//! Shared component inventory and execution plans for CLI and terminal UI.
mod self_update;
mod ui;
use super::*;
use crate::backend::updates::{self, Status};
use crate::model_store::{ModelSource, ModelStorage};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    process::Stdio,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
};

#[derive(Debug, Clone, Args)]
pub struct ComponentsArgs {
    /// Print the component inventory instead of opening the terminal UI.
    #[arg(long)]
    pub list: bool,
    #[arg(long, requires = "list")]
    pub json: bool,
    /// Use local metadata only; installation and update actions stay disabled.
    #[arg(long)]
    pub offline: bool,
    #[arg(long, hide = true, conflicts_with_all = ["list", "json", "offline"])]
    execute: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub struct UpdateArgs {
    /// Backend target or model:MODEL_ID; omit with --all.
    #[arg(required_unless_present = "all", conflicts_with = "all")]
    component: Option<String>,
    /// Update all installed, host-compatible managed backends and changed models.
    #[arg(long)]
    all: bool,
    /// Also set up missing dependencies required by installed models.
    #[arg(long, requires = "all")]
    include_missing: bool,
    /// Approve the displayed plan without prompting.
    #[arg(long, short)]
    yes: bool,
    /// Display the concrete plan without changing components.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", content = "target", rename_all = "snake_case")]
enum Action {
    InstallBackend(String),
    UpdateBackend(String),
    SetupModel(String),
    UpdateModel(String),
    UpdateWerk(String),
    UpdateProjector { model: String, file: String },
    UpdateExternalBackend { target: String, python: PathBuf },
}

impl Action {
    fn label(&self) -> String {
        match self {
            Self::UpdateProjector { model, file } => format!(
                "Update external projector {model}/{file}; verify matching main weights and retain previous file"
            ),
            Self::UpdateWerk(version) => {
                format!("Update werk1112 to {version}; retain previous executable")
            }
            Self::InstallBackend(s) => format!("Install {s} and dependencies"),
            Self::UpdateBackend(s) => {
                format!("Update {s} and dependencies (retain supported version pins)")
            }
            Self::SetupModel(s) => format!("Set up missing components for {s}"),
            Self::UpdateModel(s) => {
                format!("Update model {s} with matching vision assets; retain previous files")
            }
            Self::UpdateExternalBackend { target, python } => format!(
                "Update external {target} and dependencies in {}",
                python.display()
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Component {
    id: String,
    name: String,
    kind: String,
    status: String,
    installed: Option<String>,
    available: Option<String>,
    detail: String,
    required_by: Vec<String>,
    action: Option<Action>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Inventory {
    host: String,
    offline: bool,
    components: Vec<Component>,
}

struct Host {
    os: &'static str,
    arch: &'static str,
    cuda: bool,
    nvfp4: bool,
    rocm: bool,
    vulkan: bool,
    strix: bool,
    macos_major: Option<u32>,
}

impl Host {
    fn detect() -> Self {
        let cuda = crate::inference_service::devices::Inventory::detect().ok();
        let cuda_devices = cuda.as_ref().map(|i| i.visible()).unwrap_or_default();
        let vulkan = updates::command_output(
            Command::new("vulkaninfo").arg("--summary"),
            Duration::from_secs(5),
        )
        .is_ok_and(|(out, _)| String::from_utf8_lossy(&out).contains("deviceName"));
        Self {
            os: env::consts::OS,
            arch: env::consts::ARCH,
            cuda: !cuda_devices.is_empty(),
            nvfp4: cuda_devices.iter().any(|d| {
                d.architecture
                    .as_deref()
                    .and_then(|s| s.strip_prefix("sm_"))
                    .and_then(|s| s.parse::<u32>().ok())
                    .is_some_and(|n| n >= 80)
            }),
            rocm: crate::backend::current_selected_rocm_device_status()
                != crate::backend::SelectedRocmDeviceStatus::Unknown,
            vulkan,
            strix: current_host_is_strix_halo(),
            macos_major: if cfg!(target_os = "macos") {
                updates::command_output(
                    Command::new("sw_vers").arg("-productVersion"),
                    Duration::from_secs(3),
                )
                .ok()
                .and_then(|(bytes, _)| {
                    String::from_utf8_lossy(&bytes)
                        .split('.')
                        .next()?
                        .trim()
                        .parse()
                        .ok()
                })
            } else {
                None
            },
        }
    }

    fn compatible(&self, target: &str) -> bool {
        if !crate::backend::BACKEND_INSTALL_TARGETS.contains(&target)
            || runtime_install_target_for_platform(Some(target), self.os, self.arch, self.strix)
                .is_none()
        {
            return false;
        }
        match target {
            "omlx" => self.macos_major.is_some_and(|version| version >= 15),
            "mlx" | "mlx-vlm" => self.macos_major.is_some_and(|version| version >= 14),
            "llama-cuda-nvfp4" => self.cuda && self.nvfp4,
            "llama-cuda" | "llama-cuda-offload" | "onnx-cuda" | "vllm" => self.cuda,
            "llama-rocm" | "onnx-rocm" => self.rocm,
            "llama-vulkan" => self.vulkan,
            _ => true,
        }
    }
}

pub(super) fn ensure_compatible(_store: &ModelStore, target: &str) -> Result<()> {
    anyhow::ensure!(
        Host::detect().compatible(target),
        "{target} is not supported by the detected host/accelerator; inspect werk components --list"
    );
    Ok(())
}

fn inventory(store: &ModelStore, offline: bool) -> Result<Inventory> {
    let host = Host::detect();
    let scanned = updates::collect(store, offline)?;
    let mut backends = BTreeMap::new();
    for &target in crate::backend::BACKEND_INSTALL_TARGETS {
        if !host.compatible(target) {
            continue;
        }
        backends.insert(target.to_string(), Component {
            id: target.into(), name: target.into(), kind: "backend".into(), status: "not installed".into(),
            installed: None, available: None,
            detail: "Compatible with the detected host. Installation includes the backend's dependencies; required build tools and Python are checked by its installer.".into(),
            required_by: vec![], action: (!offline).then(|| Action::InstallBackend(target.into())),
        });
    }
    let mut components = Vec::new();
    let mut grouped: BTreeMap<String, Vec<&updates::Entry>> = BTreeMap::new();
    for entry in &scanned {
        if entry.kind == "backend" {
            let target = entry
                .name
                .split('/')
                .next()
                .unwrap_or(&entry.name)
                .replace("onnxruntime-", "onnx-");
            grouped.entry(target).or_default().push(entry);
        }
    }
    for (target, entries) in grouped {
        let Some(row) = backends.get_mut(&target) else {
            continue;
        };
        let status = if entries.iter().any(|e| e.status == Status::Error) {
            "scan error"
        } else if entries.iter().any(|e| e.status == Status::UpdateAvailable) {
            "update available"
        } else if entries.iter().any(|e| e.status == Status::Pinned) {
            "pinned"
        } else if entries.iter().any(|e| e.status == Status::Unknown) {
            "installed / unchecked"
        } else {
            "up to date"
        };
        row.status = status.into();
        row.installed = Some(
            entries
                .iter()
                .map(|e| {
                    format!(
                        "{} {}",
                        e.name.split('/').next_back().unwrap_or(&e.name),
                        e.installed.as_deref().unwrap_or("unknown")
                    )
                })
                .collect::<Vec<_>>()
                .join(", "),
        );
        row.available = Some(
            entries
                .iter()
                .filter(|e| e.status == Status::UpdateAvailable || e.status == Status::Pinned)
                .map(|e| {
                    format!(
                        "{} {}",
                        e.name.split('/').next_back().unwrap_or(&e.name),
                        e.available.as_deref().unwrap_or("pinned")
                    )
                })
                .collect::<Vec<_>>()
                .join(", "),
        );
        row.detail = entries
            .iter()
            .map(|e| format!("{}: {}", e.name, e.detail))
            .collect::<Vec<_>>()
            .join("\n");
        row.action = (!offline).then(|| Action::UpdateBackend(target));
    }
    let mut external: BTreeMap<(&str, &Path), Vec<&updates::Entry>> = BTreeMap::new();
    for entry in scanned.iter().filter(|e| e.kind == "external_backend") {
        let target = entry.name.split('/').next().unwrap_or(&entry.name);
        if host.compatible(target) {
            external
                .entry((target, &entry.path))
                .or_default()
                .push(entry);
        }
    }
    for ((target, python), entries) in external {
        components.push(Component {
            id: format!("external:{target}"), name: format!("{target} (external)"), kind: "external backend".into(),
            status: if entries.iter().any(|e| e.status == Status::Error) { "scan error" } else if entries.iter().any(|e| e.status == Status::UpdateAvailable) { "update available" } else { "installed / external" }.into(),
            installed: Some(entries.iter().filter_map(|e| e.installed.as_deref()).collect::<Vec<_>>().join(", ")),
            available: Some(entries.iter().filter_map(|e| e.available.as_deref()).collect::<Vec<_>>().join(", ")),
            detail: format!("Existing environment: {}\nUpdates affect this shared environment, including its dependencies. Editable/forked packages and downgrades require their own source workflow.\n{}", python.display(), entries.iter().map(|e| format!("{}: {}", e.name, e.detail)).collect::<Vec<_>>().join("\n")),
            required_by: vec![], action: (!offline && entries.iter().any(|e| e.installed.is_some())).then(|| Action::UpdateExternalBackend { target: target.into(), python: python.into() }),
        });
    }
    let models = store.list_existing()?;
    let mut compatible_models = HashSet::new();
    for model in &models {
        let scanned_model = scanned
            .iter()
            .find(|e| e.kind == "model" && e.name == model.id);
        let mut detail = scanned_model.map(|e| e.detail.clone()).unwrap_or_default();
        let mut status = scanned_model
            .map(|e| e.status.label())
            .unwrap_or("unknown")
            .to_string();
        let mut action = None;
        let plan = model_setup::plan(
            store,
            &model,
            BackendChoice::Auto,
            SelectionOptions::default(),
            offline,
            true,
        );
        let mut host_supported = false;
        match plan {
            Ok(plan) => {
                host_supported = plan.issues.is_empty()
                    && plan.installs.iter().all(|target| host.compatible(target));
                for target in &plan.installs {
                    if let Some(row) = backends.get_mut(*target) {
                        row.required_by.push(model.id.clone());
                        if row.installed.is_none() {
                            row.status = "missing / required".into();
                        }
                    }
                }
                detail.push_str(&format!("\n{}", plan.describe()));
                if plan.has_actions() {
                    status = "missing components".into();
                    if !offline && host_supported && plan.issues.is_empty() {
                        action = Some(Action::SetupModel(model.id.clone()));
                    }
                } else if !plan.issues.is_empty() {
                    status = "setup blocked".into();
                    host_supported = false;
                }
            }
            Err(e) => {
                detail.push_str(&format!("\nSetup check failed: {e:#}"));
                status = "setup error".into();
            }
        }
        let changed = scanned_model.is_some_and(|e| e.status == Status::UpdateAvailable)
            || scanned.iter().any(|e| {
                e.kind == "projector"
                    && e.name.starts_with(&format!("{}/", model.id))
                    && e.status == Status::UpdateAvailable
            });
        if !offline
            && host_supported
            && changed
            && matches!(&model.source, ModelSource::HuggingFace { .. })
            && matches!(model.storage, ModelStorage::Managed)
        {
            action = Some(Action::UpdateModel(model.id.clone()));
            status = "update available".into();
        }
        if host_supported {
            compatible_models.insert(model.id.clone());
        }
        components.push(Component {
            id: format!("model:{}", model.id),
            name: model.id.clone(),
            kind: "model".into(),
            status,
            installed: scanned_model.and_then(|e| e.installed.clone()),
            available: scanned_model.and_then(|e| e.available.clone()),
            detail,
            required_by: vec![],
            action,
        });
    }
    // Projectors are visible individually. Replacement is paired with the model
    // update so an unrelated latest projector cannot silently break inference.
    for entry in scanned.iter().filter(|e| e.kind == "projector") {
        let external = models.iter().find_map(|m| {
            if !matches!(m.storage, ModelStorage::External { .. })
                || !compatible_models.contains(&m.id)
            {
                return None;
            }
            m.files
                .iter()
                .find(|f| format!("{}/{}", m.id, f.path) == entry.name)
                .map(|f| (m, f))
        });
        let action = external
            .filter(|(model, _)| {
                !store
                    .model_files_dir(model)
                    .ancestors()
                    .any(|p| p.file_name().is_some_and(|n| n == "snapshots"))
            })
            .filter(|_| !offline && entry.status == Status::UpdateAvailable)
            .map(|(model, file)| Action::UpdateProjector {
                model: model.id.clone(),
                file: file.path.clone(),
            });
        components.push(Component {
            id: format!("projector:{}", entry.name),
            name: entry.name.clone(),
            kind: "projector".into(),
            status: entry.status.label().into(),
            installed: entry.installed.clone(),
            available: entry.available.clone(),
            detail: format!(
                "{}\nUpdate together with its model using the model row.",
                entry.detail
            ),
            required_by: vec![],
            action,
        });
    }
    let mut rows: Vec<_> = backends.into_values().collect();
    rows.extend(components);
    rows.push(self_update::component(offline));
    Ok(Inventory {
        host: format!(
            "{} / {} · CUDA {} · ROCm {} · Vulkan {}",
            host.os, host.arch, host.cuda, host.rocm, host.vulkan
        ),
        offline,
        components: rows,
    })
}

pub(super) fn run(store: ModelStore, args: ComponentsArgs) -> Result<()> {
    if let Some(action) = args.execute {
        return execute_action(&store, serde_json::from_str(&action)?);
    }
    if args.list {
        let inventory = inventory(&store, args.offline)?;
        if args.json {
            std::println!("{}", serde_json::to_string_pretty(&inventory)?);
        } else {
            crate::ui_println!("{}", inventory.host);
            for c in inventory.components {
                crate::ui_println!("{:<32} {:<22} {}", c.id, c.status, c.required_by.join(", "));
            }
        }
        Ok(())
    } else {
        ui::run(store, args.offline)
    }
}

fn selected_actions(inventory: &Inventory, args: &UpdateArgs) -> Result<Vec<Action>> {
    if let Some(id) = &args.component {
        let row = inventory.components.iter().find(|c| &c.id == id).context(
            "Component not found or incompatible with this host; use werk components --list",
        )?;
        let action = row
            .action
            .clone()
            .context("No supported action is available for this component")?;
        anyhow::ensure!(
            !matches!(action, Action::InstallBackend(_)),
            "Component is not installed; use werk backend install or werk components"
        );
        return Ok(vec![action]);
    }
    Ok(inventory
        .components
        .iter()
        .filter(|c| {
            c.action.as_ref().is_some_and(|a| {
                matches!(
                    a,
                    Action::UpdateBackend(_)
                        | Action::UpdateModel(_)
                        | Action::UpdateExternalBackend { .. }
                        | Action::UpdateWerk(_)
                        | Action::UpdateProjector { .. }
                ) || args.include_missing
                    && (matches!(a, Action::SetupModel(_))
                        || matches!(a, Action::InstallBackend(_)) && !c.required_by.is_empty())
            })
        })
        .filter_map(|c| c.action.as_ref())
        .cloned()
        .collect())
}

pub(super) fn update(store: ModelStore, args: UpdateArgs) -> Result<()> {
    let activity = crate::terminal::Activity::start("Preparing component update plan");
    let inventory = inventory(&store, false)?;
    let actions = selected_actions(&inventory, &args)?;
    activity.finish();
    if actions.is_empty() {
        crate::ui_println!("No compatible component updates are planned.");
        return Ok(());
    }
    for (i, action) in actions.iter().enumerate() {
        crate::ui_println!("{}. {}", i + 1, action.label());
    }
    if args.dry_run {
        return Ok(());
    }
    if !args.yes {
        anyhow::ensure!(
            io::stdin().is_terminal(),
            "Use --yes to execute this plan noninteractively, or --dry-run to review it"
        );
        crate::ui_eprintln!("Execute this plan? [y/N]");
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !matches!(
            input.trim().to_ascii_lowercase().as_str(),
            "y" | "yes" | "j" | "ja"
        ) {
            return Ok(());
        }
    }
    let mut failed = Vec::new();
    for (i, action) in actions.iter().enumerate() {
        let activity = crate::terminal::Activity::start(format!(
            "[{}/{}] {}",
            i + 1,
            actions.len(),
            action.label()
        ));
        let status = action_command(&store, action)?.status();
        activity.finish();
        if !status.is_ok_and(|s| s.success()) {
            failed.push(action.label());
        }
    }
    anyhow::ensure!(
        failed.is_empty(),
        "Some components failed; successful changes are retained:\n{}",
        failed.join("\n")
    );
    crate::ui_println!(
        "Completed {} component(s). Restart running inference services to use updated runtimes.",
        actions.len()
    );
    Ok(())
}

fn action_command(store: &ModelStore, action: &Action) -> Result<Command> {
    let mut command = Command::new(env::current_exe()?);
    command
        .arg("--model-home")
        .arg(store.home())
        .args(["components", "--execute"])
        .arg(serde_json::to_string(action)?);
    command
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .stdin(Stdio::null());
    Ok(command)
}

fn execute_action(store: &ModelStore, action: Action) -> Result<()> {
    let updating_backend = matches!(&action, Action::UpdateBackend(_));
    fs::create_dir_all(store.home())?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(store.home().join("components.lock"))?;
    fs2::FileExt::try_lock_exclusive(&lock).context("Another component operation is running")?;
    match action {
        Action::UpdateProjector { model, file } => {
            store.update_external_projector(&model, &file)?
        }
        Action::UpdateWerk(version) => self_update::install(&version)?,
        Action::UpdateExternalBackend { target, python } => {
            ensure_compatible(store, &target)?;
            anyhow::ensure!(
                updates::external_python_candidates(store)
                    .iter()
                    .any(|(t, p)| *t == target && p == &python),
                "External interpreter selection changed; refresh the component inventory"
            );
            updates::update_external_python(&target, &python)?;
        }
        Action::InstallBackend(target) | Action::UpdateBackend(target) => {
            ensure_compatible(store, &target)?;
            let update = updating_backend;
            let mut command = Command::new(env::current_exe()?);
            command
                .arg("--model-home")
                .arg(store.home())
                .args([
                    "backend",
                    if update { "update" } else { "install" },
                    &target,
                ])
                .stdin(Stdio::null());
            anyhow::ensure!(
                command.status()?.success(),
                "Backend operation failed: {target}"
            );
        }
        Action::SetupModel(id) => {
            let model = store.get_existing(&id)?;
            let plan = model_setup::plan(
                store,
                &model,
                BackendChoice::Auto,
                SelectionOptions::default(),
                false,
                true,
            )?;
            for target in &plan.installs {
                ensure_compatible(store, target)?;
            }
            model_setup::ensure(
                store,
                &id,
                BackendChoice::Auto,
                SelectionOptions {
                    setup_policy: model_setup::SetupPolicy::Approve,
                    ..Default::default()
                },
                false,
                true,
            )?;
        }
        Action::UpdateModel(id) => {
            let progress = pull_progress_bar();
            let backup = store
                .update_from_huggingface(&id, |event| update_pull_progress(&progress, event))?;
            progress.finish_and_clear();
            crate::ui_println!(
                "Updated {id}; previous model retained at {}",
                backup.display()
            );
        }
    }
    Ok(())
}

enum Event {
    Inventory(Result<Inventory, String>),
    Started(usize, String),
    Log(String),
    Done(usize, bool),
    Finished,
}

fn refresh(store: ModelStore, offline: bool, tx: mpsc::Sender<Event>, cancel: Arc<AtomicBool>) {
    thread::spawn(move || {
        let result = (|| -> Result<Inventory> {
            let mut command = Command::new(env::current_exe()?);
            command
                .arg("--model-home")
                .arg(store.home())
                .args(["components", "--list", "--json"]);
            if offline {
                command.arg("--offline");
            }
            let (out, _) = updates::command_output_with_cancel(
                &mut command,
                Duration::from_secs(600),
                Some(&cancel),
            )?;
            Ok(serde_json::from_slice(&out)?)
        })();
        let _ = tx.send(Event::Inventory(result.map_err(|e| format!("{e:#}"))));
    });
}

fn execute_queue(
    store: ModelStore,
    actions: Vec<Action>,
    cancel: Arc<AtomicBool>,
    tx: mpsc::Sender<Event>,
) {
    thread::spawn(move || {
        for (i, action) in actions.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let _ = tx.send(Event::Started(i, action.label()));
            let result = (|| -> Result<bool> {
                let mut child = action_command(&store, action)?
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()?;
                let out = child.stdout.take().context("Missing installer stdout")?;
                let err = child.stderr.take().context("Missing installer stderr")?;
                let stream = |reader: Box<dyn Read + Send>, tx: mpsc::Sender<Event>| {
                    thread::spawn(move || {
                        let mut line = Vec::new();
                        for byte in io::BufReader::new(reader)
                            .bytes()
                            .map_while(std::result::Result::ok)
                        {
                            if byte == b'\n' || byte == b'\r' || line.len() >= 8192 {
                                if !line.is_empty() {
                                    let _ = tx.send(Event::Log(crate::terminal::clean(
                                        &String::from_utf8_lossy(&line),
                                    )));
                                    line.clear();
                                }
                            } else {
                                line.push(byte);
                            }
                        }
                        if !line.is_empty() {
                            let _ = tx.send(Event::Log(crate::terminal::clean(
                                &String::from_utf8_lossy(&line),
                            )));
                        }
                    })
                };
                let a = stream(Box::new(out), tx.clone());
                let b = stream(Box::new(err), tx.clone());
                let status = child.wait()?;
                let _ = a.join();
                let _ = b.join();
                Ok(status.success())
            })();
            let success = match result {
                Ok(ok) => ok,
                Err(e) => {
                    let _ = tx.send(Event::Log(format!("{e:#}")));
                    false
                }
            };
            let _ = tx.send(Event::Done(i, success));
        }
        let _ = tx.send(Event::Finished);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_filter_requires_actual_accelerators() {
        let mut host = Host {
            os: "linux",
            arch: "x86_64",
            cuda: false,
            nvfp4: false,
            rocm: false,
            vulkan: false,
            strix: false,
            macos_major: None,
        };
        assert!(host.compatible("llama-cpu"));
        assert!(host.compatible("transformers"));
        for name in [
            "omlx",
            "mlx",
            "llama-cuda",
            "llama-rocm",
            "llama-vulkan",
            "vllm",
        ] {
            assert!(!host.compatible(name));
        }
        host.cuda = true;
        assert!(host.compatible("llama-cuda"));
        assert!(!host.compatible("llama-cuda-nvfp4"));
        host.arch = "aarch64";
        assert!(!host.compatible("vllm"));
        host.os = "macos";
        host.macos_major = Some(14);
        assert!(!host.compatible("omlx"));
        assert!(host.compatible("mlx"));
        host.macos_major = Some(15);
        assert!(host.compatible("omlx"));
        assert!(!host.compatible("llama-cuda"));
    }

    #[test]
    fn update_plan_includes_installed_and_external_but_not_optional_installs() {
        let actions = vec![
            Action::UpdateBackend("llama-cpu".into()),
            Action::InstallBackend("media".into()),
            Action::SetupModel("model".into()),
            Action::UpdateExternalBackend {
                target: "omlx".into(),
                python: PathBuf::from("/external/python"),
            },
            Action::UpdateWerk("2.0.0".into()),
        ];
        let inventory = Inventory {
            host: "fixture".into(),
            offline: false,
            components: actions
                .into_iter()
                .enumerate()
                .map(|(i, action)| Component {
                    id: i.to_string(),
                    name: i.to_string(),
                    kind: "fixture".into(),
                    status: "".into(),
                    installed: None,
                    available: None,
                    detail: "".into(),
                    required_by: vec![],
                    action: Some(action),
                })
                .collect(),
        };
        let mut args = UpdateArgs {
            component: None,
            all: true,
            include_missing: false,
            yes: false,
            dry_run: true,
        };
        assert_eq!(selected_actions(&inventory, &args).unwrap().len(), 3);
        args.include_missing = true;
        assert_eq!(selected_actions(&inventory, &args).unwrap().len(), 4);
        args.component = Some("1".into());
        assert!(selected_actions(&inventory, &args).is_err());
    }

    #[test]
    fn cli_management_commands_and_rejections() {
        for argv in [
            vec!["werk", "components"],
            vec!["werk", "components", "--list", "--json", "--offline"],
            vec!["werk", "update", "--all", "--dry-run"],
            vec!["werk", "update", "external:omlx", "--yes"],
            vec!["werk", "backend", "update", "llama-cpu"],
            vec!["werk", "update-scan", "--offline", "--json"],
        ] {
            assert!(Cli::try_parse_from(argv).is_ok());
        }
        for argv in [
            vec!["werk", "update"],
            vec!["werk", "update", "llama-cpu", "--all"],
            vec!["werk", "components", "--json"],
            vec!["werk", "components", "--offline", "--execute", "{}"],
        ] {
            assert!(Cli::try_parse_from(argv).is_err());
        }
    }
}
