//! Explicit, read-only update inventory. Remote failures belong to individual
//! components, so an unavailable registry never hides the rest of the report.
use crate::model_store::{MANIFEST_FILE, ModelManifest, ModelSource, ModelStore};
use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, clap::Args)]
pub struct ScanOptions {
    /// Print the complete report as JSON, including per-component errors.
    #[arg(long)]
    pub json: bool,
    /// Inventory local components without contacting upstream registries.
    #[arg(long)]
    pub offline: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ScanScope {
    All,
    Backends,
    Artifacts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Status {
    UpToDate,
    UpdateAvailable,
    Pinned,
    Unknown,
    Error,
}

impl Status {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::UpToDate => "up to date",
            Self::UpdateAvailable => "update available",
            Self::Pinned => "pinned",
            Self::Unknown => "unknown",
            Self::Error => "error",
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct Entry {
    pub(crate) kind: &'static str,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) installed: Option<String>,
    pub(crate) available: Option<String>,
    pub(crate) status: Status,
    pub(crate) detail: String,
}

impl Entry {
    fn new(kind: &'static str, name: impl Into<String>, path: PathBuf) -> Self {
        Self {
            kind,
            name: name.into(),
            path,
            installed: None,
            available: None,
            status: Status::Unknown,
            detail: "No verified upstream version information".into(),
        }
    }

    fn fail(&mut self, error: anyhow::Error) {
        self.status = Status::Error;
        self.detail = format!("{error:#}");
    }
}

#[derive(Debug, Default, Serialize)]
struct Summary {
    up_to_date: usize,
    update_available: usize,
    pinned: usize,
    unknown: usize,
    errors: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u8,
    offline: bool,
    summary: Summary,
    entries: Vec<Entry>,
}

impl Report {
    fn new(offline: bool, mut entries: Vec<Entry>) -> Self {
        entries.sort_by(|a, b| (&a.kind, &a.name, &a.path).cmp(&(&b.kind, &b.name, &b.path)));
        let mut summary = Summary::default();
        for entry in &entries {
            match entry.status {
                Status::UpToDate => summary.up_to_date += 1,
                Status::UpdateAvailable => summary.update_available += 1,
                Status::Pinned => summary.pinned += 1,
                Status::Unknown => summary.unknown += 1,
                Status::Error => summary.errors += 1,
            }
        }
        Self {
            schema_version: 1,
            offline,
            summary,
            entries,
        }
    }
}

pub fn run(store: &ModelStore, scope: ScanScope, options: &ScanOptions) -> Result<()> {
    let activity =
        (!options.json).then(|| crate::terminal::Activity::start("Scanning installed components"));
    let mut remote = Remote::new(store, options.offline)?;
    let report = scan(store, scope, options.offline, &mut remote)?;
    if let Some(activity) = activity {
        activity.finish();
    }
    if options.json {
        crate::ui_println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        if report.entries.is_empty() {
            crate::ui_println!("No installed components found in the selected scope.");
        }
        for entry in &report.entries {
            crate::ui_println!(
                "{} / {}: {}",
                printable(entry.kind),
                printable(&entry.name),
                entry.status.label()
            );
            crate::ui_println!(
                "  installed: {} | upstream: {}",
                printable(entry.installed.as_deref().unwrap_or("unknown")),
                printable(entry.available.as_deref().unwrap_or("not checked"))
            );
            crate::ui_println!("  {}", printable(&entry.detail));
        }
        let s = &report.summary;
        crate::ui_println!(
            "{} update(s), {} current, {} pinned, {} unknown, {} error(s).",
            s.update_available,
            s.up_to_date,
            s.pinned,
            s.unknown,
            s.errors
        );
        crate::ui_println!(
            "Scan only; nothing installed or changed. Upstream availability does not prove runtime/model compatibility."
        );
    }
    ensure!(
        report.summary.errors == 0,
        "Update scan incomplete: {} component error(s); see report",
        report.summary.errors
    );
    Ok(())
}

pub(crate) fn collect(store: &ModelStore, offline: bool) -> Result<Vec<Entry>> {
    let mut remote = Remote::new(store, offline)?;
    Ok(scan(store, ScanScope::All, offline, &mut remote)?.entries)
}

/// Recover provenance for models imported or linked from an existing HF clone
/// or snapshot. No Git fetch, checkout, or cache writes occur here.
pub(crate) fn tracked_model_source(
    store: &ModelStore,
    model: &ModelManifest,
) -> Result<Option<(String, String)>> {
    let source = store.model_dir(&model.id).join("source-revision.json");
    if let ModelSource::HuggingFace { repo } = &model.source {
        let saved: Value = if source.is_file() {
            read_json(&source)?
        } else {
            Value::Null
        };
        let revision = (saved["repo"].as_str() == Some(repo))
            .then(|| saved["revision"].as_str())
            .flatten()
            .or_else(|| {
                model
                    .metadata
                    .vision_setup
                    .as_ref()
                    .and_then(|s| s.source_revision.as_deref())
            });
        if let Some(revision) = revision.filter(|s| hex(s, 40, 40)) {
            return Ok(Some((repo.clone(), revision.into())));
        }
    }
    let root = store.model_files_dir(model);
    if root
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|s| s == "snapshots")
    {
        let revision = root.file_name().unwrap_or_default().to_string_lossy();
        if let Some(name) = root
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("models--"))
        {
            if let Some((owner, name)) = name.split_once("--") {
                let repo = format!("{owner}/{name}");
                if hex(&revision, 40, 40) && repository_url(&repo).is_ok() {
                    return Ok(Some((repo, revision.into_owned())));
                }
            }
        }
    }
    let remote = command_output(
        Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(["remote", "get-url", "origin"]),
        Duration::from_secs(3),
    );
    if let Ok((bytes, _)) = remote {
        let url = String::from_utf8_lossy(&bytes);
        if let Some(repo) = url
            .trim()
            .strip_prefix("https://huggingface.co/")
            .map(|s| s.trim_end_matches(".git"))
        {
            if repository_url(repo).is_ok() {
                let (bytes, _) = command_output(
                    Command::new("git")
                        .arg("-C")
                        .arg(&root)
                        .args(["rev-parse", "HEAD"]),
                    Duration::from_secs(3),
                )?;
                let revision = String::from_utf8(bytes)?.trim().to_string();
                if hex(&revision, 40, 40) {
                    return Ok(Some((repo.into(), revision)));
                }
            }
        }
    }
    Ok(None)
}

pub(crate) fn repository_info(store: &ModelStore, repo: &str) -> Result<Value> {
    Remote::new(store, false)?.get(&repository_url(repo)?)
}

fn printable(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

trait Metadata {
    fn get(&mut self, url: &str) -> Result<Value>;
}

struct Remote<'a> {
    client: reqwest::blocking::Client,
    store: &'a ModelStore,
    offline: bool,
    cache: HashMap<String, std::result::Result<Value, String>>,
}

impl<'a> Remote<'a> {
    fn new(store: &'a ModelStore, offline: bool) -> Result<Self> {
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(15))
                .connect_timeout(Duration::from_secs(5))
                .user_agent(concat!("werk/", env!("CARGO_PKG_VERSION"), " update-scan"))
                .build()?,
            store,
            offline,
            cache: HashMap::new(),
        })
    }
}

impl Metadata for Remote<'_> {
    fn get(&mut self, url: &str) -> Result<Value> {
        ensure!(!self.offline, "Offline mode forbids upstream requests");
        if !self.cache.contains_key(url) {
            let result = (|| -> Result<Value> {
                let mut request = self.client.get(url).header("Accept", "application/json");
                if url.starts_with("https://huggingface.co/") {
                    if let Some(token) = self.store.huggingface_http_token()? {
                        request = request.bearer_auth(token);
                    }
                }
                let response = request
                    .send()
                    .map_err(|_| anyhow!("Upstream request failed; check network access"))?;
                ensure!(
                    response.status().is_success(),
                    "Upstream returned HTTP {}",
                    response.status()
                );
                let mut bytes = Vec::new();
                response.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
                ensure!(
                    bytes.len() <= 8 * 1024 * 1024,
                    "Upstream metadata is too large"
                );
                Ok(serde_json::from_slice(&bytes).context("Invalid upstream JSON")?)
            })();
            self.cache
                .insert(url.into(), result.map_err(|e| format!("{e:#}")));
        }
        self.cache[url].clone().map_err(|e| anyhow!(e))
    }
}

fn directories(root: &Path) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(e).with_context(|| format!("Cannot read {}", root.display())),
    };
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry?;
        if entry.path().is_dir() {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

fn scan(
    store: &ModelStore,
    scope: ScanScope,
    offline: bool,
    remote: &mut impl Metadata,
) -> Result<Report> {
    let mut entries = Vec::new();
    if scope != ScanScope::Artifacts {
        scan_backends(store, offline, remote, &mut entries)?;
    }
    if scope != ScanScope::Backends {
        for dir in directories(&store.models_dir())? {
            let path = dir.join(MANIFEST_FILE);
            if !path.is_file() {
                continue;
            }
            let manifest: Result<ModelManifest> = read_json(&path);
            match manifest {
                Ok(manifest) => scan_model(store, &dir, &manifest, offline, remote, &mut entries),
                Err(e) => {
                    let mut entry = Entry::new(
                        "model",
                        dir.file_name().unwrap_or_default().to_string_lossy(),
                        path,
                    );
                    entry.fail(e);
                    entries.push(entry);
                }
            }
        }
    }
    Ok(Report::new(offline, entries))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("Cannot read {}", path.display()))?,
    )
    .with_context(|| format!("Invalid metadata in {}", path.display()))
}

fn scan_backends(
    store: &ModelStore,
    offline: bool,
    remote: &mut impl Metadata,
    entries: &mut Vec<Entry>,
) -> Result<()> {
    let mut seen = HashSet::new();
    for root in directories(&store.home().join("backends"))? {
        let name = root.file_name().unwrap_or_default().to_string_lossy();
        let python = root.join("venv").join(if cfg!(windows) {
            "Scripts/python.exe"
        } else {
            "bin/python"
        });
        if python.is_file() {
            scan_python(&name, &python, offline, false, entries);
        } else if name.starts_with("llama-") {
            let mut roots = vec![root.clone()];
            roots.extend(directories(&root)?.into_iter().filter(|p| {
                p.file_name().is_some_and(|s| {
                    s.to_string_lossy().starts_with("offload-")
                        || s.to_string_lossy().starts_with("nvfp4-")
                })
            }));
            for profile in roots {
                for path in super::llama_server::server_candidates_under(&profile)
                    .into_iter()
                    .filter(|p| p.is_file())
                    .take(1)
                {
                    if !path.is_file()
                        || !seen.insert(fs::canonicalize(&path).unwrap_or_else(|_| path.clone()))
                    {
                        continue;
                    }
                    let mut entry = Entry::new("backend", name.as_ref(), path.clone());
                    let pinned = path.components().any(|c| {
                        c.as_os_str().to_string_lossy().starts_with("nvfp4-")
                            || c.as_os_str().to_string_lossy().starts_with("offload-")
                    });
                    if pinned {
                        entry.name = if path.to_string_lossy().contains("nvfp4-") {
                            "llama-cuda-nvfp4"
                        } else {
                            "llama-cuda-offload"
                        }
                        .into();
                    }
                    let result = check_llama(&path, offline, pinned, remote, &mut entry);
                    if let Err(e) = result {
                        entry.fail(e);
                    }
                    entries.push(entry);
                }
            }
        } else if name.starts_with("onnxruntime-") {
            let executable = root.join(if cfg!(windows) {
                "werk-onnx-runner.exe"
            } else {
                "werk-onnx-runner"
            });
            if executable.is_file() {
                let mut entry = Entry::new("backend", name.as_ref(), executable);
                entry.detail = "Bundled ONNX runner has no upstream release metadata; update through a matching Werk bundle".into();
                entries.push(entry);
            }
        }
    }
    for (target, python) in external_python_candidates(store) {
        let first = entries.len();
        scan_python(target, &python, offline, true, entries);
        for entry in &mut entries[first..] {
            entry.kind = "external_backend";
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct Package {
    name: String,
    installed: String,
    available: Option<String>,
    status: Status,
    detail: String,
}

fn scan_python(name: &str, python: &Path, offline: bool, external: bool, entries: &mut Vec<Entry>) {
    let primary: &[&str] = match name {
        "mlx" => &["mlx-lm"],
        "mlx-vlm" => &["mlx-vlm"],
        "transformers" => &["transformers"],
        "media" => &["diffusers", "transformers"],
        "vllm" => &["vllm"],
        "qwen-tts" => &["qwen-tts"],
        "omlx" => &["omlx"],
        "text-analysis" => &["transformers", "sentence-transformers", "laya"],
        "onnx-cpu" => &["onnxruntime-genai"],
        _ => &[],
    };
    let mut packages = primary.to_vec();
    for dependency in [
        "torch",
        "transformers",
        "diffusers",
        "mlx",
        "mlx-lm",
        "mlx-vlm",
        "onnxruntime",
        "onnxruntime-genai",
        "accelerate",
        "tokenizers",
        "torchaudio",
        "tilelang",
    ] {
        if !packages.contains(&dependency) {
            packages.push(dependency);
        }
    }
    let pinned: &[&str] = match name {
        "omlx" => &["omlx"],
        "qwen-tts" => &["qwen-tts"],
        _ => &[],
    };
    let required = if name == "media" {
        &["diffusers"][..]
    } else {
        primary
    };
    let result = (|| -> Result<Vec<Package>> {
        let constraints = if external
            || !matches!(
                name,
                "mlx"
                    | "mlx-vlm"
                    | "media"
                    | "transformers"
                    | "onnx-cpu"
                    | "text-analysis"
                    | "qwen-tts"
            ) {
            String::new()
        } else if let Some(path) = python
            .ancestors()
            .nth(3)
            .map(|root| root.join("constraints.txt"))
        {
            match fs::read_to_string(path) {
                Ok(value) => value,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
                Err(error) => return Err(error.into()),
            }
        } else {
            String::new()
        };
        let config = json!({"packages": packages, "required": required,
            "pinned": pinned, "offline": offline, "constraints": constraints});
        let mut command = Command::new(python);
        if !external {
            command.arg("-I");
        }
        let (output, _) = command_output(
            command.args([
                "-B",
                "-c",
                include_str!("update_packages.py"),
                &config.to_string(),
            ]),
            Duration::from_secs(60),
        )?;
        let packages: Vec<Package> =
            serde_json::from_slice(&output).context("Invalid Python package inventory")?;
        ensure!(
            external || !packages.is_empty(),
            "No runtime packages found in the managed Python environment"
        );
        Ok(packages)
    })();
    match result {
        Ok(packages) => entries.extend(packages.into_iter().map(|package| Entry {
            kind: "backend",
            name: format!("{name}/{}", package.name),
            path: python.into(),
            installed: Some(package.installed),
            available: package.available,
            status: package.status,
            detail: package.detail,
        })),
        Err(e) => {
            let mut entry = Entry::new("backend", name, python.into());
            entry.fail(e);
            entries.push(entry);
        }
    }
}

/// Follow the same interpreter selection as inference. Keep venv symlink paths:
/// canonicalizing a venv's Python would accidentally select the base environment.
pub(crate) fn external_python_candidates(store: &ModelStore) -> Vec<(&'static str, PathBuf)> {
    use super::python_install::PythonBackend;
    let mut candidates = Vec::new();
    for backend in [
        PythonBackend::Mlx,
        PythonBackend::MlxVlm,
        PythonBackend::Transformers,
        PythonBackend::Media,
        PythonBackend::OnnxCpu,
    ] {
        if super::python_install::platform_rejection(
            backend.target(),
            std::env::consts::OS,
            std::env::consts::ARCH,
        )
        .is_none()
        {
            candidates.push((backend.target(), backend.discover_python(store)));
        }
    }
    for (target, variable) in [
        ("vllm", "WERK_VLLM_PYTHON"),
        ("qwen-tts", "WERK_QWEN_TTS_PYTHON"),
        ("text-analysis", "WERK_TEXT_PYTHON"),
    ] {
        if let Some(path) = std::env::var_os(variable) {
            candidates.push((target, PathBuf::from(path)));
        }
    }
    if let Ok(python) = super::OmlxBackend::discovered_python(store) {
        candidates.push(("omlx", python));
    }
    candidates
        .into_iter()
        .filter_map(|(target, path)| {
            let path = if path.components().count() == 1 {
                std::env::split_paths(&std::env::var_os("PATH")?)
                    .map(|p| p.join(&path))
                    .find(|p| p.is_file())?
            } else if path.is_absolute() {
                path
            } else {
                std::env::current_dir().ok()?.join(path)
            };
            (path.is_file() && !path.starts_with(store.home().join("backends")))
                .then_some((target, path))
        })
        .collect()
}

pub(crate) fn update_external_python(target: &str, python: &Path) -> Result<()> {
    use super::python_install::PythonBackend;
    super::python_install::ensure_install_platform(target)?;
    let (packages, validation): (Vec<&str>, &str) = match target {
        "mlx" => (
            PythonBackend::Mlx.packages().to_vec(),
            PythonBackend::Mlx.validation(),
        ),
        "mlx-vlm" => (
            PythonBackend::MlxVlm.packages().to_vec(),
            PythonBackend::MlxVlm.validation(),
        ),
        "transformers" => (
            PythonBackend::Transformers.packages().to_vec(),
            PythonBackend::Transformers.validation(),
        ),
        "media" => (
            PythonBackend::Media.packages().to_vec(),
            PythonBackend::Media.validation(),
        ),
        "onnx-cpu" => (
            PythonBackend::OnnxCpu.packages().to_vec(),
            PythonBackend::OnnxCpu.validation(),
        ),
        "omlx" => (
            vec!["git+https://github.com/jundot/omlx.git@v0.7.0"],
            "from omlx.cli import main; import mlx.core as mx; assert mx.metal.is_available()",
        ),
        "vllm" => (vec!["vllm"], "import vllm"),
        "qwen-tts" => (
            vec![super::qwen_tts::QWEN_TTS_PACKAGE],
            "from qwen_tts import Qwen3TTSModel",
        ),
        "text-analysis" => (
            include_str!("../../runtime/requirements-text-analysis.txt")
                .lines()
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect(),
            "import torch, transformers, sentence_transformers, laya",
        ),
        _ => anyhow::bail!("External updates are not supported for {target}"),
    };
    // Let pip resolve a concrete compatible transaction first. Refuse planned
    // downgrades, editable installs and changed VCS origins in shared environments.
    let report = tempfile::NamedTempFile::new()?;
    let constraints = tempfile::NamedTempFile::new()?;
    let scratch = tempfile::tempdir()?;
    ensure!(
        crate::terminal::command_status(
            Command::new(python)
                .current_dir(scratch.path())
                .args(["-B", "-m", "pip", "check"])
        )?
        .success(),
        "External environment already has dependency conflicts; no packages changed. Repair it with its own package manager, or use `werk backend install {target}` for an isolated managed environment"
    );
    let mut preview = Command::new(python);
    preview.current_dir(scratch.path());
    preview
        .args([
            "-B",
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--upgrade-strategy",
            "eager",
            "--dry-run",
            "--report",
        ])
        .arg(report.path())
        .args(&packages);
    ensure!(
        crate::terminal::command_status(&mut preview)?.success(),
        "Cannot resolve an update for this external environment; its package manager may own it"
    );
    ensure!(
        crate::terminal::command_status(
            Command::new(python)
                .current_dir(scratch.path())
                .args(["-B", "-c", include_str!("update_external_check.py")])
                .arg(report.path())
                .arg(constraints.path())
                .arg(serde_json::to_string(&packages)?)
        )?
        .success(),
        "External update would alter an incompatible/custom installation; environment retained"
    );
    ensure!(
        crate::terminal::command_status(
            Command::new(python)
                .args([
                    "-B",
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--upgrade-strategy",
                    "eager"
                ])
                .current_dir(scratch.path())
                .arg("--constraint")
                .arg(constraints.path())
                .args(packages)
        )?
        .success(),
        "External backend update failed"
    );
    ensure!(
        crate::terminal::command_status(Command::new(python).args(["-B", "-m", "pip", "check"]))?
            .success(),
        "External dependency validation failed"
    );
    ensure!(
        crate::terminal::command_status(Command::new(python).args(["-B", "-c", validation]))?
            .success(),
        "External runtime validation failed"
    );
    Ok(())
}

// Capture bounded output without a pipe deadlock; only ephemeral scratch files
// are written. Probes do not load models or modify the installation.
pub(crate) fn command_output(
    command: &mut Command,
    timeout: Duration,
) -> Result<(Vec<u8>, Vec<u8>)> {
    command_output_with_cancel(command, timeout, None)
}

pub(crate) fn command_output_with_cancel(
    command: &mut Command,
    timeout: Duration,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<(Vec<u8>, Vec<u8>)> {
    use std::io::{Seek, SeekFrom};
    struct Probe(std::process::Child, bool);
    impl Drop for Probe {
        fn drop(&mut self) {
            #[cfg(unix)]
            if self.1 {
                // Only a group created below for this owned scanner process.
                unsafe {
                    libc::kill(-(self.0.id() as i32), libc::SIGKILL);
                }
            }
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let output = tempfile::tempfile()?;
    let errors = tempfile::tempfile()?;
    #[cfg(unix)]
    if cancel.is_some() {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = Probe(
        command
            .stdin(Stdio::null())
            .stdout(output.try_clone()?)
            .stderr(errors.try_clone()?)
            .spawn()
            .context("Could not start installed runtime metadata probe")?,
        cancel.is_some(),
    );
    let start = Instant::now();
    loop {
        match child.0.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    let mut errors = errors.try_clone()?;
                    let length = errors.metadata()?.len();
                    errors.seek(SeekFrom::Start(length.saturating_sub(8192)))?;
                    let mut detail = Vec::new();
                    errors.take(8192).read_to_end(&mut detail)?;
                    anyhow::bail!(
                        "Installed runtime metadata probe exited with {status}: {}",
                        crate::terminal::clean(&String::from_utf8_lossy(&detail)).trim()
                    );
                }
                break;
            }
            Ok(None)
                if !cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                    && start.elapsed() < timeout
                    && output.metadata()?.len() <= 8 * 1024 * 1024
                    && errors.metadata()?.len() <= 8 * 1024 * 1024 =>
            {
                thread::sleep(Duration::from_millis(20))
            }
            _ => {
                anyhow::bail!(
                    "Runtime metadata probe exceeded its time/output limit or could not be monitored"
                );
            }
        }
    }
    let read = |mut file: fs::File| -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        file.seek(SeekFrom::Start(0))?;
        file.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= 8 * 1024 * 1024,
            "Metadata probe output is too large"
        );
        Ok(bytes)
    };
    Ok((read(output)?, read(errors)?))
}

fn hex(value: &str, min: usize, max: usize) -> bool {
    (min..=max).contains(&value.len()) && value.bytes().all(|c| c.is_ascii_hexdigit())
}

fn llama_revision(output: &str) -> Option<&str> {
    output.lines().find_map(|line| {
        let (_, version) = line.trim().split_once("version:")?;
        let (_, rest) = version.split_once('(')?;
        let (revision, _) = rest.split_once(')')?;
        hex(revision, 7, 40).then_some(revision)
    })
}

fn check_llama(
    path: &Path,
    offline: bool,
    pinned: bool,
    remote: &mut impl Metadata,
    entry: &mut Entry,
) -> Result<()> {
    let (mut bytes, errors) =
        command_output(Command::new(path).arg("--version"), Duration::from_secs(5))?;
    bytes.push(b'\n');
    bytes.extend(errors);
    let output = String::from_utf8_lossy(&bytes);
    let Some(revision) = llama_revision(&output) else {
        entry.detail = "The installed binary did not report a verifiable llama.cpp revision".into();
        return Ok(());
    };
    entry.installed = Some(revision.into());
    if offline {
        entry.detail = "Offline; upstream revision not checked".into();
    } else if pinned {
        entry.status = Status::Pinned;
        entry.detail = "Werk CUDA profile uses a pinned and patched source revision; update through a compatible Werk release".into();
    } else {
        let upstream =
            remote.get("https://api.github.com/repos/ggml-org/llama.cpp/commits/HEAD")?;
        let latest = upstream["sha"]
            .as_str()
            .filter(|s| hex(s, 40, 40))
            .context("Upstream commit is missing")?;
        entry.available = Some(latest.into());
        if latest.starts_with(revision) {
            entry.status = Status::UpToDate;
            entry.detail = "Binary matches upstream HEAD".into();
        } else {
            let comparison = remote.get(&format!("https://api.github.com/repos/ggml-org/llama.cpp/compare/{revision}...{latest}?per_page=1"))?;
            match comparison["status"].as_str() {
                Some("ahead") => {
                    entry.status = Status::UpdateAvailable;
                    entry.detail = "Upstream contains newer commits. Update the managed source checkout and rebuild; backend install alone reuses its existing checkout".into();
                }
                Some("identical" | "behind") => {
                    entry.status = Status::UpToDate;
                    entry.detail = "Installed binary is at or ahead of upstream HEAD".into();
                }
                _ => {
                    entry.detail =
                        "Installed and upstream histories cannot be compared as a forward update"
                            .into();
                }
            }
        }
    }
    Ok(())
}

fn repository_url(repo: &str) -> Result<String> {
    let parts: Vec<_> = repo.split('/').collect();
    ensure!(
        parts.len() == 2
            && parts.iter().all(|p| !p.is_empty()
                && *p != "."
                && *p != ".."
                && p.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))),
        "Invalid Hugging Face repository identity"
    );
    Ok(format!(
        "https://huggingface.co/api/models/{repo}?blobs=true"
    ))
}

fn scan_model(
    store: &ModelStore,
    dir: &Path,
    model: &ModelManifest,
    offline: bool,
    remote: &mut impl Metadata,
    entries: &mut Vec<Entry>,
) {
    let mut entry = Entry::new("model", &model.id, dir.into());
    let tracked = tracked_model_source(store, model);
    let result = (|| -> Result<()> {
        let Some((repo, revision)) = tracked.as_ref().map_err(|e| anyhow!("{e:#}"))? else {
            entry.detail = "Local or provider model has no tracked upstream repository".into();
            return Ok(());
        };
        ensure!(hex(revision, 40, 40), "Invalid recorded model revision");
        entry.installed = Some(revision.clone());
        if offline {
            entry.detail = "Offline; upstream revision not checked".into();
            return Ok(());
        }
        let info = remote.get(&repository_url(repo)?)?;
        let latest = info["sha"]
            .as_str()
            .filter(|s| hex(s, 40, 40))
            .context("Upstream model revision is missing")?;
        entry.available = Some(latest.into());
        entry.status = if revision == latest {
            Status::UpToDate
        } else {
            Status::UpdateAvailable
        };
        entry.detail = if revision == latest { "Recorded model revision matches upstream" } else { "Repository revision changed; changes may include metadata. Review the model and its matching vision assets together" }.into();
        Ok(())
    })();
    if let Err(e) = result {
        entry.fail(e);
    }
    entries.push(entry);
    for file in &model.files {
        let name = file.path.to_ascii_lowercase();
        if !(name.ends_with(".gguf") && (name.contains("mmproj") || name.contains("projector"))) {
            continue;
        }
        let path = store.absolute_model_file(model, &file.path);
        let mut entry = Entry::new(
            "projector",
            format!("{}/{}", model.id, file.path),
            path.clone(),
        );
        let result = (|| -> Result<()> {
            ensure!(path.is_file(), "Registered projector file is missing");
            let source = dir.join("vision-source.json");
            let saved: Value = if source.is_file() {
                read_json(&source)?
            } else {
                Value::Null
            };
            entry.installed = Some(file.checksum.clone());
            if offline {
                entry.detail = "Offline; upstream projector not checked".into();
                return Ok(());
            }
            // Receipts from werk setup refer to the selected projector, whose
            // manifest may still use CRC32. Verify its bytes against the saved
            // SHA-256 before using that receipt's repository identity.
            let selected = model
                .metadata
                .vision_setup
                .as_ref()
                .and_then(|s| s.projector.as_deref())
                == Some(&file.path);
            let uses_receipt = selected && saved["sha256"].as_str().is_some();
            let (repo, filename) = if uses_receipt {
                (
                    saved["repo"]
                        .as_str()
                        .context("Missing projector repository")?,
                    saved["file"]
                        .as_str()
                        .context("Missing projector filename")?,
                )
            } else if let Some((repo, _)) = tracked.as_ref().ok().and_then(|s| s.as_ref()) {
                // configure_vision copies external/manual projectors here; the
                // model's repository is not evidence for those copied assets.
                if !file.path.starts_with("files/") || file.path.starts_with("files/vision/") {
                    entry.detail = "Local projector has no matching source receipt".into();
                    return Ok(());
                }
                (repo.as_str(), file.path.strip_prefix("files/").unwrap())
            } else {
                return Ok(());
            };
            let checksum = projector_sha256(&path, &file.checksum)?;
            entry.installed = Some(checksum.clone());
            if uses_receipt
                && !saved["sha256"]
                    .as_str()
                    .is_some_and(|sha| sha.eq_ignore_ascii_case(&checksum))
            {
                entry.detail = "Projector differs from its saved source receipt; source identity cannot be verified".into();
                return Ok(());
            }
            let info = remote.get(&repository_url(repo)?)?;
            let upstream = info["siblings"]
                .as_array()
                .context("Upstream file list is missing")?
                .iter()
                .find(|f| f["rfilename"].as_str() == Some(filename));
            let Some(latest) = upstream
                .and_then(|f| f["lfs"]["sha256"].as_str())
                .filter(|s| hex(s, 64, 64))
            else {
                entry.detail = "Upstream file is missing or has no comparable LFS checksum".into();
                return Ok(());
            };
            entry.available = Some(latest.into());
            entry.status = if checksum.eq_ignore_ascii_case(latest) {
                Status::UpToDate
            } else {
                Status::UpdateAvailable
            };
            entry.detail = if entry.status == Status::UpToDate { "Local projector SHA-256 matches upstream" } else { "Projector changed upstream; keep it paired with a compatible model revision before replacing it" }.into();
            Ok(())
        })();
        if let Err(e) = result {
            entry.fail(e);
        }
        entries.push(entry);
    }
}

fn projector_sha256(path: &Path, recorded: &str) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = fs::File::open(path)?;
    let mut sha = Sha256::new();
    let mut crc = crc32fast::Hasher::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        sha.update(&buffer[..n]);
        crc.update(&buffer[..n]);
    }
    let sha = format!("{:x}", sha.finalize());
    let actual = if recorded.starts_with("crc32:") {
        format!("crc32:{:08x}", crc.finalize())
    } else if recorded.starts_with("sha256:") {
        format!("sha256:{sha}")
    } else {
        anyhow::bail!("Projector has no supported recorded checksum");
    };
    ensure!(
        actual.eq_ignore_ascii_case(recorded),
        "Local projector differs from its manifest checksum; verify the installed artifact before comparing updates"
    );
    Ok(sha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::{ModelFile, ModelFormat, ModelMetadata, ModelStorage};
    use sha2::{Digest, Sha256};

    struct FixtureRemote {
        info: Value,
        calls: Vec<String>,
        fail: bool,
    }
    impl Metadata for FixtureRemote {
        fn get(&mut self, url: &str) -> Result<Value> {
            self.calls.push(url.into());
            ensure!(!self.fail, "fixture registry unavailable");
            Ok(self.info.clone())
        }
    }
    fn fixture() -> (tempfile::TempDir, ModelStore, FixtureRemote) {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(dir.path().join("store"))).unwrap();
        let files = store.model_dir("test").join("files");
        fs::create_dir_all(&files).unwrap();
        let data = b"projector fixture";
        fs::write(files.join("mmproj.gguf"), data).unwrap();
        let model = ModelManifest {
            id: "test".into(),
            source: ModelSource::HuggingFace {
                repo: "owner/model".into(),
            },
            storage: ModelStorage::Managed,
            format: ModelFormat::Gguf,
            architecture: None,
            tokenizer_path: None,
            config_path: None,
            model_path: None,
            backend: "llama-server".into(),
            created_unix: 0,
            files: vec![ModelFile {
                path: "files/mmproj.gguf".into(),
                size: data.len() as u64,
                checksum: format!("crc32:{:08x}", crc32fast::hash(data)),
            }],
            artifacts: vec![],
            metadata: ModelMetadata::default(),
        };
        fs::write(
            store.model_dir("test").join(MANIFEST_FILE),
            serde_json::to_vec(&model).unwrap(),
        )
        .unwrap();
        fs::write(
            store.model_dir("test").join("source-revision.json"),
            json!({"repo": "owner/model", "revision": "a".repeat(40)}).to_string(),
        )
        .unwrap();
        let remote = FixtureRemote {
            info: json!({"sha": "a".repeat(40), "siblings": [{"rfilename": "mmproj.gguf", "lfs": {"sha256": format!("{:x}", Sha256::digest(data))}}]}),
            calls: vec![],
            fail: false,
        };
        (dir, store, remote)
    }
    #[test]
    fn crc32_projectors_are_compared_by_actual_sha256_without_changing_store() {
        let (_dir, store, mut remote) = fixture();
        let path = store.model_dir("test").join(MANIFEST_FILE);
        let before = fs::read(&path).unwrap();
        let report = scan(&store, ScanScope::Artifacts, false, &mut remote).unwrap();
        assert_eq!(report.summary.up_to_date, 2);
        remote.info["sha"] = "b".repeat(40).into();
        remote.info["siblings"][0]["lfs"]["sha256"] = "c".repeat(64).into();
        let report = scan(&store, ScanScope::Artifacts, false, &mut remote).unwrap();
        assert_eq!(report.summary.update_available, 2);
        assert_eq!(fs::read(path).unwrap(), before);
    }
    #[test]
    fn offline_scan_has_no_upstream_calls_and_keeps_missing_home_absent() {
        let (dir, store, mut remote) = fixture();
        let report = scan(&store, ScanScope::Artifacts, true, &mut remote).unwrap();
        assert_eq!(report.summary.unknown, 2);
        assert!(remote.calls.is_empty());
        let empty = ModelStore::resolve(Some(dir.path().join("absent"))).unwrap();
        assert!(
            scan(&empty, ScanScope::Artifacts, true, &mut remote)
                .unwrap()
                .entries
                .is_empty()
        );
        assert!(!empty.home().exists());
    }
    #[test]
    fn registry_failure_and_local_corruption_do_not_hide_other_entries() {
        let (_dir, store, mut remote) = fixture();
        remote.fail = true;
        let report = scan(&store, ScanScope::Artifacts, false, &mut remote).unwrap();
        assert_eq!(report.summary.errors, 2);
        remote.fail = false;
        fs::write(store.model_dir("test").join("files/mmproj.gguf"), "changed").unwrap();
        let report = scan(&store, ScanScope::Artifacts, false, &mut remote).unwrap();
        assert_eq!(report.summary.up_to_date, 1);
        assert_eq!(report.summary.errors, 1);
    }
    #[test]
    fn binary_version_and_repository_identity_are_not_guessed() {
        assert_eq!(
            llama_revision("driver info\nversion: 1234 (abcdef123)\nbuilt with gcc"),
            Some("abcdef123")
        );
        assert_eq!(llama_revision("version: 1.2.3"), None);
        for bad in [
            "owner/../repo",
            "https://host/repo",
            "owner/repo?token=x",
            "../repo",
        ] {
            assert!(repository_url(bad).is_err());
        }
    }
    #[cfg(unix)]
    #[test]
    fn broken_external_environment_is_rejected_before_install_or_resolution() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().unwrap();
        let python = temp.path().join("python");
        let calls = temp.path().join("calls");
        fs::write(&python, format!(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\nprintf 'torchvision requires torch==2.6.0\\n' >&2\nexit 1\n",
            calls.display()
        )).unwrap();
        fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
        let error = update_external_python("media", &python).unwrap_err();
        assert!(error.to_string().contains("no packages changed"));
        assert_eq!(fs::read_to_string(calls).unwrap(), "-B -m pip check\n");
    }

    #[cfg(unix)]
    #[test]
    fn probes_keep_stderr_out_of_json_and_stop_on_timeout() {
        let (out, err) = command_output(
            Command::new("sh").args(["-c", "printf '[]'; printf warning >&2"]),
            Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(out, b"[]");
        assert_eq!(err, b"warning");
        let failure = command_output(
            Command::new("sh").args(["-c", "printf 'dependency conflict' >&2; exit 3"]),
            Duration::from_secs(2),
        )
        .unwrap_err();
        assert!(failure.to_string().contains("dependency conflict"));
        let start = Instant::now();
        assert!(
            command_output(
                Command::new("sh").args(["-c", "exec sleep 5"]),
                Duration::from_millis(50)
            )
            .is_err()
        );
        assert!(start.elapsed() < Duration::from_secs(2));
    }
}
