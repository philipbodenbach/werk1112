//! Werk-owned oMLX workers. Discovery and compatibility checks never load weights.
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, VecDeque},
    env,
    ffi::OsString,
    fs,
    hash::{Hash, Hasher},
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::{Component, Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    ChatGenerationSession, GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent,
    GenerationBackend, GenerationTimings,
    openai_transport::{
        HttpDeadline, OpenAiCompletion, SseAccumulator, append_assistant_content,
        chat_completion_body, delta_content, delta_tool_calls, ensure_visible_completion,
        finalize_completion_stats, request_with_bearer, send_stream_result, send_text_chunk,
        send_tool_call_delta, stream_body, update_completion_from_event,
        update_completion_from_message,
    },
};
use crate::{
    capabilities::InferenceTask,
    inference::{TaskReadiness, TaskReadinessStatus},
    model_store::{ModelFormat, ModelManifest, ModelRuntimeIdentity, ModelStore},
    runtime_control::{BackendRuntimeAdapter, ModelResidencyStatus, StaticRuntimeAdapter},
};

const PROBE: &str = include_str!("omlx_probe.py");
const SUPERVISOR: &str = include_str!("omlx_supervisor.py");
const PROBE_TIMEOUT: Duration = Duration::from_secs(20);
const DEFAULT_HEALTH_TIMEOUT: Duration = Duration::from_secs(900);
const POLL: Duration = Duration::from_millis(100);
const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;

fn console_script_source(source: &str) -> String {
    // `python -c` otherwise imports from the caller's current directory before
    // the probe/supervisor can set the console entry point's import location.
    // sys is built in; no filesystem module is imported until this is fixed.
    format!(
        "import sys\n_werk_launcher_parent = sys.argv.pop(1)\nif not (getattr(sys.flags, 'safe_path', False) or sys.flags.isolated):\n    sys.path[0] = _werk_launcher_parent\ndel _werk_launcher_parent\n{source}"
    )
}

#[derive(Clone)]
pub struct OmlxBackend {
    store: ModelStore,
    // Snapshot once. The same invocation is used for model preflight and startup.
    invocation: std::result::Result<OmlxInvocation, String>,
    servers: Arc<Mutex<HashMap<String, Arc<OmlxProcess>>>>,
    #[cfg(test)]
    test_probe: Option<ProbeReport>,
}

#[derive(Clone)]
struct OmlxInvocation {
    launcher: PathBuf,
    python: PathBuf,
    python_args: Vec<String>,
    health_timeout: Duration,
    environment: Vec<(OsString, OsString)>,
    launcher_fingerprint: u64,
    working_directory: PathBuf,
}

impl std::fmt::Debug for OmlxInvocation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Environment values may contain credentials. Only a fingerprint enters
        // configuration identities; no environment contents enter diagnostics.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.environment.hash(&mut hasher);
        formatter
            .debug_struct("OmlxInvocation")
            .field("launcher", &self.launcher)
            .field("python", &self.python)
            .field("python_args", &self.python_args)
            .field("timeout", &self.health_timeout)
            .field("environment", &hasher.finish())
            .field("working_directory", &self.working_directory)
            .field("launcher_fingerprint", &self.launcher_fingerprint)
            .finish()
    }
}

#[derive(Clone, Debug)]
struct ProbeReport {
    detail: String,
    version: String,
    tools: bool,
    tool_calling_detail: Option<String>,
    runtime: Value,
}

struct OmlxProcess {
    child: Mutex<Child>,
    // EOF stops the owned worker even if Werk exits via a signal without Drop.
    parent_pipe: Option<ChildStdin>,
    url: String,
    api_key: String,
    model_name: String,
    model_dir: PathBuf,
    model_identity: Option<ModelRuntimeIdentity>,
    base_path: PathBuf,
    version: String,
    instance_id: String,
    tools: bool,
    log_tail: Arc<Mutex<VecDeque<String>>>,
}

struct OmlxChatSession {
    server: Arc<OmlxProcess>,
}

impl OmlxBackend {
    pub fn new(store: ModelStore) -> Self {
        Self {
            store,
            invocation: OmlxInvocation::discover().map_err(|error| format!("{error:#}")),
            servers: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(test)]
            test_probe: None,
        }
    }

    pub(crate) fn cache_identity(&self) -> String {
        format!("{:?}", self.invocation)
    }

    pub fn probe() -> Result<String> {
        let invocation = OmlxInvocation::discover()?;
        Ok(invocation.probe(None)?.detail)
    }

    pub fn probe_model(&self, manifest: &ModelManifest) -> Result<()> {
        self.model_probe(manifest).map(|_| ())
    }

    pub fn probe_tool_calling(&self, manifest: &ModelManifest) -> Result<bool> {
        let (_, report) = self.model_probe(manifest)?;
        if !report.tools {
            bail!(
                "{}",
                report
                    .tool_calling_detail
                    .as_deref()
                    .unwrap_or("oMLX model has no verified native tool parser")
            );
        }
        Ok(true)
    }

    fn invocation(&self) -> Result<&OmlxInvocation> {
        self.invocation
            .as_ref()
            .map_err(|detail| anyhow!(detail.clone()))
    }

    fn model_probe(&self, manifest: &ModelManifest) -> Result<(PathBuf, ProbeReport)> {
        let directory = resolve_model_dir(&self.store, manifest)?;
        #[cfg(test)]
        if let Some(report) = &self.test_probe {
            return Ok((directory, report.clone()));
        }
        let report = self
            .invocation()?
            .probe(Some(&directory))
            .with_context(|| format!("oMLX model '{}' is not verified compatible", manifest.id))?;
        Ok((directory, report))
    }

    fn cached_server(&self, manifest: &ModelManifest) -> Result<(Arc<OmlxProcess>, f64)> {
        let invocation = self.invocation()?;
        // Recheck metadata against this exact invocation before startup/reuse.
        let (directory, report) = self.model_probe(manifest)?;
        let identity = ModelRuntimeIdentity::from_manifest(manifest)?;
        let key = format!(
            "{}|{}|{identity}|{}",
            self.cache_identity(),
            directory.display(),
            report.runtime
        );
        // Serialize lookup + startup, so concurrent first requests cannot spawn
        // duplicate multi-hundred-GB workers for the same model.
        let mut servers = self
            .servers
            .lock()
            .map_err(|_| anyhow!("oMLX worker registry is poisoned"))?;
        if let Some(server) = servers.get(&key)
            && server.is_running()
        {
            return Ok((server.clone(), 0.0));
        }
        servers.retain(|_, server| server.is_running());
        let started = Instant::now();
        let mut server = OmlxProcess::start(&self.store, invocation, &directory, report)?;
        server.model_identity = Some(identity);
        let server = Arc::new(server);
        servers.insert(key, server.clone());
        Ok((server, started.elapsed().as_secs_f64()))
    }

    fn generate_inner(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<GenerateResponse> {
        reject_images(&request)?;
        validate_tool_options(&request)?;
        if request.requires_tool_calling() && !self.probe_tool_calling(manifest)? {
            bail!(
                "oMLX does not have a verified native tool parser for model '{}'",
                manifest.id
            );
        }
        let started = Instant::now();
        let (server, load_seconds) = self.cached_server(manifest)?;
        let mut response = server.generate(&request, tx)?;
        response.timings.load_seconds = load_seconds;
        response.timings.total_seconds = started.elapsed().as_secs_f64();
        Ok(response)
    }
}

impl GenerationBackend for OmlxBackend {
    fn supports_tool_calling(&self, manifest: &ModelManifest, has_images: bool) -> bool {
        !has_images && self.probe_tool_calling(manifest).unwrap_or(false)
    }

    fn runtime_control_adapter(&self) -> Arc<dyn BackendRuntimeAdapter> {
        let server = self.servers.lock().ok().and_then(|servers| {
            let mut active = servers.values().filter(|server| server.is_running());
            let first = active.next().cloned();
            if active.next().is_some() { None } else { first }
        });
        Arc::new(residency_adapter(server.as_deref()))
    }

    fn runtime_control_adapter_for(
        &self,
        manifest: &ModelManifest,
    ) -> Result<Arc<dyn BackendRuntimeAdapter>> {
        let directory = resolve_model_dir(&self.store, manifest)?;
        let identity = ModelRuntimeIdentity::from_manifest(manifest)?;
        let servers = self
            .servers
            .lock()
            .map_err(|_| anyhow!("oMLX worker registry is poisoned"))?;
        let server = servers.values().find(|server| {
            server.model_dir == directory
                && server.model_identity.as_ref() == Some(&identity)
                && server.is_running()
        });
        Ok(Arc::new(residency_adapter(server.map(Arc::as_ref))))
    }

    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        self.cached_server(manifest).map(|_| ())
    }

    fn start_chat_session(
        &self,
        manifest: &ModelManifest,
        _seed: Option<u64>,
    ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
        let (server, _) = self.cached_server(manifest)?;
        Ok(Some(Box::new(OmlxChatSession { server })))
    }

    fn task_readiness(
        &self,
        manifest: &ModelManifest,
        task: InferenceTask,
    ) -> Option<TaskReadiness> {
        if task != InferenceTask::TextGeneration {
            return None;
        }
        let result = self.probe_model(manifest);
        Some(TaskReadiness {
            status: if result.is_ok() {
                TaskReadinessStatus::Available
            } else {
                TaskReadinessStatus::Unavailable
            },
            detail: result
                .map(|_| {
                    "oMLX model metadata is compatible; weights have not been loaded".to_string()
                })
                .unwrap_or_else(|error| format!("{error:#}")),
            adapter: Some("omlx".to_string()),
            required_backend: Some("omlx".to_string()),
            install_command: None,
            fallback_backend: None,
            missing_dependencies: Vec::new(),
            missing_dependency_groups: Vec::new(),
        })
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        self.generate_inner(manifest, request, None)
    }

    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        let backend = self.clone();
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            let result = backend.generate_inner(&manifest, request, Some(tx.clone()));
            send_stream_result(tx, result);
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

impl ChatGenerationSession for OmlxChatSession {
    fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        self.server.generate(&request, None)
    }

    fn generate_stream(&self, request: GenerateRequest) -> GenerateStream {
        let server = self.server.clone();
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            let result = server.generate(&request, Some(tx.clone()));
            send_stream_result(tx, result);
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

fn residency_adapter(server: Option<&OmlxProcess>) -> StaticRuntimeAdapter {
    let adapter = StaticRuntimeAdapter::new("omlx").with_accelerator_family("mlx");
    match server {
        Some(server) => adapter.with_backend_version(&server.version).with_instance_id(&server.instance_id)
            .with_model_residency(ModelResidencyStatus::Supported, "Werk owns and reuses this active oMLX model worker; oMLX internal caches expose no Werk named state operations"),
        None => adapter.with_model_residency(ModelResidencyStatus::Unavailable, "no unique active oMLX worker is available to verify model residency"),
    }
}

fn reject_images(request: &GenerateRequest) -> Result<()> {
    if !request.image_urls.is_empty()
        || request
            .messages
            .iter()
            .filter_map(|message| message.content.as_ref())
            .any(|content| !content.image_urls().is_empty())
    {
        bail!("oMLX integration currently supports text requests only");
    }
    Ok(())
}

fn validate_tool_options(request: &GenerateRequest) -> Result<()> {
    use crate::openai::{ToolChoice, ToolChoiceMode};
    let Some(config) = &request.tool_config else {
        return Ok(());
    };
    if matches!(
        &config.tool_choice,
        Some(ToolChoice::Named(_) | ToolChoice::Mode(ToolChoiceMode::Required))
    ) {
        bail!("oMLX does not enforce required or named tool_choice; use auto or none");
    }
    if config.parallel_tool_calls.is_some() {
        bail!("oMLX does not implement parallel_tool_calls; omit this option");
    }
    if config
        .tools
        .as_ref()
        .is_some_and(|tools| tools.iter().any(|tool| tool.function.strict == Some(true)))
    {
        bail!("oMLX does not enforce strict tool schemas; omit strict or set it to false");
    }
    Ok(())
}

fn resolve_model_dir(store: &ModelStore, manifest: &ModelManifest) -> Result<PathBuf> {
    if !matches!(manifest.format, ModelFormat::Mlx | ModelFormat::SafeTensors) {
        bail!("oMLX supports MLX or Hugging Face safetensors text model directories");
    }
    let root = store.model_dir(&manifest.id);
    let mut candidates = Vec::new();
    if let Some(config) = &manifest.config_path {
        let relative = Path::new(config);
        if relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
        {
            bail!("oMLX manifest config_path must stay within the model directory");
        }
        if let Some(parent) = root.join(relative).parent() {
            candidates.push(parent.to_path_buf());
        }
    }
    candidates.push(root.join("files"));
    candidates.push(root);
    for candidate in candidates {
        if candidate.join("config.json").is_file() {
            return candidate
                .canonicalize()
                .context("cannot resolve oMLX physical model path");
        }
    }
    bail!(
        "oMLX requires a local model directory containing config.json for '{}'",
        manifest.id
    )
}

impl OmlxInvocation {
    fn discover() -> Result<Self> {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            bail!("oMLX local backend requires macOS on Apple Silicon");
        }
        let launcher = match env::var_os("WERK_OMLX_BIN") {
            Some(path) if !path.is_empty() => absolute_program(PathBuf::from(path))?,
            Some(_) => bail!("WERK_OMLX_BIN is empty"),
            None => find_program("omlx").context("oMLX is not installed; install oMLX and set WERK_OMLX_BIN to its Python CLI launcher")?,
        };
        let health_timeout = match env::var("WERK_OMLX_HEALTH_TIMEOUT_SECONDS") {
            Ok(value) => Duration::from_secs(
                value
                    .parse::<u64>()
                    .ok()
                    .filter(|seconds| *seconds > 0)
                    .context("WERK_OMLX_HEALTH_TIMEOUT_SECONDS must be a positive integer")?,
            ),
            Err(env::VarError::NotPresent) => DEFAULT_HEALTH_TIMEOUT,
            Err(_) => bail!("WERK_OMLX_HEALTH_TIMEOUT_SECONDS is not valid UTF-8"),
        };
        Self::from_launcher(launcher, health_timeout)
    }

    fn from_launcher(launcher: PathBuf, health_timeout: Duration) -> Result<Self> {
        let mut environment: Vec<_> = env::vars_os()
            .filter(|(name, _)| !name.to_string_lossy().starts_with("OMLX_"))
            .collect();
        environment.sort();
        let launcher = launcher
            .canonicalize()
            .context("oMLX launcher does not exist")?;
        let mut bytes = Vec::new();
        fs::File::open(&launcher)?
            .take(65537)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 65536 {
            bail!("cannot verify oMLX launcher: script exceeds 64 KiB");
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hasher);
        let launcher_fingerprint = hasher.finish();
        let text =
            std::str::from_utf8(&bytes).context("oMLX launcher must be a Python console script")?;
        let shebang = text
            .lines()
            .next()
            .and_then(|line| line.strip_prefix("#!"))
            .context("oMLX launcher must have a Python shebang")?;
        let words: Vec<&str> = shebang.split_whitespace().collect();
        let mut index = 0;
        if words.first().is_some_and(|word| {
            Path::new(word)
                .file_name()
                .is_some_and(|name| name == "env")
        }) {
            index += 1;
            if words.get(index) == Some(&"-S") {
                index += 1;
            }
        }
        let program = words
            .get(index)
            .context("oMLX shebang has no Python interpreter")?;
        if !Path::new(program)
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("python"))
        {
            bail!("cannot verify oMLX launcher Python interpreter");
        }
        let python_path = PathBuf::from(program);
        let python_path = if python_path.components().count() == 1 {
            environment
                .iter()
                .find(|(name, _)| name == "PATH")
                .and_then(|(_, path)| {
                    env::split_paths(path)
                        .map(|directory| directory.join(program))
                        .find(|candidate| candidate.is_file())
                })
                .context("oMLX launcher Python is not present in the captured PATH")?
        } else {
            python_path
        };
        let python = if python_path.is_absolute() {
            python_path
        } else {
            env::current_dir()?.join(python_path)
        };
        if !python.is_file() {
            bail!("oMLX launcher Python does not exist: {}", python.display());
        }
        let python_args: Vec<String> = words[index + 1..]
            .iter()
            .map(|word| word.to_string())
            .collect();
        if python_args.iter().any(|flag| {
            !matches!(
                flag.as_str(),
                "-s" | "-S" | "-E" | "-I" | "-P" | "-B" | "-u"
            )
        }) {
            bail!("cannot verify custom oMLX launcher Python flags");
        }
        Ok(Self {
            launcher,
            python,
            python_args,
            health_timeout,
            environment,
            launcher_fingerprint,
            working_directory: env::current_dir()?,
        })
    }

    fn python_command(&self) -> Command {
        let mut command = Command::new(&self.python);
        command.args(&self.python_args);
        command.current_dir(&self.working_directory);
        command.env_clear().envs(self.environment.iter().cloned());
        command
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .env("PYTHONDONTWRITEBYTECODE", "1");
        command
    }

    fn probe(&self, model_dir: Option<&Path>) -> Result<ProbeReport> {
        self.verify_launcher()?;
        if let Some(directory) = model_dir {
            self.verify_import_paths(directory)?;
        }
        let mut command = self.python_command();
        command
            .args(["-c", &console_script_source(PROBE)])
            .arg(
                self.launcher
                    .parent()
                    .context("oMLX launcher has no parent directory")?,
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let payload = json!({"model_dir": model_dir, "launcher": self.launcher});
        let mut child = command
            .spawn()
            .context("failed to start selected oMLX metadata probe")?;
        let mut stdin = child.stdin.take().context("oMLX probe stdin unavailable")?;
        let bytes = serde_json::to_vec(&payload)?;
        let writer = thread::spawn(move || stdin.write_all(&bytes));
        let stdout = child
            .stdout
            .take()
            .context("oMLX probe stdout unavailable")?;
        let stderr = child
            .stderr
            .take()
            .context("oMLX probe stderr unavailable")?;
        let out = thread::spawn(move || read_bounded(stdout));
        let err = thread::spawn(move || read_bounded(stderr));
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if started.elapsed() >= PROBE_TIMEOUT {
                let _ = child.kill();
                let _ = child.wait();
                bail!("oMLX metadata compatibility probe timed out after 20 seconds");
            }
            thread::sleep(Duration::from_millis(20));
        };
        let _ = writer.join();
        let stdout = out.join().unwrap_or_default();
        let stderr = err.join().unwrap_or_default();
        let value: Value = String::from_utf8_lossy(&stdout)
            .lines()
            .rev()
            .find_map(|line| serde_json::from_str(line).ok())
            .with_context(|| {
                format!(
                    "oMLX probe returned no JSON: {}",
                    String::from_utf8_lossy(&stderr)
                )
            })?;
        let detail = value
            .get("detail")
            .and_then(Value::as_str)
            .unwrap_or("oMLX probe returned no detail")
            .to_string();
        if !status.success() || value.get("ok").and_then(Value::as_bool) != Some(true) {
            bail!("{detail}");
        }
        let runtime = value
            .get("runtime")
            .cloned()
            .context("oMLX probe omitted runtime identity")?;
        let version = runtime
            .get("omlx_version")
            .and_then(Value::as_str)
            .context("oMLX probe omitted version")?
            .to_string();
        Ok(ProbeReport {
            detail,
            version,
            tools: value
                .get("supports_tool_calling")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            tool_calling_detail: value
                .get("tool_calling_detail")
                .and_then(Value::as_str)
                .map(str::to_string),
            runtime,
        })
    }

    fn verify_import_paths(&self, model_dir: &Path) -> Result<()> {
        let model_dir = model_dir.canonicalize()?;
        let mut paths = vec![self.launcher.clone(), self.python.clone()];
        for (name, value) in &self.environment {
            if name == "PYTHONPATH" || name == "PYTHONHOME" {
                paths.extend(env::split_paths(value).map(|path| {
                    if path.is_absolute() {
                        path
                    } else {
                        self.working_directory.join(path)
                    }
                }));
            }
        }
        if paths.iter().any(|path| {
            path.starts_with(&model_dir)
                || path
                    .canonicalize()
                    .is_ok_and(|path| path.starts_with(&model_dir))
        }) {
            bail!(
                "cannot verify oMLX Python environment: its launcher, interpreter or Python import paths include the model repository; use an installation outside the model directory and remove model paths from PYTHONPATH/PYTHONHOME"
            );
        }
        Ok(())
    }

    fn verify_launcher(&self) -> Result<()> {
        let mut bytes = Vec::new();
        fs::File::open(&self.launcher)?
            .take(65537)
            .read_to_end(&mut bytes)?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hasher);
        if bytes.len() > 65536 || hasher.finish() != self.launcher_fingerprint {
            bail!(
                "oMLX launcher changed after runtime selection; select the updated runtime again"
            );
        }
        Ok(())
    }
}

fn absolute_program(path: PathBuf) -> Result<PathBuf> {
    // Preserve venv Python symlink paths: canonicalizing to the system Python
    // would select a different site-packages environment.
    let path = if path.components().count() == 1 {
        find_program(path.to_str().context("Python path is not UTF-8")?)
            .context("configured program is not on PATH")?
    } else if path.is_absolute() {
        path
    } else {
        env::current_dir()?.join(path)
    };
    if !path.is_file() {
        bail!("configured oMLX program does not exist: {}", path.display());
    }
    Ok(path)
}

fn find_program(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|directory| directory.join(name))
            .find(|candidate| candidate.is_file())
    })
}

fn read_bounded(mut reader: impl Read) -> Vec<u8> {
    let mut captured = Vec::new();
    let mut bytes = [0; 4096];
    while let Ok(count) = reader.read(&mut bytes) {
        if count == 0 {
            break;
        }
        let keep = count.min(65536_usize.saturating_sub(captured.len()));
        captured.extend_from_slice(&bytes[..keep]);
    }
    captured
}

impl OmlxProcess {
    fn start(
        store: &ModelStore,
        invocation: &OmlxInvocation,
        model_dir: &Path,
        report: ProbeReport,
    ) -> Result<Self> {
        invocation.verify_launcher()?;
        invocation.verify_import_paths(model_dir)?;
        let instance_id = random_id()?;
        let port = TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port();
        let api_key = random_id()?;
        let base_path = store
            .home()
            .join("backends")
            .join("omlx")
            .join("workers")
            .join(&instance_id);
        fs::create_dir_all(&base_path).context("cannot create isolated oMLX base path")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&base_path, fs::Permissions::from_mode(0o700))?;
        }
        let mut command = invocation.python_command();
        command
            .args(["-c", &console_script_source(SUPERVISOR)])
            .arg(
                invocation
                    .launcher
                    .parent()
                    .context("oMLX launcher has no parent directory")?,
            )
            .arg(&invocation.launcher)
            .arg("serve")
            .arg("--model-dir")
            .arg(model_dir)
            .arg("--base-path")
            .arg(&base_path)
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--api-key",
                &api_key,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let deadline = HttpDeadline::new(invocation.health_timeout);
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                let _ = fs::remove_dir_all(&base_path);
                return Err(error).context("failed to start selected oMLX launcher");
            }
        };
        let log_tail = Arc::new(Mutex::new(VecDeque::new()));
        let parent_pipe = child.stdin.take();
        if let Some(stdout) = child.stdout.take() {
            spawn_log_reader(stdout, log_tail.clone(), api_key.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            spawn_log_reader(stderr, log_tail.clone(), api_key.clone());
        }
        // Create the owner before waiting. Every startup failure now kills and
        // reaps the child and removes only this process's private base path.
        let mut process = Self {
            child: Mutex::new(child),
            parent_pipe,
            url: format!("http://127.0.0.1:{port}"),
            api_key,
            model_name: String::new(),
            model_dir: model_dir.to_path_buf(),
            model_identity: None,
            base_path,
            version: report.version,
            instance_id,
            tools: report.tools,
            log_tail,
        };
        process
            .wait_and_load(deadline)
            .with_context(|| format!("oMLX startup failed{}", process.formatted_log_tail()))?;
        Ok(process)
    }

    fn wait_and_load(&mut self, deadline: HttpDeadline) -> Result<()> {
        let mut last_error = "server has not responded".to_string();
        loop {
            if !self.is_running() {
                bail!("oMLX server exited before becoming ready");
            }
            let remaining = deadline
                .remaining()
                .with_context(|| format!("timed out waiting for oMLX health: {last_error}"))?;
            match self.json_request(
                "GET",
                "/health",
                None,
                remaining.min(Duration::from_secs(3)),
            ) {
                Ok(health) if health.get("status").and_then(Value::as_str) == Some("healthy") => {
                    break;
                }
                Ok(_) => last_error = "oMLX health still reports loading".to_string(),
                Err(error) => last_error = format!("{error:#}"),
            }
            thread::sleep(POLL.min(deadline.remaining().unwrap_or(Duration::ZERO)));
        }
        let status = self.json_request(
            "GET",
            "/api/status",
            None,
            deadline.remaining()?.min(Duration::from_secs(3)),
        )?;
        if status.get("version").and_then(Value::as_str) != Some(self.version.as_str()) {
            bail!(
                "oMLX running server version does not match the probed interpreter ({})",
                self.version
            );
        }
        let models = self.json_request(
            "GET",
            "/v1/models/status",
            None,
            deadline.remaining()?.min(Duration::from_secs(3)),
        )?;
        self.model_name = physical_model_name(&models, &self.model_dir)?;
        let path = format!("/v1/models/{}/load", encode_path_segment(&self.model_name));
        let loaded = self.json_request("POST", &path, Some(&json!({})), deadline.remaining()?)?;
        if loaded.get("status").and_then(Value::as_str) != Some("ok")
            || loaded.get("model_id").and_then(Value::as_str) != Some(self.model_name.as_str())
        {
            bail!("oMLX did not confirm loading the selected physical model");
        }
        let models = self.json_request(
            "GET",
            "/v1/models/status",
            None,
            deadline.remaining()?.min(Duration::from_secs(3)),
        )?;
        if physical_model_name(&models, &self.model_dir)? != self.model_name
            || !models
                .get("models")
                .and_then(Value::as_array)
                .is_some_and(|models| {
                    models.iter().any(|entry| {
                        entry.get("id").and_then(Value::as_str) == Some(self.model_name.as_str())
                            && entry.get("loaded").and_then(Value::as_bool) == Some(true)
                    })
                })
        {
            bail!("oMLX load completed without a resident matching physical model");
        }
        Ok(())
    }

    fn json_request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        timeout: Duration,
    ) -> Result<Value> {
        let mut response = request_with_bearer(
            &self.url,
            path,
            method,
            body,
            Some(timeout),
            Some(&self.api_key),
        )
        .with_context(|| format!("oMLX {method} {path} failed"))?;
        if response.status != 200 {
            bail!("oMLX {method} {path} returned HTTP {}", response.status);
        }
        let mut bytes = Vec::new();
        stream_body(&mut response, |chunk| {
            if bytes.len().saturating_add(chunk.len()) > MAX_JSON_BYTES {
                bail!("oMLX JSON response exceeds 16 MiB");
            }
            bytes.extend_from_slice(chunk);
            Ok(())
        })?;
        let value: Value = serde_json::from_slice(&bytes).context("oMLX returned invalid JSON")?;
        reject_upstream_error(&value)?;
        Ok(value)
    }

    fn generate(
        &self,
        request: &GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<GenerateResponse> {
        reject_images(request)?;
        validate_tool_options(request)?;
        if request.requires_tool_calling() && !self.tools {
            bail!("oMLX model does not have a verified native tool parser");
        }
        if !self.is_running() {
            bail!("oMLX model worker has exited{}", self.formatted_log_tail());
        }
        let started = Instant::now();
        let body = chat_completion_body(&self.model_name, request, tx.is_some());
        let mut response = super::openai_transport::request_with_bearer_cancellable(
            &self.url,
            "/v1/chat/completions",
            "POST",
            Some(&body),
            None,
            Some(&self.api_key),
            tx.clone(),
        )
        .context("oMLX chat completion request failed")?;
        if response.status != 200 {
            bail!("oMLX chat completion returned HTTP {}", response.status);
        }
        let mut completion = OpenAiCompletion {
            finish_reason: "length".to_string(),
            ..Default::default()
        };
        if tx.is_some() {
            let mut sse = SseAccumulator::default();
            let mut done = false;
            stream_body(&mut response, |bytes| {
                sse.push(bytes, |event| {
                    if event.trim() == "[DONE]" {
                        done = true;
                        return Ok(());
                    }
                    if done {
                        bail!("oMLX returned SSE data after [DONE]");
                    }
                    let value: Value =
                        serde_json::from_str(event).context("oMLX returned invalid SSE JSON")?;
                    reject_upstream_error(&value)?;
                    update_completion_from_event(&mut completion, &value);
                    if let Some(chunk) = delta_content(&value) {
                        if !chunk.is_empty() && completion.first_token_seconds <= 0.0 {
                            completion.first_token_seconds = started.elapsed().as_secs_f64();
                        }
                        completion.text.push_str(&chunk);
                        append_assistant_content(&mut completion.assistant_content, &chunk);
                        if !chunk.is_empty() {
                            send_text_chunk(&tx, chunk)?;
                        }
                    }
                    if let Some(calls) = delta_tool_calls(&value)? {
                        if !calls.is_empty() {
                            completion.saw_tool_call_delta = true;
                            if completion.first_token_seconds <= 0.0 {
                                completion.first_token_seconds = started.elapsed().as_secs_f64();
                            }
                            send_tool_call_delta(&tx, calls)?;
                        }
                    }
                    Ok(())
                })
            })?;
            if !done || sse.pending.iter().any(|byte| !byte.is_ascii_whitespace()) {
                bail!("oMLX stream ended before a complete [DONE] event");
            }
        } else {
            let mut bytes = Vec::new();
            stream_body(&mut response, |chunk| {
                if bytes.len().saturating_add(chunk.len()) > MAX_JSON_BYTES {
                    bail!("oMLX completion exceeds 16 MiB");
                }
                bytes.extend_from_slice(chunk);
                Ok(())
            })?;
            let value: Value = serde_json::from_slice(&bytes)
                .context("oMLX returned invalid chat completion JSON")?;
            reject_upstream_error(&value)?;
            update_completion_from_event(&mut completion, &value);
            update_completion_from_message(&mut completion, &value)?;
            completion.first_token_seconds = started.elapsed().as_secs_f64();
        }
        ensure_visible_completion(&completion).context("oMLX completion has no visible answer")?;
        finalize_completion_stats(&mut completion, request, started.elapsed().as_secs_f64());
        let assistant_message = completion.assistant_message();
        Ok(GenerateResponse {
            text: completion.text,
            assistant_message: Some(assistant_message),
            prompt_tokens: completion.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            finish_reason: completion.finish_reason,
            timings: GenerationTimings {
                first_token_seconds: completion.first_token_seconds,
                prompt_seconds: completion.prompt_seconds,
                decode_seconds: completion.decode_seconds,
                total_seconds: started.elapsed().as_secs_f64(),
                ..Default::default()
            },
            backend_diagnostics: Vec::new(),
        })
    }

    fn is_running(&self) -> bool {
        self.child
            .lock()
            .ok()
            .and_then(|mut child| child.try_wait().ok())
            .is_some_and(|status| status.is_none())
    }

    fn formatted_log_tail(&self) -> String {
        self.log_tail
            .lock()
            .ok()
            .filter(|tail| !tail.is_empty())
            .map(|tail| {
                format!(
                    "\noMLX output tail:\n{}",
                    tail.iter().cloned().collect::<Vec<_>>().join("\n")
                )
            })
            .unwrap_or_default()
    }
}

impl Drop for OmlxProcess {
    fn drop(&mut self) {
        drop(self.parent_pipe.take());
        if let Ok(child) = self.child.get_mut() {
            #[cfg(unix)]
            {
                // This process group was created by us. Kill its workers too;
                // never target an externally owned oMLX service.
                if child.try_wait().ok().flatten().is_none() {
                    unsafe {
                        libc::kill(-(child.id() as i32), libc::SIGKILL);
                    }
                }
            }
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = fs::remove_dir_all(&self.base_path);
    }
}

fn physical_model_name(value: &Value, directory: &Path) -> Result<String> {
    let models = value
        .get("models")
        .and_then(Value::as_array)
        .context("oMLX model status response has no models array")?;
    let matches: Vec<&str> = models
        .iter()
        .filter(|entry| entry.get("model_type").and_then(Value::as_str) == Some("llm"))
        .filter(|entry| {
            entry
                .get("model_path")
                .and_then(Value::as_str)
                .filter(|path| !path.contains("://"))
                .and_then(|path| Path::new(path).canonicalize().ok())
                .as_deref()
                == Some(directory)
        })
        .filter_map(|entry| entry.get("id").and_then(Value::as_str))
        .collect();
    match matches.as_slice() {
        [name] if !name.is_empty() => Ok(name.to_string()),
        [] => bail!(
            "oMLX does not advertise the selected physical text model path {}",
            directory.display()
        ),
        _ => bail!(
            "oMLX advertises multiple IDs for the selected physical model path {}",
            directory.display()
        ),
    }
}

fn reject_upstream_error(value: &Value) -> Result<()> {
    if let Some(error) = value.get("error").filter(|value| !value.is_null()) {
        bail!("oMLX upstream error: {error}");
    }
    Ok(())
}

fn encode_path_segment(value: &str) -> String {
    value
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
                (byte as char).to_string()
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect()
}

fn random_id() -> Result<String> {
    let mut bytes = [0; 24];
    getrandom::getrandom(&mut bytes)
        .map_err(|error| anyhow!("cannot create oMLX worker identity: {error}"))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn spawn_log_reader(
    reader: impl Read + Send + 'static,
    tail: Arc<Mutex<VecDeque<String>>>,
    api_key: String,
) {
    thread::spawn(move || {
        // Limit individual lines as well as retained line count.
        let mut reader = BufReader::new(reader);
        let mut bytes = Vec::new();
        loop {
            bytes.clear();
            let mut limited = (&mut reader).take(4096);
            if limited.read_until(b'\n', &mut bytes).unwrap_or(0) == 0 {
                break;
            }
            let line = String::from_utf8_lossy(&bytes).replace(&api_key, "[redacted]");
            if let Ok(mut tail) = tail.lock() {
                tail.push_back(line.trim_end().to_string());
                while tail.len() > 32 {
                    tail.pop_front();
                }
            }
        }
    });
}

#[cfg(test)]
mod tests;
