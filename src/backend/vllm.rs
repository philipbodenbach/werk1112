use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, VecDeque},
    env,
    ffi::OsStr,
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    BackendDoctorCheck, ChatGenerationSession, GenerateRequest, GenerateResponse, GenerateStream,
    GenerateStreamEvent, GeneratedAssistantMessage, GenerationBackend, GenerationTimings,
    SelectedRocmDeviceStatus, current_host_is_strix_halo,
    current_selected_rocm_device_is_strix_halo, current_selected_rocm_device_status,
};
use crate::{
    capabilities::InferenceTask,
    inference::{TaskReadiness, TaskReadinessStatus},
    model_store::{ModelFormat, ModelManifest, ModelRuntimeIdentity, ModelStore},
    openai::{ChatCompletionToolCall, ChatCompletionToolCallDelta},
    runtime_control::BackendRuntimeAdapter,
};

mod runtime_control;
use self::runtime_control::VllmRuntimeControlAdapter;

const DEFAULT_HEALTH_TIMEOUT_SECONDS: u64 = 300;
const DGX_SPARK_HEALTH_TIMEOUT_SECONDS: u64 = 900;
const STRIX_HALO_HEALTH_TIMEOUT_SECONDS: u64 = 900;
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(250);
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const HEALTH_REQUEST_IO_TIMEOUT: Duration = Duration::from_secs(5);
const REMOTE_DISCOVERY_PROBE_TIMEOUT: Duration = Duration::from_secs(3);
/// Exact Hugging Face `config.json.model_type` values for vLLM-backed VLMs.
///
/// Keep this deliberately narrower than vLLM's text architecture allowlist:
/// advertising a text-only sibling such as `qwen3` as image-capable would let
/// image requests reach a model which cannot consume their multipart content.
pub(crate) const VLLM_VISION_ARCHITECTURES: &[&str] = &[
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "glm4v",
    "glm4v_moe",
];
const WSL_VLLM_MESSAGE: &str = "vLLM is a Linux-native runtime. Your environment appears to be WSL, where vLLM can fail because required GPU memory features such as UVA/CUDA IPC are unavailable. Werk will fall back to Candle CUDA. For best vLLM support use native Linux or a remote vLLM server.";
const DGX_SPARK_VLLM_MESSAGE: &str = "DGX Spark detected (Linux aarch64 / GB10). Use NVIDIA's Spark-compatible vLLM container and expose its OpenAI endpoint, then set WERK_VLLM_HOST, WERK_VLLM_PORT, and, when its served name differs from the Werk model ID, WERK_VLLM_MODEL. A generic managed `pip install vllm` is intentionally not offered on DGX Spark.";
const STRIX_HALO_VLLM_MESSAGE: &str = "AMD Strix Halo detected (Linux x86_64 / gfx1151). A generic managed `pip install vllm` is intentionally not offered because Strix Halo requires a matching ROCm vLLM build. Use an official ROCm vLLM container and set WERK_VLLM_HOST plus WERK_VLLM_PORT, or set WERK_VLLM_PYTHON to a preprovisioned Python whose PyTorch reports both torch.version.hip and gfx1151. Set WERK_VLLM_ACCELERATOR=rocm (or WERK_VLLM_ROCM=1) for a remote ROCm endpoint.";
const LINUX_ARM64_MANAGED_VLLM_MESSAGE: &str = "Linux aarch64 detected without a verified DGX Spark/GB10 signal. Werk does not offer a generic managed `pip install vllm` on ARM64 because the compatible wheel/container depends on the concrete platform. Use a vendor-supported runtime and set WERK_VLLM_PYTHON, or expose an OpenAI-compatible vLLM endpoint and set WERK_VLLM_HOST plus WERK_VLLM_PORT.";

#[derive(Clone)]
pub struct VllmBackend {
    store: ModelStore,
    accelerator: VllmAccelerator,
    automatic_prefix_caching: Option<bool>,
    servers: Arc<Mutex<HashMap<String, Arc<VllmProcess>>>>,
    #[cfg(test)]
    test_server: Option<Arc<VllmProcess>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VllmAccelerator {
    Cuda,
    Rocm,
}

impl VllmAccelerator {
    fn backend_label(self) -> &'static str {
        match self {
            Self::Cuda => "vllm-cuda",
            Self::Rocm => "vllm-rocm",
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Rocm => "ROCm",
        }
    }
}

struct VllmProcess {
    child: Option<Mutex<Child>>,
    command_label: String,
    discovery_source: String,
    args: Vec<String>,
    model_dir: PathBuf,
    model_name: String,
    model_name_source: &'static str,
    is_nemotron: bool,
    url: String,
    pid: Option<u32>,
    log_tail: Arc<Mutex<VecDeque<String>>>,
    accelerator: VllmAccelerator,
    runtime_version: String,
    runtime_instance_id: String,
}

struct VllmChatSession {
    server: Arc<VllmProcess>,
    architecture: Option<String>,
}

pub(crate) fn vllm_architecture_supports_images(architecture: Option<&str>) -> bool {
    architecture.is_some_and(|architecture| {
        VLLM_VISION_ARCHITECTURES
            .iter()
            .any(|supported| supported.eq_ignore_ascii_case(architecture))
    })
}

fn multipart_image_urls(request: &GenerateRequest) -> Vec<String> {
    request
        .messages
        .iter()
        .filter_map(|message| message.content.as_ref())
        .flat_map(crate::openai::MessageContent::image_urls)
        .collect()
}

fn validate_vllm_image_request(
    architecture: Option<&str>,
    request: &GenerateRequest,
) -> Result<()> {
    let multipart_images = multipart_image_urls(request);
    if request.image_urls.is_empty() && multipart_images.is_empty() {
        return Ok(());
    }

    let architecture_label = architecture.unwrap_or("unknown");
    if !vllm_architecture_supports_images(architecture) {
        bail!(
            "vLLM image inputs require an explicitly supported VLM architecture; model architecture '{architecture_label}' is not one of: {}",
            VLLM_VISION_ARCHITECTURES.join(", ")
        );
    }
    if multipart_images.is_empty() {
        bail!(
            "vLLM image inputs must be present in the ordered OpenAI multipart messages; separate image URLs without multipart message content cannot be forwarded safely"
        );
    }
    if multipart_images != request.image_urls {
        bail!("vLLM image routing metadata does not match the ordered multipart message images");
    }
    for source in &multipart_images {
        let source = source.trim();
        if source.starts_with("data:image/")
            || source.starts_with("http://")
            || source.starts_with("https://")
        {
            continue;
        }
        bail!(
            "vLLM vision input does not expose local filesystem paths to its OpenAI endpoint; send an image data URL or HTTP(S) URL (CLI --image paths are converted automatically)"
        );
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct VllmDiscovery {
    pub command: Option<VllmCommand>,
    pub source: String,
    pub attempts: Vec<VllmDiscoveryAttempt>,
}

#[derive(Debug, Clone)]
pub struct VllmDiscoveryAttempt {
    pub label: String,
    pub path: Option<PathBuf>,
    pub exists: bool,
    pub usable: bool,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct VllmHealthStatus {
    pub installed_label: &'static str,
    pub health_label: &'static str,
    pub healthy: bool,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub enum VllmCommand {
    Python(PathBuf),
    Executable(PathBuf),
    Remote { host: String, port: u16 },
}

#[derive(Default)]
struct VllmCompletion {
    text: String,
    assistant_content: Option<Option<String>>,
    tool_calls: Option<Vec<ChatCompletionToolCall>>,
    saw_tool_call_delta: bool,
    saw_reasoning_content: bool,
    prompt_tokens: usize,
    completion_tokens: usize,
    prompt_seconds: f64,
    decode_seconds: f64,
    first_token_seconds: f64,
    finish_reason: String,
}

impl VllmCompletion {
    fn assistant_message(&self) -> GeneratedAssistantMessage {
        GeneratedAssistantMessage {
            content: self
                .assistant_content
                .clone()
                .unwrap_or_else(|| Some(self.text.clone())),
            tool_calls: self.tool_calls.clone(),
        }
    }
}

impl VllmBackend {
    pub fn new(store: ModelStore) -> Self {
        Self::new_with_accelerator(store, VllmAccelerator::Cuda)
    }

    pub fn new_rocm(store: ModelStore) -> Self {
        Self::new_with_accelerator(store, VllmAccelerator::Rocm)
    }

    fn new_with_accelerator(store: ModelStore, accelerator: VllmAccelerator) -> Self {
        Self {
            store,
            accelerator,
            automatic_prefix_caching: None,
            servers: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(test)]
            test_server: None,
        }
    }

    pub(crate) fn with_automatic_prefix_caching(mut self, enabled: Option<bool>) -> Self {
        self.automatic_prefix_caching = enabled;
        self
    }

    #[cfg(test)]
    pub(crate) fn with_mock_http_server(
        store: ModelStore,
        url: String,
        model_name: String,
    ) -> Self {
        let model_dir = store.model_dir(&model_name);
        let server = VllmProcess {
            child: None,
            command_label: "mock remote vLLM OpenAI server".to_string(),
            discovery_source: "test HTTP server".to_string(),
            args: Vec::new(),
            model_dir,
            model_name,
            model_name_source: "test served model",
            is_nemotron: false,
            url,
            pid: None,
            log_tail: Arc::new(Mutex::new(VecDeque::new())),
            accelerator: VllmAccelerator::Cuda,
            runtime_version: "test".to_string(),
            runtime_instance_id: "vllm-test-instance".to_string(),
        };
        Self {
            store,
            accelerator: VllmAccelerator::Cuda,
            automatic_prefix_caching: None,
            servers: Arc::new(Mutex::new(HashMap::new())),
            test_server: Some(Arc::new(server)),
        }
    }

    pub fn probe(store: &ModelStore) -> Result<String> {
        let discovery = discover_vllm(store);
        if let Some(reason) = local_vllm_platform_rejection_for_discovery(&discovery) {
            bail!("{reason}");
        }
        let Some(command) = discovery.command.as_ref() else {
            bail!("{}", missing_vllm_message(&discovery));
        };
        let configured_args = configured_vllm_args()?;
        validate_vllm_args_target(command, &configured_args)?;
        ensure_vllm_platform_eligible(command)?;
        vllm_cuda_capability(command)?;
        match command {
            VllmCommand::Remote { host, port } => Ok(if remote_discovery_ready(&discovery) {
                format!("vLLM OpenAI server at http://{host}:{port}")
            } else {
                format!(
                    "vLLM OpenAI server configured at http://{host}:{port}; not ready yet, execution will wait for /v1/models"
                )
            }),
            command => Ok(format!(
                "vLLM {} ({})",
                command.display(),
                vllm_version(command).unwrap_or_else(|| "version unknown".to_string())
            )),
        }
    }

    pub fn probe_rocm(store: &ModelStore) -> Result<String> {
        let discovery = require_vllm(store)?;
        let command = discovery
            .command
            .as_ref()
            .context("vLLM discovery had no command")?;
        let configured_args = configured_vllm_args()?;
        validate_vllm_args_target(command, &configured_args)?;
        ensure_vllm_platform_eligible(command)?;
        let detail = vllm_rocm_capability(command)?;
        Ok(format!("vLLM ROCm ({detail})"))
    }

    pub fn discover(store: &ModelStore) -> VllmDiscovery {
        discover_vllm(store)
    }

    pub fn health(store: &ModelStore) -> VllmHealthStatus {
        let discovery = discover_vllm(store);
        vllm_health(&discovery)
    }

    pub fn missing_message(store: &ModelStore) -> String {
        let discovery = discover_vllm(store);
        if let Some(reason) = local_vllm_platform_rejection_for_discovery(&discovery) {
            return reason.to_string();
        }
        missing_vllm_message(&discovery)
    }

    pub fn unavailable_reason(store: &ModelStore) -> String {
        Self::cuda_unavailable_reason(store)
    }

    pub fn cuda_unavailable_reason(store: &ModelStore) -> String {
        let discovery = discover_vllm(store);
        if let Some(reason) = local_vllm_platform_rejection_for_discovery(&discovery) {
            return reason.to_string();
        }
        if let Some(command) = discovery.command.as_ref()
            && let Err(err) = ensure_vllm_platform_eligible(command)
        {
            return compact_error(&err.to_string());
        }
        if let Some(command) = discovery.command.as_ref()
            && let Err(err) = vllm_cuda_capability(command)
        {
            return compact_error(&err.to_string());
        }
        concise_vllm_unavailable_reason(&discovery)
    }

    pub fn rocm_unavailable_reason(store: &ModelStore) -> String {
        let discovery = discover_vllm(store);
        if let Some(reason) = local_vllm_platform_rejection_for_discovery(&discovery) {
            return reason.to_string();
        }
        let Some(command) = discovery.command.as_ref() else {
            return concise_vllm_unavailable_reason(&discovery);
        };
        if let Err(err) = ensure_vllm_platform_eligible(command) {
            return compact_error(&err.to_string());
        }
        vllm_rocm_capability(command)
            .err()
            .map(|err| compact_error(&err.to_string()))
            .unwrap_or_else(|| "vLLM ROCm runtime is unavailable".to_string())
    }

    fn cached_server(&self, manifest: &ModelManifest) -> Result<(Arc<VllmProcess>, bool, f64)> {
        if manifest.format != ModelFormat::SafeTensors {
            bail!("vLLM backend supports HF safetensors model directories only");
        }

        #[cfg(test)]
        if let Some(server) = &self.test_server {
            return Ok((server.clone(), true, 0.0));
        }

        let discovery = discover_vllm(&self.store);
        let configured_args = configured_vllm_args()?;
        if let Some(command) = discovery.command.as_ref() {
            validate_vllm_args_target(command, &configured_args)?;
        }
        let configured_args = effective_vllm_args_for_target(
            configured_args,
            discovery.command.as_ref(),
            self.automatic_prefix_caching,
        );
        // A remote vLLM server owns and loads its weights. The installed Werk
        // manifest is still required for routing, but its local repository may
        // intentionally contain metadata only (common when Werk runs beside a
        // Spark container).
        let model_dir = resolve_vllm_model_dir_for_discovery(&self.store, manifest, &discovery)?;
        let model_identity = ModelRuntimeIdentity::from_manifest(manifest)?;
        let key = vllm_server_cache_key(
            &model_identity,
            &model_dir,
            &discovery,
            &VllmCacheEnvironment::current(&configured_args),
        );
        if let Some(server) = self
            .servers
            .lock()
            .map_err(|_| anyhow!("vLLM server cache mutex poisoned"))?
            .get(&key)
            .cloned()
            && server.is_running()
        {
            return Ok((server, true, 0.0));
        }

        let started = Instant::now();
        let server = Arc::new(VllmProcess::start(
            &self.store,
            manifest,
            &model_dir,
            discovery,
            self.accelerator,
            configured_args,
        )?);
        let load_seconds = started.elapsed().as_secs_f64();
        self.servers
            .lock()
            .map_err(|_| anyhow!("vLLM server cache mutex poisoned"))?
            .insert(key, server.clone());
        Ok((server, false, load_seconds))
    }

    fn generate_inner(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<GenerateResponse> {
        validate_vllm_image_request(manifest.architecture.as_deref(), &request)?;

        let total_started = Instant::now();
        let (server, reused, load_seconds) = self.cached_server(manifest)?;
        server.print_debug(&request, reused);
        let completion = server.complete(&request, tx)?;
        let assistant_message = completion.assistant_message();
        Ok(GenerateResponse {
            text: completion.text,
            assistant_message: Some(assistant_message),
            prompt_tokens: completion.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            finish_reason: completion.finish_reason,
            timings: GenerationTimings {
                load_seconds,
                warmup_seconds: 0.0,
                first_token_seconds: completion.first_token_seconds,
                prompt_seconds: completion.prompt_seconds,
                decode_seconds: completion.decode_seconds,
                total_seconds: total_started.elapsed().as_secs_f64(),
            },
            backend_diagnostics: Vec::new(),
        })
    }
}

impl GenerationBackend for VllmBackend {
    fn supports_tool_calling(&self, _manifest: &ModelManifest, _has_images: bool) -> bool {
        true
    }

    fn runtime_control_adapter(&self) -> Arc<dyn BackendRuntimeAdapter> {
        Arc::new(VllmRuntimeControlAdapter::new(self.clone()))
    }

    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        self.cached_server(manifest).map(|_| ())
    }

    fn start_chat_session(
        &self,
        manifest: &ModelManifest,
        _seed: Option<u64>,
    ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
        if manifest.format != ModelFormat::SafeTensors {
            return Ok(None);
        }
        let (server, _, _) = self.cached_server(manifest)?;
        Ok(Some(Box::new(VllmChatSession {
            server,
            architecture: manifest.architecture.clone(),
        })))
    }

    fn task_readiness(
        &self,
        manifest: &ModelManifest,
        task: InferenceTask,
    ) -> Option<TaskReadiness> {
        if task != InferenceTask::ImageUnderstanding {
            return None;
        }
        let readiness = if !manifest.supports_task(task) {
            Err(anyhow!(
                "model '{}' does not advertise image-understanding",
                manifest.id
            ))
        } else if manifest.format != ModelFormat::SafeTensors {
            Err(anyhow!(
                "vLLM image understanding requires a Hugging Face safetensors model"
            ))
        } else if !vllm_architecture_supports_images(manifest.architecture.as_deref()) {
            Err(anyhow!(
                "vLLM does not support image input for architecture '{}'",
                manifest.architecture.as_deref().unwrap_or("unknown")
            ))
        } else {
            match self.accelerator {
                VllmAccelerator::Cuda => Self::probe(&self.store),
                VllmAccelerator::Rocm => Self::probe_rocm(&self.store),
            }
            .map(|_| ())
        };
        let adapter = self.accelerator.backend_label().to_string();
        Some(match readiness {
            Ok(()) => TaskReadiness {
                status: TaskReadinessStatus::Available,
                detail: format!(
                    "{} can route image understanding for architecture '{}' without loading model weights during capability discovery",
                    self.accelerator.backend_label(),
                    manifest.architecture.as_deref().unwrap_or("unknown")
                ),
                adapter: Some(adapter),
                required_backend: None,
                install_command: None,
                fallback_backend: None,
                missing_dependencies: Vec::new(),
                missing_dependency_groups: Vec::new(),
            },
            Err(error) => TaskReadiness {
                status: TaskReadinessStatus::Unavailable,
                detail: error.to_string(),
                adapter: Some(adapter.clone()),
                required_backend: Some(adapter),
                install_command: None,
                fallback_backend: None,
                missing_dependencies: Vec::new(),
                missing_dependency_groups: Vec::new(),
            },
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

impl ChatGenerationSession for VllmChatSession {
    fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        validate_vllm_image_request(self.architecture.as_deref(), &request)?;
        let total_started = Instant::now();
        self.server.print_debug(&request, true);
        let completion = self.server.complete(&request, None)?;
        let assistant_message = completion.assistant_message();
        Ok(GenerateResponse {
            text: completion.text,
            assistant_message: Some(assistant_message),
            prompt_tokens: completion.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            finish_reason: completion.finish_reason,
            timings: GenerationTimings {
                load_seconds: 0.0,
                warmup_seconds: 0.0,
                first_token_seconds: completion.first_token_seconds,
                prompt_seconds: completion.prompt_seconds,
                decode_seconds: completion.decode_seconds,
                total_seconds: total_started.elapsed().as_secs_f64(),
            },
            backend_diagnostics: Vec::new(),
        })
    }

    fn generate_stream(&self, request: GenerateRequest) -> GenerateStream {
        let server = self.server.clone();
        let architecture = self.architecture.clone();
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            let total_started = Instant::now();
            server.print_debug(&request, true);
            let result = validate_vllm_image_request(architecture.as_deref(), &request)
                .and_then(|()| server.complete(&request, Some(tx.clone())))
                .map(|completion| {
                    let assistant_message = completion.assistant_message();
                    GenerateResponse {
                        text: completion.text,
                        assistant_message: Some(assistant_message),
                        prompt_tokens: completion.prompt_tokens,
                        completion_tokens: completion.completion_tokens,
                        finish_reason: completion.finish_reason,
                        timings: GenerationTimings {
                            load_seconds: 0.0,
                            warmup_seconds: 0.0,
                            first_token_seconds: completion.first_token_seconds,
                            prompt_seconds: completion.prompt_seconds,
                            decode_seconds: completion.decode_seconds,
                            total_seconds: total_started.elapsed().as_secs_f64(),
                        },
                        backend_diagnostics: Vec::new(),
                    }
                });
            send_stream_result(tx, result);
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

impl VllmProcess {
    fn start(
        store: &ModelStore,
        manifest: &ModelManifest,
        model_dir: &Path,
        discovery: VllmDiscovery,
        accelerator: VllmAccelerator,
        configured_args: ConfiguredVllmArgs,
    ) -> Result<Self> {
        let discovery = if discovery.command.is_some() {
            discovery
        } else {
            require_vllm(store)?
        };
        let command = discovery
            .command
            .clone()
            .context("vLLM discovery had no command")?;
        validate_vllm_args_target(&command, &configured_args)?;
        match accelerator {
            VllmAccelerator::Cuda => vllm_cuda_capability(&command)?,
            VllmAccelerator::Rocm => vllm_rocm_capability(&command)?,
        };
        let log_tail = Arc::new(Mutex::new(VecDeque::new()));
        if let VllmCommand::Remote { host, port } = command {
            eprintln!("Using remote vLLM {} backend", accelerator.display_name());
            let url = format!("http://{host}:{port}");
            let configured_model = configured_vllm_model()?;
            let timeout = configured_vllm_health_timeout(current_vllm_platform()).duration;
            let model = wait_for_remote_served_model(
                &url,
                &manifest.id,
                configured_model.as_deref(),
                timeout,
            )?;
            let process = Self {
                child: None,
                command_label: "remote vLLM OpenAI server".to_string(),
                discovery_source: discovery.source,
                args: Vec::new(),
                model_dir: model_dir.to_path_buf(),
                model_name: model.name,
                model_name_source: model.source,
                is_nemotron: is_nemotron_manifest(manifest),
                url,
                pid: None,
                log_tail,
                accelerator,
                runtime_version: "unreported".to_string(),
                runtime_instance_id: new_runtime_instance_id()?,
            };
            return Ok(process);
        }

        eprintln!("Using vLLM {} backend", accelerator.display_name());
        validate_werk_managed_prefix_caching_arg(&command, &configured_args)?;
        let runtime_version = vllm_version(&command)
            .and_then(|version| sanitize_runtime_version(&version))
            .unwrap_or_else(|| "unknown".to_string());
        let port = free_local_port()?;
        let url = format!("http://127.0.0.1:{port}");
        let launch = vllm_launch_command(
            &command,
            model_dir,
            &manifest.id,
            port,
            &configured_args.args,
        )?;
        let mut child_command = Command::new(&launch.program);
        child_command.args(&launch.args);
        if env_true("WERK_VLLM_LOG") {
            child_command
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit());
        } else {
            child_command.stdout(Stdio::piped()).stderr(Stdio::piped());
        }
        let mut child = child_command.spawn().with_context(|| {
            format!(
                "failed to start vLLM server using {}",
                launch.program.display()
            )
        })?;
        if !env_true("WERK_VLLM_LOG") {
            if let Some(stdout) = child.stdout.take() {
                spawn_log_tail_reader("stdout", stdout, log_tail.clone());
            }
            if let Some(stderr) = child.stderr.take() {
                spawn_log_tail_reader("stderr", stderr, log_tail.clone());
            }
        }
        let pid = child.id();
        let process = Self {
            child: Some(Mutex::new(child)),
            command_label: command.display(),
            discovery_source: discovery.source,
            args: launch.args,
            model_dir: model_dir.to_path_buf(),
            model_name: manifest.id.clone(),
            model_name_source: "Werk model ID / local --served-model-name",
            is_nemotron: is_nemotron_manifest(manifest),
            url,
            pid: Some(pid),
            log_tail,
            accelerator,
            runtime_version,
            runtime_instance_id: new_runtime_instance_id()?,
        };
        process.wait_until_ready()?;
        Ok(process)
    }

    fn complete(
        &self,
        request: &GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<VllmCompletion> {
        let started = Instant::now();
        let streaming = tx.is_some();
        let body = chat_completion_body(&self.model_name, request, streaming);
        let mut stream = post_json(&self.url, "/v1/chat/completions", &body)?;
        let mut completion = VllmCompletion {
            finish_reason: "length".to_string(),
            ..Default::default()
        };

        if streaming {
            let mut sse = SseAccumulator::default();
            stream_body(&mut stream, |bytes| {
                sse.push(bytes, |event| {
                    if event == "[DONE]" {
                        return Ok(());
                    }
                    let value: Value = serde_json::from_str(event)
                        .with_context(|| format!("invalid vLLM SSE event: {event}"))?;
                    update_completion_from_event(&mut completion, &value);
                    if let Some(chunk) = delta_content(&value) {
                        if completion.first_token_seconds <= 0.0 && !chunk.is_empty() {
                            completion.first_token_seconds = started.elapsed().as_secs_f64();
                        }
                        completion.text.push_str(&chunk);
                        append_assistant_content(&mut completion.assistant_content, &chunk);
                        if !chunk.is_empty() {
                            send_text_chunk(&tx, chunk)?;
                        }
                    }
                    if let Some(tool_calls) = delta_tool_calls(&value)? {
                        completion.saw_tool_call_delta = true;
                        if completion.first_token_seconds <= 0.0 {
                            completion.first_token_seconds = started.elapsed().as_secs_f64();
                        }
                        send_tool_call_delta(&tx, tool_calls)?;
                    }
                    Ok(())
                })
            })?;
        } else {
            let mut response_body = Vec::new();
            stream_body(&mut stream, |bytes| {
                response_body.extend_from_slice(bytes);
                Ok(())
            })?;
            let value: Value = serde_json::from_slice(&response_body)
                .context("vLLM returned an invalid non-streaming chat completion")?;
            update_completion_from_event(&mut completion, &value);
            update_completion_from_message(&mut completion, &value)?;
            if (!completion.text.is_empty()
                || completion
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty()))
                && completion.first_token_seconds <= 0.0
            {
                completion.first_token_seconds = started.elapsed().as_secs_f64();
            }
        }

        ensure_vllm_visible_completion(&completion)?;
        finalize_completion_stats(&mut completion, request, started.elapsed().as_secs_f64());
        Ok(completion)
    }

    fn wait_until_ready(&self) -> Result<()> {
        let timeout = configured_vllm_health_timeout(current_vllm_platform()).duration;
        let deadline = HttpDeadline::new(timeout);
        loop {
            if let Some(status) = self.try_wait_status()? {
                let reason = format!(
                    "vLLM server exited before becoming healthy ({status}){}",
                    self.formatted_log_tail()
                );
                if let Some(message) = wsl_vllm_health_failure_message(&reason) {
                    bail!("{message}");
                }
                bail!("{reason}");
            }
            let health_ready = deadline
                .remaining()
                .ok()
                .and_then(|remaining| {
                    get_with_timeout(
                        &self.url,
                        "/health",
                        remaining.min(HEALTH_REQUEST_IO_TIMEOUT),
                    )
                    .ok()
                })
                .is_some_and(|response| response.status == 200);
            let models_ready = !health_ready
                && deadline
                    .remaining()
                    .ok()
                    .and_then(|remaining| {
                        get_with_timeout(
                            &self.url,
                            "/v1/models",
                            remaining.min(HEALTH_REQUEST_IO_TIMEOUT),
                        )
                        .ok()
                    })
                    .is_some_and(|response| response.status == 200);
            if health_ready || models_ready {
                return Ok(());
            }
            let Ok(remaining) = deadline.remaining() else {
                let reason = format!(
                    "timed out after {}s waiting for vLLM server at {}{}",
                    timeout.as_secs(),
                    self.url,
                    self.formatted_log_tail()
                );
                if let Some(message) = wsl_vllm_health_failure_message(&reason) {
                    bail!("{message}");
                }
                bail!("{reason}");
            };
            thread::sleep(HEALTH_POLL_INTERVAL.min(remaining));
        }
    }

    fn is_running(&self) -> bool {
        if self.child.is_none() {
            return remote_vllm_model_ids(&self.url)
                .map(|models| remote_models_include_served_name(&self.model_name, &models))
                .unwrap_or(false);
        }
        matches!(self.try_wait_status(), Ok(None))
    }

    fn try_wait_status(&self) -> Result<Option<ExitStatus>> {
        let Some(child) = &self.child else {
            return Ok(None);
        };
        let mut child = child
            .lock()
            .map_err(|_| anyhow!("vLLM child mutex poisoned"))?;
        Ok(child.try_wait()?)
    }

    fn formatted_log_tail(&self) -> String {
        let Ok(tail) = self.log_tail.lock() else {
            return String::new();
        };
        if tail.is_empty() {
            return String::new();
        }
        format!(
            "\n\nvLLM output tail:\n{}",
            tail.iter().cloned().collect::<Vec<_>>().join("\n")
        )
    }

    fn print_debug(&self, request: &GenerateRequest, reused: bool) {
        if !request.debug {
            return;
        }
        eprintln!("selected backend: {}", self.accelerator.backend_label());
        eprintln!("actual engine: vLLM OpenAI-compatible server");
        eprintln!("vLLM executable: {}", self.command_label);
        eprintln!("discovery source: {}", self.discovery_source);
        if self.args.is_empty() {
            eprintln!("full vLLM args: <remote server>");
        } else {
            eprintln!("full vLLM args: {}", shell_join(&self.args));
        }
        eprintln!("model path: {}", self.model_dir.display());
        eprintln!(
            "server PID: {}",
            self.pid
                .map(|pid| pid.to_string())
                .unwrap_or_else(|| "external".to_string())
        );
        eprintln!("server URL: {}", self.url);
        eprintln!(
            "served model name: {} ({})",
            self.model_name, self.model_name_source
        );
        eprintln!("reused existing server: {reused}");
        if self.is_nemotron {
            print_nemotron_reasoning_parser_guidance(&self.args);
        }
    }
}

impl Drop for VllmProcess {
    fn drop(&mut self) {
        if let Some(child) = &self.child
            && let Ok(mut child) = child.lock()
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl VllmCommand {
    fn executable(&self) -> PathBuf {
        match self {
            Self::Python(path) | Self::Executable(path) => path.clone(),
            Self::Remote { .. } => PathBuf::new(),
        }
    }

    fn display(&self) -> String {
        match self {
            Self::Python(path) => {
                format!("{} -m vllm.entrypoints.openai.api_server", path.display())
            }
            Self::Executable(path) => path.display().to_string(),
            Self::Remote { host, port } => format!("http://{host}:{port}"),
        }
    }
}

struct HttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    reader: BufReader<TcpStream>,
    deadline: Option<HttpDeadline>,
}

#[derive(Debug, Clone, Copy)]
struct HttpDeadline {
    started: Instant,
    timeout: Duration,
}

impl HttpDeadline {
    fn new(timeout: Duration) -> Self {
        Self {
            started: Instant::now(),
            timeout,
        }
    }

    fn remaining(self) -> Result<Duration> {
        self.timeout
            .checked_sub(self.started.elapsed())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| anyhow!("vLLM HTTP probe timed out"))
    }

    fn remaining_capped(self, cap: Duration) -> Result<Duration> {
        Ok(self.remaining()?.min(cap))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VllmRemoteConfig {
    host: String,
    port: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RemoteModelResolution {
    name: String,
    source: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VllmHealthTimeout {
    duration: Duration,
    valid: bool,
    detail: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct VllmCacheEnvironment {
    args: String,
    served_model: String,
    host: String,
    port: String,
    python: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ConfiguredVllmArgs {
    raw: String,
    args: Vec<String>,
    werk_managed_prefix_caching: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VllmLaunchCommand {
    program: PathBuf,
    args: Vec<String>,
}

impl VllmCacheEnvironment {
    fn current(configured_args: &ConfiguredVllmArgs) -> Self {
        Self {
            // Cache against the argv Werk will actually pass to the process,
            // including server defaults which did not originate in the
            // environment variable.
            args: serde_json::to_string(&configured_args.args)
                .expect("serializing a string-only vLLM argv cannot fail"),
            served_model: env::var("WERK_VLLM_MODEL").unwrap_or_default(),
            host: env::var("WERK_VLLM_HOST").unwrap_or_default(),
            port: env::var("WERK_VLLM_PORT").unwrap_or_default(),
            python: env::var("WERK_VLLM_PYTHON").unwrap_or_default(),
        }
    }
}

#[derive(Default)]
struct SseAccumulator {
    pending: Vec<u8>,
}

impl SseAccumulator {
    fn push<F>(&mut self, bytes: &[u8], mut on_event: F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        self.pending.extend_from_slice(bytes);
        while let Some(index) = find_sse_boundary(&self.pending) {
            let event = self.pending.drain(..index).collect::<Vec<_>>();
            while matches!(self.pending.first(), Some(b'\r' | b'\n')) {
                self.pending.remove(0);
            }
            let event = String::from_utf8_lossy(&event);
            for line in event.lines() {
                if let Some(data) = line.strip_prefix("data:") {
                    on_event(data.trim())?;
                }
            }
        }
        Ok(())
    }
}

fn vllm_launch_command(
    command: &VllmCommand,
    model_dir: &Path,
    model_name: &str,
    port: u16,
    extra_args: &[String],
) -> Result<VllmLaunchCommand> {
    validate_vllm_extra_args(extra_args)?;

    let mut args = match command {
        VllmCommand::Python(_) => vec![
            "-m".to_string(),
            "vllm.entrypoints.openai.api_server".to_string(),
            "--model".to_string(),
            model_dir.display().to_string(),
        ],
        VllmCommand::Executable(_) => vec!["serve".to_string(), model_dir.display().to_string()],
        VllmCommand::Remote { .. } => {
            bail!("cannot construct a local vLLM launch command for a remote vLLM server")
        }
    };
    args.extend([
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "--served-model-name".to_string(),
        model_name.to_string(),
    ]);
    args.extend_from_slice(extra_args);
    Ok(VllmLaunchCommand {
        program: command.executable(),
        args,
    })
}

fn configured_vllm_model() -> Result<Option<String>> {
    env::var("WERK_VLLM_MODEL")
        .ok()
        .map(|value| validate_configured_vllm_model(&value))
        .transpose()
}

fn validate_configured_vllm_model(value: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() {
        bail!("WERK_VLLM_MODEL must not be empty");
    }
    if value.chars().any(char::is_control) {
        bail!("WERK_VLLM_MODEL must not contain control characters");
    }
    Ok(value.to_string())
}

fn resolve_remote_served_model_with_timeout(
    base_url: &str,
    werk_model_id: &str,
    configured_model: Option<&str>,
    timeout: Duration,
) -> Result<RemoteModelResolution> {
    let advertised = remote_vllm_model_ids_with_timeout(base_url, timeout).with_context(|| {
        format!(
            "could not resolve the served model name from {base_url}/v1/models; set WERK_VLLM_MODEL to the model ID exposed by the remote vLLM server"
        )
    })?;
    if let Some(configured_model) = configured_model {
        return select_configured_remote_served_model(configured_model, &advertised);
    }
    select_remote_served_model(werk_model_id, &advertised)
}

fn wait_for_remote_served_model(
    base_url: &str,
    werk_model_id: &str,
    configured_model: Option<&str>,
    timeout: Duration,
) -> Result<RemoteModelResolution> {
    wait_for_remote_served_model_with(timeout, HEALTH_POLL_INTERVAL, |remaining| {
        resolve_remote_served_model_with_timeout(
            base_url,
            werk_model_id,
            configured_model,
            remaining.min(HEALTH_REQUEST_IO_TIMEOUT),
        )
    })
    .with_context(|| format!("remote vLLM model discovery at {base_url}/v1/models failed"))
}

fn wait_for_remote_served_model_with<F>(
    timeout: Duration,
    poll_interval: Duration,
    mut resolve: F,
) -> Result<RemoteModelResolution>
where
    F: FnMut(Duration) -> Result<RemoteModelResolution>,
{
    let deadline = HttpDeadline::new(timeout);
    let mut last_message = None;
    loop {
        let remaining = match deadline.remaining() {
            Ok(remaining) => remaining,
            Err(_) => {
                bail!(
                    "timed out after {}s waiting for remote vLLM model discovery: {}",
                    timeout.as_secs(),
                    last_message.unwrap_or_else(|| "endpoint did not respond".to_string()),
                )
            }
        };
        let error = match resolve(remaining) {
            Ok(model) => return Ok(model),
            Err(error) => error,
        };
        let message = compact_error(&error.to_string());
        let configuration_error = message.contains("serves multiple models")
            || message.contains("is not advertised by remote vLLM");
        if configuration_error {
            return Err(error);
        }
        last_message = Some(message);
        let remaining = deadline.remaining().unwrap_or(Duration::ZERO);
        if remaining.is_zero() {
            continue;
        }
        thread::sleep(poll_interval.min(remaining));
    }
}

fn select_configured_remote_served_model(
    configured_model: &str,
    advertised: &[String],
) -> Result<RemoteModelResolution> {
    if advertised.iter().any(|model| model == configured_model) {
        return Ok(RemoteModelResolution {
            name: configured_model.to_string(),
            source: "verified WERK_VLLM_MODEL",
        });
    }
    bail!(
        "WERK_VLLM_MODEL '{}' is not advertised by remote vLLM; available models: {}",
        configured_model,
        if advertised.is_empty() {
            "<none>".to_string()
        } else {
            advertised.join(", ")
        }
    )
}

fn remote_models_include_served_name(served_name: &str, advertised: &[String]) -> bool {
    advertised.iter().any(|model| model == served_name)
}

fn select_remote_served_model(
    werk_model_id: &str,
    advertised: &[String],
) -> Result<RemoteModelResolution> {
    if advertised.iter().any(|model| model == werk_model_id) {
        return Ok(RemoteModelResolution {
            name: werk_model_id.to_string(),
            source: "matching /v1/models entry",
        });
    }
    if let [only] = advertised {
        return Ok(RemoteModelResolution {
            name: only.clone(),
            source: "only /v1/models entry",
        });
    }
    if advertised.is_empty() {
        bail!(
            "remote vLLM returned no served models; set WERK_VLLM_MODEL after the server has loaded its model"
        );
    }
    bail!(
        "remote vLLM serves multiple models ({}) and none matches Werk model '{}'; set WERK_VLLM_MODEL to one advertised model ID",
        advertised.join(", "),
        werk_model_id
    )
}

fn remote_vllm_model_ids(base_url: &str) -> Result<Vec<String>> {
    remote_vllm_model_ids_with_timeout(base_url, HEALTH_REQUEST_IO_TIMEOUT)
}

fn remote_vllm_model_ids_with_timeout(base_url: &str, timeout: Duration) -> Result<Vec<String>> {
    let mut response = get_with_timeout(base_url, "/v1/models", timeout)?;
    let mut bytes = Vec::new();
    stream_body(&mut response, |chunk| {
        bytes.extend_from_slice(chunk);
        Ok(())
    })?;
    parse_remote_vllm_model_ids(&bytes)
}

fn parse_remote_vllm_model_ids(bytes: &[u8]) -> Result<Vec<String>> {
    let value: Value =
        serde_json::from_slice(bytes).context("remote vLLM /v1/models returned invalid JSON")?;
    let mut models = value
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|model| model.get("id").and_then(Value::as_str))
        .map(str::trim)
        .filter(|id| !id.is_empty() && !id.chars().any(char::is_control))
        .map(str::to_string)
        .collect::<Vec<_>>();
    models.sort();
    models.dedup();
    Ok(models)
}

fn is_nemotron_manifest(manifest: &ModelManifest) -> bool {
    manifest
        .architecture
        .as_deref()
        .is_some_and(is_nemotron_architecture_name)
}

fn is_nemotron_architecture_name(architecture: &str) -> bool {
    matches!(
        architecture.to_ascii_lowercase().as_str(),
        "nemotron_h" | "nemotron_h_moe"
    )
}

fn print_nemotron_reasoning_parser_guidance(args: &[String]) {
    if args.is_empty() {
        eprintln!(
            "Nemotron reasoning parser: controlled by the remote vLLM server; configure the parser when starting its Spark/container runtime if the model card requires one"
        );
    } else if has_reasoning_parser_arg(args) {
        eprintln!("Nemotron reasoning parser: configured through WERK_VLLM_ARGS");
    } else {
        eprintln!(
            "Nemotron reasoning parser: not configured; add `--reasoning-parser <parser>` to WERK_VLLM_ARGS only when required by the concrete model card"
        );
    }
}

fn has_reasoning_parser_arg(args: &[String]) -> bool {
    args.iter()
        .any(|arg| arg == "--reasoning-parser" || arg.starts_with("--reasoning-parser="))
}

fn resolve_vllm_model_dir(store: &ModelStore, manifest: &ModelManifest) -> Result<PathBuf> {
    let root = store.model_dir(&manifest.id);
    if !root.is_dir() {
        bail!(
            "model directory for '{}' does not exist: {}",
            manifest.id,
            root.display()
        );
    }

    if let Some(config_path) = manifest.config_path.as_deref() {
        let config = store.absolute_model_file(manifest, config_path);
        let dir = config.parent().with_context(|| {
            format!(
                "manifest config_path '{}' has no parent directory",
                config_path
            )
        })?;
        if dir.join("config.json").is_file() {
            return Ok(dir.to_path_buf());
        }
    }

    let files_dir = root.join("files");
    if files_dir.join("config.json").is_file() {
        return Ok(files_dir);
    }

    if root.join("config.json").is_file() {
        return Ok(root);
    }

    bail!(
        "vLLM requires a Hugging Face model directory containing config.json for '{}'; tried manifest config_path {}, files directory {}, and model root {}",
        manifest.id,
        manifest
            .config_path
            .as_deref()
            .map(|path| store
                .absolute_model_file(manifest, path)
                .display()
                .to_string())
            .unwrap_or_else(|| "<none>".to_string()),
        files_dir.display(),
        root.display()
    )
}

fn resolve_vllm_model_dir_for_discovery(
    store: &ModelStore,
    manifest: &ModelManifest,
    discovery: &VllmDiscovery,
) -> Result<PathBuf> {
    if vllm_uses_remote_weights(discovery) {
        Ok(store.model_dir(&manifest.id))
    } else {
        resolve_vllm_model_dir(store, manifest)
    }
}

fn vllm_uses_remote_weights(discovery: &VllmDiscovery) -> bool {
    matches!(discovery.command, Some(VllmCommand::Remote { .. }))
}

fn vllm_discovery_cache_identity(discovery: &VllmDiscovery) -> String {
    match discovery.command.as_ref() {
        Some(VllmCommand::Python(path)) => format!("python:{}", path.display()),
        Some(VllmCommand::Executable(path)) => format!("executable:{}", path.display()),
        Some(VllmCommand::Remote { host, port }) => format!("remote:{host}:{port}"),
        None => format!("missing:{}", discovery.source),
    }
}

fn vllm_server_cache_key(
    model_identity: &ModelRuntimeIdentity,
    model_dir: &Path,
    discovery: &VllmDiscovery,
    environment: &VllmCacheEnvironment,
) -> String {
    format!(
        "{}:{}:{}:{}:{}:{}:{}:{}",
        model_identity,
        model_dir.display(),
        environment.args,
        environment.served_model,
        environment.host,
        environment.port,
        environment.python,
        vllm_discovery_cache_identity(discovery),
    )
}

fn chat_completion_body(model_name: &str, request: &GenerateRequest, stream: bool) -> Value {
    let messages = if request.messages.is_empty() {
        json!([{
            "role": "user",
            "content": request.prompt,
        }])
    } else {
        vllm_chat_messages(&request.messages)
    };
    let mut body = json!({
        "model": model_name,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "stream": stream,
    });
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = json!(top_p);
    }
    if !request.stop.is_empty() {
        body["stop"] = json!(request.stop);
    }
    if let Some(seed) = request.seed {
        body["seed"] = json!(seed);
    }
    if let Some(tool_config) = &request.tool_config {
        if let Some(tools) = &tool_config.tools {
            body["tools"] = json!(tools);
        }
        if let Some(tool_choice) = &tool_config.tool_choice {
            body["tool_choice"] = json!(tool_choice);
        }
        if let Some(parallel_tool_calls) = tool_config.parallel_tool_calls {
            body["parallel_tool_calls"] = json!(parallel_tool_calls);
        }
    }
    body
}

fn vllm_chat_messages(messages: &[crate::openai::ChatMessage]) -> Value {
    let mut messages =
        serde_json::to_value(messages).expect("ChatMessage serialization cannot fail");
    if let Some(messages) = messages.as_array_mut() {
        for message in messages {
            let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts {
                if part.get("type").and_then(Value::as_str) == Some("input_image") {
                    part["type"] = json!("image_url");
                }
                if matches!(part.get("type").and_then(Value::as_str), Some("image_url"))
                    && let Some(url) = part.get("image_url").and_then(Value::as_str)
                {
                    part["image_url"] = json!({"url": url});
                }
            }
        }
    }
    messages
}

fn update_completion_from_event(completion: &mut VllmCompletion, value: &Value) {
    if delta_has_reasoning_content(value) {
        completion.saw_reasoning_content = true;
    }
    if let Some(choice) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        && let Some(reason) = choice.get("finish_reason").and_then(Value::as_str)
        && !reason.is_empty()
    {
        completion.finish_reason = reason.to_string();
    }
    if let Some(usage) = value.get("usage") {
        if let Some(tokens) = usage.get("prompt_tokens").and_then(Value::as_u64) {
            completion.prompt_tokens = tokens as usize;
        }
        if let Some(tokens) = usage.get("completion_tokens").and_then(Value::as_u64) {
            completion.completion_tokens = tokens as usize;
        }
    }
}

fn update_completion_from_message(completion: &mut VllmCompletion, value: &Value) -> Result<()> {
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .context("vLLM non-streaming response has no completion choice")?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .context("vLLM non-streaming response has no assistant message")?;

    completion.assistant_content = match message.get("content") {
        Some(Value::Null) => Some(None),
        Some(Value::String(content)) => {
            completion.text.clone_from(content);
            Some(Some(content.clone()))
        }
        Some(_) => bail!("vLLM assistant message content must be a string or null"),
        None => None,
    };
    completion.tool_calls = match message.get("tool_calls") {
        Some(Value::Null) | None => None,
        Some(tool_calls) => Some(
            serde_json::from_value(tool_calls.clone())
                .context("vLLM returned invalid assistant message.tool_calls")?,
        ),
    };
    if completion.assistant_content.is_none()
        && completion
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty())
    {
        completion.assistant_content = Some(None);
    }
    Ok(())
}

fn append_assistant_content(content: &mut Option<Option<String>>, chunk: &str) {
    match content {
        Some(Some(current)) => current.push_str(chunk),
        _ => *content = Some(Some(chunk.to_string())),
    }
}

fn delta_tool_calls(value: &Value) -> Result<Option<Vec<ChatCompletionToolCallDelta>>> {
    let Some(tool_calls) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("tool_calls"))
    else {
        return Ok(None);
    };
    if tool_calls.is_null() {
        return Ok(None);
    }
    serde_json::from_value(tool_calls.clone())
        .map(Some)
        .context("vLLM returned invalid delta.tool_calls")
}

fn delta_has_reasoning_content(value: &Value) -> bool {
    let Some(delta) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"))
    else {
        return false;
    };
    ["reasoning", "reasoning_content"].into_iter().any(|key| {
        delta.get(key).is_some_and(|value| match value {
            Value::String(text) => !text.is_empty(),
            Value::Null => false,
            _ => true,
        })
    })
}

fn ensure_vllm_visible_completion(completion: &VllmCompletion) -> Result<()> {
    let has_tool_calls = completion.saw_tool_call_delta
        || completion
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty());
    if completion.text.trim().is_empty() && completion.saw_reasoning_content && !has_tool_calls {
        bail!(
            "vLLM generated hidden reasoning but no visible answer content; increase max tokens so the model can finish its answer, or disable the configured reasoning parser/mode when hidden reasoning is not wanted"
        );
    }
    Ok(())
}

fn delta_content(value: &Value) -> Option<String> {
    value
        .get("choices")?
        .as_array()?
        .first()?
        .get("delta")?
        .get("content")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn finalize_completion_stats(
    completion: &mut VllmCompletion,
    request: &GenerateRequest,
    elapsed_seconds: f64,
) {
    if completion.prompt_tokens == 0 && !request.prompt.trim().is_empty() {
        completion.prompt_tokens = estimate_tokens(&request.prompt);
    }
    if completion.prompt_seconds <= 0.0 && completion.first_token_seconds > 0.0 {
        completion.prompt_seconds = completion.first_token_seconds;
    }
    if completion.decode_seconds <= 0.0 {
        completion.decode_seconds = if completion.first_token_seconds > 0.0
            && elapsed_seconds > completion.first_token_seconds
        {
            elapsed_seconds - completion.first_token_seconds
        } else {
            elapsed_seconds
        };
    }
    if completion.completion_tokens == 0 {
        completion.completion_tokens = estimate_tokens(&completion.text);
    }
}

pub fn install_managed_vllm(store: &ModelStore) -> Result<PathBuf> {
    let platform = current_vllm_platform();
    if let Some(reason) = managed_vllm_install_rejection_for(platform, env::consts::ARCH) {
        bail!("{reason}");
    }
    if platform == VllmPlatform::Wsl {
        eprintln!("{WSL_VLLM_MESSAGE}");
    }

    let root = managed_vllm_dir(store);
    let venv = root.join("venv");
    fs::create_dir_all(&root)
        .with_context(|| format!("failed to create vLLM backend cache {}", root.display()))?;

    let python = find_bootstrap_python().ok_or_else(|| {
        anyhow!("no Python interpreter found; install python3 or set WERK_VLLM_PYTHON")
    })?;
    if !managed_vllm_python(store).is_file() {
        eprintln!("Creating vLLM virtualenv at {}", venv.display());
        run_command(
            Command::new(&python).arg("-m").arg("venv").arg(&venv),
            "failed to create vLLM virtualenv",
        )?;
    }
    let venv_python = managed_vllm_python(store);
    eprintln!("Installing vLLM into {}", venv.display());
    run_command(
        Command::new(&venv_python)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--upgrade")
            .arg("pip"),
        "failed to upgrade pip in vLLM virtualenv",
    )?;
    run_command(
        Command::new(&venv_python)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("vllm"),
        "failed to install vLLM; check network access, Python version, CUDA/PyTorch wheel availability",
    )?;
    validate_vllm_python(&venv_python)?;
    Ok(venv_python)
}

pub fn vllm_doctor_checks(store: &ModelStore) -> Vec<BackendDoctorCheck> {
    let discovery = discover_vllm(store);
    let health = vllm_health(&discovery);
    let platform = current_vllm_platform();
    let health_timeout = configured_vllm_health_timeout(platform);
    let mut checks = Vec::new();
    checks.push(BackendDoctorCheck {
        name: "vLLM platform".to_string(),
        ok: matches!(discovery.command, Some(VllmCommand::Remote { .. }))
            || local_vllm_platform_rejection(platform).is_none(),
        detail: vllm_platform_detail(platform).to_string(),
    });
    checks.push(BackendDoctorCheck {
        name: "vLLM health timeout".to_string(),
        ok: health_timeout.valid,
        detail: health_timeout.detail,
    });
    checks.push(BackendDoctorCheck {
        name: "vLLM discovery".to_string(),
        ok: discovery.command.is_some(),
        detail: discovery.source.clone(),
    });
    checks.push(BackendDoctorCheck {
        name: "vLLM runtime source".to_string(),
        ok: discovery.command.is_some(),
        detail: match &discovery.command {
            Some(VllmCommand::Python(path)) => path.display().to_string(),
            Some(VllmCommand::Executable(path)) => format!("using executable {}", path.display()),
            Some(VllmCommand::Remote { host, port }) => {
                format!("using remote http://{host}:{port}")
            }
            None => format!("managed path {}", managed_vllm_python(store).display()),
        },
    });
    checks.push(BackendDoctorCheck {
        name: "vLLM installed".to_string(),
        ok: discovery.command.is_some(),
        detail: discovery
            .command
            .as_ref()
            .and_then(vllm_version)
            .unwrap_or_else(|| "not installed".to_string()),
    });
    checks.push(BackendDoctorCheck {
        name: "vLLM health".to_string(),
        ok: health.healthy,
        detail: format!("{}: {}", health.health_label, health.detail),
    });
    let (accelerator_ok, accelerator_detail) = discovery
        .command
        .as_ref()
        .map(vllm_accelerator_capability_detail)
        .unwrap_or_else(|| (false, "no vLLM runtime discovered".to_string()));
    checks.push(BackendDoctorCheck {
        name: "vLLM accelerator capability".to_string(),
        ok: accelerator_ok,
        detail: accelerator_detail,
    });
    checks.push(BackendDoctorCheck {
        name: "vLLM version".to_string(),
        ok: discovery.command.is_some(),
        detail: discovery
            .command
            .as_ref()
            .and_then(vllm_version)
            .unwrap_or_else(|| "unknown".to_string()),
    });
    checks
}

fn vllm_accelerator_capability_detail(command: &VllmCommand) -> (bool, String) {
    let cuda = vllm_cuda_capability(command);
    let rocm = vllm_rocm_capability(command);
    let ok = cuda.is_ok() || rocm.is_ok();
    let detail = format!(
        "CUDA: {}; ROCm: {}",
        cuda.unwrap_or_else(|error| compact_error(&error.to_string())),
        rocm.unwrap_or_else(|error| compact_error(&error.to_string()))
    );
    (ok, detail)
}

fn vllm_health(discovery: &VllmDiscovery) -> VllmHealthStatus {
    vllm_health_for_platform(discovery, current_vllm_platform())
}

fn vllm_health_for_platform(discovery: &VllmDiscovery, platform: VllmPlatform) -> VllmHealthStatus {
    match discovery.command.as_ref() {
        Some(VllmCommand::Remote { host, port }) => {
            let ready = remote_discovery_ready(discovery);
            VllmHealthStatus {
                installed_label: "remote",
                health_label: if ready { "healthy" } else { "not ready" },
                healthy: ready,
                detail: if ready {
                    format!(
                        "remote OpenAI-compatible vLLM endpoint reachable at http://{host}:{port}"
                    )
                } else {
                    format!(
                        "remote vLLM is configured at http://{host}:{port} but is not ready; inference will wait up to the configured health timeout"
                    )
                },
            }
        }
        Some(command) => match local_vllm_platform_rejection(platform) {
            Some(reason) => VllmHealthStatus {
                installed_label: "yes",
                health_label: if platform == VllmPlatform::Wsl {
                    "best-effort on WSL"
                } else {
                    "unsupported"
                },
                healthy: false,
                detail: reason.to_string(),
            },
            None => VllmHealthStatus {
                installed_label: "yes",
                health_label: "eligible",
                healthy: true,
                detail: vllm_version(command).unwrap_or_else(|| "version unknown".to_string()),
            },
        },
        None => {
            match local_vllm_platform_rejection_for_discovery_with_platform(discovery, platform) {
                Some(reason) => VllmHealthStatus {
                    installed_label: "no",
                    health_label: if platform == VllmPlatform::Wsl {
                        "best-effort on WSL"
                    } else {
                        "unsupported"
                    },
                    healthy: false,
                    detail: reason.to_string(),
                },
                None => VllmHealthStatus {
                    installed_label: "no",
                    health_label: "missing",
                    healthy: false,
                    detail: concise_vllm_unavailable_reason_for_platform(discovery, platform),
                },
            }
        }
    }
}

fn remote_discovery_ready(discovery: &VllmDiscovery) -> bool {
    discovery
        .attempts
        .iter()
        .find(|attempt| attempt.label == "WERK_VLLM_HOST/WERK_VLLM_PORT")
        .map(|attempt| attempt.usable)
        .unwrap_or(true)
}

fn ensure_vllm_platform_eligible(command: &VllmCommand) -> Result<()> {
    if matches!(command, VllmCommand::Remote { .. }) {
        return Ok(());
    }
    if let Some(reason) = local_vllm_platform_rejection(current_vllm_platform()) {
        bail!("{reason}");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VllmPlatform {
    NativeLinux,
    DgxSpark,
    StrixHalo,
    Wsl,
    NativeWindows,
    Macos,
    Unsupported,
}

fn current_vllm_platform() -> VllmPlatform {
    if cfg!(target_os = "linux") {
        if is_wsl_environment() {
            VllmPlatform::Wsl
        } else if is_dgx_spark_environment() {
            VllmPlatform::DgxSpark
        } else if strix_halo_vllm_profile_selected(
            current_host_is_strix_halo(),
            current_selected_rocm_device_status(),
        ) {
            VllmPlatform::StrixHalo
        } else {
            VllmPlatform::NativeLinux
        }
    } else if cfg!(target_os = "windows") {
        VllmPlatform::NativeWindows
    } else if cfg!(target_os = "macos") {
        VllmPlatform::Macos
    } else {
        VllmPlatform::Unsupported
    }
}

fn strix_halo_vllm_profile_selected(
    strix_halo_host: bool,
    selected_device: SelectedRocmDeviceStatus,
) -> bool {
    strix_halo_host && selected_device != SelectedRocmDeviceStatus::Other
}

fn local_vllm_platform_rejection(platform: VllmPlatform) -> Option<&'static str> {
    match platform {
        VllmPlatform::NativeLinux | VllmPlatform::DgxSpark | VllmPlatform::StrixHalo => None,
        VllmPlatform::Wsl => Some(WSL_VLLM_MESSAGE),
        VllmPlatform::NativeWindows => Some(
            "vLLM is a Linux-native runtime. Native Windows local vLLM is not eligible. Use native Linux or a remote vLLM server.",
        ),
        VllmPlatform::Macos => Some(
            "vLLM is a Linux-native runtime. Local managed vLLM is not eligible on macOS. Use native Linux or a remote vLLM server.",
        ),
        VllmPlatform::Unsupported => Some(
            "vLLM is a Linux-native runtime. Local vLLM is not eligible on this platform. Use native Linux or a remote vLLM server.",
        ),
    }
}

fn managed_vllm_install_rejection_for(
    platform: VllmPlatform,
    architecture: &str,
) -> Option<&'static str> {
    match platform {
        VllmPlatform::DgxSpark => Some(DGX_SPARK_VLLM_MESSAGE),
        VllmPlatform::StrixHalo => Some(STRIX_HALO_VLLM_MESSAGE),
        VllmPlatform::NativeLinux | VllmPlatform::Wsl if architecture == "aarch64" => {
            Some(LINUX_ARM64_MANAGED_VLLM_MESSAGE)
        }
        VllmPlatform::NativeLinux | VllmPlatform::Wsl => None,
        VllmPlatform::NativeWindows | VllmPlatform::Macos | VllmPlatform::Unsupported => {
            local_vllm_platform_rejection(platform)
        }
    }
}

fn local_vllm_platform_rejection_for_discovery(discovery: &VllmDiscovery) -> Option<&'static str> {
    local_vllm_platform_rejection_for_discovery_with_platform(discovery, current_vllm_platform())
}

fn local_vllm_platform_rejection_for_discovery_with_platform(
    discovery: &VllmDiscovery,
    platform: VllmPlatform,
) -> Option<&'static str> {
    if matches!(discovery.command, Some(VllmCommand::Remote { .. }))
        || invalid_remote_vllm_config_detail(discovery).is_some()
    {
        return None;
    }
    local_vllm_platform_rejection(platform)
}

fn is_wsl_environment() -> bool {
    env::var_os("WSL_DISTRO_NAME").is_some()
        || env::var_os("WSL_INTEROP").is_some()
        || fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|text| linux_release_looks_like_wsl(&text))
            .unwrap_or(false)
        || fs::read_to_string("/proc/version")
            .map(|text| linux_release_looks_like_wsl(&text))
            .unwrap_or(false)
}

fn is_dgx_spark_environment() -> bool {
    if !cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        return false;
    }
    let device_tree_model = fs::read_to_string("/proc/device-tree/model").ok();
    let nvidia_smi = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned());
    dgx_spark_signals(
        env::consts::ARCH,
        device_tree_model.as_deref(),
        nvidia_smi.as_deref(),
    )
}

fn dgx_spark_signals(
    architecture: &str,
    device_tree_model: Option<&str>,
    nvidia_smi: Option<&str>,
) -> bool {
    if architecture != "aarch64" {
        return false;
    }
    [device_tree_model, nvidia_smi]
        .into_iter()
        .flatten()
        .map(str::to_ascii_lowercase)
        .any(|signal| {
            signal.contains("dgx spark")
                || signal.contains("nvidia spark")
                || contains_ascii_alphanumeric_token(&signal, "gb10")
        })
}

fn contains_ascii_alphanumeric_token(text: &str, expected: &str) -> bool {
    text.split(|character: char| !character.is_ascii_alphanumeric())
        .any(|token| token.eq_ignore_ascii_case(expected))
}

fn vllm_platform_detail(platform: VllmPlatform) -> &'static str {
    match platform {
        VllmPlatform::NativeLinux => "native Linux; local or remote vLLM is eligible",
        VllmPlatform::DgxSpark => DGX_SPARK_VLLM_MESSAGE,
        VllmPlatform::StrixHalo => STRIX_HALO_VLLM_MESSAGE,
        VllmPlatform::Wsl => WSL_VLLM_MESSAGE,
        VllmPlatform::NativeWindows => "native Windows; use a remote Linux vLLM OpenAI endpoint",
        VllmPlatform::Macos => "macOS; use a remote Linux vLLM OpenAI endpoint",
        VllmPlatform::Unsupported => {
            "unsupported local vLLM platform; use a remote Linux vLLM OpenAI endpoint"
        }
    }
}

fn configured_vllm_health_timeout(platform: VllmPlatform) -> VllmHealthTimeout {
    let raw = env::var("WERK_VLLM_HEALTH_TIMEOUT_SECONDS").ok();
    vllm_health_timeout_for(platform, raw.as_deref())
}

fn vllm_health_timeout_for(platform: VllmPlatform, raw: Option<&str>) -> VllmHealthTimeout {
    let default_seconds = match platform {
        VllmPlatform::DgxSpark => DGX_SPARK_HEALTH_TIMEOUT_SECONDS,
        VllmPlatform::StrixHalo => STRIX_HALO_HEALTH_TIMEOUT_SECONDS,
        _ => DEFAULT_HEALTH_TIMEOUT_SECONDS,
    };
    match raw {
        None => VllmHealthTimeout {
            duration: Duration::from_secs(default_seconds),
            valid: true,
            detail: format!(
                "{default_seconds}s default{}; override with WERK_VLLM_HEALTH_TIMEOUT_SECONDS",
                match platform {
                    VllmPlatform::DgxSpark => " for DGX Spark cold starts",
                    VllmPlatform::StrixHalo => " for Strix Halo ROCm cold starts",
                    _ => "",
                }
            ),
        },
        Some(raw) => match raw.trim().parse::<u64>() {
            Ok(seconds) if seconds > 0 => VllmHealthTimeout {
                duration: Duration::from_secs(seconds),
                valid: true,
                detail: format!("{seconds}s from WERK_VLLM_HEALTH_TIMEOUT_SECONDS"),
            },
            _ => VllmHealthTimeout {
                duration: Duration::from_secs(default_seconds),
                valid: false,
                detail: format!(
                    "invalid WERK_VLLM_HEALTH_TIMEOUT_SECONDS={raw:?}; expected a positive integer, using {default_seconds}s"
                ),
            },
        },
    }
}

fn linux_release_looks_like_wsl(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("microsoft") || lower.contains("wsl")
}

fn wsl_vllm_health_failure_message(reason: &str) -> Option<String> {
    if current_vllm_platform() != VllmPlatform::Wsl || !is_wsl_sensitive_vllm_failure(reason) {
        return None;
    }
    Some(format!(
        "{WSL_VLLM_MESSAGE}\n\nvLLM health probe failed: {}",
        compact_error(reason)
    ))
}

fn is_wsl_sensitive_vllm_failure(reason: &str) -> bool {
    let lower = reason.to_ascii_lowercase();
    lower.contains("uva")
        || lower.contains("pin_memory")
        || lower.contains("pin memory")
        || lower.contains("engine-core")
        || lower.contains("engine_core")
        || lower.contains("engine core")
        || lower.contains("cuda ipc")
        || lower.contains("cuda_ipc")
        || lower.contains("cudaipc")
        || (lower.contains("multiprocessing")
            && (lower.contains("start") || lower.contains("spawn") || lower.contains("failed")))
}

pub fn managed_vllm_dir(store: &ModelStore) -> PathBuf {
    store.home().join("backends").join("vllm")
}

fn managed_vllm_python(store: &ModelStore) -> PathBuf {
    if cfg!(windows) {
        managed_vllm_dir(store)
            .join("venv")
            .join("Scripts")
            .join("python.exe")
    } else {
        managed_vllm_dir(store)
            .join("venv")
            .join("bin")
            .join("python")
    }
}

fn require_vllm(store: &ModelStore) -> Result<VllmDiscovery> {
    let discovery = discover_vllm(store);
    if discovery.command.is_some() {
        Ok(discovery)
    } else {
        bail!("{}", missing_vllm_message(&discovery))
    }
}

fn configured_remote_vllm_from_env() -> Result<Option<VllmRemoteConfig>> {
    let host = env::var_os("WERK_VLLM_HOST")
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("WERK_VLLM_HOST must contain valid UTF-8"))
        })
        .transpose()?;
    let port = env::var_os("WERK_VLLM_PORT")
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("WERK_VLLM_PORT must contain valid UTF-8"))
        })
        .transpose()?;
    configured_remote_vllm(host.as_deref(), port.as_deref())
}

fn configured_remote_vllm(
    host: Option<&str>,
    port: Option<&str>,
) -> Result<Option<VllmRemoteConfig>> {
    let (Some(host), Some(port)) = (host, port) else {
        if host.is_some() || port.is_some() {
            bail!("WERK_VLLM_HOST and WERK_VLLM_PORT must be set together");
        }
        return Ok(None);
    };

    let host = host.trim();
    if host.is_empty() {
        bail!("WERK_VLLM_HOST must not be empty");
    }
    if host.chars().any(char::is_control) {
        bail!("WERK_VLLM_HOST must not contain control characters");
    }
    if host.contains(':') {
        bail!(
            "WERK_VLLM_HOST does not currently accept IPv6 literals; use an IPv4 address or DNS name"
        );
    }
    if !host
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || ".-_".contains(character))
    {
        bail!(
            "WERK_VLLM_HOST contains unsupported characters; use an IPv4 address or ASCII DNS name"
        );
    }

    let port_text = port.trim();
    let port = port_text
        .parse::<u16>()
        .with_context(|| format!("invalid WERK_VLLM_PORT value {port_text:?}"))?;
    if port == 0 {
        bail!("WERK_VLLM_PORT must be between 1 and 65535");
    }
    Ok(Some(VllmRemoteConfig {
        host: host.to_string(),
        port,
    }))
}

fn discover_vllm(store: &ModelStore) -> VllmDiscovery {
    let mut attempts = Vec::new();
    let platform = current_vllm_platform();

    match configured_remote_vllm_from_env() {
        Ok(Some(config)) => {
            let usable = remote_supports_vllm(&config.host, config.port);
            attempts.push(VllmDiscoveryAttempt {
                label: "WERK_VLLM_HOST/WERK_VLLM_PORT".to_string(),
                path: None,
                exists: true,
                usable,
                detail: if usable {
                    format!(
                        "remote vLLM server reachable at http://{}:{}",
                        config.host, config.port
                    )
                } else {
                    format!(
                        "remote vLLM server is not ready at http://{}:{}",
                        config.host, config.port
                    )
                },
            });
            // Any HOST/PORT configuration is an explicit remote-runtime
            // choice. Preserve it while a large model is cold-starting instead
            // of silently loading local weights through another vLLM install.
            return VllmDiscovery {
                command: Some(VllmCommand::Remote {
                    host: config.host,
                    port: config.port,
                }),
                source: "env WERK_VLLM_HOST/WERK_VLLM_PORT".to_string(),
                attempts,
            };
        }
        Ok(None) => {}
        Err(error) => {
            attempts.push(VllmDiscoveryAttempt {
                label: "WERK_VLLM_HOST/WERK_VLLM_PORT".to_string(),
                path: None,
                exists: true,
                usable: false,
                detail: compact_error(&error.to_string()),
            });
            return VllmDiscovery {
                command: None,
                source: "invalid env WERK_VLLM_HOST/WERK_VLLM_PORT".to_string(),
                attempts,
            };
        }
    }

    if let Some(path) = env::var_os("WERK_VLLM_PYTHON").map(PathBuf::from) {
        // Discovery verifies only the vLLM API surface. The selected runtime
        // candidate performs CUDA/ROCm (and gfx1151) validation separately so
        // a hybrid Strix host can use a CUDA dGPU or another AMD GPU without
        // inheriting the integrated-device profile.
        let (usable, detail) = python_vllm_status(&path);
        attempts.push(VllmDiscoveryAttempt {
            label: "WERK_VLLM_PYTHON".to_string(),
            path: Some(path.clone()),
            exists: path.is_file(),
            usable,
            detail,
        });
        if usable {
            return VllmDiscovery {
                command: Some(VllmCommand::Python(path)),
                source: "env WERK_VLLM_PYTHON".to_string(),
                attempts,
            };
        }
    }

    if platform == VllmPlatform::StrixHalo {
        attempts.push(VllmDiscoveryAttempt {
            label: "automatic local vLLM discovery".to_string(),
            path: None,
            exists: false,
            usable: false,
            detail: "disabled on Strix Halo; set WERK_VLLM_PYTHON to an explicitly provisioned ROCm vLLM environment reporting gfx1151, or configure WERK_VLLM_HOST/WERK_VLLM_PORT"
                .to_string(),
        });
        return VllmDiscovery {
            command: None,
            source: "Strix Halo requires explicit ROCm vLLM configuration".to_string(),
            attempts,
        };
    }

    let managed_python = managed_vllm_python(store);
    let (managed_usable, managed_detail) = python_vllm_status(&managed_python);
    attempts.push(VllmDiscoveryAttempt {
        label: "managed venv".to_string(),
        path: Some(managed_python.clone()),
        exists: managed_python.is_file(),
        usable: managed_usable,
        detail: managed_detail,
    });
    if managed_usable {
        return VllmDiscovery {
            command: Some(VllmCommand::Python(managed_python)),
            source: "managed venv".to_string(),
            attempts,
        };
    }

    if let Some(path) = find_in_path(vllm_executable_name()) {
        let usable = executable_supports_vllm(&path);
        attempts.push(VllmDiscoveryAttempt {
            label: format!("PATH: {}", vllm_executable_name()),
            path: Some(path.clone()),
            exists: true,
            usable,
            detail: if usable {
                "vLLM executable ok".to_string()
            } else {
                "vLLM executable did not run".to_string()
            },
        });
        if usable {
            return VllmDiscovery {
                command: Some(VllmCommand::Executable(path)),
                source: "PATH".to_string(),
                attempts,
            };
        }
    } else {
        attempts.push(VllmDiscoveryAttempt {
            label: format!("PATH: {}", vllm_executable_name()),
            path: None,
            exists: false,
            usable: false,
            detail: "not found".to_string(),
        });
    }

    for name in ["python3", "python"] {
        if let Some(path) = find_in_path(name) {
            let (usable, detail) = python_vllm_status(&path);
            attempts.push(VllmDiscoveryAttempt {
                label: format!("PATH: {name}"),
                path: Some(path.clone()),
                exists: true,
                usable,
                detail,
            });
            if usable {
                return VllmDiscovery {
                    command: Some(VllmCommand::Python(path)),
                    source: format!("PATH {name}"),
                    attempts,
                };
            }
        }
    }

    VllmDiscovery {
        command: None,
        source: "missing".to_string(),
        attempts,
    }
}

fn missing_vllm_message(discovery: &VllmDiscovery) -> String {
    if let Some(detail) = invalid_remote_vllm_config_detail(discovery) {
        return format!("Invalid remote vLLM configuration: {detail}");
    }
    let mut message = "No vLLM runtime found.\n\nTried:".to_string();
    for attempt in &discovery.attempts {
        let path = attempt
            .path
            .as_ref()
            .map(|path| format!(": {}", path.display()))
            .unwrap_or_default();
        let exists = if attempt.exists { "exists" } else { "missing" };
        let usable = if attempt.usable {
            "usable"
        } else {
            "not usable"
        };
        message.push_str(&format!(
            "\n- {}{} ({exists}, {usable}): {}",
            attempt.label, path, attempt.detail
        ));
    }
    message.push_str("\n\nFix:");
    let platform = current_vllm_platform();
    if platform == VllmPlatform::DgxSpark {
        message
            .push_str("\n- start NVIDIA's Spark-compatible vLLM container with its OpenAI server");
        message.push_str("\n- set WERK_VLLM_HOST=127.0.0.1 and WERK_VLLM_PORT=<published-port>");
        message.push_str(
            "\n- set WERK_VLLM_MODEL=<served-model-id> when /v1/models uses a different ID",
        );
        message.push_str(
            "\n- managed `werk backend install vllm` is intentionally unavailable on DGX Spark",
        );
    } else if platform == VllmPlatform::StrixHalo {
        message.push_str(
            "\n- set WERK_VLLM_PYTHON=/path/to/python-with-rocm-vllm; its PyTorch must report torch.version.hip and gfx1151",
        );
        message.push_str(
            "\n- or start an official ROCm vLLM container and set WERK_VLLM_HOST plus WERK_VLLM_PORT",
        );
        message.push_str(
            "\n- set WERK_VLLM_ACCELERATOR=rocm (or WERK_VLLM_ROCM=1) for a remote ROCm endpoint",
        );
        message.push_str(
            "\n- generic managed `werk backend install vllm` is intentionally unavailable on Strix Halo",
        );
    } else if managed_vllm_install_rejection_for(platform, env::consts::ARCH)
        == Some(LINUX_ARM64_MANAGED_VLLM_MESSAGE)
    {
        message
            .push_str("\n- install a vendor-supported ARM64 vLLM runtime and set WERK_VLLM_PYTHON");
        message.push_str("\n- or set WERK_VLLM_HOST=127.0.0.1 and WERK_VLLM_PORT=<port>");
        message.push_str("\n- generic managed `werk backend install vllm` is intentionally unavailable on Linux aarch64");
    } else {
        message.push_str("\n- set WERK_VLLM_PYTHON=/path/to/python-with-vllm");
        message.push_str("\n- or set WERK_VLLM_HOST=127.0.0.1 and WERK_VLLM_PORT=<port>");
        message.push_str("\n- or run: werk backend install vllm");
    }
    message.push_str("\n- or use: werk --backend candle ...");
    message
}

fn concise_vllm_unavailable_reason(discovery: &VllmDiscovery) -> String {
    concise_vllm_unavailable_reason_for_platform(discovery, current_vllm_platform())
}

fn concise_vllm_unavailable_reason_for_platform(
    discovery: &VllmDiscovery,
    platform: VllmPlatform,
) -> String {
    if let Some(detail) = invalid_remote_vllm_config_detail(discovery) {
        return format!("Invalid remote vLLM configuration: {detail}");
    }
    if platform == VllmPlatform::DgxSpark {
        return DGX_SPARK_VLLM_MESSAGE.to_string();
    }
    if platform == VllmPlatform::StrixHalo {
        if let Some(attempt) = discovery
            .attempts
            .iter()
            .find(|attempt| attempt.label == "WERK_VLLM_PYTHON" && !attempt.usable)
        {
            return format!(
                "WERK_VLLM_PYTHON is not a verified Strix Halo ROCm runtime: {}",
                attempt.detail
            );
        }
        return STRIX_HALO_VLLM_MESSAGE.to_string();
    }
    if discovery.attempts.is_empty() {
        return "No vLLM runtime found; run: werk backend install vllm".to_string();
    }
    if let Some(attempt) = discovery
        .attempts
        .iter()
        .find(|attempt| attempt.exists && !attempt.usable)
        .or_else(|| discovery.attempts.iter().find(|attempt| !attempt.usable))
    {
        let path = attempt
            .path
            .as_ref()
            .map(|path| format!(" ({})", path.display()))
            .unwrap_or_default();
        format!("{}{}: {}", attempt.label, path, attempt.detail)
    } else {
        "No vLLM runtime found; run: werk backend install vllm".to_string()
    }
}

fn invalid_remote_vllm_config_detail(discovery: &VllmDiscovery) -> Option<&str> {
    (discovery.source == "invalid env WERK_VLLM_HOST/WERK_VLLM_PORT")
        .then(|| {
            discovery
                .attempts
                .first()
                .map(|attempt| attempt.detail.as_str())
        })
        .flatten()
}

fn python_vllm_status(path: &Path) -> (bool, String) {
    if !path.is_file() {
        return (false, "Python path does not exist".to_string());
    }
    match Command::new(path)
        .arg("-c")
        .arg("import vllm; import vllm.entrypoints.openai.api_server")
        .output()
    {
        Ok(output) if output.status.success() => (true, "vLLM OpenAI server import ok".to_string()),
        Ok(output) => (
            false,
            command_failure_detail("Python cannot import vLLM OpenAI server", &output),
        ),
        Err(err) => (false, format!("failed to run Python: {err}")),
    }
}

fn python_cuda_status(path: &Path) -> (bool, String) {
    if !path.is_file() {
        return (false, "Python path does not exist".to_string());
    }
    match Command::new(path)
        .arg("-c")
        .arg(cuda_python_probe_script())
        .output()
    {
        Ok(output) if output.status.success() => (
            true,
            format!(
                "PyTorch CUDA runtime detected ({})",
                String::from_utf8_lossy(&output.stdout).trim()
            ),
        ),
        Ok(output) => (
            false,
            command_failure_detail("Python does not expose a CUDA PyTorch stack", &output),
        ),
        Err(err) => (false, format!("failed to run Python: {err}")),
    }
}

fn cuda_python_probe_script() -> &'static str {
    r#"
import vllm
import vllm.entrypoints.openai.api_server
import torch

hip = getattr(getattr(torch, "version", None), "hip", None)
assert not hip, f"torch.version.hip is set ({hip}); this is a ROCm runtime, not CUDA"
cuda = getattr(getattr(torch, "version", None), "cuda", None)
assert cuda, "torch.version.cuda is not set"
assert torch.cuda.is_available(), "CUDA PyTorch reports no GPU"
print(cuda)
"#
}

fn python_rocm_status(path: &Path) -> (bool, String) {
    if !path.is_file() {
        return (false, "Python path does not exist".to_string());
    }
    match Command::new(path)
        .arg("-c")
        .arg(rocm_python_probe_script())
        .output()
    {
        Ok(output) if output.status.success() => (
            true,
            format!(
                "PyTorch ROCm/HIP runtime detected ({})",
                String::from_utf8_lossy(&output.stdout).trim()
            ),
        ),
        Ok(output) => (
            false,
            command_failure_detail("Python does not expose a ROCm/HIP PyTorch stack", &output),
        ),
        Err(err) => (false, format!("failed to run Python: {err}")),
    }
}

fn rocm_python_probe_script() -> &'static str {
    r#"
import vllm
import vllm.entrypoints.openai.api_server
import torch

hip = getattr(getattr(torch, "version", None), "hip", None)
assert hip, "torch.version.hip is not set"
assert torch.cuda.is_available(), "ROCm PyTorch reports no visible GPU"
assert int(torch.cuda.device_count()) > 0, "ROCm PyTorch reports zero logical GPUs"
print(hip)
"#
}

fn python_strix_halo_status(path: &Path) -> (bool, String) {
    if !path.is_file() {
        return (false, "Python path does not exist".to_string());
    }
    let script = strix_halo_python_probe_script();
    match Command::new(path).arg("-c").arg(script).output() {
        Ok(output) if output.status.success() => (
            true,
            String::from_utf8_lossy(&output.stdout).trim().to_string(),
        ),
        Ok(output) => (
            false,
            command_failure_detail(
                "Python is not a verified Strix Halo ROCm vLLM environment",
                &output,
            ),
        ),
        Err(err) => (false, format!("failed to run Python: {err}")),
    }
}

fn strix_halo_python_probe_script() -> &'static str {
    r#"
import vllm
import vllm.entrypoints.openai.api_server
import torch

hip = getattr(getattr(torch, "version", None), "hip", None)
assert hip, "torch.version.hip is not set (CUDA PyTorch is not a Strix Halo runtime)"
assert torch.cuda.is_available(), "ROCm PyTorch reports no GPU"
assert int(torch.cuda.device_count()) > 0, "ROCm PyTorch reports no logical GPU"

properties = torch.cuda.get_device_properties(0)
architecture = ""
for name in ("gcnArchName", "gcn_arch_name"):
    value = getattr(properties, name, None)
    if value:
        architecture = str(value)
        break

assert "gfx1151" in architecture.lower(), \
    "selected logical ROCm device 0 is not gfx1151: " + (architecture or "<unknown>")
print(f"vLLM ROCm/HIP {hip}; gfx1151; FP16 is the validated Strix Halo precision")
"#
}

fn vllm_rocm_capability(command: &VllmCommand) -> Result<String> {
    match command {
        VllmCommand::Python(path) => {
            let (usable, detail) = if current_selected_rocm_device_is_strix_halo() {
                python_strix_halo_status(path)
            } else {
                python_rocm_status(path)
            };
            if usable {
                Ok(detail)
            } else {
                bail!(
                    "vLLM is installed, but the Python environment is not ROCm-capable: {detail}. Install vLLM with a ROCm/HIP PyTorch build or use --backend cuda/auto."
                )
            }
        }
        VllmCommand::Remote { host, port } => {
            if remote_rocm_explicitly_confirmed() {
                Ok(format!(
                    "remote vLLM endpoint at http://{host}:{port} marked ROCm-capable by environment"
                ))
            } else {
                bail!(
                    "remote vLLM endpoint is reachable at http://{host}:{port}, but ROCm capability cannot be inferred. Set WERK_VLLM_ACCELERATOR=rocm or WERK_VLLM_ROCM=1 for an explicitly ROCm-backed remote server."
                )
            }
        }
        VllmCommand::Executable(path) => {
            bail!(
                "vLLM executable {} is installed, but ROCm capability cannot be verified from the executable. Set WERK_VLLM_PYTHON to a Python environment where torch.version.hip is set, or use a remote endpoint with WERK_VLLM_ACCELERATOR=rocm.",
                path.display()
            )
        }
    }
}

fn vllm_cuda_capability(command: &VllmCommand) -> Result<String> {
    vllm_cuda_capability_for(command, remote_rocm_explicitly_confirmed())
}

fn vllm_cuda_capability_for(command: &VllmCommand, remote_rocm: bool) -> Result<String> {
    match command {
        VllmCommand::Python(path) => {
            let (usable, detail) = python_cuda_status(path);
            if usable {
                Ok(detail)
            } else {
                bail!(
                    "vLLM is installed, but the Python environment is not CUDA-capable: {detail}. Install vLLM with a CUDA PyTorch build or select --backend rocm for a ROCm environment."
                )
            }
        }
        VllmCommand::Remote { host, port } => {
            if remote_rocm {
                bail!(
                    "remote vLLM endpoint at http://{host}:{port} is explicitly marked ROCm-backed and cannot satisfy the CUDA runtime candidate"
                )
            }
            Ok(format!(
                "remote vLLM endpoint at http://{host}:{port} is not marked ROCm-backed and is treated as CUDA"
            ))
        }
        VllmCommand::Executable(path) => Ok(format!(
            "vLLM executable {} uses the default CUDA runtime route",
            path.display()
        )),
    }
}

pub(crate) fn vllm_rocm_signals(accelerator: Option<&str>, legacy_rocm: Option<&str>) -> bool {
    accelerator
        .is_some_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "rocm" | "hip"))
        || legacy_rocm.is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "rocm" | "hip"
            )
        })
}

fn remote_rocm_explicitly_confirmed() -> bool {
    let accelerator = env::var("WERK_VLLM_ACCELERATOR").ok();
    let legacy_rocm = env::var("WERK_VLLM_ROCM").ok();
    vllm_rocm_signals(accelerator.as_deref(), legacy_rocm.as_deref())
}

fn executable_supports_vllm(path: &Path) -> bool {
    Command::new(path)
        .arg("serve")
        .arg("--help")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn validate_vllm_python(path: &Path) -> Result<()> {
    let (usable, detail) = python_vllm_status(path);
    if !usable {
        bail!(
            "vLLM installation validation failed for {}: {}",
            path.display(),
            detail
        );
    }
    let output = Command::new(path)
        .arg("-m")
        .arg("vllm.entrypoints.openai.api_server")
        .arg("--help")
        .output()
        .with_context(|| {
            format!(
                "failed to validate vLLM OpenAI server entrypoint with {}",
                path.display()
            )
        })?;
    if !output.status.success() {
        bail!(
            "vLLM OpenAI server entrypoint validation failed for {}: {}",
            path.display(),
            command_failure_detail("server module did not start", &output)
        );
    }
    Ok(())
}

fn remote_supports_vllm(host: &str, port: u16) -> bool {
    let url = format!("http://{host}:{port}");
    get_with_timeout(&url, "/v1/models", REMOTE_DISCOVERY_PROBE_TIMEOUT)
        .map(|response| response.status == 200)
        .unwrap_or(false)
}

fn command_failure_detail(prefix: &str, output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        output.status.to_string()
    };
    format!("{prefix}: {detail}")
}

fn compact_error(reason: &str) -> String {
    reason.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn new_runtime_instance_id() -> Result<String> {
    let mut random = [0_u8; 16];
    getrandom::getrandom(&mut random)
        .map_err(|_| anyhow!("secure vLLM process identity generation is unavailable"))?;
    Ok(format!(
        "vp_{}",
        random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

fn sanitize_runtime_version(version: &str) -> Option<String> {
    let version = version.trim();
    (!version.is_empty()
        && version.len() <= 128
        && version
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_' | b'+')))
    .then(|| version.to_string())
}

fn vllm_version(command: &VllmCommand) -> Option<String> {
    match command {
        VllmCommand::Python(path) => {
            let output = Command::new(path)
                .arg("-c")
                .arg("import vllm; print(getattr(vllm, '__version__', 'unknown'))")
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        }
        VllmCommand::Executable(path) => {
            let output = Command::new(path).arg("--version").output().ok()?;
            output.status.success().then(|| {
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .next()
                    .unwrap_or("unknown")
                    .trim()
                    .to_string()
            })
        }
        VllmCommand::Remote { .. } => {
            Some("remote server; version is not exposed by /v1/models".to_string())
        }
    }
}

fn get_with_timeout(base_url: &str, path: &str, timeout: Duration) -> Result<HttpResponse> {
    request(base_url, path, "GET", None, Some(timeout))
}

fn post_json(base_url: &str, path: &str, body: &Value) -> Result<HttpResponse> {
    request(base_url, path, "POST", Some(body), None)
}

fn request(
    base_url: &str,
    path: &str,
    method: &str,
    body: Option<&Value>,
    timeout: Option<Duration>,
) -> Result<HttpResponse> {
    let (_, host, port) = parse_http_url(base_url)?;
    let deadline = timeout.map(HttpDeadline::new);
    let mut stream = match deadline {
        Some(deadline) => connect_http_with_deadline(&host, port, deadline),
        None => connect_http(&host, port),
    }
    .with_context(|| format!("failed to connect to vLLM server at {base_url}"))?;
    stream.set_nodelay(true).ok();
    let body_text = body.map(serde_json::to_string).transpose()?;
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nAccept: text/event-stream\r\n"
    );
    if let Some(body_text) = &body_text {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", body_text.len()));
    }
    request.push_str("\r\n");
    write_http_bytes(&mut stream, request.as_bytes(), deadline)?;
    if let Some(body_text) = body_text {
        write_http_bytes(&mut stream, body_text.as_bytes(), deadline)?;
    }
    if let Some(deadline) = deadline {
        stream.set_write_timeout(Some(deadline.remaining()?))?;
    }
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let status_line = read_http_line(&mut reader, deadline)?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| anyhow!("invalid HTTP response from vLLM server: {status_line:?}"))?;
    let mut headers = Vec::new();
    loop {
        let line = read_http_line(&mut reader, deadline)?;
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    if status >= 400 {
        if deadline.is_some() {
            bail!("vLLM HTTP {status}");
        } else {
            let mut text = String::new();
            let _ = reader.read_to_string(&mut text);
            bail!("vLLM HTTP {status}: {}", text.trim());
        }
    }
    Ok(HttpResponse {
        status,
        headers,
        reader,
        deadline,
    })
}

fn write_http_bytes(
    stream: &mut TcpStream,
    mut bytes: &[u8],
    deadline: Option<HttpDeadline>,
) -> Result<()> {
    if deadline.is_none() {
        stream.write_all(bytes)?;
        return Ok(());
    }
    while !bytes.is_empty() {
        let remaining = deadline
            .context("missing vLLM HTTP deadline")?
            .remaining()?;
        stream.set_write_timeout(Some(remaining))?;
        let written = stream.write(bytes)?;
        if written == 0 {
            bail!("vLLM HTTP connection closed while writing request");
        }
        bytes = &bytes[written..];
    }
    Ok(())
}

fn read_http_line(
    reader: &mut BufReader<TcpStream>,
    deadline: Option<HttpDeadline>,
) -> Result<String> {
    let Some(deadline) = deadline else {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        return Ok(line);
    };

    let mut bytes = Vec::new();
    loop {
        reader
            .get_mut()
            .set_read_timeout(Some(deadline.remaining()?))?;
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break;
        }
        let count = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|index| index + 1)
            .unwrap_or(available.len());
        let found_newline = available[..count].ends_with(b"\n");
        bytes.extend_from_slice(&available[..count]);
        reader.consume(count);
        if found_newline {
            break;
        }
    }
    String::from_utf8(bytes).context("vLLM returned a non-UTF-8 HTTP header")
}

fn connect_http(host: &str, port: u16) -> Result<TcpStream> {
    let addresses = resolve_http_addresses(host, port)?;
    connect_http_addresses(host, port, &addresses, |_| Ok(HTTP_CONNECT_TIMEOUT))
}

fn connect_http_with_deadline(host: &str, port: u16, deadline: HttpDeadline) -> Result<TcpStream> {
    let addresses = resolve_http_addresses_with_deadline(host, port, deadline)?;
    connect_http_addresses(host, port, &addresses, |_| {
        deadline.remaining_capped(HTTP_CONNECT_TIMEOUT)
    })
}

fn resolve_http_addresses(host: &str, port: u16) -> Result<Vec<SocketAddr>> {
    let addresses = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("could not resolve vLLM host {host}"))?
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        bail!("vLLM host {host} resolved to no socket addresses");
    }
    Ok(addresses)
}

fn resolve_http_addresses_with_deadline(
    host: &str,
    port: u16,
    deadline: HttpDeadline,
) -> Result<Vec<SocketAddr>> {
    let host_owned = host.to_string();
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = (host_owned.as_str(), port)
            .to_socket_addrs()
            .map(|addresses| addresses.collect::<Vec<_>>())
            .map_err(|error| error.to_string());
        let _ = sender.send(result);
    });
    let addresses = receiver
        .recv_timeout(deadline.remaining()?)
        .map_err(|error| anyhow!("timed out resolving vLLM host {host}: {error}"))?
        .map_err(|error| anyhow!("could not resolve vLLM host {host}: {error}"))?;
    if addresses.is_empty() {
        bail!("vLLM host {host} resolved to no socket addresses");
    }
    Ok(addresses)
}

fn connect_http_addresses<F>(
    host: &str,
    port: u16,
    addresses: &[SocketAddr],
    mut timeout_for: F,
) -> Result<TcpStream>
where
    F: FnMut(&SocketAddr) -> Result<Duration>,
{
    let mut errors = Vec::new();
    for address in addresses {
        let timeout = timeout_for(address)?;
        match TcpStream::connect_timeout(address, timeout) {
            Ok(stream) => return Ok(stream),
            Err(error) => errors.push(format!("{address}: {error}")),
        }
    }
    Err(anyhow!(
        "could not connect to {host}:{port}: {}",
        errors.join("; ")
    ))
}

fn stream_body<F>(response: &mut HttpResponse, mut on_bytes: F) -> Result<()>
where
    F: FnMut(&[u8]) -> Result<()>,
{
    if header_contains(&response.headers, "transfer-encoding", "chunked") {
        loop {
            let size_line = read_http_line(&mut response.reader, response.deadline)?;
            let size_text = size_line
                .trim()
                .split_once(';')
                .map(|(size, _)| size)
                .unwrap_or_else(|| size_line.trim());
            let size = usize::from_str_radix(size_text, 16)
                .with_context(|| format!("invalid chunk size from vLLM: {size_text}"))?;
            if size == 0 {
                break;
            }
            let mut chunk = vec![0u8; size];
            read_http_exact(&mut response.reader, &mut chunk, response.deadline)?;
            on_bytes(&chunk)?;
            let mut crlf = [0u8; 2];
            read_http_exact(&mut response.reader, &mut crlf, response.deadline)?;
        }
    } else if let Some(length) = response
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.parse::<usize>().ok())
    {
        let mut remaining = length;
        let mut buffer = [0u8; 8192];
        while remaining > 0 {
            let requested = remaining.min(buffer.len());
            let count = read_http_bytes(
                &mut response.reader,
                &mut buffer[..requested],
                response.deadline,
            )?;
            if count == 0 {
                bail!("vLLM HTTP response ended before Content-Length bytes were received");
            }
            on_bytes(&buffer[..count])?;
            remaining -= count;
        }
    } else {
        let mut buffer = [0u8; 8192];
        loop {
            let n = read_http_bytes(&mut response.reader, &mut buffer, response.deadline)?;
            if n == 0 {
                break;
            }
            on_bytes(&buffer[..n])?;
        }
    }
    Ok(())
}

fn read_http_bytes(
    reader: &mut BufReader<TcpStream>,
    bytes: &mut [u8],
    deadline: Option<HttpDeadline>,
) -> Result<usize> {
    if let Some(deadline) = deadline {
        reader
            .get_mut()
            .set_read_timeout(Some(deadline.remaining()?))?;
    }
    Ok(reader.read(bytes)?)
}

fn read_http_exact(
    reader: &mut BufReader<TcpStream>,
    mut bytes: &mut [u8],
    deadline: Option<HttpDeadline>,
) -> Result<()> {
    while !bytes.is_empty() {
        let count = read_http_bytes(reader, bytes, deadline)?;
        if count == 0 {
            bail!("vLLM HTTP response ended unexpectedly");
        }
        bytes = &mut bytes[count..];
    }
    Ok(())
}

fn parse_http_url(url: &str) -> Result<(String, String, u16)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| anyhow!("only http vLLM URLs are supported: {url}"))?;
    let (host_port, _) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = host_port
        .rsplit_once(':')
        .ok_or_else(|| anyhow!("vLLM URL has no port: {url}"))?;
    Ok(("http".to_string(), host.to_string(), port.parse()?))
}

fn header_contains(headers: &[(String, String)], name: &str, needle: &str) -> bool {
    headers.iter().any(|(header, value)| {
        header.eq_ignore_ascii_case(name) && value.to_ascii_lowercase().contains(needle)
    })
}

fn find_sse_boundary(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(2)
        .position(|window| window == b"\n\n")
        .or_else(|| bytes.windows(4).position(|window| window == b"\r\n\r\n"))
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

fn send_stream_result(
    tx: mpsc::Sender<Result<GenerateStreamEvent, String>>,
    result: Result<GenerateResponse>,
) {
    match result {
        Ok(response) => {
            let _ = tx.blocking_send(Ok(GenerateStreamEvent::Done {
                finish_reason: response.finish_reason,
                prompt_tokens: response.prompt_tokens,
                completion_tokens: response.completion_tokens,
                timings: response.timings,
                backend_diagnostics: response.backend_diagnostics,
            }));
        }
        Err(err) => {
            let _ = tx.blocking_send(Err(format_error_chain(&err)));
        }
    }
}

fn send_text_chunk(
    tx: &Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    chunk: String,
) -> Result<()> {
    if let Some(tx) = tx {
        tx.blocking_send(Ok(GenerateStreamEvent::TextChunk(chunk)))
            .map_err(|err| anyhow!("stream receiver closed: {err}"))?;
    }
    Ok(())
}

fn send_tool_call_delta(
    tx: &Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    tool_calls: Vec<ChatCompletionToolCallDelta>,
) -> Result<()> {
    if let Some(tx) = tx {
        tx.blocking_send(Ok(GenerateStreamEvent::ToolCallDelta(tool_calls)))
            .map_err(|err| anyhow!("stream receiver closed: {err}"))?;
    }
    Ok(())
}

const WERK_OWNED_VLLM_ARGS: &[&str] = &["--model", "--host", "--port", "--served-model-name"];
const VLLM_ENABLE_PREFIX_CACHING_ARG: &str = "--enable-prefix-caching";
const VLLM_DISABLE_PREFIX_CACHING_ARG: &str = "--no-enable-prefix-caching";

fn configured_vllm_args() -> Result<ConfiguredVllmArgs> {
    configured_vllm_args_from(env::var_os("WERK_VLLM_ARGS"))
}

fn configured_vllm_args_from(value: Option<std::ffi::OsString>) -> Result<ConfiguredVllmArgs> {
    let Some(value) = value else {
        return Ok(ConfiguredVllmArgs::default());
    };
    let raw = value.into_string().map_err(|_| {
        anyhow!("WERK_VLLM_ARGS must be valid UTF-8; non-UTF-8 values are not supported")
    })?;
    let args = parse_vllm_args(OsStr::new(&raw))?;
    Ok(ConfiguredVllmArgs {
        raw,
        args,
        werk_managed_prefix_caching: None,
    })
}

fn effective_vllm_args_for_target(
    mut configured_args: ConfiguredVllmArgs,
    command: Option<&VllmCommand>,
    automatic_prefix_caching: Option<bool>,
) -> ConfiguredVllmArgs {
    let werk_starts_local_process = matches!(
        command,
        Some(VllmCommand::Python(_) | VllmCommand::Executable(_))
    );
    let user_selected_prefix_caching = configured_args.args.iter().any(|argument| {
        vllm_argument_matches(argument, VLLM_ENABLE_PREFIX_CACHING_ARG)
            || vllm_argument_matches(argument, VLLM_DISABLE_PREFIX_CACHING_ARG)
    });

    if werk_starts_local_process
        && !user_selected_prefix_caching
        && let Some(enabled) = automatic_prefix_caching
    {
        configured_args.args.push(
            if enabled {
                VLLM_ENABLE_PREFIX_CACHING_ARG
            } else {
                VLLM_DISABLE_PREFIX_CACHING_ARG
            }
            .to_string(),
        );
        configured_args.werk_managed_prefix_caching = Some(enabled);
    }
    configured_args
}

fn validate_werk_managed_prefix_caching_arg(
    command: &VllmCommand,
    configured_args: &ConfiguredVllmArgs,
) -> Result<()> {
    let Some(enabled) = configured_args.werk_managed_prefix_caching else {
        return Ok(());
    };
    let flag = if enabled {
        VLLM_ENABLE_PREFIX_CACHING_ARG
    } else {
        VLLM_DISABLE_PREFIX_CACHING_ARG
    };
    let mut help = match command {
        VllmCommand::Python(path) => {
            let mut command = Command::new(path);
            command
                .arg("-m")
                .arg("vllm.entrypoints.openai.api_server")
                .arg("--help");
            command
        }
        VllmCommand::Executable(path) => {
            let mut command = Command::new(path);
            command.arg("serve").arg("--help");
            command
        }
        VllmCommand::Remote { .. } => return Ok(()),
    };
    let output = help
        .output()
        .with_context(|| format!("failed to inspect vLLM support for {flag}"))?;
    if !output.status.success() {
        bail!(
            "could not verify vLLM support for {flag}: {}",
            command_failure_detail("server help failed", &output)
        );
    }
    if !vllm_help_contains_arg(&output.stdout, &output.stderr, flag) {
        bail!(
            "the installed vLLM runtime does not advertise {flag}; this flag was requested by Werk's serve persistence policy"
        );
    }
    Ok(())
}

fn vllm_help_contains_arg(stdout: &[u8], stderr: &[u8], flag: &str) -> bool {
    let flag = flag.as_bytes();
    if flag.is_empty() {
        return false;
    }
    [stdout, stderr].into_iter().any(|stream| {
        stream
            .windows(flag.len())
            .enumerate()
            .any(|(offset, candidate)| {
                candidate == flag
                    && (offset == 0 || !vllm_arg_name_byte(stream[offset - 1]))
                    && (offset + flag.len() == stream.len()
                        || !vllm_arg_name_byte(stream[offset + flag.len()]))
            })
    })
}

fn vllm_arg_name_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')
}

fn vllm_argument_matches(argument: &str, flag: &str) -> bool {
    argument == flag
        || argument
            .strip_prefix(flag)
            .is_some_and(|suffix| suffix.starts_with('='))
}

/// Parses `WERK_VLLM_ARGS` as a POSIX-style shell-word list, not a shell command.
///
/// `shell_words` only separates words: it does not run a shell or perform
/// command, variable, tilde, or glob expansion.
fn parse_vllm_args(input: &OsStr) -> Result<Vec<String>> {
    let input = input.to_str().ok_or_else(|| {
        anyhow!("WERK_VLLM_ARGS must be valid UTF-8; non-UTF-8 values are not supported")
    })?;
    let trailing_backslashes = input
        .as_bytes()
        .iter()
        .rev()
        .take_while(|byte| **byte == b'\\')
        .count();
    if trailing_backslashes % 2 == 1 {
        bail!(
            "invalid WERK_VLLM_ARGS POSIX-style shell-word list: trailing backslash is not allowed"
        );
    }
    let args = shell_words::split(input)
        .map_err(|error| anyhow!("invalid WERK_VLLM_ARGS POSIX-style shell-word list: {error}"))?;
    validate_vllm_extra_args(&args)?;
    Ok(args)
}

fn validate_vllm_extra_args(args: &[String]) -> Result<()> {
    for argument in args {
        for reserved in WERK_OWNED_VLLM_ARGS {
            if argument == reserved
                || argument
                    .strip_prefix(reserved)
                    .is_some_and(|suffix| suffix.starts_with('='))
            {
                bail!(
                    "WERK_VLLM_ARGS cannot set {reserved}: this server argument is controlled by Werk and must be configured through Werk itself"
                );
            }
        }
    }
    Ok(())
}

fn validate_vllm_args_target(
    command: &VllmCommand,
    configured_args: &ConfiguredVllmArgs,
) -> Result<()> {
    if matches!(command, VllmCommand::Remote { .. }) && !configured_args.raw.is_empty() {
        bail!(
            "WERK_VLLM_ARGS only applies when Werk starts a local vLLM process; configure process arguments on the remote vLLM server"
        );
    }
    Ok(())
}

fn run_command(command: &mut Command, context: &str) -> Result<()> {
    let status = command.status().with_context(|| context.to_string())?;
    if !status.success() {
        bail!("{context}; command exited with {status}");
    }
    Ok(())
}

fn find_bootstrap_python() -> Option<PathBuf> {
    env::var_os("WERK_VLLM_PYTHON")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| find_in_path("python3"))
        .or_else(|| find_in_path("python"))
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(name);
    if path.components().count() > 1 && path.is_file() {
        return Some(path);
    }
    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            let candidate = dir.join(format!("{name}.exe"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn vllm_executable_name() -> &'static str {
    if cfg!(windows) { "vllm.exe" } else { "vllm" }
}

fn free_local_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

fn spawn_log_tail_reader<R>(label: &'static str, reader: R, tail: Arc<Mutex<VecDeque<String>>>)
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(mut tail) = tail.lock() {
                if tail.len() >= 80 {
                    tail.pop_front();
                }
                tail.push_back(format!("{label}: {line}"));
            }
        }
    });
}

fn env_true(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|arg| {
            if arg
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=+".contains(ch))
            {
                arg.clone()
            } else {
                format!("'{}'", arg.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_error_chain(err: &anyhow::Error) -> String {
    let mut parts = err.chain().map(ToString::to_string).collect::<Vec<_>>();
    parts.dedup();
    parts.join(": ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ToolCallingConfig;
    use crate::model_store::ModelSource;
    use crate::openai::{
        ChatCompletionRequest, ChatMessage, ContentPart, ImageUrlPart, ImageUrlSpec, MessageContent,
    };
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn chat_completion_body_uses_openai_messages() {
        let request = GenerateRequest {
            prompt: "ignored rendered prompt".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text("hello".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            image_urls: Vec::new(),
            max_tokens: 32,
            temperature: Some(0.2),
            top_p: None,
            stop: vec!["stop".to_string()],
            seed: Some(7),
            stream_granularity: super::super::StreamGranularity::Chunk,
            verbose: false,
            debug: false,
            tool_config: None,
        };
        let body = chat_completion_body("model", &request, true);
        assert_eq!(body["model"], "model");
        assert_eq!(body["messages"][0]["content"], "hello");
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn vllm_vision_architecture_allowlist_is_exact() {
        for architecture in [
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "glm4v",
            "glm4v_moe",
            "QWEN3_VL",
        ] {
            assert!(
                vllm_architecture_supports_images(Some(architecture)),
                "expected {architecture} to be image-capable"
            );
        }
        for architecture in [
            None,
            Some("qwen2"),
            Some("qwen3"),
            Some("chatglm"),
            Some("glm4"),
            Some("qwen3-vl"),
        ] {
            assert!(
                !vllm_architecture_supports_images(architecture),
                "unexpected image support for {architecture:?}"
            );
        }
    }

    #[test]
    fn vllm_multimodal_body_preserves_order_urls_and_detail() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some(MessageContent::Text("Inspect carefully.".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Parts(vec![
                    ContentPart {
                        kind: "text".to_string(),
                        text: Some("Compare these screenshots:".to_string()),
                        image_url: None,
                    },
                    ContentPart {
                        kind: "image_url".to_string(),
                        text: None,
                        image_url: Some(ImageUrlSpec::Object(ImageUrlPart {
                            url: "data:image/png;base64,first".to_string(),
                            detail: Some("high".to_string()),
                        })),
                    },
                    ContentPart {
                        kind: "text".to_string(),
                        text: Some("against".to_string()),
                        image_url: None,
                    },
                    ContentPart {
                        kind: "input_image".to_string(),
                        text: None,
                        image_url: Some(ImageUrlSpec::Url(
                            "https://example.test/second.png".to_string(),
                        )),
                    },
                ])),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];
        let request = GenerateRequest {
            prompt: "rendered prompt must not replace multipart messages".to_string(),
            messages: messages.clone(),
            image_urls: vec![
                "data:image/png;base64,first".to_string(),
                "https://example.test/second.png".to_string(),
            ],
            max_tokens: 64,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            seed: None,
            stream_granularity: super::super::StreamGranularity::Chunk,
            verbose: false,
            debug: false,
            tool_config: None,
        };

        validate_vllm_image_request(Some("qwen3_vl"), &request).unwrap();
        let body = chat_completion_body("vlm", &request, true);
        assert_eq!(
            body["messages"][0],
            serde_json::to_value(&messages[0]).unwrap()
        );
        assert_eq!(body["messages"][1]["content"][0]["type"], "text");
        assert_eq!(body["messages"][1]["content"][1]["type"], "image_url");
        assert_eq!(
            body["messages"][1]["content"][1]["image_url"]["detail"],
            "high"
        );
        assert_eq!(body["messages"][1]["content"][2]["text"], "against");
        assert_eq!(body["messages"][1]["content"][3]["type"], "image_url");
        assert_eq!(
            body["messages"][1]["content"][3]["image_url"]["url"],
            "https://example.test/second.png"
        );
    }

    #[test]
    fn vllm_image_validation_rejects_local_file_urls() {
        let request = GenerateRequest {
            prompt: "inspect".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Parts(vec![ContentPart {
                    kind: "image_url".to_string(),
                    text: None,
                    image_url: Some(ImageUrlSpec::Url("file:///tmp/private.png".to_string())),
                }])),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            image_urls: vec!["file:///tmp/private.png".to_string()],
            max_tokens: 16,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            seed: None,
            stream_granularity: super::super::StreamGranularity::Chunk,
            verbose: false,
            debug: false,
            tool_config: None,
        };

        let error = validate_vllm_image_request(Some("qwen3_vl"), &request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("data URL or HTTP(S) URL"));
    }

    #[test]
    fn vllm_image_validation_rejects_text_siblings_and_unstructured_images() {
        let multipart_message = ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![ContentPart {
                kind: "image_url".to_string(),
                text: None,
                image_url: Some(ImageUrlSpec::Url("image.png".to_string())),
            }])),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };
        let mut request = GenerateRequest {
            prompt: "inspect".to_string(),
            messages: vec![multipart_message],
            image_urls: vec!["image.png".to_string()],
            max_tokens: 16,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            seed: None,
            stream_granularity: super::super::StreamGranularity::Chunk,
            verbose: false,
            debug: false,
            tool_config: None,
        };

        let error = validate_vllm_image_request(Some("qwen3"), &request).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("explicitly supported VLM architecture")
        );

        request.messages.clear();
        let error = validate_vllm_image_request(Some("qwen3_vl"), &request).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("ordered OpenAI multipart messages")
        );

        request.image_urls.clear();
        validate_vllm_image_request(Some("qwen3"), &request).unwrap();
    }

    #[test]
    fn parses_vllm_streaming_delta_and_usage() {
        let mut completion = VllmCompletion::default();
        let value = json!({
            "choices": [{"delta": {"content": "hello"}, "finish_reason": null}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2}
        });
        assert_eq!(delta_content(&value).as_deref(), Some("hello"));
        update_completion_from_event(&mut completion, &value);
        assert_eq!(completion.prompt_tokens, 4);
        assert_eq!(completion.completion_tokens, 2);
    }

    #[test]
    fn vllm_body_forwards_tool_configuration_and_history_without_rewriting() {
        let public_request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "ignored",
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Berlin\"}"
                        }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "{\"temperature\":21}"
                }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": ["string", "null"]}},
                        "required": ["city"],
                        "additionalProperties": false
                    }
                }
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": false
        }))
        .unwrap();
        let request = GenerateRequest {
            prompt: "ignored".to_string(),
            messages: public_request.messages,
            image_urls: Vec::new(),
            max_tokens: 64,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            seed: None,
            stream_granularity: super::super::StreamGranularity::Chunk,
            verbose: false,
            debug: false,
            tool_config: Some(ToolCallingConfig {
                tools: public_request.tools,
                tool_choice: public_request.tool_choice,
                parallel_tool_calls: public_request.parallel_tool_calls,
            }),
        };

        let body = chat_completion_body("served-model", &request, false);
        assert_eq!(body["stream"], false);
        assert!(body.get("stream_options").is_none());
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(
            body["tools"][0]["function"]["parameters"]["properties"]["city"]["type"],
            json!(["string", "null"])
        );
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Berlin\"}"
        );
        assert!(body["messages"][1]["content"].is_null());
        assert_eq!(body["messages"][2]["tool_call_id"], "call_123");
    }

    #[test]
    fn vllm_non_streaming_tool_response_preserves_null_content_and_call() {
        let value = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Berlin\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 19, "completion_tokens": 7}
        });
        let mut completion = VllmCompletion::default();
        update_completion_from_event(&mut completion, &value);
        update_completion_from_message(&mut completion, &value).unwrap();

        let assistant = completion.assistant_message();
        assert_eq!(assistant.content, None);
        let calls = assistant.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_123");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\":\"Berlin\"}");
        assert_eq!(completion.finish_reason, "tool_calls");
        assert_eq!(completion.prompt_tokens, 19);
        assert_eq!(completion.completion_tokens, 7);
        ensure_vllm_visible_completion(&completion).unwrap();
    }

    #[test]
    fn vllm_streaming_tool_deltas_preserve_indexes_and_partial_strings() {
        let initial = json!({
            "choices": [{
                "delta": {"tool_calls": [
                    {
                        "index": 0,
                        "id": "call_weather",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""}
                    },
                    {
                        "index": 1,
                        "id": "call_time",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": ""}
                    }
                ]},
                "finish_reason": null
            }]
        });
        let fragments = json!({
            "choices": [{
                "delta": {"tool_calls": [
                    {"index": 0, "function": {"name": "", "arguments": "{\"city\""}},
                    {"index": 1, "function": {"arguments": "{\"zone\":\"UTC\"}"}}
                ]},
                "finish_reason": null
            }]
        });

        let initial_calls = delta_tool_calls(&initial).unwrap().unwrap();
        assert_eq!(initial_calls.len(), 2);
        assert_eq!(initial_calls[0].index, 0);
        assert_eq!(initial_calls[0].id.as_deref(), Some("call_weather"));
        assert_eq!(
            initial_calls[0]
                .function
                .as_ref()
                .and_then(|function| function.arguments.as_deref()),
            Some("")
        );
        let partial_calls = delta_tool_calls(&fragments).unwrap().unwrap();
        assert_eq!(partial_calls[0].index, 0);
        assert_eq!(
            partial_calls[0]
                .function
                .as_ref()
                .and_then(|function| function.name.as_deref()),
            Some("")
        );
        assert_eq!(
            partial_calls[0]
                .function
                .as_ref()
                .and_then(|function| function.arguments.as_deref()),
            Some("{\"city\"")
        );
        assert_eq!(partial_calls[1].index, 1);
        assert_eq!(delta_content(&fragments), None);
    }

    #[test]
    fn hidden_reasoning_without_visible_answer_is_not_a_success() {
        for field in ["reasoning", "reasoning_content"] {
            let mut completion = VllmCompletion::default();
            let mut value = json!({
                "choices": [{"delta": {}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 32}
            });
            value["choices"][0]["delta"][field] = json!("internal reasoning");
            update_completion_from_event(&mut completion, &value);
            assert!(completion.saw_reasoning_content);
            let error = ensure_vllm_visible_completion(&completion)
                .unwrap_err()
                .to_string();
            assert!(error.contains("hidden reasoning but no visible answer"));
            assert!(error.contains("increase max tokens"));

            completion.text = "visible answer".to_string();
            ensure_vllm_visible_completion(&completion).unwrap();
        }

        let mut completion_without_usage = VllmCompletion::default();
        update_completion_from_event(
            &mut completion_without_usage,
            &json!({
                "choices": [{
                    "delta": {"reasoning_content": "internal reasoning"},
                    "finish_reason": "length"
                }]
            }),
        );
        assert_eq!(completion_without_usage.completion_tokens, 0);
        let error = ensure_vllm_visible_completion(&completion_without_usage)
            .unwrap_err()
            .to_string();
        assert!(error.contains("hidden reasoning but no visible answer"));
    }

    #[test]
    fn vllm_model_dir_prefers_manifest_config_parent() {
        let store = test_store("vllm-config-parent");
        let manifest = test_manifest(
            "Qwen/Qwen3-4B",
            Some("files/snapshots/main/config.json"),
            Some("files/model.safetensors"),
        );
        let root = store.model_dir(&manifest.id);
        fs::create_dir_all(root.join("files/snapshots/main")).unwrap();
        fs::create_dir_all(root.join("files")).unwrap();
        fs::write(root.join("files/snapshots/main/config.json"), b"{}").unwrap();
        fs::write(root.join("files/config.json"), b"{}").unwrap();

        let resolved = resolve_vllm_model_dir(&store, &manifest).unwrap();
        assert_eq!(resolved, root.join("files/snapshots/main"));
    }

    #[test]
    fn vllm_model_dir_falls_back_to_files_dir_with_config() {
        let store = test_store("vllm-files-dir");
        let manifest = test_manifest("Qwen/Qwen3-4B", None, Some("files/model.safetensors"));
        let root = store.model_dir(&manifest.id);
        fs::create_dir_all(root.join("files")).unwrap();
        fs::write(root.join("files/config.json"), b"{}").unwrap();

        let resolved = resolve_vllm_model_dir(&store, &manifest).unwrap();
        assert_eq!(resolved, root.join("files"));
    }

    #[test]
    fn vllm_model_dir_uses_root_only_when_root_contains_config() {
        let store = test_store("vllm-root-dir");
        let manifest = test_manifest("Qwen/Qwen3-4B", None, None);
        let root = store.model_dir(&manifest.id);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();

        let resolved = resolve_vllm_model_dir(&store, &manifest).unwrap();
        assert_eq!(resolved, root);
    }

    #[test]
    fn vllm_model_dir_rejects_store_root_without_config() {
        let store = test_store("vllm-no-config");
        let manifest = test_manifest("Qwen/Qwen3-4B", None, Some("files/model.safetensors"));
        let root = store.model_dir(&manifest.id);
        fs::create_dir_all(root.join("files")).unwrap();

        let err = resolve_vllm_model_dir(&store, &manifest).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("config.json"));
        assert!(message.contains("Qwen/Qwen3-4B"));
    }

    #[test]
    fn remote_vllm_does_not_require_local_config_or_weights() {
        let store = test_store("vllm-remote-metadata-only");
        let manifest = test_manifest("nvidia/Nemotron-Super", None, None);
        let discovery = VllmDiscovery {
            command: Some(VllmCommand::Remote {
                host: "127.0.0.1".to_string(),
                port: 8000,
            }),
            source: "test remote".to_string(),
            attempts: Vec::new(),
        };

        let resolved = resolve_vllm_model_dir_for_discovery(&store, &manifest, &discovery).unwrap();
        assert_eq!(resolved, store.model_dir(&manifest.id));
        assert!(!resolved.exists());
    }

    #[test]
    fn failed_remote_attempt_that_falls_back_local_still_requires_local_weights() {
        let store = test_store("vllm-failed-remote-local-fallback");
        let manifest = test_manifest("nvidia/Nemotron-Super", None, None);
        let discovery = VllmDiscovery {
            command: Some(VllmCommand::Python(PathBuf::from("/usr/bin/python3"))),
            source: "PATH python3".to_string(),
            attempts: vec![VllmDiscoveryAttempt {
                label: "WERK_VLLM_HOST/WERK_VLLM_PORT".to_string(),
                path: None,
                exists: false,
                usable: false,
                detail: "remote endpoint unreachable".to_string(),
            }],
        };

        let error = resolve_vllm_model_dir_for_discovery(&store, &manifest, &discovery)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not exist"));
    }

    #[test]
    fn vllm_cache_identity_distinguishes_remote_and_python_runtimes() {
        let remote_a = VllmDiscovery {
            command: Some(VllmCommand::Remote {
                host: "spark-a".to_string(),
                port: 8000,
            }),
            source: "env".to_string(),
            attempts: Vec::new(),
        };
        let remote_b = VllmDiscovery {
            command: Some(VllmCommand::Remote {
                host: "spark-b".to_string(),
                port: 9000,
            }),
            source: "env".to_string(),
            attempts: Vec::new(),
        };
        let python = VllmDiscovery {
            command: Some(VllmCommand::Python(PathBuf::from("/opt/vllm/bin/python"))),
            source: "env".to_string(),
            attempts: Vec::new(),
        };
        assert_ne!(
            vllm_discovery_cache_identity(&remote_a),
            vllm_discovery_cache_identity(&remote_b)
        );
        assert_ne!(
            vllm_discovery_cache_identity(&remote_a),
            vllm_discovery_cache_identity(&python)
        );
        assert!(vllm_discovery_cache_identity(&python).contains("/opt/vllm/bin/python"));

        let base = VllmCacheEnvironment {
            host: "spark-a".to_string(),
            port: "8000".to_string(),
            python: "/opt/vllm-a/bin/python".to_string(),
            ..Default::default()
        };
        let manifest = test_manifest("nemotron", None, None);
        let model_identity = ModelRuntimeIdentity::from_manifest(&manifest).unwrap();
        let base_key = vllm_server_cache_key(
            &model_identity,
            Path::new("/models/nemotron"),
            &remote_a,
            &base,
        );
        for changed in [
            VllmCacheEnvironment {
                host: "spark-b".to_string(),
                ..base.clone()
            },
            VllmCacheEnvironment {
                port: "9000".to_string(),
                ..base.clone()
            },
            VllmCacheEnvironment {
                python: "/opt/vllm-b/bin/python".to_string(),
                ..base.clone()
            },
        ] {
            assert_ne!(
                base_key,
                vllm_server_cache_key(
                    &model_identity,
                    Path::new("/models/nemotron"),
                    &remote_a,
                    &changed,
                )
            );
        }
    }

    #[test]
    fn vllm_args_keep_logical_served_model_name() {
        let model_dir = PathBuf::from("/tmp/werk-model/files");
        let launch = vllm_launch_command(
            &VllmCommand::Python(PathBuf::from("/usr/bin/python3")),
            &model_dir,
            "Qwen/Qwen3-4B",
            12345,
            &[],
        )
        .unwrap();
        let model_arg = launch
            .args
            .windows(2)
            .find(|pair| pair[0] == "--model")
            .map(|pair| pair[1].as_str());
        let served_name = launch
            .args
            .windows(2)
            .find(|pair| pair[0] == "--served-model-name")
            .map(|pair| pair[1].as_str());
        assert_eq!(model_arg, Some("/tmp/werk-model/files"));
        assert_eq!(served_name, Some("Qwen/Qwen3-4B"));
    }

    #[test]
    fn vllm_args_parse_separate_and_equals_forms() {
        assert_eq!(
            parse_vllm_args(OsStr::new("--max-num-seqs 16")).unwrap(),
            vec!["--max-num-seqs", "16"]
        );
        assert_eq!(
            parse_vllm_args(OsStr::new("--max-num-seqs=16")).unwrap(),
            vec!["--max-num-seqs=16"]
        );
    }

    #[test]
    fn local_vllm_prefix_caching_default_is_added_only_when_requested() {
        let command = VllmCommand::Python(PathBuf::from("/opt/vllm/bin/python"));

        let ordinary = effective_vllm_args_for_target(
            configured_vllm_args_from(None).unwrap(),
            Some(&command),
            None,
        );
        assert!(ordinary.args.is_empty());

        let persistent = effective_vllm_args_for_target(
            configured_vllm_args_from(None).unwrap(),
            Some(&command),
            Some(true),
        );
        assert_eq!(persistent.args, [VLLM_ENABLE_PREFIX_CACHING_ARG]);
        assert!(persistent.raw.is_empty());
        assert_eq!(persistent.werk_managed_prefix_caching, Some(true));

        let persistence_without_reuse = effective_vllm_args_for_target(
            configured_vllm_args_from(None).unwrap(),
            Some(&command),
            Some(false),
        );
        assert_eq!(
            persistence_without_reuse.args,
            [VLLM_DISABLE_PREFIX_CACHING_ARG]
        );
        assert_eq!(
            persistence_without_reuse.werk_managed_prefix_caching,
            Some(false)
        );

        let tuned = effective_vllm_args_for_target(
            configured_vllm_args_from(Some("--max-num-seqs 8".into())).unwrap(),
            Some(&command),
            Some(true),
        );
        assert_eq!(
            tuned.args,
            ["--max-num-seqs", "8", VLLM_ENABLE_PREFIX_CACHING_ARG]
        );

        let launch = vllm_launch_command(
            &command,
            Path::new("/models/qwen"),
            "qwen",
            43127,
            &persistent.args,
        )
        .unwrap();
        assert_eq!(
            launch.args.last().map(String::as_str),
            Some(VLLM_ENABLE_PREFIX_CACHING_ARG)
        );

        let cache_environment = VllmCacheEnvironment::current(&persistent);
        assert_eq!(
            cache_environment.args,
            serde_json::to_string(&persistent.args).unwrap()
        );
        assert!(
            cache_environment
                .args
                .contains(VLLM_ENABLE_PREFIX_CACHING_ARG)
        );
    }

    #[test]
    fn managed_prefix_cache_flag_requires_exact_installed_help_support() {
        for (stdout, stderr) in [
            (
                b"options: --enable-prefix-caching".as_slice(),
                b"".as_slice(),
            ),
            (b"[--enable-prefix-caching]".as_slice(), b"".as_slice()),
            (b"--enable-prefix-caching=true".as_slice(), b"".as_slice()),
            (
                b"".as_slice(),
                b"  --enable-prefix-caching, --other".as_slice(),
            ),
        ] {
            assert!(vllm_help_contains_arg(
                stdout,
                stderr,
                VLLM_ENABLE_PREFIX_CACHING_ARG
            ));
        }
        assert!(vllm_help_contains_arg(
            b"",
            b"--no-enable-prefix-caching",
            VLLM_DISABLE_PREFIX_CACHING_ARG
        ));
        assert!(!vllm_help_contains_arg(
            b"--no-enable-prefix-caching-legacy",
            b"",
            VLLM_DISABLE_PREFIX_CACHING_ARG
        ));

        for false_positive in [
            b"--enable-other-cache".as_slice(),
            b"--enable-prefix-caching-extra".as_slice(),
            b"prefix--enable-prefix-caching".as_slice(),
            b"x--enable-prefix-caching-y".as_slice(),
        ] {
            assert!(!vllm_help_contains_arg(
                false_positive,
                b"",
                VLLM_ENABLE_PREFIX_CACHING_ARG
            ));
        }
        assert!(!vllm_help_contains_arg(b"anything", b"", ""));
    }

    #[cfg(unix)]
    #[test]
    fn managed_prefix_cache_validation_probes_the_matching_local_entrypoint() {
        let python = fake_python(
            "prefix-help-python",
            r#"if [ "$1" = "-m" ] && [ "$2" = "vllm.entrypoints.openai.api_server" ] && [ "$3" = "--help" ]; then
    printf '%s\n' 'options: --enable-prefix-caching'
    exit 0
fi
printf '%s\n' 'unexpected python arguments' >&2
exit 9
"#,
        );
        let python_command = VllmCommand::Python(python);
        let configured = effective_vllm_args_for_target(
            configured_vllm_args_from(None).unwrap(),
            Some(&python_command),
            Some(true),
        );
        validate_werk_managed_prefix_caching_arg(&python_command, &configured).unwrap();

        let executable = fake_python(
            "prefix-help-executable",
            r#"if [ "$1" = "serve" ] && [ "$2" = "--help" ]; then
    printf '%s\n' '--no-enable-prefix-caching' >&2
    exit 0
fi
printf '%s\n' 'unexpected executable arguments' >&2
exit 9
"#,
        );
        let executable_command = VllmCommand::Executable(executable);
        let configured = effective_vllm_args_for_target(
            configured_vllm_args_from(None).unwrap(),
            Some(&executable_command),
            Some(false),
        );
        validate_werk_managed_prefix_caching_arg(&executable_command, &configured).unwrap();

        let misleading = fake_python(
            "prefix-help-misleading",
            "printf '%s\\n' '--enable-prefix-caching-extra'\nexit 0\n",
        );
        let misleading_command = VllmCommand::Executable(misleading);
        let configured = effective_vllm_args_for_target(
            configured_vllm_args_from(None).unwrap(),
            Some(&misleading_command),
            Some(true),
        );
        let error = validate_werk_managed_prefix_caching_arg(&misleading_command, &configured)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not advertise --enable-prefix-caching"));
    }

    #[test]
    fn explicit_vllm_prefix_caching_choices_override_the_serve_default() {
        let command = VllmCommand::Executable(PathBuf::from("/opt/vllm/bin/vllm"));

        let enabled =
            configured_vllm_args_from(Some("--max-num-seqs 8 --enable-prefix-caching".into()))
                .unwrap();
        let enabled_args = enabled.args.clone();
        let effective = effective_vllm_args_for_target(enabled, Some(&command), Some(false));
        assert_eq!(effective.args, enabled_args);
        assert_eq!(effective.werk_managed_prefix_caching, None);
        assert!(validate_werk_managed_prefix_caching_arg(&command, &effective).is_ok());

        let disabled =
            configured_vllm_args_from(Some("--no-enable-prefix-caching --max-num-seqs 8".into()))
                .unwrap();
        let disabled_args = disabled.args.clone();
        let effective = effective_vllm_args_for_target(disabled, Some(&command), Some(true));
        assert_eq!(effective.args, disabled_args);
        assert_eq!(effective.werk_managed_prefix_caching, None);
        assert!(
            effective
                .args
                .iter()
                .any(|arg| arg == VLLM_DISABLE_PREFIX_CACHING_ARG)
        );
        assert!(
            !effective
                .args
                .iter()
                .any(|arg| arg == VLLM_ENABLE_PREFIX_CACHING_ARG)
        );

        let similar = configured_vllm_args_from(Some("--enable-prefix-caching-extra".into()))
            .expect("syntactically valid user args");
        let effective = effective_vllm_args_for_target(similar, Some(&command), Some(true));
        assert_eq!(
            effective.args,
            [
                "--enable-prefix-caching-extra",
                VLLM_ENABLE_PREFIX_CACHING_ARG
            ]
        );
        assert_eq!(effective.werk_managed_prefix_caching, Some(true));

        let equals_form =
            configured_vllm_args_from(Some("--no-enable-prefix-caching=true".into())).unwrap();
        let effective = effective_vllm_args_for_target(equals_form, Some(&command), Some(true));
        assert_eq!(effective.args, ["--no-enable-prefix-caching=true"]);
        assert_eq!(effective.werk_managed_prefix_caching, None);
    }

    #[test]
    fn remote_vllm_never_receives_werk_generated_prefix_caching_args() {
        let remote = VllmCommand::Remote {
            host: "vllm.internal".to_string(),
            port: 8000,
        };
        let effective = effective_vllm_args_for_target(
            configured_vllm_args_from(None).unwrap(),
            Some(&remote),
            Some(true),
        );
        assert!(effective.raw.is_empty());
        assert!(effective.args.is_empty());
        assert_eq!(effective.werk_managed_prefix_caching, None);
        assert!(validate_vllm_args_target(&remote, &effective).is_ok());
    }

    #[test]
    fn effective_vllm_prefix_caching_args_change_the_server_cache_key() {
        let command = VllmCommand::Python(PathBuf::from("/opt/vllm/bin/python"));
        let discovery = VllmDiscovery {
            command: Some(command.clone()),
            source: "test".to_string(),
            attempts: Vec::new(),
        };
        let ordinary = effective_vllm_args_for_target(
            configured_vllm_args_from(None).unwrap(),
            Some(&command),
            None,
        );
        let persistent = effective_vllm_args_for_target(
            configured_vllm_args_from(None).unwrap(),
            Some(&command),
            Some(true),
        );
        let manifest = test_manifest("qwen", None, None);
        let model_identity = ModelRuntimeIdentity::from_manifest(&manifest).unwrap();

        assert_ne!(
            vllm_server_cache_key(
                &model_identity,
                Path::new("/models/qwen"),
                &discovery,
                &VllmCacheEnvironment {
                    args: serde_json::to_string(&ordinary.args).unwrap(),
                    ..Default::default()
                },
            ),
            vllm_server_cache_key(
                &model_identity,
                Path::new("/models/qwen"),
                &discovery,
                &VllmCacheEnvironment {
                    args: serde_json::to_string(&persistent.args).unwrap(),
                    ..Default::default()
                },
            )
        );
    }

    #[test]
    fn vllm_args_preserve_json_and_quoted_spaces() {
        assert_eq!(
            parse_vllm_args(OsStr::new(
                r#"--speculative-config '{"method":"mtp","num_speculative_tokens":1}' --label "two words""#,
            ))
            .unwrap(),
            vec![
                "--speculative-config",
                r#"{"method":"mtp","num_speculative_tokens":1}"#,
                "--label",
                "two words",
            ]
        );
    }

    #[test]
    fn vllm_args_preserve_explicitly_empty_words() {
        assert_eq!(
            parse_vllm_args(OsStr::new(r#"--double "" --single ''"#)).unwrap(),
            vec!["--double", "", "--single", ""]
        );
    }

    #[test]
    fn vllm_args_treat_backslashes_literally_inside_single_quotes() {
        assert_eq!(
            parse_vllm_args(OsStr::new(r"--value 'left\right'")).unwrap(),
            vec!["--value", r"left\right"]
        );
    }

    #[test]
    fn vllm_args_preserve_repeated_flags_and_order() {
        assert_eq!(
            parse_vllm_args(OsStr::new("--include first --other middle --include last")).unwrap(),
            vec![
                "--include",
                "first",
                "--other",
                "middle",
                "--include",
                "last"
            ]
        );
    }

    #[test]
    fn vllm_args_reject_malformed_shell_words() {
        for malformed in [
            "--value 'unclosed",
            "--value \"unclosed",
            "--value trailing\\",
        ] {
            let error = parse_vllm_args(OsStr::new(malformed))
                .unwrap_err()
                .to_string();
            assert!(error.contains("invalid WERK_VLLM_ARGS"), "{error}");
            assert!(error.contains("POSIX-style shell-word list"), "{error}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn vllm_args_reject_non_utf8_environment_values() {
        use std::os::unix::ffi::OsStringExt;

        let error =
            configured_vllm_args_from(Some(std::ffi::OsString::from_vec(b"--value \xff".to_vec())))
                .unwrap_err()
                .to_string();
        assert!(error.contains("WERK_VLLM_ARGS must be valid UTF-8"));
    }

    #[test]
    fn vllm_args_reject_every_werk_owned_flag_in_separate_form() {
        for reserved in WERK_OWNED_VLLM_ARGS {
            let error = parse_vllm_args(OsStr::new(&format!("{reserved} user-value")))
                .unwrap_err()
                .to_string();
            assert!(error.contains(reserved), "{error}");
            assert!(error.contains("controlled by Werk"), "{error}");
        }
    }

    #[test]
    fn vllm_args_reject_every_werk_owned_flag_in_equals_form() {
        for reserved in WERK_OWNED_VLLM_ARGS {
            let error = parse_vllm_args(OsStr::new(&format!("{reserved}=user-value")))
                .unwrap_err()
                .to_string();
            assert!(error.contains(reserved), "{error}");
            assert!(error.contains("controlled by Werk"), "{error}");
        }
    }

    #[test]
    fn remote_vllm_rejects_local_process_args() {
        let args = parse_vllm_args(OsStr::new("--max-num-seqs 16")).unwrap();
        let configured_args = ConfiguredVllmArgs {
            raw: "--max-num-seqs 16".to_string(),
            args,
            werk_managed_prefix_caching: None,
        };
        let error = validate_vllm_args_target(
            &VllmCommand::Remote {
                host: "vllm.internal".to_string(),
                port: 8000,
            },
            &configured_args,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("only applies when Werk starts a local vLLM process"));
        assert!(error.contains("remote vLLM server"));
    }

    #[test]
    fn remote_vllm_rejects_nonempty_whitespace_only_args_value() {
        let configured_args = configured_vllm_args_from(Some("   ".into())).unwrap();
        assert!(configured_args.args.is_empty());

        let error = validate_vllm_args_target(
            &VllmCommand::Remote {
                host: "vllm.internal".to_string(),
                port: 8000,
            },
            &configured_args,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("only applies when Werk starts a local vLLM process"));
    }

    #[test]
    fn vllm_args_do_not_expand_shell_syntax() {
        assert_eq!(
            parse_vllm_args(OsStr::new(
                r#"--variable '$WERK_SECRET' --command '$(touch /tmp/never)' --glob '*.safetensors'"#,
            ))
            .unwrap(),
            vec![
                "--variable",
                "$WERK_SECRET",
                "--command",
                "$(touch /tmp/never)",
                "--glob",
                "*.safetensors",
            ]
        );
    }

    #[test]
    fn vllm_python_launch_argv_preserves_advanced_flags_exactly() {
        let extra = parse_vllm_args(OsStr::new(
            r#"--quantization compressed-tensors --kv-cache-dtype fp8 --speculative-config '{"method":"mtp","num_speculative_tokens":1}' --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-num-seqs 16"#,
        ))
        .unwrap();
        let launch = vllm_launch_command(
            &VllmCommand::Python(PathBuf::from("/opt/vllm/bin/python")),
            Path::new("/srv/werk1112/models/qwen/files"),
            "Qwen-Test",
            43127,
            &extra,
        )
        .unwrap();

        assert_eq!(launch.program, PathBuf::from("/opt/vllm/bin/python"));
        assert_eq!(
            launch.args,
            vec![
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                "/srv/werk1112/models/qwen/files",
                "--host",
                "127.0.0.1",
                "--port",
                "43127",
                "--served-model-name",
                "Qwen-Test",
                "--quantization",
                "compressed-tensors",
                "--kv-cache-dtype",
                "fp8",
                "--speculative-config",
                r#"{"method":"mtp","num_speculative_tokens":1}"#,
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
                "--max-num-seqs",
                "16",
            ]
        );
    }

    #[test]
    fn vllm_executable_launch_argv_preserves_advanced_flags_exactly() {
        let extra = parse_vllm_args(OsStr::new(
            r#"--quantization compressed-tensors --kv-cache-dtype fp8 --speculative-config '{"method":"mtp","num_speculative_tokens":1}' --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-num-seqs 16"#,
        ))
        .unwrap();
        let launch = vllm_launch_command(
            &VllmCommand::Executable(PathBuf::from("/opt/vllm/bin/vllm")),
            Path::new("/srv/werk1112/models/qwen/files"),
            "Qwen-Test",
            43127,
            &extra,
        )
        .unwrap();

        assert_eq!(launch.program, PathBuf::from("/opt/vllm/bin/vllm"));
        assert_eq!(
            launch.args,
            vec![
                "serve",
                "/srv/werk1112/models/qwen/files",
                "--host",
                "127.0.0.1",
                "--port",
                "43127",
                "--served-model-name",
                "Qwen-Test",
                "--quantization",
                "compressed-tensors",
                "--kv-cache-dtype",
                "fp8",
                "--speculative-config",
                r#"{"method":"mtp","num_speculative_tokens":1}"#,
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
                "--max-num-seqs",
                "16",
            ]
        );
    }

    #[test]
    fn remote_served_model_prefers_exact_werk_id() {
        let models = vec![
            "remote-alias".to_string(),
            "nvidia/Nemotron-3-Nano".to_string(),
        ];
        let selected = select_remote_served_model("nvidia/Nemotron-3-Nano", &models).unwrap();
        assert_eq!(selected.name, "nvidia/Nemotron-3-Nano");
        assert_eq!(selected.source, "matching /v1/models entry");
    }

    #[test]
    fn remote_served_model_maps_only_advertised_model_for_spark() {
        let models = vec!["nemotron-super-nvfp4".to_string()];
        let selected = select_remote_served_model("local-nemotron", &models).unwrap();
        assert_eq!(selected.name, "nemotron-super-nvfp4");
        assert_eq!(selected.source, "only /v1/models entry");
    }

    #[test]
    fn remote_served_model_rejects_ambiguous_mapping() {
        let models = vec!["model-a".to_string(), "model-b".to_string()];
        let error = select_remote_served_model("local-nemotron", &models).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("multiple models"));
        assert!(message.contains("WERK_VLLM_MODEL"));
        assert!(message.contains("model-a, model-b"));
    }

    #[test]
    fn configured_remote_model_must_be_advertised() {
        let models = vec!["model-a".to_string(), "model-b".to_string()];
        let selected = select_configured_remote_served_model("model-b", &models).unwrap();
        assert_eq!(selected.name, "model-b");
        assert_eq!(selected.source, "verified WERK_VLLM_MODEL");

        let error = select_configured_remote_served_model("typo", &models)
            .unwrap_err()
            .to_string();
        assert!(error.contains("WERK_VLLM_MODEL 'typo' is not advertised"));
        assert!(error.contains("model-a, model-b"));
        assert!(remote_models_include_served_name("model-b", &models));
        assert!(!remote_models_include_served_name("old-model", &models));
    }

    #[test]
    fn remote_models_response_is_sanitized_and_deduplicated() {
        let models = parse_remote_vllm_model_ids(
            br#"{"data":[{"id":" model-b "},{"id":"model-a"},{"id":"model-b"},{"id":"bad\nmodel"},{"id":""},{"object":"model"}]}"#,
        )
        .unwrap();
        assert_eq!(models, vec!["model-a", "model-b"]);
    }

    #[test]
    fn configured_vllm_model_is_trimmed_and_rejects_controls() {
        assert_eq!(
            validate_configured_vllm_model("  nemotron-served  ").unwrap(),
            "nemotron-served"
        );
        assert!(validate_configured_vllm_model("  ").is_err());
        assert!(validate_configured_vllm_model("model\nsecond").is_err());
    }

    #[test]
    fn nemotron_architectures_and_reasoning_parser_args_are_explicit() {
        assert!(is_nemotron_architecture_name("nemotron_h"));
        assert!(is_nemotron_architecture_name("NEMOTRON_H_MOE"));
        assert!(!is_nemotron_architecture_name("nemotron_omni"));
        assert!(has_reasoning_parser_arg(&[
            "--reasoning-parser".to_string(),
            "deepseek_r1".to_string(),
        ]));
        assert!(has_reasoning_parser_arg(&[
            "--reasoning-parser=deepseek_r1".to_string(),
        ]));
        assert!(!has_reasoning_parser_arg(&[
            "--trust-remote-code".to_string(),
        ]));
    }

    #[test]
    fn linux_release_detection_identifies_wsl() {
        assert!(linux_release_looks_like_wsl(
            "5.15.167.4-microsoft-standard-WSL2"
        ));
        assert!(linux_release_looks_like_wsl("Linux version 6.6.0 WSL2"));
        assert!(!linux_release_looks_like_wsl("6.8.0-63-generic"));
    }

    #[test]
    fn local_vllm_policy_rejects_wsl_with_fallback_message() {
        let reason = local_vllm_platform_rejection(VllmPlatform::Wsl).unwrap();
        assert!(reason.contains("vLLM is a Linux-native runtime"));
        assert!(reason.contains("Werk will fall back to Candle CUDA"));
        assert!(reason.contains("remote vLLM server"));
        assert!(local_vllm_platform_rejection(VllmPlatform::NativeLinux).is_none());
        assert!(local_vllm_platform_rejection(VllmPlatform::StrixHalo).is_none());
    }

    #[test]
    fn failed_remote_attempt_does_not_make_local_wsl_vllm_eligible() {
        let discovery = VllmDiscovery {
            command: Some(VllmCommand::Python(PathBuf::from("/usr/bin/python3"))),
            source: "PATH python3".to_string(),
            attempts: vec![VllmDiscoveryAttempt {
                label: "WERK_VLLM_HOST/WERK_VLLM_PORT".to_string(),
                path: None,
                exists: false,
                usable: false,
                detail: "unreachable".to_string(),
            }],
        };
        let reason = local_vllm_platform_rejection_for_discovery_with_platform(
            &discovery,
            VllmPlatform::Wsl,
        )
        .unwrap();
        assert!(reason.contains("WSL"));
    }

    #[test]
    fn managed_vllm_install_policy_blocks_unverified_linux_arm64() {
        assert!(managed_vllm_install_rejection_for(VllmPlatform::NativeLinux, "x86_64").is_none());
        assert!(managed_vllm_install_rejection_for(VllmPlatform::Wsl, "x86_64").is_none());

        for platform in [VllmPlatform::NativeLinux, VllmPlatform::Wsl] {
            let arm64 = managed_vllm_install_rejection_for(platform, "aarch64").unwrap();
            assert!(arm64.contains("Linux aarch64 detected"));
            assert!(arm64.contains("WERK_VLLM_PYTHON"));
            assert!(arm64.contains("WERK_VLLM_HOST"));
        }

        let spark = managed_vllm_install_rejection_for(VllmPlatform::DgxSpark, "aarch64").unwrap();
        assert!(spark.contains("DGX Spark detected"));
        assert!(spark.contains("Spark-compatible vLLM container"));
        assert!(spark.contains("intentionally not offered"));

        let strix = managed_vllm_install_rejection_for(VllmPlatform::StrixHalo, "x86_64").unwrap();
        assert!(strix.contains("Strix Halo detected"));
        assert!(strix.contains("ROCm vLLM"));
        assert!(strix.contains("torch.version.hip"));
        assert!(strix.contains("gfx1151"));
        assert!(strix.contains("intentionally not offered"));

        let windows =
            managed_vllm_install_rejection_for(VllmPlatform::NativeWindows, "x86_64").unwrap();
        assert!(windows.contains("Native Windows local vLLM is not eligible"));

        let macos = managed_vllm_install_rejection_for(VllmPlatform::Macos, "aarch64").unwrap();
        assert!(macos.contains("not eligible on macOS"));
    }

    #[test]
    fn remote_vllm_configuration_is_strict_and_header_safe() {
        assert_eq!(configured_remote_vllm(None, None).unwrap(), None);
        assert_eq!(
            configured_remote_vllm(Some(" spark.local "), Some(" 8000 ")).unwrap(),
            Some(VllmRemoteConfig {
                host: "spark.local".to_string(),
                port: 8000,
            })
        );

        for (host, port, expected) in [
            (Some("spark.local"), None, "must be set together"),
            (None, Some("8000"), "must be set together"),
            (Some(""), Some("8000"), "must not be empty"),
            (
                Some("spark.local\r\nX-Injected: true"),
                Some("8000"),
                "control characters",
            ),
            (Some("::1"), Some("8000"), "IPv6 literals"),
            (
                Some("spark.local/path"),
                Some("8000"),
                "unsupported characters",
            ),
            (Some("spark.local"), Some("0"), "between 1 and 65535"),
            (
                Some("spark.local"),
                Some("invalid"),
                "invalid WERK_VLLM_PORT",
            ),
        ] {
            let error = configured_remote_vllm(host, port).unwrap_err().to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    fn invalid_remote_intent_is_actionable_and_never_treated_as_local() {
        let discovery = VllmDiscovery {
            command: None,
            source: "invalid env WERK_VLLM_HOST/WERK_VLLM_PORT".to_string(),
            attempts: vec![VllmDiscoveryAttempt {
                label: "WERK_VLLM_HOST/WERK_VLLM_PORT".to_string(),
                path: None,
                exists: true,
                usable: false,
                detail: "WERK_VLLM_HOST and WERK_VLLM_PORT must be set together".to_string(),
            }],
        };
        assert!(invalid_remote_vllm_config_detail(&discovery).is_some());
        assert!(
            local_vllm_platform_rejection_for_discovery_with_platform(
                &discovery,
                VllmPlatform::Wsl,
            )
            .is_none()
        );
        let message = missing_vllm_message(&discovery);
        assert!(message.starts_with("Invalid remote vLLM configuration:"));
        assert!(message.contains("must be set together"));
        assert!(!message.contains("werk backend install vllm"));
    }

    #[test]
    fn dgx_spark_detection_requires_aarch64_and_a_spark_or_gb10_signal() {
        assert!(dgx_spark_signals(
            "aarch64",
            Some("NVIDIA DGX Spark\0"),
            None,
        ));
        assert!(dgx_spark_signals("aarch64", None, Some("NVIDIA GB10"),));
        assert!(!dgx_spark_signals(
            "x86_64",
            Some("NVIDIA DGX Spark"),
            Some("NVIDIA GB10"),
        ));
        assert!(!dgx_spark_signals(
            "aarch64",
            Some("Generic ARM server"),
            Some("NVIDIA H100"),
        ));
        assert!(!dgx_spark_signals(
            "aarch64",
            Some("Generic ARM server"),
            Some("NVIDIA GB100"),
        ));
    }

    #[test]
    fn missing_vllm_on_spark_recommends_remote_container_not_managed_pip() {
        let discovery = VllmDiscovery {
            command: None,
            source: "missing".to_string(),
            attempts: Vec::new(),
        };
        let detail =
            concise_vllm_unavailable_reason_for_platform(&discovery, VllmPlatform::DgxSpark);
        assert!(detail.contains("WERK_VLLM_HOST"));
        assert!(detail.contains("WERK_VLLM_MODEL"));
        assert!(detail.contains("intentionally not offered"));
        assert!(!detail.contains("run: werk backend install vllm"));
    }

    #[test]
    fn missing_vllm_on_strix_halo_recommends_explicit_rocm_not_managed_pip() {
        let discovery = VllmDiscovery {
            command: None,
            source: "missing".to_string(),
            attempts: Vec::new(),
        };
        let detail =
            concise_vllm_unavailable_reason_for_platform(&discovery, VllmPlatform::StrixHalo);
        assert!(detail.contains("WERK_VLLM_PYTHON"));
        assert!(detail.contains("WERK_VLLM_HOST"));
        assert!(detail.contains("WERK_VLLM_ACCELERATOR=rocm"));
        assert!(detail.contains("gfx1151"));
        assert!(detail.contains("intentionally not offered"));
        assert!(!detail.contains("run: werk backend install vllm"));

        let invalid_python = VllmDiscovery {
            command: None,
            source: "Strix Halo requires explicit ROCm vLLM configuration".to_string(),
            attempts: vec![VllmDiscoveryAttempt {
                label: "WERK_VLLM_PYTHON".to_string(),
                path: Some(PathBuf::from("/tmp/python")),
                exists: true,
                usable: false,
                detail: "torch.version.hip is not set".to_string(),
            }],
        };
        let detail =
            concise_vllm_unavailable_reason_for_platform(&invalid_python, VllmPlatform::StrixHalo);
        assert!(detail.contains("not a verified Strix Halo ROCm runtime"));
        assert!(detail.contains("torch.version.hip is not set"));
    }

    #[test]
    fn rocm_environment_signals_are_normalized_consistently() {
        for value in ["rocm", "ROCM", " hip "] {
            assert!(vllm_rocm_signals(Some(value), None));
        }
        for value in ["1", "true", "YES", " on ", "rocm", "HIP"] {
            assert!(vllm_rocm_signals(None, Some(value)));
        }
        for (accelerator, legacy) in [
            (Some("cuda"), None),
            (Some("1"), None),
            (None, Some("0")),
            (None, Some("false")),
        ] {
            assert!(!vllm_rocm_signals(accelerator, legacy));
        }
    }

    #[test]
    fn vllm_health_timeout_defaults_are_unified_memory_aware_and_overrideable() {
        let linux = vllm_health_timeout_for(VllmPlatform::NativeLinux, None);
        assert_eq!(linux.duration, Duration::from_secs(300));
        assert!(linux.valid);

        let spark = vllm_health_timeout_for(VllmPlatform::DgxSpark, None);
        assert_eq!(spark.duration, Duration::from_secs(900));
        assert!(spark.detail.contains("DGX Spark cold starts"));

        let strix = vllm_health_timeout_for(VllmPlatform::StrixHalo, None);
        assert_eq!(strix.duration, Duration::from_secs(900));
        assert!(strix.detail.contains("Strix Halo ROCm cold starts"));

        let overridden = vllm_health_timeout_for(VllmPlatform::DgxSpark, Some("1800"));
        assert_eq!(overridden.duration, Duration::from_secs(1800));
        assert!(overridden.valid);
        assert!(
            overridden
                .detail
                .contains("WERK_VLLM_HEALTH_TIMEOUT_SECONDS")
        );

        for invalid in ["0", "-1", "not-a-number", ""] {
            let fallback = vllm_health_timeout_for(VllmPlatform::DgxSpark, Some(invalid));
            assert_eq!(fallback.duration, Duration::from_secs(900));
            assert!(!fallback.valid);
            assert!(fallback.detail.contains("positive integer"));
        }
    }

    #[test]
    fn vllm_health_marks_wsl_local_as_best_effort_but_remote_healthy() {
        let local = VllmDiscovery {
            command: Some(VllmCommand::Python(PathBuf::from("/tmp/python"))),
            source: "test".to_string(),
            attempts: Vec::new(),
        };
        let health = vllm_health_for_platform(&local, VllmPlatform::Wsl);
        assert_eq!(health.installed_label, "yes");
        assert_eq!(health.health_label, "best-effort on WSL");
        assert!(!health.healthy);

        let remote = VllmDiscovery {
            command: Some(VllmCommand::Remote {
                host: "127.0.0.1".to_string(),
                port: 8000,
            }),
            source: "test".to_string(),
            attempts: Vec::new(),
        };
        let health = vllm_health_for_platform(&remote, VllmPlatform::Wsl);
        assert_eq!(health.installed_label, "remote");
        assert_eq!(health.health_label, "healthy");
        assert!(health.healthy);
    }

    #[test]
    fn configured_but_unready_remote_is_not_reported_healthy() {
        let remote = VllmDiscovery {
            command: Some(VllmCommand::Remote {
                host: "spark.local".to_string(),
                port: 8000,
            }),
            source: "env".to_string(),
            attempts: vec![VllmDiscoveryAttempt {
                label: "WERK_VLLM_HOST/WERK_VLLM_PORT".to_string(),
                path: None,
                exists: false,
                usable: false,
                detail: "cold starting".to_string(),
            }],
        };
        let health = vllm_health_for_platform(&remote, VllmPlatform::DgxSpark);
        assert_eq!(health.installed_label, "remote");
        assert_eq!(health.health_label, "not ready");
        assert!(!health.healthy);
        assert!(health.detail.contains("will wait"));
    }

    #[test]
    fn remote_model_resolution_waits_through_cold_start() {
        let mut attempts = 0;
        let resolution =
            wait_for_remote_served_model_with(Duration::from_secs(2), Duration::ZERO, |_| {
                attempts += 1;
                if attempts == 1 {
                    bail!("remote endpoint is still loading");
                }
                Ok(RemoteModelResolution {
                    name: "nemotron-spark".to_string(),
                    source: "only /v1/models entry",
                })
            })
            .unwrap();
        assert_eq!(resolution.name, "nemotron-spark");
        assert_eq!(attempts, 2);
    }

    #[test]
    fn remote_model_wait_has_a_hard_bound_when_http_server_stalls() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            if let Ok((_stream, _)) = listener.accept() {
                thread::sleep(Duration::from_secs(1));
            }
        });

        let started = Instant::now();
        let error = wait_for_remote_served_model(
            &format!("http://127.0.0.1:{port}"),
            "nvidia/Nemotron-Super",
            None,
            Duration::from_millis(100),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("remote vLLM model discovery"));
        assert!(
            started.elapsed() < Duration::from_millis(750),
            "stalled HTTP probe exceeded its health deadline: {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn vllm_health_marks_wsl_missing_local_as_best_effort() {
        let discovery = VllmDiscovery {
            command: None,
            source: "missing".to_string(),
            attempts: Vec::new(),
        };
        let health = vllm_health_for_platform(&discovery, VllmPlatform::Wsl);
        assert_eq!(health.installed_label, "no");
        assert_eq!(health.health_label, "best-effort on WSL");
        assert!(!health.healthy);
        assert!(health.detail.contains("Werk will fall back to Candle CUDA"));
    }

    #[test]
    fn wsl_sensitive_vllm_failure_markers_are_detected() {
        assert!(is_wsl_sensitive_vllm_failure("UVA is not available"));
        assert!(is_wsl_sensitive_vllm_failure("pin_memory failed"));
        assert!(is_wsl_sensitive_vllm_failure("CUDA IPC handle failed"));
        assert!(is_wsl_sensitive_vllm_failure("engine-core failed to start"));
        assert!(is_wsl_sensitive_vllm_failure(
            "multiprocessing spawn failed during startup"
        ));
        assert!(!is_wsl_sensitive_vllm_failure("model file not found"));
    }

    #[cfg(unix)]
    #[test]
    fn vllm_rocm_capability_accepts_python_with_hip_stack() {
        let probe = rocm_python_probe_script();
        assert!(probe.contains("torch.cuda.is_available()"));
        assert!(probe.contains("torch.cuda.device_count()"));

        let python = fake_python("rocm-ok", "printf '6.3.0\\n'\nexit 0\n");
        let detail = vllm_rocm_capability(&VllmCommand::Python(python)).unwrap();
        assert!(detail.contains("PyTorch ROCm/HIP runtime detected"));
        assert!(detail.contains("6.3.0"));
    }

    #[cfg(unix)]
    #[test]
    fn vllm_rocm_capability_rejects_python_without_hip_stack() {
        let python = fake_python(
            "rocm-missing",
            "printf 'torch.version.hip is not set\\n' >&2\nexit 1\n",
        );
        let err = vllm_rocm_capability(&VllmCommand::Python(python)).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("not ROCm-capable"));
        assert!(message.contains("ROCm/HIP PyTorch"));

        let python = fake_python(
            "rocm-masked",
            "printf 'ROCm PyTorch reports no visible GPU\\n' >&2\nexit 1\n",
        );
        let (usable, detail) = python_rocm_status(&python);
        assert!(!usable);
        assert!(detail.contains("no visible GPU"));
    }

    #[cfg(unix)]
    #[test]
    fn vllm_cuda_capability_rejects_hip_python_and_rocm_remote() {
        let probe = cuda_python_probe_script();
        assert!(probe.contains("assert not hip"));
        assert!(probe.contains("torch.version.cuda is not set"));

        let python = fake_python(
            "cuda-is-hip",
            "printf 'torch.version.hip is set (6.3); this is a ROCm runtime, not CUDA\\n' >&2\nexit 1\n",
        );
        let error = vllm_cuda_capability_for(&VllmCommand::Python(python), false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not CUDA-capable"));
        assert!(error.contains("ROCm runtime, not CUDA"));

        let remote = VllmCommand::Remote {
            host: "strix-vllm".to_string(),
            port: 8000,
        };
        assert!(vllm_cuda_capability_for(&remote, false).is_ok());
        let error = vllm_cuda_capability_for(&remote, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("explicitly marked ROCm-backed"));
        assert!(error.contains("cannot satisfy the CUDA runtime candidate"));
    }

    #[test]
    fn vllm_backend_constructors_preserve_selected_accelerator() {
        assert_eq!(
            VllmBackend::new(test_store("vllm-cuda-constructor")).accelerator,
            VllmAccelerator::Cuda
        );
        assert_eq!(
            VllmBackend::new_rocm(test_store("vllm-rocm-constructor")).accelerator,
            VllmAccelerator::Rocm
        );
    }

    #[test]
    fn strix_vllm_profile_is_relaxed_only_for_a_confirmed_other_device() {
        assert!(strix_halo_vllm_profile_selected(
            true,
            SelectedRocmDeviceStatus::StrixHalo
        ));
        assert!(strix_halo_vllm_profile_selected(
            true,
            SelectedRocmDeviceStatus::Unknown
        ));
        assert!(!strix_halo_vllm_profile_selected(
            true,
            SelectedRocmDeviceStatus::Other
        ));
    }

    #[test]
    fn vllm_rocm_capability_rejects_plain_executable_discovery() {
        let err = vllm_rocm_capability(&VllmCommand::Executable(PathBuf::from("/usr/bin/vllm")))
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("ROCm capability cannot be verified"));
        assert!(message.contains("WERK_VLLM_PYTHON"));
    }

    #[cfg(unix)]
    #[test]
    fn strix_halo_python_validation_requires_vllm_hip_and_gfx1151() {
        let probe = strix_halo_python_probe_script();
        assert!(probe.contains("get_device_properties(0)"));
        assert!(probe.contains("selected logical ROCm device 0 is not gfx1151"));
        assert!(!probe.contains("for index in range"));
        assert!(!probe.contains("any(\"gfx1151\""));

        let python = fake_python(
            "strix-ok",
            "printf 'vLLM ROCm/HIP 7.2.1; gfx1151; FP16 is the validated Strix Halo precision\\n'\nexit 0\n",
        );
        let (usable, detail) = python_strix_halo_status(&python);
        assert!(usable);
        assert!(detail.contains("ROCm/HIP 7.2.1"));
        assert!(detail.contains("gfx1151"));
        assert!(detail.contains("FP16"));

        let python = fake_python(
            "strix-wrong-arch",
            "printf 'ROCm PyTorch did not report gfx1151: gfx1100\\n' >&2\nexit 1\n",
        );
        let (usable, detail) = python_strix_halo_status(&python);
        assert!(!usable);
        assert!(detail.contains("not a verified Strix Halo ROCm vLLM environment"));
        assert!(detail.contains("gfx1100"));
    }

    #[cfg(unix)]
    fn fake_python(name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "werk1112-vllm-{name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("python");
        fs::write(&path, format!("#!/bin/sh\n{body}")).unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }

    fn test_store(name: &str) -> ModelStore {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "werk1112-vllm-store-{name}-{}-{nanos}",
            std::process::id()
        ));
        ModelStore::resolve(Some(root)).unwrap()
    }

    fn test_manifest(
        id: &str,
        config_path: Option<&str>,
        model_path: Option<&str>,
    ) -> ModelManifest {
        ModelManifest {
            id: id.to_string(),
            source: ModelSource::LocalPath {
                path: "test".to_string(),
            },
            format: ModelFormat::SafeTensors,
            architecture: Some("qwen3".to_string()),
            tokenizer_path: Some("files/tokenizer.json".to_string()),
            config_path: config_path.map(str::to_string),
            model_path: model_path.map(str::to_string),
            backend: "test".to_string(),
            created_unix: 1,
            files: Vec::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        }
    }
}
