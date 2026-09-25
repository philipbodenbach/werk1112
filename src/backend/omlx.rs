//! Werk-owned oMLX workers. Discovery and compatibility checks never load weights.
use crate::openai::OmlxReasoningEffort;
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
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
        chat_completion_body, delta_content, delta_has_reasoning_content, delta_tool_calls,
        ensure_visible_completion, finalize_completion_stats, request_with_bearer,
        send_stream_result, send_text_chunk, send_tool_call_delta, stream_body,
        update_completion_from_event, update_completion_from_message,
    },
};
use crate::{
    capabilities::InferenceTask,
    inference::{TaskReadiness, TaskReadinessStatus},
    model_store::{ModelFormat, ModelManifest, ModelRuntimeIdentity, ModelStore},
    runtime_control::{BackendRuntimeAdapter, ModelResidencyStatus, StaticRuntimeAdapter},
};

const API_OPTIONS: &[&str] = &[
    "response_format",
    "frequency_penalty",
    "presence_penalty",
    "top_k",
    "reasoning_effort",
];

fn apply_omlx_api_options(
    body: &mut Value,
    options: std::collections::BTreeMap<String, Value>,
) -> Result<()> {
    let effort_override = options.get("reasoning_effort").cloned();
    super::openai_transport::apply_api_options(body, options, API_OPTIONS)?;
    // Keep native effort values and types in sync with the template.
    // Only recognized standard levels change the thinking switch; custom
    // strings and numeric efforts retain their model-specific meaning.
    if let Some(effort) = effort_override {
        let thinking = super::openai_transport::reasoning_effort_thinking(&effort);
        if let Some(enabled) = thinking {
            body["chat_template_kwargs"]["enable_thinking"] = json!(enabled);
        }
        if thinking == Some(false) {
            // Native oMLX forwards "none" literally to model templates.
            // Disable thinking without sending that unsupported sentinel.
            body.as_object_mut().unwrap().remove("reasoning_effort");
            body["chat_template_kwargs"]
                .as_object_mut()
                .unwrap()
                .remove("reasoning_effort");
        } else {
            body["chat_template_kwargs"]["reasoning_effort"] = effort;
        }
    }
    Ok(())
}

const PROBE: &str = include_str!("omlx_probe.py");
const SUPERVISOR: &str = include_str!("omlx_supervisor.py");
const EXPERTS: &str = include_str!("omlx_experts.py");
const PERSISTENCE: &str = include_str!("omlx_persistence.py");
const OFFLOAD: &str = include_str!("omlx_offload.py");
const OFFLOAD_RUNTIME: &str = include_str!("omlx_offload_runtime.py");
const TEXT_OFFLOAD: &str = include_str!("omlx_text_offload.py");
const TEXT_DECODE: &str = include_str!("omlx_decode.py");
const GLM_PROFILE: &str = include_str!("omlx_glm_profile.py");
mod experts;
const PROBE_TIMEOUT: Duration = Duration::from_secs(20);
const DEFAULT_HEALTH_TIMEOUT: Duration = Duration::from_secs(900);
const POLL: Duration = Duration::from_millis(100);
const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;
// Request-specific controls may select another worker, but cannot grow an
// unbounded collection or evict a worker still owned by runtime operations.
const MAX_CACHED_WORKERS: usize = 16;
const MAX_CACHED_MODEL_PROBES: usize = 32;
const MAX_PROBE_DEPENDENCIES: usize = 8192;
const MAX_PROBE_JSON_BYTES: usize = 4 * 1024 * 1024;
const MAX_PROBE_STDERR_BYTES: usize = 64 * 1024;

fn console_script_source(source: &str) -> String {
    // `python -c` otherwise imports from the caller's current directory before
    // the probe/supervisor can set the console entry point's import location.
    // sys is built in; no filesystem module is imported until this is fixed.
    format!(
        "import sys\n_werk_launcher_parent = sys.argv.pop(1)\nif not (getattr(sys.flags, 'safe_path', False) or sys.flags.isolated):\n    sys.path[0] = _werk_launcher_parent\ndel _werk_launcher_parent\n{source}"
    )
}

// Keep embedded helpers out of argv: Linux limits each argument to 128 KiB.
// Retain the private file until startup has completed; the bootstrap uses only
// builtins before console_script_source establishes the verified import path.
fn attach_python_script(command: &mut Command, source: &str) -> Result<tempfile::NamedTempFile> {
    let mut script = tempfile::Builder::new()
        .prefix("werk-omlx-")
        .suffix(".py")
        .tempfile()?;
    script.write_all(source.as_bytes())?;
    let path = serde_json::to_string(
        script
            .path()
            .to_str()
            .context("oMLX temporary script path is not UTF-8")?,
    )?;
    command.args([
        "-c",
        &format!("exec(compile(open({path}, encoding='utf-8').read(), '<werk-omlx>', 'exec'))"),
    ]);
    Ok(script)
}

fn worker_script_source(source: &str) -> String {
    // Embed Werk's helper in a private module, never import code from model paths.
    let mut script = String::from("import types\n");
    for (name, helper) in [
        ("_werk_omlx_experts", EXPERTS),
        ("_werk_omlx_offload", OFFLOAD),
        ("_werk_omlx_offload_runtime", OFFLOAD_RUNTIME),
        ("_werk_omlx_text_offload", TEXT_OFFLOAD),
        ("_werk_omlx_persistence", PERSISTENCE),
    ] {
        let literal = serde_json::to_string(helper).expect("Python source is serializable");
        script.push_str(&format!("_werk_helper = types.ModuleType('{name}')\nsys.modules['{name}'] = _werk_helper\nexec({literal}, _werk_helper.__dict__)\n"));
    }
    script.push_str(source);
    console_script_source(&script)
}

fn text_worker_script_source(source: &str, architecture: Option<&str>) -> String {
    if !matches!(architecture, Some("glm5_next" | "qwen4_exp")) {
        return worker_script_source(source);
    }
    let mut script = String::new();
    for (name, helper) in [("_werk_omlx_decode", TEXT_DECODE)].into_iter().chain(
        (architecture == Some("glm5_next")).then_some(("_werk_omlx_glm_profile", GLM_PROFILE)),
    ) {
        let literal = serde_json::to_string(helper).expect("Python source is serializable");
        script.push_str(&format!("_werk_helper = types.ModuleType('{name}')\nsys.modules['{name}'] = _werk_helper\nexec({literal}, _werk_helper.__dict__)\n"));
    }
    script.push_str(source);
    worker_script_source(&script)
}

#[derive(Clone)]
pub struct OmlxBackend {
    store: ModelStore,
    // Snapshot once. The same invocation is used for model preflight and startup.
    invocation: std::result::Result<OmlxInvocation, String>,
    servers: Arc<Mutex<HashMap<String, Arc<OmlxProcess>>>>,
    model_probes: Arc<Mutex<VecDeque<CachedModelProbe>>>,
    // Native template control is request-local; toggling it reuses weights.
    request_thinking: Option<bool>,
    request_reasoning_effort: Option<OmlxReasoningEffort>,
    #[cfg(test)]
    test_probe: Option<ProbeReport>,
}

mod telemetry;

#[derive(Clone)]
struct OmlxInvocation {
    launcher: PathBuf,
    python: PathBuf,
    python_args: Vec<String>,
    health_timeout: Duration,
    environment: Vec<(OsString, OsString)>,
    launcher_fingerprint: u64,
    working_directory: PathBuf,
    expert_cache_bytes: Option<u64>,
    ngram_cache_bytes: Option<u64>,
    expert_execution: &'static str,
    thinking: Option<bool>,
    reasoning_effort: Option<OmlxReasoningEffort>,
    persistence_dir: Option<PathBuf>,
    server_prefix_cache: bool,
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
            .field("expert_cache_bytes", &self.expert_cache_bytes)
            .field("ngram_cache_bytes", &self.ngram_cache_bytes)
            .field("expert_execution", &self.expert_execution)
            .field("thinking", &self.thinking)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("persistence_dir", &self.persistence_dir)
            .field("server_prefix_cache", &self.server_prefix_cache)
            .finish()
    }
}

#[derive(Clone, Debug)]
struct ProbeReport {
    detail: String,
    version: String,
    tools: bool,
    runtime: Value,
    cache_paths: Vec<PathBuf>,
}

#[derive(Clone, PartialEq, Eq)]
struct ProbeFileStamp {
    path: PathBuf,
    size: u64,
    modified: Option<std::time::SystemTime>,
    #[cfg(unix)]
    unix_identity: (u64, u64, i64, i64, u32),
}

impl ProbeFileStamp {
    fn read(path: PathBuf) -> Result<Self> {
        let metadata = fs::metadata(&path)?;
        #[cfg(unix)]
        let unix_identity = {
            use std::os::unix::fs::MetadataExt;
            (
                metadata.dev(),
                metadata.ino(),
                metadata.ctime(),
                metadata.ctime_nsec(),
                metadata.mode(),
            )
        };
        Ok(Self {
            path,
            size: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            unix_identity,
        })
    }

    fn unchanged(&self) -> bool {
        Self::read(self.path.clone()).is_ok_and(|current| current == *self)
    }
}

#[derive(PartialEq, Eq)]
struct ModelProbeKey {
    invocation: String,
    manifest: ModelRuntimeIdentity,
    directory: PathBuf,
    files: Vec<ProbeFileStamp>,
}

impl ModelProbeKey {
    fn read(
        invocation: &OmlxInvocation,
        manifest: &ModelManifest,
        directory: &Path,
    ) -> Result<Self> {
        let mut paths = fs::read_dir(directory)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        // These are the files consumed by omlx_probe.py and its header-only
        // expert preflight. Additions/removals are represented by the list.
        paths.retain(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    matches!(
                        name,
                        "config.json"
                            | "tokenizer_config.json"
                            | "generation_config.json"
                            | "chat_template.jinja"
                            | "chat_templates"
                    ) || (name.starts_with("model") && name.ends_with(".safetensors"))
                })
        });
        paths.sort();
        let files = paths
            .into_iter()
            .map(ProbeFileStamp::read)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            invocation: format!("{invocation:?}"),
            manifest: ModelRuntimeIdentity::from_manifest(manifest)?,
            directory: directory.to_path_buf(),
            files,
        })
    }
}

struct CachedModelProbe {
    key: ModelProbeKey,
    dependencies: Vec<ProbeFileStamp>,
    report: ProbeReport,
}

struct OmlxProcess {
    child: Mutex<Child>,
    // EOF stops the owned worker even if Werk exits via a signal without Drop.
    parent_pipe: Option<ChildStdin>,
    // These flock descriptions also belong to the child. Closing the parent's
    // copies cannot make its cache purgeable until the worker actually exits.
    _lifetime_locks: Vec<fs::File>,
    url: String,
    api_key: String,
    model_name: String,
    model_dir: PathBuf,
    model_identity: Option<ModelRuntimeIdentity>,
    logical_model_id: Option<String>,
    base_path: PathBuf,
    version: String,
    instance_id: String,
    tools: bool,
    log_tail: Arc<Mutex<VecDeque<String>>>,
    expert_offload: bool,
    expert_cache_bytes: Option<u64>,
    ngram_cache_bytes: Option<u64>,
    expert_execution: &'static str,
    thinking: Option<bool>,
    reasoning_effort: Option<OmlxReasoningEffort>,
    server_prefix_cache: bool,
}

struct OmlxChatSession {
    manifest: ModelManifest,
    server: Arc<OmlxProcess>,
    request_thinking: Option<bool>,
    request_reasoning_effort: Option<OmlxReasoningEffort>,
}

impl OmlxBackend {
    pub fn new(store: ModelStore) -> Self {
        Self {
            store,
            invocation: OmlxInvocation::discover().map_err(|error| format!("{error:#}")),
            servers: Arc::new(Mutex::new(HashMap::new())),
            model_probes: Arc::new(Mutex::new(VecDeque::new())),
            request_thinking: None,
            request_reasoning_effort: None,
            #[cfg(test)]
            test_probe: None,
        }
    }

    pub(crate) fn cache_identity(&self) -> String {
        format!("{:?}", self.invocation)
    }

    /// Retain short exact token prefixes across requests in the owned worker.
    /// Configure before `prepare`; this applies to sessions, tools and direct
    /// generation alike. It stores no chat history and ends with the worker.
    pub fn with_server_prefix_cache(mut self, enabled: bool) -> Self {
        if let Ok(invocation) = &mut self.invocation {
            invocation.server_prefix_cache = enabled;
        }
        self
    }

    pub fn probe() -> Result<String> {
        let invocation = OmlxInvocation::discover()?;
        Ok(invocation.probe(None)?.detail)
    }

    pub fn probe_model(&self, manifest: &ModelManifest) -> Result<()> {
        self.model_probe(manifest).map(|_| ())
    }

    pub fn probe_tool_calling(&self, manifest: &ModelManifest) -> Result<bool> {
        self.model_probe(manifest)?;
        Ok(true)
    }

    fn invocation(&self) -> Result<&OmlxInvocation> {
        self.invocation
            .as_ref()
            .map_err(|detail| anyhow!(detail.clone()))
    }

    fn configured_for_chat(&self, options: &crate::openai::ChatRuntimeOptions) -> Result<Self> {
        options.validate()?;
        let mut configured = self.clone();
        let invocation = configured
            .invocation
            .as_mut()
            .map_err(|detail| anyhow!(detail.clone()))?;
        if let Some(omlx) = &options.omlx {
            if let Some(thinking) = omlx.thinking {
                configured.request_thinking = Some(thinking);
            }
            if let Some(effort) = omlx.reasoning_effort {
                configured.request_reasoning_effort = Some(effort);
            }
            if let Some(megabytes) = omlx.expert_cache_mb {
                invocation.expert_cache_bytes =
                    expert_cache_bytes(Some(megabytes.to_string().into()))?;
            }
            if let Some(budget) = omlx.ngram_cache_mb {
                invocation.ngram_cache_bytes = match budget {
                    crate::openai::NgramCacheBudget::Megabytes(mb) => Some(mb * 1024 * 1024),
                    crate::openai::NgramCacheBudget::Mode(_) => None,
                };
            }
        }
        Ok(configured)
    }

    fn matches_chat_configuration(&self, server: &OmlxProcess) -> bool {
        self.invocation().is_ok_and(|invocation| {
            server.thinking == invocation.thinking
                && server.reasoning_effort == invocation.reasoning_effort
                && server.expert_cache_bytes == invocation.expert_cache_bytes
                && server.ngram_cache_bytes == invocation.ngram_cache_bytes
                && server.expert_execution == invocation.expert_execution
                && server.server_prefix_cache == invocation.server_prefix_cache
        })
    }

    fn model_probe(&self, manifest: &ModelManifest) -> Result<(PathBuf, ProbeReport)> {
        let directory = resolve_model_dir(&self.store, manifest)?;
        #[cfg(test)]
        if let Some(report) = &self.test_probe {
            return Ok((directory, report.clone()));
        }
        let invocation = self.invocation()?;
        // Always recheck the launcher and import boundary, including cache hits.
        invocation.verify_launcher()?;
        invocation.verify_import_paths(&directory)?;
        let key = ModelProbeKey::read(invocation, manifest, &directory)?;
        let mut probes = self
            .model_probes
            .lock()
            .map_err(|_| anyhow!("oMLX model probe cache mutex poisoned"))?;
        // Keep only the latest on-disk snapshot for a given model invocation.
        // A later removal must not revive an older inventory for that model.
        probes.retain(|entry| {
            entry.key.manifest != key.manifest
                || entry.key.invocation != key.invocation
                || entry.key.directory != key.directory
                || entry.key.files == key.files
        });
        if let Some(index) = probes.iter().position(|entry| entry.key == key) {
            let cached = probes
                .remove(index)
                .expect("located probe cache entry exists");
            if cached.dependencies.iter().all(ProbeFileStamp::unchanged) {
                let report = cached.report.clone();
                probes.push_back(cached);
                return Ok((directory, report));
            }
        }
        // Serialize misses so concurrent first requests do not all import the
        // same Python runtime. Failures and incomplete inventories are uncached.
        let report = invocation
            .probe(Some(&directory))
            .with_context(|| format!("oMLX model '{}' is not verified compatible", manifest.id))?;
        if !report.cache_paths.is_empty()
            && report.cache_paths.len() <= MAX_PROBE_DEPENDENCIES
            && report.cache_paths.iter().all(|path| path.is_absolute())
            && ModelProbeKey::read(invocation, manifest, &directory)? == key
            && let Ok(dependencies) = report
                .cache_paths
                .iter()
                .cloned()
                .map(ProbeFileStamp::read)
                .collect::<Result<Vec<_>>>()
        {
            probes.push_back(CachedModelProbe {
                key,
                dependencies,
                report: report.clone(),
            });
            while probes.len() > MAX_CACHED_MODEL_PROBES {
                probes.pop_front();
            }
        }
        Ok((directory, report))
    }

    fn cached_server(&self, manifest: &ModelManifest) -> Result<(Arc<OmlxProcess>, f64)> {
        let invocation = self.invocation()?;
        // Recheck metadata against this exact invocation before startup/reuse.
        let (directory, report) = self.model_probe(manifest)?;
        self.cached_server_with(manifest, invocation, directory, report)
    }

    fn cached_server_with(
        &self,
        manifest: &ModelManifest,
        invocation: &OmlxInvocation,
        directory: PathBuf,
        report: ProbeReport,
    ) -> Result<(Arc<OmlxProcess>, f64)> {
        let identity = ModelRuntimeIdentity::from_manifest(manifest)?;
        let key = format!(
            "{invocation:?}|{}|{identity}|{}",
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
        if servers.len() >= MAX_CACHED_WORKERS {
            bail!(
                "oMLX has reached its limit of {MAX_CACHED_WORKERS} retained workers; restart the Werk server before selecting additional model or cache configurations"
            );
        }
        let started = Instant::now();
        let mut server = OmlxProcess::start(&self.store, invocation, &directory, report)?;
        server.model_identity = Some(identity);
        server.logical_model_id = Some(manifest.id.clone());
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
        let native_tools = if request.requires_tool_calling() {
            self.model_probe(manifest)?.1.tools
        } else {
            true
        };
        generate_with_tool_fallback(manifest, request, tx, native_tools, |request, tx| {
            let started = Instant::now();
            let (server, load_seconds) = self.cached_server(manifest)?;
            let mut response = server.generate_with_controls(
                &request,
                tx,
                self.request_thinking,
                self.request_reasoning_effort,
            )?;
            response.timings.load_seconds = load_seconds;
            response.timings.total_seconds = started.elapsed().as_secs_f64();
            Ok(response)
        })
    }
}

impl GenerationBackend for OmlxBackend {
    fn telemetry(&self) -> Vec<crate::observability::BackendSnapshot> {
        telemetry::sample(self)
    }
    fn validate_api_options(
        &self,
        _manifest: &ModelManifest,
        request: &GenerateRequest,
        options: &std::collections::BTreeMap<String, Value>,
    ) -> Result<()> {
        super::openai_transport::validate_api_options(options, API_OPTIONS)?;
        reject_images(request)
    }
    fn generate_api(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        options: std::collections::BTreeMap<String, Value>,
        tx: Option<mpsc::Sender<Result<Value, String>>>,
    ) -> Result<Value> {
        self.validate_api_options(manifest, &request, &options)?;
        let (request, policy) = self.prepare_tool_request(manifest, request)?;
        let (server, _) = self.cached_server(manifest)?;
        let mut body = omlx_chat_completion_body(
            &server.model_name,
            &request,
            tx.is_some() && policy.is_none(),
            self.request_thinking.or(server.thinking),
            self.request_reasoning_effort.or(server.reasoning_effort),
        );
        apply_omlx_api_options(&mut body, options)?;
        if let Some(policy) = policy {
            let mut value = super::openai_transport::generate_api(
                &server.url,
                Some(&server.api_key),
                body,
                None,
            )?;
            apply_generic_api_tools(&mut value, &policy)?;
            if let Some(tx) = tx {
                value["object"] = json!("chat.completion.chunk");
                for choice in value["choices"].as_array_mut().into_iter().flatten() {
                    let choice = choice
                        .as_object_mut()
                        .context("oMLX response choice must be an object")?;
                    let mut delta = choice
                        .remove("message")
                        .context("oMLX response choice has no message")?;
                    if let Some(calls) = delta["tool_calls"].as_array_mut() {
                        for (index, call) in calls.iter_mut().enumerate() {
                            call["index"] = json!(index);
                        }
                    }
                    choice.insert("delta".into(), delta);
                }
                tx.blocking_send(Ok(value))
                    .map_err(|_| anyhow!("stream receiver closed"))?;
                return Ok(Value::Null);
            }
            Ok(value)
        } else {
            super::openai_transport::generate_api(&server.url, Some(&server.api_key), body, tx)
        }
    }
    fn count_tokens(&self, manifest: &ModelManifest, request: GenerateRequest) -> Result<usize> {
        self.count_api_tokens(manifest, request, Default::default())
    }
    fn count_api_tokens(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        options: std::collections::BTreeMap<String, Value>,
    ) -> Result<usize> {
        self.validate_api_options(manifest, &request, &options)?;
        reject_images(&request)?;
        let (request, _) = self.prepare_tool_request(manifest, request)?;
        let (server, _) = self.cached_server(manifest)?;
        let mut body = omlx_chat_completion_body(
            &server.model_name,
            &request,
            false,
            self.request_thinking.or(server.thinking),
            self.request_reasoning_effort.or(server.reasoning_effort),
        );
        apply_omlx_api_options(&mut body, options)?;
        let value = server.json_request(
            "POST",
            "/werk/tokenize",
            Some(&body),
            Duration::from_secs(60),
        )?;
        value
            .get("input_tokens")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .context("oMLX tokenizer returned no valid input_tokens")
    }
    fn with_chat_options(
        &self,
        manifest: &ModelManifest,
        options: &crate::openai::ChatRuntimeOptions,
    ) -> Result<Arc<dyn GenerationBackend>> {
        let configured = self.configured_for_chat(options)?;
        // Inspect the exact configured runtime and offload loader, without
        // loading weights or mutating the base backend's captured settings.
        configured.probe_model(manifest)?;
        Ok(Arc::new(configured))
    }

    fn supports_tool_calling(&self, _manifest: &ModelManifest, _has_images: bool) -> bool {
        true
    }

    fn runtime_control_adapter(&self) -> Arc<dyn BackendRuntimeAdapter> {
        let server = self.servers.lock().ok().and_then(|servers| {
            let mut active = servers
                .values()
                .filter(|server| server.is_running() && self.matches_chat_configuration(server));
            let first = active.next().cloned();
            if active.next().is_some() { None } else { first }
        });
        Arc::new(experts::OmlxRuntimeAdapter::new(server, self.store.clone()))
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
        let mut matching = servers.values().filter(|server| {
            server.model_dir == directory
                && server.model_identity.as_ref() == Some(&identity)
                && self.matches_chat_configuration(server)
                && server.is_running()
        });
        let first = matching.next().cloned();
        let server = if matching.next().is_some() {
            None
        } else {
            first
        };
        Ok(Arc::new(experts::OmlxRuntimeAdapter::new(
            server,
            self.store.clone(),
        )))
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
        Ok(Some(Box::new(OmlxChatSession {
            manifest: manifest.clone(),
            server,
            request_thinking: self.request_thinking,
            request_reasoning_effort: self.request_reasoning_effort,
        })))
    }

    fn start_persistent_chat_session(
        &self,
        manifest: &ModelManifest,
        _seed: Option<u64>,
        cache_directory: &Path,
    ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
        let (directory, report) = self.model_probe(manifest)?;
        // The ordinary path remains available on all previously supported
        // runtimes. Only the verified native-cache extension is version gated.
        if report.version != "0.6.4" {
            return Ok(None);
        }
        let mut invocation = self.invocation()?.clone();
        invocation.persistence_dir = Some(persistent_cache_directory(
            cache_directory,
            manifest,
            &invocation,
            &report,
        )?);
        let (server, _) = self.cached_server_with(manifest, &invocation, directory, report)?;
        let status = server.json_request(
            "GET",
            "/werk/persistence/status",
            None,
            Duration::from_secs(5),
        );
        let active =
            status.and_then(|value| verified_persistent_cache_status(&value, &server.model_name));
        match active {
            Ok(true) => Ok(Some(Box::new(OmlxChatSession {
                manifest: manifest.clone(),
                server,
                request_thinking: self.request_thinking,
                request_reasoning_effort: self.request_reasoning_effort,
            }))),
            inactive => {
                // An unsupported model cache must not leave a second large
                // model alive when the caller resumes its ordinary chat path.
                self.servers
                    .lock()
                    .map_err(|_| anyhow!("oMLX worker registry is poisoned"))?
                    .retain(|_, candidate| !Arc::ptr_eq(candidate, &server));
                drop(server);
                inactive.map(|_| None)
            }
        }
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
        generate_with_tool_fallback(
            &self.manifest,
            request,
            None,
            self.server.tools,
            |request, tx| {
                self.server.generate_with_controls(
                    &request,
                    tx,
                    self.request_thinking,
                    self.request_reasoning_effort,
                )
            },
        )
    }

    fn generate_stream(&self, request: GenerateRequest) -> GenerateStream {
        let server = self.server.clone();
        let manifest = self.manifest.clone();
        let thinking = self.request_thinking;
        let reasoning_effort = self.request_reasoning_effort;
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            let result = generate_with_tool_fallback(
                &manifest,
                request,
                Some(tx.clone()),
                server.tools,
                |request, tx| {
                    server.generate_with_controls(&request, tx, thinking, reasoning_effort)
                },
            );
            send_stream_result(tx, result);
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

impl OmlxBackend {
    fn prepare_tool_request(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<(GenerateRequest, Option<super::tool_calling::Policy>)> {
        if request.requires_tool_calling()
            && (!self.model_probe(manifest)?.1.tools || validate_tool_options(&request).is_err())
        {
            let (request, policy) = super::tool_calling::prepare(manifest, request)?;
            Ok((request, Some(policy)))
        } else {
            Ok((request, None))
        }
    }
}

fn generate_with_tool_fallback(
    manifest: &ModelManifest,
    request: GenerateRequest,
    tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    native_tools: bool,
    run: impl FnOnce(
        GenerateRequest,
        Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<GenerateResponse>,
) -> Result<GenerateResponse> {
    if !request.requires_tool_calling() || (native_tools && validate_tool_options(&request).is_ok())
    {
        return run(request, tx);
    }
    // Validate the complete fallback output before emitting executable deltas.
    let response = super::tool_calling::generate(manifest, request, |request| run(request, None))?;
    if !response.text.is_empty() {
        send_text_chunk(&tx, response.text.clone())?;
    }
    if let Some(calls) = response
        .assistant_message
        .as_ref()
        .and_then(|message| message.tool_calls.as_ref())
    {
        let deltas = calls
            .iter()
            .enumerate()
            .map(|(index, call)| crate::openai::ChatCompletionToolCallDelta {
                index,
                id: Some(call.id.clone()),
                kind: Some(call.kind.clone()),
                function: Some(crate::openai::ChatCompletionFunctionCallDelta {
                    name: Some(call.function.name.clone()),
                    arguments: Some(call.function.arguments.clone()),
                }),
            })
            .collect();
        send_tool_call_delta(&tx, deltas)?;
    }
    Ok(response)
}

fn apply_generic_api_tools(value: &mut Value, policy: &super::tool_calling::Policy) -> Result<()> {
    let choices = value["choices"]
        .as_array_mut()
        .context("oMLX response has no choices")?;
    for choice in choices {
        let finish_reason = choice["finish_reason"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        let message = choice
            .get_mut("message")
            .context("oMLX response choice has no message")?;
        anyhow::ensure!(
            message.is_object(),
            "oMLX response message must be an object"
        );
        anyhow::ensure!(
            message["tool_calls"].as_array().is_none_or(Vec::is_empty),
            "generic tool protocol received unexpected native tool calls"
        );
        let parsed = policy.parse(message["content"].as_str().unwrap_or_default())?;
        super::tool_calling::validate_finish_reason(&parsed, &finish_reason)?;
        message["content"] = serde_json::to_value(parsed.content)?;
        if let Some(calls) = parsed.tool_calls {
            message["tool_calls"] = serde_json::to_value(calls)?;
            choice["finish_reason"] = json!("tool_calls");
        }
    }
    Ok(())
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
    let root = store.model_location(manifest);
    let mut candidates = Vec::new();
    if let Some(config) = &manifest.config_path {
        let relative = Path::new(config);
        if relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
        {
            bail!("oMLX manifest config_path must stay within the model directory");
        }
        if let Some(parent) = store.absolute_model_file(manifest, config).parent() {
            candidates.push(parent.to_path_buf());
        }
    }
    candidates.push(store.model_files_dir(manifest));
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
        let expert_cache_bytes = expert_cache_bytes(env::var_os("WERK_OMLX_EXPERT_CACHE_MB"))?;
        let ngram_cache_bytes = ngram_cache_bytes(env::var_os("WERK_OMLX_NGRAM_CACHE_MB"))?;
        let expert_execution = expert_execution(env::var_os("WERK_OMLX_EXPERT_EXECUTION"))?;
        let mut environment: Vec<_> = env::vars_os()
            .filter(|(name, _)| !name.to_string_lossy().starts_with("OMLX_"))
            .collect();
        environment.sort();
        let thinking = thinking_enabled(
            environment
                .iter()
                .find(|(name, _)| name == "WERK_OMLX_THINKING")
                .map(|(_, value)| value.clone()),
        )?;
        let reasoning_effort = reasoning_effort_enabled(
            environment
                .iter()
                .find(|(name, _)| name == "WERK_OMLX_REASONING_EFFORT")
                .map(|(_, value)| value.clone()),
        )?;
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
            expert_cache_bytes,
            ngram_cache_bytes,
            expert_execution,
            thinking,
            reasoning_effort,
            persistence_dir: None,
            server_prefix_cache: false,
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
        let _script = attach_python_script(&mut command, &worker_script_source(PROBE))?;
        command
            .arg(
                self.launcher
                    .parent()
                    .context("oMLX launcher has no parent directory")?,
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let payload = json!({"model_dir": model_dir, "launcher": self.launcher,
            "expert_cache_bytes": self.expert_cache_bytes,
            "ngram_cache_bytes": self.ngram_cache_bytes});
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
        let out = thread::spawn(move || read_bounded(stdout, MAX_PROBE_JSON_BYTES));
        let err = thread::spawn(move || read_bounded(stderr, MAX_PROBE_STDERR_BYTES));
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
            runtime,
            cache_paths: value
                .get("cache_paths")
                .and_then(Value::as_array)
                .filter(|paths| paths.len() <= MAX_PROBE_DEPENDENCIES)
                .and_then(|paths| {
                    paths
                        .iter()
                        .map(|path| path.as_str().map(PathBuf::from))
                        .collect::<Option<Vec<_>>>()
                })
                .unwrap_or_default(),
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

fn expert_cache_bytes(value: Option<OsString>) -> Result<Option<u64>> {
    // Native loading is the default. Some(0) is explicit auto; public 0 disables offload.
    let Some(value) = value else {
        return Ok(None);
    };
    if value == "auto" {
        return Ok(Some(0));
    }
    let mb = value.to_str().and_then(|value| value.parse::<u64>().ok())
        .filter(|mb| *mb <= u64::MAX / (1024 * 1024))
        .context("WERK_OMLX_EXPERT_CACHE_MB must be auto or a nonnegative MiB integer that fits in u64 bytes (0 disables expert offload)")?;
    Ok((mb > 0).then_some(mb * 1024 * 1024))
}

fn reasoning_effort_enabled(value: Option<OsString>) -> Result<Option<OmlxReasoningEffort>> {
    value.map(|value| {
        let text = value.to_str().context("WERK_OMLX_REASONING_EFFORT is not valid UTF-8")?;
        match text {
            "low" => Ok(OmlxReasoningEffort::Low),
            "high" => Ok(OmlxReasoningEffort::High),
            "max" => Ok(OmlxReasoningEffort::Max),
            _ => bail!("WERK_OMLX_REASONING_EFFORT must be low, high or max; unset it to inherit the model default"),
        }
    }).transpose()
}

fn thinking_enabled(value: Option<OsString>) -> Result<Option<bool>> {
    match value.as_deref().and_then(|value| value.to_str()) {
        None if value.is_none() => Ok(None),
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        _ => bail!(
            "WERK_OMLX_THINKING must be 0 (disabled) or 1 (enabled); unset it to preserve the model's default"
        ),
    }
}

fn ngram_cache_bytes(value: Option<OsString>) -> Result<Option<u64>> {
    // Resident tables are the default; None represents explicitly requested auto.
    let Some(value) = value else {
        return Ok(Some(0));
    };
    if value == "auto" {
        return Ok(None);
    }
    let mb = value.to_str().and_then(|value| value.parse::<u64>().ok())
        .filter(|mb| *mb <= u64::MAX / (1024 * 1024))
        .context("WERK_OMLX_NGRAM_CACHE_MB must be auto or a nonnegative MiB integer (0 selects resident tables)")?;
    Ok(Some(mb * 1024 * 1024))
}

fn expert_execution(value: Option<OsString>) -> Result<&'static str> {
    match value.as_deref().and_then(|value| value.to_str()) {
        None if value.is_none() => Ok("grouped"),
        Some("grouped") => Ok("grouped"),
        Some("serial") => Ok("serial"),
        _ => bail!("WERK_OMLX_EXPERT_EXECUTION must be grouped or serial"),
    }
}

fn expert_interval_diagnostics(before: &Value, after: &Value) -> Option<String> {
    // These counters belong to the entire worker. Concurrent requests may
    // contribute; never label their differences as request-exclusive timings.
    let mut interval = serde_json::Map::new();
    for name in [
        "cache_hits",
        "cache_misses",
        "disk_bytes_read",
        "cache_evictions",
        "allocator_clears",
        "tensor_materializations",
        "output_evaluations",
        "forward_calls",
        "budget_reductions",
        "prefill_cache_growth_bytes",
        "prefill_samples_corrected",
        "ngram_requested_rows",
        "ngram_unique_rows",
        "ngram_cache_hits",
        "ngram_cache_misses",
        "ngram_cache_evictions",
    ] {
        if let (Some(a), Some(b)) = (
            before.get(name).and_then(Value::as_u64),
            after.get(name).and_then(Value::as_u64),
        ) {
            interval.insert(name.into(), json!(b.checked_sub(a)?));
        }
    }
    for name in [
        "disk_read_seconds",
        "materialize_seconds",
        "forward_seconds",
        "routing_seconds",
    ] {
        if let (Some(a), Some(b)) = (
            before.get(name).and_then(Value::as_f64),
            after.get(name).and_then(Value::as_f64),
        ) {
            if !a.is_finite() || !b.is_finite() || a < 0.0 || b < a {
                return None;
            }
            interval.insert(
                name.into(),
                json!(((b - a) * 1_000_000.0).round() / 1_000_000.0),
            );
        }
    }
    if !interval.contains_key("cache_hits") || !interval.contains_key("cache_misses") {
        return None;
    }
    for name in [
        "resident_cache_bytes",
        "cache_budget_bytes",
        "effective_cache_budget_bytes",
        "allocator_cache_bytes",
        "ngram_resident_cache_bytes",
        "ngram_cache_budget_bytes",
        "ngram_effective_cache_budget_bytes",
    ] {
        if let Some(value) = after.get(name).and_then(Value::as_u64) {
            interval.insert(name.into(), json!(value));
        }
    }
    if let Some(mode @ ("auto" | "explicit")) =
        after.get("cache_budget_mode").and_then(Value::as_str)
    {
        interval.insert("cache_budget_mode".into(), json!(mode));
    }
    if let Some(mode @ ("serial" | "grouped")) = after.get("execution").and_then(Value::as_str) {
        interval.insert("execution".into(), json!(mode));
    }
    if let Some(mode @ ("auto" | "explicit" | "disabled")) =
        after.get("ngram_cache_budget_mode").and_then(Value::as_str)
    {
        interval.insert("ngram_cache_budget_mode".into(), json!(mode));
    }
    Some(format!(
        "oMLX experts (worker interval; reads include OS file cache): {}",
        Value::Object(interval)
    ))
}

fn glm_profile_diagnostics(status: &Value) -> Option<String> {
    let profile = status.get("glm_layer_profile")?;
    if !profile.get("enabled")?.as_bool()? {
        return None;
    }
    let layers = profile.get("layers")?.as_array()?;
    if layers.len() > 1024 {
        return None;
    }
    let rows: Option<Vec<Value>> = layers
        .iter()
        .map(|layer| {
            let mut row = serde_json::Map::new();
            row.insert("layer".into(), json!(layer.get("layer")?.as_u64()?));
            for name in [
                "calls",
                "failures",
                "cache_hits",
                "cache_misses",
                "cache_evictions",
                "disk_bytes_read",
            ] {
                row.insert(name.into(), json!(layer.get(name)?.as_u64()?));
            }
            for name in [
                "wall_seconds",
                "forward_seconds",
                "routing_seconds",
                "disk_read_seconds",
            ] {
                let seconds = layer.get(name)?.as_f64()?;
                if !seconds.is_finite() || seconds < 0.0 {
                    return None;
                }
                row.insert(name.into(), json!(seconds));
            }
            Some(Value::Object(row))
        })
        .collect();
    Some(format!(
        "oMLX GLM layer profile (worker cumulative; overlapping timings; reads include OS file cache): {}",
        json!(rows?)
    ))
}

fn omlx_chat_completion_body(
    model_name: &str,
    request: &GenerateRequest,
    stream: bool,
    thinking: Option<bool>,
    reasoning_effort: Option<OmlxReasoningEffort>,
) -> Value {
    let mut body = chat_completion_body(model_name, request, stream);
    if let Some(enabled) = thinking {
        body["chat_template_kwargs"] = json!({"enable_thinking": enabled});
    }
    if let Some(effort) = reasoning_effort {
        body["reasoning_effort"] = json!(effort);
        body["chat_template_kwargs"]["reasoning_effort"] = json!(effort);
    }
    body
}

fn persistent_cache_directory(
    root: &Path,
    manifest: &ModelManifest,
    invocation: &OmlxInvocation,
    report: &ProbeReport,
) -> Result<PathBuf> {
    let metadata =
        fs::symlink_metadata(root).context("persistent chat cache directory is missing")?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("persistent chat cache path must be a regular directory");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.uid() != unsafe { libc::geteuid() } {
            bail!("persistent chat cache directory is not owned by the current user");
        }
    }
    // Runtime and model changes select another namespace. Unrelated captured
    // shell environment changes do not invalidate the cache on every restart.
    let import_environment = invocation
        .environment
        .iter()
        .filter(|(name, _)| name == "PYTHONPATH" || name == "PYTHONHOME")
        .map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                value.to_string_lossy().into_owned(),
            )
        })
        .collect::<Vec<_>>();
    let namespace = json!({
        "format": "werk-omlx-native-chat-cache-v1",
        "adapter": env!("CARGO_PKG_VERSION"),
        "model": ModelRuntimeIdentity::from_manifest(manifest)?.to_string(),
        "runtime": report.runtime,
        "launcher": invocation.launcher,
        "launcher_fingerprint": invocation.launcher_fingerprint,
        "python": invocation.python,
        "python_args": invocation.python_args,
        "import_environment": import_environment,
        "import_working_directory": (!import_environment.is_empty()).then_some(&invocation.working_directory),
        "thinking": invocation.thinking,
        "reasoning_effort": invocation.reasoning_effort,
        "expert_cache_bytes": invocation.expert_cache_bytes,
        "ngram_cache_bytes": invocation.ngram_cache_bytes,
        "expert_execution": invocation.expert_execution,
    });
    let namespace = format!("omlx-{:x}", Sha256::digest(serde_json::to_vec(&namespace)?));
    let directory = root.canonicalize()?.join(namespace);
    match fs::symlink_metadata(&directory) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            bail!("persistent oMLX cache namespace is not a regular directory");
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(&directory).context("cannot create persistent oMLX cache namespace")?;
        }
        Err(error) => return Err(error).context("cannot inspect persistent oMLX cache namespace"),
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if fs::metadata(&directory)?.uid() != unsafe { libc::geteuid() } {
            bail!("persistent oMLX cache namespace is not owned by the current user");
        }
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;
    }
    Ok(directory)
}

fn verified_persistent_cache_status(value: &Value, physical_model: &str) -> Result<bool> {
    if value.get("installed").and_then(Value::as_bool) != Some(true)
        || value.get("format").and_then(Value::as_str) != Some("omlx-exact-prefix-v1")
    {
        bail!("oMLX did not confirm the requested native persistent chat cache adapter");
    }
    match value.get("active").and_then(Value::as_bool) {
        Some(false) => Ok(false),
        Some(true) if value.get("model_id").and_then(Value::as_str) == Some(physical_model) => {
            Ok(true)
        }
        _ => bail!("oMLX persistent chat cache does not identify the selected loaded model"),
    }
}

#[derive(Default)]
struct OmlxUsageTimings {
    first_token: Option<f64>,
    prompt: Option<f64>,
    decode: Option<f64>,
    total: Option<f64>,
    cached_prompt_tokens: Option<u64>,
}

fn update_omlx_completion_from_event(
    completion: &mut OpenAiCompletion,
    timings: &mut OmlxUsageTimings,
    value: &Value,
    observed_seconds: Option<f64>,
) {
    update_completion_from_event(completion, value);
    // oMLX's Usage extension reports durations in seconds, including reasoning
    // generation. Ignore malformed fields without discarding earlier metadata.
    if let Some(usage) = value.get("usage") {
        let seconds = |key: &str| {
            usage
                .get(key)
                .and_then(Value::as_f64)
                .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
        };
        timings.first_token = seconds("time_to_first_token").or(timings.first_token);
        timings.prompt = seconds("prompt_eval_duration").or(timings.prompt);
        timings.decode = seconds("generation_duration").or(timings.decode);
        timings.total = seconds("total_time").or(timings.total);
        timings.cached_prompt_tokens = usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64)
            .or(timings.cached_prompt_tokens);
    }
    // Hidden reasoning is already token generation, not prompt evaluation.
    if completion.first_token_seconds <= 0.0
        && delta_has_reasoning_content(value)
        && let Some(seconds) = observed_seconds
    {
        completion.first_token_seconds = seconds;
    }
}

fn finalize_omlx_completion_stats(
    completion: &mut OpenAiCompletion,
    request: &GenerateRequest,
    elapsed_seconds: f64,
    timings: &OmlxUsageTimings,
) -> Vec<String> {
    let first_token = timings.first_token.or_else(|| {
        (completion.first_token_seconds > 0.0).then_some(completion.first_token_seconds)
    });
    let total = timings.total.unwrap_or(elapsed_seconds);
    let decode = timings.decode.unwrap_or_else(|| match first_token {
        Some(first) if total > first => total - first,
        _ => total,
    });
    // Keep the shared token-count fallback, then replace its timing estimates.
    finalize_completion_stats(completion, request, elapsed_seconds);
    completion.first_token_seconds = first_token.unwrap_or(0.0);
    completion.prompt_seconds = timings.prompt.or(first_token).unwrap_or(f64::NAN);
    completion.decode_seconds = decode;
    let mut diagnostics = if timings.decode.is_none() && first_token.is_none() {
        vec!["oMLX timing: separate phase durations unavailable; eval duration includes prompt processing".into()]
    } else {
        Vec::new()
    };
    if let Some(tokens) = timings.cached_prompt_tokens {
        diagnostics.push(format!("oMLX cached prompt tokens: {tokens}"));
    }
    diagnostics
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

fn read_bounded(mut reader: impl Read, limit: usize) -> Vec<u8> {
    let mut captured = Vec::new();
    let mut bytes = [0; 4096];
    while let Ok(count) = reader.read(&mut bytes) {
        if count == 0 {
            break;
        }
        let keep = count.min(limit.saturating_sub(captured.len()));
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
        // Auto is architecture/version gated by the metadata probe; unrelated
        // models keep the native loader. Explicit budgets remain strict.
        let expert_cache = invocation
            .expert_cache_bytes
            .filter(|bytes| *bytes > 0 || report.runtime.get("expert_offload").is_some());
        let native_weight_adapter = report
            .runtime
            .get("expert_offload")
            .and_then(|entry| entry.get("loader"))
            .and_then(Value::as_str)
            == Some("installed_native_text_port");
        let mut lifetime_locks = Vec::new();
        if let Some(directory) = &invocation.persistence_dir {
            // persistent_cache_directory creates exactly one runtime namespace
            // under the chat's native-cache root, whose archive lock is held by
            // the caller for the entire session.
            let cache_root = directory
                .parent()
                .context("persistent oMLX cache has no root")?;
            if let Some(lock) = crate::cache::workers::lock_persistent_worker_cache(cache_root)? {
                lifetime_locks.push(lock);
            }
        }
        let (base_path, lifetime_lock) =
            crate::cache::workers::prepare_worker(store.home(), &instance_id)?;
        lifetime_locks.extend(lifetime_lock);
        // Server requests share a worker, not a conversation archive. Keeping
        // this cache inside its private base also gives each model/configuration
        // its own writer and the existing inherited cache-purge lifetime lock.
        let server_cache_directory = (invocation.server_prefix_cache
            && report.version == "0.6.4"
            && invocation.persistence_dir.is_none())
        .then(|| base_path.join("cache").join("prefix-cache"));
        let persistence_directory = invocation
            .persistence_dir
            .as_ref()
            .or(server_cache_directory.as_ref());
        let architecture = report
            .runtime
            .get("expert_offload")
            .and_then(|offload| offload.get("architecture"))
            .and_then(Value::as_str);
        let worker_source = text_worker_script_source(SUPERVISOR, architecture);
        let mut command = invocation.python_command();
        let _script = attach_python_script(&mut command, &worker_source)?;
        command
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
        command
            .env_remove("WERK_OMLX_EXPERT_MODEL_DIR")
            .env_remove("WERK_OMLX_EXPERT_CACHE_BYTES")
            .env_remove("WERK_OMLX_NGRAM_CACHE_BYTES")
            .env_remove("WERK_OMLX_EXPERT_EXECUTION")
            .env_remove("WERK_OMLX_PERSISTENCE_DIR")
            .env_remove("WERK_OMLX_PERSISTENCE_MODEL_DIR");
        if let Some(bytes) = expert_cache {
            command
                .env("WERK_OMLX_EXPERT_MODEL_DIR", model_dir)
                .env("WERK_OMLX_EXPERT_EXECUTION", invocation.expert_execution)
                .env("WERK_OMLX_EXPERT_CACHE_BYTES", bytes.to_string());
        } else if native_weight_adapter {
            command
                .env("WERK_OMLX_EXPERT_MODEL_DIR", model_dir)
                .env("WERK_OMLX_EXPERT_CACHE_BYTES", "native");
        }
        if let Some(bytes) = invocation.ngram_cache_bytes {
            command.env("WERK_OMLX_NGRAM_CACHE_BYTES", bytes.to_string());
        }
        if let Some(directory) = persistence_directory {
            command
                .args(["--paged-ssd-cache-dir"])
                .arg(directory)
                .args(["--paged-ssd-cache-max-size", "4GB"])
                .env("WERK_OMLX_PERSISTENCE_DIR", directory)
                .env("WERK_OMLX_PERSISTENCE_MODEL_DIR", model_dir);
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        crate::cache::workers::inherit_lifetime_locks(&mut command, &lifetime_locks)?;
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
            _lifetime_locks: lifetime_locks,
            url: format!("http://127.0.0.1:{port}"),
            api_key,
            model_name: String::new(),
            model_dir: model_dir.to_path_buf(),
            model_identity: None,
            logical_model_id: None,
            base_path,
            version: report.version,
            instance_id,
            tools: report.tools,
            log_tail,
            expert_offload: false,
            expert_cache_bytes: invocation.expert_cache_bytes,
            ngram_cache_bytes: invocation.ngram_cache_bytes,
            expert_execution: invocation.expert_execution,
            thinking: invocation.thinking,
            reasoning_effort: invocation.reasoning_effort,
            server_prefix_cache: invocation.server_prefix_cache,
        };
        process
            .wait_and_load(deadline)
            .with_context(|| format!("oMLX startup failed{}", process.formatted_log_tail()))?;
        if let Some(requested_bytes) = expert_cache {
            let status =
                process.json_request("GET", "/werk/experts/status", None, deadline.remaining()?)?;
            let actual = status.get("cache_budget_bytes").and_then(Value::as_u64);
            let native_experts = native_weight_adapter
                && status.get("experts_offloaded").and_then(Value::as_bool) == Some(false)
                && actual == Some(0);
            let confirmed = if requested_bytes == 0 {
                (actual.is_some_and(|bytes| bytes > 0) || native_experts)
                    && status.get("cache_budget_mode").and_then(Value::as_str) == Some("auto")
            } else {
                actual == Some(requested_bytes)
            };
            if status.get("active").and_then(Value::as_bool) != Some(true) || !confirmed {
                bail!("oMLX did not confirm activating the requested bounded expert cache");
            }
            process.expert_offload = !native_experts;
            if native_experts {
                eprintln!(
                    "oMLX experts: native resident execution (auto; weights fit available memory)"
                );
            } else {
                eprintln!(
                    "oMLX expert cache: {} MiB ({}) upper budget; SSD offload active, native memory guard may reduce residency",
                    actual.unwrap_or_default() / (1024 * 1024),
                    if requested_bytes == 0 {
                        "auto"
                    } else {
                        "explicit"
                    }
                );
            }
        }
        if native_weight_adapter {
            let status =
                process.json_request("GET", "/werk/experts/status", None, deadline.remaining()?)?;
            if status.get("active").and_then(Value::as_bool) != Some(true) {
                bail!("native text offload worker did not confirm the loaded model");
            }
            if let Some(requested) = invocation.ngram_cache_bytes {
                let mode = status.get("ngram_offload").and_then(Value::as_str);
                let actual = status
                    .get("ngram_cache_budget_bytes")
                    .and_then(Value::as_u64);
                let capacity = status
                    .get("ngram_maximum_cache_bytes")
                    .and_then(Value::as_u64);
                let confirmed = if requested == 0 {
                    actual == Some(0) && matches!(mode, Some("disabled" | "not_applicable"))
                } else {
                    mode == Some("supported")
                        && capacity.is_some_and(|bytes| actual == Some(bytes.min(requested)))
                };
                if !confirmed {
                    bail!("oMLX worker did not confirm the requested N-gram cache setting");
                }
            }
            if status.get("ngram_offload").and_then(Value::as_str) == Some("supported") {
                eprintln!(
                    "oMLX N-gram cache: {} MiB initial, {} MiB upper budget ({}); shares memory with the expert cache",
                    status
                        .get("ngram_initial_cache_bytes")
                        .and_then(Value::as_u64)
                        .unwrap_or_default()
                        / (1024 * 1024),
                    status
                        .get("ngram_cache_budget_bytes")
                        .and_then(Value::as_u64)
                        .unwrap_or_default()
                        / (1024 * 1024),
                    status
                        .get("ngram_cache_budget_mode")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown"),
                );
            }
            process.expert_offload =
                status.get("experts_offloaded").and_then(Value::as_bool) == Some(true);
        }
        // Persistent CLI sessions verify their own cache after startup. The
        // server-only directory is intentionally absent for those sessions.
        if invocation.server_prefix_cache && invocation.persistence_dir.is_none() {
            let active = if server_cache_directory.is_some() {
                process
                    .json_request(
                        "GET",
                        "/werk/persistence/status",
                        None,
                        Duration::from_secs(5),
                    )
                    .and_then(|status| {
                        verified_persistent_cache_status(&status, &process.model_name)
                    })
            } else {
                Ok(false)
            };
            // Unsupported model cache trees keep this already loaded worker;
            // starting a fallback worker would duplicate its model weights.
            match active {
                Ok(true) => eprintln!(
                    "oMLX native short-prefix cache active for {}; worker lifetime, SSD limit 4GB",
                    model_dir.display()
                ),
                Ok(false) => eprintln!(
                    "oMLX native short-prefix cache unavailable for this runtime/model; ordinary prefix caching remains active"
                ),
                Err(error) => eprintln!(
                    "oMLX native short-prefix cache could not be verified ({error:#}); continuing with the loaded worker"
                ),
            }
        }
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

    #[cfg(test)]
    fn generate(
        &self,
        request: &GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<GenerateResponse> {
        self.generate_with_controls(request, tx, None, None)
    }

    fn generate_with_controls(
        &self,
        request: &GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
        thinking: Option<bool>,
        reasoning_effort: Option<OmlxReasoningEffort>,
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
        let body = omlx_chat_completion_body(
            &self.model_name,
            request,
            tx.is_some(),
            thinking.or(self.thinking),
            reasoning_effort.or(self.reasoning_effort),
        );
        let expert_before = ((self.expert_offload
            || self.ngram_cache_bytes.is_some_and(|bytes| bytes > 0))
            && (request.verbose || request.debug))
            .then(|| {
                self.json_request("GET", "/werk/experts/status", None, Duration::from_secs(2))
                    .ok()
            })
            .flatten();
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
        let mut usage_timings = OmlxUsageTimings::default();
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
                    update_omlx_completion_from_event(
                        &mut completion,
                        &mut usage_timings,
                        &value,
                        Some(started.elapsed().as_secs_f64()),
                    );
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
            update_omlx_completion_from_event(&mut completion, &mut usage_timings, &value, None);
            update_completion_from_message(&mut completion, &value)?;
        }
        ensure_visible_completion(&completion).context("oMLX completion has no visible answer")?;
        let generation_seconds = started.elapsed().as_secs_f64();
        let mut backend_diagnostics = finalize_omlx_completion_stats(
            &mut completion,
            request,
            generation_seconds,
            &usage_timings,
        );
        if request.verbose || request.debug {
            // Only explicitly sent values are known here; omitted sampling
            // controls continue to inherit the selected runtime's defaults.
            let controls = [
                "temperature",
                "top_p",
                "seed",
                "max_tokens",
                "chat_template_kwargs",
                "reasoning_effort",
            ]
            .into_iter()
            .filter_map(|key| body.get(key).map(|v| (key.to_owned(), v.clone())))
            .collect::<serde_json::Map<_, _>>();
            backend_diagnostics.push(format!(
                "oMLX request controls (omitted values inherit runtime defaults): {}",
                Value::Object(controls)
            ));
        }
        if let Some(before) = expert_before
            && let Ok(after) =
                self.json_request("GET", "/werk/experts/status", None, Duration::from_secs(2))
        {
            if let Some(diagnostic) = expert_interval_diagnostics(&before, &after) {
                backend_diagnostics.push(diagnostic);
            }
            if let Some(diagnostic) = glm_profile_diagnostics(&after) {
                backend_diagnostics.push(diagnostic);
            }
        }
        let assistant_message = completion.assistant_message();
        Ok(GenerateResponse {
            text: completion.text,
            assistant_message: Some(assistant_message),
            prompt_tokens: completion.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            finish_reason: completion.finish_reason,
            timings: GenerationTimings {
                cached_prompt_tokens: usage_timings
                    .cached_prompt_tokens
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|n| *n <= completion.prompt_tokens),
                first_token_seconds: completion.first_token_seconds,
                prompt_seconds: completion.prompt_seconds,
                decode_seconds: completion.decode_seconds,
                total_seconds: generation_seconds,
                ..Default::default()
            },
            backend_diagnostics,
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
        let mut stopped = false;
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
            stopped = child.wait().is_ok();
        }
        // If reaping failed, leave the private cache for later inspection. Its
        // inherited lifetime lock still protects it while the child is alive.
        if stopped {
            let _ = fs::remove_dir_all(&self.base_path);
        }
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
