use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    time::Instant,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent, GenerationBackend,
    GenerationTimings,
};
use crate::{
    media_companion::CompanionClient,
    model_store::{
        ArtifactKind, ArtifactStatus, ModelFormat, ModelManifest, ModelRuntimeIdentity, ModelStore,
    },
    runtime_control::{ModelResidencyStatus, StaticRuntimeAdapter},
};

const ONNX_GENAI_PYTHON_SCRIPT: &str = r#"
import contextlib
import gc
import json
import os
import sys
import time
from collections import OrderedDict

# Keep protocol stdout isolated from native/runtime logging. Anything emitted
# by onnxruntime-genai is diagnostic stderr, never a JSONL response frame.
protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
try:
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
except Exception:
    pass

import onnxruntime_genai as og

MODEL_CACHE = OrderedDict()

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}

def int_value(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except Exception:
            return None
    return None

def first_config_int(configs, paths):
    for config in configs:
        for path in paths:
            value = config
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            parsed = int_value(value)
            if parsed and parsed > 0:
                return parsed
    return None

def values_as_ints(value):
    if isinstance(value, list):
        return [parsed for parsed in (int_value(item) for item in value) if parsed is not None]
    parsed = int_value(value)
    return [] if parsed is None else [parsed]

def model_cache_capacity():
    raw = os.environ.get("WERK_ONNX_GENAI_MODEL_CACHE_SIZE", "1").strip()
    try:
        return max(0, min(8, int(raw)))
    except Exception:
        return 1


def collect_released_model():
    gc.collect()


def model_metadata(model_dir, tokenizer):
    genai_config = load_json(os.path.join(model_dir, "genai_config.json"))
    hf_config = load_json(os.path.join(model_dir, "config.json"))
    configs = [genai_config, hf_config]
    context_length = first_config_int(configs, [
        ("model", "context_length"),
        ("search", "max_length"),
        ("max_position_embeddings",),
        ("n_positions",),
        ("seq_length",),
        ("context_length",),
    ])
    eos_token_ids = set()
    for config in configs:
        if isinstance(config, dict):
            eos_token_ids.update(values_as_ints(config.get("eos_token_id")))
            model_config = config.get("model")
            if isinstance(model_config, dict):
                eos_token_ids.update(values_as_ints(model_config.get("eos_token_id")))
    try:
        eos_token_ids.update(values_as_ints(tokenizer.eos_token_ids))
    except Exception:
        pass
    return context_length, eos_token_ids


def validated_request(request):
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    for field in ("model", "model_key", "mode", "device"):
        if not isinstance(request.get(field), str) or not request[field]:
            raise ValueError(field + " is required")
    if not isinstance(request.get("prompt"), str):
        raise ValueError("prompt is required")
    max_tokens = int(request.get("max_tokens"))
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    stops = request.get("stop") or []
    if not isinstance(stops, list) or any(not isinstance(stop, str) for stop in stops):
        raise ValueError("stop must be a list of strings")
    request["max_tokens"] = max_tokens
    request["stop"] = stops
    return request


def load_cached_model(request):
    capacity = model_cache_capacity()
    while len(MODEL_CACHE) > capacity:
        _, evicted = MODEL_CACHE.popitem(last=False)
        del evicted
        collect_released_model()
    cache_key = (request["model_key"], request["mode"], request["device"])
    cached = MODEL_CACHE.pop(cache_key, None) if capacity else None
    if cached is not None:
        MODEL_CACHE[cache_key] = cached
        return cached, True, 0.0

    # Evict before loading a miss. With the default capacity of one this
    # avoids briefly retaining two full model allocations in RAM/VRAM.
    while MODEL_CACHE and (not capacity or len(MODEL_CACHE) >= capacity):
        _, evicted = MODEL_CACHE.popitem(last=False)
        del evicted
        collect_released_model()

    load_started = time.monotonic()
    model = og.Model(request["model"])
    tokenizer = og.Tokenizer(model)
    context_length, eos_token_ids = model_metadata(request["model"], tokenizer)
    entry = (model, tokenizer, context_length, eos_token_ids)
    load_seconds = time.monotonic() - load_started
    if capacity:
        MODEL_CACHE[cache_key] = entry
        while len(MODEL_CACHE) > capacity:
            _, evicted = MODEL_CACHE.popitem(last=False)
            del evicted
            collect_released_model()
    return entry, False, load_seconds


def generate_result(raw_request):
    request = validated_request(raw_request)
    total_started = time.monotonic()
    (model, tokenizer, context_length, eos_token_ids), model_cache_hit, load_seconds = load_cached_model(request)

    prompt_started = time.monotonic()
    input_ids = tokenizer.encode(request["prompt"])
    prompt_tokens = len(input_ids)
    prompt_seconds = time.monotonic() - prompt_started
    requested_max_new_tokens = request["max_tokens"]
    if context_length:
        available_new_tokens = max(1, context_length - prompt_tokens)
        max_new_tokens = min(requested_max_new_tokens, available_new_tokens)
    else:
        max_new_tokens = requested_max_new_tokens
    max_length = prompt_tokens + max_new_tokens

    default_stops = [
        "<|end|>",
        "<|endoftext|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "</s>",
    ]
    stop_strings = []
    for stop in list(request["stop"]) + default_stops:
        if stop and stop not in stop_strings:
            stop_strings.append(stop)

    params = og.GeneratorParams(model)
    search_options = {"max_length": max_length}
    if request.get("temperature") is not None:
        search_options["temperature"] = float(request["temperature"])
    if request.get("top_p") is not None:
        search_options["top_p"] = float(request["top_p"])
    if request.get("seed") is not None:
        search_options["random_seed"] = int(request["seed"])
    params.set_search_options(**search_options)

    # Generator state is request-local. Only immutable model/tokenizer objects
    # participate in residency; prompts and KV/generator state never do.
    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)
    stream = tokenizer.create_stream()
    generation_started = time.monotonic()
    text = ""
    completion_tokens = 0
    finish_reason = "generator_done"
    first_token_at = None

    while completion_tokens < max_new_tokens:
        if generator.is_done():
            finish_reason = "generator_done"
            break
        generator.generate_next_token()
        next_tokens = generator.get_next_tokens()
        if len(next_tokens) == 0:
            finish_reason = "generator_done" if generator.is_done() else "eos"
            break
        token_id = int(next_tokens[0])
        if first_token_at is None:
            first_token_at = time.monotonic()
        completion_tokens += 1
        candidate = text + stream.decode(token_id)
        if token_id in eos_token_ids:
            text = candidate
            finish_reason = "eos"
            break
        stop_match = None
        for stop in stop_strings:
            index = candidate.find(stop)
            if index >= 0 and (stop_match is None or index < stop_match):
                stop_match = index
        if stop_match is not None:
            text = candidate[:stop_match]
            finish_reason = "stop_sequence"
            break
        text = candidate
    else:
        if completion_tokens >= max_new_tokens:
            finish_reason = "max_new_tokens"

    for stop in stop_strings:
        index = text.find(stop)
        if index >= 0:
            text = text[:index]
            if finish_reason not in ("eos", "stop_sequence"):
                finish_reason = "stop_sequence"
            break

    return {
        "ok": True,
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "stop_reason": finish_reason,
        "requested_max_new_tokens": requested_max_new_tokens,
        "max_new_tokens": max_new_tokens,
        "max_length": max_length,
        "context_length": context_length,
        "model_cache_hit": model_cache_hit,
        "load_seconds": load_seconds,
        "prompt_seconds": prompt_seconds,
        "first_token_seconds": 0.0 if first_token_at is None else first_token_at - generation_started,
        "decode_seconds": time.monotonic() - generation_started,
        "total_seconds": time.monotonic() - total_started,
    }


def safe_error(error):
    return {
        "ok": False,
        "error": {
            "code": "onnx_genai_failed",
            "message": "ONNX GenAI generation failed",
            "detail": type(error).__name__,
        },
    }


def serve():
    for raw in sys.stdin:
        request_id = None
        try:
            frame = json.loads(raw)
            request_id = frame.get("request_id")
            if frame.get("transport_version") != 1 or isinstance(request_id, bool) or not isinstance(request_id, int):
                raise ValueError("invalid transport envelope")
            operation = frame.get("operation")
            if operation == "transport-handshake":
                response = {"ok": True, "protocol_version": 1}
            elif operation == "execute":
                with contextlib.redirect_stdout(sys.stderr):
                    response = generate_result(frame.get("payload"))
            else:
                raise ValueError("unsupported operation")
        except Exception as error:
            response = safe_error(error)
        envelope = {
            "transport_version": 1,
            "request_id": request_id,
            "response": response,
        }
        protocol_out.write(json.dumps(envelope, separators=(",", ":")) + "\n")
        protocol_out.flush()
        # Do not retain request envelopes, prompts, generator responses, or
        # serialized input while the resident worker waits for its next job.
        frame = None
        response = None
        envelope = None
        raw = None


def main():
    if len(sys.argv) != 2:
        raise SystemExit(64)
    if sys.argv[1] == "serve":
        serve()
        return
    if sys.argv[1] != "execute":
        raise SystemExit(64)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            response = generate_result(json.load(sys.stdin))
    except Exception as error:
        response = safe_error(error)
    protocol_out.write(json.dumps(response, separators=(",", ":")) + "\n")
    protocol_out.flush()


if __name__ == "__main__":
    main()
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxRuntimeMode {
    Cuda,
    Rocm,
    Cpu,
}

impl OnnxRuntimeMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cuda => "onnxruntime-cuda",
            Self::Rocm => "onnxruntime-rocm",
            Self::Cpu => "onnxruntime-cpu",
        }
    }

    pub fn display(self) -> &'static str {
        match self {
            Self::Cuda => "ONNX Runtime CUDA",
            Self::Rocm => "ONNX Runtime ROCm",
            Self::Cpu => "ONNX Runtime CPU",
        }
    }

    fn bundle_env(self) -> &'static str {
        match self {
            Self::Cuda => "WERK_ONNX_RUNTIME_BUNDLE_CUDA",
            Self::Rocm => "WERK_ONNX_RUNTIME_BUNDLE_ROCM",
            Self::Cpu => "WERK_ONNX_RUNTIME_BUNDLE_CPU",
        }
    }
}

fn onnx_genai_device(mode: OnnxRuntimeMode) -> &'static str {
    match mode {
        OnnxRuntimeMode::Cuda => "cuda",
        OnnxRuntimeMode::Rocm => "rocm",
        OnnxRuntimeMode::Cpu => "cpu",
    }
}

const DEFAULT_ONNX_GENAI_MODEL_CACHE_SIZE: usize = 1;
const MAX_ONNX_GENAI_MODEL_CACHE_SIZE: usize = 8;

fn parse_onnx_genai_model_cache_size(raw: Option<&str>) -> usize {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return DEFAULT_ONNX_GENAI_MODEL_CACHE_SIZE;
    };
    raw.parse::<i64>()
        .map(|capacity| capacity.clamp(0, MAX_ONNX_GENAI_MODEL_CACHE_SIZE as i64) as usize)
        .unwrap_or(DEFAULT_ONNX_GENAI_MODEL_CACHE_SIZE)
}

fn onnx_genai_model_cache_size() -> usize {
    parse_onnx_genai_model_cache_size(env::var("WERK_ONNX_GENAI_MODEL_CACHE_SIZE").ok().as_deref())
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OnnxProvisionOptions {
    pub install_missing_runtime: bool,
    pub verbose: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxRuntimeAvailability {
    Ready,
    Installable,
    Unavailable,
}

#[derive(Clone)]
pub struct OnnxRuntimeBackend {
    store: ModelStore,
    mode: OnnxRuntimeMode,
    python_genai_worker: Arc<Mutex<Option<OnnxGenaiWorker>>>,
}

#[derive(Clone)]
struct OnnxGenaiWorker {
    python: PathBuf,
    client: CompanionClient,
}

#[derive(Debug, Clone)]
pub struct OnnxRuntimeDiscovery {
    pub path: Option<PathBuf>,
    pub source: String,
    pub attempts: Vec<OnnxRuntimeAttempt>,
}

#[derive(Debug, Clone)]
pub struct OnnxRuntimeAttempt {
    pub label: String,
    pub path: Option<PathBuf>,
    pub exists: bool,
    pub usable: bool,
    pub detail: String,
}

impl OnnxRuntimeBackend {
    pub fn new(store: ModelStore, mode: OnnxRuntimeMode) -> Self {
        Self {
            store,
            mode,
            python_genai_worker: Arc::new(Mutex::new(None)),
        }
    }

    pub fn probe(store: &ModelStore, mode: OnnxRuntimeMode) -> Result<String> {
        let discovery = discover_onnx_runtime(store, mode);
        if let Some(path) = discovery.path.as_ref() {
            return Ok(format!("{} runner {}", mode.display(), path.display()));
        }
        if mode == OnnxRuntimeMode::Cpu
            && let Some(runtime) = discover_onnx_genai_python()
        {
            return Ok(format!(
                "{} via Python onnxruntime-genai {}",
                mode.display(),
                runtime.python.display()
            ));
        }
        Err(anyhow!(
            "{}",
            missing_message_from_discovery(mode, &discovery)
        ))
    }

    pub fn discover(store: &ModelStore, mode: OnnxRuntimeMode) -> OnnxRuntimeDiscovery {
        discover_onnx_runtime(store, mode)
    }

    pub fn missing_message(store: &ModelStore, mode: OnnxRuntimeMode) -> String {
        missing_message_from_discovery(mode, &discover_onnx_runtime(store, mode))
    }

    pub fn unavailable_reason(store: &ModelStore, mode: OnnxRuntimeMode) -> String {
        concise_unavailable_reason(&discover_onnx_runtime(store, mode))
    }

    pub fn availability(store: &ModelStore, mode: OnnxRuntimeMode) -> OnnxRuntimeAvailability {
        let discovery = discover_onnx_runtime(store, mode);
        if discovery.path.is_some() {
            OnnxRuntimeAvailability::Ready
        } else if find_bundled_runner(mode).is_some() {
            OnnxRuntimeAvailability::Installable
        } else {
            OnnxRuntimeAvailability::Unavailable
        }
    }

    pub fn ensure_available_for_model(
        store: &ModelStore,
        manifest: &ModelManifest,
        mode: OnnxRuntimeMode,
    ) -> Result<()> {
        Self::ensure_available_for_model_with_options(
            store,
            manifest,
            mode,
            OnnxProvisionOptions::default(),
        )
    }

    pub fn ensure_available_for_model_with_options(
        store: &ModelStore,
        manifest: &ModelManifest,
        mode: OnnxRuntimeMode,
        options: OnnxProvisionOptions,
    ) -> Result<()> {
        if !matches!(
            manifest.format,
            ModelFormat::SafeTensors | ModelFormat::Onnx
        ) {
            bail!("ONNX Runtime route requires a safetensors source model or direct ONNX model");
        }
        let mut discovery = discover_onnx_runtime(store, mode);
        if discovery.path.is_none() && options.install_missing_runtime {
            install_managed_onnx_runtime(store, mode)?;
            discovery = discover_onnx_runtime(store, mode);
        }
        if discovery.path.is_none() {
            if mode == OnnxRuntimeMode::Cpu
                && onnx_genai_model_dir(store, manifest).is_some()
                && discover_onnx_genai_python().is_some()
            {
                if options.verbose {
                    eprintln!(
                        "Selected runtime: {} via Python onnxruntime-genai",
                        mode.display()
                    );
                    eprintln!("Runtime status: ready");
                }
                return Ok(());
            }
            bail!("{}", missing_message_from_discovery(mode, &discovery));
        }
        if options.verbose {
            eprintln!("Selected runtime: {}", mode.display());
            eprintln!("Runtime status: ready");
        }
        if manifest.format == ModelFormat::Onnx {
            if options.verbose {
                eprintln!("Artifact: direct ONNX model");
                eprintln!("Result: runtime ready");
            }
            return Ok(());
        }
        if store.ready_onnx_artifact(manifest).is_some() {
            if options.verbose {
                eprintln!("Artifact: ready");
                eprintln!("Result: runtime ready");
            }
            return Ok(());
        }
        if options.verbose {
            eprintln!("Artifact: building ONNX export");
        }
        if let Err(err) = store
            .build_onnx_artifact(&manifest.id, false)
            .with_context(|| "ONNX artifact generation failed")
        {
            if options.verbose {
                eprintln!("Result: artifact build failed: {err}");
            }
            return Err(err);
        }
        if options.verbose {
            eprintln!("Result: runtime ready");
        }
        Ok(())
    }

    fn runner(&self) -> Result<PathBuf> {
        discover_onnx_runtime(&self.store, self.mode)
            .path
            .as_ref()
            .cloned()
            .ok_or_else(|| anyhow!("{}", Self::missing_message(&self.store, self.mode)))
    }

    fn model_path(&self, manifest: &ModelManifest) -> Result<PathBuf> {
        if manifest.format == ModelFormat::Onnx {
            let path = manifest
                .model_path
                .as_deref()
                .context("ONNX manifest has no model_path")?;
            return Ok(self.store.absolute_model_file(manifest, path));
        }
        if manifest.format != ModelFormat::SafeTensors {
            bail!(
                "ONNX Runtime backend supports safetensors source models and direct ONNX models only"
            );
        }
        if let Some(artifact) = self.store.ready_onnx_artifact(manifest) {
            if artifact.status != ArtifactStatus::Ready || artifact.kind != ArtifactKind::Onnx {
                bail!("ONNX artifact for '{}' is not ready", manifest.id);
            }
            return Ok(self.store.absolute_artifact_path(manifest, &artifact));
        }
        let artifact = self.store.build_onnx_artifact(&manifest.id, false)?;
        if artifact.status != ArtifactStatus::Ready || artifact.kind != ArtifactKind::Onnx {
            bail!("ONNX artifact for '{}' is not ready", manifest.id);
        }
        Ok(self.store.absolute_artifact_path(manifest, &artifact))
    }

    fn generate_inner(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        if !request.image_urls.is_empty() {
            bail!(
                "ONNX Runtime text backend received image inputs; use a VLM-capable model/runtime"
            );
        }
        let total_started = Instant::now();
        let model_path = self.model_path(manifest)?;
        if let Ok(runner) = self.runner() {
            return self.generate_with_runner(
                manifest,
                request,
                total_started,
                &runner,
                &model_path,
            );
        }

        if self.mode == OnnxRuntimeMode::Cpu
            && let Some(model_dir) = onnx_genai_model_dir(&self.store, manifest)
            && let Some(runtime) = discover_onnx_genai_python()
        {
            return self.generate_with_python_genai(
                manifest,
                request,
                total_started,
                &runtime.python,
                &model_dir,
            );
        }

        bail!("{}", Self::missing_message(&self.store, self.mode));
    }

    fn generate_with_runner(
        &self,
        _manifest: &ModelManifest,
        request: GenerateRequest,
        total_started: Instant,
        runner: &Path,
        model_path: &Path,
    ) -> Result<GenerateResponse> {
        if request.verbose {
            eprintln!("Starting generation...");
        }
        if request.debug {
            eprintln!("selected backend: {}", self.mode.label());
            eprintln!("ONNX Runtime runner: {}", runner.display());
            eprintln!("ONNX model: {}", model_path.display());
        }
        let started = Instant::now();
        let output = Command::new(runner)
            .arg("--model")
            .arg(model_path)
            .arg("--prompt")
            .arg(&request.prompt)
            .arg("--max-tokens")
            .arg(request.max_tokens.to_string())
            .arg("--backend")
            .arg(match self.mode {
                OnnxRuntimeMode::Cuda => "cuda",
                OnnxRuntimeMode::Rocm => "rocm",
                OnnxRuntimeMode::Cpu => "cpu",
            })
            .arg("--json")
            .output()
            .with_context(|| format!("failed to run ONNX Runtime runner {}", runner.display()))?;
        if !output.status.success() {
            bail!(
                "ONNX Runtime runner failed: {}",
                command_output_detail(&output)
            );
        }
        let value: Value = serde_json::from_slice(&output.stdout).with_context(|| {
            format!(
                "ONNX Runtime runner returned invalid JSON: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        })?;
        let text = value
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("ONNX Runtime runner JSON missing string field 'text'"))?
            .to_string();
        let prompt_tokens = value
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or_else(|| request.prompt.split_whitespace().count().max(1));
        let completion_tokens = value
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or_else(|| text.split_whitespace().count().max(1));
        let elapsed = started.elapsed().as_secs_f64();
        let prompt_seconds = value
            .get("prompt_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN);
        Ok(GenerateResponse {
            text,
            assistant_message: None,
            prompt_tokens,
            completion_tokens,
            finish_reason: value
                .get("finish_reason")
                .and_then(Value::as_str)
                .unwrap_or("stop")
                .to_string(),
            timings: GenerationTimings {
                load_seconds: 0.0,
                warmup_seconds: 0.0,
                first_token_seconds: 0.0,
                prompt_seconds,
                decode_seconds: elapsed,
                total_seconds: total_started.elapsed().as_secs_f64(),
            },
            backend_diagnostics: Vec::new(),
        })
    }

    fn generate_with_python_genai(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        total_started: Instant,
        python: &Path,
        model_dir: &Path,
    ) -> Result<GenerateResponse> {
        if request.verbose {
            eprintln!("Starting generation...");
        }
        if request.debug {
            eprintln!(
                "selected backend: {} via Python onnxruntime-genai",
                self.mode.label()
            );
            eprintln!("Python: {}", python.display());
            eprintln!("ONNX GenAI model: {}", model_dir.display());
        }
        let client = self.python_genai_client(python)?;
        self.generate_with_python_genai_client(manifest, request, total_started, model_dir, &client)
    }

    fn python_genai_client(&self, python: &Path) -> Result<CompanionClient> {
        let mut worker = self
            .python_genai_worker
            .lock()
            .map_err(|_| anyhow!("ONNX GenAI resident worker registry is poisoned"))?;
        if let Some(worker) = worker.as_ref()
            && worker.python == python
        {
            return Ok(worker.client.clone());
        }

        let client = CompanionClient::from_embedded_python(
            python.to_path_buf(),
            ONNX_GENAI_PYTHON_SCRIPT,
            "Werk embedded ONNX GenAI worker",
        )
        .with_resident_worker();
        *worker = Some(OnnxGenaiWorker {
            python: python.to_path_buf(),
            client: client.clone(),
        });
        Ok(client)
    }

    fn generate_with_python_genai_client(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        total_started: Instant,
        model_dir: &Path,
        client: &CompanionClient,
    ) -> Result<GenerateResponse> {
        let model_dir = model_dir
            .to_str()
            .context("ONNX GenAI model directory is not valid UTF-8")?;
        let model_key = ModelRuntimeIdentity::from_manifest(manifest)?.to_string();
        let worker_request = json!({
            "model": model_dir,
            "model_key": model_key,
            "mode": self.mode.label(),
            "device": onnx_genai_device(self.mode),
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seed": request.seed,
            "stop": request.stop,
        });
        let started = Instant::now();
        let value = client
            .request("execute", &worker_request)
            .context("ONNX GenAI resident worker failed")?;
        let text = value
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("ONNX GenAI worker response missing string field 'text'"))?
            .to_string();
        let prompt_tokens = value
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or_else(|| request.prompt.split_whitespace().count().max(1));
        let completion_tokens = value
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or_else(|| text.split_whitespace().count().max(1));
        let decode_seconds = value
            .get("decode_seconds")
            .and_then(Value::as_f64)
            .unwrap_or_else(|| started.elapsed().as_secs_f64());
        let first_token_seconds = value
            .get("first_token_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let load_seconds = value
            .get("load_seconds")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .context("ONNX GenAI worker response missing valid load_seconds")?;
        let model_cache_hit = value
            .get("model_cache_hit")
            .and_then(Value::as_bool)
            .context("ONNX GenAI worker response missing boolean model_cache_hit")?;
        let prompt_seconds = value
            .get("prompt_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN);
        let mut backend_diagnostics = vec![
            format!("model_cache_hit: {model_cache_hit}"),
            format!("model_load_seconds: {load_seconds:.6}"),
        ];
        for (label, key) in [
            ("stop reason", "stop_reason"),
            ("requested max new tokens", "requested_max_new_tokens"),
            ("effective max new tokens", "max_new_tokens"),
            ("effective max length", "max_length"),
            ("context length", "context_length"),
        ] {
            if let Some(value) = value.get(key) {
                match value {
                    Value::String(text) => backend_diagnostics.push(format!("{label}: {text}")),
                    Value::Number(number) => backend_diagnostics.push(format!("{label}: {number}")),
                    _ => {}
                }
            }
        }
        Ok(GenerateResponse {
            text,
            assistant_message: None,
            prompt_tokens,
            completion_tokens,
            finish_reason: value
                .get("finish_reason")
                .and_then(Value::as_str)
                .unwrap_or("stop")
                .to_string(),
            timings: GenerationTimings {
                load_seconds,
                warmup_seconds: 0.0,
                first_token_seconds,
                prompt_seconds,
                decode_seconds,
                total_seconds: total_started.elapsed().as_secs_f64(),
            },
            backend_diagnostics,
        })
    }
}

impl GenerationBackend for OnnxRuntimeBackend {
    fn runtime_control_adapter(&self) -> Arc<dyn crate::runtime_control::BackendRuntimeAdapter> {
        let route = if discover_onnx_runtime(&self.store, self.mode).path.is_some() {
            OnnxResidencyRoute::OneShotRunner
        } else {
            OnnxResidencyRoute::Unavailable
        };
        onnx_runtime_control_adapter(self.mode, route)
    }

    fn runtime_control_adapter_for(
        &self,
        manifest: &ModelManifest,
    ) -> Result<Arc<dyn crate::runtime_control::BackendRuntimeAdapter>> {
        let discovery = discover_onnx_runtime(&self.store, self.mode);
        let route = if discovery.path.is_some() {
            OnnxResidencyRoute::OneShotRunner
        } else if self.mode == OnnxRuntimeMode::Cpu
            && onnx_genai_model_dir(&self.store, manifest).is_some()
        {
            if discover_onnx_genai_python().is_some() {
                if onnx_genai_model_cache_size() == 0 {
                    OnnxResidencyRoute::EmbeddedPythonCacheDisabled
                } else {
                    OnnxResidencyRoute::EmbeddedPython
                }
            } else {
                OnnxResidencyRoute::Unavailable
            }
        } else {
            OnnxResidencyRoute::Unavailable
        };
        Ok(onnx_runtime_control_adapter(self.mode, route))
    }

    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        eprintln!("Using {} backend", self.mode.display());
        Self::ensure_available_for_model(&self.store, manifest, self.mode)
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        self.generate_inner(manifest, request)
    }

    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        let backend = self.clone();
        let (tx, rx) = mpsc::channel(4);
        tokio::task::spawn_blocking(move || {
            let result = backend.generate_inner(&manifest, request);
            match result {
                Ok(response) => {
                    if !response.text.is_empty() {
                        let _ = tx.blocking_send(Ok(GenerateStreamEvent::TextChunk(
                            response.text.clone(),
                        )));
                    }
                    let _ = tx.blocking_send(Ok(GenerateStreamEvent::Done {
                        finish_reason: response.finish_reason,
                        prompt_tokens: response.prompt_tokens,
                        completion_tokens: response.completion_tokens,
                        timings: response.timings,
                        backend_diagnostics: response.backend_diagnostics,
                    }));
                }
                Err(err) => {
                    let _ = tx.blocking_send(Err(err.to_string()));
                }
            }
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnnxResidencyRoute {
    EmbeddedPython,
    EmbeddedPythonCacheDisabled,
    OneShotRunner,
    Unavailable,
}

fn onnx_runtime_control_adapter(
    mode: OnnxRuntimeMode,
    route: OnnxResidencyRoute,
) -> Arc<dyn crate::runtime_control::BackendRuntimeAdapter> {
    let (backend, status, detail) = match route {
        OnnxResidencyRoute::EmbeddedPython => (
            format!("{}-python-genai", mode.label()),
            ModelResidencyStatus::Supported,
            "Werk's resident Python ONNX GenAI worker reuses the exact model and tokenizer with a bounded LRU; generator and prompt state remain request-local",
        ),
        OnnxResidencyRoute::EmbeddedPythonCacheDisabled => (
            format!("{}-python-genai", mode.label()),
            ModelResidencyStatus::Unsupported,
            "the Werk ONNX GenAI worker is available, but model residency is disabled by WERK_ONNX_GENAI_MODEL_CACHE_SIZE=0",
        ),
        OnnxResidencyRoute::OneShotRunner => (
            mode.label().to_string(),
            ModelResidencyStatus::Unsupported,
            "the configured ONNX Runtime runner is launched once per request and does not expose model residency",
        ),
        OnnxResidencyRoute::Unavailable => (
            mode.label().to_string(),
            ModelResidencyStatus::Unavailable,
            "no compatible ONNX Runtime execution path is currently available for this model",
        ),
    };
    Arc::new(
        StaticRuntimeAdapter::new(backend)
            .with_accelerator_family(onnx_genai_device(mode))
            .with_model_residency(status, detail),
    )
}

fn discover_onnx_runtime(store: &ModelStore, mode: OnnxRuntimeMode) -> OnnxRuntimeDiscovery {
    let mut attempts = Vec::new();
    let env_name = match mode {
        OnnxRuntimeMode::Cuda => "WERK_ONNX_RUNTIME_CUDA",
        OnnxRuntimeMode::Rocm => "WERK_ONNX_RUNTIME_ROCM",
        OnnxRuntimeMode::Cpu => "WERK_ONNX_RUNTIME_CPU",
    };
    for (label, path) in [
        (
            env_name.to_string(),
            env::var_os(env_name).map(PathBuf::from),
        ),
        (
            "WERK_ONNX_RUNTIME".to_string(),
            env::var_os("WERK_ONNX_RUNTIME").map(PathBuf::from),
        ),
        (
            "managed cache".to_string(),
            Some(managed_runner_path(store, mode)),
        ),
        (
            "PATH: werk-onnx-runner".to_string(),
            find_in_path(runner_name()),
        ),
    ] {
        let Some(path) = path else {
            attempts.push(OnnxRuntimeAttempt {
                label,
                path: None,
                exists: false,
                usable: false,
                detail: "not set".to_string(),
            });
            continue;
        };
        let usable = runner_help_ok(&path);
        attempts.push(OnnxRuntimeAttempt {
            label: label.clone(),
            path: Some(path.clone()),
            exists: path.is_file(),
            usable,
            detail: if usable {
                "runner --help ok".to_string()
            } else {
                "runner missing or --help failed".to_string()
            },
        });
        if usable {
            return OnnxRuntimeDiscovery {
                path: Some(path),
                source: label,
                attempts,
            };
        }
    }
    OnnxRuntimeDiscovery {
        path: None,
        source: "missing".to_string(),
        attempts,
    }
}

pub fn managed_runner_path(store: &ModelStore, mode: OnnxRuntimeMode) -> PathBuf {
    let name = match mode {
        OnnxRuntimeMode::Cuda => "onnxruntime-cuda",
        OnnxRuntimeMode::Rocm => "onnxruntime-rocm",
        OnnxRuntimeMode::Cpu => "onnxruntime-cpu",
    };
    store.home().join("backends").join(name).join(runner_name())
}

pub fn install_managed_onnx_runtime(store: &ModelStore, mode: OnnxRuntimeMode) -> Result<PathBuf> {
    let source =
        find_bundled_runner(mode).ok_or_else(|| anyhow!("{}", missing_bundle_message(mode)))?;
    let dest = managed_runner_path(store, mode);
    fs::create_dir_all(
        dest.parent()
            .ok_or_else(|| anyhow!("invalid managed ONNX Runtime path {}", dest.display()))?,
    )?;
    if source != dest {
        fs::copy(&source, &dest).with_context(|| {
            format!(
                "failed to copy ONNX Runtime runner from {} to {}",
                source.display(),
                dest.display()
            )
        })?;
    }
    make_executable(&dest)?;
    if !runner_help_ok(&dest) {
        bail!(
            "installed ONNX Runtime runner did not pass --help validation: {}",
            dest.display()
        );
    }
    Ok(dest)
}

fn find_bundled_runner(mode: OnnxRuntimeMode) -> Option<PathBuf> {
    bundled_runner_candidates(mode)
        .into_iter()
        .find(|path| runner_help_ok(path))
}

fn bundled_runner_candidates(mode: OnnxRuntimeMode) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os(mode.bundle_env()).map(PathBuf::from) {
        candidates.push(path);
    }
    if let Some(path) = env::var_os("WERK_ONNX_RUNTIME_BUNDLE").map(PathBuf::from) {
        candidates.push(path);
    }

    let backend_dir = match mode {
        OnnxRuntimeMode::Cuda => "onnxruntime-cuda",
        OnnxRuntimeMode::Rocm => "onnxruntime-rocm",
        OnnxRuntimeMode::Cpu => "onnxruntime-cpu",
    };

    if let Ok(exe) = env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("backends").join(backend_dir).join(runner_name()));
        candidates.push(dir.join(backend_dir).join(runner_name()));
    }

    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        let root = PathBuf::from(manifest_dir);
        candidates.push(root.join("backends").join(backend_dir).join(runner_name()));
        candidates.push(
            root.join("dist")
                .join("backends")
                .join(backend_dir)
                .join(runner_name()),
        );
    }

    candidates
}

fn missing_bundle_message(mode: OnnxRuntimeMode) -> String {
    let mut message = format!(
        "No bundled {} runner found for provisioning.",
        mode.display()
    );
    message.push_str("\n\nTried:");
    for candidate in bundled_runner_candidates(mode) {
        message.push_str(&format!("\n- {}", candidate.display()));
    }
    message.push_str("\n\nFix:");
    message.push_str(&format!(
        "\n- set {}=/path/to/{}",
        mode.bundle_env(),
        runner_name()
    ));
    message.push_str("\n- or ship the runner under backends/<runtime>/ next to the werk binary");
    message.push_str("\n- or set WERK_ONNX_RUNTIME=/path/to/werk-onnx-runner");
    message
}

fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

fn missing_message_from_discovery(
    mode: OnnxRuntimeMode,
    discovery: &OnnxRuntimeDiscovery,
) -> String {
    let mut message = format!("No {} runner found.\n\nTried:", mode.display());
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
    message.push_str("\n- set WERK_ONNX_RUNTIME=/path/to/werk-onnx-runner");
    message.push_str("\n- or install a managed ONNX Runtime runner artifact for Werk");
    message.push_str(
        "\n- or install Python ONNX GenAI support with `python3 -m pip install onnxruntime-genai`",
    );
    message
}

fn concise_unavailable_reason(discovery: &OnnxRuntimeDiscovery) -> String {
    if discovery
        .attempts
        .iter()
        .any(|attempt| attempt.exists && !attempt.usable)
    {
        "runner validation failed".to_string()
    } else {
        "runner not installed or bundled".to_string()
    }
}

#[derive(Debug, Clone)]
struct OnnxGenaiPythonRuntime {
    python: PathBuf,
}

fn discover_onnx_genai_python() -> Option<OnnxGenaiPythonRuntime> {
    let mut candidates = Vec::<PathBuf>::new();
    for env_name in ["WERK_ONNX_GENAI_PYTHON", "WERK_ONNX_RUNTIME_PYTHON"] {
        if let Some(path) = env::var_os(env_name).map(PathBuf::from) {
            candidates.push(path);
        }
    }
    if let Some(path) = find_in_path("python3") {
        candidates.push(path);
    }
    if let Some(path) = find_in_path("python") {
        candidates.push(path);
    }

    let mut seen = Vec::<PathBuf>::new();
    candidates.into_iter().find_map(|python| {
        if seen.iter().any(|path| path == &python) {
            return None;
        }
        seen.push(python.clone());
        python_supports_onnx_genai(&python).then_some(OnnxGenaiPythonRuntime { python })
    })
}

fn python_supports_onnx_genai(python: &Path) -> bool {
    python.is_file()
        && Command::new(python)
            .args(["-c", "import onnxruntime_genai"])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
}

fn onnx_genai_model_dir(store: &ModelStore, manifest: &ModelManifest) -> Option<PathBuf> {
    if manifest.format != ModelFormat::Onnx {
        return None;
    }

    if let Some(path) = manifest
        .model_path
        .as_deref()
        .map(|path| store.absolute_model_file(manifest, path))
    {
        if is_onnx_genai_dir(&path) {
            return Some(path);
        }
        if let Some(parent) = path.parent()
            && is_onnx_genai_dir(parent)
        {
            return Some(parent.to_path_buf());
        }
    }

    let mut candidates = manifest
        .files
        .iter()
        .filter(|file| file.path.ends_with("/genai_config.json"))
        .filter_map(|file| {
            store
                .model_dir(&manifest.id)
                .join(&file.path)
                .parent()
                .map(Path::to_path_buf)
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|left| onnx_genai_dir_priority(left));
    candidates.into_iter().find(|path| is_onnx_genai_dir(path))
}

fn is_onnx_genai_dir(path: &Path) -> bool {
    path.is_dir() && path.join("genai_config.json").is_file()
}

fn onnx_genai_dir_priority(path: &Path) -> (usize, String) {
    let text = path.to_string_lossy().to_ascii_lowercase();
    let priority = if text.contains("cpu_and_mobile") || text.contains("/cpu") {
        0
    } else if text.contains("cuda") {
        1
    } else if text.contains("directml") {
        2
    } else {
        3
    };
    (priority, text)
}

fn runner_help_ok(path: &Path) -> bool {
    path.is_file()
        && Command::new(path)
            .arg("--help")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
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

fn runner_name() -> &'static str {
    if cfg!(windows) {
        "werk-onnx-runner.exe"
    } else {
        "werk-onnx-runner"
    }
}

fn command_output_detail(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        output.status.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::StreamGranularity,
        model_store::{ModelFile, ModelSource},
        runtime_control::{AUTOMATIC_REUSE_OPERATION, MODEL_RESIDENCY_CAPABILITY},
        werk_protocol::CapabilityStatus,
    };
    use std::{
        io::Write,
        process::Stdio,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    const FAKE_RESIDENT_ONNX_WORKER: &str = r#"
import json
import sys
from collections import OrderedDict

if len(sys.argv) != 2 or sys.argv[1] != "serve":
    raise SystemExit(64)

cache = OrderedDict()
for raw in sys.stdin:
    frame = json.loads(raw)
    payload = frame["payload"]
    key = (payload["model_key"], payload["mode"], payload["device"])
    hit = key in cache
    if hit:
        cache.pop(key)
    cache[key] = True
    while len(cache) > 1:
        cache.popitem(last=False)
    response = {
        "ok": True,
        "text": payload["prompt"],
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "finish_reason": "stop",
        "stop_reason": "fixture",
        "requested_max_new_tokens": payload["max_tokens"],
        "max_new_tokens": payload["max_tokens"],
        "max_length": payload["max_tokens"] + 2,
        "context_length": 4096,
        "model_cache_hit": hit,
        "load_seconds": 0.0 if hit else 0.125,
        "prompt_seconds": 0.01,
        "first_token_seconds": 0.02,
        "decode_seconds": 0.03,
        "total_seconds": 0.04,
    }
    print(json.dumps({
        "transport_version": 1,
        "request_id": frame["request_id"],
        "response": response,
    }), flush=True)
"#;

    const CRASHING_ONNX_WORKER: &str = r#"
import os
import sys

if len(sys.argv) != 2 or sys.argv[1] != "serve":
    raise SystemExit(64)
for _ in sys.stdin:
    print("deterministic ONNX worker crash", file=sys.stderr, flush=True)
    os._exit(17)
"#;

    const SLOW_ONNX_WORKER: &str = r#"
import json
import sys
import time

if len(sys.argv) != 2 or sys.argv[1] != "serve":
    raise SystemExit(64)
for raw in sys.stdin:
    frame = json.loads(raw)
    time.sleep(1.0)
    print(json.dumps({
        "transport_version": 1,
        "request_id": frame["request_id"],
        "response": {"ok": True},
    }), flush=True)
"#;

    fn test_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("werk-onnxruntime-{name}-{unique}"))
    }

    fn manifest_with_model_path(id: &str, model_path: Option<&str>) -> ModelManifest {
        ModelManifest {
            id: id.to_string(),
            source: ModelSource::LocalPath {
                path: "test".to_string(),
            },
            format: ModelFormat::Onnx,
            architecture: Some("phi3".to_string()),
            tokenizer_path: None,
            config_path: None,
            model_path: model_path.map(str::to_string),
            backend: "onnxruntime".to_string(),
            created_unix: 0,
            files: Vec::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        }
    }

    fn generate_request(prompt: &str) -> GenerateRequest {
        GenerateRequest {
            prompt: prompt.to_string(),
            messages: Vec::new(),
            image_urls: Vec::new(),
            max_tokens: 8,
            temperature: Some(0.25),
            top_p: Some(0.9),
            stop: vec!["stop".to_string()],
            seed: Some(7),
            stream_granularity: StreamGranularity::Chunk,
            verbose: false,
            debug: false,
            tool_config: None,
        }
    }

    fn python_for_test() -> Option<PathBuf> {
        find_in_path("python3").or_else(|| find_in_path("python"))
    }

    #[test]
    fn embedded_onnx_genai_worker_is_valid_python() {
        let Some(python) = python_for_test() else {
            return;
        };
        let mut child = Command::new(python)
            .args([
                "-c",
                "import sys; compile(sys.stdin.read(), '<werk_onnx_genai>', 'exec')",
            ])
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(ONNX_GENAI_PYTHON_SCRIPT.as_bytes())
            .unwrap();
        assert!(child.wait().unwrap().success());
    }

    #[test]
    fn residency_contract_distinguishes_resident_one_shot_and_missing_routes() {
        let cases = [
            (
                OnnxResidencyRoute::EmbeddedPython,
                CapabilityStatus::Supported,
                true,
                "onnxruntime-cpu-python-genai",
            ),
            (
                OnnxResidencyRoute::OneShotRunner,
                CapabilityStatus::Unsupported,
                false,
                "onnxruntime-cpu",
            ),
            (
                OnnxResidencyRoute::EmbeddedPythonCacheDisabled,
                CapabilityStatus::Unsupported,
                false,
                "onnxruntime-cpu-python-genai",
            ),
            (
                OnnxResidencyRoute::Unavailable,
                CapabilityStatus::Unavailable,
                false,
                "onnxruntime-cpu",
            ),
        ];

        for (route, expected, reuses, expected_backend) in cases {
            let descriptor = onnx_runtime_control_adapter(OnnxRuntimeMode::Cpu, route).descriptor();
            assert_eq!(descriptor.backend, expected_backend);
            assert_eq!(descriptor.capabilities.len(), 1);
            let capability = &descriptor.capabilities[0];
            assert_eq!(capability.id, MODEL_RESIDENCY_CAPABILITY);
            assert_eq!(capability.status, expected);
            assert_eq!(
                capability.operations,
                reuses
                    .then(|| vec![AUTOMATIC_REUSE_OPERATION.to_string()])
                    .unwrap_or_default()
            );
            assert!(
                descriptor
                    .capabilities
                    .iter()
                    .all(|capability| !capability.id.starts_with("runtime.state."))
            );
        }
    }

    #[test]
    fn onnx_genai_cache_size_parser_is_bounded_and_defaults_to_one() {
        assert_eq!(parse_onnx_genai_model_cache_size(None), 1);
        assert_eq!(parse_onnx_genai_model_cache_size(Some("")), 1);
        assert_eq!(parse_onnx_genai_model_cache_size(Some("invalid")), 1);
        assert_eq!(parse_onnx_genai_model_cache_size(Some("0")), 0);
        assert_eq!(parse_onnx_genai_model_cache_size(Some("-4")), 0);
        assert_eq!(parse_onnx_genai_model_cache_size(Some("3")), 3);
        assert_eq!(parse_onnx_genai_model_cache_size(Some("99")), 8);
    }

    #[test]
    fn resident_protocol_reuses_exact_model_identity_with_lru_one() {
        let Some(python) = python_for_test() else {
            return;
        };
        let tmp = test_dir("resident-protocol");
        let store = ModelStore::resolve(Some(tmp.clone())).unwrap();
        let backend = OnnxRuntimeBackend::new(store, OnnxRuntimeMode::Cpu);
        let client = CompanionClient::from_embedded_python(
            python,
            FAKE_RESIDENT_ONNX_WORKER,
            "fake resident ONNX worker",
        )
        .with_timeout(Duration::from_secs(3))
        .with_resident_worker();
        let first_manifest = manifest_with_model_path("first", Some("files/model.onnx"));
        let mut second_manifest = first_manifest.clone();
        second_manifest.id = "second".to_string();

        let first = backend
            .generate_with_python_genai_client(
                &first_manifest,
                generate_request("first prompt"),
                Instant::now(),
                &tmp,
                &client,
            )
            .unwrap();
        let first_hit = backend
            .generate_with_python_genai_client(
                &first_manifest,
                generate_request("different prompt"),
                Instant::now(),
                &tmp,
                &client,
            )
            .unwrap();
        let second = backend
            .generate_with_python_genai_client(
                &second_manifest,
                generate_request("second model"),
                Instant::now(),
                &tmp,
                &client,
            )
            .unwrap();
        let evicted = backend
            .generate_with_python_genai_client(
                &first_manifest,
                generate_request("first model again"),
                Instant::now(),
                &tmp,
                &client,
            )
            .unwrap();

        assert_eq!(first.text, "first prompt");
        assert_eq!(first.timings.load_seconds, 0.125);
        assert!(
            first
                .backend_diagnostics
                .contains(&"model_cache_hit: false".to_string())
        );
        assert_eq!(first_hit.text, "different prompt");
        assert_eq!(first_hit.timings.load_seconds, 0.0);
        assert!(
            first_hit
                .backend_diagnostics
                .contains(&"model_cache_hit: true".to_string())
        );
        assert!(
            second
                .backend_diagnostics
                .contains(&"model_cache_hit: false".to_string())
        );
        assert!(
            evicted
                .backend_diagnostics
                .contains(&"model_cache_hit: false".to_string())
        );

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn resident_worker_crash_is_contextual_and_does_not_return_a_result() {
        let Some(python) = python_for_test() else {
            return;
        };
        let tmp = test_dir("resident-crash");
        let store = ModelStore::resolve(Some(tmp.clone())).unwrap();
        let backend = OnnxRuntimeBackend::new(store, OnnxRuntimeMode::Cpu);
        let client = CompanionClient::from_embedded_python(
            python,
            CRASHING_ONNX_WORKER,
            "crashing ONNX worker",
        )
        .with_timeout(Duration::from_secs(3))
        .with_resident_worker();
        let manifest = manifest_with_model_path("crash", Some("files/model.onnx"));

        let error = backend
            .generate_with_python_genai_client(
                &manifest,
                generate_request("never returned"),
                Instant::now(),
                &tmp,
                &client,
            )
            .unwrap_err();

        let message = format!("{error:#}");
        assert!(
            message.contains("ONNX GenAI resident worker failed"),
            "{message}"
        );
        assert!(
            message.contains("closed stdout") || message.contains("terminated with"),
            "{message}"
        );
        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn resident_worker_timeout_is_contextual_and_does_not_return_a_result() {
        let Some(python) = python_for_test() else {
            return;
        };
        let tmp = test_dir("resident-timeout");
        let store = ModelStore::resolve(Some(tmp.clone())).unwrap();
        let backend = OnnxRuntimeBackend::new(store, OnnxRuntimeMode::Cpu);
        let client =
            CompanionClient::from_embedded_python(python, SLOW_ONNX_WORKER, "slow ONNX worker")
                .with_timeout(Duration::from_millis(100))
                .with_resident_worker();
        let manifest = manifest_with_model_path("timeout", Some("files/model.onnx"));

        let error = backend
            .generate_with_python_genai_client(
                &manifest,
                generate_request("never returned"),
                Instant::now(),
                &tmp,
                &client,
            )
            .unwrap_err();

        let message = format!("{error:#}");
        assert!(
            message.contains("ONNX GenAI resident worker failed"),
            "{message}"
        );
        assert!(message.contains("timed out"), "{message}");
        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn onnx_genai_model_dir_uses_manifest_model_parent() {
        let tmp = test_dir("manifest-parent");
        let store = ModelStore::resolve(Some(tmp.clone())).unwrap();
        let model_dir = store
            .model_dir("phi")
            .join("files/cpu_and_mobile/cpu-int4-rtn-block-32");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("genai_config.json"), b"{}").unwrap();
        fs::write(model_dir.join("model.onnx"), b"onnx").unwrap();

        let manifest = manifest_with_model_path(
            "phi",
            Some("files/cpu_and_mobile/cpu-int4-rtn-block-32/model.onnx"),
        );

        assert_eq!(onnx_genai_model_dir(&store, &manifest), Some(model_dir));

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn onnx_genai_model_dir_prefers_cpu_candidate_from_files() {
        let tmp = test_dir("cpu-priority");
        let store = ModelStore::resolve(Some(tmp.clone())).unwrap();
        let root = store.model_dir("phi");
        let cuda_dir = root.join("files/cuda/cuda-fp16");
        let cpu_dir = root.join("files/cpu_and_mobile/cpu-int4-rtn-block-32");
        fs::create_dir_all(&cuda_dir).unwrap();
        fs::create_dir_all(&cpu_dir).unwrap();
        fs::write(cuda_dir.join("genai_config.json"), b"{}").unwrap();
        fs::write(cpu_dir.join("genai_config.json"), b"{}").unwrap();

        let mut manifest = manifest_with_model_path("phi", None);
        manifest.files = vec![
            ModelFile {
                path: "files/cuda/cuda-fp16/genai_config.json".to_string(),
                size: 2,
                checksum: "crc32:0".to_string(),
            },
            ModelFile {
                path: "files/cpu_and_mobile/cpu-int4-rtn-block-32/genai_config.json".to_string(),
                size: 2,
                checksum: "crc32:0".to_string(),
            },
        ];

        assert_eq!(onnx_genai_model_dir(&store, &manifest), Some(cpu_dir));

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn onnx_genai_model_dir_ignores_non_onnx_manifest() {
        let tmp = test_dir("non-onnx");
        let store = ModelStore::resolve(Some(tmp.clone())).unwrap();
        let mut manifest = manifest_with_model_path("phi", Some("files/model.onnx"));
        manifest.format = ModelFormat::SafeTensors;

        assert!(onnx_genai_model_dir(&store, &manifest).is_none());

        let _ = fs::remove_dir_all(tmp);
    }
}
