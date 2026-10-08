//! Resident non-generative text inference. Architecture/task checks precede loading;
//! failed workers are discarded before a compatible fallback allocates weights.
use super::runtime_cache::RuntimeCache;
use crate::{
    capabilities::InferenceTask,
    media_companion::CompanionClient,
    model_store::{ModelFormat, ModelManifest, ModelRuntimeIdentity, ModelStore},
};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    env,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

pub fn install(store: &ModelStore) -> Result<PathBuf> {
    super::python_install::ensure_install_platform("text-analysis")?;
    use std::process::Command;
    let root = store.home().join("backends/text-analysis");
    std::fs::create_dir_all(&root)?;
    let venv = root.join("venv");
    let python = venv.join(if cfg!(windows) {
        "Scripts/python.exe"
    } else {
        "bin/python"
    });
    if !python.is_file() {
        let status = crate::terminal::command_status(
            Command::new(if cfg!(windows) { "python" } else { "python3" })
                .args(["-m", "venv"])
                .arg(&venv),
        )
        .context(
            "Cannot create text-analysis environment; install Python >=3.10 with venv support",
        )?;
        ensure!(
            status.success(),
            "Python venv creation failed; install python3-venv and retry werk backend install text-analysis"
        );
    }
    let requirements = root.join("requirements.txt");
    std::fs::write(
        &requirements,
        include_str!("../../runtime/requirements-text-analysis.txt"),
    )?;
    let status = crate::terminal::command_status(
        Command::new(&python)
            .args(["-m", "pip", "install", "--upgrade", "-r"])
            .arg(&requirements),
    )?;
    ensure!(
        status.success(),
        "Text-analysis dependency installation failed. Check the pip error above, then retry werk backend install text-analysis; alternatively set WERK_TEXT_PYTHON to a compatible environment"
    );
    if cfg!(target_os = "linux") {
        let status = crate::terminal::command_status(Command::new(&python).args([
            "-m",
            "pip",
            "install",
            "tilelang>=0.1.14,<0.2",
        ]))?;
        if !status.success() {
            crate::ui_eprintln!(
                "Optional Laya TileLang acceleration could not be installed; the PyTorch CUDA/CPU fallback remains available. Install laya[fast] in {} to retry.",
                python.display()
            );
        }
    }
    Ok(python)
}

pub fn task_for(model: &ModelManifest) -> Option<InferenceTask> {
    if model.format != ModelFormat::SafeTensors {
        return None;
    }
    let task = match model.architecture.as_deref()? {
        "xlm-roberta" if model.supports_task(InferenceTask::TextReranking) => {
            InferenceTask::TextReranking
        }
        "xlm-roberta" => InferenceTask::TextClassification,
        "embedding_gemma2" => InferenceTask::TextEmbedding,
        "laya" => InferenceTask::TextClassification,
        _ => return None,
    };
    model.supports_task(task).then_some(task)
}

pub fn endpoint(task: InferenceTask) -> &'static str {
    match task {
        InferenceTask::TextEmbedding => "/v1/embeddings",
        InferenceTask::TextReranking => "/v1/rerank",
        InferenceTask::TextClassification => "/v1/classifications",
        _ => "/v1/chat/completions",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub backend: String,
    pub device: String,
    pub fallback_policy: String,
    pub batch_size: usize,
    pub max_length: usize,
    pub checkpoint: String,
    pub truncate: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            backend: "auto".into(),
            device: "auto".into(),
            fallback_policy: "compatible".into(),
            batch_size: 16,
            max_length: 512,
            checkpoint: "auto".into(),
            truncate: false,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            ["auto", "transformers", "vllm", "candle"].contains(&self.backend.as_str()),
            "backend '{}' cannot execute these text-analysis models. Candle supports XLM-RoBERTa sequence classifiers only, not Laya or EmbeddingGemma 2. Use --backend auto or transformers; vLLM supports BGE/EmbeddingGemma 2 with a compatible version.",
            self.backend
        );
        ensure!(
            ["auto", "cuda", "cpu"].contains(&self.device.as_str()),
            "device must be auto, cuda or cpu"
        );
        ensure!(
            ["compatible", "none"].contains(&self.fallback_policy.as_str()),
            "fallback_policy must be compatible or none"
        );
        ensure!(
            (1..=128).contains(&self.batch_size),
            "batch_size must be 1..128; reduce it after CUDA out-of-memory errors"
        );
        ensure!(
            (8..=8192).contains(&self.max_length),
            "max_length must be 8..8192"
        );
        ensure!(
            ["auto", "english", "multilingual", "typed-decisions"]
                .contains(&self.checkpoint.as_str()),
            "checkpoint must be auto, english, multilingual or typed-decisions"
        );
        Ok(())
    }
}

/// Process-wide explicit choices remain hard constraints on API requests.
#[derive(Clone, Default)]
pub struct Policy {
    pub backend: Option<String>,
    pub device: Option<String>,
}
impl Policy {
    pub fn apply(&self, mut options: Options) -> Result<Options> {
        if let Some(backend) = &self.backend {
            ensure!(
                options.backend == "auto" || &options.backend == backend,
                "request backend conflicts with the server --backend constraint"
            );
            options.backend = backend.clone();
            // Explicit server backends must not silently switch runtimes.
            if backend != "transformers" {
                options.fallback_policy = "none".into();
            }
        }
        if let Some(device) = &self.device {
            ensure!(
                options.device == "auto" || &options.device == device,
                "request device conflicts with the server device constraint"
            );
            options.device = device.clone();
        }
        options.validate()?;
        Ok(options)
    }
}

struct Worker {
    key: String,
    client: CompanionClient,
}
#[derive(Debug)]
pub struct InvalidInput(pub String);
impl std::fmt::Display for InvalidInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for InvalidInput {}
#[derive(Clone)]
pub struct TextAnalysisBackend {
    store: ModelStore,
    client: CompanionClient,
    vllm_client: Option<CompanionClient>,
    workers: Arc<RuntimeCache<String, Worker>>,
    native: super::text_analysis_candle::NativeBackend,
    probe_cache: Arc<OnceLock<Value>>,
    locks: Arc<super::runtime_cache::KeyedLocks<String>>,
    routes: Arc<Mutex<HashMap<String, (String, String, Vec<String>)>>>,
}
impl TextAnalysisBackend {
    #[cfg(test)]
    pub(crate) fn with_test_client(
        store: ModelStore,
        client: CompanionClient,
        probe: Value,
    ) -> Self {
        let mut backend = Self::new(store);
        backend.client = client;
        backend.vllm_client = None;
        backend.probe_cache.set(probe).unwrap();
        backend
    }

    pub fn new(store: ModelStore) -> Self {
        let managed = store
            .home()
            .join("backends/text-analysis/venv")
            .join(if cfg!(windows) {
                "Scripts/python.exe"
            } else {
                "bin/python"
            });
        let python = env::var_os("WERK_TEXT_PYTHON")
            .or_else(|| env::var_os("WERK_TRANSFORMERS_PYTHON"))
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if managed.is_file() {
                    managed
                } else {
                    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
                }
            });
        let vllm_python = env::var_os("WERK_VLLM_PYTHON")
            .map(PathBuf::from)
            .or_else(|| {
                let path = super::managed_vllm_dir(&store)
                    .join("venv")
                    .join(if cfg!(windows) {
                        "Scripts/python.exe"
                    } else {
                        "bin/python"
                    });
                path.is_file().then_some(path)
            });
        let vllm_client = vllm_python.map(|python| {
            CompanionClient::from_embedded_python(
                python,
                include_str!("../../runtime/werk_text_analysis.py"),
                "Werk vLLM pooling worker",
            )
        });
        Self {
            store,
            vllm_client,
            client: CompanionClient::from_embedded_python(
                python,
                include_str!("../../runtime/werk_text_analysis.py"),
                "Werk text-analysis worker",
            ),
            workers: Arc::new(RuntimeCache::bounded(2)),
            native: Default::default(),
            probe_cache: Arc::new(OnceLock::new()),
            locks: Arc::new(Default::default()),
            routes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn probe(&self) -> Result<Value> {
        if let Some(probe) = self.probe_cache.get() {
            return Ok(probe.clone());
        }
        let mut probe = self.client.request("probe-model", &json!({})).context(
            "Text-analysis Python unavailable. Install torch, transformers>=5.19, sentence-transformers>=6.1 and laya[serve,fast], then set WERK_TEXT_PYTHON to that environment's Python. For XLM-RoBERTa a native Candle fallback is available; Laya and EmbeddingGemma 2 require their Python runtimes")
            .unwrap_or_else(|error| json!({"cuda":false,"probe_error":format!("{error:#}")}));
        if let Some(client) = &self.vllm_client {
            match client.request("probe-model", &json!({})) {
                Ok(vllm) => {
                    probe["vllm"] = vllm["vllm"].clone();
                    probe["vllm_cuda"] = vllm["cuda"].clone();
                }
                Err(error) => {
                    probe["vllm"] = json!(false);
                    probe["vllm_error"] = json!(format!("{error:#}"));
                }
            }
        }
        let _ = self.probe_cache.set(probe.clone());
        Ok(probe)
    }

    pub fn diagnostics(&self, manifest: &ModelManifest, options: &Options) -> Result<Value> {
        options.validate()?;
        let probe = if options.backend == "candle" {
            json!({"cuda": cfg!(feature="candle-cuda")})
        } else {
            self.probe()
                .unwrap_or_else(|error| json!({"cuda":false,"probe_error":format!("{error:#}")}))
        };
        let (choices, notes) = candidates(manifest, options, &probe)?;
        let arch = manifest.architecture.as_deref().unwrap_or("");
        let ready = choices
            .iter()
            .any(|(runtime, device)| match runtime.as_str() {
                "candle" => {
                    arch == "xlm-roberta" && (device == "cpu" || cfg!(feature = "candle-cuda"))
                }
                "vllm" => {
                    probe["vllm"].as_bool() == Some(true)
                        && device == "cuda"
                        && probe.get("vllm_cuda").unwrap_or(&probe["cuda"]).as_bool() == Some(true)
                }
                _ => {
                    probe["architectures"][arch].as_bool() == Some(true)
                        && (device == "cpu" || probe["cuda"].as_bool() == Some(true))
                }
            });
        Ok(
            json!({"architecture": arch, "ready": ready, "environment":probe,
            "candidates": choices.iter().map(|(runtime,device)|json!({"runtime":runtime,"device":device})).collect::<Vec<_>>(),
            "diagnostics":notes,
            "detail": if ready { "Runtime dependencies found; the first inference validates model weights and memory fit" }
                else { "Required text-analysis dependencies are missing.\n\nInstall dependencies:\n\n    werk backend install text-analysis\n\nThen restart werk serve.\n\nAlternatively, set WERK_TEXT_PYTHON to a compatible Python environment.\nWhen selecting vllm, install or update vLLM separately." }}),
        )
    }

    pub fn execute(
        &self,
        manifest: &ModelManifest,
        task: InferenceTask,
        payload: Value,
        options: &Options,
    ) -> Result<Value> {
        self.execute_observed(manifest, task, payload, options, |_, _| {})
    }

    pub fn execute_observed(
        &self,
        manifest: &ModelManifest,
        task: InferenceTask,
        payload: Value,
        options: &Options,
        mut observe: impl FnMut(&str, Option<(&str, &str)>),
    ) -> Result<Value> {
        options.validate()?;
        ensure!(
            task_for(manifest) == Some(task),
            "model '{}' does not support {}. Its declared tasks are {:?}; use the matching endpoint",
            manifest.id,
            task,
            manifest.metadata.tasks
        );
        let gate = self.locks.get(manifest.id.clone())?;
        observe("waiting for model", None);
        let _guard = gate.lock().map_err(|_| {
            anyhow::anyhow!("text-analysis model lock poisoned; restart werk serve")
        })?;
        observe("resolving runtime", None);
        let probe = if options.backend == "candle" {
            json!({"cuda": cfg!(feature="candle-cuda")})
        } else {
            self.probe()
                .unwrap_or_else(|error| json!({"cuda":false,"probe_error":format!("{error:#}")}))
        };
        let (mut candidates, mut diagnostics) = candidates(manifest, options, &probe)?;
        let route_key = format!(
            "{}:{}:{}",
            manifest.id,
            ModelRuntimeIdentity::from_manifest(manifest)?,
            serde_json::to_string(options)?
        );
        if let Some((runtime, device, notes)) = self.routes.lock().unwrap().get(&route_key) {
            if let Some(index) = candidates
                .iter()
                .position(|candidate| candidate == &(runtime.clone(), device.clone()))
            {
                let preferred = candidates.remove(index);
                candidates.insert(0, preferred);
                diagnostics.extend(notes.clone());
            }
        }
        let mut failures = Vec::new();
        for (runtime, device) in candidates {
            observe("loading / inference", Some((&runtime, &device)));
            let key = format!(
                "{}:{}:{runtime}:{device}:{}:{}",
                manifest.id,
                ModelRuntimeIdentity::from_manifest(manifest)?,
                options.checkpoint,
                options.max_length
            );
            let result = if runtime == "candle" {
                self.native
                    .execute(&self.store, manifest, task, &payload, options, &device)
            } else {
                let (worker, _, _) = self.workers.get_or_try_init(
                    key.clone(),
                    |_| true,
                    || {
                        Ok(Worker {
                            key: key.clone(),
                            client: (if runtime == "vllm" {
                                self.vllm_client.as_ref().unwrap_or(&self.client)
                            } else {
                                &self.client
                            })
                            .clone()
                            .with_resident_worker()
                            .with_model_cache(&self.store, manifest),
                        })
                    },
                )?;
                let request = json!({"model": self.store.model_files_dir(manifest),
                "architecture": manifest.architecture, "task": task.to_string(),
                "runtime": runtime, "device": device, "options": options, "payload": payload});
                let result = worker.client.request("execute", &request);
                drop(worker);
                result
            };
            if let Err(error) = &result {
                if error.is::<InvalidInput>() {
                    return result;
                }
            }
            match result {
                Ok(mut response) if response.get("ok").and_then(Value::as_bool) == Some(true) => {
                    if let Some(message) = response.get("invalid_input").and_then(Value::as_str) {
                        return Err(InvalidInput(message.into()).into());
                    }
                    if let Some(warnings) = response.get("warnings").and_then(Value::as_array) {
                        diagnostics.extend(
                            warnings
                                .iter()
                                .filter_map(Value::as_str)
                                .map(str::to_string),
                        );
                    }
                    for diagnostic in &diagnostics {
                        if crate::logging::enabled() {
                            crate::logging::emit(
                                crate::logging::Level::Warn,
                                "backend.warning",
                                diagnostic,
                                json!({"model":manifest.id,"runtime":runtime,"device":device}),
                            );
                        } else {
                            crate::ui_eprintln!("[werk text-analysis] {diagnostic}");
                        }
                    }
                    let mut routes = self.routes.lock().unwrap();
                    if routes.len() >= 32 {
                        routes.clear();
                    }
                    routes.insert(route_key.clone(), (runtime, device, diagnostics.clone()));
                    response["werk"]["diagnostics"] = json!(diagnostics);
                    return Ok(response);
                }
                result => {
                    self.workers.remove_if(|worker| worker.key == key);
                    let error = match result {
                        Ok(response) => response["error"]["message"]
                            .as_str()
                            .unwrap_or("invalid worker response")
                            .to_string(),
                        Err(error) => format!("{error:#}"),
                    };
                    let detail = format!("{runtime}/{device} failed: {error}");
                    if crate::logging::enabled() {
                        crate::logging::emit(
                            crate::logging::Level::Warn,
                            "backend.attempt_failed",
                            &detail,
                            json!({"model":manifest.id,"runtime":runtime,"device":device}),
                        );
                    } else {
                        crate::ui_eprintln!("[werk text-analysis] {detail}");
                    }
                    diagnostics.push(format!(
                        "{detail}; trying the next compatible runtime if permitted"
                    ));
                    failures.push(detail);
                }
            }
        }
        bail!(
            "No compatible runtime could execute '{}': {}. Mitigation: install the required packages in WERK_TEXT_PYTHON; for CUDA OOM reduce werk.batch_size/max_length or free GPU memory. Use werk.device=cpu (slower) or auto to permit CPU fallback. Candle fallback is limited to XLM-RoBERTa sequence classifiers; other architectures require their Python runtime.",
            manifest.id,
            failures.join("; ")
        )
    }
}

fn candidates(
    model: &ModelManifest,
    options: &Options,
    probe: &Value,
) -> Result<(Vec<(String, String)>, Vec<String>)> {
    let native = model.architecture.as_deref() == Some("xlm-roberta");
    ensure!(
        options.backend != "candle" || native,
        "Candle has no compatible loader for this architecture. Use backend=auto or transformers; install dependencies with werk backend install text-analysis"
    );
    let cuda = probe["cuda"].as_bool().unwrap_or(false)
        || probe["vllm_cuda"].as_bool() == Some(true)
        || native && cfg!(feature = "candle-cuda") && options.backend != "transformers";
    ensure!(
        options.device != "cuda" || cuda,
        "CUDA was explicitly requested but PyTorch cannot access it. Install a CUDA-enabled PyTorch build in WERK_TEXT_PYTHON and check nvidia-smi; select device=auto or cpu to permit CPU execution"
    );
    let device = if options.device == "cpu" || !cuda {
        "cpu"
    } else {
        "cuda"
    };
    let laya = model.architecture.as_deref() == Some("laya");
    let classification = task_for(model) == Some(InferenceTask::TextClassification);
    let mut candidates = Vec::new();
    let mut notes = Vec::new();
    let mut push = |runtime: &str, device: &str| candidates.push((runtime.into(), device.into()));
    if options.backend == "candle" {
        push("candle", device);
        if device == "cuda" && options.device == "auto" && options.fallback_policy != "none" {
            push("candle", "cpu");
        }
        return Ok((candidates, notes));
    }
    if let Some(error) = probe["probe_error"].as_str() {
        notes.push(error.to_string());
    }
    if options.backend == "vllm" {
        ensure!(
            !classification,
            "This classification architecture uses its dedicated PyTorch runtime. Use backend=auto or transformers"
        );
        push("vllm", device);
    } else if options.backend == "auto"
        && device == "cuda"
        && !classification
        && probe["vllm"].as_bool() == Some(true)
        && probe["wsl"].as_bool() != Some(true)
    {
        push("vllm", device);
    }
    if options.backend != "vllm" || options.fallback_policy != "none" {
        if laya && device == "cuda" {
            if probe["tilelang"].as_bool() == Some(true) {
                push("laya-fast", device);
            } else {
                notes.push("Laya TileLang fast path unavailable; using PyTorch CUDA. Install laya[fast] in WERK_TEXT_PYTHON to enable fused kernels.".into());
            }
        }
        push(if laya { "laya" } else { "transformers" }, device);
        if !laya && options.fallback_policy != "none" {
            push("transformers-eager", device);
        }
        if native
            && options.backend == "auto"
            && options.fallback_policy != "none"
            && (device != "cuda" || cfg!(feature = "candle-cuda"))
        {
            push("candle", device);
        }
        if options.device == "auto" && device != "cpu" && options.fallback_policy != "none" {
            push(if laya { "laya" } else { "transformers" }, "cpu");
            if !laya {
                push("transformers-eager", "cpu");
                if native && options.backend == "auto" {
                    push("candle", "cpu");
                }
            }
        }
    }
    if options.fallback_policy == "none" {
        candidates.truncate(1);
    }
    if options.device == "auto" && device == "cpu" {
        notes.push("CUDA is unavailable in the text-analysis Python environment; using CPU. Install CUDA-enabled torch and set WERK_TEXT_PYTHON to use GPU acceleration.".into());
    }
    Ok((candidates, notes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(
        root: &std::path::Path,
        architecture: &str,
        task: InferenceTask,
    ) -> (ModelStore, ModelManifest) {
        let source = root.join("source");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("model.safetensors"), b"fixture").unwrap();
        std::fs::write(source.join("config.json"), "{}").unwrap();
        let store = ModelStore::resolve(Some(root.join("store"))).unwrap();
        let mut model = store
            .import_path(&source, "another-organization/custom-finetune")
            .unwrap();
        model.architecture = Some(architecture.into());
        model.metadata.tasks = vec![task];
        (store, model)
    }

    #[test]
    fn architecture_selection_does_not_depend_on_repository_name() {
        let dir = tempfile::tempdir().unwrap();
        let (_, mut model) = model(dir.path(), "xlm-roberta", InferenceTask::TextReranking);
        assert_eq!(task_for(&model), Some(InferenceTask::TextReranking));
        model.metadata.tasks = vec![InferenceTask::TextGeneration];
        assert_eq!(task_for(&model), None);
        let options = Options::default();
        assert!(
            Policy {
                backend: Some("onnx".into()),
                device: None
            }
            .apply(options)
            .unwrap_err()
            .to_string()
            .contains("cannot execute")
        );
    }

    #[test]
    fn explicit_device_and_no_fallback_are_honored() {
        let dir = tempfile::tempdir().unwrap();
        let (_, model) = model(dir.path(), "embedding_gemma2", InferenceTask::TextEmbedding);
        let probe = json!({"cuda": true, "vllm": true, "wsl": false});
        let mut options = Options {
            device: "cpu".into(),
            ..Default::default()
        };
        assert!(
            candidates(&model, &options, &probe)
                .unwrap()
                .0
                .iter()
                .all(|(_, device)| device == "cpu")
        );
        options.device = "cuda".into();
        options.fallback_policy = "none".into();
        assert_eq!(
            candidates(&model, &options, &probe).unwrap().0,
            vec![("vllm".into(), "cuda".into())]
        );
        assert!(
            candidates(&model, &options, &json!({"cuda":false}))
                .unwrap_err()
                .to_string()
                .contains("CUDA was explicitly requested")
        );
        options.fallback_policy = "compatible".into();
        assert!(
            candidates(&model, &options, &probe)
                .unwrap()
                .0
                .iter()
                .all(|(_, device)| device == "cuda")
        );
    }

    #[test]
    fn readiness_requires_dependencies_for_an_allowed_candidate() {
        let dir = tempfile::tempdir().unwrap();
        let (store, model) = model(dir.path(), "embedding_gemma2", InferenceTask::TextEmbedding);
        let backend = TextAnalysisBackend::new(store);
        backend
            .probe_cache
            .set(json!({"cuda":true,"vllm":false,
            "architectures":{"embedding_gemma2":true}}))
            .unwrap();
        let mut options = Options {
            backend: "vllm".into(),
            fallback_policy: "none".into(),
            ..Default::default()
        };
        assert_eq!(
            backend.diagnostics(&model, &options).unwrap()["ready"],
            false
        );
        options.fallback_policy = "compatible".into();
        assert_eq!(
            backend.diagnostics(&model, &options).unwrap()["ready"],
            true
        );
        options.backend = "candle".into();
        assert!(
            backend
                .diagnostics(&model, &options)
                .unwrap_err()
                .to_string()
                .contains("no compatible loader")
        );
    }

    #[test]
    fn failed_cuda_workers_are_replaced_and_successful_fallback_is_reused() {
        let dir = tempfile::tempdir().unwrap();
        let (store, model) = model(dir.path(), "xlm-roberta", InferenceTask::TextReranking);
        let log = dir.path().join("calls");
        let script = r#"
import json,sys
log=sys.argv[1]
def call(op,p):
 if op=='probe-model': return {'ok':True,'cuda':True,'vllm':False}
 if op=='transport-handshake': return {'ok':True}
 with open(log,'a') as f: f.write(p['runtime']+'/'+p['device']+'\n')
 if p['device']=='cuda': return {'ok':False,'error':{'code':'oom','message':'CUDA out of memory'}}
 return {'ok':True,'results':[], 'werk':{'runtime':p['runtime'],'device':p['device']}}
if sys.argv[-1]=='serve':
 for line in sys.stdin:
  f=json.loads(line)
  print(json.dumps({'transport_version':1,'request_id':f['request_id'],'response':call(f['operation'],f['payload'])}),flush=True)
else: print(json.dumps(call(sys.argv[-1],json.load(sys.stdin))))
"#;
        let mut backend = TextAnalysisBackend::new(store);
        backend.client = CompanionClient::from_command(
            "python3",
            ["-c".into(), script.into(), log.into_os_string()],
        );
        backend.vllm_client = None;
        let options = Options {
            backend: "transformers".into(),
            ..Default::default()
        };
        let first = backend
            .execute(&model, InferenceTask::TextReranking, json!({}), &options)
            .unwrap();
        assert_eq!(first["werk"]["device"], "cpu");
        assert_eq!(first["werk"]["diagnostics"].as_array().unwrap().len(), 2);
        backend
            .execute(&model, InferenceTask::TextReranking, json!({}), &options)
            .unwrap();
        let log = std::fs::read_to_string(dir.path().join("calls")).unwrap();
        assert_eq!(
            log.lines().collect::<Vec<_>>(),
            vec![
                "transformers/cuda",
                "transformers-eager/cuda",
                "transformers/cpu",
                "transformers/cpu"
            ]
        );
    }
}
