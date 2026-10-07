//! Server-owned, immutable single-host execution plans. No request supplies argv.
use crate::{
    backend::{
        GenerationBackend, LlamaCppMode, LlamaRuntimeOptions, LlamaServerBackend, VllmBackend,
    },
    inference_service::devices::{Device, Inventory},
    model_store::{ModelFormat, ModelManifest, ModelRuntimeIdentity, ModelStore},
};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Runtime {
    LlamaCuda,
    LlamaCudaOffload,
    VllmCuda,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    Single,
    Layer,
    Tensor,
    Pipeline,
    Expert,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Offload {
    #[default]
    None,
    /// Static layer computation on CPU; this is not weight streaming.
    CpuLayers {
        gpu_layers: u32,
    },
    CpuExperts {
        layers: u32,
    },
    CpuWeights {
        gib_per_gpu: f64,
    },
    ExpertCache {
        slots: u32,
        pinned_mib: u64,
    },
    HelperGpu,
    SsdStreaming,
}

/// All fields are bytes, per device, and are operator estimates, not telemetry.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct DeviceBudget {
    pub weights: u64,
    #[serde(default)]
    pub replicated: u64,
    pub kv: u64,
    #[serde(default)]
    pub experts: u64,
    pub compute: u64,
    pub overhead: u64,
    pub reserve: u64,
}
impl DeviceBudget {
    pub fn total(&self) -> Result<u64> {
        [
            self.weights,
            self.replicated,
            self.kv,
            self.experts,
            self.compute,
            self.overhead,
            self.reserve,
        ]
        .into_iter()
        .try_fold(0u64, |a, b| {
            a.checked_add(b).context("device budget overflow")
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub id: String,
    pub alias: String,
    pub model: String,
    pub runtime: Runtime,
    pub executable: PathBuf,
    /// Source revision / wheel build identifier; kept alongside binary digest.
    pub build: String,
    pub gpus: Vec<String>,
    pub strategy: Strategy,
    #[serde(default = "one")]
    pub tensor_parallel: usize,
    #[serde(default = "one")]
    pub pipeline_parallel: usize,
    #[serde(default)]
    pub split: Vec<f64>,
    pub context: usize,
    pub batch: usize,
    #[serde(default = "one")]
    pub parallel: usize,
    #[serde(default = "one")]
    pub cpu_threads: usize,
    #[serde(default)]
    pub offload: Offload,
    #[serde(default)]
    pub kv_cpu: bool,
    pub memory: Vec<DeviceBudget>,
    pub host_bytes: u64,
    #[serde(default)]
    pub native_args: Vec<String>,
}
fn one() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    pub version: u32,
    pub profiles: Vec<Profile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub profile: Profile,
    pub model_identity: String,
    pub executable_sha256: String,
    pub runtime_version: String,
    pub devices: Vec<Device>,
    pub strategy: Strategy,
    pub args: Vec<String>,
    pub environment: BTreeMap<String, String>,
    pub fingerprint: String,
    pub reasons: Vec<String>,
}

pub fn binary_digest(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("cannot read runtime {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

impl Configuration {
    pub fn load(path: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            config.version == 1,
            "unsupported deployment configuration version"
        );
        ensure!(
            !config.profiles.is_empty() && config.profiles.len() <= 64,
            "deployment configuration must contain 1..64 profiles"
        );
        Ok(config)
    }
    pub fn resolve(&self, store: &ModelStore, inventory: &Inventory) -> Result<Vec<Plan>> {
        ensure!(
            self.version == 1,
            "unsupported deployment configuration version"
        );
        ensure!(
            !self.profiles.is_empty() && self.profiles.len() <= 64,
            "deployment configuration must contain 1..64 profiles"
        );
        let mut ids = BTreeSet::new();
        let mut used = BTreeSet::new();
        let mut aliases = BTreeMap::new();
        let mut plans = Vec::new();
        let mut host = 0u64;
        for profile in &self.profiles {
            ensure!(
                ids.insert(profile.id.clone()),
                "duplicate deployment ID {}",
                profile.id
            );
            let manifest = store.get(&profile.model)?;
            if profile.runtime != Runtime::VllmCuda {
                LlamaServerBackend::validate_deployment_offload(store, &manifest, profile)?;
            }
            validate_model_parallel(store, profile, &manifest)?;
            if let Some(model) = aliases.insert(&profile.alias, &profile.model) {
                ensure!(
                    model == &profile.model,
                    "replicas sharing an alias must use the same model"
                );
            }
            // Automatic placement must consider devices already claimed by earlier profiles.
            let mut available = inventory.clone();
            for device in &mut available.devices {
                if used.contains(&device.id) {
                    device.visible_index = None;
                }
            }
            let plan = profile.resolve(&manifest, &available)?;
            for device in &plan.devices {
                ensure!(
                    used.insert(device.id.clone()),
                    "GPU {} overlaps independent profiles; sharing is unsupported",
                    device.id
                );
            }
            host = host
                .checked_add(profile.host_bytes)
                .context("host budget overflow")?;
            plans.push(plan);
        }
        if host > 0 {
            ensure!(
                host <= inventory
                    .host_available_bytes
                    .context("host RAM availability is unknown")?,
                "combined host budgets exceed available RAM"
            );
        }
        Ok(plans)
    }
}

impl Profile {
    pub fn resolve(&self, manifest: &ModelManifest, inventory: &Inventory) -> Result<Plan> {
        ensure!(
            self.id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.')),
            "deployment IDs require ASCII letters, digits, dot, dash or underscore"
        );
        ensure!(
            inventory.memory_topology != Some(crate::inference::MemoryTopology::Unified),
            "unified-memory deployment admission is not implemented; use the existing platform backend path to avoid double-counting RAM and GPU memory"
        );
        ensure!(
            !self.id.is_empty()
                && self.id.len() <= 128
                && !self.alias.is_empty()
                && self.alias.len() <= 128,
            "deployment IDs and aliases must be 1..128 bytes"
        );
        ensure!(
            self.executable.is_absolute() && !self.build.trim().is_empty(),
            "profile needs an absolute executable and explicit build identity"
        );
        ensure!(
            self.context > 0
                && self.batch > 0
                && self.parallel > 0
                && self.parallel <= u32::MAX as usize
                && self.cpu_threads > 0
                && self.cpu_threads <= u32::MAX as usize,
            "context, batch, parallel and CPU threads must be positive and in range"
        );
        ensure!(
            self.context.checked_mul(self.parallel).is_some(),
            "context capacity overflow"
        );
        ensure!(
            self.host_bytes > 0,
            "profile must budget host RAM, runtime and transfer buffers"
        );
        let expected = if self.runtime == Runtime::VllmCuda {
            ModelFormat::SafeTensors
        } else {
            ModelFormat::Gguf
        };
        ensure!(
            manifest.format == expected,
            "profile runtime is incompatible with model format"
        );
        for name in ["WERK_LLAMA_ARGS", "WERK_VLLM_ARGS"] {
            ensure!(
                std::env::var(name).unwrap_or_default().trim().is_empty(),
                "{name} conflicts with managed deployments; move permitted options into native_args"
            );
        }
        validate_native_args(self.runtime, &self.native_args)?;
        let devices: Vec<Device> = if self.strategy == Strategy::Auto && self.gpus.is_empty() {
            ensure!(
                self.memory.len() == 1,
                "auto requires one explicit per-GPU budget"
            );
            let needed = self.memory[0].total()?;
            vec![inventory.visible().into_iter().filter(|d| d.free_bytes.is_some_and(|b| b >= needed))
                .min_by_key(|d| (d.total_bytes, d.id.clone())).context("no single visible GPU fits the declared auto budget; configure an explicit group")?.clone()]
        } else {
            ensure!(
                !self.gpus.is_empty(),
                "explicit placement requires GPU UUIDs"
            );
            let mut seen = BTreeSet::new();
            self.gpus.iter().map(|id| {
                ensure!(seen.insert(id), "duplicate GPU in profile");
                inventory.devices.iter().find(|d| &d.id == id && d.visible_index.is_some()).cloned()
                    .with_context(|| format!("GPU {id} is unknown or outside inherited visibility; full UUIDs are required"))
            }).collect::<Result<_>>()?
        };
        ensure!(
            devices.len() == self.memory.len(),
            "memory must contain one budget per selected GPU, in the same order"
        );
        for (device, budget) in devices.iter().zip(&self.memory) {
            ensure!(
                device.runtime == "cuda",
                "selected runtime requires proven CUDA devices"
            );
            ensure!(
                budget.reserve > 0 && budget.overhead > 0 && budget.compute > 0,
                "each device needs compute, overhead and reserve budgets"
            );
            ensure!(
                budget.total()? <= device.free_bytes.context("GPU free memory is unknown")?,
                "GPU {} cannot fit its declared budget",
                device.id
            );
        }
        let strategy = if self.strategy == Strategy::Auto {
            ensure!(
                devices.len() == 1,
                "auto does not invent a multi-GPU strategy; select layer, tensor, pipeline or expert explicitly"
            );
            Strategy::Single
        } else {
            self.strategy
        };
        if strategy == Strategy::Single {
            ensure!(devices.len() == 1, "single uses exactly one GPU");
        }
        let mut args = Vec::<String>::new();
        let mut add = |flag: &str, value: String| {
            args.extend([flag.into(), value]);
        };
        if self.runtime != Runtime::VllmCuda {
            ensure!(
                matches!(strategy, Strategy::Single | Strategy::Layer),
                "llama profiles currently support single and layer; row/tensor require separate model/build combination validation"
            );
            ensure!(
                self.tensor_parallel == 1 && self.pipeline_parallel == 1,
                "vLLM TP/PP parameters are not llama tensor-split ratios"
            );
            add(
                "--device",
                (0..devices.len())
                    .map(|i| format!("CUDA{i}"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            add(
                "--split-mode",
                if strategy == Strategy::Single {
                    "none"
                } else {
                    "layer"
                }
                .into(),
            );
            add("--main-gpu", "0".into());
            if strategy == Strategy::Layer && self.split.is_empty() {
                ensure!(
                    self.memory.iter().all(|b| b.weights > 0),
                    "layer placement requires positive resident weight estimates on every GPU"
                );
                add(
                    "--tensor-split",
                    self.memory
                        .iter()
                        .map(|b| b.weights.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                );
            } else if !self.split.is_empty() {
                ensure!(
                    strategy == Strategy::Layer
                        && self.split.len() == devices.len()
                        && self.split.iter().all(|v| v.is_finite() && *v > 0.0),
                    "layer split requires one finite positive ratio per GPU"
                );
                add(
                    "--tensor-split",
                    self.split
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                );
            }
            match self.offload {
                Offload::None => {}
                Offload::CpuLayers { gpu_layers } => {
                    ensure!(
                        gpu_layers <= i32::MAX as u32,
                        "GPU layer count is out of range"
                    );
                }
                Offload::CpuExperts { layers } => {
                    ensure!(layers > 0, "CPU expert layer count must be positive");
                    add("--n-cpu-moe", layers.to_string());
                }
                Offload::ExpertCache { slots, pinned_mib } => {
                    ensure!(
                        self.runtime == Runtime::LlamaCudaOffload && slots > 0,
                        "expert cache requires the pinned offload flavor and nonzero slots"
                    );
                    ensure!(
                        self.build == "907a73da9a149faa8c42ccde890f1d575586810f",
                        "expert cache is restricted to the reviewed Werk offload pin"
                    );
                    ensure!(
                        self.memory.iter().all(|b| b.experts > 0),
                        "expert cache must be budgeted on every GPU"
                    );
                    ensure!(
                        pinned_mib
                            .checked_mul(1024 * 1024)
                            .is_some_and(|b| b <= self.host_bytes),
                        "pinned cache exceeds host budget"
                    );
                    add("--moe-expert-cache-size", slots.to_string());
                    add("--moe-expert-cache-host-pinned-mb", pinned_mib.to_string());
                }
                _ => bail!(
                    "this llama build/profile does not support the requested offload mechanism"
                ),
            }
        } else {
            ensure!(
                self.split.is_empty() && !self.kv_cpu,
                "llama split ratios and CPU KV controls do not apply to vLLM"
            );
            ensure!(
                self.tensor_parallel > 0
                    && self.pipeline_parallel > 0
                    && self.tensor_parallel.checked_mul(self.pipeline_parallel)
                        == Some(devices.len()),
                "TP * PP must equal selected GPU count"
            );
            match strategy {
                Strategy::Single => ensure!(
                    self.tensor_parallel == 1 && self.pipeline_parallel == 1,
                    "single requires TP=PP=1"
                ),
                Strategy::Tensor => ensure!(
                    self.tensor_parallel > 1 && self.pipeline_parallel == 1,
                    "tensor requires TP>1 and PP=1"
                ),
                Strategy::Pipeline => ensure!(self.pipeline_parallel > 1, "pipeline requires PP>1"),
                Strategy::Expert => ensure!(
                    self.tensor_parallel > 1
                        && self.pipeline_parallel == 1
                        && matches!(self.offload, Offload::None),
                    "expert requires TP>1, PP=1, no combined offload"
                ),
                _ => bail!("vLLM does not implement llama layer split"),
            }
            add("--tensor-parallel-size", self.tensor_parallel.to_string());
            add(
                "--pipeline-parallel-size",
                self.pipeline_parallel.to_string(),
            );
            add("--distributed-executor-backend", "mp".into());
            add("--max-model-len", self.context.to_string());
            add("--max-num-seqs", self.parallel.to_string());
            add("--max-num-batched-tokens", self.batch.to_string());
            // vLLM applies one utilization fraction to all ranks. Use the smallest
            // fraction, never aggregate VRAM, and include the reserved headroom.
            let fraction = devices
                .iter()
                .zip(&self.memory)
                .map(|(d, b)| Ok((b.total()? - b.reserve) as f64 / d.total_bytes as f64))
                .collect::<Result<Vec<f64>>>()?
                .into_iter()
                .fold(1.0f64, f64::min);
            ensure!(
                fraction > 0.0 && fraction < 1.0,
                "invalid vLLM memory fraction"
            );
            add("--gpu-memory-utilization", format!("{fraction:.8}"));
            match self.offload {
                Offload::None => {}
                Offload::CpuWeights { gib_per_gpu } => {
                    ensure!(
                        gib_per_gpu.is_finite()
                            && gib_per_gpu > 0.0
                            && gib_per_gpu * devices.len() as f64 * 1073741824.0
                                <= self.host_bytes as f64,
                        "CPU weight offload exceeds host budget"
                    );
                    add("--cpu-offload-gb", gib_per_gpu.to_string());
                }
                _ => bail!("requested vLLM offload mechanism has no validated profile translation"),
            }
            if strategy == Strategy::Expert {
                args.push("--enable-expert-parallel".into());
            }
        }
        args.extend(self.native_args.clone());
        let environment = BTreeMap::from([
            (
                "CUDA_VISIBLE_DEVICES".into(),
                devices
                    .iter()
                    .map(|d| d.id.clone())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            ("CUDA_DEVICE_ORDER".into(), "PCI_BUS_ID".into()),
            ("OMP_NUM_THREADS".into(), self.cpu_threads.to_string()),
        ]);
        let mut plan = Plan {
            profile: self.clone(),
            model_identity: ModelRuntimeIdentity::from_manifest(manifest)?.to_string(),
            executable_sha256: binary_digest(&self.executable)?,
            runtime_version: runtime_version(&self.executable)?,
            devices, strategy, args, environment,
            fingerprint: String::new(),
            reasons: vec![
                "Exclusive GPU ownership; component budgets are declared estimates, not measured usage or hard allocator limits.".into(),
                "Unknown P2P/NVLink is not treated as a prerequisite; no throughput prediction was made.".into(),
            ],
        };
        if self.strategy == Strategy::Auto {
            plan.reasons.push("Selected the smallest fitting single GPU; additional devices lack a measured performance benefit.".into());
        }
        let mut command = Command::new(&self.executable);
        if self.runtime == Runtime::VllmCuda {
            command.arg("serve");
        }
        let help = command
            .arg("--help")
            .envs(&plan.environment)
            .output()
            .context("runtime capability probe failed")?;
        ensure!(help.status.success(), "runtime --help probe failed");
        let help = format!(
            "{}\n{}",
            String::from_utf8_lossy(&help.stdout),
            String::from_utf8_lossy(&help.stderr)
        );
        plan.validate_help(&help)?;
        // Free memory and topology observations change between starts and are
        // evidence, not execution identity. Do not invalidate KV for a scrape.
        plan.fingerprint = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&serde_json::json!({
                "profile": plan.profile, "model": plan.model_identity, "binary": plan.executable_sha256, "runtime_version": plan.runtime_version,
                "devices": plan.devices.iter().map(|d| (&d.id, &d.pci_bus_id, d.total_bytes)).collect::<Vec<_>>(),
                "strategy": plan.strategy, "args": plan.args, "environment": plan.environment,
            }))?)
        );
        Ok(plan)
    }
}

fn validate_native_args(runtime: Runtime, args: &[String]) -> Result<()> {
    // Restrictive by design: native config files and CLI abbreviations could
    // bypass visibility, worker ports and budgeting even with a deny-list.
    for arg in args {
        let allowed = match runtime {
            Runtime::VllmCuda => ["--enforce-eager", "--disable-log-requests"].as_slice(),
            _ => ["--no-warmup", "--jinja", "--log-disable"].as_slice(),
        };
        ensure!(
            allowed.contains(&arg.as_str()),
            "native option {arg:?} is not permitted in a managed deployment; process/device/memory options must be structured"
        );
    }
    Ok(())
}

fn validate_model_parallel(
    store: &ModelStore,
    profile: &Profile,
    manifest: &ModelManifest,
) -> Result<()> {
    if profile.runtime != Runtime::VllmCuda
        || profile.tensor_parallel == 1 && profile.pipeline_parallel == 1
    {
        return Ok(());
    }
    let path = manifest
        .config_path
        .as_deref()
        .context("vLLM parallel placement requires local model config metadata")?;
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(store.absolute_model_file(manifest, path))?)?;
    let config = config.get("text_config").unwrap_or(&config);
    ensure!(
        config
            .get("quantization_config")
            .is_none_or(serde_json::Value::is_null),
        "quantized vLLM TP/PP/EP combinations need a model/build-specific validation and are not enabled in managed profiles yet"
    );
    let heads = config
        .get("num_attention_heads")
        .and_then(serde_json::Value::as_u64)
        .context("num_attention_heads is required for TP planning")?;
    let tp = profile.tensor_parallel as u64;
    ensure!(
        tp > 0 && heads > 0 && heads % tp == 0,
        "attention heads must divide evenly across TP ranks"
    );
    if let Some(kv) = config
        .get("num_key_value_heads")
        .and_then(serde_json::Value::as_u64)
    {
        ensure!(
            kv > 0 && (if kv >= tp { kv % tp == 0 } else { tp % kv == 0 }),
            "KV heads cannot be partitioned or replicated over this TP group"
        );
    }
    if profile.pipeline_parallel > 1 {
        let layers = config
            .get("num_hidden_layers")
            .and_then(serde_json::Value::as_u64)
            .context("num_hidden_layers is required for PP planning")?;
        ensure!(
            layers >= profile.pipeline_parallel as u64,
            "PP exceeds model layer count"
        );
    }
    if profile.strategy == Strategy::Expert {
        let experts = config
            .get("num_experts")
            .or_else(|| config.get("num_local_experts"))
            .or_else(|| config.get("n_routed_experts"))
            .and_then(serde_json::Value::as_u64)
            .context("EP requires a model with routed expert metadata")?;
        ensure!(
            experts > 0 && experts % tp == 0,
            "experts must divide evenly across EP ranks"
        );
    }
    Ok(())
}

impl Plan {
    pub(crate) fn capability(&self) -> crate::werk_protocol::Capability {
        crate::werk_protocol::Capability {
            id: "runtime.placement".into(),
            status: crate::werk_protocol::CapabilityStatus::Supported,
            detail: format!(
                "Deployment {}: {:?}, {} CUDA devices; build {}, fingerprint {}. Estimates and native-start configuration only; no live weight/expert migration.",
                self.profile.id,
                self.strategy,
                self.devices.len(),
                self.profile.build,
                self.fingerprint
            ),
            operations: vec!["inspect".into()],
        }
    }
    fn validate_help(&self, help: &str) -> Result<()> {
        let tokens: BTreeSet<_> = help
            .split(|c: char| c.is_whitespace() || matches!(c, ',' | '=' | '[' | ']' | '(' | ')'))
            .collect();
        for arg in self.args.iter().filter(|s| s.starts_with("--")) {
            ensure!(
                tokens.contains(arg.as_str()),
                "selected build does not advertise {arg}"
            );
        }
        if self.profile.runtime == Runtime::LlamaCudaOffload {
            ensure!(
                tokens.contains("--moe-expert-cache-size"),
                "offload flavor does not advertise native expert cache support"
            );
        }
        Ok(())
    }
    pub fn verify_binary(&self) -> Result<()> {
        ensure!(
            binary_digest(&self.profile.executable)? == self.executable_sha256,
            "runtime binary changed since plan resolution; reload profiles"
        );
        ensure!(
            runtime_version(&self.profile.executable)? == self.runtime_version,
            "runtime version changed since plan resolution; reload profiles"
        );
        Ok(())
    }
    pub fn apply(&self, command: &mut Command) {
        command.envs(&self.environment);
    }
    pub fn llama_options(&self) -> LlamaRuntimeOptions {
        LlamaRuntimeOptions {
            gpu_layers: match self.profile.offload {
                Offload::CpuLayers { gpu_layers } => Some(gpu_layers as i32),
                _ => None,
            },
            ctx_size: Some(self.profile.context),
            batch_size: Some(self.profile.batch),
            parallel: Some(self.profile.parallel as u32),
            threads: Some(self.profile.cpu_threads as u32),
            threads_batch: Some(self.profile.cpu_threads as u32),
            kv_offload: Some(!self.profile.kv_cpu),
            ..Default::default()
        }
    }
}

fn runtime_version(executable: &Path) -> Result<String> {
    let output = Command::new(executable)
        .arg("--version")
        .output()
        .context("runtime version probe failed")?;
    ensure!(output.status.success(), "runtime --version probe failed");
    let text = if output.stdout.is_empty() {
        String::from_utf8_lossy(&output.stderr).into_owned()
    } else {
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    ensure!(
        !text.trim().is_empty() && text.len() <= 16384,
        "invalid runtime version response"
    );
    // vLLM may log timestamped platform discovery before argparse prints its
    // version. llama prints a version line followed by compiler information.
    let version = text
        .lines()
        .find(|line| line.trim_start().starts_with("version:"))
        .or_else(|| text.lines().rev().find(|line| !line.trim().is_empty()))
        .context("runtime version is empty")?;
    Ok(version.trim().into())
}

#[derive(Clone)]
pub(crate) struct Execution {
    pub plan: Arc<Plan>,
    pub resources: Arc<crate::inference_service::resources::Reservations>,
    #[cfg(test)]
    pub inventory: Option<Inventory>,
}
impl Execution {
    pub fn backend(&self, store: ModelStore) -> Arc<dyn GenerationBackend> {
        match self.plan.profile.runtime {
            Runtime::LlamaCuda | Runtime::LlamaCudaOffload => Arc::new(
                LlamaServerBackend::new(store, LlamaCppMode::Cuda, self.plan.llama_options())
                    .with_deployment(self.clone()),
            ),
            Runtime::VllmCuda => Arc::new(VllmBackend::new(store).with_deployment(self.clone())),
        }
    }
    pub fn reserve(&self) -> Result<crate::inference_service::resources::Reservation> {
        self.plan.verify_binary()?;
        if self.plan.profile.runtime != Runtime::VllmCuda {
            let mut command = Command::new(&self.plan.profile.executable);
            self.plan.apply(&mut command);
            let output = command.arg("--list-devices").output()?;
            ensure!(output.status.success(), "native CUDA device probe failed");
            let devices = format!(
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            for i in 0..self.plan.devices.len() {
                ensure!(
                    devices
                        .lines()
                        .any(|line| line.trim_start().starts_with(&format!("CUDA{i}:"))),
                    "selected runtime build cannot see logical CUDA{i} under the reserved UUID mask"
                );
            }
        }
        #[cfg(test)]
        if let Some(inventory) = &self.inventory {
            return self.resources.acquire(&self.plan, inventory);
        }
        self.resources.acquire(&self.plan, &Inventory::detect()?)
    }
}

#[cfg(test)]
pub(crate) mod tests;
