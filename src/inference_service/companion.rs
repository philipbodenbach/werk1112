use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::Duration,
};

use super::{
    backend::{BackendExecution, BackendOutput, BackendProbe, MediaInferenceBackend},
    output::ensure_output_path,
    resources::{configured_media_accelerator, detected_accelerator},
};
use crate::{
    backend::{BackendAccelerator, BackendRuntime, RuntimeId, runtime_registry},
    capabilities::{InferenceTask, RepositoryLayout},
    inference::{
        EffectiveInferenceRequest, InferenceRuntimeCandidate, ParameterSource,
        ParameterSupportStatus, ParameterValue, RuntimeAccelerator, WorkloadEstimate,
    },
    media_companion::{CompanionClient, CompanionExecution, CompanionHealth, CompanionOutput},
    model_store::{ModelManifest, ModelStore},
};

#[derive(Debug, Clone)]
pub struct CompanionMediaBackend {
    client: std::result::Result<CompanionClient, String>,
    health_cache: Arc<OnceLock<std::result::Result<CompanionHealth, String>>>,
}

impl CompanionMediaBackend {
    pub fn discover() -> Self {
        Self {
            client: CompanionClient::discover().map_err(|error| error.to_string()),
            health_cache: Arc::new(OnceLock::new()),
        }
    }

    pub fn with_client(client: CompanionClient) -> Self {
        Self {
            client: Ok(client),
            health_cache: Arc::new(OnceLock::new()),
        }
    }

    fn health(&self, client: &CompanionClient) -> std::result::Result<CompanionHealth, String> {
        self.health_cache
            .get_or_init(|| client.health().map_err(|error| error.to_string()))
            .clone()
    }
}

impl Default for CompanionMediaBackend {
    fn default() -> Self {
        Self::discover()
    }
}

impl MediaInferenceBackend for CompanionMediaBackend {
    fn probe(
        &self,
        store: &ModelStore,
        manifest: &ModelManifest,
        task: InferenceTask,
        schema_paths: &[String],
    ) -> BackendProbe {
        let client = match &self.client {
            Ok(client) => client,
            Err(error) => {
                return BackendProbe {
                    available: false,
                    detail: error.clone(),
                    candidates: default_companion_candidates(
                        false,
                        Some(error.clone()),
                        schema_paths,
                    ),
                    parameter_support: default_companion_parameter_support(schema_paths),
                };
            }
        };
        let health = match self.health(client) {
            Ok(health) => health,
            Err(error) => {
                return BackendProbe {
                    available: false,
                    detail: error.clone(),
                    candidates: default_companion_candidates(false, Some(error), schema_paths),
                    parameter_support: default_companion_parameter_support(schema_paths),
                };
            }
        };
        let probe_request = json!({
            "model_path": companion_model_path(store, manifest),
            "task": task.to_string(),
            "layout": manifest.metadata.repository_layout.to_string(),
            "family": manifest.metadata.family,
            "architecture": manifest.architecture,
        });
        let model_probe = client.probe_model(&probe_request);
        let (available, detail) = match model_probe {
            Ok(value) => (
                value
                    .get("supported")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                value
                    .get("detail")
                    .and_then(Value::as_str)
                    .unwrap_or("companion model probe succeeded")
                    .to_string(),
            ),
            Err(error) => (false, error.to_string()),
        };
        let parameter_support = companion_parameter_support(schema_paths, Some(&health));
        BackendProbe {
            available,
            detail: detail.clone(),
            candidates: companion_candidates(
                available,
                (!available).then_some(detail),
                schema_paths,
                Some(&health),
            ),
            parameter_support,
        }
    }

    fn execute(
        &self,
        store: &ModelStore,
        manifest: &ModelManifest,
        request: &EffectiveInferenceRequest,
        output_dir: &Path,
        runtime: &str,
    ) -> Result<BackendExecution> {
        let client = self.client.as_ref().map_err(|error| anyhow!("{error}"))?;
        let client = request
            .u64_parameter("routing.timeout")
            .filter(|seconds| *seconds > 0)
            .map(|seconds| {
                client
                    .clone()
                    .with_execute_timeout(Duration::from_secs(seconds))
            })
            .unwrap_or_else(|| client.clone());
        let model_path = companion_model_path(store, manifest);
        let mut parameters = companion_execution_parameters(request, runtime);
        if let Some(accelerator) = companion_runtime_accelerator(runtime) {
            parameters.insert(
                "routing.accelerator".to_string(),
                ParameterValue::String(accelerator.to_string()),
            );
        }
        let staged_inputs = output_dir.join(".inputs");
        let inputs = match companion_inputs(request, &staged_inputs) {
            Ok(inputs) => inputs,
            Err(error) => {
                let _ = fs::remove_dir_all(&staged_inputs);
                return Err(error);
            }
        };
        let companion_request = json!({
            "protocol_version": 1,
            "model_path": model_path,
            "model": manifest.id,
            "task": request.task.to_string(),
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "inputs": inputs,
            "parameters": parameters,
            "effective_parameters": parameters,
            "explicit_parameters": request.explicit_parameters,
            "parameter_policy": request.parameter_policy,
            "output_dir": output_dir,
            "runtime": runtime,
            "local_files_only": true
        });
        let response = client.execute(&companion_request);
        let _ = fs::remove_dir_all(&staged_inputs);
        let response = response?;
        companion_execution(response, output_dir, runtime, request.task)
    }

    fn estimate(
        &self,
        store: &ModelStore,
        manifest: &ModelManifest,
        request: &EffectiveInferenceRequest,
    ) -> Result<Option<WorkloadEstimate>> {
        let client = match &self.client {
            Ok(client) => client,
            Err(_) => return Ok(None),
        };
        let parameters = request.values_only();
        let companion_request = json!({
            "protocol_version": 1,
            "model_path": companion_model_path(store, manifest),
            "model": manifest.id,
            "task": request.task.to_string(),
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "parameters": parameters,
            "effective_parameters": parameters,
            "explicit_parameters": request.explicit_parameters,
            "parameter_policy": request.parameter_policy,
            "local_files_only": true
        });
        let mut value = client.estimate(&companion_request)?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| anyhow!("media companion estimate response must be an object"))?;
        if !object.contains_key("fit")
            && let Some(fit) = object.remove("fit_assessment")
        {
            object.insert("fit".to_string(), fit);
        }
        let estimate = serde_json::from_value(value)
            .context("invalid media companion workload estimate response")?;
        Ok(Some(estimate))
    }
}

fn companion_model_path(store: &ModelStore, manifest: &ModelManifest) -> PathBuf {
    if manifest.metadata.repository_layout == RepositoryLayout::SingleFile
        && let Some(model_path) = manifest.model_path.as_deref()
    {
        return store.absolute_model_file(manifest, model_path);
    }
    store.model_dir(&manifest.id).join("files")
}

pub(super) fn companion_inputs(
    request: &EffectiveInferenceRequest,
    staging_dir: &Path,
) -> Result<Value> {
    let mut inputs = serde_json::Map::new();
    for (index, input) in request.inputs.iter().enumerate() {
        let value = match &input.source {
            crate::inference::InferenceInputSource::Path { path } => Value::String(path.clone()),
            crate::inference::InferenceInputSource::Url { .. } => {
                bail!(
                    "the offline media companion does not fetch URL inputs; download the media and use a local path"
                )
            }
            crate::inference::InferenceInputSource::Base64 { data } => {
                if data.len() > 512 * 1024 * 1024 {
                    bail!("inline base64 input exceeds the 512 MiB encoded-size limit");
                }
                fs::create_dir_all(staging_dir)?;
                let extension = input_extension(input.mime_type.as_deref());
                let path = staging_dir.join(format!("input-{index}.{extension}"));
                fs::write(&path, decode_base64(data)?)?;
                Value::String(path.display().to_string())
            }
            crate::inference::InferenceInputSource::Text { text } => Value::String(text.clone()),
        };
        match inputs.get_mut(&input.role) {
            Some(Value::Array(values)) => values.push(value),
            Some(existing) => {
                let first = existing.take();
                *existing = Value::Array(vec![first, value]);
            }
            None => {
                inputs.insert(input.role.clone(), value);
            }
        }
    }
    Ok(Value::Object(inputs))
}

fn input_extension(mime_type: Option<&str>) -> &'static str {
    match mime_type.unwrap_or_default().to_ascii_lowercase().as_str() {
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "video/mp4" => "mp4",
        "video/webm" => "webm",
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/flac" => "flac",
        "audio/mpeg" => "mp3",
        "audio/ogg" => "ogg",
        _ => "bin",
    }
}

pub(super) fn decode_base64(data: &str) -> Result<Vec<u8>> {
    let mut encoded = data
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect::<Vec<_>>();
    if encoded.len() % 4 == 1 {
        bail!("invalid base64 input length");
    }
    while encoded.len() % 4 != 0 {
        encoded.push(b'=');
    }
    let mut output = Vec::with_capacity(encoded.len() / 4 * 3);
    for (chunk_index, chunk) in encoded.chunks_exact(4).enumerate() {
        let last = chunk_index + 1 == encoded.len() / 4;
        let padding = chunk.iter().rev().take_while(|byte| **byte == b'=').count();
        if padding > 2 || (!last && padding > 0) || chunk[..chunk.len() - padding].contains(&b'=') {
            bail!("invalid base64 padding");
        }
        let sextet = |byte: u8| -> Result<u8> {
            match byte {
                b'A'..=b'Z' => Ok(byte - b'A'),
                b'a'..=b'z' => Ok(byte - b'a' + 26),
                b'0'..=b'9' => Ok(byte - b'0' + 52),
                b'+' | b'-' => Ok(62),
                b'/' | b'_' => Ok(63),
                b'=' => Ok(0),
                _ => bail!("invalid base64 character"),
            }
        };
        let a = sextet(chunk[0])?;
        let b = sextet(chunk[1])?;
        let c = sextet(chunk[2])?;
        let d = sextet(chunk[3])?;
        output.push((a << 2) | (b >> 4));
        if padding < 2 {
            output.push((b << 4) | (c >> 2));
        }
        if padding == 0 {
            output.push((c << 6) | d);
        }
    }
    Ok(output)
}

fn companion_execution(
    response: CompanionExecution,
    output_dir: &Path,
    runtime: &str,
    expected_task: InferenceTask,
) -> Result<BackendExecution> {
    if !response.ok {
        bail!("media companion returned an unsuccessful execution");
    }
    let response_task = response.task.trim().replace('-', "_");
    let expected_task_name = expected_task.to_string().replace('-', "_");
    if response_task != expected_task_name {
        bail!(
            "media companion response task mismatch: expected {}, got '{}'",
            expected_task,
            response.task
        );
    }
    let outputs = response
        .outputs
        .into_iter()
        .map(|output| companion_output(output, output_dir))
        .collect::<Result<Vec<_>>>()?;
    if outputs.is_empty() {
        bail!("media companion completed without producing an output");
    }
    Ok(BackendExecution {
        runtime: runtime.to_string(),
        outputs,
        warnings: response.warnings,
        metadata: sanitized_companion_metadata(response.metadata),
    })
}

fn sanitized_companion_metadata(metadata: Value) -> Value {
    let Some(source) = metadata.as_object() else {
        return Value::Null;
    };
    let mut sanitized = serde_json::Map::new();
    copy_metadata_fields(
        source,
        &mut sanitized,
        &["runtime", "created_unix", "elapsed_seconds", "offline"],
    );

    if let Some(backend) = source.get("backend").and_then(Value::as_object) {
        let mut safe_backend = serde_json::Map::new();
        copy_metadata_fields(
            backend,
            &mut safe_backend,
            &[
                "runtime",
                "pipeline_task",
                "pipeline_class",
                "device",
                "dtype",
                "seed",
                "model_load_seconds",
                "inference_seconds",
                "encoding_seconds",
            ],
        );
        if let Some(parameters) = backend.get("translated_parameters").and_then(string_array) {
            safe_backend.insert("translated_parameters".to_string(), parameters);
        }
        if let Some(support) = backend
            .get("parameter_support")
            .and_then(sanitized_parameter_support)
        {
            safe_backend.insert("parameter_support".to_string(), support);
        }
        sanitized.insert("backend".to_string(), Value::Object(safe_backend));
    }
    Value::Object(sanitized)
}

fn copy_metadata_fields(
    source: &serde_json::Map<String, Value>,
    destination: &mut serde_json::Map<String, Value>,
    fields: &[&str],
) {
    for field in fields {
        if let Some(value) = source.get(*field)
            && matches!(
                value,
                Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
            )
        {
            destination.insert((*field).to_string(), value.clone());
        }
    }
}

fn string_array(value: &Value) -> Option<Value> {
    let values = value
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .map(|value| Value::String(value.to_string()))
        .collect::<Vec<_>>();
    Some(Value::Array(values))
}

fn sanitized_parameter_support(value: &Value) -> Option<Value> {
    let source = value.as_object()?;
    let mut sanitized = serde_json::Map::new();
    if let Some(policy) = source.get("policy").and_then(Value::as_str) {
        sanitized.insert("policy".to_string(), Value::String(policy.to_string()));
    }
    for field in ["explicit_parameters", "unsupported_explicit_parameters"] {
        if let Some(values) = source.get(field).and_then(string_array) {
            sanitized.insert(field.to_string(), values);
        }
    }
    Some(Value::Object(sanitized))
}

fn companion_output(output: CompanionOutput, output_dir: &Path) -> Result<BackendOutput> {
    let path = PathBuf::from(output.path);
    let path = if path.is_absolute() {
        path
    } else {
        output_dir.join(path)
    };
    ensure_output_path(output_dir, &path)?;
    Ok(BackendOutput {
        path,
        mime_type: output.mime_type,
        width: output.width,
        height: output.height,
        duration: output.duration,
        metadata: output.metadata,
    })
}

pub(super) fn default_companion_candidates(
    available: bool,
    reason: Option<String>,
    schema_paths: &[String],
) -> Vec<InferenceRuntimeCandidate> {
    companion_candidates(available, reason, schema_paths, None)
}

fn companion_candidates(
    available: bool,
    reason: Option<String>,
    schema_paths: &[String],
    health: Option<&CompanionHealth>,
) -> Vec<InferenceRuntimeCandidate> {
    companion_candidates_with_override(
        available,
        reason,
        schema_paths,
        health,
        configured_media_accelerator(),
    )
}

fn companion_candidates_with_override(
    available: bool,
    reason: Option<String>,
    schema_paths: &[String],
    health: Option<&CompanionHealth>,
    configured: Option<RuntimeAccelerator>,
) -> Vec<InferenceRuntimeCandidate> {
    let detected = available_companion_accelerators_with_override(health, configured);
    runtime_registry()
        .iter()
        .filter(|descriptor| descriptor.runtime == BackendRuntime::MediaCompanion)
        .filter_map(|descriptor| {
            let id = media_runtime_label(descriptor.id)?;
            let accelerator = descriptor
                .accelerators
                .first()
                .copied()
                .map(runtime_accelerator)
                .unwrap_or(RuntimeAccelerator::Other);
            let hardware_available =
                accelerator == RuntimeAccelerator::Cpu || detected.contains(&accelerator);
            let candidate_available = available && hardware_available;
            let availability_reason = if !available {
                reason.clone()
            } else if !hardware_available {
                Some(
                    companion_accelerator_unavailable_reason_with_override(
                        health,
                        accelerator,
                        configured,
                    )
                    .unwrap_or_else(|| {
                        format!(
                            "{} accelerator is not detected on this host",
                            format!("{accelerator:?}").to_ascii_lowercase()
                        )
                    }),
                )
            } else {
                None
            };
            Some(InferenceRuntimeCandidate {
                id: id.to_string(),
                backend: "media-companion".to_string(),
                accelerator,
                available: candidate_available,
                availability_reason,
                supported_tasks: descriptor.supported_tasks.to_vec(),
                supported_layouts: descriptor.supported_layouts.to_vec(),
                supported_formats: descriptor.supported_formats.to_vec(),
                supported_families: Vec::new(),
                supported_architectures: descriptor
                    .supported_architectures
                    .iter()
                    .map(|architecture| (*architecture).to_string())
                    .collect(),
                parameter_support: schema_paths
                    .iter()
                    .map(|path| {
                        (
                            path.clone(),
                            descriptor.parameter_support_status(path.as_str()),
                        )
                    })
                    .collect(),
                supports_offloading: descriptor.supports_offloading
                    && matches!(
                        accelerator,
                        RuntimeAccelerator::Cuda | RuntimeAccelerator::Rocm
                    ),
                supports_compile: descriptor.supports_compile,
                supports_batching: descriptor.supports_batching,
                priority: descriptor.priority,
            })
        })
        .collect()
}

fn default_companion_parameter_support(
    schema_paths: &[String],
) -> BTreeMap<String, ParameterSupportStatus> {
    companion_parameter_support(schema_paths, None)
}

fn companion_parameter_support(
    schema_paths: &[String],
    health: Option<&CompanionHealth>,
) -> BTreeMap<String, ParameterSupportStatus> {
    let detected = available_companion_accelerators(health);
    let descriptor = runtime_registry()
        .iter()
        .filter(|descriptor| descriptor.runtime == BackendRuntime::MediaCompanion)
        .find(|descriptor| {
            descriptor
                .accelerators
                .first()
                .copied()
                .map(runtime_accelerator)
                .is_some_and(|accelerator| detected.contains(&accelerator))
        })
        .or_else(|| {
            runtime_registry().iter().find(|descriptor| {
                descriptor.runtime == BackendRuntime::MediaCompanion
                    && descriptor.accelerators.contains(&BackendAccelerator::Cpu)
            })
        });
    schema_paths
        .iter()
        .map(|path| {
            (
                path.clone(),
                descriptor
                    .map(|descriptor| descriptor.parameter_support_status(path))
                    .unwrap_or(ParameterSupportStatus::ModelDependent),
            )
        })
        .collect()
}

fn available_companion_accelerators(health: Option<&CompanionHealth>) -> Vec<RuntimeAccelerator> {
    available_companion_accelerators_with_override(health, configured_media_accelerator())
}

fn available_companion_accelerators_with_override(
    health: Option<&CompanionHealth>,
    configured: Option<RuntimeAccelerator>,
) -> Vec<RuntimeAccelerator> {
    if let Some(configured) = configured {
        return vec![configured];
    }
    if let Some(health) = health
        && !health.accelerators.is_empty()
    {
        return health
            .accelerators
            .iter()
            .filter(|(_, status)| status.available)
            .filter_map(|(name, _)| companion_accelerator_from_name(name))
            .collect();
    }
    vec![detected_accelerator()]
}

fn companion_accelerator_unavailable_reason_with_override(
    health: Option<&CompanionHealth>,
    accelerator: RuntimeAccelerator,
    configured: Option<RuntimeAccelerator>,
) -> Option<String> {
    if configured.is_some() {
        return None;
    }
    let health = health.filter(|health| !health.accelerators.is_empty())?;
    let name = companion_accelerator_name(accelerator)?;
    let status = health.accelerators.get(name)?;
    if status.available {
        return None;
    }
    Some(match status.detail.as_deref() {
        Some(detail) => format!("{name} is unavailable in media companion Python: {detail}"),
        None => format!("{name} is unavailable in media companion Python"),
    })
}

fn companion_accelerator_from_name(name: &str) -> Option<RuntimeAccelerator> {
    match name.trim().to_ascii_lowercase().as_str() {
        "cpu" => Some(RuntimeAccelerator::Cpu),
        "cuda" => Some(RuntimeAccelerator::Cuda),
        "rocm" | "hip" => Some(RuntimeAccelerator::Rocm),
        "mps" | "metal" => Some(RuntimeAccelerator::Mps),
        "mlx" => Some(RuntimeAccelerator::Mlx),
        _ => None,
    }
}

fn companion_accelerator_name(accelerator: RuntimeAccelerator) -> Option<&'static str> {
    match accelerator {
        RuntimeAccelerator::Cpu => Some("cpu"),
        RuntimeAccelerator::Cuda => Some("cuda"),
        RuntimeAccelerator::Rocm => Some("rocm"),
        RuntimeAccelerator::Mps => Some("mps"),
        RuntimeAccelerator::Mlx => Some("mlx"),
        _ => None,
    }
}

fn media_runtime_label(id: RuntimeId) -> Option<&'static str> {
    match id {
        RuntimeId::MediaCompanionCuda => Some("media-companion-cuda"),
        RuntimeId::MediaCompanionRocm => Some("media-companion-rocm"),
        RuntimeId::MediaCompanionMetal => Some("media-companion-metal"),
        RuntimeId::MediaCompanionCpu => Some("media-companion-cpu"),
        _ => None,
    }
}

fn runtime_accelerator(accelerator: BackendAccelerator) -> RuntimeAccelerator {
    match accelerator {
        BackendAccelerator::Cpu => RuntimeAccelerator::Cpu,
        BackendAccelerator::Cuda => RuntimeAccelerator::Cuda,
        BackendAccelerator::Rocm => RuntimeAccelerator::Rocm,
        BackendAccelerator::Metal => RuntimeAccelerator::Mps,
        BackendAccelerator::Mlx => RuntimeAccelerator::Mlx,
        _ => RuntimeAccelerator::Other,
    }
}

fn companion_runtime_accelerator(runtime: &str) -> Option<&'static str> {
    match runtime {
        "media-companion-cuda" => Some("cuda"),
        "media-companion-rocm" => Some("rocm"),
        "media-companion-metal" => Some("metal"),
        "media-companion-cpu" => Some("cpu"),
        _ => None,
    }
}

pub(super) fn companion_execution_parameters(
    request: &EffectiveInferenceRequest,
    runtime: &str,
) -> BTreeMap<String, ParameterValue> {
    let mut parameters = request.values_only();
    let gpu_runtime = matches!(
        companion_runtime_accelerator(runtime),
        Some("cuda" | "rocm")
    );
    for (permission, execution_flag) in [
        ("routing.allow_cpu_offload", "_werk_enable_cpu_offload"),
        (
            "routing.allow_sequential_offload",
            "_werk_enable_sequential_offload",
        ),
        (
            "routing.allow_component_offload",
            "_werk_enable_component_offload",
        ),
    ] {
        let selected = gpu_runtime
            && request.parameters.get(permission).is_some_and(|parameter| {
                parameter.source == ParameterSource::BackendAdjustment
                    && parameter.value.as_bool() == Some(true)
            });
        parameters.insert(execution_flag.to_string(), selected.into());
    }
    parameters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media_companion::CompanionAccelerator;

    fn health_with_accelerators(
        accelerators: impl IntoIterator<Item = (&'static str, bool, &'static str)>,
    ) -> CompanionHealth {
        CompanionHealth {
            ok: true,
            status: "ok".to_string(),
            protocol_version: 1,
            companion_version: Some("test".to_string()),
            python_version: Some("3.12".to_string()),
            dependencies: BTreeMap::new(),
            accelerators: accelerators
                .into_iter()
                .map(|(name, available, detail)| {
                    (
                        name.to_string(),
                        CompanionAccelerator {
                            available,
                            version: None,
                            detail: Some(detail.to_string()),
                        },
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn companion_health_controls_media_accelerator_availability() {
        let health = health_with_accelerators([
            ("cpu", true, "CPU available"),
            ("cuda", true, "RTX 3090"),
            ("rocm", false, "ROCm unavailable"),
        ]);

        let available = available_companion_accelerators_with_override(Some(&health), None);

        assert!(available.contains(&RuntimeAccelerator::Cpu));
        assert!(available.contains(&RuntimeAccelerator::Cuda));
        assert!(!available.contains(&RuntimeAccelerator::Rocm));
        assert_eq!(
            companion_accelerator_unavailable_reason_with_override(
                Some(&health),
                RuntimeAccelerator::Rocm,
                None,
            )
            .as_deref(),
            Some("rocm is unavailable in media companion Python: ROCm unavailable")
        );

        let candidates = companion_candidates_with_override(true, None, &[], Some(&health), None);
        let cuda = candidates
            .iter()
            .find(|candidate| candidate.id == "media-companion-cuda")
            .unwrap();
        let rocm = candidates
            .iter()
            .find(|candidate| candidate.id == "media-companion-rocm")
            .unwrap();
        assert!(cuda.available);
        assert!(!rocm.available);
        assert!(
            rocm.availability_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("ROCm unavailable"))
        );
    }

    #[test]
    fn configured_media_accelerator_overrides_health_detection() {
        let health = health_with_accelerators([
            ("cpu", true, "CPU available"),
            ("cuda", false, "CUDA unavailable"),
        ]);

        assert_eq!(
            available_companion_accelerators_with_override(
                Some(&health),
                Some(RuntimeAccelerator::Cuda),
            ),
            vec![RuntimeAccelerator::Cuda]
        );
    }

    #[test]
    fn persisted_companion_metadata_keeps_diagnostics_but_drops_private_duplicates() {
        let metadata = json!({
            "runtime": "werk-media-companion",
            "model_path": "/private/models/secret",
            "effective_parameters": {
                "prompt": "private prompt",
                "lyrics": "private lyrics"
            },
            "outputs": [{"path": "/private/output.png"}],
            "elapsed_seconds": 12.5,
            "offline": true,
            "backend": {
                "runtime": "diffusers",
                "device": "cuda",
                "dtype": "float16",
                "seed": 17,
                "model_load_seconds": 8.0,
                "inference_seconds": 4.0,
                "encoding_seconds": 0.5,
                "text": "private transcript",
                "translated_parameters": ["prompt", "width", 42],
                "parameter_support": {
                    "policy": "strict",
                    "explicit_parameters": ["image.width"],
                    "unsupported_explicit_parameters": [],
                    "unsupported_reasons": {
                        "image.path": "/private/reason"
                    }
                }
            }
        });

        let sanitized = sanitized_companion_metadata(metadata);

        assert_eq!(sanitized["elapsed_seconds"], 12.5);
        assert_eq!(sanitized["backend"]["device"], "cuda");
        assert_eq!(sanitized["backend"]["inference_seconds"], 4.0);
        assert_eq!(
            sanitized["backend"]["translated_parameters"],
            json!(["prompt", "width"])
        );
        assert!(sanitized.get("model_path").is_none());
        assert!(sanitized.get("effective_parameters").is_none());
        assert!(sanitized.get("outputs").is_none());
        assert!(sanitized["backend"].get("text").is_none());
        assert!(
            sanitized["backend"]["parameter_support"]
                .get("unsupported_reasons")
                .is_none()
        );
        let encoded = serde_json::to_string(&sanitized).unwrap();
        assert!(!encoded.contains("private"));
    }
}
