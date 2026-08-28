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
    backend::{
        BackendAccelerator, BackendRuntime, RuntimeId, require_qwen_tts_python, runtime_registry,
    },
    capabilities::{InferenceTask, RepositoryLayout},
    inference::{
        EffectiveInferenceRequest, InferenceRuntimeCandidate, ParameterSource,
        ParameterSupportStatus, ParameterValue, RuntimeAccelerator, TaskReadiness,
        TaskReadinessStatus, WorkloadEstimate,
    },
    media_companion::{CompanionClient, CompanionExecution, CompanionHealth, CompanionOutput},
    model_store::{ModelManifest, ModelStore},
};

#[derive(Debug, Clone)]
pub struct CompanionMediaBackend {
    client: std::result::Result<CompanionClient, String>,
    health_cache: Arc<OnceLock<CompanionHealth>>,
    qwen_client_cache: Arc<OnceLock<CompanionClient>>,
    qwen_health_cache: Arc<OnceLock<CompanionHealth>>,
}

impl CompanionMediaBackend {
    pub fn discover() -> Self {
        Self {
            client: CompanionClient::discover()
                .map(CompanionClient::with_resident_worker)
                .map_err(|error| error.to_string()),
            health_cache: Arc::new(OnceLock::new()),
            qwen_client_cache: Arc::new(OnceLock::new()),
            qwen_health_cache: Arc::new(OnceLock::new()),
        }
    }

    pub fn with_client(client: CompanionClient) -> Self {
        Self {
            client: Ok(client),
            health_cache: Arc::new(OnceLock::new()),
            qwen_client_cache: Arc::new(OnceLock::new()),
            qwen_health_cache: Arc::new(OnceLock::new()),
        }
    }

    fn client_for_manifest(
        &self,
        store: &ModelStore,
        manifest: &ModelManifest,
    ) -> std::result::Result<(CompanionClient, bool), String> {
        if !is_qwen3_tts_manifest(manifest) {
            return self.client.clone().map(|client| (client, false));
        }
        if let Some(client) = self.qwen_client_cache.get() {
            return Ok((client.clone(), true));
        }

        // Cache only a usable client. A failed lookup commonly means the user
        // has not installed the managed backend yet; keeping that error in a
        // OnceLock would make `werk backend install qwen-tts` ineffective until
        // the Werk process is restarted.
        let python = require_qwen_tts_python(store).map_err(|error| error.to_string())?;
        let client = CompanionClient::from_python(python).map_err(|error| error.to_string())?;
        Ok((self.qwen_client_cache.get_or_init(|| client).clone(), true))
    }

    fn health(
        &self,
        client: &CompanionClient,
        qwen_tts: bool,
    ) -> std::result::Result<CompanionHealth, String> {
        let cache = if qwen_tts {
            &self.qwen_health_cache
        } else {
            &self.health_cache
        };
        if let Some(health) = cache.get() {
            return Ok(health.clone());
        }
        // Preflight operations must not queue behind a long resident media
        // generation. They are lightweight one-shot calls; only execute owns
        // the serialized resident worker and its loaded pipeline.
        let health = client
            .clone()
            .without_resident_worker()
            .health()
            .map_err(|error| error.to_string())?;
        let _ = cache.set(health.clone());
        Ok(health)
    }
}

impl Default for CompanionMediaBackend {
    fn default() -> Self {
        Self::discover()
    }
}

fn is_qwen3_tts_manifest(manifest: &ModelManifest) -> bool {
    manifest.architecture.as_deref() == Some("qwen3_tts")
        || manifest
            .metadata
            .family
            .as_deref()
            .is_some_and(|family| family.starts_with("qwen3-tts"))
}

fn unavailable_task_readiness(detail: String) -> TaskReadiness {
    TaskReadiness {
        status: TaskReadinessStatus::Unavailable,
        detail,
        adapter: None,
        required_backend: None,
        install_command: None,
        fallback_backend: None,
        missing_dependencies: Vec::new(),
        missing_dependency_groups: Vec::new(),
    }
}

fn normalize_task_readiness(mut readiness: TaskReadiness) -> TaskReadiness {
    // The install command is an executable recommendation, not descriptive
    // metadata. Old companions exposed it as a top-level field even when the
    // task was unavailable; never let that become an installation suggestion.
    if readiness.status != TaskReadinessStatus::Installable {
        readiness.install_command = None;
    }
    readiness
}

fn task_readiness_from_model_probe(value: &Value, available: bool, detail: &str) -> TaskReadiness {
    let readiness = value
        .get("readiness")
        .cloned()
        .and_then(|readiness| serde_json::from_value(readiness).ok())
        .unwrap_or_else(|| TaskReadiness {
            status: if available {
                TaskReadinessStatus::Available
            } else {
                TaskReadinessStatus::Unavailable
            },
            detail: detail.to_string(),
            adapter: value
                .get("adapter")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            required_backend: value
                .get("required_backend")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            install_command: value
                .get("install_command")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            fallback_backend: None,
            missing_dependencies: value
                .get("missing_dependencies")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect(),
            missing_dependency_groups: Vec::new(),
        });
    normalize_task_readiness(readiness)
}

fn confirms_installable_qwen_voice_design(readiness: &TaskReadiness) -> bool {
    readiness.status == TaskReadinessStatus::Installable
        && readiness.adapter.as_deref() == Some("qwen3_tts_voice_design")
        && readiness.required_backend.as_deref() == Some("qwen-tts")
        && readiness.install_command.as_deref() == Some("werk backend install qwen-tts")
}

impl MediaInferenceBackend for CompanionMediaBackend {
    fn probe(
        &self,
        store: &ModelStore,
        manifest: &ModelManifest,
        task: InferenceTask,
        schema_paths: &[String],
    ) -> BackendProbe {
        let (client, qwen_tts, qwen_runtime_error) = match self.client_for_manifest(store, manifest)
        {
            Ok((client, qwen_tts)) => (client, qwen_tts, None),
            Err(error) if is_qwen3_tts_manifest(manifest) => match self.client.clone() {
                // The general companion can inspect config.json without
                // loading Qwen weights. This distinguishes an installable
                // VoiceDesign model from a recognized but unsupported
                // CustomVoice/Base variant even when the isolated backend
                // is absent.
                Ok(client) => (client, false, Some(error)),
                Err(_) => {
                    return BackendProbe {
                        available: false,
                        detail: error.clone(),
                        candidates: companion_candidates_for_model(
                            false,
                            Some(error.clone()),
                            schema_paths,
                            task,
                            manifest.metadata.repository_layout,
                            None,
                        ),
                        parameter_support: default_companion_parameter_support(schema_paths),
                        readiness: Some(unavailable_task_readiness(error)),
                    };
                }
            },
            Err(error) => {
                return BackendProbe {
                    available: false,
                    detail: error.clone(),
                    candidates: companion_candidates_for_model(
                        false,
                        Some(error.clone()),
                        schema_paths,
                        task,
                        manifest.metadata.repository_layout,
                        None,
                    ),
                    parameter_support: default_companion_parameter_support(schema_paths),
                    readiness: Some(unavailable_task_readiness(error)),
                };
            }
        };
        let health = match self.health(&client, qwen_tts) {
            Ok(health) => health,
            Err(error) => {
                return BackendProbe {
                    available: false,
                    detail: error.clone(),
                    candidates: companion_candidates_for_model(
                        false,
                        Some(error.clone()),
                        schema_paths,
                        task,
                        manifest.metadata.repository_layout,
                        None,
                    ),
                    parameter_support: default_companion_parameter_support(schema_paths),
                    readiness: Some(unavailable_task_readiness(error)),
                };
            }
        };
        let probe_request = json!({
            "model_path": companion_model_path(store, manifest),
            "task": companion_wire_task(task).to_string(),
            "layout": manifest.metadata.repository_layout.to_string(),
            "family": manifest.metadata.family,
            "architecture": manifest.architecture,
        });
        let model_probe = client
            .clone()
            .without_resident_worker()
            .probe_model(&probe_request);
        let (mut available, mut detail, mut readiness) = match model_probe {
            Ok(value) => {
                let available = value
                    .get("supported")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let detail = value
                    .get("detail")
                    .and_then(Value::as_str)
                    .unwrap_or("companion model probe succeeded")
                    .to_string();
                let readiness = task_readiness_from_model_probe(&value, available, &detail);
                (available, detail, readiness)
            }
            Err(error) => {
                let detail = error.to_string();
                (false, detail.clone(), unavailable_task_readiness(detail))
            }
        };
        if let Some(error) = qwen_runtime_error {
            available = false;
            if confirms_installable_qwen_voice_design(&readiness) {
                detail = error.clone();
                readiness.detail = error;
            } else if readiness.status == TaskReadinessStatus::Available {
                detail = error.clone();
                readiness = unavailable_task_readiness(error);
            } else {
                // Preserve a concrete model-probe rejection (broken layout,
                // unsupported variant, or unavailable dependencies) instead
                // of replacing it with an unrelated install recommendation.
                detail = readiness.detail.clone();
            }
        }
        let parameter_support = companion_parameter_support(schema_paths, Some(&health));
        BackendProbe {
            available,
            detail: detail.clone(),
            candidates: companion_candidates_for_model(
                available,
                (!available).then_some(detail),
                schema_paths,
                task,
                manifest.metadata.repository_layout,
                Some(&health),
            ),
            parameter_support,
            readiness: Some(readiness),
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
        let (client, _) = self
            .client_for_manifest(store, manifest)
            .map_err(|error| anyhow!("{error}"))?;
        let client = request
            .u64_parameter("routing.timeout")
            .filter(|seconds| *seconds > 0)
            .map(|seconds| {
                client
                    .clone()
                    .with_execute_timeout(Duration::from_secs(seconds))
            })
            .unwrap_or(client);
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
            "task": companion_wire_task(request.task).to_string(),
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
        let (client, _) = match self.client_for_manifest(store, manifest) {
            Ok(client) => client,
            Err(_) => return Ok(None),
        };
        let mut parameters = request.values_only();
        apply_companion_task_parameters(request.task, &mut parameters);
        let companion_request = json!({
            "protocol_version": 1,
            "model_path": companion_model_path(store, manifest),
            "model": manifest.id,
            "task": companion_wire_task(request.task).to_string(),
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "parameters": parameters,
            "effective_parameters": parameters,
            "explicit_parameters": request.explicit_parameters,
            "parameter_policy": request.parameter_policy,
            "local_files_only": true
        });
        let mut value = client
            .clone()
            .without_resident_worker()
            .estimate(&companion_request)?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| anyhow!("media companion estimate response must be an object"))?;
        if !object.contains_key("fit")
            && let Some(fit) = object.remove("fit_assessment")
        {
            object.insert("fit".to_string(), fit);
        }
        let estimate: WorkloadEstimate = serde_json::from_value(value)
            .context("invalid media companion workload estimate response")?;
        Ok(Some(estimate))
    }
}

fn companion_wire_task(task: InferenceTask) -> InferenceTask {
    // The companion protocol supports the complete canonical task taxonomy.
    // Preserve task identity so adapter selection, errors, and metadata retain
    // the precise public operation instead of silently degrading to a generic
    // adapter task.
    task
}

fn apply_companion_task_parameters(
    task: InferenceTask,
    parameters: &mut BTreeMap<String, ParameterValue>,
) {
    if task == InferenceTask::SpeechTranslation {
        parameters.insert(
            "stt.operation".to_string(),
            ParameterValue::String("translate".to_string()),
        );
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
    for (chunk_index, chunk) in encoded.as_chunks::<4>().0.iter().enumerate() {
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
    let expected_task_name = companion_wire_task(expected_task)
        .to_string()
        .replace('-', "_");
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
                "model_cache_hit",
                "inference_seconds",
                "encoding_seconds",
            ],
        );
        copy_metadata_enum_field(
            backend,
            &mut safe_backend,
            "offload_mode",
            &["none", "model_cpu", "sequential_cpu"],
        );
        copy_metadata_enum_field(
            backend,
            &mut safe_backend,
            "offload_request",
            &["none", "model_cpu", "sequential_cpu", "component"],
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

fn copy_metadata_enum_field(
    source: &serde_json::Map<String, Value>,
    destination: &mut serde_json::Map<String, Value>,
    field: &str,
    allowed: &[&str],
) {
    if let Some(value) = source.get(field).and_then(Value::as_str)
        && allowed.contains(&value)
    {
        destination.insert(field.to_string(), Value::String(value.to_string()));
    }
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

#[cfg(test)]
pub(super) fn default_companion_candidates(
    available: bool,
    reason: Option<String>,
    schema_paths: &[String],
) -> Vec<InferenceRuntimeCandidate> {
    companion_candidates_with_override(
        available,
        reason,
        schema_paths,
        None,
        configured_media_accelerator(),
        None,
    )
}

fn companion_candidates_for_model(
    available: bool,
    reason: Option<String>,
    schema_paths: &[String],
    task: InferenceTask,
    layout: RepositoryLayout,
    health: Option<&CompanionHealth>,
) -> Vec<InferenceRuntimeCandidate> {
    companion_candidates_with_override(
        available,
        reason,
        schema_paths,
        health,
        configured_media_accelerator(),
        Some((task, layout)),
    )
}

fn companion_candidates_with_override(
    available: bool,
    reason: Option<String>,
    schema_paths: &[String],
    health: Option<&CompanionHealth>,
    configured: Option<RuntimeAccelerator>,
    model_context: Option<(InferenceTask, RepositoryLayout)>,
) -> Vec<InferenceRuntimeCandidate> {
    let detected = available_companion_accelerators_with_override(health, configured);
    let adapter_supports_offloading = model_context
        .is_some_and(|(task, layout)| companion_adapter_supports_offloading(task, layout));
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
                supports_offloading: adapter_supports_offloading
                    && descriptor.supports_offloading
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

/// Mirrors the adapter choice in `runtime/werk_media_companion.py` without
/// advertising an offload path for model/task combinations whose execution is
/// handled by Transformers.
fn companion_adapter_supports_offloading(task: InferenceTask, layout: RepositoryLayout) -> bool {
    match task {
        InferenceTask::ImageGeneration
        | InferenceTask::ImageEditing
        | InferenceTask::ImageVariation
        | InferenceTask::ImageInpainting
        | InferenceTask::ImageOutpainting
        | InferenceTask::ImageUpscaling
        | InferenceTask::VideoGeneration
        | InferenceTask::ImageToVideo
        | InferenceTask::VideoToVideo
        | InferenceTask::VideoInpainting
        | InferenceTask::VideoExtension
        | InferenceTask::VideoUpscaling
        | InferenceTask::FrameInterpolation => matches!(
            layout,
            RepositoryLayout::Diffusers | RepositoryLayout::SingleFile
        ),
        InferenceTask::AudioGeneration | InferenceTask::MusicGeneration => {
            layout == RepositoryLayout::Diffusers
        }
        _ => false,
    }
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
    apply_companion_task_parameters(request.task, &mut parameters);
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
    use crate::{
        backend::managed_qwen_tts_python,
        media_companion::CompanionAccelerator,
        model_store::{ModelFormat, ModelMetadata, ModelSource},
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    struct CompanionTestDirectory(PathBuf);

    impl CompanionTestDirectory {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "werk-companion-{name}-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create companion test directory");
            Self(path)
        }

        fn store(&self) -> ModelStore {
            ModelStore::resolve(Some(self.0.join("store"))).expect("resolve test model store")
        }
    }

    impl Drop for CompanionTestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn qwen_voice_design_manifest() -> ModelManifest {
        ModelManifest {
            id: "qwen-voice-design".to_string(),
            source: ModelSource::LocalPath {
                path: "qwen-voice-design".to_string(),
            },
            format: ModelFormat::SafeTensors,
            architecture: Some("qwen3_tts".to_string()),
            tokenizer_path: None,
            config_path: Some("files/config.json".to_string()),
            model_path: None,
            backend: "media-companion".to_string(),
            created_unix: 1,
            files: Vec::new(),
            artifacts: Vec::new(),
            metadata: ModelMetadata {
                family: Some("qwen3-tts-voice-design".to_string()),
                repository_layout: RepositoryLayout::Transformers,
                tasks: vec![InferenceTask::TextToSpeech],
                ..ModelMetadata::default()
            },
        }
    }

    #[cfg(unix)]
    fn write_executable(path: &Path, contents: &str) {
        use std::os::unix::fs::PermissionsExt;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create executable parent");
        }
        fs::write(path, contents).expect("write test executable");
        let mut permissions = fs::metadata(path)
            .expect("test executable metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).expect("make test executable executable");
    }

    #[cfg(unix)]
    fn static_probe_client(directory: &Path, name: &str, probe: &Value) -> CompanionClient {
        let path = directory.join(name);
        let health = json!({
            "ok": true,
            "status": "ok",
            "protocol_version": 1,
            "companion_version": "test",
            "python_version": "test",
            "dependencies": {},
            "accelerators": {
                "cpu": {
                    "available": true,
                    "version": null,
                    "detail": "test CPU"
                }
            }
        });
        let health = serde_json::to_string(&health).expect("serialize health response");
        let probe = serde_json::to_string(probe).expect("serialize probe response");
        let quote = |value: &str| value.replace('\'', "'\"'\"'");
        write_executable(
            &path,
            &format!(
                "#!/bin/sh\ncat >/dev/null\ncase \"$1\" in\n  health) printf '%s\\n' '{}' ;;\n  probe-model) printf '%s\\n' '{}' ;;\n  *) exit 64 ;;\nesac\n",
                quote(&health),
                quote(&probe),
            ),
        );
        CompanionClient::from_command(path, Vec::<std::ffi::OsString>::new())
    }

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
    fn legacy_unavailable_readiness_drops_install_command() {
        let value = json!({
            "supported": false,
            "detail": "the repository config is incomplete",
            "adapter": "qwen3_tts_voice_design",
            "required_backend": "qwen-tts",
            "install_command": "werk backend install qwen-tts",
            "missing_dependencies": ["qwen_tts"]
        });

        let readiness =
            task_readiness_from_model_probe(&value, false, "the repository config is incomplete");

        assert_eq!(readiness.status, TaskReadinessStatus::Unavailable);
        assert_eq!(readiness.required_backend.as_deref(), Some("qwen-tts"));
        assert_eq!(readiness.install_command, None);
    }

    #[test]
    fn structured_non_installable_readiness_drops_install_command() {
        let value = json!({
            "supported": false,
            "detail": "the repository layout is broken",
            "readiness": {
                "status": "unavailable",
                "detail": "the repository layout is broken",
                "adapter": null,
                "required_backend": null,
                "install_command": "werk backend install qwen-tts",
                "fallback_backend": null,
                "missing_dependencies": []
            }
        });

        let readiness =
            task_readiness_from_model_probe(&value, false, "the repository layout is broken");

        assert_eq!(readiness.status, TaskReadinessStatus::Unavailable);
        assert_eq!(readiness.install_command, None);
    }

    #[test]
    fn structured_readiness_preserves_alternative_dependency_routes() {
        let value = json!({
            "supported": false,
            "detail": "an audio generation framework is missing",
            "readiness": {
                "status": "unavailable",
                "detail": "an audio generation framework is missing",
                "missing_dependencies": ["torch", "numpy"],
                "missing_dependency_groups": [{
                    "purpose": "audio_generation_framework",
                    "any_of": [
                        {"all_of": ["diffusers"]},
                        {"all_of": ["transformers"]}
                    ]
                }]
            }
        });

        let readiness = task_readiness_from_model_probe(
            &value,
            false,
            "an audio generation framework is missing",
        );

        assert_eq!(readiness.missing_dependencies, ["torch", "numpy"]);
        assert_eq!(readiness.missing_dependency_groups.len(), 1);
        assert_eq!(
            readiness.missing_dependency_groups[0].purpose,
            "audio_generation_framework"
        );
        assert_eq!(
            readiness.missing_dependency_groups[0].any_of[1].all_of,
            ["transformers"]
        );
    }

    #[cfg(unix)]
    #[test]
    fn qwen_client_discovery_retries_after_backend_appears() {
        let directory = CompanionTestDirectory::new("qwen-retry");
        let store = directory.store();
        let manifest = qwen_voice_design_manifest();
        let general_client = static_probe_client(
            &directory.0,
            "general-companion",
            &json!({"ok": true, "supported": false}),
        );
        let backend = CompanionMediaBackend::with_client(general_client);

        let first = backend.client_for_manifest(&store, &manifest);
        assert!(first.is_err());
        assert!(backend.qwen_client_cache.get().is_none());

        let managed_python = managed_qwen_tts_python(&store);
        write_executable(&managed_python, "#!/bin/sh\nprintf '%s\\n' '0.1.1'\n");

        let (_client, isolated) = backend
            .client_for_manifest(&store, &manifest)
            .expect("newly installed Qwen backend must be discovered");
        assert!(isolated);
        assert!(backend.qwen_client_cache.get().is_some());
    }

    #[cfg(unix)]
    #[test]
    fn broken_qwen_model_is_not_reclassified_as_installable() {
        let directory = CompanionTestDirectory::new("qwen-broken-model");
        let store = directory.store();
        let manifest = qwen_voice_design_manifest();
        let files = store.model_dir(&manifest.id).join("files");
        fs::create_dir_all(&files).expect("create model files directory");
        fs::write(files.join("config.json"), "{broken-json").expect("write broken model config");
        let client = static_probe_client(
            &directory.0,
            "broken-model-companion",
            &json!({
                "ok": true,
                "supported": false,
                "detail": "the repository config/layout is broken",
                "readiness": {
                    "status": "unavailable",
                    "detail": "the repository config/layout is broken",
                    "adapter": null,
                    "required_backend": null,
                    "install_command": "werk backend install qwen-tts",
                    "fallback_backend": null,
                    "missing_dependencies": []
                }
            }),
        );
        let backend = CompanionMediaBackend::with_client(client);

        let probe = backend.probe(&store, &manifest, InferenceTask::TextToSpeech, &[]);
        let readiness = probe.readiness.expect("probe readiness");

        assert!(!probe.available);
        assert_eq!(readiness.status, TaskReadinessStatus::Unavailable);
        assert_eq!(readiness.detail, "the repository config/layout is broken");
        assert_eq!(readiness.install_command, None);
        assert!(probe.detail.contains("config/layout is broken"));
    }

    #[cfg(unix)]
    #[test]
    fn missing_qwen_runtime_is_installable_only_after_voice_design_confirmation() {
        let directory = CompanionTestDirectory::new("qwen-installable");
        let store = directory.store();
        let manifest = qwen_voice_design_manifest();
        let files = store.model_dir(&manifest.id).join("files");
        fs::create_dir_all(&files).expect("create model files directory");
        fs::write(
            files.join("config.json"),
            r#"{"model_type":"qwen3_tts","tts_model_type":"voice_design"}"#,
        )
        .expect("write model config");
        let client = static_probe_client(
            &directory.0,
            "installable-model-companion",
            &json!({
                "ok": true,
                "supported": false,
                "detail": "Qwen VoiceDesign backend is missing",
                "readiness": {
                    "status": "installable",
                    "detail": "Qwen VoiceDesign backend is missing",
                    "adapter": "qwen3_tts_voice_design",
                    "required_backend": "qwen-tts",
                    "install_command": "werk backend install qwen-tts",
                    "fallback_backend": null,
                    "missing_dependencies": ["qwen_tts"]
                }
            }),
        );
        let backend = CompanionMediaBackend::with_client(client);

        let probe = backend.probe(&store, &manifest, InferenceTask::TextToSpeech, &[]);
        let readiness = probe.readiness.expect("probe readiness");

        assert!(!probe.available);
        assert_eq!(readiness.status, TaskReadinessStatus::Installable);
        assert_eq!(
            readiness.install_command.as_deref(),
            Some("werk backend install qwen-tts")
        );
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

        let candidates =
            companion_candidates_with_override(true, None, &[], Some(&health), None, None);
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
    fn companion_protocol_preserves_canonical_audio_tasks() {
        for task in [
            InferenceTask::SpeechTranslation,
            InferenceTask::AudioEventDetection,
            InferenceTask::VoiceActivityDetection,
            InferenceTask::SpeakerIdentification,
            InferenceTask::LanguageIdentification,
            InferenceTask::SpeechEmotionRecognition,
        ] {
            assert_eq!(companion_wire_task(task), task);
        }

        let mut parameters = BTreeMap::new();
        apply_companion_task_parameters(InferenceTask::SpeechTranslation, &mut parameters);
        assert_eq!(
            parameters.get("stt.operation"),
            Some(&ParameterValue::String("translate".to_string()))
        );

        for (expected, legacy_alias) in [
            (InferenceTask::SpeechTranslation, "speech_to_text"),
            (InferenceTask::AudioEventDetection, "audio_classification"),
        ] {
            let error = companion_execution(
                CompanionExecution {
                    ok: true,
                    task: legacy_alias.to_string(),
                    outputs: Vec::new(),
                    metadata: Value::Null,
                    warnings: Vec::new(),
                },
                Path::new("."),
                "media-companion-cpu",
                expected,
            )
            .unwrap_err();
            assert!(error.to_string().contains("response task mismatch"));
        }
    }

    #[test]
    fn companion_candidates_only_offer_offloading_for_diffusers_adapters() {
        let cases = [
            (
                InferenceTask::ImageGeneration,
                RepositoryLayout::Diffusers,
                true,
            ),
            (
                InferenceTask::ImageGeneration,
                RepositoryLayout::SingleFile,
                true,
            ),
            (
                InferenceTask::VideoGeneration,
                RepositoryLayout::Diffusers,
                true,
            ),
            (
                InferenceTask::AudioGeneration,
                RepositoryLayout::Diffusers,
                true,
            ),
            (
                InferenceTask::AudioGeneration,
                RepositoryLayout::Transformers,
                false,
            ),
            (
                InferenceTask::MusicGeneration,
                RepositoryLayout::Transformers,
                false,
            ),
            (
                InferenceTask::TextToSpeech,
                RepositoryLayout::Diffusers,
                false,
            ),
            (
                InferenceTask::SpeechToText,
                RepositoryLayout::Transformers,
                false,
            ),
        ];

        for (task, layout, adapter_supports_offloading) in cases {
            let candidates = companion_candidates_with_override(
                true,
                None,
                &[],
                None,
                Some(RuntimeAccelerator::Cuda),
                Some((task, layout)),
            );

            for candidate in candidates {
                let accelerator_supports_offloading = matches!(
                    candidate.accelerator,
                    RuntimeAccelerator::Cuda | RuntimeAccelerator::Rocm
                );
                assert_eq!(
                    candidate.supports_offloading,
                    adapter_supports_offloading && accelerator_supports_offloading,
                    "unexpected offload support for task {task}, layout {layout}, runtime {}",
                    candidate.id
                );
            }
        }
    }

    #[test]
    fn generic_companion_candidates_do_not_guess_offload_support() {
        let candidates = default_companion_candidates(true, None, &[]);

        assert!(
            candidates
                .iter()
                .all(|candidate| !candidate.supports_offloading)
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
                "model_cache_hit": true,
                "inference_seconds": 4.0,
                "encoding_seconds": 0.5,
                "offload_mode": "model_cpu",
                "offload_request": "component",
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
        assert_eq!(sanitized["backend"]["model_cache_hit"], true);
        assert_eq!(sanitized["backend"]["inference_seconds"], 4.0);
        assert_eq!(sanitized["backend"]["offload_mode"], "model_cpu");
        assert_eq!(sanitized["backend"]["offload_request"], "component");
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

    #[test]
    fn persisted_companion_metadata_rejects_unknown_offload_values() {
        let sanitized = sanitized_companion_metadata(json!({
            "backend": {
                "offload_mode": "component",
                "offload_request": "invented"
            }
        }));

        assert!(sanitized["backend"].get("offload_mode").is_none());
        assert!(sanitized["backend"].get("offload_request").is_none());
    }
}
