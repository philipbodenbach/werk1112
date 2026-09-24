use std::{collections::HashMap, fmt};

use crate::{
    backend::{
        BackendAccelerator, BackendRuntime, RuntimeId, StaticModelSupport, backend_supports_images,
        explain_backend_rejection, is_transformers_compat_model, runtime_descriptor,
        runtime_registry, runtime_static_model_support, runtime_supports_layout,
        runtime_supports_model, runtime_supports_task, vllm_architecture_supports_images,
    },
    capabilities::{InferenceTask, InputModality, OutputModality},
    model_store::{ModelFormat, ModelManifest},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestedBackend {
    Auto,
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
    Mlx,
    Omlx,
    Burn,
    Candle,
    Transformers,
    Vllm,
    LlamaLegacy,
    LlamaHighlevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestCapabilities {
    pub text_generation: bool,
    pub image_input: bool,
    pub embeddings: bool,
    pub streaming: bool,
    pub tool_calling: bool,
    pub task: Option<InferenceTask>,
    pub input_modality: Option<InputModality>,
    pub output_modality: Option<OutputModality>,
}

impl RequestCapabilities {
    pub fn text(streaming: bool) -> Self {
        Self {
            text_generation: true,
            image_input: false,
            embeddings: false,
            streaming,
            tool_calling: false,
            task: None,
            input_modality: None,
            output_modality: None,
        }
    }

    pub fn text_with_images(streaming: bool, image_input: bool) -> Self {
        Self {
            image_input,
            ..Self::text(streaming)
        }
    }

    pub fn with_tool_calling(mut self, tool_calling: bool) -> Self {
        self.tool_calling = tool_calling;
        self
    }

    pub fn for_task(task: InferenceTask) -> Self {
        Self::for_task_with_streaming(task, false)
    }

    pub fn for_task_with_streaming(task: InferenceTask, streaming: bool) -> Self {
        Self {
            text_generation: task == InferenceTask::TextGeneration,
            image_input: task
                .required_input_modalities()
                .contains(&InputModality::Image),
            embeddings: matches!(
                task,
                InferenceTask::TextEmbedding | InferenceTask::AudioEmbedding
            ),
            streaming,
            tool_calling: false,
            task: Some(task),
            input_modality: task.required_input_modalities().first().copied(),
            output_modality: Some(task.output_modality()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAvailability {
    pub runtime_id: RuntimeId,
    /// Availability for this model and request, after the concrete runtime's
    /// compatibility probe. A successful package import alone is insufficient.
    pub available: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeCandidate {
    pub runtime_id: RuntimeId,
    pub priority: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDecisionStatus {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeDecision {
    pub runtime_id: RuntimeId,
    pub display_name: &'static str,
    pub status: RuntimeDecisionStatus,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedRuntime {
    pub model_id: String,
    pub runtime_id: RuntimeId,
    pub display_name: &'static str,
    pub accelerator: BackendAccelerator,
    pub reason: String,
    pub fallback_chain: Vec<RuntimeDecision>,
    pub rejection_reasons: Vec<RuntimeDecision>,
}

impl SelectedRuntime {
    /// Render the routing decision itself; callers must not run selection again
    /// to reconstruct a fallback warning.
    pub fn fallback_note(&self) -> Option<String> {
        let preferred = self.fallback_chain.first()?;
        let reason = preferred
            .reason
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let mut note = format!(
            "Model {}: preferred runtime {} cannot be used ({}). Using compatible fallback {}.",
            self.model_id, preferred.display_name, reason, self.display_name,
        );
        if self.accelerator == BackendAccelerator::Cpu
            && !runtime_descriptor(preferred.runtime_id)
                .accelerators
                .contains(&BackendAccelerator::Cpu)
        {
            note.push_str(" The selected route uses CPU execution.");
        }
        Some(note)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePlan {
    pub requested_backend: RequestedBackend,
    pub request_capabilities: RequestCapabilities,
    pub candidates: Vec<RuntimeDecision>,
    pub selected: Option<SelectedRuntime>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePlanError {
    pub model_id: String,
    pub requested_backend: RequestedBackend,
    pub decisions: Vec<RuntimeDecision>,
}

impl fmt::Display for RuntimePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "no available compatible runtime for model '{}' and requested backend {:?}",
            self.model_id, self.requested_backend
        )?;
        if self.decisions.is_empty() {
            return write!(f, "no runtime candidates matched this request");
        }
        writeln!(f, "tried:")?;
        for decision in &self.decisions {
            writeln!(f, "- {}: {}", decision.display_name, decision.reason)?;
        }
        Ok(())
    }
}

impl std::error::Error for RuntimePlanError {}

pub fn select_runtime(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    available_runtimes: &[RuntimeAvailability],
) -> Result<SelectedRuntime, RuntimePlanError> {
    let plan = plan_runtime(
        manifest,
        requested_backend,
        request_capabilities,
        available_runtimes,
    );
    plan.selected.clone().ok_or(RuntimePlanError {
        model_id: manifest.id.clone(),
        requested_backend,
        decisions: plan.candidates,
    })
}

/// Select from an already constrained and ordered set, preserving explicit
/// backend/device bindings while applying the same compatibility rules as auto.
pub fn select_runtime_from_candidates(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    available_runtimes: &[RuntimeAvailability],
    candidates: &[RuntimeId],
) -> Result<SelectedRuntime, RuntimePlanError> {
    let plan = plan_runtime_from_candidates(
        manifest,
        requested_backend,
        request_capabilities,
        available_runtimes,
        candidates,
    );
    plan.selected.ok_or(RuntimePlanError {
        model_id: manifest.id.clone(),
        requested_backend,
        decisions: plan.candidates,
    })
}

pub fn select_runtime_for_task(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    task: InferenceTask,
    available_runtimes: &[RuntimeAvailability],
) -> Result<SelectedRuntime, RuntimePlanError> {
    select_runtime(
        manifest,
        requested_backend,
        RequestCapabilities::for_task(task),
        available_runtimes,
    )
}

pub fn plan_runtime(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    available_runtimes: &[RuntimeAvailability],
) -> RuntimePlan {
    let candidate_ids = runtime_candidate_ids_for_plan(
        manifest,
        requested_backend,
        request_capabilities,
        available_runtimes,
    );
    plan_runtime_from_candidates(
        manifest,
        requested_backend,
        request_capabilities,
        available_runtimes,
        &candidate_ids,
    )
}

fn plan_runtime_from_candidates(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    available_runtimes: &[RuntimeAvailability],
    candidate_ids: &[RuntimeId],
) -> RuntimePlan {
    let availability = availability_map(available_runtimes);
    let decisions = candidate_ids.iter().copied().map(|runtime_id| {
        (
            candidate_decision(
                manifest,
                requested_backend,
                request_capabilities,
                runtime_id,
                availability.get(&runtime_id),
            ),
            is_preferred_route(
                manifest,
                requested_backend,
                request_capabilities,
                runtime_id,
            ),
        )
    });
    plan_from_decisions(manifest, requested_backend, request_capabilities, decisions)
}

fn plan_from_decisions(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    decisions: impl IntoIterator<Item = (RuntimeDecision, bool)>,
) -> RuntimePlan {
    let mut candidates = Vec::new();
    let mut selected = None;
    let mut rejections = Vec::new();
    let mut fallback_chain = Vec::new();

    for (decision, preferred_route) in decisions {
        let runtime_id = decision.runtime_id;
        let descriptor = runtime_descriptor(runtime_id);
        if decision.status == RuntimeDecisionStatus::Accepted {
            selected = Some(SelectedRuntime {
                model_id: manifest.id.clone(),
                runtime_id,
                display_name: decision.display_name,
                accelerator: descriptor
                    .accelerators
                    .first()
                    .copied()
                    .unwrap_or(BackendAccelerator::Auto),
                reason: decision.reason.clone(),
                fallback_chain,
                rejection_reasons: rejections.clone(),
            });
            candidates.push(decision);
            break;
        }
        if preferred_route {
            fallback_chain.push(decision.clone());
        }
        rejections.push(decision.clone());
        candidates.push(decision);
    }

    RuntimePlan {
        requested_backend,
        request_capabilities,
        candidates,
        selected,
    }
}

pub fn plan_runtime_for_task(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    task: InferenceTask,
    available_runtimes: &[RuntimeAvailability],
) -> RuntimePlan {
    plan_runtime(
        manifest,
        requested_backend,
        RequestCapabilities::for_task(task),
        available_runtimes,
    )
}

pub fn runtime_candidates(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
) -> Vec<RuntimeCandidate> {
    runtime_candidate_ids(manifest, requested_backend)
        .into_iter()
        .map(|runtime_id| RuntimeCandidate {
            priority: runtime_descriptor(runtime_id).priority,
            runtime_id,
        })
        .collect()
}

pub fn runtime_candidates_for_task(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    task: InferenceTask,
) -> Vec<RuntimeCandidate> {
    runtime_candidate_ids_for_task(manifest, requested_backend, task)
        .into_iter()
        .map(|runtime_id| RuntimeCandidate {
            priority: runtime_descriptor(runtime_id).priority,
            runtime_id,
        })
        .collect()
}

pub fn runtime_candidate_ids_for_task(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    task: InferenceTask,
) -> Vec<RuntimeId> {
    typed_runtime_candidate_ids(manifest, requested_backend, task)
}

pub fn runtime_candidate_ids(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
) -> Vec<RuntimeId> {
    let mut candidates = match requested_backend {
        RequestedBackend::Auto => auto_candidates(manifest),
        RequestedBackend::Cpu => cpu_candidates(manifest),
        RequestedBackend::Cuda => cuda_candidates(manifest),
        RequestedBackend::Rocm => rocm_candidates(manifest),
        RequestedBackend::Vulkan => vulkan_candidates(manifest),
        RequestedBackend::Metal => metal_candidates(manifest),
        RequestedBackend::Mlx => vec![RuntimeId::MlxVlm, RuntimeId::Mlx],
        RequestedBackend::Omlx => vec![RuntimeId::Omlx],
        RequestedBackend::Burn => burn_candidates(),
        RequestedBackend::Candle => candle_candidates(manifest),
        RequestedBackend::Transformers => vec![RuntimeId::TransformersCompat],
        RequestedBackend::Vllm => vec![RuntimeId::VllmCuda],
        RequestedBackend::LlamaLegacy | RequestedBackend::LlamaHighlevel => Vec::new(),
    };
    // The format/architecture rules determine membership; registry priorities
    // determine the actual order. Hardware profile overrides are applied later.
    candidates.sort_by(|left, right| {
        runtime_descriptor(*right)
            .priority
            .cmp(&runtime_descriptor(*left).priority)
    });
    candidates
}

fn runtime_candidate_ids_for_plan(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    available_runtimes: &[RuntimeAvailability],
) -> Vec<RuntimeId> {
    if let Some(task) = request_capabilities
        .task
        .filter(|task| is_media_task(*task))
    {
        return typed_runtime_candidate_ids(manifest, requested_backend, task);
    }
    let mut candidates = runtime_candidate_ids(manifest, requested_backend);
    if requested_backend == RequestedBackend::Auto
        && manifest.format == ModelFormat::Gguf
        && available_runtimes
            .iter()
            .any(|availability| availability.runtime_id == RuntimeId::LlamaServerRocm)
        && !candidates.contains(&RuntimeId::LlamaServerRocm)
    {
        let prefer_rocm = available_runtimes
            .iter()
            .position(|availability| availability.runtime_id == RuntimeId::LlamaServerRocm)
            .is_some_and(|rocm| {
                available_runtimes
                    .iter()
                    .position(|availability| availability.runtime_id == RuntimeId::LlamaServerCuda)
                    .is_none_or(|cuda| rocm < cuda)
            });
        let insert_at = if prefer_rocm {
            0
        } else {
            llama_rocm_insert_position(&candidates)
        };
        candidates.insert(insert_at, RuntimeId::LlamaServerRocm);
    }
    if matches!(
        requested_backend,
        RequestedBackend::Auto | RequestedBackend::Vllm
    ) && manifest.format == ModelFormat::SafeTensors
        && available_runtimes
            .iter()
            .any(|availability| availability.runtime_id == RuntimeId::VllmRocm)
        && !candidates.contains(&RuntimeId::VllmRocm)
        && runtime_supports_model(
            runtime_descriptor(RuntimeId::VllmRocm),
            &manifest.format,
            manifest.architecture.as_deref(),
        )
    {
        // A verified ROCm endpoint/interpreter must be tried before the
        // generic CUDA vLLM candidate. This keeps both auto and the
        // accelerator-neutral explicit `vllm` route correctly labelled.
        candidates.insert(0, RuntimeId::VllmRocm);
    }
    if let Some(task) = request_capabilities.task {
        candidates
            .retain(|runtime_id| runtime_supports_task(runtime_descriptor(*runtime_id), task));
    }
    candidates
}

fn typed_runtime_candidate_ids(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    task: InferenceTask,
) -> Vec<RuntimeId> {
    if !is_media_task(task) {
        return runtime_candidate_ids(manifest, requested_backend)
            .into_iter()
            .filter(|runtime_id| runtime_supports_task(runtime_descriptor(*runtime_id), task))
            .collect();
    }

    let mut candidates = runtime_registry()
        .iter()
        .filter(|descriptor| descriptor.runtime == BackendRuntime::MediaCompanion)
        .filter(|descriptor| runtime_supports_task(descriptor, task))
        .filter(|descriptor| {
            runtime_supports_layout(descriptor, manifest.metadata.repository_layout)
        })
        .filter(|descriptor| requested_backend_matches_descriptor(requested_backend, descriptor))
        .map(|descriptor| descriptor.id)
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        runtime_descriptor(*right)
            .priority
            .cmp(&runtime_descriptor(*left).priority)
    });
    candidates
}

fn is_media_task(task: InferenceTask) -> bool {
    !matches!(
        task,
        InferenceTask::TextGeneration
            | InferenceTask::TextEmbedding
            | InferenceTask::ImageUnderstanding
    )
}

fn requested_backend_matches_descriptor(
    requested_backend: RequestedBackend,
    descriptor: &crate::backend::RuntimeDescriptor,
) -> bool {
    match requested_backend {
        RequestedBackend::Auto => true,
        RequestedBackend::Cpu => descriptor.accelerators.contains(&BackendAccelerator::Cpu),
        RequestedBackend::Cuda => descriptor.accelerators.contains(&BackendAccelerator::Cuda),
        RequestedBackend::Rocm => descriptor.accelerators.contains(&BackendAccelerator::Rocm),
        RequestedBackend::Metal => descriptor.accelerators.contains(&BackendAccelerator::Metal),
        RequestedBackend::Mlx => {
            matches!(
                descriptor.runtime,
                BackendRuntime::Mlx | BackendRuntime::MlxVlm
            ) || (descriptor.runtime == BackendRuntime::MediaCompanion
                && descriptor.accelerators.contains(&BackendAccelerator::Metal))
        }
        RequestedBackend::Vulkan => descriptor
            .accelerators
            .contains(&BackendAccelerator::Vulkan),
        RequestedBackend::Burn => descriptor.runtime == BackendRuntime::Burn,
        RequestedBackend::Candle => descriptor.runtime == BackendRuntime::Candle,
        RequestedBackend::Transformers => descriptor.runtime == BackendRuntime::TransformersCompat,
        RequestedBackend::Vllm => descriptor.runtime == BackendRuntime::Vllm,
        RequestedBackend::Omlx => descriptor.runtime == BackendRuntime::Omlx,
        RequestedBackend::LlamaLegacy | RequestedBackend::LlamaHighlevel => false,
    }
}

fn llama_rocm_insert_position(candidates: &[RuntimeId]) -> usize {
    candidates
        .iter()
        .position(|id| {
            matches!(
                id,
                RuntimeId::LlamaServerVulkan
                    | RuntimeId::LlamaServerMetal
                    | RuntimeId::LlamaServerCpu
            )
        })
        .unwrap_or(candidates.len())
}

fn auto_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => gguf_auto_candidates(),
        ModelFormat::SafeTensors => safetensors_auto_candidates(manifest),
        ModelFormat::Onnx => onnx_auto_candidates(),
        ModelFormat::Mlx => {
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                vec![
                    RuntimeId::MlxVlm,
                    RuntimeId::Mlx,
                    RuntimeId::Omlx,
                    RuntimeId::CandleMetal,
                    RuntimeId::CandleCpu,
                ]
            } else {
                vec![RuntimeId::MlxVlm, RuntimeId::Mlx]
            }
        }
        ModelFormat::TensorRt
        | ModelFormat::OpenVino
        | ModelFormat::CoreMl
        | ModelFormat::PyTorch
        | ModelFormat::TensorFlow
        | ModelFormat::Unknown => Vec::new(),
    }
}

fn onnx_auto_candidates() -> Vec<RuntimeId> {
    if cfg!(any(windows, target_os = "linux")) {
        vec![RuntimeId::OnnxRuntimeCuda, RuntimeId::OnnxRuntimeCpu]
    } else {
        vec![RuntimeId::OnnxRuntimeCpu]
    }
}

fn gguf_auto_candidates() -> Vec<RuntimeId> {
    if cfg!(any(windows, target_os = "linux")) {
        vec![
            RuntimeId::LlamaServerCuda,
            // ROCm is added by the CLI selection layer only when a ROCm/HIP
            // llama-server is detected or explicitly signaled. Keeping it out
            // of the default pure planner avoids noisy NVIDIA-only auto output.
            RuntimeId::LlamaServerVulkan,
            RuntimeId::LlamaServerCpu,
            RuntimeId::CandleCuda,
            RuntimeId::CandleCpu,
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            RuntimeId::LlamaServerMetal,
            RuntimeId::LlamaServerCpu,
            RuntimeId::CandleMetal,
            RuntimeId::CandleCpu,
        ]
    } else {
        vec![RuntimeId::LlamaServerCpu, RuntimeId::CandleCpu]
    }
}

fn safetensors_auto_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    if is_transformers_compat_model(manifest) {
        return vec![RuntimeId::TransformersCompat];
    }
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        vec![
            RuntimeId::MlxVlm,
            RuntimeId::Mlx,
            RuntimeId::Omlx,
            RuntimeId::CandleMetal,
            RuntimeId::CandleCpu,
        ]
    } else if cfg!(target_os = "macos") {
        vec![RuntimeId::CandleMetal, RuntimeId::CandleCpu]
    } else if cfg!(any(windows, target_os = "linux")) {
        let mut candidates = Vec::new();
        candidates.extend(vllm_auto_candidates(manifest));
        candidates.extend([RuntimeId::CandleCuda, RuntimeId::CandleCpu]);
        candidates
    } else {
        vec![RuntimeId::CandleCpu]
    }
}

fn vllm_auto_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    if runtime_supports_model(
        runtime_descriptor(RuntimeId::VllmCuda),
        &manifest.format,
        manifest.architecture.as_deref(),
    ) {
        vec![RuntimeId::VllmCuda]
    } else {
        Vec::new()
    }
}

fn cpu_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => vec![RuntimeId::LlamaServerCpu],
        ModelFormat::SafeTensors => vec![RuntimeId::CandleCpu],
        ModelFormat::Onnx => vec![RuntimeId::OnnxRuntimeCpu],
        _ => Vec::new(),
    }
}

fn cuda_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => vec![RuntimeId::LlamaServerCuda],
        ModelFormat::SafeTensors => {
            let mut candidates = Vec::new();
            candidates.extend(vllm_auto_candidates(manifest));
            candidates.push(RuntimeId::CandleCuda);
            candidates
        }
        ModelFormat::Onnx => vec![RuntimeId::OnnxRuntimeCuda],
        _ => Vec::new(),
    }
}

fn rocm_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => vec![RuntimeId::LlamaServerRocm],
        ModelFormat::SafeTensors => vec![RuntimeId::VllmRocm],
        ModelFormat::Onnx => vec![RuntimeId::OnnxRuntimeRocm],
        _ => Vec::new(),
    }
}

fn burn_candidates() -> Vec<RuntimeId> {
    if cfg!(feature = "burn-cuda") && cfg!(any(windows, target_os = "linux")) {
        vec![RuntimeId::BurnCuda]
    } else if cfg!(feature = "burn-cpu") {
        vec![RuntimeId::BurnCpu]
    } else {
        Vec::new()
    }
}

fn vulkan_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => vec![RuntimeId::LlamaServerVulkan],
        ModelFormat::SafeTensors => Vec::new(),
        _ => Vec::new(),
    }
}

fn metal_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf if cfg!(target_os = "macos") => {
            vec![RuntimeId::LlamaServerMetal, RuntimeId::CandleMetal]
        }
        ModelFormat::SafeTensors => vec![RuntimeId::CandleMetal],
        _ => Vec::new(),
    }
}

fn candle_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf | ModelFormat::SafeTensors => {
            if cfg!(target_os = "macos") {
                vec![RuntimeId::CandleMetal, RuntimeId::CandleCpu]
            } else if cfg!(any(windows, target_os = "linux")) {
                vec![RuntimeId::CandleCuda, RuntimeId::CandleCpu]
            } else {
                vec![RuntimeId::CandleCpu]
            }
        }
        _ => Vec::new(),
    }
}

fn candidate_decision(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    runtime_id: RuntimeId,
    availability: Option<&RuntimeAvailability>,
) -> RuntimeDecision {
    let descriptor = runtime_descriptor(runtime_id);
    let reason = rejection_reason(
        manifest,
        requested_backend,
        request_capabilities,
        runtime_id,
        availability,
    );
    match reason {
        Some(reason) => RuntimeDecision {
            runtime_id,
            display_name: descriptor.display_name,
            status: RuntimeDecisionStatus::Rejected,
            reason,
        },
        None => RuntimeDecision {
            runtime_id,
            display_name: descriptor.display_name,
            status: RuntimeDecisionStatus::Accepted,
            reason: selection_reason(manifest, requested_backend, descriptor.runtime),
        },
    }
}

/// Static checks shared by execution callers before running any external probe.
pub fn runtime_static_rejection(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    runtime_id: RuntimeId,
) -> Option<String> {
    rejection_reason(
        manifest,
        requested_backend,
        request_capabilities,
        runtime_id,
        Some(&RuntimeAvailability {
            runtime_id,
            available: true,
            reason: None,
        }),
    )
}

fn is_preferred_route(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    runtime_id: RuntimeId,
) -> bool {
    // A backend for a different task, architecture, modality, or host platform
    // was never a preferred route. Keep its rejection in the full diagnostics.
    let host_eligible = match runtime_id {
        RuntimeId::LlamaServerMetal | RuntimeId::CandleMetal | RuntimeId::MediaCompanionMetal => {
            cfg!(target_os = "macos")
        }
        RuntimeId::Mlx | RuntimeId::MlxVlm | RuntimeId::Omlx => {
            cfg!(all(target_os = "macos", target_arch = "aarch64"))
        }
        RuntimeId::LlamaServerCuda
        | RuntimeId::LlamaServerRocm
        | RuntimeId::LlamaServerVulkan
        | RuntimeId::CandleCuda
        | RuntimeId::VllmCuda
        | RuntimeId::VllmRocm
        | RuntimeId::OnnxRuntimeCuda
        | RuntimeId::OnnxRuntimeRocm
        | RuntimeId::MediaCompanionCuda
        | RuntimeId::MediaCompanionRocm => cfg!(any(windows, target_os = "linux")),
        _ => true,
    };
    host_eligible
        && runtime_static_rejection(
            manifest,
            requested_backend,
            request_capabilities,
            runtime_id,
        )
        .is_none()
}

fn rejection_reason(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    runtime_id: RuntimeId,
    availability: Option<&RuntimeAvailability>,
) -> Option<String> {
    let descriptor = runtime_descriptor(runtime_id);
    if !requested_backend_matches_descriptor(requested_backend, descriptor) {
        return Some(format!(
            "runtime does not match the explicit {requested_backend:?} backend/device binding"
        ));
    }
    if request_capabilities.tool_calling && !descriptor.supports_tool_calling() {
        return Some("runtime does not support OpenAI tool calling".to_string());
    }
    if let Some(task) = request_capabilities.task {
        if !manifest.supports_task(task) {
            return Some(format!("model does not advertise support for task {task}"));
        }
        if !runtime_supports_task(descriptor, task) {
            return Some(format!("runtime does not support task {task}"));
        }
        if !runtime_supports_layout(descriptor, manifest.metadata.repository_layout) {
            return Some(format!(
                "runtime does not support {} repository layout",
                manifest.metadata.repository_layout
            ));
        }
        if let Some(input_modality) = request_capabilities.input_modality {
            if !task.required_input_modalities().contains(&input_modality) {
                return Some(format!(
                    "input modality {input_modality} is incompatible with task {task}"
                ));
            }
            if !manifest.metadata.input_modalities.is_empty()
                && !manifest.metadata.input_modalities.contains(&input_modality)
            {
                return Some(format!(
                    "model does not advertise support for {input_modality} input"
                ));
            }
        }
        if let Some(output_modality) = request_capabilities.output_modality {
            if output_modality != task.output_modality() {
                return Some(format!(
                    "output modality {output_modality} is incompatible with task {task}"
                ));
            }
            if !manifest.metadata.output_modalities.is_empty()
                && !manifest
                    .metadata
                    .output_modalities
                    .contains(&output_modality)
            {
                return Some(format!(
                    "model does not advertise support for {output_modality} output"
                ));
            }
        }
    }
    if runtime_static_model_support(
        descriptor,
        &manifest.format,
        manifest.architecture.as_deref(),
    ) == StaticModelSupport::Unsupported
    {
        return Some(model_support_rejection(manifest, descriptor.runtime));
    }
    if descriptor.runtime == BackendRuntime::Candle
        && manifest.format == ModelFormat::SafeTensors
        && let Some(quantization) = manifest.metadata.quantization.as_deref()
        && [
            "awq", "gptq", "nf4", "int4", "int8", "fp8", "mxfp4", "affine",
        ]
        .iter()
        .any(|layout| quantization.to_ascii_lowercase().contains(layout))
    {
        return Some(format!(
            "Candle does not support the quantized safetensors layout '{quantization}'"
        ));
    }
    if runtime_id == RuntimeId::MlxVlm
        && !request_capabilities.image_input
        && request_capabilities.task != Some(InferenceTask::ImageUnderstanding)
    {
        return Some(
            "MLX-VLM is reserved for image requests; text-only MLX uses mlx-lm".to_string(),
        );
    }
    if request_capabilities.task.is_none() {
        if request_capabilities.image_input
            && !manifest.supports_task(InferenceTask::ImageUnderstanding)
        {
            return Some("model does not advertise image-understanding".to_string());
        }
        if request_capabilities.text_generation && !descriptor.capabilities.text_generation {
            return Some("runtime does not support text generation".to_string());
        }
        if request_capabilities.image_input && !descriptor.capabilities.vision_language {
            return Some("runtime is not VLM-capable".to_string());
        }
        if request_capabilities.embeddings && !descriptor.capabilities.embeddings {
            return Some("runtime does not support embeddings".to_string());
        }
    }
    let requests_vision = request_capabilities.image_input
        || request_capabilities.task == Some(InferenceTask::ImageUnderstanding);
    if requests_vision
        && descriptor.runtime == BackendRuntime::Vllm
        && !vllm_architecture_supports_images(manifest.architecture.as_deref())
    {
        return Some("vLLM does not support image input for this architecture".to_string());
    }
    if request_capabilities.streaming && !descriptor.capabilities.streaming {
        return Some("runtime does not support streaming".to_string());
    }
    if let Some(reason) = explain_backend_rejection(
        descriptor.runtime,
        &manifest.format,
        request_capabilities.task.is_none() && request_capabilities.image_input,
    ) {
        return Some(reason.to_string());
    }
    if request_capabilities.task.is_none()
        && request_capabilities.image_input
        && !backend_supports_images(descriptor.runtime)
    {
        return Some("runtime is not VLM-capable".to_string());
    }
    if !descriptor.implemented {
        return Some(unimplemented_runtime_rejection(
            manifest,
            descriptor.runtime,
        ));
    }
    match availability {
        Some(availability) if availability.available => None,
        Some(availability) => Some(
            availability
                .reason
                .clone()
                .unwrap_or_else(|| "runtime is unavailable".to_string()),
        ),
        None => Some("runtime availability was not reported".to_string()),
    }
}

fn model_support_rejection(manifest: &ModelManifest, runtime: BackendRuntime) -> String {
    if runtime == BackendRuntime::Candle {
        return match manifest.architecture.as_deref() {
            Some(architecture) => format!(
                "Candle does not support architecture '{architecture}' in {:?} format",
                manifest.format
            ),
            None => "model metadata is missing the architecture required by Candle".to_string(),
        };
    }
    match (runtime, &manifest.format) {
        (BackendRuntime::Vllm, ModelFormat::SafeTensors) => {
            "vLLM is not selected for this architecture".to_string()
        }
        (BackendRuntime::MlxVlm, ModelFormat::Mlx | ModelFormat::SafeTensors) => {
            "MLX-VLM is selected for supported VLM architectures".to_string()
        }
        (BackendRuntime::TransformersCompat, ModelFormat::SafeTensors) => {
            "Transformers compatibility is selected for raw ChatGLM/GLM repositories".to_string()
        }
        _ => "model format or architecture is not supported".to_string(),
    }
}

fn unimplemented_runtime_rejection(manifest: &ModelManifest, runtime: BackendRuntime) -> String {
    let _ = (manifest, runtime);
    "runtime integration is not implemented yet".to_string()
}

fn selection_reason(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    runtime: BackendRuntime,
) -> String {
    match (manifest.format.clone(), runtime, requested_backend) {
        (ModelFormat::Gguf, BackendRuntime::LlamaServer, _) => {
            "GGUF hot path uses persistent llama.cpp server".to_string()
        }
        (ModelFormat::SafeTensors, BackendRuntime::OnnxRuntime, _) => {
            "HF safetensors hot path uses managed ONNX Runtime artifacts".to_string()
        }
        (ModelFormat::SafeTensors, BackendRuntime::Burn, _) => {
            "HF safetensors hot path uses Burn".to_string()
        }
        (ModelFormat::SafeTensors, BackendRuntime::TransformersCompat, _) => {
            "raw ChatGLM/GLM compatibility route uses Transformers trust_remote_code".to_string()
        }
        (ModelFormat::SafeTensors, BackendRuntime::Vllm, RequestedBackend::Vllm) => {
            "explicit vLLM route requested".to_string()
        }
        (ModelFormat::SafeTensors, BackendRuntime::Vllm, _) => {
            "HF safetensors CUDA hot path uses vLLM for supported architectures".to_string()
        }
        (_, BackendRuntime::MediaCompanion, _) => {
            "typed media task uses the managed media companion".to_string()
        }
        (_, BackendRuntime::Candle, RequestedBackend::Candle) => {
            "explicit Candle route requested".to_string()
        }
        (_, BackendRuntime::Candle, _) => {
            "fallback runtime supports the selected model architecture".to_string()
        }
        (ModelFormat::Mlx, BackendRuntime::MlxVlm, _) => {
            "MLX VLM image request uses mlx-vlm".to_string()
        }
        (ModelFormat::Mlx, BackendRuntime::Mlx, _) => "MLX model uses mlx-lm".to_string(),
        (_, BackendRuntime::MlxVlm, _) => {
            "MLX VLM runtime selected for compatible model".to_string()
        }
        (_, BackendRuntime::Mlx, _) => "MLX runtime selected for compatible model".to_string(),
        (_, BackendRuntime::Omlx, RequestedBackend::Omlx) => {
            "explicit oMLX runtime selected for compatible model".to_string()
        }
        (_, BackendRuntime::Omlx, _) => "oMLX runtime selected for compatible model".to_string(),
        (_, _, RequestedBackend::Cpu) => "best CPU runtime for this model".to_string(),
        (_, _, RequestedBackend::Cuda) => "best CUDA runtime for this model".to_string(),
        (_, _, RequestedBackend::Rocm) => "best ROCm runtime for this model".to_string(),
        (_, _, RequestedBackend::Vulkan) => "best Vulkan runtime for this model".to_string(),
        (_, _, RequestedBackend::Metal) => "best Metal runtime for this model".to_string(),
        (_, _, RequestedBackend::Transformers) => {
            "explicit Transformers compatibility route requested".to_string()
        }
        _ => "best available runtime for this model".to_string(),
    }
}

fn availability_map(
    available_runtimes: &[RuntimeAvailability],
) -> HashMap<RuntimeId, RuntimeAvailability> {
    available_runtimes
        .iter()
        .cloned()
        .map(|availability| (availability.runtime_id, availability))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::{InferenceTask, RepositoryLayout};
    use crate::model_store::{ModelManifest, ModelSource};

    #[test]
    fn preferred_available_route_has_no_fallback_diagnostic() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let selected = select_runtime(
            &manifest,
            RequestedBackend::Cuda,
            RequestCapabilities::text(true),
            &[
                available(RuntimeId::VllmCuda),
                available(RuntimeId::CandleCuda),
            ],
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::VllmCuda);
        assert!(selected.fallback_chain.is_empty());
        assert_eq!(selected.fallback_note(), None);
    }

    #[test]
    fn supported_standard_and_gpt_oss_models_keep_the_mlx_lm_route() {
        for (format, architecture, quantization) in [
            (ModelFormat::Mlx, "llama", None),
            (ModelFormat::Mlx, "gpt_oss", Some("MXFP4")),
            (ModelFormat::SafeTensors, "gpt_oss", Some("MXFP4")),
        ] {
            let mut manifest = manifest(format, Some(architecture));
            let backends: &[RequestedBackend] = if manifest.format == ModelFormat::SafeTensors {
                manifest.metadata.repository_layout = RepositoryLayout::Transformers;
                &[RequestedBackend::Mlx]
            } else {
                manifest.metadata.repository_layout = RepositoryLayout::Mlx;
                &[RequestedBackend::Auto, RequestedBackend::Mlx]
            };
            manifest.metadata.quantization = quantization.map(str::to_owned);
            // Simulate a successful model-specific probe. The Python tests
            // verify the installed loader and GPT-OSS quantization_config path.
            for &requested_backend in backends {
                let selected = select_runtime(
                    &manifest,
                    requested_backend,
                    RequestCapabilities::text(true),
                    &[available(RuntimeId::Mlx), available(RuntimeId::CandleCpu)],
                )
                .unwrap();
                assert_eq!(selected.runtime_id, RuntimeId::Mlx);
                if manifest.format == ModelFormat::Mlx {
                    assert_eq!(selected.reason, "MLX model uses mlx-lm");
                }
                assert!(selected.fallback_chain.is_empty());
                assert!(selected.fallback_note().is_none());
            }
        }
    }

    #[test]
    fn missing_or_incompatible_preferred_runtime_reports_the_actual_fallback() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        for reason in [
            "vLLM is not installed",
            "installed vLLM fixture version cannot resolve phi3",
        ] {
            let selected = select_runtime(
                &manifest,
                RequestedBackend::Cuda,
                RequestCapabilities::text(true),
                &[
                    RuntimeAvailability {
                        runtime_id: RuntimeId::VllmCuda,
                        available: false,
                        reason: Some(reason.into()),
                    },
                    available(RuntimeId::CandleCuda),
                ],
            )
            .unwrap();
            assert_eq!(selected.runtime_id, RuntimeId::CandleCuda);
            let note = selected.fallback_note().unwrap();
            assert!(note.contains(&manifest.id));
            assert!(note.contains("preferred runtime vLLM CUDA"));
            assert!(note.contains(reason));
            assert!(note.contains("compatible fallback Candle CUDA"));
            assert_eq!(selected.fallback_chain, selected.rejection_reasons);
        }
    }

    #[test]
    fn typed_text_request_preserves_detected_rocm_hardware_preference() {
        let mut manifest = manifest(ModelFormat::SafeTensors, Some("qwen3"));
        manifest.metadata.tasks = vec![InferenceTask::TextGeneration];
        let selected = select_runtime_for_task(
            &manifest,
            RequestedBackend::Auto,
            InferenceTask::TextGeneration,
            &[
                available(RuntimeId::VllmRocm),
                available(RuntimeId::VllmCuda),
            ],
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::VllmRocm);
        assert!(selected.fallback_note().is_none());
    }

    #[test]
    fn candidate_order_applies_registry_priorities_before_hardware_overrides() {
        for format in [
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            ModelFormat::Mlx,
            ModelFormat::Onnx,
        ] {
            let manifest = manifest(format, Some("qwen3"));
            let candidates = runtime_candidates(&manifest, RequestedBackend::Auto);
            assert!(
                candidates
                    .windows(2)
                    .all(|pair| pair[0].priority >= pair[1].priority)
            );
        }
    }

    #[test]
    fn tool_calling_preserves_restricted_candidates_and_device_requirements() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let available = [
            available(RuntimeId::CandleCpu),
            available(RuntimeId::CandleCuda),
        ];
        let selected = select_runtime_from_candidates(
            &manifest,
            RequestedBackend::Cuda,
            RequestCapabilities::text(true).with_tool_calling(true),
            &available,
            &[RuntimeId::CandleCuda],
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::CandleCuda);
        let error = select_runtime_from_candidates(
            &manifest,
            RequestedBackend::Cuda,
            RequestCapabilities::text(true),
            &available,
            &[RuntimeId::CandleCpu],
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("explicit Cuda backend/device binding")
        );
    }

    #[test]
    fn installed_candle_rejects_unsupported_architecture_and_quantization_before_generation() {
        for (architecture, quantization, expected) in [
            ("deepseek_v4", None, "architecture 'deepseek_v4'"),
            ("llama", Some("MXFP4/MXFP8"), "quantized safetensors layout"),
            ("qwen3", Some("gptq-4bit"), "quantized safetensors layout"),
        ] {
            let mut manifest = manifest(ModelFormat::SafeTensors, Some(architecture));
            manifest.metadata.quantization = quantization.map(str::to_owned);
            let error = select_runtime(
                &manifest,
                RequestedBackend::Cpu,
                RequestCapabilities::text(true),
                &[available(RuntimeId::CandleCpu)],
            )
            .unwrap_err();
            assert!(error.to_string().contains(expected));
        }
    }

    fn deepseek_fixture() -> ModelManifest {
        // Metadata only: no weights and no claim of a real supporting backend.
        let mut manifest = manifest(ModelFormat::Mlx, Some("deepseek_v4"));
        manifest.id = "Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX".into();
        manifest.metadata.repository_layout = RepositoryLayout::Mlx;
        manifest.metadata.quantization = Some("MXFP4/MXFP8".into());
        manifest
    }

    fn unsupported_deepseek_mlx() -> RuntimeAvailability {
        RuntimeAvailability {
            runtime_id: RuntimeId::Mlx,
            available: false,
            reason: Some("installed mlx-lm fixture version does not support architecture deepseek_v4 (mixed MXFP4/MXFP8)".into()),
        }
    }

    #[test]
    fn deepseek_simulated_supported_mlx_is_selected_without_fallback() {
        let manifest = deepseek_fixture();
        // Availability represents a probe confirming both deepseek_v4 and
        // mixed MXFP4/MXFP8 support, not evidence of real runtime support.
        let selected = select_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &[available(RuntimeId::Mlx)],
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::Mlx);
        assert_eq!(selected.reason, "MLX model uses mlx-lm");
        assert!(selected.fallback_chain.is_empty());
        assert!(selected.fallback_note().is_none());
    }

    #[test]
    fn installed_omlx_does_not_displace_compatible_mlx() {
        // Explicit candidate injection exercises Apple routing on Linux CI too.
        // Availability fixtures represent model-specific probes, not inference.
        for manifest in [
            manifest(ModelFormat::Mlx, Some("llama")),
            manifest(ModelFormat::SafeTensors, Some("gpt_oss")),
            deepseek_fixture(),
        ] {
            let mut candidates = [RuntimeId::Omlx, RuntimeId::Mlx];
            candidates.sort_by_key(|id| std::cmp::Reverse(runtime_descriptor(*id).priority));
            let selected = select_runtime_from_candidates(
                &manifest,
                RequestedBackend::Auto,
                RequestCapabilities::text(true),
                &[available(RuntimeId::Omlx), available(RuntimeId::Mlx)],
                &candidates,
            )
            .unwrap();
            assert_eq!(selected.runtime_id, RuntimeId::Mlx);
            assert!(selected.rejection_reasons.is_empty());
            assert!(selected.fallback_note().is_none());
        }
    }

    #[test]
    fn compatible_omlx_handles_deepseek_after_mlx_probe_rejection() {
        let manifest = deepseek_fixture();
        let selected = select_runtime_from_candidates(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &[unsupported_deepseek_mlx(), available(RuntimeId::Omlx)],
            &[RuntimeId::Mlx, RuntimeId::Omlx, RuntimeId::CandleCpu],
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::Omlx);
        assert_eq!(selected.rejection_reasons.len(), 1);
        assert!(selected.rejection_reasons[0].reason.contains("deepseek_v4"));
        // The real logger suppresses Apple-only preferred routes on other hosts.
        // Exercise its same decision reducer with the Apple host eligibility.
        let selected = plan_from_decisions(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            [
                (selected.rejection_reasons[0].clone(), true),
                (
                    candidate_decision(
                        &manifest,
                        RequestedBackend::Auto,
                        RequestCapabilities::text(true),
                        RuntimeId::Omlx,
                        Some(&available(RuntimeId::Omlx)),
                    ),
                    true,
                ),
            ],
        )
        .selected
        .unwrap();
        let note = selected.fallback_note().unwrap();
        assert!(note.contains("deepseek_v4"));
        assert!(note.contains("compatible fallback oMLX"));
    }

    #[test]
    fn missing_or_incompatible_omlx_preserves_each_probe_failure() {
        let manifest = deepseek_fixture();
        for reason in [
            "oMLX executable is missing",
            "oMLX MXFP8 support is unverified",
        ] {
            let error = select_runtime_from_candidates(
                &manifest,
                RequestedBackend::Auto,
                RequestCapabilities::text(true),
                &[
                    unsupported_deepseek_mlx(),
                    RuntimeAvailability {
                        runtime_id: RuntimeId::Omlx,
                        available: false,
                        reason: Some(reason.into()),
                    },
                    available(RuntimeId::CandleCpu),
                ],
                &[RuntimeId::Mlx, RuntimeId::Omlx, RuntimeId::CandleCpu],
            )
            .unwrap_err();
            assert!(error.to_string().contains(reason));
            assert!(error.to_string().contains("deepseek_v4"));
            assert_eq!(error.decisions.len(), 3);
        }
    }

    #[test]
    fn explicit_omlx_is_bound_and_does_not_change_explicit_mlx() {
        let manifest = deepseek_fixture();
        let both = [available(RuntimeId::Mlx), available(RuntimeId::Omlx)];
        for (requested, expected) in [
            (RequestedBackend::Mlx, RuntimeId::Mlx),
            (RequestedBackend::Omlx, RuntimeId::Omlx),
        ] {
            let selected =
                select_runtime(&manifest, requested, RequestCapabilities::text(true), &both)
                    .unwrap();
            assert_eq!(selected.runtime_id, expected);
            assert!(selected.fallback_note().is_none());
        }
        let error = select_runtime(
            &manifest,
            RequestedBackend::Omlx,
            RequestCapabilities::text(true),
            &[available(RuntimeId::Mlx)],
        )
        .unwrap_err();
        assert_eq!(error.decisions.len(), 1);
        assert_eq!(error.decisions[0].runtime_id, RuntimeId::Omlx);
    }

    #[test]
    fn omlx_respects_devices_formats_modalities_and_runtime_availability() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("llama"));
        for requested in [
            RequestedBackend::Cpu,
            RequestedBackend::Cuda,
            RequestedBackend::Rocm,
            RequestedBackend::Metal,
        ] {
            let error = select_runtime_from_candidates(
                &manifest,
                requested,
                RequestCapabilities::text(true),
                &[available(RuntimeId::Omlx)],
                &[RuntimeId::Omlx],
            )
            .unwrap_err();
            assert!(error.to_string().contains("backend/device binding"));
        }
        let tool_request = RequestCapabilities::text(true).with_tool_calling(true);
        let selected = select_runtime(
            &manifest,
            RequestedBackend::Omlx,
            tool_request,
            &[available(RuntimeId::Omlx)],
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::Omlx);
        let error = select_runtime(
            &manifest,
            RequestedBackend::Omlx,
            tool_request,
            &[RuntimeAvailability {
                runtime_id: RuntimeId::Omlx,
                available: false,
                reason: Some("installed oMLX cannot load this model".into()),
            }],
        )
        .unwrap_err();
        assert!(error.to_string().contains("cannot load this model"));
        for request in [
            RequestCapabilities::text_with_images(true, true),
            RequestCapabilities::for_task(InferenceTask::TextEmbedding),
        ] {
            assert!(
                select_runtime(
                    &manifest,
                    RequestedBackend::Omlx,
                    request,
                    &[available(RuntimeId::Omlx)]
                )
                .is_err()
            );
        }
        let mut gguf = manifest.clone();
        gguf.format = ModelFormat::Gguf;
        assert!(
            select_runtime(
                &gguf,
                RequestedBackend::Omlx,
                RequestCapabilities::text(true),
                &[available(RuntimeId::Omlx)]
            )
            .is_err()
        );
    }

    #[test]
    fn auto_omlx_membership_is_limited_to_apple_text_routes() {
        for format in [ModelFormat::Mlx, ModelFormat::SafeTensors] {
            let manifest = manifest(format, Some("llama"));
            let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Auto);
            assert_eq!(
                candidates.contains(&RuntimeId::Omlx),
                cfg!(all(target_os = "macos", target_arch = "aarch64"))
            );
            if let Some(omlx) = candidates.iter().position(|id| *id == RuntimeId::Omlx) {
                assert_eq!(candidates[omlx - 1], RuntimeId::Mlx);
            }
        }
    }

    #[test]
    fn deepseek_without_a_compatible_runtime_fails_with_architecture_probe_reason() {
        let manifest = deepseek_fixture();
        let error = select_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &[unsupported_deepseek_mlx(), available(RuntimeId::CandleCpu)],
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains(&manifest.id));
        assert!(message.contains(
            "installed mlx-lm fixture version does not support architecture deepseek_v4"
        ));
        assert!(message.contains("MXFP4/MXFP8"));
        assert!(
            error
                .decisions
                .iter()
                .all(|decision| decision.status == RuntimeDecisionStatus::Rejected)
        );
    }

    #[test]
    fn deepseek_simulated_compatible_candidate_is_selected_and_reported() {
        let manifest = deepseek_fixture();
        let capabilities = RequestCapabilities::text(true);
        let mlx_rejection = candidate_decision(
            &manifest,
            RequestedBackend::Auto,
            capabilities,
            RuntimeId::Mlx,
            Some(&unsupported_deepseek_mlx()),
        );
        assert_eq!(mlx_rejection.status, RuntimeDecisionStatus::Rejected);
        // Inject a synthetic accepted decision only at the common decision
        // reducer. The production registry still rejects this model for Candle;
        // no real DeepSeek, Candle, MLX-VLM, or oMLX support is invented here.
        let simulated = RuntimeDecision {
            runtime_id: RuntimeId::CandleCpu,
            display_name: "simulated compatible runtime",
            status: RuntimeDecisionStatus::Accepted,
            reason: "test fixture reports model-specific compatibility".into(),
        };
        let selected = plan_from_decisions(
            &manifest,
            RequestedBackend::Auto,
            capabilities,
            [(mlx_rejection, true), (simulated, true)],
        )
        .selected
        .unwrap();
        assert_eq!(selected.display_name, "simulated compatible runtime");
        let note = selected.fallback_note().unwrap();
        assert!(note.contains(&manifest.id));
        assert!(note.contains("deepseek_v4"));
        assert!(note.contains("compatible fallback simulated compatible runtime"));
        assert!(!runtime_supports_model(
            runtime_descriptor(RuntimeId::CandleCpu),
            &manifest.format,
            manifest.architecture.as_deref()
        ));
    }

    fn available(runtime_id: RuntimeId) -> RuntimeAvailability {
        RuntimeAvailability {
            runtime_id,
            available: true,
            reason: None,
        }
    }

    #[test]
    fn tool_calling_can_fall_back_to_generic_adapter_within_device_binding() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let tool_request = RequestCapabilities::text(true).with_tool_calling(true);
        let unavailable_vllm = [
            RuntimeAvailability {
                runtime_id: RuntimeId::VllmCuda,
                available: false,
                reason: Some("vLLM unavailable".to_string()),
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::CandleCuda,
                available: true,
                reason: None,
            },
        ];

        let rejected = plan_runtime(
            &manifest,
            RequestedBackend::Cuda,
            tool_request,
            &unavailable_vllm,
        );
        assert_eq!(rejected.selected.unwrap().runtime_id, RuntimeId::CandleCuda);

        let available_vllm = [
            RuntimeAvailability {
                runtime_id: RuntimeId::VllmCuda,
                available: true,
                reason: None,
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::CandleCuda,
                available: true,
                reason: None,
            },
        ];
        let selected = select_runtime(
            &manifest,
            RequestedBackend::Cuda,
            tool_request,
            &available_vllm,
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::VllmCuda);
    }

    #[test]
    fn gguf_auto_prefers_llama_server_before_candle() {
        let manifest = manifest(ModelFormat::Gguf, Some("llama"));
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Auto);
        if cfg!(any(windows, target_os = "linux")) {
            assert_eq!(candidates[0], RuntimeId::LlamaServerCuda);
            assert_eq!(candidates[1], RuntimeId::LlamaServerVulkan);
            assert_eq!(candidates[2], RuntimeId::LlamaServerCpu);
        } else if cfg!(target_os = "macos") {
            assert_eq!(candidates[0], RuntimeId::LlamaServerMetal);
            assert_eq!(candidates[1], RuntimeId::LlamaServerCpu);
        } else {
            assert_eq!(candidates[0], RuntimeId::LlamaServerCpu);
        }
        assert!(candidates.contains(&RuntimeId::CandleCpu));
    }

    #[test]
    fn safetensors_cuda_uses_vllm_then_candle_without_cpu_fallback() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Cuda);
        assert_eq!(candidates[0], RuntimeId::VllmCuda);
        assert!(
            candidates
                .iter()
                .position(|id| *id == RuntimeId::VllmCuda)
                .unwrap()
                < candidates
                    .iter()
                    .position(|id| *id == RuntimeId::CandleCuda)
                    .unwrap()
        );
        assert!(candidates.contains(&RuntimeId::CandleCuda));
        assert!(!candidates.contains(&RuntimeId::CandleCpu));
        assert!(!candidates.contains(&RuntimeId::BurnCuda));
        assert!(!candidates.contains(&RuntimeId::BurnCpu));
    }

    #[test]
    fn safetensors_auto_tries_vllm_before_candle_for_supported_architectures() {
        let qwen = manifest(ModelFormat::SafeTensors, Some("qwen2"));
        let candidates = runtime_candidate_ids(&qwen, RequestedBackend::Auto);
        if cfg!(any(windows, target_os = "linux")) {
            assert_eq!(candidates[0], RuntimeId::VllmCuda);
            assert!(
                candidates
                    .iter()
                    .position(|id| *id == RuntimeId::VllmCuda)
                    .unwrap()
                    < candidates
                        .iter()
                        .position(|id| *id == RuntimeId::CandleCuda)
                        .unwrap()
            );
            assert!(candidates.contains(&RuntimeId::CandleCpu));
        }
        assert!(!candidates.contains(&RuntimeId::BurnCuda));
        assert!(!candidates.contains(&RuntimeId::BurnCpu));
    }

    #[test]
    fn safetensors_auto_prefers_an_available_rocm_vllm_runtime() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("nemotron_h"));
        let available = [
            RuntimeAvailability {
                runtime_id: RuntimeId::VllmRocm,
                available: true,
                reason: None,
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::VllmCuda,
                available: true,
                reason: None,
            },
        ];

        let selected = select_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &available,
        )
        .unwrap();

        assert_eq!(selected.runtime_id, RuntimeId::VllmRocm);

        let unavailable_rocm = [
            RuntimeAvailability {
                runtime_id: RuntimeId::VllmRocm,
                available: false,
                reason: Some("ROCm runtime unavailable".to_string()),
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::VllmCuda,
                available: true,
                reason: None,
            },
        ];
        let fallback = select_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &unavailable_rocm,
        )
        .unwrap();
        assert_eq!(fallback.runtime_id, RuntimeId::VllmCuda);
        assert!(
            fallback
                .fallback_chain
                .iter()
                .any(|decision| decision.runtime_id == RuntimeId::VllmRocm)
        );
    }

    #[test]
    fn explicit_vllm_uses_rocm_provenance_when_that_runtime_is_available() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("nemotron_h"));
        let available = [
            RuntimeAvailability {
                runtime_id: RuntimeId::VllmRocm,
                available: true,
                reason: None,
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::VllmCuda,
                available: true,
                reason: None,
            },
        ];

        let selected = select_runtime(
            &manifest,
            RequestedBackend::Vllm,
            RequestCapabilities::text(true),
            &available,
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::VllmRocm);
        assert_eq!(selected.accelerator, BackendAccelerator::Rocm);
    }

    #[test]
    fn nemotron_h_safetensors_routes_to_text_only_vllm() {
        for architecture in ["nemotron_h", "nemotron_h_moe"] {
            let nemotron = manifest(ModelFormat::SafeTensors, Some(architecture));
            let candidates = runtime_candidate_ids(&nemotron, RequestedBackend::Vllm);
            assert_eq!(candidates, vec![RuntimeId::VllmCuda]);

            let available = [RuntimeAvailability {
                runtime_id: RuntimeId::VllmCuda,
                available: true,
                reason: None,
            }];
            let selected = select_runtime(
                &nemotron,
                RequestedBackend::Vllm,
                RequestCapabilities::text(true),
                &available,
            )
            .unwrap();
            assert_eq!(selected.runtime_id, RuntimeId::VllmCuda);

            let image_plan = plan_runtime(
                &nemotron,
                RequestedBackend::Vllm,
                RequestCapabilities::text_with_images(true, true),
                &available,
            );
            assert!(image_plan.selected.is_none());
            assert!(
                image_plan
                    .candidates
                    .iter()
                    .any(|decision| decision.reason.contains("image-understanding"))
            );
        }
    }

    #[test]
    fn qwen3_vl_image_request_routes_to_vllm_but_text_qwen3_does_not() {
        let available = [RuntimeAvailability {
            runtime_id: RuntimeId::VllmCuda,
            available: true,
            reason: None,
        }];
        let mut vision = manifest(ModelFormat::SafeTensors, Some("qwen3_vl_moe"));
        vision.metadata.tasks = vec![
            InferenceTask::TextGeneration,
            InferenceTask::ImageUnderstanding,
        ];

        let selected = select_runtime(
            &vision,
            RequestedBackend::Vllm,
            RequestCapabilities::text_with_images(true, true),
            &available,
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::VllmCuda);

        let mut text_qwen = manifest(ModelFormat::SafeTensors, Some("qwen3"));
        text_qwen.metadata.tasks = vision.metadata.tasks.clone();
        let rejected = plan_runtime(
            &text_qwen,
            RequestedBackend::Vllm,
            RequestCapabilities::text_with_images(true, true),
            &available,
        );
        assert!(rejected.selected.is_none());
        assert!(rejected.candidates.iter().any(|decision| {
            decision
                .reason
                .contains("vLLM does not support image input for this architecture")
        }));

        let task_probe = plan_runtime(
            &text_qwen,
            RequestedBackend::Vllm,
            RequestCapabilities::for_task(InferenceTask::ImageUnderstanding),
            &available,
        );
        assert!(task_probe.selected.is_none());
        assert!(task_probe.candidates.iter().any(|decision| {
            decision
                .reason
                .contains("vLLM does not support image input for this architecture")
        }));
    }

    #[test]
    fn gguf_vlm_image_request_routes_to_llama_server() {
        let mut vision = manifest(ModelFormat::Gguf, Some("qwen3_vl"));
        vision.metadata.tasks = vec![
            InferenceTask::TextGeneration,
            InferenceTask::ImageUnderstanding,
        ];
        let available = [RuntimeAvailability {
            runtime_id: RuntimeId::LlamaServerCpu,
            available: true,
            reason: None,
        }];

        let selected = select_runtime(
            &vision,
            RequestedBackend::Cpu,
            RequestCapabilities::text_with_images(true, true),
            &available,
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::LlamaServerCpu);
    }

    #[test]
    fn safetensors_auto_omits_burn_and_keeps_cpu_only_as_auto_fallback() {
        let unknown = manifest(ModelFormat::SafeTensors, Some("unknown"));
        let candidates = runtime_candidate_ids(&unknown, RequestedBackend::Auto);
        if cfg!(any(windows, target_os = "linux")) {
            assert_eq!(
                candidates,
                vec![RuntimeId::CandleCuda, RuntimeId::CandleCpu]
            );
        }
        assert!(!candidates.contains(&RuntimeId::BurnCuda));
        assert!(!candidates.contains(&RuntimeId::BurnCpu));
    }

    #[test]
    fn chatglm_safetensors_auto_uses_transformers_compatibility_route() {
        let chatglm = manifest(ModelFormat::SafeTensors, Some("chatglm"));
        let candidates = runtime_candidate_ids(&chatglm, RequestedBackend::Auto);

        assert_eq!(candidates, vec![RuntimeId::TransformersCompat]);
    }

    #[test]
    fn explicit_vllm_request_has_no_candle_fallback_candidates() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Vllm);
        assert_eq!(candidates, vec![RuntimeId::VllmCuda]);
    }

    #[test]
    fn explicit_burn_request_has_no_candle_fallback_candidates() {
        let supported = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let candidates = runtime_candidate_ids(&supported, RequestedBackend::Burn);
        if cfg!(feature = "burn-cuda") && cfg!(any(windows, target_os = "linux")) {
            assert_eq!(candidates, vec![RuntimeId::BurnCuda]);
        } else if cfg!(feature = "burn-cpu") {
            assert_eq!(candidates, vec![RuntimeId::BurnCpu]);
        } else {
            assert!(candidates.is_empty());
        }

        let unsupported = manifest(ModelFormat::SafeTensors, Some("unknown"));
        assert!(runtime_candidate_ids(&unsupported, RequestedBackend::Burn).is_empty());
    }

    #[test]
    fn explicit_rocm_routes_to_rocm_candidates_only_for_compatible_formats() {
        let safetensors = manifest(ModelFormat::SafeTensors, Some("qwen3"));
        assert_eq!(
            runtime_candidate_ids(&safetensors, RequestedBackend::Rocm),
            vec![RuntimeId::VllmRocm]
        );
        let unknown_safetensors = manifest(ModelFormat::SafeTensors, Some("unknown"));
        assert_eq!(
            runtime_candidate_ids(&unknown_safetensors, RequestedBackend::Rocm),
            vec![RuntimeId::VllmRocm]
        );

        let gguf = manifest(ModelFormat::Gguf, Some("llama"));
        assert_eq!(
            runtime_candidate_ids(&gguf, RequestedBackend::Rocm),
            vec![RuntimeId::LlamaServerRocm]
        );

        let onnx = manifest(ModelFormat::Onnx, None);
        assert_eq!(
            runtime_candidate_ids(&onnx, RequestedBackend::Rocm),
            vec![RuntimeId::OnnxRuntimeRocm]
        );
    }

    #[test]
    fn gguf_auto_adds_rocm_only_when_availability_mentions_it() {
        let manifest = manifest(ModelFormat::Gguf, Some("llama"));
        let plain = plan_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &[],
        );
        assert!(
            !plain
                .candidates
                .iter()
                .any(|decision| decision.runtime_id == RuntimeId::LlamaServerRocm)
        );

        let gated = plan_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &[RuntimeAvailability {
                runtime_id: RuntimeId::LlamaServerRocm,
                available: false,
                reason: Some("ROCm probe unavailable".to_string()),
            }],
        );
        let ids = gated
            .candidates
            .iter()
            .map(|decision| decision.runtime_id)
            .collect::<Vec<_>>();
        let rocm = ids
            .iter()
            .position(|id| *id == RuntimeId::LlamaServerRocm)
            .unwrap();
        let first_lower_priority_llama = ids
            .iter()
            .position(|id| {
                matches!(
                    id,
                    RuntimeId::LlamaServerVulkan
                        | RuntimeId::LlamaServerMetal
                        | RuntimeId::LlamaServerCpu
                )
            })
            .unwrap();
        assert!(rocm <= first_lower_priority_llama);

        let preferred = select_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &[
                RuntimeAvailability {
                    runtime_id: RuntimeId::LlamaServerRocm,
                    available: true,
                    reason: None,
                },
                RuntimeAvailability {
                    runtime_id: RuntimeId::LlamaServerCuda,
                    available: true,
                    reason: None,
                },
            ],
        )
        .unwrap();
        assert_eq!(preferred.runtime_id, RuntimeId::LlamaServerRocm);
    }

    #[test]
    fn onnx_auto_routes_to_onnxruntime() {
        let manifest = manifest(ModelFormat::Onnx, None);
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Auto);
        assert!(matches!(
            candidates.first(),
            Some(RuntimeId::OnnxRuntimeCuda | RuntimeId::OnnxRuntimeCpu)
        ));
        assert!(!candidates.contains(&RuntimeId::CandleCpu));
        let selected = select_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &[RuntimeAvailability {
                runtime_id: candidates[0],
                available: true,
                reason: None,
            }],
        )
        .unwrap();
        assert_eq!(selected.runtime_id, candidates[0]);
    }

    #[test]
    fn image_request_rejects_text_only_runtime() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let available = [RuntimeAvailability {
            runtime_id: RuntimeId::CandleCuda,
            available: true,
            reason: None,
        }];
        let plan = plan_runtime(
            &manifest,
            RequestedBackend::Candle,
            RequestCapabilities::text_with_images(true, true),
            &available,
        );
        assert!(plan.selected.is_none());
        assert!(
            plan.candidates
                .iter()
                .any(|decision| decision.reason.contains("image-understanding"))
        );
    }

    #[test]
    fn gemma4_unified_image_request_prefers_mlx_vlm() {
        let mut manifest = manifest(ModelFormat::Mlx, Some("gemma4_unified"));
        manifest.metadata.tasks = vec![
            InferenceTask::TextGeneration,
            InferenceTask::ImageUnderstanding,
        ];
        let available = [
            RuntimeAvailability {
                runtime_id: RuntimeId::MlxVlm,
                available: true,
                reason: None,
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::Mlx,
                available: true,
                reason: None,
            },
        ];

        let selected = select_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text_with_images(true, true),
            &available,
        )
        .unwrap();

        assert_eq!(selected.runtime_id, RuntimeId::MlxVlm);
    }

    #[test]
    fn gemma4_unified_text_request_uses_mlx_not_mlx_vlm() {
        let manifest = manifest(ModelFormat::Mlx, Some("gemma4_unified"));
        let available = [
            RuntimeAvailability {
                runtime_id: RuntimeId::MlxVlm,
                available: true,
                reason: None,
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::Mlx,
                available: true,
                reason: None,
            },
        ];

        let plan = plan_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &available,
        );

        let selected = plan.selected.unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::Mlx);
        assert!(selected.fallback_chain.is_empty());
        assert!(selected.fallback_note().is_none());
        assert!(
            selected
                .rejection_reasons
                .iter()
                .any(|decision| decision.runtime_id == RuntimeId::MlxVlm)
        );
        assert!(
            plan.candidates
                .iter()
                .any(|decision| decision.runtime_id == RuntimeId::MlxVlm
                    && decision.reason.contains("text-only MLX uses mlx-lm"))
        );
    }

    #[test]
    fn typed_text_request_does_not_treat_mlx_vlm_as_a_preferred_route() {
        let mut manifest = manifest(ModelFormat::Mlx, Some("gemma4_unified"));
        manifest.metadata.repository_layout = RepositoryLayout::Mlx;
        manifest.metadata.tasks = vec![InferenceTask::TextGeneration];
        let selected = select_runtime_for_task(
            &manifest,
            RequestedBackend::Mlx,
            InferenceTask::TextGeneration,
            &[available(RuntimeId::MlxVlm), available(RuntimeId::Mlx)],
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::Mlx);
        assert!(selected.fallback_note().is_none());
    }

    #[test]
    fn candle_can_be_selected_as_explicit_route() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let available = [RuntimeAvailability {
            runtime_id: RuntimeId::CandleCpu,
            available: true,
            reason: None,
        }];
        let selected = select_runtime(
            &manifest,
            RequestedBackend::Candle,
            RequestCapabilities::text(true),
            &available,
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::CandleCpu);
    }

    #[test]
    fn image_task_selects_available_media_companion_cuda() {
        let manifest = media_manifest(InferenceTask::ImageGeneration);
        let available = [
            RuntimeAvailability {
                runtime_id: RuntimeId::MediaCompanionCuda,
                available: true,
                reason: None,
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::MediaCompanionCpu,
                available: true,
                reason: None,
            },
        ];

        let selected = select_runtime_for_task(
            &manifest,
            RequestedBackend::Auto,
            InferenceTask::ImageGeneration,
            &available,
        )
        .unwrap();

        assert_eq!(selected.runtime_id, RuntimeId::MediaCompanionCuda);
        assert_eq!(selected.accelerator, BackendAccelerator::Cuda);
    }

    #[test]
    fn audio_embedding_requests_advertise_embedding_capability() {
        let capabilities = RequestCapabilities::for_task(InferenceTask::AudioEmbedding);

        assert!(capabilities.embeddings);
        assert_eq!(capabilities.input_modality, Some(InputModality::Audio));
        assert_eq!(
            capabilities.output_modality,
            Some(OutputModality::Embedding)
        );
    }

    #[test]
    fn video_task_auto_prefers_fastest_available_media_accelerator() {
        let manifest = media_manifest(InferenceTask::VideoGeneration);
        let available = [
            RuntimeAvailability {
                runtime_id: RuntimeId::MediaCompanionCpu,
                available: true,
                reason: None,
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::MediaCompanionCuda,
                available: true,
                reason: None,
            },
        ];

        let selected = select_runtime_for_task(
            &manifest,
            RequestedBackend::Auto,
            InferenceTask::VideoGeneration,
            &available,
        )
        .unwrap();

        assert_eq!(selected.runtime_id, RuntimeId::MediaCompanionCuda);
        assert_eq!(selected.accelerator, BackendAccelerator::Cuda);
    }

    #[test]
    fn explicit_video_cpu_route_overrides_available_cuda() {
        let manifest = media_manifest(InferenceTask::VideoGeneration);
        let available = [
            RuntimeAvailability {
                runtime_id: RuntimeId::MediaCompanionCuda,
                available: true,
                reason: None,
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::MediaCompanionCpu,
                available: true,
                reason: None,
            },
        ];

        let selected = select_runtime_for_task(
            &manifest,
            RequestedBackend::Cpu,
            InferenceTask::VideoGeneration,
            &available,
        )
        .unwrap();

        assert_eq!(selected.runtime_id, RuntimeId::MediaCompanionCpu);
        assert_eq!(selected.accelerator, BackendAccelerator::Cpu);
    }

    #[test]
    fn typed_task_mismatch_rejects_media_runtime() {
        let manifest = media_manifest(InferenceTask::ImageGeneration);
        let plan = plan_runtime_for_task(
            &manifest,
            RequestedBackend::Auto,
            InferenceTask::VideoGeneration,
            &[RuntimeAvailability {
                runtime_id: RuntimeId::MediaCompanionCuda,
                available: true,
                reason: None,
            }],
        );

        assert!(plan.selected.is_none());
        assert!(plan.candidates.iter().any(|decision| {
            decision.runtime_id == RuntimeId::MediaCompanionCuda
                && decision.reason.contains("model does not advertise")
        }));
    }

    #[test]
    fn typed_media_candidates_filter_layout_and_accelerator() {
        let mut manifest = media_manifest(InferenceTask::ImageGeneration);
        assert_eq!(
            runtime_candidate_ids_for_task(
                &manifest,
                RequestedBackend::Cuda,
                InferenceTask::ImageGeneration,
            ),
            vec![RuntimeId::MediaCompanionCuda]
        );

        manifest.metadata.repository_layout = RepositoryLayout::TensorRtEngine;
        assert!(
            runtime_candidate_ids_for_task(
                &manifest,
                RequestedBackend::Auto,
                InferenceTask::ImageGeneration,
            )
            .is_empty()
        );
        let rejection = rejection_reason(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::for_task(InferenceTask::ImageGeneration),
            RuntimeId::MediaCompanionCuda,
            None,
        )
        .unwrap();
        assert!(rejection.contains("repository layout"));
    }

    #[test]
    fn legacy_text_auto_path_never_includes_media_companion() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("qwen2"));
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Auto);

        assert!(candidates.iter().all(|runtime_id| {
            runtime_descriptor(*runtime_id).runtime != BackendRuntime::MediaCompanion
        }));
        let plan = plan_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &[RuntimeAvailability {
                runtime_id: RuntimeId::MediaCompanionCuda,
                available: true,
                reason: None,
            }],
        );
        assert!(plan.candidates.iter().all(|decision| {
            runtime_descriptor(decision.runtime_id).runtime != BackendRuntime::MediaCompanion
        }));
    }

    fn media_manifest(task: InferenceTask) -> ModelManifest {
        let mut manifest = manifest(ModelFormat::SafeTensors, Some("fixture-media"));
        manifest.metadata.repository_layout = RepositoryLayout::Diffusers;
        manifest.metadata.tasks = vec![task];
        manifest.metadata.input_modalities = task.required_input_modalities().to_vec();
        manifest.metadata.output_modalities = vec![task.output_modality()];
        manifest
    }

    fn manifest(format: ModelFormat, architecture: Option<&str>) -> ModelManifest {
        ModelManifest {
            storage: Default::default(),
            id: "test-model".to_string(),
            source: ModelSource::LocalPath {
                path: "test".to_string(),
            },
            format,
            architecture: architecture.map(str::to_string),
            tokenizer_path: None,
            config_path: None,
            model_path: Some("files/model.bin".to_string()),
            backend: "test".to_string(),
            created_unix: 0,
            files: Vec::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        }
    }
}
