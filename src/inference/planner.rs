use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{
    capabilities::{InferenceTask, OutputModality, RepositoryLayout},
    model_store::{ModelFormat, ModelManifest},
};

use super::{
    estimate::{
        FitAssessment, WorkloadEstimate, format_memory_bytes, projected_host_peak_with_offload,
    },
    types::{EffectiveInferenceRequest, ParameterPolicy, ParameterSupportStatus},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeAccelerator {
    Cpu,
    Cuda,
    Rocm,
    Mps,
    Mlx,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRuntimeCandidate {
    pub id: String,
    pub backend: String,
    pub accelerator: RuntimeAccelerator,
    pub available: bool,
    pub availability_reason: Option<String>,
    pub supported_tasks: Vec<InferenceTask>,
    pub supported_layouts: Vec<RepositoryLayout>,
    #[serde(default)]
    pub supported_formats: Vec<ModelFormat>,
    #[serde(default)]
    pub supported_families: Vec<String>,
    #[serde(default)]
    pub supported_architectures: Vec<String>,
    #[serde(default)]
    pub parameter_support: BTreeMap<String, ParameterSupportStatus>,
    pub supports_offloading: bool,
    pub supports_compile: bool,
    pub supports_batching: bool,
    pub priority: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanCandidateStatus {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanCandidateDecision {
    pub runtime_id: String,
    pub backend: String,
    pub status: PlanCandidateStatus,
    pub score: Option<i32>,
    pub reasons: Vec<String>,
    pub degradations: Vec<ExecutionDegradation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionDegradation {
    CpuOffload,
    SequentialOffload,
    ComponentOffload,
    VaeTiling,
    TemporalWindowing,
    SlowerAttention { backend: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub task: InferenceTask,
    pub selected_runtime: Option<String>,
    pub selected_backend: Option<String>,
    pub score: Option<i32>,
    pub candidates: Vec<PlanCandidateDecision>,
    pub backend_fallback: bool,
    pub degradations: Vec<ExecutionDegradation>,
    pub model_or_quality_downgrades: Vec<String>,
}

pub fn plan_execution(
    manifest: &ModelManifest,
    request: &EffectiveInferenceRequest,
    estimate: &WorkloadEstimate,
    candidates: &[InferenceRuntimeCandidate],
) -> ExecutionPlan {
    let requested_backend = request
        .string_parameter("routing.backend")
        .filter(|value| *value != "auto");
    // The media companion treats a concrete device as the authoritative
    // execution target, so the planner must resolve the same way when both
    // values are present.
    let requested_accelerator = request
        .string_parameter("routing.device")
        .and_then(concrete_routing_target)
        .or_else(|| {
            request
                .string_parameter("routing.accelerator")
                .and_then(concrete_routing_target)
        });
    let fallback_policy = request
        .string_parameter("routing.fallback_policy")
        .unwrap_or("backend");
    let allow_degradation = fallback_policy == "degrade";
    let accelerator_memory_exceeded = estimate
        .accelerator_peak_bytes
        .zip(estimate.accelerator_memory_limit_bytes)
        .is_some_and(|(peak, limit)| peak >= limit);
    let host_memory_exceeded = estimate
        .host_peak_bytes
        .zip(estimate.host_memory_limit_bytes)
        .is_some_and(|(peak, limit)| peak >= limit);
    let memory_pressure = estimate.fit == FitAssessment::LikelyOom
        || accelerator_memory_exceeded
        || host_memory_exceeded;
    let mut decisions = Vec::new();

    for candidate in candidates {
        let mut reasons = Vec::new();
        let mut degradations = Vec::new();
        if !candidate.available {
            reasons.push(
                candidate
                    .availability_reason
                    .clone()
                    .unwrap_or_else(|| "runtime is unavailable".to_string()),
            );
        }
        if !candidate.supported_tasks.contains(&request.task) {
            reasons.push(format!("runtime does not support task {}", request.task));
        }
        if !candidate.supported_layouts.is_empty()
            && !candidate
                .supported_layouts
                .contains(&manifest.metadata.repository_layout)
        {
            reasons.push(format!(
                "runtime does not support {:?} repository layout",
                manifest.metadata.repository_layout
            ));
        }
        if !candidate.supported_formats.is_empty()
            && !candidate.supported_formats.contains(&manifest.format)
        {
            reasons.push(format!(
                "runtime does not support {:?} model format",
                manifest.format
            ));
        }
        if let Some(family) = manifest.metadata.family.as_deref()
            && !candidate.supported_families.is_empty()
            && !candidate
                .supported_families
                .iter()
                .any(|supported| supported.eq_ignore_ascii_case(family))
        {
            reasons.push(format!("runtime does not support model family '{family}'"));
        }
        if let Some(architecture) = manifest.architecture.as_deref()
            && !candidate.supported_architectures.is_empty()
            && !candidate
                .supported_architectures
                .iter()
                .any(|supported| supported.eq_ignore_ascii_case(architecture))
        {
            reasons.push(format!(
                "runtime does not support architecture '{architecture}'"
            ));
        }
        if let Some(backend) = requested_backend
            && fallback_policy == "none"
            && !candidate.backend.eq_ignore_ascii_case(backend)
        {
            reasons.push(format!("explicit backend '{backend}' was requested"));
        }
        if let Some(accelerator) = requested_accelerator
            && !runtime_accelerator_matches(candidate.accelerator, accelerator)
        {
            reasons.push(format!("accelerator '{accelerator}' was requested"));
        }
        for path in &request.explicit_parameters {
            if matches!(
                candidate
                    .parameter_support
                    .get(path)
                    .copied()
                    .unwrap_or(ParameterSupportStatus::ModelDependent),
                ParameterSupportStatus::Ignored | ParameterSupportStatus::Unsupported
            ) && request.parameter_policy == ParameterPolicy::Strict
            {
                reasons.push(format!("explicit parameter '{path}' is unsupported"));
            }
        }
        if memory_pressure {
            if host_memory_exceeded {
                reasons.push(host_memory_rejection_reason(estimate));
            } else {
                let accelerator_can_offload = matches!(
                    candidate.accelerator,
                    RuntimeAccelerator::Cuda | RuntimeAccelerator::Rocm
                );
                let selected_offload = (candidate.supports_offloading && accelerator_can_offload)
                    .then(|| selected_offload(request, allow_degradation))
                    .flatten();
                if let Some(offload) = selected_offload {
                    if let Some(reason) = offload_host_memory_rejection_reason(estimate) {
                        reasons.push(reason);
                    } else {
                        degradations.push(offload);
                        // Task-level memory reductions accompany an offload plan but
                        // cannot rescue an estimate that is still over capacity on
                        // their own: their effect is already included in `estimate`.
                        if allow_degradation
                            && request.task.output_modality() == OutputModality::Image
                            && request.bool_parameter("image.vae_tiling") == Some(true)
                        {
                            degradations.push(ExecutionDegradation::VaeTiling);
                        }
                        if allow_degradation
                            && request.task.output_modality() == OutputModality::Video
                            && (request.bool_parameter("video.temporal_vae_tiling") == Some(true)
                                || request.u64_parameter("video.window_size").is_some())
                        {
                            degradations.push(ExecutionDegradation::TemporalWindowing);
                        }
                    }
                } else {
                    reasons.push(memory_rejection_reason(estimate, candidate.accelerator));
                }
            }
        }

        let accepted = reasons.is_empty();
        let mut score = candidate.priority;
        if requested_backend.is_some_and(|backend| candidate.backend.eq_ignore_ascii_case(backend))
        {
            score += 500;
        }
        score -= i32::try_from(degradations.len()).unwrap_or(i32::MAX) * 35;
        if candidate.accelerator == RuntimeAccelerator::Cpu {
            score -= 120;
        }
        decisions.push(PlanCandidateDecision {
            runtime_id: candidate.id.clone(),
            backend: candidate.backend.clone(),
            status: if accepted {
                PlanCandidateStatus::Accepted
            } else {
                PlanCandidateStatus::Rejected
            },
            score: accepted.then_some(score),
            reasons,
            degradations,
        });
    }

    decisions.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.runtime_id.cmp(&right.runtime_id))
    });
    let selected = decisions
        .iter()
        .find(|decision| decision.status == PlanCandidateStatus::Accepted)
        .cloned();
    let selected_backend = selected.as_ref().map(|decision| decision.backend.clone());
    let backend_fallback = match (requested_backend, selected_backend.as_deref()) {
        (Some(requested), Some(selected)) => !requested.eq_ignore_ascii_case(selected),
        _ => false,
    };
    let model_or_quality_downgrades = if memory_pressure {
        vec![
            "consider a smaller model or stronger quantization".to_string(),
            match request.task.output_modality() {
                OutputModality::Image => "consider a lower image resolution".to_string(),
                OutputModality::Video => {
                    "consider fewer frames or a lower video resolution".to_string()
                }
                OutputModality::Audio => "consider a shorter duration".to_string(),
                OutputModality::Text | OutputModality::Embedding => {
                    "consider a shorter context".to_string()
                }
            },
        ]
    } else {
        Vec::new()
    };

    ExecutionPlan {
        task: request.task,
        selected_runtime: selected
            .as_ref()
            .map(|decision| decision.runtime_id.clone()),
        selected_backend,
        score: selected.as_ref().and_then(|decision| decision.score),
        candidates: decisions,
        backend_fallback,
        degradations: selected
            .map(|decision| decision.degradations)
            .unwrap_or_default(),
        model_or_quality_downgrades,
    }
}

fn selected_offload(
    request: &EffectiveInferenceRequest,
    allow_degradation: bool,
) -> Option<ExecutionDegradation> {
    let options = [
        (
            "routing.allow_cpu_offload",
            ExecutionDegradation::CpuOffload,
        ),
        (
            "routing.allow_sequential_offload",
            ExecutionDegradation::SequentialOffload,
        ),
        (
            "routing.allow_component_offload",
            ExecutionDegradation::ComponentOffload,
        ),
    ];

    // An explicit permission is an instruction, not a general fallback. It
    // therefore applies even under the default `backend` fallback policy.
    if let Some((_, degradation)) = options.iter().find(|(path, _)| {
        request.explicit_parameters.contains(*path) && request.bool_parameter(path) == Some(true)
    }) {
        return Some(degradation.clone());
    }

    if !allow_degradation {
        return None;
    }

    // Component and sequential offload use CPU-offload hooks too. An explicit
    // CPU-offload denial suppresses inherited alternatives; a separately
    // explicit alternative was handled above and remains authoritative.
    if request
        .explicit_parameters
        .contains("routing.allow_cpu_offload")
        && request.bool_parameter("routing.allow_cpu_offload") == Some(false)
    {
        return None;
    }

    options
        .into_iter()
        .find(|(path, _)| request.bool_parameter(path) == Some(true))
        .map(|(_, degradation)| degradation)
}

fn memory_rejection_reason(estimate: &WorkloadEstimate, accelerator: RuntimeAccelerator) -> String {
    if accelerator == RuntimeAccelerator::Cuda
        && let Some((peak, limit)) = estimate
            .accelerator_peak_bytes
            .zip(estimate.accelerator_memory_limit_bytes)
        && peak >= limit
    {
        format!(
            "estimated accelerator peak of {} reaches or exceeds detected accelerator memory limit of {} and no permitted offload could be selected",
            format_memory_bytes(peak),
            format_memory_bytes(limit)
        )
    } else {
        "workload is likely out of memory and no permitted offload could be selected".to_string()
    }
}

fn host_memory_rejection_reason(estimate: &WorkloadEstimate) -> String {
    match estimate
        .host_peak_bytes
        .zip(estimate.host_memory_limit_bytes)
    {
        Some((peak, limit)) => format!(
            "estimated host peak of {} reaches or exceeds detected host memory limit of {}; accelerator offload cannot resolve host-memory pressure",
            format_memory_bytes(peak),
            format_memory_bytes(limit)
        ),
        None => "workload is likely out of host memory; accelerator offload cannot resolve host-memory pressure".to_string(),
    }
}

fn offload_host_memory_rejection_reason(estimate: &WorkloadEstimate) -> Option<String> {
    let limit = estimate.host_memory_limit_bytes?;
    let required = projected_host_peak_with_offload(estimate)?;
    (required >= limit).then(|| {
        format!(
            "projected host memory required with offload is {}, which reaches or exceeds detected host memory limit of {}; offload was not selected",
            format_memory_bytes(required),
            format_memory_bytes(limit)
        )
    })
}

fn runtime_accelerator_matches(accelerator: RuntimeAccelerator, requested: &str) -> bool {
    matches!(
        (accelerator, requested.to_ascii_lowercase().as_str()),
        (RuntimeAccelerator::Cpu, "cpu")
            | (RuntimeAccelerator::Cuda, "cuda")
            | (RuntimeAccelerator::Rocm, "rocm" | "hip")
            | (RuntimeAccelerator::Mps, "mps" | "metal")
            | (RuntimeAccelerator::Mlx, "mlx" | "metal")
    )
}

fn concrete_routing_target(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty() && !value.eq_ignore_ascii_case("auto")).then_some(value)
}
