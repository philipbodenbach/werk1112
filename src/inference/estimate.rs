use serde::{Deserialize, Serialize};

use crate::{
    capabilities::{InferenceTask, OutputModality},
    model_store::ModelManifest,
};

use super::types::{EffectiveInferenceRequest, ParameterSource, ParameterValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateConfidence {
    Exact,
    BackendMeasured,
    ArchitectureModel,
    Heuristic,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitAssessment {
    Fits,
    Tight,
    LikelyOom,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTopology {
    Discrete,
    Unified,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct HostResources {
    pub host_memory_bytes: Option<u64>,
    pub accelerator_memory_bytes: Option<u64>,
    pub accelerator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_topology: Option<MemoryTopology>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkloadEstimate {
    pub task: InferenceTask,
    pub download_size_bytes: Option<u64>,
    pub weight_payload_bytes: Option<u64>,
    pub accelerator_peak_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accelerator_memory_limit_bytes: Option<u64>,
    pub host_peak_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_memory_limit_bytes: Option<u64>,
    pub output_size_bytes: Option<u64>,
    pub fit: FitAssessment,
    pub confidence: EstimateConfidence,
    pub assumptions: Vec<String>,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

pub fn estimate_workload(
    manifest: &ModelManifest,
    request: &EffectiveInferenceRequest,
    resources: &HostResources,
) -> WorkloadEstimate {
    let download_size = manifest
        .files
        .iter()
        .map(|file| file.size)
        .fold(0_u64, u64::saturating_add);
    let weight_payload = manifest
        .files
        .iter()
        .filter(|file| is_weight_path(&file.path))
        .map(|file| file.size)
        .fold(0_u64, u64::saturating_add);
    let weights = if weight_payload == 0 {
        download_size
    } else {
        weight_payload
    };
    let batch = request
        .u64_parameter(&format!(
            "{}.batch_size",
            request.task.parameter_namespace()
        ))
        .unwrap_or(1)
        .max(1);
    let mut assumptions = vec![
        "weights are estimated from stored artifact sizes".to_string(),
        "runtime allocator fragmentation is approximated".to_string(),
    ];
    let (mut activation, output_size) = match request.task.output_modality() {
        OutputModality::Image => {
            let typed = request.image_generation_options();
            let width = typed
                .as_ref()
                .map(|options| u64::from(options.width))
                .unwrap_or(1024);
            let height = typed
                .as_ref()
                .map(|options| u64::from(options.height))
                .unwrap_or(1024);
            let count = request.u64_parameter("image.num_images").unwrap_or(1);
            let pixels = width
                .saturating_mul(height)
                .saturating_mul(batch)
                .saturating_mul(count);
            let mut activation = pixels.saturating_mul(96);
            if request.bool_parameter("image.vae_tiling") == Some(true) {
                activation = activation.saturating_mul(3) / 5;
                assumptions.push("VAE tiling reduces estimated activation peak".to_string());
            }
            if request.bool_parameter("image.vae_slicing") == Some(true) {
                activation = activation.saturating_mul(4) / 5;
                assumptions.push("VAE slicing reduces estimated activation peak".to_string());
            }
            assumptions.push(format!(
                "{} denoising step(s); steps affect runtime more than peak memory",
                typed.as_ref().map(|options| options.steps).unwrap_or(28)
            ));
            (activation, pixels.saturating_mul(4))
        }
        OutputModality::Video => {
            let typed = request.video_generation_options();
            let width = typed
                .as_ref()
                .map(|options| u64::from(options.width))
                .unwrap_or(832);
            let height = typed
                .as_ref()
                .map(|options| u64::from(options.height))
                .unwrap_or(480);
            let frames = typed
                .as_ref()
                .map(|options| u64::from(options.frames))
                .unwrap_or(81);
            let count = request.u64_parameter("video.num_videos").unwrap_or(1);
            let pixels = width
                .saturating_mul(height)
                .saturating_mul(frames)
                .saturating_mul(batch)
                .saturating_mul(count);
            let mut activation = pixels.saturating_mul(36);
            if request.bool_parameter("video.temporal_vae_tiling") == Some(true)
                || request.u64_parameter("video.window_size").is_some()
            {
                activation = activation.saturating_mul(11) / 20;
                assumptions.push(
                    "temporal tiling/windowing reduces estimated activation peak".to_string(),
                );
            }
            assumptions.push(format!(
                "{} denoising step(s); steps affect runtime more than peak memory",
                typed.as_ref().map(|options| options.steps).unwrap_or(30)
            ));
            let output_size = request
                .u64_parameter("video.bitrate")
                .map(|bitrate| {
                    let fps = typed.as_ref().map(|options| options.fps).unwrap_or(24.0);
                    let duration = frames as f64 / fps.max(0.1);
                    saturating_f64_to_u64(duration * bitrate as f64 / 8.0).saturating_mul(count)
                })
                .unwrap_or_else(|| pixels.saturating_mul(3) / 20);
            (activation, output_size)
        }
        OutputModality::Audio => {
            let namespace = request.task.parameter_namespace();
            let typed = request.audio_generation_options();
            let duration = request
                .f64_parameter(&format!("{namespace}.duration"))
                .or_else(|| typed.as_ref().map(|options| options.duration))
                .unwrap_or(30.0)
                .max(0.0);
            let rate = request
                .u64_parameter(&format!("{namespace}.sample_rate"))
                .or_else(|| typed.as_ref().map(|options| u64::from(options.sample_rate)))
                .unwrap_or(44_100);
            let channels = request
                .u64_parameter(&format!("{namespace}.channels"))
                .or_else(|| typed.as_ref().map(|options| u64::from(options.channels)))
                .unwrap_or(2);
            let variations = request.u64_parameter("audio.variations").unwrap_or(1);
            let stems = request
                .parameter("audio.stems")
                .and_then(|value| match value {
                    ParameterValue::List(items) => Some(items.len() as u64),
                    _ => None,
                })
                .unwrap_or(1)
                .max(1);
            let samples = saturating_f64_to_u64(duration * rate as f64)
                .saturating_mul(channels)
                .saturating_mul(variations);
            let bit_depth = request
                .u64_parameter(&format!("{namespace}.bit_depth"))
                .unwrap_or(16)
                .max(1);
            let output_size = request
                .u64_parameter(&format!("{namespace}.bitrate"))
                .map(|bitrate| {
                    saturating_f64_to_u64(duration * bitrate as f64 / 8.0)
                        .saturating_mul(variations)
                })
                .unwrap_or_else(|| samples.saturating_mul(bit_depth).div_ceil(8))
                .saturating_mul(stems);
            (
                samples.saturating_mul(24).saturating_mul(stems),
                output_size,
            )
        }
        OutputModality::Text | OutputModality::Embedding => (weights / 10, 4_u64 * 1024 * 1024),
    };
    let precision = request
        .string_parameter("routing.precision")
        .unwrap_or("auto")
        .to_ascii_lowercase();
    let precision_scale = match precision.as_str() {
        "fp16" | "float16" | "half" | "bf16" | "bfloat16" => 0.6,
        "fp8" | "float8" | "int8" => 0.4,
        "int4" | "nf4" => 0.3,
        _ => 1.0,
    };
    if precision != "auto" {
        assumptions.push(format!(
            "activation estimate scaled for requested precision '{precision}'"
        ));
    }
    let attention = request
        .string_parameter("routing.attention_backend")
        .unwrap_or("auto")
        .to_ascii_lowercase();
    let attention_scale = match attention.as_str() {
        "flash" | "flash_attention" | "flash-attention" | "xformers" => 0.78,
        "sliced" => 0.65,
        "sdpa" => 0.88,
        _ => 1.0,
    };
    if attention != "auto" {
        assumptions.push(format!(
            "activation estimate scaled for attention backend '{attention}'"
        ));
    }
    activation = saturating_f64_to_u64(activation as f64 * precision_scale * attention_scale);
    let component_overhead = u64::try_from(manifest.metadata.components.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(32 * 1024 * 1024);
    let runtime_overhead = weights / 8 + 256 * 1024 * 1024 + component_overhead;
    assumptions.push(format!(
        "{} detected model component(s) contribute runtime overhead",
        manifest.metadata.components.len()
    ));
    let accelerator_peak = weights
        .saturating_add(activation)
        .saturating_add(runtime_overhead);
    // The public allow_* values are routing permissions. They affect the
    // execution estimate only after the planner selected an offload
    // degradation and recorded it as a backend adjustment.
    let offload_selected = [
        "routing.allow_cpu_offload",
        "routing.allow_sequential_offload",
        "routing.allow_component_offload",
    ]
    .iter()
    .any(|path| {
        request.parameters.get(*path).is_some_and(|parameter| {
            parameter.source == ParameterSource::BackendAdjustment
                && parameter.value.as_bool() == Some(true)
        })
    });
    let host_peak = if offload_selected {
        weights
            .saturating_add(activation / 3)
            .saturating_add(512 * 1024 * 1024)
    } else {
        weights / 4 + 512 * 1024 * 1024
    };

    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();
    if resources.memory_topology == Some(MemoryTopology::Unified) {
        assumptions.push(
            "host MemAvailable is the conservative available capacity of the shared CPU/GPU unified-memory pool"
                .to_string(),
        );
        let uses_accelerator = resources.accelerator.as_deref().is_some_and(|accelerator| {
            matches!(
                accelerator.to_ascii_lowercase().as_str(),
                "cuda" | "rocm" | "hip" | "mps" | "metal" | "mlx"
            )
        });
        if uses_accelerator {
            if resources.accelerator_memory_bytes.is_none() {
                warnings.push(
                    "DGX Spark unified memory has no independently reported VRAM capacity; accelerator fit remains unknown and CPU offload does not create a separate memory pool"
                        .to_string(),
                );
            } else {
                warnings.push(
                    "the configured accelerator-memory limit is part of DGX Spark's shared unified-memory pool, not additional discrete VRAM"
                        .to_string(),
                );
            }
        }
    }
    let fit = classify_workload_fit(Some(accelerator_peak), Some(host_peak), resources);
    if let Some(limit) = resources.accelerator_memory_bytes {
        assumptions.push(format!(
            "detected accelerator memory limit is {}",
            format_memory_bytes(limit)
        ));
        if accelerator_peak >= limit {
            warnings.push(format!(
                "estimated accelerator peak of {} reaches or exceeds detected accelerator memory limit of {}",
                format_memory_bytes(accelerator_peak),
                format_memory_bytes(limit)
            ));
        }
    }
    if let Some(limit) = resources.host_memory_bytes {
        assumptions.push(format!(
            "detected host memory limit is {}",
            format_memory_bytes(limit)
        ));
        if host_peak >= limit {
            warnings.push(format!(
                "estimated host peak of {} reaches or exceeds detected host memory limit of {}",
                format_memory_bytes(host_peak),
                format_memory_bytes(limit)
            ));
        }
    }
    match fit {
        FitAssessment::Tight => {
            warnings.push("estimated peak is close to available memory".to_string());
            recommendations
                .push("enable component offload or task-specific tiling/windowing".to_string());
        }
        FitAssessment::LikelyOom => {
            warnings.push("estimated workload does not fit reported resources".to_string());
            recommendations.push(
                "try another runtime or enable CPU/sequential offload and tiling".to_string(),
            );
            recommendations.push(
                "resolution, frames, duration, or model downgrades are recommendations only"
                    .to_string(),
            );
        }
        FitAssessment::Fits | FitAssessment::Unknown => {}
    }

    WorkloadEstimate {
        task: request.task,
        download_size_bytes: Some(download_size),
        weight_payload_bytes: Some(weights),
        accelerator_peak_bytes: Some(accelerator_peak),
        accelerator_memory_limit_bytes: resources.accelerator_memory_bytes,
        host_peak_bytes: Some(host_peak),
        host_memory_limit_bytes: resources.host_memory_bytes,
        output_size_bytes: Some(output_size),
        fit,
        confidence: EstimateConfidence::Heuristic,
        assumptions,
        warnings,
        recommendations,
    }
}

/// Conservatively projects host memory after moving accelerator state to the
/// CPU. The current host working set remains resident, all model weights may
/// need a host copy, and a third of the remaining accelerator working set is
/// reserved for staging and activations.
pub(crate) fn projected_host_peak_with_offload(estimate: &WorkloadEstimate) -> Option<u64> {
    let current_host_peak = estimate.host_peak_bytes?;
    let weights = estimate
        .weight_payload_bytes
        .or(estimate.download_size_bytes)?;
    let accelerator_working_set = estimate
        .accelerator_peak_bytes
        .unwrap_or(weights)
        .saturating_sub(weights);

    Some(
        current_host_peak
            .saturating_add(weights)
            .saturating_add(accelerator_working_set / 3),
    )
}

pub(crate) fn format_memory_bytes(bytes: u64) -> String {
    const GIBIBYTE: u64 = 1024 * 1024 * 1024;
    const MEBIBYTE: u64 = 1024 * 1024;
    const KIBIBYTE: u64 = 1024;

    if bytes >= GIBIBYTE {
        format!("{:.2} GiB", bytes as f64 / GIBIBYTE as f64)
    } else if bytes >= MEBIBYTE {
        format!("{:.2} MiB", bytes as f64 / MEBIBYTE as f64)
    } else if bytes >= KIBIBYTE {
        format!("{:.2} KiB", bytes as f64 / KIBIBYTE as f64)
    } else {
        format!("{bytes} B")
    }
}

fn saturating_f64_to_u64(value: f64) -> u64 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value.round() as u64
    }
}

fn is_weight_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    [
        ".safetensors",
        ".gguf",
        ".bin",
        ".pt",
        ".pth",
        ".onnx",
        ".engine",
        ".plan",
        ".npz",
    ]
    .iter()
    .any(|extension| lower.ends_with(extension))
}

pub(crate) fn classify_workload_fit(
    accelerator_peak: Option<u64>,
    host_peak: Option<u64>,
    resources: &HostResources,
) -> FitAssessment {
    let accelerator_oom = accelerator_peak
        .zip(resources.accelerator_memory_bytes)
        .is_some_and(|(peak, limit)| peak >= limit);
    let host_oom = host_peak
        .zip(resources.host_memory_bytes)
        .is_some_and(|(peak, limit)| peak >= limit);
    if accelerator_oom || host_oom {
        return FitAssessment::LikelyOom;
    }
    let accelerator_ratio = accelerator_peak
        .zip(resources.accelerator_memory_bytes)
        .map(|(peak, limit)| peak as f64 / limit.max(1) as f64);
    let host_ratio = host_peak
        .zip(resources.host_memory_bytes)
        .map(|(peak, limit)| peak as f64 / limit.max(1) as f64);
    let accelerator_capacity_unknown = accelerator_peak.is_some()
        && resources.accelerator_memory_bytes.is_none()
        && resources.accelerator.as_deref().is_some_and(|accelerator| {
            matches!(
                accelerator.to_ascii_lowercase().as_str(),
                "cuda" | "rocm" | "hip" | "mps" | "metal" | "mlx"
            )
        });
    if accelerator_capacity_unknown {
        return FitAssessment::Unknown;
    }
    let Some(max_ratio) = [accelerator_ratio, host_ratio]
        .into_iter()
        .flatten()
        .reduce(f64::max)
    else {
        return FitAssessment::Unknown;
    };
    if max_ratio >= 0.85 {
        FitAssessment::Tight
    } else {
        FitAssessment::Fits
    }
}
