use serde::Serialize;
use serde_json::Value;
use std::io::{self, Write};

use super::{format_bytes, format_duration, trim_float, yes_no};
use crate::{
    capabilities::{InferenceTask, OutputModality},
    inference::{
        EffectiveInferenceRequest, ExecutionDegradation, ExecutionPlan, InferenceInputSource,
        ParameterSource, ParameterValue, PlanCandidateStatus, WorkloadEstimate,
    },
    inference_service::{InferenceResult, RuntimeAttemptTiming},
};

#[derive(Debug, Clone, Copy)]
pub(super) struct MediaCliTimings {
    pub(super) total_seconds: f64,
    pub(super) service_seconds: f64,
    pub(super) publication_seconds: f64,
}

pub(super) fn write_media_failed_attempts<W: Write>(
    writer: &mut W,
    attempts: &[RuntimeAttemptTiming],
    service_seconds: f64,
) -> io::Result<()> {
    writeln!(writer)?;
    writeln!(writer, "media execution failed:")?;
    write_media_stat(
        writer,
        "service duration:",
        &format_duration(service_seconds),
    )?;
    writeln!(writer, "attempt details:")?;
    for attempt in attempts {
        writeln!(
            writer,
            "  {} outcome={} duration={}",
            attempt.runtime,
            serialized_enum_label(&attempt.outcome),
            format_duration(attempt.duration_seconds)
        )?;
    }
    Ok(())
}

pub(super) fn write_media_verbose_stats<W: Write>(
    writer: &mut W,
    result: &InferenceResult,
    timings: MediaCliTimings,
) -> io::Result<()> {
    let backend = result
        .backend_metadata
        .get("backend")
        .unwrap_or(&Value::Null);
    let inference_seconds = media_phase_seconds(backend, "inference_seconds");
    let output_bytes = result
        .outputs
        .iter()
        .map(|output| output.size_bytes)
        .fold(0_u64, u64::saturating_add);

    writeln!(writer)?;
    writeln!(writer, "media stats:")?;
    write_media_stat(writer, "runtime:", &result.runtime)?;
    if let Some(selected_backend) = result.plan.selected_backend.as_deref() {
        write_media_stat(writer, "backend:", selected_backend)?;
    }
    for (label, key) in [
        ("adapter:", "runtime"),
        ("device:", "device"),
        ("precision:", "dtype"),
        ("pipeline:", "pipeline_task"),
    ] {
        if let Some(value) = backend.get(key).and_then(Value::as_str) {
            write_media_stat(writer, label, value)?;
        }
    }
    write_media_stat(
        writer,
        "total duration:",
        &format_duration(timings.total_seconds),
    )?;
    write_media_stat(
        writer,
        "service duration:",
        &format_duration(timings.service_seconds),
    )?;
    for (label, seconds) in [
        ("request resolution:", result.timings.resolve_seconds),
        ("workload estimate:", result.timings.estimate_seconds),
        ("routing plan:", result.timings.planning_seconds),
        ("output setup:", result.timings.output_setup_seconds),
        ("runtime execution:", result.timings.execution_seconds),
        ("result finalization:", result.timings.finalization_seconds),
    ] {
        if seconds.is_finite() && seconds > 0.0 {
            write_media_stat(writer, label, &format_duration(seconds))?;
        }
    }
    if let Some(seconds) = media_metadata_seconds(&result.backend_metadata, "elapsed_seconds") {
        write_media_stat(writer, "companion duration:", &format_duration(seconds))?;
    }
    for (label, key) in [
        ("model load/setup:", "model_load_seconds"),
        ("inference:", "inference_seconds"),
        ("encoding:", "encoding_seconds"),
    ] {
        if let Some(seconds) = media_phase_seconds(backend, key) {
            write_media_stat(writer, label, &format_duration(seconds))?;
        }
    }
    write_media_stat(
        writer,
        "publication:",
        &format_duration(timings.publication_seconds),
    )?;
    write_media_stat(writer, "outputs:", &result.outputs.len().to_string())?;
    write_media_stat(writer, "output size:", &format_bytes(output_bytes))?;
    if let Some(formats) = media_output_formats(result) {
        write_media_stat(writer, "output format:", &formats)?;
    }

    match result.task.output_modality() {
        OutputModality::Image => {
            write_media_image_stats(writer, result, inference_seconds)?;
        }
        OutputModality::Video => {
            write_media_video_stats(writer, result, inference_seconds)?;
        }
        OutputModality::Audio => {
            write_media_audio_stats(writer, result, inference_seconds)?;
        }
        OutputModality::Text if result.task == InferenceTask::SpeechToText => {
            write_media_transcription_stats(writer, result)?;
        }
        OutputModality::Text | OutputModality::Embedding => {}
    }

    write_media_stat(
        writer,
        "workload fit:",
        &format!(
            "{} ({})",
            serialized_enum_label(&result.estimate.fit),
            serialized_enum_label(&result.estimate.confidence)
        ),
    )?;
    if let Some(bytes) = result.estimate.accelerator_peak_bytes {
        write_media_stat(writer, "estimated accel peak:", &format_bytes(bytes))?;
    }
    if let Some(bytes) = result.estimate.host_peak_bytes {
        write_media_stat(writer, "estimated host peak:", &format_bytes(bytes))?;
    }
    write_media_stat(
        writer,
        "backend fallback:",
        yes_no(result.plan.backend_fallback),
    )?;
    if !result.timings.runtime_attempts.is_empty() {
        writeln!(writer, "runtime attempts:")?;
        for attempt in &result.timings.runtime_attempts {
            writeln!(
                writer,
                "  {} outcome={} duration={}",
                attempt.runtime,
                serialized_enum_label(&attempt.outcome),
                format_duration(attempt.duration_seconds)
            )?;
        }
    }
    if !result.plan.degradations.is_empty() {
        write_media_stat(
            writer,
            "degradations:",
            &result
                .plan
                .degradations
                .iter()
                .map(format_media_degradation)
                .collect::<Vec<_>>()
                .join(", "),
        )?;
    }
    Ok(())
}

fn write_media_image_stats<W: Write>(
    writer: &mut W,
    result: &InferenceResult,
    inference_seconds: Option<f64>,
) -> io::Result<()> {
    if let Some(resolution) = media_output_resolutions(result) {
        write_media_stat(writer, "resolution:", &resolution)?;
    }
    if let Some(steps) = media_effective_u64(result, "steps") {
        write_media_stat(writer, "steps:", &steps.to_string())?;
    }
    if let Some(seed) = media_seed(result) {
        write_media_stat(writer, "seed:", &seed.to_string())?;
    }
    let pixels = result
        .outputs
        .iter()
        .filter_map(|output| Some(u64::from(output.width?) * u64::from(output.height?)))
        .fold(0_u64, u64::saturating_add);
    if let Some(seconds) = positive_finite(inference_seconds)
        && !result.outputs.is_empty()
    {
        write_media_stat(
            writer,
            "seconds / image:",
            &format_duration(seconds / result.outputs.len() as f64),
        )?;
        if pixels > 0 {
            write_media_stat(
                writer,
                "generation rate:",
                &format!("{:.2} MP/s", pixels as f64 / 1_000_000.0 / seconds),
            )?;
        }
    }
    Ok(())
}

fn write_media_video_stats<W: Write>(
    writer: &mut W,
    result: &InferenceResult,
    inference_seconds: Option<f64>,
) -> io::Result<()> {
    if let Some(resolution) = media_output_resolutions(result) {
        write_media_stat(writer, "resolution:", &resolution)?;
    }
    let frames =
        media_output_u64_sum(result, "frames").or_else(|| media_effective_u64(result, "frames"));
    if let Some(frames) = frames {
        write_media_stat(writer, "generated frames:", &frames.to_string())?;
    }
    if let Some(fps) =
        media_output_f64_first(result, "fps").or_else(|| media_effective_f64(result, "fps"))
    {
        write_media_stat(writer, "playback rate:", &format!("{fps:.2} fps"))?;
    }
    if let Some(duration) = media_output_duration_summary(result) {
        write_media_stat(writer, "output duration:", &duration)?;
    }
    if let Some(steps) = media_effective_u64(result, "steps") {
        write_media_stat(writer, "steps:", &steps.to_string())?;
    }
    if let Some(seed) = media_seed(result) {
        write_media_stat(writer, "seed:", &seed.to_string())?;
    }
    if let (Some(frames), Some(seconds)) = (frames, positive_finite(inference_seconds)) {
        write_media_stat(
            writer,
            "generation rate:",
            &format!("{:.2} frames/s", frames as f64 / seconds),
        )?;
    }
    Ok(())
}

fn write_media_audio_stats<W: Write>(
    writer: &mut W,
    result: &InferenceResult,
    inference_seconds: Option<f64>,
) -> io::Result<()> {
    if let Some(duration_summary) = media_output_duration_summary(result) {
        write_media_stat(writer, "output duration:", &duration_summary)?;
    }
    if let Some(duration) = media_output_duration(result)
        && media_audio_real_time_factor_is_meaningful(result.task)
        && let Some(inference_seconds) = positive_finite(inference_seconds)
        && duration > 0.0
    {
        let label = if media_output_duration_count(result) > 1 {
            "aggregate RTF:"
        } else {
            "real-time factor:"
        };
        write_media_stat(
            writer,
            label,
            &format!("{:.3}", inference_seconds / duration),
        )?;
    }
    if let Some(sample_rate) = media_output_u64_first(result, "sample_rate")
        .or_else(|| media_effective_u64(result, "sample_rate"))
    {
        write_media_stat(writer, "sample rate:", &format!("{sample_rate} Hz"))?;
    }
    if let Some(channels) = media_output_u64_first(result, "channels")
        .or_else(|| media_effective_u64(result, "channels"))
    {
        write_media_stat(writer, "channels:", &channels.to_string())?;
    }
    if let Some(seed) = media_seed(result) {
        write_media_stat(writer, "seed:", &seed.to_string())?;
    }
    Ok(())
}

fn write_media_transcription_stats<W: Write>(
    writer: &mut W,
    result: &InferenceResult,
) -> io::Result<()> {
    if let Some(format) = result
        .effective_request
        .string_parameter("stt.output_format")
    {
        write_media_stat(writer, "transcript format:", format)?;
    }
    Ok(())
}

pub(super) fn write_media_routing_debug<W: Write>(
    writer: &mut W,
    request: &EffectiveInferenceRequest,
    estimate: &WorkloadEstimate,
    plan: &ExecutionPlan,
) -> io::Result<()> {
    writeln!(writer, "media routing debug:")?;
    write_media_stat(writer, "task:", &request.task.to_string())?;
    write_media_stat(writer, "model:", &request.model)?;
    write_media_stat(
        writer,
        "parameter policy:",
        &request.parameter_policy.to_string(),
    )?;
    write_media_stat(
        writer,
        "prompt:",
        &media_text_presence(request.prompt.as_deref()),
    )?;
    write_media_stat(
        writer,
        "negative prompt:",
        &media_text_presence(request.negative_prompt.as_deref()),
    )?;
    if request.inputs.is_empty() {
        write_media_stat(writer, "inputs:", "none")?;
    } else {
        writeln!(writer, "{:<24}", "inputs:")?;
        for input in &request.inputs {
            writeln!(
                writer,
                "  {} role={} source={}",
                input.modality,
                input.role,
                media_input_source_kind(&input.source)
            )?;
        }
    }
    for (label, path) in [
        ("routing backend:", "routing.backend"),
        ("routing accelerator:", "routing.accelerator"),
        ("routing device:", "routing.device"),
        ("routing precision:", "routing.precision"),
        ("fallback policy:", "routing.fallback_policy"),
    ] {
        if let Some(value) = request.string_parameter(path) {
            let source = request
                .parameters
                .get(path)
                .map(|parameter| serialized_enum_label(&parameter.source))
                .unwrap_or_else(|| "unknown".to_string());
            write_media_stat(writer, label, &format!("{value} ({source})"))?;
        }
    }
    write_media_stat(
        writer,
        "workload fit:",
        &format!(
            "{} ({})",
            serialized_enum_label(&estimate.fit),
            serialized_enum_label(&estimate.confidence)
        ),
    )?;
    if let Some(bytes) = estimate.accelerator_peak_bytes {
        write_media_stat(writer, "estimated accel peak:", &format_bytes(bytes))?;
    }
    if let Some(bytes) = estimate.host_peak_bytes {
        write_media_stat(writer, "estimated host peak:", &format_bytes(bytes))?;
    }

    writeln!(writer, "candidate runtimes:")?;
    for candidate in &plan.candidates {
        let status = match candidate.status {
            PlanCandidateStatus::Accepted => "accepted",
            PlanCandidateStatus::Rejected => "rejected",
        };
        let score = candidate
            .score
            .map(|score| score.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        writeln!(
            writer,
            "  {} backend={} status={} score={}",
            candidate.runtime_id, candidate.backend, status, score
        )?;
        for reason in &candidate.reasons {
            writeln!(writer, "    reason: {}", media_sanitize_debug_text(reason))?;
        }
        for degradation in &candidate.degradations {
            writeln!(
                writer,
                "    degradation: {}",
                format_media_degradation(degradation)
            )?;
        }
    }
    write_media_stat(
        writer,
        "selected runtime:",
        plan.selected_runtime.as_deref().unwrap_or("none"),
    )?;
    write_media_stat(
        writer,
        "selected backend:",
        plan.selected_backend.as_deref().unwrap_or("none"),
    )?;
    write_media_stat(writer, "selected score:", &media_optional_i32(plan.score))?;

    writeln!(writer, "resolved parameters:")?;
    for (path, parameter) in &request.parameters {
        let explicit = if request.explicit_parameters.contains(path) {
            " explicit"
        } else {
            ""
        };
        writeln!(
            writer,
            "  {path} = {} ({}{explicit})",
            media_debug_parameter_value(path, &parameter.value),
            serialized_enum_label(&parameter.source)
        )?;
    }
    Ok(())
}

pub(super) fn write_media_backend_debug<W: Write>(
    writer: &mut W,
    result: &InferenceResult,
) -> io::Result<()> {
    let backend = result
        .backend_metadata
        .get("backend")
        .unwrap_or(&Value::Null);
    writeln!(writer)?;
    writeln!(writer, "media execution debug:")?;
    write_media_stat(writer, "actual runtime:", &result.runtime)?;
    write_media_stat(
        writer,
        "backend fallback:",
        yes_no(result.plan.backend_fallback),
    )?;
    if !result.timings.runtime_attempts.is_empty() {
        writeln!(writer, "attempted runtimes:")?;
        for attempt in &result.timings.runtime_attempts {
            writeln!(
                writer,
                "  {} -> {}",
                attempt.runtime,
                serialized_enum_label(&attempt.outcome)
            )?;
        }
    }
    for (label, key) in [
        ("adapter:", "runtime"),
        ("pipeline:", "pipeline_task"),
        ("device:", "device"),
        ("precision:", "dtype"),
    ] {
        if let Some(value) = backend.get(key).and_then(Value::as_str) {
            write_media_stat(writer, label, value)?;
        }
    }
    if let Some(seed) = backend.get("seed").and_then(json_value_u64) {
        write_media_stat(writer, "backend seed:", &seed.to_string())?;
    }
    if let Some(parameters) = backend
        .get("translated_parameters")
        .and_then(Value::as_array)
    {
        let names = parameters
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        if !names.is_empty() {
            write_media_stat(writer, "translated params:", &names.join(", "))?;
        }
    }
    if let Some(support) = backend.get("parameter_support") {
        if let Some(policy) = support.get("policy").and_then(Value::as_str) {
            write_media_stat(writer, "backend policy:", policy)?;
        }
        for (label, key) in [
            ("explicit params:", "explicit_parameters"),
            ("unsupported params:", "unsupported_explicit_parameters"),
        ] {
            if let Some(values) = support.get(key).and_then(Value::as_array) {
                let names = values.iter().filter_map(Value::as_str).collect::<Vec<_>>();
                if !names.is_empty() {
                    write_media_stat(writer, label, &names.join(", "))?;
                }
            }
        }
    }
    let mut backend_adjustments = result
        .effective_request
        .parameters
        .iter()
        .filter(|(_, parameter)| parameter.source == ParameterSource::BackendAdjustment)
        .peekable();
    if backend_adjustments.peek().is_some() {
        writeln!(writer, "backend adjustments:")?;
        for (path, parameter) in backend_adjustments {
            writeln!(
                writer,
                "  {path} = {}",
                media_debug_parameter_value(path, &parameter.value)
            )?;
        }
    }
    if !result.plan.degradations.is_empty() {
        write_media_stat(
            writer,
            "degradations:",
            &result
                .plan
                .degradations
                .iter()
                .map(format_media_degradation)
                .collect::<Vec<_>>()
                .join(", "),
        )?;
    }
    if !result.plan.model_or_quality_downgrades.is_empty() {
        write_media_stat(
            writer,
            "quality downgrades:",
            &result.plan.model_or_quality_downgrades.join(", "),
        )?;
    }
    Ok(())
}

fn write_media_stat<W: Write>(writer: &mut W, label: &str, value: &str) -> io::Result<()> {
    writeln!(writer, "{label:<24}{value}")
}

fn media_text_presence(text: Option<&str>) -> String {
    text.map(|text| format!("present ({} character(s), redacted)", text.chars().count()))
        .unwrap_or_else(|| "none".to_string())
}

fn media_input_source_kind(source: &InferenceInputSource) -> &'static str {
    match source {
        InferenceInputSource::Path { .. } => "path (redacted)",
        InferenceInputSource::Url { .. } => "url (redacted)",
        InferenceInputSource::Base64 { .. } => "base64 (redacted)",
        InferenceInputSource::Text { .. } => "text (redacted)",
    }
}

fn media_debug_parameter_value(path: &str, value: &ParameterValue) -> String {
    match value {
        ParameterValue::Null => "null".to_string(),
        ParameterValue::Boolean(value) => value.to_string(),
        ParameterValue::Integer(value) => value.to_string(),
        ParameterValue::Number(value) => trim_float(format!("{value:.8}")),
        ParameterValue::String(value)
            if media_parameter_is_sensitive(path)
                || contains_absolute_path(value)
                || value.contains("http://")
                || value.contains("https://")
                || value.starts_with("data:") =>
        {
            "<redacted>".to_string()
        }
        ParameterValue::String(value) => {
            serde_json::to_string(value).unwrap_or_else(|_| "\"<unprintable>\"".to_string())
        }
        ParameterValue::List(values) => format!("[{} item(s)]", values.len()),
        ParameterValue::Object(values) => format!("{{{} field(s)}}", values.len()),
    }
}

fn media_parameter_is_sensitive(path: &str) -> bool {
    let leaf = path.rsplit('.').next().unwrap_or(path).to_ascii_lowercase();
    let sensitive_token = leaf
        .split(|character: char| !character.is_ascii_alphanumeric())
        .any(|token| {
            matches!(
                token,
                "prompt"
                    | "lyrics"
                    | "text"
                    | "hotword"
                    | "path"
                    | "file"
                    | "url"
                    | "token"
                    | "reference"
                    | "model"
                    | "checkpoint"
                    | "vae"
                    | "refiner"
                    | "image"
                    | "video"
                    | "audio"
                    | "lora"
                    | "adapter"
            )
        });
    sensitive_token
        || matches!(
            leaf.as_str(),
            "api_key" | "apikey" | "filename" | "filepath" | "uri" | "base64"
        )
}

fn media_sanitize_debug_text(text: &str) -> String {
    text.split_whitespace()
        .map(|part| {
            let candidate = part.trim_matches(|character: char| {
                matches!(
                    character,
                    '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';'
                )
            });
            if candidate.contains("http://") || candidate.contains("https://") {
                "<url>".to_string()
            } else if contains_absolute_path(candidate) {
                "<path>".to_string()
            } else {
                part.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_absolute_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.iter().enumerate().any(|(index, byte)| {
        (*byte == b'/'
            && (index == 0
                || matches!(
                    bytes[index - 1],
                    b'=' | b':' | b'\'' | b'"' | b'(' | b'[' | b'{'
                )))
            || (*byte == b'\\'
                && (index == 0
                    || matches!(
                        bytes[index - 1],
                        b'=' | b':' | b'\'' | b'"' | b'(' | b'[' | b'{'
                    )))
            || (byte.is_ascii_alphabetic()
                && bytes.get(index + 1) == Some(&b':')
                && bytes
                    .get(index + 2)
                    .is_some_and(|separator| matches!(separator, b'/' | b'\\')))
    })
}

fn media_metadata_seconds(metadata: &Value, key: &str) -> Option<f64> {
    metadata
        .get(key)
        .and_then(json_value_f64)
        .and_then(|value| (value.is_finite() && value >= 0.0).then_some(value))
}

fn media_phase_seconds(metadata: &Value, key: &str) -> Option<f64> {
    media_metadata_seconds(metadata, key).or_else(|| {
        metadata
            .get("timings")
            .and_then(|timings| media_metadata_seconds(timings, key))
    })
}

fn json_value_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn json_value_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn positive_finite(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite() && *value > 0.0)
}

fn media_output_u64_first(result: &InferenceResult, key: &str) -> Option<u64> {
    result
        .outputs
        .iter()
        .find_map(|output| output.backend_metadata.get(key).and_then(json_value_u64))
}

fn media_output_u64_sum(result: &InferenceResult, key: &str) -> Option<u64> {
    let values = result
        .outputs
        .iter()
        .filter_map(|output| output.backend_metadata.get(key).and_then(json_value_u64))
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.into_iter().fold(0_u64, u64::saturating_add))
}

fn media_output_f64_first(result: &InferenceResult, key: &str) -> Option<f64> {
    result.outputs.iter().find_map(|output| {
        output
            .backend_metadata
            .get(key)
            .and_then(json_value_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
    })
}

fn media_output_duration(result: &InferenceResult) -> Option<f64> {
    let durations = result
        .outputs
        .iter()
        .filter_map(|output| output.duration)
        .filter(|duration| duration.is_finite() && *duration >= 0.0)
        .collect::<Vec<_>>();
    let has_durations = !durations.is_empty();
    let total = durations.into_iter().sum::<f64>();
    (has_durations && total.is_finite() && total >= 0.0).then_some(total)
}

fn media_output_duration_count(result: &InferenceResult) -> usize {
    result
        .outputs
        .iter()
        .filter_map(|output| output.duration)
        .filter(|duration| duration.is_finite() && *duration >= 0.0)
        .count()
}

fn media_output_duration_summary(result: &InferenceResult) -> Option<String> {
    let durations = result
        .outputs
        .iter()
        .filter_map(|output| output.duration)
        .filter(|duration| duration.is_finite() && *duration >= 0.0)
        .collect::<Vec<_>>();
    let first = *durations.first()?;
    if durations.len() == 1 {
        return Some(format_duration(first));
    }
    if durations
        .iter()
        .all(|duration| (duration - first).abs() <= (first.abs() * 1e-6).max(1e-9))
    {
        Some(format!(
            "{} each ({} outputs)",
            format_duration(first),
            durations.len()
        ))
    } else {
        let total = durations.iter().sum::<f64>();
        if !total.is_finite() {
            return None;
        }
        Some(format!(
            "{} total ({} outputs)",
            format_duration(total),
            durations.len()
        ))
    }
}

fn media_audio_real_time_factor_is_meaningful(task: InferenceTask) -> bool {
    !matches!(
        task,
        InferenceTask::StemGeneration | InferenceTask::StemSeparation
    )
}

fn media_effective_u64(result: &InferenceResult, leaf: &str) -> Option<u64> {
    result
        .effective_request
        .u64_parameter(&format!("{}.{leaf}", result.task.parameter_namespace()))
}

fn media_effective_f64(result: &InferenceResult, leaf: &str) -> Option<f64> {
    result
        .effective_request
        .f64_parameter(&format!("{}.{leaf}", result.task.parameter_namespace()))
}

fn media_seed(result: &InferenceResult) -> Option<u64> {
    result
        .outputs
        .iter()
        .find_map(|output| output.seed)
        .or_else(|| media_effective_u64(result, "seed"))
}

fn media_output_resolutions(result: &InferenceResult) -> Option<String> {
    let mut resolutions = result
        .outputs
        .iter()
        .filter_map(|output| Some((output.width?, output.height?)))
        .collect::<Vec<_>>();
    resolutions.sort_unstable();
    resolutions.dedup();
    match resolutions.as_slice() {
        [] => None,
        [(width, height)] => Some(format!("{width}x{height}")),
        values => Some(
            values
                .iter()
                .map(|(width, height)| format!("{width}x{height}"))
                .collect::<Vec<_>>()
                .join(", "),
        ),
    }
}

fn media_output_formats(result: &InferenceResult) -> Option<String> {
    let mut formats = result
        .outputs
        .iter()
        .map(|output| output.mime_type.as_str())
        .collect::<Vec<_>>();
    formats.sort_unstable();
    formats.dedup();
    (!formats.is_empty()).then(|| formats.join(", "))
}

fn serialized_enum_label<T: Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}

fn format_media_degradation(degradation: &ExecutionDegradation) -> String {
    match degradation {
        ExecutionDegradation::CpuOffload => "cpu offload".to_string(),
        ExecutionDegradation::SequentialOffload => "sequential offload".to_string(),
        ExecutionDegradation::ComponentOffload => "component offload".to_string(),
        ExecutionDegradation::VaeTiling => "VAE tiling".to_string(),
        ExecutionDegradation::TemporalWindowing => "temporal windowing".to_string(),
        ExecutionDegradation::SlowerAttention { backend } => {
            format!("slower attention ({backend})")
        }
    }
}

fn media_optional_i32(value: Option<i32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "N/A".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        capabilities::{InferenceTask, InputModality, OutputModality},
        inference::{
            EffectiveInferenceRequest, EstimateConfidence, ExecutionPlan, FitAssessment,
            InferenceInput, InferenceInputSource, ParameterPolicy, ParameterSource, ParameterValue,
            PlanCandidateDecision, PlanCandidateStatus, ResolvedParameter, WorkloadEstimate,
        },
        inference_service::{
            InferenceResult, OutputMetadata, RuntimeAttemptOutcome, RuntimeAttemptTiming,
        },
    };
    use serde_json::{Value, json};

    #[test]
    fn media_verbose_stats_are_task_specific_and_never_report_tokens() {
        let mut image = test_media_diagnostic_result("verbose-image");
        image.backend_metadata = json!({
            "elapsed_seconds": 9.5,
            "backend": {
                "runtime": "diffusers",
                "device": "cuda",
                "dtype": "float16",
                "model_load_seconds": 6.0,
                "inference_seconds": 2.0,
                "encoding_seconds": 0.5
            }
        });
        image.outputs[0].width = Some(512);
        image.outputs[0].height = Some(512);
        image.effective_request.parameters.insert(
            "image.steps".to_string(),
            ResolvedParameter {
                value: ParameterValue::Integer(20),
                source: ParameterSource::RequestOverride,
            },
        );
        image.timings.resolve_seconds = 0.1;
        image.timings.estimate_seconds = 0.2;
        image.timings.planning_seconds = 0.1;
        image.timings.output_setup_seconds = 0.01;
        image.timings.execution_seconds = 9.5;
        image.timings.finalization_seconds = 0.02;
        image.timings.runtime_attempts = vec![RuntimeAttemptTiming {
            runtime: "media-companion-cuda".to_string(),
            duration_seconds: 9.5,
            outcome: RuntimeAttemptOutcome::Succeeded,
            error: None,
        }];
        image.timings.update_total();

        let mut output = Vec::new();
        write_media_verbose_stats(
            &mut output,
            &image,
            MediaCliTimings {
                total_seconds: 10.0,
                service_seconds: 9.8,
                publication_seconds: 0.2,
            },
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("device:                 cuda"));
        assert!(output.contains("model load/setup:       6s"));
        assert!(output.contains("resolution:             512x512"));
        assert!(output.contains("steps:                  20"));
        assert!(output.contains("generation rate:"));
        assert!(output.contains("MP/s"));
        assert!(!output.contains("token"));

        let mut video = image.clone();
        retask_diagnostic_result(
            &mut video,
            InferenceTask::VideoGeneration,
            OutputModality::Video,
        );
        video.outputs[0].duration = Some(2.0);
        video.outputs[0].backend_metadata = json!({"frames": 48, "fps": 24.0});
        video.backend_metadata["backend"]["inference_seconds"] = json!(4.0);
        let mut output = Vec::new();
        write_media_verbose_stats(
            &mut output,
            &video,
            MediaCliTimings {
                total_seconds: 10.0,
                service_seconds: 9.8,
                publication_seconds: 0.2,
            },
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("generated frames:       48"));
        assert!(output.contains("playback rate:          24.00 fps"));
        assert!(output.contains("generation rate:        12.00 frames/s"));
        assert!(!output.contains("tokens/s"));

        let mut audio = image;
        retask_diagnostic_result(
            &mut audio,
            InferenceTask::TextToSpeech,
            OutputModality::Audio,
        );
        audio.outputs[0].width = None;
        audio.outputs[0].height = None;
        audio.outputs[0].duration = Some(5.0);
        audio.outputs[0].backend_metadata = json!({"sample_rate": 24_000, "channels": 1});
        audio.backend_metadata["backend"]["inference_seconds"] = json!(2.0);
        let mut output = Vec::new();
        write_media_verbose_stats(
            &mut output,
            &audio,
            MediaCliTimings {
                total_seconds: 10.0,
                service_seconds: 9.8,
                publication_seconds: 0.2,
            },
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("output duration:        5s"));
        assert!(output.contains("sample rate:            24000 Hz"));
        assert!(output.contains("channels:               1"));
        assert!(output.contains("real-time factor:       0.400"));
        assert!(!output.contains("token"));

        let mut second = audio.outputs[0].clone();
        second.id = "out-diagnostic-1".to_string();
        audio.outputs.push(second);
        let mut output = Vec::new();
        write_media_verbose_stats(
            &mut output,
            &audio,
            MediaCliTimings {
                total_seconds: 10.0,
                service_seconds: 9.8,
                publication_seconds: 0.2,
            },
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("output duration:        5s each (2 outputs)"));
        assert!(output.contains("aggregate RTF:          0.200"));
    }

    #[test]
    fn media_debug_reports_routing_without_leaking_private_values() {
        let mut result = test_media_diagnostic_result("debug-redaction");
        result.effective_request.prompt = Some("PROMPT_SECRET".to_string());
        result.effective_request.negative_prompt = Some("NEGATIVE_SECRET".to_string());
        result.effective_request.inputs = vec![
            InferenceInput {
                modality: InputModality::Image,
                role: "initial_image".to_string(),
                source: InferenceInputSource::Path {
                    path: "/private/input/SECRET_IMAGE.png".to_string(),
                },
                mime_type: Some("image/png".to_string()),
            },
            InferenceInput {
                modality: InputModality::Image,
                role: "reference_image".to_string(),
                source: InferenceInputSource::Base64 {
                    data: "BASE64_SECRET".to_string(),
                },
                mime_type: Some("image/png".to_string()),
            },
        ];
        for (path, value) in [
            (
                "image.prompt",
                ParameterValue::String("PARAMETER_SECRET".to_string()),
            ),
            ("image.width", ParameterValue::Integer(512)),
            (
                "image.reference_image",
                ParameterValue::String("/private/reference.png".to_string()),
            ),
        ] {
            result.effective_request.parameters.insert(
                path.to_string(),
                ResolvedParameter {
                    value,
                    source: ParameterSource::RequestOverride,
                },
            );
            result
                .effective_request
                .explicit_parameters
                .insert(path.to_string());
        }
        result.effective_request.parameters.insert(
            "routing.backend".to_string(),
            ResolvedParameter {
                value: ParameterValue::String("media-companion".to_string()),
                source: ParameterSource::TaskDefault,
            },
        );
        result.plan.candidates = vec![
            PlanCandidateDecision {
                runtime_id: "media-companion-cuda".to_string(),
                backend: "media-companion".to_string(),
                status: PlanCandidateStatus::Rejected,
                score: None,
                reasons: vec![
                    "runner missing at /home/private/bin/runner https://example.test/?token=URL_SECRET"
                        .to_string(),
                    "runner=/home/private/bin/EMBEDDED_PATH_SECRET url=https://example.test/?token=EMBEDDED_URL_SECRET"
                        .to_string(),
                    "windows=C:\\Users\\private\\WINDOWS_PATH_SECRET quoted='/home/private/QUOTED_PATH_SECRET'"
                        .to_string(),
                ],
                degradations: Vec::new(),
            },
            PlanCandidateDecision {
                runtime_id: "media-companion-cpu".to_string(),
                backend: "media-companion".to_string(),
                status: PlanCandidateStatus::Accepted,
                score: Some(25),
                reasons: Vec::new(),
                degradations: Vec::new(),
            },
        ];
        result.plan.selected_runtime = Some("media-companion-cpu".to_string());
        result.backend_metadata = json!({
            "model_path": "/private/model/SECRET_MODEL",
            "effective_parameters": {"prompt": "BACKEND_PROMPT_SECRET"},
            "backend": {
                "runtime": "diffusers",
                "device": "cpu",
                "dtype": "float32",
                "text": "TRANSCRIPT_SECRET",
                "translated_parameters": ["prompt", "width"]
            }
        });

        let mut output = Vec::new();
        write_media_routing_debug(
            &mut output,
            &result.effective_request,
            &result.estimate,
            &result.plan,
        )
        .unwrap();
        result.effective_request.parameters.insert(
            "image.steps".to_string(),
            ResolvedParameter {
                value: ParameterValue::Integer(16),
                source: ParameterSource::BackendAdjustment,
            },
        );
        result.effective_request.parameters.insert(
            "image.backend_reference".to_string(),
            ResolvedParameter {
                value: ParameterValue::String("BACKEND_REFERENCE_SECRET".to_string()),
                source: ParameterSource::BackendAdjustment,
            },
        );
        write_media_backend_debug(&mut output, &result).unwrap();
        let output = String::from_utf8(output).unwrap();

        assert!(output.contains("candidate runtimes:"));
        assert!(output.contains("status=rejected"));
        assert!(output.contains("routing backend:        media-companion (task_default)"));
        assert!(output.contains("resolved parameters:"));
        assert!(output.contains("image.width = 512 (request_override explicit)"));
        assert!(output.contains("image.prompt = <redacted>"));
        assert!(output.contains("backend adjustments:"));
        assert!(output.contains("image.steps = 16"));
        assert!(output.contains("image.backend_reference = <redacted>"));
        assert!(output.contains("source=path (redacted)"));
        assert!(output.contains("source=base64 (redacted)"));
        assert!(output.contains("<path>"));
        assert!(output.contains("<url>"));
        for secret in [
            "PROMPT_SECRET",
            "NEGATIVE_SECRET",
            "PARAMETER_SECRET",
            "SECRET_IMAGE",
            "BASE64_SECRET",
            "reference.png",
            "SECRET_MODEL",
            "BACKEND_PROMPT_SECRET",
            "BACKEND_REFERENCE_SECRET",
            "TRANSCRIPT_SECRET",
            "URL_SECRET",
            "EMBEDDED_PATH_SECRET",
            "EMBEDDED_URL_SECRET",
            "WINDOWS_PATH_SECRET",
            "QUOTED_PATH_SECRET",
        ] {
            assert!(!output.contains(secret), "debug output leaked {secret}");
        }
    }

    #[test]
    fn media_debug_redacts_generic_media_and_adapter_parameters() {
        for path in [
            "image.reference",
            "image.model_path",
            "image.control_model",
            "image.checkpoint",
            "image.refiner",
            "image.vae",
            "image.control_image",
            "video.guide_video",
            "audio.guide_audio",
            "image.lora",
            "image.adapter_name",
        ] {
            assert_eq!(
                media_debug_parameter_value(
                    path,
                    &ParameterValue::String("PRIVATE_VALUE".to_string())
                ),
                "<redacted>",
                "{path} was not treated as sensitive"
            );
        }
        for value in [
            "/private/scheduler.json",
            "C:\\Users\\private\\scheduler.json",
            "https://example.test/model?token=secret",
            "data:application/octet-stream;base64,secret",
        ] {
            assert_eq!(
                media_debug_parameter_value(
                    "image.scheduler",
                    &ParameterValue::String(value.to_string())
                ),
                "<redacted>",
                "{value} was not treated as sensitive"
            );
        }
        assert_eq!(
            media_debug_parameter_value("image.reference_weight", &ParameterValue::Number(0.75)),
            "0.75"
        );
        assert_eq!(
            media_debug_parameter_value("image.vae_tiling", &ParameterValue::Boolean(true)),
            "true"
        );
        assert_eq!(
            media_debug_parameter_value(
                "routing.performance_preference",
                &ParameterValue::String("balanced".to_string())
            ),
            "\"balanced\""
        );
        assert_eq!(
            media_debug_parameter_value(
                "routing.profile",
                &ParameterValue::String("quality".to_string())
            ),
            "\"quality\""
        );
    }

    #[test]
    fn media_verbose_omits_unmeasured_or_invalid_phases_and_rates() {
        let mut result = test_media_diagnostic_result("verbose-missing-timings");
        result.backend_metadata = json!({
            "backend": {
                "runtime": "external",
                "inference_seconds": "NaN"
            }
        });

        let mut output = Vec::new();
        write_media_verbose_stats(
            &mut output,
            &result,
            MediaCliTimings {
                total_seconds: 1.0,
                service_seconds: 0.9,
                publication_seconds: 0.1,
            },
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();

        assert!(!output.contains("model load/setup:"));
        assert!(!output.contains("\ninference:"));
        assert!(!output.contains("encoding:"));
        assert!(!output.contains("generation rate:"));
        assert!(!output.contains("NaN"));
    }

    #[test]
    fn failed_attempt_report_keeps_timings_without_dumping_backend_errors() {
        let attempts = vec![RuntimeAttemptTiming {
            runtime: "media-companion-cuda".to_string(),
            duration_seconds: 3.5,
            outcome: RuntimeAttemptOutcome::Failed,
            error: Some(
                "load failed at /private/model with prompt BACKEND_ERROR_SECRET".to_string(),
            ),
        }];
        let mut output = Vec::new();

        write_media_failed_attempts(&mut output, &attempts, 4.0).unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("media execution failed:"));
        assert!(output.contains("service duration:       4s"));
        assert!(output.contains("media-companion-cuda outcome=failed duration=3.5s"));
        assert!(!output.contains("/private/model"));
        assert!(!output.contains("BACKEND_ERROR_SECRET"));
    }

    fn test_media_diagnostic_result(_name: &str) -> InferenceResult {
        let task = InferenceTask::ImageGeneration;
        InferenceResult {
            id: "out-diagnostic".to_string(),
            task,
            model: "test-model".to_string(),
            runtime: "test-runtime".to_string(),
            outputs: vec![OutputMetadata {
                id: "out-diagnostic-0".to_string(),
                task,
                model: "test-model".to_string(),
                runtime: "test-runtime".to_string(),
                path: "diagnostic.png".to_string(),
                mime_type: "image/png".to_string(),
                size_bytes: 10,
                width: Some(16),
                height: Some(16),
                duration: None,
                seed: Some(1),
                effective_parameters: Default::default(),
                created_unix: 1,
                backend_metadata: Value::Null,
            }],
            effective_request: EffectiveInferenceRequest {
                model: "test-model".to_string(),
                task,
                prompt: Some("test".to_string()),
                negative_prompt: None,
                inputs: Vec::new(),
                output_modality: OutputModality::Image,
                parameters: Default::default(),
                explicit_parameters: Default::default(),
                parameter_policy: ParameterPolicy::Strict,
                warnings: Vec::new(),
            },
            estimate: WorkloadEstimate {
                task,
                download_size_bytes: None,
                weight_payload_bytes: None,
                accelerator_peak_bytes: None,
                host_peak_bytes: None,
                output_size_bytes: Some(10),
                fit: FitAssessment::Fits,
                confidence: EstimateConfidence::Exact,
                assumptions: Vec::new(),
                warnings: Vec::new(),
                recommendations: Vec::new(),
            },
            plan: ExecutionPlan {
                task,
                selected_runtime: Some("test-runtime".to_string()),
                selected_backend: Some("test-backend".to_string()),
                score: Some(1),
                candidates: Vec::new(),
                backend_fallback: false,
                degradations: Vec::new(),
                model_or_quality_downgrades: Vec::new(),
            },
            backend_metadata: Value::Null,
            timings: Default::default(),
            warnings: Vec::new(),
            created_unix: 1,
        }
    }

    fn retask_diagnostic_result(
        result: &mut InferenceResult,
        task: InferenceTask,
        modality: OutputModality,
    ) {
        result.task = task;
        result.effective_request.task = task;
        result.effective_request.output_modality = modality;
        result.estimate.task = task;
        result.plan.task = task;
        for output in &mut result.outputs {
            output.task = task;
        }
    }
}
