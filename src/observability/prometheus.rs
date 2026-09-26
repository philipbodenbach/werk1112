use super::Snapshot;
use std::fmt::Write;

fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}
pub fn render(s: &Snapshot) -> String {
    let mut out = String::new();
    let mut metric = |name: &str, kind: &str, help: &str, value: f64| {
        if value.is_finite() {
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}"
            );
        }
    };
    metric(
        "werk_uptime_seconds",
        "gauge",
        "Werk server uptime.",
        s.uptime_seconds,
    );
    if let Some(swap) = s.host_swap_used_bytes {
        metric(
            "werk_host_swap_used_bytes",
            "gauge",
            "Host-wide used swap.",
            swap as f64,
        );
    }
    if let Some(free) = s.host_memory_free_bytes {
        metric(
            "werk_host_memory_free_bytes",
            "gauge",
            "Physically free host memory, excluding reclaimable cache.",
            free as f64,
        );
    }
    for (name, value) in [
        ("started", s.totals.started),
        ("completed", s.totals.completed),
        ("errors", s.totals.errors),
        ("cancelled", s.totals.cancelled),
    ] {
        metric(
            &format!("werk_requests_{name}_total"),
            "counter",
            "Chat generation requests since server start.",
            value as f64,
        );
    }
    metric(
        "werk_requests_active",
        "gauge",
        "Chat generations currently active.",
        s.totals.active as f64,
    );
    for (name, value) in [
        ("prompt", s.totals.prompt_tokens),
        ("output", s.totals.output_tokens),
        ("cached", s.totals.cached_tokens),
    ] {
        metric(
            &format!("werk_{name}_tokens_total"),
            "counter",
            "Tokens from completed backend usage reports.",
            value as f64,
        );
    }
    metric(
        "werk_request_duration_seconds_total",
        "counter",
        "Total observed chat generation duration including failed and cancelled requests.",
        s.totals.duration_seconds,
    );
    if let Some(time) = s.backend_observed_at_ms {
        metric(
            "werk_backend_sample_age_seconds",
            "gauge",
            "Age of the cached backend sample.",
            s.observed_at_ms.saturating_sub(time) as f64 / 1000.,
        );
    }
    if let Some(memory) = &s.memory {
        for (tier, m) in [("host", &memory.host), ("accelerator", &memory.accelerator)] {
            for (field, v) in [
                ("capacity", m.capacity_bytes),
                ("available", m.available_bytes),
            ] {
                if let Some(v) = v {
                    metric(
                        &format!("werk_{tier}_memory_{field}_bytes"),
                        "gauge",
                        "Physical memory telemetry when available.",
                        v as f64,
                    );
                }
            }
        }
    }
    let mut latest = std::collections::BTreeMap::new();
    for request in &s.requests {
        if request.output_tokens.is_some() {
            latest.entry(&request.model).or_insert(request);
        }
    }
    for (suffix, help) in [
        (
            "decode_tokens_per_second",
            "Last completed request backend decode rate.",
        ),
        (
            "prefill_tokens_per_second",
            "Last completed request backend uncached prefill rate.",
        ),
        (
            "first_output_seconds",
            "Last completed request time to first output.",
        ),
        (
            "duration_seconds",
            "Last completed request generation duration.",
        ),
    ] {
        let name = format!("werk_last_request_{suffix}");
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} gauge");
        for (model, request) in &latest {
            let value = match suffix {
                "decode_tokens_per_second" => request.decode_tokens_per_second,
                "prefill_tokens_per_second" => request.prefill_tokens_per_second,
                "first_output_seconds" => request.first_output_seconds,
                _ => Some(request.elapsed_seconds),
            };
            if let Some(value) = value.filter(|v| v.is_finite()) {
                let _ = writeln!(out, "{name}{{model=\"{}\"}} {value}", escape(model));
            }
        }
    }
    // Group metadata once per family even when several models are loaded.
    let mut families = std::collections::BTreeMap::<(String, &str), Vec<(String, f64)>>::new();
    for b in &s.backends {
        let labels = format!(
            "backend=\"{}\",model=\"{}\",worker=\"{}\"",
            escape(&b.backend),
            escape(&b.model),
            escape(&b.instance)
        );
        families
            .entry(("available".into(), "gauge"))
            .or_default()
            .push((labels.clone(), if b.available { 1. } else { 0. }));
        if !b.available {
            continue;
        }
        for (name, value) in &b.counters {
            families
                .entry((name.clone(), "counter"))
                .or_default()
                .push((labels.clone(), *value as f64));
        }
        for (name, value) in &b.gauges {
            if value.is_finite() {
                families
                    .entry((name.clone(), "gauge"))
                    .or_default()
                    .push((labels.clone(), *value));
            }
        }
    }
    for ((name, kind), values) in families {
        if !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
            continue;
        }
        let name = format!("werk_backend_{name}");
        let _ = writeln!(
            out,
            "# HELP {name} Native backend observation.\n# TYPE {name} {kind}"
        );
        for (labels, v) in values {
            let _ = writeln!(out, "{name}{{{labels}}} {v}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn labels_are_escaped_and_missing_values_omitted() {
        let t = super::super::Telemetry::default();
        let mut s = t.snapshot();
        let mut b = super::super::BackendSnapshot {
            backend: "test".into(),
            model: "a\"\nb".into(),
            available: true,
            ..Default::default()
        };
        b.gauges.insert("bad".into(), f64::NAN);
        s.backends.push(b);
        let text = render(&s);
        assert!(text.contains("model=\"a\\\"\\nb\""));
        assert!(!text.contains("NaN"));
        assert!(text.ends_with('\n'));
    }
}
