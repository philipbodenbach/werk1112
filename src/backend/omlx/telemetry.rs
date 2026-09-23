use super::OmlxBackend;
use crate::observability::BackendSnapshot;
use serde_json::Value;
use std::time::Duration;

pub(super) fn sample(backend: &OmlxBackend) -> Vec<BackendSnapshot> {
    // Clone references before I/O: monitoring never holds the worker registry
    // lock while a request is running or waits on generation locks.
    let servers: Vec<_> = backend
        .servers
        .lock()
        .map(|s| s.values().cloned().collect())
        .unwrap_or_default();
    servers
        .into_iter()
        .take(8)
        .map(|server| {
            let mut sample = BackendSnapshot {
                backend: "omlx".into(),
                instance: server.instance_id.clone(),
                model: server
                    .logical_model_id
                    .clone()
                    .unwrap_or_else(|| server.model_name.clone()),
                ..Default::default()
            };
            if !server.is_running() {
                return sample;
            }
            let Ok(status) =
                server.json_request("GET", "/api/status", None, Duration::from_secs(2))
            else {
                return sample;
            };
            sample.available = true;
            counters(
                &mut sample,
                &status,
                &[
                    ("total_requests", "requests_completed_total"),
                    ("total_prompt_tokens", "prompt_tokens_total"),
                    ("total_completion_tokens", "output_tokens_total"),
                    ("total_cached_tokens", "cached_tokens_total"),
                ],
            );
            gauges(
                &mut sample,
                &status,
                &[
                    ("active_requests", "requests_active"),
                    ("waiting_requests", "requests_waiting"),
                    ("uptime_seconds", "uptime_seconds"),
                    ("avg_prefill_tps", "prefill_tokens_per_second_average"),
                    ("avg_generation_tps", "decode_tokens_per_second_average"),
                ],
            );
            if server.expert_offload {
                if let Ok(experts) =
                    server.json_request("GET", "/werk/experts/status", None, Duration::from_secs(2))
                {
                    counters(
                        &mut sample,
                        &experts,
                        &[
                            ("cache_hits", "expert_cache_hits_total"),
                            ("cache_misses", "expert_cache_misses_total"),
                            ("cache_evictions", "expert_cache_evictions_total"),
                            ("disk_bytes_read", "expert_read_bytes_total"),
                            ("budget_reductions", "expert_budget_reductions_total"),
                            ("ngram_cache_hits", "ngram_cache_hits_total"),
                            ("ngram_cache_misses", "ngram_cache_misses_total"),
                        ],
                    );
                    gauges(
                        &mut sample,
                        &experts,
                        &[
                            ("cache_budget_bytes", "expert_cache_budget_bytes"),
                            (
                                "effective_cache_budget_bytes",
                                "expert_cache_effective_budget_bytes",
                            ),
                            ("resident_cache_bytes", "expert_cache_resident_bytes"),
                            ("ngram_resident_cache_bytes", "ngram_cache_resident_bytes"),
                            (
                                "ngram_effective_cache_budget_bytes",
                                "ngram_cache_budget_bytes",
                            ),
                        ],
                    );
                    gauges(
                        &mut sample,
                        &experts["last_decode_admission"],
                        &[
                            ("cached_tokens", "decode_context_tokens"),
                            ("current_bytes", "decode_memory_observed_bytes"),
                            ("ceiling_bytes", "decode_memory_ceiling_bytes"),
                        ],
                    );
                }
            }
            sample
        })
        .collect()
}
fn counters(sample: &mut BackendSnapshot, value: &Value, names: &[(&str, &str)]) {
    for (source, target) in names {
        if let Some(n) = value[*source].as_u64() {
            sample.counters.insert((*target).into(), n);
        }
    }
}
fn gauges(sample: &mut BackendSnapshot, value: &Value, names: &[(&str, &str)]) {
    for (source, target) in names {
        if let Some(n) = value[*source]
            .as_f64()
            .filter(|n| n.is_finite() && *n >= 0.)
        {
            sample.gauges.insert((*target).into(), n);
        }
    }
}

#[cfg(test)]
mod observability_tests {
    use super::*;
    #[test]
    fn missing_and_invalid_native_values_are_not_zero_metrics() {
        let mut snapshot = BackendSnapshot::default();
        let value = serde_json::json!({"hits":12,"memory":4096,"missing":null,"negative":-1});
        counters(
            &mut snapshot,
            &value,
            &[("hits", "hits_total"), ("missing", "missing_total")],
        );
        gauges(
            &mut snapshot,
            &value,
            &[
                ("memory", "memory_bytes"),
                ("negative", "negative"),
                ("absent", "absent"),
            ],
        );
        assert_eq!(snapshot.counters.len(), 1);
        assert_eq!(snapshot.gauges.len(), 1);
        assert_eq!(snapshot.gauges["memory_bytes"], 4096.);
    }
}
