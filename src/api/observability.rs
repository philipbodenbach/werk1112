use super::state::ApiState;
use crate::{
    observability::{BackendSnapshot, Snapshot, now_ms},
    werk_protocol::{ControlContext, MemoryStatusResponse},
};
use axum::{
    extract::State,
    http::{HeaderMap, header},
    response::{IntoResponse, Response},
};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Default)]
pub(super) struct SampleCache {
    data: Mutex<Option<Cached>>,
    gate: Arc<tokio::sync::Mutex<()>>,
}
#[derive(Clone)]
struct Cached {
    swap: Option<u64>,
    host_free: Option<u64>,
    time: Instant,
    timestamp: u64,
    backends: Vec<BackendSnapshot>,
    memory: Option<MemoryStatusResponse>,
}

async fn collect(state: &ApiState) -> Snapshot {
    let cached = state
        .telemetry_cache
        .data
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let stale = cached
        .as_ref()
        .is_none_or(|c| c.time.elapsed() >= Duration::from_secs(2));
    if stale {
        if let Ok(gate) = state.telemetry_cache.gate.clone().try_lock_owned() {
            // Recheck after acquisition: another scrape may just have filled it.
            if state
                .telemetry_cache
                .data
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .is_some_and(|c| c.time.elapsed() < Duration::from_secs(2))
            {
                return current(state);
            }
            let backend = state.backend.clone();
            let Ok((_gate, mut backends, swap, host_free)) =
                tokio::task::spawn_blocking(move || {
                    let mut system = sysinfo::System::new();
                    system.refresh_memory();
                    (
                        gate,
                        backend.telemetry(),
                        sysinfo::IS_SUPPORTED_SYSTEM.then(|| system.used_swap()),
                        sysinfo::IS_SUPPORTED_SYSTEM.then(|| system.free_memory()),
                    )
                })
                .await
            else {
                return current(state);
            };
            backends.sort_by(|a, b| (&a.backend, &a.instance).cmp(&(&b.backend, &b.instance)));
            let timestamp = now_ms();
            if let Some(previous) = &cached {
                let elapsed = timestamp.saturating_sub(previous.timestamp) as f64 / 1000.;
                for current in &mut backends {
                    if let Some(old) = previous
                        .backends
                        .iter()
                        .find(|b| b.instance == current.instance && b.backend == current.backend)
                    {
                        let rates = crate::observability::Rates::between(old, current, elapsed);
                        for (key, value) in [
                            ("decode_tokens_per_second_estimate", rates.decode_estimate),
                            ("expert_cache_hit_ratio", rates.expert_hit_ratio),
                            ("expert_read_bytes_per_second", rates.read_bytes_per_second),
                        ] {
                            if let Some(value) = value {
                                current.gauges.insert(key.into(), value);
                            }
                        }
                    }
                }
            }
            let memory = state
                .werk_control
                .memory_status(ControlContext::local("observability"))
                .await
                .ok();
            *state
                .telemetry_cache
                .data
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(Cached {
                time: Instant::now(),
                timestamp,
                swap,
                host_free,
                backends,
                memory,
            });
        }
    }
    current(state)
}
fn current(state: &ApiState) -> Snapshot {
    let mut snapshot = state.telemetry.snapshot();
    if let Some(c) = state
        .telemetry_cache
        .data
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
    {
        snapshot.backends = c.backends.clone();
        snapshot.host_swap_used_bytes = c.swap;
        snapshot.host_memory_free_bytes = c.host_free;
        snapshot.memory = c.memory.clone();
        snapshot.backend_observed_at_ms = Some(c.timestamp);
    }
    snapshot
}
pub(super) async fn metrics(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let text = crate::observability::prometheus::render(&collect(&state).await);
    (
        [
            (
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            ),
            (header::CACHE_CONTROL, "no-store"),
        ],
        text,
    )
        .into_response()
}
pub(super) async fn snapshot(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    let (id, _) = match super::werk::request_context(&state, &headers).await {
        Ok(context) => context,
        Err(response) => return response,
    };
    super::werk::success(id, collect(&state).await)
}
