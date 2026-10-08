//! Bounded, content-free inference telemetry shared by the TUI and Prometheus.
use crate::backend::{GenerateResponse, GenerateStream, GenerateStreamEvent, GenerationTimings};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
#[cfg(test)]
use tokio_stream::StreamExt;

pub mod prometheus;

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackendSnapshot {
    pub backend: String,
    pub instance: String,
    pub model: String,
    pub available: bool,
    pub counters: BTreeMap<String, u64>,
    pub gauges: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RequestSnapshot {
    pub id: u64,
    pub model: String,
    pub state: String,
    pub started_ms: u64,
    pub elapsed_seconds: f64,
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub first_output_seconds: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis: Option<AnalysisSnapshot>,
}

/// Scalar, content-free metadata for non-generative inference. A model weight
/// cache hit is distinct from reuse of cached prompt tokens.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalysisSnapshot {
    pub task: String,
    pub runtime: Option<String>,
    pub device: Option<String>,
    pub dtype: Option<String>,
    pub model_cache_hit: Option<bool>,
    pub load_seconds: Option<f64>,
    pub inference_seconds: Option<f64>,
    pub worker_seconds: Option<f64>,
    pub results: Option<u64>,
    pub attempts: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Totals {
    pub started: u64,
    pub completed: u64,
    pub errors: u64,
    pub cancelled: u64,
    pub active: u64,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub duration_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    #[serde(default)]
    pub host_swap_used_bytes: Option<u64>,
    #[serde(default)]
    pub host_memory_free_bytes: Option<u64>,
    pub schema_version: u32,
    pub observed_at_ms: u64,
    pub server_started_ms: u64,
    pub uptime_seconds: f64,
    pub totals: Totals,
    pub requests: Vec<RequestSnapshot>,
    pub backends: Vec<BackendSnapshot>,
    pub memory: Option<crate::werk_protocol::MemoryStatusResponse>,
    pub backend_observed_at_ms: Option<u64>,
}

struct Records {
    totals: Totals,
    active: BTreeMap<u64, RequestSnapshot>,
    recent: VecDeque<RequestSnapshot>,
}
pub struct Telemetry {
    start: Instant,
    started_ms: u64,
    records: Mutex<Records>,
}
impl Default for Telemetry {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            started_ms: now_ms(),
            records: Mutex::new(Records {
                totals: Totals::default(),
                active: BTreeMap::new(),
                recent: VecDeque::new(),
            }),
        }
    }
}
impl Telemetry {
    pub fn begin(self: &Arc<Self>, model: &str) -> RequestGuard {
        let mut r = self.records.lock().unwrap_or_else(|e| e.into_inner());
        r.totals.started += 1;
        r.totals.active += 1;
        let id = r.totals.started;
        let entry = RequestSnapshot {
            id,
            model: model.chars().take(256).collect(),
            state: "waiting / prefill".into(),
            started_ms: now_ms(),
            ..Default::default()
        };
        // Aggregate counters remain accurate even when the detail table is full.
        if r.active.len() < 128 {
            r.active.insert(id, entry.clone());
        }
        drop(r);
        let log_context = crate::logging::capture();
        crate::logging::emit_in(
            &log_context,
            crate::logging::Level::Debug,
            "inference.started",
            "Inference accepted",
            serde_json::json!({"inference_id":id,"model":entry.model}),
        );
        RequestGuard {
            log_context,
            telemetry: self.clone(),
            entry,
            start: Instant::now(),
            finished: false,
            first: false,
            raw_timings: RawTimings::default(),
        }
    }
    pub fn snapshot(&self) -> Snapshot {
        let r = self.records.lock().unwrap_or_else(|e| e.into_inner());
        let time = now_ms();
        let mut requests: Vec<_> = r
            .active
            .values()
            .cloned()
            .map(|mut e| {
                e.elapsed_seconds = time.saturating_sub(e.started_ms) as f64 / 1000.;
                e
            })
            .collect();
        requests.extend(r.recent.iter().rev().cloned());
        Snapshot {
            host_swap_used_bytes: None,
            host_memory_free_bytes: None,
            schema_version: 1,
            observed_at_ms: time,
            server_started_ms: self.started_ms,
            uptime_seconds: self.start.elapsed().as_secs_f64(),
            totals: r.totals.clone(),
            requests,
            backends: vec![],
            memory: None,
            backend_observed_at_ms: None,
        }
    }
}

// Retain only scalar metadata from native API responses, never response content.
#[derive(Default)]
struct RawTimings {
    usage_prompt_tokens: Option<u64>,
    usage_output_tokens: Option<u64>,
    usage_cached_tokens: Option<u64>,
    prompt_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    prompt_seconds: Option<f64>,
    decode_seconds: Option<f64>,
    first_token_seconds: Option<f64>,
}

pub struct RequestGuard {
    log_context: crate::logging::Context,
    telemetry: Arc<Telemetry>,
    entry: RequestSnapshot,
    start: Instant,
    finished: bool,
    first: bool,
    raw_timings: RawTimings,
}
impl RequestGuard {
    pub fn id(&self) -> u64 {
        self.entry.id
    }

    pub fn analysis_begin(&mut self, task: &str) {
        self.entry.analysis = Some(AnalysisSnapshot {
            task: task.into(),
            ..Default::default()
        });
        self.analysis_phase("preparing", None);
    }

    pub fn analysis_phase(&mut self, phase: &str, runtime: Option<(&str, &str)>) {
        if self.finished {
            return;
        }
        self.entry.state = phase.into();
        crate::logging::emit_in(
            &self.log_context,
            crate::logging::Level::Debug,
            "inference.phase",
            &format!("Inference #{} · {phase}", self.entry.id),
            serde_json::json!({"inference_id":self.entry.id,"model":self.entry.model,"phase":phase,"runtime":runtime.map(|v|v.0),"device":runtime.map(|v|v.1)}),
        );
        if let (Some(analysis), Some((runtime, device))) = (&mut self.entry.analysis, runtime) {
            analysis.runtime = Some(runtime.chars().take(64).collect());
            analysis.device = Some(device.chars().take(64).collect());
            analysis.attempts += 1;
        }
        if let Some(entry) = self
            .telemetry
            .records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active
            .get_mut(&self.entry.id)
        {
            *entry = self.entry.clone();
        }
    }

    pub fn analysis_complete(&mut self, value: &serde_json::Value) {
        if self.finished {
            return;
        }
        let usage = &value["usage"];
        self.entry.prompt_tokens = usage["input_tokens"]
            .as_u64()
            .or_else(|| usage["prompt_tokens"].as_u64())
            .or_else(|| usage["total_tokens"].as_u64());
        // Non-generative tasks have no output-token/decode measurements.
        self.entry.output_tokens = None;
        if let Some(analysis) = &mut self.entry.analysis {
            let werk = &value["werk"];
            let text = |key: &str| werk[key].as_str().map(|s| s.chars().take(64).collect());
            let seconds = |key: &str| werk[key].as_f64().filter(|s| s.is_finite() && *s >= 0.0);
            analysis.runtime = text("runtime").or(analysis.runtime.take());
            analysis.device = text("device").or(analysis.device.take());
            analysis.dtype = text("dtype");
            analysis.model_cache_hit = werk["model_cache_hit"].as_bool();
            analysis.load_seconds = seconds("load_seconds");
            analysis.inference_seconds = seconds("inference_seconds");
            analysis.worker_seconds = seconds("total_seconds");
            analysis.results = value["results"]
                .as_array()
                .or_else(|| value["data"].as_array())
                .map(|rows| rows.len() as u64)
                .or_else(|| {
                    value["answers"]
                        .as_object()
                        .map(|answers| answers.len() as u64)
                });
        }
        self.finish("done");
    }

    fn raw_metadata(&mut self, value: &serde_json::Value) {
        let usage = &value["usage"];
        let timings = &value["timings"];
        let raw = &mut self.raw_timings;
        raw.usage_prompt_tokens = usage["prompt_tokens"].as_u64().or(raw.usage_prompt_tokens);
        raw.usage_output_tokens = usage["completion_tokens"]
            .as_u64()
            .or(raw.usage_output_tokens);
        raw.usage_cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .or(raw.usage_cached_tokens);
        raw.prompt_tokens = timings["prompt_n"].as_u64().or(raw.prompt_tokens);
        raw.output_tokens = timings["predicted_n"].as_u64().or(raw.output_tokens);
        raw.cached_tokens = timings["cache_n"].as_u64().or(raw.cached_tokens);
        let seconds = |v: &serde_json::Value| v.as_f64().filter(|s| s.is_finite() && *s >= 0.);
        // llama.cpp reports milliseconds; oMLX extends usage with seconds.
        raw.prompt_seconds = seconds(&timings["prompt_ms"])
            .map(|ms| ms / 1000.)
            .or_else(|| seconds(&usage["prompt_eval_duration"]))
            .or(raw.prompt_seconds);
        raw.decode_seconds = seconds(&timings["predicted_ms"])
            .map(|ms| ms / 1000.)
            .or_else(|| seconds(&usage["generation_duration"]))
            .or(raw.decode_seconds);
        raw.first_token_seconds =
            seconds(&usage["time_to_first_token"]).or(raw.first_token_seconds);
    }
    pub fn raw_complete(&mut self, value: &serde_json::Value) {
        if self.finished {
            return;
        }
        self.raw_metadata(value);
        let raw = &self.raw_timings;
        self.entry.prompt_tokens = raw.usage_prompt_tokens;
        self.entry.output_tokens = raw.usage_output_tokens;
        self.entry.cached_tokens = raw.usage_cached_tokens.or(raw.cached_tokens);
        self.entry.decode_tokens_per_second = raw
            .output_tokens
            .or(self.entry.output_tokens)
            .zip(raw.decode_seconds)
            .and_then(|(n, s)| positive_rate(n, s));
        let evaluated = raw.prompt_tokens.or_else(|| {
            self.entry
                .prompt_tokens
                .map(|n| n.saturating_sub(self.entry.cached_tokens.unwrap_or(0)))
        });
        self.entry.prefill_tokens_per_second = evaluated
            .zip(raw.prompt_seconds)
            .and_then(|(n, s)| positive_rate(n, s));
        self.entry.first_output_seconds =
            raw.first_token_seconds.or(self.entry.first_output_seconds);
        let reason = value["choices"]
            .as_array()
            .and_then(|c| c.first())
            .and_then(|c| c["finish_reason"].as_str())
            .unwrap_or("done");
        self.finish(match reason {
            "tool_calls" => "tool_calls",
            "length" => "length",
            _ => "done",
        });
    }
    pub fn first_output(&mut self) {
        if self.first {
            return;
        }
        self.first = true;
        self.entry.first_output_seconds = Some(self.start.elapsed().as_secs_f64());
        self.entry.state = "streaming".into();
        if let Some(entry) = self
            .telemetry
            .records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active
            .get_mut(&self.entry.id)
        {
            *entry = self.entry.clone();
        }
    }
    pub fn complete(&mut self, response: &GenerateResponse) {
        self.done(
            &response.finish_reason,
            response.prompt_tokens as u64,
            response.completion_tokens as u64,
            response.timings,
        );
    }
    pub fn done(&mut self, reason: &str, prompt: u64, output: u64, timings: GenerationTimings) {
        if self.finished {
            return;
        }
        self.entry.prompt_tokens = Some(prompt);
        self.entry.output_tokens = Some(output);
        self.entry.cached_tokens = timings.cached_prompt_tokens.map(|n| n as u64);
        self.entry.decode_tokens_per_second = positive_rate(output, timings.decode_seconds);
        self.entry.prefill_tokens_per_second = positive_rate(
            prompt.saturating_sub(self.entry.cached_tokens.unwrap_or(0)),
            timings.prompt_seconds,
        );
        if timings.first_token_seconds > 0. && timings.first_token_seconds.is_finite() {
            self.entry.first_output_seconds = Some(timings.first_token_seconds);
        }
        self.finish(match reason {
            "tool_calls" => "tool_calls",
            "length" => "length",
            _ => "done",
        });
    }
    pub fn error(&mut self) {
        self.finish("error");
    }
    fn finish(&mut self, status: &str) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.entry.state = status.into();
        self.entry.elapsed_seconds = self.start.elapsed().as_secs_f64();
        let mut r = self
            .telemetry
            .records
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        r.active.remove(&self.entry.id);
        r.totals.active = r.totals.active.saturating_sub(1);
        match status {
            "error" => r.totals.errors += 1,
            "cancelled" => r.totals.cancelled += 1,
            _ => r.totals.completed += 1,
        }
        r.totals.prompt_tokens += self.entry.prompt_tokens.unwrap_or(0);
        r.totals.output_tokens += self.entry.output_tokens.unwrap_or(0);
        r.totals.cached_tokens += self.entry.cached_tokens.unwrap_or(0);
        r.totals.duration_seconds += self.entry.elapsed_seconds;
        if r.recent.len() == 64 {
            r.recent.pop_front();
        }
        r.recent.push_back(self.entry.clone());
        drop(r);
        crate::logging::emit_in(
            &self.log_context,
            match status {
                "error" => crate::logging::Level::Error,
                "cancelled" => crate::logging::Level::Warn,
                _ => crate::logging::Level::Info,
            },
            match status {
                "error" => "inference.failed",
                "cancelled" => "inference.cancelled",
                _ => "inference.completed",
            },
            &format!(
                "Inference #{} {status} · {} · {:.2} s",
                self.entry.id, self.entry.model, self.entry.elapsed_seconds
            ),
            serde_json::json!({
                "inference_id":self.entry.id,"model":self.entry.model,"state":status,
                "duration_seconds":self.entry.elapsed_seconds,"input_tokens":self.entry.prompt_tokens,
                "output_tokens":self.entry.output_tokens,"cached_tokens":self.entry.cached_tokens,
                "first_output_seconds":self.entry.first_output_seconds,
                "decode_tokens_per_second":self.entry.decode_tokens_per_second,
                "prefill_tokens_per_second":self.entry.prefill_tokens_per_second,"analysis":self.entry.analysis
            }),
        );
    }
}
impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.finish(if std::thread::panicking() {
            "error"
        } else {
            "cancelled"
        });
    }
}
fn positive_rate(tokens: u64, seconds: f64) -> Option<f64> {
    (seconds > 0. && seconds.is_finite())
        .then(|| tokens as f64 / seconds)
        .filter(|v| v.is_finite())
}

pub fn observe_stream(stream: GenerateStream, guard: RequestGuard) -> GenerateStream {
    Box::pin(ObservedStream {
        stream,
        guard,
        ended: false,
    })
}
struct ObservedStream {
    stream: GenerateStream,
    guard: RequestGuard,
    ended: bool,
}
impl tokio_stream::Stream for ObservedStream {
    type Item = Result<GenerateStreamEvent, String>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        if self.ended {
            return Poll::Ready(None);
        }
        match self.stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(event)) => {
                match &event {
                    Ok(GenerateStreamEvent::TextChunk(text)) if !text.is_empty() => {
                        self.guard.first_output()
                    }
                    Ok(GenerateStreamEvent::ToolCallDelta(delta)) if !delta.is_empty() => {
                        self.guard.first_output()
                    }
                    Ok(GenerateStreamEvent::Done {
                        finish_reason,
                        prompt_tokens,
                        completion_tokens,
                        timings,
                        ..
                    }) => self.guard.done(
                        finish_reason,
                        *prompt_tokens as u64,
                        *completion_tokens as u64,
                        *timings,
                    ),
                    Err(_) => self.guard.error(),
                    _ => {}
                }
                Poll::Ready(Some(event))
            }
            Poll::Ready(None) => {
                self.ended = true;
                if !self.guard.finished {
                    self.guard.error();
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

pub fn observe_raw_stream(
    stream: crate::backend::ApiGenerateStream,
    guard: RequestGuard,
    expected: usize,
) -> crate::backend::ApiGenerateStream {
    Box::pin(RawStream {
        stream,
        guard,
        expected,
        finished: std::collections::BTreeSet::new(),
        reason: String::new(),
        ended: false,
    })
}
struct RawStream {
    stream: crate::backend::ApiGenerateStream,
    guard: RequestGuard,
    expected: usize,
    finished: std::collections::BTreeSet<u64>,
    reason: String,
    ended: bool,
}
impl tokio_stream::Stream for RawStream {
    type Item = Result<serde_json::Value, String>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        if self.ended {
            return Poll::Ready(None);
        }
        match self.stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(value))) => {
                if value.get("error").is_some() {
                    self.guard.error();
                }
                self.guard.raw_metadata(&value);
                if let Some(choices) = value["choices"].as_array() {
                    for choice in choices {
                        if choice["delta"]["content"]
                            .as_str()
                            .is_some_and(|v| !v.is_empty())
                            || choice["delta"]["tool_calls"].is_array()
                        {
                            self.guard.first_output();
                        }
                        if let Some(reason) = choice["finish_reason"].as_str() {
                            if let Some(i) = choice["index"].as_u64() {
                                if i < self.expected as u64 {
                                    self.finished.insert(i);
                                    self.reason = reason.to_owned();
                                }
                            }
                        }
                    }
                }
                Poll::Ready(Some(Ok(value)))
            }
            Poll::Ready(Some(Err(e))) => {
                self.guard.error();
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                self.ended = true;
                if self.finished.len() == self.expected {
                    let raw = serde_json::json!({"choices":[{"finish_reason":self.reason}]});
                    self.guard.raw_complete(&raw);
                } else {
                    self.guard.error();
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Only derive interval rates within one worker generation; resets and missing
/// counters are gaps, never fabricated zeroes. Decode context growth is an
/// explicitly labelled estimate, valid only for a single uninterrupted request.
#[derive(Clone, Debug, Default)]
pub struct Rates {
    pub decode_estimate: Option<f64>,
    pub expert_hit_ratio: Option<f64>,
    pub read_bytes_per_second: Option<f64>,
}
impl Rates {
    pub fn between(old: &BackendSnapshot, new: &BackendSnapshot, seconds: f64) -> Self {
        if old.instance != new.instance || !old.available || !new.available || seconds <= 0. {
            return Self::default();
        }
        let delta = |name: &str| {
            new.counters
                .get(name)?
                .checked_sub(*old.counters.get(name)?)
        };
        let hits = delta("expert_cache_hits_total");
        let misses = delta("expert_cache_misses_total");
        let expert_hit_ratio = hits
            .zip(misses)
            .and_then(|(h, m)| (h + m > 0).then(|| h as f64 / (h + m) as f64));
        let same_request = old.counters.get("requests_completed_total").is_some()
            && old.counters.get("requests_completed_total")
                == new.counters.get("requests_completed_total")
            && old.gauges.get("requests_active") == Some(&1.)
            && new.gauges.get("requests_active") == Some(&1.);
        // llama.cpp exposes per-slot generation counts. Task IDs prevent a new
        // request (including prompt evaluation) from becoming a false decode spike.
        let slot_deltas: Vec<_> = new
            .gauges
            .iter()
            .filter_map(|(key, task)| {
                let slot = key.strip_prefix("slot_")?.strip_suffix("_task")?;
                if old.gauges.get(key) != Some(task) {
                    return None;
                }
                let key = format!("slot_{slot}_decoded");
                let (a, b) = (old.gauges.get(&key)?, new.gauges.get(&key)?);
                (a > &0. && b >= a).then_some((b - a) / seconds)
            })
            .collect();
        let decode_estimate = if !slot_deltas.is_empty() {
            Some(slot_deltas.iter().sum())
        } else if same_request {
            old.gauges
                .get("decode_context_tokens")
                .zip(new.gauges.get("decode_context_tokens"))
                .and_then(|(a, b)| (b > a).then(|| (b - a) / seconds))
        } else {
            None
        };
        Self {
            decode_estimate,
            expert_hit_ratio,
            read_bytes_per_second: delta("expert_read_bytes_total").map(|d| d as f64 / seconds),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analysis_requests_share_accounting_without_fabricating_generation_metrics() {
        let t = Arc::new(Telemetry::default());
        for (task, result, usage) in [
            (
                "text-classification",
                serde_json::json!({"answers":{"private-question":{"value":"private-answer"}}}),
                serde_json::json!({"input_tokens":47}),
            ),
            (
                "text-reranking",
                serde_json::json!({"results":[{"text":"private-document"},{"text":"private-document"}]}),
                serde_json::json!({"total_tokens":47}),
            ),
            (
                "text-embedding",
                serde_json::json!({"data":[{"embedding":[0.123]}]}),
                serde_json::json!({"prompt_tokens":47}),
            ),
        ] {
            let mut guard = t.begin(task);
            guard.analysis_begin(task);
            guard.analysis_phase("loading / inference", Some(("transformers", "cuda")));
            let live = t.snapshot();
            assert_eq!(live.totals.active, 1);
            assert_eq!(live.requests[0].state, "loading / inference");
            assert_eq!(live.requests[0].analysis.as_ref().unwrap().attempts, 1);
            assert_eq!(
                live.requests[0]
                    .analysis
                    .as_ref()
                    .unwrap()
                    .inference_seconds,
                None
            );
            let mut response = result;
            response["usage"] = usage;
            response["werk"] = serde_json::json!({"runtime":"transformers","device":"cuda","dtype":"torch.bfloat16","model_cache_hit":true,"load_seconds":0.1,"inference_seconds":0.4,"total_seconds":0.5,"diagnostics":["private-diagnostic"]});
            guard.analysis_complete(&response);
            guard.analysis_complete(&response); // Completion is idempotent.
            let snapshot = t.snapshot();
            let request = &snapshot.requests[0];
            assert_eq!(request.state, "done");
            assert_eq!(request.prompt_tokens, Some(47));
            assert!(request.output_tokens.is_none());
            assert!(request.cached_tokens.is_none());
            assert!(request.first_output_seconds.is_none());
            assert!(request.decode_tokens_per_second.is_none());
            assert!(request.prefill_tokens_per_second.is_none());
            let analysis = request.analysis.as_ref().unwrap();
            assert_eq!(
                analysis.results,
                Some(if task == "text-reranking" { 2 } else { 1 })
            );
            assert_eq!(analysis.inference_seconds, Some(0.4));
            assert_eq!(analysis.model_cache_hit, Some(true));
            assert!(
                !serde_json::to_string(&snapshot)
                    .unwrap()
                    .contains("private-")
            );
        }
        let snapshot = t.snapshot();
        assert_eq!(snapshot.totals.active, 0);
        assert_eq!(snapshot.totals.completed, 3);
        assert_eq!(snapshot.totals.prompt_tokens, 141);
        assert_eq!(snapshot.totals.output_tokens, 0);
        assert_eq!(snapshot.totals.cached_tokens, 0);
    }

    #[test]
    fn analysis_failure_cancellation_and_old_snapshots_keep_the_shared_contract() {
        let t = Arc::new(Telemetry::default());
        let mut guard = t.begin("fixture");
        guard.analysis_begin("text-classification");
        guard.error();
        drop(guard);
        let mut guard = t.begin("fixture");
        guard.analysis_begin("text-embedding");
        drop(guard);
        let snapshot = t.snapshot();
        assert_eq!(snapshot.totals.errors, 1);
        assert_eq!(snapshot.totals.cancelled, 1);
        assert_eq!(snapshot.totals.active, 0);
        assert_eq!(snapshot.totals.completed, 0);
        let mut legacy = serde_json::to_value(&snapshot.requests[0]).unwrap();
        legacy.as_object_mut().unwrap().remove("analysis");
        assert!(
            serde_json::from_value::<RequestSnapshot>(legacy)
                .unwrap()
                .analysis
                .is_none()
        );
    }

    #[test]
    fn raw_completion_preserves_backend_rates() {
        let t = Arc::new(Telemetry::default());
        t.begin("mock").raw_complete(&serde_json::json!({
            "choices": [{"finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30,
                      "prompt_tokens_details": {"cached_tokens": 80}},
            "timings": {"prompt_n": 20, "prompt_ms": 500,
                        "predicted_n": 30, "predicted_ms": 2000}
        }));
        let s = t.snapshot();
        let r = &s.requests[0];
        assert_eq!(r.state, "tool_calls");
        assert_eq!(r.decode_tokens_per_second, Some(15.));
        assert_eq!(r.prefill_tokens_per_second, Some(40.));
        assert_eq!(r.cached_tokens, Some(80));
        assert_eq!(s.totals.output_tokens, 30);
    }

    #[tokio::test]
    async fn raw_stream_retains_timings_across_separate_usage_and_finish_chunks() {
        // Timings may arrive before or after the separate usage chunk.
        for timings_first in [false, true] {
            let t = Arc::new(Telemetry::default());
            let timing = serde_json::json!({"choices": [], "timings": {
                "prompt_n": 20, "prompt_ms": 500, "cache_n": 80,
                "predicted_n": 30, "predicted_ms": 2000
            }});
            let usage = serde_json::json!({"choices": [], "usage": {
                "prompt_tokens": 100, "completion_tokens": 30
            }});
            let mut items = vec![serde_json::json!({"choices": [{
                "index": 0, "delta": {"tool_calls": []}, "finish_reason": "tool_calls"
            }]})];
            items.extend(if timings_first {
                [timing, usage]
            } else {
                [usage, timing]
            });
            let output: Vec<_> = observe_raw_stream(
                Box::pin(tokio_stream::iter(
                    items.iter().cloned().map(Ok).collect::<Vec<_>>(),
                )),
                t.begin("mock"),
                1,
            )
            .collect()
            .await;
            assert_eq!(
                output.into_iter().map(Result::unwrap).collect::<Vec<_>>(),
                items
            );
            let s = t.snapshot();
            let r = &s.requests[0];
            assert_eq!(r.state, "tool_calls");
            assert_eq!(r.decode_tokens_per_second, Some(15.));
            assert_eq!(r.prefill_tokens_per_second, Some(40.));
            assert_eq!(r.cached_tokens, Some(80));
            assert_eq!(s.totals.completed, 1);
            assert_eq!(s.totals.output_tokens, 30);
        }
    }

    #[tokio::test]
    async fn raw_omlx_usage_timings_survive_partial_and_invalid_usage_chunks() {
        let t = Arc::new(Telemetry::default());
        let values = vec![
            serde_json::json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30,
                    "prompt_tokens_details": {"cached_tokens": 80},
                    "time_to_first_token": 0.75, "prompt_eval_duration": 0.5,
                    "generation_duration": 2.0}}),
            serde_json::json!({"choices": [], "usage": {
                "generation_duration": -1, "prompt_eval_duration": "invalid",
                "time_to_first_token": null}}),
        ];
        let _: Vec<_> = observe_raw_stream(
            Box::pin(tokio_stream::iter(values.into_iter().map(Ok))),
            t.begin("omlx"),
            1,
        )
        .collect()
        .await;
        let s = t.snapshot();
        let r = &s.requests[0];
        assert_eq!(r.decode_tokens_per_second, Some(15.));
        assert_eq!(r.prefill_tokens_per_second, Some(40.));
        assert_eq!(r.first_output_seconds, Some(0.75));
        assert_eq!(r.cached_tokens, Some(80));
        assert_eq!(s.totals.output_tokens, 30);
    }

    #[tokio::test]
    async fn typed_backend_timings_reach_both_completed_request_paths() {
        let t = Arc::new(Telemetry::default());
        let response = GenerateResponse {
            text: "private".into(),
            assistant_message: None,
            prompt_tokens: 100,
            completion_tokens: 30,
            finish_reason: "stop".into(),
            timings: GenerationTimings {
                cached_prompt_tokens: Some(80),
                prompt_seconds: 0.5,
                decode_seconds: 2.,
                first_token_seconds: 0.75,
                ..Default::default()
            },
            backend_diagnostics: vec![],
        };
        t.begin("typed").complete(&response);
        let _: Vec<_> = observe_stream(
            Box::pin(tokio_stream::iter(vec![Ok(GenerateStreamEvent::Done {
                finish_reason: response.finish_reason,
                prompt_tokens: response.prompt_tokens,
                completion_tokens: response.completion_tokens,
                timings: response.timings,
                backend_diagnostics: vec![],
            })])),
            t.begin("typed"),
        )
        .collect()
        .await;
        let s = t.snapshot();
        assert_eq!(s.totals.completed, 2);
        for r in s.requests {
            assert_eq!(r.decode_tokens_per_second, Some(15.));
            assert_eq!(r.prefill_tokens_per_second, Some(40.));
            assert_eq!(r.first_output_seconds, Some(0.75));
        }
    }

    #[test]
    fn raw_completion_without_valid_timings_keeps_rates_unavailable() {
        for ms in [
            serde_json::Value::Null,
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!("invalid"),
        ] {
            let t = Arc::new(Telemetry::default());
            t.begin("mock").raw_complete(&serde_json::json!({
                "choices": [{"finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                "timings": {"prompt_ms": ms, "predicted_ms": ms}
            }));
            let s = t.snapshot();
            assert_eq!(s.requests[0].decode_tokens_per_second, None);
            assert_eq!(s.requests[0].prefill_tokens_per_second, None);
            assert_eq!(s.totals.output_tokens, 30);
        }
    }

    #[tokio::test]
    async fn cancelled_stream_and_premature_eof_are_distinct() {
        let t = Arc::new(Telemetry::default());
        let stream = observe_stream(Box::pin(tokio_stream::pending()), t.begin("mock"));
        drop(stream);
        let _: Vec<_> = observe_stream(Box::pin(tokio_stream::empty()), t.begin("mock"))
            .collect()
            .await;
        let s = t.snapshot();
        assert_eq!(s.totals.cancelled, 1);
        assert_eq!(s.totals.errors, 1);
        assert_eq!(s.totals.active, 0);
    }
    #[tokio::test]
    async fn rich_stream_finishes_after_usage_and_reports_incomplete_choices() {
        let t = Arc::new(Telemetry::default());
        let items = vec![
            Ok(
                serde_json::json!({"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}),
            ),
            Ok(
                serde_json::json!({"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3}}),
            ),
        ];
        let _: Vec<_> = observe_raw_stream(
            Box::pin(tokio_stream::iter(items.clone())),
            t.begin("mock"),
            1,
        )
        .collect()
        .await;
        let _: Vec<_> = observe_raw_stream(Box::pin(tokio_stream::iter(items)), t.begin("mock"), 2)
            .collect()
            .await;
        let s = t.snapshot();
        assert_eq!(s.totals.completed, 1);
        assert_eq!(s.totals.errors, 1);
        assert_eq!(s.totals.output_tokens, 3);
    }
    #[test]
    fn records_are_bounded_and_cancellation_is_accounted() {
        let t = Arc::new(Telemetry::default());
        for _ in 0..100 {
            drop(t.begin("model"));
        }
        let s = t.snapshot();
        assert_eq!(s.requests.len(), 64);
        assert_eq!(s.totals.cancelled, 100);
        assert_eq!(s.totals.active, 0);
    }
    #[tokio::test]
    async fn stream_counts_tokens_from_usage_not_chunks() {
        let t = Arc::new(Telemetry::default());
        let events = vec![
            Ok(GenerateStreamEvent::TextChunk(
                "many tokens in one chunk".into(),
            )),
            Ok(GenerateStreamEvent::Done {
                finish_reason: "stop".into(),
                prompt_tokens: 20,
                completion_tokens: 6,
                timings: GenerationTimings::default(),
                backend_diagnostics: vec![],
            }),
        ];
        let _: Vec<_> = observe_stream(Box::pin(tokio_stream::iter(events)), t.begin("model"))
            .collect()
            .await;
        assert_eq!(t.snapshot().totals.output_tokens, 6);
        assert_eq!(t.snapshot().totals.completed, 1);
    }
    #[test]
    fn observability_llama_decode_uses_task_identity_and_excludes_prefill() {
        let mut a = BackendSnapshot {
            available: true,
            instance: "worker".into(),
            ..Default::default()
        };
        a.gauges
            .extend([("slot_0_task".into(), 42.), ("slot_0_decoded".into(), 100.)]);
        let mut b = a.clone();
        b.gauges.insert("slot_0_decoded".into(), 124.);
        assert_eq!(Rates::between(&a, &b, 2.).decode_estimate, Some(12.));
        b.gauges.insert("slot_0_task".into(), 43.);
        assert!(Rates::between(&a, &b, 2.).decode_estimate.is_none());
        b = a.clone();
        b.gauges.insert("slot_0_decoded".into(), 2.);
        assert!(Rates::between(&a, &b, 2.).decode_estimate.is_none());
        a.gauges.insert("slot_0_decoded".into(), 0.);
        assert!(Rates::between(&a, &b, 2.).decode_estimate.is_none());
        b.gauges.clear(); // Idle slots must not reuse the last request's speed.
        assert!(Rates::between(&a, &b, 2.).decode_estimate.is_none());
    }

    #[test]
    fn interval_resets_and_idle_are_not_hits() {
        let mut a = BackendSnapshot {
            available: true,
            instance: "a".into(),
            ..Default::default()
        };
        a.counters.insert("expert_cache_hits_total".into(), 100);
        a.counters.insert("expert_cache_misses_total".into(), 10);
        assert!(Rates::between(&a, &a, 1.).expert_hit_ratio.is_none());
        let mut b = a.clone();
        b.counters.insert("expert_cache_hits_total".into(), 1);
        assert!(Rates::between(&a, &b, 1.).expert_hit_ratio.is_none());
        b = a.clone();
        b.counters.insert("expert_cache_hits_total".into(), 109);
        b.counters.insert("expert_cache_misses_total".into(), 11);
        assert_eq!(Rates::between(&a, &b, 1.).expert_hit_ratio, Some(0.9));
    }
}
