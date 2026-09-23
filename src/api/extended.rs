//! Rich API fields use a separate opt-in transport; ordinary chats retain their path.
use super::{generation::Prepared, response::api_error};
use axum::{
    Json,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, convert::Infallible};
use tokio_stream::StreamExt;

pub(super) fn validate(options: &BTreeMap<String, Value>, anthropic: bool) -> Result<(), String> {
    for (key, v) in options {
        if v.is_null()
            && matches!(
                key.as_str(),
                "frequency_penalty"
                    | "presence_penalty"
                    | "n"
                    | "logprobs"
                    | "top_logprobs"
                    | "logit_bias"
                    | "reasoning_effort"
                    | "response_format"
                    | "store"
            )
        {
            continue;
        }
        let valid = match key.as_str() {
            "__werk_matched_stop" => anthropic && v.is_array(),
            "frequency_penalty" | "presence_penalty" => {
                v.as_f64().is_some_and(|n| (-2.0..=2.0).contains(&n))
            }
            "n" => !anthropic && v.as_u64().is_some_and(|n| (1..=16).contains(&n)),
            "logprobs" => !anthropic && v.is_boolean(),
            "top_logprobs" => {
                !anthropic
                    && v.as_u64().is_some_and(|n| n <= 20)
                    && options.get("logprobs") == Some(&json!(true))
            }
            "logit_bias" => {
                !anthropic
                    && v.as_object().is_some_and(|m| {
                        m.len() <= 65536
                            && m.iter().all(|(k, v)| {
                                k.parse::<u32>().is_ok()
                                    && v.as_f64().is_some_and(|n| (-100.0..=100.0).contains(&n))
                            })
                    })
            }
            "reasoning_effort" => {
                !anthropic
                    && v.as_str().is_some_and(|s| {
                        matches!(s, "none" | "minimal" | "low" | "medium" | "high" | "xhigh")
                    })
            }
            "top_k" => anthropic && v.as_u64().is_some_and(|n| n <= u32::MAX as u64),
            "response_format" => valid_format(v),
            "store" => !anthropic && v == &json!(false),
            _ => return Err(format!("unsupported request field: {key}")),
        };
        if !valid {
            return Err(format!("unsupported or invalid value for {key}"));
        }
    }
    Ok(())
}
fn valid_format(v: &Value) -> bool {
    let Some(m) = v.as_object() else {
        return false;
    };
    match m.get("type").and_then(Value::as_str) {
        Some("text" | "json_object") => m.len() == 1,
        Some("json_schema") => {
            m.len() == 2
                && m.get("json_schema")
                    .and_then(Value::as_object)
                    .is_some_and(|s| {
                        s.keys().all(|k| {
                            matches!(k.as_str(), "name" | "description" | "schema" | "strict")
                        }) && s.get("name").and_then(Value::as_str).is_some_and(|n| {
                            !n.is_empty()
                                && n.len() <= 64
                                && n.bytes()
                                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                        }) && s.get("schema").is_some_and(Value::is_object)
                            && s.get("strict")
                                .is_none_or(|v| v.is_boolean() || v.is_null())
                            && s.get("description").is_none_or(Value::is_string)
                    })
        }
        _ => false,
    }
}

pub(super) fn normalize(options: &mut BTreeMap<String, Value>) {
    options.retain(|key, v| {
        !v.is_null()
            && !matches!(key.as_str(), "store")
            && !(key == "n" && v == &json!(1))
            && !(key == "logprobs" && v == &json!(false))
            && !(key == "response_format" && v == &json!({"type":"text"}))
    });
}

pub(super) async fn openai(prepared: Prepared) -> Response {
    let model = prepared.manifest.id.clone();
    let n = prepared
        .api_options
        .get("n")
        .and_then(Value::as_u64)
        .unwrap_or(1) as usize;
    if prepared.stream {
        let include_usage = prepared.include_usage;
        let source = raw_stream(prepared);
        let mut failed = false;
        let mut finished = std::collections::HashSet::new();
        let stream = source.filter_map(move |event| {
            let data = match event {
                Ok(mut chunk) => {
                    if !include_usage {
                        if chunk["choices"].as_array().is_some_and(Vec::is_empty)
                            && chunk["usage"].is_object()
                        {
                            return None;
                        }
                        if let Some(object) = chunk.as_object_mut() {
                            object.remove("usage");
                        }
                    }
                    chunk["model"] = model.clone().into();
                    if let Some(choices) = chunk["choices"].as_array() {
                        for c in choices {
                            if c["finish_reason"].is_string() {
                                if let Some(index) = c["index"].as_u64() {
                                    finished.insert(index);
                                }
                            }
                        }
                    }
                    chunk.to_string()
                }
                Err(error) => {
                    failed = true;
                    json!({"error":{"type":"server_error","message":error}}).to_string()
                }
            };
            Some((Event::default().data(data), failed, finished.len()))
        });
        // Track terminal success in the pull-driven adapter rather than append
        // [DONE] unconditionally after an error or premature upstream EOF.
        let output = OpenAiStream {
            source: Box::pin(stream),
            failed: false,
            finished: 0,
            expected: n,
            ended: false,
        };
        Sse::new(output)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match tokio::task::spawn_blocking(move || raw_generate(prepared)).await {
            Ok(Ok(mut value)) => {
                if value["choices"].as_array().is_none_or(|c| c.len() != n) {
                    return api_error(
                        StatusCode::BAD_GATEWAY,
                        "backend returned the wrong number of choices".into(),
                        None,
                    );
                }
                value["model"] = model.into();
                Json(value).into_response()
            }
            Ok(Err(e)) => api_error(StatusCode::BAD_REQUEST, e.to_string(), None),
            Err(_) => api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "extended generation task failed".into(),
                None,
            ),
        }
    }
}

pub(super) fn raw_stream(prepared: Prepared) -> crate::backend::ApiGenerateStream {
    let guard = prepared.state.telemetry.begin(&prepared.manifest.id);
    let expected = prepared
        .api_options
        .get("n")
        .and_then(Value::as_u64)
        .unwrap_or(1) as usize;
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    tokio::task::spawn_blocking(move || {
        if tx.is_closed() {
            return;
        }
        if let Err(e) = prepared.state.backend.generate_api(
            &prepared.manifest,
            prepared.request,
            prepared.api_options,
            Some(tx.clone()),
        ) {
            let _ = tx.blocking_send(Err(e.to_string()));
        }
    });
    crate::observability::observe_raw_stream(
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)),
        guard,
        expected,
    )
}
pub(super) fn raw_generate(prepared: Prepared) -> anyhow::Result<Value> {
    let mut guard = prepared.state.telemetry.begin(&prepared.manifest.id);
    let result = prepared.state.backend.generate_api(
        &prepared.manifest,
        prepared.request,
        prepared.api_options,
        None,
    );
    match &result {
        Ok(value) => guard.raw_complete(value),
        Err(_) => guard.error(),
    }
    result
}
struct OpenAiStream {
    source: std::pin::Pin<Box<dyn tokio_stream::Stream<Item = (Event, bool, usize)> + Send>>,
    failed: bool,
    finished: usize,
    expected: usize,
    ended: bool,
}
impl tokio_stream::Stream for OpenAiStream {
    type Item = Result<Event, Infallible>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        if self.ended {
            return Poll::Ready(None);
        }
        match self.source.as_mut().poll_next(cx) {
            Poll::Ready(Some((event, failed, finished))) => {
                self.failed = failed;
                self.finished = finished;
                Poll::Ready(Some(Ok(event)))
            }
            Poll::Ready(None) => {
                self.ended = true;
                if self.failed {
                    Poll::Ready(None)
                } else if self.finished == self.expected {
                    Poll::Ready(Some(Ok(Event::default().data("[DONE]"))))
                } else {
                    Poll::Ready(Some(Ok(Event::default().data(json!({"error":{"type":"server_error","message":"backend ended without completing every choice"}}).to_string()))))
                }
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
