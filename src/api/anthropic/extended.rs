use super::{response, stream::MessagesStream};
use crate::{
    api::generation::Prepared,
    backend::{
        ApiGenerateStream, GenerateResponse, GenerateStreamEvent, GeneratedAssistantMessage,
    },
};
use axum::{
    Json,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{KeepAlive, Sse},
    },
};
use serde_json::Value;
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};
use tokio_stream::Stream;

pub(super) async fn handle(
    prepared: Prepared,
    id: &str,
    request_id: &str,
    tool_names: Vec<String>,
) -> Response {
    let model = prepared.manifest.id.clone();
    let matched_requested = prepared.api_options.contains_key("__werk_matched_stop");
    if prepared.stream {
        let matched = Arc::new(Mutex::new(None));
        let bridge = Bridge {
            source: Some(crate::api::extended::raw_stream(prepared)),
            queue: VecDeque::new(),
            finish: None,
            usage: None,
            matched: matched.clone(),
            matched_requested,
        };
        let mut stream = MessagesStream::new(Box::pin(bridge), &model, id, request_id);
        stream.tool_names = tool_names;
        stream.matched_stop = Some(matched);
        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match tokio::task::spawn_blocking(move || {
            prepared.state.backend.generate_api(
                &prepared.manifest,
                prepared.request,
                prepared.api_options,
                None,
            )
        })
        .await
        {
            Ok(Ok(raw)) => match convert(raw, matched_requested).and_then(|(generated, matched)| {
                response::message_with_stop(id, &model, generated, matched)
            }) {
                Ok(message) => Json(message).into_response(),
                Err(e) => response::error(StatusCode::BAD_GATEWAY, e, request_id),
            },
            Ok(Err(e)) => response::error(StatusCode::BAD_REQUEST, e.to_string(), request_id),
            Err(_) => response::error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "extended generation task failed",
                request_id,
            ),
        }
    }
}
fn usage(raw: &Value) -> Result<(usize, usize), String> {
    Ok((
        raw["prompt_tokens"]
            .as_u64()
            .ok_or("native API omitted prompt usage")? as usize,
        raw["completion_tokens"]
            .as_u64()
            .ok_or("native API omitted output usage")? as usize,
    ))
}
fn convert(
    raw: Value,
    matched_requested: bool,
) -> Result<(GenerateResponse, Option<String>), String> {
    let choices = raw["choices"]
        .as_array()
        .filter(|c| c.len() == 1)
        .ok_or("native API must return exactly one choice")?;
    let choice = &choices[0];
    let (prompt_tokens, completion_tokens) = usage(&raw["usage"])?;
    let message = &choice["message"];
    let text = message["content"].as_str().map(str::to_owned);
    let tool_calls = if message["tool_calls"].is_null() {
        None
    } else {
        Some(
            serde_json::from_value(message["tool_calls"].clone())
                .map_err(|_| "native API returned invalid tool calls")?,
        )
    };
    Ok((
        GenerateResponse {
            text: text.clone().unwrap_or_default(),
            assistant_message: Some(GeneratedAssistantMessage {
                content: text,
                tool_calls,
            }),
            prompt_tokens,
            completion_tokens,
            finish_reason: choice["finish_reason"]
                .as_str()
                .ok_or("native API omitted finish reason")?
                .into(),
            timings: Default::default(),
            backend_diagnostics: vec![],
        },
        if matched_requested {
            choice["stop_reason"].as_str().map(str::to_owned)
        } else {
            None
        },
    ))
}
struct Bridge {
    source: Option<ApiGenerateStream>,
    queue: VecDeque<GenerateStreamEvent>,
    finish: Option<String>,
    usage: Option<(usize, usize)>,
    matched: Arc<Mutex<Option<String>>>,
    matched_requested: bool,
}
impl Bridge {
    fn event(&mut self, value: Value) -> Result<(), String> {
        if value["usage"].is_object() {
            self.usage = Some(usage(&value["usage"])?);
        }
        let choices = value["choices"]
            .as_array()
            .ok_or("native API omitted choices")?;
        if choices.len() > 1 {
            return Err("Anthropic response requires one choice".into());
        }
        if let Some(choice) = choices.first() {
            if choice["index"].as_u64() != Some(0) {
                return Err("native API returned invalid choice index".into());
            }
            if let Some(text) = choice["delta"]["content"]
                .as_str()
                .filter(|s| !s.is_empty())
            {
                self.queue
                    .push_back(GenerateStreamEvent::TextChunk(text.into()));
            }
            if choice["delta"]["tool_calls"].is_array() {
                self.queue.push_back(GenerateStreamEvent::ToolCallDelta(
                    serde_json::from_value(choice["delta"]["tool_calls"].clone())
                        .map_err(|_| "invalid native tool deltas")?,
                ));
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                self.finish = Some(reason.into());
                if self.matched_requested {
                    *self
                        .matched
                        .lock()
                        .map_err(|_| "stop metadata unavailable")? =
                        choice["stop_reason"].as_str().map(str::to_owned);
                }
            }
        }
        Ok(())
    }
}
impl Stream for Bridge {
    type Item = Result<GenerateStreamEvent, String>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        for _ in 0..32 {
            if let Some(event) = self.queue.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            let Some(source) = self.source.as_mut() else {
                return Poll::Ready(None);
            };
            match source.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(v))) => {
                    if let Err(e) = self.event(v) {
                        self.source = None;
                        return Poll::Ready(Some(Err(e)));
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    self.source = None;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    self.source = None;
                    return Poll::Ready(Some(match (self.finish.take(), self.usage) {
                        (Some(finish_reason), Some((prompt_tokens, completion_tokens))) => {
                            Ok(GenerateStreamEvent::Done {
                                finish_reason,
                                prompt_tokens,
                                completion_tokens,
                                timings: Default::default(),
                                backend_diagnostics: vec![],
                            })
                        }
                        _ => {
                            Err("native stream ended without final usage and finish reason".into())
                        }
                    }));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}
