//! Pull-driven SSE: dropping the body drops the backend stream; no detached pump.
use super::{
    request::valid_name,
    response::{error_value, stop_reason, validate_finish},
};
use crate::{
    backend::{GenerateStream, GenerateStreamEvent},
    openai::ChatCompletionToolCallDelta,
};
use axum::response::sse::Event;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    convert::Infallible,
    pin::Pin,
    task::{Context, Poll},
};
use tokio_stream::Stream;

const MAX_TOOL_BYTES: usize = 8 * 1024 * 1024;
const MAX_TOOLS: usize = 128;

#[derive(Default)]
struct Tool {
    id: String,
    name: String,
    arguments: String,
}

pub(super) struct MessagesStream {
    pub verbose: bool,
    model: String,
    source: Option<GenerateStream>,
    queue: VecDeque<Event>,
    request_id: String,
    tools: BTreeMap<usize, Tool>,
    bytes: usize,
    text_started: bool,
    tool_started: bool,
}

impl MessagesStream {
    pub fn new(source: GenerateStream, model: &str, id: &str, request_id: &str) -> Self {
        let mut stream = Self {
            verbose: false,
            model: model.into(),
            source: Some(source),
            queue: VecDeque::new(),
            request_id: request_id.into(),
            tools: BTreeMap::new(),
            bytes: 0,
            text_started: false,
            tool_started: false,
        };
        stream.push(
            "message_start",
            json!({"message":{
                "id":id,"type":"message","role":"assistant","model":model,"content":[],
                "stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}
            }}),
        );
        stream
    }
    fn push(&mut self, kind: &str, mut value: Value) {
        value["type"] = kind.into();
        self.queue
            .push_back(Event::default().event(kind).data(value.to_string()));
    }
    fn fail(&mut self, message: String) {
        self.source = None;
        self.tools.clear();
        self.queue.clear();
        self.queue.push_back(
            Event::default()
                .event("error")
                .data(error_value("api_error", message, &self.request_id).to_string()),
        );
    }
    fn tool_delta(&mut self, delta: ChatCompletionToolCallDelta) -> Result<(), String> {
        if delta.kind.as_deref().is_some_and(|kind| kind != "function") {
            return Err("unsupported backend tool type".into());
        }
        if !self.tools.contains_key(&delta.index) && self.tools.len() >= MAX_TOOLS {
            return Err("too many tool calls in one response".into());
        }
        let name = delta
            .function
            .as_ref()
            .and_then(|f| f.name.as_deref())
            .unwrap_or_default();
        let args = delta
            .function
            .as_ref()
            .and_then(|f| f.arguments.as_deref())
            .unwrap_or_default();
        let id = delta.id.as_deref().unwrap_or_default();
        self.bytes = self
            .bytes
            .saturating_add(id.len())
            .saturating_add(name.len())
            .saturating_add(args.len());
        if self.bytes > MAX_TOOL_BYTES {
            return Err("tool response exceeds the 8 MiB stream buffer limit".into());
        }
        self.tool_started = true;
        let tool = self.tools.entry(delta.index).or_default();
        tool.id.push_str(id);
        tool.name.push_str(name);
        tool.arguments.push_str(args);
        if tool.id.len() > 128 || tool.name.len() > 128 {
            return Err("backend tool identity is too long".into());
        }
        Ok(())
    }
    fn event(&mut self, event: GenerateStreamEvent) -> Result<(), String> {
        match event {
            GenerateStreamEvent::TextChunk(text) => {
                if text.is_empty() {
                    return Ok(());
                }
                if self.tool_started {
                    return Err(
                        "backend text after tool deltas cannot be represented without reordering"
                            .into(),
                    );
                }
                if !self.text_started {
                    self.push(
                        "content_block_start",
                        json!({"index":0,"content_block":{"type":"text","text":""}}),
                    );
                    self.text_started = true;
                }
                self.push(
                    "content_block_delta",
                    json!({"index":0,"delta":{"type":"text_delta","text":text}}),
                );
            }
            GenerateStreamEvent::ToolCallDelta(deltas) => {
                for delta in deltas {
                    self.tool_delta(delta)?;
                }
            }
            GenerateStreamEvent::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
                timings,
                backend_diagnostics,
            } => {
                if self.verbose {
                    super::log_completion(
                        &self.model,
                        &finish_reason,
                        prompt_tokens,
                        completion_tokens,
                        timings,
                        &backend_diagnostics,
                    );
                }
                let stop = stop_reason(&finish_reason)?;
                validate_finish(stop, !self.tools.is_empty())?;
                let mut ids = HashSet::new();
                for tool in self.tools.values() {
                    if !valid_name(&tool.id) || !valid_name(&tool.name) || !ids.insert(&tool.id) {
                        return Err("backend returned an invalid or duplicate tool identity".into());
                    }
                    // A token limit can cut JSON mid-value. Keep the raw fragment and
                    // max_tokens stop; the client must not execute an incomplete call.
                    if stop != "max_tokens"
                        && !serde_json::from_str::<Value>(&tool.arguments)
                            .is_ok_and(|v| v.is_object())
                    {
                        return Err("backend returned incomplete or invalid tool JSON".into());
                    }
                }
                if self.text_started {
                    self.push("content_block_stop", json!({"index":0}));
                }
                // Tool IDs/names can be fragmented with no header-complete signal.
                // Buffer tools until Done so the immutable Anthropic block header is
                // correct. Ordinary text above is emitted immediately. Storage is bounded.
                let tools = std::mem::take(&mut self.tools);
                for (offset, tool) in tools.into_values().enumerate() {
                    let index = offset + usize::from(self.text_started);
                    self.push(
                        "content_block_start",
                        json!({"index":index,"content_block":{
                        "type":"tool_use","id":tool.id,"name":tool.name,"input":{}}}),
                    );
                    self.push("content_block_delta", json!({"index":index,"delta":{"type":"input_json_delta","partial_json":tool.arguments}}));
                    self.push("content_block_stop", json!({"index":index}));
                }
                if !self.text_started && !self.tool_started {
                    self.push(
                        "content_block_start",
                        json!({"index":0,"content_block":{"type":"text","text":""}}),
                    );
                    self.push("content_block_stop", json!({"index":0}));
                }
                self.push(
                    "message_delta",
                    json!({"delta":{"stop_reason":stop,"stop_sequence":null},
                    "usage":{"input_tokens":prompt_tokens,"output_tokens":completion_tokens}}),
                );
                self.push("message_stop", json!({}));
                self.source = None;
            }
        }
        Ok(())
    }
}

impl Stream for MessagesStream {
    type Item = Result<Event, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Bound work per poll even when the backend immediately yields only tool fragments.
        for _ in 0..32 {
            if let Some(event) = self.queue.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            let Some(source) = self.source.as_mut() else {
                return Poll::Ready(None);
            };
            match source.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(event))) => {
                    if let Err(error) = self.event(event) {
                        self.fail(error);
                    }
                }
                Poll::Ready(Some(Err(error))) => self.fail(error),
                Poll::Ready(None) => {
                    self.fail("backend stream ended without a completion event".into())
                }
            }
        }
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    use tokio_stream::StreamExt;

    struct Probe {
        polls: Arc<AtomicUsize>,
        dropped: Arc<AtomicBool>,
    }
    impl Drop for Probe {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }
    impl Stream for Probe {
        type Item = Result<GenerateStreamEvent, String>;
        fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(Some(Ok(GenerateStreamEvent::TextChunk("Grüße 🌍".into()))))
        }
    }
    #[tokio::test]
    async fn text_is_pull_driven_and_cancellation_drops_source_without_background_work() {
        let polls = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        let source = Box::pin(Probe {
            polls: polls.clone(),
            dropped: dropped.clone(),
        });
        let mut stream = MessagesStream::new(source, "mock", "msg_test", "req_test");
        let _ = stream.next().await.unwrap(); // message_start does not prefetch
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        let _ = stream.next().await.unwrap(); // content_block_start
        let _ = stream.next().await.unwrap(); // queued Unicode delta; no extra backend read
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        drop(stream);
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(polls.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn limits_tool_count_even_with_empty_argument_fragments() {
        let mut stream = MessagesStream::new(
            Box::pin(tokio_stream::empty()),
            "mock",
            "msg_test",
            "req_test",
        );
        for index in 0..MAX_TOOLS {
            stream
                .tool_delta(ChatCompletionToolCallDelta {
                    index,
                    id: None,
                    kind: None,
                    function: None,
                })
                .unwrap();
        }
        assert!(
            stream
                .tool_delta(ChatCompletionToolCallDelta {
                    index: MAX_TOOLS,
                    id: None,
                    kind: None,
                    function: None
                })
                .is_err()
        );
    }
}
