//! Assemble native OpenAI streaming tool calls while forwarding original deltas.
use crate::openai::{
    ChatCompletionFunctionCall, ChatCompletionToolCall, ChatCompletionToolCallDelta,
};
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct ToolCallAccumulator {
    calls: BTreeMap<usize, ChatCompletionToolCall>,
}

impl ToolCallAccumulator {
    pub(super) fn push(&mut self, deltas: &[ChatCompletionToolCallDelta]) -> Result<()> {
        for delta in deltas {
            let call = self
                .calls
                .entry(delta.index)
                .or_insert_with(|| ChatCompletionToolCall {
                    id: String::new(),
                    kind: "function".to_string(),
                    function: ChatCompletionFunctionCall {
                        name: String::new(),
                        arguments: String::new(),
                    },
                });
            if let Some(id) = &delta.id {
                call.id.push_str(id);
            }
            if let Some(kind) = &delta.kind {
                if kind != "function" {
                    bail!("llama-server returned unsupported tool-call type {kind}");
                }
                call.kind.clone_from(kind);
            }
            if let Some(function) = &delta.function {
                if let Some(name) = &function.name {
                    call.function.name.push_str(name);
                }
                if let Some(arguments) = &function.arguments {
                    call.function.arguments.push_str(arguments);
                }
            }
        }
        Ok(())
    }

    pub(super) fn finish(self) -> Result<Option<Vec<ChatCompletionToolCall>>> {
        if self.calls.is_empty() {
            return Ok(None);
        }
        let calls: Vec<_> = self.calls.into_values().collect();
        if calls
            .iter()
            .any(|call| call.id.is_empty() || call.function.name.is_empty())
        {
            bail!("llama-server returned an incomplete tool call (missing id or function name)");
        }
        let mut ids = std::collections::HashSet::new();
        for call in &calls {
            if !ids.insert(call.id.as_str()) {
                bail!("llama-server returned duplicate tool-call IDs");
            }
            let arguments: serde_json::Value = serde_json::from_str(&call.function.arguments)
                .context("llama-server returned incomplete or invalid tool-call arguments")?;
            if !arguments.is_object() {
                bail!("llama-server tool-call arguments must be a JSON object");
            }
        }
        Ok(Some(calls))
    }
}
