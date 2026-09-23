use crate::{
    anthropic::{self as wire, Block, Content},
    openai::*,
};
use std::collections::HashSet;

pub(super) fn valid_name(name: &str) -> bool {
    (1..=128).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

pub(super) fn translate(request: wire::MessagesRequest) -> Result<ChatCompletionRequest, String> {
    if request.model.trim().is_empty() || request.max_tokens == 0 || request.messages.is_empty() {
        return Err("model, nonempty messages and positive max_tokens are required".into());
    }
    if request
        .temperature
        .is_some_and(|v| !(0.0..=1.0).contains(&v))
        || request.top_p.is_some_and(|v| !(0.0..=1.0).contains(&v))
    {
        return Err("temperature and top_p must be between 0 and 1".into());
    }
    if !request.stop_sequences.is_empty() {
        return Err("nonempty stop_sequences are not supported: the backend cannot report the matched sequence".into());
    }
    let mut names = HashSet::new();
    let tools = request
        .tools
        .map(|tools| {
            tools
                .into_iter()
                .map(|tool| {
                    if !valid_name(&tool.name) || !names.insert(tool.name.clone()) {
                        return Err(
                            "tool names must be unique and match [a-zA-Z0-9_-]{1,128}".to_string()
                        );
                    }
                    if tool.kind.as_deref().is_some_and(|kind| kind != "custom") {
                        return Err("only custom client tools are supported".into());
                    }
                    if !tool.input_schema.is_object()
                        || tool.input_schema.get("type").and_then(|v| v.as_str()) != Some("object")
                    {
                        return Err(
                            "tools[].input_schema must be a JSON object schema with type object"
                                .into(),
                        );
                    }
                    Ok(ChatCompletionTool {
                        kind: "function".into(),
                        function: ChatCompletionToolFunction {
                            name: tool.name,
                            description: tool.description,
                            parameters: Some(tool.input_schema),
                            strict: tool.strict,
                        },
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?
        .filter(|tools| !tools.is_empty());

    let mut parallel_tool_calls = None;
    let tool_choice = request
        .tool_choice
        .map(|choice| {
            use wire::ToolChoice as C;
            let (choice, disable) = match choice {
                C::None => (ToolChoice::Mode(ToolChoiceMode::None), None),
                C::Auto {
                    disable_parallel_tool_use,
                } => (
                    ToolChoice::Mode(ToolChoiceMode::Auto),
                    disable_parallel_tool_use,
                ),
                C::Any {
                    disable_parallel_tool_use,
                } => (
                    ToolChoice::Mode(ToolChoiceMode::Required),
                    disable_parallel_tool_use,
                ),
                C::Tool {
                    name,
                    disable_parallel_tool_use,
                } => {
                    if !names.contains(&name) {
                        return Err("tool_choice names an undefined tool".to_string());
                    }
                    (
                        ToolChoice::Named(NamedToolChoice {
                            kind: "function".into(),
                            function: NamedToolChoiceFunction { name },
                        }),
                        disable_parallel_tool_use,
                    )
                }
            };
            if tools.is_none() && !matches!(choice, ToolChoice::Mode(ToolChoiceMode::None)) {
                return Err("tool_choice requires tools".into());
            }
            // False permits the backend default; true is a constraint it must enforce.
            if disable == Some(true) {
                parallel_tool_calls = Some(false);
            }
            Ok(choice)
        })
        .transpose()?;

    let mut messages = Vec::new();
    if let Some(system) = request.system {
        messages.push(text_message("system", system.into_text()));
    }
    let mut seen = HashSet::new();
    let mut pending = HashSet::new();
    for message in request.messages {
        if !matches!(message.role.as_str(), "user" | "assistant") {
            return Err(
                "messages[].role must be user or assistant; use the separate system field".into(),
            );
        }
        if !pending.is_empty() && message.role != "user" {
            return Err("tool_use must be followed immediately by user tool_result blocks".into());
        }
        let blocks = match message.content {
            Content::Text(text) => vec![Block::Text { text }],
            Content::Blocks(blocks) => blocks,
        };
        if blocks.is_empty() {
            return Err("message content must not be empty".into());
        }
        let mut text = Vec::new();
        let mut calls = Vec::new();
        let mut had_results = false;
        for block in blocks {
            match block {
                Block::Text { text: value } => {
                    if !calls.is_empty() {
                        return Err("assistant text after tool_use is not supported; place text before tool calls".into());
                    }
                    if !pending.is_empty() {
                        return Err("all pending tool_result blocks must precede user text".into());
                    }
                    text.push(value);
                }
                Block::ToolUse { id, name, input } => {
                    if message.role != "assistant"
                        || !valid_name(&name)
                        || !valid_name(&id)
                        || !input.is_object()
                    {
                        return Err(
                            "tool_use requires assistant role, valid id/name and object input"
                                .into(),
                        );
                    }
                    if !seen.insert(id.clone()) {
                        return Err("duplicate tool_use id".into());
                    }
                    // Pending results are checked on the following message, not sibling calls.
                    calls.push(ChatCompletionToolCall {
                        id,
                        kind: "function".into(),
                        function: ChatCompletionFunctionCall {
                            name,
                            arguments: input.to_string(),
                        },
                    });
                }
                Block::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    if message.role != "user" || !text.is_empty() || !pending.remove(&tool_use_id) {
                        return Err("tool_result must precede text and match a pending tool_use exactly once".into());
                    }
                    let content = content
                        .map(wire::TextContent::into_text)
                        .unwrap_or_default();
                    let content = if is_error {
                        serde_json::json!({"is_error": true, "content": content}).to_string()
                    } else {
                        content
                    };
                    let mut result = text_message("tool", content);
                    result.tool_call_id = Some(tool_use_id);
                    messages.push(result);
                    had_results = true;
                }
            }
        }
        if !pending.is_empty() {
            return Err("user message is missing a tool_result for a pending tool_use".into());
        }
        pending.extend(calls.iter().map(|call| call.id.clone()));
        if !text.is_empty() || !calls.is_empty() || !had_results {
            let mut translated = text_message(&message.role, text.join("\n"));
            if text.is_empty() {
                translated.content = None;
            }
            if !calls.is_empty() {
                translated.tool_calls = Some(calls);
            }
            messages.push(translated);
        }
    }
    if !pending.is_empty() {
        return Err(
            "history ends with tool_use; provide its user tool_result before generating again"
                .into(),
        );
    }
    Ok(ChatCompletionRequest {
        model: Some(request.model),
        messages,
        stream: Some(request.stream),
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: Some(request.max_tokens),
        max_completion_tokens: None,
        stream_options: None,
        stop: None,
        seed: None,
        tools,
        tool_choice,
        parallel_tool_calls,
        werk: None,
    })
}

fn text_message(role: &str, text: String) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: Some(MessageContent::Text(text)),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}
