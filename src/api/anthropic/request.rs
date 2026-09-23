use crate::{
    anthropic::{self as wire, Block, Content},
    openai::*,
};
use base64::Engine;
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
    if request.messages.len() > 100_000 {
        return Err("messages may contain at most 100000 entries".into());
    }
    if request
        .metadata
        .as_ref()
        .and_then(|m| m.user_id.as_ref())
        .is_some_and(|id| id.len() > 256)
    {
        return Err("metadata.user_id must not exceed 256 bytes".into());
    }
    if request
        .temperature
        .is_some_and(|v| !(0.0..=1.0).contains(&v))
        || request.top_p.is_some_and(|v| !(0.0..=1.0).contains(&v))
    {
        return Err("temperature and top_p must be between 0 and 1".into());
    }
    if request.stop_sequences.iter().any(String::is_empty) || request.stop_sequences.len() > 4 {
        return Err("stop_sequences accepts at most four nonempty strings".into());
    }
    let mut extra = std::collections::BTreeMap::new();
    if let Some(k) = request.top_k {
        extra.insert("top_k".into(), serde_json::json!(k));
    }
    if let Some(config) = request.output_config {
        let wire::OutputFormat::JsonSchema { schema } = config.format;
        extra.insert("response_format".into(),serde_json::json!({"type":"json_schema","json_schema":{"name":"response","schema":schema,"strict":true}}));
    }
    if !request.stop_sequences.is_empty() {
        extra.insert(
            "__werk_matched_stop".into(),
            serde_json::json!(request.stop_sequences),
        );
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
        let mut parts = Vec::new();
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
                    parts.push(text_part(value));
                }
                Block::Image { source } => {
                    if message.role != "user" || !pending.is_empty() {
                        return Err(
                            "images require a user message after all pending tool results".into(),
                        );
                    }
                    parts.push(image_part(source)?);
                }
                Block::Document {
                    source,
                    title,
                    context,
                    citations,
                } => {
                    if message.role != "user" || !pending.is_empty() {
                        return Err(
                            "documents require a user message after all pending tool results"
                                .into(),
                        );
                    }
                    parts.push(document_part(source, title, context, citations)?);
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
                    if message.role != "user" || !parts.is_empty() || !pending.remove(&tool_use_id)
                    {
                        return Err("tool_result must precede text and match a pending tool_use exactly once".into());
                    }
                    let mut result = text_message("tool", String::new());
                    result.content = Some(tool_result_content(content, is_error)?);
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
        if !parts.is_empty() || !calls.is_empty() || !had_results {
            let mut translated = text_message(&message.role, String::new());
            translated.content = if parts.is_empty() {
                None
            } else {
                Some(parts_content(parts))
            };
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
        extra,
        model: Some(request.model),
        messages,
        stream: Some(request.stream),
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: Some(request.max_tokens),
        max_completion_tokens: None,
        stream_options: None,
        stop: if request.stop_sequences.is_empty() {
            None
        } else {
            Some(StopSpec::Many(request.stop_sequences))
        },
        seed: None,
        tools,
        tool_choice,
        parallel_tool_calls,
        werk: request.werk,
    })
}

fn text_part(text: String) -> ContentPart {
    ContentPart {
        file: None,
        kind: "text".into(),
        text: Some(text),
        image_url: None,
    }
}

fn document_part(
    source: crate::documents::DocumentSource,
    title: Option<String>,
    context: Option<String>,
    citations: Option<wire::DocumentCitations>,
) -> Result<ContentPart, String> {
    if citations.is_some_and(|c| c.enabled) {
        return Err("document citations are not yet supported by the response adapter".into());
    }
    Ok(ContentPart {
        kind: "file".into(),
        text: None,
        image_url: None,
        file: Some(crate::documents::FilePart {
            source: Some(source),
            filename: title,
            context,
            ..Default::default()
        }),
    })
}

fn image_part(source: wire::ImageSource) -> Result<ContentPart, String> {
    let url = match source {
        wire::ImageSource::Base64 { media_type, data } => {
            if !matches!(
                media_type.as_str(),
                "image/jpeg" | "image/png" | "image/gif" | "image/webp"
            ) {
                return Err(
                    "image media_type must be image/jpeg, image/png, image/gif or image/webp"
                        .into(),
                );
            }
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(&data)
                .map_err(|_| "image source contains invalid base64")?;
            if decoded.is_empty() {
                return Err("image source must not be empty".into());
            }
            format!("data:{media_type};base64,{data}")
        }
        wire::ImageSource::Url { url } => {
            let uri = url
                .parse::<axum::http::Uri>()
                .map_err(|_| "image URL must be a valid HTTP(S) URL")?;
            if !matches!(uri.scheme_str(), Some("https" | "http"))
                || uri.host().is_none_or(str::is_empty)
            {
                return Err("image URL must use http or https and include a host".into());
            }
            url
        }
    };
    Ok(ContentPart {
        file: None,
        kind: "image_url".into(),
        text: None,
        image_url: Some(ImageUrlSpec::Url(url)),
    })
}

fn parts_content(parts: Vec<ContentPart>) -> MessageContent {
    if parts.iter().all(|part| part.kind == "text") {
        MessageContent::Text(
            parts
                .into_iter()
                .filter_map(|part| part.text)
                .collect::<Vec<_>>()
                .join("\n"),
        )
    } else {
        MessageContent::Parts(parts)
    }
}

fn tool_result_content(
    content: Option<wire::ToolResultContent>,
    is_error: bool,
) -> Result<MessageContent, String> {
    let parts = match content {
        None => vec![text_part(String::new())],
        Some(wire::ToolResultContent::Text(text)) => vec![text_part(text)],
        Some(wire::ToolResultContent::Blocks(blocks)) => blocks
            .into_iter()
            .map(|block| match block {
                wire::ToolResultBlock::Text { text } => Ok(text_part(text)),
                wire::ToolResultBlock::Image { source } => image_part(source),
                wire::ToolResultBlock::Document {
                    source,
                    title,
                    context,
                    citations,
                } => document_part(source, title, context, citations),
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    Ok(match (parts_content(parts), is_error) {
        (MessageContent::Text(text), true) => {
            MessageContent::Text(serde_json::json!({"is_error":true,"content":text}).to_string())
        }
        (MessageContent::Parts(mut parts), true) => {
            parts.insert(0, text_part("{\"is_error\":true}".into()));
            MessageContent::Parts(parts)
        }
        (content, false) => content,
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
