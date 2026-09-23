use super::request::valid_name;
use crate::backend::{GenerateResponse, GeneratedAssistantMessage};
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

pub(super) fn stop_reason(reason: &str) -> Result<&'static str, String> {
    match reason {
        "stop" => Ok("end_turn"),
        "length" => Ok("max_tokens"),
        "tool_calls" => Ok("tool_use"),
        _ => Err(format!("unsupported backend finish reason: {reason}")),
    }
}

pub(super) fn message(id: &str, model: &str, response: GenerateResponse) -> Result<Value, String> {
    let stop = stop_reason(&response.finish_reason)?;
    let assistant = response
        .assistant_message
        .unwrap_or(GeneratedAssistantMessage {
            content: Some(response.text),
            tool_calls: None,
        });
    let mut content = Vec::new();
    if let Some(text) = assistant.content.filter(|text| !text.is_empty()) {
        content.push(json!({"type":"text", "text":text}));
    }
    let mut ids = std::collections::HashSet::new();
    for call in assistant.tool_calls.unwrap_or_default() {
        if call.kind != "function"
            || !valid_name(&call.id)
            || !valid_name(&call.function.name)
            || !ids.insert(call.id.clone())
        {
            return Err("backend returned an invalid or duplicate tool identity".into());
        }
        let input: Value = serde_json::from_str(&call.function.arguments)
            .map_err(|_| "backend returned incomplete or invalid tool JSON; increase max_tokens if truncated")?;
        if !input.is_object() {
            return Err("backend tool arguments must be a JSON object".into());
        }
        content
            .push(json!({"type":"tool_use", "id":call.id,"name":call.function.name,"input":input}));
    }
    validate_finish(stop, !ids.is_empty())?;
    if content.is_empty() {
        content.push(json!({"type":"text", "text":""}));
    }
    Ok(
        json!({"id":id,"type":"message","role":"assistant","model":model,
        "content":content,"stop_reason":stop,"stop_sequence":null,
        "usage":{"input_tokens":response.prompt_tokens,"output_tokens":response.completion_tokens}}),
    )
}

pub(super) fn validate_finish(stop: &str, has_tools: bool) -> Result<(), String> {
    if (stop == "tool_use" && !has_tools) || (stop == "end_turn" && has_tools) {
        return Err("backend finish reason disagrees with its tool calls".into());
    }
    Ok(())
}

pub(super) fn error_value(kind: &str, message: impl Into<String>, request_id: &str) -> Value {
    json!({"type":"error", "error":{"type":kind,"message":message.into()},"request_id":request_id})
}

pub(super) fn error(status: StatusCode, message: impl Into<String>, request_id: &str) -> Response {
    let kind = match status {
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        status if status.is_server_error() => "api_error",
        _ => "invalid_request_error",
    };
    (status, Json(error_value(kind, message, request_id))).into_response()
}
