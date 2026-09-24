//! Authenticated, explicit execution of model-selected Werk function calls.
//! Chat inference only returns nested tool calls; it never executes them.
use super::{response::api_error, state::ApiState};
use crate::{
    capabilities::InferenceTask,
    inference_service::{tool_definitions, tool_inference_request},
    openai::ChatCompletionRequest,
};
use axum::{
    Json,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Call {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: Function,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Function {
    name: String,
    arguments: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JobArguments {
    id: String,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ToolsQuery {
    task: Option<String>,
}

pub(super) async fn list(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(query): Query<ToolsQuery>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let tools = if let Some(task) = query.task {
        let task = match task.parse::<InferenceTask>() {
            Ok(task) => task,
            Err(error) => return api_error(StatusCode::BAD_REQUEST, error, Some("task".into())),
        };
        let name = task.to_string().replace('-', "_");
        tool_definitions()
            .into_iter()
            .filter(|tool| {
                let tool_name = tool["function"]["name"].as_str().unwrap_or_default();
                tool_name == name || matches!(tool_name, "get_job" | "cancel_job")
            })
            .collect()
    } else {
        tool_definitions()
    };
    Json(json!({"object":"werk.tools","tools":tools,"call_url":"/v1/tools/call"})).into_response()
}

pub(super) async fn call(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(call): Json<Call>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    if call.id.trim().is_empty() || call.id.len() > 256 || call.kind != "function" {
        return invalid(
            "expected a function tool call with a nonempty id (at most 256 bytes)".into(),
        );
    }
    let arguments: Value = match serde_json::from_str(&call.function.arguments) {
        Ok(Value::Object(object)) => Value::Object(object),
        _ => return invalid("function.arguments must encode a JSON object".into()),
    };
    if matches!(call.function.name.as_str(), "get_job" | "cancel_job") {
        let args: JobArguments = match serde_json::from_value(arguments) {
            Ok(args) => args,
            Err(error) => return invalid(error.to_string()),
        };
        let manager = state.job_manager.clone();
        let cancel = call.function.name == "cancel_job";
        return match tokio::task::spawn_blocking(move || {
            if cancel {
                manager.store().cancel(&args.id)
            } else {
                manager.store().get(&args.id)
            }
        })
        .await
        {
            Ok(Ok(record)) => result(&call, StatusCode::OK, job_value(record)),
            Ok(Err(error)) => invalid(error.to_string()),
            Err(error) => internal(error.to_string()),
        };
    }
    let task = match call.function.name.parse::<InferenceTask>() {
        Ok(task) if call.function.name == task.to_string().replace('-', "_") => task,
        _ => return invalid(format!("unknown Werk function '{}'", call.function.name)),
    };
    if matches!(
        task,
        InferenceTask::TextGeneration | InferenceTask::ImageUnderstanding
    ) {
        // Match the published function schema and avoid hidden streaming/extra fields.
        let allowed = [
            "model",
            "messages",
            "max_tokens",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        ];
        if let Some(key) = arguments
            .as_object()
            .unwrap()
            .keys()
            .find(|key| !allowed.contains(&key.as_str()))
        {
            return invalid(format!("unknown function argument '{key}'"));
        }
        if !arguments["model"]
            .as_str()
            .is_some_and(|s| !s.trim().is_empty())
        {
            return invalid("model must be a nonempty string".into());
        }
        let request: ChatCompletionRequest = match serde_json::from_value(arguments) {
            Ok(request) => request,
            Err(error) => return invalid(error.to_string()),
        };
        if task == InferenceTask::ImageUnderstanding
            && !crate::openai::image_urls_from_messages(&request.messages)
                .iter()
                .any(|url| !url.trim().is_empty())
        {
            return invalid(
                "image_understanding requires an image_url or input_image content part".into(),
            );
        }
        let response =
            super::chat::chat_completions_handler(State(state), headers, Ok(Json(request))).await;
        if !response.status().is_success() {
            return response;
        }
        return match axum::body::to_bytes(response.into_body(), 16 * 1024 * 1024).await {
            Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                Ok(value) => result(&call, StatusCode::OK, value),
                Err(error) => internal(error.to_string()),
            },
            Err(error) => internal(error.to_string()),
        };
    }
    let request = match tool_inference_request(task, arguments) {
        Ok(request) => request,
        Err(error) => return invalid(error.to_string()),
    };
    let service = state.inference_service.clone();
    let validation = request.clone();
    match tokio::task::spawn_blocking(move || service.resolve(validation)).await {
        Ok(Ok(_)) => {}
        Ok(Err(error)) => return invalid(error.to_string()),
        Err(error) => return internal(error.to_string()),
    }
    match state.job_manager.submit(request) {
        Ok(record) => result(&call, StatusCode::ACCEPTED, job_value(record)),
        Err(error) => internal(error.to_string()),
    }
}

fn job_value(record: crate::inference_service::JobRecord) -> Value {
    // A tool result goes straight back into a model's context. Never echo input
    // files or the effective request, including copies in result metadata.
    let mut value = json!({
        "id": record.id,
        "status": record.status,
        "model": record.request.model,
        "task": record.request.task,
        "created_unix": record.created_unix,
        "updated_unix": record.updated_unix,
        "error": record.error,
        "url": format!("/v1/jobs/{}", record.id),
        "result": null,
    });
    if let Some(result) = record.result {
        value["result"] = json!({
            "id": result.id,
            "runtime": result.runtime,
            "warnings": result.warnings,
        });
        value["outputs"] = json!(result.outputs.iter().map(|output|json!({
            "id":output.id,"url":format!("/v1/outputs/{}",output.id),"mime_type":output.mime_type,
            "size_bytes":output.size_bytes,"width":output.width,"height":output.height,"duration":output.duration
        })).collect::<Vec<_>>());
    }
    value
}

fn result(call: &Call, status: StatusCode, value: Value) -> Response {
    (status, Json(json!({"object":"werk.tool_result","tool_call_id":call.id,"name":call.function.name,
        "message":{"role":"tool","tool_call_id":call.id,"content":value.to_string()},"result":value}))).into_response()
}
fn invalid(message: String) -> Response {
    api_error(
        StatusCode::BAD_REQUEST,
        message,
        Some("function.arguments".into()),
    )
}
fn internal(message: String) -> Response {
    api_error(StatusCode::INTERNAL_SERVER_ERROR, message, None)
}
