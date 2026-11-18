//! Werk Protocol transport for installed inference routes.
use super::*;
use crate::api::{chat, media, tools};
use crate::openai::ChatCompletionRequest;
use crate::werk_protocol::{Capability, CapabilityStatus};
use axum::extract::{FromRequest, Request};
use axum::middleware::{self, Next};
use serde_json::{Value, json};

pub(super) fn routes(state: &ApiState, body_limit: usize) -> Router<ApiState> {
    let json = Router::new()
        .route("/werk/v1/models", get(chat::models_handler))
        .route("/werk/v1/tasks", get(media::capabilities_handler))
        .route("/werk/v1/tools", get(tools::list))
        .route("/werk/v1/tools/call", post(tools::call))
        .route(
            "/werk/v1/jobs/{id}",
            get(media::get_job_handler).delete(media::cancel_job_handler),
        )
        .layer(middleware::from_fn_with_state(state.clone(), json_route));
    json.route("/werk/v1/chat/completions", post(completions))
        .route("/werk/v1/outputs/{id}", get(output))
        .layer(DefaultBodyLimit::max(body_limit))
}

pub(super) fn advertise(capabilities: &mut Vec<Capability>) {
    for (id, detail, operations) in [
        (
            "api.models.list",
            "Lists configured models and deployment aliases",
            &["list"][..],
        ),
        (
            "api.chat.completions",
            "Complete chat history, including tool calls; model support is checked on each request",
            &["complete", "stream"][..],
        ),
        (
            "api.tools",
            "Discovery and invocation of built-in inference functions",
            &["list", "tasks", "call"][..],
        ),
        (
            "api.jobs",
            "Status and cancellation of inference jobs",
            &["get", "cancel"][..],
        ),
        (
            "api.outputs",
            "Reads stored inference artifacts by ID",
            &["read"][..],
        ),
    ] {
        capabilities.retain(|entry| entry.id != id);
        capabilities.push(Capability {
            id: id.into(),
            status: CapabilityStatus::Supported,
            detail: detail.into(),
            operations: operations.iter().map(|s| (*s).into()).collect(),
        });
    }
    capabilities.sort_by(|a, b| a.id.cmp(&b.id));
}

async fn json_route(State(state): State<ApiState>, mut request: Request, next: Next) -> Response {
    let (id, context) = match request_context(&state, request.headers()).await {
        Ok(context) => context,
        Err(error) => return error,
    };
    request.extensions_mut().insert(context);
    wrap_json(id, next.run(request).await).await
}

async fn completions(State(state): State<ApiState>, request: Request) -> Response {
    let headers = request.headers().clone();
    let (id, _) =
        match request_context_for(&state, &headers, &["application/json", "text/event-stream"])
            .await
        {
            Ok(context) => context,
            Err(error) => return error,
        };
    let request = match Json::<ChatCompletionRequest>::from_request(request, &state).await {
        Ok(Json(request)) => request,
        Err(error) => return json_rejection(id, error),
    };
    let stream = request.stream == Some(true);
    let representation = if stream {
        "text/event-stream"
    } else {
        "application/json"
    };
    if !accepts_protocol_response(&headers, representation) {
        return protocol_error(id, incompatible_protocol());
    }
    if !chat::portable_image_inputs(&request) {
        return invalid_request(id, "Use an inline image or an HTTP(S) image URL");
    }
    let response = chat::complete(state, headers, request, "/werk/v1/chat/completions").await;
    if stream && response.status().is_success() {
        if response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.split(';').next())
            != Some("text/event-stream")
        {
            return protocol_error(
                id,
                ProtocolError::new(
                    ProtocolErrorCode::Internal,
                    "Invalid streaming representation",
                ),
            );
        }
        let mut response = response;
        set_protocol_headers(response.headers_mut(), &id);
        return response;
    }
    wrap_json(id, response).await
}

async fn output(
    State(state): State<ApiState>,
    headers: HeaderMap,
    path: Result<Path<String>, PathRejection>,
) -> Response {
    let (id, context) =
        match request_context_for(&state, &headers, &["application/octet-stream"]).await {
            Ok(context) => context,
            Err(error) => return error,
        };
    let Ok(path) = path else {
        return invalid_request(id, "Invalid output ID");
    };
    let state = state.for_inference_principal(context.principal_id());
    let mut response = media::output_handler(State(state), headers, path).await;
    if !response.status().is_success() {
        return wrap_json(id, response).await;
    }
    set_protocol_headers(response.headers_mut(), &id);
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    response
}

async fn wrap_json(id: String, response: Response) -> Response {
    let (parts, body) = response.into_parts();
    let bytes = match axum::body::to_bytes(body, MAX_WERK_RESPONSE_BYTES).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return protocol_error(
                id,
                ProtocolError::new(
                    ProtocolErrorCode::ResourceExhausted,
                    "Inference response exceeds protocol limit",
                ),
            );
        }
    };
    if parts.status.is_success() {
        return match serde_json::from_slice::<Value>(&bytes) {
            Ok(value) => json_response(parts.status, id.clone(), ProtocolEnvelope::v1(id, value)),
            Err(_) => protocol_error(
                id,
                ProtocolError::new(
                    ProtocolErrorCode::Internal,
                    "Invalid response from inference handler",
                ),
            ),
        };
    }
    let status = parts.status;
    let code = match status {
        StatusCode::UNAUTHORIZED => ProtocolErrorCode::Unauthorized,
        StatusCode::FORBIDDEN => ProtocolErrorCode::Forbidden,
        StatusCode::NOT_FOUND => ProtocolErrorCode::NotFound,
        StatusCode::CONFLICT => ProtocolErrorCode::Conflict,
        StatusCode::TOO_MANY_REQUESTS | StatusCode::PAYLOAD_TOO_LARGE => {
            ProtocolErrorCode::ResourceExhausted
        }
        StatusCode::SERVICE_UNAVAILABLE | StatusCode::BAD_GATEWAY | StatusCode::GATEWAY_TIMEOUT => {
            ProtocolErrorCode::Unavailable
        }
        _ if status.is_client_error() => ProtocolErrorCode::InvalidRequest,
        _ => ProtocolErrorCode::Internal,
    };
    let mut message = status
        .canonical_reason()
        .unwrap_or("Inference request failed")
        .to_string();
    let mut details = None;
    if status.is_client_error()
        && let Ok(value) = serde_json::from_slice::<Value>(&bytes)
    {
        let error = &value["error"];
        // Publish only structured diagnostics defined by the inference transport.
        if let Some(specific) = error["code"].as_str().filter(|v| {
            matches!(
                *v,
                "context_length_exceeded"
                    | "output_limit_exceeded"
                    | "unsupported_tool_calling"
                    | "images_unsupported"
                    | "image_input_unsupported"
                    | "vision_unsupported"
                    | "multimodal_not_supported"
                    | "missing_mmproj"
                    | "unsupported_vision"
                    | "unsupported_image_input"
                    | "unsupported_multimodal_input"
                    | "invalid_reasoning_effort"
                    | "invalid_reasoning_tier"
                    | "unsupported_chat_options"
                    | "unsupported_api_options"
            )
        }) {
            let mut fields = json!({"inference_code": specific});
            if specific == "context_length_exceeded" {
                for name in ["context_length", "prompt_tokens", "max_tokens"] {
                    if let Some(number) = error[name].as_u64() {
                        fields[name] = number.into();
                    }
                }
                if let Some(method) = error["token_count_method"].as_str().filter(|v| {
                    v.len() <= 64 && v.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
                }) {
                    fields["token_count_method"] = method.into();
                }
            }
            if let Some(detail) = error["message"].as_str() {
                message = detail.chars().take(2048).collect();
            }
            details = Some(fields);
        }
    }
    let mut response = json_response(
        status,
        id.clone(),
        ProtocolErrorEnvelope {
            protocol: ProtocolVersion::V1,
            request_id: id,
            error: ProtocolErrorBody {
                code,
                message,
                retryable: status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error(),
                details,
            },
        },
    );
    for name in [header::RETRY_AFTER, header::WWW_AUTHENTICATE] {
        if let Some(value) = parts.headers.get(&name) {
            response.headers_mut().insert(name, value.clone());
        }
    }
    response
}
