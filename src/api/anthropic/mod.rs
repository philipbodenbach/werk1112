mod extended;
mod request;
mod response;
mod stream;

use super::{
    generation::{self, ContextPolicy},
    state::ApiState,
};
use crate::anthropic::{CountTokensRequest, MessagesRequest};
use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{KeepAlive, Sse},
    },
};

pub(super) async fn messages_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    request: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let (id, request_id) = request_ids();
    with_request_id(
        handle(state, headers, request, &id, &request_id).await,
        &request_id,
    )
}

fn request_ids() -> (String, String) {
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let suffix = format!("{stamp:x}_{sequence:x}");
    let request_id = format!("req_werk_{suffix}");
    let id = format!("msg_werk_{suffix}");
    (id, request_id)
}

fn with_request_id(mut response: Response, request_id: &str) -> Response {
    response
        .headers_mut()
        .insert("request-id", request_id.parse().expect("ASCII request ID"));
    response
}

pub(super) async fn count_tokens_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    request: Result<Json<CountTokensRequest>, JsonRejection>,
) -> Response {
    let (_, request_id) = request_ids();
    let response = async {
        if let Err(error) = validate_headers(&state, &headers, &request_id) {
            return error;
        }
        let request = match request {
            Ok(Json(request)) => request,
            Err(error) => {
                return response::error(
                    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
                        StatusCode::PAYLOAD_TOO_LARGE
                    } else {
                        StatusCode::BAD_REQUEST
                    },
                    error.body_text(),
                    &request_id,
                );
            }
        };
        let request = match request::translate(request.into()) {
            Ok(request) => request,
            Err(error) => return response::error(StatusCode::BAD_REQUEST, error, &request_id),
        };
        let prepared = match generation::prepare(
            state,
            request,
            ContextPolicy::Count,
            "/v1/messages/count_tokens",
            &headers,
        )
        .await
        {
            Ok(prepared) => prepared,
            Err(error) => return response::error(error.status, error.message, &request_id),
        };
        match tokio::task::spawn_blocking(move || {
            prepared
                .state
                .backend
                .count_tokens(&prepared.manifest, prepared.request)
        })
        .await
        {
            Ok(Ok(count)) => Json(serde_json::json!({"input_tokens":count})).into_response(),
            Ok(Err(error)) => {
                response::error(StatusCode::BAD_REQUEST, error.to_string(), &request_id)
            }
            Err(_) => response::error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "token counting task failed",
                &request_id,
            ),
        }
    }
    .await;
    with_request_id(response, &request_id)
}

async fn handle(
    state: ApiState,
    headers: HeaderMap,
    request: Result<Json<MessagesRequest>, JsonRejection>,
    id: &str,
    request_id: &str,
) -> Response {
    if let Err(error) = validate_headers(&state, &headers, request_id) {
        return error;
    }
    handle_request(state, request, id, request_id, &headers).await
}

fn validate_headers(
    state: &ApiState,
    headers: &HeaderMap,
    request_id: &str,
) -> Result<(), Response> {
    if state.authorize(&headers).is_err() {
        let mut error = response::error(
            StatusCode::UNAUTHORIZED,
            "invalid or missing API key",
            request_id,
        );
        error
            .headers_mut()
            .insert("www-authenticate", "Bearer".parse().unwrap());
        return Err(error);
    }
    if headers
        .get("anthropic-version")
        .and_then(|v| v.to_str().ok())
        != Some("2023-06-01")
    {
        return Err(response::error(
            StatusCode::BAD_REQUEST,
            "anthropic-version must be 2023-06-01",
            request_id,
        ));
    }
    if headers
        .get("anthropic-beta")
        .is_some_and(|v| v != "files-api-2025-04-14")
    {
        return Err(response::error(
            StatusCode::BAD_REQUEST,
            "anthropic-beta features are not supported",
            request_id,
        ));
    }
    Ok(())
}

async fn handle_request(
    state: ApiState,
    request: Result<Json<MessagesRequest>, JsonRejection>,
    id: &str,
    request_id: &str,
    headers: &HeaderMap,
) -> Response {
    let request = match request {
        Ok(Json(request)) => request,
        Err(error) => {
            return response::error(
                if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
                    StatusCode::PAYLOAD_TOO_LARGE
                } else {
                    StatusCode::BAD_REQUEST
                },
                error.body_text(),
                request_id,
            );
        }
    };
    let request = match request::translate(request) {
        Ok(request) => request,
        Err(error) => return response::error(StatusCode::BAD_REQUEST, error, request_id),
    };
    let tool_names = request
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tool| tool.function.name.clone())
        .collect();
    let prepared = match generation::prepare(
        state,
        request,
        ContextPolicy::Reject,
        "/v1/messages",
        headers,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(error) => return response::error(error.status, error.message, request_id),
    };
    let model = prepared.manifest.id.clone();
    if !prepared.api_options.is_empty() {
        return extended::handle(prepared, id, request_id, tool_names).await;
    }
    let verbose = prepared.state.verbose;
    if prepared.stream {
        let source = generation::generate_stream(
            &prepared.state,
            prepared.manifest,
            prepared.request,
            prepared.explicit_runtime_options,
        );
        let mut stream = stream::MessagesStream::new(source, &model, id, request_id);
        stream.verbose = verbose;
        stream.tool_names = tool_names;
        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match generation::generate(
            prepared.state,
            prepared.manifest,
            prepared.request,
            prepared.explicit_runtime_options,
        )
        .await
        {
            Ok(result) => {
                if verbose {
                    log_completion(
                        &model,
                        &result.finish_reason,
                        result.prompt_tokens,
                        result.completion_tokens,
                        result.timings,
                        &result.backend_diagnostics,
                    );
                }
                match response::message(id, &model, result) {
                    Ok(message) => Json(message).into_response(),
                    Err(error) => response::error(StatusCode::BAD_GATEWAY, error, request_id),
                }
            }
            Err(error) => response::error(StatusCode::BAD_REQUEST, error.to_string(), request_id),
        }
    }
}

fn log_completion(
    model: &str,
    finish_reason: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    timings: crate::backend::GenerationTimings,
    diagnostics: &[String],
) {
    eprintln!(
        "[werk serve] anthropic {}",
        serde_json::json!({"model":model,"finish_reason":finish_reason,
        "prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,"timings":timings,
        "backend_diagnostics":diagnostics})
    );
}
