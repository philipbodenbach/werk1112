mod request;
mod response;
mod stream;

use super::{
    generation::{self, ContextPolicy},
    state::ApiState,
};
use crate::anthropic::MessagesRequest;
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
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let suffix = format!("{stamp:x}_{sequence:x}");
    let request_id = format!("req_werk_{suffix}");
    let id = format!("msg_werk_{suffix}");
    let mut response = handle(state, headers, request, &id, &request_id).await;
    response
        .headers_mut()
        .insert("request-id", request_id.parse().expect("ASCII request ID"));
    response
}

async fn handle(
    state: ApiState,
    headers: HeaderMap,
    request: Result<Json<MessagesRequest>, JsonRejection>,
    id: &str,
    request_id: &str,
) -> Response {
    if state.authorize(&headers).is_err() {
        let mut error = response::error(
            StatusCode::UNAUTHORIZED,
            "invalid or missing API key",
            request_id,
        );
        error
            .headers_mut()
            .insert("www-authenticate", "Bearer".parse().unwrap());
        return error;
    }
    if headers
        .get("anthropic-version")
        .and_then(|v| v.to_str().ok())
        != Some("2023-06-01")
    {
        return response::error(
            StatusCode::BAD_REQUEST,
            "anthropic-version must be 2023-06-01",
            request_id,
        );
    }
    if headers.contains_key("anthropic-beta") {
        return response::error(
            StatusCode::BAD_REQUEST,
            "anthropic-beta features are not supported",
            request_id,
        );
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
                request_id,
            );
        }
    };
    let request = match request::translate(request) {
        Ok(request) => request,
        Err(error) => return response::error(StatusCode::BAD_REQUEST, error, request_id),
    };
    let prepared =
        match generation::prepare(state, request, ContextPolicy::Reject, "/v1/messages").await {
            Ok(prepared) => prepared,
            Err(error) => return response::error(error.status, error.message, request_id),
        };
    let model = prepared.manifest.id.clone();
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
