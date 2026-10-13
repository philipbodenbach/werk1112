//! Task-specific extensions preserve vendor contracts without changing chat/media defaults.
use super::state::ApiState;
use crate::{
    model_support,
    providers::{action, http::ProviderError, media},
};
use axum::{
    Json,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;

fn failure(error: anyhow::Error) -> Response {
    let (code, status) = if let Some(e) = error.downcast_ref::<ProviderError>() {
        (
            e.code,
            match e.code {
                "missing_credentials" | "provider_access_denied" => StatusCode::SERVICE_UNAVAILABLE,
                "provider_rate_limited" => StatusCode::TOO_MANY_REQUESTS,
                "content_moderated" | "upstream_invalid_request" => StatusCode::BAD_REQUEST,
                "upstream_timeout" => StatusCode::GATEWAY_TIMEOUT,
                _ => StatusCode::BAD_GATEWAY,
            },
        )
    } else if error.to_string().starts_with("invalid_request:") {
        ("invalid_request", StatusCode::BAD_REQUEST)
    } else {
        ("inference_error", StatusCode::BAD_GATEWAY)
    };
    let upstream_status = error.downcast_ref::<ProviderError>().and_then(|e| e.status);
    (status,Json(json!({"error":{"code":code,"message":error.to_string(),"upstream_status":upstream_status}}))).into_response()
}
fn parse<T: for<'de> Deserialize<'de>>(
    request: Result<Json<T>, axum::extract::rejection::JsonRejection>,
) -> Result<T, Box<Response>> {
    request.map(|Json(v)| v).map_err(|e| {
        Box::new(
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"error":{"code":"invalid_request","message":e.body_text()}})),
            )
                .into_response(),
        )
    })
}
pub(super) async fn catalog(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    Json(json!({"checked_at":"2026-10-13","models":model_support::CATALOG})).into_response()
}
pub(super) async fn image(
    State(state): State<ApiState>,
    headers: HeaderMap,
    request: Result<Json<media::FluxImageRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let request = match parse(request) {
        Ok(v) => v,
        Err(e) => return *e,
    };
    let permit = match state.provider_gate.clone().acquire_owned().await {
        Ok(p) => p,
        Err(e) => return failure(e.into()),
    };
    match crate::logging::spawn_blocking(move || {
        let _permit = permit;
        media::image(&request)
    })
    .await
    {
        Ok(Ok(output)) => ([(header::CONTENT_TYPE, output.mime)], output.bytes).into_response(),
        Ok(Err(e)) => failure(e),
        Err(e) => failure(e.into()),
    }
}
pub(super) async fn speech(
    State(state): State<ApiState>,
    headers: HeaderMap,
    request: Result<Json<media::SpeechRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let request = match parse(request) {
        Ok(v) => v,
        Err(e) => return *e,
    };
    let permit = match state.provider_gate.clone().acquire_owned().await {
        Ok(p) => p,
        Err(e) => return failure(e.into()),
    };
    let mime = match request.validate() {
        Ok(m) => m,
        Err(e) => return failure(e),
    };
    if !request.stream {
        return match crate::logging::spawn_blocking(move || {
            let _permit = permit;
            media::speech(&request)
        })
        .await
        {
            Ok(Ok(output)) => ([(header::CONTENT_TYPE, output.mime)], output.bytes).into_response(),
            Ok(Err(e)) => failure(e),
            Err(e) => failure(e.into()),
        };
    }
    // Authenticate and obtain upstream status before sending HTTP 200. Bounded
    // channels exert backpressure; disconnects stop reads and drop the response.
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(8);
    crate::logging::spawn_blocking(move || {
        let _permit = permit;
        use std::io::Read;
        let mut response = match media::speech_response(&request) {
            Ok(r) => r,
            Err(e) => {
                let _ = ready_tx.send(Err(e));
                return;
            }
        };
        if ready_tx.send(Ok(())).is_err() {
            return;
        }
        let mut chunk = [0u8; 16384];
        let mut total = 0usize;
        while !tx.is_closed() {
            match response.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    total += n;
                    if total > 64 * 1024 * 1024 {
                        let _ = tx.blocking_send(Err(std::io::Error::other(
                            "audio response exceeds size limit",
                        )));
                        break;
                    }
                    if tx.blocking_send(Ok(chunk[..n].to_vec())).is_err() {
                        break;
                    }
                }
                Err(_) => {
                    let _ = tx
                        .blocking_send(Err(std::io::Error::other("upstream audio stream failed")));
                    break;
                }
            }
        }
    });
    match ready_rx.await {
        Ok(Ok(())) => (
            [(header::CONTENT_TYPE, mime)],
            Body::from_stream(ReceiverStream::new(rx)),
        )
            .into_response(),
        Ok(Err(e)) => failure(e),
        Err(e) => failure(e.into()),
    }
}
pub(super) async fn action(
    State(state): State<ApiState>,
    headers: HeaderMap,
    request: Result<Json<action::ActionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let request = match parse(request) {
        Ok(v) => v,
        Err(e) => return *e,
    };
    let permit = match state.provider_gate.clone().acquire_owned().await {
        Ok(p) => p,
        Err(e) => return failure(e.into()),
    };
    let store = state.store.clone();
    let actions = state.actions.clone();
    match crate::logging::spawn_blocking(move || {
        let _permit = permit;
        actions.execute(&store, request)
    })
    .await
    {
        Ok(Ok(output)) => Json(output).into_response(),
        Ok(Err(e)) => failure(e),
        Err(e) => failure(e.into()),
    }
}
