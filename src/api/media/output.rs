use axum::{
    Json,
    body::{Body, Bytes},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Serialize;
use serde_json::Value;
use std::fs;
use tokio::{io::AsyncReadExt, sync::mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    capabilities::InferenceTask,
    inference::InferenceRequest,
    inference_service::{InferenceResult, OutputMetadata, OutputStore},
};

use super::requests::DirectResponseFormat;
use crate::api::{response::api_error, state::ApiState};

pub(super) async fn submit_job(state: ApiState, request: InferenceRequest) -> Response {
    let service = state.inference_service.clone();
    let request_to_validate = request.clone();
    match tokio::task::spawn_blocking(move || service.resolve(request_to_validate)).await {
        Ok(Ok(_)) => {}
        Ok(Err(error)) => {
            return api_error(StatusCode::BAD_REQUEST, error.to_string(), None);
        }
        Err(error) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("request validation task failed: {error}"),
                None,
            );
        }
    }
    match state.job_manager.submit(request) {
        Ok(record) => (StatusCode::ACCEPTED, Json(record)).into_response(),
        Err(error) => api_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string(), None),
    }
}

#[derive(Debug, Serialize)]
struct DirectMediaResponse {
    created: u64,
    data: Vec<DirectMediaOutput>,
    werk: InferenceResult,
}

#[derive(Debug, Serialize)]
struct DirectMediaOutput {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    mime_type: String,
    size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<f64>,
}

enum DirectExecutionError {
    Inference(String),
    Response(String),
}

struct ManagedResultCleanup {
    output_store: OutputStore,
    result_id: Option<String>,
}

impl ManagedResultCleanup {
    fn new(output_store: OutputStore, result_id: String) -> Self {
        Self {
            output_store,
            result_id: Some(result_id),
        }
    }
}

impl Drop for ManagedResultCleanup {
    fn drop(&mut self) {
        let Some(result_id) = self.result_id.take() else {
            return;
        };
        if let Err(error) = self.output_store.remove_result(&result_id) {
            eprintln!("warning: failed to clean up streamed media result '{result_id}': {error:#}");
        }
    }
}

pub(super) async fn execute_direct(
    state: ApiState,
    request: InferenceRequest,
    response_format: DirectResponseFormat,
) -> Response {
    let service = state.inference_service.clone();
    match tokio::task::spawn_blocking(move || {
        let result = service
            .execute(request)
            .map_err(|error| DirectExecutionError::Inference(error.to_string()))?;
        let release_result = matches!(response_format, DirectResponseFormat::Base64)
            || matches!(
                result.task,
                InferenceTask::SpeechToText | InferenceTask::SpeechTranslation
            );
        let result_id = result.id.clone();
        let response = direct_media_response(result, response_format);
        match response {
            Ok(mut response) => {
                if release_result
                    && let Err(error) = service.output_store().remove_result(&result_id)
                {
                    response.werk.warnings.push(format!(
                        "embedded response was returned, but temporary output cleanup failed: {error:#}"
                    ));
                }
                Ok(response)
            }
            Err(error) => {
                if release_result {
                    let _ = service.output_store().remove_result(&result_id);
                }
                Err(DirectExecutionError::Response(error))
            }
        }
    })
    .await
    {
        Ok(Ok(response)) => Json(response).into_response(),
        Ok(Err(DirectExecutionError::Inference(error))) => {
            api_error(StatusCode::BAD_REQUEST, error, None)
        }
        Ok(Err(DirectExecutionError::Response(error))) => {
            api_error(StatusCode::INTERNAL_SERVER_ERROR, error, None)
        }
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("inference task failed: {error}"),
            None,
        ),
    }
}

pub(super) async fn execute_audio_bytes(state: ApiState, request: InferenceRequest) -> Response {
    let service = state.inference_service.clone();
    let output_store = service.output_store().clone();
    match tokio::task::spawn_blocking(move || {
        service.execute(request).map(|result| {
            let result_id = result.id.clone();
            let cleanup = ManagedResultCleanup::new(output_store, result_id);
            (result, cleanup)
        })
    })
    .await
    {
        Ok(Ok((result, cleanup))) => {
            let Some(output) = result
                .outputs
                .iter()
                .find(|output| output.mime_type.starts_with("audio/"))
            else {
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "text-to-speech backend did not produce an audio output".to_string(),
                    None,
                );
            };
            let body = match ephemeral_streaming_file_body(&output.path, cleanup).await {
                Ok(body) => body,
                Err(error) => {
                    return api_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("failed to read speech output: {error}"),
                        None,
                    );
                }
            };
            let mut response = Response::new(body);
            *response.status_mut() = StatusCode::OK;
            if let Ok(content_type) = HeaderValue::from_str(&output.mime_type) {
                response
                    .headers_mut()
                    .insert(header::CONTENT_TYPE, content_type);
            }
            if let Ok(content_length) = HeaderValue::from_str(&output.size_bytes.to_string()) {
                response
                    .headers_mut()
                    .insert(header::CONTENT_LENGTH, content_length);
            }
            if let Ok(output_id) = HeaderValue::from_str(&output.id) {
                response.headers_mut().insert("x-werk-output-id", output_id);
            }
            response
        }
        Ok(Err(error)) => api_error(StatusCode::BAD_REQUEST, error.to_string(), None),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("inference task failed: {error}"),
            None,
        ),
    }
}

pub(super) async fn streaming_file_body(path: &str) -> std::io::Result<Body> {
    let file = tokio::fs::File::open(path).await?;
    Ok(streaming_body(file, None))
}

async fn ephemeral_streaming_file_body(
    path: &str,
    cleanup: ManagedResultCleanup,
) -> std::io::Result<Body> {
    let file = tokio::fs::File::open(path).await?;
    Ok(streaming_body(file, Some(cleanup)))
}

fn streaming_body(mut file: tokio::fs::File, cleanup: Option<ManagedResultCleanup>) -> Body {
    let (sender, receiver) = mpsc::channel::<std::io::Result<Bytes>>(8);
    tokio::spawn(async move {
        loop {
            let mut buffer = vec![0_u8; 64 * 1024];
            match file.read(&mut buffer).await {
                Ok(0) => break,
                Ok(read) => {
                    buffer.truncate(read);
                    if sender.send(Ok(Bytes::from(buffer))).await.is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(error)).await;
                    break;
                }
            }
        }
        drop(file);
        drop(cleanup);
    });
    Body::from_stream(ReceiverStream::new(receiver))
}

fn direct_media_response(
    result: InferenceResult,
    response_format: DirectResponseFormat,
) -> Result<DirectMediaResponse, String> {
    let data = result
        .outputs
        .iter()
        .map(|output| direct_media_output(result.task, output, response_format))
        .collect::<Result<Vec<_>, _>>()?;
    if data.is_empty() {
        return Err(
            if matches!(
                result.task,
                InferenceTask::SpeechToText | InferenceTask::SpeechTranslation
            ) {
                "speech transcription backend did not produce text".to_string()
            } else {
                "inference backend did not produce an output".to_string()
            },
        );
    }
    Ok(DirectMediaResponse {
        created: result.created_unix,
        data,
        werk: result,
    })
}

fn direct_media_output(
    task: InferenceTask,
    output: &OutputMetadata,
    response_format: DirectResponseFormat,
) -> Result<DirectMediaOutput, String> {
    let text = if matches!(
        task,
        InferenceTask::SpeechToText | InferenceTask::SpeechTranslation
    ) {
        let body = fs::read_to_string(&output.path)
            .map_err(|error| format!("failed to read speech text output: {error}"))?;
        if output.mime_type == "application/json" {
            serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|value| {
                    value
                        .get("text")
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                })
                .or(Some(body))
        } else {
            Some(body)
        }
    } else {
        None
    };
    let (url, b64_json) = if text.is_some() {
        (None, None)
    } else {
        match response_format {
            DirectResponseFormat::Url => (Some(format!("/v1/outputs/{}", output.id)), None),
            DirectResponseFormat::Base64 => {
                let bytes = fs::read(&output.path).map_err(|error| {
                    format!("failed to read output for base64 response: {error}")
                })?;
                (None, Some(encode_base64(&bytes)))
            }
        }
    };
    Ok(DirectMediaOutput {
        id: output.id.clone(),
        url,
        b64_json,
        text,
        mime_type: output.mime_type.clone(),
        size_bytes: output.size_bytes,
        width: output.width,
        height: output.height,
        duration: output.duration,
    })
}

pub(in crate::api) fn encode_base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let first = chunk[0];
        let second = chunk.get(1).copied().unwrap_or(0);
        let third = chunk.get(2).copied().unwrap_or(0);
        encoded.push(TABLE[(first >> 2) as usize] as char);
        encoded.push(TABLE[(((first & 0x03) << 4) | (second >> 4)) as usize] as char);
        if chunk.len() > 1 {
            encoded.push(TABLE[(((second & 0x0f) << 2) | (third >> 6)) as usize] as char);
        } else {
            encoded.push('=');
        }
        if chunk.len() > 2 {
            encoded.push(TABLE[(third & 0x3f) as usize] as char);
        } else {
            encoded.push('=');
        }
    }
    encoded
}
