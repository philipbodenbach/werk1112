use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use serde_json::json;
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};
use tokio_stream::{StreamExt, once};

use crate::{
    backend::{GenerateRequest, GenerateResponse, GenerateStreamEvent, GeneratedAssistantMessage},
    model_store::{ModelManifest, unix_ts},
    openai::{
        AssistantMessage, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse,
        ModelListResponse, ModelObject, Usage,
    },
};

use super::{response::api_error, state::ApiState};

pub(super) async fn models_handler(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    match state.store.list() {
        Ok(manifests) => {
            state.log_verbose(format!(
                "[werk serve] GET /v1/models -> {} model(s)",
                manifests.len()
            ));
            let data = manifests.into_iter().map(model_object).collect();
            Json(ModelListResponse {
                object: "list",
                data,
            })
            .into_response()
        }
        Err(err) => api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string(), None),
    }
}

pub(super) async fn model_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    match state.store.get(&id) {
        Ok(manifest) => Json(model_object(manifest)).into_response(),
        Err(error) => api_error(
            StatusCode::NOT_FOUND,
            error.to_string(),
            Some("model".to_string()),
        ),
    }
}

fn model_object(manifest: ModelManifest) -> ModelObject {
    ModelObject {
        id: manifest.id,
        object: "model",
        created: manifest.created_unix,
        owned_by: "local",
    }
}

pub(super) async fn chat_completions_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let prepared = match super::generation::prepare(
        state,
        request,
        super::generation::ContextPolicy::Trim,
        "/v1/chat/completions",
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(error) => return error.openai_response(),
    };
    if prepared.stream {
        stream_chat_response(
            prepared.state,
            prepared.manifest,
            prepared.request,
            prepared.explicit_runtime_options,
            prepared.include_usage,
        )
    } else {
        complete_chat_response(
            prepared.state,
            prepared.manifest,
            prepared.request,
            prepared.explicit_runtime_options,
        )
        .await
    }
}

async fn complete_chat_response(
    state: ApiState,
    manifest: ModelManifest,
    generate_request: GenerateRequest,
    explicit_runtime_options: bool,
) -> Response {
    let verbose = state.verbose;
    let model = manifest.id.clone();
    let result =
        super::generation::generate(state, manifest, generate_request, explicit_runtime_options)
            .await;

    match result {
        Ok(response) => {
            if verbose {
                eprintln!(
                    "[werk serve] complete model={} finish={} prompt_tokens={} completion_tokens={} total={} load={} eval_rate={}",
                    model,
                    response.finish_reason,
                    response.prompt_tokens,
                    response.completion_tokens,
                    format_duration(response.timings.total_seconds),
                    format_duration(response.timings.load_seconds),
                    format_token_rate(response.completion_tokens, response.timings.decode_seconds)
                );
                log_generation_phases(response.prompt_tokens, response.timings);
                log_backend_diagnostics(&response.backend_diagnostics);
            }
            let metadata = json!({"timings": response.timings, "backend_diagnostics": response.backend_diagnostics});
            let cached = response.timings.cached_prompt_tokens;
            let mut body = serde_json::to_value(to_chat_completion(model, response))
                .expect("serializable completion");
            body["werk"] = metadata;
            if let Some(cached) = cached {
                body["usage"]["prompt_tokens_details"] = json!({"cached_tokens": cached});
            }
            Json(body).into_response()
        }
        Err(err) => {
            eprintln!("[werk serve] complete model={model} -> error: {err}");
            api_error(StatusCode::BAD_REQUEST, err.to_string(), None)
        }
    }
}

fn stream_chat_response(
    state: ApiState,
    manifest: ModelManifest,
    generate_request: GenerateRequest,
    explicit_runtime_options: bool,
    include_usage: bool,
) -> Response {
    let model = manifest.id.clone();
    let created = unix_ts();
    let id = format!("chatcmpl-{created}");

    let role_id = id.clone();
    let role_model = model.clone();
    let role = once(Ok::<Event, Infallible>(
        Event::default().data(
            json!({
                "id": role_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": role_model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": null
                }]
            })
            .to_string(),
        ),
    ));

    let body_id = id.clone();
    let body_model = model.clone();
    let body_model_for_log = model.clone();
    let verbose = state.verbose;
    let body_stream = super::generation::generate_stream(
        &state,
        manifest,
        generate_request,
        explicit_runtime_options,
    );
    let final_usage = Arc::new(Mutex::new(None));
    let body_usage = final_usage.clone();
    let body = body_stream.map(move |event| {
            let data = match event {
                Ok(GenerateStreamEvent::TextChunk(text)) => json!({
                    "id": body_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": null
                        }]
                    }),
                Ok(GenerateStreamEvent::ToolCallDelta(tool_calls)) => json!({
                    "id": body_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": tool_calls},
                        "finish_reason": null
                    }]
                }),
                Ok(GenerateStreamEvent::Done {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                    timings,
                    backend_diagnostics,
                }) => {
                    if include_usage {
                        let mut usage = json!({
                            "id": body_id, "object": "chat.completion.chunk", "created": created,
                            "model": body_model, "choices": [],
                            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                      "total_tokens": prompt_tokens.saturating_add(completion_tokens)}
                        });
                        if let Some(cached) = timings.cached_prompt_tokens {
                            usage["usage"]["prompt_tokens_details"] = json!({"cached_tokens": cached});
                        }
                        *body_usage.lock().expect("stream usage mutex poisoned") = Some(usage);
                    }
                    if verbose {
                        eprintln!(
                            "[werk serve] stream model={} finish={} prompt_tokens={} completion_tokens={} total={} load={} eval_rate={}",
                            body_model_for_log,
                            finish_reason,
                            prompt_tokens,
                            completion_tokens,
                            format_duration(timings.total_seconds),
                            format_duration(timings.load_seconds),
                            format_token_rate(completion_tokens, timings.decode_seconds)
                        );
                        log_generation_phases(prompt_tokens, timings);
                        log_backend_diagnostics(&backend_diagnostics);
                    }
                    json!({
                        "id": body_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body_model,
                        "werk": {"timings": timings, "backend_diagnostics": backend_diagnostics},
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason
                        }]
                    })
                }
                Err(message) => {
                    eprintln!("[werk serve] stream model={} -> error: {message}", body_model_for_log);
                    json!({
                        "error": {
                            "message": message,
                            "type": "invalid_request_error",
                            "param": null,
                            "code": null
                        }
                    })
                }
            };
            Ok::<Event, Infallible>(Event::default().data(data.to_string()))
        });

    let done = once(Ok::<Event, Infallible>(Event::default().data("[DONE]")));
    let usage = once(()).filter_map(move |_| {
        final_usage
            .lock()
            .expect("stream usage mutex poisoned")
            .take()
            .map(|value: serde_json::Value| {
                Ok::<Event, Infallible>(Event::default().data(value.to_string()))
            })
    });
    let stream = role.chain(body).chain(usage).chain(done);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn format_duration(seconds: f64) -> String {
    let seconds = seconds.max(0.0);
    if seconds >= 1.0 {
        trim_float(format!("{seconds:.6}")) + "s"
    } else if seconds >= 0.001 {
        trim_float(format!("{:.4}", seconds * 1000.0)) + "ms"
    } else {
        trim_float(format!("{:.3}", seconds * 1_000_000.0)) + "us"
    }
}

fn format_token_rate(tokens: usize, seconds: f64) -> String {
    if seconds <= 0.0 {
        return "-".to_string();
    }
    format!("{:.2} tok/s", tokens as f64 / seconds)
}

fn trim_float(mut value: String) -> String {
    while value.contains('.') && value.ends_with('0') {
        value.pop();
    }
    if value.ends_with('.') {
        value.pop();
    }
    value
}

fn log_generation_phases(prompt_tokens: usize, timings: crate::backend::GenerationTimings) {
    let Some(cached) = timings.cached_prompt_tokens.filter(|n| *n <= prompt_tokens) else {
        return;
    };
    eprintln!(
        "[werk serve] phases {}",
        json!({
            "first_token_seconds": timings.first_token_seconds,
            "prompt_seconds": timings.prompt_seconds,
            "decode_seconds": timings.decode_seconds,
            "cached_prompt_tokens": cached,
            "evaluated_prompt_tokens": prompt_tokens - cached,
        })
    );
}

fn log_backend_diagnostics(diagnostics: &[String]) {
    for diagnostic in diagnostics {
        eprintln!("[werk serve]   {diagnostic}");
    }
}

fn to_chat_completion(model: String, response: GenerateResponse) -> ChatCompletionResponse {
    let created = unix_ts();
    let GeneratedAssistantMessage {
        content,
        tool_calls,
    } = response
        .assistant_message
        .unwrap_or(GeneratedAssistantMessage {
            content: Some(response.text),
            tool_calls: None,
        });
    ChatCompletionResponse {
        id: format!("chatcmpl-{created}"),
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content,
                tool_calls,
            },
            finish_reason: response.finish_reason,
        }],
        usage: Usage {
            prompt_tokens: response.prompt_tokens,
            completion_tokens: response.completion_tokens,
            total_tokens: response.prompt_tokens + response.completion_tokens,
        },
    }
}
