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
use std::convert::Infallible;
use tokio_stream::{StreamExt, once};

use crate::{
    backend::{GenerateRequest, GenerateResponse, GenerateStreamEvent, StreamGranularity},
    capabilities::InferenceTask,
    model_store::{ModelManifest, unix_ts},
    openai::{
        AssistantMessage, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse,
        ModelListResponse, ModelObject, Usage, generation_messages_for_prompt,
        image_urls_from_messages, messages_to_prompt_for_model_with_template,
    },
};

use super::{response::api_error, state::ApiState};

const DEFAULT_LLAMA_CONTEXT_SIZE: usize = 4096;

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
    Json(mut request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let model_id = match request.model.as_deref().or(state.default_model.as_deref()) {
        Some(model) => model,
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "request must include model, or start the server with --model <id>".to_string(),
                Some("model".to_string()),
            );
        }
    };

    let manifest = match state.store.get(model_id) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("[werk serve] POST /v1/chat/completions model={model_id} -> 404");
            return api_error(
                StatusCode::NOT_FOUND,
                err.to_string(),
                Some("model".to_string()),
            );
        }
    };

    if !manifest.metadata.tasks.is_empty()
        && !manifest.supports_task(InferenceTask::TextGeneration)
        && !manifest.supports_task(InferenceTask::ImageUnderstanding)
    {
        let message = if manifest.supports_task(InferenceTask::ImageGeneration) {
            format!(
                "model '{}' is an image-generation model and cannot be used with /v1/chat/completions; use /v1/images/generations instead",
                manifest.id
            )
        } else {
            format!(
                "model '{}' does not declare text-generation or image-understanding and cannot be used with /v1/chat/completions",
                manifest.id
            )
        };
        return api_error(StatusCode::BAD_REQUEST, message, Some("model".to_string()));
    }

    let max_tokens = request.max_completion_tokens();
    let context_size = effective_chat_context_size(state.chat_context_size, &manifest);
    let removed_messages = if let Some(context_size) = context_size {
        match trim_messages_to_context(&mut request.messages, context_size, max_tokens) {
            Ok(removed) => removed,
            Err(message) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    message,
                    Some("messages".to_string()),
                );
            }
        }
    } else {
        0
    };
    if removed_messages > 0 {
        eprintln!(
            "[werk serve] chat context model={} removed_messages={} context_size={} max_tokens={}",
            manifest.id,
            removed_messages,
            context_size.unwrap_or_default(),
            max_tokens
        );
    }

    let image_urls = image_urls_from_messages(&request.messages);
    let stream = request.stream.unwrap_or(false);
    state.log_verbose(format!(
        "[werk serve] POST /v1/chat/completions model={} stream={} messages={} images={} max_tokens={}",
        manifest.id,
        yes_no(stream),
        request.messages.len(),
        image_urls.len(),
        request.max_completion_tokens()
    ));
    let prompt_options = match state.prompt_options(&manifest, !image_urls.is_empty()) {
        Ok(options) => options,
        Err(err) => {
            eprintln!(
                "[werk serve] POST /v1/chat/completions model={} -> routing error: {err}",
                manifest.id
            );
            return api_error(StatusCode::BAD_REQUEST, err.to_string(), None);
        }
    };
    let prompt =
        messages_to_prompt_for_model_with_template(&manifest, &request.messages, prompt_options);
    let generation_messages = generation_messages_for_prompt(&prompt, request.messages.clone());
    let mut stop = prompt.stop;
    stop.extend(request.stop_strings());

    let generate_request = GenerateRequest {
        prompt: prompt.prompt,
        messages: generation_messages,
        image_urls,
        max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        stop,
        seed: request.seed,
        stream_granularity: StreamGranularity::Chunk,
        verbose: state.verbose,
        debug: false,
    };

    if stream {
        stream_chat_response(state, manifest, generate_request)
    } else {
        complete_chat_response(state, manifest, generate_request).await
    }
}

fn effective_chat_context_size(configured: usize, manifest: &ModelManifest) -> Option<usize> {
    if manifest.format != crate::model_store::ModelFormat::Gguf {
        return None;
    }
    if configured != 0 {
        return Some(configured);
    }
    manifest
        .metadata
        .parameter_constraints
        .get("max_position_embeddings")
        .or_else(|| {
            manifest
                .metadata
                .parameter_constraints
                .get("model_max_length")
        })
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .or(Some(DEFAULT_LLAMA_CONTEXT_SIZE))
}

fn trim_messages_to_context(
    messages: &mut Vec<crate::openai::ChatMessage>,
    context_size: usize,
    max_tokens: usize,
) -> Result<usize, String> {
    const SAFETY_TOKENS: usize = 64;
    let reserved = max_tokens
        .checked_add(SAFETY_TOKENS)
        .ok_or_else(|| "chat response token reserve overflowed".to_string())?;
    if reserved >= context_size {
        return Err(format!(
            "response budget ({max_tokens} tokens) leaves no prompt space in the {context_size}-token context; reduce max_tokens or increase the server --ctx-size"
        ));
    }
    let prompt_budget = context_size - reserved;
    let mut removed = 0;

    while estimate_message_tokens(messages) > prompt_budget {
        let Some(start) = messages
            .iter()
            .position(|message| !message.role.eq_ignore_ascii_case("system"))
        else {
            return Err(format!(
                "system prompt is too large for the {context_size}-token context"
            ));
        };
        if start + 1 >= messages.len() {
            return Err(format!(
                "current message is too large for the {context_size}-token context after reserving {max_tokens} response tokens"
            ));
        }
        let remove_count = if messages
            .get(start + 1)
            .is_some_and(|message| message.role.eq_ignore_ascii_case("assistant"))
        {
            2
        } else {
            1
        };
        messages.drain(start..start + remove_count);
        removed += remove_count;
    }
    Ok(removed)
}

fn estimate_message_tokens(messages: &[crate::openai::ChatMessage]) -> usize {
    messages
        .iter()
        .map(|message| {
            let content_bytes = message
                .content
                .as_ref()
                .map(crate::openai::MessageContent::as_text)
                .map(|content| content.len())
                .unwrap_or_default();
            content_bytes.div_ceil(3) + 16
        })
        .sum::<usize>()
        + 16
}

async fn complete_chat_response(
    state: ApiState,
    manifest: ModelManifest,
    generate_request: GenerateRequest,
) -> Response {
    let backend = state.backend.clone();
    let verbose = state.verbose;
    let model = manifest.id.clone();
    let chat_session = match state.chat_session(&manifest, generate_request.seed) {
        Ok(session) => session,
        Err(err) => {
            eprintln!("[werk serve] complete model={model} -> session error: {err}");
            return api_error(StatusCode::BAD_REQUEST, err.to_string(), None);
        }
    };
    let result = tokio::task::spawn_blocking(move || {
        if let Some(session) = chat_session.as_ref() {
            session.generate(generate_request)
        } else {
            backend.generate(&manifest, generate_request)
        }
    })
    .await
    .map_err(|err| anyhow::anyhow!("generation task failed: {err}"))
    .and_then(|inner| inner);

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
                log_backend_diagnostics(&response.backend_diagnostics);
            }
            Json(to_chat_completion(model, response)).into_response()
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
    let body_stream = match state.chat_session(&manifest, generate_request.seed) {
        Ok(Some(session)) => session.generate_stream(generate_request),
        Ok(None) => state.backend.generate_stream(manifest, generate_request),
        Err(err) => Box::pin(tokio_stream::iter(vec![Err(err.to_string())])),
    };
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
                Ok(GenerateStreamEvent::Done {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                    timings,
                    backend_diagnostics,
                }) => {
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
                        log_backend_diagnostics(&backend_diagnostics);
                    }
                    json!({
                        "id": body_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body_model,
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
    let stream = role.chain(body).chain(done);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
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

fn log_backend_diagnostics(diagnostics: &[String]) {
    for diagnostic in diagnostics {
        eprintln!("[werk serve]   {diagnostic}");
    }
}

fn to_chat_completion(model: String, response: GenerateResponse) -> ChatCompletionResponse {
    let created = unix_ts();
    ChatCompletionResponse {
        id: format!("chatcmpl-{created}"),
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: response.text,
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
