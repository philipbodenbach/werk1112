//! Shared preparation and execution; wire protocols own serialization and errors.
use super::state::ApiState;
use crate::{
    backend::{
        GenerateRequest, GenerateResponse, GenerateStream, StreamGranularity, ToolCallingConfig,
    },
    capabilities::InferenceTask,
    model_store::ModelManifest,
    openai::{
        ChatCompletionRequest, ChatTemplateOptions, ChatTemplateSource,
        generation_messages_for_prompt, image_urls_from_messages,
        messages_to_prompt_for_model_with_template,
    },
};
use axum::http::StatusCode;
const DEFAULT_LLAMA_CONTEXT_SIZE: usize = 4096;

pub(super) struct GenerationError {
    pub status: StatusCode,
    pub message: String,
    pub param: Option<String>,
    pub code: Option<String>,
}
impl GenerationError {
    fn new(status: StatusCode, message: String, param: Option<String>) -> Self {
        Self {
            status,
            message,
            param,
            code: None,
        }
    }
    fn with_code(
        status: StatusCode,
        message: String,
        param: Option<String>,
        code: Option<String>,
    ) -> Self {
        Self {
            status,
            message,
            param,
            code,
        }
    }
    pub fn openai_response(self) -> axum::response::Response {
        super::response::api_error_with_code(self.status, self.message, self.param, self.code)
    }
}
#[derive(Clone, Copy)]
pub(super) enum ContextPolicy {
    Trim,
    Reject,
    Count,
}
pub(super) struct Prepared {
    pub api_options: std::collections::BTreeMap<String, serde_json::Value>,
    pub state: ApiState,
    pub manifest: ModelManifest,
    pub request: GenerateRequest,
    pub explicit_runtime_options: bool,
    pub stream: bool,
    pub include_usage: bool,
}
pub(super) async fn prepare(
    mut state: ApiState,
    mut request: ChatCompletionRequest,
    context_policy: ContextPolicy,
    endpoint: &str,
    headers: &axum::http::HeaderMap,
) -> Result<Prepared, GenerationError> {
    if let Err(error) =
        super::extended::validate(&request.extra, endpoint.starts_with("/v1/messages"))
    {
        return Err(GenerationError::new(StatusCode::BAD_REQUEST, error, None));
    }
    super::extended::normalize(&mut request.extra);
    if let Some(options) = &request.werk
        && let Err(error) = options.validate()
    {
        return Err(GenerationError::new(
            StatusCode::BAD_REQUEST,
            error.to_string(),
            Some("werk.omlx".into()),
        ));
    }
    let model_id = match request.model.as_deref().or(state.default_model.as_deref()) {
        Some(model) => model,
        None => {
            return Err(GenerationError::new(
                StatusCode::BAD_REQUEST,
                "request must include model, or start the server with --model <id>".to_string(),
                Some("model".to_string()),
            ));
        }
    };

    let manifest = match state.store.get(model_id) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("[werk serve] POST {endpoint} model={model_id} -> 404");
            return Err(GenerationError::new(
                StatusCode::NOT_FOUND,
                err.to_string(),
                Some("model".to_string()),
            ));
        }
    };

    if !manifest.metadata.tasks.is_empty()
        && !manifest.supports_task(InferenceTask::TextGeneration)
        && !manifest.supports_task(InferenceTask::ImageUnderstanding)
    {
        let message = if manifest.supports_task(InferenceTask::ImageGeneration) {
            format!(
                "model '{}' is an image-generation model and cannot be used with {endpoint}; use /v1/images/generations instead",
                manifest.id
            )
        } else {
            format!(
                "model '{}' does not declare text-generation or image-understanding and cannot be used with {endpoint}",
                manifest.id
            )
        };
        return Err(GenerationError::new(
            StatusCode::BAD_REQUEST,
            message,
            Some("model".to_string()),
        ));
    }

    if let Err(error) = super::documents::expand(&state, headers, &manifest, &mut request).await {
        return Err(GenerationError::new(
            StatusCode::BAD_REQUEST,
            error.to_string(),
            Some("messages".into()),
        ));
    }
    let max_tokens = request.max_completion_tokens();
    let context_size =
        effective_chat_context_size(state.chat_context_size, &manifest).or_else(|| {
            if matches!(context_policy, ContextPolicy::Reject) {
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
                    .and_then(|v| usize::try_from(v).ok())
                    .filter(|v| *v > 0)
            } else {
                None
            }
        });
    let removed_messages = if let Some(context_size) = context_size {
        match match context_policy {
            ContextPolicy::Trim => {
                trim_messages_to_context(&mut request.messages, context_size, max_tokens)
            }
            ContextPolicy::Reject => check_context(&request, context_size).map(|()| 0),
            ContextPolicy::Count => Ok(0),
        } {
            Ok(removed) => removed,
            Err(message) => {
                return Err(GenerationError::new(
                    StatusCode::BAD_REQUEST,
                    message,
                    Some("messages".to_string()),
                ));
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
    let runtime_options = request.werk.as_ref().filter(|options| !options.is_empty());
    let explicit_runtime_options = runtime_options.is_some();
    if let Some(options) = runtime_options {
        if !image_urls.is_empty() {
            return Err(GenerationError::with_code(
                StatusCode::BAD_REQUEST,
                "werk.omlx options currently support text chat only".into(),
                Some("werk.omlx".into()),
                Some("unsupported_chat_options".into()),
            ));
        }
        let backend = state.backend.clone();
        let selected_model = manifest.clone();
        let options = options.clone();
        // Compatibility probing may invoke Python. Keep it off the async
        // executor, and never modify the process environment for a request.
        match tokio::task::spawn_blocking(move || {
            backend.with_chat_options(&selected_model, &options)
        })
        .await
        {
            Ok(Ok(backend)) => state.backend = backend,
            result => {
                let message = match result {
                    Ok(Err(error)) => format!("{error:#}"),
                    Err(error) => format!("chat runtime configuration failed: {error}"),
                    Ok(Ok(_)) => unreachable!(),
                };
                return Err(GenerationError::with_code(
                    StatusCode::BAD_REQUEST,
                    message,
                    Some("werk.omlx".into()),
                    Some("unsupported_chat_options".into()),
                ));
            }
        }
    }
    let requires_tool_calling = request.requires_tool_calling();
    if requires_tool_calling
        && !state
            .backend
            .supports_tool_calling(&manifest, !image_urls.is_empty())
    {
        return Err(GenerationError::with_code(
            StatusCode::BAD_REQUEST,
            "the configured backend does not support OpenAI tool calling; use --backend vllm or --backend omlx with a compatible model and native tool parser"
                .to_string(),
            Some(tool_calling_parameter(&request).to_string()),
            Some("unsupported_tool_calling".to_string()),
        ));
    }
    let stream = request.stream.unwrap_or(false);
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    if state.verbose {
        let tools = request.tools.as_deref().unwrap_or_default();
        // Clients can attach a large tool catalog to a single short message.
        // Report its size without logging schemas, prompts or credentials.
        let tool_schema_bytes = if tools.is_empty() {
            0
        } else {
            serde_json::to_vec(tools)
                .map(|bytes| bytes.len())
                .unwrap_or_default()
        };
        state.log_verbose(format!(
            "[werk serve] POST {endpoint} model={} stream={} messages={} images={} max_tokens={} tools={} tool_schema_bytes={}",
            manifest.id,
            yes_no(stream),
            request.messages.len(),
            image_urls.len(),
            max_tokens,
            tools.len(),
            tool_schema_bytes,
        ));
    }
    let prompt_options = match if explicit_runtime_options {
        // These options select the native oMLX text route. A resolver for the
        // server's default route must not template or redirect this request.
        Ok(ChatTemplateOptions {
            default_source: ChatTemplateSource::Model,
            model_template_preferred: true,
            override_name: Some("model"),
        })
    } else {
        state.prompt_options(&manifest, !image_urls.is_empty())
    } {
        Ok(options) => options,
        Err(err) => {
            eprintln!(
                "[werk serve] POST {endpoint} model={} -> routing error: {err}",
                manifest.id
            );
            return Err(GenerationError::new(
                StatusCode::BAD_REQUEST,
                err.to_string(),
                None,
            ));
        }
    };
    let prompt =
        messages_to_prompt_for_model_with_template(&manifest, &request.messages, prompt_options);
    let generation_messages = if requires_tool_calling {
        request.messages.clone()
    } else {
        generation_messages_for_prompt(&prompt, request.messages.clone())
    };
    let mut stop = prompt.stop;
    stop.extend(request.stop_strings());
    let tool_config = (request.tools.is_some()
        || request.tool_choice.is_some()
        || request.parallel_tool_calls.is_some())
    .then_some(ToolCallingConfig {
        tools: request.tools,
        tool_choice: request.tool_choice,
        parallel_tool_calls: request.parallel_tool_calls,
    });

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
        tool_config,
    };

    Ok(Prepared {
        api_options: request.extra,
        state,
        manifest,
        request: generate_request,
        explicit_runtime_options,
        stream,
        include_usage,
    })
}

// Admission estimate, not an exact tokenizer. Includes tool schemas and history;
// never drop messages from a protocol with paired tool-use/result blocks.
fn check_context(request: &ChatCompletionRequest, context_size: usize) -> Result<(), String> {
    let has_images = request.messages.iter().any(|m|matches!(&m.content,
        Some(crate::openai::MessageContent::Parts(parts)) if parts.iter().any(|p|p.image_url.is_some())));
    let history = if has_images {
        // Encoded pixels are not text tokens. Use the shared visual estimate,
        // still accounting for tool arguments/identities without copying image data.
        let mut bytes = estimate_message_tokens(&request.messages).saturating_mul(3);
        for message in &request.messages {
            bytes = bytes.saturating_add(
                serde_json::to_vec(&(
                    &message.role,
                    &message.name,
                    &message.tool_calls,
                    &message.tool_call_id,
                ))
                .map_err(|e| e.to_string())?
                .len(),
            );
        }
        bytes
    } else {
        serde_json::to_vec(&request.messages)
            .map_err(|e| e.to_string())?
            .len()
    };
    let tools = request
        .tools
        .as_ref()
        .map(serde_json::to_vec)
        .transpose()
        .map_err(|e| e.to_string())?
        .map_or(0, |v| v.len());
    let estimate = history
        .saturating_add(tools)
        .div_ceil(3)
        .saturating_add(16usize.saturating_mul(request.messages.len()))
        .saturating_add(request.max_completion_tokens())
        .saturating_add(64);
    if estimate > context_size {
        return Err(format!(
            "estimated prompt and response exceed the {context_size}-token context; reduce messages, tools or max_tokens"
        ));
    }
    Ok(())
}

pub(super) async fn generate(
    state: ApiState,
    manifest: ModelManifest,
    request: GenerateRequest,
    explicit_runtime_options: bool,
) -> anyhow::Result<GenerateResponse> {
    let mut guard = state.telemetry.begin(&manifest.id);
    let session = match select_session(&state, &manifest, &request, explicit_runtime_options) {
        Ok(session) => session,
        Err(error) => {
            guard.error();
            return Err(error);
        }
    };
    tokio::task::spawn_blocking(move || {
        let result = match session {
            Some(session) => session.generate(request),
            None => state.backend.generate(&manifest, request),
        };
        match &result {
            Ok(response) => guard.complete(response),
            Err(_) => guard.error(),
        }
        result
    })
    .await
    .map_err(|e| anyhow::anyhow!("generation task failed: {e}"))?
}

pub(super) fn generate_stream(
    state: &ApiState,
    manifest: ModelManifest,
    request: GenerateRequest,
    explicit_runtime_options: bool,
) -> GenerateStream {
    let guard = state.telemetry.begin(&manifest.id);
    let stream = match select_session(state, &manifest, &request, explicit_runtime_options) {
        Ok(Some(session)) => session.generate_stream(request),
        Ok(None) => state.backend.generate_stream(manifest, request),
        Err(e) => Box::pin(tokio_stream::iter(vec![Err(e.to_string())])),
    };
    crate::observability::observe_stream(stream, guard)
}
fn select_session(
    state: &ApiState,
    manifest: &ModelManifest,
    request: &GenerateRequest,
    explicit_runtime_options: bool,
) -> anyhow::Result<Option<std::sync::Arc<dyn crate::backend::ChatGenerationSession>>> {
    if !request.image_urls.is_empty() || request.requires_tool_calling() || explicit_runtime_options
    {
        Ok(None)
    } else {
        state.chat_session(manifest, request.seed)
    }
}
fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}
fn tool_calling_parameter(request: &ChatCompletionRequest) -> &'static str {
    if request.tools.is_some() {
        "tools"
    } else if request.tool_choice.is_some() {
        "tool_choice"
    } else if request.parallel_tool_calls.is_some() {
        "parallel_tool_calls"
    } else {
        "messages"
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
            message
                .content
                .as_ref()
                .map(estimate_content_tokens)
                .unwrap_or_default()
                .saturating_add(16)
        })
        .sum::<usize>()
        + 16
}

fn estimate_content_tokens(content: &crate::openai::MessageContent) -> usize {
    use crate::openai::{ImageUrlSpec, MessageContent};

    match content {
        MessageContent::Text(text) => text.len().div_ceil(3),
        MessageContent::Parts(parts) => parts.iter().fold(0usize, |tokens, part| {
            let part_tokens = match part.kind.as_str() {
                "text" => part.text.as_deref().unwrap_or_default().len().div_ceil(3),
                "image_url" | "input_image" => {
                    let detail = part.image_url.as_ref().and_then(|image| match image {
                        ImageUrlSpec::Object(image) => image.detail.as_deref(),
                        ImageUrlSpec::Url(_) => None,
                    });
                    match detail {
                        Some("low") => 256,
                        Some("high") => 2048,
                        _ => 1024,
                    }
                }
                _ => 0,
            };
            tokens.saturating_add(part_tokens)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{ContentPart, ImageUrlPart, ImageUrlSpec, MessageContent};

    #[test]
    fn anthropic_visual_admission_counts_images_instead_of_base64_bytes_and_keeps_tool_cost() {
        let mut request:ChatCompletionRequest=serde_json::from_value(serde_json::json!({
            "max_tokens":128,"messages":[{"role":"user","content":[
                {"type":"text","text":"Describe the page"},
                {"type":"image_url","image_url":format!("data:image/png;base64,{}","A".repeat(1024*1024))}
            ]}]})).unwrap();
        assert!(check_context(&request, 4096).is_ok());
        request.messages.push(serde_json::from_value(serde_json::json!({"role":"assistant","tool_calls":[
            {"id":"call_1","type":"function","function":{"name":"tool","arguments":"x".repeat(20000)}}
        ]})).unwrap());
        assert!(check_context(&request, 4096).is_err());
    }

    #[test]
    fn multimodal_context_estimate_accounts_for_image_detail() {
        let low = MessageContent::Parts(vec![ContentPart {
            file: None,
            kind: "image_url".to_string(),
            text: None,
            image_url: Some(ImageUrlSpec::Object(ImageUrlPart {
                url: "data:image/png;base64,AAAA".to_string(),
                detail: Some("low".to_string()),
            })),
        }]);
        let high = MessageContent::Parts(vec![ContentPart {
            file: None,
            kind: "input_image".to_string(),
            text: None,
            image_url: Some(ImageUrlSpec::Object(ImageUrlPart {
                url: "data:image/png;base64,AAAA".to_string(),
                detail: Some("high".to_string()),
            })),
        }]);

        assert_eq!(estimate_content_tokens(&low), 256);
        assert_eq!(estimate_content_tokens(&high), 2048);
    }
}
