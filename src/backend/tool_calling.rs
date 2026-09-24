//! Tool transport for generators exposing only text. The protocol is explicit:
//! ordinary JSON prose is never promoted to an executable function call.
//! Native chat adapters should keep using their native tool transport.

use std::{
    collections::VecDeque,
    pin::Pin,
    task::{Context, Poll},
};

use anyhow::{Context as _, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::Stream;

use super::{
    GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent,
    GeneratedAssistantMessage, ToolCallingConfig,
};
use crate::{
    model_store::ModelManifest,
    openai::{
        ChatCompletionFunctionCall, ChatCompletionFunctionCallDelta, ChatCompletionToolCall,
        ChatCompletionToolCallDelta, ChatMessage, ContentPart, MessageContent, ToolChoice,
        ToolChoiceMode, messages_to_prompt_for_model,
    },
};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
// Bound validation buffers independently of whether upstream honors max_tokens.
const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
const DIAGNOSTIC: &str =
    "tool calling: werk-tool-call-v1 generic prompt protocol; model compliance is not guaranteed";

pub(crate) fn generate(
    manifest: &ModelManifest,
    request: GenerateRequest,
    run: impl FnOnce(GenerateRequest) -> Result<GenerateResponse>,
) -> Result<GenerateResponse> {
    if !request.requires_tool_calling() {
        return run(request);
    }
    let (request, policy) = prepare(manifest, request)?;
    policy.finish(run(request)?)
}

pub(crate) fn generate_stream(
    manifest: &ModelManifest,
    request: GenerateRequest,
    run: impl FnOnce(GenerateRequest) -> GenerateStream,
) -> GenerateStream {
    if !request.requires_tool_calling() {
        return run(request);
    }
    match prepare(manifest, request) {
        Ok((request, policy)) => Box::pin(ToolStream {
            upstream: Some(run(request)),
            policy,
            text: String::new(),
            pending: VecDeque::new(),
        }),
        Err(error) => Box::pin(tokio_stream::iter([Err(error.to_string())])),
    }
}

#[derive(Debug)]
pub(super) struct Policy {
    names: Vec<String>,
    choice: ToolChoice,
    parallel: bool,
}

impl Policy {
    pub(super) fn finish(&self, mut response: GenerateResponse) -> Result<GenerateResponse> {
        ensure!(
            response
                .assistant_message
                .as_ref()
                .is_none_or(|message| message.tool_calls.as_ref().is_none_or(Vec::is_empty)),
            "generic tool protocol received unexpected native tool calls"
        );
        let output = response
            .assistant_message
            .as_ref()
            .and_then(|message| message.content.as_deref())
            .unwrap_or(&response.text);
        let message = self.parse(output)?;
        validate_finish_reason(&message, &response.finish_reason)?;
        response.text = message.content.clone().unwrap_or_default();
        if message.tool_calls.is_some() {
            response.finish_reason = "tool_calls".into();
        }
        response.assistant_message = Some(message);
        response.backend_diagnostics.push(DIAGNOSTIC.into());
        Ok(response)
    }

    fn new(config: &ToolCallingConfig) -> Result<Self> {
        let mut names = Vec::new();
        for tool in config.tools.iter().flatten() {
            ensure!(
                tool.function.strict != Some(true),
                "strict function schemas require native constrained tool decoding; the generic tool protocol cannot guarantee strict schema conformance"
            );
            ensure!(
                tool.kind == "function",
                "generic tool protocol supports function tools only"
            );
            ensure!(
                !tool.function.name.is_empty(),
                "tool function name must not be empty"
            );
            ensure!(
                !names.contains(&tool.function.name),
                "duplicate tool function name: {}",
                tool.function.name
            );
            names.push(tool.function.name.clone());
        }
        let choice = config
            .tool_choice
            .clone()
            .unwrap_or(ToolChoice::Mode(if names.is_empty() {
                ToolChoiceMode::None
            } else {
                ToolChoiceMode::Auto
            }));
        match &choice {
            ToolChoice::Mode(ToolChoiceMode::Required) => ensure!(
                !names.is_empty(),
                "tool_choice required needs at least one function tool"
            ),
            ToolChoice::Named(named) => {
                ensure!(
                    named.kind == "function",
                    "named tool_choice must have type function"
                );
                ensure!(
                    names.contains(&named.function.name),
                    "tool_choice selects unknown function: {}",
                    named.function.name
                );
            }
            _ => {}
        }
        Ok(Self {
            names,
            choice,
            parallel: config.parallel_tool_calls.unwrap_or(true),
        })
    }

    pub(super) fn parse(&self, output: &str) -> Result<GeneratedAssistantMessage> {
        ensure!(
            output.len() <= MAX_OUTPUT_BYTES,
            "generic tool output exceeds the {MAX_OUTPUT_BYTES}-byte validation limit"
        );
        let mut rest = output;
        let mut content = String::new();
        let mut calls = Vec::new();
        while let Some(start) = rest.find(OPEN) {
            let before = &rest[..start];
            ensure_no_markup(before)?;
            content.push_str(before);
            let body = &rest[start + OPEN.len()..];
            // Parse JSON before locating the closing marker: argument strings
            // may themselves contain literal </tool_call> text.
            let mut decoder = serde_json::Deserializer::from_str(body).into_iter::<WireCall>();
            let wire = decoder
                .next()
                .context("empty tool_call body")?
                .context("malformed JSON in tool_call")?;
            let after_json = body[decoder.byte_offset()..].trim_start();
            ensure!(
                after_json.starts_with(CLOSE),
                "tool_call is missing its closing </tool_call> marker"
            );
            rest = &after_json[CLOSE.len()..];
            ensure!(
                !matches!(self.choice, ToolChoice::Mode(ToolChoiceMode::None)),
                "model emitted a tool call despite tool_choice none"
            );
            ensure!(
                self.names.contains(&wire.name),
                "model emitted unknown tool function: {}",
                wire.name
            );
            if let ToolChoice::Named(named) = &self.choice {
                ensure!(
                    wire.name == named.function.name,
                    "model emitted {} instead of required function {}",
                    wire.name,
                    named.function.name
                );
            }
            ensure!(
                self.parallel || calls.is_empty(),
                "model emitted multiple calls despite parallel_tool_calls false"
            );
            let arguments = match wire.arguments {
                Value::String(encoded) => serde_json::from_str(&encoded)
                    .context("tool arguments string is not valid JSON")?,
                arguments => arguments,
            };
            ensure!(
                arguments.is_object(),
                "tool function arguments must be a JSON object"
            );
            calls.push(ChatCompletionToolCall {
                id: call_id()?,
                kind: "function".into(),
                function: ChatCompletionFunctionCall {
                    name: wire.name,
                    arguments: serde_json::to_string(&arguments)?,
                },
            });
        }
        ensure_no_markup(rest)?;
        content.push_str(rest);
        if matches!(
            self.choice,
            ToolChoice::Mode(ToolChoiceMode::Required) | ToolChoice::Named(_)
        ) {
            ensure!(
                !calls.is_empty(),
                "model did not emit the required tool call"
            );
        }
        let has_calls = !calls.is_empty();
        Ok(GeneratedAssistantMessage {
            content: if has_calls && content.trim().is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls: has_calls.then_some(calls),
        })
    }
}

// A syntactically complete first call can still belong to a truncated series.
// Publish executable calls only after the generator reports a normal stop.
pub(super) fn validate_finish_reason(
    message: &GeneratedAssistantMessage,
    finish_reason: &str,
) -> Result<()> {
    ensure!(
        message.tool_calls.as_ref().is_none_or(Vec::is_empty)
            || matches!(finish_reason, "stop" | "tool_calls"),
        "generic tool generation ended with {finish_reason:?} before successful tool completion"
    );
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCall {
    name: String,
    arguments: Value,
}

fn ensure_no_markup(text: &str) -> Result<()> {
    ensure!(
        !text.contains("<tool_") && !text.contains("</tool_"),
        "malformed or incomplete tool_call marker in model output"
    );
    Ok(())
}

fn call_id() -> Result<String> {
    let mut bytes = [0_u8; 16];
    getrandom::getrandom(&mut bytes)
        .map_err(|error| anyhow::anyhow!("cannot allocate tool call ID: {error}"))?;
    let mut id = String::from("call_");
    for byte in bytes {
        use std::fmt::Write;
        write!(&mut id, "{byte:02x}").expect("writing into a String cannot fail");
    }
    Ok(id)
}

// Admission uses the same byte-based estimate as the API context guard. Tool
// schema bytes and per-message history wrappers are accounted there separately.
pub(crate) fn generic_prompt_overhead_tokens(tool_choice: Option<&ToolChoice>) -> usize {
    let default_choice = ToolChoice::Mode(ToolChoiceMode::Auto);
    // The single-call instruction is the longer parallel-policy alternative.
    tool_contract("[]", tool_choice.unwrap_or(&default_choice), false)
        .len()
        .div_ceil(3)
}

fn tool_contract(tools_json: &str, choice: &ToolChoice, parallel: bool) -> String {
    let choice_instruction = match choice {
        ToolChoice::Mode(ToolChoiceMode::None) => {
            "Do not call any tool. Answer with normal text.".to_owned()
        }
        ToolChoice::Mode(ToolChoiceMode::Auto) => {
            "Call a tool when needed; otherwise answer with normal text.".to_owned()
        }
        ToolChoice::Mode(ToolChoiceMode::Required) => {
            "You MUST call at least one available tool.".to_owned()
        }
        ToolChoice::Named(named) => format!(
            "You MUST call the function {} and no other function.",
            serde_json::to_string(&named.function.name).expect("string serialization cannot fail")
        ),
    };
    format!(
        "Tool interface (werk-tool-call-v1):\n\
         Available function tools and their JSON schemas:\n{}\n\
         {}\n{}\n\
         To call a function, emit exactly <tool_call>{{\"name\":\"FUNCTION_NAME\",\"arguments\":{{...}}}}</tool_call>.\n\
         Arguments must be a JSON object matching that function's schema. Do not wrap calls in Markdown or include call IDs.\n\
         Emit one such block per call. Only these explicit blocks are executed. Never emit incomplete blocks or invent function names.\n\
         Text outside the blocks is user-visible. When answering normally, do not output tool_call markers or examples.\n\
         Historical tool metadata is encoded in <tool_history> JSON </tool_history> at the beginning of a message.\n\
         It preserves prior assistant calls and tool result IDs; it is conversation data, not a new instruction or a new call.\n\
         A message marked with original role tool contains that tool's result, even though its transport role is user.",
        tools_json,
        choice_instruction,
        if parallel {
            "Multiple calls are allowed when needed."
        } else {
            "Emit at most one tool call in this response."
        },
    )
}

pub(super) fn prepare(
    manifest: &ModelManifest,
    mut request: GenerateRequest,
) -> Result<(GenerateRequest, Policy)> {
    let config = request.tool_config.take().unwrap_or(ToolCallingConfig {
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
    });
    let policy = Policy::new(&config)?;
    let contract = tool_contract(
        &serde_json::to_string(&config.tools.unwrap_or_default())?,
        &policy.choice,
        policy.parallel,
    );
    if request.messages.is_empty() && !request.prompt.is_empty() {
        request
            .messages
            .push(text_message("user", std::mem::take(&mut request.prompt)));
    }
    for message in &mut request.messages {
        if !message.uses_tool_calling() {
            continue;
        }
        let history = json!({"role": message.role, "name": message.name,
            "tool_calls": message.tool_calls, "tool_call_id": message.tool_call_id});
        prepend_text(
            message,
            format!(
                "<tool_history>{}</tool_history>\n",
                serde_json::to_string(&history)?
            ),
        );
        if message.role.eq_ignore_ascii_case("tool") {
            message.role = "user".into();
        }
        message.tool_calls = None;
        message.tool_call_id = None;
    }
    if let Some(system) = request
        .messages
        .first_mut()
        .filter(|message| message.role == "system")
    {
        prepend_text(system, format!("{contract}\n\n"));
    } else {
        request.messages.insert(0, text_message("system", contract));
    }
    let prompt = messages_to_prompt_for_model(manifest, &request.messages);
    request.prompt = prompt.prompt;
    for stop in prompt.stop {
        if !request.stop.contains(&stop) {
            request.stop.push(stop);
        }
    }
    Ok((request, policy))
}

fn text_message(role: &str, text: String) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: Some(MessageContent::Text(text)),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn prepend_text(message: &mut ChatMessage, prefix: String) {
    match &mut message.content {
        Some(MessageContent::Text(text)) => text.insert_str(0, &prefix),
        Some(MessageContent::Parts(parts)) => parts.insert(
            0,
            ContentPart {
                kind: "text".into(),
                text: Some(prefix),
                image_url: None,
                file: None,
            },
        ),
        None => message.content = Some(MessageContent::Text(prefix)),
    }
}

struct ToolStream {
    // No producer task: dropping this stream immediately drops the upstream
    // receiver, allowing its normal cancellation mechanism to stop generation.
    upstream: Option<GenerateStream>,
    policy: Policy,
    text: String,
    pending: VecDeque<Result<GenerateStreamEvent, String>>,
}

impl ToolStream {
    fn finish(&mut self, event: GenerateStreamEvent) -> Result<()> {
        let message = self.policy.parse(&self.text)?;
        let GenerateStreamEvent::Done { finish_reason, .. } = &event else {
            bail!("generic tool stream received an invalid terminal event");
        };
        validate_finish_reason(&message, finish_reason)?;
        if let Some(text) = message.content.filter(|text| !text.is_empty()) {
            self.pending
                .push_back(Ok(GenerateStreamEvent::TextChunk(text)));
        }
        let has_calls = message.tool_calls.is_some();
        if let Some(calls) = message.tool_calls {
            self.pending
                .push_back(Ok(GenerateStreamEvent::ToolCallDelta(
                    calls
                        .into_iter()
                        .enumerate()
                        .map(|(index, call)| ChatCompletionToolCallDelta {
                            index,
                            id: Some(call.id),
                            kind: Some(call.kind),
                            function: Some(ChatCompletionFunctionCallDelta {
                                name: Some(call.function.name),
                                arguments: Some(call.function.arguments),
                            }),
                        })
                        .collect(),
                )));
        }
        if let GenerateStreamEvent::Done {
            mut finish_reason,
            prompt_tokens,
            completion_tokens,
            timings,
            mut backend_diagnostics,
        } = event
        {
            if has_calls {
                finish_reason = "tool_calls".into();
            }
            backend_diagnostics.push(DIAGNOSTIC.into());
            backend_diagnostics.push("generic tool stream buffered until validation".into());
            self.pending.push_back(Ok(GenerateStreamEvent::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
                timings,
                backend_diagnostics,
            }));
        } else {
            bail!("generic tool stream received an invalid terminal event");
        }
        Ok(())
    }
}

impl Stream for ToolStream {
    type Item = Result<GenerateStreamEvent, String>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let mut chunks_polled = 0;
        loop {
            if let Some(event) = this.pending.pop_front() {
                return Poll::Ready(Some(event));
            }
            let Some(upstream) = this.upstream.as_mut() else {
                return Poll::Ready(None);
            };
            let result = match upstream.as_mut().poll_next(context) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(result) => result,
            };
            match result {
                Some(Ok(GenerateStreamEvent::TextChunk(text))) => {
                    if text.len() > MAX_OUTPUT_BYTES.saturating_sub(this.text.len()) {
                        this.upstream = None;
                        this.text.clear();
                        return Poll::Ready(Some(Err(format!(
                            "generic tool output exceeds the {MAX_OUTPUT_BYTES}-byte validation limit"
                        ))));
                    }
                    this.text.push_str(&text);
                    chunks_polled += 1;
                    // Cooperate even if upstream yields ready chunks indefinitely.
                    if chunks_polled == 64 {
                        context.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                }
                Some(Ok(event @ GenerateStreamEvent::Done { .. })) => {
                    this.upstream = None;
                    if let Err(error) = this.finish(event) {
                        return Poll::Ready(Some(Err(error.to_string())));
                    }
                }
                Some(Ok(GenerateStreamEvent::ToolCallDelta(_))) => {
                    this.upstream = None;
                    return Poll::Ready(Some(Err(
                        "generic tool stream received unexpected native tool calls".into(),
                    )));
                }
                Some(Err(error)) => {
                    this.upstream = None;
                    return Poll::Ready(Some(Err(error)));
                }
                None => {
                    this.upstream = None;
                    return Poll::Ready(Some(Err(
                        "tool generation stream ended before completion metadata".into(),
                    )));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{GenerationTimings, StreamGranularity},
        model_store::{ModelFormat, ModelSource},
        openai::{ChatCompletionTool, ImageUrlSpec, NamedToolChoice, NamedToolChoiceFunction},
    };
    use tokio_stream::StreamExt;

    fn manifest() -> ModelManifest {
        ModelManifest {
            id: "Qwen3-test".into(),
            source: ModelSource::LocalPath {
                path: "test".into(),
            },
            storage: Default::default(),
            format: ModelFormat::SafeTensors,
            architecture: Some("qwen3".into()),
            tokenizer_path: None,
            config_path: None,
            model_path: None,
            backend: "test".into(),
            created_unix: 0,
            files: Vec::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        }
    }
    fn tool(name: &str) -> ChatCompletionTool {
        serde_json::from_value(json!({"type":"function", "function": {"name":name,
            "description":"Read the weather", "parameters":{"type":"object",
                "properties":{"city":{"type":"string"}}, "required":["city"]}}}))
        .unwrap()
    }
    fn request() -> GenerateRequest {
        GenerateRequest {
            prompt: "original".into(),
            messages: vec![text_message("user", "Weather in Köln?".into())],
            image_urls: Vec::new(),
            max_tokens: 123,
            temperature: Some(0.3),
            top_p: Some(0.9),
            stop: vec!["END".into()],
            seed: Some(7),
            stream_granularity: StreamGranularity::Token,
            verbose: false,
            debug: false,
            tool_config: Some(ToolCallingConfig {
                tools: Some(vec![tool("weather"), tool("clock")]),
                tool_choice: None,
                parallel_tool_calls: None,
            }),
        }
    }
    fn response(text: &str) -> GenerateResponse {
        GenerateResponse {
            text: text.into(),
            assistant_message: None,
            prompt_tokens: 81,
            completion_tokens: 29,
            finish_reason: "stop".into(),
            timings: GenerationTimings {
                prompt_seconds: 0.3,
                decode_seconds: 0.9,
                total_seconds: 1.2,
                ..Default::default()
            },
            backend_diagnostics: vec!["backend metadata".into()],
        }
    }
    fn done() -> GenerateStreamEvent {
        let response = response("");
        GenerateStreamEvent::Done {
            finish_reason: response.finish_reason,
            prompt_tokens: response.prompt_tokens,
            completion_tokens: response.completion_tokens,
            timings: response.timings,
            backend_diagnostics: response.backend_diagnostics,
        }
    }
    fn call(name: &str) -> String {
        format!(
            "<tool_call>{{\"name\":\"{name}\",\"arguments\":{{\"city\":\"Köln\"}}}}</tool_call>"
        )
    }
    fn named(name: &str) -> ToolChoice {
        ToolChoice::Named(NamedToolChoice {
            kind: "function".into(),
            function: NamedToolChoiceFunction { name: name.into() },
        })
    }

    #[test]
    fn round_trip_preserves_schema_sampling_history_and_usage() {
        let mut input = request();
        input.messages.push(ChatMessage {
            role: "assistant".into(),
            content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Some(vec![ChatCompletionToolCall {
                id: "call_previous".into(),
                kind: "function".into(),
                function: ChatCompletionFunctionCall {
                    name: "weather".into(),
                    arguments: "{\"city\":\"Berlin\"}".into(),
                },
            }]),
        });
        input.messages.push(ChatMessage {
            role: "tool".into(),
            content: Some(MessageContent::Text("Sunny, 21°C".into())),
            name: Some("weather".into()),
            tool_calls: None,
            tool_call_id: Some("call_previous".into()),
        });
        let output = generate(&manifest(), input, |prepared| {
            assert!(!prepared.requires_tool_calling());
            for expected in [
                "<|im_start|>system",
                "\"properties\":{\"city\":{\"type\":\"string\"}}",
                "call_previous",
                "Sunny, 21°C",
                "Berlin",
            ] {
                assert!(prepared.prompt.contains(expected), "missing {expected}");
            }
            assert_eq!(prepared.messages.last().unwrap().role, "user");
            assert!(
                prepared
                    .messages
                    .last()
                    .unwrap()
                    .content
                    .as_ref()
                    .unwrap()
                    .as_text()
                    .contains("\"role\":\"tool\"")
            );
            assert_eq!(prepared.max_tokens, 123);
            assert_eq!(prepared.temperature, Some(0.3));
            assert_eq!(prepared.top_p, Some(0.9));
            assert_eq!(prepared.seed, Some(7));
            assert!(prepared.stop.contains(&"END".to_owned()));
            Ok(response(&call("weather")))
        })
        .unwrap();
        assert_eq!(output.finish_reason, "tool_calls");
        assert!(output.text.is_empty());
        let message = output.assistant_message.unwrap();
        assert!(message.content.is_none());
        let calls = message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            json!({"city":"Köln"})
        );
        assert_eq!((output.prompt_tokens, output.completion_tokens), (81, 29));
        assert_eq!(output.timings.total_seconds, 1.2);
        assert_eq!(output.backend_diagnostics[0], "backend metadata");
        assert!(output.backend_diagnostics.contains(&DIAGNOSTIC.to_string()));
    }

    #[test]
    fn historical_tool_round_trip_works_without_new_tool_definitions() {
        let first = generate(&manifest(), request(), |_| Ok(response(&call("weather")))).unwrap();
        let calls = first.assistant_message.unwrap().tool_calls.unwrap();
        let mut next = request();
        next.tool_config = None;
        next.messages.push(ChatMessage {
            role: "assistant".into(),
            content: None,
            name: None,
            tool_calls: Some(calls.clone()),
            tool_call_id: None,
        });
        next.messages.push(ChatMessage {
            role: "tool".into(),
            content: Some(MessageContent::Text("21 degrees".into())),
            name: None,
            tool_calls: None,
            tool_call_id: Some(calls[0].id.clone()),
        });
        let output = generate(&manifest(), next, |prepared| {
            assert!(prepared.prompt.contains(&calls[0].id));
            assert!(prepared.prompt.contains("21 degrees"));
            assert!(!prepared.requires_tool_calling());
            Ok(response("It is 21 degrees."))
        })
        .unwrap();
        assert_eq!(output.text, "It is 21 degrees.");
        assert_eq!(output.finish_reason, "stop");
    }

    #[test]
    fn image_parts_and_image_urls_survive_tool_preparation() {
        let mut input = request();
        let image = "data:image/png;base64,cGl4ZWxz";
        input.image_urls.push(image.into());
        let parts = vec![ContentPart {
            kind: "image_url".into(),
            text: None,
            image_url: Some(ImageUrlSpec::Url(image.into())),
            file: None,
        }];
        input.messages.push(ChatMessage {
            role: "tool".into(),
            content: Some(MessageContent::Parts(parts.clone())),
            name: None,
            tool_call_id: Some("call_camera".into()),
            tool_calls: None,
        });
        generate(&manifest(), input, |prepared| {
            assert_eq!(prepared.image_urls, [image]);
            let MessageContent::Parts(actual) =
                prepared.messages.last().unwrap().content.as_ref().unwrap()
            else {
                panic!("image parts flattened");
            };
            assert_eq!(actual.len(), 2);
            assert_eq!(
                serde_json::to_value(&actual[1]).unwrap(),
                serde_json::to_value(&parts[0]).unwrap()
            );
            assert!(actual[0].text.as_ref().unwrap().contains("call_camera"));
            Ok(response("A photo."))
        })
        .unwrap();
    }

    #[test]
    fn ordinary_text_requests_are_exact_pass_through() {
        let mut input = request();
        input.tool_config = None;
        let output = generate(&manifest(), input.clone(), |prepared| {
            assert_eq!(prepared.prompt, input.prompt);
            assert_eq!(
                serde_json::to_value(&prepared.messages).unwrap(),
                serde_json::to_value(&input.messages).unwrap()
            );
            assert_eq!(prepared.stop, input.stop);
            Ok(response(&call("weather")))
        })
        .unwrap();
        assert_eq!(output.text, call("weather"));
        assert!(output.assistant_message.is_none());
        assert_eq!(output.backend_diagnostics.len(), 1);
    }

    #[test]
    fn plain_json_is_never_promoted_to_a_call() {
        let prose = r#"{"name":"weather","arguments":{"city":"Köln"}}"#;
        let output = generate(&manifest(), request(), |_| Ok(response(prose))).unwrap();
        assert_eq!(output.text, prose);
        assert!(output.assistant_message.unwrap().tool_calls.is_none());
    }

    #[test]
    fn choices_and_parallel_constraints_are_enforced() {
        for (choice, output, should_pass) in [
            (
                ToolChoice::Mode(ToolChoiceMode::Auto),
                "No tools needed".into(),
                true,
            ),
            (
                ToolChoice::Mode(ToolChoiceMode::None),
                "No tools needed".into(),
                true,
            ),
            (
                ToolChoice::Mode(ToolChoiceMode::None),
                call("weather"),
                false,
            ),
            (
                ToolChoice::Mode(ToolChoiceMode::Required),
                "No tools needed".into(),
                false,
            ),
            (
                ToolChoice::Mode(ToolChoiceMode::Required),
                call("weather"),
                true,
            ),
            (named("weather"), call("clock"), false),
            (named("weather"), call("weather"), true),
        ] {
            let mut input = request();
            input.tool_config.as_mut().unwrap().tool_choice = Some(choice.clone());
            assert_eq!(
                generate(&manifest(), input, |_| Ok(response(&output))).is_ok(),
                should_pass,
                "{choice:?}, {output}"
            );
        }
        let two_calls = format!("{}\n{}", call("weather"), call("clock"));
        let output = generate(&manifest(), request(), |_| Ok(response(&two_calls))).unwrap();
        let calls = output.assistant_message.unwrap().tool_calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
        let mut input = request();
        input.tool_config.as_mut().unwrap().parallel_tool_calls = Some(false);
        assert!(generate(&manifest(), input, |_| Ok(response(&two_calls))).is_err());
    }

    #[test]
    fn malformed_and_unknown_calls_fail_explicitly() {
        for malformed in [
            "<tool_call>{bad json}</tool_call>",
            "<tool_call>{\"name\":\"weather\",\"arguments\":{}}",
            "<tool_call>{\"name\":\"missing\",\"arguments\":{}}</tool_call>",
            "<tool_call>{\"name\":\"weather\",\"arguments\":[]}</tool_call>",
            "<tool_call>{\"name\":\"weather\",\"arguments\":\"broken\"}</tool_call>",
            "<tool_call>{\"name\":\"weather\",\"arguments\":{},\"extra\":1}</tool_call>",
            "</tool_call>",
            "<tool_call",
            "<tool_ca",
        ] {
            assert!(
                generate(&manifest(), request(), |_| Ok(response(malformed))).is_err(),
                "accepted {malformed}"
            );
        }
    }

    #[test]
    fn truncated_or_filtered_generations_never_publish_completed_tool_prefixes() {
        for reason in ["length", "content_filter", "error", ""] {
            let mut output = response(&call("weather"));
            output.finish_reason = reason.into();
            let error = generate(&manifest(), request(), |_| Ok(output)).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("before successful tool completion")
            );

            let mut output = response("Partial ordinary text");
            output.finish_reason = reason.into();
            let output = generate(&manifest(), request(), |_| Ok(output)).unwrap();
            assert_eq!(output.finish_reason, reason);
            assert!(output.assistant_message.unwrap().tool_calls.is_none());
        }
    }

    #[tokio::test]
    async fn truncated_streams_emit_no_buffered_text_or_completed_tool_prefixes() {
        for reason in ["length", "content_filter", "error", ""] {
            let mut terminal = done();
            if let GenerateStreamEvent::Done { finish_reason, .. } = &mut terminal {
                *finish_reason = reason.into();
            }
            let events = vec![
                Ok(GenerateStreamEvent::TextChunk(format!(
                    "Checking. {}",
                    call("weather")
                ))),
                Ok(terminal),
            ];
            let mut stream = generate_stream(&manifest(), request(), |_| {
                Box::pin(tokio_stream::iter(events))
            });
            assert!(
                stream
                    .next()
                    .await
                    .unwrap()
                    .unwrap_err()
                    .contains("before successful tool completion")
            );
            assert!(stream.next().await.is_none());
        }
    }

    #[test]
    fn invalid_configuration_fails_before_generation() {
        let mut input = request();
        input.tool_config.as_mut().unwrap().tool_choice = Some(named("unknown"));
        assert!(generate(&manifest(), input, |_| panic!("must not call backend")).is_err());
        let mut input = request();
        let config = input.tool_config.as_mut().unwrap();
        config.tools = Some(vec![]);
        config.tool_choice = Some(ToolChoice::Mode(ToolChoiceMode::Required));
        assert!(generate(&manifest(), input, |_| panic!("must not call backend")).is_err());
        let mut input = request();
        input.tool_config.as_mut().unwrap().tools.as_mut().unwrap()[0]
            .function
            .strict = Some(true);
        assert!(generate(&manifest(), input, |_| panic!("must not call backend")).is_err());
    }

    #[test]
    fn arguments_can_contain_markers_or_encoded_json() {
        for arguments in [
            json!({"city":"literal </tool_call> text"}),
            json!("{\"city\":\"literal </tool_call> text\"}"),
        ] {
            let output = format!(
                "Checking.\n<tool_call>{}</tool_call>",
                json!({"name":"weather","arguments":arguments})
            );
            let output = generate(&manifest(), request(), |_| Ok(response(&output))).unwrap();
            assert_eq!(output.text, "Checking.\n");
            let calls = output.assistant_message.unwrap().tool_calls.unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
                json!({"city":"literal </tool_call> text"})
            );
        }
    }

    #[tokio::test]
    async fn split_stream_never_leaks_markup_and_preserves_usage() {
        let text = format!("Checking.\n{}", call("weather"));
        let mut events = text
            .chars()
            .map(|character| Ok(GenerateStreamEvent::TextChunk(character.to_string())))
            .collect::<Vec<_>>();
        events.push(Ok(done()));
        let stream = generate_stream(&manifest(), request(), |_| {
            Box::pin(tokio_stream::iter(events))
        });
        let result = stream.collect::<Result<Vec<_>, _>>().await.unwrap();
        assert_eq!(result.len(), 3);
        assert!(
            matches!(&result[0], GenerateStreamEvent::TextChunk(text) if text == "Checking.\n")
        );
        match &result[1] {
            GenerateStreamEvent::ToolCallDelta(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].index, 0);
                assert!(calls[0].id.as_ref().unwrap().starts_with("call_"));
                assert_eq!(
                    calls[0].function.as_ref().unwrap().name.as_deref(),
                    Some("weather")
                );
            }
            _ => panic!("missing structured call"),
        }
        match &result[2] {
            GenerateStreamEvent::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
                timings,
                backend_diagnostics,
            } => {
                assert_eq!(finish_reason, "tool_calls");
                assert_eq!((*prompt_tokens, *completion_tokens), (81, 29));
                assert_eq!(timings.total_seconds, 1.2);
                assert!(backend_diagnostics.contains(&"backend metadata".to_string()));
            }
            _ => panic!("missing completion metadata"),
        }
    }

    #[tokio::test]
    async fn stream_errors_expose_no_buffered_output_or_fabricated_usage() {
        for events in [
            vec![
                Ok(GenerateStreamEvent::TextChunk(call("weather"))),
                Err("upstream failed".into()),
            ],
            vec![Ok(GenerateStreamEvent::TextChunk(call("weather")))],
            vec![
                Ok(GenerateStreamEvent::TextChunk(
                    "<tool_call>{bad}</tool_call>".into(),
                )),
                Ok(done()),
            ],
        ] {
            let mut stream = generate_stream(&manifest(), request(), |_| {
                Box::pin(tokio_stream::iter(events))
            });
            assert!(stream.next().await.unwrap().is_err());
            assert!(stream.next().await.is_none());
        }
    }

    #[tokio::test]
    async fn ordinary_streams_pass_through_without_buffering() {
        let mut input = request();
        input.tool_config = None;
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let mut stream = generate_stream(&manifest(), input, |_| {
            Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
        });
        tx.send(Ok(GenerateStreamEvent::TextChunk("immediate".into())))
            .await
            .unwrap();
        assert!(
            matches!(stream.next().await, Some(Ok(GenerateStreamEvent::TextChunk(text))) if text == "immediate")
        );
    }

    #[tokio::test]
    async fn dropping_tool_stream_closes_upstream_receiver_immediately() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let stream = generate_stream(&manifest(), request(), |_| {
            Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
        });
        assert!(!tx.is_closed());
        drop(stream);
        assert!(tx.is_closed());
    }
    #[test]
    fn buffered_output_has_an_independent_byte_limit() {
        let output = "a".repeat(MAX_OUTPUT_BYTES + 1);
        let error = generate(&manifest(), request(), |_| Ok(response(&output))).unwrap_err();
        assert!(error.to_string().contains("validation limit"));
    }

    #[tokio::test]
    async fn oversized_stream_fails_without_exposing_buffered_output() {
        let (tx, rx) = tokio::sync::mpsc::channel(3);
        tx.send(Ok(GenerateStreamEvent::TextChunk(
            "a".repeat(MAX_OUTPUT_BYTES),
        )))
        .await
        .unwrap();
        tx.send(Ok(GenerateStreamEvent::TextChunk("b".into())))
            .await
            .unwrap();
        tx.send(Ok(done())).await.unwrap();
        let mut stream = generate_stream(&manifest(), request(), |_| {
            Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
        });
        assert!(
            stream
                .next()
                .await
                .unwrap()
                .unwrap_err()
                .contains("validation limit")
        );
        assert!(tx.is_closed());
        assert!(stream.next().await.is_none());
    }
}
