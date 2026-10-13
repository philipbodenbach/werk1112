//! Official hosted chat provider, sharing the local backends' OpenAI parsing.
use super::{
    GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent, GenerationBackend,
    GenerationTimings, PreloadHandle, openai_transport as wire,
};
use crate::{
    model_store::ModelManifest,
    model_support::{self, Provider},
    providers::http,
};
use anyhow::{Result, ensure};
use serde_json::Value;
use std::{collections::BTreeMap, io::Read, time::Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub struct ProviderBackend;
const OPTIONS: &[&str] = &[
    "response_format",
    "frequency_penalty",
    "presence_penalty",
    "n",
    "logprobs",
    "top_logprobs",
    "top_k",
    "reasoning_effort",
];
fn model(manifest: &ModelManifest) -> Result<&str> {
    let Some((Provider::Mistral, model @ "mistral-large-4")) = model_support::remote(manifest)
    else {
        anyhow::bail!("unsupported_task: this provider model does not support chat");
    };
    Ok(model)
}
fn body(
    manifest: &ModelManifest,
    request: &GenerateRequest,
    options: BTreeMap<String, Value>,
    stream: bool,
) -> Result<Value> {
    let mut value = wire::chat_completion_body(model(manifest)?, request, stream);
    value.as_object_mut().unwrap().remove("stream_options");
    if let Some(seed) = value.as_object_mut().unwrap().remove("seed") {
        value["random_seed"] = seed;
    }
    wire::apply_api_options(&mut value, options, OPTIONS)?;
    Ok(value)
}
// Mistral's official ContentChunk union includes text and thinking blocks.
// Keep visible text separate from reasoning in the OpenAI-compatible response.
fn normalize_content(value: &mut Value) -> Result<()> {
    if let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) {
        for choice in choices {
            for field in ["message", "delta"] {
                let Some(message) = choice.get_mut(field) else {
                    continue;
                };
                let Some(parts) = message.get("content").and_then(Value::as_array) else {
                    continue;
                };
                let mut text = String::new();
                let mut reasoning = Vec::new();
                for part in parts {
                    match part["type"].as_str() {
                        Some("text") => {
                            text.push_str(part["text"].as_str().ok_or_else(|| {
                                http::error("upstream_schema", "invalid text chunk")
                            })?)
                        }
                        Some("thinking") => reasoning.push(part.clone()),
                        _ => {
                            return Err(http::error(
                                "unsupported_task",
                                "provider returned a non-text content chunk unsupported by this chat adapter",
                            ));
                        }
                    }
                }
                message["content"] = Value::String(text);
                if !reasoning.is_empty() {
                    message["reasoning_content"] = Value::Array(reasoning);
                }
            }
        }
    }
    Ok(())
}
fn execute(
    client: &http::ProviderClient,
    body: &Value,
    mut event: impl FnMut(Value) -> Result<()>,
) -> Result<Value> {
    let mut response = client.post("/v1/chat/completions", body)?;
    if body["stream"] != true {
        let mut value = http::read_json(response)?;
        normalize_content(&mut value)?;
        ensure!(
            value["choices"].as_array().is_some_and(|s| !s.is_empty()),
            "upstream_schema: missing completion choices"
        );
        return Ok(value);
    }
    let mut accumulator = wire::SseAccumulator::default();
    let mut chunk = [0; 8192];
    let mut done = false;
    loop {
        let count = response
            .read(&mut chunk)
            .map_err(|_| http::error("upstream_transport", "chat stream read failed"))?;
        if count == 0 {
            break;
        }
        accumulator
            .push(&chunk[..count], |data| {
                if data.trim() == "[DONE]" {
                    done = true;
                    return Ok(());
                }
                ensure!(!done, "upstream_schema: data after stream terminator");
                let mut value: Value = serde_json::from_str(data)
                    .map_err(|_| http::error("upstream_schema", "invalid chat stream JSON"))?;
                ensure!(
                    value.get("error").is_none(),
                    "upstream_error: provider stream failed"
                );
                normalize_content(&mut value)?;
                event(value)
            })
            .map_err(|_| {
                http::error(
                    "upstream_stream",
                    "chat stream failed or consumer disconnected",
                )
            })?;
        if done {
            break;
        }
    }
    ensure!(done, "upstream_stream: chat stream ended without DONE");
    Ok(Value::Null)
}
impl ProviderBackend {
    fn completion(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<GenerateResponse> {
        let start = Instant::now();
        let body = body(manifest, &request, BTreeMap::new(), tx.is_some())?;
        let client = http::client(Provider::Mistral)?;
        let mut completion = wire::OpenAiCompletion::default();
        let response = execute(&client, &body, |event| {
            wire::update_completion_from_event(&mut completion, &event);
            if let Some(tx) = &tx {
                if let Some(text) = wire::delta_content(&event) {
                    completion.text.push_str(&text);
                    if completion.first_token_seconds == 0.0 && !text.is_empty() {
                        completion.first_token_seconds = start.elapsed().as_secs_f64();
                    }
                    tx.blocking_send(Ok(GenerateStreamEvent::TextChunk(text)))
                        .map_err(|_| anyhow::anyhow!("consumer disconnected"))?;
                }
                if let Some(deltas) = wire::delta_tool_calls(&event)? {
                    completion.saw_tool_call_delta = true;
                    tx.blocking_send(Ok(GenerateStreamEvent::ToolCallDelta(deltas)))
                        .map_err(|_| anyhow::anyhow!("consumer disconnected"))?;
                }
            }
            Ok(())
        })?;
        if tx.is_none() {
            wire::update_completion_from_event(&mut completion, &response);
            wire::update_completion_from_message(&mut completion, &response)?;
        }
        wire::ensure_visible_completion(&completion)?;
        let assistant_message = completion.assistant_message();
        Ok(GenerateResponse {
            text: completion.text,
            assistant_message: Some(assistant_message),
            prompt_tokens: completion.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            finish_reason: completion.finish_reason,
            timings: GenerationTimings {
                first_token_seconds: completion.first_token_seconds,
                total_seconds: start.elapsed().as_secs_f64(),
                ..Default::default()
            },
            backend_diagnostics: vec![
                "Mistral hosted API; inference verification: IMPLEMENTED_UNTESTED".into(),
            ],
        })
    }
}
impl GenerationBackend for ProviderBackend {
    fn supports_tool_calling(&self, manifest: &ModelManifest, _images: bool) -> bool {
        model(manifest).is_ok()
    }
    fn preload(
        &self,
        manifest: &ModelManifest,
        _request: GenerateRequest,
    ) -> Result<PreloadHandle> {
        model(manifest)?;
        http::preload(Provider::Mistral)
    }
    fn validate_api_options(
        &self,
        manifest: &ModelManifest,
        _request: &GenerateRequest,
        options: &BTreeMap<String, Value>,
    ) -> Result<()> {
        model(manifest)?;
        wire::validate_api_options(options, OPTIONS)?;
        if let Some(effort) = options.get("reasoning_effort") {
            ensure!(
                effort
                    .as_str()
                    .is_some_and(
                        |s| ["none", "minimal", "low", "medium", "high", "xhigh"].contains(&s)
                    ),
                "invalid_request: unsupported Mistral reasoning_effort"
            );
        }
        Ok(())
    }
    fn generate_api(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        options: BTreeMap<String, Value>,
        tx: Option<mpsc::Sender<Result<Value, String>>>,
    ) -> Result<Value> {
        self.validate_api_options(manifest, &request, &options)?;
        let body = body(manifest, &request, options, tx.is_some())?;
        let client = http::client(Provider::Mistral)?;
        execute(&client, &body, |value| {
            if let Some(tx) = &tx {
                tx.blocking_send(Ok(value))
                    .map_err(|_| anyhow::anyhow!("consumer disconnected"))?;
            }
            Ok(())
        })
    }
    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        self.completion(manifest, request, None)
    }
    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        let (tx, rx) = mpsc::channel(32);
        crate::logging::spawn_blocking(move || {
            let result = Self.completion(&manifest, request, Some(tx.clone()));
            wire::send_stream_result(tx, result);
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{StreamGranularity, ToolCallingConfig},
        model_store::ModelStore,
        openai::ChatCompletionRequest,
        providers::tests::mock,
    };
    use serde_json::json;
    fn request(value: Value) -> GenerateRequest {
        let chat: ChatCompletionRequest = serde_json::from_value(value).unwrap();
        GenerateRequest {
            prompt: String::new(),
            messages: chat.messages,
            image_urls: vec![],
            max_tokens: 32,
            temperature: Some(0.5),
            top_p: Some(0.9),
            stop: vec![],
            seed: Some(42),
            stream_granularity: StreamGranularity::Token,
            verbose: false,
            debug: false,
            tool_config: Some(ToolCallingConfig {
                tools: chat.tools,
                tool_choice: chat.tool_choice,
                parallel_tool_calls: chat.parallel_tool_calls,
            }),
        }
    }
    #[test]
    fn mistral_thinking_blocks_do_not_become_visible_text() {
        let mut response = json!({"choices":[{"message":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"reasoning"}],"signature":"opaque"},{"type":"text","text":"answer"}]}}]});
        normalize_content(&mut response).unwrap();
        assert_eq!(response["choices"][0]["message"]["content"], "answer");
        assert_eq!(
            response["choices"][0]["message"]["reasoning_content"][0]["signature"],
            "opaque"
        );
    }
    #[test]
    fn mistral_mapping_and_tool_round_trip_keep_messages_intact() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::resolve(Some(dir.path().into())).unwrap();
        let manifest = model_support::register_remote(&store, "mistral-large-4", None).unwrap();
        let first = request(
            json!({"model":"mistral-large-4","messages":[{"role":"user","content":"weather"}],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{}}}}],"tool_choice":"required"}),
        );
        let value=body(&manifest,&first,BTreeMap::from([("response_format".into(),json!({"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object"}}}))]),true).unwrap();
        assert_eq!(value["random_seed"], 42);
        assert!(value.get("seed").is_none());
        assert!(value.get("stream_options").is_none());
        assert_eq!(value["tools"][0]["function"]["name"], "weather");
        assert_eq!(value["response_format"]["type"], "json_schema");
        let call = json!({"id":"Abc123xyz","type":"function","function":{"name":"weather","arguments":"{}"}});
        let first_response = json!({"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[call.clone()]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}});
        let second_response = json!({"choices":[{"message":{"role":"assistant","content":"sunny"},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":2}});
        let (url, captured, server) = mock(vec![
            (
                200,
                "application/json",
                serde_json::to_vec(&first_response).unwrap(),
            ),
            (
                200,
                "application/json",
                serde_json::to_vec(&second_response).unwrap(),
            ),
        ]);
        let client = http::ProviderClient::for_test(Provider::Mistral, &url).unwrap();
        let reply = execute(
            &client,
            &body(&manifest, &first, BTreeMap::new(), false).unwrap(),
            |_| Ok(()),
        )
        .unwrap();
        let mut completion = wire::OpenAiCompletion::default();
        wire::update_completion_from_message(&mut completion, &reply).unwrap();
        assert_eq!(completion.tool_calls.unwrap()[0].id, "Abc123xyz");
        let next = request(
            json!({"model":"mistral-large-4","messages":[{"role":"user","content":"weather"},{"role":"assistant","content":null,"tool_calls":[call]},{"role":"tool","tool_call_id":"Abc123xyz","content":"sunny"}]}),
        );
        execute(
            &client,
            &body(&manifest, &next, BTreeMap::new(), false).unwrap(),
            |_| Ok(()),
        )
        .unwrap();
        server.join().unwrap();
        assert_eq!(
            captured.lock().unwrap()[1].1["messages"][2]["tool_call_id"],
            "Abc123xyz"
        );
    }
    #[test]
    fn mistral_stream_delivers_tool_deltas_and_requires_done() {
        let event = json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"Abc123xyz","type":"function","function":{"name":"weather","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":2}});
        let (url, _, server) = mock(vec![(
            200,
            "text/event-stream",
            format!("data: {event}\n\ndata: [DONE]\n\n").into_bytes(),
        )]);
        let client = http::ProviderClient::for_test(Provider::Mistral, &url).unwrap();
        let mut events = vec![];
        execute(&client, &json!({"stream":true}), |v| {
            events.push(v);
            Ok(())
        })
        .unwrap();
        server.join().unwrap();
        assert_eq!(events.len(), 1);
        assert!(wire::delta_tool_calls(&events[0]).unwrap().is_some());
        let (url, _, server) = mock(vec![(
            200,
            "text/event-stream",
            format!("data: {event}\n\n").into_bytes(),
        )]);
        let client = http::ProviderClient::for_test(Provider::Mistral, &url).unwrap();
        assert!(execute(&client, &json!({"stream":true}), |_| Ok(())).is_err());
        server.join().unwrap();
    }
}
