use super::support::*;
use crate::backend::{ChatGenerationSession, tool_calling};
use std::{collections::VecDeque, sync::Mutex};

// Token generation is scripted, but the actual adapter protocol and HTTP routes
// prepare history, preserve images, parse calls and translate API responses.
struct GenericBackend {
    outputs: Mutex<VecDeque<String>>,
    prepared: Mutex<Vec<GenerateRequest>>,
}
impl GenericBackend {
    fn output(&self, request: GenerateRequest) -> String {
        assert!(!request.requires_tool_calling());
        self.prepared.lock().unwrap().push(request);
        self.outputs
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected generation")
    }
}
impl GenerationBackend for GenericBackend {
    fn supports_tool_calling(&self, _: &ModelManifest, _: bool) -> bool {
        true
    }
    fn start_chat_session(
        &self,
        _: &ModelManifest,
        _: Option<u64>,
    ) -> anyhow::Result<Option<Box<dyn ChatGenerationSession>>> {
        anyhow::bail!("tool and image requests must bypass text chat session caching")
    }
    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> anyhow::Result<GenerateResponse> {
        tool_calling::generate(manifest, request, |prepared| {
            Ok(GenerateResponse {
                text: self.output(prepared),
                assistant_message: None,
                prompt_tokens: 31,
                completion_tokens: 17,
                finish_reason: "stop".into(),
                timings: GenerationTimings::default(),
                backend_diagnostics: vec![],
            })
        })
    }
    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        tool_calling::generate_stream(&manifest, request, |prepared| {
            let mut events: Vec<_> = self
                .output(prepared)
                .chars()
                .map(|c| Ok(GenerateStreamEvent::TextChunk(c.to_string())))
                .collect();
            events.push(Ok(GenerateStreamEvent::Done {
                finish_reason: "stop".into(),
                prompt_tokens: 31,
                completion_tokens: 17,
                timings: GenerationTimings::default(),
                backend_diagnostics: vec![],
            }));
            Box::pin(tokio_stream::iter(events))
        })
    }
}
fn app(outputs: &[&str]) -> (Router, Arc<GenericBackend>) {
    let store = test_store();
    let manifest = ModelManifest {
        storage: Default::default(),
        id: "generic-qwen".into(),
        source: ModelSource::LocalPath {
            path: "test".into(),
        },
        format: ModelFormat::Unknown,
        architecture: Some("qwen3".into()),
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "mock-generic".into(),
        created_unix: 1,
        files: vec![],
        artifacts: vec![],
        metadata: Default::default(),
    };
    fs::create_dir_all(store.model_dir(&manifest.id)).unwrap();
    fs::write(
        store
            .model_dir(&manifest.id)
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let backend = Arc::new(GenericBackend {
        outputs: Mutex::new(outputs.iter().map(|s| (*s).to_owned()).collect()),
        prepared: Mutex::new(vec![]),
    });
    let state = ApiState::new_with_default_model_and_prompt_options(
        store,
        backend.clone(),
        None,
        Some(Arc::new(|_, _, _| {
            Ok(ChatTemplateOptions {
                default_source: ChatTemplateSource::Model,
                model_template_preferred: true,
                override_name: Some("model"),
            })
        })),
    );
    (router(state), backend)
}
fn openai_tool() -> Value {
    json!({"type":"function","function":{"name":"inspect","description":"Inspect the image",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}})
}
fn anthropic_tool() -> Value {
    let tool = openai_tool();
    json!({"name":tool["function"]["name"],"description":tool["function"]["description"],
        "input_schema":tool["function"]["parameters"]})
}
fn call() -> &'static str {
    "<tool_call>{\"name\":\"inspect\",\"arguments\":{\"city\":\"Köln\"}}</tool_call>"
}
async fn post(app: &Router, path: &str, payload: Value) -> Response {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}
async fn json_body(response: Response) -> Value {
    let status = response.status();
    let bytes = body::to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    serde_json::from_slice(&bytes).unwrap()
}
async fn sse(response: Response) -> (String, Vec<Value>) {
    let status = response.status();
    let bytes = body::to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert_eq!(status, StatusCode::OK, "{text}");
    let events = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    (text, events)
}
fn assert_image(request: &GenerateRequest, expected: &str) {
    assert!(request.image_urls.contains(&expected.to_owned()));
    assert!(request.messages.iter().any(|m| match &m.content {
        Some(crate::openai::MessageContent::Parts(parts)) => parts.iter().any(|part| {
            part.image_url
                .as_ref()
                .is_some_and(|url| serde_json::to_string(url).unwrap().contains(expected))
        }),
        _ => false,
    }));
}
#[tokio::test]
async fn openai_generic_tool_roundtrip_preserves_vision_and_tool_result_identity() {
    let (app, backend) = app(&[call(), "The image shows Köln."]);
    let image = "data:image/png;base64,AQID";
    let user = json!({"role":"user","content":[{"type":"text","text":"Inspect this"},
        {"type":"image_url","image_url":{"url":image,"detail":"high"}}]});
    let first = json_body(post(&app, "/v1/chat/completions", json!({
        "model":"generic-qwen","max_tokens":256,"messages":[user],"tools":[openai_tool()],
        "tool_choice":{"type":"function","function":{"name":"inspect"}},"parallel_tool_calls":false
    })).await).await;
    assert_eq!(first["choices"][0]["finish_reason"], "tool_calls");
    let assistant = first["choices"][0]["message"].clone();
    assert!(assistant["content"].is_null());
    let function_call = &assistant["tool_calls"][0];
    assert_eq!(function_call["function"]["name"], "inspect");
    assert_eq!(
        serde_json::from_str::<Value>(function_call["function"]["arguments"].as_str().unwrap())
            .unwrap(),
        json!({"city":"Köln"})
    );
    let id = function_call["id"].as_str().unwrap();
    assert!(id.starts_with("call_"));
    let followup_image = "https://example.test/inspection.png";
    let final_response = json_body(post(&app, "/v1/chat/completions", json!({
        "model":"generic-qwen","max_tokens":256,"messages":[user,assistant,
            {"role":"tool","tool_call_id":id,"content":[{"type":"text","text":"Köln confirmed"},
                {"type":"image_url","image_url":followup_image}]}]
    })).await).await;
    assert_eq!(
        final_response["choices"][0]["message"]["content"],
        "The image shows Köln."
    );
    assert_eq!(final_response["choices"][0]["finish_reason"], "stop");
    assert_eq!(final_response["usage"]["completion_tokens"], 17);
    let prepared = backend.prepared.lock().unwrap();
    assert_eq!(prepared.len(), 2);
    assert_image(&prepared[0], image);
    assert_image(&prepared[1], followup_image);
    assert!(prepared[0].prompt.contains("\"required\":[\"city\"]"));
    assert!(prepared[1].prompt.contains(id));
    assert!(prepared[1].prompt.contains("Köln confirmed"));
    assert_eq!(prepared[1].messages.last().unwrap().role, "user");
}
#[tokio::test]
async fn anthropic_generic_tool_roundtrip_preserves_vision_and_tool_results() {
    let (app, backend) = app(&[call(), "The image shows Köln."]);
    let user = json!({"role":"user","content":[{"type":"text","text":"Inspect this"},
        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}}]});
    let first = json_body(post(&app, "/v1/messages", json!({
        "model":"generic-qwen","max_tokens":256,"messages":[user],"tools":[anthropic_tool()],
        "tool_choice":{"type":"tool","name":"inspect","disable_parallel_tool_use":true}
    })).await).await;
    assert_eq!(first["stop_reason"], "tool_use");
    let tool_use = &first["content"][0];
    assert_eq!(tool_use["type"], "tool_use");
    assert_eq!(tool_use["name"], "inspect");
    assert_eq!(tool_use["input"], json!({"city":"Köln"}));
    let id = tool_use["id"].as_str().unwrap();
    let final_response = json_body(post(&app, "/v1/messages", json!({
        "model":"generic-qwen","max_tokens":256,"messages":[user,
            {"role":"assistant","content":first["content"]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":id,"content":[
                {"type":"text","text":"Köln confirmed"},
                {"type":"image","source":{"type":"url","url":"https://example.test/inspection.png"}}
            ]}]}]
    })).await).await;
    assert_eq!(final_response["stop_reason"], "end_turn");
    assert_eq!(
        final_response["content"][0]["text"],
        "The image shows Köln."
    );
    assert_eq!(final_response["usage"]["output_tokens"], 17);
    let prepared = backend.prepared.lock().unwrap();
    assert_eq!(prepared.len(), 2);
    assert_image(&prepared[0], "data:image/png;base64,AQID");
    assert_image(&prepared[1], "https://example.test/inspection.png");
    assert!(prepared[1].prompt.contains(id));
    assert!(prepared[1].prompt.contains("Köln confirmed"));
}
#[tokio::test]
async fn openai_generic_stream_emits_validated_parallel_calls_and_usage_without_markup() {
    let output = format!("Checking.\n{}\n{}", call(), call());
    let (app, _) = app(&[&output]);
    let (text, events) = sse(post(&app, "/v1/chat/completions", json!({
        "model":"generic-qwen","max_tokens":256,"stream":true,"stream_options":{"include_usage":true},
        "messages":[{"role":"user","content":"Inspect both"}],"tools":[openai_tool()],
        "tool_choice":"required","parallel_tool_calls":true
    })).await).await;
    assert!(text.contains("[DONE]"));
    assert!(!text.contains("<tool_call>"));
    let chunks: Vec<_> = events
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["tool_calls"].as_array())
        .collect();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].len(), 2);
    for (index, c) in chunks[0].iter().enumerate() {
        assert_eq!(c["index"], index);
        assert!(c["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(c["function"]["name"], "inspect");
        assert_eq!(
            serde_json::from_str::<Value>(c["function"]["arguments"].as_str().unwrap()).unwrap(),
            json!({"city":"Köln"})
        );
    }
    assert_ne!(chunks[0][0]["id"], chunks[0][1]["id"]);
    assert!(
        events
            .iter()
            .any(|e| e["choices"][0]["finish_reason"] == "tool_calls")
    );
    assert!(events.iter().any(|e| e["usage"]["completion_tokens"] == 17));
}
#[tokio::test]
async fn anthropic_generic_stream_emits_tool_use_blocks_and_valid_json_deltas() {
    let output = format!("Checking.\n{}\n{}", call(), call());
    let (app, _) = app(&[&output]);
    let (text, events) = sse(post(
        &app,
        "/v1/messages",
        json!({
            "model":"generic-qwen","max_tokens":256,"stream":true,
            "messages":[{"role":"user","content":"Inspect both"}],"tools":[anthropic_tool()],
            "tool_choice":{"type":"any"}
        }),
    )
    .await)
    .await;
    assert!(!text.contains("<tool_call>"));
    let starts: Vec<_> = events
        .iter()
        .filter(|e| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use")
        .collect();
    assert_eq!(starts.len(), 2);
    assert_ne!(
        starts[0]["content_block"]["id"],
        starts[1]["content_block"]["id"]
    );
    for start in starts {
        assert_eq!(start["content_block"]["name"], "inspect");
        let args: String = events
            .iter()
            .filter(|e| e["index"] == start["index"] && e["delta"]["type"] == "input_json_delta")
            .map(|e| e["delta"]["partial_json"].as_str().unwrap())
            .collect();
        assert_eq!(
            serde_json::from_str::<Value>(&args).unwrap(),
            json!({"city":"Köln"})
        );
    }
    assert!(events.iter().any(|e| e["type"] == "message_delta"
        && e["delta"]["stop_reason"] == "tool_use"
        && e["usage"]["output_tokens"] == 17));
    assert_eq!(events.last().unwrap()["type"], "message_stop");
}
