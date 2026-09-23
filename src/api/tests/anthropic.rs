use super::support::*;
use crate::{
    backend::{ChatGenerationSession, GeneratedAssistantMessage},
    openai::{ChatCompletionFunctionCallDelta, ChatCompletionToolCallDelta},
};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct Fixture {
    requests: Mutex<Vec<GenerateRequest>>,
    events: Option<Vec<Result<GenerateStreamEvent, String>>>,
    response: Option<GenerateResponse>,
    sessions: Arc<AtomicUsize>,
    support_tools: bool,
}
impl Fixture {
    fn tools() -> Self {
        Self {
            support_tools: true,
            ..Self::default()
        }
    }
}

fn tool_response() -> GenerateResponse {
    let mut result = MockBackend
        .generate(&dummy_manifest(), empty_request())
        .unwrap();
    result.text.clear();
    result.finish_reason = "tool_calls".into();
    result.prompt_tokens = 17;
    result.completion_tokens = 8;
    result.assistant_message = Some(GeneratedAssistantMessage { content: Some("Ich rechne.".into()), tool_calls: Some(serde_json::from_value(json!([
        {"id":"call_one","type":"function","function":{"name":"add","arguments":"{\"a\":2,\"b\":3}"}},
        {"id":"call_two","type":"function","function":{"name":"add","arguments":"{\"a\":5,\"b\":6}"}}
    ])).unwrap()) });
    result
}
fn empty_request() -> GenerateRequest {
    GenerateRequest {
        prompt: String::new(),
        messages: vec![],
        image_urls: vec![],
        max_tokens: 10,
        temperature: None,
        top_p: None,
        stop: vec![],
        seed: None,
        stream_granularity: crate::backend::StreamGranularity::Chunk,
        verbose: false,
        debug: false,
        tool_config: None,
    }
}
fn wants_call(request: &GenerateRequest) -> bool {
    request
        .tool_config
        .as_ref()
        .is_some_and(|c| c.tools.as_ref().is_some_and(|t| !t.is_empty()))
        && (!request.messages.iter().any(|m| m.role == "tool")
            || request.messages.last().is_some_and(|m| m.role == "user"
                && matches!(&m.content, Some(crate::openai::MessageContent::Text(text)) if text.starts_with("Use the add tool"))))
}
fn turn_suffix(request: &GenerateRequest) -> String {
    let results = request.messages.iter().filter(|m| m.role == "tool").count();
    if results == 0 {
        String::new()
    } else {
        format!("_{results}")
    }
}
impl GenerationBackend for Fixture {
    fn count_tokens(&self, _: &ModelManifest, request: GenerateRequest) -> anyhow::Result<usize> {
        self.requests.lock().unwrap().push(request);
        Ok(23)
    }
    fn supports_tool_calling(&self, _: &ModelManifest, _: bool) -> bool {
        self.support_tools
    }
    fn start_chat_session(
        &self,
        _: &ModelManifest,
        _: Option<u64>,
    ) -> anyhow::Result<Option<Box<dyn ChatGenerationSession>>> {
        self.sessions.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    }
    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> anyhow::Result<GenerateResponse> {
        self.requests.lock().unwrap().push(request.clone());
        if let Some(response) = &self.response {
            return Ok(response.clone());
        }
        if wants_call(&request) {
            let mut response = tool_response();
            for call in response
                .assistant_message
                .as_mut()
                .unwrap()
                .tool_calls
                .as_mut()
                .unwrap()
            {
                call.id.push_str(&turn_suffix(&request));
            }
            return Ok(response);
        }
        MockBackend.generate(manifest, request)
    }
    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        self.requests.lock().unwrap().push(request.clone());
        if let Some(events) = &self.events {
            return Box::pin(tokio_stream::iter(events.clone()));
        }
        if wants_call(&request) {
            let suffix = turn_suffix(&request);
            return Box::pin(tokio_stream::iter(vec![
                Ok(GenerateStreamEvent::TextChunk("Ich rechne.".into())),
                delta(0, Some("call_"), Some("a"), "{\"a\":"),
                delta(
                    1,
                    Some(&format!("call_two{suffix}")),
                    Some("add"),
                    "{\"a\":5,",
                ),
                delta(0, Some(&format!("one{suffix}")), Some("dd"), "2,\"b\":3}"),
                delta(1, None, None, "\"b\":6}"),
                done("tool_calls"),
            ]));
        }
        MockBackend.generate_stream(manifest, request)
    }
}
fn delta(
    index: usize,
    id: Option<&str>,
    name: Option<&str>,
    args: &str,
) -> Result<GenerateStreamEvent, String> {
    Ok(GenerateStreamEvent::ToolCallDelta(vec![
        ChatCompletionToolCallDelta {
            index,
            id: id.map(String::from),
            kind: None,
            function: Some(ChatCompletionFunctionCallDelta {
                name: name.map(String::from),
                arguments: Some(args.into()),
            }),
        },
    ]))
}
fn done(reason: &str) -> Result<GenerateStreamEvent, String> {
    Ok(GenerateStreamEvent::Done {
        finish_reason: reason.into(),
        prompt_tokens: 17,
        completion_tokens: 8,
        timings: GenerationTimings::default(),
        backend_diagnostics: vec![],
    })
}
fn dummy_manifest() -> ModelManifest {
    ModelManifest {
        storage: Default::default(),
        id: "mock".into(),
        source: ModelSource::LocalPath {
            path: "test".into(),
        },
        format: ModelFormat::Unknown,
        architecture: None,
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "mock".into(),
        created_unix: 1,
        files: vec![],
        artifacts: vec![],
        metadata: Default::default(),
    }
}
fn state(backend: Arc<dyn GenerationBackend>) -> ApiState {
    let store = test_store();
    let manifest = dummy_manifest();
    fs::create_dir_all(store.model_dir("mock")).unwrap();
    fs::write(
        store
            .model_dir("mock")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    ApiState::new(store, backend)
}
fn request() -> Value {
    json!({"model":"mock","max_tokens":96,"messages":[{"role":"user","content":"Hi"}]})
}
fn tools() -> Value {
    json!([{"name":"add","description":"Add integers","input_schema":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}])
}
async fn post(app: &Router, value: Value) -> Response {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(value.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}
async fn sse(response: Response) -> Vec<Value> {
    assert_eq!(response.status(), StatusCode::OK);
    let body = body::to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    text.split("\n\n")
        .filter(|event| !event.is_empty())
        .map(|event| {
            let kind = event
                .lines()
                .find_map(|line| line.strip_prefix("event: "))
                .unwrap();
            let value: Value = serde_json::from_str(
                event
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(value["type"], kind);
            value
        })
        .collect()
}

#[tokio::test]
async fn text_and_stream_have_anthropic_shapes_and_final_usage() {
    let app = router(state(Arc::new(Fixture::tools())));
    let response = post(&app, request()).await;
    assert!(response.headers().get("request-id").is_some());
    let value = response_json(response).await;
    assert_eq!(value["type"], "message");
    assert_eq!(value["content"], json!([{"type":"text","text":"hello"}]));
    assert_eq!(value["stop_reason"], "end_turn");
    let mut req = request();
    req["stream"] = true.into();
    let events = sse(post(&app, req).await).await;
    assert_eq!(
        events
            .iter()
            .map(|v| v["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]
    );
    assert_eq!(events[0]["message"]["usage"]["input_tokens"], 0);
    assert_eq!(
        events[4]["usage"],
        json!({"input_tokens":2,"output_tokens":1})
    );
}

#[tokio::test]
async fn tool_round_trip_preserves_ids_error_results_and_trailing_text() {
    let backend = Arc::new(Fixture::tools());
    let app = router(state(backend.clone()));
    let mut req = request();
    req["tools"] = tools();
    let first = response_json(post(&app, req.clone()).await).await;
    assert_eq!(first["stop_reason"], "tool_use");
    req["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":first["content"]}),
        json!({"role":"user","content":[
            {"type":"tool_result","tool_use_id":"call_one","content":"5"},
            {"type":"tool_result","tool_use_id":"call_two","content":[{"type":"text","text":"failed"}],"is_error":true},
            {"type":"text","text":"Please explain."}]}),
    ]);
    let final_msg = response_json(post(&app, req).await).await;
    assert_eq!(final_msg["stop_reason"], "end_turn");
    let requests = backend.requests.lock().unwrap();
    let history = &requests[1].messages;
    assert_eq!(
        history.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
        ["user", "assistant", "tool", "tool", "user"]
    );
    assert_eq!(history[2].tool_call_id.as_deref(), Some("call_one"));
    assert_eq!(
        serde_json::to_value(&history[3].content).unwrap(),
        json!("{\"content\":\"failed\",\"is_error\":true}")
    );
    assert_eq!(
        serde_json::to_value(&history[4].content).unwrap(),
        "Please explain."
    );
    assert_eq!(
        backend.sessions.load(Ordering::SeqCst),
        0,
        "tool requests must bypass text-only sessions"
    );
}

#[tokio::test]
async fn interleaved_tool_fragments_produce_stable_sequential_blocks() {
    let app = router(state(Arc::new(Fixture::tools())));
    let mut req = request();
    req["tools"] = tools();
    req["stream"] = true.into();
    let events = sse(post(&app, req).await).await;
    let starts: Vec<_> = events
        .iter()
        .filter(|v| v["type"] == "content_block_start")
        .collect();
    assert_eq!(starts.len(), 3);
    assert!(
        starts[1]["content_block"]["id"]
            .as_str()
            .unwrap()
            .starts_with("toolu_")
    );
    assert_eq!(starts[1]["content_block"]["name"], "add");
    assert_eq!(starts[2]["index"], 2);
    let mut fragments = BTreeMap::<usize, String>::new();
    for event in events
        .iter()
        .filter(|v| v["delta"]["type"] == "input_json_delta")
    {
        fragments
            .entry(event["index"].as_u64().unwrap() as usize)
            .or_default()
            .push_str(event["delta"]["partial_json"].as_str().unwrap());
    }
    let arguments: Vec<Value> = fragments
        .values()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(arguments, vec![json!({"a":2,"b":3}), json!({"a":5,"b":6})]);
    assert_eq!(events[events.len() - 2]["delta"]["stop_reason"], "tool_use");
}

#[tokio::test]
async fn validation_rejects_unsupported_fields_and_invalid_histories_before_generation() {
    let backend = Arc::new(Fixture::tools());
    let app = router(state(backend.clone()));
    for source in [
        json!({"type":"url","url":"http://"}),
        json!({"type":"url","url":"file:///tmp/image.png"}),
        json!({"type":"url","url":"https://host/\r\ninjected"}),
        json!({"type":"base64","media_type":"image/png","data":"!invalid!"}),
        json!({"type":"base64","media_type":"image/png","data":""}),
        json!({"type":"base64","media_type":"text/plain","data":"AQID"}),
    ] {
        let mut req = request();
        req["messages"] = json!([{"role":"user","content":[{"type":"image","source":source}]}]);
        assert_eq!(post(&app, req).await.status(), StatusCode::BAD_REQUEST);
    }
    let invalid = vec![
        ("max_tokens", json!(0)),
        ("model", json!("")),
        ("messages", json!([])),
        ("temperature", json!(1.1)),
        ("top_p", json!(-0.1)),
        ("stop_sequences", json!(["END"])),
        ("thinking", json!({"type":"enabled","budget_tokens":100})),
        ("metadata", json!({"user_id":"x".repeat(257)})),
        (
            "system",
            json!([{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]),
        ),
        ("messages", json!([{"role":"system","content":"hi"}])),
        (
            "messages",
            json!([{"role":"user","content":[{"type":"image","source":{}}]}]),
        ),
        (
            "messages",
            json!([{"role":"user","content":[{"type":"tool_result","tool_use_id":"unknown","content":"5"}]}]),
        ),
        (
            "messages",
            json!([{"role":"assistant","content":[{"type":"tool_use","id":"x","name":"add","input":{}},{"type":"text","text":"late"}]}]),
        ),
        (
            "messages",
            json!([{"role":"assistant","content":[{"type":"tool_use","id":"x","name":"add","input":{}}]}]),
        ),
        (
            "tools",
            json!([{"type":"web_search_20250305","name":"web_search"}]),
        ),
        ("tool_choice", json!({"type":"any"})),
    ];
    for (field, value) in invalid {
        let mut req = request();
        req[field] = value;
        let response = post(&app, req.clone()).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{req}");
        assert_eq!(
            response_json(response).await["error"]["type"],
            "invalid_request_error"
        );
    }
    let use_block = json!({"type":"tool_use","id":"x","name":"add","input":{}});
    for results in [
        json!([]),
        json!([{"type":"text","text":"missing"}]),
        json!([
        {"type":"tool_result","tool_use_id":"x","content":"5"}, {"type":"tool_result","tool_use_id":"x","content":"5"}]),
    ] {
        let mut req = request();
        req["messages"] =
            json!([{"role":"assistant","content":[use_block]}, {"role":"user","content":results}]);
        assert_eq!(post(&app, req).await.status(), StatusCode::BAD_REQUEST);
    }
    assert!(backend.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn error_envelopes_cover_auth_version_json_size_missing_model_and_capabilities() {
    let app = router(state(Arc::new(MockBackend)).with_api_keys(vec!["secret".into()]));
    let response = post(&app, request()).await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let id = response.headers()["request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let value = response_json(response).await;
    assert_eq!(value["request_id"], id);
    assert_eq!(value["error"]["type"], "authentication_error");
    for header_name in ["x-api-key", "authorization"] {
        let value = if header_name == "x-api-key" {
            "secret"
        } else {
            "Bearer secret"
        };
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(header_name, value)
                    .header("content-type", "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(request().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    let app = router(state(Arc::new(MockBackend)));
    for (version, beta, body) in [
        (None, false, request().to_string()),
        (Some("wrong"), false, request().to_string()),
        (Some("2023-06-01"), true, request().to_string()),
        (Some("2023-06-01"), false, "{".into()),
    ] {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json");
        if let Some(version) = version {
            builder = builder.header("anthropic-version", version)
        }
        if beta {
            builder = builder.header("anthropic-beta", "future-beta")
        }
        let response = app
            .clone()
            .oneshot(builder.body(Body::from(body)).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response_json(response).await["type"], "error");
    }
    let limited = super::super::router::router_with_body_limit(state(Arc::new(MockBackend)), 16);
    let response = post(&limited, request()).await;
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        response_json(response).await["error"]["type"],
        "request_too_large"
    );
    let mut req = request();
    req["model"] = "absent".into();
    assert_eq!(post(&app, req).await.status(), StatusCode::NOT_FOUND);
    let mut req = request();
    req["tools"] = tools();
    let response = post(&app, req).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        response_json(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("does not support")
    );
}

#[tokio::test]
async fn stream_errors_and_truncation_never_claim_successful_tool_completion() {
    let cases = vec![
        vec![
            Ok(GenerateStreamEvent::TextChunk("Grüße 🌍".into())),
            Err("failure".into()),
            done("stop"),
        ],
        vec![Ok(GenerateStreamEvent::TextChunk("early EOF".into()))],
        vec![delta(0, Some("x"), Some("add"), "{"), done("tool_calls")],
        vec![done("unknown")],
        vec![
            delta(0, Some("x"), Some("add"), "{}"),
            Ok(GenerateStreamEvent::TextChunk("late".into())),
            done("tool_calls"),
        ],
        vec![
            delta(0, Some("x"), Some("add"), &"x".repeat(8 * 1024 * 1024)),
            done("tool_calls"),
        ],
    ];
    for events in cases {
        let app = router(state(Arc::new(Fixture {
            events: Some(events),
            ..Fixture::tools()
        })));
        let mut req = request();
        req["stream"] = true.into();
        let events = sse(post(&app, req).await).await;
        assert_eq!(events.last().unwrap()["type"], "error");
        assert!(!events.iter().any(|v| v["type"] == "message_stop"));
    }
    let app = router(state(Arc::new(Fixture {
        events: Some(vec![
            delta(0, Some("x"), Some("add"), "{\"a\":"),
            done("length"),
        ]),
        ..Fixture::tools()
    })));
    let mut req = request();
    req["stream"] = true.into();
    let events = sse(post(&app, req).await).await;
    assert_eq!(
        events[events.len() - 2]["delta"]["stop_reason"],
        "max_tokens"
    );
    assert_eq!(events.last().unwrap()["type"], "message_stop");
}

#[tokio::test]
async fn malformed_nonstream_tool_arguments_are_a_backend_error() {
    let mut result = tool_response();
    result
        .assistant_message
        .as_mut()
        .unwrap()
        .tool_calls
        .as_mut()
        .unwrap()[0]
        .function
        .arguments = "{".into();
    let app = router(state(Arc::new(Fixture {
        response: Some(result),
        ..Fixture::tools()
    })));
    let response = post(&app, request()).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(response_json(response).await["error"]["type"], "api_error");
}

#[tokio::test]
async fn tool_choice_constraints_survive_translation_for_backend_validation() {
    use crate::openai::{ToolChoice, ToolChoiceMode};
    let backend = Arc::new(Fixture::tools());
    let app = router(state(backend.clone()));
    for choice in [
        json!({"type":"auto"}),
        json!({"type":"auto","disable_parallel_tool_use":false}),
        json!({"type":"none"}),
        json!({"type":"any"}),
        json!({"type":"tool","name":"add","disable_parallel_tool_use":true}),
    ] {
        let mut req = request();
        req["tools"] = tools();
        req["tools"][0]["strict"] = true.into();
        req["tool_choice"] = choice;
        assert_eq!(post(&app, req).await.status(), StatusCode::OK);
    }
    let requests = backend.requests.lock().unwrap();
    let configs: Vec<_> = requests
        .iter()
        .map(|r| r.tool_config.as_ref().unwrap())
        .collect();
    assert_eq!(
        configs[0].tool_choice,
        Some(ToolChoice::Mode(ToolChoiceMode::Auto))
    );
    assert_eq!(configs[0].parallel_tool_calls, None);
    assert_eq!(configs[1].parallel_tool_calls, None);
    assert_eq!(
        configs[2].tool_choice,
        Some(ToolChoice::Mode(ToolChoiceMode::None))
    );
    assert_eq!(
        configs[3].tool_choice,
        Some(ToolChoice::Mode(ToolChoiceMode::Required))
    );
    assert!(matches!(configs[4].tool_choice, Some(ToolChoice::Named(_))));
    assert_eq!(configs[4].parallel_tool_calls, Some(false));
    assert_eq!(
        configs[4].tools.as_ref().unwrap()[0].function.strict,
        Some(true)
    );
}

struct SessionBackend {
    starts: Arc<AtomicUsize>,
}
struct Session;
impl ChatGenerationSession for Session {
    fn generate(&self, request: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        MockBackend.generate(&dummy_manifest(), request)
    }
    fn generate_stream(&self, request: GenerateRequest) -> GenerateStream {
        MockBackend.generate_stream(dummy_manifest(), request)
    }
}
impl GenerationBackend for SessionBackend {
    fn start_chat_session(
        &self,
        _: &ModelManifest,
        _: Option<u64>,
    ) -> anyhow::Result<Option<Box<dyn ChatGenerationSession>>> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        Ok(Some(Box::new(Session)))
    }
    fn generate(&self, _: &ModelManifest, _: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        panic!("must reuse session")
    }
    fn generate_stream(&self, _: ModelManifest, _: GenerateRequest) -> GenerateStream {
        panic!("must reuse session")
    }
}
#[tokio::test]
async fn both_protocols_share_one_text_session_for_streaming_and_json() {
    let starts = Arc::new(AtomicUsize::new(0));
    let app = router(state(Arc::new(SessionBackend {
        starts: starts.clone(),
    })));
    for stream in [false, true] {
        let mut req = request();
        req["stream"] = stream.into();
        let response = post_json(&app, "/v1/chat/completions", req.clone(), None).await;
        assert_eq!(response.status(), StatusCode::OK);
        body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let response = post(&app, req).await;
        assert_eq!(response.status(), StatusCode::OK);
        body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
    }
    assert_eq!(starts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn anthropic_cors_allows_version_header_and_exposes_request_id() {
    let app = router(
        state(Arc::new(MockBackend))
            .with_cors_origins(vec!["https://client.example".parse().unwrap()]),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/messages")
                .header("origin", "https://client.example")
                .header("access-control-request-method", "POST")
                .header(
                    "access-control-request-headers",
                    "x-api-key,anthropic-version,content-type",
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response.headers()["access-control-allow-headers"]
            .to_str()
            .unwrap()
            .contains("anthropic-version")
    );
}

#[tokio::test]
async fn protocols_prepare_identical_backend_requests_and_reject_oversized_tool_history() {
    let backend = Arc::new(Fixture::tools());
    let state = state(backend.clone());
    let app = router(state.clone());
    let mut req = request();
    req["system"] = "Be brief.".into();
    req["tools"] = tools();
    req["temperature"] = 0.into();
    assert_eq!(post(&app, req.clone()).await.status(), StatusCode::OK);
    let openai = json!({"model":"mock","max_tokens":96,"temperature":0,"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}],
        "tools":[{"type":"function","function":{"name":"add","description":"Add integers","parameters":tools()[0]["input_schema"]}}]});
    assert_eq!(
        post_json(&app, "/v1/chat/completions", openai, None)
            .await
            .status(),
        StatusCode::OK
    );
    let requests = backend.requests.lock().unwrap();
    assert_eq!(requests[0].prompt, requests[1].prompt);
    assert_eq!(
        serde_json::to_value(&requests[0].messages).unwrap(),
        serde_json::to_value(&requests[1].messages).unwrap()
    );
    assert_eq!(
        requests[0].tool_config.as_ref().unwrap().tools,
        requests[1].tool_config.as_ref().unwrap().tools
    );
    assert_eq!(requests[0].stop, requests[1].stop);
    assert_eq!(requests[0].temperature, requests[1].temperature);
    drop(requests);
    let mut manifest = dummy_manifest();
    manifest.format = ModelFormat::Gguf;
    fs::write(
        state
            .store
            .model_dir("mock")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let limited = router(state.with_chat_context_size(Some(512)));
    req["tools"][0]["description"] = "x".repeat(2000).into();
    let response = post(&limited, req).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        response_json(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("exceed")
    );
}

#[tokio::test]
#[ignore = "requires local TCP and anthropic==0.86.0; run with WERK_TEST_ANTHROPIC_PYTHON"]
async fn official_anthropic_sdk_text_stream_and_tool_loop() {
    let python = std::env::var("WERK_TEST_ANTHROPIC_PYTHON")
        .expect("set Python executable with anthropic==0.86.0 installed");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = router(state(Arc::new(Fixture::tools())).with_api_keys(vec!["sdk-test".into()]));
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let result = tokio::process::Command::new(&python)
        .args([
            "tests/anthropic_sdk.py",
            "--base-url",
            &format!("http://{address}"),
            "--model",
            "mock",
            "--api-key",
            "sdk-test",
            "--fixture",
            "--catalog-size",
            "64",
        ])
        .output()
        .await
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let result = tokio::process::Command::new(python)
        .args([
            "utils/benchmarks/anthropic_api.py",
            "--base-url",
            &format!("http://{address}"),
            "--model",
            "mock",
            "--api-key",
            "sdk-test",
            "--rounds",
            "1",
        ])
        .output()
        .await
        .unwrap();
    server.abort();
    assert!(
        result.status.success(),
        "benchmark: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = String::from_utf8(result.stdout).unwrap();
    let rows: Vec<Value> = output
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 6);
    assert_eq!(
        rows[..4]
            .iter()
            .map(|v| v["protocol"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["openai", "anthropic", "anthropic", "openai"]
    );
    assert!(
        rows[..4]
            .iter()
            .all(|v| v["text_sha256"] == rows[0]["text_sha256"])
    );
}

#[tokio::test]
async fn vision_and_tool_result_images_keep_content_order_and_use_modality_aware_route() {
    let backend = Arc::new(Fixture::tools());
    let base = state(backend.clone());
    let app = router(ApiState::new_with_default_model_and_prompt_options(
        base.store.as_ref().clone(),
        backend.clone(),
        None,
        Some(Arc::new(|_, _, _| {
            Ok(ChatTemplateOptions {
                default_source: ChatTemplateSource::Model,
                model_template_preferred: true,
                override_name: Some("model"),
            })
        })),
    ));
    let mut req = request();
    req["metadata"] = json!({"user_id":"client-42"});
    req["messages"] = json!([{"role":"user","content":[
        {"type":"text","text":"Before"},
        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}},
        {"type":"text","text":"After"},
        {"type":"image","source":{"type":"url","url":"https://example.test/image.png"}}
    ]}]);
    assert_eq!(post(&app, req.clone()).await.status(), StatusCode::OK);
    let captured = backend.requests.lock().unwrap()[0].clone();
    assert_eq!(
        captured.image_urls,
        [
            "data:image/png;base64,AQID",
            "https://example.test/image.png"
        ]
    );
    let parts = serde_json::to_value(&captured.messages[0].content).unwrap();
    assert_eq!(parts[0]["text"], "Before");
    assert_eq!(parts[2]["text"], "After");
    assert_eq!(backend.sessions.load(Ordering::SeqCst), 0);
    req["messages"] = json!([
        {"role":"assistant","content":[{"type":"tool_use","id":"image_call","name":"add","input":{}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"image_call","is_error":true,
            "content":[{"type":"text","text":"Here"},{"type":"image","source":{"type":"url","url":"https://example.test/image.png"}}]}]}
    ]);
    assert_eq!(post(&app, req).await.status(), StatusCode::OK);
    let captured = backend.requests.lock().unwrap()[1].clone();
    assert_eq!(captured.messages[1].role, "tool");
    let parts = serde_json::to_value(&captured.messages[1].content).unwrap();
    assert_eq!(parts[0]["text"], "{\"is_error\":true}");
    assert_eq!(parts[1]["text"], "Here");
    assert_eq!(parts[2]["type"], "image_url");
}

#[tokio::test]
async fn count_endpoint_uses_native_counter_without_generation_or_context_trimming() {
    let backend = Arc::new(Fixture::tools());
    let state = state(backend.clone()).with_chat_context_size(Some(32));
    let mut manifest = dummy_manifest();
    manifest.format = ModelFormat::Gguf;
    fs::write(
        state
            .store
            .model_dir("mock")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let app = router(state);
    let req = json!({"model":"mock","system":"Be brief.","tools":tools(),"messages":[{"role":"user","content":"hello".repeat(500)}]});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(req.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers().contains_key("request-id"));
    assert_eq!(response_json(response).await, json!({"input_tokens":23}));
    let captured = backend.requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].messages.len(), 2);
    assert_eq!(
        captured[0]
            .tool_config
            .as_ref()
            .unwrap()
            .tools
            .as_ref()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(backend.sessions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn count_without_native_support_fails_instead_of_estimating() {
    let app = router(state(Arc::new(MockBackend)));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(
                    json!({"model":"mock","messages":[{"role":"user","content":"hi"}]}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value = response_json(response).await;
    assert!(value.get("input_tokens").is_none());
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("native chat token counting")
    );
}
