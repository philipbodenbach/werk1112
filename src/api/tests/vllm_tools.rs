use super::support::*;
use crate::backend::VllmBackend;
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

struct MockHttpResponse {
    content_type: &'static str,
    warning: bool,
    body: String,
}

impl MockHttpResponse {
    fn json(value: Value) -> Self {
        Self {
            content_type: "application/json",
            warning: false,
            body: value.to_string(),
        }
    }

    fn sse(events: Vec<Value>) -> Self {
        let mut body = events
            .into_iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>();
        body.push_str("data: [DONE]\n\n");
        Self {
            content_type: "text/event-stream",
            warning: false,
            body,
        }
    }
}

struct MockVllmServer {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl MockVllmServer {
    fn start(responses: Vec<MockHttpResponse>) -> Self {
        Self::start_at(responses, "/v1/chat/completions")
    }

    fn start_at(responses: Vec<MockHttpResponse>, path: &'static str) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let handle = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            for response in responses {
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(
                                Instant::now() < deadline,
                                "timed out waiting for Werk's vLLM request"
                            );
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("mock vLLM accept failed: {error}"),
                    }
                };
                let request = read_json_request(&mut stream, path);
                recorded.lock().unwrap().push(request);
                write_response(&mut stream, response);
            }
        });
        Self {
            url,
            requests,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> Vec<Value> {
        self.handle.take().unwrap().join().unwrap();
        self.requests.lock().unwrap().clone()
    }
}

fn read_json_request(stream: &mut TcpStream, path: &str) -> Value {
    // macOS may inherit O_NONBLOCK from the listening socket.
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).unwrap();
    assert_eq!(request_line.trim_end(), format!("POST {path} HTTP/1.1"));
    let mut content_length = None;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            content_length = Some(value.trim().parse::<usize>().unwrap());
        }
    }
    let mut body = vec![0; content_length.expect("Werk request has Content-Length")];
    reader.read_exact(&mut body).unwrap();
    serde_json::from_slice(&body).unwrap()
}

fn write_response(stream: &mut TcpStream, response: MockHttpResponse) {
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n",
        response.content_type,
        response.body.len(),
        if response.warning {
            "Warning: 299 - grammar fallback\r\n"
        } else {
            ""
        }
    );
    stream.write_all(headers.as_bytes()).unwrap();
    stream.write_all(response.body.as_bytes()).unwrap();
    stream.flush().unwrap();
}

fn vllm_app(server_url: String) -> Router {
    router(vllm_state(server_url))
}

fn vllm_state(server_url: String) -> ApiState {
    let store = test_store();
    let manifest = ModelManifest {
        storage: Default::default(),
        id: "qwen-test".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::SafeTensors,
        architecture: Some("qwen2".to_string()),
        tokenizer_path: None,
        config_path: None,
        model_path: Some("model.safetensors".to_string()),
        backend: "vllm".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: ModelMetadata {
            tasks: vec![InferenceTask::TextGeneration],
            input_modalities: vec![InputModality::Text],
            output_modalities: vec![OutputModality::Text],
            ..Default::default()
        },
    };
    fs::create_dir_all(store.model_dir(&manifest.id)).unwrap();
    fs::write(
        store
            .model_dir(&manifest.id)
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let backend =
        VllmBackend::with_mock_http_server(store.clone(), server_url, "Qwen-Test".to_string());
    ApiState::new(store, Arc::new(backend))
}

fn weather_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "city": {"type": ["string", "null"], "minLength": 1}
                },
                "required": ["city"],
                "additionalProperties": false
            }
        }
    })
}

#[tokio::test]
async fn rich_openai_responses_preserve_multiple_choices_logprobs_and_schema_controls() {
    let logprobs = json!({"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]});
    let server = MockVllmServer::start(vec![MockHttpResponse::json(json!({
        "id":"native-id","object":"chat.completion","created":1,"model":"Qwen-Test",
        "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop","logprobs":logprobs},
            {"index":1,"message":{"role":"assistant","content":"world"},"finish_reason":"stop","logprobs":logprobs}],
        "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}))]);
    let app = vllm_app(server.url.clone());
    let response=post_json(&app,"/v1/chat/completions",json!({"model":"qwen-test","messages":[{"role":"user","content":"Hello"}],
        "n":2,"logprobs":true,"top_logprobs":1,"frequency_penalty":0.2,"presence_penalty":0.1,"logit_bias":{"12":-5},"reasoning_effort":"low"}),None).await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["model"], "qwen-test");
    assert_eq!(value["choices"].as_array().unwrap().len(), 2);
    assert_eq!(value["choices"][1]["logprobs"], logprobs);
    let sent = server.finish();
    assert_eq!(sent[0]["n"], 2);
    assert_eq!(sent[0]["logit_bias"]["12"], -5);
    assert_eq!(sent[0]["reasoning_effort"], "low");
    assert_eq!(sent[0]["chat_template_kwargs"]["enable_thinking"], true);
}

#[tokio::test]
async fn vllm_reasoning_effort_overrides_thinking_per_request_for_json_and_sse() {
    for stream in [false, true] {
        // Change the mode on one backend, then omit it to prove no override
        // leaks into later requests that should retain the native default.
        let efforts = [Some("low"), Some("none"), None];
        let responses = efforts.iter().map(|_| {
            if stream {
                MockHttpResponse::sse(vec![json!({
                    "choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}
                })])
            } else {
                MockHttpResponse::json(json!({
                    "choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}
                }))
            }
        }).collect();
        let server = MockVllmServer::start(responses);
        let app = vllm_app(server.url.clone());
        for effort in efforts {
            let mut payload = json!({"model":"qwen-test","stream":stream,"max_tokens":32,
                "messages":[{"role":"user","content":"Hello"}],"frequency_penalty":0.1});
            if let Some(effort) = effort {
                payload["reasoning_effort"] = effort.into();
            }
            let response = post_json(&app, "/v1/chat/completions", payload, None).await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
            if stream {
                assert!(
                    String::from_utf8(bytes.to_vec())
                        .unwrap()
                        .contains("data: [DONE]")
                );
            } else {
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["choices"][0]["message"]["content"], "answer");
            }
        }
        let sent = server.finish();
        assert_eq!(sent.len(), efforts.len());
        for (body, effort) in sent.iter().zip(efforts) {
            if let Some(effort) = effort {
                assert_eq!(body["reasoning_effort"], effort);
                assert_eq!(
                    body["chat_template_kwargs"]["enable_thinking"],
                    effort != "none"
                );
            } else {
                assert!(body.get("reasoning_effort").is_none());
                assert!(body.get("chat_template_kwargs").is_none());
            }
            assert_eq!(body["stream"], stream);
        }
    }
}

#[tokio::test]
async fn vllm_reasoning_effort_rejects_values_outside_native_schema_before_json_or_sse() {
    let server = MockVllmServer::start(vec![]);
    let app = vllm_app(server.url.clone());
    for effort in [
        json!("off"),
        json!("default"),
        json!("custom_thinking"),
        json!(42),
        json!(0.5),
    ] {
        for stream in [false, true] {
            let response = post_json(
                &app,
                "/v1/chat/completions",
                json!({
                    "model":"qwen-test","messages":[{"role":"user","content":"Hello"}],
                    "reasoning_effort":effort,"stream":stream
                }),
                None,
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
            let body = response_json(response).await;
            assert_eq!(body["error"]["code"], "invalid_reasoning_effort");
            assert_eq!(body["error"]["param"], "reasoning_effort");
            assert_eq!(
                body["error"]["supported_values"],
                json!(["none", "minimal", "low", "medium", "high", "xhigh", "max"])
            );
            assert_eq!(body["error"]["supported_types"], json!(["string"]));
            assert_eq!(body["error"]["values_scope"], "backend");
            assert_eq!(body["error"]["values_depend_on_model"], true);
            assert!(body["error"]["message"].as_str().unwrap().contains("vLLM"));
            let message = body["error"]["message"].as_str().unwrap();
            for value in ["none", "minimal", "low", "medium", "high", "xhigh", "max"] {
                assert!(message.contains(value), "{message}");
            }
        }
    }
    assert!(server.finish().is_empty());
}

#[tokio::test]
async fn extended_controls_fail_when_native_enforcement_or_metadata_is_missing() {
    for (field, value, warning, expected) in [
        (
            "response_format",
            json!({"type":"json_object"}),
            true,
            "could not enforce",
        ),
        (
            "logprobs",
            json!(true),
            false,
            "did not provide requested logprobs",
        ),
    ] {
        let mut native = MockHttpResponse::json(
            json!({"choices":[{"index":0,"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}),
        );
        native.warning = warning;
        let server = MockVllmServer::start(vec![native]);
        let app = vllm_app(server.url.clone());
        let mut request =
            json!({"model":"qwen-test","messages":[{"role":"user","content":"hello"}]});
        request[field] = value;
        let response = post_json(&app, "/v1/chat/completions", request, None).await;
        assert!(!response.status().is_success());
        let text = String::from_utf8(
            body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(text.contains(expected), "{text}");
        server.finish();
    }
}

#[tokio::test]
async fn anthropic_native_template_stop_is_not_a_client_stop_sequence() {
    let server = MockVllmServer::start(vec![MockHttpResponse::json(
        json!({"choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop","stop_reason":"<|im_end|>"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}),
    )]);
    let app = vllm_app(server.url.clone());
    let request = json!({"model":"qwen-test","max_tokens":32,"top_k":10,"messages":[{"role":"user","content":"hello"}]});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value: Value = serde_json::from_slice(
        &body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(value["stop_reason"], "end_turn");
    assert!(value["stop_sequence"].is_null());
    server.finish();
}

#[tokio::test]
async fn anthropic_schema_top_k_and_matched_stop_reach_native_backend() {
    let server = MockVllmServer::start(vec![MockHttpResponse::json(
        json!({"id":"native","model":"Qwen-Test",
        "choices":[{"index":0,"message":{"role":"assistant","content":"{\"ok\":true}"},"finish_reason":"stop","stop_reason":"END"}],
        "usage":{"prompt_tokens":10,"completion_tokens":5}}),
    )]);
    let app = vllm_app(server.url.clone());
    let body = json!({"model":"qwen-test","max_tokens":32,"top_k":20,"stop_sequences":["END"],
        "output_config":{"format":{"type":"json_schema","schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}},
        "messages":[{"role":"user","content":"Reply in JSON"}]});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["stop_reason"], "stop_sequence");
    assert_eq!(value["stop_sequence"], "END");
    assert_eq!(value["content"][0]["text"], "{\"ok\":true}");
    let sent = server.finish();
    assert_eq!(sent[0]["top_k"], 20);
    assert_eq!(sent[0]["response_format"]["json_schema"]["strict"], true);
    assert!(sent[0].get("__werk_matched_stop").is_none());
}

#[tokio::test]
async fn rich_streams_keep_choice_indexes_usage_and_error_termination() {
    for include_usage in [false, true] {
        let server = MockVllmServer::start(vec![MockHttpResponse::sse(vec![
            json!({"choices":[{"index":1,"delta":{"content":"second"},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"content":"first"},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"},{"index":1,"delta":{},"finish_reason":"length"}]}),
            json!({"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}),
        ])]);
        let app = vllm_app(server.url.clone());
        let response=post_json(&app,"/v1/chat/completions",json!({"model":"qwen-test","messages":[{"role":"user","content":"hello"}],"n":2,"stream":true,"stream_options":{"include_usage":include_usage}}),None).await;
        let text = String::from_utf8(
            body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(text.contains("[DONE]"), "{text}");
        assert_eq!(text.contains("prompt_tokens"), include_usage);
        assert!(text.contains("\"index\":1"));
        assert!(text.contains("\"index\":0"));
        server.finish();
    }
    let server = MockVllmServer::start(vec![MockHttpResponse::sse(vec![
        json!({"choices":[{"index":0,"delta":{"content":"unfinished"},"finish_reason":null}]}),
    ])]);
    let app = vllm_app(server.url.clone());
    let response=post_json(&app,"/v1/chat/completions",json!({"model":"qwen-test","messages":[{"role":"user","content":"hello"}],"presence_penalty":0.1,"stream":true}),None).await;
    let text = String::from_utf8(
        body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(text.contains("error"));
    assert!(!text.contains("[DONE]"));
    server.finish();
}

#[tokio::test]
async fn anthropic_rich_stream_reports_real_matched_stop_and_final_usage() {
    let server = MockVllmServer::start(vec![MockHttpResponse::sse(vec![
        json!({"choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop","stop_reason":"END"}]}),
        json!({"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}),
    ])]);
    let app = vllm_app(server.url.clone());
    let request = json!({"model":"qwen-test","messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stop_sequences":["END"],"stream":true});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let text = String::from_utf8(
        body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(text.contains("\"stop_reason\":\"stop_sequence\""), "{text}");
    assert!(text.contains("\"stop_sequence\":\"END\""));
    assert!(text.contains("\"input_tokens\":12"));
    assert!(text.contains("event: message_stop"));
    assert!(!text.contains("event: error"));
    server.finish();
}

#[tokio::test]
async fn anthropic_count_uses_vllm_tokenizer_with_tool_history_and_physical_model() {
    let server = MockVllmServer::start_at(
        vec![MockHttpResponse::json(
            json!({"count": 47, "tokens": [1, 2]}),
        )],
        "/tokenize",
    );
    let app = vllm_app(server.url.clone());
    let body = json!({
        "model":"qwen-test",
        "system":"Use tools when needed.",
        "tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{}}}],
        "messages":[
            {"role":"user","content":"Weather?"},
            {"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"get_weather","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"21 C"}]}
        ]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response_json(response).await, json!({"input_tokens":47}));
    let requests = server.finish();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request["model"], "Qwen-Test");
    assert_eq!(request["messages"][0]["role"], "system");
    assert_eq!(request["messages"][2]["tool_calls"][0]["id"], "call_1");
    assert_eq!(request["messages"][3]["tool_call_id"], "call_1");
    assert_eq!(request["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(request["add_generation_prompt"], true);
    assert_eq!(request["add_special_tokens"], false);
    assert!(request.get("max_tokens").is_none());
}

#[tokio::test]
async fn vllm_non_streaming_tools_and_tool_result_continue_through_werk_handler() {
    let tool_call = json!({
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"Berlin\"}"
        }
    });
    let server = MockVllmServer::start(vec![
        MockHttpResponse::json(json!({
            "id": "vllm-completion-1",
            "object": "chat.completion",
            "created": 1,
            "model": "Qwen-Test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [tool_call.clone()]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
        })),
        MockHttpResponse::json(json!({
            "id": "vllm-completion-2",
            "object": "chat.completion",
            "created": 2,
            "model": "Qwen-Test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "It is 21 C in Berlin."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 31, "completion_tokens": 9, "total_tokens": 40}
        })),
    ]);
    let app = vllm_app(server.url.clone());
    let tool = weather_tool();

    let response = post_json(
        &app,
        "/v1/chat/completions",
        json!({
            "model": "qwen-test",
            "messages": [{"role": "user", "content": "What is the weather in Berlin?"}],
            "tools": [tool.clone()],
            "tool_choice": "auto",
            "parallel_tool_calls": false
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let response = response_json(response).await;
    assert_eq!(response["object"], "chat.completion");
    assert_eq!(response["model"], "qwen-test");
    assert!(
        response["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("chatcmpl-"))
    );
    assert!(response["choices"][0]["message"]["content"].is_null());
    assert_eq!(
        response["choices"][0]["message"]["tool_calls"][0],
        tool_call
    );
    assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(response["usage"]["prompt_tokens"], 20);
    assert_eq!(response["usage"]["completion_tokens"], 8);
    assert_eq!(response["usage"]["total_tokens"], 28);

    let continuation_messages = json!([
        {"role": "user", "content": "What is the weather in Berlin?"},
        {"role": "assistant", "content": null, "tool_calls": [tool_call.clone()]},
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "{\"temperature\":21}"
        }
    ]);
    let continuation = post_json(
        &app,
        "/v1/chat/completions",
        json!({
            "model": "qwen-test",
            "messages": continuation_messages.clone(),
            "tools": [tool.clone()],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "parallel_tool_calls": false
        }),
        None,
    )
    .await;
    assert_eq!(continuation.status(), StatusCode::OK);
    let continuation = response_json(continuation).await;
    assert_eq!(
        continuation["choices"][0]["message"]["content"],
        "It is 21 C in Berlin."
    );
    assert_eq!(continuation["choices"][0]["finish_reason"], "stop");

    let requests = server.finish();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["model"], "Qwen-Test");
    assert_eq!(requests[0]["stream"], false);
    assert!(requests[0].get("stream_options").is_none());
    assert_eq!(
        requests[0]["messages"],
        json!([{"role": "user", "content": "What is the weather in Berlin?"}])
    );
    assert_eq!(requests[0]["tools"], json!([tool.clone()]));
    assert_eq!(requests[0]["tool_choice"], "auto");
    assert_eq!(requests[0]["parallel_tool_calls"], false);
    assert_eq!(
        requests[0]["tools"][0]["function"]["parameters"],
        tool["function"]["parameters"]
    );
    assert_eq!(requests[1]["messages"], continuation_messages);
    assert_eq!(requests[1]["tools"], json!([tool]));
    assert_eq!(
        requests[1]["tool_choice"],
        json!({"type": "function", "function": {"name": "get_weather"}})
    );
    assert_eq!(requests[1]["parallel_tool_calls"], false);
}

#[tokio::test]
async fn vllm_streaming_tool_deltas_keep_indexes_fragments_finish_and_done() {
    let server = MockVllmServer::start(vec![MockHttpResponse::sse(vec![
        json!({
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        }),
        json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {
                    "index": 0,
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "get_", "arguments": ""}
                },
                {
                    "index": 1,
                    "id": "call_time",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": ""}
                }
            ]}, "finish_reason": null}]
        }),
        json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"name": "weather", "arguments": "{\"city\""}}
            ]}, "finish_reason": null}]
        }),
        json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ":\"Berlin\"}"}},
                {"index": 1, "function": {"arguments": "{\"zone\":\"UTC\"}"}}
            ]}, "finish_reason": null}]
        }),
        json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 22, "completion_tokens": 11, "total_tokens": 33}
        }),
    ])]);
    let app = vllm_app(server.url.clone());

    let response = post_json(
        &app,
        "/v1/chat/completions",
        json!({
            "model": "qwen-test",
            "stream": true,
            "messages": [{"role": "user", "content": "Weather and UTC time?"}],
            "tools": [
                weather_tool(),
                {"type": "function", "function": {
                    "name": "get_time",
                    "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}}
                }}
            ],
            "tool_choice": "required",
            "parallel_tool_calls": true
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let stream = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(stream.contains("data: [DONE]"));
    let events = stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str::<Value>(data).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");
    let tool_events = events
        .iter()
        .filter(|event| event["choices"][0]["delta"].get("tool_calls").is_some())
        .collect::<Vec<_>>();
    assert_eq!(tool_events.len(), 3);
    assert_eq!(
        tool_events[0]["choices"][0]["delta"]["tool_calls"],
        json!([
            {
                "index": 0,
                "id": "call_weather",
                "type": "function",
                "function": {"name": "get_", "arguments": ""}
            },
            {
                "index": 1,
                "id": "call_time",
                "type": "function",
                "function": {"name": "get_time", "arguments": ""}
            }
        ])
    );
    assert_eq!(
        tool_events[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "weather"
    );
    assert_eq!(
        tool_events[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\""
    );
    assert_eq!(
        tool_events[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        ":\"Berlin\"}"
    );
    assert_eq!(
        tool_events[2]["choices"][0]["delta"]["tool_calls"][1]["index"],
        1
    );
    assert_eq!(
        tool_events[2]["choices"][0]["delta"]["tool_calls"][1]["function"]["arguments"],
        "{\"zone\":\"UTC\"}"
    );
    assert_eq!(
        events.last().unwrap()["choices"][0]["finish_reason"],
        "tool_calls"
    );

    let requests = server.finish();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["stream"], true);
    assert_eq!(requests[0]["stream_options"]["include_usage"], true);
    assert_eq!(requests[0]["tool_choice"], "required");
    assert_eq!(requests[0]["parallel_tool_calls"], true);
    assert_eq!(requests[0]["tools"].as_array().unwrap().len(), 2);
}

#[test]
fn vllm_token_count_preserves_request_reasoning_template_options() {
    for (effort, thinking) in [("low", true), ("none", false)] {
        let server = MockVllmServer::start_at(
            vec![MockHttpResponse::json(json!({"count":47,"tokens":[]}))],
            "/tokenize",
        );
        let state = vllm_state(server.url.clone());
        let manifest = state.store.get("qwen-test").unwrap();
        let request = GenerateRequest {
            prompt: String::new(),
            messages: serde_json::from_value(json!([{"role":"user","content":"Weather?"}]))
                .unwrap(),
            image_urls: vec![],
            max_tokens: 32,
            temperature: None,
            top_p: None,
            stop: vec![],
            seed: None,
            stream_granularity: crate::backend::StreamGranularity::Chunk,
            verbose: false,
            debug: false,
            tool_config: Some(crate::backend::ToolCallingConfig {
                tools: Some(serde_json::from_value(json!([weather_tool()])).unwrap()),
                tool_choice: None,
                parallel_tool_calls: None,
            }),
        };
        let count = state
            .backend
            .count_api_tokens(
                &manifest,
                request,
                BTreeMap::from([("reasoning_effort".into(), json!(effort))]),
            )
            .unwrap();
        assert_eq!(count, 47);
        let requests = server.finish();
        assert_eq!(
            requests[0]["chat_template_kwargs"]["reasoning_effort"],
            effort
        );
        assert_eq!(
            requests[0]["chat_template_kwargs"]["enable_thinking"],
            thinking
        );
        assert_eq!(requests[0]["tools"][0]["function"]["name"], "get_weather");
    }
}

#[tokio::test]
async fn extended_api_preserves_native_timing_formats_in_request_history() {
    // Exercise the shared raw HTTP transport and API observation with both
    // native timing formats, plus plain OpenAI usage without phase durations.
    for format in ["llama.cpp", "omlx", "plain"] {
        for stream in [false, true] {
            let mut metadata = json!({"choices": [], "usage": {
                "prompt_tokens": 100, "completion_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 80}
            }});
            match format {
                "llama.cpp" => {
                    metadata["timings"] = json!({
                        "prompt_n": 20, "prompt_ms": 500,
                        "predicted_n": 30, "predicted_ms": 2000
                    })
                }
                "omlx" => {
                    metadata["usage"]["prompt_eval_duration"] = json!(0.5);
                    metadata["usage"]["generation_duration"] = json!(2.0);
                }
                _ => {}
            }
            let upstream = if stream {
                MockHttpResponse::sse(vec![
                    json!({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}),
                    metadata,
                ])
            } else {
                metadata["choices"] = json!([{"index": 0,
                    "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]);
                MockHttpResponse::json(metadata)
            };
            let server = MockVllmServer::start(vec![upstream]);
            let state = vllm_state(server.url.clone());
            let telemetry = state.telemetry.clone();
            let app = router(state);
            let response = post_json(
                &app,
                "/v1/chat/completions",
                json!({
                    "model": "qwen-test", "messages": [{"role": "user", "content": "hello"}],
                    "frequency_penalty": 0.2, "stream": stream
                }),
                None,
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            let _ = body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
            let snapshot = telemetry.snapshot();
            assert_eq!(snapshot.totals.completed, 1, "{format}, stream={stream}");
            let row = &snapshot.requests[0];
            assert_eq!(row.output_tokens, Some(30));
            assert_eq!(row.cached_tokens, Some(80));
            assert_eq!(
                row.decode_tokens_per_second,
                (format != "plain").then_some(15.),
                "{format}, stream={stream}"
            );
            assert_eq!(
                row.prefill_tokens_per_second,
                (format != "plain").then_some(40.)
            );
            server.finish();
        }
    }
}
