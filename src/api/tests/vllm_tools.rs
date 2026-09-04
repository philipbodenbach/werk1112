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
    body: String,
}

impl MockHttpResponse {
    fn json(value: Value) -> Self {
        Self {
            content_type: "application/json",
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
                let request = read_json_request(&mut stream);
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

fn read_json_request(stream: &mut TcpStream) -> Value {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).unwrap();
    assert_eq!(
        request_line.trim_end(),
        "POST /v1/chat/completions HTTP/1.1"
    );
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
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.content_type,
        response.body.len()
    );
    stream.write_all(headers.as_bytes()).unwrap();
    stream.write_all(response.body.as_bytes()).unwrap();
    stream.flush().unwrap();
}

fn vllm_app(server_url: String) -> Router {
    let store = test_store();
    let manifest = ModelManifest {
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
    router(ApiState::new(store, Arc::new(backend)))
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
