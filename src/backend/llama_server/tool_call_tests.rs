use super::*;
use crate::backend::{StreamGranularity, ToolCallingConfig};
use crate::openai::{ChatMessage, ToolChoice};

fn tool_request() -> GenerateRequest {
    GenerateRequest {
        prompt: "Inspect and call the tools".into(),
        messages: Vec::new(),
        image_urls: Vec::new(),
        max_tokens: 64,
        temperature: Some(0.0),
        top_p: None,
        stop: Vec::new(),
        seed: None,
        stream_granularity: StreamGranularity::Chunk,
        verbose: false,
        debug: false,
        tool_config: Some(ToolCallingConfig {
            tools: Some(serde_json::from_value(json!([
                {"type":"function","function":{"name":"weather","description":"Weather in a place",
                  "parameters":{"type":"object","properties":{"place":{"type":"string"}},"required":["place"],"additionalProperties":false},"strict":true}},
                {"type":"function","function":{"name":"echo","parameters":{"type":"object","properties":{"text":{"type":"string"}}}}}
            ])).unwrap()),
            tool_choice: Some(serde_json::from_value(json!("auto")).unwrap()),
            parallel_tool_calls: Some(true),
        }),
    }
}

fn messages(value: Value) -> Vec<ChatMessage> {
    serde_json::from_value(value).unwrap()
}

// Exercise real TCP request serialization and response parsing without a model
// download, GPU, process discovery, or mutations to global runtime variables.
fn mock_http(
    responses: Vec<(&'static str, &'static str, String)>,
) -> (String, thread::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        let mut requests = Vec::new();
        for (path, mime, response) in responses {
            let (socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(socket);
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            assert_eq!(line.trim(), format!("POST {path} HTTP/1.1"));
            let mut length = 0;
            loop {
                line.clear();
                reader.read_line(&mut line).unwrap();
                if line.trim().is_empty() {
                    break;
                }
                if let Some((key, value)) = line.split_once(':') {
                    if key.eq_ignore_ascii_case("content-length") {
                        length = value.trim().parse().unwrap();
                    }
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            requests.push(serde_json::from_slice(&body).unwrap());
            let socket = reader.get_mut();
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", response.len()).unwrap();
            // Deliberately split through SSE lines and UTF-8 byte boundaries.
            for fragment in response.as_bytes().chunks(7) {
                socket.write_all(fragment).unwrap();
            }
            socket.flush().unwrap();
        }
        requests
    });
    (url, handle)
}

fn sse(events: Vec<Value>, done: bool) -> String {
    let mut data = events
        .into_iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>();
    if done {
        data.push_str("data: [DONE]\n\n");
    }
    data
}

fn parallel_call_events() -> Vec<Value> {
    vec![
        json!({"choices":[{"index":0,"delta":{"reasoning_content":"select tools"}}]}),
        json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":1,"id":"call_echo","type":"function","function":{"name":"echo","arguments":"{\"text\":"}},
            {"index":0,"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{\"place\":"}}
        ]}}]}),
        json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"function":{"arguments":"\"Berlin\"}"}},
            {"index":1,"function":{"arguments":"\"Grüße\"}"}}
        ]}}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],
            "usage":{"prompt_tokens":112,"completion_tokens":30,"prompt_tokens_details":{"cached_tokens":80}}}),
    ]
}

#[test]
fn native_tools_without_history_use_chat_endpoint_and_preserve_all_choices() {
    for choice in [
        json!("auto"),
        json!("required"),
        json!("none"),
        json!({"type":"function","function":{"name":"weather"}}),
    ] {
        let mut request = tool_request();
        request.tool_config.as_mut().unwrap().tool_choice =
            Some(serde_json::from_value::<ToolChoice>(choice.clone()).unwrap());
        request.tool_config.as_mut().unwrap().parallel_tool_calls = Some(false);
        let (url, server) = mock_http(vec![(
            "/v1/chat/completions",
            "text/event-stream",
            sse(
                vec![
                    json!({"choices":[{"index":0,"delta":{"content":"ready"},"finish_reason":"stop"}]}),
                ],
                true,
            ),
        )]);
        let completion = complete_request(&url, &request, None, Instant::now()).unwrap();
        assert_eq!(completion.text, "ready");
        let bodies = server.join().unwrap();
        assert_eq!(bodies[0]["tool_choice"], choice);
        assert_eq!(bodies[0]["parallel_tool_calls"], false);
        assert_eq!(bodies[0]["messages"][0]["content"], request.prompt);
        assert_eq!(bodies[0]["tools"][0]["function"]["strict"], true);
    }
}

#[test]
fn native_streaming_forwards_deltas_and_assembles_parallel_tool_only_result() {
    let request = tool_request();
    let (url, server) = mock_http(vec![(
        "/v1/chat/completions",
        "text/event-stream",
        sse(parallel_call_events(), true),
    )]);
    let (tx, mut rx) = mpsc::channel(16);
    let completion = complete_request(&url, &request, Some(tx), Instant::now()).unwrap();
    assert_eq!(completion.text, "");
    assert_eq!(completion.finish_reason, "tool_calls");
    assert_eq!(completion.prompt_tokens, 112);
    assert_eq!(completion.cached_prompt_tokens, Some(80));
    assert!(completion.first_token_seconds > 0.0);
    let message = completion.assistant_message.unwrap();
    assert_eq!(message.content, None);
    let calls = message.tool_calls.unwrap();
    assert_eq!(calls[0].id, "call_weather");
    assert_eq!(calls[0].function.arguments, r#"{"place":"Berlin"}"#);
    assert_eq!(calls[1].id, "call_echo");
    assert_eq!(calls[1].function.arguments, r#"{"text":"Grüße"}"#);
    let first = rx.try_recv().unwrap().unwrap();
    let second = rx.try_recv().unwrap().unwrap();
    match (first, second) {
        (GenerateStreamEvent::ToolCallDelta(first), GenerateStreamEvent::ToolCallDelta(second)) => {
            assert_eq!(first[0].index, 1);
            assert_eq!(first[0].id.as_deref(), Some("call_echo"));
            assert_eq!(second[0].index, 0);
            assert_eq!(
                second[0].function.as_ref().unwrap().arguments.as_deref(),
                Some("\"Berlin\"}")
            );
        }
        other => panic!("expected untouched tool deltas, received {other:?}"),
    }
    assert!(rx.try_recv().is_err());
    assert_eq!(server.join().unwrap()[0]["parallel_tool_calls"], true);
}

#[test]
fn native_nonstream_generation_assembles_tools_and_optional_assistant_text() {
    for text in [None, Some("I will check.")] {
        let mut events = parallel_call_events();
        if let Some(text) = text {
            events.insert(0, json!({"choices":[{"delta":{"content":text}}]}));
        }
        let (url, server) = mock_http(vec![(
            "/v1/chat/completions",
            "text/event-stream",
            sse(events, true),
        )]);
        let completion = complete_request(&url, &tool_request(), None, Instant::now()).unwrap();
        let message = completion.assistant_message.unwrap();
        assert_eq!(message.content.as_deref(), text);
        assert_eq!(message.tool_calls.unwrap().len(), 2);
        server.join().unwrap();
    }
}

#[test]
fn native_vision_tool_continuation_preserves_images_calls_and_result_ids() {
    let mut request = tool_request();
    request.messages = messages(json!([
        {"role":"user","content":[{"type":"text","text":"Inspect it"},{"type":"input_image","image_url":{"url":"data:image/png;base64,AAAA","detail":"high"}}]},
        {"role":"assistant","content":null,"tool_calls":[{"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{\"place\":\"Berlin\"}"}}]},
        {"role":"tool","tool_call_id":"call_weather","name":"weather","content":"Sunny"}
    ]));
    let (url, server) = mock_http(vec![(
        "/v1/chat/completions",
        "text/event-stream",
        sse(
            vec![json!({"choices":[{"delta":{"content":"Looks sunny."},"finish_reason":"stop"}]})],
            true,
        ),
    )]);
    assert!(request_has_images(&request));
    validate_llama_image_sources(&request).unwrap();
    let completion = complete_request(&url, &request, None, Instant::now()).unwrap();
    assert_eq!(completion.text, "Looks sunny.");
    let bodies = server.join().unwrap();
    assert_eq!(
        bodies[0]["messages"][0]["content"][1],
        json!({"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA","detail":"high"}})
    );
    assert!(bodies[0]["messages"][1]["content"].is_null());
    assert_eq!(
        bodies[0]["messages"][1]["tool_calls"][0]["id"],
        "call_weather"
    );
    assert_eq!(bodies[0]["messages"][2]["tool_call_id"], "call_weather");
    assert!(bodies[0]["tools"].is_array());
}

#[test]
fn native_api_nonstream_preserves_tool_only_response_and_tool_schema() {
    let request = tool_request();
    let expected = json!({"id":"chatcmpl-tools","object":"chat.completion","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{\"place\":\"Berlin\"}"}}]}}],"usage":{"prompt_tokens":99,"completion_tokens":20,"total_tokens":119}});
    let (url, server) = mock_http(vec![(
        "/v1/chat/completions",
        "application/json",
        expected.to_string(),
    )]);
    let response = crate::backend::openai_transport::generate_api(
        &url,
        None,
        chat_api_body(&request, false),
        None,
    )
    .unwrap();
    assert_eq!(response, expected);
    let bodies = server.join().unwrap();
    assert_eq!(bodies[0]["stream"], false);
    assert!(bodies[0].get("stream_options").is_none());
    assert_eq!(bodies[0]["tools"][0]["function"]["name"], "weather");
}

#[test]
fn token_count_templates_tool_schemas_and_tool_result_history() {
    let mut request = tool_request();
    request.messages = messages(json!([
        {"role":"user","content":"Check Berlin"},
        {"role":"assistant","content":null,"tool_calls":[{"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{\"place\":\"Berlin\"}"}}]},
        {"role":"tool","tool_call_id":"call_weather","content":"Sunny"}
    ]));
    let (url, server) = mock_http(vec![
        (
            "/apply-template",
            "application/json",
            json!({"prompt":"tools+history+assistant"}).to_string(),
        ),
        (
            "/tokenize",
            "application/json",
            json!({"tokens":[1,2,3,4,5,6]}).to_string(),
        ),
    ]);
    assert_eq!(count_request_tokens(&url, &request).unwrap(), 6);
    let bodies = server.join().unwrap();
    assert_eq!(bodies[0]["tools"][0]["function"]["name"], "weather");
    assert_eq!(bodies[0]["messages"][2]["tool_call_id"], "call_weather");
    assert_eq!(bodies[0]["add_generation_prompt"], true);
    assert_eq!(bodies[1]["content"], "tools+history+assistant");
    assert_eq!(bodies[1]["parse_special"], true);
}

#[test]
fn truncated_or_incomplete_tool_stream_is_an_error() {
    for (events, done, expected) in [
        (
            vec![
                json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_bad","type":"function","function":{"arguments":"{"}}]}}]}),
            ],
            true,
            "incomplete tool call",
        ),
        (
            vec![
                json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_bad","type":"function","function":{"name":"weather","arguments":"{"}}]}}]}),
            ],
            false,
            "before a completion marker",
        ),
    ] {
        let (url, server) = mock_http(vec![(
            "/v1/chat/completions",
            "text/event-stream",
            sse(events, done),
        )]);
        let error = complete_request(&url, &tool_request(), None, Instant::now())
            .err()
            .unwrap();
        assert!(error.to_string().contains(expected), "{error}");
        server.join().unwrap();
    }
}

#[test]
fn jinja_defaults_on_when_supported_and_respects_explicit_overrides() {
    assert!(supported_args_from_help("--jinja, --no-jinja").jinja);
    assert!(!supported_args_from_help("--jinja-config").jinja);
    let mut args = Vec::new();
    append_jinja_default(&mut args, true, &[]);
    assert_eq!(args, ["--jinja"]);
    for extra in [
        vec!["--jinja".into()],
        vec!["--no-jinja".into()],
        vec!["--jinja=false".into()],
    ] {
        let mut args = Vec::new();
        append_jinja_default(&mut args, true, &extra);
        assert!(args.is_empty());
    }
    let mut args = Vec::new();
    append_jinja_default(&mut args, false, &[]);
    assert!(args.is_empty());
}

#[test]
fn tool_calls_require_complete_json_and_a_matching_finish_reason() {
    for (arguments, reason, expected) in [
        ("{", "tool_calls", "invalid tool-call arguments"),
        ("[]", "tool_calls", "must be a JSON object"),
        ("{}", "length", "without a tool_calls finish reason"),
        ("{}", "stop", "without a tool_calls finish reason"),
    ] {
        let events = vec![
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"name":"weather","arguments":arguments}}]},"finish_reason":reason}]}),
        ];
        let (url, server) = mock_http(vec![(
            "/v1/chat/completions",
            "text/event-stream",
            sse(events, true),
        )]);
        let error = complete_request(&url, &tool_request(), None, Instant::now())
            .err()
            .unwrap();
        assert!(error.to_string().contains(expected), "{error}");
        server.join().unwrap();
    }
    let (url, server) = mock_http(vec![(
        "/v1/chat/completions",
        "text/event-stream",
        sse(
            vec![json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]})],
            true,
        ),
    )]);
    let error = complete_request(&url, &tool_request(), None, Instant::now())
        .err()
        .unwrap();
    assert!(error.to_string().contains("without tool calls"));
    server.join().unwrap();
}

#[test]
fn native_stream_rejects_tool_data_after_done_without_forwarding_it() {
    let mut response = sse(
        vec![json!({"choices":[{"delta":{"content":"ready"},"finish_reason":"stop"}]})],
        true,
    );
    response.push_str(&sse(
        vec![json!({"choices":[{
            "delta":{"tool_calls":[{"index":0,"id":"call_after_done","type":"function",
                "function":{"name":"weather","arguments":"{}"}}]},
            "finish_reason":"tool_calls"
        }]})],
        false,
    ));
    let (url, server) = mock_http(vec![(
        "/v1/chat/completions",
        "text/event-stream",
        response,
    )]);
    let (tx, mut rx) = mpsc::channel(16);
    let error = complete_request(&url, &tool_request(), Some(tx), Instant::now())
        .err()
        .expect("post-DONE data must fail");
    assert!(error.to_string().contains("data after DONE"), "{error}");
    assert!(
        matches!(rx.try_recv().unwrap().unwrap(), GenerateStreamEvent::TextChunk(text) if text == "ready")
    );
    assert!(rx.try_recv().is_err(), "post-DONE tool call was forwarded");
    server.join().unwrap();
}

#[test]
fn snapshot_requests_pin_the_slot_while_api_requests_allow_native_cache_selection() {
    for chat in [false, true] {
        for slot in [None, Some(STATE_SLOT_ID)] {
            let mut request = tool_request();
            if !chat {
                request.tool_config = None;
            }
            let path = if chat {
                "/v1/chat/completions"
            } else {
                "/completion"
            };
            let event = if chat {
                json!({"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]})
            } else {
                json!({"content":"ok","stop":true})
            };
            let (url, server) =
                mock_http(vec![(path, "text/event-stream", sse(vec![event], true))]);
            let result =
                complete_request_in_slot(&url, &request, None, Instant::now(), slot).unwrap();
            assert_eq!(result.text, "ok");
            let bodies = server.join().unwrap();
            assert_eq!(bodies[0]["cache_prompt"], true);
            assert_eq!(
                bodies[0].get("id_slot").and_then(Value::as_u64),
                slot.map(u64::from)
            );
        }
    }
}
