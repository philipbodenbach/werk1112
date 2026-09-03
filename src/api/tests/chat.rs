use super::support::*;
use crate::{
    backend::GeneratedAssistantMessage,
    openai::{
        ChatCompletionFunctionCall, ChatCompletionFunctionCallDelta, ChatCompletionToolCall,
        ChatCompletionToolCallDelta,
    },
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone)]
struct VisionRecordingBackend {
    request: Arc<std::sync::Mutex<Option<GenerateRequest>>>,
}

#[derive(Clone)]
struct ToolCallingBackend {
    calls: Arc<AtomicUsize>,
    session_calls: Arc<AtomicUsize>,
    request: Arc<std::sync::Mutex<Option<GenerateRequest>>>,
}

impl GenerationBackend for ToolCallingBackend {
    fn supports_tool_calling(&self, _manifest: &ModelManifest, _has_images: bool) -> bool {
        true
    }

    fn start_chat_session(
        &self,
        _manifest: &ModelManifest,
        _seed: Option<u64>,
    ) -> anyhow::Result<Option<Box<dyn crate::backend::ChatGenerationSession>>> {
        self.session_calls.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("tool requests must bypass the text chat-session cache")
    }

    fn generate(
        &self,
        _manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> anyhow::Result<GenerateResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.request.lock().unwrap() = Some(request);
        Ok(GenerateResponse {
            text: String::new(),
            assistant_message: Some(GeneratedAssistantMessage {
                content: None,
                tool_calls: Some(vec![ChatCompletionToolCall {
                    id: "call_next".to_string(),
                    kind: "function".to_string(),
                    function: ChatCompletionFunctionCall {
                        name: "get_weather".to_string(),
                        arguments: r#"{"city":"Hamburg"}"#.to_string(),
                    },
                }]),
            }),
            prompt_tokens: 17,
            completion_tokens: 8,
            finish_reason: "tool_calls".to_string(),
            timings: GenerationTimings::default(),
            backend_diagnostics: Vec::new(),
        })
    }

    fn generate_stream(
        &self,
        _manifest: ModelManifest,
        request: GenerateRequest,
    ) -> GenerateStream {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.request.lock().unwrap() = Some(request);
        let events = vec![
            Ok(GenerateStreamEvent::ToolCallDelta(vec![
                ChatCompletionToolCallDelta {
                    index: 0,
                    id: Some("call_weather".to_string()),
                    kind: Some("function".to_string()),
                    function: Some(ChatCompletionFunctionCallDelta {
                        name: Some("get_weather".to_string()),
                        arguments: Some(r#"{"city":"#.to_string()),
                    }),
                },
                ChatCompletionToolCallDelta {
                    index: 1,
                    id: Some("call_time".to_string()),
                    kind: Some("function".to_string()),
                    function: Some(ChatCompletionFunctionCallDelta {
                        name: Some("get_time".to_string()),
                        arguments: Some(r#"{"zone":"#.to_string()),
                    }),
                },
            ])),
            Ok(GenerateStreamEvent::ToolCallDelta(vec![
                ChatCompletionToolCallDelta {
                    index: 0,
                    id: None,
                    kind: None,
                    function: Some(ChatCompletionFunctionCallDelta {
                        name: None,
                        arguments: Some(r#"Berlin"}"#.to_string()),
                    }),
                },
                ChatCompletionToolCallDelta {
                    index: 1,
                    id: None,
                    kind: None,
                    function: Some(ChatCompletionFunctionCallDelta {
                        name: None,
                        arguments: Some(r#"Europe/Berlin"}"#.to_string()),
                    }),
                },
            ])),
            Ok(GenerateStreamEvent::Done {
                finish_reason: "tool_calls".to_string(),
                prompt_tokens: 12,
                completion_tokens: 9,
                timings: GenerationTimings::default(),
                backend_diagnostics: Vec::new(),
            }),
        ];
        Box::pin(tokio_stream::iter(events))
    }
}

#[derive(Clone)]
struct UnsupportedToolBackend {
    calls: Arc<AtomicUsize>,
}

impl GenerationBackend for UnsupportedToolBackend {
    fn generate(
        &self,
        _manifest: &ModelManifest,
        _request: GenerateRequest,
    ) -> anyhow::Result<GenerateResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("unsupported backend must not be called")
    }

    fn generate_stream(
        &self,
        _manifest: ModelManifest,
        _request: GenerateRequest,
    ) -> GenerateStream {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(tokio_stream::empty())
    }
}

impl GenerationBackend for VisionRecordingBackend {
    fn start_chat_session(
        &self,
        _manifest: &ModelManifest,
        _seed: Option<u64>,
    ) -> anyhow::Result<Option<Box<dyn crate::backend::ChatGenerationSession>>> {
        anyhow::bail!("image requests must bypass modality-blind API chat sessions")
    }

    fn generate(
        &self,
        _manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> anyhow::Result<GenerateResponse> {
        *self.request.lock().unwrap() = Some(request);
        Ok(GenerateResponse {
            text: "visual inspection complete".to_string(),
            assistant_message: None,
            prompt_tokens: 32,
            completion_tokens: 3,
            finish_reason: "stop".to_string(),
            timings: GenerationTimings::default(),
            backend_diagnostics: Vec::new(),
        })
    }

    fn generate_stream(
        &self,
        _manifest: ModelManifest,
        _request: GenerateRequest,
    ) -> GenerateStream {
        Box::pin(tokio_stream::iter(vec![Err(
            "streaming is not used by this test".to_string(),
        )]))
    }
}

#[tokio::test]
async fn models_and_chat_routes_use_openai_shapes() {
    let store = test_store();
    let manifest = ModelManifest {
        id: "mock".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::Unknown,
        architecture: None,
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "mock".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: Default::default(),
    };
    fs::create_dir_all(store.model_dir("mock")).unwrap();
    fs::write(
        store
            .model_dir("mock")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let app = router(ApiState::new(store, Arc::new(MockBackend)));

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"mock","messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["object"], "chat.completion");
    assert_eq!(value["choices"][0]["message"]["role"], "assistant");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"mock","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let stream = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(stream.contains("\"object\":\"chat.completion.chunk\""));
    assert!(stream.contains("\"content\":\"hello\""));
    assert!(stream.contains("data: [DONE]"));
}

#[tokio::test]
async fn tool_calling_nonstream_preserves_request_history_and_nullable_response() {
    let store = test_store();
    install_tool_chat_model(&store);
    let calls = Arc::new(AtomicUsize::new(0));
    let session_calls = Arc::new(AtomicUsize::new(0));
    let recorded = Arc::new(std::sync::Mutex::new(None));
    let app = router(ApiState::new(
        store,
        Arc::new(ToolCallingBackend {
            calls: calls.clone(),
            session_calls: session_calls.clone(),
            request: recorded.clone(),
        }),
    ));
    let payload = json!({
        "model": "tool-model",
        "messages": [
            {"role": "user", "content": "Continue the lookup"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_previous",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Berlin\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_previous",
                "content": "{\"temperature_c\":21}"
            }
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"}
        },
        "parallel_tool_calls": false
    });

    let response = post_json(&app, "/v1/chat/completions", payload.clone(), None).await;

    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["choices"][0]["message"]["content"], Value::Null);
    assert_eq!(
        value["choices"][0]["message"]["tool_calls"][0],
        json!({
            "id": "call_next",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\":\"Hamburg\"}"
            }
        })
    );
    assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(session_calls.load(Ordering::SeqCst), 0);

    let request = recorded.lock().unwrap().clone().unwrap();
    assert!(request.requires_tool_calling());
    assert_eq!(request.messages.len(), 3);
    assert!(request.messages[1].content.is_none());
    assert_eq!(
        request.messages[1].tool_calls.as_ref().unwrap()[0]
            .function
            .arguments,
        r#"{"city":"Berlin"}"#
    );
    assert_eq!(
        request.messages[2].tool_call_id.as_deref(),
        Some("call_previous")
    );
    let config = serde_json::to_value(request.tool_config.as_ref().unwrap()).unwrap();
    assert_eq!(config["tools"], payload["tools"]);
    assert_eq!(config["tool_choice"], payload["tool_choice"]);
    assert_eq!(config["parallel_tool_calls"], false);
}

#[tokio::test]
async fn tool_calling_stream_preserves_indexes_fragments_finish_and_done() {
    let store = test_store();
    install_tool_chat_model(&store);
    let calls = Arc::new(AtomicUsize::new(0));
    let session_calls = Arc::new(AtomicUsize::new(0));
    let app = router(ApiState::new(
        store,
        Arc::new(ToolCallingBackend {
            calls: calls.clone(),
            session_calls: session_calls.clone(),
            request: Arc::new(std::sync::Mutex::new(None)),
        }),
    ));

    let response = post_json(
        &app,
        "/v1/chat/completions",
        json!({
            "model": "tool-model",
            "stream": true,
            "messages": [{"role": "user", "content": "Weather and time?"}],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}}
            }],
            "tool_choice": "auto",
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
    let data = stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect::<Vec<_>>();
    assert_eq!(data.last().copied(), Some("[DONE]"));
    let chunks = data[..data.len() - 1]
        .iter()
        .map(|value| serde_json::from_str::<Value>(value).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(
        chunks[1]["choices"][0]["delta"]["tool_calls"][0]["index"],
        0
    );
    assert_eq!(
        chunks[1]["choices"][0]["delta"]["tool_calls"][1]["index"],
        1
    );
    assert_eq!(
        chunks[1]["choices"][0]["delta"]["tool_calls"][0]["id"],
        "call_weather"
    );
    assert_eq!(
        chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        r#"{"city":"#
    );
    assert_eq!(
        chunks[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        r#"Berlin"}"#
    );
    assert_eq!(chunks[3]["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(session_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn unsupported_backend_rejects_tool_calling_before_backend_execution() {
    let store = test_store();
    install_tool_chat_model(&store);
    let calls = Arc::new(AtomicUsize::new(0));
    let app = router(ApiState::new(
        store,
        Arc::new(UnsupportedToolBackend {
            calls: calls.clone(),
        }),
    ));

    let response = post_json(
        &app,
        "/v1/chat/completions",
        json!({
            "model": "tool-model",
            "messages": [{"role": "user", "content": "Use a tool"}],
            "tools": [{
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}}
            }]
        }),
        None,
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value = response_json(response).await;
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert_eq!(value["error"]["code"], "unsupported_tool_calling");
    assert_eq!(value["error"]["param"], "tools");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("--backend vllm")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

fn install_tool_chat_model(store: &ModelStore) {
    let manifest = ModelManifest {
        id: "tool-model".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::Unknown,
        architecture: None,
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "mock".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: Default::default(),
    };
    fs::create_dir_all(store.model_dir("tool-model")).unwrap();
    fs::write(
        store
            .model_dir("tool-model")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

#[tokio::test]
async fn vision_chat_preserves_ordered_parts_and_bypasses_text_session_cache() {
    let store = test_store();
    let manifest = ModelManifest {
        id: "qwen3-vl-test".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::SafeTensors,
        architecture: Some("qwen3_vl".to_string()),
        tokenizer_path: None,
        config_path: None,
        model_path: Some("model.safetensors".to_string()),
        backend: "test".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: ModelMetadata {
            tasks: vec![
                InferenceTask::TextGeneration,
                InferenceTask::ImageUnderstanding,
            ],
            input_modalities: vec![InputModality::Text, InputModality::Image],
            output_modalities: vec![OutputModality::Text],
            ..Default::default()
        },
    };
    fs::create_dir_all(store.model_dir("qwen3-vl-test")).unwrap();
    fs::write(
        store
            .model_dir("qwen3-vl-test")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let recorded = Arc::new(std::sync::Mutex::new(None));
    let backend = VisionRecordingBackend {
        request: recorded.clone(),
    };
    let resolver: PromptOptionsResolver = Arc::new(|_, _, has_images| {
        assert!(has_images);
        Ok(ChatTemplateOptions {
            default_source: ChatTemplateSource::Model,
            model_template_preferred: true,
            override_name: None,
        })
    });
    let app = router(ApiState::new_with_default_model_and_prompt_options(
        store,
        Arc::new(backend),
        None,
        Some(resolver),
    ));
    let payload = json!({
        "model": "qwen3-vl-test",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare the heading"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,AAAA", "detail": "high"
                }},
                {"type": "text", "text": "Then check the button alignment"},
                {"type": "input_image", "image_url": "https://example.test/page-2.png"}
            ]
        }]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let request = recorded.lock().unwrap().clone().unwrap();
    assert_eq!(
        request.image_urls,
        [
            "data:image/png;base64,AAAA".to_string(),
            "https://example.test/page-2.png".to_string(),
        ]
    );
    let content = request.messages[0].content.as_ref().unwrap();
    let crate::openai::MessageContent::Parts(parts) = content else {
        panic!("vision message content was flattened")
    };
    assert_eq!(
        parts
            .iter()
            .map(|part| part.kind.as_str())
            .collect::<Vec<_>>(),
        ["text", "image_url", "text", "input_image"]
    );
    let crate::openai::ImageUrlSpec::Object(image) = parts[1].image_url.as_ref().unwrap() else {
        panic!("image detail object was flattened")
    };
    assert_eq!(image.detail.as_deref(), Some("high"));
}

#[tokio::test]
async fn model_retrieve_route_returns_openai_model_object() {
    let app = model_retrieve_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/mock")
                .header(header::AUTHORIZATION, "Bearer sk-test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response_json(response).await,
        json!({
            "id": "mock",
            "object": "model",
            "created": 1,
            "owned_by": "local"
        })
    );
}

#[tokio::test]
async fn model_retrieve_route_returns_openai_not_found_error() {
    let app = model_retrieve_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/missing")
                .header(header::AUTHORIZATION, "Bearer sk-test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let value = response_json(response).await;
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert_eq!(value["error"]["param"], "model");
    assert!(
        value["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("model 'missing' is not installed"))
    );
}

#[tokio::test]
async fn model_retrieve_route_requires_bearer_auth() {
    let app = model_retrieve_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/mock")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response.headers().get(header::WWW_AUTHENTICATE).unwrap(),
        "Bearer"
    );
    assert_eq!(
        response_json(response).await["error"]["message"],
        "missing bearer token"
    );
}

fn model_retrieve_app() -> Router {
    let store = test_store();
    let manifest = ModelManifest {
        id: "mock".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::Unknown,
        architecture: None,
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "mock".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: Default::default(),
    };
    fs::create_dir_all(store.model_dir("mock")).unwrap();
    fs::write(
        store
            .model_dir("mock")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    router(ApiState::new(store, Arc::new(MockBackend)).with_api_keys(vec!["sk-test".to_string()]))
}

#[tokio::test]
async fn server_api_keys_require_matching_bearer_token() {
    let store = test_store();
    let manifest = ModelManifest {
        id: "mock".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::Unknown,
        architecture: None,
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "mock".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: Default::default(),
    };
    fs::create_dir_all(store.model_dir("mock")).unwrap();
    fs::write(
        store
            .model_dir("mock")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let app = router(
        ApiState::new(store, Arc::new(MockBackend)).with_api_keys(vec!["sk-test".to_string()]),
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response.headers().get(header::WWW_AUTHENTICATE).unwrap(),
        "Bearer"
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header(header::AUTHORIZATION, "Bearer wrong")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header(header::AUTHORIZATION, "Bearer sk-test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn server_default_model_is_used_when_request_omits_model() {
    let store = test_store();
    let manifest = ModelManifest {
        id: "mock".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::Unknown,
        architecture: None,
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "mock".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: Default::default(),
    };
    fs::create_dir_all(store.model_dir("mock")).unwrap();
    fs::write(
        store
            .model_dir("mock")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let app = router(ApiState::new_with_default_model(
        store,
        Arc::new(MockBackend),
        Some("mock".to_string()),
    ));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["model"], "mock");
}

#[tokio::test]
async fn chat_route_rejects_image_generation_only_models_before_backend_loading() {
    let store = test_store();
    let manifest = ModelManifest {
        id: "image-only".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::SafeTensors,
        architecture: Some("flux_transformer_2d_model".to_string()),
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "onnxruntime".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: ModelMetadata {
            tasks: vec![InferenceTask::ImageGeneration],
            input_modalities: vec![InputModality::Text],
            output_modalities: vec![OutputModality::Image],
            ..Default::default()
        },
    };
    fs::create_dir_all(store.model_dir("image-only")).unwrap();
    fs::write(
        store
            .model_dir("image-only")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let app = router(ApiState::new(store, Arc::new(MockBackend)));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"image-only","messages":[{"role":"user","content":"draw a cat"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value = response_json(response).await;
    assert_eq!(value["error"]["param"], "model");
    assert!(
        value["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("/v1/images/generations"))
    );
}

#[tokio::test]
async fn chat_route_uses_prompt_options_resolver_before_generation() {
    let store = test_store();
    let manifest = ModelManifest {
        id: "mock".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::SafeTensors,
        architecture: Some("starcoder2".to_string()),
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "onnxruntime".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: Default::default(),
    };
    fs::create_dir_all(store.model_dir("mock")).unwrap();
    fs::write(
        store
            .model_dir("mock")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let resolver: PromptOptionsResolver = Arc::new(|_, _, _| {
        Ok(ChatTemplateOptions {
            default_source: ChatTemplateSource::Model,
            model_template_preferred: true,
            override_name: None,
        })
    });
    let app = router(ApiState::new_with_default_model_and_prompt_options(
        store,
        Arc::new(PromptEchoBackend),
        None,
        Some(resolver),
    ));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"mock","messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["choices"][0]["message"]["content"], "hi");
}

#[tokio::test]
async fn chat_route_trims_complete_old_turns_for_gguf_context() {
    let store = test_store();
    let manifest = ModelManifest {
        id: "mock-gguf".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::Gguf,
        architecture: Some("llama".to_string()),
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "llama-server".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: Default::default(),
    };
    fs::create_dir_all(store.model_dir("mock-gguf")).unwrap();
    fs::write(
        store
            .model_dir("mock-gguf")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let app =
        router(ApiState::new(store, Arc::new(PromptEchoBackend)).with_chat_context_size(Some(256)));
    let old = format!("OLD_MARKER {}", "x".repeat(700));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "mock-gguf",
                        "max_tokens": 32,
                        "messages": [
                            {"role": "user", "content": old},
                            {"role": "assistant", "content": "OLD_REPLY"},
                            {"role": "user", "content": "LATEST_MESSAGE"}
                        ]
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    let prompt = value["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(prompt.contains("LATEST_MESSAGE"));
    assert!(!prompt.contains("OLD_MARKER"));
    assert!(!prompt.contains("OLD_REPLY"));
}

#[tokio::test]
async fn chat_route_reports_when_current_message_cannot_fit_gguf_context() {
    let store = test_store();
    let manifest = ModelManifest {
        id: "mock-gguf".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::Gguf,
        architecture: Some("llama".to_string()),
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "llama-server".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata: Default::default(),
    };
    fs::create_dir_all(store.model_dir("mock-gguf")).unwrap();
    fs::write(
        store
            .model_dir("mock-gguf")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let app =
        router(ApiState::new(store, Arc::new(PromptEchoBackend)).with_chat_context_size(Some(256)));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "mock-gguf",
                        "max_tokens": 32,
                        "messages": [{"role": "user", "content": "x".repeat(700)}]
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value = response_json(response).await;
    assert_eq!(value["error"]["param"], "messages");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("current message is too large")
    );
}
