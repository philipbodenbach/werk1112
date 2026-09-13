use super::support::*;
use crate::openai::{ChatRuntimeOptions, OmlxChatOptions};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Default)]
struct OptionsBackend {
    selected: Option<ChatRuntimeOptions>,
    requests: Arc<Mutex<Vec<(Option<ChatRuntimeOptions>, GenerateRequest)>>>,
    configure_calls: Arc<AtomicUsize>,
    session_calls: Arc<AtomicUsize>,
}

impl GenerationBackend for OptionsBackend {
    fn with_chat_options(
        &self,
        _manifest: &ModelManifest,
        options: &ChatRuntimeOptions,
    ) -> anyhow::Result<Arc<dyn GenerationBackend>> {
        self.configure_calls.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(Self {
            selected: Some(options.clone()),
            ..self.clone()
        }))
    }

    fn start_chat_session(
        &self,
        _manifest: &ModelManifest,
        _seed: Option<u64>,
    ) -> anyhow::Result<Option<Box<dyn crate::backend::ChatGenerationSession>>> {
        self.session_calls.fetch_add(1, Ordering::SeqCst);
        assert!(
            self.selected.is_none(),
            "request-specific options must bypass the shared session cache"
        );
        Ok(None)
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> anyhow::Result<GenerateResponse> {
        self.requests
            .lock()
            .unwrap()
            .push((self.selected.clone(), request.clone()));
        MockBackend.generate(manifest, request)
    }

    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        self.requests
            .lock()
            .unwrap()
            .push((self.selected.clone(), request.clone()));
        MockBackend.generate_stream(manifest, request)
    }
}

fn install_model(store: &ModelStore) {
    let manifest = ModelManifest {
        id: "options-model".into(),
        source: ModelSource::LocalPath {
            path: "fixture".into(),
        },
        format: ModelFormat::Unknown,
        architecture: Some("qwen3".into()),
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "fixture".into(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
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
}

fn request(options: Option<Value>, stream: bool) -> Value {
    let mut body = json!({"model":"options-model","stream":stream,
        "messages":[{"role":"system","content":"Answer in German."},{"role":"user","content":"Hi"}]});
    if let Some(options) = options {
        body["werk"] = options;
    }
    body
}

#[tokio::test]
async fn stream_usage_is_opt_in_and_follows_finish_before_done() {
    let store = test_store();
    install_model(&store);
    let app = router(ApiState::new(store, Arc::new(MockBackend)));
    for enabled in [None, Some(false), Some(true)] {
        let mut payload = request(None, true);
        if let Some(enabled) = enabled {
            payload["stream_options"] = json!({"include_usage": enabled});
        }
        let response = post_json(&app, "/v1/chat/completions", payload, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let events: Vec<_> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect();
        assert_eq!(events.last(), Some(&"[DONE]"));
        let chunks: Vec<Value> = events[..events.len() - 1]
            .iter()
            .map(|event| serde_json::from_str(event).unwrap())
            .collect();
        assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "hello");
        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
        if enabled == Some(true) {
            assert_eq!(chunks.len(), 4);
            assert_eq!(chunks[3]["choices"], json!([]));
            assert_eq!(
                chunks[3]["usage"],
                json!({"prompt_tokens":2,"completion_tokens":1,"total_tokens":3})
            );
            assert_eq!(chunks[3]["id"], chunks[0]["id"]);
        } else {
            assert_eq!(chunks.len(), 3);
            assert!(chunks.iter().all(|chunk| chunk.get("usage").is_none()));
        }
    }
}

#[tokio::test]
async fn omlx_chat_options_are_request_scoped_on_json_and_streaming_paths() {
    let store = test_store();
    install_model(&store);
    let backend = OptionsBackend::default();
    let resolver: PromptOptionsResolver =
        Arc::new(|_, _, _| anyhow::bail!("default route must not resolve configured requests"));
    let app = router(ApiState::new_with_default_model_and_prompt_options(
        store,
        Arc::new(backend.clone()),
        None,
        Some(resolver),
    ));
    let cases = [
        (false, false, 8192, json!(1024)),
        (true, true, 0, json!(0)),
        (false, false, 8192, json!("auto")),
        (true, true, 0, json!("auto")),
    ];
    for (stream, thinking, budget, ngram) in &cases {
        let response = post_json(
            &app,
            "/v1/chat/completions",
            request(
                Some(json!({"omlx":{"thinking":thinking,"expert_cache_mb":budget,"ngram_cache_mb":ngram}})),
                *stream,
            ),
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        if *stream {
            assert!(String::from_utf8_lossy(&bytes).contains("[DONE]"));
        }
    }
    assert_eq!(backend.configure_calls.load(Ordering::SeqCst), cases.len());
    assert_eq!(backend.session_calls.load(Ordering::SeqCst), 0);
    let calls = backend.requests.lock().unwrap();
    assert_eq!(calls.len(), cases.len());
    for ((selected, generation), (_, thinking, budget, ngram)) in calls.iter().zip(cases) {
        assert_eq!(
            selected,
            &Some(ChatRuntimeOptions {
                omlx: Some(OmlxChatOptions {
                    thinking: Some(thinking),
                    expert_cache_mb: Some(budget),
                    ngram_cache_mb: Some(serde_json::from_value(ngram).unwrap()),
                })
            })
        );
        assert_eq!(generation.messages.len(), 2);
        assert_eq!(generation.messages[0].role, "system");
        assert_eq!(generation.messages[1].role, "user");
        assert_eq!(
            generation.messages[1].content.as_ref().unwrap().as_text(),
            "Hi"
        );
    }
}

#[tokio::test]
async fn empty_chat_options_inherit_and_explicit_values_do_not_leak_to_later_requests() {
    let store = test_store();
    install_model(&store);
    let backend = OptionsBackend::default();
    let app = router(ApiState::new(store, Arc::new(backend.clone())));
    for options in [
        Some(json!({"omlx":{"thinking":false}})),
        None,
        Some(json!({})),
        Some(json!({"omlx":{}})),
    ] {
        let response = post_json(&app, "/v1/chat/completions", request(options, false), None).await;
        assert_eq!(response.status(), StatusCode::OK);
    }
    assert_eq!(backend.configure_calls.load(Ordering::SeqCst), 1);
    let calls = backend.requests.lock().unwrap();
    assert!(calls[0].0.is_some());
    assert!(calls[1..].iter().all(|(options, _)| options.is_none()));
}

#[tokio::test]
async fn chat_runtime_options_reject_invalid_values_and_unsupported_backends_before_generation() {
    let store = test_store();
    install_model(&store);
    let app = router(ApiState::new(store, Arc::new(MockBackend)));
    for stream in [false, true] {
        let response = post_json(
            &app,
            "/v1/chat/completions",
            request(Some(json!({"omlx":{"thinking":false}})), stream),
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["code"], "unsupported_chat_options");
        assert_eq!(body["error"]["param"], "werk.omlx");
    }
    for options in [
        json!({"omlx":{"expert_cache_mb":1048577}}),
        json!({"omlx":{"expert_cache_mb":-1}}),
        json!({"omlx":{"expert_cache_mb":1.5}}),
        json!({"omlx":{"expert_cache_mb":true}}),
        json!({"omlx":{"ngram_cache_mb":1048577}}),
        json!({"omlx":{"ngram_cache_mb":-1}}),
        json!({"omlx":{"ngram_cache_mb":1.5}}),
        json!({"omlx":{"ngram_cache_mb":true}}),
        json!({"omlx":{"ngram_cache_mb":"automatic"}}),
        json!({"omlx":{"ngram_cache_mb":"1024"}}),
        json!({"omlx":{"thinking":0}}),
        json!({"omlx":{"thinking":"false"}}),
        json!({"omlx":{"unknown":1}}),
        json!({"unknown":{}}),
    ] {
        let response = post_json(
            &app,
            "/v1/chat/completions",
            request(Some(options), false),
            None,
        )
        .await;
        assert!(response.status().is_client_error());
    }
}

#[tokio::test]
async fn omlx_chat_options_reject_images_before_backend_configuration() {
    let store = test_store();
    install_model(&store);
    let backend = OptionsBackend::default();
    let app = router(ApiState::new(store, Arc::new(backend.clone())));
    let mut payload = request(Some(json!({"omlx":{"thinking":false}})), false);
    payload["messages"][1]["content"] =
        json!([{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]);
    let response = post_json(&app, "/v1/chat/completions", payload, None).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(backend.configure_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn discovery_advertises_chat_options_without_claiming_model_compatibility() {
    let app = router(ApiState::new(test_store(), Arc::new(MockBackend)));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/werk/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    let capability = body["data"]["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["id"] == "api.chat.omlx_options")
        .unwrap();
    assert_eq!(capability["status"], "supported");
    assert_eq!(
        capability["operations"],
        json!(["thinking", "expert_cache_mb", "ngram_cache_mb"])
    );
}
