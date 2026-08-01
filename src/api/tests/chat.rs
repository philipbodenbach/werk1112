use super::support::*;

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
