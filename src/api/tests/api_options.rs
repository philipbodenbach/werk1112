use super::support::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct UnsupportedBackend;

impl GenerationBackend for UnsupportedBackend {
    fn generate(&self, _: &ModelManifest, _: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        panic!("unsupported options must not start generation")
    }

    fn generate_stream(&self, _: ModelManifest, _: GenerateRequest) -> GenerateStream {
        panic!("unsupported options must not start streaming")
    }

    fn generate_api(
        &self,
        _: &ModelManifest,
        _: GenerateRequest,
        _: BTreeMap<String, Value>,
        _: Option<tokio::sync::mpsc::Sender<Result<Value, String>>>,
    ) -> anyhow::Result<Value> {
        panic!("unsupported options must fail before extended generation")
    }
}

struct ReasoningLimitsBackend;

impl GenerationBackend for ReasoningLimitsBackend {
    fn validate_api_options(
        &self,
        _: &ModelManifest,
        _: &GenerateRequest,
        _: &BTreeMap<String, Value>,
    ) -> anyhow::Result<()> {
        Err(
            anyhow::Error::new(crate::backend::ApiOptionError::reasoning_effort(
                "fixture native schema",
                Some(&["none", "minimal", "low", "medium", "high", "xhigh", "max"]),
                &["string", "null"],
            ))
            .context("fixture option validation"),
        )
    }

    fn generate(&self, _: &ModelManifest, _: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        panic!("invalid reasoning must not start generation")
    }

    fn generate_stream(&self, _: ModelManifest, _: GenerateRequest) -> GenerateStream {
        panic!("invalid reasoning must not start streaming")
    }
}

#[derive(Default)]
struct SelectiveBackend {
    validations: AtomicUsize,
    generations: AtomicUsize,
}

impl GenerationBackend for SelectiveBackend {
    fn validate_api_options(
        &self,
        _: &ModelManifest,
        request: &GenerateRequest,
        options: &BTreeMap<String, Value>,
    ) -> anyhow::Result<()> {
        self.validations.fetch_add(1, Ordering::SeqCst);
        assert_eq!(request.max_tokens, 32);
        if options.contains_key("reasoning_effort") {
            anyhow::bail!("fixture does not support reasoning_effort")
        }
        Ok(())
    }

    fn generate_api(
        &self,
        _: &ModelManifest,
        _: GenerateRequest,
        options: BTreeMap<String, Value>,
        tx: Option<tokio::sync::mpsc::Sender<Result<Value, String>>>,
    ) -> anyhow::Result<Value> {
        self.generations.fetch_add(1, Ordering::SeqCst);
        assert_eq!(options.get("frequency_penalty"), Some(&json!(0.2)));
        if let Some(tx) = tx {
            tx.blocking_send(Ok(
                json!({"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}),
            ))?;
        }
        Ok(
            json!({"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}),
        )
    }

    fn generate(&self, _: &ModelManifest, _: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        panic!("extended options must use extended generation")
    }

    fn generate_stream(&self, _: ModelManifest, _: GenerateRequest) -> GenerateStream {
        panic!("extended options must use extended streaming")
    }
}

fn app(backend: Arc<dyn GenerationBackend>) -> Router {
    let store = test_store();
    let manifest = ModelManifest {
        storage: Default::default(),
        id: "api-options".into(),
        source: ModelSource::LocalPath {
            path: "fixture".into(),
        },
        format: ModelFormat::Unknown,
        architecture: None,
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
    router(ApiState::new(store, backend))
}

fn request(stream: bool) -> Value {
    json!({"model":"api-options","stream":stream,"max_tokens":32,
        "messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"low"})
}

async fn rejection(app: &Router, payload: Value) -> Value {
    let response = post_json(app, "/v1/chat/completions", payload, None).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn assert_rejected(app: &Router, stream: bool, expected_message: &str) {
    let value = rejection(app, request(stream)).await;
    assert_eq!(value["error"].as_object().unwrap().len(), 4);
    assert!(value["error"]["param"].is_null());
    assert_eq!(value["error"]["code"], "unsupported_api_options");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains(expected_message)
    );
}

#[tokio::test]
async fn extended_api_options_default_backend_rejects_before_json_or_sse() {
    let app = app(Arc::new(UnsupportedBackend));
    for stream in [false, true] {
        assert_rejected(&app, stream, "does not support").await;
    }
}

#[tokio::test]
async fn extended_api_options_preflight_delegates_to_concrete_backend() {
    let backend = Arc::new(SelectiveBackend::default());
    let app = app(backend.clone());
    for stream in [false, true] {
        assert_rejected(&app, stream, "fixture does not support reasoning_effort").await;
    }
    assert_eq!(backend.validations.load(Ordering::SeqCst), 2);
    assert_eq!(backend.generations.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn extended_api_options_supported_fields_reach_json_and_sse_generation() {
    let backend = Arc::new(SelectiveBackend::default());
    let app = app(backend.clone());
    for stream in [false, true] {
        let mut payload = request(stream);
        payload.as_object_mut().unwrap().remove("reasoning_effort");
        payload["frequency_penalty"] = json!(0.2);
        let response = post_json(&app, "/v1/chat/completions", payload, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
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
            assert_eq!(value["choices"][0]["message"]["content"], "ok");
        }
    }
    assert_eq!(backend.validations.load(Ordering::SeqCst), 2);
    assert_eq!(backend.generations.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn reasoning_effort_native_hints_are_returned_before_json_or_sse() {
    let app = app(Arc::new(ReasoningLimitsBackend));
    for stream in [false, true] {
        let mut payload = request(stream);
        payload["reasoning_effort"] = json!("custom");
        let value = rejection(&app, payload).await;
        let error = &value["error"];
        assert_eq!(error["type"], "invalid_request_error");
        assert_eq!(error["param"], "reasoning_effort");
        assert_eq!(error["code"], "invalid_reasoning_effort");
        assert_eq!(
            error["supported_values"],
            json!(["none", "minimal", "low", "medium", "high", "xhigh", "max"])
        );
        assert_eq!(error["supported_types"], json!(["string", "null"]));
        assert_eq!(error["values_scope"], "backend");
        assert_eq!(error["values_depend_on_model"], true);
        assert!(error.get("details").is_none());
        let message = error["message"].as_str().unwrap();
        for effort in ["none", "minimal", "low", "medium", "high", "xhigh", "max"] {
            assert!(message.contains(effort), "missing {effort}: {message}");
        }
    }
}

#[tokio::test]
async fn reasoning_effort_invalid_types_return_schema_hints_before_json_or_sse() {
    let app = app(Arc::new(UnsupportedBackend));
    for stream in [false, true] {
        for invalid in [json!(true), json!([]), json!({})] {
            let mut payload = request(stream);
            payload["reasoning_effort"] = invalid;
            let value = rejection(&app, payload).await;
            let error = &value["error"];
            assert_eq!(error["param"], "reasoning_effort");
            assert_eq!(error["code"], "invalid_reasoning_effort");
            assert_eq!(
                error["supported_types"],
                json!(["string", "number", "null"])
            );
            assert_eq!(error["values_depend_on_model"], true);
            assert!(error.get("supported_values").is_none());
            assert!(error.get("values_scope").is_none());
            let message = error["message"].as_str().unwrap();
            for kind in ["string", "number", "null"] {
                assert!(message.contains(kind), "missing {kind}: {message}");
            }
        }
    }
}

#[tokio::test]
async fn unrelated_api_option_errors_do_not_include_reasoning_hints() {
    let app = app(Arc::new(UnsupportedBackend));
    for stream in [false, true] {
        let mut payload = request(stream);
        payload["frequency_penalty"] = json!("invalid");
        let value = rejection(&app, payload).await;
        assert_eq!(value["error"].as_object().unwrap().len(), 4);
        assert!(value["error"]["param"].is_null());
        assert!(value["error"]["code"].is_null());
        assert!(
            value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("frequency_penalty")
        );
    }
}
