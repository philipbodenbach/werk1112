use super::support::*;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

struct Counter {
    native: Option<usize>,
    counts: AtomicUsize,
    generations: AtomicUsize,
    counted: Mutex<Option<Value>>,
}
impl Counter {
    fn new(native: Option<usize>) -> Arc<Self> {
        Arc::new(Self {
            native,
            counts: AtomicUsize::new(0),
            generations: AtomicUsize::new(0),
            counted: Mutex::new(None),
        })
    }
}
fn signature(request: &GenerateRequest, options: &BTreeMap<String, Value>) -> Value {
    json!({"messages":request.messages,"tools":request.tool_config,"options":options,"max_tokens":request.max_tokens})
}
impl GenerationBackend for Counter {
    fn supports_tool_calling(&self, _: &ModelManifest, _: bool) -> bool {
        true
    }
    fn validate_api_options(
        &self,
        _: &ModelManifest,
        _: &GenerateRequest,
        _: &BTreeMap<String, Value>,
    ) -> anyhow::Result<()> {
        Ok(())
    }
    fn count_api_tokens(
        &self,
        _: &ModelManifest,
        request: GenerateRequest,
        options: BTreeMap<String, Value>,
    ) -> anyhow::Result<usize> {
        self.counts.fetch_add(1, Ordering::SeqCst);
        *self.counted.lock().unwrap() = Some(signature(&request, &options));
        self.native
            .ok_or_else(|| anyhow::anyhow!("tokenizer unavailable"))
    }
    fn generate_api(
        &self,
        _: &ModelManifest,
        request: GenerateRequest,
        options: BTreeMap<String, Value>,
        tx: Option<tokio::sync::mpsc::Sender<Result<Value, String>>>,
    ) -> anyhow::Result<Value> {
        self.generations.fetch_add(1, Ordering::SeqCst);
        if let Some(counted) = self.counted.lock().unwrap().as_ref() {
            assert_eq!(
                *counted,
                signature(&request, &options),
                "count and generation must preserve messages, tool pairs and API options"
            );
        }
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
        panic!("extended request")
    }
    fn generate_stream(&self, _: ModelManifest, _: GenerateRequest) -> GenerateStream {
        panic!("extended request")
    }
}
fn app(backend: Arc<Counter>) -> Router {
    let store = test_store();
    let manifest = ModelManifest {
        storage: Default::default(),
        id: "context-test".into(),
        source: ModelSource::LocalPath {
            path: "fixture".into(),
        },
        format: ModelFormat::Gguf,
        architecture: None,
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "fixture".into(),
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
    router(ApiState::new(store, backend).with_chat_context_size(Some(4096)))
}
fn request(stream: bool, large: bool) -> Value {
    json!({"model":"context-test","stream":stream,"max_tokens":32,"reasoning_effort":"low",
    "tools":[{"type":"function","function":{"name":"inspect","parameters":{"type":"object","properties":{}}}}],
    "messages":[
        {"role":"user","content":"Inspect the file"},
        {"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"inspect","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"call_1","content": if large { "\\\"".repeat(12000) } else { "ok".into() }}
    ]})
}
#[tokio::test]
async fn native_count_admits_overestimated_tool_history_without_trimming() {
    // Exact boundary is valid, despite JSON escaping making the estimate huge.
    let backend = Counter::new(Some(4096 - 32));
    let app = app(backend.clone());
    for stream in [false, true] {
        let response = post_json(&app, "/v1/chat/completions", request(stream, true), None).await;
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
        }
    }
    assert_eq!(backend.counts.load(Ordering::SeqCst), 2);
    assert_eq!(backend.generations.load(Ordering::SeqCst), 2);
}
#[tokio::test]
async fn native_overflow_returns_recognizable_error_before_json_or_sse() {
    let backend = Counter::new(Some(4096 - 32 + 1));
    let app = app(backend.clone());
    for stream in [false, true] {
        let response = post_json(&app, "/v1/chat/completions", request(stream, true), None).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        let error = &value["error"];
        assert_eq!(error["code"], "context_length_exceeded");
        assert_eq!(error["param"], "messages");
        assert_eq!(error["prompt_tokens"], 4065);
        assert_eq!(error["max_tokens"], 32);
        assert_eq!(error["context_length"], 4096);
        assert_eq!(error["token_count_method"], "native");
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .contains("maximum context length is 4096 tokens")
        );
    }
    assert_eq!(backend.generations.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn unavailable_native_counter_keeps_labeled_estimate_and_compaction_signal() {
    let backend = Counter::new(None);
    let app = app(backend.clone());
    let response = post_json(&app, "/v1/chat/completions", request(true, true), None).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["error"]["code"], "context_length_exceeded");
    assert_eq!(value["error"]["token_count_method"], "estimate");
    assert_eq!(backend.generations.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn requests_well_within_budget_do_not_add_tokenizer_round_trips() {
    let backend = Counter::new(None);
    let app = app(backend.clone());
    let response = post_json(&app, "/v1/chat/completions", request(false, false), None).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(backend.counts.load(Ordering::SeqCst), 0);
    assert_eq!(backend.generations.load(Ordering::SeqCst), 1);
}
