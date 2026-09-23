use super::support::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ObservedBackend(Arc<AtomicUsize>);
impl GenerationBackend for ObservedBackend {
    fn telemetry(&self) -> Vec<crate::observability::BackendSnapshot> {
        self.0.fetch_add(1, Ordering::Relaxed);
        vec![crate::observability::BackendSnapshot {
            backend: "mock".into(),
            instance: "worker-1".into(),
            model: "mock".into(),
            available: true,
            ..Default::default()
        }]
    }
    fn generate(&self, m: &ModelManifest, r: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        MockBackend.generate(m, r)
    }
    fn generate_stream(&self, m: ModelManifest, r: GenerateRequest) -> GenerateStream {
        MockBackend.generate_stream(m, r)
    }
}
fn app() -> (Router, Arc<AtomicUsize>) {
    let store = test_store();
    let calls = Arc::new(AtomicUsize::new(0));
    let manifest = ModelManifest {
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
    };
    fs::create_dir_all(store.model_dir("mock")).unwrap();
    fs::write(
        store
            .model_dir("mock")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    (
        router(
            ApiState::new(store, Arc::new(ObservedBackend(calls.clone())))
                .with_api_keys(vec!["test-key".into()]),
        ),
        calls,
    )
}
fn get(path: &str, auth: bool) -> Request<Body> {
    let mut r = Request::builder()
        .uri(path)
        .header("x-werk-protocol-version", "1.0");
    if auth {
        r = r.header(header::AUTHORIZATION, "Bearer test-key");
    }
    r.body(Body::empty()).unwrap()
}
#[tokio::test]
async fn observability_auth_and_shared_sampling_cache() {
    let (app, calls) = app();
    for path in ["/metrics", "/werk/v1/observability"] {
        assert_eq!(
            app.clone()
                .oneshot(get(path, false))
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let response = app
        .clone()
        .oneshot(get("/werk/v1/observability", true))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let data = response_json(response).await;
    assert_eq!(data["data"]["schema_version"], 1);
    assert_eq!(data["data"]["backends"][0]["backend"], "mock");
    let response = app.oneshot(get("/metrics", true)).await.unwrap();
    assert!(
        response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .contains("version=0.0.4")
    );
    let body = body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("werk_requests_active 0"));
    assert!(!text.contains("test-key"));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}
#[tokio::test]
async fn observability_counts_both_protocols_and_stream_modes_without_content() {
    let (app, _) = app();
    for streaming in [false, true] {
        for path in ["/v1/chat/completions", "/v1/messages"] {
            let response=app.clone().oneshot(Request::builder().method("POST").uri(path).header(header::AUTHORIZATION,"Bearer test-key").header(header::CONTENT_TYPE,"application/json").header("anthropic-version","2023-06-01").body(Body::from(json!({"model":"mock","messages":[{"role":"user","content":"private-prompt-never-in-metrics"}],"max_tokens":64,"stream":streaming}).to_string())).unwrap()).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let _ = body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
        }
    }
    let result = response_json(
        app.oneshot(get("/werk/v1/observability", true))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(result["data"]["totals"]["completed"], 4);
    assert_eq!(result["data"]["totals"]["output_tokens"], 4);
    assert_eq!(result["data"]["totals"]["active"], 0);
    assert!(!result.to_string().contains("private-prompt"));
}
