use super::support::*;
use crate::backend::{PreloadHandle, runtime_cache::RuntimeCache};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

struct ResidentBackend {
    cache: RuntimeCache<String, String>,
    loads: Mutex<Vec<String>>,
    warmups: AtomicUsize,
}

impl GenerationBackend for ResidentBackend {
    fn preload(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> anyhow::Result<PreloadHandle> {
        assert_eq!(request.max_tokens, 32);
        assert!(!request.verbose && !request.debug);
        assert!(request.tool_config.is_none());
        self.warmups.fetch_add(1, Ordering::SeqCst);
        crate::backend::preload_resident(self, manifest, request)
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> anyhow::Result<GenerateResponse> {
        let _model = self.cache.get_or_try_init(
            manifest.id.clone(),
            |_| true,
            || {
                self.loads.lock().unwrap().push(manifest.id.clone());
                Ok(manifest.id.clone())
            },
        )?;
        anyhow::ensure!(
            manifest.id != "broken",
            "simulated warmup allocation failure"
        );
        MockBackend.generate(manifest, request)
    }

    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        let result = self
            .generate(&manifest, request)
            .map(|r| GenerateStreamEvent::Done {
                finish_reason: r.finish_reason,
                prompt_tokens: r.prompt_tokens,
                completion_tokens: r.completion_tokens,
                timings: r.timings,
                backend_diagnostics: r.backend_diagnostics,
            })
            .map_err(|e| e.to_string());
        Box::pin(tokio_stream::iter(vec![result]))
    }
}

fn install(store: &ModelStore, id: &str) {
    fs::create_dir_all(store.model_dir(id)).unwrap();
    store
        .write_manifest(&ModelManifest {
            storage: Default::default(),
            id: id.into(),
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
        })
        .unwrap();
}

async fn chat(app: Router, model: &str, stream: bool) -> Response {
    app.oneshot(Request::builder().method("POST").uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(json!({"model":model,"messages":[{"role":"user","content":"hello"}],"stream":stream}).to_string())).unwrap()).await.unwrap()
}

#[tokio::test]
async fn preload_large_list_survives_failure_and_reuses_workers_through_http() {
    let dir = tempfile::tempdir().unwrap();
    let store = ModelStore::resolve(Some(dir.path().to_path_buf())).unwrap();
    let backend = Arc::new(ResidentBackend {
        cache: RuntimeCache::bounded(2),
        loads: Mutex::new(vec![]),
        warmups: AtomicUsize::new(0),
    });
    let models: Vec<_> = (0..70).map(|i| format!("model-{i}")).collect();
    for model in models.iter().chain(std::iter::once(&"broken".to_string())) {
        install(&store, model);
    }
    let state = ApiState::new_with_default_model(store, backend.clone(), Some(models[0].clone()));
    let mut requested = models.clone();
    requested.insert(1, "broken".into());
    requested.insert(2, "missing".into());
    requested.push(models[0].clone());
    let report = state.preload_models(&requested).await;
    assert_eq!((report.ready, report.attempted), (70, 72));
    assert!(state.telemetry.snapshot().requests.is_empty());
    assert_eq!(state.telemetry.snapshot().totals.output_tokens, 0);
    assert_eq!(backend.warmups.load(Ordering::SeqCst), 71);
    let loaded = backend.loads.lock().unwrap().clone();
    assert_eq!(&loaded[..3], &["model-0", "broken", "model-1"]);
    assert_eq!(backend.cache.snapshot().len(), 70);
    assert!(
        backend
            .cache
            .snapshot()
            .iter()
            .all(|v| v.as_str() != "broken")
    );
    let app = router(state);
    // Exercise both HTTP paths and concurrent requests without a second load.
    let (a, b, c) = tokio::join!(
        chat(app.clone(), "model-0", false),
        chat(app.clone(), "model-69", true),
        chat(app.clone(), "model-1", false)
    );
    for response in [a, b, c] {
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(!bytes.is_empty());
    }
    assert_eq!(
        chat(app.clone(), "model-0", false).await.status(),
        StatusCode::OK
    );
    assert_eq!(*backend.loads.lock().unwrap(), loaded);
    backend.cache.remove_if(|_| true);
    assert_eq!(backend.cache.snapshot().len(), 70);
    drop(app);
    backend.cache.remove_if(|_| true);
    assert!(backend.cache.snapshot().is_empty());
}

#[tokio::test]
async fn unsupported_preload_keeps_server_usable() {
    let dir = tempfile::tempdir().unwrap();
    let store = ModelStore::resolve(Some(dir.path().to_path_buf())).unwrap();
    install(&store, "one-shot");
    let state = ApiState::new(store, Arc::new(MockBackend));
    let report = state
        .preload_models(&["one-shot".into(), "missing".into(), "one-shot".into()])
        .await;
    assert_eq!((report.ready, report.attempted), (0, 2));
    assert_eq!(
        chat(router(state), "one-shot", false).await.status(),
        StatusCode::OK
    );
}
