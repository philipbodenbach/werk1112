use super::support::*;
use crate::backend::ChatGenerationSession;
use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

#[derive(Default)]
struct Gate(Mutex<bool>, Condvar);
impl Gate {
    fn wait(&self) {
        let mut open = self.0.lock().unwrap();
        while !*open {
            open = self.1.wait(open).unwrap();
        }
    }
    fn release(&self) {
        *self.0.lock().unwrap() = true;
        self.1.notify_all();
    }
}
struct Release(Arc<Gate>);
impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}

#[derive(Clone)]
struct Backend {
    starts: Arc<AtomicUsize>,
    entered: tokio::sync::mpsc::UnboundedSender<String>,
    gate: Arc<Gate>,
    init_gate: Arc<Gate>,
    alive: Arc<AtomicBool>,
}
#[derive(Clone)]
struct Session {
    backend: Backend,
    model: ModelManifest,
}
impl ChatGenerationSession for Session {
    fn is_available(&self) -> bool {
        self.backend.alive.load(Ordering::SeqCst)
    }
    fn generate(&self, request: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        self.backend.entered.send(self.model.id.clone()).unwrap();
        self.backend.gate.wait();
        if self.model.id == "broken" {
            anyhow::bail!("isolated runtime failure");
        }
        MockBackend.generate(&self.model, request)
    }
    fn generate_stream(&self, request: GenerateRequest) -> GenerateStream {
        let session = self.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tokio::task::spawn_blocking(move || {
            let result = session
                .generate(request)
                .map(|result| GenerateStreamEvent::Done {
                    finish_reason: result.finish_reason,
                    prompt_tokens: result.prompt_tokens,
                    completion_tokens: result.completion_tokens,
                    timings: result.timings,
                    backend_diagnostics: result.backend_diagnostics,
                })
                .map_err(|error| error.to_string());
            let _ = tx.blocking_send(result);
        });
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}
impl GenerationBackend for Backend {
    fn start_chat_session(
        &self,
        model: &ModelManifest,
        _: Option<u64>,
    ) -> anyhow::Result<Option<Box<dyn ChatGenerationSession>>> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        if model.id == "a" {
            self.entered.send("initializing-a".into()).unwrap();
            self.init_gate.wait();
        }
        Ok(Some(Box::new(Session {
            backend: self.clone(),
            model: model.clone(),
        })))
    }
    fn generate(&self, _: &ModelManifest, _: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        unreachable!()
    }
    fn generate_stream(&self, _: ModelManifest, _: GenerateRequest) -> GenerateStream {
        unreachable!()
    }
}
fn fixture() -> (
    tempfile::TempDir,
    ApiState,
    Backend,
    tokio::sync::mpsc::UnboundedReceiver<String>,
) {
    let root = tempfile::tempdir().unwrap();
    let store = ModelStore::resolve(Some(root.path().to_path_buf())).unwrap();
    for id in ["a", "b", "c", "broken"] {
        fs::create_dir_all(store.model_dir(id)).unwrap();
        store
            .write_manifest(&ModelManifest {
                storage: Default::default(),
                id: id.into(),
                source: ModelSource::LocalPath {
                    path: "test".into(),
                },
                format: ModelFormat::SafeTensors,
                architecture: Some("llama".into()),
                tokenizer_path: None,
                config_path: None,
                model_path: None,
                backend: "mock".into(),
                created_unix: 1,
                files: vec![],
                artifacts: vec![],
                metadata: ModelMetadata::default(),
            })
            .unwrap();
    }
    let (entered, events) = tokio::sync::mpsc::unbounded_channel();
    let backend = Backend {
        starts: Arc::new(AtomicUsize::new(0)),
        entered,
        gate: Arc::new(Gate::default()),
        init_gate: Arc::new(Gate::default()),
        alive: Arc::new(AtomicBool::new(true)),
    };
    let state = ApiState::new(store, Arc::new(backend.clone()));
    (root, state, backend, events)
}
fn launch(
    app: Router,
    model: &str,
    anthropic: bool,
    stream: bool,
) -> tokio::task::JoinHandle<(StatusCode, Vec<u8>)> {
    let request = Request::builder()
        .method("POST")
        .uri(if anthropic {
            "/v1/messages"
        } else {
            "/v1/chat/completions"
        })
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .body(Body::from(
            json!({"model":model,"max_tokens":16,"stream":stream,
            "messages":[{"role":"user","content":"hello"}]})
            .to_string(),
        ))
        .unwrap();
    tokio::spawn(async move {
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let body = body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        (status, body.to_vec())
    })
}
async fn event(events: &mut tokio::sync::mpsc::UnboundedReceiver<String>) -> String {
    tokio::time::timeout(Duration::from_secs(10), events.recv())
        .await
        .expect("request must progress without releasing other requests")
        .unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn both_apis_overlap_across_three_models_and_reuse_sessions() {
    let (_root, state, backend, mut events) = fixture();
    let _release = Release(backend.gate.clone());
    backend.init_gate.release();
    let app = router(state);
    let requests = [
        launch(app.clone(), "a", false, false),
        launch(app.clone(), "b", true, true),
        launch(app.clone(), "c", false, true),
    ];
    let mut entered = Vec::new();
    while entered.len() < 3 {
        let model = event(&mut events).await;
        if !model.starts_with("initializing") {
            entered.push(model);
        }
    }
    entered.sort();
    assert_eq!(entered, ["a", "b", "c"]);
    assert_eq!(backend.starts.load(Ordering::SeqCst), 3);
    backend.gate.release();
    for request in requests {
        assert_eq!(request.await.unwrap().0, StatusCode::OK);
    }
    assert_eq!(
        launch(app.clone(), "a", true, false).await.unwrap().0,
        StatusCode::OK
    );
    assert_eq!(backend.starts.load(Ordering::SeqCst), 3);
    assert!(
        !launch(app.clone(), "broken", false, false)
            .await
            .unwrap()
            .0
            .is_success()
    );
    assert_eq!(
        launch(app, "b", false, false).await.unwrap().0,
        StatusCode::OK
    );
}

#[tokio::test(flavor = "current_thread")]
async fn six_same_model_requests_share_one_session_and_overlap() {
    let (_root, state, backend, mut events) = fixture();
    backend.init_gate.release();
    let _release = Release(backend.gate.clone());
    let app = router(state);
    let requests: Vec<_> = (0..6)
        .map(|i| launch(app.clone(), "a", i % 2 == 0, i % 3 == 0))
        .collect();
    let mut entered = 0;
    while entered < 6 {
        if event(&mut events).await == "a" {
            entered += 1;
        }
    }
    assert_eq!(backend.starts.load(Ordering::SeqCst), 1);
    backend.gate.release();
    for request in requests {
        assert_eq!(request.await.unwrap().0, StatusCode::OK);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn cold_session_creation_does_not_block_executor_or_other_models() {
    let (_root, state, backend, mut events) = fixture();
    let _release = Release(backend.init_gate.clone());
    backend.gate.release();
    let app = router(state);
    let a = launch(app.clone(), "a", false, true);
    assert_eq!(event(&mut events).await, "initializing-a");
    let b = launch(app, "b", true, false);
    assert_eq!(event(&mut events).await, "b");
    assert_eq!(b.await.unwrap().0, StatusCode::OK);
    backend.init_gate.release();
    assert_eq!(a.await.unwrap().0, StatusCode::OK);
}

#[tokio::test]
async fn unavailable_session_is_recreated_on_the_next_request() {
    let (_root, state, backend, _events) = fixture();
    backend.init_gate.release();
    backend.gate.release();
    let app = router(state);
    assert_eq!(
        launch(app.clone(), "a", false, false).await.unwrap().0,
        StatusCode::OK
    );
    backend.alive.store(false, Ordering::SeqCst);
    assert_eq!(
        launch(app, "a", false, false).await.unwrap().0,
        StatusCode::OK
    );
    assert_eq!(backend.starts.load(Ordering::SeqCst), 2);
}
