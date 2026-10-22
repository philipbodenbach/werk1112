use super::support::*;
use crate::{
    observability::ReasoningDefaults,
    reasoning::{BackendReasoning, profile},
};
use axum::http::HeaderMap;
use std::sync::Mutex;

fn manifest() -> ModelManifest {
    ModelManifest {
        storage: Default::default(),
        id: "reasoning-fixture".into(),
        source: ModelSource::LocalPath {
            path: "fixture".into(),
        },
        format: ModelFormat::Gguf,
        architecture: Some("qwen35".into()),
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "fixture".into(),
        created_unix: 1,
        files: vec![],
        artifacts: vec![],
        metadata: Default::default(),
    }
}
fn config(thinking: bool, budget: i64) -> BackendReasoning {
    BackendReasoning {
        backend: "llama.cpp".into(),
        budget_control: true,
        native_efforts: vec![],
        defaults: ReasoningDefaults {
            source: "worker_launch_configuration".into(),
            thinking: Some(thinking),
            budget_tokens: Some(budget),
            effort: None,
        },
    }
}

#[test]
fn generation_limits_distinguish_defaults_from_ceilings_and_bound_reasoning() {
    let mut m = manifest();
    m.metadata
        .generation_defaults
        .insert("max_new_tokens".into(), json!(256));
    assert_eq!(
        crate::generation_limits::profile(&m, Some(16384)).max_output_tokens,
        Some(16383)
    );
    assert_eq!(
        crate::generation_limits::profile(&m, None).max_output_tokens,
        None
    );
    m.metadata
        .parameter_constraints
        .insert("max_completion_tokens".into(), json!({"max":8192}));
    m.metadata
        .parameter_constraints
        .insert("max_output_tokens".into(), json!(6144));
    let limits = crate::generation_limits::profile(&m, Some(16384));
    assert_eq!(limits.max_output_tokens, Some(6144));
    assert_eq!(limits.context_tokens, Some(16384));
    let reasoning = profile(&m, Some(&config(true, 8192)), Some(16384), 32768);
    assert_eq!(reasoning.limits.output_tokens, 6144);
    assert!(
        reasoning
            .tiers
            .iter()
            .filter(|tier| tier.available)
            .all(|tier| tier.minimum_output_tokens <= 6144)
    );
}

#[tokio::test]
async fn generation_limits_alias_uses_smallest_replica_without_advancing_routing() {
    use crate::api::deployments::Registry;
    let backend = Arc::new(Recorder::default());
    let store = test_store();
    let mut m = manifest();
    // Explicit deployment context must also work for non-GGUF backends.
    m.format = ModelFormat::Unknown;
    fs::create_dir_all(store.model_dir(&m.id)).unwrap();
    fs::write(
        store
            .model_dir(&m.id)
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&m).unwrap(),
    )
    .unwrap();
    let mut entries = Vec::new();
    for (id, context) in [("a", 16384), ("b", 8192)] {
        let plan = serde_json::from_value(json!({
            "profile":{"id":id,"alias":"shared","model":"reasoning-fixture","runtime":"vllm_cuda",
                "executable":"unused","build":"fixture","gpus":[],"strategy":"single","context":context,
                "batch":1,"memory":[],"host_bytes":0},
            "model_identity":"fixture","executable_sha256":"unused","runtime_version":"fixture","devices":[],
            "strategy":"single","args":[],"environment":{},"fingerprint":"fixture","reasons":[]
        })).unwrap();
        entries.push((
            Arc::new(plan),
            ApiState::new(store.clone(), backend.clone()).with_chat_context_size(Some(context)),
        ));
    }
    let baseline = Registry::new(entries.clone()).unwrap();
    let expected_context = baseline
        .select("shared", &HeaderMap::new())
        .unwrap()
        .0
        .chat_context_size;
    let registry = Arc::new(Registry::new(entries).unwrap());
    let mut state = ApiState::new(store, backend.clone());
    state.deployments = Some(registry.clone());
    let app = router(state);
    for _ in 0..3 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/werk/v1/chat/limits?model=shared")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value: Value = serde_json::from_slice(
            &body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            value["data"],
            json!({"schema_version":1,"model":"shared","context_tokens":8192,"max_output_tokens":8191})
        );
    }
    let (selected, _) = registry.select("shared", &HeaderMap::new()).unwrap();
    assert_eq!(
        selected.chat_context_size, expected_context,
        "discovery must not advance round-robin selection"
    );
    assert!(backend.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn generation_limits_discovery_and_admission_share_the_model_ceiling() {
    let backend = Arc::new(Recorder::default());
    let store = test_store();
    let mut m = manifest();
    m.metadata
        .parameter_constraints
        .insert("max_output_tokens".into(), json!(8192));
    fs::create_dir_all(store.model_dir(&m.id)).unwrap();
    fs::write(
        store
            .model_dir(&m.id)
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&m).unwrap(),
    )
    .unwrap();
    let mut state = ApiState::new(store, backend.clone());
    state.chat_context_size = 16384;
    let app = router(state);
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/werk/v1/chat/limits?model=reasoning-fixture")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value: Value = serde_json::from_slice(
        &body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        value["data"],
        json!({"schema_version":1,"model":"reasoning-fixture","context_tokens":16384,"max_output_tokens":8192})
    );
    assert!(
        backend.calls.lock().unwrap().is_empty(),
        "discovery must not infer or load a model"
    );
    let response = post_json(&app, "/v1/chat/completions", json!({"model":"reasoning-fixture","max_completion_tokens":8193,"messages":[{"role":"user","content":"hi"}]}), None).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value: Value = serde_json::from_slice(
        &body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(value["error"]["code"], "output_limit_exceeded");
    assert!(backend.calls.lock().unwrap().is_empty());
    let response = post_json(&app, "/v1/chat/completions", json!({"model":"reasoning-fixture","max_completion_tokens":8192,"messages":[{"role":"user","content":"hi"}],"werk":{"reasoning":{"tier":"default"}}}), None).await;
    assert_eq!(response.status(), StatusCode::OK);
}
#[test]
fn reasoning_defaults_and_every_available_tier_obey_limits() {
    for thinking in [false, true] {
        for budget in [-1, 0, 255, 512, 2048, 4096, 10000] {
            for output in [1, 511, 512, 513, 1024, 1536, 2048, 2560, 4096, 16000] {
                let p = profile(
                    &manifest(),
                    Some(&config(thinking, budget)),
                    Some(32768),
                    output,
                );
                let default = p.default_tier.as_ref().unwrap();
                assert!(p.tiers.iter().any(|t| &t.id == default && t.available));
                for tier in p.tiers.iter().filter(|t| t.available) {
                    assert!(tier.minimum_output_tokens <= p.limits.output_tokens);
                    let tokens = tier.reasoning_tokens.unwrap();
                    assert!(tokens + p.limits.answer_reserve_tokens <= p.limits.output_tokens);
                    if budget >= 0 {
                        assert!(tokens <= budget as u64);
                    }
                }
                let options = p.resolve("default").unwrap();
                if !thinking || budget == 0 {
                    assert_eq!(default, "none");
                    assert_eq!(options["__werk_thinking"], false);
                }
            }
        }
    }
    let p = profile(&manifest(), Some(&config(true, 2048)), Some(32768), 2048);
    assert_eq!(p.default_tier.as_deref(), Some("medium"));
    assert!(p.default_adjusted_to_limits);
    assert!(p.resolve("high").is_err());
    assert_eq!(
        p.resolve("default").unwrap()["reasoning_budget_tokens"],
        1024
    );
    let p = profile(&manifest(), Some(&config(true, 2048)), Some(512), 2048);
    assert!(p.default_tier.is_none());
    assert!(p.resolve("default").is_err());
}
#[test]
fn reasoning_native_vocabulary_is_model_specific_and_unknown_stays_unknown() {
    let mut m = manifest();
    let mut runtime = config(true, 2048);
    runtime.budget_control = false;
    runtime.native_efforts = vec![json!("none"), json!("low"), json!("high"), json!("max")];
    runtime.defaults.effort = Some(json!("high"));
    m.metadata.parameter_constraints.insert(
        "reasoning_effort".into(),
        json!({"enum":["none","low","high"]}),
    );
    let p = profile(&m, Some(&runtime), None, 2048);
    assert_eq!(
        p.tiers.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(),
        vec!["none", "low", "high"]
    );
    assert_eq!(p.resolve("default").unwrap()["reasoning_effort"], "high");
    assert_eq!(p.limits.reasoning_tokens, None);
    runtime.defaults.thinking = Some(false);
    assert_eq!(
        profile(&m, Some(&runtime), None, 2048)
            .default_tier
            .as_deref(),
        Some("none")
    );
    let unknown = profile(&m, None, None, 2048);
    assert!(!unknown.supported);
    assert!(unknown.tiers.is_empty());
}

#[test]
fn reasoning_named_launch_default_is_preserved_or_reduced_to_fit() {
    let mut runtime = config(true, 8192);
    runtime.defaults.effort = Some(json!("low"));
    assert_eq!(
        profile(&manifest(), Some(&runtime), None, 12000)
            .default_tier
            .as_deref(),
        Some("low")
    );
    runtime.defaults.effort = Some(json!("high"));
    let p = profile(&manifest(), Some(&runtime), None, 2048);
    assert_eq!(p.default_tier.as_deref(), Some("medium"));
    assert!(p.default_adjusted_to_limits);
    runtime.defaults.effort = Some(json!("none"));
    assert_eq!(
        profile(&manifest(), Some(&runtime), None, 12000)
            .default_tier
            .as_deref(),
        Some("none")
    );
}
#[derive(Default)]
struct Recorder {
    calls: Mutex<Vec<BTreeMap<String, Value>>>,
}
impl GenerationBackend for Recorder {
    fn reasoning_config(&self, _: &ModelManifest) -> Option<BackendReasoning> {
        Some(config(true, 2048))
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
        _: GenerateRequest,
        options: BTreeMap<String, Value>,
    ) -> anyhow::Result<usize> {
        self.calls.lock().unwrap().push(options);
        Ok(8)
    }
    fn generate_api(
        &self,
        _: &ModelManifest,
        _: GenerateRequest,
        options: BTreeMap<String, Value>,
        _: Option<tokio::sync::mpsc::Sender<Result<Value, String>>>,
    ) -> anyhow::Result<Value> {
        self.calls.lock().unwrap().push(options);
        Ok(
            json!({"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}),
        )
    }
    fn generate(&self, _: &ModelManifest, _: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        panic!("use managed API options")
    }
    fn generate_stream(&self, _: ModelManifest, _: GenerateRequest) -> GenerateStream {
        panic!("no native model")
    }
}
fn app(backend: Arc<Recorder>) -> Router {
    let store = test_store();
    let m = manifest();
    fs::create_dir_all(store.model_dir(&m.id)).unwrap();
    fs::write(
        store
            .model_dir(&m.id)
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&m).unwrap(),
    )
    .unwrap();
    router(ApiState::new(store, backend))
}
#[tokio::test]
async fn reasoning_protocol_profile_and_both_requests_share_resolution() {
    let backend = Arc::new(Recorder::default());
    let app = app(backend.clone());
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/werk/v1/chat/reasoning?model=reasoning-fixture&max_output_tokens=2048")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-werk-protocol-version"], "1.0");
    let value: Value = serde_json::from_slice(
        &body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(value["data"]["default_tier"], "medium");
    assert_eq!(value["data"]["limits"]["reasoning_tokens"], 1536);
    assert!(backend.calls.lock().unwrap().is_empty());
    for endpoint in ["/werk/v1/chat/count_tokens", "/v1/chat/completions"] {
        let response=post_json(&app,endpoint,json!({"model":"reasoning-fixture","max_tokens":2048,"messages":[{"role":"user","content":"hi"}],"werk":{"reasoning":{"tier":"default"}}}),None).await;
        assert_eq!(response.status(), StatusCode::OK);
    }
    let calls = backend.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0], calls[1]);
    assert_eq!(calls[0]["reasoning_budget_tokens"], 1024);
}
#[tokio::test]
async fn reasoning_invalid_tiers_and_conflicts_fail_before_backend_calls() {
    let backend = Arc::new(Recorder::default());
    let app = app(backend.clone());
    for (tier, native) in [
        ("high", None),
        ("ultra", None),
        ("made-up", None),
        ("low", Some("low")),
    ] {
        let mut payload = json!({"model":"reasoning-fixture","max_tokens":2048,"messages":[{"role":"user","content":"hi"}],"werk":{"reasoning":{"tier":tier}}});
        if let Some(effort) = native {
            payload["reasoning_effort"] = json!(effort);
        }
        let response = post_json(&app, "/v1/chat/completions", payload, None).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_slice(
            &body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(value["error"]["code"], "invalid_reasoning_tier");
    }
    assert!(backend.calls.lock().unwrap().is_empty());
}
