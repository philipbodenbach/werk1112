//! Alias routing uses independent existing ApiState/session/control instances.
use super::state::ApiState;
use crate::deployments::Plan;
use anyhow::{Result, ensure};
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub(super) struct Instance {
    pub plan: Arc<Plan>,
    pub state: ApiState,
    slots: Arc<Semaphore>,
}
pub(super) struct Registry {
    instances: BTreeMap<String, Instance>,
    aliases: BTreeMap<String, Vec<String>>,
    next: AtomicUsize,
}
impl Registry {
    pub fn new(entries: Vec<(Arc<Plan>, ApiState)>) -> Result<Self> {
        let mut instances = BTreeMap::new();
        let mut aliases: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (plan, state) in entries {
            ensure!(
                !instances.contains_key(&plan.profile.id),
                "duplicate deployment ID"
            );
            aliases
                .entry(plan.profile.alias.clone())
                .or_default()
                .push(plan.profile.id.clone());
            instances.insert(
                plan.profile.id.clone(),
                Instance {
                    slots: Arc::new(Semaphore::new(plan.profile.parallel)),
                    plan,
                    state,
                },
            );
        }
        for (alias, ids) in &mut aliases {
            ensure!(
                !instances.contains_key(alias) || ids.len() == 1 && ids[0] == *alias,
                "alias collides with a deployment ID: {alias}"
            );
            ids.sort();
        }
        Ok(Self {
            instances,
            aliases,
            next: AtomicUsize::new(0),
        })
    }
    pub fn contains(&self, alias: &str) -> bool {
        self.instances.contains_key(alias) || self.aliases.contains_key(alias)
    }
    pub fn models(&self) -> Vec<(String, String)> {
        self.aliases
            .iter()
            .map(|(alias, ids)| {
                (
                    alias.clone(),
                    self.instances[&ids[0]].plan.profile.model.clone(),
                )
            })
            .collect()
    }
    pub fn control(&self, id: &str) -> Result<ApiState> {
        Ok(self
            .instances
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("unknown deployment ID {id}"))?
            .state
            .clone())
    }
    pub fn select(&self, alias: &str, headers: &HeaderMap) -> Result<(ApiState, String)> {
        let ids = self.aliases.get(alias).cloned().or_else(|| self.instances.contains_key(alias).then(|| vec![alias.into()]))
            .ok_or_else(|| anyhow::anyhow!("unknown deployment alias {alias}; unmanaged model starts are disabled in profile mode"))?;
        let session = headers
            .get("x-werk-session-id")
            .map(|s| s.to_str())
            .transpose()?;
        if let Some(session) = session {
            ensure!(
                !session.is_empty() && session.len() <= 256,
                "x-werk-session-id must contain 1..256 bytes"
            );
        }
        let selected = if let Some(session) = session {
            let mut hash = Sha256::new();
            // Scope client affinity to the bearer principal; never retain tokens.
            hash.update(
                headers
                    .get("authorization")
                    .or_else(|| headers.get("x-api-key"))
                    .map_or(&[][..], |v| v.as_bytes()),
            );
            hash.update([0]);
            hash.update(alias.as_bytes());
            hash.update([0]);
            hash.update(session.as_bytes());
            let bytes: [u8; 32] = hash.finalize().into();
            u64::from_be_bytes(bytes[..8].try_into().unwrap()) as usize % ids.len()
        } else {
            let start = self.next.fetch_add(1, Ordering::Relaxed) % ids.len();
            (0..ids.len())
                .map(|i| (start + i) % ids.len())
                .max_by_key(|i| self.instances[&ids[*i]].slots.available_permits())
                .unwrap()
        };
        let instance = &self.instances[&ids[selected]];
        // Deliberate fail-fast queue policy: no unbounded HTTP waiters. An
        // affinity-bound request never spills onto another replica when busy.
        let permit = instance.slots.clone().try_acquire_owned().map_err(|_| {
            anyhow::anyhow!("deployment busy; queue capacity is zero; retry this complete request")
        })?;
        let mut state = instance.state.clone();
        state.deployment_permit = Some(Arc::new(permit));
        state.requested_alias = Some(alias.into());
        Ok((state, instance.plan.profile.model.clone()))
    }
    pub fn telemetry(&self) -> Vec<crate::observability::BackendSnapshot> {
        self.instances
            .values()
            .flat_map(|i| {
                let mut workers = i.state.backend.telemetry();
                workers.push(crate::observability::BackendSnapshot {
                    backend: "deployment".into(),
                    instance: i.plan.profile.id.clone(),
                    model: i.plan.profile.alias.clone(),
                    available: true,
                    counters: Default::default(),
                    gauges: BTreeMap::from([
                        ("gpu_count".into(), i.plan.devices.len() as f64),
                        ("slots_available".into(), i.slots.available_permits() as f64),
                        ("queue_capacity".into(), 0.0),
                        (
                            "gpu_budget_bytes_estimate".into(),
                            i.plan
                                .profile
                                .memory
                                .iter()
                                .filter_map(|b| b.total().ok())
                                .sum::<u64>() as f64,
                        ),
                        (
                            "host_budget_bytes_estimate".into(),
                            i.plan.profile.host_bytes as f64,
                        ),
                    ]),
                });
                workers
            })
            .collect()
    }
    pub fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({ "profiles": self.instances.values().map(|i| serde_json::json!({"plan": i.plan.as_ref(), "available_slots": i.slots.available_permits(), "queue_capacity": 0, "queue_policy": "reject", "workers": i.state.backend.telemetry()})).collect::<Vec<_>>() })
    }
}

pub(super) async fn diagnostics(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let registry = state.deployments.clone();
    match tokio::task::spawn_blocking(move || -> Result<serde_json::Value> {
        Ok(serde_json::json!({"inventory": crate::inference_service::devices::Inventory::detect()?, "deployments": registry.map(|r| r.diagnostics())}))
    }).await {
        Ok(Ok(value)) => Json(value).into_response(),
        error => super::response::api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("deployment diagnostics failed: {error:?}"), None),
    }
}

pub(super) fn control_state(
    state: ApiState,
    headers: &HeaderMap,
) -> std::result::Result<ApiState, Response> {
    let Some(registry) = &state.deployments else {
        return Ok(state);
    };
    state.authorize(headers)?;
    let selected = headers.get("x-werk-deployment").and_then(|v| v.to_str().ok())
        .ok_or_else(|| anyhow::anyhow!("profile mode requires x-werk-deployment with an exact instance ID for runtime control"))
        .and_then(|id| registry.control(id));
    selected.map_err(|e| {
        super::response::api_error(
            StatusCode::BAD_REQUEST,
            e.to_string(),
            Some("x-werk-deployment".into()),
        )
    })
}

/// Keeps admission until the stream finishes or its consumer disconnects.
pub(super) struct AdmittedStream<S> {
    pub stream: S,
    pub _permit: Option<Arc<OwnedSemaphorePermit>>,
}
impl<S: tokio_stream::Stream + Unpin> tokio_stream::Stream for AdmittedStream<S> {
    type Item = S::Item;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.stream).poll_next(cx)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::{
        backend::{GenerateRequest, GenerateResponse, GenerateStream, GenerationBackend},
        model_store::{ModelManifest, ModelStore},
    };
    struct NeverGenerate;
    impl GenerationBackend for NeverGenerate {
        fn generate(&self, _: &ModelManifest, _: GenerateRequest) -> Result<GenerateResponse> {
            anyhow::bail!("test backend")
        }
        fn generate_stream(&self, _: ModelManifest, _: GenerateRequest) -> GenerateStream {
            Box::pin(tokio_stream::empty())
        }
    }
    fn registry() -> (tempfile::TempDir, Registry) {
        let (dir, plan) = crate::deployments::tests::plan();
        let store = ModelStore::resolve(Some(dir.path().join("store"))).unwrap();
        let mut other = plan.clone();
        other.profile.id = "worker-b".into();
        let registry = Registry::new(vec![
            (
                Arc::new(plan),
                ApiState::new(store.clone(), Arc::new(NeverGenerate)),
            ),
            (
                Arc::new(other),
                ApiState::new(store, Arc::new(NeverGenerate)),
            ),
        ])
        .unwrap();
        (dir, registry)
    }
    #[test]
    fn sessions_stay_on_the_same_replica_and_do_not_spill_when_busy() {
        let (_dir, registry) = registry();
        let mut headers = HeaderMap::new();
        headers.insert("x-werk-session-id", "conversation-7".parse().unwrap());
        let (state, _) = registry.select("chat", &headers).unwrap();
        assert!(
            registry
                .select("chat", &headers)
                .err()
                .unwrap()
                .to_string()
                .contains("deployment busy")
        );
        let selected = state.backend.clone();
        drop(state);
        let (again, _) = registry.select("chat", &headers).unwrap();
        assert!(Arc::ptr_eq(&selected, &again.backend));
        assert!(registry.control("chat").is_err());
        assert!(registry.control("worker-a").is_ok());
    }
    #[test]
    fn stateless_requests_use_free_replicas_and_are_bounded() {
        let (_dir, registry) = registry();
        let headers = HeaderMap::new();
        let (a, _) = registry.select("chat", &headers).unwrap();
        let (b, _) = registry.select("chat", &headers).unwrap();
        assert!(!Arc::ptr_eq(&a.backend, &b.backend));
        assert!(registry.select("chat", &headers).is_err());
        drop(a);
        assert!(registry.select("chat", &headers).is_ok());
        assert!(registry.select("model", &headers).is_err());
    }
    #[test]
    fn admission_lives_until_stream_drop() {
        let (_dir, registry) = registry();
        let (state, _) = registry.select("worker-a", &HeaderMap::new()).unwrap();
        let stream = AdmittedStream {
            stream: tokio_stream::empty::<()>(),
            _permit: state.deployment_permit.clone(),
        };
        drop(state);
        assert!(registry.select("worker-a", &HeaderMap::new()).is_err());
        drop(stream);
        assert!(registry.select("worker-a", &HeaderMap::new()).is_ok());
    }

    #[tokio::test]
    async fn http_aliases_and_control_target_are_enforced() {
        use axum::{
            body::{Body, to_bytes},
            http::Request,
        };
        use tower::ServiceExt;
        let (dir, registry) = registry();
        let store = ModelStore::resolve(Some(dir.path().join("store"))).unwrap();
        store.ensure().unwrap();
        let manifest = crate::deployments::tests::manifest();
        std::fs::create_dir_all(store.model_dir(&manifest.id)).unwrap();
        std::fs::write(
            store.model_dir(&manifest.id).join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let mut state = ApiState::new(store, Arc::new(NeverGenerate));
        state.deployments = Some(Arc::new(registry));
        let app = crate::api::router(state);
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
        let models: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(models["data"][0]["id"], "chat");
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/werk/v1/info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/werk/v1/info")
                    .header("x-werk-deployment", "worker-a")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"model","messages":[{"role":"user","content":"hello"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let error: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap();
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("unmanaged model starts")
        );
    }

    #[tokio::test]
    async fn protocol_client_reaches_selected_instance_over_real_http() {
        let (dir, registry) = registry();
        let store = ModelStore::resolve(Some(dir.path().join("store"))).unwrap();
        let mut state = ApiState::new(store, Arc::new(NeverGenerate));
        state.deployments = Some(Arc::new(registry));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(crate::api::serve_with_listener(listener, state));
        let result = tokio::task::spawn_blocking(move || {
            crate::werk_protocol::WerkProtocolClient::new(&format!("http://{address}"), None)
                .unwrap()
                .with_deployment(Some("worker-a".into()))
                .unwrap()
                .info()
        })
        .await
        .unwrap();
        server.abort();
        assert!(result.is_ok(), "{result:?}");
    }
}
