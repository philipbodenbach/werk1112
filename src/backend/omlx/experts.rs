//! Expert controls bound to one verified, Werk-owned oMLX worker.
//!
//! The worker owns the actual tensors and disk reads. This adapter only routes
//! the existing expert protocol; it does not manufacture residency metadata.

use super::{OmlxProcess, encode_path_segment, residency_adapter};
use crate::{
    model_store::{ModelManifest, ModelRuntimeIdentity, ModelStore},
    runtime_control::{
        BackendRuntimeAdapter, BackendRuntimeDescriptor, require_expert_capability,
        validate_expert_action, validate_expert_action_response, validate_expert_filter,
        validate_expert_list_response,
    },
    werk_protocol::{
        Capability, CapabilityStatus, ExpertAction, ExpertActionRequest, ExpertActionResponse,
        ExpertListFilter, ExpertListResponse, ExpertSummary, ExpertTier, ProtocolError,
        ProtocolErrorCode,
    },
};
use std::{sync::Arc, time::Duration};

const EXPERT_CAPABILITY: &str = "runtime.experts.residency";
const MAX_PAGE_SIZE: u16 = 1_000;
const MAX_ACTION_IDS: usize = 4_096;
const CONTROL_TIMEOUT: Duration = Duration::from_secs(120);
const STATUS_TIMEOUT: Duration = Duration::from_secs(2);

pub(super) struct OmlxRuntimeAdapter {
    server: Option<Arc<OmlxProcess>>,
    store: ModelStore,
}

impl OmlxRuntimeAdapter {
    pub(super) fn new(server: Option<Arc<OmlxProcess>>, store: ModelStore) -> Self {
        Self { server, store }
    }

    fn active_server(&self) -> Result<&OmlxProcess, ProtocolError> {
        self.server
            .as_deref()
            .filter(|server| server.is_running())
            .ok_or_else(|| unavailable("the selected oMLX worker is no longer active"))
    }

    fn verify_manifest(&self, manifest: &ModelManifest) -> Result<(), ProtocolError> {
        let server = self.active_server()?;
        let identity = ModelRuntimeIdentity::from_manifest(manifest).map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "cannot identify the selected model manifest",
            )
        })?;
        if server.model_identity.as_ref() != Some(&identity)
            || server.logical_model_id.as_deref() != Some(manifest.id.as_str())
        {
            return Err(unavailable(
                "the selected model does not match this oMLX worker; prepare the model again",
            ));
        }
        Ok(())
    }

    fn logical_model_id(&self, requested: Option<&str>) -> Result<String, ProtocolError> {
        let server = self.active_server()?;
        let logical_id = server
            .logical_model_id
            .as_deref()
            .ok_or_else(|| unavailable("the oMLX worker has no verified logical model identity"))?;
        if requested.is_some_and(|id| id != logical_id) {
            return Err(unavailable(
                "the requested model is not owned by this oMLX worker",
            ));
        }
        // This is deliberately the read-only lookup: telemetry and dry runs
        // must not initialize an absent store or silently load another model.
        let manifest = self
            .store
            .get_existing(logical_id)
            .map_err(|_| unavailable("the active oMLX model manifest is no longer available"))?;
        self.verify_manifest(&manifest)?;
        Ok(logical_id.to_string())
    }

    fn expert_model_is_active(server: &OmlxProcess) -> bool {
        server
            .json_request("GET", "/werk/experts/status", None, STATUS_TIMEOUT)
            .ok()
            .is_some_and(|status| {
                status.get("active").and_then(serde_json::Value::as_bool) == Some(true)
                    && status.get("model_id").and_then(serde_json::Value::as_str)
                        == Some(server.model_name.as_str())
            })
    }
}

impl BackendRuntimeAdapter for OmlxRuntimeAdapter {
    fn descriptor(&self) -> BackendRuntimeDescriptor {
        let active = self.server.as_deref().filter(|server| server.is_running());
        let mut descriptor = residency_adapter(active).descriptor();
        let (status, detail, operations) = match active {
            Some(server) if server.expert_offload && Self::expert_model_is_active(server) => (
                CapabilityStatus::Experimental,
                "Werk owns disk-backed MoE expert tensors and a bounded unified-memory RAM cache in this oMLX worker; prefetch supports ram only",
                vec!["list", "prefetch", "pin", "unpin", "evict"],
            ),
            Some(server) if server.expert_offload => (
                CapabilityStatus::Unavailable,
                "the oMLX worker cannot confirm an active offloaded model; load the model or retry after generation completes",
                vec![],
            ),
            Some(_) => (
                CapabilityStatus::Unsupported,
                "this oMLX worker has no active expert offload adapter; configure WERK_OMLX_EXPERT_CACHE_MB before loading a compatible model",
                vec![],
            ),
            None => (
                CapabilityStatus::Unavailable,
                "no unique active oMLX worker is available to verify expert residency",
                vec![],
            ),
        };
        descriptor.capabilities.push(Capability {
            id: EXPERT_CAPABILITY.to_string(),
            status,
            detail: detail.to_string(),
            operations: operations.into_iter().map(str::to_string).collect(),
        });
        descriptor
    }

    fn descriptor_for_model(
        &self,
        manifest: &ModelManifest,
    ) -> Result<BackendRuntimeDescriptor, ProtocolError> {
        if self.active_server().is_err() {
            // Preserve static metadata inspection before a worker is loaded.
            // Prepared operations still fail their unavailable capability gate.
            return Ok(self.descriptor());
        }
        self.verify_manifest(manifest)?;
        Ok(self.descriptor())
    }

    fn list_experts(&self, filter: &ExpertListFilter) -> Result<ExpertListResponse, ProtocolError> {
        validate_expert_filter(filter, MAX_PAGE_SIZE)?;
        require_expert_capability(&self.descriptor(), filter.allow_experimental, false)?;
        let logical_id = self.logical_model_id(filter.model_id.as_deref())?;
        let server = self.active_server()?;
        let value = server
            .json_request(
                "GET",
                &list_path(filter, &server.model_name),
                None,
                CONTROL_TIMEOUT,
            )
            .map_err(worker_error)?;
        let mut response: ExpertListResponse =
            serde_json::from_value(value).map_err(|_| invalid_response())?;
        rebind_summaries(&mut response.experts, &server.model_name, &logical_id)?;
        let logical_filter = ExpertListFilter {
            model_id: Some(logical_id),
            ..filter.clone()
        };
        validate_expert_list_response(&response, &logical_filter, MAX_PAGE_SIZE)?;
        Ok(response)
    }

    fn expert_action(
        &self,
        request: &ExpertActionRequest,
    ) -> Result<ExpertActionResponse, ProtocolError> {
        validate_expert_action(request, MAX_ACTION_IDS)?;
        require_expert_capability(&self.descriptor(), request.allow_experimental, true)?;
        if request.action == ExpertAction::Prefetch && request.target_tier == Some(ExpertTier::Vram)
        {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Unsupported,
                "oMLX expert prefetch uses unified-memory RAM; select target_tier ram",
            ));
        }
        let logical_id = self.logical_model_id(Some(&request.model_id))?;
        let server = self.active_server()?;
        let physical_request = ExpertActionRequest {
            model_id: server.model_name.clone(),
            ..request.clone()
        };
        let body = serde_json::to_value(physical_request).map_err(|_| invalid_response())?;
        let value = server
            .json_request(
                "POST",
                "/werk/experts/actions",
                Some(&body),
                CONTROL_TIMEOUT,
            )
            .map_err(worker_error)?;
        let mut response: ExpertActionResponse =
            serde_json::from_value(value).map_err(|_| invalid_response())?;
        rebind_summaries(&mut response.experts, &server.model_name, &logical_id)?;
        validate_expert_action_response(&response, request)?;
        Ok(response)
    }
}

fn list_path(filter: &ExpertListFilter, physical_model_id: &str) -> String {
    let mut query = vec![format!(
        "model_id={}",
        encode_path_segment(physical_model_id)
    )];
    if let Some(tier) = filter.tier {
        query.push(format!(
            "tier={}",
            match tier {
                ExpertTier::Vram => "vram",
                ExpertTier::Ram => "ram",
                ExpertTier::External => "external",
            }
        ));
    }
    if let Some(limit) = filter.limit {
        query.push(format!("limit={limit}"));
    }
    if let Some(cursor) = &filter.cursor {
        query.push(format!("cursor={}", encode_path_segment(cursor)));
    }
    if filter.allow_experimental {
        query.push("allow_experimental=true".to_string());
    }
    format!("/werk/experts?{}", query.join("&"))
}

fn rebind_summaries(
    experts: &mut [ExpertSummary],
    physical_model_id: &str,
    logical_model_id: &str,
) -> Result<(), ProtocolError> {
    for expert in experts {
        if expert.model_id != physical_model_id {
            return Err(invalid_response());
        }
        expert.model_id = logical_model_id.to_string();
    }
    Ok(())
}

fn unavailable(message: &str) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::Unavailable, message).retryable(true)
}

fn invalid_response() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::Internal,
        "oMLX returned an invalid expert control response",
    )
}

fn worker_error(error: anyhow::Error) -> ProtocolError {
    // A timed-out action may already have taken effect. Do not advertise an
    // automatic retry; callers can inspect the real worker state first.
    // Only fixed messages from our embedded worker cross the public control
    // plane. Arbitrary upstream exceptions can contain private shard paths.
    let detail = format!("{error:#}");
    for (reason, code) in [
        (
            "requested experts and pins exceed the expert cache budget",
            ProtocolErrorCode::ResourceExhausted,
        ),
        (
            "pins must leave room for at least one demand-loaded expert",
            ProtocolErrorCode::ResourceExhausted,
        ),
        (
            "expert cache is full of pinned experts; unpin experts or raise its budget",
            ProtocolErrorCode::ResourceExhausted,
        ),
        (
            "cannot evict pinned experts; unpin them first",
            ProtocolErrorCode::Conflict,
        ),
        (
            "streamed model is not loaded",
            ProtocolErrorCode::Unavailable,
        ),
        (
            "checkpoint changed after expert streaming preflight",
            ProtocolErrorCode::Unavailable,
        ),
    ] {
        if detail.contains(reason) {
            return ProtocolError::new(code, format!("oMLX: {reason}"));
        }
    }
    ProtocolError::new(
        ProtocolErrorCode::Unavailable,
        "oMLX expert control request failed; inspect the worker log and refresh expert telemetry before retrying an action",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[cfg(unix)]
    struct Fixture {
        adapter: OmlxRuntimeAdapter,
        manifest: ModelManifest,
        root: std::path::PathBuf,
        status_worker: Option<(
            Arc<std::sync::atomic::AtomicBool>,
            std::thread::JoinHandle<String>,
        )>,
        status_active: Arc<std::sync::atomic::AtomicBool>,
    }

    #[cfg(unix)]
    impl Fixture {
        fn new(expert_offload: bool, url: &str) -> Self {
            use std::{
                collections::VecDeque,
                os::unix::process::CommandExt,
                process::{Command, Stdio},
                sync::Mutex,
            };
            let root = std::env::temp_dir().join(format!(
                "werk-omlx-expert-test-{}",
                super::super::random_id().unwrap()
            ));
            let store = ModelStore::resolve(Some(root.clone())).unwrap();
            let manifest = ModelManifest {
                id: "owner/model".to_string(),
                source: crate::model_store::ModelSource::LocalPath {
                    path: "fixture".to_string(),
                },
                format: crate::model_store::ModelFormat::Mlx,
                architecture: Some("deepseek_v4".to_string()),
                tokenizer_path: None,
                config_path: None,
                model_path: None,
                backend: "omlx".to_string(),
                created_unix: 0,
                files: vec![],
                artifacts: vec![],
                metadata: Default::default(),
            };
            std::fs::create_dir_all(store.model_dir(&manifest.id)).unwrap();
            store.write_manifest(&manifest).unwrap();
            let manifest = store.get_existing(&manifest.id).unwrap();
            let status_active = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let (url, status_worker) = if url == "http://127.0.0.1:1" {
                let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let (url, worker) = mock_http(None, status_active.clone(), stop.clone());
                (url, Some((stop, worker)))
            } else {
                (url.to_string(), None)
            };
            let mut child = Command::new("/bin/sh")
                .args(["-c", "read werk_lifetime"])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .process_group(0)
                .spawn()
                .unwrap();
            let parent_pipe = child.stdin.take();
            let server = OmlxProcess {
                child: Mutex::new(child),
                parent_pipe,
                _lifetime_locks: Vec::new(),
                url,
                api_key: "fixture-secret".to_string(),
                model_name: "physical / model".to_string(),
                model_dir: store.model_dir(&manifest.id),
                model_identity: Some(ModelRuntimeIdentity::from_manifest(&manifest).unwrap()),
                logical_model_id: Some(manifest.id.clone()),
                base_path: root.join("worker"),
                version: "fixture".to_string(),
                instance_id: "fixture-instance".to_string(),
                tools: false,
                log_tail: Arc::new(Mutex::new(VecDeque::new())),
                expert_offload,
                expert_cache_bytes: None,
                expert_execution: "grouped",
                thinking: None,
                server_prefix_cache: false,
            };
            Self {
                adapter: OmlxRuntimeAdapter::new(Some(Arc::new(server)), store),
                manifest,
                root,
                status_worker,
                status_active,
            }
        }

        fn request(&self) -> ExpertActionRequest {
            ExpertActionRequest {
                model_id: self.manifest.id.clone(),
                expert_ids: vec!["layer-1-expert-2".to_string()],
                action: ExpertAction::Prefetch,
                target_tier: Some(ExpertTier::Ram),
                dry_run: true,
                allow_experimental: true,
            }
        }
    }

    #[cfg(unix)]
    impl Drop for Fixture {
        fn drop(&mut self) {
            self.adapter.server.take();
            if let Some((stop, worker)) = self.status_worker.take() {
                stop.store(true, std::sync::atomic::Ordering::SeqCst);
                worker.join().unwrap();
            }
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[cfg(unix)]
    fn respond_once(body: serde_json::Value) -> (String, std::thread::JoinHandle<String>) {
        mock_http(
            Some(body),
            Arc::new(std::sync::atomic::AtomicBool::new(true)),
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )
    }

    #[cfg(unix)]
    fn mock_http(
        body: Option<serde_json::Value>,
        active: Arc<std::sync::atomic::AtomicBool>,
        stop: Arc<std::sync::atomic::AtomicBool>,
    ) -> (String, std::thread::JoinHandle<String>) {
        use std::{
            io::{Read, Write},
            net::TcpListener,
            thread,
            time::Instant,
        };
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            loop {
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut stream = loop {
                    if stop.load(std::sync::atomic::Ordering::SeqCst) {
                        return String::new();
                    }
                    if let Ok((stream, _)) = listener.accept() {
                        break stream;
                    }
                    assert!(Instant::now() < deadline, "adapter made no HTTP request");
                    thread::sleep(Duration::from_millis(5));
                };
                // Accepted sockets inherit listener nonblocking mode on macOS.
                // Keep only accept polling nonblocking; request reads use the
                // bounded blocking timeout below.
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    let count = stream.read(&mut chunk).unwrap();
                    assert!(count > 0, "incomplete HTTP request");
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(end) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
                        let header = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
                        let length: usize = header
                            .lines()
                            .find_map(|line| line.strip_prefix("content-length:"))
                            .map(|value| value.trim().parse().unwrap())
                            .unwrap_or(0);
                        if bytes.len() >= end + 4 + length {
                            break;
                        }
                    }
                }
                let request = String::from_utf8(bytes).unwrap();
                assert!(
                    request
                        .to_ascii_lowercase()
                        .contains("authorization: bearer fixture-secret\r\n")
                );
                let is_status = request.starts_with("GET /werk/experts/status ");
                let response = if is_status {
                json!({"active":active.load(std::sync::atomic::Ordering::SeqCst),"model_id":"physical / model"})
            } else {
                body.clone().expect("unexpected expert operation")
            }.to_string();
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", response.len(), response).unwrap();
                if !is_status {
                    return request;
                }
            }
        });
        (url, handle)
    }

    fn summary(model_id: &str) -> ExpertSummary {
        ExpertSummary {
            id: "layer-1-expert-2".to_string(),
            model_id: model_id.to_string(),
            tier: ExpertTier::External,
            bytes: Some(1024),
            hotness: 0.0,
            pinned: false,
            last_used_unix_ms: None,
        }
    }

    #[test]
    fn query_encodes_physical_model_and_cursor_without_query_injection() {
        let path = list_path(
            &ExpertListFilter {
                model_id: Some("owner/logical-model".to_string()),
                tier: Some(ExpertTier::External),
                limit: Some(17),
                cursor: Some("offset&model_id=other /+".to_string()),
                allow_experimental: true,
            },
            "physical / model",
        );
        assert_eq!(
            path,
            "/werk/experts?model_id=physical%20%2F%20model&tier=external&limit=17&cursor=offset%26model_id%3Dother%20%2F%2B&allow_experimental=true"
        );
        assert!(!path.contains("owner"));
    }

    #[test]
    fn rebinds_only_the_selected_physical_model() {
        let mut experts = vec![summary("physical")];
        rebind_summaries(&mut experts, "physical", "owner/logical").unwrap();
        assert_eq!(experts[0].model_id, "owner/logical");
        let mut unrelated = vec![summary("other")];
        assert_eq!(
            rebind_summaries(&mut unrelated, "physical", "owner/logical")
                .unwrap_err()
                .code,
            ProtocolErrorCode::Internal
        );
    }

    #[test]
    fn worker_errors_preserve_known_capacity_reason_without_private_details() {
        let error = worker_error(anyhow::anyhow!(
            "HTTP 400 /private/checkpoint: requested experts and pins exceed the expert cache budget"
        ));
        assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);
        assert!(!error.message.contains("/private"));
        assert!(error.message.contains("cache budget"));
        let unknown = worker_error(anyhow::anyhow!("open /private/checkpoint: IO failure"));
        assert_eq!(unknown.code, ProtocolErrorCode::Unavailable);
        assert!(!unknown.message.contains("/private"));
        assert!(!unknown.retryable);
    }

    #[test]
    fn absent_worker_never_claims_experts_or_named_state() {
        let path = std::env::temp_dir().join(format!(
            "werk-omlx-expert-missing-{}",
            super::super::random_id().unwrap()
        ));
        let adapter =
            OmlxRuntimeAdapter::new(None, ModelStore::resolve(Some(path.clone())).unwrap());
        let descriptor = adapter.descriptor();
        let expert = descriptor
            .capabilities
            .iter()
            .find(|capability| capability.id == EXPERT_CAPABILITY)
            .unwrap();
        assert_eq!(expert.status, CapabilityStatus::Unavailable);
        assert!(expert.operations.is_empty());
        assert_eq!(descriptor.capabilities.len(), 2);
        assert_eq!(
            adapter
                .list_experts(&ExpertListFilter {
                    allow_experimental: true,
                    ..Default::default()
                })
                .unwrap_err()
                .code,
            ProtocolErrorCode::Unavailable
        );
        assert!(!path.exists());
    }

    #[test]
    #[cfg(unix)]
    fn prepared_list_uses_authenticated_worker_and_rebinds_logical_model() {
        let (url, request) = respond_once(json!({
            "experts": [summary("physical / model")], "next_cursor": null
        }));
        let fixture = Fixture::new(true, &url);
        let filter = ExpertListFilter {
            model_id: Some(fixture.manifest.id.clone()),
            allow_experimental: true,
            ..Default::default()
        };
        let plan = fixture
            .adapter
            .prepare_expert_list(Some(&fixture.manifest), &filter)
            .unwrap();
        let response = fixture.adapter.list_experts_prepared(plan).unwrap();
        assert_eq!(response.experts[0].model_id, fixture.manifest.id);
        let sent = request.join().unwrap();
        assert!(sent.starts_with(
            "GET /werk/experts?model_id=physical%20%2F%20model&allow_experimental=true "
        ));
        assert!(
            sent.to_ascii_lowercase()
                .contains("authorization: bearer fixture-secret\r\n")
        );
    }

    #[test]
    #[cfg(unix)]
    fn prepared_action_preserves_dry_run_and_uses_physical_model() {
        let (url, request) = respond_once(json!({
            "experts": [summary("physical / model")], "changed": 0, "dry_run": true
        }));
        let fixture = Fixture::new(true, &url);
        let action = fixture.request();
        let plan = fixture
            .adapter
            .prepare_expert_action(&fixture.manifest, &action)
            .unwrap();
        let response = fixture.adapter.expert_action_prepared(plan).unwrap();
        assert_eq!(response.experts[0].model_id, fixture.manifest.id);
        assert!(response.dry_run);
        let sent = request.join().unwrap();
        assert!(sent.starts_with("POST /werk/experts/actions "));
        assert!(
            sent.to_ascii_lowercase()
                .contains("authorization: bearer fixture-secret\r\n")
        );
        let body: serde_json::Value =
            serde_json::from_str(sent.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["model_id"], "physical / model");
        assert_eq!(body["target_tier"], "ram");
        assert_eq!(body["dry_run"], true);
    }

    #[test]
    #[cfg(unix)]
    fn opt_in_and_model_binding_are_checked_before_http() {
        let fixture = Fixture::new(true, "http://127.0.0.1:1");
        let mut action = fixture.request();
        action.allow_experimental = false;
        assert_eq!(
            fixture.adapter.expert_action(&action).unwrap_err().code,
            ProtocolErrorCode::ExperimentalOptInRequired
        );
        action.allow_experimental = true;
        action.model_id = "other/model".to_string();
        assert_eq!(
            fixture.adapter.expert_action(&action).unwrap_err().code,
            ProtocolErrorCode::Unavailable
        );
        action.model_id = fixture.manifest.id.clone();
        action.target_tier = Some(ExpertTier::Vram);
        assert_eq!(
            fixture.adapter.expert_action(&action).unwrap_err().code,
            ProtocolErrorCode::Unsupported
        );

        let mut changed = fixture.manifest.clone();
        changed.created_unix = 1;
        assert_eq!(
            fixture
                .adapter
                .prepare_expert_action(&changed, &fixture.request())
                .unwrap_err()
                .code,
            ProtocolErrorCode::Unavailable
        );
        fixture.adapter.store.write_manifest(&changed).unwrap();
        assert_eq!(
            fixture
                .adapter
                .expert_action(&fixture.request())
                .unwrap_err()
                .code,
            ProtocolErrorCode::Unavailable
        );
    }

    #[test]
    #[cfg(unix)]
    fn normal_worker_remains_unsupported_and_dead_worker_invalidates_prepared_plan() {
        let ordinary = Fixture::new(false, "http://127.0.0.1:1");
        assert_eq!(
            ordinary
                .adapter
                .expert_action(&ordinary.request())
                .unwrap_err()
                .code,
            ProtocolErrorCode::Unsupported
        );
        let fixture = Fixture::new(true, "http://127.0.0.1:1");
        let plan = fixture
            .adapter
            .prepare_expert_action(&fixture.manifest, &fixture.request())
            .unwrap();
        let server = fixture.adapter.server.as_ref().unwrap();
        server.child.lock().unwrap().kill().unwrap();
        server.child.lock().unwrap().wait().unwrap();
        assert_eq!(
            fixture
                .adapter
                .expert_action_prepared(plan)
                .unwrap_err()
                .code,
            ProtocolErrorCode::Unavailable
        );
        let capability = fixture
            .adapter
            .descriptor()
            .capabilities
            .into_iter()
            .find(|capability| capability.id == EXPERT_CAPABILITY)
            .unwrap();
        assert_eq!(capability.status, CapabilityStatus::Unavailable);
        assert!(capability.operations.is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn model_unload_clears_experimental_capability_without_killing_worker() {
        let fixture = Fixture::new(true, "http://127.0.0.1:1");
        let capability = |descriptor: BackendRuntimeDescriptor| {
            descriptor
                .capabilities
                .into_iter()
                .find(|capability| capability.id == EXPERT_CAPABILITY)
                .unwrap()
        };
        assert_eq!(
            capability(fixture.adapter.descriptor()).status,
            CapabilityStatus::Experimental
        );
        let plan = fixture
            .adapter
            .prepare_expert_action(&fixture.manifest, &fixture.request())
            .unwrap();
        fixture
            .status_active
            .store(false, std::sync::atomic::Ordering::SeqCst);
        assert!(fixture.adapter.active_server().is_ok());
        let unloaded = capability(fixture.adapter.descriptor());
        assert_eq!(unloaded.status, CapabilityStatus::Unavailable);
        assert!(unloaded.operations.is_empty());
        assert_eq!(
            fixture
                .adapter
                .expert_action_prepared(plan)
                .unwrap_err()
                .code,
            ProtocolErrorCode::Unavailable
        );
    }
}
