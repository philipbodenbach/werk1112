use super::backend::require_expert_capability;
use super::{
    BackendDecodeRequest, BackendDecodeResult, BackendExpertOperationPlan,
    BackendMemoryRequirement, BackendPersistedStatePlan, BackendPersistedStateResolution,
    BackendPrefillRequest, BackendPrefillResult, BackendRuntimeAdapter, BackendRuntimeDescriptor,
    BackendSnapshot, BackendState, UnsupportedRuntimeAdapter, validate_compatibility,
    validate_compatibility_envelope, validate_runtime_descriptor,
};
use crate::{
    backend::{
        ChatGenerationSession, GenerateRequest, GenerateResponse, GenerateStream, GenerationBackend,
    },
    capabilities::InferenceTask,
    inference::TaskReadiness,
    model_store::ModelManifest,
    werk_protocol::{
        CompatibilityEnvelope, ExpertActionRequest, ExpertActionResponse, ExpertListFilter,
        ExpertListResponse, ProtocolError, ProtocolErrorCode, StateTier,
    },
};
use sha2::{Digest, Sha256};
use std::{collections::VecDeque, io::Write, sync::Arc, sync::Mutex};
use tokio_stream::StreamExt;

const MAX_MODEL_ROUTES: usize = 64;
// Keep every currently admissible handoff routable while remaining bounded by
// the control plane's own 1,024-entry handoff registry.
const MAX_COMPATIBILITY_ROUTES: usize = 1_024;
const MAX_INSTANCE_ROUTES: usize = 1_024;

/// Resolves runtime control through the same model-aware backend router used
/// by ordinary generation requests.
///
/// Route tables are deliberately small and bounded. A missing or expired
/// route fails closed instead of guessing which backend owns opaque state.
pub struct RoutedRuntimeAdapter {
    backend: Arc<dyn GenerationBackend>,
    routes: Mutex<RouteCache>,
}

/// Generation-side companion that keeps the control-plane descriptor aligned
/// with the concrete backend selected by ordinary API traffic.
pub(crate) struct RuntimeRoutedGenerationBackend {
    backend: Arc<dyn GenerationBackend>,
    runtime: Arc<RoutedRuntimeAdapter>,
}

struct RuntimeRoutedChatSession {
    session: Box<dyn ChatGenerationSession>,
    runtime: Arc<RoutedRuntimeAdapter>,
    adapter: Arc<dyn BackendRuntimeAdapter>,
}

impl ChatGenerationSession for RuntimeRoutedChatSession {
    fn generate(&self, request: GenerateRequest) -> anyhow::Result<GenerateResponse> {
        let response = self.session.generate(request)?;
        let _ = self.runtime.set_active(self.adapter.clone());
        Ok(response)
    }

    fn generate_stream(&self, request: GenerateRequest) -> GenerateStream {
        let stream = self.session.generate_stream(request);
        let runtime = self.runtime.clone();
        let mut adapter = Some(self.adapter.clone());
        Box::pin(stream.map(move |event| {
            if event.is_ok()
                && let Some(adapter) = adapter.take()
            {
                let _ = runtime.set_active(adapter);
            }
            event
        }))
    }
}

impl RuntimeRoutedGenerationBackend {
    pub(crate) fn new(
        backend: Arc<dyn GenerationBackend>,
        runtime: Arc<RoutedRuntimeAdapter>,
    ) -> Self {
        Self { backend, runtime }
    }
}

struct RouteCache {
    active: Arc<dyn BackendRuntimeAdapter>,
    models: VecDeque<ModelRoute>,
    compatibility: VecDeque<CompatibilityRoute>,
    instances: VecDeque<InstanceRoute>,
    pending_state_routes: usize,
}

struct ModelRoute {
    model_id: String,
    manifest_fingerprint: [u8; 32],
    has_images: bool,
    adapter: Arc<dyn BackendRuntimeAdapter>,
}

#[derive(Clone, PartialEq, Eq)]
struct CompatibilityKey(CompatibilityEnvelope);

impl CompatibilityKey {
    fn new(compatibility: &CompatibilityEnvelope) -> Self {
        Self(compatibility.clone())
    }
}

struct CompatibilityRoute {
    key: CompatibilityKey,
    model_id: String,
    adapter: Arc<dyn BackendRuntimeAdapter>,
}

struct InstanceRoute {
    instance_id: String,
    adapter: Arc<dyn BackendRuntimeAdapter>,
    live_states: usize,
}

struct StateRouteReservation<'a> {
    router: &'a RoutedRuntimeAdapter,
    kind: StateRouteReservationKind,
    active: bool,
}

enum StateRouteReservationKind {
    Existing {
        instance_id: String,
        adapter: Arc<dyn BackendRuntimeAdapter>,
    },
    New,
}

impl RoutedRuntimeAdapter {
    pub fn new(backend: Arc<dyn GenerationBackend>) -> Self {
        let initial = backend.runtime_control_adapter();
        Self {
            backend,
            routes: Mutex::new(RouteCache {
                active: initial,
                models: VecDeque::new(),
                compatibility: VecDeque::new(),
                instances: VecDeque::new(),
                pending_state_routes: 0,
            }),
        }
    }

    fn adapter_for_model(
        &self,
        manifest: &ModelManifest,
    ) -> Result<Arc<dyn BackendRuntimeAdapter>, ProtocolError> {
        self.resolve_model_route(manifest, false, false)
    }

    fn adapter_for_request(
        &self,
        manifest: &ModelManifest,
        has_images: bool,
    ) -> Result<Arc<dyn BackendRuntimeAdapter>, ProtocolError> {
        if !has_images {
            return self.adapter_for_model(manifest);
        }
        self.resolve_model_route(manifest, true, true)
    }

    fn resolve_model_route(
        &self,
        manifest: &ModelManifest,
        has_images: bool,
        request_aware: bool,
    ) -> Result<Arc<dyn BackendRuntimeAdapter>, ProtocolError> {
        let manifest_fingerprint = manifest_fingerprint(manifest)?;
        if let Some(adapter) = self.cached_model_adapter(&manifest_fingerprint, has_images)? {
            return Ok(adapter);
        }

        let selected = if request_aware {
            self.backend
                .runtime_control_adapter_for_request(manifest, has_images)
        } else {
            self.backend.runtime_control_adapter_for(manifest)
        }
        .map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::Unavailable,
                "runtime-control backend selection failed for the requested model",
            )
        })?;

        // A concurrent resolver may have populated the route while selection
        // ran without the routing lock. Keep exactly one adapter per model.
        let mut routes = self.lock_routes()?;
        if let Some(index) = routes.models.iter().position(|route| {
            route.manifest_fingerprint == manifest_fingerprint && route.has_images == has_images
        }) {
            let route = routes
                .models
                .remove(index)
                .expect("a located model route must exist");
            let adapter = route.adapter.clone();
            routes.models.push_back(route);
            return Ok(adapter);
        }
        push_bounded(
            &mut routes.models,
            ModelRoute {
                model_id: manifest.id.clone(),
                manifest_fingerprint,
                has_images,
                adapter: selected.clone(),
            },
            MAX_MODEL_ROUTES,
        );
        Ok(selected)
    }

    fn cached_model_adapter(
        &self,
        manifest_fingerprint: &[u8; 32],
        has_images: bool,
    ) -> Result<Option<Arc<dyn BackendRuntimeAdapter>>, ProtocolError> {
        let mut routes = self.lock_routes()?;
        let Some(index) = routes.models.iter().position(|route| {
            &route.manifest_fingerprint == manifest_fingerprint && route.has_images == has_images
        }) else {
            return Ok(None);
        };
        let route = routes
            .models
            .remove(index)
            .expect("a located model route must exist");
        let adapter = route.adapter.clone();
        routes.models.push_back(route);
        Ok(Some(adapter))
    }

    fn active_adapter(&self) -> Result<Arc<dyn BackendRuntimeAdapter>, ProtocolError> {
        Ok(self.lock_routes()?.active.clone())
    }

    fn known_model_adapter(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn BackendRuntimeAdapter>, ProtocolError> {
        let mut routes = self.lock_routes()?;
        let Some(index) = routes
            .models
            .iter()
            .rposition(|route| route.model_id == model_id && !route.has_images)
        else {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Unavailable,
                "no runtime adapter has been resolved for the requested model",
            )
            .retryable(true));
        };
        let route = routes
            .models
            .remove(index)
            .expect("a located model route must exist");
        let adapter = route.adapter.clone();
        routes.models.push_back(route);
        Ok(adapter)
    }

    fn set_active(&self, adapter: Arc<dyn BackendRuntimeAdapter>) -> Result<(), ProtocolError> {
        self.lock_routes()?.active = adapter;
        Ok(())
    }

    fn record_compatibility(
        &self,
        model_id: &str,
        compatibility: &CompatibilityEnvelope,
        adapter: Arc<dyn BackendRuntimeAdapter>,
    ) -> Result<(), ProtocolError> {
        let key = CompatibilityKey::new(compatibility);
        let mut routes = self.lock_routes()?;
        routes
            .compatibility
            .retain(|route| route.key != key || route.model_id != model_id);
        push_bounded(
            &mut routes.compatibility,
            CompatibilityRoute {
                key,
                model_id: model_id.to_string(),
                adapter: adapter.clone(),
            },
            MAX_COMPATIBILITY_ROUTES,
        );
        routes.active = adapter;
        Ok(())
    }

    fn compatibility_route(
        &self,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<(String, Arc<dyn BackendRuntimeAdapter>), ProtocolError> {
        let key = CompatibilityKey::new(compatibility);
        let mut routes = self.lock_routes()?;
        let Some(index) = routes
            .compatibility
            .iter()
            .rposition(|route| route.key == key)
        else {
            return Err(incompatible_route());
        };
        let route = routes
            .compatibility
            .remove(index)
            .expect("a located compatibility route must exist");
        let result = (route.model_id.clone(), route.adapter.clone());
        routes.compatibility.push_back(route);
        Ok(result)
    }

    fn compatibility_route_for_model(
        &self,
        model_id: &str,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<Arc<dyn BackendRuntimeAdapter>, ProtocolError> {
        let key = CompatibilityKey::new(compatibility);
        let mut routes = self.lock_routes()?;
        let Some(index) = routes
            .compatibility
            .iter()
            .rposition(|route| route.key == key && route.model_id == model_id)
        else {
            return Err(incompatible_route());
        };
        let route = routes
            .compatibility
            .remove(index)
            .expect("a located compatibility route must exist");
        let adapter = route.adapter.clone();
        routes.compatibility.push_back(route);
        Ok(adapter)
    }

    fn adapter_for_compatibility(
        &self,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<Arc<dyn BackendRuntimeAdapter>, ProtocolError> {
        self.compatibility_route(compatibility)
            .map(|(_, adapter)| adapter)
    }

    fn record_state(
        &self,
        state: &BackendState,
        adapter: Arc<dyn BackendRuntimeAdapter>,
    ) -> Result<(), ProtocolError> {
        let mut routes = self.lock_routes()?;
        record_instance_locked(&mut routes, state.instance_id().to_string(), adapter)
    }

    fn reserve_state_route(
        &self,
        adapter: &Arc<dyn BackendRuntimeAdapter>,
    ) -> Result<StateRouteReservation<'_>, ProtocolError> {
        let descriptor = adapter.descriptor();
        validate_runtime_descriptor(&descriptor)?;
        let expected_instance = descriptor.instance_id;
        let mut routes = self.lock_routes()?;
        if let Some(index) = routes
            .instances
            .iter()
            .position(|route| route.instance_id == expected_instance)
        {
            if !Arc::ptr_eq(&routes.instances[index].adapter, adapter) {
                return Err(ProtocolError::new(
                    ProtocolErrorCode::Internal,
                    "backend process identity is not unique across runtime adapters",
                ));
            }
            routes.instances[index].live_states = routes.instances[index]
                .live_states
                .checked_add(1)
                .ok_or_else(|| {
                    ProtocolError::new(
                        ProtocolErrorCode::ResourceExhausted,
                        "runtime state ownership count is exhausted",
                    )
                })?;
            return Ok(StateRouteReservation {
                router: self,
                kind: StateRouteReservationKind::Existing {
                    instance_id: expected_instance,
                    adapter: adapter.clone(),
                },
                active: true,
            });
        }
        if routes
            .instances
            .len()
            .saturating_add(routes.pending_state_routes)
            >= MAX_INSTANCE_ROUTES
        {
            return Err(ProtocolError::new(
                ProtocolErrorCode::ResourceExhausted,
                "runtime state process-route capacity is exhausted",
            ));
        }
        routes.pending_state_routes = routes.pending_state_routes.saturating_add(1);
        Ok(StateRouteReservation {
            router: self,
            kind: StateRouteReservationKind::New,
            active: true,
        })
    }

    fn release_state_route(
        &self,
        state: &BackendState,
        adapter: &Arc<dyn BackendRuntimeAdapter>,
    ) -> Result<(), ProtocolError> {
        let mut routes = self.lock_routes()?;
        release_instance_locked(&mut routes, state.instance_id(), adapter)
    }

    fn adapter_for_state(
        &self,
        state: &BackendState,
    ) -> Result<Arc<dyn BackendRuntimeAdapter>, ProtocolError> {
        let mut routes = self.lock_routes()?;
        let Some(index) = routes
            .instances
            .iter()
            .rposition(|route| route.instance_id == state.instance_id())
        else {
            return Err(incompatible_route());
        };
        let route = routes
            .instances
            .remove(index)
            .expect("a located state route must exist");
        let adapter = route.adapter.clone();
        routes.instances.push_back(route);
        Ok(adapter)
    }

    fn adapter_for_model_id_or_compatibility(
        &self,
        model_id: &str,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<Arc<dyn BackendRuntimeAdapter>, ProtocolError> {
        self.compatibility_route_for_model(model_id, compatibility)
    }

    fn lock_routes(&self) -> Result<std::sync::MutexGuard<'_, RouteCache>, ProtocolError> {
        self.routes.lock().map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "runtime-control routing is unavailable",
            )
        })
    }
}

impl StateRouteReservation<'_> {
    fn commit(
        mut self,
        state: &BackendState,
        committed_adapter: Arc<dyn BackendRuntimeAdapter>,
    ) -> Result<(), ProtocolError> {
        let mut routes = self.router.lock_routes()?;
        match &self.kind {
            StateRouteReservationKind::Existing {
                instance_id,
                adapter: reserved_adapter,
            } if state.instance_id() == instance_id
                && Arc::ptr_eq(reserved_adapter, &committed_adapter) =>
            {
                self.active = false;
                return Ok(());
            }
            StateRouteReservationKind::Existing {
                instance_id,
                adapter: reserved_adapter,
            } => {
                release_instance_locked(&mut routes, instance_id, reserved_adapter)?;
            }
            StateRouteReservationKind::New => {
                routes.pending_state_routes = routes.pending_state_routes.saturating_sub(1);
            }
        }
        self.active = false;
        record_instance_locked(
            &mut routes,
            state.instance_id().to_string(),
            committed_adapter,
        )
    }
}

impl Drop for StateRouteReservation<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Ok(mut routes) = self.router.routes.lock() {
            match &self.kind {
                StateRouteReservationKind::Existing {
                    instance_id,
                    adapter,
                } => {
                    let _ = release_instance_locked(&mut routes, instance_id, adapter);
                }
                StateRouteReservationKind::New => {
                    routes.pending_state_routes = routes.pending_state_routes.saturating_sub(1);
                }
            }
        }
        self.active = false;
    }
}

impl GenerationBackend for RuntimeRoutedGenerationBackend {
    fn runtime_control_adapter(&self) -> Arc<dyn BackendRuntimeAdapter> {
        self.runtime.clone()
    }

    fn runtime_control_adapter_for(
        &self,
        manifest: &ModelManifest,
    ) -> anyhow::Result<Arc<dyn BackendRuntimeAdapter>> {
        self.runtime
            .adapter_for_model(manifest)
            .map_err(anyhow::Error::new)
    }

    fn runtime_control_adapter_for_request(
        &self,
        manifest: &ModelManifest,
        has_images: bool,
    ) -> anyhow::Result<Arc<dyn BackendRuntimeAdapter>> {
        self.runtime
            .adapter_for_request(manifest, has_images)
            .map_err(anyhow::Error::new)
    }

    fn prepare(&self, manifest: &ModelManifest) -> anyhow::Result<()> {
        let selected = self.runtime.adapter_for_model(manifest).ok();
        self.backend.prepare(manifest)?;
        if let Some(adapter) = selected {
            let _ = self.runtime.set_active(adapter);
        }
        Ok(())
    }

    fn start_chat_session(
        &self,
        manifest: &ModelManifest,
        seed: Option<u64>,
    ) -> anyhow::Result<Option<Box<dyn ChatGenerationSession>>> {
        let selected = self.runtime.adapter_for_model(manifest).ok();
        let session = self.backend.start_chat_session(manifest, seed)?;
        match (session, selected) {
            (Some(session), Some(adapter)) => {
                let _ = self.runtime.set_active(adapter.clone());
                Ok(Some(Box::new(RuntimeRoutedChatSession {
                    session,
                    runtime: self.runtime.clone(),
                    adapter,
                })))
            }
            (session, _) => Ok(session),
        }
    }

    fn task_readiness(
        &self,
        manifest: &ModelManifest,
        task: InferenceTask,
    ) -> Option<TaskReadiness> {
        self.backend.task_readiness(manifest, task)
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> anyhow::Result<GenerateResponse> {
        let has_images = !request.image_urls.is_empty();
        let selected = self.runtime.adapter_for_request(manifest, has_images).ok();
        let response = self.backend.generate(manifest, request)?;
        if let Some(adapter) = selected {
            let _ = self.runtime.set_active(adapter);
        }
        Ok(response)
    }

    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        let has_images = !request.image_urls.is_empty();
        let mut selected = self.runtime.adapter_for_request(&manifest, has_images).ok();
        let stream = self.backend.generate_stream(manifest, request);
        let runtime = self.runtime.clone();
        Box::pin(stream.map(move |event| {
            if event.is_ok()
                && let Some(adapter) = selected.take()
            {
                let _ = runtime.set_active(adapter);
            }
            event
        }))
    }
}

impl BackendRuntimeAdapter for RoutedRuntimeAdapter {
    fn descriptor(&self) -> BackendRuntimeDescriptor {
        let active = match self.routes.lock() {
            Ok(routes) => routes.active.clone(),
            Err(_) => return UnsupportedRuntimeAdapter::new("runtime-router").descriptor(),
        };
        active.descriptor()
    }

    fn descriptor_for_model(
        &self,
        manifest: &ModelManifest,
    ) -> Result<BackendRuntimeDescriptor, ProtocolError> {
        let descriptor = self.adapter_for_model(manifest)?.descriptor();
        validate_runtime_descriptor(&descriptor)?;
        Ok(descriptor)
    }

    fn descriptor_for_compatibility(
        &self,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<BackendRuntimeDescriptor, ProtocolError> {
        let descriptor = self.adapter_for_compatibility(compatibility)?.descriptor();
        validate_runtime_descriptor(&descriptor)?;
        Ok(descriptor)
    }

    fn descriptor_for_state(
        &self,
        state: &BackendState,
    ) -> Result<BackendRuntimeDescriptor, ProtocolError> {
        let descriptor = self.adapter_for_state(state)?.descriptor();
        validate_runtime_descriptor(&descriptor)?;
        Ok(descriptor)
    }

    fn compatibility(
        &self,
        manifest: &ModelManifest,
        prompt_fingerprint: &str,
    ) -> Result<CompatibilityEnvelope, ProtocolError> {
        let adapter = self.adapter_for_model(manifest)?;
        let compatibility = adapter.compatibility(manifest, prompt_fingerprint)?;
        validate_compatibility_envelope(&compatibility).map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend returned an invalid or unbounded compatibility envelope",
            )
        })?;
        self.record_compatibility(&manifest.id, &compatibility, adapter)?;
        Ok(compatibility)
    }

    fn inspect_compatibility(
        &self,
        manifest: &ModelManifest,
        prompt_fingerprint: &str,
    ) -> Result<CompatibilityEnvelope, ProtocolError> {
        let compatibility = self
            .adapter_for_model(manifest)?
            .inspect_compatibility(manifest, prompt_fingerprint)?;
        validate_compatibility_envelope(&compatibility).map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend returned an invalid or unbounded compatibility envelope",
            )
        })?;
        Ok(compatibility)
    }

    fn resolve_persisted_state(
        &self,
        manifest: &ModelManifest,
        snapshot: &BackendSnapshot,
        expected: &CompatibilityEnvelope,
    ) -> Result<BackendPersistedStateResolution, ProtocolError> {
        let adapter = match self.compatibility_route_for_model(&manifest.id, expected) {
            Ok(adapter) => adapter,
            Err(error) if error.code == ProtocolErrorCode::IncompatibleState => {
                self.adapter_for_model(manifest)?
            }
            Err(error) => return Err(error),
        };
        let resolution = adapter.resolve_persisted_state(manifest, snapshot, expected)?;
        validate_compatibility_envelope(&resolution.compatibility).map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend resolved an invalid or unbounded compatibility envelope",
            )
        })?;
        validate_compatibility(expected, &resolution.compatibility)?;
        self.record_compatibility(&manifest.id, expected, adapter)?;
        Ok(resolution)
    }

    fn prepare_persisted_state(
        &self,
        manifest: &ModelManifest,
        snapshot: &BackendSnapshot,
        expected: &CompatibilityEnvelope,
    ) -> Result<BackendPersistedStatePlan, ProtocolError> {
        validate_compatibility_envelope(expected).map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::IncompatibleState,
                "persisted-state restore compatibility is invalid",
            )
        })?;
        let adapter = match self.compatibility_route_for_model(&manifest.id, expected) {
            Ok(adapter) => adapter,
            Err(error) if error.code == ProtocolErrorCode::IncompatibleState => {
                self.adapter_for_model(manifest)?
            }
            Err(error) => return Err(error),
        };
        let resolution = adapter.resolve_persisted_state(manifest, snapshot, expected)?;
        validate_compatibility_envelope(&resolution.compatibility).map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend resolved an invalid or unbounded compatibility envelope",
            )
        })?;
        validate_compatibility(expected, &resolution.compatibility)?;
        let descriptor = adapter.descriptor();
        validate_runtime_descriptor(&descriptor)?;
        let requirement = adapter.restore_memory_requirement(snapshot, expected)?;
        let plan = BackendPersistedStatePlan::routed(
            resolution,
            descriptor,
            requirement,
            snapshot,
            expected,
            adapter.clone(),
        );
        plan.validate_current_descriptor(&adapter.descriptor())?;
        self.record_compatibility(&manifest.id, expected, adapter)?;
        Ok(plan)
    }

    fn inspect_persisted_state(
        &self,
        manifest: &ModelManifest,
        snapshot: &BackendSnapshot,
        expected: &CompatibilityEnvelope,
    ) -> Result<
        (
            BackendPersistedStateResolution,
            Option<BackendMemoryRequirement>,
        ),
        ProtocolError,
    > {
        let adapter = self.adapter_for_model(manifest)?;
        let (resolution, requirement) =
            adapter.inspect_persisted_state(manifest, snapshot, expected)?;
        validate_compatibility_envelope(&resolution.compatibility).map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend inspected an invalid or unbounded compatibility envelope",
            )
        })?;
        validate_compatibility(expected, &resolution.compatibility)?;
        Ok((resolution, requirement))
    }

    fn validate_state(
        &self,
        state: &BackendState,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<(), ProtocolError> {
        self.adapter_for_state(state)?
            .validate_state(state, compatibility)
    }

    fn prefill_memory_requirement(
        &self,
        request: &BackendPrefillRequest,
    ) -> Result<Option<BackendMemoryRequirement>, ProtocolError> {
        self.adapter_for_model_id_or_compatibility(&request.model_id, &request.compatibility)?
            .prefill_memory_requirement(request)
    }

    fn restore_memory_requirement(
        &self,
        snapshot: &BackendSnapshot,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<Option<BackendMemoryRequirement>, ProtocolError> {
        self.adapter_for_compatibility(compatibility)?
            .restore_memory_requirement(snapshot, compatibility)
    }

    fn decode_memory_requirement(
        &self,
        request: &BackendDecodeRequest,
    ) -> Result<Option<BackendMemoryRequirement>, ProtocolError> {
        self.adapter_for_state(request.state.state())?
            .decode_memory_requirement(request)
    }

    fn prefill(
        &self,
        request: BackendPrefillRequest,
    ) -> Result<BackendPrefillResult, ProtocolError> {
        let adapter =
            self.adapter_for_model_id_or_compatibility(&request.model_id, &request.compatibility)?;
        let route = self.reserve_state_route(&adapter)?;
        let result = adapter.prefill(request)?;
        commit_state_route(route, &result.state, adapter)?;
        Ok(result)
    }

    fn decode(&self, request: BackendDecodeRequest) -> Result<BackendDecodeResult, ProtocolError> {
        let adapter = self.adapter_for_state(request.state.state())?;
        adapter.validate_state(request.state.state(), &request.compatibility)?;
        let route = self.reserve_state_route(&adapter)?;
        let result = adapter.decode(request)?;
        if let Some(state) = &result.state {
            commit_state_route(route, state, adapter)?;
        } else {
            drop(route);
        }
        Ok(result)
    }

    fn restore(
        &self,
        snapshot: BackendSnapshot,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<BackendState, ProtocolError> {
        let adapter = self.adapter_for_compatibility(compatibility)?;
        let route = self.reserve_state_route(&adapter)?;
        let state = adapter.restore(snapshot, compatibility)?;
        commit_state_route(route, &state, adapter)?;
        Ok(state)
    }

    fn restore_prepared_state(
        &self,
        plan: BackendPersistedStatePlan,
        snapshot: BackendSnapshot,
        expected: &CompatibilityEnvelope,
    ) -> Result<BackendState, ProtocolError> {
        plan.validate_restore(&snapshot, expected)?;
        let adapter = plan.routed_adapter()?;
        plan.validate_current_descriptor(&adapter.descriptor())?;
        let route = self.reserve_state_route(&adapter)?;
        let state = adapter.restore(snapshot, expected)?;
        commit_state_route(route, &state, adapter)?;
        Ok(state)
    }

    fn snapshot(&self, state: &BackendState) -> Result<BackendState, ProtocolError> {
        let adapter = self.adapter_for_state(state)?;
        let route = self.reserve_state_route(&adapter)?;
        let snapshot = adapter.snapshot(state)?;
        commit_state_route(route, &snapshot, adapter)?;
        Ok(snapshot)
    }

    fn inspect_snapshot_export(
        &self,
        state: &BackendState,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<(), ProtocolError> {
        self.adapter_for_state(state)?
            .inspect_snapshot_export(state, compatibility)
    }

    fn validate_snapshot(
        &self,
        snapshot: &BackendState,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<(), ProtocolError> {
        self.adapter_for_state(snapshot)?
            .validate_snapshot(snapshot, compatibility)
    }

    fn move_state(
        &self,
        state: Arc<BackendState>,
        target: StateTier,
    ) -> Result<BackendState, ProtocolError> {
        let adapter = self.adapter_for_state(&state)?;
        let route = self.reserve_state_route(&adapter)?;
        let moved = adapter.move_state(state, target)?;
        commit_state_route(route, &moved, adapter)?;
        Ok(moved)
    }

    fn release(&self, state: &BackendState) -> Result<(), ProtocolError> {
        let adapter = self.adapter_for_state(state)?;
        adapter.release(state)?;
        self.release_state_route(state, &adapter)
    }

    fn list_experts(&self, filter: &ExpertListFilter) -> Result<ExpertListResponse, ProtocolError> {
        let adapter = match filter.model_id.as_deref() {
            Some(model_id) => self.known_model_adapter(model_id)?,
            None => self.active_adapter()?,
        };
        require_expert_capability(&adapter.descriptor(), filter.allow_experimental, false)?;
        adapter.list_experts(filter)
    }

    fn prepare_expert_list(
        &self,
        manifest: Option<&ModelManifest>,
        filter: &ExpertListFilter,
    ) -> Result<BackendExpertOperationPlan, ProtocolError> {
        let adapter = match manifest {
            Some(manifest) => self.adapter_for_model(manifest)?,
            None => self.active_adapter()?,
        };
        let descriptor = adapter.descriptor();
        BackendExpertOperationPlan::routed_list(manifest, descriptor, filter, adapter)
    }

    fn list_experts_prepared(
        &self,
        plan: BackendExpertOperationPlan,
    ) -> Result<ExpertListResponse, ProtocolError> {
        let (adapter, filter) = plan.into_routed_list()?;
        adapter.list_experts(&filter)
    }

    fn expert_action(
        &self,
        request: &ExpertActionRequest,
    ) -> Result<ExpertActionResponse, ProtocolError> {
        let adapter = self.known_model_adapter(&request.model_id)?;
        require_expert_capability(&adapter.descriptor(), request.allow_experimental, true)?;
        adapter.expert_action(request)
    }

    fn prepare_expert_action(
        &self,
        manifest: &ModelManifest,
        request: &ExpertActionRequest,
    ) -> Result<BackendExpertOperationPlan, ProtocolError> {
        let adapter = self.adapter_for_model(manifest)?;
        let descriptor = adapter.descriptor();
        BackendExpertOperationPlan::routed_action(manifest, descriptor, request, adapter)
    }

    fn expert_action_prepared(
        &self,
        plan: BackendExpertOperationPlan,
    ) -> Result<ExpertActionResponse, ProtocolError> {
        let (adapter, request) = plan.into_routed_action()?;
        adapter.expert_action(&request)
    }
}

fn record_instance_locked(
    routes: &mut RouteCache,
    instance_id: String,
    adapter: Arc<dyn BackendRuntimeAdapter>,
) -> Result<(), ProtocolError> {
    if let Some(index) = routes
        .instances
        .iter()
        .position(|route| route.instance_id == instance_id)
    {
        let existing = routes
            .instances
            .remove(index)
            .expect("a located instance route must exist");
        if !Arc::ptr_eq(&existing.adapter, &adapter) {
            routes.instances.insert(index, existing);
            return Err(ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend process identity is not unique across runtime adapters",
            ));
        }
        let mut existing = existing;
        let Some(live_states) = existing.live_states.checked_add(1) else {
            routes.instances.insert(index, existing);
            return Err(ProtocolError::new(
                ProtocolErrorCode::ResourceExhausted,
                "runtime state ownership count is exhausted",
            ));
        };
        existing.live_states = live_states;
        routes.instances.push_back(existing);
        return Ok(());
    }
    if routes.instances.len() == MAX_INSTANCE_ROUTES {
        return Err(ProtocolError::new(
            ProtocolErrorCode::ResourceExhausted,
            "runtime state process-route capacity is exhausted",
        ));
    }
    routes.instances.push_back(InstanceRoute {
        instance_id,
        adapter,
        live_states: 1,
    });
    Ok(())
}

fn release_instance_locked(
    routes: &mut RouteCache,
    instance_id: &str,
    adapter: &Arc<dyn BackendRuntimeAdapter>,
) -> Result<(), ProtocolError> {
    let index = routes
        .instances
        .iter()
        .position(|route| route.instance_id == instance_id)
        .ok_or_else(incompatible_route)?;
    if !Arc::ptr_eq(&routes.instances[index].adapter, adapter)
        || routes.instances[index].live_states == 0
    {
        return Err(incompatible_route());
    }
    routes.instances[index].live_states -= 1;
    if routes.instances[index].live_states == 0 {
        routes.instances.remove(index);
    }
    Ok(())
}

fn commit_state_route(
    reservation: StateRouteReservation<'_>,
    state: &BackendState,
    adapter: Arc<dyn BackendRuntimeAdapter>,
) -> Result<(), ProtocolError> {
    if let Err(error) = reservation.commit(state, adapter.clone()) {
        return if adapter.release(state).is_ok() {
            Err(error)
        } else {
            Err(ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend state ownership could not be recorded and cleanup could not be confirmed",
            )
            .with_backend_cleanup_unconfirmed(state.natural_tier(), state.bytes()))
        };
    }
    Ok(())
}

fn push_bounded<T>(routes: &mut VecDeque<T>, value: T, maximum: usize) {
    if routes.len() == maximum {
        routes.pop_front();
    }
    routes.push_back(value);
}

fn manifest_fingerprint(manifest: &ModelManifest) -> Result<[u8; 32], ProtocolError> {
    let mut writer = DigestWriter(Sha256::new());
    serde_json::to_writer(&mut writer, manifest).map_err(|_| {
        ProtocolError::new(
            ProtocolErrorCode::Internal,
            "runtime-control model routing could not identify the requested manifest",
        )
    })?;
    Ok(writer.0.finalize().into())
}

struct DigestWriter(Sha256);

impl Write for DigestWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn incompatible_route() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::IncompatibleState,
        "no resolved runtime adapter owns this opaque state",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{GenerateRequest, GenerateResponse, GenerateStream, GenerationTimings},
        model_store::{ModelFormat, ModelMetadata, ModelSource},
        runtime_control::{BackendDecodeOptions, BackendPersistedStateScope, BackendStateLease},
        werk_protocol::{
            Capability, CapabilityStatus, ContextCompatibility, ExpertAction, ExpertSummary,
            ExpertTier, PersistencePolicy, PrefillInput, ProtocolVersion,
        },
    };
    use std::sync::{
        Barrier,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    struct RecordingAdapter {
        name: &'static str,
        instance: &'static str,
        prefill_calls: AtomicUsize,
        decode_calls: AtomicUsize,
        resolution_calls: AtomicUsize,
        restore_requirement_calls: AtomicUsize,
        restore_calls: AtomicUsize,
        snapshot_calls: AtomicUsize,
        release_calls: AtomicUsize,
        fail_release: AtomicBool,
        resolution_barrier: Mutex<Option<Arc<Barrier>>>,
    }

    impl RecordingAdapter {
        fn new(name: &'static str, instance: &'static str) -> Self {
            Self {
                name,
                instance,
                prefill_calls: AtomicUsize::new(0),
                decode_calls: AtomicUsize::new(0),
                resolution_calls: AtomicUsize::new(0),
                restore_requirement_calls: AtomicUsize::new(0),
                restore_calls: AtomicUsize::new(0),
                snapshot_calls: AtomicUsize::new(0),
                release_calls: AtomicUsize::new(0),
                fail_release: AtomicBool::new(false),
                resolution_barrier: Mutex::new(None),
            }
        }

        fn block_resolution_on(&self, barrier: Arc<Barrier>) {
            *self.resolution_barrier.lock().unwrap() = Some(barrier);
        }

        fn envelope(
            &self,
            manifest: &ModelManifest,
            prompt_fingerprint: &str,
        ) -> CompatibilityEnvelope {
            CompatibilityEnvelope {
                model_fingerprint: format!("model:{}", manifest.id),
                tokenizer_fingerprint: format!("tokenizer:{}", manifest.id),
                prompt_fingerprint: prompt_fingerprint.to_string(),
                chat_template_fingerprint: None,
                backend: self.name.to_string(),
                backend_version: "1".to_string(),
                runtime_adapter_version: "1".to_string(),
                accelerator_family: self.name.to_string(),
                tensor_dtype: "f16".to_string(),
                kv_dtype: "f16".to_string(),
                quantization: "none".to_string(),
                cache_layout: "test".to_string(),
                block_size: None,
                context: ContextCompatibility {
                    context_size: 4096,
                    batch_size: None,
                    rope_configuration_fingerprint: None,
                },
                multimodal_processor_fingerprints: Vec::new(),
                producer_protocol: ProtocolVersion::V1,
            }
        }
    }

    impl BackendRuntimeAdapter for RecordingAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            BackendRuntimeDescriptor {
                backend: self.name.to_string(),
                backend_version: "1".to_string(),
                adapter_version: "1".to_string(),
                accelerator_family: self.name.to_string(),
                instance_id: self.instance.to_string(),
                capabilities: vec![Capability {
                    id: format!("runtime.test.{}", self.name),
                    status: CapabilityStatus::Supported,
                    detail: "test adapter".to_string(),
                    operations: vec!["read".to_string()],
                }],
            }
        }

        fn compatibility(
            &self,
            manifest: &ModelManifest,
            prompt_fingerprint: &str,
        ) -> Result<CompatibilityEnvelope, ProtocolError> {
            Ok(self.envelope(manifest, prompt_fingerprint))
        }

        fn resolve_persisted_state(
            &self,
            manifest: &ModelManifest,
            _snapshot: &BackendSnapshot,
            expected: &CompatibilityEnvelope,
        ) -> Result<BackendPersistedStateResolution, ProtocolError> {
            self.resolution_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(barrier) = self.resolution_barrier.lock().unwrap().clone() {
                barrier.wait();
                barrier.wait();
            }
            Ok(BackendPersistedStateResolution {
                compatibility: self.envelope(manifest, &expected.prompt_fingerprint),
                scope: BackendPersistedStateScope::CrossRestart,
            })
        }

        fn restore_memory_requirement(
            &self,
            snapshot: &BackendSnapshot,
            _compatibility: &CompatibilityEnvelope,
        ) -> Result<Option<BackendMemoryRequirement>, ProtocolError> {
            self.restore_requirement_calls
                .fetch_add(1, Ordering::SeqCst);
            Ok(Some(BackendMemoryRequirement {
                tier: StateTier::Ram,
                bytes: snapshot.bytes,
                demotion_target: None,
            }))
        }

        fn prefill(
            &self,
            request: BackendPrefillRequest,
        ) -> Result<BackendPrefillResult, ProtocolError> {
            self.prefill_calls.fetch_add(1, Ordering::SeqCst);
            Ok(BackendPrefillResult {
                state: BackendState::InProcess {
                    handle: format!("handle:{}", request.model_id),
                    bytes: Some(64),
                    tier: StateTier::Ram,
                    instance_id: self.instance.to_string(),
                },
                prompt_tokens: 1,
                reused: false,
            })
        }

        fn decode(
            &self,
            _request: BackendDecodeRequest,
        ) -> Result<BackendDecodeResult, ProtocolError> {
            self.decode_calls.fetch_add(1, Ordering::SeqCst);
            Ok(BackendDecodeResult {
                text: self.name.to_string(),
                state: None,
                completion_tokens: 1,
                finish_reason: "stop".to_string(),
            })
        }

        fn restore(
            &self,
            snapshot: BackendSnapshot,
            _compatibility: &CompatibilityEnvelope,
        ) -> Result<BackendState, ProtocolError> {
            self.restore_calls.fetch_add(1, Ordering::SeqCst);
            Ok(BackendState::InProcess {
                handle: format!("restored:{}", self.name),
                bytes: Some(snapshot.bytes),
                tier: StateTier::Ram,
                instance_id: self.instance.to_string(),
            })
        }

        fn snapshot(&self, state: &BackendState) -> Result<BackendState, ProtocolError> {
            self.snapshot_calls.fetch_add(1, Ordering::SeqCst);
            Ok(state.clone())
        }

        fn release(&self, _state: &BackendState) -> Result<(), ProtocolError> {
            self.release_calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_release.load(Ordering::SeqCst) {
                Err(ProtocolError::new(
                    ProtocolErrorCode::Unavailable,
                    "deterministic release failure",
                ))
            } else {
                Ok(())
            }
        }
    }

    struct ModelRoutingBackend {
        first: Arc<RecordingAdapter>,
        second: Arc<RecordingAdapter>,
        resolutions: AtomicUsize,
    }

    struct DirectBackend {
        adapter: Arc<RecordingAdapter>,
    }

    struct ExpertRaceAdapter {
        marker: &'static str,
        list_calls: AtomicUsize,
        action_calls: AtomicUsize,
    }

    impl ExpertRaceAdapter {
        fn new(marker: &'static str) -> Self {
            Self {
                marker,
                list_calls: AtomicUsize::new(0),
                action_calls: AtomicUsize::new(0),
            }
        }

        fn expert(&self, model_id: &str) -> ExpertSummary {
            ExpertSummary {
                id: format!("expert-{}", self.marker),
                model_id: model_id.to_string(),
                tier: ExpertTier::Ram,
                bytes: Some(1),
                hotness: 1.0,
                pinned: false,
                last_used_unix_ms: None,
            }
        }
    }

    impl BackendRuntimeAdapter for ExpertRaceAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            BackendRuntimeDescriptor {
                // Both objects intentionally claim the exact same public
                // process identity and capabilities. Pointer identity is the
                // only distinction available to the operation plan.
                backend: "expert-race".to_string(),
                backend_version: "1".to_string(),
                adapter_version: "1".to_string(),
                accelerator_family: "cpu".to_string(),
                instance_id: "shared-instance".to_string(),
                capabilities: vec![Capability {
                    id: "runtime.experts.residency".to_string(),
                    status: CapabilityStatus::Supported,
                    detail: "deterministic expert test".to_string(),
                    operations: vec!["read".to_string(), "control".to_string()],
                }],
            }
        }

        fn list_experts(
            &self,
            filter: &ExpertListFilter,
        ) -> Result<ExpertListResponse, ProtocolError> {
            self.list_calls.fetch_add(1, Ordering::SeqCst);
            Ok(ExpertListResponse {
                experts: vec![self.expert(filter.model_id.as_deref().unwrap_or("active"))],
                next_cursor: None,
            })
        }

        fn expert_action(
            &self,
            request: &ExpertActionRequest,
        ) -> Result<ExpertActionResponse, ProtocolError> {
            self.action_calls.fetch_add(1, Ordering::SeqCst);
            Ok(ExpertActionResponse {
                experts: vec![self.expert(&request.model_id)],
                changed: u64::from(!request.dry_run),
                dry_run: request.dry_run,
            })
        }
    }

    struct ExpertRaceBackend {
        first: Arc<ExpertRaceAdapter>,
        second: Arc<ExpertRaceAdapter>,
    }

    impl GenerationBackend for ExpertRaceBackend {
        fn runtime_control_adapter_for(
            &self,
            manifest: &ModelManifest,
        ) -> anyhow::Result<Arc<dyn BackendRuntimeAdapter>> {
            if manifest.architecture.as_deref() == Some("route-b") {
                Ok(self.second.clone())
            } else {
                Ok(self.first.clone())
            }
        }

        fn generate(
            &self,
            _manifest: &ModelManifest,
            _request: GenerateRequest,
        ) -> anyhow::Result<GenerateResponse> {
            anyhow::bail!("not used")
        }

        fn generate_stream(
            &self,
            _manifest: ModelManifest,
            _request: GenerateRequest,
        ) -> GenerateStream {
            Box::pin(tokio_stream::empty())
        }
    }

    impl GenerationBackend for DirectBackend {
        fn runtime_control_adapter(&self) -> Arc<dyn BackendRuntimeAdapter> {
            self.adapter.clone()
        }

        fn generate(
            &self,
            _manifest: &ModelManifest,
            _request: GenerateRequest,
        ) -> anyhow::Result<GenerateResponse> {
            anyhow::bail!("not used")
        }

        fn generate_stream(
            &self,
            _manifest: ModelManifest,
            _request: GenerateRequest,
        ) -> GenerateStream {
            Box::pin(tokio_stream::empty())
        }
    }

    impl GenerationBackend for ModelRoutingBackend {
        fn runtime_control_adapter_for(
            &self,
            manifest: &ModelManifest,
        ) -> anyhow::Result<Arc<dyn BackendRuntimeAdapter>> {
            self.resolutions.fetch_add(1, Ordering::SeqCst);
            if manifest.id == "second" {
                Ok(self.second.clone())
            } else {
                Ok(self.first.clone())
            }
        }

        fn runtime_control_adapter_for_request(
            &self,
            manifest: &ModelManifest,
            has_images: bool,
        ) -> anyhow::Result<Arc<dyn BackendRuntimeAdapter>> {
            self.resolutions.fetch_add(1, Ordering::SeqCst);
            if has_images || manifest.id == "second" {
                Ok(self.second.clone())
            } else {
                Ok(self.first.clone())
            }
        }

        fn generate(
            &self,
            _manifest: &ModelManifest,
            _request: GenerateRequest,
        ) -> anyhow::Result<GenerateResponse> {
            Ok(GenerateResponse {
                text: "generated".to_string(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: "stop".to_string(),
                timings: GenerationTimings::default(),
                backend_diagnostics: Vec::new(),
            })
        }

        fn generate_stream(
            &self,
            _manifest: ModelManifest,
            _request: GenerateRequest,
        ) -> GenerateStream {
            Box::pin(tokio_stream::empty())
        }
    }

    fn manifest(id: &str) -> ModelManifest {
        ModelManifest {
            id: id.to_string(),
            source: ModelSource::LocalPath {
                path: "test".to_string(),
            },
            format: ModelFormat::Gguf,
            architecture: Some("test".to_string()),
            tokenizer_path: None,
            config_path: None,
            model_path: Some("model.gguf".to_string()),
            backend: "test".to_string(),
            created_unix: 1,
            files: Vec::new(),
            artifacts: Vec::new(),
            metadata: ModelMetadata::default(),
        }
    }

    fn generation_request(has_images: bool) -> GenerateRequest {
        GenerateRequest {
            prompt: "hello".to_string(),
            messages: Vec::new(),
            image_urls: has_images
                .then(|| "data:image/png;base64,AA==".to_string())
                .into_iter()
                .collect(),
            max_tokens: 1,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            seed: None,
            stream_granularity: crate::backend::StreamGranularity::Token,
            verbose: false,
            debug: false,
        }
    }

    #[test]
    fn model_router_reaches_concrete_adapters_and_routes_old_state_by_instance() {
        let first = Arc::new(RecordingAdapter::new("first", "instance-first"));
        let second = Arc::new(RecordingAdapter::new("second", "instance-second"));
        let backend = Arc::new(ModelRoutingBackend {
            first: first.clone(),
            second: second.clone(),
            resolutions: AtomicUsize::new(0),
        });
        let routed = Arc::new(RoutedRuntimeAdapter::new(backend.clone()));

        assert_eq!(routed.descriptor().backend, "generation-backend");
        let first_compatibility = routed
            .compatibility(&manifest("first"), "prompt-a")
            .unwrap();
        assert_eq!(routed.descriptor().backend, "first");
        assert_eq!(routed.descriptor().capabilities[0].id, "runtime.test.first");
        let first_state = routed
            .prefill(BackendPrefillRequest {
                model_id: "first".to_string(),
                input: PrefillInput::Text {
                    text: "hello".to_string(),
                },
                compatibility: first_compatibility.clone(),
                policy: PersistencePolicy::default(),
            })
            .unwrap()
            .state;

        routed
            .compatibility(&manifest("second"), "prompt-b")
            .unwrap();
        assert_eq!(routed.descriptor().backend, "second");
        assert_eq!(
            routed.descriptor().capabilities[0].id,
            "runtime.test.second"
        );

        let snapshot = routed.snapshot(&first_state).unwrap();
        assert_eq!(routed.descriptor().backend, "second");
        routed.release(&snapshot).unwrap();
        assert_eq!(first.prefill_calls.load(Ordering::SeqCst), 1);
        assert_eq!(first.snapshot_calls.load(Ordering::SeqCst), 1);
        assert_eq!(first.release_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second.prefill_calls.load(Ordering::SeqCst), 0);

        routed
            .compatibility(&manifest("second"), "prompt-for-active-switch")
            .unwrap();
        assert_eq!(
            routed
                .descriptor_for_model(&manifest("first"))
                .unwrap()
                .backend,
            "first"
        );
        assert_eq!(
            routed
                .descriptor_for_compatibility(&first_compatibility)
                .unwrap()
                .backend,
            "first"
        );
        assert_eq!(routed.descriptor().backend, "second");
        routed
            .validate_state(&first_state, &first_compatibility)
            .unwrap();
        assert_eq!(routed.descriptor().backend, "second");
        let decoded = routed
            .decode(BackendDecodeRequest {
                state: BackendStateLease::new(routed.clone(), first_state),
                compatibility: first_compatibility,
                options: BackendDecodeOptions {
                    max_tokens: 1,
                    temperature: None,
                    top_p: None,
                    seed: None,
                    stop: Vec::new(),
                },
            })
            .unwrap();
        assert_eq!(decoded.text, "first");
        assert_eq!(first.decode_calls.load(Ordering::SeqCst), 1);
        assert_eq!(routed.descriptor().backend, "second");

        routed
            .compatibility(&manifest("first"), "prompt-c")
            .unwrap();
        assert_eq!(backend.resolutions.load(Ordering::SeqCst), 2);
        assert_eq!(routed.descriptor().backend, "first");
    }

    #[test]
    fn cold_router_requires_manifest_aware_resolution_before_restore() {
        let first = Arc::new(RecordingAdapter::new("first", "instance-first"));
        let second = Arc::new(RecordingAdapter::new("second", "instance-second"));
        let backend = Arc::new(ModelRoutingBackend {
            first: first.clone(),
            second: second.clone(),
            resolutions: AtomicUsize::new(0),
        });
        let routed = RoutedRuntimeAdapter::new(backend.clone());
        let model = manifest("first");
        let expected = first.envelope(&model, "stored-prompt-fingerprint");
        let snapshot = test_snapshot(321);

        let unresolved = routed
            .restore_memory_requirement(&snapshot, &expected)
            .unwrap_err();
        assert_eq!(unresolved.code, ProtocolErrorCode::IncompatibleState);
        assert_eq!(routed.descriptor().backend, "generation-backend");

        let resolution = routed
            .resolve_persisted_state(&model, &snapshot, &expected)
            .unwrap();
        assert_eq!(resolution.compatibility, expected);
        assert_eq!(resolution.scope, BackendPersistedStateScope::CrossRestart);
        assert_eq!(routed.descriptor().backend, "first");
        assert_eq!(first.resolution_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second.resolution_calls.load(Ordering::SeqCst), 0);

        let requirement = routed
            .restore_memory_requirement(&snapshot, &expected)
            .unwrap()
            .unwrap();
        assert_eq!(requirement.bytes, 321);
        let restored = routed.restore(snapshot, &expected).unwrap();
        assert_eq!(restored.instance_id(), "instance-first");
        routed.release(&restored).unwrap();
        assert_eq!(first.restore_requirement_calls.load(Ordering::SeqCst), 1);
        assert_eq!(first.restore_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn prepared_restore_stays_bound_to_one_adapter_while_the_route_changes() {
        let first = Arc::new(RecordingAdapter::new("shared", "instance-first"));
        let second = Arc::new(RecordingAdapter::new("shared", "instance-second"));
        let routed = Arc::new(RoutedRuntimeAdapter::new(Arc::new(DirectBackend {
            adapter: first.clone(),
        })));
        let model = manifest("shared");
        let expected = first.envelope(&model, "stored-prompt-fingerprint");
        routed
            .record_compatibility(&model.id, &expected, first.clone())
            .unwrap();

        let barrier = Arc::new(Barrier::new(2));
        first.block_resolution_on(barrier.clone());
        let route_switch = {
            let routed = routed.clone();
            let second = second.clone();
            let expected = expected.clone();
            let model_id = model.id.clone();
            std::thread::spawn(move || {
                barrier.wait();
                routed
                    .record_compatibility(&model_id, &expected, second)
                    .unwrap();
                barrier.wait();
            })
        };

        let snapshot = test_snapshot(321);
        let plan = routed
            .prepare_persisted_state(&model, &snapshot, &expected)
            .unwrap();
        route_switch.join().unwrap();
        routed
            .record_compatibility(&model.id, &expected, second.clone())
            .unwrap();

        assert_eq!(plan.descriptor().instance_id, "instance-first");
        assert_eq!(first.resolution_calls.load(Ordering::SeqCst), 1);
        assert_eq!(first.restore_requirement_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second.resolution_calls.load(Ordering::SeqCst), 0);
        assert_eq!(second.restore_requirement_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            routed
                .descriptor_for_compatibility(&expected)
                .unwrap()
                .instance_id,
            "instance-second"
        );

        let restored = routed
            .restore_prepared_state(plan, snapshot, &expected)
            .unwrap();
        assert_eq!(restored.instance_id(), "instance-first");
        assert_eq!(first.restore_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second.restore_calls.load(Ordering::SeqCst), 0);
        routed.release(&restored).unwrap();
    }

    #[test]
    fn prepared_restore_rejects_a_different_snapshot_or_envelope() {
        let adapter = Arc::new(RecordingAdapter::new("direct", "instance-direct"));
        let routed = RoutedRuntimeAdapter::new(Arc::new(DirectBackend {
            adapter: adapter.clone(),
        }));
        let model = manifest("direct");
        let expected = routed.compatibility(&model, "prompt").unwrap();

        let first = test_snapshot(7);
        let first_plan = routed
            .prepare_persisted_state(&model, &first, &expected)
            .unwrap();
        let error = routed
            .restore_prepared_state(first_plan, test_snapshot(7), &expected)
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::IncompatibleState);

        let second = test_snapshot(7);
        let second_plan = routed
            .prepare_persisted_state(&model, &second, &expected)
            .unwrap();
        let mut foreign = expected.clone();
        foreign.prompt_fingerprint = "another-prompt".to_string();
        let error = routed
            .restore_prepared_state(second_plan, second, &foreign)
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::IncompatibleState);
        assert_eq!(adapter.restore_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn ordinary_generation_updates_the_shared_runtime_descriptor() {
        let first = Arc::new(RecordingAdapter::new("first", "instance-first"));
        let second = Arc::new(RecordingAdapter::new("second", "instance-second"));
        let backend = Arc::new(ModelRoutingBackend {
            first,
            second,
            resolutions: AtomicUsize::new(0),
        });
        let runtime = Arc::new(RoutedRuntimeAdapter::new(backend.clone()));
        let generation = RuntimeRoutedGenerationBackend::new(backend, runtime.clone());

        assert_eq!(runtime.descriptor().backend, "generation-backend");
        generation
            .generate(&manifest("first"), generation_request(false))
            .unwrap();
        assert_eq!(runtime.descriptor().backend, "first");
        generation
            .generate(&manifest("first"), generation_request(true))
            .unwrap();
        assert_eq!(runtime.descriptor().backend, "second");
    }

    #[test]
    fn cold_router_does_not_select_by_backend_label_and_reports_exact_mismatches() {
        let first = Arc::new(RecordingAdapter::new("first", "instance-first"));
        let second = Arc::new(RecordingAdapter::new("second", "instance-second"));
        let routed = RoutedRuntimeAdapter::new(Arc::new(ModelRoutingBackend {
            first: first.clone(),
            second: second.clone(),
            resolutions: AtomicUsize::new(0),
        }));
        let model = manifest("first");
        let mut expected = first.envelope(&model, "stored-prompt-fingerprint");
        expected.backend = "second".to_string();
        expected.kv_dtype = "bf16".to_string();
        let error = routed
            .resolve_persisted_state(&model, &test_snapshot(1), &expected)
            .unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::IncompatibleState);
        assert_eq!(
            error.details,
            Some(serde_json::json!({
                "mismatch_fields": ["backend", "kv_dtype"]
            }))
        );
        assert_eq!(first.resolution_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second.resolution_calls.load(Ordering::SeqCst), 0);
        assert_eq!(routed.descriptor().backend, "generation-backend");
    }

    #[test]
    fn live_instance_routes_are_never_evicted_and_existing_instances_work_at_capacity() {
        let adapter = Arc::new(RecordingAdapter::new("direct", "new-instance"));
        let routed = RoutedRuntimeAdapter::new(Arc::new(DirectBackend {
            adapter: adapter.clone(),
        }));
        let compatibility = routed.compatibility(&manifest("direct"), "prompt").unwrap();
        let mut live_states = Vec::with_capacity(MAX_INSTANCE_ROUTES);
        for index in 0..MAX_INSTANCE_ROUTES {
            let state = BackendState::InProcess {
                handle: format!("handle-{index}"),
                bytes: Some(1),
                tier: StateTier::Ram,
                instance_id: if index == 0 {
                    "new-instance".to_string()
                } else {
                    format!("instance-{index}")
                },
            };
            routed.record_state(&state, adapter.clone()).unwrap();
            live_states.push(state);
        }

        let other: Arc<dyn BackendRuntimeAdapter> =
            Arc::new(RecordingAdapter::new("other", "other-instance"));
        let error = match routed.reserve_state_route(&other) {
            Err(error) => error,
            Ok(_) => panic!("a new process route must not be admitted at capacity"),
        };
        assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);

        let admitted = routed
            .prefill(BackendPrefillRequest {
                model_id: "direct".to_string(),
                input: PrefillInput::Text {
                    text: "hello".to_string(),
                },
                compatibility: compatibility.clone(),
                policy: PersistencePolicy::default(),
            })
            .unwrap()
            .state;
        assert_eq!(adapter.prefill_calls.load(Ordering::SeqCst), 1);
        assert_eq!(adapter.release_calls.load(Ordering::SeqCst), 0);

        routed.release(&live_states[0]).unwrap();
        routed.release(&live_states[1]).unwrap();
        routed.release(&admitted).unwrap();
        assert_eq!(adapter.prefill_calls.load(Ordering::SeqCst), 1);
        assert_eq!(adapter.release_calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn failed_backend_release_preserves_the_owning_instance_route() {
        let adapter = Arc::new(RecordingAdapter::new("direct", "instance-direct"));
        let routed = RoutedRuntimeAdapter::new(Arc::new(DirectBackend {
            adapter: adapter.clone(),
        }));
        let compatibility = routed.compatibility(&manifest("direct"), "prompt").unwrap();
        let state = routed
            .prefill(BackendPrefillRequest {
                model_id: "direct".to_string(),
                input: PrefillInput::Text {
                    text: "hello".to_string(),
                },
                compatibility,
                policy: PersistencePolicy::default(),
            })
            .unwrap()
            .state;

        adapter.fail_release.store(true, Ordering::SeqCst);
        assert_eq!(
            routed.release(&state).unwrap_err().code,
            ProtocolErrorCode::Unavailable
        );
        assert!(routed.adapter_for_state(&state).is_ok());

        adapter.fail_release.store(false, Ordering::SeqCst);
        routed.release(&state).unwrap();
        assert!(routed.adapter_for_state(&state).is_err());
    }

    #[test]
    fn direct_backend_keeps_its_existing_runtime_adapter_path() {
        let adapter = Arc::new(RecordingAdapter::new("direct", "instance-direct"));
        let routed = RoutedRuntimeAdapter::new(Arc::new(DirectBackend {
            adapter: adapter.clone(),
        }));

        assert_eq!(routed.descriptor().backend, "direct");
        let compatibility = routed.compatibility(&manifest("direct"), "prompt").unwrap();
        routed
            .prefill(BackendPrefillRequest {
                model_id: "direct".to_string(),
                input: PrefillInput::Text {
                    text: "hello".to_string(),
                },
                compatibility,
                policy: PersistencePolicy::default(),
            })
            .unwrap();

        assert_eq!(adapter.prefill_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn concurrent_model_routes_never_cross_backend_ownership() {
        let first = Arc::new(RecordingAdapter::new("first", "instance-first"));
        let second = Arc::new(RecordingAdapter::new("second", "instance-second"));
        let routed = Arc::new(RoutedRuntimeAdapter::new(Arc::new(ModelRoutingBackend {
            first: first.clone(),
            second: second.clone(),
            resolutions: AtomicUsize::new(0),
        })));

        let workers = (0..32)
            .map(|index| {
                let routed = routed.clone();
                std::thread::spawn(move || {
                    let model_id = if index % 2 == 0 { "first" } else { "second" };
                    let compatibility = routed
                        .compatibility(&manifest(model_id), &format!("prompt-{index}"))
                        .unwrap();
                    let result = routed
                        .prefill(BackendPrefillRequest {
                            model_id: model_id.to_string(),
                            input: PrefillInput::Text {
                                text: "hello".to_string(),
                            },
                            compatibility,
                            policy: PersistencePolicy::default(),
                        })
                        .unwrap();
                    assert_eq!(result.state.instance_id(), format!("instance-{model_id}"));
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }

        assert_eq!(first.prefill_calls.load(Ordering::SeqCst), 16);
        assert_eq!(second.prefill_calls.load(Ordering::SeqCst), 16);
    }

    #[test]
    fn prepared_expert_operations_never_switch_adapter_even_with_the_same_instance_id() {
        let first = Arc::new(ExpertRaceAdapter::new("a"));
        let second = Arc::new(ExpertRaceAdapter::new("b"));
        let routed = Arc::new(RoutedRuntimeAdapter::new(Arc::new(ExpertRaceBackend {
            first: first.clone(),
            second: second.clone(),
        })));
        let mut first_manifest = manifest("shared-model");
        first_manifest.architecture = Some("route-a".to_string());
        first_manifest.source = ModelSource::LocalPath {
            path: "private-route-a-path".to_string(),
        };
        let mut second_manifest = first_manifest.clone();
        second_manifest.architecture = Some("route-b".to_string());
        second_manifest.source = ModelSource::LocalPath {
            path: "private-route-b-path".to_string(),
        };
        let filter = ExpertListFilter {
            model_id: Some("shared-model".to_string()),
            ..ExpertListFilter::default()
        };
        let request = ExpertActionRequest {
            model_id: "shared-model".to_string(),
            expert_ids: vec!["expert-a".to_string()],
            action: ExpertAction::Pin,
            target_tier: None,
            dry_run: false,
            allow_experimental: false,
        };

        let list_plan = routed
            .prepare_expert_list(Some(&first_manifest), &filter)
            .unwrap();
        let action_plan = routed
            .prepare_expert_action(&first_manifest, &request)
            .unwrap();
        let debug = format!("{list_plan:?} {action_plan:?}");
        for secret in ["shared-model", "expert-a", "private-route-a-path"] {
            assert!(!debug.contains(secret));
        }

        // Both workers stop after capability/descriptor preparation. The main
        // thread then installs a newer route for the same model ID. Both
        // adapters deliberately expose the same instance ID and descriptor,
        // so an ID-only check would route these operations to B.
        let phase = Arc::new(Barrier::new(3));
        let list_worker = {
            let routed = routed.clone();
            let phase = phase.clone();
            std::thread::spawn(move || {
                phase.wait();
                phase.wait();
                routed.list_experts_prepared(list_plan)
            })
        };
        let action_worker = {
            let routed = routed.clone();
            let phase = phase.clone();
            std::thread::spawn(move || {
                phase.wait();
                phase.wait();
                routed.expert_action_prepared(action_plan)
            })
        };

        phase.wait();
        routed.descriptor_for_model(&second_manifest).unwrap();
        let latest_route = routed.known_model_adapter("shared-model").unwrap();
        let second_adapter: Arc<dyn BackendRuntimeAdapter> = second.clone();
        assert!(Arc::ptr_eq(&latest_route, &second_adapter));
        phase.wait();

        match list_worker.join().unwrap() {
            Ok(response) => assert_eq!(response.experts[0].id, "expert-a"),
            Err(error) => {
                assert_eq!(error.code, ProtocolErrorCode::Unavailable);
                assert!(error.retryable);
            }
        }
        match action_worker.join().unwrap() {
            Ok(response) => assert_eq!(response.experts[0].id, "expert-a"),
            Err(error) => {
                assert_eq!(error.code, ProtocolErrorCode::Unavailable);
                assert!(error.retryable);
            }
        }
        assert_eq!(second.list_calls.load(Ordering::SeqCst), 0);
        assert_eq!(second.action_calls.load(Ordering::SeqCst), 0);
        assert!(first.list_calls.load(Ordering::SeqCst) <= 1);
        assert!(first.action_calls.load(Ordering::SeqCst) <= 1);
    }

    fn test_snapshot(bytes: u64) -> BackendSnapshot {
        let file = std::fs::File::open(std::env::current_exe().expect("test executable path"))
            .expect("open test executable");
        BackendSnapshot::from_verified_file(file, bytes)
    }
}
