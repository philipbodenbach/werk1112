use super::{
    BackendDecodeOptions, BackendDecodeRequest, BackendMemoryRequirement,
    BackendPersistedStatePlan, BackendPersistedStateScope, BackendPrefillRequest,
    BackendRuntimeAdapter, BackendRuntimeDescriptor, BackendSnapshot, BackendState,
    BackendStateLease, PrincipalDeriver,
    experts::{valid_opaque_id, validate_expert_action},
    handoff::{HandoffRecord, HandoffRegistry, HandoffReservation, now_unix_ms},
    memory::{
        AllocationId, MemoryError, MemoryManager, MemoryManagerConfig, MemoryObservation,
        MemoryReservation, MemoryTelemetry, MemoryTier, MemoryTopology, PressureAction,
        PressureThresholds, SystemMemoryClock, TierBudget,
    },
    store::{
        LoadedStoredState, NewStoredState, OpaquePayloadSource, StateStore,
        validate_filter as validate_state_list_filter, validate_principal_id,
    },
    validate_compatibility, validate_compatibility_envelope, validate_runtime_descriptor,
};
use crate::{
    inference::MemoryTopology as InferenceMemoryTopology,
    inference_service::detect_host_resources,
    model_store::{ModelManifest, ModelStore},
    werk_protocol::{
        BoxControlFuture, CapabilitiesResponse, Capability, CapabilityStatus, ControlContext,
        DecodeRequest, DecodeResponse, ExpertActionRequest, ExpertActionResponse, ExpertListFilter,
        ExpertListResponse, ExpertSummary, MemoryStatusResponse, MemoryTierStatus, PersistenceMode,
        PrefillInput, PrefillRequest, PrefillResponse, PressureLevel, ProtocolError,
        ProtocolErrorCode, ProtocolLimits, ProtocolResult, ProtocolVersion, PruneStatesRequest,
        PruneStatesResponse, ReuseMode, RuntimeInfo, StateAction, StateActionRequest,
        StateActionResponse, StateListFilter, StateListResponse, StateSelector, StateStatus,
        StateSummary, StateTier, WerkControl,
    },
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::Read,
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::Semaphore;

const MAX_BLOCKING_OPERATIONS: usize = 8;
const MAX_VOLATILE_STATES: usize = 1024;
const MAX_VOLATILE_STATES_PER_PRINCIPAL: usize = 128;
const HANDOFF_TTL_MILLIS: u64 = 15 * 60 * 1000;
const MAX_PREFILL_BYTES: usize = 512 * 1024;
const MAX_MESSAGES: usize = 256;
const MAX_DECODE_TOKENS: u64 = 32 * 1024;
// Leave enough headroom beneath the 8-MiB transport response bound for JSON
// escaping, the protocol envelope and metadata (one byte can expand to six
// bytes as a JSON escape).
const MAX_DECODE_TEXT_BYTES: usize = 1024 * 1024;
const MEMORY_ACTION_COOLDOWN_MILLIS: u64 = 5_000;
const MAX_MEMORY_ALLOCATIONS: usize = 1_024;
const MAX_MEMORY_RESERVATIONS: usize = 1_024;
const MAX_MEMORY_ACTIONS_PER_CYCLE: usize = 128;

#[derive(Clone)]
pub struct LocalWerkControl {
    inner: Arc<LocalInner>,
}

struct LocalInner {
    store: ModelStore,
    state_store: StateStore,
    principal_deriver: PrincipalDeriver,
    adapter: Arc<dyn BackendRuntimeAdapter>,
    handoffs: HandoffRegistry,
    volatile_states: Mutex<HashMap<(String, String), VolatileState>>,
    operation_gate: Arc<Semaphore>,
    memory: Option<MemoryManager>,
    next_allocation_id: AtomicU64,
    backend_cleanup_latched: Arc<AtomicBool>,
    backend_cleanup_failures: Arc<AtomicU64>,
    blocking: Arc<Semaphore>,
    limits: ProtocolLimits,
}

#[derive(Clone)]
struct VolatileState {
    principal_id: String,
    summary: StateSummary,
    compatibility: crate::werk_protocol::CompatibilityEnvelope,
    prompt_tokens: u64,
    lease: BackendStateLease,
    persisted: bool,
    allocation_id: Option<AllocationId>,
}

struct PendingMemoryLoad {
    allocation_id: AllocationId,
    tier: MemoryTier,
    bytes: u64,
    reservation: MemoryReservation,
}

impl LocalWerkControl {
    pub fn new(store: ModelStore, adapter: Arc<dyn BackendRuntimeAdapter>) -> Self {
        let memory = build_system_memory_manager();
        Self::with_memory_manager(store, adapter, memory)
    }

    fn with_memory_manager(
        store: ModelStore,
        adapter: Arc<dyn BackendRuntimeAdapter>,
        memory: Option<MemoryManager>,
    ) -> Self {
        let limits = ProtocolLimits::default();
        Self {
            inner: Arc::new(LocalInner {
                state_store: StateStore::new(store.home()),
                principal_deriver: PrincipalDeriver::new(&store),
                store,
                adapter,
                handoffs: HandoffRegistry::default(),
                volatile_states: Mutex::new(HashMap::new()),
                operation_gate: Arc::new(Semaphore::new(1)),
                memory,
                next_allocation_id: AtomicU64::new(1),
                backend_cleanup_latched: Arc::new(AtomicBool::new(false)),
                backend_cleanup_failures: Arc::new(AtomicU64::new(0)),
                blocking: Arc::new(Semaphore::new(MAX_BLOCKING_OPERATIONS)),
                limits,
            }),
        }
    }

    #[cfg(test)]
    fn new_with_memory_manager(
        store: ModelStore,
        adapter: Arc<dyn BackendRuntimeAdapter>,
        memory: MemoryManager,
    ) -> Self {
        Self::with_memory_manager(store, adapter, Some(memory))
    }

    pub(crate) fn principal_deriver(&self) -> PrincipalDeriver {
        self.inner.principal_deriver.clone()
    }
}

impl WerkControl for LocalWerkControl {
    fn info(&self, context: ControlContext) -> BoxControlFuture<'_, RuntimeInfo> {
        let inner = self.inner.clone();
        Box::pin(run_blocking(inner, move |inner| {
            validate_principal_id(context.principal_id())?;
            let descriptor = inner.adapter.descriptor();
            validate_runtime_descriptor(&descriptor)?;
            Ok(RuntimeInfo {
                service: "werk1112".to_string(),
                service_version: env!("CARGO_PKG_VERSION").to_string(),
                protocol: ProtocolVersion::V1,
                active_backend: descriptor.backend,
                limits: inner.limits.clone(),
            })
        }))
    }

    fn capabilities(&self, context: ControlContext) -> BoxControlFuture<'_, CapabilitiesResponse> {
        let inner = self.inner.clone();
        Box::pin(run_blocking(inner, move |inner| {
            validate_principal_id(context.principal_id())?;
            Ok(CapabilitiesResponse {
                capabilities: capabilities_for(&inner)?,
            })
        }))
    }

    fn list_states(
        &self,
        context: ControlContext,
        filter: StateListFilter,
    ) -> BoxControlFuture<'_, StateListResponse> {
        let inner = self.inner.clone();
        Box::pin(run_blocking(inner, move |inner| {
            validate_principal_id(context.principal_id())?;
            list_states_inner(&inner, context.principal_id(), &filter)
        }))
    }

    fn state_action(
        &self,
        context: ControlContext,
        state_id: String,
        request: StateActionRequest,
    ) -> BoxControlFuture<'_, StateActionResponse> {
        let inner = self.inner.clone();
        Box::pin(run_mutating(inner, move |inner| {
            validate_principal_id(context.principal_id())?;
            state_action_inner(&inner, context.principal_id(), &state_id, &request)
        }))
    }

    fn prune_states(
        &self,
        context: ControlContext,
        request: PruneStatesRequest,
    ) -> BoxControlFuture<'_, PruneStatesResponse> {
        let inner = self.inner.clone();
        Box::pin(run_mutating(inner, move |inner| {
            validate_principal_id(context.principal_id())?;
            prune_states_inner(&inner, context.principal_id(), &request)
        }))
    }

    fn memory_status(&self, context: ControlContext) -> BoxControlFuture<'_, MemoryStatusResponse> {
        let inner = self.inner.clone();
        Box::pin(run_blocking(inner, move |inner| {
            validate_principal_id(context.principal_id())?;
            Ok(memory_status_for(&inner))
        }))
    }

    fn list_experts(
        &self,
        context: ControlContext,
        filter: ExpertListFilter,
    ) -> BoxControlFuture<'_, ExpertListResponse> {
        let inner = self.inner.clone();
        Box::pin(run_blocking(inner, move |inner| {
            validate_principal_id(context.principal_id())?;
            validate_expert_filter(&filter, inner.limits.max_page_size)?;
            let manifest = match filter.model_id.as_deref() {
                Some(model_id) => Some(manifest_for_existing_model(&inner, model_id)?),
                None => None,
            };
            let plan = inner
                .adapter
                .prepare_expert_list(manifest.as_ref(), &filter)?;
            let response = inner.adapter.list_experts_prepared(plan)?;
            validate_expert_list_response(&response, &filter, inner.limits.max_page_size)?;
            Ok(response)
        }))
    }

    fn expert_action(
        &self,
        context: ControlContext,
        request: ExpertActionRequest,
    ) -> BoxControlFuture<'_, ExpertActionResponse> {
        let inner = self.inner.clone();
        Box::pin(run_mutating(inner, move |inner| {
            validate_principal_id(context.principal_id())?;
            if !request.dry_run {
                ensure_backend_cleanup_healthy(&inner)?;
            }
            validate_expert_action(
                &request,
                usize::from(inner.limits.max_expert_ids_per_operation),
            )?;
            let manifest = if request.dry_run {
                manifest_for_existing_model(&inner, &request.model_id)?
            } else {
                manifest_for_model(&inner, &request.model_id)?
            };
            let plan = inner.adapter.prepare_expert_action(&manifest, &request)?;
            let response = inner.adapter.expert_action_prepared(plan)?;
            validate_expert_action_response(&response, &request)?;
            Ok(response)
        }))
    }

    fn prefill(
        &self,
        context: ControlContext,
        request: PrefillRequest,
    ) -> BoxControlFuture<'_, PrefillResponse> {
        let inner = self.inner.clone();
        Box::pin(run_mutating(inner, move |inner| {
            validate_principal_id(context.principal_id())?;
            prefill_inner(&inner, context.principal_id(), request)
        }))
    }

    fn decode(
        &self,
        context: ControlContext,
        request: DecodeRequest,
    ) -> BoxControlFuture<'_, DecodeResponse> {
        let inner = self.inner.clone();
        Box::pin(run_mutating(inner, move |inner| {
            validate_principal_id(context.principal_id())?;
            decode_inner(&inner, context.principal_id(), request)
        }))
    }
}

async fn run_blocking<T, F>(inner: Arc<LocalInner>, operation: F) -> ProtocolResult<T>
where
    T: Send + 'static,
    F: FnOnce(Arc<LocalInner>) -> ProtocolResult<T> + Send + 'static,
{
    let permit = inner
        .blocking
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| unavailable("runtime control is shutting down"))?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        operation(inner).map_err(bound_protocol_error)
    })
    .await
    .map_err(|_| internal())?
}

async fn run_mutating<T, F>(inner: Arc<LocalInner>, operation: F) -> ProtocolResult<T>
where
    T: Send + 'static,
    F: FnOnce(Arc<LocalInner>) -> ProtocolResult<T> + Send + 'static,
{
    let mutation_permit = inner
        .operation_gate
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| unavailable("runtime control is shutting down"))?;
    let blocking_permit = inner
        .blocking
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| unavailable("runtime control is shutting down"))?;
    tokio::task::spawn_blocking(move || {
        // Both permits live in the blocking task. If the requesting future is
        // cancelled, Tokio lets that task finish and mutations remain serialized.
        let _mutation_permit = mutation_permit;
        let _blocking_permit = blocking_permit;
        operation(inner).map_err(bound_protocol_error)
    })
    .await
    .map_err(|_| internal())?
}

fn bound_protocol_error(mut error: ProtocolError) -> ProtocolError {
    const MAX_ERROR_MESSAGE_BYTES: usize = 4 * 1024;
    const MAX_ERROR_DETAILS_BYTES: usize = 16 * 1024;
    if error.message.is_empty()
        || error.message.len() > MAX_ERROR_MESSAGE_BYTES
        || error.message.chars().any(char::is_control)
    {
        error.message = "runtime control operation failed".to_string();
    }
    if error.details.as_ref().is_some_and(|details| {
        serde_json::to_vec(details)
            .map(|encoded| encoded.len() > MAX_ERROR_DETAILS_BYTES)
            .unwrap_or(true)
    }) {
        error.details = None;
    }
    error
}

fn capabilities_for(inner: &LocalInner) -> ProtocolResult<Vec<Capability>> {
    capabilities_for_descriptor(inner, inner.adapter.descriptor())
}

fn capabilities_for_descriptor(
    inner: &LocalInner,
    descriptor: BackendRuntimeDescriptor,
) -> ProtocolResult<Vec<Capability>> {
    validate_runtime_descriptor(&descriptor)?;
    let mut capabilities = BTreeMap::<String, Capability>::new();
    let unsupported = [
        "runtime.model_residency",
        "runtime.state.prefix_cache",
        "runtime.state.persistence",
        "runtime.state.restore",
        "runtime.state.tier.vram",
        "runtime.state.tier.ram",
        "runtime.state.tier.disk",
        "runtime.experts.residency",
        "runtime.pd.prefill",
        "runtime.pd.decode",
        "runtime.pd.handoff",
    ];
    for id in unsupported {
        capabilities.insert(
            id.to_string(),
            Capability {
                id: id.to_string(),
                status: CapabilityStatus::Unsupported,
                detail: format!(
                    "{} does not expose this operation through its active Werk adapter",
                    descriptor.backend
                ),
                operations: Vec::new(),
            },
        );
    }
    for capability in descriptor.capabilities {
        capabilities.insert(capability.id.clone(), capability);
    }
    let memory = memory_status_for(inner);
    capabilities.insert(
        "runtime.memory.telemetry.host".to_string(),
        Capability {
            id: "runtime.memory.telemetry.host".to_string(),
            status: if memory.host.capacity_bytes.is_some() {
                CapabilityStatus::Supported
            } else {
                CapabilityStatus::Unavailable
            },
            detail: if memory.host.capacity_bytes.is_some() {
                "host memory is sampled dynamically".to_string()
            } else {
                "host memory telemetry is unavailable on this runtime".to_string()
            },
            operations: vec!["read".to_string()],
        },
    );
    capabilities.insert(
        "runtime.memory.telemetry.accelerator".to_string(),
        Capability {
            id: "runtime.memory.telemetry.accelerator".to_string(),
            status: if memory.accelerator.capacity_bytes.is_some()
                && memory.accelerator.available_bytes.is_some()
            {
                CapabilityStatus::Supported
            } else {
                CapabilityStatus::Unavailable
            },
            detail: if memory.accelerator.capacity_bytes.is_some()
                && memory.accelerator.available_bytes.is_some()
            {
                "accelerator memory is sampled dynamically".to_string()
            } else {
                "accelerator free-memory telemetry is unavailable".to_string()
            },
            operations: vec!["read".to_string()],
        },
    );
    let backend_reservations = capabilities.get("runtime.memory.reservations").cloned();
    let reservations = match (inner.memory.is_some(), backend_reservations) {
        (true, Some(capability))
            if matches!(
                capability.status,
                CapabilityStatus::Supported | CapabilityStatus::Experimental
            ) =>
        {
            capability
        }
        (false, Some(_)) => Capability {
            id: "runtime.memory.reservations".to_string(),
            status: CapabilityStatus::Unavailable,
            detail: "dynamic memory telemetry is unavailable, so reservations cannot be enforced"
                .to_string(),
            operations: Vec::new(),
        },
        _ => Capability {
            id: "runtime.memory.reservations".to_string(),
            status: CapabilityStatus::Unavailable,
            detail: "the active backend has not supplied bounded pre-load memory estimates"
                .to_string(),
            operations: Vec::new(),
        },
    };
    capabilities.insert("runtime.memory.reservations".to_string(), reservations);
    Ok(capabilities.into_values().collect())
}

fn require_backend_capability(
    descriptor: &BackendRuntimeDescriptor,
    id: &str,
    allow_experimental: bool,
    require_control: bool,
) -> ProtocolResult<()> {
    validate_runtime_descriptor(descriptor)?;
    let capability = descriptor
        .capabilities
        .iter()
        .find(|capability| capability.id == id)
        .ok_or_else(|| {
            unsupported(format!(
                "{} did not declare the requested capability",
                descriptor.backend
            ))
        })?;
    require_capability_status(capability, id, allow_experimental, require_control)
}

fn manifest_for_model(inner: &LocalInner, model_id: &str) -> ProtocolResult<ModelManifest> {
    inner
        .store
        .get(model_id)
        .map_err(|_| ProtocolError::new(ProtocolErrorCode::NotFound, "model is not installed"))
}

fn manifest_for_existing_model(
    inner: &LocalInner,
    model_id: &str,
) -> ProtocolResult<ModelManifest> {
    inner
        .store
        .get_existing(model_id)
        .map_err(|_| ProtocolError::new(ProtocolErrorCode::NotFound, "model is not installed"))
}

fn require_capability_status(
    capability: &Capability,
    id: &str,
    allow_experimental: bool,
    require_control: bool,
) -> ProtocolResult<()> {
    match capability.status {
        CapabilityStatus::Supported => Ok(()),
        CapabilityStatus::Experimental if allow_experimental => Ok(()),
        CapabilityStatus::Experimental => Err(ProtocolError::new(
            ProtocolErrorCode::ExperimentalOptInRequired,
            format!("{id} is experimental and requires explicit opt-in"),
        )),
        CapabilityStatus::Unavailable => Err(unavailable(capability.detail.clone())),
        CapabilityStatus::ExternallyManaged if !require_control => Ok(()),
        CapabilityStatus::ExternallyManaged => Err(unsupported(format!(
            "{id} is managed externally and cannot be controlled through this adapter"
        ))),
        CapabilityStatus::MetadataOnly => Err(unsupported(format!(
            "{id} exposes metadata only; the requested operation is unavailable"
        ))),
        CapabilityStatus::Unsupported => Err(unsupported(capability.detail.clone())),
    }
}

fn prepare_persisted_snapshot(
    inner: &LocalInner,
    stored: &LoadedStoredState,
    allow_experimental: bool,
) -> ProtocolResult<(BackendSnapshot, BackendPersistedStatePlan)> {
    let manifest = inner.store.get(&stored.summary.model_id).map_err(|_| {
        unavailable("the model required by this persisted runtime state is not installed")
    })?;
    let snapshot = BackendSnapshot::from_verified_file(
        stored.payload_file.try_clone().map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::CorruptState,
                "runtime state payload handle is unavailable",
            )
        })?,
        stored.payload_bytes,
    );
    let plan =
        inner
            .adapter
            .prepare_persisted_state(&manifest, &snapshot, &stored.compatibility)?;
    validate_compatibility_envelope(&plan.resolution().compatibility).map_err(|_| {
        ProtocolError::new(
            ProtocolErrorCode::Internal,
            "backend resolved an invalid or unbounded compatibility envelope",
        )
    })?;
    validate_compatibility(&stored.compatibility, &plan.resolution().compatibility)?;

    let descriptor = plan.descriptor();
    validate_runtime_descriptor(&descriptor)?;
    validate_descriptor_ownership(descriptor, &stored.compatibility)?;
    let require_declared = |id: &str| {
        let capability = descriptor
            .capabilities
            .iter()
            .find(|capability| capability.id == id)
            .ok_or_else(|| {
                unsupported("the resolved backend did not declare the requested capability")
            })?;
        require_capability_status(capability, id, allow_experimental, true)
    };
    require_declared("runtime.state.restore")?;
    if plan.resolution().scope == BackendPersistedStateScope::CrossRestart {
        require_declared("runtime.state.restore.cross_restart")?;
    }
    Ok((snapshot, plan))
}

fn prefill_inner(
    inner: &LocalInner,
    principal_id: &str,
    request: PrefillRequest,
) -> ProtocolResult<PrefillResponse> {
    validate_prefill(&request, &inner.limits)?;
    ensure_backend_cleanup_healthy(inner)?;
    // Admission is transactional: no backend state or persistent catalog
    // entry is created unless response capacity is already held.
    let handoff_reservation = inner.handoffs.reserve(principal_id)?;
    ensure_backend_cleanup_healthy(inner)?;
    let manifest = inner
        .store
        .get(&request.model_id)
        .map_err(|_| ProtocolError::new(ProtocolErrorCode::NotFound, "model is not installed"))?;
    // Some adapters (notably llama.cpp) perform their one-time functional
    // probe while constructing the compatibility envelope. Never let a
    // request without experimental opt-in trigger that backend work.
    let preflight_descriptor = inner.adapter.descriptor_for_model(&manifest)?;
    preflight_prefill_capability(
        &preflight_descriptor,
        "runtime.pd.prefill",
        request.allow_experimental,
    )?;
    preflight_prefill_capability(
        &preflight_descriptor,
        "runtime.pd.handoff",
        request.allow_experimental,
    )?;
    let prompt_bytes = serde_json::to_vec(&request.input).map_err(|_| invalid("invalid input"))?;
    let mut scoped_prompt = Vec::with_capacity(principal_id.len() + 1 + prompt_bytes.len());
    scoped_prompt.extend_from_slice(principal_id.as_bytes());
    scoped_prompt.push(0);
    scoped_prompt.extend_from_slice(&prompt_bytes);
    let prompt_fingerprint = inner
        .principal_deriver
        .fingerprint("prompt-cache", &scoped_prompt)
        .map_err(|_| unavailable("secure prompt fingerprinting is unavailable"))?;
    let compatibility = inner
        .adapter
        .compatibility(&manifest, &prompt_fingerprint)?;
    validate_compatibility_envelope(&compatibility).map_err(|_| {
        ProtocolError::new(
            ProtocolErrorCode::Internal,
            "backend returned an invalid or unbounded compatibility envelope",
        )
    })?;
    if compatibility.prompt_fingerprint != prompt_fingerprint {
        return Err(ProtocolError::new(
            ProtocolErrorCode::Internal,
            "backend compatibility did not preserve the scoped prompt fingerprint",
        ));
    }
    let descriptor = inner.adapter.descriptor_for_compatibility(&compatibility)?;
    validate_descriptor_ownership(&descriptor, &compatibility)?;
    require_backend_capability(
        &descriptor,
        "runtime.pd.prefill",
        request.allow_experimental,
        true,
    )?;
    require_backend_capability(
        &descriptor,
        "runtime.pd.handoff",
        request.allow_experimental,
        true,
    )?;
    if request.policy.mode == PersistenceMode::Memory {
        let tiers = capabilities_for_descriptor(inner, descriptor.clone())?;
        let has_memory_tier = tiers.into_iter().any(|capability| {
            matches!(
                capability.id.as_str(),
                "runtime.state.tier.ram" | "runtime.state.tier.vram"
            ) && (capability.status == CapabilityStatus::Supported
                || (request.allow_experimental
                    && capability.status == CapabilityStatus::Experimental))
        });
        if !has_memory_tier {
            return Err(unsupported(
                "the active backend cannot retain runtime state in RAM or VRAM",
            ));
        }
    }
    if request.policy.mode == PersistenceMode::Disk {
        require_backend_capability(
            &descriptor,
            "runtime.state.persistence",
            request.allow_experimental,
            true,
        )?;
        require_state_tier_capability(&descriptor, StateTier::Disk, request.allow_experimental)?;
    }

    if request.policy.reuse != ReuseMode::Disabled {
        let volatile_reuse =
            find_volatile_reuse(inner, principal_id, &request.model_id, &compatibility)?;
        ensure_backend_cleanup_healthy(inner)?;
        if let Some(mut reused) = volatile_reuse {
            let reused_descriptor = inner.adapter.descriptor_for_state(reused.lease.state())?;
            require_policy_state_tier(
                request.policy.mode,
                reused.lease.state(),
                &reused_descriptor,
                request.allow_experimental,
            )?;
            if !reused.persisted
                && matches!(
                    request.policy.mode,
                    PersistenceMode::Disk | PersistenceMode::Auto
                )
            {
                let persistence = require_backend_capability(
                    &reused_descriptor,
                    "runtime.state.persistence",
                    request.allow_experimental,
                    true,
                )
                .and_then(|()| {
                    require_state_tier_capability(
                        &reused_descriptor,
                        StateTier::Disk,
                        request.allow_experimental,
                    )
                });
                match persistence {
                    Ok(()) => match persist_reused_volatile_state(inner, principal_id, &reused) {
                        Ok(()) => reused.persisted = true,
                        Err(_)
                            if request.policy.mode == PersistenceMode::Auto
                                && !inner.backend_cleanup_latched.load(Ordering::Acquire) => {}
                        Err(error) => return Err(error),
                    },
                    Err(_) if request.policy.mode == PersistenceMode::Auto => {}
                    Err(error) => return Err(error),
                }
            }
            return issue_handoff(
                handoff_reservation,
                principal_id,
                &request.model_id,
                reused.summary.id.into(),
                reused.prompt_tokens,
                true,
                reused.lease,
                compatibility,
            );
        }
        let stored = inner.state_store.find_compatible(
            principal_id,
            &request.model_id,
            &compatibility,
            request.policy.reuse == ReuseMode::Required,
        )?;
        if let Some(stored) = stored {
            let (snapshot, plan) =
                prepare_persisted_snapshot(inner, &stored, request.allow_experimental)?;
            let restore_descriptor = plan.descriptor().clone();
            let requirement = plan.memory_requirement();
            require_policy_requirement_tier(
                request.policy.mode,
                requirement.as_ref(),
                &restore_descriptor,
                request.allow_experimental,
            )?;
            let pending = reserve_backend_memory(inner, requirement, true)?;
            let state = inner
                .adapter
                .restore_prepared_state(plan, snapshot, &stored.compatibility)
                .map_err(|error| backend_operation_error(inner, error))?;
            if let Err(error) = require_policy_state_tier(
                request.policy.mode,
                &state,
                &restore_descriptor,
                request.allow_experimental,
            ) {
                return Err(cleanup_rejected_backend_state(
                    inner,
                    &state,
                    pending.as_ref(),
                    error,
                ));
            }
            let (lease, allocation_id) =
                commit_backend_memory(inner, state, pending, &stored.compatibility)?;
            let mut summary = stored.summary;
            summary.tier = lease.state().natural_tier();
            summary.bytes = lease.state().bytes();
            summary.last_accessed_unix_ms = now_unix_ms();
            if let (Some(memory), Some(allocation_id)) = (&inner.memory, allocation_id) {
                memory
                    .set_pinned(allocation_id, summary.pinned)
                    .map_err(memory_error)?;
            }
            let state_id = summary.id.clone();
            insert_volatile(
                inner,
                VolatileState {
                    principal_id: principal_id.to_string(),
                    summary,
                    compatibility: stored.compatibility,
                    prompt_tokens: stored.prompt_tokens,
                    lease: lease.clone(),
                    persisted: true,
                    allocation_id,
                },
            )?;
            return issue_handoff(
                handoff_reservation,
                principal_id,
                &request.model_id,
                Some(state_id),
                stored.prompt_tokens,
                true,
                lease,
                compatibility,
            );
        }
        if request.policy.reuse == ReuseMode::Required {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Conflict,
                "reuse was required, but no compatible runtime state exists",
            ));
        }
    }

    let backend_request = BackendPrefillRequest {
        model_id: request.model_id.clone(),
        input: request.input,
        compatibility: compatibility.clone(),
        policy: request.policy.clone(),
    };
    let requirement = inner.adapter.prefill_memory_requirement(&backend_request)?;
    require_policy_requirement_tier(
        request.policy.mode,
        requirement.as_ref(),
        &descriptor,
        request.allow_experimental,
    )?;
    let pending = reserve_backend_memory(inner, requirement, true)?;
    let result = inner
        .adapter
        .prefill(backend_request)
        .map_err(|error| backend_operation_error(inner, error))?;
    if let Err(error) = require_policy_state_tier(
        request.policy.mode,
        &result.state,
        &descriptor,
        request.allow_experimental,
    ) {
        return Err(cleanup_rejected_backend_state(
            inner,
            &result.state,
            pending.as_ref(),
            error,
        ));
    }
    let (lease, allocation_id) =
        commit_backend_memory(inner, result.state, pending, &compatibility)?;
    let state_id = retain_prefill_state(
        inner,
        principal_id,
        &request.model_id,
        &request.policy,
        result.prompt_tokens,
        &compatibility,
        lease.clone(),
        allocation_id,
        request.allow_experimental,
    )?;
    issue_handoff(
        handoff_reservation,
        principal_id,
        &request.model_id,
        state_id,
        result.prompt_tokens,
        result.reused,
        lease,
        compatibility,
    )
}

fn decode_inner(
    inner: &LocalInner,
    principal_id: &str,
    request: DecodeRequest,
) -> ProtocolResult<DecodeResponse> {
    validate_decode(&request)?;
    ensure_backend_cleanup_healthy(inner)?;
    let preview = inner.handoffs.inspect(principal_id, &request.handoff)?;
    ensure_backend_cleanup_healthy(inner)?;
    inner
        .adapter
        .validate_state(preview.state.state(), &preview.compatibility)?;
    let descriptor = inner.adapter.descriptor_for_state(preview.state.state())?;
    require_backend_capability(
        &descriptor,
        "runtime.pd.decode",
        request.allow_experimental,
        true,
    )?;
    drop(preview);
    let (record, replacement_handoff) = inner
        .handoffs
        .take_with_replacement(principal_id, &request.handoff)?;
    ensure_backend_cleanup_healthy(inner)?;
    inner
        .adapter
        .validate_state(record.state.state(), &record.compatibility)?;
    touch_handoff_state(inner, principal_id, &record)?;
    let backend_request = BackendDecodeRequest {
        state: record.state.clone(),
        compatibility: record.compatibility.clone(),
        options: BackendDecodeOptions {
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
            stop: request.stop,
        },
    };
    let replacement_requirement = inner.adapter.decode_memory_requirement(&backend_request)?;
    let pending_replacement = reserve_backend_memory(inner, replacement_requirement, true)?;
    let result = inner
        .adapter
        .decode(backend_request)
        .map_err(|error| backend_operation_error(inner, error))?;
    if result.text.len() > MAX_DECODE_TEXT_BYTES
        || result.completion_tokens > request.max_tokens
        || result.finish_reason.is_empty()
        || result.finish_reason.len() > 64
        || result.finish_reason.chars().any(char::is_control)
    {
        let original = ProtocolError::new(
            ProtocolErrorCode::Internal,
            "backend returned an invalid decode result",
        );
        if let Some(state) = &result.state {
            return Err(cleanup_rejected_backend_state(
                inner,
                state,
                pending_replacement.as_ref(),
                original,
            ));
        }
        return Err(original);
    }
    let record_model_id = record.model_id.clone();
    let record_state_id = record.state_id.clone();
    let record_compatibility = record.compatibility.clone();
    drop(record);
    if let Err(error) = ensure_backend_cleanup_healthy(inner) {
        if let Some(state) = &result.state {
            return Err(cleanup_rejected_backend_state(
                inner,
                state,
                pending_replacement.as_ref(),
                error,
            ));
        }
        return Err(error);
    }
    let updated_handoff = if let Some(state) = result.state {
        let (lease, _) =
            commit_backend_memory(inner, state, pending_replacement, &record_compatibility)?;
        let expires = now_unix_ms().saturating_add(HANDOFF_TTL_MILLIS);
        Some(replacement_handoff.issue(HandoffRecord {
            principal_id: principal_id.to_string(),
            model_id: record_model_id,
            state_id: record_state_id,
            state: lease,
            compatibility: record_compatibility,
            expires_unix_ms: expires,
        })?)
    } else {
        drop(pending_replacement);
        drop(replacement_handoff);
        None
    };
    Ok(DecodeResponse {
        text: result.text,
        handoff: updated_handoff,
        completion_tokens: result.completion_tokens,
        finish_reason: result.finish_reason,
    })
}

fn touch_handoff_state(
    inner: &LocalInner,
    principal_id: &str,
    record: &HandoffRecord,
) -> ProtocolResult<()> {
    let Some(state_id) = record.state_id.as_deref() else {
        return Ok(());
    };
    let key = (principal_id.to_string(), state_id.to_string());
    let mut states = inner.volatile_states.lock().map_err(|_| internal())?;
    let Some(state) = states.get_mut(&key) else {
        return Ok(());
    };
    let now = now_unix_ms();
    if volatile_state_is_expired(state, now) {
        return Ok(());
    }
    if let (Some(memory), Some(allocation_id)) = (&inner.memory, state.allocation_id) {
        memory.touch(allocation_id).map_err(memory_error)?;
    }
    state.summary.last_accessed_unix_ms = now;
    Ok(())
}

fn reserve_backend_memory(
    inner: &LocalInner,
    requirement: Option<BackendMemoryRequirement>,
    initially_pinned: bool,
) -> ProtocolResult<Option<PendingMemoryLoad>> {
    let Some(requirement) = requirement else {
        return Ok(None);
    };
    let memory = inner.memory.as_ref().ok_or_else(|| {
        unavailable("dynamic memory telemetry is unavailable; the backend load was not started")
    })?;
    let tier = managed_memory_tier(requirement.tier)?;
    let demotion_target = requirement
        .demotion_target
        .map(managed_memory_tier)
        .transpose()?;
    let allocation_id = AllocationId::new(inner.next_allocation_id.fetch_add(1, Ordering::Relaxed))
        .map_err(memory_error)?;
    let reserve = || {
        memory.reserve_load(
            allocation_id,
            tier,
            requirement.bytes,
            initially_pinned,
            demotion_target,
        )
    };
    let reservation = match reserve() {
        Ok(reservation) => reservation,
        Err(MemoryError::ReservationDenied { .. }) => {
            relieve_memory_pressure(inner, tier, requirement.bytes)?;
            reserve().map_err(memory_error)?
        }
        Err(error) => return Err(memory_error(error)),
    };
    Ok(Some(PendingMemoryLoad {
        allocation_id,
        tier,
        bytes: requirement.bytes,
        reservation,
    }))
}

fn commit_backend_memory(
    inner: &LocalInner,
    state: BackendState,
    pending: Option<PendingMemoryLoad>,
    compatibility: &crate::werk_protocol::CompatibilityEnvelope,
) -> ProtocolResult<(BackendStateLease, Option<AllocationId>)> {
    if let Err(error) = validate_backend_state(inner, &state, compatibility) {
        return Err(cleanup_rejected_backend_state(
            inner,
            &state,
            pending.as_ref(),
            error,
        ));
    }
    let Some(pending) = pending else {
        if optional_managed_memory_tier(state.natural_tier()).is_some() {
            return Err(cleanup_rejected_backend_state(
                inner,
                &state,
                None,
                ProtocolError::new(
                    ProtocolErrorCode::Internal,
                    "backend returned in-memory state without a pre-load reservation",
                ),
            ));
        }
        return Ok((local_backend_lease(inner, state, None), None));
    };
    let state_tier = match optional_managed_memory_tier(state.natural_tier()) {
        Some(tier) => tier,
        None => {
            return Err(cleanup_rejected_backend_state(
                inner,
                &state,
                Some(&pending),
                ProtocolError::new(
                    ProtocolErrorCode::Internal,
                    "backend returned a non-memory state for a reserved memory load",
                ),
            ));
        }
    };
    if state_tier != pending.tier || state.bytes().is_some_and(|bytes| bytes > pending.bytes) {
        return Err(cleanup_rejected_backend_state(
            inner,
            &state,
            Some(&pending),
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend state exceeded or contradicted its pre-load memory reservation",
            ),
        ));
    }
    let allocation_id = pending.allocation_id;
    let orphaned_tier = pending.tier;
    let orphaned_bytes = Some(pending.bytes);
    if let Err(error) = pending.reservation.commit_load() {
        return Err(cleanup_rejected_backend_state_with_hint(
            inner,
            &state,
            Some((orphaned_tier, orphaned_bytes)),
            memory_error(error),
        ));
    }
    let lease = local_backend_lease(inner, state, Some(allocation_id));
    Ok((lease, Some(allocation_id)))
}

fn local_backend_lease(
    inner: &LocalInner,
    state: BackendState,
    allocation_id: Option<AllocationId>,
) -> BackendStateLease {
    let memory = inner.memory.clone();
    let cleanup_latched = inner.backend_cleanup_latched.clone();
    let cleanup_failures = inner.backend_cleanup_failures.clone();
    BackendStateLease::with_release_hook(inner.adapter.clone(), state, move |released| {
        if released {
            if let (Some(memory), Some(allocation_id)) = (&memory, allocation_id) {
                let _ = memory.remove_allocation(allocation_id);
            }
            return;
        }
        if let (Some(memory), Some(allocation_id)) = (&memory, allocation_id) {
            let _ = memory.record_failed_release(allocation_id);
        }
        cleanup_failures.fetch_add(1, Ordering::Relaxed);
        cleanup_latched.store(true, Ordering::Release);
    })
}

fn cleanup_rejected_backend_state(
    inner: &LocalInner,
    state: &BackendState,
    pending: Option<&PendingMemoryLoad>,
    original: ProtocolError,
) -> ProtocolError {
    let reserved = pending.map(|pending| (pending.tier, Some(pending.bytes)));
    cleanup_rejected_backend_state_with_fallback(inner, state, reserved, original)
}

fn cleanup_rejected_backend_state_with_fallback(
    inner: &LocalInner,
    state: &BackendState,
    fallback: Option<(MemoryTier, Option<u64>)>,
    original: ProtocolError,
) -> ProtocolError {
    let actual =
        optional_managed_memory_tier(state.natural_tier()).map(|tier| (tier, state.bytes()));
    let hint = match (actual, fallback) {
        (Some((actual_tier, actual_bytes)), Some((fallback_tier, fallback_bytes)))
            if actual_tier == fallback_tier =>
        {
            Some((
                actual_tier,
                match (actual_bytes, fallback_bytes) {
                    (Some(actual), Some(fallback)) => Some(actual.max(fallback)),
                    _ => None,
                },
            ))
        }
        (Some(actual), _) => Some(actual),
        (None, fallback) => fallback,
    };
    cleanup_rejected_backend_state_with_hint(inner, state, hint, original)
}

fn cleanup_rejected_backend_state_with_hint(
    inner: &LocalInner,
    state: &BackendState,
    orphan_hint: Option<(MemoryTier, Option<u64>)>,
    original: ProtocolError,
) -> ProtocolError {
    if inner.adapter.release(state).is_ok() {
        return original;
    }
    inner
        .backend_cleanup_failures
        .fetch_add(1, Ordering::Relaxed);
    inner.backend_cleanup_latched.store(true, Ordering::Release);
    let accounted = match (&inner.memory, orphan_hint) {
        (Some(memory), Some((tier, bytes))) => memory.record_orphaned_release(tier, bytes).is_ok(),
        _ => false,
    };
    if accounted {
        ProtocolError::new(
            ProtocolErrorCode::Internal,
            "backend state was rejected and cleanup could not be confirmed; memory remains conservatively accounted",
        )
    } else {
        unavailable(
            "backend cleanup could not be confirmed; new backend state work is disabled for this process",
        )
    }
}

fn backend_operation_error(inner: &LocalInner, error: ProtocolError) -> ProtocolError {
    let Some((tier, bytes)) = error.backend_cleanup_unconfirmed() else {
        return error;
    };
    let tier = match tier {
        StateTier::Ram => Some(MemoryTier::Host),
        StateTier::Vram => Some(MemoryTier::Accelerator(0)),
        StateTier::Disk | StateTier::External => None,
    };
    inner
        .backend_cleanup_failures
        .fetch_add(1, Ordering::Relaxed);
    inner.backend_cleanup_latched.store(true, Ordering::Release);
    let accounted = match (&inner.memory, tier) {
        (Some(memory), Some(tier)) => memory.record_orphaned_release(tier, bytes).is_ok(),
        _ => false,
    };
    if accounted {
        ProtocolError::new(
            ProtocolErrorCode::Internal,
            "backend routing cleanup could not be confirmed; memory remains conservatively accounted",
        )
    } else {
        unavailable(
            "backend routing cleanup could not be confirmed; new backend state work is disabled for this process",
        )
    }
}

fn ensure_backend_cleanup_healthy(inner: &LocalInner) -> ProtocolResult<()> {
    if inner.backend_cleanup_latched.load(Ordering::Acquire) {
        return Err(unavailable(
            "new backend state work is disabled after an unconfirmed cleanup failure",
        ));
    }
    Ok(())
}

fn managed_memory_tier(tier: StateTier) -> ProtocolResult<MemoryTier> {
    optional_managed_memory_tier(tier).ok_or_else(|| {
        ProtocolError::new(
            ProtocolErrorCode::Internal,
            "backend requested an in-memory reservation for a non-memory state tier",
        )
    })
}

fn require_policy_state_tier(
    mode: PersistenceMode,
    state: &BackendState,
    descriptor: &BackendRuntimeDescriptor,
    allow_experimental: bool,
) -> ProtocolResult<()> {
    if mode == PersistenceMode::Memory {
        if optional_managed_memory_tier(state.natural_tier()).is_none() {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend returned a non-memory state for persistence mode memory",
            ));
        }
        require_state_tier_capability(descriptor, state.natural_tier(), allow_experimental)?;
    }
    Ok(())
}

fn require_policy_requirement_tier(
    mode: PersistenceMode,
    requirement: Option<&BackendMemoryRequirement>,
    descriptor: &BackendRuntimeDescriptor,
    allow_experimental: bool,
) -> ProtocolResult<()> {
    if mode == PersistenceMode::Memory
        && let Some(requirement) = requirement
    {
        require_state_tier_capability(descriptor, requirement.tier, allow_experimental)?;
    }
    Ok(())
}

fn optional_managed_memory_tier(tier: StateTier) -> Option<MemoryTier> {
    match tier {
        StateTier::Ram => Some(MemoryTier::Host),
        StateTier::Vram => Some(MemoryTier::Accelerator(0)),
        StateTier::Disk | StateTier::External => None,
    }
}

fn state_tier_for_memory(tier: MemoryTier) -> StateTier {
    match tier {
        MemoryTier::Host => StateTier::Ram,
        MemoryTier::Accelerator(_) => StateTier::Vram,
    }
}

fn relieve_memory_pressure(
    inner: &LocalInner,
    tier: MemoryTier,
    requested_bytes: u64,
) -> ProtocolResult<()> {
    let Some(memory) = &inner.memory else {
        return Ok(());
    };
    let eligible = {
        let states = inner.volatile_states.lock().map_err(|_| internal())?;
        states
            .values()
            .filter(|state| state.lease.strong_count() == 1)
            .filter_map(|state| state.allocation_id)
            .collect::<BTreeSet<_>>()
    };
    let plan = if requested_bytes == 0 {
        memory.plan_pressure_actions_among(tier, &eligible)
    } else {
        memory.plan_for_reservation_among(tier, requested_bytes, &eligible)
    }
    .map_err(memory_error)?;
    if requested_bytes > 0 && plan.unresolved_pressure_bytes > 0 {
        return Err(ProtocolError::new(
            ProtocolErrorCode::ResourceExhausted,
            "memory capacity cannot be made available from the eligible runtime states",
        )
        .retryable(true));
    }
    for action in plan.actions {
        let allocation_id = action.allocation_id();
        let state_key = {
            let states = inner.volatile_states.lock().map_err(|_| internal())?;
            states
                .iter()
                .find(|(_, state)| {
                    state.allocation_id == Some(allocation_id) && state.lease.strong_count() == 1
                })
                .map(|(key, _)| key.clone())
        };
        let Some(state_key) = state_key else {
            continue;
        };
        let permit = match memory.begin_pressure_action(&action) {
            Ok(permit) => permit,
            Err(MemoryError::StaleAction(_) | MemoryError::ActionInFlight(_)) => continue,
            Err(error) => return Err(memory_error(error)),
        };
        match action {
            PressureAction::Evict { .. } => {
                let mut removed = inner
                    .volatile_states
                    .lock()
                    .map_err(|_| internal())?
                    .remove(&state_key)
                    .ok_or_else(internal)?;
                if !removed.lease.has_release_hook() {
                    inner
                        .volatile_states
                        .lock()
                        .map_err(|_| internal())?
                        .insert(state_key, removed);
                    return Err(internal());
                }
                let accounting = match removed.lease.release_backend_and_take_hook() {
                    Ok(Some(accounting)) => accounting,
                    Ok(None) => {
                        inner
                            .volatile_states
                            .lock()
                            .map_err(|_| internal())?
                            .insert(state_key, removed);
                        return Err(internal());
                    }
                    Err(error) => {
                        inner
                            .volatile_states
                            .lock()
                            .map_err(|_| internal())?
                            .insert(state_key, removed);
                        return Err(error);
                    }
                };
                drop(removed);
                if let Err(error) = permit.commit() {
                    // Backend release already succeeded. Reconcile the
                    // allocation directly if the preclaimed transition became
                    // stale unexpectedly.
                    accounting(true);
                    return Err(memory_error(error));
                }
            }
            PressureAction::Demote { to, bytes, .. } => {
                let mut state = inner
                    .volatile_states
                    .lock()
                    .map_err(|_| internal())?
                    .remove(&state_key)
                    .ok_or_else(internal)?;
                let moved = match inner.adapter.move_state(
                    Arc::new(state.lease.state().clone()),
                    state_tier_for_memory(to),
                ) {
                    Ok(moved) => moved,
                    Err(error) => {
                        inner
                            .volatile_states
                            .lock()
                            .map_err(|_| internal())?
                            .insert(state_key, state);
                        return Err(backend_operation_error(inner, error));
                    }
                };
                if let Err(error) = validate_backend_state(inner, &moved, &state.compatibility) {
                    let error = cleanup_rejected_backend_state_with_fallback(
                        inner,
                        &moved,
                        Some((to, Some(bytes))),
                        error,
                    );
                    inner
                        .volatile_states
                        .lock()
                        .map_err(|_| internal())?
                        .insert(state_key, state);
                    return Err(error);
                }
                if optional_managed_memory_tier(moved.natural_tier()) != Some(to)
                    || moved.bytes().is_some_and(|moved_bytes| moved_bytes > bytes)
                {
                    let error = cleanup_rejected_backend_state_with_fallback(
                        inner,
                        &moved,
                        Some((to, Some(bytes))),
                        ProtocolError::new(
                            ProtocolErrorCode::Internal,
                            "backend demotion contradicted the reserved target tier or size",
                        ),
                    );
                    inner
                        .volatile_states
                        .lock()
                        .map_err(|_| internal())?
                        .insert(state_key, state);
                    return Err(error);
                }
                if !state.lease.has_release_hook() {
                    let error = cleanup_rejected_backend_state_with_fallback(
                        inner,
                        &moved,
                        Some((to, Some(bytes))),
                        internal(),
                    );
                    inner
                        .volatile_states
                        .lock()
                        .map_err(|_| internal())?
                        .insert(state_key, state);
                    return Err(error);
                }
                let accounting = match state.lease.release_backend_and_take_hook() {
                    Ok(Some(accounting)) => accounting,
                    Ok(None) => {
                        let error = cleanup_rejected_backend_state_with_fallback(
                            inner,
                            &moved,
                            Some((to, Some(bytes))),
                            internal(),
                        );
                        inner
                            .volatile_states
                            .lock()
                            .map_err(|_| internal())?
                            .insert(state_key, state);
                        return Err(error);
                    }
                    Err(error) => {
                        let error = cleanup_rejected_backend_state_with_fallback(
                            inner,
                            &moved,
                            Some((to, Some(bytes))),
                            error,
                        );
                        inner
                            .volatile_states
                            .lock()
                            .map_err(|_| internal())?
                            .insert(state_key, state);
                        return Err(error);
                    }
                };
                let replacement = BackendStateLease::with_boxed_release_hook(
                    inner.adapter.clone(),
                    moved,
                    accounting,
                );
                state.summary.tier = replacement.state().natural_tier();
                state.summary.bytes = replacement.state().bytes();
                state.lease = replacement;
                if let Err(error) = permit.commit() {
                    drop(state);
                    return Err(memory_error(error));
                }
                inner
                    .volatile_states
                    .lock()
                    .map_err(|_| internal())?
                    .insert(state_key, state);
            }
        }
    }
    Ok(())
}

fn memory_error(error: MemoryError) -> ProtocolError {
    match error {
        MemoryError::ReservationDenied { .. }
        | MemoryError::AllocationLimitReached(_)
        | MemoryError::ReservationLimitReached(_) => ProtocolError::new(
            ProtocolErrorCode::ResourceExhausted,
            format!("memory capacity could not be reserved: {error}"),
        )
        .retryable(true),
        MemoryError::InvalidTelemetry(_) | MemoryError::UnknownTier(_) => {
            unavailable(format!("memory accounting is unavailable: {error}"))
        }
        _ => ProtocolError::new(
            ProtocolErrorCode::Internal,
            format!("memory accounting rejected an inconsistent operation: {error}"),
        ),
    }
}

fn retain_prefill_state(
    inner: &LocalInner,
    principal_id: &str,
    model_id: &str,
    policy: &crate::werk_protocol::PersistencePolicy,
    prompt_tokens: u64,
    compatibility: &crate::werk_protocol::CompatibilityEnvelope,
    lease: BackendStateLease,
    allocation_id: Option<AllocationId>,
    allow_experimental: bool,
) -> ProtocolResult<Option<String>> {
    if policy.mode == PersistenceMode::Ephemeral {
        return Ok(None);
    }
    if matches!(policy.mode, PersistenceMode::Disk | PersistenceMode::Auto) {
        let descriptor = inner.adapter.descriptor_for_state(lease.state())?;
        let persistence = require_backend_capability(
            &descriptor,
            "runtime.state.persistence",
            allow_experimental,
            true,
        )
        .and_then(|()| {
            require_state_tier_capability(&descriptor, StateTier::Disk, allow_experimental)
        });
        if persistence.is_ok() {
            let prepared = match prepare_snapshot_payload(inner, &lease, compatibility) {
                Ok(prepared) => Some(prepared),
                Err(_) if policy.mode == PersistenceMode::Auto => {
                    // Snapshot preparation can create an independently owned
                    // backend export before discovering an unusable payload.
                    // The helper drops that export before this cleanup-health
                    // decision, so Auto never hides an unconfirmed release.
                    ensure_backend_cleanup_healthy(inner)?;
                    None
                }
                Err(error) => return Err(error),
            };
            if let Some((snapshot, payload)) = prepared {
                let expires_unix_ms = policy
                    .ttl_seconds
                    .map(|seconds| now_unix_ms().saturating_add(seconds.saturating_mul(1000)));
                let state = inner.state_store.commit(
                    principal_id,
                    NewStoredState {
                        model_id: model_id.to_string(),
                        backend: compatibility.backend.clone(),
                        compatibility: compatibility.clone(),
                        payload,
                        prompt_tokens,
                        expires_unix_ms,
                        pinned: policy.pin,
                    },
                )?;
                drop(snapshot);
                ensure_backend_cleanup_healthy(inner)?;
                return Ok(Some(state.id));
            }
        }
        if policy.mode == PersistenceMode::Disk {
            persistence?;
            return Err(unsupported(
                "the active backend did not produce a disk-persistable opaque snapshot",
            ));
        }
    }

    let id = random_state_id()?;
    let now = now_unix_ms();
    let summary = StateSummary {
        id: id.clone(),
        model_id: model_id.to_string(),
        tier: lease.state().natural_tier(),
        status: StateStatus::Ready,
        bytes: lease.state().bytes(),
        created_unix_ms: now,
        last_accessed_unix_ms: now,
        expires_unix_ms: policy
            .ttl_seconds
            .map(|seconds| now.saturating_add(seconds.saturating_mul(1000))),
        pinned: policy.pin,
        backend: compatibility.backend.clone(),
        reusable: true,
    };
    if let (Some(memory), Some(allocation_id)) = (&inner.memory, allocation_id) {
        memory
            .set_pinned(allocation_id, policy.pin)
            .map_err(memory_error)?;
    }
    insert_volatile(
        inner,
        VolatileState {
            principal_id: principal_id.to_string(),
            summary,
            compatibility: compatibility.clone(),
            prompt_tokens,
            lease,
            persisted: false,
            allocation_id,
        },
    )?;
    Ok(Some(id))
}

fn payload_source(state: &BackendState) -> ProtocolResult<Option<OpaquePayloadSource>> {
    match state {
        BackendState::OpaqueBytes { bytes, .. } => {
            Ok(Some(OpaquePayloadSource::Bytes(bytes.clone())))
        }
        BackendState::OpaqueFile { path, bytes, .. } => {
            OpaquePayloadSource::open_file(path, *bytes).map(Some)
        }
        BackendState::InProcess { .. } | BackendState::External { .. } => Ok(None),
    }
}

fn validate_live_opaque_snapshot(
    lease: &BackendStateLease,
    compatibility: &crate::werk_protocol::CompatibilityEnvelope,
) -> ProtocolResult<()> {
    if !matches!(
        lease.state(),
        BackendState::OpaqueBytes { .. } | BackendState::OpaqueFile { .. }
    ) {
        return Err(ProtocolError::new(
            ProtocolErrorCode::Internal,
            "only opaque backend state can be persisted without snapshot export",
        ));
    }
    lease
        .adapter()
        .validate_snapshot(lease.state(), compatibility)?;
    validate_nonzero_backend_state(lease.state())
}

fn inspect_snapshot_export(
    lease: &BackendStateLease,
    compatibility: &crate::werk_protocol::CompatibilityEnvelope,
) -> ProtocolResult<()> {
    match lease.state() {
        BackendState::OpaqueBytes { .. } | BackendState::OpaqueFile { .. } => {
            validate_live_opaque_snapshot(lease, compatibility)?;
            if payload_source(lease.state())?.is_none() {
                return Err(unsupported(
                    "the active backend state is not a disk-persistable opaque snapshot",
                ));
            }
            Ok(())
        }
        state => lease
            .adapter()
            .inspect_snapshot_export(state, compatibility),
    }
}

fn prepare_snapshot_payload(
    inner: &LocalInner,
    lease: &BackendStateLease,
    compatibility: &crate::werk_protocol::CompatibilityEnvelope,
) -> ProtocolResult<(BackendStateLease, OpaquePayloadSource)> {
    let snapshot = match lease.state() {
        BackendState::OpaqueBytes { .. } | BackendState::OpaqueFile { .. } => {
            validate_live_opaque_snapshot(lease, compatibility)?;
            lease.clone()
        }
        state => snapshot_state_lease(inner, state, compatibility)?,
    };
    let payload = match payload_source(snapshot.state()) {
        Ok(Some(payload)) => payload,
        Ok(None) => {
            drop(snapshot);
            return Err(unsupported(
                "the active backend did not produce a disk-persistable opaque snapshot",
            ));
        }
        Err(error) => {
            drop(snapshot);
            return Err(error);
        }
    };
    Ok((snapshot, payload))
}

fn persist_volatile_state(
    inner: &LocalInner,
    principal_id: &str,
    state: &VolatileState,
) -> ProtocolResult<StateSummary> {
    let (snapshot, payload) = prepare_snapshot_payload(inner, &state.lease, &state.compatibility)?;
    let committed = inner.state_store.commit_with_id(
        principal_id,
        &state.summary.id,
        NewStoredState {
            model_id: state.summary.model_id.clone(),
            backend: state.compatibility.backend.clone(),
            compatibility: state.compatibility.clone(),
            payload,
            prompt_tokens: state.prompt_tokens,
            expires_unix_ms: state.summary.expires_unix_ms,
            pinned: state.summary.pinned,
        },
    )?;
    drop(snapshot);
    ensure_backend_cleanup_healthy(inner)?;
    Ok(committed)
}

fn persist_reused_volatile_state(
    inner: &LocalInner,
    principal_id: &str,
    state: &VolatileState,
) -> ProtocolResult<()> {
    persist_volatile_state(inner, principal_id, state)?;
    let key = (principal_id.to_string(), state.summary.id.clone());
    let mut states = inner.volatile_states.lock().map_err(|_| internal())?;
    let current = states.get_mut(&key).ok_or_else(|| {
        ProtocolError::new(
            ProtocolErrorCode::Conflict,
            "runtime state changed while it was being persisted",
        )
    })?;
    if current.compatibility != state.compatibility {
        return Err(ProtocolError::new(
            ProtocolErrorCode::Conflict,
            "runtime state compatibility changed while it was being persisted",
        ));
    }
    current.persisted = true;
    Ok(())
}

fn snapshot_state_lease(
    inner: &LocalInner,
    state: &BackendState,
    compatibility: &crate::werk_protocol::CompatibilityEnvelope,
) -> ProtocolResult<BackendStateLease> {
    let snapshot = inner
        .adapter
        .snapshot(state)
        .map_err(|error| backend_operation_error(inner, error))?;
    let validation = inner
        .adapter
        .validate_snapshot(&snapshot, compatibility)
        .and_then(|()| validate_nonzero_backend_state(&snapshot));
    if let Err(error) = validation {
        return Err(cleanup_rejected_backend_state(
            inner, &snapshot, None, error,
        ));
    }
    Ok(local_backend_lease(inner, snapshot, None))
}

fn require_state_tier_capability(
    descriptor: &BackendRuntimeDescriptor,
    target: StateTier,
    allow_experimental: bool,
) -> ProtocolResult<()> {
    let capability = match target {
        StateTier::Vram => "runtime.state.tier.vram",
        StateTier::Ram => "runtime.state.tier.ram",
        StateTier::Disk => "runtime.state.tier.disk",
        StateTier::External => {
            return Err(unsupported(
                "external state ownership cannot be selected as a local tier",
            ));
        }
    };
    require_backend_capability(descriptor, capability, allow_experimental, true)
}

fn preflight_prefill_capability(
    descriptor: &BackendRuntimeDescriptor,
    id: &str,
    allow_experimental: bool,
) -> ProtocolResult<()> {
    validate_runtime_descriptor(descriptor)?;
    let capability = descriptor
        .capabilities
        .iter()
        .find(|capability| capability.id == id)
        .ok_or_else(|| {
            unsupported("the resolved backend did not declare the requested capability")
        })?;
    if capability.status == CapabilityStatus::Unavailable && allow_experimental {
        // The unavailable state can be resolved only by the explicitly
        // opted-in functional probe. Its result is checked authoritatively
        // immediately after compatibility construction.
        return Ok(());
    }
    require_capability_status(capability, id, allow_experimental, true)
}

fn validate_descriptor_ownership(
    descriptor: &BackendRuntimeDescriptor,
    compatibility: &crate::werk_protocol::CompatibilityEnvelope,
) -> ProtocolResult<()> {
    validate_runtime_descriptor(descriptor)?;
    let mut mismatch_fields = Vec::new();
    if descriptor.backend != compatibility.backend {
        mismatch_fields.push("backend");
    }
    if descriptor.backend_version != compatibility.backend_version {
        mismatch_fields.push("backend_version");
    }
    if descriptor.adapter_version != compatibility.runtime_adapter_version {
        mismatch_fields.push("runtime_adapter_version");
    }
    if descriptor.accelerator_family != compatibility.accelerator_family {
        mismatch_fields.push("accelerator_family");
    }
    if mismatch_fields.is_empty() {
        Ok(())
    } else {
        Err(ProtocolError::new(
            ProtocolErrorCode::IncompatibleState,
            "persisted state metadata does not match the resolved backend runtime",
        )
        .with_details(serde_json::json!({ "mismatch_fields": mismatch_fields })))
    }
}

fn issue_handoff(
    reservation: HandoffReservation<'_>,
    principal_id: &str,
    model_id: &str,
    state_id: Option<String>,
    prompt_tokens: u64,
    reused: bool,
    lease: BackendStateLease,
    compatibility: crate::werk_protocol::CompatibilityEnvelope,
) -> ProtocolResult<PrefillResponse> {
    let tier = lease.state().natural_tier();
    let expires_unix_ms = now_unix_ms().saturating_add(HANDOFF_TTL_MILLIS);
    let handoff = reservation.issue(HandoffRecord {
        principal_id: principal_id.to_string(),
        model_id: model_id.to_string(),
        state_id: state_id.clone(),
        state: lease,
        compatibility,
        expires_unix_ms,
    })?;
    Ok(PrefillResponse {
        handoff,
        state_id,
        prompt_tokens,
        reused,
        tier,
        expires_unix_ms,
    })
}

fn find_volatile_reuse(
    inner: &LocalInner,
    principal_id: &str,
    model_id: &str,
    compatibility: &crate::werk_protocol::CompatibilityEnvelope,
) -> ProtocolResult<Option<VolatileState>> {
    find_volatile_reuse_with_clock(inner, principal_id, model_id, compatibility, now_unix_ms)
}

fn find_volatile_reuse_with_clock(
    inner: &LocalInner,
    principal_id: &str,
    model_id: &str,
    compatibility: &crate::werk_protocol::CompatibilityEnvelope,
    clock: impl Fn() -> u64,
) -> ProtocolResult<Option<VolatileState>> {
    let scan_now = clock();
    let mut states = inner.volatile_states.lock().map_err(|_| internal())?;
    let expired = take_expired_volatile_states(&mut states, Some(principal_id), scan_now);
    let mut candidates = states
        .values()
        .filter(|state| {
            state.principal_id == principal_id
                && !volatile_state_is_expired(state, scan_now)
                && state.summary.model_id == model_id
                && state.compatibility.prompt_fingerprint == compatibility.prompt_fingerprint
                && state
                    .compatibility
                    .mismatch_fields(compatibility)
                    .is_empty()
        })
        .cloned()
        .collect::<Vec<_>>();
    drop(states);
    drop(expired);
    candidates.sort_by(|left, right| {
        right
            .summary
            .last_accessed_unix_ms
            .cmp(&left.summary.last_accessed_unix_ms)
            .then_with(|| left.summary.id.cmp(&right.summary.id))
    });
    for mut state in candidates {
        if state
            .lease
            .adapter()
            .validate_state(state.lease.state(), &state.compatibility)
            .is_err()
        {
            continue;
        }
        if let (Some(memory), Some(allocation_id)) = (&inner.memory, state.allocation_id) {
            memory.touch(allocation_id).map_err(memory_error)?;
        }
        let key = (principal_id.to_string(), state.summary.id.clone());
        let mut states = inner.volatile_states.lock().map_err(|_| internal())?;
        let validation_now = clock();
        let current = states.get_mut(&key).ok_or_else(|| {
            ProtocolError::new(
                ProtocolErrorCode::Conflict,
                "runtime state changed while reuse was being validated",
            )
        })?;
        if volatile_state_is_expired(current, validation_now)
            || current.compatibility != state.compatibility
        {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Conflict,
                "runtime state compatibility changed while reuse was being validated",
            ));
        }
        current.summary.last_accessed_unix_ms = validation_now;
        state = current.clone();
        drop(states);
        return Ok(Some(state));
    }
    Ok(None)
}

fn insert_volatile(inner: &LocalInner, state: VolatileState) -> ProtocolResult<()> {
    let expired = {
        let mut states = inner.volatile_states.lock().map_err(|_| internal())?;
        take_expired_volatile_states(&mut states, None, now_unix_ms())
    };
    drop(expired);
    ensure_backend_cleanup_healthy(inner)?;
    let mut states = inner.volatile_states.lock().map_err(|_| internal())?;
    let result = (|| {
        let key = (state.principal_id.clone(), state.summary.id.clone());
        if states.contains_key(&key) {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Conflict,
                "runtime state ID already exists in this security namespace",
            ));
        }
        let principal_states = states
            .values()
            .filter(|existing| existing.principal_id == state.principal_id)
            .count();
        if principal_states >= MAX_VOLATILE_STATES_PER_PRINCIPAL {
            return Err(ProtocolError::new(
                ProtocolErrorCode::ResourceExhausted,
                "volatile runtime state limit is reached for this security namespace",
            )
            .retryable(true));
        }
        if states.len() >= MAX_VOLATILE_STATES {
            return Err(ProtocolError::new(
                ProtocolErrorCode::ResourceExhausted,
                "global volatile runtime state limit is reached",
            )
            .retryable(true));
        }
        states.insert(key, state);
        Ok(())
    })();
    drop(states);
    result
}

fn list_states_inner(
    inner: &LocalInner,
    principal_id: &str,
    filter: &StateListFilter,
) -> ProtocolResult<StateListResponse> {
    validate_state_list_filter(filter, inner.limits.max_page_size)?;
    if let Some(cursor) = filter.cursor.as_deref() {
        let _ = decode_state_cursor(cursor)?;
    }
    let limit = filter.limit.unwrap_or(inner.limits.max_page_size);
    // The store validates catalog integrity without mutating it. Disk-only
    // entries nevertheless stay fail-closed until an explicit restore proves
    // that the concrete backend can still consume them. A live volatile entry
    // with the same ID replaces this conservative summary below after
    // adapter-owned validation.
    let disk_states = inner
        .state_store
        .all_summaries(principal_id)?
        .into_iter()
        .map(|mut state| {
            state.status = StateStatus::Unavailable;
            state.reusable = false;
            state
        })
        .collect::<Vec<_>>();
    let mut states_by_id = disk_states
        .into_iter()
        .map(|state| (state.id.clone(), state))
        .collect::<BTreeMap<_, _>>();
    let now = now_unix_ms();
    let volatile = {
        let volatile = inner.volatile_states.lock().map_err(|_| internal())?;
        volatile
            .values()
            .filter(|state| {
                state.principal_id == principal_id && !volatile_state_is_expired(state, now)
            })
            .cloned()
            .collect::<Vec<_>>()
    };
    for state in volatile {
        let mut summary = state.summary.clone();
        if inner
            .adapter
            .validate_state(state.lease.state(), &state.compatibility)
            .is_err()
        {
            summary.status = StateStatus::Unavailable;
            summary.reusable = false;
        }
        states_by_id.insert(summary.id.clone(), summary);
    }
    let mut states = states_by_id.into_values().collect::<Vec<_>>();
    states.retain(|state| {
        filter
            .model_id
            .as_deref()
            .is_none_or(|model| state.model_id == model)
            && filter.tier.is_none_or(|tier| state.tier == tier)
    });
    states.sort_by(|left, right| left.id.cmp(&right.id));
    if let Some(cursor) = filter.cursor.as_deref() {
        let after = decode_state_cursor(cursor)?;
        states.retain(|state| state.id > after);
    }
    let has_more = states.len() > usize::from(limit);
    states.truncate(usize::from(limit));
    let next_cursor = has_more
        .then(|| states.last().map(|state| encode_state_cursor(&state.id)))
        .flatten();
    Ok(StateListResponse {
        states,
        next_cursor,
    })
}

fn state_action_inner(
    inner: &LocalInner,
    principal_id: &str,
    state_id: &str,
    request: &StateActionRequest,
) -> ProtocolResult<StateActionResponse> {
    validate_state_identifier(state_id)?;
    match request.action {
        StateAction::Pin | StateAction::Unpin | StateAction::Evict
            if request.target_tier.is_some() =>
        {
            return Err(invalid(
                "target_tier is only valid for promote and demote actions",
            ));
        }
        StateAction::Promote | StateAction::Demote if request.target_tier.is_none() => {
            return Err(invalid("promote and demote actions require target_tier"));
        }
        _ => {}
    }
    if !request.dry_run && matches!(request.action, StateAction::Promote | StateAction::Demote) {
        ensure_backend_cleanup_healthy(inner)?;
    }
    let state_key = (principal_id.to_string(), state_id.to_string());
    let (volatile, expired) = {
        let now = now_unix_ms();
        let mut states = inner.volatile_states.lock().map_err(|_| internal())?;
        let expired = if request.dry_run {
            Vec::new()
        } else {
            take_expired_volatile_states(&mut states, Some(principal_id), now)
        };
        let volatile = states
            .get(&state_key)
            .filter(|state| !volatile_state_is_expired(state, now))
            .cloned();
        (volatile, expired)
    };
    drop(expired);
    if !request.dry_run && matches!(request.action, StateAction::Promote | StateAction::Demote) {
        ensure_backend_cleanup_healthy(inner)?;
    }
    if let Some(mut state) = volatile {
        match request.action {
            StateAction::Pin | StateAction::Unpin => {
                let pinned = request.action == StateAction::Pin;
                let changed = state.summary.pinned != pinned;
                state.summary.pinned = pinned;
                if changed && !request.dry_run {
                    if let (Some(memory), Some(allocation_id)) =
                        (&inner.memory, state.allocation_id)
                    {
                        memory
                            .set_pinned(allocation_id, pinned)
                            .map_err(memory_error)?;
                    }
                    if state.persisted {
                        if let Err(error) =
                            inner
                                .state_store
                                .set_pinned(principal_id, state_id, pinned, false)
                        {
                            if let (Some(memory), Some(allocation_id)) =
                                (&inner.memory, state.allocation_id)
                            {
                                let _ = memory.set_pinned(allocation_id, !pinned);
                            }
                            return Err(error);
                        }
                    }
                    drop(replace_volatile_state(
                        inner,
                        state_key.clone(),
                        state.clone(),
                    )?);
                }
                return Ok(StateActionResponse {
                    state: state.summary,
                    changed,
                    dry_run: request.dry_run,
                });
            }
            StateAction::Evict => {
                if !request.dry_run {
                    let mut removed = take_volatile_state(inner, &state_key)?.ok_or_else(|| {
                        ProtocolError::new(
                            ProtocolErrorCode::Conflict,
                            "runtime state changed while it was being evicted",
                        )
                    })?;
                    drop(state);
                    if let Err(error) = release_volatile_state(&mut removed) {
                        drop(replace_volatile_state(inner, state_key, removed)?);
                        return Err(error);
                    }
                    if removed.persisted {
                        let _ = inner.state_store.prune(
                            principal_id,
                            &PruneStatesRequest {
                                selector: StateSelector::Ids {
                                    ids: vec![state_id.to_string()],
                                },
                                dry_run: false,
                            },
                        )?;
                    }
                    removed.summary.status = StateStatus::Unavailable;
                    return Ok(StateActionResponse {
                        state: removed.summary,
                        changed: true,
                        dry_run: false,
                    });
                }
                state.summary.status = StateStatus::Unavailable;
                return Ok(StateActionResponse {
                    state: state.summary,
                    changed: true,
                    dry_run: request.dry_run,
                });
            }
            StateAction::Promote | StateAction::Demote => {
                let descriptor = inner.adapter.descriptor_for_state(state.lease.state())?;
                validate_backend_state(inner, state.lease.state(), &state.compatibility)?;
                let target = request
                    .target_tier
                    .ok_or_else(|| invalid("promote and demote actions require target_tier"))?;
                if target == state.summary.tier {
                    return Ok(StateActionResponse {
                        state: state.summary,
                        changed: false,
                        dry_run: request.dry_run,
                    });
                }
                let direction_is_valid = matches!(
                    (request.action, state.summary.tier, target),
                    (StateAction::Promote, StateTier::Ram, StateTier::Vram)
                        | (StateAction::Demote, StateTier::Vram, StateTier::Ram)
                        | (
                            StateAction::Demote,
                            StateTier::Ram | StateTier::Vram,
                            StateTier::Disk
                        )
                );
                if !direction_is_valid {
                    return Err(invalid(
                        "state tier action does not move in the requested promote/demote direction",
                    ));
                }
                require_state_tier_capability(&descriptor, target, request.allow_experimental)?;
                if target == StateTier::Disk {
                    require_backend_capability(
                        &descriptor,
                        "runtime.state.persistence",
                        request.allow_experimental,
                        true,
                    )?;
                    if request.dry_run {
                        inspect_snapshot_export(&state.lease, &state.compatibility)?;
                    }
                }
                let movement_bytes = if target == StateTier::Disk {
                    None
                } else {
                    Some(state.summary.bytes.ok_or_else(|| {
                        unsupported(
                            "state tier movement requires an exact backend memory-size estimate",
                        )
                    })?)
                };
                if request.dry_run {
                    state.summary.tier = target;
                    return Ok(StateActionResponse {
                        state: state.summary,
                        changed: true,
                        dry_run: true,
                    });
                }
                if target == StateTier::Disk {
                    let summary = if state.persisted {
                        inner.state_store.load(principal_id, state_id)?.summary
                    } else {
                        persist_volatile_state(inner, principal_id, &state)?
                    };
                    let mut removed = take_volatile_state(inner, &state_key)?.ok_or_else(|| {
                        ProtocolError::new(
                            ProtocolErrorCode::Conflict,
                            "runtime state changed while it was being demoted",
                        )
                    })?;
                    removed.persisted = true;
                    drop(state);
                    if let Err(error) = release_volatile_state(&mut removed) {
                        drop(replace_volatile_state(inner, state_key, removed)?);
                        return Err(error);
                    }
                    return Ok(StateActionResponse {
                        state: summary,
                        changed: true,
                        dry_run: false,
                    });
                }
                let bytes = movement_bytes.expect("non-disk movements require a byte estimate");
                let requirement = BackendMemoryRequirement {
                    tier: target,
                    bytes,
                    demotion_target: (target == StateTier::Vram).then_some(StateTier::Ram),
                };
                let pending = reserve_backend_memory(inner, Some(requirement), true)?;
                let moved = inner
                    .adapter
                    .move_state(Arc::new(state.lease.state().clone()), target)
                    .map_err(|error| backend_operation_error(inner, error))?;
                let (lease, allocation_id) =
                    commit_backend_memory(inner, moved, pending, &state.compatibility)?;
                state.summary.tier = lease.state().natural_tier();
                state.summary.bytes = lease.state().bytes();
                state.lease = lease;
                state.allocation_id = allocation_id;
                if let (Some(memory), Some(allocation_id)) = (&inner.memory, allocation_id) {
                    memory
                        .set_pinned(allocation_id, state.summary.pinned)
                        .map_err(memory_error)?;
                }
                let mut old = take_volatile_state(inner, &state_key)?.ok_or_else(|| {
                    ProtocolError::new(
                        ProtocolErrorCode::Conflict,
                        "runtime state changed while its tier was being moved",
                    )
                })?;
                if let Err(error) = release_volatile_state(&mut old) {
                    drop(replace_volatile_state(inner, state_key, old)?);
                    drop(state);
                    return Err(error);
                }
                let response_state = state.summary.clone();
                drop(replace_volatile_state(inner, state_key, state)?);
                return Ok(StateActionResponse {
                    state: response_state,
                    changed: true,
                    dry_run: false,
                });
            }
        }
    }

    match request.action {
        StateAction::Pin | StateAction::Unpin => {
            let pinned = request.action == StateAction::Pin;
            let (state, changed) =
                inner
                    .state_store
                    .set_pinned(principal_id, state_id, pinned, request.dry_run)?;
            Ok(StateActionResponse {
                state,
                changed,
                dry_run: request.dry_run,
            })
        }
        StateAction::Evict => {
            let mut state = if request.dry_run {
                inner.state_store.inspect(principal_id, state_id)?
            } else {
                inner.state_store.load(principal_id, state_id)?
            }
            .summary;
            let result = inner.state_store.prune(
                principal_id,
                &PruneStatesRequest {
                    selector: StateSelector::Ids {
                        ids: vec![state_id.to_string()],
                    },
                    dry_run: request.dry_run,
                },
            )?;
            state.status = StateStatus::Unavailable;
            Ok(StateActionResponse {
                state,
                changed: result.matched == 1,
                dry_run: request.dry_run,
            })
        }
        StateAction::Promote => {
            let target = request
                .target_tier
                .filter(|tier| matches!(tier, StateTier::Ram | StateTier::Vram))
                .ok_or_else(|| invalid("disk state promotion requires target_tier ram or vram"))?;
            let stored = if request.dry_run {
                inner.state_store.inspect(principal_id, state_id)?
            } else {
                inner.state_store.load(principal_id, state_id)?
            };
            if request.dry_run {
                let manifest =
                    inner
                        .store
                        .get_existing(&stored.summary.model_id)
                        .map_err(|_| {
                            ProtocolError::new(
                                ProtocolErrorCode::NotFound,
                                "the state model is not installed",
                            )
                        })?;
                let descriptor = inner.adapter.descriptor_for_model(&manifest)?;
                validate_descriptor_ownership(&descriptor, &stored.compatibility)?;
                require_backend_capability(
                    &descriptor,
                    "runtime.state.restore",
                    request.allow_experimental,
                    true,
                )?;
                require_state_tier_capability(&descriptor, target, request.allow_experimental)?;
                let current = inner
                    .adapter
                    .inspect_compatibility(&manifest, &stored.compatibility.prompt_fingerprint)?;
                validate_compatibility_envelope(&current).map_err(|_| {
                    ProtocolError::new(
                        ProtocolErrorCode::Internal,
                        "backend returned invalid dry-run compatibility metadata",
                    )
                })?;
                if current.prompt_fingerprint != stored.compatibility.prompt_fingerprint {
                    return Err(ProtocolError::new(
                        ProtocolErrorCode::Internal,
                        "backend compatibility did not preserve the stored prompt fingerprint",
                    ));
                }
                validate_compatibility(&stored.compatibility, &current)?;
                let snapshot = BackendSnapshot::from_verified_file(
                    stored.payload_file.try_clone().map_err(|_| {
                        ProtocolError::new(
                            ProtocolErrorCode::CorruptState,
                            "runtime state payload handle is unavailable",
                        )
                    })?,
                    stored.payload_bytes,
                );
                let (resolution, requirement) = inner.adapter.inspect_persisted_state(
                    &manifest,
                    &snapshot,
                    &stored.compatibility,
                )?;
                validate_compatibility_envelope(&resolution.compatibility).map_err(|_| {
                    ProtocolError::new(
                        ProtocolErrorCode::Internal,
                        "backend inspected invalid dry-run persisted-state metadata",
                    )
                })?;
                validate_compatibility(&stored.compatibility, &resolution.compatibility)?;
                if resolution.scope == BackendPersistedStateScope::CrossRestart {
                    require_backend_capability(
                        &descriptor,
                        "runtime.state.restore.cross_restart",
                        request.allow_experimental,
                        true,
                    )?;
                }
                let requirement = requirement.ok_or_else(|| {
                    unavailable(
                        "the backend cannot bound restore memory, so promotion would not start",
                    )
                })?;
                if requirement.bytes == 0 || requirement.tier != target {
                    return Err(unsupported(
                        "backend restore cannot produce the requested target tier",
                    ));
                }
                let mut summary = stored.summary;
                summary.tier = target;
                summary.bytes = Some(requirement.bytes);
                return Ok(StateActionResponse {
                    state: summary,
                    changed: true,
                    dry_run: true,
                });
            }
            let (snapshot, plan) =
                prepare_persisted_snapshot(inner, &stored, request.allow_experimental)?;
            let descriptor = plan.descriptor().clone();
            require_state_tier_capability(&descriptor, target, request.allow_experimental)?;
            let requirement = plan.memory_requirement().ok_or_else(|| {
                unavailable("the backend cannot bound restore memory, so promotion was not started")
            })?;
            if requirement.tier != target {
                return Err(unsupported(
                    "backend restore cannot produce the requested target tier",
                ));
            }
            let mut summary = stored.summary;
            summary.tier = target;
            summary.bytes = Some(requirement.bytes);
            let pending = reserve_backend_memory(inner, Some(requirement), true)?;
            let restored = inner
                .adapter
                .restore_prepared_state(plan, snapshot, &stored.compatibility)
                .map_err(|error| backend_operation_error(inner, error))?;
            let (lease, allocation_id) =
                commit_backend_memory(inner, restored, pending, &stored.compatibility)?;
            summary.tier = lease.state().natural_tier();
            summary.bytes = lease.state().bytes();
            if let (Some(memory), Some(allocation_id)) = (&inner.memory, allocation_id) {
                memory
                    .set_pinned(allocation_id, summary.pinned)
                    .map_err(memory_error)?;
            }
            insert_volatile(
                inner,
                VolatileState {
                    principal_id: principal_id.to_string(),
                    summary: summary.clone(),
                    compatibility: stored.compatibility,
                    prompt_tokens: stored.prompt_tokens,
                    lease,
                    persisted: true,
                    allocation_id,
                },
            )?;
            Ok(StateActionResponse {
                state: summary,
                changed: true,
                dry_run: false,
            })
        }
        StateAction::Demote => Err(invalid("a disk state cannot be demoted further")),
    }
}

fn prune_states_inner(
    inner: &LocalInner,
    principal_id: &str,
    request: &PruneStatesRequest,
) -> ProtocolResult<PruneStatesResponse> {
    // Preview first so backend releases never occur after an invalid selector.
    // A real disk mutation is intentionally performed only after every live
    // state accepted its explicit release; an error therefore never reports
    // a false successful purge.
    let mut preview_request = request.clone();
    preview_request.dry_run = true;
    let preview = inner
        .state_store
        .prune_detailed(principal_id, &preview_request)?;
    let now = now_unix_ms();
    let (entries, expired, mut removed) = {
        let mut volatile = inner.volatile_states.lock().map_err(|_| internal())?;
        let expired = if request.dry_run {
            Vec::new()
        } else {
            take_expired_volatile_states(&mut volatile, Some(principal_id), now)
        };
        let entries = volatile
            .iter()
            .filter(|(_, state)| {
                state.principal_id == principal_id
                    && !volatile_state_is_expired(state, now)
                    && selector_matches(&request.selector, &state.summary)
            })
            .map(|(key, state)| (key.clone(), state.summary.clone()))
            .collect::<Vec<_>>();
        let mut removed = Vec::new();
        if !request.dry_run {
            for (key, _) in &entries {
                if let Some(state) = volatile.remove(key) {
                    removed.push(state);
                }
            }
        }
        (entries, expired, removed)
    };
    if !request.dry_run {
        // Expiry cleanup is part of this mutation too. Release those leases
        // explicitly so a backend failure cannot be hidden by `Drop` while
        // this operation goes on to report a successful prune. Process the
        // expired states first and restore every lease that has not yet been
        // released if any cleanup fails.
        removed.extend(expired);
        while let Some(mut state) = removed.pop() {
            if let Err(error) = release_volatile_state(&mut state) {
                let key = (state.principal_id.clone(), state.summary.id.clone());
                drop(replace_volatile_state(inner, key, state)?);
                while let Some(state) = removed.pop() {
                    let key = (state.principal_id.clone(), state.summary.id.clone());
                    drop(replace_volatile_state(inner, key, state)?);
                }
                return Err(error);
            }
        }
    }

    let disk = if request.dry_run {
        preview
    } else {
        inner.state_store.prune_detailed(principal_id, request)?
    };
    if !request.dry_run {
        let disk_ids = disk
            .matched_states
            .iter()
            .map(|state| state.id.as_str())
            .collect::<std::collections::HashSet<_>>();
        let mut volatile = inner.volatile_states.lock().map_err(|_| internal())?;
        for state in volatile.values_mut() {
            if state.principal_id == principal_id
                && state.persisted
                && disk_ids.contains(state.summary.id.as_str())
            {
                state.persisted = false;
            }
        }
    }
    let mut matched_states = BTreeMap::<String, Option<u64>>::new();
    for state in disk.matched_states {
        matched_states.insert(state.id, state.bytes);
    }
    // A volatile overlay is the currently visible representation and replaces
    // the disk byte count for the same opaque state ID.
    for (_, state) in entries {
        matched_states.insert(state.id, state.bytes);
    }
    let bytes = matched_states.values().try_fold(0_u64, |total, bytes| {
        bytes.map(|bytes| total.saturating_add(bytes))
    });
    let matched = matched_states.len() as u64;
    Ok(PruneStatesResponse {
        matched,
        removed: if request.dry_run { 0 } else { matched },
        bytes,
        dry_run: request.dry_run,
    })
}

fn volatile_state_is_expired(state: &VolatileState, now_unix_ms: u64) -> bool {
    state
        .summary
        .expires_unix_ms
        .is_some_and(|expires| expires <= now_unix_ms)
}

fn take_expired_volatile_states(
    states: &mut HashMap<(String, String), VolatileState>,
    principal_id: Option<&str>,
    now_unix_ms: u64,
) -> Vec<VolatileState> {
    let expired = states
        .iter()
        .filter_map(|(key, state)| {
            (principal_id.is_none_or(|principal| state.principal_id == principal)
                && volatile_state_is_expired(state, now_unix_ms)
                && state.lease.strong_count() == 1)
                .then_some(key.clone())
        })
        .collect::<Vec<_>>();
    expired
        .into_iter()
        .filter_map(|key| states.remove(&key))
        .collect()
}

fn replace_volatile_state(
    inner: &LocalInner,
    key: (String, String),
    state: VolatileState,
) -> ProtocolResult<Option<VolatileState>> {
    let mut states = inner.volatile_states.lock().map_err(|_| internal())?;
    let previous = states.insert(key, state);
    drop(states);
    Ok(previous)
}

fn take_volatile_state(
    inner: &LocalInner,
    key: &(String, String),
) -> ProtocolResult<Option<VolatileState>> {
    let mut states = inner.volatile_states.lock().map_err(|_| internal())?;
    let removed = states.remove(key);
    drop(states);
    Ok(removed)
}

fn release_volatile_state(state: &mut VolatileState) -> ProtocolResult<()> {
    // A live handoff is an independent owner. Removing the named volatile
    // overlay must not invalidate that handoff; its final lease drop performs
    // the backend release and accounting transition later.
    if state.lease.strong_count() > 1 {
        return Ok(());
    }

    if state.allocation_id.is_some() {
        let accounting = state.lease.release_backend_and_take_required_hook()?;
        accounting(true);
    } else if let Some(accounting) = state.lease.release_backend_and_take_hook()? {
        accounting(true);
    }
    Ok(())
}

fn selector_matches(selector: &StateSelector, state: &StateSummary) -> bool {
    match selector {
        StateSelector::Ids { ids } => ids.iter().any(|id| id == &state.id),
        StateSelector::Filter {
            model_id,
            tier,
            older_than_unix_ms,
        } => {
            model_id
                .as_deref()
                .is_none_or(|model| state.model_id == model)
                && tier.is_none_or(|tier| tier == state.tier)
                && older_than_unix_ms.is_none_or(|cutoff| state.last_accessed_unix_ms < cutoff)
        }
        StateSelector::All { confirm } => *confirm,
    }
}

fn validate_backend_state(
    inner: &LocalInner,
    state: &BackendState,
    compatibility: &crate::werk_protocol::CompatibilityEnvelope,
) -> ProtocolResult<()> {
    inner.adapter.validate_state(state, compatibility)?;
    validate_nonzero_backend_state(state)
}

fn validate_nonzero_backend_state(state: &BackendState) -> ProtocolResult<()> {
    if matches!(state.bytes(), Some(0)) {
        return Err(ProtocolError::new(
            ProtocolErrorCode::Internal,
            "backend returned an invalid zero-byte state measurement",
        ));
    }
    Ok(())
}

fn validate_prefill(request: &PrefillRequest, limits: &ProtocolLimits) -> ProtocolResult<()> {
    if request.model_id.trim().is_empty()
        || request.model_id.len() > 256
        || request.model_id.chars().any(char::is_control)
        || request.model_id.contains("..")
    {
        return Err(invalid("invalid model_id"));
    }
    match &request.input {
        PrefillInput::Text { text } => {
            if text.is_empty() || text.len() > MAX_PREFILL_BYTES {
                return Err(invalid(
                    "prefill text must contain between 1 and 524288 bytes",
                ));
            }
        }
        PrefillInput::Messages { messages } => {
            if messages.is_empty() || messages.len() > MAX_MESSAGES {
                return Err(invalid(
                    "prefill messages must contain between 1 and 256 items",
                ));
            }
            let mut total = 0usize;
            for message in messages {
                if message.role.is_empty()
                    || message.role.len() > 32
                    || !message
                        .role
                        .bytes()
                        .all(|byte| byte.is_ascii_lowercase() || byte == b'_' || byte == b'-')
                    || message.content.is_empty()
                {
                    return Err(invalid("prefill message role or content is invalid"));
                }
                total = total.saturating_add(message.content.len());
            }
            if total > MAX_PREFILL_BYTES {
                return Err(invalid("prefill message content exceeds 524288 bytes"));
            }
        }
    }
    if let Some(ttl) = request.policy.ttl_seconds
        && (ttl == 0 || ttl > limits.max_ttl_seconds)
    {
        return Err(invalid(format!(
            "ttl_seconds must be between 1 and {}",
            limits.max_ttl_seconds
        )));
    }
    Ok(())
}

fn validate_decode(request: &DecodeRequest) -> ProtocolResult<()> {
    if request.handoff.len() < 32 || request.handoff.len() > 4096 {
        return Err(ProtocolError::new(
            ProtocolErrorCode::ExpiredHandoff,
            "handoff is invalid or expired",
        ));
    }
    if request.max_tokens == 0 || request.max_tokens > MAX_DECODE_TOKENS {
        return Err(invalid(format!(
            "max_tokens must be between 1 and {MAX_DECODE_TOKENS}"
        )));
    }
    if request
        .temperature
        .is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value))
        || request
            .top_p
            .is_some_and(|value| !value.is_finite() || !(0.0 < value && value <= 1.0))
    {
        return Err(invalid(
            "decode sampling parameters are outside their bounds",
        ));
    }
    if request.stop.len() > 16
        || request
            .stop
            .iter()
            .any(|value| value.is_empty() || value.len() > 1024)
    {
        return Err(invalid(
            "stop must contain at most 16 non-empty bounded strings",
        ));
    }
    Ok(())
}

fn validate_expert_filter(filter: &ExpertListFilter, max_page_size: u16) -> ProtocolResult<()> {
    if filter.model_id.as_deref().is_some_and(|id| {
        id.trim().is_empty()
            || id.len() > 256
            || id.chars().any(char::is_control)
            || id.contains("..")
    }) {
        return Err(invalid("invalid expert model_id filter"));
    }
    if filter
        .limit
        .is_some_and(|limit| limit == 0 || limit > max_page_size)
    {
        return Err(invalid(format!(
            "expert limit must be between 1 and {max_page_size}"
        )));
    }
    if filter
        .cursor
        .as_deref()
        .is_some_and(|cursor| cursor.len() > 256)
    {
        return Err(invalid("expert cursor is too long"));
    }
    Ok(())
}

fn validate_expert_list_response(
    response: &ExpertListResponse,
    filter: &ExpertListFilter,
    max_page_size: u16,
) -> ProtocolResult<()> {
    let maximum = usize::from(filter.limit.unwrap_or(max_page_size));
    if response.experts.len() > maximum {
        return Err(invalid_backend_expert_response());
    }
    validate_expert_summaries(&response.experts, filter.model_id.as_deref(), filter.tier)?;
    if response.next_cursor.as_deref().is_some_and(|cursor| {
        cursor.is_empty() || cursor.len() > 256 || cursor.chars().any(char::is_control)
    }) {
        return Err(invalid_backend_expert_response());
    }
    Ok(())
}

fn validate_expert_action_response(
    response: &ExpertActionResponse,
    request: &ExpertActionRequest,
) -> ProtocolResult<()> {
    if response.dry_run != request.dry_run
        || response.experts.len() > request.expert_ids.len()
        || response.changed > response.experts.len() as u64
    {
        return Err(invalid_backend_expert_response());
    }
    validate_expert_summaries(&response.experts, Some(&request.model_id), None)?;
    let requested = request
        .expert_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if response
        .experts
        .iter()
        .any(|expert| !requested.contains(expert.id.as_str()))
    {
        return Err(invalid_backend_expert_response());
    }
    Ok(())
}

fn validate_expert_summaries(
    experts: &[ExpertSummary],
    model_id: Option<&str>,
    tier: Option<crate::werk_protocol::ExpertTier>,
) -> ProtocolResult<()> {
    let mut ids = BTreeSet::new();
    for expert in experts {
        if !valid_opaque_id(&expert.id)
            || expert.model_id.trim().is_empty()
            || expert.model_id.len() > 256
            || expert.model_id.chars().any(char::is_control)
            || model_id.is_some_and(|model| expert.model_id != model)
            || tier.is_some_and(|expected| expert.tier != expected)
            || expert.bytes == Some(0)
            || !expert.hotness.is_finite()
            || expert.hotness < 0.0
            || !ids.insert(expert.id.as_str())
        {
            return Err(invalid_backend_expert_response());
        }
    }
    Ok(())
}

fn invalid_backend_expert_response() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::Internal,
        "backend returned an invalid or unbounded expert response",
    )
}

fn validate_state_identifier(id: &str) -> ProtocolResult<()> {
    if !id.starts_with("st_")
        || id.len() > 128
        || id.contains("..")
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(invalid("invalid state ID"));
    }
    Ok(())
}

fn random_state_id() -> ProtocolResult<String> {
    let mut random = [0u8; 18];
    getrandom::getrandom(&mut random)
        .map_err(|_| unavailable("secure runtime state identity generation is unavailable"))?;
    Ok(format!("st_{}", URL_SAFE_NO_PAD.encode(random)))
}

fn encode_state_cursor(id: &str) -> String {
    URL_SAFE_NO_PAD.encode(id.as_bytes())
}

fn decode_state_cursor(cursor: &str) -> ProtocolResult<String> {
    if cursor.len() > 256 {
        return Err(invalid("invalid cursor"));
    }
    let bytes = URL_SAFE_NO_PAD
        .decode(cursor)
        .map_err(|_| invalid("invalid cursor"))?;
    let id = String::from_utf8(bytes).map_err(|_| invalid("invalid cursor"))?;
    validate_state_identifier(&id)?;
    Ok(id)
}

#[derive(Debug)]
struct SystemRuntimeMemoryTelemetry {
    topology: MemoryTopology,
    accelerator: Option<String>,
    include_accelerator: bool,
}

impl MemoryTelemetry for SystemRuntimeMemoryTelemetry {
    fn observe(&self) -> Result<Vec<MemoryObservation>, MemoryError> {
        let (host_total, host_available) = host_memory_observation()
            .and_then(|(total, available)| available.map(|available| (total, available)))
            .ok_or_else(|| {
                MemoryError::InvalidTelemetry(
                    "host total and available memory could not be sampled".to_string(),
                )
            })?;
        let mut observations = vec![MemoryObservation {
            tier: MemoryTier::Host,
            total_bytes: host_total,
            available_bytes: host_available,
        }];
        if self.include_accelerator {
            let (total_bytes, available_bytes) = if self.topology == MemoryTopology::Unified {
                (host_total, host_available)
            } else {
                accelerator_memory_observation(self.accelerator.as_deref())
                    .and_then(|(total, available)| available.map(|available| (total, available)))
                    .ok_or_else(|| {
                        MemoryError::InvalidTelemetry(
                            "accelerator total and available memory could not be sampled"
                                .to_string(),
                        )
                    })?
            };
            observations.push(MemoryObservation {
                tier: MemoryTier::Accelerator(0),
                total_bytes,
                available_bytes,
            });
        }
        Ok(observations)
    }
}

fn build_system_memory_manager() -> Option<MemoryManager> {
    let resources = detect_host_resources();
    let topology = match resources.memory_topology {
        Some(InferenceMemoryTopology::Unified) => MemoryTopology::Unified,
        Some(InferenceMemoryTopology::Discrete | InferenceMemoryTopology::Unknown) | None => {
            MemoryTopology::Discrete
        }
    };
    let accelerator = resources.accelerator.clone();
    let accelerator_name = accelerator.as_deref().unwrap_or_default();
    let candidate_has_accelerator =
        !accelerator_name.is_empty() && accelerator_name != "cpu" && accelerator_name != "unknown";
    let candidate = SystemRuntimeMemoryTelemetry {
        topology,
        accelerator: accelerator.clone(),
        include_accelerator: candidate_has_accelerator,
    };
    let include_accelerator = candidate_has_accelerator
        && candidate
            .observe()
            .is_ok_and(|observations| observations.len() == 2);
    let telemetry: Arc<dyn MemoryTelemetry> = Arc::new(SystemRuntimeMemoryTelemetry {
        topology,
        accelerator,
        include_accelerator,
    });
    let observations = telemetry.observe().ok()?;
    let budgets = observations
        .iter()
        .map(|observation| TierBudget::new(observation.tier, observation.total_bytes))
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    let config = MemoryManagerConfig::new(
        budgets,
        topology,
        PressureThresholds::default(),
        MEMORY_ACTION_COOLDOWN_MILLIS,
        MAX_MEMORY_ALLOCATIONS,
        MAX_MEMORY_RESERVATIONS,
        MAX_MEMORY_ACTIONS_PER_CYCLE,
    )
    .ok()?;
    Some(MemoryManager::new(
        config,
        telemetry,
        Arc::new(SystemMemoryClock::new()),
    ))
}

fn memory_status_for(inner: &LocalInner) -> MemoryStatusResponse {
    let Some(manager) = &inner.memory else {
        let mut status = system_memory_status();
        append_backend_cleanup_counters(inner, &mut status.counters);
        return status;
    };
    let (snapshot, telemetry_available) = match manager.refresh() {
        Ok(snapshot) => (snapshot, true),
        Err(_) => (manager.accounting_snapshot(), false),
    };
    let mut host = MemoryTierStatus {
        capacity_bytes: None,
        available_bytes: None,
        managed_bytes: 0,
        reserved_bytes: 0,
        pressure: PressureLevel::Unknown,
    };
    let mut accelerator = host.clone();
    let mut active_reservations = 0_u64;
    let mut managed_allocations = 0_u64;
    for tier in snapshot.tiers {
        let status = MemoryTierStatus {
            capacity_bytes: telemetry_available.then_some(tier.total_bytes),
            available_bytes: telemetry_available.then_some(tier.available_bytes),
            managed_bytes: tier.managed_used_bytes,
            reserved_bytes: tier.reserved_bytes,
            pressure: if telemetry_available {
                protocol_pressure(tier.pressure)
            } else {
                PressureLevel::Unknown
            },
        };
        active_reservations = active_reservations.saturating_add(tier.active_reservations as u64);
        managed_allocations = managed_allocations.saturating_add(tier.managed_allocations as u64);
        match tier.tier {
            MemoryTier::Host => host = status,
            MemoryTier::Accelerator(0) => accelerator = status,
            MemoryTier::Accelerator(_) => {}
        }
    }
    let mut counters = BTreeMap::new();
    counters.insert("active_reservations".to_string(), active_reservations);
    counters.insert("managed_allocations".to_string(), managed_allocations);
    counters.insert(
        "pressure_actions_in_flight".to_string(),
        snapshot.actions_in_flight as u64,
    );
    counters.insert(
        "completed_demotions".to_string(),
        snapshot.completed_demotions,
    );
    counters.insert(
        "completed_evictions".to_string(),
        snapshot.completed_evictions,
    );
    counters.insert("failed_releases".to_string(), snapshot.failed_releases);
    counters.insert(
        "orphaned_release_bytes".to_string(),
        snapshot.orphaned_release_bytes,
    );
    if !telemetry_available {
        counters.insert("telemetry_errors".to_string(), 1);
    }
    append_backend_cleanup_counters(inner, &mut counters);
    MemoryStatusResponse {
        observed_at_unix_ms: snapshot.observed_at_unix_millis,
        overall_pressure: max_pressure(host.pressure, accelerator.pressure),
        topology: match snapshot.topology {
            MemoryTopology::Discrete => "discrete",
            MemoryTopology::Unified => "unified",
        }
        .to_string(),
        host,
        accelerator,
        last_action_unix_ms: snapshot.last_action_unix_millis,
        counters,
    }
}

fn append_backend_cleanup_counters(inner: &LocalInner, counters: &mut BTreeMap<String, u64>) {
    counters.insert(
        "backend_cleanup_failures".to_string(),
        inner.backend_cleanup_failures.load(Ordering::Relaxed),
    );
    counters.insert(
        "backend_cleanup_latched".to_string(),
        u64::from(inner.backend_cleanup_latched.load(Ordering::Acquire)),
    );
}

fn protocol_pressure(level: super::memory::PressureLevel) -> PressureLevel {
    match level {
        super::memory::PressureLevel::Normal => PressureLevel::Normal,
        super::memory::PressureLevel::Soft => PressureLevel::Soft,
        super::memory::PressureLevel::Hard => PressureLevel::Hard,
        super::memory::PressureLevel::Emergency => PressureLevel::Emergency,
    }
}

fn system_memory_status() -> MemoryStatusResponse {
    let resources = detect_host_resources();
    let host_observation = host_memory_observation();
    let topology = match resources.memory_topology {
        Some(InferenceMemoryTopology::Unified) => "unified",
        Some(InferenceMemoryTopology::Discrete) => "discrete",
        Some(InferenceMemoryTopology::Unknown) | None => "unknown",
    }
    .to_string();
    let accelerator_observation = if topology == "unified" {
        host_observation
    } else {
        accelerator_memory_observation(resources.accelerator.as_deref()).or_else(|| {
            resources
                .accelerator_memory_bytes
                .map(|total| (total, None))
        })
    };
    let host = memory_tier_status(host_observation);
    let accelerator = memory_tier_status(accelerator_observation);
    MemoryStatusResponse {
        observed_at_unix_ms: now_unix_ms(),
        overall_pressure: max_pressure(host.pressure, accelerator.pressure),
        topology,
        host,
        accelerator,
        last_action_unix_ms: None,
        counters: BTreeMap::new(),
    }
}

fn memory_tier_status(observation: Option<(u64, Option<u64>)>) -> MemoryTierStatus {
    let (capacity_bytes, available_bytes) = observation
        .map(|(capacity, available)| (Some(capacity), available))
        .unwrap_or((None, None));
    MemoryTierStatus {
        capacity_bytes,
        available_bytes,
        managed_bytes: 0,
        reserved_bytes: 0,
        pressure: pressure(capacity_bytes, available_bytes),
    }
}

fn pressure(total: Option<u64>, available: Option<u64>) -> PressureLevel {
    let (Some(total), Some(available)) = (total, available) else {
        return PressureLevel::Unknown;
    };
    if total == 0 || available > total {
        return PressureLevel::Unknown;
    }
    let used_basis_points =
        u128::from(total.saturating_sub(available)).saturating_mul(10_000) / u128::from(total);
    if used_basis_points >= 9_500 {
        PressureLevel::Emergency
    } else if used_basis_points >= 8_500 {
        PressureLevel::Hard
    } else if used_basis_points >= 7_500 {
        PressureLevel::Soft
    } else {
        PressureLevel::Normal
    }
}

fn max_pressure(left: PressureLevel, right: PressureLevel) -> PressureLevel {
    fn rank(level: PressureLevel) -> u8 {
        match level {
            PressureLevel::Unknown => 0,
            PressureLevel::Normal => 1,
            PressureLevel::Soft => 2,
            PressureLevel::Hard => 3,
            PressureLevel::Emergency => 4,
        }
    }
    if rank(left) >= rank(right) {
        left
    } else {
        right
    }
}

#[cfg(unix)]
fn host_memory_observation() -> Option<(u64, Option<u64>)> {
    // SAFETY: sysconf has no pointer arguments and is safe to query. Negative
    // values are treated as unavailable rather than cast.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    // SAFETY: same as above.
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    // SAFETY: same as above.
    let available_pages = unsafe { libc::sysconf(libc::_SC_AVPHYS_PAGES) };
    if page_size <= 0 || pages <= 0 {
        return None;
    }
    let total = (page_size as u64).saturating_mul(pages as u64);
    let available =
        (available_pages > 0).then(|| (page_size as u64).saturating_mul(available_pages as u64));
    Some((total, available))
}

#[cfg(not(unix))]
fn host_memory_observation() -> Option<(u64, Option<u64>)> {
    None
}

fn accelerator_memory_observation(accelerator: Option<&str>) -> Option<(u64, Option<u64>)> {
    let accelerator = accelerator?.to_ascii_lowercase();
    if !accelerator.contains("cuda") {
        return None;
    }
    let stdout = bounded_command_stdout(
        "nvidia-smi",
        &[
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        Duration::from_secs(2),
        64 * 1024,
    )?;
    let line = String::from_utf8(stdout).ok()?.lines().next()?.to_string();
    let mut values = line.split(',').map(str::trim);
    let total_mib = values.next()?.parse::<u64>().ok()?;
    let free_mib = values.next()?.parse::<u64>().ok()?;
    let mib = 1024 * 1024;
    Some((
        total_mib.saturating_mul(mib),
        Some(free_mib.saturating_mul(mib)),
    ))
}

fn bounded_command_stdout(
    program: &str,
    arguments: &[&str],
    timeout: Duration,
    max_stdout_bytes: usize,
) -> Option<Vec<u8>> {
    let mut child = Command::new(program)
        .args(arguments)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait().ok()? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    if !status.success() {
        return None;
    }
    let mut stdout = child.stdout.take()?;
    let mut bytes = Vec::new();
    stdout
        .by_ref()
        .take((max_stdout_bytes + 1) as u64)
        .read_to_end(&mut bytes)
        .ok()?;
    (bytes.len() <= max_stdout_bytes).then_some(bytes)
}

fn invalid(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::InvalidRequest, message)
}

fn unsupported(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::Unsupported, message)
}

fn unavailable(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::Unavailable, message).retryable(true)
}

fn internal() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::Internal,
        "runtime control operation failed",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model_store::{ModelFormat, ModelManifest, ModelMetadata, ModelSource},
        runtime_control::{
            AUTOMATIC_REUSE_OPERATION, BackendDecodeResult, BackendPersistedStateResolution,
            BackendPersistedStateScope, BackendPrefillResult, BackendRuntimeDescriptor,
            MODEL_RESIDENCY_CAPABILITY, ModelResidencyStatus, StaticRuntimeAdapter,
        },
        werk_protocol::{CompatibilityEnvelope, ContextCompatibility, ProtocolMessage},
    };
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicBool, AtomicU64, AtomicUsize},
        time::{SystemTime, UNIX_EPOCH},
    };

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(1);

    #[derive(Clone)]
    struct FakeAdapter {
        incompatible: bool,
        restore_calls: Arc<AtomicUsize>,
        resolution_calls: Arc<AtomicUsize>,
    }

    impl FakeAdapter {
        fn new() -> Self {
            Self {
                incompatible: false,
                restore_calls: Arc::new(AtomicUsize::new(0)),
                resolution_calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl BackendRuntimeAdapter for FakeAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            BackendRuntimeDescriptor {
                backend: "fake".to_string(),
                backend_version: if self.incompatible { "2" } else { "1" }.to_string(),
                adapter_version: "1".to_string(),
                accelerator_family: "cpu".to_string(),
                instance_id: "fake-process".to_string(),
                capabilities: [
                    "runtime.state.persistence",
                    "runtime.state.restore",
                    "runtime.state.restore.cross_restart",
                    "runtime.state.tier.disk",
                    "runtime.state.tier.ram",
                    "runtime.pd.prefill",
                    "runtime.pd.decode",
                    "runtime.pd.handoff",
                    "runtime.memory.reservations",
                ]
                .into_iter()
                .map(|id| Capability {
                    id: id.to_string(),
                    status: CapabilityStatus::Supported,
                    detail: "deterministic fake".to_string(),
                    operations: vec!["test".to_string()],
                })
                .collect(),
            }
        }

        fn compatibility(
            &self,
            _manifest: &crate::model_store::ModelManifest,
            prompt_fingerprint: &str,
        ) -> ProtocolResult<CompatibilityEnvelope> {
            Ok(CompatibilityEnvelope {
                model_fingerprint: "sha256:model".to_string(),
                tokenizer_fingerprint: "sha256:tokenizer".to_string(),
                prompt_fingerprint: prompt_fingerprint.to_string(),
                chat_template_fingerprint: Some("sha256:template".to_string()),
                backend: "fake".to_string(),
                backend_version: if self.incompatible { "2" } else { "1" }.to_string(),
                runtime_adapter_version: "1".to_string(),
                accelerator_family: "cpu".to_string(),
                tensor_dtype: "f32".to_string(),
                kv_dtype: "f32".to_string(),
                quantization: "none".to_string(),
                cache_layout: "fake-v1".to_string(),
                block_size: None,
                context: ContextCompatibility {
                    context_size: 4096,
                    batch_size: Some(512),
                    rope_configuration_fingerprint: None,
                },
                multimodal_processor_fingerprints: Vec::new(),
                producer_protocol: ProtocolVersion::V1,
            })
        }

        fn inspect_compatibility(
            &self,
            manifest: &ModelManifest,
            prompt_fingerprint: &str,
        ) -> ProtocolResult<CompatibilityEnvelope> {
            self.compatibility(manifest, prompt_fingerprint)
        }

        fn resolve_persisted_state(
            &self,
            manifest: &ModelManifest,
            _snapshot: &BackendSnapshot,
            expected: &CompatibilityEnvelope,
        ) -> ProtocolResult<BackendPersistedStateResolution> {
            self.resolution_calls.fetch_add(1, Ordering::SeqCst);
            Ok(BackendPersistedStateResolution {
                compatibility: self.compatibility(manifest, &expected.prompt_fingerprint)?,
                scope: BackendPersistedStateScope::CrossRestart,
            })
        }

        fn inspect_persisted_state(
            &self,
            manifest: &ModelManifest,
            snapshot: &BackendSnapshot,
            expected: &CompatibilityEnvelope,
        ) -> ProtocolResult<(
            BackendPersistedStateResolution,
            Option<BackendMemoryRequirement>,
        )> {
            Ok((
                BackendPersistedStateResolution {
                    compatibility: self.compatibility(manifest, &expected.prompt_fingerprint)?,
                    scope: BackendPersistedStateScope::CrossRestart,
                },
                self.restore_memory_requirement(snapshot, expected)?,
            ))
        }

        fn prefill(
            &self,
            _request: super::super::BackendPrefillRequest,
        ) -> ProtocolResult<BackendPrefillResult> {
            Ok(BackendPrefillResult {
                state: BackendState::OpaqueBytes {
                    bytes: Arc::from(b"opaque-kv".as_slice()),
                    tier: StateTier::Ram,
                    instance_id: "fake-process".to_string(),
                },
                prompt_tokens: 4,
                reused: false,
            })
        }

        fn prefill_memory_requirement(
            &self,
            _request: &BackendPrefillRequest,
        ) -> ProtocolResult<Option<BackendMemoryRequirement>> {
            Ok(Some(BackendMemoryRequirement {
                tier: StateTier::Ram,
                bytes: 9,
                demotion_target: None,
            }))
        }

        fn restore_memory_requirement(
            &self,
            snapshot: &BackendSnapshot,
            _compatibility: &CompatibilityEnvelope,
        ) -> ProtocolResult<Option<BackendMemoryRequirement>> {
            Ok(Some(BackendMemoryRequirement {
                tier: StateTier::Ram,
                bytes: snapshot.bytes,
                demotion_target: None,
            }))
        }

        fn restore(
            &self,
            snapshot: BackendSnapshot,
            _compatibility: &CompatibilityEnvelope,
        ) -> ProtocolResult<BackendState> {
            self.restore_calls.fetch_add(1, Ordering::SeqCst);
            let valid = snapshot
                .try_clone_file()
                .and_then(|file| file.metadata())
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() == snapshot.bytes);
            if !valid {
                return Err(ProtocolError::new(
                    ProtocolErrorCode::CorruptState,
                    "fake snapshot failed validation",
                ));
            }
            Ok(BackendState::InProcess {
                handle: "restored".to_string(),
                bytes: Some(snapshot.bytes),
                tier: StateTier::Ram,
                instance_id: "fake-process".to_string(),
            })
        }

        fn snapshot(&self, _state: &BackendState) -> ProtocolResult<BackendState> {
            Ok(BackendState::OpaqueBytes {
                bytes: Arc::from(b"opaque-kv".as_slice()),
                tier: StateTier::Disk,
                instance_id: "fake-process".to_string(),
            })
        }

        fn inspect_snapshot_export(
            &self,
            state: &BackendState,
            compatibility: &CompatibilityEnvelope,
        ) -> ProtocolResult<()> {
            self.validate_state(state, compatibility)
        }

        fn decode(&self, request: BackendDecodeRequest) -> ProtocolResult<BackendDecodeResult> {
            assert_eq!(request.state.state().instance_id(), "fake-process");
            Ok(BackendDecodeResult {
                text: "decoded".to_string(),
                state: None,
                completion_tokens: 1,
                finish_reason: "stop".to_string(),
            })
        }
    }

    #[derive(Clone)]
    struct ReservationObservingAdapter {
        inner: FakeAdapter,
        memory: MemoryManager,
        saw_reservation: Arc<AtomicBool>,
    }

    impl BackendRuntimeAdapter for ReservationObservingAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            self.inner.descriptor()
        }

        fn compatibility(
            &self,
            manifest: &ModelManifest,
            prompt_fingerprint: &str,
        ) -> ProtocolResult<CompatibilityEnvelope> {
            self.inner.compatibility(manifest, prompt_fingerprint)
        }

        fn prefill_memory_requirement(
            &self,
            request: &BackendPrefillRequest,
        ) -> ProtocolResult<Option<BackendMemoryRequirement>> {
            self.inner.prefill_memory_requirement(request)
        }

        fn prefill(&self, request: BackendPrefillRequest) -> ProtocolResult<BackendPrefillResult> {
            let snapshot = self.memory.refresh().map_err(memory_error)?;
            self.saw_reservation.store(
                snapshot
                    .tiers
                    .iter()
                    .any(|tier| tier.tier == MemoryTier::Host && tier.reserved_bytes == 9),
                Ordering::SeqCst,
            );
            self.inner.prefill(request)
        }

        fn decode(&self, request: BackendDecodeRequest) -> ProtocolResult<BackendDecodeResult> {
            self.inner.decode(request)
        }
    }

    #[derive(Clone)]
    struct ReplacementDecodeAdapter {
        inner: FakeAdapter,
        declare_requirement: bool,
        releases: Arc<AtomicUsize>,
    }

    impl BackendRuntimeAdapter for ReplacementDecodeAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            self.inner.descriptor()
        }

        fn compatibility(
            &self,
            manifest: &ModelManifest,
            prompt_fingerprint: &str,
        ) -> ProtocolResult<CompatibilityEnvelope> {
            self.inner.compatibility(manifest, prompt_fingerprint)
        }

        fn prefill_memory_requirement(
            &self,
            request: &BackendPrefillRequest,
        ) -> ProtocolResult<Option<BackendMemoryRequirement>> {
            self.inner.prefill_memory_requirement(request)
        }

        fn prefill(&self, request: BackendPrefillRequest) -> ProtocolResult<BackendPrefillResult> {
            self.inner.prefill(request)
        }

        fn decode_memory_requirement(
            &self,
            _request: &BackendDecodeRequest,
        ) -> ProtocolResult<Option<BackendMemoryRequirement>> {
            Ok(self
                .declare_requirement
                .then_some(BackendMemoryRequirement {
                    tier: StateTier::Ram,
                    bytes: 9,
                    demotion_target: None,
                }))
        }

        fn decode(&self, request: BackendDecodeRequest) -> ProtocolResult<BackendDecodeResult> {
            assert_eq!(request.state.state().instance_id(), "fake-process");
            Ok(BackendDecodeResult {
                text: "decoded".to_string(),
                state: Some(BackendState::OpaqueBytes {
                    bytes: Arc::from(b"opaque-kv".as_slice()),
                    tier: StateTier::Ram,
                    instance_id: "fake-process".to_string(),
                }),
                completion_tokens: 1,
                finish_reason: "length".to_string(),
            })
        }

        fn release(&self, _state: &BackendState) -> ProtocolResult<()> {
            self.releases.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[derive(Clone)]
    struct FailingReleaseAdapter {
        inner: FakeAdapter,
        attempts: Arc<AtomicUsize>,
    }

    impl BackendRuntimeAdapter for FailingReleaseAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            self.inner.descriptor()
        }

        fn release(&self, _state: &BackendState) -> ProtocolResult<()> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            Err(ProtocolError::new(
                ProtocolErrorCode::Unavailable,
                "deterministic release failure",
            ))
        }
    }

    #[derive(Clone)]
    struct PruneReleaseAdapter {
        inner: FakeAdapter,
        attempts: Arc<AtomicUsize>,
        fail_on_attempt: Option<usize>,
    }

    impl BackendRuntimeAdapter for PruneReleaseAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            self.inner.descriptor()
        }

        fn compatibility(
            &self,
            manifest: &ModelManifest,
            prompt_fingerprint: &str,
        ) -> ProtocolResult<CompatibilityEnvelope> {
            self.inner.compatibility(manifest, prompt_fingerprint)
        }

        fn release(&self, _state: &BackendState) -> ProtocolResult<()> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_on_attempt == Some(attempt) {
                return Err(ProtocolError::new(
                    ProtocolErrorCode::Unavailable,
                    "deterministic prune release failure",
                ));
            }
            Ok(())
        }
    }

    #[derive(Clone)]
    struct SnapshotRejectingAdapter {
        inner: FakeAdapter,
        snapshot_calls: Arc<AtomicUsize>,
        validation_calls: Arc<AtomicUsize>,
    }

    impl BackendRuntimeAdapter for SnapshotRejectingAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            self.inner.descriptor()
        }

        fn compatibility(
            &self,
            manifest: &ModelManifest,
            prompt_fingerprint: &str,
        ) -> ProtocolResult<CompatibilityEnvelope> {
            self.inner.compatibility(manifest, prompt_fingerprint)
        }

        fn prefill_memory_requirement(
            &self,
            request: &BackendPrefillRequest,
        ) -> ProtocolResult<Option<BackendMemoryRequirement>> {
            self.inner.prefill_memory_requirement(request)
        }

        fn prefill(&self, request: BackendPrefillRequest) -> ProtocolResult<BackendPrefillResult> {
            self.inner.prefill(request)
        }

        fn snapshot(&self, state: &BackendState) -> ProtocolResult<BackendState> {
            self.snapshot_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.snapshot(state)
        }

        fn validate_snapshot(
            &self,
            _snapshot: &BackendState,
            _compatibility: &CompatibilityEnvelope,
        ) -> ProtocolResult<()> {
            self.validation_calls.fetch_add(1, Ordering::SeqCst);
            Err(ProtocolError::new(
                ProtocolErrorCode::IncompatibleState,
                "deterministic snapshot-contract rejection",
            ))
        }
    }

    #[derive(Clone)]
    struct MissingOpaqueFileAdapter {
        inner: FakeAdapter,
        missing_path: PathBuf,
    }

    impl BackendRuntimeAdapter for MissingOpaqueFileAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            self.inner.descriptor()
        }

        fn compatibility(
            &self,
            manifest: &ModelManifest,
            prompt_fingerprint: &str,
        ) -> ProtocolResult<CompatibilityEnvelope> {
            self.inner.compatibility(manifest, prompt_fingerprint)
        }

        fn prefill_memory_requirement(
            &self,
            request: &BackendPrefillRequest,
        ) -> ProtocolResult<Option<BackendMemoryRequirement>> {
            self.inner.prefill_memory_requirement(request)
        }

        fn prefill(&self, _request: BackendPrefillRequest) -> ProtocolResult<BackendPrefillResult> {
            Ok(BackendPrefillResult {
                state: BackendState::OpaqueFile {
                    path: self.missing_path.clone(),
                    bytes: 9,
                    tier: StateTier::Ram,
                    instance_id: "fake-process".to_string(),
                },
                prompt_tokens: 4,
                reused: false,
            })
        }

        fn validate_snapshot(
            &self,
            snapshot: &BackendState,
            compatibility: &CompatibilityEnvelope,
        ) -> ProtocolResult<()> {
            self.validate_state(snapshot, compatibility)
        }
    }

    #[derive(Clone)]
    struct ContractBreakingAdapter {
        inner: FakeAdapter,
        prefill_tier: StateTier,
        restore_tier: StateTier,
        fail_release: bool,
        releases: Arc<AtomicUsize>,
        restore_calls: Arc<AtomicUsize>,
    }

    impl BackendRuntimeAdapter for ContractBreakingAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            self.inner.descriptor()
        }

        fn compatibility(
            &self,
            manifest: &ModelManifest,
            prompt_fingerprint: &str,
        ) -> ProtocolResult<CompatibilityEnvelope> {
            self.inner.compatibility(manifest, prompt_fingerprint)
        }

        fn resolve_persisted_state(
            &self,
            manifest: &ModelManifest,
            snapshot: &BackendSnapshot,
            expected: &CompatibilityEnvelope,
        ) -> ProtocolResult<BackendPersistedStateResolution> {
            self.inner
                .resolve_persisted_state(manifest, snapshot, expected)
        }

        fn prefill(&self, _request: BackendPrefillRequest) -> ProtocolResult<BackendPrefillResult> {
            Ok(BackendPrefillResult {
                state: BackendState::OpaqueBytes {
                    bytes: Arc::from(b"opaque-kv".as_slice()),
                    tier: self.prefill_tier,
                    instance_id: "fake-process".to_string(),
                },
                prompt_tokens: 4,
                reused: false,
            })
        }

        fn restore(
            &self,
            snapshot: BackendSnapshot,
            _compatibility: &CompatibilityEnvelope,
        ) -> ProtocolResult<BackendState> {
            self.restore_calls.fetch_add(1, Ordering::SeqCst);
            Ok(BackendState::InProcess {
                handle: "unreserved-restore".to_string(),
                bytes: Some(snapshot.bytes),
                tier: self.restore_tier,
                instance_id: "fake-process".to_string(),
            })
        }

        fn release(&self, _state: &BackendState) -> ProtocolResult<()> {
            self.releases.fetch_add(1, Ordering::SeqCst);
            if self.fail_release {
                Err(ProtocolError::new(
                    ProtocolErrorCode::Unavailable,
                    "deterministic rejected-state cleanup failure",
                ))
            } else {
                Ok(())
            }
        }
    }

    #[derive(Clone)]
    struct ExperimentalPrefillAdapter {
        inner: FakeAdapter,
        compatibility_calls: Arc<AtomicUsize>,
    }

    #[derive(Clone, Copy)]
    enum InvalidCompatibilityKind {
        WrongPrompt,
        TooManyProcessors,
    }

    #[derive(Clone)]
    struct InvalidCompatibilityAdapter {
        inner: FakeAdapter,
        kind: InvalidCompatibilityKind,
        prefill_calls: Arc<AtomicUsize>,
    }

    impl BackendRuntimeAdapter for InvalidCompatibilityAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            self.inner.descriptor()
        }

        fn compatibility(
            &self,
            manifest: &ModelManifest,
            prompt_fingerprint: &str,
        ) -> ProtocolResult<CompatibilityEnvelope> {
            let mut compatibility = self.inner.compatibility(manifest, prompt_fingerprint)?;
            match self.kind {
                InvalidCompatibilityKind::WrongPrompt => {
                    compatibility.prompt_fingerprint = "sha256:constant".to_string();
                }
                InvalidCompatibilityKind::TooManyProcessors => {
                    compatibility.multimodal_processor_fingerprints =
                        (0..65).map(|index| format!("sha256:mm-{index}")).collect();
                }
            }
            Ok(compatibility)
        }

        fn prefill(&self, request: BackendPrefillRequest) -> ProtocolResult<BackendPrefillResult> {
            self.prefill_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.prefill(request)
        }
    }

    impl BackendRuntimeAdapter for ExperimentalPrefillAdapter {
        fn descriptor(&self) -> BackendRuntimeDescriptor {
            let mut descriptor = self.inner.descriptor();
            for capability in &mut descriptor.capabilities {
                capability.status = CapabilityStatus::Experimental;
            }
            descriptor
        }

        fn compatibility(
            &self,
            manifest: &ModelManifest,
            prompt_fingerprint: &str,
        ) -> ProtocolResult<CompatibilityEnvelope> {
            self.compatibility_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.compatibility(manifest, prompt_fingerprint)
        }

        fn prefill_memory_requirement(
            &self,
            request: &BackendPrefillRequest,
        ) -> ProtocolResult<Option<BackendMemoryRequirement>> {
            self.inner.prefill_memory_requirement(request)
        }

        fn prefill(&self, request: BackendPrefillRequest) -> ProtocolResult<BackendPrefillResult> {
            self.inner.prefill(request)
        }
    }

    struct TestControl {
        root: PathBuf,
        control: LocalWerkControl,
    }

    #[derive(Debug)]
    struct TestMemoryTelemetry;

    impl MemoryTelemetry for TestMemoryTelemetry {
        fn observe(&self) -> Result<Vec<MemoryObservation>, MemoryError> {
            Ok(vec![MemoryObservation {
                tier: MemoryTier::Host,
                total_bytes: 1024 * 1024,
                available_bytes: 1024 * 1024,
            }])
        }
    }

    #[derive(Debug)]
    struct MutableMemoryTelemetry {
        available: Arc<AtomicU64>,
    }

    impl MemoryTelemetry for MutableMemoryTelemetry {
        fn observe(&self) -> Result<Vec<MemoryObservation>, MemoryError> {
            Ok(vec![MemoryObservation {
                tier: MemoryTier::Host,
                total_bytes: 1024,
                available_bytes: self.available.load(Ordering::SeqCst),
            }])
        }
    }

    #[derive(Debug)]
    struct FailableMemoryTelemetry {
        fail: Arc<AtomicBool>,
    }

    impl MemoryTelemetry for FailableMemoryTelemetry {
        fn observe(&self) -> Result<Vec<MemoryObservation>, MemoryError> {
            if self.fail.load(Ordering::SeqCst) {
                return Err(MemoryError::InvalidTelemetry(
                    "deterministic telemetry failure".to_string(),
                ));
            }
            Ok(vec![MemoryObservation {
                tier: MemoryTier::Host,
                total_bytes: 1024,
                available_bytes: 1024,
            }])
        }
    }

    fn test_memory_manager() -> MemoryManager {
        let config = MemoryManagerConfig::new(
            vec![TierBudget::new(MemoryTier::Host, 1024 * 1024).unwrap()],
            MemoryTopology::Discrete,
            PressureThresholds::default(),
            0,
            128,
            128,
            128,
        )
        .unwrap();
        MemoryManager::new(
            config,
            Arc::new(TestMemoryTelemetry),
            Arc::new(SystemMemoryClock::new()),
        )
    }

    fn mutable_test_memory_manager(available: Arc<AtomicU64>) -> MemoryManager {
        let config = MemoryManagerConfig::new(
            vec![TierBudget::new(MemoryTier::Host, 1024).unwrap()],
            MemoryTopology::Discrete,
            PressureThresholds::default(),
            0,
            128,
            128,
            128,
        )
        .unwrap();
        MemoryManager::new(
            config,
            Arc::new(MutableMemoryTelemetry { available }),
            Arc::new(SystemMemoryClock::new()),
        )
    }

    impl TestControl {
        fn new(adapter: Arc<dyn BackendRuntimeAdapter>) -> Self {
            Self::new_with_memory(adapter, test_memory_manager())
        }

        fn new_with_memory(adapter: Arc<dyn BackendRuntimeAdapter>, memory: MemoryManager) -> Self {
            let root = std::env::temp_dir().join(format!(
                "werk-local-control-{}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir(&root).unwrap();
            let store = ModelStore::resolve(Some(root.clone())).unwrap();
            store.ensure().unwrap();
            let manifest = ModelManifest {
                id: "model".to_string(),
                source: ModelSource::HuggingFace {
                    repo: "test/model".to_string(),
                },
                format: ModelFormat::Gguf,
                architecture: Some("test".to_string()),
                tokenizer_path: None,
                config_path: None,
                model_path: None,
                backend: "test".to_string(),
                created_unix: 1,
                files: Vec::new(),
                artifacts: Vec::new(),
                metadata: ModelMetadata::default(),
            };
            fs::create_dir(store.model_dir("model")).unwrap();
            store.write_manifest(&manifest).unwrap();
            Self {
                root,
                control: LocalWerkControl::new_with_memory_manager(store, adapter, memory),
            }
        }
    }

    impl Drop for TestControl {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn backend_model_residency_overrides_fallback_without_enabling_state_operations() {
        let test = TestControl::new(Arc::new(
            StaticRuntimeAdapter::new("resident-test-backend").with_model_residency(
                ModelResidencyStatus::Supported,
                "the backend keeps the model loaded between requests",
            ),
        ));

        let capabilities = capabilities_for(&test.control.inner).unwrap();
        let residency = capabilities
            .iter()
            .find(|capability| capability.id == MODEL_RESIDENCY_CAPABILITY)
            .unwrap();
        assert_eq!(residency.status, CapabilityStatus::Supported);
        assert_eq!(
            residency.operations,
            vec![AUTOMATIC_REUSE_OPERATION.to_string()]
        );

        let state_capabilities = capabilities
            .iter()
            .filter(|capability| capability.id.starts_with("runtime.state."))
            .collect::<Vec<_>>();
        assert!(!state_capabilities.is_empty());
        assert!(state_capabilities.iter().all(|capability| {
            capability.status == CapabilityStatus::Unsupported && capability.operations.is_empty()
        }));
    }

    #[test]
    fn missing_backend_model_residency_is_fail_closed() {
        let test = TestControl::new(Arc::new(StaticRuntimeAdapter::new(
            "non-resident-test-backend",
        )));

        let capabilities = capabilities_for(&test.control.inner).unwrap();
        let residency = capabilities
            .iter()
            .find(|capability| capability.id == MODEL_RESIDENCY_CAPABILITY)
            .unwrap();
        assert_eq!(residency.status, CapabilityStatus::Unsupported);
        assert!(residency.operations.is_empty());
        assert!(residency.detail.contains("non-resident-test-backend"));
    }

    fn prefill(policy: PersistenceMode) -> PrefillRequest {
        PrefillRequest {
            model_id: "model".to_string(),
            input: PrefillInput::Messages {
                messages: vec![ProtocolMessage {
                    role: "user".to_string(),
                    content: "private prompt".to_string(),
                }],
            },
            policy: crate::werk_protocol::PersistencePolicy {
                mode: policy,
                reuse: ReuseMode::Prefer,
                ttl_seconds: Some(60),
                pin: false,
            },
            allow_experimental: false,
        }
    }

    fn volatile_state(
        test: &TestControl,
        principal_id: &str,
        state_id: &str,
        expires_unix_ms: Option<u64>,
    ) -> VolatileState {
        let manifest = test.control.inner.store.get("model").unwrap();
        let compatibility = test
            .control
            .inner
            .adapter
            .compatibility(&manifest, "sha256:prompt")
            .unwrap();
        let now = now_unix_ms();
        VolatileState {
            principal_id: principal_id.to_string(),
            summary: StateSummary {
                id: state_id.to_string(),
                model_id: "model".to_string(),
                tier: StateTier::Ram,
                status: StateStatus::Ready,
                bytes: Some(1),
                created_unix_ms: now,
                last_accessed_unix_ms: now,
                expires_unix_ms,
                pinned: false,
                backend: compatibility.backend.clone(),
                reusable: true,
            },
            compatibility,
            prompt_tokens: 1,
            lease: BackendStateLease::new(
                test.control.inner.adapter.clone(),
                BackendState::InProcess {
                    handle: state_id.to_string(),
                    bytes: Some(1),
                    tier: StateTier::Ram,
                    instance_id: test.control.inner.adapter.descriptor().instance_id,
                },
            ),
            persisted: false,
            allocation_id: None,
        }
    }

    #[tokio::test]
    async fn model_reads_and_dry_runs_leave_an_absent_custom_home_absent() {
        let parent = std::env::temp_dir().join(format!(
            "werk-local-read-only-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&parent).unwrap();
        let home = parent.join("custom-home");
        let store = ModelStore::resolve(Some(home.clone())).unwrap();
        let control = LocalWerkControl::new_with_memory_manager(
            store,
            Arc::new(FakeAdapter::new()),
            test_memory_manager(),
        );

        let list_error = control
            .list_experts(
                ControlContext::local("req_list"),
                ExpertListFilter {
                    model_id: Some("missing-model".to_string()),
                    ..ExpertListFilter::default()
                },
            )
            .await
            .unwrap_err();
        assert_eq!(list_error.code, ProtocolErrorCode::NotFound);
        assert!(!home.exists());

        let action_error = control
            .expert_action(
                ControlContext::local("req_expert_preview"),
                ExpertActionRequest {
                    model_id: "missing-model".to_string(),
                    expert_ids: vec!["expert-1".to_string()],
                    action: crate::werk_protocol::ExpertAction::Pin,
                    target_tier: None,
                    dry_run: true,
                    allow_experimental: false,
                },
            )
            .await
            .unwrap_err();
        assert_eq!(action_error.code, ProtocolErrorCode::NotFound);
        assert!(!home.exists());

        let promote_error = control
            .state_action(
                ControlContext::local("req_state_preview"),
                "st_missing".to_string(),
                StateActionRequest {
                    action: StateAction::Promote,
                    target_tier: Some(StateTier::Ram),
                    dry_run: true,
                    allow_experimental: false,
                },
            )
            .await
            .unwrap_err();
        assert_eq!(promote_error.code, ProtocolErrorCode::NotFound);
        assert!(!home.exists());

        drop(control);
        let _ = fs::remove_dir_all(parent);
    }

    #[tokio::test]
    async fn prefill_rejects_invalid_backend_compatibility_before_backend_work() {
        for kind in [
            InvalidCompatibilityKind::WrongPrompt,
            InvalidCompatibilityKind::TooManyProcessors,
        ] {
            let prefill_calls = Arc::new(AtomicUsize::new(0));
            let adapter = Arc::new(InvalidCompatibilityAdapter {
                inner: FakeAdapter::new(),
                kind,
                prefill_calls: prefill_calls.clone(),
            });
            let test = TestControl::new(adapter);

            let error = test
                .control
                .prefill(
                    ControlContext::new("p_alice", "req_invalid_compatibility"),
                    prefill(PersistenceMode::Ephemeral),
                )
                .await
                .unwrap_err();

            assert_eq!(error.code, ProtocolErrorCode::Internal);
            assert_eq!(prefill_calls.load(Ordering::SeqCst), 0);
        }
    }

    #[tokio::test]
    async fn disk_state_is_reused_without_persisting_prompt_or_path_in_protocol() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let context = ControlContext::new("p_alice", "req_1");
        let first = test
            .control
            .prefill(context.clone(), prefill(PersistenceMode::Disk))
            .await
            .unwrap();
        assert!(!first.reused);
        assert!(first.state_id.is_some());
        let second = test
            .control
            .prefill(context, prefill(PersistenceMode::Disk))
            .await
            .unwrap();
        assert!(second.reused);
        assert_eq!(first.state_id, second.state_id);
        let json = serde_json::to_string(&second).unwrap();
        assert!(!json.contains("private prompt"));
        assert!(!json.contains(test.root.to_string_lossy().as_ref()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_disk_reuse_restores_once_and_reuses_the_volatile_overlay() {
        let adapter = Arc::new(FakeAdapter::new());
        let test = TestControl::new(adapter.clone());
        let original = test
            .control
            .prefill(
                ControlContext::new("p_alice", "req_seed"),
                prefill(PersistenceMode::Disk),
            )
            .await
            .unwrap();
        let expected_state_id = original.state_id.unwrap();

        let first = test.control.prefill(
            ControlContext::new("p_alice", "req_restore_a"),
            prefill(PersistenceMode::Memory),
        );
        let second = test.control.prefill(
            ControlContext::new("p_alice", "req_restore_b"),
            prefill(PersistenceMode::Memory),
        );
        let (first, second) = tokio::join!(first, second);
        let first = first.unwrap();
        let second = second.unwrap();

        assert!(first.reused);
        assert!(second.reused);
        assert_eq!(first.state_id.as_deref(), Some(expected_state_id.as_str()));
        assert_eq!(second.state_id.as_deref(), Some(expected_state_id.as_str()));
        assert_eq!(adapter.restore_calls.load(Ordering::SeqCst), 1);
        let states = test.control.inner.volatile_states.lock().unwrap();
        let overlay = states
            .get(&("p_alice".to_string(), expected_state_id))
            .unwrap();
        assert!(overlay.persisted);
        assert_eq!(overlay.summary.tier, StateTier::Ram);
    }

    #[tokio::test]
    async fn disk_policy_persists_a_reused_memory_state_before_reporting_success() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let memory = test
            .control
            .prefill(
                ControlContext::new("p_alice", "req_memory"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap();
        let state_id = memory.state_id.unwrap();

        let disk = test
            .control
            .prefill(
                ControlContext::new("p_alice", "req_disk"),
                prefill(PersistenceMode::Disk),
            )
            .await
            .unwrap();
        assert!(disk.reused);
        assert_eq!(disk.state_id.as_deref(), Some(state_id.as_str()));
        assert!(
            test.control
                .inner
                .state_store
                .inspect("p_alice", &state_id)
                .is_ok()
        );
        assert!(
            test.control
                .inner
                .volatile_states
                .lock()
                .unwrap()
                .get(&("p_alice".to_string(), state_id))
                .unwrap()
                .persisted
        );
    }

    #[tokio::test]
    async fn state_and_handoff_are_principal_partitioned_and_handoff_is_single_use() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let alice = test
            .control
            .prefill(
                ControlContext::new("p_alice", "req_1"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap();
        assert!(
            test.control
                .list_states(
                    ControlContext::new("p_bob", "req_2"),
                    StateListFilter::default(),
                )
                .await
                .unwrap()
                .states
                .is_empty()
        );
        let request = DecodeRequest {
            handoff: alice.handoff,
            max_tokens: 4,
            temperature: None,
            top_p: None,
            seed: None,
            stop: Vec::new(),
            allow_experimental: false,
        };
        assert_eq!(
            test.control
                .decode(ControlContext::new("p_alice", "req_3"), request.clone())
                .await
                .unwrap()
                .text,
            "decoded"
        );
        assert_eq!(
            test.control
                .decode(ControlContext::new("p_alice", "req_4"), request)
                .await
                .unwrap_err()
                .code,
            ProtocolErrorCode::ExpiredHandoff
        );
    }

    #[tokio::test]
    async fn full_handoff_capacity_rejects_prefill_before_state_creation() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let template = volatile_state(&test, "p_alice", "st_template", None);
        let mut tokens = Vec::new();
        loop {
            match test.control.inner.handoffs.issue(HandoffRecord {
                principal_id: "p_alice".to_string(),
                model_id: "model".to_string(),
                state_id: None,
                state: template.lease.clone(),
                compatibility: template.compatibility.clone(),
                expires_unix_ms: now_unix_ms().saturating_add(60_000),
            }) {
                Ok(token) => tokens.push(token),
                Err(error) => {
                    assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);
                    break;
                }
            }
        }
        assert_eq!(tokens.len(), 128);

        let error = test
            .control
            .prefill(
                ControlContext::new("p_alice", "req_full"),
                prefill(PersistenceMode::Disk),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);
        assert!(error.retryable);
        assert!(
            test.control
                .inner
                .state_store
                .all_summaries("p_alice")
                .unwrap()
                .is_empty()
        );
        assert!(
            test.control
                .inner
                .handoffs
                .inspect("p_alice", &tokens[0])
                .is_ok()
        );
    }

    #[tokio::test]
    async fn reuse_required_never_silently_prefills() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let mut request = prefill(PersistenceMode::Disk);
        request.policy.reuse = ReuseMode::Required;
        assert_eq!(
            test.control
                .prefill(ControlContext::local("req"), request)
                .await
                .unwrap_err()
                .code,
            ProtocolErrorCode::Conflict
        );
    }

    #[tokio::test]
    async fn backend_memory_is_reserved_and_accounted_for_live_state() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let response = test
            .control
            .prefill(
                ControlContext::local("req_prefill"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap();
        assert!(response.state_id.is_some());

        let status = test
            .control
            .memory_status(ControlContext::local("req_memory"))
            .await
            .unwrap();
        assert_eq!(status.host.managed_bytes, 9);
        assert_eq!(status.host.reserved_bytes, 0);
        assert_eq!(status.counters.get("managed_allocations"), Some(&1));
    }

    #[tokio::test]
    async fn telemetry_failure_preserves_manager_accounting_in_memory_status() {
        let fail = Arc::new(AtomicBool::new(false));
        let config = MemoryManagerConfig::new(
            vec![TierBudget::new(MemoryTier::Host, 1024).unwrap()],
            MemoryTopology::Discrete,
            PressureThresholds::default(),
            0,
            128,
            128,
            128,
        )
        .unwrap();
        let memory = MemoryManager::new(
            config,
            Arc::new(FailableMemoryTelemetry { fail: fail.clone() }),
            Arc::new(SystemMemoryClock::new()),
        );
        memory
            .reserve_load(
                AllocationId::new(1).unwrap(),
                MemoryTier::Host,
                100,
                false,
                None,
            )
            .unwrap()
            .commit_load()
            .unwrap();
        let _active_reservation = memory
            .reserve_load(
                AllocationId::new(2).unwrap(),
                MemoryTier::Host,
                50,
                false,
                None,
            )
            .unwrap();
        memory
            .record_orphaned_release(MemoryTier::Host, Some(7))
            .unwrap();
        fail.store(true, Ordering::SeqCst);
        let test = TestControl::new_with_memory(Arc::new(FakeAdapter::new()), memory);

        let status = test
            .control
            .memory_status(ControlContext::local("req_failed_telemetry"))
            .await
            .unwrap();

        assert_eq!(status.host.capacity_bytes, None);
        assert_eq!(status.host.available_bytes, None);
        assert_eq!(status.host.pressure, PressureLevel::Unknown);
        assert_eq!(status.overall_pressure, PressureLevel::Unknown);
        assert_eq!(status.host.managed_bytes, 107);
        assert_eq!(status.host.reserved_bytes, 50);
        assert_eq!(status.counters.get("active_reservations"), Some(&1));
        assert_eq!(status.counters.get("managed_allocations"), Some(&1));
        assert_eq!(status.counters.get("failed_releases"), Some(&1));
        assert_eq!(status.counters.get("orphaned_release_bytes"), Some(&7));
        assert_eq!(status.counters.get("telemetry_errors"), Some(&1));
    }

    #[test]
    fn insufficient_admission_plan_does_not_partially_release_states() {
        let releases = Arc::new(AtomicUsize::new(0));
        let adapter = Arc::new(ContractBreakingAdapter {
            inner: FakeAdapter::new(),
            prefill_tier: StateTier::Ram,
            restore_tier: StateTier::Ram,
            fail_release: false,
            releases: releases.clone(),
            restore_calls: Arc::new(AtomicUsize::new(0)),
        });
        let config = MemoryManagerConfig::new(
            vec![TierBudget::new(MemoryTier::Host, 1000).unwrap()],
            MemoryTopology::Discrete,
            PressureThresholds::default(),
            0,
            128,
            128,
            1,
        )
        .unwrap();
        let memory = MemoryManager::new(
            config,
            Arc::new(MutableMemoryTelemetry {
                available: Arc::new(AtomicU64::new(1000)),
            }),
            Arc::new(SystemMemoryClock::new()),
        );
        let test = TestControl::new_with_memory(adapter, memory);
        let compatibility = FakeAdapter::new()
            .compatibility(
                &test.control.inner.store.get("model").unwrap(),
                "sha256:prompt",
            )
            .unwrap();

        for state_id in ["st_pressure_one", "st_pressure_two"] {
            let pending = reserve_backend_memory(
                &test.control.inner,
                Some(BackendMemoryRequirement {
                    tier: StateTier::Ram,
                    bytes: 100,
                    demotion_target: None,
                }),
                false,
            )
            .unwrap();
            let (lease, allocation_id) = commit_backend_memory(
                &test.control.inner,
                BackendState::OpaqueBytes {
                    bytes: Arc::from(vec![0_u8; 100]),
                    tier: StateTier::Ram,
                    instance_id: "fake-process".to_string(),
                },
                pending,
                &compatibility,
            )
            .unwrap();
            let mut state = volatile_state(&test, "local", state_id, None);
            state.lease = lease;
            state.allocation_id = allocation_id;
            state.summary.bytes = Some(100);
            insert_volatile(&test.control.inner, state).unwrap();
        }
        releases.store(0, Ordering::SeqCst);

        let error = relieve_memory_pressure(&test.control.inner, MemoryTier::Host, 800)
            .expect_err("one permitted action cannot provide the required 151 bytes of relief");

        assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);
        assert!(error.retryable);
        assert_eq!(releases.load(Ordering::SeqCst), 0);
        assert_eq!(test.control.inner.volatile_states.lock().unwrap().len(), 2);
        let snapshot = test
            .control
            .inner
            .memory
            .as_ref()
            .unwrap()
            .refresh()
            .unwrap();
        assert_eq!(snapshot.tiers[0].managed_used_bytes, 200);
        assert_eq!(snapshot.tiers[0].reserved_bytes, 0);
        assert_eq!(snapshot.tiers[0].managed_allocations, 2);
        assert_eq!(snapshot.failed_releases, 0);
    }

    #[tokio::test]
    async fn emergency_eviction_transfers_release_hook_to_memory_action() {
        let available = Arc::new(AtomicU64::new(1024));
        let memory = mutable_test_memory_manager(available.clone());
        let test = TestControl::new_with_memory(Arc::new(FakeAdapter::new()), memory);
        let prefetched = test
            .control
            .prefill(
                ControlContext::local("req_prefill"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap();
        let state_id = prefetched.state_id.clone().unwrap();
        test.control
            .decode(
                ControlContext::local("req_decode"),
                DecodeRequest {
                    handoff: prefetched.handoff,
                    max_tokens: 1,
                    temperature: None,
                    top_p: None,
                    seed: None,
                    stop: Vec::new(),
                    allow_experimental: false,
                },
            )
            .await
            .unwrap();

        available.store(0, Ordering::SeqCst);
        relieve_memory_pressure(&test.control.inner, MemoryTier::Host, 0).unwrap();

        assert!(
            !test
                .control
                .inner
                .volatile_states
                .lock()
                .unwrap()
                .contains_key(&("local".to_string(), state_id))
        );
        let snapshot = test
            .control
            .inner
            .memory
            .as_ref()
            .unwrap()
            .refresh()
            .unwrap();
        assert_eq!(snapshot.tiers[0].managed_used_bytes, 0);
        assert_eq!(snapshot.tiers[0].managed_allocations, 0);
        assert_eq!(snapshot.completed_evictions, 1);
        assert!(snapshot.last_action_unix_millis.is_some());

        let status = test
            .control
            .memory_status(ControlContext::local("req_memory"))
            .await
            .unwrap();
        assert_eq!(status.counters.get("completed_evictions"), Some(&1));
        assert!(status.last_action_unix_ms.is_some());
    }

    #[tokio::test]
    async fn reservation_exists_before_backend_prefill_begins() {
        let memory = test_memory_manager();
        let saw_reservation = Arc::new(AtomicBool::new(false));
        let adapter = Arc::new(ReservationObservingAdapter {
            inner: FakeAdapter::new(),
            memory: memory.clone(),
            saw_reservation: saw_reservation.clone(),
        });
        let test = TestControl::new_with_memory(adapter, memory);
        test.control
            .prefill(
                ControlContext::local("req_prefill"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap();
        assert!(saw_reservation.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn experimental_prefill_is_rejected_before_compatibility_probe_without_opt_in() {
        let compatibility_calls = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(ExperimentalPrefillAdapter {
            inner: FakeAdapter::new(),
            compatibility_calls: compatibility_calls.clone(),
        }));

        let error = test
            .control
            .prefill(
                ControlContext::local("req_without_opt_in"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::ExperimentalOptInRequired);
        assert_eq!(compatibility_calls.load(Ordering::SeqCst), 0);

        let mut opted_in = prefill(PersistenceMode::Memory);
        opted_in.allow_experimental = true;
        test.control
            .prefill(ControlContext::local("req_with_opt_in"), opted_in)
            .await
            .unwrap();
        assert_eq!(compatibility_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn memory_policy_rejects_a_non_memory_backend_state() {
        let releases = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(ContractBreakingAdapter {
            inner: FakeAdapter::new(),
            prefill_tier: StateTier::Disk,
            restore_tier: StateTier::Ram,
            fail_release: false,
            releases: releases.clone(),
            restore_calls: Arc::new(AtomicUsize::new(0)),
        }));

        let error = test
            .control
            .prefill(
                ControlContext::local("req_wrong_tier"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::Internal);
        assert_eq!(releases.load(Ordering::SeqCst), 1);
        let snapshot = test
            .control
            .inner
            .memory
            .as_ref()
            .unwrap()
            .refresh()
            .unwrap();
        assert_eq!(snapshot.tiers[0].managed_used_bytes, 0);
        assert_eq!(snapshot.orphaned_release_bytes, 0);
    }

    #[tokio::test]
    async fn unreserved_backend_memory_is_rejected_and_released() {
        let releases = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(ContractBreakingAdapter {
            inner: FakeAdapter::new(),
            prefill_tier: StateTier::Ram,
            restore_tier: StateTier::Ram,
            fail_release: false,
            releases: releases.clone(),
            restore_calls: Arc::new(AtomicUsize::new(0)),
        }));

        let error = test
            .control
            .prefill(
                ControlContext::local("req_unreserved"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::Internal);
        assert_eq!(releases.load(Ordering::SeqCst), 1);
        let snapshot = test
            .control
            .inner
            .memory
            .as_ref()
            .unwrap()
            .refresh()
            .unwrap();
        assert_eq!(snapshot.tiers[0].managed_used_bytes, 0);
        assert_eq!(snapshot.tiers[0].managed_allocations, 0);
    }

    #[tokio::test]
    async fn unreserved_restore_memory_is_rejected_and_released() {
        let seed = TestControl::new(Arc::new(FakeAdapter::new()));
        seed.control
            .prefill(
                ControlContext::new("p_alice", "req_seed"),
                prefill(PersistenceMode::Disk),
            )
            .await
            .unwrap();

        let releases = Arc::new(AtomicUsize::new(0));
        let restore_calls = Arc::new(AtomicUsize::new(0));
        let adapter = Arc::new(ContractBreakingAdapter {
            inner: FakeAdapter::new(),
            prefill_tier: StateTier::Ram,
            restore_tier: StateTier::Ram,
            fail_release: false,
            releases: releases.clone(),
            restore_calls: restore_calls.clone(),
        });
        let control = LocalWerkControl::new_with_memory_manager(
            seed.control.inner.store.clone(),
            adapter,
            test_memory_manager(),
        );
        let error = control
            .prefill(
                ControlContext::new("p_alice", "req_restore"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::Internal);
        assert_eq!(restore_calls.load(Ordering::SeqCst), 1);
        assert_eq!(releases.load(Ordering::SeqCst), 1);
        let snapshot = control.inner.memory.as_ref().unwrap().refresh().unwrap();
        assert_eq!(snapshot.tiers[0].managed_used_bytes, 0);
        assert_eq!(snapshot.tiers[0].managed_allocations, 0);
    }

    #[tokio::test]
    async fn failed_rejected_state_cleanup_is_conservatively_accounted() {
        let releases = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(ContractBreakingAdapter {
            inner: FakeAdapter::new(),
            prefill_tier: StateTier::Ram,
            restore_tier: StateTier::Ram,
            fail_release: true,
            releases: releases.clone(),
            restore_calls: Arc::new(AtomicUsize::new(0)),
        }));

        let error = test
            .control
            .prefill(
                ControlContext::local("req_orphan"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::Internal);
        assert!(error.message.contains("conservatively accounted"));
        assert_eq!(releases.load(Ordering::SeqCst), 1);
        let snapshot = test
            .control
            .inner
            .memory
            .as_ref()
            .unwrap()
            .refresh()
            .unwrap();
        assert_eq!(snapshot.tiers[0].managed_used_bytes, 9);
        assert_eq!(snapshot.tiers[0].managed_allocations, 0);
        assert_eq!(snapshot.failed_releases, 1);
        assert_eq!(snapshot.orphaned_release_bytes, 9);
        let status = test
            .control
            .memory_status(ControlContext::local("req_orphan_status"))
            .await
            .unwrap();
        assert_eq!(status.counters.get("backend_cleanup_failures"), Some(&1));
        assert_eq!(status.counters.get("backend_cleanup_latched"), Some(&1));

        let retry = test
            .control
            .prefill(
                ControlContext::local("req_after_orphan"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap_err();
        assert_eq!(retry.code, ProtocolErrorCode::Unavailable);
        assert_eq!(releases.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn public_error_details_cannot_spoof_backend_cleanup_accounting() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let error = ProtocolError::new(
            ProtocolErrorCode::Unavailable,
            "adapter supplied an ordinary failure",
        )
        .with_details(serde_json::json!({
            "backend_cleanup_unconfirmed": true,
            "tier": "ram",
            "bytes": 99,
        }));

        let returned = backend_operation_error(&test.control.inner, error.clone());

        assert_eq!(returned, error);
        assert!(
            !test
                .control
                .inner
                .backend_cleanup_latched
                .load(Ordering::Acquire)
        );
        assert_eq!(
            test.control
                .inner
                .backend_cleanup_failures
                .load(Ordering::Relaxed),
            0
        );
        let memory = test
            .control
            .inner
            .memory
            .as_ref()
            .unwrap()
            .refresh()
            .unwrap();
        assert_eq!(memory.failed_releases, 0);
        assert_eq!(memory.orphaned_release_bytes, 0);
    }

    #[tokio::test]
    async fn decode_replacement_state_is_reserved_and_remains_accounted() {
        let releases = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(ReplacementDecodeAdapter {
            inner: FakeAdapter::new(),
            declare_requirement: true,
            releases: releases.clone(),
        }));
        let prefetched = test
            .control
            .prefill(
                ControlContext::local("req_prefill"),
                prefill(PersistenceMode::Ephemeral),
            )
            .await
            .unwrap();
        let decoded = test
            .control
            .decode(
                ControlContext::local("req_decode"),
                DecodeRequest {
                    handoff: prefetched.handoff,
                    max_tokens: 1,
                    temperature: None,
                    top_p: None,
                    seed: None,
                    stop: Vec::new(),
                    allow_experimental: false,
                },
            )
            .await
            .unwrap();
        assert!(decoded.handoff.is_some());
        let snapshot = test
            .control
            .inner
            .memory
            .as_ref()
            .unwrap()
            .refresh()
            .unwrap();
        assert_eq!(snapshot.tiers[0].managed_used_bytes, 9);
        assert_eq!(snapshot.tiers[0].managed_allocations, 1);
        assert_eq!(snapshot.tiers[0].reserved_bytes, 0);
        assert_eq!(releases.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn decode_rejects_and_releases_an_unreserved_memory_replacement() {
        let releases = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(ReplacementDecodeAdapter {
            inner: FakeAdapter::new(),
            declare_requirement: false,
            releases: releases.clone(),
        }));
        let prefetched = test
            .control
            .prefill(
                ControlContext::local("req_prefill"),
                prefill(PersistenceMode::Ephemeral),
            )
            .await
            .unwrap();
        let error = test
            .control
            .decode(
                ControlContext::local("req_decode"),
                DecodeRequest {
                    handoff: prefetched.handoff,
                    max_tokens: 1,
                    temperature: None,
                    top_p: None,
                    seed: None,
                    stop: Vec::new(),
                    allow_experimental: false,
                },
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::Internal);
        let snapshot = test
            .control
            .inner
            .memory
            .as_ref()
            .unwrap()
            .refresh()
            .unwrap();
        assert_eq!(snapshot.tiers[0].managed_used_bytes, 0);
        assert_eq!(snapshot.tiers[0].managed_allocations, 0);
        assert_eq!(releases.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn failed_backend_release_remains_conservatively_accounted() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(FailingReleaseAdapter {
            inner: FakeAdapter::new(),
            attempts: attempts.clone(),
        }));
        let pending = reserve_backend_memory(
            &test.control.inner,
            Some(BackendMemoryRequirement {
                tier: StateTier::Ram,
                bytes: 9,
                demotion_target: None,
            }),
            false,
        )
        .unwrap();
        let compatibility = FakeAdapter::new()
            .compatibility(
                &test.control.inner.store.get("model").unwrap(),
                "sha256:prompt",
            )
            .unwrap();
        let (lease, _) = commit_backend_memory(
            &test.control.inner,
            BackendState::OpaqueBytes {
                bytes: Arc::from(b"opaque-kv".as_slice()),
                tier: StateTier::Ram,
                instance_id: "fake-process".to_string(),
            },
            pending,
            &compatibility,
        )
        .unwrap();
        drop(lease);

        let snapshot = test
            .control
            .inner
            .memory
            .as_ref()
            .unwrap()
            .refresh()
            .unwrap();
        assert_eq!(snapshot.tiers[0].managed_used_bytes, 9);
        assert_eq!(snapshot.tiers[0].managed_allocations, 1);
        assert_eq!(snapshot.failed_releases, 1);
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn managed_unique_release_without_accounting_hook_is_transactional() {
        let releases = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(ReplacementDecodeAdapter {
            inner: FakeAdapter::new(),
            declare_requirement: true,
            releases: releases.clone(),
        }));
        let mut state = volatile_state(&test, "p_alice", "st_unhooked", None);
        state.allocation_id = Some(AllocationId::new(777).unwrap());
        let original_summary = state.summary.clone();

        let error = release_volatile_state(&mut state).unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::Internal);
        assert_eq!(releases.load(Ordering::SeqCst), 0);
        assert_eq!(state.summary, original_summary);
        assert_eq!(state.allocation_id, Some(AllocationId::new(777).unwrap()));
        assert_eq!(state.lease.strong_count(), 1);

        let key = ("p_alice".to_string(), "st_unhooked".to_string());
        assert!(
            replace_volatile_state(&test.control.inner, key.clone(), state)
                .unwrap()
                .is_none()
        );
        let mut round_tripped = take_volatile_state(&test.control.inner, &key)
            .unwrap()
            .expect("the unchanged state must remain reinsertable");
        assert_eq!(round_tripped.summary, original_summary);
        assert_eq!(releases.load(Ordering::SeqCst), 0);

        // The backend lease itself was not consumed by the rejected managed
        // release and remains synchronously releasable as an unmanaged lease.
        round_tripped.allocation_id = None;
        release_volatile_state(&mut round_tripped).unwrap();
        assert_eq!(releases.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_cleanup_accounts_the_actual_tier_and_oversized_state() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(FailingReleaseAdapter {
            inner: FakeAdapter::new(),
            attempts: attempts.clone(),
        }));
        let state = BackendState::InProcess {
            handle: "wrong-tier-oversized".to_string(),
            bytes: Some(99),
            tier: StateTier::Ram,
            instance_id: "fake-process".to_string(),
        };
        let error = cleanup_rejected_backend_state_with_fallback(
            &test.control.inner,
            &state,
            Some((MemoryTier::Accelerator(0), Some(9))),
            internal(),
        );

        assert_eq!(error.code, ProtocolErrorCode::Internal);
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        let snapshot = test
            .control
            .inner
            .memory
            .as_ref()
            .unwrap()
            .refresh()
            .unwrap();
        assert_eq!(snapshot.tiers[0].tier, MemoryTier::Host);
        assert_eq!(snapshot.tiers[0].managed_used_bytes, 99);
        assert_eq!(snapshot.orphaned_release_bytes, 99);
    }

    #[tokio::test]
    async fn disk_promotion_dry_run_never_calls_backend_restore() {
        let adapter = Arc::new(FakeAdapter::new());
        let test = TestControl::new(adapter.clone());
        let prefetched = test
            .control
            .prefill(
                ControlContext::local("req_prefill"),
                prefill(PersistenceMode::Disk),
            )
            .await
            .unwrap();
        let state_id = prefetched.state_id.unwrap();
        let result = test
            .control
            .state_action(
                ControlContext::local("req_promote"),
                state_id,
                StateActionRequest {
                    action: StateAction::Promote,
                    target_tier: Some(StateTier::Ram),
                    dry_run: true,
                    allow_experimental: false,
                },
            )
            .await
            .unwrap();
        assert!(result.changed);
        assert!(result.dry_run);
        assert_eq!(adapter.restore_calls.load(Ordering::SeqCst), 0);
        assert_eq!(adapter.resolution_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn disk_promotion_dry_run_does_not_recreate_a_missing_model_directory() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let prefetched = test
            .control
            .prefill(
                ControlContext::local("req_prefill"),
                prefill(PersistenceMode::Disk),
            )
            .await
            .unwrap();
        let state_id = prefetched.state_id.unwrap();
        let models = test.control.inner.store.models_dir();
        fs::remove_dir_all(&models).unwrap();

        let error = test
            .control
            .state_action(
                ControlContext::local("req_promote_missing_model"),
                state_id,
                StateActionRequest {
                    action: StateAction::Promote,
                    target_tier: Some(StateTier::Ram),
                    dry_run: true,
                    allow_experimental: false,
                },
            )
            .await
            .unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::NotFound);
        assert!(!models.exists());
    }

    #[tokio::test]
    async fn direct_opaque_persistence_requires_snapshot_contract_validation() {
        let disk_snapshot_calls = Arc::new(AtomicUsize::new(0));
        let disk_validation_calls = Arc::new(AtomicUsize::new(0));
        let disk = TestControl::new(Arc::new(SnapshotRejectingAdapter {
            inner: FakeAdapter::new(),
            snapshot_calls: disk_snapshot_calls.clone(),
            validation_calls: disk_validation_calls.clone(),
        }));

        let error = disk
            .control
            .prefill(
                ControlContext::local("req_disk_snapshot_rejected"),
                prefill(PersistenceMode::Disk),
            )
            .await
            .unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::IncompatibleState);
        assert_eq!(disk_validation_calls.load(Ordering::SeqCst), 1);
        assert_eq!(disk_snapshot_calls.load(Ordering::SeqCst), 0);
        assert!(
            disk.control
                .inner
                .state_store
                .all_summaries("local")
                .unwrap()
                .is_empty()
        );
        assert!(
            disk.control
                .inner
                .volatile_states
                .lock()
                .unwrap()
                .is_empty()
        );

        let auto_snapshot_calls = Arc::new(AtomicUsize::new(0));
        let auto_validation_calls = Arc::new(AtomicUsize::new(0));
        let auto = TestControl::new(Arc::new(SnapshotRejectingAdapter {
            inner: FakeAdapter::new(),
            snapshot_calls: auto_snapshot_calls.clone(),
            validation_calls: auto_validation_calls.clone(),
        }));
        let response = auto
            .control
            .prefill(
                ControlContext::local("req_auto_snapshot_rejected"),
                prefill(PersistenceMode::Auto),
            )
            .await
            .unwrap();
        let state_id = response.state_id.expect("Auto must retain live state");

        assert_eq!(auto_validation_calls.load(Ordering::SeqCst), 1);
        assert_eq!(auto_snapshot_calls.load(Ordering::SeqCst), 0);
        assert!(
            auto.control
                .inner
                .state_store
                .all_summaries("local")
                .unwrap()
                .is_empty()
        );
        let states = auto.control.inner.volatile_states.lock().unwrap();
        let retained = states
            .get(&("local".to_string(), state_id))
            .expect("Auto must fall back to a volatile state");
        assert!(!retained.persisted);
        assert_eq!(retained.summary.tier, StateTier::Ram);
    }

    #[tokio::test]
    async fn auto_falls_back_when_an_opaque_snapshot_payload_is_unreadable() {
        let missing_parent = std::env::temp_dir().join(format!(
            "werk-missing-snapshot-{}-{}",
            std::process::id(),
            NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed),
        ));
        let test = TestControl::new(Arc::new(MissingOpaqueFileAdapter {
            inner: FakeAdapter::new(),
            missing_path: missing_parent.join("gone.bin"),
        }));

        let response = test
            .control
            .prefill(
                ControlContext::local("req_auto_missing_snapshot"),
                prefill(PersistenceMode::Auto),
            )
            .await
            .unwrap();
        let state_id = response.state_id.expect("Auto must retain live state");

        assert!(
            test.control
                .inner
                .state_store
                .all_summaries("local")
                .unwrap()
                .is_empty()
        );
        let states = test.control.inner.volatile_states.lock().unwrap();
        let retained = states
            .get(&("local".to_string(), state_id))
            .expect("the unreadable snapshot must fall back to volatile state");
        assert!(!retained.persisted);
        assert_eq!(retained.summary.tier, StateTier::Ram);
    }

    #[tokio::test]
    async fn volatile_disk_dry_run_fails_closed_without_snapshot_inspection() {
        let snapshot_calls = Arc::new(AtomicUsize::new(0));
        let validation_calls = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(SnapshotRejectingAdapter {
            inner: FakeAdapter::new(),
            snapshot_calls: snapshot_calls.clone(),
            validation_calls,
        }));
        let state_id = "st_no_snapshot_inspection";
        insert_volatile(
            &test.control.inner,
            volatile_state(&test, "p_alice", state_id, None),
        )
        .unwrap();

        let error = test
            .control
            .state_action(
                ControlContext::new("p_alice", "req_dry_disk"),
                state_id.to_string(),
                StateActionRequest {
                    action: StateAction::Demote,
                    target_tier: Some(StateTier::Disk),
                    dry_run: true,
                    allow_experimental: false,
                },
            )
            .await
            .unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::Unsupported);
        assert_eq!(snapshot_calls.load(Ordering::SeqCst), 0);
        let states = test.control.inner.volatile_states.lock().unwrap();
        let retained = states
            .get(&("p_alice".to_string(), state_id.to_string()))
            .unwrap();
        assert_eq!(retained.summary.tier, StateTier::Ram);
        assert!(!retained.persisted);
        assert!(!test.root.join("runtime-state").exists());
    }

    #[tokio::test]
    async fn memory_state_can_be_demoted_to_crash_safe_disk_state() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let prefetched = test
            .control
            .prefill(
                ControlContext::local("req_prefill"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap();
        let state_id = prefetched.state_id.clone().unwrap();
        test.control
            .decode(
                ControlContext::local("req_decode"),
                DecodeRequest {
                    handoff: prefetched.handoff,
                    max_tokens: 1,
                    temperature: None,
                    top_p: None,
                    seed: None,
                    stop: Vec::new(),
                    allow_experimental: false,
                },
            )
            .await
            .unwrap();
        let result = test
            .control
            .state_action(
                ControlContext::local("req_demote"),
                state_id.clone(),
                StateActionRequest {
                    action: StateAction::Demote,
                    target_tier: Some(StateTier::Disk),
                    dry_run: false,
                    allow_experimental: false,
                },
            )
            .await
            .unwrap();
        assert_eq!(result.state.id, state_id);
        assert_eq!(result.state.tier, StateTier::Disk);
        assert_eq!(
            test.control
                .memory_status(ControlContext::local("req_memory"))
                .await
                .unwrap()
                .host
                .managed_bytes,
            0
        );
    }

    #[tokio::test]
    async fn disk_only_prune_detaches_a_live_memory_overlay_before_re_persisting() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let seeded = test
            .control
            .prefill(
                ControlContext::new("p_alice", "req_seed"),
                prefill(PersistenceMode::Disk),
            )
            .await
            .unwrap();
        let state_id = seeded.state_id.unwrap();
        test.control
            .prefill(
                ControlContext::new("p_alice", "req_restore"),
                prefill(PersistenceMode::Memory),
            )
            .await
            .unwrap();

        let pruned = test
            .control
            .prune_states(
                ControlContext::new("p_alice", "req_prune_disk"),
                PruneStatesRequest {
                    selector: StateSelector::Filter {
                        model_id: None,
                        tier: Some(StateTier::Disk),
                        older_than_unix_ms: None,
                    },
                    dry_run: false,
                },
            )
            .await
            .unwrap();
        assert_eq!(pruned.matched, 1);
        assert_eq!(pruned.removed, 1);
        assert!(
            !test
                .control
                .inner
                .volatile_states
                .lock()
                .unwrap()
                .get(&("p_alice".to_string(), state_id.clone()))
                .unwrap()
                .persisted
        );

        let demoted = test
            .control
            .state_action(
                ControlContext::new("p_alice", "req_re_persist"),
                state_id.clone(),
                StateActionRequest {
                    action: StateAction::Demote,
                    target_tier: Some(StateTier::Disk),
                    dry_run: false,
                    allow_experimental: false,
                },
            )
            .await
            .unwrap();
        assert_eq!(demoted.state.id, state_id);
        assert_eq!(demoted.state.tier, StateTier::Disk);
        assert_eq!(
            test.control
                .inner
                .state_store
                .all_summaries("p_alice")
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn prune_release_failure_returns_an_error_and_restores_unreleased_states() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(PruneReleaseAdapter {
            inner: FakeAdapter::new(),
            attempts: attempts.clone(),
            fail_on_attempt: Some(1),
        }));
        let expired_at = now_unix_ms().saturating_sub(1);
        {
            let mut states = test.control.inner.volatile_states.lock().unwrap();
            states.insert(
                ("p_alice".to_string(), "st_expired".to_string()),
                volatile_state(&test, "p_alice", "st_expired", Some(expired_at)),
            );
            states.insert(
                ("p_alice".to_string(), "st_live".to_string()),
                volatile_state(&test, "p_alice", "st_live", None),
            );
        }

        let error = test
            .control
            .prune_states(
                ControlContext::new("p_alice", "req_prune_release_failure"),
                PruneStatesRequest {
                    selector: StateSelector::All { confirm: true },
                    dry_run: false,
                },
            )
            .await
            .unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::Unavailable);
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        let states = test.control.inner.volatile_states.lock().unwrap();
        assert_eq!(states.len(), 2);
        assert!(states.contains_key(&("p_alice".to_string(), "st_expired".to_string())));
        assert!(states.contains_key(&("p_alice".to_string(), "st_live".to_string())));
    }

    #[test]
    fn volatile_state_limits_are_principal_scoped_and_never_evict_silently() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        for index in 0..MAX_VOLATILE_STATES_PER_PRINCIPAL {
            let id = format!("st_alice_{index}");
            insert_volatile(
                &test.control.inner,
                volatile_state(&test, "p_alice", &id, None),
            )
            .unwrap();
        }

        let overflow_id = "st_alice_overflow";
        let error = insert_volatile(
            &test.control.inner,
            volatile_state(&test, "p_alice", overflow_id, None),
        )
        .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);
        assert!(error.retryable);

        let shared_id = "st_shared";
        insert_volatile(
            &test.control.inner,
            volatile_state(&test, "p_bob", shared_id, None),
        )
        .unwrap();
        let states = test.control.inner.volatile_states.lock().unwrap();
        assert_eq!(states.len(), MAX_VOLATILE_STATES_PER_PRINCIPAL + 1);
        assert!(states.contains_key(&("p_alice".to_string(), "st_alice_0".to_string())));
        assert!(states.contains_key(&("p_bob".to_string(), shared_id.to_string())));
        assert!(!states.contains_key(&("p_alice".to_string(), overflow_id.to_string())));
    }

    #[test]
    fn identical_volatile_state_ids_are_isolated_by_principal() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        for principal in ["p_alice", "p_bob"] {
            insert_volatile(
                &test.control.inner,
                volatile_state(&test, principal, "st_same", None),
            )
            .unwrap();
        }
        let states = test.control.inner.volatile_states.lock().unwrap();
        assert_eq!(states.len(), 2);
        assert!(states.contains_key(&("p_alice".to_string(), "st_same".to_string())));
        assert!(states.contains_key(&("p_bob".to_string(), "st_same".to_string())));
    }

    #[tokio::test]
    async fn expired_pinned_volatile_state_cannot_be_mutated() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let mut expired = volatile_state(
            &test,
            "p_alice",
            "st_expired",
            Some(now_unix_ms().saturating_sub(1)),
        );
        expired.summary.pinned = true;
        insert_volatile(&test.control.inner, expired).unwrap();

        let error = test
            .control
            .state_action(
                ControlContext::new("p_alice", "req_expired"),
                "st_expired".to_string(),
                StateActionRequest {
                    action: StateAction::Unpin,
                    target_tier: None,
                    dry_run: false,
                    allow_experimental: false,
                },
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::NotFound);
        assert!(
            test.control
                .inner
                .volatile_states
                .lock()
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn failed_release_of_an_expired_handoff_latches_before_new_backend_work() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(FailingReleaseAdapter {
            inner: FakeAdapter::new(),
            attempts: attempts.clone(),
        }));
        let compatibility = FakeAdapter::new()
            .compatibility(
                &test.control.inner.store.get("model").unwrap(),
                "sha256:prompt",
            )
            .unwrap();
        test.control
            .inner
            .handoffs
            .issue(HandoffRecord {
                principal_id: "local".to_string(),
                model_id: "model".to_string(),
                state_id: None,
                state: local_backend_lease(
                    &test.control.inner,
                    BackendState::InProcess {
                        handle: "expired-handoff".to_string(),
                        bytes: Some(1),
                        tier: StateTier::Ram,
                        instance_id: "fake-process".to_string(),
                    },
                    None,
                ),
                compatibility,
                expires_unix_ms: now_unix_ms().saturating_sub(1),
            })
            .unwrap();

        let error = test
            .control
            .prefill(
                ControlContext::local("req_after_expired_handoff"),
                prefill(PersistenceMode::Ephemeral),
            )
            .await
            .unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::Unavailable);
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert!(
            test.control
                .inner
                .backend_cleanup_latched
                .load(Ordering::Acquire)
        );
    }

    #[tokio::test]
    async fn failed_release_of_an_expired_volatile_state_latches_before_tier_work() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let test = TestControl::new(Arc::new(FailingReleaseAdapter {
            inner: FakeAdapter::new(),
            attempts: attempts.clone(),
        }));
        let compatibility = FakeAdapter::new()
            .compatibility(
                &test.control.inner.store.get("model").unwrap(),
                "sha256:prompt",
            )
            .unwrap();
        let expired_at = now_unix_ms().saturating_sub(1);
        test.control.inner.volatile_states.lock().unwrap().insert(
            ("p_alice".to_string(), "st_expired_cleanup".to_string()),
            VolatileState {
                principal_id: "p_alice".to_string(),
                summary: StateSummary {
                    id: "st_expired_cleanup".to_string(),
                    model_id: "model".to_string(),
                    tier: StateTier::Ram,
                    status: StateStatus::Ready,
                    bytes: Some(1),
                    created_unix_ms: expired_at,
                    last_accessed_unix_ms: expired_at,
                    expires_unix_ms: Some(expired_at),
                    pinned: false,
                    backend: compatibility.backend.clone(),
                    reusable: true,
                },
                compatibility,
                prompt_tokens: 1,
                lease: local_backend_lease(
                    &test.control.inner,
                    BackendState::InProcess {
                        handle: "expired-volatile".to_string(),
                        bytes: Some(1),
                        tier: StateTier::Ram,
                        instance_id: "fake-process".to_string(),
                    },
                    None,
                ),
                persisted: false,
                allocation_id: None,
            },
        );

        let error = test
            .control
            .state_action(
                ControlContext::new("p_alice", "req_expired_cleanup"),
                "st_expired_cleanup".to_string(),
                StateActionRequest {
                    action: StateAction::Demote,
                    target_tier: Some(StateTier::Disk),
                    dry_run: false,
                    allow_experimental: false,
                },
            )
            .await
            .unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::Unavailable);
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert!(
            test.control
                .inner
                .backend_cleanup_latched
                .load(Ordering::Acquire)
        );
        assert!(
            test.control
                .inner
                .volatile_states
                .lock()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn volatile_reuse_rechecks_ttl_after_backend_validation() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let mut state = volatile_state(&test, "p_alice", "st_expiring", Some(101));
        state.summary.created_unix_ms = 1;
        state.summary.last_accessed_unix_ms = 7;
        let compatibility = state.compatibility.clone();
        test.control
            .inner
            .volatile_states
            .lock()
            .unwrap()
            .insert(("p_alice".to_string(), "st_expiring".to_string()), state);
        let clock_calls = AtomicUsize::new(0);

        let error = find_volatile_reuse_with_clock(
            &test.control.inner,
            "p_alice",
            "model",
            &compatibility,
            || match clock_calls.fetch_add(1, Ordering::SeqCst) {
                0 => 100,
                _ => 101,
            },
        )
        .err()
        .expect("reuse must fail once the candidate reaches its expiry");

        assert_eq!(error.code, ProtocolErrorCode::Conflict);
        assert!(clock_calls.load(Ordering::SeqCst) >= 2);
        let states = test.control.inner.volatile_states.lock().unwrap();
        let current = states
            .get(&("p_alice".to_string(), "st_expiring".to_string()))
            .unwrap();
        assert_eq!(current.summary.last_accessed_unix_ms, 7);
    }

    #[tokio::test]
    async fn reads_and_dry_runs_do_not_cleanup_expired_volatile_states() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let expired_at = Some(now_unix_ms().saturating_sub(1));
        {
            let mut states = test.control.inner.volatile_states.lock().unwrap();
            states.insert(
                ("p_alice".to_string(), "st_expired_alice".to_string()),
                volatile_state(&test, "p_alice", "st_expired_alice", expired_at),
            );
            states.insert(
                ("p_bob".to_string(), "st_expired_bob".to_string()),
                volatile_state(&test, "p_bob", "st_expired_bob", expired_at),
            );
            states.insert(
                ("p_alice".to_string(), "st_live_alice".to_string()),
                volatile_state(&test, "p_alice", "st_live_alice", None),
            );
        }

        let listed = test
            .control
            .list_states(
                ControlContext::new("p_alice", "req_list"),
                StateListFilter::default(),
            )
            .await
            .unwrap();
        assert_eq!(
            listed
                .states
                .iter()
                .map(|state| state.id.as_str())
                .collect::<Vec<_>>(),
            ["st_live_alice"]
        );

        let error = test
            .control
            .state_action(
                ControlContext::new("p_alice", "req_preview"),
                "st_expired_alice".to_string(),
                StateActionRequest {
                    action: StateAction::Pin,
                    target_tier: None,
                    dry_run: true,
                    allow_experimental: false,
                },
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::NotFound);

        let preview = test
            .control
            .prune_states(
                ControlContext::new("p_alice", "req_prune_preview"),
                PruneStatesRequest {
                    selector: StateSelector::All { confirm: true },
                    dry_run: true,
                },
            )
            .await
            .unwrap();
        assert_eq!(preview.matched, 1);
        assert_eq!(preview.removed, 0);
        assert_eq!(test.control.inner.volatile_states.lock().unwrap().len(), 3);

        let removed = test
            .control
            .prune_states(
                ControlContext::new("p_alice", "req_prune"),
                PruneStatesRequest {
                    selector: StateSelector::All { confirm: true },
                    dry_run: false,
                },
            )
            .await
            .unwrap();
        assert_eq!(removed.matched, 1);
        let states = test.control.inner.volatile_states.lock().unwrap();
        assert_eq!(states.len(), 1);
        assert!(states.contains_key(&("p_bob".to_string(), "st_expired_bob".to_string(),)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queued_mutations_do_not_consume_the_read_blocking_pool() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let held_mutation = test
            .control
            .inner
            .operation_gate
            .clone()
            .acquire_owned()
            .await
            .unwrap();
        let queued = (0..(MAX_BLOCKING_OPERATIONS * 2))
            .map(|index| {
                let control = test.control.clone();
                tokio::spawn(async move {
                    control
                        .state_action(
                            ControlContext::local(format!("queued_{index}")),
                            format!("st_missing_{index}"),
                            StateActionRequest {
                                action: StateAction::Pin,
                                target_tier: None,
                                dry_run: true,
                                allow_experimental: false,
                            },
                        )
                        .await
                })
            })
            .collect::<Vec<_>>();
        tokio::task::yield_now().await;

        let info = test
            .control
            .info(ControlContext::local("read_while_queued"))
            .await
            .unwrap();
        assert_eq!(info.service, "werk1112");
        assert_eq!(
            test.control.inner.blocking.available_permits(),
            MAX_BLOCKING_OPERATIONS
        );

        drop(held_mutation);
        for task in queued {
            assert_eq!(
                task.await.unwrap().unwrap_err().code,
                ProtocolErrorCode::NotFound
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_mutation_keeps_gate_until_blocking_work_finishes() {
        let test = TestControl::new(Arc::new(FakeAdapter::new()));
        let first_started = Arc::new(AtomicBool::new(false));
        let release_first = Arc::new(AtomicBool::new(false));
        let second_started = Arc::new(AtomicBool::new(false));

        let first = {
            let inner = test.control.inner.clone();
            let first_started = first_started.clone();
            let release_first = release_first.clone();
            tokio::spawn(run_mutating(inner, move |_| {
                first_started.store(true, Ordering::SeqCst);
                while !release_first.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
                Ok(())
            }))
        };
        tokio::time::timeout(Duration::from_secs(1), async {
            while !first_started.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        // Cancelling the caller must not release the mutation gate while the
        // non-cancellable blocking operation is still running.
        first.abort();
        let second = {
            let inner = test.control.inner.clone();
            let second_started = second_started.clone();
            tokio::spawn(run_mutating(inner, move |_| {
                second_started.store(true, Ordering::SeqCst);
                Ok(())
            }))
        };
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!second_started.load(Ordering::SeqCst));

        release_first.store(true, Ordering::SeqCst);
        tokio::time::timeout(Duration::from_secs(1), second)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(second_started.load(Ordering::SeqCst));
        assert!(first.await.unwrap_err().is_cancelled());
    }
}
