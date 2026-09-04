use crate::model_store::ModelManifest;
use crate::werk_protocol::{
    Capability, CapabilityStatus, CompatibilityEnvelope, ExpertActionRequest, ExpertActionResponse,
    ExpertListFilter, ExpertListResponse, PersistencePolicy, PrefillInput, ProtocolError,
    ProtocolErrorCode, StateTier,
};
use serde_json::json;
use std::{collections::BTreeSet, fmt, fs::File, path::PathBuf, sync::Arc};

const MAX_COMPATIBILITY_FIELD_BYTES: usize = 512;
const MAX_COMPATIBILITY_MULTIMODAL_PROCESSORS: usize = 64;
const MAX_BACKEND_CAPABILITIES: usize = 256;
const MAX_CAPABILITY_OPERATIONS: usize = 64;
pub const MODEL_RESIDENCY_CAPABILITY: &str = "runtime.model_residency";
pub const AUTOMATIC_REUSE_OPERATION: &str = "automatic_reuse";
// JSON can escape one input byte as six output bytes. Keeping the
// conservative serialized upper bound below 4 MiB leaves ample room for the
// protocol envelope and derived control-plane capabilities under the 8-MiB
// transport limit.
const MAX_ENCODED_RUNTIME_DESCRIPTOR_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendRuntimeDescriptor {
    pub backend: String,
    pub backend_version: String,
    pub adapter_version: String,
    pub accelerator_family: String,
    /// Random identity for the concrete backend process generation. Volatile
    /// handles are valid only while this exact instance remains alive.
    pub instance_id: String,
    pub capabilities: Vec<Capability>,
}

/// Truthful ownership status for automatic model or pipeline reuse.
///
/// Residency is deliberately separate from named prefix/KV state. Declaring
/// it never enables prefill, decode, snapshots, restore, or state movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelResidencyStatus {
    Supported,
    ExternallyManaged,
    Unavailable,
    Unsupported,
}

impl ModelResidencyStatus {
    fn capability_status(self) -> CapabilityStatus {
        match self {
            Self::Supported => CapabilityStatus::Supported,
            Self::ExternallyManaged => CapabilityStatus::ExternallyManaged,
            Self::Unavailable => CapabilityStatus::Unavailable,
            Self::Unsupported => CapabilityStatus::Unsupported,
        }
    }

    fn has_automatic_reuse(self) -> bool {
        matches!(self, Self::Supported | Self::ExternallyManaged)
    }
}

/// Builds the standard model-residency capability without implying any
/// backend-owned prefix/KV-state operations.
pub fn model_residency_capability(
    status: ModelResidencyStatus,
    detail: impl Into<String>,
) -> Capability {
    Capability {
        id: MODEL_RESIDENCY_CAPABILITY.to_string(),
        status: status.capability_status(),
        detail: detail.into(),
        operations: status
            .has_automatic_reuse()
            .then(|| vec![AUTOMATIC_REUSE_OPERATION.to_string()])
            .unwrap_or_default(),
    }
}

/// Static runtime metadata for backends that can describe residency but do
/// not expose named runtime state.
///
/// The default descriptor is fail-closed. Callers opt into an exact residency
/// status explicitly; all prefix/KV adapter methods retain their unsupported
/// defaults.
#[derive(Debug, Clone)]
pub struct StaticRuntimeAdapter {
    descriptor: BackendRuntimeDescriptor,
}

impl StaticRuntimeAdapter {
    pub fn new(backend: impl Into<String>) -> Self {
        Self {
            descriptor: BackendRuntimeDescriptor {
                backend: backend.into(),
                backend_version: "unknown".to_string(),
                adapter_version: env!("CARGO_PKG_VERSION").to_string(),
                accelerator_family: "unknown".to_string(),
                instance_id: "metadata-only".to_string(),
                capabilities: Vec::new(),
            },
        }
    }

    pub fn with_backend_version(mut self, backend_version: impl Into<String>) -> Self {
        self.descriptor.backend_version = backend_version.into();
        self
    }

    pub fn with_accelerator_family(mut self, accelerator_family: impl Into<String>) -> Self {
        self.descriptor.accelerator_family = accelerator_family.into();
        self
    }

    pub fn with_instance_id(mut self, instance_id: impl Into<String>) -> Self {
        self.descriptor.instance_id = instance_id.into();
        self
    }

    pub fn with_model_residency(
        mut self,
        status: ModelResidencyStatus,
        detail: impl Into<String>,
    ) -> Self {
        self.descriptor
            .capabilities
            .retain(|capability| capability.id != MODEL_RESIDENCY_CAPABILITY);
        self.descriptor
            .capabilities
            .push(model_residency_capability(status, detail));
        self
    }
}

impl BackendRuntimeAdapter for StaticRuntimeAdapter {
    fn descriptor(&self) -> BackendRuntimeDescriptor {
        self.descriptor.clone()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BackendMemoryRequirement {
    pub tier: StateTier,
    pub bytes: u64,
    pub demotion_target: Option<StateTier>,
}

#[derive(Clone)]
pub enum BackendState {
    InProcess {
        handle: String,
        bytes: Option<u64>,
        tier: StateTier,
        instance_id: String,
    },
    /// Small backend-owned opaque state. Large snapshots should use
    /// `OpaqueFile` so the control plane never buffers multi-GB KV state.
    OpaqueBytes {
        bytes: Arc<[u8]>,
        tier: StateTier,
        instance_id: String,
    },
    OpaqueFile {
        path: PathBuf,
        bytes: u64,
        tier: StateTier,
        instance_id: String,
    },
    External {
        key: String,
        bytes: Option<u64>,
        tier: StateTier,
        instance_id: String,
    },
}

impl fmt::Debug for BackendState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackendState")
            .field(
                "kind",
                &match self {
                    Self::InProcess { .. } => "in_process",
                    Self::OpaqueBytes { .. } => "opaque_bytes",
                    Self::OpaqueFile { .. } => "opaque_file",
                    Self::External { .. } => "external",
                },
            )
            .field("bytes", &self.bytes())
            .field("tier", &self.natural_tier())
            .field("identity", &"[redacted]")
            .finish()
    }
}

impl BackendState {
    pub fn bytes(&self) -> Option<u64> {
        match self {
            Self::InProcess { bytes, .. } | Self::External { bytes, .. } => *bytes,
            Self::OpaqueFile { bytes, .. } => Some(*bytes),
            Self::OpaqueBytes { bytes, .. } => Some(bytes.len() as u64),
        }
    }

    pub fn natural_tier(&self) -> StateTier {
        match self {
            Self::InProcess { tier, .. }
            | Self::OpaqueBytes { tier, .. }
            | Self::OpaqueFile { tier, .. }
            | Self::External { tier, .. } => *tier,
        }
    }

    pub fn instance_id(&self) -> &str {
        match self {
            Self::InProcess { instance_id, .. }
            | Self::OpaqueBytes { instance_id, .. }
            | Self::OpaqueFile { instance_id, .. }
            | Self::External { instance_id, .. } => instance_id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendPrefillRequest {
    pub model_id: String,
    pub input: PrefillInput,
    pub compatibility: CompatibilityEnvelope,
    pub policy: PersistencePolicy,
}

#[derive(Debug, Clone)]
pub struct BackendPrefillResult {
    pub state: BackendState,
    pub prompt_tokens: u64,
    pub reused: bool,
}

pub struct BackendSnapshot {
    file: File,
    pub bytes: u64,
    identity: Arc<()>,
}

impl BackendSnapshot {
    /// Builds a snapshot from the exact file descriptor opened and
    /// integrity-checked by the state store. Retaining the descriptor avoids
    /// reopening a filesystem path that could have been replaced meanwhile.
    pub(crate) fn from_verified_file(file: File, bytes: u64) -> Self {
        Self {
            file,
            bytes,
            identity: Arc::new(()),
        }
    }

    /// Duplicates the already-verified file descriptor for a backend reader.
    /// The duplicate continues to name the same file after a rename, unlink,
    /// or path replacement.
    pub fn try_clone_file(&self) -> std::io::Result<File> {
        self.file.try_clone()
    }
}

impl fmt::Debug for BackendSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackendSnapshot")
            .field("source", &"[verified file handle]")
            .field("bytes", &self.bytes)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPersistedStateScope {
    SameProcess,
    CrossRestart,
}

#[derive(Debug, Clone)]
pub struct BackendPersistedStateResolution {
    pub compatibility: CompatibilityEnvelope,
    pub scope: BackendPersistedStateScope,
}

/// Opaque, one-use proof that persisted-state ownership, capabilities, and
/// restore memory were resolved against one concrete backend process.
///
/// The plan deliberately does not expose or format the verified snapshot
/// handle, its filesystem path, compatibility material, or the bound adapter.
pub struct BackendPersistedStatePlan {
    resolution: BackendPersistedStateResolution,
    descriptor: BackendRuntimeDescriptor,
    requirement: Option<BackendMemoryRequirement>,
    expected: CompatibilityEnvelope,
    snapshot_identity: Arc<()>,
    snapshot_bytes: u64,
    binding: PersistedStateBinding,
}

enum PersistedStateBinding {
    Direct,
    Routed(Arc<dyn BackendRuntimeAdapter>),
}

impl BackendPersistedStatePlan {
    fn direct(
        resolution: BackendPersistedStateResolution,
        descriptor: BackendRuntimeDescriptor,
        requirement: Option<BackendMemoryRequirement>,
        snapshot: &BackendSnapshot,
        expected: &CompatibilityEnvelope,
    ) -> Self {
        Self::new(
            resolution,
            descriptor,
            requirement,
            snapshot,
            expected,
            PersistedStateBinding::Direct,
        )
    }

    pub(crate) fn routed(
        resolution: BackendPersistedStateResolution,
        descriptor: BackendRuntimeDescriptor,
        requirement: Option<BackendMemoryRequirement>,
        snapshot: &BackendSnapshot,
        expected: &CompatibilityEnvelope,
        adapter: Arc<dyn BackendRuntimeAdapter>,
    ) -> Self {
        Self::new(
            resolution,
            descriptor,
            requirement,
            snapshot,
            expected,
            PersistedStateBinding::Routed(adapter),
        )
    }

    fn new(
        resolution: BackendPersistedStateResolution,
        descriptor: BackendRuntimeDescriptor,
        requirement: Option<BackendMemoryRequirement>,
        snapshot: &BackendSnapshot,
        expected: &CompatibilityEnvelope,
        binding: PersistedStateBinding,
    ) -> Self {
        Self {
            resolution,
            descriptor,
            requirement,
            expected: expected.clone(),
            snapshot_identity: snapshot.identity.clone(),
            snapshot_bytes: snapshot.bytes,
            binding,
        }
    }

    pub fn resolution(&self) -> &BackendPersistedStateResolution {
        &self.resolution
    }

    pub fn descriptor(&self) -> &BackendRuntimeDescriptor {
        &self.descriptor
    }

    pub fn memory_requirement(&self) -> Option<BackendMemoryRequirement> {
        self.requirement
    }

    pub(crate) fn validate_restore(
        &self,
        snapshot: &BackendSnapshot,
        expected: &CompatibilityEnvelope,
    ) -> Result<(), ProtocolError> {
        validate_compatibility_envelope(expected).map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::IncompatibleState,
                "persisted-state restore compatibility is invalid",
            )
        })?;
        validate_compatibility(expected, &self.expected)?;
        validate_compatibility(expected, &self.resolution.compatibility)?;
        if snapshot.bytes != self.snapshot_bytes
            || !Arc::ptr_eq(&snapshot.identity, &self.snapshot_identity)
        {
            return Err(ProtocolError::new(
                ProtocolErrorCode::IncompatibleState,
                "persisted-state restore plan does not belong to this verified snapshot",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_current_descriptor(
        &self,
        current: &BackendRuntimeDescriptor,
    ) -> Result<(), ProtocolError> {
        validate_runtime_descriptor(current)?;
        if current != &self.descriptor {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Unavailable,
                "the backend process changed after persisted-state restore was prepared",
            )
            .retryable(true));
        }
        Ok(())
    }

    pub(crate) fn routed_adapter(&self) -> Result<Arc<dyn BackendRuntimeAdapter>, ProtocolError> {
        match &self.binding {
            PersistedStateBinding::Routed(adapter) => Ok(adapter.clone()),
            PersistedStateBinding::Direct => Err(ProtocolError::new(
                ProtocolErrorCode::Internal,
                "persisted-state restore plan has invalid backend ownership",
            )),
        }
    }

    fn is_direct(&self) -> bool {
        matches!(self.binding, PersistedStateBinding::Direct)
    }
}

impl fmt::Debug for BackendPersistedStatePlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackendPersistedStatePlan")
            .field("scope", &self.resolution.scope)
            .field("requirement", &self.requirement)
            .field("snapshot_bytes", &self.snapshot_bytes)
            .finish_non_exhaustive()
    }
}

/// Opaque, one-use proof that an expert operation was authorized against one
/// exact model manifest, runtime descriptor, capability set, and adapter.
///
/// The plan is intentionally neither cloneable nor serializable. Its custom
/// debug representation omits the manifest, request, cursor, expert IDs,
/// process identity, and bound adapter.
pub struct BackendExpertOperationPlan {
    manifest: Option<ModelManifest>,
    descriptor: BackendRuntimeDescriptor,
    operation: ExpertOperation,
    binding: ExpertOperationBinding,
}

enum ExpertOperation {
    List(ExpertListFilter),
    Action(ExpertActionRequest),
}

enum ExpertOperationBinding {
    Direct,
    Routed(Arc<dyn BackendRuntimeAdapter>),
}

impl BackendExpertOperationPlan {
    fn direct_list(
        manifest: Option<&ModelManifest>,
        descriptor: BackendRuntimeDescriptor,
        filter: &ExpertListFilter,
    ) -> Result<Self, ProtocolError> {
        Self::list(manifest, descriptor, filter, ExpertOperationBinding::Direct)
    }

    pub(crate) fn routed_list(
        manifest: Option<&ModelManifest>,
        descriptor: BackendRuntimeDescriptor,
        filter: &ExpertListFilter,
        adapter: Arc<dyn BackendRuntimeAdapter>,
    ) -> Result<Self, ProtocolError> {
        Self::list(
            manifest,
            descriptor,
            filter,
            ExpertOperationBinding::Routed(adapter),
        )
    }

    fn list(
        manifest: Option<&ModelManifest>,
        descriptor: BackendRuntimeDescriptor,
        filter: &ExpertListFilter,
        binding: ExpertOperationBinding,
    ) -> Result<Self, ProtocolError> {
        validate_expert_manifest_binding(manifest, filter.model_id.as_deref())?;
        validate_runtime_descriptor(&descriptor)?;
        require_expert_capability(&descriptor, filter.allow_experimental, false)?;
        Ok(Self {
            manifest: manifest.cloned(),
            descriptor,
            operation: ExpertOperation::List(filter.clone()),
            binding,
        })
    }

    fn direct_action(
        manifest: &ModelManifest,
        descriptor: BackendRuntimeDescriptor,
        request: &ExpertActionRequest,
    ) -> Result<Self, ProtocolError> {
        Self::action(
            manifest,
            descriptor,
            request,
            ExpertOperationBinding::Direct,
        )
    }

    pub(crate) fn routed_action(
        manifest: &ModelManifest,
        descriptor: BackendRuntimeDescriptor,
        request: &ExpertActionRequest,
        adapter: Arc<dyn BackendRuntimeAdapter>,
    ) -> Result<Self, ProtocolError> {
        Self::action(
            manifest,
            descriptor,
            request,
            ExpertOperationBinding::Routed(adapter),
        )
    }

    fn action(
        manifest: &ModelManifest,
        descriptor: BackendRuntimeDescriptor,
        request: &ExpertActionRequest,
        binding: ExpertOperationBinding,
    ) -> Result<Self, ProtocolError> {
        validate_expert_manifest_binding(Some(manifest), Some(&request.model_id))?;
        validate_runtime_descriptor(&descriptor)?;
        require_expert_capability(&descriptor, request.allow_experimental, true)?;
        Ok(Self {
            manifest: Some(manifest.clone()),
            descriptor,
            operation: ExpertOperation::Action(request.clone()),
            binding,
        })
    }

    fn validate_current_descriptor(
        &self,
        current: &BackendRuntimeDescriptor,
    ) -> Result<(), ProtocolError> {
        validate_runtime_descriptor(current)?;
        if current != &self.descriptor {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Unavailable,
                "the expert backend changed after the operation was prepared",
            )
            .retryable(true));
        }
        Ok(())
    }

    fn into_direct_list(
        self,
        current: &BackendRuntimeDescriptor,
    ) -> Result<ExpertListFilter, ProtocolError> {
        self.validate_current_descriptor(current)?;
        validate_bound_expert_operation(&self.manifest, &self.operation)?;
        if !matches!(self.binding, ExpertOperationBinding::Direct) {
            return Err(invalid_expert_plan());
        }
        match self.operation {
            ExpertOperation::List(filter) => Ok(filter),
            ExpertOperation::Action(_) => Err(invalid_expert_plan()),
        }
    }

    fn into_direct_action(
        self,
        current: &BackendRuntimeDescriptor,
    ) -> Result<ExpertActionRequest, ProtocolError> {
        self.validate_current_descriptor(current)?;
        validate_bound_expert_operation(&self.manifest, &self.operation)?;
        if !matches!(self.binding, ExpertOperationBinding::Direct) {
            return Err(invalid_expert_plan());
        }
        match self.operation {
            ExpertOperation::Action(request) => Ok(request),
            ExpertOperation::List(_) => Err(invalid_expert_plan()),
        }
    }

    pub(crate) fn into_routed_list(
        self,
    ) -> Result<(Arc<dyn BackendRuntimeAdapter>, ExpertListFilter), ProtocolError> {
        validate_bound_expert_operation(&self.manifest, &self.operation)?;
        let adapter = match &self.binding {
            ExpertOperationBinding::Routed(adapter) => adapter.clone(),
            ExpertOperationBinding::Direct => return Err(invalid_expert_plan()),
        };
        self.validate_current_descriptor(&adapter.descriptor())?;
        match self.operation {
            ExpertOperation::List(filter) => Ok((adapter, filter)),
            ExpertOperation::Action(_) => Err(invalid_expert_plan()),
        }
    }

    pub(crate) fn into_routed_action(
        self,
    ) -> Result<(Arc<dyn BackendRuntimeAdapter>, ExpertActionRequest), ProtocolError> {
        validate_bound_expert_operation(&self.manifest, &self.operation)?;
        let adapter = match &self.binding {
            ExpertOperationBinding::Routed(adapter) => adapter.clone(),
            ExpertOperationBinding::Direct => return Err(invalid_expert_plan()),
        };
        self.validate_current_descriptor(&adapter.descriptor())?;
        match self.operation {
            ExpertOperation::Action(request) => Ok((adapter, request)),
            ExpertOperation::List(_) => Err(invalid_expert_plan()),
        }
    }
}

impl fmt::Debug for BackendExpertOperationPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackendExpertOperationPlan")
            .field(
                "operation",
                &match self.operation {
                    ExpertOperation::List(_) => "list",
                    ExpertOperation::Action(_) => "action",
                },
            )
            .field("manifest", &self.manifest.as_ref().map(|_| "[bound]"))
            .field("descriptor", &"[bound]")
            .field("adapter", &"[bound]")
            .finish()
    }
}

fn validate_expert_manifest_binding(
    manifest: Option<&ModelManifest>,
    model_id: Option<&str>,
) -> Result<(), ProtocolError> {
    if match (manifest, model_id) {
        (None, None) => false,
        (Some(manifest), Some(model_id)) => manifest.id != model_id,
        (None, Some(_)) | (Some(_), None) => true,
    } {
        Err(invalid_expert_plan())
    } else {
        Ok(())
    }
}

fn validate_bound_expert_operation(
    manifest: &Option<ModelManifest>,
    operation: &ExpertOperation,
) -> Result<(), ProtocolError> {
    match operation {
        ExpertOperation::List(filter) => {
            validate_expert_manifest_binding(manifest.as_ref(), filter.model_id.as_deref())
        }
        ExpertOperation::Action(request) => {
            validate_expert_manifest_binding(manifest.as_ref(), Some(&request.model_id))
        }
    }
}

fn invalid_expert_plan() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::Internal,
        "expert operation plan has invalid backend ownership",
    )
}

pub(crate) fn require_expert_capability(
    descriptor: &BackendRuntimeDescriptor,
    allow_experimental: bool,
    require_control: bool,
) -> Result<(), ProtocolError> {
    validate_runtime_descriptor(descriptor)?;
    let capability = descriptor
        .capabilities
        .iter()
        .find(|capability| capability.id == "runtime.experts.residency")
        .ok_or_else(|| {
            ProtocolError::new(
                ProtocolErrorCode::Unsupported,
                "the selected backend did not declare expert residency support",
            )
        })?;
    match capability.status {
        CapabilityStatus::Supported => Ok(()),
        CapabilityStatus::Experimental if allow_experimental => Ok(()),
        CapabilityStatus::Experimental => Err(ProtocolError::new(
            ProtocolErrorCode::ExperimentalOptInRequired,
            "runtime.experts.residency requires explicit experimental opt-in",
        )),
        CapabilityStatus::ExternallyManaged if !require_control => Ok(()),
        CapabilityStatus::Unavailable => Err(ProtocolError::new(
            ProtocolErrorCode::Unavailable,
            capability.detail.clone(),
        )
        .retryable(true)),
        CapabilityStatus::ExternallyManaged
        | CapabilityStatus::MetadataOnly
        | CapabilityStatus::Unsupported => Err(ProtocolError::new(
            ProtocolErrorCode::Unsupported,
            capability.detail.clone(),
        )),
    }
}

#[derive(Debug, Clone)]
pub struct BackendDecodeRequest {
    pub state: BackendStateLease,
    pub compatibility: CompatibilityEnvelope,
    pub options: BackendDecodeOptions,
}

#[derive(Clone)]
pub struct BackendDecodeOptions {
    pub max_tokens: u64,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
}

impl fmt::Debug for BackendDecodeOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackendDecodeOptions")
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("seed", &self.seed)
            .field("stop", &"[redacted]")
            .field("stop_count", &self.stop.len())
            .finish()
    }
}

#[derive(Clone)]
pub struct BackendDecodeResult {
    pub text: String,
    /// Optional replacement state for another one-time handoff. Decode always
    /// consumes the input lease; returning `None` ends the state lifecycle.
    pub state: Option<BackendState>,
    pub completion_tokens: u64,
    pub finish_reason: String,
}

impl fmt::Debug for BackendDecodeResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackendDecodeResult")
            .field("text", &"[redacted]")
            .field("text_bytes", &self.text.len())
            .field("state", &self.state)
            .field("completion_tokens", &self.completion_tokens)
            .field("finish_reason", &self.finish_reason)
            .finish()
    }
}

#[derive(Clone)]
pub struct BackendStateLease(Arc<BackendStateLeaseInner>);

struct BackendStateLeaseInner {
    adapter: Arc<dyn BackendRuntimeAdapter>,
    state: BackendState,
    on_release: Option<Box<dyn FnOnce(bool) + Send + Sync>>,
    released: bool,
}

impl Drop for BackendStateLeaseInner {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        let released = self.adapter.release(&self.state).is_ok();
        if let Some(on_release) = self.on_release.take() {
            on_release(released);
        }
    }
}

impl std::fmt::Debug for BackendStateLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BackendStateLease")
            .field("state", &"[opaque]")
            .finish()
    }
}

impl BackendStateLease {
    pub fn new(adapter: Arc<dyn BackendRuntimeAdapter>, state: BackendState) -> Self {
        Self(Arc::new(BackendStateLeaseInner {
            adapter,
            state,
            on_release: None,
            released: false,
        }))
    }

    pub fn with_release_hook(
        adapter: Arc<dyn BackendRuntimeAdapter>,
        state: BackendState,
        on_release: impl FnOnce(bool) + Send + Sync + 'static,
    ) -> Self {
        Self::with_boxed_release_hook(adapter, state, Box::new(on_release))
    }

    pub(crate) fn with_boxed_release_hook(
        adapter: Arc<dyn BackendRuntimeAdapter>,
        state: BackendState,
        on_release: Box<dyn FnOnce(bool) + Send + Sync>,
    ) -> Self {
        Self(Arc::new(BackendStateLeaseInner {
            adapter,
            state,
            on_release: Some(on_release),
            released: false,
        }))
    }

    pub fn state(&self) -> &BackendState {
        &self.0.state
    }

    pub fn adapter(&self) -> &Arc<dyn BackendRuntimeAdapter> {
        &self.0.adapter
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    pub(crate) fn has_release_hook(&self) -> bool {
        self.0.on_release.is_some()
    }

    /// Releases the backend object synchronously and transfers its accounting
    /// hook to the caller. On failure the lease remains intact and can be put
    /// back into its owner; no accounting transition has occurred.
    pub(crate) fn release_backend_and_take_hook(
        &mut self,
    ) -> Result<Option<Box<dyn FnOnce(bool) + Send + Sync>>, ProtocolError> {
        let inner = Arc::get_mut(&mut self.0).ok_or_else(|| {
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend state still has live leases and cannot be released",
            )
        })?;
        if inner.released {
            return Ok(inner.on_release.take());
        }
        inner.adapter.release(&inner.state)?;
        inner.released = true;
        Ok(inner.on_release.take())
    }

    /// Releases a uniquely owned state whose accounting hook is mandatory.
    /// Every precondition is checked before invoking the backend, so an error
    /// guarantees that the lease can still be returned to its owner.
    pub(crate) fn release_backend_and_take_required_hook(
        &mut self,
    ) -> Result<Box<dyn FnOnce(bool) + Send + Sync>, ProtocolError> {
        let inner = Arc::get_mut(&mut self.0).ok_or_else(|| {
            ProtocolError::new(
                ProtocolErrorCode::Conflict,
                "backend state still has live handoffs and cannot be released",
            )
        })?;
        if inner.released || inner.on_release.is_none() {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend state does not have a releasable accounting lease",
            ));
        }
        inner.adapter.release(&inner.state)?;
        inner.released = true;
        Ok(inner
            .on_release
            .take()
            .expect("the release hook was checked before backend release"))
    }
}

/// Backend-owned state operations. Werk never interprets opaque KV payloads.
pub trait BackendRuntimeAdapter: Send + Sync {
    fn descriptor(&self) -> BackendRuntimeDescriptor;

    /// Resolves the descriptor that owns operations for one installed model.
    /// Simple adapters have one descriptor; routing adapters override this so
    /// capability checks cannot race with an unrelated request changing the
    /// globally active backend.
    fn descriptor_for_model(
        &self,
        _manifest: &ModelManifest,
    ) -> Result<BackendRuntimeDescriptor, ProtocolError> {
        Ok(self.descriptor())
    }

    /// Resolves the descriptor that produced a compatibility envelope.
    fn descriptor_for_compatibility(
        &self,
        _compatibility: &CompatibilityEnvelope,
    ) -> Result<BackendRuntimeDescriptor, ProtocolError> {
        Ok(self.descriptor())
    }

    /// Resolves the descriptor that owns an opaque live state.
    fn descriptor_for_state(
        &self,
        _state: &BackendState,
    ) -> Result<BackendRuntimeDescriptor, ProtocolError> {
        Ok(self.descriptor())
    }

    /// Constructs every compatibility dimension owned by the backend. A
    /// missing or unknown required dimension must be returned as an error,
    /// never as a wildcard.
    fn compatibility(
        &self,
        _manifest: &ModelManifest,
        _prompt_fingerprint: &str,
    ) -> Result<CompatibilityEnvelope, ProtocolError> {
        Err(unsupported("verifiable runtime-state compatibility"))
    }

    /// Computes compatibility without probing, activating, loading, or
    /// otherwise mutating a backend. Adapters that cannot prove the complete
    /// envelope from already-available metadata must fail closed. This hook is
    /// used only for dry-run validation.
    fn inspect_compatibility(
        &self,
        _manifest: &ModelManifest,
        _prompt_fingerprint: &str,
    ) -> Result<CompatibilityEnvelope, ProtocolError> {
        Err(unsupported("side-effect-free compatibility inspection"))
    }

    /// Resolves and validates an opaque persisted snapshot before restore
    /// capabilities are consulted. Implementations must state whether their
    /// proof is limited to the current process or survives a restart.
    fn resolve_persisted_state(
        &self,
        _manifest: &ModelManifest,
        _snapshot: &BackendSnapshot,
        _expected: &CompatibilityEnvelope,
    ) -> Result<BackendPersistedStateResolution, ProtocolError> {
        Err(unsupported("validated persisted-state resolution"))
    }

    /// Resolves all restore decisions into a one-use plan. The default is for
    /// a single-process adapter; routing adapters override this method so the
    /// plan retains the exact selected adapter object.
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
        let resolution = self.resolve_persisted_state(manifest, snapshot, expected)?;
        validate_compatibility_envelope(&resolution.compatibility).map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::Internal,
                "backend resolved an invalid or unbounded compatibility envelope",
            )
        })?;
        validate_compatibility(expected, &resolution.compatibility)?;
        let descriptor = self.descriptor();
        validate_runtime_descriptor(&descriptor)?;
        let requirement = self.restore_memory_requirement(snapshot, expected)?;
        let plan = BackendPersistedStatePlan::direct(
            resolution,
            descriptor,
            requirement,
            snapshot,
            expected,
        );
        plan.validate_current_descriptor(&self.descriptor())?;
        Ok(plan)
    }

    /// Side-effect-free persisted-state validation for strict dry runs. The
    /// adapter must not probe, activate, load, restore, or mutate backend
    /// state. It returns the same ownership scope and bounded restore-memory
    /// requirement that a real restore would enforce.
    fn inspect_persisted_state(
        &self,
        _manifest: &ModelManifest,
        _snapshot: &BackendSnapshot,
        _expected: &CompatibilityEnvelope,
    ) -> Result<
        (
            BackendPersistedStateResolution,
            Option<BackendMemoryRequirement>,
        ),
        ProtocolError,
    > {
        Err(unsupported("side-effect-free persisted-state inspection"))
    }

    /// Verifies that a live opaque state is still owned by this exact runtime
    /// process and compatibility envelope.
    fn validate_state(
        &self,
        state: &BackendState,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<(), ProtocolError> {
        let descriptor = self.descriptor();
        let mut mismatch_fields = Vec::new();
        if state.instance_id() != descriptor.instance_id {
            mismatch_fields.push("process_instance");
        }
        if compatibility.backend != descriptor.backend {
            mismatch_fields.push("backend");
        }
        if compatibility.backend_version != descriptor.backend_version {
            mismatch_fields.push("backend_version");
        }
        if compatibility.runtime_adapter_version != descriptor.adapter_version {
            mismatch_fields.push("runtime_adapter_version");
        }
        if compatibility.accelerator_family != descriptor.accelerator_family {
            mismatch_fields.push("accelerator_family");
        }
        if mismatch_fields.is_empty() {
            Ok(())
        } else {
            Err(ProtocolError::new(
                ProtocolErrorCode::IncompatibleState,
                "runtime state no longer belongs to the resolved backend process",
            )
            .with_details(json!({ "mismatch_fields": mismatch_fields })))
        }
    }

    fn prefill_memory_requirement(
        &self,
        _request: &BackendPrefillRequest,
    ) -> Result<Option<BackendMemoryRequirement>, ProtocolError> {
        Ok(None)
    }

    fn restore_memory_requirement(
        &self,
        _snapshot: &BackendSnapshot,
        _compatibility: &CompatibilityEnvelope,
    ) -> Result<Option<BackendMemoryRequirement>, ProtocolError> {
        Ok(None)
    }

    /// Declares the additional allocation required when decode returns a
    /// replacement state. A backend that returns an in-memory replacement
    /// must provide an exact bound here so Werk can reserve capacity before
    /// decode begins. Returning `None` promises that decode will not create a
    /// new RAM/VRAM-owned state.
    fn decode_memory_requirement(
        &self,
        _request: &BackendDecodeRequest,
    ) -> Result<Option<BackendMemoryRequirement>, ProtocolError> {
        Ok(None)
    }

    fn prefill(
        &self,
        _request: BackendPrefillRequest,
    ) -> Result<BackendPrefillResult, ProtocolError> {
        Err(unsupported("prefill-only execution"))
    }

    fn decode(&self, _request: BackendDecodeRequest) -> Result<BackendDecodeResult, ProtocolError> {
        Err(unsupported("decode-from-state"))
    }

    fn restore(
        &self,
        _snapshot: BackendSnapshot,
        _compatibility: &CompatibilityEnvelope,
    ) -> Result<BackendState, ProtocolError> {
        Err(unsupported("persistent state restore"))
    }

    /// Consumes a previously prepared restore plan. Snapshot identity and the
    /// complete compatibility envelope are checked again after admission but
    /// before backend state is created.
    fn restore_prepared_state(
        &self,
        plan: BackendPersistedStatePlan,
        snapshot: BackendSnapshot,
        expected: &CompatibilityEnvelope,
    ) -> Result<BackendState, ProtocolError> {
        plan.validate_restore(&snapshot, expected)?;
        if !plan.is_direct() {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Internal,
                "persisted-state restore plan has invalid backend ownership",
            ));
        }
        plan.validate_current_descriptor(&self.descriptor())?;
        self.restore(snapshot, expected)
    }

    /// Produces a backend-owned opaque snapshot without changing ownership of
    /// the live state. The returned state is independently releasable.
    fn snapshot(&self, _state: &BackendState) -> Result<BackendState, ProtocolError> {
        Err(unsupported("persistent state snapshots"))
    }

    /// Proves, without creating a snapshot or otherwise mutating backend
    /// state, that `snapshot` can export this exact live state as an
    /// independently releasable opaque snapshot accepted by
    /// `validate_snapshot`. Dry runs fail closed unless an adapter provides
    /// this proof.
    fn inspect_snapshot_export(
        &self,
        _state: &BackendState,
        _compatibility: &CompatibilityEnvelope,
    ) -> Result<(), ProtocolError> {
        Err(unsupported("side-effect-free snapshot export inspection"))
    }

    /// Verifies an independently releasable snapshot returned by `snapshot`.
    ///
    /// Snapshot representations may differ from live-state representations
    /// (for example, an in-process handle may export an opaque file). Simple
    /// adapters can keep the default when both use the same validation rules.
    fn validate_snapshot(
        &self,
        snapshot: &BackendState,
        compatibility: &CompatibilityEnvelope,
    ) -> Result<(), ProtocolError> {
        self.validate_state(snapshot, compatibility)
    }

    fn move_state(
        &self,
        _state: Arc<BackendState>,
        _target: StateTier,
    ) -> Result<BackendState, ProtocolError> {
        Err(unsupported("state tier movement"))
    }

    fn release(&self, _state: &BackendState) -> Result<(), ProtocolError> {
        Ok(())
    }

    fn list_experts(
        &self,
        _filter: &ExpertListFilter,
    ) -> Result<ExpertListResponse, ProtocolError> {
        Err(unsupported("expert residency telemetry"))
    }

    /// Prepares expert telemetry against one exact manifest, descriptor, and
    /// capability set. Routing adapters override this method to retain the
    /// concrete selected adapter in the one-use plan.
    fn prepare_expert_list(
        &self,
        manifest: Option<&ModelManifest>,
        filter: &ExpertListFilter,
    ) -> Result<BackendExpertOperationPlan, ProtocolError> {
        let descriptor = match manifest {
            Some(manifest) => self.descriptor_for_model(manifest)?,
            None => self.descriptor(),
        };
        BackendExpertOperationPlan::direct_list(manifest, descriptor, filter)
    }

    /// Consumes a prepared expert-list plan. The full descriptor, including
    /// capabilities, is checked immediately before invoking the backend.
    fn list_experts_prepared(
        &self,
        plan: BackendExpertOperationPlan,
    ) -> Result<ExpertListResponse, ProtocolError> {
        let current = match plan.manifest.as_ref() {
            Some(manifest) => self.descriptor_for_model(manifest)?,
            None => self.descriptor(),
        };
        let filter = plan.into_direct_list(&current)?;
        self.list_experts(&filter)
    }

    fn expert_action(
        &self,
        _request: &ExpertActionRequest,
    ) -> Result<ExpertActionResponse, ProtocolError> {
        Err(unsupported("expert residency control"))
    }

    /// Prepares expert control against one exact manifest, descriptor, and
    /// capability set. Routing adapters override this method to retain the
    /// concrete selected adapter in the one-use plan.
    fn prepare_expert_action(
        &self,
        manifest: &ModelManifest,
        request: &ExpertActionRequest,
    ) -> Result<BackendExpertOperationPlan, ProtocolError> {
        let descriptor = self.descriptor_for_model(manifest)?;
        BackendExpertOperationPlan::direct_action(manifest, descriptor, request)
    }

    /// Consumes a prepared expert-action plan. The full descriptor, including
    /// capabilities, is checked immediately before invoking the backend.
    fn expert_action_prepared(
        &self,
        plan: BackendExpertOperationPlan,
    ) -> Result<ExpertActionResponse, ProtocolError> {
        let manifest = plan.manifest.as_ref().ok_or_else(invalid_expert_plan)?;
        let current = self.descriptor_for_model(manifest)?;
        let request = plan.into_direct_action(&current)?;
        self.expert_action(&request)
    }
}

#[derive(Debug, Clone)]
pub struct UnsupportedRuntimeAdapter {
    descriptor: BackendRuntimeDescriptor,
}

impl UnsupportedRuntimeAdapter {
    pub fn new(backend: impl Into<String>) -> Self {
        Self {
            descriptor: BackendRuntimeDescriptor {
                backend: backend.into(),
                backend_version: "unknown".to_string(),
                adapter_version: env!("CARGO_PKG_VERSION").to_string(),
                accelerator_family: "unknown".to_string(),
                instance_id: "unavailable".to_string(),
                capabilities: Vec::new(),
            },
        }
    }
}

impl BackendRuntimeAdapter for UnsupportedRuntimeAdapter {
    fn descriptor(&self) -> BackendRuntimeDescriptor {
        self.descriptor.clone()
    }
}

fn unsupported(operation: &str) -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::Unsupported,
        format!("the active backend does not support {operation}"),
    )
}

pub(crate) fn validate_compatibility(
    expected: &CompatibilityEnvelope,
    actual: &CompatibilityEnvelope,
) -> Result<(), ProtocolError> {
    let mismatch_fields = expected.mismatch_fields(actual);
    if mismatch_fields.is_empty() {
        return Ok(());
    }
    Err(ProtocolError::new(
        ProtocolErrorCode::IncompatibleState,
        "runtime-state compatibility does not match the resolved backend",
    )
    .with_details(json!({ "mismatch_fields": mismatch_fields })))
}

pub(crate) fn validate_compatibility_envelope(
    value: &CompatibilityEnvelope,
) -> Result<(), ProtocolError> {
    let required = [
        &value.model_fingerprint,
        &value.tokenizer_fingerprint,
        &value.prompt_fingerprint,
        &value.backend,
        &value.backend_version,
        &value.runtime_adapter_version,
        &value.accelerator_family,
        &value.tensor_dtype,
        &value.kv_dtype,
        &value.quantization,
        &value.cache_layout,
    ];
    if required.into_iter().any(invalid_bounded_text) {
        return Err(invalid_envelope());
    }
    let invalid_optional = |field: Option<&String>| field.is_some_and(invalid_bounded_text);
    if invalid_optional(value.chat_template_fingerprint.as_ref())
        || invalid_optional(value.context.rope_configuration_fingerprint.as_ref())
        || value.multimodal_processor_fingerprints.len() > MAX_COMPATIBILITY_MULTIMODAL_PROCESSORS
        || value
            .multimodal_processor_fingerprints
            .iter()
            .any(invalid_bounded_text)
        || value.context.context_size == 0
        || value.context.batch_size == Some(0)
        || value.block_size == Some(0)
    {
        return Err(invalid_envelope());
    }
    if !crate::werk_protocol::ProtocolVersion::V1.accepts(value.producer_protocol) {
        return Err(ProtocolError::new(
            ProtocolErrorCode::IncompatibleState,
            "runtime state was produced by an incompatible protocol version",
        ));
    }
    Ok(())
}

pub(crate) fn validate_runtime_descriptor(
    descriptor: &BackendRuntimeDescriptor,
) -> Result<(), ProtocolError> {
    let mut encoded_upper_bound = 256usize;
    for value in [
        &descriptor.backend,
        &descriptor.backend_version,
        &descriptor.adapter_version,
        &descriptor.accelerator_family,
        &descriptor.instance_id,
    ] {
        if invalid_bounded_text(value) {
            return Err(invalid_descriptor());
        }
        encoded_upper_bound = encoded_upper_bound
            .checked_add(value.len().saturating_mul(6).saturating_add(32))
            .ok_or_else(invalid_descriptor)?;
    }
    if descriptor.capabilities.len() > MAX_BACKEND_CAPABILITIES {
        return Err(invalid_descriptor());
    }
    let mut ids = BTreeSet::new();
    for capability in &descriptor.capabilities {
        if invalid_bounded_text(&capability.id)
            || capability.detail.trim().is_empty()
            || capability.detail.len() > 4 * 1024
            || capability.detail.chars().any(char::is_control)
            || capability.operations.len() > MAX_CAPABILITY_OPERATIONS
            || capability.operations.iter().any(invalid_bounded_text)
            || !ids.insert(capability.id.as_str())
        {
            return Err(invalid_descriptor());
        }
        encoded_upper_bound = encoded_upper_bound
            .checked_add(
                capability
                    .id
                    .len()
                    .saturating_add(capability.detail.len())
                    .saturating_mul(6)
                    .saturating_add(128),
            )
            .ok_or_else(invalid_descriptor)?;
        for operation in &capability.operations {
            encoded_upper_bound = encoded_upper_bound
                .checked_add(operation.len().saturating_mul(6).saturating_add(8))
                .ok_or_else(invalid_descriptor)?;
        }
        if encoded_upper_bound > MAX_ENCODED_RUNTIME_DESCRIPTOR_BYTES {
            return Err(invalid_descriptor());
        }
    }
    Ok(())
}

fn invalid_bounded_text(value: &String) -> bool {
    value.trim().is_empty()
        || value.len() > MAX_COMPATIBILITY_FIELD_BYTES
        || value.chars().any(char::is_control)
}

fn invalid_envelope() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::InvalidRequest,
        "compatibility envelope contains a missing, invalid, or unbounded field",
    )
}

fn invalid_descriptor() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::Internal,
        "backend returned an invalid or ambiguous runtime descriptor",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_residency_statuses_are_exact_and_do_not_declare_state_operations() {
        let cases = [
            (
                ModelResidencyStatus::Supported,
                CapabilityStatus::Supported,
                true,
            ),
            (
                ModelResidencyStatus::ExternallyManaged,
                CapabilityStatus::ExternallyManaged,
                true,
            ),
            (
                ModelResidencyStatus::Unavailable,
                CapabilityStatus::Unavailable,
                false,
            ),
            (
                ModelResidencyStatus::Unsupported,
                CapabilityStatus::Unsupported,
                false,
            ),
        ];

        for (residency, expected_status, operational) in cases {
            let capability = model_residency_capability(residency, "exact test status");
            assert_eq!(capability.id, MODEL_RESIDENCY_CAPABILITY);
            assert_eq!(capability.status, expected_status);
            assert_eq!(
                capability.operations,
                operational
                    .then(|| vec![AUTOMATIC_REUSE_OPERATION.to_string()])
                    .unwrap_or_default()
            );
            assert!(!capability.id.starts_with("runtime.state."));
        }
    }

    #[test]
    fn static_runtime_adapter_is_fail_closed_except_for_explicit_residency() {
        let adapter = StaticRuntimeAdapter::new("precise-backend")
            .with_backend_version("runtime-1")
            .with_accelerator_family("cpu")
            .with_instance_id("process-1")
            .with_model_residency(
                ModelResidencyStatus::Supported,
                "model weights remain in this Werk-owned process",
            );
        let descriptor = adapter.descriptor();

        assert_eq!(descriptor.backend, "precise-backend");
        assert_eq!(descriptor.backend_version, "runtime-1");
        assert_eq!(descriptor.accelerator_family, "cpu");
        assert_eq!(descriptor.instance_id, "process-1");
        assert_eq!(descriptor.capabilities.len(), 1);
        assert_eq!(
            descriptor.capabilities[0],
            model_residency_capability(
                ModelResidencyStatus::Supported,
                "model weights remain in this Werk-owned process",
            )
        );
        assert!(
            adapter
                .prefill(BackendPrefillRequest {
                    model_id: "model".to_string(),
                    input: PrefillInput::Text {
                        text: "not retained".to_string(),
                    },
                    compatibility: CompatibilityEnvelope {
                        model_fingerprint: "model".to_string(),
                        tokenizer_fingerprint: "tokenizer".to_string(),
                        prompt_fingerprint: "prompt".to_string(),
                        chat_template_fingerprint: None,
                        backend: "precise-backend".to_string(),
                        backend_version: "runtime-1".to_string(),
                        runtime_adapter_version: env!("CARGO_PKG_VERSION").to_string(),
                        accelerator_family: "cpu".to_string(),
                        tensor_dtype: "f32".to_string(),
                        kv_dtype: "f32".to_string(),
                        quantization: "none".to_string(),
                        cache_layout: "none".to_string(),
                        block_size: None,
                        context: crate::werk_protocol::ContextCompatibility {
                            context_size: 1,
                            batch_size: None,
                            rope_configuration_fingerprint: None,
                        },
                        multimodal_processor_fingerprints: Vec::new(),
                        producer_protocol: crate::werk_protocol::ProtocolVersion::V1,
                    },
                    policy: PersistencePolicy::default(),
                })
                .is_err()
        );
    }

    #[test]
    fn replacing_static_residency_status_does_not_duplicate_capabilities() {
        let descriptor = StaticRuntimeAdapter::new("backend")
            .with_model_residency(ModelResidencyStatus::Unavailable, "not started")
            .with_model_residency(ModelResidencyStatus::ExternallyManaged, "remote-owned")
            .descriptor();

        assert_eq!(descriptor.capabilities.len(), 1);
        assert_eq!(
            descriptor.capabilities[0].status,
            CapabilityStatus::ExternallyManaged
        );
        assert_eq!(
            descriptor.capabilities[0].operations,
            vec![AUTOMATIC_REUSE_OPERATION.to_string()]
        );
    }
}
