//! Single-node runtime persistence and memory-management implementation.

mod backend;
// The policy engine is intentionally retained as a tested building block for
// adapters that expose expert residency. The oMLX disk-backed cache owns its
// own tensor lifecycle and uses the shared validators; it does not activate
// this separate RAM/VRAM pressure-policy surface.
#[cfg_attr(not(test), allow(dead_code))]
mod experts;
mod handoff;
mod local;
pub(crate) mod memory;
mod persistence;
mod routing;
mod security;
mod store;

pub use backend::{
    AUTOMATIC_REUSE_OPERATION, BackendDecodeOptions, BackendDecodeRequest, BackendDecodeResult,
    BackendExpertOperationPlan, BackendMemoryRequirement, BackendPersistedStatePlan,
    BackendPersistedStateResolution, BackendPersistedStateScope, BackendPrefillRequest,
    BackendPrefillResult, BackendRuntimeAdapter, BackendRuntimeDescriptor, BackendSnapshot,
    BackendState, BackendStateLease, MODEL_RESIDENCY_CAPABILITY, ModelResidencyStatus,
    StaticRuntimeAdapter, UnsupportedRuntimeAdapter, model_residency_capability,
};
pub(crate) use backend::{
    require_expert_capability, validate_compatibility, validate_compatibility_envelope,
    validate_runtime_descriptor,
};
pub(crate) use experts::validate_expert_action;
pub use local::LocalWerkControl;
pub(crate) use local::{
    validate_expert_action_response, validate_expert_filter, validate_expert_list_response,
};
pub(crate) use persistence::ServerPersistenceConfig;
pub use routing::RoutedRuntimeAdapter;
pub(crate) use routing::RuntimeRoutedGenerationBackend;
pub(crate) use security::PrincipalDeriver;
pub(crate) use store::{StateStore, StoredCacheEntry};
