//! Single-node runtime persistence and memory-management implementation.

mod backend;
// The policy engine is intentionally retained as a tested building block for
// adapters that expose expert residency. Production adapters currently use
// only its shared validation helpers and truthfully report expert control as
// unsupported, so the remaining policy surface is dormant by design.
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
    validate_compatibility, validate_compatibility_envelope, validate_runtime_descriptor,
};
pub use local::LocalWerkControl;
pub(crate) use persistence::ServerPersistenceConfig;
pub use routing::RoutedRuntimeAdapter;
pub(crate) use routing::RuntimeRoutedGenerationBackend;
pub(crate) use security::PrincipalDeriver;
