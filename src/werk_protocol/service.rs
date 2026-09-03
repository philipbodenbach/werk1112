use super::{
    CapabilitiesResponse, DecodeRequest, DecodeResponse, ExpertActionRequest, ExpertActionResponse,
    ExpertListFilter, ExpertListResponse, MemoryStatusResponse, PrefillRequest, PrefillResponse,
    PruneStatesRequest, PruneStatesResponse, RuntimeInfo, StateActionRequest, StateActionResponse,
    StateListFilter, StateListResponse, StateTier,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fmt, future::Future, pin::Pin};

pub type BoxControlFuture<'a, T> = Pin<Box<dyn Future<Output = ProtocolResult<T>> + Send + 'a>>;
pub type ProtocolResult<T> = Result<T, ProtocolError>;

#[derive(Clone, PartialEq, Eq)]
pub struct ControlContext {
    principal_id: String,
    request_id: String,
}

impl ControlContext {
    pub fn new(principal_id: impl Into<String>, request_id: impl Into<String>) -> Self {
        Self {
            principal_id: principal_id.into(),
            request_id: request_id.into(),
        }
    }

    pub fn local(request_id: impl Into<String>) -> Self {
        Self::new("local", request_id)
    }

    pub fn principal_id(&self) -> &str {
        &self.principal_id
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }
}

impl fmt::Debug for ControlContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlContext")
            .field("principal_id", &"[redacted]")
            .field("request_id", &self.request_id)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolErrorCode {
    InvalidRequest,
    IncompatibleProtocol,
    Unauthorized,
    Forbidden,
    NotFound,
    Conflict,
    IncompatibleState,
    ExpiredHandoff,
    Unsupported,
    Unavailable,
    ExperimentalOptInRequired,
    ResourceExhausted,
    CorruptState,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtocolErrorBody {
    pub code: ProtocolErrorCode,
    pub message: String,
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProtocolError {
    pub code: ProtocolErrorCode,
    pub message: String,
    pub retryable: bool,
    pub details: Option<Value>,
    internal: Option<InternalProtocolError>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum InternalProtocolError {
    BackendCleanupUnconfirmed { tier: StateTier, bytes: Option<u64> },
}

impl ProtocolError {
    pub fn new(code: ProtocolErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retryable: false,
            details: None,
            internal: None,
        }
    }

    pub fn retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    pub(crate) fn with_backend_cleanup_unconfirmed(
        mut self,
        tier: StateTier,
        bytes: Option<u64>,
    ) -> Self {
        self.internal = Some(InternalProtocolError::BackendCleanupUnconfirmed { tier, bytes });
        self
    }

    pub(crate) fn backend_cleanup_unconfirmed(&self) -> Option<(StateTier, Option<u64>)> {
        match self.internal {
            Some(InternalProtocolError::BackendCleanupUnconfirmed { tier, bytes }) => {
                Some((tier, bytes))
            }
            None => None,
        }
    }

    pub fn public_body(&self) -> ProtocolErrorBody {
        ProtocolErrorBody {
            code: self.code,
            message: self.message.clone(),
            retryable: self.retryable,
            details: self.details.clone(),
        }
    }
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ProtocolError {}

/// Semantic Werk control plane. Implementations may be local or remote; the
/// contract has no dependency on HTTP, Axum, or filesystem representations.
pub trait WerkControl: Send + Sync {
    fn info(&self, context: ControlContext) -> BoxControlFuture<'_, RuntimeInfo>;
    fn capabilities(&self, context: ControlContext) -> BoxControlFuture<'_, CapabilitiesResponse>;
    fn list_states(
        &self,
        context: ControlContext,
        filter: StateListFilter,
    ) -> BoxControlFuture<'_, StateListResponse>;
    fn state_action(
        &self,
        context: ControlContext,
        state_id: String,
        request: StateActionRequest,
    ) -> BoxControlFuture<'_, StateActionResponse>;
    fn prune_states(
        &self,
        context: ControlContext,
        request: PruneStatesRequest,
    ) -> BoxControlFuture<'_, PruneStatesResponse>;
    fn memory_status(&self, context: ControlContext) -> BoxControlFuture<'_, MemoryStatusResponse>;
    fn list_experts(
        &self,
        context: ControlContext,
        filter: ExpertListFilter,
    ) -> BoxControlFuture<'_, ExpertListResponse>;
    fn expert_action(
        &self,
        context: ControlContext,
        request: ExpertActionRequest,
    ) -> BoxControlFuture<'_, ExpertActionResponse>;
    fn prefill(
        &self,
        context: ControlContext,
        request: PrefillRequest,
    ) -> BoxControlFuture<'_, PrefillResponse>;
    fn decode(
        &self,
        context: ControlContext,
        request: DecodeRequest,
    ) -> BoxControlFuture<'_, DecodeResponse>;
}
