use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fmt, str::FromStr};

pub const PROTOCOL_MAJOR: u16 = 1;
pub const PROTOCOL_MINOR: u16 = 0;
pub const PROTOCOL_VERSION_HEADER: &str = "x-werk-protocol-version";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    pub const V1: Self = Self {
        major: PROTOCOL_MAJOR,
        minor: PROTOCOL_MINOR,
    };

    pub fn accepts(self, producer: Self) -> bool {
        self.major == producer.major && producer.minor <= self.minor
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}", self.major, self.minor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseProtocolVersionError;

impl fmt::Display for ParseProtocolVersionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("protocol version must be MAJOR.MINOR")
    }
}

impl std::error::Error for ParseProtocolVersionError {}

impl FromStr for ProtocolVersion {
    type Err = ParseProtocolVersionError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (major, minor) = value
            .trim()
            .split_once('.')
            .ok_or(ParseProtocolVersionError)?;
        if major.is_empty() || minor.is_empty() || minor.contains('.') {
            return Err(ParseProtocolVersionError);
        }
        Ok(Self {
            major: major.parse().map_err(|_| ParseProtocolVersionError)?,
            minor: minor.parse().map_err(|_| ParseProtocolVersionError)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolEnvelope<T> {
    pub protocol: ProtocolVersion,
    pub request_id: String,
    pub data: T,
}

impl<T> ProtocolEnvelope<T> {
    pub fn v1(request_id: impl Into<String>, data: T) -> Self {
        Self {
            protocol: ProtocolVersion::V1,
            request_id: request_id.into(),
            data,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityStatus {
    Supported,
    Unsupported,
    Unavailable,
    Experimental,
    ExternallyManaged,
    MetadataOnly,
}

impl CapabilityStatus {
    pub fn is_operational(&self, allow_experimental: bool) -> bool {
        matches!(self, Self::Supported)
            || (allow_experimental && matches!(self, Self::Experimental))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    pub id: String,
    pub status: CapabilityStatus,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub operations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolLimits {
    pub max_page_size: u16,
    pub max_state_ids_per_operation: u16,
    pub max_expert_ids_per_operation: u16,
    pub max_request_bytes: u64,
    pub max_handoff_bytes: u64,
    pub max_ttl_seconds: u64,
}

impl Default for ProtocolLimits {
    fn default() -> Self {
        Self {
            max_page_size: 100,
            max_state_ids_per_operation: 100,
            max_expert_ids_per_operation: 256,
            max_request_bytes: 1024 * 1024,
            max_handoff_bytes: 4 * 1024,
            max_ttl_seconds: 30 * 24 * 60 * 60,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeInfo {
    pub service: String,
    pub service_version: String,
    pub protocol: ProtocolVersion,
    pub active_backend: String,
    pub limits: ProtocolLimits,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitiesResponse {
    pub capabilities: Vec<Capability>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTier {
    Vram,
    Ram,
    Disk,
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateStatus {
    Ready,
    Loading,
    Moving,
    Unavailable,
    Quarantined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersistenceMode {
    Ephemeral,
    Memory,
    Disk,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReuseMode {
    Disabled,
    Prefer,
    Required,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistencePolicy {
    #[serde(default = "default_persistence_mode")]
    pub mode: PersistenceMode,
    #[serde(default = "default_reuse_mode")]
    pub reuse: ReuseMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    #[serde(default)]
    pub pin: bool,
}

fn default_persistence_mode() -> PersistenceMode {
    PersistenceMode::Auto
}

fn default_reuse_mode() -> ReuseMode {
    ReuseMode::Prefer
}

impl Default for PersistencePolicy {
    fn default() -> Self {
        Self {
            mode: PersistenceMode::Auto,
            reuse: ReuseMode::Prefer,
            ttl_seconds: None,
            pin: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextCompatibility {
    pub context_size: u64,
    pub batch_size: Option<u64>,
    pub rope_configuration_fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompatibilityEnvelope {
    pub model_fingerprint: String,
    pub tokenizer_fingerprint: String,
    pub prompt_fingerprint: String,
    pub chat_template_fingerprint: Option<String>,
    pub backend: String,
    pub backend_version: String,
    pub runtime_adapter_version: String,
    pub accelerator_family: String,
    pub tensor_dtype: String,
    pub kv_dtype: String,
    pub quantization: String,
    pub cache_layout: String,
    pub block_size: Option<u64>,
    pub context: ContextCompatibility,
    #[serde(default)]
    pub multimodal_processor_fingerprints: Vec<String>,
    pub producer_protocol: ProtocolVersion,
}

impl CompatibilityEnvelope {
    pub fn mismatch_fields(&self, other: &Self) -> Vec<&'static str> {
        let mut fields = Vec::new();
        macro_rules! compare {
            ($field:ident) => {
                if self.$field != other.$field {
                    fields.push(stringify!($field));
                }
            };
        }
        compare!(model_fingerprint);
        compare!(tokenizer_fingerprint);
        compare!(prompt_fingerprint);
        compare!(chat_template_fingerprint);
        compare!(backend);
        compare!(backend_version);
        compare!(runtime_adapter_version);
        compare!(accelerator_family);
        compare!(tensor_dtype);
        compare!(kv_dtype);
        compare!(quantization);
        compare!(cache_layout);
        compare!(block_size);
        compare!(context);
        compare!(multimodal_processor_fingerprints);
        compare!(producer_protocol);
        fields
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSummary {
    pub id: String,
    pub model_id: String,
    pub tier: StateTier,
    pub status: StateStatus,
    pub bytes: Option<u64>,
    pub created_unix_ms: u64,
    pub last_accessed_unix_ms: u64,
    pub expires_unix_ms: Option<u64>,
    pub pinned: bool,
    pub backend: String,
    pub reusable: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateListFilter {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<StateTier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateListResponse {
    pub states: Vec<StateSummary>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateAction {
    Pin,
    Unpin,
    Promote,
    Demote,
    Evict,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateActionRequest {
    pub action: StateAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_tier: Option<StateTier>,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub allow_experimental: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateActionResponse {
    pub state: StateSummary,
    pub changed: bool,
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StateSelector {
    Ids {
        ids: Vec<String>,
    },
    Filter {
        model_id: Option<String>,
        tier: Option<StateTier>,
        older_than_unix_ms: Option<u64>,
    },
    All {
        confirm: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PruneStatesRequest {
    pub selector: StateSelector,
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PruneStatesResponse {
    pub matched: u64,
    pub removed: u64,
    pub bytes: Option<u64>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureLevel {
    Normal,
    Soft,
    Hard,
    Emergency,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryTierStatus {
    pub capacity_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub managed_bytes: u64,
    pub reserved_bytes: u64,
    pub pressure: PressureLevel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryStatusResponse {
    pub observed_at_unix_ms: u64,
    pub overall_pressure: PressureLevel,
    pub topology: String,
    pub host: MemoryTierStatus,
    pub accelerator: MemoryTierStatus,
    pub last_action_unix_ms: Option<u64>,
    pub counters: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertTier {
    Vram,
    Ram,
    External,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertSummary {
    pub id: String,
    pub model_id: String,
    pub tier: ExpertTier,
    pub bytes: Option<u64>,
    pub hotness: f64,
    pub pinned: bool,
    pub last_used_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertListFilter {
    pub model_id: Option<String>,
    pub tier: Option<ExpertTier>,
    pub limit: Option<u16>,
    pub cursor: Option<String>,
    #[serde(default)]
    pub allow_experimental: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertListResponse {
    pub experts: Vec<ExpertSummary>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertAction {
    Prefetch,
    Pin,
    Unpin,
    Evict,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertActionRequest {
    pub model_id: String,
    pub expert_ids: Vec<String>,
    pub action: ExpertAction,
    pub target_tier: Option<ExpertTier>,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub allow_experimental: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertActionResponse {
    pub experts: Vec<ExpertSummary>,
    pub changed: u64,
    pub dry_run: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum PrefillInput {
    Text { text: String },
    Messages { messages: Vec<ProtocolMessage> },
}

impl fmt::Debug for PrefillInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text { text } => formatter
                .debug_struct("Text")
                .field("text", &"[redacted]")
                .field("bytes", &text.len())
                .finish(),
            Self::Messages { messages } => formatter
                .debug_struct("Messages")
                .field("messages", &"[redacted]")
                .field("count", &messages.len())
                .finish(),
        }
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolMessage {
    pub role: String,
    pub content: String,
}

impl fmt::Debug for ProtocolMessage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProtocolMessage")
            .field("role", &self.role)
            .field("content", &"[redacted]")
            .field("content_bytes", &self.content.len())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillRequest {
    pub model_id: String,
    pub input: PrefillInput,
    #[serde(default)]
    pub policy: PersistencePolicy,
    #[serde(default)]
    pub allow_experimental: bool,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillResponse {
    pub handoff: String,
    pub state_id: Option<String>,
    pub prompt_tokens: u64,
    pub reused: bool,
    pub tier: StateTier,
    pub expires_unix_ms: u64,
}

impl fmt::Debug for PrefillResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrefillResponse")
            .field("handoff", &"[redacted]")
            .field("state_id", &self.state_id)
            .field("prompt_tokens", &self.prompt_tokens)
            .field("reused", &self.reused)
            .field("tier", &self.tier)
            .field("expires_unix_ms", &self.expires_unix_ms)
            .finish()
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeRequest {
    pub handoff: String,
    pub max_tokens: u64,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub allow_experimental: bool,
}

impl fmt::Debug for DecodeRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodeRequest")
            .field("handoff", &"[redacted]")
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("seed", &self.seed)
            .field("stop", &"[redacted]")
            .field("stop_count", &self.stop.len())
            .field("allow_experimental", &self.allow_experimental)
            .finish()
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeResponse {
    pub text: String,
    pub handoff: Option<String>,
    pub completion_tokens: u64,
    pub finish_reason: String,
}

impl fmt::Debug for DecodeResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodeResponse")
            .field("text", &"[redacted]")
            .field("text_bytes", &self.text.len())
            .field("handoff", &self.handoff.as_ref().map(|_| "[redacted]"))
            .field("completion_tokens", &self.completion_tokens)
            .field("finish_reason", &self.finish_reason)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_major_must_match_and_minor_is_backward_compatible() {
        assert!(
            ProtocolVersion { major: 1, minor: 3 }.accepts(ProtocolVersion { major: 1, minor: 2 })
        );
        assert!(
            !ProtocolVersion { major: 1, minor: 1 }.accepts(ProtocolVersion { major: 1, minor: 2 })
        );
        assert!(
            !ProtocolVersion { major: 1, minor: 9 }.accepts(ProtocolVersion { major: 2, minor: 0 })
        );
        assert_eq!(ProtocolVersion::V1.to_string(), "1.0");
        assert_eq!("1.0".parse(), Ok(ProtocolVersion::V1));
        assert!("1".parse::<ProtocolVersion>().is_err());
        assert!("1.0.1".parse::<ProtocolVersion>().is_err());
    }

    #[test]
    fn compatibility_reports_every_changed_dimension() {
        let left = compatibility();
        let mut right = left.clone();
        right.backend = "vllm".to_string();
        right.context.context_size = 8192;
        right.multimodal_processor_fingerprints = vec!["sha256:other".to_string()];
        assert_eq!(
            left.mismatch_fields(&right),
            ["backend", "context", "multimodal_processor_fingerprints"]
        );
    }

    #[test]
    fn capability_status_wire_values_cover_every_truthful_state() {
        let statuses = [
            CapabilityStatus::Supported,
            CapabilityStatus::Unsupported,
            CapabilityStatus::Unavailable,
            CapabilityStatus::Experimental,
            CapabilityStatus::ExternallyManaged,
            CapabilityStatus::MetadataOnly,
        ];
        assert_eq!(
            serde_json::to_value(statuses).unwrap(),
            serde_json::json!([
                "supported",
                "unsupported",
                "unavailable",
                "experimental",
                "externally_managed",
                "metadata_only"
            ])
        );
        assert!(CapabilityStatus::Supported.is_operational(false));
        assert!(!CapabilityStatus::Experimental.is_operational(false));
        assert!(CapabilityStatus::Experimental.is_operational(true));
        assert!(!CapabilityStatus::ExternallyManaged.is_operational(true));
        assert!(!CapabilityStatus::MetadataOnly.is_operational(true));
    }

    #[test]
    fn destructive_selector_is_required_and_tagged() {
        assert!(serde_json::from_str::<PruneStatesRequest>(r#"{"dry_run":true}"#).is_err());
        let request: PruneStatesRequest =
            serde_json::from_str(r#"{"selector":{"kind":"ids","ids":["state_1"]},"dry_run":true}"#)
                .unwrap();
        assert!(matches!(request.selector, StateSelector::Ids { .. }));
    }

    #[test]
    fn sensitive_dto_debug_output_is_redacted() {
        let prefill = PrefillRequest {
            model_id: "model".to_string(),
            input: PrefillInput::Text {
                text: "private prompt".to_string(),
            },
            policy: PersistencePolicy::default(),
            allow_experimental: false,
        };
        let decode = DecodeRequest {
            handoff: "handoff-secret".to_string(),
            max_tokens: 1,
            temperature: None,
            top_p: None,
            seed: None,
            stop: vec!["private stop".to_string()],
            allow_experimental: false,
        };
        let response = PrefillResponse {
            handoff: "response-secret".to_string(),
            state_id: None,
            prompt_tokens: 1,
            reused: false,
            tier: StateTier::Ram,
            expires_unix_ms: 1,
        };
        let decode_response = DecodeResponse {
            text: "private generated text".to_string(),
            handoff: Some("next-handoff-secret".to_string()),
            completion_tokens: 1,
            finish_reason: "stop".to_string(),
        };

        let debug = format!("{prefill:?} {decode:?} {response:?} {decode_response:?}");
        for secret in [
            "private prompt",
            "handoff-secret",
            "private stop",
            "response-secret",
            "private generated text",
            "next-handoff-secret",
        ] {
            assert!(!debug.contains(secret));
        }
        assert!(debug.contains("[redacted]"));
    }

    fn compatibility() -> CompatibilityEnvelope {
        CompatibilityEnvelope {
            model_fingerprint: "sha256:model".to_string(),
            tokenizer_fingerprint: "sha256:tokenizer".to_string(),
            prompt_fingerprint: "sha256:prompt".to_string(),
            chat_template_fingerprint: Some("sha256:template".to_string()),
            backend: "llama.cpp".to_string(),
            backend_version: "1".to_string(),
            runtime_adapter_version: "1".to_string(),
            accelerator_family: "cuda".to_string(),
            tensor_dtype: "f16".to_string(),
            kv_dtype: "f16".to_string(),
            quantization: "q4_k_m".to_string(),
            cache_layout: "llama_slot".to_string(),
            block_size: None,
            context: ContextCompatibility {
                context_size: 4096,
                batch_size: Some(512),
                rope_configuration_fingerprint: None,
            },
            multimodal_processor_fingerprints: vec!["sha256:mm".to_string()],
            producer_protocol: ProtocolVersion::V1,
        }
    }
}
