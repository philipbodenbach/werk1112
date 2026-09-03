//! Backend-neutral mixture-of-experts residency policy.
//!
//! This module tracks only public model/expert identifiers and aggregate
//! residency metadata. It does not own backend handles and never claims that a
//! move happened: callers plan an action, claim it with [`ExpertActionPermit`],
//! execute the backend operation, and commit the permit only after success.

use crate::werk_protocol::{
    ExpertAction, ExpertActionRequest, ExpertListFilter, ExpertListResponse, ExpertSummary,
    ExpertTier, ProtocolError, ProtocolErrorCode,
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MAX_MODELS: usize = 4_096;
const MAX_EXPERTS_PER_MODEL: usize = 65_536;
const MAX_TOTAL_EXPERTS: usize = 262_144;
const MAX_ACCESS_BATCH: usize = 4_096;
const MAX_PAGE_SIZE: usize = 1_000;
const MAX_ACTIONS: usize = 4_096;
const MAX_HALF_LIFE_MILLIS: u64 = 30 * 24 * 60 * 60 * 1_000;
const MAX_COOLDOWN_MILLIS: u64 = 24 * 60 * 60 * 1_000;
const MAX_ACCESS_WEIGHT: f64 = 1_000_000.0;
const MAX_HOTNESS: f64 = 1_000_000_000_000.0;
const CURSOR_DOMAIN: &[u8] = b"werk-expert-cursor-v1\0";
const CURSOR_BYTES: usize = 64;

#[derive(Debug, Clone)]
pub(crate) struct ExpertResidencyConfig {
    pub(crate) max_models: usize,
    pub(crate) max_experts_per_model: usize,
    pub(crate) max_total_experts: usize,
    pub(crate) max_access_batch: usize,
    pub(crate) default_page_size: usize,
    pub(crate) max_page_size: usize,
    pub(crate) max_actions: usize,
    /// Half-life of the exponentially decayed access score.
    pub(crate) hotness_half_life_millis: u64,
    /// RAM experts at or above this score are eligible for VRAM prefetch.
    pub(crate) promote_hotness: f64,
    /// VRAM experts at or below this score are eligible for ordinary demotion.
    pub(crate) demote_hotness: f64,
    /// Minimum time between automatic movements of one expert.
    pub(crate) transition_cooldown_millis: u64,
}

impl Default for ExpertResidencyConfig {
    fn default() -> Self {
        Self {
            max_models: 128,
            max_experts_per_model: 4_096,
            max_total_experts: 65_536,
            max_access_batch: 256,
            default_page_size: 100,
            max_page_size: 100,
            max_actions: 256,
            hotness_half_life_millis: 60_000,
            promote_hotness: 2.0,
            demote_hotness: 0.5,
            transition_cooldown_millis: 5_000,
        }
    }
}

impl ExpertResidencyConfig {
    fn validate(&self) -> Result<(), ExpertResidencyError> {
        validate_bound("model", self.max_models, MAX_MODELS)?;
        validate_bound(
            "per-model expert",
            self.max_experts_per_model,
            MAX_EXPERTS_PER_MODEL,
        )?;
        validate_bound("total expert", self.max_total_experts, MAX_TOTAL_EXPERTS)?;
        validate_bound("access batch", self.max_access_batch, MAX_ACCESS_BATCH)?;
        validate_bound("page", self.max_page_size, MAX_PAGE_SIZE)?;
        validate_bound("action", self.max_actions, MAX_ACTIONS)?;
        if self.default_page_size == 0 || self.default_page_size > self.max_page_size {
            return Err(ExpertResidencyError::InvalidConfiguration(
                "default page size must be nonzero and no larger than max_page_size".to_string(),
            ));
        }
        if self.hotness_half_life_millis == 0
            || self.hotness_half_life_millis > MAX_HALF_LIFE_MILLIS
        {
            return Err(ExpertResidencyError::InvalidConfiguration(format!(
                "hotness half-life must be between 1 and {MAX_HALF_LIFE_MILLIS} milliseconds"
            )));
        }
        if self.transition_cooldown_millis > MAX_COOLDOWN_MILLIS {
            return Err(ExpertResidencyError::InvalidConfiguration(format!(
                "transition cooldown must not exceed {MAX_COOLDOWN_MILLIS} milliseconds"
            )));
        }
        if !self.demote_hotness.is_finite()
            || !self.promote_hotness.is_finite()
            || self.demote_hotness < 0.0
            || self.demote_hotness >= self.promote_hotness
        {
            return Err(ExpertResidencyError::InvalidConfiguration(
                "hotness thresholds must satisfy 0 <= demote < promote".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_bound(label: &str, value: usize, maximum: usize) -> Result<(), ExpertResidencyError> {
    if value == 0 || value > maximum {
        return Err(ExpertResidencyError::InvalidConfiguration(format!(
            "{label} limit must be between 1 and {maximum}"
        )));
    }
    Ok(())
}

pub(crate) trait ExpertClock: Send + Sync {
    fn monotonic_millis(&self) -> u64;
    fn unix_millis(&self) -> u64;
}

#[derive(Debug)]
pub(crate) struct SystemExpertClock {
    started: Instant,
    unix_at_start_millis: u64,
}

impl SystemExpertClock {
    pub(crate) fn new() -> Self {
        let unix_at_start_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        Self {
            started: Instant::now(),
            unix_at_start_millis,
        }
    }
}

impl Default for SystemExpertClock {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertClock for SystemExpertClock {
    fn monotonic_millis(&self) -> u64 {
        self.started
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }

    fn unix_millis(&self) -> u64 {
        self.unix_at_start_millis
            .saturating_add(self.monotonic_millis())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpertObservation {
    pub model_id: String,
    pub expert_id: String,
    pub tier: ExpertTier,
    pub bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExpertAccess {
    pub expert_id: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertPressureLevel {
    Soft,
    Hard,
    Emergency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpertPressureRequest {
    pub source_tier: ExpertTier,
    pub level: ExpertPressureLevel,
    pub relief_bytes: u64,
    /// Capacity already reserved by the caller for VRAM -> RAM demotions.
    pub ram_headroom_bytes: u64,
    pub max_actions: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExpertPolicyAction {
    Move {
        model_id: String,
        expert_id: String,
        from: ExpertTier,
        to: ExpertTier,
        bytes: Option<u64>,
        revision: u64,
    },
    Evict {
        model_id: String,
        expert_id: String,
        from: ExpertTier,
        bytes: Option<u64>,
        revision: u64,
    },
}

impl ExpertPolicyAction {
    pub(crate) fn model_id(&self) -> &str {
        match self {
            Self::Move { model_id, .. } | Self::Evict { model_id, .. } => model_id,
        }
    }

    pub(crate) fn expert_id(&self) -> &str {
        match self {
            Self::Move { expert_id, .. } | Self::Evict { expert_id, .. } => expert_id,
        }
    }

    pub(crate) fn source_tier(&self) -> ExpertTier {
        match self {
            Self::Move { from, .. } | Self::Evict { from, .. } => *from,
        }
    }

    pub(crate) fn target_tier(&self) -> Option<ExpertTier> {
        match self {
            Self::Move { to, .. } => Some(*to),
            Self::Evict { .. } => None,
        }
    }

    pub(crate) fn bytes(&self) -> Option<u64> {
        match self {
            Self::Move { bytes, .. } | Self::Evict { bytes, .. } => *bytes,
        }
    }

    fn revision(&self) -> u64 {
        match self {
            Self::Move { revision, .. } | Self::Evict { revision, .. } => *revision,
        }
    }

    fn key(&self) -> ExpertKey {
        ExpertKey {
            model_id: self.model_id().to_string(),
            expert_id: self.expert_id().to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpertCandidatePlan {
    pub target_tier: ExpertTier,
    pub actions: Vec<ExpertPolicyAction>,
    pub deferred_by_cooldown: usize,
    pub deferred_by_hysteresis: usize,
    pub protected_by_pin: usize,
    pub in_flight: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpertPressurePlan {
    pub source_tier: ExpertTier,
    pub level: ExpertPressureLevel,
    pub actions: Vec<ExpertPolicyAction>,
    pub requested_relief_bytes: u64,
    pub known_planned_relief_bytes: u64,
    pub unresolved_relief_bytes: u64,
    pub unknown_size_actions: usize,
    pub deferred_by_cooldown: usize,
    pub deferred_by_hysteresis: usize,
    pub protected_by_pin: usize,
    pub in_flight: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExpertCommitOutcome {
    pub updated: Vec<ExpertSummary>,
    pub evicted: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExpertResidencyError {
    InvalidConfiguration(String),
    InvalidModelId,
    InvalidExpertId,
    InvalidAccessWeight,
    InvalidTierTransition { from: ExpertTier, to: ExpertTier },
    InvalidCursor,
    InvalidLimit { label: &'static str, maximum: usize },
    UnknownModel(String),
    UnknownExpert { model_id: String, expert_id: String },
    ModelLimitReached(usize),
    ModelExpertLimitReached { model_id: String, limit: usize },
    TotalExpertLimitReached(usize),
    ActionInFlight { model_id: String, expert_id: String },
    StaleAction { model_id: String, expert_id: String },
    PinnedExpert { model_id: String, expert_id: String },
    DuplicateExpert { model_id: String, expert_id: String },
    CursorCollision,
}

impl fmt::Display for ExpertResidencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(
                    formatter,
                    "invalid expert residency configuration: {message}"
                )
            }
            Self::InvalidModelId => formatter.write_str("invalid model identifier"),
            Self::InvalidExpertId => formatter.write_str("invalid expert identifier"),
            Self::InvalidAccessWeight => formatter.write_str("invalid expert access weight"),
            Self::InvalidTierTransition { from, to } => {
                write!(
                    formatter,
                    "invalid expert tier transition {from:?} -> {to:?}"
                )
            }
            Self::InvalidCursor => formatter.write_str("invalid expert list cursor"),
            Self::InvalidLimit { label, maximum } => {
                write!(formatter, "{label} limit must be between 1 and {maximum}")
            }
            Self::UnknownModel(model_id) => write!(formatter, "model {model_id} is not tracked"),
            Self::UnknownExpert {
                model_id,
                expert_id,
            } => write!(
                formatter,
                "expert {expert_id} for model {model_id} is not tracked"
            ),
            Self::ModelLimitReached(limit) => {
                write!(formatter, "tracked model limit {limit} was reached")
            }
            Self::ModelExpertLimitReached { model_id, limit } => write!(
                formatter,
                "tracked expert limit {limit} for model {model_id} was reached"
            ),
            Self::TotalExpertLimitReached(limit) => {
                write!(formatter, "total tracked expert limit {limit} was reached")
            }
            Self::ActionInFlight {
                model_id,
                expert_id,
            } => write!(
                formatter,
                "expert {expert_id} for model {model_id} already has an action in flight"
            ),
            Self::StaleAction {
                model_id,
                expert_id,
            } => write!(
                formatter,
                "expert action for {expert_id} on model {model_id} is stale"
            ),
            Self::PinnedExpert {
                model_id,
                expert_id,
            } => write!(
                formatter,
                "expert {expert_id} for model {model_id} is pinned"
            ),
            Self::DuplicateExpert {
                model_id,
                expert_id,
            } => write!(
                formatter,
                "expert {expert_id} for model {model_id} appears more than once"
            ),
            Self::CursorCollision => formatter.write_str("expert cursor hash collision"),
        }
    }
}

impl Error for ExpertResidencyError {}

impl From<ExpertResidencyError> for ProtocolError {
    fn from(error: ExpertResidencyError) -> Self {
        let code = match error {
            ExpertResidencyError::UnknownModel(_) | ExpertResidencyError::UnknownExpert { .. } => {
                ProtocolErrorCode::NotFound
            }
            ExpertResidencyError::ModelLimitReached(_)
            | ExpertResidencyError::ModelExpertLimitReached { .. }
            | ExpertResidencyError::TotalExpertLimitReached(_) => {
                ProtocolErrorCode::ResourceExhausted
            }
            ExpertResidencyError::ActionInFlight { .. }
            | ExpertResidencyError::StaleAction { .. }
            | ExpertResidencyError::PinnedExpert { .. } => ProtocolErrorCode::Conflict,
            ExpertResidencyError::CursorCollision => ProtocolErrorCode::Internal,
            ExpertResidencyError::InvalidConfiguration(_)
            | ExpertResidencyError::InvalidModelId
            | ExpertResidencyError::InvalidExpertId
            | ExpertResidencyError::InvalidAccessWeight
            | ExpertResidencyError::InvalidTierTransition { .. }
            | ExpertResidencyError::InvalidCursor
            | ExpertResidencyError::InvalidLimit { .. }
            | ExpertResidencyError::DuplicateExpert { .. } => ProtocolErrorCode::InvalidRequest,
        };
        ProtocolError::new(code, error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ExpertKey {
    model_id: String,
    expert_id: String,
}

#[derive(Debug, Clone)]
struct TrackedExpert {
    tier: ExpertTier,
    bytes: Option<u64>,
    hotness: f64,
    hotness_updated_millis: u64,
    pinned: bool,
    last_used_monotonic_millis: Option<u64>,
    last_used_unix_millis: Option<u64>,
    last_transition_millis: Option<u64>,
    revision: u64,
}

#[derive(Debug, Default)]
struct ModelExperts {
    experts: BTreeMap<String, TrackedExpert>,
}

#[derive(Debug, Default)]
struct ExpertState {
    models: BTreeMap<String, ModelExperts>,
    pending: BTreeSet<ExpertKey>,
    total_experts: usize,
}

struct ExpertInner {
    config: ExpertResidencyConfig,
    clock: Arc<dyn ExpertClock>,
    state: Mutex<ExpertState>,
}

#[derive(Clone)]
pub(crate) struct ExpertResidencyManager {
    inner: Arc<ExpertInner>,
}

impl fmt::Debug for ExpertResidencyManager {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = lock_state(&self.inner.state);
        formatter
            .debug_struct("ExpertResidencyManager")
            .field("models", &state.models.len())
            .field("experts", &state.total_experts)
            .field("actions_in_flight", &state.pending.len())
            .finish_non_exhaustive()
    }
}

impl ExpertResidencyManager {
    pub(crate) fn new(
        config: ExpertResidencyConfig,
        clock: Arc<dyn ExpertClock>,
    ) -> Result<Self, ExpertResidencyError> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(ExpertInner {
                config,
                clock,
                state: Mutex::new(ExpertState::default()),
            }),
        })
    }

    pub(crate) fn with_system_clock(
        config: ExpertResidencyConfig,
    ) -> Result<Self, ExpertResidencyError> {
        Self::new(config, Arc::new(SystemExpertClock::new()))
    }

    /// Adds or refreshes backend-observed residency. Existing hotness and pins
    /// remain local policy state; an observed tier change starts cooldown.
    pub(crate) fn observe(
        &self,
        observation: ExpertObservation,
    ) -> Result<ExpertSummary, ExpertResidencyError> {
        validate_model_id(&observation.model_id)?;
        validate_expert_id(&observation.expert_id)?;
        if observation.bytes == Some(0) {
            return Err(ExpertResidencyError::InvalidConfiguration(
                "expert byte size must be greater than zero when known".to_string(),
            ));
        }
        let now_mono = self.inner.clock.monotonic_millis();
        let key = ExpertKey {
            model_id: observation.model_id.clone(),
            expert_id: observation.expert_id.clone(),
        };
        let mut state = lock_state(&self.inner.state);
        if state.pending.contains(&key) {
            return Err(action_in_flight(&key));
        }

        if let Some(model) = state.models.get_mut(&observation.model_id) {
            if let Some(expert) = model.experts.get_mut(&observation.expert_id) {
                let mut changed = false;
                if expert.tier != observation.tier {
                    expert.tier = observation.tier;
                    expert.last_transition_millis = Some(now_mono);
                    changed = true;
                }
                if expert.bytes != observation.bytes {
                    expert.bytes = observation.bytes;
                    changed = true;
                }
                if changed {
                    expert.revision = expert.revision.saturating_add(1);
                }
                return Ok(summary_from_entry(
                    &observation.model_id,
                    &observation.expert_id,
                    expert,
                    now_mono,
                    self.inner.config.hotness_half_life_millis,
                ));
            }
        }

        if !state.models.contains_key(&observation.model_id)
            && state.models.len() >= self.inner.config.max_models
        {
            return Err(ExpertResidencyError::ModelLimitReached(
                self.inner.config.max_models,
            ));
        }
        if state.total_experts >= self.inner.config.max_total_experts {
            return Err(ExpertResidencyError::TotalExpertLimitReached(
                self.inner.config.max_total_experts,
            ));
        }
        let model = state
            .models
            .entry(observation.model_id.clone())
            .or_default();
        if model.experts.len() >= self.inner.config.max_experts_per_model {
            return Err(ExpertResidencyError::ModelExpertLimitReached {
                model_id: observation.model_id,
                limit: self.inner.config.max_experts_per_model,
            });
        }
        let entry = TrackedExpert {
            tier: observation.tier,
            bytes: observation.bytes,
            hotness: 0.0,
            hotness_updated_millis: now_mono,
            pinned: false,
            last_used_monotonic_millis: None,
            last_used_unix_millis: None,
            last_transition_millis: None,
            revision: 1,
        };
        let summary = summary_from_entry(
            &observation.model_id,
            &observation.expert_id,
            &entry,
            now_mono,
            self.inner.config.hotness_half_life_millis,
        );
        model.experts.insert(observation.expert_id, entry);
        state.total_experts = state.total_experts.saturating_add(1);
        Ok(summary)
    }

    pub(crate) fn record_access(
        &self,
        model_id: &str,
        expert_id: &str,
    ) -> Result<ExpertSummary, ExpertResidencyError> {
        let mut summaries = self.record_accesses(
            model_id,
            &[ExpertAccess {
                expert_id: expert_id.to_string(),
                weight: 1.0,
            }],
        )?;
        Ok(summaries
            .pop()
            .expect("one validated expert access produces one summary"))
    }

    /// Atomically records a bounded access batch. Duplicate expert IDs are
    /// combined, and validation completes before any score is changed.
    pub(crate) fn record_accesses(
        &self,
        model_id: &str,
        accesses: &[ExpertAccess],
    ) -> Result<Vec<ExpertSummary>, ExpertResidencyError> {
        validate_model_id(model_id)?;
        if accesses.is_empty() || accesses.len() > self.inner.config.max_access_batch {
            return Err(ExpertResidencyError::InvalidLimit {
                label: "access batch",
                maximum: self.inner.config.max_access_batch,
            });
        }
        let mut combined = BTreeMap::<String, f64>::new();
        for access in accesses {
            validate_expert_id(&access.expert_id)?;
            if !access.weight.is_finite()
                || access.weight <= 0.0
                || access.weight > MAX_ACCESS_WEIGHT
            {
                return Err(ExpertResidencyError::InvalidAccessWeight);
            }
            let value = combined.entry(access.expert_id.clone()).or_default();
            *value += access.weight;
            if !value.is_finite() || *value > MAX_ACCESS_WEIGHT {
                return Err(ExpertResidencyError::InvalidAccessWeight);
            }
        }

        let now_mono = self.inner.clock.monotonic_millis();
        let now_unix = self.inner.clock.unix_millis();
        let mut state = lock_state(&self.inner.state);
        let model = state
            .models
            .get(model_id)
            .ok_or_else(|| ExpertResidencyError::UnknownModel(model_id.to_string()))?;
        for expert_id in combined.keys() {
            if !model.experts.contains_key(expert_id) {
                return Err(unknown_expert(model_id, expert_id));
            }
            let key = ExpertKey {
                model_id: model_id.to_string(),
                expert_id: expert_id.clone(),
            };
            if state.pending.contains(&key) {
                return Err(action_in_flight(&key));
            }
        }

        let model = state
            .models
            .get_mut(model_id)
            .expect("model existence was checked above");
        let mut summaries = Vec::with_capacity(combined.len());
        for (expert_id, weight) in combined {
            let expert = model
                .experts
                .get_mut(&expert_id)
                .expect("expert existence was checked above");
            expert.hotness =
                (decayed_hotness(expert, now_mono, self.inner.config.hotness_half_life_millis)
                    + weight)
                    .min(MAX_HOTNESS);
            expert.hotness_updated_millis = now_mono;
            expert.last_used_monotonic_millis = Some(now_mono);
            expert.last_used_unix_millis = Some(now_unix);
            expert.revision = expert.revision.saturating_add(1);
            summaries.push(summary_from_entry(
                model_id,
                &expert_id,
                expert,
                now_mono,
                self.inner.config.hotness_half_life_millis,
            ));
        }
        Ok(summaries)
    }

    /// Applies pin state atomically. Dry-run summaries show the proposed state
    /// without changing the tracked entries.
    pub(crate) fn set_pinned(
        &self,
        model_id: &str,
        expert_ids: &[String],
        pinned: bool,
        dry_run: bool,
    ) -> Result<Vec<ExpertSummary>, ExpertResidencyError> {
        validate_model_id(model_id)?;
        validate_id_batch(model_id, expert_ids, self.inner.config.max_actions)?;
        let now = self.inner.clock.monotonic_millis();
        let mut state = lock_state(&self.inner.state);
        let model = state
            .models
            .get(model_id)
            .ok_or_else(|| ExpertResidencyError::UnknownModel(model_id.to_string()))?;
        for expert_id in expert_ids {
            if !model.experts.contains_key(expert_id) {
                return Err(unknown_expert(model_id, expert_id));
            }
            let key = ExpertKey {
                model_id: model_id.to_string(),
                expert_id: expert_id.clone(),
            };
            if state.pending.contains(&key) {
                return Err(action_in_flight(&key));
            }
        }

        let model = state
            .models
            .get_mut(model_id)
            .expect("model existence was checked above");
        let mut summaries = Vec::with_capacity(expert_ids.len());
        for expert_id in expert_ids {
            let expert = model
                .experts
                .get_mut(expert_id)
                .expect("expert existence was checked above");
            if !dry_run && expert.pinned != pinned {
                expert.pinned = pinned;
                expert.revision = expert.revision.saturating_add(1);
            }
            let mut summary = summary_from_entry(
                model_id,
                expert_id,
                expert,
                now,
                self.inner.config.hotness_half_life_millis,
            );
            if dry_run {
                summary.pinned = pinned;
            }
            summaries.push(summary);
        }
        Ok(summaries)
    }

    /// Selects hot RAM -> VRAM prefetches or cold VRAM -> RAM movements. The
    /// caller must reserve target memory before claiming and executing a move.
    pub(crate) fn prefetch_candidates(
        &self,
        model_id: &str,
        target_tier: ExpertTier,
        limit: usize,
    ) -> Result<ExpertCandidatePlan, ExpertResidencyError> {
        validate_model_id(model_id)?;
        validate_action_limit(limit, self.inner.config.max_actions)?;
        let source_tier = opposite_memory_tier(target_tier)?;
        let now = self.inner.clock.monotonic_millis();
        let state = lock_state(&self.inner.state);
        let model = state
            .models
            .get(model_id)
            .ok_or_else(|| ExpertResidencyError::UnknownModel(model_id.to_string()))?;
        let mut candidates = Vec::new();
        let mut deferred_by_cooldown = 0;
        let mut deferred_by_hysteresis = 0;
        let mut protected_by_pin = 0;
        let mut in_flight = 0;
        for (expert_id, expert) in &model.experts {
            if expert.tier != source_tier {
                continue;
            }
            if target_tier == ExpertTier::Ram && expert.pinned {
                protected_by_pin += 1;
                continue;
            }
            let key = ExpertKey {
                model_id: model_id.to_string(),
                expert_id: expert_id.clone(),
            };
            if state.pending.contains(&key) {
                in_flight += 1;
                continue;
            }
            if in_cooldown(expert, now, self.inner.config.transition_cooldown_millis) {
                deferred_by_cooldown += 1;
                continue;
            }
            let hotness = decayed_hotness(expert, now, self.inner.config.hotness_half_life_millis);
            let crosses_hysteresis = match target_tier {
                ExpertTier::Vram => hotness >= self.inner.config.promote_hotness,
                ExpertTier::Ram => hotness <= self.inner.config.demote_hotness,
                ExpertTier::External => unreachable!("target was validated above"),
            };
            if !crosses_hysteresis {
                deferred_by_hysteresis += 1;
                continue;
            }
            candidates.push(ScoredCandidate {
                model_id: model_id.to_string(),
                expert_id: expert_id.clone(),
                expert: expert.clone(),
                hotness,
            });
        }
        sort_movement_candidates(&mut candidates, target_tier);
        let actions = candidates
            .into_iter()
            .take(limit)
            .map(|candidate| ExpertPolicyAction::Move {
                model_id: candidate.model_id,
                expert_id: candidate.expert_id,
                from: source_tier,
                to: target_tier,
                bytes: candidate.expert.bytes,
                revision: candidate.expert.revision,
            })
            .collect();
        Ok(ExpertCandidatePlan {
            target_tier,
            actions,
            deferred_by_cooldown,
            deferred_by_hysteresis,
            protected_by_pin,
            in_flight,
        })
    }

    /// Plans deterministic cold-first pressure relief. Hard pressure respects
    /// the hotness hysteresis and cooldown. Emergency pressure bypasses those
    /// two policy guards, but never bypasses pins or an in-flight operation.
    pub(crate) fn plan_pressure(
        &self,
        request: ExpertPressureRequest,
    ) -> Result<ExpertPressurePlan, ExpertResidencyError> {
        validate_action_limit(request.max_actions, self.inner.config.max_actions)?;
        if request.source_tier == ExpertTier::External {
            return Err(ExpertResidencyError::InvalidTierTransition {
                from: ExpertTier::External,
                to: ExpertTier::External,
            });
        }
        let now = self.inner.clock.monotonic_millis();
        let state = lock_state(&self.inner.state);
        let mut candidates = Vec::new();
        let mut deferred_by_cooldown = 0;
        let mut deferred_by_hysteresis = 0;
        let mut protected_by_pin = 0;
        let mut in_flight = 0;
        for (model_id, model) in &state.models {
            for (expert_id, expert) in &model.experts {
                if expert.tier != request.source_tier {
                    continue;
                }
                if expert.pinned {
                    protected_by_pin += 1;
                    continue;
                }
                let key = ExpertKey {
                    model_id: model_id.clone(),
                    expert_id: expert_id.clone(),
                };
                if state.pending.contains(&key) {
                    in_flight += 1;
                    continue;
                }
                let hotness =
                    decayed_hotness(expert, now, self.inner.config.hotness_half_life_millis);
                if request.level != ExpertPressureLevel::Emergency {
                    if in_cooldown(expert, now, self.inner.config.transition_cooldown_millis) {
                        deferred_by_cooldown += 1;
                        continue;
                    }
                    if request.level == ExpertPressureLevel::Soft
                        || hotness > self.inner.config.demote_hotness
                    {
                        deferred_by_hysteresis += 1;
                        continue;
                    }
                }
                candidates.push(ScoredCandidate {
                    model_id: model_id.clone(),
                    expert_id: expert_id.clone(),
                    expert: expert.clone(),
                    hotness,
                });
            }
        }
        candidates.sort_by(cold_candidate_order);

        let mut actions = Vec::new();
        let mut known_relief = 0_u64;
        let mut unknown_size_actions = 0;
        let mut ram_headroom = request.ram_headroom_bytes;
        let mut demoted = BTreeSet::new();

        // Prefer non-destructive VRAM -> RAM demotion whenever the caller has
        // already established enough RAM headroom. Unknown sizes cannot be
        // admitted safely and therefore remain candidates for eviction.
        if request.source_tier == ExpertTier::Vram {
            for candidate in &candidates {
                if actions.len() >= request.max_actions || known_relief >= request.relief_bytes {
                    break;
                }
                let Some(bytes) = candidate.expert.bytes else {
                    continue;
                };
                if bytes > ram_headroom {
                    continue;
                }
                ram_headroom = ram_headroom.saturating_sub(bytes);
                known_relief = known_relief.saturating_add(bytes);
                demoted.insert((candidate.model_id.clone(), candidate.expert_id.clone()));
                actions.push(ExpertPolicyAction::Move {
                    model_id: candidate.model_id.clone(),
                    expert_id: candidate.expert_id.clone(),
                    from: ExpertTier::Vram,
                    to: ExpertTier::Ram,
                    bytes: Some(bytes),
                    revision: candidate.expert.revision,
                });
            }
        }

        for candidate in candidates {
            if actions.len() >= request.max_actions || known_relief >= request.relief_bytes {
                break;
            }
            if demoted.contains(&(candidate.model_id.clone(), candidate.expert_id.clone())) {
                continue;
            }
            match candidate.expert.bytes {
                Some(bytes) => known_relief = known_relief.saturating_add(bytes),
                None => unknown_size_actions += 1,
            }
            actions.push(ExpertPolicyAction::Evict {
                model_id: candidate.model_id,
                expert_id: candidate.expert_id,
                from: request.source_tier,
                bytes: candidate.expert.bytes,
                revision: candidate.expert.revision,
            });
        }
        Ok(ExpertPressurePlan {
            source_tier: request.source_tier,
            level: request.level,
            actions,
            requested_relief_bytes: request.relief_bytes,
            known_planned_relief_bytes: known_relief,
            unresolved_relief_bytes: request.relief_bytes.saturating_sub(known_relief),
            unknown_size_actions,
            deferred_by_cooldown,
            deferred_by_hysteresis,
            protected_by_pin,
            in_flight,
        })
    }

    /// Builds an explicit move or eviction action without applying automatic
    /// hotness/cooldown policy. Pins continue to protect eviction.
    pub(crate) fn explicit_action(
        &self,
        model_id: &str,
        expert_id: &str,
        target_tier: Option<ExpertTier>,
    ) -> Result<ExpertPolicyAction, ExpertResidencyError> {
        validate_model_id(model_id)?;
        validate_expert_id(expert_id)?;
        let state = lock_state(&self.inner.state);
        let expert = lookup_expert(&state, model_id, expert_id)?;
        let key = ExpertKey {
            model_id: model_id.to_string(),
            expert_id: expert_id.to_string(),
        };
        if state.pending.contains(&key) {
            return Err(action_in_flight(&key));
        }
        match target_tier {
            Some(to) => {
                validate_memory_transition(expert.tier, to)?;
                Ok(ExpertPolicyAction::Move {
                    model_id: model_id.to_string(),
                    expert_id: expert_id.to_string(),
                    from: expert.tier,
                    to,
                    bytes: expert.bytes,
                    revision: expert.revision,
                })
            }
            None => {
                if expert.tier == ExpertTier::External {
                    return Err(ExpertResidencyError::InvalidTierTransition {
                        from: ExpertTier::External,
                        to: ExpertTier::External,
                    });
                }
                if expert.pinned {
                    return Err(ExpertResidencyError::PinnedExpert {
                        model_id: model_id.to_string(),
                        expert_id: expert_id.to_string(),
                    });
                }
                Ok(ExpertPolicyAction::Evict {
                    model_id: model_id.to_string(),
                    expert_id: expert_id.to_string(),
                    from: expert.tier,
                    bytes: expert.bytes,
                    revision: expert.revision,
                })
            }
        }
    }

    /// Claims an action batch before backend work. Dropping the permit rolls
    /// the claims back. Commit only after the whole backend operation succeeds.
    pub(crate) fn begin_actions(
        &self,
        actions: &[ExpertPolicyAction],
    ) -> Result<ExpertActionPermit, ExpertResidencyError> {
        validate_action_limit(actions.len(), self.inner.config.max_actions)?;
        let mut unique = BTreeSet::new();
        let mut state = lock_state(&self.inner.state);
        for action in actions {
            let key = action.key();
            if !unique.insert(key.clone()) {
                return Err(ExpertResidencyError::DuplicateExpert {
                    model_id: key.model_id,
                    expert_id: key.expert_id,
                });
            }
            validate_planned_action(&state, action)?;
        }
        for key in unique {
            state.pending.insert(key);
        }
        Ok(ExpertActionPermit {
            inner: Arc::clone(&self.inner),
            actions: actions.to_vec(),
            finished: false,
        })
    }

    pub(crate) fn begin_action(
        &self,
        action: &ExpertPolicyAction,
    ) -> Result<ExpertActionPermit, ExpertResidencyError> {
        self.begin_actions(std::slice::from_ref(action))
    }

    pub(crate) fn list(
        &self,
        filter: &ExpertListFilter,
    ) -> Result<ExpertListResponse, ExpertResidencyError> {
        if let Some(model_id) = &filter.model_id {
            validate_model_id(model_id)?;
        }
        let limit = usize::from(
            filter.limit.unwrap_or(
                self.inner
                    .config
                    .default_page_size
                    .try_into()
                    .unwrap_or(u16::MAX),
            ),
        );
        if limit == 0 || limit > self.inner.config.max_page_size {
            return Err(ExpertResidencyError::InvalidLimit {
                label: "page",
                maximum: self.inner.config.max_page_size,
            });
        }
        let filter_hash = hash_filter(filter.model_id.as_deref(), filter.tier);
        let after = filter
            .cursor
            .as_deref()
            .map(|cursor| decode_cursor(cursor, &filter_hash))
            .transpose()?;
        let now = self.inner.clock.monotonic_millis();
        let state = lock_state(&self.inner.state);
        let mut rows = Vec::<([u8; 32], ExpertSummary)>::new();
        for (model_id, model) in &state.models {
            if filter
                .model_id
                .as_deref()
                .is_some_and(|wanted| wanted != model_id)
            {
                continue;
            }
            for (expert_id, expert) in &model.experts {
                if filter.tier.is_some_and(|tier| tier != expert.tier) {
                    continue;
                }
                let key_hash = hash_key(model_id, expert_id);
                if after.is_some_and(|after| key_hash <= after) {
                    continue;
                }
                rows.push((
                    key_hash,
                    summary_from_entry(
                        model_id,
                        expert_id,
                        expert,
                        now,
                        self.inner.config.hotness_half_life_millis,
                    ),
                ));
            }
        }
        rows.sort_by_key(|(hash, _)| *hash);
        if rows.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(ExpertResidencyError::CursorCollision);
        }
        let has_more = rows.len() > limit;
        rows.truncate(limit);
        let next_cursor = if has_more {
            rows.last()
                .map(|(key_hash, _)| encode_cursor(&filter_hash, key_hash))
        } else {
            None
        };
        Ok(ExpertListResponse {
            experts: rows.into_iter().map(|(_, summary)| summary).collect(),
            next_cursor,
        })
    }

    #[cfg(test)]
    fn tracked_counts(&self) -> (usize, usize, usize) {
        let state = lock_state(&self.inner.state);
        (state.models.len(), state.total_experts, state.pending.len())
    }
}

pub(crate) struct ExpertActionPermit {
    inner: Arc<ExpertInner>,
    actions: Vec<ExpertPolicyAction>,
    finished: bool,
}

impl fmt::Debug for ExpertActionPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExpertActionPermit")
            .field("actions", &self.actions.len())
            .field("finished", &self.finished)
            .finish()
    }
}

impl ExpertActionPermit {
    pub(crate) fn actions(&self) -> &[ExpertPolicyAction] {
        &self.actions
    }

    pub(crate) fn commit(mut self) -> Result<ExpertCommitOutcome, ExpertResidencyError> {
        let now = self.inner.clock.monotonic_millis();
        let mut state = lock_state(&self.inner.state);
        for action in &self.actions {
            validate_committable_action(&state, action)?;
        }

        let mut updated = Vec::new();
        let mut evicted = Vec::new();
        for action in &self.actions {
            match action {
                ExpertPolicyAction::Move {
                    model_id,
                    expert_id,
                    to,
                    ..
                } => {
                    let expert = state
                        .models
                        .get_mut(model_id)
                        .and_then(|model| model.experts.get_mut(expert_id))
                        .expect("commit validation guarantees expert existence");
                    expert.tier = *to;
                    expert.last_transition_millis = Some(now);
                    expert.revision = expert.revision.saturating_add(1);
                    updated.push(summary_from_entry(
                        model_id,
                        expert_id,
                        expert,
                        now,
                        self.inner.config.hotness_half_life_millis,
                    ));
                }
                ExpertPolicyAction::Evict {
                    model_id,
                    expert_id,
                    ..
                } => {
                    let removed = state
                        .models
                        .get_mut(model_id)
                        .and_then(|model| model.experts.remove(expert_id));
                    debug_assert!(removed.is_some());
                    state.total_experts = state.total_experts.saturating_sub(1);
                    evicted.push((model_id.clone(), expert_id.clone()));
                }
            }
        }
        let empty_models = state
            .models
            .iter()
            .filter_map(|(model_id, model)| model.experts.is_empty().then(|| model_id.clone()))
            .collect::<Vec<_>>();
        for model_id in empty_models {
            state.models.remove(&model_id);
        }
        for action in &self.actions {
            state.pending.remove(&action.key());
        }
        self.finished = true;
        Ok(ExpertCommitOutcome { updated, evicted })
    }
}

impl Drop for ExpertActionPermit {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let mut state = lock_state(&self.inner.state);
        for action in &self.actions {
            state.pending.remove(&action.key());
        }
    }
}

#[derive(Clone)]
struct ScoredCandidate {
    model_id: String,
    expert_id: String,
    expert: TrackedExpert,
    hotness: f64,
}

fn sort_movement_candidates(candidates: &mut [ScoredCandidate], target_tier: ExpertTier) {
    match target_tier {
        ExpertTier::Vram => candidates.sort_by(|left, right| {
            right
                .hotness
                .total_cmp(&left.hotness)
                .then_with(|| {
                    right
                        .expert
                        .last_used_monotonic_millis
                        .cmp(&left.expert.last_used_monotonic_millis)
                })
                .then_with(|| left.model_id.cmp(&right.model_id))
                .then_with(|| left.expert_id.cmp(&right.expert_id))
        }),
        ExpertTier::Ram => candidates.sort_by(cold_candidate_order),
        ExpertTier::External => unreachable!("external target is rejected"),
    }
}

fn cold_candidate_order(left: &ScoredCandidate, right: &ScoredCandidate) -> std::cmp::Ordering {
    left.hotness
        .total_cmp(&right.hotness)
        .then_with(|| {
            left.expert
                .last_used_monotonic_millis
                .cmp(&right.expert.last_used_monotonic_millis)
        })
        .then_with(|| left.model_id.cmp(&right.model_id))
        .then_with(|| left.expert_id.cmp(&right.expert_id))
}

fn validate_planned_action(
    state: &ExpertState,
    action: &ExpertPolicyAction,
) -> Result<(), ExpertResidencyError> {
    let key = action.key();
    if state.pending.contains(&key) {
        return Err(action_in_flight(&key));
    }
    let expert = lookup_expert(state, action.model_id(), action.expert_id())?;
    if expert.revision != action.revision()
        || expert.tier != action.source_tier()
        || expert.bytes != action.bytes()
    {
        return Err(stale_action(&key));
    }
    match action {
        ExpertPolicyAction::Move { from, to, .. } => validate_memory_transition(*from, *to),
        ExpertPolicyAction::Evict { .. } => {
            if expert.tier == ExpertTier::External {
                return Err(ExpertResidencyError::InvalidTierTransition {
                    from: ExpertTier::External,
                    to: ExpertTier::External,
                });
            }
            if expert.pinned {
                return Err(ExpertResidencyError::PinnedExpert {
                    model_id: key.model_id,
                    expert_id: key.expert_id,
                });
            }
            Ok(())
        }
    }
}

fn validate_committable_action(
    state: &ExpertState,
    action: &ExpertPolicyAction,
) -> Result<(), ExpertResidencyError> {
    let key = action.key();
    if !state.pending.contains(&key) {
        return Err(stale_action(&key));
    }
    let expert = lookup_expert(state, action.model_id(), action.expert_id())?;
    if expert.revision != action.revision()
        || expert.tier != action.source_tier()
        || expert.bytes != action.bytes()
    {
        return Err(stale_action(&key));
    }
    Ok(())
}

fn lookup_expert<'a>(
    state: &'a ExpertState,
    model_id: &str,
    expert_id: &str,
) -> Result<&'a TrackedExpert, ExpertResidencyError> {
    state
        .models
        .get(model_id)
        .ok_or_else(|| ExpertResidencyError::UnknownModel(model_id.to_string()))?
        .experts
        .get(expert_id)
        .ok_or_else(|| unknown_expert(model_id, expert_id))
}

fn summary_from_entry(
    model_id: &str,
    expert_id: &str,
    expert: &TrackedExpert,
    now_millis: u64,
    half_life_millis: u64,
) -> ExpertSummary {
    ExpertSummary {
        id: expert_id.to_string(),
        model_id: model_id.to_string(),
        tier: expert.tier,
        bytes: expert.bytes,
        hotness: decayed_hotness(expert, now_millis, half_life_millis),
        pinned: expert.pinned,
        last_used_unix_ms: expert.last_used_unix_millis,
    }
}

fn decayed_hotness(expert: &TrackedExpert, now_millis: u64, half_life_millis: u64) -> f64 {
    let elapsed = now_millis.saturating_sub(expert.hotness_updated_millis);
    if elapsed == 0 || expert.hotness == 0.0 {
        return expert.hotness;
    }
    let exponent = -(elapsed as f64 / half_life_millis as f64);
    let value = expert.hotness * 2.0_f64.powf(exponent);
    if value.is_finite() {
        value.clamp(0.0, MAX_HOTNESS)
    } else {
        0.0
    }
}

fn in_cooldown(expert: &TrackedExpert, now_millis: u64, cooldown_millis: u64) -> bool {
    cooldown_millis > 0
        && expert
            .last_transition_millis
            .is_some_and(|last| now_millis.saturating_sub(last) < cooldown_millis)
}

fn opposite_memory_tier(target: ExpertTier) -> Result<ExpertTier, ExpertResidencyError> {
    match target {
        ExpertTier::Vram => Ok(ExpertTier::Ram),
        ExpertTier::Ram => Ok(ExpertTier::Vram),
        ExpertTier::External => Err(ExpertResidencyError::InvalidTierTransition {
            from: ExpertTier::External,
            to: ExpertTier::External,
        }),
    }
}

fn validate_memory_transition(
    from: ExpertTier,
    to: ExpertTier,
) -> Result<(), ExpertResidencyError> {
    if matches!(
        (from, to),
        (ExpertTier::Ram, ExpertTier::Vram) | (ExpertTier::Vram, ExpertTier::Ram)
    ) {
        Ok(())
    } else {
        Err(ExpertResidencyError::InvalidTierTransition { from, to })
    }
}

fn validate_action_limit(limit: usize, maximum: usize) -> Result<(), ExpertResidencyError> {
    if limit == 0 || limit > maximum {
        Err(ExpertResidencyError::InvalidLimit {
            label: "action",
            maximum,
        })
    } else {
        Ok(())
    }
}

fn validate_id_batch(
    model_id: &str,
    ids: &[String],
    maximum: usize,
) -> Result<(), ExpertResidencyError> {
    validate_action_limit(ids.len(), maximum)?;
    let mut unique = BTreeSet::new();
    for expert_id in ids {
        validate_expert_id(expert_id)?;
        if !unique.insert(expert_id) {
            return Err(ExpertResidencyError::DuplicateExpert {
                model_id: model_id.to_string(),
                expert_id: expert_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_model_id(model_id: &str) -> Result<(), ExpertResidencyError> {
    if model_id.trim().is_empty() || model_id.len() > 256 || model_id.chars().any(char::is_control)
    {
        Err(ExpertResidencyError::InvalidModelId)
    } else {
        Ok(())
    }
}

fn validate_expert_id(expert_id: &str) -> Result<(), ExpertResidencyError> {
    if valid_opaque_id(expert_id) {
        Ok(())
    } else {
        Err(ExpertResidencyError::InvalidExpertId)
    }
}

fn unknown_expert(model_id: &str, expert_id: &str) -> ExpertResidencyError {
    ExpertResidencyError::UnknownExpert {
        model_id: model_id.to_string(),
        expert_id: expert_id.to_string(),
    }
}

fn action_in_flight(key: &ExpertKey) -> ExpertResidencyError {
    ExpertResidencyError::ActionInFlight {
        model_id: key.model_id.clone(),
        expert_id: key.expert_id.clone(),
    }
}

fn stale_action(key: &ExpertKey) -> ExpertResidencyError {
    ExpertResidencyError::StaleAction {
        model_id: key.model_id.clone(),
        expert_id: key.expert_id.clone(),
    }
}

fn hash_filter(model_id: Option<&str>, tier: Option<ExpertTier>) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(CURSOR_DOMAIN);
    hash.update(b"filter\0");
    match model_id {
        Some(model_id) => {
            hash.update(b"model\0");
            hash.update((model_id.len() as u64).to_be_bytes());
            hash.update(model_id.as_bytes());
        }
        None => hash.update(b"all-models\0"),
    }
    hash.update([match tier {
        None => 0,
        Some(ExpertTier::Vram) => 1,
        Some(ExpertTier::Ram) => 2,
        Some(ExpertTier::External) => 3,
    }]);
    hash.finalize().into()
}

fn hash_key(model_id: &str, expert_id: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(CURSOR_DOMAIN);
    hash.update(b"key\0");
    hash.update((model_id.len() as u64).to_be_bytes());
    hash.update(model_id.as_bytes());
    hash.update((expert_id.len() as u64).to_be_bytes());
    hash.update(expert_id.as_bytes());
    hash.finalize().into()
}

fn encode_cursor(filter_hash: &[u8; 32], key_hash: &[u8; 32]) -> String {
    let mut bytes = [0_u8; CURSOR_BYTES];
    bytes[..32].copy_from_slice(filter_hash);
    bytes[32..].copy_from_slice(key_hash);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn decode_cursor(
    cursor: &str,
    expected_filter_hash: &[u8; 32],
) -> Result<[u8; 32], ExpertResidencyError> {
    if cursor.len() > 128 {
        return Err(ExpertResidencyError::InvalidCursor);
    }
    let bytes = URL_SAFE_NO_PAD
        .decode(cursor)
        .map_err(|_| ExpertResidencyError::InvalidCursor)?;
    if bytes.len() != CURSOR_BYTES || &bytes[..32] != expected_filter_hash {
        return Err(ExpertResidencyError::InvalidCursor);
    }
    let mut key_hash = [0_u8; 32];
    key_hash.copy_from_slice(&bytes[32..]);
    Ok(key_hash)
}

pub(crate) fn validate_expert_action(
    request: &ExpertActionRequest,
    max_ids: usize,
) -> Result<(), ProtocolError> {
    if request.model_id.trim().is_empty()
        || request.model_id.len() > 256
        || request.model_id.chars().any(char::is_control)
        || request.model_id.contains("..")
    {
        return Err(invalid(
            "model_id must be a safe identifier between 1 and 256 characters",
        ));
    }
    if request.expert_ids.is_empty() {
        return Err(invalid("expert_ids must not be empty"));
    }
    if request.expert_ids.len() > max_ids {
        return Err(invalid(format!(
            "expert_ids exceeds the maximum of {max_ids}"
        )));
    }
    let mut unique = BTreeSet::new();
    for id in &request.expert_ids {
        if !valid_opaque_id(id) {
            return Err(invalid("expert_ids contains an invalid identifier"));
        }
        if !unique.insert(id) {
            return Err(invalid("expert_ids contains a duplicate identifier"));
        }
    }
    match (request.action, request.target_tier) {
        (ExpertAction::Prefetch, Some(ExpertTier::Vram | ExpertTier::Ram)) => {}
        (ExpertAction::Prefetch, Some(ExpertTier::External)) => {
            return Err(invalid("prefetch target_tier must be either vram or ram"));
        }
        (ExpertAction::Prefetch, None) => {
            return Err(invalid("prefetch requires target_tier"));
        }
        (ExpertAction::Pin | ExpertAction::Unpin | ExpertAction::Evict, None) => {}
        (ExpertAction::Pin | ExpertAction::Unpin | ExpertAction::Evict, Some(_)) => {
            return Err(invalid(
                "target_tier is only valid for prefetch expert actions",
            ));
        }
    }
    Ok(())
}

pub(crate) fn valid_opaque_id(id: &str) -> bool {
    (1..=128).contains(&id.len())
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        && !id.contains("..")
}

fn invalid(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::InvalidRequest, message)
}

fn lock_state(mutex: &Mutex<ExpertState>) -> MutexGuard<'_, ExpertState> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::werk_protocol::{ExpertAction, ExpertActionRequest};
    use std::sync::{
        Barrier,
        atomic::{AtomicU64, Ordering},
    };
    use std::thread;

    #[derive(Debug)]
    struct TestClock {
        monotonic: AtomicU64,
        unix_base: u64,
    }

    impl TestClock {
        fn new() -> Self {
            Self {
                monotonic: AtomicU64::new(0),
                unix_base: 1_700_000_000_000,
            }
        }

        fn advance(&self, millis: u64) {
            self.monotonic.fetch_add(millis, Ordering::SeqCst);
        }
    }

    impl ExpertClock for TestClock {
        fn monotonic_millis(&self) -> u64 {
            self.monotonic.load(Ordering::SeqCst)
        }

        fn unix_millis(&self) -> u64 {
            self.unix_base
                .saturating_add(self.monotonic.load(Ordering::SeqCst))
        }
    }

    fn manager() -> (ExpertResidencyManager, Arc<TestClock>) {
        let clock = Arc::new(TestClock::new());
        let config = ExpertResidencyConfig {
            hotness_half_life_millis: 1_000,
            promote_hotness: 2.0,
            demote_hotness: 0.5,
            transition_cooldown_millis: 500,
            ..ExpertResidencyConfig::default()
        };
        (
            ExpertResidencyManager::new(config, clock.clone()).unwrap(),
            clock,
        )
    }

    fn observe(
        manager: &ExpertResidencyManager,
        model_id: &str,
        expert_id: &str,
        tier: ExpertTier,
        bytes: Option<u64>,
    ) {
        manager
            .observe(ExpertObservation {
                model_id: model_id.to_string(),
                expert_id: expert_id.to_string(),
                tier,
                bytes,
            })
            .unwrap();
    }

    fn request(ids: Vec<&str>) -> ExpertActionRequest {
        ExpertActionRequest {
            model_id: "model".to_string(),
            expert_ids: ids.into_iter().map(str::to_string).collect(),
            action: ExpertAction::Prefetch,
            target_tier: Some(ExpertTier::Ram),
            dry_run: true,
            allow_experimental: false,
        }
    }

    #[test]
    fn expert_actions_require_explicit_unique_safe_ids() {
        assert!(validate_expert_action(&request(Vec::new()), 8).is_err());
        assert!(validate_expert_action(&request(vec!["same", "same"]), 8).is_err());
        assert!(validate_expert_action(&request(vec!["../outside"]), 8).is_err());
        assert!(validate_expert_action(&request(vec!["expert-1"]), 8).is_ok());

        let mut unsafe_model = request(vec!["expert-1"]);
        unsafe_model.model_id = "../outside".to_string();
        assert!(validate_expert_action(&unsafe_model, 8).is_err());
        unsafe_model.model_id = "model\nother".to_string();
        assert!(validate_expert_action(&unsafe_model, 8).is_err());
    }

    #[test]
    fn expert_action_target_tier_matches_the_action() {
        let mut action = request(vec!["expert-1"]);
        action.target_tier = None;
        assert!(validate_expert_action(&action, 8).is_err());

        action.target_tier = Some(ExpertTier::External);
        assert!(validate_expert_action(&action, 8).is_err());

        action.action = ExpertAction::Pin;
        action.target_tier = Some(ExpertTier::Ram);
        assert!(validate_expert_action(&action, 8).is_err());

        action.target_tier = None;
        assert!(validate_expert_action(&action, 8).is_ok());
    }

    #[test]
    fn tracking_is_bounded_per_model_and_globally() {
        let clock = Arc::new(TestClock::new());
        let config = ExpertResidencyConfig {
            max_models: 1,
            max_experts_per_model: 2,
            max_total_experts: 2,
            ..ExpertResidencyConfig::default()
        };
        let manager = ExpertResidencyManager::new(config, clock).unwrap();
        observe(&manager, "m1", "e1", ExpertTier::Ram, Some(10));
        observe(&manager, "m1", "e2", ExpertTier::Ram, Some(10));
        assert!(matches!(
            manager.observe(ExpertObservation {
                model_id: "m1".to_string(),
                expert_id: "e3".to_string(),
                tier: ExpertTier::Ram,
                bytes: Some(10),
            }),
            Err(ExpertResidencyError::TotalExpertLimitReached(2))
        ));
        assert!(matches!(
            manager.observe(ExpertObservation {
                model_id: "m2".to_string(),
                expert_id: "e1".to_string(),
                tier: ExpertTier::Ram,
                bytes: Some(10),
            }),
            Err(ExpertResidencyError::ModelLimitReached(1))
        ));
        assert_eq!(manager.tracked_counts(), (1, 2, 0));
    }

    #[test]
    fn hotness_decays_and_prefetch_uses_hysteresis() {
        let (manager, clock) = manager();
        observe(&manager, "m", "hot", ExpertTier::Ram, Some(10));
        observe(&manager, "m", "warm", ExpertTier::Ram, Some(10));
        manager
            .record_accesses(
                "m",
                &[
                    ExpertAccess {
                        expert_id: "hot".to_string(),
                        weight: 4.0,
                    },
                    ExpertAccess {
                        expert_id: "warm".to_string(),
                        weight: 1.0,
                    },
                ],
            )
            .unwrap();
        let plan = manager
            .prefetch_candidates("m", ExpertTier::Vram, 8)
            .unwrap();
        assert_eq!(plan.actions.len(), 1);
        assert_eq!(plan.actions[0].expert_id(), "hot");
        assert_eq!(plan.deferred_by_hysteresis, 1);

        clock.advance(1_000);
        let listed = manager.list(&ExpertListFilter::default()).unwrap();
        let hot = listed
            .experts
            .iter()
            .find(|expert| expert.id == "hot")
            .unwrap();
        assert!((hot.hotness - 2.0).abs() < 1e-9);
        assert_eq!(hot.last_used_unix_ms, Some(1_700_000_000_000));
    }

    #[test]
    fn pressure_is_cold_first_demotes_before_evicting_and_protects_pins() {
        let (manager, _) = manager();
        observe(&manager, "m", "a", ExpertTier::Vram, Some(100));
        observe(&manager, "m", "b", ExpertTier::Vram, Some(200));
        observe(&manager, "m", "pinned", ExpertTier::Vram, Some(500));
        manager
            .set_pinned("m", &["pinned".to_string()], true, false)
            .unwrap();
        let plan = manager
            .plan_pressure(ExpertPressureRequest {
                source_tier: ExpertTier::Vram,
                level: ExpertPressureLevel::Hard,
                relief_bytes: 250,
                ram_headroom_bytes: 100,
                max_actions: 8,
            })
            .unwrap();
        assert_eq!(plan.known_planned_relief_bytes, 300);
        assert_eq!(plan.unresolved_relief_bytes, 0);
        assert_eq!(plan.protected_by_pin, 1);
        assert!(matches!(
            &plan.actions[0],
            ExpertPolicyAction::Move {
                expert_id,
                from: ExpertTier::Vram,
                to: ExpertTier::Ram,
                ..
            } if expert_id == "a"
        ));
        assert!(matches!(
            &plan.actions[1],
            ExpertPolicyAction::Evict { expert_id, .. } if expert_id == "b"
        ));
    }

    #[test]
    fn unknown_sizes_are_never_assumed_to_fit_demotion_headroom() {
        let (manager, _) = manager();
        observe(&manager, "m", "unknown", ExpertTier::Vram, None);
        let plan = manager
            .plan_pressure(ExpertPressureRequest {
                source_tier: ExpertTier::Vram,
                level: ExpertPressureLevel::Emergency,
                relief_bytes: 1,
                ram_headroom_bytes: u64::MAX,
                max_actions: 1,
            })
            .unwrap();
        assert!(matches!(plan.actions[0], ExpertPolicyAction::Evict { .. }));
        assert_eq!(plan.unknown_size_actions, 1);
        assert_eq!(plan.known_planned_relief_bytes, 0);
        assert_eq!(plan.unresolved_relief_bytes, 1);
    }

    #[test]
    fn permit_rollback_and_commit_keep_backend_truthful_and_apply_cooldown() {
        let (manager, clock) = manager();
        observe(&manager, "m", "e", ExpertTier::Ram, Some(100));
        manager
            .record_accesses(
                "m",
                &[ExpertAccess {
                    expert_id: "e".to_string(),
                    weight: 4.0,
                }],
            )
            .unwrap();
        let action = manager
            .prefetch_candidates("m", ExpertTier::Vram, 1)
            .unwrap()
            .actions
            .remove(0);
        {
            let permit = manager.begin_action(&action).unwrap();
            assert_eq!(permit.actions().len(), 1);
            assert!(matches!(
                manager.record_access("m", "e"),
                Err(ExpertResidencyError::ActionInFlight { .. })
            ));
        }
        assert_eq!(manager.tracked_counts().2, 0);
        let outcome = manager.begin_action(&action).unwrap().commit().unwrap();
        assert_eq!(outcome.updated[0].tier, ExpertTier::Vram);
        assert!(outcome.evicted.is_empty());

        let reverse = manager
            .prefetch_candidates("m", ExpertTier::Ram, 1)
            .unwrap();
        assert!(reverse.actions.is_empty());
        assert_eq!(reverse.deferred_by_cooldown, 1);
        clock.advance(500);
        assert_eq!(
            manager
                .prefetch_candidates("m", ExpertTier::Ram, 1)
                .unwrap()
                .actions
                .len(),
            0,
            "hotness still lies above the demotion hysteresis"
        );
    }

    #[test]
    fn action_becomes_stale_after_access_changes_hotness() {
        let (manager, _) = manager();
        observe(&manager, "m", "e", ExpertTier::Ram, Some(10));
        let action = manager
            .explicit_action("m", "e", Some(ExpertTier::Vram))
            .unwrap();
        manager.record_access("m", "e").unwrap();
        assert!(matches!(
            manager.begin_action(&action),
            Err(ExpertResidencyError::StaleAction { .. })
        ));
    }

    #[test]
    fn access_batches_validate_atomically() {
        let (manager, _) = manager();
        observe(&manager, "m", "known", ExpertTier::Ram, Some(10));
        assert!(
            manager
                .record_accesses(
                    "m",
                    &[
                        ExpertAccess {
                            expert_id: "known".to_string(),
                            weight: 2.0,
                        },
                        ExpertAccess {
                            expert_id: "missing".to_string(),
                            weight: 1.0,
                        },
                    ],
                )
                .is_err()
        );
        let summary = manager
            .list(&ExpertListFilter::default())
            .unwrap()
            .experts
            .pop()
            .unwrap();
        assert_eq!(summary.hotness, 0.0);
    }

    #[test]
    fn list_pagination_is_stable_and_cursor_is_bound_to_filter() {
        let (manager, _) = manager();
        for (model, expert, tier) in [
            ("m1", "e1", ExpertTier::Ram),
            ("m1", "e2", ExpertTier::Vram),
            ("m2", "e1", ExpertTier::Ram),
            ("m2", "e2", ExpertTier::Vram),
            ("m3", "e1", ExpertTier::Ram),
        ] {
            observe(&manager, model, expert, tier, Some(10));
        }
        let mut filter = ExpertListFilter {
            limit: Some(2),
            ..ExpertListFilter::default()
        };
        let mut seen = BTreeSet::new();
        loop {
            let page = manager.list(&filter).unwrap();
            for expert in page.experts {
                assert!(seen.insert((expert.model_id, expert.id)));
            }
            let Some(cursor) = page.next_cursor else {
                break;
            };
            filter.cursor = Some(cursor);
        }
        assert_eq!(seen.len(), 5);

        let first = manager
            .list(&ExpertListFilter {
                tier: Some(ExpertTier::Ram),
                limit: Some(1),
                ..ExpertListFilter::default()
            })
            .unwrap();
        assert!(matches!(
            manager.list(&ExpertListFilter {
                tier: Some(ExpertTier::Vram),
                limit: Some(1),
                cursor: first.next_cursor,
                ..ExpertListFilter::default()
            }),
            Err(ExpertResidencyError::InvalidCursor)
        ));
    }

    #[test]
    fn concurrent_access_recording_has_no_lost_updates() {
        let (manager, _) = manager();
        observe(&manager, "m", "e", ExpertTier::Ram, Some(10));
        let threads = (0..8)
            .map(|_| {
                let manager = manager.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        manager.record_access("m", "e").unwrap();
                    }
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
        let hotness = manager.list(&ExpertListFilter::default()).unwrap().experts[0].hotness;
        assert_eq!(hotness, 800.0);
    }

    #[test]
    fn action_claim_excludes_concurrent_claimants() {
        let (manager, _) = manager();
        observe(&manager, "m", "e", ExpertTier::Ram, Some(10));
        let action = manager
            .explicit_action("m", "e", Some(ExpertTier::Vram))
            .unwrap();
        let permit = manager.begin_action(&action).unwrap();
        let barrier = Arc::new(Barrier::new(9));
        let threads = (0..8)
            .map(|_| {
                let manager = manager.clone();
                let action = action.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    matches!(
                        manager.begin_action(&action),
                        Err(ExpertResidencyError::ActionInFlight { .. })
                    )
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        assert!(threads.into_iter().all(|thread| thread.join().unwrap()));
        drop(permit);
        assert_eq!(manager.tracked_counts().2, 0);
    }
}
