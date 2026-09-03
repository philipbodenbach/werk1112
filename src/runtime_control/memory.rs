//! Backend-neutral memory accounting and pressure policy.
//!
//! This module deliberately has no dependency on a concrete inference backend.
//! Backends reserve capacity before loading or promoting an allocation, then
//! commit that reservation only after the operation succeeds. Pressure handling
//! produces a deterministic action plan; callers perform the backend operation
//! and record it here afterwards.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const BASIS_POINTS: u32 = 10_000;
const MAX_MEMORY_TIERS: usize = 65;
const MAX_MANAGED_ALLOCATIONS: usize = 65_536;
const MAX_ACTIVE_RESERVATIONS: usize = 65_536;
const MAX_ACTIONS_PER_CYCLE: usize = 4_096;
const MAX_ACTION_COOLDOWN_MILLIS: u64 = 24 * 60 * 60 * 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum MemoryTier {
    Host,
    Accelerator(u16),
}

impl fmt::Display for MemoryTier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host => formatter.write_str("host"),
            Self::Accelerator(index) => write!(formatter, "accelerator:{index}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryTopology {
    Discrete,
    Unified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PressureLevel {
    Normal,
    Soft,
    Hard,
    Emergency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PressureThresholds {
    soft_basis_points: u16,
    hard_basis_points: u16,
    emergency_basis_points: u16,
    hysteresis_basis_points: u16,
}

impl PressureThresholds {
    pub(crate) fn new(
        soft_basis_points: u16,
        hard_basis_points: u16,
        emergency_basis_points: u16,
        hysteresis_basis_points: u16,
    ) -> Result<Self, MemoryError> {
        if soft_basis_points == 0
            || soft_basis_points >= hard_basis_points
            || hard_basis_points >= emergency_basis_points
            || u32::from(emergency_basis_points) > BASIS_POINTS
        {
            return Err(MemoryError::InvalidConfiguration(
                "pressure thresholds must satisfy 0 < soft < hard < emergency <= 10000".to_string(),
            ));
        }
        if hysteresis_basis_points >= soft_basis_points
            || hysteresis_basis_points >= hard_basis_points.saturating_sub(soft_basis_points)
            || hysteresis_basis_points >= emergency_basis_points.saturating_sub(hard_basis_points)
        {
            return Err(MemoryError::InvalidConfiguration(
                "pressure hysteresis must be smaller than the soft threshold and every threshold gap"
                    .to_string(),
            ));
        }
        Ok(Self {
            soft_basis_points,
            hard_basis_points,
            emergency_basis_points,
            hysteresis_basis_points,
        })
    }

    pub(crate) const fn soft_basis_points(self) -> u16 {
        self.soft_basis_points
    }

    pub(crate) const fn hard_basis_points(self) -> u16 {
        self.hard_basis_points
    }

    pub(crate) const fn emergency_basis_points(self) -> u16 {
        self.emergency_basis_points
    }

    pub(crate) const fn hysteresis_basis_points(self) -> u16 {
        self.hysteresis_basis_points
    }

    fn classify(self, utilization_basis_points: u32) -> PressureLevel {
        if utilization_basis_points >= u32::from(self.emergency_basis_points) {
            PressureLevel::Emergency
        } else if utilization_basis_points >= u32::from(self.hard_basis_points) {
            PressureLevel::Hard
        } else if utilization_basis_points >= u32::from(self.soft_basis_points) {
            PressureLevel::Soft
        } else {
            PressureLevel::Normal
        }
    }

    fn classify_with_hysteresis(
        self,
        previous: PressureLevel,
        utilization_basis_points: u32,
    ) -> PressureLevel {
        let candidate = self.classify(utilization_basis_points);
        if candidate >= previous {
            return candidate;
        }
        for (level, entry) in [
            (PressureLevel::Emergency, self.emergency_basis_points),
            (PressureLevel::Hard, self.hard_basis_points),
            (PressureLevel::Soft, self.soft_basis_points),
        ] {
            if previous >= level
                && utilization_basis_points
                    >= u32::from(entry.saturating_sub(self.hysteresis_basis_points))
            {
                return level;
            }
        }
        candidate
    }

    fn action_target_basis_points(self, pressure: PressureLevel) -> Option<u32> {
        match pressure {
            PressureLevel::Normal | PressureLevel::Soft => None,
            PressureLevel::Hard => Some(u32::from(
                self.soft_basis_points
                    .saturating_sub(self.hysteresis_basis_points),
            )),
            PressureLevel::Emergency => Some(u32::from(
                self.hard_basis_points
                    .saturating_sub(self.hysteresis_basis_points),
            )),
        }
    }
}

impl Default for PressureThresholds {
    fn default() -> Self {
        Self {
            soft_basis_points: 7_500,
            hard_basis_points: 8_500,
            emergency_basis_points: 9_500,
            hysteresis_basis_points: 500,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TierBudget {
    pub tier: MemoryTier,
    pub bytes: u64,
}

impl TierBudget {
    pub(crate) fn new(tier: MemoryTier, bytes: u64) -> Result<Self, MemoryError> {
        if bytes == 0 {
            return Err(MemoryError::InvalidConfiguration(format!(
                "memory budget for {tier} must be greater than zero"
            )));
        }
        Ok(Self { tier, bytes })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryManagerConfig {
    budgets: Vec<TierBudget>,
    topology: MemoryTopology,
    thresholds: PressureThresholds,
    action_cooldown_millis: u64,
    max_allocations: usize,
    max_reservations: usize,
    max_actions_per_cycle: usize,
}

impl MemoryManagerConfig {
    pub(crate) fn new(
        budgets: Vec<TierBudget>,
        topology: MemoryTopology,
        thresholds: PressureThresholds,
        action_cooldown_millis: u64,
        max_allocations: usize,
        max_reservations: usize,
        max_actions_per_cycle: usize,
    ) -> Result<Self, MemoryError> {
        if budgets.is_empty() || budgets.len() > MAX_MEMORY_TIERS {
            return Err(MemoryError::InvalidConfiguration(format!(
                "memory manager requires between 1 and {MAX_MEMORY_TIERS} tier budgets"
            )));
        }
        let mut tiers = BTreeSet::new();
        for budget in &budgets {
            if budget.bytes == 0 {
                return Err(MemoryError::InvalidConfiguration(format!(
                    "memory budget for {} must be greater than zero",
                    budget.tier
                )));
            }
            if !tiers.insert(budget.tier) {
                return Err(MemoryError::InvalidConfiguration(format!(
                    "memory budget for {} is duplicated",
                    budget.tier
                )));
            }
        }
        if !tiers.contains(&MemoryTier::Host) {
            return Err(MemoryError::InvalidConfiguration(
                "memory manager requires a host tier budget".to_string(),
            ));
        }
        if action_cooldown_millis > MAX_ACTION_COOLDOWN_MILLIS {
            return Err(MemoryError::InvalidConfiguration(format!(
                "action cooldown must not exceed {MAX_ACTION_COOLDOWN_MILLIS} milliseconds"
            )));
        }
        validate_limit(
            "managed allocation",
            max_allocations,
            MAX_MANAGED_ALLOCATIONS,
        )?;
        validate_limit(
            "active reservation",
            max_reservations,
            MAX_ACTIVE_RESERVATIONS,
        )?;
        validate_limit(
            "pressure action",
            max_actions_per_cycle,
            MAX_ACTIONS_PER_CYCLE,
        )?;
        Ok(Self {
            budgets,
            topology,
            thresholds,
            action_cooldown_millis,
            max_allocations,
            max_reservations,
            max_actions_per_cycle,
        })
    }

    pub(crate) fn thresholds(&self) -> PressureThresholds {
        self.thresholds
    }

    pub(crate) fn topology(&self) -> MemoryTopology {
        self.topology
    }
}

fn validate_limit(label: &str, value: usize, maximum: usize) -> Result<(), MemoryError> {
    if value == 0 || value > maximum {
        return Err(MemoryError::InvalidConfiguration(format!(
            "{label} limit must be between 1 and {maximum}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryObservation {
    pub tier: MemoryTier,
    pub total_bytes: u64,
    pub available_bytes: u64,
}

pub(crate) trait MemoryTelemetry: Send + Sync {
    fn observe(&self) -> Result<Vec<MemoryObservation>, MemoryError>;
}

pub(crate) trait MemoryClock: Send + Sync {
    fn monotonic_millis(&self) -> u64;
    fn unix_millis(&self) -> u64;
}

#[derive(Debug)]
pub(crate) struct SystemMemoryClock {
    started: Instant,
    unix_at_start_millis: u64,
}

impl SystemMemoryClock {
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

impl Default for SystemMemoryClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryClock for SystemMemoryClock {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct AllocationId(u64);

impl AllocationId {
    pub(crate) fn new(value: u64) -> Result<Self, MemoryError> {
        if value == 0 {
            return Err(MemoryError::InvalidAllocation(
                "allocation id must be nonzero".to_string(),
            ));
        }
        Ok(Self(value))
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TierMemorySnapshot {
    pub tier: MemoryTier,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub observed_used_bytes: u64,
    pub managed_used_bytes: u64,
    pub reserved_bytes: u64,
    pub budget_bytes: u64,
    pub managed_available_bytes: u64,
    pub active_reservations: usize,
    pub managed_allocations: usize,
    pub pressure: PressureLevel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemorySnapshot {
    pub observed_at_unix_millis: u64,
    pub topology: MemoryTopology,
    pub tiers: Vec<TierMemorySnapshot>,
    pub last_action_unix_millis: Option<u64>,
    pub completed_demotions: u64,
    pub completed_evictions: u64,
    pub failed_releases: u64,
    pub orphaned_release_bytes: u64,
    pub actions_in_flight: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PressureAction {
    Demote {
        allocation_id: AllocationId,
        from: MemoryTier,
        to: MemoryTier,
        bytes: u64,
    },
    Evict {
        allocation_id: AllocationId,
        tier: MemoryTier,
        bytes: u64,
    },
}

impl PressureAction {
    pub(crate) fn allocation_id(&self) -> AllocationId {
        match self {
            Self::Demote { allocation_id, .. } | Self::Evict { allocation_id, .. } => {
                *allocation_id
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PressureActionPlan {
    pub tier: MemoryTier,
    pub pressure: PressureLevel,
    pub actions: Vec<PressureAction>,
    pub planned_relief_bytes: u64,
    pub unresolved_pressure_bytes: u64,
    pub deferred_by_cooldown: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MemoryError {
    InvalidConfiguration(String),
    InvalidTelemetry(String),
    InvalidAllocation(String),
    UnknownTier(MemoryTier),
    UnknownAllocation(AllocationId),
    DuplicateAllocation(AllocationId),
    AllocationLimitReached(usize),
    ReservationLimitReached(usize),
    ReservationDenied {
        tier: MemoryTier,
        requested_bytes: u64,
        reason: String,
    },
    InvalidReservation(String),
    ActionInFlight(AllocationId),
    StaleAction(AllocationId),
    PinnedAllocation(AllocationId),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(formatter, "invalid memory configuration: {message}")
            }
            Self::InvalidTelemetry(message) => {
                write!(formatter, "invalid memory telemetry: {message}")
            }
            Self::InvalidAllocation(message) => write!(formatter, "invalid allocation: {message}"),
            Self::UnknownTier(tier) => write!(formatter, "memory tier {tier} is not configured"),
            Self::UnknownAllocation(id) => write!(formatter, "allocation {} is unknown", id.get()),
            Self::DuplicateAllocation(id) => {
                write!(formatter, "allocation {} already exists", id.get())
            }
            Self::AllocationLimitReached(limit) => {
                write!(formatter, "managed allocation limit {limit} was reached")
            }
            Self::ReservationLimitReached(limit) => {
                write!(formatter, "active reservation limit {limit} was reached")
            }
            Self::ReservationDenied {
                tier,
                requested_bytes,
                reason,
            } => write!(
                formatter,
                "reservation of {requested_bytes} bytes on {tier} was denied: {reason}"
            ),
            Self::InvalidReservation(message) => {
                write!(formatter, "invalid reservation: {message}")
            }
            Self::ActionInFlight(id) => write!(
                formatter,
                "allocation {} already has a pressure action in flight",
                id.get()
            ),
            Self::StaleAction(id) => write!(
                formatter,
                "pressure action for allocation {} is stale",
                id.get()
            ),
            Self::PinnedAllocation(id) => {
                write!(
                    formatter,
                    "allocation {} is pinned and cannot be changed by pressure policy",
                    id.get()
                )
            }
        }
    }
}

impl Error for MemoryError {}

#[derive(Debug, Clone)]
struct ManagedAllocation {
    id: AllocationId,
    tier: MemoryTier,
    bytes: u64,
    pinned: bool,
    demotion_target: Option<MemoryTier>,
    last_used_millis: u64,
}

#[derive(Debug, Clone)]
struct TierAccount {
    budget_bytes: u64,
    managed_used_bytes: u64,
    reserved_bytes: u64,
    active_reservations: usize,
    pressure: PressureLevel,
    last_action_millis: Option<u64>,
    observation: Option<MemoryObservation>,
}

#[derive(Debug)]
struct ManagerState {
    tiers: BTreeMap<MemoryTier, TierAccount>,
    allocations: BTreeMap<AllocationId, ManagedAllocation>,
    pending_loads: BTreeSet<AllocationId>,
    pending_promotions: BTreeSet<AllocationId>,
    pending_actions: BTreeSet<AllocationId>,
    active_reservations: usize,
    last_action_unix_millis: Option<u64>,
    completed_demotions: u64,
    completed_evictions: u64,
    failed_releases: u64,
    orphaned_release_bytes: u64,
}

struct ManagerInner {
    config: MemoryManagerConfig,
    telemetry: Arc<dyn MemoryTelemetry>,
    clock: Arc<dyn MemoryClock>,
    state: Mutex<ManagerState>,
}

#[derive(Clone)]
pub(crate) struct MemoryManager {
    inner: Arc<ManagerInner>,
}

impl MemoryManager {
    pub(crate) fn new(
        config: MemoryManagerConfig,
        telemetry: Arc<dyn MemoryTelemetry>,
        clock: Arc<dyn MemoryClock>,
    ) -> Self {
        let tiers = config
            .budgets
            .iter()
            .map(|budget| {
                (
                    budget.tier,
                    TierAccount {
                        budget_bytes: budget.bytes,
                        managed_used_bytes: 0,
                        reserved_bytes: 0,
                        active_reservations: 0,
                        pressure: PressureLevel::Normal,
                        last_action_millis: None,
                        observation: None,
                    },
                )
            })
            .collect();
        Self {
            inner: Arc::new(ManagerInner {
                config,
                telemetry,
                clock,
                state: Mutex::new(ManagerState {
                    tiers,
                    allocations: BTreeMap::new(),
                    pending_loads: BTreeSet::new(),
                    pending_promotions: BTreeSet::new(),
                    pending_actions: BTreeSet::new(),
                    active_reservations: 0,
                    last_action_unix_millis: None,
                    completed_demotions: 0,
                    completed_evictions: 0,
                    failed_releases: 0,
                    orphaned_release_bytes: 0,
                }),
            }),
        }
    }

    pub(crate) fn refresh(&self) -> Result<MemorySnapshot, MemoryError> {
        let observations = self.sample_telemetry()?;
        let mut state = lock_state(&self.inner.state);
        refresh_accounts(
            &mut state,
            &observations,
            self.inner.config.topology,
            self.inner.config.thresholds,
        );
        Ok(snapshot_from_state(
            &state,
            self.inner.clock.unix_millis(),
            self.inner.config.topology,
        ))
    }

    /// Returns manager-owned accounting even when physical telemetry cannot
    /// currently be sampled. Callers must treat capacity, availability and
    /// pressure as unknown; reservations, allocations and cleanup counters
    /// remain authoritative.
    pub(crate) fn accounting_snapshot(&self) -> MemorySnapshot {
        let state = lock_state(&self.inner.state);
        let tiers = state
            .tiers
            .iter()
            .map(|(tier, account)| TierMemorySnapshot {
                tier: *tier,
                total_bytes: account
                    .observation
                    .map(|observation| observation.total_bytes)
                    .unwrap_or(0),
                available_bytes: account
                    .observation
                    .map(|observation| observation.available_bytes)
                    .unwrap_or(0),
                observed_used_bytes: account
                    .observation
                    .map(|observation| {
                        observation
                            .total_bytes
                            .saturating_sub(observation.available_bytes)
                    })
                    .unwrap_or(0),
                managed_used_bytes: account.managed_used_bytes,
                reserved_bytes: account.reserved_bytes,
                budget_bytes: account.budget_bytes,
                managed_available_bytes: account
                    .budget_bytes
                    .saturating_sub(account.managed_used_bytes)
                    .saturating_sub(account.reserved_bytes),
                active_reservations: account.active_reservations,
                managed_allocations: state
                    .allocations
                    .values()
                    .filter(|allocation| allocation.tier == *tier)
                    .count(),
                pressure: account.pressure,
            })
            .collect();
        MemorySnapshot {
            observed_at_unix_millis: self.inner.clock.unix_millis(),
            topology: self.inner.config.topology,
            tiers,
            last_action_unix_millis: state.last_action_unix_millis,
            completed_demotions: state.completed_demotions,
            completed_evictions: state.completed_evictions,
            failed_releases: state.failed_releases,
            orphaned_release_bytes: state.orphaned_release_bytes,
            actions_in_flight: state.pending_actions.len(),
        }
    }

    pub(crate) fn reserve_load(
        &self,
        allocation_id: AllocationId,
        tier: MemoryTier,
        bytes: u64,
        pinned: bool,
        demotion_target: Option<MemoryTier>,
    ) -> Result<MemoryReservation, MemoryError> {
        let observations = self.sample_telemetry()?;
        let mut state = lock_state(&self.inner.state);
        refresh_accounts(
            &mut state,
            &observations,
            self.inner.config.topology,
            self.inner.config.thresholds,
        );
        if state.allocations.contains_key(&allocation_id)
            || state.pending_loads.contains(&allocation_id)
        {
            return Err(MemoryError::DuplicateAllocation(allocation_id));
        }
        if state
            .allocations
            .len()
            .saturating_add(state.pending_loads.len())
            >= self.inner.config.max_allocations
        {
            return Err(MemoryError::AllocationLimitReached(
                self.inner.config.max_allocations,
            ));
        }
        validate_demotion_target(&state, tier, demotion_target)?;
        let reservation = reserve_locked(
            &self.inner,
            &mut state,
            tier,
            bytes,
            ReservationBinding::Load {
                allocation_id,
                pinned,
                demotion_target,
            },
        )?;
        state.pending_loads.insert(allocation_id);
        Ok(reservation)
    }

    pub(crate) fn reserve_promotion(
        &self,
        allocation_id: AllocationId,
        target: MemoryTier,
    ) -> Result<MemoryReservation, MemoryError> {
        let observations = self.sample_telemetry()?;
        let mut state = lock_state(&self.inner.state);
        refresh_accounts(
            &mut state,
            &observations,
            self.inner.config.topology,
            self.inner.config.thresholds,
        );
        let allocation = state
            .allocations
            .get(&allocation_id)
            .ok_or(MemoryError::UnknownAllocation(allocation_id))?;
        if allocation.tier == target {
            return Err(MemoryError::InvalidReservation(format!(
                "allocation {} is already on {target}",
                allocation_id.get()
            )));
        }
        if state.pending_promotions.contains(&allocation_id) {
            return Err(MemoryError::InvalidReservation(format!(
                "allocation {} already has a pending promotion",
                allocation_id.get()
            )));
        }
        if state.pending_actions.contains(&allocation_id) {
            return Err(MemoryError::ActionInFlight(allocation_id));
        }
        let source = allocation.tier;
        let bytes = allocation.bytes;
        if source != MemoryTier::Host || !matches!(target, MemoryTier::Accelerator(_)) {
            return Err(MemoryError::InvalidReservation(
                "promotion must move an allocation from host memory to an accelerator".to_string(),
            ));
        }
        let reservation = reserve_locked(
            &self.inner,
            &mut state,
            target,
            bytes,
            ReservationBinding::Promotion {
                allocation_id,
                source,
            },
        )?;
        state.pending_promotions.insert(allocation_id);
        Ok(reservation)
    }

    pub(crate) fn touch(&self, allocation_id: AllocationId) -> Result<(), MemoryError> {
        let now = self.inner.clock.monotonic_millis();
        let mut state = lock_state(&self.inner.state);
        if state.pending_actions.contains(&allocation_id) {
            return Err(MemoryError::ActionInFlight(allocation_id));
        }
        let allocation = state
            .allocations
            .get_mut(&allocation_id)
            .ok_or(MemoryError::UnknownAllocation(allocation_id))?;
        allocation.last_used_millis = now;
        Ok(())
    }

    pub(crate) fn set_pinned(
        &self,
        allocation_id: AllocationId,
        pinned: bool,
    ) -> Result<(), MemoryError> {
        let mut state = lock_state(&self.inner.state);
        if state.pending_actions.contains(&allocation_id) {
            return Err(MemoryError::ActionInFlight(allocation_id));
        }
        let allocation = state
            .allocations
            .get_mut(&allocation_id)
            .ok_or(MemoryError::UnknownAllocation(allocation_id))?;
        allocation.pinned = pinned;
        Ok(())
    }

    pub(crate) fn remove_allocation(&self, allocation_id: AllocationId) -> Result<(), MemoryError> {
        let mut state = lock_state(&self.inner.state);
        if state.pending_actions.contains(&allocation_id)
            || state.pending_promotions.contains(&allocation_id)
        {
            return Err(MemoryError::ActionInFlight(allocation_id));
        }
        let allocation = state
            .allocations
            .remove(&allocation_id)
            .ok_or(MemoryError::UnknownAllocation(allocation_id))?;
        if let Some(account) = state.tiers.get_mut(&allocation.tier) {
            account.managed_used_bytes =
                account.managed_used_bytes.saturating_sub(allocation.bytes);
        }
        state.pending_promotions.remove(&allocation_id);
        Ok(())
    }

    /// Keeps a conservatively accounted allocation visible when its backend
    /// refused cleanup. The orphan is pinned because Werk no longer has a
    /// safe state handle with which to retry a policy movement.
    pub(crate) fn record_failed_release(
        &self,
        allocation_id: AllocationId,
    ) -> Result<(), MemoryError> {
        let mut state = lock_state(&self.inner.state);
        if state.pending_actions.contains(&allocation_id) {
            return Err(MemoryError::ActionInFlight(allocation_id));
        }
        let allocation = state
            .allocations
            .get_mut(&allocation_id)
            .ok_or(MemoryError::UnknownAllocation(allocation_id))?;
        allocation.pinned = true;
        state.failed_releases = state.failed_releases.saturating_add(1);
        Ok(())
    }

    /// Conservatively accounts a backend allocation whose cleanup failed
    /// before a normal managed-allocation record could be committed. With no
    /// trustworthy size, charge the whole configured tier budget.
    pub(crate) fn record_orphaned_release(
        &self,
        tier: MemoryTier,
        bytes: Option<u64>,
    ) -> Result<(), MemoryError> {
        let mut state = lock_state(&self.inner.state);
        let account = state
            .tiers
            .get_mut(&tier)
            .ok_or(MemoryError::UnknownTier(tier))?;
        let accounted = bytes
            .filter(|bytes| *bytes > 0)
            .unwrap_or(account.budget_bytes);
        account.managed_used_bytes = account.managed_used_bytes.saturating_add(accounted);
        state.failed_releases = state.failed_releases.saturating_add(1);
        state.orphaned_release_bytes = state.orphaned_release_bytes.saturating_add(accounted);
        Ok(())
    }

    pub(crate) fn plan_pressure_actions(
        &self,
        tier: MemoryTier,
    ) -> Result<PressureActionPlan, MemoryError> {
        self.plan_pressure_actions_for(tier, 0, None)
    }

    pub(crate) fn plan_pressure_actions_among(
        &self,
        tier: MemoryTier,
        eligible: &BTreeSet<AllocationId>,
    ) -> Result<PressureActionPlan, MemoryError> {
        self.plan_pressure_actions_for(tier, 0, Some(eligible))
    }

    /// Plans enough relief for a rejected admission, even when current
    /// pressure is still normal or soft. The requested bytes are included in
    /// the projected utilization so the retry cannot depend on crossing a
    /// pressure threshold first.
    pub(crate) fn plan_for_reservation(
        &self,
        tier: MemoryTier,
        requested_bytes: u64,
    ) -> Result<PressureActionPlan, MemoryError> {
        if requested_bytes == 0 {
            return Err(MemoryError::InvalidReservation(
                "reservation size must be greater than zero".to_string(),
            ));
        }
        self.plan_pressure_actions_for(tier, requested_bytes, None)
    }

    pub(crate) fn plan_for_reservation_among(
        &self,
        tier: MemoryTier,
        requested_bytes: u64,
        eligible: &BTreeSet<AllocationId>,
    ) -> Result<PressureActionPlan, MemoryError> {
        if requested_bytes == 0 {
            return Err(MemoryError::InvalidReservation(
                "reservation size must be greater than zero".to_string(),
            ));
        }
        self.plan_pressure_actions_for(tier, requested_bytes, Some(eligible))
    }

    fn plan_pressure_actions_for(
        &self,
        tier: MemoryTier,
        requested_bytes: u64,
        eligible: Option<&BTreeSet<AllocationId>>,
    ) -> Result<PressureActionPlan, MemoryError> {
        let observations = self.sample_telemetry()?;
        let now = self.inner.clock.monotonic_millis();
        let mut state = lock_state(&self.inner.state);
        refresh_accounts(
            &mut state,
            &observations,
            self.inner.config.topology,
            self.inner.config.thresholds,
        );
        let account = state
            .tiers
            .get(&tier)
            .ok_or(MemoryError::UnknownTier(tier))?;
        let pressure = account.pressure;
        let pressure_target = self
            .inner
            .config
            .thresholds
            .action_target_basis_points(pressure);
        let target_basis_points = match (pressure_target, requested_bytes) {
            (Some(target), _) => target,
            (None, 0) => return Ok(empty_action_plan(tier, pressure, false)),
            // Reservations are rejected at the hard threshold. Staying one
            // basis point below it makes the plan agree exactly with the
            // admission predicate without inventing extra policy headroom.
            (None, _) => u32::from(
                self.inner
                    .config
                    .thresholds
                    .hard_basis_points
                    .saturating_sub(1),
            ),
        };
        let (effective_budget, effective_used) =
            effective_accounting(&state, tier, self.inner.config.topology);
        let target_bytes = ((u128::from(effective_budget) * u128::from(target_basis_points))
            / u128::from(BASIS_POINTS))
        .try_into()
        .unwrap_or(u64::MAX);
        let required_relief = effective_used
            .saturating_add(requested_bytes)
            .saturating_sub(target_bytes);
        if required_relief == 0 {
            return Ok(empty_action_plan(tier, pressure, false));
        }
        let last_action_millis = match self.inner.config.topology {
            MemoryTopology::Discrete => account.last_action_millis,
            MemoryTopology::Unified => state
                .tiers
                .values()
                .filter_map(|account| account.last_action_millis)
                .max(),
        };
        if last_action_millis
            .is_some_and(|last| now.saturating_sub(last) < self.inner.config.action_cooldown_millis)
        {
            return Ok(PressureActionPlan {
                tier,
                pressure,
                actions: Vec::new(),
                planned_relief_bytes: 0,
                unresolved_pressure_bytes: required_relief,
                deferred_by_cooldown: true,
            });
        }

        let mut candidates = state
            .allocations
            .values()
            .filter(|allocation| {
                (self.inner.config.topology == MemoryTopology::Unified || allocation.tier == tier)
                    && !allocation.pinned
                    && eligible.is_none_or(|eligible| eligible.contains(&allocation.id))
                    && !state.pending_promotions.contains(&allocation.id)
                    && !state.pending_actions.contains(&allocation.id)
            })
            .cloned()
            .collect::<Vec<_>>();
        candidates.sort_by_key(|allocation| (allocation.last_used_millis, allocation.id));

        let mut actions = Vec::new();
        let mut planned_ids = BTreeSet::new();
        let mut planned_incoming = BTreeMap::<MemoryTier, u64>::new();
        let mut relieved = 0_u64;

        if self.inner.config.topology == MemoryTopology::Discrete {
            for allocation in candidates
                .iter()
                .filter(|allocation| allocation.demotion_target.is_some())
            {
                if relieved >= required_relief
                    || actions.len() >= self.inner.config.max_actions_per_cycle
                {
                    break;
                }
                let target = allocation
                    .demotion_target
                    .expect("filtered allocations have a demotion target");
                let already_planned = planned_incoming.get(&target).copied().unwrap_or(0);
                if !can_accept_bytes(
                    &state,
                    target,
                    allocation.bytes,
                    already_planned,
                    self.inner.config.topology,
                    self.inner.config.thresholds,
                ) {
                    continue;
                }
                planned_incoming.insert(target, already_planned.saturating_add(allocation.bytes));
                actions.push(PressureAction::Demote {
                    allocation_id: allocation.id,
                    from: tier,
                    to: target,
                    bytes: allocation.bytes,
                });
                planned_ids.insert(allocation.id);
                relieved = relieved.saturating_add(allocation.bytes);
            }
        }

        for allocation in &candidates {
            if relieved >= required_relief
                || actions.len() >= self.inner.config.max_actions_per_cycle
            {
                break;
            }
            if planned_ids.contains(&allocation.id) {
                continue;
            }
            actions.push(PressureAction::Evict {
                allocation_id: allocation.id,
                tier: allocation.tier,
                bytes: allocation.bytes,
            });
            planned_ids.insert(allocation.id);
            relieved = relieved.saturating_add(allocation.bytes);
        }

        Ok(PressureActionPlan {
            tier,
            pressure,
            actions,
            planned_relief_bytes: relieved,
            unresolved_pressure_bytes: required_relief.saturating_sub(relieved),
            deferred_by_cooldown: false,
        })
    }

    /// Claims an action before backend work begins. Dropping the permit rolls
    /// the claim back; committing it records a successfully completed action.
    pub(crate) fn begin_pressure_action(
        &self,
        action: &PressureAction,
    ) -> Result<PressureActionPermit, MemoryError> {
        let mut state = lock_state(&self.inner.state);
        let allocation_id = action.allocation_id();
        if state.pending_actions.contains(&allocation_id)
            || state.pending_promotions.contains(&allocation_id)
        {
            return Err(MemoryError::ActionInFlight(allocation_id));
        }
        let allocation = state
            .allocations
            .get(&allocation_id)
            .cloned()
            .ok_or(MemoryError::StaleAction(allocation_id))?;
        if allocation.pinned {
            return Err(MemoryError::PinnedAllocation(allocation_id));
        }
        match action {
            PressureAction::Demote {
                from, to, bytes, ..
            } => {
                if allocation.tier != *from
                    || allocation.bytes != *bytes
                    || allocation.demotion_target != Some(*to)
                    || self.inner.config.topology != MemoryTopology::Discrete
                {
                    return Err(MemoryError::StaleAction(allocation_id));
                }
                if state.active_reservations >= self.inner.config.max_reservations {
                    return Err(MemoryError::ReservationLimitReached(
                        self.inner.config.max_reservations,
                    ));
                }
                if !can_accept_bytes(
                    &state,
                    *to,
                    *bytes,
                    0,
                    self.inner.config.topology,
                    self.inner.config.thresholds,
                ) {
                    return Err(MemoryError::ReservationDenied {
                        tier: *to,
                        requested_bytes: *bytes,
                        reason: "demotion target capacity changed after planning".to_string(),
                    });
                }
                let target = state
                    .tiers
                    .get_mut(to)
                    .ok_or(MemoryError::UnknownTier(*to))?;
                target.reserved_bytes = target.reserved_bytes.saturating_add(*bytes);
                target.active_reservations = target.active_reservations.saturating_add(1);
                state.active_reservations = state.active_reservations.saturating_add(1);
            }
            PressureAction::Evict { tier, bytes, .. } => {
                if allocation.tier != *tier || allocation.bytes != *bytes {
                    return Err(MemoryError::StaleAction(allocation_id));
                }
            }
        }
        state.pending_actions.insert(allocation_id);
        Ok(PressureActionPermit {
            inner: Arc::clone(&self.inner),
            action: action.clone(),
            active: true,
        })
    }

    fn sample_telemetry(&self) -> Result<BTreeMap<MemoryTier, MemoryObservation>, MemoryError> {
        let samples = self.inner.telemetry.observe()?;
        if samples.is_empty() || samples.len() > MAX_MEMORY_TIERS {
            return Err(MemoryError::InvalidTelemetry(format!(
                "telemetry must contain between 1 and {MAX_MEMORY_TIERS} tiers"
            )));
        }
        let mut observations = BTreeMap::new();
        for sample in samples {
            if sample.total_bytes == 0 {
                return Err(MemoryError::InvalidTelemetry(format!(
                    "{} reported zero total bytes",
                    sample.tier
                )));
            }
            if sample.available_bytes > sample.total_bytes {
                return Err(MemoryError::InvalidTelemetry(format!(
                    "{} reported more available bytes than total bytes",
                    sample.tier
                )));
            }
            if observations.insert(sample.tier, sample).is_some() {
                return Err(MemoryError::InvalidTelemetry(format!(
                    "{} was reported more than once",
                    sample.tier
                )));
            }
        }
        let state = lock_state(&self.inner.state);
        for tier in state.tiers.keys() {
            if !observations.contains_key(tier) {
                return Err(MemoryError::InvalidTelemetry(format!(
                    "configured tier {tier} was not reported"
                )));
            }
        }
        Ok(observations)
    }
}

fn lock_state(mutex: &Mutex<ManagerState>) -> MutexGuard<'_, ManagerState> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn empty_action_plan(
    tier: MemoryTier,
    pressure: PressureLevel,
    deferred_by_cooldown: bool,
) -> PressureActionPlan {
    PressureActionPlan {
        tier,
        pressure,
        actions: Vec::new(),
        planned_relief_bytes: 0,
        unresolved_pressure_bytes: 0,
        deferred_by_cooldown,
    }
}

fn refresh_accounts(
    state: &mut ManagerState,
    observations: &BTreeMap<MemoryTier, MemoryObservation>,
    topology: MemoryTopology,
    thresholds: PressureThresholds,
) {
    if topology == MemoryTopology::Unified {
        for (tier, account) in &mut state.tiers {
            account.observation = Some(
                observations
                    .get(tier)
                    .copied()
                    .expect("telemetry was validated before updating accounts"),
            );
        }
        let previous = state
            .tiers
            .values()
            .map(|account| account.pressure)
            .max()
            .unwrap_or(PressureLevel::Normal);
        let (budget, used) = effective_accounting(state, MemoryTier::Host, topology);
        let pressure =
            thresholds.classify_with_hysteresis(previous, utilization_basis_points(used, budget));
        for account in state.tiers.values_mut() {
            account.pressure = pressure;
        }
        return;
    }
    for (tier, account) in &mut state.tiers {
        let observation = observations
            .get(tier)
            .copied()
            .expect("telemetry was validated before updating accounts");
        let used = observation
            .total_bytes
            .saturating_sub(observation.available_bytes)
            .max(account.managed_used_bytes)
            .saturating_add(account.reserved_bytes);
        let budget = account.budget_bytes.min(observation.total_bytes);
        let utilization = utilization_basis_points(used, budget);
        account.pressure = thresholds.classify_with_hysteresis(account.pressure, utilization);
        account.observation = Some(observation);
    }
}

fn effective_accounting(
    state: &ManagerState,
    tier: MemoryTier,
    topology: MemoryTopology,
) -> (u64, u64) {
    if topology == MemoryTopology::Discrete {
        let account = state
            .tiers
            .get(&tier)
            .expect("effective accounting requires a configured tier");
        let observation = account
            .observation
            .expect("effective accounting requires refreshed telemetry");
        return (
            account.budget_bytes.min(observation.total_bytes),
            observation
                .total_bytes
                .saturating_sub(observation.available_bytes)
                .max(account.managed_used_bytes)
                .saturating_add(account.reserved_bytes),
        );
    }

    let budget = state
        .tiers
        .values()
        .map(|account| {
            account.budget_bytes.min(
                account
                    .observation
                    .expect("effective accounting requires refreshed telemetry")
                    .total_bytes,
            )
        })
        .min()
        .unwrap_or(0);
    let observed_used = state
        .tiers
        .values()
        .map(|account| {
            let observation = account
                .observation
                .expect("effective accounting requires refreshed telemetry");
            observation
                .total_bytes
                .saturating_sub(observation.available_bytes)
        })
        .max()
        .unwrap_or(0);
    let managed_used = state.tiers.values().fold(0_u64, |total, account| {
        total.saturating_add(account.managed_used_bytes)
    });
    let reserved = state.tiers.values().fold(0_u64, |total, account| {
        total.saturating_add(account.reserved_bytes)
    });
    (
        budget,
        observed_used.max(managed_used).saturating_add(reserved),
    )
}

fn utilization_basis_points(used_bytes: u64, budget_bytes: u64) -> u32 {
    if budget_bytes == 0 {
        return u32::MAX;
    }
    ((u128::from(used_bytes) * u128::from(BASIS_POINTS)) / u128::from(budget_bytes))
        .try_into()
        .unwrap_or(u32::MAX)
}

fn snapshot_from_state(
    state: &ManagerState,
    observed_at: u64,
    topology: MemoryTopology,
) -> MemorySnapshot {
    let unified_managed_available = (topology == MemoryTopology::Unified).then(|| {
        let budget = state
            .tiers
            .values()
            .map(|account| account.budget_bytes)
            .min()
            .unwrap_or(0);
        let managed = state.tiers.values().fold(0_u64, |total, account| {
            total.saturating_add(account.managed_used_bytes)
        });
        let reserved = state.tiers.values().fold(0_u64, |total, account| {
            total.saturating_add(account.reserved_bytes)
        });
        budget.saturating_sub(managed).saturating_sub(reserved)
    });
    let tiers = state
        .tiers
        .iter()
        .map(|(tier, account)| {
            let observation = account
                .observation
                .expect("snapshot is created only after valid telemetry refresh");
            TierMemorySnapshot {
                tier: *tier,
                total_bytes: observation.total_bytes,
                available_bytes: observation.available_bytes,
                observed_used_bytes: observation
                    .total_bytes
                    .saturating_sub(observation.available_bytes),
                managed_used_bytes: account.managed_used_bytes,
                reserved_bytes: account.reserved_bytes,
                budget_bytes: account.budget_bytes,
                managed_available_bytes: unified_managed_available.unwrap_or_else(|| {
                    account
                        .budget_bytes
                        .saturating_sub(account.managed_used_bytes)
                        .saturating_sub(account.reserved_bytes)
                }),
                active_reservations: account.active_reservations,
                managed_allocations: state
                    .allocations
                    .values()
                    .filter(|allocation| allocation.tier == *tier)
                    .count(),
                pressure: account.pressure,
            }
        })
        .collect();
    MemorySnapshot {
        observed_at_unix_millis: observed_at,
        topology,
        tiers,
        last_action_unix_millis: state.last_action_unix_millis,
        completed_demotions: state.completed_demotions,
        completed_evictions: state.completed_evictions,
        failed_releases: state.failed_releases,
        orphaned_release_bytes: state.orphaned_release_bytes,
        actions_in_flight: state.pending_actions.len(),
    }
}

fn can_accept_bytes(
    state: &ManagerState,
    tier: MemoryTier,
    bytes: u64,
    additional_planned_bytes: u64,
    topology: MemoryTopology,
    thresholds: PressureThresholds,
) -> bool {
    let Some(account) = state.tiers.get(&tier) else {
        return false;
    };
    let Some(observation) = account.observation else {
        return false;
    };
    let incoming = additional_planned_bytes.saturating_add(bytes);
    if account
        .managed_used_bytes
        .saturating_add(account.reserved_bytes)
        .saturating_add(incoming)
        > account.budget_bytes
        || account.reserved_bytes.saturating_add(incoming) > observation.available_bytes
    {
        return false;
    }
    let projected = observation
        .total_bytes
        .saturating_sub(observation.available_bytes)
        .max(account.managed_used_bytes)
        .saturating_add(account.reserved_bytes)
        .saturating_add(incoming);
    let effective_budget = account.budget_bytes.min(observation.total_bytes);
    if thresholds.classify(utilization_basis_points(projected, effective_budget))
        >= PressureLevel::Hard
    {
        return false;
    }
    if topology == MemoryTopology::Unified {
        let (shared_budget, shared_used) = effective_accounting(state, tier, topology);
        let shared_incoming = additional_planned_bytes.saturating_add(bytes);
        let shared_available = state
            .tiers
            .values()
            .map(|account| {
                account
                    .observation
                    .expect("capacity checks require refreshed telemetry")
                    .available_bytes
            })
            .min()
            .unwrap_or(0);
        let shared_reserved = state.tiers.values().fold(0_u64, |total, account| {
            total.saturating_add(account.reserved_bytes)
        });
        if shared_reserved.saturating_add(shared_incoming) > shared_available
            || thresholds.classify(utilization_basis_points(
                shared_used.saturating_add(shared_incoming),
                shared_budget,
            )) >= PressureLevel::Hard
        {
            return false;
        }
    }
    true
}

#[derive(Debug, Clone, Copy)]
enum ReservationBinding {
    Load {
        allocation_id: AllocationId,
        pinned: bool,
        demotion_target: Option<MemoryTier>,
    },
    Promotion {
        allocation_id: AllocationId,
        source: MemoryTier,
    },
}

pub(crate) struct MemoryReservation {
    inner: Arc<ManagerInner>,
    tier: MemoryTier,
    bytes: u64,
    binding: ReservationBinding,
    active: bool,
}

impl fmt::Debug for MemoryReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryReservation")
            .field("tier", &self.tier)
            .field("bytes", &self.bytes)
            .field("active", &self.active)
            .finish_non_exhaustive()
    }
}

impl MemoryReservation {
    pub(crate) const fn tier(&self) -> MemoryTier {
        self.tier
    }

    pub(crate) const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn commit_load(mut self) -> Result<(), MemoryError> {
        let ReservationBinding::Load {
            allocation_id,
            pinned,
            demotion_target,
        } = self.binding
        else {
            return Err(MemoryError::InvalidReservation(
                "a promotion reservation cannot commit a new load".to_string(),
            ));
        };
        let now = self.inner.clock.monotonic_millis();
        let mut state = lock_state(&self.inner.state);
        if !state.pending_loads.contains(&allocation_id)
            || state.allocations.contains_key(&allocation_id)
        {
            return Err(MemoryError::InvalidReservation(format!(
                "allocation {} is no longer reserved for loading",
                allocation_id.get()
            )));
        }
        release_reservation_locked(&mut state, self.tier, self.bytes, self.binding);
        let account = state
            .tiers
            .get_mut(&self.tier)
            .expect("reservation tier must remain configured");
        account.managed_used_bytes = account.managed_used_bytes.saturating_add(self.bytes);
        state.allocations.insert(
            allocation_id,
            ManagedAllocation {
                id: allocation_id,
                tier: self.tier,
                bytes: self.bytes,
                pinned,
                demotion_target,
                last_used_millis: now,
            },
        );
        self.active = false;
        Ok(())
    }

    pub(crate) fn commit_promotion(mut self) -> Result<(), MemoryError> {
        let ReservationBinding::Promotion {
            allocation_id,
            source,
        } = self.binding
        else {
            return Err(MemoryError::InvalidReservation(
                "a load reservation cannot commit a promotion".to_string(),
            ));
        };
        let now = self.inner.clock.monotonic_millis();
        let mut state = lock_state(&self.inner.state);
        let allocation = state
            .allocations
            .get(&allocation_id)
            .ok_or(MemoryError::UnknownAllocation(allocation_id))?;
        if allocation.tier != source || allocation.bytes != self.bytes {
            return Err(MemoryError::InvalidReservation(format!(
                "allocation {} changed while its promotion was reserved",
                allocation_id.get()
            )));
        }
        release_reservation_locked(&mut state, self.tier, self.bytes, self.binding);
        let source_account = state
            .tiers
            .get_mut(&source)
            .expect("allocation source tier must remain configured");
        source_account.managed_used_bytes =
            source_account.managed_used_bytes.saturating_sub(self.bytes);
        let target_account = state
            .tiers
            .get_mut(&self.tier)
            .expect("reservation target tier must remain configured");
        target_account.managed_used_bytes =
            target_account.managed_used_bytes.saturating_add(self.bytes);
        let allocation = state
            .allocations
            .get_mut(&allocation_id)
            .expect("allocation existence was checked above");
        allocation.tier = self.tier;
        allocation.demotion_target = Some(source);
        allocation.last_used_millis = now;
        self.active = false;
        Ok(())
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut state = lock_state(&self.inner.state);
        release_reservation_locked(&mut state, self.tier, self.bytes, self.binding);
        self.active = false;
    }
}

pub(crate) struct PressureActionPermit {
    inner: Arc<ManagerInner>,
    action: PressureAction,
    active: bool,
}

impl fmt::Debug for PressureActionPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PressureActionPermit")
            .field("action", &self.action)
            .field("active", &self.active)
            .finish()
    }
}

impl PressureActionPermit {
    pub(crate) fn action(&self) -> &PressureAction {
        &self.action
    }

    /// Records the accounting transition after the backend action succeeds.
    pub(crate) fn commit(mut self) -> Result<(), MemoryError> {
        let allocation_id = self.action.allocation_id();
        let completed_at_millis = self.inner.clock.monotonic_millis();
        let completed_at_unix_millis = self.inner.clock.unix_millis();
        let mut state = lock_state(&self.inner.state);
        if !state.pending_actions.contains(&allocation_id) {
            return Err(MemoryError::StaleAction(allocation_id));
        }
        let allocation = state
            .allocations
            .get(&allocation_id)
            .cloned()
            .ok_or(MemoryError::StaleAction(allocation_id))?;
        match &self.action {
            PressureAction::Demote {
                from, to, bytes, ..
            } => {
                if allocation.tier != *from
                    || allocation.bytes != *bytes
                    || allocation.demotion_target != Some(*to)
                {
                    return Err(MemoryError::StaleAction(allocation_id));
                }
                release_pressure_action_locked(&mut state, &self.action);
                let source = state
                    .tiers
                    .get_mut(from)
                    .expect("allocation tier must be configured");
                source.managed_used_bytes = source.managed_used_bytes.saturating_sub(*bytes);
                let target = state
                    .tiers
                    .get_mut(to)
                    .expect("permit target tier must remain configured");
                target.managed_used_bytes = target.managed_used_bytes.saturating_add(*bytes);
                let allocation = state
                    .allocations
                    .get_mut(&allocation_id)
                    .expect("allocation existence was checked above");
                allocation.tier = *to;
                allocation.demotion_target = None;
                state.completed_demotions = state.completed_demotions.saturating_add(1);
            }
            PressureAction::Evict { tier, bytes, .. } => {
                if allocation.tier != *tier || allocation.bytes != *bytes {
                    return Err(MemoryError::StaleAction(allocation_id));
                }
                release_pressure_action_locked(&mut state, &self.action);
                state.allocations.remove(&allocation_id);
                let account = state
                    .tiers
                    .get_mut(tier)
                    .expect("allocation tier must be configured");
                account.managed_used_bytes = account.managed_used_bytes.saturating_sub(*bytes);
                state.pending_promotions.remove(&allocation_id);
                state.completed_evictions = state.completed_evictions.saturating_add(1);
            }
        }
        match self.inner.config.topology {
            MemoryTopology::Discrete => {
                let action_tier = match &self.action {
                    PressureAction::Demote { from, .. } => *from,
                    PressureAction::Evict { tier, .. } => *tier,
                };
                let account = state
                    .tiers
                    .get_mut(&action_tier)
                    .expect("action tier must remain configured");
                account.last_action_millis = Some(completed_at_millis);
            }
            MemoryTopology::Unified => {
                for account in state.tiers.values_mut() {
                    account.last_action_millis = Some(completed_at_millis);
                }
            }
        }
        state.last_action_unix_millis = Some(completed_at_unix_millis);
        self.active = false;
        Ok(())
    }
}

impl Drop for PressureActionPermit {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut state = lock_state(&self.inner.state);
        release_pressure_action_locked(&mut state, &self.action);
        self.active = false;
    }
}

fn release_pressure_action_locked(state: &mut ManagerState, action: &PressureAction) {
    state.pending_actions.remove(&action.allocation_id());
    if let PressureAction::Demote { to, bytes, .. } = action {
        if let Some(account) = state.tiers.get_mut(to) {
            account.reserved_bytes = account.reserved_bytes.saturating_sub(*bytes);
            account.active_reservations = account.active_reservations.saturating_sub(1);
        }
        state.active_reservations = state.active_reservations.saturating_sub(1);
    }
}

fn reserve_locked(
    inner: &Arc<ManagerInner>,
    state: &mut ManagerState,
    tier: MemoryTier,
    bytes: u64,
    binding: ReservationBinding,
) -> Result<MemoryReservation, MemoryError> {
    if bytes == 0 {
        return Err(MemoryError::InvalidReservation(
            "reservation size must be greater than zero".to_string(),
        ));
    }
    if state.active_reservations >= inner.config.max_reservations {
        return Err(MemoryError::ReservationLimitReached(
            inner.config.max_reservations,
        ));
    }
    let account = state
        .tiers
        .get(&tier)
        .ok_or(MemoryError::UnknownTier(tier))?;
    let observation = account
        .observation
        .expect("telemetry refresh precedes every reservation");
    if account
        .managed_used_bytes
        .saturating_add(account.reserved_bytes)
        .saturating_add(bytes)
        > account.budget_bytes
    {
        return Err(reservation_denied(
            tier,
            bytes,
            "managed budget would be exceeded",
        ));
    }
    if account.reserved_bytes.saturating_add(bytes) > observation.available_bytes {
        return Err(reservation_denied(
            tier,
            bytes,
            "observed available memory is insufficient",
        ));
    }
    let projected = observation
        .total_bytes
        .saturating_sub(observation.available_bytes)
        .max(account.managed_used_bytes)
        .saturating_add(account.reserved_bytes)
        .saturating_add(bytes);
    let effective_budget = account.budget_bytes.min(observation.total_bytes);
    if inner
        .config
        .thresholds
        .classify(utilization_basis_points(projected, effective_budget))
        >= PressureLevel::Hard
    {
        return Err(reservation_denied(
            tier,
            bytes,
            "projected pressure would be hard or emergency",
        ));
    }
    if inner.config.topology == MemoryTopology::Unified {
        let (shared_budget, shared_used) = effective_accounting(state, tier, inner.config.topology);
        let shared_available = state
            .tiers
            .values()
            .map(|account| {
                account
                    .observation
                    .expect("reservation requires refreshed telemetry")
                    .available_bytes
            })
            .min()
            .unwrap_or(0);
        let shared_reserved = state.tiers.values().fold(0_u64, |total, account| {
            total.saturating_add(account.reserved_bytes)
        });
        if shared_reserved.saturating_add(bytes) > shared_available
            || inner.config.thresholds.classify(utilization_basis_points(
                shared_used.saturating_add(bytes),
                shared_budget,
            )) >= PressureLevel::Hard
        {
            return Err(reservation_denied(
                tier,
                bytes,
                "unified-memory pressure would be hard or emergency",
            ));
        }
    }
    let account = state
        .tiers
        .get_mut(&tier)
        .expect("tier existence was checked above");
    account.reserved_bytes = account.reserved_bytes.saturating_add(bytes);
    account.active_reservations = account.active_reservations.saturating_add(1);
    state.active_reservations = state.active_reservations.saturating_add(1);
    Ok(MemoryReservation {
        inner: Arc::clone(inner),
        tier,
        bytes,
        binding,
        active: true,
    })
}

fn reservation_denied(tier: MemoryTier, bytes: u64, reason: &str) -> MemoryError {
    MemoryError::ReservationDenied {
        tier,
        requested_bytes: bytes,
        reason: reason.to_string(),
    }
}

fn release_reservation_locked(
    state: &mut ManagerState,
    tier: MemoryTier,
    bytes: u64,
    binding: ReservationBinding,
) {
    if let Some(account) = state.tiers.get_mut(&tier) {
        account.reserved_bytes = account.reserved_bytes.saturating_sub(bytes);
        account.active_reservations = account.active_reservations.saturating_sub(1);
    }
    state.active_reservations = state.active_reservations.saturating_sub(1);
    match binding {
        ReservationBinding::Load { allocation_id, .. } => {
            state.pending_loads.remove(&allocation_id);
        }
        ReservationBinding::Promotion { allocation_id, .. } => {
            state.pending_promotions.remove(&allocation_id);
        }
    }
}

fn validate_demotion_target(
    state: &ManagerState,
    source: MemoryTier,
    target: Option<MemoryTier>,
) -> Result<(), MemoryError> {
    if let Some(target) = target
        && (!matches!(
            (source, target),
            (MemoryTier::Accelerator(_), MemoryTier::Host)
        ) || !state.tiers.contains_key(&target))
    {
        return Err(MemoryError::InvalidAllocation(
            "demotion must move an accelerator allocation to configured host memory".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Default)]
    struct FakeClock {
        monotonic: AtomicU64,
        unix: AtomicU64,
    }

    impl FakeClock {
        fn set(&self, millis: u64) {
            self.monotonic.store(millis, Ordering::SeqCst);
            self.unix
                .store(1_700_000_000_000 + millis, Ordering::SeqCst);
        }

        fn advance(&self, millis: u64) {
            self.monotonic.fetch_add(millis, Ordering::SeqCst);
            self.unix.fetch_add(millis, Ordering::SeqCst);
        }
    }

    impl MemoryClock for FakeClock {
        fn monotonic_millis(&self) -> u64 {
            self.monotonic.load(Ordering::SeqCst)
        }

        fn unix_millis(&self) -> u64 {
            self.unix.load(Ordering::SeqCst)
        }
    }

    struct FakeTelemetry {
        observations: Mutex<Vec<MemoryObservation>>,
    }

    impl FakeTelemetry {
        fn new(observations: Vec<MemoryObservation>) -> Self {
            Self {
                observations: Mutex::new(observations),
            }
        }

        fn set(&self, tier: MemoryTier, total_bytes: u64, available_bytes: u64) {
            let mut observations = self.observations.lock().unwrap();
            let observation = observations
                .iter_mut()
                .find(|observation| observation.tier == tier)
                .unwrap();
            observation.total_bytes = total_bytes;
            observation.available_bytes = available_bytes;
        }
    }

    impl MemoryTelemetry for FakeTelemetry {
        fn observe(&self) -> Result<Vec<MemoryObservation>, MemoryError> {
            Ok(self.observations.lock().unwrap().clone())
        }
    }

    fn observation(tier: MemoryTier, total: u64, available: u64) -> MemoryObservation {
        MemoryObservation {
            tier,
            total_bytes: total,
            available_bytes: available,
        }
    }

    fn manager_with_limits(
        telemetry: Arc<FakeTelemetry>,
        clock: Arc<FakeClock>,
        thresholds: PressureThresholds,
        cooldown: u64,
        max_allocations: usize,
        max_reservations: usize,
        max_actions: usize,
    ) -> MemoryManager {
        let config = MemoryManagerConfig::new(
            vec![
                TierBudget::new(MemoryTier::Host, 1_000).unwrap(),
                TierBudget::new(MemoryTier::Accelerator(0), 1_000).unwrap(),
            ],
            MemoryTopology::Discrete,
            thresholds,
            cooldown,
            max_allocations,
            max_reservations,
            max_actions,
        )
        .unwrap();
        MemoryManager::new(config, telemetry, clock)
    }

    fn test_manager(telemetry: Arc<FakeTelemetry>, clock: Arc<FakeClock>) -> MemoryManager {
        manager_with_limits(
            telemetry,
            clock,
            PressureThresholds::default(),
            1_000,
            64,
            64,
            64,
        )
    }

    fn tier(snapshot: &MemorySnapshot, wanted: MemoryTier) -> &TierMemorySnapshot {
        snapshot
            .tiers
            .iter()
            .find(|snapshot| snapshot.tier == wanted)
            .unwrap()
    }

    #[test]
    fn thresholds_and_configuration_are_strictly_validated() {
        let valid = PressureThresholds::new(7_500, 8_500, 9_500, 500).unwrap();
        assert_eq!(valid.soft_basis_points(), 7_500);
        assert_eq!(valid.hard_basis_points(), 8_500);
        assert_eq!(valid.emergency_basis_points(), 9_500);
        assert_eq!(valid.hysteresis_basis_points(), 500);
        for values in [
            (0, 8_500, 9_500, 500),
            (8_500, 8_500, 9_500, 500),
            (7_500, 9_500, 9_000, 500),
            (7_500, 8_500, 10_001, 500),
            (7_500, 8_500, 9_500, 7_500),
            (7_500, 8_000, 8_500, 500),
        ] {
            assert!(PressureThresholds::new(values.0, values.1, values.2, values.3).is_err());
        }

        let thresholds = PressureThresholds::default();
        assert!(
            MemoryManagerConfig::new(vec![], MemoryTopology::Discrete, thresholds, 0, 1, 1, 1,)
                .is_err()
        );
        assert!(
            MemoryManagerConfig::new(
                vec![TierBudget::new(MemoryTier::Accelerator(0), 100).unwrap()],
                MemoryTopology::Discrete,
                thresholds,
                0,
                1,
                1,
                1,
            )
            .is_err()
        );
        assert!(
            MemoryManagerConfig::new(
                vec![
                    TierBudget::new(MemoryTier::Host, 100).unwrap(),
                    TierBudget::new(MemoryTier::Host, 200).unwrap(),
                ],
                MemoryTopology::Discrete,
                thresholds,
                0,
                1,
                1,
                1,
            )
            .is_err()
        );
        assert!(TierBudget::new(MemoryTier::Host, 0).is_err());
        let config = MemoryManagerConfig::new(
            vec![TierBudget::new(MemoryTier::Host, 100).unwrap()],
            MemoryTopology::Unified,
            thresholds,
            0,
            1,
            1,
            1,
        )
        .unwrap();
        assert_eq!(config.topology(), MemoryTopology::Unified);
        assert_eq!(config.thresholds(), thresholds);
        assert!(
            MemoryManagerConfig::new(
                vec![TierBudget::new(MemoryTier::Host, 100).unwrap()],
                MemoryTopology::Discrete,
                thresholds,
                MAX_ACTION_COOLDOWN_MILLIS + 1,
                1,
                1,
                1,
            )
            .is_err()
        );

        let clock = SystemMemoryClock::new();
        assert!(clock.unix_millis() >= clock.monotonic_millis());
    }

    #[test]
    fn pressure_transitions_use_hysteresis() {
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 1_000),
            observation(MemoryTier::Accelerator(0), 1_000, 240),
        ]));
        let clock = Arc::new(FakeClock::default());
        let manager = test_manager(Arc::clone(&telemetry), clock);

        assert_eq!(
            tier(&manager.refresh().unwrap(), MemoryTier::Accelerator(0)).pressure,
            PressureLevel::Soft
        );
        telemetry.set(MemoryTier::Accelerator(0), 1_000, 280);
        assert_eq!(
            tier(&manager.refresh().unwrap(), MemoryTier::Accelerator(0)).pressure,
            PressureLevel::Soft
        );
        telemetry.set(MemoryTier::Accelerator(0), 1_000, 301);
        assert_eq!(
            tier(&manager.refresh().unwrap(), MemoryTier::Accelerator(0)).pressure,
            PressureLevel::Normal
        );

        telemetry.set(MemoryTier::Accelerator(0), 1_000, 140);
        assert_eq!(
            tier(&manager.refresh().unwrap(), MemoryTier::Accelerator(0)).pressure,
            PressureLevel::Hard
        );
        telemetry.set(MemoryTier::Accelerator(0), 1_000, 190);
        assert_eq!(
            tier(&manager.refresh().unwrap(), MemoryTier::Accelerator(0)).pressure,
            PressureLevel::Hard
        );
        telemetry.set(MemoryTier::Accelerator(0), 1_000, 201);
        assert_eq!(
            tier(&manager.refresh().unwrap(), MemoryTier::Accelerator(0)).pressure,
            PressureLevel::Soft
        );
        telemetry.set(MemoryTier::Accelerator(0), 1_000, 260);
        assert_eq!(
            tier(&manager.refresh().unwrap(), MemoryTier::Accelerator(0)).pressure,
            PressureLevel::Soft
        );
        telemetry.set(MemoryTier::Accelerator(0), 1_000, 301);
        assert_eq!(
            tier(&manager.refresh().unwrap(), MemoryTier::Accelerator(0)).pressure,
            PressureLevel::Normal
        );
    }

    #[test]
    fn reservation_accounting_is_separate_and_drop_releases_capacity() {
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 900),
            observation(MemoryTier::Accelerator(0), 1_000, 900),
        ]));
        let clock = Arc::new(FakeClock::default());
        clock.set(42);
        let manager = test_manager(Arc::clone(&telemetry), clock);

        let reservation = manager
            .reserve_load(
                AllocationId::new(1).unwrap(),
                MemoryTier::Host,
                100,
                false,
                None,
            )
            .unwrap();
        assert_eq!(reservation.tier(), MemoryTier::Host);
        assert_eq!(reservation.bytes(), 100);
        let snapshot = manager.refresh().unwrap();
        let host = tier(&snapshot, MemoryTier::Host);
        assert_eq!(host.managed_used_bytes, 0);
        assert_eq!(host.reserved_bytes, 100);
        assert_eq!(host.budget_bytes, 1_000);
        assert_eq!(host.active_reservations, 1);
        assert_eq!(host.managed_available_bytes, 900);

        drop(reservation);
        let host = manager.refresh().unwrap();
        let host = tier(&host, MemoryTier::Host);
        assert_eq!(host.reserved_bytes, 0);
        assert_eq!(host.active_reservations, 0);

        manager
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
        let host = manager.refresh().unwrap();
        let host = tier(&host, MemoryTier::Host);
        assert_eq!(host.reserved_bytes, 0);
        assert_eq!(host.managed_used_bytes, 100);
        assert_eq!(host.managed_allocations, 1);
        manager
            .remove_allocation(AllocationId::new(1).unwrap())
            .unwrap();
        assert_eq!(
            tier(&manager.refresh().unwrap(), MemoryTier::Host).managed_used_bytes,
            0
        );
    }

    #[test]
    fn load_ids_and_allocation_slots_are_reserved_before_backend_work() {
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 1_000),
            observation(MemoryTier::Accelerator(0), 1_000, 1_000),
        ]));
        let clock = Arc::new(FakeClock::default());
        let manager = manager_with_limits(
            Arc::clone(&telemetry),
            clock,
            PressureThresholds::default(),
            1_000,
            1,
            4,
            4,
        );
        let id = AllocationId::new(1).unwrap();
        let reservation = manager
            .reserve_load(id, MemoryTier::Host, 100, false, None)
            .unwrap();

        assert!(matches!(
            manager.reserve_load(id, MemoryTier::Host, 50, false, None),
            Err(MemoryError::DuplicateAllocation(duplicate)) if duplicate == id
        ));
        assert!(matches!(
            manager.reserve_load(
                AllocationId::new(2).unwrap(),
                MemoryTier::Host,
                50,
                false,
                None,
            ),
            Err(MemoryError::AllocationLimitReached(1))
        ));
        drop(reservation);
        let replacement = manager
            .reserve_load(id, MemoryTier::Host, 50, false, None)
            .unwrap();
        replacement.commit_load().unwrap();
        let snapshot = manager.refresh().unwrap();
        assert_eq!(tier(&snapshot, MemoryTier::Host).reserved_bytes, 0);
        assert_eq!(tier(&snapshot, MemoryTier::Host).active_reservations, 0);
    }

    #[test]
    fn reservations_enforce_pressure_availability_and_count_limits() {
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 900),
            observation(MemoryTier::Accelerator(0), 1_000, 900),
        ]));
        let clock = Arc::new(FakeClock::default());
        let manager = manager_with_limits(
            Arc::clone(&telemetry),
            clock,
            PressureThresholds::default(),
            0,
            10,
            1,
            10,
        );
        let first = manager
            .reserve_load(
                AllocationId::new(1).unwrap(),
                MemoryTier::Host,
                10,
                false,
                None,
            )
            .unwrap();
        assert!(matches!(
            manager.reserve_load(
                AllocationId::new(2).unwrap(),
                MemoryTier::Host,
                10,
                false,
                None,
            ),
            Err(MemoryError::ReservationLimitReached(1))
        ));
        drop(first);

        telemetry.set(MemoryTier::Host, 1_000, 200);
        assert!(matches!(
            manager.reserve_load(
                AllocationId::new(2).unwrap(),
                MemoryTier::Host,
                60,
                false,
                None,
            ),
            Err(MemoryError::ReservationDenied { .. })
        ));
        telemetry.set(MemoryTier::Host, 1_000, 20);
        assert!(matches!(
            manager.reserve_load(
                AllocationId::new(3).unwrap(),
                MemoryTier::Host,
                30,
                false,
                None,
            ),
            Err(MemoryError::ReservationDenied { .. })
        ));
        assert!(matches!(
            manager.reserve_load(
                AllocationId::new(4).unwrap(),
                MemoryTier::Host,
                0,
                false,
                None,
            ),
            Err(MemoryError::InvalidReservation(_))
        ));
    }

    #[test]
    fn promotion_is_bound_to_an_raii_reservation() {
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 900),
            observation(MemoryTier::Accelerator(0), 1_000, 900),
        ]));
        let clock = Arc::new(FakeClock::default());
        let manager = test_manager(Arc::clone(&telemetry), clock);
        let id = AllocationId::new(7).unwrap();
        manager
            .reserve_load(id, MemoryTier::Host, 100, false, None)
            .unwrap()
            .commit_load()
            .unwrap();

        let promotion = manager
            .reserve_promotion(id, MemoryTier::Accelerator(0))
            .unwrap();
        let snapshot = manager.refresh().unwrap();
        assert_eq!(tier(&snapshot, MemoryTier::Host).managed_used_bytes, 100);
        assert_eq!(
            tier(&snapshot, MemoryTier::Accelerator(0)).reserved_bytes,
            100
        );
        assert!(
            manager
                .reserve_promotion(id, MemoryTier::Accelerator(0))
                .is_err()
        );
        assert_eq!(
            manager.remove_allocation(id),
            Err(MemoryError::ActionInFlight(id))
        );
        assert!(matches!(
            manager.begin_pressure_action(&PressureAction::Evict {
                allocation_id: id,
                tier: MemoryTier::Host,
                bytes: 100,
            }),
            Err(MemoryError::ActionInFlight(blocked)) if blocked == id
        ));

        telemetry.set(MemoryTier::Host, 1_000, 40);
        let plan = manager.plan_pressure_actions(MemoryTier::Host).unwrap();
        assert!(plan.actions.is_empty());
        assert!(plan.unresolved_pressure_bytes > 0);

        promotion.commit_promotion().unwrap();
        let snapshot = manager.refresh().unwrap();
        assert_eq!(tier(&snapshot, MemoryTier::Host).managed_used_bytes, 0);
        assert_eq!(
            tier(&snapshot, MemoryTier::Accelerator(0)).managed_used_bytes,
            100
        );
        assert_eq!(
            tier(&snapshot, MemoryTier::Accelerator(0)).reserved_bytes,
            0
        );
    }

    #[test]
    fn pressure_plan_demotes_before_evicting_and_skips_pinned_allocations() {
        let thresholds = PressureThresholds::new(7_000, 8_500, 9_500, 500).unwrap();
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 900),
            observation(MemoryTier::Accelerator(0), 1_000, 1_000),
        ]));
        let clock = Arc::new(FakeClock::default());
        let manager = manager_with_limits(
            Arc::clone(&telemetry),
            Arc::clone(&clock),
            thresholds,
            1_000,
            64,
            64,
            64,
        );
        let accelerator = MemoryTier::Accelerator(0);
        clock.set(1);
        manager
            .reserve_load(
                AllocationId::new(1).unwrap(),
                accelerator,
                40,
                true,
                Some(MemoryTier::Host),
            )
            .unwrap()
            .commit_load()
            .unwrap();
        clock.set(2);
        manager
            .reserve_load(
                AllocationId::new(2).unwrap(),
                accelerator,
                80,
                false,
                Some(MemoryTier::Host),
            )
            .unwrap()
            .commit_load()
            .unwrap();
        clock.set(3);
        manager
            .reserve_load(AllocationId::new(3).unwrap(), accelerator, 100, false, None)
            .unwrap()
            .commit_load()
            .unwrap();

        telemetry.set(accelerator, 1_000, 40);
        clock.set(10);
        let plan = manager.plan_pressure_actions(accelerator).unwrap();
        assert_eq!(plan.pressure, PressureLevel::Emergency);
        assert_eq!(plan.planned_relief_bytes, 180);
        assert_eq!(plan.unresolved_pressure_bytes, 0);
        assert_eq!(
            plan.actions,
            vec![
                PressureAction::Demote {
                    allocation_id: AllocationId::new(2).unwrap(),
                    from: accelerator,
                    to: MemoryTier::Host,
                    bytes: 80,
                },
                PressureAction::Evict {
                    allocation_id: AllocationId::new(3).unwrap(),
                    tier: accelerator,
                    bytes: 100,
                },
            ]
        );
        assert!(
            !plan
                .actions
                .iter()
                .any(|action| action.allocation_id() == AllocationId::new(1).unwrap())
        );
    }

    #[test]
    fn reservation_plan_accounts_for_projected_pressure_before_it_is_hard() {
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 400),
            observation(MemoryTier::Accelerator(0), 1_000, 1_000),
        ]));
        let clock = Arc::new(FakeClock::default());
        let manager = manager_with_limits(
            Arc::clone(&telemetry),
            clock,
            PressureThresholds::default(),
            0,
            64,
            64,
            64,
        );
        let allocation = AllocationId::new(1).unwrap();
        manager
            .reserve_load(allocation, MemoryTier::Host, 200, false, None)
            .unwrap()
            .commit_load()
            .unwrap();

        assert!(matches!(
            manager.reserve_load(
                AllocationId::new(2).unwrap(),
                MemoryTier::Host,
                300,
                false,
                None,
            ),
            Err(MemoryError::ReservationDenied { .. })
        ));
        assert!(
            manager
                .plan_pressure_actions(MemoryTier::Host)
                .unwrap()
                .actions
                .is_empty()
        );

        let plan = manager.plan_for_reservation(MemoryTier::Host, 300).unwrap();
        assert_eq!(plan.pressure, PressureLevel::Normal);
        assert_eq!(
            plan.actions,
            vec![PressureAction::Evict {
                allocation_id: allocation,
                tier: MemoryTier::Host,
                bytes: 200,
            }]
        );
        assert_eq!(plan.planned_relief_bytes, 200);
        assert_eq!(plan.unresolved_pressure_bytes, 0);
    }

    #[test]
    fn unified_memory_is_accounted_once_and_does_not_plan_false_demotions() {
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 1_000),
            observation(MemoryTier::Accelerator(0), 1_000, 1_000),
        ]));
        let clock = Arc::new(FakeClock::default());
        let config = MemoryManagerConfig::new(
            vec![
                TierBudget::new(MemoryTier::Host, 1_000).unwrap(),
                TierBudget::new(MemoryTier::Accelerator(0), 1_000).unwrap(),
            ],
            MemoryTopology::Unified,
            PressureThresholds::default(),
            1_000,
            64,
            64,
            64,
        )
        .unwrap();
        let manager = MemoryManager::new(config, telemetry.clone(), clock);
        let accelerator = MemoryTier::Accelerator(0);

        manager
            .reserve_load(
                AllocationId::new(1).unwrap(),
                accelerator,
                400,
                false,
                Some(MemoryTier::Host),
            )
            .unwrap()
            .commit_load()
            .unwrap();
        manager
            .reserve_load(
                AllocationId::new(2).unwrap(),
                MemoryTier::Host,
                400,
                false,
                None,
            )
            .unwrap()
            .commit_load()
            .unwrap();

        let snapshot = manager.refresh().unwrap();
        assert_eq!(snapshot.topology, MemoryTopology::Unified);
        assert_eq!(
            tier(&snapshot, MemoryTier::Host).pressure,
            PressureLevel::Soft
        );
        assert_eq!(tier(&snapshot, accelerator).pressure, PressureLevel::Soft);
        assert_eq!(
            tier(&snapshot, MemoryTier::Host).managed_available_bytes,
            200
        );
        assert_eq!(tier(&snapshot, accelerator).managed_available_bytes, 200);
        assert!(matches!(
            manager.reserve_load(
                AllocationId::new(3).unwrap(),
                MemoryTier::Host,
                60,
                false,
                None,
            ),
            Err(MemoryError::ReservationDenied { .. })
        ));

        telemetry.set(MemoryTier::Host, 1_000, 40);
        telemetry.set(accelerator, 1_000, 40);
        let plan = manager.plan_pressure_actions(accelerator).unwrap();
        assert_eq!(plan.pressure, PressureLevel::Emergency);
        assert!(matches!(
            plan.actions.first(),
            Some(PressureAction::Evict {
                allocation_id,
                tier,
                bytes: 400,
            }) if *allocation_id == AllocationId::new(1).unwrap() && *tier == accelerator
        ));
        assert!(
            plan.actions
                .iter()
                .all(|action| !matches!(action, PressureAction::Demote { .. }))
        );
        manager
            .begin_pressure_action(&plan.actions[0])
            .unwrap()
            .commit()
            .unwrap();
        let shared_cooldown = manager.plan_pressure_actions(MemoryTier::Host).unwrap();
        assert!(shared_cooldown.actions.is_empty());
        assert!(shared_cooldown.deferred_by_cooldown);
    }

    #[test]
    fn action_cooldown_uses_the_injected_monotonic_clock() {
        let thresholds = PressureThresholds::new(7_000, 8_500, 9_500, 500).unwrap();
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 900),
            observation(MemoryTier::Accelerator(0), 1_000, 1_000),
        ]));
        let clock = Arc::new(FakeClock::default());
        let manager = manager_with_limits(
            Arc::clone(&telemetry),
            Arc::clone(&clock),
            thresholds,
            1_000,
            64,
            64,
            64,
        );
        let accelerator = MemoryTier::Accelerator(0);
        manager
            .reserve_load(AllocationId::new(1).unwrap(), accelerator, 200, false, None)
            .unwrap()
            .commit_load()
            .unwrap();
        manager
            .reserve_load(AllocationId::new(2).unwrap(), accelerator, 200, false, None)
            .unwrap()
            .commit_load()
            .unwrap();
        telemetry.set(accelerator, 1_000, 40);

        let first = manager.plan_pressure_actions(accelerator).unwrap();
        assert!(!first.actions.is_empty());
        manager
            .begin_pressure_action(&first.actions[0])
            .unwrap()
            .commit()
            .unwrap();
        let deferred = manager.plan_pressure_actions(accelerator).unwrap();
        assert!(deferred.actions.is_empty());
        assert!(deferred.deferred_by_cooldown);
        assert!(deferred.unresolved_pressure_bytes > 0);
        clock.advance(999);
        assert!(
            manager
                .plan_pressure_actions(accelerator)
                .unwrap()
                .deferred_by_cooldown
        );
        clock.advance(1);
        let after = manager.plan_pressure_actions(accelerator).unwrap();
        assert!(!after.deferred_by_cooldown);
        assert!(!after.actions.is_empty());
    }

    #[test]
    fn action_permits_reserve_demotions_and_recheck_pin_state_before_backend_work() {
        let thresholds = PressureThresholds::new(7_000, 8_500, 9_500, 500).unwrap();
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 900),
            observation(MemoryTier::Accelerator(0), 1_000, 1_000),
        ]));
        let clock = Arc::new(FakeClock::default());
        let manager = manager_with_limits(Arc::clone(&telemetry), clock, thresholds, 0, 64, 64, 64);
        let accelerator = MemoryTier::Accelerator(0);
        let demoted = AllocationId::new(1).unwrap();
        let evicted = AllocationId::new(2).unwrap();
        manager
            .reserve_load(demoted, accelerator, 80, false, Some(MemoryTier::Host))
            .unwrap()
            .commit_load()
            .unwrap();
        manager
            .reserve_load(evicted, accelerator, 100, false, None)
            .unwrap()
            .commit_load()
            .unwrap();
        telemetry.set(accelerator, 1_000, 40);
        let plan = manager.plan_pressure_actions(accelerator).unwrap();

        let demotion = manager.begin_pressure_action(&plan.actions[0]).unwrap();
        assert_eq!(demotion.action(), &plan.actions[0]);
        assert_eq!(
            tier(&manager.refresh().unwrap(), MemoryTier::Host).reserved_bytes,
            80
        );
        assert_eq!(
            manager.set_pinned(demoted, true),
            Err(MemoryError::ActionInFlight(demoted))
        );
        demotion.commit().unwrap();
        manager.set_pinned(evicted, true).unwrap();
        assert!(matches!(
            manager.begin_pressure_action(&plan.actions[1]),
            Err(MemoryError::PinnedAllocation(id)) if id == evicted
        ));
        manager.set_pinned(evicted, false).unwrap();
        let eviction = manager.begin_pressure_action(&plan.actions[1]).unwrap();
        assert!(matches!(
            manager.begin_pressure_action(&plan.actions[1]),
            Err(MemoryError::ActionInFlight(id)) if id == evicted
        ));
        drop(eviction);
        let eviction = manager.begin_pressure_action(&plan.actions[1]).unwrap();
        eviction.commit().unwrap();

        let snapshot = manager.refresh().unwrap();
        assert_eq!(tier(&snapshot, MemoryTier::Host).managed_used_bytes, 80);
        assert_eq!(tier(&snapshot, accelerator).managed_used_bytes, 0);
        assert_eq!(snapshot.completed_demotions, 1);
        assert_eq!(snapshot.completed_evictions, 1);
        assert_eq!(snapshot.actions_in_flight, 0);
        assert!(snapshot.last_action_unix_millis.is_some());
        assert!(matches!(
            manager.touch(evicted),
            Err(MemoryError::UnknownAllocation(id)) if id == evicted
        ));
    }

    #[test]
    fn telemetry_and_arithmetic_are_bounded_and_saturating() {
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, u64::MAX, 0),
            observation(MemoryTier::Accelerator(0), u64::MAX, u64::MAX),
        ]));
        let clock = Arc::new(FakeClock::default());
        let manager = test_manager(Arc::clone(&telemetry), clock);
        let snapshot = manager.refresh().unwrap();
        assert_eq!(
            tier(&snapshot, MemoryTier::Host).pressure,
            PressureLevel::Emergency
        );
        assert_eq!(
            tier(&snapshot, MemoryTier::Host).observed_used_bytes,
            u64::MAX
        );

        telemetry.set(MemoryTier::Host, 0, 0);
        assert!(matches!(
            manager.refresh(),
            Err(MemoryError::InvalidTelemetry(_))
        ));
        telemetry.set(MemoryTier::Host, 100, 101);
        assert!(matches!(
            manager.refresh(),
            Err(MemoryError::InvalidTelemetry(_))
        ));
    }

    #[test]
    fn concurrent_reservations_are_serialized_and_all_release_on_drop() {
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 1_000),
            observation(MemoryTier::Accelerator(0), 1_000, 1_000),
        ]));
        let clock = Arc::new(FakeClock::default());
        let manager = test_manager(telemetry, clock);
        let reservations = (1..=8)
            .map(|id| {
                let manager = manager.clone();
                std::thread::spawn(move || {
                    manager
                        .reserve_load(
                            AllocationId::new(id).unwrap(),
                            MemoryTier::Host,
                            10,
                            false,
                            None,
                        )
                        .unwrap()
                })
            })
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        let snapshot = manager.refresh().unwrap();
        assert_eq!(tier(&snapshot, MemoryTier::Host).reserved_bytes, 80);
        assert_eq!(tier(&snapshot, MemoryTier::Host).active_reservations, 8);
        drop(reservations);
        let snapshot = manager.refresh().unwrap();
        assert_eq!(tier(&snapshot, MemoryTier::Host).reserved_bytes, 0);
        assert_eq!(tier(&snapshot, MemoryTier::Host).active_reservations, 0);
    }

    #[test]
    fn pressure_action_count_is_bounded_deterministically() {
        let thresholds = PressureThresholds::new(7_000, 8_500, 9_500, 500).unwrap();
        let telemetry = Arc::new(FakeTelemetry::new(vec![
            observation(MemoryTier::Host, 1_000, 900),
            observation(MemoryTier::Accelerator(0), 1_000, 1_000),
        ]));
        let clock = Arc::new(FakeClock::default());
        let manager = manager_with_limits(Arc::clone(&telemetry), clock, thresholds, 0, 64, 64, 1);
        let accelerator = MemoryTier::Accelerator(0);
        for id in 1..=3 {
            manager
                .reserve_load(AllocationId::new(id).unwrap(), accelerator, 50, false, None)
                .unwrap()
                .commit_load()
                .unwrap();
        }
        telemetry.set(accelerator, 1_000, 20);
        let plan = manager.plan_pressure_actions(accelerator).unwrap();
        assert_eq!(plan.actions.len(), 1);
        assert_eq!(
            plan.actions[0].allocation_id(),
            AllocationId::new(1).unwrap()
        );
        assert!(plan.unresolved_pressure_bytes > 0);
    }
}
