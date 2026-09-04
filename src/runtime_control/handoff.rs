use crate::{
    runtime_control::backend::BackendStateLease,
    werk_protocol::{CompatibilityEnvelope, ProtocolError, ProtocolErrorCode},
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Mutex,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const TOKEN_BYTES: usize = 32;
const MAX_HANDOFFS: usize = 1024;
const MAX_HANDOFFS_PER_PRINCIPAL: usize = 128;
const DEFAULT_HANDOFF_TTL: Duration = Duration::from_secs(15 * 60);

#[derive(Clone)]
pub(crate) struct HandoffRecord {
    pub principal_id: String,
    pub model_id: String,
    pub state_id: Option<String>,
    pub state: BackendStateLease,
    pub compatibility: CompatibilityEnvelope,
    pub expires_unix_ms: u64,
}

impl fmt::Debug for HandoffRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HandoffRecord")
            .field("principal_id", &"[redacted]")
            .field("model_id", &self.model_id)
            .field("state_id", &self.state_id)
            .field("state", &"[opaque]")
            .field("expires_unix_ms", &self.expires_unix_ms)
            .finish()
    }
}

#[derive(Default)]
struct HandoffEntries {
    entries: HashMap<String, HandoffRecord>,
    reserved_tokens: HashSet<String>,
    reserved_by_principal: HashMap<String, usize>,
}

#[derive(Default)]
pub(crate) struct HandoffRegistry {
    entries: Mutex<HandoffEntries>,
}

impl fmt::Debug for HandoffRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HandoffRegistry { entries: [redacted] }")
    }
}

impl HandoffRegistry {
    pub fn reserve(&self, principal_id: &str) -> Result<HandoffReservation<'_>, ProtocolError> {
        let (token, digest) = new_token()?;
        let mut entries = self.entries.lock().map_err(|_| internal())?;
        let expired = take_expired(&mut entries.entries, now_unix_ms());
        let result = (|| {
            ensure_capacity(&entries, principal_id)?;
            if entries.entries.contains_key(&digest)
                || !entries.reserved_tokens.insert(digest.clone())
            {
                return Err(token_collision());
            }
            *entries
                .reserved_by_principal
                .entry(principal_id.to_string())
                .or_default() += 1;
            Ok(HandoffReservation {
                registry: self,
                principal_id: principal_id.to_string(),
                token: Some(token),
                digest: Some(digest),
            })
        })();
        drop(entries);
        drop(expired);
        result
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn issue(&self, record: HandoffRecord) -> Result<String, ProtocolError> {
        self.reserve(&record.principal_id)?.issue(record)
    }

    fn commit_reserved(
        &self,
        principal_id: &str,
        digest: &str,
        mut record: HandoffRecord,
    ) -> Result<(), ProtocolError> {
        if record.principal_id != principal_id {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Internal,
                "runtime handoff reservation principal does not match its record",
            ));
        }
        if record.expires_unix_ms == 0 {
            record.expires_unix_ms = now_unix_ms()
                .saturating_add(DEFAULT_HANDOFF_TTL.as_millis().min(u64::MAX as u128) as u64);
        }
        let mut entries = self.entries.lock().map_err(|_| internal())?;
        if !entries.reserved_tokens.remove(digest) {
            return Err(internal());
        }
        decrement_principal_reservation(&mut entries, principal_id)?;
        if entries.entries.contains_key(digest) {
            return Err(token_collision());
        }
        entries.entries.insert(digest.to_string(), record);
        Ok(())
    }

    /// Validates and clones a handoff without consuming it. This is used only
    /// to resolve the owning backend before capability checks; `take` remains
    /// the sole transition into decode ownership.
    pub fn inspect(&self, principal_id: &str, token: &str) -> Result<HandoffRecord, ProtocolError> {
        if token.len() > 4096 || token.len() < 32 {
            return Err(expired());
        }
        let digest = token_digest(token);
        let mut entries = self.entries.lock().map_err(|_| internal())?;
        let now = now_unix_ms();
        let expired_records = take_expired(&mut entries.entries, now);
        let result = entries
            .entries
            .get(&digest)
            .ok_or_else(expired)
            .and_then(|candidate| {
                if candidate.principal_id != principal_id || candidate.expires_unix_ms <= now {
                    Err(expired())
                } else {
                    Ok(candidate.clone())
                }
            });
        drop(entries);
        drop(expired_records);
        result
    }

    /// Handoffs are single-use: ownership is taken before backend decode.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn take(&self, principal_id: &str, token: &str) -> Result<HandoffRecord, ProtocolError> {
        if token.len() > 4096 || token.len() < 32 {
            return Err(expired());
        }
        let digest = token_digest(token);
        let mut entries = self.entries.lock().map_err(|_| internal())?;
        let now = now_unix_ms();
        let expired_records = take_expired(&mut entries.entries, now);
        let result = match entries.entries.get(&digest) {
            Some(candidate)
                if candidate.principal_id == principal_id && candidate.expires_unix_ms > now =>
            {
                entries.entries.remove(&digest).ok_or_else(internal)
            }
            _ => Err(expired()),
        };
        drop(entries);
        drop(expired_records);
        result
    }

    /// Consumes one handoff while atomically reserving capacity for an
    /// optional replacement. The replacement reservation has the same
    /// principal, so a full registry cannot race between decode and issuance.
    pub fn take_with_replacement(
        &self,
        principal_id: &str,
        token: &str,
    ) -> Result<(HandoffRecord, HandoffReservation<'_>), ProtocolError> {
        if token.len() > 4096 || token.len() < 32 {
            return Err(expired());
        }
        let (replacement_token, replacement_digest) = new_token()?;
        let digest = token_digest(token);
        let mut entries = self.entries.lock().map_err(|_| internal())?;
        let now = now_unix_ms();
        let expired_records = take_expired(&mut entries.entries, now);
        let result = (|| {
            let candidate = entries.entries.get(&digest).ok_or_else(expired)?;
            if candidate.principal_id != principal_id || candidate.expires_unix_ms <= now {
                return Err(expired());
            }
            if entries.entries.contains_key(&replacement_digest)
                || !entries.reserved_tokens.insert(replacement_digest.clone())
            {
                return Err(token_collision());
            }
            let record = entries.entries.remove(&digest).ok_or_else(internal)?;
            *entries
                .reserved_by_principal
                .entry(principal_id.to_string())
                .or_default() += 1;
            Ok((
                record,
                HandoffReservation {
                    registry: self,
                    principal_id: principal_id.to_string(),
                    token: Some(replacement_token),
                    digest: Some(replacement_digest),
                },
            ))
        })();
        drop(entries);
        drop(expired_records);
        result
    }
}

pub(crate) struct HandoffReservation<'a> {
    registry: &'a HandoffRegistry,
    principal_id: String,
    token: Option<String>,
    digest: Option<String>,
}

impl HandoffReservation<'_> {
    pub fn issue(mut self, record: HandoffRecord) -> Result<String, ProtocolError> {
        let token = self.token.as_ref().ok_or_else(internal)?.clone();
        let digest = self.digest.as_deref().ok_or_else(internal)?;
        self.registry
            .commit_reserved(&self.principal_id, digest, record)?;
        self.token = None;
        self.digest = None;
        Ok(token)
    }
}

impl Drop for HandoffReservation<'_> {
    fn drop(&mut self) {
        let Some(digest) = self.digest.take() else {
            return;
        };
        let Ok(mut entries) = self.registry.entries.lock() else {
            return;
        };
        entries.reserved_tokens.remove(&digest);
        let _ = decrement_principal_reservation(&mut entries, &self.principal_id);
    }
}

fn ensure_capacity(entries: &HandoffEntries, principal_id: &str) -> Result<(), ProtocolError> {
    if entries
        .entries
        .len()
        .saturating_add(entries.reserved_tokens.len())
        >= MAX_HANDOFFS
    {
        return Err(capacity_exhausted(
            "the runtime handoff registry is at its global capacity",
        ));
    }
    let live = entries
        .entries
        .values()
        .filter(|existing| existing.principal_id == principal_id)
        .count();
    let reserved = entries
        .reserved_by_principal
        .get(principal_id)
        .copied()
        .unwrap_or(0);
    if live.saturating_add(reserved) >= MAX_HANDOFFS_PER_PRINCIPAL {
        return Err(capacity_exhausted(
            "the principal runtime handoff limit has been reached",
        ));
    }
    Ok(())
}

fn decrement_principal_reservation(
    entries: &mut HandoffEntries,
    principal_id: &str,
) -> Result<(), ProtocolError> {
    let remove = {
        let count = entries
            .reserved_by_principal
            .get_mut(principal_id)
            .ok_or_else(internal)?;
        *count = count.checked_sub(1).ok_or_else(internal)?;
        *count == 0
    };
    if remove {
        entries.reserved_by_principal.remove(principal_id);
    }
    Ok(())
}

fn new_token() -> Result<(String, String), ProtocolError> {
    let mut random = [0u8; TOKEN_BYTES];
    getrandom::getrandom(&mut random).map_err(|_| {
        ProtocolError::new(
            ProtocolErrorCode::Unavailable,
            "secure handoff token generation is unavailable",
        )
    })?;
    let token = URL_SAFE_NO_PAD.encode(random);
    let digest = token_digest(&token);
    Ok((token, digest))
}

fn token_collision() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::Unavailable,
        "secure handoff token allocation is temporarily unavailable",
    )
    .retryable(true)
}

fn token_digest(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"werk1112-handoff-v1\0");
    hasher.update(token.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn take_expired(entries: &mut HashMap<String, HandoffRecord>, now: u64) -> Vec<HandoffRecord> {
    let expired = entries
        .iter()
        .filter_map(|(digest, record)| (record.expires_unix_ms <= now).then_some(digest.clone()))
        .collect::<Vec<_>>();
    expired
        .into_iter()
        .filter_map(|digest| entries.remove(&digest))
        .collect()
}

pub(crate) fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn expired() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::ExpiredHandoff,
        "handoff is invalid, expired, already consumed, or belongs to another principal",
    )
}

fn internal() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::Internal,
        "runtime handoff registry is unavailable",
    )
}

fn capacity_exhausted(message: &'static str) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::ResourceExhausted, message).retryable(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::werk_protocol::{ContextCompatibility, ProtocolVersion};

    #[test]
    fn handoff_is_opaque_partitioned_and_single_use() {
        let registry = HandoffRegistry::default();
        let token = registry.issue(record("alice")).unwrap();
        assert!(!format!("{registry:?}").contains(&token));
        assert!(registry.take("bob", &token).is_err());
        assert_eq!(registry.take("alice", &token).unwrap().model_id, "m");

        let token = registry.issue(record("alice")).unwrap();
        assert_eq!(registry.inspect("alice", &token).unwrap().model_id, "m");
        assert!(registry.inspect("bob", &token).is_err());
        assert_eq!(registry.take("alice", &token).unwrap().model_id, "m");
        assert!(registry.take("alice", &token).is_err());
    }

    #[test]
    fn per_principal_capacity_never_evicts_a_valid_handoff() {
        let registry = HandoffRegistry::default();
        let tokens = (0..MAX_HANDOFFS_PER_PRINCIPAL)
            .map(|_| registry.issue(record("alice")).unwrap())
            .collect::<Vec<_>>();

        let error = registry.issue(record("alice")).unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);
        assert!(error.retryable);
        assert!(registry.inspect("alice", &tokens[0]).is_ok());
        assert!(registry.issue(record("bob")).is_ok());
    }

    #[test]
    fn global_capacity_never_evicts_a_valid_handoff() {
        let registry = HandoffRegistry::default();
        let mut first = None;
        for principal in 0..(MAX_HANDOFFS / MAX_HANDOFFS_PER_PRINCIPAL) {
            for _ in 0..MAX_HANDOFFS_PER_PRINCIPAL {
                let token = registry.issue(record(&format!("p_{principal}"))).unwrap();
                first.get_or_insert_with(|| (format!("p_{principal}"), token));
            }
        }

        let error = registry.issue(record("overflow")).unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);
        assert!(error.retryable);
        let (principal, token) = first.unwrap();
        assert!(registry.inspect(&principal, &token).is_ok());
    }

    #[test]
    fn expired_handoffs_are_purged_before_capacity_checks() {
        let registry = HandoffRegistry::default();
        {
            let mut entries = registry.entries.lock().unwrap();
            for index in 0..MAX_HANDOFFS {
                let mut expired = record("alice");
                expired.expires_unix_ms = now_unix_ms().saturating_sub(1);
                entries.entries.insert(format!("{index:064x}"), expired);
            }
        }

        assert!(registry.issue(record("alice")).is_ok());
        assert_eq!(registry.entries.lock().unwrap().entries.len(), 1);
    }

    #[test]
    fn reservations_hold_capacity_and_rollback_on_drop() {
        let registry = HandoffRegistry::default();
        let reservation = registry.reserve("alice").unwrap();
        for _ in 1..MAX_HANDOFFS_PER_PRINCIPAL {
            registry.issue(record("alice")).unwrap();
        }
        assert_eq!(
            registry.issue(record("alice")).unwrap_err().code,
            ProtocolErrorCode::ResourceExhausted
        );

        drop(reservation);
        assert!(registry.issue(record("alice")).is_ok());
    }

    #[test]
    fn consuming_a_full_handoff_atomically_holds_its_replacement_slot() {
        let registry = HandoffRegistry::default();
        let tokens = (0..MAX_HANDOFFS_PER_PRINCIPAL)
            .map(|_| registry.issue(record("alice")).unwrap())
            .collect::<Vec<_>>();

        let (consumed, replacement) = registry.take_with_replacement("alice", &tokens[0]).unwrap();
        assert_eq!(consumed.model_id, "m");
        assert_eq!(
            registry.issue(record("alice")).unwrap_err().code,
            ProtocolErrorCode::ResourceExhausted
        );
        let replacement_token = replacement.issue(record("alice")).unwrap();
        assert!(registry.inspect("alice", &replacement_token).is_ok());
    }

    fn record(principal: &str) -> HandoffRecord {
        HandoffRecord {
            principal_id: principal.to_string(),
            model_id: "m".to_string(),
            state_id: None,
            state: crate::runtime_control::BackendStateLease::new(
                std::sync::Arc::new(crate::runtime_control::UnsupportedRuntimeAdapter::new(
                    "test",
                )),
                crate::runtime_control::BackendState::InProcess {
                    handle: "opaque".to_string(),
                    bytes: Some(1),
                    tier: crate::werk_protocol::StateTier::Ram,
                    instance_id: "instance".to_string(),
                },
            ),
            compatibility: CompatibilityEnvelope {
                model_fingerprint: "m".to_string(),
                tokenizer_fingerprint: "t".to_string(),
                prompt_fingerprint: "p".to_string(),
                chat_template_fingerprint: None,
                backend: "test".to_string(),
                backend_version: "1".to_string(),
                runtime_adapter_version: "1".to_string(),
                accelerator_family: "cpu".to_string(),
                tensor_dtype: "f32".to_string(),
                kv_dtype: "f32".to_string(),
                quantization: "none".to_string(),
                cache_layout: "test".to_string(),
                block_size: None,
                context: ContextCompatibility {
                    context_size: 1,
                    batch_size: None,
                    rope_configuration_fingerprint: None,
                },
                multimodal_processor_fingerprints: Vec::new(),
                producer_protocol: ProtocolVersion::V1,
            },
            expires_unix_ms: now_unix_ms() + 60_000,
        }
    }
}
