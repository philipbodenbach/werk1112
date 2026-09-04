use super::{experts::valid_opaque_id, handoff::now_unix_ms};
use crate::werk_protocol::{
    CompatibilityEnvelope, ProtocolError, ProtocolErrorCode, PruneStatesRequest,
    PruneStatesResponse, StateListFilter, StateListResponse, StateSelector, StateStatus,
    StateSummary, StateTier,
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
#[cfg(windows)]
use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
use std::{
    collections::HashSet,
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const STATE_SCHEMA_VERSION: u32 = 1;
const METADATA_FILE: &str = "metadata.json";
const METADATA_CHECKSUM_FILE: &str = "metadata.sha256";
const PAYLOAD_FILE: &str = "payload.bin";
const PIN_FILE: &str = "pinned";
const LAST_ACCESSED_FILE: &str = "last-accessed";
const QUOTA_EVICTIONS_FILE: &str = ".quota-evictions";
const PIN_MARKER: &[u8] = b"pinned\n";
const QUOTA_EVICTIONS_HEADER: &[u8] = b"werk-quota-evictions-v1\n";
const MAX_QUOTA_EVICTIONS_BYTES: u64 = 1024 * 1024;
const MAX_STATE_DIRECTORY_ENTRIES: usize = 6;
const MAX_METADATA_BYTES: u64 = 1024 * 1024;
const DEFAULT_MAX_PAYLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const DEFAULT_MAX_NAMESPACE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_MAX_NAMESPACE_ENTRIES: usize = 1024;
const DEFAULT_MAX_QUARANTINE_BYTES: u64 = DEFAULT_MAX_PAYLOAD_BYTES + 4 * 1024 * 1024;
const DEFAULT_MAX_QUARANTINE_ENTRIES: usize = 64;
const MAX_QUARANTINE_TREE_ENTRIES: usize = 64;

#[derive(Debug, Clone)]
pub(crate) struct StateStoreLimits {
    pub max_payload_bytes: u64,
    pub max_namespace_bytes: u64,
    pub max_namespace_entries: usize,
    pub max_quarantine_bytes: u64,
    pub max_quarantine_entries: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    pub max_page_size: u16,
    pub max_operation_ids: usize,
}

impl Default for StateStoreLimits {
    fn default() -> Self {
        Self {
            max_payload_bytes: DEFAULT_MAX_PAYLOAD_BYTES,
            max_namespace_bytes: DEFAULT_MAX_NAMESPACE_BYTES,
            max_namespace_entries: DEFAULT_MAX_NAMESPACE_ENTRIES,
            max_quarantine_bytes: DEFAULT_MAX_QUARANTINE_BYTES,
            max_quarantine_entries: DEFAULT_MAX_QUARANTINE_ENTRIES,
            max_page_size: 100,
            max_operation_ids: 100,
        }
    }
}

#[derive(Clone)]
pub(crate) struct NewStoredState {
    pub model_id: String,
    pub backend: String,
    pub compatibility: CompatibilityEnvelope,
    pub payload: OpaquePayloadSource,
    pub prompt_tokens: u64,
    pub expires_unix_ms: Option<u64>,
    pub pinned: bool,
}

impl fmt::Debug for NewStoredState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NewStoredState")
            .field("model_id", &self.model_id)
            .field("backend", &self.backend)
            .field("compatibility", &"[redacted]")
            .field("payload", &self.payload)
            .field("prompt_tokens", &self.prompt_tokens)
            .field("expires_unix_ms", &self.expires_unix_ms)
            .field("pinned", &self.pinned)
            .finish()
    }
}

#[derive(Clone)]
pub(crate) enum OpaquePayloadSource {
    Bytes(Arc<[u8]>),
    File { file: Arc<File>, bytes: u64 },
}

impl fmt::Debug for OpaquePayloadSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes(bytes) => formatter
                .debug_struct("Bytes")
                .field("bytes", &bytes.len())
                .finish(),
            Self::File { bytes, .. } => formatter
                .debug_struct("File")
                .field("bytes", bytes)
                .field("handle", &"[opaque]")
                .finish(),
        }
    }
}

impl OpaquePayloadSource {
    pub(crate) fn open_file(path: &Path, expected_bytes: u64) -> Result<Self, ProtocolError> {
        let metadata = fs::symlink_metadata(path).map_err(|_| corrupt())?;
        if metadata_is_link_or_reparse(&metadata)
            || !metadata.is_file()
            || metadata.len() != expected_bytes
        {
            return Err(corrupt());
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        #[cfg(windows)]
        options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
        let file = options.open(path).map_err(|_| corrupt())?;
        let opened_metadata = file.metadata().map_err(|_| corrupt())?;
        #[cfg(windows)]
        let is_reparse_point = opened_metadata.file_attributes() & 0x0000_0400 != 0;
        #[cfg(not(windows))]
        let is_reparse_point = false;
        if is_reparse_point || !opened_metadata.is_file() || opened_metadata.len() != expected_bytes
        {
            return Err(corrupt());
        }
        Ok(Self::File {
            file: Arc::new(file),
            bytes: expected_bytes,
        })
    }
}

pub(crate) struct LoadedStoredState {
    pub summary: StateSummary,
    pub compatibility: CompatibilityEnvelope,
    /// The exact regular file whose checksum was verified while the catalog
    /// lock was held. Restore paths must consume this handle instead of
    /// reopening a filesystem path.
    pub payload_file: File,
    pub payload_bytes: u64,
    pub prompt_tokens: u64,
}

pub(crate) struct StatePruneResult {
    pub summary: PruneStatesResponse,
    /// States matched under the same catalog lock as the summary and mutation.
    pub matched_states: Vec<StateSummary>,
}

impl fmt::Debug for LoadedStoredState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LoadedStoredState")
            .field("summary", &self.summary)
            .field("compatibility", &"[redacted]")
            .field("payload_file", &"[opaque verified file]")
            .field("payload_bytes", &self.payload_bytes)
            .field("prompt_tokens", &self.prompt_tokens)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredStateMetadata {
    schema_version: u32,
    id: String,
    model_id: String,
    backend: String,
    compatibility: CompatibilityEnvelope,
    compatibility_digest: String,
    payload_bytes: u64,
    payload_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    quota_evictions_sha256: Option<String>,
    prompt_tokens: u64,
    created_unix_ms: u64,
    last_accessed_unix_ms: u64,
    expires_unix_ms: Option<u64>,
}

pub(crate) struct StateStore {
    root: PathBuf,
    limits: StateStoreLimits,
    process_gate: Mutex<StoreProcessState>,
}

#[derive(Default)]
struct StoreProcessState {
    _private: (),
}

impl StateStore {
    pub fn new(home: &Path) -> Self {
        Self::with_limits(home, StateStoreLimits::default())
    }

    pub fn with_limits(home: &Path, limits: StateStoreLimits) -> Self {
        Self {
            root: home.join("runtime-state").join("v1"),
            limits,
            process_gate: Mutex::new(StoreProcessState::default()),
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn reconcile(&self, principal_id: &str) -> Result<u64, ProtocolError> {
        validate_principal_id(principal_id)?;
        let _process = self.process_gate.lock().map_err(|_| internal())?;
        let _file = self.lock_root()?;
        let namespace = self.ensure_namespace(principal_id)?;
        let reconciled = self.reconcile_locked(&namespace)?;
        Ok(reconciled)
    }

    pub fn commit(
        &self,
        principal_id: &str,
        state: NewStoredState,
    ) -> Result<StateSummary, ProtocolError> {
        self.commit_inner(principal_id, None, state)
    }

    pub fn commit_with_id(
        &self,
        principal_id: &str,
        state_id: &str,
        state: NewStoredState,
    ) -> Result<StateSummary, ProtocolError> {
        validate_state_id(state_id)?;
        self.commit_inner(principal_id, Some(state_id), state)
    }

    fn commit_inner(
        &self,
        principal_id: &str,
        requested_id: Option<&str>,
        state: NewStoredState,
    ) -> Result<StateSummary, ProtocolError> {
        self.commit_inner_with_cleanup(principal_id, requested_id, state, true)
    }

    fn commit_inner_with_cleanup(
        &self,
        principal_id: &str,
        requested_id: Option<&str>,
        state: NewStoredState,
        finish_quota_cleanup: bool,
    ) -> Result<StateSummary, ProtocolError> {
        validate_principal_id(principal_id)?;
        validate_model_id(&state.model_id)?;
        if state.backend != state.compatibility.backend {
            return Err(invalid(
                "state backend must match its compatibility envelope",
            ));
        }
        let incoming_payload_bytes = payload_source_len(&state.payload)?;
        if incoming_payload_bytes > self.limits.max_payload_bytes {
            return Err(resource(format!(
                "state payload exceeds the {} byte limit",
                self.limits.max_payload_bytes
            )));
        }
        validate_compatibility(&state.compatibility)?;
        let _process = self.process_gate.lock().map_err(|_| internal())?;
        let _file = self.lock_root()?;
        let namespace = self.ensure_namespace(principal_id)?;
        self.reconcile_locked(&namespace)?;
        let quota_evictions =
            self.quota_eviction_plan_locked(&namespace, incoming_payload_bytes, 1)?;

        let id = match requested_id {
            Some(id) => id.to_string(),
            None => random_id("st_", 18)?,
        };
        if fs::symlink_metadata(namespace.join(&id)).is_ok() {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Conflict,
                "a persistent runtime state with this ID already exists",
            ));
        }
        let staging = namespace.join(random_id(".staging-", 12)?);
        create_owned_directory(&staging)?;
        let now = now_unix_ms();
        let compatibility_digest = sha256_json(&state.compatibility)?;
        let (payload_bytes, payload_sha256) = match write_payload_new(
            &staging.join(PAYLOAD_FILE),
            &state.payload,
            self.limits.max_payload_bytes,
        ) {
            Ok(result) => result,
            Err(error) => {
                let _ = remove_entry_without_following(&staging);
                return Err(error);
            }
        };
        let quota_evictions_journal = if quota_evictions.is_empty() {
            None
        } else {
            Some(encode_quota_evictions(&quota_evictions)?)
        };
        let metadata = StoredStateMetadata {
            schema_version: STATE_SCHEMA_VERSION,
            id: id.clone(),
            model_id: state.model_id,
            backend: state.backend,
            compatibility: state.compatibility,
            compatibility_digest,
            payload_bytes,
            payload_sha256,
            quota_evictions_sha256: quota_evictions_journal.as_deref().map(sha256_bytes),
            prompt_tokens: state.prompt_tokens,
            created_unix_ms: now,
            last_accessed_unix_ms: now,
            expires_unix_ms: state.expires_unix_ms,
        };
        let write_result: Result<(), ProtocolError> = (|| {
            let metadata_bytes = serde_json::to_vec_pretty(&metadata).map_err(|_| internal())?;
            if metadata_bytes.len() as u64 > MAX_METADATA_BYTES {
                return Err(invalid(
                    "serialized runtime state metadata exceeds its limit",
                ));
            }
            write_new_file(&staging.join(METADATA_FILE), &metadata_bytes)?;
            write_new_file(
                &staging.join(METADATA_CHECKSUM_FILE),
                format!("{}\n", sha256_bytes(&metadata_bytes)).as_bytes(),
            )?;
            if state.pinned {
                write_new_file(&staging.join(PIN_FILE), PIN_MARKER)?;
            }
            if let Some(journal) = &quota_evictions_journal {
                write_new_file(&staging.join(QUOTA_EVICTIONS_FILE), journal)?;
            }
            sync_directory(&staging)?;
            fs::rename(&staging, namespace.join(&id)).map_err(|_| internal())?;
            sync_directory(&namespace)?;
            Ok(())
        })();
        if write_result.is_err() {
            let _ = remove_entry_without_following(&staging);
        }
        write_result?;
        if finish_quota_cleanup && !quota_evictions.is_empty() {
            // The new state is already durable at this point. A failed
            // post-commit eviction must not turn a successful publish into an
            // ambiguous error or remove the new state. The marker remains and
            // the next mutating reconciliation deterministically finishes the
            // same quota repair.
            let _ = self.finish_quota_evictions_locked(&namespace, &id, &quota_evictions);
        }
        Ok(summary_from_metadata(&metadata, state.pinned))
    }

    pub fn load(
        &self,
        principal_id: &str,
        state_id: &str,
    ) -> Result<LoadedStoredState, ProtocolError> {
        self.load_with_access_update(principal_id, state_id, true)
    }

    /// Verifies and returns a stored state without changing its LRU metadata.
    ///
    /// This is used by dry-run control paths so inspecting an existing state
    /// does not change its persisted last-accessed value.
    pub fn inspect(
        &self,
        principal_id: &str,
        state_id: &str,
    ) -> Result<LoadedStoredState, ProtocolError> {
        self.load_with_access_update(principal_id, state_id, false)
    }

    fn load_with_access_update(
        &self,
        principal_id: &str,
        state_id: &str,
        update_access: bool,
    ) -> Result<LoadedStoredState, ProtocolError> {
        validate_principal_id(principal_id)?;
        validate_state_id(state_id)?;
        let _process = self.process_gate.lock().map_err(|_| internal())?;
        if update_access {
            let _file = self.lock_root()?;
            let namespace = self.ensure_namespace(principal_id)?;
            self.reconcile_locked_except(&namespace, Some(state_id))?;
            self.load_locked(&namespace, state_id, true, true)
        } else {
            let Some((_file, namespace)) = self.existing_locked_namespace(principal_id)? else {
                return Err(not_found());
            };
            self.load_locked(&namespace, state_id, false, false)
        }
    }

    pub fn find_compatible(
        &self,
        principal_id: &str,
        model_id: &str,
        expected: &CompatibilityEnvelope,
        reject_incompatible: bool,
    ) -> Result<Option<LoadedStoredState>, ProtocolError> {
        validate_principal_id(principal_id)?;
        validate_model_id(model_id)?;
        validate_compatibility(expected)?;
        let _process = self.process_gate.lock().map_err(|_| internal())?;
        let _file = self.lock_root()?;
        let namespace = self.ensure_namespace(principal_id)?;
        self.reconcile_locked(&namespace)?;
        let mut compatible = Vec::<StoredStateMetadata>::new();
        let mut mismatches = HashSet::<&'static str>::new();
        for entry in read_dir_bounded(&namespace, self.namespace_entry_scan_limit())? {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("st_") || !valid_opaque_id(&name) {
                continue;
            }
            let Ok((metadata, _)) = read_state(&entry.path(), self.limits.max_payload_bytes) else {
                continue;
            };
            if metadata.model_id != model_id
                || metadata.compatibility.prompt_fingerprint != expected.prompt_fingerprint
                || metadata
                    .expires_unix_ms
                    .is_some_and(|expires| expires <= now_unix_ms())
            {
                continue;
            }
            let changed = metadata.compatibility.mismatch_fields(expected);
            if changed.is_empty() {
                compatible.push(metadata);
            } else {
                mismatches.extend(changed);
            }
        }
        compatible.sort_by(|left, right| {
            right
                .last_accessed_unix_ms
                .cmp(&left.last_accessed_unix_ms)
                .then_with(|| left.id.cmp(&right.id))
        });
        if let Some(metadata) = compatible.first() {
            return self
                .load_locked(&namespace, &metadata.id, true, true)
                .map(Some);
        }
        if reject_incompatible && !mismatches.is_empty() {
            let mut mismatches = mismatches.into_iter().collect::<Vec<_>>();
            mismatches.sort_unstable();
            return Err(ProtocolError::new(
                ProtocolErrorCode::IncompatibleState,
                "a reusable state exists, but its compatibility could not be proven",
            )
            .with_details(json!({"mismatch_fields": mismatches})));
        }
        Ok(None)
    }

    fn load_locked(
        &self,
        namespace: &Path,
        state_id: &str,
        update_access: bool,
        quarantine_corrupt: bool,
    ) -> Result<LoadedStoredState, ProtocolError> {
        if !has_exact_child_name(
            namespace,
            state_id,
            self.limits.max_namespace_entries.saturating_add(64),
        )? {
            return Err(not_found());
        }
        let state_dir = namespace.join(state_id);
        let (mut metadata, pinned) = match read_state(&state_dir, self.limits.max_payload_bytes) {
            Ok(state) => state,
            Err(error) if quarantine_corrupt && error.code == ProtocolErrorCode::CorruptState => {
                self.quarantine_locked(&state_dir)?;
                return Err(ProtocolError::new(
                    ProtocolErrorCode::CorruptState,
                    "runtime state failed integrity verification and was removed from the active catalog",
                ));
            }
            Err(error) => return Err(error),
        };
        if metadata
            .expires_unix_ms
            .is_some_and(|expires| expires <= now_unix_ms())
        {
            if quarantine_corrupt {
                remove_state_dir(&state_dir)?;
                sync_directory(namespace)?;
            }
            return Err(expired_state());
        }
        let payload_path = state_dir.join(PAYLOAD_FILE);
        let (payload_file, payload_bytes, payload_checksum) =
            open_and_sha256_regular_file(&payload_path, self.limits.max_payload_bytes)?;
        if payload_bytes != metadata.payload_bytes || payload_checksum != metadata.payload_sha256 {
            if quarantine_corrupt {
                self.quarantine_locked(&state_dir)?;
                return Err(ProtocolError::new(
                    ProtocolErrorCode::CorruptState,
                    "runtime state payload failed integrity verification and was removed from the active catalog",
                ));
            }
            return Err(corrupt());
        }
        if update_access {
            let accessed = now_unix_ms().max(metadata.last_accessed_unix_ms);
            write_last_accessed(&state_dir, accessed)?;
            metadata.last_accessed_unix_ms = accessed;
        }
        Ok(LoadedStoredState {
            summary: summary_from_metadata(&metadata, pinned),
            compatibility: metadata.compatibility,
            payload_file,
            payload_bytes,
            prompt_tokens: metadata.prompt_tokens,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn list(
        &self,
        principal_id: &str,
        filter: &StateListFilter,
    ) -> Result<StateListResponse, ProtocolError> {
        validate_filter(filter, self.limits.max_page_size)?;
        let _process = self.process_gate.lock().map_err(|_| internal())?;
        let Some((_file, namespace)) = self.existing_locked_namespace(principal_id)? else {
            return Ok(StateListResponse {
                states: Vec::new(),
                next_cursor: None,
            });
        };
        let mut states = self.scan_locked(&namespace)?;
        let now = now_unix_ms();
        states.retain(|state| {
            !state.expires_unix_ms.is_some_and(|expires| expires <= now)
                && filter
                    .model_id
                    .as_deref()
                    .is_none_or(|model| state.model_id == model)
                && filter.tier.is_none_or(|tier| state.tier == tier)
        });
        states.sort_by(|left, right| left.id.cmp(&right.id));
        let after = filter.cursor.as_deref().map(decode_cursor).transpose()?;
        if let Some(after) = after {
            states.retain(|state| state.id > after);
        }
        let limit = usize::from(filter.limit.unwrap_or(self.limits.max_page_size));
        let has_more = states.len() > limit;
        states.truncate(limit);
        let next_cursor = has_more
            .then(|| states.last().map(|state| encode_cursor(&state.id)))
            .flatten();
        Ok(StateListResponse {
            states,
            next_cursor,
        })
    }

    pub fn all_summaries(&self, principal_id: &str) -> Result<Vec<StateSummary>, ProtocolError> {
        validate_principal_id(principal_id)?;
        let _process = self.process_gate.lock().map_err(|_| internal())?;
        let Some((_file, namespace)) = self.existing_locked_namespace(principal_id)? else {
            return Ok(Vec::new());
        };
        let mut states = self.scan_locked(&namespace)?;
        let now = now_unix_ms();
        states.retain(|state| !state.expires_unix_ms.is_some_and(|expires| expires <= now));
        states.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(states)
    }

    pub fn set_pinned(
        &self,
        principal_id: &str,
        state_id: &str,
        pinned: bool,
        dry_run: bool,
    ) -> Result<(StateSummary, bool), ProtocolError> {
        validate_principal_id(principal_id)?;
        validate_state_id(state_id)?;
        let _process = self.process_gate.lock().map_err(|_| internal())?;
        let (_file, namespace) = if dry_run {
            let Some(context) = self.existing_locked_namespace(principal_id)? else {
                return Err(not_found());
            };
            context
        } else {
            let file = self.lock_root()?;
            let namespace = self.ensure_namespace(principal_id)?;
            self.reconcile_locked_except(&namespace, Some(state_id))?;
            (file, namespace)
        };
        let state_dir = namespace.join(state_id);
        let loaded = self.load_locked(&namespace, state_id, false, !dry_run)?;
        let was_pinned = loaded.summary.pinned;
        let changed = was_pinned != pinned;
        if changed && !dry_run {
            let marker = state_dir.join(PIN_FILE);
            if pinned {
                publish_new_state_marker(&state_dir, PIN_FILE, PIN_MARKER)?;
            } else {
                remove_regular_file(&marker)?;
                sync_directory(&state_dir)?;
            }
        }
        let mut summary = loaded.summary;
        summary.pinned = pinned;
        Ok((summary, changed))
    }

    pub fn prune(
        &self,
        principal_id: &str,
        request: &PruneStatesRequest,
    ) -> Result<PruneStatesResponse, ProtocolError> {
        Ok(self.prune_detailed(principal_id, request)?.summary)
    }

    pub(crate) fn prune_detailed(
        &self,
        principal_id: &str,
        request: &PruneStatesRequest,
    ) -> Result<StatePruneResult, ProtocolError> {
        validate_principal_id(principal_id)?;
        validate_selector(&request.selector, self.limits.max_operation_ids)?;
        let _process = self.process_gate.lock().map_err(|_| internal())?;
        let context = if request.dry_run {
            self.existing_locked_namespace(principal_id)?
        } else {
            let file = self.lock_root()?;
            let namespace = self.ensure_namespace(principal_id)?;
            self.reconcile_locked(&namespace)?;
            Some((file, namespace))
        };
        let Some((_file, namespace)) = context else {
            return Ok(StatePruneResult {
                summary: PruneStatesResponse {
                    matched: 0,
                    removed: 0,
                    bytes: Some(0),
                    dry_run: request.dry_run,
                },
                matched_states: Vec::new(),
            });
        };
        let mut states = self.scan_locked(&namespace)?;
        let now = now_unix_ms();
        states.retain(|state| !state.expires_unix_ms.is_some_and(|expires| expires <= now));
        states.retain(|state| selector_matches(&request.selector, state));
        states.sort_by(|left, right| left.id.cmp(&right.id));
        let matched = states.len() as u64;
        let bytes = states.iter().fold(0u64, |total, state| {
            total.saturating_add(state.bytes.unwrap_or(0))
        });
        if !request.dry_run {
            for state in &states {
                remove_state_dir(&namespace.join(&state.id))?;
            }
            sync_directory(&namespace)?;
        }
        Ok(StatePruneResult {
            summary: PruneStatesResponse {
                matched,
                removed: if request.dry_run { 0 } else { matched },
                bytes: Some(bytes),
                dry_run: request.dry_run,
            },
            matched_states: states,
        })
    }

    fn lock_root(&self) -> Result<RootLock, ProtocolError> {
        ensure_owned_directory_tree(&self.root)?;
        let path = self.root.join(".lock");
        let lock_existed = match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata_is_link_or_reparse(&metadata) || !metadata.is_file() => {
                return Err(ProtocolError::new(
                    ProtocolErrorCode::Forbidden,
                    "runtime state lock is unsafe",
                ));
            }
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(_) => return Err(internal()),
        };
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);
        #[cfg(unix)]
        options
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        #[cfg(windows)]
        options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
        let file = options.open(&path).map_err(|_| internal())?;
        if !lock_existed {
            sync_directory(&self.root)?;
        }
        #[cfg(windows)]
        if file.metadata().map_err(|_| internal())?.file_attributes() & 0x0000_0400 != 0 {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Forbidden,
                "runtime state lock is unsafe",
            ));
        }
        #[cfg(unix)]
        {
            let metadata = file.metadata().map_err(|_| internal())?;
            if metadata.uid() != unsafe { libc::geteuid() } {
                return Err(forbidden(
                    "runtime state lock is not owned by the current user",
                ));
            }
            file.set_permissions(fs::Permissions::from_mode(0o600))
                .map_err(|_| internal())?;
        }
        FileExt::try_lock_exclusive(&file).map_err(|_| {
            ProtocolError::new(
                ProtocolErrorCode::Unavailable,
                "runtime state catalog is locked by another process",
            )
            .retryable(true)
        })?;
        let lock = RootLock(file);
        self.maintain_quarantine_locked()?;
        Ok(lock)
    }

    /// Opens an already-initialized catalog without creating directories,
    /// lock files, or changing permissions. This is the only catalog entry
    /// point used by read-only and dry-run operations.
    fn existing_locked_namespace(
        &self,
        principal_id: &str,
    ) -> Result<Option<(RootLock, PathBuf)>, ProtocolError> {
        validate_principal_id(principal_id)?;
        let parent = self.root.parent().ok_or_else(internal)?;
        if !validate_existing_owned_directory(parent)?
            || !validate_existing_owned_directory(&self.root)?
        {
            return Ok(None);
        }

        let lock_path = self.root.join(".lock");
        let lock_metadata = match fs::symlink_metadata(&lock_path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(unavailable(
                    "runtime state catalog is not initialized for safe read-only access",
                ));
            }
            Err(_) => return Err(internal()),
        };
        if metadata_is_link_or_reparse(&lock_metadata) || !lock_metadata.is_file() {
            return Err(forbidden("runtime state lock is unsafe"));
        }
        #[cfg(unix)]
        if lock_metadata.uid() != unsafe { libc::geteuid() }
            || lock_metadata.permissions().mode() & 0o077 != 0
        {
            return Err(forbidden("runtime state lock is not owner-only"));
        }

        let mut options = OpenOptions::new();
        options.read(true).write(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        #[cfg(windows)]
        options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
        let file = options.open(&lock_path).map_err(|_| internal())?;
        #[cfg(unix)]
        if file.metadata().map_err(|_| internal())?.uid() != unsafe { libc::geteuid() } {
            return Err(forbidden(
                "runtime state lock is not owned by the current user",
            ));
        }
        #[cfg(windows)]
        if file.metadata().map_err(|_| internal())?.file_attributes() & 0x0000_0400 != 0 {
            return Err(forbidden("runtime state lock is unsafe"));
        }
        FileExt::try_lock_exclusive(&file)
            .map_err(|_| unavailable("runtime state catalog is locked by another process"))?;

        let namespace = self.root.join(principal_id);
        if !validate_existing_owned_directory(&namespace)? {
            return Ok(None);
        }
        Ok(Some((RootLock(file), namespace)))
    }

    fn ensure_namespace(&self, principal_id: &str) -> Result<PathBuf, ProtocolError> {
        validate_principal_id(principal_id)?;
        let namespace = self.root.join(principal_id);
        create_or_validate_owned_directory(&namespace)?;
        Ok(namespace)
    }

    fn scan_locked(&self, namespace: &Path) -> Result<Vec<StateSummary>, ProtocolError> {
        let mut states = Vec::new();
        let mut scan_budget = self.payload_scan_budget();
        for entry in read_dir_bounded(namespace, self.namespace_entry_scan_limit())? {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            if !name.starts_with("st_") || !valid_opaque_id(&name) {
                continue;
            }
            let path = entry.path();
            if let Ok((metadata, pinned)) = read_state(&path, self.limits.max_payload_bytes) {
                scan_budget.charge(metadata.payload_bytes)?;
                let payload =
                    sha256_regular_file(&path.join(PAYLOAD_FILE), self.limits.max_payload_bytes);
                if payload.is_ok_and(|(bytes, checksum)| {
                    bytes == metadata.payload_bytes && checksum == metadata.payload_sha256
                }) {
                    states.push(summary_from_metadata(&metadata, pinned));
                }
            }
        }
        Ok(states)
    }

    fn reconcile_locked(&self, namespace: &Path) -> Result<u64, ProtocolError> {
        self.reconcile_locked_except(namespace, None)
    }

    fn reconcile_locked_except(
        &self,
        namespace: &Path,
        excluded_state_id: Option<&str>,
    ) -> Result<u64, ProtocolError> {
        let mut reconciled = self.complete_pending_quota_evictions_locked(namespace)?;
        let now = now_unix_ms();
        let mut scan_budget = self.payload_scan_budget();
        for entry in read_dir_bounded(namespace, self.namespace_entry_scan_limit())? {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            if excluded_state_id == Some(name.as_str()) {
                continue;
            }
            if name.starts_with(".staging-") {
                remove_entry_without_following(&path)?;
                reconciled = reconciled.saturating_add(1);
                continue;
            }
            let state = (name.starts_with("st_") && valid_opaque_id(&name))
                .then(|| read_state(&path, self.limits.max_payload_bytes))
                .transpose()
                .and_then(|state| state.ok_or_else(corrupt));
            match state {
                Ok((metadata, _))
                    if metadata
                        .expires_unix_ms
                        .is_some_and(|expires| expires <= now) =>
                {
                    // Expiry is an unconditional retention boundary. Pinning
                    // protects policy eviction only while the state is live.
                    remove_state_dir(&path)?;
                    reconciled = reconciled.saturating_add(1);
                }
                Ok((metadata, _)) => {
                    scan_budget.charge(metadata.payload_bytes)?;
                    let payload = sha256_regular_file(
                        &path.join(PAYLOAD_FILE),
                        self.limits.max_payload_bytes,
                    );
                    let valid = payload.is_ok_and(|(bytes, checksum)| {
                        bytes == metadata.payload_bytes && checksum == metadata.payload_sha256
                    });
                    if !valid {
                        self.quarantine_locked(&path)?;
                        reconciled = reconciled.saturating_add(1);
                    }
                }
                Err(_) => {
                    self.quarantine_locked(&path)?;
                    reconciled = reconciled.saturating_add(1);
                }
            }
        }
        if reconciled > 0 {
            sync_directory(namespace)?;
        }
        Ok(reconciled)
    }

    fn complete_pending_quota_evictions_locked(
        &self,
        namespace: &Path,
    ) -> Result<u64, ProtocolError> {
        let mut journals = Vec::new();
        let mut scan_budget = self.payload_scan_budget();
        for entry in read_dir_bounded(namespace, self.namespace_entry_scan_limit())? {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("st_") || !valid_opaque_id(&name) {
                continue;
            }
            let path = entry.path();
            let Ok((metadata, _)) = read_state(&path, self.limits.max_payload_bytes) else {
                continue;
            };
            scan_budget.charge(metadata.payload_bytes)?;
            let payload =
                sha256_regular_file(&path.join(PAYLOAD_FILE), self.limits.max_payload_bytes);
            if !payload.is_ok_and(|(bytes, checksum)| {
                bytes == metadata.payload_bytes && checksum == metadata.payload_sha256
            }) {
                continue;
            }
            let Ok(Some(victims)) = read_quota_evictions(&path) else {
                continue;
            };
            let journal_digest = sha256_bytes(&encode_quota_evictions(&victims)?);
            if metadata.quota_evictions_sha256.as_deref() != Some(journal_digest.as_str()) {
                continue;
            }
            journals.push((name, victims));
        }
        if journals.is_empty() {
            return Ok(0);
        }

        let owners = journals
            .iter()
            .map(|(state_id, _)| state_id.clone())
            .collect::<HashSet<_>>();
        let mut seen = HashSet::new();
        let journaled = journals
            .iter()
            .flat_map(|(_, victims)| victims.iter())
            .filter(|victim| !owners.contains(*victim) && seen.insert((*victim).clone()))
            .cloned()
            .collect::<Vec<_>>();
        let mut removed = self.remove_quota_victims_locked(namespace, &journaled)?;
        // If an earlier crash happened between victim removals, finish any
        // remaining quota repair without ever selecting a marker-bearing
        // committed state as the recovery victim.
        let additional = self.quota_eviction_plan_locked_excluding(namespace, 0, 0, &owners)?;
        removed = removed.saturating_add(self.remove_quota_victims_locked(namespace, &additional)?);

        // Only clear journals after every required victim deletion and the
        // namespace sync have succeeded. A crash or error before this point
        // leaves a durable marker for the next reconciliation.
        for (state_id, _) in journals {
            self.clear_quota_marker_locked(&namespace.join(state_id))?;
        }
        Ok(removed)
    }

    fn quarantine_locked(&self, path: &Path) -> Result<(), ProtocolError> {
        let source_parent = path.parent().ok_or_else(internal)?;
        let metadata = fs::symlink_metadata(path).map_err(|_| corrupt())?;
        if metadata_is_link_or_reparse(&metadata) || !metadata.is_dir() {
            remove_entry_without_following(path)?;
            sync_directory(source_parent)?;
            return Ok(());
        }

        let source_bytes = match logical_regular_bytes_bounded(
            path,
            MAX_QUARANTINE_TREE_ENTRIES,
            self.limits.max_quarantine_bytes,
        ) {
            Ok(bytes) => bytes,
            Err(_) => {
                // An unmeasurable or structurally unbounded corrupt tree must
                // not escape retention limits merely because it is evidence.
                remove_entry_without_following(path)?;
                sync_directory(source_parent)?;
                return Ok(());
            }
        };
        if self.limits.max_quarantine_entries == 0
            || source_bytes > self.limits.max_quarantine_bytes
        {
            remove_entry_without_following(path)?;
            sync_directory(source_parent)?;
            return Ok(());
        }

        let quarantine = self.root.join(".quarantine");
        create_or_validate_owned_directory(&quarantine)?;
        // The quarantine directory itself may have been created just above;
        // persist that root entry before moving evidence into it.
        sync_directory(&self.root)?;
        if !self.make_quarantine_room_locked(&quarantine, source_bytes, 1)? {
            remove_entry_without_following(path)?;
            sync_directory(source_parent)?;
            return Ok(());
        }
        let prefix = format!("q_{:020}_", now_unix_ms());
        let target = quarantine.join(random_id(&prefix, 18)?);
        fs::rename(path, target).map_err(|_| corrupt())?;
        sync_directory(&quarantine)?;
        sync_directory(source_parent)?;
        Ok(())
    }

    fn make_quarantine_room_locked(
        &self,
        quarantine: &Path,
        incoming_bytes: u64,
        incoming_entries: usize,
    ) -> Result<bool, ProtocolError> {
        if incoming_bytes > self.limits.max_quarantine_bytes
            || incoming_entries > self.limits.max_quarantine_entries
        {
            return Ok(false);
        }
        let scan_limit = self
            .limits
            .max_quarantine_entries
            .saturating_add(64)
            .max(64);
        let entries = match read_dir_bounded(quarantine, scan_limit) {
            Ok(entries) => entries,
            Err(error) if error.code == ProtocolErrorCode::ResourceExhausted => {
                // An inherited or externally modified quarantine that cannot
                // be audited within its entry bound is discarded wholesale.
                // It contains only already-invalid catalog material.
                remove_entry_without_following(quarantine)?;
                sync_directory(&self.root)?;
                create_owned_directory(quarantine)?;
                sync_directory(&self.root)?;
                return Ok(incoming_entries <= self.limits.max_quarantine_entries
                    && incoming_bytes <= self.limits.max_quarantine_bytes);
            }
            Err(error) => return Err(error),
        };
        let mut retained = Vec::new();
        let mut changed = false;
        for entry in entries {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("q_") {
                remove_entry_without_following(&path)?;
                changed = true;
                continue;
            }
            match logical_regular_bytes_bounded(
                &path,
                MAX_QUARANTINE_TREE_ENTRIES,
                self.limits.max_quarantine_bytes,
            ) {
                Ok(bytes) => retained.push((name, path, bytes)),
                Err(_) => {
                    remove_entry_without_following(&path)?;
                    changed = true;
                }
            }
        }
        retained.sort_by(|left, right| left.0.cmp(&right.0));
        let mut total_bytes = retained
            .iter()
            .fold(0u64, |total, entry| total.saturating_add(entry.2));
        while retained.len().saturating_add(incoming_entries) > self.limits.max_quarantine_entries
            || total_bytes.saturating_add(incoming_bytes) > self.limits.max_quarantine_bytes
        {
            if retained.is_empty() {
                break;
            }
            let (_, path, bytes) = retained.remove(0);
            remove_entry_without_following(&path)?;
            total_bytes = total_bytes.saturating_sub(bytes);
            changed = true;
        }
        if changed {
            sync_directory(quarantine)?;
        }
        Ok(
            retained.len().saturating_add(incoming_entries) <= self.limits.max_quarantine_entries
                && total_bytes.saturating_add(incoming_bytes) <= self.limits.max_quarantine_bytes,
        )
    }

    fn maintain_quarantine_locked(&self) -> Result<(), ProtocolError> {
        let quarantine = self.root.join(".quarantine");
        let metadata = match fs::symlink_metadata(&quarantine) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(_) => return Err(internal()),
        };
        if metadata_is_link_or_reparse(&metadata) || !metadata.is_dir() {
            return Err(forbidden("runtime state quarantine is unsafe"));
        }
        if self.limits.max_quarantine_entries == 0 {
            remove_entry_without_following(&quarantine)?;
            sync_directory(&self.root)?;
            return Ok(());
        }
        let _ = self.make_quarantine_room_locked(&quarantine, 0, 0)?;
        Ok(())
    }

    fn quota_eviction_plan_locked(
        &self,
        namespace: &Path,
        incoming_bytes: u64,
        incoming_entries: usize,
    ) -> Result<Vec<String>, ProtocolError> {
        self.quota_eviction_plan_locked_excluding(
            namespace,
            incoming_bytes,
            incoming_entries,
            &HashSet::new(),
        )
    }

    fn quota_eviction_plan_locked_excluding(
        &self,
        namespace: &Path,
        incoming_bytes: u64,
        incoming_entries: usize,
        excluded: &HashSet<String>,
    ) -> Result<Vec<String>, ProtocolError> {
        if incoming_bytes > self.limits.max_namespace_bytes
            || incoming_entries > self.limits.max_namespace_entries
        {
            return Err(resource("runtime state exceeds the namespace quota"));
        }
        let mut states = self.scan_locked(namespace)?;
        let mut total_bytes = states.iter().fold(0u64, |total, state| {
            total.saturating_add(state.bytes.unwrap_or(0))
        });
        let mut total_entries = states.len();
        states.sort_by(|left, right| {
            left.last_accessed_unix_ms
                .cmp(&right.last_accessed_unix_ms)
                .then_with(|| left.id.cmp(&right.id))
        });
        let mut evictions = Vec::new();
        for state in states
            .into_iter()
            .filter(|state| !state.pinned && !excluded.contains(&state.id))
        {
            if total_bytes.saturating_add(incoming_bytes) <= self.limits.max_namespace_bytes
                && total_entries.saturating_add(incoming_entries)
                    <= self.limits.max_namespace_entries
            {
                break;
            }
            total_bytes = total_bytes.saturating_sub(state.bytes.unwrap_or(0));
            total_entries = total_entries.saturating_sub(1);
            evictions.push(state.id);
        }
        if total_bytes.saturating_add(incoming_bytes) > self.limits.max_namespace_bytes
            || total_entries.saturating_add(incoming_entries) > self.limits.max_namespace_entries
        {
            return Err(resource(
                "runtime state namespace quota cannot be satisfied because remaining states are pinned",
            ));
        }
        Ok(evictions)
    }

    fn finish_quota_evictions_locked(
        &self,
        namespace: &Path,
        committed_state_id: &str,
        evictions: &[String],
    ) -> Result<(), ProtocolError> {
        self.remove_quota_victims_locked(namespace, evictions)?;
        self.clear_quota_marker_locked(&namespace.join(committed_state_id))
    }

    fn remove_quota_victims_locked(
        &self,
        namespace: &Path,
        evictions: &[String],
    ) -> Result<u64, ProtocolError> {
        let mut removed = 0u64;
        for id in evictions {
            validate_state_id(id).map_err(|_| corrupt())?;
            let path = namespace.join(id);
            match fs::symlink_metadata(&path) {
                Ok(_) => {
                    remove_state_dir(&path)?;
                    removed = removed.saturating_add(1);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => return Err(internal()),
            }
        }
        if !evictions.is_empty() {
            sync_directory(namespace)?;
        }
        Ok(removed)
    }

    fn clear_quota_marker_locked(&self, state_dir: &Path) -> Result<(), ProtocolError> {
        match fs::symlink_metadata(state_dir) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(_) => return Err(internal()),
            Ok(metadata) if metadata_is_link_or_reparse(&metadata) || !metadata.is_dir() => {
                return Err(corrupt());
            }
            Ok(_) => {}
        }
        let marker = state_dir.join(QUOTA_EVICTIONS_FILE);
        if read_quota_evictions(state_dir)?.is_none() {
            return Ok(());
        }
        remove_regular_file(&marker)?;
        sync_directory(state_dir)
    }

    fn namespace_entry_scan_limit(&self) -> usize {
        self.limits.max_namespace_entries.saturating_add(64)
    }

    fn payload_scan_budget(&self) -> PayloadScanBudget {
        // A crash may leave exactly one newly published state awaiting its
        // quota evictions. Permit that bounded, recoverable overage, but never
        // hash an attacker-expanded namespace without an aggregate ceiling.
        PayloadScanBudget::new(
            self.limits
                .max_namespace_bytes
                .saturating_add(self.limits.max_payload_bytes),
        )
    }
}

struct RootLock(File);

impl Drop for RootLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

struct PayloadScanBudget {
    remaining_bytes: u64,
}

impl PayloadScanBudget {
    fn new(max_bytes: u64) -> Self {
        Self {
            remaining_bytes: max_bytes,
        }
    }

    fn charge(&mut self, bytes: u64) -> Result<(), ProtocolError> {
        self.remaining_bytes = self.remaining_bytes.checked_sub(bytes).ok_or_else(|| {
            resource("runtime state catalog exceeds the aggregate payload scan limit")
        })?;
        Ok(())
    }
}

fn read_state(
    state_dir: &Path,
    max_payload_bytes: u64,
) -> Result<(StoredStateMetadata, bool), ProtocolError> {
    let directory_metadata = fs::symlink_metadata(state_dir).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ProtocolError::new(ProtocolErrorCode::NotFound, "runtime state was not found")
        } else {
            corrupt()
        }
    })?;
    if metadata_is_link_or_reparse(&directory_metadata) || !directory_metadata.is_dir() {
        return Err(corrupt());
    }
    if !metadata_is_owned_private(&directory_metadata) {
        return Err(corrupt());
    }
    validate_state_children(state_dir)?;
    let metadata_bytes =
        read_bounded_regular_file(&state_dir.join(METADATA_FILE), MAX_METADATA_BYTES)?;
    let checksum = read_bounded_regular_file(&state_dir.join(METADATA_CHECKSUM_FILE), 128)?;
    let checksum = std::str::from_utf8(&checksum)
        .map_err(|_| corrupt())?
        .trim();
    if checksum != sha256_bytes(&metadata_bytes) {
        return Err(corrupt());
    }
    let mut metadata: StoredStateMetadata =
        serde_json::from_slice(&metadata_bytes).map_err(|_| corrupt())?;
    if metadata.schema_version != STATE_SCHEMA_VERSION
        || state_dir.file_name().and_then(|name| name.to_str()) != Some(&metadata.id)
        || validate_state_id(&metadata.id).is_err()
        || metadata.payload_bytes > max_payload_bytes
        || metadata.backend != metadata.compatibility.backend
        || sha256_json(&metadata.compatibility)? != metadata.compatibility_digest
    {
        return Err(corrupt());
    }
    validate_model_id(&metadata.model_id).map_err(|_| corrupt())?;
    validate_compatibility(&metadata.compatibility).map_err(|_| corrupt())?;
    let payload_metadata =
        fs::symlink_metadata(state_dir.join(PAYLOAD_FILE)).map_err(|_| corrupt())?;
    if metadata_is_link_or_reparse(&payload_metadata)
        || !payload_metadata.is_file()
        || payload_metadata.len() != metadata.payload_bytes
        || !metadata_is_owned_private(&payload_metadata)
    {
        return Err(corrupt());
    }
    if let Some(accessed) = read_last_accessed(state_dir)? {
        metadata.last_accessed_unix_ms = metadata.last_accessed_unix_ms.max(accessed);
    }
    let pinned = exact_marker_exists(state_dir.join(PIN_FILE), PIN_MARKER)?;
    if let Some(evictions) = read_quota_evictions(state_dir)? {
        let encoded = encode_quota_evictions(&evictions)?;
        let journal_digest = sha256_bytes(&encoded);
        if metadata.quota_evictions_sha256.as_deref() != Some(journal_digest.as_str()) {
            return Err(corrupt());
        }
    }
    Ok((metadata, pinned))
}

fn validate_state_children(state_dir: &Path) -> Result<(), ProtocolError> {
    let entries =
        read_dir_bounded(state_dir, MAX_STATE_DIRECTORY_ENTRIES).map_err(|_| corrupt())?;
    for entry in entries {
        let name = entry.file_name().into_string().map_err(|_| corrupt())?;
        if !matches!(
            name.as_str(),
            METADATA_FILE
                | METADATA_CHECKSUM_FILE
                | PAYLOAD_FILE
                | PIN_FILE
                | LAST_ACCESSED_FILE
                | QUOTA_EVICTIONS_FILE
        ) {
            return Err(corrupt());
        }
    }
    Ok(())
}

fn exact_marker_exists(path: PathBuf, expected: &[u8]) -> Result<bool, ProtocolError> {
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata_is_link_or_reparse(&metadata) || !metadata.is_file() => {
            Err(corrupt())
        }
        Ok(metadata) if metadata.len() != expected.len() as u64 => Err(corrupt()),
        Ok(_) => {
            let bytes = read_bounded_regular_file(&path, expected.len() as u64)?;
            if bytes == expected {
                Ok(true)
            } else {
                Err(corrupt())
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(_) => Err(corrupt()),
    }
}

fn encode_quota_evictions(evictions: &[String]) -> Result<Vec<u8>, ProtocolError> {
    let mut encoded = Vec::from(QUOTA_EVICTIONS_HEADER);
    let mut seen = HashSet::new();
    for state_id in evictions {
        validate_state_id(state_id).map_err(|_| corrupt())?;
        if !seen.insert(state_id.as_str()) {
            return Err(corrupt());
        }
        encoded.extend_from_slice(state_id.as_bytes());
        encoded.push(b'\n');
        if encoded.len() as u64 > MAX_QUOTA_EVICTIONS_BYTES {
            return Err(resource("runtime state quota journal exceeds its bound"));
        }
    }
    if evictions.is_empty() {
        return Err(corrupt());
    }
    Ok(encoded)
}

fn read_quota_evictions(state_dir: &Path) -> Result<Option<Vec<String>>, ProtocolError> {
    let path = state_dir.join(QUOTA_EVICTIONS_FILE);
    let metadata = match fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(corrupt()),
        Ok(metadata) => metadata,
    };
    if metadata_is_link_or_reparse(&metadata)
        || !metadata.is_file()
        || metadata.len() > MAX_QUOTA_EVICTIONS_BYTES
    {
        return Err(corrupt());
    }
    let bytes = read_bounded_regular_file(&path, MAX_QUOTA_EVICTIONS_BYTES)?;
    let body = bytes
        .strip_prefix(QUOTA_EVICTIONS_HEADER)
        .ok_or_else(corrupt)?;
    if body.is_empty() || !body.ends_with(b"\n") {
        return Err(corrupt());
    }
    let body = std::str::from_utf8(body).map_err(|_| corrupt())?;
    let mut victims = Vec::new();
    let mut seen = HashSet::new();
    for state_id in body.lines() {
        validate_state_id(state_id).map_err(|_| corrupt())?;
        if !seen.insert(state_id.to_string()) {
            return Err(corrupt());
        }
        victims.push(state_id.to_string());
    }
    if victims.is_empty() {
        return Err(corrupt());
    }
    Ok(Some(victims))
}

fn summary_from_metadata(metadata: &StoredStateMetadata, pinned: bool) -> StateSummary {
    StateSummary {
        id: metadata.id.clone(),
        model_id: metadata.model_id.clone(),
        tier: StateTier::Disk,
        status: StateStatus::Ready,
        bytes: Some(metadata.payload_bytes),
        created_unix_ms: metadata.created_unix_ms,
        last_accessed_unix_ms: metadata.last_accessed_unix_ms,
        expires_unix_ms: metadata.expires_unix_ms,
        pinned,
        backend: metadata.backend.clone(),
        reusable: true,
    }
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
                .is_none_or(|model_id| state.model_id == model_id)
                && tier.is_none_or(|tier| state.tier == tier)
                && older_than_unix_ms.is_none_or(|cutoff| state.last_accessed_unix_ms < cutoff)
        }
        StateSelector::All { confirm } => *confirm,
    }
}

fn validate_selector(selector: &StateSelector, max_ids: usize) -> Result<(), ProtocolError> {
    match selector {
        StateSelector::Ids { ids } => {
            if ids.is_empty() || ids.len() > max_ids {
                return Err(invalid(format!(
                    "ids selector requires between 1 and {max_ids} state IDs"
                )));
            }
            let mut unique = HashSet::new();
            for id in ids {
                validate_state_id(id)?;
                if !unique.insert(id) {
                    return Err(invalid("ids selector contains a duplicate state ID"));
                }
            }
        }
        StateSelector::Filter {
            model_id,
            tier,
            older_than_unix_ms,
        } => {
            if model_id.is_none() && tier.is_none() && older_than_unix_ms.is_none() {
                return Err(invalid(
                    "filter selector requires at least one explicit restriction",
                ));
            }
            if let Some(model_id) = model_id {
                validate_model_id(model_id)?;
            }
        }
        StateSelector::All { confirm } if !confirm => {
            return Err(invalid("all selector requires confirm=true"));
        }
        StateSelector::All { .. } => {}
    }
    Ok(())
}

pub(crate) fn validate_filter(
    filter: &StateListFilter,
    max_page_size: u16,
) -> Result<(), ProtocolError> {
    if let Some(limit) = filter.limit
        && (limit == 0 || limit > max_page_size)
    {
        return Err(invalid(format!(
            "limit must be between 1 and {max_page_size}"
        )));
    }
    if let Some(model_id) = filter.model_id.as_deref() {
        validate_model_id(model_id)?;
    }
    if let Some(cursor) = filter.cursor.as_deref()
        && cursor.len() > 256
    {
        return Err(invalid("cursor is too long"));
    }
    Ok(())
}

fn validate_state_id(id: &str) -> Result<(), ProtocolError> {
    if !id.starts_with("st_") || !valid_opaque_id(id) || id.contains(['*', '?']) {
        return Err(invalid("invalid state ID"));
    }
    Ok(())
}

fn validate_model_id(id: &str) -> Result<(), ProtocolError> {
    if id.trim().is_empty()
        || id.len() > 256
        || id.chars().any(char::is_control)
        || id.contains("..")
    {
        return Err(invalid("invalid model ID"));
    }
    Ok(())
}

fn validate_compatibility(value: &CompatibilityEnvelope) -> Result<(), ProtocolError> {
    super::validate_compatibility_envelope(value)
}

fn random_id(prefix: &str, bytes: usize) -> Result<String, ProtocolError> {
    let mut random = vec![0u8; bytes];
    getrandom::getrandom(&mut random).map_err(|_| {
        ProtocolError::new(
            ProtocolErrorCode::Unavailable,
            "secure runtime state identity generation is unavailable",
        )
    })?;
    Ok(format!("{prefix}{}", URL_SAFE_NO_PAD.encode(random)))
}

#[cfg_attr(not(test), allow(dead_code))]
fn encode_cursor(id: &str) -> String {
    URL_SAFE_NO_PAD.encode(id.as_bytes())
}

#[cfg_attr(not(test), allow(dead_code))]
fn decode_cursor(cursor: &str) -> Result<String, ProtocolError> {
    let bytes = URL_SAFE_NO_PAD
        .decode(cursor)
        .map_err(|_| invalid("invalid cursor"))?;
    let id = String::from_utf8(bytes).map_err(|_| invalid("invalid cursor"))?;
    validate_state_id(&id)?;
    Ok(id)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn payload_source_len(source: &OpaquePayloadSource) -> Result<u64, ProtocolError> {
    match source {
        OpaquePayloadSource::Bytes(bytes) => Ok(bytes.len() as u64),
        OpaquePayloadSource::File { bytes, .. } => Ok(*bytes),
    }
}

fn write_payload_new(
    target: &Path,
    source: &OpaquePayloadSource,
    max_bytes: u64,
) -> Result<(u64, String), ProtocolError> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut output = options.open(target).map_err(|_| internal())?;
    let mut hasher = Sha256::new();
    let mut written = 0u64;
    match source {
        OpaquePayloadSource::Bytes(bytes) => {
            written = bytes.len() as u64;
            if written > max_bytes {
                return Err(resource("runtime state payload exceeds its size limit"));
            }
            output.write_all(bytes).map_err(|_| internal())?;
            hasher.update(bytes);
        }
        OpaquePayloadSource::File { file, bytes } => {
            if *bytes > max_bytes {
                return Err(resource("runtime state payload exceeds its size limit"));
            }
            let mut input = file.try_clone().map_err(|_| corrupt())?;
            input.seek(SeekFrom::Start(0)).map_err(|_| corrupt())?;
            let mut buffer = [0u8; 1024 * 1024];
            loop {
                let read = input.read(&mut buffer).map_err(|_| corrupt())?;
                if read == 0 {
                    break;
                }
                written = written.saturating_add(read as u64);
                if written > max_bytes {
                    return Err(resource("runtime state payload exceeds its size limit"));
                }
                output.write_all(&buffer[..read]).map_err(|_| internal())?;
                hasher.update(&buffer[..read]);
            }
            if written != *bytes {
                return Err(corrupt());
            }
        }
    }
    output.sync_all().map_err(|_| internal())?;
    Ok((written, format!("sha256:{:x}", hasher.finalize())))
}

fn sha256_regular_file(path: &Path, max_bytes: u64) -> Result<(u64, String), ProtocolError> {
    let (_, bytes, checksum) = open_and_sha256_regular_file(path, max_bytes)?;
    Ok((bytes, checksum))
}

fn open_and_sha256_regular_file(
    path: &Path,
    max_bytes: u64,
) -> Result<(File, u64, String), ProtocolError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| corrupt())?;
    if metadata_is_link_or_reparse(&metadata)
        || !metadata.is_file()
        || metadata.len() > max_bytes
        || !metadata_is_owned_private(&metadata)
    {
        return Err(corrupt());
    }
    let mut input = open_regular_file_without_following(path).map_err(|_| corrupt())?;
    let opened_metadata = input.metadata().map_err(|_| corrupt())?;
    if !opened_metadata.is_file()
        || opened_metadata.len() != metadata.len()
        || !metadata_is_owned_private(&opened_metadata)
    {
        return Err(corrupt());
    }
    let mut hasher = Sha256::new();
    let mut read_total = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = input.read(&mut buffer).map_err(|_| corrupt())?;
        if read == 0 {
            break;
        }
        read_total = read_total.saturating_add(read as u64);
        if read_total > max_bytes {
            return Err(corrupt());
        }
        hasher.update(&buffer[..read]);
    }
    if read_total != metadata.len() {
        return Err(corrupt());
    }
    input.seek(SeekFrom::Start(0)).map_err(|_| corrupt())?;
    Ok((input, read_total, format!("sha256:{:x}", hasher.finalize())))
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String, ProtocolError> {
    let bytes = serde_json::to_vec(value).map_err(|_| internal())?;
    Ok(sha256_bytes(&bytes))
}

fn read_bounded_regular_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>, ProtocolError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| corrupt())?;
    if metadata_is_link_or_reparse(&metadata)
        || !metadata.is_file()
        || metadata.len() > max_bytes
        || !metadata_is_owned_private(&metadata)
    {
        return Err(corrupt());
    }
    let file = open_regular_file_without_following(path).map_err(|_| corrupt())?;
    let opened_metadata = file.metadata().map_err(|_| corrupt())?;
    if !opened_metadata.is_file()
        || opened_metadata.len() != metadata.len()
        || !metadata_is_owned_private(&opened_metadata)
    {
        return Err(corrupt());
    }
    let mut bytes = Vec::with_capacity(metadata.len().min(1024 * 1024) as usize);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| corrupt())?;
    if bytes.len() as u64 > max_bytes {
        return Err(corrupt());
    }
    Ok(bytes)
}

fn open_regular_file_without_following(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    #[cfg(windows)]
    options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    let file = options.open(path)?;
    #[cfg(windows)]
    if file.metadata()?.file_attributes() & 0x0000_0400 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "reparse points are not regular runtime-state files",
        ));
    }
    Ok(file)
}

fn read_last_accessed(state_dir: &Path) -> Result<Option<u64>, ProtocolError> {
    let path = state_dir.join(LAST_ACCESSED_FILE);
    match fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(_) => Err(corrupt()),
        Ok(metadata) if metadata_is_link_or_reparse(&metadata) || !metadata.is_file() => {
            Err(corrupt())
        }
        Ok(_) => {
            let bytes = read_bounded_regular_file(&path, 21)?;
            let digits = bytes.strip_suffix(b"\n").ok_or_else(corrupt)?;
            if digits.is_empty()
                || !digits.iter().all(u8::is_ascii_digit)
                || (digits.len() > 1 && digits.first() == Some(&b'0'))
            {
                return Err(corrupt());
            }
            let value = std::str::from_utf8(digits)
                .map_err(|_| corrupt())?
                .parse::<u64>()
                .map_err(|_| corrupt())?;
            Ok(Some(value))
        }
    }
}

fn write_last_accessed(state_dir: &Path, value: u64) -> Result<(), ProtocolError> {
    let namespace = state_dir.parent().ok_or_else(internal)?;
    let staging = namespace.join(random_id(".staging-last-accessed-", 12)?);
    let target = state_dir.join(LAST_ACCESSED_FILE);
    let result: Result<(), ProtocolError> = (|| {
        write_new_file(&staging, format!("{value}\n").as_bytes())?;
        #[cfg(windows)]
        match fs::symlink_metadata(&target) {
            Ok(metadata) if metadata.is_dir() && !metadata_is_link_or_reparse(&metadata) => {
                return Err(corrupt());
            }
            Ok(_) => fs::remove_file(&target).map_err(|_| internal())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(internal()),
        }
        fs::rename(&staging, &target).map_err(|_| internal())?;
        sync_directory(state_dir)?;
        sync_directory(namespace)
    })();
    if result.is_err() {
        let _ = fs::remove_file(staging);
    }
    result
}

fn publish_new_state_marker(
    state_dir: &Path,
    name: &str,
    contents: &[u8],
) -> Result<(), ProtocolError> {
    let namespace = state_dir.parent().ok_or_else(internal)?;
    let staging = namespace.join(random_id(".staging-marker-", 12)?);
    let target = state_dir.join(name);
    let result = (|| {
        write_new_file(&staging, contents)?;
        fs::hard_link(&staging, &target).map_err(|_| internal())?;
        fs::remove_file(&staging).map_err(|_| internal())?;
        sync_directory(state_dir)?;
        sync_directory(namespace)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&staging);
    }
    result
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), ProtocolError> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(path).map_err(|_| internal())?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| internal())?;
    Ok(())
}

fn remove_regular_file(path: &Path) -> Result<(), ProtocolError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| corrupt())?;
    if metadata_is_link_or_reparse(&metadata) || !metadata.is_file() {
        return Err(corrupt());
    }
    fs::remove_file(path).map_err(|_| internal())
}

fn remove_state_dir(path: &Path) -> Result<(), ProtocolError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| corrupt())?;
    if metadata_is_link_or_reparse(&metadata) || !metadata.is_dir() {
        return Err(corrupt());
    }
    fs::remove_dir_all(path).map_err(|_| internal())
}

fn remove_entry_without_following(path: &Path) -> Result<(), ProtocolError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| corrupt())?;
    if metadata.is_dir() && !metadata_is_link_or_reparse(&metadata) {
        fs::remove_dir_all(path).map_err(|_| internal())
    } else {
        fs::remove_file(path).map_err(|_| internal())
    }
}

fn read_dir_bounded(path: &Path, max: usize) -> Result<Vec<fs::DirEntry>, ProtocolError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| internal())?;
    if metadata_is_link_or_reparse(&metadata) || !metadata.is_dir() {
        return Err(ProtocolError::new(
            ProtocolErrorCode::Forbidden,
            "runtime state directory is unsafe",
        ));
    }
    let mut entries = Vec::new();
    for entry in fs::read_dir(path).map_err(|_| internal())? {
        if entries.len() >= max {
            return Err(resource("runtime state catalog entry limit exceeded"));
        }
        entries.push(entry.map_err(|_| internal())?);
    }
    Ok(entries)
}

fn has_exact_child_name(parent: &Path, name: &str, max: usize) -> Result<bool, ProtocolError> {
    Ok(read_dir_bounded(parent, max)?
        .into_iter()
        .any(|entry| entry.file_name() == std::ffi::OsStr::new(name)))
}

fn metadata_is_link_or_reparse(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        metadata.file_attributes() & 0x0000_0400 != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn metadata_is_owned_private(metadata: &fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        metadata.uid() == unsafe { libc::geteuid() } && metadata.permissions().mode() & 0o077 == 0
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        true
    }
}

fn logical_regular_bytes_bounded(
    root: &Path,
    max_entries: usize,
    max_bytes: u64,
) -> Result<u64, ProtocolError> {
    if max_entries == 0 {
        return Err(resource("quarantine entry is structurally unbounded"));
    }
    let mut pending = vec![root.to_path_buf()];
    let mut discovered = 1usize;
    let mut bytes = 0u64;
    while let Some(path) = pending.pop() {
        let metadata = fs::symlink_metadata(&path).map_err(|_| corrupt())?;
        if metadata_is_link_or_reparse(&metadata) {
            continue;
        }
        if metadata.is_file() {
            bytes = bytes.saturating_add(metadata.len());
            if bytes > max_bytes {
                return Err(resource("quarantine entry exceeds its byte limit"));
            }
            continue;
        }
        if !metadata.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&path).map_err(|_| corrupt())? {
            if discovered >= max_entries {
                return Err(resource("quarantine entry is structurally unbounded"));
            }
            discovered = discovered.saturating_add(1);
            pending.push(entry.map_err(|_| corrupt())?.path());
        }
    }
    Ok(bytes)
}

fn ensure_owned_directory_tree(path: &Path) -> Result<(), ProtocolError> {
    let parent = path.parent().ok_or_else(internal)?;
    create_or_validate_owned_directory(parent)?;
    create_or_validate_owned_directory(path)
}

fn validate_existing_owned_directory(path: &Path) -> Result<bool, ProtocolError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(_) => return Err(internal()),
    };
    if metadata_is_link_or_reparse(&metadata) || !metadata.is_dir() {
        return Err(forbidden("runtime state directory is unsafe"));
    }
    #[cfg(unix)]
    if metadata.uid() != unsafe { libc::geteuid() } || metadata.permissions().mode() & 0o077 != 0 {
        return Err(forbidden("runtime state directory is not owner-only"));
    }
    Ok(true)
}

fn create_or_validate_owned_directory(path: &Path) -> Result<(), ProtocolError> {
    let mut created = false;
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata_is_link_or_reparse(&metadata) || !metadata.is_dir() => {
            return Err(ProtocolError::new(
                ProtocolErrorCode::Forbidden,
                "runtime state directory is unsafe",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match fs::create_dir(path) {
                Ok(()) => created = true,
                Err(create_error) if create_error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(_) => return Err(internal()),
            }
            let metadata = fs::symlink_metadata(path).map_err(|_| internal())?;
            if metadata_is_link_or_reparse(&metadata) || !metadata.is_dir() {
                return Err(ProtocolError::new(
                    ProtocolErrorCode::Forbidden,
                    "runtime state directory is unsafe",
                ));
            }
        }
        Err(_) => return Err(internal()),
    }
    #[cfg(unix)]
    {
        let metadata = fs::symlink_metadata(path).map_err(|_| internal())?;
        if metadata.uid() != unsafe { libc::geteuid() } {
            return Err(forbidden(
                "runtime state directory is not owned by the current user",
            ));
        }
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|_| internal())?;
    }
    if created {
        sync_directory(path.parent().ok_or_else(internal)?)?;
    }
    Ok(())
}

fn create_owned_directory(path: &Path) -> Result<(), ProtocolError> {
    fs::create_dir(path).map_err(|_| internal())?;
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|_| internal())?;
    sync_directory(path.parent().ok_or_else(internal)?)?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), ProtocolError> {
    #[cfg(unix)]
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|_| internal())?;
    let _ = path;
    Ok(())
}

fn invalid(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::InvalidRequest, message)
}

pub(crate) fn validate_principal_id(principal_id: &str) -> Result<(), ProtocolError> {
    if principal_id != "local"
        && (!principal_id.starts_with("p_")
            || principal_id.len() == 2
            || !valid_opaque_id(principal_id)
            || !principal_id.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
            }))
    {
        return Err(invalid("invalid principal namespace"));
    }
    Ok(())
}

fn not_found() -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::NotFound, "runtime state was not found")
}

fn expired_state() -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::NotFound, "runtime state has expired")
}

fn forbidden(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::Forbidden, message)
}

fn unavailable(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::Unavailable, message).retryable(true)
}

fn resource(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::ResourceExhausted, message)
}

fn corrupt() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::CorruptState,
        "runtime state failed integrity verification",
    )
}

fn internal() -> ProtocolError {
    ProtocolError::new(
        ProtocolErrorCode::Internal,
        "runtime state storage operation failed",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::werk_protocol::{ContextCompatibility, ProtocolVersion};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "werk-state-store-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn commit_is_namespaced_integrity_checked_and_listed_without_paths() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        let summary = store.commit("p_alice", new_state(b"secret kv")).unwrap();
        assert_eq!(summary.tier, StateTier::Disk);
        assert_eq!(summary.bytes, Some(9));
        assert!(summary.pinned);
        assert!(
            store
                .list("p_bob", &StateListFilter::default())
                .unwrap()
                .states
                .is_empty()
        );
        let listed = store.list("p_alice", &StateListFilter::default()).unwrap();
        assert_eq!(listed.states, [summary.clone()]);
        let json = serde_json::to_string(&listed).unwrap();
        assert!(!json.contains(temp.0.to_string_lossy().as_ref()));
        assert!(!json.contains("secret kv"));
        let mut loaded = store.load("p_alice", &summary.id).unwrap();
        assert_eq!(loaded.payload_bytes, 9);
        let mut bytes = Vec::new();
        loaded.payload_file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"secret kv");
    }

    #[test]
    fn verified_payload_handle_survives_path_replacement() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        let summary = store.commit("local", new_state(b"original")).unwrap();
        let mut loaded = store.inspect("local", &summary.id).unwrap();
        let debug = format!("{loaded:?}");
        assert!(!debug.contains(temp.0.to_string_lossy().as_ref()));
        assert!(!debug.contains("original"));
        let state_dir = temp.0.join("runtime-state/v1/local").join(&summary.id);
        let payload = state_dir.join(PAYLOAD_FILE);
        let displaced = state_dir.join("payload.displaced");

        fs::rename(&payload, &displaced).unwrap();
        fs::write(&payload, b"replaced").unwrap();

        let mut bytes = Vec::new();
        loaded.payload_file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"original");
        assert_eq!(fs::read(payload).unwrap(), b"replaced");
    }

    #[test]
    fn corruption_is_quarantined_and_never_returned() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        let summary = store.commit("local", new_state(b"payload")).unwrap();
        let payload = temp
            .0
            .join("runtime-state/v1/local")
            .join(&summary.id)
            .join(PAYLOAD_FILE);
        fs::write(payload, b"tampered").unwrap();
        let error = store.load("local", &summary.id).unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::CorruptState);
        assert!(
            store
                .list("local", &StateListFilter::default())
                .unwrap()
                .states
                .is_empty()
        );
    }

    #[test]
    fn invalid_persisted_metadata_is_corruption_and_is_quarantined() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        let summary = store.commit("local", new_state(b"payload")).unwrap();
        let state_dir = temp.0.join("runtime-state/v1/local").join(&summary.id);
        let metadata_path = state_dir.join(METADATA_FILE);
        let mut metadata: StoredStateMetadata =
            serde_json::from_slice(&fs::read(&metadata_path).unwrap()).unwrap();
        metadata.model_id = "../invalid".to_string();
        let metadata_bytes = serde_json::to_vec_pretty(&metadata).unwrap();
        fs::write(&metadata_path, &metadata_bytes).unwrap();
        fs::write(
            state_dir.join(METADATA_CHECKSUM_FILE),
            format!("{}\n", sha256_bytes(&metadata_bytes)),
        )
        .unwrap();

        let error = store.load("local", &summary.id).unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::CorruptState);
        assert!(!state_dir.exists());
        assert_eq!(
            fs::read_dir(temp.0.join("runtime-state/v1/.quarantine"))
                .unwrap()
                .count(),
            1
        );
    }

    #[test]
    fn startup_reconciliation_removes_staging_and_quarantines_unknown_entries() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        store.reconcile("local").unwrap();
        let namespace = temp.0.join("runtime-state/v1/local");
        fs::create_dir(namespace.join(".staging-abandoned")).unwrap();
        fs::write(namespace.join("not-a-state"), b"bad").unwrap();
        assert_eq!(store.reconcile("local").unwrap(), 2);
        assert!(!namespace.join(".staging-abandoned").exists());
        assert!(!namespace.join("not-a-state").exists());
    }

    #[test]
    fn read_only_and_dry_run_paths_do_not_initialize_a_missing_catalog() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        let root = temp.0.join("runtime-state");
        let missing_id = "st_missing";

        assert!(
            store
                .list("local", &StateListFilter::default())
                .unwrap()
                .states
                .is_empty()
        );
        assert!(store.all_summaries("local").unwrap().is_empty());
        assert_eq!(
            store.inspect("local", missing_id).unwrap_err().code,
            ProtocolErrorCode::NotFound
        );
        assert_eq!(
            store
                .set_pinned("local", missing_id, true, true)
                .unwrap_err()
                .code,
            ProtocolErrorCode::NotFound
        );
        let preview = store
            .prune(
                "local",
                &PruneStatesRequest {
                    selector: StateSelector::Ids {
                        ids: vec![missing_id.to_string()],
                    },
                    dry_run: true,
                },
            )
            .unwrap();
        assert_eq!((preview.matched, preview.removed), (0, 0));
        assert!(!root.exists());
    }

    #[test]
    fn path_like_principal_cannot_alias_the_catalog_root() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);

        let error = store.reconcile(".").unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::InvalidRequest);
        assert!(!temp.0.join("runtime-state").exists());
    }

    #[test]
    fn dry_runs_do_not_reconcile_or_change_existing_catalog_entries() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        let state = store.commit("local", new_state(b"payload")).unwrap();
        let namespace = temp.0.join("runtime-state/v1/local");
        let state_dir = namespace.join(&state.id);
        let staging = namespace.join(".staging-abandoned");
        let corrupt = namespace.join("st_corrupt");
        fs::create_dir(&staging).unwrap();
        fs::create_dir(&corrupt).unwrap();

        let preview = store
            .prune(
                "local",
                &PruneStatesRequest {
                    selector: StateSelector::All { confirm: true },
                    dry_run: true,
                },
            )
            .unwrap();
        let (projected, changed) = store.set_pinned("local", &state.id, false, true).unwrap();

        assert_eq!((preview.matched, preview.removed), (1, 0));
        assert!(changed);
        assert!(!projected.pinned);
        assert!(state_dir.join(PIN_FILE).is_file());
        assert!(!state_dir.join(LAST_ACCESSED_FILE).exists());
        assert!(staging.is_dir());
        assert!(corrupt.is_dir());
        assert!(!temp.0.join("runtime-state/v1/.quarantine").exists());
    }

    #[test]
    fn expired_pinned_state_is_hidden_read_only_and_removed_before_quota() {
        let temp = TestDir::new();
        let limits = StateStoreLimits {
            max_namespace_entries: 1,
            max_namespace_bytes: 100,
            ..StateStoreLimits::default()
        };
        let store = StateStore::with_limits(&temp.0, limits);
        let mut expired = new_state(b"expired");
        expired.expires_unix_ms = Some(now_unix_ms().saturating_sub(1));
        expired.pinned = true;
        let expired = store.commit("local", expired).unwrap();
        let expired_dir = temp.0.join("runtime-state/v1/local").join(&expired.id);

        assert!(
            store
                .list("local", &StateListFilter::default())
                .unwrap()
                .states
                .is_empty()
        );
        assert_eq!(
            store.inspect("local", &expired.id).unwrap_err().code,
            ProtocolErrorCode::NotFound
        );
        assert_eq!(
            store
                .set_pinned("local", &expired.id, false, true)
                .unwrap_err()
                .code,
            ProtocolErrorCode::NotFound
        );
        let preview = store
            .prune(
                "local",
                &PruneStatesRequest {
                    selector: StateSelector::All { confirm: true },
                    dry_run: true,
                },
            )
            .unwrap();
        assert_eq!((preview.matched, preview.removed), (0, 0));
        assert!(expired_dir.is_dir());

        let replacement = store.commit("local", new_state(b"live")).unwrap();
        assert!(!expired_dir.exists());
        assert!(
            temp.0
                .join("runtime-state/v1/local")
                .join(replacement.id)
                .is_dir()
        );
    }

    #[test]
    fn mutation_quarantines_corruption_introduced_after_first_use() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        let first = store.commit("local", new_state(b"payload")).unwrap();
        let first_dir = temp.0.join("runtime-state/v1/local").join(&first.id);
        fs::write(first_dir.join(PAYLOAD_FILE), b"PAYLOAD").unwrap();

        assert!(
            store
                .list("local", &StateListFilter::default())
                .unwrap()
                .states
                .is_empty()
        );
        let preview = store
            .prune(
                "local",
                &PruneStatesRequest {
                    selector: StateSelector::All { confirm: true },
                    dry_run: true,
                },
            )
            .unwrap();
        assert_eq!((preview.matched, preview.removed), (0, 0));
        assert_eq!(
            store.inspect("local", &first.id).unwrap_err().code,
            ProtocolErrorCode::CorruptState
        );
        assert!(first_dir.is_dir());
        assert!(!temp.0.join("runtime-state/v1/.quarantine").exists());

        let second = store.commit("local", new_state(b"second")).unwrap();

        assert!(!first_dir.exists());
        assert!(
            temp.0
                .join("runtime-state/v1/local")
                .join(second.id)
                .is_dir()
        );
        assert_eq!(
            fs::read_dir(temp.0.join("runtime-state/v1/.quarantine"))
                .unwrap()
                .count(),
            1
        );
    }

    #[test]
    fn quarantine_retention_is_bounded_by_entry_count_and_bytes() {
        let temp = TestDir::new();
        let limits = StateStoreLimits {
            max_quarantine_entries: 2,
            max_quarantine_bytes: DEFAULT_MAX_QUARANTINE_BYTES,
            ..StateStoreLimits::default()
        };
        let store = StateStore::with_limits(&temp.0, limits);
        let states = (0..3)
            .map(|_| store.commit("local", new_state(b"payload")).unwrap())
            .collect::<Vec<_>>();
        let namespace = temp.0.join("runtime-state/v1/local");
        for state in &states {
            fs::write(namespace.join(&state.id).join(PAYLOAD_FILE), b"PAYLOAD").unwrap();
        }

        assert_eq!(store.reconcile("local").unwrap(), 3);
        assert_eq!(
            fs::read_dir(temp.0.join("runtime-state/v1/.quarantine"))
                .unwrap()
                .count(),
            2
        );

        let quarantine = temp.0.join("runtime-state/v1/.quarantine");
        for index in 0..65 {
            fs::create_dir(quarantine.join(format!("q_manual_{index:03}"))).unwrap();
        }
        assert_eq!(fs::read_dir(&quarantine).unwrap().count(), 67);
        assert_eq!(store.reconcile("local").unwrap(), 0);
        assert_eq!(fs::read_dir(&quarantine).unwrap().count(), 0);

        let tiny = TestDir::new();
        let tiny_store = StateStore::with_limits(
            &tiny.0,
            StateStoreLimits {
                max_quarantine_entries: 2,
                max_quarantine_bytes: 1,
                ..StateStoreLimits::default()
            },
        );
        let state = tiny_store.commit("local", new_state(b"payload")).unwrap();
        let state_dir = tiny.0.join("runtime-state/v1/local").join(state.id);
        fs::write(state_dir.join(PAYLOAD_FILE), b"PAYLOAD").unwrap();
        assert_eq!(tiny_store.reconcile("local").unwrap(), 1);
        assert!(!state_dir.exists());
        assert!(!tiny.0.join("runtime-state/v1/.quarantine").exists());
    }

    #[test]
    fn prune_requires_explicit_constrained_selector_and_defaults_to_dry_run() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        let state = store.commit("local", new_state(b"payload")).unwrap();
        let empty_filter = PruneStatesRequest {
            selector: StateSelector::Filter {
                model_id: None,
                tier: None,
                older_than_unix_ms: None,
            },
            dry_run: true,
        };
        assert!(store.prune("local", &empty_filter).is_err());
        let request: PruneStatesRequest = serde_json::from_str(&format!(
            r#"{{"selector":{{"kind":"ids","ids":["{}"]}}}}"#,
            state.id
        ))
        .unwrap();
        let dry = store.prune("local", &request).unwrap();
        assert_eq!((dry.matched, dry.removed, dry.dry_run), (1, 0, true));
        let real = store
            .prune(
                "local",
                &PruneStatesRequest {
                    dry_run: false,
                    ..request
                },
            )
            .unwrap();
        assert_eq!((real.matched, real.removed), (1, 1));
    }

    #[test]
    fn quota_evicts_oldest_unpinned_but_never_pinned_state() {
        let temp = TestDir::new();
        let limits = StateStoreLimits {
            max_namespace_entries: 2,
            max_namespace_bytes: 100,
            ..StateStoreLimits::default()
        };
        let store = StateStore::with_limits(&temp.0, limits);
        let pinned = store.commit("local", new_state(b"one")).unwrap();
        let mut second = new_state(b"two");
        second.pinned = false;
        let second = store.commit("local", second).unwrap();
        let mut third = new_state(b"three");
        third.pinned = false;
        let third = store.commit("local", third).unwrap();
        let ids = store
            .list("local", &StateListFilter::default())
            .unwrap()
            .states
            .into_iter()
            .map(|state| state.id)
            .collect::<HashSet<_>>();
        assert!(ids.contains(&pinned.id));
        assert!(!ids.contains(&second.id));
        assert!(ids.contains(&third.id));
    }

    #[test]
    fn published_quota_commit_is_recovered_without_precommit_eviction() {
        let temp = TestDir::new();
        let limits = StateStoreLimits {
            max_namespace_entries: 2,
            max_namespace_bytes: 100,
            ..StateStoreLimits::default()
        };
        let store = StateStore::with_limits(&temp.0, limits.clone());
        let mut evictable = new_state(b"first");
        evictable.pinned = false;
        let evictable = store
            .commit_with_id("local", "st_first", evictable)
            .unwrap();
        let retained = store
            .commit_with_id("local", "st_second", new_state(b"second"))
            .unwrap();

        let committed = store
            .commit_inner_with_cleanup("local", Some("st_third"), new_state(b"third"), false)
            .unwrap();
        let namespace = temp.0.join("runtime-state/v1/local");

        assert!(namespace.join(&evictable.id).is_dir());
        assert!(namespace.join(&retained.id).is_dir());
        assert!(namespace.join(&committed.id).is_dir());
        assert_eq!(
            fs::read(namespace.join(&committed.id).join(QUOTA_EVICTIONS_FILE)).unwrap(),
            encode_quota_evictions(std::slice::from_ref(&evictable.id)).unwrap()
        );

        drop(store);
        let recovered = StateStore::with_limits(&temp.0, limits);
        assert_eq!(recovered.reconcile("local").unwrap(), 1);
        assert!(!namespace.join(&evictable.id).exists());
        assert!(namespace.join(&retained.id).is_dir());
        assert!(namespace.join(&committed.id).is_dir());
        assert!(
            !namespace
                .join(&committed.id)
                .join(QUOTA_EVICTIONS_FILE)
                .exists()
        );
    }

    #[test]
    fn corrupt_quota_journal_owner_never_evicts_valid_states() {
        for corruption in ["payload", "journal"] {
            let temp = TestDir::new();
            let limits = StateStoreLimits {
                max_namespace_entries: 2,
                max_namespace_bytes: 100,
                ..StateStoreLimits::default()
            };
            let store = StateStore::with_limits(&temp.0, limits.clone());
            let mut evictable = new_state(b"first");
            evictable.pinned = false;
            let evictable = store
                .commit_with_id("local", "st_first", evictable)
                .unwrap();
            let retained = store
                .commit_with_id("local", "st_second", new_state(b"second"))
                .unwrap();
            let committed = store
                .commit_inner_with_cleanup("local", Some("st_third"), new_state(b"third"), false)
                .unwrap();
            let namespace = temp.0.join("runtime-state/v1/local");
            let committed_dir = namespace.join(&committed.id);

            match corruption {
                "payload" => fs::write(committed_dir.join(PAYLOAD_FILE), b"THIRD").unwrap(),
                "journal" => fs::write(
                    committed_dir.join(QUOTA_EVICTIONS_FILE),
                    encode_quota_evictions(std::slice::from_ref(&retained.id)).unwrap(),
                )
                .unwrap(),
                _ => unreachable!(),
            }

            drop(store);
            let recovered = StateStore::with_limits(&temp.0, limits);
            assert_eq!(recovered.reconcile("local").unwrap(), 1);
            assert!(namespace.join(&evictable.id).is_dir());
            assert!(namespace.join(&retained.id).is_dir());
            assert!(!committed_dir.exists());
        }
    }

    #[cfg(unix)]
    #[test]
    fn state_directory_and_persistence_files_must_remain_owner_only() {
        for permissive_target in ["state-directory", "payload"] {
            let temp = TestDir::new();
            let store = StateStore::new(&temp.0);
            let state = store
                .commit_with_id("local", "st_permissions", new_state(b"secret"))
                .unwrap();
            let state_dir = temp.0.join("runtime-state/v1/local").join(&state.id);
            let target = if permissive_target == "state-directory" {
                state_dir.clone()
            } else {
                state_dir.join(PAYLOAD_FILE)
            };
            fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();

            assert_eq!(
                store.load("local", &state.id).unwrap_err().code,
                ProtocolErrorCode::CorruptState
            );
            assert!(
                !state_dir.exists(),
                "unsafe {permissive_target} remained active"
            );
        }
    }

    #[test]
    fn state_children_and_fixed_markers_are_exact_and_bounded() {
        for (name, corrupt_child, contents) in [
            ("unknown-child", "unexpected", b"data".as_slice()),
            ("pin-marker", PIN_FILE, b"yes\n".as_slice()),
            (
                "quota-marker",
                QUOTA_EVICTIONS_FILE,
                b"unexpected\n".as_slice(),
            ),
        ] {
            let temp = TestDir::new();
            let store = StateStore::new(&temp.0);
            let state = store
                .commit_with_id("local", &format!("st_{name}"), new_state(b"payload"))
                .unwrap();
            let state_dir = temp.0.join("runtime-state/v1/local").join(&state.id);
            fs::write(state_dir.join(corrupt_child), contents).unwrap();

            assert_eq!(
                store.inspect("local", &state.id).unwrap_err().code,
                ProtocolErrorCode::CorruptState
            );
            assert_eq!(
                store.load("local", &state.id).unwrap_err().code,
                ProtocolErrorCode::CorruptState
            );
            assert!(!state_dir.exists());
        }

        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        let state = store
            .commit_with_id("local", "st_too_many_children", new_state(b"payload"))
            .unwrap();
        let state_dir = temp.0.join("runtime-state/v1/local").join(&state.id);
        for index in 0..=MAX_STATE_DIRECTORY_ENTRIES {
            fs::write(state_dir.join(format!("unexpected-{index}")), b"x").unwrap();
        }
        assert_eq!(
            store.inspect("local", &state.id).unwrap_err().code,
            ProtocolErrorCode::CorruptState
        );
    }

    #[test]
    fn namespace_payload_scans_have_one_commit_of_aggregate_headroom() {
        let temp = TestDir::new();
        let roomy = StateStore::with_limits(
            &temp.0,
            StateStoreLimits {
                max_payload_bytes: 4,
                max_namespace_bytes: 100,
                max_namespace_entries: 10,
                ..StateStoreLimits::default()
            },
        );
        for id in ["st_one", "st_two", "st_three"] {
            roomy
                .commit_with_id("local", id, new_state(b"1234"))
                .unwrap();
        }
        drop(roomy);

        let bounded = StateStore::with_limits(
            &temp.0,
            StateStoreLimits {
                max_payload_bytes: 4,
                max_namespace_bytes: 4,
                max_namespace_entries: 10,
                ..StateStoreLimits::default()
            },
        );
        let error = bounded
            .list("local", &StateListFilter::default())
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);
        assert!(error.message.contains("aggregate payload scan limit"));
    }

    #[test]
    fn namespace_scan_limits_saturate_instead_of_wrapping() {
        let temp = TestDir::new();
        let store = StateStore::with_limits(
            &temp.0,
            StateStoreLimits {
                max_payload_bytes: 2,
                max_namespace_bytes: u64::MAX - 1,
                max_namespace_entries: usize::MAX,
                ..StateStoreLimits::default()
            },
        );

        assert_eq!(store.namespace_entry_scan_limit(), usize::MAX);
        assert_eq!(store.payload_scan_budget().remaining_bytes, u64::MAX);
    }

    #[test]
    fn unsatisfiable_quota_does_not_partially_evict_valid_states() {
        let temp = TestDir::new();
        let limits = StateStoreLimits {
            max_namespace_entries: 10,
            max_namespace_bytes: 8,
            ..StateStoreLimits::default()
        };
        let store = StateStore::with_limits(&temp.0, limits);
        let pinned = store.commit("local", new_state(b"1234567")).unwrap();
        let mut evictable = new_state(b"x");
        evictable.pinned = false;
        let evictable = store.commit("local", evictable).unwrap();

        let error = store.commit("local", new_state(b"yy")).unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);
        let ids = store
            .list("local", &StateListFilter::default())
            .unwrap()
            .states
            .into_iter()
            .map(|state| state.id)
            .collect::<HashSet<_>>();
        assert_eq!(ids, HashSet::from([pinned.id, evictable.id]));
    }

    #[test]
    fn successful_load_records_access_and_lru_quota_preserves_recent_state() {
        let temp = TestDir::new();
        let limits = StateStoreLimits {
            max_namespace_entries: 2,
            max_namespace_bytes: 100,
            ..StateStoreLimits::default()
        };
        let store = StateStore::with_limits(&temp.0, limits);
        let mut first = new_state(b"one");
        first.pinned = false;
        let first = store.commit("local", first).unwrap();
        let mut second = new_state(b"two");
        second.pinned = false;
        let second = store.commit("local", second).unwrap();
        let first_dir = temp.0.join("runtime-state/v1/local").join(&first.id);

        let loaded = store.load("local", &first.id).unwrap();
        assert!(first_dir.join(LAST_ACCESSED_FILE).is_file());
        assert!(loaded.summary.last_accessed_unix_ms >= first.last_accessed_unix_ms);
        write_last_accessed(&first_dir, u64::MAX - 1).unwrap();

        let mut third = new_state(b"three");
        third.pinned = false;
        let third = store.commit("local", third).unwrap();
        let ids = store
            .list("local", &StateListFilter::default())
            .unwrap()
            .states
            .into_iter()
            .map(|state| state.id)
            .collect::<HashSet<_>>();
        assert!(ids.contains(&first.id));
        assert!(!ids.contains(&second.id));
        assert!(ids.contains(&third.id));
    }

    #[test]
    fn inspect_verifies_state_without_recording_access() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        let state = store.commit("local", new_state(b"payload")).unwrap();
        let state_dir = temp.0.join("runtime-state/v1/local").join(&state.id);

        let inspected = store.inspect("local", &state.id).unwrap();

        assert_eq!(
            inspected.summary.last_accessed_unix_ms,
            state.last_accessed_unix_ms
        );
        assert!(!state_dir.join(LAST_ACCESSED_FILE).exists());
    }

    #[test]
    fn corrupt_access_marker_is_quarantined_on_load() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        let state = store.commit("local", new_state(b"payload")).unwrap();
        let state_dir = temp.0.join("runtime-state/v1/local").join(&state.id);
        fs::write(state_dir.join(LAST_ACCESSED_FILE), b"not-a-timestamp").unwrap();

        let error = store.load("local", &state.id).unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::CorruptState);
        assert!(!state_dir.exists());
    }

    #[test]
    fn externally_locked_catalog_fails_fast_and_retryably() {
        let temp = TestDir::new();
        let store = StateStore::new(&temp.0);
        store.reconcile("local").unwrap();
        let lock_path = temp.0.join("runtime-state/v1/.lock");
        let external = OpenOptions::new()
            .read(true)
            .write(true)
            .open(lock_path)
            .unwrap();
        FileExt::lock_exclusive(&external).unwrap();

        let error = store
            .list("local", &StateListFilter::default())
            .unwrap_err();

        assert_eq!(error.code, ProtocolErrorCode::Unavailable);
        assert!(error.retryable);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_runtime_root_is_rejected() {
        use std::os::unix::fs::symlink;
        let temp = TestDir::new();
        let outside = TestDir::new();
        fs::create_dir(temp.0.join("runtime-state")).unwrap();
        symlink(&outside.0, temp.0.join("runtime-state/v1")).unwrap();
        let store = StateStore::new(&temp.0);
        assert_eq!(
            store
                .list("local", &StateListFilter::default())
                .unwrap_err()
                .code,
            ProtocolErrorCode::Forbidden
        );
    }

    fn new_state(payload: &[u8]) -> NewStoredState {
        NewStoredState {
            model_id: "model".to_string(),
            backend: "test".to_string(),
            compatibility: CompatibilityEnvelope {
                model_fingerprint: "sha256:model".to_string(),
                tokenizer_fingerprint: "sha256:tokenizer".to_string(),
                prompt_fingerprint: "hmac-sha256:prompt".to_string(),
                chat_template_fingerprint: Some("sha256:template".to_string()),
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
                    context_size: 4096,
                    batch_size: None,
                    rope_configuration_fingerprint: None,
                },
                multimodal_processor_fingerprints: Vec::new(),
                producer_protocol: ProtocolVersion::V1,
            },
            payload: OpaquePayloadSource::Bytes(Arc::from(payload)),
            prompt_tokens: 3,
            expires_unix_ms: None,
            pinned: true,
        }
    }
}
