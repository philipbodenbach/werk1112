//! Narrow, fail-closed runtime-state integration for a concrete llama.cpp server process.
//!
//! Nothing in this module starts or loads a model. Capabilities become
//! experimental only after the already-running process passes the exact slot
//! save/erase/restore/replay probe below.

use super::{
    DEFAULT_CTX_SIZE, LlamaServerBackend, LlamaServerProcess, SupportedArgs, discover_llama_server,
    label, parse_local_url,
};
use crate::{
    backend::{LlamaCppMode, LlamaRuntimeOptions},
    model_store::{ModelFormat, ModelManifest, ModelRuntimeIdentity, ModelStore},
    runtime_control::{
        BackendDecodeOptions, BackendDecodeRequest, BackendDecodeResult,
        BackendPersistedStateResolution, BackendPersistedStateScope, BackendPrefillRequest,
        BackendPrefillResult, BackendRuntimeAdapter, BackendRuntimeDescriptor, BackendSnapshot,
        BackendState, ModelResidencyStatus, model_residency_capability, validate_compatibility,
    },
    werk_protocol::{
        Capability, CapabilityStatus, CompatibilityEnvelope, ContextCompatibility, PersistenceMode,
        PrefillInput, ProtocolError, ProtocolErrorCode, ProtocolVersion, StateTier,
    },
};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    env, fs,
    fs::OpenOptions,
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
    net::{IpAddr, SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex, MutexGuard, TryLockError},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const STATE_HTTP_TIMEOUT: Duration = Duration::from_secs(180);
const STATE_HTTP_MAX_REQUEST_BYTES: usize = 1024 * 1024;
const STATE_HTTP_MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const STATE_SNAPSHOT_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const STATE_SLOT_ID: u32 = 0;
const MAX_CANONICAL_STATE_RECORDS: usize = 1024;
const MAX_DEFERRED_CLEANUPS: usize = MAX_CANONICAL_STATE_RECORDS;
const STATE_CAPABILITY_PROBE_PROMPT: &str =
    "Werk llama.cpp runtime-state capability probe. Evaluate this exact private prompt.";

#[derive(Clone, PartialEq, Eq)]
struct LlamaExecutableIdentity {
    version: String,
    binary_sha256: String,
    help_sha256: String,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct LlamaProcessIdentity {
    executable: LlamaExecutableIdentity,
    args_sha256: String,
}

pub(super) struct LlamaProcessStateRuntime {
    pub(super) generation_id: Option<String>,
    pub(super) snapshot_dir: Option<PathBuf>,
    pub(super) identity: Option<LlamaProcessIdentity>,
    pub(super) configured: bool,
}

pub(super) struct LlamaRuntimeStateAdapter {
    backend: LlamaServerBackend,
    unavailable_instance_id: String,
    discovered_identity: Option<LlamaExecutableIdentity>,
    state: Mutex<LlamaAdapterState>,
    deferred_cleanup: Mutex<DeferredCleanupQueue>,
}

#[derive(Default)]
struct LlamaAdapterState {
    active: Option<LlamaActiveRuntime>,
    failed_generations: HashSet<String>,
    records: HashMap<String, LlamaStateRecord>,
    /// Principal-scoped prompt fingerprints point only at opaque context IDs.
    /// Canonical input is held separately as backend execution state and is
    /// never embedded in this reuse index.
    retained_contexts: HashMap<String, RetainedContextIndex>,
    decode_contexts: HashMap<String, LlamaDecodeContext>,
    exported_snapshots: HashSet<PathBuf>,
    access_clock: u64,
}

struct LlamaActiveRuntime {
    server: Arc<LlamaServerProcess>,
    validated: bool,
}

#[derive(Clone)]
struct LlamaStateRecord {
    server: Arc<LlamaServerProcess>,
    snapshot_name: String,
    context_id: String,
}

#[derive(Clone)]
struct LlamaDecodeContext {
    server: Arc<LlamaServerProcess>,
    input: PrefillInput,
    compatibility: CompatibilityEnvelope,
    prompt_tokens: u64,
}

#[derive(Clone)]
struct RetainedContextIndex {
    context_id: String,
    last_access: u64,
    expires_unix_ms: Option<u64>,
}

#[derive(Clone, PartialEq, Eq)]
enum DeferredCleanup {
    InProcess { handle: String, instance_id: String },
    OpaqueFile { path: PathBuf },
}

#[derive(Default)]
struct DeferredCleanupQueue {
    entries: VecDeque<DeferredCleanup>,
}

impl LlamaRuntimeStateAdapter {
    pub(super) fn new(backend: LlamaServerBackend) -> Self {
        let discovered_identity = discover_llama_server(&backend.store, backend.mode)
            .path
            .as_deref()
            .and_then(|path| probe_llama_executable_identity(path).ok());
        Self {
            backend,
            unavailable_instance_id: random_private_id("unavailable_", 16)
                .unwrap_or_else(|_| "unavailable".to_string()),
            discovered_identity,
            state: Mutex::new(LlamaAdapterState::default()),
            deferred_cleanup: Mutex::new(DeferredCleanupQueue::default()),
        }
    }

    fn lock_state(&self) -> std::result::Result<MutexGuard<'_, LlamaAdapterState>, ProtocolError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| protocol_internal("llama.cpp runtime-state adapter is unavailable"))?;
        let mut deferred = self
            .deferred_cleanup
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        drain_deferred_cleanup(&mut state, &mut deferred);
        purge_expired_retained_contexts_at(&mut state, current_unix_ms());
        Ok(state)
    }

    fn existing_server(
        &self,
        manifest: &ModelManifest,
    ) -> std::result::Result<Arc<LlamaServerProcess>, ProtocolError> {
        if manifest.format != ModelFormat::Gguf {
            return Err(protocol_unsupported(
                "llama.cpp runtime state is available only for GGUF models",
            ));
        }
        let model_identity = ModelRuntimeIdentity::from_manifest(manifest).map_err(|_| {
            protocol_internal("llama.cpp runtime could not identify the requested model manifest")
        })?;
        let servers = self
            .backend
            .servers
            .lock()
            .map_err(|_| protocol_internal("llama.cpp runtime registry is unavailable"))?;
        let mut candidates = servers
            .values()
            .filter(|server| server.model_identity == model_identity)
            .cloned()
            .collect::<Vec<_>>();
        drop(servers);
        candidates.sort_by_key(|server| std::cmp::Reverse(server.pid));
        candidates
            .into_iter()
            .find(|server| server.is_running())
            .ok_or_else(|| {
                protocol_unavailable(
                    "runtime state requires an already-running llama.cpp model process",
                )
            })
    }

    fn activate_and_probe(
        &self,
        manifest: &ModelManifest,
    ) -> std::result::Result<Arc<LlamaServerProcess>, ProtocolError> {
        let server = self.existing_server(manifest)?;
        let generation_id = server
            .state_runtime
            .generation_id
            .as_deref()
            .ok_or_else(|| protocol_unavailable(LLAMA_STATE_PREREQUISITES_DETAIL))?;
        if !server.state_runtime.configured || server.state_runtime.identity.is_none() {
            return Err(protocol_unavailable(LLAMA_STATE_PREREQUISITES_DETAIL));
        }

        let mut state = self.lock_state()?;
        let same_generation = state.active.as_ref().is_some_and(|active| {
            active.server.state_runtime.generation_id.as_deref() == Some(generation_id)
        });
        if !same_generation {
            cleanup_adapter_state(&mut state);
            state.active = Some(LlamaActiveRuntime {
                server: server.clone(),
                validated: false,
            });
        }
        if state.failed_generations.contains(generation_id) {
            return Err(protocol_unavailable(
                "the active llama.cpp process failed runtime-state validation",
            ));
        }
        if !state.active.as_ref().is_some_and(|active| active.validated) {
            let probe_result = server
                .state_gate
                .lock()
                .map_err(|_| protocol_internal("llama.cpp state operation gate is unavailable"))
                .and_then(|_operation| {
                    functional_probe_llama_state(&server).map_err(|_| {
                        protocol_unavailable(
                            "the active llama.cpp process failed runtime-state validation",
                        )
                    })
                });
            if let Err(error) = probe_result {
                state.failed_generations.insert(generation_id.to_string());
                if let Some(active) = state.active.as_mut() {
                    active.validated = false;
                }
                return Err(error);
            }
            if let Some(active) = state.active.as_mut() {
                active.validated = true;
            }
        }
        Ok(server)
    }

    fn active_validated_server(
        &self,
        state: &LlamaAdapterState,
    ) -> std::result::Result<Arc<LlamaServerProcess>, ProtocolError> {
        let active = state.active.as_ref().ok_or_else(|| {
            protocol_unavailable("no functionally validated llama.cpp model process is active")
        })?;
        if !active.validated || !active.server.is_running() {
            return Err(protocol_unavailable(
                "the validated llama.cpp model process is no longer available",
            ));
        }
        Ok(active.server.clone())
    }
}

const LLAMA_STATE_PREREQUISITES_DETAIL: &str = "the exact llama.cpp binary/process does not provide a private, single-slot, functionally validated state interface";

impl BackendRuntimeAdapter for LlamaRuntimeStateAdapter {
    fn descriptor(&self) -> BackendRuntimeDescriptor {
        let Ok(mut state) = self.lock_state() else {
            return unavailable_llama_descriptor(
                self.backend.mode,
                self.discovered_identity.as_ref(),
                self.unavailable_instance_id.clone(),
                "llama.cpp runtime-state adapter synchronization failed",
            );
        };
        let dead = state
            .active
            .as_ref()
            .is_some_and(|active| !active.server.is_running());
        if dead {
            cleanup_adapter_state(&mut state);
            state.active = None;
        }
        let Some(active) = state.active.as_ref().filter(|active| active.validated) else {
            return unavailable_llama_descriptor(
                self.backend.mode,
                self.discovered_identity.as_ref(),
                self.unavailable_instance_id.clone(),
                "runtime state requires an already-running, functionally validated llama.cpp model process",
            );
        };
        let Some(identity) = active.server.state_runtime.identity.as_ref() else {
            return unavailable_llama_descriptor(
                self.backend.mode,
                self.discovered_identity.as_ref(),
                self.unavailable_instance_id.clone(),
                "the active llama.cpp process has no exact runtime identity",
            );
        };
        let Some(instance_id) = active.server.state_runtime.generation_id.clone() else {
            return unavailable_llama_descriptor(
                self.backend.mode,
                self.discovered_identity.as_ref(),
                self.unavailable_instance_id.clone(),
                "the active llama.cpp process has no generation identity",
            );
        };
        BackendRuntimeDescriptor {
            backend: label(self.backend.mode).to_string(),
            backend_version: llama_process_version(identity),
            adapter_version: env!("CARGO_PKG_VERSION").to_string(),
            accelerator_family: llama_accelerator_family(self.backend.mode).to_string(),
            instance_id,
            capabilities: llama_runtime_capabilities(
                llama_state_capabilities(true, ""),
                ModelResidencyStatus::Supported,
                "Werk keeps this exact llama.cpp model process resident and enables automatic prompt-cache reuse",
            ),
        }
    }

    fn compatibility(
        &self,
        manifest: &ModelManifest,
        prompt_fingerprint: &str,
    ) -> std::result::Result<CompatibilityEnvelope, ProtocolError> {
        if prompt_fingerprint.is_empty() || prompt_fingerprint.len() > 256 {
            return Err(protocol_incompatible(
                "runtime-state prompt fingerprint is invalid",
            ));
        }
        let server = self.activate_and_probe(manifest)?;
        build_llama_compatibility(
            manifest,
            &server,
            &self.backend.runtime_options,
            prompt_fingerprint,
        )
    }

    fn resolve_persisted_state(
        &self,
        manifest: &ModelManifest,
        _snapshot: &BackendSnapshot,
        expected: &CompatibilityEnvelope,
    ) -> std::result::Result<BackendPersistedStateResolution, ProtocolError> {
        let retained_context_id = {
            let state = self.lock_state()?;
            let server = self.active_validated_server(&state)?;
            let retained = state
                .retained_contexts
                .get(&expected.prompt_fingerprint)
                .ok_or_else(|| {
                    protocol_unavailable(
                        "llama.cpp snapshots can be resolved only by their original live process",
                    )
                })?;
            let context = state
                .decode_contexts
                .get(&retained.context_id)
                .ok_or_else(|| {
                    protocol_unavailable("llama.cpp snapshot decode context is no longer retained")
                })?;
            if server_generation_id(&server)? != server_generation_id(&context.server)? {
                return Err(protocol_unavailable(
                    "llama.cpp snapshot ownership ended with its original process",
                ));
            }
            validate_compatibility(expected, &context.compatibility)?;
            retained.context_id.clone()
        };

        let compatibility = self.compatibility(manifest, &expected.prompt_fingerprint)?;
        // Capability probing can take long enough for a named retention policy
        // to expire. Re-check after it, so resolution never reports a context
        // that is no longer eligible for named reuse.
        let state = self.lock_state()?;
        if state
            .retained_contexts
            .get(&expected.prompt_fingerprint)
            .is_none_or(|retained| retained.context_id != retained_context_id)
        {
            return Err(protocol_unavailable(
                "llama.cpp snapshot decode context is no longer retained",
            ));
        }
        Ok(BackendPersistedStateResolution {
            compatibility,
            scope: BackendPersistedStateScope::SameProcess,
        })
    }

    fn validate_state(
        &self,
        backend_state: &BackendState,
        compatibility: &CompatibilityEnvelope,
    ) -> std::result::Result<(), ProtocolError> {
        let state = self.lock_state()?;
        let server = self.active_validated_server(&state)?;
        let (handle, instance_id) = in_process_handle(backend_state)?;
        if instance_id != server_generation_id(&server)? {
            return Err(protocol_incompatible(
                "llama.cpp state belongs to another process generation",
            )
            .with_details(json!({ "mismatch_fields": ["process_instance"] })));
        }
        let record = state
            .records
            .get(handle)
            .ok_or_else(|| protocol_incompatible("llama.cpp state handle is invalid or expired"))?;
        let context = state
            .decode_contexts
            .get(&record.context_id)
            .ok_or_else(|| {
                protocol_incompatible("llama.cpp state decode context is unavailable")
            })?;
        if server_generation_id(&record.server)? != instance_id || !record.server.is_running() {
            return Err(protocol_incompatible(
                "llama.cpp state owner is no longer the active process generation",
            )
            .with_details(json!({ "mismatch_fields": ["process_instance"] })));
        }
        validate_compatibility(compatibility, &context.compatibility)
    }

    fn prefill(
        &self,
        request: BackendPrefillRequest,
    ) -> std::result::Result<BackendPrefillResult, ProtocolError> {
        let mut state = self.lock_state()?;
        if state.records.len() >= MAX_CANONICAL_STATE_RECORDS {
            return Err(protocol_resource_exhausted(
                "the llama.cpp runtime-state handle limit has been reached",
            ));
        }
        let server = self.active_validated_server(&state)?;
        validate_request_generation(&server, &request.model_id, &request.compatibility)?;
        let handle = random_private_id("lh_", 24)
            .map_err(|_| protocol_unavailable("secure state allocation is unavailable"))?;
        let context_id = random_private_id("lc_", 24)
            .map_err(|_| protocol_unavailable("secure decode context allocation is unavailable"))?;
        let snapshot_name = random_private_id("state_", 24)
            .map(|value| format!("{value}.bin"))
            .map_err(|_| protocol_unavailable("secure state allocation is unavailable"))?;
        let _operation = server
            .state_gate
            .lock()
            .map_err(|_| protocol_internal("llama.cpp state operation gate is unavailable"))?;
        erase_slot_best_effort(&server);
        let prompt_tokens = match run_llama_prefill(&server, &request.input) {
            Ok(prompt_tokens) => prompt_tokens,
            Err(_) => {
                erase_slot_best_effort(&server);
                return Err(protocol_unavailable("llama.cpp prefill validation failed"));
            }
        };
        let snapshot = match save_llama_slot(&server, &snapshot_name, prompt_tokens) {
            Ok(snapshot) => snapshot,
            Err(_) => {
                erase_slot_best_effort(&server);
                remove_private_snapshot(&server, &snapshot_name);
                return Err(protocol_unavailable(
                    "llama.cpp state snapshot validation failed",
                ));
            }
        };
        if erase_llama_slot(&server, Some(prompt_tokens)).is_err() {
            erase_slot_best_effort(&server);
            remove_private_snapshot(&server, &snapshot_name);
            return Err(protocol_unavailable(
                "llama.cpp slot cleanup failed after prefill",
            ));
        }
        state.access_clock = state.access_clock.saturating_add(1);
        let access = state.access_clock;
        let expires_unix_ms =
            retained_context_expiration(current_unix_ms(), request.policy.ttl_seconds);
        let context = LlamaDecodeContext {
            server: server.clone(),
            input: request.input.clone(),
            compatibility: request.compatibility.clone(),
            prompt_tokens,
        };
        if state.decode_contexts.contains_key(&context_id) {
            remove_private_snapshot(&server, &snapshot_name);
            return Err(protocol_unavailable(
                "secure decode context allocation collided",
            ));
        }
        state.decode_contexts.insert(context_id.clone(), context);
        let record = LlamaStateRecord {
            server: server.clone(),
            snapshot_name: snapshot_name.clone(),
            context_id: context_id.clone(),
        };
        state.records.insert(handle.clone(), record);
        if matches!(
            request.policy.mode,
            PersistenceMode::Disk | PersistenceMode::Auto
        ) {
            let replaced = state.retained_contexts.insert(
                request.compatibility.prompt_fingerprint.clone(),
                RetainedContextIndex {
                    context_id,
                    last_access: access,
                    expires_unix_ms,
                },
            );
            if let Some(replaced) = replaced {
                remove_decode_context_if_unused(&mut state, &replaced.context_id);
            }
            trim_retained_contexts(&mut state);
        }
        Ok(BackendPrefillResult {
            state: BackendState::InProcess {
                handle,
                bytes: Some(snapshot.bytes),
                tier: StateTier::Disk,
                instance_id: server_generation_id(&server)?.to_string(),
            },
            prompt_tokens,
            reused: false,
        })
    }

    fn decode(
        &self,
        request: BackendDecodeRequest,
    ) -> std::result::Result<BackendDecodeResult, ProtocolError> {
        let (record, context) = {
            let state = self.lock_state()?;
            let server = self.active_validated_server(&state)?;
            let (handle, instance_id) = in_process_handle(request.state.state())?;
            if instance_id != server_generation_id(&server)? {
                return Err(protocol_incompatible(
                    "llama.cpp state belongs to another process generation",
                ));
            }
            let record = state.records.get(handle).cloned().ok_or_else(|| {
                protocol_incompatible("llama.cpp state handle is invalid or expired")
            })?;
            let context = state
                .decode_contexts
                .get(&record.context_id)
                .cloned()
                .ok_or_else(|| {
                    protocol_incompatible("llama.cpp state decode context is unavailable")
                })?;
            if !context
                .compatibility
                .mismatch_fields(&request.compatibility)
                .is_empty()
            {
                return Err(protocol_incompatible(
                    "llama.cpp state compatibility validation failed",
                ));
            }
            (record, context)
        };
        if !record.server.is_running() {
            return Err(protocol_unavailable(
                "the llama.cpp process owning this state is no longer running",
            ));
        }
        let _operation = record
            .server
            .state_gate
            .lock()
            .map_err(|_| protocol_internal("llama.cpp state operation gate is unavailable"))?;
        let snapshot_bytes = validate_private_snapshot(&record.server, &record.snapshot_name, None)
            .map_err(|_| protocol_incompatible("llama.cpp state snapshot is unavailable"))?;
        erase_slot_best_effort(&record.server);
        restore_llama_slot(
            &record.server,
            &record.snapshot_name,
            context.prompt_tokens,
            snapshot_bytes,
        )
        .map_err(|_| {
            erase_slot_best_effort(&record.server);
            protocol_incompatible("llama.cpp state restore validation failed")
        })?;
        let decoded = match run_llama_decode(&record.server, &context.input, &request.options) {
            Ok(decoded) => decoded,
            Err(_) => {
                erase_slot_best_effort(&record.server);
                return Err(protocol_unavailable("llama.cpp decode failed"));
            }
        };
        let cache_valid = llama_slot_status(&record.server).is_ok_and(|slot| {
            !slot.is_processing
                && slot.prompt_tokens_cache
                    >= context
                        .prompt_tokens
                        .saturating_sub(1)
                        .min(context.prompt_tokens)
                && (context.prompt_tokens <= 1 || slot.prompt_tokens_cache > 0)
        });
        erase_slot_best_effort(&record.server);
        if !cache_valid {
            return Err(protocol_incompatible(
                "llama.cpp did not prove restored prompt-cache reuse",
            ));
        }
        Ok(BackendDecodeResult {
            text: decoded.text,
            state: None,
            completion_tokens: decoded.completion_tokens,
            finish_reason: decoded.finish_reason,
        })
    }

    fn restore(
        &self,
        snapshot: BackendSnapshot,
        compatibility: &CompatibilityEnvelope,
    ) -> std::result::Result<BackendState, ProtocolError> {
        let mut state = self.lock_state()?;
        if state.records.len() >= MAX_CANONICAL_STATE_RECORDS {
            return Err(protocol_resource_exhausted(
                "the llama.cpp runtime-state handle limit has been reached",
            ));
        }
        let server = self.active_validated_server(&state)?;
        state.access_clock = state.access_clock.saturating_add(1);
        let access = state.access_clock;
        let retained = state
            .retained_contexts
            .get(&compatibility.prompt_fingerprint)
            .cloned()
            .ok_or_else(|| {
                protocol_unavailable(
                    "llama.cpp snapshots can be restored only by their original live process",
                )
            })?;
        let context = state
            .decode_contexts
            .get(&retained.context_id)
            .cloned()
            .ok_or_else(|| {
                protocol_unavailable("llama.cpp snapshot decode context is no longer retained")
            })?;
        if let Some(index) = state
            .retained_contexts
            .get_mut(&compatibility.prompt_fingerprint)
        {
            index.last_access = access;
        }
        if server_generation_id(&server)? != server_generation_id(&context.server)?
            || !context
                .compatibility
                .mismatch_fields(compatibility)
                .is_empty()
        {
            return Err(protocol_incompatible(
                "llama.cpp snapshot belongs to another process or runtime configuration",
            ));
        }
        let snapshot_name = random_private_id("restore_", 24)
            .map(|value| format!("{value}.bin"))
            .map_err(|_| protocol_unavailable("secure state allocation is unavailable"))?;
        let handle = random_private_id("lh_", 24)
            .map_err(|_| protocol_unavailable("secure state allocation is unavailable"))?;
        copy_snapshot_into_private_dir(&server, &snapshot, &snapshot_name)
            .map_err(|_| protocol_incompatible("llama.cpp snapshot validation failed"))?;
        let operation_result = server
            .state_gate
            .lock()
            .map_err(|_| protocol_internal("llama.cpp state operation gate is unavailable"))
            .and_then(|_operation| {
                erase_slot_best_effort(&server);
                if restore_llama_slot(
                    &server,
                    &snapshot_name,
                    context.prompt_tokens,
                    snapshot.bytes,
                )
                .is_err()
                {
                    erase_slot_best_effort(&server);
                    return Err(protocol_incompatible("llama.cpp snapshot restore failed"));
                }
                if erase_llama_slot(&server, Some(context.prompt_tokens)).is_err() {
                    erase_slot_best_effort(&server);
                    return Err(protocol_unavailable("llama.cpp slot cleanup failed"));
                }
                Ok(())
            });
        if let Err(error) = operation_result {
            remove_private_snapshot(&server, &snapshot_name);
            return Err(error);
        }
        state.records.insert(
            handle.clone(),
            LlamaStateRecord {
                server: server.clone(),
                snapshot_name,
                context_id: retained.context_id,
            },
        );
        Ok(BackendState::InProcess {
            handle,
            bytes: Some(snapshot.bytes),
            tier: StateTier::Disk,
            instance_id: server_generation_id(&server)?.to_string(),
        })
    }

    fn snapshot(
        &self,
        backend_state: &BackendState,
    ) -> std::result::Result<BackendState, ProtocolError> {
        let mut state = self.lock_state()?;
        let server = self.active_validated_server(&state)?;
        let (handle, instance_id) = in_process_handle(backend_state)?;
        if instance_id != server_generation_id(&server)? {
            return Err(protocol_incompatible(
                "llama.cpp state belongs to another process generation",
            ));
        }
        let record =
            state.records.get(handle).cloned().ok_or_else(|| {
                protocol_incompatible("llama.cpp state handle is invalid or expired")
            })?;
        let export_name = random_private_id("export_", 24)
            .map(|value| format!("{value}.bin"))
            .map_err(|_| protocol_unavailable("secure snapshot allocation is unavailable"))?;
        let (path, bytes) =
            copy_private_snapshot(&record.server, &record.snapshot_name, &export_name)
                .map_err(|_| protocol_unavailable("llama.cpp state snapshot failed"))?;
        state.exported_snapshots.insert(path.clone());
        Ok(BackendState::OpaqueFile {
            path,
            bytes,
            tier: StateTier::Disk,
            instance_id: server_generation_id(&server)?.to_string(),
        })
    }

    fn inspect_snapshot_export(
        &self,
        backend_state: &BackendState,
        compatibility: &CompatibilityEnvelope,
    ) -> std::result::Result<(), ProtocolError> {
        // This is the dry-run proof path: do not use `lock_state`, which may
        // drain deferred cleanup or expire retained contexts.
        let state = self
            .state
            .lock()
            .map_err(|_| protocol_internal("llama.cpp runtime-state adapter is unavailable"))?;
        let server = self.active_validated_server(&state)?;
        let (handle, instance_id) = in_process_handle(backend_state)?;
        if instance_id != server_generation_id(&server)? {
            return Err(protocol_incompatible(
                "llama.cpp state belongs to another process generation",
            ));
        }
        let record = state
            .records
            .get(handle)
            .ok_or_else(|| protocol_incompatible("llama.cpp state handle is invalid or expired"))?;
        let context = state
            .decode_contexts
            .get(&record.context_id)
            .ok_or_else(|| {
                protocol_incompatible("llama.cpp state decode context is unavailable")
            })?;
        if !Arc::ptr_eq(&server, &record.server)
            || !Arc::ptr_eq(&record.server, &context.server)
            || server_generation_id(&record.server)? != instance_id
            || server_generation_id(&context.server)? != instance_id
        {
            return Err(protocol_incompatible(
                "llama.cpp state owner is no longer the active process generation",
            ));
        }
        validate_compatibility(compatibility, &context.compatibility)?;
        let expected_bytes = backend_state.bytes().ok_or_else(|| {
            protocol_incompatible("llama.cpp state snapshot byte count is unavailable")
        })?;
        inspect_private_snapshot(&record.server, &record.snapshot_name, Some(expected_bytes))
            .map_err(|_| protocol_incompatible("llama.cpp state snapshot is unavailable"))?;
        Ok(())
    }

    fn validate_snapshot(
        &self,
        snapshot: &BackendState,
        _compatibility: &CompatibilityEnvelope,
    ) -> std::result::Result<(), ProtocolError> {
        let BackendState::OpaqueFile {
            path,
            bytes,
            tier,
            instance_id,
        } = snapshot
        else {
            return Err(protocol_incompatible(
                "llama.cpp snapshots must be registered opaque files",
            ));
        };
        if *tier != StateTier::Disk || *bytes == 0 {
            return Err(protocol_incompatible(
                "llama.cpp snapshot metadata is invalid",
            ));
        }

        let state = self.lock_state()?;
        let server = self.active_validated_server(&state)?;
        if instance_id != server_generation_id(&server)? {
            return Err(protocol_incompatible(
                "llama.cpp snapshot belongs to another process generation",
            )
            .with_details(json!({ "mismatch_fields": ["process_instance"] })));
        }
        if !state.exported_snapshots.contains(path) {
            return Err(protocol_incompatible(
                "llama.cpp snapshot export is invalid or expired",
            ));
        }
        Ok(())
    }

    fn release(&self, backend_state: &BackendState) -> std::result::Result<(), ProtocolError> {
        let Some(cleanup) = deferred_cleanup_for_state(backend_state) else {
            return Ok(());
        };

        // A lease can be released from `Drop`; neither adapter lock may be
        // waited on. If the main state is busy, accept the exact opaque target
        // into a bounded, deduplicated queue. A full or concurrently busy
        // queue is an explicit retryable failure rather than false success.
        match self.state.try_lock() {
            Ok(mut state) => {
                purge_expired_retained_contexts_at(&mut state, current_unix_ms());
                apply_deferred_cleanup(&mut state, cleanup);
                match self.deferred_cleanup.try_lock() {
                    Ok(mut deferred) => {
                        drain_deferred_cleanup(&mut state, &mut deferred);
                    }
                    Err(TryLockError::Poisoned(poisoned)) => {
                        let mut deferred = poisoned.into_inner();
                        drain_deferred_cleanup(&mut state, &mut deferred);
                    }
                    // The requested state has already been released. A busy
                    // queue contains unrelated cleanup and must not turn that
                    // completed release into a retryable failure.
                    Err(TryLockError::WouldBlock) => {}
                }
                Ok(())
            }
            Err(TryLockError::Poisoned(_)) => Err(protocol_internal(
                "llama.cpp runtime-state adapter is unavailable",
            )),
            Err(TryLockError::WouldBlock) => {
                let mut deferred = match self.deferred_cleanup.try_lock() {
                    Ok(deferred) => deferred,
                    Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                    Err(TryLockError::WouldBlock) => return Err(deferred_cleanup_busy()),
                };
                enqueue_deferred_cleanup(&mut deferred, cleanup)
            }
        }
    }
}

fn unavailable_llama_descriptor(
    mode: LlamaCppMode,
    identity: Option<&LlamaExecutableIdentity>,
    instance_id: String,
    reason: &str,
) -> BackendRuntimeDescriptor {
    let residency_status = if identity.is_some() {
        ModelResidencyStatus::Supported
    } else {
        ModelResidencyStatus::Unavailable
    };
    let residency_detail = if identity.is_some() {
        "Werk can keep an exact llama.cpp model process resident; named state remains unavailable until the running process passes functional validation"
    } else {
        "llama.cpp model residency is unavailable because no compatible managed runtime was discovered"
    };
    BackendRuntimeDescriptor {
        backend: label(mode).to_string(),
        backend_version: identity
            .map(llama_executable_version)
            .unwrap_or_else(|| "unavailable".to_string()),
        adapter_version: env!("CARGO_PKG_VERSION").to_string(),
        accelerator_family: llama_accelerator_family(mode).to_string(),
        instance_id,
        capabilities: llama_runtime_capabilities(
            llama_state_capabilities(false, reason),
            residency_status,
            residency_detail,
        ),
    }
}

fn llama_runtime_capabilities(
    mut capabilities: Vec<Capability>,
    residency_status: ModelResidencyStatus,
    residency_detail: &str,
) -> Vec<Capability> {
    capabilities.push(model_residency_capability(
        residency_status,
        residency_detail,
    ));
    capabilities
}

fn llama_state_capabilities(validated: bool, unavailable_reason: &str) -> Vec<Capability> {
    let status = if validated {
        CapabilityStatus::Experimental
    } else {
        CapabilityStatus::Unavailable
    };
    let detail = |operation: &str| {
        if validated {
            format!(
                "experimental llama.cpp {operation}; valid only in the exact current model process generation"
            )
        } else {
            unavailable_reason.to_string()
        }
    };
    let mut capabilities = [
        ("runtime.state.prefix_cache", "explicit-slot prefix cache"),
        (
            "runtime.state.persistence",
            "same-process opaque slot snapshots",
        ),
        (
            "runtime.state.restore",
            "same-process opaque slot snapshot restore",
        ),
        ("runtime.state.tier.disk", "disk-backed slot snapshots"),
        ("runtime.pd.prefill", "prefill-only execution"),
        ("runtime.pd.decode", "decode from an explicit slot snapshot"),
        ("runtime.pd.handoff", "opaque single-use handoff"),
    ]
    .into_iter()
    .map(|(id, operation)| Capability {
        id: id.to_string(),
        status: status.clone(),
        detail: detail(operation),
        operations: if validated {
            vec![operation.to_string()]
        } else {
            Vec::new()
        },
    })
    .collect::<Vec<_>>();
    capabilities.push(Capability {
        id: "runtime.state.restore.cross_restart".to_string(),
        status: CapabilityStatus::Unavailable,
        detail: "llama.cpp slot snapshots are deliberately rejected after any process restart"
            .to_string(),
        operations: Vec::new(),
    });
    capabilities
}

fn llama_accelerator_family(mode: LlamaCppMode) -> &'static str {
    match mode {
        LlamaCppMode::Cpu => "cpu",
        LlamaCppMode::Cuda => "cuda",
        LlamaCppMode::Rocm => "rocm",
        LlamaCppMode::Metal => "metal",
        LlamaCppMode::Vulkan => "vulkan",
    }
}

fn llama_executable_version(identity: &LlamaExecutableIdentity) -> String {
    format!(
        "{};binary={};help={}",
        identity.version, identity.binary_sha256, identity.help_sha256
    )
}

fn llama_process_version(identity: &LlamaProcessIdentity) -> String {
    format!(
        "{};args={}",
        llama_executable_version(&identity.executable),
        identity.args_sha256
    )
}

fn server_generation_id(server: &LlamaServerProcess) -> std::result::Result<&str, ProtocolError> {
    server
        .state_runtime
        .generation_id
        .as_deref()
        .ok_or_else(|| protocol_unavailable("llama.cpp process generation is unavailable"))
}

fn validate_request_generation(
    server: &LlamaServerProcess,
    model_id: &str,
    compatibility: &CompatibilityEnvelope,
) -> std::result::Result<(), ProtocolError> {
    let identity = server
        .state_runtime
        .identity
        .as_ref()
        .ok_or_else(|| protocol_unavailable("llama.cpp runtime identity is unavailable"))?;
    if server.model_id != model_id
        || compatibility.backend != label(server.mode)
        || compatibility.backend_version != llama_process_version(identity)
    {
        return Err(protocol_incompatible(
            "llama.cpp request does not match the validated model process",
        ));
    }
    Ok(())
}

fn in_process_handle(state: &BackendState) -> std::result::Result<(&str, &str), ProtocolError> {
    match state {
        BackendState::InProcess {
            handle,
            instance_id,
            ..
        } => Ok((handle, instance_id)),
        _ => Err(protocol_incompatible(
            "llama.cpp expected a private in-process state handle",
        )),
    }
}

fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn retained_context_expiration(now_unix_ms: u64, ttl_seconds: Option<u64>) -> Option<u64> {
    ttl_seconds.map(|seconds| now_unix_ms.saturating_add(seconds.saturating_mul(1_000)))
}

fn purge_expired_retained_contexts_at(state: &mut LlamaAdapterState, now_unix_ms: u64) {
    let expired = state
        .retained_contexts
        .iter()
        .filter(|(_, retained)| {
            retained
                .expires_unix_ms
                .is_some_and(|expires| expires <= now_unix_ms)
        })
        .map(|(fingerprint, retained)| (fingerprint.clone(), retained.context_id.clone()))
        .collect::<Vec<_>>();
    for (fingerprint, context_id) in expired {
        state.retained_contexts.remove(&fingerprint);
        // Live opaque state handles retain their decode context independently.
        // Expiry only ends named reuse/restore eligibility.
        remove_decode_context_if_unused(state, &context_id);
    }
}

fn deferred_cleanup_for_state(state: &BackendState) -> Option<DeferredCleanup> {
    match state {
        BackendState::InProcess {
            handle,
            instance_id,
            ..
        } => Some(DeferredCleanup::InProcess {
            handle: handle.clone(),
            instance_id: instance_id.clone(),
        }),
        BackendState::OpaqueFile { path, .. } => {
            Some(DeferredCleanup::OpaqueFile { path: path.clone() })
        }
        BackendState::OpaqueBytes { .. } | BackendState::External { .. } => None,
    }
}

fn enqueue_deferred_cleanup(
    queue: &mut DeferredCleanupQueue,
    cleanup: DeferredCleanup,
) -> std::result::Result<(), ProtocolError> {
    if queue.entries.contains(&cleanup) {
        return Ok(());
    }
    if queue.entries.len() >= MAX_DEFERRED_CLEANUPS {
        return Err(protocol_resource_exhausted(
            "the llama.cpp deferred state-cleanup limit has been reached; retry release",
        ));
    }
    queue.entries.push_back(cleanup);
    Ok(())
}

fn drain_deferred_cleanup(state: &mut LlamaAdapterState, queue: &mut DeferredCleanupQueue) {
    while let Some(cleanup) = queue.entries.pop_front() {
        apply_deferred_cleanup(state, cleanup);
    }
}

fn apply_deferred_cleanup(state: &mut LlamaAdapterState, cleanup: DeferredCleanup) {
    match cleanup {
        DeferredCleanup::InProcess {
            handle,
            instance_id,
        } => {
            let belongs_to_generation = state.records.get(&handle).is_some_and(|record| {
                record.server.state_runtime.generation_id.as_deref() == Some(&instance_id)
            });
            if belongs_to_generation && let Some(record) = state.records.remove(&handle) {
                remove_private_snapshot(&record.server, &record.snapshot_name);
                remove_decode_context_if_unused(state, &record.context_id);
            }
        }
        DeferredCleanup::OpaqueFile { path } => {
            if state.exported_snapshots.remove(&path) {
                remove_exported_snapshot(&path);
            }
        }
    }
}

fn deferred_cleanup_busy() -> ProtocolError {
    protocol_resource_exhausted(
        "llama.cpp state cleanup is busy and release was not accepted; retry release",
    )
}

fn cleanup_adapter_state(state: &mut LlamaAdapterState) {
    for (_, record) in state.records.drain() {
        remove_private_snapshot(&record.server, &record.snapshot_name);
    }
    for path in state.exported_snapshots.drain() {
        remove_exported_snapshot(&path);
    }
    state.retained_contexts.clear();
    state.decode_contexts.clear();
}

fn trim_retained_contexts(state: &mut LlamaAdapterState) {
    while state.retained_contexts.len() > MAX_CANONICAL_STATE_RECORDS {
        let Some(oldest) = state
            .retained_contexts
            .iter()
            .min_by_key(|(_, record)| record.last_access)
            .map(|(fingerprint, _)| fingerprint.clone())
        else {
            break;
        };
        if let Some(removed) = state.retained_contexts.remove(&oldest) {
            remove_decode_context_if_unused(state, &removed.context_id);
        }
    }
}

fn remove_decode_context_if_unused(state: &mut LlamaAdapterState, context_id: &str) {
    let referenced_by_state = state
        .records
        .values()
        .any(|record| record.context_id == context_id);
    let retained_for_reuse = state
        .retained_contexts
        .values()
        .any(|record| record.context_id == context_id);
    if !referenced_by_state && !retained_for_reuse {
        state.decode_contexts.remove(context_id);
    }
}

fn build_llama_compatibility(
    manifest: &ModelManifest,
    server: &LlamaServerProcess,
    runtime_options: &LlamaRuntimeOptions,
    prompt_fingerprint: &str,
) -> std::result::Result<CompatibilityEnvelope, ProtocolError> {
    let identity = server
        .state_runtime
        .identity
        .as_ref()
        .ok_or_else(|| protocol_unavailable("llama.cpp runtime identity is unavailable"))?;
    let mut files = manifest
        .files
        .iter()
        .map(|file| (&file.path, file.size, &file.checksum))
        .collect::<Vec<_>>();
    files.sort_by(|left, right| left.0.cmp(right.0));
    let model_fingerprint = sha256_json_value(&json!({
        "format": manifest.format,
        "architecture": manifest.architecture,
        "model_path": manifest.model_path,
        "files": files,
    }))
    .map_err(|_| protocol_internal("failed to fingerprint the llama.cpp model"))?;
    let tokenizer_fingerprint = if let Some(tokenizer_path) = manifest.tokenizer_path.as_deref() {
        let tokenizer = manifest
            .files
            .iter()
            .find(|file| file.path == tokenizer_path)
            .map(|file| json!([file.path, file.size, file.checksum]));
        sha256_json_value(&tokenizer)
    } else {
        sha256_json_value(&json!(["embedded-gguf-tokenizer", model_fingerprint]))
    }
    .map_err(|_| protocol_internal("failed to fingerprint the llama.cpp tokenizer"))?;
    let chat_template_fingerprint = sha256_json_value(&json!({
        "resolved": manifest.metadata.chat_template,
        "embedded_model": model_fingerprint,
    }))
    .map_err(|_| protocol_internal("failed to fingerprint the llama.cpp chat template"))?;
    let mut multimodal = manifest
        .files
        .iter()
        .filter(|file| {
            let lower = file.path.to_ascii_lowercase();
            lower.ends_with(".gguf") && (lower.contains("mmproj") || lower.contains("projector"))
        })
        .map(|file| {
            sha256_bytes(format!("{}:{}:{}", file.path, file.size, file.checksum).as_bytes())
        })
        .collect::<Vec<_>>();
    multimodal.sort();
    let context_size = last_option_value(&server.args, &["-c", "--ctx-size"])
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(runtime_options.ctx_size.unwrap_or(DEFAULT_CTX_SIZE) as u64);
    let batch_size = last_option_value(&server.args, &["-b", "--batch-size"])
        .and_then(|value| value.parse::<u64>().ok())
        .or_else(|| runtime_options.batch_size.map(|value| value as u64));
    let kv_dtype_k = last_option_value(&server.args, &["-ctk", "--cache-type-k"])
        .unwrap_or("backend-default")
        .to_string();
    let kv_dtype_v = last_option_value(&server.args, &["-ctv", "--cache-type-v"])
        .unwrap_or("backend-default")
        .to_string();
    Ok(CompatibilityEnvelope {
        model_fingerprint: model_fingerprint.clone(),
        tokenizer_fingerprint,
        prompt_fingerprint: prompt_fingerprint.to_string(),
        chat_template_fingerprint: Some(chat_template_fingerprint),
        backend: label(server.mode).to_string(),
        backend_version: llama_process_version(identity),
        runtime_adapter_version: env!("CARGO_PKG_VERSION").to_string(),
        accelerator_family: llama_accelerator_family(server.mode).to_string(),
        tensor_dtype: manifest
            .metadata
            .precision
            .clone()
            .unwrap_or_else(|| "gguf-mixed".to_string()),
        kv_dtype: format!("k={kv_dtype_k};v={kv_dtype_v}"),
        quantization: manifest
            .metadata
            .quantization
            .clone()
            .unwrap_or_else(|| "gguf".to_string()),
        cache_layout: format!("llama.cpp-slot0:args={}", identity.args_sha256),
        block_size: None,
        context: ContextCompatibility {
            context_size,
            batch_size,
            rope_configuration_fingerprint: Some(sha256_bytes(
                format!("gguf-rope:{model_fingerprint}").as_bytes(),
            )),
        },
        multimodal_processor_fingerprints: multimodal,
        producer_protocol: ProtocolVersion::V1,
    })
}

fn protocol_unsupported(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::Unsupported, message)
}

fn protocol_unavailable(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::Unavailable, message).retryable(true)
}

fn protocol_incompatible(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::IncompatibleState, message)
}

fn protocol_internal(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::Internal, message)
}

fn protocol_resource_exhausted(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ProtocolErrorCode::ResourceExhausted, message).retryable(true)
}

#[derive(Debug, Clone, Copy)]
struct LlamaSlotStatus {
    prompt_tokens: u64,
    prompt_tokens_cache: u64,
    is_processing: bool,
}

#[derive(Debug, Clone, Copy)]
struct LlamaSnapshotInfo {
    bytes: u64,
}

struct LlamaDecoded {
    text: String,
    completion_tokens: u64,
    finish_reason: String,
}

fn functional_probe_llama_state(server: &LlamaServerProcess) -> Result<()> {
    let probe_input = PrefillInput::Text {
        text: STATE_CAPABILITY_PROBE_PROMPT.to_string(),
    };
    let probe_name = format!("{}.bin", random_private_id("probe_", 24)?);
    let result = (|| {
        erase_slot_best_effort(server);
        let prompt_tokens = run_llama_prefill(server, &probe_input)?;
        if prompt_tokens < 2 {
            bail!("llama.cpp capability probe produced too few prompt tokens");
        }
        let snapshot = save_llama_slot(server, &probe_name, prompt_tokens)?;
        erase_llama_slot(server, Some(prompt_tokens))?;
        restore_llama_slot(server, &probe_name, prompt_tokens, snapshot.bytes)?;
        let replay_tokens = run_llama_prefill(server, &probe_input)?;
        if replay_tokens != prompt_tokens {
            bail!("llama.cpp capability probe replay changed tokenization");
        }
        let status = llama_slot_status(server)?;
        if status.is_processing || status.prompt_tokens_cache < prompt_tokens.saturating_sub(1) {
            bail!("llama.cpp capability probe did not prove restored cache reuse");
        }
        erase_llama_slot(server, Some(prompt_tokens))?;
        Ok(())
    })();
    erase_slot_best_effort(server);
    remove_private_snapshot(server, &probe_name);
    result
}

fn run_llama_prefill(server: &LlamaServerProcess, input: &PrefillInput) -> Result<u64> {
    let (path, body) = llama_state_request_body(input, Some(0), None);
    let response = control_json_request(&server.url, path, "POST", Some(&body))?;
    if !response.is_object() {
        bail!("llama.cpp prefill returned an invalid response");
    }
    if let Some(id_slot) = response.get("id_slot").and_then(Value::as_u64)
        && id_slot != u64::from(STATE_SLOT_ID)
    {
        bail!("llama.cpp prefill used an unexpected slot");
    }
    let status = llama_slot_status(server)?;
    if status.is_processing || status.prompt_tokens == 0 {
        bail!("llama.cpp prefill did not leave a valid idle prompt slot");
    }
    Ok(status.prompt_tokens)
}

fn run_llama_decode(
    server: &LlamaServerProcess,
    input: &PrefillInput,
    options: &BackendDecodeOptions,
) -> Result<LlamaDecoded> {
    let (path, body) = llama_state_request_body(input, Some(options.max_tokens), Some(options));
    let response = control_json_request(&server.url, path, "POST", Some(&body))?;
    match input {
        PrefillInput::Text { .. } => {
            let text = response
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("llama.cpp completion response has no content"))?
                .to_string();
            let completion_tokens = response
                .get("tokens_predicted")
                .and_then(Value::as_u64)
                .or_else(|| {
                    response
                        .get("timings")
                        .and_then(|timings| timings.get("predicted_n"))
                        .and_then(Value::as_u64)
                })
                .ok_or_else(|| anyhow!("llama.cpp completion response has no token count"))?;
            let finish_reason = match response
                .get("stop_type")
                .and_then(Value::as_str)
                .unwrap_or("limit")
            {
                "eos" | "word" => "stop",
                "limit" | "none" => "length",
                other if !other.is_empty() && other.len() <= 64 => other,
                _ => "unknown",
            }
            .to_string();
            Ok(LlamaDecoded {
                text,
                completion_tokens,
                finish_reason,
            })
        }
        PrefillInput::Messages { .. } => {
            let choice = response
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .ok_or_else(|| anyhow!("llama.cpp chat response has no choice"))?;
            let text = choice
                .get("message")
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let completion_tokens = response
                .get("usage")
                .and_then(|usage| usage.get("completion_tokens"))
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("llama.cpp chat response has no token count"))?;
            let finish_reason = choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .filter(|reason| !reason.is_empty() && reason.len() <= 64)
                .unwrap_or("unknown")
                .to_string();
            Ok(LlamaDecoded {
                text,
                completion_tokens,
                finish_reason,
            })
        }
    }
}

fn llama_state_request_body<'a>(
    input: &'a PrefillInput,
    max_tokens: Option<u64>,
    options: Option<&BackendDecodeOptions>,
) -> (&'static str, Value) {
    let token_limit = max_tokens.unwrap_or(0);
    let mut body = match input {
        PrefillInput::Text { text } => json!({
            "prompt": text,
            "n_predict": token_limit,
            "stream": false,
            "cache_prompt": true,
            "id_slot": STATE_SLOT_ID,
        }),
        PrefillInput::Messages { messages } => json!({
            "messages": messages,
            "max_tokens": token_limit,
            "stream": false,
            "cache_prompt": true,
            "id_slot": STATE_SLOT_ID,
        }),
    };
    if let Some(options) = options {
        if let Some(temperature) = options.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = options.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(seed) = options.seed {
            body["seed"] = json!(seed);
        }
        if !options.stop.is_empty() {
            body["stop"] = json!(options.stop);
        }
    }
    let path = match input {
        PrefillInput::Text { .. } => "/completion",
        PrefillInput::Messages { .. } => "/v1/chat/completions",
    };
    (path, body)
}

fn llama_slot_status(server: &LlamaServerProcess) -> Result<LlamaSlotStatus> {
    let response = control_json_request(&server.url, "/slots", "GET", None)?;
    let slots = response
        .as_array()
        .ok_or_else(|| anyhow!("llama.cpp slots response is not an array"))?;
    let slot = slots
        .iter()
        .find(|slot| slot.get("id").and_then(Value::as_u64) == Some(u64::from(STATE_SLOT_ID)))
        .ok_or_else(|| anyhow!("llama.cpp explicit state slot is missing"))?;
    Ok(LlamaSlotStatus {
        prompt_tokens: slot
            .get("n_prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        prompt_tokens_cache: slot
            .get("n_prompt_tokens_cache")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        is_processing: slot
            .get("is_processing")
            .and_then(Value::as_bool)
            .ok_or_else(|| anyhow!("llama.cpp slot status is incomplete"))?,
    })
}

fn save_llama_slot(
    server: &LlamaServerProcess,
    filename: &str,
    expected_tokens: u64,
) -> Result<LlamaSnapshotInfo> {
    let response = control_json_request(
        &server.url,
        &format!("/slots/{STATE_SLOT_ID}?action=save"),
        "POST",
        Some(&json!({"filename": filename})),
    )?;
    validate_slot_action_identity(&response, filename)?;
    let saved = required_u64(&response, "n_saved")?;
    let written = required_u64(&response, "n_written")?;
    if saved != expected_tokens || written == 0 || written > STATE_SNAPSHOT_MAX_BYTES {
        bail!("llama.cpp slot save response failed validation");
    }
    let file_bytes = validate_private_snapshot(server, filename, Some(written))?;
    if file_bytes != written {
        bail!("llama.cpp slot save byte count does not match the snapshot");
    }
    Ok(LlamaSnapshotInfo { bytes: written })
}

fn restore_llama_slot(
    server: &LlamaServerProcess,
    filename: &str,
    expected_tokens: u64,
    expected_bytes: u64,
) -> Result<()> {
    validate_private_snapshot(server, filename, Some(expected_bytes))?;
    let response = control_json_request(
        &server.url,
        &format!("/slots/{STATE_SLOT_ID}?action=restore"),
        "POST",
        Some(&json!({"filename": filename})),
    )?;
    validate_slot_action_identity(&response, filename)?;
    if required_u64(&response, "n_restored")? != expected_tokens
        || required_u64(&response, "n_read")? != expected_bytes
    {
        bail!("llama.cpp slot restore response failed validation");
    }
    let slot = llama_slot_status(server)?;
    if slot.is_processing || slot.prompt_tokens != expected_tokens {
        bail!("llama.cpp restored slot status failed validation");
    }
    Ok(())
}

fn erase_llama_slot(server: &LlamaServerProcess, expected_tokens: Option<u64>) -> Result<()> {
    let response = control_json_request(
        &server.url,
        &format!("/slots/{STATE_SLOT_ID}?action=erase"),
        "POST",
        Some(&json!({})),
    )?;
    if response.get("id_slot").and_then(Value::as_u64) != Some(u64::from(STATE_SLOT_ID)) {
        bail!("llama.cpp slot erase returned an unexpected slot");
    }
    let erased = required_u64(&response, "n_erased")?;
    if expected_tokens.is_some_and(|expected| erased != expected) {
        bail!("llama.cpp slot erase token count failed validation");
    }
    Ok(())
}

fn erase_slot_best_effort(server: &LlamaServerProcess) {
    let _ = erase_llama_slot(server, None);
}

fn validate_slot_action_identity(response: &Value, filename: &str) -> Result<()> {
    if response.get("id_slot").and_then(Value::as_u64) != Some(u64::from(STATE_SLOT_ID))
        || response.get("filename").and_then(Value::as_str) != Some(filename)
    {
        bail!("llama.cpp slot action identity failed validation");
    }
    Ok(())
}

fn required_u64(value: &Value, field: &str) -> Result<u64> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("llama.cpp response is missing {field}"))
}

fn control_json_request(
    base_url: &str,
    path: &str,
    method: &str,
    body: Option<&Value>,
) -> Result<Value> {
    let (_, host, port) = parse_local_url(base_url)?;
    let ip = host
        .parse::<IpAddr>()
        .with_context(|| "llama.cpp state endpoint is not an IP literal".to_string())?;
    if !ip.is_loopback() {
        bail!("llama.cpp state endpoint is not loopback-only");
    }
    if !matches!(method, "GET" | "POST")
        || !path.starts_with('/')
        || path.bytes().any(|byte| matches!(byte, b'\r' | b'\n'))
    {
        bail!("invalid llama.cpp state request target");
    }

    let body = body.map(serde_json::to_vec).transpose()?;
    if body
        .as_ref()
        .is_some_and(|body| body.len() > STATE_HTTP_MAX_REQUEST_BYTES)
    {
        bail!("llama.cpp state request exceeds its size limit");
    }

    let address = SocketAddr::new(ip, port);
    let mut stream = TcpStream::connect_timeout(&address, STATE_HTTP_TIMEOUT)
        .with_context(|| "failed to connect to the local llama.cpp state endpoint".to_string())?;
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(STATE_HTTP_TIMEOUT))?;
    stream.set_write_timeout(Some(STATE_HTTP_TIMEOUT))?;

    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nAccept: application/json\r\n"
    );
    if let Some(body) = &body {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes())?;
    if let Some(body) = body {
        stream.write_all(&body)?;
    }
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut header_bytes = 0usize;
    let status_line = read_bounded_http_line(&mut reader, 8 * 1024, &mut header_bytes)?;
    let mut status_parts = status_line.split_whitespace();
    let protocol = status_parts.next().unwrap_or_default();
    let status = status_parts
        .next()
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| anyhow!("invalid HTTP status from llama.cpp state endpoint"))?;
    if !matches!(protocol, "HTTP/1.0" | "HTTP/1.1") {
        bail!("invalid HTTP protocol from llama.cpp state endpoint");
    }

    let mut content_length = None;
    let mut chunked = false;
    loop {
        let line = read_bounded_http_line(&mut reader, 8 * 1024, &mut header_bytes)?;
        if line.is_empty() {
            break;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| anyhow!("invalid HTTP header from llama.cpp state endpoint"))?;
        let name = name.trim();
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            let parsed = value.parse::<usize>().map_err(|_| {
                anyhow!("invalid HTTP content length from llama.cpp state endpoint")
            })?;
            if content_length.is_some_and(|existing| existing != parsed) {
                bail!("conflicting HTTP content lengths from llama.cpp state endpoint");
            }
            content_length = Some(parsed);
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            let encodings = value
                .split(',')
                .map(str::trim)
                .filter(|encoding| !encoding.is_empty())
                .collect::<Vec<_>>();
            if encodings.len() != 1 || !encodings[0].eq_ignore_ascii_case("chunked") {
                bail!("unsupported HTTP transfer encoding from llama.cpp state endpoint");
            }
            chunked = true;
        }
    }
    if !(200..300).contains(&status) {
        bail!("llama.cpp state endpoint returned HTTP {status}");
    }
    if chunked && content_length.is_some() {
        bail!("ambiguous HTTP response framing from llama.cpp state endpoint");
    }

    let response = if chunked {
        read_bounded_chunked_body(&mut reader, STATE_HTTP_MAX_RESPONSE_BYTES)?
    } else if let Some(length) = content_length {
        if length > STATE_HTTP_MAX_RESPONSE_BYTES {
            bail!("llama.cpp state response exceeds its size limit");
        }
        let mut response = vec![0u8; length];
        reader.read_exact(&mut response)?;
        response
    } else {
        let mut response = Vec::new();
        reader
            .by_ref()
            .take((STATE_HTTP_MAX_RESPONSE_BYTES + 1) as u64)
            .read_to_end(&mut response)?;
        if response.len() > STATE_HTTP_MAX_RESPONSE_BYTES {
            bail!("llama.cpp state response exceeds its size limit");
        }
        response
    };
    if response.is_empty() {
        bail!("llama.cpp state endpoint returned an empty response");
    }
    serde_json::from_slice(&response)
        .with_context(|| "llama.cpp state endpoint returned invalid JSON".to_string())
}

fn read_bounded_http_line<R: BufRead>(
    reader: &mut R,
    line_limit: usize,
    total: &mut usize,
) -> Result<String> {
    let mut bytes = Vec::new();
    loop {
        if bytes.len() >= line_limit || *total >= 64 * 1024 {
            bail!("llama.cpp state response headers exceed their size limit");
        }
        let mut byte = [0u8; 1];
        let read = reader.read(&mut byte)?;
        if read == 0 {
            bail!("truncated HTTP headers from llama.cpp state endpoint");
        }
        *total += 1;
        if byte[0] == b'\n' {
            break;
        }
        bytes.push(byte[0]);
    }
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    String::from_utf8(bytes)
        .map_err(|_| anyhow!("non-UTF-8 HTTP headers from llama.cpp state endpoint"))
}

fn read_bounded_chunked_body<R: BufRead>(reader: &mut R, limit: usize) -> Result<Vec<u8>> {
    let mut response = Vec::new();
    loop {
        let mut framing_bytes = 0usize;
        let size_line = read_bounded_http_line(reader, 1024, &mut framing_bytes)?;
        let size_text = size_line
            .split_once(';')
            .map(|(size, _)| size)
            .unwrap_or(&size_line)
            .trim();
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|_| anyhow!("invalid HTTP chunk size from llama.cpp state endpoint"))?;
        if size == 0 {
            loop {
                let trailer = read_bounded_http_line(reader, 8 * 1024, &mut framing_bytes)?;
                if trailer.is_empty() {
                    break;
                }
                if !trailer.contains(':') {
                    bail!("invalid HTTP trailer from llama.cpp state endpoint");
                }
            }
            return Ok(response);
        }
        if size > limit.saturating_sub(response.len()) {
            bail!("llama.cpp state response exceeds its size limit");
        }
        let start = response.len();
        response.resize(start + size, 0);
        reader.read_exact(&mut response[start..])?;
        let mut terminator = [0u8; 2];
        reader.read_exact(&mut terminator)?;
        if terminator != *b"\r\n" {
            bail!("invalid HTTP chunk framing from llama.cpp state endpoint");
        }
    }
}

fn private_snapshot_path(server: &LlamaServerProcess, filename: &str) -> Result<PathBuf> {
    if filename.is_empty()
        || filename.len() > 128
        || filename == "."
        || filename == ".."
        || !filename
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        || Path::new(filename).components().count() != 1
    {
        bail!("invalid private llama.cpp snapshot name");
    }
    let directory = server
        .state_runtime
        .snapshot_dir
        .as_deref()
        .ok_or_else(|| anyhow!("private llama.cpp snapshot storage is unavailable"))?;
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("private llama.cpp snapshot storage is not a real directory");
    }
    #[cfg(unix)]
    if metadata.permissions().mode() & 0o077 != 0 {
        bail!("private llama.cpp snapshot storage is not owner-only");
    }
    Ok(directory.join(filename))
}

fn open_bounded_regular_file(path: &Path, expected_bytes: Option<u64>) -> Result<(fs::File, u64)> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!("runtime-state snapshot is not a regular file");
    }
    let bytes = metadata.len();
    if bytes == 0 || bytes > STATE_SNAPSHOT_MAX_BYTES || expected_bytes.is_some_and(|v| v != bytes)
    {
        bail!("runtime-state snapshot size failed validation");
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let file = options.open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file() || opened.len() != bytes {
        bail!("runtime-state snapshot changed while it was opened");
    }
    Ok((file, bytes))
}

fn validate_private_snapshot(
    server: &LlamaServerProcess,
    filename: &str,
    expected_bytes: Option<u64>,
) -> Result<u64> {
    let path = private_snapshot_path(server, filename)?;
    let (file, bytes) = open_bounded_regular_file(&path, expected_bytes)?;
    #[cfg(unix)]
    {
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
        if file.metadata()?.permissions().mode() & 0o077 != 0 {
            bail!("private llama.cpp snapshot is not owner-only");
        }
    }
    Ok(bytes)
}

fn inspect_private_snapshot(
    server: &LlamaServerProcess,
    filename: &str,
    expected_bytes: Option<u64>,
) -> Result<u64> {
    let path = private_snapshot_path(server, filename)?;
    let (file, bytes) = open_bounded_regular_file(&path, expected_bytes)?;
    #[cfg(unix)]
    if file.metadata()?.permissions().mode() & 0o077 != 0 {
        bail!("private llama.cpp snapshot is not owner-only");
    }
    Ok(bytes)
}

fn remove_private_snapshot(server: &LlamaServerProcess, filename: &str) {
    let Ok(path) = private_snapshot_path(server, filename) else {
        return;
    };
    remove_exported_snapshot(&path);
}

fn remove_exported_snapshot(path: &Path) {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return;
    };
    if metadata.file_type().is_symlink() || metadata.is_file() {
        let _ = fs::remove_file(path);
    }
}

fn copy_private_snapshot(
    server: &LlamaServerProcess,
    source_name: &str,
    target_name: &str,
) -> Result<(PathBuf, u64)> {
    let source = private_snapshot_path(server, source_name)?;
    let target = private_snapshot_path(server, target_name)?;
    let (_, expected_bytes) = open_bounded_regular_file(&source, None)?;
    let bytes = copy_bounded_snapshot(&source, &target, expected_bytes)?;
    validate_private_snapshot(server, target_name, Some(bytes))?;
    Ok((target, bytes))
}

fn copy_snapshot_into_private_dir(
    server: &LlamaServerProcess,
    snapshot: &BackendSnapshot,
    target_name: &str,
) -> Result<()> {
    let target = private_snapshot_path(server, target_name)?;
    let source = snapshot
        .try_clone_file()
        .context("failed to duplicate the verified runtime-state snapshot")?;
    copy_bounded_snapshot_file(source, &target, snapshot.bytes)?;
    validate_private_snapshot(server, target_name, Some(snapshot.bytes))?;
    Ok(())
}

fn copy_bounded_snapshot(source: &Path, target: &Path, expected_bytes: u64) -> Result<u64> {
    let (source_file, bytes) = open_bounded_regular_file(source, Some(expected_bytes))?;
    if bytes != expected_bytes {
        bail!("runtime-state snapshot changed while it was copied");
    }
    copy_bounded_snapshot_file(source_file, target, expected_bytes)
}

fn copy_bounded_snapshot_file(
    mut source_file: fs::File,
    target: &Path,
    expected_bytes: u64,
) -> Result<u64> {
    let source_metadata = source_file.metadata()?;
    if !source_metadata.is_file()
        || source_metadata.len() != expected_bytes
        || expected_bytes > STATE_SNAPSHOT_MAX_BYTES
    {
        bail!("runtime-state snapshot changed while it was copied");
    }
    source_file.seek(SeekFrom::Start(0))?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600).custom_flags(libc::O_CLOEXEC);
    let mut target_file = options.open(target)?;
    let result = (|| {
        let mut copied = 0u64;
        let mut buffer = [0u8; 1024 * 1024];
        loop {
            let read = source_file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            copied = copied.saturating_add(read as u64);
            if copied > expected_bytes || copied > STATE_SNAPSHOT_MAX_BYTES {
                bail!("runtime-state snapshot changed while it was copied");
            }
            target_file.write_all(&buffer[..read])?;
        }
        if copied != expected_bytes {
            bail!("runtime-state snapshot changed while it was copied");
        }
        target_file.flush()?;
        target_file.sync_all()?;
        Ok(copied)
    })();
    if result.is_err() {
        let _ = fs::remove_file(target);
    }
    result
}

pub(super) fn prepare_llama_state_snapshot_dir(
    store: &ModelStore,
    supported: &SupportedArgs,
) -> Result<(Option<String>, Option<PathBuf>)> {
    if !(supported.help_succeeded
        && supported.slots
        && supported.slot_save_path
        && supported.parallel)
    {
        return Ok((None, None));
    }
    #[cfg(not(unix))]
    {
        let _ = store;
        return Ok((None, None));
    }
    #[cfg(unix)]
    {
        let generation_id = random_private_id("pg_", 24)?;
        let backends = store.home().join("backends");
        ensure_real_directory(&backends, false)?;
        let runtime_root = backends.join(".runtime-state");
        ensure_real_directory(&runtime_root, true)?;
        let llama_root = runtime_root.join("llama.cpp");
        ensure_real_directory(&llama_root, true)?;
        let snapshot_dir = llama_root.join(&generation_id);
        fs::create_dir(&snapshot_dir).with_context(|| {
            "failed to create the private llama.cpp state working directory".to_string()
        })?;
        fs::set_permissions(&snapshot_dir, fs::Permissions::from_mode(0o700)).with_context(
            || "failed to protect the private llama.cpp state working directory".to_string(),
        )?;
        Ok((Some(generation_id), Some(snapshot_dir)))
    }
}

fn ensure_real_directory(path: &Path, private: bool) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                bail!("runtime-state working directory is not a real directory");
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(path)
                .with_context(|| "failed to create runtime-state working directory".to_string())?;
        }
        Err(error) => return Err(error.into()),
    }
    #[cfg(unix)]
    if private {
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| "failed to protect runtime-state working directory".to_string())?;
        let mode = fs::symlink_metadata(path)?.permissions().mode();
        if mode & 0o077 != 0 {
            bail!("runtime-state working directory is not owner-only");
        }
    }
    Ok(())
}

pub(super) fn cleanup_llama_snapshot_dir(path: &Path) {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return;
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        let _ = fs::remove_file(path);
    } else {
        let _ = fs::remove_dir_all(path);
    }
}

fn random_private_id(prefix: &str, random_bytes: usize) -> Result<String> {
    let mut random = vec![0u8; random_bytes];
    getrandom::getrandom(&mut random)
        .map_err(|_| anyhow!("secure runtime-state identifier generation is unavailable"))?;
    let mut value = String::with_capacity(prefix.len() + random_bytes * 2);
    value.push_str(prefix);
    for byte in random {
        use std::fmt::Write as _;
        write!(&mut value, "{byte:02x}").expect("writing hexadecimal to String cannot fail");
    }
    Ok(value)
}

pub(super) fn probe_llama_process_identity(
    executable: &Path,
    args: &[String],
) -> Result<LlamaProcessIdentity> {
    Ok(LlamaProcessIdentity {
        executable: probe_llama_executable_identity(executable)?,
        args_sha256: sha256_json_value(args)?,
    })
}

fn probe_llama_executable_identity(executable: &Path) -> Result<LlamaExecutableIdentity> {
    let metadata = fs::metadata(executable)
        .with_context(|| "failed to inspect the llama-server executable".to_string())?;
    if !metadata.is_file() {
        bail!("llama-server executable is not a regular file");
    }
    let binary_sha256 = sha256_regular_file(executable, 4 * 1024 * 1024 * 1024)?;
    let help = bounded_command_output(executable, "--help", 2 * 1024 * 1024)?;
    let version = bounded_command_output(executable, "--version", 64 * 1024)?;
    let version = version
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .ok_or_else(|| anyhow!("llama-server --version returned no version"))?;
    if version.len() > 512 || version.chars().any(char::is_control) {
        bail!("llama-server returned an invalid version string");
    }
    Ok(LlamaExecutableIdentity {
        version: version.to_string(),
        binary_sha256,
        help_sha256: sha256_bytes(help.as_bytes()),
    })
}

fn bounded_command_output(executable: &Path, argument: &str, max_bytes: usize) -> Result<String> {
    let output = Command::new(executable)
        .arg(argument)
        .output()
        .with_context(|| format!("failed to execute llama-server {argument}"))?;
    if !output.status.success() {
        bail!("llama-server {argument} failed");
    }
    if output.stdout.len().saturating_add(output.stderr.len()) > max_bytes {
        bail!("llama-server {argument} output exceeds its size limit");
    }
    let mut text = String::from_utf8(output.stdout)
        .map_err(|_| anyhow!("llama-server {argument} output is not UTF-8"))?;
    if !output.stderr.is_empty() {
        text.push('\n');
        text.push_str(
            std::str::from_utf8(&output.stderr)
                .map_err(|_| anyhow!("llama-server {argument} output is not UTF-8"))?,
        );
    }
    Ok(text)
}

fn sha256_regular_file(path: &Path, max_bytes: u64) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        bail!("file is not a bounded regular file");
    }
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if total > max_bytes {
            bail!("file exceeds its size limit");
        }
        hasher.update(&buffer[..read]);
    }
    if total != metadata.len() {
        bail!("file changed while it was being hashed");
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn sha256_json_value<T: serde::Serialize + ?Sized>(value: &T) -> Result<String> {
    Ok(sha256_bytes(&serde_json::to_vec(value)?))
}

pub(super) fn llama_state_args_are_effective(
    args: &[String],
    snapshot_dir: &Path,
    model_path: &Path,
    port: u16,
) -> bool {
    last_option_value(args, &["-np", "--parallel"]) == Some("1")
        && last_option_value(args, &["--slot-save-path"]) == snapshot_dir.to_str()
        && last_option_value(args, &["--model", "-m"]) == model_path.to_str()
        && last_option_value(args, &["--host"]) == Some("127.0.0.1")
        && last_option_value(args, &["--port"]).and_then(|value| value.parse::<u16>().ok())
            == Some(port)
        && last_toggle_value(args, "--slots", "--no-slots", true)
        && (!args.iter().any(|argument| argument == "--cache-ram")
            || last_option_value(args, &["--cache-ram"]) == Some("0"))
        && !last_toggle_value(args, "--cache-idle-slots", "--no-cache-idle-slots", false)
        && last_option_value(args, &["--api-prefix"]).is_none_or(str::is_empty)
}

fn last_option_value<'a>(args: &'a [String], names: &[&str]) -> Option<&'a str> {
    let mut value = None;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if names.iter().any(|name| argument == name) {
            value = args.get(index + 1).map(String::as_str);
            index += 2;
            continue;
        }
        for name in names {
            if let Some(next) = argument.strip_prefix(&format!("{name}=")) {
                value = Some(next);
            }
        }
        index += 1;
    }
    value
}

fn last_toggle_value(args: &[String], positive: &str, negative: &str, default: bool) -> bool {
    args.iter().fold(default, |value, argument| {
        if argument == positive {
            true
        } else if argument == negative {
            false
        } else {
            value
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::{ModelMetadata, ModelSource};
    use crate::werk_protocol::ProtocolMessage;
    use std::{
        collections::VecDeque,
        io::Cursor,
        net::TcpListener,
        process::Stdio,
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    struct FakeStateServer {
        url: String,
        stop: mpsc::Sender<()>,
        join: thread::JoinHandle<std::result::Result<Vec<String>, String>>,
    }

    impl FakeStateServer {
        fn finish(self) -> Vec<String> {
            let _ = self.stop.send(());
            self.join
                .join()
                .expect("fake state server thread panicked")
                .expect("fake state server failed")
        }
    }

    #[test]
    fn capability_declaration_is_fail_closed_and_never_claims_cross_restart_restore() {
        let unavailable = llama_state_capabilities(false, "not validated");
        assert!(
            unavailable
                .iter()
                .all(|capability| capability.status == CapabilityStatus::Unavailable)
        );

        let validated = llama_state_capabilities(true, "");
        for capability in &validated {
            if capability.id == "runtime.state.restore.cross_restart" {
                assert_eq!(capability.status, CapabilityStatus::Unavailable);
                assert!(capability.operations.is_empty());
            } else {
                assert_eq!(capability.status, CapabilityStatus::Experimental);
                assert!(!capability.operations.is_empty());
            }
        }
    }

    #[test]
    fn state_args_must_still_select_the_exact_private_single_slot_configuration() {
        let root = test_root("args");
        let model = root.join("model.gguf");
        let snapshots = root.join("private");
        let base = vec![
            "--model".to_string(),
            model.display().to_string(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            "32123".to_string(),
            "--parallel".to_string(),
            "1".to_string(),
            "--slots".to_string(),
            "--slot-save-path".to_string(),
            snapshots.display().to_string(),
            "--cache-ram".to_string(),
            "0".to_string(),
            "--no-cache-idle-slots".to_string(),
        ];
        assert!(llama_state_args_are_effective(
            &base, &snapshots, &model, 32123
        ));

        for override_args in [
            vec!["--parallel", "2"],
            vec!["--no-slots", ""],
            vec!["--slot-save-path", "/different"],
            vec!["--cache-ram", "1"],
            vec!["--cache-idle-slots", ""],
            vec!["--api-prefix", "/api"],
        ] {
            let mut args = base.clone();
            args.extend(
                override_args
                    .into_iter()
                    .filter(|value| !value.is_empty())
                    .map(str::to_string),
            );
            assert!(!llama_state_args_are_effective(
                &args, &snapshots, &model, 32123
            ));
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_help_option_matching_rejects_similar_names() {
        assert!(super::super::help_has_exact_option(
            "  --slots, enable endpoint",
            "--slots"
        ));
        assert!(super::super::help_has_exact_option(
            "[-np, --parallel N]",
            "--parallel"
        ));
        assert!(!super::super::help_has_exact_option(
            "--slots-view --parallelism",
            "--slots"
        ));
    }

    #[test]
    fn prefill_and_decode_bodies_pin_slot_zero_and_preserve_canonical_input() {
        let input = PrefillInput::Messages {
            messages: vec![ProtocolMessage {
                role: "user".to_string(),
                content: "exact message".to_string(),
            }],
        };
        let (path, prefill) = llama_state_request_body(&input, Some(0), None);
        assert_eq!(path, "/v1/chat/completions");
        assert_eq!(prefill["id_slot"], STATE_SLOT_ID);
        assert_eq!(prefill["max_tokens"], 0);
        assert_eq!(prefill["stream"], false);
        assert_eq!(prefill["cache_prompt"], true);
        assert_eq!(prefill["messages"][0]["content"], "exact message");
        assert!(prefill.get("prompt").is_none());

        let options = BackendDecodeOptions {
            max_tokens: 5,
            temperature: Some(0.25),
            top_p: Some(0.8),
            seed: Some(9),
            stop: vec!["done".to_string()],
        };
        let (_, decode) =
            llama_state_request_body(&input, Some(options.max_tokens), Some(&options));
        assert_eq!(decode["max_tokens"], 5);
        assert_eq!(decode["temperature"], 0.25);
        assert_eq!(decode["top_p"], 0.8);
        assert_eq!(decode["seed"], 9);
        assert_eq!(decode["stop"][0], "done");
    }

    #[test]
    fn bounded_http_parser_accepts_chunking_and_rejects_oversized_chunks() {
        let mut valid = Cursor::new(b"7\r\n{\"ok\":t\r\n4\r\nrue}\r\n0\r\n\r\n");
        assert_eq!(
            read_bounded_chunked_body(&mut valid, 32).unwrap(),
            br#"{"ok":true}"#
        );

        let mut oversized = Cursor::new(b"21\r\n");
        assert!(read_bounded_chunked_body(&mut oversized, 32).is_err());
    }

    #[test]
    fn state_http_rejects_non_loopback_and_oversized_request_before_connecting() {
        assert!(control_json_request("http://192.0.2.1:9", "/slots", "GET", None).is_err());
        let large = json!({"prompt": "x".repeat(STATE_HTTP_MAX_REQUEST_BYTES + 1)});
        let error = control_json_request("http://127.0.0.1:9", "/completion", "POST", Some(&large))
            .unwrap_err();
        assert!(error.to_string().contains("size limit"));
    }

    #[test]
    fn state_http_rejects_declared_oversized_response_without_reading_its_body() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = [0u8; 2048];
            let _ = stream.read(&mut request);
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                STATE_HTTP_MAX_RESPONSE_BYTES + 1
            )
            .unwrap();
        });
        let error =
            control_json_request(&format!("http://{address}"), "/slots", "GET", None).unwrap_err();
        assert!(error.to_string().contains("size limit"));
        join.join().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn private_snapshot_copy_is_owner_only_and_never_follows_symlinks() {
        use std::os::unix::fs::{PermissionsExt as _, symlink};

        let root = test_root("snapshot-copy");
        let source = root.join("source.bin");
        let target = root.join("target.bin");
        fs::write(&source, b"opaque-state").unwrap();
        assert_eq!(copy_bounded_snapshot(&source, &target, 12).unwrap(), 12);
        assert_eq!(fs::read(&target).unwrap(), b"opaque-state");
        assert_eq!(
            fs::metadata(&target).unwrap().permissions().mode() & 0o077,
            0
        );

        let external = root.join("external.bin");
        let link = root.join("link.bin");
        fs::write(&external, b"do-not-follow").unwrap();
        symlink(&external, &link).unwrap();
        assert!(open_bounded_regular_file(&link, None).is_err());
        assert_eq!(fs::read(&external).unwrap(), b"do-not-follow");

        let verified_source = root.join("verified-source.bin");
        let former_source = root.join("former-source.bin");
        let verified_target = root.join("verified-target.bin");
        fs::write(&verified_source, b"original-kv!").unwrap();
        let verified_handle = fs::File::open(&verified_source).unwrap();
        fs::rename(&verified_source, &former_source).unwrap();
        fs::write(&verified_source, b"replacement!").unwrap();
        copy_bounded_snapshot_file(verified_handle, &verified_target, 12).unwrap();
        assert_eq!(fs::read(&verified_target).unwrap(), b"original-kv!");
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn process_state_directory_is_private_and_unique() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = test_root("state-directory");
        let store = ModelStore::resolve(Some(root.clone())).unwrap();
        let supported = SupportedArgs {
            slots: true,
            slot_save_path: true,
            parallel: true,
            help_succeeded: true,
            ..SupportedArgs::default()
        };
        let (first_generation, first) =
            prepare_llama_state_snapshot_dir(&store, &supported).unwrap();
        let (second_generation, second) =
            prepare_llama_state_snapshot_dir(&store, &supported).unwrap();
        let first = first.unwrap();
        let second = second.unwrap();
        assert_ne!(first_generation, second_generation);
        assert_ne!(first, second);
        assert_eq!(
            fs::metadata(&first).unwrap().permissions().mode() & 0o077,
            0
        );
        assert_eq!(
            fs::metadata(&second).unwrap().permissions().mode() & 0o077,
            0
        );
        cleanup_llama_snapshot_dir(&first);
        cleanup_llama_snapshot_dir(&second);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn functional_probe_validates_the_full_private_slot_round_trip() {
        let root = test_root("functional-probe");
        let snapshot_dir = root.join("snapshots");
        fs::create_dir(&snapshot_dir).unwrap();
        #[cfg(unix)]
        fs::set_permissions(&snapshot_dir, fs::Permissions::from_mode(0o700)).unwrap();
        let fake = spawn_fake_state_server(snapshot_dir.clone(), false);
        let process = test_process(fake.url.clone(), snapshot_dir.clone());

        let result = functional_probe_llama_state(&process);
        let requests = fake.finish();
        result.unwrap();
        assert!(requests.len() >= 10);
        assert_eq!(
            requests.first().map(String::as_str),
            Some("POST /slots/0?action=erase")
        );
        assert!(requests.iter().any(|request| request == "POST /completion"));
        assert!(
            requests
                .iter()
                .any(|request| request == "POST /slots/0?action=save")
        );
        assert!(
            requests
                .iter()
                .any(|request| request == "POST /slots/0?action=restore")
        );
        assert!(fs::read_dir(&snapshot_dir).unwrap().next().is_none());

        drop(process);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn functional_probe_fails_closed_on_untruthful_slot_save_metadata() {
        let root = test_root("functional-probe-invalid");
        let snapshot_dir = root.join("snapshots");
        fs::create_dir(&snapshot_dir).unwrap();
        #[cfg(unix)]
        fs::set_permissions(&snapshot_dir, fs::Permissions::from_mode(0o700)).unwrap();
        let fake = spawn_fake_state_server(snapshot_dir.clone(), true);
        let process = test_process(fake.url.clone(), snapshot_dir.clone());

        let result = functional_probe_llama_state(&process);
        let _ = fake.finish();
        assert!(result.is_err());
        assert!(fs::read_dir(&snapshot_dir).unwrap().next().is_none());

        drop(process);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn retained_context_ttl_expires_named_reuse_but_preserves_a_live_handle() {
        assert_eq!(retained_context_expiration(10, Some(2)), Some(2_010));
        assert_eq!(retained_context_expiration(10, None), None);
        assert_eq!(
            retained_context_expiration(u64::MAX - 10, Some(1)),
            Some(u64::MAX)
        );

        let root = test_root("retained-ttl");
        let snapshots = root.join("snapshots");
        fs::create_dir(&snapshots).unwrap();
        let server = Arc::new(test_process("http://127.0.0.1:9".to_string(), snapshots));
        let mut state = LlamaAdapterState::default();
        for context_id in ["live-context", "orphan-context", "future-context"] {
            state.decode_contexts.insert(
                context_id.to_string(),
                LlamaDecodeContext {
                    server: server.clone(),
                    input: PrefillInput::Text {
                        text: format!("private-{context_id}"),
                    },
                    compatibility: test_compatibility(context_id),
                    prompt_tokens: 7,
                },
            );
        }
        state.records.insert(
            "live-handle".to_string(),
            LlamaStateRecord {
                server: server.clone(),
                snapshot_name: "missing.bin".to_string(),
                context_id: "live-context".to_string(),
            },
        );
        state.retained_contexts.insert(
            "expired-live".to_string(),
            RetainedContextIndex {
                context_id: "live-context".to_string(),
                last_access: 1,
                expires_unix_ms: Some(100),
            },
        );
        state.retained_contexts.insert(
            "expired-orphan".to_string(),
            RetainedContextIndex {
                context_id: "orphan-context".to_string(),
                last_access: 2,
                expires_unix_ms: Some(99),
            },
        );
        state.retained_contexts.insert(
            "future".to_string(),
            RetainedContextIndex {
                context_id: "future-context".to_string(),
                last_access: 3,
                expires_unix_ms: Some(101),
            },
        );

        purge_expired_retained_contexts_at(&mut state, 100);

        assert!(!state.retained_contexts.contains_key("expired-live"));
        assert!(!state.retained_contexts.contains_key("expired-orphan"));
        assert!(state.retained_contexts.contains_key("future"));
        assert!(matches!(
            state
                .decode_contexts
                .get("live-context")
                .map(|context| &context.input),
            Some(PrefillInput::Text { text }) if text == "private-live-context"
        ));
        assert!(!state.decode_contexts.contains_key("orphan-context"));

        apply_deferred_cleanup(
            &mut state,
            DeferredCleanup::InProcess {
                handle: "live-handle".to_string(),
                instance_id: "test-process-generation".to_string(),
            },
        );
        assert!(!state.decode_contexts.contains_key("live-context"));

        cleanup_adapter_state(&mut state);
        drop(state);
        drop(server);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn snapshot_export_inspection_is_pure_and_proves_the_registered_state() {
        let root = test_root("inspect-export");
        let snapshot_dir = root.join("snapshots");
        fs::create_dir(&snapshot_dir).unwrap();
        #[cfg(unix)]
        fs::set_permissions(&snapshot_dir, fs::Permissions::from_mode(0o700)).unwrap();
        let snapshot_name = "state.bin";
        let snapshot_path = snapshot_dir.join(snapshot_name);
        fs::write(&snapshot_path, b"opaque-state").unwrap();
        #[cfg(unix)]
        fs::set_permissions(&snapshot_path, fs::Permissions::from_mode(0o600)).unwrap();

        let server = Arc::new(test_process(
            "http://127.0.0.1:9".to_string(),
            snapshot_dir.clone(),
        ));
        let adapter = test_adapter(&root);
        let compatibility = test_compatibility("prompt");
        let exported_sentinel = root.join("existing-export.bin");
        let queued_sentinel = root.join("queued-export.bin");
        {
            let mut state = adapter.state.lock().unwrap();
            state.active = Some(LlamaActiveRuntime {
                server: server.clone(),
                validated: true,
            });
            state.records.insert(
                "live-handle".to_string(),
                LlamaStateRecord {
                    server: server.clone(),
                    snapshot_name: snapshot_name.to_string(),
                    context_id: "live-context".to_string(),
                },
            );
            state.decode_contexts.insert(
                "live-context".to_string(),
                LlamaDecodeContext {
                    server: server.clone(),
                    input: PrefillInput::Text {
                        text: "private-context".to_string(),
                    },
                    compatibility: compatibility.clone(),
                    prompt_tokens: 7,
                },
            );
            state.exported_snapshots.insert(exported_sentinel.clone());
            state.access_clock = 41;
        }
        adapter
            .deferred_cleanup
            .lock()
            .unwrap()
            .entries
            .push_back(DeferredCleanup::OpaqueFile {
                path: queued_sentinel.clone(),
            });
        let before_contents = fs::read(&snapshot_path).unwrap();
        #[cfg(unix)]
        let before_mode = fs::metadata(&snapshot_path).unwrap().permissions().mode();

        adapter
            .inspect_snapshot_export(
                &BackendState::InProcess {
                    handle: "live-handle".to_string(),
                    bytes: Some(12),
                    tier: StateTier::Disk,
                    instance_id: "test-process-generation".to_string(),
                },
                &compatibility,
            )
            .unwrap();

        let state = adapter.state.lock().unwrap();
        assert_eq!(state.access_clock, 41);
        assert_eq!(state.records.len(), 1);
        assert_eq!(state.decode_contexts.len(), 1);
        assert_eq!(state.exported_snapshots.len(), 1);
        assert!(state.exported_snapshots.contains(&exported_sentinel));
        drop(state);
        let deferred = adapter.deferred_cleanup.lock().unwrap();
        assert_eq!(deferred.entries.len(), 1);
        assert!(matches!(
            deferred.entries.front(),
            Some(DeferredCleanup::OpaqueFile { path }) if path == &queued_sentinel
        ));
        drop(deferred);
        assert_eq!(fs::read(&snapshot_path).unwrap(), before_contents);
        assert_eq!(fs::read_dir(&snapshot_dir).unwrap().count(), 1);
        #[cfg(unix)]
        assert_eq!(
            fs::metadata(&snapshot_path).unwrap().permissions().mode(),
            before_mode
        );

        drop(adapter);
        drop(server);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn contended_release_is_deduplicated_and_drained_by_the_next_operation() {
        let root = test_root("deferred-release");
        let export = root.join("export.bin");
        fs::write(&export, b"opaque-state").unwrap();
        let adapter = test_adapter(&root);
        let backend_state = BackendState::OpaqueFile {
            path: export.clone(),
            bytes: 12,
            tier: StateTier::Disk,
            instance_id: "generation".to_string(),
        };
        let mut guard = adapter.state.lock().unwrap();
        guard.exported_snapshots.insert(export.clone());

        adapter.release(&backend_state).unwrap();
        adapter.release(&backend_state).unwrap();
        assert_eq!(adapter.deferred_cleanup.lock().unwrap().entries.len(), 1);
        assert!(export.is_file());

        drop(guard);
        let _ = adapter.descriptor();
        assert!(adapter.deferred_cleanup.lock().unwrap().entries.is_empty());
        assert!(!export.exists());

        drop(adapter);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn deferred_release_capacity_fails_honestly_without_evicting_queued_work() {
        let root = test_root("deferred-capacity");
        let adapter = test_adapter(&root);
        let guard = adapter.state.lock().unwrap();
        for index in 0..MAX_DEFERRED_CLEANUPS {
            adapter
                .release(&BackendState::OpaqueFile {
                    path: root.join(format!("export-{index}.bin")),
                    bytes: 1,
                    tier: StateTier::Disk,
                    instance_id: "generation".to_string(),
                })
                .unwrap();
        }
        let error = adapter
            .release(&BackendState::OpaqueFile {
                path: root.join("overflow.bin"),
                bytes: 1,
                tier: StateTier::Disk,
                instance_id: "generation".to_string(),
            })
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);
        assert!(error.retryable);
        let deferred = adapter.deferred_cleanup.lock().unwrap();
        assert_eq!(deferred.entries.len(), MAX_DEFERRED_CLEANUPS);
        assert!(matches!(
            deferred.entries.front(),
            Some(DeferredCleanup::OpaqueFile { path }) if path == &root.join("export-0.bin")
        ));
        drop(deferred);

        drop(guard);
        let _ = adapter.descriptor();
        drop(adapter);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn release_reports_queue_lock_contention_instead_of_false_success() {
        let root = test_root("deferred-contention");
        let adapter = test_adapter(&root);
        let state_guard = adapter.state.lock().unwrap();
        let queue_guard = adapter.deferred_cleanup.lock().unwrap();
        let error = adapter
            .release(&BackendState::OpaqueFile {
                path: root.join("export.bin"),
                bytes: 1,
                tier: StateTier::Disk,
                instance_id: "generation".to_string(),
            })
            .unwrap_err();
        assert_eq!(error.code, ProtocolErrorCode::ResourceExhausted);
        assert!(error.retryable);
        assert!(queue_guard.entries.is_empty());

        drop(queue_guard);
        drop(state_guard);
        drop(adapter);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn release_succeeds_after_target_cleanup_when_unrelated_queue_is_locked() {
        let root = test_root("partial-release");
        let export = root.join("export.bin");
        fs::write(&export, b"opaque-state").unwrap();
        let adapter = test_adapter(&root);
        adapter
            .state
            .lock()
            .unwrap()
            .exported_snapshots
            .insert(export.clone());
        let queue_guard = adapter.deferred_cleanup.lock().unwrap();

        adapter
            .release(&BackendState::OpaqueFile {
                path: export.clone(),
                bytes: 12,
                tier: StateTier::Disk,
                instance_id: "generation".to_string(),
            })
            .unwrap();

        assert!(!export.exists());
        assert!(
            !adapter
                .state
                .lock()
                .unwrap()
                .exported_snapshots
                .contains(&export)
        );
        assert!(queue_guard.entries.is_empty());

        drop(queue_guard);
        drop(adapter);
        fs::remove_dir_all(root).unwrap();
    }

    fn spawn_fake_state_server(snapshot_dir: PathBuf, corrupt_save: bool) -> FakeStateServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let (stop_tx, stop_rx) = mpsc::channel();
        let join = thread::spawn(move || {
            let started = Instant::now();
            let mut requests = Vec::new();
            let mut slot_tokens = 0u64;
            let mut cache_tokens = 0u64;
            loop {
                if stop_rx.try_recv().is_ok() {
                    return Ok(requests);
                }
                if started.elapsed() > Duration::from_secs(10) {
                    return Err("fake llama.cpp state server timed out".to_string());
                }
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(error) => return Err(error.to_string()),
                };
                let (method, target, body) =
                    read_fake_request(&mut stream).map_err(|error| error.to_string())?;
                requests.push(format!("{method} {target}"));
                let snapshot_bytes = b"fake-slot-state";
                let response = match (method.as_str(), target.as_str()) {
                    ("POST", "/slots/0?action=erase") => {
                        let erased = slot_tokens;
                        slot_tokens = 0;
                        cache_tokens = 0;
                        json!({"id_slot": 0, "n_erased": erased})
                    }
                    ("POST", "/completion") => {
                        if body.get("prompt").and_then(Value::as_str)
                            != Some(STATE_CAPABILITY_PROBE_PROMPT)
                            || body.get("n_predict").and_then(Value::as_u64) != Some(0)
                            || body.get("id_slot").and_then(Value::as_u64) != Some(0)
                            || body.get("cache_prompt").and_then(Value::as_bool) != Some(true)
                        {
                            return Err("probe request was not exact".to_string());
                        }
                        if slot_tokens == 7 {
                            cache_tokens = 7;
                        }
                        slot_tokens = 7;
                        json!({"id_slot": 0, "content": "", "tokens_predicted": 0})
                    }
                    ("GET", "/slots") => json!([{
                        "id": 0,
                        "n_prompt_tokens": slot_tokens,
                        "n_prompt_tokens_cache": cache_tokens,
                        "is_processing": false,
                    }]),
                    ("POST", "/slots/0?action=save") => {
                        let filename = fake_filename(&body)?;
                        fs::write(snapshot_dir.join(filename), snapshot_bytes)
                            .map_err(|error| error.to_string())?;
                        json!({
                            "id_slot": 0,
                            "filename": filename,
                            "n_saved": if corrupt_save { 6 } else { slot_tokens },
                            "n_written": snapshot_bytes.len(),
                        })
                    }
                    ("POST", "/slots/0?action=restore") => {
                        let filename = fake_filename(&body)?;
                        let bytes = fs::read(snapshot_dir.join(filename))
                            .map_err(|error| error.to_string())?;
                        slot_tokens = 7;
                        cache_tokens = 7;
                        json!({
                            "id_slot": 0,
                            "filename": filename,
                            "n_restored": slot_tokens,
                            "n_read": bytes.len(),
                        })
                    }
                    _ => return Err(format!("unexpected fake request: {method} {target}")),
                };
                write_fake_json(&mut stream, &response).map_err(|error| error.to_string())?;
            }
        });
        FakeStateServer {
            url: format!("http://{address}"),
            stop: stop_tx,
            join,
        }
    }

    fn read_fake_request(stream: &mut TcpStream) -> Result<(String, String, Value)> {
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut request_line = String::new();
        reader.read_line(&mut request_line)?;
        let mut parts = request_line.split_whitespace();
        let method = parts
            .next()
            .ok_or_else(|| anyhow!("missing fake request method"))?
            .to_string();
        let target = parts
            .next()
            .ok_or_else(|| anyhow!("missing fake request target"))?
            .to_string();
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            if line == "\r\n" || line == "\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                content_length = value.trim().parse()?;
            }
        }
        let mut bytes = vec![0u8; content_length];
        reader.read_exact(&mut bytes)?;
        let body = if bytes.is_empty() {
            json!({})
        } else {
            serde_json::from_slice(&bytes)?
        };
        Ok((method, target, body))
    }

    fn write_fake_json(stream: &mut TcpStream, value: &Value) -> Result<()> {
        let bytes = serde_json::to_vec(value)?;
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            bytes.len()
        )?;
        stream.write_all(&bytes)?;
        stream.flush()?;
        Ok(())
    }

    fn fake_filename(body: &Value) -> std::result::Result<&str, String> {
        let filename = body
            .get("filename")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing fake snapshot filename".to_string())?;
        if filename.contains(['/', '\\']) {
            return Err("unsafe fake snapshot filename".to_string());
        }
        Ok(filename)
    }

    const TEST_PROCESS_CHILD_ENV: &str = "WERK_INTERNAL_LLAMA_TEST_PROCESS_CHILD";
    const TEST_PROCESS_CHILD_NAME: &str =
        "backend::llama_server::runtime_state::tests::test_process_child_waits_for_parent_stdin";

    #[test]
    fn test_process_child_waits_for_parent_stdin() {
        if std::env::var_os(TEST_PROCESS_CHILD_ENV).is_none() {
            return;
        }
        let mut input = Vec::new();
        std::io::stdin()
            .read_to_end(&mut input)
            .expect("test process child reads parent stdin");
    }

    fn test_process(url: String, snapshot_dir: PathBuf) -> LlamaServerProcess {
        let executable = std::env::current_exe().unwrap();
        let child = Command::new(&executable)
            .arg(TEST_PROCESS_CHILD_NAME)
            .arg("--exact")
            .env(TEST_PROCESS_CHILD_ENV, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        LlamaServerProcess {
            child: Mutex::new(child),
            executable,
            discovery_source: "test".to_string(),
            args: Vec::new(),
            model_path: PathBuf::from("model.gguf"),
            projector_path: None,
            model_id: "test-model".to_string(),
            model_identity: ModelRuntimeIdentity::from_manifest(&ModelManifest {
                id: "test-model".to_string(),
                source: ModelSource::LocalPath {
                    path: "test".to_string(),
                },
                format: ModelFormat::Gguf,
                architecture: Some("llama".to_string()),
                tokenizer_path: None,
                config_path: None,
                model_path: Some("model.gguf".to_string()),
                backend: "llama-server".to_string(),
                created_unix: 1,
                files: Vec::new(),
                artifacts: Vec::new(),
                metadata: ModelMetadata::default(),
            })
            .unwrap(),
            url,
            pid: std::process::id(),
            mode: LlamaCppMode::Cpu,
            log_tail: Arc::new(Mutex::new(VecDeque::new())),
            state_gate: Mutex::new(()),
            state_runtime: LlamaProcessStateRuntime {
                generation_id: Some("test-process-generation".to_string()),
                snapshot_dir: Some(snapshot_dir),
                identity: Some(LlamaProcessIdentity {
                    executable: LlamaExecutableIdentity {
                        version: "test".to_string(),
                        binary_sha256: "sha256:test".to_string(),
                        help_sha256: "sha256:test".to_string(),
                    },
                    args_sha256: "sha256:test".to_string(),
                }),
                configured: true,
            },
        }
    }

    fn test_adapter(root: &Path) -> LlamaRuntimeStateAdapter {
        let store = ModelStore::resolve(Some(root.to_path_buf())).unwrap();
        LlamaRuntimeStateAdapter {
            backend: LlamaServerBackend::new(
                store,
                LlamaCppMode::Cpu,
                LlamaRuntimeOptions::default(),
            ),
            unavailable_instance_id: "unavailable-test".to_string(),
            discovered_identity: None,
            state: Mutex::new(LlamaAdapterState::default()),
            deferred_cleanup: Mutex::new(DeferredCleanupQueue::default()),
        }
    }

    fn test_compatibility(prompt_fingerprint: &str) -> CompatibilityEnvelope {
        CompatibilityEnvelope {
            model_fingerprint: "model".to_string(),
            tokenizer_fingerprint: "tokenizer".to_string(),
            prompt_fingerprint: prompt_fingerprint.to_string(),
            chat_template_fingerprint: None,
            backend: "llama-cpu".to_string(),
            backend_version: "test".to_string(),
            runtime_adapter_version: "test".to_string(),
            accelerator_family: "cpu".to_string(),
            tensor_dtype: "f32".to_string(),
            kv_dtype: "f32".to_string(),
            quantization: "none".to_string(),
            cache_layout: "test".to_string(),
            block_size: None,
            context: ContextCompatibility {
                context_size: 4_096,
                batch_size: Some(512),
                rope_configuration_fingerprint: None,
            },
            multimodal_processor_fingerprints: Vec::new(),
            producer_protocol: ProtocolVersion::V1,
        }
    }

    fn test_root(name: &str) -> PathBuf {
        let suffix = random_private_id("", 12).unwrap();
        let root = std::env::temp_dir().join(format!("werk-llama-state-{name}-{suffix}"));
        fs::create_dir(&root).unwrap();
        root
    }
}
