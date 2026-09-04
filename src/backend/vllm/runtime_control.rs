//! Truthful control-plane metadata for the active vLLM OpenAI server.
//!
//! vLLM's automatic prefix cache is backend-owned and does not expose Werk's
//! named state lifecycle. This adapter therefore reports only what can be
//! proven from a concrete, already-started local process. It deliberately
//! implements none of the opaque state operations.

use super::{VllmBackend, VllmProcess};
use crate::{
    runtime_control::{
        BackendRuntimeAdapter, BackendRuntimeDescriptor, ModelResidencyStatus,
        UnsupportedRuntimeAdapter, model_residency_capability,
    },
    werk_protocol::{Capability, CapabilityStatus},
};
use std::sync::Arc;

const PREFIX_CACHE_CAPABILITY: &str = "runtime.state.prefix_cache";

pub(super) struct VllmRuntimeControlAdapter {
    backend: VllmBackend,
    fallback: UnsupportedRuntimeAdapter,
}

impl VllmRuntimeControlAdapter {
    pub(super) fn new(backend: VllmBackend) -> Self {
        let fallback = UnsupportedRuntimeAdapter::new(backend.accelerator.backend_label());
        Self { backend, fallback }
    }
}

impl BackendRuntimeAdapter for VllmRuntimeControlAdapter {
    fn descriptor(&self) -> BackendRuntimeDescriptor {
        let Ok(servers) = self.backend.servers.lock() else {
            let mut descriptor = self.fallback.descriptor();
            descriptor.capabilities = vec![
                model_residency_capability(
                    ModelResidencyStatus::Unavailable,
                    "vLLM model-residency metadata is unavailable because its process registry could not be read",
                ),
                prefix_cache_capability(
                    CapabilityStatus::Unavailable,
                    "vLLM runtime metadata is unavailable because its process registry could not be read",
                ),
            ];
            return descriptor;
        };
        let processes = servers
            .values()
            .filter(|server| process_is_active_without_remote_io(server))
            .cloned()
            .collect::<Vec<_>>();
        drop(servers);

        let (status, detail) = active_prefix_cache_status(&processes);
        let runtime_version =
            single_runtime_value(&processes, |process| process.runtime_version.as_str());
        let instance_id =
            single_runtime_value(&processes, |process| process.runtime_instance_id.as_str());
        BackendRuntimeDescriptor {
            backend: self.backend.accelerator.backend_label().to_string(),
            backend_version: runtime_version.unwrap_or_else(|| {
                if processes.is_empty() {
                    "unavailable".to_string()
                } else {
                    "mixed-or-unknown".to_string()
                }
            }),
            adapter_version: env!("CARGO_PKG_VERSION").to_string(),
            accelerator_family: self.backend.accelerator.display_name().to_ascii_lowercase(),
            // No vLLM state handle is issued by this adapter. A multi-process
            // label is therefore metadata, never a portable process identity.
            instance_id: instance_id.unwrap_or_else(|| "vllm-metadata-only".to_string()),
            capabilities: vec![
                active_model_residency_capability(&processes),
                prefix_cache_capability(status, detail),
            ],
        }
    }
}

fn active_model_residency_capability(processes: &[Arc<VllmProcess>]) -> Capability {
    if processes.is_empty() {
        return model_residency_capability(
            ModelResidencyStatus::Unavailable,
            "no active vLLM process is available to verify model residency",
        );
    }

    let local = processes
        .iter()
        .filter(|process| process.child.is_some())
        .count();
    if local == processes.len() {
        model_residency_capability(
            ModelResidencyStatus::Supported,
            "Werk owns and reuses the active local vLLM model processes",
        )
    } else if local == 0 {
        model_residency_capability(
            ModelResidencyStatus::ExternallyManaged,
            "the configured remote vLLM service owns model residency",
        )
    } else {
        model_residency_capability(
            ModelResidencyStatus::ExternallyManaged,
            "model residency spans Werk-owned local and externally managed remote vLLM processes",
        )
    }
}

fn process_is_active_without_remote_io(process: &&Arc<VllmProcess>) -> bool {
    match &process.child {
        None => true,
        Some(child) => child
            .lock()
            .ok()
            .and_then(|mut child| child.try_wait().ok())
            .is_some_and(|status| status.is_none()),
    }
}

fn active_prefix_cache_status(processes: &[Arc<VllmProcess>]) -> (CapabilityStatus, &'static str) {
    if processes.is_empty() {
        return (
            CapabilityStatus::Unavailable,
            "no active vLLM process is available to verify automatic prefix caching",
        );
    }
    if processes.iter().any(|process| process.child.is_none()) {
        return (
            CapabilityStatus::MetadataOnly,
            "remote vLLM configuration is not introspectable through its OpenAI endpoint",
        );
    }
    let settings = processes
        .iter()
        .map(|process| prefix_cache_setting(&process.args))
        .collect::<Vec<_>>();
    if settings
        .iter()
        .all(|setting| *setting == PrefixCacheSetting::Enabled)
    {
        (
            CapabilityStatus::ExternallyManaged,
            "automatic prefix caching is explicitly enabled and owned by the active vLLM process; Werk cannot name, persist, move, or prune its cache entries",
        )
    } else if settings
        .iter()
        .all(|setting| *setting == PrefixCacheSetting::Disabled)
    {
        (
            CapabilityStatus::Unsupported,
            "automatic prefix caching is explicitly disabled for the active vLLM process",
        )
    } else {
        (
            CapabilityStatus::MetadataOnly,
            "the active vLLM process does not have one unambiguous, explicitly enabled prefix-cache configuration",
        )
    }
}

fn prefix_cache_capability(status: CapabilityStatus, detail: &str) -> Capability {
    Capability {
        id: PREFIX_CACHE_CAPABILITY.to_string(),
        operations: matches!(status, CapabilityStatus::ExternallyManaged)
            .then(|| vec!["automatic_reuse".to_string()])
            .unwrap_or_default(),
        status,
        detail: detail.to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixCacheSetting {
    Enabled,
    Disabled,
    Ambiguous,
}

fn prefix_cache_setting(args: &[String]) -> PrefixCacheSetting {
    let enabled = args.iter().any(|arg| arg == "--enable-prefix-caching");
    let disabled = args.iter().any(|arg| arg == "--no-enable-prefix-caching");
    match (enabled, disabled) {
        (true, false) => PrefixCacheSetting::Enabled,
        (false, true) => PrefixCacheSetting::Disabled,
        _ => PrefixCacheSetting::Ambiguous,
    }
}

fn single_runtime_value(
    processes: &[Arc<VllmProcess>],
    value: impl Fn(&VllmProcess) -> &str,
) -> Option<String> {
    let first = value(processes.first()?.as_ref());
    processes
        .iter()
        .all(|process| value(process.as_ref()) == first)
        .then(|| first.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::ModelStore;
    use crate::runtime_control::MODEL_RESIDENCY_CAPABILITY;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "werk-vllm-runtime-control-{}-{}",
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

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn prefix_cache_requires_one_explicit_non_conflicting_flag() {
        assert_eq!(
            prefix_cache_setting(&args(&["--enable-prefix-caching"])),
            PrefixCacheSetting::Enabled
        );
        assert_eq!(
            prefix_cache_setting(&args(&["--no-enable-prefix-caching"])),
            PrefixCacheSetting::Disabled
        );
        assert_eq!(prefix_cache_setting(&[]), PrefixCacheSetting::Ambiguous);
        assert_eq!(
            prefix_cache_setting(&args(&[
                "--enable-prefix-caching",
                "--no-enable-prefix-caching"
            ])),
            PrefixCacheSetting::Ambiguous
        );
        assert_eq!(
            prefix_cache_setting(&args(&["--enable-prefix-caching=true"])),
            PrefixCacheSetting::Ambiguous
        );
    }

    #[test]
    fn inactive_backend_never_claims_operational_prefix_cache() {
        let temp = TestDir::new();
        let store = ModelStore::resolve(Some(temp.0.clone())).unwrap();
        let descriptor = VllmRuntimeControlAdapter::new(VllmBackend::new(store)).descriptor();
        let capability = descriptor
            .capabilities
            .iter()
            .find(|capability| capability.id == PREFIX_CACHE_CAPABILITY)
            .unwrap();
        assert_eq!(capability.status, CapabilityStatus::Unavailable);
        assert!(capability.operations.is_empty());

        let residency = descriptor
            .capabilities
            .iter()
            .find(|capability| capability.id == MODEL_RESIDENCY_CAPABILITY)
            .unwrap();
        assert_eq!(residency.status, CapabilityStatus::Unavailable);
        assert!(residency.operations.is_empty());
    }
}
