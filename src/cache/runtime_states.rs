//! Runtime snapshots are administered through their catalog, never as raw trees.
use super::{CacheEntry, CacheKind};
use crate::runtime_control::{StateStore, StoredCacheEntry};
use anyhow::{Context, Result, bail};
use std::path::Path;

pub(super) fn list(home: &Path) -> Result<Vec<CacheEntry>> {
    Ok(StateStore::new(home)
        .cache_inventory()?
        .into_iter()
        .map(cache_entry)
        .collect())
}

pub(super) fn purge(home: &Path, id: &str, dry_run: bool) -> Result<CacheEntry> {
    let suffix = id
        .strip_prefix("runtime-state:")
        .context("invalid runtime state cache ID")?;
    let (namespace, state_id) = suffix
        .split_once(':')
        .context("runtime state cache ID requires its namespace and state ID")?;
    if namespace.is_empty() || state_id.is_empty() || state_id.contains(':') {
        bail!("invalid runtime state cache ID; use the exact ID from werk cache list");
    }
    Ok(cache_entry(
        StateStore::new(home).cache_prune(namespace, state_id, dry_run)?,
    ))
}

fn cache_entry(entry: StoredCacheEntry) -> CacheEntry {
    CacheEntry {
        id: format!("runtime-state:{}:{}", entry.namespace, entry.id),
        kind: CacheKind::RuntimeState,
        backend: entry.backend,
        bytes: entry.bytes,
        modified_unix_seconds: entry.modified_unix_ms.map(|value| value / 1000),
        active: entry.active,
        blocked_reason: entry.blocked_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_cache_ids_are_rejected_before_catalog_access() {
        let home = Path::new("/missing-runtime-state-cache-fixture");
        for id in [
            "runtime-state:",
            "runtime-state:local",
            "runtime-state:local:",
            "runtime-state:local:st_a:extra",
            "runtime-state:../outside:st_a",
            "runtime-state:local:../outside",
            "runtime-state:local:st_*",
            "chat-kv:local:st_a",
        ] {
            assert!(purge(home, id, true).is_err(), "accepted {id}");
        }
    }
}
