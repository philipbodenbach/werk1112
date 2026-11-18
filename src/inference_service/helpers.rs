use anyhow::{Result, anyhow, bail};
use getrandom::getrandom;
use serde::Serialize;
use std::{fs, path::Path};

use crate::model_store::unix_ts;

pub(super) fn validate_safe_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || value
            .chars()
            .any(|ch| !(ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.')))
        || value == "."
        || value == ".."
    {
        bail!("unsafe storage identifier '{value}'");
    }
    Ok(())
}

/// Scoped IDs persist with the existing records, indexes and global output-retention budget.
/// The namespace comes only from the authenticated server principal. IDs remain opaque to clients.
#[derive(Clone, Debug, Default)]
pub(super) struct InferenceScope(Option<String>);

impl InferenceScope {
    pub fn for_principal(principal: &str) -> Self {
        use sha2::{Digest, Sha256};
        Self(Some(format!("{:x}", Sha256::digest(principal.as_bytes()))))
    }

    pub fn new_id(&self, kind: &str) -> Result<String> {
        match &self.0 {
            Some(scope) => new_id(&format!("{kind}-w1-{scope}")),
            None => new_id(kind),
        }
    }

    pub fn permits(&self, kind: &str, id: &str) -> bool {
        match &self.0 {
            Some(scope) => id.starts_with(&format!("{kind}-w1-{scope}-")),
            None => !id.to_ascii_lowercase().starts_with(&format!("{kind}-w1-")),
        }
    }
}

pub(super) fn new_id(prefix: &str) -> Result<String> {
    let mut random = [0_u8; 8];
    getrandom(&mut random).map_err(|error| anyhow!("failed to generate id: {error}"))?;
    let suffix = random
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("{prefix}-{}-{suffix}", unix_ts()))
}

pub(super) fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension(new_id("tmp")?);
    let data = serde_json::to_vec_pretty(value)?;
    fs::write(&temporary, data)?;
    fs::rename(&temporary, path)?;
    Ok(())
}
