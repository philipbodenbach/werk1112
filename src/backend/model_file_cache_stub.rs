//! Other platforms retain the operating system's normal file-cache policy.

use crate::model_store::{ModelManifest, ModelStore};
use std::path::PathBuf;

pub(crate) struct CacheReleaseGuard;

impl CacheReleaseGuard {
    pub(crate) fn prepare_paths(_paths: Vec<PathBuf>) -> Option<Self> {
        None
    }

    pub(crate) fn prepare_manifest(_store: &ModelStore, _manifest: &ModelManifest) -> Option<Self> {
        None
    }
}
