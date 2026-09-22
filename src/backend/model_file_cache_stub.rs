//! Other platforms retain the operating system's normal file-cache policy.

use crate::model_store::{ModelManifest, ModelStore};
use std::{
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

pub(crate) struct CacheReleaseGuard;

impl CacheReleaseGuard {
    pub(crate) fn prepare(
        _model_path: &Path,
        _projector_path: Option<&Path>,
        _args: &[String],
    ) -> Option<Self> {
        None
    }

    pub(crate) fn prepare_paths(_paths: Vec<PathBuf>) -> Option<Self> {
        None
    }

    pub(crate) fn prepare_manifest(_store: &ModelStore, _manifest: &ModelManifest) -> Option<Self> {
        None
    }

    pub(crate) fn with_preparation<T>(&self, prepare: impl FnOnce(Option<&AtomicBool>) -> T) -> T {
        prepare(None)
    }
}
