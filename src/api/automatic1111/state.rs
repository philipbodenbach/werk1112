use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::Mutex as AsyncMutex;

use crate::model_store::unix_ts;

pub(in crate::api) struct Automatic1111State {
    pub(super) generation_lock: Arc<AsyncMutex<()>>,
    selected_checkpoint: Mutex<Option<String>>,
    progress: Mutex<ProgressSnapshot>,
}

impl Default for Automatic1111State {
    fn default() -> Self {
        Self {
            generation_lock: Arc::new(AsyncMutex::new(())),
            selected_checkpoint: Mutex::new(None),
            progress: Mutex::new(ProgressSnapshot::default()),
        }
    }
}

impl Automatic1111State {
    pub(in crate::api) fn set_selected_checkpoint(&self, checkpoint: Option<String>) {
        *lock_unpoisoned(&self.selected_checkpoint) = checkpoint;
    }

    pub(super) fn selected_checkpoint(&self) -> Option<String> {
        lock_unpoisoned(&self.selected_checkpoint).clone()
    }

    pub(super) fn begin(self: &Arc<Self>, steps: u32) -> ActiveGeneration {
        *lock_unpoisoned(&self.progress) = ProgressSnapshot {
            active: true,
            started_unix: unix_ts(),
            steps,
        };
        ActiveGeneration {
            state: Arc::clone(self),
        }
    }

    pub(super) fn progress(&self) -> ProgressSnapshot {
        lock_unpoisoned(&self.progress).clone()
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(super) struct ActiveGeneration {
    state: Arc<Automatic1111State>,
}

impl Drop for ActiveGeneration {
    fn drop(&mut self) {
        *lock_unpoisoned(&self.state.progress) = ProgressSnapshot::default();
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct ProgressSnapshot {
    pub(super) active: bool,
    pub(super) started_unix: u64,
    pub(super) steps: u32,
}
