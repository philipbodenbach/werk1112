//! Resident resources, keyed by the backend's existing model/configuration identity.
//! The registry only publishes slots. Loading is single-flight per slot; inference
//! owns an Arc and never holds the registry or initialization lock.
use anyhow::{Result, anyhow, bail};
use std::{
    cell::RefCell,
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

type Slot<V> = Arc<CacheSlot<V>>;

#[derive(Debug)]
struct CacheSlot<V> {
    value: Mutex<Option<Arc<V>>>,
    pins: AtomicUsize,
}
impl<V> CacheSlot<V> {
    fn new(value: Option<Arc<V>>) -> Self {
        Self {
            value: Mutex::new(value),
            pins: AtomicUsize::new(0),
        }
    }
}
impl<V> std::ops::Deref for CacheSlot<V> {
    type Target = Mutex<Option<Arc<V>>>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

struct CapturedSlot {
    id: usize,
    release: Option<Box<dyn FnOnce(bool) + Send + Sync>>,
}
impl Drop for CapturedSlot {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release(false);
        }
    }
}

// Explicit startup transaction. Backend loading and generation are synchronous;
// the scope follows that call only, never unrelated requests or other threads.
thread_local! {
    static PRELOAD: RefCell<Option<Vec<CapturedSlot>>> = const { RefCell::new(None) };
}

/// Keeps successful preload slots outside the ordinary cache capacity/eviction
/// policy. Owners also keep routed backend discovery identities reusable.
pub struct PreloadHandle {
    slots: Vec<CapturedSlot>,
    owners: Vec<Arc<dyn super::GenerationBackend>>,
}
impl PreloadHandle {
    pub(crate) fn with_owner(mut self, owner: Arc<dyn super::GenerationBackend>) -> Self {
        self.owners.push(owner);
        self
    }
}
impl Drop for PreloadHandle {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            if let Some(release) = slot.release.take() {
                release(true);
            }
        }
    }
}

pub(crate) fn preload(operation: impl FnOnce() -> Result<()>) -> Result<PreloadHandle> {
    anyhow::ensure!(
        !PRELOAD.with(|scope| scope.borrow().is_some()),
        "nested preload transaction"
    );
    PRELOAD.with(|scope| *scope.borrow_mut() = Some(Vec::new()));
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            let slots = PRELOAD.with(|scope| scope.borrow_mut().take());
            drop(slots); // rollback outside the RefCell borrow, including on panic
        }
    }
    let _reset = Reset;
    operation()?;
    anyhow::ensure!(
        PRELOAD.with(|scope| !scope.borrow().as_ref().unwrap().is_empty()),
        "backend did not retain a resident model during warmup"
    );
    let slots = PRELOAD.with(|scope| scope.borrow_mut().take().unwrap());
    Ok(PreloadHandle {
        slots,
        owners: Vec::new(),
    })
}

fn capture<V: Send + Sync + 'static>(slot: &Slot<V>, loaded: &Arc<V>, fresh: bool) {
    PRELOAD.with(|scope| {
        let mut scope = scope.borrow_mut();
        let Some(slots) = scope.as_mut() else { return };
        let id = Arc::as_ptr(slot) as usize;
        if slots.iter().any(|s| s.id == id) {
            return;
        }
        slot.pins.fetch_add(1, Ordering::Relaxed);
        let slot = slot.clone();
        let loaded = Arc::downgrade(loaded);
        slots.push(CapturedSlot {
            id,
            release: Some(Box::new(move |committed| {
                // Only remove the resource created by this failed attempt. A cache
                // hit, or a subsequently replaced worker, belongs to its old owner.
                let removed = if !committed && fresh {
                    let mut value = slot.lock().unwrap_or_else(|e| e.into_inner());
                    if value
                        .as_ref()
                        .is_some_and(|v| loaded.ptr_eq(&Arc::downgrade(v)))
                    {
                        value.take()
                    } else {
                        None
                    }
                } else {
                    None
                };
                slot.pins.fetch_sub(1, Ordering::Relaxed);
                drop(removed);
            })),
        });
    });
}

#[derive(Debug)]
pub(crate) struct RuntimeCache<K, V> {
    entries: Mutex<VecDeque<(K, Slot<V>)>>,
    capacity: Option<usize>,
    evict_idle: bool,
}

impl<K, V> Default for RuntimeCache<K, V> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
            capacity: None,
            evict_idle: true,
        }
    }
}

impl<K: Eq, V: Send + Sync + 'static> RuntimeCache<K, V> {
    pub(crate) fn bounded(capacity: usize) -> Self {
        Self {
            capacity: Some(capacity.max(1)),
            ..Self::default()
        }
    }

    pub(crate) fn retained(capacity: usize) -> Self {
        Self {
            evict_idle: false,
            ..Self::bounded(capacity)
        }
    }

    #[cfg(test)]
    pub(crate) fn insert_fixture(&self, key: K, value: Arc<V>) {
        self.entries
            .lock()
            .unwrap()
            .push_back((key, Arc::new(CacheSlot::new(Some(value)))));
    }

    pub(crate) fn get_or_try_init(
        &self,
        key: K,
        valid: impl FnOnce(&V) -> bool,
        load: impl FnOnce() -> Result<V>,
    ) -> Result<(Arc<V>, bool, f64)> {
        let (slot, evicted) = {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| anyhow!("runtime registry poisoned"))?;
            if let Some(index) = entries.iter().position(|(candidate, _)| candidate == &key) {
                let entry = entries.remove(index).expect("located runtime slot");
                let slot = entry.1.clone();
                entries.push_back(entry);
                (slot, None)
            } else {
                // Failed loads carry no resident resource and must not consume
                // the capacity forever. Waiters retain their initialization slot.
                entries.retain(|(_, slot)| {
                    Arc::strong_count(slot) > 1
                        || !slot.try_lock().is_ok_and(|value| value.is_none())
                });
                let ordinary_entries = entries
                    .iter()
                    .filter(|(_, slot)| slot.pins.load(Ordering::Relaxed) == 0)
                    .count();
                let evicted = if !PRELOAD.with(|scope| scope.borrow().is_some())
                    && self
                        .capacity
                        .is_some_and(|capacity| ordinary_entries >= capacity)
                {
                    if !self.evict_idle {
                        bail!(
                            "runtime has reached its limit of {} retained workers",
                            self.capacity.unwrap()
                        );
                    }
                    // A caller holds the slot during initialization and the value
                    // during inference. Never evict either or duplicate active weights.
                    let index = entries.iter().position(|(_, slot)| {
                        Arc::strong_count(slot) == 1
                            && slot.try_lock().is_ok_and(|value| {
                                value
                                    .as_ref()
                                    .is_none_or(|value| Arc::strong_count(value) == 1)
                            })
                    });
                    let Some(index) = index else {
                        bail!(
                            "resident runtime capacity reached; all entries are in use; retry when a request finishes or increase the backend model cache capacity"
                        );
                    };
                    entries.remove(index)
                } else {
                    None
                };
                let slot = Arc::new(CacheSlot::new(None));
                entries.push_back((key, slot.clone()));
                (slot, evicted)
            }
        };
        // Destructors may stop/reap a child. Never run them under the registry.
        drop(evicted);
        let mut value = slot
            .lock()
            .map_err(|_| anyhow!("runtime initialization poisoned"))?;
        if let Some(existing) = value.as_ref().filter(|existing| valid(existing)) {
            capture(&slot, existing, false);
            return Ok((existing.clone(), true, 0.0));
        }
        // Release a failed instance before allocating its replacement. Ordinary
        // errors leave this slot retryable and do not affect any other key.
        *value = None;
        let started = Instant::now();
        let loaded = Arc::new(load()?);
        let seconds = started.elapsed().as_secs_f64();
        *value = Some(loaded.clone());
        capture(&slot, &loaded, true);
        Ok((loaded, false, seconds))
    }

    /// Monitoring/control must not wait for a model being loaded.
    pub(crate) fn snapshot(&self) -> Vec<Arc<V>> {
        let slots: Vec<_> = self
            .entries
            .lock()
            .map(|entries| entries.iter().map(|(_, slot)| slot.clone()).collect())
            .unwrap_or_default();
        slots
            .into_iter()
            .filter_map(|slot| slot.try_lock().ok()?.clone())
            .collect()
    }

    pub(crate) fn remove_if(&self, predicate: impl Fn(&V) -> bool) {
        let removed = {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let mut removed = Vec::new();
            let mut index = 0;
            while index < entries.len() {
                let remove = Arc::strong_count(&entries[index].1) == 1
                    && entries[index]
                        .1
                        .try_lock()
                        .is_ok_and(|value| value.as_ref().is_some_and(|value| predicate(value)));
                if remove {
                    removed.push(entries.remove(index));
                } else {
                    index += 1;
                }
            }
            removed
        };
        drop(removed);
    }
}

/// Short-lived per-key creation gates for caches with their own retention policy.
pub(crate) struct KeyedLocks<K>(Mutex<Vec<(K, std::sync::Weak<Mutex<()>>)>>);
impl<K: Eq> KeyedLocks<K> {
    pub(crate) fn get(&self, key: K) -> Result<Arc<Mutex<()>>> {
        let mut gates = self
            .0
            .lock()
            .map_err(|_| anyhow!("creation gate registry poisoned"))?;
        gates.retain(|(_, gate)| gate.strong_count() > 0);
        if let Some(gate) = gates
            .iter()
            .find(|(candidate, _)| candidate == &key)
            .and_then(|(_, gate)| gate.upgrade())
        {
            return Ok(gate);
        }
        let gate = Arc::new(Mutex::new(()));
        gates.push((key, Arc::downgrade(&gate)));
        Ok(gate)
    }
}

impl<K> Default for KeyedLocks<K> {
    fn default() -> Self {
        Self(Mutex::new(Vec::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Barrier,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    };
    use std::{thread, time::Duration};

    #[test]
    fn preload_has_no_count_limit_and_preserves_ordinary_capacity() {
        for cache in [RuntimeCache::bounded(2), RuntimeCache::retained(2)] {
            let handles: Vec<_> = (0..100)
                .map(|id| {
                    preload(|| {
                        cache.get_or_try_init(id, |_| true, || Ok(id))?;
                        Ok(())
                    })
                    .unwrap()
                })
                .collect();
            assert_eq!(cache.snapshot().len(), 100);
            // Even explicit cleanup cannot evict pinned slots.
            cache.remove_if(|_| true);
            assert_eq!(cache.snapshot().len(), 100);
            let a = cache.get_or_try_init(100, |_| true, || Ok(100)).unwrap().0;
            let b = cache.get_or_try_init(101, |_| true, || Ok(101)).unwrap().0;
            assert!(
                cache
                    .get_or_try_init(102, |_| true, || panic!("ordinary capacity exceeded"))
                    .is_err()
            );
            for id in 0..100 {
                assert_eq!(
                    *cache
                        .get_or_try_init(id, |_| true, || panic!("preloaded model evicted"))
                        .unwrap()
                        .0,
                    id
                );
            }
            drop((a, b, handles));
            cache.remove_if(|_| true);
            assert!(cache.snapshot().is_empty());
        }
    }

    #[test]
    fn failed_preload_rolls_back_only_new_resources_and_releases_capacity() {
        let cache = RuntimeCache::retained(1);
        let existing = cache
            .get_or_try_init("existing", |_| true, || Ok(1))
            .unwrap()
            .0;
        let failed = preload(|| {
            cache.get_or_try_init("existing", |_| true, || panic!("cache hit"))?;
            cache.get_or_try_init("failed", |_| true, || Ok(2))?;
            // Accessing a slot twice must not overwrite its rollback ownership.
            cache.get_or_try_init("failed", |_| true, || panic!("cache hit"))?;
            anyhow::bail!("warmup failed")
        });
        assert!(failed.is_err());
        assert_eq!(cache.snapshot().len(), 1);
        assert!(Arc::ptr_eq(&existing, &cache.snapshot()[0]));
        let handle = preload(|| {
            cache.get_or_try_init("next", |_| true, || Ok(3))?;
            Ok(())
        })
        .unwrap();
        assert_eq!(cache.snapshot().len(), 2);
        drop(handle);
    }

    #[test]
    fn preload_scope_is_thread_local_and_unwinds_without_leaking() {
        let cache = Arc::new(RuntimeCache::retained(1));
        cache.get_or_try_init(0, |_| true, || Ok(0)).unwrap();
        let failure = std::panic::catch_unwind(|| {
            let _ = preload(|| {
                let other = cache.clone();
                thread::spawn(move || {
                    assert!(
                        other
                            .get_or_try_init(1, |_| true, || panic!("must respect capacity"))
                            .is_err()
                    )
                })
                .join()
                .unwrap();
                cache.get_or_try_init(2, |_| true, || Ok(2))?;
                panic!("warmup panic")
            });
        });
        assert!(failure.is_err());
        assert_eq!(cache.snapshot().len(), 1);
        assert!(!PRELOAD.with(|scope| scope.borrow().is_some()));
        assert!(preload(|| Ok(())).is_err());
    }

    #[test]
    fn concurrent_cold_requests_load_exactly_one_runtime() {
        let cache = Arc::new(RuntimeCache::default());
        let starts = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(6));
        let threads: Vec<_> = (0..6)
            .map(|_| {
                let (cache, starts, barrier) = (cache.clone(), starts.clone(), barrier.clone());
                thread::spawn(move || {
                    barrier.wait();
                    cache
                        .get_or_try_init(
                            "model",
                            |_| true,
                            || {
                                starts.fetch_add(1, Ordering::SeqCst);
                                Ok(Mutex::new(()))
                            },
                        )
                        .unwrap()
                        .0
                })
            })
            .collect();
        let models: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert!(models.iter().all(|model| Arc::ptr_eq(model, &models[0])));
    }

    #[test]
    fn unrelated_initialization_and_inference_do_not_share_locks() {
        let cache = Arc::new(RuntimeCache::default());
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let first_cache = cache.clone();
        let first = thread::spawn(move || {
            first_cache
                .get_or_try_init(
                    "a",
                    |_| true,
                    || {
                        entered_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                        Ok(Mutex::new(()))
                    },
                )
                .unwrap()
                .0
        });
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(cache.snapshot().is_empty());
        let (ready_tx, ready_rx) = mpsc::channel();
        let other_cache = cache.clone();
        let second = thread::spawn(move || {
            let model = other_cache
                .get_or_try_init("b", |_| true, || Ok(Mutex::new(())))
                .unwrap()
                .0;
            ready_tx.send(model).unwrap();
        });
        let result = ready_rx.recv_timeout(Duration::from_secs(5));
        release_tx.send(()).unwrap();
        let a = first.join().unwrap();
        second.join().unwrap();
        let b = result.expect("model b must load while a is blocked");
        let _a = a.lock().unwrap();
        let same_a = cache
            .get_or_try_init("a", |_| true, || panic!("duplicate"))
            .unwrap()
            .0;
        assert!(same_a.try_lock().is_err());
        assert!(b.try_lock().is_ok());
    }

    #[test]
    fn failure_is_retryable_and_does_not_poison_other_models_or_capacity() {
        let cache = RuntimeCache::retained(2);
        let good = cache
            .get_or_try_init("good", |_| true, || Ok(42))
            .unwrap()
            .0;
        for _ in 0..3 {
            assert!(
                cache
                    .get_or_try_init("bad", |_| true, || anyhow::bail!("load failed"))
                    .is_err()
            );
        }
        let recovered = cache
            .get_or_try_init("recovered", |_| true, || Ok(7))
            .unwrap()
            .0;
        assert_eq!(*recovered, 7);
        assert!(Arc::ptr_eq(
            &good,
            &cache
                .get_or_try_init("good", |_| true, || panic!("duplicate"))
                .unwrap()
                .0
        ));
    }

    #[test]
    fn dead_runtime_replacement_is_single_flight() {
        let cache = Arc::new(RuntimeCache::default());
        let old = cache
            .get_or_try_init("a", |_| true, || Ok(AtomicBool::new(true)))
            .unwrap()
            .0;
        old.store(false, Ordering::SeqCst);
        let starts = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(4));
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let (cache, starts, barrier) = (cache.clone(), starts.clone(), barrier.clone());
                thread::spawn(move || {
                    barrier.wait();
                    cache
                        .get_or_try_init(
                            "a",
                            |alive| alive.load(Ordering::SeqCst),
                            || {
                                starts.fetch_add(1, Ordering::SeqCst);
                                Ok(AtomicBool::new(true))
                            },
                        )
                        .unwrap()
                        .0
                })
            })
            .collect();
        let models: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert!(!Arc::ptr_eq(&old, &models[0]));
        assert!(models.iter().all(|model| Arc::ptr_eq(model, &models[0])));
    }

    #[test]
    fn bounded_cache_never_evicts_active_resources_and_drops_idle_lru() {
        struct Runtime(Arc<AtomicUsize>);
        impl Drop for Runtime {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let cache = RuntimeCache::bounded(1);
        let a = cache
            .get_or_try_init("a", |_| true, || Ok(Runtime(drops.clone())))
            .unwrap()
            .0;
        assert!(
            cache
                .get_or_try_init("b", |_| true, || panic!("must not allocate over capacity"))
                .is_err()
        );
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(a);
        let b = cache
            .get_or_try_init("b", |_| true, || Ok(Runtime(drops.clone())))
            .unwrap()
            .0;
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(cache);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(b);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn panicking_loader_is_confined_to_its_key() {
        let cache = Arc::new(RuntimeCache::<&str, u32>::default());
        let failing = cache.clone();
        assert!(
            thread::spawn(move || failing.get_or_try_init(
                "bad",
                |_| true,
                || panic!("native panic")
            ))
            .join()
            .is_err()
        );
        assert_eq!(
            *cache
                .get_or_try_init("good", |_| true, || Ok(42))
                .unwrap()
                .0,
            42
        );
    }
}
