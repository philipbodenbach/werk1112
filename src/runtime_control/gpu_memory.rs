//! Demand-driven GPU telemetry with one NVML connection per server process.
//!
//! Repeated short-lived nvidia-smi clients can accumulate Windows driver
//! allocations under WSL. Keep the library, device and sampling thread alive;
//! never fall back to spawning a process for each dashboard refresh.
use libloading::Library;
use std::{
    ffi::{c_uint, c_void},
    sync::{Arc, Condvar, Mutex, OnceLock, mpsc},
    time::{Duration, Instant},
};

type Memory = (u64, u64);

pub(super) fn observe() -> Option<Memory> {
    static SAMPLER: OnceLock<Option<Sampler>> = OnceLock::new();
    SAMPLER
        .get_or_init(|| {
            Sampler::start(
                || {
                    let session = Nvml::open()?;
                    Some(move || session.memory())
                },
                Policy::default(),
            )
        })
        .as_ref()?
        .observe()
}

#[derive(Clone, Copy)]
struct Policy {
    fresh_for: Duration,
    retry_after: Duration,
    wait: Duration,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            fresh_for: Duration::from_secs(2),
            retry_after: Duration::from_secs(30),
            wait: Duration::from_secs(2),
        }
    }
}

#[derive(Default)]
struct State {
    sample: Option<(Instant, Option<Memory>)>,
    running: bool,
    closed: bool,
}

type Shared = Arc<(Mutex<State>, Condvar)>;

struct Sampler {
    request: mpsc::SyncSender<()>,
    shared: Shared,
    policy: Policy,
}

impl Sampler {
    fn start<I, Q>(initialize: I, policy: Policy) -> Option<Self>
    where
        I: FnOnce() -> Option<Q> + Send + 'static,
        Q: FnMut() -> Option<Memory> + 'static,
    {
        let shared = Shared::default();
        let worker = shared.clone();
        let (request, requests) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("werk-gpu-memory".into())
            .spawn(move || {
                let _exit = WorkerExit(worker.clone());
                // Initialize only on demand, on this same thread. Neither a
                // blocked init nor a blocked query can cause more workers.
                if requests.recv().is_err() {
                    return;
                }
                let Some(mut query) = initialize() else {
                    return;
                };
                loop {
                    let value = query();
                    let mut state = worker.0.lock().unwrap_or_else(|e| e.into_inner());
                    state.sample = Some((Instant::now(), value));
                    state.running = false;
                    worker.1.notify_all();
                    drop(state);
                    if requests.recv().is_err() {
                        return;
                    }
                }
            })
            .ok()?;
        Some(Self {
            request,
            shared,
            policy,
        })
    }

    fn observe(&self) -> Option<Memory> {
        let mut state = self.shared.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed {
            return None;
        }
        if let Some((at, value)) = state.sample {
            let ttl = if value.is_some() {
                self.policy.fresh_for
            } else {
                self.policy.retry_after
            };
            if at.elapsed() < ttl {
                return value;
            }
        }
        if state.running {
            return None;
        }
        state.running = true;
        if self.request.try_send(()).is_err() {
            state.running = false;
            state.closed = true;
            return None;
        }
        let (state, _) = self
            .shared
            .1
            .wait_timeout_while(state, self.policy.wait, |s| s.running && !s.closed)
            .unwrap_or_else(|e| e.into_inner());
        // Never publish expired data as fresh after a driver timeout/failure.
        if state.running || state.closed {
            None
        } else {
            state.sample.and_then(|(_, value)| value)
        }
    }
}

struct WorkerExit(Shared);

impl Drop for WorkerExit {
    fn drop(&mut self) {
        let mut state = self.0.0.lock().unwrap_or_else(|e| e.into_inner());
        state.closed = true;
        state.running = false;
        self.0.1.notify_all();
    }
}

// Stable NVML v1 memory ABI from nvml.h; values are bytes, not MiB.
#[repr(C)]
#[derive(Default)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

type Init = unsafe extern "C" fn() -> c_uint;
type Shutdown = unsafe extern "C" fn() -> c_uint;
type Device = unsafe extern "C" fn(c_uint, *mut *mut c_void) -> c_uint;
type MemoryInfo = unsafe extern "C" fn(*mut c_void, *mut NvmlMemory) -> c_uint;

struct Nvml {
    // The library outlives every function pointer and the device handle.
    _library: Library,
    shutdown: Shutdown,
    memory_info: MemoryInfo,
    device: *mut c_void,
}

impl Nvml {
    fn open() -> Option<Self> {
        let library = load_library()?;
        // SAFETY: these exported NVML symbols have the C signatures above.
        // Resolve everything before init, so early symbol errors need no cleanup.
        let (init, shutdown, device, memory_info) = unsafe {
            (
                *library.get::<Init>(b"nvmlInit_v2\0").ok()?,
                *library.get::<Shutdown>(b"nvmlShutdown\0").ok()?,
                *library
                    .get::<Device>(b"nvmlDeviceGetHandleByIndex_v2\0")
                    .ok()?,
                *library
                    .get::<MemoryInfo>(b"nvmlDeviceGetMemoryInfo\0")
                    .ok()?,
            )
        };
        // SAFETY: all calls stay on the sampler thread while the library is loaded.
        if unsafe { init() } != 0 {
            return None;
        }
        let mut session = Self {
            _library: library,
            shutdown,
            memory_info,
            device: std::ptr::null_mut(),
        };
        // Preserve the existing telemetry scope: the first physical GPU.
        // SAFETY: session.device is valid writable storage for the opaque handle.
        if unsafe { device(0, &mut session.device) } != 0 || session.device.is_null() {
            return None; // Drop balances the successful init, including on error.
        }
        Some(session)
    }

    fn memory(&self) -> Option<Memory> {
        let mut memory = NvmlMemory::default();
        // SAFETY: a live device and a correctly sized nvmlMemory_t are supplied.
        let status = unsafe { (self.memory_info)(self.device, &mut memory) };
        (status == 0 && memory.total > 0 && memory.free <= memory.total)
            .then_some((memory.total, memory.free))
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: each successfully initialized session shuts down exactly once,
        // on its owning thread and before unloading the library.
        unsafe { (self.shutdown)() };
    }
}

fn load_library() -> Option<Library> {
    #[cfg(target_os = "linux")]
    let candidates = [
        std::path::PathBuf::from("libnvidia-ml.so.1"),
        std::path::PathBuf::from("/usr/lib/wsl/lib/libnvidia-ml.so.1"),
    ];
    #[cfg(target_os = "windows")]
    let candidates = [
        std::path::PathBuf::from(std::env::var_os("SystemRoot")?)
            .join("System32")
            .join("nvml.dll"),
        std::path::PathBuf::from(std::env::var_os("ProgramFiles")?)
            .join("NVIDIA Corporation")
            .join("NVSMI")
            .join("nvml.dll"),
    ];
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    let candidates: [std::path::PathBuf; 0] = [];
    // SAFETY: only the system NVIDIA driver library is loaded. Its C entry
    // points are resolved and used according to the NVML ABI in Nvml::open.
    candidates
        .iter()
        .find_map(|path| unsafe { Library::new(path).ok() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn policy() -> Policy {
        Policy {
            fresh_for: Duration::ZERO,
            retry_after: Duration::from_secs(60),
            wait: Duration::from_secs(1),
        }
    }

    fn expire(sampler: &Sampler) {
        sampler.shared.0.lock().unwrap().sample.as_mut().unwrap().0 =
            Instant::now() - Duration::from_secs(120);
    }

    #[test]
    fn observability_gpu_reuses_one_session_and_cleans_up_once() {
        struct Cleanup(mpsc::Sender<()>);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }
        let inits = Arc::new(AtomicUsize::new(0));
        let starts = inits.clone();
        let (done, cleaned) = mpsc::channel();
        let sampler = Sampler::start(
            move || {
                starts.fetch_add(1, Ordering::SeqCst);
                let guard = Cleanup(done);
                let mut queries = 0;
                Some(move || {
                    let _keep = &guard;
                    queries += 1;
                    Some((1000, queries))
                })
            },
            policy(),
        )
        .unwrap();
        for query in 1..=100 {
            assert_eq!(sampler.observe(), Some((1000, query)));
        }
        assert_eq!(inits.load(Ordering::SeqCst), 1);
        drop(sampler);
        cleaned.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(cleaned.try_recv().is_err());
    }

    #[test]
    fn observability_gpu_caches_reads_and_does_no_work_without_demand() {
        let calls = Arc::new(AtomicUsize::new(0));
        let count = calls.clone();
        let sampler = Sampler::start(
            move || Some(move || Some((1000, count.fetch_add(1, Ordering::SeqCst) as u64))),
            Policy {
                fresh_for: Duration::from_secs(60),
                ..policy()
            },
        )
        .unwrap();
        for _ in 0..50 {
            assert_eq!(sampler.observe(), Some((1000, 0)));
        }
        expire(&sampler);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(sampler.observe(), Some((1000, 1)));
    }

    #[test]
    fn observability_gpu_failure_backs_off_without_reopening_or_stale_data() {
        let sampler = Sampler::start(
            || {
                let mut calls = 0;
                Some(move || {
                    calls += 1;
                    match calls {
                        1 => Some((1000, 900)),
                        2 => None,
                        _ => Some((1000, 800)),
                    }
                })
            },
            policy(),
        )
        .unwrap();
        assert_eq!(sampler.observe(), Some((1000, 900)));
        for _ in 0..20 {
            assert_eq!(sampler.observe(), None);
        }
        expire(&sampler);
        assert_eq!(sampler.observe(), Some((1000, 800)));
    }

    #[test]
    fn observability_gpu_missing_driver_is_not_repeatedly_initialized() {
        let inits = Arc::new(AtomicUsize::new(0));
        let starts = inits.clone();
        let sampler = Sampler::start(
            move || {
                starts.fetch_add(1, Ordering::SeqCst);
                None::<fn() -> Option<Memory>>
            },
            policy(),
        )
        .unwrap();
        for _ in 0..20 {
            assert_eq!(sampler.observe(), None);
        }
        assert_eq!(inits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn observability_gpu_hung_init_or_query_keeps_a_single_bounded_worker() {
        for block_init in [true, false] {
            let (release, blocked) = mpsc::channel();
            let sampler = Sampler::start(
                move || {
                    if block_init {
                        blocked.recv().unwrap();
                    }
                    Some(move || {
                        if !block_init {
                            blocked.recv().unwrap();
                        }
                        Some((1000, 900))
                    })
                },
                Policy {
                    wait: Duration::from_millis(20),
                    ..policy()
                },
            )
            .unwrap();
            let start = Instant::now();
            for _ in 0..100 {
                assert_eq!(sampler.observe(), None);
            }
            assert!(start.elapsed() < Duration::from_secs(1));
            assert!(sampler.shared.0.lock().unwrap().running);
            // Dropping the owner must never join a stuck driver thread.
            drop(sampler);
            release.send(()).unwrap();
        }
    }

    #[test]
    #[ignore = "requires an NVIDIA driver; samples one live connection for 40 seconds"]
    fn observability_gpu_live_nvml_connection() {
        for _ in 0..20 {
            let (total, free) = observe().expect("NVML reports available memory");
            assert!(total > 0 && free <= total);
            println!("gpu memory: total={total}, free={free}");
            std::thread::sleep(Duration::from_secs(2));
        }
    }
}
