//! Ownership and interruption cleanup for Werk's inference worker processes.
use std::{
    ops::Deref,
    process::{Child, Command},
    sync::{Arc, Mutex, OnceLock, Weak},
};

#[derive(Default)]
struct Children {
    stopping: bool,
    children: Vec<Weak<Mutex<Child>>>,
    cleanups: Vec<Weak<Mutex<Option<CleanupCallback>>>>,
}

type CleanupCallback = Box<dyn FnOnce() + Send + 'static>;

fn children() -> &'static Mutex<Children> {
    static CHILDREN: OnceLock<Mutex<Children>> = OnceLock::new();
    CHILDREN.get_or_init(Default::default)
}

pub(crate) struct ManagedChild(Arc<Mutex<Child>>);

/// Register before preparing resources that must also be released if startup
/// fails or a signal interrupts preparation. The owner must outlive its worker;
/// normal destruction must reap that worker before dropping this token.
pub(crate) struct ManagedCleanup(Arc<Mutex<Option<CleanupCallback>>>);

impl ManagedCleanup {
    pub(crate) fn register(callback: CleanupCallback) -> std::io::Result<Self> {
        let mut registry = children().lock().unwrap_or_else(|e| e.into_inner());
        if registry.stopping {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "Werk is shutting down",
            ));
        }
        let cleanup = Arc::new(Mutex::new(Some(callback)));
        registry.cleanups.retain(|entry| entry.strong_count() > 0);
        registry.cleanups.push(Arc::downgrade(&cleanup));
        Ok(Self(cleanup))
    }
}

fn run_cleanup(cleanup: &Mutex<Option<CleanupCallback>>) {
    // Keep the mutex through execution: a racing signal/Drop must wait for the
    // callback to finish, not merely observe that another caller took it.
    let mut callback = cleanup.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(callback) = callback.take() {
        callback();
    }
}

impl Drop for ManagedCleanup {
    fn drop(&mut self) {
        run_cleanup(&self.0);
    }
}

impl ManagedChild {
    pub(crate) fn spawn(command: &mut Command) -> std::io::Result<Self> {
        // Serialize registration with shutdown: no child can escape between
        // spawning and becoming visible to the interruption handler.
        let mut registry = children().lock().unwrap_or_else(|e| e.into_inner());
        if registry.stopping {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "Werk is shutting down",
            ));
        }
        let child = Arc::new(Mutex::new(command.spawn()?));
        registry.children.retain(|child| child.strong_count() > 0);
        registry.children.push(Arc::downgrade(&child));
        Ok(Self(child))
    }
}

impl Deref for ManagedChild {
    type Target = Mutex<Child>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        let mut child = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn shutdown_children() {
    let (active, cleanups) = {
        let mut registry = children().lock().unwrap_or_else(|e| e.into_inner());
        registry.stopping = true;
        let active = registry
            .children
            .drain(..)
            .filter_map(|c| c.upgrade())
            .collect::<Vec<_>>();
        let cleanups = registry
            .cleanups
            .drain(..)
            .filter_map(|cleanup| cleanup.upgrade())
            .collect::<Vec<_>>();
        (active, cleanups)
    };
    // Stop all owned workers first, then reap them before exiting Werk.
    for child in &active {
        let _ = child.lock().unwrap_or_else(|e| e.into_inner()).kill();
    }
    for child in &active {
        let _ = child.lock().unwrap_or_else(|e| e.into_inner()).wait();
    }
    // Every wait has been attempted before cleanup. Callbacks are advisory and
    // must tolerate an unsuccessful wait; they cannot assume exclusive file
    // ownership. No registry lock is held while callbacks cancel/join work.
    for cleanup in cleanups {
        run_cleanup(&cleanup);
    }
}

/// Install only at the CLI boundary, never for library users. The listener is
/// a separate task so blocking model loading cannot prevent interruption.
pub(crate) fn install_shutdown_handler() -> std::io::Result<tokio::task::JoinHandle<()>> {
    #[cfg(unix)]
    let wait = {
        use tokio::signal::unix::{SignalKind, signal};
        let mut interrupt = signal(SignalKind::interrupt())?;
        let mut terminate = signal(SignalKind::terminate())?;
        let mut hangup = signal(SignalKind::hangup())?;
        async move {
            tokio::select! {
                _ = interrupt.recv() => 130,
                _ = terminate.recv() => 143,
                _ = hangup.recv() => 129,
            }
        }
    };
    #[cfg(windows)]
    let wait = {
        let mut interrupt = tokio::signal::windows::ctrl_c()?;
        let mut ctrl_break = tokio::signal::windows::ctrl_break()?;
        async move {
            tokio::select! {
                _ = interrupt.recv() => 130,
                _ = ctrl_break.recv() => 130,
            }
        }
    };
    Ok(tokio::spawn(async move {
        let status = wait.await;
        shutdown_children();
        std::process::exit(status);
    }))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        process::Stdio,
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn interruption_fixture() {
        let Some(ready) = std::env::var_os("WERK_TEST_CHILD_READY") else {
            return;
        };
        let ready = PathBuf::from(ready);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let _listener = install_shutdown_handler().unwrap();
            let cleanup_ready = ready.clone();
            let _cleanup = ManagedCleanup::register(Box::new(move || {
                let pid = fs::read_to_string(&cleanup_ready)
                    .unwrap()
                    .parse::<i32>()
                    .unwrap();
                let gone = unsafe { libc::kill(pid, 0) } == -1
                    && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
                fs::write(
                    cleanup_ready.with_extension("cleanup"),
                    if gone { "reaped" } else { "still running" },
                )
                .unwrap();
            }))
            .unwrap();
            let _child = ManagedChild::spawn(Command::new("sh")
                .args(["-c", "trap '' INT TERM; printf '%s' $$ > \"$WERK_TEST_CHILD_READY\"; exec sleep 60"])
                .env("WERK_TEST_CHILD_READY", &ready)
                .stdout(Stdio::null()).stderr(Stdio::null())).unwrap();
            // Simulate synchronous loading/inference that never yields.
            thread::sleep(Duration::from_secs(60));
        });
    }

    fn assert_interruption_reaps_child(signal: i32, expected_exit: i32) {
        let ready = std::env::temp_dir().join(format!(
            "werk-child-ready-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut parent = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "backend::llama_process_lifecycle::tests::interruption_fixture",
            ])
            .env("WERK_TEST_CHILD_READY", &ready)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let started = Instant::now();
        let child_pid = loop {
            if let Some(pid) = fs::read_to_string(&ready)
                .ok()
                .and_then(|s| s.parse::<i32>().ok())
            {
                break pid;
            }
            if started.elapsed() > Duration::from_secs(10) {
                let _ = parent.kill();
                let _ = parent.wait();
                panic!("interruption fixture did not become ready");
            }
            thread::sleep(Duration::from_millis(10));
        };
        // Only the Werk-like parent receives the signal. Its child ignores
        // graceful termination, so the test proves explicit owned-child cleanup.
        assert_eq!(unsafe { libc::kill(parent.id() as i32, signal) }, 0);
        let status = loop {
            if let Some(status) = parent.try_wait().unwrap() {
                break status;
            }
            if started.elapsed() > Duration::from_secs(15) {
                let _ = parent.kill();
                let _ = parent.wait();
                unsafe {
                    libc::kill(child_pid, libc::SIGKILL);
                }
                panic!("interruption did not stop parent");
            }
            thread::sleep(Duration::from_millis(10));
        };
        let child_gone = unsafe { libc::kill(child_pid, 0) } == -1
            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
        if !child_gone {
            unsafe {
                libc::kill(child_pid, libc::SIGKILL);
            }
        }
        let cleanup = fs::read_to_string(ready.with_extension("cleanup"));
        let _ = fs::remove_file(&ready);
        let _ = fs::remove_file(ready.with_extension("cleanup"));
        assert_eq!(status.code(), Some(expected_exit));
        assert!(child_gone, "child must be reaped before parent exits");
        assert_eq!(
            cleanup.unwrap(),
            "reaped",
            "cleanup must observe the reaped child and complete before parent exit"
        );
    }

    #[test]
    fn interrupt_reaps_owned_worker_even_during_blocking_work() {
        assert_interruption_reaps_child(libc::SIGINT, 130);
    }

    #[test]
    fn termination_reaps_owned_worker_even_during_blocking_work() {
        assert_interruption_reaps_child(libc::SIGTERM, 143);
    }

    #[test]
    fn normal_drop_reaps_worker_without_killing_unrelated_process() {
        let child = ManagedChild::spawn(Command::new("sleep").arg("60")).unwrap();
        let pid = child.lock().unwrap().id();
        let mut unrelated = Command::new("sleep").arg("60").spawn().unwrap();
        drop(child);
        let gone = unsafe { libc::kill(pid as i32, 0) } == -1
            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
        let unrelated_running = unrelated.try_wait().unwrap().is_none();
        let _ = unrelated.kill();
        let _ = unrelated.wait();
        assert!(gone);
        assert!(unrelated_running);
    }

    #[test]
    fn normal_drop_runs_registered_cleanup_once() {
        let count = Arc::new(AtomicUsize::new(0));
        let called = count.clone();
        let cleanup = ManagedCleanup::register(Box::new(move || {
            called.fetch_add(1, Ordering::SeqCst);
        }))
        .unwrap();
        let registered = cleanup.0.clone();
        drop(cleanup);
        run_cleanup(&registered);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn racing_cleanup_waits_for_the_in_progress_callback() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let count = Arc::new(AtomicUsize::new(0));
        let called = count.clone();
        let cleanup = ManagedCleanup::register(Box::new(move || {
            called.fetch_add(1, Ordering::SeqCst);
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        }))
        .unwrap();
        let registered = cleanup.0.clone();
        let dropping = thread::spawn(move || drop(cleanup));
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(registered.try_lock().is_err());
        let racing = thread::spawn(move || {
            run_cleanup(&registered);
            done_tx.send(()).unwrap();
        });
        assert!(matches!(
            done_rx.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        release_tx.send(()).unwrap();
        dropping.join().unwrap();
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        racing.join().unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stopped_registration_fixture() {
        if std::env::var_os("WERK_TEST_STOP_REGISTRATION").is_none() {
            return;
        }
        let count = Arc::new(AtomicUsize::new(0));
        let called = count.clone();
        let cleanup = ManagedCleanup::register(Box::new(move || {
            // Registration from inside cleanup also proves that callbacks run
            // outside the registry lock, and cannot escape ongoing shutdown.
            assert!(ManagedCleanup::register(Box::new(|| {})).is_err());
            assert!(ManagedChild::spawn(&mut Command::new("true")).is_err());
            called.fetch_add(1, Ordering::SeqCst);
        }))
        .unwrap();
        shutdown_children();
        drop(cleanup);
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(ManagedCleanup::register(Box::new(|| {})).is_err());
    }

    #[test]
    fn shutdown_rejects_new_cleanup_and_child_registrations() {
        // Shutdown is process-global, so isolate it from parallel test cases.
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "backend::llama_process_lifecycle::tests::stopped_registration_fixture",
            ])
            .env("WERK_TEST_STOP_REGISTRATION", "1")
            .status()
            .unwrap();
        assert!(status.success());
    }
}
