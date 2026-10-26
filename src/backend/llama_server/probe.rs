//! Reuse CLI inspection without repeatedly initializing the GPU driver.
//! A timed-out caller does not release admission until process/pipe cleanup
//! really finishes. Failures are cached too; a changed executable/environment
//! or a new Werk process permits a fresh inspection.
use super::runtime_state::FileStamp;
use anyhow::{Context, Result, anyhow, ensure};
use sha2::{Digest, Sha256};
use std::{
    collections::VecDeque,
    ffi::OsString,
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

const TIMEOUT: Duration = Duration::from_secs(5);
const CACHE_ENTRIES: usize = 32;
type Outcome = std::result::Result<Arc<str>, Arc<str>>;

#[derive(Clone, PartialEq, Eq)]
struct Key {
    executable: PathBuf,
    stamp: FileStamp,
    argument: String,
    environment: [u8; 32],
    max_bytes: usize,
}

#[derive(Default)]
struct Entry {
    result: Mutex<Option<Outcome>>,
    ready: Condvar,
    cancel: AtomicBool,
}

#[derive(Default)]
struct State {
    entries: VecDeque<(Key, Arc<Entry>)>,
    running: bool,
}

#[derive(Default)]
struct ProbeCache(Mutex<State>, Condvar);

impl ProbeCache {
    fn get(
        self: &Arc<Self>,
        key: Key,
        timeout: Duration,
        operation: impl FnOnce(&AtomicBool) -> Result<String> + Send + 'static,
    ) -> Result<Arc<str>> {
        let started = Instant::now();
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        while state.running && !state.entries.iter().any(|(k, _)| k == &key) {
            let remaining = timeout.saturating_sub(started.elapsed());
            ensure!(
                !remaining.is_zero(),
                "another runtime inspection is still running or cleaning up"
            );
            state = self
                .1
                .wait_timeout(state, remaining)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        let entry = if let Some((_, entry)) = state.entries.iter().find(|(k, _)| k == &key) {
            entry.clone()
        } else {
            // Serialize even different CLI inspections: both --help and
            // --version may initialize CUDA. Never launch another process
            // while cleanup is stuck in a driver syscall.
            ensure!(
                started.elapsed() < timeout,
                "runtime inspection admission timed out"
            );
            let entry = Arc::new(Entry::default());
            if state.entries.len() == CACHE_ENTRIES {
                state.entries.pop_front();
            }
            state.entries.push_back((key, entry.clone()));
            state.running = true;
            let cache = self.clone();
            let pending = entry.clone();
            if let Err(error) = std::thread::Builder::new()
                .name("werk-runtime-probe".into())
                .spawn(move || {
                    let result = operation(&pending.cancel)
                        .map(Arc::<str>::from)
                        .map_err(|error| Arc::<str>::from(format!("{error:#}")));
                    // Operation returns only after owned children/readers are
                    // reaped. A caller's timeout alone must not clear this bit.
                    cache.0.lock().unwrap_or_else(|e| e.into_inner()).running = false;
                    cache.1.notify_all();
                    let mut saved = pending.result.lock().unwrap_or_else(|e| e.into_inner());
                    if saved.is_none() {
                        *saved = Some(result);
                    }
                    pending.ready.notify_all();
                })
            {
                state.running = false;
                self.1.notify_all();
                *entry.result.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(Err(Arc::from(error.to_string())));
            }
            entry
        };
        drop(state);
        let result = entry.result.lock().unwrap_or_else(|e| e.into_inner());
        let (mut result, _) = entry
            .ready
            .wait_timeout_while(result, timeout.saturating_sub(started.elapsed()), |r| {
                r.is_none()
            })
            .unwrap_or_else(|e| e.into_inner());
        if result.is_none() {
            entry.cancel.store(true, Ordering::Release);
            *result = Some(Err(Arc::from(
                "runtime inspection timed out; this failure is cached for the unchanged runtime",
            )));
            entry.ready.notify_all();
        }
        result
            .as_ref()
            .unwrap()
            .clone()
            .map_err(|error| anyhow!(error.to_string()))
    }
}

pub(super) fn output(executable: &Path, argument: &str, max_bytes: usize) -> Result<Arc<str>> {
    static CACHE: OnceLock<Arc<ProbeCache>> = OnceLock::new();
    let executable = std::fs::canonicalize(executable)?;
    let stamp = FileStamp::read(&executable)?;
    // Pin the same inherited environment used for the cache key. Do not log
    // its values (which can contain credentials).
    let mut environment: Vec<(OsString, OsString)> = std::env::vars_os().collect();
    environment.sort();
    let mut digest = Sha256::new();
    for (key, value) in &environment {
        for part in [key, value] {
            let bytes = part.as_encoded_bytes();
            digest.update((bytes.len() as u64).to_le_bytes());
            digest.update(bytes);
        }
    }
    let key = Key {
        executable,
        stamp,
        argument: argument.into(),
        environment: digest.finalize().into(),
        max_bytes,
    };
    let command_key = key.clone();
    CACHE
        .get_or_init(|| Arc::new(ProbeCache::default()))
        .get(key, TIMEOUT, move |cancel| {
            let text = command_output(&command_key, &environment, cancel, TIMEOUT)?;
            ensure!(
                FileStamp::read(&command_key.executable)? == command_key.stamp,
                "runtime changed during CLI inspection"
            );
            Ok(text)
        })
}

fn kill_and_reap(child: &mut Child) {
    #[cfg(unix)]
    // A probe has its own process group, so inherited pipes cannot leave a
    // grandchild alive. This never signals the resident model server.
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
    // This may block in a broken driver. The caller already has its deadline;
    // the cache keeps admission closed until this worker really finishes.
    let _ = child.wait();
}

fn command_output(
    key: &Key,
    environment: &[(OsString, OsString)],
    cancel: &AtomicBool,
    timeout: Duration,
) -> Result<String> {
    ensure!(
        !cancel.load(Ordering::Acquire),
        "runtime inspection cancelled before launch"
    );
    let mut command = Command::new(&key.executable);
    command
        .arg(&key.argument)
        .env_clear()
        .envs(environment.iter().cloned())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command.spawn().context("cannot start runtime inspection")?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let deadline = Instant::now() + timeout;
    std::thread::scope(|scope| {
        let (tx, rx) = mpsc::channel();
        let max = key.max_bytes;
        for (index, reader) in [
            (0, Box::new(stdout) as Box<dyn Read + Send>),
            (1, Box::new(stderr)),
        ] {
            let tx = tx.clone();
            let reader = std::thread::Builder::new()
                .name("werk-probe-output".into())
                .spawn_scoped(scope, move || {
                    let mut bytes = Vec::new();
                    let result = reader
                        .take(max as u64 + 1)
                        .read_to_end(&mut bytes)
                        .map_err(anyhow::Error::from)
                        .and_then(|_| {
                            ensure!(
                                bytes.len() <= max,
                                "runtime inspection output exceeds limit"
                            );
                            Ok(bytes)
                        });
                    let _ = tx.send((index, result));
                });
            if let Err(error) = reader {
                kill_and_reap(&mut child);
                return Err(error).context("cannot start runtime output reader");
            }
        }
        drop(tx);
        let result = (|| {
            let mut received = 0;
            let mut outputs = [Vec::new(), Vec::new()];
            loop {
                ensure!(
                    !cancel.load(Ordering::Acquire) && Instant::now() < deadline,
                    "runtime inspection timed out"
                );
                while let Ok((index, bytes)) = rx.try_recv() {
                    outputs[index] = bytes?;
                    received += 1;
                }
                ensure!(
                    outputs.iter().map(Vec::len).sum::<usize>() <= max,
                    "runtime inspection output exceeds limit"
                );
                if let Some(status) = child.try_wait()?
                    && received == 2
                {
                    if !status.success() {
                        let stderr = String::from_utf8_lossy(&outputs[1]);
                        let stdout = String::from_utf8_lossy(&outputs[0]);
                        // Keep failure diagnostics without growing the cached
                        // error to the full help-output allowance.
                        let detail: String =
                            stderr.chars().chain(stdout.chars()).take(2048).collect();
                        anyhow::bail!(
                            "runtime inspection {} failed with {status}: {detail}",
                            key.argument
                        );
                    }
                    let mut text = String::from_utf8(std::mem::take(&mut outputs[0]))?;
                    if !outputs[1].is_empty() {
                        text.push('\n');
                        text.push_str(std::str::from_utf8(&outputs[1])?);
                    }
                    return Ok(text);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        })();
        if result.is_err() {
            kill_and_reap(&mut child);
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn key(path: &Path) -> Key {
        Key {
            executable: path.into(),
            stamp: FileStamp::read(path).unwrap(),
            argument: "--help".into(),
            environment: [0; 32],
            max_bytes: 1024,
        }
    }

    #[test]
    fn timed_out_probe_keeps_admission_until_cleanup_finishes() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let key = key(file.path());
        let cache = Arc::new(ProbeCache::default());
        let (release, blocked) = mpsc::channel();
        let start = Instant::now();
        assert!(
            cache
                .get(key.clone(), Duration::from_millis(40), move |_| {
                    blocked.recv().unwrap();
                    Ok("late result".into())
                })
                .is_err()
        );
        assert!(start.elapsed() < Duration::from_secs(1));
        for _ in 0..8 {
            assert!(
                cache
                    .get(key.clone(), Duration::from_millis(40), |_| {
                        panic!("same-runtime timeout must remain cached")
                    })
                    .is_err()
            );
            let mut other = key.clone();
            other.argument = "--version".into();
            assert!(
                cache
                    .get(other, Duration::from_millis(40), |_| {
                        panic!("cleanup must retain admission, even for another probe")
                    })
                    .is_err()
            );
        }
        release.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while cache.0.lock().unwrap().running && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(!cache.0.lock().unwrap().running);
        let mut other = key.clone();
        other.argument = "--version".into();
        assert_eq!(
            &*cache
                .get(other, Duration::from_secs(1), |_| Ok("new".into()))
                .unwrap(),
            "new"
        );
        assert!(
            cache
                .get(key, Duration::from_secs(1), |_| panic!())
                .is_err()
        );
    }

    #[test]
    fn concurrent_readers_share_one_inspection_and_reuse_its_result() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let key = key(file.path());
        let cache = Arc::new(ProbeCache::default());
        let calls = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let (cache, calls, key) = (cache.clone(), calls.clone(), key.clone());
                scope.spawn(move || {
                    assert_eq!(
                        &*cache
                            .get(key, Duration::from_secs(2), move |_| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                std::thread::sleep(Duration::from_millis(40));
                                Ok("--mmproj".into())
                            })
                            .unwrap(),
                        "--mmproj"
                    );
                });
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn runtime_variants_do_not_grow_the_cache_without_bound() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let key = key(file.path());
        let cache = Arc::new(ProbeCache::default());
        for variant in 0..CACHE_ENTRIES + 2 {
            let mut next = key.clone();
            next.environment[0] = variant as u8;
            assert_eq!(
                &*cache
                    .get(next, Duration::from_secs(1), move |_| Ok(
                        variant.to_string()
                    ))
                    .unwrap(),
                variant.to_string()
            );
            assert!(cache.0.lock().unwrap().entries.len() <= CACHE_ENTRIES);
        }
        let mut last = key.clone();
        last.environment[0] = (CACHE_ENTRIES + 1) as u8;
        cache
            .get(last, Duration::from_secs(1), |_| {
                panic!("recent result must be reused")
            })
            .unwrap();
        assert_eq!(
            &*cache
                .get(key, Duration::from_secs(1), |_| Ok(
                    "evicted and reinspected".into()
                ))
                .unwrap(),
            "evicted and reinspected"
        );
    }

    #[cfg(unix)]
    fn script(path: &Path, body: &str) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn real_cli_results_are_reused_but_replaced_binary_is_reinspected() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("runtime");
        let calls = dir.path().join("calls");
        script(
            &exe,
            &format!(
                "echo called >> '{}'; echo --mmproj; echo diagnostic >&2",
                calls.display()
            ),
        );
        for _ in 0..10 {
            assert_eq!(
                &*output(&exe, "--help", 1024).unwrap(),
                "--mmproj\n\ndiagnostic\n"
            );
        }
        assert_eq!(std::fs::read_to_string(&calls).unwrap().lines().count(), 1);
        let replacement = dir.path().join("replacement");
        script(
            &replacement,
            &format!("echo called >> '{}'; echo --new-option", calls.display()),
        );
        std::fs::rename(replacement, &exe).unwrap();
        assert!(
            output(&exe, "--help", 1024)
                .unwrap()
                .contains("--new-option")
        );
        assert_eq!(std::fs::read_to_string(&calls).unwrap().lines().count(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn failed_cli_is_not_restarted_for_each_request() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("runtime");
        let calls = dir.path().join("calls");
        script(
            &exe,
            &format!(
                "echo called >> '{}'; echo driver-unavailable >&2; exit 1",
                calls.display()
            ),
        );
        for _ in 0..8 {
            assert!(
                output(&exe, "--help", 1024)
                    .unwrap_err()
                    .to_string()
                    .contains("driver-unavailable")
            );
        }
        assert_eq!(std::fs::read_to_string(calls).unwrap().lines().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn command_deadline_and_output_limits_reap_only_the_probe() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("runtime");
        for body in [
            "exec sleep 5",
            "while :; do printf 1234567890; done",
            "while :; do printf 1234567890 >&2; done",
        ] {
            script(&exe, body);
            let start = Instant::now();
            assert!(
                command_output(
                    &key(&exe),
                    &std::env::vars_os().collect::<Vec<_>>(),
                    &AtomicBool::new(false),
                    Duration::from_millis(80)
                )
                .is_err()
            );
            assert!(start.elapsed() < Duration::from_secs(2));
        }
    }
}
