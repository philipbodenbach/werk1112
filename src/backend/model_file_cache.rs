//! File-scoped, advisory model page-cache release after the owning worker exits.
//!
//! Shared leases coordinate participating Werk workers. Other applications do
//! not participate: mapped pages are protected by the kernel, but their unmapped
//! file cache can also be discarded. Advice never changes file contents and does
//! not guarantee that every requested page is reclaimed or returned to Windows.

use crate::backend::llama_process_lifecycle::ManagedCleanup;
use crate::model_store::{ModelManifest, ModelStore};
use anyhow::{Context, Result, ensure};
use std::{
    collections::HashSet,
    fs::{self, File, Metadata, OpenOptions},
    io,
    os::{
        fd::AsRawFd,
        unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime},
};

const MAX_FILES: usize = 4096;
const MAX_GGUF_SHARDS: usize = 256;
const LEASE_WAIT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelFileIdentity {
    length: u64,
    modified: Option<SystemTime>,
    unix: (u64, u64, i64, i64, i64, i64),
}

impl ModelFileIdentity {
    pub(crate) fn from_metadata(metadata: &Metadata) -> Self {
        Self {
            length: metadata.len(),
            modified: metadata.modified().ok(),
            unix: (
                metadata.dev(),
                metadata.ino(),
                metadata.mtime(),
                metadata.mtime_nsec(),
                metadata.ctime(),
                metadata.ctime_nsec(),
            ),
        }
    }

    pub(crate) fn matches(&self, metadata: &Metadata) -> bool {
        metadata.is_file() && *self == Self::from_metadata(metadata)
    }
}

pub(crate) struct CacheReleaseGuard {
    cancelled: Arc<AtomicBool>,
    preparation_gate: Arc<Mutex<()>>,
    retained_preparation: Arc<Mutex<Vec<Box<dyn Send>>>>,
    // The owner must reap its native child before dropping this token.
    _cleanup: ManagedCleanup,
}

impl CacheReleaseGuard {
    pub(crate) fn prepare(
        model_path: &Path,
        projector_path: Option<&Path>,
        args: &[String],
    ) -> Option<Self> {
        if !selected_paths_match(model_path, projector_path, args)
            || [
                "LLAMA_ARG_MODEL_URL",
                "LLAMA_ARG_HF_REPO",
                "LLAMA_ARG_HF_FILE",
                "LLAMA_ARG_MMPROJ_URL",
                "LLAMA_ARG_MODELS_DIR",
            ]
            .iter()
            .any(|name| std::env::var_os(name).is_some_and(|value| !value.is_empty()))
        {
            return None;
        }
        let paths = match gguf_paths(model_path, projector_path) {
            Ok(paths) => paths,
            Err(error) => {
                eprintln!("[werk] model file cache release unavailable: {error:#}");
                return None;
            }
        };
        Self::prepare_inner(paths, true)
    }

    /// For backends whose manifest already identifies the actual model assets.
    /// Callers must retain the guard until their worker has stopped and join any
    /// preparation using `with_preparation` before dropping it.
    pub(crate) fn prepare_paths(paths: Vec<PathBuf>) -> Option<Self> {
        Self::prepare_inner(paths, false)
    }

    pub(crate) fn prepare_manifest(store: &ModelStore, manifest: &ModelManifest) -> Option<Self> {
        Self::prepare_paths(
            manifest
                .files
                .iter()
                .map(|file| store.absolute_model_file(manifest, &file.path))
                .collect(),
        )
    }

    fn prepare_inner(paths: Vec<PathBuf>, require_gguf: bool) -> Option<Self> {
        if paths.is_empty() {
            return None;
        }
        let release = match release_policy(
            std::env::var("WERK_MODEL_CACHE_RELEASE").ok().as_deref(),
            fs::read_to_string("/proc/sys/kernel/osrelease")
                .ok()
                .as_deref(),
        ) {
            Some(release) => release,
            None => {
                eprintln!(
                    "[werk] invalid WERK_MODEL_CACHE_RELEASE; expected auto, off or on; release disabled"
                );
                false
            }
        };
        // Retain leases even with release off: an active reader must protect its
        // files against another participating worker whose policy enables it.
        let leases = match open_leases(&paths, require_gguf, LEASE_WAIT) {
            Ok(leases) => leases,
            Err(error) => {
                eprintln!(
                    "[werk] model file cache lease unavailable; scoped release skipped: {error:#}"
                );
                return None;
            }
        };
        Self::from_leases(leases, release)
    }

    fn from_leases(leases: Vec<FileLease>, release: bool) -> Option<Self> {
        let cancelled = Arc::new(AtomicBool::new(false));
        let preparation_gate = Arc::new(Mutex::new(()));
        let retained_preparation = Arc::new(Mutex::new(Vec::<Box<dyn Send>>::new()));
        let cleanup_cancelled = Arc::clone(&cancelled);
        let cleanup_gate = Arc::clone(&preparation_gate);
        let cleanup_retained = Arc::clone(&retained_preparation);
        let cleanup = ManagedCleanup::register(Box::new(move || {
            cleanup_cancelled.store(true, Ordering::Release);
            let _gate = cleanup_gate.lock().unwrap_or_else(|error| error.into_inner());
            let retained = std::mem::take(
                &mut *cleanup_retained
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()),
            );
            // Populated mappings must be gone before file-cache advice, also
            // when signal shutdown runs this callback without dropping owners.
            drop(retained);
            if release {
                let result = release_leases(&leases);
                eprintln!(
                    "[werk] model file cache release requested: {:.2} GiB across {} file(s); {} shared, {} changed, {} failed (advisory)",
                    result.requested_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    result.advised_files,
                    result.shared_files,
                    result.changed_files,
                    result.failed_files,
                );
            }
            drop(leases);
        }))
        .ok()?;
        Some(Self {
            cancelled,
            preparation_gate,
            retained_preparation,
            _cleanup: cleanup,
        })
    }

    pub(crate) fn with_preparation<T>(&self, prepare: impl FnOnce(Option<&AtomicBool>) -> T) -> T {
        let _gate = self
            .preparation_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        prepare(Some(&self.cancelled))
    }

    pub(crate) fn with_retained_preparation<T, R: Send + 'static>(
        &self,
        prepare: impl FnOnce(Option<&AtomicBool>) -> (T, Option<R>),
    ) -> T {
        self.with_preparation(|cancelled| {
            let cancelled_before = self.cancelled.load(Ordering::Acquire);
            let (result, retained) = prepare(cancelled);
            if let Some(retained) = retained
                && !cancelled_before
                && !self.cancelled.load(Ordering::Acquire)
            {
                self.retained_preparation
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(Box::new(retained));
            }
            result
        })
    }
}

fn release_policy(value: Option<&str>, kernel_release: Option<&str>) -> Option<bool> {
    match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
        "on" => Some(true),
        "off" => Some(false),
        "auto" => Some(kernel_release.is_some_and(|release| {
            let release = release.to_ascii_lowercase();
            release.contains("microsoft") || release.contains("wsl")
        })),
        _ => None,
    }
}

fn selected_paths_match(model: &Path, projector: Option<&Path>, args: &[String]) -> bool {
    if option(args, &["--model", "-m"]).map(Path::new) != Some(model)
        || option(args, &["--mmproj", "-mm"]).map(Path::new) != projector
    {
        return false;
    }
    !args.iter().any(|arg| {
        let key = arg.split('=').next().unwrap_or(arg);
        matches!(
            key,
            "--model-url"
                | "-mu"
                | "--hf-repo"
                | "-hf"
                | "-hfr"
                | "--hf-file"
                | "-hff"
                | "--models-dir"
                | "--models-preset"
                | "--mmproj-url"
                | "-mmu"
        ) || (projector.is_some() && key == "--no-mmproj")
    })
}

fn option<'a>(args: &'a [String], names: &[&str]) -> Option<&'a str> {
    let mut selected = None;
    for (index, arg) in args.iter().enumerate() {
        if names.contains(&arg.as_str()) {
            selected = args.get(index + 1).map(String::as_str);
        } else if let Some((key, value)) = arg.split_once('=')
            && names.contains(&key)
        {
            selected = Some(value);
        }
    }
    selected
}

fn gguf_paths(model: &Path, projector: Option<&Path>) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for path in std::iter::once(model).chain(projector) {
        ensure!(
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf")),
            "selected model asset is not a GGUF file"
        );
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .context("GGUF filename is unavailable")?;
        let names =
            crate::model_store::gguf_shard_paths(name)?.unwrap_or_else(|| vec![name.to_string()]);
        ensure!(names.len() <= MAX_GGUF_SHARDS, "too many GGUF shards");
        let parent = path.parent().context("GGUF parent is unavailable")?;
        paths.extend(names.into_iter().map(|name| parent.join(name)));
    }
    Ok(paths)
}

struct FileLease {
    file: File,
    identity: ModelFileIdentity,
    bytes: u64,
}

fn open_leases(paths: &[PathBuf], require_gguf: bool, wait: Duration) -> Result<Vec<FileLease>> {
    ensure!(paths.len() <= MAX_FILES, "too many model assets to retain");
    let deadline = Instant::now() + wait;
    let mut seen = HashSet::new();
    let mut leases = Vec::new();
    for path in paths {
        // Nonblocking open prevents a replaced file/FIFO from hanging startup.
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)
            .with_context(|| format!("cannot open model asset {}", path.display()))?;
        let metadata = file.metadata()?;
        ensure!(metadata.is_file(), "model asset is not a regular file");
        if !seen.insert((metadata.dev(), metadata.ino())) {
            continue;
        }
        let identity = ModelFileIdentity::from_metadata(&metadata);
        if require_gguf {
            let mut magic = [0_u8; 4];
            file.read_exact_at(&mut magic, 0)?;
            ensure!(&magic == b"GGUF", "model asset is not a GGUF file");
        }
        loop {
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) } == 0 {
                break;
            }
            let error = io::Error::last_os_error();
            if matches!(
                error.raw_os_error(),
                Some(libc::EWOULDBLOCK) | Some(libc::EINTR)
            ) && Instant::now() < deadline
            {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
            return Err(error).context("cannot acquire shared model file lease");
        }
        ensure!(
            identity.matches(&file.metadata()?),
            "model asset changed while acquiring its lease"
        );
        leases.push(FileLease {
            file,
            identity,
            bytes: metadata.len(),
        });
    }
    Ok(leases)
}

#[derive(Default, Debug)]
struct ReleaseResult {
    requested_bytes: u64,
    advised_files: usize,
    shared_files: usize,
    changed_files: usize,
    failed_files: usize,
}

fn release_leases(leases: &[FileLease]) -> ReleaseResult {
    let mut result = ReleaseResult::default();
    for lease in leases {
        // Upgrade only after this worker has ended. A failed Linux upgrade may
        // drop our shared lock; that is safe now that our reader has stopped.
        if unsafe { libc::flock(lease.file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            if io::Error::last_os_error().raw_os_error() == Some(libc::EWOULDBLOCK) {
                result.shared_files += 1;
            } else {
                result.failed_files += 1;
            }
            continue;
        }
        let unchanged = lease
            .file
            .metadata()
            .is_ok_and(|metadata| lease.identity.matches(&metadata));
        if !unchanged {
            result.changed_files += 1;
            continue;
        }
        // Zero length means through EOF. The retained descriptor and original
        // identity keep advice scoped to the exact asset we acquired earlier.
        let error =
            unsafe { libc::posix_fadvise(lease.file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        if error == 0 {
            result.advised_files += 1;
            result.requested_bytes = result.requested_bytes.saturating_add(lease.bytes);
        } else {
            result.failed_files += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Write,
        process::{Command, Stdio},
        time::{SystemTime, UNIX_EPOCH},
    };

    // linux/magic.h; not exposed by every supported libc crate version.
    const RAMFS_MAGIC: u32 = 0x8584_58f6;

    struct Fixture {
        path: PathBuf,
        contents: Vec<u8>,
    }

    impl Fixture {
        fn new() -> Self {
            let id = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "werk-model-file-cache-{}-{id}.gguf",
                std::process::id()
            ));
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            let mut contents = vec![0x5a; page_size * 4];
            contents[..4].copy_from_slice(b"GGUF");
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            file.write_all(&contents).unwrap();
            file.sync_all().unwrap();
            Self { path, contents }
        }

        fn leases(&self) -> Vec<FileLease> {
            open_leases(std::slice::from_ref(&self.path), true, Duration::ZERO).unwrap()
        }

        fn supports_eviction(&self) -> bool {
            let file = File::open(&self.path).unwrap();
            let mut filesystem = std::mem::MaybeUninit::<libc::statfs>::uninit();
            assert_eq!(
                unsafe { libc::fstatfs(file.as_raw_fd(), filesystem.as_mut_ptr()) },
                0
            );
            let kind = unsafe { filesystem.assume_init() }.f_type;
            kind != libc::TMPFS_MAGIC && kind as u32 != RAMFS_MAGIC
        }

        fn resident_pages(&self) -> usize {
            let file = File::open(&self.path).unwrap();
            let length = self.contents.len();
            let mapping = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            assert_ne!(mapping, libc::MAP_FAILED);
            let mut pages = [0_u8; 4];
            let checked = unsafe { libc::mincore(mapping, length, pages.as_mut_ptr()) };
            let unmapped = unsafe { libc::munmap(mapping, length) };
            assert_eq!(checked, 0);
            assert_eq!(unmapped, 0);
            pages.into_iter().filter(|page| page & 1 != 0).count()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
            let _ = fs::remove_file(self.path.with_extension("unmapped"));
        }
    }

    struct RetainedMapping {
        address: usize,
        length: usize,
        unmapped_marker: PathBuf,
    }

    impl RetainedMapping {
        fn new(path: &Path) -> Self {
            let file = File::open(path).unwrap();
            let length = file.metadata().unwrap().len() as usize;
            let mapping = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            assert_ne!(mapping, libc::MAP_FAILED);
            let retained = Self {
                address: mapping as usize,
                length,
                unmapped_marker: path.with_extension("unmapped"),
            };
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            // Real PTEs prevent advice from discarding these pages until Drop.
            for offset in (0..length).step_by(page) {
                unsafe {
                    std::ptr::read_volatile(mapping.cast::<u8>().add(offset));
                }
            }
            retained
        }
    }

    impl Drop for RetainedMapping {
        fn drop(&mut self) {
            assert_eq!(
                unsafe { libc::munmap(self.address as *mut libc::c_void, self.length) },
                0
            );
            fs::write(&self.unmapped_marker, b"done").unwrap();
        }
    }

    #[test]
    fn retained_preparation_unmaps_before_release_and_with_release_disabled() {
        for release in [true, false] {
            let fixture = Fixture::new();
            let guard = CacheReleaseGuard::from_leases(fixture.leases(), release).unwrap();
            assert_eq!(
                guard.with_retained_preparation(|_| (7, Some(RetainedMapping::new(&fixture.path)))),
                7
            );
            assert!(!fixture.path.with_extension("unmapped").exists());
            assert_eq!(fixture.resident_pages(), 4);
            drop(guard);
            assert!(fixture.path.with_extension("unmapped").exists());
            if fixture.supports_eviction() {
                assert_eq!(fixture.resident_pages(), if release { 0 } else { 4 });
            }
        }
    }

    #[test]
    fn already_cancelled_preparation_drops_returned_resources() {
        let fixture = Fixture::new();
        let guard = CacheReleaseGuard::from_leases(fixture.leases(), false).unwrap();
        guard.cancelled.store(true, Ordering::Release);
        guard.with_retained_preparation(|cancelled| {
            assert!(cancelled.unwrap().load(Ordering::Acquire));
            ((), Some(RetainedMapping::new(&fixture.path)))
        });
        assert!(fixture.path.with_extension("unmapped").exists());
        assert!(guard.retained_preparation.lock().unwrap().is_empty());
    }

    #[test]
    fn default_release_is_wsl_only_and_explicit_policy_overrides() {
        assert_eq!(
            release_policy(None, Some("6.6.87.2-microsoft-standard-WSL2")),
            Some(true)
        );
        assert_eq!(release_policy(None, Some("6.8.0-generic")), Some(false));
        assert_eq!(release_policy(None, None), Some(false));
        assert_eq!(release_policy(Some("off"), Some("WSL2")), Some(false));
        assert_eq!(release_policy(Some("on"), None), Some(true));
        assert_eq!(release_policy(Some("bogus"), Some("WSL2")), None);
    }

    #[test]
    fn native_model_and_projector_overrides_abstain() {
        let args = |values: &[&str]| {
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
        };
        let model = Path::new("/model.gguf");
        assert!(selected_paths_match(
            model,
            None,
            &args(&["--model", "/model.gguf"])
        ));
        assert!(selected_paths_match(
            model,
            Some(Path::new("/vision.gguf")),
            &args(&["-m=/model.gguf", "-mm", "/vision.gguf"])
        ));
        for extra in [
            vec!["-m", "/other.gguf"],
            vec!["--model-url=https://example.invalid/model.gguf"],
            vec!["--mmproj", "/vision.gguf"],
        ] {
            let mut values = args(&["--model", "/model.gguf"]);
            values.extend(args(&extra));
            assert!(!selected_paths_match(model, None, &values));
        }
    }

    #[test]
    fn advice_preserves_file_bytes_and_deduplicates_inode() {
        let fixture = Fixture::new();
        let alias = fixture.path.with_extension("alias.gguf");
        fs::hard_link(&fixture.path, &alias).unwrap();
        let leases =
            open_leases(&[fixture.path.clone(), alias.clone()], true, Duration::ZERO).unwrap();
        assert_eq!(leases.len(), 1);
        let result = release_leases(&leases);
        assert_eq!(result.advised_files, 1);
        assert_eq!(result.requested_bytes, fixture.contents.len() as u64);
        assert_eq!(fs::read(&fixture.path).unwrap(), fixture.contents);
        fs::remove_file(alias).unwrap();
    }

    #[test]
    fn another_shared_reader_protects_its_file() {
        let fixture = Fixture::new();
        let first = fixture.leases();
        let second = fixture.leases();
        let result = release_leases(&first);
        assert_eq!(result.shared_files, 1);
        assert_eq!(result.requested_bytes, 0);
        drop(first);
        assert_eq!(release_leases(&second).advised_files, 1);
    }

    #[test]
    fn partial_setup_failure_releases_previously_acquired_leases() {
        let fixture = Fixture::new();
        let missing = fixture.path.with_extension("missing");
        assert!(open_leases(&[fixture.path.clone(), missing], true, Duration::ZERO).is_err());
        let file = File::open(&fixture.path).unwrap();
        assert_eq!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
    }

    #[test]
    fn new_reader_abstains_when_exclusive_cleanup_does_not_finish() {
        let fixture = Fixture::new();
        let file = File::open(&fixture.path).unwrap();
        assert_eq!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        assert!(open_leases(std::slice::from_ref(&fixture.path), true, Duration::ZERO).is_err());
        drop(file);
        assert_eq!(fixture.leases().len(), 1);
    }

    #[test]
    fn changed_file_is_not_advised() {
        let fixture = Fixture::new();
        let leases = fixture.leases();
        let file = OpenOptions::new().write(true).open(&fixture.path).unwrap();
        file.set_len(fixture.contents.len() as u64 + 1).unwrap();
        let result = release_leases(&leases);
        assert_eq!(result.changed_files, 1);
        assert_eq!(result.requested_bytes, 0);
    }

    #[test]
    fn generic_assets_do_not_require_gguf_magic() {
        let fixture = Fixture::new();
        fs::write(&fixture.path, b"model config or another weights format").unwrap();
        assert!(open_leases(std::slice::from_ref(&fixture.path), true, Duration::ZERO).is_err());
        assert_eq!(
            open_leases(std::slice::from_ref(&fixture.path), false, Duration::ZERO)
                .unwrap()
                .len(),
            1
        );
    }

    /// Runs only in an isolated child because signal shutdown intentionally
    /// exits the process and permanently closes its native-child registry.
    #[test]
    fn interrupted_preparation_fixture() {
        let Some(path) = std::env::var_os("WERK_TEST_CACHE_PREPARATION_FILE") else {
            return;
        };
        let path = PathBuf::from(path);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            use crate::backend::llama_process_lifecycle::{ManagedChild, install_shutdown_handler};
            let _listener = install_shutdown_handler().unwrap();
            let guard = CacheReleaseGuard::prepare_paths(vec![path.clone()]).unwrap();
            if std::env::var_os("WERK_TEST_CACHE_RETAIN_BEFORE_SIGNAL").is_some() {
                guard.with_retained_preparation(|_| ((), Some(RetainedMapping::new(&path))));
                fs::write(path.with_extension("ready"), b"retained").unwrap();
            } else {
                guard.with_retained_preparation(|cancelled| {
                    let mapping = RetainedMapping::new(&path);
                    fs::write(path.with_extension("ready"), b"mapped").unwrap();
                    let started = Instant::now();
                    let cancelled = cancelled.unwrap();
                    while !cancelled.load(Ordering::Acquire) {
                        assert!(started.elapsed() < Duration::from_secs(15));
                        thread::sleep(Duration::from_millis(5));
                    }
                    // Cleanup must join preparation and drop its late resource
                    // before issuing advice, even if the caller ignores cancel.
                    thread::sleep(Duration::from_millis(25));
                    match ManagedChild::spawn(Command::new("sh").args(["-c", "exit 0"])) {
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                            fs::write(path.with_extension("blocked"), b"no late child").unwrap();
                        }
                        _ => panic!("native child creation was not blocked during shutdown"),
                    }
                    ((), Some(mapping))
                });
            }
            // Shutdown exits after the cleanup callback finishes. No normal
            // Drop can accidentally make the interrupted path appear correct.
            thread::sleep(Duration::from_secs(30));
        });
    }

    fn assert_interruption_cleanup(retain_before_signal: bool) {
        let fixture = Fixture::new();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "backend::model_file_cache::tests::interrupted_preparation_fixture",
                "--nocapture",
            ])
            .env("WERK_TEST_CACHE_PREPARATION_FILE", &fixture.path)
            .env("WERK_MODEL_CACHE_RELEASE", "on")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if retain_before_signal {
            command.env("WERK_TEST_CACHE_RETAIN_BEFORE_SIGNAL", "1");
        } else {
            command.env_remove("WERK_TEST_CACHE_RETAIN_BEFORE_SIGNAL");
        }
        let mut child = command.spawn().unwrap();
        let started = Instant::now();
        while !fixture.path.with_extension("ready").exists() {
            if started.elapsed() > Duration::from_secs(15) {
                let _ = child.kill();
                let output = child.wait_with_output().unwrap();
                panic!(
                    "preparation fixture did not become ready: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
        let interrupted = Instant::now();
        while child.try_wait().unwrap().is_none() {
            if interrupted.elapsed() > Duration::from_secs(15) {
                let _ = child.kill();
                let output = child.wait_with_output().unwrap();
                panic!(
                    "preparation cleanup did not complete: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        assert_eq!(output.status.code(), Some(143));
        assert!(fixture.path.with_extension("unmapped").exists());
        if !retain_before_signal {
            assert!(fixture.path.with_extension("blocked").exists());
        }
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("across 1 file(s); 0 shared, 0 changed, 0 failed")
        );

        if fixture.supports_eviction() {
            assert_eq!(fixture.resident_pages(), 0);
        }
        for extension in ["ready", "unmapped"] {
            fs::remove_file(fixture.path.with_extension(extension)).unwrap();
        }
        if !retain_before_signal {
            fs::remove_file(fixture.path.with_extension("blocked")).unwrap();
        }
    }

    #[test]
    fn interruption_cancels_preparation_before_advice_and_blocks_late_child() {
        assert_interruption_cleanup(false);
    }

    #[test]
    fn interruption_drops_retained_preparation_before_advice_without_owner_drop() {
        assert_interruption_cleanup(true);
    }

    #[test]
    fn clean_unmapped_scratch_pages_can_be_released() {
        let fixture = Fixture::new();
        let leases = fixture.leases();
        let file = &leases[0].file;
        let mut filesystem = std::mem::MaybeUninit::<libc::statfs>::uninit();
        assert_eq!(
            unsafe { libc::fstatfs(file.as_raw_fd(), filesystem.as_mut_ptr()) },
            0
        );
        // tmpfs/ramfs have no backing store to discard to; the API remains only
        // advisory there. Production does not promise page reclamation either.
        let kind = unsafe { filesystem.assume_init() }.f_type;
        if kind == libc::TMPFS_MAGIC || kind as u32 == RAMFS_MAGIC {
            return;
        }
        let resident = || {
            let length = fixture.contents.len();
            let mapping = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            assert_ne!(mapping, libc::MAP_FAILED);
            let mut pages = vec![0_u8; 4];
            let result = unsafe { libc::mincore(mapping, length, pages.as_mut_ptr()) };
            let unmap = unsafe { libc::munmap(mapping, length) };
            assert_eq!(result, 0);
            assert_eq!(unmap, 0);
            pages.into_iter().filter(|page| page & 1 != 0).count()
        };
        assert_eq!(resident(), 4);
        assert_eq!(release_leases(&leases).advised_files, 1);
        assert_eq!(resident(), 0);
    }
}
