//! Only private oMLX cache payloads are removable. Lifetime locks live outside
//! disposable worker bases, and their open descriptions are inherited by Python.
use super::{
    CacheEntry, CacheKind, inspect_cache_tree, metadata_is_link_or_reparse, read_directory_bounded,
    remove_cache_tree, validate_owned_directory,
};
use anyhow::{Context, Result, bail};
use fs2::FileExt;
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
};

const MARKER: &[u8] = b"werk-omlx-worker-lifetime-v1\n";
const LOCK_DIRECTORY: &str = ".locks";
const PERSISTENT_LOCK: &str = ".worker.lock";
const MAX_WORKERS: usize = 16_384;
const MAX_LEGACY_DIRECTORIES: usize = 1024;
const MAX_LEGACY_DEPTH: usize = 16;
pub(crate) const LIFETIME_FDS_ENV: &str = "WERK_OMLX_LIFETIME_FDS";

pub(super) enum WorkerLock {
    Held(File),
    Active,
    Missing,
}

fn valid_worker_id(id: &str) -> bool {
    id.len() == 48
        && id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn workers_root(home: &Path) -> Result<Option<PathBuf>> {
    let mut path = home.to_owned();
    for component in [None, Some("backends"), Some("omlx"), Some("workers")] {
        if let Some(component) = component {
            path.push(component);
        }
        match fs::symlink_metadata(&path) {
            Ok(_) => validate_owned_directory(&path)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).context("cannot inspect oMLX worker directory"),
        }
    }
    Ok(Some(path))
}

fn ensure_directory(path: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    builder.mode(0o700);
    match builder.create(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error).context("cannot create oMLX worker directory"),
    }
    validate_owned_directory(path)?;
    #[cfg(unix)]
    if fs::symlink_metadata(path)?.permissions().mode() & 0o022 != 0 {
        bail!("oMLX worker directory is writable by another user");
    }
    Ok(())
}

/// Acquire before creating the worker base. The stable lock must never be
/// unlinked: a cache purge and a running worker must observe the same inode.
pub(crate) fn prepare_worker(home: &Path, id: &str) -> Result<(PathBuf, Option<File>)> {
    if !valid_worker_id(id) {
        bail!("invalid oMLX worker identity");
    }
    fs::create_dir_all(home).context("cannot create Werk storage directory")?;
    ensure_directory(home)?;
    let mut root = home.to_owned();
    for component in ["backends", "omlx", "workers"] {
        root.push(component);
        ensure_directory(&root)?;
    }
    #[cfg(unix)]
    let lock = {
        let locks = root.join(LOCK_DIRECTORY);
        ensure_directory(&locks)?;
        Some(create_lifetime_lock(&locks.join(format!("{id}.lock")))?)
    };
    // Without inherited POSIX flock semantics, no marker may claim that the
    // worker lifetime is known. Its cache consequently remains protected.
    #[cfg(not(unix))]
    let lock = None;
    let base = root.join(id);
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    builder.mode(0o700);
    builder
        .create(&base)
        .context("cannot create isolated oMLX base path")?;
    Ok((base, lock))
}

/// The chat archive lock belongs to Werk; this second lock also belongs to the
/// native cache writer after Werk dies, until that worker actually exits.
pub(crate) fn lock_persistent_worker_cache(directory: &Path) -> Result<Option<File>> {
    validate_owned_directory(directory)?;
    #[cfg(unix)]
    return create_lifetime_lock(&directory.join(PERSISTENT_LOCK)).map(Some);
    #[cfg(not(unix))]
    Ok(None)
}

pub(super) fn try_persistent_cache_lock(directory: &Path) -> Result<WorkerLock> {
    validate_owned_directory(directory)?;
    try_lifetime_lock(&directory.join(PERSISTENT_LOCK))
}

pub(crate) fn inherit_lifetime_locks(command: &mut Command, files: &[File]) -> Result<()> {
    command.env_remove(LIFETIME_FDS_ENV);
    #[cfg(unix)]
    {
        use std::os::{fd::AsRawFd, unix::process::CommandExt};
        let descriptors = files.iter().map(AsRawFd::as_raw_fd).collect::<Vec<_>>();
        if !descriptors.is_empty() {
            command.env(
                LIFETIME_FDS_ENV,
                descriptors
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            // Only fcntl, an async-signal-safe operation, runs after fork. The
            // parent keeps CLOEXEC; unrelated subprocesses cannot retain locks.
            unsafe {
                command.pre_exec(move || {
                    for descriptor in &descriptors {
                        let flags = libc::fcntl(*descriptor, libc::F_GETFD);
                        if flags < 0
                            || libc::fcntl(*descriptor, libc::F_SETFD, flags & !libc::FD_CLOEXEC)
                                < 0
                        {
                            return Err(std::io::Error::last_os_error());
                        }
                    }
                    Ok(())
                });
            }
        }
    }
    #[cfg(not(unix))]
    let _ = files;
    Ok(())
}

fn validate_lock_metadata(metadata: &fs::Metadata) -> Result<()> {
    if metadata_is_link_or_reparse(metadata) || !metadata.is_file() {
        bail!("oMLX lifetime lock is not a regular file");
    }
    #[cfg(unix)]
    if metadata.uid() != unsafe { libc::geteuid() }
        || metadata.nlink() != 1
        || metadata.permissions().mode() & 0o077 != 0
    {
        bail!("oMLX lifetime lock is not a private, singly linked owner file");
    }
    Ok(())
}

fn lock_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    #[cfg(unix)]
    options
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x0020_0000);
    }
    options
}

fn read_marker(file: &File) -> Result<()> {
    let mut marker = Vec::new();
    file.take((MARKER.len() + 1) as u64)
        .read_to_end(&mut marker)?;
    if marker != MARKER {
        bail!("oMLX cache has an unrecognized lifetime marker; worker activity is unknown");
    }
    Ok(())
}

fn try_lifetime_lock(path: &Path) -> Result<WorkerLock> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => validate_lock_metadata(&metadata)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(WorkerLock::Missing);
        }
        Err(error) => return Err(error).context("cannot inspect oMLX lifetime lock"),
    }
    let file = lock_options()
        .open(path)
        .context("cannot open oMLX lifetime lock")?;
    validate_lock_metadata(&file.metadata()?)?;
    match FileExt::try_lock_exclusive(&file) {
        Ok(()) => {
            read_marker(&file)?;
            Ok(WorkerLock::Held(file))
        }
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(WorkerLock::Active),
        Err(error) => Err(error).context("cannot check oMLX worker activity"),
    }
}

#[cfg(unix)]
fn create_lifetime_lock(path: &Path) -> Result<File> {
    match lock_options().create_new(true).open(path) {
        Ok(mut file) => {
            validate_lock_metadata(&file.metadata()?)?;
            FileExt::try_lock_exclusive(&file).context("oMLX worker cache is already active")?;
            file.write_all(MARKER)?;
            file.sync_all()?;
            Ok(file)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            match try_lifetime_lock(path)? {
                WorkerLock::Held(file) => Ok(file),
                WorkerLock::Active => {
                    bail!("oMLX worker cache is already active; wait for its worker to exit")
                }
                WorkerLock::Missing => bail!("oMLX lifetime lock disappeared during startup"),
            }
        }
        Err(error) => Err(error).context("cannot create oMLX lifetime lock"),
    }
}

fn worker_lock(root: &Path, id: &str) -> Result<WorkerLock> {
    let locks = root.join(LOCK_DIRECTORY);
    match fs::symlink_metadata(&locks) {
        Ok(_) => validate_owned_directory(&locks)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(WorkerLock::Missing);
        }
        Err(error) => return Err(error).context("cannot inspect oMLX lifecycle directory"),
    }
    try_lifetime_lock(&locks.join(format!("{id}.lock")))
}

fn blank_entry(id: &str) -> CacheEntry {
    CacheEntry {
        id: format!("omlx-worker:{id}"),
        kind: CacheKind::OmlxWorker,
        backend: Some("omlx".into()),
        bytes: None,
        modified_unix_seconds: None,
        active: false,
        blocked_reason: None,
    }
}

/// oMLX 0.6.4 creates `_boundary_snapshots/<pid>-<uuid.hex>` in its cache.
/// Its SSD writers are threads of that owner process. In an old, uniquely
/// named Werk worker base, dead owners and a tree containing only directories
/// allow empty-directory cleanup without inventing a lifetime lock for it.
struct LegacyEmptyCache {
    directories: Vec<PathBuf>,
    owner_pids: Vec<i32>,
}

impl LegacyEmptyCache {
    fn inspect(cache: &Path) -> Result<Self> {
        let mut directories = Vec::new();
        let mut pending = vec![(cache.to_owned(), 0usize)];
        while let Some((path, depth)) = pending.pop() {
            if directories.len() >= MAX_LEGACY_DIRECTORIES || depth > MAX_LEGACY_DEPTH {
                bail!("legacy cache exceeds the empty-directory inspection limit");
            }
            if depth == 1 {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .context("unrecognized legacy oMLX cache layout")?;
                if name != "_boundary_snapshots"
                    && name != "response-state"
                    && !(name.len() == 1 && b"0123456789abcdef".contains(&name.as_bytes()[0]))
                {
                    bail!("unrecognized legacy oMLX cache layout");
                }
            }
            // A byte sum of zero is insufficient: even zero-length files must
            // remain protected. Reject files, links and special entries here.
            validate_owned_directory(&path)?;
            let remaining =
                MAX_LEGACY_DIRECTORIES.saturating_sub(directories.len() + pending.len() + 1);
            pending.extend(
                read_directory_bounded(&path, remaining)?
                    .into_iter()
                    .map(|child| (child, depth + 1)),
            );
            directories.push(path);
        }
        let owners = cache.join("_boundary_snapshots");
        validate_owned_directory(&owners)?;
        let mut owner_pids = Vec::new();
        for owner in read_directory_bounded(&owners, MAX_LEGACY_DIRECTORIES)? {
            let name = owner
                .file_name()
                .and_then(|name| name.to_str())
                .context("unrecognized legacy oMLX process marker")?;
            let (pid, nonce) = name
                .split_once('-')
                .context("unrecognized legacy oMLX process marker")?;
            if pid.is_empty()
                || !pid.bytes().all(|byte| byte.is_ascii_digit())
                || nonce.len() != 32
                || !nonce
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                bail!("unrecognized legacy oMLX process marker");
            }
            let pid: i32 = pid.parse().context("invalid legacy oMLX owner PID")?;
            if pid <= 0 {
                bail!("invalid legacy oMLX owner PID");
            }
            owner_pids.push(pid);
        }
        if owner_pids.is_empty() {
            bail!("legacy oMLX cache has no process markers");
        }
        Ok(Self {
            directories,
            owner_pids,
        })
    }

    fn owners_have_exited(&self) -> Result<bool> {
        #[cfg(unix)]
        {
            for &pid in &self.owner_pids {
                // Signal zero checks existence and sends no signal. Reused
                // PIDs are conservatively treated as active, too.
                if unsafe { libc::kill(pid, 0) } == 0 {
                    return Ok(false);
                }
                if std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) {
                    bail!("cannot verify that the legacy oMLX owner process has exited");
                }
            }
            Ok(true)
        }
        #[cfg(not(unix))]
        bail!("legacy oMLX process verification is unavailable on this platform")
    }

    fn remove(self) -> Result<()> {
        let current = Self::inspect(
            self.directories
                .first()
                .context("missing legacy cache root")?,
        )?;
        if !current.owners_have_exited()? {
            bail!("legacy oMLX cache owner is still running; stop it before purging");
        }
        // Parents precede children during inspection. rmdir is atomic with
        // respect to emptiness, so a file written after inspection is never
        // deleted. Do not replace this with recursive tree removal.
        for directory in current.directories.into_iter().rev() {
            validate_owned_directory(&directory)?;
            fs::remove_dir(directory)
                .context("legacy cache changed or is no longer empty; cleanup stopped")?;
        }
        Ok(())
    }
}

fn inspect_legacy_empty_cache(cache: &Path) -> Result<LegacyEmptyCache> {
    LegacyEmptyCache::inspect(cache)
        .context("legacy oMLX cache has no lifetime lock; worker activity is unknown")
}

pub(super) fn list(home: &Path) -> Result<Vec<CacheEntry>> {
    let Some(root) = workers_root(home)? else {
        return Ok(Vec::new());
    };
    let mut entries = Vec::new();
    for worker in read_directory_bounded(&root, MAX_WORKERS)? {
        let Some(id) = worker
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| valid_worker_id(name))
        else {
            continue;
        };
        let mut entry = blank_entry(id);
        if let Err(error) = validate_owned_directory(&worker) {
            entry.blocked_reason = Some(format!("{error:#}"));
            entries.push(entry);
            continue;
        }
        let cache = worker.join("cache");
        match fs::symlink_metadata(&cache) {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                entry.blocked_reason = Some(format!("cannot inspect oMLX cache: {error}"));
                entries.push(entry);
                continue;
            }
        }
        let mut legacy = false;
        let guard = match worker_lock(&root, id) {
            Ok(WorkerLock::Held(file)) => Some(file),
            Ok(WorkerLock::Active) => {
                entry.active = true;
                entry.blocked_reason = Some("oMLX worker is active".into());
                None
            }
            Ok(WorkerLock::Missing) => {
                legacy = true;
                None
            }
            Err(error) => {
                entry.blocked_reason = Some(format!("{error:#}"));
                None
            }
        };
        match validate_owned_directory(&cache).and_then(|()| inspect_cache_tree(&cache)) {
            Ok(stats) => {
                entry.bytes = Some(stats.bytes);
                entry.modified_unix_seconds = stats.modified_unix_seconds;
            }
            Err(error) if entry.blocked_reason.is_none() => {
                entry.blocked_reason = Some(format!("{error:#}"))
            }
            Err(_) => {}
        }
        if legacy && entry.blocked_reason.is_none() {
            match inspect_legacy_empty_cache(&cache).and_then(|plan| plan.owners_have_exited()) {
                Ok(true) => {}
                Ok(false) => {
                    entry.active = true;
                    entry.blocked_reason = Some("legacy oMLX cache owner is still running".into());
                }
                Err(error) => entry.blocked_reason = Some(format!("{error:#}")),
            }
        }
        drop(guard);
        entries.push(entry);
    }
    Ok(entries)
}

pub(super) fn purge(home: &Path, id: &str, dry_run: bool) -> Result<CacheEntry> {
    let worker_id = id
        .strip_prefix("omlx-worker:")
        .filter(|id| valid_worker_id(id))
        .context("invalid oMLX worker cache ID; use the exact ID from cache list")?;
    let root = workers_root(home)?.context("oMLX worker cache does not exist")?;
    let worker = root.join(worker_id);
    validate_owned_directory(&worker)?;
    let _guard = match worker_lock(&root, worker_id)? {
        WorkerLock::Held(file) => Some(file),
        WorkerLock::Active => bail!("oMLX worker is active; stop it before purging its cache"),
        WorkerLock::Missing => None,
    };
    // Recheck the exact payload only after acquiring its lifetime lock.
    validate_owned_directory(&worker)?;
    let cache = worker.join("cache");
    validate_owned_directory(&cache)?;
    let stats = inspect_cache_tree(&cache)?;
    let mut entry = blank_entry(worker_id);
    entry.bytes = Some(stats.bytes);
    entry.modified_unix_seconds = stats.modified_unix_seconds;
    let legacy = if _guard.is_none() {
        let plan = inspect_legacy_empty_cache(&cache)?;
        if !plan.owners_have_exited()? {
            bail!("legacy oMLX cache owner is still running; stop it before purging");
        }
        Some(plan)
    } else {
        None
    };
    if !dry_run {
        if let Some(plan) = legacy {
            plan.remove()?;
        } else {
            remove_cache_tree(&cache)?;
        }
        #[cfg(unix)]
        File::open(&worker)?
            .sync_all()
            .context("cannot flush oMLX worker directory")?;
    }
    Ok(entry)
}

#[cfg(all(test, unix))]
mod legacy_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let mut bytes = [0u8; 8];
            getrandom::getrandom(&mut bytes).unwrap();
            let path = std::env::temp_dir()
                .join(format!("werk-worker-cache-{:x}", u64::from_ne_bytes(bytes)));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn worker(&self) -> (String, PathBuf, Option<File>) {
            let id = "a".repeat(48);
            let (worker, guard) = prepare_worker(&self.0, &id).unwrap();
            fs::create_dir(worker.join("cache")).unwrap();
            fs::write(worker.join("cache/payload"), b"cache bytes").unwrap();
            fs::write(worker.join("settings.json"), b"private settings").unwrap();
            fs::write(worker.join("server.log"), b"private log").unwrap();
            (id, worker, guard)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    #[cfg(unix)]
    fn idle_worker_purge_only_removes_cache_and_preserves_lock_inode() {
        let fixture = Fixture::new();
        let (id, worker, guard) = fixture.worker();
        let lock_path = worker.parent().unwrap().join(format!(".locks/{id}.lock"));
        let inode = fs::metadata(&lock_path).unwrap().ino();
        drop(guard);
        let entries = list(&fixture.0).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].bytes, Some(11));
        assert!(!entries[0].active);
        assert!(entries[0].blocked_reason.is_none());
        purge(&fixture.0, &format!("omlx-worker:{id}"), true).unwrap();
        assert!(worker.join("cache/payload").exists());
        purge(&fixture.0, &format!("omlx-worker:{id}"), false).unwrap();
        assert!(!worker.join("cache").exists());
        assert_eq!(
            fs::read(worker.join("settings.json")).unwrap(),
            b"private settings"
        );
        assert_eq!(fs::read(worker.join("server.log")).unwrap(), b"private log");
        assert_eq!(fs::metadata(lock_path).unwrap().ino(), inode);
        assert!(list(&fixture.0).unwrap().is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn active_worker_blocks_inventory_removal_and_rechecks_before_purge() {
        let fixture = Fixture::new();
        let (id, worker, guard) = fixture.worker();
        let cache_id = format!("omlx-worker:{id}");
        assert!(list(&fixture.0).unwrap()[0].active);
        for dry_run in [true, false] {
            assert!(purge(&fixture.0, &cache_id, dry_run).is_err());
        }
        drop(guard);
        assert!(!list(&fixture.0).unwrap()[0].active);
        let lock_path = worker.parent().unwrap().join(format!(".locks/{id}.lock"));
        let _guard = create_lifetime_lock(&lock_path).unwrap();
        assert!(purge(&fixture.0, &cache_id, false).is_err());
        assert!(worker.join("cache/payload").exists());
    }

    #[test]
    fn legacy_and_unrecognized_lifetime_caches_remain_blocked() {
        let fixture = Fixture::new();
        let id = "b".repeat(48);
        let worker = fixture.0.join("backends/omlx/workers").join(&id);
        fs::create_dir_all(worker.join("cache")).unwrap();
        fs::write(worker.join("cache/legacy"), b"do not remove").unwrap();
        let entry = list(&fixture.0).unwrap().pop().unwrap();
        assert_eq!(entry.bytes, Some(13));
        assert!(entry.blocked_reason.unwrap().contains("unknown"));
        assert!(purge(&fixture.0, &format!("omlx-worker:{id}"), false).is_err());
        assert!(!worker.parent().unwrap().join(".locks").exists());
        #[cfg(unix)]
        {
            let locks = worker.parent().unwrap().join(".locks");
            ensure_directory(&locks).unwrap();
            let path = locks.join(format!("{id}.lock"));
            let mut file = lock_options().create_new(true).open(path).unwrap();
            file.write_all(b"unknown version\n").unwrap();
            assert!(purge(&fixture.0, &format!("omlx-worker:{id}"), false).is_err());
        }
        assert!(worker.join("cache/legacy").exists());
    }

    #[test]
    fn arbitrary_or_traversing_ids_are_rejected() {
        let fixture = Fixture::new();
        for id in [
            "../cache",
            "omlx-worker:../cache",
            "omlx-worker:ABC",
            "omlx-worker:",
        ] {
            assert!(purge(&fixture.0, id, false).is_err());
        }
        assert!(!valid_worker_id(&"a".repeat(64)));
    }

    #[test]
    #[cfg(unix)]
    fn symlinked_payload_and_hardlinked_files_are_never_removed() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        let (id, worker, guard) = fixture.worker();
        drop(guard);
        let unrelated = fixture.0.join("weights.safetensors");
        fs::write(&unrelated, b"model weights").unwrap();
        let linked = worker.join("cache/linked");
        symlink(&unrelated, &linked).unwrap();
        assert!(purge(&fixture.0, &format!("omlx-worker:{id}"), false).is_err());
        fs::remove_file(&linked).unwrap();
        fs::hard_link(&unrelated, &linked).unwrap();
        assert!(purge(&fixture.0, &format!("omlx-worker:{id}"), false).is_err());
        assert_eq!(fs::read(unrelated).unwrap(), b"model weights");
        assert!(worker.join("cache/payload").exists());
    }

    #[test]
    #[cfg(unix)]
    fn inherited_worker_lock_outlives_parent_handle_until_child_exits() {
        use std::process::Stdio;
        let fixture = Fixture::new();
        let (id, worker, guard) = fixture.worker();
        let native_cache = fixture.0.join("native-cache");
        fs::create_dir(&native_cache).unwrap();
        let native_guard = lock_persistent_worker_cache(&native_cache)
            .unwrap()
            .unwrap();
        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", "echo ready; read answer"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        let files = vec![guard.unwrap(), native_guard];
        inherit_lifetime_locks(&mut command, &files).unwrap();
        let mut child = command.spawn().unwrap();
        let mut ready = String::new();
        std::io::BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut ready)
            .unwrap();
        assert_eq!(ready.trim(), "ready");
        drop(files);
        assert!(list(&fixture.0).unwrap()[0].active);
        assert!(matches!(
            try_persistent_cache_lock(&native_cache).unwrap(),
            WorkerLock::Active
        ));
        assert!(purge(&fixture.0, &format!("omlx-worker:{id}"), false).is_err());
        drop(child.stdin.take());
        child.wait().unwrap();
        assert!(matches!(
            try_persistent_cache_lock(&native_cache).unwrap(),
            WorkerLock::Held(_)
        ));
        purge(&fixture.0, &format!("omlx-worker:{id}"), false).unwrap();
        assert!(!worker.join("cache").exists());
    }

    #[test]
    #[cfg(unix)]
    fn persistent_native_cache_lock_is_shared_with_the_worker() {
        let fixture = Fixture::new();
        assert!(matches!(
            try_persistent_cache_lock(&fixture.0).unwrap(),
            WorkerLock::Missing
        ));
        let guard = lock_persistent_worker_cache(&fixture.0).unwrap().unwrap();
        assert!(matches!(
            try_persistent_cache_lock(&fixture.0).unwrap(),
            WorkerLock::Active
        ));
        assert!(lock_persistent_worker_cache(&fixture.0).is_err());
        drop(guard);
        assert!(matches!(
            try_persistent_cache_lock(&fixture.0).unwrap(),
            WorkerLock::Held(_)
        ));
    }
}
