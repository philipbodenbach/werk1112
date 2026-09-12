//! Cache inventory and explicit removal. Model weights and unrelated files are excluded.
use anyhow::{Context, Result, bail};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
#[cfg(windows)]
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::{
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

mod runtime_states;
pub(crate) mod workers;

const MAX_ROOT_ENTRIES: usize = 16384;
const MAX_TREE_ENTRIES: usize = 100_000;
const MAX_TREE_DEPTH: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheKind {
    ChatKv,
    ChatHistory,
    OmlxWorker,
    RuntimeState,
}

impl CacheKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ChatKv => "chat-kv",
            Self::ChatHistory => "chat-history",
            Self::OmlxWorker => "omlx-worker",
            Self::RuntimeState => "runtime-state",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheEntry {
    pub id: String,
    pub kind: CacheKind,
    pub backend: Option<String>,
    pub bytes: Option<u64>,
    pub modified_unix_seconds: Option<u64>,
    pub active: bool,
    pub blocked_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub enum CacheSelection {
    Id(String),
    All {
        kind: Option<CacheKind>,
        include_history: bool,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheBlocked {
    pub id: String,
    pub reason: String,
}

#[derive(Debug, Default, Serialize)]
pub struct CacheInventory {
    pub entries: Vec<CacheEntry>,
    /// Provider failures have a provider name instead of a removable cache ID.
    pub blocked: Vec<CacheBlocked>,
}

#[derive(Debug, Serialize)]
pub struct CachePurgeReport {
    pub entries: Vec<CacheEntry>,
    pub removed: Vec<String>,
    pub blocked: Vec<CacheBlocked>,
    pub dry_run: bool,
}

pub fn list(home: &Path) -> Result<CacheInventory> {
    Ok(inventory(home, None))
}

fn inventory(home: &Path, kind: Option<CacheKind>) -> CacheInventory {
    let mut report = CacheInventory::default();
    if kind.is_none_or(|kind| matches!(kind, CacheKind::ChatKv | CacheKind::ChatHistory)) {
        collect_provider(&mut report, "chat", list_chat(home));
    }
    if kind.is_none_or(|kind| kind == CacheKind::OmlxWorker) {
        collect_provider(&mut report, "omlx-worker", workers::list(home));
    }
    if kind.is_none_or(|kind| kind == CacheKind::RuntimeState) {
        collect_provider(&mut report, "runtime-state", runtime_states::list(home));
    }
    report.entries.sort_by(|left, right| left.id.cmp(&right.id));
    report
}

fn collect_provider(report: &mut CacheInventory, provider: &str, result: Result<Vec<CacheEntry>>) {
    match result {
        Ok(entries) => report.entries.extend(entries),
        Err(error) => report.blocked.push(CacheBlocked {
            id: provider.into(),
            reason: format!("{error:#}"),
        }),
    }
}

pub fn purge(home: &Path, selection: &CacheSelection, dry_run: bool) -> Result<CachePurgeReport> {
    let (selected, blocked) = match selection {
        CacheSelection::Id(id) => (vec![id.clone()], Vec::new()),
        CacheSelection::All {
            kind,
            include_history,
        } => {
            let inventory = inventory(home, *kind);
            let selected = inventory
                .entries
                .into_iter()
                .filter(|entry| kind.is_none_or(|kind| kind == entry.kind))
                .filter(|entry| {
                    entry.kind != CacheKind::ChatHistory
                        || *include_history
                        || *kind == Some(CacheKind::ChatHistory)
                })
                .map(|entry| entry.id)
                .collect();
            (selected, inventory.blocked)
        }
    };
    let mut report = CachePurgeReport {
        entries: Vec::new(),
        removed: Vec::new(),
        blocked,
        dry_run,
    };
    for id in selected {
        let result = match id.split_once(':').map(|(kind, _)| kind) {
            Some("chat-kv" | "chat-history") => purge_chat(home, &id, dry_run),
            Some("omlx-worker") => workers::purge(home, &id, dry_run),
            Some("runtime-state") => runtime_states::purge(home, &id, dry_run),
            _ => Err(anyhow::anyhow!(
                "unknown cache ID; use an exact ID from werk cache list"
            )),
        };
        match result {
            Ok(entry) => {
                if !dry_run {
                    report.removed.push(id);
                }
                report.entries.push(entry);
            }
            Err(error) => report.blocked.push(CacheBlocked {
                id,
                reason: format!("{error:#}"),
            }),
        }
    }
    Ok(report)
}

fn chat_root(home: &Path) -> Result<Option<PathBuf>> {
    let root = home.join("chat-sessions");
    match fs::symlink_metadata(&root) {
        Ok(_) => {
            validate_owned_directory(&root)?;
            #[cfg(unix)]
            if fs::symlink_metadata(&root)?.permissions().mode() & 0o077 != 0 {
                bail!("chat cache directory is not private to its owner");
            }
            Ok(Some(root))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).context("cannot inspect chat cache directory"),
    }
}

fn list_chat(home: &Path) -> Result<Vec<CacheEntry>> {
    let Some(root) = chat_root(home)? else {
        return Ok(Vec::new());
    };
    let mut entries = Vec::new();
    for path in read_directory_bounded(&root, MAX_ROOT_ENTRIES)? {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some((hash, extension)) = name.rsplit_once('.') else {
            continue;
        };
        if !valid_chat_hash(hash) {
            continue;
        }
        let kind = match extension {
            "cache" => CacheKind::ChatKv,
            "json" => CacheKind::ChatHistory,
            _ => continue,
        };
        entries.push(chat_entry(&root, &path, hash, kind)?);
    }
    Ok(entries)
}

fn valid_chat_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn chat_entry(root: &Path, path: &Path, hash: &str, kind: CacheKind) -> Result<CacheEntry> {
    let mut entry = CacheEntry {
        id: format!("{}:{hash}", kind.as_str()),
        kind,
        backend: None,
        bytes: None,
        modified_unix_seconds: None,
        active: false,
        blocked_reason: None,
    };
    match inspect_chat_target(path, kind) {
        Ok(stats) => {
            entry.bytes = Some(stats.bytes);
            entry.modified_unix_seconds = stats.modified_unix_seconds;
        }
        Err(error) => entry.blocked_reason = Some(format!("{error:#}")),
    }
    if entry.blocked_reason.is_none() {
        entry.backend = chat_backend(path, kind)?;
    }
    match chat_lock(root, hash, false) {
        Ok(ChatLock::Active) => {
            entry.active = true;
            entry.blocked_reason = Some("chat session is active".into());
        }
        Ok(ChatLock::Held(_)) | Ok(ChatLock::Missing) => {}
        Err(error) => entry.blocked_reason = Some(format!("{error:#}")),
    }
    if kind == CacheKind::ChatKv && !entry.active && entry.blocked_reason.is_none() {
        match workers::try_persistent_cache_lock(path) {
            Ok(workers::WorkerLock::Active) => {
                entry.active = true;
                entry.blocked_reason = Some("native cache worker is active".into());
            }
            Ok(workers::WorkerLock::Held(_)) | Ok(workers::WorkerLock::Missing) => {}
            Err(error) => entry.blocked_reason = Some(format!("{error:#}")),
        }
    }
    Ok(entry)
}

fn purge_chat(home: &Path, id: &str, dry_run: bool) -> Result<CacheEntry> {
    let (prefix, hash) = id.split_once(':').context("invalid chat cache ID")?;
    if !valid_chat_hash(hash) {
        bail!("invalid chat cache ID; expected the exact ID from cache list");
    }
    let (kind, extension) = match prefix {
        "chat-kv" => (CacheKind::ChatKv, "cache"),
        "chat-history" => (CacheKind::ChatHistory, "json"),
        _ => bail!("invalid chat cache kind"),
    };
    let root = chat_root(home)?.context("chat cache does not exist")?;
    let path = root.join(format!("{hash}.{extension}"));
    // Inspect once before lock creation so a nonexistent ID cannot create an
    // unlimited number of lock files. Recheck under the lock before removal.
    inspect_chat_target(&path, kind)?;
    let guard = match chat_lock(&root, hash, !dry_run)? {
        ChatLock::Active => bail!("chat session is active; close it before purging its cache"),
        ChatLock::Held(file) => Some(file),
        ChatLock::Missing => None,
    };
    // New workers inherit a second lock, covering the interval after their
    // parent dies. Legacy caches use the existing main session lock alone.
    let _native_guard = if kind == CacheKind::ChatKv {
        match workers::try_persistent_cache_lock(&path)? {
            workers::WorkerLock::Active => {
                bail!("native cache worker is active; stop it before purging its cache")
            }
            workers::WorkerLock::Held(file) => Some(file),
            workers::WorkerLock::Missing => None,
        }
    } else {
        None
    };
    let stats = inspect_chat_target(&path, kind)?;
    let entry = CacheEntry {
        id: id.into(),
        kind,
        backend: chat_backend(&path, kind)?,
        bytes: Some(stats.bytes),
        modified_unix_seconds: stats.modified_unix_seconds,
        active: false,
        blocked_reason: None,
    };
    if !dry_run {
        // The lifetime chat lock protects both its history and native KV tree.
        if guard.is_none() {
            bail!("could not lock chat cache for removal");
        }
        match kind {
            CacheKind::ChatKv => remove_cache_tree(&path)?,
            CacheKind::ChatHistory => {
                fs::remove_file(&path).context("cannot remove selected chat history")?
            }
            _ => unreachable!(),
        }
        sync_directory(&root)?;
    }
    Ok(entry)
}

fn inspect_chat_target(path: &Path, kind: CacheKind) -> Result<TreeStats> {
    let metadata = fs::symlink_metadata(path).context("selected chat cache does not exist")?;
    validate_owned_metadata(&metadata)?;
    if (kind == CacheKind::ChatKv && !metadata.is_dir())
        || (kind == CacheKind::ChatHistory && !metadata.is_file())
    {
        bail!("chat cache has an unexpected file type");
    }
    inspect_cache_tree(path)
}

fn chat_backend(path: &Path, kind: CacheKind) -> Result<Option<String>> {
    if kind != CacheKind::ChatKv {
        return Ok(None);
    }
    let namespaces = read_directory_bounded(path, MAX_ROOT_ENTRIES)?
        .into_iter()
        .filter(|path| path.file_name().and_then(|name| name.to_str()) != Some(".worker.lock"))
        .collect::<Vec<_>>();
    Ok((!namespaces.is_empty()
        && namespaces.iter().all(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("omlx-"))
        }))
    .then(|| "omlx".into()))
}

enum ChatLock {
    Held(File),
    Active,
    Missing,
}

fn chat_lock(root: &Path, hash: &str, create: bool) -> Result<ChatLock> {
    let path = root.join(format!("{hash}.lock"));
    match fs::symlink_metadata(&path) {
        Ok(metadata) => {
            validate_owned_metadata(&metadata)?;
            if !metadata.is_file() {
                bail!("chat session lock is not a regular file");
            }
            #[cfg(unix)]
            if metadata.permissions().mode() & 0o077 != 0 {
                bail!("chat session lock is not private");
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !create => {
            return Ok(ChatLock::Missing);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("cannot inspect chat session lock"),
    }
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(create);
    #[cfg(unix)]
    options
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK);
    #[cfg(windows)]
    options.custom_flags(0x0020_0000);
    let file = options
        .open(&path)
        .context("cannot open chat session lock")?;
    let metadata = file.metadata()?;
    validate_owned_metadata(&metadata)?;
    if !metadata.is_file() {
        bail!("chat session lock is not a regular file");
    }
    match FileExt::try_lock_exclusive(&file) {
        Ok(()) => Ok(ChatLock::Held(file)),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(ChatLock::Active),
        Err(error) => Err(error).context("cannot check chat session activity"),
    }
}

pub(super) struct TreeStats {
    pub(super) bytes: u64,
    pub(super) modified_unix_seconds: Option<u64>,
}

pub(super) fn inspect_cache_tree(path: &Path) -> Result<TreeStats> {
    let mut pending = vec![(path.to_owned(), 0usize)];
    let mut count = 0;
    let mut stats = TreeStats {
        bytes: 0,
        modified_unix_seconds: None,
    };
    while let Some((path, depth)) = pending.pop() {
        count += 1;
        if count > MAX_TREE_ENTRIES || depth > MAX_TREE_DEPTH {
            bail!("cache inspection exceeds its bounded walk limit");
        }
        let metadata = fs::symlink_metadata(&path).context("cannot inspect cache entry")?;
        validate_owned_metadata(&metadata)?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|time| time.as_secs());
        stats.modified_unix_seconds = stats.modified_unix_seconds.max(modified);
        if metadata.is_file() {
            stats.bytes = stats
                .bytes
                .checked_add(metadata.len())
                .context("cache byte count overflow")?;
        } else if metadata.is_dir() {
            let remaining = MAX_TREE_ENTRIES.saturating_sub(count + pending.len());
            let children = read_directory_bounded(&path, remaining)?;
            pending.extend(children.into_iter().map(|child| (child, depth + 1)));
        } else {
            bail!("cache contains a special file; removal is blocked");
        }
    }
    Ok(stats)
}

pub(super) fn remove_cache_tree(path: &Path) -> Result<()> {
    validate_owned_directory(path)?;
    inspect_cache_tree(path)?;
    fs::remove_dir_all(path).context("cannot remove selected cache directory")
}

pub(super) fn read_directory_bounded(path: &Path, limit: usize) -> Result<Vec<PathBuf>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(path).context("cannot inspect cache directory")? {
        if entries.len() == limit {
            bail!("cache directory has too many entries to inspect safely");
        }
        entries.push(entry?.path());
    }
    Ok(entries)
}

pub(super) fn validate_owned_directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).context("cannot inspect cache directory")?;
    validate_owned_metadata(&metadata)?;
    if !metadata.is_dir() {
        bail!("cache path is not a regular directory");
    }
    Ok(())
}

fn validate_owned_metadata(metadata: &fs::Metadata) -> Result<()> {
    if metadata_is_link_or_reparse(metadata) {
        bail!("cache contains a symlink or reparse point; removal is blocked");
    }
    #[cfg(unix)]
    {
        if metadata.uid() != unsafe { libc::geteuid() } {
            bail!("cache entry is not owned by the current user");
        }
        if metadata.is_file() && metadata.nlink() != 1 {
            bail!("cache file has hard links; removal is blocked");
        }
    }
    Ok(())
}

pub(super) fn metadata_is_link_or_reparse(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    return metadata.file_attributes() & 0x0000_0400 != 0;
    #[cfg(not(windows))]
    false
}

fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(path)?
        .sync_all()
        .context("cannot flush cache directory")?;
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        home: PathBuf,
        hash: String,
        kv: PathBuf,
        history: PathBuf,
        lock: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let mut random = [0u8; 8];
            getrandom::getrandom(&mut random).unwrap();
            let home = std::env::temp_dir()
                .join(format!("werk-cache-tests-{:x}", u64::from_le_bytes(random)));
            let root = home.join("chat-sessions");
            fs::create_dir_all(&root).unwrap();
            #[cfg(unix)]
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
            let hash = "a".repeat(64);
            let kv = root.join(format!("{hash}.cache"));
            let namespace = kv.join(format!("omlx-{}", "b".repeat(64))).join("native");
            fs::create_dir_all(&namespace).unwrap();
            fs::write(namespace.join("block.safetensors"), b"cache bytes").unwrap();
            let history = root.join(format!("{hash}.json"));
            // Inventory must not deserialize or expose conversation contents.
            fs::write(
                &history,
                b"private conversation fixture, deliberately not JSON",
            )
            .unwrap();
            let lock = root.join(format!("{hash}.lock"));
            let models = home.join("models");
            fs::create_dir(&models).unwrap();
            fs::write(
                models.join("weights.safetensors"),
                b"never remove model weights",
            )
            .unwrap();
            Self {
                home,
                hash,
                kv,
                history,
                lock,
            }
        }

        fn id(&self, kind: CacheKind) -> String {
            format!("{}:{}", kind.as_str(), self.hash)
        }

        fn hold_lock(&self) -> File {
            let mut options = OpenOptions::new();
            options.create(true).read(true).write(true);
            #[cfg(unix)]
            options.mode(0o600);
            let lock = options.open(&self.lock).unwrap();
            FileExt::try_lock_exclusive(&lock).unwrap();
            lock
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.home);
        }
    }

    #[test]
    fn inventory_distinguishes_native_cache_from_private_history_without_reading_it() {
        let fixture = Fixture::new();
        let entries = list_chat(&fixture.home).unwrap();
        assert_eq!(entries.len(), 2);
        let kv = entries
            .iter()
            .find(|entry| entry.kind == CacheKind::ChatKv)
            .unwrap();
        assert_eq!(kv.id, fixture.id(CacheKind::ChatKv));
        assert_eq!(kv.backend.as_deref(), Some("omlx"));
        assert_eq!(kv.bytes, Some(b"cache bytes".len() as u64));
        assert!(kv.modified_unix_seconds.is_some());
        assert!(!kv.active);
        assert!(kv.blocked_reason.is_none());
        let history = entries
            .iter()
            .find(|entry| entry.kind == CacheKind::ChatHistory)
            .unwrap();
        assert_eq!(
            history.bytes,
            Some(fs::metadata(&fixture.history).unwrap().len())
        );
        let json = serde_json::to_string(&entries).unwrap();
        assert!(!json.contains("private conversation"));
        assert!(!fixture.lock.exists(), "list must not create locks");
    }

    #[test]
    fn backend_detection_ignores_only_the_worker_coordination_marker() {
        let fixture = Fixture::new();
        fs::write(fixture.kv.join(".worker.lock"), b"coordination marker").unwrap();
        assert_eq!(
            chat_backend(&fixture.kv, CacheKind::ChatKv)
                .unwrap()
                .as_deref(),
            Some("omlx")
        );
        fs::create_dir(fixture.kv.join(".another-runtime")).unwrap();
        assert!(
            chat_backend(&fixture.kv, CacheKind::ChatKv)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn bulk_dry_run_is_read_only_and_default_purge_preserves_histories_and_models() {
        let fixture = Fixture::new();
        let selection = CacheSelection::All {
            kind: None,
            include_history: false,
        };
        let preview = purge(&fixture.home, &selection, true).unwrap();
        assert_eq!(preview.entries.len(), 1);
        assert_eq!(preview.entries[0].id, fixture.id(CacheKind::ChatKv));
        assert!(preview.removed.is_empty());
        assert!(preview.blocked.is_empty());
        assert!(fixture.kv.exists());
        assert!(fixture.history.exists());
        assert!(!fixture.lock.exists());
        let removed = purge(&fixture.home, &selection, false).unwrap();
        assert_eq!(removed.removed, vec![fixture.id(CacheKind::ChatKv)]);
        assert!(removed.blocked.is_empty());
        assert!(!fixture.kv.exists());
        assert!(fixture.history.exists());
        assert!(
            fixture.lock.is_file(),
            "stable session lock inode must be retained"
        );
        assert!(fixture.home.join("models/weights.safetensors").is_file());
    }

    #[test]
    fn busy_runtime_catalog_does_not_hide_or_block_idle_chat_caches() {
        let fixture = Fixture::new();
        let runtime = fixture.home.join("runtime-state");
        let catalog = runtime.join("v1");
        fs::create_dir_all(&catalog).unwrap();
        #[cfg(unix)]
        for directory in [&runtime, &catalog] {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut options = OpenOptions::new();
        options.create_new(true).read(true).write(true);
        #[cfg(unix)]
        options.mode(0o600);
        let catalog_lock = options.open(catalog.join(".lock")).unwrap();
        FileExt::try_lock_exclusive(&catalog_lock).unwrap();

        let inventory = list(&fixture.home).unwrap();
        assert_eq!(inventory.entries.len(), 2);
        assert_eq!(inventory.blocked.len(), 1);
        assert_eq!(inventory.blocked[0].id, "runtime-state");
        assert!(inventory.blocked[0].reason.contains("locked"));
        let filtered = purge(
            &fixture.home,
            &CacheSelection::All {
                kind: Some(CacheKind::ChatKv),
                include_history: false,
            },
            true,
        )
        .unwrap();
        assert_eq!(filtered.entries.len(), 1);
        assert!(filtered.blocked.is_empty());
        let all = CacheSelection::All {
            kind: None,
            include_history: false,
        };
        let preview = purge(&fixture.home, &all, true).unwrap();
        assert_eq!(preview.entries.len(), 1);
        assert_eq!(preview.blocked.len(), 1);
        assert_eq!(preview.blocked[0].id, "runtime-state");
        assert!(fixture.kv.exists());
        assert!(!fixture.lock.exists());

        let report = purge(&fixture.home, &all, false).unwrap();
        assert_eq!(report.removed, vec![fixture.id(CacheKind::ChatKv)]);
        assert_eq!(report.blocked.len(), 1);
        assert_eq!(report.blocked[0].id, "runtime-state");
        assert!(!fixture.kv.exists());
        assert!(fixture.history.exists());
        assert!(catalog.join(".lock").exists());
        assert!(fixture.home.join("models/weights.safetensors").exists());
        drop(catalog_lock);
    }

    #[test]
    fn explicit_history_id_and_history_filter_remove_only_selected_transcript() {
        for selection in [
            CacheSelection::Id(format!("chat-history:{}", "a".repeat(64))),
            CacheSelection::All {
                kind: Some(CacheKind::ChatHistory),
                include_history: false,
            },
        ] {
            let fixture = Fixture::new();
            let report = purge(&fixture.home, &selection, false).unwrap();
            assert_eq!(report.removed, vec![fixture.id(CacheKind::ChatHistory)]);
            assert!(!fixture.history.exists());
            assert!(fixture.kv.exists());
        }
    }

    #[test]
    fn explicit_bulk_history_inclusion_removes_both_types_but_keeps_lock_inode() {
        let fixture = Fixture::new();
        let lock = fixture.hold_lock();
        drop(lock);
        let before = fs::metadata(&fixture.lock).unwrap();
        let report = purge(
            &fixture.home,
            &CacheSelection::All {
                kind: None,
                include_history: true,
            },
            false,
        )
        .unwrap();
        assert_eq!(report.removed.len(), 2);
        assert!(report.blocked.is_empty());
        assert!(!fixture.history.exists());
        assert!(!fixture.kv.exists());
        #[cfg(unix)]
        assert_eq!(before.ino(), fs::metadata(&fixture.lock).unwrap().ino());
        #[cfg(not(unix))]
        let _ = before;
        assert!(fixture.lock.is_file());
    }

    #[test]
    fn active_chat_lock_blocks_both_data_types_and_releases_after_exit() {
        let fixture = Fixture::new();
        let lock = fixture.hold_lock();
        let entries = list_chat(&fixture.home).unwrap();
        assert!(entries.iter().all(|entry| entry.active));
        let report = purge(
            &fixture.home,
            &CacheSelection::All {
                kind: None,
                include_history: true,
            },
            false,
        )
        .unwrap();
        assert!(report.removed.is_empty());
        assert!(report.entries.is_empty());
        assert_eq!(report.blocked.len(), 2);
        assert!(fixture.kv.exists());
        assert!(fixture.history.exists());
        drop(lock);
        assert!(purge_chat(&fixture.home, &fixture.id(CacheKind::ChatKv), false).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn native_worker_lock_blocks_cache_after_main_chat_lock_releases() {
        let fixture = Fixture::new();
        let chat_lock = fixture.hold_lock();
        let worker_lock = workers::lock_persistent_worker_cache(&fixture.kv)
            .unwrap()
            .unwrap();
        drop(chat_lock);
        let entries = list_chat(&fixture.home).unwrap();
        let kv = entries
            .iter()
            .find(|entry| entry.kind == CacheKind::ChatKv)
            .unwrap();
        assert!(kv.active);
        assert_eq!(kv.backend.as_deref(), Some("omlx"));
        assert!(
            entries
                .iter()
                .find(|entry| entry.kind == CacheKind::ChatHistory)
                .is_some_and(|entry| !entry.active)
        );
        let selection = CacheSelection::Id(fixture.id(CacheKind::ChatKv));
        for dry_run in [true, false] {
            let report = purge(&fixture.home, &selection, dry_run).unwrap();
            assert!(report.entries.is_empty());
            assert!(report.removed.is_empty());
            assert_eq!(report.blocked.len(), 1);
            assert!(fixture.kv.exists());
        }
        drop(worker_lock);
        let report = purge(&fixture.home, &selection, false).unwrap();
        assert_eq!(report.removed, vec![fixture.id(CacheKind::ChatKv)]);
        assert!(!fixture.kv.exists());
        assert!(fixture.history.exists());
        assert!(fixture.lock.exists());
    }

    #[test]
    fn invalid_ids_and_missing_caches_never_create_files_or_touch_models() {
        let fixture = Fixture::new();
        for id in [
            "../../models",
            "chat-kv:../../models",
            "chat-kv:/tmp",
            "chat-kv:",
            "models:weights",
            &format!("chat-kv:{}", "f".repeat(64)),
        ] {
            let report = purge(&fixture.home, &CacheSelection::Id(id.into()), false).unwrap();
            assert!(report.entries.is_empty());
            assert_eq!(report.blocked.len(), 1);
            assert!(!fixture.lock.exists());
        }
        assert!(fixture.home.join("models/weights.safetensors").exists());
        assert!(fixture.kv.exists());
        assert!(fixture.history.exists());
    }

    #[test]
    fn excessive_nesting_is_reported_blocked_before_any_removal() {
        let fixture = Fixture::new();
        let mut path = fixture.kv.clone();
        for _ in 0..=MAX_TREE_DEPTH {
            path = path.join("nested");
        }
        fs::create_dir_all(path).unwrap();
        let entries = list_chat(&fixture.home).unwrap();
        let cache = entries
            .iter()
            .find(|entry| entry.kind == CacheKind::ChatKv)
            .unwrap();
        assert!(cache.bytes.is_none());
        assert!(cache.blocked_reason.is_some());
        assert!(purge_chat(&fixture.home, &fixture.id(CacheKind::ChatKv), false).is_err());
        assert!(fixture.kv.exists());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_cache_children_and_targets_are_not_followed_or_removed() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        let outside = fixture.home.join("models");
        let link = fixture.kv.join("outside");
        symlink(&outside, &link).unwrap();
        let entries = list_chat(&fixture.home).unwrap();
        let cache = entries
            .iter()
            .find(|entry| entry.kind == CacheKind::ChatKv)
            .unwrap();
        assert!(cache.bytes.is_none());
        assert!(cache.blocked_reason.as_ref().unwrap().contains("symlink"));
        assert!(purge_chat(&fixture.home, &fixture.id(CacheKind::ChatKv), false).is_err());
        assert!(outside.join("weights.safetensors").is_file());
        fs::remove_file(link).unwrap();
        fs::remove_dir_all(&fixture.kv).unwrap();
        symlink(&outside, &fixture.kv).unwrap();
        assert!(purge_chat(&fixture.home, &fixture.id(CacheKind::ChatKv), false).is_err());
        assert!(outside.join("weights.safetensors").is_file());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_session_locks_cannot_bypass_active_cache_protection() {
        let fixture = Fixture::new();
        let target = fixture.home.join("lock-target");
        fs::write(&target, "untouched").unwrap();
        std::os::unix::fs::symlink(&target, &fixture.lock).unwrap();
        assert!(
            list_chat(&fixture.home)
                .unwrap()
                .iter()
                .all(|entry| entry.blocked_reason.is_some())
        );
        assert!(purge_chat(&fixture.home, &fixture.id(CacheKind::ChatHistory), false).is_err());
        assert_eq!(fs::read_to_string(target).unwrap(), "untouched");
        assert!(fixture.history.exists());
    }
}
