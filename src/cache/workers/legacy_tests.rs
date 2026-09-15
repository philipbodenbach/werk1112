#![cfg(unix)]

use super::*;
use crate::cache::CacheSelection;
use std::time::SystemTime;

struct Fixture {
    home: PathBuf,
    id: String,
    worker: PathBuf,
    cache: PathBuf,
    history: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let mut bytes = [0u8; 8];
        getrandom::getrandom(&mut bytes).unwrap();
        let home = std::env::temp_dir().join(format!(
            "werk-legacy-worker-cache-{:x}",
            u64::from_ne_bytes(bytes)
        ));
        fs::create_dir(&home).unwrap();
        fs::set_permissions(&home, fs::Permissions::from_mode(0o700)).unwrap();
        let id = "c".repeat(48);
        let worker = home.join("backends/omlx/workers").join(&id);
        let cache = worker.join("cache");
        fs::create_dir_all(cache.join("0/nested")).unwrap();
        fs::create_dir_all(cache.join("response-state/nested")).unwrap();
        fs::write(worker.join("settings.json"), b"preserve settings").unwrap();
        fs::write(worker.join("server.log"), b"preserve log").unwrap();
        let chat_root = home.join("chat-sessions");
        fs::create_dir(&chat_root).unwrap();
        fs::set_permissions(&chat_root, fs::Permissions::from_mode(0o700)).unwrap();
        let history = chat_root.join(format!("{}.json", "d".repeat(64)));
        fs::write(&history, b"preserve conversation").unwrap();
        fs::write(home.join("model.safetensors"), b"preserve model").unwrap();
        Self {
            home,
            id,
            worker,
            cache,
            history,
        }
    }

    fn cache_id(&self) -> String {
        format!("omlx-worker:{}", self.id)
    }

    fn owner(&self, pid: u32, suffix: char) -> PathBuf {
        let name = format!("{pid}-{}", suffix.to_string().repeat(32));
        let path = self.cache.join("_boundary_snapshots").join(name);
        fs::create_dir_all(path.join("request/nested")).unwrap();
        path
    }

    fn assert_preserved(&self) {
        assert_eq!(
            fs::read(self.worker.join("settings.json")).unwrap(),
            b"preserve settings"
        );
        assert_eq!(
            fs::read(self.worker.join("server.log")).unwrap(),
            b"preserve log"
        );
        assert_eq!(fs::read(&self.history).unwrap(), b"preserve conversation");
        assert_eq!(
            fs::read(self.home.join("model.safetensors")).unwrap(),
            b"preserve model"
        );
    }

    fn assert_blocked(&self) -> CacheEntry {
        let before = snapshot(&self.home);
        let entry = list(&self.home).unwrap().pop().unwrap();
        assert!(entry.blocked_reason.is_some());
        assert!(purge(&self.home, &self.cache_id(), true).is_err());
        assert_eq!(snapshot(&self.home), before);
        assert!(purge(&self.home, &self.cache_id(), false).is_err());
        assert!(self.cache.exists());
        self.assert_preserved();
        entry
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.home);
    }
}

fn reaped_child_pid() -> u32 {
    let mut child = Command::new("/bin/sh")
        .args(["-c", "exit 0"])
        .spawn()
        .unwrap();
    let pid = child.id();
    assert!(child.wait().unwrap().success());
    assert_eq!(unsafe { libc::kill(pid as libc::pid_t, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH),
        "a reused PID must never be treated as an exited owner"
    );
    pid
}

#[derive(Debug, PartialEq, Eq)]
struct SnapshotEntry {
    path: PathBuf,
    directory: bool,
    symlink: bool,
    bytes: u64,
    modified: Option<SystemTime>,
}

fn snapshot(root: &Path) -> Vec<SnapshotEntry> {
    let mut pending = vec![root.to_path_buf()];
    let mut entries = Vec::new();
    while let Some(path) = pending.pop() {
        let metadata = fs::symlink_metadata(&path).unwrap();
        if metadata.is_dir() {
            pending.extend(
                fs::read_dir(&path)
                    .unwrap()
                    .map(|entry| entry.unwrap().path()),
            );
        }
        entries.push(SnapshotEntry {
            path: path.strip_prefix(root).unwrap().to_path_buf(),
            directory: metadata.is_dir(),
            symlink: metadata.file_type().is_symlink(),
            bytes: metadata.len(),
            modified: metadata.modified().ok(),
        });
    }
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    entries
}

#[test]
fn exited_legacy_empty_cache_lists_and_dry_runs_without_writes_then_purges_only_cache() {
    for bulk in [false, true] {
        let fixture = Fixture::new();
        let pid = reaped_child_pid();
        fixture.owner(pid, 'a');
        fixture.owner(pid, 'b');
        let before = snapshot(&fixture.home);
        let entry = list(&fixture.home).unwrap().pop().unwrap();
        assert_eq!(entry.id, fixture.cache_id());
        assert_eq!(entry.bytes, Some(0));
        assert!(!entry.active);
        assert!(entry.blocked_reason.is_none());
        let preview = purge(&fixture.home, &fixture.cache_id(), true).unwrap();
        assert_eq!(preview.bytes, Some(0));
        assert_eq!(snapshot(&fixture.home), before);
        assert!(!fixture.worker.parent().unwrap().join(".locks").exists());

        if bulk {
            let report = crate::cache::purge(
                &fixture.home,
                &CacheSelection::All {
                    kind: None,
                    include_history: false,
                },
                false,
            )
            .unwrap();
            assert_eq!(report.removed, vec![fixture.cache_id()]);
            assert!(report.blocked.is_empty());
        } else {
            purge(&fixture.home, &fixture.cache_id(), false).unwrap();
        }
        assert!(!fixture.cache.exists());
        assert!(fixture.worker.is_dir());
        fixture.assert_preserved();
        assert!(list(&fixture.home).unwrap().is_empty());
    }
}

#[test]
fn any_live_legacy_owner_keeps_even_empty_cache_active() {
    let fixture = Fixture::new();
    fixture.owner(reaped_child_pid(), 'a');
    fixture.owner(std::process::id(), 'b');
    let entry = fixture.assert_blocked();
    assert!(entry.active);
    assert_eq!(entry.bytes, Some(0));
}

#[test]
fn zero_length_file_disqualifies_an_exited_legacy_empty_cache() {
    let fixture = Fixture::new();
    let owner = fixture.owner(reaped_child_pid(), 'a');
    fs::write(owner.join("request/zero-byte.safetensors"), b"").unwrap();
    let entry = fixture.assert_blocked();
    assert_eq!(entry.bytes, Some(0));
}

#[test]
fn missing_or_malformed_legacy_owner_markers_remain_blocked() {
    let pid = reaped_child_pid();
    for marker in [
        None,
        Some(String::new()),
        Some("unknown-owner".into()),
        Some(format!("0-{}", "a".repeat(32))),
        Some(format!("-1-{}", "a".repeat(32))),
        Some(format!("{pid}-{}", "A".repeat(32))),
        Some(format!("{pid}-short")),
        Some(format!("{pid}-{}-extra", "a".repeat(32))),
    ] {
        let fixture = Fixture::new();
        if let Some(marker) = marker {
            let owners = fixture.cache.join("_boundary_snapshots");
            fs::create_dir_all(owners.join(marker)).unwrap();
        }
        fixture.assert_blocked();
    }
}

#[test]
fn active_new_worker_lock_cannot_be_bypassed_with_dead_legacy_markers() {
    let fixture = Fixture::new();
    let new_id = "e".repeat(48);
    let (worker, guard) = prepare_worker(&fixture.home, &new_id).unwrap();
    let cache = worker.join("cache");
    fs::create_dir_all(cache.join("_boundary_snapshots").join(format!(
        "{}-{}",
        reaped_child_pid(),
        "f".repeat(32)
    )))
    .unwrap();
    let cache_id = format!("omlx-worker:{new_id}");
    let entry = list(&fixture.home)
        .unwrap()
        .into_iter()
        .find(|entry| entry.id == cache_id)
        .unwrap();
    assert!(entry.active);
    assert!(entry.blocked_reason.is_some());
    assert_eq!(entry.bytes, Some(0));
    for dry_run in [true, false] {
        assert!(purge(&fixture.home, &cache_id, dry_run).is_err());
    }
    assert!(cache.exists());
    drop(guard);
}

#[test]
fn symlink_in_exited_legacy_cache_is_never_followed_or_removed() {
    let fixture = Fixture::new();
    fixture.owner(reaped_child_pid(), 'a');
    let outside = fixture.home.join("unrelated-empty-directory");
    fs::create_dir(&outside).unwrap();
    let link = fixture.cache.join("0/link");
    std::os::unix::fs::symlink(&outside, &link).unwrap();
    fixture.assert_blocked();
    assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink());
    assert!(outside.is_dir());
}

#[test]
fn unknown_legacy_directory_layout_remains_protected_even_when_empty() {
    let fixture = Fixture::new();
    fixture.owner(reaped_child_pid(), 'a');
    fs::create_dir(fixture.cache.join("unrecognized-cache-data")).unwrap();
    fixture.assert_blocked();
}

#[test]
fn file_created_after_legacy_inspection_is_preserved_by_atomic_directory_removal() {
    let fixture = Fixture::new();
    let owner = fixture.owner(reaped_child_pid(), 'a');
    let plan = LegacyEmptyCache::inspect(&fixture.cache).unwrap();
    let new_file = owner.join("request/nested/late-write.safetensors");
    fs::write(&new_file, b"written after inspection").unwrap();
    assert!(plan.remove().is_err());
    assert_eq!(fs::read(new_file).unwrap(), b"written after inspection");
    assert!(fixture.cache.is_dir());
    fixture.assert_preserved();
}
