//! Local build/backup/system-temp maintenance, separate from model persistence.
use anyhow::Result;
use fs2::FileExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, File, Metadata},
    io,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    Build,
    Backup,
    Shared,
    Temporary,
}
impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Backup => "backup",
            Self::Shared => "shared",
            Self::Temporary => "temporary",
        }
    }
}
pub struct Selection {
    pub id: Option<String>,
    pub all: bool,
    pub dry_run: bool,
}
fn cleanup_error(code: &str, message: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{code}: {message}")
}

pub struct Locations {
    pub workspace: PathBuf,
    pub state: PathBuf,
    pub home: PathBuf,
    pub cache: PathBuf,
    pub cargo: PathBuf,
    pub temporary: Vec<PathBuf>,
}

impl Locations {
    pub fn resolve(workspace: &Path) -> Result<Self> {
        let home = user_home()?;
        let root = fs::canonicalize(std::env::temp_dir())?;
        let temporary = if cfg!(unix) && root == Path::new("/tmp") && Path::new("/var/tmp").is_dir()
        {
            vec![root, PathBuf::from("/var/tmp")]
        } else {
            vec![root]
        };
        Ok(Self {
            workspace: fs::canonicalize(workspace)?,
            state: state_home()?,
            cache: env_path("XDG_CACHE_HOME").unwrap_or_else(|| home.join(".cache")),
            cargo: env_path("CARGO_HOME").unwrap_or_else(|| home.join(".cargo")),
            home,
            temporary,
        })
    }
}

pub fn temporary_candidates(
    home: &Path,
    store_entries: Vec<PathBuf>,
    roots: &[PathBuf],
) -> Result<Vec<Candidate>> {
    let mut out = Vec::new();
    if !store_entries.is_empty() {
        let root = fs::canonicalize(home)?;
        for path in store_entries {
            out.push(Candidate {
                path: root.join("tmp").join(path.file_name().unwrap()),
                root: root.clone(),
                kind: Kind::Temporary,
            });
        }
    }
    for root in roots {
        let root = fs::canonicalize(root)?;
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_lowercase();
            if ["werk1112-", "werk-", "werk_"]
                .iter()
                .any(|p| name.starts_with(p))
                && !name.starts_with("werk-worker-")
                && !name.starts_with("werk_doctor_")
            {
                out.push(Candidate {
                    root: root.clone(),
                    path: entry.path(),
                    kind: Kind::Temporary,
                });
            }
        }
    }
    Ok(out)
}

pub fn cache_candidates(
    locations: &Locations,
    kind: Option<Kind>,
    shared: bool,
) -> Result<Vec<Candidate>> {
    let mut out = Vec::new();
    if kind.is_none_or(|kind| kind == Kind::Build)
        && locations.workspace.join("Cargo.toml").is_file()
    {
        discover_builds(&locations.workspace, &locations.workspace, 0, &mut out)?;
    }
    if kind.is_none_or(|kind| kind == Kind::Backup) {
        for suffix in ["werk1112/backups", "werk-update-backups"] {
            add_children(&locations.state, suffix, Kind::Backup, &mut out)?;
        }
    }
    if (shared && kind.is_none()) || kind == Some(Kind::Shared) {
        for (root, suffix) in [
            (&locations.cache, "pip"),
            (&locations.cache, "uv"),
            (&locations.cache, "pipenv"),
            (&locations.cache, "vllm"),
            (&locations.home, ".npm/_cacache"),
            (&locations.home, ".nv/ComputeCache"),
            (&locations.home, ".triton/cache"),
            (&locations.cargo, "registry/cache"),
            (&locations.cargo, "registry/src"),
            (&locations.cargo, "registry/index"),
        ] {
            add(root, suffix, Kind::Shared, &mut out)?;
        }
    }
    Ok(out)
}

pub fn format_report(report: &Report, purge: bool) -> String {
    let mut lines = Vec::new();
    for entry in &report.entries {
        lines.push(format!("{}  {}", entry.id, entry.path.display()));
        lines.push(format!(
            "  {:.2} MiB | {} files | {}{}",
            entry.allocated_bytes as f64 / 1048576.0,
            entry.files,
            entry.blocked.as_deref().unwrap_or(if !purge {
                "available"
            } else if report.dry_run {
                "would remove"
            } else {
                "removed"
            }),
            if entry.size_complete {
                ""
            } else {
                " (size incomplete)"
            }
        ));
    }
    lines.push(format!(
        "{} entries; {:.2} MiB eligible; {} removed{}.",
        report.entries.len(),
        report.eligible_bytes as f64 / 1048576.0,
        report.removed.len(),
        if report.dry_run { " (dry run)" } else { "" }
    ));
    lines.join("\n")
}

#[derive(Clone)]
pub struct Candidate {
    pub root: PathBuf,
    pub path: PathBuf,
    pub kind: Kind,
}

#[derive(Serialize)]
pub struct Entry {
    pub id: String,
    pub kind: Kind,
    pub path: PathBuf,
    pub files: u64,
    pub allocated_bytes: u64,
    pub size_complete: bool,
    pub blocked: Option<String>,
}

#[derive(Default, Serialize)]
pub struct Report {
    pub dry_run: bool,
    pub entries: Vec<Entry>,
    pub eligible_bytes: u64,
    pub removed: Vec<PathBuf>,
    pub errors: Vec<String>,
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn user_home() -> Result<PathBuf> {
    env_path(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .ok_or_else(|| cleanup_error("cleanup_home", "User home directory is not configured"))
}

fn state_home() -> Result<PathBuf> {
    if cfg!(windows) {
        env_path("LOCALAPPDATA")
            .ok_or_else(|| cleanup_error("cleanup_home", "LOCALAPPDATA is not configured"))
    } else {
        match env_path("XDG_STATE_HOME") {
            Some(path) => Ok(path),
            None => Ok(user_home()?.join(".local/state")),
        }
    }
}

fn add(root: &Path, suffix: &str, kind: Kind, out: &mut Vec<Candidate>) -> Result<()> {
    if !root.is_absolute() {
        return Err(cleanup_error(
            "cleanup_path",
            "Cleanup roots must be absolute paths",
        ));
    }
    if !root.exists() {
        return Ok(());
    }
    let root = fs::canonicalize(root)?;
    let path = root.join(suffix);
    match fs::symlink_metadata(&path) {
        Ok(_) => out.push(Candidate { root, path, kind }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn add_children(root: &Path, suffix: &str, kind: Kind, out: &mut Vec<Candidate>) -> Result<()> {
    let mut parents = Vec::new();
    add(root, suffix, kind, &mut parents)?;
    for parent in parents {
        // Never traverse a redirected backup directory.
        if safe_path(&parent).is_err() {
            out.push(parent);
            continue;
        }
        for child in fs::read_dir(&parent.path)? {
            out.push(Candidate {
                path: child?.path(),
                ..parent.clone()
            });
        }
    }
    Ok(())
}

fn discover_builds(
    root: &Path,
    directory: &Path,
    depth: usize,
    out: &mut Vec<Candidate>,
) -> Result<()> {
    if directory.join("Cargo.toml").is_file() {
        let suffix = directory.strip_prefix(root).unwrap().join("target");
        add(root, &suffix.to_string_lossy(), Kind::Build, out)?;
    }
    if depth >= 4 {
        return Ok(());
    }
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || matches!(name.as_ref(), "target" | "node_modules" | "vendor") {
            continue;
        }
        if entry.file_type()?.is_dir() {
            discover_builds(root, &entry.path(), depth + 1, out)?;
        }
    }
    Ok(())
}

fn safe_path(candidate: &Candidate) -> io::Result<()> {
    let relative = candidate
        .path
        .strip_prefix(&candidate.root)
        .map_err(|_| io::Error::other("Path is outside its cleanup root"))?;
    if relative.as_os_str().is_empty() {
        return Err(io::Error::other("Cannot delete a cleanup root"));
    }
    let mut path = candidate.root.clone();
    for component in relative.components() {
        if !matches!(component, std::path::Component::Normal(_)) {
            return Err(io::Error::other("Invalid cleanup path"));
        }
        path.push(component);
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            return Err(io::Error::other("Symbolic links are protected"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if metadata.uid() != unsafe { libc::geteuid() } {
                return Err(io::Error::other("Entry belongs to another user"));
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct Usage {
    files: u64,
    bytes: u64,
    newest: Option<SystemTime>,
    seen: HashSet<(u64, u64)>,
    locks: Vec<File>,
    blocked: Option<String>,
}

fn measure(path: &Path, usage: &mut Usage) -> io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    usage.newest = usage.newest.max(Some(metadata.modified()?));
    if metadata.is_file() {
        usage.files += 1;
    }
    usage.bytes = usage
        .bytes
        .saturating_add(allocated(&metadata, &mut usage.seen));
    if metadata.file_type().is_symlink() {
        return Ok(());
    }
    if metadata.is_file()
        && path
            .file_name()
            .is_some_and(|name| name == ".cargo-lock" || name == ".package-cache")
    {
        let file = File::open(path)?;
        if file.try_lock_exclusive().is_ok() {
            usage.locks.push(file);
        } else {
            // Listing sizes remains useful during a build; never purge the tree.
            usage.blocked = Some("Cargo is using this directory".into());
        }
    }
    if metadata.is_dir() {
        for entry in fs::read_dir(path)? {
            measure(&entry?.path(), usage)?;
        }
    } else if !metadata.is_file() {
        return Err(io::Error::other("Special files are protected"));
    }
    Ok(())
}

fn allocated(metadata: &Metadata, seen: &mut HashSet<(u64, u64)>) -> u64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if seen.insert((metadata.dev(), metadata.ino())) {
            metadata.blocks().saturating_mul(512)
        } else {
            0
        }
    }
    #[cfg(not(unix))]
    {
        let _ = seen;
        metadata.len()
    }
}

/// Best-effort Linux process snapshot: never read process environments or file contents.
#[cfg(target_os = "linux")]
fn active_paths() -> Vec<(PathBuf, u32)> {
    let mut paths = Vec::new();
    if let Ok(processes) = fs::read_dir("/proc") {
        use std::os::unix::fs::MetadataExt;
        for process in processes.flatten() {
            let Ok(pid) = process.file_name().to_string_lossy().parse::<u32>() else {
                continue;
            };
            if pid == std::process::id()
                || process
                    .metadata()
                    .map_or(true, |m| m.uid() != unsafe { libc::geteuid() })
            {
                continue;
            }
            for name in ["exe", "cwd"] {
                if let Ok(path) = fs::read_link(process.path().join(name)) {
                    paths.push((path, pid));
                }
            }
            if let Ok(fds) = fs::read_dir(process.path().join("fd")) {
                for fd in fds.flatten() {
                    if let Ok(path) = fs::read_link(fd.path()) {
                        paths.push((path, pid));
                    }
                }
            }
        }
    }
    paths
}

#[cfg(not(target_os = "linux"))]
fn active_paths() -> Vec<(PathBuf, u32)> {
    Vec::new()
}

fn inspect(candidate: &Candidate, hours: u64, active: &[(PathBuf, u32)]) -> (Entry, Usage) {
    let hash = format!(
        "{:x}",
        Sha256::digest(format!("{:?}", candidate.path).as_bytes())
    )[..16]
        .to_owned();
    let id = format!("{}:{hash}", candidate.kind.as_str());
    let mut usage = Usage::default();
    let measured = safe_path(candidate).and_then(|()| measure(&candidate.path, &mut usage));
    let size_complete = measured.is_ok();
    let mut blocked = measured
        .err()
        .map(|e| e.to_string())
        .or_else(|| usage.blocked.clone());
    #[cfg(target_os = "linux")]
    if blocked.is_none() && candidate.kind == Kind::Temporary {
        use std::os::unix::fs::MetadataExt;
        let name = candidate
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        for prefix in [
            "werk1112-api-session-cache-",
            "werk-candle-residency-",
            "werk-burn-residency-",
            "werk-cpu-prefetch-",
        ] {
            if let Some(pid) = name
                .strip_prefix(prefix)
                .and_then(|tail| tail.split('-').next())
                .and_then(|pid| pid.parse::<u32>().ok())
                && fs::metadata(format!("/proc/{pid}"))
                    .is_ok_and(|metadata| metadata.uid() == unsafe { libc::geteuid() })
            {
                blocked = Some(format!("Owned by live Werk process {pid}"));
            }
        }
    }
    if blocked.is_none()
        && hours > 0
        && usage.newest.is_some_and(|time| {
            SystemTime::now().duration_since(time).unwrap_or_default()
                < Duration::from_secs(hours.saturating_mul(3600))
        })
    {
        blocked = Some(format!("Modified within the last {hours} hours"));
    }
    if blocked.is_none()
        && let Some((_, pid)) = active
            .iter()
            .find(|(path, _)| path.starts_with(&candidate.path))
    {
        blocked = Some(format!("In use by process {pid}"));
    }
    (
        Entry {
            id,
            kind: candidate.kind,
            path: candidate.path.clone(),
            files: usage.files,
            allocated_bytes: usage.bytes,
            size_complete,
            blocked,
        },
        usage,
    )
}

pub fn run(
    mut candidates: Vec<Candidate>,
    hours: u64,
    selection: Option<&Selection>,
) -> Result<Report> {
    candidates.sort_by(|a, b| a.path.cmp(&b.path));
    candidates.dedup_by(|a, b| a.path == b.path);
    let active = active_paths();
    let mut report = Report {
        dry_run: selection.is_some_and(|s| s.dry_run),
        ..Default::default()
    };
    let mut selected = false;
    for candidate in candidates {
        let (mut entry, _locks) = inspect(&candidate, hours, &active);
        if let Some(selection) = selection {
            if !selection.all && selection.id.as_deref() != Some(&entry.id) {
                continue;
            }
            selected = true;
            if entry.blocked.is_none() && !selection.dry_run {
                // Recheck roots immediately before removal. remove_dir_all does not follow child links.
                let removed = safe_path(&candidate).and_then(|()| {
                    if fs::symlink_metadata(&candidate.path)?.is_dir() {
                        fs::remove_dir_all(&candidate.path)
                    } else {
                        fs::remove_file(&candidate.path)
                    }
                });
                match removed {
                    Ok(()) => report.removed.push(candidate.path.clone()),
                    Err(error) => {
                        let message = format!("{}: {error}", candidate.path.display());
                        entry.blocked = Some(message.clone());
                        report.errors.push(message);
                    }
                }
            }
        }
        if entry.blocked.is_none() {
            report.eligible_bytes = report.eligible_bytes.saturating_add(entry.allocated_bytes);
        }
        report.entries.push(entry);
    }
    if selection.is_some_and(|s| s.id.is_some()) && !selected {
        return Err(cleanup_error(
            "cleanup_not_found",
            "Unknown ID in this scope; run list with the same options first",
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{self, CacheKind, CacheSelection};
    struct Fixture {
        root: tempfile::TempDir,
        locations: Locations,
    }
    impl Fixture {
        fn new() -> Self {
            let root = tempfile::tempdir().unwrap();
            for name in [
                "workspace",
                "state",
                "home",
                "cache",
                "cargo",
                "tmp",
                "store",
            ] {
                fs::create_dir(root.path().join(name)).unwrap();
            }
            let locations = Locations {
                workspace: root.path().join("workspace"),
                state: root.path().join("state"),
                home: root.path().join("home"),
                cache: root.path().join("cache"),
                cargo: root.path().join("cargo"),
                temporary: vec![root.path().join("tmp")],
            };
            Self { root, locations }
        }
        fn path(&self, name: &str) -> PathBuf {
            self.root.path().join(name)
        }
        fn write(&self, name: &str) {
            let path = self.path(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, "fixture").unwrap();
        }
        fn temps(&self) -> Vec<Candidate> {
            temporary_candidates(&self.path("store"), vec![], &self.locations.temporary).unwrap()
        }
    }
    fn all() -> Selection {
        Selection {
            id: None,
            all: true,
            dry_run: false,
        }
    }

    #[test]
    fn expanded_temp_cleanup_keeps_recent_and_unrelated_entries_and_dry_run_is_read_only() {
        let f = Fixture::new();
        for name in [
            "tmp/werk-old.log",
            "tmp/werk1112-recent/file",
            "tmp/ww-other.log",
            "tmp/werk-worker-other.log",
            "tmp/werk_doctor_other",
            "tmp/unrelated",
            "store/tmp/pull",
            "store/models/model",
        ] {
            f.write(name);
        }
        File::options()
            .write(true)
            .open(f.path("tmp/werk-old.log"))
            .unwrap()
            .set_modified(SystemTime::now() - Duration::from_secs(48 * 3600))
            .unwrap();
        let candidates = temporary_candidates(
            &f.path("store"),
            vec![f.path("store/tmp/pull")],
            &f.locations.temporary,
        )
        .unwrap();
        let report = run(candidates.clone(), 24, None).unwrap();
        assert_eq!(report.entries.len(), 3);
        assert_eq!(
            report
                .entries
                .iter()
                .filter(|e| e.blocked.is_none())
                .count(),
            1
        );
        run(
            candidates.clone(),
            0,
            Some(&Selection {
                dry_run: true,
                ..all()
            }),
        )
        .unwrap();
        assert!(f.path("tmp/werk-old.log").exists());
        run(candidates, 24, Some(&all())).unwrap();
        assert!(!f.path("tmp/werk-old.log").exists());
        assert!(f.path("tmp/werk1112-recent/file").exists());
        run(f.temps(), 0, Some(&all())).unwrap();
        for name in [
            "tmp/ww-other.log",
            "tmp/werk-worker-other.log",
            "tmp/werk_doctor_other",
            "tmp/unrelated",
            "store/tmp/pull",
            "store/models/model",
        ] {
            assert!(f.path(name).exists(), "{name}");
        }
    }

    #[test]
    fn cache_all_includes_builds_and_backups_but_shared_caches_need_opt_in() {
        let f = Fixture::new();
        for name in [
            "workspace/Cargo.toml",
            "workspace/src/main.rs",
            "workspace/target/build.o",
            "state/werk1112/backups/old/werk",
            "cache/pip/download",
            "store/models/model",
            "store/auth/token",
            "cache/huggingface/hub/model",
            "cache/huggingface/token",
            "cargo/bin/werk",
        ] {
            f.write(name);
        }
        let home = f.path("store");
        let inventory = cache::list_local(&home, &f.locations, None, false).unwrap();
        assert_eq!(inventory.entries.len(), 2);
        let selection = CacheSelection::All {
            kind: None,
            include_history: false,
        };
        cache::purge_local(&home, &f.locations, &selection, false, true).unwrap();
        assert!(f.path("workspace/target/build.o").exists());
        let report = cache::purge_local(&home, &f.locations, &selection, false, false).unwrap();
        assert_eq!(report.removed.len(), 2);
        assert!(f.path("cache/pip/download").exists());
        let inventory =
            cache::list_local(&home, &f.locations, Some(CacheKind::Shared), false).unwrap();
        assert_eq!(inventory.entries.len(), 1);
        cache::purge_local(
            &home,
            &f.locations,
            &CacheSelection::Id(inventory.entries[0].id.clone()),
            false,
            false,
        )
        .unwrap();
        assert!(!f.path("cache/pip").exists());
        for name in [
            "workspace/src/main.rs",
            "store/models/model",
            "store/auth/token",
            "cache/huggingface/hub/model",
            "cache/huggingface/token",
            "cargo/bin/werk",
        ] {
            assert!(f.path(name).exists(), "{name}");
        }
        assert!(
            cache::purge_local(
                &home,
                &f.locations,
                &CacheSelection::Id("build:../../etc".into()),
                false,
                false
            )
            .is_err()
        );
    }

    #[test]
    fn cargo_lock_protects_builds_while_size_remains_complete() {
        let f = Fixture::new();
        f.write("workspace/Cargo.toml");
        f.write("workspace/target/debug/.cargo-lock");
        f.write("workspace/target/debug/compiled.o");
        let lock = File::open(f.path("workspace/target/debug/.cargo-lock")).unwrap();
        lock.lock_exclusive().unwrap();
        let candidates = cache_candidates(&f.locations, Some(Kind::Build), false).unwrap();
        let report = run(candidates, 0, Some(&all())).unwrap();
        assert!(report.removed.is_empty());
        assert!(report.entries[0].size_complete);
        assert_eq!(report.entries[0].files, 2);
        assert!(
            report.entries[0]
                .blocked
                .as_ref()
                .unwrap()
                .contains("Cargo")
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_and_parent_redirections_do_not_expand_deletion_scope() {
        use std::os::unix::fs::symlink;
        let f = Fixture::new();
        f.write("outside/keep");
        f.write("tmp/werk-owned/file");
        symlink(f.path("outside"), f.path("tmp/werk-outside")).unwrap();
        symlink(f.path("outside"), f.path("tmp/werk-owned/link")).unwrap();
        run(f.temps(), 0, Some(&all())).unwrap();
        assert!(f.path("outside/keep").exists());
        assert!(f.path("tmp/werk-outside").is_symlink());
        fs::create_dir(f.path("state/werk1112")).unwrap();
        symlink(f.path("outside"), f.path("state/werk1112/backups")).unwrap();
        let report = run(
            cache_candidates(&f.locations, Some(Kind::Backup), false).unwrap(),
            0,
            Some(&all()),
        )
        .unwrap();
        assert!(report.entries[0].blocked.is_some());
        assert!(f.path("outside/keep").exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn active_temporary_work_is_skipped() {
        let f = Fixture::new();
        f.write("tmp/werk-active/file");
        f.write(&format!(
            "tmp/werk1112-api-session-cache-{}/file",
            std::process::id()
        ));
        let mut process = std::process::Command::new("sleep")
            .arg("30")
            .current_dir(f.path("tmp/werk-active"))
            .spawn()
            .unwrap();
        let result = run(f.temps(), 0, Some(&all()));
        let _ = process.kill();
        let _ = process.wait();
        let report = result.unwrap();
        assert_eq!(report.entries.len(), 2);
        assert!(
            report
                .entries
                .iter()
                .all(|entry| entry.blocked.as_ref().unwrap().contains("process"))
        );
        assert!(f.path("tmp/werk-active/file").exists());
    }
}
