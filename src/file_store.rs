//! Local file objects shared by API adapters, partitioned by authenticated principal.
use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

pub const MAX_FILE_BYTES: usize = 50 * 1024 * 1024;
const MAX_FILES: usize = 1024;
const MAX_STORE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_PRINCIPAL_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone)]
pub struct FileStore {
    root: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredFile {
    pub id: String,
    pub filename: String,
    pub mime_type: String,
    pub bytes: u64,
    pub created_at: u64,
    pub expires_at: Option<u64>,
    pub purpose: String,
}

impl FileStore {
    pub fn new(home: &Path) -> Self {
        Self {
            root: home.join("api-files"),
        }
    }

    fn lock(&self) -> Result<File> {
        fs::create_dir_all(self.root.parent().context("invalid file store root")?)?;
        private_dir(&self.root)?;
        let lock = private_file(&self.root.join(".lock"), false)?;
        lock.lock_exclusive()?;
        Ok(lock)
    }

    fn namespace(&self, principal: &str) -> Result<PathBuf> {
        ensure!(safe_component(principal), "invalid file namespace");
        Ok(self.root.join(principal))
    }

    fn path(&self, principal: &str, id: &str) -> Result<PathBuf> {
        ensure!(
            id.starts_with("file_") && safe_component(id),
            "file not found"
        );
        let ns = self.namespace(principal)?;
        ensure_plain_dir(&ns)?;
        let path = ns.join(id);
        ensure_plain_dir(&path)?;
        Ok(path)
    }

    /// The cross-process lock covers quota accounting, expiration and publication.
    /// No file is evicted to admit another upload.
    pub fn put(
        &self,
        principal: &str,
        filename: String,
        mime_type: String,
        purpose: String,
        expires_at: Option<u64>,
        data: &[u8],
    ) -> Result<StoredFile> {
        ensure!(
            !data.is_empty() && data.len() <= MAX_FILE_BYTES,
            "file must contain 1..50 MiB"
        );
        ensure!(
            !filename.is_empty()
                && filename.len() <= 255
                && !filename
                    .chars()
                    .any(|c| c.is_control() || "<>:\"|?*\\/".contains(c)),
            "invalid filename"
        );
        ensure!(
            !mime_type.is_empty()
                && mime_type.len() <= 255
                && !mime_type.chars().any(char::is_control),
            "invalid MIME type"
        );
        ensure!(
            expires_at.is_none_or(|at| at > crate::model_store::unix_ts()),
            "file expiration must be in the future"
        );
        let _lock = self.lock()?;
        let entries = self.scan()?;
        let total: u64 = entries.iter().map(|(_, f)| f.bytes).sum();
        let owned: Vec<_> = entries.iter().filter(|(p, _)| p == principal).collect();
        let owned_bytes: u64 = owned.iter().map(|(_, f)| f.bytes).sum();
        ensure!(
            entries.len() < MAX_FILES
                && owned.len() < 256
                && total + data.len() as u64 <= MAX_STORE_BYTES
                && owned_bytes + data.len() as u64 <= MAX_PRINCIPAL_BYTES,
            "file storage quota exceeded; delete unused files"
        );
        let ns = self.namespace(principal)?;
        private_dir(&ns)?;
        let mut random = [0_u8; 24];
        getrandom::getrandom(&mut random)
            .map_err(|_| anyhow::anyhow!("file ID generation failed"))?;
        let id = format!(
            "file_{}",
            random
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );
        let entry = StoredFile {
            id: id.clone(),
            filename,
            mime_type,
            bytes: data.len() as u64,
            created_at: crate::model_store::unix_ts(),
            expires_at,
            purpose,
        };
        let stage = tempfile::Builder::new().prefix(".stage-").tempdir_in(&ns)?;
        let mut blob = private_file(&stage.path().join("data"), true)?;
        blob.write_all(data)?;
        blob.sync_all()?;
        let mut metadata = private_file(&stage.path().join("metadata.json"), true)?;
        metadata.write_all(&serde_json::to_vec(&entry)?)?;
        metadata.sync_all()?;
        drop(blob);
        drop(metadata);
        fs::rename(stage.path(), ns.join(id))?;
        Ok(entry)
    }

    pub fn list(&self, principal: &str) -> Result<Vec<StoredFile>> {
        let _lock = self.lock()?;
        Ok(self
            .scan()?
            .into_iter()
            .filter(|(p, _)| p == principal)
            .map(|(_, f)| f)
            .collect())
    }

    pub fn get(&self, principal: &str, id: &str) -> Result<(StoredFile, Vec<u8>)> {
        let _lock = self.lock()?;
        let path = self.path(principal, id)?;
        let meta = read_metadata(&path)?;
        if meta
            .expires_at
            .is_some_and(|at| at <= crate::model_store::unix_ts())
        {
            fs::remove_dir_all(path)?;
            bail!("file not found");
        }
        ensure!(meta.id == id, "invalid file metadata");
        let data = read_regular(&path.join("data"), MAX_FILE_BYTES)?;
        ensure!(data.len() as u64 == meta.bytes, "stored file size mismatch");
        Ok((meta, data))
    }

    pub fn metadata(&self, principal: &str, id: &str) -> Result<StoredFile> {
        let _lock = self.lock()?;
        let path = self.path(principal, id)?;
        let meta = read_metadata(&path)?;
        ensure!(meta.id == id, "invalid file metadata");
        if meta
            .expires_at
            .is_some_and(|at| at <= crate::model_store::unix_ts())
        {
            fs::remove_dir_all(path)?;
            bail!("file not found");
        }
        Ok(meta)
    }

    pub fn delete(&self, principal: &str, id: &str) -> Result<()> {
        let _lock = self.lock()?;
        let path = self.path(principal, id)?;
        read_metadata(&path)?;
        fs::remove_dir_all(path)?;
        Ok(())
    }

    fn scan(&self) -> Result<Vec<(String, StoredFile)>> {
        let mut result = Vec::new();
        let mut visited = 0;
        for ns in fs::read_dir(&self.root)? {
            visited += 1;
            ensure!(
                visited <= MAX_FILES * 3,
                "file inventory exceeds safety limit"
            );
            let ns = ns?;
            if ns.file_name() == ".lock" {
                continue;
            }
            ensure_plain_dir(&ns.path())?;
            let principal = ns
                .file_name()
                .to_str()
                .context("invalid file namespace")?
                .to_owned();
            ensure!(safe_component(&principal), "invalid file namespace");
            for path in fs::read_dir(ns.path())? {
                let path = path?;
                visited += 1;
                ensure!(
                    visited <= MAX_FILES * 3,
                    "file inventory exceeds safety limit"
                );
                ensure_plain_dir(&path.path())?;
                if path.file_name().to_string_lossy().starts_with(".stage-") {
                    // No writer can be active while we own the store lock.
                    fs::remove_dir_all(path.path())?;
                    continue;
                }
                let meta = read_metadata(&path.path())?;
                ensure!(
                    path.file_name().to_str() == Some(meta.id.as_str()),
                    "invalid file metadata"
                );
                if meta
                    .expires_at
                    .is_some_and(|at| at <= crate::model_store::unix_ts())
                {
                    fs::remove_dir_all(path.path())?;
                } else {
                    result.push((principal.clone(), meta));
                }
            }
            if fs::read_dir(ns.path())?.next().is_none() {
                fs::remove_dir(ns.path())?;
            }
        }
        Ok(result)
    }
}

fn safe_component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 100
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
}
fn ensure_plain_dir(path: &Path) -> Result<()> {
    let meta = fs::symlink_metadata(path).map_err(|_| anyhow::anyhow!("file not found"))?;
    ensure!(
        meta.is_dir() && !meta.file_type().is_symlink(),
        "invalid file storage directory"
    );
    Ok(())
}
fn private_dir(path: &Path) -> Result<()> {
    match fs::create_dir(path) {
        Ok(()) => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(e) => return Err(e.into()),
    }
    ensure_plain_dir(path)
}
fn private_file(path: &Path, create_new: bool) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    if create_new {
        options.create_new(true);
    } else {
        options.create(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000);
    }
    let file = options.open(path)?;
    ensure!(file.metadata()?.is_file(), "invalid file storage entry");
    Ok(file)
}
fn read_regular(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink() && metadata.len() <= limit as u64,
        "invalid file storage entry"
    );
    let mut data = Vec::new();
    File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut data)?;
    ensure!(data.len() <= limit, "file exceeds limit");
    Ok(data)
}
fn read_metadata(path: &Path) -> Result<StoredFile> {
    let meta: StoredFile =
        serde_json::from_slice(&read_regular(&path.join("metadata.json"), 8192)?)?;
    ensure!(
        meta.bytes <= MAX_FILE_BYTES as u64,
        "invalid stored file size"
    );
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn put(store: &FileStore) -> StoredFile {
        store
            .put(
                "alice",
                "a.txt".into(),
                "text/plain".into(),
                "user_data".into(),
                None,
                b"hello",
            )
            .unwrap()
    }
    #[test]
    fn expiration_reclaims_bytes_and_orphan_staging_is_removed() {
        let root = tempfile::tempdir().unwrap();
        let store = FileStore::new(root.path());
        let mut file = put(&store);
        file.expires_at = Some(1);
        fs::write(
            store
                .root
                .join("alice")
                .join(&file.id)
                .join("metadata.json"),
            serde_json::to_vec(&file).unwrap(),
        )
        .unwrap();
        let stale = store.root.join("alice/.stage-abandoned");
        fs::create_dir(&stale).unwrap();
        fs::write(stale.join("data"), b"abandoned").unwrap();
        assert!(store.list("alice").unwrap().is_empty());
        assert!(!stale.exists());
        assert!(!store.root.join("alice").join(file.id).exists());
    }
    #[test]
    fn quota_rejection_never_evicts_existing_files() {
        let root = tempfile::tempdir().unwrap();
        let store = FileStore::new(root.path());
        let first = put(&store);
        // Account genuine-size metadata without allocating hundreds of MiB in a unit test.
        for _ in 0..5 {
            let mut file = put(&store);
            file.bytes = MAX_FILE_BYTES as u64;
            fs::write(
                store
                    .root
                    .join("alice")
                    .join(&file.id)
                    .join("metadata.json"),
                serde_json::to_vec(&file).unwrap(),
            )
            .unwrap();
        }
        let mut file = put(&store);
        file.bytes = 6 * 1024 * 1024;
        fs::write(
            store
                .root
                .join("alice")
                .join(&file.id)
                .join("metadata.json"),
            serde_json::to_vec(&file).unwrap(),
        )
        .unwrap();
        assert!(
            store
                .put(
                    "alice",
                    "new.txt".into(),
                    "text/plain".into(),
                    "user_data".into(),
                    None,
                    b"no"
                )
                .unwrap_err()
                .to_string()
                .contains("quota")
        );
        assert_eq!(store.get("alice", &first.id).unwrap().1, b"hello");
    }
    #[test]
    fn file_ids_and_principals_cannot_escape_storage() {
        let root = tempfile::tempdir().unwrap();
        let store = FileStore::new(root.path());
        let file = put(&store);
        assert!(store.get("bob", &file.id).is_err());
        assert!(store.get("../alice", &file.id).is_err());
        assert!(store.get("alice", "file_../../outside").is_err());
    }
    #[cfg(unix)]
    #[test]
    fn symlinked_blob_cannot_be_read() {
        let root = tempfile::tempdir().unwrap();
        let store = FileStore::new(root.path());
        let file = put(&store);
        let blob = store.root.join("alice").join(&file.id).join("data");
        fs::remove_file(&blob).unwrap();
        let outside = root.path().join("outside");
        fs::write(&outside, b"secret").unwrap();
        std::os::unix::fs::symlink(&outside, &blob).unwrap();
        assert!(store.get("alice", &file.id).is_err());
    }
}
