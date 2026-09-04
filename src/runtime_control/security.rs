use crate::model_store::ModelStore;
use fs2::FileExt;
use hmac::{Hmac, Mac};
use sha2::Sha256;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt, PermissionsExt};
#[cfg(windows)]
use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

type HmacSha256 = Hmac<Sha256>;
const SECRET_BYTES: usize = 32;
const MAX_AUTH_DIRECTORY_ENTRIES: usize = 256;
const MAX_STAGING_KEYS_PER_CLEANUP: usize = 64;

#[derive(Clone)]
pub(crate) struct PrincipalDeriver {
    path: PathBuf,
    secret: Arc<OnceLock<[u8; SECRET_BYTES]>>,
    load_gate: Arc<Mutex<()>>,
}

impl PrincipalDeriver {
    pub fn new(store: &ModelStore) -> Self {
        Self {
            path: store.home().join("auth").join("runtime-namespace.key"),
            secret: Arc::new(OnceLock::new()),
            load_gate: Arc::new(Mutex::new(())),
        }
    }

    pub fn derive(&self, api_key: &str) -> Result<String, String> {
        self.derive_scoped("principal", api_key)
            .map(|digest| format!("p_{digest}"))
    }

    pub fn fingerprint(&self, scope: &str, value: &[u8]) -> Result<String, String> {
        if scope.is_empty()
            || scope.len() > 64
            || !scope
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err("invalid fingerprint scope".to_string());
        }
        self.derive_scoped_bytes(scope, value)
            .map(|digest| format!("hmac-sha256:{digest}"))
    }

    fn derive_scoped(&self, scope: &str, value: &str) -> Result<String, String> {
        self.derive_scoped_bytes(scope, value.as_bytes())
    }

    fn derive_scoped_bytes(&self, scope: &str, value: &[u8]) -> Result<String, String> {
        if self.secret.get().is_none() {
            let _loading = self
                .load_gate
                .lock()
                .map_err(|_| "runtime namespace key loader is unavailable".to_string())?;
            if self.secret.get().is_none() {
                let secret = load_or_create_secret(&self.path)?;
                let _ = self.secret.set(secret);
            }
        }
        let secret = self
            .secret
            .get()
            .ok_or_else(|| "runtime namespace key is unavailable".to_string())?;
        let mut mac = HmacSha256::new_from_slice(secret)
            .map_err(|_| "could not initialize principal derivation".to_string())?;
        mac.update(b"werk1112-scoped-identity-v1\0");
        mac.update(scope.as_bytes());
        mac.update(b"\0");
        mac.update(value);
        Ok(format!("{:x}", mac.finalize().into_bytes()))
    }
}

fn load_or_create_secret(path: &Path) -> Result<[u8; SECRET_BYTES], String> {
    let parent = path.parent().ok_or("runtime namespace key has no parent")?;
    ensure_safe_parent(parent)?;
    let lock_path = parent.join(".runtime-namespace.lock");
    if fs::symlink_metadata(&lock_path)
        .is_ok_and(|metadata| metadata_is_link_or_reparse(&metadata) || !metadata.is_file())
    {
        return Err("runtime namespace lock is not a regular file".to_string());
    }
    let mut lock_options = OpenOptions::new();
    lock_options.create(true).read(true).write(true);
    #[cfg(unix)]
    lock_options
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    #[cfg(windows)]
    lock_options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    let lock = lock_options
        .open(&lock_path)
        .map_err(|error| format!("could not open runtime namespace lock: {error}"))?;
    let lock_metadata = lock
        .metadata()
        .map_err(|error| format!("could not inspect runtime namespace lock: {error}"))?;
    if metadata_is_link_or_reparse(&lock_metadata) || !lock_metadata.is_file() {
        return Err("runtime namespace lock is not a regular file".to_string());
    }
    #[cfg(unix)]
    {
        if lock_metadata.uid() != unsafe { libc::geteuid() } {
            return Err("runtime namespace lock is not owned by the current user".to_string());
        }
        lock.set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("could not restrict runtime namespace lock: {error}"))?;
    }
    #[cfg(windows)]
    if lock
        .metadata()
        .map_err(|error| format!("could not inspect runtime namespace lock: {error}"))?
        .file_attributes()
        & 0x0000_0400
        != 0
    {
        return Err("runtime namespace lock is a reparse point".to_string());
    }
    FileExt::try_lock_exclusive(&lock)
        .map_err(|error| format!("could not lock runtime namespace key: {error}"))?;
    cleanup_staging_keys(parent)?;

    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata_is_link_or_reparse(&metadata) || !metadata.is_file() {
                return Err("runtime namespace key is not a regular file".to_string());
            }
            return read_secret(path);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("could not inspect runtime namespace key: {error}")),
    }

    let mut secret = [0u8; SECRET_BYTES];
    getrandom::getrandom(&mut secret)
        .map_err(|_| "secure runtime namespace key generation is unavailable".to_string())?;
    let mut suffix = [0u8; 12];
    getrandom::getrandom(&mut suffix)
        .map_err(|_| "secure runtime namespace staging is unavailable".to_string())?;
    let staging = parent.join(format!(
        ".runtime-namespace-{}.tmp",
        suffix
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ));
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    options.mode(0o600);
    let result = (|| {
        let mut file = options
            .open(&staging)
            .map_err(|error| format!("could not create runtime namespace staging key: {error}"))?;
        file.write_all(&secret)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("could not persist runtime namespace key: {error}"))?;
        drop(file);
        commit_staged_secret(&staging, path, parent)?;
        Ok(secret)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&staging);
    }
    result
}

fn read_secret(path: &Path) -> Result<[u8; SECRET_BYTES], String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    #[cfg(windows)]
    options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    let mut file = options
        .open(path)
        .map_err(|error| format!("could not open runtime namespace key: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("could not inspect runtime namespace key: {error}"))?;
    if metadata_is_link_or_reparse(&metadata)
        || !metadata.is_file()
        || metadata.len() != SECRET_BYTES as u64
    {
        return Err("runtime namespace key is not a safe 32-byte regular file".to_string());
    }
    #[cfg(unix)]
    if metadata.uid() != unsafe { libc::geteuid() } || metadata.permissions().mode() & 0o077 != 0 {
        return Err("runtime namespace key is not private to the current owner".to_string());
    }
    let mut bytes = Vec::with_capacity(SECRET_BYTES + 1);
    Read::by_ref(&mut file)
        .take((SECRET_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("could not read runtime namespace key: {error}"))?;
    bytes.try_into().map_err(|_| {
        "runtime namespace key has an invalid length; refusing to replace it".to_string()
    })
}

fn cleanup_staging_keys(parent: &Path) -> Result<(), String> {
    let entries = fs::read_dir(parent)
        .map_err(|error| format!("could not inspect runtime auth directory: {error}"))?;
    let mut inspected = 0usize;
    let mut removed = 0usize;
    for entry in entries {
        inspected = inspected.saturating_add(1);
        if inspected > MAX_AUTH_DIRECTORY_ENTRIES {
            return Err("runtime auth directory contains too many entries".to_string());
        }
        let entry =
            entry.map_err(|error| format!("could not inspect runtime auth entry: {error}"))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with(".runtime-namespace-") || !name.ends_with(".tmp") {
            continue;
        }
        removed = removed.saturating_add(1);
        if removed > MAX_STAGING_KEYS_PER_CLEANUP {
            return Err("runtime auth directory contains too many staging keys".to_string());
        }
        let metadata = fs::symlink_metadata(entry.path())
            .map_err(|error| format!("could not inspect runtime auth staging key: {error}"))?;
        if metadata.is_dir() && !metadata_is_link_or_reparse(&metadata) {
            return Err("runtime auth staging entry is unexpectedly a directory".to_string());
        }
        fs::remove_file(entry.path())
            .map_err(|error| format!("could not remove runtime auth staging key: {error}"))?;
    }
    Ok(())
}

fn ensure_safe_parent(path: &Path) -> Result<(), String> {
    let created = match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata_is_link_or_reparse(&metadata) || !metadata.is_dir() {
                return Err("runtime auth directory is not a safe directory".to_string());
            }
            false
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(path)
                .map_err(|error| format!("could not create runtime auth directory: {error}"))?;
            true
        }
        Err(error) => {
            return Err(format!("could not inspect runtime auth directory: {error}"));
        }
    };
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("could not inspect runtime auth directory: {error}"))?;
    if metadata_is_link_or_reparse(&metadata) || !metadata.is_dir() {
        return Err("runtime auth directory is not a safe directory".to_string());
    }
    #[cfg(unix)]
    if metadata.uid() != unsafe { libc::geteuid() } {
        return Err("runtime auth directory is not owned by the current user".to_string());
    }
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|error| format!("could not restrict runtime auth directory: {error}"))?;
    if created {
        if let Some(parent) = path.parent() {
            sync_directory(parent)?;
        }
    }
    Ok(())
}

fn commit_staged_secret(staging: &Path, target: &Path, parent: &Path) -> Result<(), String> {
    // A hard-link publish is an atomic no-replace operation on the same
    // filesystem. An uncooperative process racing us can never have its key
    // silently overwritten.
    fs::hard_link(staging, target)
        .map_err(|error| format!("could not commit runtime namespace key: {error}"))?;
    sync_directory(parent)?;
    fs::remove_file(staging)
        .map_err(|error| format!("could not remove runtime namespace staging key: {error}"))?;
    sync_directory(parent)?;
    Ok(())
}

fn metadata_is_link_or_reparse(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        return metadata.file_attributes() & 0x0000_0400 != 0;
    }
    #[cfg(not(windows))]
    false
}

fn sync_directory(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        fs::File::open(path)
            .and_then(|file| file.sync_all())
            .map_err(|error| format!("could not sync runtime auth directory: {error}"))?;
    }
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Arc, Barrier},
        thread,
        time::{SystemTime, UNIX_EPOCH},
    };

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "werk-principal-{label}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn principal_ids_are_stable_partitioned_and_do_not_contain_keys() {
        let root = TestDir::new("stable");
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();
        let first = PrincipalDeriver::new(&store);
        let alice = first.derive("alice-secret").unwrap();
        assert_eq!(alice, first.derive("alice-secret").unwrap());
        assert_ne!(alice, first.derive("bob-secret").unwrap());
        assert!(!alice.contains("alice-secret"));
        assert_eq!(
            alice,
            PrincipalDeriver::new(&store)
                .derive("alice-secret")
                .unwrap()
        );
    }

    #[test]
    fn concurrent_first_use_commits_one_stable_secret() {
        let root = TestDir::new("concurrent");
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();
        let deriver = Arc::new(PrincipalDeriver::new(&store));
        let barrier = Arc::new(Barrier::new(8));
        let threads = (0..8)
            .map(|_| {
                let deriver = deriver.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    deriver.derive("same-credential").unwrap()
                })
            })
            .collect::<Vec<_>>();
        let identities = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert!(identities.iter().all(|identity| identity == &identities[0]));
        assert_eq!(
            fs::metadata(root.0.join("auth/runtime-namespace.key"))
                .unwrap()
                .len(),
            SECRET_BYTES as u64
        );
    }

    #[test]
    fn transient_external_lock_failure_is_not_cached() {
        let root = TestDir::new("retry-lock");
        let auth = root.0.join("auth");
        fs::create_dir(&auth).unwrap();
        let lock = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(auth.join(".runtime-namespace.lock"))
            .unwrap();
        FileExt::lock_exclusive(&lock).unwrap();
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();
        let deriver = PrincipalDeriver::new(&store);

        assert!(deriver.derive("credential").is_err());
        FileExt::unlock(&lock).unwrap();
        assert!(deriver.derive("credential").is_ok());
    }

    #[test]
    fn invalid_existing_key_is_never_silently_replaced() {
        let root = TestDir::new("invalid");
        let auth = root.0.join("auth");
        fs::create_dir(&auth).unwrap();
        let key = auth.join("runtime-namespace.key");
        fs::write(&key, b"short").unwrap();
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();

        assert!(PrincipalDeriver::new(&store).derive("credential").is_err());
        assert_eq!(fs::read(key).unwrap(), b"short");
    }

    #[test]
    fn oversized_existing_key_is_rejected_with_a_bounded_read() {
        let root = TestDir::new("oversized");
        let auth = root.0.join("auth");
        fs::create_dir(&auth).unwrap();
        let key = auth.join("runtime-namespace.key");
        fs::write(&key, [5_u8; SECRET_BYTES + 1]).unwrap();
        #[cfg(unix)]
        fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).unwrap();
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();

        assert!(PrincipalDeriver::new(&store).derive("credential").is_err());
        assert_eq!(fs::metadata(key).unwrap().len(), (SECRET_BYTES + 1) as u64);
    }

    #[test]
    fn staged_secret_publish_never_replaces_an_existing_key() {
        let root = TestDir::new("no-replace");
        let staging = root.0.join("staging");
        let target = root.0.join("target");
        fs::write(&staging, [1_u8; SECRET_BYTES]).unwrap();
        fs::write(&target, [2_u8; SECRET_BYTES]).unwrap();

        assert!(commit_staged_secret(&staging, &target, &root.0).is_err());
        assert_eq!(fs::read(&target).unwrap(), [2_u8; SECRET_BYTES]);
        assert_eq!(fs::read(&staging).unwrap(), [1_u8; SECRET_BYTES]);
    }

    #[test]
    fn auth_directory_scanning_is_bounded() {
        let root = TestDir::new("bounded-scan");
        for index in 0..=MAX_AUTH_DIRECTORY_ENTRIES {
            fs::write(root.0.join(format!("entry-{index}")), b"x").unwrap();
        }

        assert!(cleanup_staging_keys(&root.0).is_err());
    }

    #[test]
    fn abandoned_staging_cleanup_work_is_bounded() {
        let root = TestDir::new("bounded-cleanup");
        for index in 0..=MAX_STAGING_KEYS_PER_CLEANUP {
            fs::write(
                root.0.join(format!(".runtime-namespace-{index}.tmp")),
                b"partial",
            )
            .unwrap();
        }

        assert!(cleanup_staging_keys(&root.0).is_err());
        assert!(
            fs::read_dir(&root.0)
                .unwrap()
                .filter_map(Result::ok)
                .any(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
        );
    }

    #[test]
    fn abandoned_staging_key_is_removed_before_commit() {
        let root = TestDir::new("staging");
        let auth = root.0.join("auth");
        fs::create_dir(&auth).unwrap();
        let staging = auth.join(".runtime-namespace-abandoned.tmp");
        fs::write(&staging, b"partial").unwrap();
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();

        PrincipalDeriver::new(&store).derive("credential").unwrap();
        assert!(!staging.exists());
        assert!(auth.join("runtime-namespace.key").is_file());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_key_is_rejected_without_touching_target() {
        use std::os::unix::fs::symlink;

        let root = TestDir::new("symlink");
        let outside = TestDir::new("outside");
        let target = outside.0.join("target");
        let original = [7_u8; SECRET_BYTES];
        fs::write(&target, original).unwrap();
        let auth = root.0.join("auth");
        fs::create_dir(&auth).unwrap();
        symlink(&target, auth.join("runtime-namespace.key")).unwrap();
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();

        assert!(PrincipalDeriver::new(&store).derive("credential").is_err());
        assert_eq!(fs::read(target).unwrap(), original);
    }

    #[cfg(unix)]
    #[test]
    fn secret_and_auth_directory_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let root = TestDir::new("permissions");
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();
        PrincipalDeriver::new(&store).derive("credential").unwrap();
        assert_eq!(
            fs::metadata(root.0.join("auth"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(root.0.join("auth/runtime-namespace.key"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }

    #[cfg(unix)]
    #[test]
    fn existing_owned_auth_directory_is_restricted() {
        use std::os::unix::fs::PermissionsExt;

        let root = TestDir::new("existing-auth-permissions");
        let auth = root.0.join("auth");
        fs::create_dir(&auth).unwrap();
        fs::set_permissions(&auth, fs::Permissions::from_mode(0o755)).unwrap();
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();

        PrincipalDeriver::new(&store).derive("credential").unwrap();

        assert_eq!(
            fs::metadata(auth).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }

    #[cfg(unix)]
    #[test]
    fn permissive_existing_secret_is_rejected_without_rewriting_it() {
        use std::os::unix::fs::PermissionsExt;

        let root = TestDir::new("permissive-secret");
        let auth = root.0.join("auth");
        fs::create_dir(&auth).unwrap();
        let key = auth.join("runtime-namespace.key");
        let original = [9_u8; SECRET_BYTES];
        fs::write(&key, original).unwrap();
        fs::set_permissions(&key, fs::Permissions::from_mode(0o644)).unwrap();
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();

        assert!(PrincipalDeriver::new(&store).derive("credential").is_err());
        assert_eq!(fs::read(&key).unwrap(), original);
        assert_eq!(
            fs::metadata(key).unwrap().permissions().mode() & 0o777,
            0o644
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_auth_directory_is_rejected_without_touching_target() {
        use std::os::unix::fs::symlink;

        let root = TestDir::new("symlinked-auth");
        let outside = TestDir::new("outside-auth");
        symlink(&outside.0, root.0.join("auth")).unwrap();
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();

        assert!(PrincipalDeriver::new(&store).derive("credential").is_err());
        assert!(fs::read_dir(&outside.0).unwrap().next().is_none());
    }

    #[cfg(windows)]
    #[test]
    fn reparse_auth_directory_is_rejected_without_touching_target() {
        use std::{io::ErrorKind, os::windows::fs::symlink_dir};

        let root = TestDir::new("reparse-auth");
        let outside = TestDir::new("outside-auth");
        if let Err(error) = symlink_dir(&outside.0, root.0.join("auth")) {
            if error.kind() == ErrorKind::PermissionDenied {
                return;
            }
            panic!("could not create test directory reparse point: {error}");
        }
        let store = ModelStore::resolve(Some(root.0.clone())).unwrap();

        assert!(PrincipalDeriver::new(&store).derive("credential").is_err());
        assert!(fs::read_dir(&outside.0).unwrap().next().is_none());
    }
}
