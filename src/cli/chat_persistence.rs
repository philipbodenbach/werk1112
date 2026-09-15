//! Portable chat history. This does not serialize model weights or native KV caches.
use anyhow::{Context, Result, bail};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
#[cfg(windows)]
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    openai::{ChatMessage, MessageContent},
    werk_protocol::{PersistenceMode, PersistencePolicy, ReuseMode},
};

const SCHEMA_VERSION: u32 = 1;
const MAX_ARCHIVE_BYTES: usize = 16 * 1024 * 1024;
const MAX_MESSAGES: usize = 4096;
const MAX_TTL_SECONDS: u64 = 30 * 24 * 60 * 60;

pub(super) struct ChatPersistence {
    disk: Option<DiskSession>,
    policy: PersistencePolicy,
    resumed: bool,
    notice: Option<String>,
}

struct DiskSession {
    root: PathBuf,
    path: PathBuf,
    fingerprint: String,
    // Never unlink lock files: another process could otherwise hold a different
    // inode for the same logical session. Closing this handle releases the lock.
    _lock: File,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatArchive<T> {
    schema_version: u32,
    session_fingerprint: String,
    updated_unix_seconds: u64,
    expires_unix_seconds: Option<u64>,
    messages: T,
}

impl ChatPersistence {
    pub(super) fn open(
        home: &Path,
        session: &str,
        model_id: &str,
        policy: PersistencePolicy,
    ) -> Result<(Self, Vec<ChatMessage>)> {
        Self::open_at(home, session, model_id, policy, unix_seconds()?)
    }

    fn open_at(
        home: &Path,
        session: &str,
        model_id: &str,
        policy: PersistencePolicy,
        now: u64,
    ) -> Result<(Self, Vec<ChatMessage>)> {
        validate_label(session, "session", 256)?;
        validate_label(model_id, "model", 1024)?;
        if policy.pin {
            bail!("chat history does not support persistence pinning");
        }
        if policy
            .ttl_seconds
            .is_some_and(|ttl| ttl == 0 || ttl > MAX_TTL_SECONDS)
        {
            bail!("chat persistence TTL must be from 1 to 2592000 seconds");
        }
        let durable = matches!(policy.mode, PersistenceMode::Disk | PersistenceMode::Auto);
        let mut persistence = Self {
            disk: None,
            policy,
            resumed: false,
            notice: None,
        };
        if !durable {
            if persistence.policy.reuse == ReuseMode::Required {
                bail!("required chat history reuse needs disk or auto persistence mode");
            }
            return Ok((persistence, Vec::new()));
        }

        // Length-delimited JSON prevents namespace collisions. Only portable
        // message history lives here; backend-specific cache identities do not.
        let fingerprint = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(model_id, session))?)
        );
        let root = storage_directory(home)?;
        let lock_path = root.join(format!("{fingerprint}.lock"));
        let lock = private_options(true, false)
            .open(&lock_path)
            .context("cannot open chat session lock")?;
        validate_regular_private(&lock.metadata()?)?;
        FileExt::try_lock_exclusive(&lock)
            .context("chat session is already open in another process; choose another --session")?;
        let path = root.join(format!("{fingerprint}.json"));
        // Reject links and unsafe files even with reuse disabled, before there
        // is any opportunity to replace data outside this session namespace.
        let exists = validate_existing_archive(&path)?;
        persistence.disk = Some(DiskSession {
            root,
            path,
            fingerprint,
            _lock: lock,
        });
        if persistence.policy.reuse == ReuseMode::Disabled {
            persistence.notice = Some("starting a new chat history with reuse disabled".into());
            return Ok((persistence, Vec::new()));
        }
        if !exists {
            if persistence.policy.reuse == ReuseMode::Required {
                bail!("required chat history was not found for this model and session");
            }
            return Ok((persistence, Vec::new()));
        }

        let disk = persistence.disk.as_ref().expect("disk session initialized");
        let archive = read_archive(&disk.path, &disk.fingerprint).context(
            "saved chat history is invalid; choose another --session or explicitly start over with --persistence-reuse disabled",
        )?;
        if archive
            .expires_unix_seconds
            .is_some_and(|expiry| expiry <= now)
        {
            if persistence.policy.reuse == ReuseMode::Required {
                bail!("required chat history has expired");
            }
            persistence.notice =
                Some("saved chat history has expired; starting a new conversation".into());
            return Ok((persistence, Vec::new()));
        }
        persistence.resumed = true;
        Ok((persistence, archive.messages))
    }

    /// The caller must pass the full archive only after successful generation.
    /// Request-local context trimming must not discard older archived messages.
    pub(super) fn save_completed_turn(&mut self, messages: &[ChatMessage]) -> Result<()> {
        self.save_at(messages, unix_seconds()?)
    }

    fn save_at(&mut self, messages: &[ChatMessage], now: u64) -> Result<()> {
        validate_messages(messages)?;
        let Some(disk) = &self.disk else {
            return Ok(());
        };
        let archive = ChatArchive {
            schema_version: SCHEMA_VERSION,
            session_fingerprint: disk.fingerprint.clone(),
            updated_unix_seconds: now,
            expires_unix_seconds: self
                .policy
                .ttl_seconds
                .map(|ttl| {
                    now.checked_add(ttl)
                        .context("chat history expiry exceeds clock range")
                })
                .transpose()?,
            messages,
        };
        let mut bytes = BoundedBytes(Vec::new());
        serde_json::to_writer(&mut bytes, &archive)
            .context("chat history exceeds its 16 MiB archive limit or cannot be encoded")?;
        validate_directory(&disk.root)?;
        validate_existing_archive(&disk.path)?;
        let mut suffix = [0u8; 12];
        getrandom::getrandom(&mut suffix)
            .map_err(|_| anyhow::anyhow!("cannot create a private chat staging name"))?;
        let suffix: String = suffix.iter().map(|byte| format!("{byte:02x}")).collect();
        let staging = disk
            .root
            .join(format!(".{}-{suffix}.tmp", disk.fingerprint));
        let result = (|| {
            let mut file = private_options(false, true)
                .open(&staging)
                .context("cannot create private chat history staging file")?;
            file.write_all(&bytes.0)
                .context("cannot write chat history")?;
            file.sync_all().context("cannot flush chat history")?;
            drop(file);
            fs::rename(&staging, &disk.path).context("cannot atomically replace chat history")?;
            sync_directory(&disk.root)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&staging);
        }
        result
    }

    pub(super) fn is_durable(&self) -> bool {
        self.disk.is_some()
    }

    pub(super) fn resumed(&self) -> bool {
        self.resumed
    }

    pub(super) fn notice(&self) -> Option<&str> {
        self.notice.as_deref()
    }

    pub(super) fn path(&self) -> Option<&Path> {
        self.disk.as_ref().map(|disk| disk.path.as_path())
    }

    pub(super) fn native_cache_directory(&self) -> Result<PathBuf> {
        let disk = self
            .disk
            .as_ref()
            .context("native disk cache requires durable chat history")?;
        validate_directory(&disk.root)?;
        let path = disk.path.with_extension("cache");
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        builder.mode(0o700);
        match builder.create(&path) {
            Ok(()) => sync_directory(&disk.root)?,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).context("cannot create private native chat cache directory");
            }
        }
        validate_directory(&path)?;
        Ok(path)
    }
}

fn validate_label(value: &str, label: &str, max: usize) -> Result<()> {
    if value.trim().is_empty() || value.len() > max || value.chars().any(char::is_control) {
        bail!(
            "chat {label} must be nonempty, at most {max} bytes and contain no control characters"
        );
    }
    Ok(())
}

fn validate_messages(messages: &[ChatMessage]) -> Result<()> {
    if messages.is_empty() || messages.len() > MAX_MESSAGES {
        bail!("saved chat history must contain from 1 to 4096 messages");
    }
    if messages.iter().any(|message| {
        !matches!(
            message.role.as_str(),
            "system" | "developer" | "user" | "assistant" | "tool"
        )
    }) {
        bail!("saved chat history contains an unsupported message role");
    }
    let last = messages.last().expect("nonempty checked above");
    let has_content = match &last.content {
        Some(MessageContent::Text(text)) => !text.trim().is_empty(),
        Some(MessageContent::Parts(parts)) => parts.iter().any(|part| {
            part.text
                .as_ref()
                .is_some_and(|text| !text.trim().is_empty())
                || part.image_url.is_some()
        }),
        None => false,
    };
    if last.role != "assistant"
        || (!has_content
            && !last
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty()))
    {
        bail!("chat history must end with a completed assistant turn");
    }
    Ok(())
}

fn read_archive(path: &Path, fingerprint: &str) -> Result<ChatArchive<Vec<ChatMessage>>> {
    let mut file = private_read_options()
        .open(path)
        .context("cannot read chat history")?;
    let metadata = file.metadata()?;
    validate_regular_private(&metadata)?;
    if metadata.len() > MAX_ARCHIVE_BYTES as u64 {
        bail!("saved chat history exceeds 16 MiB");
    }
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take((MAX_ARCHIVE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_ARCHIVE_BYTES {
        bail!("saved chat history exceeds 16 MiB");
    }
    let archive: ChatArchive<Vec<ChatMessage>> =
        serde_json::from_slice(&bytes).context("saved chat history is corrupt")?;
    if archive.schema_version != SCHEMA_VERSION || archive.session_fingerprint != fingerprint {
        bail!("saved chat history has an incompatible schema or session identity");
    }
    if archive
        .expires_unix_seconds
        .is_some_and(|expiry| expiry <= archive.updated_unix_seconds)
    {
        bail!("saved chat history has invalid expiry metadata");
    }
    validate_messages(&archive.messages)?;
    Ok(archive)
}

fn storage_directory(home: &Path) -> Result<PathBuf> {
    fs::create_dir_all(home).context("cannot create Werk home for chat history")?;
    let root = home.canonicalize()?.join("chat-sessions");
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    builder.mode(0o700);
    match builder.create(&root) {
        Ok(()) => sync_directory(root.parent().expect("chat directory has parent"))?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error).context("cannot create private chat history directory"),
    }
    validate_directory(&root)?;
    Ok(root)
}

fn validate_directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if is_link_or_reparse(&metadata) || !metadata.is_dir() {
        bail!("chat history directory must be a regular directory, not a link");
    }
    validate_owner_and_mode(&metadata)
}

fn validate_existing_archive(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            validate_regular_private(&metadata)?;
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).context("cannot inspect saved chat history"),
    }
}

fn validate_regular_private(metadata: &fs::Metadata) -> Result<()> {
    if is_link_or_reparse(metadata) || !metadata.is_file() {
        bail!("chat history and lock files must be regular files, not links");
    }
    #[cfg(unix)]
    if metadata.nlink() != 1 {
        bail!("chat history and lock files must not have hard links");
    }
    validate_owner_and_mode(metadata)
}

fn validate_owner_and_mode(metadata: &fs::Metadata) -> Result<()> {
    #[cfg(unix)]
    if metadata.uid() != unsafe { libc::geteuid() } || metadata.permissions().mode() & 0o077 != 0 {
        bail!("chat history must be private to the current user (directories 0700, files 0600)");
    }
    let _ = metadata;
    Ok(())
}

fn is_link_or_reparse(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    return metadata.file_attributes() & 0x0000_0400 != 0;
    #[cfg(not(windows))]
    false
}

fn private_read_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK);
    #[cfg(windows)]
    options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    options
}

fn private_options(create: bool, create_new: bool) -> OpenOptions {
    let mut options = private_read_options();
    options.write(true).create(create).create_new(create_new);
    #[cfg(unix)]
    options.mode(0o600);
    options
}

fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(path)?
        .sync_all()
        .context("cannot flush chat history directory")?;
    let _ = path;
    Ok(())
}

fn unix_seconds() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs())
}

struct BoundedBytes(Vec<u8>);

impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_ARCHIVE_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other("chat history exceeds 16 MiB"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::MessageContent;
    use serde_json::json;

    struct TestDir(PathBuf);
    impl TestDir {
        fn new() -> Self {
            let mut suffix = [0u8; 8];
            getrandom::getrandom(&mut suffix).unwrap();
            Self(std::env::temp_dir().join(format!(
                "werk-chat-history-{:x}",
                u64::from_le_bytes(suffix)
            )))
        }
    }
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn messages() -> Vec<ChatMessage> {
        serde_json::from_value(json!([
            {"role": "system", "content": "Remember this conversation."},
            {"role": "user", "name": "owner", "content": [
                {"type": "text", "text": "Hi"},
                {"type": "image_url", "image_url": {"url": "file:///picture.png", "detail": "low"}}
            ]},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call1", "content": "Found"},
            {"role": "assistant", "content": "Hello"}
        ])).unwrap()
    }

    fn open(
        root: &TestDir,
        policy: PersistencePolicy,
        now: u64,
    ) -> Result<(ChatPersistence, Vec<ChatMessage>)> {
        ChatPersistence::open_at(&root.0, "default", "owner/model", policy, now)
    }

    #[test]
    fn portable_full_history_survives_restarts_and_retains_message_fields() {
        let root = TestDir::new();
        let original = messages();
        let (mut session, empty) = open(&root, PersistencePolicy::default(), 100).unwrap();
        assert!(empty.is_empty());
        assert!(!session.resumed());
        assert!(session.is_durable());
        session.save_at(&original, 101).unwrap();
        let path = session.path().unwrap().to_owned();
        drop(session);
        let (session, loaded) = open(&root, PersistencePolicy::default(), 102).unwrap();
        assert!(session.resumed());
        assert_eq!(
            serde_json::to_value(&loaded).unwrap(),
            serde_json::to_value(original).unwrap()
        );
        assert_eq!(session.path(), Some(path.as_path()));
        #[cfg(unix)]
        {
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(path.parent().unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
    }

    #[test]
    fn native_cache_directory_is_private_stable_and_requires_disk_mode() {
        let root = TestDir::new();
        let (session, _) = open(&root, PersistencePolicy::default(), 100).unwrap();
        let path = session.native_cache_directory().unwrap();
        assert_eq!(path, session.path().unwrap().with_extension("cache"));
        assert_eq!(path, session.native_cache_directory().unwrap());
        #[cfg(unix)]
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o700
        );
        let (memory, _) = open(
            &root,
            PersistencePolicy {
                mode: PersistenceMode::Memory,
                ..Default::default()
            },
            100,
        )
        .unwrap();
        assert!(memory.native_cache_directory().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn native_cache_directory_rejects_symlinks() {
        let root = TestDir::new();
        let (session, _) = open(&root, PersistencePolicy::default(), 100).unwrap();
        let target = root.0.join("outside-cache");
        fs::create_dir(&target).unwrap();
        std::os::unix::fs::symlink(&target, session.path().unwrap().with_extension("cache"))
            .unwrap();
        assert!(session.native_cache_directory().is_err());
        assert_eq!(fs::read_dir(target).unwrap().count(), 0);
    }

    #[test]
    fn sessions_and_models_are_isolated_without_path_interpolation() {
        let root = TestDir::new();
        let (mut session, _) = open(&root, PersistencePolicy::default(), 100).unwrap();
        session.save_at(&messages(), 101).unwrap();
        for (name, model) in [("../default", "owner/model"), ("default", "other/model")] {
            let (other, history) =
                ChatPersistence::open_at(&root.0, name, model, PersistencePolicy::default(), 102)
                    .unwrap();
            assert!(history.is_empty());
            assert_ne!(other.path(), session.path());
            assert_eq!(
                other.path().unwrap().parent(),
                session.path().unwrap().parent()
            );
        }
    }

    #[test]
    fn active_session_lock_blocks_concurrent_writers_and_releases_on_drop() {
        let root = TestDir::new();
        let (session, _) = open(&root, PersistencePolicy::default(), 100).unwrap();
        let error = open(&root, PersistencePolicy::default(), 100)
            .err()
            .unwrap();
        assert!(error.to_string().contains("already open"));
        drop(session);
        assert!(open(&root, PersistencePolicy::default(), 100).is_ok());
    }

    #[test]
    fn required_missing_or_expired_history_fails_and_prefer_expiry_is_explicit() {
        let root = TestDir::new();
        let required = PersistencePolicy {
            reuse: ReuseMode::Required,
            ..Default::default()
        };
        assert!(
            open(&root, required.clone(), 100)
                .err()
                .unwrap()
                .to_string()
                .contains("not found")
        );
        let policy = PersistencePolicy {
            ttl_seconds: Some(5),
            ..Default::default()
        };
        let (mut session, _) = open(&root, policy, 100).unwrap();
        session.save_at(&messages(), 100).unwrap();
        drop(session);
        let (session, loaded) = open(&root, required.clone(), 104).unwrap();
        assert!(!loaded.is_empty());
        drop(session);
        assert!(
            open(&root, required, 105)
                .err()
                .unwrap()
                .to_string()
                .contains("expired")
        );
        let (session, loaded) = open(&root, PersistencePolicy::default(), 105).unwrap();
        assert!(loaded.is_empty());
        assert!(session.notice().unwrap().contains("expired"));
    }

    #[test]
    fn disabled_reuse_preserves_existing_file_until_completed_turn_is_saved() {
        let root = TestDir::new();
        let (mut session, _) = open(&root, PersistencePolicy::default(), 100).unwrap();
        session.save_at(&messages(), 100).unwrap();
        let path = session.path().unwrap().to_owned();
        let original = fs::read(&path).unwrap();
        drop(session);
        let policy = PersistencePolicy {
            reuse: ReuseMode::Disabled,
            ..Default::default()
        };
        let (mut session, history) = open(&root, policy, 101).unwrap();
        assert!(history.is_empty());
        assert_eq!(fs::read(&path).unwrap(), original);
        let mut unfinished = messages();
        unfinished.pop();
        assert!(session.save_at(&unfinished, 101).is_err());
        assert_eq!(fs::read(&path).unwrap(), original);
        let mut empty_answer = messages();
        empty_answer.last_mut().unwrap().content = Some(MessageContent::Text(String::new()));
        assert!(session.save_at(&empty_answer, 101).is_err());
        assert_eq!(fs::read(&path).unwrap(), original);
        let replacement = vec![messages().pop().unwrap()];
        session.save_at(&replacement, 102).unwrap();
        drop(session);
        assert_eq!(
            open(&root, PersistencePolicy::default(), 103)
                .unwrap()
                .1
                .len(),
            1
        );
    }

    #[test]
    fn corrupt_archive_is_not_silently_overwritten() {
        let root = TestDir::new();
        let (mut session, _) = open(&root, PersistencePolicy::default(), 100).unwrap();
        session.save_at(&messages(), 100).unwrap();
        let path = session.path().unwrap().to_owned();
        fs::write(&path, b"{broken").unwrap();
        drop(session);
        assert!(
            open(&root, PersistencePolicy::default(), 101)
                .err()
                .unwrap()
                .to_string()
                .contains("invalid")
        );
        assert_eq!(fs::read(&path).unwrap(), b"{broken");
    }

    #[test]
    fn schema_identity_and_size_are_validated_before_reuse() {
        let root = TestDir::new();
        let (mut session, _) = open(&root, PersistencePolicy::default(), 100).unwrap();
        session.save_at(&messages(), 100).unwrap();
        let path = session.path().unwrap().to_owned();
        let original: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        drop(session);
        for (field, bad) in [
            ("schema_version", json!(999)),
            ("session_fingerprint", json!("other")),
        ] {
            let mut malformed = original.clone();
            malformed[field] = bad;
            fs::write(&path, serde_json::to_vec(&malformed).unwrap()).unwrap();
            assert!(open(&root, PersistencePolicy::default(), 101).is_err());
        }
        let file = private_options(false, false).open(&path).unwrap();
        file.set_len((MAX_ARCHIVE_BYTES + 1) as u64).unwrap();
        drop(file);
        assert!(open(&root, PersistencePolicy::default(), 101).is_err());
    }

    #[test]
    fn oversized_save_preserves_the_previous_archive() {
        let root = TestDir::new();
        let (mut session, _) = open(&root, PersistencePolicy::default(), 100).unwrap();
        session.save_at(&messages(), 100).unwrap();
        let original = fs::read(session.path().unwrap()).unwrap();
        let mut oversized = messages();
        oversized.last_mut().unwrap().content =
            Some(MessageContent::Text("x".repeat(MAX_ARCHIVE_BYTES)));
        assert!(session.save_at(&oversized, 101).is_err());
        assert_eq!(fs::read(session.path().unwrap()).unwrap(), original);
        assert!(
            session
                .save_at(&vec![messages().pop().unwrap(); MAX_MESSAGES + 1], 101)
                .is_err()
        );
    }

    #[test]
    fn memory_and_ephemeral_modes_do_not_write_to_disk_and_pinning_is_explicitly_rejected() {
        let root = TestDir::new();
        for mode in [PersistenceMode::Memory, PersistenceMode::Ephemeral] {
            let (mut session, history) = open(
                &root,
                PersistencePolicy {
                    mode,
                    ..Default::default()
                },
                100,
            )
            .unwrap();
            assert!(!session.is_durable());
            assert!(!session.resumed());
            assert!(session.path().is_none());
            assert!(history.is_empty());
            session.save_at(&messages(), 100).unwrap();
            assert!(!root.0.exists());
            assert!(
                open(
                    &root,
                    PersistencePolicy {
                        mode,
                        reuse: ReuseMode::Required,
                        ..Default::default()
                    },
                    100
                )
                .is_err()
            );
        }
        assert!(
            open(
                &root,
                PersistencePolicy {
                    pin: true,
                    ..Default::default()
                },
                100
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn links_and_shared_permissions_cannot_redirect_or_expose_archives() {
        use std::os::unix::fs::symlink;
        let root = TestDir::new();
        let (mut session, _) = open(&root, PersistencePolicy::default(), 100).unwrap();
        session.save_at(&messages(), 100).unwrap();
        let path = session.path().unwrap().to_owned();
        drop(session);
        let outside = root.0.join("outside");
        fs::write(&outside, "untouched").unwrap();
        fs::remove_file(&path).unwrap();
        symlink(&outside, &path).unwrap();
        assert!(open(&root, PersistencePolicy::default(), 101).is_err());
        assert_eq!(fs::read_to_string(&outside).unwrap(), "untouched");
        fs::remove_file(&path).unwrap();
        fs::hard_link(&outside, &path).unwrap();
        assert!(open(&root, PersistencePolicy::default(), 101).is_err());
        fs::remove_file(&path).unwrap();
        let (mut session, _) = open(&root, PersistencePolicy::default(), 102).unwrap();
        session.save_at(&messages(), 102).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        drop(session);
        assert!(open(&root, PersistencePolicy::default(), 103).is_err());
    }
}
