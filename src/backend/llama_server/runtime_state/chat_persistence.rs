//! Durable terminal-chat snapshots. Named runtime-control states remain process-bound.

use super::*;
use serde::{Deserialize, Serialize};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

pub(crate) struct LlamaChatPersistence {
    directory: PathBuf,
    restore_attempted: AtomicBool,
    pub(crate) probe_seconds: f64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u32,
    filename: String,
    tokens: u64,
    bytes: u64,
    sha256: String,
}

impl LlamaChatPersistence {
    pub(crate) fn open(
        server: &LlamaServerProcess,
        store: &ModelStore,
        manifest: &ModelManifest,
        root: &Path,
    ) -> Result<Self> {
        if !server.state_runtime.configured {
            bail!("llama.cpp does not expose verified private single-slot snapshot storage");
        }
        let identity = server
            .state_runtime
            .identity
            .as_ref()
            .context("missing runtime identity")?;
        let mut files = Vec::new();
        for file in &manifest.files {
            let path = store.absolute_model_file(manifest, &file.path);
            let metadata = fs::metadata(&path)?;
            files.push(json!([
                path,
                metadata.len(),
                metadata
                    .modified()?
                    .duration_since(UNIX_EPOCH)?
                    .as_nanos()
                    .to_string()
            ]));
        }
        files.sort_by_key(Value::to_string);
        let mut runtime_environment = env::vars()
            .filter(|(key, _)| {
                key.starts_with("LLAMA_ARG_")
                    || key.starts_with("GGML_")
                    || matches!(
                        key.as_str(),
                        "CUDA_VISIBLE_DEVICES" | "LD_LIBRARY_PATH" | "LD_PRELOAD"
                    )
            })
            .collect::<Vec<_>>();
        runtime_environment.sort();
        let key = sha256_json_value(&json!({
            "format": "werk-llama-chat-v1",
            "model": server.model_identity.to_string(),
            "files": files,
            "runtime": identity.executable.binary_sha256,
            "version": identity.executable.version,
            "libraries": runtime_libraries(server)?,
            "environment": runtime_environment,
            "mode": label(server.mode),
            "args": persistent_args(&server.args),
        }))?;
        ensure_real_directory(root, true)?;
        let directory = root.join(format!("llama-{}", key.trim_start_matches("sha256:")));
        ensure_real_directory(&directory, true)?;
        let _gate = server
            .state_gate
            .lock()
            .map_err(|_| anyhow!("llama.cpp state gate unavailable"))?;
        let probe_started = Instant::now();
        functional_probe_llama_chat_state(server)?;
        Ok(Self {
            directory,
            restore_attempted: AtomicBool::new(false),
            probe_seconds: probe_started.elapsed().as_secs_f64(),
        })
    }

    // Called while holding the process state gate, including the following generation.
    pub(crate) fn restore(&self, server: &LlamaServerProcess) -> Result<String> {
        if self.restore_attempted.swap(true, Ordering::Relaxed) {
            return Ok(
                "llama.cpp native KV persistence: using live slot; reuse reported by backend usage"
                    .into(),
            );
        }
        Ok(match self.restore_inner(server) {
            Ok(Some(tokens)) => format!(
                "llama.cpp native KV snapshot restored: {tokens} tokens; actual prefix reuse reported by backend usage"
            ),
            Ok(None) => "llama.cpp native KV persistence: cold session".into(),
            Err(error) => {
                if let Err(clear_error) = erase_llama_slot(server, None) {
                    self.restore_attempted.store(false, Ordering::Relaxed);
                    return Err(clear_error)
                        .context("cannot clear slot after failed native cache restore");
                }
                format!(
                    "llama.cpp native KV snapshot not reused: {error}; rebuilding from conversation"
                )
            }
        })
    }

    fn read_record(&self) -> Result<Option<Record>> {
        let path = self.directory.join("latest.json");
        if !path.try_exists()? {
            return Ok(None);
        }
        let (file, bytes) = open_bounded_regular_file(&path, None)?;
        if bytes > 4096 {
            bail!("native cache index exceeds its size limit");
        }
        let record: Record = serde_json::from_reader(file.take(4097))?;
        validate_record(&record)?;
        Ok(Some(record))
    }

    fn restore_inner(&self, server: &LlamaServerProcess) -> Result<Option<u64>> {
        let Some(record) = self.read_record()? else {
            return Ok(None);
        };
        let name = format!("{}.bin", random_private_id("chat_restore_", 16)?);
        let target = private_snapshot_path(server, &name)?;
        let result = (|| {
            let sha256 = copy_snapshot_with_sha256(
                &self.directory.join(&record.filename),
                &target,
                record.bytes,
            )?;
            if sha256 != record.sha256 {
                bail!("native cache checksum mismatch");
            }
            restore_llama_slot(server, &name, record.tokens, record.bytes)?;
            Ok(Some(record.tokens))
        })();
        remove_private_snapshot(server, &name);
        result
    }

    pub(crate) fn save(&self, server: &LlamaServerProcess) -> String {
        match self.save_inner(server) {
            Ok(tokens) => format!("llama.cpp native KV snapshot saved: {tokens} tokens"),
            Err(error) => format!("llama.cpp native KV snapshot not saved: {error}"),
        }
    }

    fn save_inner(&self, server: &LlamaServerProcess) -> Result<u64> {
        let status = llama_slot_status(server)?;
        if status.is_processing || status.prompt_tokens == 0 {
            bail!("native cache slot is busy or empty");
        }
        let name = format!("{}.bin", random_private_id("chat_", 16)?);
        let result = (|| {
            let info = save_llama_slot(server, &name, status.prompt_tokens)?;
            let source = private_snapshot_path(server, &name)?;
            let target = self.directory.join(&name);
            let sha256 = copy_snapshot_with_sha256(&source, &target, info.bytes)?;
            let record = Record {
                version: 1,
                filename: name.clone(),
                tokens: status.prompt_tokens,
                bytes: info.bytes,
                sha256,
            };
            let previous = self.read_record().ok().flatten();
            if let Err(error) = self.publish(&record) {
                // A directory fsync can fail after the new index was renamed.
                // Keep the bytes if that index may already reference them.
                if self
                    .read_record()
                    .is_ok_and(|current| current.is_none_or(|v| v.filename != name))
                {
                    let _ = fs::remove_file(target);
                }
                return Err(error);
            }
            if let Some(previous) = previous {
                let _ = fs::remove_file(self.directory.join(previous.filename));
            }
            Ok(status.prompt_tokens)
        })();
        remove_private_snapshot(server, &name);
        result
    }

    fn publish(&self, record: &Record) -> Result<()> {
        let temporary = self.directory.join(random_private_id("index_", 16)?);
        let result = (|| {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            options.mode(0o600);
            let mut file = options.open(&temporary)?;
            serde_json::to_writer(&mut file, record)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            fs::rename(&temporary, self.directory.join("latest.json"))?;
            #[cfg(unix)]
            fs::File::open(&self.directory)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(temporary);
        }
        result
    }
}

fn runtime_libraries(server: &LlamaServerProcess) -> Result<Vec<(PathBuf, String)>> {
    let mut paths = std::collections::BTreeSet::new();
    let is_runtime_library = |path: &Path| {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        ["libllama", "libggml", "libmtmd", "llama", "ggml"]
            .iter()
            .any(|prefix| name.starts_with(prefix))
            && (name.contains(".so") || name.ends_with(".dylib") || name.ends_with(".dll"))
    };
    #[cfg(target_os = "linux")]
    for line in fs::read_to_string(format!("/proc/{}/maps", server.pid))?.lines() {
        if let Some(start) = line.find('/') {
            let path = PathBuf::from(&line[start..]);
            if is_runtime_library(&path) {
                paths.insert(fs::canonicalize(path)?);
            }
        }
    }
    if let Some(parent) = server.executable.parent() {
        for entry in fs::read_dir(parent)? {
            let path = entry?.path();
            if is_runtime_library(&path) {
                paths.insert(fs::canonicalize(path)?);
            }
        }
    }
    paths
        .into_iter()
        .map(|path| {
            let hash = sha256_regular_file(&path, 4 * 1024 * 1024 * 1024)?;
            Ok((path, hash))
        })
        .collect()
}

fn validate_record(record: &Record) -> Result<()> {
    let id = record
        .filename
        .strip_prefix("chat_")
        .and_then(|v| v.strip_suffix(".bin"));
    if record.version != 1
        || record.tokens == 0
        || record.bytes == 0
        || record.bytes > STATE_SNAPSHOT_MAX_BYTES
        || !id.is_some_and(|id| id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()))
        || !record
            .sha256
            .strip_prefix("sha256:")
            .is_some_and(|hash| hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        bail!("invalid native cache index");
    }
    Ok(())
}

fn persistent_args(args: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if matches!(arg.as_str(), "--port" | "--slot-save-path") {
            let _ = args.next();
        } else {
            result.push(arg.clone());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_identity_ignores_only_ephemeral_arguments() {
        let args = [
            "--port",
            "1234",
            "--slot-save-path",
            "/tmp/random",
            "-c",
            "4096",
            "--reasoning",
            "off",
        ]
        .map(str::to_string);
        assert_eq!(persistent_args(&args), ["-c", "4096", "--reasoning", "off"]);
    }

    #[test]
    fn cache_record_rejects_traversal_and_unbounded_files() {
        let mut record = Record {
            version: 1,
            filename: format!("chat_{}.bin", "a".repeat(32)),
            tokens: 10,
            bytes: 100,
            sha256: format!("sha256:{}", "b".repeat(64)),
        };
        validate_record(&record).unwrap();
        record.filename = "../state.bin".into();
        assert!(validate_record(&record).is_err());
        record.filename = format!("chat_{}.bin", "a".repeat(32));
        record.bytes = STATE_SNAPSHOT_MAX_BYTES + 1;
        assert!(validate_record(&record).is_err());
    }
}
