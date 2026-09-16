use super::{MANIFEST_FILE, ModelManifest, ModelStore, sanitize_id, validate_id};
use anyhow::{Context, Result, bail};
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
};

/// One complete model found directly inside a collection directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelImportCandidate {
    pub path: PathBuf,
    pub id: String,
    pub is_werk_model: bool,
}

impl ModelStore {
    /// Discover and preflight a collection without changing either directory.
    /// Child repositories remain intact, including their components and shards.
    pub fn discover_import_collection(&self, source: &Path) -> Result<Vec<ModelImportCandidate>> {
        if !source.is_dir() {
            bail!(
                "--all requires a model collection directory: {}",
                source.display()
            );
        }
        let entries = visible_children(source)?;
        reject_werk_home(source)?;
        if [MANIFEST_FILE, "config.json", "model_index.json"]
            .iter()
            .any(|name| source.join(name).is_file())
            || entries.iter().any(|path| is_weight_index(path))
        {
            bail!(
                "{} appears to be one model repository; import it without --all to keep its files and components together",
                source.display()
            );
        }

        let mut candidates = Vec::new();
        for path in entries {
            let candidate = if path.is_dir() {
                reject_werk_home(&path)?;
                let manifest_path = path.join(MANIFEST_FILE);
                if manifest_path.is_file() {
                    // Parsing metadata directly avoids enrichment, weight reads,
                    // and any dependency on the original store being mounted.
                    let manifest: ModelManifest =
                        serde_json::from_slice(&fs::read(&manifest_path).with_context(|| {
                            format!("cannot read model manifest {}", manifest_path.display())
                        })?)
                        .with_context(|| {
                            format!("invalid model manifest {}", manifest_path.display())
                        })?;
                    Some(ModelImportCandidate {
                        path,
                        id: manifest.id,
                        is_werk_model: true,
                    })
                } else if directory_has_weights(&path)? {
                    let id = path_name(&path, false)?;
                    Some(ModelImportCandidate {
                        path,
                        id,
                        is_werk_model: false,
                    })
                } else {
                    None
                }
            } else if path.is_file() {
                if is_sharded_weight(&path) {
                    bail!(
                        "{} is a model shard; put all shards and their configuration in one child model directory, or import their containing directory without --all",
                        path.display()
                    );
                }
                if is_weight_file(&path) {
                    let id = path_name(&path, true)?;
                    Some(ModelImportCandidate {
                        path,
                        id,
                        is_werk_model: false,
                    })
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(candidate) = candidate {
                candidates.push(candidate);
            }
        }

        if candidates.is_empty() {
            bail!(
                "no model directories or supported model files found directly in {}",
                source.display()
            );
        }
        candidates.sort_by(|left, right| (&left.id, &left.path).cmp(&(&right.id, &right.path)));

        let mut targets = BTreeMap::new();
        let physical_storage_roots = [self.models_dir(), self.shared_artifacts_dir()]
            .into_iter()
            .map(|path| resolve_future_path(&path))
            .collect::<Result<Vec<_>>>()?;
        for candidate in &candidates {
            validate_id(&candidate.id)
                .with_context(|| format!("invalid model name for {}", candidate.path.display()))?;
            let storage_name = sanitize_id(&candidate.id);
            if storage_name.is_empty() || matches!(storage_name.as_str(), "." | "..") {
                bail!("invalid model storage name for '{}'", candidate.id);
            }
            if candidate.path.is_dir() {
                let physical_source = candidate.path.canonicalize().with_context(|| {
                    format!(
                        "cannot resolve model directory {}",
                        candidate.path.display()
                    )
                })?;
                if physical_storage_roots
                    .iter()
                    .any(|root| root.starts_with(&physical_source))
                {
                    bail!(
                        "the destination model home must be outside source model directory {}; choose another --model-home",
                        candidate.path.display()
                    );
                }
            }
            if let Some(previous) = targets.insert(storage_name, candidate) {
                bail!(
                    "model names '{}' ({}) and '{}' ({}) resolve to the same model directory; rename a source model before importing the collection",
                    previous.id,
                    previous.path.display(),
                    candidate.id,
                    candidate.path.display()
                );
            }
            let target = self.model_dir(&candidate.id);
            match fs::symlink_metadata(&target) {
                Ok(_) => bail!(
                    "model '{}' already exists at {}; no models were imported",
                    candidate.id,
                    target.display()
                ),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("cannot inspect model registration {}", target.display())
                    });
                }
            }
        }
        Ok(candidates)
    }
}

fn reject_werk_home(directory: &Path) -> Result<()> {
    let models = directory.join("models");
    if models.is_dir()
        && (directory.join("artifacts").is_dir()
            || directory.join("backends").is_dir()
            || visible_children(&models)?
                .iter()
                .any(|path| path.is_dir() && path.join(MANIFEST_FILE).is_file()))
    {
        bail!(
            "{} appears to be a Werk home; import its model collection with --all using {}",
            directory.display(),
            models.display()
        );
    }
    Ok(())
}

/// Resolve the physical location before the destination store exists.
fn resolve_future_path(path: &Path) -> Result<PathBuf> {
    match path.canonicalize() {
        Ok(path) => return Ok(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("cannot resolve model storage {}", path.display()));
        }
    }
    let name = path
        .file_name()
        .with_context(|| format!("cannot resolve model storage {}", path.display()))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(resolve_future_path(parent)?.join(name))
}

fn visible_children(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut children = Vec::new();
    for entry in fs::read_dir(directory)
        .with_context(|| format!("cannot read model directory {}", directory.display()))?
    {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with('.') {
            children.push(entry.path());
        }
    }
    children.sort();
    Ok(children)
}

fn path_name(path: &Path, file: bool) -> Result<String> {
    let name = if file {
        path.file_stem()
    } else {
        path.file_name()
    };
    name.and_then(OsStr::to_str)
        .map(str::to_owned)
        .with_context(|| format!("model path has no valid UTF-8 name: {}", path.display()))
}

fn is_weight_index(path: &Path) -> bool {
    path.is_file()
        && path
            .file_name()
            .and_then(OsStr::to_str)
            .is_some_and(|name| {
                let name = name.to_ascii_lowercase();
                name.ends_with(".safetensors.index.json") || name.ends_with(".bin.index.json")
            })
}

fn is_weight_file(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "gguf" | "safetensors" | "onnx" | "pt" | "pth" | "ckpt" | "npz" | "mlmodel" | "engine"
        | "plan" => true,
        "bin" => path
            .file_name()
            .and_then(OsStr::to_str)
            .is_some_and(|name| {
                matches!(
                    name.to_ascii_lowercase().as_str(),
                    "pytorch_model.bin" | "diffusion_pytorch_model.bin"
                ) || is_sharded_weight(path)
            }),
        "pb" => path
            .file_name()
            .and_then(OsStr::to_str)
            .is_some_and(|name| {
                matches!(
                    name.to_ascii_lowercase().as_str(),
                    "saved_model.pb" | "frozen_inference_graph.pb"
                )
            }),
        _ => false,
    }
}

fn is_sharded_weight(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(
        extension.as_str(),
        "gguf" | "safetensors" | "bin" | "pt" | "pth"
    ) {
        return false;
    }
    let stem = path
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let is_number = |value: &str| !value.is_empty() && value.bytes().all(|ch| ch.is_ascii_digit());
    if let Some((prefix, total)) = stem.rsplit_once("-of-") {
        if is_number(total)
            && prefix
                .rsplit_once('-')
                .is_some_and(|(_, index)| is_number(index))
        {
            return true;
        }
    }
    stem.strip_prefix("consolidated.").is_some_and(is_number)
        || stem
            .strip_prefix("mp_rank_")
            .and_then(|rest| rest.strip_suffix("_model_states"))
            .is_some_and(is_number)
}

fn directory_has_weights(directory: &Path) -> Result<bool> {
    let mut pending = VecDeque::from([directory.to_path_buf()]);
    let mut visited = HashSet::new();
    while let Some(directory) = pending.pop_front() {
        let physical = directory
            .canonicalize()
            .with_context(|| format!("cannot resolve model directory {}", directory.display()))?;
        if !visited.insert(physical) {
            continue;
        }
        for path in visible_children(&directory)? {
            if path.is_dir() {
                pending.push_back(path);
            } else if path.is_file() && is_weight_file(&path) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture {
        root: PathBuf,
        source: PathBuf,
        store: ModelStore,
    }

    impl Fixture {
        fn new() -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "werk-import-collection-{}-{}-{}",
                std::process::id(),
                super::super::unix_ts(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ));
            let source = root.join("collection");
            fs::create_dir_all(&source).unwrap();
            let store = ModelStore::resolve(Some(root.join("home"))).unwrap();
            Self {
                root,
                source,
                store,
            }
        }

        fn file(&self, name: &str) -> PathBuf {
            let path = self.source.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, b"fake weights or metadata").unwrap();
            path
        }

        fn werk_model(&self, directory: &str, id: &str) {
            let path = self.file(&format!("{directory}/manifest.json"));
            fs::write(
                path,
                serde_json::to_vec(&json!({
                    "id": id,
                    "source": {"kind": "local_path", "path": "/old/source"},
                    "format": "gguf",
                    "architecture": "llama",
                    "tokenizer_path": null,
                    "config_path": null,
                    "model_path": "files/model.gguf",
                    "backend": "llama-server",
                    "created_unix": 1,
                    "files": [{"path": "files/model.gguf", "size": 1, "checksum": "crc32:0"}]
                }))
                .unwrap(),
            )
            .unwrap();
            self.file(&format!("{directory}/files/model.gguf"));
        }

        fn discover(&self) -> Result<Vec<ModelImportCandidate>> {
            self.store.discover_import_collection(&self.source)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn collection_preserves_werk_ids_and_does_not_create_home() {
        let fixture = Fixture::new();
        fixture.werk_model("owner-zeta", "owner/zeta");
        fixture.werk_model("owner-alpha", "owner/alpha");

        let candidates = fixture.discover().unwrap();

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].id, "owner/alpha");
        assert_eq!(candidates[1].id, "owner/zeta");
        assert!(candidates.iter().all(|candidate| candidate.is_werk_model));
        assert_eq!(candidates[0].path, fixture.source.join("owner-alpha"));
        assert!(!fixture.store.home().exists());
    }

    #[test]
    fn collection_keeps_repository_components_and_shards_together() {
        let fixture = Fixture::new();
        fixture.file("text/config.json");
        fixture.file("text/model-00001-of-00002.safetensors");
        fixture.file("text/model-00002-of-00002.safetensors");
        fixture.file("image/model_index.json");
        fixture.file("image/transformer/config.json");
        fixture.file("image/transformer/diffusion_pytorch_model.safetensors");
        fixture.file("image/vae/diffusion_pytorch_model.safetensors");

        let candidates = fixture.discover().unwrap();

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].id, "image");
        assert_eq!(candidates[1].id, "text");
        assert!(candidates.iter().all(|candidate| !candidate.is_werk_model));
    }

    #[test]
    fn collection_imports_loose_models_and_ignores_unrelated_entries() {
        let fixture = Fixture::new();
        fixture.file("text.Q4_K_M.gguf");
        fixture.file("encoder.onnx");
        fixture.file("image.safetensors");
        fixture.file("README.md");
        fixture.file("tokenizer.json");
        fixture.file("tokenizer.bin");
        fixture.file("notes/data.txt");
        fixture.file(".hidden/model.gguf");
        fixture.file(".git/model.gguf");
        fixture.file("notes/.cache/model.safetensors");
        fs::create_dir_all(fixture.source.join("empty")).unwrap();

        let candidates = fixture.discover().unwrap();

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect::<Vec<_>>(),
            vec!["encoder", "image", "text.Q4_K_M"]
        );
    }

    #[test]
    fn collection_rejects_a_single_repository_root() {
        for marker in [
            "manifest.json",
            "config.json",
            "model_index.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ] {
            let fixture = Fixture::new();
            fixture.file(marker);
            fixture.file("model.gguf");

            let error = fixture.discover().unwrap_err().to_string();

            assert!(error.contains("without --all"), "{marker}: {error}");
            assert!(!fixture.store.home().exists());
        }
    }

    #[test]
    fn collection_rejects_loose_shards_with_actionable_error() {
        for shard in [
            "model-00001-of-00002.gguf",
            "model-00001-of-00002.safetensors",
            "pytorch_model-00001-of-00002.bin",
            "consolidated.00.pth",
            "mp_rank_00_model_states.pt",
        ] {
            let fixture = Fixture::new();
            fixture.file(shard);
            fixture.file("another.gguf");

            let error = fixture.discover().unwrap_err().to_string();

            assert!(error.contains("shard"), "{shard}: {error}");
            assert!(error.contains("child model directory"));
            assert!(!fixture.store.home().exists());
        }
    }

    #[test]
    fn collection_rejects_duplicate_and_sanitized_names_before_importing() {
        for names in [["model", "model"], ["owner/model", "owner-model"]] {
            let fixture = Fixture::new();
            fixture.werk_model("first", names[0]);
            fixture.werk_model("second", names[1]);

            let error = fixture.discover().unwrap_err().to_string();

            assert!(error.contains("same model directory"));
            assert!(!fixture.store.home().exists());
        }
    }

    #[test]
    fn collection_rejects_file_and_directory_name_collision() {
        let fixture = Fixture::new();
        fixture.file("model.gguf");
        fixture.file("model/weights.safetensors");

        assert!(
            fixture
                .discover()
                .unwrap_err()
                .to_string()
                .contains("same model directory")
        );
    }

    #[test]
    fn collection_rejects_existing_registration_without_writes() {
        let fixture = Fixture::new();
        fixture.file("alpha.gguf");
        fixture.file("zeta.gguf");
        fs::create_dir_all(fixture.store.model_dir("zeta")).unwrap();

        let error = fixture.discover().unwrap_err().to_string();

        assert!(error.contains("already exists"));
        assert!(error.contains("zeta"));
        assert!(!fixture.store.model_dir("alpha").exists());
    }

    #[test]
    fn collection_rejects_home_inside_source_model_before_creating_it() {
        let fixture = Fixture::new();
        fixture.file("model/model.gguf");
        let home = fixture.source.join("model/new/home");
        let store = ModelStore::resolve(Some(home.clone())).unwrap();

        let error = store
            .discover_import_collection(&fixture.source)
            .unwrap_err()
            .to_string();

        assert!(error.contains("outside source model directory"));
        assert!(!home.exists());
    }

    #[test]
    fn collection_rejects_whole_werk_home_as_root_or_child() {
        let fixture = Fixture::new();
        fixture.werk_model("old-home/models/model", "model");
        fixture.file("old-home/backends/library.pt");

        for source in [&fixture.source, &fixture.source.join("old-home")] {
            let error = fixture
                .store
                .discover_import_collection(source)
                .unwrap_err()
                .to_string();
            assert!(error.contains("appears to be a Werk home"));
            assert!(error.contains("old-home/models"));
        }
        assert!(!fixture.store.home().exists());
    }

    #[test]
    fn collection_rejects_invalid_manifest_and_invalid_model_id() {
        let fixture = Fixture::new();
        fixture.file("broken/manifest.json");

        assert!(
            fixture
                .discover()
                .unwrap_err()
                .to_string()
                .contains("invalid model manifest")
        );

        fixture.werk_model("broken", "../escape");
        assert!(
            format!("{:#}", fixture.discover().unwrap_err())
                .contains("unsupported path-like syntax")
        );
        assert!(!fixture.store.home().exists());
    }

    #[test]
    fn collection_rejects_empty_directory_and_file_source() {
        let fixture = Fixture::new();
        fixture.file("notes/README.md");
        assert!(
            fixture
                .discover()
                .unwrap_err()
                .to_string()
                .contains("no model directories")
        );

        let file = fixture.file("model.gguf");
        assert!(
            fixture
                .store
                .discover_import_collection(&file)
                .unwrap_err()
                .to_string()
                .contains("requires a model collection directory")
        );
    }

    #[cfg(unix)]
    #[test]
    fn collection_treats_dangling_target_symlink_as_existing() {
        let fixture = Fixture::new();
        fixture.file("model.gguf");
        fs::create_dir_all(fixture.store.models_dir()).unwrap();
        std::os::unix::fs::symlink(
            fixture.root.join("missing"),
            fixture.store.model_dir("model"),
        )
        .unwrap();

        assert!(
            fixture
                .discover()
                .unwrap_err()
                .to_string()
                .contains("already exists")
        );
    }

    #[cfg(unix)]
    #[test]
    fn collection_scan_handles_directory_symlink_cycles() {
        let fixture = Fixture::new();
        let notes = fixture.source.join("notes");
        fs::create_dir_all(&notes).unwrap();
        std::os::unix::fs::symlink(&notes, notes.join("cycle")).unwrap();
        fixture.file("model.gguf");

        let candidates = fixture.discover().unwrap();

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, "model");
    }
}
