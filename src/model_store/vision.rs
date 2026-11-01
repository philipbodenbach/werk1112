use super::*;
use crate::backend::vision_setup::{ModelVisionSetup, atomic_json};
use anyhow::ensure;

impl ModelManifest {
    pub fn vision_setup_backend(&self) -> Result<&'static str> {
        match self.format {
            ModelFormat::Gguf if self.supports_task(InferenceTask::ImageUnderstanding) => {
                Ok("llama-server")
            }
            ModelFormat::SafeTensors
                if crate::backend::vllm_architecture_supports_images(
                    self.architecture.as_deref(),
                ) =>
            {
                Ok("vllm")
            }
            ModelFormat::SafeTensors | ModelFormat::Mlx
                if self.architecture.as_deref() == Some("gemma4_unified") =>
            {
                Ok("mlx-vlm")
            }
            _ => bail!(
                "Image-understanding setup is not supported for this model/adapter combination; no optional components will be installed"
            ),
        }
    }
}

pub(super) fn validate_projector_identity(main: &Path, projector: &Path) -> Result<()> {
    ensure!(
        detect_gguf_architecture(projector)?.as_deref() == Some("clip"),
        "Selected GGUF does not declare the clip projector architecture"
    );
    if let (Some(main), Some(projector)) = (
        super::gguf_conversion::gguf_model_identity(main)?,
        super::gguf_conversion::gguf_model_identity(projector)?,
    ) {
        ensure!(
            main == projector,
            "Projector model identity differs from the main GGUF; select artifacts from the same model and revision"
        );
    }
    if let Some(architecture) = detect_gguf_architecture(main)? {
        let input = super::gguf_conversion::gguf_unsigned_metadata(
            main,
            &format!("{architecture}.embedding_length"),
        )?;
        let output = super::gguf_conversion::gguf_unsigned_metadata(
            projector,
            "clip.vision.projection_dim",
        )?;
        if let (Some(input), Some(output)) = (input, output) {
            ensure!(
                input == output,
                "Projector output dimension {output} differs from model embedding dimension {input}"
            );
        }
    }
    Ok(())
}

/// A choice of precision is allowed only within one projector family. Two
/// different model families must never be resolved by filename sorting.
pub(super) fn projector_family(path: &str) -> String {
    let mut name = path.to_ascii_lowercase();
    for part in ["bf16", "f16", "f32", "q8_0"] {
        name = name.replace(part, "<precision>");
    }
    name
}

pub(super) fn select_projector(root: &Path, explicit: Option<&str>) -> Result<Option<String>> {
    if let Some(explicit) = explicit {
        let path = resolve_pull_file(root, explicit)?;
        ensure!(
            is_gguf_projector_path(&path),
            "Selected file is not a recognized GGUF projector"
        );
        ensure!(
            !path.contains([',', '*', '?', '[', ']']),
            "Projector filename cannot contain Git LFS filter metacharacters"
        );
        return Ok(Some(path));
    }
    let mut files = Vec::new();
    collect_files(root, &mut files)?;
    let candidates: Vec<_> = files
        .iter()
        .filter_map(|p| relative_string(root, p))
        .filter(|p| is_gguf_projector_path(p))
        .collect();
    let families: std::collections::BTreeSet<_> =
        candidates.iter().map(|p| projector_family(p)).collect();
    ensure!(
        families.len() <= 1,
        "Ambiguous projector families; select a compatible repository file explicitly with --projector. Candidates: {}",
        candidates.join(", ")
    );
    let selected = first_gguf_projector_path(root, &files);
    if let Some(path) = &selected {
        ensure!(
            !path.contains([',', '*', '?', '[', ']']),
            "Projector filename cannot contain Git LFS filter metacharacters"
        );
    }
    Ok(selected)
}

pub(super) fn pull_vision_backend(
    root: &Path,
    primary: Option<&str>,
) -> Result<Option<&'static str>> {
    let config = fs::read(root.join("config.json"))
        .ok()
        .and_then(|data| serde_json::from_slice::<Value>(&data).ok());
    let mut files = Vec::new();
    collect_files(root, &mut files)?;
    let has_projector = files
        .iter()
        .filter_map(|p| p.to_str())
        .any(is_gguf_projector_path);
    let is_vlm = has_projector
        || config
            .as_ref()
            .is_some_and(|c| c["vision_config"].is_object() && c["language_model_only"] != true);
    if !is_vlm {
        return Ok(None);
    }
    if primary.is_some_and(|p| extension_eq(Path::new(p), "gguf")) {
        return Ok(Some("llama-server"));
    }
    let architecture = config.as_ref().and_then(|c| c["model_type"].as_str());
    if crate::backend::vllm_architecture_supports_images(architecture) {
        return Ok(Some("vllm"));
    }
    if architecture == Some("gemma4_unified") {
        return Ok(Some("mlx-vlm"));
    }
    Ok(None)
}

pub(super) fn pull_vision_description(
    root: &Path,
    repo: &str,
    revision: &str,
    destination: &Path,
    backend: &str,
) -> Result<String> {
    let mut description = format!(
        "Optional image input for {backend}; source {repo}@{revision}; destination {}. ",
        destination.display()
    );
    if backend == "llama-server" {
        let mut files = Vec::new();
        collect_files(root, &mut files)?;
        for file in files
            .iter()
            .filter_map(|p| relative_string(root, p))
            .filter(|p| is_gguf_projector_path(p))
        {
            let bytes = lfs_pointer_total_for_files(root, std::slice::from_ref(&file))?
                .map(|n| format!("{n} bytes"))
                .unwrap_or_else(|| "size unavailable".into());
            description.push_str(&format!("Projector candidate {file}: {bytes}. "));
        }
        description.push_str("Only one selected projector is downloaded; its runtime/model compatibility must still be tested.");
    } else {
        description.push_str("Uses this repository's existing safetensors, tokenizer, processor and config; vision weights may be inseparable from the main checkpoint. No separate GGUF projector or runtime installation is included.");
    }
    Ok(description)
}

impl ModelStore {
    pub(crate) fn validate_vision_projector(
        &self,
        manifest: &ModelManifest,
        projector: &Path,
    ) -> Result<()> {
        let main = manifest
            .model_path
            .as_deref()
            .context("Missing main model file")?;
        validate_projector_identity(&self.absolute_model_file(manifest, main), projector)
    }

    /// Select existing assets or explicitly import one local projector, without
    /// downloading/replacing the main model or installing a runtime.
    pub fn configure_vision(
        &self,
        id: &str,
        enabled: bool,
        projector: Option<&Path>,
    ) -> Result<ModelManifest> {
        let activity = crate::terminal::Activity::start(format!("Configuring vision for {id}"));
        activity.phase("Validating and saving model components");
        let mut manifest = self.get(id)?;
        let previous = manifest.metadata.vision_setup.clone();
        let backend = if projector.is_some() && manifest.format == ModelFormat::Gguf {
            "llama-server"
        } else {
            manifest.vision_setup_backend()?
        };
        ensure!(
            !enabled
                || projector.is_some()
                || manifest.supports_task(InferenceTask::ImageUnderstanding),
            "Model does not advertise image-understanding; inspect its architecture and model configuration first"
        );
        ensure!(
            projector.is_none() || (enabled && manifest.format == ModelFormat::Gguf),
            "--projector requires a GGUF vision setup"
        );
        let root = self.model_files_dir(&manifest);
        let mut selected = previous.as_ref().and_then(|p| p.projector.clone());
        if let Some(source) = projector {
            let source = source
                .canonicalize()
                .context("cannot open selected projector")?;
            ensure!(
                is_gguf_projector_path(&source.to_string_lossy()),
                "Expected a named mmproj/projector GGUF file"
            );
            let main = manifest
                .model_path
                .as_ref()
                .context("Missing main model file")?;
            validate_projector_identity(&self.absolute_model_file(&manifest, main), &source)?;
            push_unique(
                &mut manifest.metadata.tasks,
                InferenceTask::ImageUnderstanding,
            );
            push_unique(
                &mut manifest.metadata.input_modalities,
                InputModality::Image,
            );
            let canonical_root = root.canonicalize()?;
            let relative = if source.starts_with(&canonical_root) {
                relative_string(&canonical_root, &source).context("invalid projector path")?
            } else {
                ensure!(
                    matches!(manifest.storage, ModelStorage::Managed),
                    "For an external model, put the matching projector in its source directory before selecting it"
                );
                let checksum = crc32(&source)?;
                let relative = format!("vision/mmproj-{checksum:08x}.gguf");
                let target = root.join(&relative);
                fs::create_dir_all(target.parent().unwrap())?;
                if !target.is_file() {
                    let mut staged = tempfile::NamedTempFile::new_in(target.parent().unwrap())?;
                    std::io::copy(&mut fs::File::open(&source)?, &mut staged)?;
                    staged.as_file().sync_all()?;
                    staged.persist_noclobber(&target).map_err(|e| e.error)?;
                }
                ensure!(
                    crc32(&target)? == checksum,
                    "Existing projector checksum differs; no manifest change was applied"
                );
                relative
            };
            let path = format!("files/{relative}");
            if !manifest.files.iter().any(|f| f.path == path) {
                manifest.files.push(ModelFile {
                    path: path.clone(),
                    size: fs::metadata(&source)?.len(),
                    checksum: format!("crc32:{:08x}", crc32(&source)?),
                });
            }
            selected = Some(path);
        }
        if enabled && backend == "llama-server" && selected.is_none() {
            // The selection is constrained to manifest-listed local assets.
            let selected_file = select_projector(&root, None)?.context("Missing model-specific projector. Import the matching file with werk artifacts vision MODEL --vision --projector PATH; no download is performed by an image request")?;
            let path = format!("files/{selected_file}");
            ensure!(
                manifest.files.iter().any(|f| f.path == path),
                "Projector is not registered; select it explicitly with --projector PATH"
            );
            selected = Some(path);
        }
        if enabled && backend == "llama-server" {
            let path = selected.as_ref().context("Missing selected projector")?;
            ensure!(
                manifest.files.iter().any(|f| &f.path == path),
                "Selected projector is not registered in the manifest"
            );
            let main = manifest
                .model_path
                .as_ref()
                .context("Missing main model file")?;
            validate_projector_identity(
                &self.absolute_model_file(&manifest, main),
                &self.absolute_model_file(&manifest, path),
            )?;
            let absolute = self.absolute_model_file(&manifest, path);
            let checksum = format!("crc32:{:08x}", crc32(&absolute)?);
            let size = fs::metadata(&absolute)?.len();
            let entry = manifest.files.iter_mut().find(|f| &f.path == path).unwrap();
            entry.checksum = checksum;
            entry.size = size;
        }
        manifest.metadata.vision_setup = Some(ModelVisionSetup {
            enabled,
            backend: backend.into(),
            projector: selected,
            source_revision: previous.and_then(|p| p.source_revision),
        });
        if enabled && backend != "llama-server" {
            if backend == "vllm" {
                crate::backend::VllmBackend::validate_image_assets(self, &manifest)?;
            } else {
                crate::backend::vision_setup::validate_processor_assets(self, &manifest)?;
            }
        }
        atomic_json(&self.model_dir(id).join(MANIFEST_FILE), &manifest)?;
        activity.finish();
        Ok(manifest)
    }

    /// Fetch only a projector from the exact revision recorded at model pull.
    /// LFS selection, authentication and integrity checks reuse the normal pull
    /// machinery. Failed/declined setup never changes the working manifest.
    pub fn download_vision_projector<C>(
        &self,
        id: &str,
        file: Option<&str>,
        mut consent: C,
    ) -> Result<ModelManifest>
    where
        C: FnMut(&str) -> Result<bool>,
    {
        let manifest = self.get(id)?;
        ensure!(
            manifest.format == ModelFormat::Gguf,
            "Separate projector downloads apply only to GGUF/llama-server models"
        );
        if file.is_none()
            && manifest.files.iter().any(|f| {
                is_gguf_projector_path(&f.path)
                    && self.absolute_model_file(&manifest, &f.path).is_file()
            })
        {
            // Re-enable/validate local assets even when the previous choice was
            // text-only; existence alone must not bypass identity validation.
            return self.configure_vision(id, true, None);
        }
        let ModelSource::HuggingFace { repo } = &manifest.source else {
            bail!(
                "Local import has no verified repository revision; associate an existing matching projector with --projector PATH"
            );
        };
        let revision = manifest.metadata.vision_setup.as_ref().and_then(|s| s.source_revision.as_deref())
            .context("No source revision recorded for this older import. Select a verified matching local projector with --projector PATH; guessing a repository's current projector is unsafe")?;
        ensure!(
            revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()),
            "Invalid recorded source revision"
        );
        validate_hf_repo(repo)?;
        self.ensure()?;
        let staging = tempfile::tempdir_in(self.tmp_dir())?;
        let root = staging.path();
        let token = self.huggingface_token()?;
        let url = format!("https://huggingface.co/{repo}");
        let checking =
            crate::terminal::Activity::start(format!("Checking projector source for {id}"));
        checking.phase("Fetching recorded repository revision");
        for args in [
            vec!["init"],
            vec!["remote", "add", "origin", &url],
            vec!["fetch", "--depth", "1", "origin", revision],
            vec!["checkout", "--detach", "FETCH_HEAD"],
        ] {
            let mut command = Command::new("git");
            command
                .arg("-C")
                .arg(root)
                .env("GIT_LFS_SKIP_SMUDGE", "1")
                .args(args);
            configure_huggingface_git_auth(&mut command, token.as_deref());
            run_git_with_progress(
                &mut command,
                "Cannot prepare the recorded model revision",
                None,
                None,
                |_| {},
            )?;
        }
        let selected = select_projector(root, file)?
            .context("No model-specific projector at the recorded revision")?;
        let bytes = lfs_pointer_total_for_files(root, std::slice::from_ref(&selected))?;
        checking.finish();
        let approved = consent(&format!(
            "Image setup for model {id}; source {repo}@{revision}; file {selected}; additional download {}; destination {}. Main model weights are reused.",
            bytes
                .map(|n| format!("{n} bytes"))
                .unwrap_or_else(|| "size unavailable".into()),
            self.model_files_dir(&manifest).join("vision").display()
        ))?;
        if !approved {
            return Ok(manifest);
        }
        let activity =
            crate::terminal::Activity::start(format!("Downloading vision projector {selected}"));
        if let Some(total) = bytes {
            activity.download(total);
        } else {
            activity.phase("Downloading projector (size unknown)");
        }
        for args in [
            vec!["lfs", "install", "--local"],
            vec![
                "lfs",
                "pull",
                "origin",
                "--include",
                &selected,
                "--exclude",
                "",
            ],
            vec!["lfs", "checkout", &selected],
        ] {
            let mut command = Command::new("git");
            command.arg("-C").arg(root).args(args);
            configure_huggingface_git_auth(&mut command, token.as_deref());
            run_git_with_progress(
                &mut command,
                "Cannot download selected projector; previous model setup is unchanged",
                Some(root),
                bytes,
                |event| {
                    if let GitCommandProgress::Stats { bytes, .. } = event {
                        activity.position(bytes);
                    }
                },
            )?;
        }
        activity.phase("Verifying and activating downloaded projector");
        ensure_lfs_file_downloaded(root, &selected)?;
        let manifest = self.configure_vision(id, true, Some(&root.join(selected)))?;
        activity.finish();
        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn gguf(path: &Path, architecture: &str, name: &str) {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(2u64.to_le_bytes());
        for (key, value) in [
            ("general.architecture", architecture),
            ("general.name", name),
        ] {
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.extend(8u32.to_le_bytes());
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.as_bytes());
        }
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn matching_projector_name_cannot_hide_incompatible_dimensions() {
        let temp = tempfile::tempdir().unwrap();
        let main = temp.path().join("model.gguf");
        let projector = temp.path().join("mmproj.gguf");
        for (path, architecture, key, dimension) in [
            (&main, "qwen35", "qwen35.embedding_length", 2560u32),
            (&projector, "clip", "clip.vision.projection_dim", 5120u32),
        ] {
            gguf(path, architecture, "same model name");
            let mut bytes = fs::read(path).unwrap();
            bytes[16..24].copy_from_slice(&3u64.to_le_bytes());
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.extend(4u32.to_le_bytes());
            bytes.extend(dimension.to_le_bytes());
            fs::write(path, bytes).unwrap();
        }
        let error = validate_projector_identity(&main, &projector).unwrap_err();
        assert!(error.to_string().contains("dimension"));
        let mut bytes = fs::read(&projector).unwrap();
        let end = bytes.len();
        bytes[end - 4..].copy_from_slice(&2560u32.to_le_bytes());
        fs::write(&projector, bytes).unwrap();
        validate_projector_identity(&main, &projector).unwrap();
    }

    #[test]
    fn late_setup_reuses_assets_rejects_wrong_identity_and_preserves_working_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source");
        fs::create_dir(&source).unwrap();
        gguf(&source.join("model.gguf"), "qwen35", "Qwen3.8 27B");
        gguf(&source.join("mmproj-BF16.gguf"), "clip", "Qwen3.8-27B");
        let store = ModelStore::resolve(Some(tmp.path().join("store"))).unwrap();
        let original = store.import_path(&source, "vlm").unwrap();
        assert!(original.supports_task(InferenceTask::ImageUnderstanding));
        let disabled = store.configure_vision("vlm", false, None).unwrap();
        assert!(disabled.require_image_setup().is_err());
        let enabled = store
            .download_vision_projector("vlm", None, |_| panic!("local assets need no download"))
            .unwrap();
        assert!(enabled.require_image_setup().is_ok());
        assert_eq!(enabled.files.len(), original.files.len());
        assert_ne!(
            ModelRuntimeIdentity::from_manifest(&disabled)
                .unwrap()
                .to_string(),
            ModelRuntimeIdentity::from_manifest(&enabled)
                .unwrap()
                .to_string()
        );
        // Re-associating a changed file updates its cache fingerprint even if
        // its path and declared model identity stay the same.
        let chosen = enabled
            .metadata
            .vision_setup
            .as_ref()
            .unwrap()
            .projector
            .as_ref()
            .unwrap();
        let path = store.absolute_model_file(&enabled, chosen);
        use std::io::Write;
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0; 32])
            .unwrap();
        let refreshed = store.configure_vision("vlm", true, None).unwrap();
        assert_ne!(
            ModelRuntimeIdentity::from_manifest(&enabled)
                .unwrap()
                .to_string(),
            ModelRuntimeIdentity::from_manifest(&refreshed)
                .unwrap()
                .to_string()
        );
        // No repository access/consent callback when the selected asset exists.
        store
            .download_vision_projector("vlm", None, |_| panic!("must reuse installed asset"))
            .unwrap();
        let before = fs::read(store.model_dir("vlm").join(MANIFEST_FILE)).unwrap();
        let wrong = tmp.path().join("mmproj-Flash-Next.gguf");
        gguf(&wrong, "clip", "Qwen3.8 Flash Next");
        let error = store
            .configure_vision("vlm", true, Some(&wrong))
            .unwrap_err();
        assert!(error.to_string().contains("identity differs"));
        assert_eq!(
            before,
            fs::read(store.model_dir("vlm").join(MANIFEST_FILE)).unwrap()
        );
        let selected = enabled
            .metadata
            .vision_setup
            .as_ref()
            .unwrap()
            .projector
            .as_ref()
            .unwrap();
        fs::remove_file(store.absolute_model_file(&enabled, selected)).unwrap();
        assert!(store.configure_vision("vlm", true, None).is_err());
        assert_eq!(
            before,
            fs::read(store.model_dir("vlm").join(MANIFEST_FILE)).unwrap()
        );
    }

    #[test]
    fn pull_plan_requires_a_real_vision_configuration() {
        let root = tempfile::tempdir().unwrap();
        for (config, backend) in [
            (
                serde_json::json!({"model_type":"qwen3_5", "vision_config":{}}),
                Some("vllm"),
            ),
            (
                serde_json::json!({"model_type":"qwen3_5_moe", "vision_config":{}}),
                Some("vllm"),
            ),
            (
                serde_json::json!({"model_type":"qwen3_next", "vision_config":null}),
                None,
            ),
            (
                serde_json::json!({"model_type":"qwen3_5", "vision_config":{}, "language_model_only":true}),
                None,
            ),
            (
                serde_json::json!({"model_type":"gemma4_unified", "vision_config":{}}),
                Some("mlx-vlm"),
            ),
        ] {
            fs::write(root.path().join("config.json"), config.to_string()).unwrap();
            assert_eq!(pull_vision_backend(root.path(), None).unwrap(), backend);
        }
    }
    #[test]
    fn projector_precision_choice_never_crosses_model_families() {
        let root = tempfile::tempdir().unwrap();
        for name in [
            "mmproj-Qwen3.8-27B-F16.gguf",
            "mmproj-Qwen3.8-27B-BF16.gguf",
        ] {
            fs::write(root.path().join(name), b"fixture").unwrap();
        }
        assert_eq!(
            select_projector(root.path(), None).unwrap().as_deref(),
            Some("mmproj-Qwen3.8-27B-F16.gguf")
        );
        fs::write(
            root.path().join("mmproj-Qwen3.8-Flash-Next-F16.gguf"),
            b"fixture",
        )
        .unwrap();
        assert!(
            select_projector(root.path(), None)
                .unwrap_err()
                .to_string()
                .contains("Ambiguous")
        );
        assert!(
            select_projector(root.path(), Some("mmproj-Qwen3.8-27B-F16.gguf"))
                .unwrap()
                .is_some()
        );
    }
}
