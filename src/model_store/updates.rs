//! Stage an entire model revision before replacing its registration. The old
//! model stays usable during download and is retained as a rollback copy.
use super::*;
use anyhow::ensure;

impl ModelStore {
    /// Update an externally installed projector only if the upstream main
    /// weights still match the local model. Never mutate a content-addressed HF
    /// snapshot. Ordinary external directories retain their old projector.
    pub(crate) fn update_external_projector(&self, id: &str, relative: &str) -> Result<()> {
        let mut model = self.get_existing(id)?;
        ensure!(
            matches!(model.storage, ModelStorage::External { .. }),
            "This operation requires a linked external model"
        );
        let root = self.model_files_dir(&model);
        ensure!(
            !root
                .ancestors()
                .any(|p| p.file_name().is_some_and(|n| n == "snapshots")),
            "HF snapshots are immutable; link the newer complete snapshot instead"
        );
        let index = model
            .files
            .iter()
            .position(|f| f.path == relative)
            .context("Projector is not registered for this model")?;
        ensure!(
            is_gguf_projector_path(relative),
            "Selected file is not a projector"
        );
        let (repo, _) = crate::backend::updates::tracked_model_source(self, &model)?.context(
            "No verified HF origin; register/link the original HF repository before updating",
        )?;
        let info = crate::backend::updates::repository_info(self, &repo)?;
        let plan =
            feature_setup::projector_from_info(&repo, &info, relative.strip_prefix("files/"))?
                .context("Projector no longer exists upstream")?;
        let weights: Vec<_> = model
            .files
            .iter()
            .filter(|f| f.path.ends_with(".gguf") && !is_gguf_projector_path(&f.path))
            .collect();
        ensure!(
            !weights.is_empty(),
            "Cannot verify the linked model weights"
        );
        let activity = crate::terminal::Activity::start("Verifying external projector update");
        activity.phase("Checking that upstream main weights match the installed model");
        for file in weights {
            let name = file
                .path
                .strip_prefix("files/")
                .context("Invalid model file path")?;
            let expected = info["siblings"]
                .as_array()
                .context("Missing upstream files")?
                .iter()
                .find(|f| f["rfilename"] == name)
                .and_then(|f| f["lfs"]["sha256"].as_str())
                .context("Cannot verify upstream model checksum")?;
            let actual = file_sha256(&self.absolute_model_file(&model, &file.path))?;
            ensure!(
                actual == expected,
                "Main weights differ upstream; update/link the complete model and matching projector together"
            );
        }
        let target = self.absolute_model_file(&model, relative);
        let parent = target
            .parent()
            .context("Projector has no parent directory")?;
        let mut staged = tempfile::NamedTempFile::new_in(parent)?;
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(1800))
            .build()?;
        let mut request = client.get(format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            plan.repo, plan.revision, plan.file
        ));
        if let Some(token) = self.huggingface_token()? {
            request = request.bearer_auth(token);
        }
        let mut response = request.send()?.error_for_status()?.take(
            plan.size
                .checked_add(1)
                .context("Projector size overflow")?,
        );
        activity.phase("Downloading matching projector");
        activity.download(plan.size);
        let mut size = 0u64;
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let n = response.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            size += n as u64;
            ensure!(size <= plan.size, "Projector exceeds expected size");
            staged.write_all(&buffer[..n])?;
            activity.position(size);
        }
        staged.as_file().sync_all()?;
        ensure!(
            size == plan.size && file_sha256(staged.path())? == plan.sha256,
            "Projector integrity check failed"
        );
        self.validate_vision_projector(&model, staged.path())?;
        let backup = tempfile::Builder::new()
            .prefix(".werk-projector-previous-")
            .tempdir_in(parent)?
            .keep()
            .join("projector.gguf");
        fs::copy(&target, &backup)?;
        staged.persist(&target).map_err(|e| e.error)?;
        model.files[index].size = plan.size;
        model.files[index].checksum = format!("sha256:{}", plan.sha256);
        if let Err(e) = self.write_manifest(&model) {
            fs::copy(&backup, &target).context("Manifest update failed and projector restore failed; recover from the retained backup")?;
            return Err(e);
        }
        crate::backend::vision_setup::atomic_json(
            &self.model_dir(id).join("vision-source.json"),
            &plan,
        )?;
        activity.finish();
        crate::ui_println!(
            "Projector updated; previous file retained at {}",
            backup.display()
        );
        Ok(())
    }

    pub(crate) fn update_from_huggingface(
        &self,
        id: &str,
        progress: impl FnMut(PullProgress),
    ) -> Result<PathBuf> {
        let model = self.get_existing(id)?;
        ensure!(
            matches!(model.storage, ModelStorage::Managed),
            "Linked models are updated at their original source"
        );
        let ModelSource::HuggingFace { repo } = &model.source else {
            bail!("Only managed Hugging Face models have a verified update source");
        };
        self.ensure()?;
        let original = fs::read(self.model_dir(id).join(MANIFEST_FILE))?;
        let stage = tempfile::Builder::new()
            .prefix("model-update-")
            .tempdir_in(self.tmp_dir())?;
        let staged = ModelStore::resolve(Some(stage.path().into()))?;
        if let Some(token) = self.huggingface_token()? {
            staged.save_huggingface_token(&token)?;
        }
        let file = (model.format == ModelFormat::Gguf)
            .then(|| {
                model
                    .model_path
                    .as_deref()
                    .and_then(|s| s.strip_prefix("files/"))
            })
            .flatten();
        let vision = model
            .metadata
            .vision_setup
            .as_ref()
            .is_some_and(|s| s.enabled);
        let updated = staged.pull_from_huggingface_with_vision(
            repo,
            Some(id),
            file,
            None,
            false,
            |_| Ok(false),
            progress,
        )?;
        ensure!(
            updated.format == model.format,
            "New revision changes model format; review it as a separate import"
        );
        ensure!(
            updated.architecture == model.architecture,
            "New revision changes model architecture; review it as a separate import"
        );
        if vision {
            match staged.vision_assets(&updated, false)? {
                feature_setup::VisionAssets::Local => {
                    staged.configure_vision(id, true, None)?;
                }
                feature_setup::VisionAssets::Download(projector) => {
                    staged.install_projector(id, &projector)?;
                }
                _ => bail!(
                    "New revision has no verified matching vision assets; current model retained"
                ),
            }
        }
        ensure!(
            fs::read(self.model_dir(id).join(MANIFEST_FILE))? == original,
            "Model registration changed during download; current model retained"
        );
        let backups = self.home().join("backups/model-updates");
        fs::create_dir_all(&backups)?;
        let backup = tempfile::Builder::new()
            .prefix("previous-")
            .tempdir_in(&backups)?
            .keep();
        let previous = backup.join("model");
        activate(&self.model_dir(id), &staged.model_dir(id), &previous)?;
        Ok(backup.join("model"))
    }
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn activate(current: &Path, staged: &Path, backup: &Path) -> Result<()> {
    fs::rename(current, backup)
        .context("Cannot retain the previous model; current installation unchanged")?;
    if let Err(error) = fs::rename(staged, current) {
        if let Err(rollback) = fs::rename(backup, current) {
            bail!(
                "Activation failed: {error}; rollback failed: {rollback}; previous model is at {}",
                backup.display()
            );
        }
        return Err(error).context("Model activation failed; previous installation restored");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn activation_keeps_old_files_and_rolls_back_on_failure() {
        let dir = tempfile::tempdir().unwrap();
        let current = dir.path().join("current");
        let staged = dir.path().join("staged");
        let backup = dir.path().join("backup");
        fs::create_dir(&current).unwrap();
        fs::write(current.join("weights"), "old").unwrap();
        assert!(activate(&current, &staged, &backup).is_err());
        assert_eq!(fs::read(current.join("weights")).unwrap(), b"old");
        fs::create_dir(&staged).unwrap();
        fs::write(staged.join("weights"), "new").unwrap();
        activate(&current, &staged, &backup).unwrap();
        assert_eq!(fs::read(current.join("weights")).unwrap(), b"new");
        assert_eq!(fs::read(backup.join("weights")).unwrap(), b"old");
    }
}
