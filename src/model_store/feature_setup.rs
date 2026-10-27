//! Resolve missing vision assets without downloading weights or changing a model.
//! A confirmed plan pins its repository revision, file, length and SHA-256.
use super::*;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectorDownload {
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub size: u64,
    pub sha256: String,
    pub base_model: Option<String>,
}

impl ProjectorDownload {
    fn checked(&self) -> Result<Self> {
        projector_from_info(&self.repo, &serde_json::json!({
            "sha": self.revision, "cardData": {"base_model": self.base_model},
            "siblings": [{"rfilename": self.file, "lfs": {"size": self.size, "sha256": self.sha256}}]
        }), Some(&self.file))?.context("Invalid projector setup plan")
    }
}

#[derive(Debug, Clone)]
pub enum VisionAssets {
    Local,
    Download(ProjectorDownload),
    Unavailable(String),
    NotAdvertised,
}

fn base_model(info: &Value) -> Option<&str> {
    let base = &info["cardData"]["base_model"];
    base.as_str().or_else(|| {
        base.as_array()
            .filter(|items| items.len() == 1)
            .and_then(|items| items[0].as_str())
    })
}

fn image_capable(info: &Value) -> bool {
    info["pipeline_tag"] == "image-text-to-text"
        || info["cardData"]["pipeline_tag"] == "image-text-to-text"
        || info["config"]["vision_config"].is_object()
}

fn projector_from_info(
    repo: &str,
    info: &Value,
    selected: Option<&str>,
) -> Result<Option<ProjectorDownload>> {
    validate_hf_repo(repo)?;
    let revision = info["sha"]
        .as_str()
        .context("Repository revision is missing")?;
    anyhow::ensure!(
        revision.len() == 40 && revision.bytes().all(|c| c.is_ascii_hexdigit()),
        "Invalid projector repository revision"
    );
    let mut candidates: Vec<_> = info["siblings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|entry| {
            entry["rfilename"]
                .as_str()
                .is_some_and(is_gguf_projector_path)
        })
        .collect();
    if let Some(selected) = selected {
        candidates.retain(|entry| entry["rfilename"].as_str() == Some(selected));
        anyhow::ensure!(
            candidates.len() == 1,
            "Selected projector {selected} is absent from {repo}"
        );
    }
    let families: std::collections::BTreeSet<_> = candidates
        .iter()
        .map(|entry| super::vision::projector_family(entry["rfilename"].as_str().unwrap()))
        .collect();
    anyhow::ensure!(
        families.len() <= 1,
        "Multiple projector families in {repo}; select the matching file with werk artifacts vision MODEL --vision --projector PATH"
    );
    candidates.sort_by_key(|entry| {
        let name = entry["rfilename"].as_str().unwrap();
        (gguf_projector_priority(name), name)
    });
    let Some(entry) = candidates.first() else {
        return Ok(None);
    };
    let file = entry["rfilename"].as_str().unwrap();
    anyhow::ensure!(
        !Path::new(file).is_absolute()
            && file
                .split('/')
                .all(|p| !p.is_empty() && p != "." && p != "..")
            && !file.contains(['\\', '?', '#', '%']),
        "Unsafe projector repository path"
    );
    let size = entry["lfs"]["size"]
        .as_u64()
        .context("Projector size is missing")?;
    let sha256 = entry["lfs"]["sha256"]
        .as_str()
        .context("Projector checksum is missing")?;
    anyhow::ensure!(
        size > 0 && sha256.len() == 64 && sha256.bytes().all(|c| c.is_ascii_hexdigit()),
        "Invalid projector integrity metadata"
    );
    Ok(Some(ProjectorDownload {
        repo: repo.into(),
        revision: revision.into(),
        file: file.into(),
        size,
        sha256: sha256.into(),
        base_model: base_model(info).map(str::to_owned),
    }))
}

fn resolve_repository_vision(
    repo: &str,
    info: &Value,
    known: bool,
    selected: Option<&str>,
    mut companion_lookup: impl FnMut(&str) -> Result<Value>,
) -> Result<VisionAssets> {
    if let Some(projector) = projector_from_info(repo, &info, selected)? {
        return Ok(VisionAssets::Download(projector));
    }
    if !known && !image_capable(&info) {
        return Ok(VisionAssets::NotAdvertised);
    }
    if let Some(base) = base_model(&info) {
        validate_hf_repo(base)?;
        let companion_repo = format!("ggml-org/{}-GGUF", base.split('/').nth(1).unwrap());
        if companion_repo != repo {
            let companion = companion_lookup(&companion_repo)?;
            anyhow::ensure!(
                base_model(&companion) == Some(base),
                "Companion repository does not declare the same base model; no projector selected"
            );
            if let Some(projector) = projector_from_info(&companion_repo, &companion, None)? {
                return Ok(VisionAssets::Download(projector));
            }
        }
    }
    Ok(VisionAssets::Unavailable("The model advertises image input, but no matching projector was found in its source or the verified llama.cpp model package".into()))
}

impl ModelStore {
    fn setup_repository_info(&self, repo: &str, revision: Option<&str>) -> Result<Value> {
        validate_hf_repo(repo)?;
        let suffix = match revision {
            Some(revision) => {
                anyhow::ensure!(
                    revision.len() == 40 && revision.bytes().all(|c| c.is_ascii_hexdigit()),
                    "Invalid model source revision"
                );
                format!("/revision/{revision}")
            }
            None => String::new(),
        };
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()?;
        let mut request = client.get(format!(
            "https://huggingface.co/api/models/{repo}{suffix}?blobs=true"
        ));
        if let Some(token) = self.huggingface_token()? {
            request = request.bearer_auth(token);
        }
        let mut bytes = Vec::new();
        request
            .send()?
            .error_for_status()?
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        anyhow::ensure!(
            bytes.len() <= 8 * 1024 * 1024,
            "Model setup metadata exceeds its limit"
        );
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Existing assets take precedence. Cross-repository lookup is restricted
    /// to the llama.cpp publisher and requires the same explicit base-model ID.
    pub fn vision_assets(&self, manifest: &ModelManifest, offline: bool) -> Result<VisionAssets> {
        let has_local = manifest.files.iter().any(|f| {
            is_gguf_projector_path(&f.path) && self.absolute_model_file(manifest, &f.path).is_file()
        });
        if has_local
            || (manifest.format != ModelFormat::Gguf
                && manifest.supports_task(InferenceTask::ImageUnderstanding))
        {
            return Ok(VisionAssets::Local);
        }
        if manifest.format != ModelFormat::Gguf {
            return Ok(VisionAssets::NotAdvertised);
        }
        // A previously installed companion is stored under a local checksum
        // filename. Repair from its pinned source, not that local filename.
        let saved = self.model_dir(&manifest.id).join("vision-source.json");
        if saved.is_file() {
            let download: ProjectorDownload = serde_json::from_slice(&fs::read(saved)?)?;
            return Ok(VisionAssets::Download(download.checked()?));
        }
        let known = manifest.supports_task(InferenceTask::ImageUnderstanding);
        let ModelSource::HuggingFace { repo } = &manifest.source else {
            return Ok(if known {
                VisionAssets::Unavailable("Missing local model-specific vision projector".into())
            } else {
                VisionAssets::NotAdvertised
            });
        };
        if offline {
            return Ok(if known {
                VisionAssets::Unavailable(
                    "Missing vision projector; offline mode only uses existing assets".into(),
                )
            } else {
                VisionAssets::NotAdvertised
            });
        }
        let source_path = self.model_dir(&manifest.id).join("source-revision.json");
        let source: Value = if source_path.is_file() {
            serde_json::from_slice(&fs::read(source_path)?)?
        } else {
            Value::Null
        };
        let revision = manifest
            .metadata
            .vision_setup
            .as_ref()
            .and_then(|s| s.source_revision.as_deref())
            .or_else(|| {
                (source["repo"].as_str() == Some(repo))
                    .then(|| source["revision"].as_str())
                    .flatten()
            });
        let selected = manifest
            .metadata
            .vision_setup
            .as_ref()
            .and_then(|s| s.projector.as_deref())
            .map(|s| s.strip_prefix("files/").unwrap_or(s));
        let info = self.setup_repository_info(repo, revision)?;
        resolve_repository_vision(repo, &info, known, selected, |repo| {
            self.setup_repository_info(repo, None)
        })
    }

    /// Called only after the user confirms the displayed, concrete setup plan.
    pub fn install_projector(&self, id: &str, plan: &ProjectorDownload) -> Result<ModelManifest> {
        // Revalidate serialized plans before using their URL or length.
        let checked = plan.checked()?;
        self.ensure()?;
        let staging = tempfile::tempdir_in(self.tmp_dir())?;
        let path = staging.path().join("mmproj.gguf");
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(1800))
            .build()?;
        let mut request = client.get(format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            checked.repo, checked.revision, checked.file
        ));
        if let Some(token) = self.huggingface_token()? {
            request = request.bearer_auth(token);
        }
        let mut response = request.send()?.error_for_status()?.take(
            checked
                .size
                .checked_add(1)
                .context("Projector size overflow")?,
        );
        let mut file = fs::File::create(&path)?;
        let mut hash = Sha256::new();
        let mut length = 0u64;
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let n = response.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            length += n as u64;
            anyhow::ensure!(
                length <= checked.size,
                "Projector is larger than its approved download"
            );
            hash.update(&buffer[..n]);
            file.write_all(&buffer[..n])?;
        }
        anyhow::ensure!(
            length == checked.size
                && format!("{:x}", hash.finalize()) == checked.sha256.to_ascii_lowercase(),
            "Projector download failed its size/checksum verification; model unchanged"
        );
        file.sync_all()?;
        let manifest = self.configure_vision(id, true, Some(&path))?;
        crate::backend::vision_setup::atomic_json(
            &self.model_dir(id).join("vision-source.json"),
            &checked,
        )?;
        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flash_companion_requires_matching_base_model_and_pins_the_download() {
        let source = serde_json::json!({"sha": "a".repeat(40), "pipeline_tag": "image-text-to-text",
            "cardData": {"base_model": "Qwen/Qwen3.8-Flash-Next"}, "siblings": []});
        let companion = serde_json::json!({"sha": "c".repeat(40),
            "cardData": {"base_model": ["Qwen/Qwen3.8-Flash-Next"]},
            "siblings": [{"rfilename": "mmproj-Qwen3.8-Flash-Next-Q8_0.gguf", "lfs": {"size": 616703104, "sha256": "b".repeat(64)}}]});
        let assets = resolve_repository_vision(
            "vumpt/Qwen3.8-Flash-Next-GGUF",
            &source,
            false,
            None,
            |repo| {
                assert_eq!(repo, "ggml-org/Qwen3.8-Flash-Next-GGUF");
                Ok(companion.clone())
            },
        )
        .unwrap();
        let VisionAssets::Download(download) = assets else {
            panic!("expected projector")
        };
        assert_eq!(download.revision, "c".repeat(40));
        assert_eq!(download.size, 616703104);
        let mut wrong = companion;
        wrong["cardData"]["base_model"] = "Qwen/Other".into();
        assert!(
            resolve_repository_vision(
                "vumpt/Qwen3.8-Flash-Next-GGUF",
                &source,
                false,
                None,
                |_| Ok(wrong.clone())
            )
            .is_err()
        );
    }

    #[test]
    fn explicit_projector_wins_and_text_models_do_not_search_companions() {
        let info = serde_json::json!({"sha": "a".repeat(40), "siblings": [
            {"rfilename": "mmproj-A-F16.gguf", "lfs": {"size": 42, "sha256": "b".repeat(64)}},
            {"rfilename": "mmproj-B-F16.gguf", "lfs": {"size": 84, "sha256": "c".repeat(64)}}
        ]});
        assert!(projector_from_info("test/model", &info, None).is_err());
        assert_eq!(
            projector_from_info("test/model", &info, Some("mmproj-B-F16.gguf"))
                .unwrap()
                .unwrap()
                .size,
            84
        );
        assert!(projector_from_info("test/model", &info, Some("absent.gguf")).is_err());
        let text = serde_json::json!({"sha": "a".repeat(40), "pipeline_tag": "text-generation", "siblings": []});
        assert!(matches!(
            resolve_repository_vision("test/model", &text, false, None, |_| panic!(
                "no companion lookup for text-only model"
            ))
            .unwrap(),
            VisionAssets::NotAdvertised
        ));
    }

    #[test]
    fn companion_plan_requires_integrity_and_unambiguous_identity() {
        let mut info = serde_json::json!({"sha": "a".repeat(40), "cardData": {"base_model": ["Qwen/Test"]},
            "siblings": [{"rfilename": "mmproj-Test-F16.gguf", "lfs": {"size": 42, "sha256": "b".repeat(64)}}]});
        assert_eq!(base_model(&info), Some("Qwen/Test"));
        assert_eq!(
            projector_from_info("ggml-org/Test-GGUF", &info, None)
                .unwrap()
                .unwrap()
                .size,
            42
        );
        info["siblings"][0]["rfilename"] = "../mmproj-Test-F16.gguf".into();
        assert!(projector_from_info("ggml-org/Test-GGUF", &info, None).is_err());
        info["siblings"][0]["rfilename"] = "mmproj-Test-F16.gguf".into();
        info["siblings"][0]["lfs"]["sha256"] = "missing".into();
        assert!(projector_from_info("ggml-org/Test-GGUF", &info, None).is_err());
        info["cardData"]["base_model"] = serde_json::json!(["Qwen/Test", "Qwen/Other"]);
        assert_eq!(base_model(&info), None);
    }
}
