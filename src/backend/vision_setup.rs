//! Optional setup choices. Runtime eligibility remains in RUNTIME_REGISTRY;
//! this module does not grant network/tool access or start inference.
use super::{BACKEND_INSTALL_TARGETS, BackendRuntime, RUNTIME_REGISTRY};
use crate::model_store::ModelStore;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, IsTerminal, Write},
    path::Path,
};

#[derive(Debug, Clone, Default, clap::Args)]
pub struct VisionOptions {
    /// Set up image input for this backend/model; does not prove image inference.
    #[arg(long, conflicts_with = "text_only")]
    pub vision: bool,
    /// Skip optional image components and keep this setup text-only.
    #[arg(long, conflicts_with = "vision")]
    pub text_only: bool,
    /// Do not contact repositories or install packages (existing-asset setup only).
    #[arg(long)]
    pub offline: bool,
}

impl VisionOptions {
    pub fn decide(&self, explanation: &str, previous: Option<bool>) -> Result<bool> {
        crate::ui_eprintln!("{explanation}");
        if self.vision {
            return Ok(true);
        }
        if self.text_only {
            return Ok(false);
        }
        if let Some(previous) = previous {
            return Ok(previous);
        }
        if !io::stdin().is_terminal() || !crate::terminal::interactive(crate::terminal::Stream::Err)
        {
            return Ok(false);
        }
        crate::ui_eprintln!(
            "Set up image inputs for supported models? [y = Set up / Enter = Skip / q = Cancel] (default: Skip)"
        );
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .context("cannot read optional vision choice")?;
        Self::parse_choice(&line)
    }

    fn parse_choice(line: &str) -> Result<bool> {
        match line.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" | "j" | "ja" => Ok(true),
            "" | "n" | "no" | "nein" => Ok(false),
            _ => bail!("Setup cancelled; no vision components were approved"),
        }
    }
}

pub fn installer_components(target: &str) -> Option<&'static str> {
    // Special llama builds share the server adapter, not the legacy FFI path.
    let target = match target {
        "llama-cuda-nvfp4" | "llama-cuda-offload" => "llama-cuda",
        other => other,
    };
    RUNTIME_REGISTRY.iter().find_map(|runtime| {
        if runtime.install_target != Some(target) || !runtime.capabilities.vision_language { return None; }
        match runtime.runtime {
            BackendRuntime::LlamaServer => Some("llama-server with libmtmd and support for the concrete model architecture; a matching per-model GGUF projector is selected separately at model setup. No projector is downloaded by backend install."),
            BackendRuntime::Vllm => Some("vLLM and a compatible accelerator/PyTorch build; model-specific safetensors, config, tokenizer and image processor. No GGUF projector and no model download are part of backend install."),
            BackendRuntime::MlxVlm => Some("mlx-vlm on Apple Silicon/Metal; model-specific safetensors, config and processor. mlx-lm alone does not enable image input. No model download is part of backend install."),
            _ => None,
        }
    })
}

pub fn installer_choice(
    store: &ModelStore,
    target: &str,
    options: &VisionOptions,
) -> Result<Option<bool>> {
    ensure!(
        BACKEND_INSTALL_TARGETS.contains(&target),
        "Unknown installer target"
    );
    let Some(components) = installer_components(target) else {
        ensure!(
            !options.vision,
            "Image-understanding setup is not implemented/applicable for {target}; use an eligible image-understanding adapter, not a media-generation or text-only installer"
        );
        crate::ui_eprintln!("{target}: no optional image-understanding setup in this adapter.");
        return Ok(None);
    };
    let path = store
        .home()
        .join("backends")
        .join(target)
        .join("vision-setup.json");
    let previous = if path.is_file() {
        Some(serde_json::from_slice::<bool>(&fs::read(&path)?)?)
    } else {
        None
    };
    options
        .decide(
            &format!("Optional image setup for backend {target}: {components}"),
            previous,
        )
        .map(Some)
}

pub fn save_installer_choice(store: &ModelStore, target: &str, enabled: bool) -> Result<()> {
    ensure!(
        BACKEND_INSTALL_TARGETS.contains(&target),
        "Unknown installer target"
    );
    atomic_json(
        &store
            .home()
            .join("backends")
            .join(target)
            .join("vision-setup.json"),
        &enabled,
    )
}

pub(crate) fn validate_processor_assets(
    store: &ModelStore,
    manifest: &crate::model_store::ModelManifest,
) -> Result<()> {
    manifest.require_image_setup()?;
    ensure!(
        manifest.supports_task(crate::capabilities::InferenceTask::ImageUnderstanding),
        "Model does not advertise image-understanding"
    );
    let has_processor = manifest.files.iter().any(|file| {
        let name = Path::new(&file.path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        ["preprocessor_config.json", "processor_config.json"].contains(&name)
            && store.absolute_model_file(manifest, &file.path).is_file()
    });
    ensure!(
        has_processor,
        "Image-capable model is missing its repository-specific processor/preprocessor configuration; import the complete matching model repository. A GGUF mmproj cannot replace safetensors processor assets"
    );
    Ok(())
}

pub(crate) fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().context("setup destination has no parent")?;
    fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.as_file().sync_all()?;
    file.persist(path).map_err(|e| e.error)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelVisionSetup {
    pub enabled: bool,
    pub backend: String,
    pub projector: Option<String>,
    /// Main-model source revision, also used for later repository downloads.
    /// A manually associated local projector is not thereby revision-verified.
    pub source_revision: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interactive_choice_defaults_to_skip_and_cancel_is_not_consent() {
        for choice in ["", "\n", "no", "nein"] {
            assert!(!VisionOptions::parse_choice(choice).unwrap());
        }
        for choice in ["y", "YES", "ja"] {
            assert!(VisionOptions::parse_choice(choice).unwrap());
        }
        for choice in ["q", "cancel", "perhaps"] {
            assert!(VisionOptions::parse_choice(choice).is_err());
        }
    }
    #[test]
    fn every_installer_has_an_honest_vision_choice() {
        for target in BACKEND_INSTALL_TARGETS {
            let expected = target.starts_with("llama-") || ["vllm", "mlx-vlm"].contains(target);
            assert_eq!(installer_components(target).is_some(), expected, "{target}");
        }
        for descriptor in RUNTIME_REGISTRY {
            if descriptor.capabilities.vision_language {
                assert!(installer_components(descriptor.install_target.unwrap()).is_some());
            }
        }
    }
    #[test]
    fn omitted_choice_does_not_approve_downloads_and_explicit_choices_persist() {
        let options = VisionOptions::default();
        assert!(!options.decide("fixture", None).unwrap());
        assert!(options.decide("fixture", Some(true)).unwrap());
        assert!(
            !VisionOptions {
                text_only: true,
                ..Default::default()
            }
            .decide("fixture", Some(true))
            .unwrap()
        );
        assert!(
            VisionOptions {
                vision: true,
                ..Default::default()
            }
            .decide("fixture", Some(false))
            .unwrap()
        );
    }
}
