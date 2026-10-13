//! Evidence-backed support catalog. Availability is not an inference verification.
use crate::{
    capabilities::{InferenceTask, InputModality, OutputModality},
    model_store::{ModelFormat, ModelManifest, ModelMetadata, ModelSource, ModelStore, unix_ts},
};
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportStatus {
    VerifiedLocal,
    VerifiedRemote,
    ImplementedUntested,
    BlockedUpstream,
    Unavailable,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    Mistral,
    Bfl,
    ElevenLabs,
}
impl Provider {
    pub fn key_env(self) -> &'static str {
        match self {
            Self::Mistral => "MISTRAL_API_KEY",
            Self::Bfl => "BFL_API_KEY",
            Self::ElevenLabs => "ELEVENLABS_API_KEY",
        }
    }
    pub fn origin(self) -> &'static str {
        match self {
            Self::Mistral => "https://api.mistral.ai",
            Self::Bfl => "https://api.bfl.ai",
            Self::ElevenLabs => "https://api.elevenlabs.io",
        }
    }
    pub fn header(self) -> &'static str {
        match self {
            Self::Mistral => "authorization",
            Self::Bfl => "x-key",
            Self::ElevenLabs => "xi-api-key",
        }
    }
}

#[derive(Serialize)]
pub struct SupportEntry {
    pub id: &'static str,
    pub vendor: &'static str,
    pub local: SupportStatus,
    pub remote: SupportStatus,
    pub runtime: &'static str,
    pub capabilities: &'static [&'static str],
    pub license: &'static str,
    pub blocker: &'static str,
    pub source: &'static str,
}
use SupportStatus::*;
pub const CATALOG: &[SupportEntry] = &[
    SupportEntry {
        id: "Aleph-Alpha/Kolibri-1",
        vendor: "Aleph Alpha",
        local: ImplementedUntested,
        remote: Unavailable,
        runtime: "vllm + aleph-alpha-inference>=1",
        capabilities: &["text_generation", "tool_calling", "reasoning", "streaming"],
        license: "Apache-2.0",
        blocker: "Local inference requires FP8 weights (~78 GB), compatible NVIDIA hardware and vendor plugin; no official hosted endpoint verified.",
        source: "https://huggingface.co/Aleph-Alpha/Kolibri-1",
    },
    SupportEntry {
        id: "mistral-large-4",
        vendor: "Mistral AI",
        local: BlockedUpstream,
        remote: ImplementedUntested,
        runtime: "Mistral HTTPS API",
        capabilities: &[
            "text_generation",
            "image_understanding",
            "tool_calling",
            "structured_output",
            "streaming",
        ],
        license: "API terms; weight license must be checked at release",
        blocker: "Weights announced for 2026-10-31; remote inference needs MISTRAL_API_KEY and account access. No billable live test performed.",
        source: "https://docs.mistral.ai/models/mistral-large-4-0",
    },
    SupportEntry {
        id: "black-forest-labs/flux-3-action-droid",
        vendor: "Black Forest Labs",
        local: ImplementedUntested,
        remote: Unavailable,
        runtime: "flux-action (DROID policy)",
        capabilities: &["action_prediction", "image_input", "state_input"],
        license: "FLUX Kommunity License v1.0 (checkpoint); separate upstream code license",
        blocker: "Requires DROID checkpoint and pinned encoders, Linux/NVIDIA, matched torch/CUDA/NATTEN environment. No official hosted API verified. SO-101 is a different contract.",
        source: "https://github.com/black-forest-labs/flux-action/blob/main/docs/setup.md",
    },
    SupportEntry {
        id: "flux-3-image",
        vendor: "Black Forest Labs",
        local: Unavailable,
        remote: ImplementedUntested,
        runtime: "BFL HTTPS API (async polling)",
        capabilities: &["image_generation", "image_editing", "image_input"],
        license: "BFL API service terms",
        blocker: "No downloadable FLUX 3 Image checkpoint verified; BFL_API_KEY required. No billable live test performed.",
        source: "https://docs.bfl.ai/api-reference/utility/generate-an-image-with-flux-3",
    },
    SupportEntry {
        id: "eleven_v4",
        vendor: "ElevenLabs",
        local: Unavailable,
        remote: ImplementedUntested,
        runtime: "ElevenLabs text-to-dialogue HTTPS API",
        capabilities: &["speech_synthesis", "dialogue", "audio_streaming"],
        license: "ElevenLabs service terms",
        blocker: "No public weights; ELEVENLABS_API_KEY and voice IDs required. No billable live test performed.",
        source: "https://elevenlabs.io/docs/overview/models",
    },
];

pub fn remote(manifest: &ModelManifest) -> Option<(Provider, &str)> {
    if let ModelSource::Provider { provider, model } = &manifest.source {
        Some((*provider, model))
    } else {
        None
    }
}
pub fn is_kolibri(manifest: &ModelManifest) -> bool {
    manifest.architecture.as_deref().is_some_and(|s| {
        s.eq_ignore_ascii_case("kolibri1") || s.eq_ignore_ascii_case("Kolibri1ForCausalLM")
    })
}
pub fn is_action(manifest: &ModelManifest) -> bool {
    manifest.architecture.as_deref() == Some("flux3_action_droid")
}

pub fn register_remote(
    store: &ModelStore,
    model: &str,
    name: Option<&str>,
) -> Result<ModelManifest> {
    let (provider, tasks) = match model {
        "mistral-large-4" => (
            Provider::Mistral,
            vec![
                InferenceTask::TextGeneration,
                InferenceTask::ImageUnderstanding,
            ],
        ),
        "flux-3-image" => (
            Provider::Bfl,
            vec![InferenceTask::ImageGeneration, InferenceTask::ImageEditing],
        ),
        "eleven_v4" => (Provider::ElevenLabs, vec![InferenceTask::TextToSpeech]),
        _ => bail!(
            "unsupported_model: no verified official provider contract for {model}; use werk support list"
        ),
    };
    let id = name.unwrap_or(model);
    ensure!(
        !id.is_empty()
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
            && id != "."
            && id != "..",
        "invalid_model: choose a simple model name"
    );
    ensure!(
        !store
            .model_dir(id)
            .join(crate::model_store::MANIFEST_FILE)
            .exists(),
        "model already registered: {id}"
    );
    let outputs = tasks
        .iter()
        .map(|task| task.output_modality())
        .collect::<std::collections::HashSet<OutputModality>>()
        .into_iter()
        .collect();
    let mut inputs = vec![InputModality::Text];
    if provider != Provider::ElevenLabs {
        inputs.push(InputModality::Image);
    }
    let manifest = ModelManifest {
        id: id.into(),
        source: ModelSource::Provider {
            provider,
            model: model.into(),
        },
        format: ModelFormat::Unknown,
        architecture: None,
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "provider".into(),
        created_unix: unix_ts(),
        files: vec![],
        artifacts: vec![],
        storage: Default::default(),
        metadata: ModelMetadata {
            tasks,
            input_modalities: inputs,
            output_modalities: outputs,
            compatible_runtimes: vec!["provider".into()],
            ..Default::default()
        },
    };
    store.ensure()?;
    std::fs::create_dir_all(store.model_dir(id))?;
    store.write_manifest(&manifest)?;
    Ok(manifest)
}
