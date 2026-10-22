//! Read-only generation ceilings shared by discovery, reasoning and admission.
//! A default generation length is not a maximum and is never treated as one.
use crate::model_store::ModelManifest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatGenerationLimits {
    pub schema_version: u32,
    pub model: String,
    pub context_tokens: Option<usize>,
    pub max_output_tokens: Option<usize>,
}

pub fn profile(manifest: &ModelManifest, context: Option<usize>) -> ChatGenerationLimits {
    let positive = |value: &serde_json::Value| {
        value
            .as_u64()
            .or_else(|| value.get("max").and_then(|v| v.as_u64()))
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0 && *n <= 2_000_000)
    };
    let constraints = &manifest.metadata.parameter_constraints;
    let context = context.or_else(|| {
        [
            "max_position_embeddings",
            "model_max_length",
            "max_sequence_length",
        ]
        .iter()
        .filter_map(|key| constraints.get(*key).and_then(positive))
        .min()
    });
    let maximum = [
        "max_output_tokens",
        "max_completion_tokens",
        "max_new_tokens",
    ]
    .iter()
    .filter_map(|key| constraints.get(*key).and_then(positive))
    .chain(context.map(|n| n.saturating_sub(1)))
    .min();
    ChatGenerationLimits {
        schema_version: 1,
        model: manifest.id.clone(),
        context_tokens: context,
        max_output_tokens: maximum,
    }
}
