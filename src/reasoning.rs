//! Server-owned reasoning profiles. Clients select IDs; only Werk maps them to
//! native controls and reconciles defaults with request/output limits.
use crate::{model_store::ModelManifest, observability::ReasoningDefaults};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct BackendReasoning {
    pub backend: String,
    pub defaults: ReasoningDefaults,
    pub budget_control: bool,
    pub native_efforts: Vec<Value>,
}

pub fn consistent(values: impl IntoIterator<Item = BackendReasoning>) -> Option<BackendReasoning> {
    let mut values = values.into_iter();
    let first = values.next()?;
    values.all(|next| next == first).then_some(first)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReasoningTier {
    pub id: String,
    pub label: String,
    pub available: bool,
    pub reasoning_tokens: Option<u64>,
    pub minimum_output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_effort: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReasoningLimits {
    pub context_tokens: Option<u64>,
    pub output_tokens: u64,
    pub answer_reserve_tokens: u64,
    pub reasoning_tokens: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReasoningProfile {
    pub schema_version: u32,
    pub model: String,
    pub backend: Option<String>,
    pub supported: bool,
    pub source: String,
    pub tiers: Vec<ReasoningTier>,
    pub default_tier: Option<String>,
    pub default_adjusted_to_limits: bool,
    pub limits: ReasoningLimits,
}

pub fn profile(
    manifest: &ModelManifest,
    runtime: Option<&BackendReasoning>,
    context: Option<usize>,
    output: usize,
) -> ReasoningProfile {
    let limits = crate::generation_limits::profile(manifest, context);
    let context = limits.context_tokens;
    let output = limits
        .max_output_tokens
        .map_or(output, |cap| output.min(cap));
    let output = context.map_or(output, |ctx| output.min(ctx.saturating_sub(512))) as u64;
    let reserve = output.min(512);
    let mut profile = ReasoningProfile {
        schema_version: 1,
        model: manifest.id.clone(),
        backend: runtime.map(|r| r.backend.clone()),
        supported: false,
        source: "unknown".into(),
        tiers: vec![],
        default_tier: None,
        default_adjusted_to_limits: false,
        limits: ReasoningLimits {
            context_tokens: context.map(|n| n as u64),
            output_tokens: output,
            answer_reserve_tokens: 0,
            reasoning_tokens: None,
        },
    };
    let Some(runtime) = runtime else {
        return profile;
    };
    profile.source = runtime.defaults.source.clone();
    if runtime.budget_control {
        profile.limits.answer_reserve_tokens = reserve;
        let configured = runtime
            .defaults
            .budget_tokens
            .filter(|n| *n >= 0)
            .map(|n| n as u64);
        let ceiling = configured
            .unwrap_or(u64::MAX)
            .min(output.saturating_sub(reserve));
        profile.limits.reasoning_tokens = Some(ceiling);
        // These are Werk's bounded token-budget tiers, not claims that every
        // model's template implements native effort strings with these names.
        for (id, label, tokens) in [
            ("none", "Off", 0),
            ("low", "Low", 512),
            ("medium", "Medium", 1024),
            ("high", "High", 2048),
            ("max", "Max", 4096),
            ("ultra", "Ultra", 8192),
        ] {
            profile.tiers.push(ReasoningTier {
                id: id.into(),
                label: label.into(),
                available: output > 0 && tokens <= ceiling,
                reasoning_tokens: Some(tokens),
                minimum_output_tokens: if tokens == 0 { 1 } else { tokens + 512 },
                native_effort: None,
            });
        }
        let named_default = runtime
            .defaults
            .effort
            .as_ref()
            .and_then(Value::as_str)
            .and_then(|id| profile.tiers.iter().find(|tier| tier.id == id))
            .and_then(|tier| tier.reasoning_tokens);
        let default = if runtime.defaults.thinking == Some(false) || configured == Some(0) {
            profile.tiers.first().filter(|tier| tier.available)
        } else {
            profile.tiers.iter().rev().find(|tier| {
                tier.available
                    && named_default.is_none_or(|limit| {
                        tier.reasoning_tokens.is_some_and(|tokens| tokens <= limit)
                    })
            })
        };
        profile.default_tier = default.map(|tier| tier.id.clone());
        profile.default_adjusted_to_limits = runtime.defaults.thinking != Some(false)
            && named_default.or(configured) != default.and_then(|tier| tier.reasoning_tokens);
    } else {
        // Native effort strings do not provide an enforceable token reserve.
        profile.limits.answer_reserve_tokens = 0;
        // Explicit model metadata narrows a backend vocabulary. Never advertise
        // arbitrary strings merely because a transport accepts string values.
        let allowed = manifest
            .metadata
            .parameter_constraints
            .get("reasoning_effort")
            .and_then(|value| value.get("enum"))
            .and_then(Value::as_array)
            .unwrap_or(&runtime.native_efforts);
        for value in allowed {
            let Some(id) = value.as_str().filter(|s| {
                !s.is_empty()
                    && s.len() <= 64
                    && s.bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
            }) else {
                continue;
            };
            if !runtime.native_efforts.contains(value)
                || profile.tiers.iter().any(|tier| tier.id == id)
            {
                continue;
            }
            profile.tiers.push(ReasoningTier {
                id: id.into(),
                label: id.into(),
                available: output > 0,
                reasoning_tokens: None,
                minimum_output_tokens: 1,
                native_effort: Some(value.clone()),
            });
        }
        let off = json!("none");
        let default = if runtime.defaults.thinking == Some(false) {
            Some(&off)
        } else {
            runtime.defaults.effort.as_ref().or_else(|| {
                manifest
                    .metadata
                    .generation_defaults
                    .get("reasoning_effort")
            })
        };
        profile.default_tier = profile
            .tiers
            .iter()
            .find(|tier| tier.native_effort.as_ref() == default && tier.available)
            .map(|tier| tier.id.clone());
    }
    profile.supported = !profile.tiers.is_empty();
    profile
}

impl ReasoningProfile {
    pub fn resolve(&self, requested: &str) -> anyhow::Result<BTreeMap<String, Value>> {
        anyhow::ensure!(
            self.supported,
            "No verified reasoning tiers are available for this model/runtime"
        );
        anyhow::ensure!(
            self.limits.output_tokens > 0,
            "No output budget remains within the context limit"
        );
        let id = if requested == "default" {
            let Some(default) = &self.default_tier else {
                // A native template may own the default without reporting its
                // value. Preserve it; no made-up tier is sent to the backend.
                return Ok(BTreeMap::new());
            };
            default.as_str()
        } else {
            requested
        };
        let tier = self.tiers.iter().find(|tier| tier.id == id && tier.available)
            .ok_or_else(|| anyhow::anyhow!("Reasoning tier '{id}' is not available within this model's configured and output limits"))?;
        Ok(if let Some(tokens) = tier.reasoning_tokens {
            BTreeMap::from([
                ("reasoning_budget_tokens".into(), json!(tokens)),
                ("__werk_thinking".into(), json!(tokens > 0)),
            ])
        } else {
            BTreeMap::from([(
                "reasoning_effort".into(),
                tier.native_effort.clone().unwrap(),
            )])
        })
    }
}
