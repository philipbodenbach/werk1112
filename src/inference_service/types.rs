use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::{
    capabilities::InferenceTask,
    inference::{EffectiveInferenceRequest, ExecutionPlan, ResolvedParameter, WorkloadEstimate},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceResult {
    pub id: String,
    pub task: InferenceTask,
    pub model: String,
    pub runtime: String,
    pub outputs: Vec<OutputMetadata>,
    pub effective_request: EffectiveInferenceRequest,
    pub estimate: WorkloadEstimate,
    pub plan: ExecutionPlan,
    #[serde(default)]
    pub backend_metadata: Value,
    #[serde(default)]
    pub timings: InferenceTimings,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub created_unix: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct InferenceTimings {
    /// Sum of the service phases below. Observer callbacks and result-metadata
    /// serialization are deliberately excluded from this measured duration.
    pub total_seconds: f64,
    pub resolve_seconds: f64,
    pub estimate_seconds: f64,
    pub planning_seconds: f64,
    pub output_setup_seconds: f64,
    pub execution_seconds: f64,
    pub finalization_seconds: f64,
    pub runtime_attempts: Vec<RuntimeAttemptTiming>,
}

impl InferenceTimings {
    pub(crate) fn update_total(&mut self) {
        self.total_seconds = self.resolve_seconds
            + self.estimate_seconds
            + self.planning_seconds
            + self.output_setup_seconds
            + self.execution_seconds
            + self.finalization_seconds;
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RuntimeAttemptTiming {
    pub runtime: String,
    pub duration_seconds: f64,
    pub outcome: RuntimeAttemptOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeAttemptOutcome {
    #[default]
    Unknown,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputMetadata {
    pub id: String,
    pub task: InferenceTask,
    pub model: String,
    pub runtime: String,
    pub path: String,
    pub mime_type: String,
    pub size_bytes: u64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration: Option<f64>,
    pub seed: Option<u64>,
    pub effective_parameters: BTreeMap<String, ResolvedParameter>,
    pub created_unix: u64,
    #[serde(default)]
    pub backend_metadata: Value,
}
