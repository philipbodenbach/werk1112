use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, sync::Arc, time::Instant};

use super::{
    backend::{BackendExecution, BackendProbe, MediaInferenceBackend},
    companion::CompanionMediaBackend,
    helpers::{new_id, validate_safe_name, write_json_atomic},
    output::{OutputStore, output_metadata, remove_output_dir},
    resources::detect_host_resources,
    types::{InferenceResult, InferenceTimings, RuntimeAttemptOutcome, RuntimeAttemptTiming},
};
use crate::{
    inference::{
        EffectiveInferenceRequest, ExecutionDegradation, ExecutionPlan, HostResources,
        InferenceRequest, ParameterSource, ParameterValue, PlanCandidateStatus, ResolutionContext,
        ResolvedParameter, RuntimeAccelerator, WorkloadEstimate, classify_workload_fit,
        estimate_workload, parameter_schema, parameter_schema_for_manifest, plan_execution,
        resolve_request,
    },
    media_companion::CompanionProtocolError,
    model_store::{ModelManifest, ModelStore, unix_ts},
};

#[derive(Clone)]
pub struct InferenceService {
    store: ModelStore,
    backend: Arc<dyn MediaInferenceBackend>,
    outputs: OutputStore,
    resource_detector: Arc<dyn Fn() -> HostResources + Send + Sync>,
}

impl InferenceService {
    pub fn new(store: ModelStore) -> Self {
        let outputs = OutputStore::new(store.home());
        Self {
            store,
            backend: Arc::new(CompanionMediaBackend::discover()),
            outputs,
            resource_detector: Arc::new(detect_host_resources),
        }
    }

    pub fn with_backend(store: ModelStore, backend: Arc<dyn MediaInferenceBackend>) -> Self {
        let outputs = OutputStore::new(store.home());
        Self {
            store,
            backend,
            outputs,
            resource_detector: Arc::new(detect_host_resources),
        }
    }

    #[cfg(test)]
    pub(super) fn with_backend_and_resources(
        store: ModelStore,
        backend: Arc<dyn MediaInferenceBackend>,
        resources: HostResources,
    ) -> Self {
        let outputs = OutputStore::new(store.home());
        Self {
            store,
            backend,
            outputs,
            resource_detector: Arc::new(move || resources.clone()),
        }
    }

    pub fn store(&self) -> &ModelStore {
        &self.store
    }

    pub fn output_store(&self) -> &OutputStore {
        &self.outputs
    }

    pub fn parameter_probe(
        &self,
        manifest: &ModelManifest,
        task: crate::capabilities::InferenceTask,
    ) -> Result<BackendProbe> {
        let schema_paths = parameter_schema_for_manifest(task, manifest)?
            .into_iter()
            .map(|descriptor| descriptor.path)
            .collect::<Vec<_>>();
        Ok(self
            .backend
            .probe(&self.store, manifest, task, &schema_paths))
    }

    pub fn resolve(&self, request: InferenceRequest) -> Result<EffectiveInferenceRequest> {
        let manifest = self.store.get(&request.model)?;
        let schema = parameter_schema_for_manifest(request.task, &manifest)?;
        let schema_paths = schema
            .iter()
            .map(|descriptor| descriptor.path.clone())
            .collect::<Vec<_>>();
        let probe = self
            .backend
            .probe(&self.store, &manifest, request.task, &schema_paths);
        let context = ResolutionContext {
            runtime_defaults: runtime_defaults(&probe),
            hardware_profile: hardware_defaults(&request),
            user_profile: self.load_profile(request.routing.profile.as_deref())?,
            parameter_support: probe.parameter_support,
        };
        resolve_request(&manifest, request, &context)
    }

    pub fn estimate(&self, request: InferenceRequest) -> Result<WorkloadEstimate> {
        let manifest = self.store.get(&request.model)?;
        let effective = self.resolve(request)?;
        self.estimate_effective(&manifest, &effective)
    }

    pub fn plan(
        &self,
        request: InferenceRequest,
    ) -> Result<(EffectiveInferenceRequest, WorkloadEstimate, ExecutionPlan)> {
        let manifest = self.store.get(&request.model)?;
        let mut effective = self.resolve(request)?;
        let estimate = self.estimate_effective(&manifest, &effective)?;
        let schema_paths = parameter_schema_for_manifest(effective.task, &manifest)?
            .into_iter()
            .map(|descriptor| descriptor.path)
            .collect::<Vec<_>>();
        let probe = self
            .backend
            .probe(&self.store, &manifest, effective.task, &schema_paths);
        let plan = plan_execution(&manifest, &effective, &estimate, &probe.candidates);
        apply_plan_adjustments(&mut effective, &plan);
        Ok((effective, estimate, plan))
    }

    fn estimate_effective(
        &self,
        manifest: &ModelManifest,
        effective: &EffectiveInferenceRequest,
    ) -> Result<WorkloadEstimate> {
        let resources = resources_for_request((self.resource_detector)(), effective);
        let fallback = estimate_workload(manifest, effective, &resources);
        match self.backend.estimate(&self.store, manifest, effective) {
            Ok(Some(mut estimate)) => {
                if estimate.task != effective.task {
                    bail!(
                        "media backend estimate task mismatch: expected {}, got {}",
                        effective.task,
                        estimate.task
                    );
                }
                complete_backend_fit(&mut estimate, &resources);
                append_unique(&mut estimate.warnings, fallback.warnings);
                append_unique(&mut estimate.recommendations, fallback.recommendations);
                Ok(estimate)
            }
            Ok(None) => Ok(fallback),
            Err(error)
                if error
                    .downcast_ref::<CompanionProtocolError>()
                    .is_some_and(|error| {
                        matches!(
                            error.code.as_str(),
                            "unsupported_parameter" | "invalid_parameter" | "invalid_request"
                        )
                    }) =>
            {
                Err(error)
            }
            Err(error) => {
                let mut fallback = fallback;
                fallback.warnings.push(format!(
                    "backend estimate unavailable; using Werk heuristic: {error:#}"
                ));
                Ok(fallback)
            }
        }
    }

    pub fn execute(&self, request: InferenceRequest) -> Result<InferenceResult> {
        self.execute_with_observers(request, |_, _, _| {}, |_| {})
    }

    pub fn execute_with_plan_observer<F>(
        &self,
        request: InferenceRequest,
        observer: F,
    ) -> Result<InferenceResult>
    where
        F: FnOnce(&EffectiveInferenceRequest, &WorkloadEstimate, &ExecutionPlan),
    {
        self.execute_with_observers(request, observer, |_| {})
    }

    pub(crate) fn execute_with_observers<P, A>(
        &self,
        request: InferenceRequest,
        plan_observer: P,
        mut attempt_observer: A,
    ) -> Result<InferenceResult>
    where
        P: FnOnce(&EffectiveInferenceRequest, &WorkloadEstimate, &ExecutionPlan),
        A: FnMut(&RuntimeAttemptTiming),
    {
        let mut timings = InferenceTimings::default();

        let resolve_started = Instant::now();
        let manifest = self.store.get(&request.model)?;
        let mut effective = self.resolve(request)?;
        timings.resolve_seconds = resolve_started.elapsed().as_secs_f64();

        let estimate_started = Instant::now();
        let estimate = self.estimate_effective(&manifest, &effective)?;
        timings.estimate_seconds = estimate_started.elapsed().as_secs_f64();

        let planning_started = Instant::now();
        let schema_paths = parameter_schema_for_manifest(effective.task, &manifest)?
            .into_iter()
            .map(|descriptor| descriptor.path)
            .collect::<Vec<_>>();
        let probe = self
            .backend
            .probe(&self.store, &manifest, effective.task, &schema_paths);
        let mut plan = plan_execution(&manifest, &effective, &estimate, &probe.candidates);
        timings.planning_seconds = planning_started.elapsed().as_secs_f64();

        plan_observer(&effective, &estimate, &plan);

        let selected_runtime = plan.selected_runtime.clone().ok_or_else(|| {
            anyhow!(
                "no executable runtime for {}: {}",
                effective.task,
                format_plan_rejections(&plan)
            )
        })?;

        let output_setup_started = Instant::now();
        self.outputs.enforce_retention()?;
        let result_id = new_id("out")?;
        let output_dir = self.outputs.create_output_dir(&result_id)?;
        let fallback_policy = effective
            .string_parameter("routing.fallback_policy")
            .unwrap_or("backend");
        let runtime_attempts = plan
            .candidates
            .iter()
            .filter(|candidate| candidate.status == PlanCandidateStatus::Accepted)
            .filter(|candidate| {
                fallback_policy != "none" || candidate.runtime_id == selected_runtime
            })
            .cloned()
            .collect::<Vec<_>>();
        timings.output_setup_seconds = output_setup_started.elapsed().as_secs_f64();

        let execution_started = Instant::now();
        let mut execution_errors = Vec::new();
        let mut execution = None;
        for (index, candidate) in runtime_attempts.iter().enumerate() {
            if index > 0 {
                remove_output_dir(&self.outputs.root, &output_dir)?;
                fs::create_dir_all(&output_dir)?;
            }
            let mut attempt_effective = effective.clone();
            apply_candidate_adjustments(
                &mut attempt_effective,
                &candidate.runtime_id,
                &candidate.backend,
                &candidate.degradations,
            );
            let attempt_started = Instant::now();
            let attempt = self.backend.execute(
                &self.store,
                &manifest,
                &attempt_effective,
                &output_dir,
                &candidate.runtime_id,
            );
            let attempt_seconds = attempt_started.elapsed().as_secs_f64();
            match attempt {
                Ok(value) => {
                    let attempt_timing = RuntimeAttemptTiming {
                        runtime: candidate.runtime_id.clone(),
                        duration_seconds: attempt_seconds,
                        outcome: RuntimeAttemptOutcome::Succeeded,
                        error: None,
                    };
                    attempt_observer(&attempt_timing);
                    timings.runtime_attempts.push(attempt_timing);
                    if candidate.runtime_id != selected_runtime {
                        plan.selected_runtime = Some(candidate.runtime_id.clone());
                        plan.selected_backend = Some(candidate.backend.clone());
                        plan.score = candidate.score;
                        plan.degradations = candidate.degradations.clone();
                        plan.backend_fallback = true;
                    }
                    effective = attempt_effective;
                    execution = Some(value);
                    break;
                }
                Err(error) => {
                    let error = format!("{error:#}");
                    let attempt_timing = RuntimeAttemptTiming {
                        runtime: candidate.runtime_id.clone(),
                        duration_seconds: attempt_seconds,
                        outcome: RuntimeAttemptOutcome::Failed,
                        error: Some(error.clone()),
                    };
                    attempt_observer(&attempt_timing);
                    timings.runtime_attempts.push(attempt_timing);
                    execution_errors.push(format!("{}: {error}", candidate.runtime_id));
                }
            }
        }
        timings.execution_seconds = execution_started.elapsed().as_secs_f64();
        let Some(execution) = execution else {
            let _ = remove_output_dir(&self.outputs.root, &output_dir);
            bail!(
                "all accepted runtimes failed for {}: {}",
                effective.task,
                execution_errors.join("; ")
            );
        };

        let finalization_started = Instant::now();
        let BackendExecution {
            runtime,
            outputs: backend_outputs,
            warnings: execution_warnings,
            metadata: backend_metadata,
        } = execution;
        let created_unix = unix_ts();
        let seed =
            effective.u64_parameter(&format!("{}.seed", effective.task.parameter_namespace()));
        let outputs = match backend_outputs
            .into_iter()
            .enumerate()
            .map(|(index, output)| {
                output_metadata(
                    &result_id,
                    index,
                    &manifest,
                    &effective,
                    &runtime,
                    output,
                    seed,
                    created_unix,
                    &output_dir,
                )
            })
            .collect::<Result<Vec<_>>>()
        {
            Ok(outputs) => outputs,
            Err(error) => {
                let _ = remove_output_dir(&self.outputs.root, &output_dir);
                return Err(error);
            }
        };
        let mut warnings = effective.warnings.clone();
        warnings.extend(estimate.warnings.clone());
        warnings.extend(execution_warnings);
        if !execution_errors.is_empty() {
            warnings.push(format!(
                "runtime fallback was used after: {}",
                execution_errors.join("; ")
            ));
        }
        timings.finalization_seconds = finalization_started.elapsed().as_secs_f64();
        timings.update_total();
        let mut result = InferenceResult {
            id: result_id,
            task: effective.task,
            model: manifest.id,
            runtime,
            outputs,
            effective_request: effective,
            estimate,
            plan,
            backend_metadata,
            timings,
            warnings,
            created_unix,
        };
        if let Err(error) = write_json_atomic(&output_dir.join("metadata.json"), &result) {
            let _ = remove_output_dir(&self.outputs.root, &output_dir);
            return Err(error);
        }
        if self.outputs.enforce_retention_preserving(&output_dir)? {
            result.warnings.push(format!(
                "current output exceeds the configured {} byte output-store budget; it was preserved while older outputs were removed",
                self.outputs.max_bytes
            ));
            write_json_atomic(&output_dir.join("metadata.json"), &result)?;
        }
        Ok(result)
    }

    pub fn capabilities(&self) -> Result<Value> {
        let models = self
            .store
            .list()?
            .into_iter()
            .map(|manifest| {
                let available_tasks = manifest
                    .metadata
                    .tasks
                    .iter()
                    .copied()
                    .filter(|task| {
                        let paths = parameter_schema_for_manifest(*task, &manifest)
                            .unwrap_or_else(|_| parameter_schema(*task))
                            .into_iter()
                            .map(|descriptor| descriptor.path)
                            .collect::<Vec<_>>();
                        self.backend
                            .probe(&self.store, &manifest, *task, &paths)
                            .candidates
                            .iter()
                            .any(|candidate| {
                                candidate.available && candidate.supported_tasks.contains(task)
                            })
                    })
                    .collect::<Vec<_>>();
                json!({
                    "id": manifest.id,
                    "family": manifest.metadata.family,
                    "layout": manifest.metadata.repository_layout,
                    "tasks": manifest.metadata.tasks,
                    "available_tasks": available_tasks,
                    "input_modalities": manifest.metadata.input_modalities,
                    "output_modalities": manifest.metadata.output_modalities
                })
            })
            .collect::<Vec<_>>();
        let tools = dynamic_tools(&models);
        Ok(json!({
            "object": "werk.capabilities",
            "models": models,
            "tools": tools
        }))
    }

    fn load_profile(&self, profile: Option<&str>) -> Result<BTreeMap<String, ParameterValue>> {
        let Some(profile) = profile else {
            return Ok(BTreeMap::new());
        };
        validate_safe_name(profile)?;
        let path = self
            .store
            .home()
            .join("profiles")
            .join(format!("{profile}.json"));
        if !path.is_file() {
            bail!(
                "saved profile '{}' does not exist at {}",
                profile,
                path.display()
            );
        }
        let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
        let object = value
            .as_object()
            .ok_or_else(|| anyhow!("profile {} must contain a JSON object", path.display()))?;
        object
            .iter()
            .map(|(path, value)| Ok((path.clone(), ParameterValue::from_json(value.clone())?)))
            .collect()
    }
}

pub(super) fn resources_for_request(
    mut resources: HostResources,
    request: &EffectiveInferenceRequest,
) -> HostResources {
    let Some(requested) = request
        .string_parameter("routing.device")
        .map(str::trim)
        .filter(|device| !device.is_empty() && !device.eq_ignore_ascii_case("auto"))
        .or_else(|| {
            request
                .string_parameter("routing.accelerator")
                .map(str::trim)
                .filter(|accelerator| {
                    !accelerator.is_empty() && !accelerator.eq_ignore_ascii_case("auto")
                })
        })
        .map(normalized_accelerator_label)
    else {
        return resources;
    };
    let detected = resources
        .accelerator
        .as_deref()
        .map(normalized_accelerator_label);
    if requested == "cpu" || detected.as_deref() != Some(requested.as_str()) {
        resources.accelerator_memory_bytes = None;
    }
    resources.accelerator = Some(requested);
    resources
}

fn normalized_accelerator_label(accelerator: &str) -> String {
    match accelerator.trim().to_ascii_lowercase().as_str() {
        "hip" => "rocm".to_string(),
        "mps" => "metal".to_string(),
        value => value.to_string(),
    }
}

fn dynamic_tools(models: &[Value]) -> Vec<Value> {
    let mut tasks = models
        .iter()
        .filter_map(|model| model.get("available_tasks").and_then(Value::as_array))
        .flatten()
        .filter_map(Value::as_str)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    tasks.sort();
    tasks.dedup();
    tasks
        .into_iter()
        .map(|task| {
            json!({
                "type": "function",
                "function": {
                    "name": task.replace('-', "_"),
                    "description": format!("Run Werk task {task} using an available local model")
                }
            })
        })
        .collect()
}

fn append_unique(target: &mut Vec<String>, values: Vec<String>) {
    for value in values {
        if !target.contains(&value) {
            target.push(value);
        }
    }
}

pub(super) fn complete_backend_fit(estimate: &mut WorkloadEstimate, resources: &HostResources) {
    if resources.accelerator_memory_bytes.is_some() {
        estimate.accelerator_memory_limit_bytes = resources.accelerator_memory_bytes;
    }
    if resources.host_memory_bytes.is_some() {
        estimate.host_memory_limit_bytes = resources.host_memory_bytes;
    }
    let detected_fit = classify_workload_fit(
        estimate.accelerator_peak_bytes,
        estimate.host_peak_bytes,
        resources,
    );
    if fit_severity(detected_fit) > fit_severity(estimate.fit) {
        estimate.fit = detected_fit;
    }
}

fn fit_severity(fit: crate::inference::FitAssessment) -> u8 {
    match fit {
        crate::inference::FitAssessment::Unknown => 0,
        crate::inference::FitAssessment::Fits => 1,
        crate::inference::FitAssessment::Tight => 2,
        crate::inference::FitAssessment::LikelyOom => 3,
    }
}

fn runtime_defaults(probe: &BackendProbe) -> BTreeMap<String, ParameterValue> {
    let mut defaults = BTreeMap::new();
    if probe
        .candidates
        .iter()
        .filter(|candidate| candidate.available)
        .max_by_key(|candidate| candidate.priority)
        .is_some_and(|candidate| candidate.accelerator == RuntimeAccelerator::Cpu)
    {
        defaults.insert("routing.compile".to_string(), false.into());
    }
    defaults
}

fn hardware_defaults(request: &InferenceRequest) -> BTreeMap<String, ParameterValue> {
    let mut defaults = BTreeMap::new();
    if request
        .routing
        .quality
        .as_deref()
        .is_some_and(|quality| quality == "draft")
    {
        let namespace = request.task.parameter_namespace();
        defaults.insert(format!("{namespace}.steps"), 8_i64.into());
    }
    defaults
}

fn apply_plan_adjustments(request: &mut EffectiveInferenceRequest, plan: &ExecutionPlan) {
    let Some(runtime) = plan.selected_runtime.as_deref() else {
        return;
    };
    let backend = plan.selected_backend.as_deref().unwrap_or("unknown");
    apply_candidate_adjustments(request, runtime, backend, &plan.degradations);
}

fn apply_candidate_adjustments(
    request: &mut EffectiveInferenceRequest,
    runtime: &str,
    backend: &str,
    degradations: &[ExecutionDegradation],
) {
    insert_backend_adjustment(request, "routing.backend", backend.into());
    let accelerator = runtime_accelerator_for_id(runtime);
    if let Some(label) = runtime_accelerator_label(accelerator) {
        insert_backend_adjustment(request, "routing.accelerator", label.into());
    }
    for degradation in degradations {
        match degradation {
            ExecutionDegradation::CpuOffload => {
                insert_backend_adjustment(request, "routing.allow_cpu_offload", true.into());
            }
            ExecutionDegradation::SequentialOffload => {
                insert_backend_adjustment(request, "routing.allow_sequential_offload", true.into())
            }
            ExecutionDegradation::ComponentOffload => {
                insert_backend_adjustment(request, "routing.allow_component_offload", true.into())
            }
            ExecutionDegradation::VaeTiling => {
                insert_backend_adjustment(request, "image.vae_tiling", true.into());
            }
            ExecutionDegradation::TemporalWindowing => {
                insert_backend_adjustment(request, "video.temporal_vae_tiling", true.into());
            }
            ExecutionDegradation::SlowerAttention { backend } => insert_backend_adjustment(
                request,
                "routing.attention_backend",
                backend.clone().into(),
            ),
        }
    }
}

fn insert_backend_adjustment(
    request: &mut EffectiveInferenceRequest,
    path: &str,
    value: ParameterValue,
) {
    request.parameters.insert(
        path.to_string(),
        ResolvedParameter {
            value,
            source: ParameterSource::BackendAdjustment,
        },
    );
}

fn runtime_accelerator_for_id(runtime: &str) -> RuntimeAccelerator {
    let runtime = runtime.to_ascii_lowercase();
    if runtime.ends_with("-cuda") {
        RuntimeAccelerator::Cuda
    } else if runtime.ends_with("-rocm") || runtime.ends_with("-hip") {
        RuntimeAccelerator::Rocm
    } else if runtime.ends_with("-metal") || runtime.ends_with("-mps") {
        RuntimeAccelerator::Mps
    } else if runtime.ends_with("-mlx") || runtime.starts_with("mlx") {
        RuntimeAccelerator::Mlx
    } else if runtime.ends_with("-cpu") {
        RuntimeAccelerator::Cpu
    } else {
        RuntimeAccelerator::Other
    }
}

fn runtime_accelerator_label(accelerator: RuntimeAccelerator) -> Option<&'static str> {
    match accelerator {
        RuntimeAccelerator::Cpu => Some("cpu"),
        RuntimeAccelerator::Cuda => Some("cuda"),
        RuntimeAccelerator::Rocm => Some("rocm"),
        RuntimeAccelerator::Mps => Some("metal"),
        RuntimeAccelerator::Mlx => Some("mlx"),
        RuntimeAccelerator::Other => None,
    }
}

fn format_plan_rejections(plan: &ExecutionPlan) -> String {
    plan.candidates
        .iter()
        .map(|candidate| format!("{}: {}", candidate.runtime_id, candidate.reasons.join(", ")))
        .collect::<Vec<_>>()
        .join("; ")
}
