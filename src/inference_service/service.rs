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
    backend::validated_backend_install_command,
    inference::{
        EffectiveInferenceRequest, ExecutionDegradation, ExecutionPlan, HostResources,
        InferenceRequest, ParameterSource, ParameterValue, PlanCandidateStatus, ResolutionContext,
        ResolvedParameter, RuntimeAccelerator, TaskReadiness, TaskReadinessStatus,
        WorkloadEstimate, classify_workload_fit, estimate_workload, parameter_schema,
        parameter_schema_for_manifest, plan_execution, resolve_request,
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
        let mut probe = self
            .backend
            .probe(&self.store, manifest, task, &schema_paths);
        normalize_probe_readiness(&mut probe);
        Ok(probe)
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
        let mut plan = plan_execution(&manifest, &effective, &estimate, &probe.candidates);
        attach_probe_readiness(&mut plan, &probe);
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
        attach_probe_readiness(&mut plan, &probe);
        timings.planning_seconds = planning_started.elapsed().as_secs_f64();

        plan_observer(&effective, &estimate, &plan);

        let selected_runtime = plan.selected_runtime.clone().ok_or_else(|| {
            anyhow!(
                "no executable runtime for {}: {}",
                effective.task,
                format_plan_rejections(&plan, &manifest.id)
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
                "all accepted runtimes failed for {}: {}. Recommendation: inspect this exact model/task with `werk doctor --model {} --task {}`; Werk will not silently switch models",
                effective.task,
                execution_errors.join("; "),
                manifest.id,
                effective.task,
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
                let task_probes = manifest
                    .metadata
                    .tasks
                    .iter()
                    .copied()
                    .map(|task| {
                        let paths = parameter_schema_for_manifest(task, &manifest)
                            .unwrap_or_else(|_| parameter_schema(task))
                            .into_iter()
                            .map(|descriptor| descriptor.path)
                            .collect::<Vec<_>>();
                        let probe = self.backend.probe(&self.store, &manifest, task, &paths);
                        (task, probe)
                    })
                    .collect::<Vec<_>>();
                let available_tasks = task_probes
                    .iter()
                    .filter(|(task, probe)| {
                        probe.candidates.iter().any(|candidate| {
                            candidate.available && candidate.supported_tasks.contains(task)
                        })
                    })
                    .map(|(task, _)| *task)
                    .collect::<Vec<_>>();
                let task_statuses = task_probes
                    .into_iter()
                    .map(|(task, probe)| {
                        let mut readiness = probe.readiness.unwrap_or_else(|| {
                            let available = probe.candidates.iter().any(|candidate| {
                                candidate.available && candidate.supported_tasks.contains(&task)
                            });
                            TaskReadiness {
                                status: if available {
                                    TaskReadinessStatus::Available
                                } else {
                                    TaskReadinessStatus::Unavailable
                                },
                                detail: probe.detail,
                                adapter: None,
                                required_backend: None,
                                install_command: None,
                                fallback_backend: None,
                                missing_dependencies: Vec::new(),
                                missing_dependency_groups: Vec::new(),
                            }
                        });
                        normalize_task_readiness(&mut readiness);
                        readiness.detail = sanitize_capability_detail(&readiness.detail);
                        let task_key = serde_json::to_value(task)
                            .ok()
                            .and_then(|value| value.as_str().map(ToString::to_string))
                            .unwrap_or_else(|| task.to_string());
                        (task_key, readiness)
                    })
                    .collect::<BTreeMap<_, _>>();
                json!({
                    "id": manifest.id,
                    "family": manifest.metadata.family,
                    "layout": manifest.metadata.repository_layout,
                    "tasks": manifest.metadata.tasks,
                    "available_tasks": available_tasks,
                    "task_statuses": task_statuses,
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

fn format_plan_rejections(plan: &ExecutionPlan, model: &str) -> String {
    let rejections = plan
        .candidates
        .iter()
        .map(|candidate| format!("{}: {}", candidate.runtime_id, candidate.reasons.join(", ")))
        .collect::<Vec<_>>()
        .join("; ");
    match plan
        .task_readiness
        .as_ref()
        .and_then(|readiness| task_readiness_recommendation(readiness, model, plan))
    {
        Some(recommendation) => format!("{rejections}. {recommendation}"),
        None => rejections,
    }
}

fn attach_probe_readiness(plan: &mut ExecutionPlan, probe: &BackendProbe) {
    plan.task_readiness = probe.readiness.clone();
    if let Some(readiness) = plan.task_readiness.as_mut() {
        normalize_task_readiness(readiness);
    }
    if plan.selected_runtime.is_some() {
        return;
    }

    let constrained_fallback = plan.candidates.iter().find_map(|decision| {
        let constraints = explicit_routing_constraints(&decision.reasons)?;
        let candidate_was_available = probe
            .candidates
            .iter()
            .any(|candidate| candidate.id == decision.runtime_id && candidate.available);
        candidate_was_available.then_some((decision.runtime_id.clone(), constraints))
    });
    let Some((fallback_runtime, constraints)) = constrained_fallback else {
        return;
    };

    let constraint_label = match constraints {
        ExplicitRoutingConstraints {
            backend: true,
            accelerator: true,
        } => "explicit backend and accelerator/device constraints",
        ExplicitRoutingConstraints { backend: true, .. } => "the explicit backend constraint",
        ExplicitRoutingConstraints {
            accelerator: true, ..
        } => "the explicit accelerator/device constraint",
        _ => return,
    };
    let detail = format!(
        "runtime '{fallback_runtime}' passed its backend probe but was rejected only by {constraint_label}"
    );
    let readiness = plan.task_readiness.get_or_insert_with(|| TaskReadiness {
        status: TaskReadinessStatus::FallbackAvailable,
        detail: detail.clone(),
        adapter: None,
        required_backend: None,
        install_command: None,
        fallback_backend: Some(fallback_runtime.clone()),
        missing_dependencies: Vec::new(),
        missing_dependency_groups: Vec::new(),
    });
    readiness.status = TaskReadinessStatus::FallbackAvailable;
    readiness.detail = detail;
    readiness.install_command = None;
    readiness.fallback_backend = Some(fallback_runtime);
}

fn task_readiness_recommendation(
    readiness: &TaskReadiness,
    model: &str,
    plan: &ExecutionPlan,
) -> Option<String> {
    match readiness.status {
        TaskReadinessStatus::Available => None,
        TaskReadinessStatus::Installable => readiness
            .install_command
            .as_deref()
            .and_then(validated_backend_install_command)
            .map(|command| {
                format!("Recommendation: run `{command}`; no compatible fallback was verified")
            })
            .or_else(|| {
                Some(format!(
                    "Recommendation: inspect this exact model/task with `werk doctor --model {model} --task {}`",
                    plan.task
                ))
            }),
        TaskReadinessStatus::NotImplemented => Some(format!(
            "No Werk execution adapter is implemented for model '{model}' and task {}; installing an unrelated package will not enable it",
            plan.task
        )),
        TaskReadinessStatus::FallbackAvailable => {
            let constraints = readiness
                .fallback_backend
                .as_deref()
                .and_then(|runtime| {
                    plan.candidates
                        .iter()
                        .find(|candidate| candidate.runtime_id == runtime)
                })
                .and_then(|candidate| explicit_routing_constraints(&candidate.reasons));
            let action = match constraints {
                Some(ExplicitRoutingConstraints {
                    backend: true,
                    accelerator: true,
                }) => {
                    "change or remove the explicit accelerator/device constraint and change the explicit backend or allow backend fallback"
                }
                Some(ExplicitRoutingConstraints { backend: true, .. }) => {
                    "change the explicit backend or use a fallback policy that permits backend fallback"
                }
                Some(ExplicitRoutingConstraints {
                    accelerator: true, ..
                }) => "change or remove the explicit accelerator/device constraint",
                _ => "change the explicit routing constraints",
            };
            Some(format!(
                "A verified fallback runtime{} exists but the current routing constraints rejected it; {action}",
                readiness
                    .fallback_backend
                    .as_deref()
                    .map(|backend| format!(" '{backend}'"))
                    .unwrap_or_default()
            ))
        }
        TaskReadinessStatus::Unavailable => Some(format!(
            "Recommendation: inspect this exact model/task with `werk doctor --model {model} --task {}`",
            plan.task
        )),
    }
}

#[derive(Debug, Clone, Copy)]
struct ExplicitRoutingConstraints {
    backend: bool,
    accelerator: bool,
}

fn explicit_routing_constraints(reasons: &[String]) -> Option<ExplicitRoutingConstraints> {
    if reasons.is_empty() {
        return None;
    }
    let mut constraints = ExplicitRoutingConstraints {
        backend: false,
        accelerator: false,
    };
    for reason in reasons {
        if reason.starts_with("explicit backend '") && reason.ends_with("' was requested") {
            constraints.backend = true;
        } else if reason.starts_with("accelerator '") && reason.ends_with("' was requested") {
            constraints.accelerator = true;
        } else {
            return None;
        }
    }
    (constraints.backend || constraints.accelerator).then_some(constraints)
}

fn normalize_probe_readiness(probe: &mut BackendProbe) {
    if let Some(readiness) = probe.readiness.as_mut() {
        normalize_task_readiness(readiness);
    }
}

fn normalize_task_readiness(readiness: &mut TaskReadiness) {
    readiness.install_command = if readiness.status == TaskReadinessStatus::Installable {
        readiness
            .install_command
            .as_deref()
            .and_then(validated_backend_install_command)
    } else {
        None
    };
}

fn sanitize_capability_detail(detail: &str) -> String {
    let without_controls = detail
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    without_controls
        .split_whitespace()
        .map(|part| {
            let candidate = part.trim_matches(|character: char| {
                matches!(
                    character,
                    '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';'
                )
            });
            if candidate.contains("http://") || candidate.contains("https://") {
                "<url>".to_string()
            } else if contains_absolute_path(candidate) {
                "<path>".to_string()
            } else {
                part.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_absolute_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.iter().enumerate().any(|(index, byte)| {
        (*byte == b'/'
            && (index == 0
                || matches!(
                    bytes[index - 1],
                    b'=' | b':' | b'\'' | b'"' | b'(' | b'[' | b'{'
                )))
            || (*byte == b'\\'
                && (index == 0
                    || matches!(
                        bytes[index - 1],
                        b'=' | b':' | b'\'' | b'"' | b'(' | b'[' | b'{'
                    )))
            || (byte.is_ascii_alphabetic()
                && bytes.get(index + 1) == Some(&b':')
                && bytes
                    .get(index + 2)
                    .is_some_and(|separator| matches!(separator, b'/' | b'\\')))
    })
}
