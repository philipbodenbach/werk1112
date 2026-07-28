use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use super::{
    backend::{BackendExecution, BackendOutput, BackendProbe, MediaInferenceBackend},
    companion::{
        companion_execution_parameters, companion_inputs, decode_base64,
        default_companion_candidates,
    },
    jobs::{JobStatus, JobStore},
    output::OutputStore,
    service::{InferenceService, complete_backend_fit, resources_for_request},
    types::{InferenceResult, InferenceTimings, RuntimeAttemptOutcome},
};
use crate::{
    capabilities::{InferenceTask, InputModality, OutputModality, RepositoryLayout},
    inference::{
        EffectiveInferenceRequest, EstimateConfidence, FitAssessment, HostResources,
        InferenceRequest, InferenceRuntimeCandidate, ParameterPolicy, ParameterSource,
        ParameterSupportStatus, ParameterValue, ResolutionContext, RoutingOverrides,
        RuntimeAccelerator, WorkloadEstimate, classify_workload_fit, resolve_request,
    },
    model_store::{ModelManifest, ModelStore, unix_ts},
};

#[derive(Clone)]
struct MockMediaBackend;

impl MediaInferenceBackend for MockMediaBackend {
    fn probe(
        &self,
        _store: &ModelStore,
        _manifest: &ModelManifest,
        task: InferenceTask,
        schema_paths: &[String],
    ) -> BackendProbe {
        BackendProbe {
            available: true,
            detail: "mock".to_string(),
            candidates: vec![InferenceRuntimeCandidate {
                id: "mock-cpu".to_string(),
                backend: "mock".to_string(),
                accelerator: RuntimeAccelerator::Cpu,
                available: true,
                availability_reason: None,
                supported_tasks: vec![task],
                supported_layouts: vec![RepositoryLayout::Diffusers],
                supported_formats: Vec::new(),
                supported_families: Vec::new(),
                supported_architectures: Vec::new(),
                parameter_support: schema_paths
                    .iter()
                    .cloned()
                    .map(|path| (path, ParameterSupportStatus::Native))
                    .collect(),
                supports_offloading: true,
                supports_compile: false,
                supports_batching: true,
                priority: 100,
            }],
            parameter_support: schema_paths
                .iter()
                .cloned()
                .map(|path| (path, ParameterSupportStatus::Native))
                .collect(),
        }
    }

    fn execute(
        &self,
        _store: &ModelStore,
        _manifest: &ModelManifest,
        _request: &EffectiveInferenceRequest,
        output_dir: &Path,
        runtime: &str,
    ) -> Result<BackendExecution> {
        let path = output_dir.join("image.png");
        fs::write(&path, b"png fixture")?;
        Ok(BackendExecution {
            runtime: runtime.to_string(),
            outputs: vec![BackendOutput {
                path,
                mime_type: Some("image/png".to_string()),
                width: Some(1024),
                height: Some(1024),
                duration: None,
                metadata: json!({"mock": true}),
            }],
            warnings: Vec::new(),
            metadata: json!({
                "adapter": "mock",
                "device": "cpu",
            }),
        })
    }
}

#[derive(Clone)]
struct PreflightOomBackend {
    execute_calls: Arc<AtomicUsize>,
}

impl MediaInferenceBackend for PreflightOomBackend {
    fn probe(
        &self,
        _store: &ModelStore,
        _manifest: &ModelManifest,
        task: InferenceTask,
        schema_paths: &[String],
    ) -> BackendProbe {
        let parameter_support = schema_paths
            .iter()
            .cloned()
            .map(|path| (path, ParameterSupportStatus::Native))
            .collect::<BTreeMap<_, _>>();
        BackendProbe {
            available: true,
            detail: "preflight OOM fixture".to_string(),
            candidates: vec![InferenceRuntimeCandidate {
                id: "mock-cuda".to_string(),
                backend: "mock".to_string(),
                accelerator: RuntimeAccelerator::Cuda,
                available: true,
                availability_reason: None,
                supported_tasks: vec![task],
                supported_layouts: vec![RepositoryLayout::Diffusers],
                supported_formats: Vec::new(),
                supported_families: Vec::new(),
                supported_architectures: Vec::new(),
                parameter_support: parameter_support.clone(),
                supports_offloading: true,
                supports_compile: false,
                supports_batching: true,
                priority: 100,
            }],
            parameter_support,
        }
    }

    fn estimate(
        &self,
        _store: &ModelStore,
        _manifest: &ModelManifest,
        request: &EffectiveInferenceRequest,
    ) -> Result<Option<WorkloadEstimate>> {
        Ok(Some(WorkloadEstimate {
            task: request.task,
            download_size_bytes: Some(1),
            weight_payload_bytes: Some(1),
            accelerator_peak_bytes: Some(20 * 1024 * 1024 * 1024),
            accelerator_memory_limit_bytes: Some(12 * 1024 * 1024 * 1024),
            host_peak_bytes: Some(1),
            host_memory_limit_bytes: None,
            output_size_bytes: Some(1),
            fit: FitAssessment::Fits,
            confidence: EstimateConfidence::BackendMeasured,
            assumptions: Vec::new(),
            warnings: Vec::new(),
            recommendations: Vec::new(),
        }))
    }

    fn execute(
        &self,
        _store: &ModelStore,
        _manifest: &ModelManifest,
        _request: &EffectiveInferenceRequest,
        _output_dir: &Path,
        _runtime: &str,
    ) -> Result<BackendExecution> {
        self.execute_calls.fetch_add(1, Ordering::SeqCst);
        bail!("backend execute must not run after preflight rejection")
    }
}

#[derive(Clone)]
struct FallbackMediaBackend;

impl MediaInferenceBackend for FallbackMediaBackend {
    fn probe(
        &self,
        _store: &ModelStore,
        _manifest: &ModelManifest,
        task: InferenceTask,
        schema_paths: &[String],
    ) -> BackendProbe {
        let support = schema_paths
            .iter()
            .cloned()
            .map(|path| (path, ParameterSupportStatus::Native))
            .collect::<BTreeMap<_, _>>();
        let candidate = |id: &str, accelerator, priority| InferenceRuntimeCandidate {
            id: id.to_string(),
            backend: "mock".to_string(),
            accelerator,
            available: true,
            availability_reason: None,
            supported_tasks: vec![task],
            supported_layouts: vec![RepositoryLayout::Diffusers],
            supported_formats: Vec::new(),
            supported_families: Vec::new(),
            supported_architectures: Vec::new(),
            parameter_support: support.clone(),
            supports_offloading: true,
            supports_compile: false,
            supports_batching: true,
            priority,
        };
        BackendProbe {
            available: true,
            detail: "fallback mock".to_string(),
            candidates: vec![
                candidate("mock-cuda", RuntimeAccelerator::Cuda, 1_000),
                candidate("mock-cpu", RuntimeAccelerator::Cpu, 500),
            ],
            parameter_support: support,
        }
    }

    fn execute(
        &self,
        _store: &ModelStore,
        _manifest: &ModelManifest,
        _request: &EffectiveInferenceRequest,
        output_dir: &Path,
        runtime: &str,
    ) -> Result<BackendExecution> {
        if runtime == "mock-cuda" {
            fs::write(output_dir.join("partial.bin"), b"partial")?;
            bail!("simulated accelerator load failure");
        }
        let path = output_dir.join("image.png");
        fs::write(&path, b"fallback fixture")?;
        Ok(BackendExecution {
            runtime: runtime.to_string(),
            outputs: vec![BackendOutput {
                path,
                mime_type: Some("image/png".to_string()),
                width: Some(512),
                height: Some(512),
                duration: None,
                metadata: Value::Null,
            }],
            warnings: Vec::new(),
            metadata: json!({
                "adapter": "mock",
                "device": "cpu",
            }),
        })
    }
}

fn test_home(name: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "werk-inference-service-{name}-{}-{}",
        std::process::id(),
        unix_ts()
    ))
}

fn image_store(name: &str) -> (PathBuf, ModelStore) {
    let root = test_home(name);
    if root.exists() {
        fs::remove_dir_all(&root).unwrap();
    }
    let source = root.join("source");
    fs::create_dir_all(source.join("transformer")).unwrap();
    fs::write(
        source.join("model_index.json"),
        br#"{"_class_name":"FluxPipeline"}"#,
    )
    .unwrap();
    fs::write(
        source.join("transformer/config.json"),
        br#"{"_class_name":"FluxTransformer2DModel"}"#,
    )
    .unwrap();
    fs::write(source.join("transformer/model.safetensors"), b"weights").unwrap();
    let store = ModelStore::resolve(Some(root.join("store"))).unwrap();
    let manifest = store.import_path(&source, "flux").unwrap();
    assert!(manifest.supports_task(InferenceTask::ImageGeneration));
    assert!(
        manifest
            .metadata
            .input_modalities
            .contains(&InputModality::Text)
    );
    assert!(
        manifest
            .metadata
            .output_modalities
            .contains(&OutputModality::Image)
    );
    (root, store)
}

#[test]
fn service_resolves_plans_executes_and_persists_output_metadata() {
    let (root, store) = image_store("execute");
    let service = InferenceService::with_backend(store, Arc::new(MockMediaBackend));
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("an orbital station".to_string());
    request
        .parameters
        .insert("width".to_string(), 1024_i64.into());
    let result = service.execute(request).unwrap();
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.runtime, "mock-cpu");
    assert_eq!(result.backend_metadata["adapter"], "mock");
    assert_eq!(result.backend_metadata["device"], "cpu");
    assert!(result.timings.total_seconds.is_finite());
    assert_eq!(result.timings.runtime_attempts.len(), 1);
    assert_eq!(
        result.timings.runtime_attempts[0].outcome,
        RuntimeAttemptOutcome::Succeeded
    );
    assert!(result.timings.runtime_attempts[0].error.is_none());
    assert!(result.timings.total_seconds >= result.timings.runtime_attempts[0].duration_seconds);
    assert_eq!(result.outputs[0].mime_type, "image/png");
    assert!(Path::new(&result.outputs[0].path).is_file());
    let metadata_path = service
        .output_store()
        .root()
        .join(&result.id)
        .join("metadata.json");
    let persisted: InferenceResult =
        serde_json::from_slice(&fs::read(&metadata_path).unwrap()).unwrap();
    assert_eq!(persisted.backend_metadata, result.backend_metadata);
    assert_eq!(
        persisted.timings.runtime_attempts,
        result.timings.runtime_attempts
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn preflight_vram_rejection_never_calls_execute_or_creates_an_attempt() {
    let (root, store) = image_store("preflight-vram");
    let execute_calls = Arc::new(AtomicUsize::new(0));
    let service = InferenceService::with_backend_and_resources(
        store,
        Arc::new(PreflightOomBackend {
            execute_calls: Arc::clone(&execute_calls),
        }),
        HostResources {
            host_memory_bytes: Some(64 * 1024 * 1024 * 1024),
            accelerator_memory_bytes: Some(12 * 1024 * 1024 * 1024),
            accelerator: Some("cuda".to_string()),
        },
    );
    let output_entries_before = fs::read_dir(service.output_store().root())
        .map(|entries| entries.count())
        .unwrap_or(0);
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("preflight rejection".to_string());
    request.routing.allow_cpu_offload = crate::inference::OverrideBool::Disabled;
    let plan_observed = Cell::new(false);
    let attempts = RefCell::new(Vec::new());

    let error = service
        .execute_with_observers(
            request,
            |_, estimate, plan| {
                plan_observed.set(true);
                assert_eq!(
                    estimate.accelerator_memory_limit_bytes,
                    Some(12 * 1024 * 1024 * 1024)
                );
                assert_eq!(plan.selected_runtime, None);
            },
            |attempt| attempts.borrow_mut().push(attempt.clone()),
        )
        .unwrap_err();

    assert!(plan_observed.get());
    assert_eq!(execute_calls.load(Ordering::SeqCst), 0);
    assert!(attempts.borrow().is_empty());
    assert!(format!("{error:#}").contains("20.00 GiB"));
    assert_eq!(
        fs::read_dir(service.output_store().root())
            .map(|entries| entries.count())
            .unwrap_or(0),
        output_entries_before
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn execution_retries_accepted_runtime_and_records_backend_adjustment() {
    let (root, store) = image_store("fallback");
    let service = InferenceService::with_backend(store, Arc::new(FallbackMediaBackend));
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("fallback test".to_string());

    let result = service.execute(request).unwrap();
    assert_eq!(result.runtime, "mock-cpu");
    assert_eq!(result.plan.selected_runtime.as_deref(), Some("mock-cpu"));
    assert!(result.plan.backend_fallback);
    assert_eq!(result.timings.runtime_attempts.len(), 2);
    assert_eq!(
        result.timings.runtime_attempts[0].outcome,
        RuntimeAttemptOutcome::Failed
    );
    assert_eq!(result.timings.runtime_attempts[0].runtime, "mock-cuda");
    assert!(
        result.timings.runtime_attempts[0]
            .error
            .as_deref()
            .is_some_and(|error| error.contains("simulated accelerator load failure"))
    );
    assert_eq!(
        result.timings.runtime_attempts[1].outcome,
        RuntimeAttemptOutcome::Succeeded
    );
    assert_eq!(result.timings.runtime_attempts[1].runtime, "mock-cpu");
    assert!(result.timings.runtime_attempts[1].error.is_none());
    assert_eq!(result.backend_metadata["device"], "cpu");
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.contains("mock-cuda"))
    );
    assert_eq!(
        result.effective_request.parameters["routing.accelerator"].source,
        ParameterSource::BackendAdjustment
    );
    assert_eq!(
        result
            .effective_request
            .string_parameter("routing.accelerator"),
        Some("cpu")
    );
    assert!(
        !service
            .output_store()
            .root()
            .join(&result.id)
            .join("partial.bin")
            .exists()
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn plan_and_attempt_observers_report_a_later_backend_failure() {
    let (root, store) = image_store("plan-observer");
    let service = InferenceService::with_backend(store, Arc::new(FallbackMediaBackend));
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("observe failed execution".to_string());
    request.routing.fallback_policy = Some("none".to_string());
    let calls = Cell::new(0);
    let attempts = RefCell::new(Vec::new());

    let result = service.execute_with_observers(
        request,
        |effective, estimate, plan| {
            calls.set(calls.get() + 1);
            assert_eq!(effective.model, "flux");
            assert_eq!(estimate.task, InferenceTask::ImageGeneration);
            assert_eq!(plan.selected_runtime.as_deref(), Some("mock-cuda"));
        },
        |attempt| attempts.borrow_mut().push(attempt.clone()),
    );

    assert!(result.is_err());
    assert_eq!(calls.get(), 1);
    let attempts = attempts.borrow();
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].runtime, "mock-cuda");
    assert_eq!(attempts[0].outcome, RuntimeAttemptOutcome::Failed);
    assert!(attempts[0].duration_seconds.is_finite());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn inference_result_diagnostics_are_backward_compatible_with_legacy_metadata() {
    let (root, store) = image_store("legacy-result");
    let service = InferenceService::with_backend(store, Arc::new(MockMediaBackend));
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("legacy metadata".to_string());
    let result = service.execute(request).unwrap();
    let mut value = serde_json::to_value(result).unwrap();
    let fields = value.as_object_mut().unwrap();
    fields.remove("backend_metadata");
    fields.remove("timings");

    let legacy: InferenceResult = serde_json::from_value(value).unwrap();
    assert_eq!(legacy.backend_metadata, Value::Null);
    assert_eq!(legacy.timings, Default::default());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn inference_timing_types_accept_partial_serialized_records() {
    let timings: InferenceTimings = serde_json::from_value(json!({
        "runtime_attempts": [{
            "runtime": "legacy-runtime"
        }]
    }))
    .unwrap();

    assert_eq!(timings.total_seconds, 0.0);
    assert_eq!(timings.runtime_attempts.len(), 1);
    assert_eq!(timings.runtime_attempts[0].duration_seconds, 0.0);
    assert_eq!(
        timings.runtime_attempts[0].outcome,
        RuntimeAttemptOutcome::Unknown
    );
    assert!(timings.runtime_attempts[0].error.is_none());
}

#[test]
fn inline_base64_inputs_are_staged_for_the_offline_companion() {
    let (root, store) = image_store("base64-model");
    let staging = root.join("staged");
    let mut manifest = store.get("flux").unwrap();
    manifest.metadata.tasks.push(InferenceTask::ImageEditing);
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageEditing);
    request.prompt = Some("edit".to_string());
    request.inputs.push(crate::inference::InferenceInput {
        modality: InputModality::Image,
        role: "image".to_string(),
        source: crate::inference::InferenceInputSource::Base64 {
            data: "AAEC".to_string(),
        },
        mime_type: Some("image/png".to_string()),
    });
    let effective = resolve_request(&manifest, request, &ResolutionContext::default()).unwrap();
    let inputs = companion_inputs(&effective, &staging).unwrap();
    let path = PathBuf::from(inputs["image"].as_str().unwrap());
    assert_eq!(fs::read(path).unwrap(), vec![0, 1, 2]);
    assert!(decode_base64("%%%").is_err());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn companion_executes_only_planner_selected_gpu_offload() {
    let (root, store) = image_store("offload-parameters");
    let manifest = store.get("flux").unwrap();
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("test".to_string());
    request.routing.allow_cpu_offload = crate::inference::OverrideBool::Enabled;
    let mut effective = resolve_request(&manifest, request, &ResolutionContext::default()).unwrap();

    let permitted = companion_execution_parameters(&effective, "media-companion-cuda");
    assert_eq!(
        permitted["_werk_enable_cpu_offload"],
        ParameterValue::Boolean(false)
    );

    effective
        .parameters
        .get_mut("routing.allow_cpu_offload")
        .unwrap()
        .source = ParameterSource::BackendAdjustment;
    let selected = companion_execution_parameters(&effective, "media-companion-cuda");
    assert_eq!(
        selected["_werk_enable_cpu_offload"],
        ParameterValue::Boolean(true)
    );
    let cpu_fallback = companion_execution_parameters(&effective, "media-companion-cpu");
    assert_eq!(
        cpu_fallback["_werk_enable_cpu_offload"],
        ParameterValue::Boolean(false)
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn backend_fit_is_derived_from_its_own_peaks() {
    let resources = HostResources {
        host_memory_bytes: Some(100),
        accelerator_memory_bytes: Some(100),
        accelerator: Some("cuda".to_string()),
    };
    let mut backend = WorkloadEstimate {
        task: InferenceTask::ImageGeneration,
        download_size_bytes: Some(10),
        weight_payload_bytes: Some(10),
        accelerator_peak_bytes: Some(150),
        accelerator_memory_limit_bytes: None,
        host_peak_bytes: Some(20),
        host_memory_limit_bytes: None,
        output_size_bytes: Some(1),
        fit: FitAssessment::Unknown,
        confidence: EstimateConfidence::Heuristic,
        assumptions: Vec::new(),
        warnings: Vec::new(),
        recommendations: Vec::new(),
    };
    let fallback_fit = classify_workload_fit(Some(20), Some(20), &resources);
    assert_eq!(fallback_fit, FitAssessment::Fits);

    complete_backend_fit(&mut backend, &resources);
    assert_eq!(backend.fit, FitAssessment::LikelyOom);
    assert_eq!(backend.accelerator_memory_limit_bytes, Some(100));
    assert_eq!(backend.host_memory_limit_bytes, Some(100));
}

#[test]
fn detected_accelerator_limit_overrides_optimistic_backend_fit() {
    let resources = HostResources {
        host_memory_bytes: Some(1_000),
        accelerator_memory_bytes: Some(100),
        accelerator: Some("cuda".to_string()),
    };
    let mut backend = WorkloadEstimate {
        task: InferenceTask::ImageGeneration,
        download_size_bytes: Some(10),
        weight_payload_bytes: Some(10),
        accelerator_peak_bytes: Some(101),
        accelerator_memory_limit_bytes: None,
        host_peak_bytes: Some(20),
        host_memory_limit_bytes: None,
        output_size_bytes: Some(1),
        fit: FitAssessment::Fits,
        confidence: EstimateConfidence::BackendMeasured,
        assumptions: Vec::new(),
        warnings: Vec::new(),
        recommendations: Vec::new(),
    };

    complete_backend_fit(&mut backend, &resources);

    assert_eq!(backend.fit, FitAssessment::LikelyOom);
    assert_eq!(backend.accelerator_memory_limit_bytes, Some(100));
    assert_eq!(backend.host_memory_limit_bytes, Some(1_000));
}

#[test]
fn explicit_cpu_route_does_not_inherit_cuda_memory_limit() {
    let (root, store) = image_store("cpu-resource-scope");
    let manifest = store.get("flux").unwrap();
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("CPU route".to_string());
    request.routing.accelerator = Some("cpu".to_string());
    let effective = resolve_request(&manifest, request, &ResolutionContext::default()).unwrap();
    let resources = resources_for_request(
        HostResources {
            host_memory_bytes: Some(64 * 1024 * 1024 * 1024),
            accelerator_memory_bytes: Some(24 * 1024 * 1024 * 1024),
            accelerator: Some("cuda".to_string()),
        },
        &effective,
    );

    assert_eq!(resources.accelerator.as_deref(), Some("cpu"));
    assert_eq!(resources.accelerator_memory_bytes, None);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn explicit_cpu_device_takes_precedence_over_cuda_accelerator_for_resources() {
    let (root, store) = image_store("cpu-device-resource-scope");
    let manifest = store.get("flux").unwrap();
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("CPU device route".to_string());
    request.routing.accelerator = Some("cuda".to_string());
    request.routing.device = Some("cpu".to_string());
    let effective = resolve_request(&manifest, request, &ResolutionContext::default()).unwrap();
    let resources = resources_for_request(
        HostResources {
            host_memory_bytes: Some(64 * 1024 * 1024 * 1024),
            accelerator_memory_bytes: Some(24 * 1024 * 1024 * 1024),
            accelerator: Some("cuda".to_string()),
        },
        &effective,
    );

    assert_eq!(resources.accelerator.as_deref(), Some("cpu"));
    assert_eq!(resources.accelerator_memory_bytes, None);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn job_store_enforces_state_machine_and_persists_cancellation() {
    let root = test_home("jobs");
    let store = JobStore::new(&root);
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("test".to_string());
    request.routing = RoutingOverrides {
        parameter_policy: ParameterPolicy::Strict,
        ..Default::default()
    };
    let job = store.create(request).unwrap();
    assert_eq!(job.status, JobStatus::Queued);
    let loading = store
        .transition(&job.id, JobStatus::Loading, None, None)
        .unwrap();
    assert_eq!(loading.status, JobStatus::Loading);
    let cancelled = store.cancel(&job.id).unwrap();
    assert_eq!(cancelled.status, JobStatus::Cancelled);
    assert!(
        store
            .transition(&job.id, JobStatus::Running, None, None)
            .is_err()
    );
    assert_eq!(store.get(&job.id).unwrap().status, JobStatus::Cancelled);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn job_store_recovers_nonterminal_records_after_restart() {
    let root = test_home("job-recovery");
    let store = JobStore::new(&root);
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("test".to_string());
    let job = store.create(request).unwrap();
    store
        .transition(&job.id, JobStatus::Loading, None, None)
        .unwrap();
    store
        .transition(&job.id, JobStatus::Running, None, None)
        .unwrap();

    let restarted = JobStore::new(&root);
    restarted.recover_interrupted().unwrap();
    let record = restarted.get(&job.id).unwrap();
    assert_eq!(record.status, JobStatus::Failed);
    assert!(
        record
            .error
            .as_deref()
            .is_some_and(|error| error.contains("restart"))
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn companion_candidates_derive_capabilities_and_parameter_support_from_registry() {
    let schema_paths = vec![
        "image.width".to_string(),
        "image.output_path".to_string(),
        "routing.allow_disk_offload".to_string(),
    ];
    let candidates = default_companion_candidates(false, Some("test".to_string()), &schema_paths);

    assert_eq!(candidates.len(), 4);
    assert!(candidates.iter().all(|candidate| !candidate.available));
    assert!(candidates.iter().any(|candidate| {
        candidate.id == "media-companion-cpu" && candidate.accelerator == RuntimeAccelerator::Cpu
    }));
    for candidate in candidates {
        assert_eq!(
            candidate.parameter_support["image.width"],
            ParameterSupportStatus::Translated
        );
        assert_eq!(
            candidate.parameter_support["image.output_path"],
            ParameterSupportStatus::Unsupported
        );
        assert_eq!(
            candidate.parameter_support["routing.allow_disk_offload"],
            ParameterSupportStatus::Unsupported
        );
        assert!(
            !candidate
                .supported_tasks
                .contains(&InferenceTask::StemSeparation)
        );
        if candidate.accelerator == RuntimeAccelerator::Cpu {
            assert!(!candidate.supports_offloading);
        }
    }
}

#[test]
fn output_retention_only_removes_output_children() {
    let root = test_home("retention");
    let outputs = OutputStore::with_limits(&root, 1, u64::MAX);
    let first = outputs.create_output_dir("out-old").unwrap();
    fs::write(first.join("data.bin"), vec![0_u8; 32]).unwrap();
    let direct_cli_output = outputs
        .root()
        .join("tiny-sd-image-generation-1784968751-807fe1a8.png");
    fs::write(&direct_cli_output, vec![0_u8; 32]).unwrap();
    let models = root.join("models");
    fs::create_dir_all(&models).unwrap();
    fs::write(models.join("keep"), b"model").unwrap();
    outputs.enforce_retention().unwrap();
    assert!(!first.exists());
    assert!(!direct_cli_output.exists());
    assert!(models.join("keep").is_file());
    let _ = fs::remove_dir_all(root);
}
