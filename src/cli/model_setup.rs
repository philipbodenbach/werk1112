//! One setup review for the model's complete supported feature set.
//! Used after a pull and before local execution; never from an HTTP worker.
use super::*;
use crate::model_store::feature_setup::{ProjectorDownload, VisionAssets};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum SetupPolicy {
    Ask,
    Approve,
    #[default]
    Never,
}

#[derive(Debug)]
pub(super) struct SetupPlan {
    model: String,
    runtime: Option<RuntimeId>,
    pub(super) installs: Vec<&'static str>,
    pub(super) configure_vision: bool,
    pub(super) projector: Option<ProjectorDownload>,
    pub(super) issues: Vec<String>,
}

impl SetupPlan {
    pub(super) fn has_actions(&self) -> bool {
        !self.installs.is_empty() || self.configure_vision || self.projector.is_some()
    }

    pub(super) fn describe(&self) -> String {
        let mut lines = vec![format!("Model setup: {}", self.model)];
        if let Some(runtime) = self.runtime {
            lines.push(format!(
                "Backend: {}{}",
                runtime_descriptor(runtime).display_name,
                if !self.installs.is_empty() {
                    " (installation/update needed)"
                } else {
                    " (available)"
                }
            ));
            lines.push("Features: text, streaming and tool calls".into());
        }
        for target in &self.installs {
            lines.push(format!("Install: werk backend install {target}"));
        }
        if let Some(projector) = &self.projector {
            lines.push(format!(
                "Vision: download {} from {}@{} ({:.1} MiB); existing model weights are reused",
                projector.file,
                projector.repo,
                projector.revision,
                projector.size as f64 / 1048576.0
            ));
        } else if self.configure_vision {
            lines.push("Vision: activate and validate existing model components".into());
        }
        for issue in &self.issues {
            lines.push(format!("Unavailable: {issue}"));
        }
        lines.join("\n")
    }
}

pub(super) fn plan(
    store: &ModelStore,
    manifest: &ModelManifest,
    backend: BackendChoice,
    options: SelectionOptions,
    offline: bool,
    include_vision: bool,
) -> Result<SetupPlan> {
    plan_with_service(
        &InferenceService::new(store.clone()),
        manifest,
        backend,
        options,
        offline,
        include_vision,
    )
}

/// Share runtime health probes across all models in one component inventory.
pub(super) fn plan_with_service(
    service: &InferenceService,
    manifest: &ModelManifest,
    backend: BackendChoice,
    options: SelectionOptions,
    offline: bool,
    include_vision: bool,
) -> Result<SetupPlan> {
    let store = service.store();
    let mut plan = SetupPlan {
        model: manifest.id.clone(),
        runtime: None,
        installs: Vec::new(),
        configure_vision: false,
        projector: None,
        issues: Vec::new(),
    };
    // Check every advertised specialist task as well as the chat route.
    for task in &manifest.metadata.tasks {
        if matches!(
            task,
            InferenceTask::TextGeneration | InferenceTask::ImageUnderstanding
        ) {
            continue;
        }
        let probe = service.parameter_probe(manifest, *task)?;
        if !probe.available {
            let target = probe
                .readiness
                .as_ref()
                .filter(|r| r.status == TaskReadinessStatus::Installable)
                .and_then(|r| r.install_command.as_deref())
                .and_then(validated_backend_install_command)
                .and_then(|command| {
                    crate::backend::BACKEND_INSTALL_TARGETS
                        .iter()
                        .copied()
                        .find(|target| command == format!("werk backend install {target}"))
                })
                .and_then(|target| runtime_install_target_for_current_host(Some(target)));
            if let Some(target) = target {
                if !plan.installs.contains(&target) {
                    plan.installs.push(target);
                }
            } else {
                plan.issues.push(format!("{task}: {}", probe.detail));
            }
        }
    }
    if !manifest.supports_task(InferenceTask::TextGeneration)
        && !manifest.supports_task(InferenceTask::ImageUnderstanding)
    {
        return Ok(plan);
    }
    let mut preview = manifest.clone();
    let mut images = include_vision && manifest.supports_task(InferenceTask::ImageUnderstanding);
    if include_vision {
        match store.vision_assets(manifest, offline) {
            Ok(VisionAssets::Local) => {
                images = true;
                plan.configure_vision = manifest
                    .metadata
                    .vision_setup
                    .as_ref()
                    .is_some_and(|s| !s.enabled)
                    || (manifest.format == ModelFormat::Gguf
                        && LlamaServerBackend::validate_image_model(store, manifest).is_err());
            }
            Ok(VisionAssets::Download(projector)) => {
                images = true;
                plan.projector = Some(projector);
            }
            Ok(VisionAssets::Unavailable(reason)) => {
                images = true;
                plan.issues.push(reason);
            }
            Ok(VisionAssets::NotAdvertised) => {}
            Err(error) => {
                plan.issues
                    .push(format!("Model component discovery: {error:#}"));
            }
        }
    }
    if images && !preview.supports_task(InferenceTask::ImageUnderstanding) {
        preview
            .metadata
            .tasks
            .push(InferenceTask::ImageUnderstanding);
        preview.metadata.input_modalities.push(InputModality::Image);
    }
    // Validate existing assets with the proposed activation before installing a runtime.
    if images && plan.projector.is_none() {
        if let Some(setup) = preview.metadata.vision_setup.as_mut() {
            setup.enabled = true;
        }
        let assets = match preview.format {
            ModelFormat::Gguf => LlamaServerBackend::validate_image_model(store, &preview)
                .and_then(|path| store.validate_vision_projector(&preview, &path)),
            _ => crate::backend::vision_setup::validate_processor_assets(store, &preview),
        };
        if let Err(error) = assets {
            plan.issues.push(format!("Vision assets: {error:#}"));
        }
    }
    // Setup checks never install. The single confirmed plan owns all mutations.
    let options = SelectionOptions {
        provision_missing_backends: false,
        ..options
    };
    let requested = requested_backend_for_choice(backend);
    let capabilities = RequestCapabilities::text_with_images(true, images).with_tool_calling(true);
    let candidates = runtime_candidate_ids_for_selection(store, &preview, requested);
    let mut installable = None;
    let mut cuda_available = None;
    let mut reasons = Vec::new();
    for id in candidates {
        if let Some(reason) = runtime_static_rejection(&preview, requested, capabilities, id) {
            reasons.push(reason);
            continue;
        }
        let Some(choice) = runtime_id_to_backend_for_request(id, requested) else {
            continue;
        };
        let mut reason = backend_unavailability_reason(store, choice, &preview, options);
        if reason.is_none() && images {
            reason = match choice {
                BackendChoice::LlamaServer(mode) => LlamaServerBackend::probe_image_runtime(
                    store,
                    &preview,
                    mode,
                    options.fp4_kernel,
                )
                .err()
                .map(|e| format!("{e:#}")),
                _ => None,
            };
        }
        if reason.is_none() {
            plan.runtime = Some(id);
            break;
        }
        let mut target =
            runtime_install_target_for_current_host(runtime_descriptor(id).install_target);
        if id == RuntimeId::LlamaServerCuda
            && options.fp4_kernel != Some(Fp4Kernel::Ggml)
            && LlamaServerBackend::needs_nvfp4_runtime(store, &preview, LlamaCppMode::Cuda)
                .unwrap_or(false)
        {
            target = Some("llama-cuda-nvfp4");
        }
        // An installable package is not evidence that its accelerator exists.
        let host_eligible = match id {
            RuntimeId::LlamaServerCuda | RuntimeId::VllmCuda => {
                *cuda_available.get_or_insert_with(|| {
                    crate::inference_service::devices::Inventory::detect()
                        .is_ok_and(|i| i.devices.iter().any(|d| d.visible_index.is_some()))
                })
            }
            RuntimeId::LlamaServerRocm | RuntimeId::VllmRocm => {
                llama_rocm_auto_probeable(store) || vllm_rocm_auto_probeable()
            }
            RuntimeId::LlamaServerMetal | RuntimeId::Mlx | RuntimeId::MlxVlm | RuntimeId::Omlx => {
                cfg!(all(target_os = "macos", target_arch = "aarch64"))
            }
            // Vulkan hardware detection remains the installer's responsibility when explicitly selected.
            RuntimeId::LlamaServerVulkan => requested != RequestedBackend::Auto,
            _ => true,
        };
        if host_eligible && let Some(target) = target {
            if installable.is_none() {
                installable = Some((id, target));
            }
        }
        if let Some(reason) = reason {
            reasons.push(reason);
        }
    }
    if plan.runtime.is_none() {
        if let Some((id, target)) = installable {
            plan.runtime = Some(id);
            plan.installs.push(target);
        } else {
            plan.issues.push(format!(
                "No runtime provides the required model features: {}",
                reasons.join("; ")
            ));
        }
    }
    // MLX uses different adapters for plain text and images. Full setup must
    // include both, even when the vision adapter already works.
    if images && plan.runtime == Some(RuntimeId::MlxVlm) {
        let text = self::plan(store, manifest, backend, options, offline, false)?;
        for target in text.installs {
            if !plan.installs.contains(&target) {
                plan.installs.push(target);
            }
        }
        plan.issues.extend(text.issues);
    }
    if offline && (!plan.installs.is_empty() || plan.projector.is_some()) {
        plan.issues
            .push("Offline mode cannot install missing runtime or model components".into());
    }
    Ok(plan)
}

fn confirm(policy: SetupPolicy, description: &str) -> Result<bool> {
    crate::ui_eprintln!("{description}");
    match policy {
        SetupPolicy::Approve => Ok(true),
        SetupPolicy::Never => Ok(false),
        SetupPolicy::Ask => {
            if !io::stdin().is_terminal()
                || !crate::terminal::interactive(crate::terminal::Stream::Err)
            {
                crate::ui_eprintln!(
                    "Setup needs confirmation. Run werk setup MODEL in a terminal, or pass --auto-install-backends to approve the displayed components."
                );
                return Ok(false);
            }
            crate::ui_eprintln!(
                "Install/configure all listed components? [y = Set up all / Enter = Later / q = Cancel]"
            );
            let mut answer = String::new();
            io::stdin()
                .read_line(&mut answer)
                .context("Cannot read model setup choice")?;
            match answer.trim().to_ascii_lowercase().as_str() {
                "y" | "yes" | "j" | "ja" => Ok(true),
                "" | "n" | "no" | "nein" => Ok(false),
                _ => bail!("Model setup cancelled; downloaded model files are retained"),
            }
        }
    }
}

pub(super) fn ensure(
    store: &ModelStore,
    id: &str,
    backend: BackendChoice,
    options: SelectionOptions,
    offline: bool,
    include_vision: bool,
) -> Result<ModelManifest> {
    let manifest = store.get(id)?;
    if options.setup_policy == SetupPolicy::Never {
        return Ok(manifest);
    }
    let checking = crate::terminal::Activity::start(format!("Checking model setup {id}"));
    checking.phase("Checking backend, tools and model components");
    let setup = plan(store, &manifest, backend, options, offline, include_vision)?;
    checking.finish();
    if !setup.issues.is_empty() {
        crate::ui_eprintln!("{}", setup.describe());
        bail!("Full model setup is blocked by the listed prerequisites; model files are retained");
    }
    if !setup.has_actions() {
        return Ok(manifest);
    }
    if !confirm(options.setup_policy, &setup.describe())? {
        return Ok(manifest);
    }
    anyhow::ensure!(
        !offline || (setup.installs.is_empty() && setup.projector.is_none()),
        "Offline mode forbids this setup download; model files are retained"
    );
    if let Some(projector) = &setup.projector {
        crate::ui_eprintln!("Installing the verified model-specific vision projector...");
        store.install_projector(id, projector)?;
    } else if setup.configure_vision {
        store.configure_vision(id, true, None)?;
    }
    for target in setup.installs {
        let mut command = Command::new(env::current_exe()?);
        command
            .arg("--model-home")
            .arg(store.home())
            .args(["backend", "install", target]);
        if crate::logging::raw_console() {
            command.env(
                "WERK_LOG_FORMAT",
                if crate::logging::json_console() {
                    "json"
                } else {
                    "text"
                },
            );
        }
        if crate::backend::vision_setup::installer_components(target).is_some() {
            command.arg("--vision");
        }
        anyhow::ensure!(
            command.status()?.success(),
            "Backend installation failed; model files are retained. Retry with werk setup {id}"
        );
    }
    let verifying = crate::terminal::Activity::start(format!("Verifying model setup {id}"));
    let updated = store.get(id)?;
    let verify = plan(store, &updated, backend, options, true, include_vision)?;
    anyhow::ensure!(
        !verify.has_actions() && verify.issues.is_empty(),
        "Model setup is incomplete:\n{}",
        verify.describe()
    );
    verifying.finish();
    crate::ui_eprintln!(
        "Model setup complete: prerequisites for the advertised features are installed. Actual model inference remains untested."
    );
    Ok(updated)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn fixture() -> (tempfile::TempDir, ModelStore, ModelManifest) {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        fs::create_dir(&source).unwrap();
        for (file, architecture) in [("model.gguf", "qwen35"), ("mmproj-F16.gguf", "clip")] {
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3u32.to_le_bytes());
            bytes.extend(0u64.to_le_bytes());
            bytes.extend(2u64.to_le_bytes());
            for (key, value) in [
                ("general.architecture", architecture),
                ("general.name", "test VLM"),
            ] {
                bytes.extend((key.len() as u64).to_le_bytes());
                bytes.extend(key.as_bytes());
                bytes.extend(8u32.to_le_bytes());
                bytes.extend((value.len() as u64).to_le_bytes());
                bytes.extend(value.as_bytes());
            }
            fs::write(source.join(file), bytes).unwrap();
        }
        let store = ModelStore::resolve(Some(temp.path().join("store"))).unwrap();
        let manifest = store.import_path(&source, "vlm").unwrap();
        (temp, store, manifest)
    }

    fn runtime(store: &ModelStore, vision: bool) {
        use std::os::unix::fs::PermissionsExt;
        let executable = managed_backend_dir(store, LlamaCppMode::Cpu).join("llama-server");
        fs::create_dir_all(executable.parent().unwrap()).unwrap();
        fs::write(
            &executable,
            if vision {
                "#!/bin/sh\necho --mmproj\n"
            } else {
                "#!/bin/sh\nexit 0\n"
            },
        )
        .unwrap();
        fs::set_permissions(executable, fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn complete_legacy_model_needs_no_setup_or_manifest_rewrite() {
        let (_temp, store, manifest) = fixture();
        runtime(&store, true);
        let setup = plan(
            &store,
            &manifest,
            BackendChoice::LlamaServer(LlamaCppMode::Cpu),
            SelectionOptions::default(),
            true,
            true,
        )
        .unwrap();
        assert!(!setup.has_actions(), "{setup:?}");
        assert!(setup.issues.is_empty(), "{setup:?}");
    }

    #[test]
    fn disabled_vision_and_runtime_upgrade_share_one_plan() {
        let (_temp, store, _) = fixture();
        runtime(&store, false);
        let manifest = store.configure_vision("vlm", false, None).unwrap();
        let before = fs::read(store.model_dir("vlm").join("manifest.json")).unwrap();
        let setup = plan(
            &store,
            &manifest,
            BackendChoice::LlamaServer(LlamaCppMode::Cpu),
            SelectionOptions::default(),
            false,
            true,
        )
        .unwrap();
        assert!(setup.configure_vision);
        assert_eq!(setup.installs, ["llama-cpu"]);
        assert!(setup.issues.is_empty(), "{setup:?}");
        assert_eq!(
            before,
            fs::read(store.model_dir("vlm").join("manifest.json")).unwrap()
        );
    }

    #[test]
    fn confirmed_local_setup_activates_vision_and_preserves_weights() {
        let (_temp, store, original) = fixture();
        runtime(&store, true);
        store.configure_vision("vlm", false, None).unwrap();
        let updated = ensure(
            &store,
            "vlm",
            BackendChoice::LlamaServer(LlamaCppMode::Cpu),
            SelectionOptions::from_cli(BackendArg::Cpu, true, false),
            true,
            true,
        )
        .unwrap();
        assert!(updated.metadata.vision_setup.unwrap().enabled);
        assert_eq!(
            serde_json::to_value(&original.files).unwrap(),
            serde_json::to_value(&updated.files).unwrap()
        );
    }

    #[test]
    fn removed_companion_is_repaired_from_its_saved_revision() {
        let (_temp, store, manifest) = fixture();
        runtime(&store, true);
        let file = manifest
            .files
            .iter()
            .find(|f| f.path.contains("mmproj"))
            .unwrap();
        fs::remove_file(store.absolute_model_file(&manifest, &file.path)).unwrap();
        let saved = ProjectorDownload {
            repo: "ggml-org/test-GGUF".into(),
            revision: "a".repeat(40),
            file: "mmproj-original-Q8_0.gguf".into(),
            size: 42,
            sha256: "b".repeat(64),
            base_model: Some("test/original".into()),
        };
        fs::write(
            store.model_dir("vlm").join("vision-source.json"),
            serde_json::to_vec(&saved).unwrap(),
        )
        .unwrap();
        let setup = plan(
            &store,
            &manifest,
            BackendChoice::LlamaServer(LlamaCppMode::Cpu),
            SelectionOptions::default(),
            true,
            true,
        )
        .unwrap();
        let download = setup.projector.unwrap();
        assert_eq!(download.file, saved.file);
        assert_eq!(download.revision, saved.revision);
        assert!(setup.issues.iter().any(|issue| issue.contains("Offline")));
    }

    #[test]
    fn missing_local_projector_is_explicit_and_never_silently_downgraded() {
        let (_temp, store, manifest) = fixture();
        runtime(&store, true);
        let file = manifest
            .files
            .iter()
            .find(|f| f.path.contains("mmproj"))
            .unwrap();
        fs::remove_file(store.absolute_model_file(&manifest, &file.path)).unwrap();
        let error = ensure(
            &store,
            "vlm",
            BackendChoice::LlamaServer(LlamaCppMode::Cpu),
            SelectionOptions::from_cli(BackendArg::Cpu, true, false),
            true,
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("prerequisites"));
        assert!(store.get("vlm").is_ok());
    }
}
