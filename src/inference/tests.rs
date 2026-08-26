use super::*;
use crate::{
    capabilities::{
        InferenceTask, InputModality, ModelComponent, OutputModality, RepositoryLayout,
    },
    media_cli::{AudioCommands, ImageCommands, VideoCommands},
    model_store::{
        CURRENT_MANIFEST_SCHEMA_VERSION, ModelFile, ModelFormat, ModelManifest, ModelMetadata,
        ModelSource,
    },
};
use clap::CommandFactory;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(clap::Parser)]
struct ImageSchemaCli {
    #[command(subcommand)]
    command: ImageCommands,
}

#[derive(clap::Parser)]
struct VideoSchemaCli {
    #[command(subcommand)]
    command: VideoCommands,
}

#[derive(clap::Parser)]
struct AudioSchemaCli {
    #[command(subcommand)]
    command: AudioCommands,
}

fn assert_schema_flags_exist_on_subcommand<C: CommandFactory>(
    task: InferenceTask,
    subcommand: &str,
) {
    let command = C::command();
    let public_flags = command
        .find_subcommand(subcommand)
        .unwrap_or_else(|| panic!("missing public subcommand {subcommand}"))
        .get_arguments()
        .filter_map(|argument| argument.get_long())
        .map(|flag| format!("--{flag}"))
        .collect::<BTreeSet<_>>();
    let missing = parameter_schema(task)
        .into_iter()
        .filter(|descriptor| !descriptor.path.starts_with("routing."))
        .filter(|descriptor| !public_flags.contains(&descriptor.cli_flag))
        .map(|descriptor| (descriptor.path, descriptor.cli_flag))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "{task} advertises flags absent from '{subcommand}': {missing:?}"
    );
}

fn assert_schema_flags_exist_on_nested_subcommand<C: CommandFactory>(
    task: InferenceTask,
    parent: &str,
    subcommand: &str,
) {
    let command = C::command();
    let parent_command = command
        .find_subcommand(parent)
        .unwrap_or_else(|| panic!("missing public subcommand {parent}"));
    let public_flags = parent_command
        .find_subcommand(subcommand)
        .unwrap_or_else(|| panic!("missing public subcommand {parent} {subcommand}"))
        .get_arguments()
        .filter_map(|argument| argument.get_long())
        .map(|flag| format!("--{flag}"))
        .collect::<BTreeSet<_>>();
    let missing = parameter_schema(task)
        .into_iter()
        .filter(|descriptor| !descriptor.path.starts_with("routing."))
        .filter(|descriptor| !public_flags.contains(&descriptor.cli_flag))
        .map(|descriptor| (descriptor.path, descriptor.cli_flag))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "{task} advertises flags absent from '{parent} {subcommand}': {missing:?}"
    );
}

fn image_manifest() -> ModelManifest {
    let mut metadata = ModelMetadata {
        schema_version: CURRENT_MANIFEST_SCHEMA_VERSION,
        family: Some("flux".to_string()),
        repository_layout: RepositoryLayout::Diffusers,
        tasks: vec![InferenceTask::ImageGeneration, InferenceTask::ImageEditing],
        input_modalities: vec![InputModality::Text, InputModality::Image],
        output_modalities: vec![OutputModality::Image],
        components: vec![ModelComponent::new(
            crate::capabilities::ModelComponentKind::Transformer,
            "files/transformer",
        )],
        ..Default::default()
    };
    metadata
        .generation_defaults
        .insert("image.steps".to_string(), Value::from(35));
    ModelManifest {
        id: "flux".to_string(),
        source: ModelSource::LocalPath {
            path: "fixture".to_string(),
        },
        format: ModelFormat::SafeTensors,
        architecture: Some("FluxTransformer2DModel".to_string()),
        tokenizer_path: None,
        config_path: None,
        model_path: Some("files/transformer/model.safetensors".to_string()),
        backend: "media-companion".to_string(),
        created_unix: 1,
        files: vec![ModelFile {
            path: "files/transformer/model.safetensors".to_string(),
            size: 1_000_000_000,
            checksum: "crc32:0".to_string(),
        }],
        artifacts: Vec::new(),
        metadata,
    }
}

fn video_manifest(family: &str) -> ModelManifest {
    ModelManifest {
        id: family.to_string(),
        source: ModelSource::LocalPath {
            path: "fixture".to_string(),
        },
        format: ModelFormat::SafeTensors,
        architecture: Some("WanTransformer3DModel".to_string()),
        tokenizer_path: None,
        config_path: None,
        model_path: Some("files/transformer/model.safetensors".to_string()),
        backend: "media-companion".to_string(),
        created_unix: 1,
        files: vec![ModelFile {
            path: "files/transformer/model.safetensors".to_string(),
            size: 1_000_000_000,
            checksum: "crc32:0".to_string(),
        }],
        artifacts: Vec::new(),
        metadata: ModelMetadata {
            schema_version: CURRENT_MANIFEST_SCHEMA_VERSION,
            family: Some(family.to_string()),
            repository_layout: RepositoryLayout::Diffusers,
            tasks: vec![InferenceTask::VideoGeneration],
            input_modalities: vec![InputModality::Text],
            output_modalities: vec![OutputModality::Video],
            components: vec![ModelComponent::new(
                crate::capabilities::ModelComponentKind::Transformer,
                "files/transformer",
            )],
            ..Default::default()
        },
    }
}

fn request() -> InferenceRequest {
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("an orbital station".to_string());
    request
}

#[test]
fn boolean_override_has_three_states() {
    assert!(OverrideBool::Inherit.resolve(true));
    assert!(!OverrideBool::Inherit.resolve(false));
    assert!(OverrideBool::Enabled.resolve(false));
    assert!(!OverrideBool::Disabled.resolve(true));
    assert_eq!(OverrideBool::Inherit.explicit(), None);
}

#[test]
fn list_override_distinguishes_inherit_replace_add_and_clear() {
    let inherited = vec!["base"];
    assert_eq!(
        ListOverride::<&str>::Inherit.resolve(&inherited),
        vec!["base"]
    );
    assert_eq!(
        ListOverride::Replace(vec!["new"]).resolve(&inherited),
        vec!["new"]
    );
    assert_eq!(
        ListOverride::Add(vec!["extra"]).resolve(&inherited),
        vec!["base", "extra"]
    );
    assert!(ListOverride::<&str>::Clear.resolve(&inherited).is_empty());
}

#[test]
fn list_override_operations_apply_end_to_end_with_provenance() {
    let mut manifest = image_manifest();
    manifest
        .metadata
        .generation_defaults
        .insert("image.loras".to_string(), json!(["base.safetensors"]));
    let mut add = request();
    add.parameters.insert(
        "loras".to_string(),
        ParameterValue::from_json(json!({
            "operation": "add",
            "values": ["style.safetensors"]
        }))
        .unwrap(),
    );
    let effective = resolve_request(&manifest, add, &ResolutionContext::default()).unwrap();
    assert_eq!(
        effective.parameter("image.loras"),
        Some(&ParameterValue::List(vec![
            "base.safetensors".into(),
            "style.safetensors".into()
        ]))
    );
    assert_eq!(
        effective.parameters["image.loras"].source,
        ParameterSource::RequestOverride
    );

    let mut clear = request();
    clear.parameters.insert(
        "image.loras".to_string(),
        ParameterValue::from_json(json!({"operation": "clear"})).unwrap(),
    );
    let effective = resolve_request(&manifest, clear, &ResolutionContext::default()).unwrap();
    assert_eq!(
        effective.parameter("image.loras"),
        Some(&ParameterValue::List(Vec::new()))
    );
}

#[test]
fn defaults_follow_specificity_and_track_provenance() {
    let manifest = image_manifest();
    let mut request = request();
    request
        .parameters
        .insert("steps".to_string(), 42_i64.into());
    let mut context = ResolutionContext::default();
    context
        .runtime_defaults
        .insert("image.guidance".to_string(), 2.0_f64.into());
    context
        .hardware_profile
        .insert("image.width".to_string(), 768_i64.into());
    context
        .user_profile
        .insert("width".to_string(), 896_i64.into());

    let effective = resolve_request(&manifest, request, &context).unwrap();
    assert_eq!(effective.u64_parameter("image.steps"), Some(42));
    assert_eq!(
        effective.parameters["image.steps"].source,
        ParameterSource::RequestOverride
    );
    assert_eq!(effective.u64_parameter("image.width"), Some(896));
    assert_eq!(
        effective.parameters["image.width"].source,
        ParameterSource::UserProfile
    );
    assert_eq!(effective.f64_parameter("image.guidance"), Some(2.0));
    assert_eq!(
        effective.parameters["image.guidance"].source,
        ParameterSource::RuntimeDefault
    );
}

#[test]
fn generic_schemas_do_not_advertise_artificial_upper_limits() {
    for task in [
        InferenceTask::ImageGeneration,
        InferenceTask::VideoGeneration,
        InferenceTask::MusicGeneration,
        InferenceTask::TextToSpeech,
        InferenceTask::SpeechToText,
        InferenceTask::AudioClassification,
    ] {
        for descriptor in parameter_schema(task) {
            if descriptor.path.ends_with(".seed") {
                assert_eq!(descriptor.maximum, Some(i64::MAX.into()));
            } else {
                assert_eq!(
                    descriptor.maximum, None,
                    "{} still has a generic upper limit",
                    descriptor.path
                );
            }
        }
    }
}

#[test]
fn values_above_the_previous_generic_caps_are_forwarded_unchanged() {
    let manifest = image_manifest();
    let mut request = request();
    for (path, value) in [
        ("image.width", 32_769_i64.into()),
        ("image.height", 32_769_i64.into()),
        ("image.batch_size", 257_i64.into()),
        ("image.num_images", 1_025_i64.into()),
        ("image.steps", 1_001_i64.into()),
        ("image.guidance", 100.1_f64.into()),
    ] {
        request.parameters.insert(path.to_string(), value);
    }

    let effective = resolve_request(&manifest, request, &ResolutionContext::default()).unwrap();
    assert_eq!(effective.u64_parameter("image.width"), Some(32_769));
    assert_eq!(effective.u64_parameter("image.height"), Some(32_769));
    assert_eq!(effective.u64_parameter("image.batch_size"), Some(257));
    assert_eq!(effective.u64_parameter("image.num_images"), Some(1_025));
    assert_eq!(effective.u64_parameter("image.steps"), Some(1_001));
    assert_eq!(effective.f64_parameter("image.guidance"), Some(100.1));
}

#[test]
fn unbounded_numeric_parameters_still_require_finite_values() {
    let manifest = image_manifest();
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let mut request = request();
        request
            .parameters
            .insert("image.guidance".to_string(), value.into());
        assert!(
            resolve_request(&manifest, request, &ResolutionContext::default())
                .unwrap_err()
                .to_string()
                .contains("must be finite")
        );
    }
}

#[test]
fn model_reported_maximum_warns_but_does_not_rewrite_or_reject() {
    let mut manifest = image_manifest();
    manifest.metadata.parameter_constraints.insert(
        "width".to_string(),
        json!({
            "maximum": 768
        }),
    );
    let schema = parameter_schema_for_manifest(InferenceTask::ImageGeneration, &manifest).unwrap();
    let width = schema
        .iter()
        .find(|descriptor| descriptor.path == "image.width")
        .unwrap();
    assert_eq!(width.minimum, Some(64_i64.into()));
    assert_eq!(width.maximum, Some(768_i64.into()));

    let mut below_minimum = request();
    below_minimum
        .parameters
        .insert("image.width".to_string(), 1_i64.into());
    assert!(
        resolve_request(&manifest, below_minimum, &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("below its minimum")
    );

    let mut request = request();
    request
        .parameters
        .insert("image.width".to_string(), 1_024_i64.into());

    let effective = resolve_request(&manifest, request, &ResolutionContext::default()).unwrap();
    assert_eq!(effective.u64_parameter("image.width"), Some(1_024));
    assert!(effective.warnings.iter().any(|warning| {
        warning.contains("image.width")
            && warning.contains("model-reported maximum 768")
            && warning.contains("forwarded unchanged")
    }));
}

#[test]
fn manifest_constraints_enrich_schema_and_validate_effective_values() {
    let mut manifest = image_manifest();
    manifest.metadata.parameter_constraints.insert(
        "width".to_string(),
        json!({
            "default": 512,
            "minimum": 320,
            "maximum": 768,
            "step": 64,
            "allowed_values": [320, 512, 768]
        }),
    );
    let schema = parameter_schema_for_manifest(InferenceTask::ImageGeneration, &manifest).unwrap();
    let width = schema
        .iter()
        .find(|descriptor| descriptor.path == "image.width")
        .unwrap();
    assert_eq!(width.minimum, Some(320_i64.into()));
    assert_eq!(width.maximum, Some(768_i64.into()));
    assert_eq!(width.step, Some(64_i64.into()));

    let mut invalid = request();
    invalid
        .parameters
        .insert("image.width".to_string(), 640_i64.into());
    assert!(
        resolve_request(&manifest, invalid, &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("must be one of")
    );
}

#[test]
fn invalid_model_default_is_reported_instead_of_silently_dropped() {
    let mut manifest = image_manifest();
    manifest
        .metadata
        .generation_defaults
        .insert("image.width".to_string(), json!({"not": "an integer"}));
    assert!(
        resolve_request(&manifest, request(), &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("expects Integer")
    );
}

#[test]
fn family_default_is_overridden_by_model_default() {
    let manifest = image_manifest();
    let effective = resolve_request(&manifest, request(), &ResolutionContext::default()).unwrap();
    assert_eq!(effective.u64_parameter("image.steps"), Some(35));
    assert_eq!(
        effective.parameters["image.steps"].source,
        ParameterSource::ModelDefault
    );
    assert_eq!(effective.f64_parameter("image.guidance"), Some(3.5));
    assert_eq!(
        effective.parameters["image.guidance"].source,
        ParameterSource::ModelFamilyDefault
    );
}

#[test]
fn wan22_ti2v_family_uses_official_video_defaults() {
    let manifest = video_manifest("wan2.2-ti2v");
    let mut request = InferenceRequest::new(&manifest.id, InferenceTask::VideoGeneration);
    request.prompt = Some("clouds over a mountain lake".to_string());

    let effective = resolve_request(&manifest, request, &ResolutionContext::default()).unwrap();
    let expected = [
        ("video.width", 1280_i64.into()),
        ("video.height", 704_i64.into()),
        ("video.frames", 121_i64.into()),
        ("video.fps", 24.0_f64.into()),
        ("video.steps", 50_i64.into()),
        ("video.guidance", 5.0_f64.into()),
    ];
    for (path, value) in expected {
        assert_eq!(effective.parameter(path), Some(&value), "{path}");
        assert_eq!(
            effective.parameters[path].source,
            ParameterSource::ModelFamilyDefault,
            "{path}"
        );
    }
}

#[test]
fn generic_wan_family_keeps_existing_video_defaults() {
    let manifest = video_manifest("wan");
    let mut request = InferenceRequest::new(&manifest.id, InferenceTask::VideoGeneration);
    request.prompt = Some("clouds over a mountain lake".to_string());

    let effective = resolve_request(&manifest, request, &ResolutionContext::default()).unwrap();
    assert_eq!(effective.u64_parameter("video.frames"), Some(81));
    assert_eq!(effective.f64_parameter("video.fps"), Some(16.0));
    assert_eq!(effective.u64_parameter("video.width"), Some(832));
    assert_eq!(effective.u64_parameter("video.height"), Some(480));
    assert_eq!(effective.u64_parameter("video.steps"), Some(30));
    assert_eq!(effective.f64_parameter("video.guidance"), Some(6.0));
}

#[test]
fn validation_rejects_ranges_and_unknown_parameters() {
    let manifest = image_manifest();
    let mut invalid = request();
    invalid.parameters.insert("width".to_string(), 1_i64.into());
    assert!(
        resolve_request(&manifest, invalid, &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("minimum")
    );

    let mut unknown = request();
    unknown
        .parameters
        .insert("telepathy".to_string(), true.into());
    assert!(
        resolve_request(&manifest, unknown, &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("unknown parameter")
    );
}

#[test]
fn strict_parameter_support_rejects_ignored_override() {
    let manifest = image_manifest();
    let mut request = request();
    request
        .parameters
        .insert("steps".to_string(), 12_i64.into());
    let mut context = ResolutionContext::default();
    context
        .parameter_support
        .insert("image.steps".to_string(), ParameterSupportStatus::Ignored);
    assert!(
        resolve_request(&manifest, request.clone(), &context)
            .unwrap_err()
            .to_string()
            .contains("ignored")
    );
    request.routing.parameter_policy = ParameterPolicy::Warn;
    let effective = resolve_request(&manifest, request, &context).unwrap();
    assert_eq!(effective.warnings.len(), 1);
}

#[test]
fn schema_is_machine_readable_and_complete_for_core_image_values() {
    let schema = parameter_schema(InferenceTask::ImageGeneration);
    let paths = schema
        .iter()
        .map(|descriptor| descriptor.path.as_str())
        .collect::<BTreeSet<_>>();
    assert!(paths.contains("routing.parameter_policy"));
    assert!(paths.contains("image.width"));
    assert!(paths.contains("image.controls"));
    assert!(paths.contains("image.loras"));
    assert!(paths.contains("image.vae_tiling"));
    assert!(paths.contains("image.output_path"));
    let width = schema
        .iter()
        .find(|descriptor| descriptor.path == "image.width")
        .unwrap();
    assert!(width.affects_memory);
    assert!(width.affects_runtime);
    assert_eq!(width.minimum, Some(64_i64.into()));
}

#[test]
fn schema_uses_real_primary_flags_for_structured_and_repeatable_values() {
    let flags = [
        (
            InferenceTask::ImageGeneration,
            "image.controls",
            "--image-control",
        ),
        (
            InferenceTask::ImageGeneration,
            "image.loras",
            "--image-lora",
        ),
        (
            InferenceTask::VideoGeneration,
            "video.prompt_keyframes",
            "--video-prompt-keyframe",
        ),
        (InferenceTask::MusicGeneration, "audio.genres", "--genre"),
        (
            InferenceTask::MusicGeneration,
            "audio.instruments",
            "--music-instrument",
        ),
        (
            InferenceTask::MusicGeneration,
            "audio.variations",
            "--num-variations",
        ),
        (
            InferenceTask::MusicGeneration,
            "audio.output_format",
            "--format",
        ),
        (InferenceTask::TextToSpeech, "tts.output_format", "--format"),
        (
            InferenceTask::TextToSpeech,
            "tts.loudness",
            "--loudness-lufs",
        ),
        (InferenceTask::SpeechToText, "stt.operation", "--task"),
        (
            InferenceTask::SpeechToText,
            "stt.temperature_fallbacks",
            "--temperature-fallback",
        ),
    ];
    for (task, path, expected) in flags {
        let schema = parameter_schema(task);
        assert_eq!(
            schema
                .iter()
                .find(|descriptor| descriptor.path == path)
                .map(|descriptor| descriptor.cli_flag.as_str()),
            Some(expected),
            "{path}"
        );
    }
}

#[test]
fn task_schemas_only_advertise_flags_on_their_public_subcommands() {
    for task in [
        InferenceTask::ImageGeneration,
        InferenceTask::ImageEditing,
        InferenceTask::ImageVariation,
        InferenceTask::ImageInpainting,
        InferenceTask::ImageOutpainting,
    ] {
        let subcommand = if task == InferenceTask::ImageGeneration {
            "generate"
        } else {
            "edit"
        };
        assert_schema_flags_exist_on_subcommand::<ImageSchemaCli>(task, subcommand);
    }
    assert_schema_flags_exist_on_subcommand::<ImageSchemaCli>(
        InferenceTask::ImageUpscaling,
        "upscale",
    );

    assert_schema_flags_exist_on_subcommand::<VideoSchemaCli>(
        InferenceTask::VideoGeneration,
        "generate",
    );
    assert_schema_flags_exist_on_subcommand::<VideoSchemaCli>(
        InferenceTask::ImageToVideo,
        "animate",
    );
    for task in [
        InferenceTask::VideoToVideo,
        InferenceTask::VideoInpainting,
        InferenceTask::VideoExtension,
    ] {
        assert_schema_flags_exist_on_subcommand::<VideoSchemaCli>(task, "transform");
    }
    for task in [
        InferenceTask::VideoUpscaling,
        InferenceTask::FrameInterpolation,
    ] {
        assert_schema_flags_exist_on_subcommand::<VideoSchemaCli>(task, "upscale");
    }

    for task in [
        InferenceTask::AudioGeneration,
        InferenceTask::MusicGeneration,
        InferenceTask::SongContinuation,
        InferenceTask::SongVariation,
    ] {
        assert_schema_flags_exist_on_subcommand::<AudioSchemaCli>(task, "generate");
    }
    assert_schema_flags_exist_on_subcommand::<AudioSchemaCli>(InferenceTask::TextToSpeech, "speak");
    assert_schema_flags_exist_on_subcommand::<AudioSchemaCli>(
        InferenceTask::SpeechToText,
        "transcribe",
    );
    assert_schema_flags_exist_on_subcommand::<AudioSchemaCli>(
        InferenceTask::SpeechTranslation,
        "translate",
    );
    assert_schema_flags_exist_on_subcommand::<AudioSchemaCli>(
        InferenceTask::StemSeparation,
        "separate",
    );
    for (task, subcommand) in [
        (InferenceTask::AudioEventDetection, "event"),
        (InferenceTask::VoiceActivityDetection, "voice"),
        (InferenceTask::SpeakerIdentification, "speaker"),
        (InferenceTask::LanguageIdentification, "language"),
        (InferenceTask::SpeechEmotionRecognition, "emotion"),
    ] {
        assert_schema_flags_exist_on_nested_subcommand::<AudioSchemaCli>(
            task, "detect", subcommand,
        );
    }
    assert_schema_flags_exist_on_nested_subcommand::<AudioSchemaCli>(
        InferenceTask::AudioClassification,
        "analyze",
        "classify",
    );
    assert_schema_flags_exist_on_nested_subcommand::<AudioSchemaCli>(
        InferenceTask::AudioCaptioning,
        "analyze",
        "caption",
    );
    assert_schema_flags_exist_on_nested_subcommand::<AudioSchemaCli>(
        InferenceTask::AudioUnderstanding,
        "analyze",
        "understand",
    );
    assert_schema_flags_exist_on_subcommand::<AudioSchemaCli>(
        InferenceTask::AudioEmbedding,
        "embed",
    );
}

#[test]
fn prepared_tasks_without_public_subcommands_do_not_advertise_task_flags() {
    for task in [
        InferenceTask::VoiceConversion,
        InferenceTask::StemGeneration,
        InferenceTask::AudioEnhancement,
        InferenceTask::SpeakerDiarization,
        InferenceTask::AudioEditing,
    ] {
        assert!(
            parameter_schema(task)
                .iter()
                .all(|descriptor| descriptor.path.starts_with("routing.")),
            "{task}"
        );
    }
}

#[test]
fn audio_classification_and_translation_schemas_expose_runtime_parameters() {
    assert!(InferenceTask::AudioUnderstanding.requires_prompt());
    assert!(!InferenceTask::AudioCaptioning.requires_prompt());
    for task in [
        InferenceTask::AudioEventDetection,
        InferenceTask::VoiceActivityDetection,
        InferenceTask::SpeakerIdentification,
        InferenceTask::LanguageIdentification,
        InferenceTask::SpeechEmotionRecognition,
        InferenceTask::AudioClassification,
    ] {
        let paths = parameter_schema(task)
            .into_iter()
            .map(|descriptor| descriptor.path)
            .collect::<BTreeSet<_>>();
        assert!(paths.contains("audio.top_k"), "{task}");
        assert!(paths.contains("audio.output_format"), "{task}");
    }

    let translation = parameter_schema(InferenceTask::SpeechTranslation)
        .into_iter()
        .map(|descriptor| descriptor.path)
        .collect::<BTreeSet<_>>();
    assert!(translation.contains("stt.operation"));
    assert!(translation.contains("stt.language"));
    assert!(translation.contains("stt.output_format"));

    for task in [
        InferenceTask::AudioCaptioning,
        InferenceTask::AudioUnderstanding,
    ] {
        let paths = parameter_schema(task)
            .into_iter()
            .map(|descriptor| descriptor.path)
            .collect::<BTreeSet<_>>();
        assert!(paths.contains("audio.max_new_tokens"), "{task}");
        assert!(paths.contains("audio.temperature"), "{task}");
        assert!(paths.contains("audio.top_k"), "{task}");
        assert!(paths.contains("audio.top_p"), "{task}");
        assert!(paths.contains("audio.output_format"), "{task}");
    }

    let embedding = parameter_schema(InferenceTask::AudioEmbedding)
        .into_iter()
        .map(|descriptor| descriptor.path)
        .collect::<BTreeSet<_>>();
    assert!(embedding.contains("audio.normalize"));
    assert!(embedding.contains("audio.pooling"));
    assert!(embedding.contains("audio.output_format"));
}

#[test]
fn audio_analysis_rejects_prompts_that_the_runtime_would_ignore() {
    let mut manifest = image_manifest();
    manifest.id = "audio".to_string();
    manifest.metadata.tasks = vec![
        InferenceTask::SpeechToText,
        InferenceTask::SpeechTranslation,
        InferenceTask::TextToSpeech,
        InferenceTask::AudioClassification,
        InferenceTask::AudioCaptioning,
        InferenceTask::AudioEmbedding,
    ];
    manifest.metadata.input_modalities = vec![InputModality::Audio, InputModality::Text];
    manifest.metadata.output_modalities = vec![
        OutputModality::Text,
        OutputModality::Embedding,
        OutputModality::Audio,
    ];
    let audio_input = || InferenceInput {
        modality: InputModality::Audio,
        role: "input_audio".to_string(),
        source: InferenceInputSource::Path {
            path: "clip.wav".to_string(),
        },
        mime_type: Some("audio/wav".to_string()),
    };

    let mut transcription = InferenceRequest::new("audio", InferenceTask::SpeechToText);
    transcription.prompt = Some("ignored ASR hint".to_string());
    transcription.inputs.push(audio_input());
    assert!(
        resolve_request(&manifest, transcription, &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("does not consume a prompt")
    );

    let mut contradictory_translation =
        InferenceRequest::new("audio", InferenceTask::SpeechTranslation);
    contradictory_translation.inputs.push(audio_input());
    contradictory_translation.parameters.insert(
        "stt.operation".to_string(),
        ParameterValue::String("transcribe".to_string()),
    );
    assert!(
        resolve_request(
            &manifest,
            contradictory_translation,
            &ResolutionContext::default()
        )
        .is_err()
    );

    let mut speech = InferenceRequest::new("audio", InferenceTask::TextToSpeech);
    speech.prompt = Some("Read this aloud".to_string());
    speech.negative_prompt = Some("ignored delivery constraint".to_string());
    let error = resolve_request(&manifest, speech, &ResolutionContext::default())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("does not consume a negative prompt"),
        "{error}"
    );

    let mut classification = InferenceRequest::new("audio", InferenceTask::AudioClassification);
    classification.prompt = Some("ignored label hint".to_string());
    classification.inputs.push(audio_input());
    assert!(
        resolve_request(&manifest, classification, &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("does not consume a prompt")
    );

    let mut embedding = InferenceRequest::new("audio", InferenceTask::AudioEmbedding);
    embedding.prompt = Some("ignored embedding hint".to_string());
    embedding.inputs.push(audio_input());
    assert!(
        resolve_request(&manifest, embedding, &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("does not consume a prompt")
    );

    let mut caption = InferenceRequest::new("audio", InferenceTask::AudioCaptioning);
    caption.prompt = Some("Describe only the foreground sound".to_string());
    caption.negative_prompt = Some("silence".to_string());
    caption.inputs.push(audio_input());
    assert!(
        resolve_request(&manifest, caption, &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("does not consume a negative prompt")
    );
}

#[test]
fn task_specific_schema_filtering_keeps_resolution_consistent() {
    let image_upscale = parameter_schema(InferenceTask::ImageUpscaling);
    assert!(
        image_upscale
            .iter()
            .all(|descriptor| descriptor.path != "image.steps")
    );
    assert!(
        image_upscale
            .iter()
            .all(|descriptor| descriptor.path != "image.controls")
    );

    let video_upscale = parameter_schema(InferenceTask::VideoUpscaling);
    assert!(
        video_upscale
            .iter()
            .all(|descriptor| descriptor.path != "video.reference_images")
    );
    assert!(
        video_upscale
            .iter()
            .all(|descriptor| descriptor.path != "video.prompt_keyframes")
    );

    let stem_paths = parameter_schema(InferenceTask::StemSeparation)
        .into_iter()
        .filter(|descriptor| descriptor.path.starts_with("audio."))
        .map(|descriptor| descriptor.path)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        stem_paths,
        [
            "audio.channels",
            "audio.normalization",
            "audio.num_stems",
            "audio.output_format",
            "audio.output_path",
            "audio.sample_rate",
            "audio.segment_duration",
            "audio.segment_overlap",
            "audio.stems",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    );

    let mut manifest = image_manifest();
    manifest.metadata.tasks = vec![InferenceTask::ImageUpscaling];
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageUpscaling);
    request.inputs.push(InferenceInput {
        modality: InputModality::Image,
        role: "image".to_string(),
        source: InferenceInputSource::Path {
            path: "input.png".to_string(),
        },
        mime_type: Some("image/png".to_string()),
    });
    let effective = resolve_request(&manifest, request, &ResolutionContext::default()).unwrap();
    assert!(!effective.parameters.contains_key("image.steps"));
    assert!(effective.parameters.contains_key("image.upscale_scale"));
}

#[test]
fn image_estimate_scales_with_resolution_and_tiling() {
    let manifest = image_manifest();
    let low = resolve_request(&manifest, request(), &ResolutionContext::default()).unwrap();
    let mut high_request = request();
    high_request
        .parameters
        .insert("width".to_string(), 2048_i64.into());
    high_request
        .parameters
        .insert("height".to_string(), 2048_i64.into());
    let high = resolve_request(&manifest, high_request, &ResolutionContext::default()).unwrap();
    let resources = HostResources::default();
    let low_estimate = estimate_workload(&manifest, &low, &resources);
    let high_estimate = estimate_workload(&manifest, &high, &resources);
    assert!(
        high_estimate.accelerator_peak_bytes.unwrap()
            > low_estimate.accelerator_peak_bytes.unwrap()
    );

    let mut tiled_request = request();
    tiled_request
        .parameters
        .insert("vae_tiling".to_string(), true.into());
    let tiled = resolve_request(&manifest, tiled_request, &ResolutionContext::default()).unwrap();
    assert!(
        estimate_workload(&manifest, &tiled, &resources).accelerator_peak_bytes
            < low_estimate.accelerator_peak_bytes
    );
}

#[test]
fn fit_rejects_zero_accelerator_headroom() {
    let resources = HostResources {
        host_memory_bytes: Some(1_000),
        accelerator_memory_bytes: Some(100),
        accelerator: Some("cuda".to_string()),
        memory_topology: None,
    };

    assert_eq!(
        classify_workload_fit(Some(100), Some(20), &resources),
        FitAssessment::LikelyOom
    );
}

#[test]
fn cuda_fit_stays_unknown_without_detected_vram_unless_host_is_already_oom() {
    let resources = HostResources {
        host_memory_bytes: Some(1_000),
        accelerator_memory_bytes: None,
        accelerator: Some("cuda".to_string()),
        memory_topology: None,
    };

    assert_eq!(
        classify_workload_fit(Some(100), Some(20), &resources),
        FitAssessment::Unknown
    );
    assert_eq!(
        classify_workload_fit(Some(100), Some(1_000), &resources),
        FitAssessment::LikelyOom
    );
}

#[test]
fn unified_memory_uses_memavailable_as_accelerator_hard_bound() {
    const GIB: u64 = 1024 * 1024 * 1024;
    let resources = HostResources {
        host_memory_bytes: Some(96 * GIB),
        accelerator_memory_bytes: None,
        accelerator: Some("rocm".to_string()),
        memory_topology: Some(MemoryTopology::Unified),
    };

    assert_eq!(
        classify_workload_fit(Some(120 * GIB), Some(20 * GIB), &resources),
        FitAssessment::LikelyOom
    );
}

#[test]
fn unified_workload_estimate_blocks_accelerator_peak_above_shared_hard_bound() {
    let manifest = image_manifest();
    let effective = resolve_request(&manifest, request(), &ResolutionContext::default()).unwrap();
    let shared_limit = 1024 * 1024 * 1024;
    let estimate = estimate_workload(
        &manifest,
        &effective,
        &HostResources {
            host_memory_bytes: Some(shared_limit),
            accelerator_memory_bytes: None,
            accelerator: Some("rocm".to_string()),
            memory_topology: Some(MemoryTopology::Unified),
        },
    );

    assert!(estimate.host_peak_bytes.unwrap() < shared_limit);
    assert!(estimate.accelerator_peak_bytes.unwrap() >= shared_limit);
    assert_eq!(estimate.fit, FitAssessment::LikelyOom);
    assert!(
        estimate
            .warnings
            .iter()
            .any(|warning| warning.contains("unified-memory host-visible hard bound"))
    );
}

#[test]
fn unified_memory_estimate_is_explicit_about_unknown_accelerator_fit() {
    let manifest = image_manifest();
    let effective = resolve_request(&manifest, request(), &ResolutionContext::default()).unwrap();
    let estimate = estimate_workload(
        &manifest,
        &effective,
        &HostResources {
            host_memory_bytes: Some(128 * 1024 * 1024 * 1024),
            accelerator_memory_bytes: None,
            accelerator: Some("cuda".to_string()),
            memory_topology: Some(MemoryTopology::Unified),
        },
    );

    assert_eq!(estimate.fit, FitAssessment::Unknown);
    assert_eq!(estimate.accelerator_memory_limit_bytes, None);
    assert!(
        estimate
            .assumptions
            .iter()
            .any(|value| value.contains("shared CPU/GPU unified-memory system"))
    );
    assert!(
        estimate
            .warnings
            .iter()
            .any(|value| value.contains("no independently reported accelerator-memory capacity"))
    );

    let cpu_estimate = estimate_workload(
        &manifest,
        &effective,
        &HostResources {
            host_memory_bytes: Some(128 * 1024 * 1024 * 1024),
            accelerator_memory_bytes: None,
            accelerator: Some("cpu".to_string()),
            memory_topology: Some(MemoryTopology::Unified),
        },
    );
    assert!(
        !cpu_estimate
            .warnings
            .iter()
            .any(|value| value.contains("no independently reported accelerator-memory capacity"))
    );

    let configured_limit_estimate = estimate_workload(
        &manifest,
        &effective,
        &HostResources {
            host_memory_bytes: Some(128 * 1024 * 1024 * 1024),
            accelerator_memory_bytes: Some(96 * 1024 * 1024 * 1024),
            accelerator: Some("rocm".to_string()),
            memory_topology: Some(MemoryTopology::Unified),
        },
    );
    assert!(configured_limit_estimate.warnings.iter().any(|value| {
        value.contains("same unified-memory system") && value.contains("not added to host memory")
    }));
    assert!(
        configured_limit_estimate
            .warnings
            .iter()
            .all(|value| !value.contains("DGX Spark"))
    );
}

#[test]
fn workload_estimate_deserializes_without_memory_limits() {
    let manifest = image_manifest();
    let effective = resolve_request(&manifest, request(), &ResolutionContext::default()).unwrap();
    let estimate = estimate_workload(
        &manifest,
        &effective,
        &HostResources {
            host_memory_bytes: Some(64 * 1024 * 1024 * 1024),
            accelerator_memory_bytes: Some(24 * 1024 * 1024 * 1024),
            accelerator: Some("cuda".to_string()),
            memory_topology: None,
        },
    );
    let mut legacy = serde_json::to_value(estimate).unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("accelerator_memory_limit_bytes");
    legacy
        .as_object_mut()
        .unwrap()
        .remove("host_memory_limit_bytes");

    let decoded: WorkloadEstimate = serde_json::from_value(legacy).unwrap();

    assert_eq!(decoded.accelerator_memory_limit_bytes, None);
    assert_eq!(decoded.host_memory_limit_bytes, None);
}

#[test]
fn offload_permission_does_not_change_estimate_until_planner_selects_it() {
    let manifest = image_manifest();
    let baseline = resolve_request(&manifest, request(), &ResolutionContext::default()).unwrap();
    let mut permitted_request = request();
    permitted_request.routing.allow_cpu_offload = OverrideBool::Enabled;
    let permitted =
        resolve_request(&manifest, permitted_request, &ResolutionContext::default()).unwrap();
    let resources = HostResources::default();

    let baseline_estimate = estimate_workload(&manifest, &baseline, &resources);
    let permitted_estimate = estimate_workload(&manifest, &permitted, &resources);
    assert_eq!(
        permitted_estimate.accelerator_peak_bytes,
        baseline_estimate.accelerator_peak_bytes
    );
    assert_eq!(
        permitted_estimate.host_peak_bytes,
        baseline_estimate.host_peak_bytes
    );

    let mut selected = permitted;
    selected
        .parameters
        .get_mut("routing.allow_cpu_offload")
        .unwrap()
        .source = ParameterSource::BackendAdjustment;
    assert!(
        estimate_workload(&manifest, &selected, &resources).host_peak_bytes
            > baseline_estimate.host_peak_bytes
    );
}

#[test]
fn planner_honors_explicit_gpu_offload_under_default_fallback_policy() {
    let manifest = image_manifest();
    let mut permitted_request = request();
    permitted_request.routing.allow_cpu_offload = OverrideBool::Enabled;
    permitted_request.routing.allow_sequential_offload = OverrideBool::Disabled;
    permitted_request.routing.allow_component_offload = OverrideBool::Disabled;
    let effective =
        resolve_request(&manifest, permitted_request, &ResolutionContext::default()).unwrap();
    let mut estimate = estimate_workload(&manifest, &effective, &HostResources::default());
    estimate.fit = FitAssessment::LikelyOom;
    estimate.accelerator_peak_bytes = Some(20 * 1024 * 1024 * 1024);
    estimate.accelerator_memory_limit_bytes = Some(12 * 1024 * 1024 * 1024);
    let candidate = |id: &str, accelerator: RuntimeAccelerator| InferenceRuntimeCandidate {
        id: id.to_string(),
        backend: "media-companion".to_string(),
        accelerator,
        available: true,
        availability_reason: None,
        supported_tasks: vec![InferenceTask::ImageGeneration],
        supported_layouts: vec![RepositoryLayout::Diffusers],
        supported_formats: vec![ModelFormat::SafeTensors],
        supported_families: Vec::new(),
        supported_architectures: Vec::new(),
        parameter_support: BTreeMap::new(),
        supports_offloading: true,
        supports_compile: false,
        supports_batching: true,
        priority: 100,
    };
    let plan = plan_execution(
        &manifest,
        &effective,
        &estimate,
        &[
            candidate("media-companion-cpu", RuntimeAccelerator::Cpu),
            candidate("media-companion-cuda", RuntimeAccelerator::Cuda),
        ],
    );
    let cpu = plan
        .candidates
        .iter()
        .find(|candidate| candidate.runtime_id == "media-companion-cpu")
        .unwrap();
    assert_eq!(cpu.status, PlanCandidateStatus::Rejected);
    assert!(cpu.degradations.is_empty());
    let cuda = plan
        .candidates
        .iter()
        .find(|candidate| candidate.runtime_id == "media-companion-cuda")
        .unwrap();
    assert_eq!(cuda.status, PlanCandidateStatus::Accepted);
    assert_eq!(cuda.degradations, vec![ExecutionDegradation::CpuOffload]);
}

#[test]
fn planner_rejects_before_loading_when_vram_is_exceeded_and_cpu_offload_is_disabled() {
    let manifest = image_manifest();
    let mut denied_request = request();
    denied_request.routing.allow_cpu_offload = OverrideBool::Disabled;
    let effective =
        resolve_request(&manifest, denied_request, &ResolutionContext::default()).unwrap();
    let mut estimate = estimate_workload(&manifest, &effective, &HostResources::default());
    estimate.fit = FitAssessment::Fits;
    estimate.accelerator_peak_bytes = Some(20 * 1024 * 1024 * 1024);
    estimate.accelerator_memory_limit_bytes = Some(12 * 1024 * 1024 * 1024);
    let candidate = InferenceRuntimeCandidate {
        id: "media-companion-cuda".to_string(),
        backend: "media-companion".to_string(),
        accelerator: RuntimeAccelerator::Cuda,
        available: true,
        availability_reason: None,
        supported_tasks: vec![InferenceTask::ImageGeneration],
        supported_layouts: vec![RepositoryLayout::Diffusers],
        supported_formats: vec![ModelFormat::SafeTensors],
        supported_families: Vec::new(),
        supported_architectures: Vec::new(),
        parameter_support: BTreeMap::new(),
        supports_offloading: true,
        supports_compile: false,
        supports_batching: true,
        priority: 100,
    };
    let plan = plan_execution(&manifest, &effective, &estimate, &[candidate]);

    assert_eq!(plan.selected_runtime, None);
    assert_eq!(plan.candidates[0].status, PlanCandidateStatus::Rejected);
    assert!(plan.candidates[0].degradations.is_empty());
    assert!(plan.candidates[0].reasons.iter().any(|reason| {
        reason.contains("20.00 GiB")
            && reason.contains("12.00 GiB")
            && reason.contains("no permitted offload")
    }));
}

#[test]
fn planner_rejects_every_offload_mode_when_projected_host_memory_does_not_fit() {
    let manifest = image_manifest();
    let candidate = InferenceRuntimeCandidate {
        id: "media-companion-cuda".to_string(),
        backend: "media-companion".to_string(),
        accelerator: RuntimeAccelerator::Cuda,
        available: true,
        availability_reason: None,
        supported_tasks: vec![InferenceTask::ImageGeneration],
        supported_layouts: vec![RepositoryLayout::Diffusers],
        supported_formats: vec![ModelFormat::SafeTensors],
        supported_families: Vec::new(),
        supported_architectures: Vec::new(),
        parameter_support: BTreeMap::new(),
        supports_offloading: true,
        supports_compile: false,
        supports_batching: true,
        priority: 100,
    };

    for offload in [
        ExecutionDegradation::CpuOffload,
        ExecutionDegradation::SequentialOffload,
        ExecutionDegradation::ComponentOffload,
    ] {
        let mut offload_request = request();
        offload_request.routing.allow_cpu_offload = OverrideBool::Disabled;
        offload_request.routing.allow_sequential_offload = OverrideBool::Disabled;
        offload_request.routing.allow_component_offload = OverrideBool::Disabled;
        match offload {
            ExecutionDegradation::CpuOffload => {
                offload_request.routing.allow_cpu_offload = OverrideBool::Enabled;
            }
            ExecutionDegradation::SequentialOffload => {
                offload_request.routing.allow_sequential_offload = OverrideBool::Enabled;
            }
            ExecutionDegradation::ComponentOffload => {
                offload_request.routing.allow_component_offload = OverrideBool::Enabled;
            }
            _ => unreachable!(),
        }
        let effective =
            resolve_request(&manifest, offload_request, &ResolutionContext::default()).unwrap();
        let mut estimate = estimate_workload(&manifest, &effective, &HostResources::default());
        estimate.fit = FitAssessment::LikelyOom;
        estimate.weight_payload_bytes = Some(10 * 1024 * 1024 * 1024);
        estimate.accelerator_peak_bytes = Some(20 * 1024 * 1024 * 1024);
        estimate.accelerator_memory_limit_bytes = Some(12 * 1024 * 1024 * 1024);
        estimate.host_peak_bytes = Some(4 * 1024 * 1024 * 1024);
        estimate.host_memory_limit_bytes = Some(16 * 1024 * 1024 * 1024);

        let plan = plan_execution(
            &manifest,
            &effective,
            &estimate,
            std::slice::from_ref(&candidate),
        );

        assert_eq!(plan.selected_runtime, None, "offload mode {offload:?}");
        assert!(plan.candidates[0].degradations.is_empty());
        assert!(plan.candidates[0].reasons.iter().any(|reason| {
            reason.contains("17.33 GiB")
                && reason.contains("16.00 GiB")
                && reason.contains("offload was not selected")
        }));
    }
}

#[test]
fn planner_never_uses_accelerator_offload_to_mask_known_host_oom() {
    let manifest = image_manifest();
    let mut offload_request = request();
    offload_request.routing.allow_cpu_offload = OverrideBool::Enabled;
    let effective =
        resolve_request(&manifest, offload_request, &ResolutionContext::default()).unwrap();
    let mut estimate = estimate_workload(&manifest, &effective, &HostResources::default());
    estimate.fit = FitAssessment::LikelyOom;
    estimate.accelerator_peak_bytes = Some(20 * 1024 * 1024 * 1024);
    estimate.accelerator_memory_limit_bytes = Some(12 * 1024 * 1024 * 1024);
    estimate.host_peak_bytes = Some(16 * 1024 * 1024 * 1024);
    estimate.host_memory_limit_bytes = Some(16 * 1024 * 1024 * 1024);
    let candidate = InferenceRuntimeCandidate {
        id: "media-companion-cuda".to_string(),
        backend: "media-companion".to_string(),
        accelerator: RuntimeAccelerator::Cuda,
        available: true,
        availability_reason: None,
        supported_tasks: vec![InferenceTask::ImageGeneration],
        supported_layouts: vec![RepositoryLayout::Diffusers],
        supported_formats: vec![ModelFormat::SafeTensors],
        supported_families: Vec::new(),
        supported_architectures: Vec::new(),
        parameter_support: BTreeMap::new(),
        supports_offloading: true,
        supports_compile: false,
        supports_batching: true,
        priority: 100,
    };

    let plan = plan_execution(&manifest, &effective, &estimate, &[candidate]);

    assert_eq!(plan.selected_runtime, None);
    assert!(plan.candidates[0].degradations.is_empty());
    assert!(plan.candidates[0].reasons.iter().any(|reason| {
        reason.contains("16.00 GiB")
            && reason.contains("accelerator offload cannot resolve host-memory pressure")
    }));
}

#[test]
fn planner_gives_device_precedence_over_conflicting_accelerator() {
    let manifest = image_manifest();
    let mut routed_request = request();
    routed_request.routing.accelerator = Some("cuda".to_string());
    routed_request.routing.device = Some(" cpu ".to_string());
    let effective =
        resolve_request(&manifest, routed_request, &ResolutionContext::default()).unwrap();
    let estimate = estimate_workload(&manifest, &effective, &HostResources::default());
    let candidate = |id: &str, accelerator: RuntimeAccelerator| InferenceRuntimeCandidate {
        id: id.to_string(),
        backend: "media-companion".to_string(),
        accelerator,
        available: true,
        availability_reason: None,
        supported_tasks: vec![InferenceTask::ImageGeneration],
        supported_layouts: vec![RepositoryLayout::Diffusers],
        supported_formats: vec![ModelFormat::SafeTensors],
        supported_families: Vec::new(),
        supported_architectures: Vec::new(),
        parameter_support: BTreeMap::new(),
        supports_offloading: true,
        supports_compile: false,
        supports_batching: true,
        priority: 100,
    };

    let plan = plan_execution(
        &manifest,
        &effective,
        &estimate,
        &[
            candidate("media-companion-cuda", RuntimeAccelerator::Cuda),
            candidate("media-companion-cpu", RuntimeAccelerator::Cpu),
        ],
    );

    assert_eq!(
        plan.selected_runtime.as_deref(),
        Some("media-companion-cpu")
    );
    let cuda = plan
        .candidates
        .iter()
        .find(|candidate| candidate.runtime_id == "media-companion-cuda")
        .unwrap();
    assert!(
        cuda.reasons
            .iter()
            .any(|reason| reason == "accelerator 'cpu' was requested")
    );
}

#[test]
fn degrade_policy_selects_one_offload_and_respects_explicit_cpu_denial() {
    let manifest = image_manifest();
    let candidate = InferenceRuntimeCandidate {
        id: "media-companion-cuda".to_string(),
        backend: "media-companion".to_string(),
        accelerator: RuntimeAccelerator::Cuda,
        available: true,
        availability_reason: None,
        supported_tasks: vec![InferenceTask::ImageGeneration],
        supported_layouts: vec![RepositoryLayout::Diffusers],
        supported_formats: vec![ModelFormat::SafeTensors],
        supported_families: Vec::new(),
        supported_architectures: Vec::new(),
        parameter_support: BTreeMap::new(),
        supports_offloading: true,
        supports_compile: false,
        supports_batching: true,
        priority: 100,
    };
    let mut degrade_request = request();
    degrade_request.routing.fallback_policy = Some("degrade".to_string());
    let effective =
        resolve_request(&manifest, degrade_request, &ResolutionContext::default()).unwrap();
    let mut estimate = estimate_workload(&manifest, &effective, &HostResources::default());
    estimate.fit = FitAssessment::LikelyOom;
    let plan = plan_execution(
        &manifest,
        &effective,
        &estimate,
        std::slice::from_ref(&candidate),
    );
    assert_eq!(plan.degradations, vec![ExecutionDegradation::CpuOffload]);

    let mut denied_request = request();
    denied_request.routing.fallback_policy = Some("degrade".to_string());
    denied_request.routing.allow_cpu_offload = OverrideBool::Disabled;
    let denied = resolve_request(&manifest, denied_request, &ResolutionContext::default()).unwrap();
    let denied_plan = plan_execution(
        &manifest,
        &denied,
        &estimate,
        std::slice::from_ref(&candidate),
    );
    assert_eq!(denied_plan.selected_runtime, None);
    assert!(denied_plan.candidates[0].degradations.is_empty());

    let mut alternative_request = request();
    alternative_request.routing.fallback_policy = Some("degrade".to_string());
    alternative_request.routing.allow_cpu_offload = OverrideBool::Disabled;
    alternative_request.routing.allow_component_offload = OverrideBool::Enabled;
    let alternative = resolve_request(
        &manifest,
        alternative_request,
        &ResolutionContext::default(),
    )
    .unwrap();
    let alternative_plan = plan_execution(&manifest, &alternative, &estimate, &[candidate]);
    assert_eq!(
        alternative_plan.degradations,
        vec![ExecutionDegradation::ComponentOffload]
    );
}

#[test]
fn planner_scores_fallbacks_and_never_applies_quality_downgrade() {
    let manifest = image_manifest();
    let effective = resolve_request(&manifest, request(), &ResolutionContext::default()).unwrap();
    let estimate = estimate_workload(
        &manifest,
        &effective,
        &HostResources {
            host_memory_bytes: Some(16_000_000_000),
            accelerator_memory_bytes: Some(16_000_000_000),
            accelerator: Some("cuda".to_string()),
            memory_topology: None,
        },
    );
    let candidates = vec![
        InferenceRuntimeCandidate {
            id: "unavailable".to_string(),
            backend: "diffusers".to_string(),
            accelerator: RuntimeAccelerator::Cuda,
            available: false,
            availability_reason: Some("missing".to_string()),
            supported_tasks: vec![InferenceTask::ImageGeneration],
            supported_layouts: vec![RepositoryLayout::Diffusers],
            supported_formats: vec![ModelFormat::SafeTensors],
            supported_families: Vec::new(),
            supported_architectures: Vec::new(),
            parameter_support: BTreeMap::new(),
            supports_offloading: true,
            supports_compile: true,
            supports_batching: true,
            priority: 1000,
        },
        InferenceRuntimeCandidate {
            id: "working".to_string(),
            backend: "media-companion".to_string(),
            accelerator: RuntimeAccelerator::Cpu,
            available: true,
            availability_reason: None,
            supported_tasks: vec![InferenceTask::ImageGeneration],
            supported_layouts: vec![RepositoryLayout::Diffusers],
            supported_formats: vec![ModelFormat::SafeTensors],
            supported_families: Vec::new(),
            supported_architectures: Vec::new(),
            parameter_support: BTreeMap::new(),
            supports_offloading: true,
            supports_compile: false,
            supports_batching: true,
            priority: 500,
        },
    ];
    let plan = plan_execution(&manifest, &effective, &estimate, &candidates);
    assert_eq!(plan.selected_runtime.as_deref(), Some("working"));
    assert!(
        plan.candidates
            .iter()
            .any(|candidate| candidate.runtime_id == "unavailable"
                && candidate.status == PlanCandidateStatus::Rejected)
    );
    assert!(plan.model_or_quality_downgrades.is_empty());
}

#[test]
fn conversation_content_serializes_typed_media_and_tools() {
    let content = ConversationContent::ToolResult(ToolResultContent {
        call_id: "call-1".to_string(),
        result: json!({"ok": true}),
        content: vec![ConversationContent::Image(MediaContent {
            url: Some("/v1/outputs/out.png".to_string()),
            path: None,
            mime_type: "image/png".to_string(),
            metadata: BTreeMap::new(),
        })],
    });
    let value = serde_json::to_value(content).unwrap();
    assert_eq!(value["type"], "tool_result");
    assert_eq!(value["content"]["content"][0]["type"], "image");
}
