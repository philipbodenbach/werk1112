#!/usr/bin/env python3
"""Offline media companion for Werk.

The traditional command mode handles exactly one JSON request. ``serve`` uses
a newline-delimited resident transport so configured Diffusers and Transformers
media models can remain warm between requests. Model loading is deliberately
local-only: this module never installs packages and never downloads a model.
"""

import contextlib
import gc
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import wave
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path


PROTOCOL_VERSION = 1
COMPANION_VERSION = "1.5.1"
TRANSPORT_VERSION = 1
PIPELINE_CACHE_SIZE_ENV = "WERK_MEDIA_PIPELINE_CACHE_SIZE"
QWEN3_TTS_ADAPTER = "qwen3_tts_voice_design"
QWEN3_TTS_PYTHON_ENV = "WERK_QWEN_TTS_PYTHON"

for _name in (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "DIFFUSERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
):
    os.environ[_name] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


IMAGE_TASKS = {
    "image_generation",
    "image_editing",
    "image_variation",
    "image_inpainting",
    "image_outpainting",
    "image_upscaling",
}
VIDEO_TASKS = {
    "video_generation",
    "image_to_video",
    "video_to_video",
    "video_inpainting",
    "video_extension",
    "video_upscaling",
    "frame_interpolation",
}
VIDEO_INPUT_TASKS = {
    "video_to_video",
    "video_inpainting",
    "video_extension",
    "video_upscaling",
    "frame_interpolation",
}
AUDIO_GENERATION_TASKS = {
    "audio_generation",
    "music_generation",
}
TTS_TASKS = {"text_to_speech"}
ASR_TASKS = {"speech_to_text", "speech_translation"}
AUDIO_CLASSIFICATION_TASKS = {
    "audio_event_detection",
    "voice_activity_detection",
    "speaker_identification",
    "language_identification",
    "speech_emotion_recognition",
    "audio_classification",
}
AUDIO_TEXT_TASKS = {"audio_captioning", "audio_understanding"}
AUDIO_EMBEDDING_TASKS = {"audio_embedding"}
AUDIO_SOURCE_TASKS = (
    ASR_TASKS
    | AUDIO_CLASSIFICATION_TASKS
    | AUDIO_TEXT_TASKS
    | AUDIO_EMBEDDING_TASKS
)
ALL_AUDIO_TASKS = (
    AUDIO_GENERATION_TASKS
    | TTS_TASKS
    | AUDIO_SOURCE_TASKS
)
DECLARED_UNSUPPORTED_TASKS = {
    "song_continuation",
    "song_variation",
    "voice_conversion",
    "stem_generation",
    "stem_separation",
    "audio_enhancement",
    "speaker_diarization",
    "audio_editing",
}

ORCHESTRATOR_PARAMETERS = {
    "routing.backend",
    "routing.fallback_policy",
    "routing.parameter_policy",
    "routing.performance_preference",
    "routing.profile",
    "routing.quality",
    "routing.timeout",
}
TORCH_RUNTIME_PARAMETERS = {
    "routing.accelerator",
    "routing.device",
    "routing.precision",
}
DIFFUSERS_ROUTING_PARAMETERS = {
    "routing.allow_component_offload",
    "routing.allow_cpu_offload",
    "routing.allow_sequential_offload",
}
IMAGE_ADAPTER_PARAMETERS = {
    "image.batch_size",
    "image.guidance",
    "image.height",
    "image.loras",
    "image.num_images",
    "image.output_format",
    "image.seed",
    "image.steps",
    "image.vae_slicing",
    "image.vae_tiling",
    "image.width",
}
VIDEO_ADAPTER_PARAMETERS = {
    "video.batch_size",
    "video.fps",
    "video.frames",
    "video.guidance",
    "video.height",
    "video.num_videos",
    "video.output_format",
    "video.seed",
    "video.steps",
    "video.temporal_vae_tiling",
    "video.width",
}
DIFFUSERS_AUDIO_PARAMETERS = {
    "audio.duration",
    "audio.guidance",
    "audio.lyrics",
    "audio.output_format",
    "audio.seed",
    "audio.steps",
    "audio.variations",
}
TRANSFORMERS_AUDIO_PARAMETERS = {
    "audio.duration",
    "audio.guidance",
    "audio.lyrics",
    "audio.output_format",
    "audio.seed",
    "audio.temperature",
    "audio.top_k",
    "audio.top_p",
    "audio.variations",
}
TTS_ADAPTER_PARAMETERS = {
    "tts.output_format",
    "tts.seed",
}
QWEN3_TTS_VOICE_DESIGN_PARAMETERS = TTS_ADAPTER_PARAMETERS | {
    "tts.language",
    "tts.speaking_style",
}

# Architecture-specific Python packages live outside the generic Diffusers and
# Transformers adapters. Keep their identity, dependency and installation
# metadata together so model probing and execution routing cannot disagree.
# A future VoxCPM (or other audio) adapter is added as another entry rather
# than as repository-name heuristics or a generic Transformers fallback.
ARCHITECTURE_ADAPTER_REGISTRY = (
    {
        "adapter": QWEN3_TTS_ADAPTER,
        "model_types": frozenset({"qwen3_tts"}),
        "architectures": frozenset(),
        "variant_field": "tts_model_type",
        "variants": frozenset({"voice_design"}),
        "tasks": frozenset(TTS_TASKS),
        "dependencies": ("torch", "numpy", "qwen_tts"),
        "required_backend": "qwen-tts",
        "install_command": "werk backend install qwen-tts",
        "fallback_possible": False,
        "dependency_reason": (
            "Qwen3-TTS VoiceDesign requires torch, numpy, and an importable "
            "qwen-tts package in its companion Python environment; run "
            "'werk backend install qwen-tts' or select an isolated interpreter "
            f"with {QWEN3_TTS_PYTHON_ENV}"
        ),
    },
)
ASR_ADAPTER_PARAMETERS = {
    "stt.beam_size",
    "stt.initial_prompt",
    "stt.language",
    "stt.operation",
    "stt.output_format",
    "stt.segment_timestamps",
    "stt.temperature",
    "stt.word_timestamps",
}
AUDIO_CLASSIFICATION_PARAMETERS = {
    "audio.output_format",
    "audio.top_k",
}
AUDIO_TEXT_PARAMETERS = {
    "audio.max_new_tokens",
    "audio.output_format",
    "audio.temperature",
    "audio.top_k",
    "audio.top_p",
}
AUDIO_EMBEDDING_PARAMETERS = {
    "audio.normalize",
    "audio.output_format",
    "audio.pooling",
}


class CompanionFailure(Exception):
    def __init__(self, code, message, detail=None):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.detail = detail


def fail(code, message, detail=None):
    raise CompanionFailure(code, message, detail)


class DiffusersConfigurationOutcome:
    """Reusable result of configuring a cached Diffusers pipeline.

    Configuration mutates a pipeline, so it must only run on a cache miss.
    Parameter-policy handling, however, is request-local and must be replayed
    on every cache hit.  This object keeps those two concerns separate.
    """

    def __init__(self):
        self.unsupported = {}
        self.implicit_warnings = {}
        self.fatal_when_implicit = {}

    def reject(
        self,
        path,
        reason,
        *,
        implicit_warning=None,
        fatal_when_implicit=None,
    ):
        path = normalized_name(path)
        self.unsupported[path] = str(reason)
        if implicit_warning:
            self.implicit_warnings[path] = str(implicit_warning)
        if fatal_when_implicit:
            self.fatal_when_implicit[path] = fatal_when_implicit


class DiffusersPipelineEntry:
    def __init__(
        self,
        pipeline,
        torch,
        device,
        dtype,
        offload_metadata,
        configuration_outcome,
        model_load_seconds,
        configuration_warnings=None,
    ):
        self.pipeline = pipeline
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.offload_metadata = dict(offload_metadata)
        self.configuration_outcome = configuration_outcome
        self.model_load_seconds = float(model_load_seconds)
        self.configuration_warnings = list(configuration_warnings or [])


class TransformersAudioEntry:
    """One cached Transformers pipeline or processor/model pair."""

    def __init__(
        self,
        torch,
        device,
        dtype,
        model_load_seconds,
        *,
        adapter,
        pipeline_task,
        pipeline=None,
        processor=None,
        model=None,
    ):
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.model_load_seconds = float(model_load_seconds)
        self.adapter = str(adapter)
        self.pipeline_task = str(pipeline_task)
        self.pipeline = pipeline
        self.processor = processor
        self.model = model


def cleanup_torch_allocator(torch, device):
    gc.collect()
    try:
        if device == "cuda" and bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()
        elif device == "mps":
            empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
    except Exception:
        # Eviction is best-effort. A cleanup failure must not poison the
        # resident protocol or hide the original inference result/error.
        pass


def cleanup_diffusers_pipeline_entry(entry):
    """Release a cached pipeline and return allocator caches to the runtime."""
    if entry is None:
        return
    pipeline = entry.pipeline
    entry.pipeline = None
    del pipeline
    cleanup_torch_allocator(entry.torch, entry.device)


def cleanup_transformers_audio_entry(entry):
    """Release all references owned by a Transformers audio cache entry."""
    if entry is None:
        return
    entry.pipeline = None
    entry.processor = None
    entry.model = None
    cleanup_torch_allocator(entry.torch, entry.device)


def cleanup_media_pipeline_entry(entry):
    if isinstance(entry, TransformersAudioEntry):
        cleanup_transformers_audio_entry(entry)
    else:
        cleanup_diffusers_pipeline_entry(entry)


class DiffusersPipelineCache:
    """Synchronous bounded LRU retained under its original public name."""

    def __init__(self, max_size=1, cleanup=cleanup_diffusers_pipeline_entry):
        try:
            max_size = int(max_size)
        except (TypeError, ValueError):
            fail("invalid_configuration", "pipeline cache size must be an integer")
        if max_size < 0:
            fail("invalid_configuration", "pipeline cache size must not be negative")
        self.max_size = max_size
        self._cleanup = cleanup
        self._entries = OrderedDict()

    @property
    def enabled(self):
        return self.max_size > 0

    def __len__(self):
        return len(self._entries)

    def get(self, key):
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._entries[key] = entry
        return entry

    def prepare_for_load(self, key):
        """Make room before loading, avoiding simultaneous old/new weights."""
        if not self.enabled or key in self._entries:
            return
        while len(self._entries) >= self.max_size:
            _old_key, entry = self._entries.popitem(last=False)
            self._cleanup(entry)

    def put(self, key, entry):
        if not self.enabled:
            return False
        previous = self._entries.pop(key, None)
        if previous is not None and previous is not entry:
            self._cleanup(previous)
        self._entries[key] = entry
        while len(self._entries) > self.max_size:
            _old_key, evicted = self._entries.popitem(last=False)
            self._cleanup(evicted)
        return True

    def evict(self, key):
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._cleanup(entry)
            return True
        return False

    def clear(self):
        entries = list(self._entries.values())
        self._entries.clear()
        for entry in entries:
            self._cleanup(entry)


class MediaPipelineCache(DiffusersPipelineCache):
    """Shared resident LRU for Diffusers and Transformers media weights."""

    def __init__(self, max_size=1):
        super().__init__(max_size, cleanup=cleanup_media_pipeline_entry)


class CompanionRuntime:
    """State that exists only for the long-lived ``serve`` transport."""

    def __init__(self, pipeline_cache_size=None):
        if pipeline_cache_size is None:
            pipeline_cache_size = os.environ.get(PIPELINE_CACHE_SIZE_ENV, "1")
        # One global bound prevents a Diffusers pipeline and a Transformers
        # model from independently occupying the full resident-cache budget.
        self.pipeline_cache = MediaPipelineCache(pipeline_cache_size)

    def close(self):
        self.pipeline_cache.clear()


def normalized_name(value):
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalized_parameters(payload):
    value = payload.get("effective_parameters", payload.get("parameters", {}))
    if value is None:
        return {}
    if not isinstance(value, dict):
        fail("invalid_request", "effective_parameters must be a JSON object")
    normalized = {}
    for key, item in value.items():
        if isinstance(item, dict) and "value" in item:
            item = item["value"]
        # EffectiveInferenceRequest intentionally carries every schema path so
        # diagnostics can explain where values came from.  Unset paths are
        # serialized as JSON null, which must retain its "not supplied"
        # meaning inside adapters instead of masking their local defaults.
        if item is None:
            continue
        name = normalized_name(key)
        normalized[name] = item
        # Werk's canonical schema uses dotted paths (for example
        # ``image.width``).  Pipeline adapters consume the task-local leaf
        # while the full path remains available for diagnostics.
        if "." in name:
            normalized.setdefault(name.rsplit(".", 1)[1], item)
    if payload.get("prompt") is not None:
        normalized["prompt"] = payload["prompt"]
    if payload.get("negative_prompt") is not None:
        normalized["negative_prompt"] = payload["negative_prompt"]
    return normalized


def explicit_parameter_paths(payload):
    raw = payload.get("explicit_parameters", [])
    if raw is None:
        return set()
    if not isinstance(raw, list):
        fail("invalid_request", "explicit_parameters must be a JSON array of parameter paths")
    paths = set()
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            fail(
                "invalid_request",
                "explicit_parameters entries must be non-empty strings",
            )
        paths.add(normalized_name(value))
    return paths


def requested_parameter_policy(payload, parameters):
    value = payload.get("parameter_policy")
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if value is None:
        value = parameters.get("routing.parameter_policy")
    if value is None:
        value = parameters.get("parameter_policy")
    policy = normalized_name(value or "strict")
    if policy not in {"strict", "warn", "permissive"}:
        fail(
            "invalid_parameter",
            "parameter_policy must be strict, warn, or permissive",
        )
    return policy


class ExplicitParameterGuard:
    def __init__(self, payload, task, adapter, parameters):
        self.task = task
        self.adapter = adapter
        self.policy = requested_parameter_policy(payload, parameters)
        self.explicit = explicit_parameter_paths(payload)
        if payload.get("negative_prompt") is not None:
            self.explicit.add("negative_prompt")
        if task in ASR_TASKS and payload.get("prompt") is not None:
            self.explicit.add("stt.initial_prompt")
        self.unsupported = {}
        self.warnings = []

    def validate_supported(self, supported):
        unsupported = {
            path: (
                f"the {self.adapter} adapter for task '{self.task}' does not "
                "consume this parameter"
            )
            for path in self.explicit
            if path not in supported
        }
        self._record(unsupported)

    def reject(self, path, reason):
        path = normalized_name(path)
        if path in self.explicit:
            self._record({path: reason})

    def reject_overridden(self, path, winner):
        self.reject(
            path,
            f"it is overridden by explicit parameter '{normalized_name(winner)}'",
        )

    def metadata(self):
        return {
            "policy": self.policy,
            "explicit_parameters": sorted(self.explicit),
            "unsupported_explicit_parameters": sorted(self.unsupported),
            "unsupported_reasons": dict(sorted(self.unsupported.items())),
        }

    def without_unsupported(self, parameters):
        sanitized = dict(parameters)
        for path in self.unsupported:
            sanitized.pop(path, None)
            if "." in path:
                sanitized.pop(path.rsplit(".", 1)[1], None)
            if path == "stt.initial_prompt" and self.task in ASR_TASKS:
                sanitized.pop("prompt", None)
        return sanitized

    def _record(self, unsupported):
        added = {}
        for path, reason in unsupported.items():
            if path not in self.unsupported:
                self.unsupported[path] = reason
                added[path] = reason
        if not added:
            return
        if self.policy == "strict":
            paths = sorted(added)
            fail(
                "unsupported_parameter",
                "explicit parameters are not supported by the selected media adapter",
                {
                    "adapter": self.adapter,
                    "task": self.task,
                    "parameters": paths,
                    "reasons": {path: added[path] for path in paths},
                },
            )
        if self.policy == "warn":
            for path in sorted(added):
                self.warnings.append(
                    f"explicit parameter '{path}' is unsupported by the "
                    f"{self.adapter} adapter and was ignored: {added[path]}"
                )


def has_diffusers_pipeline_manifest(root):
    manifest = read_json_file(root / "model_index.json")
    return bool(str(manifest.get("_class_name") or "").strip())


def has_transformers_model_config(root):
    if not root.is_dir():
        return False
    config = read_json_file(root / "config.json")
    return bool(config)


def qwen3_tts_model_variant(root):
    """Return the declared Qwen3-TTS variant for a local repository.

    Qwen3-TTS is not a generic Transformers text-to-audio pipeline.  Detect it
    from its local config before the generic adapter gets a chance to claim
    the repository.  Repository names are deliberately ignored.
    """
    if not root.is_dir():
        return None
    config = read_json_file(root / "config.json")
    if normalized_name(config.get("model_type")) != "qwen3_tts":
        return None
    return normalized_name(config.get("tts_model_type")) or "unknown"


def is_qwen3_tts_voice_design(root):
    return qwen3_tts_model_variant(root) == "voice_design"


def architecture_adapter_registration(config, task=None):
    """Return an exact adapter registration and any recognized special model.

    The second result deliberately survives a task or variant mismatch. It
    lets routing block generic pipelines for a known special architecture even
    when the companion has not implemented that particular operation yet.
    """
    if not isinstance(config, dict):
        return None, None
    model_type = normalized_name(config.get("model_type"))
    architectures = config.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = [architectures]
    architecture_names = {
        normalized_name(architecture)
        for architecture in architectures
        if str(architecture or "").strip()
    }
    requested_task = normalized_name(task)
    identity_registration = None
    for registration in ARCHITECTURE_ADAPTER_REGISTRY:
        model_types = registration.get("model_types", ())
        registered_architectures = registration.get("architectures", ())
        identity_matches = (
            bool(model_type) and model_type in model_types
        ) or bool(architecture_names & set(registered_architectures))
        if not identity_matches:
            continue
        if identity_registration is None:
            identity_registration = registration
        tasks = registration.get("tasks", ())
        if requested_task and requested_task not in tasks:
            continue
        variant_field = registration.get("variant_field")
        variants = registration.get("variants", ())
        if variant_field and variants:
            variant = normalized_name(config.get(variant_field))
            if variant not in variants:
                continue
        return registration, identity_registration
    return None, identity_registration


def architecture_adapter_for_root(root, task=None):
    if not root.is_dir():
        return None, None
    return architecture_adapter_registration(
        read_json_file(root / "config.json"),
        task,
    )


def architecture_adapter_by_name(adapter):
    for registration in ARCHITECTURE_ADAPTER_REGISTRY:
        if registration.get("adapter") == adapter:
            return registration
    return None


def missing_adapter_dependencies(registration, dependencies):
    return [
        name
        for name in registration.get("dependencies", ())
        if not bool(dependencies.get(name, {}).get("available", False))
    ]


def architecture_backend_hint(probe, task, dependencies):
    matched, recognized = architecture_adapter_registration(
        probe.get("config", {}),
        task,
    )
    registration = matched or recognized
    if registration is None:
        return None
    missing = missing_adapter_dependencies(registration, dependencies)
    adapter_supported = matched is not None
    backend_available = not missing
    return {
        "required_backend": registration["required_backend"],
        # Do not suggest an installation as a fix for a variant/task whose
        # execution adapter has not been implemented in this companion.
        "install_command": (
            registration["install_command"]
            if adapter_supported and not backend_available
            else None
        ),
        "backend_available": backend_available,
        "fallback_possible": bool(registration.get("fallback_possible", False)),
        "architecture_adapter_supported": adapter_supported,
        "missing_dependencies": missing,
    }


def execution_adapter(model_path, task):
    root = model_path if model_path.is_dir() else model_path.parent
    registered_adapter, special_model = architecture_adapter_for_root(root, task)
    if special_model is not None:
        return (
            registered_adapter.get("adapter")
            if registered_adapter is not None
            else None
        )
    if task in IMAGE_TASKS | VIDEO_TASKS:
        # A Diffusers directory is identified by its component manifest.  A
        # generic config.json alone may describe a native checkpoint layout
        # (for example a framework-specific transformer shard) which
        # DiffusionPipeline.from_pretrained cannot execute.  Single-file
        # pipelines remain model-dependent and are validated while loading.
        if model_path.is_file() or has_diffusers_pipeline_manifest(root):
            return "diffusers"
        return None
    if task in AUDIO_GENERATION_TASKS:
        if has_diffusers_pipeline_manifest(root):
            return "diffusers_audio"
        if has_transformers_model_config(root):
            return "transformers_audio"
        return None
    if task in TTS_TASKS:
        if has_transformers_model_config(root):
            return "transformers_tts"
    if task in ASR_TASKS and has_transformers_model_config(root):
        return "transformers_asr"
    if task in AUDIO_CLASSIFICATION_TASKS and has_transformers_model_config(root):
        return "transformers_audio_classification"
    if task in AUDIO_TEXT_TASKS and has_transformers_model_config(root):
        return "transformers_audio_text"
    if task in AUDIO_EMBEDDING_TASKS and has_transformers_model_config(root):
        return "transformers_audio_embedding"
    return None


def selected_offload_request(parameters):
    if bool(parameters.get("_werk_enable_sequential_offload")):
        return "sequential_cpu"
    if bool(parameters.get("_werk_enable_component_offload")):
        return "component"
    if bool(parameters.get("_werk_enable_cpu_offload")):
        return "model_cpu"
    return "none"


def validate_adapter_offload(adapter, parameters):
    offload_request = selected_offload_request(parameters)
    if offload_request != "none" and adapter not in {"diffusers", "diffusers_audio"}:
        fail(
            "backend_configuration_failed",
            "selected CPU offload is unsupported by the chosen media adapter",
            {
                "adapter": adapter,
                "offload_request": offload_request,
            },
        )
    return offload_request


def supported_explicit_parameters(task, adapter):
    supported = set(ORCHESTRATOR_PARAMETERS)
    supported.update(TORCH_RUNTIME_PARAMETERS)
    if adapter in {"diffusers", "diffusers_audio"}:
        supported.update(DIFFUSERS_ROUTING_PARAMETERS)
        supported.add("negative_prompt")
    if task in IMAGE_TASKS:
        supported.update(IMAGE_ADAPTER_PARAMETERS)
    elif task in VIDEO_TASKS:
        supported.update(VIDEO_ADAPTER_PARAMETERS)
    elif adapter == "diffusers_audio":
        supported.update(DIFFUSERS_AUDIO_PARAMETERS)
    elif adapter == "transformers_audio":
        supported.update(TRANSFORMERS_AUDIO_PARAMETERS)
    elif adapter == "transformers_tts":
        supported.update(TTS_ADAPTER_PARAMETERS)
    elif adapter == QWEN3_TTS_ADAPTER:
        supported.update(QWEN3_TTS_VOICE_DESIGN_PARAMETERS)
    elif adapter == "transformers_asr":
        supported.update(ASR_ADAPTER_PARAMETERS)
    elif adapter == "transformers_audio_classification":
        supported.update(AUDIO_CLASSIFICATION_PARAMETERS)
    elif adapter == "transformers_audio_text":
        supported.update(AUDIO_TEXT_PARAMETERS)
    elif adapter == "transformers_audio_embedding":
        supported.update(AUDIO_EMBEDDING_PARAMETERS)
    return supported


def input_values(payload, parameters):
    values = payload.get("inputs", {})
    if values is None:
        values = {}
    if not isinstance(values, dict):
        fail("invalid_request", "inputs must be a JSON object")
    merged = {
        normalized_name(key): value
        for key, value in values.items()
        if value is not None
    }
    for key in (
        "source",
        "input",
        "image",
        "input_image",
        "initial_image",
        "final_image",
        "mask",
        "mask_image",
        "mask_video",
        "source_video",
        "input_video",
        "video",
        "input_audio",
        "source_audio",
        "audio",
        "reference_audio",
    ):
        if payload.get(key) is not None:
            merged[key] = payload[key]
        elif parameters.get(key) is not None and key not in merged:
            merged[key] = parameters[key]
    if "video" in merged:
        merged.setdefault("source_video", merged["video"])
    if "audio" in merged:
        merged.setdefault("source_audio", merged["audio"])
    return merged


def validate_adapter_inputs(task, adapter, inputs):
    allowed = set()
    if task in IMAGE_TASKS:
        allowed.update(
            {
                "image",
                "input_image",
                "initial_image",
                "final_image",
                "mask",
                "mask_image",
            }
        )
    elif task in VIDEO_TASKS:
        allowed.update(
            {
                "image",
                "input_image",
                "initial_image",
                "final_image",
                "video",
                "source_video",
                "input_video",
                "mask",
                "mask_video",
            }
        )
    elif adapter in {
        "transformers_asr",
        "transformers_audio_classification",
        "transformers_audio_text",
        "transformers_audio_embedding",
    }:
        allowed.update(
            {
                "audio",
                "source_audio",
                "input_audio",
                "source",
                "input",
            }
        )
    unsupported = sorted(set(inputs) - allowed)
    if unsupported:
        fail(
            "unsupported_parameter",
            f"task '{task}' does not support the supplied media input roles",
            {"input_roles": unsupported},
        )


def required_string(value, name):
    if not isinstance(value, str) or not value.strip():
        fail("invalid_request", f"{name} must be a non-empty string")
    return value.strip()


def prompt_with_lyrics(prompt, lyrics):
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    lyrics = lyrics.strip() if isinstance(lyrics, str) else ""
    if not lyrics:
        return prompt or None
    if not prompt:
        return lyrics
    lyrics_block = f"Lyrics:\n{lyrics}"
    if prompt == lyrics or prompt == lyrics_block or prompt.endswith(f"\n\n{lyrics_block}"):
        return prompt
    return f"{prompt}\n\n{lyrics_block}"


def local_model_path(payload):
    raw = required_string(payload.get("model_path"), "model_path")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        fail("model_not_found", f"local model path does not exist: {path}")
    return path


def output_directory(payload):
    raw = required_string(payload.get("output_dir"), "output_dir")
    path = Path(raw).expanduser().resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        fail("output_error", f"cannot create output directory: {path}", str(error))
    if not path.is_dir():
        fail("output_error", f"output path is not a directory: {path}")
    return path


def module_status(module_name, distribution=None):
    available = importlib.util.find_spec(module_name) is not None
    version = None
    detail = None
    if available:
        try:
            version = importlib.metadata.version(distribution or module_name)
        except Exception:
            detail = "module found; version unavailable"
    else:
        detail = "not installed (optional)"
    return {
        "available": available,
        "version": version,
        "detail": detail,
    }


def importable_module_status(module_name, distribution=None, required_attribute=None):
    """Report whether an optional adapter module can actually be imported.

    ``find_spec`` alone is insufficient for provider packages with pinned
    transitive dependencies: a partially installed qwen-tts distribution can
    be discoverable while failing during import.  Capability and model probes
    must not advertise such an environment as executable.
    """
    status = module_status(module_name, distribution)
    if not status["available"]:
        return status
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        status["available"] = False
        status["detail"] = f"installed but import failed: {error}"
        return status
    if required_attribute and not callable(getattr(module, required_attribute, None)):
        status["available"] = False
        status["detail"] = (
            f"installed module does not expose required API '{required_attribute}'"
        )
    return status


def dependency_snapshot():
    dependencies = {
        "torch": module_status("torch"),
        "diffusers": module_status("diffusers"),
        "transformers": module_status("transformers"),
        "qwen_tts": importable_module_status(
            "qwen_tts",
            "qwen-tts",
            "Qwen3TTSModel",
        ),
        "PIL": module_status("PIL", "Pillow"),
        "numpy": module_status("numpy"),
        "soundfile": module_status("soundfile"),
        "scipy": module_status("scipy"),
        "librosa": module_status("librosa"),
        "torchaudio": module_status("torchaudio"),
        "imageio": module_status("imageio"),
        "imageio_ffmpeg": module_status("imageio_ffmpeg", "imageio-ffmpeg"),
        "av": module_status("av"),
    }
    ffmpeg = shutil.which("ffmpeg")
    dependencies["ffmpeg"] = {
        "available": ffmpeg is not None,
        "version": None,
        "detail": ffmpeg or "ffmpeg executable not found (optional)",
    }
    if not dependencies["qwen_tts"]["available"]:
        detail = dependencies["qwen_tts"].get("detail") or "not available"
        dependencies["qwen_tts"]["detail"] = (
            f"{detail}; run 'werk backend install qwen-tts' or select an "
            f"isolated interpreter with {QWEN3_TTS_PYTHON_ENV}"
        )
    return dependencies


def video_encoder_ready(dependencies):
    # ImageIO alone does not guarantee that an MP4 writer plugin is present.
    # PyAV, a system ffmpeg, or imageio-ffmpeg's bundled executable each give
    # the companion a concrete encoder path.
    return any(
        bool(dependencies.get(name, {}).get("available"))
        for name in ("av", "imageio_ffmpeg", "ffmpeg")
    )


def video_decoder_ready(dependencies):
    if bool(dependencies.get("av", {}).get("available")):
        return True
    return bool(
        dependencies.get("imageio", {}).get("available")
    ) and bool(
        dependencies.get("imageio_ffmpeg", {}).get("available")
    )


def audio_decoder_ready(dependencies):
    return bool(
        dependencies.get("soundfile", {}).get("available")
    ) or bool(
        dependencies.get("ffmpeg", {}).get("available")
    )


def require_module(module_name, distribution=None, purpose=None):
    try:
        return importlib.import_module(module_name)
    except Exception as error:
        package = distribution or module_name
        suffix = f" for {purpose}" if purpose else ""
        fail(
            "missing_dependency",
            f"optional dependency '{package}' is required{suffix}",
            str(error),
        )


def torch_hip_version(torch):
    return getattr(getattr(torch, "version", None), "hip", None)


def torch_gpu_architectures(torch):
    """Return architectures reported by the active devices, not wheel targets."""
    architectures = []
    try:
        device_count = int(torch.cuda.device_count())
    except Exception:
        return architectures
    for index in range(device_count):
        try:
            properties = torch.cuda.get_device_properties(index)
        except Exception:
            continue
        for name in ("gcnArchName", "gcn_arch_name"):
            value = getattr(properties, name, None)
            if value:
                architectures.append(str(value))
                break
    return architectures


def torch_is_strix_halo(torch):
    return any(
        "gfx1151" in architecture.lower()
        for architecture in torch_gpu_architectures(torch)
    )


def accelerator_snapshot():
    snapshot = {
        "cpu": {
            "available": True,
            "version": None,
            "detail": "CPU execution is available",
        }
    }
    unavailable = {
        "available": False,
        "version": None,
        "detail": "PyTorch is not available",
    }
    snapshot["cuda"] = dict(unavailable)
    snapshot["rocm"] = dict(unavailable)
    snapshot["mps"] = dict(unavailable)
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        detail = f"PyTorch could not be imported: {error}"
        for name in ("cuda", "rocm", "mps"):
            snapshot[name]["detail"] = detail
        return snapshot

    torch_version = str(getattr(torch, "__version__", "unknown"))
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    hip_version = torch_hip_version(torch)
    try:
        torch_gpu_available = bool(torch.cuda.is_available())
    except Exception as error:
        torch_gpu_available = False
        gpu_error = str(error)
    else:
        gpu_error = None

    if torch_gpu_available:
        try:
            device_count = int(torch.cuda.device_count())
            names = [
                str(torch.cuda.get_device_name(index))
                for index in range(device_count)
            ]
            device_detail = ", ".join(names) if names else "GPU available"
        except Exception as error:
            device_count = 0
            device_detail = f"GPU available; device detail unavailable: {error}"
        architectures = torch_gpu_architectures(torch)
        architecture_detail = (
            f"; architecture(s): {', '.join(architectures)}"
            if architectures
            else ""
        )
        strix_detail = (
            "; Strix Halo/gfx1151 detected; use FP16 (the validated ROCm precision) or auto"
            if hip_version and torch_is_strix_halo(torch)
            else ""
        )
        kind = "rocm" if hip_version else "cuda"
        version = hip_version if hip_version else cuda_version
        snapshot[kind] = {
            "available": True,
            "version": str(version) if version else None,
            "detail": (
                f"torch {torch_version}; {device_count} device(s): {device_detail}"
                f"{architecture_detail}{strix_detail}"
            ),
        }
        other = "cuda" if kind == "rocm" else "rocm"
        snapshot[other]["detail"] = (
            f"torch {torch_version} uses {kind.upper()}, not {other.upper()}"
        )
    else:
        detail = (
            f"torch {torch_version} reports no GPU"
            if gpu_error is None
            else f"torch {torch_version} GPU check failed: {gpu_error}"
        )
        if cuda_version:
            detail += f"; CUDA build {cuda_version}"
        if hip_version:
            detail += f"; ROCm build {hip_version}"
        snapshot["cuda"]["version"] = (
            str(cuda_version) if cuda_version else None
        )
        snapshot["rocm"]["version"] = (
            str(hip_version) if hip_version else None
        )
        snapshot["cuda"]["detail"] = detail
        snapshot["rocm"]["detail"] = detail

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    try:
        mps_available = (
            mps_backend is not None and bool(mps_backend.is_available())
        )
    except Exception as error:
        mps_available = False
        mps_detail = f"torch {torch_version} MPS check failed: {error}"
    else:
        mps_detail = (
            f"torch {torch_version} reports MPS available"
            if mps_available
            else f"torch {torch_version} reports MPS unavailable"
        )
    snapshot["mps"] = {
        "available": mps_available,
        "version": None,
        "detail": mps_detail,
    }
    return snapshot


def command_health(_payload):
    return {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "companion_version": COMPANION_VERSION,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "offline": True,
        "dependencies": dependency_snapshot(),
        "accelerators": accelerator_snapshot(),
    }


def task_capability(task, available, runtime, reason=None):
    return {
        "task": task,
        "available": bool(available),
        "runtime": runtime,
        "model_dependent": True,
        "reason": reason,
    }


def command_capabilities(_payload):
    deps = dependency_snapshot()
    image_ready = (
        deps["torch"]["available"]
        and deps["diffusers"]["available"]
        and deps["PIL"]["available"]
    )
    video_ready = image_ready and video_encoder_ready(deps)
    transformers_audio_ready = (
        deps["torch"]["available"]
        and deps["transformers"]["available"]
        and deps["numpy"]["available"]
    )
    qwen3_tts_ready = (
        deps["torch"]["available"]
        and deps["numpy"]["available"]
        and deps.get("qwen_tts", {}).get("available", False)
    )
    transformers_audio_input_ready = (
        transformers_audio_ready and audio_decoder_ready(deps)
    )
    generative_audio_ready = (
        deps["torch"]["available"]
        and deps["numpy"]["available"]
        and (
            deps["transformers"]["available"]
            or deps["diffusers"]["available"]
        )
    )
    model_dependent_reason = (
        "adapter dependencies are available; exact model, pipeline, and "
        "parameter compatibility is confirmed when the concrete pipeline is loaded for execution"
    )
    capabilities = []
    for task in sorted(IMAGE_TASKS):
        capabilities.append(
            task_capability(
                task,
                image_ready,
                "diffusers",
                model_dependent_reason
                if image_ready
                else "requires torch, diffusers and Pillow",
            )
        )
    for task in sorted(VIDEO_TASKS):
        task_ready = video_ready and (
            task not in VIDEO_INPUT_TASKS or video_decoder_ready(deps)
        )
        capabilities.append(
            task_capability(
                task,
                task_ready,
                "diffusers",
                model_dependent_reason
                if task_ready
                else (
                    "requires torch, diffusers, Pillow, a video encoder and a video decoder"
                    if task in VIDEO_INPUT_TASKS
                    else "requires torch, diffusers, Pillow and a video encoder"
                ),
            )
        )
    for task in sorted(AUDIO_GENERATION_TASKS):
        capabilities.append(
            task_capability(
                task,
                generative_audio_ready,
                "diffusers-or-transformers",
                model_dependent_reason
                if generative_audio_ready
                else "requires torch, numpy, and either diffusers or transformers",
            )
        )
    for task in sorted(TTS_TASKS):
        tts_ready = transformers_audio_ready or qwen3_tts_ready
        if tts_ready:
            tts_reason = model_dependent_reason
            if not qwen3_tts_ready:
                tts_reason += (
                    "; Qwen3-TTS models additionally require an importable "
                    "qwen-tts package in the companion Python environment"
                )
        else:
            tts_reason = (
                "requires torch and numpy plus either transformers for generic "
                "TTS or qwen-tts for Qwen3-TTS"
            )
        capabilities.append(
            task_capability(
                task,
                tts_ready,
                "qwen3-tts-or-transformers",
                tts_reason,
            )
        )
    for task in sorted(AUDIO_SOURCE_TASKS - AUDIO_TEXT_TASKS):
        capabilities.append(
            task_capability(
                task,
                transformers_audio_input_ready,
                "transformers",
                model_dependent_reason
                if transformers_audio_input_ready
                else (
                    "requires torch, transformers, numpy, and either "
                    "soundfile or ffmpeg for local audio decoding"
                ),
            )
        )
    transformers_audio_text_ready = transformers_audio_input_ready
    for task in sorted(AUDIO_TEXT_TASKS):
        capabilities.append(
            task_capability(
                task,
                transformers_audio_text_ready,
                "transformers",
                model_dependent_reason
                if transformers_audio_text_ready
                else (
                    "requires torch, transformers, numpy, and either "
                    "soundfile or ffmpeg for local audio decoding"
                ),
            )
        )
    for task in sorted(DECLARED_UNSUPPORTED_TASKS):
        capabilities.append(
            task_capability(
                task,
                False,
                None,
                "the generic companion has no reliable local adapter for this task",
            )
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "offline": True,
        "capabilities": capabilities,
        "parameter_policy": {
            "default": "strict",
            "supported": ["strict", "warn", "permissive"],
            "scope": "explicit_parameters are checked against the selected task and adapter",
        },
        "input_modalities": ["text", "image", "video", "audio"],
        "output_modalities": ["image", "video", "audio", "text"],
    }


def read_json_file(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def file_inventory(root):
    if root.is_file():
        return [root]
    files = []
    for directory, names, file_names in os.walk(root, followlinks=False):
        names[:] = [
            name
            for name in names
            if name not in {".git", ".cache", "__pycache__", "outputs"}
        ]
        base = Path(directory)
        for name in file_names:
            files.append(base / name)
    return files


def model_probe(path):
    root = path if path.is_dir() else path.parent
    model_index = read_json_file(root / "model_index.json")
    config = read_json_file(root / "config.json")
    generation_config = read_json_file(root / "generation_config.json")
    files = file_inventory(path)
    names = {item.name.lower() for item in files}
    directories = {
        item.name
        for item in root.iterdir()
        if item.is_dir()
    } if root.is_dir() else set()

    if model_index:
        layout = "diffusers"
    elif config or "config.json" in names:
        layout = "transformers"
    elif path.is_file():
        layout = "single_file"
    else:
        layout = "custom"

    class_name = str(model_index.get("_class_name") or "")
    model_type = str(config.get("model_type") or "")
    architectures = config.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = [str(architectures)]
    pipeline_tag = str(
        config.get("pipeline_tag")
        or model_index.get("pipeline_tag")
        or generation_config.get("pipeline_tag")
        or ""
    )
    # Repository names are intentionally excluded. Probe claims come only
    # from the local model's declared pipeline/config classes and tags.
    searchable = " ".join(
        [class_name, model_type, pipeline_tag]
        + [str(item) for item in architectures]
    ).lower()
    normalized_searchable = normalized_name(searchable)
    normalized_pipeline_tag = normalized_name(pipeline_tag)
    architecture_names = [str(item).lower() for item in architectures]
    audio_backbone_hints = (
        "audio_spectrogram_transformer",
        "clap",
        "data2vec_audio",
        "hubert",
        "sew",
        "unispeech",
        "wav2vec2",
        "wavlm",
    )
    has_audio_backbone = any(
        word in normalized_searchable for word in audio_backbone_hints
    ) or normalized_name(model_type) == "ast"

    tasks = []
    if model_index:
        if any(word in searchable for word in ("video", "animatediff", "cogvideo", "wan")):
            tasks.extend(["video_generation", "image_to_video"])
        elif any(word in searchable for word in ("audio", "music", "audioldm")):
            tasks.extend(["audio_generation", "music_generation"])
        else:
            tasks.append("image_generation")
            if "inpaint" in searchable:
                tasks.append("image_inpainting")
            if any(word in searchable for word in ("img2img", "image2image")):
                tasks.append("image_editing")
    if normalized_pipeline_tag == "automatic_speech_recognition" or any(
        word in searchable
        for word in (
            "whisper",
            "speech_to_text",
            "speechrecognition",
            "forctc",
        )
    ):
        tasks.append("speech_to_text")
    task_to_id = generation_config.get("task_to_id")
    if (
        isinstance(task_to_id, dict)
        and "translate" in task_to_id
    ) or normalized_pipeline_tag in {
        "speech_translation",
        "automatic_speech_translation",
    }:
        tasks.append("speech_translation")
    if any(
        word in searchable
        for word in (
            "qwen3_tts",
            "qwen3tts",
            "speecht5",
            "bark",
            "vits",
            "fastspeech",
            "parler",
            "text-to-speech",
            "texttospeech",
        )
    ):
        tasks.append("text_to_speech")
    if any(
        word in searchable
        for word in ("musicgen", "audiogen", "text-to-audio", "text_to_audio")
    ):
        tasks.extend(["audio_generation", "music_generation"])
    classification_architecture = any(
        "foraudioclassification" in architecture
        or (
            "forsequenceclassification" in architecture
            and has_audio_backbone
        )
        for architecture in architecture_names
    )
    if normalized_pipeline_tag in {
        "audio_classification",
        "zero_shot_audio_classification",
    } or classification_architecture:
        tasks.append("audio_classification")
    has_audio_config = any(
        name in config
        for name in (
            "audio_config",
            "audio_encoder_config",
            "audio_tokenizer_config",
        )
    )
    multimodal_generation_architecture = any(
        "forconditionalgeneration" in architecture
        or "formultimodalgeneration" in architecture
        for architecture in architecture_names
    )
    if normalized_pipeline_tag in {
        "any_to_any",
        "audio_text_to_text",
        "audio_captioning",
    } or (
        has_audio_config
        and multimodal_generation_architecture
    ):
        tasks.extend(["audio_captioning", "audio_understanding"])
    if normalized_pipeline_tag in {
        "audio_feature_extraction",
        "audio_embedding",
    } or has_audio_backbone:
        tasks.append("audio_embedding")
    if not tasks and config:
        # A generic Transformers repository remains cataloguable. Execution is
        # intentionally model-dependent instead of guessing a text model is media.
        tasks = []

    components = sorted(
        name
        for name in directories
        if normalized_name(name)
        in {
            "transformer",
            "unet",
            "vae",
            "text_encoder",
            "text_encoder_2",
            "tokenizer",
            "tokenizer_2",
            "scheduler",
            "encoder",
            "decoder",
            "vocoder",
            "feature_extractor",
            "controlnet",
            "adapter",
            "speech_tokenizer",
        }
    )
    weight_extensions = {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".onnx",
        ".gguf",
        ".ckpt",
        ".npz",
    }
    weight_files = [
        item for item in files if item.suffix.lower() in weight_extensions
    ]
    return {
        "layout": layout,
        "class_name": class_name or None,
        "model_type": model_type or None,
        "architectures": architectures,
        "pipeline_tag": pipeline_tag or None,
        "model_variant": (
            normalized_name(config.get("tts_model_type")) or None
            if normalized_name(model_type) == "qwen3_tts"
            else None
        ),
        "tasks": sorted(set(tasks)),
        "components": components,
        "file_count": len(files),
        "weight_file_count": len(weight_files),
        "weight_payload_bytes": sum(safe_size(item) for item in weight_files),
        "config": config,
    }


def task_dependency_ready(task, deps):
    if task in IMAGE_TASKS:
        return (
            deps["torch"]["available"]
            and deps["diffusers"]["available"]
            and deps["PIL"]["available"]
        ), "requires torch, diffusers and Pillow"
    if task in VIDEO_TASKS:
        base = (
            deps["torch"]["available"]
            and deps["diffusers"]["available"]
            and deps["PIL"]["available"]
        )
        encoder = video_encoder_ready(deps)
        decoder = task not in VIDEO_INPUT_TASKS or video_decoder_ready(deps)
        reason = (
            "requires torch, diffusers, Pillow, a video encoder and a video decoder"
            if task in VIDEO_INPUT_TASKS
            else "requires torch, diffusers, Pillow and a video encoder"
        )
        return base and encoder and decoder, reason
    if task in AUDIO_GENERATION_TASKS:
        return (
            deps["torch"]["available"]
            and deps["numpy"]["available"]
            and (
                deps["diffusers"]["available"]
                or deps["transformers"]["available"]
            )
        ), "requires torch, numpy, and either diffusers or transformers"
    if task in TTS_TASKS:
        return (
            deps["torch"]["available"]
            and deps["transformers"]["available"]
            and deps["numpy"]["available"]
        ), "requires torch, transformers and numpy"
    if task in AUDIO_SOURCE_TASKS:
        ready = (
            deps["torch"]["available"]
            and deps["transformers"]["available"]
            and deps["numpy"]["available"]
            and audio_decoder_ready(deps)
        )
        return ready, (
            "requires torch, transformers, numpy, and either soundfile or "
            "ffmpeg for local audio decoding"
        )
    return False, "task has no generic companion adapter"


def adapter_dependency_ready(task, adapter, deps):
    registration = architecture_adapter_by_name(adapter)
    if registration is not None:
        ready = not missing_adapter_dependencies(registration, deps)
        return ready, registration["dependency_reason"]
    if adapter == "diffusers_audio":
        return (
            deps["torch"]["available"]
            and deps["numpy"]["available"]
            and deps["diffusers"]["available"]
        ), "requires torch, numpy and diffusers for this Diffusers repository"
    if adapter == "transformers_audio":
        return (
            deps["torch"]["available"]
            and deps["numpy"]["available"]
            and deps["transformers"]["available"]
        ), "requires torch, numpy and transformers for this Transformers repository"
    return task_dependency_ready(task, deps)


def generic_dependency_readiness(task, dependencies, adapter=None):
    """Return mandatory dependencies and unsatisfied alternative groups.

    ``missing_dependencies`` remains the backward-compatible flat field, but
    now contains only packages which are individually mandatory for the task.
    Alternative execution routes are represented separately.  Each ``any_of``
    entry is one viable route, whose ``all_of`` packages must be available
    together.  This can express both simple choices such as
    diffusers-or-transformers and compound routes such as
    PyAV-or-(imageio-and-imageio-ffmpeg) without implying that every listed
    package must be installed.
    """

    def available(name):
        return bool(dependencies.get(name, {}).get("available", False))

    missing = []
    missing_groups = []

    def require(*names):
        missing.extend(name for name in names if not available(name))

    def require_any_of(purpose, *routes):
        normalized_routes = [tuple(route) for route in routes]
        if any(all(available(name) for name in route) for route in normalized_routes):
            return
        missing_groups.append(
            {
                "purpose": purpose,
                "any_of": [
                    {"all_of": list(route)} for route in normalized_routes
                ],
            }
        )

    if task in IMAGE_TASKS:
        require("torch", "diffusers", "PIL")
    elif task in VIDEO_TASKS:
        require("torch", "diffusers", "PIL")
        require_any_of(
            "video_encoder",
            ("av",),
            ("ffmpeg",),
            ("imageio_ffmpeg",),
        )
        if task in VIDEO_INPUT_TASKS:
            require_any_of(
                "video_decoder",
                ("av",),
                ("imageio", "imageio_ffmpeg"),
            )
    elif task in AUDIO_GENERATION_TASKS:
        require("torch", "numpy")
        if adapter == "diffusers_audio":
            require("diffusers")
        elif adapter == "transformers_audio":
            require("transformers")
        else:
            require_any_of(
                "audio_generation_framework",
                ("diffusers",),
                ("transformers",),
            )
    elif task in TTS_TASKS:
        require("torch", "transformers", "numpy")
    elif task in AUDIO_SOURCE_TASKS:
        require("torch", "transformers", "numpy")
        require_any_of(
            "audio_decoder",
            ("soundfile",),
            ("ffmpeg",),
        )

    return list(dict.fromkeys(missing)), missing_groups


def model_probe_readiness(
    requested_task,
    supported,
    adapter,
    detail,
    backend_hint,
    dependencies,
):
    """Build the additive, machine-readable readiness result for a probe."""

    architecture_not_implemented = bool(
        backend_hint is not None
        and backend_hint.get("architecture_adapter_supported") is False
    )
    task_not_implemented = requested_task in DECLARED_UNSUPPORTED_TASKS

    if supported is True:
        status = "available"
    elif task_not_implemented or architecture_not_implemented:
        status = "not_implemented"
    elif backend_hint and backend_hint.get("install_command"):
        status = "installable"
    else:
        status = "unavailable"

    if backend_hint is not None:
        required_backend = backend_hint.get("required_backend")
        missing_dependencies = list(
            backend_hint.get("missing_dependencies") or []
        )
        missing_dependency_groups = []
    else:
        required_backend = None
        (
            missing_dependencies,
            missing_dependency_groups,
        ) = generic_dependency_readiness(
            requested_task,
            dependencies,
            adapter,
        )

    # An installation command is authoritative only when it came from the
    # existing explicit architecture-backend hint and that adapter is actually
    # implemented.  Generic dependency failures and unimplemented variants
    # must never manufacture or repeat one.
    install_command = (
        backend_hint.get("install_command")
        if status == "installable" and backend_hint is not None
        else None
    )

    return {
        "status": status,
        "detail": detail,
        "adapter": adapter,
        "required_backend": required_backend,
        "install_command": install_command,
        "fallback_backend": None,
        "missing_dependencies": missing_dependencies,
        "missing_dependency_groups": missing_dependency_groups,
    }


def command_probe_model(payload):
    path = local_model_path(payload)
    probe = model_probe(path)
    requested_task = normalized_name(payload.get("task"))
    dependencies = dependency_snapshot()
    supported = None
    reasons = []
    adapter = None
    if requested_task:
        adapter = execution_adapter(path, requested_task)
        ready, dependency_reason = adapter_dependency_ready(
            requested_task,
            adapter,
            dependencies,
        )
        advertised_tasks = set(probe["tasks"])
        same_media_family = (
            requested_task in IMAGE_TASKS
            and bool(advertised_tasks & IMAGE_TASKS)
        ) or (
            requested_task in VIDEO_TASKS
            and bool(advertised_tasks & VIDEO_TASKS)
        ) or (
            requested_task in AUDIO_CLASSIFICATION_TASKS
            and bool(advertised_tasks & AUDIO_CLASSIFICATION_TASKS)
        ) or (
            requested_task in AUDIO_TEXT_TASKS
            and bool(advertised_tasks & AUDIO_TEXT_TASKS)
        )
        recognized = requested_task in advertised_tasks or (
            same_media_family
            and probe["layout"] in {"diffusers", "transformers"}
        )
        supported = ready and recognized and adapter is not None
        if not ready:
            reasons.append(dependency_reason)
        if not recognized:
            reasons.append(
                "model metadata does not advertise this task; support remains model-dependent"
            )
        if adapter is None:
            reasons.append(
                "the local repository layout has no executable companion adapter; "
                "use a Diffusers pipeline repository, a supported single-file model, "
                "or configure another media backend"
            )
    if reasons:
        detail = "; ".join(reasons)
    elif requested_task and supported:
        detail = (
            "local dependencies and model metadata match the requested task; "
            "exact pipeline and parameter compatibility remains model-dependent"
        )
    elif requested_task:
        detail = "requested task support could not be established"
    else:
        detail = "model metadata probe completed; no task was requested"
    backend_hint = architecture_backend_hint(
        probe,
        requested_task or None,
        dependencies,
    )
    result = {
        "model_path": str(path),
        "supported": supported,
        "adapter": adapter,
        "reasons": reasons,
        "detail": detail,
        "probe": {
            key: value
            for key, value in probe.items()
            if key != "config"
        },
        "offline": True,
    }
    result.update(
        backend_hint
        or {
            "required_backend": None,
            "install_command": None,
            "backend_available": None,
            "fallback_possible": None,
            "architecture_adapter_supported": None,
            "missing_dependencies": [],
        }
    )
    result["readiness"] = model_probe_readiness(
        requested_task,
        supported,
        adapter,
        detail,
        backend_hint,
        dependencies,
    )
    return result


def safe_size(path):
    try:
        return path.stat().st_size
    except Exception:
        return 0


def positive_int(parameters, key, default, minimum=1):
    value = parameters.get(key, default)
    if isinstance(value, bool):
        fail("invalid_parameter", f"{key} must be an integer")
    try:
        value = int(value)
    except Exception:
        fail("invalid_parameter", f"{key} must be an integer")
    if value < minimum:
        fail("invalid_parameter", f"{key} must be at least {minimum}")
    return value


def positive_float(parameters, key, default, minimum=0.0):
    value = parameters.get(key, default)
    if isinstance(value, bool):
        fail("invalid_parameter", f"{key} must be numeric")
    try:
        value = float(value)
    except Exception:
        fail("invalid_parameter", f"{key} must be numeric")
    if not math.isfinite(value) or value < minimum:
        fail("invalid_parameter", f"{key} must be at least {minimum}")
    return value


def command_estimate(payload):
    path = local_model_path(payload)
    task = normalized_name(payload.get("task"))
    if not task:
        fail("invalid_request", "task is required for workload estimation")
    parameters = normalized_parameters(payload)
    adapter = execution_adapter(path, task)
    if adapter is None:
        fail("unsupported_task", f"no estimate adapter for task '{task}'")
    parameter_guard = ExplicitParameterGuard(
        payload,
        task,
        adapter,
        parameters,
    )
    parameter_guard.validate_supported(
        supported_explicit_parameters(task, adapter)
    )
    parameters = parameter_guard.without_unsupported(parameters)
    probe = model_probe(path)
    files = file_inventory(path)
    download_size = sum(safe_size(item) for item in files)
    weight_payload = probe["weight_payload_bytes"]
    if weight_payload == 0:
        weight_payload = download_size

    batch = positive_int(parameters, "batch_size", 1)
    if task in IMAGE_TASKS:
        count = positive_int(parameters, "num_images", 1)
    elif task in VIDEO_TASKS:
        count = positive_int(parameters, "num_videos", 1)
    else:
        # ``audio.variations`` is the canonical Werk schema path and is
        # normalized to the ``variations`` leaf. Keep ``num_variations`` only
        # as a compatibility alias for older direct companion callers.
        count = positive_int(
            {
                "variations": parameters.get(
                    "variations",
                    parameters.get("num_variations", 1),
                )
            },
            "variations",
            1,
        )
    assumptions = [
        "weights are loaded from the local repository without conversion",
        "activation memory is a conservative architecture-independent heuristic",
    ]
    warnings = []
    recommendations = []
    if adapter == QWEN3_TTS_ADAPTER:
        assumptions.append(
            "Qwen3-TTS VoiceDesign loads the language model and bundled speech tokenizer"
        )
        ready, dependency_reason = adapter_dependency_ready(
            task,
            adapter,
            dependency_snapshot(),
        )
        if not ready:
            warnings.append(dependency_reason)
            recommendations.append(
                "run 'werk backend install qwen-tts' or set "
                f"{QWEN3_TTS_PYTHON_ENV} to an isolated qwen-tts interpreter"
            )
    # The public allow_* values are planner permissions, not execution
    # commands.  Only Werk's internal flags represent a degradation selected
    # for the concrete runtime.
    offload = bool(
        parameters.get("_werk_enable_cpu_offload")
        or parameters.get("_werk_enable_sequential_offload")
        or parameters.get("_werk_enable_component_offload")
    )
    precision = normalized_name(parameters.get("precision") or "auto")
    precision_scale = {
        "fp16": 0.6,
        "float16": 0.6,
        "half": 0.6,
        "bf16": 0.6,
        "bfloat16": 0.6,
        "fp8": 0.4,
        "float8": 0.4,
        "int8": 0.4,
        "int4": 0.3,
        "nf4": 0.3,
    }.get(precision, 1.0)
    if precision != "auto":
        assumptions.append(
            f"activation estimate is scaled for requested precision '{precision}'"
        )
    attention = normalized_name(parameters.get("attention_backend") or "auto")
    attention_scale = {
        "flash": 0.78,
        "flash_attention": 0.78,
        "xformers": 0.78,
        "sliced": 0.65,
        "sdpa": 0.88,
    }.get(attention, 1.0)
    if attention != "auto":
        assumptions.append(
            f"activation estimate is scaled for attention backend '{attention}'"
        )
    component_overhead = len(probe["components"]) * 32 * 1024**2
    assumptions.append(
        f"{len(probe['components'])} detected model component(s) contribute runtime overhead"
    )

    if task in IMAGE_TASKS:
        width = positive_int(parameters, "width", 1024)
        height = positive_int(parameters, "height", 1024)
        steps = positive_int(parameters, "steps", 28)
        pixels = width * height * batch
        activations = int(pixels * 4 * 32 * precision_scale * attention_scale)
        accelerator_peak = int(
            weight_payload * (0.72 if offload else 1.12)
            + activations
            + component_overhead
        )
        host_peak = int(
            weight_payload * (1.15 if offload else 0.30)
            + 768 * 1024**2
            + component_overhead
        )
        output_size = int(width * height * 1.6 * count)
        assumptions.append(f"{width}x{height}, batch {batch}, {steps} denoising steps")
        if width * height > 1024 * 1024:
            recommendations.append("enable VAE tiling if the selected pipeline supports it")
    elif task in VIDEO_TASKS:
        width = positive_int(parameters, "width", 832)
        height = positive_int(parameters, "height", 480)
        frames = positive_int(parameters, "frames", parameters.get("num_frames", 49))
        fps = positive_float(parameters, "fps", 24.0, 0.1)
        pixels = width * height * frames * batch
        window = positive_int(parameters, "window_size", frames)
        active_frames = min(frames, window)
        activations = int(
            width
            * height
            * active_frames
            * batch
            * 4
            * 42
            * precision_scale
            * attention_scale
        )
        accelerator_peak = int(
            weight_payload * (0.68 if offload else 1.10)
            + activations
            + component_overhead
        )
        host_peak = int(
            weight_payload * (1.20 if offload else 0.35)
            + 1536 * 1024**2
            + component_overhead
        )
        duration = frames / fps
        bitrate = positive_int(parameters, "bitrate", 8_000_000)
        output_size = int(duration * bitrate / 8 * count)
        assumptions.append(
            f"{width}x{height}, {frames} frames at {fps:g} fps, active window {active_frames}"
        )
        recommendations.append("use temporal windowing or tiling when full-frame fit is marginal")
    elif task in ALL_AUDIO_TASKS:
        duration = positive_float(parameters, "duration", 30.0, 0.01)
        if adapter == QWEN3_TTS_ADAPTER:
            # VoiceDesign emits native 24 kHz mono.  Sample-rate/channel
            # controls are intentionally not advertised until an explicit
            # resampling stage exists.
            sample_rate = 24_000
            channels = 1
        else:
            sample_rate = positive_int(parameters, "sample_rate", 44_100)
            channels = positive_int(parameters, "channels", 2)
        stems = max(1, len(parameters.get("stems", []) or []))
        working_audio = int(duration * sample_rate * channels * 4 * stems)
        accelerator_peak = int(
            weight_payload * (0.80 if offload else 1.08)
            + working_audio * 8 * precision_scale
            + component_overhead
        )
        host_peak = int(
            weight_payload * (1.15 if offload else 0.35)
            + working_audio * 3
            + component_overhead
        )
        bit_depth = positive_int(parameters, "bit_depth", 16)
        output_size = int(
            duration * sample_rate * channels * bit_depth / 8 * stems * count
        )
        assumptions.append(
            f"{duration:g}s, {sample_rate} Hz, {channels} channel(s), {stems} output stem(s)"
        )
        if task in (
            ASR_TASKS
            | AUDIO_CLASSIFICATION_TASKS
            | AUDIO_TEXT_TASKS
            | AUDIO_EMBEDDING_TASKS
        ):
            output_size = max(4096, int(duration * 80))
            assumptions.append(
                "this task returns structured text/embedding data rather than PCM audio"
            )
    else:
        fail("unsupported_task", f"no estimate adapter for task '{task}'")

    if weight_payload == download_size:
        warnings.append("weight files were not identified precisely; all repository bytes were counted")
    warnings.extend(parameter_guard.warnings)
    confidence = "heuristic"
    return {
        "task": task,
        "model_path": str(path),
        "download_size_bytes": download_size,
        "weight_payload_bytes": weight_payload,
        "accelerator_peak_bytes": max(0, accelerator_peak),
        "host_peak_bytes": max(0, host_peak),
        "output_size_bytes": max(0, output_size),
        "confidence": confidence,
        "fit_assessment": "unknown",
        "assumptions": assumptions,
        "warnings": warnings,
        "parameter_support": parameter_guard.metadata(),
        "recommendations": recommendations,
        "backend": "werk-media-companion",
        "offline": True,
    }


def torch_runtime(parameters):
    torch = require_module("torch", purpose="media inference")
    requested = normalized_name(
        parameters.get("device")
        or parameters.get("accelerator")
        or "auto"
    )
    hip_version = torch_hip_version(torch)
    try:
        torch_gpu_available = bool(torch.cuda.is_available())
    except Exception as error:
        fail(
            "accelerator_unavailable",
            "PyTorch GPU availability check failed",
            str(error),
        )

    if requested in {"auto", ""}:
        if torch_gpu_available:
            device = "cuda"
        elif (
            getattr(torch.backends, "mps", None) is not None
            and bool(torch.backends.mps.is_available())
        ):
            device = "mps"
        else:
            device = "cpu"
    elif requested == "cuda":
        if hip_version:
            fail(
                "accelerator_unavailable",
                "CUDA was requested, but the selected PyTorch environment is ROCm/HIP; use accelerator=rocm",
                f"torch.version.hip={hip_version}",
            )
        if not torch_gpu_available:
            fail("accelerator_unavailable", "CUDA was requested but torch reports no GPU")
        device = "cuda"
    elif requested in {"rocm", "hip"}:
        if not hip_version:
            cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
            detail = (
                f"selected PyTorch is a CUDA build ({cuda_version})"
                if cuda_version
                else "torch.version.hip is not set"
            )
            fail(
                "accelerator_unavailable",
                "ROCm/HIP was requested, but the selected PyTorch environment is not ROCm-capable",
                detail,
            )
        if not torch_gpu_available:
            fail(
                "accelerator_unavailable",
                "ROCm/HIP was requested and torch.version.hip is set, but torch reports no GPU",
                f"torch.version.hip={hip_version}",
            )
        device = "cuda"
    elif requested in {"mps", "metal"}:
        if (
            getattr(torch.backends, "mps", None) is None
            or not bool(torch.backends.mps.is_available())
        ):
            fail("accelerator_unavailable", "MPS/Metal was requested but is unavailable")
        device = "mps"
    elif requested == "cpu":
        device = "cpu"
    else:
        fail("unsupported_parameter", f"unsupported device/accelerator '{requested}'")

    precision = normalized_name(parameters.get("precision") or "auto")
    if precision in {"float32", "fp32", "f32"}:
        dtype = torch.float32
    elif precision in {"bfloat16", "bf16"}:
        dtype = torch.bfloat16
    elif precision in {"float16", "fp16", "f16", "half"}:
        dtype = torch.float16
    elif precision in {"auto", ""}:
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    else:
        fail("unsupported_parameter", f"unsupported precision '{precision}'")
    return torch, device, dtype


def synchronize_torch_device(torch, device):
    """Wait for queued accelerator work before reading a wall-clock timer."""
    if device == "cuda":
        # PyTorch exposes both CUDA and ROCm/HIP synchronization here.
        torch.cuda.synchronize()
        return
    if device == "mps":
        synchronize = getattr(getattr(torch, "mps", None), "synchronize", None)
        if callable(synchronize):
            synchronize()


def seeded_generator(torch, device, parameters):
    seed = parameters.get("seed")
    if seed is None:
        return None, None
    try:
        seed = int(seed)
    except Exception:
        fail("invalid_parameter", "seed must be an integer")
    generator_device = device if device in {"cuda", "cpu"} else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed & ((1 << 63) - 1))
    return generator, seed


def local_input_path(value, name):
    if isinstance(value, list):
        if not value:
            fail("invalid_request", f"{name} list is empty")
        value = value[0]
    path = Path(required_string(value, name)).expanduser().resolve()
    if not path.is_file():
        fail("input_not_found", f"{name} does not exist or is not a file: {path}")
    return path


def path_cache_identity(path):
    """Return a lightweight identity for immutable/local model assets."""
    path = Path(path).expanduser().resolve()

    def stat_identity(item):
        try:
            stat = item.stat()
            return (
                str(item),
                int(getattr(stat, "st_dev", 0)),
                int(getattr(stat, "st_ino", 0)),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        except Exception:
            return (str(item), None, None, None, None)

    identity = [stat_identity(path)]
    if path.is_dir():
        # These files determine the concrete Diffusers pipeline and are cheap
        # to stat. Werk model-store directories are otherwise immutable.
        for name in ("model_index.json", "config.json"):
            candidate = path / name
            if candidate.exists():
                identity.append(stat_identity(candidate))
    return tuple(identity)


def normalized_lora_specs(parameters):
    raw = parameters.get("loras") or parameters.get("lora") or []
    if isinstance(raw, dict):
        raw = [raw]
    if not raw:
        return ()
    if not isinstance(raw, list):
        fail("invalid_parameter", "LoRA adapters must be a list")
    specs = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            path = local_input_path(item, f"loras[{index}]")
            weight = 1.0
        elif isinstance(item, dict):
            path = local_input_path(
                item.get("model") or item.get("path"),
                f"loras[{index}].model",
            )
            try:
                weight = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                fail("invalid_parameter", f"loras[{index}].weight must be numeric")
            if not math.isfinite(weight):
                fail("invalid_parameter", f"loras[{index}].weight must be finite")
        else:
            fail("invalid_parameter", "each LoRA entry must be a path or object")
        specs.append((path, weight, path_cache_identity(path)))
    return tuple(specs)


def diffusers_pipeline_cache_key(
    payload,
    model_path,
    task,
    has_image,
    device,
    dtype,
    parameters,
    lora_specs,
):
    """Describe only state that affects a loaded/configured pipeline."""
    cuda_index = None
    if device == "cuda":
        try:
            torch = importlib.import_module("torch")
            cuda_index = int(torch.cuda.current_device())
        except Exception:
            cuda_index = 0
    return (
        "diffusers",
        str(payload.get("model") or ""),
        path_cache_identity(model_path),
        str(task),
        bool(has_image),
        str(device),
        cuda_index,
        str(dtype).replace("torch.", ""),
        selected_offload_request(parameters),
        bool(parameters.get("vae_tiling")),
        bool(parameters.get("temporal_vae_tiling")),
        bool(parameters.get("vae_slicing")),
        bool(parameters.get("attention_slicing")),
        tuple(
            (identity, float(weight))
            for _path, weight, identity in lora_specs
        ),
    )


def accelerator_cache_identity(torch, device):
    if device != "cuda":
        return None
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return 0


def transformers_audio_cache_key(
    model_path,
    adapter,
    pipeline_task,
    torch,
    device,
    dtype,
):
    return (
        "transformers-audio",
        path_cache_identity(model_path),
        str(adapter),
        str(pipeline_task),
        str(device),
        accelerator_cache_identity(torch, device),
        str(dtype).replace("torch.", ""),
    )


def load_image(value, name):
    image_module = require_module("PIL.Image", "Pillow", "image input")
    path = local_input_path(value, name)
    try:
        with image_module.open(path) as image:
            return image.convert("RGB")
    except Exception as error:
        fail("invalid_input", f"failed to load {name}: {path}", str(error))


def load_video_frames(value, name):
    path = local_input_path(value, name)
    image_module = require_module("PIL.Image", "Pillow", "video input")
    errors = []
    try:
        av = importlib.import_module("av")
        with av.open(str(path)) as container:
            frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
        if frames:
            return frames
        errors.append("pyav: decoder returned no frames")
    except Exception as error:
        errors.append(f"pyav: {error}")
    try:
        imageio = importlib.import_module("imageio.v3")
        frames = [
            image_module.fromarray(frame).convert("RGB")
            for frame in imageio.imiter(path)
        ]
        if frames:
            return frames
        errors.append("imageio: decoder returned no frames")
    except Exception as error:
        errors.append(f"imageio: {error}")
    if not errors:
        errors.append("the decoder returned no frames")
    fail("invalid_input", f"failed to decode {name}: {path}", errors)


def supports_keyword(callable_value, keyword):
    try:
        signature = inspect.signature(callable_value)
    except Exception:
        return True
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def filtered_kwargs(
    callable_value,
    values,
    required=(),
    parameter_guard=None,
    parameter_paths=None,
):
    result = {}
    unsupported = []
    parameter_paths = parameter_paths or {}
    for key, value in values.items():
        if value is None:
            continue
        if supports_keyword(callable_value, key):
            result[key] = value
        elif key in required:
            unsupported.append(key)
        elif parameter_guard is not None and key in parameter_paths:
            paths = parameter_paths[key]
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                parameter_guard.reject(
                    path,
                    f"the selected pipeline call does not accept keyword '{key}'",
                )
    if unsupported:
        fail(
            "unsupported_parameter",
            "selected pipeline cannot accept required parameters",
            {"parameters": unsupported},
        )
    return result


def diffusers_video_mapping_name(task):
    """Return Diffusers' task registry for a Werk video task when available."""
    if task == "video_generation":
        return "AUTO_TEXT2VIDEO_PIPELINES_MAPPING"
    if task == "image_to_video":
        return "AUTO_IMAGE2VIDEO_PIPELINES_MAPPING"
    if task in {
        "video_to_video",
        "video_inpainting",
        "video_extension",
        "video_upscaling",
        "frame_interpolation",
    }:
        return "AUTO_VIDEO2VIDEO_PIPELINES_MAPPING"
    return None


def resolve_diffusers_video_pipeline_class(diffusers, model_path, task):
    """Resolve a video pipeline through the installed Diffusers registry.

    Diffusers does not expose an ``AutoPipelineForImage2Video`` class in all
    releases.  Its internal auto-pipeline registry is nevertheless the
    authoritative architecture-to-task mapping.  Looking up that registry
    keeps Werk architecture-neutral and also permits task variants whose
    registry key extends the repository's base key (for example ``*-i2v``).
    Unknown architectures deliberately fall back to ``DiffusionPipeline`` so
    a repository's own ``_class_name`` remains usable.
    """
    public_name = {
        "video_generation": "AutoPipelineForText2Video",
        "image_to_video": "AutoPipelineForImage2Video",
        "video_to_video": "AutoPipelineForVideo2Video",
    }.get(task)
    public_class = getattr(diffusers, public_name, None) if public_name else None
    if public_class is not None:
        return public_class

    mapping_name = diffusers_video_mapping_name(task)
    if mapping_name is None or model_path.is_file():
        return None
    model_index = read_json_file(model_path / "model_index.json")
    original_class_name = str(model_index.get("_class_name") or "")
    if not original_class_name:
        return None
    try:
        auto_pipeline = importlib.import_module("diffusers.pipelines.auto_pipeline")
        mapping = getattr(auto_pipeline, mapping_name, None)
        resolver = getattr(auto_pipeline, "_get_task_class", None)
        if mapping is None or not callable(resolver):
            return None
        direct = resolver(mapping, original_class_name, throw_error_if_not_exist=False)
        if direct is not None:
            return direct

        model_resolver = getattr(auto_pipeline, "_get_model", None)
        source_key = model_resolver(original_class_name) if callable(model_resolver) else None
        if not source_key:
            return None
        related = []
        for candidate_key, candidate_class in mapping.items():
            if (
                candidate_key == source_key
                or candidate_key.startswith(f"{source_key}-")
                or source_key.startswith(f"{candidate_key}-")
            ):
                related.append(candidate_class)
        if len(related) == 1:
            return related[0]
    except Exception:
        # Registry internals vary between Diffusers releases.  The repository
        # manifest still provides a sound generic fallback below.
        return None
    return None


def diffusers_component_class_name(model_path, component):
    if not model_path.is_dir():
        return None
    manifest = read_json_file(model_path / "model_index.json")
    descriptor = manifest.get(component)
    if isinstance(descriptor, str):
        class_name = descriptor
    elif isinstance(descriptor, (list, tuple)):
        class_name = next(
            (
                item
                for item in reversed(descriptor)
                if isinstance(item, str) and item.strip()
            ),
            None,
        )
    elif isinstance(descriptor, dict):
        class_name = descriptor.get("_class_name") or descriptor.get("class_name")
    else:
        class_name = None
    if class_name:
        return str(class_name)
    component_config = read_json_file(model_path / component / "config.json")
    class_name = component_config.get("_class_name")
    return str(class_name) if class_name else None


def diffusers_load_dtype(model_path, torch, dtype):
    """Use component precision requirements declared by the local pipeline.

    AutoencoderKLWan is documented and implemented as an fp32 VAE. Detecting
    the component class in the Diffusers manifest/config keeps the policy
    independent of repository names and leaves all other architectures alone.
    """
    vae_class = normalized_name(
        diffusers_component_class_name(model_path, "vae") or ""
    ).replace("_", "")
    float32 = getattr(torch, "float32", None)
    if vae_class == "autoencoderklwan" and float32 is not None and dtype != float32:
        return {"default": dtype, "vae": float32}
    return dtype


def load_diffusers_pipeline(model_path, task, has_image, torch, dtype):
    diffusers = require_module("diffusers", purpose=task)
    class_name = None
    if task in {"image_inpainting", "image_outpainting"}:
        class_name = "AutoPipelineForInpainting"
    elif task in {
        "image_editing",
        "image_variation",
        "image_upscaling",
    }:
        class_name = "AutoPipelineForImage2Image"
    elif task in VIDEO_TASKS:
        pipeline_class = resolve_diffusers_video_pipeline_class(
            diffusers,
            model_path,
            task,
        )
    elif task == "image_generation":
        class_name = "AutoPipelineForText2Image"

    if task not in VIDEO_TASKS:
        pipeline_class = getattr(diffusers, class_name, None) if class_name else None
    if pipeline_class is None:
        pipeline_class = getattr(diffusers, "DiffusionPipeline", None)
    if pipeline_class is None:
        fail("missing_dependency", "installed diffusers has no compatible pipeline class")

    load_dtype = diffusers_load_dtype(model_path, torch, dtype)
    load_kwargs = {"local_files_only": True, "torch_dtype": load_dtype}
    try:
        if model_path.is_file() and hasattr(pipeline_class, "from_single_file"):
            return pipeline_class.from_single_file(str(model_path), **load_kwargs)
        return pipeline_class.from_pretrained(str(model_path), **load_kwargs)
    except TypeError:
        # Older Diffusers releases may use the deprecated dtype spelling.
        load_kwargs.pop("torch_dtype", None)
        load_kwargs["dtype"] = load_dtype
        try:
            if model_path.is_file() and hasattr(pipeline_class, "from_single_file"):
                return pipeline_class.from_single_file(str(model_path), **load_kwargs)
            return pipeline_class.from_pretrained(str(model_path), **load_kwargs)
        except Exception as error:
            fail("model_load_failed", f"failed to load local Diffusers model: {model_path}", str(error))
    except Exception as error:
        fail("model_load_failed", f"failed to load local Diffusers model: {model_path}", str(error))


def configure_diffusers_pipeline(
    pipeline,
    device,
    task,
    parameters,
    warnings,
    parameter_guard,
    configuration_outcome=None,
):
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    sequential = bool(parameters.get("_werk_enable_sequential_offload"))
    component = bool(parameters.get("_werk_enable_component_offload"))
    model_cpu = bool(parameters.get("_werk_enable_cpu_offload"))
    offload_request = selected_offload_request(parameters)

    # Diffusers/Accelerate CPU-offload hooks target a CUDA-style accelerator.
    # Never invoke them for a CPU (or MPS) execution attempt.
    can_cpu_offload = device == "cuda"
    offload_mode = "none"
    try:
        if offload_request != "none" and not can_cpu_offload:
            fail(
                "backend_configuration_failed",
                "selected CPU offload requires a CUDA-style accelerator",
                {
                    "device": device,
                    "offload_request": offload_request,
                },
            )

        if sequential:
            if not callable(getattr(pipeline, "enable_sequential_cpu_offload", None)):
                fail(
                    "backend_configuration_failed",
                    "selected pipeline has no sequential CPU offload hook",
                    {"offload_request": offload_request},
                )
            pipeline.enable_sequential_cpu_offload()
            offload_mode = "sequential_cpu"

        model_cpu_requested = component or model_cpu
        if offload_mode == "none" and model_cpu_requested:
            if not callable(getattr(pipeline, "enable_model_cpu_offload", None)):
                fail(
                    "backend_configuration_failed",
                    "selected pipeline has no model CPU offload hook",
                    {"offload_request": offload_request},
                )
            pipeline.enable_model_cpu_offload()
            offload_mode = "model_cpu"
        if offload_mode == "none" and hasattr(pipeline, "to"):
            pipeline.to(device)
    except CompanionFailure:
        raise
    except Exception as error:
        fail("backend_configuration_failed", "failed to configure pipeline device/offload", str(error))

    tiling = bool(
        parameters.get("vae_tiling")
        or parameters.get("temporal_vae_tiling")
    )
    tiling_path = (
        "video.temporal_vae_tiling"
        if task in VIDEO_TASKS
        else "image.vae_tiling"
    )
    if tiling:
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        elif getattr(pipeline, "vae", None) is not None and hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        else:
            reason = "VAE tiling is unavailable on the selected pipeline"
            implicit_warning = (
                "VAE tiling requested by resolved defaults but unavailable on "
                "the selected pipeline"
            )
            if configuration_outcome is not None:
                configuration_outcome.reject(
                    tiling_path,
                    reason,
                    implicit_warning=implicit_warning,
                )
            else:
                parameter_guard.reject(tiling_path, reason)
                if tiling_path not in parameter_guard.explicit:
                    warnings.append(implicit_warning)
    if bool(parameters.get("vae_slicing")):
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        elif getattr(pipeline, "vae", None) is not None and hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        else:
            path = "image.vae_slicing"
            reason = "VAE slicing is unavailable on the selected pipeline"
            implicit_warning = (
                "VAE slicing requested by resolved defaults but unavailable on "
                "the selected pipeline"
            )
            if configuration_outcome is not None:
                configuration_outcome.reject(
                    path,
                    reason,
                    implicit_warning=implicit_warning,
                )
            else:
                parameter_guard.reject(path, reason)
                if path not in parameter_guard.explicit:
                    warnings.append(implicit_warning)
    if bool(parameters.get("attention_slicing")) and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    return {
        "offload_mode": offload_mode,
        "offload_request": offload_request,
    }


def apply_loras(
    pipeline,
    parameters,
    warnings,
    parameter_guard,
    configuration_outcome=None,
    lora_specs=None,
):
    if lora_specs is None:
        lora_specs = normalized_lora_specs(parameters)
    if not lora_specs:
        return
    if not hasattr(pipeline, "load_lora_weights"):
        path = "image.loras"
        reason = "the selected pipeline has no LoRA loading hook"
        if configuration_outcome is not None:
            configuration_outcome.reject(
                path,
                reason,
                fatal_when_implicit=(
                    "unsupported_parameter",
                    "LoRA adapters are not supported by the selected pipeline",
                ),
            )
            return
        parameter_guard.reject(path, reason)
        if path in parameter_guard.explicit:
            return
        fail("unsupported_parameter", "LoRA adapters are not supported by the selected pipeline")
    names = []
    weights = []
    for index, (path, weight, _identity) in enumerate(lora_specs):
        name = f"werk_lora_{index}"
        try:
            pipeline.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=name,
                local_files_only=True,
            )
        except TypeError:
            pipeline.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=name,
            )
        except Exception as error:
            fail("adapter_load_failed", f"failed to load LoRA: {path}", str(error))
        names.append(name)
        weights.append(weight)
    if hasattr(pipeline, "set_adapters"):
        pipeline.set_adapters(names, adapter_weights=weights)
    elif any(weight != 1.0 for weight in weights):
        path = "image.loras"
        reason = (
            "the pipeline loaded LoRA files but cannot apply their explicit weights"
        )
        implicit_warning = (
            "pipeline loaded LoRA adapters but cannot apply adapter weights "
            "from resolved defaults"
        )
        if configuration_outcome is not None:
            configuration_outcome.reject(
                path,
                reason,
                implicit_warning=implicit_warning,
            )
        else:
            parameter_guard.reject(path, reason)
            if path not in parameter_guard.explicit:
                warnings.append(implicit_warning)


def replay_diffusers_configuration(outcome, parameter_guard, warnings):
    if outcome is None:
        return
    for path, reason in outcome.unsupported.items():
        explicit = path in parameter_guard.explicit
        fatal = outcome.fatal_when_implicit.get(path)
        if not explicit and fatal is not None:
            code, message = fatal
            fail(code, message)
        parameter_guard.reject(path, reason)
        if not explicit and path in outcome.implicit_warnings:
            warnings.append(outcome.implicit_warnings[path])


def diffusers_count_parameter(namespace, parameters, parameter_guard):
    """Resolve Werk's task count aliases for a Diffusers pipeline call.

    Werk resolves defaults before invoking the companion, so checking only
    whether ``num_images``/``num_videos`` is present makes that default mask an
    explicitly requested ``batch_size``. Explicitness determines precedence;
    the resolved values are only used after that choice has been made.
    """
    if namespace == "image":
        count_name = "num_images"
        call_name = "num_images_per_prompt"
    elif namespace == "video":
        count_name = "num_videos"
        call_name = "num_videos_per_prompt"
    else:
        fail("invalid_request", f"unsupported Diffusers count namespace '{namespace}'")

    count_path = f"{namespace}.{count_name}"
    batch_path = f"{namespace}.batch_size"
    count_explicit = count_path in parameter_guard.explicit
    batch_explicit = batch_path in parameter_guard.explicit

    if count_explicit and batch_explicit:
        # Preserve the existing conflict semantics: the task-specific count
        # wins under warn/permissive and strict policy rejects the request.
        parameter_guard.reject_overridden(batch_path, count_path)

    if batch_explicit and not count_explicit:
        return call_name, parameters.get("batch_size"), batch_path
    if parameters.get(count_name) is not None:
        return call_name, parameters.get(count_name), count_path
    return call_name, parameters.get("batch_size"), batch_path


def diffusers_call_values(
    task,
    parameters,
    inputs,
    torch,
    device,
    parameter_guard,
):
    prompt = parameters.get("prompt") or parameters.get("description") or ""
    if task not in {"frame_interpolation", "video_upscaling", "image_upscaling"}:
        prompt = required_string(prompt, "effective_parameters.prompt")
    generator, seed = seeded_generator(torch, device, parameters)
    namespace = "video" if task in VIDEO_TASKS else "image"
    count_key, count_value, count_path = diffusers_count_parameter(
        namespace,
        parameters,
        parameter_guard,
    )
    values = {
        "prompt": prompt or None,
        "negative_prompt": parameters.get("negative_prompt"),
        "width": parameters.get("width"),
        "height": parameters.get("height"),
        "num_inference_steps": parameters.get("steps"),
        "guidance_scale": parameters.get("guidance_scale", parameters.get("guidance")),
        "guidance_rescale": parameters.get("guidance_rescale"),
        "strength": parameters.get(
            "image_strength",
            parameters.get("video_strength", parameters.get("strength")),
        ),
        "eta": parameters.get("eta"),
        "num_frames": parameters.get("frames", parameters.get("num_frames")),
        "fps": parameters.get("fps"),
        "decode_chunk_size": parameters.get("decode_chunk_size"),
        "motion_bucket_id": parameters.get("motion_bucket"),
        "noise_aug_strength": parameters.get("noise_augmentation"),
        "generator": generator,
        # PIL output avoids architecture-specific tensor/NumPy layouts and is
        # supported by modern Diffusers video pipelines.  Older pipelines
        # simply have this optional hint filtered from their call signature.
        "output_type": "pil" if task in VIDEO_TASKS else None,
    }
    values[count_key] = count_value
    parameter_paths = {
        "negative_prompt": "negative_prompt",
        "width": f"{namespace}.width",
        "height": f"{namespace}.height",
        "num_inference_steps": f"{namespace}.steps",
        "guidance_scale": f"{namespace}.guidance",
        count_key: count_path,
        "num_frames": "video.frames",
        "generator": f"{namespace}.seed",
    }
    image_value = (
        inputs.get("input_image")
        or inputs.get("initial_image")
        or inputs.get("image")
        or (
            inputs.get("input")
            if task in IMAGE_TASKS | {"image_to_video"}
            else None
        )
        or (
            inputs.get("source")
            if task in IMAGE_TASKS | {"image_to_video"}
            else None
        )
    )
    if image_value is not None:
        values["image"] = load_image(image_value, "input image")
    final_image = inputs.get("final_image")
    if final_image is not None:
        values["last_image"] = load_image(final_image, "final image")
    mask_value = inputs.get("mask_image") or inputs.get("mask")
    if mask_value is not None:
        values["mask_image"] = load_image(mask_value, "mask image")
    mask_video_value = inputs.get("mask_video")
    if mask_video_value is not None:
        values["mask_video"] = load_video_frames(mask_video_value, "mask video")
    video_value = (
        inputs.get("source_video")
        or inputs.get("input_video")
        or inputs.get("video")
        or inputs.get("source")
        or inputs.get("input")
    )
    if video_value is not None:
        values["video"] = load_video_frames(video_value, "source video")
    required = ["prompt"]
    if task in {
        "image_editing",
        "image_variation",
        "image_inpainting",
        "image_outpainting",
        "image_upscaling",
        "image_to_video",
    }:
        required.append("image")
        if "image" not in values:
            fail("invalid_request", f"task '{task}' requires a local input image")
    if task == "image_inpainting":
        required.append("mask_image")
        if "mask_image" not in values:
            fail("invalid_request", f"task '{task}' requires a local mask input")
    if task == "video_inpainting":
        required.append("mask_video")
        if "mask_video" not in values:
            fail("invalid_request", f"task '{task}' requires a local mask video")
    if task in {
        "video_to_video",
        "video_inpainting",
        "video_extension",
        "video_upscaling",
        "frame_interpolation",
    }:
        required.append("video")
        if "video" not in values:
            fail("invalid_request", f"task '{task}' requires a local source video")
    return values, required, seed, parameter_paths


def image_format(parameters):
    value = normalized_name(parameters.get("output_format") or parameters.get("format") or "png")
    aliases = {"jpg": "jpeg"}
    value = aliases.get(value, value)
    if value not in {"png", "jpeg", "webp"}:
        fail("unsupported_parameter", f"unsupported image output format '{value}'")
    return value


def image_mime(format_name):
    return {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }[format_name]


def ensure_pil_image(value):
    image_module = require_module("PIL.Image", "Pillow", "image output")
    if hasattr(value, "save"):
        if getattr(value, "mode", None) == "RGB" or not hasattr(value, "convert"):
            return value
        return value.convert("RGB")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    numpy = require_module("numpy", purpose="image output")
    array = numpy.asarray(value)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = numpy.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim not in {2, 3}:
        fail(
            "backend_error",
            "media pipeline returned an unsupported frame layout",
            {"shape": list(array.shape)},
        )
    if array.dtype.kind == "f":
        finite = array[numpy.isfinite(array)]
        minimum = float(finite.min()) if finite.size else 0.0
        maximum = float(finite.max()) if finite.size else 0.0
        if minimum >= -1.0 and maximum <= 1.0 and minimum < 0.0:
            array = (array + 1.0) * 127.5
        elif minimum >= 0.0 and maximum <= 1.0:
            array = array * 255.0
        array = numpy.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = numpy.clip(array, 0, 255).astype("uint8")
    elif array.dtype != numpy.uint8:
        array = numpy.clip(array, 0, 255).astype("uint8")
    try:
        return image_module.fromarray(array).convert("RGB")
    except Exception as error:
        fail(
            "backend_error",
            "media pipeline returned a frame that Pillow cannot decode",
            {"shape": list(array.shape), "dtype": str(array.dtype), "error": str(error)},
        )


def save_images(images, output_dir, task, parameters, identifier):
    if not isinstance(images, (list, tuple)):
        images = [images]
    format_name = image_format(parameters)
    suffix = "jpg" if format_name == "jpeg" else format_name
    outputs = []
    for index, raw in enumerate(images):
        image = ensure_pil_image(raw)
        path = output_dir / f"{task}-{identifier}-{index + 1}.{suffix}"
        save_kwargs = {}
        if format_name == "jpeg":
            # ``quality`` is the leaf alias of canonical
            # ``routing.quality`` (quality/balanced/latency policy), not a
            # numeric JPEG encoder setting.
            save_kwargs["quality"] = 95
        image.save(path, format=format_name.upper(), **save_kwargs)
        outputs.append(
            output_record(
                path,
                image_mime(format_name),
                width=int(image.width),
                height=int(image.height),
            )
        )
    return outputs


def frame_batches(value, expected_frames=None):
    if value is None:
        return []
    try:
        expected_frames = int(expected_frames) if expected_frames is not None else None
    except (TypeError, ValueError):
        expected_frames = None
    if expected_frames is not None and expected_frames < 1:
        expected_frames = None
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        first = value[0]
        if isinstance(first, (list, tuple)):
            return [list(batch) for batch in value]
        first_ndim = getattr(first, "ndim", None)
        if first_ndim is None and hasattr(first, "shape"):
            first_ndim = len(first.shape)
        if first_ndim in {4, 5}:
            batches = []
            for batch in value:
                batches.extend(frame_batches(batch, expected_frames))
            return batches
        return [list(value)]

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    ndim = getattr(value, "ndim", None)
    if ndim is None:
        return [[value]]
    if ndim == 5:
        shape = value.shape
        layouts = []
        if shape[-1] in {1, 3, 4}:
            layouts.append(("bfhwc", shape[1]))
        if shape[2] in {1, 3, 4}:
            layouts.append(("bfchw", shape[1]))
        if shape[1] in {1, 3, 4}:
            layouts.append(("bcfhw", shape[2]))
        if expected_frames is not None:
            matching = [layout for layout in layouts if layout[1] == expected_frames]
            if matching:
                layouts = matching
        if not layouts:
            fail(
                "backend_error",
                "video pipeline returned an unsupported batch layout",
                {"shape": list(shape)},
            )
        layout = layouts[0][0]
        if layout == "bfhwc":  # batch, frames, height, width, channels
            return [[value[batch, frame] for frame in range(shape[1])] for batch in range(shape[0])]
        if layout == "bfchw":  # batch, frames, channels, height, width
            return [[value[batch, frame] for frame in range(shape[1])] for batch in range(shape[0])]
        return [[value[batch, :, frame] for frame in range(shape[2])] for batch in range(shape[0])]
    if ndim == 4:
        shape = value.shape
        layouts = []
        if shape[-1] in {1, 3, 4}:
            layouts.append(("fhwc", shape[0]))
        if shape[1] in {1, 3, 4}:
            layouts.append(("fchw", shape[0]))
        if shape[0] in {1, 3, 4}:
            layouts.append(("cfhw", shape[1]))
        if expected_frames is not None:
            matching = [layout for layout in layouts if layout[1] == expected_frames]
            if matching:
                layouts = matching
        if not layouts:
            fail(
                "backend_error",
                "video pipeline returned an unsupported frame layout",
                {"shape": list(shape)},
            )
        if layouts[0][0] == "cfhw":  # channels, frames, height, width
            return [[value[:, frame] for frame in range(shape[1])]]
        return [[value[frame] for frame in range(shape[0])]]
    if ndim in {2, 3}:
        return [[value]]
    fail(
        "backend_error",
        "video pipeline returned an unsupported frame batch layout",
        {"shape": list(value.shape)},
    )


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    if not executable or not Path(executable).is_file():
        return None
    return str(executable)


def encode_video_with_pyav(frames, path, fps):
    av = importlib.import_module("av")
    numpy = require_module("numpy", purpose="video export")
    width, height = frames[0].size
    container = None
    try:
        container = av.open(str(path), mode="w", format="mp4")
        stream = container.add_stream("h264", rate=Fraction(str(float(fps))).limit_denominator(1000))
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = "yuv420p"
        for image in frames:
            if image.size != (width, height):
                raise ValueError("all video frames must have the same dimensions")
            array = numpy.asarray(image, dtype="uint8")
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        if container is not None:
            container.close()


def encode_video_with_ffmpeg(frames, path, fps):
    executable = ffmpeg_executable()
    if executable is None:
        raise FileNotFoundError("ffmpeg executable not found on PATH or in imageio-ffmpeg")
    numpy = require_module("numpy", purpose="video export")
    width, height = frames[0].size
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for image in frames:
            if image.size != (width, height):
                raise ValueError("all video frames must have the same dimensions")
            array = numpy.asarray(image, dtype="uint8")
            process.stdin.write(array.tobytes())
        process.stdin.close()
        detail = process.stderr.read().decode("utf-8", errors="replace")
        status = process.wait()
    except BaseException:
        with contextlib.suppress(Exception):
            process.kill()
        with contextlib.suppress(Exception):
            process.wait()
        raise
    if status != 0:
        raise RuntimeError(detail.strip() or f"ffmpeg exited with status {status}")


def export_video(frames, path, fps, format_name):
    frames = [ensure_pil_image(frame) for frame in frames]
    if not frames:
        fail("backend_error", "video pipeline returned no frames")
    if format_name == "gif":
        try:
            duration_ms = max(1, round(1000.0 / float(fps)))
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
            )
            return
        except Exception as error:
            fail("encoding_failed", f"failed to encode animated GIF: {path}", str(error))
    errors = []
    for encoder in (encode_video_with_pyav, encode_video_with_ffmpeg):
        try:
            encoder(frames, path, fps)
            return
        except Exception as error:
            errors.append(f"{encoder.__name__}: {error}")
            with contextlib.suppress(Exception):
                path.unlink()
    try:
        imageio = importlib.import_module("imageio.v3")
        numpy = require_module("numpy", purpose="video export")
        imageio.imwrite(
            path,
            numpy.stack([numpy.asarray(frame) for frame in frames]),
            fps=float(fps),
        )
        return
    except Exception as error:
        errors.append(f"imageio: {error}")
        with contextlib.suppress(Exception):
            path.unlink()
    fail("encoding_failed", f"failed to encode video: {path}", errors)


def prepare_diffusers_pipeline(
    runtime,
    payload,
    model_path,
    task,
    has_image,
    torch,
    device,
    dtype,
    parameters,
    parameter_guard,
):
    """Load or reuse a fully configured image/video Diffusers pipeline."""
    cache = runtime.pipeline_cache if runtime is not None else None
    cache_enabled = cache is not None and cache.enabled
    lora_specs = normalized_lora_specs(parameters)
    cache_key = diffusers_pipeline_cache_key(
        payload,
        model_path,
        task,
        has_image,
        device,
        dtype,
        parameters,
        lora_specs,
    )
    if cache_enabled:
        entry = cache.get(cache_key)
        if entry is not None:
            return entry, cache_key, True
        # Eviction deliberately precedes model loading. Keeping both the old
        # and new pipeline alive can otherwise turn a normal model switch into
        # an avoidable accelerator OOM.
        cache.prepare_for_load(cache_key)

    load_started = time.perf_counter()
    pipeline = None
    configuration_outcome = (
        DiffusersConfigurationOutcome() if cache_enabled else None
    )
    warnings = []
    try:
        pipeline = load_diffusers_pipeline(model_path, task, has_image, torch, dtype)
        offload_metadata = configure_diffusers_pipeline(
            pipeline,
            device,
            task,
            parameters,
            warnings,
            parameter_guard,
            configuration_outcome=configuration_outcome,
        )
        apply_loras(
            pipeline,
            parameters,
            warnings,
            parameter_guard,
            configuration_outcome=configuration_outcome,
            lora_specs=lora_specs,
        )
        synchronize_torch_device(torch, device)
        model_load_seconds = max(0.0, time.perf_counter() - load_started)
        entry = DiffusersPipelineEntry(
            pipeline,
            torch,
            device,
            dtype,
            offload_metadata,
            configuration_outcome,
            model_load_seconds,
            warnings,
        )
        if cache_enabled:
            cache.put(cache_key, entry)
        return entry, cache_key, False
    except BaseException:
        if pipeline is not None:
            cleanup_diffusers_pipeline_entry(
                DiffusersPipelineEntry(
                    pipeline,
                    torch,
                    device,
                    dtype,
                    {},
                    configuration_outcome,
                    0.0,
                    warnings,
                )
            )
        raise


def execute_diffusers(
    payload,
    model_path,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    parameter_guard,
    runtime=None,
):
    torch, device, dtype = torch_runtime(parameters)
    has_image = any(
        key in inputs for key in ("image", "input_image", "initial_image")
    )
    entry, cache_key, model_cache_hit = prepare_diffusers_pipeline(
        runtime,
        payload,
        model_path,
        task,
        has_image,
        torch,
        device,
        dtype,
        parameters,
        parameter_guard,
    )
    cached = runtime is not None and runtime.pipeline_cache.enabled
    try:
        return execute_prepared_diffusers_pipeline(
            entry,
            cache_key,
            model_cache_hit,
            runtime,
            torch,
            device,
            dtype,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            parameter_guard,
        )
    finally:
        # A one-shot process and a resident worker with cache size zero both
        # own an uncached pipeline for exactly one request. Explicit cleanup is
        # required even when validation or encoding raises: dropping Python
        # locals alone does not return PyTorch allocator caches to the device.
        if not cached:
            cleanup_diffusers_pipeline_entry(entry)


def execute_prepared_diffusers_pipeline(
    entry,
    cache_key,
    model_cache_hit,
    runtime,
    torch,
    device,
    dtype,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    parameter_guard,
):
    pipeline = entry.pipeline
    warnings = list(entry.configuration_warnings)
    replay_diffusers_configuration(
        entry.configuration_outcome,
        parameter_guard,
        warnings,
    )
    values, required, seed, parameter_paths = diffusers_call_values(
        task,
        parameters,
        inputs,
        torch,
        device,
        parameter_guard,
    )
    if (
        "num_videos_per_prompt" in values
        and not supports_keyword(pipeline.__call__, "num_videos_per_prompt")
        and supports_keyword(pipeline.__call__, "num_images_per_prompt")
    ):
        values["num_images_per_prompt"] = values.pop("num_videos_per_prompt")
        parameter_paths["num_images_per_prompt"] = parameter_paths.pop(
            "num_videos_per_prompt"
        )
    if (
        "mask_video" in values
        and not supports_keyword(pipeline.__call__, "mask_video")
    ):
        alternate_mask_keyword = next(
            (
                keyword
                for keyword in ("mask", "mask_image")
                if supports_keyword(pipeline.__call__, keyword)
            ),
            None,
        )
        if alternate_mask_keyword is not None:
            values[alternate_mask_keyword] = values.pop("mask_video")
            required = [
                alternate_mask_keyword if key == "mask_video" else key
                for key in required
            ]
    kwargs = filtered_kwargs(
        pipeline.__call__,
        values,
        required=required,
        parameter_guard=parameter_guard,
        parameter_paths=parameter_paths,
    )
    try:
        synchronize_torch_device(torch, device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            result = pipeline(**kwargs)
        synchronize_torch_device(torch, device)
        inference_seconds = max(0.0, time.perf_counter() - inference_started)
    except Exception as error:
        # A call-time failure may leave scheduler/module state inconsistent.
        # Signature validation happens before this block, so an unsupported
        # warm parameter does not discard an otherwise healthy resident model.
        if runtime is not None and runtime.pipeline_cache.enabled:
            pipeline = None
            runtime.pipeline_cache.evict(cache_key)
        fail("execution_failed", f"Diffusers pipeline failed for task '{task}'", str(error))

    encoding_started = time.perf_counter()
    if task in IMAGE_TASKS:
        images = getattr(result, "images", None)
        if images is None and isinstance(result, dict):
            images = result.get("images")
        if not images:
            fail("backend_error", "Diffusers pipeline returned no images")
        outputs = save_images(images, output_dir, task, parameters, identifier)
    else:
        frames = getattr(result, "frames", None)
        if frames is None and isinstance(result, dict):
            frames = result.get("frames")
        batches = frame_batches(
            frames,
            parameters.get("frames", parameters.get("num_frames")),
        )
        if not batches:
            fail("backend_error", "Diffusers pipeline returned no video frames")
        fps = positive_float(parameters, "fps", 24.0, 0.1)
        format_name = normalized_name(
            parameters.get("output_format") or parameters.get("format") or "mp4"
        )
        if format_name not in {"mp4", "gif"}:
            fail(
                "unsupported_parameter",
                f"unsupported direct video output format '{format_name}'",
            )
        outputs = []
        for index, batch in enumerate(batches):
            path = output_dir / f"{task}-{identifier}-{index + 1}.{format_name}"
            export_video(batch, path, fps, format_name)
            first = ensure_pil_image(batch[0])
            outputs.append(
                output_record(
                    path,
                    "video/mp4" if format_name == "mp4" else "image/gif",
                    width=int(first.width),
                    height=int(first.height),
                    duration=len(batch) / fps,
                    metadata={"frames": len(batch), "fps": fps},
                )
            )
    encoding_seconds = max(0.0, time.perf_counter() - encoding_started)
    return outputs, warnings, {
        "runtime": "diffusers",
        "pipeline_task": task,
        "pipeline_class": type(entry.pipeline).__name__,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
        "translated_parameters": sorted(kwargs),
        "model_load_seconds": 0.0 if model_cache_hit else entry.model_load_seconds,
        "model_cache_hit": model_cache_hit,
        "inference_seconds": inference_seconds,
        "encoding_seconds": encoding_seconds,
        **entry.offload_metadata,
    }


def declared_keyword(callable_value, keyword):
    try:
        return keyword in inspect.signature(callable_value).parameters
    except Exception:
        return False


def diffusers_audio_duration_keyword(pipeline):
    for keyword in ("audio_length_in_s", "audio_end_in_s"):
        if declared_keyword(pipeline.__call__, keyword):
            return keyword
    return None


def execute_diffusers_audio(
    payload,
    model_path,
    task,
    parameters,
    output_dir,
    identifier,
    parameter_guard,
    runtime=None,
):
    torch, device, dtype = torch_runtime(parameters)
    entry, cache_key, model_cache_hit = prepare_diffusers_pipeline(
        runtime,
        payload,
        model_path,
        task,
        False,
        torch,
        device,
        dtype,
        parameters,
        parameter_guard,
    )
    cached = runtime is not None and runtime.pipeline_cache.enabled
    try:
        return execute_prepared_diffusers_audio(
            entry,
            cache_key,
            model_cache_hit,
            runtime,
            torch,
            device,
            dtype,
            task,
            parameters,
            output_dir,
            identifier,
            parameter_guard,
        )
    finally:
        if not cached:
            cleanup_diffusers_pipeline_entry(entry)


def execute_prepared_diffusers_audio(
    entry,
    cache_key,
    model_cache_hit,
    runtime,
    torch,
    device,
    dtype,
    task,
    parameters,
    output_dir,
    identifier,
    parameter_guard,
):
    pipeline = entry.pipeline
    warnings = list(entry.configuration_warnings)
    replay_diffusers_configuration(
        entry.configuration_outcome,
        parameter_guard,
        warnings,
    )
    prompt = prompt_with_lyrics(
        parameters.get("prompt") or parameters.get("description"),
        parameters.get("lyrics"),
    )
    prompt = required_string(prompt, "effective_parameters.prompt/description")
    generator, seed = seeded_generator(torch, device, parameters)
    duration = parameters.get("duration")
    duration_keyword = diffusers_audio_duration_keyword(pipeline)
    if duration is not None and duration_keyword is None:
        parameter_guard.reject(
            "audio.duration",
            "the selected Diffusers audio pipeline exposes no duration keyword",
        )
        if "audio.duration" not in parameter_guard.explicit:
            warnings.append(
                "resolved audio.duration was not applied because the selected "
                "Diffusers pipeline exposes no duration control"
            )
    count = positive_int(
        {
            "variations": parameters.get(
                "num_variations",
                parameters.get("variations", 1),
            )
        },
        "variations",
        1,
    )
    values = {
        "prompt": prompt,
        "negative_prompt": parameters.get("negative_prompt"),
        "num_inference_steps": parameters.get("steps"),
        "guidance_scale": parameters.get("guidance_scale", parameters.get("guidance")),
        "num_waveforms_per_prompt": count,
        "generator": generator,
        "output_type": "np",
    }
    parameter_paths = {
        "negative_prompt": "negative_prompt",
        "num_inference_steps": "audio.steps",
        "guidance_scale": "audio.guidance",
        "num_waveforms_per_prompt": "audio.variations",
        "generator": "audio.seed",
    }
    if duration_keyword is not None:
        values[duration_keyword] = duration
        parameter_paths[duration_keyword] = "audio.duration"
    kwargs = filtered_kwargs(
        pipeline.__call__,
        values,
        required=("prompt",),
        parameter_guard=parameter_guard,
        parameter_paths=parameter_paths,
    )
    effective_count = count if "num_waveforms_per_prompt" in kwargs else 1
    try:
        synchronize_torch_device(torch, device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            result = pipeline(**kwargs)
        synchronize_torch_device(torch, device)
        inference_seconds = max(0.0, time.perf_counter() - inference_started)
    except Exception as error:
        if runtime is not None and runtime.pipeline_cache.enabled:
            pipeline = None
            runtime.pipeline_cache.evict(cache_key)
        fail("execution_failed", f"Diffusers audio pipeline failed for '{task}'", str(error))

    encoding_started = time.perf_counter()
    audios = getattr(result, "audios", None)
    if audios is None and isinstance(result, dict):
        audios = result.get("audios")
        if audios is None:
            audios = result.get("audio")
    if audios is None:
        fail("backend_error", "Diffusers pipeline returned no audio")
    # Older Diffusers vocoders can return mono batches as (batch, samples),
    # while newer AudioPipelineOutput implementations use a channel axis.
    waveforms = split_audio_waveforms(
        audios,
        effective_count,
        allow_2d_batch=True,
    )
    if len(waveforms) != effective_count:
        fail(
            "backend_error",
            "Diffusers audio pipeline returned a different number of variations "
            f"than requested ({len(waveforms)} != {effective_count})",
        )
    sample_rate = pipeline_sample_rate(pipeline)
    if sample_rate is None:
        fail(
            "backend_error",
            "Diffusers audio pipeline/config did not expose its sampling rate",
        )
    format_name = requested_audio_format(parameters)
    outputs = write_audio_outputs(
        task,
        identifier,
        output_dir,
        waveforms,
        sample_rate,
        format_name,
    )
    encoding_seconds = max(0.0, time.perf_counter() - encoding_started)
    translated = []
    if duration is not None and duration_keyword in kwargs:
        translated.append(f"audio.duration->{duration_keyword}")
    if "num_waveforms_per_prompt" in kwargs:
        translated.append("audio.variations->num_waveforms_per_prompt")
    for path, keyword in (
        ("negative_prompt", "negative_prompt"),
        ("audio.steps", "num_inference_steps"),
        ("audio.guidance", "guidance_scale"),
        ("audio.seed", "generator"),
    ):
        if keyword in kwargs:
            translated.append(f"{path}->{keyword}")
    return outputs, warnings, {
        "runtime": "diffusers",
        "pipeline_task": task,
        "pipeline_class": type(entry.pipeline).__name__,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
        "translated_parameters": sorted(set(translated)),
        "model_load_seconds": 0.0 if model_cache_hit else entry.model_load_seconds,
        "model_cache_hit": model_cache_hit,
        "inference_seconds": inference_seconds,
        "encoding_seconds": encoding_seconds,
        **entry.offload_metadata,
    }


def transformers_device(device):
    if device == "cuda":
        return 0
    if device == "mps":
        return "mps"
    return -1


def load_transformers_pipeline(
    model_path,
    pipeline_task,
    parameters,
    runtime_values=None,
):
    if runtime_values is None:
        torch, device, dtype = torch_runtime(parameters)
    else:
        torch, device, dtype = runtime_values
    transformers = require_module("transformers", purpose=pipeline_task)
    factory = getattr(transformers, "pipeline", None)
    if factory is None:
        fail("missing_dependency", "installed transformers package has no pipeline API")
    kwargs = {
        "task": pipeline_task,
        "model": str(model_path),
        "device": transformers_device(device),
        "trust_remote_code": False,
    }
    # Do not repeat local_files_only in model_kwargs. Transformers builds its
    # own hub kwargs and expands both mappings into from_pretrained(), which
    # raises for duplicate keys. Offline-only loading is enforced process-wide
    # above through the Hugging Face/Transformers offline environment flags.
    if device != "cpu":
        try:
            factory_parameters = inspect.signature(factory).parameters
        except Exception:
            factory_parameters = {}
        if "dtype" in factory_parameters:
            kwargs["dtype"] = dtype
        else:
            kwargs["torch_dtype"] = dtype
    try:
        pipeline = factory(**kwargs)
    except Exception as error:
        fail(
            "model_load_failed",
            f"failed to load local Transformers pipeline '{pipeline_task}' from {model_path}",
            str(error),
        )
    return pipeline, torch, device, dtype


def prepare_transformers_audio_entry(
    runtime,
    model_path,
    adapter,
    pipeline_task,
    parameters,
    loader,
):
    """Load or reuse one entry in the shared resident media LRU."""
    cache = runtime.pipeline_cache if runtime is not None else None
    cache_enabled = cache is not None and cache.enabled
    runtime_values = torch_runtime(parameters)
    cache_key = None
    if cache_enabled:
        torch, device, dtype = runtime_values
        cache_key = transformers_audio_cache_key(
            model_path,
            adapter,
            pipeline_task,
            torch,
            device,
            dtype,
        )
        entry = cache.get(cache_key)
        if entry is not None:
            if not isinstance(entry, TransformersAudioEntry):
                cache.evict(cache_key)
                fail("internal_error", "resident media cache entry has an invalid type")
            return entry, cache_key, True, True
        # Eviction happens before loading so two heavyweight frameworks never
        # temporarily exceed the one shared resident-cache budget.
        cache.prepare_for_load(cache_key)

    load_started = time.perf_counter()
    entry = None
    try:
        entry = loader(runtime_values)
        synchronize_torch_device(entry.torch, entry.device)
        entry.model_load_seconds = max(
            0.0,
            time.perf_counter() - load_started,
        )
        if cache_enabled:
            cache.put(cache_key, entry)
        return entry, cache_key, False, cache_enabled
    except BaseException:
        if entry is not None:
            cleanup_transformers_audio_entry(entry)
        else:
            cleanup_torch_allocator(runtime_values[0], runtime_values[1])
        raise


def prepare_transformers_pipeline(
    runtime,
    model_path,
    adapter,
    pipeline_task,
    parameters,
):
    def loader(runtime_values):
        pipeline, torch, device, dtype = load_transformers_pipeline(
            model_path,
            pipeline_task,
            parameters,
            runtime_values=runtime_values,
        )
        return TransformersAudioEntry(
            torch,
            device,
            dtype,
            0.0,
            adapter=adapter,
            pipeline_task=pipeline_task,
            pipeline=pipeline,
        )

    return prepare_transformers_audio_entry(
        runtime,
        model_path,
        adapter,
        pipeline_task,
        parameters,
        loader,
    )


def evict_failed_transformers_entry(runtime, cache_key):
    if (
        runtime is not None
        and cache_key is not None
        and runtime.pipeline_cache.enabled
    ):
        runtime.pipeline_cache.evict(cache_key)


def object_value(value, name):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None) if value is not None else None


def pipeline_sample_rate(pipeline, result=None):
    candidates = []
    if result is not None:
        candidates.extend([
            object_value(result, "sampling_rate"),
            object_value(result, "sample_rate"),
        ])
    component_names = (
        "processor",
        "audio_processor",
        "feature_extractor",
        "audio_feature_extractor",
        "audio_encoder",
        "vocoder",
        "codec",
        "vae",
        "model",
        "config",
    )
    pending = [(pipeline, 0)]
    visited = set()
    while pending:
        component, depth = pending.pop(0)
        if component is None or isinstance(
            component,
            (str, bytes, bytearray, int, float, bool),
        ):
            continue
        identity = id(component)
        if identity in visited:
            continue
        visited.add(identity)
        candidates.extend([
            object_value(component, "sampling_rate"),
            object_value(component, "sample_rate"),
        ])
        if depth >= 4:
            continue
        for component_name in component_names:
            child = object_value(component, component_name)
            if child is not None:
                pending.append((child, depth + 1))
    for value in candidates:
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def numpy_audio(value):
    numpy = require_module("numpy", purpose="audio output")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            # NumPy cannot directly represent Torch bfloat16 tensors. Audio
            # pipelines and embedding models commonly run at that dtype, so
            # convert only the host-side value used for serialization.
            float_value = getattr(value, "float", None)
            if callable(float_value):
                value = float_value()
                if hasattr(value, "numpy"):
                    value = value.numpy()
    return numpy.asarray(value)


def split_audio_waveforms(audios, expected_count=1, allow_2d_batch=False):
    array = numpy_audio(audios)
    if array.size == 0:
        fail("backend_error", "audio pipeline returned an empty waveform")
    try:
        expected_count = max(1, int(expected_count or 1))
    except (TypeError, ValueError):
        expected_count = 1
    if array.ndim == 1:
        return [array]
    if array.ndim == 2:
        if array.shape[0] == 1:
            return [array[0]]
        if (
            allow_2d_batch
            and expected_count > 1
            and array.shape[0] == expected_count
        ):
            return [array[index] for index in range(array.shape[0])]
        # Transformers specifies each TextToAudio result as channel-first 2-D
        # audio. Never reinterpret stereo channels as requested variations.
        return [array]
    if array.ndim == 3:
        return [array[index] for index in range(array.shape[0])]
    fail("backend_error", f"unsupported audio batch shape: {array.shape}")


def audio_result_items(result, pipeline):
    values = result if isinstance(result, list) else [result]
    if not values:
        fail("backend_error", "audio pipeline returned no results")
    items = []
    for value in values:
        audio = object_value(value, "audio")
        if audio is None:
            audio = object_value(value, "audios")
        if audio is None:
            fail("backend_error", "audio pipeline result has no 'audio' value")
        rate = pipeline_sample_rate(pipeline, value)
        if rate is None:
            fail("backend_error", "audio pipeline/config did not expose its sampling rate")
        items.append((audio, rate))
    return items


def requested_audio_format(parameters):
    format_name = normalized_name(
        parameters.get("output_format") or parameters.get("format") or "wav"
    )
    if format_name not in {"wav", "flac", "mp3", "ogg"}:
        fail(
            "unsupported_parameter",
            f"unsupported direct audio output format '{format_name}'",
        )
    return format_name


def audio_mime_type(format_name):
    return {
        "wav": "audio/wav",
        "flac": "audio/flac",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
    }[format_name]


def normalized_audio_array(audio):
    numpy = require_module("numpy", purpose="audio output")
    array = numpy.squeeze(numpy_audio(audio))
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 1:
        channels = 1
    elif array.ndim == 2:
        # Diffusers and Transformers audio model outputs use channel-first
        # two-dimensional waveforms. Normalize that public adapter contract to
        # the frame-major layout expected by soundfile and PCM interleaving.
        array = array.T
        channels = int(array.shape[1])
    else:
        fail("backend_error", f"unsupported audio tensor shape: {array.shape}")
    if array.size == 0:
        fail("backend_error", "cannot encode an empty audio waveform")
    if channels < 1:
        fail("backend_error", f"unsupported audio channel count: {channels}")
    if array.dtype.kind == "f":
        array = numpy.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
        pcm = (numpy.clip(array, -1.0, 1.0) * 32767.0).astype("<i2")
    elif array.dtype.itemsize <= 2:
        pcm = array.astype("<i2")
    else:
        pcm = numpy.clip(array, -32768, 32767).astype("<i2")
    interleaved = pcm.reshape(-1)
    frames = int(array.shape[0] if array.ndim > 1 else array.size)
    return array, interleaved, channels, frames


def encode_audio_with_ffmpeg(path, pcm, sample_rate, channels, format_name):
    executable = shutil.which("ffmpeg")
    if executable is None:
        fail(
            "missing_dependency",
            f"ffmpeg is required for {format_name} audio output",
        )
    codec_arguments = {
        "flac": ["-c:a", "flac"],
        "mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
        "ogg": ["-c:a", "libvorbis", "-q:a", "5"],
    }.get(format_name)
    if codec_arguments is None:
        fail("internal_error", f"no ffmpeg codec mapping for '{format_name}'")
    command = [
        executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-vn",
        *codec_arguments,
        "-y",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            input=pcm.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as error:
        fail("encoding_failed", f"failed to start ffmpeg for {format_name} output", str(error))
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        fail("encoding_failed", f"ffmpeg failed to encode {format_name} audio", detail)


def write_audio(path, audio, sample_rate, format_name):
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        fail("backend_error", "audio sampling rate must be an integer")
    if sample_rate <= 0:
        fail("backend_error", "audio sampling rate must be positive")
    array, pcm, channels, frames = normalized_audio_array(audio)
    if format_name == "wav":
        try:
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(channels)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(pcm.tobytes())
        except Exception as error:
            fail("encoding_failed", "failed to encode wav audio", str(error))
    elif format_name in {"flac", "ogg"}:
        try:
            soundfile = importlib.import_module("soundfile")
            soundfile.write(
                str(path),
                array,
                sample_rate,
                format=format_name.upper(),
            )
        except Exception:
            encode_audio_with_ffmpeg(
                path,
                pcm,
                sample_rate,
                channels,
                format_name,
            )
    elif format_name == "mp3":
        encode_audio_with_ffmpeg(path, pcm, sample_rate, channels, format_name)
    else:
        fail("unsupported_parameter", f"unsupported audio output format '{format_name}'")
    return channels, frames / float(sample_rate)


def write_audio_outputs(
    task,
    identifier,
    output_dir,
    waveforms,
    sample_rate,
    format_name,
):
    outputs = []
    multiple = len(waveforms) > 1
    for index, waveform in enumerate(waveforms, start=1):
        suffix = f"-{index}" if multiple else ""
        path = output_dir / f"{task}-{identifier}{suffix}.{format_name}"
        channels, duration = write_audio(path, waveform, sample_rate, format_name)
        outputs.append(
            output_record(
                path,
                audio_mime_type(format_name),
                duration=duration,
                metadata={"sample_rate": sample_rate, "channels": channels},
            )
        )
    return outputs


def local_audio_source(inputs, parameters):
    source = (
        inputs.get("input_audio")
        or inputs.get("source_audio")
        or inputs.get("audio")
        or inputs.get("source")
        or inputs.get("input")
        or parameters.get("input_audio")
    )
    return local_input_path(source, "input audio")


def resample_audio(array, source_rate, target_rate):
    numpy = require_module("numpy", purpose="audio input")
    if array.size == 0:
        fail("invalid_input", "decoded input audio is empty")
    if source_rate == target_rate:
        return array.astype("float32", copy=False)
    try:
        scipy_signal = importlib.import_module("scipy.signal")
        divisor = math.gcd(int(source_rate), int(target_rate))
        result = scipy_signal.resample_poly(
            array,
            int(target_rate) // divisor,
            int(source_rate) // divisor,
        )
    except Exception:
        target_frames = max(1, round(array.size * target_rate / source_rate))
        source_positions = numpy.linspace(0.0, 1.0, num=array.size, endpoint=False)
        target_positions = numpy.linspace(0.0, 1.0, num=target_frames, endpoint=False)
        result = numpy.interp(target_positions, source_positions, array)
    return numpy.asarray(result, dtype="float32")


def decode_local_audio(path, target_sample_rate):
    numpy = require_module("numpy", purpose="audio input")
    try:
        target_sample_rate = int(target_sample_rate)
    except (TypeError, ValueError):
        fail("backend_configuration_failed", "selected pipeline exposes no valid audio sampling rate")
    if target_sample_rate <= 0:
        fail("backend_configuration_failed", "selected pipeline exposes no valid audio sampling rate")
    errors = []
    try:
        soundfile = importlib.import_module("soundfile")
        array, source_rate = soundfile.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )
        array = numpy.asarray(array, dtype="float32")
        if array.ndim == 2:
            array = array.mean(axis=1)
        elif array.ndim != 1:
            raise ValueError(f"unsupported decoded audio shape {array.shape}")
        return {
            "raw": resample_audio(array, int(source_rate), target_sample_rate),
            "sampling_rate": target_sample_rate,
        }
    except Exception as error:
        errors.append(f"soundfile: {error}")
    executable = shutil.which("ffmpeg")
    if executable is not None:
        command = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(target_sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode == 0 and completed.stdout:
                return {
                    "raw": numpy.frombuffer(completed.stdout, dtype="<f4").copy(),
                    "sampling_rate": target_sample_rate,
                }
            errors.append(
                "ffmpeg: "
                + completed.stderr.decode("utf-8", "replace").strip()
            )
        except Exception as error:
            errors.append(f"ffmpeg: {error}")
    else:
        errors.append("ffmpeg: executable not found")
    fail("invalid_input", f"failed to decode input audio: {path}", errors)


def decoded_audio_for_pipeline(pipeline, source_path):
    sample_rate = pipeline_sample_rate(pipeline)
    if sample_rate is None:
        fail(
            "backend_configuration_failed",
            "selected Transformers pipeline/config exposes no audio sampling rate",
        )
    return decode_local_audio(source_path, sample_rate)


def seed_transformers_runtime(torch, parameters):
    seed = parameters.get("seed")
    if seed is None:
        return None
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        fail("invalid_parameter", "seed must be an integer")
    seed = seed & ((1 << 63) - 1)
    manual_seed = getattr(torch, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    manual_seed_all = getattr(cuda, "manual_seed_all", None)
    if callable(manual_seed_all):
        manual_seed_all(seed)
    return seed


def transformer_audio_frame_rate(pipeline):
    model = getattr(pipeline, "model", None)
    config = getattr(model, "config", None)
    candidates = [
        object_value(object_value(config, "audio_encoder"), "frame_rate"),
        object_value(
            object_value(getattr(model, "audio_encoder", None), "config"),
            "frame_rate",
        ),
        object_value(config, "frame_rate"),
    ]
    for value in candidates:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def transformer_audio_duration_metadata(pipeline, parameters, max_new_tokens):
    """Describe duration translation without turning model defaults into caps."""

    duration = parameters.get("duration")
    if duration is None or max_new_tokens is None:
        return None
    duration = positive_float({"duration": duration}, "duration", 30.0, 0.01)
    model_type = normalized_name(
        object_value(getattr(getattr(pipeline, "model", None), "config", None), "model_type")
    )
    metadata = {
        "requested_seconds": duration,
        "audio_tokens": max_new_tokens,
        "hard_limit_applied": False,
    }
    if model_type in {"musicgen", "musicgen_melody"}:
        metadata.update(
            {
                "model_default_seconds": 30.0,
                "exceeds_model_default": duration > 30.0,
            }
        )
    return metadata


def transformer_duration_tokens(
    pipeline,
    parameters,
    parameter_guard,
    warnings,
):
    duration = parameters.get("duration")
    if duration is None:
        return None
    duration = positive_float({"duration": duration}, "duration", 30.0, 0.01)
    model_type = normalized_name(
        object_value(getattr(getattr(pipeline, "model", None), "config", None), "model_type")
    )
    if model_type in {"musicgen", "musicgen_melody"} and duration > 30.0:
        warnings.append(
            f"resolved audio.duration ({duration:g}s) exceeds MusicGen's "
            "30-second model default; it is forwarded unchanged because the "
            "companion does not impose a hard duration limit. The selected "
            "model/runtime may reject the request or require continuation generation."
        )
    frame_rate = transformer_audio_frame_rate(pipeline)
    if frame_rate is None:
        parameter_guard.reject(
            "audio.duration",
            "the selected Transformers model/config exposes no audio token frame rate",
        )
        if "audio.duration" not in parameter_guard.explicit:
            warnings.append(
                "resolved audio.duration was not applied because the selected "
                "Transformers model exposes no audio token frame rate; the model "
                "generation default is used"
            )
        return None
    return max(1, math.ceil(duration * frame_rate))


def execute_audio_generation(
    model_path,
    task,
    parameters,
    output_dir,
    identifier,
    parameter_guard,
    runtime=None,
):
    entry, cache_key, model_cache_hit, resident = prepare_transformers_pipeline(
        runtime,
        model_path,
        "transformers_audio",
        "text-to-audio",
        parameters,
    )
    try:
        return execute_prepared_audio_generation(
            entry,
            cache_key,
            model_cache_hit,
            runtime,
            task,
            parameters,
            output_dir,
            identifier,
            parameter_guard,
        )
    finally:
        if not resident:
            cleanup_transformers_audio_entry(entry)


def execute_prepared_audio_generation(
    entry,
    cache_key,
    model_cache_hit,
    runtime,
    task,
    parameters,
    output_dir,
    identifier,
    parameter_guard,
):
    pipeline = entry.pipeline
    torch = entry.torch
    device = entry.device
    dtype = entry.dtype
    prompt = prompt_with_lyrics(
        parameters.get("prompt") or parameters.get("description"),
        parameters.get("lyrics"),
    )
    prompt = required_string(prompt, "effective_parameters.prompt/description")
    seed = seed_transformers_runtime(torch, parameters)
    warnings = []
    max_new_tokens = transformer_duration_tokens(
        pipeline,
        parameters,
        parameter_guard,
        warnings,
    )
    duration_metadata = transformer_audio_duration_metadata(
        pipeline,
        parameters,
        max_new_tokens,
    )
    temperature = parameters.get("temperature")
    if temperature is not None:
        temperature = positive_float(parameters, "temperature", 0.0)
    top_k = parameters.get("top_k")
    if top_k is not None:
        top_k = positive_int(parameters, "top_k", 0, minimum=0)
    top_p = parameters.get("top_p")
    if top_p is not None:
        top_p = positive_float(parameters, "top_p", 1.0)
        if top_p > 1.0:
            fail("invalid_parameter", "top_p must not exceed 1")
    guidance = parameters.get("guidance_scale", parameters.get("guidance"))
    if guidance is not None:
        guidance = positive_float({"guidance": guidance}, "guidance", 0.0)
    sampling_requested = any(
        value is not None for value in (temperature, top_k, top_p)
    )
    generate_kwargs = {
        key: value
        for key, value in {
            "guidance_scale": guidance,
            "temperature": temperature if temperature not in {None, 0, 0.0} else None,
            "do_sample": False if temperature == 0.0 else (
                True if sampling_requested else None
            ),
            "top_k": top_k,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }.items()
        if value is not None
    }
    call_values = {"generate_kwargs": generate_kwargs} if generate_kwargs else {}
    count = positive_int(
        {"variations": parameters.get("num_variations", parameters.get("variations", 1))},
        "variations",
        1,
    )
    prompts = prompt if count == 1 else [prompt] * count
    synchronize_torch_device(torch, device)
    inference_started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipeline(prompts, **call_values)
    except Exception as error:
        pipeline = None
        evict_failed_transformers_entry(runtime, cache_key)
        fail("execution_failed", f"Transformers audio pipeline failed for '{task}'", str(error))
    synchronize_torch_device(torch, device)
    inference_seconds = max(0.0, time.perf_counter() - inference_started)

    encoding_started = time.perf_counter()
    items = audio_result_items(result, pipeline)
    if len(items) == 1 and count > 1:
        audio, sample_rate = items[0]
        waveforms = split_audio_waveforms(audio, count)
        if len(waveforms) != count:
            fail(
                "backend_error",
                "Transformers audio pipeline did not return the requested number of variations",
            )
        items = [(waveform, sample_rate) for waveform in waveforms]
    if len(items) != count:
        fail(
            "backend_error",
            "Transformers audio pipeline returned a different number of variations "
            f"than requested ({len(items)} != {count})",
        )
    format_name = requested_audio_format(parameters)
    outputs = []
    for index, (audio, sample_rate) in enumerate(items, start=1):
        suffix = f"-{index}" if count > 1 else ""
        path = output_dir / f"{task}-{identifier}{suffix}.{format_name}"
        channels, duration = write_audio(path, audio, sample_rate, format_name)
        outputs.append(
            output_record(
                path,
                audio_mime_type(format_name),
                duration=duration,
                metadata={"sample_rate": sample_rate, "channels": channels},
            )
        )
    encoding_seconds = max(0.0, time.perf_counter() - encoding_started)
    translated_parameters = []
    if max_new_tokens is not None:
        translated_parameters.append("audio.duration->generate_kwargs.max_new_tokens")
    if count > 1:
        translated_parameters.append("audio.variations->batched prompts")
    if seed is not None:
        translated_parameters.append("audio.seed->torch.manual_seed")
    if parameters.get("lyrics"):
        translated_parameters.append("audio.lyrics->prompt")
    for path, keyword in (
        ("audio.guidance", "guidance_scale"),
        ("audio.temperature", "temperature"),
        ("audio.top_k", "top_k"),
        ("audio.top_p", "top_p"),
    ):
        if keyword in generate_kwargs:
            translated_parameters.append(f"{path}->generate_kwargs.{keyword}")
    if "do_sample" in generate_kwargs:
        translated_parameters.append("audio.sampling->generate_kwargs.do_sample")
    return outputs, warnings, {
        "runtime": "transformers",
        "pipeline_task": "text-to-audio",
        "pipeline_class": type(pipeline).__name__,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
        "translated_parameters": sorted(translated_parameters),
        "model_load_seconds": 0.0 if model_cache_hit else entry.model_load_seconds,
        "model_cache_hit": model_cache_hit,
        "inference_seconds": inference_seconds,
        "encoding_seconds": encoding_seconds,
        "offload_mode": "none",
        "offload_request": "none",
        **(
            {"duration_control": duration_metadata}
            if duration_metadata is not None
            else {}
        ),
    }


QWEN3_TTS_LANGUAGE_NAMES = {
    "auto": "Auto",
    "zh": "Chinese",
    "zho": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "eng": "English",
    "english": "English",
    "ja": "Japanese",
    "jpn": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "kor": "Korean",
    "korean": "Korean",
    "de": "German",
    "de_de": "German",
    "deu": "German",
    "ger": "German",
    "german": "German",
    "fr": "French",
    "fra": "French",
    "fre": "French",
    "french": "French",
    "ru": "Russian",
    "rus": "Russian",
    "russian": "Russian",
    "pt": "Portuguese",
    "por": "Portuguese",
    "portuguese": "Portuguese",
    "es": "Spanish",
    "spa": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "ita": "Italian",
    "italian": "Italian",
}


def qwen3_tts_language(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return "Auto"
    normalized = normalized_name(value)
    language = QWEN3_TTS_LANGUAGE_NAMES.get(normalized)
    if language is None:
        supported = sorted(set(QWEN3_TTS_LANGUAGE_NAMES.values()))
        fail(
            "invalid_parameter",
            f"unsupported Qwen3-TTS language '{value}'",
            {"supported_languages": supported},
        )
    return language


def qwen3_tts_device_map(torch, device):
    if device == "cuda":
        try:
            index = int(torch.cuda.current_device())
        except Exception:
            index = 0
        return f"cuda:{index}"
    return device


def load_qwen3_tts_voice_design(
    model_path,
    parameters,
    runtime_values=None,
):
    if runtime_values is None:
        torch, device, dtype = torch_runtime(parameters)
    else:
        torch, device, dtype = runtime_values
    model_root = model_path if model_path.is_dir() else model_path.parent
    if not is_qwen3_tts_voice_design(model_root):
        variant = qwen3_tts_model_variant(model_root)
        fail(
            "unsupported_model",
            "the Qwen3-TTS adapter currently supports only VoiceDesign models",
            {"tts_model_type": variant},
        )
    qwen_tts = require_module(
        "qwen_tts",
        "qwen-tts",
        "Qwen3-TTS VoiceDesign inference; run 'werk backend install qwen-tts' "
        f"or configure {QWEN3_TTS_PYTHON_ENV} with an isolated interpreter",
    )
    model_class = getattr(qwen_tts, "Qwen3TTSModel", None)
    from_pretrained = getattr(model_class, "from_pretrained", None)
    if not callable(from_pretrained):
        fail(
            "missing_dependency",
            "installed qwen-tts package does not expose Qwen3TTSModel.from_pretrained",
        )
    load_kwargs = {
        "device_map": qwen3_tts_device_map(torch, device),
        "dtype": dtype,
    }
    try:
        model = from_pretrained(str(model_root), **load_kwargs)
    except Exception as error:
        fail(
            "model_load_failed",
            f"failed to load local Qwen3-TTS VoiceDesign model from {model_root}",
            str(error),
        )
    if not callable(getattr(model, "generate_voice_design", None)):
        fail(
            "model_load_failed",
            "loaded qwen-tts model does not expose generate_voice_design",
        )
    return model, torch, device, dtype


def prepare_qwen3_tts_voice_design(runtime, model_path, parameters):
    def loader(runtime_values):
        model, torch, device, dtype = load_qwen3_tts_voice_design(
            model_path,
            parameters,
            runtime_values=runtime_values,
        )
        return TransformersAudioEntry(
            torch,
            device,
            dtype,
            0.0,
            adapter=QWEN3_TTS_ADAPTER,
            pipeline_task="voice-design",
            model=model,
        )

    return prepare_transformers_audio_entry(
        runtime,
        model_path,
        QWEN3_TTS_ADAPTER,
        "voice-design",
        parameters,
        loader,
    )


def execute_qwen3_tts_voice_design(
    model_path,
    task,
    parameters,
    output_dir,
    identifier,
    runtime=None,
):
    entry, cache_key, model_cache_hit, resident = prepare_qwen3_tts_voice_design(
        runtime,
        model_path,
        parameters,
    )
    try:
        return execute_prepared_qwen3_tts_voice_design(
            entry,
            cache_key,
            model_cache_hit,
            runtime,
            task,
            parameters,
            output_dir,
            identifier,
        )
    finally:
        if not resident:
            cleanup_transformers_audio_entry(entry)


def execute_prepared_qwen3_tts_voice_design(
    entry,
    cache_key,
    model_cache_hit,
    runtime,
    task,
    parameters,
    output_dir,
    identifier,
):
    model = entry.model
    torch = entry.torch
    device = entry.device
    dtype = entry.dtype
    text = required_string(
        parameters.get("text") or parameters.get("prompt"),
        "effective_parameters.text",
    )
    language = qwen3_tts_language(parameters.get("language"))
    raw_instruct = parameters.get("speaking_style")
    warnings = []
    if raw_instruct is None:
        instruct = "A clear, natural, high-quality studio voice with precise articulation."
        warnings.append(
            "tts.speaking_style was not supplied; a neutral high-quality VoiceDesign instruction was used"
        )
    else:
        instruct = required_string(
            raw_instruct,
            "effective_parameters.tts.speaking_style",
        )
    seed = seed_transformers_runtime(torch, parameters)
    synchronize_torch_device(torch, device)
    inference_started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
                non_streaming_mode=True,
            )
    except Exception as error:
        model = None
        evict_failed_transformers_entry(runtime, cache_key)
        fail("execution_failed", "Qwen3-TTS VoiceDesign generation failed", str(error))
    synchronize_torch_device(torch, device)
    inference_seconds = max(0.0, time.perf_counter() - inference_started)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        fail(
            "backend_error",
            "Qwen3-TTS generate_voice_design returned an invalid result",
        )
    waveforms, sample_rate = result
    if not isinstance(waveforms, (tuple, list)) or len(waveforms) != 1:
        fail(
            "backend_error",
            "Qwen3-TTS VoiceDesign must return exactly one waveform",
        )

    encoding_started = time.perf_counter()
    format_name = requested_audio_format(parameters)
    outputs = write_audio_outputs(
        task,
        identifier,
        output_dir,
        waveforms,
        sample_rate,
        format_name,
    )
    encoding_seconds = max(0.0, time.perf_counter() - encoding_started)
    translated = [
        "tts.language->language",
        "tts.speaking_style->instruct"
        if raw_instruct is not None
        else "adapter_default->instruct",
    ]
    if seed is not None:
        translated.append("tts.seed->torch.manual_seed")
    if parameters.get("precision") is not None:
        translated.append("routing.precision->dtype")
    if parameters.get("accelerator") is not None or parameters.get("device") is not None:
        translated.append("routing.accelerator->device_map")
    return outputs, warnings, {
        "runtime": "qwen-tts",
        "pipeline_task": "text-to-speech",
        "pipeline_class": type(entry.model).__name__,
        "model_variant": "voice_design",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "language": language,
        "seed": seed,
        "translated_parameters": sorted(translated),
        "model_load_seconds": 0.0 if model_cache_hit else entry.model_load_seconds,
        "model_cache_hit": model_cache_hit,
        "inference_seconds": inference_seconds,
        "encoding_seconds": encoding_seconds,
        "offload_mode": "none",
        "offload_request": "none",
    }


def execute_tts(
    model_path,
    task,
    parameters,
    output_dir,
    identifier,
    runtime=None,
):
    entry, cache_key, model_cache_hit, resident = prepare_transformers_pipeline(
        runtime,
        model_path,
        "transformers_tts",
        "text-to-audio",
        parameters,
    )
    try:
        return execute_prepared_tts(
            entry,
            cache_key,
            model_cache_hit,
            runtime,
            task,
            parameters,
            output_dir,
            identifier,
        )
    finally:
        if not resident:
            cleanup_transformers_audio_entry(entry)


def execute_prepared_tts(
    entry,
    cache_key,
    model_cache_hit,
    runtime,
    task,
    parameters,
    output_dir,
    identifier,
):
    pipeline = entry.pipeline
    torch = entry.torch
    device = entry.device
    dtype = entry.dtype
    text = required_string(parameters.get("text") or parameters.get("prompt"), "effective_parameters.text")
    seed = seed_transformers_runtime(torch, parameters)
    kwargs = {}
    for key in ("speaker_embeddings", "vocoder", "generate_kwargs"):
        if key in parameters:
            kwargs[key] = parameters[key]
    synchronize_torch_device(torch, device)
    inference_started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipeline(text, **kwargs)
    except Exception as error:
        pipeline = None
        evict_failed_transformers_entry(runtime, cache_key)
        fail("execution_failed", "Transformers text-to-speech pipeline failed", str(error))
    synchronize_torch_device(torch, device)
    inference_seconds = max(0.0, time.perf_counter() - inference_started)

    encoding_started = time.perf_counter()
    items = audio_result_items(result, pipeline)
    if len(items) != 1:
        fail("backend_error", "text-to-speech pipeline returned multiple audio results")
    audio, sample_rate = items[0]
    format_name = requested_audio_format(parameters)
    path = output_dir / f"{task}-{identifier}.{format_name}"
    channels, duration = write_audio(path, audio, sample_rate, format_name)
    encoding_seconds = max(0.0, time.perf_counter() - encoding_started)
    return [output_record(
        path,
        audio_mime_type(format_name),
        duration=duration,
        metadata={"sample_rate": sample_rate, "channels": channels},
    )], [], {
        "runtime": "transformers",
        "pipeline_task": "text-to-audio",
        "pipeline_class": type(pipeline).__name__,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
        "translated_parameters": (
            ["tts.seed->torch.manual_seed"] if seed is not None else []
        ),
        "model_load_seconds": 0.0 if model_cache_hit else entry.model_load_seconds,
        "model_cache_hit": model_cache_hit,
        "inference_seconds": inference_seconds,
        "encoding_seconds": encoding_seconds,
        "offload_mode": "none",
        "offload_request": "none",
    }


def transcript_segments(result):
    segments = []
    chunks = result.get("chunks") if isinstance(result, dict) else None
    if not isinstance(chunks, list):
        return segments
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        timestamp = chunk.get("timestamp") or chunk.get("timestamps")
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) < 2:
            continue
        try:
            start = max(0.0, float(timestamp[0] or 0.0))
            end = max(start, float(timestamp[1] or start))
        except (TypeError, ValueError):
            continue
        segments.append((start, end, str(chunk.get("text") or "").strip()))
    return segments


def transcript_timestamp(seconds, decimal_marker="."):
    milliseconds = max(0, round(float(seconds) * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
        f"{decimal_marker}{millis:03d}"
    )


def save_transcription_outputs(result, output_dir, task, identifier, parameters):
    format_name = normalized_name(parameters.get("output_format") or "json")
    format_name = {"txt": "text"}.get(format_name, format_name)
    if format_name not in {"json", "text", "srt", "vtt", "tsv"}:
        fail(
            "unsupported_parameter",
            f"unsupported transcription output format '{format_name}'",
        )
    text = str(result.get("text") or "")
    if format_name == "json":
        path = output_dir / f"{task}-{identifier}.json"
        atomic_json_write(path, result)
        return [output_record(path, "application/json")]
    if format_name == "text":
        path = output_dir / f"{task}-{identifier}.txt"
        path.write_text(text, encoding="utf-8")
        return [output_record(path, "text/plain")]

    segments = transcript_segments(result)
    if not segments:
        fail(
            "backend_error",
            f"transcription format '{format_name}' requires timestamped pipeline chunks",
        )
    if format_name == "srt":
        body = "\n\n".join(
            f"{index}\n"
            f"{transcript_timestamp(start, ',')} --> "
            f"{transcript_timestamp(end, ',')}\n{text}"
            for index, (start, end, text) in enumerate(segments, start=1)
        )
        mime_type = "application/x-subrip"
    elif format_name == "vtt":
        entries = "\n\n".join(
            f"{transcript_timestamp(start)} --> {transcript_timestamp(end)}\n{text}"
            for start, end, text in segments
        )
        body = f"WEBVTT\n\n{entries}"
        mime_type = "text/vtt"
    else:
        body = "start\tend\ttext\n" + "\n".join(
            f"{round(start * 1000)}\t{round(end * 1000)}\t"
            f"{text.replace(chr(9), ' ').replace(chr(10), ' ')}"
            for start, end, text in segments
        )
        mime_type = "text/tab-separated-values"
    path = output_dir / f"{task}-{identifier}.{format_name}"
    path.write_text(f"{body.rstrip()}\n", encoding="utf-8")
    return [output_record(path, mime_type)]


def asr_supports_generation(pipeline):
    model = getattr(pipeline, "model", None)
    can_generate = getattr(model, "can_generate", None)
    if callable(can_generate):
        try:
            return bool(can_generate())
        except Exception:
            pass
    return normalized_name(getattr(pipeline, "type", "")).startswith("seq2seq")


def asr_supports_translation(pipeline):
    if normalized_name(getattr(pipeline, "type", "")) == "seq2seq_whisper":
        return True
    for owner in (pipeline, getattr(pipeline, "model", None)):
        generation_config = getattr(owner, "generation_config", None)
        task_to_id = object_value(generation_config, "task_to_id")
        if isinstance(task_to_id, dict) and "translate" in task_to_id:
            return True
    return False


def asr_timestamp_value(pipeline, parameters, parameter_guard, warnings):
    word = bool(parameters.get("word_timestamps"))
    segment = bool(parameters.get("segment_timestamps"))
    if word and segment:
        parameter_guard.reject_overridden(
            "stt.segment_timestamps",
            "stt.word_timestamps",
        )
    if not word and not segment:
        return None, []
    pipeline_type = normalized_name(getattr(pipeline, "type", ""))
    paths = ["stt.word_timestamps" if word else "stt.segment_timestamps"]
    if word:
        if pipeline_type == "seq2seq":
            parameter_guard.reject(
                paths[0],
                "the selected non-Whisper seq2seq ASR pipeline cannot return timestamps",
            )
            if paths[0] not in parameter_guard.explicit:
                warnings.append(
                    "resolved word timestamps were not requested because the selected "
                    "ASR pipeline cannot produce them"
                )
            return None, paths
        return "word", paths
    if pipeline_type == "seq2seq_whisper" or not pipeline_type:
        return True, paths
    if pipeline_type in {"ctc", "ctc_with_lm"}:
        # CTC registries expose timestamped chunks at word granularity.
        return "word", paths
    parameter_guard.reject(
        paths[0],
        "the selected ASR pipeline cannot return segment timestamps",
    )
    if paths[0] not in parameter_guard.explicit:
        warnings.append(
            "resolved segment timestamps were not requested because the selected "
            "ASR pipeline cannot produce them"
        )
    return None, paths


def execute_asr(
    model_path,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    parameter_guard,
    runtime=None,
):
    entry, cache_key, model_cache_hit, resident = prepare_transformers_pipeline(
        runtime,
        model_path,
        "transformers_asr",
        "automatic-speech-recognition",
        parameters,
    )
    try:
        return execute_prepared_asr(
            entry,
            cache_key,
            model_cache_hit,
            runtime,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            parameter_guard,
        )
    finally:
        if not resident:
            cleanup_transformers_audio_entry(entry)


def execute_prepared_asr(
    entry,
    cache_key,
    model_cache_hit,
    runtime,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    parameter_guard,
):
    pipeline = entry.pipeline
    torch = entry.torch
    device = entry.device
    dtype = entry.dtype
    source_path = local_audio_source(inputs, parameters)
    decoded_audio = decoded_audio_for_pipeline(pipeline, source_path)
    warnings = []
    kwargs = {}
    return_timestamps, timestamp_paths = asr_timestamp_value(
        pipeline,
        parameters,
        parameter_guard,
        warnings,
    )
    if return_timestamps is not None:
        kwargs["return_timestamps"] = return_timestamps
    requested_operation = normalized_name(
        parameters.get("operation")
        or parameters.get("mode")
        or parameters.get("transcription_task")
        or "transcribe"
    )
    if task == "speech_translation":
        if requested_operation not in {"", "translate", "transcribe"}:
            fail("invalid_parameter", f"unsupported ASR operation '{requested_operation}'")
        if (
            requested_operation == "transcribe"
            and "stt.operation" in parameter_guard.explicit
        ):
            parameter_guard.reject(
                "stt.operation",
                "speech-translation requires operation=translate",
            )
        requested_operation = "translate"
    if requested_operation not in {"transcribe", "translate"}:
        fail("invalid_parameter", f"unsupported ASR operation '{requested_operation}'")
    if requested_operation == "translate" and not asr_supports_translation(pipeline):
        if task == "speech_translation":
            fail(
                "unsupported_task",
                "the selected ASR architecture does not advertise speech translation",
            )
        parameter_guard.reject(
            "stt.operation",
            "the selected ASR architecture does not advertise translation",
        )
        requested_operation = "transcribe"
    generative = asr_supports_generation(pipeline)
    generate_kwargs = {
        key: value
        for key, value in {
            "language": parameters.get("language") if generative else None,
            "task": requested_operation if generative else None,
            "temperature": parameters.get("temperature") if generative else None,
            "num_beams": parameters.get("beam_size") if generative else None,
        }.items()
        if value is not None
    }
    if not generative:
        for path, value in (
            ("stt.language", parameters.get("language")),
            ("stt.temperature", parameters.get("temperature")),
            ("stt.beam_size", parameters.get("beam_size")),
        ):
            if value is not None:
                parameter_guard.reject(
                    path,
                    "the selected CTC/forward-only ASR architecture has no generation controls",
                )
    generate_paths = []
    if generative and parameters.get("language") is not None:
        generate_paths.append("stt.language")
    if generative and requested_operation is not None:
        generate_paths.append("stt.operation")
    if generative and parameters.get("temperature") is not None:
        generate_paths.append("stt.temperature")
    if generative and parameters.get("beam_size") is not None:
        generate_paths.append("stt.beam_size")
    initial_prompt = parameters.get("initial_prompt") or parameters.get("prompt")
    if initial_prompt:
        prompt_ids = getattr(
            getattr(pipeline, "tokenizer", None),
            "get_prompt_ids",
            None,
        )
        if prompt_ids is None:
            parameter_guard.reject(
                "stt.initial_prompt",
                "the selected ASR pipeline tokenizer cannot encode prompt IDs",
            )
        else:
            try:
                generate_kwargs["prompt_ids"] = prompt_ids(
                    required_string(initial_prompt, "initial_prompt"),
                    return_tensors="pt",
                )
                generate_paths.append("stt.initial_prompt")
            except Exception as error:
                parameter_guard.reject(
                    "stt.initial_prompt",
                    f"the ASR tokenizer rejected the initial prompt: {error}",
                )
    if generate_kwargs:
        kwargs["generate_kwargs"] = generate_kwargs
    kwargs = filtered_kwargs(
        pipeline.__call__,
        kwargs,
        parameter_guard=parameter_guard,
        parameter_paths={
            "return_timestamps": timestamp_paths,
            "generate_kwargs": generate_paths,
        },
    )
    synchronize_torch_device(torch, device)
    inference_started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipeline(decoded_audio, **kwargs)
    except Exception as error:
        pipeline = None
        evict_failed_transformers_entry(runtime, cache_key)
        fail("execution_failed", "Transformers speech-to-text pipeline failed", str(error))
    synchronize_torch_device(torch, device)
    inference_seconds = max(0.0, time.perf_counter() - inference_started)

    encoding_started = time.perf_counter()
    if not isinstance(result, dict):
        result = {"text": str(result)}
    text = str(result.get("text") or "")
    outputs = save_transcription_outputs(
        result,
        output_dir,
        task,
        identifier,
        parameters,
    )
    encoding_seconds = max(0.0, time.perf_counter() - encoding_started)
    translated_parameters = []
    if "return_timestamps" in kwargs:
        translated_parameters.append("stt.timestamps->return_timestamps")
    if "generate_kwargs" in kwargs:
        translated_parameters.extend(
            f"stt.{key}->generate_kwargs.{target}"
            for key, target in (
                ("language", "language"),
                ("operation", "task"),
                ("temperature", "temperature"),
                ("beam_size", "num_beams"),
                ("initial_prompt", "prompt_ids"),
            )
            if target in kwargs["generate_kwargs"]
        )
    return outputs, warnings, {
        "runtime": "transformers",
        "pipeline_task": "automatic-speech-recognition",
        "pipeline_class": type(pipeline).__name__,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "text": text,
        "translated_parameters": sorted(translated_parameters),
        "model_load_seconds": 0.0 if model_cache_hit else entry.model_load_seconds,
        "model_cache_hit": model_cache_hit,
        "inference_seconds": inference_seconds,
        "encoding_seconds": encoding_seconds,
        "offload_mode": "none",
        "offload_request": "none",
    }


def requested_text_output_format(parameters, parameter_guard=None):
    format_name = normalized_name(parameters.get("output_format") or "json")
    format_name = {"txt": "text"}.get(format_name, format_name)
    if format_name in {"wav", "flac", "mp3", "ogg"} and (
        parameter_guard is None
        or "audio.output_format" not in parameter_guard.explicit
    ):
        # Audio schema defaults can be inherited while a text-producing task
        # uses its own deterministic JSON default.
        return "json"
    if format_name not in {"json", "text"}:
        fail(
            "unsupported_parameter",
            f"unsupported structured audio-analysis output format '{format_name}'",
        )
    return format_name


def save_structured_audio_text(
    value,
    text,
    output_dir,
    task,
    identifier,
    format_name,
):
    if format_name == "json":
        path = output_dir / f"{task}-{identifier}.json"
        atomic_json_write(path, value)
        return [output_record(path, "application/json")]
    path = output_dir / f"{task}-{identifier}.txt"
    path.write_text(f"{str(text).rstrip()}\n", encoding="utf-8")
    return [output_record(path, "text/plain")]


def normalized_classification_results(result):
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        fail("backend_error", "audio classification returned an unsupported result shape")
    labels = []
    for item in result:
        if not isinstance(item, dict) or "label" not in item or "score" not in item:
            fail("backend_error", "audio classification result lacks label/score values")
        try:
            score = float(item["score"])
        except (TypeError, ValueError):
            fail("backend_error", "audio classification returned an invalid score")
        if not math.isfinite(score):
            fail("backend_error", "audio classification returned a non-finite score")
        labels.append({"label": str(item["label"]), "score": score})
    labels.sort(key=lambda item: (-item["score"], item["label"]))
    return labels


def execute_audio_classification(
    model_path,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    parameter_guard,
    runtime=None,
):
    entry, cache_key, model_cache_hit, resident = prepare_transformers_pipeline(
        runtime,
        model_path,
        "transformers_audio_classification",
        "audio-classification",
        parameters,
    )
    try:
        return execute_prepared_audio_classification(
            entry,
            cache_key,
            model_cache_hit,
            runtime,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            parameter_guard,
        )
    finally:
        if not resident:
            cleanup_transformers_audio_entry(entry)


def execute_prepared_audio_classification(
    entry,
    cache_key,
    model_cache_hit,
    runtime,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    parameter_guard,
):
    format_name = requested_text_output_format(parameters, parameter_guard)
    top_k = parameters.get("top_k")
    if top_k is not None:
        top_k = positive_int(parameters, "top_k", 1)
    pipeline = entry.pipeline
    torch = entry.torch
    device = entry.device
    dtype = entry.dtype
    source_path = local_audio_source(inputs, parameters)
    decoded_audio = decoded_audio_for_pipeline(pipeline, source_path)
    kwargs = {}
    if top_k is not None:
        kwargs["top_k"] = top_k
    synchronize_torch_device(torch, device)
    inference_started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipeline(decoded_audio, **kwargs)
    except Exception as error:
        pipeline = None
        evict_failed_transformers_entry(runtime, cache_key)
        fail(
            "execution_failed",
            f"Transformers audio-classification pipeline failed for '{task}'",
            str(error),
        )
    synchronize_torch_device(torch, device)
    inference_seconds = max(0.0, time.perf_counter() - inference_started)
    encoding_started = time.perf_counter()
    labels = normalized_classification_results(result)
    text = "\n".join(
        f"{item['label']}\t{item['score']:.9g}" for item in labels
    )
    outputs = save_structured_audio_text(
        {"labels": labels},
        text,
        output_dir,
        task,
        identifier,
        format_name,
    )
    encoding_seconds = max(0.0, time.perf_counter() - encoding_started)
    return outputs, [], {
        "runtime": "transformers",
        "pipeline_task": "audio-classification",
        "pipeline_class": type(pipeline).__name__,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "text": text,
        "translated_parameters": ["audio.top_k->top_k"] if "top_k" in kwargs else [],
        "model_load_seconds": 0.0 if model_cache_hit else entry.model_load_seconds,
        "model_cache_hit": model_cache_hit,
        "inference_seconds": inference_seconds,
        "encoding_seconds": encoding_seconds,
        "offload_mode": "none",
        "offload_request": "none",
    }


def generated_audio_text(result):
    values = result if isinstance(result, list) else [result]
    texts = []
    for item in values:
        value = object_value(item, "generated_text")
        if isinstance(value, list) and value:
            last = value[-1]
            value = last.get("content") if isinstance(last, dict) else last
        if value is not None:
            texts.append(str(value))
    if not texts:
        fail("backend_error", "audio text pipeline returned no generated text")
    return "\n".join(texts).strip()


def execute_audio_text(
    model_path,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    parameter_guard,
    runtime=None,
):
    entry, cache_key, model_cache_hit, resident = prepare_transformers_pipeline(
        runtime,
        model_path,
        "transformers_audio_text",
        "any-to-any",
        parameters,
    )
    try:
        return execute_prepared_audio_text(
            entry,
            cache_key,
            model_cache_hit,
            runtime,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            parameter_guard,
        )
    finally:
        if not resident:
            cleanup_transformers_audio_entry(entry)


def execute_prepared_audio_text(
    entry,
    cache_key,
    model_cache_hit,
    runtime,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    parameter_guard,
):
    format_name = requested_text_output_format(parameters, parameter_guard)
    pipeline = entry.pipeline
    torch = entry.torch
    device = entry.device
    dtype = entry.dtype
    source_path = local_audio_source(inputs, parameters)
    decoded_audio = decoded_audio_for_pipeline(pipeline, source_path)
    prompt = parameters.get("prompt") or parameters.get("description")
    if task == "audio_captioning" and not str(prompt or "").strip():
        prompt = "Describe this audio."
    prompt = required_string(prompt, "effective_parameters.prompt/description")
    max_new_tokens = parameters.get("max_new_tokens")
    if max_new_tokens is not None:
        max_new_tokens = positive_int(parameters, "max_new_tokens", 1)
    temperature = parameters.get("temperature")
    if temperature is not None:
        temperature = positive_float(parameters, "temperature", 0.0)
    top_k = parameters.get("top_k")
    if top_k is not None:
        top_k = positive_int(parameters, "top_k", 0, minimum=0)
    top_p = parameters.get("top_p")
    if top_p is not None:
        top_p = positive_float(parameters, "top_p", 1.0)
        if top_p > 1.0:
            fail("invalid_parameter", "top_p must not exceed 1")
    sampling_requested = any(
        value is not None for value in (temperature, top_k, top_p)
    )
    generate_kwargs = {
        key: value
        for key, value in {
            "temperature": temperature if temperature not in {None, 0.0} else None,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": False if temperature == 0.0 else (
                True if sampling_requested else None
            ),
        }.items()
        if value is not None
    }
    kwargs = {"return_full_text": False}
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    if generate_kwargs:
        kwargs["generate_kwargs"] = generate_kwargs
    synchronize_torch_device(torch, device)
    inference_started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipeline(prompt, audio=decoded_audio["raw"], **kwargs)
    except Exception as error:
        pipeline = None
        evict_failed_transformers_entry(runtime, cache_key)
        fail(
            "execution_failed",
            f"Transformers any-to-any audio pipeline failed for '{task}'",
            str(error),
        )
    synchronize_torch_device(torch, device)
    inference_seconds = max(0.0, time.perf_counter() - inference_started)
    encoding_started = time.perf_counter()
    text = generated_audio_text(result)
    outputs = save_structured_audio_text(
        {"text": text},
        text,
        output_dir,
        task,
        identifier,
        format_name,
    )
    encoding_seconds = max(0.0, time.perf_counter() - encoding_started)
    return outputs, [], {
        "runtime": "transformers",
        "pipeline_task": "any-to-any",
        "pipeline_class": type(pipeline).__name__,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "text": text,
        "translated_parameters": sorted(
            (
                ["audio.max_new_tokens->max_new_tokens"]
                if max_new_tokens is not None
                else []
            )
            + [
                f"audio.{key}->generate_kwargs.{key}"
                for key in generate_kwargs
            ]
        ),
        "model_load_seconds": 0.0 if model_cache_hit else entry.model_load_seconds,
        "model_cache_hit": model_cache_hit,
        "inference_seconds": inference_seconds,
        "encoding_seconds": encoding_seconds,
        "offload_mode": "none",
        "offload_request": "none",
    }


def load_transformers_audio_embedder(
    model_path,
    parameters,
    runtime_values=None,
):
    if runtime_values is None:
        torch, device, dtype = torch_runtime(parameters)
    else:
        torch, device, dtype = runtime_values
    transformers = require_module("transformers", purpose="audio embedding")
    processor = None
    processor_errors = []
    for class_name in ("AutoProcessor", "AutoFeatureExtractor"):
        factory = getattr(transformers, class_name, None)
        if factory is None:
            continue
        try:
            processor = factory.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            break
        except Exception as error:
            processor_errors.append(f"{class_name}: {error}")
    if processor is None:
        fail(
            "model_load_failed",
            "failed to load a local audio processor/feature extractor",
            processor_errors,
        )
    auto_model = getattr(transformers, "AutoModel", None)
    if auto_model is None:
        fail("missing_dependency", "installed transformers has no AutoModel registry")
    model_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if device != "cpu":
        model_kwargs["dtype"] = dtype
    try:
        model = auto_model.from_pretrained(str(model_path), **model_kwargs)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model.to(device)
    except Exception as error:
        fail("model_load_failed", "failed to load local audio embedding model", str(error))
    return processor, model, torch, device, dtype


def prepare_transformers_audio_embedder(runtime, model_path, parameters):
    adapter = "transformers_audio_embedding"
    pipeline_task = "audio-embedding"

    def loader(runtime_values):
        processor, model, torch, device, dtype = (
            load_transformers_audio_embedder(
                model_path,
                parameters,
                runtime_values=runtime_values,
            )
        )
        return TransformersAudioEntry(
            torch,
            device,
            dtype,
            0.0,
            adapter=adapter,
            pipeline_task=pipeline_task,
            processor=processor,
            model=model,
        )

    return prepare_transformers_audio_entry(
        runtime,
        model_path,
        adapter,
        pipeline_task,
        parameters,
        loader,
    )


def process_audio_waveform(processor, waveform, sample_rate):
    """Call a registered processor without confusing audio for another modality."""
    call_kwargs = {
        "sampling_rate": sample_rate,
        "return_tensors": "pt",
    }
    try:
        parameters = inspect.signature(processor.__call__).parameters
    except Exception:
        parameters = {}
    for keyword in ("audio", "audios", "raw_speech", "raw_audio", "waveform"):
        if keyword in parameters:
            call_kwargs[keyword] = waveform
            return processor(**call_kwargs)
    return processor(waveform, **call_kwargs)


def embedding_vector(value, inputs, normalize):
    torch = require_module("torch", purpose="audio embedding")
    if value is None:
        fail("backend_error", "audio embedding model returned no hidden representation")
    if not hasattr(value, "ndim"):
        value = torch.as_tensor(value)
    if value.ndim >= 3:
        attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None
        if (
            attention_mask is not None
            and value.ndim == 3
            and attention_mask.ndim == 2
            and attention_mask.shape[1] == value.shape[1]
        ):
            mask = attention_mask.to(value.device, dtype=value.dtype).unsqueeze(-1)
            value = (value * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            while value.ndim > 2:
                value = value.mean(dim=1)
    if value.ndim == 2:
        if value.shape[0] != 1:
            fail("backend_error", "audio embedding model returned an unexpected batch size")
        value = value[0]
    value = value.reshape(-1)
    if normalize:
        norm = value.float().norm(p=2)
        norm_value = norm.item() if hasattr(norm, "item") else float(norm)
        if float(norm_value) > 0:
            value = value / norm.to(value.dtype)
    array = numpy_audio(value).astype("float32", copy=False)
    if array.size == 0 or not bool(require_module("numpy").isfinite(array).all()):
        fail("backend_error", "audio embedding contains no finite values")
    return [float(item) for item in array.tolist()]


def execute_audio_embedding(
    model_path,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    parameter_guard,
    runtime=None,
):
    format_name = requested_text_output_format(parameters, parameter_guard)
    pooling = normalized_name(parameters.get("pooling") or "mean")
    if pooling != "mean":
        parameter_guard.reject(
            "audio.pooling",
            "the generic audio embedding adapter supports only mean pooling",
        )
    entry, cache_key, model_cache_hit, resident = (
        prepare_transformers_audio_embedder(
            runtime,
            model_path,
            parameters,
        )
    )
    try:
        return execute_prepared_audio_embedding(
            entry,
            cache_key,
            model_cache_hit,
            runtime,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            format_name,
        )
    finally:
        if not resident:
            cleanup_transformers_audio_entry(entry)


def execute_prepared_audio_embedding(
    entry,
    cache_key,
    model_cache_hit,
    runtime,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    format_name,
):
    processor = entry.processor
    model = entry.model
    torch = entry.torch
    device = entry.device
    dtype = entry.dtype
    holder = type("AudioEmbeddingPipeline", (), {"processor": processor, "model": model})()
    source_path = local_audio_source(inputs, parameters)
    decoded_audio = decoded_audio_for_pipeline(holder, source_path)
    try:
        model_inputs = process_audio_waveform(
            processor,
            decoded_audio["raw"],
            decoded_audio["sampling_rate"],
        )
    except Exception as error:
        fail("invalid_input", "audio processor rejected the decoded waveform", str(error))
    if not isinstance(model_inputs, dict):
        model_inputs = dict(model_inputs)
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }
    synchronize_torch_device(torch, device)
    inference_started = time.perf_counter()
    try:
        with torch.inference_mode():
            get_audio_features = getattr(model, "get_audio_features", None)
            if callable(get_audio_features):
                representation = get_audio_features(**model_inputs)
            else:
                result = model(**model_inputs)
                representation = object_value(result, "audio_embeds")
                if representation is None:
                    representation = object_value(result, "embeddings")
                if representation is None:
                    representation = object_value(result, "last_hidden_state")
                if representation is None:
                    hidden_states = object_value(result, "hidden_states")
                    if isinstance(hidden_states, (list, tuple)) and hidden_states:
                        representation = hidden_states[-1]
    except Exception as error:
        model = None
        evict_failed_transformers_entry(runtime, cache_key)
        fail("execution_failed", "Transformers audio embedding model failed", str(error))
    synchronize_torch_device(torch, device)
    inference_seconds = max(0.0, time.perf_counter() - inference_started)
    encoding_started = time.perf_counter()
    normalize = bool(parameters.get("normalize", True))
    vector = embedding_vector(representation, model_inputs, normalize)
    output_value = {"embedding": vector, "dimensions": len(vector)}
    outputs = save_structured_audio_text(
        output_value,
        " ".join(f"{value:.9g}" for value in vector),
        output_dir,
        task,
        identifier,
        format_name,
    )
    outputs[0]["metadata"] = {
        "dimensions": len(vector),
        "normalized": normalize,
    }
    encoding_seconds = max(0.0, time.perf_counter() - encoding_started)
    return outputs, [], {
        "runtime": "transformers",
        "pipeline_task": "audio-embedding",
        "pipeline_class": type(model).__name__,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "embedding_dimensions": len(vector),
        "translated_parameters": [
            "audio.normalize->l2_normalize",
            "audio.pooling->mean_pooling",
        ],
        "model_load_seconds": 0.0 if model_cache_hit else entry.model_load_seconds,
        "model_cache_hit": model_cache_hit,
        "inference_seconds": inference_seconds,
        "encoding_seconds": encoding_seconds,
        "offload_mode": "none",
        "offload_request": "none",
    }


def output_record(
    path,
    mime_type,
    width=None,
    height=None,
    duration=None,
    metadata=None,
):
    return {
        "path": str(path),
        "mime_type": mime_type,
        "size": safe_size(path),
        "width": width,
        "height": height,
        "duration": duration,
        "metadata": metadata or {},
    }


def atomic_json_write(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(value), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except Exception:
            pass


def metadata_effective_parameters(parameters, adapter):
    values = dict(parameters)
    if adapter == QWEN3_TTS_ADAPTER:
        # Generated speech and natural-language voice instructions can contain
        # user-sensitive content.  Keep only parameter names/translation
        # metadata, never their text, in persisted Qwen execution metadata.
        for key in (
            "text",
            "prompt",
            "tts.speaking_style",
            "speaking_style",
            "tts.instruct",
            "instruct",
        ):
            values.pop(key, None)
    return values


def command_execute(payload, runtime=None):
    model_path = local_model_path(payload)
    task = normalized_name(payload.get("task"))
    if not task:
        fail("invalid_request", "task is required")
    parameters = normalized_parameters(payload)
    adapter = execution_adapter(model_path, task)
    if adapter is None:
        fail("unsupported_task", f"no executable companion adapter for task '{task}'")
    validate_adapter_offload(adapter, parameters)
    parameter_guard = ExplicitParameterGuard(
        payload,
        task,
        adapter,
        parameters,
    )
    parameter_guard.validate_supported(
        supported_explicit_parameters(task, adapter)
    )
    parameters = parameter_guard.without_unsupported(parameters)
    inputs = input_values(payload, parameters)
    validate_adapter_inputs(task, adapter, inputs)
    output_dir = output_directory(payload)
    identifier = uuid.uuid4().hex
    created_unix = time.time()
    started = time.perf_counter()

    if adapter == "diffusers":
        outputs, warnings, backend_metadata = execute_diffusers(
            payload,
            model_path,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            parameter_guard,
            runtime=runtime,
        )
    elif adapter == "diffusers_audio":
        outputs, warnings, backend_metadata = execute_diffusers_audio(
            payload,
            model_path,
            task,
            parameters,
            output_dir,
            identifier,
            parameter_guard,
            runtime=runtime,
        )
    elif adapter == "transformers_audio":
        outputs, warnings, backend_metadata = execute_audio_generation(
            model_path,
            task,
            parameters,
            output_dir,
            identifier,
            parameter_guard,
            runtime=runtime,
        )
    elif adapter == "transformers_tts":
        outputs, warnings, backend_metadata = execute_tts(
            model_path,
            task,
            parameters,
            output_dir,
            identifier,
            runtime=runtime,
        )
    elif adapter == QWEN3_TTS_ADAPTER:
        outputs, warnings, backend_metadata = execute_qwen3_tts_voice_design(
            model_path,
            task,
            parameters,
            output_dir,
            identifier,
            runtime=runtime,
        )
    elif adapter == "transformers_asr":
        outputs, warnings, backend_metadata = execute_asr(
            model_path,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            parameter_guard,
            runtime=runtime,
        )
    elif adapter == "transformers_audio_classification":
        outputs, warnings, backend_metadata = execute_audio_classification(
            model_path,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            parameter_guard,
            runtime=runtime,
        )
    elif adapter == "transformers_audio_text":
        outputs, warnings, backend_metadata = execute_audio_text(
            model_path,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            parameter_guard,
            runtime=runtime,
        )
    elif adapter == "transformers_audio_embedding":
        outputs, warnings, backend_metadata = execute_audio_embedding(
            model_path,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            parameter_guard,
            runtime=runtime,
        )
    else:
        fail("internal_error", f"unknown execution adapter '{adapter}'")

    warnings.extend(parameter_guard.warnings)
    parameters = parameter_guard.without_unsupported(parameters)
    backend_metadata["parameter_support"] = parameter_guard.metadata()

    metadata = {
        "id": identifier,
        "task": task,
        "model_path": str(model_path),
        "runtime": "werk-media-companion",
        "backend": backend_metadata,
        "effective_parameters": metadata_effective_parameters(parameters, adapter),
        "outputs": outputs,
        "warnings": warnings,
        "created_unix": int(created_unix),
        "elapsed_seconds": max(0.0, time.perf_counter() - started),
        "offline": True,
    }
    metadata_path = output_dir / f"{task}-{identifier}.metadata.json"
    atomic_json_write(metadata_path, metadata)
    metadata["metadata_path"] = str(metadata_path)
    return {
        "task": task,
        "outputs": outputs,
        "metadata": metadata,
        "warnings": warnings,
    }


def dispatch(operation, payload, runtime=None):
    commands = {
        "health": command_health,
        "capabilities": command_capabilities,
        "probe-model": command_probe_model,
        "estimate": command_estimate,
        "execute": command_execute,
    }
    handler = commands.get(operation)
    if handler is None:
        fail("unknown_command", f"unknown companion command '{operation}'")
    if operation == "execute":
        return handler(payload, runtime=runtime)
    return handler(payload)


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


def response_error(error):
    if isinstance(error, CompanionFailure):
        code = error.code
        message = error.message
        detail = error.detail
    else:
        code = "internal_error"
        message = str(error) or error.__class__.__name__
        detail = None
        if os.environ.get("WERK_MEDIA_DEBUG") in {"1", "true", "yes", "on"}:
            detail = traceback.format_exc()
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "detail": json_safe(detail),
        },
    }


def write_json_line(stream, value):
    encoded = json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))
    stream.write(encoded)
    stream.write("\n")
    stream.flush()


def isolate_resident_protocol_output():
    """Reserve the original stdout pipe for JSONL and redirect fd 1 to stderr.

    ``redirect_stdout`` only replaces Python's ``sys.stdout`` object. Native
    extensions and child processes can still write directly to file descriptor
    1 and corrupt a long-lived framed transport. Duplicating the original pipe
    first gives protocol writes a private descriptor while all later fd-1
    output is routed to the diagnostic stream.
    """
    original_stdout = sys.stdout
    protocol_fd = None
    protocol_stream = None
    try:
        original_stdout.flush()
        sys.stderr.flush()
        protocol_fd = os.dup(original_stdout.fileno())
        protocol_stream = os.fdopen(
            protocol_fd,
            "w",
            encoding=getattr(original_stdout, "encoding", None) or "utf-8",
            errors="backslashreplace",
            buffering=1,
        )
        protocol_fd = None
        os.dup2(sys.stderr.fileno(), original_stdout.fileno())
        return protocol_stream, True
    except Exception:
        if protocol_stream is not None:
            try:
                protocol_stream.close()
            except Exception:
                pass
        if protocol_fd is not None:
            try:
                os.close(protocol_fd)
            except Exception:
                pass
        return original_stdout, False


def resident_response(request, runtime):
    request_id = request.get("request_id") if isinstance(request, dict) else None
    try:
        if not isinstance(request, dict):
            fail("invalid_request", "resident request must be a JSON object")
        if request.get("transport_version") != TRANSPORT_VERSION:
            fail(
                "transport_version_mismatch",
                f"resident transport requires version {TRANSPORT_VERSION}",
                {"expected": TRANSPORT_VERSION, "received": request.get("transport_version")},
            )
        if request_id is None or isinstance(request_id, (dict, list)):
            fail("invalid_request", "request_id must be a JSON scalar")
        operation = normalized_name(request.get("operation")).replace("_", "-")
        if not operation:
            fail("invalid_request", "operation is required")
        payload = request.get("payload", {})
        if not isinstance(payload, dict):
            fail("invalid_request", "payload must be a JSON object")
        if operation == "shutdown":
            body = {"status": "shutting_down"}
            keep_running = False
        else:
            body = dispatch(operation, payload, runtime=runtime)
            keep_running = True
        response = {"ok": True}
        if isinstance(body, dict):
            response.update(body)
        else:
            response["result"] = body
        # Envelope fields are transport-owned and cannot be overwritten by an
        # operation response.
        response.update(
            {
                "transport_version": TRANSPORT_VERSION,
                "request_id": request_id,
                "ok": True,
            }
        )
        return response, keep_running
    except Exception as error:
        response = {
            "transport_version": TRANSPORT_VERSION,
            "request_id": request_id,
        }
        response.update(response_error(error))
        return response, True


def serve_loop(input_stream, output_stream, runtime=None):
    runtime = runtime or CompanionRuntime()
    try:
        for line in input_stream:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except Exception as error:
                response = {
                    "transport_version": TRANSPORT_VERSION,
                    "request_id": None,
                }
                response.update(
                    response_error(
                        CompanionFailure(
                            "invalid_json",
                            "resident request line must contain one JSON object",
                            str(error),
                        )
                    )
                )
                write_json_line(output_stream, response)
                continue
            response, keep_running = resident_response(request, runtime)
            write_json_line(output_stream, response)
            if not keep_running:
                break
    finally:
        runtime.close()


def main():
    original_stdout = sys.stdout
    if len(sys.argv) == 2 and normalized_name(sys.argv[1]) == "serve":
        protocol_stdout, owns_protocol_stdout = isolate_resident_protocol_output()
        try:
            runtime = CompanionRuntime()
        except Exception as error:
            response = {
                "transport_version": TRANSPORT_VERSION,
                "request_id": None,
            }
            response.update(response_error(error))
            write_json_line(protocol_stdout, response)
            if owns_protocol_stdout:
                protocol_stdout.close()
            return
        # Keep the entire lifetime protected: third-party libraries may print
        # after a call has returned. Python stdout and native fd 1 both point
        # at stderr; only protocol_stdout retains the framed transport pipe.
        try:
            with contextlib.redirect_stdout(sys.stderr):
                serve_loop(sys.stdin, protocol_stdout, runtime=runtime)
        finally:
            if owns_protocol_stdout:
                protocol_stdout.close()
        return
    try:
        if len(sys.argv) != 2:
            fail("invalid_command", "expected exactly one command argument")
        operation = normalized_name(sys.argv[1]).replace("_", "-")
        try:
            payload = json.load(sys.stdin)
        except Exception as error:
            fail("invalid_json", "stdin must contain one JSON object", str(error))
        if not isinstance(payload, dict):
            fail("invalid_request", "stdin JSON value must be an object")
        # Third-party imports and pipelines occasionally print progress or
        # warnings to stdout. Redirect all such output to stderr so stdout
        # remains a single protocol object.
        with contextlib.redirect_stdout(sys.stderr):
            body = dispatch(operation, payload)
        response = {"ok": True}
        if isinstance(body, dict):
            response.update(body)
        else:
            response["result"] = body
    except BaseException as error:
        response = response_error(error)

    write_json_line(original_stdout, response)


if __name__ == "__main__":
    main()
