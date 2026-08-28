"""Native Werk1112 nodes for ComfyUI."""

from __future__ import annotations

import json
import math
import time
from typing import Any, Mapping
from urllib.parse import quote

try:
    from .client import WerkApiError, WerkClient
    from .config import (
        DEFAULT_TIMEOUT_SECONDS,
        WerkAudioConfig,
        WerkConnection,
        WerkImageConfig,
        WerkRoutingConfig,
        WerkVisionConfig,
        WerkVideoConfig,
        environment_api_key,
        environment_max_audio_input_bytes,
        environment_max_audio_bytes,
        environment_max_image_pixels,
        environment_max_vision_input_bytes,
        environment_max_video_bytes,
        environment_server_url,
    )
    from .audio_utils import audio_bytes_to_comfy, comfy_audio_to_api_input
    from .image_utils import (
        batch_image_tensors,
        comfy_images_to_data_urls,
        decode_base64_image,
        image_bytes_to_tensor,
    )
    from .video_utils import image_tensor_to_api_input, video_bytes_to_comfy
except ImportError:  # pragma: no cover - direct-module development
    from client import WerkApiError, WerkClient
    from config import (
        DEFAULT_TIMEOUT_SECONDS,
        WerkAudioConfig,
        WerkConnection,
        WerkImageConfig,
        WerkRoutingConfig,
        WerkVisionConfig,
        WerkVideoConfig,
        environment_api_key,
        environment_max_audio_input_bytes,
        environment_max_audio_bytes,
        environment_max_image_pixels,
        environment_max_vision_input_bytes,
        environment_max_video_bytes,
        environment_server_url,
    )
    from audio_utils import audio_bytes_to_comfy, comfy_audio_to_api_input
    from image_utils import (
        batch_image_tensors,
        comfy_images_to_data_urls,
        decode_base64_image,
        image_bytes_to_tensor,
    )
    from video_utils import image_tensor_to_api_input, video_bytes_to_comfy


IMAGE_TASK = "image-generation"
VISION_TASK = "image-understanding"
VIDEO_GENERATION_TASK = "video-generation"
IMAGE_TO_VIDEO_TASK = "image-to-video"
VIDEO_TASKS = (VIDEO_GENERATION_TASK, IMAGE_TO_VIDEO_TASK)
AUDIO_GENERATION_TASK = "audio-generation"
MUSIC_GENERATION_TASK = "music-generation"
TEXT_TO_SPEECH_TASK = "text-to-speech"
AUDIO_GENERATION_TASKS = (
    AUDIO_GENERATION_TASK,
    MUSIC_GENERATION_TASK,
    TEXT_TO_SPEECH_TASK,
)
AUDIO_TRANSCRIPTION_TASKS = ("speech-to-text", "speech-translation")
AUDIO_DETECTION_TASKS = (
    "audio-event-detection",
    "voice-activity-detection",
    "speaker-identification",
    "language-identification",
    "speech-emotion-recognition",
)
AUDIO_ANALYSIS_TASKS = (
    "audio-captioning",
    "speaker-diarization",
    "audio-classification",
    "audio-understanding",
)
AUDIO_TRANSFORM_TASKS = (
    "voice-conversion",
    "stem-separation",
    "audio-enhancement",
    "audio-editing",
)
AUDIO_EMBEDDING_TASKS = ("audio-embedding",)
AUDIO_TEXT_OUTPUT_TASKS = (
    *AUDIO_TRANSCRIPTION_TASKS,
    *AUDIO_DETECTION_TASKS,
    *AUDIO_ANALYSIS_TASKS,
    *AUDIO_EMBEDDING_TASKS,
)
AUDIO_TASKS = (
    *AUDIO_GENERATION_TASKS,
    *AUDIO_TRANSCRIPTION_TASKS,
    *AUDIO_DETECTION_TASKS,
    *AUDIO_ANALYSIS_TASKS,
    *AUDIO_TRANSFORM_TASKS,
    *AUDIO_EMBEDDING_TASKS,
)
MAX_AUDIO_TEXT_BYTES = 16 * 1024 * 1024
DEDICATED_PARAMETERS = {
    "image.width",
    "image.height",
    "image.num_images",
    "image.steps",
    "image.guidance",
    "image.seed",
    "image.output_format",
}
IMAGE_CONFIG_DEDICATED_PARAMETERS = DEDICATED_PARAMETERS | {
    "image.batch_size",
    "image.vae_tiling",
    "image.vae_slicing",
}
VIDEO_CONFIG_DEDICATED_PARAMETERS = {
    "video.width",
    "video.height",
    "video.frames",
    "video.fps",
    "video.batch_size",
    "video.num_videos",
    "video.steps",
    "video.guidance",
    "video.seed",
    "video.temporal_vae_tiling",
    "video.output_format",
}
AUDIO_CONFIG_DEDICATED_PARAMETERS = {
    "audio.duration",
    "audio.variations",
    "audio.seed",
    "audio.sample_rate",
    "audio.channels",
    "audio.instrumental",
    "audio.output_format",
}
TTS_CONFIG_DEDICATED_PARAMETERS = {
    "tts.voice",
    "tts.speed",
    "tts.seed",
    "tts.sample_rate",
    "tts.channels",
    "tts.output_format",
}
ROUTING_OPTION_PATHS = {
    "backend": "routing.backend",
    "accelerator": "routing.accelerator",
    "device": "routing.device",
    "precision": "routing.precision",
    "quantization": "routing.quantization",
    "profile": "routing.profile",
    "quality": "routing.quality",
    "performance_preference": "routing.performance_preference",
    "fallback_policy": "routing.fallback_policy",
    "parameter_policy": "routing.parameter_policy",
    "allow_cpu_offload": "routing.allow_cpu_offload",
    "allow_sequential_offload": "routing.allow_sequential_offload",
    "allow_component_offload": "routing.allow_component_offload",
    "allow_disk_offload": "routing.allow_disk_offload",
    "attention_backend": "routing.attention_backend",
    "compile": "routing.compile",
    "timeout_seconds": "routing.timeout",
}
ROUTING_DEDICATED_PARAMETERS = set(ROUTING_OPTION_PATHS.values())
ROUTING_ENUM_VALUES = {
    "quality": {"draft", "balanced", "high", "maximum"},
    "performance_preference": {
        "quality",
        "balanced",
        "speed",
        "latency",
        "throughput",
        "memory",
    },
    "fallback_policy": {"none", "backend", "degrade"},
    "parameter_policy": {"strict", "warn", "permissive"},
}
TRISTATE_VALUES = {"inherit", "enabled", "disabled"}
RESERVED_ADDITIONAL_KEYS = {
    "model",
    "prompt",
    "negative_prompt",
    "initial_image",
    "input",
    "task",
    "async",
    "background",
    "job",
    "voice",
    "speed",
    "n",
    "size",
    "response_format",
    "output_format",
    "style",
    "quality",
    "parameter_policy",
    "routing",
    "backend",
    "accelerator",
    "device",
    "precision",
    "quantization",
    "profile",
    "performance_preference",
    "fallback_policy",
    "allow_cpu_offload",
    "allow_sequential_offload",
    "allow_component_offload",
    "allow_disk_offload",
    "attention_backend",
    "compile",
    "timeout_seconds",
    "user",
    "server_url",
    "api_key",
    "authorization",
    "headers",
    "url",
}
ALLOWED_IMAGE_REQUEST_FIELDS = {
    "n",
    "size",
    "response_format",
    "output_format",
    "style",
}
ALLOWED_VIDEO_REQUEST_FIELDS = {"n", "size", "response_format"}
ALLOWED_AUDIO_GENERATION_REQUEST_FIELDS = {"n", "response_format"}
ALLOWED_TTS_REQUEST_FIELDS = {"voice", "speed", "response_format"}
ALLOWED_VISION_REQUEST_FIELDS = {
    "temperature",
    "top_p",
    "max_completion_tokens",
    "stop",
    "seed",
}


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Werk {label} response must be a JSON object")
    return value


def _model_entries(payload: Any) -> list[dict[str, Any]]:
    root = _object(payload, "models")
    entries = root.get("data", [])
    if not isinstance(entries, list):
        raise ValueError("Werk models response field 'data' must be an array")
    return [
        entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]


def _capability_entries(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return []
    return [
        entry
        for entry in payload["models"]
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]


def _normalized_tasks(value: Any) -> list[str]:
    """Normalize Werk's JSON enum spelling and older display spelling."""

    if not isinstance(value, list):
        return []
    tasks: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        task = item.strip().lower().replace("_", "-")
        if task and task not in tasks:
            tasks.append(task)
    return tasks


def _task_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("_", "-")


def _task_statuses(capability: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Keep the server's structured readiness metadata intact for consumers."""

    statuses = capability.get("task_statuses", model.get("task_statuses", {}))
    return statuses if isinstance(statuses, dict) else {}


def _task_status_for_model(model: dict[str, Any], task: str) -> dict[str, Any]:
    statuses = model.get("task_statuses", {})
    if not isinstance(statuses, dict):
        return {}
    for status_task, status in statuses.items():
        if _task_name(status_task) == task and isinstance(status, dict):
            return status
    return {}


def _unavailable_task_message(
    task: str,
    model_ids: list[str],
    metadata: list[dict[str, Any]],
) -> str:
    """Render only authoritative readiness actions returned by Werk.

    Install commands are never inferred from backend or dependency names. A
    missing generic adapter is materially different from a missing package and
    must not be presented as something pip/cargo can fix.
    """

    models_by_id = {model["id"]: model for model in metadata}
    statuses = [
        (model_id, _task_status_for_model(models_by_id.get(model_id, {}), task))
        for model_id in model_ids
    ]
    base = (
        f"Werk models declare {task}, but none is currently runtime "
        f"probe-eligible: {', '.join(model_ids)}."
    )

    install_actions: list[tuple[str, str]] = []
    for model_id, status in statuses:
        command = status.get("install_command")
        if (
            _task_name(status.get("status")) == "installable"
            and isinstance(command, str)
            and command.strip()
        ):
            action = (model_id, command.strip())
            if action not in install_actions:
                install_actions.append(action)
    if install_actions:
        rendered = "; ".join(
            f"{model_id}: {command}" for model_id, command in install_actions
        )
        return f"{base} Install the required backend: {rendered}"

    not_implemented = [
        model_id
        for model_id, status in statuses
        if _task_name(status.get("status")) == "not-implemented"
    ]
    if not_implemented:
        return (
            f"{base} No registered Werk adapter exists for {task} on: "
            f"{', '.join(not_implemented)}. Installing a package alone will not "
            "add support; a compatible adapter must be implemented or configured."
        )

    doctor_commands = "; ".join(
        f"werk doctor --model {model_id} --task {task}" for model_id in model_ids
    )
    return f"{base} Diagnose the runtime with: {doctor_commands}"


def classify_image_models(
    models_payload: Any, capabilities_payload: Any
) -> dict[str, Any]:
    installed = _model_entries(models_payload)
    installed_ids = [entry["id"] for entry in installed]
    capability_by_id = {
        entry["id"]: entry for entry in _capability_entries(capabilities_payload)
    }
    declared: list[str] = []
    available: list[str] = []
    metadata: list[dict[str, Any]] = []
    for model in installed:
        model_id = model["id"]
        capability = capability_by_id.get(model_id, model)
        tasks = _normalized_tasks(capability.get("tasks", model.get("tasks", [])))
        available_tasks = _normalized_tasks(
            capability.get("available_tasks", model.get("available_tasks", []))
        )
        task_statuses = _task_statuses(capability, model)
        if IMAGE_TASK in tasks:
            declared.append(model_id)
        if IMAGE_TASK in available_tasks:
            available.append(model_id)
        metadata.append(
            {
                "id": model_id,
                "declares_image_generation": IMAGE_TASK in tasks,
                "image_generation_probe_eligible": IMAGE_TASK in available_tasks,
                "tasks": tasks,
                "available_tasks": available_tasks,
                "task_statuses": task_statuses,
            }
        )
    return {
        "installed": installed_ids,
        "declared": declared,
        "available": available,
        "models": metadata,
    }


def classify_vision_models(
    models_payload: Any, capabilities_payload: Any
) -> dict[str, Any]:
    """Classify models by Werk's authoritative image-understanding task."""

    installed = _model_entries(models_payload)
    installed_ids = [entry["id"] for entry in installed]
    capability_by_id = {
        entry["id"]: entry for entry in _capability_entries(capabilities_payload)
    }
    declared: list[str] = []
    available: list[str] = []
    metadata: list[dict[str, Any]] = []
    for model in installed:
        model_id = model["id"]
        capability = capability_by_id.get(model_id, model)
        tasks = _normalized_tasks(capability.get("tasks", model.get("tasks", [])))
        available_tasks = _normalized_tasks(
            capability.get("available_tasks", model.get("available_tasks", []))
        )
        task_statuses = _task_statuses(capability, model)
        if VISION_TASK in tasks:
            declared.append(model_id)
        if VISION_TASK in available_tasks:
            available.append(model_id)
        metadata.append(
            {
                "id": model_id,
                "declares_image_understanding": VISION_TASK in tasks,
                "image_understanding_probe_eligible": (
                    VISION_TASK in available_tasks
                ),
                "tasks": tasks,
                "available_tasks": available_tasks,
                "task_statuses": task_statuses,
            }
        )
    return {
        "installed": installed_ids,
        "declared": declared,
        "available": available,
        "models": metadata,
    }


def classify_video_models(
    models_payload: Any, capabilities_payload: Any
) -> dict[str, Any]:
    """Classify text-to-video and image-to-video support independently."""

    installed = _model_entries(models_payload)
    installed_ids = [entry["id"] for entry in installed]
    capability_by_id = {
        entry["id"]: entry for entry in _capability_entries(capabilities_payload)
    }
    by_task = {
        task: {"declared": [], "available": []} for task in VIDEO_TASKS
    }
    declared: list[str] = []
    available: list[str] = []
    metadata: list[dict[str, Any]] = []
    for model in installed:
        model_id = model["id"]
        capability = capability_by_id.get(model_id, model)
        tasks = _normalized_tasks(capability.get("tasks", model.get("tasks", [])))
        available_tasks = _normalized_tasks(
            capability.get("available_tasks", model.get("available_tasks", []))
        )
        task_statuses = _task_statuses(capability, model)
        for task in VIDEO_TASKS:
            if task in tasks:
                by_task[task]["declared"].append(model_id)
                if model_id not in declared:
                    declared.append(model_id)
            if task in available_tasks:
                by_task[task]["available"].append(model_id)
                if model_id not in available:
                    available.append(model_id)
        metadata.append(
            {
                "id": model_id,
                "declares_video_generation": VIDEO_GENERATION_TASK in tasks,
                "video_generation_probe_eligible": (
                    VIDEO_GENERATION_TASK in available_tasks
                ),
                "declares_image_to_video": IMAGE_TO_VIDEO_TASK in tasks,
                "image_to_video_probe_eligible": IMAGE_TO_VIDEO_TASK in available_tasks,
                "tasks": tasks,
                "available_tasks": available_tasks,
                "task_statuses": task_statuses,
            }
        )
    return {
        "installed": installed_ids,
        "declared": declared,
        "available": available,
        "by_task": by_task,
        "models": metadata,
    }


def classify_audio_models(
    models_payload: Any, capabilities_payload: Any
) -> dict[str, Any]:
    """Classify every audio task accepted by Werk's generic job API."""

    installed = _model_entries(models_payload)
    installed_ids = [entry["id"] for entry in installed]
    capability_by_id = {
        entry["id"]: entry for entry in _capability_entries(capabilities_payload)
    }
    by_task = {task: {"declared": [], "available": []} for task in AUDIO_TASKS}
    declared: list[str] = []
    available: list[str] = []
    metadata: list[dict[str, Any]] = []
    for model in installed:
        model_id = model["id"]
        capability = capability_by_id.get(model_id, model)
        tasks = _normalized_tasks(capability.get("tasks", model.get("tasks", [])))
        available_tasks = _normalized_tasks(
            capability.get("available_tasks", model.get("available_tasks", []))
        )
        task_statuses = _task_statuses(capability, model)
        for task in AUDIO_TASKS:
            if task in tasks:
                by_task[task]["declared"].append(model_id)
                if model_id not in declared:
                    declared.append(model_id)
            if task in available_tasks:
                by_task[task]["available"].append(model_id)
                if model_id not in available:
                    available.append(model_id)
        metadata.append(
            {
                "id": model_id,
                "declares_audio_generation": AUDIO_GENERATION_TASK in tasks,
                "audio_generation_probe_eligible": (
                    AUDIO_GENERATION_TASK in available_tasks
                ),
                "declares_music_generation": MUSIC_GENERATION_TASK in tasks,
                "music_generation_probe_eligible": (
                    MUSIC_GENERATION_TASK in available_tasks
                ),
                "declares_text_to_speech": TEXT_TO_SPEECH_TASK in tasks,
                "text_to_speech_probe_eligible": TEXT_TO_SPEECH_TASK in available_tasks,
                "tasks": tasks,
                "available_tasks": available_tasks,
                "task_statuses": task_statuses,
            }
        )
    return {
        "installed": installed_ids,
        "declared": declared,
        "available": available,
        "by_task": by_task,
        "models": metadata,
    }


def _video_task(value: str) -> str:
    task = str(value or "").strip().lower().replace("_", "-")
    if task not in VIDEO_TASKS:
        raise ValueError(
            "task must be video-generation or image-to-video"
        )
    return task


def _audio_task(value: str) -> str:
    task = str(value or "").strip().lower().replace("_", "-")
    if task not in AUDIO_TASKS:
        raise ValueError("task is not a supported Werk audio task")
    return task


def _audio_generation_task(value: str) -> str:
    task = _audio_task(value)
    if task not in AUDIO_GENERATION_TASKS:
        raise ValueError(
            "generation task must be audio-generation, music-generation, "
            "or text-to-speech"
        )
    return task


def _normalize_parameter_object(
    value: str,
    *,
    namespace: str,
    dedicated_parameters: set[str],
    label: str,
) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object")

    flattened: list[tuple[str, Any]] = []
    for raw_name, parameter_value in parsed.items():
        if raw_name == namespace:
            if not isinstance(parameter_value, dict):
                raise ValueError(
                    f"{label} field '{namespace}' must contain a JSON object"
                )
            flattened.extend(
                (f"{namespace}.{name}", child)
                for name, child in parameter_value.items()
            )
        else:
            flattened.append((raw_name, parameter_value))

    normalized: dict[str, Any] = {}
    for raw_name, parameter_value in flattened:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("additional parameter names must be non-empty strings")
        name = raw_name.strip()
        lowered = name.lower()
        first_segment = lowered.split(".", 1)[0]
        namespaced_name = lowered.removeprefix(f"{namespace}.")
        canonical = name if "." in name else f"{namespace}.{name}"
        if canonical in dedicated_parameters:
            raise ValueError(
                f"additional parameter '{name}' duplicates a dedicated node input"
            )
        if (
            (lowered != namespace and lowered in RESERVED_ADDITIONAL_KEYS)
            or (
                first_segment != namespace
                and first_segment in RESERVED_ADDITIONAL_KEYS
            )
            or namespaced_name in RESERVED_ADDITIONAL_KEYS
        ):
            raise ValueError(
                f"additional parameter '{name}' is a reserved request or routing field"
            )
        if not canonical.startswith(f"{namespace}."):
            raise ValueError(
                f"additional parameter '{name}' must use the '{namespace}.' namespace"
            )
        if canonical in normalized:
            raise ValueError(
                f"additional parameter '{name}' is duplicated after normalization"
            )
        normalized[canonical] = parameter_value
    return normalized


def normalize_image_config_parameters(value: str) -> dict[str, Any]:
    return _normalize_parameter_object(
        value,
        namespace="image",
        dedicated_parameters=IMAGE_CONFIG_DEDICATED_PARAMETERS,
        label="additional_image_parameters_json",
    )


def normalize_video_config_parameters(value: str) -> dict[str, Any]:
    return _normalize_parameter_object(
        value,
        namespace="video",
        dedicated_parameters=VIDEO_CONFIG_DEDICATED_PARAMETERS,
        label="additional_video_parameters_json",
    )


def normalize_audio_config_parameters(value: str, task: str) -> dict[str, Any]:
    selected_task = _audio_generation_task(task)
    namespace = "tts" if selected_task == TEXT_TO_SPEECH_TASK else "audio"
    dedicated = (
        TTS_CONFIG_DEDICATED_PARAMETERS
        if namespace == "tts"
        else AUDIO_CONFIG_DEDICATED_PARAMETERS
    )
    return _normalize_parameter_object(
        value,
        namespace=namespace,
        dedicated_parameters=dedicated,
        label="additional_audio_parameters_json",
    )


def normalize_routing_config_parameters(value: str) -> dict[str, Any]:
    return _normalize_parameter_object(
        value,
        namespace="routing",
        dedicated_parameters=ROUTING_DEDICATED_PARAMETERS,
        label="additional_routing_parameters_json",
    )


def _optional_text(value: str) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _optional_enum(name: str, value: str) -> str | None:
    normalized = str(value or "inherit").strip().lower()
    if normalized == "inherit":
        return None
    allowed = ROUTING_ENUM_VALUES[name]
    if normalized not in allowed:
        raise ValueError(
            f"{name} must be 'inherit' or one of: {', '.join(sorted(allowed))}"
        )
    return normalized


def _optional_bool(name: str, value: str) -> bool | None:
    normalized = str(value or "inherit").strip().lower()
    if normalized not in TRISTATE_VALUES:
        raise ValueError(f"{name} must be inherit, enabled, or disabled")
    if normalized == "inherit":
        return None
    return normalized == "enabled"


def build_routing_config(
    *,
    backend: str = "",
    accelerator: str = "",
    device: str = "",
    precision: str = "",
    quantization: str = "",
    profile: str = "",
    quality: str = "inherit",
    performance_preference: str = "inherit",
    fallback_policy: str = "inherit",
    parameter_policy: str = "inherit",
    allow_cpu_offload: str = "inherit",
    allow_sequential_offload: str = "inherit",
    allow_component_offload: str = "inherit",
    allow_disk_offload: str = "inherit",
    attention_backend: str = "",
    compile: str = "inherit",
    inference_timeout_seconds: int = 0,
    additional_routing_parameters_json: str = "{}",
) -> WerkRoutingConfig:
    """Build all current Werk routing overrides without inventing defaults."""

    request_options: dict[str, Any] = {}
    for name, value in {
        "backend": backend,
        "accelerator": accelerator,
        "device": device,
        "precision": precision,
        "quantization": quantization,
        "profile": profile,
        "attention_backend": attention_backend,
    }.items():
        if normalized := _optional_text(value):
            request_options[name] = normalized

    for name, value in {
        "quality": quality,
        "performance_preference": performance_preference,
        "fallback_policy": fallback_policy,
        "parameter_policy": parameter_policy,
    }.items():
        if normalized := _optional_enum(name, value):
            request_options[name] = normalized

    for name, value in {
        "allow_cpu_offload": allow_cpu_offload,
        "allow_sequential_offload": allow_sequential_offload,
        "allow_component_offload": allow_component_offload,
        "allow_disk_offload": allow_disk_offload,
        "compile": compile,
    }.items():
        normalized = _optional_bool(name, value)
        if normalized is not None:
            request_options[name] = normalized

    timeout = int(inference_timeout_seconds)
    if timeout < 0:
        raise ValueError("inference_timeout_seconds must be at least 0")
    if timeout:
        request_options["timeout_seconds"] = timeout

    parameters = normalize_routing_config_parameters(additional_routing_parameters_json)
    return WerkRoutingConfig(request_options=request_options, parameters=parameters)


def routing_config_payload(config: WerkRoutingConfig) -> dict[str, Any]:
    if not isinstance(config, WerkRoutingConfig):
        raise TypeError("config must be a WerkRoutingConfig")
    return {
        "request_options": dict(config.request_options),
        "parameters": dict(config.parameters),
    }


def _vision_stop_sequences(value: str) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(
            f"stop_sequences_json is invalid JSON: {error.msg}"
        ) from error
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) for item in parsed
    ):
        raise ValueError("stop_sequences_json must contain an array of strings")
    if any(not item for item in parsed):
        raise ValueError("stop sequences must not be empty")
    return parsed


def build_vision_config(
    *,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_completion_tokens: int = 1024,
    seed: int = 0,
    image_detail: str = "auto",
    stop_sequences_json: str = "[]",
) -> WerkVisionConfig:
    """Build only fields accepted by Werk's chat-completions contract."""

    temperature_value = float(temperature)
    top_p_value = float(top_p)
    token_value = int(max_completion_tokens)
    seed_value = int(seed)
    detail_value = str(image_detail or "auto").strip().lower()
    if not math.isfinite(temperature_value) or temperature_value < 0.0:
        raise ValueError("temperature must be finite and at least 0")
    if not math.isfinite(top_p_value) or not 0.0 <= top_p_value <= 1.0:
        raise ValueError("top_p must be finite and between 0 and 1")
    if token_value < 1:
        raise ValueError("max_completion_tokens must be at least 1")
    if not 0 <= seed_value <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError("seed must be between 0 and 9223372036854775807")
    if detail_value not in {"auto", "low", "high"}:
        raise ValueError("image_detail must be auto, low, or high")
    stop = _vision_stop_sequences(stop_sequences_json)
    request_fields: dict[str, Any] = {
        "temperature": temperature_value,
        "top_p": top_p_value,
        "max_completion_tokens": token_value,
        "seed": seed_value,
    }
    if stop:
        request_fields["stop"] = stop
    return WerkVisionConfig(
        request_fields=request_fields,
        image_detail=detail_value,
    )


def vision_config_payload(config: WerkVisionConfig) -> dict[str, Any]:
    if not isinstance(config, WerkVisionConfig):
        raise TypeError("config must be a WerkVisionConfig")
    return {
        "request_fields": dict(config.request_fields),
        "image_detail": config.image_detail,
    }


def build_image_config(
    *,
    width: int = 1024,
    height: int = 1024,
    count: int = 1,
    batch_size: int = 1,
    steps: int = 28,
    guidance: float = 7.0,
    seed: int = 0,
    output_format: str = "png",
    response_format: str = "b64_json",
    style: str = "none",
    vae_tiling: str = "inherit",
    vae_slicing: str = "inherit",
    additional_image_parameters_json: str = "{}",
    routing: WerkRoutingConfig | None = None,
) -> WerkImageConfig:
    """Build common image controls plus arbitrary schema-discovered parameters."""

    width_value = int(width)
    height_value = int(height)
    count_value = int(count)
    batch_value = int(batch_size)
    steps_value = int(steps)
    seed_value = int(seed)
    guidance_value = float(guidance)
    if width_value < 64 or height_value < 64:
        raise ValueError("width and height must be at least 64")
    if count_value < 1:
        raise ValueError("count must be at least 1")
    if batch_value < 1:
        raise ValueError("batch_size must be at least 1")
    if count_value > 1 and batch_value > 1:
        raise ValueError(
            "count and batch_size cannot both be greater than 1; "
            "the selected media adapter treats them as alternative image-count controls"
        )
    if steps_value < 1:
        raise ValueError("steps must be at least 1")
    if not math.isfinite(guidance_value) or guidance_value < 0.0:
        raise ValueError("guidance must be finite and at least 0")
    if not 0 <= seed_value <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError("seed must be between 0 and 9223372036854775807")
    if output_format not in {"png", "jpeg", "webp"}:
        raise ValueError("output_format must be png, jpeg, or webp")
    if response_format not in {"b64_json", "url"}:
        raise ValueError("response_format must be b64_json or url")
    if style not in {"none", "vivid", "natural"}:
        raise ValueError("style must be none, vivid, or natural")

    request_fields: dict[str, Any] = {
        "size": f"{width_value}x{height_value}",
        "response_format": response_format,
        "output_format": output_format,
    }
    if batch_value == 1:
        request_fields["n"] = count_value
    if style != "none":
        request_fields["style"] = style
    parameters: dict[str, Any] = {
        "image.steps": steps_value,
        "image.guidance": guidance_value,
        "image.seed": seed_value,
    }
    if batch_value > 1:
        parameters["image.batch_size"] = batch_value
    for name, value in {
        "image.vae_tiling": vae_tiling,
        "image.vae_slicing": vae_slicing,
    }.items():
        normalized = _optional_bool(name, value)
        if normalized is not None:
            parameters[name] = normalized
    parameters.update(
        normalize_image_config_parameters(additional_image_parameters_json)
    )

    routing_config = routing or WerkRoutingConfig()
    if not isinstance(routing_config, WerkRoutingConfig):
        raise TypeError("routing must be a WerkRoutingConfig")
    return WerkImageConfig(
        request_fields=request_fields,
        parameters=parameters,
        routing=routing_config,
    )


def image_config_payload(config: WerkImageConfig) -> dict[str, Any]:
    if not isinstance(config, WerkImageConfig):
        raise TypeError("config must be a WerkImageConfig")
    return {
        "request_fields": dict(config.request_fields),
        "parameters": dict(config.parameters),
        "routing": routing_config_payload(config.routing),
    }


def build_video_config(
    *,
    width: int = 832,
    height: int = 480,
    count: int = 1,
    batch_size: int = 1,
    frames: int = 81,
    fps: float = 24.0,
    steps: int = 30,
    guidance: float = 6.0,
    seed: int = 0,
    output_format: str = "mp4",
    temporal_vae_tiling: str = "inherit",
    additional_video_parameters_json: str = "{}",
    routing: WerkRoutingConfig | None = None,
) -> WerkVideoConfig:
    """Build common video controls plus arbitrary schema-discovered parameters."""

    width_value = int(width)
    height_value = int(height)
    count_value = int(count)
    batch_value = int(batch_size)
    frames_value = int(frames)
    fps_value = float(fps)
    steps_value = int(steps)
    guidance_value = float(guidance)
    seed_value = int(seed)
    output_value = str(output_format or "").strip().lower()
    if width_value < 64 or height_value < 64:
        raise ValueError("width and height must be at least 64")
    if count_value < 1:
        raise ValueError("count must be at least 1")
    if batch_value < 1:
        raise ValueError("batch_size must be at least 1")
    if count_value > 1 and batch_value > 1:
        raise ValueError(
            "count and batch_size cannot both be greater than 1; "
            "the selected media adapter treats them as alternative video-count controls"
        )
    if frames_value < 1:
        raise ValueError("frames must be at least 1")
    if not math.isfinite(fps_value) or fps_value < 0.1:
        raise ValueError("fps must be finite and at least 0.1")
    if steps_value < 1:
        raise ValueError("steps must be at least 1")
    if not math.isfinite(guidance_value) or guidance_value < 0.0:
        raise ValueError("guidance must be finite and at least 0")
    if not 0 <= seed_value <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError("seed must be between 0 and 9223372036854775807")
    if output_value not in {"mp4", "gif"}:
        raise ValueError("output_format must be mp4 or gif")

    request_fields: dict[str, Any] = {
        "size": f"{width_value}x{height_value}",
        # Werk's video API names its container field response_format.
        "response_format": output_value,
    }
    if batch_value == 1:
        request_fields["n"] = count_value
    parameters: dict[str, Any] = {
        "video.frames": frames_value,
        "video.fps": fps_value,
        "video.steps": steps_value,
        "video.guidance": guidance_value,
        "video.seed": seed_value,
    }
    if batch_value > 1:
        parameters["video.batch_size"] = batch_value
    temporal_tiling = _optional_bool(
        "video.temporal_vae_tiling", temporal_vae_tiling
    )
    if temporal_tiling is not None:
        parameters["video.temporal_vae_tiling"] = temporal_tiling
    parameters.update(
        normalize_video_config_parameters(additional_video_parameters_json)
    )

    routing_config = routing or WerkRoutingConfig()
    if not isinstance(routing_config, WerkRoutingConfig):
        raise TypeError("routing must be a WerkRoutingConfig")
    return WerkVideoConfig(
        request_fields=request_fields,
        parameters=parameters,
        routing=routing_config,
    )


def video_config_payload(config: WerkVideoConfig) -> dict[str, Any]:
    if not isinstance(config, WerkVideoConfig):
        raise TypeError("config must be a WerkVideoConfig")
    return {
        "request_fields": dict(config.request_fields),
        "parameters": dict(config.parameters),
        "routing": routing_config_payload(config.routing),
    }


def build_audio_config(
    *,
    task: str = AUDIO_GENERATION_TASK,
    duration: float = 30.0,
    variations: int = 1,
    seed: int = 0,
    sample_rate: int = 0,
    channels: int = 0,
    output_format: str = "wav",
    instrumental: str = "inherit",
    voice: str = "",
    speed: float = 1.0,
    language: str = "",
    speaking_style: str = "",
    additional_audio_parameters_json: str = "{}",
    routing: WerkRoutingConfig | None = None,
) -> WerkAudioConfig:
    """Build portable audio-generation or speech-synthesis controls."""

    selected_task = _audio_generation_task(task)
    duration_value = float(duration)
    variations_value = int(variations)
    seed_value = int(seed)
    sample_rate_value = int(sample_rate)
    channels_value = int(channels)
    output_value = str(output_format or "").strip().lower()
    speed_value = float(speed)
    voice_value = str(voice or "").strip()
    language_value = str(language or "").strip()
    speaking_style_value = str(speaking_style or "").strip()
    if not math.isfinite(duration_value) or duration_value < 0.1:
        raise ValueError("duration must be finite and at least 0.1 seconds")
    if variations_value < 1:
        raise ValueError("variations must be at least 1")
    if not 0 <= seed_value <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError("seed must be between 0 and 9223372036854775807")
    if sample_rate_value != 0 and sample_rate_value < 8_000:
        raise ValueError("sample_rate must be 0 (inherit) or at least 8000")
    if channels_value != 0 and channels_value < 1:
        raise ValueError("channels must be 0 (inherit) or at least 1")
    if output_value not in {"wav", "flac", "ogg"}:
        raise ValueError("output_format must be wav, flac, or ogg")
    if not math.isfinite(speed_value) or speed_value < 0.1:
        raise ValueError("speed must be finite and at least 0.1")

    request_fields: dict[str, Any] = {"response_format": output_value}
    parameters: dict[str, Any]
    if selected_task == TEXT_TO_SPEECH_TASK:
        if variations_value != 1:
            raise ValueError("text-to-speech supports exactly one variation")
        if duration_value != 30.0:
            raise ValueError("duration applies only to audio or music generation")
        if str(instrumental or "inherit").strip().lower() != "inherit":
            raise ValueError("instrumental applies only to audio or music generation")
        if voice_value:
            request_fields["voice"] = voice_value
        if speed_value != 1.0:
            request_fields["speed"] = speed_value
        # Zero is the portable inherit/default value.  Sending it explicitly
        # would make strict backends reject otherwise valid TTS requests when
        # they do not expose deterministic synthesis.
        parameters = {}
        if seed_value:
            parameters["tts.seed"] = seed_value
        if language_value:
            parameters["tts.language"] = language_value
        if speaking_style_value:
            parameters["tts.speaking_style"] = speaking_style_value
        namespace = "tts"
    else:
        if voice_value:
            raise ValueError("voice applies only to text-to-speech")
        if speed_value != 1.0:
            raise ValueError("speed applies only to text-to-speech")
        if language_value:
            raise ValueError("language applies only to text-to-speech")
        if speaking_style_value:
            raise ValueError("speaking_style applies only to text-to-speech")
        request_fields["n"] = variations_value
        parameters = {
            "audio.duration": duration_value,
            "audio.seed": seed_value,
        }
        instrumental_value = _optional_bool("audio.instrumental", instrumental)
        if instrumental_value is not None:
            parameters["audio.instrumental"] = instrumental_value
        namespace = "audio"
    if sample_rate_value:
        parameters[f"{namespace}.sample_rate"] = sample_rate_value
    if channels_value:
        parameters[f"{namespace}.channels"] = channels_value
    additional_parameters = normalize_audio_config_parameters(
        additional_audio_parameters_json,
        selected_task,
    )
    duplicates = set(parameters) & set(additional_parameters)
    if duplicates:
        raise ValueError(
            "additional audio parameter duplicates a populated node input: "
            + ", ".join(sorted(duplicates))
        )
    parameters.update(additional_parameters)

    routing_config = routing or WerkRoutingConfig()
    if not isinstance(routing_config, WerkRoutingConfig):
        raise TypeError("routing must be a WerkRoutingConfig")
    return WerkAudioConfig(
        task=selected_task,
        request_fields=request_fields,
        parameters=parameters,
        routing=routing_config,
    )


def audio_config_payload(config: WerkAudioConfig) -> dict[str, Any]:
    if not isinstance(config, WerkAudioConfig):
        raise TypeError("config must be a WerkAudioConfig")
    return {
        "task": config.task,
        "request_fields": dict(config.request_fields),
        "parameters": dict(config.parameters),
        "routing": routing_config_payload(config.routing),
    }


def build_configured_image_request(
    *,
    model: str,
    prompt: str,
    negative_prompt: str = "",
    config: WerkImageConfig | None = None,
) -> dict[str, Any]:
    """Merge a typed image config into one unambiguous Werk API request."""

    model_value = str(model or "").strip()
    if not model_value:
        raise ValueError("model must not be empty; connect WERK Image Models")
    if not str(prompt or "").strip():
        raise ValueError("prompt must not be empty")
    image_config = config or build_image_config()
    if not isinstance(image_config, WerkImageConfig):
        raise TypeError("config must be a WerkImageConfig")

    unknown_request_fields = (
        set(image_config.request_fields) - ALLOWED_IMAGE_REQUEST_FIELDS
    )
    if unknown_request_fields:
        raise ValueError(
            "image config contains unsupported request field(s): "
            + ", ".join(sorted(unknown_request_fields))
        )
    unknown_routing_fields = set(image_config.routing.request_options) - set(
        ROUTING_OPTION_PATHS
    )
    if unknown_routing_fields:
        raise ValueError(
            "routing config contains unsupported request option(s): "
            + ", ".join(sorted(unknown_routing_fields))
        )

    parameters = dict(image_config.routing.parameters)
    duplicates = set(parameters) & set(image_config.parameters)
    if duplicates:
        raise ValueError("config parameter collision: " + ", ".join(sorted(duplicates)))
    parameters.update(image_config.parameters)
    request: dict[str, Any] = {
        "model": model_value,
        "prompt": str(prompt),
        **dict(image_config.request_fields),
        **dict(image_config.routing.request_options),
        "parameters": parameters,
    }
    if str(negative_prompt or "").strip():
        request["negative_prompt"] = str(negative_prompt)
    return request


def build_vision_request(
    *,
    model: str,
    prompt: str,
    images: Any,
    config: WerkVisionConfig,
    system_prompt: str = "",
) -> dict[str, Any]:
    """Build one ordered, non-streaming multimodal chat-completion request."""

    model_value = str(model or "").strip()
    prompt_value = str(prompt or "").strip()
    if not model_value:
        raise ValueError("model must not be empty; connect WERK Vision Models")
    if not prompt_value:
        raise ValueError("prompt must not be empty")
    if not isinstance(config, WerkVisionConfig):
        raise TypeError("config must be a WerkVisionConfig")
    vision_config = config
    unknown_fields = set(vision_config.request_fields) - ALLOWED_VISION_REQUEST_FIELDS
    if unknown_fields:
        raise ValueError(
            "vision config contains unsupported chat field(s): "
            + ", ".join(sorted(unknown_fields))
        )

    data_urls = comfy_images_to_data_urls(
        images,
        max_pixels=environment_max_image_pixels(),
        max_bytes=environment_max_vision_input_bytes(),
    )
    content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {
                "url": data_url,
                "detail": vision_config.image_detail,
            },
        }
        for data_url in data_urls
    ]
    content.append({"type": "text", "text": prompt_value})
    messages: list[dict[str, Any]] = []
    if str(system_prompt or "").strip():
        messages.append(
            {"role": "system", "content": str(system_prompt).strip()}
        )
    messages.append({"role": "user", "content": content})
    return {
        "model": model_value,
        "messages": messages,
        "stream": False,
        **dict(vision_config.request_fields),
    }


def build_configured_video_request(
    *,
    model: str,
    prompt: str,
    negative_prompt: str = "",
    initial_image: Any | None = None,
    config: WerkVideoConfig | None = None,
) -> dict[str, Any]:
    """Merge a typed video config and optional first frame into a Werk request."""

    model_value = str(model or "").strip()
    if not model_value:
        raise ValueError("model must not be empty; connect WERK Video Models")
    if not str(prompt or "").strip():
        raise ValueError("prompt must not be empty")
    video_config = config or build_video_config()
    if not isinstance(video_config, WerkVideoConfig):
        raise TypeError("config must be a WerkVideoConfig")

    unknown_request_fields = (
        set(video_config.request_fields) - ALLOWED_VIDEO_REQUEST_FIELDS
    )
    if unknown_request_fields:
        raise ValueError(
            "video config contains unsupported request field(s): "
            + ", ".join(sorted(unknown_request_fields))
        )
    unknown_routing_fields = set(video_config.routing.request_options) - set(
        ROUTING_OPTION_PATHS
    )
    if unknown_routing_fields:
        raise ValueError(
            "routing config contains unsupported request option(s): "
            + ", ".join(sorted(unknown_routing_fields))
        )

    parameters = dict(video_config.routing.parameters)
    duplicates = set(parameters) & set(video_config.parameters)
    if duplicates:
        raise ValueError("config parameter collision: " + ", ".join(sorted(duplicates)))
    parameters.update(video_config.parameters)
    request: dict[str, Any] = {
        "model": model_value,
        "prompt": str(prompt),
        **dict(video_config.request_fields),
        **dict(video_config.routing.request_options),
        "parameters": parameters,
    }
    if str(negative_prompt or "").strip():
        request["negative_prompt"] = str(negative_prompt)
    if initial_image is not None:
        request["initial_image"] = image_tensor_to_api_input(initial_image)
    return request


def build_configured_audio_request(
    *,
    task: str,
    model: str,
    prompt: str,
    negative_prompt: str = "",
    config: WerkAudioConfig | None = None,
) -> dict[str, Any]:
    """Merge one task-specific audio config into its public Werk API shape."""

    selected_task = _audio_generation_task(task)
    model_value = str(model or "").strip()
    if not model_value:
        raise ValueError("model must not be empty; connect WERK Audio Models")
    prompt_value = str(prompt or "")
    if not prompt_value.strip():
        raise ValueError("prompt must not be empty")
    audio_config = config or build_audio_config(task=selected_task)
    if not isinstance(audio_config, WerkAudioConfig):
        raise TypeError("config must be a WerkAudioConfig")
    if audio_config.task != selected_task:
        raise ValueError(
            f"audio config task '{audio_config.task}' does not match generator task "
            f"'{selected_task}'"
        )

    allowed_request_fields = (
        ALLOWED_TTS_REQUEST_FIELDS
        if selected_task == TEXT_TO_SPEECH_TASK
        else ALLOWED_AUDIO_GENERATION_REQUEST_FIELDS
    )
    unknown_request_fields = set(audio_config.request_fields) - allowed_request_fields
    if unknown_request_fields:
        raise ValueError(
            "audio config contains unsupported request field(s): "
            + ", ".join(sorted(unknown_request_fields))
        )
    unknown_routing_fields = set(audio_config.routing.request_options) - set(
        ROUTING_OPTION_PATHS
    )
    if unknown_routing_fields:
        raise ValueError(
            "routing config contains unsupported request option(s): "
            + ", ".join(sorted(unknown_routing_fields))
        )

    parameters = dict(audio_config.routing.parameters)
    duplicates = set(parameters) & set(audio_config.parameters)
    if duplicates:
        raise ValueError("config parameter collision: " + ", ".join(sorted(duplicates)))
    parameters.update(audio_config.parameters)
    request: dict[str, Any] = {
        "model": model_value,
        **dict(audio_config.request_fields),
        **dict(audio_config.routing.request_options),
        "parameters": parameters,
    }
    negative_value = str(negative_prompt or "")
    if selected_task == TEXT_TO_SPEECH_TASK:
        if negative_value.strip():
            raise ValueError("negative_prompt is not supported for text-to-speech")
        request["input"] = prompt_value
        request["async"] = True
    else:
        request["task"] = selected_task
        request["prompt"] = prompt_value
        if negative_value.strip():
            request["negative_prompt"] = negative_value
    return request


def normalize_audio_job_parameters(value: str, task: str) -> dict[str, Any]:
    """Normalize free-form parameters for an audio-input generic job."""

    selected_task = _audio_task(task)
    if selected_task in AUDIO_GENERATION_TASKS:
        raise ValueError("audio-input jobs do not accept generation-only tasks")
    namespace = "stt" if selected_task in AUDIO_TRANSCRIPTION_TASKS else "audio"
    return _normalize_parameter_object(
        value,
        namespace=namespace,
        dedicated_parameters=set(),
        label="additional_audio_parameters_json",
    )


def build_audio_input_job_request(
    *,
    task: str,
    model: str,
    audio: Any,
    reference_audio: Any | None = None,
    prompt: str = "",
    negative_prompt: str = "",
    additional_audio_parameters_json: str = "{}",
    routing: WerkRoutingConfig | None = None,
) -> dict[str, Any]:
    """Build a generic ``/v1/jobs`` request from native ComfyUI audio."""

    selected_task = _audio_task(task)
    if selected_task in AUDIO_GENERATION_TASKS:
        raise ValueError("use WERK Audio Generate for audio-generation tasks")
    model_value = str(model or "").strip()
    if not model_value:
        raise ValueError("model must not be empty; connect WERK Audio Models")
    prompt_value = str(prompt or "")
    if (
        selected_task in {"audio-understanding", "audio-editing"}
        and not prompt_value.strip()
    ):
        raise ValueError(f"prompt must not be empty for {selected_task}")
    if reference_audio is not None and selected_task != "voice-conversion":
        raise ValueError("reference_audio is supported only for voice-conversion")
    routing_config = routing or WerkRoutingConfig()
    if not isinstance(routing_config, WerkRoutingConfig):
        raise TypeError("routing must be a WerkRoutingConfig")
    unknown_routing_fields = set(routing_config.request_options) - set(
        ROUTING_OPTION_PATHS
    )
    if unknown_routing_fields:
        raise ValueError(
            "routing config contains unsupported request option(s): "
            + ", ".join(sorted(unknown_routing_fields))
        )
    parameters = dict(routing_config.parameters)
    task_parameters = normalize_audio_job_parameters(
        additional_audio_parameters_json,
        selected_task,
    )
    duplicates = set(parameters) & set(task_parameters)
    if duplicates:
        raise ValueError("config parameter collision: " + ", ".join(sorted(duplicates)))
    parameters.update(task_parameters)
    max_input_bytes = environment_max_audio_input_bytes()
    primary_input = comfy_audio_to_api_input(
        audio,
        max_bytes=max_input_bytes,
    )
    inputs = [primary_input]
    if reference_audio is not None:
        encoded = primary_input["source"]["data"]
        padding = len(encoded) - len(encoded.rstrip("="))
        primary_bytes = (len(encoded) * 3 // 4) - padding
        remaining_bytes = max_input_bytes - primary_bytes
        if remaining_bytes <= 0:
            raise ValueError(
                f"combined encoded audio inputs exceed {max_input_bytes} bytes"
            )
        try:
            reference_input = comfy_audio_to_api_input(
                reference_audio,
                max_bytes=remaining_bytes,
                role="reference_audio",
            )
        except ValueError as error:
            if "encoded audio input exceeds" in str(error):
                raise ValueError(
                    f"combined encoded audio inputs exceed {max_input_bytes} bytes"
                ) from error
            raise
        inputs.append(reference_input)
    request: dict[str, Any] = {
        "model": model_value,
        "task": selected_task,
        "inputs": inputs,
        **dict(routing_config.request_options),
        "parameters": parameters,
    }
    if prompt_value.strip():
        request["prompt"] = prompt_value
    if str(negative_prompt or "").strip():
        request["negative_prompt"] = str(negative_prompt)
    return request


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        if str(value.get("kind", "")).lower() == "base64":
            return {
                key: _sanitize_metadata(child)
                for key, child in value.items()
                if str(key).lower() != "data"
            } | {"embedded": True}
        safe = {}
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {
                "path",
                "output_path",
                "local_path",
                "filesystem_path",
                "base64",
                "b64_json",
            }:
                continue
            safe[key] = _sanitize_metadata(child)
        return safe
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    return value


def safe_result_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    data_metadata = []
    data = response.get("data", [])
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                data_metadata.append(
                    {
                        key: item[key]
                        for key in (
                            "id",
                            "mime_type",
                            "size_bytes",
                            "width",
                            "height",
                            "duration",
                        )
                        if key in item
                    }
                )
    werk = response.get("werk", {})
    safe_werk = {}
    if isinstance(werk, dict):
        for key in (
            "id",
            "task",
            "model",
            "runtime",
            "effective_request",
            "estimate",
            "plan",
            "backend_metadata",
            "timings",
            "warnings",
            "created_unix",
        ):
            if key in werk:
                safe_werk[key] = _sanitize_metadata(werk[key])
    return {
        "created": response.get("created"),
        "data": data_metadata,
        "werk": safe_werk,
    }


def _safe_inference_result(result: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in (
        "id",
        "task",
        "model",
        "runtime",
        "effective_request",
        "estimate",
        "plan",
        "backend_metadata",
        "timings",
        "warnings",
        "created_unix",
    ):
        if key in result:
            safe[key] = _sanitize_metadata(result[key])
    outputs = result.get("outputs", [])
    if isinstance(outputs, list):
        safe["outputs"] = [
            {
                key: _sanitize_metadata(item[key])
                for key in (
                    "id",
                    "task",
                    "model",
                    "runtime",
                    "mime_type",
                    "size_bytes",
                    "width",
                    "height",
                    "duration",
                    "seed",
                    "effective_parameters",
                    "created_unix",
                    "backend_metadata",
                )
                if key in item
            }
            for item in outputs
            if isinstance(item, dict)
        ]
    return safe


def safe_media_job_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    safe = {
        key: record[key]
        for key in ("id", "status", "created_unix", "updated_unix")
        if key in record
    }
    result = record.get("result")
    if isinstance(result, dict):
        safe["result"] = _safe_inference_result(result)
    if isinstance(record.get("error"), str):
        safe["error"] = record["error"]
    return safe


def safe_video_job_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible video-specific name for shared job sanitization."""

    return safe_media_job_metadata(record)


def _check_comfy_interrupted() -> None:
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:  # pragma: no cover - only available inside ComfyUI
        return
    throw_exception_if_processing_interrupted()


def _cancel_job_best_effort(client: WerkClient, job_id: str) -> None:
    try:
        client.delete_json(f"/v1/jobs/{quote(job_id, safe='')}")
    except Exception:
        pass


def wait_for_media_job(
    client: WerkClient,
    initial_record: Mapping[str, Any],
    *,
    media_kind: str,
    timeout_seconds: float,
    poll_interval_seconds: float = 1.0,
    sleep_fn=None,
    monotonic_fn=None,
    interrupt_check=None,
) -> dict[str, Any]:
    """Poll one Werk media job and cancel it best-effort if waiting aborts."""

    kind = str(media_kind or "media").strip().lower()
    record = _object(initial_record, f"{kind} job")
    job_id = record.get("id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError(f"Werk {kind} job response contains no valid job id")
    timeout = float(timeout_seconds)
    interval = float(poll_interval_seconds)
    if timeout <= 0:
        raise ValueError(f"{kind} job timeout must be greater than zero")
    if interval <= 0:
        raise ValueError(f"{kind} job poll interval must be greater than zero")
    sleep = sleep_fn or time.sleep
    monotonic = monotonic_fn or time.monotonic
    check_interrupted = interrupt_check or _check_comfy_interrupted
    deadline = monotonic() + timeout
    active = True
    try:
        while True:
            status = str(record.get("status", "")).strip().lower()
            if status == "completed":
                active = False
                if not isinstance(record.get("result"), dict):
                    raise ValueError(
                        f"Werk completed {kind} job contains no inference result"
                    )
                return record
            if status == "failed":
                active = False
                detail = record.get("error")
                message = detail.strip() if isinstance(detail, str) else "unknown failure"
                raise ValueError(f"Werk {kind} job failed: {message}")
            if status == "cancelled":
                active = False
                raise ValueError(f"Werk {kind} job was cancelled")
            if status not in {"queued", "loading", "running", "encoding"}:
                raise ValueError(f"Werk {kind} job returned unknown status '{status}'")

            check_interrupted()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Werk {kind} job did not complete within {timeout:g} seconds"
                )
            sleep(min(interval, remaining))
            record = _object(
                client.get_json(f"/v1/jobs/{quote(job_id, safe='')}"),
                f"{kind} job",
            )
            if record.get("id") != job_id:
                raise ValueError(f"Werk {kind} job id changed while polling")
    except BaseException:
        if active:
            _cancel_job_best_effort(client, job_id)
        raise


def wait_for_video_job(
    client: WerkClient,
    initial_record: Mapping[str, Any],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 1.0,
    sleep_fn=None,
    monotonic_fn=None,
    interrupt_check=None,
) -> dict[str, Any]:
    return wait_for_media_job(
        client,
        initial_record,
        media_kind="video",
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
        interrupt_check=interrupt_check,
    )


def wait_for_audio_job(
    client: WerkClient,
    initial_record: Mapping[str, Any],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 1.0,
    sleep_fn=None,
    monotonic_fn=None,
    interrupt_check=None,
) -> dict[str, Any]:
    return wait_for_media_job(
        client,
        initial_record,
        media_kind="audio",
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
        interrupt_check=interrupt_check,
    )


def _video_mime_type(value: str) -> bool:
    return value.startswith("video/") or value == "image/gif"


def _audio_mime_type(value: str) -> bool:
    return value.startswith("audio/") or value == "application/ogg"


def execute_media_job_request(
    connection: WerkConnection,
    *,
    endpoint: str,
    media_kind: str,
    request: Mapping[str, Any],
    seed: int,
    max_bytes: int,
    mime_validator,
    converter,
):
    client = WerkClient(connection)
    initial = _object(
        client.post_json(endpoint, dict(request)),
        f"{media_kind} generation",
    )
    record = wait_for_media_job(
        client,
        initial,
        media_kind=media_kind,
        timeout_seconds=connection.timeout_seconds,
    )
    result = _object(record.get("result"), f"{media_kind} result")
    outputs = result.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"Werk completed {media_kind} job contains no outputs")

    values = []
    output_ids = []
    for item in outputs:
        if not isinstance(item, dict):
            raise ValueError(
                f"Werk {media_kind} result contains an invalid output entry"
            )
        output_id = item.get("id")
        if not isinstance(output_id, str) or not output_id.strip():
            raise ValueError(f"Werk {media_kind} output contains no valid output id")
        mime_type = str(item.get("mime_type", "")).lower().split(";", 1)[0]
        if not mime_validator(mime_type):
            raise ValueError(
                f"Werk {media_kind} output returned non-{media_kind} MIME type "
                f"'{mime_type or 'unknown'}'"
            )
        raw, content_type = client.download_bytes(
            f"/v1/outputs/{quote(output_id, safe='')}",
            max_bytes=max_bytes,
        )
        downloaded_type = (
            content_type.lower().split(";", 1)[0] if content_type else ""
        )
        if downloaded_type and not mime_validator(downloaded_type):
            raise ValueError(
                f"Werk output returned non-{media_kind} content type '{content_type}'"
            )
        values.append(converter(raw))
        output_ids.append(output_id)

    result_id = result.get("id")
    if not isinstance(result_id, str):
        result_id = ""
    return (
        values,
        _json_text(safe_media_job_metadata(record)),
        int(seed),
        str(record["id"]),
        str(result_id),
        "\n".join(output_ids),
    )


def execute_video_request(
    connection: WerkConnection,
    request: Mapping[str, Any],
    seed: int,
):
    return execute_media_job_request(
        connection,
        endpoint="/v1/videos/generations",
        media_kind="video",
        request=request,
        seed=seed,
        max_bytes=environment_max_video_bytes(),
        mime_validator=_video_mime_type,
        converter=video_bytes_to_comfy,
    )


def execute_audio_request(
    connection: WerkConnection,
    task: str,
    request: Mapping[str, Any],
    seed: int,
):
    selected_task = _audio_generation_task(task)
    endpoint = (
        "/v1/audio/speech"
        if selected_task == TEXT_TO_SPEECH_TASK
        else "/v1/audio/generations"
    )
    return execute_media_job_request(
        connection,
        endpoint=endpoint,
        media_kind="audio",
        request=request,
        seed=seed,
        max_bytes=environment_max_audio_bytes(),
        mime_validator=_audio_mime_type,
        converter=audio_bytes_to_comfy,
    )


def execute_audio_process_request(
    connection: WerkConnection,
    task: str,
    request: Mapping[str, Any],
):
    selected_task = _audio_task(task)
    if selected_task not in AUDIO_TRANSFORM_TASKS:
        raise ValueError("audio process task must produce audio")
    seed = 0
    parameters = request.get("parameters")
    if isinstance(parameters, Mapping):
        value = parameters.get("audio.seed", 0)
        if isinstance(value, int) and not isinstance(value, bool):
            seed = value
    return execute_media_job_request(
        connection,
        endpoint="/v1/jobs",
        media_kind="audio",
        request=request,
        seed=seed,
        max_bytes=environment_max_audio_bytes(),
        mime_validator=_audio_mime_type,
        converter=audio_bytes_to_comfy,
    )


def _text_output_mime_type(value: str) -> bool:
    return value.startswith("text/") or value in {
        "application/json",
        "application/x-ndjson",
    }


def _decode_text_output(data: bytes, mime_type: str) -> str:
    try:
        value = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Werk audio analysis output is not valid UTF-8") from error
    if mime_type in {"application/json", "application/x-ndjson"}:
        if mime_type == "application/json":
            try:
                return _json_text(json.loads(value))
            except json.JSONDecodeError as error:
                raise ValueError("Werk audio analysis returned invalid JSON") from error
    return value


def execute_audio_analysis_request(
    connection: WerkConnection,
    task: str,
    request: Mapping[str, Any],
):
    selected_task = _audio_task(task)
    if selected_task not in AUDIO_TEXT_OUTPUT_TASKS:
        raise ValueError("audio analysis task must produce text or JSON")
    client = WerkClient(connection)
    initial = _object(client.post_json("/v1/jobs", dict(request)), "audio analysis")
    record = wait_for_media_job(
        client,
        initial,
        media_kind="audio analysis",
        timeout_seconds=connection.timeout_seconds,
    )
    result = _object(record.get("result"), "audio analysis result")
    outputs = result.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Werk completed audio analysis job contains no outputs")
    values: list[str] = []
    output_ids: list[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            raise ValueError("Werk audio analysis result has an invalid output entry")
        output_id = item.get("id")
        if not isinstance(output_id, str) or not output_id.strip():
            raise ValueError("Werk audio analysis output has no valid output id")
        mime_type = str(item.get("mime_type", "")).lower().split(";", 1)[0]
        if not _text_output_mime_type(mime_type):
            raise ValueError(
                "Werk audio analysis returned non-text MIME type "
                f"'{mime_type or 'unknown'}'"
            )
        raw, content_type = client.download_bytes(
            f"/v1/outputs/{quote(output_id, safe='')}",
            max_bytes=MAX_AUDIO_TEXT_BYTES,
        )
        downloaded_type = (
            content_type.lower().split(";", 1)[0] if content_type else mime_type
        )
        if not _text_output_mime_type(downloaded_type):
            raise ValueError(
                "Werk audio analysis download returned non-text content type "
                f"'{content_type}'"
            )
        values.append(_decode_text_output(raw, downloaded_type))
        output_ids.append(output_id)
    result_id = result.get("id")
    return (
        values,
        _json_text(safe_media_job_metadata(record)),
        str(record["id"]),
        str(result_id) if isinstance(result_id, str) else "",
        "\n".join(output_ids),
    )


def execute_vision_request(
    connection: WerkConnection,
    request: Mapping[str, Any],
):
    response = _object(
        WerkClient(connection).post_json("/v1/chat/completions", dict(request)),
        "vision analysis",
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Werk vision response contains no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Werk vision response contains an invalid choice")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Werk vision response contains no assistant text")
    completion_id = response.get("id")
    finish_reason = choice.get("finish_reason")
    metadata = {
        key: value
        for key, value in response.items()
        if key != "choices"
    }
    metadata["choice"] = {
        "index": choice.get("index", 0),
        "finish_reason": finish_reason,
        "role": message.get("role", "assistant"),
    }
    return (
        message["content"],
        _json_text(metadata),
        str(completion_id) if isinstance(completion_id, str) else "",
        str(finish_reason) if isinstance(finish_reason, str) else "",
    )


def execute_image_request(
    connection: WerkConnection,
    request: Mapping[str, Any],
    seed: int,
):
    client = WerkClient(connection)
    response = _object(
        client.post_json("/v1/images/generations", dict(request)),
        "image generation",
    )
    data = response.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("Werk image generation response contains no images")
    tensors = []
    output_ids = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Werk image response contains an invalid data entry")
        output_id = item.get("id")
        if isinstance(output_id, str):
            output_ids.append(output_id)
        if isinstance(item.get("b64_json"), str):
            raw = decode_base64_image(item["b64_json"])
        elif isinstance(item.get("url"), str):
            raw, content_type = client.download_bytes(item["url"])
            if content_type and not content_type.lower().split(";", 1)[0].startswith(
                "image/"
            ):
                raise ValueError(
                    f"Werk output returned non-image content type '{content_type}'"
                )
        else:
            raise ValueError("Werk image entry contains neither b64_json nor url")
        tensors.append(
            image_bytes_to_tensor(raw, max_pixels=environment_max_image_pixels())
        )
    werk = response.get("werk", {})
    result_id = werk.get("id", "") if isinstance(werk, dict) else ""
    metadata = safe_result_metadata(response)
    return (
        batch_image_tensors(tensors),
        _json_text(metadata),
        int(seed),
        str(result_id),
        "\n".join(output_ids),
    )


class WerkConnectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": environment_server_url()}),
                "api_key": ("STRING", {"default": environment_api_key()}),
                "timeout_seconds": (
                    "INT",
                    {"default": DEFAULT_TIMEOUT_SECONDS, "min": 1},
                ),
                "verify_tls": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("WERK_CONNECTION", "STRING")
    RETURN_NAMES = ("connection", "status")
    FUNCTION = "connect"
    CATEGORY = "WERK/Configuration"

    def connect(
        self, server_url: str, api_key: str, timeout_seconds: int, verify_tls: bool
    ):
        connection = WerkConnection(server_url, api_key, timeout_seconds, verify_tls)
        return connection, connection.safe_status


class WerkServerInfoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "refresh_token": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("models", "capabilities", "metadata_json")
    FUNCTION = "discover"
    CATEGORY = "WERK/Discovery"

    def discover(self, connection: WerkConnection, refresh_token: int):
        del refresh_token
        client = WerkClient(connection)
        models = client.get_json("/v1/models")
        capabilities: Any = {}
        optional_error = None
        try:
            capabilities = client.get_json("/v1/capabilities")
        except WerkApiError as error:
            optional_error = str(error)
        classified = classify_image_models(models, capabilities)
        vision_classified = classify_vision_models(models, capabilities)
        video_classified = classify_video_models(models, capabilities)
        audio_classified = classify_audio_models(models, capabilities)
        metadata = {
            "models": models,
            "capabilities": capabilities,
            "classification": classified,
            "vision_classification": vision_classified,
            "video_classification": video_classified,
            "audio_classification": audio_classified,
        }
        if optional_error:
            metadata["capabilities_warning"] = optional_error
        return (
            "\n".join(classified["installed"]),
            _json_text(capabilities),
            _json_text(metadata),
        )


class WerkImageModelsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "refresh_token": ("INT", {"default": 0}),
                "preferred_model": ("STRING", {"default": ""}),
                "require_available": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "available_models", "metadata_json")
    FUNCTION = "select"
    CATEGORY = "WERK/Discovery"

    def select(
        self,
        connection: WerkConnection,
        refresh_token: int,
        preferred_model: str,
        require_available: bool,
    ):
        del refresh_token
        client = WerkClient(connection)
        models = client.get_json("/v1/models")
        try:
            capabilities = client.get_json("/v1/capabilities")
        except WerkApiError:
            capabilities = {}
        classified = classify_image_models(models, capabilities)
        candidates = (
            classified["available"] if require_available else classified["declared"]
        )
        preferred = preferred_model.strip()
        if preferred:
            if preferred in candidates:
                selected = preferred
            elif preferred in classified["declared"] and require_available:
                raise ValueError(
                    _unavailable_task_message(
                        IMAGE_TASK,
                        [preferred],
                        classified["models"],
                    )
                )
            elif preferred in classified["installed"]:
                raise ValueError(
                    f"preferred Werk model {preferred!r} does not declare {IMAGE_TASK}"
                )
            else:
                raise ValueError(f"preferred Werk model {preferred!r} is not installed")
        elif len(candidates) == 1:
            selected = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                "multiple matching Werk image models; set preferred_model to one of: "
                + ", ".join(candidates)
            )
        elif classified["declared"] and require_available:
            raise ValueError(
                _unavailable_task_message(
                    IMAGE_TASK,
                    classified["declared"],
                    classified["models"],
                )
            )
        else:
            raise ValueError("no installed Werk model declares image-generation")
        return selected, "\n".join(candidates), _json_text(classified)


class WerkVisionModelsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "refresh_token": ("INT", {"default": 0}),
                "preferred_model": ("STRING", {"default": ""}),
                "require_available": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "available_models", "metadata_json")
    FUNCTION = "select"
    CATEGORY = "WERK/Discovery"

    def select(
        self,
        connection: WerkConnection,
        refresh_token: int,
        preferred_model: str,
        require_available: bool,
    ):
        del refresh_token
        client = WerkClient(connection)
        models = client.get_json("/v1/models")
        try:
            capabilities = client.get_json("/v1/capabilities")
        except WerkApiError:
            capabilities = {}
        classified = classify_vision_models(models, capabilities)
        candidates = (
            classified["available"] if require_available else classified["declared"]
        )
        preferred = preferred_model.strip()
        if preferred:
            if preferred in candidates:
                selected = preferred
            elif preferred in classified["declared"] and require_available:
                raise ValueError(
                    _unavailable_task_message(
                        VISION_TASK,
                        [preferred],
                        classified["models"],
                    )
                )
            elif preferred in classified["installed"]:
                raise ValueError(
                    f"preferred Werk model {preferred!r} does not declare {VISION_TASK}"
                )
            else:
                raise ValueError(f"preferred Werk model {preferred!r} is not installed")
        elif len(candidates) == 1:
            selected = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                "multiple matching Werk vision models; set preferred_model to one of: "
                + ", ".join(candidates)
            )
        elif classified["declared"] and require_available:
            raise ValueError(
                _unavailable_task_message(
                    VISION_TASK,
                    classified["declared"],
                    classified["models"],
                )
            )
        else:
            raise ValueError("no installed Werk model declares image-understanding")
        return selected, "\n".join(candidates), _json_text(classified)


class WerkVideoModelsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "task": (
                    [VIDEO_GENERATION_TASK, IMAGE_TO_VIDEO_TASK],
                    {"default": VIDEO_GENERATION_TASK},
                ),
                "refresh_token": ("INT", {"default": 0}),
                "preferred_model": ("STRING", {"default": ""}),
                "require_available": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "available_models", "metadata_json")
    FUNCTION = "select"
    CATEGORY = "WERK/Discovery"

    def select(
        self,
        connection: WerkConnection,
        task: str,
        refresh_token: int,
        preferred_model: str,
        require_available: bool,
    ):
        del refresh_token
        selected_task = _video_task(task)
        client = WerkClient(connection)
        models = client.get_json("/v1/models")
        try:
            capabilities = client.get_json("/v1/capabilities")
        except WerkApiError:
            capabilities = {}
        classified = classify_video_models(models, capabilities)
        task_models = classified["by_task"][selected_task]
        candidates = (
            task_models["available"]
            if require_available
            else task_models["declared"]
        )
        preferred = preferred_model.strip()
        if preferred:
            if preferred in candidates:
                selected = preferred
            elif preferred in task_models["declared"] and require_available:
                raise ValueError(
                    _unavailable_task_message(
                        selected_task,
                        [preferred],
                        classified["models"],
                    )
                )
            elif preferred in classified["installed"]:
                raise ValueError(
                    f"preferred Werk model {preferred!r} does not declare {selected_task}"
                )
            else:
                raise ValueError(f"preferred Werk model {preferred!r} is not installed")
        elif len(candidates) == 1:
            selected = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                f"multiple matching Werk {selected_task} models; "
                "set preferred_model to one of: " + ", ".join(candidates)
            )
        elif task_models["declared"] and require_available:
            raise ValueError(
                _unavailable_task_message(
                    selected_task,
                    task_models["declared"],
                    classified["models"],
                )
            )
        else:
            raise ValueError(
                f"no installed Werk model declares {selected_task}"
            )
        metadata = dict(classified)
        metadata["selected_task"] = selected_task
        return selected, "\n".join(candidates), _json_text(metadata)


class WerkAudioModelsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "task": (
                    list(AUDIO_TASKS),
                    {"default": AUDIO_GENERATION_TASK},
                ),
                "refresh_token": ("INT", {"default": 0}),
                "preferred_model": ("STRING", {"default": ""}),
                "require_available": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "available_models", "metadata_json")
    FUNCTION = "select"
    CATEGORY = "WERK/Discovery"

    def select(
        self,
        connection: WerkConnection,
        task: str,
        refresh_token: int,
        preferred_model: str,
        require_available: bool,
    ):
        del refresh_token
        selected_task = _audio_task(task)
        client = WerkClient(connection)
        models = client.get_json("/v1/models")
        try:
            capabilities = client.get_json("/v1/capabilities")
        except WerkApiError:
            capabilities = {}
        classified = classify_audio_models(models, capabilities)
        task_models = classified["by_task"][selected_task]
        candidates = (
            task_models["available"]
            if require_available
            else task_models["declared"]
        )
        preferred = preferred_model.strip()
        if preferred:
            if preferred in candidates:
                selected = preferred
            elif preferred in task_models["declared"] and require_available:
                raise ValueError(
                    _unavailable_task_message(
                        selected_task,
                        [preferred],
                        classified["models"],
                    )
                )
            elif preferred in classified["installed"]:
                raise ValueError(
                    f"preferred Werk model {preferred!r} does not declare {selected_task}"
                )
            else:
                raise ValueError(f"preferred Werk model {preferred!r} is not installed")
        elif len(candidates) == 1:
            selected = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                f"multiple matching Werk {selected_task} models; "
                "set preferred_model to one of: " + ", ".join(candidates)
            )
        elif task_models["declared"] and require_available:
            raise ValueError(
                _unavailable_task_message(
                    selected_task,
                    task_models["declared"],
                    classified["models"],
                )
            )
        else:
            raise ValueError(f"no installed Werk model declares {selected_task}")
        metadata = dict(classified)
        metadata["selected_task"] = selected_task
        return selected, "\n".join(candidates), _json_text(metadata)


class WerkImageParametersNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model": ("STRING", {"default": ""}),
                "backend": ("STRING", {"default": "auto"}),
                "refresh_token": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("parameters_json", "summary")
    FUNCTION = "parameters"
    CATEGORY = "WERK/Discovery"

    def parameters(
        self, connection: WerkConnection, model: str, backend: str, refresh_token: int
    ):
        del refresh_token
        model = model.strip()
        if not model:
            raise ValueError("model must not be empty")
        payload = WerkClient(connection).get_json(
            "/v1/parameters",
            {"task": IMAGE_TASK, "model": model, "backend": backend.strip() or "auto"},
        )
        root = _object(payload, "parameters")
        descriptors = root.get("parameters", root.get("data", []))
        count = len(descriptors) if isinstance(descriptors, (list, dict)) else 0
        summary = f"{model}: {count} parameter descriptor(s) returned by Werk"
        return _json_text(payload), summary


class WerkVideoParametersNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model": ("STRING", {"default": ""}),
                "task": (
                    [VIDEO_GENERATION_TASK, IMAGE_TO_VIDEO_TASK],
                    {"default": VIDEO_GENERATION_TASK},
                ),
                "backend": ("STRING", {"default": "auto"}),
                "refresh_token": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("parameters_json", "summary")
    FUNCTION = "parameters"
    CATEGORY = "WERK/Discovery"

    def parameters(
        self,
        connection: WerkConnection,
        model: str,
        task: str,
        backend: str,
        refresh_token: int,
    ):
        del refresh_token
        model = model.strip()
        if not model:
            raise ValueError("model must not be empty")
        selected_task = _video_task(task)
        payload = WerkClient(connection).get_json(
            "/v1/parameters",
            {
                "task": selected_task,
                "model": model,
                "backend": backend.strip() or "auto",
            },
        )
        root = _object(payload, "parameters")
        descriptors = root.get("parameters", root.get("data", []))
        count = len(descriptors) if isinstance(descriptors, (list, dict)) else 0
        summary = (
            f"{model} ({selected_task}): {count} parameter descriptor(s) "
            "returned by Werk"
        )
        return _json_text(payload), summary


class WerkAudioParametersNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model": ("STRING", {"default": ""}),
                "task": (
                    list(AUDIO_TASKS),
                    {"default": AUDIO_GENERATION_TASK},
                ),
                "backend": ("STRING", {"default": "auto"}),
                "refresh_token": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("parameters_json", "summary")
    FUNCTION = "parameters"
    CATEGORY = "WERK/Discovery"

    def parameters(
        self,
        connection: WerkConnection,
        model: str,
        task: str,
        backend: str,
        refresh_token: int,
    ):
        del refresh_token
        model = model.strip()
        if not model:
            raise ValueError("model must not be empty")
        selected_task = _audio_task(task)
        payload = WerkClient(connection).get_json(
            "/v1/parameters",
            {
                "task": selected_task,
                "model": model,
                "backend": backend.strip() or "auto",
            },
        )
        root = _object(payload, "parameters")
        descriptors = root.get("parameters", root.get("data", []))
        count = len(descriptors) if isinstance(descriptors, (list, dict)) else 0
        summary = (
            f"{model} ({selected_task}): {count} parameter descriptor(s) "
            "returned by Werk"
        )
        return _json_text(payload), summary


class WerkRoutingConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        tri_state = (["inherit", "enabled", "disabled"], {"default": "inherit"})
        return {
            "required": {
                "backend": ("STRING", {"default": ""}),
                "accelerator": ("STRING", {"default": ""}),
                "device": ("STRING", {"default": ""}),
                "precision": ("STRING", {"default": ""}),
                "quantization": ("STRING", {"default": ""}),
                "profile": ("STRING", {"default": ""}),
                "quality": (
                    ["inherit", "draft", "balanced", "high", "maximum"],
                    {"default": "inherit"},
                ),
                "performance_preference": (
                    [
                        "inherit",
                        "quality",
                        "balanced",
                        "speed",
                        "latency",
                        "throughput",
                        "memory",
                    ],
                    {"default": "inherit"},
                ),
                "fallback_policy": (
                    ["inherit", "none", "backend", "degrade"],
                    {"default": "inherit"},
                ),
                "parameter_policy": (
                    ["inherit", "strict", "warn", "permissive"],
                    {"default": "inherit"},
                ),
                "allow_cpu_offload": tri_state,
                "allow_sequential_offload": tri_state,
                "allow_component_offload": tri_state,
                "allow_disk_offload": tri_state,
                "attention_backend": ("STRING", {"default": ""}),
                "compile": tri_state,
                "inference_timeout_seconds": (
                    "INT",
                    {"default": 0, "min": 0},
                ),
                "additional_routing_parameters_json": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
            }
        }

    RETURN_TYPES = ("WERK_ROUTING_CONFIG", "STRING")
    RETURN_NAMES = ("routing", "config_json")
    FUNCTION = "configure"
    CATEGORY = "WERK/Configuration"

    def configure(self, **inputs):
        config = build_routing_config(**inputs)
        return config, _json_text(routing_config_payload(config))


class WerkVisionConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_completion_tokens": (
                    "INT",
                    {"default": 1024, "min": 1},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF},
                ),
                "image_detail": (
                    ["auto", "low", "high"],
                    {"default": "auto"},
                ),
                "stop_sequences_json": (
                    "STRING",
                    {"default": "[]", "multiline": True},
                ),
            }
        }

    RETURN_TYPES = ("WERK_VISION_CONFIG", "STRING")
    RETURN_NAMES = ("config", "config_json")
    FUNCTION = "configure"
    CATEGORY = "WERK/Configuration"

    def configure(self, **inputs):
        config = build_vision_config(**inputs)
        return config, _json_text(vision_config_payload(config))


class WerkImageConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        tri_state = (["inherit", "enabled", "disabled"], {"default": "inherit"})
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "step": 8}),
                "height": (
                    "INT",
                    {"default": 1024, "min": 64, "step": 8},
                ),
                "count": ("INT", {"default": 1, "min": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "steps": ("INT", {"default": 28, "min": 1}),
                "guidance": (
                    "FLOAT",
                    {"default": 7.0, "min": 0.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "response_format": (["b64_json", "url"], {"default": "b64_json"}),
                "style": (["none", "vivid", "natural"], {"default": "none"}),
                "vae_tiling": tri_state,
                "vae_slicing": tri_state,
                "additional_image_parameters_json": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
            },
            "optional": {"routing": ("WERK_ROUTING_CONFIG",)},
        }

    RETURN_TYPES = ("WERK_IMAGE_CONFIG", "STRING")
    RETURN_NAMES = ("config", "config_json")
    FUNCTION = "configure"
    CATEGORY = "WERK/Configuration"

    def configure(self, routing: WerkRoutingConfig | None = None, **inputs):
        config = build_image_config(routing=routing, **inputs)
        return config, _json_text(image_config_payload(config))


class WerkVideoConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        tri_state = (["inherit", "enabled", "disabled"], {"default": "inherit"})
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 832, "min": 64, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 480, "min": 64, "step": 8},
                ),
                "count": ("INT", {"default": 1, "min": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "frames": ("INT", {"default": 81, "min": 1}),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 0.1, "step": 0.1},
                ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "guidance": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "step": 0.1},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF},
                ),
                "output_format": (["mp4", "gif"], {"default": "mp4"}),
                "temporal_vae_tiling": tri_state,
                "additional_video_parameters_json": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
            },
            "optional": {"routing": ("WERK_ROUTING_CONFIG",)},
        }

    RETURN_TYPES = ("WERK_VIDEO_CONFIG", "STRING")
    RETURN_NAMES = ("config", "config_json")
    FUNCTION = "configure"
    CATEGORY = "WERK/Configuration"

    def configure(self, routing: WerkRoutingConfig | None = None, **inputs):
        config = build_video_config(routing=routing, **inputs)
        return config, _json_text(video_config_payload(config))


class WerkAudioConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (
                    list(AUDIO_GENERATION_TASKS),
                    {"default": AUDIO_GENERATION_TASK},
                ),
                "duration": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.1, "step": 0.1},
                ),
                "variations": ("INT", {"default": 1, "min": 1}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF},
                ),
                "sample_rate": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": "0 inherits the model/task sample rate.",
                    },
                ),
                "channels": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": "0 inherits the model/task channel count.",
                    },
                ),
                "output_format": (
                    ["wav", "flac", "ogg"],
                    {"default": "wav"},
                ),
                "instrumental": (
                    ["inherit", "enabled", "disabled"],
                    {"default": "inherit"},
                ),
                "voice": ("STRING", {"default": ""}),
                "speed": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "step": 0.01},
                ),
                "additional_audio_parameters_json": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
            },
            "optional": {
                "routing": ("WERK_ROUTING_CONFIG",),
                "language": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "TTS language, for example German. Empty inherits the "
                            "model default. Used only for text-to-speech."
                        ),
                    },
                ),
                "speaking_style": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "TTS voice/style instruction. For Qwen3-TTS VoiceDesign "
                            "this is passed as instruct."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("WERK_AUDIO_CONFIG", "STRING")
    RETURN_NAMES = ("config", "config_json")
    FUNCTION = "configure"
    CATEGORY = "WERK/Configuration"

    def configure(self, routing: WerkRoutingConfig | None = None, **inputs):
        config = build_audio_config(routing=routing, **inputs)
        return config, _json_text(audio_config_payload(config))


class WerkImageGenerateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect the model output from WERK Image Models.",
                    },
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {"config": ("WERK_IMAGE_CONFIG",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("images", "metadata_json", "seed", "result_id", "output_ids")
    FUNCTION = "generate"
    CATEGORY = "WERK/Image"
    OUTPUT_NODE = True

    def generate(
        self,
        connection: WerkConnection,
        model: str,
        prompt: str,
        negative_prompt: str,
        config: WerkImageConfig | None = None,
    ):
        request = build_configured_image_request(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            config=config,
        )
        image_config = config or build_image_config()
        seed = int(image_config.parameters["image.seed"])
        return execute_image_request(connection, request, seed)


class WerkVideoGenerateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect the model output from WERK Video Models.",
                    },
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
            },
            "optional": {
                "config": ("WERK_VIDEO_CONFIG",),
                "initial_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "videos",
        "metadata_json",
        "seed",
        "job_id",
        "result_id",
        "output_ids",
    )
    OUTPUT_IS_LIST = (True, False, False, False, False, False)
    FUNCTION = "generate"
    CATEGORY = "WERK/Video"
    OUTPUT_NODE = True

    def generate(
        self,
        connection: WerkConnection,
        model: str,
        prompt: str,
        negative_prompt: str,
        config: WerkVideoConfig | None = None,
        initial_image=None,
    ):
        request = build_configured_video_request(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            initial_image=initial_image,
            config=config,
        )
        video_config = config or build_video_config()
        seed = int(video_config.parameters["video.seed"])
        return execute_video_request(connection, request, seed)


class WerkAudioGenerateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect the model output from WERK Audio Models.",
                    },
                ),
                "config": (
                    "WERK_AUDIO_CONFIG",
                    {
                        "tooltip": (
                            "Required: connect an active WERK Audio Config. "
                            "Bypassing it would discard all audio/TTS settings."
                        ),
                    },
                ),
                "task": (
                    list(AUDIO_GENERATION_TASKS),
                    {"default": AUDIO_GENERATION_TASK},
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Text to generate from; for TTS this is the spoken text.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "audio",
        "metadata_json",
        "seed",
        "job_id",
        "result_id",
        "output_ids",
    )
    OUTPUT_IS_LIST = (True, False, False, False, False, False)
    FUNCTION = "generate"
    CATEGORY = "WERK/Audio"
    OUTPUT_NODE = True

    def generate(
        self,
        connection: WerkConnection,
        model: str,
        task: str,
        prompt: str,
        negative_prompt: str,
        config: WerkAudioConfig,
    ):
        selected_task = _audio_generation_task(task)
        if selected_task == TEXT_TO_SPEECH_TASK and config is None:
            raise ValueError(
                "text-to-speech requires a connected, active WERK Audio Config; "
                "un-bypass the config node before queueing the workflow"
            )
        request = build_configured_audio_request(
            task=selected_task,
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            config=config,
        )
        audio_config = config or build_audio_config(task=selected_task)
        namespace = "tts" if selected_task == TEXT_TO_SPEECH_TASK else "audio"
        seed = int(audio_config.parameters.get(f"{namespace}.seed", 0))
        return execute_audio_request(connection, selected_task, request, seed)


class WerkAudioProcessNode:
    """Run audio-to-audio transforms through Werk's generic job API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect WERK Audio Models with the same task selected.",
                    },
                ),
                "task": (
                    list(AUDIO_TRANSFORM_TASKS),
                    {"default": AUDIO_TRANSFORM_TASKS[0]},
                ),
                "source_audio": ("AUDIO",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "additional_audio_parameters_json": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
            },
            "optional": {
                "routing": ("WERK_ROUTING_CONFIG",),
                "reference_audio": (
                    "AUDIO",
                    {
                        "tooltip": "Optional target voice reference for voice-conversion.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "audio",
        "metadata_json",
        "seed",
        "job_id",
        "result_id",
        "output_ids",
    )
    OUTPUT_IS_LIST = (True, False, False, False, False, False)
    FUNCTION = "process"
    CATEGORY = "WERK/Audio"
    OUTPUT_NODE = True

    def process(
        self,
        connection: WerkConnection,
        model: str,
        task: str,
        source_audio: Any,
        prompt: str,
        negative_prompt: str,
        additional_audio_parameters_json: str,
        routing: WerkRoutingConfig | None = None,
        reference_audio: Any | None = None,
    ):
        selected_task = _audio_task(task)
        if selected_task not in AUDIO_TRANSFORM_TASKS:
            raise ValueError("task is not an audio transform task")
        request = build_audio_input_job_request(
            task=selected_task,
            model=model,
            audio=source_audio,
            reference_audio=reference_audio,
            prompt=prompt,
            negative_prompt=negative_prompt,
            additional_audio_parameters_json=additional_audio_parameters_json,
            routing=routing,
        )
        return execute_audio_process_request(connection, selected_task, request)


class WerkAudioAnalyzeNode:
    """Transcribe, detect, analyze, or embed native ComfyUI audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect WERK Audio Models with the same task selected.",
                    },
                ),
                "task": (
                    list(AUDIO_TEXT_OUTPUT_TASKS),
                    {"default": AUDIO_TRANSCRIPTION_TASKS[0]},
                ),
                "source_audio": ("AUDIO",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "additional_audio_parameters_json": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
            },
            "optional": {"routing": ("WERK_ROUTING_CONFIG",)},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "results",
        "metadata_json",
        "job_id",
        "result_id",
        "output_ids",
    )
    OUTPUT_IS_LIST = (True, False, False, False, False)
    FUNCTION = "analyze"
    CATEGORY = "WERK/Audio"
    OUTPUT_NODE = True

    def analyze(
        self,
        connection: WerkConnection,
        model: str,
        task: str,
        source_audio: Any,
        prompt: str,
        additional_audio_parameters_json: str,
        routing: WerkRoutingConfig | None = None,
    ):
        selected_task = _audio_task(task)
        if selected_task not in AUDIO_TEXT_OUTPUT_TASKS:
            raise ValueError("task is not an audio analysis task")
        request = build_audio_input_job_request(
            task=selected_task,
            model=model,
            audio=source_audio,
            prompt=prompt,
            additional_audio_parameters_json=additional_audio_parameters_json,
            routing=routing,
        )
        return execute_audio_analysis_request(connection, selected_task, request)


class WerkVisionAnalyzeNode:
    """Inspect one or more native ComfyUI images through Werk chat vision."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect the model output from WERK Vision Models.",
                    },
                ),
                "images": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "system_prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "config": ("WERK_VISION_CONFIG",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "analysis",
        "metadata_json",
        "completion_id",
        "finish_reason",
    )
    FUNCTION = "analyze"
    CATEGORY = "WERK/Vision"
    OUTPUT_NODE = True

    def analyze(
        self,
        connection: WerkConnection,
        model: str,
        images: Any,
        prompt: str,
        system_prompt: str,
        config: WerkVisionConfig,
    ):
        request = build_vision_request(
            model=model,
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            config=config,
        )
        return execute_vision_request(connection, request)


NODE_CLASS_MAPPINGS = {
    "WerkConnection": WerkConnectionNode,
    "WerkServerInfo": WerkServerInfoNode,
    "WerkImageModels": WerkImageModelsNode,
    "WerkVisionModels": WerkVisionModelsNode,
    "WerkImageParameters": WerkImageParametersNode,
    "WerkRoutingConfig": WerkRoutingConfigNode,
    "WerkVisionConfig": WerkVisionConfigNode,
    "WerkImageConfig": WerkImageConfigNode,
    "WerkImageGenerate": WerkImageGenerateNode,
    "WerkVideoModels": WerkVideoModelsNode,
    "WerkVideoParameters": WerkVideoParametersNode,
    "WerkVideoConfig": WerkVideoConfigNode,
    "WerkVideoGenerate": WerkVideoGenerateNode,
    "WerkAudioModels": WerkAudioModelsNode,
    "WerkAudioParameters": WerkAudioParametersNode,
    "WerkAudioConfig": WerkAudioConfigNode,
    "WerkAudioGenerate": WerkAudioGenerateNode,
    "WerkAudioProcess": WerkAudioProcessNode,
    "WerkAudioAnalyze": WerkAudioAnalyzeNode,
    "WerkVisionAnalyze": WerkVisionAnalyzeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WerkConnection": "WERK Connection (Beta)",
    "WerkServerInfo": "WERK Server Info (Beta)",
    "WerkImageModels": "WERK Image Models (Beta)",
    "WerkVisionModels": "WERK Vision Models (Beta)",
    "WerkImageParameters": "WERK Image Parameters (Beta)",
    "WerkRoutingConfig": "WERK Routing Config (Beta)",
    "WerkVisionConfig": "WERK Vision Config (Beta)",
    "WerkImageConfig": "WERK Image Config (Beta)",
    "WerkImageGenerate": "WERK Image Generate (Beta)",
    "WerkVideoModels": "WERK Video Models (Beta)",
    "WerkVideoParameters": "WERK Video Parameters (Beta)",
    "WerkVideoConfig": "WERK Video Config (Beta)",
    "WerkVideoGenerate": "WERK Video Generate (Beta)",
    "WerkAudioModels": "WERK Audio Models (Beta)",
    "WerkAudioParameters": "WERK Audio Parameters (Beta)",
    "WerkAudioConfig": "WERK Audio Config (Beta)",
    "WerkAudioGenerate": "WERK Audio Generate (Beta)",
    "WerkAudioProcess": "WERK Audio Process (Beta)",
    "WerkAudioAnalyze": "WERK Audio Analyze (Beta)",
    "WerkVisionAnalyze": "WERK Vision Analyze (Beta)",
}
