"""Native Werk1112 nodes for ComfyUI."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping
from urllib.parse import quote

try:
    from .client import WerkApiError, WerkClient
    from .config import (
        DEFAULT_TIMEOUT_SECONDS,
        WerkConnection,
        WerkImageConfig,
        WerkRoutingConfig,
        WerkVideoConfig,
        environment_api_key,
        environment_max_image_pixels,
        environment_max_video_bytes,
        environment_server_url,
    )
    from .image_utils import (
        batch_image_tensors,
        decode_base64_image,
        image_bytes_to_tensor,
    )
    from .video_utils import image_tensor_to_api_input, video_bytes_to_comfy
except ImportError:  # pragma: no cover - direct-module development
    from client import WerkApiError, WerkClient
    from config import (
        DEFAULT_TIMEOUT_SECONDS,
        WerkConnection,
        WerkImageConfig,
        WerkRoutingConfig,
        WerkVideoConfig,
        environment_api_key,
        environment_max_image_pixels,
        environment_max_video_bytes,
        environment_server_url,
    )
    from image_utils import (
        batch_image_tensors,
        decode_base64_image,
        image_bytes_to_tensor,
    )
    from video_utils import image_tensor_to_api_input, video_bytes_to_comfy


IMAGE_TASK = "image-generation"
VIDEO_GENERATION_TASK = "video-generation"
IMAGE_TO_VIDEO_TASK = "image-to-video"
VIDEO_TASKS = (VIDEO_GENERATION_TASK, IMAGE_TO_VIDEO_TASK)
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
    if timeout < 0 or timeout > 0xFFFFFFFF:
        raise ValueError("inference_timeout_seconds must be between 0 and 4294967295")
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
    if not 64 <= width_value <= 32768 or not 64 <= height_value <= 32768:
        raise ValueError("width and height must be between 64 and 32768")
    if not 1 <= count_value <= 1024:
        raise ValueError("count must be between 1 and 1024")
    if not 1 <= batch_value <= 256:
        raise ValueError("batch_size must be between 1 and 256")
    if count_value > 1 and batch_value > 1:
        raise ValueError(
            "count and batch_size cannot both be greater than 1; "
            "the selected media adapter treats them as alternative image-count controls"
        )
    if not 1 <= steps_value <= 1000:
        raise ValueError("steps must be between 1 and 1000")
    if not 0.0 <= guidance_value <= 100.0:
        raise ValueError("guidance must be between 0 and 100")
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
    if not 64 <= width_value <= 16384 or not 64 <= height_value <= 16384:
        raise ValueError("width and height must be between 64 and 16384")
    if not 1 <= count_value <= 256:
        raise ValueError("count must be between 1 and 256")
    if not 1 <= batch_value <= 64:
        raise ValueError("batch_size must be between 1 and 64")
    if count_value > 1 and batch_value > 1:
        raise ValueError(
            "count and batch_size cannot both be greater than 1; "
            "the selected media adapter treats them as alternative video-count controls"
        )
    if not 1 <= frames_value <= 100_000:
        raise ValueError("frames must be between 1 and 100000")
    if not 0.1 <= fps_value <= 1000.0:
        raise ValueError("fps must be between 0.1 and 1000")
    if not 1 <= steps_value <= 2000:
        raise ValueError("steps must be between 1 and 2000")
    if not 0.0 <= guidance_value <= 100.0:
        raise ValueError("guidance must be between 0 and 100")
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


def safe_video_job_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
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
    """Poll one Werk media job and cancel it best-effort if waiting aborts."""

    record = _object(initial_record, "video job")
    job_id = record.get("id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("Werk video job response contains no valid job id")
    timeout = float(timeout_seconds)
    interval = float(poll_interval_seconds)
    if timeout <= 0:
        raise ValueError("video job timeout must be greater than zero")
    if interval <= 0:
        raise ValueError("video job poll interval must be greater than zero")
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
                        "Werk completed video job contains no inference result"
                    )
                return record
            if status == "failed":
                active = False
                detail = record.get("error")
                message = detail.strip() if isinstance(detail, str) else "unknown failure"
                raise ValueError(f"Werk video job failed: {message}")
            if status == "cancelled":
                active = False
                raise ValueError("Werk video job was cancelled")
            if status not in {"queued", "loading", "running", "encoding"}:
                raise ValueError(f"Werk video job returned unknown status '{status}'")

            check_interrupted()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Werk video job did not complete within {timeout:g} seconds"
                )
            sleep(min(interval, remaining))
            record = _object(
                client.get_json(f"/v1/jobs/{quote(job_id, safe='')}"),
                "video job",
            )
            if record.get("id") != job_id:
                raise ValueError("Werk video job id changed while polling")
    except BaseException:
        if active:
            _cancel_job_best_effort(client, job_id)
        raise


def execute_video_request(
    connection: WerkConnection,
    request: Mapping[str, Any],
    seed: int,
):
    client = WerkClient(connection)
    initial = _object(
        client.post_json("/v1/videos/generations", dict(request)),
        "video generation",
    )
    record = wait_for_video_job(
        client,
        initial,
        timeout_seconds=connection.timeout_seconds,
    )
    result = _object(record.get("result"), "video result")
    outputs = result.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Werk completed video job contains no outputs")

    videos = []
    output_ids = []
    max_bytes = environment_max_video_bytes()
    for item in outputs:
        if not isinstance(item, dict):
            raise ValueError("Werk video result contains an invalid output entry")
        output_id = item.get("id")
        if not isinstance(output_id, str) or not output_id.strip():
            raise ValueError("Werk video output contains no valid output id")
        mime_type = str(item.get("mime_type", "")).lower().split(";", 1)[0]
        if not (mime_type.startswith("video/") or mime_type == "image/gif"):
            raise ValueError(
                f"Werk video output returned non-video MIME type '{mime_type or 'unknown'}'"
            )
        raw, content_type = client.download_bytes(
            f"/v1/outputs/{quote(output_id, safe='')}",
            max_bytes=max_bytes,
        )
        downloaded_type = (
            content_type.lower().split(";", 1)[0] if content_type else ""
        )
        if downloaded_type and not (
            downloaded_type.startswith("video/") or downloaded_type == "image/gif"
        ):
            raise ValueError(
                f"Werk output returned non-video content type '{content_type}'"
            )
        videos.append(video_bytes_to_comfy(raw))
        output_ids.append(output_id)

    result_id = result.get("id")
    if not isinstance(result_id, str):
        result_id = ""
    return (
        videos,
        _json_text(safe_video_job_metadata(record)),
        int(seed),
        str(record["id"]),
        str(result_id),
        "\n".join(output_ids),
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
                    {"default": DEFAULT_TIMEOUT_SECONDS, "min": 1, "max": 86400},
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
        video_classified = classify_video_models(models, capabilities)
        metadata = {
            "models": models,
            "capabilities": capabilities,
            "classification": classified,
            "video_classification": video_classified,
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
        if preferred and preferred in candidates:
            selected = preferred
        elif len(candidates) == 1:
            selected = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                "multiple matching Werk image models; set preferred_model to one of: "
                + ", ".join(candidates)
            )
        elif classified["declared"] and require_available:
            raise ValueError(
                "Werk models declare image-generation, but none is currently runtime probe-eligible: "
                + ", ".join(classified["declared"])
            )
        else:
            raise ValueError("no installed Werk model declares image-generation")
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
        if preferred and preferred in candidates:
            selected = preferred
        elif len(candidates) == 1:
            selected = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                f"multiple matching Werk {selected_task} models; "
                "set preferred_model to one of: " + ", ".join(candidates)
            )
        elif task_models["declared"] and require_available:
            raise ValueError(
                f"Werk models declare {selected_task}, but none is currently "
                "runtime probe-eligible: " + ", ".join(task_models["declared"])
            )
        else:
            raise ValueError(
                f"no installed Werk model declares {selected_task}"
            )
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
                    {"default": 0, "min": 0, "max": 0xFFFFFFFF},
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


class WerkImageConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        tri_state = (["inherit", "enabled", "disabled"], {"default": "inherit"})
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 32768, "step": 8}),
                "height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 32768, "step": 8},
                ),
                "count": ("INT", {"default": 1, "min": 1, "max": 1024}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 256}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 1000}),
                "guidance": (
                    "FLOAT",
                    {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1},
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
                    {"default": 832, "min": 64, "max": 16384, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 480, "min": 64, "max": 16384, "step": 8},
                ),
                "count": ("INT", {"default": 1, "min": 1, "max": 256}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "frames": ("INT", {"default": 81, "min": 1, "max": 100000}),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 0.1, "max": 1000.0, "step": 0.1},
                ),
                "steps": ("INT", {"default": 30, "min": 1, "max": 2000}),
                "guidance": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1},
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


NODE_CLASS_MAPPINGS = {
    "WerkConnection": WerkConnectionNode,
    "WerkServerInfo": WerkServerInfoNode,
    "WerkImageModels": WerkImageModelsNode,
    "WerkImageParameters": WerkImageParametersNode,
    "WerkRoutingConfig": WerkRoutingConfigNode,
    "WerkImageConfig": WerkImageConfigNode,
    "WerkImageGenerate": WerkImageGenerateNode,
    "WerkVideoModels": WerkVideoModelsNode,
    "WerkVideoParameters": WerkVideoParametersNode,
    "WerkVideoConfig": WerkVideoConfigNode,
    "WerkVideoGenerate": WerkVideoGenerateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WerkConnection": "WERK Connection (Beta)",
    "WerkServerInfo": "WERK Server Info (Beta)",
    "WerkImageModels": "WERK Image Models (Beta)",
    "WerkImageParameters": "WERK Image Parameters (Beta)",
    "WerkRoutingConfig": "WERK Routing Config (Beta)",
    "WerkImageConfig": "WERK Image Config (Beta)",
    "WerkImageGenerate": "WERK Image Generate (Beta)",
    "WerkVideoModels": "WERK Video Models (Beta)",
    "WerkVideoParameters": "WERK Video Parameters (Beta)",
    "WerkVideoConfig": "WERK Video Config (Beta)",
    "WerkVideoGenerate": "WERK Video Generate (Beta)",
}
