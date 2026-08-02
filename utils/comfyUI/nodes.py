"""Native Werk1112 nodes for ComfyUI."""

from __future__ import annotations

import json
from typing import Any, Mapping

try:
    from .client import WerkApiError, WerkClient
    from .config import (
        DEFAULT_TIMEOUT_SECONDS,
        WerkConnection,
        WerkImageConfig,
        WerkRoutingConfig,
        environment_api_key,
        environment_max_image_pixels,
        environment_server_url,
    )
    from .image_utils import (
        batch_image_tensors,
        decode_base64_image,
        image_bytes_to_tensor,
    )
except ImportError:  # pragma: no cover - direct-module development
    from client import WerkApiError, WerkClient
    from config import (
        DEFAULT_TIMEOUT_SECONDS,
        WerkConnection,
        WerkImageConfig,
        WerkRoutingConfig,
        environment_api_key,
        environment_max_image_pixels,
        environment_server_url,
    )
    from image_utils import (
        batch_image_tensors,
        decode_base64_image,
        image_bytes_to_tensor,
    )


IMAGE_TASK = "image-generation"
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


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        safe = {}
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"path", "output_path", "local_path", "filesystem_path"}:
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
        metadata = {
            "models": models,
            "capabilities": capabilities,
            "classification": classified,
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


NODE_CLASS_MAPPINGS = {
    "WerkConnection": WerkConnectionNode,
    "WerkServerInfo": WerkServerInfoNode,
    "WerkImageModels": WerkImageModelsNode,
    "WerkImageParameters": WerkImageParametersNode,
    "WerkRoutingConfig": WerkRoutingConfigNode,
    "WerkImageConfig": WerkImageConfigNode,
    "WerkImageGenerate": WerkImageGenerateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WerkConnection": "WERK Connection",
    "WerkServerInfo": "WERK Server Info",
    "WerkImageModels": "WERK Image Models",
    "WerkImageParameters": "WERK Image Parameters",
    "WerkRoutingConfig": "WERK Routing Config",
    "WerkImageConfig": "WERK Image Config",
    "WerkImageGenerate": "WERK Image Generate",
}
