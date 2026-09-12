"""Text generation and optional per-request oMLX controls for ComfyUI."""

from __future__ import annotations

import json
from typing import Any, Mapping

try:
    from .client import WerkApiError, WerkClient
    from .config import WerkConnection, WerkTextConfig
    from .protocol import WerkProtocolClient, WerkProtocolError, require_capability
    from .nodes import (
        _capability_entries,
        _json_text,
        _model_entries,
        _normalized_tasks,
        _object,
        _task_statuses,
        _unavailable_task_message,
        build_vision_config,
    )
except ImportError:  # pragma: no cover - direct-module development
    from client import WerkApiError, WerkClient
    from config import WerkConnection, WerkTextConfig
    from protocol import WerkProtocolClient, WerkProtocolError, require_capability
    from nodes import (
        _capability_entries,
        _json_text,
        _model_entries,
        _normalized_tasks,
        _object,
        _task_statuses,
        _unavailable_task_message,
        build_vision_config,
    )

TEXT_TASK = "text-generation"
MAX_MESSAGES = 4096
MAX_MESSAGES_BYTES = 16 * 1024 * 1024
CHAT_FIELDS = {"temperature", "top_p", "max_completion_tokens", "seed", "stop"}


def classify_text_models(models_payload: Any, capabilities_payload: Any) -> dict[str, Any]:
    installed = _model_entries(models_payload)
    capabilities = {entry["id"]: entry for entry in _capability_entries(capabilities_payload)}
    result: dict[str, Any] = {
        "installed": [model["id"] for model in installed],
        "declared": [],
        "available": [],
        "models": [],
    }
    for model in installed:
        capability = capabilities.get(model["id"], model)
        tasks = _normalized_tasks(capability.get("tasks", model.get("tasks", [])))
        available = _normalized_tasks(
            capability.get("available_tasks", model.get("available_tasks", []))
        )
        if TEXT_TASK in tasks:
            result["declared"].append(model["id"])
        if TEXT_TASK in available:
            result["available"].append(model["id"])
        result["models"].append({
            "id": model["id"],
            "tasks": tasks,
            "available_tasks": available,
            "task_statuses": _task_statuses(capability, model),
            "declares_text_generation": TEXT_TASK in tasks,
            "text_generation_probe_eligible": TEXT_TASK in available,
        })
    return result


def _tristate(value: str, label: str) -> bool | None:
    if value not in {"inherit", "enabled", "disabled"}:
        raise ValueError(f"{label} must be inherit, enabled, or disabled")
    return None if value == "inherit" else value == "enabled"


def build_text_config(
    *,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_completion_tokens: int = 1024,
    seed: int = 0,
    stop_sequences_json: str = "[]",
    omlx_thinking: str = "inherit",
    omlx_expert_offload: str = "inherit",
    omlx_expert_cache_mb: int = 8192,
    inherit_sampling: bool = False,
) -> WerkTextConfig:
    # Reuse validation for the common /v1/chat/completions sampling fields.
    common = build_vision_config(
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        seed=seed,
        stop_sequences_json=stop_sequences_json,
    )
    if type(inherit_sampling) is not bool:
        raise ValueError("inherit_sampling must be a boolean")
    fields = dict(common.request_fields)
    if inherit_sampling:
        for key in ("temperature", "top_p", "seed"):
            fields.pop(key, None)
    options: dict[str, Any] = {}
    thinking = _tristate(omlx_thinking, "omlx_thinking")
    offload = _tristate(omlx_expert_offload, "omlx_expert_offload")
    if thinking is not None:
        options["thinking"] = thinking
    if offload is False:
        options["expert_cache_mb"] = 0
    elif offload is True:
        if type(omlx_expert_cache_mb) is not int or not 1 <= omlx_expert_cache_mb <= 1048576:
            raise ValueError("enabled oMLX expert cache must be an integer from 1 to 1048576 MiB")
        options["expert_cache_mb"] = omlx_expert_cache_mb
    return WerkTextConfig(request_fields=fields, omlx_options=options)


def text_config_payload(config: WerkTextConfig) -> dict[str, Any]:
    if not isinstance(config, WerkTextConfig):
        raise TypeError("config must be a WerkTextConfig")
    if set(config.request_fields) - CHAT_FIELDS:
        raise ValueError("text config contains unsupported chat fields")
    payload = dict(config.request_fields)
    if "stop" in payload:
        payload["stop"] = list(payload["stop"])
    if config.omlx_options:
        payload["werk"] = {"omlx": dict(config.omlx_options)}
    return payload


def build_text_request(
    *,
    model: str,
    prompt: str,
    system_prompt: str = "",
    messages_json: str = "[]",
    config: WerkTextConfig | None = None,
) -> dict[str, Any]:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must not be empty")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string")
    if not isinstance(messages_json, str):
        raise TypeError("messages_json must be a string")
    if len(messages_json.encode("utf-8")) > MAX_MESSAGES_BYTES:
        raise ValueError("messages_json exceeds the 16 MiB limit")
    try:
        history = json.loads(messages_json or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(f"messages_json is invalid JSON: {error.msg}") from error
    if not isinstance(history, list) or len(history) > MAX_MESSAGES - 2:
        raise ValueError("messages_json must be an array with at most 4094 messages")
    for message in history:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or message["role"] not in {"system", "developer", "user", "assistant"}
            or not isinstance(message["content"], str)
        ):
            raise ValueError("messages_json entries require a system/developer/user/assistant role and string content")
    messages = []
    if system_prompt.strip() and (not history or history[0] != {"role": "system", "content": system_prompt}):
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return {
        "model": model.strip(),
        "messages": messages,
        "stream": False,
        **text_config_payload(config if config is not None else build_text_config()),
    }


def execute_text_request(connection: WerkConnection, request: Mapping[str, Any]):
    omlx = request.get("werk", {}).get("omlx", {})
    if omlx:
        try:
            capability = require_capability(
                WerkProtocolClient(connection).capabilities(),
                "api.chat.omlx_options",
                allow_experimental=False,
            )
            if set(omlx) - set(capability.get("operations", [])):
                raise ValueError("Werk did not declare the requested oMLX option operations")
        except (WerkProtocolError, ValueError) as error:
            raise ValueError(
                "Cannot verify oMLX request controls. Update and restart the Werk server "
                f"with api.chat.omlx_options support before using explicit options: {error}"
            ) from error
    response = _object(
        WerkClient(connection).post_json("/v1/chat/completions", dict(request)),
        "text completion",
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Werk text response contains no valid choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Werk text response contains no assistant text")
    # Deliberately omit message content/reasoning and any echoed request fields.
    metadata = {key: response[key] for key in ("id", "object", "created", "model", "usage") if key in response}
    metadata["choice"] = {
        "index": choice.get("index", 0),
        "finish_reason": choice.get("finish_reason"),
        "role": message.get("role", "assistant"),
    }
    return (
        message["content"],
        _json_text(metadata),
        response.get("id") if isinstance(response.get("id"), str) else "",
        choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else "",
    )


class WerkTextModelsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "connection": ("WERK_CONNECTION",),
            "refresh_token": ("INT", {"default": 0}),
            "preferred_model": ("STRING", {"default": ""}),
            "require_available": ("BOOLEAN", {
                "default": False,
                "tooltip": "When enabled, require the server's default runtime probe to pass. Leave disabled when Text Config overrides oMLX offload; generation verifies the configured route.",
            }),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "available_models", "metadata_json")
    FUNCTION = "select"
    CATEGORY = "WERK/Discovery"

    def select(self, connection, refresh_token, preferred_model, require_available):
        del refresh_token
        client = WerkClient(connection)
        models = client.get_json("/v1/models")
        try:
            capabilities = client.get_json("/v1/capabilities")
        except WerkApiError:
            capabilities = {}
        classified = classify_text_models(models, capabilities)
        candidates = classified["available"] if require_available else classified["declared"]
        preferred = preferred_model.strip()
        if preferred and preferred not in candidates:
            if preferred in classified["declared"] and require_available:
                raise ValueError(_unavailable_task_message(TEXT_TASK, [preferred], classified["models"]))
            if preferred in classified["installed"]:
                raise ValueError(f"preferred Werk model {preferred!r} does not declare {TEXT_TASK}")
            raise ValueError(f"preferred Werk model {preferred!r} is not installed")
        if preferred:
            selected = preferred
        elif len(candidates) == 1:
            selected = candidates[0]
        elif len(candidates) > 1:
            raise ValueError("multiple matching Werk text models; set preferred_model to one of: " + ", ".join(candidates))
        elif classified["declared"] and require_available:
            raise ValueError(_unavailable_task_message(TEXT_TASK, classified["declared"], classified["models"]))
        else:
            raise ValueError("no installed Werk model declares text-generation")
        return selected, "\n".join(candidates), _json_text(classified)


class WerkTextConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_completion_tokens": ("INT", {"default": 1024, "min": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
            "stop_sequences_json": ("STRING", {"default": "[]", "multiline": True}),
            "omlx_thinking": (["inherit", "enabled", "disabled"], {
                "default": "inherit", "tooltip": "Override oMLX thinking for this request. Inherit uses the Werk server default.",
            }),
            "omlx_expert_offload": (["inherit", "enabled", "disabled"], {
                "default": "inherit", "tooltip": "Inherit uses the server default: automatic hardware/model sizing unless a fixed server budget is configured. Enabled sets a manual cache ceiling; disabled selects native loading. This is separate from prompt/KV persistence.",
            }),
            "omlx_expert_cache_mb": ("INT", {
                "default": 8192, "min": 1, "max": 1048576,
                "tooltip": "Manual expert cache ceiling, used only when offload is enabled. Inherit ignores this field and keeps server Auto sizing. Native memory guards may shrink residency; other weights, KV and workspace need additional memory.",
            }),
        }, "optional": {"inherit_sampling": ("BOOLEAN", {"default": False, "tooltip": "Use server temperature, top-p and seed. Completion limit and explicit stop sequences still apply."})}}

    RETURN_TYPES = ("WERK_TEXT_CONFIG", "STRING")
    RETURN_NAMES = ("config", "config_json")
    FUNCTION = "configure"
    CATEGORY = "WERK/Configuration"

    def configure(self, **inputs):
        config = build_text_config(**inputs)
        return config, _json_text(text_config_payload(config))


class WerkTextGenerateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model": ("STRING", {"forceInput": True, "tooltip": "Connect WERK Text Models.model or an installed model ID."}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "config": ("WERK_TEXT_CONFIG",),
                "messages_json": ("STRING", {"default": "[]", "multiline": True, "tooltip": "Prior text messages as role/content objects; the prompt is appended as the final user message."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "metadata_json", "completion_id", "finish_reason", "model_id", "messages_json")
    FUNCTION = "generate"
    CATEGORY = "WERK/Text"
    OUTPUT_NODE = True

    def generate(self, connection, model, prompt, system_prompt="", config=None, messages_json="[]"):
        request = build_text_request(
            model=model, prompt=prompt, system_prompt=system_prompt,
            config=config, messages_json=messages_json,
        )
        result = execute_text_request(connection, request)
        history = [*request["messages"], {"role": "assistant", "content": result[0]}]
        return {"ui": {"text": [result[0]]}, "result": (*result, request["model"], _json_text(history))}


NODE_CLASS_MAPPINGS = {
    "WerkTextModels": WerkTextModelsNode,
    "WerkTextConfig": WerkTextConfigNode,
    "WerkTextGenerate": WerkTextGenerateNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WerkTextModels": "WERK Text Models (Beta)",
    "WerkTextConfig": "WERK Text Config (Beta)",
    "WerkTextGenerate": "WERK Text Generate (Beta)",
}
