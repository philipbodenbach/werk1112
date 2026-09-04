"""Additive ComfyUI nodes for Werk's versioned runtime-control protocol."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import quote

try:
    from .config import WerkConnection
    from .protocol import WerkProtocolClient, require_capability
except ImportError:  # pragma: no cover - direct-module development
    from config import WerkConnection
    from protocol import WerkProtocolClient, require_capability


PERSISTENCE_MODES = ("auto", "ephemeral", "memory", "disk")
REUSE_MODES = ("prefer", "disabled", "required")
STATE_TIERS = ("vram", "ram", "disk", "external")
STATE_ACTIONS = ("pin", "unpin", "promote", "demote", "evict")
PROMOTION_TIERS = ("vram", "ram")
DEMOTION_TIERS = ("ram", "disk")
EXPERT_TIERS = ("vram", "ram", "external")
EXPERT_ACTIONS = ("prefetch", "pin", "unpin", "evict")
MAX_LOCAL_STATE_IDS = 4096
MAX_LOCAL_EXPERT_IDS = 4096
MAX_LOCAL_MESSAGES = 256
MAX_LOCAL_PREFILL_BYTES = 512 * 1024
MAX_LOCAL_STOP_SEQUENCES = 16


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _nonempty(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _safe_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")  # noqa: TRY004
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if converted < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return converted


@dataclass(frozen=True, repr=False)
class WerkRuntimeDescriptor:
    """Validated runtime information and capability discovery result."""

    info: Mapping[str, Any]
    capabilities: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))
        object.__setattr__(
            self,
            "capabilities",
            tuple(MappingProxyType(dict(entry)) for entry in self.capabilities),
        )

    def __repr__(self) -> str:
        return (
            "WerkRuntimeDescriptor("
            f"service={self.info.get('service')!r}, "
            f"service_version={self.info.get('service_version')!r}, "
            f"active_backend={self.info.get('active_backend')!r}, "
            f"capability_count={len(self.capabilities)})"
        )


@dataclass(frozen=True, repr=False)
class WerkPersistencePolicy:
    mode: str = "auto"
    reuse: str = "prefer"
    ttl_seconds: int | None = None
    pin: bool = False

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        reuse = str(self.reuse).strip().lower()
        if mode not in PERSISTENCE_MODES:
            raise ValueError(f"unknown persistence mode {mode!r}")
        if reuse not in REUSE_MODES:
            raise ValueError(f"unknown reuse mode {reuse!r}")
        ttl = self.ttl_seconds
        if ttl is not None:
            ttl = _safe_int(ttl, "ttl_seconds", minimum=1)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "reuse", reuse)
        object.__setattr__(self, "ttl_seconds", ttl)
        object.__setattr__(self, "pin", bool(self.pin))

    def payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mode": self.mode,
            "reuse": self.reuse,
            "pin": self.pin,
        }
        if self.ttl_seconds is not None:
            result["ttl_seconds"] = self.ttl_seconds
        return result

    def __repr__(self) -> str:
        return (
            "WerkPersistencePolicy("
            f"mode={self.mode!r}, reuse={self.reuse!r}, "
            f"ttl_seconds={self.ttl_seconds!r}, pin={self.pin!r})"
        )


class WerkStateHandoff:
    """Opaque, non-JSON workflow value carrying a one-time handoff token."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not 32 <= len(value) <= 4096:
            raise ValueError("Werk state handoff is invalid")
        self.__value = value

    def _request_value(self) -> str:
        """Return the wire value only to the adjacent Decode node."""

        return self.__value

    def __repr__(self) -> str:
        return "WerkStateHandoff([opaque])"

    __str__ = __repr__


def _client(connection: WerkConnection) -> WerkProtocolClient:
    if not isinstance(connection, WerkConnection):
        raise TypeError("connection must be a WERK_CONNECTION")
    return WerkProtocolClient(connection)


def _discover(client: WerkProtocolClient) -> WerkRuntimeDescriptor:
    info = client.info()
    capabilities = client.capabilities()
    return WerkRuntimeDescriptor(info, tuple(capabilities["capabilities"]))


def _capabilities(client: WerkProtocolClient) -> Mapping[str, Any]:
    """Mandatory discovery: runtime nodes never fall back to legacy routes."""

    return client.capabilities()


def _parse_messages(value: str) -> list[dict[str, str]]:
    try:
        messages = json.loads(value)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"messages_json must be valid JSON: {error}") from error
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_LOCAL_MESSAGES:
        raise ValueError("messages_json must contain between 1 and 256 messages")
    validated = []
    total = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(
                f"messages_json[{index}] must contain exactly role and content"
            )
        role = message.get("role")
        content = message.get("content")
        if (
            not isinstance(role, str)
            or not role
            or len(role) > 32
            or any(
                not (character.isascii() and (character.islower() or character in "_-"))
                for character in role
            )
        ):
            raise ValueError(f"messages_json[{index}].role is invalid")
        if not isinstance(content, str) or not content:
            raise ValueError(f"messages_json[{index}].content must not be empty")
        total += len(content.encode("utf-8"))
        if total > MAX_LOCAL_PREFILL_BYTES:
            raise ValueError("message content exceeds the 512-KiB prefill limit")
        validated.append({"role": role, "content": content})
    return validated


def _parse_stop(value: str) -> list[str]:
    try:
        stop = json.loads(value)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"stop_sequences_json must be valid JSON: {error}") from error
    if not isinstance(stop, list) or len(stop) > MAX_LOCAL_STOP_SEQUENCES:
        raise ValueError("stop_sequences_json must be an array with at most 16 values")
    if any(
        not isinstance(item, str)
        or not item
        or len(item.encode("utf-8")) > 1024
        for item in stop
    ):
        raise ValueError("stop sequences must be non-empty strings of at most 1024 characters")
    return stop


def _state_ids(value: str) -> list[str]:
    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError(
                "state_ids must be valid JSON or newline-separated IDs: "
                f"{error}"
            ) from error
        if not isinstance(parsed, list):
            raise ValueError("state_ids JSON must be an array")
        values = parsed
    else:
        values = [item for line in raw.splitlines() for item in line.split(",")]
    result = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("state_ids must contain only non-empty strings")
        state_id = item.strip()
        if state_id not in result:
            result.append(state_id)
    if len(result) > MAX_LOCAL_STATE_IDS:
        raise ValueError("state_ids exceeds the local safety limit")
    return result


def _expert_model_id(value: str, *, required: bool) -> str:
    model_id = str(value).strip()
    if not model_id and not required:
        return ""
    try:
        encoded_length = len(model_id.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError(
            "model_id must contain between 1 and 256 safe characters"
        ) from None
    if (
        not model_id
        or encoded_length > 256
        or ".." in model_id
        or any(character < " " or character == "\x7f" for character in model_id)
    ):
        raise ValueError("model_id must contain between 1 and 256 safe characters")
    return model_id


def _expert_ids(value: str) -> list[str]:
    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError(
                "expert_ids must be valid JSON or newline-separated IDs: "
                f"{error}"
            ) from error
        if not isinstance(parsed, list):
            raise ValueError("expert_ids JSON must be an array")
        values = parsed
    else:
        values = [item for line in raw.splitlines() for item in line.split(",")]
    result = []
    for item in values:
        if not isinstance(item, str):
            raise TypeError("expert_ids must contain only strings")
        expert_id = item.strip()
        if (
            not expert_id
            or len(expert_id) > 128
            or ".." in expert_id
            or not all(
                character.isascii()
                and (character.isalnum() or character in "_-.")
                for character in expert_id
            )
        ):
            raise ValueError("expert_ids contains an invalid opaque expert ID")
        if expert_id in result:
            raise ValueError("expert_ids contains a duplicate expert ID")
        result.append(expert_id)
    if len(result) > MAX_LOCAL_EXPERT_IDS:
        raise ValueError("expert_ids exceeds the local safety limit")
    return result


class WerkRuntimeInfoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "refresh_token": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("WERK_RUNTIME_INFO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("runtime_info", "summary", "info_json", "capabilities_json")
    FUNCTION = "discover"
    CATEGORY = "WERK/Runtime"

    def discover(self, connection: WerkConnection, refresh_token: int):
        del refresh_token
        runtime = _discover(_client(connection))
        info = dict(runtime.info)
        capabilities = [dict(value) for value in runtime.capabilities]
        protocol = info["protocol"]
        summary = (
            f"{info['service']} {info['service_version']} | "
            f"protocol {protocol['major']}.{protocol['minor']} | "
            f"backend {info['active_backend']} | {len(capabilities)} capabilities"
        )
        return runtime, summary, _json_text(info), _json_text({"capabilities": capabilities})


class WerkPersistencePolicyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(PERSISTENCE_MODES), {"default": "auto"}),
                "reuse": (list(REUSE_MODES), {"default": "prefer"}),
                "ttl_seconds": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF},
                ),
                "pin": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("WERK_PERSISTENCE_POLICY", "STRING")
    RETURN_NAMES = ("policy", "policy_json")
    FUNCTION = "configure"
    CATEGORY = "WERK/Runtime"

    def configure(self, mode: str, reuse: str, ttl_seconds: int, pin: bool):
        ttl = _safe_int(ttl_seconds, "ttl_seconds")
        policy = WerkPersistencePolicy(mode, reuse, ttl or None, pin)
        return policy, _json_text(policy.payload())


class WerkRuntimeStatesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model_id": ("STRING", {"default": ""}),
                "tier": (["all", *STATE_TIERS], {"default": "all"}),
                "limit": ("INT", {"default": 50, "min": 1, "max": 4096}),
                "cursor": ("STRING", {"default": ""}),
                "refresh_token": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("states_json", "state_ids", "next_cursor", "count")
    FUNCTION = "list_states"
    CATEGORY = "WERK/Runtime"

    def list_states(
        self,
        connection: WerkConnection,
        model_id: str,
        tier: str,
        limit: int,
        cursor: str,
        refresh_token: int,
    ):
        del refresh_token
        client = _client(connection)
        _capabilities(client)
        info = client.info()
        selected_tier = str(tier).strip().lower()
        if selected_tier not in ("all", *STATE_TIERS):
            raise ValueError("tier has an unknown value")
        page_limit = _safe_int(limit, "limit", minimum=1)
        maximum = info["limits"]["max_page_size"]
        if page_limit > maximum:
            raise ValueError(
                f"the server accepts a state page limit of at most {maximum}"
            )
        payload = client.states(
            {
                "model_id": str(model_id).strip() or None,
                "tier": None if selected_tier == "all" else selected_tier,
                "limit": page_limit,
                "cursor": str(cursor).strip() or None,
            }
        )
        ids = [state["id"] for state in payload["states"]]
        return (
            _json_text(payload["states"]),
            "\n".join(ids),
            payload["next_cursor"] or "",
            len(ids),
        )


class WerkStateControlNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "state_id": ("STRING", {"default": ""}),
                "action": (list(STATE_ACTIONS), {"default": "pin"}),
                "target_tier": (
                    ["unchanged", *PROMOTION_TIERS, "disk"],
                    {"default": "unchanged"},
                ),
                "dry_run": ("BOOLEAN", {"default": True}),
                "allow_experimental": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("state_json", "changed", "dry_run", "summary")
    FUNCTION = "control"
    CATEGORY = "WERK/Runtime"
    OUTPUT_NODE = True

    def control(
        self,
        connection: WerkConnection,
        state_id: str,
        action: str,
        target_tier: str,
        dry_run: bool,
        allow_experimental: bool,
    ):
        if not isinstance(dry_run, bool) or not isinstance(allow_experimental, bool):
            raise TypeError("dry_run and allow_experimental must be booleans")
        state_id = _nonempty(state_id, "state_id")
        selected_action = str(action).strip().lower()
        if selected_action not in STATE_ACTIONS:
            raise ValueError("action has an unknown value")
        selected_tier = str(target_tier).strip().lower()
        if selected_tier not in ("unchanged", *PROMOTION_TIERS, "disk"):
            raise ValueError("target_tier has an unknown value")
        if selected_action == "promote" and selected_tier not in PROMOTION_TIERS:
            raise ValueError("promote requires an explicit vram or ram target_tier")
        if selected_action == "demote" and selected_tier not in DEMOTION_TIERS:
            raise ValueError("demote requires an explicit ram or disk target_tier")
        if selected_action not in {"promote", "demote"} and selected_tier != "unchanged":
            raise ValueError("target_tier is only valid for promote and demote")
        client = _client(connection)
        _capabilities(client)
        request: dict[str, Any] = {
            "action": selected_action,
            "dry_run": bool(dry_run),
            "allow_experimental": bool(allow_experimental),
        }
        if selected_action in {"promote", "demote"}:
            request["target_tier"] = selected_tier
        response = client.state_action(quote(state_id, safe=""), request)
        outcome = "unchanged"
        if response["changed"]:
            outcome = "would change" if response["dry_run"] else "changed"
        summary = f"{selected_action} {state_id}: {outcome}"
        return (
            _json_text(response["state"]),
            response["changed"],
            response["dry_run"],
            summary,
        )


class WerkStatePruneNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "selector": (
                    ["ids", "filter", "all"],
                    {"default": "ids"},
                ),
                "state_ids": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "model_id": ("STRING", {"default": ""}),
                "tier": (["all", *STATE_TIERS], {"default": "all"}),
                "older_than_unix_ms": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF},
                ),
                "confirm_all": ("BOOLEAN", {"default": False}),
                "dry_run": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("matched", "removed", "bytes", "dry_run", "summary", "result_json")
    FUNCTION = "prune"
    CATEGORY = "WERK/Runtime"
    OUTPUT_NODE = True

    def prune(
        self,
        connection: WerkConnection,
        selector: str,
        state_ids: str,
        model_id: str,
        tier: str,
        older_than_unix_ms: int,
        confirm_all: bool,
        dry_run: bool,
    ):
        selected = str(selector).strip().lower()
        selected_tier = str(tier).strip().lower()
        if selected not in {"ids", "filter", "all"}:
            raise ValueError("selector has an unknown value")
        if selected_tier not in ("all", *STATE_TIERS):
            raise ValueError("tier has an unknown value")
        client = _client(connection)
        info = client.info()
        _capabilities(client)
        if selected == "ids":
            ids = _state_ids(state_ids)
            if not ids:
                raise ValueError("the ids selector requires at least one state ID")
            maximum = info["limits"]["max_state_ids_per_operation"]
            if len(ids) > maximum:
                raise ValueError(f"the server accepts at most {maximum} state IDs")
            selector_payload: dict[str, Any] = {"kind": "ids", "ids": ids}
        elif selected == "filter":
            cutoff = _safe_int(older_than_unix_ms, "older_than_unix_ms")
            selector_payload = {
                "kind": "filter",
                "model_id": str(model_id).strip() or None,
                "tier": None if selected_tier == "all" else selected_tier,
                "older_than_unix_ms": cutoff or None,
            }
            selector_payload = {
                key: value for key, value in selector_payload.items() if value is not None
            }
            if len(selector_payload) == 1:
                raise ValueError("the filter selector requires at least one constraint")
        else:
            if not confirm_all:
                raise ValueError("the all selector requires confirm_all")
            selector_payload = {"kind": "all", "confirm": True}
        response = client.prune_states(
            {"selector": selector_payload, "dry_run": bool(dry_run)}
        )
        verb = "would remove" if response["dry_run"] else "removed"
        removal_count = (
            response["matched"] if response["dry_run"] else response["removed"]
        )
        summary = (
            f"matched {response['matched']} runtime state(s); "
            f"{verb} {removal_count}"
        )
        return (
            response["matched"],
            response["removed"],
            "" if response["bytes"] is None else str(response["bytes"]),
            response["dry_run"],
            summary,
            _json_text(response),
        )


class WerkMemoryStatusNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "refresh_token": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("overall_pressure", "host", "accelerator", "status_json")
    FUNCTION = "status"
    CATEGORY = "WERK/Runtime"

    def status(self, connection: WerkConnection, refresh_token: int):
        del refresh_token
        client = _client(connection)
        _capabilities(client)
        status = client.memory()

        def render(name: str) -> str:
            tier = status[name]
            capacity = tier["capacity_bytes"]
            available = tier["available_bytes"]
            return (
                f"{name}: pressure={tier['pressure']}, "
                f"capacity={'unknown' if capacity is None else capacity}, "
                f"available={'unknown' if available is None else available}, "
                f"managed={tier['managed_bytes']}, reserved={tier['reserved_bytes']}"
            )

        return (
            status["overall_pressure"],
            render("host"),
            render("accelerator"),
            _json_text(status),
        )


class WerkRuntimeExpertsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model_id": ("STRING", {"default": ""}),
                "tier": (["all", *EXPERT_TIERS], {"default": "all"}),
                "limit": ("INT", {"default": 50, "min": 1, "max": 4096}),
                "cursor": ("STRING", {"default": ""}),
                "allow_experimental": ("BOOLEAN", {"default": False}),
                "refresh_token": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("experts_json", "expert_ids", "next_cursor", "count")
    FUNCTION = "list_experts"
    CATEGORY = "WERK/Runtime"

    def list_experts(
        self,
        connection: WerkConnection,
        model_id: str,
        tier: str,
        limit: int,
        cursor: str,
        allow_experimental: bool,
        refresh_token: int,
    ):
        del refresh_token
        if not isinstance(allow_experimental, bool):
            raise TypeError("allow_experimental must be a boolean")
        model = _expert_model_id(model_id, required=False)
        selected_tier = str(tier).strip().lower()
        if selected_tier not in ("all", *EXPERT_TIERS):
            raise ValueError("tier has an unknown value")
        selected_cursor = str(cursor).strip()
        try:
            cursor_length = len(selected_cursor.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError("cursor is invalid") from None
        if cursor_length > 256 or any(
            character < " " or character == "\x7f"
            for character in selected_cursor
        ):
            raise ValueError("cursor is invalid")
        page_limit = _safe_int(limit, "limit", minimum=1)
        client = _client(connection)
        capabilities = _capabilities(client)
        require_capability(
            capabilities,
            "runtime.experts.residency",
            allow_experimental=bool(allow_experimental),
            allow_externally_managed=True,
        )
        info = client.info()
        maximum = info["limits"]["max_page_size"]
        if page_limit > maximum:
            raise ValueError(f"the server accepts an expert page limit of at most {maximum}")
        response = client.experts(
            {
                "model_id": model or None,
                "tier": None if selected_tier == "all" else selected_tier,
                "limit": page_limit,
                "cursor": selected_cursor or None,
                "allow_experimental": allow_experimental,
            }
        )
        ids = [expert["id"] for expert in response["experts"]]
        return (
            _json_text(response["experts"]),
            "\n".join(ids),
            response["next_cursor"] or "",
            len(ids),
        )


class WerkExpertControlNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model_id": ("STRING", {"default": ""}),
                "expert_ids": ("STRING", {"default": "", "multiline": True}),
                "action": (list(EXPERT_ACTIONS), {"default": "pin"}),
                "target_tier": (
                    ["unchanged", *EXPERT_TIERS],
                    {"default": "unchanged"},
                ),
                "dry_run": ("BOOLEAN", {"default": True}),
                "allow_experimental": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("experts_json", "changed", "dry_run", "summary")
    FUNCTION = "control"
    CATEGORY = "WERK/Runtime"
    OUTPUT_NODE = True

    def control(
        self,
        connection: WerkConnection,
        model_id: str,
        expert_ids: str,
        action: str,
        target_tier: str,
        dry_run: bool,
        allow_experimental: bool,
    ):
        if not isinstance(dry_run, bool) or not isinstance(allow_experimental, bool):
            raise TypeError("dry_run and allow_experimental must be booleans")
        model = _expert_model_id(model_id, required=True)
        ids = _expert_ids(expert_ids)
        if not ids:
            raise ValueError("expert_ids must contain at least one explicit expert ID")
        selected_action = str(action).strip().lower()
        if selected_action not in EXPERT_ACTIONS:
            raise ValueError("action has an unknown value")
        selected_tier = str(target_tier).strip().lower()
        if selected_tier not in ("unchanged", *EXPERT_TIERS):
            raise ValueError("target_tier has an unknown value")
        if selected_action == "prefetch":
            if selected_tier not in {"vram", "ram"}:
                raise ValueError("prefetch requires an explicit vram or ram target_tier")
        elif selected_tier != "unchanged":
            raise ValueError("target_tier is only valid for prefetch")

        client = _client(connection)
        capabilities = _capabilities(client)
        require_capability(
            capabilities,
            "runtime.experts.residency",
            allow_experimental=bool(allow_experimental),
        )
        info = client.info()
        maximum = info["limits"]["max_expert_ids_per_operation"]
        if len(ids) > maximum:
            raise ValueError(f"the server accepts at most {maximum} expert IDs")
        request: dict[str, Any] = {
            "model_id": model,
            "expert_ids": ids,
            "action": selected_action,
            "dry_run": bool(dry_run),
            "allow_experimental": bool(allow_experimental),
        }
        if selected_action == "prefetch":
            request["target_tier"] = selected_tier
        response = client.expert_action(request)
        verb = "would change" if response["dry_run"] else "changed"
        summary = (
            f"{selected_action} matched {len(response['experts'])} expert(s); "
            f"{verb} {response['changed']}"
        )
        return (
            _json_text(response["experts"]),
            response["changed"],
            response["dry_run"],
            summary,
        )


class WerkPrefillNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "model_id": ("STRING", {"default": ""}),
                "input_type": (["text", "messages"], {"default": "text"}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "messages_json": (
                    "STRING",
                    {
                        "default": '[{"role":"user","content":""}]',
                        "multiline": True,
                    },
                ),
                "allow_experimental": ("BOOLEAN", {"default": False}),
            },
            "optional": {"policy": ("WERK_PERSISTENCE_POLICY",)},
        }

    RETURN_TYPES = (
        "WERK_STATE_HANDOFF",
        "STRING",
        "INT",
        "BOOLEAN",
        "STRING",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "handoff",
        "state_id",
        "prompt_tokens",
        "reused",
        "tier",
        "expires_unix_ms",
        "metadata_json",
    )
    FUNCTION = "prefill"
    CATEGORY = "WERK/Runtime"

    def prefill(
        self,
        connection: WerkConnection,
        model_id: str,
        input_type: str,
        text: str,
        messages_json: str,
        allow_experimental: bool,
        policy: WerkPersistencePolicy | None = None,
    ):
        model = _nonempty(model_id, "model_id")
        selected_input = str(input_type).strip().lower()
        if selected_input == "text":
            if not isinstance(text, str) or not text:
                raise ValueError("text prefill input must not be empty")
            if len(text.encode("utf-8")) > MAX_LOCAL_PREFILL_BYTES:
                raise ValueError("text exceeds the 512-KiB prefill limit")
            input_payload: dict[str, Any] = {"type": "text", "text": text}
        elif selected_input == "messages":
            input_payload = {"type": "messages", "messages": _parse_messages(messages_json)}
        else:
            raise ValueError("input_type has an unknown value")
        if policy is not None and not isinstance(policy, WerkPersistencePolicy):
            raise TypeError("policy must be a WERK_PERSISTENCE_POLICY")
        client = _client(connection)
        capabilities = _capabilities(client)
        require_capability(
            capabilities,
            "runtime.pd.prefill",
            allow_experimental=bool(allow_experimental),
            allow_unavailable_probe=True,
        )
        require_capability(
            capabilities,
            "runtime.pd.handoff",
            allow_experimental=bool(allow_experimental),
            allow_unavailable_probe=True,
        )
        request = {
            "model_id": model,
            "input": input_payload,
            "allow_experimental": bool(allow_experimental),
        }
        # An unconnected policy deliberately stays absent on the wire. This
        # preserves the protocol defaults on older servers while allowing an
        # explicitly configured Werk server to supply its own prefill policy.
        if policy is not None:
            request["policy"] = policy.payload()
        response = client.prefill(request)
        handoff = WerkStateHandoff(response["handoff"])
        metadata = {
            key: response[key]
            for key in (
                "state_id",
                "prompt_tokens",
                "reused",
                "tier",
                "expires_unix_ms",
            )
        }
        return (
            handoff,
            response["state_id"] or "",
            response["prompt_tokens"],
            response["reused"],
            response["tier"],
            response["expires_unix_ms"],
            _json_text(metadata),
        )


class WerkDecodeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("WERK_CONNECTION",),
                "handoff": ("WERK_STATE_HANDOFF",),
                "max_tokens": (
                    "INT",
                    {"default": 256, "min": 1, "max": 32 * 1024},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF},
                ),
                "stop_sequences_json": (
                    "STRING",
                    {"default": "[]", "multiline": True},
                ),
                "allow_experimental": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "WERK_STATE_HANDOFF",
        "STRING",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "text",
        "updated_handoff",
        "finish_reason",
        "completion_tokens",
        "metadata_json",
    )
    FUNCTION = "decode"
    CATEGORY = "WERK/Runtime"
    OUTPUT_NODE = True

    def decode(
        self,
        connection: WerkConnection,
        handoff: WerkStateHandoff,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        stop_sequences_json: str,
        allow_experimental: bool,
    ):
        if not isinstance(handoff, WerkStateHandoff):
            raise TypeError("handoff must come from a WERK_STATE_HANDOFF socket")
        maximum = _safe_int(max_tokens, "max_tokens", minimum=1)
        if maximum > 32 * 1024:
            raise ValueError("max_tokens must not exceed 32768")
        try:
            temperature_value = float(temperature)
            top_p_value = float(top_p)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("temperature and top_p must be finite numbers") from error
        if not math.isfinite(temperature_value) or (
            temperature_value != -1.0 and not 0 <= temperature_value <= 2
        ):
            raise ValueError("temperature must be -1 or between 0 and 2")
        if not math.isfinite(top_p_value) or (
            top_p_value != -1.0 and not 0 < top_p_value <= 1
        ):
            raise ValueError("top_p must be -1 or greater than 0 and at most 1")
        if isinstance(seed, bool):
            raise ValueError("seed must be -1 or non-negative")  # noqa: TRY004
        try:
            seed_value = int(seed)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("seed must be -1 or non-negative") from error
        if seed_value < -1 or seed_value > 0x7FFFFFFFFFFFFFFF:
            raise ValueError("seed must be -1 or non-negative")
        client = _client(connection)
        capabilities = _capabilities(client)
        require_capability(
            capabilities,
            "runtime.pd.decode",
            allow_experimental=bool(allow_experimental),
        )
        require_capability(
            capabilities,
            "runtime.pd.handoff",
            allow_experimental=bool(allow_experimental),
        )
        token = handoff._request_value()
        request: dict[str, Any] = {
            "handoff": token,
            "max_tokens": maximum,
            "stop": _parse_stop(stop_sequences_json),
            "allow_experimental": bool(allow_experimental),
        }
        if temperature_value != -1.0:
            request["temperature"] = temperature_value
        if top_p_value != -1.0:
            request["top_p"] = top_p_value
        if seed_value >= 0:
            request["seed"] = seed_value
        response = client.decode(request, handoff_secret=token)
        updated = (
            WerkStateHandoff(response["handoff"])
            if response["handoff"] is not None
            else None
        )
        metadata = {
            "completion_tokens": response["completion_tokens"],
            "finish_reason": response["finish_reason"],
            "has_updated_handoff": updated is not None,
        }
        return (
            response["text"],
            updated,
            response["finish_reason"],
            response["completion_tokens"],
            _json_text(metadata),
        )


NODE_CLASS_MAPPINGS = {
    "WerkRuntimeInfo": WerkRuntimeInfoNode,
    "WerkPersistencePolicy": WerkPersistencePolicyNode,
    "WerkRuntimeStates": WerkRuntimeStatesNode,
    "WerkStateControl": WerkStateControlNode,
    "WerkStatePrune": WerkStatePruneNode,
    "WerkMemoryStatus": WerkMemoryStatusNode,
    "WerkRuntimeExperts": WerkRuntimeExpertsNode,
    "WerkExpertControl": WerkExpertControlNode,
    "WerkPrefill": WerkPrefillNode,
    "WerkDecode": WerkDecodeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WerkRuntimeInfo": "WERK Runtime Info (Beta)",
    "WerkPersistencePolicy": "WERK Persistence Policy (Beta)",
    "WerkRuntimeStates": "WERK Runtime States (Beta)",
    "WerkStateControl": "WERK State Control (Beta)",
    "WerkStatePrune": "WERK State Prune (Beta)",
    "WerkMemoryStatus": "WERK Memory Status (Beta)",
    "WerkRuntimeExperts": "WERK Runtime Experts (Beta)",
    "WerkExpertControl": "WERK Expert Control (Beta)",
    "WerkPrefill": "WERK Prefill (Beta)",
    "WerkDecode": "WERK Decode (Beta)",
}
