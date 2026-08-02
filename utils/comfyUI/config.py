"""Connection configuration for the Werk1112 ComfyUI nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

DEFAULT_SERVER_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_MAX_IMAGE_PIXELS = 64 * 1024 * 1024


def normalize_server_url(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Werk server URL must not be empty")
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Werk server URL must use http or https")
    if not parsed.hostname:
        raise ValueError("Werk server URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Werk server URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Werk server URL must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def environment_server_url() -> str:
    return os.environ.get("WERK_BASE_URL", DEFAULT_SERVER_URL)


def environment_api_key() -> str:
    return os.environ.get("WERK_API_KEY", "")


def environment_max_image_pixels() -> int:
    raw = os.environ.get("WERK_MAX_IMAGE_PIXELS", str(DEFAULT_MAX_IMAGE_PIXELS))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("WERK_MAX_IMAGE_PIXELS must be an integer") from error
    if value <= 0:
        raise ValueError("WERK_MAX_IMAGE_PIXELS must be greater than zero")
    return value


@dataclass(frozen=True, repr=False)
class WerkConnection:
    server_url: str
    api_key: str = ""
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    verify_tls: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "server_url", normalize_server_url(self.server_url))
        object.__setattr__(self, "api_key", self.api_key.strip())
        if int(self.timeout_seconds) <= 0:
            raise ValueError("Werk HTTP timeout must be greater than zero")
        object.__setattr__(self, "timeout_seconds", int(self.timeout_seconds))

    def __repr__(self) -> str:
        return (
            "WerkConnection("
            f"server_url={self.server_url!r}, "
            f"authentication_configured={bool(self.api_key)!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"verify_tls={self.verify_tls!r})"
        )

    @property
    def safe_status(self) -> str:
        auth = "configured" if self.api_key else "not configured"
        tls = "enabled" if self.verify_tls else "disabled"
        return f"{self.server_url} | authentication: {auth} | TLS verification: {tls}"


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy a config mapping so downstream nodes cannot mutate shared state."""

    return MappingProxyType(dict(value))


@dataclass(frozen=True, repr=False, eq=False)
class WerkRoutingConfig:
    """Validated request routing overrides produced by a ComfyUI config node."""

    request_options: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_options", _immutable_mapping(self.request_options)
        )
        object.__setattr__(self, "parameters", _immutable_mapping(self.parameters))

    def __repr__(self) -> str:
        return (
            "WerkRoutingConfig("
            f"request_options={dict(self.request_options)!r}, "
            f"parameters={dict(self.parameters)!r})"
        )


@dataclass(frozen=True, repr=False, eq=False)
class WerkImageConfig:
    """Validated image request fields and canonical Werk parameters."""

    request_fields: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    routing: WerkRoutingConfig = field(default_factory=WerkRoutingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.routing, WerkRoutingConfig):
            raise TypeError("routing must be a WerkRoutingConfig")
        object.__setattr__(
            self, "request_fields", _immutable_mapping(self.request_fields)
        )
        object.__setattr__(self, "parameters", _immutable_mapping(self.parameters))

    def __repr__(self) -> str:
        return (
            "WerkImageConfig("
            f"request_fields={dict(self.request_fields)!r}, "
            f"parameters={dict(self.parameters)!r}, "
            f"routing={self.routing!r})"
        )
