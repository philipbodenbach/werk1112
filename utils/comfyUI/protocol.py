"""Strict, credential-safe client for Werk's versioned runtime protocol."""

from __future__ import annotations

import json
import math
import ssl
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

try:
    from .config import WerkConnection
except ImportError:  # pragma: no cover - direct-module development
    from config import WerkConnection


PROTOCOL_MAJOR = 1
PROTOCOL_MINOR = 0
PROTOCOL_VERSION_HEADER = "X-Werk-Protocol-Version"
PROTOCOL_VERSION_VALUE = f"{PROTOCOL_MAJOR}.{PROTOCOL_MINOR}"
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
MAX_CAPABILITIES = 4096
MAX_STATES = 4096
MAX_EXPERTS = 4096

CAPABILITY_STATUSES = frozenset(
    {
        "supported",
        "unsupported",
        "unavailable",
        "experimental",
        "externally_managed",
        "metadata_only",
    }
)
PROTOCOL_ERROR_CODES = frozenset(
    {
        "invalid_request",
        "unauthorized",
        "forbidden",
        "not_found",
        "conflict",
        "incompatible_state",
        "expired_handoff",
        "unsupported",
        "unavailable",
        "experimental_opt_in_required",
        "resource_exhausted",
        "corrupt_state",
        "internal",
        "incompatible_protocol",
    }
)
STATE_TIERS = frozenset({"vram", "ram", "disk", "external"})
STATE_STATUSES = frozenset(
    {"ready", "loading", "moving", "unavailable", "quarantined"}
)
EXPERT_TIERS = frozenset({"vram", "ram", "external"})
PRESSURE_LEVELS = frozenset(
    {"normal", "soft", "hard", "emergency", "unknown"}
)


@dataclass(frozen=True, repr=False)
class WerkProtocolError(Exception):
    """Typed protocol/transport failure containing only display-safe fields."""

    operation: str
    sanitized_url: str
    status_code: int | None
    code: str
    safe_message: str
    retryable: bool = False
    request_id: str | None = None

    def __str__(self) -> str:
        status = (
            f" (HTTP {self.status_code})"
            if self.status_code is not None
            else ""
        )
        request = f" [request {self.request_id}]" if self.request_id else ""
        return (
            f"Werk Protocol {self.operation} failed{status} at "
            f"{self.sanitized_url}: {self.code}: {self.safe_message}{request}"
        )

    __repr__ = __str__


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _IncompatibleProtocolVersion(ValueError):
    """A syntactically valid peer version this client cannot consume."""


def _effective_port(parts) -> int | None:
    if parts.port is not None:
        return parts.port
    if parts.scheme.lower() == "https":
        return 443
    if parts.scheme.lower() == "http":
        return 80
    return None


def _same_origin(left: str, right: str) -> bool:
    first, second = urlsplit(left), urlsplit(right)
    return (
        first.scheme.lower() == second.scheme.lower()
        and (first.hostname or "").lower() == (second.hostname or "").lower()
        and _effective_port(first) == _effective_port(second)
    )


def _sanitized_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return f"{parts.scheme}://{host}{parts.path}"


def _redacted(value: object, secrets: Iterable[str]) -> str:
    safe = str(value)
    for secret in secrets:
        if secret:
            safe = safe.replace(secret, "[redacted]")
    safe = "".join(
        character if character >= " " and character != "\x7f" else " "
        for character in safe
    )
    return safe[:1000]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")  # noqa: TRY004
    return value


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _request_id(value: Any) -> str:
    request_id = _string(value, "request_id")
    if len(request_id) > 128 or not all(
        character.isascii()
        and (character.isalnum() or character in "._-")
        for character in request_id
    ):
        raise ValueError("request_id is invalid")
    return request_id


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _optional_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")  # noqa: TRY004
    return value


def _finite_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004
            f"{label} must be a finite number >= {minimum:g}"
        )
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        raise ValueError(
            f"{label} must be a finite number >= {minimum:g}"
        ) from None
    if not math.isfinite(converted) or converted < minimum:
        raise ValueError(f"{label} must be a finite number >= {minimum:g}")
    return converted


def _version(value: Any, label: str = "protocol") -> dict[str, int]:
    root = _object(value, label)
    major = _integer(root.get("major"), f"{label}.major")
    minor = _integer(root.get("minor"), f"{label}.minor")
    if major != PROTOCOL_MAJOR or minor > PROTOCOL_MINOR:
        raise _IncompatibleProtocolVersion(
            f"incompatible Werk Protocol version {major}.{minor}; "
            f"this node supports {PROTOCOL_MAJOR}.{PROTOCOL_MINOR}"
        )
    return {"major": major, "minor": minor}


def _response_protocol_version(headers: Any) -> dict[str, int] | None:
    """Parse the optional HTTP version declaration without guessing.

    Older 1.0 servers may omit the header, in which case the JSON envelope is
    still authoritative. A present declaration must be unique, well formed,
    supported, and later match that envelope.
    """

    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        values = list(get_all(PROTOCOL_VERSION_HEADER) or [])
    else:
        value = headers.get(PROTOCOL_VERSION_HEADER)
        values = [] if value is None else [value]
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("duplicate Werk Protocol version header")
    value = values[0]
    if not isinstance(value, str):
        raise TypeError("invalid Werk Protocol version header")
    parts = value.strip().split(".")
    if (
        len(parts) != 2
        or not all(part and part.isascii() and part.isdigit() for part in parts)
    ):
        raise ValueError("invalid Werk Protocol version header")
    major, minor = (int(part, 10) for part in parts)
    if major > 65535 or minor > 65535:
        raise ValueError("invalid Werk Protocol version header")
    return _version({"major": major, "minor": minor}, "protocol header")


def _match_protocol_versions(
    declared: dict[str, int] | None, envelope: dict[str, int]
) -> None:
    if declared is not None and declared != envelope:
        raise ValueError(
            "Werk Protocol version header does not match its envelope"
        )


def _state_summary(value: Any, label: str = "state") -> dict[str, Any]:
    state = _object(value, label)
    tier = _string(state.get("tier"), f"{label}.tier")
    status = _string(state.get("status"), f"{label}.status")
    if tier not in STATE_TIERS:
        raise ValueError(f"{label}.tier has an unknown value")
    if status not in STATE_STATUSES:
        raise ValueError(f"{label}.status has an unknown value")
    bytes_value = _optional_integer(state.get("bytes"), f"{label}.bytes")
    expires = _optional_integer(
        state.get("expires_unix_ms"), f"{label}.expires_unix_ms"
    )
    return {
        "id": _string(state.get("id"), f"{label}.id"),
        "model_id": _string(state.get("model_id"), f"{label}.model_id"),
        "tier": tier,
        "status": status,
        "bytes": bytes_value,
        "created_unix_ms": _integer(
            state.get("created_unix_ms"), f"{label}.created_unix_ms"
        ),
        "last_accessed_unix_ms": _integer(
            state.get("last_accessed_unix_ms"),
            f"{label}.last_accessed_unix_ms",
        ),
        "expires_unix_ms": expires,
        "pinned": _boolean(state.get("pinned"), f"{label}.pinned"),
        "backend": _string(state.get("backend"), f"{label}.backend"),
        "reusable": _boolean(state.get("reusable"), f"{label}.reusable"),
    }


def _tier_status(value: Any, label: str) -> dict[str, Any]:
    tier = _object(value, label)
    pressure = _string(tier.get("pressure"), f"{label}.pressure")
    if pressure not in PRESSURE_LEVELS:
        raise ValueError(f"{label}.pressure has an unknown value")
    return {
        "capacity_bytes": _optional_integer(
            tier.get("capacity_bytes"), f"{label}.capacity_bytes"
        ),
        "available_bytes": _optional_integer(
            tier.get("available_bytes"), f"{label}.available_bytes"
        ),
        "managed_bytes": _integer(
            tier.get("managed_bytes"), f"{label}.managed_bytes"
        ),
        "reserved_bytes": _integer(
            tier.get("reserved_bytes"), f"{label}.reserved_bytes"
        ),
        "pressure": pressure,
    }


def _expert_identifier(value: Any, label: str) -> str:
    expert_id = _string(value, label)
    if (
        len(expert_id) > 128
        or ".." in expert_id
        or not all(
            character.isascii()
            and (character.isalnum() or character in "_-.")
            for character in expert_id
        )
    ):
        raise ValueError(f"{label} is not a valid opaque expert ID")
    return expert_id


def _expert_model_identifier(value: Any, label: str) -> str:
    model_id = _string(value, label)
    try:
        encoded_length = len(model_id.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError(f"{label} is invalid") from None
    if encoded_length > 256 or any(
        character < " " or character == "\x7f" for character in model_id
    ):
        raise ValueError(f"{label} is invalid")
    return model_id


def _expert_summary(value: Any, label: str = "expert") -> dict[str, Any]:
    expert = _object(value, label)
    tier = _string(expert.get("tier"), f"{label}.tier")
    if tier not in EXPERT_TIERS:
        raise ValueError(f"{label}.tier has an unknown value")
    return {
        "id": _expert_identifier(expert.get("id"), f"{label}.id"),
        "model_id": _expert_model_identifier(
            expert.get("model_id"), f"{label}.model_id"
        ),
        "tier": tier,
        "bytes": _optional_integer(expert.get("bytes"), f"{label}.bytes"),
        "hotness": _finite_number(expert.get("hotness"), f"{label}.hotness"),
        "pinned": _boolean(expert.get("pinned"), f"{label}.pinned"),
        "last_used_unix_ms": _optional_integer(
            expert.get("last_used_unix_ms"), f"{label}.last_used_unix_ms"
        ),
    }


def _expert_summaries(data: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    experts = data.get("experts")
    if not isinstance(experts, list) or len(experts) > MAX_EXPERTS:
        raise ValueError(f"{label} experts must be a bounded array")
    validated = [
        _expert_summary(expert, f"experts[{index}]")
        for index, expert in enumerate(experts)
    ]
    identities = [(expert["model_id"], expert["id"]) for expert in validated]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{label} returned a duplicate expert identity")
    return validated


def _expert_page(value: Any, label: str) -> dict[str, Any]:
    data = _object(value, label)
    validated = _expert_summaries(data, label)
    cursor = data.get("next_cursor")
    if cursor is not None:
        cursor = _string(cursor, f"{label} next_cursor")
        try:
            cursor_length = len(cursor.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError(f"{label} next_cursor is invalid") from None
        if cursor_length > 256 or any(
            character < " " or character == "\x7f" for character in cursor
        ):
            raise ValueError(f"{label} next_cursor is invalid")
    return {"experts": validated, "next_cursor": cursor}


class WerkProtocolClient:
    """Synchronous bounded transport used only by additive runtime nodes."""

    def __init__(
        self,
        connection: WerkConnection,
        *,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ) -> None:
        self.connection = connection
        self.max_response_bytes = int(max_response_bytes)
        self.max_request_bytes = int(max_request_bytes)
        if self.max_response_bytes <= 0 or self.max_request_bytes <= 0:
            raise ValueError("Werk Protocol byte limits must be greater than zero")
        context = None
        if connection.server_url.startswith("https://"):
            context = (
                ssl.create_default_context()
                if connection.verify_tls
                else ssl._create_unverified_context()
            )
        handlers = [_NoRedirect()]
        if context is not None:
            handlers.append(HTTPSHandler(context=context))
        self._opener = build_opener(*handlers)

    def __repr__(self) -> str:
        return (
            "WerkProtocolClient("
            f"server_url={self.connection.server_url!r}, "
            f"authentication_configured={bool(self.connection.api_key)!r}, "
            f"max_response_bytes={self.max_response_bytes!r}, "
            f"max_request_bytes={self.max_request_bytes!r})"
        )

    def _url(self, path: str, query: Mapping[str, Any] | None = None) -> str:
        parsed_path = urlsplit(path)
        segments = parsed_path.path.split("/")
        if (
            parsed_path.scheme
            or parsed_path.netloc
            or parsed_path.query
            or parsed_path.fragment
            or not parsed_path.path.startswith("/werk/v1/")
            or any(part == ".." for part in segments)
            or any(character in path for character in ("\r", "\n", " "))
        ):
            raise self._error("build request", path, None, "invalid_path", "invalid protocol path")
        url = urljoin(self.connection.server_url + "/", path.lstrip("/"))
        if not _same_origin(self.connection.server_url, url):
            raise self._error(
                "build request",
                url,
                None,
                "invalid_path",
                "protocol endpoint escaped the Werk origin",
            )
        if query:
            values = {
                key: value
                for key, value in query.items()
                if value is not None and value != ""
            }
            if values:
                url = f"{url}?{urlencode(values, doseq=True)}"
        return url

    def _error(
        self,
        operation: str,
        url: str,
        status: int | None,
        code: str,
        message: object,
        *,
        retryable: bool = False,
        request_id: str | None = None,
        secrets: Iterable[str] = (),
    ) -> WerkProtocolError:
        redactions = (self.connection.api_key, *tuple(secrets))
        safe_request_id = request_id
        if safe_request_id is not None:
            safe_request_id = _redacted(safe_request_id, redactions)
            if "[redacted]" in safe_request_id:
                safe_request_id = None
        return WerkProtocolError(
            operation=operation,
            sanitized_url=_sanitized_url(url),
            status_code=status,
            code=code,
            safe_message=_redacted(message, redactions),
            retryable=retryable,
            request_id=safe_request_id,
        )

    def _read_bounded(self, response, operation: str, url: str) -> bytes:
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                if int(length) > self.max_response_bytes:
                    raise self._error(
                        operation,
                        url,
                        None,
                        "response_too_large",
                        f"response exceeds {self.max_response_bytes} bytes",
                    )
            except ValueError:
                pass
        body = response.read(self.max_response_bytes + 1)
        if len(body) > self.max_response_bytes:
            raise self._error(
                operation,
                url,
                None,
                "response_too_large",
                f"response exceeds {self.max_response_bytes} bytes",
            )
        return body

    def _typed_http_error(
        self,
        error: HTTPError,
        operation: str,
        url: str,
        secrets: Iterable[str],
    ) -> WerkProtocolError:
        if error.code in {301, 302, 303, 307, 308}:
            return self._error(
                operation,
                url,
                error.code,
                "redirect_rejected",
                "redirects are not allowed for Werk Protocol requests",
                secrets=secrets,
            )
        body = error.read(MAX_ERROR_BYTES + 1)
        if len(body) > MAX_ERROR_BYTES:
            return self._error(
                operation,
                url,
                error.code,
                "invalid_error_envelope",
                "Werk Protocol error response exceeded its safe display limit",
                secrets=secrets,
            )
        try:
            payload = json.loads(body.decode("utf-8"))
            root = _object(payload, "error envelope")
            declared_version = _response_protocol_version(error.headers)
            envelope_version = _version(root.get("protocol"))
            _match_protocol_versions(declared_version, envelope_version)
            request_id = _request_id(root.get("request_id"))
            detail = _object(root.get("error"), "error")
            code = _string(detail.get("code"), "error.code")
            if code not in PROTOCOL_ERROR_CODES:
                raise ValueError("error.code has an unknown value")
            message = _string(detail.get("message"), "error.message")
            retryable = _boolean(detail.get("retryable"), "error.retryable")
            return self._error(
                operation,
                url,
                error.code,
                code,
                message,
                retryable=retryable,
                request_id=request_id,
                secrets=secrets,
            )
        except _IncompatibleProtocolVersion as parse_error:
            return self._error(
                operation,
                url,
                error.code,
                "incompatible_protocol",
                parse_error,
                secrets=secrets,
            )
        except (
            json.JSONDecodeError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as parse_error:
            return self._error(
                operation,
                url,
                error.code,
                "invalid_error_envelope",
                f"server returned an invalid Werk Protocol error: {parse_error}",
                secrets=secrets,
            )

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
        *,
        secrets: Iterable[str] = (),
    ) -> Any:
        url = self._url(path, query)
        operation = f"{method} {urlsplit(url).path}"
        body: bytes | None = None
        headers = {
            "Accept": "application/json",
            PROTOCOL_VERSION_HEADER: PROTOCOL_VERSION_VALUE,
        }
        if payload is not None:
            try:
                body = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeError) as error:
                raise self._error(
                    operation,
                    url,
                    None,
                    "invalid_request",
                    error,
                    secrets=secrets,
                ) from None
            if len(body) > self.max_request_bytes:
                raise self._error(
                    operation,
                    url,
                    None,
                    "request_too_large",
                    f"request exceeds {self.max_request_bytes} bytes",
                    secrets=secrets,
                )
            headers["Content-Type"] = "application/json"
        if self.connection.api_key:
            if "\r" in self.connection.api_key or "\n" in self.connection.api_key:
                raise self._error(
                    operation,
                    url,
                    None,
                    "invalid_request",
                    "API key contains invalid header characters",
                )
            headers["Authorization"] = f"Bearer {self.connection.api_key}"
        try:
            request = Request(url, data=body, method=method, headers=headers)
        except (TypeError, ValueError) as error:
            raise self._error(
                operation,
                url,
                None,
                "invalid_request",
                error,
                secrets=secrets,
            ) from None
        try:
            with self._opener.open(
                request, timeout=self.connection.timeout_seconds
            ) as response:
                try:
                    declared_version = _response_protocol_version(response.headers)
                except _IncompatibleProtocolVersion as error:
                    raise self._error(
                        operation,
                        url,
                        None,
                        "incompatible_protocol",
                        error,
                        secrets=secrets,
                    ) from None
                except (TypeError, ValueError) as error:
                    raise self._error(
                        operation,
                        url,
                        None,
                        "invalid_response",
                        error,
                        secrets=secrets,
                    ) from None
                content_type = response.headers.get("Content-Type", "")
                if content_type.split(";", 1)[0].strip().lower() != "application/json":
                    raise self._error(
                        operation,
                        url,
                        None,
                        "invalid_response",
                        "Werk Protocol response is not application/json",
                        secrets=secrets,
                    )
                raw = self._read_bounded(response, operation, url)
        except HTTPError as error:
            raise self._typed_http_error(error, operation, url, secrets) from None
        except WerkProtocolError:
            raise
        except (URLError, OSError, TimeoutError) as error:
            reason = getattr(error, "reason", None)
            raise self._error(
                operation,
                url,
                None,
                "transport_error",
                reason or error,
                retryable=True,
                secrets=secrets,
            ) from None
        try:
            envelope = _object(
                json.loads(raw.decode("utf-8")), "Werk Protocol envelope"
            )
            envelope_version = _version(envelope.get("protocol"))
            _match_protocol_versions(declared_version, envelope_version)
            _request_id(envelope.get("request_id"))
            if "data" not in envelope:
                raise ValueError("Werk Protocol envelope is missing data")
            return envelope["data"]
        except _IncompatibleProtocolVersion as error:
            raise self._error(
                operation,
                url,
                None,
                "incompatible_protocol",
                error,
                secrets=secrets,
            ) from None
        except (json.JSONDecodeError, UnicodeError, ValueError) as error:
            raise self._error(
                operation,
                url,
                None,
                "invalid_response",
                error,
                secrets=secrets,
            ) from None

    def info(self) -> dict[str, Any]:
        data = _object(self._request("GET", "/werk/v1/info"), "runtime info")
        _version(data.get("protocol"), "runtime info protocol")
        limits = _object(data.get("limits"), "runtime info limits")
        validated_limits = {
            key: _integer(limits.get(key), f"limits.{key}", minimum=1)
            for key in (
                "max_page_size",
                "max_state_ids_per_operation",
                "max_expert_ids_per_operation",
                "max_request_bytes",
                "max_handoff_bytes",
                "max_ttl_seconds",
            )
        }
        return {
            "service": _string(data.get("service"), "runtime info service"),
            "service_version": _string(
                data.get("service_version"), "runtime info service_version"
            ),
            "protocol": _version(data.get("protocol"), "runtime info protocol"),
            "active_backend": _string(
                data.get("active_backend"), "runtime info active_backend"
            ),
            "limits": validated_limits,
        }

    def capabilities(self) -> dict[str, Any]:
        data = _object(
            self._request("GET", "/werk/v1/capabilities"), "capabilities"
        )
        entries = data.get("capabilities")
        if not isinstance(entries, list) or len(entries) > MAX_CAPABILITIES:
            raise ValueError("capabilities must be a bounded array")
        validated = []
        seen = set()
        for index, value in enumerate(entries):
            entry = _object(value, f"capabilities[{index}]")
            capability_id = _string(entry.get("id"), f"capabilities[{index}].id")
            status = _string(entry.get("status"), f"capabilities[{index}].status")
            if status not in CAPABILITY_STATUSES:
                raise ValueError(
                    f"capabilities[{index}].status has an unknown value"
                )
            if capability_id in seen:
                raise ValueError(f"capability {capability_id!r} was returned twice")
            seen.add(capability_id)
            operations = entry.get("operations", [])
            if not isinstance(operations, list) or len(operations) > 128:
                raise ValueError(f"capabilities[{index}].operations must be bounded")
            validated.append(
                {
                    "id": capability_id,
                    "status": status,
                    "detail": _string(
                        entry.get("detail"),
                        f"capabilities[{index}].detail",
                        allow_empty=True,
                    ),
                    "operations": [
                        _string(
                            operation,
                            f"capabilities[{index}].operations",
                        )
                        for operation in operations
                    ],
                }
            )
        return {"capabilities": validated}

    def states(self, query: Mapping[str, Any]) -> dict[str, Any]:
        data = _object(
            self._request("GET", "/werk/v1/states", query=query),
            "runtime states",
        )
        states = data.get("states")
        if not isinstance(states, list) or len(states) > MAX_STATES:
            raise ValueError("runtime states must be a bounded array")
        cursor = data.get("next_cursor")
        if cursor is not None:
            cursor = _string(cursor, "next_cursor")
        return {
            "states": [
                _state_summary(value, f"states[{index}]")
                for index, value in enumerate(states)
            ],
            "next_cursor": cursor,
        }

    def state_action(
        self, state_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        data = _object(
            self._request(
                "POST", f"/werk/v1/states/{state_id}/actions", payload
            ),
            "state action",
        )
        return {
            "state": _state_summary(data.get("state")),
            "changed": _boolean(data.get("changed"), "state action changed"),
            "dry_run": _boolean(data.get("dry_run"), "state action dry_run"),
        }

    def prune_states(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        data = _object(
            self._request("POST", "/werk/v1/states/prune", payload),
            "state prune",
        )
        return {
            "matched": _integer(data.get("matched"), "state prune matched"),
            "removed": _integer(data.get("removed"), "state prune removed"),
            "bytes": _optional_integer(data.get("bytes"), "state prune bytes"),
            "dry_run": _boolean(data.get("dry_run"), "state prune dry_run"),
        }

    def memory(self) -> dict[str, Any]:
        data = _object(
            self._request("GET", "/werk/v1/memory"), "memory status"
        )
        pressure = _string(data.get("overall_pressure"), "overall_pressure")
        if pressure not in PRESSURE_LEVELS:
            raise ValueError("overall_pressure has an unknown value")
        counters = _object(data.get("counters"), "memory counters")
        if len(counters) > 1024:
            raise ValueError("memory counters must be bounded")
        last_action = _optional_integer(
            data.get("last_action_unix_ms"), "last_action_unix_ms"
        )
        return {
            "observed_at_unix_ms": _integer(
                data.get("observed_at_unix_ms"), "observed_at_unix_ms"
            ),
            "overall_pressure": pressure,
            "topology": _string(data.get("topology"), "topology"),
            "host": _tier_status(data.get("host"), "host"),
            "accelerator": _tier_status(data.get("accelerator"), "accelerator"),
            "last_action_unix_ms": last_action,
            "counters": {
                _string(key, "memory counter name"): _integer(
                    value, f"memory counter {key}"
                )
                for key, value in counters.items()
            },
        }

    def experts(self, query: Mapping[str, Any]) -> dict[str, Any]:
        allow_experimental = query.get("allow_experimental", False)
        if not isinstance(allow_experimental, bool):
            raise TypeError("expert list allow_experimental must be a boolean")
        wire_query = dict(query)
        wire_query["allow_experimental"] = (
            "true" if allow_experimental else "false"
        )
        secrets = tuple(
            str(value)
            for key, value in query.items()
            if key in {"model_id", "cursor"} and value
        )
        response = _expert_page(
            self._request(
                "GET",
                "/werk/v1/experts",
                query=wire_query,
                secrets=secrets,
            ),
            "expert list",
        )
        limit = query.get("limit")
        if _is_int(limit) and len(response["experts"]) > limit:
            raise ValueError("expert list exceeded the requested page limit")
        return response

    def expert_action(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        secret_values = [payload.get("model_id")]
        expert_ids = payload.get("expert_ids")
        if not isinstance(expert_ids, list):
            raise TypeError("expert action expert_ids must be an array")
        secret_values.extend(expert_ids)
        data = _object(
            self._request(
                "POST",
                "/werk/v1/experts/actions",
                payload,
                secrets=tuple(
                    value for value in secret_values if isinstance(value, str) and value
                ),
            ),
            "expert action",
        )
        experts = _expert_summaries(data, "expert action")
        changed = _integer(data.get("changed"), "expert action changed")
        dry_run = _boolean(data.get("dry_run"), "expert action dry_run")
        requested_model = payload.get("model_id")
        requested_ids = expert_ids
        allowed_ids = {
            value for value in requested_ids if isinstance(value, str)
        }
        if (
            len(experts) > len(requested_ids)
            or changed > len(requested_ids)
            or any(
                expert["model_id"] != requested_model
                or expert["id"] not in allowed_ids
                for expert in experts
            )
        ):
            raise ValueError("expert action response did not match its explicit selection")
        requested_dry_run = payload.get("dry_run", False)
        if not isinstance(requested_dry_run, bool) or dry_run != requested_dry_run:
            raise ValueError("expert action response dry_run did not match the request")
        return {"experts": experts, "changed": changed, "dry_run": dry_run}

    def prefill(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        data = _object(
            self._request("POST", "/werk/v1/prefill", payload), "prefill"
        )
        handoff = _string(data.get("handoff"), "prefill handoff")
        if not 32 <= len(handoff) <= 4096:
            raise ValueError("prefill handoff has an invalid length")
        state_id = data.get("state_id")
        if state_id is not None:
            state_id = _string(state_id, "prefill state_id")
        tier = _string(data.get("tier"), "prefill tier")
        if tier not in STATE_TIERS:
            raise ValueError("prefill tier has an unknown value")
        return {
            "handoff": handoff,
            "state_id": state_id,
            "prompt_tokens": _integer(
                data.get("prompt_tokens"), "prefill prompt_tokens"
            ),
            "reused": _boolean(data.get("reused"), "prefill reused"),
            "tier": tier,
            "expires_unix_ms": _integer(
                data.get("expires_unix_ms"), "prefill expires_unix_ms"
            ),
        }

    def decode(
        self, payload: Mapping[str, Any], *, handoff_secret: str
    ) -> dict[str, Any]:
        data = _object(
            self._request(
                "POST",
                "/werk/v1/decode",
                payload,
                secrets=(handoff_secret,),
            ),
            "decode",
        )
        updated_handoff = data.get("handoff")
        if updated_handoff is not None:
            updated_handoff = _string(updated_handoff, "decode handoff")
            if not 32 <= len(updated_handoff) <= 4096:
                raise ValueError("decode handoff has an invalid length")
        return {
            "text": _string(data.get("text"), "decode text", allow_empty=True),
            "handoff": updated_handoff,
            "completion_tokens": _integer(
                data.get("completion_tokens"), "decode completion_tokens"
            ),
            "finish_reason": _string(
                data.get("finish_reason"), "decode finish_reason"
            ),
        }


def capability_by_id(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries = payload.get("capabilities", [])
    if not isinstance(entries, list):
        raise ValueError("capabilities must be an array")  # noqa: TRY004
    return {entry["id"]: entry for entry in entries}


def require_capability(
    payload: Mapping[str, Any],
    capability_id: str,
    *,
    allow_experimental: bool,
    allow_unavailable_probe: bool = False,
    allow_externally_managed: bool = False,
) -> Mapping[str, Any]:
    capability = capability_by_id(payload).get(capability_id)
    if capability is None:
        raise ValueError(
            f"Werk runtime did not declare required capability {capability_id!r}"
        )
    status = capability["status"]
    if status == "supported" or (
        status == "experimental" and allow_experimental
    ) or (
        status == "externally_managed" and allow_externally_managed
    ):
        return capability
    # Some adapters can become experimental only after a model-scoped,
    # functional prefill probe. This opt-in is intentionally narrow: the
    # server still performs compatibility, capability, and experimental gates
    # before backend work, and all non-prefill callers remain fail-closed.
    if status == "unavailable" and allow_experimental and allow_unavailable_probe:
        return capability
    if status == "experimental":
        raise ValueError(
            f"Werk capability {capability_id!r} is experimental; enable explicit opt-in"
        )
    detail = capability.get("detail") or "no detail supplied"
    raise ValueError(
        f"Werk capability {capability_id!r} is {status}: {detail}"
    )
