"""Small, credential-safe HTTP client for the Werk API."""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, HTTPSHandler

try:
    from .config import WerkConnection
except ImportError:  # pragma: no cover - direct-module development
    from config import WerkConnection


DEFAULT_MAX_JSON_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_IMAGE_BYTES = 128 * 1024 * 1024
MAX_ERROR_BYTES = 16 * 1024


@dataclass(frozen=True)
class WerkApiError(Exception):
    operation: str
    sanitized_url: str
    status_code: int | None
    safe_message: str

    def __str__(self) -> str:
        status = f" (HTTP {self.status_code})" if self.status_code is not None else ""
        return f"Werk {self.operation} failed{status} at {self.sanitized_url}: {self.safe_message}"

    __repr__ = __str__


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _effective_port(parts) -> int | None:  # noqa: ANN001
    if parts.port is not None:
        return parts.port
    return 443 if parts.scheme.lower() == "https" else 80 if parts.scheme.lower() == "http" else None


def same_origin(left: str, right: str) -> bool:
    a, b = urlsplit(left), urlsplit(right)
    return (
        a.scheme.lower() == b.scheme.lower()
        and (a.hostname or "").lower() == (b.hostname or "").lower()
        and _effective_port(a) == _effective_port(b)
    )


def _sanitized_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return f"{parts.scheme}://{host}{parts.path}"


class WerkClient:
    def __init__(
        self,
        connection: WerkConnection,
        *,
        max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    ) -> None:
        self.connection = connection
        self.max_json_bytes = int(max_json_bytes)
        self.max_image_bytes = int(max_image_bytes)
        context = None
        if connection.server_url.startswith("https://"):
            context = ssl.create_default_context() if connection.verify_tls else ssl._create_unverified_context()
        handlers = [_NoRedirect()]
        if context is not None:
            handlers.append(HTTPSHandler(context=context))
        self._opener = build_opener(*handlers)

    def _url(self, path: str, query: Mapping[str, Any] | None = None) -> str:
        base = self.connection.server_url + "/"
        url = urljoin(base, path.lstrip("/"))
        parts = urlsplit(url)
        if parts.username is not None or parts.password is not None:
            raise WerkApiError("build request", _sanitized_url(url), None, "URL credentials are not allowed")
        if not same_origin(self.connection.server_url, url):
            raise WerkApiError("build request", _sanitized_url(url), None, "API endpoint escaped the Werk origin")
        if query:
            values = {key: value for key, value in query.items() if value is not None and value != ""}
            url = f"{url}?{urlencode(values)}"
        return url

    def _headers(self, *, include_auth: bool, json_body: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if include_auth and self.connection.api_key:
            headers["Authorization"] = f"Bearer {self.connection.api_key}"
        return headers

    @staticmethod
    def _read_bounded(response, limit: int, operation: str, url: str) -> bytes:  # noqa: ANN001
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                if int(length) > limit:
                    raise WerkApiError(operation, _sanitized_url(url), None, f"response exceeds {limit} bytes")
            except ValueError:
                pass
        data = response.read(limit + 1)
        if len(data) > limit:
            raise WerkApiError(operation, _sanitized_url(url), None, f"response exceeds {limit} bytes")
        return data

    @staticmethod
    def _error_message(body: bytes, fallback: str) -> str:
        text = body[:MAX_ERROR_BYTES].decode("utf-8", "replace")
        try:
            payload = json.loads(text)
            error = payload.get("error", payload) if isinstance(payload, dict) else payload
            if isinstance(error, dict):
                value = error.get("message") or error.get("detail") or error.get("error")
                if isinstance(value, str) and value.strip():
                    return value.strip()[:1000]
            if isinstance(error, str) and error.strip():
                return error.strip()[:1000]
        except (json.JSONDecodeError, UnicodeError):
            pass
        if text.strip() and "<html" not in text.lower():
            return text.strip()[:1000]
        return fallback[:1000]

    def _open(self, request: Request, operation: str, limit: int) -> tuple[bytes, Mapping[str, str]]:
        safe_url = _sanitized_url(request.full_url)
        try:
            with self._opener.open(request, timeout=self.connection.timeout_seconds) as response:
                return self._read_bounded(response, limit, operation, request.full_url), response.headers
        except HTTPError as error:
            location = error.headers.get("Location") if error.headers else None
            if error.code in {301, 302, 303, 307, 308} and location:
                target = urljoin(request.full_url, location)
                message = "redirect rejected"
                if not same_origin(request.full_url, target):
                    message = "cross-origin redirect rejected to protect credentials"
                raise WerkApiError(operation, safe_url, error.code, message) from None
            body = error.read(MAX_ERROR_BYTES + 1)
            message = self._error_message(body, error.reason or "HTTP request failed")
            if self.connection.api_key:
                message = message.replace(self.connection.api_key, "[redacted]")
            raise WerkApiError(operation, safe_url, error.code, message) from None
        except (URLError, OSError, TimeoutError) as error:
            reason = getattr(error, "reason", None)
            safe = str(reason or error)
            if self.connection.api_key:
                safe = safe.replace(self.connection.api_key, "[redacted]")
            raise WerkApiError(operation, safe_url, None, safe[:1000]) from None

    def _json(self, request: Request, operation: str) -> Any:
        body, _ = self._open(request, operation, self.max_json_bytes)
        try:
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeError) as error:
            raise WerkApiError(operation, _sanitized_url(request.full_url), None, f"invalid JSON response: {error}") from None

    def get_json(self, path: str, query: Mapping[str, Any] | None = None) -> Any:
        url = self._url(path, query)
        return self._json(Request(url, headers=self._headers(include_auth=True)), f"GET {urlsplit(url).path}")

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Any:
        url = self._url(path)
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(url, data=data, method="POST", headers=self._headers(include_auth=True, json_body=True))
        return self._json(request, f"POST {urlsplit(url).path}")

    def download_bytes(self, value: str) -> tuple[bytes, str | None]:
        url = urljoin(self.connection.server_url + "/", value)
        parts = urlsplit(url)
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            raise WerkApiError("download output", _sanitized_url(url), None, "output URL must use http or https")
        if parts.username is not None or parts.password is not None:
            raise WerkApiError("download output", _sanitized_url(url), None, "output URL credentials are not allowed")
        include_auth = same_origin(self.connection.server_url, url)
        request = Request(url, headers=self._headers(include_auth=include_auth))
        body, headers = self._open(request, "download output", self.max_image_bytes)
        return body, headers.get("Content-Type")
