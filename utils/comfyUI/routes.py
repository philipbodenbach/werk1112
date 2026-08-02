"""Local ComfyUI routes used by the Werk1112 frontend widgets.

The browser talks only to its own ComfyUI origin. ComfyUI then performs Werk
discovery with the same credential-safe client used by the execution nodes.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping

try:
    from .client import WerkApiError, WerkClient
    from .config import WerkConnection
    from .nodes import classify_image_models
except ImportError:  # pragma: no cover - direct-module development
    from client import WerkApiError, WerkClient
    from config import WerkConnection
    from nodes import classify_image_models


def _connection_from_payload(payload: Mapping[str, Any]) -> WerkConnection:
    server_url = payload.get("server_url", "")
    api_key = payload.get("api_key", "")
    timeout_seconds = payload.get("timeout_seconds", 900)
    verify_tls = payload.get("verify_tls", True)
    if not isinstance(server_url, str):
        raise ValueError("server_url must be a string")
    if not isinstance(api_key, str):
        raise ValueError("api_key must be a string")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise ValueError("timeout_seconds must be an integer")
    if not isinstance(verify_tls, bool):
        raise ValueError("verify_tls must be a boolean")
    return WerkConnection(server_url, api_key, timeout_seconds, verify_tls)


def discover_connection(
    payload: Mapping[str, Any],
    *,
    client_factory: Callable[[WerkConnection], WerkClient] = WerkClient,
) -> dict[str, Any]:
    """Verify one connection and return safe model discovery information."""

    if not isinstance(payload, Mapping):
        raise ValueError("request body must be a JSON object")
    connection = _connection_from_payload(payload)
    client = client_factory(connection)
    models = client.get_json("/v1/models")
    capabilities: Any = {}
    warning = None
    try:
        capabilities = client.get_json("/v1/capabilities")
    except WerkApiError as error:
        warning = str(error)
    classification = classify_image_models(models, capabilities)
    model_count = len(classification["installed"])
    available_count = len(classification["available"])
    model_word = "model" if model_count == 1 else "models"
    image_word = "image" if available_count == 1 else "images"
    status = f"Connected · {model_count} {model_word} · {available_count} {image_word}"
    response = {
        "ok": True,
        "status": status,
        "server_url": connection.server_url,
        "authentication_configured": bool(connection.api_key),
        "verify_tls": connection.verify_tls,
        "models": classification["installed"],
        "image_models": {
            "declared": classification["declared"],
            "available": classification["available"],
        },
    }
    if warning:
        response["warning"] = warning
    return response


_ROUTES_REGISTERED = False


def register_routes() -> bool:
    """Register ComfyUI routes when imported by a running ComfyUI server."""

    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return False

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return False

    @prompt_server.routes.post("/werk1112/verify")
    async def verify_werk_connection(request):  # noqa: ANN001
        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"ok": False, "status": "Connection failed", "error": "request body must be valid JSON"},
                status=400,
            )
        try:
            result = await asyncio.to_thread(discover_connection, payload)
            return web.json_response(result)
        except ValueError as error:
            return web.json_response(
                {"ok": False, "status": "Connection failed", "error": str(error)},
                status=400,
            )
        except WerkApiError as error:
            return web.json_response(
                {"ok": False, "status": "Connection failed", "error": str(error)},
                status=502,
            )
        except Exception:
            return web.json_response(
                {
                    "ok": False,
                    "status": "Connection failed",
                    "error": "unexpected connection verification failure",
                },
                status=500,
            )

    _ROUTES_REGISTERED = True
    return True
