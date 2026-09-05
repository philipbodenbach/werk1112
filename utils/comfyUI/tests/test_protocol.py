import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ..config import WerkConnection
from ..protocol import (
    CAPABILITY_STATUSES,
    PROTOCOL_VERSION_HEADER,
    WerkProtocolClient,
    WerkProtocolError,
)


class Server:
    def __init__(self, responder):
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.requests.append((self.path, dict(self.headers), None))
                responder(self, owner)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                owner.requests.append((self.path, dict(self.headers), body))
                responder(self, owner)

            def log_message(self, *_args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()


@pytest.fixture
def servers():
    values = []

    def create(responder):
        server = Server(responder)
        values.append(server)
        return server

    yield create
    for server in values:
        server.close()


def send(handler, status=200, payload=None, body=None, headers=None):
    if body is None:
        body = json.dumps(payload or {}).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    for key, value in (headers or {}).items():
        handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(body)


def envelope(data, *, major=1, minor=0, request_id="req_test"):
    return {
        "protocol": {"major": major, "minor": minor},
        "request_id": request_id,
        "data": data,
    }


def test_info_and_all_six_capability_statuses_are_validated(servers):
    statuses = sorted(CAPABILITY_STATUSES)

    def responder(handler, _owner):
        if handler.path == "/werk/v1/info":
            send(
                handler,
                payload=envelope(
                    {
                        "service": "werk1112",
                        "service_version": "1.5.1",
                        "protocol": {"major": 1, "minor": 0},
                        "active_backend": "llama.cpp",
                        "limits": {
                            "max_page_size": 100,
                            "max_state_ids_per_operation": 100,
                            "max_expert_ids_per_operation": 256,
                            "max_request_bytes": 1048576,
                            "max_handoff_bytes": 4096,
                            "max_ttl_seconds": 2592000,
                        },
                    }
                ),
                headers={PROTOCOL_VERSION_HEADER: "1.0"},
            )
            return
        send(
            handler,
            payload=envelope(
                {
                    "capabilities": [
                        {
                            "id": f"runtime.test.{status}",
                            "status": status,
                            "detail": status,
                            "operations": ["read"],
                        }
                        for status in statuses
                    ]
                }
            ),
            headers={PROTOCOL_VERSION_HEADER: "1.0"},
        )

    server = servers(responder)
    client = WerkProtocolClient(WerkConnection(server.url, "api-secret"))
    assert client.info()["active_backend"] == "llama.cpp"
    capabilities = client.capabilities()["capabilities"]
    assert sorted(item["status"] for item in capabilities) == statuses
    assert all(
        request[1].get("Authorization") == "Bearer api-secret"
        for request in server.requests
    )
    assert all(
        request[1].get("Accept") == "application/json"
        and request[1].get(PROTOCOL_VERSION_HEADER) == "1.0"
        for request in server.requests
    )
    assert "api-secret" not in repr(client)


def test_success_envelope_rejects_newer_minor_and_wrong_major(servers):
    for major, minor in ((1, 1), (2, 0)):
        server = servers(
            lambda handler, _owner, major=major, minor=minor: send(
                handler,
                payload=envelope({}, major=major, minor=minor),
            )
        )
        with pytest.raises(WerkProtocolError, match="incompatible Werk Protocol"):
            WerkProtocolClient(WerkConnection(server.url)).info()


def test_typed_incompatible_protocol_error_is_preserved(servers):
    server = servers(
        lambda handler, _owner: send(
            handler,
            status=406,
            payload={
                "protocol": {"major": 1, "minor": 0},
                "request_id": "req_version",
                "error": {
                    "code": "incompatible_protocol",
                    "message": "the requested protocol version is not supported",
                    "retryable": False,
                },
            },
            headers={PROTOCOL_VERSION_HEADER: "1.0"},
        )
    )
    with pytest.raises(WerkProtocolError) as captured:
        WerkProtocolClient(WerkConnection(server.url)).info()
    assert captured.value.status_code == 406
    assert captured.value.code == "incompatible_protocol"


@pytest.mark.parametrize(
    ("header", "expected_code"),
    [
        ("1.1", "incompatible_protocol"),
        ("not-a-version", "invalid_response"),
    ],
)
def test_success_response_protocol_header_is_validated(
    servers, header, expected_code
):
    server = servers(
        lambda handler, _owner: send(
            handler,
            payload=envelope({"capabilities": []}),
            headers={PROTOCOL_VERSION_HEADER: header},
        )
    )
    with pytest.raises(WerkProtocolError) as captured:
        WerkProtocolClient(WerkConnection(server.url)).capabilities()
    assert captured.value.code == expected_code


def test_error_response_protocol_header_must_match_a_supported_envelope(servers):
    server = servers(
        lambda handler, _owner: send(
            handler,
            status=503,
            payload={
                "protocol": {"major": 1, "minor": 0},
                "request_id": "req_unavailable",
                "error": {
                    "code": "unavailable",
                    "message": "not ready",
                    "retryable": True,
                },
            },
            headers={PROTOCOL_VERSION_HEADER: "2.0"},
        )
    )
    with pytest.raises(WerkProtocolError) as captured:
        WerkProtocolClient(WerkConnection(server.url)).info()
    assert captured.value.status_code == 503
    assert captured.value.code == "incompatible_protocol"


def test_response_protocol_header_remains_optional_for_1_0_servers(servers):
    server = servers(
        lambda handler, _owner: send(
            handler, payload=envelope({"capabilities": []})
        )
    )
    assert WerkProtocolClient(WerkConnection(server.url)).capabilities() == {
        "capabilities": []
    }


def test_typed_error_preserves_fields_and_redacts_api_key_and_handoff(servers):
    handoff = "handoff-secret-" + "x" * 32

    def responder(handler, _owner):
        send(
            handler,
            status=410,
            payload={
                "protocol": {"major": 1, "minor": 0},
                "request_id": "req_expired",
                "error": {
                    "code": "expired_handoff",
                    "message": f"expired {handoff} api-secret",
                    "retryable": False,
                },
            },
        )

    server = servers(responder)
    client = WerkProtocolClient(WerkConnection(server.url, "api-secret"))
    with pytest.raises(WerkProtocolError) as captured:
        client.decode(
            {"handoff": handoff, "max_tokens": 1}, handoff_secret=handoff
        )
    error = captured.value
    assert error.status_code == 410
    assert error.code == "expired_handoff"
    assert error.request_id == "req_expired"
    assert error.retryable is False
    assert handoff not in str(error)
    assert "api-secret" not in repr(error)
    assert str(error).count("[redacted]") == 2


def test_typed_error_drops_a_request_id_that_reflects_a_secret(servers):
    handoff = "reflected-handoff-" + "x" * 32

    def responder(handler, _owner):
        send(
            handler,
            status=410,
            payload={
                "protocol": {"major": 1, "minor": 0},
                "request_id": handoff,
                "error": {
                    "code": "expired_handoff",
                    "message": "expired",
                    "retryable": False,
                },
            },
        )

    server = servers(responder)
    client = WerkProtocolClient(WerkConnection(server.url, "api-secret"))
    with pytest.raises(WerkProtocolError) as captured:
        client.decode(
            {"handoff": handoff, "max_tokens": 1}, handoff_secret=handoff
        )
    assert captured.value.request_id is None
    assert handoff not in str(captured.value)


def test_runtime_transport_never_follows_redirects(servers):
    target = servers(
        lambda handler, _owner: send(handler, payload=envelope({"stolen": True}))
    )

    def redirect(handler, _owner):
        send(
            handler,
            status=302,
            body=b"",
            headers={"Location": target.url + "/stolen"},
        )

    werk = servers(redirect)
    with pytest.raises(WerkProtocolError) as captured:
        WerkProtocolClient(WerkConnection(werk.url, "secret")).info()
    assert captured.value.code == "redirect_rejected"
    assert target.requests == []


def test_runtime_transport_bounds_responses_and_requests(servers):
    server = servers(lambda handler, _owner: send(handler, body=b"x" * 64))
    with pytest.raises(WerkProtocolError, match="exceeds 32 bytes"):
        WerkProtocolClient(
            WerkConnection(server.url), max_response_bytes=32
        ).info()

    client = WerkProtocolClient(
        WerkConnection(server.url), max_request_bytes=16
    )
    with pytest.raises(WerkProtocolError, match="request exceeds 16 bytes"):
        client.prune_states(
            {"selector": {"kind": "ids", "ids": ["state_long"]}}
        )


def test_old_or_malformed_servers_fail_closed(servers):
    old = servers(
        lambda handler, _owner: send(
            handler,
            status=404,
            payload={"error": {"message": "not found"}},
        )
    )
    with pytest.raises(WerkProtocolError) as captured:
        WerkProtocolClient(WerkConnection(old.url)).info()
    assert captured.value.code == "invalid_error_envelope"

    malformed = servers(
        lambda handler, _owner: send(
            handler,
            payload={"protocol": {"major": 1, "minor": 0}, "data": {}},
        )
    )
    with pytest.raises(WerkProtocolError, match="request_id"):
        WerkProtocolClient(WerkConnection(malformed.url)).info()


def test_unknown_capability_status_is_rejected(servers):
    server = servers(
        lambda handler, _owner: send(
            handler,
            payload=envelope(
                {
                    "capabilities": [
                        {
                            "id": "runtime.future",
                            "status": "maybe",
                            "detail": "unknown",
                            "operations": [],
                        }
                    ]
                }
            ),
        )
    )
    with pytest.raises(ValueError, match="unknown value"):
        WerkProtocolClient(WerkConnection(server.url)).capabilities()


def expert(expert_id="expert_1", model_id="private/model"):
    return {
        "id": expert_id,
        "model_id": model_id,
        "tier": "vram",
        "bytes": 4096,
        "hotness": 2.5,
        "pinned": False,
        "last_used_unix_ms": 123,
    }


def test_expert_list_and_action_use_versioned_bounded_typed_contract(servers):
    def responder(handler, _owner):
        if handler.path.startswith("/werk/v1/experts?"):
            send(
                handler,
                payload=envelope(
                    {"experts": [expert()], "next_cursor": "cursor_2"}
                ),
            )
            return
        request = json.loads(_owner.requests[-1][2])
        assert request == {
            "model_id": "private/model",
            "expert_ids": ["expert_1"],
            "action": "pin",
            "dry_run": True,
            "allow_experimental": False,
        }
        send(
            handler,
            payload=envelope(
                {
                    "experts": [{**expert(), "pinned": True}],
                    "changed": 1,
                    "dry_run": True,
                }
            ),
        )

    server = servers(responder)
    client = WerkProtocolClient(WerkConnection(server.url, "api-secret"))
    listed = client.experts(
        {
            "model_id": "private/model",
            "tier": "vram",
            "limit": 1,
            "cursor": "cursor_1",
            "allow_experimental": False,
        }
    )
    assert listed == {"experts": [expert()], "next_cursor": "cursor_2"}
    assert server.requests[0][0] == (
        "/werk/v1/experts?model_id=private%2Fmodel&tier=vram&limit=1"
        "&cursor=cursor_1&allow_experimental=false"
    )
    result = client.expert_action(
        {
            "model_id": "private/model",
            "expert_ids": ["expert_1"],
            "action": "pin",
            "dry_run": True,
            "allow_experimental": False,
        }
    )
    assert result["experts"][0]["pinned"] is True
    assert result["changed"] == 1 and result["dry_run"] is True

    with pytest.raises(TypeError, match="must be a boolean"):
        client.experts({"limit": 1, "allow_experimental": "false"})


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            {"experts": [{**expert(), "hotness": float("nan")}], "next_cursor": None},
            "finite number",
        ),
        (
            {"experts": [expert("../escape")], "next_cursor": None},
            "valid opaque expert ID",
        ),
        (
            {"experts": [expert(), expert()], "next_cursor": None},
            "duplicate expert identity",
        ),
    ],
)
def test_expert_list_rejects_malformed_or_ambiguous_responses(
    servers, response, message
):
    server = servers(
        lambda handler, _owner: send(handler, payload=envelope(response))
    )
    with pytest.raises(ValueError, match=message):
        WerkProtocolClient(WerkConnection(server.url)).experts({"limit": 10})


def test_expert_action_validates_selection_and_redacts_identifiers(servers):
    model_id = "private/model"
    expert_id = "expert_secret"

    def mismatched(handler, _owner):
        send(
            handler,
            payload=envelope(
                {
                    "experts": [expert("other", model_id)],
                    "changed": 1,
                    "dry_run": False,
                }
            ),
        )

    mismatch_server = servers(mismatched)
    client = WerkProtocolClient(WerkConnection(mismatch_server.url))
    with pytest.raises(ValueError, match="explicit selection"):
        client.expert_action(
            {
                "model_id": model_id,
                "expert_ids": [expert_id],
                "action": "pin",
                "dry_run": False,
                "allow_experimental": False,
            }
        )

    def rejected(handler, _owner):
        send(
            handler,
            status=400,
            payload={
                "protocol": {"major": 1, "minor": 0},
                "request_id": "req_expert",
                "error": {
                    "code": "invalid_request",
                    "message": f"rejected {model_id} {expert_id} api-secret",
                    "retryable": False,
                },
            },
        )

    error_server = servers(rejected)
    protected = WerkProtocolClient(WerkConnection(error_server.url, "api-secret"))
    with pytest.raises(WerkProtocolError) as captured:
        protected.expert_action(
            {
                "model_id": model_id,
                "expert_ids": [expert_id],
                "action": "pin",
                "dry_run": True,
                "allow_experimental": False,
            }
        )
    rendered = str(captured.value)
    assert model_id not in rendered
    assert expert_id not in rendered
    assert "api-secret" not in rendered
