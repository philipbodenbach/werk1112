import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ..client import WerkApiError, WerkClient, same_origin
from ..config import WerkConnection, normalize_server_url


class Server:
    def __init__(self, responder):
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.requests.append((self.path, dict(self.headers)))
                responder(self, owner)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                owner.requests.append((self.path, dict(self.headers), body))
                responder(self, owner)

            def do_DELETE(self):
                owner.requests.append((self.path, dict(self.headers)))
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


def send(handler, status=200, body=b"{}", content_type="application/json", headers=None):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    for key, value in (headers or {}).items():
        handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(body)


def test_url_normalization_and_same_origin():
    assert normalize_server_url(" HTTPS://Example.COM:443/base/// ") == "https://Example.COM:443/base"
    assert same_origin("https://example.com", "https://EXAMPLE.com:443/v1/output")


@pytest.mark.parametrize("url", ["ftp://example.com", "file:///tmp/socket", "example.com"])
def test_unsupported_or_missing_url_scheme_rejected(url):
    with pytest.raises(ValueError, match="http or https"):
        normalize_server_url(url)


def test_embedded_credentials_rejected():
    with pytest.raises(ValueError, match="credentials"):
        normalize_server_url("http://user:secret@example.com")


def test_connection_repr_and_status_hide_api_key():
    connection = WerkConnection("http://localhost:11434", "super-secret")
    assert "super-secret" not in repr(connection)
    assert "super-secret" not in connection.safe_status


def test_bearer_header_and_json_get(servers):
    def responder(handler, _owner):
        send(handler, body=b'{"data":[]}')

    server = servers(responder)
    payload = WerkClient(WerkConnection(server.url, "secret")).get_json("/v1/models")
    assert payload == {"data": []}
    assert server.requests[0][1]["Authorization"] == "Bearer secret"


def test_json_post(servers):
    def responder(handler, _owner):
        send(handler, body=b'{"ok":true}')

    server = servers(responder)
    assert WerkClient(WerkConnection(server.url)).post_json("/v1/images/generations", {"n": 2}) == {"ok": True}
    assert json.loads(server.requests[0][2]) == {"n": 2}


def test_authenticated_json_delete(servers):
    server = servers(lambda handler, _owner: send(handler, body=b'{"status":"cancelled"}'))
    payload = WerkClient(WerkConnection(server.url, "secret")).delete_json("/v1/jobs/job-1")
    assert payload == {"status": "cancelled"}
    assert server.requests[0][0] == "/v1/jobs/job-1"
    assert server.requests[0][1]["Authorization"] == "Bearer secret"


def test_api_key_absent_from_401_and_500_errors(servers):
    for status in (401, 500):
        def responder(handler, _owner, status=status):
            send(handler, status=status, body=b'{"error":{"message":"safe failure"}}')

        server = servers(responder)
        with pytest.raises(WerkApiError) as captured:
            WerkClient(WerkConnection(server.url, "never-print-this")).get_json("/v1/models")
        assert captured.value.status_code == status
        assert "safe failure" in str(captured.value)
        assert "never-print-this" not in str(captured.value)
        assert "never-print-this" not in repr(captured.value)


def test_api_key_is_redacted_when_server_echoes_it(servers):
    server = servers(
        lambda handler, _owner: send(
            handler,
            status=401,
            body=b'{"error":{"message":"rejected echoed-secret"}}',
        )
    )
    with pytest.raises(WerkApiError) as captured:
        WerkClient(WerkConnection(server.url, "echoed-secret")).get_json("/v1/models")
    assert "echoed-secret" not in str(captured.value)
    assert "[redacted]" in str(captured.value)


def test_invalid_json_error_is_safe(servers):
    server = servers(lambda handler, _owner: send(handler, body=b"not-json"))
    with pytest.raises(WerkApiError, match="invalid JSON"):
        WerkClient(WerkConnection(server.url)).get_json("/v1/models")


def test_bounded_response_protection(servers):
    server = servers(lambda handler, _owner: send(handler, body=b"x" * 32))
    with pytest.raises(WerkApiError, match="exceeds 8 bytes"):
        WerkClient(WerkConnection(server.url), max_json_bytes=8).get_json("/v1/models")


def test_relative_output_download_reuses_auth_on_werk_origin(servers):
    server = servers(lambda handler, _owner: send(handler, body=b"png", content_type="image/png"))
    body, mime = WerkClient(WerkConnection(server.url, "secret")).download_bytes("/v1/outputs/id")
    assert body == b"png"
    assert mime == "image/png"
    assert server.requests[0][1]["Authorization"] == "Bearer secret"


def test_output_download_accepts_a_call_specific_byte_limit(servers):
    server = servers(lambda handler, _owner: send(handler, body=b"video"))
    client = WerkClient(WerkConnection(server.url))
    with pytest.raises(WerkApiError, match="exceeds 4 bytes"):
        client.download_bytes("/v1/outputs/video", max_bytes=4)


def test_cross_origin_download_never_receives_auth(servers):
    target = servers(lambda handler, _owner: send(handler, body=b"png", content_type="image/png"))
    werk = servers(lambda handler, _owner: send(handler))
    WerkClient(WerkConnection(werk.url, "secret")).download_bytes(target.url + "/image.png")
    assert "Authorization" not in target.requests[0][1]


def test_cross_origin_redirect_is_rejected_without_contacting_target(servers):
    target = servers(lambda handler, _owner: send(handler, body=b"should-not-run"))

    def redirect(handler, _owner):
        send(handler, status=302, body=b"", headers={"Location": target.url + "/stolen"})

    werk = servers(redirect)
    with pytest.raises(WerkApiError, match="cross-origin redirect rejected"):
        WerkClient(WerkConnection(werk.url, "secret")).download_bytes("/v1/outputs/id")
    assert target.requests == []
