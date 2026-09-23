"""Exercise the built CLI against a separate empty CPU server, never live inference."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--binary", default="target/debug/werk.exe" if os.name == "nt" else "target/debug/werk")
args = parser.parse_args()
binary = str(Path(args.binary).resolve())
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
url = f"http://127.0.0.1:{port}"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
env = os.environ.copy()
env["WERK_API_KEY"] = "observability-smoke-fixture"
with tempfile.TemporaryDirectory(prefix="werk-observability-smoke-") as directory:
    log_path = Path(directory) / "server.log"
    with log_path.open("wb") as log:
        server = subprocess.Popen(
            [binary, "--model-home", directory, "--device", "cpu",
             "--no-auto-install-backends", "serve", "--host", "127.0.0.1", "--port", str(port)],
            stdout=log, stderr=log, env=env,
        )
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    request = urllib.request.Request(url + "/metrics", headers={"Authorization": "Bearer " + env["WERK_API_KEY"]})
                    with opener.open(request, timeout=2) as response:
                        metrics = response.read().decode()
                    break
                except (urllib.error.URLError, TimeoutError):
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("isolated test server did not start: " + log_path.read_text(errors="replace")[-2000:])
                    time.sleep(0.1)
            assert "werk_requests_active 0" in metrics
            assert "werk_uptime_seconds" in metrics
            try:
                opener.open(url + "/metrics", timeout=2)
            except urllib.error.HTTPError as error:
                assert error.code == 401
            else:
                raise AssertionError("metrics accepted an unauthenticated request")
            result = subprocess.run([binary, "top", "--url", url, "--once", "--json"], env=env, capture_output=True, text=True, timeout=10, check=True)
            snapshot = json.loads(result.stdout)
            assert snapshot["schema_version"] == 1
            assert snapshot["totals"]["started"] == 0
            assert snapshot["backends"] == []
            assert snapshot["requests"] == []
            print("Passed: isolated CPU server, authenticated metrics, werk top --once --json; no model loaded")
        finally:
            # Only the child created above is signalled. Never discover/stop other processes.
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
