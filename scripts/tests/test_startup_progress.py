"""Startup progress and failure contracts, using tiny GGUFs and a fake runtime.

No model downloads, GPU work or package installation. Build werk first.
"""
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import tempfile
import time
import unittest
from urllib.request import urlopen

from test_vision_setup import BINARY, gguf


RUNTIME = r'''#!/usr/bin/env python3
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
if "--help" in sys.argv:
    time.sleep(float(os.environ.get("FIXTURE_PROBE_DELAY", "0")))
    print("--model --host --port --mmproj --parallel --ctx-size --jinja --no-warmup --reasoning --reasoning-budget --flash-attn")
    sys.exit(0)
if "--version" in sys.argv:
    print("fixture llama-server 1")
    sys.exit(0)
with open(os.environ["FIXTURE_TRACE"], "a") as trace:
    trace.write(json.dumps({"pid": os.getpid(), "args": sys.argv[1:]}) + "\n")
print("srv load_model: loading model", file=sys.stderr, flush=True)
if os.environ.get("FIXTURE_FAIL") == "1":
    print("W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user", file=sys.stderr, flush=True)
time.sleep(float(os.environ.get("FIXTURE_LOAD_DELAY", "0")))
if os.environ.get("FIXTURE_FAIL") == "1":
    print("fixture native load failure", file=sys.stderr, flush=True)
    sys.exit(42)
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        chunks = [
            {"choices":[{"delta":{"content":"one two"},"finish_reason":None}]},
            {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}
        ]
        body = ("".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port")+1])), Handler).serve_forever()
'''


@unittest.skipUnless(os.name == "posix" and BINARY.is_file(), "requires local werk build and PTY")
class StartupProgress(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="werk-startup-progress-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.store = root / "store"
        self.trace = root / "launches.jsonl"
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("WERK_", "LLAMA_ARG_"))}
        self.env.update(TERM="xterm-256color", NO_COLOR="1", FIXTURE_TRACE=str(self.trace))
        self.command = [str(BINARY), "--model-home", str(self.store)]
        source = root / "source"
        source.mkdir()
        (source / "model.gguf").write_bytes(gguf("qwen35"))
        (source / "mmproj-F16.gguf").write_bytes(gguf("clip"))
        imported = subprocess.run(self.command + ["--no-auto-install-backends", "import", str(source),
                                  "--name", "tiny", "--text-only", "--offline"],
                                  capture_output=True, env=self.env, timeout=10)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        runtime = self.store / "backends/llama-cpu/llama-server"
        runtime.parent.mkdir(parents=True)
        runtime.write_text(RUNTIME)
        runtime.chmod(0o755)
        self.env.update(WERK_LLAMA_SERVER=str(runtime), WERK_LLAMA_SERVER_CPU=str(runtime))

    def serve(self, preload=True, setup=False):
        args = ["--auto-install-backends" if setup else "--no-auto-install-backends",
                "--backend", "cpu", "serve", "--model", "tiny", "--port", "0",
                "--allow-unauthenticated", "--gpu-layers", "3", "--ctx-size", "4096",
                "--parallel", "2", "--batch-size", "256", "--ubatch-size", "128"]
        if preload:
            args += ["--preload", "tiny"]
        return args

    def capture(self, args, tty=False, ready=False, on_ready=None):
        master = slave = None
        if tty:
            import pty
            master, slave = pty.openpty()
        process = subprocess.Popen(self.command + args, stdin=subprocess.DEVNULL,
                                   stdout=slave if tty else subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=self.env)
        if slave is not None:
            os.close(slave)
        fd = master if tty else process.stdout.fileno()
        output = bytearray()
        deadline = time.monotonic() + 25
        stopped = False
        try:
            while time.monotonic() < deadline:
                if select.select([fd], [], [], 0.1)[0]:
                    try:
                        chunk = os.read(fd, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    output.extend(chunk)
                address = re.search(rb"Werk server listening at (http://127\.0\.0\.1:\d+)", output)
                if ready and not stopped and address:
                    if on_ready:
                        on_ready(address[1].decode())
                    stopped = True
                    process.send_signal(signal.SIGTERM)
            else:
                self.fail("CLI did not exit or become ready:\n" + output.decode(errors="replace"))
            return process.wait(timeout=5), output.decode(errors="replace")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if master is not None:
                os.close(master)
            if process.stdout:
                process.stdout.close()

    def launches(self):
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def assert_failed_without_restart(self, code, output):
        self.assertNotEqual(code, 0, output)
        self.assertIn("werk serve stopped", output)
        self.assertNotIn("server.started", output)
        self.assertNotIn("Server ready", output)
        self.assertEqual(len(self.launches()), 1)
        with self.assertRaises(ProcessLookupError):
            os.kill(self.launches()[0]["pid"], 0)

    def test_slow_failure_shows_spinner_and_keeps_launch_parameters(self):
        self.env.update(FIXTURE_FAIL="1", FIXTURE_LOAD_DELAY="1.5",
                        WERK_LLAMA_ARGS="--reasoning-budget 2048")
        code, output = self.capture(self.serve(), tty=True)
        self.assert_failed_without_restart(code, output)
        self.assertGreater(sum(output.count(c) for c in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"), 3, output)
        plain = " ".join(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output).replace("│", " ").split())
        self.assertIn("could not automatically fit", plain)
        self.assertIn("timeout 180s", output)
        args = self.launches()[0]["args"]
        for flag, value in [("-ngl", "3"), ("-c", "8192"), ("-np", "2"),
                            ("-b", "256"), ("-ub", "128"), ("--reasoning-budget", "2048")]:
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_json_logs_have_periodic_status_during_silent_load(self):
        self.env.update(FIXTURE_FAIL="1", FIXTURE_LOAD_DELAY="5.5")
        code, output = self.capture(["--log-format", "json"] + self.serve())
        self.assert_failed_without_restart(code, output)
        self.assertNotIn("\x1b", output)
        events = [json.loads(line) for line in output.splitlines()]
        progress = [e for e in events if e["event"] == "operation.progress"]
        self.assertTrue(progress, output)
        self.assertTrue(any(e["fields"].get("duration_seconds", 0) >= 5 for e in progress), output)

    def test_failure_after_vision_setup_exits_and_retains_installed_assets(self):
        self.env["FIXTURE_FAIL"] = "1"
        code, output = self.capture(self.serve(setup=True))
        self.assert_failed_without_restart(code, output)
        self.assertIn("Model setup complete", output)
        manifest = json.loads((self.store / "models/tiny/manifest.json").read_text())
        self.assertTrue(manifest["vision_setup"]["enabled"])
        self.assertIn("--mmproj", self.launches()[0]["args"])

    def test_default_model_failure_without_preload_exits(self):
        self.env["FIXTURE_FAIL"] = "1"
        code, output = self.capture(self.serve(preload=False))
        self.assert_failed_without_restart(code, output)

    def test_failed_preload_does_not_start_later_models(self):
        imported = subprocess.run(self.command + ["--no-auto-install-backends", "import",
                                  str(Path(self.temp.name) / "source"), "--name", "second",
                                  "--text-only", "--offline"],
                                  capture_output=True, env=self.env, timeout=10)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        self.env["FIXTURE_FAIL"] = "1"
        code, output = self.capture(self.serve() + ["second"])
        self.assert_failed_without_restart(code, output)
        self.assertNotIn("Preparing preload target 'second'", output)

    def test_backend_install_failure_stops_serve_before_native_launch(self):
        runtime = Path(self.env["WERK_LLAMA_SERVER"])
        runtime.write_text(RUNTIME.replace("--mmproj ", ""))
        commands = Path(self.temp.name) / "commands"
        commands.mkdir()
        git = commands / "git"
        git.write_text("#!/bin/sh\necho 'fixture installation failed' >&2\nexit 7\n")
        git.chmod(0o755)
        self.env["PATH"] = str(commands) + os.pathsep + self.env["PATH"]
        code, output = self.capture(self.serve(setup=True))
        self.assertNotEqual(code, 0, output)
        self.assertIn("Backend installation failed", output)
        self.assertIn("werk serve stopped", output)
        self.assertNotIn("server.started", output)
        self.assertFalse(self.trace.exists(), output)

    def test_successful_preload_starts_request_service(self):
        limits = []
        def inspect_capacity(url):
            with urlopen(url + "/werk/v1/chat/limits?model=tiny", timeout=3) as response:
                limits.append(json.load(response)["data"])
        code, output = self.capture(self.serve(), ready=True, on_ready=inspect_capacity)
        self.assertIn("serve.preload.ready", output)
        self.assertIn("server.started", output)
        self.assertNotIn("startup failed", output)
        self.assertLess(output.index("serve.preload.ready"), output.index("server.started"))
        self.assertEqual(len(self.launches()), 1)
        self.assertEqual(limits[0]["max_concurrent_requests"], 2)
        self.assertEqual(limits[0]["context_tokens"], 4096)

    def test_backend_install_has_status_and_spinner(self):
        self.env["FIXTURE_PROBE_DELAY"] = "0.5"
        code, output = self.capture(["backend", "install", "llama-cpu", "--vision"], tty=True)
        self.assertEqual(code, 0, output)
        self.assertIn("Installing backend llama-cpu", output)
        self.assertIn("completed", output)
        self.assertTrue(any(c in output for c in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"), output)


if __name__ == "__main__":
    unittest.main()
