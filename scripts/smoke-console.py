#!/usr/bin/env python3
"""Exercise the real CLI in pipes and Unix PTYs, without downloading models.

Usage: python3 scripts/smoke-console.py [target/debug/werk]
Uses an isolated temporary store and a stub installer; no real packages installed.
"""
import errno
import fcntl
import http.server
import json
import os
from pathlib import Path
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import urllib.request

SGR = re.compile(r"\x1b\[[0-9;]*m")
BINARY = str(Path(sys.argv[1] if len(sys.argv) > 1 else "target/debug/werk").resolve())


def read_pty(fd, process, timeout=40):
    chunks, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], .1)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        except OSError as error:
            if error.errno == errno.EIO:
                break
            raise
    else:
        process.kill()
        process.wait()
        raise AssertionError("CLI did not exit within timeout")
    return b"".join(chunks).decode().replace("\r\n", "\n")


def run(args, env, terminal=False, width=100, code=0):
    if terminal:
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, width, 0, 0))
        process = subprocess.Popen([BINARY, *args], stdin=subprocess.DEVNULL,
                                   stdout=slave, stderr=slave, env=env)
        os.close(slave)
        try:
            out = read_pty(master, process)
            actual = process.wait(timeout=3)
        finally:
            os.close(master)
            if process.poll() is None:
                process.kill()
                process.wait()
    else:
        process = subprocess.run([BINARY, *args], capture_output=True, text=True,
                                 stdin=subprocess.DEVNULL, env=env, timeout=40)
        actual, out = process.returncode, process.stdout + process.stderr
    assert actual == code, (args, actual, out)
    return out


class Chat(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/chat/completions" and body["stream"]
        events = [
            {"choices": [{"delta": {"content": "Hello λ"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
        ]
        payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload.encode())))
        self.end_headers()
        self.wfile.write(payload.encode())


def main():
    with tempfile.TemporaryDirectory(prefix="werk-console-") as temporary:
        root = Path(temporary)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("WERK_", "NO_COLOR", "CLICOLOR", "FORCE_COLOR"))}
        env.update(WERK_HOME=str(root / "store"), TERM="xterm-256color")
        for args in (["--help"], ["serve", "--help"], ["backend", "install", "--help"],
                     ["list"], ["cache", "list"], ["temp", "list"], ["backend", "list"]):
            rich = run(args, env, terminal=True)
            plain = run(args, env)
            assert SGR.search(rich), (args, rich)
            assert "\x1b" not in plain, (args, plain)
        print("Help, model/backend/cache/temp views: colored PTY and clean pipes OK")

        for args in (["list", "--json"], ["cache", "list", "--json"],
                     ["parameters", "--task", "text-generation", "--example"]):
            for tty in (False, True):
                data = run(args, env, terminal=tty)
                json.loads(data)
                assert "\x1b" not in data and "╭" not in data
        for tty in (False, True):
            assert run(["temp", "path"], env, terminal=tty).strip() == str(root / "store" / "tmp")
        print("JSON, parameter examples and path output remain unadorned in PTYs too")

        for extra in ({"NO_COLOR": "1"}, {"TERM": "dumb"}):
            for args in (["list"], ["--help"], ["serve", "--model", "missing", "--allow-unauthenticated"]):
                output = run(args, dict(env, **extra), terminal=True, code=1 if args[0] == "serve" else 0)
                assert not SGR.search(output), (args, extra, output)
        narrow = run(["serve", "--model", "missing", "--allow-unauthenticated"], env,
                     terminal=True, width=40, code=1)
        assert "Inference Router" in narrow and "could not complete" in narrow
        print("NO_COLOR, TERM=dumb, narrow terminal and error exit status OK")

        source = root / "checkpoint"
        source.mkdir()
        (source / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))
        (source / "model.safetensors").write_bytes(b"fixture")
        run(["import", str(source), "--name", "console/chat"], env)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Chat)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            args = ["run", "console/chat", "hello", "--server", f"http://127.0.0.1:{server.server_port}"]
            assert run(args, env).strip() == "Hello λ"
            assert run(args + ["--stream"], env).strip() == "Hello λ"
            for tty in (False, True):
                output = run(args + ["--json", "--stream"], env, terminal=tty)
                values = [json.loads(line) for line in output.splitlines() if line.strip()]
                assert values[-1]["type"] == "completion"
                assert values[-1]["message"]["content"] == "Hello λ"
                assert "\x1b" not in output
        finally:
            server.shutdown()
            server.server_close()
        print("Real streamed CLI path: content and JSONL preserved in pipes and PTYs")

        (source / "config.json").write_text(json.dumps({"model_type": "embedding_gemma2"}))
        run(["import", str(source), "--name", "console/embed"], env)
        missing = dict(env, WERK_TEXT_PYTHON=str(root / "missing-python"),
                       WERK_VLLM_PYTHON=str(root / "missing-python"))
        output = run(["serve", "--port", "0", "--model", "console/embed", "--allow-unauthenticated"],
                     missing, terminal=True, code=1)
        assert "Backend setup required" in output
        assert "\x1b[1;38;2;0;220;255mwerk backend install text-analysis\x1b[0m" in output
        print("Missing dependencies: actionable panel with the logo's cyan command OK")

        python = root / "store/backends/text-analysis/venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\nprintf 'Installer stdout\\n'\nprintf 'Installer stderr\\n' >&2\nexit ${WERK_CONSOLE_STUB_EXIT:-0}\n")
        python.chmod(0o755)
        output = run(["backend", "install", "text-analysis"], env, terminal=True)
        assert "Installer stdout" in output and "Installer stderr" in output and "╰─ Done" in output
        failed = run(["backend", "install", "text-analysis"], dict(env, WERK_CONSOLE_STUB_EXIT="7"), terminal=True, code=1)
        assert "dependency installation failed" in failed
        print("Installer stdout/stderr are streamed and failures retain a nonzero exit")

        master, slave = pty.openpty()
        process = subprocess.Popen([BINARY, "serve", "--port", "0", "--api-key", "console-test-key"],
                                   stdin=subprocess.DEVNULL, stdout=slave, stderr=slave, env=env)
        os.close(slave)
        output = b""
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if select.select([master], [], [], .1)[0]:
                    output += os.read(master, 65536)
                    match = re.search(rb"http://127\.0\.0\.1:(\d+)", output)
                    if match:
                        break
            else:
                raise AssertionError("serve did not report its address")
            url = "http://127.0.0.1:" + match[1].decode()
            request = urllib.request.Request(url + "/v1/models", headers={"Authorization": "Bearer console-test-key"})
            with urllib.request.urlopen(request, timeout=5) as response:
                json.load(response)
            time.sleep(.2)
            process.send_signal(signal.SIGINT)
            output += read_pty(master, process).encode()
            assert process.wait(timeout=5) == 130
            text = output.decode()
            assert "Server ready" in text and "GET /v1/models" in text and "Stopped." in text
            assert "console-test-key" not in text
        finally:
            os.close(master)
            if process.poll() is None:
                process.kill()
                process.wait()
        print("Serve panel, live request status, authentication and Ctrl+C cleanup OK")


if __name__ == "__main__":
    main()
