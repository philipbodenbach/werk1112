#!/usr/bin/env python3
"""Exercise operational logs in pipes and PTYs without downloading models.

Usage: python3 scripts/smoke-logging.py [target/debug/werk]
Uses an isolated store, ephemeral HTTP port and disposable log files.
"""
import collections
import json
import os
from pathlib import Path
import pty
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

BINARY = str(Path(sys.argv[1] if len(sys.argv) > 1 else "target/debug/werk").resolve())
KEY = "logging-smoke-secret-credential"


def events(text):
    assert "\x1b" not in text, text
    return [json.loads(line) for line in text.splitlines()]


def request(url, path, auth=True, payload=None):
    headers = {"Authorization": "Bearer " + KEY} if auth else {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url + path, data=payload, headers=headers)
    try:
        response = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, response.headers, response.read()


def serve(root, env, terminal):
    log = root / ("tty.jsonl" if terminal else "pipe.jsonl")
    args = [BINARY, "serve", "--port", "0", "--api-key", KEY,
            "--verbose-lite" if terminal else "--verbose-pure", "--log-file", str(log)]
    captured = bytearray()
    if terminal:
        master, slave = pty.openpty()
        process = subprocess.Popen(args, stdout=slave, stderr=slave, stdin=subprocess.DEVNULL, env=env)
        os.close(slave)
        def drain():
            while True:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                captured.extend(chunk)
        reader = threading.Thread(target=drain)
        reader.start()
    else:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   stdin=subprocess.DEVNULL, env=env)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            assert process.poll() is None, "serve exited during startup"
            try:
                started = next(e for e in events(log.read_text()) if e["event"] == "server.started")
                break
            except (FileNotFoundError, StopIteration, json.JSONDecodeError):
                time.sleep(.05)
        else:
            raise AssertionError("server did not start")
        url = "http://" + started["fields"]["address"]
        ids = []
        for path, auth, payload, expected in [
            ("/v1/models?secret=PRIVATE_QUERY", False, None, 401),
            ("/v1/models", True, None, 200),
            ("/PRIVATE_PATH", True, None, 404),
            ("/v1/classifications", True, b'{"PRIVATE_BODY":', 400),
        ]:
            status, headers, body = request(url, path, auth, payload)
            assert status == expected, (status, body)
            ids.append(headers["x-request-id"])
        assert len(set(ids)) == len(ids)
        status, _, metrics = request(url, "/metrics")
        assert status == 200
        for metric in [b"werk_log_records_dropped_total 0", b"werk_log_write_errors_total 0",
                       b"werk_requests_started_total"]:
            assert metric in metrics, metric
        process.send_signal(signal.SIGTERM)
        if terminal:
            assert process.wait(timeout=10) == 143
            reader.join(timeout=5)
            assert not reader.is_alive()
            raw = captured.decode().replace("\r\n", "\n")
        else:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 143
            assert stdout == b"", stdout
            raw = stderr.decode()
        file_raw = log.read_text()
        console_events, file_events = events(raw), events(file_raw)
        normalized = lambda rows: collections.Counter(json.dumps(row, sort_keys=True) for row in rows)
        assert normalized(console_events) == normalized(file_events)
        assert any(e["event"] == "process.stopped" for e in file_events)
        for request_id in ids:
            completed = [e for e in file_events if e.get("request_id") == request_id and e["event"] == "http.request.completed"]
            assert len(completed) == 1, completed
            assert completed[0]["level"] == ("info" if completed[0]["fields"]["status"] == 200 else "warn")
        for private in [KEY, "PRIVATE_QUERY", "PRIVATE_PATH", "PRIVATE_BODY"]:
            assert private not in file_raw and private not in raw, private
        assert not any(e["fields"].get("route") == "/metrics" for e in file_events)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if terminal:
            reader.join(timeout=5)
            os.close(master)


def main():
    with tempfile.TemporaryDirectory(prefix="werk-logging-") as temporary:
        root = Path(temporary)
        env = {k:v for k,v in os.environ.items() if not k.startswith(("WERK_", "NO_COLOR", "CLICOLOR", "FORCE_COLOR"))}
        env.update(WERK_HOME=str(root / "store"), TERM="xterm-256color")
        for terminal in (False, True):
            serve(root, env, terminal)
            print("Raw JSONL, correlation, redaction, metrics, shutdown/file flush: " + ("PTY OK" if terminal else "pipes OK"))
        result = subprocess.run([BINARY, "list", "--json", "--log-format", "json", "--log-level", "debug"],
                                env=env, capture_output=True, text=True, timeout=20, check=True)
        json.loads(result.stdout)
        assert {e["event"] for e in events(result.stderr)} >= {"command.started", "command.completed"}
        result = subprocess.run([BINARY, "list", "--json", "--log-level", "off"],
                                env=env, capture_output=True, text=True, timeout=20, check=True)
        json.loads(result.stdout)
        assert not result.stderr
        result = subprocess.run([BINARY, "serve", "--model", "missing", "--allow-unauthenticated", "--verbose-pure"],
                                env=env, capture_output=True, text=True, timeout=20)
        assert result.returncode == 1 and not result.stdout
        assert any(e["event"] == "command.failed" and e["level"] == "error" for e in events(result.stderr))
        print("CLI stdout preserved, OFF filtering and JSONL command failures OK")


if __name__ == "__main__":
    main()
