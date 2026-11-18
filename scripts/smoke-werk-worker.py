"""Exercise two native binaries with generated local Candle weights and a loopback-only store."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import queue
import secrets
import socket
import struct
import subprocess
import tempfile
import threading
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def cancel_worker_run(command, working_dir, environment, await_generation, await_cancel):
    """Cancel via the worker's local control interface while real inference is active."""
    with subprocess.Popen(command, cwd=working_dir, env=environment, text=True,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as worker:
        replies = queue.Queue()

        def read_replies():
            for line in worker.stdout:
                replies.put(line)

        reader = threading.Thread(target=read_replies, daemon=True)
        reader.start()
        sequence = 0

        def call(method, params):
            nonlocal sequence
            sequence += 1
            worker.stdin.write(json.dumps({"id": sequence, "method": method, "params": params}) + "\n")
            worker.stdin.flush()
            response = json.loads(replies.get(timeout=8))
            assert response.get("id") == sequence and "result" in response, (method, response)
            return response["result"]

        try:
            call("initialize", {"protocol_version": 1})
            session = call("session.create", {"profile": "general-readonly"})["session_id"]
            run = call("run.submit", {"session_id": session, "input": "Repeat OK.", "command_id": "submit"})
            handle = {"session_id": session, "run_id": run["run_id"]}
            live_id = await_generation()["id"]
            call("run.cancel", {**handle, "command_id": "cancel"})
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                status = call("run.status", handle)["status"]
                if status == "cancelled":
                    break
                time.sleep(.05)
            assert status == "cancelled", status
            await_cancel(live_id)
            call("shutdown", None)
            assert worker.wait(5) == 0
            return live_id
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.wait(5)
            reader.join(timeout=2)


def tiny_model(store, name, repeat):
    """A one-layer Llama with an identity residual and a deterministic LM head."""
    root = store / "models" / name
    root.mkdir(parents=True)
    width = 8
    eye = [[float(i == j) for j in range(width)] for i in range(width)]
    tokens = ["<unk>", "OK", "</s>", "x", "y", "z", "u", "v"]
    cfg = dict(model_type="llama", architectures=["LlamaForCausalLM"], hidden_size=width,
               intermediate_size=width, vocab_size=len(tokens), num_hidden_layers=1,
               num_attention_heads=1, num_key_value_heads=1, rms_norm_eps=0.00001,
               max_position_embeddings=8192, eos_token_id=2, torch_dtype="float32")
    write_json(root / "config.json", cfg)
    write_json(root / "tokenizer.json", {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
        "normalizer": None, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": None,
        "decoder": None, "model": {"type": "WordLevel", "unk_token": tokens[0],
                                   "vocab": {s: i for i, s in enumerate(tokens)}}})
    header, weights = {}, bytearray()

    def tensor(key, shape, values):
        start = len(weights)
        weights.extend(struct.pack("<" + str(len(values)) + "f", *values))
        header[key] = dict(dtype="F32", shape=shape, data_offsets=[start, len(weights)])

    tensor("model.embed_tokens.weight", [width, width], [x for row in eye for x in row])
    head = [[0.0] * width for _ in range(width)]
    for i in range(width):
        head[2 if i == 1 and not repeat else 1][i] = 20.0
    tensor("lm_head.weight", [width, width], [x for row in head for x in row])
    for key in ["model.norm", "model.layers.0.input_layernorm",
                "model.layers.0.post_attention_layernorm"]:
        tensor(key + ".weight", [width], [1.0] * width)
    for key in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        tensor("model.layers.0." + key + ".weight", [width, width], [0.0] * width ** 2)
    index = json.dumps(header, separators=(",", ":")).encode()
    index += b" " * (-len(index) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(index)) + index + weights)
    files = [dict(path=p.name, size=p.stat().st_size, checksum=sha(p)) for p in sorted(root.iterdir())]
    write_json(root / "manifest.json", {
        "id": name, "source": {"kind": "local_path", "path": str(root)}, "backend": "candle",
        "created_unix": 1, "format": "safe_tensors", "architecture": "llama",
        "tokenizer_path": "tokenizer.json", "config_path": "config.json",
        "model_path": "model.safetensors", "files": files, "tasks": ["text_generation"]})
    return sha(root / "model.safetensors")


def check(args):
    if not __debug__:
        raise RuntimeError("The native assertions require Python without -O.")
    werk, worker = args.werk.resolve(strict=True), args.worker.resolve(strict=True)
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "host": platform.platform(),
              "transport": "authenticated Werk Protocol 1.0 over loopback",
              "inference": "built-in Candle CPU, generated one-layer deterministic Llama F32 Safetensors",
              "server_sha256": sha(werk), "worker_sha256": sha(worker)}
    key = secrets.token_hex(32)
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("WW_", "WERK_"))}
    env.update(WERK_API_KEY=key, RAYON_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="werk-worker-cpu-") as tmp:
        root = Path(tmp)
        workspace = root / "workspace"
        workspace.mkdir()
        model, long_model = "cpu-once", "cpu-repeat"
        result["fixture_weight_sha256"] = {
            x: tiny_model(root / "store", x, x == long_model) for x in [model, long_model]}
        with closing(socket.socket()) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        origin = f"http://127.0.0.1:{port}"
        server_log = root / "server.log"
        with server_log.open("wb") as log:
            proc = subprocess.Popen([str(werk), "--model-home", str(root / "store"),
                                     "--device", "cpu", "--no-auto-install-backends",
                                     "serve", "--host", "127.0.0.1", "--port", str(port)],
                                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        def send(path, payload=None, *, auth=True, version="1.0", accept="application/json"):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=12)
            headers = {"Accept": accept, "Content-Type": "application/json",
                       "X-Werk-Protocol-Version": version}
            if auth:
                headers["Authorization"] = "Bearer " + key
            try:
                conn.request("POST" if payload is not None else "GET", "/werk/v1/" + path,
                             json.dumps(payload).encode() if payload is not None else None, headers)
                return conn
            except BaseException:
                conn.close()
                raise

        def json_call(path, payload=None, status=200, **kwargs):
            with closing(send(path, payload, **kwargs)) as conn, conn.getresponse() as resp:
                data = json.loads(resp.read())
                assert resp.status == status, (path, resp.status, data)
                assert resp.getheader("X-Werk-Protocol-Version") == "1.0"
                assert data["request_id"] == resp.getheader("X-Werk-Request-Id")
                assert data["protocol"] == {"major": 1, "minor": 0}
                return data["data" if status == 200 else "error"]

        def chat(which, count, stream):
            body = {"model": which, "messages": [{"role": "user", "content": "Local smoke"}],
                    "temperature": 0, "max_completion_tokens": count, "stream": stream}
            if stream:
                body["stream_options"] = {"include_usage": True}
            return body

        def observed(matches):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                events = json_call("observability")["requests"]
                for event in events:
                    if event["model"] == long_model and matches(event):
                        return event
                time.sleep(.05)
            raise AssertionError(("missing telemetry for " + long_model, events))

        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    json_call("info")
                    break
                except OSError:
                    if proc.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(server_log.read_text(errors="replace").replace(key, "[redacted]"))
                    time.sleep(.1)
            result["missing_key"] = json_call("info", auth=False, status=401)["code"]
            result["bad_version"] = json_call("info", version="2.0", status=406)["code"]
            caps = {c["id"]: c.get("operations", []) for c in json_call("capabilities")["capabilities"]
                    if c["status"] == "supported"}
            assert {"complete", "stream"} <= set(caps["api.chat.completions"])
            result["invalid_chat"] = json_call("chat/completions", {}, status=400)["code"]
            assert model in {m["id"] for m in json_call("models")["data"]}
            msg = json_call("chat/completions", chat(model, 8, False))
            assert msg["choices"][0]["message"]["content"].strip() == "OK", msg
            result["json_usage"] = msg["usage"]
            assert result["json_usage"]["completion_tokens"] == 1 and result["json_usage"]["prompt_tokens"] > 0
            conn = send("chat/completions", chat(model, 8, True), accept="application/json, text/event-stream")
            with closing(conn), conn.getresponse() as resp:
                assert resp.status == 200 and resp.getheader("Content-Type").startswith("text/event-stream")
                assert resp.getheader("X-Werk-Protocol-Version") == "1.0"
                result["sse_request_id"] = resp.getheader("X-Werk-Request-Id")
                assert result["sse_request_id"]
                frames = [line[6:] for line in resp.read().decode().splitlines() if line.startswith("data: ")]
                assert frames[-1] == "[DONE]"
                chunks = [json.loads(frame) for frame in frames[:-1]]
                assert "".join(c["delta"].get("content", "") for x in chunks for c in x["choices"]).strip() == "OK"
                result["sse_usage"] = next(x["usage"] for x in chunks if "usage" in x)
                assert result["sse_usage"] == result["json_usage"]
            for streaming in [False, True]:
                config = root / "config.json"
                write_json(config, {"base_url": origin, "model": model, "stream": streaming,
                                    "timeout_ms": 30000, "max_body_bytes": 8388608,
                                    "options": {"temperature": 0}, "capabilities": {
                                        "tool_calling": "supported", "streaming": "supported", "context_tokens": 8192,
                                        "parallel_tools": "unsupported", "structured_output": "unknown",
                                        "images": "unsupported", "reasoning": "unknown"}})
                cli = subprocess.run([str(worker), "--werk-config", str(config), "--secret-env", "WERK_API_KEY",
                                      "--store", str(root / f"session-{streaming}.sqlite"), "--workspace", str(workspace),
                                      "--harness", "minimal", "run", "Respond briefly."],
                                     cwd=root, env=env, capture_output=True, text=True, timeout=35)
                assert cli.returncode == 0 and cli.stdout.strip() == "OK", (
                    cli.returncode, cli.stdout.replace(key, "[redacted]"), cli.stderr.replace(key, "[redacted]"))
                result["worker_sse" if streaming else "worker_json"] = "OK"
            conn = send("chat/completions", chat(long_model, 4000, True), accept="application/json, text/event-stream")
            with closing(conn), conn.getresponse() as resp:
                assert resp.status == 200
                while True:
                    line = resp.readline()
                    assert line and b"[DONE]" not in line
                    if line.startswith(b"data: "):
                        chunks = json.loads(line[6:])["choices"]
                        if any(c["delta"].get("content") for c in chunks):
                            break
            stream_cancel = observed(lambda event: event["state"] == "cancelled")
            result["cancelled_sse"] = stream_cancel["state"]
            with closing(send("chat/completions", chat(long_model, 4000, False))) as conn:
                pending = observed(lambda event: event["id"] > stream_cancel["id"] and
                                   event["state"] == "waiting / prefill")
                # A TCP send-half-close lets a peer legally continue computing
                # its response. Abort the connection to signal loss of the client.
                layout = "HH" if os.name == "nt" else "ii"
                conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack(layout, 1, 0))
            json_cancel = observed(lambda event: event["id"] == pending["id"] and event["state"] == "cancelled")
            result["cancelled_json"] = json_cancel["state"]
            settings = json.loads(config.read_text(encoding="utf-8"))
            previous_id = json_cancel["id"]
            for mode in ["json", "sse"]:
                settings.update(model=long_model, stream=mode == "sse")
                write_json(config, settings)
                rpc = [str(worker), "--werk-config", str(config), "--secret-env", "WERK_API_KEY",
                       "--store", str(root / f"rpc-{mode}.sqlite"), "--workspace", str(workspace),
                       "--harness", "minimal", "rpc"]
                live_id = cancel_worker_run(
                    rpc, root, env, lambda: observed(lambda event: event["id"] > previous_id and
                                                      event["state"] not in {"cancelled", "done", "error"}),
                    lambda ident: observed(lambda event: event["id"] == ident and event["state"] == "cancelled"))
                result["worker_run_cancel_" + mode] = {"local_run": "cancelled", "inference": "cancelled"}
                previous_id = live_id
            resumed = json_call("chat/completions", chat(long_model, 2, False))
            assert resumed["usage"]["completion_tokens"] == 2, resumed
            result["same_model_after_cancel"] = resumed["choices"][0]["message"]["content"].strip()
        finally:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)
        assert key not in server_log.read_text(errors="replace")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--werk", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--evidence", type=Path)
    opts = parser.parse_args()
    record = check(opts)
    if opts.evidence:
        write_json(opts.evidence, record)
    print(json.dumps(record))
