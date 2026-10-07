#!/usr/bin/env python3
"""Exercise an explicit deployment configuration; never download models or build engines.

Example: python utils/multi_gpu/validate.py --config profiles.json --out /tmp/result.json
Outputs observed HTTP timings, native telemetry, exact plans and GPU memory samples.
Cold means a new worker, NOT a dropped OS page cache. Run separate configs to compare.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


def get(url):
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


def infer(base, alias, prompt, tokens, session, timeout):
    started = time.perf_counter()
    first = None
    text = ""
    usage = None
    body = {"model": alias, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "seed": 42, "max_tokens": tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "x-werk-session-id": session})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            for choice in event.get("choices", []):
                delta = choice.get("delta", {}).get("content", "") or ""
                if delta and first is None:
                    first = time.perf_counter() - started
                text += delta
            usage = event.get("usage") or usage
    return {"alias": alias, "session": session, "ttft_seconds": first,
            "total_seconds": time.perf_counter() - started, "usage": usage, "text": text}


def quality_checks(base, alias, timeout):
    def chat(body):
        body = {"model": alias, "temperature": 0, "seed": 42, "max_tokens": 128, **body}
        request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json", "x-werk-session-id": "quality-reference"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    schema = chat({"messages": [{"role": "user", "content": "Compute 17 + 25 and return it as result."}],
                   "response_format": {"type": "json_schema", "json_schema": {"name": "arithmetic", "strict": True,
                    "schema": {"type": "object", "properties": {"result": {"type": "integer"}},
                               "required": ["result"], "additionalProperties": False}}}})
    structured = json.loads(schema["choices"][0]["message"]["content"])
    if structured != {"result": 42}:
        raise AssertionError(f"structured arithmetic differs: {structured}")
    messages = [{"role": "user", "content": "Call add with a=17 and b=25."}]
    tools = [{"type": "function", "function": {"name": "add", "description": "Add two integers",
              "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                             "required": ["a", "b"], "additionalProperties": False}}}]
    called = chat({"messages": messages, "tools": tools,
                   "tool_choice": {"type": "function", "function": {"name": "add"}}})
    assistant = called["choices"][0]["message"]
    call = assistant["tool_calls"][0]
    if call["function"]["name"] != "add" or json.loads(call["function"]["arguments"]) != {"a": 17, "b": 25}:
        raise AssertionError(f"unexpected tool call: {call}")
    messages.extend([assistant, {"role": "tool", "tool_call_id": call["id"], "content": "42"},
                     {"role": "user", "content": "Return only the resulting integer."}])
    continued = chat({"messages": messages, "tools": tools, "tool_choice": "none"})
    if continued["choices"][0]["message"]["content"].strip() != "42":
        raise AssertionError("tool continuation did not return 42")
    return {"structured": schema, "tool_call": called, "continuation": continued, "status": "passed_reference_cases"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--werk", default="target/debug/werk")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--prompt", default="Return the integer result of 17 + 25, with no explanation.")
    parser.add_argument("--required-gpus", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--quality", action="store_true", help="Also assert structured JSON, forced tool call and tool continuation reference cases")
    args = parser.parse_args()
    if args.repetitions < 1 or args.tokens < 1:
        parser.error("repetitions and tokens must be positive")
    werk = str(Path(args.werk).resolve())
    inventory = json.loads(subprocess.check_output([werk, "gpus"], text=True))
    result = {"schema": 1, "parameters": vars(args), "inventory_before": inventory,
              "werk_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "dirty_diff": subprocess.check_output(["git", "diff", "--stat"], text=True),
              "cold_definition": "fresh worker; OS page cache not flushed", "samples": [],
              "hardware_cases": {"single": "configured", "two_equal": "requires explicit config and >=2 GPUs",
                                 "two_unequal": "requires explicit config and >=2 unequal GPUs",
                                 "independent": "requires explicit config and >=2 GPUs",
                                 "mixed": "requires explicit config and >=3 GPUs",
                                 "replicas": "requires explicit config and >=2 GPUs"}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    visible = [d for d in inventory["devices"] if d["visible_index"] is not None]
    if len(visible) < args.required_gpus:
        result.update(status="skipped", reason="insufficient visible GPUs")
        out.write_text(json.dumps(result, indent=2))
        return
    try:
        result["plans"] = json.loads(subprocess.check_output([werk, "deployment-plan", args.config], text=True, stderr=subprocess.PIPE))
    except subprocess.CalledProcessError as error:
        result.update(status="rejected_plan", reason=error.stderr.strip())
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps({"status": result["status"], "reason": result["reason"], "output": str(out)}))
        raise SystemExit(1) from None
    # Exact instance IDs exercise every replica. Concurrent groups share a round.
    ids = [p["profile"]["id"] for p in result["plans"]]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    stop = threading.Event()
    memory = []

    def sample_memory():
        while not stop.is_set():
            try:
                value = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=5)
                memory.append({"time": time.time(), "used_mib": {line.split(",")[0].strip(): int(line.split(",")[1]) for line in value.splitlines()}})
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop.wait(0.2)

    log_path = out.with_suffix(".server.log")
    command = [werk, "--deployments", str(Path(args.config).resolve()), "serve", "--port", str(port), "--allow-unauthenticated"]
    with log_path.open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        monitor = threading.Thread(target=sample_memory, daemon=True)
        monitor.start()
        try:
            deadline = time.monotonic() + 60
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Werk exited {server.returncode}; see {log_path}")
                try:
                    get(base + "/v1/models")
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Werk API readiness timed out")
                    time.sleep(0.2)
            for repetition in range(args.repetitions):
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(ids)) as pool:
                    futures = [pool.submit(infer, base, alias, args.prompt, args.tokens,
                                           f"validation-{alias}", args.timeout) for alias in ids]
                    round_samples = [f.result() for f in futures]
                for sample in round_samples:
                    sample.update(repetition=repetition, worker_state="cold" if repetition == 0 else "warm")
                result["samples"].extend(round_samples)
                result.setdefault("telemetry", []).append(get(base + "/werk/v1/observability"))
            result["diagnostics_after"] = get(base + "/werk/v1/deployments")
            if args.quality:
                result["quality"] = {alias: quality_checks(base, alias, args.timeout) for alias in ids}
            result["status"] = "passed_transport"
            result["correctness"] = "Outputs retained for reference comparison; transport success alone is not numerical equivalence."
            latencies = [s["total_seconds"] for s in result["samples"]]
            result["latency_seconds"] = {"median": statistics.median(latencies), "min": min(latencies), "max": max(latencies)}
        except Exception as error:
            result.update(status="failed", error=str(error))
        finally:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            stop.set()
            monitor.join(timeout=6)
            result["gpu_memory_samples"] = memory
            result["gpu_peak_used_mib_including_other_processes"] = {d["id"]: max((s["used_mib"].get(d["id"], 0) for s in memory), default=None) for d in visible}
            result["unavailable_measurements"] = ["PCIe transfers", "pinned RAM", "per-worker GPU allocation", "logit/logprob equivalence"]
            out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "output": str(out), "samples": len(result["samples"])}))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
