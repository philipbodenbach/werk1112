#!/usr/bin/env python3
"""Compare existing Werk endpoints sequentially; never start/load/stop a model.

Default order is OpenAI, Anthropic, Anthropic, OpenAI (ABBA). For old/new
binary comparison run with --order openai and distinct --label values.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request


def workers():
    try:
        rows = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
        return sorted(int(line.split(None, 1)[0]) for line in rows.splitlines()
                      if len(line.split(None, 1)) == 2
                      and Path(line.split(None, 1)[1]).name == "omlx-server")
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--api-key", default="werk-local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--order", default="openai,anthropic,anthropic,openai")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--server-log", type=Path, help="Verbose log, for Anthropic native decode/cache metrics")
    parser.add_argument("--prompt", default="Explain what a token is in three sentences.")
    args = parser.parse_args()
    order = args.order.split(",")
    assert order and set(order) <= {"openai", "anthropic"}
    assert args.rounds > 0
    samples = []
    for round_index in range(args.rounds):
        for protocol in order:
            offset = args.server_log.stat().st_size if args.server_log else 0
            payload = dict(model=args.model, messages=[{"role": "user", "content": args.prompt}],
                           max_tokens=args.max_tokens, temperature=0, stream=True)
            if protocol == "openai":
                payload["stream_options"] = {"include_usage": True}
            request = urllib.request.Request(
                args.base_url.rstrip("/") + ("/v1/messages" if protocol == "anthropic" else "/v1/chat/completions"),
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + args.api_key,
                         "anthropic-version": "2023-06-01"})
            before = workers()
            start = time.monotonic()
            first = last = None
            text = ""
            input_tokens = output_tokens = None
            timings = None
            stop = None
            terminal = False
            with urllib.request.urlopen(request, timeout=600) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        terminal = True
                        continue
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    content = None
                    if protocol == "openai":
                        choices = event.get("choices", [])
                        if choices:
                            content = choices[0].get("delta", {}).get("content")
                            stop = choices[0].get("finish_reason") or stop
                        if event.get("usage"):
                            input_tokens = event["usage"]["prompt_tokens"]
                            output_tokens = event["usage"]["completion_tokens"]
                        timings = event.get("werk", {}).get("timings") or timings
                    else:
                        if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta":
                            content = event["delta"]["text"]
                        if event["type"] == "message_delta":
                            input_tokens = event["usage"]["input_tokens"]
                            output_tokens = event["usage"]["output_tokens"]
                            stop = event["delta"]["stop_reason"]
                        if event["type"] == "message_stop":
                            terminal = True
                    if content:
                        last = time.monotonic() - start
                        first = last if first is None else first
                        text += content
            elapsed = time.monotonic() - start
            assert terminal and output_tokens is not None, "Incomplete stream"
            if args.server_log and protocol == "anthropic":
                with args.server_log.open("rb") as log:
                    log.seek(offset)
                    for line in log.read().decode(errors="replace").splitlines():
                        prefix = "[werk serve] anthropic "
                        if line.startswith(prefix):
                            record = json.loads(line[len(prefix):])
                            if record["model"] == args.model:
                                timings = record["timings"]
            decode_seconds = (timings or {}).get("decode_seconds", 0)
            row = {"label": args.label, "protocol": protocol, "round": round_index,
                   "first_in_series": not samples, "ttft_seconds": first, "total_seconds": elapsed,
                   "input_tokens": input_tokens, "output_tokens": output_tokens, "stop_reason": stop,
                   "decode_tokens_per_second": output_tokens / decode_seconds if decode_seconds else None,
                   "client_tokens_per_second": (output_tokens - 1) / (last - first)
                   if last is not None and last > first and output_tokens > 1 else None,
                   "cached_prompt_tokens": (timings or {}).get("cached_prompt_tokens"),
                   "omlx_worker_pids_before": before, "omlx_worker_pids_after": workers(),
                   "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
            samples.append(row)
            print(json.dumps(row), flush=True)
    for protocol in dict.fromkeys(order):
        rows = [row for row in samples if row["protocol"] == protocol and not row["first_in_series"]]
        medians = {key: statistics.median(values) for key in
                   ("ttft_seconds", "total_seconds", "decode_tokens_per_second")
                   if (values := [row[key] for row in rows if row[key] is not None])}
        print(json.dumps({"summary": args.label, "protocol": protocol, "warm_samples": len(rows), "medians": medians}))


if __name__ == "__main__":
    main()
