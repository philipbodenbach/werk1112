#!/usr/bin/env python3
"""Compare a running Werk node at request concurrency 1/2/4/8 (stdlib only)."""
import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.error
import urllib.request


def fetch(url, headers, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout)


def snapshot(base, headers):
    try:
        with fetch(base + "/werk/v1/observability", headers, timeout=10) as response:
            return json.load(response)
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        return {"unavailable": str(error)}


def request(base, headers, model, prompt, tokens):
    started = time.perf_counter()
    first = None
    usage = {}
    try:
        with fetch(base + "/v1/chat/completions", headers, {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens, "stream": True, "stream_options": {"include_usage": True},
        }) as response:
            done = False
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    done = True
                    break
                value = json.loads(raw)
                if value.get("error"):
                    raise ValueError(str(value["error"]))
                for choice in value.get("choices", []):
                    delta = choice.get("delta", {})
                    if first is None and (delta.get("content") or delta.get("tool_calls")):
                        first = time.perf_counter()
                usage = value.get("usage") or usage
            if not done:
                raise ValueError("stream ended without [DONE]")
        ended = time.perf_counter()
        count = usage.get("completion_tokens")
        return {
            "model": model, "elapsed_seconds": ended - started,
            "ttft_seconds": None if first is None else first - started,
            "completion_tokens": count, "usage": usage,
            "request_tokens_per_second": None if count is None else count / (ended - started),
            # Werk has no portable per-request backend queue-time measurement.
            "queue_seconds": None,
        }
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        return {"model": model, "elapsed_seconds": time.perf_counter() - started, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", action="append", required=True, help="repeat to interleave models")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--requests", type=int, default=16, help="requests per concurrency level")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Explain how a compiler checks types. Give a detailed example.")
    parser.add_argument("--warmup", action="store_true", help="one request per model before measuring")
    parser.add_argument("--output", default="concurrency-results.json")
    args = parser.parse_args()
    if min(args.concurrency + [args.requests, args.max_tokens]) < 1:
        parser.error("concurrency, requests and max-tokens must be positive")
    base = args.url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("WERK_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["WERK_API_KEY"]
    if args.warmup:
        for model in args.model:
            result = request(base, headers, model, args.prompt, args.max_tokens)
            if "error" in result:
                parser.error("warmup failed: " + result["error"])
    report = {"models": args.model, "warmup": args.warmup, "runs": [],
              "notes": ["TTFT includes Werk and native queue time; queue time is unavailable separately.",
                        "Snapshots expose backend-specific memory, instance and cache metrics where available; they are not peak memory.",
                        "Repeated prompts exercise native prefix reuse; use a separate cold server run to compare cold loads."]}
    for concurrency in args.concurrency:
        before = snapshot(base, headers)
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(request, base, headers, args.model[index % len(args.model)], args.prompt, args.max_tokens)
                       for index in range(args.requests)]
            results = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
        known_tokens = [result["completion_tokens"] for result in results if result.get("completion_tokens") is not None]
        ttfts = [result["ttft_seconds"] for result in results if result.get("ttft_seconds") is not None]
        run = {"concurrency": concurrency, "elapsed_seconds": elapsed,
               "successful_requests": sum("error" not in result for result in results),
               "aggregate_tokens_per_second": sum(known_tokens) / elapsed if known_tokens else None,
               "ttft_median_seconds": statistics.median(ttfts) if ttfts else None,
               "requests": results, "observability_before": before, "observability_after": snapshot(base, headers)}
        report["runs"].append(run)
        print(json.dumps({key: run[key] for key in ("concurrency", "successful_requests", "aggregate_tokens_per_second", "ttft_median_seconds")}))
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    return 0 if all(run["successful_requests"] == args.requests for run in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
