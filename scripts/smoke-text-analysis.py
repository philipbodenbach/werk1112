#!/usr/bin/env python3
"""Exercise an already-running Werk server's text-analysis endpoints.

Example: python scripts/smoke-text-analysis.py --reranker ORG/RERANKER \
    --embedder ORG/EMBEDDER --decisions ORG/DECISIONS
Uses WERK_API_KEY when set. Does not start servers or install dependencies.
"""
import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--reranker")
    parser.add_argument("--embedder")
    parser.add_argument("--decisions")
    parser.add_argument("--backend", default="auto", choices=["auto", "transformers", "candle", "vllm"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if not any([args.reranker, args.embedder, args.decisions]):
        parser.error("specify at least one model")
    cases = []
    if args.reranker:
        cases.append(("/v1/rerank", {"model": args.reranker, "query": "Rust installieren",
            "documents": ["cargo install installiert Rust-Programme.", "Tomaten brauchen Licht."]}))
    if args.embedder:
        cases.append(("/v1/embeddings", {"model": args.embedder,
            "input": ["Rust installieren", "Tomaten pflanzen"], "input_type": "query", "dimensions": 128}))
    if args.decisions:
        cases.append(("/v1/classifications", {"model": args.decisions,
            "state": {"text": "Meine Rechnung wurde zweimal abgebucht."},
            "questions": {"billing": {"type": "noul", "instructions": "Geht es um eine Zahlung?"}}}))
    report = []
    headers = {"Content-Type": "application/json"}
    if os.environ.get("WERK_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["WERK_API_KEY"]
    for endpoint, payload in cases:
        payload["werk"] = {"backend": args.backend, "device": args.device}
        samples = []
        for _ in range(3):
            started = time.monotonic()
            request = urllib.request.Request(args.base_url.rstrip("/") + endpoint,
                json.dumps(payload).encode(), headers)
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    result = json.load(response)
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"{endpoint}: HTTP {error.code}: {error.read().decode()}") from None
            elapsed = time.monotonic() - started
            if endpoint.endswith("embeddings"):
                assert len(result["data"]) == 2
                for row in result["data"]:
                    vector = row["embedding"]
                    assert len(vector) == 128 and all(math.isfinite(x) for x in vector)
                    assert abs(sum(x*x for x in vector) - 1.0) < 0.015
            elif endpoint.endswith("rerank"):
                assert result["results"][0]["index"] == 0, "Relevant document must rank first"
                assert all(0 <= row["relevance_score"] <= 1 for row in result["results"])
            else:
                assert 0 <= result["answers"]["billing"]["noul"] <= 1
            samples.append({"http_seconds": elapsed, "werk": result["werk"], "usage": result["usage"]})
        report.append({"model": payload["model"], "endpoint": endpoint, "samples": samples})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
