#!/usr/bin/env python3
"""Controlled native-worker contract fixture; never performs GPU inference."""
import json
import os
from pathlib import Path
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

root = Path(__file__).parent
if "--help" in sys.argv:
    print("--device --split-mode --main-gpu --tensor-split --n-cpu-moe --no-warmup --parallel --fit")
    raise SystemExit(0)
if "--version" in sys.argv:
    print("version: test (controlled-worker)")
    raise SystemExit(0)
if "--list-devices" in sys.argv:
    for index, _ in enumerate(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")):
        print(f"CUDA{index}: controlled fixture")
    raise SystemExit(0)
if (root / "fail-start").exists():
    raise SystemExit(47)
with (root / "starts.jsonl").open("a") as output:
    output.write(json.dumps({"pid": os.getpid(), "argv": sys.argv[1:],
                             "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
                             "threads": os.environ.get("OMP_NUM_THREADS")}) + "\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send([] if self.path == "/slots" else {"status": "ok"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send({"content": "fixture", "stop": True, "tokens_predicted": 1,
                   "tokens_evaluated": 1, "timings": {"predicted_n": 1, "prompt_n": 1}})


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Handler).serve_forever()
