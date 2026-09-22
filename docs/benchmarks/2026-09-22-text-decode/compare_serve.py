#!/usr/bin/env python3
"""Sequential ABBA comparison of legacy/direct decode on one candidate binary.

Each fresh, private Werk server receives the same two independent requests:
first and repeat. No status polling occurs during generation. Requires
the local models and the usual Werk/oMLX installation; does not install code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/benchmarks"))
import chat  # noqa: E402

STATUS_KEYS = (
    "architecture", "cache_policy", "cache_hits", "cache_misses", "cache_evictions",
    "forward_calls", "forward_seconds", "routing_seconds", "materialize_seconds",
    "disk_bytes_read", "disk_read_calls", "disk_read_seconds", "output_evaluations",
    "resident_cache_bytes", "effective_cache_budget_bytes", "cache_budget_bytes",
    "last_prefill_admission", "last_decode_admission", "glm_layer_profile",
)


def process_table():
    # Only executable names are read: command arguments can contain API keys.
    output = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,lstart=,comm="], text=True
    )
    rows = {}
    for line in output.splitlines():
        fields = line.split(None, 7)
        if len(fields) == 8:
            rows[int(fields[0])] = {
                "parent": int(fields[1]), "started": " ".join(fields[2:7]),
                "comm": fields[7],
            }
    return rows


def refuse_existing_worker():
    if any(Path(row["comm"]).name == "omlx-server" for row in process_table().values()):
        raise RuntimeError("An oMLX server is active; refusing concurrent model loads")


class OwnedTree:
    """Track descendants and verify start times before signalling surviving PIDs."""

    def __init__(self, child):
        self.child = child
        self.owned = {}
        self.refresh()

    def refresh(self):
        rows = process_table()
        seeds = {pid for pid, started in self.owned.items()
                 if pid in rows and rows[pid]["started"] == started}
        if self.child.poll() is None and self.child.pid in rows:
            seeds.add(self.child.pid)
        while True:
            found = {pid for pid, row in rows.items() if row["parent"] in seeds}
            if found <= seeds:
                break
            seeds |= found
        for pid in seeds:
            self.owned[pid] = rows[pid]["started"]
        return rows

    def stop(self):
        self.refresh()
        if self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
        # oMLX creates its own process group, so killing Werk's group alone
        # would miss it. Never signal an unrecorded PID or a reused PID.
        for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            rows = self.refresh()
            remaining = [pid for pid, started in self.owned.items()
                         if pid in rows and rows[pid]["started"] == started]
            for pid in remaining:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            until = time.monotonic() + wait
            while remaining and time.monotonic() < until:
                time.sleep(0.1)
                self.child.poll()
                rows = process_table()
                remaining = [pid for pid in remaining if pid in rows and
                             rows[pid]["started"] == self.owned[pid]]
            if not remaining:
                break
        self.child.wait(timeout=1)
        if remaining:
            raise RuntimeError("An owned benchmark process did not stop")


def json_request(url, key, timeout=5):
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def phases(log_path):
    prefix = "[werk serve] phases "
    return [json.loads(line[len(prefix):])
            for line in log_path.read_text(errors="replace").splitlines()
            if line.startswith(prefix)]


def worker_settings(directory, previous):
    # Never select another worker's settings by newest timestamp.
    created = set(directory.glob("*/settings.json")) - previous
    if len(created) != 1:
        raise RuntimeError(f"Expected one newly created worker, found {len(created)}")
    return json.loads(created.pop().read_text())


def status(settings):
    value = json_request(
        f"http://127.0.0.1:{settings['server']['port']}/werk/experts/status",
        settings["auth"]["api_key"],
    )
    return {key: value[key] for key in STATUS_KEYS if key in value}


def add_metrics(sample):
    before, after = sample["before"], sample["after"]
    counters = ("cache_hits", "cache_misses", "cache_evictions", "forward_calls",
                "forward_seconds", "routing_seconds", "materialize_seconds",
                "disk_bytes_read", "disk_read_calls", "disk_read_seconds",
                "output_evaluations")
    delta = {key: after[key] - before[key] for key in counters
             if isinstance(before.get(key), (float, int))
             and isinstance(after.get(key), (float, int))}
    sample["expert_delta"] = delta
    hits, misses = delta.get("cache_hits", 0), delta.get("cache_misses", 0)
    sample["expert_hit_rate"] = hits / (hits + misses) if hits + misses else None
    sample["expert_read_GiB"] = delta.get("disk_bytes_read", 0) / 1024 ** 3
    duration = (sample.get("backend_phases") or {}).get("decode_seconds")
    tokens = sample.get("completion_tokens")
    sample["decode_tokens_per_second"] = tokens / duration if duration and tokens else None
    sample["answer_sha256"] = hashlib.sha256(sample["answer"].encode()).hexdigest()


def summarize(samples):
    result = {}
    for state in ("first", "repeat"):
        groups = {mode: [row for row in samples if row["mode"] == mode and
                         row["state"] == state and not row.get("error")]
                  for mode in ("legacy", "direct")}
        rates = {mode: statistics.mean(row["decode_tokens_per_second"] for row in rows)
                 for mode, rows in groups.items()
                 if rows and all(row["decode_tokens_per_second"] for row in rows)}
        result[state] = {
            "samples_per_mode": {mode: len(rows) for mode, rows in groups.items()},
            "mean_decode_tokens_per_second": rates,
            "direct_change_percent": 100 * (rates["direct"] / rates["legacy"] - 1)
            if len(rates) == 2 else None,
            "all_answers_identical": len({row["answer_sha256"]
                                           for rows in groups.values() for row in rows}) == 1
            if all(groups.values()) else None,
        }
    return result


def run_server(args, mode, index, report, report_path):
    refuse_existing_worker()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    key = secrets.token_hex(32)
    secret_values = [key]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("WERK_OMLX_", "OMLX_"))}
    env.pop("WERK_API_KEYS", None)
    env.update(WERK_API_KEY=key, WERK_OMLX_EXPERT_CACHE_MB=str(args.cache_mb),
               WERK_OMLX_EXPERT_EXECUTION="grouped", WERK_OMLX_NGRAM_CACHE_MB="auto",
               WERK_OMLX_THINKING="0", WERK_OMLX_TEXT_DECODE=mode)
    workers = args.werk_home / "backends/omlx/workers"
    settings_before = set(workers.glob("*/settings.json"))
    command = [str(args.binary), "--model-home", str(args.werk_home), "--backend", "omlx",
               "serve", "--model", args.model, "--host", "127.0.0.1", "--port", str(port),
               "--persistence", "--persistence-mode", "disk", "--persistence-reuse", "prefer",
               "--verbose"]
    log_path = args.output / f"{index}-{mode}.log"
    base_url = f"http://127.0.0.1:{port}"
    started = time.perf_counter()
    log_fd = os.open(log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tree = None
    with os.fdopen(log_fd, "w") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            tree = OwnedTree(child)
            until = time.monotonic() + args.startup_timeout
            while True:
                try:
                    json_request(base_url + "/v1/models", key, timeout=2)
                    break
                except OSError:
                    if child.poll() is not None or time.monotonic() >= until:
                        raise RuntimeError(f"Server failed to become ready; see {log_path}")
                    tree.refresh()
                    time.sleep(0.25)
            startup_seconds = time.perf_counter() - started
            settings = worker_settings(workers, settings_before)
            secret_values.append(settings["auth"]["api_key"])
            for state in ("first", "repeat"):
                tree.refresh()
                before = status(settings)
                payload = {"model": args.model,
                           "messages": [{"role": "user", "content": args.prompt}],
                           "temperature": 0, "max_tokens": args.max_tokens,
                           "stream": True, "stream_options": {"include_usage": True}}
                result = chat.request_chat(base_url + "/v1/chat/completions", key, payload,
                                           args.request_timeout, args.request_timeout)
                result.update(mode=mode, run=index, state=state, before=before,
                              after=status(settings), startup_seconds=startup_seconds,
                              log=str(log_path))
                # The phases line can be flushed just after the terminal SSE event.
                expected = 1 if state == "first" else 2
                until = time.monotonic() + 2
                found = phases(log_path)
                while len(found) < expected and time.monotonic() < until:
                    time.sleep(0.05)
                    found = phases(log_path)
                result["backend_phases"] = found[expected - 1] if len(found) >= expected else None
                add_metrics(result)
                report["samples"].append(result)
                report["summary"] = summarize(report["samples"])
                chat.atomic_json(report_path, report)
                print(json.dumps({name: result.get(name) for name in (
                    "run", "mode", "state", "prompt_tokens", "completion_tokens",
                    "decode_tokens_per_second", "first_text_seconds", "total_seconds",
                    "answer_sha256", "error")}), flush=True)
                if result["error"] or not result["backend_phases"]:
                    raise RuntimeError(f"Request failed or backend phases missing; see {log_path}")
        finally:
            try:
                if tree is None:
                    child.terminate()
                    child.wait(timeout=20)
                else:
                    tree.stop()
            finally:
                log.flush()
                content = log_path.read_text(errors="replace")
                for secret in secret_values:
                    content = content.replace(secret, "[REDACTED]")
                log_path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    default_home = os.environ.get("WERK_HOME") or str(
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "werk1112")
    parser.add_argument("--werk-home", type=Path, default=Path(default_home))
    parser.add_argument("--cache-mb", type=int, default=22528)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--prompt", default="Erkläre in etwa 200 Wörtern, wie ein Sprachmodell Text erzeugt.")
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--settle-seconds", type=float, default=10,
                        help="allow macOS to reclaim a stopped worker's GPU allocation")
    args = parser.parse_args()
    if min(args.cache_mb, args.max_tokens, args.startup_timeout, args.request_timeout) <= 0:
        parser.error("budgets and timeouts must be positive")
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be nonnegative")
    args.binary = args.binary.resolve(strict=True)
    args.werk_home = args.werk_home.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("--output must be empty; existing results are never overwritten")
    report_path = args.output / "report.json"
    report = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "binary": str(args.binary),
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "model": args.model, "cache_mb": args.cache_mb, "max_tokens": args.max_tokens,
              "prompt": args.prompt, "thinking": False, "temperature": 0,
              "expert_execution": "grouped", "ngram_cache_mb": "auto",
              "settle_seconds": args.settle_seconds,
              "persistence": {"mode": "disk", "reuse": "prefer"},
              "order": ["legacy", "direct", "direct", "legacy"], "samples": [],
              "note": "First/repeat refer to each fresh worker; first does not imply a "
                      "cold OS or persistence cache. No existing caches are deleted. "
                      "Status snapshots are taken only before and after each request. "
                      "Decode rate = completion_tokens / backend decode_seconds."}
    chat.atomic_json(report_path, report)
    try:
        for index, mode in enumerate(report["order"]):
            if index:
                time.sleep(args.settle_seconds)
            run_server(args, mode, index, report, report_path)
        report["completed"] = True
    except BaseException as exc:
        report["completed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["summary"] = summarize(report["samples"])
        chat.atomic_json(report_path, report)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    main()
