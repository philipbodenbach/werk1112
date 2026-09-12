#!/usr/bin/env python3
"""Repeatable, dependency-free benchmark for OpenAI-compatible chat streams."""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import math
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


class StreamError(ValueError):
    pass


def sse_events(lines):
    """Yield (event name, data) with SSE multiline and comment semantics."""
    data = []
    event = "message"
    for line in lines:
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        line = line.rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            data, event = [], "message"
        elif not line.startswith(":"):
            field, sep, value = line.partition(":")
            if sep and value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "event":
                event = value
    # A final unterminated event is deliberately ignored by the SSE protocol.


def token_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def summarize_usage(usage, first_text_seconds, total_seconds, has_reasoning=False, estimate_decode_rate=False):
    prompt = token_count(usage.get("prompt_tokens"))
    completion = token_count(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details") or {}
    cached = token_count(details.get("cached_tokens")) if isinstance(details, dict) else None
    if cached is None:
        cached = token_count(usage.get("cached_tokens"))
    reasoning = usage.get("completion_tokens_details") or {}
    reasoning_tokens = token_count(reasoning.get("reasoning_tokens")) if isinstance(reasoning, dict) else None
    duration = total_seconds - first_text_seconds if first_text_seconds is not None else None
    # Chunks may contain multiple tokens. This is a client estimate, never a backend decode rate.
    # Some proxies hide reasoning deltas while retaining their tokens in completion_tokens.
    visible_usage = reasoning_tokens == 0 or (estimate_decode_rate and reasoning_tokens is None)
    estimate = None
    if (completion is not None and completion > 1 and duration is not None and duration > 0
            and not has_reasoning and visible_usage):
        estimate = (completion - 1) / duration
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "client_decode_tokens_per_second_estimate": estimate,
    }


def consume_stream(lines, started, clock=time.perf_counter, deadline_seconds=600, estimate_decode_rate=False):
    answer = []
    first_text = None
    finish = None
    usage = {}
    done = False
    has_reasoning = False
    error = None
    try:
        def bounded_lines():
            for line in lines:
                if clock() - started > deadline_seconds:
                    raise TimeoutError("stream exceeded the total deadline")
                yield line

        for event, data in sse_events(bounded_lines()):
            if data.strip() == "[DONE]":
                done = True
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                raise StreamError("invalid JSON in SSE data") from exc
            if not isinstance(chunk, dict):
                raise StreamError("SSE data must be a JSON object")
            if event == "error" or chunk.get("error"):
                raise StreamError("server returned a streaming error: " + json.dumps(chunk.get("error", chunk)))
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices", [])
            if not isinstance(choices, list):
                raise StreamError("SSE choices must be an array")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise StreamError("SSE choice must be an object")
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise StreamError("SSE delta must be an object")
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    has_reasoning = True
                text = delta.get("content")
                if text is not None and not isinstance(text, str):
                    raise StreamError("SSE text content must be a string")
                if text:
                    if first_text is None:
                        first_text = clock() - started
                    answer.append(text)
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
        if not done:
            raise StreamError("stream ended without [DONE]")
        if finish is None:
            raise StreamError("stream ended without a finish reason")
    except (OSError, ValueError, http.client.HTTPException) as exc:
        error = str(exc)
    total = clock() - started
    return {
        "first_text_seconds": first_text,
        "total_seconds": total,
        "finish_reason": finish,
        "truncated": finish == "length",
        "answer": "".join(answer),
        "usage": usage or None,
        "received_done": done,
        "has_reasoning": has_reasoning,
        "error": error,
        **summarize_usage(usage, first_text, total, has_reasoning, estimate_decode_rate),
    }


def check_answer(answer, checks):
    results = []
    for check in checks:
        kind = check["type"]
        if kind == "json_equals":
            try:
                actual = json.loads(answer)
                passed = json.dumps(actual, sort_keys=True) == json.dumps(check["value"], sort_keys=True)
            except json.JSONDecodeError:
                passed = False
            results.append({"type": kind, "passed": passed})
        elif kind == "exact_text":
            results.append({"type": kind, "passed": answer.strip() == check["value"]})
        elif kind == "manual":
            results.append({"type": kind, "passed": None, "criterion": check["criterion"]})
        else:
            raise ValueError("unsupported quality check: " + kind)
    return results


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def redact(value, secret):
    if isinstance(value, str):
        for form in (secret, json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1]):
            value = value.replace(form, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {redact(key, secret): redact(item, secret) for key, item in value.items()}
    return value


def request_chat(url, api_key, payload, timeout, deadline, estimate_decode_rate=False):
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise StreamError("expected Content-Type text/event-stream")
            result = consume_stream(response, started, deadline_seconds=deadline, estimate_decode_rate=estimate_decode_rate)
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
        result = {
            "first_text_seconds": None, "total_seconds": time.perf_counter() - started,
            "finish_reason": None, "truncated": False, "answer": "", "usage": None,
            "received_done": False, "has_reasoning": False, "error": str(exc),
            **summarize_usage({}, None, 0, estimate_decode_rate=estimate_decode_rate),
        }
    # An upstream error can echo request headers. Never retain the configured secret.
    if api_key:
        result = redact(result, api_key)
    return result


def load_cases(path):
    document = json.loads(Path(path).read_text())
    cases = document["cases"]
    ids = set()
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixtures must contain a non-empty cases array")
    for case in cases:
        if not isinstance(case.get("id"), str) or case["id"] in ids:
            raise ValueError("fixture IDs must be unique strings")
        ids.add(case["id"])
        if not isinstance(case.get("turns"), list) or not case["turns"]:
            raise ValueError("each fixture needs at least one turn")
        if token_count(case.get("max_tokens", 128)) in (None, 0):
            raise ValueError("max_tokens must be a positive integer")
        for turn in case["turns"]:
            if not isinstance(turn.get("prompt"), str) or not turn["prompt"]:
                raise ValueError("each turn needs a non-empty prompt")
            check_answer("", turn.get("checks", []))
    return cases


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def sampling_temperature(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return value


def sampling_top_p(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise argparse.ArgumentTypeError("must be finite and greater than 0 through 1")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="name of environment variable containing the key")
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).with_name("fixtures.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--temperature", type=sampling_temperature, default=0, help="sampling temperature (default: 0)")
    parser.add_argument("--top-p", type=sampling_top_p, default=0.95, help="nucleus sampling probability (default: 0.95)")
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument("--seed", type=int, help="sampling seed (default: 42)")
    seed_options.add_argument("--unseeded", action="store_true", help="omit seed and inherit the server's seed behavior")
    parser.add_argument("--estimate-decode-rate", action="store_true", help="assume completion usage counts visible tokens only when reasoning-token details are absent")
    parser.add_argument("--timeout", type=positive_float, default=180, help="socket I/O timeout in seconds")
    parser.add_argument("--deadline", type=positive_float, default=600, help="total deadline checked after each received line")
    parser.add_argument("--case", action="append", help="fixture ID to run; may be repeated")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("set --model or OPENAI_MODEL")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    parts = urllib.parse.urlsplit(args.base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        parser.error("--base-url must be an HTTP(S) URL without credentials, query, or fragment")
    try:
        cases = load_cases(args.fixtures)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error("invalid fixtures: " + str(exc))
    if args.case:
        unknown = set(args.case) - {case["id"] for case in cases}
        if unknown:
            parser.error("unknown fixture IDs: " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case["id"] in args.case]
    api_key = os.environ.get(args.api_key_env, "")
    url = args.base_url.rstrip("/") + "/chat/completions"
    sampling = {"temperature": args.temperature, "top_p": args.top_p}
    if not args.unseeded:
        sampling["seed"] = 42 if args.seed is None else args.seed
    report = {
        "schema_version": 1,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "settings": {**sampling, "repeats": args.repeats},
        "measurement_options": {"estimate_decode_rate": args.estimate_decode_rate},
        "notes": [
            "First sample does not imply a cold worker or cold prefix cache.",
            "Repeat samples start the fixture conversation again; multi-turn samples replay actual assistant answers.",
            "First text latency includes network, queue, loading, prefill, and any preceding reasoning.",
            "Client decode rate is only an estimate: SSE chunks can contain multiple tokens and final events add overhead.",
            "Without --estimate-decode-rate, an estimate requires explicit completion_tokens_details.reasoning_tokens=0 and no observed reasoning.",
            "--estimate-decode-rate assumes completion usage counts visible tokens only when reasoning details are missing. Hidden reasoning may be undetectable even when has_reasoning is false.",
            "Missing usage/cache metrics remain null. Temperature zero does not guarantee identical results on every backend.",
            "Language and answer usefulness require manual review; automated checks cover only the declared constraints.",
        ],
        "samples": [],
    }
    atomic_json(args.output, report)
    failed = False
    try:
        for case in cases:
            for repetition in range(args.repeats):
                messages = []
                for turn_index, turn in enumerate(case["turns"]):
                    messages.append({"role": "user", "content": turn["prompt"]})
                    payload = {
                        "model": args.model, "messages": messages,
                        **sampling,
                        "max_tokens": case.get("max_tokens", 128), "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    result = request_chat(url, api_key, payload, args.timeout, args.deadline, estimate_decode_rate=args.estimate_decode_rate)
                    result.update({
                        "case": case["id"], "repetition": repetition + 1, "turn": turn_index + 1,
                        "sample_kind": "first" if repetition == 0 else "repeat",
                        "prompt": turn["prompt"], "max_tokens": payload["max_tokens"],
                        "checks": check_answer(result["answer"], turn.get("checks", [])),
                    })
                    bad = bool(result["error"] or result["truncated"] or not result["answer"]
                               or any(check["passed"] is False for check in result["checks"]))
                    failed = failed or bad
                    report["samples"].append(result)
                    atomic_json(args.output, report)
                    first = result["first_text_seconds"]
                    first_label = f"{first:.2f}s" if first is not None else "unavailable"
                    print(f"{case['id']} repeat={repetition + 1} turn={turn_index + 1}: "
                          f"first_text={first_label} total={result['total_seconds']:.2f}s "
                          f"tokens={result['completion_tokens']} cached={result['cached_tokens']} "
                          f"status={'review' if bad else 'ok'}", flush=True)
                    if result["error"] or not result["answer"]:
                        break
                    messages.append({"role": "assistant", "content": result["answer"]})
    except KeyboardInterrupt:
        report["interrupted"] = True
        failed = True
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["needs_review"] = failed
    atomic_json(args.output, report)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
