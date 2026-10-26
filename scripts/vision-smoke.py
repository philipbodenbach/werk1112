#!/usr/bin/env python3
"""Opt-in real image smoke against an already running, configured Werk server.

No downloads, service restarts, filesystem path inputs or cloud selection.
Only the generated pixels contain the expected colors. Reports no image bytes
or credentials. WERK_API_KEY or --api-keys supplies existing authorization.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import random
import struct
import time
import tomllib
import urllib.request
import zlib


def picture():
    palette = {"red": (240, 0, 0), "green": (0, 160, 0),
               "blue": (0, 0, 240), "yellow": (255, 235, 0), "purple": (150, 0, 180)}
    names = random.SystemRandom().sample(list(palette), 2)
    pixels = bytearray()
    for y in range(128):
        pixels.append(0)
        for x in range(256):
            color = palette[names[x // 128]] if 16 <= y < 112 and 16 <= x % 128 < 112 else (255, 255, 255)
            pixels.extend(color)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 128, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
    return names, "data:image/png;base64," + base64.b64encode(png).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-keys", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--modes", nargs="+", choices=["normal", "stream", "tool-follow-up"], default=["normal", "stream", "tool-follow-up"])
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    key = os.environ.get("WERK_API_KEY", "")
    if args.api_keys:
        key = tomllib.loads(args.api_keys.read_text())["keys"][0]["key"]
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key

    for mode in args.modes:
        expected, url = picture()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": 'Identify the colors of the two squares in this image. Reply only with JSON {"left":"color","right":"color"} using English color names.'},
            {"type": "image_url", "image_url": {"url": url}},
        ]}]
        if mode == "tool-follow-up":
            messages.extend([
                {"role": "assistant", "content": None, "tool_calls": [{"id": "visual_check", "type": "function", "function": {"name": "check", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "visual_check", "content": "Continue with the image inspection."},
                {"role": "user", "content": "Report the two colors from the earlier image as requested."},
            ])
        body = {"model": args.model, "messages": messages, "stream": mode == "stream",
                "max_tokens": args.max_tokens, "temperature": 0, "reasoning_effort": "none"}
        if mode == "tool-follow-up":
            body["tools"] = [{"type": "function", "function": {"name": "check", "description": "Check readiness", "parameters": {"type": "object", "properties": {}}}}]
            body["tool_choice"] = "none"
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            if mode == "stream":
                parts = []
                done = False
                for line in response:
                    if line.startswith(b"data:"):
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            done = True
                            break
                        event = json.loads(payload)
                        if "error" in event:
                            raise RuntimeError("stream returned an error")
                        for choice in event.get("choices", []):
                            parts.append(choice.get("delta", {}).get("content") or "")
                assert done, "missing stream terminator"
                answer = "".join(parts)
            else:
                answer = json.load(response)["choices"][0]["message"]["content"]
        start, end = answer.find("{"), answer.rfind("}")
        decoded = json.loads(answer[start:end + 1])
        actual = [str(decoded.get(side, "")).lower().strip() for side in ["left", "right"]]
        assert actual == expected, f"{mode}: expected {expected}, received {actual}"
        print(json.dumps({"mode": mode, "model": args.model, "result": "passed", "colors": expected, "seconds": round(time.monotonic() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()
