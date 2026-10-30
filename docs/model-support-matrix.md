# Model support matrix

Checked **2026-10-13**. `VERIFIED_LOCAL` / `VERIFIED_REMOTE` require executed real
inference. `IMPLEMENTED_UNTESTED` means a runnable adapter exists and its protocol
is tested, but actual model inference has not been executed here.
`BLOCKED_UPSTREAM` identifies a pending release; `UNAVAILABLE` means no verified
path for that mode. No new path is marked verified by this change.

| Exact model | Local | Remote | Backend / requirements | Tests executed / next action |
| --- | --- | --- | --- | --- |
| `Aleph-Alpha/Kolibri-1` | IMPLEMENTED_UNTESTED | UNAVAILABLE (official hosted API not verified) | vLLM + `aleph-alpha-inference>=1`; FP8 checkpoint, native tokenizer/template, sufficient NVIDIA memory; Apache-2.0 | Parser/effort/routing tests and existing vLLM/tool/preload regressions. Next: install plugin and weights on vendor-supported hardware; run completion and tool round trip. |
| `mistral-large-4` | BLOCKED_UPSTREAM | IMPLEMENTED_UNTESTED | Official Mistral HTTPS API; `MISTRAL_API_KEY` + preview access; service terms | Mock chat, tools, structured field mapping, SSE, errors/retries. Next: authorized live smoke; check weights/license after announced 2026-10-31 release. |
| `black-forest-labs/flux-3-action-droid` | IMPLEMENTED_UNTESTED | UNAVAILABLE (official hosted API not verified) | Official `flux-action`, checkpoint + pinned encoders, Linux/NVIDIA, matching CUDA/PyTorch/NATTEN; FLUX Kommunity License v1.0 | Typed schema, fake resident worker warmup/reuse/rollback/shutdown; no real trajectory generated. Next: install official environment and use a recorded DROID observation. |
| `flux-3-image` | UNAVAILABLE (weights/runtime not verified) | IMPLEMENTED_UNTESTED | BFL HTTPS API; `BFL_API_KEY`; BFL API terms | Mock submit/poll/moderation/download, native field validation, secret isolation. Next: authorized live image/edit smoke. |
| `eleven_v4` | UNAVAILABLE (proprietary, no weights verified) | IMPLEMENTED_UNTESTED | ElevenLabs dialogue API; `ELEVENLABS_API_KEY`, accessible voice IDs; service/subscription terms | Mock dialogue mapping, binary audio/stream route, language/format/limits. Next: authorized live audio and streamed dialogue smoke. |

Sources, architecture/context details and exact blockers are in the
[audit](model-support-audit.md). No credentials were present for the three paid
APIs, so no billable smoke calls were made.

## Discovery and registration

```sh
werk support list
werk support register mistral-large-4
werk support register flux-3-image
werk support register eleven_v4
werk list --json
werk serve --preload mistral-large-4 flux-3-image eleven_v4
```

Keys belong in `MISTRAL_API_KEY`, `BFL_API_KEY`, `ELEVENLABS_API_KEY`; neither
registration nor listing requires them. Preload without a needed key logs a
per-model failure and stops server startup. With a key, cloud preload performs **no network
call**, no inference and no billing. `serve.preload.remote_prepared` explicitly
reports this distinction. Local-only servers do not need any provider keys.

`GET /werk/v1/model-support` returns the evidence catalog.
`GET /v1/models` lists registered aliases with the proper provider owner;
`GET /v1/capabilities` includes task-specific endpoints and verification status.
All new HTTP endpoints honor the existing Werk server bearer authentication.
The server requires authentication by default: configure `WERK_API_KEY` and add
`-H "Authorization: Bearer $WERK_API_KEY"` to the curl examples below, or append
`--allow-unauthenticated` to `werk serve` for loopback development. Vendor keys
stay on the Werk server.

## Kolibri: local vLLM

Use a separate environment with the vendor plugin. Its dependencies select the
compatible vLLM version; generic `werk backend install vllm` alone is insufficient.
Follow the [official setup](https://huggingface.co/Aleph-Alpha/Kolibri-1) and ensure
hardware satisfies the vendor's requirements before downloading the ~78 GB model.

```sh
python -m venv /path/to/kolibri-env
/path/to/kolibri-env/bin/pip install 'aleph-alpha-inference>=1'
export WERK_VLLM_PYTHON=/path/to/kolibri-env/bin/python
werk pull Aleph-Alpha/Kolibri-1 --name kolibri
werk --backend vllm serve --model kolibri --preload kolibri
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kolibri","messages":[{"role":"user","content":"Guten Tag!"}],"max_tokens":128,"reasoning_effort":"none"}'
```

Werk adds `--reasoning-parser kolibri1 --tool-call-parser kolibri1
--enable-auto-tool-choice --kv-cache-dtype fp8 --max-model-len 262144` unless
explicit flags already supply values. Multi-GPU deployments still need the
appropriate tensor-parallel setting (for example `WERK_VLLM_ARGS='--tensor-parallel-size 2'`).
The server applies the checkpoint's tokenizer/chat template, including tool-result
continuation messages. Effort values are `none`, `low`, `medium`, `high`.
For an independently managed vLLM server, configure these flags upstream; Werk
never changes a remote process's launch settings. No GGUF/MLX conversion is claimed.

## Mistral Large 4: hosted chat

After registering and supplying `MISTRAL_API_KEY`:

```sh
werk run mistral-large-4 'Antworte kurz auf Deutsch.' --max-tokens 128
werk serve --model mistral-large-4 --preload mistral-large-4
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/model-support/mistral-chat.json
```

The existing chat endpoint accepts normal messages, tools/tool_choice,
parallel_tool_calls, tool results, response_format and streaming. The provider
translates `seed` to `random_seed`. Limits and account access are enforced by
Mistral, not guessed from local memory. Native token-counting is unavailable.
Local backend overrides (for example `--backend cuda`) are rejected for a hosted
model. Use `--backend auto` (default).

## FLUX 3 Image: generation and reference editing

```sh
werk support image --request examples/model-support/flux-3-image.json --output flux3.png
curl http://127.0.0.1:11434/werk/v1/images/flux-3 \
  -H 'Content-Type: application/json' \
  --data-binary @examples/model-support/flux-3-image.json -o flux3-api.png
```

To edit, add `"images":["https://.../reference.png"]` (or base64) and describe the
change in `prompt`; up to ten references are accepted. `aspect_ratio`,
`resolution` (`768sq`, `1k`, `1.5k`, `2k`, `4k`), `safety_tolerance` (0–4),
`grounding` and `version:"latest"` follow BFL's schema. Seed, negative prompt,
width/height, mode and output_format are rejected. Returned PNG/JPEG bytes carry
the matching MIME type. Werk polls the returned regional URL and downloads the
signed result immediately, without forwarding credentials to storage.

This extension uses the vendor schema. Existing canonical Diffusers image jobs,
`/v1/images/generations` and FLUX.1 Schnell retain their existing behavior; they
are not a FLUX 3 parameter translation layer.

## Eleven v4: audio and streaming dialogue

Replace the sample voice ID with an accessible voice, then:

```sh
werk support speech --request examples/model-support/eleven-v4.json --output speech.mp3
curl http://127.0.0.1:11434/werk/v1/audio/dialogue \
  -H 'Content-Type: application/json' \
  --data-binary @examples/model-support/eleven-v4.json -o speech-api.mp3
```

Set `stream:true` for incremental HTTP audio from the upstream `/stream` endpoint.
The CLI writes the completed audio file. Add text/voice pairs for dialogue.
The endpoint enforces ≤2,000 characters and ≤10 voices. Formats encode codec,
rate and bitrate, e.g. `mp3_44100_128`, `wav_24000`, `pcm_24000`, `ulaw_8000`;
the adapter validates the official format allowlist. Raw PCM is returned as raw
PCM, not falsely labeled WAV. High quality formats may require a subscription.
Audio tags remain in text; optional language_code, seed, settings and normalization
are forwarded. No automatic voice cloning or synthetic default voice is provided.

## FLUX 3 Action: local DROID predictions

Install the [official environment](https://github.com/black-forest-labs/flux-action/blob/main/docs/setup.md)
and download the complete DROID root package, including its native config,
manifest and pinned encoder references. The upstream example environment uses
Python 3.12, PyTorch 2.10.0+cu128, Transformers 5.16.1 and NATTEN
0.21.6+torch2100cu128. These are upstream requirements, **not locally tested versions**.

```sh
export WERK_FLUX_ACTION_PYTHON=/path/to/flux-action/.venv/bin/python
werk support register-action flux3-droid --checkpoint /path/to/droid
"$WERK_FLUX_ACTION_PYTHON" examples/model-support/droid-request.py \
  /path/to/observation.npz 'the recorded task instruction' > /tmp/action.json
werk support action --request /tmp/action.json --output /tmp/trajectory.json
werk serve --preload flux3-droid
curl http://127.0.0.1:11434/werk/v1/actions/predictions \
  -H 'Content-Type: application/json' --data-binary @/tmp/action.json
```

Images are synchronized 640×360 RGB base64 PNG/JPEG observations. State contains
seven absolute joint positions in radians plus a gripper closed fraction (0–1).
The response supplies 32 rows of eight numeric targets, explicit units and
`executed:false`. Prediction does not actuate hardware. A resident worker loads
and prepares the official policy once; preload performs a synthetic prediction
and pins that worker until server shutdown. Multiple preloaded policies survive
the ordinary one-worker cache limit. Failed warmups roll back their newly loaded
worker. SO-101, custom embodiments, training and live control loops need their
own contracts and are not silently routed through DROID.
