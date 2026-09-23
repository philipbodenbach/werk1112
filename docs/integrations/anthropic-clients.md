# Anthropic Messages clients

Werk serves an Anthropic-compatible **Messages subset** at `POST /v1/messages`
alongside its existing OpenAI API, on the same port. Both adapters use the same
model routing, sessions and backend workers. There is no loopback HTTP proxy or
second model load. Use the actual installed Werk model ID, not a Claude alias.
This does not claim full Anthropic API or Claude Code compatibility.

## Start and call

An existing `werk serve` needs no additional API flag. For a local console test:

```bash
env -u WERK_OMLX_REASONING_EFFORT \
WERK_OMLX_EXPERT_CACHE_MB=22528 \
WERK_OMLX_EXPERT_EXECUTION=grouped \
WERK_OMLX_NGRAM_CACHE_MB=auto \
WERK_OMLX_THINKING=0 \
werk --backend omlx serve \
  --model pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit \
  --persistence --persistence-mode disk --persistence-reuse prefer \
  --port 11434 --verbose --allow-unauthenticated
```

For GLM, stop the Qwen server first and run the same command with
`--model Vontra/GLM-5.3-Flash-MLX-oQ2-MTP`. The unauthenticated option above is
for a local test. With normal server authentication, supply the generated Werk
key via `x-api-key` or `Authorization: Bearer <key>`.

```bash
curl -fsS -N http://127.0.0.1:11434/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model":"pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit",
    "max_tokens":96,
    "temperature":0,
    "stream":true,
    "messages":[{"role":"user","content":"Explain what a token is in three sentences."}]
  }'
```

Set `stream` to `false` for a normal JSON message. For a tool request:

```bash
curl -fsS http://127.0.0.1:11434/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model":"pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit",
    "max_tokens":512,"temperature":0,
    "tools":[{"name":"add","description":"Add two integers",
      "input_schema":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}],
    "tool_choice":{"type":"auto"},
    "messages":[{"role":"user","content":"Use add to calculate 2+3."}]
  }'
```

The client executes the returned `tool_use`, appends the assistant content
unchanged, and follows it with a user message containing
`{"type":"tool_result","tool_use_id":"<returned-id>","content":"5"}`.
Send the complete history and tool catalog again. Werk never executes client
tools. Multiple calls require one result per ID in that immediately following
user message. Check `stop_reason == "tool_use"` before executing a call;
`max_tokens` is incomplete output, even if an SDK can parse part of its JSON.

## Supported contract

| Input | Behavior |
| --- | --- |
| `anthropic-version` | Required, exactly `2023-06-01` |
| `anthropic-beta` | Rejected, including unknown beta flags |
| `model`, `messages`, `max_tokens` | Required; nonempty model/history and positive token budget |
| `system` | String or text-block array; separate from messages |
| Message roles | `user`, `assistant` |
| Content | String or text/tool-use/tool-result block array; adjacent text blocks join with a newline |
| `temperature`, `top_p` | Optional numbers in `[0, 1]` |
| Tools | Custom tools with unique names, descriptions, object `input_schema` |
| `tool_choice` | `auto`, `none`, `any`, named `tool`; capability constraints remain backend-dependent |
| `disable_parallel_tool_use` | `true` forwards a constraint; absent/false leaves backend parallelism unconstrained |
| `strict` | Passed to backend; never silently weakened |
| `tool_result.is_error` | Error content becomes the JSON string `{"content":"…","is_error":true}` in the backend tool message |
| `stop_sequences` | Absent/empty accepted; nonempty rejected until matched-stop metadata exists |
| Unknown fields | Rejected; includes `metadata`, `top_k`, `thinking`, `cache_control`, server tools and media blocks |

oMLX currently supports `auto`/`none`; it rejects forced/named choices,
`disable_parallel_tool_use: true` and `strict: true`. Other backends retain their
own capabilities. Backend validation errors during an already-started stream
arrive as an SSE `error` event. Tool calling cannot make a model/backend support
tools when it lacks a native tool parser.

Assistant text must precede its tool-use blocks. User tool results must precede
any trailing user text. Unsupported interleaving, orphaned/duplicate IDs,
missing results, nonobject tool arguments and malformed JSON are rejected.
Requests exceeding a known context limit are rejected rather than trimming a
tool cycle. Admission uses a byte-based estimate including tool schemas,
arguments and results; it is not exact tokenization. If model metadata provides
no context ceiling, the backend's own context/memory guard is authoritative.

Responses contain `id`, `type: message`, `role: assistant`, `model`, content,
`stop_reason`, `stop_sequence: null` and usage. Stops map to `end_turn`,
`max_tokens` and `tool_use`. Invalid completed tool JSON yields HTTP 502 rather
than fabricated arguments. Request validation is HTTP 400, authentication 401,
missing models 404 and oversized bodies 413, with an Anthropic error envelope.
Responses carry `request-id`; errors also include it in the body. The existing
`WERK_API_BODY_LIMIT_BYTES` setting and API key checks apply to this endpoint.

## Streaming and resource use

Named SSE events are `message_start`, `content_block_start`,
`content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`.
There is no OpenAI `[DONE]` event. Text is forwarded as soon as it arrives.
Tool headers may arrive fragmented from a backend, without a header-complete
marker. Tool output is therefore buffered until backend completion and emitted
as `tool_use` blocks with `input_json_delta`. This first version does **not**
provide live partial tool-argument delivery. The buffer is bounded to 128 calls
and 8 MiB of tool ID/name/argument bytes; exceeding either limit emits an error.

There is no detached adapter task or unbounded queue. Dropping the response
drops the backend stream; actual inference cancellation continues to follow the
selected backend's existing implementation. A backend error or premature EOF
emits `error`, with no successful `message_stop` afterward.

`message_start.usage` has provisional zero counts. Actual input/output counts
arrive in the final `message_delta`; the tested SDK accumulates both. Native
cache hits are not presented as Anthropic cache billing or TTL promises.
`--verbose` reports backend timing and cache diagnostics in the server log.

## SDK and model acceptance tests

The official Python SDK **0.86.0** is tested against a local Rust mock-backend
server for `messages.create`, `messages.stream`, exact final usage and complete
two successive client tool rounds with a 64-tool catalog. The ABBA script is
also smoke-tested against that fixture. Install the SDK in an isolated environment if needed:

```bash
python3 -m venv /tmp/werk-anthropic-sdk
/tmp/werk-anthropic-sdk/bin/pip install -r tests/anthropic-requirements.txt
env WERK_TEST_ANTHROPIC_PYTHON=/tmp/werk-anthropic-sdk/bin/python \
  cargo test --locked official_anthropic_sdk_text_stream_and_tool_loop --lib -- --ignored --nocapture
```

The SDK base URL is `http://127.0.0.1:11434` (without an extra `/v1`).
Against the running Qwen server:

```bash
/tmp/werk-anthropic-sdk/bin/python tests/anthropic_sdk.py \
  --model pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit
/tmp/werk-anthropic-sdk/bin/python tests/anthropic_sdk.py \
  --model pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit --catalog-size 64
```

After switching the server to GLM:

```bash
/tmp/werk-anthropic-sdk/bin/python tests/anthropic_sdk.py \
  --model Vontra/GLM-5.3-Flash-MLX-oQ2-MTP
/tmp/werk-anthropic-sdk/bin/python tests/anthropic_sdk.py \
  --model Vontra/GLM-5.3-Flash-MLX-oQ2-MTP --catalog-size 64
```

For authenticated servers add `--api-key <werk-key>`. These tests execute only
integer addition locally. They fail if the model skips the requested tools,
returns incomplete arguments or fails to finish the cycle; this distinguishes
model/tool-parser quality from successful transport with the mock fixture.

## Performance acceptance

Run with the same fixed model/cache budgets as above. Capture `--verbose`
server stderr in a file, then run:

```bash
python3 utils/benchmarks/anthropic_api.py \
  --model pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit \
  --server-log /tmp/werk-qwen-serve.log > /tmp/werk-qwen-api-abba.jsonl
```

Repeat for GLM with its model ID and log file. This sends three ABBA rounds to
one existing server, records first versus warm requests, TTFT, output counts,
native decode rate/cache hits (when available), output hash and observed
`omlx-server` PIDs. It neither starts nor stops a model. The process observation
is by executable name; inspect the server log as well if that runtime uses a
different name. Cache counts can be absent when the backend does not report them.

For old/new OpenAI regression testing, preserve the old binary before installing
the new build. Run servers sequentially with the same arguments and use
`--order openai --rounds 4 --label old` or `--label new`. Repeat build order as
old/new/new/old. Compare warm medians and native decode timing across repeats,
not a single first request. A hash/token-count mismatch needs investigation
before treating timings as equivalent work. No reproducible Qwen/GLM regression
is acceptable. Real-model results and worker reuse still require these console
tests; passing mock/SDK tests alone is not a hardware performance claim.

Deferred: `/v1/messages/count_tokens`, thinking, vision/documents, prompt-cache
controls, server-side tools, Batch/Files APIs and full Claude Code compatibility.

## Verification record (2026-09-23)

- `cargo test --locked --offline api:: --lib`: 73 passed; the opt-in SDK test is
  ignored in this ordinary run.
- `WERK_TEST_ANTHROPIC_PYTHON=python3 cargo test --locked --offline
  official_anthropic_sdk_text_stream_and_tool_loop --lib -- --ignored`: passed,
  using official SDK 0.86.0, two tool rounds per mode, 64 tools and the ABBA harness.
- `cargo check --locked --offline --no-default-features --features
  release-macos-apple-silicon`: passed.
- Full library suite, serial: 1009 passed, 5 failed, 1 ignored. All five failures
  also reproduce in a clean copy of pre-change HEAD `ce1ab91`: two CLI fallback
  diagnostic tests, two runtime-planner CUDA/ROCm selection tests and
  `werk_protocol::client::tests::client_parses_envelope_and_sends_bearer_without_leaking_it`.
  A parallel run additionally hit a persistence-expiry test that passes serially;
  the baseline also showed an intermittent llama-server restore test failure.
  These unrelated modules were not changed by this implementation.
- No installation or real Qwen/GLM inference was performed for this adapter.
