# oMLX server short-prefix cache validation

Validated locally on 2026-09-11 with oMLX 0.6.4 and
`mlx-community/DeepSeek-V4-Flash-2bit-DQ`.

## Fix

`serve --persistence` now configures the oMLX backend before eager model
preparation. The existing native exact-prefix helper applies to all requests
served by that worker: sessions, direct generation, streaming, tools and
request-local thinking overrides. Automatic routing preserves this configuration.

Server caches live under the private worker's `cache/prefix-cache` directory,
with the existing lifetime lock and a 4 GB SSD limit. They store matching token
prefix state, not conversation history. Worker restarts start with a cold cache.
The separate CLI chat archive and durable KV namespace retain their existing
behavior. See [client configuration](integrations/openai-clients.md).

## Controlled comparison

Both runs used `WERK_OMLX_THINKING=0`,
`WERK_OMLX_EXPERT_CACHE_MB=8192`, and `serve --persistence` with automatic
backend selection. Each run sent two sequential streaming OpenAI-compatible
requests. The second included the exact first assistant answer and repeated
the user message:

> Write me a sentence about the programming language rust.

Sampling was `temperature=0`, `max_tokens=64`, with no tools. All four answers
were identical and contained 38 generated tokens. First-text and total times
below were measured at the HTTP client; token/cache counts came from the worker
and Werk logs because this endpoint does not implement stream usage summaries.

| Measurement | Before | After |
| --- | ---: | ---: |
| Turn 1: first text | 15.384 s | 17.600 s |
| Turn 1: total HTTP duration | 38.298 s | 41.697 s |
| Turn 1: cached prompt tokens | 0 / 14 | 0 / 14 |
| Turn 2: first text | 26.063 s | 11.529 s |
| Turn 2: total HTTP duration | 50.299 s | 38.163 s |
| Turn 2: cached prompt tokens | 0 / 66 | 53 / 66 |

In this single comparison, the follow-up's first text arrived 56% earlier and
total HTTP duration fell 24%. This establishes working prefix reuse, not a
general throughput guarantee. Cold requests still require prompt evaluation;
the fix does not accelerate each generated token.

## Open WebUI check

A new chat was submitted through the already running Open WebUI in Safari.
It used the same Rust prompt and the warmed worker. Werk logged 13 cached tokens
out of 14, 28 generated tokens, and 19.849 seconds of backend generation time.
Open WebUI stored the completed assistant output with `done=true` and no error.
This time is the backend duration, not a browser end-to-end measurement.
The actual local test chat is
`http://localhost:8080/c/59a1e77d-6a49-493e-b09d-5650116da5e1`.

No client-specific conversation identifier or cache extension was required.
werkStation can use the same `/v1/chat/completions` conversation requests.

## Regression checks

- 48 oMLX Rust tests passed, including worker reuse through eager preparation,
  sessions, streaming/tools and thinking overrides; expert-budget isolation;
  and disabled/unsupported cache fallback without a second worker.
- Five CLI routing/persistence tests and five API chat-option tests passed.
- All 18 Python persistence tests passed, including the real DeepSeek native
  SSD/Metal round trip and independently mutable restored states for two chats.
- Release build and installation succeeded; the running server was restarted
  with the fix on the original port and API key.
