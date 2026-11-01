# Five-model implementation report

Evidence checked on 2026-10-13. See the [audit](model-support-audit.md) for official
sources and the [matrix](model-support-matrix.md) for exact IDs, mode-specific
status, requirements and reproducible commands. No real inference of these five
models was performed; no new path is `VERIFIED_LOCAL` or `VERIFIED_REMOTE`.

## Design and changed surfaces

- `src/model_support.rs` separates vendor/model identity, capabilities, license,
  runtime requirements and inference verification. `ModelSource::Provider` in
  `src/model_store.rs` represents hosted registrations without fake weight files.
  `werk support list`, `/werk/v1/model-support`, model listing and capabilities
  expose these distinctions. Registration never persists credentials.
- `src/providers/http.rs` shares HTTPS transport, environment-based authentication,
  client caching and typed upstream errors. It disables redirects, limits body
  sizes, redacts response bodies/URLs in errors, bounds a request to 300 seconds,
  and retries explicit 429 rejection up to three attempts. GET 502/504 can retry;
  ambiguous POST failures cannot, because generation might already be billed.
- `src/backend/provider.rs` implements Mistral chat through the existing generation
  trait and OpenAI wire parser. It preserves tool messages/calls and JSON schema
  options, translates `seed` to `random_seed`, handles SSE completion/usage, and
  separates native thinking blocks from visible text. Hosted chat is routed before
  local format/runtime selection in `src/cli.rs`.
- `src/backend/vllm.rs` recognizes Kolibri's native architecture, checks the vendor
  plugin before local launch, supplies its tool/reasoning parsers and FP8 KV cache,
  and defaults to the recommended 262,144-token context. Explicit context and
  parallelism flags remain authoritative. The upstream tokenizer/template handles
  native messages. Reasoning effort reaches the template as well as the API.
- `src/providers/media.rs` implements the actual BFL and ElevenLabs schemas.
  BFL submits/polls/downloads a PNG/JPEG and reports moderation as an error; the
  storage download receives no API key. Eleven v4 uses dialogue inputs with real
  voice IDs and returns binary audio with a codec-appropriate MIME type, including
  a streaming endpoint. These contracts are documented extensions, not changes
  to the existing OpenAI/Diffusers image or speech contracts.
- `src/providers/action.rs` and `runtime/werk_flux_action.py` implement a resident
  DROID policy worker using the official `FluxActionPolicy`. It validates camera,
  state and trajectory shapes, preprocesses RGB images, returns explicit units
  and `executed:false`, and never controls a robot. Workers are scoped to server
  state, serialize policy access and are terminated/reaped on eviction/shutdown.
  `register-action` links a complete local package and preserves file inventory.
- `src/providers/cli.rs`, `src/api/providers.rs`, router/state and discovery wiring
  provide CLI commands plus authenticated task-specific endpoints. Existing access
  logging covers the routes; blocking work retains logging context. A four-permit
  gate bounds concurrent provider work, body limits constrain large image input,
  and bounded streaming channels provide backpressure.
- README, changelog, API/environment references and `examples/model-support/`
  document keys, registration, commands and payloads. No dependencies or lockfile
  changes were needed. Existing FLUX.1 Schnell runtime code is unchanged; a loader
  regression fixture was added to `runtime/test_werk_media_companion.py`.

## Latest startup preload behavior

`src/api/preload.rs` runs an ordered, deduplicated startup sequence and stops at
the first failed target. The CLI exits on failure instead of starting a partially
prepared service. Successful handles remain owned by the server until shutdown.

Kolibri uses the existing vLLM runtime-cache transaction and a real short warmup
with `reasoning_effort:none` / `enable_thinking:false`, so hidden reasoning does
not consume the entire 32-token preload response. FLUX Action loads/prepares the
policy, predicts once on synthetic observations, then pins its worker in the
same runtime-cache mechanism. Fresh failed warmups roll back their worker;
multiple pinned policies survive the ordinary cache capacity of one. Tests use
fake policies/processes and do not verify model inference.

Cloud preloading for Mistral, BFL and ElevenLabs validates local credential syntax
and pins a reusable HTTPS client. It makes **zero network requests**. Its separate
`serve.preload.remote_prepared` event explicitly states that it did not warm
provider weights or test account access. The ordinary readiness count therefore
means preparation succeeded, not that hosted inference was verified. This avoids
an implicit billable generation every time `werk serve --preload` starts.

## Verification evidence

Environment: Linux x86_64, Rust/Cargo 1.98.0, Python 3.12.8, RTX 3090 with 24 GiB
VRAM and about 98 GiB host RAM. The official Kolibri/FLUX Action Python runtimes,
checkpoint packages and all three provider keys were absent. This GPU does not
meet Kolibri's documented minimum. No paid API requests were made.

Commands executed:

```sh
cargo check --locked
cargo fmt --all --check
git diff --check
python3 -m unittest discover -s runtime -p 'test_werk*.py'
RUST_MIN_STACK=16777216 cargo test --lib --quiet -- --test-threads=1
RUST_MIN_STACK=16777216 cargo test --lib --locked kolibri -- --test-threads=1
RUST_MIN_STACK=16777216 cargo test --lib --locked provider -- --test-threads=1
cargo clippy --all-targets --locked -- -D warnings
cargo build --locked --bin werk
```

The full Rust library run passed **1,306 tests**, with **4 ignored**, in 180.67s.
The Python run passed **87 tests**, including DROID validation and the Schnell
loader fixture. An earlier ordinary parallel Rust run failed in existing
process/cache tests and then overflowed a CLI parser test's stack. The passing run
used the larger stack configured by `.github/workflows/observability.yml` and
serial execution to isolate shared-process tests. This is a test-environment
limitation, not evidence that the parallel run passed.

The new mock tests cover provider registration/status, schema rejection, tools
and tool-result continuation, structured-output request mapping, Mistral SSE,
reasoning blocks, binary audio, BFL polling/moderation/download, 429 retry,
ambiguous-failure non-replay, stalled-request timeout, credential isolation,
authentication and invalid HTTP requests. Resident-worker tests exercise warmup,
reuse, pinning beyond cache capacity, failed-warmup rollback and process cleanup.
Additional Kolibri tests check launch parameters and the non-reasoning preload
request; they do not execute a real Kolibri tool call.

Strict Clippy is not clean repository-wide: existing warnings in untouched code
include collapsible conditionals, type complexity, clone-on-copy and suspicious
open options. Warnings introduced in the new provider/API code were fixed. The final check results below distinguish these existing warnings from the
completed focused checks and CLI/server smoke.

## Final focused checks and server smoke

- Both focused Kolibri tests passed; the provider/API test selection passed all
  16 tests, including the subsequently added timeout and thinking-block tests.
- `cargo build --locked --bin werk` succeeded. Formatting and whitespace checks
  succeeded. Final `cargo check --locked` succeeded.
- `cargo clippy --all-targets --locked` completed successfully with existing
  warnings (61 for the library, 69 for library tests including duplicates).
  No warning location fell in a new module or an added/changed diff line.
  The stricter `-D warnings` run remains blocked by these baseline warnings.
- A built-binary smoke used a temporary `WERK_HOME`: registered/listed all three
  hosted aliases, read the five-entry catalog and checked that Mistral CLI routing
  fails with `missing_credentials` before sending an upstream request.
- Two actual HTTP server starts used an ordered preload list with three hosted
  models, one nonexistent target and a duplicate Mistral entry. With placeholder
  credential syntax, exactly three of four targets prepared successfully. Without
  provider keys, zero of four prepared. Both servers continued serving discovery,
  preserved deduplication, enforced bearer authentication, omitted secrets from
  logs and exited on SIGINT. Only local discovery/auth requests were sent; no
  provider generation or availability request was made.

## Limits and follow-up

- Execute authorized real-model/API smoke tests before promoting any verification
  status. Include Kolibri ordinary chat and tools, Mistral vision/tools/schema/SSE,
  BFL generation/reference editing and Eleven dialogue/audio streaming.
- Mistral local support awaits released weights, their license and a verified
  runtime. No substitute model or speculative conversion is registered.
- DROID support is the native root policy contract. SO-101, other embodiments and
  deployment to a real robot need separate adapters and validation. Synthetic
  preload proves the installed policy executes only when run against real weights.
- The task-specific extensions do not participate in canonical media job history
  or generation-token accounting. Access logs and request IDs still apply. They
  currently have no remote job-cancellation endpoint. Disconnecting audio/chat
  consumers stops further delivery/reads, but an in-flight blocking read can last
  until its timeout; disconnect does not guarantee cancellation or refund at the
  vendor. BFL polling is bounded, but a submitted generation can finish upstream
  after a caller disconnects. Action predictions finish under the worker's bounded
  reply timeout. These are limitations, not claims of upstream cancellation.
- BFL/Eleven cloud preload prepares only the local client. Provider account access,
  quota, model availability, voice permissions and latency remain unverified.
- Real FLUX.1 Schnell image inference was not repeated; its loader and existing
  media/text regressions passed with fixtures.
