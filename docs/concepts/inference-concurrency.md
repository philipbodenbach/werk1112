# Inference concurrency

## Audit baseline (2026-09-27, before changes)

One `werk serve` already resolves multiple installed model IDs. Both OpenAI
chat completions and Anthropic messages use `api/generation.rs`, `ModelStore::get`,
manifest identity, the CLI routing backend and the concrete generation backend.
`--model` selects a default; it is not an allowlist. Models load lazily (the default
can be prepared at startup). Routing and runtime-control registries are distinct
from weight/process caches. Manifest/configuration identity, not agent identity,
selects a resident runtime.

| Runtime | Multiple models | Same-model overlapping inference | Cross-model inference | Native batching |
| --- | --- | --- | --- | --- |
| llama-server (CPU/CUDA/ROCm/Vulkan/Metal) | Supported | Partially supported: exclusive state gate and default one slot | Supported after load | Backend supports it; Werk gate prevents overlap |
| vLLM, owned or external | Supported | Supported; native scheduler | Supported | Native |
| oMLX | Supported, 16 configurations | Native scheduler | Supported after startup; registry/probe locks block unrelated startup | Native, model/runtime dependent |
| Candle | Supported | Serialized mutable model/KV state | Supported | No Werk continuous batching |
| ONNX GenAI Python | Partial, bounded LRU, default one | Serialized resident transport | Missing: shared worker serializes models | No Werk sequence batching |
| Experimental Burn | Missing simultaneous residency (one cached model) | Serialized mutable model | Missing: backend-wide inference lock | No Werk continuous batching |
| Legacy llama FFI / llama_cpp | Supported | Session/context lock; weights shared | Supported | Independent contexts, no server scheduler |
| Transformers compatibility Python | Partial, bounded LRU | Serialized resident transport | Missing: shared worker | No Werk continuous batching |
| MLX / MLX-VLM external CLI | One-shot invocation | Separate invocations can overlap and duplicate weights | Separate invocations | Command dependent; no resident contract |
| External ONNX runners / media companions | Invocation/transport dependent | Companion contract dependent | Shared resident transports serialize | Companion dependent |

The baseline answer is **partially**: cross-model routing and most resident caches
already exist, but safe, efficient overlap is not uniform. Simultaneous cold misses
can duplicate llama-server/vLLM processes and Candle/legacy llama models. oMLX avoids
that race by holding its entire worker registry during startup. API session creation
also races and runs synchronously on an async executor before dispatch. Native
server startup selects and releases an ephemeral port before the child binds it,
allowing concurrent owned startups to collide. Existing session handles can pin
a failed server. ONNX, Burn and Transformers serialize
unrelated models. These are implementation findings, not assumptions about hardware.

## Concurrency concepts and implementation boundaries

Clients != HTTP connections != HTTP requests != conversations != active sequences
!= runtime instances != model instances. One connection can carry many requests;
one agent can have several conversations and spend most of its time running tools.
A runtime instance may own a child process or in-process model state. Native slots
bound active sequences, not clients. CPU worker threads and GPU command queues are
backend resources, not agent identities.

Creation must synchronize by exact runtime key. Registry locks must cover only
lookup/publication, never model loading or inference. Mutable model/context locks
belong to one runtime. Native HTTP schedulers own batching, KV allocation, prefix
reuse and GPU scheduling. Explicit slot save/restore must exclude ordinary requests
on that worker. No generic token scheduler or agent-to-process mapping is needed.

The existing process lifecycle manager atomically registers children against
shutdown, kills/reaps owned workers and rejects creation after shutdown begins.
Unix and Windows signal handlers terminate the CLI after cleanup; this is abortive
shutdown, not graceful completion of queued requests. External servers are not
owned or terminated. Native in-process calls are not cooperatively cancellable.

Audit evidence: `src/api/{router,state,generation}.rs`, `src/cli.rs` routing
backends, `src/runtime_control/routing.rs`, `src/backend/{llama_server,vllm,omlx,
candle,onnxruntime,burn,external,llama_fast}.rs`, `src/media_companion.rs`, and
`src/backend/llama_process_lifecycle.rs`.

## Resulting behavior

One node can expose multiple resident models to independent clients. Same-model
requests reuse weights; different runtime keys initialize independently. Whether
same-model decoding overlaps depends on the selected backend and its configuration.

| Runtime | Residency and maximum active calls after this change | Sequence/batching owner | KV/prefix ownership | CPU and GPU execution |
| --- | --- | --- | --- | --- |
| llama-server | One healthy process per exact key; ordinary HTTP calls overlap. Default one native slot; configure `--parallel N`. | llama.cpp slots and continuous batching; Werk does not schedule tokens. | Native slot KV and prompt caching; explicit state operations take the worker's exclusive gate. | Native thread pools (`--threads`, `--threads-batch`), native CPU/CUDA/ROCm/Vulkan/Metal execution. Multiple workers compete for device resources. |
| vLLM | One owned process per exact launch key, or references to a remote service. Werk does not cap ordinary concurrent calls. | Native active sequences, continuous batching; pass `--max-num-seqs`, `--max-num-batched-tokens` through `WERK_VLLM_ARGS`. | vLLM owns KV and APC. `serve --persistence` can request native APC on supported local launch targets; remote configuration remains external. | Native Python/engine workers, native accelerator scheduler; per-process memory budgets must fit together. |
| oMLX | One worker per model/invocation/configuration, maximum 16 retained configurations. Lookup and initialization are per key; compatibility probes serialize only per model. | Installed oMLX scheduler; supported architectures/runtime version determine sequence and batching limits. No numeric limit is imposed by Werk's ordinary HTTP path. | oMLX owns native caches; verified server prefix reuse remains configuration/model dependent. | Native MLX/Metal command scheduling in each worker; no Werk GPU scheduling. Apple Silicon execution only. |
| Candle | One in-process model per manifest identity; one active generation per model, other calls wait on that model's mutex. | Werk isolates mutable model state; no continuous batching in this adapter. | Model-owned mutable KV cleared at each generation; no cross-request prefix cache. | Candle/device implementation owns CPU threads and CUDA/Metal queues. `Send + Sync` wrapper serializes mutable use; overlapping different models need hardware headroom. |
| ONNX Python GenAI | One serialized resident Python transport per manifest identity and interpreter. Bounded model-worker LRU, default 1, configurable up to 8. | One request/generator at a time per worker; independent resident models can overlap. | Generator/request-local KV; resident model/tokenizer reused; no cross-request prefix contract. | Current embedded fallback uses CPU. ORT owns its thread pools; Werk does not equate CPU threads to agents. External CUDA/ROCm runners are a separate path. |
| Transformers compatibility | One serialized resident transport per existing manifest/device/dtype key. Bounded worker LRU, default 1, up to 8. | One generation at a time per worker; independent models can overlap. | Request-local generator/cache; model/tokenizer resident. | PyTorch owns CPU/CUDA/MPS execution; each worker has its own runtime overhead and thread pools. |
| Burn (experimental Phi-3) | Bounded exact-model cache, default 1, up to 8; one mutable inference lock per model. | No continuous batching in this adapter. | Request-local generation cache; no prefix reuse contract. | Burn CPU/CUDA implementation owns execution. Compile-time feature and model support remain required. |
| Legacy llama FFI / high-level | Single-flight shared weights. API text session keyed by manifest and seed holds its own context mutex; distinct contexts can use shared weights. | Context generation; no llama-server continuous scheduler. | Session-context KV and prefix trimming. Tools/images bypass the API session cache. | Native thread/device execution. The context lock protects mutable state, not other models. |
| External MLX / MLX-VLM / ONNX commands | One-shot processes; overlapping calls can reload/duplicate weights. No resident contract is claimed. | Opaque command implementation. | No validated cross-request cache contract. | Command/platform dependent. For resident MLX text concurrency select compatible oMLX. |
| Generic media / managed Qwen TTS companion | Existing separate serialized resident transports and pipeline LRUs; unchanged. | Companion owns execution; media job state locks protect metadata, not text inference. | Pipeline caches, not text KV. | Companion/device dependent. Different models sharing one media transport still serialize. |
| Existing Werk server client / external OpenAI-compatible vLLM | HTTP transport does not own model weights or impose a token scheduler. | Remote service. | Remote service. | Remote service; Werk cannot infer capacity or cache behavior from OpenAI compatibility alone. |

The upstream [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
confirms parallel decoding and continuous batching. Native vLLM capacity controls
are described in its [optimization documentation](https://docs.vllm.ai/en/latest/configuration/optimization/).
Werk's integration behavior above is based on the repository, not an assumption
that every installed backend version implements every upstream feature.

### What changed, and what was retained

- `backend/runtime_cache.rs` replaces the existing process/model maps with shared
  initialization slots. A registry mutex only publishes slots; loading holds a
  per-key mutex. Failures are retryable, failed empty slots do not consume capacity,
  and poisoning is isolated to the affected initialization slot. Monitoring takes
  nonblocking snapshots and never waits for a cold model.
- llama-server, vLLM, Candle and both legacy llama loaders now use single-flight
  initialization. Existing manifest and runtime/configuration keys remain intact.
  oMLX keeps its hard 16-worker ceiling while allowing different model loads to
  overlap. Concurrent creation of shared snapshot directories is also race-safe.
  llama-server, vLLM and oMLX share a portable startup-port reservation until
  readiness, preventing concurrent owned startups from choosing the same port.
  Their CLI socket handoff still cannot exclude unrelated external processes; an
  external bind conflict can fail startup and be retried.
- llama.cpp ordinary and extended API inference take a shared read gate. Explicit
  slot-state operations and terminal persistent restore/infer/save transactions
  take its exclusive write gate. Native scheduling remains intact.
- API session creation uses short-lived per-key gates and retains its existing
  64-entry LRU and seed behavior. Sessions wrapping dead owned workers are replaced
  on the next request. A failed inference is not automatically replayed. Remote
  health I/O never runs while the global API session cache is locked.
- Model lookup, prompt routing, tool-route probing and session creation execute on
  Tokio's blocking pool. Both streaming and nonstreaming API paths use this rule.
  Synchronous inference and blocking HTTP retain the existing blocking-pool
  architecture and bounded streaming channels.
- ONNX/Transformers transports and Burn's mutable runtime state are now per model.
  Their defaults remain conservative. No generic active-request/queue knobs were
  added: native backends keep their schedulers, and mutable runtimes keep local
  serialization. Configurable cache capacity controls residency, not agent count.

### Configuration and memory

`werk --parallel 4 --ctx-size 8192 serve` gives each llama-server worker four native
slots with a requested **8192 tokens per sequence**. Werk passes total context
capacity 32768 and native `-np 4`. The option defaults to one slot. It requires the
selected executable to advertise `--parallel`; it does not configure vLLM, Candle,
or legacy FFI context concurrency. `--ctx-size 0` still delegates sizing to the
native runtime. Raw `WERK_LLAMA_ARGS` are appended last; overriding native `-c` or
`-np` there bypasses the typed per-sequence sizing contract and API context estimate.

Increasing slots can increase KV memory substantially. Independent agents do not
require independent weights, but active sequences need independent state. Prefix
reuse is a backend optimization and not a throughput or cache-hit guarantee.
The existing named llama.cpp state/persistent-chat contract is verified only for
one slot: with `--parallel > 1` those capabilities remain unavailable. Ordinary
native prompt caching still works according to native behavior. This preserves
snapshot safety instead of claiming unverified multi-slot restore support.

For embedded ONNX/Transformers, `WERK_ONNX_GENAI_MODEL_CACHE_SIZE` and
`WERK_TRANSFORMERS_MODEL_CACHE_SIZE` now bound independent model workers. Default
`1` retains the previous one-model memory policy; set `2` or more to keep multiple
models resident and execute across them. Values remain clamped to `0..8`; `0`
keeps a single transport but disables native model retention. Burn uses
`WERK_BURN_MODEL_CACHE_SIZE` (default `1`, range `1..8`). These settings are captured
when the backend is constructed; restart Werk to change them.

The bounded cache evicts an idle least-recently-used entry before loading a new
one. Active requests, queued same-model calls and initializing models pin their
entries. If all slots are pinned, a **new distinct model** gets a capacity error;
raise the configured model capacity or retry after a request completes. This is
an explicit change from waiting behind the old shared worker. Same-model callers
continue using that model's serialized transport/mutex. There is no automatic
VRAM/RAM-based eviction policy. llama-server/vLLM/Candle/legacy resident maps retain
their previous unbounded configuration retention; changing a manifest selects a
new identity and may retain the old runtime until its owner exits. There is no
new universal model-unload endpoint. `werk cache purge` removes eligible disk
caches, not live weights.

### Deployment examples

Install/register each model first and confirm its selected backend with Werk's
existing doctor/readiness tools. Explicit backend selection binds all requests to
that backend; `auto` can mix backends according to existing manifest policy.

Single machine, one model, several agents:

```sh
werk --backend llama-cpu --parallel 4 --ctx-size 4096 serve --model MODEL_A
# Every agent uses this URL and model ID; no per-agent Werk process is needed.
```

Single machine, several models and agents:

```sh
WERK_ONNX_GENAI_MODEL_CACHE_SIZE=3 werk --backend auto --parallel 4 serve
# Requests select MODEL_A, MODEL_B or MODEL_C through the standard model field.
```

Apple Silicon: install a compatible oMLX runtime for resident MLX text and a Metal
llama-server for GGUF. `werk --backend auto --parallel 4 serve` can expose both
according to the existing selection policy. Check the selected route: plain
`mlx`/`mlx-vlm` commands remain one-shot. Memory and native MLX model compatibility
limit achievable concurrency.

CUDA workstation: with local vLLM and llama-server installed, for example:

```sh
WERK_VLLM_ARGS='--max-num-seqs 8 --gpu-memory-utilization 0.4' \
  werk --backend auto --parallel 4 --ctx-size 8192 serve
```

This passes the native vLLM budget to each vLLM process; the example is not an
automatic memory partition. Size all selected models, KV caches, CUDA contexts and
other workloads together. CPU-only ONNX plus llama-server uses the same registry
architecture; ORT and llama CPU thread pools may oversubscribe the machine.

Two DGX Spark nodes: run one `werk serve` on each node, using the same per-node
configuration approach as the workstation example and a memory budget suited to
each model. Register the required models independently on each node. Six coding
agents can route requests to either node while spending other time running tools.
Two 128-GB nodes do not become one 256-GB model cache; interconnect speed does not
change this. Native distributed vLLM execution, if configured externally, is a
separate concern. No cluster scheduler is introduced here.

### Validation and benchmark

Portable deterministic tests exercise the registry with barriers, channels,
reference counts and per-model mutexes. API tests use a single-thread Tokio
executor to prove cold preparation does not stall unrelated requests, and mix
OpenAI/Anthropic streaming and nonstreaming calls across three model IDs and six
same-model clients. Tests also verify session reuse/recreation, failure isolation,
ONNX transport separation, native slot configuration and bounded eviction.

A fake native HTTP server holds two real llama-server adapter requests until both
arrive, verifies state mutation is excluded, then releases them. The shutdown
fixture runs in an isolated process: it reaps an active child and rejects both a
blocked initializer and a queued initializer after shutdown starts. Existing
runtime-state save/restore, process interruption and companion failure tests remain.
Timeouts are watchdogs; successful overlap depends on observed events, not sleeps.

Run the standard-library benchmark against an already running node:

```sh
python3 utils/concurrency_bench.py --model MODEL_A --warmup --requests 16 \
  --concurrency 1 2 4 8 --output single-model.json
python3 utils/concurrency_bench.py --model MODEL_A --model MODEL_B --model MODEL_C \
  --warmup --requests 24 --concurrency 1 2 4 8 --output mixed-models.json
```

Set `WERK_API_KEY` for authenticated nodes. Run a separate fresh-server comparison
without `--warmup` for cold creation; keep model/backend versions, seed policy,
prompt, token budget and hardware constant. The script measures aggregate and
per-request output tokens/second, TTFT, failures and latency. It stores available
observability snapshots before/after each level for worker identities, residency
and cache diagnostics. These are not peak-memory measurements. Queue time is null
unless measured separately by native tooling; TTFT includes queueing. Repeated
prompts exercise prefix caching, while interleaved models exercise resident reuse.
Record native process count during the run to check that six clients do not cause
six resident model copies. Use `werk top`, native metrics and platform tools for
peak GPU/unified-memory data that the common API cannot supply.

### Platform and architectural limits

Validation performed here uses Linux x86-64, backend-neutral Rust tests, fake HTTP
servers and Python worker fixtures; optional Burn CPU and legacy llama feature
compilation is also checked. These tests prove Werk's synchronization behavior,
not CUDA/Metal/Vulkan throughput. No macOS, Windows, DGX Spark, real MLX, real
vLLM or multi-node hardware benchmark is claimed. The common cache/gates use
portable Rust synchronization. Process lifetime handling retains its existing
Unix/Windows implementations and existing platform limits for state persistence.

Native engines determine actual throughput, thread safety below their public
contracts, sequence limits and GPU overlap. Async API concurrency does not mean
unlimited resources: blocking tasks and backend wait queues still consume memory;
there is no bounded generic admission queue or tenant fairness policy. Disconnects
can stop supported HTTP streams but cannot universally cancel synchronous native
inference. Signal shutdown is abortive worker cleanup rather than graceful request
draining. Library embedders own their server shutdown policy. Media companions
and opaque one-shot commands retain the limitations listed in the matrix.

For a later multi-node scheduler, begin with a separate model-aware router using
node readiness/capacity telemetry, explicit per-node model placement and affinity
for useful prefix caches. Define retry/idempotency behavior for interrupted
streams first. Keep native token scheduling on each node, and do not treat a
cross-node KV transfer or pooled memory abstraction as already available.

## Changed files and validation record

The audit started from commit `00bd1914012e2ce5970f16458b2b979246d60f14`.
Implementation and validation completed on 2026-09-28. No new dependency or API
endpoint was introduced.

| Files | Purpose |
| --- | --- |
| `src/backend/runtime_cache.rs`, `src/backend/mod.rs` | Shared per-key resident cache, weak creation gates, session health contract, native parallel option. |
| `src/backend/llama_server.rs`, `vllm.rs`, `omlx.rs`, `candle.rs`, `llama_fast.rs`, `external.rs`, `onnxruntime.rs`, `burn.rs` | Integrate single-flight residency and backend-specific execution/worker isolation. |
| `src/backend/llama_server/runtime_state.rs`, `llama_server/telemetry.rs`, `omlx/telemetry.rs`, `vllm/runtime_control.rs`, `llama_server/fp4.rs` | Exclusive state operations, nonblocking registry snapshots, safe directory creation and adjusted imports. |
| `src/backend/llama_process_lifecycle.rs` | Shared native startup-port reservation and queued-creation shutdown coverage. |
| `src/api/generation.rs`, `state.rs`, `chat.rs`, `anthropic/mod.rs`, `src/runtime_control/routing.rs` | Blocking-pool preparation, single-flight API sessions, dead-session replacement and health forwarding. |
| `src/cli.rs` | `--parallel` / `WERK_LLAMA_PARALLEL` and parsing test. |
| `src/api/tests/concurrency.rs`, `src/api/tests/mod.rs`, `src/backend/omlx/tests.rs`, `llama_server/tool_call_tests.rs` | API overlap tests and existing cache fixture updates; other regression tests live alongside their implementations. |
| `utils/concurrency_bench.py` | Portable SSE concurrency benchmark, no additional packages. |
| `docs/concepts/inference-concurrency.md`, `runtime-persistence-and-memory.md`, `docs/reference/environment-variables.md`, `docs/README.md` | Audit, matrices, deployment/benchmark guidance and updated residency configuration. |

Checks executed:

- `cargo check --tests --quiet`: passed before the final port reservation addition;
  the subsequent complete library test build also checks the final Rust changes.
- `cargo check --tests --features burn-cpu,llama-fast,llama-cpp`: passed on Linux
  x86-64 (optional CPU/legacy configurations; no native model inference claim).
- `cargo test --lib runtime_cache --quiet`: 6 passed.
- `cargo test --lib api::tests::concurrency --quiet`: 4 passed.
- `cargo test --lib --quiet -- --test-threads=1` after the port fix: **1229 passed,
  0 failed, 4 ignored**, 135.81 seconds. Fifteen new tests were added and existing
  shutdown/cache tests were extended.
- `cargo test --lib backend::omlx::tests --quiet -- --test-threads=16` after the
  port fix: **62 passed**, 9.67 seconds; this exercises overlapping native starts.
- Final `cargo clippy --all-targets`: exit 0, 53 test-target warnings (43 shared
  with the library); untouched HEAD reports 54. Strict warnings-as-errors is not
  a clean baseline in this repository.
- `cargo fmt --all --check` and `git diff --check`: passed.
- `python3 utils/concurrency_bench.py --help`: passed. A synthetic local SSE server
  exercised all four levels with eight requests each, model interleaving, token
  usage and TTFT. All 32 requests passed. These figures are transport validation,
  **not hardware inference benchmarks**.
- Normal parallel test execution is not clean in this environment. The modified
  suite had intermittent file-lock failures and one native startup HTTP 401;
  that startup result prompted the additional port reservation fix. A fresh
  archive of untouched HEAD also failed parallel execution (1211 passed,
  3 failed, 4 ignored), including
  `cache::tests::native_worker_lock_blocks_cache_after_main_chat_lock_releases`
  and two existing CLI archive-lock tests. This confirms a baseline parallel
  file-lock testing problem; it does not establish that every stress failure has
  the same cause. Serial execution is the reliable complete validation here.
- Strict `cargo clippy --all-targets -- -D warnings` was attempted and failed.
  Newly introduced findings were corrected. Ordinary Clippy completes with
  baseline warnings; untouched HEAD itself reports 54 test-target warnings.

Performance expectations are limited to the synchronization changes: concurrent
cold callers no longer allocate duplicate resident weights, unrelated model loads
can proceed independently, and native llama.cpp can receive simultaneous requests.
Separate Python model workers cost extra interpreter/runtime memory and CPU thread
pools, bounded by their configured residency limits. Additional blocking-pool
handoffs may affect small-request latency; no numeric improvement is claimed
without running the supplied benchmark on actual models and target hardware.
