# Backends, routing and platform support

Werk1112 separates the stable inference contract from concrete execution
runtimes. A model can be installed and classified without every backend being
present, and one model may have several eligible runtime candidates.

This page documents the current implementation. “Installer exists” does not
mean “every model and accelerator combination is verified.”

Inference eligibility is separate from runtime-state control. The exact model
residency, prefix-state, persistence, memory, prefill/decode and
expert-residency statuses for each active adapter are in the
[runtime-control capability matrix](concepts/runtime-persistence-and-memory.md#current-production-capability-matrix).

## Three separate questions

Backend troubleshooting is easier when these questions are kept separate:

1. **Can Werk discover the runtime?**
2. **Can the runtime accept this model layout, architecture, task and explicit parameters?**
3. **Can it load and execute the model on this machine?**

A successful install answers only the first question. Model probing and
planning answer most of the second. The first cold inference is still the
definitive load and memory test.

## Residency is not named Prefill state

A faster second request can come from several different mechanisms. Capability
`runtime.model_residency` with operation `automatic_reuse` means only that the
selected backend can retain exact model weights or a media pipeline across
requests. It never implies a reusable prompt, a KV snapshot, a `state_id`, or
cross-restart restore.

| Execution path | Model/pipeline lifetime | Prompt/KV reuse | Named `/werk/v1/prefill` state |
| --- | --- | --- | --- |
| Werk-managed `llama-server` | Child process and weights can remain resident while Werk runs. | llama.cpp owns its automatic prompt cache. | Experimental, and only after the exact live process passes Werk's functional state probe. |
| In-process Candle, Burn and compiled llama.cpp adapters | Exact model entries remain in Werk-owned process caches where the concrete runtime is available. | Backend-specific; no generic cross-backend KV contract. | No. |
| Local Werk-started vLLM | The exact vLLM model process is reused. | vLLM-owned APC. `werk serve --persistence` supplies `--enable-prefix-caching` unless explicit `WERK_VLLM_ARGS` wins. | No; Werk cannot name, snapshot, restore, move or prune APC entries. |
| Remote vLLM | The remote operator owns process and model lifetime. | Opaque to Werk; the OpenAI endpoint does not prove its cache configuration. | No. |
| Local oMLX | Werk starts an installed oMLX CLI on demand and reuses its exact model/runtime process. | oMLX owns its native caches. `serve --persistence` also enables verified short exact-prefix reuse in the worker with `auto`/`disk` mode and reuse enabled. | No; native caches do not expose Werk named state, snapshots or restore. |
| Werk-owned Transformers or ONNX GenAI CPU-fallback worker | Exact model/tokenizer entries use independent bounded LRUs. | Generator, prompt and KV state are request-local. | No. |
| External ONNX runner, MLX or MLX-VLM command | One process is invoked per request; Werk has no validated resident cache. | No declared cross-request reuse. | No. |
| Generic media and managed Qwen workers | Separate bounded pipeline/model LRUs remain warm while their workers live. | Not a text KV-state contract. | No; durable media jobs are request/status/result records only. |

`werk serve --persistence` supplies defaults for omitted fields on
`POST /werk/v1/prefill` and the native APC default for a local Werk-started vLLM
process. With `auto` or `disk` mode and reuse enabled, it also enables verified
short exact-prefix caching in local oMLX workers. Model residency remains
automatic; remote vLLM, MLX and external ONNX execution remain unchanged.
Ordinary `/v1` and media calls continue through their normal inference routes.
Consult the live capability response before using optional runtime controls.

## Runtime selection

For a typed request Werk:

1. loads the model manifest;
2. resolves task and routing parameters;
3. estimates accelerator and host-memory demand;
4. probes registered runtimes;
5. rejects candidates that do not match task, format, layout, architecture,
   accelerator or strict explicit parameters;
6. scores the remaining candidates;
7. executes the highest-scoring candidate;
8. optionally retries another already accepted candidate according to the
   fallback policy.

Automatic text selection also executes the existing model/platform preference
order, with the registered priorities applied before the existing hardware
profile overrides. Availability is evaluated for the requested model and
capabilities: an importable runtime alone is not proof of compatibility.
Candle eligibility follows its implemented architecture loaders, including
format and known packed-quantization restrictions.

If a preferred eligible runtime is missing or its installed implementation
cannot accept the model, Werk selects the next compatible available runtime
and reports, for example:

~~~text
Model example/phi3: preferred runtime vLLM CUDA cannot be used (runtime unavailable). Using compatible fallback Candle CUDA.
~~~

The message contains the actual routing decision and rejection reason. It is
written to stderr without `--verbose` or `--debug`; `werk serve` logs it when
selecting a fallback or changing routes and suppresses identical repeats.
Diagnostics never become generated text, OpenAI response fields or SSE token
content. Candidates for another task or modality, such as MLX-VLM for a text
request, do not create a fallback warning. If no compatible runtime exists,
the request fails before generation with the relevant candidates and causes.
Known managed installation hints remain available in the error diagnostics.

Media retries retain the same accepted candidates and explicit constraints,
record failed attempts, and log the actual replacement route. Text generation
errors propagate without restarting a stream after text or tool-call fragments
have been emitted. `fallback_policy=none` continues to disable media execution
retries.

### MLX compatibility before loading

MLX text preflight reads local `config.json` and uses the installed loader's
architecture resolver and model argument validation in the Python environment
that will execute generation. It checks quantization support in that loader
and installed MLX core using only bounded synthetic inputs, without loading
model weights or running repository-provided model code. Missing architectures,
incompatible runtime implementations, damaged metadata and properties that
cannot be verified produce distinct actionable diagnostics. A successful
preflight does not prove that model weights fit in memory or guarantee inference
success.

`WERK_MLX_MODULE` selects a module in `WERK_MLX_PYTHON`; an explicit
`WERK_MLX_GENERATE` selects its executable when no module override is supplied.
With only `WERK_MLX_PYTHON`, Werk uses that interpreter's default generation
module, so a PATH launcher cannot silently switch environments. Recognized
Python console launchers are probed through their own interpreter. For opaque
custom launchers whose model resolver cannot be verified, diagnostics explain
the limitation. Existing Werk Gemma4 compatibility metadata is checked against
the installed loader as well. Model probe results are not globally cached.

A model such as `Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX` with
`model_type=deepseek_v4` and mixed MXFP4/MXFP8 quantization is rejected when the
selected installed `mlx-lm` cannot resolve that architecture or support that
layout. That candidate's rejection does not rule out the separate oMLX runtime
described below. Regression fixtures simulate compatibility outcomes; they do
not establish successful inference of the actual DeepSeek checkpoint.

Use diagnostics before a large request:

~~~bash
werk inspect MODEL
werk doctor --model MODEL --task TASK --debug
werk backend list
werk backend doctor --debug
~~~

Qwen-TTS is currently an architecture adapter behind the media companion, not
a separately rendered runtime row. Its managed environment can be installed,
but <code>backend list</code> and <code>backend doctor</code> do not yet print a
complete Qwen-specific status block. A model-specific doctor run and companion
diagnostics report the missing or selected Qwen interpreter.

The media commands additionally expose the resolved request and complete
candidate decision:

~~~bash
werk audio generate speech TTS_MODEL \
  --text "Diagnostic test." \
  --backend auto --verbose --debug
~~~

## Automatic and explicit selection

The default backend value is <code>auto</code>. It does not mean that an
arbitrary installed runtime will be tried. Only candidates accepted for the
specific task and model participate.

A concrete backend increases the score of matching candidates. For typed media
requests it becomes a hard backend constraint when
<code>fallback_policy=none</code>. A concrete accelerator or device remains a
hard target constraint independently of backend retry.

For a strict reproducibility test use all relevant constraints:

~~~bash
werk video generate VIDEO_MODEL \
  --prompt "A short camera movement" \
  --backend auto \
  --accelerator cuda \
  --fallback-policy none \
  --precision bf16 \
  --verbose --debug
~~~

## Optional local oMLX backend

On Apple Silicon, `auto` considers an installed oMLX after MLX-LM for compatible
MLX and Hugging Face safetensors text models. Requests stay on MLX-LM when its loader supports the model. If its actual loader cannot
accept a model but oMLX's loader can, Werk
selects oMLX and reports the MLX-LM rejection and compatible fallback on stderr.
`--backend omlx` binds execution to oMLX; `--backend mlx` retains its existing
MLX-LM/MLX-VLM meaning.

~~~bash
werk --backend auto run MODEL "Hello"
werk --backend omlx run MODEL "Hello"
werk --backend omlx serve
werk backend doctor --debug
werk doctor --model MODEL --task text-generation --debug
werk --backend omlx doctor --model MODEL --task text-generation
~~~

Install the [upstream oMLX CLI](https://github.com/jundot/omlx#install) separately.
Werk checks `WERK_OMLX_BIN` first, then `omlx` on `PATH` when that override is
unset. An empty or invalid override fails discovery. The macOS application
alone does not install that CLI. Werk does not install or upgrade oMLX,
change the normal MLX-LM environment, manage a
system service, or connect to an external oMLX server. `omlx` is not a
`werk backend install` target, including when automatic provisioning is enabled.

Werk captures the executable and its Python environment together. Recognized
Python console entry points can be verified; opaque custom wrappers fail with
an explanation. The same captured installation is used for model preflight and
execution. The integration is based on upstream v0.6.4 loader and API behavior;
a package version string alone is not proof of model compatibility.
Inherited upstream `OMLX_*` settings are excluded from the private worker;
see the [environment-variable reference](reference/environment-variables.md)
for the supported overrides.

An independent bounded probe activates installed oMLX pre-load patches, resolves
the model architecture and validates metadata and quantization support without
constructing the model, loading weights or executing model-repository Python.
Mixed MXFP4/MXFP8 layouts are preserved. This matters for DeepSeek V4 because
oMLX provides architecture and loader patches absent from some vanilla MLX-LM
installations. Unverifiable loader contracts remain explicit compatibility
failures. Safetensors headers with raw `F8_E8M0` tensors are rejected because the
upstream loader can temporarily rewrite those files; use an already converted
MLX checkpoint. Werk does not convert or modify model files.

Only after selection does Werk start an oMLX child on loopback with a free port,
the actual local model directory and an isolated `--base-path`. It verifies the
physical model path advertised by `/v1/models/status`, then loads that exact
model before generation. Startup and loading share a 900-second default
deadline; `WERK_OMLX_HEALTH_TIMEOUT_SECONDS` accepts a positive integer override.
Invalid values fail discovery. The metadata probe has a separate fixed
20-second limit, and this setting does not change generation timeouts.
No duplicate weight download or model-directory copy is required. Exact model
and runtime processes are reused while their backend lives and stopped when
Werk releases them. A parent-lifetime pipe also stops the worker if Werk exits
without normal cleanup. Failed startup and load attempts retain their diagnostics.

Text, chat, streaming and native tool calls use `/v1/chat/completions`. Native
parsing uses a compatible installed parser; other tool requests use Werk's
generic tool protocol instead of being rejected by model capability checks. Verified paths include DeepSeek V4's
native DSML parser/template and Qwen `qwen4_exp` text offload with the installed
`qwen3_coder` XML parser, plus GLM `glm5_next` with the installed `glm47`
`<arg_key>`/`<arg_value>` parser. Both text adapters check the actual local
template, tokenizer-loader wiring and typed argument parser without loading
model weights. Explicit incompatible parser/template overrides remain rejected. The verified oMLX API supports
omitted, `auto` and `none` tool choice. Required/named choices and explicit
`parallel_tool_calls` use the generic protocol, which validates returned calls.
Function definitions with `strict: true` remain unsupported because neither
this upstream API nor the generic protocol guarantees constrained schema decoding.
Omit `strict` or set it to `false`. Image, embedding and media requests are
outside this adapter's scope. Separate upstream reasoning fields are not
rendered as answer text; empty reasoning-only results are errors. Errors after
text or tool-call fragments have been emitted never trigger another generation.
oMLX's internal caches do not imply Werk named Prefill, KV snapshot or restore
support.

On oMLX 0.6.4, `werk serve --persistence` enables the same verified short
exact-prefix helper used by persistent CLI chat. It applies with persistence
mode `auto` or `disk` and reuse other than `disabled`, including when `auto`
routing selects oMLX. Configuration is applied before model preparation and
covers normal chat, supported tool requests and request-specific oMLX options.
Clients send their ordinary messages; no conversation ID or cache metadata is
required. Reuse requires matching token prefixes.
Startup reports whether the native short-prefix cache is active. Unsupported
runtime versions or cache types continue with the already loaded worker and
ordinary oMLX caching.

The server helper stores native SSD cache data under the private worker's
`cache/prefix-cache` directory with a 4 GB native cache limit. It lasts for that
worker and is included in managed worker cache inventory and cleanup. It does
not save conversation history or promise reuse after a server restart. Modes
`memory` and `ephemeral`, or reuse `disabled`, leave this additional helper off.
TTL, pinning and `required` reuse retain their named Prefill policy meaning;
ordinary OpenAI chat requests with a cold cache still perform normal prefill.

`werk chat --persistence` separately saves portable conversation messages for
every backend. On oMLX 0.6.4 it also enables an optional durable native SSD
prefix cache in a private namespace tied to the exact checkpoint, tokenizer,
runtime and chat settings. The worker is still stopped on chat exit; compatible native
cache files remain for the next invocation. Werk's additional exact-prefix
index covers short prefixes up to the native block size (typically 2048 tokens
for DeepSeek V4, capped at 8192); longer inputs retain ordinary oMLX caching.
Cache misses, unsupported cache types or failed writes fall back to normal
prefill. The sampler and token streaming stay on the normal oMLX path. Verbose
output reports `cached prompt tokens` when the runtime supplies the count.
This does not expose named Werk Prefill/KV-state handles or change their
capability status.

A successful metadata probe is not a memory-capacity or inference guarantee.
Validate a small model on the actual Apple Silicon installation first, then
test a large DeepSeek checkpoint only with sufficient available memory for the
selected loading mode.

### Experimental oMLX expert offload

Without offload settings, Werk preserves native oMLX loading.
For supported DeepSeek V4 affine checkpoints on oMLX 0.6.4, explicitly setting
`WERK_OMLX_EXPERT_CACHE_MB=auto` selects an automatic SSD-backed expert cache.
Both `chat` and `serve` use the same selection: at worker startup,
retain at most the model's expert bytes and the space left by current available
system memory and the effective Metal cap, reserving base weights, workspace,
and system headroom. Native admission further reduces residency for KV/attention
work. There is no fixed 24 GiB budget or 1 TiB cap on automatic selection.
On unsupported architectures/runtime versions, auto leaves the native loader
in place. `WERK_OMLX_EXPERT_CACHE_MB=0` explicitly selects native loading.

Positive MiB values remain explicit ceilings, including small-machine budgets.
For example, an 8 GiB expert cache is selected with:

~~~bash
WERK_OMLX_EXPERT_CACHE_MB=8192 \
  werk --backend omlx chat mlx-community/DeepSeek-V4-Flash-2bit-DQ --verbose
~~~

Routed expert weights remain backed by the existing local checkpoint and are
read into a bounded cache when needed. Dense and shared weights, attention,
the KV cache and temporary computation buffers still need memory in addition
to this budget. This setting is an expert-cache limit, not a total-process
memory limit or a guarantee that every MoE checkpoint fits. It does not raise
the system's Metal limit. Model layout and the installed runtime must pass the
offload checks; unsupported layouts fail explicitly.

Expert offload uses `grouped` execution by default. It materializes the nine
weight, scale and bias tensors for one expert in a single MLX evaluation,
holds active expert groups in the cache until their GPU computation finishes,
and evaluates outputs in bounded groups. The cache budget still constrains
which experts can be retained. Temporary allocator storage is recycled between
loads; the grouped path clears it when retained allocator memory exceeds
64 MiB. This temporary storage belongs to the additional workspace allowance.

For a controlled comparison, select the serial execution path before starting
Werk:

~~~bash
WERK_OMLX_EXPERT_EXECUTION=serial \
WERK_OMLX_EXPERT_CACHE_MB=8192 \
  werk --backend omlx chat mlx-community/DeepSeek-V4-Flash-2bit-DQ --verbose
~~~

`WERK_OMLX_EXPERT_EXECUTION` accepts `grouped` or `serial`; omission selects
`grouped`. Changing execution mode changes the worker configuration and the
durable CLI prefix-cache namespace. Keep cache budget, prompts and sampling
settings identical when comparing modes. This control applies to expert
offload; the expert cache budget remains separate from native prompt/KV caching.

With `--verbose` or `--debug`, generation diagnostics include the explicitly
sent sampling and thinking controls and, when available, expert counters from
the worker before and after generation. These cover cache hits, misses and
evictions, logical bytes read, tensor materializations, output evaluations,
allocator clears and selected timing counters. `disk_bytes_read` measures
logical `pread` bytes, which can be served by the OS file cache; it does not
measure physical SSD traffic. Counters describe the whole worker during the
measurement interval, so overlapping requests or expert actions may contribute.
Timing counters can contain overlapping work and must not be summed as
independent request phases. Missing diagnostics do not indicate a zero count.

Use the [HTTP chat benchmark](../utils/benchmarks/README.md) to compare repeated
prompts and multi-turn conversations through the API used by Open WebUI and
werkStation. Review generated answers alongside latency and token counts.

The initial implementation supports oMLX **0.6.4** and sanitized DeepSeek V4
checkpoints with stacked affine expert tensors. It rejects custom quantization
loaders, embedded speculative drafters and unsupported layouts before admitting
the smaller working set. The ordinary, non-offloaded adapter retains its
existing runtime compatibility checks.

### Qwen and GLM text offload

The additional private oMLX 0.6.4 text adapters support stacked affine experts
for `qwen4_exp` and `glm5_next`, preserving each installed architecture's
activation, routing, normalization and cache implementation. They do not execute
checkpoint Python files. Mixed 2/4/8-bit projections retain their own metadata;
vision and MTP tensors are excluded from this text path. Qwen/GLM share the bounded grouping control with DeepSeek: `grouped` batches
evaluation while `serial` retains the reference order. Tiny budgets reduce the
group size automatically; no active expert may be evicted.

The Vontra GLM-5.3-Flash oQ2 checkpoint requires two additional loader details:
removing its vision child through MLX's attribute API before native sanitization,
and mapping forget-gate weights, affine scales/biases and per-module quantization
together into the native `forget_gate` namespace. The tiny regression uses the
checkpoint's flat Q8 forget-gate layout as well as a vision configuration.

Its supplied GLM chat template always starts a reasoning block and does not
consume `enable_thinking`. Consequently, `WERK_OMLX_THINKING=0` is a request
that this template does not honor; allow enough generation tokens for reasoning
and the final answer. This limitation also applies to the equivalent node/API
thinking setting. Use `WERK_OMLX_REASONING_EFFORT=low` (also `high` or `max`)
or API `werk.omlx.reasoning_effort` to request less reasoning. Omission preserves
the native default; low effort does not guarantee zero reasoning. Both Text
Config nodes expose this independently as **oMLX Reasoning Effort**.

GLM linear attention fuses input projections at load time. The original
projection modules now reference row views of those same native allocations,
so packed weights and quantization metadata occupy RAM once. On the installed
Vontra checkpoint this removes 3.43 GiB of duplicate weights and leaves more
room for Auto expert residency. The unfused fallback still sees identical
projection values. Mixed quantization that cannot fuse keeps its native path.
The status field `attention_fusion_bytes` measures fused storage and
`attention_shared_bytes` records how much is shared with original modules;
shared bytes are included once in base memory. Memory guards remain active;
explicit expert budgets remain supported.

For a single decoding request with no waiting requests or interleaved prefills,
Werk revisits the shared cache budget after native decode responses. This lets
expert capacity recover after temporary prefill allocations are released,
within the measured native memory limits and configured upper budget.
The expert status endpoint reports this as `last_decode_admission`.

GLM, Qwen and DeepSeek expert offload use a segmented LRU: repeated accesses protect hot experts
within at most 80% of the existing expert budget, leaving a recency-adaptive
region for new demand. Parallel file readers fetch missing tensors of each
bounded expert group: Qwen/GLM use the logical CPU count capped at 16 (four
when unavailable), while DeepSeek retains four readers. Threads start on
demand and the bounded staging allocation does not grow with thread count.
Pins, leases and native memory limits remain
authoritative; all MLX operations stay on the owning executor. These are
automatic adapter choices, including when the expert budget is `auto`.
DeepSeek retains its native BF16/FP16 expert metadata conversion and uses the
shared bounded reader in grouped execution; owned read buffers also avoid its
previous intermediate bytearray copy. No new CLI, ComfyUI or n8n option is
required. See the [Qwen/DeepSeek comparison](benchmarks/2026-09-12-flash-offload/README.md#qwen--deepseek-geschützter-experten-cache-2026-09-15)
for measurements at an unchanged 20-GiB expert budget.
They do not enable MTP. See the [GLM decode measurements and design](glm-decode-optimization.md).

Native GLM tool parsing uses the verified `glm47` parser described above.
The text adapter does not enable vision or MTP.

For the verified Qwen/GLM text adapters, automatic expert selection first checks
whether all expert weights fit alongside base weights, auxiliary caches and the
existing system/workspace reserves. If they fit, Werk keeps the installed native
expert modules and reports `native resident execution`; their weights are charged
as base memory rather than evictable cache. If automatic Qwen N-gram tables also
fit, they remain resident. Otherwise the row cache remains active independently.
Positive expert-cache limits always retain offload. Selection happens once at
worker startup; native KV/attention admission still applies to later requests.

The offloaded Qwen/GLM single-token path reuses the shared input directly and
combines each leased group's outputs before scattering them, avoiding repeated
input gathers, per-expert output scatters and redundant synchronization. It does
not allocate a second packed copy of expert weights. Prefill retains its bounded
chunked path. See the [local comparison](benchmarks/2026-09-19-omlx-comparison/README.md).

For mixed cached/missing expert groups during Qwen/GLM decode, bounded parallel
reads now start before evaluating cached experts. GPU work stays on the owning
executor and every routed expert stays leased through its completed evaluation.
The same staging ceiling applies; small-budget groups and groups without both
hits and misses retain the ordinary path. Exceptions drain all readers before
releasing temporary storage. DeepSeek retains its separate execution path.
With overlap, `disk_read_seconds` counts foreground scheduling and waiting,
excluding cached-expert computation performed while reads run; it is not total
background I/O duration. Compare full request/decode time to assess speedup.

Qwen PLE N-gram tables have a separate row cache, controlled through
`WERK_OMLX_NGRAM_CACHE_MB` or API `werk.omlx.ngram_cache_mb`:

- Unset/`0`: native resident tables.
- Explicit `auto`: resident tables when the automatic resident selection above fits; otherwise demand-driven row caching, starting at at most 64 MiB. After demand evictions, the target can double at request/prefill boundaries and isolated decode budget checks up to a device- and model-dependent ceiling. With streamed experts, N-grams receive at most one eighth of shared cache room; both caches remain subject to native memory admission. Auto also works with native experts, whose full weights are then charged as base memory.
- API `ngram_cache_mb: "auto"`: explicitly override a fixed server setting with Auto. Omission inherits.
- Positive integer: manual upper bound in MiB; the API accepts 1–1048576.
- `0`: resident tables. It does not disable expert offload.

Both caches share the native memory limit and leave room for base weights,
attention, recurrent states and workspace. Changing either request budget
selects a separate worker. Verbose diagnostics expose separate N-gram row,
hit/miss/eviction and residency counters; bytes read are logical file reads.

```sh
WERK_OMLX_EXPERT_CACHE_MB=8192 WERK_OMLX_NGRAM_CACHE_MB=1024 \
WERK_OMLX_THINKING=0 werk --backend omlx chat \
  pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit --verbose
```

The checked DeepSeek V4 and Vontra GLM-5.3-Flash tensor inventories contain no
N-gram tables. For them this feature is not applicable; their independent MoE
offload remains available. An explicit positive N-gram budget on an unsupported
layout is rejected, rather than ignored. ComfyUI and n8n expose the same controls.

Native exact-prefix persistence now includes installed Qwen Arrays/QSA and GLM
hybrid cache states. Persistent Qwen/GLM workers use complete embedded native
snapshots because oMLX 0.6.4 cannot commit short exact prefixes with split GDN
sidecars. Cache entries remain managed by `werk cache` and their existing locks.

Validation includes tiny installed models with forced eviction, joint and
independent offload, ten decode steps, exact continuation after SSD-index restart,
and real private workers producing complete HTTP streams. Full checkpoint
acceptance and measured limitations are tracked in the
[implementation report](qwen-glm-offload-plan.md); tiny tests alone are not a
large-model quality or speed claim. Native tool parsing requires the separate
verified tokenizer/parser path; vision is not enabled by these text adapters.

For ComfyUI, n8n or another HTTP client, the same variable supplies a default
for the persistent Werk service:

~~~bash
WERK_OMLX_EXPERT_CACHE_MB=8192 werk --backend omlx serve
~~~

Alternatively, configure expert offload and its cache budget in the ComfyUI
**WERK Text Config** node or n8n **WERK Text → Chat Options**.
The nodes send `werk.omlx.expert_cache_mb` on the chat request: a positive
integer enables offload, `0` disables it, and omission inherits the server
default. **oMLX Thinking** similarly overrides the server's thinking default
for that request. Thinking changes reuse the worker; different expert budgets
use separate worker configurations and may require loading the model again.
These options work with `auto` or `omlx` server routing and still require a
compatible runtime and model. The [chat API contract](api.md#omlx-chat-options)
describes validation and the capability check used by both integrations.

Use the normal generation path to load the chosen model first. Expert telemetry
and actions do not start a model. After the worker confirms that offload is
active, `runtime.experts.residency` reports `experimental`; expert requests must
include `allow_experimental: true`. Existing ComfyUI and n8n runtime nodes
already expose this opt-in and the required fields.

When a model has been used with several cache configurations, subsequent
expert requests target the worker most recently used for that model. In
ComfyUI, connect **Werk Text Generate**'s `model_id` output to the expert
node's model input to ensure generation runs first.

The existing expert tiers describe oMLX's two locations: `external` means
checkpoint-backed outside the active expert cache, and `ram` means present in
the Apple unified-memory cache. This tier label is separate from the capability
status `externally_managed`. Prefetch accepts `ram`; `vram` is rejected because
this adapter has one shared memory pool. Pin and unpin control cache retention;
evict releases a cached expert while keeping its checkpoint source available
for later inference. Dry-run previews do not perform those cache operations.
Pins and prefetches remain constrained by the configured cache budget.

On 2026-09-10, the local `mlx-community/DeepSeek-V4-Flash-2bit-DQ` checkpoint
completed a real 16-token request through Werk and oMLX with an 8 GiB expert
cache. The API test also completed dry-run, prefetch, pin, unpin and eviction.
Sampled peak RSS across the test process tree was approximately 12.3 GiB;
the request took about 19.4 seconds including preparation. This is a short
functional smoke test, not a long-context, quality or throughput benchmark.
RSS is not a complete measurement of all system or GPU memory use.

Automated coverage uses installed-loader source fixtures, synthetic quantization
inputs and local mock HTTP workers. It checks routing, read-only preflight,
process reuse/teardown, tool calls and stream failures. Run it with
`python -m unittest discover -s src/backend -p 'test_*mlx*.py'` and
`cargo test --no-default-features`. Optional numerical tests run with
`WERK_TEST_MLX_EXPERTS=1` using the installed oMLX Python interpreter and
`python -m unittest discover -s src/backend -p 'test_omlx_experts.py'`.
These compare streamed expert outputs and a complete tiny DeepSeek model
against resident MLX computations on Metal. The Vontra checkpoint and long
contexts remain unvalidated.

## vLLM launch arguments and tool calling

`WERK_VLLM_ARGS` supplies advanced arguments only to a vLLM process that Werk
starts locally. It uses POSIX shell-word quoting to construct a direct process
argument vector; it is not evaluated by a shell. Command substitution,
environment-variable expansion, tilde expansion and globbing therefore never
occur. Malformed quoting, a trailing unescaped backslash and non-UTF-8 values
fail before process creation.

For example, after importing or pulling a compatible model as
`qwen3-coder`, a local launch can be configured as follows. The global backend
option belongs before the `serve` subcommand:

~~~bash
export WERK_API_KEY="replace-with-generated-key"
export WERK_VLLM_ARGS="--quantization compressed-tensors --kv-cache-dtype fp8 --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}' --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-num-seqs 16"

werk --backend vllm serve --model qwen3-coder
~~~

To let the server supply vLLM's native APC default as well as omitted Werk
Prefill-policy defaults, add the serve option:

~~~bash
werk --backend vllm serve --model qwen3-coder --persistence
~~~

This adds `--enable-prefix-caching` only to a local process that Werk starts and
only when `WERK_VLLM_ARGS` does not already select the enable or disable form.
It is not a named-state implementation and does not configure a remote vLLM
endpoint.

The exact parser name and flags are examples for a compatible vLLM/model
combination, not Werk defaults. Verify them against the installed vLLM version
and the selected model. vLLM generally requires `--enable-auto-tool-choice`
when `tool_choice` is `auto`, together with a compatible tool-call parser. Werk
does not validate, replace or rewrite arbitrary parser names, and does not add
that enable flag automatically.

Conceptually, Werk's effective local child argv is one of these forms, with a
resolved model directory and an internally selected loopback port:

~~~text
$WERK_VLLM_PYTHON -m vllm.entrypoints.openai.api_server --model RESOLVED_MODEL_DIR --host 127.0.0.1 --port INTERNAL_PORT --served-model-name qwen3-coder [WERK_VLLM_ARGS...]
vllm serve RESOLVED_MODEL_DIR --host 127.0.0.1 --port INTERNAL_PORT --served-model-name qwen3-coder [WERK_VLLM_ARGS...]
~~~

Werk owns `--model`, `--host`, `--port` and `--served-model-name`. Supplying
any of them in separate or `--flag=value` form through `WERK_VLLM_ARGS` is an
error. Repeated non-reserved flags, JSON values, embedded spaces and quoted
empty arguments remain distinct argv elements and retain their order.

For an already running remote vLLM endpoint, configure these process arguments
where that server is launched. A nonempty `WERK_VLLM_ARGS` is rejected when
Werk uses `WERK_VLLM_HOST` and `WERK_VLLM_PORT`; it is never silently ignored.

`POST /v1/chat/completions` supports OpenAI function-tool requests through the
vLLM adapter, for both Werk-started and remote vLLM servers. Werk forwards the
tool definitions, `tool_choice`, `parallel_tool_calls`, assistant tool calls and
tool-result messages to vLLM without translating their contents. It likewise
preserves structured tool calls in normal and streaming responses. Werk does
not execute tools or select a vLLM tool parser for the operator.

All implemented chat/vision adapters now expose tool calling. Native llama.cpp
and vLLM transports preserve the request and response fields. oMLX selects
native parsing or the generic protocol according to runtime support. Other
adapters use Werk's generic protocol with validation of generated function
calls. Automatic routing can therefore choose any compatible adapter, while
an explicit `--backend vllm` binding still remains strict. Merely setting
`WERK_VLLM_ARGS` does not activate or prefer vLLM.

Pure media runtimes are exposed as executable function tools using the same
inference planner and job service. See [Tool calling](tool-calling.md) for the
backend matrix, generic protocol limits and media tool execution API.

These guarantees cover Werk's argv construction and HTTP transport. Actual
tool-call quality, model support, vLLM-version compatibility, quantization,
speculative decoding and accelerator compatibility still require a live
runtime test before relying on a combination in production.

## Vision-language routing

An image attached to `werk run`, `werk chat` or
`POST /v1/chat/completions` changes runtime eligibility. The model manifest must
advertise image input and `image-understanding`; a text-only model is rejected
even if an installed backend can execute some other VLM.

| Runtime route | Current eligible model shape | Additional requirement |
| --- | --- | --- |
| Persistent llama.cpp server | Compatible VLM GGUF | Exactly one safe local projector GGUF listed in the manifest; its filename contains `mmproj` or `projector`, and `llama-server --help` advertises `--mmproj` |
| vLLM | Transformers safetensors with exact architecture `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`, `qwen3_vl_moe`, `glm4v` or `glm4v_moe` | Compatible installed vLLM version and model-specific processor; local or explicitly configured remote endpoint |
| MLX-VLM | MLX or safetensors `gemma4_unified` repository | Importable `mlx-vlm` environment on Apple Silicon |
| Candle | None | The in-process Candle adapter is currently text-only |

vLLM is optional and is not what makes a model visual. The vision encoder,
projector and preprocessing belong to the checkpoint/runtime implementation.
The primary non-vLLM path is llama.cpp plus the model's matching multimodal
projector. Model weights remain resident in persistent llama.cpp/vLLM server
processes, although cache details and image-embedding reuse remain
backend-specific.

Werk preserves ordered text/image parts and the image `detail` hint for the
llama.cpp and vLLM chat transports. The MLX-VLM subprocess currently receives
the prompt and image list but not arbitrary interleaving or `detail` semantics.
See [Vision and visual quality assurance](integrations/vision.md) for API and
CLI examples, body limits and the rendered HTML/slide inspection workflow.

## Fallback policies

| Policy | Candidate behavior | Quality or memory adjustment |
| --- | --- | --- |
| <code>none</code> | Execute only the selected runtime. A requested backend is a hard filter. | No inherited degradation. Explicitly requested offload still remains explicit. |
| <code>backend</code> | Default. Retry another already accepted runtime if execution or model loading fails. | No automatic inherited degradation. |
| <code>degrade</code> | Retry accepted runtimes and permit registered memory-saving adjustments. | May enable allowed offload or media tiling/windowing under memory pressure. |

Fallback never silently changes the model ID. Suggested lower resolutions,
shorter clips or smaller models are diagnostic recommendations, not automatic
request rewrites.

An unavailable architecture-specific runtime should produce:

- a rejected candidate and reason in debug output;
- an installation or configuration hint when Werk knows one;
- another accepted route only when policy and accelerator constraints permit it.

Media probes expose the same decision as a structured `task_readiness` value:
`available`, `fallback_available`, `installable`, `not_implemented`, or
`unavailable`. A concrete install command is shown only when the registered
adapter supplied that command. For example, a supported Qwen3-TTS VoiceDesign
model can recommend `werk backend install qwen-tts`; a task or model variant
without an implemented adapter explicitly says so and does not invent a pip or
Werk install command. `werk doctor --model MODEL --task TASK`, `--debug`,
`werk parameters MODEL --json`, and the HTTP discovery routes expose this same
status.

### Missing-backend negative smoke test

The repository includes a non-destructive test for the `installable` case:

~~~bash
./scripts/test-missing-media-backend.sh
~~~

It creates a metadata-only Qwen3-TTS VoiceDesign fixture and a temporary,
isolated model store, disables automatic backend provisioning, and verifies
both discovery and execution preflight. It never removes or changes the real
model store or an installed Qwen backend. To test a binary built from the
current checkout instead of the `werk` on `PATH`, select it explicitly:

~~~bash
WERK_BIN=./target/debug/werk ./scripts/test-missing-media-backend.sh
~~~

The relevant output is:

~~~text
Task readiness: installable
  Adapter: qwen3_tts_voice_design
  Required backend: qwen-tts
  Recommendation: werk backend install qwen-tts
...
Recommendation: run `werk backend install qwen-tts`; no compatible fallback was verified
PASS: missing managed backend was detected before inference and no output was created.
~~~

A missing required backend is a preflight failure, not a warning attached to a
successful inference. Werk reports `fallback_available` instead only when a
different runtime for the same model and task actually passed its probe and
the request policy permits that route.

## Parameter policy

The parameter policy is independent of backend fallback:

| Policy | Unsupported explicit parameter |
| --- | --- |
| <code>strict</code> | Reject the runtime/request. This is the default. |
| <code>warn</code> | Continue only where the resolver/adapter can safely do so and report a warning. |
| <code>permissive</code> | Allow the broadest adapter behavior, still subject to runtime validation. |

Use <code>strict</code> for production and compatibility testing. It prevents a
voice, sampler, offload or quality option from being silently ignored.

## Backend commands

The current command surface is:

~~~text
werk backend install TARGET
werk backend list
werk backend doctor [--debug]
~~~

The install targets are:

~~~text
llama-cuda
llama-cuda-nvfp4
llama-cuda-offload
llama-rocm
llama-vulkan
llama-metal
llama-cpu
onnx-cuda
onnx-rocm
onnx-cpu
vllm
qwen-tts
~~~

There is currently no <code>werk backend uninstall</code> command. See
[Uninstall and cleanup](#uninstall-and-cleanup).

## What each managed installer does

| Target | Provisioning behavior | Main prerequisites | Validation |
| --- | --- | --- | --- |
| <code>llama-cuda</code> | Shallow-clones current llama.cpp and builds llama-server with CMake and GGML CUDA. | Git, CMake, C/C++ compiler, NVIDIA driver and CUDA toolkit. | llama-server help plus known CUDA initialization failures. |
| <code>llama-cuda-nvfp4</code> | Builds pinned llama.cpp with native-first NVFP4 dispatch and standalone Marlin CUDA kernels; WERK retains persistence. | CUDA toolchain; native Blackwell requires architecture-specific kernels. | Device/build capability probe and hashed build receipt; see [NVFP4 support](nvfp4.md) for conversion, limits and overrides. |
| <code>llama-cuda-offload</code> | Builds a pinned experimental CUDA expert-cache fork through the same installer and selects it for the existing CUDA adapter. | Same CUDA toolchain; see the limits below. | CUDA initialization and advertised expert/lazy-row/slot controls; installation does not certify a model. |
| <code>llama-rocm</code> | Builds llama-server with GGML HIP. | Git, CMake, C/C++ compiler and compatible ROCm/HIP toolchain. | Executable help; a real HIP inference is not part of installation validation. |
| <code>llama-vulkan</code> | Builds llama-server with GGML Vulkan. | Git, CMake, C/C++ compiler and Vulkan development SDK. | Executable help; a real Vulkan inference is not part of installation validation. |
| <code>llama-metal</code> | Builds llama-server with GGML Metal. | macOS, Xcode command-line tools and CMake. | Rejected before build outside macOS; executable help after build. |
| <code>llama-cpu</code> | Builds the default CPU llama-server. | Git, CMake and a C/C++ compiler. | Executable help. |
| <code>onnx-cuda</code> | Copies an existing platform-specific Werk ONNX runner bundle. | A compatible bundled or explicitly configured runner. | Runner help only. |
| <code>onnx-rocm</code> | Copies an existing platform-specific Werk ONNX runner bundle. | A compatible bundled or explicitly configured runner. | Runner help only. |
| <code>onnx-cpu</code> | Copies an existing platform-specific Werk ONNX runner bundle. | A compatible bundled or explicitly configured runner. | Runner help only. |
| <code>vllm</code> | Creates an isolated virtual environment and installs vLLM with pip on eligible generic Linux hosts. On DGX Spark and AMD Strix Halo this target stops with platform-specific container/environment guidance instead of installing an unverified generic wheel. | Native Linux x86_64, Python/venv, pip, compatible PyTorch and accelerator stack. | Import/version and runtime health checks. |
| <code>qwen-tts</code> | Creates an isolated virtual environment and installs exactly qwen-tts 0.1.1. | Python 3.9+, venv, pip and platform-compatible PyTorch/audio dependencies. | Exact package version and Qwen3TTSModel import. |

### Experimental CUDA expert offload

`werk backend install llama-cuda-offload` uses the existing llama.cpp installer,
process adapter, streaming chat and persistence path. It builds
[GenerelSchwerz/llama.cpp](https://github.com/GenerelSchwerz/llama.cpp/blob/907a73da9a149faa8c42ccde890f1d575586810f/docs/fork-features.md)
at `907a73da9a149faa8c42ccde890f1d575586810f`. The fork supplies the native CUDA
expert kernels and cache; the ordinary upstream build does not expose these
controls. Werk applies one loading change: when expert caching is enabled,
disable eager mmap prefetch so startup does not fault in the whole expert file.

The source and build live below `backends/llama-cuda/offload-<revision>/`.
The existing CUDA discovery pointer selects the resulting executable after
validation. `werk backend install llama-cuda` selects the normal build again.
An explicit `WERK_LLAMA_SERVER_CUDA` or `WERK_LLAMA_SERVER` still takes precedence.

After installing Werk normally and building the optional runtime:

```bash
werk backend install llama-cuda-offload

WERK_LLAMA_ARGS='--moe-expert-cache-size 8 --moe-expert-cache-host-pinned-mb 512 --load-mode mmap --lazy-mode on --fit off --no-warmup --reasoning off' \
werk --backend cuda --ctx-size 4096 --ubatch-size 64 \
  chat ggml-org/DeepSeek-V4-Flash-GGUF \
  --verbose --persistence --session auto-test
```

Use the installed ID for the complete selected quantization; both GGUF shards
must be present. Under WSL, keep the weights on the Linux filesystem for this
test. Use `WERK_LLAMA_LOG=1` to expose the native expert hit/miss/eviction logs.

- `8` is the number of resident expert slabs **per expert tensor**, not MiB and
  not a process-wide automatic budget. Native metadata, scratch space, dense
  weights and KV memory consume additional VRAM. Increase it only after measuring
  available VRAM. `512` bounds the model's pinned host source/staging allocation;
  the OS file cache remains separately managed and reclaimable.
- `--lazy-mode on` reads supported PLE/N-gram table rows on demand. It uses the
  native mmap/file-cache path, not oMLX's separately bounded row LRU. Models
  without such tensors, including the inspected DeepSeek V4 architecture, have
  no N-gram weights to offload. This is unrelated to speculative N-gram decoding.
- The expert cache supports layer split. Tensor split is rejected by the fork.
  Quantization, auxiliary tensors and platform CUDA behavior remain native
  compatibility constraints.
- Persistent chat snapshots reuse the existing private slot machinery. A
  restored conversation alone is not evidence of KV reuse: check a nonzero
  `prompt cached count` after restarting the same session. Model weights and
  hot VRAM expert slots are loaded again on restart.
- This runtime is experimental. The WSL/RTX 3090 native suite passed cached
  matmul/prefill/overflow and pageable staging cases but failed a host-registration
  boundary fixture (`auxiliary_alias_not_identity`). A complete DeepSeek Q2_K_S
  end-to-end offload benchmark has not been verified. Do not infer full macOS
  oMLX parity or support for every MoE model from the available controls.

Important limitations:

- Ordinary llama.cpp provisioning follows the current upstream default branch rather
  than a Werk-pinned commit, so identical Werk versions can build different
  upstream revisions at different times;
- the ONNX installers do not download or build a runner today;
- ONNX installation verifies the executable, not the requested CUDA/ROCm
  execution provider;
- successful Python import does not prove that a large model fits or that the
  requested GPU kernel is available.

## Platform support matrix

The table distinguishes practical primary support from best-effort or
upstream-unconfirmed paths.

| Target | Native Linux x86_64 | AMD Strix Halo / `gfx1151` | Linux aarch64 / DGX Spark | WSL2 | Native Windows | macOS Apple Silicon |
| --- | --- | --- | --- | --- | --- | --- |
| Werk release binary | Generic x86_64 artifact | Backend-neutral Strix Halo x86_64 artifact | Spark-only arm64 `sm_121` artifact | Linux x86_64 artifact | x86_64 artifact | arm64 artifact |
| llama CPU | Supported build path | Supported build path | Supported build path | Supported build path | Supported build path | Supported build path |
| llama CUDA | Primary NVIDIA path | Not applicable | Primary GB10 path; build upstream natively | Best-effort Linux/CUDA path | Build path with CUDA toolchain | Not applicable |
| llama ROCm | Primary practical ROCm path | Primary HIP path; real `gfx1151` smoke required | Not applicable to GB10 | Not recommended | Not practically supported | Not applicable |
| llama Vulkan | Build path with Vulkan SDK | Implemented alternative; benchmark on target hardware | Not a primary Spark path | Best effort | Build path with Vulkan SDK | Not a primary path |
| llama Metal | Rejected | Rejected | Rejected | Rejected | Rejected | Supported build path |
| local vLLM | Eligible | Operator-provisioned ROCm environment only; generic managed pip install rejected | Native-Linux eligible; ARM64 package/model support remains upstream-dependent | Installer allowed, local execution currently rejected/cautioned | Rejected | Rejected |
| remote vLLM | Supported | Supported; declare ROCm | Supported | Supported | Supported | Supported |
| local oMLX | Rejected | Rejected | Rejected | Rejected | Rejected | Installed CLI and model-specific compatibility required; real inference depends on the available memory and upstream runtime |
| Qwen-TTS | CUDA is the primary documented path; CPU possible | ROCm/model dependent and hardware-unvalidated | Experimental/upstream-dependent | Experimental/upstream-unconfirmed | Experimental/upstream-unconfirmed | CPU/MPS experimental and upstream-unconfirmed |
| ONNX targets | Requires matching runner bundle | Requires matching ROCm runner bundle | Requires matching Linux aarch64 runner bundle | Requires matching Linux bundle | Requires matching Windows bundle | Requires matching macOS bundle |

Werk's release tooling produces profiles for generic Linux x86_64, AMD Strix
Halo x86_64, Linux aarch64/DGX Spark, Windows x86_64 and macOS arm64. Both
hardware profiles must be packaged and smoke-tested on their named host. The
Spark artifact uses a CUDA 13+ toolchain and targets GB10 compute capability
12.1 (`sm_121`); the Strix artifact remains backend-neutral and discovers ROCm
or Vulkan companion runtimes later. Windows arm64 and macOS x86_64 are not
current release targets.

### DGX Spark and Nemotron

Werk recognizes text-only Nemotron-H safetensors architectures and can route
them to vLLM. On Spark the recommended deployment is NVIDIA's compatible vLLM
container with Werk attached to its OpenAI endpoint. A separately provisioned
local vLLM interpreter can also be selected explicitly, but the managed generic
pip installer is deliberately disabled on Spark.

GGUF checkpoints can use a compatible llama.cpp server; the managed CUDA build
detects GB10 and selects the architecture-specific CMake target. Werk's vLLM
adapter remains text-only, so recognizing Nemotron-H does not imply support for
Nemotron Omni image, audio or video inputs. See the complete
[DGX Spark guide](integrations/dgx-spark.md).

### AMD Strix Halo

On Linux x86_64 Ryzen AI Max systems, Werk recognizes the Strix Halo CPU or
the `gfx1151` ROCm agent. GGUF can use an external llama.cpp ROCm/HIP or Vulkan
server. Supported safetensors text architectures, including eligible
Nemotron-H repositories, can use a separately provisioned ROCm vLLM
interpreter or endpoint. The generic managed vLLM pip install is deliberately
disabled for this profile so it cannot install a CUDA-oriented or otherwise
incompatible wheel.

Strix Halo uses physically shared CPU/GPU memory. Model estimates must not add
host RAM and GPU-visible capacity as independent pools, and CPU offload does
not create another physical tier. The integration and diagnostics are
implemented, but each runtime still requires a real `gfx1151` inference smoke
before being called hardware-validated. NVIDIA NVFP4 checkpoints are not
claimed as AMD-compatible. See the complete
[Strix Halo guide](integrations/strix-halo.md).

### WSL and vLLM

The current vLLM installer permits WSL and prints a warning, but local runtime
eligibility subsequently rejects WSL because vLLM can depend on GPU memory
features such as UVA and CUDA IPC. This is a known inconsistency. Prefer native
Linux or configure a remote vLLM endpoint.

## Qwen-TTS isolation

Qwen3-TTS does not use the generic Transformers text-to-audio pipeline. It uses
the qwen_tts package and its Qwen3TTSModel wrapper.

The package pins versions of shared libraries such as Transformers. Werk
therefore keeps it outside the general media-companion Python environment:

~~~text
WERK_HOME/
└── backends/
    └── qwen-tts/
        └── venv/
~~~

Install it explicitly:

~~~bash
werk backend install qwen-tts
werk backend doctor --debug
~~~

An externally managed compatible interpreter can be selected with:

~~~text
WERK_QWEN_TTS_PYTHON=/absolute/path/to/python
~~~

Discovery checks only the explicit interpreter and Werk's managed environment.
It deliberately does not select an arbitrary qwen-tts package from PATH.

While `werk serve` remains alive, Qwen-TTS execution uses its own resident,
serialized companion process and bounded model LRU. It is separate from the
generic media companion and its Diffusers/Transformers LRU;
`WERK_MEDIA_PIPELINE_CACHE_SIZE` sets the capacity of each worker independently
(default `1`). This is process-local model residency, not named Werk Prefill
state or cross-restart persistence. Restarting Werk makes the next Qwen-TTS
request cold.

### Qwen platform statement

Qwen does not publish a complete operating-system/accelerator support matrix.
Its documented reference examples use CUDA, BF16 and optionally
FlashAttention 2. The official CLI also accepts CPU, but this is not a
performance guarantee.

The qwen-tts 0.1.1 package is published as a platform-neutral Python wheel.
That describes the wheel, not the native PyTorch, audio or accelerator
dependencies. Until Werk has platform CI and real inference fixtures:

- Linux with NVIDIA CUDA is the primary documented target;
- CPU is expected to be much slower and remains model-dependent;
- Windows CUDA/CPU and WSL2 are experimental;
- macOS CPU/MPS is experimental;
- AMD ROCm is upstream-unconfirmed.

Upstream references:

- [Qwen3-TTS environment and CUDA examples](https://github.com/QwenLM/Qwen3-TTS#environment-setup)
- [qwen-tts package configuration](https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml)
- [qwen-tts 0.1.1 package files](https://pypi.org/project/qwen-tts/0.1.1/)

## Managed locations

The store root is selected in this order:

1. global <code>--model-home</code>;
2. <code>WERK_HOME</code>;
3. <code>XDG_DATA_HOME/werk1112</code>;
4. on Windows, <code>LOCALAPPDATA/werk1112</code> or the corresponding
   UserProfile fallback;
5. otherwise <code>HOME/.local/share/werk1112</code>.

Managed backend children are:

| Target | Child below the store root |
| --- | --- |
| llama targets | <code>backends/llama-cuda</code>, <code>llama-rocm</code>, <code>llama-vulkan</code>, <code>llama-metal</code> or <code>llama-cpu</code> |
| ONNX targets | <code>backends/onnxruntime-cuda</code>, <code>onnxruntime-rocm</code> or <code>onnxruntime-cpu</code> |
| vLLM | <code>backends/vllm</code> |
| oMLX worker data | <code>backends/omlx/workers/PROCESS_ID</code>; isolated data for a Werk-owned child, removed during normal worker cleanup. The installed oMLX CLI remains external. |
| Qwen-TTS | <code>backends/qwen-tts</code> |

Models, optimized model artifacts, outputs and jobs are separate siblings.
Removing one backend directory must not remove the store root.

## External runtime overrides

Managed installation is optional. Relevant explicit overrides include:

| Runtime | Override |
| --- | --- |
| llama.cpp | <code>WERK_LLAMA_SERVER_CUDA</code>, <code>WERK_LLAMA_SERVER_ROCM</code>, <code>WERK_LLAMA_SERVER_VULKAN</code>, <code>WERK_LLAMA_SERVER_METAL</code>, <code>WERK_LLAMA_SERVER_CPU</code> |
| ONNX | <code>WERK_ONNX_RUNTIME_*</code> for execution and <code>WERK_ONNX_RUNTIME_BUNDLE_*</code> for provisioning bundles |
| vLLM | <code>WERK_VLLM_PYTHON</code>, or remote <code>WERK_VLLM_HOST</code>, <code>WERK_VLLM_PORT</code> and optional <code>WERK_VLLM_MODEL</code> |
| oMLX | <code>WERK_OMLX_BIN</code> for the installed local CLI; <code>WERK_OMLX_HEALTH_TIMEOUT_SECONDS</code> for startup and loading |
| Qwen-TTS | <code>WERK_QWEN_TTS_PYTHON</code> |
| general media companion | <code>WERK_MEDIA_PYTHON</code> or <code>WERK_MEDIA_COMPANION</code> |

Explicit paths remain the operator's responsibility and are not removed by
Werk.

## Uninstall and cleanup

There is no managed backend-uninstall subcommand in the current CLI:

~~~text
werk backend uninstall qwen-tts
# not implemented
~~~

The current safe manual procedure is:

1. stop Werk servers and active inference using the target;
2. determine the resolved store root from the same <code>--model-home</code> or
   <code>WERK_HOME</code> configuration used during installation;
3. select exactly one child listed in [Managed locations](#managed-locations);
4. remove that child with the operating system's normal file-management tools;
5. run <code>werk backend list</code> and <code>werk backend doctor --debug</code>.

Do not remove the complete store root. That would also target models, outputs,
jobs and shared artifacts. Manual backend removal is not recoverable except by
reinstallation.

For Qwen-TTS, removing only
<code>WERK_HOME/backends/qwen-tts</code> removes its isolated environment. It
does not remove Qwen model repositories stored under
<code>WERK_HOME/models</code>.

A future clean command should be idempotent, resolve and validate the target
inside the active store, stop a managed worker, support a dry run, remove only
target-owned state and never delete models or outputs.

## Known backend-management gaps

- no managed uninstall command
- backend list/doctor do not yet expose a dedicated Qwen-TTS status row
- backend CLI summary text historically referred only to llama.cpp
- no pinned llama.cpp revision or reproducible build receipt
- no automatic ONNX runner download
- no execution-provider validation during ONNX provisioning
- WSL vLLM install/runtime eligibility mismatch
- Qwen-TTS platform support is not yet verified by a multi-OS inference matrix
- no single machine-readable backend capability manifest

These gaps should remain visible in documentation and diagnostics until the
implementation changes.
