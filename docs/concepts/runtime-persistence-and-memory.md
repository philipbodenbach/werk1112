---
title: Runtime persistence and memory architecture
description: Architecture, safety boundaries and backend capability matrix for Werk Protocol 1.0 runtime state.
---

# Runtime persistence and memory architecture

[Back to the documentation index](../documentation.md)

Werk Protocol 1.0 adds a control plane for backend-owned prefix/runtime state,
memory telemetry and split prefill/decode. It is additive: the established
`/v1`, Comfy compatibility and AUTOMATIC1111 routes keep their existing
contracts.

This page describes the architecture and the current implementation boundary.
The exact HTTP shapes are in the [Werk Protocol 1.0 reference](../reference/werk-protocol-v1.md).

## Architecture

| Layer | Responsibility | Deliberate boundary |
| --- | --- | --- |
| Typed protocol DTOs | Versions, capabilities, state and memory summaries, policies, actions, prefill and decode | No Axum, filesystem or backend-process types |
| `WerkControl` service | Transport-neutral, object-safe async operations | No HTTP headers or status codes |
| Local control service | Principal isolation, policy, catalog, handoffs, reservations and backend delegation | Never interprets a backend's KV representation |
| HTTP adapter | Authenticates, creates an opaque request ID, maps typed errors and serves `/werk/v1` | Does not replace `/v1` or forward arbitrary backend error JSON |
| Backend runtime adapter | Proves compatibility and owns prefill, decode, snapshot, restore, movement and expert operations | Unsupported methods fail explicitly |
| Bundled HTTP client | Bounded typed access used by `werk runtime` | Loopback-oriented HTTP only; no redirects or TLS termination |

Backend state is one of an in-process handle, opaque bytes, an opaque file or
an external connector key. Werk accounts for and transports that state, but
does not deserialize it into a fictional generic KV-cache format. A state can
only be reused when its producing adapter proves compatibility.

This is not a scheduler, worker fleet or distributed cache. Prefill/decode does
not add request placement, retries between workers or cross-backend state
conversion.

## Capability status semantics

Capability discovery is mandatory before optional runtime-control work. Every
reported status has one exact meaning:

| Status | Meaning |
| --- | --- |
| `supported` | Operational through the active Werk adapter without an experimental opt-in. |
| `unsupported` | The active adapter does not implement the operation. Retrying without changing the backend will not help. |
| `unavailable` | The contract exists, but a runtime, process, probe or telemetry prerequisite is missing now. |
| `experimental` | Implemented but accepted only when the effective request has `allow_experimental: true`; Prefill can receive this from an explicit request or an enabled server default. |
| `externally_managed` | The backend or another component owns the behavior; Werk may report it but refuses to present it as a Werk-controlled mutation. |
| `metadata_only` | Informational reporting only. It does not imply a state handle or control operation. |

Clients must not collapse these values into one Boolean. In particular,
`metadata_only` process reuse is not prefix-state support, and a backend's own
cache is not Werk-controlled merely because it exists.

## Current production capability matrix

Statuses below describe the production adapters in this repository. Host and
accelerator telemetry depend on what the running host can sample, so those
cells show both truthful outcomes.

| Capability ID(s) | Functionally validated llama.cpp process | llama.cpp before/after a failed state probe | local vLLM | remote vLLM | Other production backends |
| --- | --- | --- | --- | --- | --- |
| `runtime.memory.telemetry.host` | `supported` or `unavailable` | `supported` or `unavailable` | `supported` or `unavailable` | `supported` or `unavailable` | `supported` or `unavailable` |
| `runtime.memory.telemetry.accelerator` | `supported` or `unavailable` | `supported` or `unavailable` | `supported` or `unavailable` | `supported` or `unavailable` | `supported` or `unavailable` |
| `runtime.memory.reservations` | `unavailable` | `unavailable` | `unavailable` | `unavailable` | `unavailable` |
| `runtime.model_residency` | `supported` | `supported` when a compatible managed runtime is available; otherwise `unavailable` | `supported` with an active Werk-owned process; otherwise `unavailable` | `externally_managed` with an active endpoint | `supported`, `unavailable`, or `unsupported` according to the concrete adapter and runtime described below |
| `runtime.state.prefix_cache` | `experimental` | `unavailable` | `externally_managed` only when every active process explicitly enables APC; otherwise `unavailable`, `metadata_only`, or `unsupported` | `metadata_only` because the OpenAI endpoint does not expose an introspection/control contract | `unsupported` |
| `runtime.state.persistence`, `runtime.state.restore`, `runtime.state.tier.disk` | `experimental` | `unavailable` | `unsupported` | `unsupported` | `unsupported` |
| `runtime.state.restore.cross_restart` | `unavailable` | `unavailable` | Not advertised; no Werk adapter | Not advertised; no Werk adapter | Not advertised; no Werk adapter |
| `runtime.state.tier.ram`, `runtime.state.tier.vram` | `unsupported` | `unsupported` | `unsupported` | `unsupported` | `unsupported` |
| `runtime.pd.prefill`, `runtime.pd.decode`, `runtime.pd.handoff` | `experimental` | `unavailable` | `unsupported` | `unsupported` | `unsupported` |
| `runtime.experts.residency` | `unsupported` | `unsupported` | `unsupported` | `unsupported` | `unsupported` |

For vLLM, `runtime.state.prefix_cache` describes backend-owned automatic reuse,
not a Werk state handle. With no active process it is `unavailable`. It is
`externally_managed` with the sole operation `automatic_reuse` only when there
is at least one active, Werk-started local vLLM process, no active remote
process, and every active process's effective arguments contain the exact
`--enable-prefix-caching` flag without `--no-enable-prefix-caching`. If every
active process explicitly disables APC it is `unsupported`; remote,
mixed or ambiguous effective arguments are `metadata_only`. Even in the
`externally_managed` case, Werk cannot name, persist, move, prune or hand off a
vLLM cache entry. With `werk serve --persistence`, Werk adds the native APC
flag to a local vLLM launch unless `WERK_VLLM_ARGS` explicitly enables or
disables it. `--persistence-reuse disabled` instead adds the native disable
flag. Werk verifies that the installed server help advertises a generated flag
before spawning it. KV offload, LMCache and expert controls remain outside the
adapter and are not inferred from APC.

### Execution lifetime and reuse matrix

The runtime-control capabilities above are only one kind of persistence. The
following matrix separates four mechanisms that otherwise look similar during
a warm benchmark:

| Execution path | Execution lifetime | Model or pipeline residency | Automatic prefix/KV reuse | Named Werk state (`/werk/v1/prefill`) | Durable JobRecord |
| --- | --- | --- | --- | --- | --- |
| Werk-managed `llama-server` | One child server is reused for an exact manifest/runtime key while `werk serve` remains alive. Replacing model files under the same ID selects a new process. | Yes; weights stay in the child process. | `cache_prompt: true` is sent on completion and OpenAI chat/vision paths; exact reuse remains llama.cpp-owned. | Experimental only after the exact live process passes the functional state probe. Snapshots are usable only by that process generation. | No for ordinary chat/vision requests. |
| Local vLLM | One Werk-started server process is reused for its exact manifest/runtime/argv key. | Yes; weights stay in the vLLM process. | Backend-owned APC when effective arguments enable `--enable-prefix-caching`; `werk serve --persistence` supplies that default unless explicitly overridden. | No. Werk cannot name, snapshot, restore, move or prune APC entries. | No for ordinary chat/vision requests. |
| Remote vLLM | The separately operated endpoint owns its lifetime; it may outlive Werk. | Endpoint-owned, not guaranteed or controlled by Werk. | Endpoint-owned and opaque to Werk; remote configuration is reported as metadata, not as a controllable cache. | No. | No for ordinary chat/vision requests. |
| In-process llama.cpp high-level and legacy FFI | Rust backend objects and chat sessions live inside one `werk serve` process. | Yes; model weights and the server chat-session LRU use an exact, checksum-sensitive manifest identity. | A same-process text chat session can retain and trim its KV context for a shared prefix. Image and tool-calling requests bypass that Werk chat-session cache. | No. | No. |
| Candle | In-process Rust backend lives for the `werk serve` process. | Yes; loaded weights are keyed by exact, checksum-sensitive manifest identity. | No cross-request KV reuse; the model KV cache is cleared before each generation. | No. | No. |
| Burn (currently Phi-3) | In-process Rust backend lives for the `werk serve` process. | Yes; one prepared model is cached by exact, checksum-sensitive manifest identity. | No cross-request prefix/KV contract. | No. | No. |
| ONNX Runtime external runner | The configured opaque runner is invoked per request. | No; the runner exposes no validated residency protocol. | No. | No. | No. |
| ONNX Runtime Python GenAI fallback | One serialized Werk-owned Python worker is reused for CPU execution. | Yes; exact model and tokenizer entries use a bounded LRU, default capacity `1`. | No; each request gets a new generator and prompt state. | No. | No. |
| Transformers compatibility | One serialized Werk-owned Python worker is reused. | Yes; exact model/tokenizer entries use a bounded LRU, default capacity `1`. | Only generation-local cache; prompts and generator state are not shared across requests. | No. | No. |
| MLX / MLX-VLM | The configured command or Python module is invoked per request. | No Werk resident model cache; the subprocess reloads. | No cross-request prefix/KV reuse. | No. | No. |
| Generic media companion | One serialized resident execution worker is reused while `werk serve` lives. | Yes; a bounded Diffusers/Transformers LRU, default capacity `1`. | Not applicable; this is model/pipeline residency, not a text KV cache. | No. | Yes on job-backed media routes; direct routes remain synchronous. |
| Managed Qwen3-TTS media | A separate serialized resident Qwen execution worker is reused while `werk serve` lives. | Yes; a separate bounded Qwen LRU, also default capacity `1`. | Not applicable. | No. | Yes when speech is requested asynchronously, as the ComfyUI node does. |

All entries in the residency columns are process-local. A Werk or owned backend
restart makes the next inference cold. A remote vLLM service can remain alive
across a Werk restart, but that is external service lifetime rather than Werk
persistence. `WERK_MEDIA_PIPELINE_CACHE_SIZE` applies independently to the
generic and managed-Qwen worker, so using both can retain up to that many
entries in each process. Transformers and ONNX Python GenAI have independent
LRUs controlled by `WERK_TRANSFORMERS_MODEL_CACHE_SIZE` and
`WERK_ONNX_GENAI_MODEL_CACHE_SIZE` respectively.

Job records are different again. A job-backed route stores the request, status
and eventual result under `WERK_HOME/jobs`, but does not serialize a loaded
model, pipeline, KV cache or resumable computation. Terminal records survive a
restart; a nonterminal record found during startup is marked failed instead of
being resumed. Job durability therefore provides polling and diagnostics, not
inference acceleration or runtime-state reuse.

### llama.cpp's experimental boundary

llama.cpp state support starts fail-closed. It requires an already-running
Werk-managed process for the exact installed GGUF model. The executable must
advertise the required slot flags, and Werk must prove that its final argument
set still selects one private explicit slot and the private snapshot
directory. The adapter records fingerprints for the executable, help output
and effective process arguments plus a random process-generation identity.

The first model-scoped Prefill request performs an end-to-end functional probe
against that exact process: prefill the private slot, save it, erase it,
restore it, replay from it and verify the observed slot/token state. This is
why an explicitly opted-in client may submit Prefill while discovery is still
`unavailable`; the server remains the authoritative gate. Only that process
generation then reports the state and prefill/decode capabilities as
`experimental`. A failed generation remains `unavailable`; Werk does not infer
support from a version string or the presence of a flag alone.

Snapshots are usable only by their original live process. A restart changes
the generation identity and invalidates the in-memory handles and retained
restore context. Disk catalog entries may remain visible, but cross-restart
restore is explicitly `unavailable`. There is no cross-backend or
cross-accelerator portability.

## Persistence and reuse policy

A prefill request carries a policy with four retention modes:

| Mode | Behavior |
| --- | --- |
| `ephemeral` | Issue only the short-lived decode handoff; do not create a named reusable state. |
| `memory` | Require an operational RAM or VRAM state tier; fail if the adapter cannot retain one. |
| `disk` | Require operational persistence and a disk-persistable opaque snapshot; fail instead of silently weakening the request. |
| `auto` | Prefer a disk snapshot when the adapter can produce one, otherwise retain the backend state as a volatile named state. |

`reuse` can be `disabled`, `prefer` or `required`. `prefer` uses an exactly
compatible state when available and otherwise performs a new prefill.
`required` fails if compatible reuse cannot be proven; it never falls through
to a fresh prefill. `ttl_seconds` bounds named-state retention, and `pin`
protects a live state from policy eviction. TTL is the stronger boundary: an
expired state is unavailable even when pinned and is deleted by the next
mutating catalog operation. Pinning is not a lease on the backend process and
cannot make an invalid process-generation handle portable.

State summaries expose opaque IDs, model/backend identity, tier, status,
logical byte size when known, timestamps, pinning and reusability. They do not
expose backend handles, cache keys, snapshot paths, prompts or credentials.

### Server-side Prefill defaults

The server can provide persistence policy defaults for clients that omit them:

~~~bash
werk serve --model my-gguf-model --persistence
~~~

This means `auto` retention, `prefer` reuse, no TTL and no pinning. It also
supplies the experimental opt-in when `allow_experimental` is omitted. For a
local vLLM selected by this server, it additionally defaults the native
automatic prefix cache on. Model/pipeline residency in Werk-owned in-process
backends and resident workers is automatic and does not require this flag.
Granular controls select a different default policy, and any one of them
implies `--persistence`:

~~~bash
werk serve --model my-gguf-model \
  --persistence-mode disk \
  --persistence-reuse prefer \
  --persistence-ttl-seconds 3600 \
  --persistence-pin
~~~

The modes are `ephemeral`, `memory`, `disk` and `auto`; reuse is `disabled`,
`prefer` or `required`; TTL is bounded to 1 through 2592000 seconds. These are
defaults only for absent top-level members of `POST /werk/v1/prefill`. A
present `policy` object owns the complete request policy, and a present
`allow_experimental` Boolean owns the opt-in decision. Explicit `false` is
never promoted to `true` by server configuration.

For local vLLM, `--persistence-reuse disabled` defaults the native prefix cache
off; an explicit enable or disable in `WERK_VLLM_ARGS` wins. Remote vLLM stays
externally managed and receives no generated process argument.

This switch does not create a generic Werk KV format or make one-shot external
commands persistent. Today the positive named-state path is experimental and
requires a functionally validated Werk-managed llama-server process for the
exact installed GGUF model. The adapter owns the opaque state; Werk owns the
policy, lifecycle, accounting and compatibility envelope around it. OpenAI
`/v1` and media inference are not redirected through Prefill, semantic output
caching is not introduced, and cross-restart restore remains unavailable.

### Compatibility envelope

Reuse is an equality check over the complete producer envelope:

- model, tokenizer and principal-scoped prompt fingerprints;
- optional chat-template and multimodal-processor fingerprints;
- backend, backend version, adapter version and accelerator family;
- tensor dtype, KV dtype, quantization, cache layout and optional block size;
- context size, optional batch size and optional RoPE configuration
  fingerprint;
- producer protocol version.

There are no wildcard dimensions. A changed backend binary, model selection,
runtime arguments, cache layout or process generation cannot be waved through
as compatible. `incompatible_state` is distinct from a simple cache miss.

## Persistent store and crash safety

Disk state lives under the active server store:

~~~text
WERK_HOME/
├── auth/runtime-namespace.key
└── runtime-state/v1/
    ├── .lock
    ├── .quarantine/
    └── p_<opaque-HMAC-principal>/
        └── st_<opaque-state-id>/
            ├── metadata.json
            ├── metadata.sha256
            ├── payload.bin
            ├── last-accessed     # created after a successful load
            └── pinned            # only when pinned
~~~

The catalog uses a process mutex and an advisory file lock. A commit streams
the opaque payload into a same-directory staging entry, records SHA-256
integrity data, synchronizes files and, where the platform exposes it,
directories, and finishes with an atomic directory rename. Before mutation,
reconciliation removes abandoned staging entries and expired states,
quarantines malformed or corrupt state directories, and unlinks unsafe
non-directory entries without following them. Quarantine retention is itself
bounded; an entry that cannot be inspected safely within those bounds is
deleted instead. Directory synchronization covers both the source namespace
and quarantine after a move. Payloads are bounded and streamed rather than
copied into one unbounded memory buffer.

Store directories and files use owner-only permissions where the platform
supports them. Symlink/reparse-point checks prevent state reads from following
an unexpected filesystem target. Pruning accepts only an explicit ID set, a
non-empty constrained filter or `all` with a separate confirmation, and all
mutation surfaces support a dry run. Lists, inspection and dry-run previews
are strictly read-only: they do not create or repair the catalog, update access
metadata, reconcile entries, quarantine corruption or delete expiry. They fail
closed by omitting invalid or expired entries, while the next real mutation
performs that cleanup.

Runtime-state pruning touches only the selected runtime-state catalog. It does
not remove models, optimized artifacts, outputs, jobs, authentication files,
backend installations, the temporary directory or external output paths.
The ordinary full reset is still an explicit, principal-scoped prune with a
dry-run preview. If a broken server process cannot perform it, an administrator
must stop the server before moving `runtime-state/v1` aside as a recoverable
offline reset; replacing the namespace key or deleting the surrounding store
is neither required nor safe.

## Principal and prompt isolation

When authentication is enabled, Werk derives a stable opaque principal from
the accepted API key with HMAC-SHA-256. The HMAC secret is generated inside
`WERK_HOME/auth`, committed atomically and kept out of protocol responses.
The raw API key never becomes a directory name or service-layer value.

Prompt fingerprints are also keyed and principal-scoped. Consequently, two
API keys cannot discover or reuse one another's runtime states even if they
submit identical prompts. In explicitly unauthenticated local mode, requests
share the `local` principal; that mode is therefore not an isolation boundary.

## Dynamic memory accounting

`GET /werk/v1/memory` samples host memory and, when available, accelerator
memory. Unified-memory systems share accounting; discrete systems keep host
and accelerator tiers separate. Unknown observations remain `null`/`unknown`
instead of being guessed.

The manager combines observed use with Werk-managed allocations and pending
reservations. Current defaults classify utilization at 75% (`soft`), 85%
(`hard`) and 95% (`emergency`), with 5 percentage points of downward
hysteresis and a five-second action cooldown. These are internal policy
defaults, not a public environment-variable interface.

A backend participates in admission control only when it supplies a positive,
bounded memory requirement before load, restore or replacement decode and
advertises the reservation capability. Werk reserves capacity before invoking
the backend, commits accounting only after the returned state agrees with the
reservation, and releases it with the state lease. If backend release fails,
Werk does not claim that capacity was freed: the allocation remains
conservatively accounted, becomes pinned against policy movement and increments
the public `failed_releases` counter. The dynamic counter map also exposes
`orphaned_release_bytes`, `telemetry_errors`, `backend_cleanup_failures` and a
numeric `backend_cleanup_latched` flag. A failed telemetry sample leaves
capacity and availability unknown without erasing these accounting signals.

On a denied reservation, Werk can make one bounded pressure-relief pass,
preferring eligible least-recently-used demotions before eviction, skipping
pinned states and states with a live handoff, then retry the reservation once.
The planner includes the requested bytes in projected utilization and targets
below the hard threshold even when pre-request pressure is only `normal` or
`soft`; cooldown and per-cycle action bounds still apply. No current production
adapter supplies such a bound, so `runtime.memory.reservations` currently
reports `unavailable`; the manager is not presented as protecting unaccounted
backend loads.

Dry-run tier changes use a backend inspection operation that is required to be
side-effect free. An adapter that cannot inspect a snapshot export without
creating or changing backend state must reject the preview; Werk never obtains
a real snapshot merely to predict a disk promotion.

## Expert-residency boundary

The core includes a backend-neutral expert policy engine. Given
backend-observed expert IDs, tiers, optional sizes and access events, it can
maintain a decaying hotness score, pins and transition cooldowns; produce
bounded prefetch or pressure-relief plans; and claim an action before backend
work so local metadata changes only after the adapter reports success. Unknown
sizes are kept unknown and are never assumed to fit a target tier.

That policy engine does not discover MoE modules or move tensors itself. A
production adapter must supply stable expert identities, truthful residency
observations and the actual movement operation. No current production adapter,
including vLLM, supplies that integration, so expert routes return
`unsupported`. ComfyUI exposes capability-gated expert telemetry and explicit,
dry-run-first control nodes, but they fail closed while that status is active
and never convert route presence into a support claim. There is no in-process
or external Krasis connector in this version.

## Prefill/decode handoff

Prefill produces an opaque random handoff, not serialized KV data. The server
stores only a hash as the lookup key, binds the record to the authenticated
principal and the exact compatibility/process identity, and expires it after
15 minutes. The in-memory registry allows at most 128 live handoffs per
principal and 1024 live handoffs in total. It purges expired records before
checking either bound; at capacity it rejects issuance with retryable
`resource_exhausted` and never evicts a still-valid handoff. Prefill reserves a
slot before backend or persistence work and rolls it back on failure. This
prevents successful state creation followed by a late registry-capacity
failure.

Decode consumes the handoff before calling the backend. It is therefore
single-use even when decoding fails, and a retry must begin with a new
prefill/handoff. The server does not fall back to another backend or replay the
prompt silently. The consume operation atomically holds the slot for a possible
replacement token so another request cannot steal it during decode; the slot is
released when no updated state is returned. A backend may return an updated
state and a new handoff, but the current llama.cpp adapter completes decode
without one. Backend-returned completion text is bounded to 1 MiB of UTF-8
bytes.

ComfyUI preserves this boundary with a private `WERK_STATE_HANDOFF` socket;
the value is not exposed as a `STRING` or JSON output. See the
[ComfyUI custom-node guide](https://github.com/philipbodenbach/werk1112/blob/main/utils/comfyUI/README.md#runtime-persistence-experts-and-split-prefilldecode).

## Operations checklist

1. Start `werk serve` with authentication unless this is an intentionally
   isolated local development process.
2. Read runtime info and capabilities; do not assume a backend feature from
   its name.
3. Treat `experimental` as unavailable unless the user deliberately opts in,
   either in the request or through the Prefill-only server default.
4. Preview every state mutation, then repeat it with the explicit execute flag
   or `dry_run: false` only after checking the selector/result.
5. Expect handoffs and same-process states to expire when the server or backend
   process exits.

CLI examples are in the [runtime-control CLI section](../reference/cli.md#runtime-control),
and the transport contract is in the [Werk Protocol 1.0 reference](../reference/werk-protocol-v1.md).
