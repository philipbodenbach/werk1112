# CLI reference

This page describes the stable command groups and their operational semantics.
The installed binary remains authoritative for exact flags and accepted values:

```bash
werk --help
werk COMMAND --help
```

## Global options

Global options precede the subcommand:

```bash
werk --model-home /srv/werk \
  --backend auto \
  chat model-id
```

The most important global controls are:

| Option | Meaning |
| --- | --- |
| `--model-home PATH` | Select the complete Werk store; equivalent to `WERK_HOME`. |
| `--backend BACKEND` | Select `auto` or constrain routing to a named runtime/backend family. |
| `--device DEVICE` | Legacy Candle-only device override. Prefer `--backend` or typed media accelerator controls. |
| `--auto-install-backends` | Permit managed provisioning during runtime selection. Installation is otherwise explicit. |
| `--no-auto-install-backends` | Prohibit automatic managed provisioning. |
| `--ctx-size`, `--batch-size`, `--ubatch-size` | Advanced llama.cpp context and batching controls. |
| `--gpu-layers`, `--main-gpu` | Advanced llama.cpp accelerator placement controls. |
| `--kv-cache-type`, `--flash-attn`, `--kv-offload` | Advanced llama.cpp cache and attention controls. |
| `--threads`, `--threads-batch` | CPU helper-thread controls for generation and prompt evaluation. |

Backend availability, explicit constraints and fallback semantics are covered
in [Backends](../backends.md).

## Model lifecycle

### Import local data

`import` copies a local file or repository directory into the managed store:

```bash
werk import /absolute/path/to/model --name local-model
```

The original path is not used as mutable runtime state after a successful
copy. Removing the managed model does not remove the original source.

Add `--link` to register existing files without copying them or creating an
operating-system symlink:

```bash
werk import /mnt/f/Werk1112/models/wan22-ti2v-5b --name wan22-ti2v-5b --link
```

The source can be a file, a repository directory, or an existing Werk model
directory containing `manifest.json` and `files/`. Existing model metadata is
retained. The active store holds the registration and optimized artifacts,
while model files stay at their external location. This supports local and
external models together; `--model-home` and `WERK_HOME` still select the
complete store. `import` without `--link` continues to copy files.

Import a collection containing several models with `--all`:

```bash
# Register every model from the RAID in the active store:
werk import /mnt/f/Werk1112/models --all --link

# Copy a collection into the active store instead:
werk import /path/to/model-collection --all
```

`--all` examines the collection's immediate children for model directories and
supported model files, ignoring unrelated entries. Each repository remains one
model; its components and weight shards are not registered separately. The
collection directory must not itself be a model repository. Discovery is not
recursive.

Existing Werk model directories retain their model IDs and metadata, including
when copied. Other models use the directory name or file stem as their ID.
`--name` is required for a single model and cannot be combined with `--all`.
Before importing a collection, Werk checks for duplicate IDs, destination name
collisions and models already installed in the active store. A conflict stops
the operation before any models are imported; existing models are not
overwritten. Successful imports print each model followed by the total count.

### Pull from Hugging Face

```bash
werk pull organization/repository --name local-name
```

Select a variant from a multi-quant repository:

```bash
werk pull organization/model-GGUF \
  --file model.Q4_K_M.gguf \
  --name model-q4
```

For a split GGUF, select its first shard. Werk downloads every matching shard
in the same directory, validates that the set is complete, and uses the first
shard as the runtime entry point. Other quantization variants are excluded:

```bash
werk pull ggml-org/DeepSeek-V4-Flash-GGUF \
  --file DeepSeek-V4-Flash-Q2_K_S-00001-of-00002.gguf \
  --name deepseek-v4-flash-q2-k-s
```

Automatic GGUF selection also includes the complete shard set. A separate
`--name` lets multiple variants coexist; pulling into an existing model ID
does not replace that model.

Pull currently uses Git plus Git LFS. Gated repositories require accepted
upstream conditions and a token from `werk auth huggingface login`, an accepted
environment variable or the standard Hugging Face token cache.

### List and inspect

```bash
werk list
werk list --task image-generation --layout diffusers
werk list --input-modality audio --output-modality text
werk list --family flux --json
werk inspect model-id
```

`list` shows model metadata, a `managed` or `external` storage label, and the
absolute storage path in `PATH`. For managed models, this is the directory
containing `manifest.json` and `files/`; for external bindings, it is the
external files' source path. It supports metadata filters and sizes table
columns to their contents. When the table
would exceed the terminal width, each model is shown as a labeled block with
its full name and path. `list --json` prints the stored/enriched manifests.
External bindings include a `storage` record with `kind: "external"` and their
absolute path. They remain listed if the external disk is unmounted; loading
them requires the recorded location to be accessible.
`inspect` prints the full manifest as JSON. Declared task support is not proof
that an installed runtime can execute the model.

### Select a tracked model file

For an installed repository with several model files:

```bash
werk select-file model-id model.Q4_K_M.gguf
```

Both a path relative to the model's `files` directory and a `files/...` path
are accepted. Inspect the manifest to obtain exact filenames.

### Remove a model

```bash
werk remove model-id
werk rm model-id
```

This removes the managed model directory and its local optimized artifacts.
For external bindings, only the local registration and artifacts are removed;
external model files are never deleted. Backend environments, unrelated models
and original import sources are separate.

The directory layout and retention rules are documented in
[Models, manifests and the store](../concepts/models-manifests-and-store.md).

## Temporary files

Print or list the temporary directory for the active Werk store, preview its
cleanup or purge it:

```bash
werk temp path
werk temp list
werk temp purge --dry-run
werk temp purge
```

These commands resolve the same active `WERK_HOME` as the rest of the CLI,
including a global `--model-home PATH` override. `temp list` prints each direct
child path in sorted order, including hidden entries, without creating or
changing the temporary directory. It does not recursively expand directories or
follow child symlinks. Purging removes every child of that store's `tmp`
directory, including any concurrently active temporary work, but preserves the
`tmp` directory itself. The `--dry-run` form reports the planned cleanup without
changing the filesystem.

Models, artifacts, managed outputs, jobs, authentication data, backends, files
at the store root and output paths outside the store are persistent boundaries
and are not touched by temporary-file purging.

## Local persistence caches

Inspect local persistence storage and remove one listed entry or all eligible
caches in the active `WERK_HOME` (or global `--model-home PATH`):

```bash
werk cache list
werk cache list --kind chat-kv
werk cache list --json
werk cache purge <CACHE-ID>
werk cache purge --all --dry-run
werk cache purge --all
werk cache purge --all --kind chat-kv
werk cache purge --all --include-history
```

Copy the exact ID from `cache list`; paths and wildcard selectors are not
accepted. `purge` requires either that ID or `--all`. Deletion takes effect
immediately unless `--dry-run` is supplied. Both subcommands support `--json`;
list JSON separates `entries` from `blocked` storage-provider errors so one
unavailable catalog does not hide the remaining caches.

| Kind | Storage beneath the Werk home | Cleanup scope |
| --- | --- | --- |
| `chat-kv` | `chat-sessions/<session-hash>.cache/` | Rebuildable native KV caches for one persistent conversation, across its runtime namespaces. |
| `chat-history` | `chat-sessions/<session-hash>.json` | Saved conversation messages; excluded from `--all` unless `--include-history` or `--kind chat-history` is supplied. An exact history ID also explicitly selects its deletion. |
| `omlx-worker` | `backends/omlx/workers/<worker>/cache/` | Inactive worker cache files only; settings, credentials and logs are retained. |
| `runtime-state` | `runtime-state/v1/` | Individual disk states from the managed runtime catalog; pinned or in-use states are protected. |

The inventory reports each entry's kind, backend when known, size and protection
status. Active chats and workers are skipped; runtime-state cleanup is blocked
while a runtime holds the catalog open. An explicitly selected protected entry
returns an error; bulk cleanup reports skipped entries and continues. Unknown
or unsafe storage is protected. For legacy oMLX worker caches without a lifetime
lock, Werk can remove the known empty directory layout when its native
`_boundary_snapshots` process markers identify owners that have all exited.
This removes directories only: any file (including zero-byte files), live or
unverifiable owner, or unknown layout blocks the legacy cleanup. Existing
lifetime locks continue to protect active workers even when their caches are
empty. Legacy runtime-state catalogs without a usage lease
also remain protected: use `werk runtime prune` on their running server, or
initialize the lease through a state mutation in a current runtime and stop
that runtime before offline cleanup. Stop older Werk processes and their
workers before using cache cleanup; they predate the new runtime usage locks.

Chat KV and history entries share a session hash but can be removed separately.
Deleting KV data retains the conversation and triggers prompt recomputation on
the next run. Deleting history removes the saved messages. Lock files remain
to preserve process coordination. Model weights, original SSD-backed MoE expert
tensors, installed backends, remote/runtime-owned RAM caches and authentication
data are outside this command's scope. Temporary downloads remain managed by
`werk temp`; live named state operations remain available through `werk runtime`.

## Runtime control

`werk runtime` is a quiet, pretty-JSON client for the versioned `/werk/v1`
control plane of an already-running server. Begin with discovery:

```bash
werk runtime info
werk runtime capabilities
werk runtime memory
```

For another trusted HTTP endpoint, put the connection options before the
runtime subcommand:

```bash
export WERK_API_KEY="replace-with-generated-key"
werk runtime --url http://werk-host:11434 capabilities
werk runtime --timeout-seconds 75 info
```

`--url` defaults to `http://127.0.0.1:11434`, must contain an explicit port and
currently accepts only plain `http://` with no path or embedded credentials.
`--api-key` overrides the `WERK_API_KEY` default, but the environment variable
avoids placing a key directly in shell history. The bundled client does not
follow redirects or provide TLS; use it only over loopback or a trusted private
hop. `--timeout-seconds` is a total per-request deadline, defaults to 30 and
accepts values from 1 through 86400; like `--url` and `--api-key`, it must
appear before the runtime subcommand.

The active store belongs to the server. A client-side global `--model-home`
does not redirect remote runtime state; start the server with the intended
`WERK_HOME`/`--model-home` instead.

### List states

```bash
werk runtime states
werk runtime states --model my-model --tier disk --limit 50
werk runtime states --cursor OPAQUE_CURSOR
```

Filters are optional. Tier is `vram`, `ram`, `disk` or `external`; limit is 1
through 100. The JSON result contains opaque state IDs and an optional next
cursor. It never prints a backend handle, snapshot path, prompt or API key.

### Control one state

Every mutation previews by default. Repeat it with `--execute` only after
checking the JSON response:

```bash
werk runtime state st_OPAQUE_ID pin
werk runtime state st_OPAQUE_ID pin --execute
werk runtime state st_OPAQUE_ID unpin --execute
werk runtime state st_OPAQUE_ID evict --execute

werk runtime state st_OPAQUE_ID promote ram --allow-experimental
werk runtime state st_OPAQUE_ID promote ram --allow-experimental --execute
werk runtime state st_OPAQUE_ID demote disk --allow-experimental --execute
```

Promotion targets are `ram` or `vram`; demotion targets are `ram` or `disk`.
The requested direction must be valid for the state's current tier. Supplying
`--allow-experimental` acknowledges the backend capability status for that one
request; it does not enable experimental behavior globally.

### Prune selected states

Prune also defaults to preview and requires exactly one selector form.
`purge` is a visible alias with identical safety semantics:

```bash
# One or more exact IDs
werk runtime prune --id st_FIRST --id st_SECOND
werk runtime prune --id st_FIRST --execute

# A non-empty model/tier/time filter
werk runtime prune \
  --model my-model \
  --tier disk \
  --older-than-unix-ms 1788444000000

# Every state visible to this authenticated principal
werk runtime prune --all --confirm-all
werk runtime prune --all --confirm-all --execute
```

`--all` is rejected without `--confirm-all`; `--confirm-all` is invalid for
the ID and filter forms. `--execute` changes `dry_run` from true to false.
Pruning affects only the selected runtime states. It does not purge temporary
files and cannot remove models, artifacts, outputs, jobs, authentication data,
backend installations or external paths.

To clear every runtime state visible to the current authenticated principal,
preview and then execute the explicit all-selector:

```bash
werk runtime purge --all --confirm-all
werk runtime purge --all --confirm-all --execute
```

This is the normal recovery path when persisted state is no longer useful.
With multiple API keys, each key has a separate opaque namespace and can purge
only its own states. Handoff values cannot be listed: they are intentionally
short-lived, single-use secrets held only in server memory.

If the running process or its backend is too unhealthy to complete that
operation, stop `werk serve` first. As a local administrator, move the exact
active server store's `runtime-state/v1` directory to a separately named backup
and restart Werk. Moving it instead of deleting it keeps recovery possible;
Werk recreates an empty catalog. Do not move the surrounding store or
`auth/runtime-namespace.key`, and never do this while the server is running.
This offline recovery clears disk state for every principal in that store;
models, artifacts, outputs, jobs, credentials, backends and `tmp` are siblings
and remain untouched.

The CLI currently exposes info, capabilities, memory and state maintenance.
Prefill/decode and expert contracts are HTTP/SDK surfaces; the ComfyUI package
provides typed prefill/decode nodes plus capability-gated expert telemetry and
dry-run-first expert-control nodes. The nodes do not imply production backend
support. See the
[Werk Protocol 1.0 reference](werk-protocol-v1.md), the
[runtime architecture and capability matrix](../concepts/runtime-persistence-and-memory.md),
and the
[ComfyUI custom-node guide](https://github.com/philipbodenbach/werk1112/blob/main/utils/comfyUI/README.md#runtime-persistence-experts-and-split-prefilldecode).

## Authentication

Hugging Face credentials:

```bash
werk auth huggingface login
werk auth huggingface status
werk auth huggingface logout
```

Generate an API key file for `werk serve`:

```bash
werk auth api-key generate
werk auth api-key generate --name comfyui --path /tmp/comfyui-key.toml
```

One API-key file can contain multiple `[[keys]]` entries. Generation does not
append to an existing file; merge a newly generated block deliberately.
`--force` overwrites the complete target file and must not be used as an append
operation.

## Estimation

`estimate` has two intentionally different modes.

Model-fit estimation does not require a canonical task:

```bash
werk estimate model-id
werk estimate organization/repository --file model.Q4_K_M.gguf --verbose
werk estimate model-id --json
```

For an installed model it accounts for selected weights, runtime overhead and
a text KV-cache estimate. A repository-looking ID can be estimated remotely
from Hugging Face metadata without downloading the weights.

Workload estimation includes a canonical typed task and runs parameter
resolution first:

```bash
werk estimate flux-dev --task image-generation \
  --width 1024 --height 1024 --steps 28
werk estimate wan-i2v --task image-to-video \
  --width 832 --height 480 --frames 81
```

Task estimates require an installed model. They report estimated accelerator,
host and output demand, fit status, confidence, assumptions, warnings and
recommendations. An estimate is a planning aid, not a guarantee that a
third-party backend can load the architecture.

## Parameters and diagnostics

```bash
werk parameters --task image-generation
werk parameters flux-dev --backend auto --json
werk parameters flux-dev --example
werk parameters flux-dev --sources

werk doctor
werk doctor --task image-generation
werk doctor --model flux-dev --debug
werk backend doctor --debug
```

`parameters` describes the typed schema and, with a model, model/runtime
support. `doctor` checks the host and can add a non-executing model/task probe.
Neither command performs a full cold model load.

## Optimized artifacts

```bash
werk artifacts build model-id
werk artifacts list model-id
werk artifacts rebuild model-id
```

Artifacts are runtime-specific derivatives stored separately from source model
files. An explicit ONNX route may attempt an artifact build when no usable ONNX
artifact exists; normal automatic safetensors routing does not require ONNX.

## Text and vision inference

Start an interactive chat:

```bash
werk chat model-id --max-tokens 128
```

`--max-tokens` is a hard completion cap and can stop text mid-sentence. Terminal
chat streams decoded pieces by default. Use `--stream-granularity chunk` to
reduce terminal flushes and `--verbose` for prompt/decode timing and throughput.

For oMLX `run` and `chat`, the first completed turn includes worker/model
preparation in `load duration` and `total duration`, including persistent-session
startup. Later turns do not count that startup again. Prompt/decode timings and
first-token latency describe the request to the prepared backend.

### One-shot inference with `run`

`run` shares chat's conversation/session implementation and the media inference
service used by `serve`. It exits after one completed response:

```bash
werk run model-id "Explain Rust ownership" --max-tokens 256
werk run model-id "Remember: my project is called Atlas" --session project
werk run model-id "What is my project called?" --session project --stream
werk chat model-id --session project
```

A model/session pair uses the same archive in `run` and `chat`. Backend, device,
llama.cpp tuning, sampling, templates, image inputs, verbose diagnostics and
supported native KV persistence use the existing chat implementation. Ordinary
output is buffered; `--stream` streams text, and `--stream-granularity token|chunk`
also enables streaming. `--no-history` (alias `--single-turn`) conflicts with
persistence, as it does in `chat`.

For local llama-server runs, `--warmup-tokens 0` disables native synthetic warmup
when the runtime supports `--no-warmup`. Local llama.cpp `run`/`chat` never evaluate
a separate test prompt for session persistence. A new session runs the user
request and saves its state. Existing snapshots are checked on restore, and
successful real requests report observed reuse from backend cache counts.
Verbose output distinguishes unverified restore from historical observed reuse.
Worker loading still occurs on every local process start; cold file reads may
remain expensive, including during the first real prefill.

To reuse a worker across separate text/vision/tool `run` processes, start the
existing server once and point the client at it:

```bash
# Terminal 1: worker settings belong to serve.
WERK_LLAMA_ARGS='--n-cpu-moe 38 --reasoning off' \
werk --backend cuda --threads 24 --threads-batch 24 --ctx-size 4096 \
  serve --model vumpt/Qwen3.8-Flash-Next-GGUF --persistence \
  --host 127.0.0.1 --port 11434 --allow-unauthenticated

# Terminal 2: reuse the running worker; each invocation exits normally.
werk run vumpt/Qwen3.8-Flash-Next-GGUF "Explain Rust ownership briefly." \
  --server http://127.0.0.1:11434 --session qwen-test --persistence \
  --stream --verbose --max-tokens 128
```

This Qwen example uses 38 CPU expert layers out of the installed GGUF's 48
blocks. It retains the measured 24-thread configuration; it is not a claim of
optimal settings for every machine or quantization. `--n-cpu-moe` is static CPU
expert placement, not an adaptive expert cache.

`--server` uses the existing `/v1/chat/completions` API, including images, tools,
sampling and request-scoped `werk.omlx` controls. It starts no local worker and
never silently falls back to local inference. The model must be registered in
both client and server stores under the same ID. `--model-home` on the client
selects its model metadata and conversation archive; it does not redirect the
server store. CLI `--image` files are read on the client and sent as data URLs.
Paths inside JSON requests refer to the server filesystem; use data URLs for
separate hosts. Media `run` requests continue to use the local media
service; combining them with `--server` is rejected.

Set backend/device/context/thread options and `WERK_LLAMA_ARGS` on the server.
Worker CLI options on a remote `run` are rejected. For authenticated servers,
provide `--api-key` or `WERK_API_KEY` to the client. The example explicitly binds
an unauthenticated development server to loopback; HTTP URLs require host and
port. There is no automatic server discovery or background daemon.

Sessions still save the transcript locally. Prefix reuse now depends on the
server's live cache: restarting the server or another request replacing that
prefix may require prefill again. This route does not upload or restore local
KV snapshots. Local `run` without `--server` retains native disk snapshots where
supported. Neither transcript persistence nor live cache hits imply persistent
MoE expert residency.

Verbose text diagnostics now include `preparation duration` (worker preparation
and optional capability probe), plus llama.cpp worker startup, cache probe,
restore and save durations. Preparation is included once in `load duration`
and `total duration`. `first token` continues to measure the generation request;
`run first token including preparation` measures from entry to `run` until the
first nonempty text/tool delta arrives. With `--stream`, the first text delta is
flushed immediately. `run elapsed through completion` also includes snapshot
saving; final transcript publication and process exit follow it. Interactive
chat waiting for input is excluded. Native prefill/decode rates are unchanged.
CLI JSON completions also include `timings` and `backend_diagnostics`; the HTTP
API includes native values under `werk.timings` and `werk.backend_diagnostics`.
An available native cache count appears as `usage.prompt_tokens_details.cached_tokens`;
unknown counts remain absent.

For structured conversations and tool calls, pass the existing OpenAI chat
request schema. `--request -` reads JSON from stdin:

```bash
werk run model-id --request request.json --json --session tools
```

```json
{
  "messages": [{"role": "user", "content": "What is the weather in Berlin?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "weather",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto",
  "max_completion_tokens": 256
}
```

This accepts messages (including image parts and tool results), tools,
`tool_choice`, `parallel_tool_calls`, stop strings, sampling, streaming, and
supported `werk.omlx` controls, as in `/v1/chat/completions`. The positional model
selects the installed model; an optional JSON `model` must match its canonical
ID. Explicit CLI sampling options override JSON values. Without an explicit
limit, the JSON limit wins, otherwise the default is 256 tokens.

Tool calls are returned to the caller, as with the HTTP API; Werk does not execute
the declared functions. With a persistent session, the next invocation can send
only new tool-result messages with the corresponding `tool_call_id`. Do not
resubmit the saved transcript, since new messages are appended to that session.

`--json` prints a completion object containing `model`, `message`, `finish_reason`
and `usage`. Combined with streaming it emits newline-delimited `text_delta` and
`tool_call_delta` events followed by that `completion` object. These are CLI JSON
events, not HTTP SSE. Diagnostics go to stderr, with no startup banner on stdout.

For media, `run` accepts all canonical inference tasks and parameters supported
by the existing inference service and selected model/runtime:

```bash
werk run image-model "A mountain lake" --task image-generation \
  --set image.width=1024 --set image.height=1024 --set image.steps=20 \
  --output lake.png
werk run speech-model "Hello from Werk" --task text-to-speech --output hello.wav
werk run whisper-model --task speech-to-text --input audio=recording.wav --json
werk run video-model "Clouds drifting across mountains" --task video-generation \
  --output clouds.mp4
werk run model-id --request media-request.json --json
```

`--input MODALITY[:ROLE]=PATH_OR_URL` is repeatable; use roles such as
`image:mask_image`, `image:initial_image`, or `audio:reference_audio` for conditioned
tasks. `--set PATH=VALUE` accepts canonical parameters and `routing.*` overrides
(e.g. `routing.precision=float16`). A media request file uses the existing
`InferenceRequest` schema:

```json
{
  "task": "image_generation",
  "prompt": "A mountain lake",
  "parameters": {"image.width": 1024, "image.height": 1024, "image.steps": 20}
}
```

If no task is supplied, text/vision models use conversation inference; a model
with exactly one media task uses that task. Ambiguous media models require
`--task`. `werk parameters MODEL --task TASK` lists applicable parameters.
JSON task names use snake_case (`image_generation`); CLI names use hyphens
(`image-generation`), following the existing schemas.
Media outputs use the existing output store and CLI publication rules; `--json`
returns the canonical inference result. Conversation persistence and text
streaming flags do not apply to media tasks and produce an explicit error.
Server transport settings and named Werk Protocol state management remain on
`serve` and `werk runtime`; `run` does not start an HTTP listener.

### Persistent terminal chat

`--persistence` saves completed conversation turns and resumes them on the
next invocation of `run` or `chat`. It is available for every chat backend, including automatic
routing, and preserves the normal streaming path:

```bash
werk chat model-id --persistence
werk --backend mlx chat model-id --persistence --session project
werk --backend omlx chat model-id --session project --verbose
```

The model ID and session name identify the conversation; changing the backend
does not create a different transcript. The default session name is `default`.
`--session NAME` implies persistence. `--persistence-reuse prefer` resumes if
the conversation exists (the default), `required` fails if it does not exist,
and `disabled` starts a fresh conversation that replaces the previous archive
only after a completed answer. The reuse option also implies persistence.
These options conflict with `--no-history` and its `--single-turn` alias.
The reuse policy controls the saved conversation and the default prefix-cache
settings for managed vLLM/oMLX, as in `serve`. Explicit native runtime cache
settings still take precedence; this flag does not purge existing caches.

Private conversation archives live under `$WERK_HOME/chat-sessions/`, or the
equivalent default Werk home. Only one process may open a given model/session
at once. Writes are atomic; failed or interrupted streams do not replace the
last completed conversation. Context-window trimming affects the model input
and preserves the full saved transcript. Archives are limited to 4096 messages
and 16 MiB; corrupt or oversized archives fail explicitly. Image parts remain
in the saved conversation and require an image-capable backend when resumed.

Conversation history is portable text/message data, not a model or KV snapshot.
Werk reports whether the selected route also enables persistent native KV
caching. Unsupported native caches leave conversation persistence operational
and the backend recomputes the prompt. Local vLLM and oMLX receive the same
prefix-cache defaults as `serve --persistence`; explicit runtime
arguments still win. This cache remains vLLM-owned and does not survive its
process restart. Exiting `run` or `chat` still stops its owned backend workers.

The existing llama.cpp route also saves native slot snapshots for persistent
text chats on Unix when the server exposes the required private slot operations.
It restores a compatible snapshot on the next process start and reports
actual prefix hits as `prompt cached count` in `--verbose` output. Snapshots are
namespaced by model files, runtime executable/libraries, native environment and
effective arguments; incompatible or corrupt snapshots are rebuilt from the
conversation. Each runtime namespace retains its latest completed snapshot,
with a 2 GiB snapshot limit. This does not persist loaded model weights or add
cross-restart named Werk Protocol states.

For the optional CUDA expert-cache runtime and an offload test command, see
[CUDA expert offload](../backends.md#experimental-cuda-expert-offload).

Archives and supported native KV caches survive exit. Use
[`werk cache list` and `werk cache purge`](#local-persistence-caches) to inspect
or remove them; deleting just the KV entry preserves the conversation.

`run` and `chat` also accept `--persistence-mode disk|auto|memory|ephemeral`
and `--persistence-ttl-seconds N` (1–2592000). Disk/auto keeps the conversation
across process restarts; memory/ephemeral does not save a transcript or native
snapshot to disk. TTL expires the saved conversation. These options imply
persistence. `--persistence-pin` remains a server option for named Werk Protocol
states; terminal conversations do not create `/werk/v1/prefill` state handles.

### Vision input

Attach one or more images to a compatible vision-language model with repeatable
`--image` values:

```bash
werk --backend auto run vision-model \
  "Inspect this render for clipped text and alignment defects." \
  --image /absolute/path/to/render.png \
  --max-tokens 512 --debug

werk --backend auto chat vision-model \
  --image /absolute/path/to/render.png \
  --no-history --debug
```

The model manifest must advertise image understanding and an image-capable
runtime must pass its probe. For GGUF, the llama.cpp server path additionally
requires a manifest-listed multimodal projector. See
[Vision and visual quality assurance](../integrations/vision.md).

## Text benchmarks

Benchmark an installed model with a fixed prompt and sampling settings:

```bash
werk --backend omlx bench model-id \
  --prompt "Write one English sentence about Rust." \
  --max-tokens 64 --runs 3 --warmups 1 \
  --temperature 0 --top-p 0.95 --seed 42 \
  --json --include-output --debug
```

`--runs` defaults to 5 measured runs after 1 unmeasured `--warmups` run.
Temperature defaults to 0 and seed to 42; omitted `--top-p` inherits the
backend's behavior. A warmup does not guarantee that every later request
reuses its prompt cache. Keep runtime settings and system load consistent
between comparisons.

JSON samples contain token counts, timings, `finish_reason` and
`backend_diagnostics`. Add `--include-output` with `--json` to retain generated
text in each sample's `output` field for quality review. Without that flag,
the field is omitted. Inspect `finish_reason: "length"` for truncation before
comparing answer quality or total duration. `--debug` requests additional
backend diagnostics, including available oMLX expert counters.

The benchmark sends the original structured user message through the selected
backend's prompt handling, as `run` and `chat` do. Runtime-owned chat templates
therefore receive the original message instead of an already formatted prompt.

For HTTP streaming and multi-turn conversations, use the separate
[HTTP chat benchmark](../../utils/benchmarks/README.md). It records time to first
visible text, final API usage when supplied, answers and declared quality checks
through the same endpoint used by Open WebUI and werkStation.

## Media inference

Top-level media command groups are:

```text
werk image generate|edit|upscale
werk video generate|animate|transform|upscale
werk audio generate|transcribe|translate|detect|analyze|transform|embed
```

The model is explicit. Typed commands share task schemas, resolution,
estimation, planning, output publication and diagnostic behavior with the HTTP
service, but do not invoke the HTTP routes internally.

Prompt-capable commands resolve text from an explicit value, a text file,
piped standard input and then an interactive prompt where supported. Generated
outputs go to the requested `--output` destination or Werk's managed output
store. Prompt text is not included in automatic filenames.

See [Media inference](../media-inference.md) for the canonical task tree,
parameters and runnable examples.

## Server

```bash
werk serve \
  --host 127.0.0.1 --port 11434 \
  --model chat-model \
  --image-model image-model
```

The default address is `127.0.0.1:11434`. Authentication is enabled by default
through `--api-key`, `WERK_API_KEY`, `--api-keys` or the default key file.
`--allow-unauthenticated` is intended only for deliberate local development.

Browser CORS is disabled by default. Add exact trusted origins with repeatable
`--cors-origin`; wildcard and opaque `null` origins are rejected.

Enable persistence defaults and supported backend prefix caching:

```bash
werk serve --model chat-model --persistence
```

`--persistence` supplies `auto` retention, `prefer` reuse, no TTL and no pinning
when a `POST /werk/v1/prefill` request omits its top-level `policy` member. It
also supplies `allow_experimental: true` when that member is omitted. For a
local vLLM process started by this server, it defaults vLLM's native automatic
prefix cache on. Werk verifies that the installed vLLM help advertises the
generated flag before starting the process. A remote vLLM endpoint remains
externally managed and receives no generated launch argument.

For local oMLX 0.6.4, persistence mode `auto` or `disk` with reuse other than
`disabled` also enables verified short exact-prefix caching. This applies to
ordinary OpenAI chat and supported tool requests, including when `auto`
routing selects oMLX. The native SSD cache is limited to 4 GB in the private
worker's `cache/prefix-cache` directory and lasts for that worker. It does not
save chat history or guarantee reuse after restart. Use `chat --persistence` for
durable CLI conversation history and its supported native KV cache.

For compatible oMLX expert offload, set the expert cache and thinking default
when starting the service:

```bash
WERK_OMLX_THINKING=0 \
WERK_OMLX_EXPERT_CACHE_MB=8192 \
  werk --backend omlx serve --model chat-model --persistence --verbose
```

The expert cache budget is in MiB and is separate from the native prompt/KV
cache. Expert execution defaults to `grouped`; set
`WERK_OMLX_EXPERT_EXECUTION=serial` before starting the service for a comparison.
Grouped execution batches tensor materialization, retains active expert groups
through GPU evaluation and reduces allocator clearing. These settings apply to
ordinary HTTP chat clients as well as CLI generation. See
[experimental oMLX expert offload](../backends.md#experimental-omlx-expert-offload)
for runtime and model compatibility requirements.

With `serve --verbose`, available expert diagnostics describe the whole
worker's measurement interval. Overlapping requests can contribute to those
counters. Logical read bytes include reads served by the OS file cache and do
not establish physical SSD throughput; nested timing counters are not
independent phases to add together.

The defaults can be selected individually; any granular option implies
`--persistence`:

```bash
werk serve --model chat-model \
  --persistence-mode disk \
  --persistence-reuse prefer \
  --persistence-ttl-seconds 3600 \
  --persistence-pin
```

Persistence mode is `ephemeral`, `memory`, `disk` or `auto`; reuse is
`disabled`, `prefer` or `required`; TTL is 1 through 2592000 seconds. If the
request contains `policy`, that complete object wins, including protocol
defaults for fields omitted inside it. An explicitly supplied
`allow_experimental` value also wins, including `false`.

For a local vLLM launch, `--persistence-reuse disabled` defaults native prefix
caching off. An explicit `--enable-prefix-caching` or
`--no-enable-prefix-caching` in `WERK_VLLM_ARGS` wins over the generated
default. These backend-native cache entries remain opaque: they are not named
Werk state and cannot be listed, moved, persisted or pruned by Werk.

For oMLX, modes `memory` and `ephemeral`, or reuse `disabled`, leave the
additional exact-prefix helper off. TTL and pinning apply to named Prefill
state; `required` reuse does not make an ordinary OpenAI cache miss an error.
Existing native backend caching remains independent of this helper.

These flags do not redirect OpenAI-compatible `/v1` or media
requests through Prefill, add semantic output caching, or enable cross-restart
restore. Exact model/pipeline residency is already automatic in supported
Werk-owned in-process and resident-worker paths. Current named state/prefill
support is experimental and limited to a functionally validated, Werk-managed
llama-server process for the exact installed GGUF model. The backend owns the
opaque runtime state; Werk owns its policy, lifecycle, accounting and
compatibility checks. Inspect `werk runtime capabilities` before relying on it.
The separate model-, pipeline-, and backend-owned reuse paths are listed in the
[execution lifetime and reuse matrix](../concepts/runtime-persistence-and-memory.md#execution-lifetime-and-reuse-matrix).

The route inventory and request contracts are documented in the
[HTTP API reference](../api.md).

## Backend management

```bash
werk backend list
werk backend doctor --debug
werk backend install TARGET
```

The supported install targets, operating-system matrix and manual cleanup
procedure are documented in [Backends](../backends.md). There is currently no
managed `werk backend uninstall` command.
