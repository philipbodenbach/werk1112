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

### Pull from Hugging Face

```bash
werk pull organization/repository --name local-name
```

Select one file from a multi-quant repository:

```bash
werk pull organization/model-GGUF \
  --file model.Q4_K_M.gguf \
  --name model-q4
```

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

`list` shows summaries and supports metadata filters. `inspect` prints the full
stored/enriched manifest as JSON. Declared task support is not proof that an
installed runtime can execute the model.

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

This removes only the managed copy beneath the active Werk store. Backend
environments, unrelated models and original import sources are separate.

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

Enable server-side persistence defaults for Werk Protocol Prefill requests:

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

Apart from that local-vLLM default, these flags affect only omitted fields on
`/werk/v1/prefill`. They do not redirect OpenAI-compatible `/v1` or media
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
