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
