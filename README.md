# Werk1112

<p align="center">
  <img src="docs/assets/banner_werk.png" alt="Werk1112 startup banner: WERK1112 - Inference Router." />
</p>

Werk1112 is a local-first, multimodal inference runtime and router written in
Rust.
Applications use one CLI and HTTP service; Werk resolves models, parameters,
hardware and installed runtimes, then selects an executable backend.

Werk supports text, image, video and audio workflows without coupling clients
to llama.cpp, vLLM, Candle, MLX, oMLX, ONNX Runtime, Diffusers, Transformers or
architecture-specific companion runtimes.

## Core capabilities

- managed local and Hugging Face model store
- explicit or automatic runtime and accelerator selection
- typed chat, image, video and audio commands
- workload estimation, parameter validation and provenance
- OpenAI-compatible and Anthropic Messages API subsets with text, streaming and
  client tool calling, plus Werk-native media and job APIs
- optional ComfyUI nodes with native IMAGE, VIDEO and AUDIO values
- optional [native n8n nodes (Beta)](utils/n8n/README.md),
  with manual installation, binary media and runtime operations
- experimental [SSD expert offload for oMLX](docs/backends.md#experimental-omlx-expert-offload)
  with a bounded cache and existing ComfyUI/n8n expert controls

Werk is an inference runtime and router, not an agent framework, workflow
engine or GUI.

Save and resume a terminal conversation with any chat backend:

~~~bash
werk chat MODEL --persistence --session project
werk run MODEL "Continue with a short summary" --session project --stream
~~~

Both commands share the same model/session history. Completed turns are saved locally,
including when runtime routing changes. Without `--session`, the name is
`default` for that model. Native KV reuse is separate and backend-dependent;
conversation persistence works even when the backend must recompute the prompt.
See the [CLI reference](docs/reference/cli.md#persistent-terminal-chat).

Inspect and clean up persisted caches:

~~~bash
werk cache list
werk cache purge <CACHE-ID>
werk cache purge --all --dry-run
werk cache purge --all
~~~

The list distinguishes chat KV caches, saved chat histories, oMLX worker
caches and runtime states. `--all` preserves saved histories and skips active
or protected entries. Use `--include-history` to explicitly remove histories
too. See [cache management](docs/reference/cli.md#local-persistence-caches).

For experimental DeepSeek V4 chat with an 8 GiB expert cache and thinking
explicitly disabled:

~~~bash
WERK_OMLX_EXPERT_CACHE_MB=8192 WERK_OMLX_THINKING=0 \
  werk --backend omlx chat mlx-community/DeepSeek-V4-Flash-2bit-DQ --persistence --verbose
~~~

oMLX enables thinking by default for this model; its hidden reasoning can delay
the first visible answer. Omit `WERK_OMLX_THINKING` to preserve that default.
SSD expert offload saves memory but can substantially reduce generation speed.
For ComfyUI or n8n, start `werk --backend omlx serve` and select the options
directly in **WERK Text Config** (ComfyUI) or **WERK Text → Chat Options**
(n8n): thinking `disabled`, expert offload `enabled`, and an expert cache
budget of `8192` MiB. The default `inherit` keeps the server's
settings. See the [chat API options](docs/api.md#omlx-chat-options) for the
request contract and server compatibility check.

## Status

Werk1112 is under active development. Model discovery is broader than model
execution: a repository may be imported and classified even when no installed
runtime can execute its architecture. Use the following before a large run:

~~~bash
werk inspect MODEL
werk doctor --model MODEL --task TASK
~~~

The detailed support levels and known gaps are documented rather than hidden
behind an “all models supported” claim.

## What’s new in v1.6.0

Werk Core, Media Companion, ComfyUI and the new n8n package now share release
version **1.6.0**. ComfyUI and n8n remain Beta integrations.

- Optional local oMLX on Apple Silicon supports text, chat, streaming and
  verified native tool calls. Compatible MLX-LM remains preferred; use
  `--backend omlx` to select oMLX explicitly. Model preflight checks the
  installed oMLX patches, including DeepSeek V4 support, before loading weights.
- Eight native n8n nodes cover discovery, inference, jobs and runtime control,
  with binary media, example workflows and manual custom-directory installation.
- MLX-LM preflight checks the actual loader, architecture and quantization
  metadata, including supported MXFP4 layouts. Routing preserves explicit
  backend/device choices and explains compatible fallbacks.

See the [v1.6.0 changelog](CHANGELOG.md#160---2026-09-07),
[oMLX backend guide](docs/backends.md#optional-local-omlx-backend),
[n8n integration](utils/n8n/README.md) and
[runtime persistence and memory architecture](docs/concepts/runtime-persistence-and-memory.md)
for support boundaries. The oMLX tests use source fixtures and mock servers;
real Apple Silicon and Vontra-checkpoint inference remains to be validated.

## Install

End-user installers install the Werk binary only. Models, drivers, Python
packages and optional inference runtimes remain separate.

Linux or macOS:

~~~bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/philipbodenbach/werk1112/main/scripts/install.sh)"
~~~

Windows PowerShell:

~~~powershell
irm https://raw.githubusercontent.com/philipbodenbach/werk1112/main/scripts/install.ps1 | iex
~~~

Install from source:

~~~bash
cargo +stable install --path . --locked
~~~

Installer options, first-model setup and uninstall behavior are documented in
[Getting started](docs/getting-started.md).

Target-specific source builds use the checked-in Cargo aliases
`build-linux`, `build-linux-strix-halo`, `build-linux-aarch64`,
`build-windows`, and `build-macos-apple-silicon` on the matching host
platform. The dedicated profiles cover AMD Strix Halo and NVIDIA DGX
Spark/GB10. See
[Building from source](docs/development/build.md) for the feature graph,
platform prerequisites and troubleshooting.

## Quick start

Import a local model or pull a Hugging Face repository:

~~~bash
werk import /path/to/model --name local-model
werk pull org/model-repository --name model-name
werk list
werk inspect model-name
~~~

Keep large models on a RAID alongside small models in the normal local store:

~~~bash
werk import /mnt/f/Werk1112/models/wan22-ti2v-5b --name wan22-ti2v-5b --link
# Register all models in an existing collection:
werk import /mnt/f/Werk1112/models --all --link
~~~

`--link` registers existing files without copying them, including an existing
Werk model directory with its metadata. Removing the registration leaves the
external files intact. `--all` discovers separate models directly inside a
collection directory, retaining existing Werk model IDs or using directory
names and file stems. Omit `--link` to copy the collection into the active store.
Ordinary imports still copy files; `--model-home` and
`WERK_HOME` still select the complete store. See
[Models, manifests and the store](docs/concepts/models-manifests-and-store.md).

Run text and media inference:

~~~bash
werk chat local-model

werk image generate IMAGE_MODEL \
  --prompt "A quiet orbital greenhouse"

werk video generate VIDEO_MODEL \
  --prompt "Sunlight breaking through a forest canopy"

werk audio generate speech TTS_MODEL \
  --text "Werk elf zwölf ist bereit." \
  --output speech.wav
~~~

Inspect a rendered page or slide with a compatible vision-language model:

~~~bash
werk --backend auto run VISION_MODEL \
  "Find clipped text, missing controls, overlap and grid misalignment." \
  --image /absolute/path/to/render.png \
  --debug
~~~

Vision is supplied by the model and its processor, not by vLLM itself. Werk
can route compatible requests through llama.cpp with a GGUF projector, optional
vLLM, or the supported MLX-VLM path. See
[Vision and visual quality assurance](docs/integrations/vision.md).

Werk chooses among accepted runtimes when the backend is <code>auto</code>.
Use verbose diagnostics to see the effective request and decision:

~~~bash
werk video generate VIDEO_MODEL \
  --prompt "Clouds moving above a mountain ridge" \
  --backend auto --verbose --debug
~~~

## Backend management

Inspect discovered runtimes and their prerequisites:

~~~bash
werk backend list
werk backend doctor --debug
~~~

Managed installers are explicit:

~~~bash
werk backend install llama-cuda
werk backend install llama-cpu
werk backend install vllm
werk backend install qwen-tts
~~~

Installation support is not the same as verified model execution on every
operating system. See [Backends, installation and platform support](docs/backends.md)
for the complete matrix, fallback rules, managed paths and current uninstall
limitations.

## HTTP service

Authentication is enabled by default:

~~~bash
werk auth api-key generate
export WERK_API_KEY="replace-with-generated-key"
werk serve --model local-model
~~~

OpenAI-compatible clients use:

~~~text
http://127.0.0.1:11434/v1
~~~

Werk exposes OpenAI-compatible Chat Completions (`POST /v1/chat/completions`)
and Anthropic-compatible Messages (`POST /v1/messages`) subsets on the same
server, alongside OpenAI-inspired media routes, Werk-native discovery/jobs/outputs
and a small AUTOMATIC1111 compatibility
surface. A separate `/werk/v1` protocol provides versioned runtime capability,
state and memory control without changing those existing routes. These classes
are intentionally documented separately.

`POST /v1/messages` also supports Anthropic-style text, streaming and client tool
cycles on the same server. See [Anthropic clients](docs/integrations/anthropic-clients.md)
for SDK setup, protocol limits and Qwen/GLM test commands.

~~~bash
curl -fsS http://127.0.0.1:11434/v1/models \
  -H "Authorization: Bearer $WERK_API_KEY"
~~~

OpenAI function tools are supported through compatible local or remote vLLM
servers. Werk preserves normal and streaming tool-call structures, while the
operator remains responsible for selecting the model-specific vLLM tool parser
and enabling any required vLLM flags. Other chat adapters reject tool requests
explicitly instead of ignoring them. See the
[chat API contract](docs/api.md#post-v1chatcompletions) and
[vLLM launch configuration](docs/backends.md#vllm-launch-arguments-and-tool-calling).

See the [HTTP API reference and coverage matrix](docs/api.md) for all 33
method/path operations, exact request fields, task coverage, responses,
authentication, limits, persistence and known gaps.

## Documentation

| Topic | Document |
| --- | --- |
| Published documentation | [GitHub Pages documentation](https://philipbodenbach.github.io/werk1112/documentation.html) |
| Release notes and upgrade changes | [CHANGELOG.md](CHANGELOG.md) |
| Documentation home and wiki roadmap | [docs/README.md](docs/README.md) |
| Installation and first run | [docs/getting-started.md](docs/getting-started.md) |
| CLI command groups and semantics | [docs/reference/cli.md](docs/reference/cli.md) |
| HTTP API contract and coverage | [docs/api.md](docs/api.md) |
| Werk Protocol 1.0 HTTP contract | [docs/reference/werk-protocol-v1.md](docs/reference/werk-protocol-v1.md) |
| Runtime persistence, memory and capability boundaries | [docs/concepts/runtime-persistence-and-memory.md](docs/concepts/runtime-persistence-and-memory.md) |
| Backends, routing, installation and OS support | [docs/backends.md](docs/backends.md) |
| Tasks, modalities, repository layouts and formats | [docs/reference/tasks-and-formats.md](docs/reference/tasks-and-formats.md) |
| Models, manifests and managed storage | [docs/concepts/models-manifests-and-store.md](docs/concepts/models-manifests-and-store.md) |
| Environment-variable index | [docs/reference/environment-variables.md](docs/reference/environment-variables.md) |
| Media tasks, parameters, jobs and examples | [docs/media-inference.md](docs/media-inference.md) |
| Building from source | [docs/development/build.md](docs/development/build.md) |
| Packaging and releases | [docs/development/packaging-releases.md](docs/development/packaging-releases.md) |
| DGX Spark and AMD Strix Halo | [docs/integrations/dgx-spark.md](docs/integrations/dgx-spark.md) · [docs/integrations/strix-halo.md](docs/integrations/strix-halo.md) |
| Vision models and rendered-output QA | [docs/integrations/vision.md](docs/integrations/vision.md) |
| Client integration guides | [docs/README.md#integrations](docs/README.md#integrations) |
| ComfyUI custom nodes | [utils/comfyUI/README.md](utils/comfyUI/README.md) |

The versioned files under <code>docs/</code> are the source of truth. A public
wiki may mirror tutorials later, but should not replace versioned API and
backend contracts.

## Platform overview

Werk's release packaging currently targets:

- Linux x86_64
- Linux x86_64 AMD Strix Halo (`gfx1151` profile)
- Linux aarch64 (DGX Spark/GB10 only)
- Windows x86_64
- macOS Apple Silicon

Runtime support depends on the backend, accelerator and upstream packages.
For example, llama.cpp Metal is macOS-only, local vLLM is native-Linux-only,
and Qwen-TTS currently has Linux with NVIDIA CUDA as its primary documented
path. The detailed and experimental combinations are listed in
[docs/backends.md](docs/backends.md).

## Development

Common checks:

~~~bash
cargo +stable fmt --all --check
cargo +stable check --all-targets
cargo +stable test
python -m unittest runtime.test_werk_media_companion
python -m pytest utils/comfyUI/tests
~~~

The project deliberately keeps optional backend dependencies outside the main
Werk process where version conflicts would otherwise affect unrelated
architectures.

## License

Unless a file or subdirectory explicitly states otherwise, the current
Werk1112 source tree—including the core runtime, server, APIs, media
companions, utilities, and ComfyUI nodes—is licensed under the
[Elastic License 2.0](LICENSE). This repository-wide default also applies to
future integrations added here unless they explicitly declare another license.

ELv2 permits use, modification, preparation of derivative works, and
redistribution, subject to its terms. Its restrictions include not providing
the software to third parties as a hosted or managed service where users receive
access to a substantial set of the software's features or functionality. The
[`LICENSE`](LICENSE) file contains the authoritative terms.

Source versions and releases published before this licensing change remain
available under the license terms under which they were originally published.
Third-party dependencies, models, runtimes, and other materials retain their
own licenses and are not relicensed by this repository-wide default.
