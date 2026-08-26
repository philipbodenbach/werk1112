# Werk1112 documentation

This directory is the versioned documentation source for Werk1112. It is
reviewed with the code and describes the current repository state.

The project README remains intentionally short. Detailed contracts, support
matrices and operational behavior belong here.

## Start here

| Need | Page |
| --- | --- |
| Install and run a first command | [Getting started](getting-started.md) |
| Look up CLI commands and semantics | [CLI reference](reference/cli.md) |
| Understand every HTTP route | [HTTP API](api.md) |
| Install or select a backend | [Backends](backends.md) |
| Run Werk and Nemotron on NVIDIA DGX Spark | [DGX Spark](integrations/dgx-spark.md) |
| Run Werk on AMD Strix Halo | [AMD Strix Halo](integrations/strix-halo.md) |
| Look up tasks, modalities, layouts or formats | [Tasks and formats](reference/tasks-and-formats.md) |
| Understand manifests and managed model storage | [Models, manifests and the store](concepts/models-manifests-and-store.md) |
| Look up an environment variable | [Environment variables](reference/environment-variables.md) |
| Run image, video or audio inference | [Media inference](media-inference.md) |
| Build Werk from source | [Build guide](development/build.md) |
| Package release artifacts | [Packaging and releases](development/packaging-releases.md) |
| Install and use ComfyUI nodes | [ComfyUI integration](../utils/comfyUI/README.md) |

## Documentation vocabulary

Werk uses precise support labels:

| Label | Meaning |
| --- | --- |
| Implemented | The Werk request, schema and routing path exists. |
| Executable | At least one registered runtime adapter can execute the task. |
| Model-dependent | Execution additionally depends on the concrete architecture, repository layout and installed packages. |
| Compatible subset | Werk intentionally implements only documented parts of an external protocol. |
| Experimental | The path exists but lacks a complete platform/architecture test matrix or upstream guarantee. |
| Prepared | The task is represented by the typed contract but has no generic execution adapter yet. |
| Planned | No public contract should be assumed yet. |

Importing and classifying a model does not imply that it is executable.
Likewise, successfully installing a Python package does not prove that its
model works on every accelerator.

## Current reference pages

### API

[api.md](api.md) documents:

- all 21 paths and 23 method/path operations
- OpenAI-compatible, OpenAI-inspired, Werk-native, Comfy and A1111 surfaces
- authentication, CORS, content types and body limits
- chat, image, video, audio and generic-job request contracts
- task-to-endpoint coverage
- direct, raw-file and persisted-job responses
- errors, output retention and cancellation
- explicit compatibility gaps and sensitive metadata behavior

### CLI

[reference/cli.md](reference/cli.md) documents the command groups, global
controls, model lifecycle, authentication, estimation, diagnostics, artifacts,
text/media inference and serving semantics. The installed `--help` output is
authoritative for exact flags.

### Backends

[backends.md](backends.md) documents:

- automatic routing and explicit backend constraints
- fallback policies
- all managed install targets
- prerequisites and managed locations
- Linux, Windows, WSL and macOS support levels
- Qwen-TTS isolation and platform status
- the current absence of a managed uninstall command

### Media

[media-inference.md](media-inference.md) currently contains the complete media
guide: canonical tasks, schemas, parameter provenance, estimation, planning,
the Python companion, jobs, outputs, Wan video examples and audio smoke tests.
It will be split into smaller image, video, audio and operations pages later.

### Models and configuration

[reference/tasks-and-formats.md](reference/tasks-and-formats.md) is the
canonical task, modality, repository-layout and model-format index.

[concepts/models-manifests-and-store.md](concepts/models-manifests-and-store.md)
documents import, pull, removal, manifest schema v2, selection, inspection and
optimized artifacts.

[reference/environment-variables.md](reference/environment-variables.md)
collects the environment variables read by Werk, its runtime adapters,
installers and the ComfyUI integration.

### Development and releases

[development/build.md](development/build.md) documents the Cargo feature
graph, checked-in build aliases, native platform prerequisites, custom
acceleration builds and source-build troubleshooting.

[development/packaging-releases.md](development/packaging-releases.md)
documents the exact packaging scripts, target/archive matrix, checksums,
installer naming contract and current manual release process.

### Integrations

Client-specific setup and compatibility boundaries are documented separately:

- [OpenAI-compatible clients](integrations/openai-clients.md)
- [Open WebUI](integrations/open-webui.md)
- [AUTOMATIC1111-compatible clients](integrations/automatic1111.md)
- [ComfyUI integration choices](integrations/comfyui.md)
- [NVIDIA DGX Spark and Nemotron](integrations/dgx-spark.md)
- [AMD Strix Halo, ROCm and Vulkan](integrations/strix-halo.md)

The self-contained [ComfyUI custom-node documentation](../utils/comfyUI/README.md)
stays beside the package source because it is also the package and Registry
documentation. The ComfyUI integration guide links to it instead of duplicating
its installation, node and workflow reference.

## Wiki roadmap

The repository docs should remain canonical even if a GitHub Wiki is added.
A separate wiki repository is useful for discoverability and tutorials, but a
poor sole source for versioned API contracts because it can drift independently
from releases.

The intended information architecture is:

~~~text
docs/
├── README.md
├── getting-started.md
├── concepts/
│   ├── architecture-and-routing.md
│   ├── models-manifests-and-store.md
│   └── parameters-estimation-and-jobs.md
├── guides/
│   ├── image.md
│   ├── video.md
│   ├── audio.md
│   └── serving.md
├── backends/
│   ├── index.md
│   ├── support-matrix.md
│   ├── managed-installation.md
│   └── qwen-tts.md
├── integrations/
│   ├── openai-clients.md
│   ├── open-webui.md
│   ├── automatic1111.md
│   ├── comfyui.md
│   ├── dgx-spark.md
│   └── strix-halo.md
├── reference/
│   ├── cli.md
│   ├── api.md
│   ├── openapi.yaml
│   ├── tasks-and-formats.md
│   └── environment-variables.md
├── operations/
│   ├── security.md
│   ├── outputs-retention.md
│   └── troubleshooting.md
└── development/
    ├── build.md
    └── packaging-releases.md
~~~

The first migration step deliberately keeps three flat pages—this index,
API and backends—plus the existing media guide. They can move into the final
tree without changing their content.

## Planned documentation work

- split image, video and audio guides out of media-inference.md
- add operations/security and retention pages
- add a machine-readable OpenAPI description
- add a CI check comparing OpenAPI operations with the Axum router
- publish a tested hardware matrix separately from upstream-supported claims

Historical README material remains available through Git history. It is not
duplicated into a second monolithic reference while the focused pages are
being written.

## Keeping docs honest

When behavior changes:

1. update code and tests;
2. update the relevant versioned page in the same change;
3. label architecture- or platform-dependent support explicitly;
4. distinguish request validation from successful model loading;
5. avoid calling a route OpenAI-compatible when it uses a Werk-specific body;
6. keep examples runnable and name any required model layout or package.

The CLI help output is authoritative for available flags. The API router and
request structs are authoritative for the HTTP surface until OpenAPI becomes a
tested source of truth.
