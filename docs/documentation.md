---
title: Documentation
description: Versioned guides and references for Werk1112.
---

# Werk1112 documentation

[Back to the Werk1112 overview](index.html#documentation)

This is the published index for Werk1112's versioned documentation. The pages
are maintained with the code in the repository's [`docs/` directory][docs-source]
and describe the current repository state.

## Start here

| Need | Page |
| --- | --- |
| Install Werk1112 and run a first request | [Getting started](getting-started.md) |
| Look up a command or global option | [CLI reference](reference/cli.md) |
| Understand every HTTP route | [HTTP API](api.md) |
| Integrate runtime state, memory or prefill/decode | [Werk Protocol 1.0](reference/werk-protocol-v1.md) |
| Understand persistence safety and backend support | [Runtime persistence and memory architecture](concepts/runtime-persistence-and-memory.md) |
| Install or select an inference runtime | [Backends](backends.md) |
| Configure vLLM arguments or OpenAI tool calling | [vLLM launch and tools](backends.md#vllm-launch-arguments-and-tool-calling) |
| Run image, video or audio inference | [Media inference](media-inference.md) |

## Models and configuration

- [Models, manifests and managed storage](concepts/models-manifests-and-store.md)
- [Runtime persistence, memory and capability boundaries](concepts/runtime-persistence-and-memory.md)
- [Tasks, modalities, repository layouts and formats](reference/tasks-and-formats.md)
- [Environment variables](reference/environment-variables.md)

## Integrations and platforms

- [OpenAI-compatible clients](integrations/openai-clients.md)
- [Open WebUI](integrations/open-webui.md)
- [AUTOMATIC1111-compatible clients](integrations/automatic1111.md)
- [ComfyUI integration choices](integrations/comfyui.md)
- [Vision and visual quality assurance](integrations/vision.md)
- [NVIDIA DGX Spark and Nemotron](integrations/dgx-spark.md)
- [AMD Strix Halo, ROCm and Vulkan](integrations/strix-halo.md)

The self-contained [ComfyUI custom-node documentation][comfyui-source] stays
beside the package source so that it can also serve as the package and Registry
documentation.

## Development and releases

- [Building from source](development/build.md)
- [Packaging and releases](development/packaging-releases.md)
- [Changelog][changelog]

The installed `werk --help` output is authoritative for exact CLI flags. The
API router and request structs are authoritative for the HTTP surface until an
OpenAPI description becomes a tested source of truth.

[changelog]: https://github.com/philipbodenbach/werk1112/blob/main/CHANGELOG.md
[comfyui-source]: https://github.com/philipbodenbach/werk1112/blob/main/utils/comfyUI/README.md
[docs-source]: https://github.com/philipbodenbach/werk1112/tree/main/docs
