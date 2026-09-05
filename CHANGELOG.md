# Changelog

All notable changes to Werk1112 are documented in this file. The project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.5.1] - 2026-09-05

### Changed

- Advanced the synchronized Werk Core, Werk ComfyUI package and Werk Media
  Companion release version to `1.5.1`.

### Fixed

- Fixed macOS builds by isolating Linux-specific resource detection and using
  Darwin Mach VM statistics for bounded available-memory telemetry.

## [1.5.0] - 2026-09-05

### Added

- Added Werk Protocol 1.0 and the `werk runtime` CLI as a backend-aware
  runtime-control layer for capability discovery, state policy and lifecycle,
  memory telemetry, pressure-aware runtime management and explicit state
  maintenance.
- Added a crash-safe runtime-state catalog with integrity validation,
  API-key-principal isolation when authentication is enabled, bounded opaque
  handoffs and dry-run-capable lifecycle operations; CLI and ComfyUI lifecycle
  controls default to dry-run where applicable.
- Added capability-gated state and expert-control abstractions. Capability
  discovery reports unsupported backend operations explicitly; `unsupported`
  is a complete, truthful result and is never promoted to success.
- Added experimental split prefill/decode primitives for the functionally
  validated, Werk-managed `llama-server` path, including opaque handoffs and
  process-generation-bound state snapshots.
- Added ten ComfyUI runtime-control nodes for discovery, persistence policies,
  state inspection and maintenance, memory telemetry, expert controls and
  split prefill/decode. Together with the 20 existing inference nodes, the
  package now registers 30 nodes.
- Added OpenAI-compatible vLLM tool calling for local and remote vLLM,
  including unchanged forwarding of tool definitions, tool choice,
  parallel-tool configuration and tool-result history, plus structured normal
  and streaming tool-call responses.
- Added `werk temp path`, `werk temp list` and dry-run-capable
  `werk temp purge` commands for narrowly scoped temporary-store maintenance.

### Changed

- Expanded Werk from an inference router into an inference runtime and router.
  The established routing architecture remains a core capability and the new
  runtime/control layer is additive.
- Starting with this release, the shipped Werk Core, Werk ComfyUI package and
  Werk Media Companion versions are synchronized at `1.5.0`. Historical
  ComfyUI Registry releases retain their independent version numbers.
- Expanded process-local model residency from the existing generic media cache
  to separate generic-media and managed-Qwen workers plus bounded Transformers
  and ONNX GenAI model caches. Positive media probe and estimate results are
  cached separately without caching failures.
- Hardened `WERK_VLLM_ARGS` for locally started vLLM processes: POSIX
  shell-word parsing builds a direct argv, Werk-owned launch flags are rejected,
  malformed input fails before launch and non-reserved arguments retain their
  values and order. Non-empty local launch arguments are rejected for remote
  vLLM endpoints instead of being ignored.
- Made tool calling a required routing capability: automatic routing selects
  a compatible vLLM path, explicit vLLM routing remains strict and other
  production chat backends reject tool-required requests instead of silently
  dropping them. The vLLM transport continues to pass through messages,
  `max_tokens`, `temperature`, `top_p`, `stop` and `seed`, and now also carries
  the implemented tool configuration and tool-result history.
- Local vLLM launch arguments can configure native vLLM functionality,
  including Automatic Prefix Caching (APC). When `werk serve --persistence`
  supplies an APC default, Werk adds the native flag only in the absence of an
  explicit user choice and verifies that the installed runtime advertises it.
  APC remains owned and delegated to vLLM; Werk exposes the observed capability
  state but cannot name, snapshot, restore, move or prune vLLM KV-cache entries.
  Remote vLLM receives no generated launch flag.

### Fixed

- Fixed audio workload estimates so canonical `audio.variations` and its
  legacy alias scale the estimated output size correctly.
- Hardened runtime-state pruning so a failed backend release reports an error
  and restores unreleased catalog entries instead of leaving a partial
  in-memory purge.

### Compatibility notes

- Werk Protocol remains `1.0`. The media protocol, transport version, manifest,
  workflow and persisted-state schema versions are unchanged by the `1.5.0`
  product release.
- Runtime persistence and restore are capability-gated and backend-specific;
  v1.5.0 does not introduce a universal KV format, universal RAM/VRAM state
  movement or cross-restart restoration. Current named-state support remains
  experimental and snapshots cannot outlive their validated managed
  `llama-server` process generation.
- Split prefill/decode is experimental and available only where the active
  runtime reports the required capabilities. Expert residency/control is an
  implemented abstraction, but no current production backend advertises
  operational expert movement; `unsupported` is the expected truthful result.
- Memory telemetry preserves unknown observations, and pressure-aware
  reservation or movement remains unavailable when an adapter cannot provide
  the required bounded accounting and lifecycle hooks.
- Model and pipeline residency is process-local and separate from named
  runtime state. Remote vLLM and other externally managed facilities keep
  their own lifecycle and configuration.
- Werk forwards tool-call contracts but does not execute tools, choose a vLLM
  parser or automatically enable model-specific tool-choice flags.

## [1.4.0] - 2026-08-28

### Added

- Added typed multimodal inference contracts, CLI commands and HTTP routes for
  image, video and audio workloads, including parameter provenance, workload
  estimates and capability-aware planning.
- Added persistent media jobs, managed outputs, cancellation, retention and
  authenticated output retrieval.
- Added visual chat and image-understanding support for ordered image content,
  including GGUF projectors through llama.cpp, supported Qwen/GLM VLMs through
  optional vLLM, and the supported MLX-VLM path.
- Added the media companion for Diffusers and Transformers pipelines and a
  managed Qwen3-TTS VoiceDesign adapter.
- Added Werk-native ComfyUI nodes for model discovery, routing, image, video,
  audio, speech and visual-inspection workflows.
- Added OpenAI-compatible image generation and multimodal chat support, plus a
  documented AUTOMATIC1111 compatibility subset.
- Added dedicated NVIDIA DGX Spark/GB10 and AMD Strix Halo release profiles,
  runtime diagnostics and unified-memory-aware estimation.
- Added structured backend readiness results with actionable installation,
  configuration and unsupported-adapter recommendations.

### Changed

- Expanded automatic routing across model format, architecture, task,
  accelerator, runtime availability and explicit fallback policy. Text-only
  backends are no longer accepted for requests carrying visual input.
- Forwarded model-requested media parameters without arbitrary generic upper
  caps. Concrete model, runtime, representation and codec limits still apply
  and are reported separately.
- Hardened release installers and archives with platform-specific artifact
  selection, checksum verification and archive-content validation.
- Reworked the documentation into versioned API, backend, media, platform and
  integration references.
- Changed the project license from Apache License 2.0 to Elastic License 2.0.
  Review the new license terms before upgrading or redistributing Werk1112.

### Compatibility notes

- Model discovery remains broader than executable runtime support. Use
  `werk doctor --model MODEL --task TASK --debug` before relying on a newly
  classified model or task.
- Optional models, accelerator drivers and companion runtimes are provisioned
  separately from the Werk binary.
- The ComfyUI package keeps its independent `0.1.0` Registry version; it does
  not follow the Werk binary version.

[Unreleased]: https://github.com/philipbodenbach/werk1112/compare/v1.5.1...HEAD
[1.5.1]: https://github.com/philipbodenbach/werk1112/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/philipbodenbach/werk1112/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/philipbodenbach/werk1112/compare/v1.3.3...v1.4.0
