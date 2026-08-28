# Changelog

All notable changes to Werk1112 are documented in this file. The project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/philipbodenbach/werk1112/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/philipbodenbach/werk1112/compare/v1.3.3...v1.4.0
