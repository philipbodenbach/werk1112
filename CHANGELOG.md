# Changelog

All notable changes to Werk1112 are documented in this file. The project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- oMLX: use the same automatic device/model-dependent expert cache budget for CLI chat and Serve on supported DeepSeek V4 checkpoints. Preserve explicit small budgets and SSD offload; `0` selects native loading.
- Report uncached prefill token counts/rates in CLI verbose output and native phase timings in Serve logs for comparable performance measurements.

### Added

- `werk run` now shares chat sessions, transcript/native KV persistence,
  streaming, and context handling with `chat`. JSON requests expose the existing
  OpenAI tool/vision/runtime controls and canonical media inference service,
  including image, audio, video and embedding tasks, parameters, inputs, routing,
  and output publication. Structured output supports JSON and streamed NDJSON.
- Conversation storage mode and TTL options for both `run` and `chat`, with the
  same managed vLLM/oMLX prefix-cache defaults as `serve`.

- Native persistent terminal-chat snapshots through the existing llama.cpp slot
  adapter, with runtime/model compatibility namespaces, corruption checks and
  backend-reported prefix hits. Newer servers' response counters are accepted
  when idle-slot counters reset after completion.
- Opt-in `werk backend install llama-cuda-offload` builds a pinned experimental
  CUDA MoE-cache runtime through the existing installer and adapter; supported
  PLE tables can use native lazy reads. See the documented WSL test limitation
  and differences from oMLX's automatic budgets and row cache.
- `werk import DIRECTORY --all` imports a collection of model directories and
  supported model files, retaining existing Werk model IDs and metadata. Combine
  it with `--link` to register a RAID collection without copying weights.
  Collection imports check model ID and destination conflicts before writing;
  individual repositories remain intact, including components and shards.
- `werk import --link` registers existing external model files without copying
  weights, including existing Werk model directories and their metadata. Mix
  local and RAID models in one store; `werk list` identifies external storage,
  and removal preserves external files. Whole-store `--model-home` and
  `WERK_HOME` selection remains available.
- Grouped oMLX expert execution with batched tensor materialization, protected
  active groups and reduced allocator flushing; `WERK_OMLX_EXPERT_EXECUTION=serial`
  retains the prior path for comparison. Verbose diagnostics expose worker
  interval cache, logical-read and evaluation counters.
- Optional chat SSE usage summaries via `stream_options.include_usage`, benchmark
  output/finish diagnostics, and a reusable HTTP multi-turn quality/performance harness.
- Local `werk cache list` and `werk cache purge <CACHE-ID>` / `--all`, with
  storage-kind filters, size and protection status, JSON output and dry runs.
  Saved chat histories require explicit inclusion; active caches and pinned
  runtime states are protected from cleanup.
- Backend-independent `werk chat --persistence` with named conversations,
  automatic resume, explicit reuse policy, private atomic archives and
  protection against concurrent writers and incomplete streamed turns.
  Local vLLM also receives a validated automatic-prefix-cache default.
- Optional native exact-prefix KV caching for persistent oMLX 0.6.4 chats,
  with checkpoint/runtime binding, durable writes and reported cache hits.
  Other chat backends retain portable conversation persistence.
- Experimental SSD-backed expert loading for packed affine DeepSeek V4 MLX
  checkpoints in a private oMLX 0.6.4 worker. `WERK_OMLX_EXPERT_CACHE_MB`
  enables a bounded cache while ordinary oMLX loading remains the default.
  Active workers expose expert telemetry and RAM prefetch, pin, unpin and
  eviction through the existing Werk, ComfyUI and n8n expert controls.
- Optional `WERK_OMLX_THINKING=0|1` controls thinking for oMLX model templates
  that support it, including DeepSeek V4. Unset preserves existing defaults.
- Per-request oMLX thinking and expert-cache settings through the chat API,
  n8n WERK Text options, and new ComfyUI text model/config/generation nodes.
  Explicit settings require the server's `api.chat.omlx_options` capability;
  inherited settings preserve existing workflows. Expert controls follow the
  worker used for the selected model, including request-specific cache budgets.

### Fixed

- Download all matching shards when selecting a split GGUF with `werk pull`,
  including automatic selection. Reject incomplete shard sets before downloading
  and count the complete selected variant in memory estimates.
- Buffer GGUF architecture and chat metadata reads to avoid excessive small
  reads and long delays on mounted storage, including Windows drives in WSL.
- Show each model's absolute local storage path in `werk list`, align columns
  to their contents and use labeled blocks when the table exceeds the terminal width.
- Reuse successful oMLX compatibility probes across chat requests and sessions,
  with bounded caching and model/runtime dependency invalidation, avoiding
  repeated Python imports before each HTTP response.
- Claim the Serve port before loading the default model, so an occupied port
  fails immediately without starting a second large model worker.
- Preserve original structured messages in `werk bench` for native templates,
  avoiding a benchmark-only second application of the chat template.
- Allow cleanup of empty legacy oMLX worker cache directories when native
  process markers confirm their owners have exited. Files, active workers and
  unverifiable legacy layouts remain protected.
- Accepted standard uv MLX launchers and Homebrew Python console launchers
  alongside the previously supported entry-point formats.
- oMLX timing statistics use the runtime's prompt and generation durations
  when available, so hidden reasoning is no longer reported as prompt work.

## [1.6.0] - 2026-09-07

### Added

- Added optional local oMLX text, chat, streaming and native tool-call support
  through an already installed CLI on Apple Silicon. Compatible MLX-LM keeps
  priority; model-specific oMLX preflight enables compatible fallbacks and
  explicit `--backend omlx` selection. Werk reuses owned oMLX processes without
  installing the runtime, copying weights or exposing named KV-state support.
- Added the `n8n-nodes-werk1112` 1.6.0 Beta under `utils/n8n`:
  eight native discovery, text, image, vision, video, audio, jobs and runtime
  nodes; shared credentials, native binary data, combined in-memory
  prefill/decode, manual custom-directory installation and example workflows.
  The package remains private and manually installed. See the
  [n8n guide](utils/n8n/README.md).

### Changed

- Synchronized Werk Core, Werk Media Companion, the ComfyUI package and the
  n8n package at `1.6.0`. ComfyUI and n8n retain their Beta status.

### Fixed

- Fixed MLX preflight rejecting the regular MLX-LM loader's positional `lazy`
  argument by binding calls to the installed `load_model` signature while
  retaining resolver-override checks.
- Restored supported MXFP4 `quantization_config` metadata (including GPT-OSS)
  when the installed loader provides the matching normalization path, while
  retaining architecture and runtime quantization checks.
- Applied runtime priorities to actual selection while retaining model, task,
  platform and explicit device/backend constraints. Excluded unsupported Candle
  architectures and known incompatible packed safetensors quantization.
- Added bounded MLX model preflight in the configured execution environment:
  architecture resolution, model metadata and installed quantization support
  are checked before weight loading, including the existing Gemma4 compatibility
  path. Importing `mlx-lm` alone no longer establishes model compatibility.
- Preserved structured routing reasons through CLI and service execution and
  report compatible fallbacks on stderr without verbose/debug flags, with
  repeated server diagnostics deduplicated. Media runtime retries preserve the
  actual route and failure reasons; streaming text/tool errors do not restart
  generation. Added simulated DeepSeek metadata and routing regressions; the
  separate oMLX integration supplies DeepSeek V4 support through the installed
  upstream runtime.

### Compatibility notes

- Werk Protocol remains `1.0`; media protocol, transport, manifest, workflow
  and persisted-state schema versions are unchanged.
- oMLX requires an installed compatible CLI on Apple Silicon. Explicit
  `--backend omlx` still runs model preflight and fails without switching
  backends. Raw `F8_E8M0` checkpoints are rejected before loading because the
  upstream loader can rewrite their headers; use an MLX-converted checkpoint.
- Native oMLX tool calls require verified model-parser wiring. The initial
  verified path is DeepSeek V4 DSML; required/named tool choices, explicit
  `parallel_tool_calls` and strict schemas remain unsupported. Model residency
  does not expose Werk named KV snapshots, Prefill or restore operations.
- oMLX acceptance tests cover source fixtures and simulated HTTP workers.
  Real Apple Silicon inference with a small model and the Vontra checkpoint
  remains unvalidated in the Linux/WSL development environment.

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

[Unreleased]: https://github.com/philipbodenbach/werk1112/compare/v1.6.0...HEAD
[1.6.0]: https://github.com/philipbodenbach/werk1112/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/philipbodenbach/werk1112/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/philipbodenbach/werk1112/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/philipbodenbach/werk1112/compare/v1.3.3...v1.4.0
