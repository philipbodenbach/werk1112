# ComfyUI parity and n8n adaptations

The reference is Werk/ComfyUI **1.5.1**, checked against `nodes.py`,
`runtime_nodes.py`, their request builders, protocol client, and the actual
Rust routes/DTOs in this checkout. `__init__.py` merges **20 inference and
configuration registrations + 10 runtime registrations = 30 public nodes**.
The test suite checks the table against those registrations without importing
Torch or ComfyUI.

All eight n8n nodes display **(Beta)** and use internal node version **1**.
Names below identify node operations, not separately serialized config nodes.
IDs and parameter names are workflow contracts: incompatible changes require
node versioning or a documented migration.

| Public ComfyUI registration | n8n equivalent | Preserved options and outputs; adaptation |
| --- | --- | --- |
| `WerkConnection` | WERK API credential | Base URL, masked API key, explicit unauthenticated mode, real read-only discovery test. n8n stores credentials; no connection JSON items or environment-key fallback. |
| `WerkServerInfo` | Discovery / Server Info | Models plus capabilities, joined by exact model ID; structured metadata replaces JSON strings. |
| `WerkImageModels` | Discovery / Models; Image model selector | `image-generation`, installed/declared/available distinction, exact IDs and task statuses. Expressions/manual IDs replace ComfyUI sockets. |
| `WerkVisionModels` | Discovery / Models; Vision model selector | `image-understanding`, authoritative task discovery and unavailable reasons, never model-name heuristics. |
| `WerkImageParameters` | Discovery / Parameters | Complete schema for explicit task/model/backend, preserved as structured JSON. |
| `WerkRoutingConfig` | Image/Video/Audio routing group | backend, accelerator, device, precision, quantization, profile, quality, performance_preference, fallback_policy, parameter_policy, all four offload switches, attention_backend, compile, timeout_seconds. Inherit is omitted; explicit false is retained. No separate JSON-only node. |
| `WerkVisionConfig` | Vision options group | Temperature, top-p, completion-token budget, safe integer seed, image detail and ordered stop strings. No ineffective media-routing controls on chat. |
| `WerkImageConfig` | Image configuration and additional parameters | Dimensions, count/batch distinction, steps, guidance, seed, output/response formats, style, VAE switches; validated image namespace and list operations. Unselected fields inherit. |
| `WerkImageGenerate` | Image / Generate | Prompt, negative prompt, model/config/routing; image bytes, model/task/seed and sanitized result metadata. One binary item per result with `pairedItem`; embedded base64 output IDs are not presented as durable downloads. |
| `WerkVideoModels` | Discovery / Models; Video model selector | Explicit `video-generation` or `image-to-video` task, declared/available models and statuses. |
| `WerkVideoParameters` | Discovery / Parameters | Full parameters for selected video task, model and backend. |
| `WerkVideoConfig` | Video configuration and additional parameters | Dimensions, count/batch, frame count/rate, steps, guidance, seed, container, temporal VAE switch, video namespace and routing. |
| `WerkVideoGenerate` | Video / Generate or Image to Video; Jobs | Text-to-video or exactly one input image, persisted job and output metadata. Native bytes replace ComfyUI `VIDEO`; submit-only and separate get/wait/cancel/download support n8n workflow scheduling. |
| `WerkAudioModels` | Discovery / Models; Audio model selector | All 19 concrete audio tasks, declared versus runtime-available models and statuses. Task visibility does not promise an adapter. |
| `WerkAudioParameters` | Discovery / Parameters | Full task/model/backend schema; schema output is not an execution configuration. |
| `WerkAudioConfig` | Audio configuration and additional parameters | Duration, variations, seed, sample rate/channels, output format, instrumental, voice, speed, language, speaking style, task namespaces and routing. Zero sample rate/channels and TTS seed zero retain sentinel omission semantics. |
| `WerkAudioGenerate` | Audio / Generate | Audio/music generation and TTS; TTS uses `input` and `async: true`, rejects negative prompt. Native binary outputs replace tensor waveform dictionaries. |
| `WerkAudioProcess` | Audio / Process | Voice conversion, stem separation, enhancement, editing; `input_audio` binary and optional voice-conversion `reference_audio`, text/config/routing, output bytes and metadata. No transcoding. |
| `WerkAudioAnalyze` | Audio / Analyze | Speech-to-text/translation; event/activity/speaker/language/emotion detection; captioning/diarization/classification/understanding; embeddings. `stt`/`audio` namespaces, input role, text and structured JSON/NDJSON results preserved. Non-audio artifacts are not mislabeled as audio. |
| `WerkVisionAnalyze` | Vision / Analyze | Prompt, optional system message, ordered images and chat options. Binary helper reads become ordered image data URLs in exactly one user message; assistant text/usage/finish reason/model/tool calls remain structured. |
| `WerkRuntimeInfo` | Runtime / Info and Capabilities | Strict protocol info/capabilities, backend, limits and all six exact capability statuses. No legacy fallback. |
| `WerkPersistencePolicy` | Runtime / Prefill & Decode persistence group | Optional full policy: auto/ephemeral/memory/disk, prefer/disabled/required, pin and optional TTL. Disabled group is absent; enabled TTL zero omits TTL and means no TTL. |
| `WerkRuntimeStates` | Runtime / States | Model/tier/page/cursor filters, validated server bounds, complete safe state summaries and next cursor. Structured arrays replace JSON/newline strings. |
| `WerkStateControl` | Runtime / State Action | Explicit ID; pin/unpin/promote/demote/evict; valid action-specific target tier; dry-run defaults true; changed/state metadata. |
| `WerkStatePrune` | Runtime / Prune States | Explicit IDs, nonempty filters or separately confirmed all; server limits; dry-run true; matched/removed/bytes/result metadata. Applies only to runtime states. |
| `WerkMemoryStatus` | Runtime / Memory | Host/accelerator memory, pressure and topology as reported, no invented telemetry. |
| `WerkRuntimeExperts` | Runtime / Experts | Model/tier/page/cursor, expert residency telemetry, IDs and next cursor; `externally_managed` read-only semantics. |
| `WerkExpertControl` | Runtime / Expert Action | Explicit model and unique IDs, prefetch/pin/unpin/evict, action-specific target, dry-run true, capability/limit gates; no inferred operative MoE support. |
| `WerkPrefill` | Runtime / Prefill & Decode | Text or ordered messages, model, complete optional policy and explicit experimental opt-in. Single-use handoff stays in a local variable within one execute call; safe state/reuse/tier/expiry metadata remains available. |
| `WerkDecode` | Runtime / Prefill & Decode | Decode budget/options and text/token metadata; capabilities rechecked after prefill. No persisted handoff, separate decode node, static-data token cache or automatic retry. |

## Deliberate interface differences

- **Text** adds ordinary non-streaming chat beyond ComfyUI parity. It is a
  workflow node, not an n8n AI-Agent chat-model subnode. Tool calls are returned
  as data; this package never executes them.
- **Jobs** exposes get/wait/cancel/output-download independently of submission.
  Output downloads require an output ID, never a job or result ID.
- n8n optional parameter collections keep absent fields absent. ComfyUI's
  default-filled config widgets are not mechanically copied into every request.
  JavaScript seeds outside the safe-integer range are rejected explicitly.
- n8n binary storage can be filesystem-backed or external. Only the official
  helpers read or create data; ComfyUI tensors and Python objects are not used.
- ComfyUI's non-JSON `WerkStateHandoff` cannot safely cross persisted n8n nodes.
  Combining prefill/decode intentionally removes that unsafe workflow boundary.
- Image edits/inpainting and a generic editor for every canonical job task are
  outside this first Beta. No server route is added to implement this package.

## Contract clarifications checked in source

`/v1/models` returns `data`; `/v1/capabilities` returns `models`. Model IDs join
exactly, while task hyphens/underscores normalize only for comparison.
`/v1/chat/completions` does not honor media-routing overrides. TTS defaults to
raw bytes unless `async: true` selects the job path. Runtime service versions
are independent of Protocol 1.0 and optional response version headers remain
compatible when absent. Embedded image `b64_json` can outlive its temporary
Werk output ID, which may already have been removed. Current production expert
adapters do not promise operational expert residency; metadata remains metadata.

Text discovery has an additional server-specific distinction:
`src/inference_service/service.rs` augments generation-backend readiness for
`image-understanding`, while other `available_tasks` can come from media
readiness. A working text-only chat backend can consequently have a
media-unavailable text status. Text validates the exact installed model and
its declared text/vision task (or the empty legacy task list accepted by
`chat.rs`); the actual non-streaming chat request decides execution readiness.
It does not incorrectly use a media-adapter availability gate for text.
