# Model support audit — 2026-10-13

This audit checks the exact five requested products against vendor sources. An
upstream release is **not** proof of executed Werk inference. All new executable
paths remain `IMPLEMENTED_UNTESTED` until a real model/API smoke test succeeds;
mock tests validate adapters, not model quality or account access.

## Repository architecture and regression boundary

Werk is a Rust CLI/server, with canonical model manifests, task/capability
metadata, runtime planning, generation adapters and a separate media service.
`src/backend/mod.rs` inventories implemented runtimes: persistent llama.cpp
server plus optional legacy bindings; Candle 0.9.2; managed/external vLLM;
MLX/MLX-VLM/oMLX; ONNX Runtime; a Transformers compatibility adapter; experimental
Burn; and Python media/text-analysis companions. TensorRT, OpenVINO, TensorFlow
and CoreML format recognition is not evidence of a universal executable backend.
ComfyUI and Automatic1111 are client integrations/API compatibility surfaces,
not five additional native model engines. There was no general hosted-vendor
provider abstraction before this change.

Existing FLUX.1 Schnell flow:

1. `werk image generate` / canonical `werk run`, or `/v1/images/generations`.
2. Shared `InferenceService` resolves the manifest, parameters and runtime plan.
3. `CompanionMediaBackend` sends the execution to its resident `CompanionClient`.
4. `runtime/werk_media_companion.py::load_diffusers_pipeline` selects
   `diffusers.AutoPipelineForText2Image.from_pretrained(..., local_files_only=True)`.
   Diffusers resolves the checkpoint's `FluxPipeline` configuration. There is no
   separate native Candle FLUX implementation in this repository.
5. The companion caches the configured pipeline, generates an image and writes
   an output artifact; the Rust output store supplies bytes/base64/URLs.

The FLUX 3 adapters do not replace or redirect that pipeline. A dedicated
Schnell loader regression fixture and the existing companion/media tests cover
this boundary. A real Schnell generation was not repeated in this environment.

## Verified upstream facts and proposed/implemented routes

| Product / exact identity | Input → output; capabilities | Weights, license, runtime and hardware | Official API / decision |
| --- | --- | --- | --- |
| Aleph Alpha `Aleph-Alpha/Kolibri-1` | Text → text; reasoning, tools, streaming through vLLM. German/English tokenizer and native chat template. 1,048,576 advertised context; ≤262,144 recommended. | 78B total / 3.46B active, FP8 block-quantized MoE. Apache-2.0. `kolibri1` / `Kolibri1ForCausalLM`; `aleph-alpha-inference>=1` supplies matching vLLM/plugin/parsers. About 78 GB weights; vendor minimum includes 2×A100 80GB or 1×H200. | Extend existing vLLM path. No official hosted model endpoint verified. No GGUF/MLX/Candle/ONNX compatibility claim. [Model card](https://huggingface.co/Aleph-Alpha/Kolibri-1), [config](https://huggingface.co/Aleph-Alpha/Kolibri-1/blob/main/config.json), [plugin](https://github.com/Aleph-Alpha/aleph-alpha-inference). |
| Mistral Large 4, API `mistral-large-4` | Text/images → text; tools, JSON/schema output and SSE. Public preview; vendor page advertises 1M context. Account/deployment limits remain authoritative. | Product page: granular MoE, 1.05T total / 52B active, 1.6B vision encoder. Actual downloadable artifacts are pending, so checkpoint architecture, tokenizer files, weight license and local engine compatibility cannot be validated. | Implement hosted `/v1/chat/completions`, Bearer key, `random_seed`, structured options and tool messages. [Model](https://docs.mistral.ai/models/mistral-large-4-0), [API](https://docs.mistral.ai/api/endpoint/chat), [release announcement](https://mistral.ai/news/mistral-large-4/). |
| FLUX 3 Action, `black-forest-labs/flux-3-action-droid` | Synchronized wrist/left/right RGB frames + instruction + 8 state values → 32×8 absolute joint/gripper commands. Robot action prediction, not image generation or chat. | 7B world-action model; DROID root and optimized variants, separate SO-101 package. DROID checkpoint: FLUX Kommunity License v1.0; code has separate terms. Shared encoders include Qwen3-VL-4B. Native Linux/NVIDIA PyTorch/NATTEN environment, per-package saved config/normalization. | Implement resident DROID worker using `FluxActionPolicy`; typed JSON predictions only. No official hosted API verified. SO-101's 42×6 history/degree contract is deliberately not treated as DROID. [Setup and schemas](https://github.com/black-forest-labs/flux-action/blob/main/docs/setup.md), [DROID card](https://huggingface.co/black-forest-labs/flux-3-action-droid), [checkpoint license](https://huggingface.co/black-forest-labs/flux-3-action-droid/blob/main/LICENSE.md). |
| FLUX 3 Image, `flux-3-image` | Prompt plus optional 1–10 references → image. Generation/editing/layout instructions, aspect-ratio enum and resolution tiers. | No publicly downloadable checkpoint or compatible local runtime verified for this exact image model. FLUX.1 Schnell architecture cannot be assumed compatible. | Implement `https://api.bfl.ai/v1/flux-3-image`, `x-key`, returned regional polling URL, moderation and binary download. No seed, width, height, negative prompt, output-format or mode field in this endpoint. [Schema](https://docs.bfl.ai/api-reference/utility/generate-an-image-with-flux-3), [guide](https://docs.bfl.ai/flux_3/flux3_image_generate), [product](https://bfl.ai/models/flux-3-image). |
| Eleven v4, `eleven_v4` | Text/voice pairs → speech audio; dialogue, language, stability/similarity, seed, normalization and streaming. | Proprietary service; no public weights/local engine verified. Model overview advertises 90+ languages / 10,000 characters. Dialogue endpoint recommends ≤2,000 total characters and ≤10 voices; adapter enforces that conservative endpoint limit. | Implement `/v1/text-to-dialogue` and `/stream`, `xi-api-key`, validated codec/rate/bitrate. Models page directs v4 to dialogue; product example also shows TTS, so the adapter uses the documented dialogue contract. [Models](https://elevenlabs.io/docs/overview/models), [dialogue](https://elevenlabs.io/docs/api-reference/text-to-dialogue/convert), [streaming](https://elevenlabs.io/docs/api-reference/text-to-dialogue/stream), [OpenAPI](https://api.elevenlabs.io/openapi.json). |

The official [Mistral weight destination](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)
shows an upcoming release with an ETA of **2026-10-31**, after this audit date.
The older `Mistral-Large-4.0-1T05-A52B` URL redirects there. No conversion,
sharding configuration, weight license or local support is invented for it.

## Feature negotiation and risk controls

- Model identity, provider and physical storage are separate: a provider source
  has no local checkpoint format; local registrations keep their inventory.
- Text/tool messages use existing chat abstractions. Action outputs use a
  separate typed endpoint with explicit units, dimensions and `executed:false`.
- BFL/Eleven extensions expose their actual contracts instead of silently
  translating unsupported Diffusers/OpenAI parameters.
- New support status is independent of credentials configured, backend installed
  or preload success. Discovery never calls a paid generation endpoint.
- Kolibri uses the vendor parsers and native template; reasoning effort is
  forwarded to template kwargs. Its synthetic 32-token preload disables thinking
  so hidden reasoning alone cannot exhaust the short warmup.
- Existing startup preload ordering, deduplication, failure continuation and
  retention remain in place. DROID workers use the same pinned runtime-cache
  transaction; cloud preloads only prepare the HTTPS client and validate local
  credential syntax. They do not promise provider readiness or warmed weights.
- POST network/5xx failures are not automatically replayed: a generation may
  already have been billed. Explicit 429 rejection is retried with bounded
  backoff. Redirects are disabled; BFL result downloads never receive the key.

See [implementation and test evidence](model-support-implementation.md) and the
[status matrix with runnable commands](model-support-matrix.md).
