# Tasks and formats

Werk separates four facts that are easy to conflate:

1. the canonical inference task;
2. the model's input and output modalities;
3. the repository layout;
4. the weight or model format.

All four participate in runtime selection. A model being catalogued with a
known task or format does not prove that an installed backend can execute that
combination. See [Backends, routing and platform support](../backends.md) for
runtime selection and [Models, manifests and the store](../concepts/models-manifests-and-store.md)
for how these values are detected and persisted.
[Environment variables](environment-variables.md) lists the runtime and
backend discovery overrides referenced by those pages.

## Naming and serialization

CLI task and layout parsers accept either hyphens or underscores. The generic
job API also parses task names this way. Human-facing examples therefore use
hyphens, such as `image-generation`, while manifest and response JSON serialize
the same enum as `image_generation`.

The stable modalities are:

- input: `text`, `image`, `video`, `audio`;
- output: `text`, `image`, `video`, `audio`, `embedding`.

## Canonical tasks

The following table is derived from the `InferenceTask` enum and its modality
and parameter-namespace mappings. “Prompt” means the normalized request must
carry prompt text in addition to any required media input.

| CLI/API task name | Required input | Output | Parameter namespace | Prompt |
| --- | --- | --- | --- | --- |
| `text-generation` | text | text | `text` | yes |
| `text-embedding` | text | embedding | `text` | yes |
| `image-understanding` | image | text | `text` | yes |
| `image-generation` | text | image | `image` | yes |
| `image-editing` | image | image | `image` | yes |
| `image-variation` | image | image | `image` | no |
| `image-inpainting` | image | image | `image` | yes |
| `image-outpainting` | image | image | `image` | yes |
| `image-upscaling` | image | image | `image` | no |
| `video-generation` | text | video | `video` | yes |
| `image-to-video` | image | video | `video` | yes |
| `video-to-video` | video | video | `video` | yes |
| `video-inpainting` | video | video | `video` | yes |
| `video-extension` | video | video | `video` | yes |
| `video-upscaling` | video | video | `video` | no |
| `frame-interpolation` | video | video | `video` | no |
| `audio-generation` | text | audio | `audio` | yes |
| `music-generation` | text | audio | `audio` | yes |
| `song-continuation` | audio | audio | `audio` | no |
| `song-variation` | audio | audio | `audio` | no |
| `text-to-speech` | text | audio | `tts` | yes |
| `speech-to-text` | audio | text | `stt` | no |
| `speech-translation` | audio | text | `stt` | no |
| `audio-event-detection` | audio | text | `audio` | no |
| `voice-activity-detection` | audio | text | `audio` | no |
| `speaker-identification` | audio | text | `audio` | no |
| `language-identification` | audio | text | `audio` | no |
| `speech-emotion-recognition` | audio | text | `audio` | no |
| `audio-captioning` | audio | text | `audio` | no |
| `speaker-diarization` | audio | text | `audio` | no |
| `audio-classification` | audio | text | `audio` | no |
| `audio-understanding` | audio | text | `audio` | yes |
| `audio-embedding` | audio | embedding | `audio` | no |
| `voice-conversion` | audio | audio | `audio` | no |
| `stem-generation` | audio | audio | `audio` | no |
| `stem-separation` | audio | audio | `audio` | no |
| `audio-enhancement` | audio | audio | `audio` | no |
| `audio-editing` | audio | audio | `audio` | yes |

One manifest can declare several tasks. Task inference from repository metadata
is heuristic; the runtime planner still checks task, format, layout,
architecture, accelerator and explicit parameters before accepting a
candidate.

The generic media companion currently registers execution for the task subset
listed in [Media companion execution support](../media-inference.md#execution-support).
The remaining canonical tasks can still be represented, imported, inspected
and estimated even when no generic execution adapter is registered.

## Model formats

Format detection uses the managed file inventory. The checks are ordered, so a
repository containing several kinds of weights is classified by the first
matching rule. `werk select-file` can select another tracked weight and refresh
the format and derived metadata.

| Format | Detection evidence | Current routing status |
| --- | --- | --- |
| GGUF | `.gguf` | Persistent llama.cpp server for CPU/CUDA/ROCm/Vulkan/Metal when the matching runtime is available; Candle is a legacy compatibility path for selected architectures. |
| MLX | MLX metadata in a safetensors header or `.npz`; an MLX path-name hint is a later fallback | External `mlx-lm`/MLX routes when configured. |
| Safetensors | `.safetensors` not classified as MLX | Text routes are architecture/runtime dependent; compatible Transformers or Diffusers media repositories can use the media companion. |
| ONNX | `.onnx` | ONNX Runtime route when a compatible Werk runner is installed. |
| TensorRT | `.engine` or `.plan` | Catalog/import only; no registered TensorRT execution backend. |
| OpenVINO | both `.xml` and `.bin` | Catalog/import only; no registered OpenVINO execution backend. |
| Core ML | `.mlmodel` or a path containing `.mlpackage` | Catalog/import only; no registered Core ML execution backend. |
| PyTorch | `.pt`, `.pth` or `pytorch_model.bin` | Model-dependent media execution through the companion; no generic text route. |
| TensorFlow | `.pb`, `.ckpt`, a file named `checkpoint`, or a `.ckpt-` path | Catalog/import; there is no registered general TensorFlow execution route. |
| Unknown | no preceding rule matched | Catalog/import and model probing only; Werk does not assume an execution backend. |

The `backend` string stored in a manifest is a human-facing hint. Runtime
eligibility comes from the runtime registry and planner, not from that string
alone.

## Repository layouts

Repository layout is independent of weight format. Manifest JSON stores these
values with underscores; the CLI accepts the corresponding hyphenated form.

| Manifest value | Meaning and detection |
| --- | --- |
| `single_file` | One recognized model artifact without a repository component layout. |
| `gguf` | A GGUF model collection. |
| `transformers` | Safetensors/PyTorch repository with `config.json`, `tokenizer.json` or `tokenizer_config.json`. |
| `diffusers` | Root `model_index.json`, or at least two known component roots including `transformer` or `unet`. |
| `mlx` | Repository classified as MLX. |
| `onnx_bundle` | Multiple ONNX files or ONNX plus bundle metadata/tokenizer/subdirectories. |
| `tensorrt_engine` | TensorRT engine/plan collection. |
| `custom` | No more specific layout rule matched. |

Known component roots include `transformer`, `unet`, `vae`, `scheduler`, text
encoders/tokenizers, `encoder`, `decoder`, `vocoder`, `feature_extractor`,
`controlnet` and `adapter`. A multi-component repository remains one installed
model; a Diffusers component file is not treated as the repository's primary
model path.

## Generated media formats

The included companion can directly write these result encodings when their
required codec libraries are installed:

| Result kind | Formats |
| --- | --- |
| Image | `png`, `jpeg`, `webp` |
| Video | `mp4`, `gif` |
| Generated audio and TTS | `wav`, `flac`, `ogg` |
| ASR and structured audio analysis | `json`, `text`, `srt`, `vtt`, `tsv` |

These are output encodings, not model formats. See
[Media inference](../media-inference.md) for codec prerequisites and parameter
support.

## Source of truth

- canonical task semantics: [`src/capabilities/task.rs`](../../src/capabilities/task.rs)
- modalities and layouts: [`src/capabilities`](../../src/capabilities)
- model format and manifest detection: [`src/model_store.rs`](../../src/model_store.rs)
- registered runtime constraints: [`src/backend/mod.rs`](../../src/backend/mod.rs)
