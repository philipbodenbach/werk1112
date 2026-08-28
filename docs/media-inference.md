# Werk1112 media inference

[Documentation home](README.md) · [HTTP API](api.md) ·
[Backends and platform support](backends.md)

Werk1112 is an inference router, not a workflow engine. A request names one
model and one concrete task. Werk normalizes its inputs, resolves defaults,
validates parameters, estimates the workload, scores runtime candidates,
executes the best accepted runtime (with policy-controlled runtime retry), and
manages outputs and metadata.

## Public commands

```text
werk chat MODEL
werk image generate|edit|upscale MODEL
werk video generate|animate|transform|upscale MODEL
werk audio generate speech|music|sound MODEL
werk audio transcribe|translate MODEL
werk audio detect event|voice|speaker|language|emotion MODEL
werk audio analyze caption|diarize|classify|understand MODEL
werk audio transform voice|separate|enhance|edit MODEL
werk audio embed MODEL
werk serve
```

The old `werk run` parser remains hidden for compatibility. New applications
should use `chat` for text and the typed media commands for generated files.
The former `audio generate MODEL`, `audio speak`, and `audio separate` forms
remain available as compatibility aliases.

Interactive chat and media commands print the Werk1112 startup banner. During
local inference, the CLI renders a transient modality-specific activity line
on terminal stderr (including an audio waveform). This is an indeterminate
liveness indicator, not backend progress or a percentage. It is cleared before
the result is printed and omitted when terminal output is redirected.

For a direct CLI request without `--output`, Werk publishes completed media
directly under `WERK_HOME/outputs` with a generated name such as
`tiny-sd-image-generation-1784968751-807fe1a83ad7fb22.png`. The name combines a
sanitized model ID, task, Unix timestamp, and random suffix; it never includes
prompt text. `--output PATH` selects another file or directory instead. Werk
publishes all outputs first and then removes the request's temporary managed
result directory, avoiding a persistent duplicate. If publication fails, the
temporary result remains intact. Persisted jobs and explicit HTTP URL responses
retain their managed result metadata and output IDs. Embedded Base64/text
responses are removed after encoding, and synchronous raw output is removed
after its response stream finishes.

Generative prompt priority is:

1. `--prompt`, `--text`, or `--lyrics`;
2. its corresponding `--*-file`;
3. piped standard input;
4. interactive terminal input.

## Local Wan video example

Werk's video commands express canonical tasks; they do not select a pipeline
class by matching a model name. The requested model remains explicit, and the
planner accepts a runtime only when its adapter registry, repository-layout
probe, task registry, dependencies, accelerator, and parameter support all
match. `--backend auto` leaves that choice to the accepted candidates. It is
not a claim that every video architecture or checkpoint layout is executable.
An explicit backend constrains the candidates, and `--fallback-policy none`
prevents retry through a different accepted runtime.

This distinction matters for Wan2.2 TI2V 5B:

- [`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
  contains the native Wan checkpoint layout;
- [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
  contains the local Diffusers pipeline layout used by the bundled companion.

The native repository can be cataloged, but it has no Diffusers
`model_index.json`. The local companion therefore reports that no executable
adapter accepts the layout instead of passing it to
`DiffusionPipeline.from_pretrained` and failing ambiguously. Pull the Diffusers
variant before running either task:

```bash
werk pull Wan-AI/Wan2.2-TI2V-5B-Diffusers --name wan22-ti2v-5b
werk inspect wan22-ti2v-5b
werk doctor --model wan22-ti2v-5b --task video-generation
werk doctor --model wan22-ti2v-5b --task image-to-video
```

Text-to-video smoke test:

```bash
werk video generate wan22-ti2v-5b \
  --prompt "A quiet lunar sunrise beyond an orbital station" \
  --negative-prompt "text, watermark, static frame" \
  --width 1280 --height 704 --frames 121 --fps 24 \
  --steps 50 --guidance 5 \
  --backend auto --precision bf16 --allow-cpu-offload \
  --output wan22-t2v.mp4 --verbose --debug
```

Image-to-video smoke test:

```bash
werk video animate wan22-ti2v-5b \
  --image station.png \
  --prompt "A slow, stable camera orbit around the station" \
  --negative-prompt "text, watermark, abrupt camera shake" \
  --width 1280 --height 704 --frames 121 --fps 24 \
  --steps 50 --guidance 5 \
  --backend auto --precision bf16 --allow-cpu-offload \
  --output wan22-i2v.mp4 --verbose --debug
```

`video generate` selects the `video-generation` task. `video animate` plus its
required `--image` selects `image-to-video`. For video tasks, the companion
asks the installed Diffusers task registry for a compatible pipeline class. If
that registry has no unambiguous mapping, the repository's own `_class_name`
is the fallback; the first cold load or call can still reject a task that the
concrete pipeline does not implement. This keeps the route architecture-neutral
without pretending unknown families are supported.

The Wan examples request bf16 for the main pipeline components. The companion
detects `AutoencoderKLWan` from the repository metadata and keeps that VAE in
fp32, independent of the repository name.

The [official Wan2.2 configuration](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)
uses 121 frames at 24 FPS, 50 sampling steps, guidance 5, and flow shift 5. The
Diffusers repository already records `flow_shift: 5.0` in its
[scheduler configuration](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/blob/main/scheduler/scheduler_config.json),
so the portable request above inherits it rather than sending an unsupported
pipeline-call keyword. The
[official model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) describes
1280x704 or 704x1280 as its 720P TI2V sizes and supports both T2V and I2V. The
repository download is about 34.2 GB. Wan's native single-GPU command requires
at least 24 GB VRAM with model offload, dtype conversion, and T5 on CPU, and
the project reports under nine minutes for a five-second 720P clip on a
consumer GPU without special optimization. These are upstream reference
figures, not guarantees for another Diffusers version or Werk route. Offload
also needs sufficient host RAM, and `--allow-cpu-offload` is permission for the
planner rather than proof that a hook was installed; `--verbose` reports the
active hook after execution.

For a smaller T2V-only transport and encoder check, the official
[`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
layout uses the same typed Werk path:

```bash
werk pull Wan-AI/Wan2.1-T2V-1.3B-Diffusers --name wan21-t2v-1.3b
werk video generate wan21-t2v-1.3b \
  --prompt "Clouds moving above a mountain ridge" \
  --width 832 --height 480 --frames 81 --fps 15 \
  --steps 50 --guidance 5 \
  --backend auto --precision bf16 --allow-cpu-offload \
  --output wan21-smoke.mp4 --verbose --debug
```

The [official Diffusers model card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
uses 832x480, 81 frames, guidance 5, and a 15 FPS export; its bundled
[scheduler configuration](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/blob/main/scheduler/scheduler_config.json)
contains `flow_shift: 3.0`. These Diffusers values should not be replaced with
the native Wan runner's differently parameterized `sample_shift`. Wan's
[official 1.3B model card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
reports 8.19 GB VRAM and about four minutes for a five-second 480P clip on an
RTX 4090 without quantization. It recommends 480P because 720P is less stable.
The Diffusers repository is still about 28.9 GB, so this option saves more
accelerator memory and generation time than download time. It does not validate
Wan2.2's hybrid I2V route, new VAE, or 720P behavior, so the Wan2.2 test remains
the acceptance test.

## Audio tasks and smoke tests

The audio command hierarchy separates user intent from the model and backend:

```text
generate speech|music|sound
transcribe
translate
detect event|voice|speaker|language|emotion
analyze caption|diarize|classify|understand
transform voice|separate|enhance|edit
embed
```

Each leaf still names one installed model. Werk derives declared tasks from
repository configuration and architecture metadata, then probes registered
Diffusers or Transformers adapters. `--backend auto` scores only compatible
candidates; an explicit backend is honored, and `--fallback-policy none`
prevents backend retry. A repository can therefore be cataloged for a task
without being executable when no generic adapter accepts its layout.

Short, independent smoke tests:

```bash
# VITS text-to-speech
werk pull facebook/mms-tts-deu --name mms-tts-deu
werk audio generate speech mms-tts-deu \
  --text "Werk elf zwölf ist bereit." \
  --output speech.wav --verbose --debug

# MusicGen text-to-music
werk pull facebook/musicgen-small --name musicgen-small
werk audio generate music musicgen-small \
  --prompt "Five-second cinematic technology ident, restrained analogue synth pulse" \
  --duration 5 --seed 1112 \
  --output music.wav --verbose --debug

# Whisper transcription and speech-to-English translation
werk pull openai/whisper-tiny --name whisper-tiny
werk audio transcribe whisper-tiny --input speech.wav \
  --output-format text --output transcript.txt
werk audio translate whisper-tiny --input speech.wav \
  --output-format text --output translation.txt

# AudioSet classification
werk pull MIT/ast-finetuned-audioset-10-10-0.4593 --name ast-audioset
werk audio analyze classify ast-audioset \
  --input field-recording.wav --top-k 5 --output events.json
```

Review each model's license before use; in particular the referenced MMS-TTS
and MusicGen checkpoints are non-commercial. Generation output is
`wav`/`flac`/`ogg`; ASR, classification, audio-to-text, and embedding adapters
write structured text or JSON. `werk doctor --model MODEL --task TASK --debug`
checks metadata and dependencies without loading the full model. The first
cold execution remains the definitive model-registry and memory check.

## CLI diagnostics

Every typed image, video, and audio command accepts two independent,
combinable diagnostic flags:

```bash
# Performance and output statistics
werk image generate tiny-sd --prompt "a small service robot" --verbose

# Request resolution and inference routing
werk video animate wan-i2v --image first-frame.png \
  --prompt "slow camera movement" --debug

# Both views for one request
werk audio generate music musicgen --prompt "quiet analogue ambience" \
  --verbose --debug
```

`--verbose` is the performance/result view. It is printed after execution and
uses values that make sense for the concrete task rather than copying chat's
token counters:

- common values include measured total/service/publication time, the selected
  runtime, actual output count and bytes, and the workload fit/confidence with
  accelerator and host peak estimates clearly marked as estimates;
- image values include actual dimensions/count, effective steps and seed, plus
  seconds per image and generated megapixels per second when inference timing
  is available;
- video values keep playback FPS separate from generated-frames-per-second and
  include actual dimensions, frame count, and duration when the encoder
  reports them;
- generated audio and TTS values include actual output duration, sample rate,
  channels, and real-time performance when those measurements are available;
- transcription and other audio-input tasks report only safe structural and
  timing information, never the transcript itself.

Audio real-time factor is measured inference wall time divided by produced
audio duration; values below `1.0` are faster than real-time. When one request
produces multiple variations, Werk labels the throughput-oriented value
`aggregate RTF` because it divides by their combined duration. Stem outputs do
not use this metric because summing parallel stem durations would be
misleading.

Backends may additionally measure model loading, inference, and encoding as
separate phases. Werk prints a phase only when that phase was actually
measured. It does not relabel an entire backend call as inference time, derive
fake percentages, or turn a missing measurement into zero.

Offload diagnostics deliberately distinguish intent from reality. `planned
offload` comes from the Rust execution plan and is printed before model
loading. `active offload` is emitted only after successful runtime
configuration; its values are `none`, `model-cpu`, or `sequential-cpu`, and a
non-`none` value confirms a concrete Diffusers hook. A permission such as
`--allow-cpu-offload` alone is never reported as an active hook. When a plan
uses offload, post-execution verbose
output labels the original fit and peak estimates as `before offload`; they are
the values that caused routing to select the strategy, not fabricated
measurements of memory retained by the running pipeline.

If routing succeeds but every accepted runtime later fails while loading or
executing the model, either diagnostic flag prints the attempted runtime,
outcome, and elapsed attempt time before the normal error. Backend error text
is not copied into that timing block.

`--debug` is the request/routing view. It reports the canonical task, resolved
routing policy, workload fit, candidate runtimes with scores and rejection
reasons, selected backend, fallback and permitted degradation state, explicit
and effective parameter sources, and safe backend details such as adapter,
device, dtype, and translated parameter names. When accelerator capacity is
known it shows that value beside the estimated peak. If no permitted strategy
can satisfy the limit, the plan has no selected runtime and debug reports
`preflight outcome: blocked before runtime execution`. The command
`werk doctor --model MODEL --task TASK` remains useful as a non-executing
preflight, while `werk parameters MODEL --json` provides the complete
parameter support matrix.

Both reports are written to stderr; normal result and durable output-path lines
stay on stdout. `--verbose` and `--debug` do not imply one another, and using
both emits both reports. Debug mode disables the transient activity animation
so diagnostic lines remain stable.

Werk-authored diagnostic sections are safe to paste into an issue by default.
They never print raw prompts, negative prompts, lyrics, TTS input, initial ASR
prompts, hotwords, transcription text, inline base64 media, URL query strings,
or raw private input/model paths. Output paths continue to follow the normal
CLI output contract. A separate error or warning originating in a third-party
runtime can contain runtime-specific detail and should still be reviewed
before sharing.

## Canonical tasks

The manifest and API use typed tasks rather than a matrix of booleans:

- text generation and embedding, image understanding;
- image generation, editing, variation, inpainting, outpainting, and upscaling;
- video generation, image-to-video, video-to-video, inpainting, extension,
  upscaling, and frame interpolation;
- audio and music generation, song continuation/variation, TTS, ASR and speech
  translation, event/voice/speaker/language/emotion detection, captioning,
  diarization, classification, prompted understanding, embeddings, voice
  conversion, stem generation/separation, enhancement, and editing.

One model can declare several tasks. Input modalities (`text`, `image`,
`video`, `audio`) and output modalities (`text`, `image`, `video`, `audio`,
`embedding`) are recorded separately.

## Manifest schema v2

Schema-v2 fields are flattened into the existing manifest JSON:

- `schema_version`, `family`, `architecture`, and `repository_layout`;
- `tasks`, `input_modalities`, and `output_modalities`;
- first-class components and their files, format, precision, and quantization;
- detected generation defaults and parameter constraints;
- compatible runtime hints and optimized artifacts.

Old manifests deserialize with schema version 1 and are enriched from the
installed repository in memory. Existing identity, source, file inventory, and
selected model path remain intact.

Supported layouts are `single_file`, `gguf`, `transformers`, `diffusers`,
`mlx`, `onnx_bundle`, `tensorrt_engine`, and `custom`. Diffusers detection uses
`model_index.json` and well-known component roots:

```text
transformer  unet  vae  scheduler  text_encoder  text_encoder_2
tokenizer    tokenizer_2  encoder  decoder  vocoder
feature_extractor  controlnet  adapter
```

Components remain part of one installed model.

## Effective parameters and provenance

Transport requests contain overrides. The resolver produces an effective value
for every parameter descriptor:

```text
system → task → family → model → runtime → hardware/quality
       → saved profile → request → backend adjustment
```

Every effective value records the winning source. Booleans have inherited,
explicitly enabled, and explicitly disabled states. Internal list overrides
distinguish inherit, replace, add, and clear.

`werk parameters MODEL --backend auto --json` and
`GET /v1/parameters?task=TASK&model=MODEL&backend=auto`
return descriptors with path, CLI flag, type, label, category, default, range,
allowed values, repeatability, advanced status, and memory/quality/runtime
impact. With a model, manifest defaults/constraints and per-runtime parameter
support are included.

Generic Werk schemas do not impose policy upper bounds on explicit numeric
generation values. Defaults are used only when a value is omitted. A maximum
reported by model metadata remains visible as a capability hint; an explicit
override that exceeds it adds a warning and is forwarded unchanged so the
concrete backend can either execute it or return its own capability error. Genuine type,
format, minimum, model-context, memory-safety, and transport-security checks
remain enforced.

Backends report parameters as `native`, `translated`, `emulated`, `ignored`,
`unsupported`, or `model_dependent`. Explicit ignored/unsupported values fail
under `strict` (the API default), warn under `warn`, and continue under
`permissive`.

## Estimate and planning

Media estimates distinguish:

- download and weight payload;
- accelerator and host peak;
- output size;
- fit (`fits`, `tight`, `likely_oom`, or `unknown`);
- confidence (`exact`, `backend_measured`, `architecture_model`, `heuristic`,
  or `unknown`);
- assumptions, warnings, and recommendations.

Image estimates scale with pixels, batches/count, VAE behavior, and offload.
Video estimates additionally scale with frames and temporal windowing. Audio
estimates scale with duration, sample rate, channels, variations, and stems.

The scored planner checks model task, runtime task, repository layout,
family/architecture probe, the adapter/task registry, runtime availability,
accelerator, explicitly set parameters, and workload fit. A model can be
listed and inspected even when its layout has no executable local adapter; in
that case planning rejects it with the recorded reason. It distinguishes:

- backend fallback: the same model through another runtime;
- execution degradation: offload, tiling/windowing, or a slower attention path;
- model/quality downgrade: a recommendation that is never silently executed.

The model/task probe also reports `task_readiness`. This keeps five different
conditions separate: a ready adapter, a verified fallback, a known managed
backend that can be installed, a task for which no execution adapter has been
implemented, and an unavailable environment or model layout. Missing generic
Python dependencies are reported as mandatory packages plus explicit
`any_of`/`all_of` alternative groups without synthesizing an unsafe install
command. These facts are diagnostic only: Werk still executes the explicitly
named model and never substitutes another model automatically.

For CUDA, Werk obtains the accelerator-memory limit from `nvidia-smi`, with
`WERK_ACCELERATOR_MEMORY_BYTES` as an explicit override. Heterogeneous
multi-GPU configurations without a stable GPU UUID intentionally report the
limit as unknown rather than assigning another card's capacity. If an
estimated peak reaches or exceeds a known limit, exactly one explicitly
permitted offload strategy may keep the GPU candidate eligible. The projected
post-offload host working set must also remain below known available host RAM;
offload cannot mask an existing host-memory OOM. Without a viable strategy,
Werk rejects the plan before creating an output request directory or invoking
the backend. `--fallback-policy degrade` allows inherited degradation
permissions; an explicit `--allow-cpu-offload` permits that strategy even under
the default backend-fallback policy. If either required capacity cannot be
detected or projected, Werk reports it as unknown and cannot promise the
corresponding preflight guarantee.

The companion advertises offload only for CUDA/ROCm Diffusers adapters:
image/video pipelines and Diffusers-layout audio. Transformers audio, TTS, and
ASR adapters are routed without a fictional offload capability; the companion
also validates this again before execution.

## Media companion

Rust remains the control and routing plane. The included Python companion uses
a versioned JSON process protocol with:

```text
health  capabilities  probe-model  estimate  execute
```

Under `werk serve`, media execution runs through one persistent, serialized
worker. Lightweight health, model-probe, and estimate preflights use independent
one-shot calls so concurrent requests do not fail behind a long generation.
The worker performs lazy, local-only Diffusers/Transformers execution and
caches one configured media model by default. Diffusers image/video/audio
pipelines and Transformers audio pipelines or processor/model pairs share this
same bound. The first compatible request is cold; subsequent requests with the
same model and runtime configuration are warm. Prompt, seed, size, steps, and
count are request-local and do not invalidate the cache. Model, task adapter,
device, dtype, offload/tiling configuration, and LoRA changes may select a
different entry and therefore cause a cold load. At the default cache size, the
existing entry is evicted before a replacement is loaded.

`WERK_MEDIA_PIPELINE_CACHE_SIZE` controls the maximum number of resident
media entries and defaults to `1`; `0` disables the model cache while the
worker can remain persistent. Cached models retain VRAM and/or host RAM
until eviction or server shutdown. Execution metadata exposes
`model_cache_hit` and `model_load_seconds` for warm/cold diagnostics. A worker
crash or request timeout discards the process and cache, and the next request
starts cold. The resident transport never replays the same `execute` frame;
Werk's higher-level fallback policy may still try another accepted runtime
candidate.

The companion sets Hugging Face offline variables and passes
`local_files_only=True`; it never installs a package or downloads model
weights. `WERK_MEDIA_COMPANION` can point to a compatible executable, while
`WERK_MEDIA_PYTHON` chooses the Python interpreter for the included adapter. A
legacy external companion that does not support the persistent-worker transport
automatically uses the one-shot protocol. `WERK_MEDIA_ACCELERATOR` can
explicitly select `cuda`, `rocm`, `mps`/`metal`, or `cpu`. MLX media models and
native framework checkpoints without a registered execution adapter remain
catalogable but are not routed to Diffusers merely because they contain
safetensors or a generic `config.json`.

`werk doctor` reports the protocol and optional dependencies. Missing
Diffusers, Transformers, Pillow, audio/video codecs, or accelerator packages
only disables affected tasks. Accelerator availability comes from the exact
Python/PyTorch process selected for the media companion; the report includes
CUDA/ROCm/MPS availability and device details. Host probing, including WSL's
`/dev/dxg`, is used only when an older external companion does not report
accelerators. Candle is not currently a Diffusers image fallback; unsupported
media GPU routes fall back to the compatible CPU companion.

Direct MP4 output needs NumPy plus PyAV, a system `ffmpeg`, or
`imageio-ffmpeg`. Tasks that consume a source video additionally need PyAV, or
`imageio` together with `imageio-ffmpeg`, for decoding. The capability probe
does not treat `imageio` alone as proof that an encoder is installed.

### Execution support

| Task group | Catalog / inspect / estimate | Companion execution |
| --- | --- | --- |
| Image generation/edit/inpaint/upscale | Yes | Diffusers pipeline/model dependent |
| Video generation/animate/transform/upscale | Yes | Diffusers plus image/video codec dependencies |
| Audio/music generation | Yes | Diffusers or Transformers pipeline/model dependent |
| Song continuation/variation | Yes | Prepared; no generic adapter yet |
| Text-to-speech | Yes | Transformers TTS pipeline/model dependent |
| Speech-to-text/translation | Yes | Transformers ASR pipeline/model dependent |
| Event/VAD/speaker/language/emotion detection and classification | Yes | Transformers audio-classification pipeline/model dependent |
| Captioning and prompted audio understanding | Yes | Transformers any-to-any pipeline/model dependent |
| Audio embeddings | Yes | Transformers processor/model registry dependent |
| Speaker diarization | Yes | Prepared; no generic adapter yet |
| Voice conversion | Yes | Prepared; no generic adapter yet |
| Stem generation/separation | Yes | Prepared; no generic adapter yet |
| Audio enhancement | Yes | Prepared; no generic adapter yet |
| Audio editing | Yes | Prepared; no generic adapter yet |

Parameters not accepted by a concrete pipeline are reported rather than
silently discarded.

Direct companion output formats are `png`/`jpeg`/`webp` for images,
`mp4`/`gif` for video, `wav`/`flac`/`ogg` for generated audio and TTS, and
`json`/`text`/`srt`/`vtt`/`tsv` for ASR and structured audio analysis. Audio
input tasks decode locally through `soundfile` or `ffmpeg`; compatible
audio-to-text processors can additionally require `librosa`. Codec libraries
required by the selected format must already be installed.

## HTTP API and jobs

This section explains how media jobs map into the inference service. The
complete route-by-route contract, common routing fields, response schemas,
authentication, limits and compatibility gaps are maintained in the
[HTTP API reference](api.md).

Direct endpoints:

```text
POST /v1/chat/completions
POST /v1/images/generations
POST /v1/images/edits
POST /proxy/openai/images/generations
POST /proxy/openai/images/edits
POST /v1/videos/generations
POST /v1/audio/generations
POST /v1/audio/speech
POST /v1/audio/transcriptions
POST /v1/audio/translations
GET  /v1/capabilities
GET  /v1/parameters
GET  /v1/outputs/{id}
```

Third-party image clients can use three deliberately bounded compatibility
surfaces:

- Open WebUI should use its `openai` image engine with
  `IMAGES_OPENAI_API_BASE_URL=http://HOST:11434/v1`, a Werk API key, and an
  OpenAI image alias such as `gpt-image-1`; the alias resolves to
  `werk serve --image-model MODEL`.
- Basic AUTOMATIC1111 clients can use `/sdapi/v1/txt2img`, `/sdapi/v1/sd-models`,
  `/sdapi/v1/options`, and `/sdapi/v1/progress`. The adapter translates core
  prompt, negative-prompt, size, image-count, step, guidance, seed, and model
  fields. It rejects meaningful unsupported advanced behavior or reports an
  explicit compatibility warning instead of silently claiming A1111 scripts,
  high-resolution passes, face restoration, requested samplers, or img2img.
- ComfyUI's hosted OpenAI image node can use
  `--comfy-api-base http://HOST:11434`. Werk accepts its `X-API-Key` header and
  generation proxy path. Multipart editing is reported as unsupported. Werk
  does not emulate ComfyUI's native workflow-graph `/prompt`, custom-node,
  history, view, or WebSocket protocol.

Embedded image responses are returned as Base64 and their managed temporary
results are removed after encoding. This is the portable choice for server-side
third-party clients; explicit Werk URL responses remain authenticated and
retained.

Long-running jobs:

```text
POST   /v1/jobs
GET    /v1/jobs/{id}
DELETE /v1/jobs/{id}
```

JSON bodies on `/v1/chat/completions`, `/v1/jobs`,
`/v1/audio/transcriptions`, and `/v1/audio/translations` default to 128 MiB.
This bounds inline Base64 images and audio; other routes retain Axum's smaller
default. `WERK_API_BODY_LIMIT_BYTES` can set a positive upload limit up to
512 MiB for trusted deployments. Local-path inputs avoid Base64 expansion when
the Werk server and selected runtime can read the same filesystem.

Persisted states are `queued`, `loading`, `running`, `encoding`, `completed`,
`failed`, and `cancelled`. `/v1/audio/speech` returns audio bytes directly;
image endpoints default to self-contained Base64 JSON, transcription embeds its
text, and an explicit URL response returns Werk metadata plus authenticated
`/v1/outputs/{id}` URLs.

The shared conversation content model can represent text, image, video, audio,
tool calls, and tool results. Available media tools are derived dynamically
from installed models and successful runtime probes. Automatic LLM tool-call
orchestration is intentionally left to Station or another client; Serve never
starts a CLI subprocess.

## Storage and limits

```text
WERK_HOME/
├── models/
├── artifacts/
├── outputs/
└── jobs/
```

HTTP and job output metadata records include ID, task, model, runtime, path,
MIME type, size, dimensions/duration where applicable, seed, effective
parameters, and creation time. Direct CLI output files and retained result
directories are both subject to output retention. Retention only targets direct
children of `outputs/`, never models. Defaults are 30 days and 20 GiB; use
`WERK_OUTPUT_RETENTION_SECONDS` and `WERK_OUTPUT_MAX_BYTES` to override them.

## Current limitations

- `probe-model` checks local repository metadata, task hints, and dependencies.
  The concrete Diffusers/Transformers pipeline is first loaded during
  execution and may then remain resident, so model-specific incompatibility
  can still surface on the first cold request.
- A repository that declares a media task can still be non-executable when its
  native layout has no registered local adapter. Discovery reports declared and
  currently probe-eligible tasks separately; neither state promises that an
  arbitrary architecture will load.
- JSON local-path and inline-base64 inputs are supported. The offline companion
  does not fetch remote HTTP(S) URLs. OpenAI multipart upload compatibility is
  not implemented yet, including the hosted ComfyUI image-edit proxy.
- OpenAI-compatible image generation defaults to embedded Base64 for portable
  third-party clients. Explicit URL responses are authenticated relative Werk
  output URLs, not public object-storage URLs.
- Persisted job cancellation is cooperative. A native third-party call that
  lacks cancellation may release resources only when it returns.
- The companion currently returns one terminal response rather than granular
  progress events. The CLI animation therefore indicates liveness only, while
  measured phase timings are printed after completion by `--verbose`.
  Persisted phases are available, but `encoding` can be too brief to observe
  for fast jobs.
- Werk serves authenticated whole-file outputs. HTTP byte ranges and
  object-storage export are future work.
- Generic adapters for song continuation/variation, speaker diarization, voice
  conversion, stems, enhancement, and audio editing are described by the
  contract but not executable yet.
- The generic Transformers TTS path uses the model's native voice and sample
  rate; explicit voice, speed, pitch, and output resampling remain
  model-specific and are reported as unsupported by this adapter.
