# Werk1112 media inference

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
werk audio generate|speak|transcribe|separate MODEL
werk serve
```

The old `werk run` parser remains hidden for compatibility. New applications
should use `chat` for text and the typed media commands for generated files.

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
werk audio generate musicgen --prompt "quiet analogue ambience" \
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
- audio and music generation, song continuation/variation, TTS, ASR, voice
  conversion, stem generation/separation, and audio enhancement.

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
family/architecture probe, runtime availability, accelerator, explicitly set
parameters, and workload fit. It distinguishes:

- backend fallback: the same model through another runtime;
- execution degradation: offload, tiling/windowing, or a slower attention path;
- model/quality downgrade: a recommendation that is never silently executed.

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
caches one fully configured Diffusers image/video pipeline by default. The
first compatible request is cold; subsequent requests with the same model and
runtime configuration are warm. Prompt, seed, size, steps, and count are
request-local and do not invalidate the cache. Model, device, dtype,
offload/tiling configuration, and LoRA changes may select a different entry and
therefore cause a cold load. At the default cache size, the existing pipeline
is evicted before a replacement is loaded.

`WERK_MEDIA_PIPELINE_CACHE_SIZE` controls the maximum number of resident
Diffusers entries and defaults to `1`; `0` disables the pipeline cache while
the worker can remain persistent. Cached pipelines retain VRAM and/or host RAM
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
explicitly select `cuda`, `rocm`, `mps`/`metal`, or `cpu`. MLX media models
remain catalogable but have no executable adapter in this release.

`werk doctor` reports the protocol and optional dependencies. Missing
Diffusers, Transformers, Pillow, audio/video codecs, or accelerator packages
only disables affected tasks. Accelerator availability comes from the exact
Python/PyTorch process selected for the media companion; the report includes
CUDA/ROCm/MPS availability and device details. Host probing, including WSL's
`/dev/dxg`, is used only when an older external companion does not report
accelerators. Candle is not currently a Diffusers image fallback; unsupported
media GPU routes fall back to the compatible CPU companion.

### Execution support

| Task group | Catalog / inspect / estimate | Companion execution |
| --- | --- | --- |
| Image generation/edit/inpaint/upscale | Yes | Diffusers pipeline/model dependent |
| Video generation/animate/transform/upscale | Yes | Diffusers plus image/video codec dependencies |
| Audio/music generation | Yes | Diffusers or Transformers pipeline/model dependent |
| Song continuation/variation | Yes | Prepared; no generic adapter yet |
| Text-to-speech | Yes | Transformers TTS pipeline/model dependent |
| Speech-to-text/translation | Yes | Transformers ASR pipeline/model dependent |
| Voice conversion | Yes | Prepared; no generic adapter yet |
| Stem generation/separation | Yes | Prepared; no generic adapter yet |
| Audio enhancement | Yes | Prepared; no generic adapter yet |

Parameters not accepted by a concrete pipeline are reported rather than
silently discarded.

Direct companion output formats are `png`/`jpeg`/`webp` for images,
`mp4`/`gif` for video, `wav`/`flac`/`ogg` for generated audio and TTS, and
`json`/`text`/`srt`/`vtt`/`tsv` for ASR. Codec libraries required by the
selected format must already be installed.

## HTTP API and jobs

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
- Generic adapters for voice conversion, stems, and enhancement are described
  by the contract but not executable yet.
- The generic Transformers TTS path uses the model's native voice and sample
  rate; explicit voice, speed, pitch, and output resampling remain
  model-specific and are reported as unsupported by this adapter.
