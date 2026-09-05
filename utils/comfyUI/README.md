# ComfyUI Werk1112 nodes (Beta)

> [!WARNING]
> This integration is currently in beta. Node inputs, outputs, and discovery
> behavior may still change before the first stable release.

Starting with v1.5.0, the Werk ComfyUI package uses the same release version as
Werk Core and Werk Media Companion. This synchronized versioning does not
change the integration's beta status.

This package connects native ComfyUI nodes to `werk serve` over HTTP. ComfyUI
still owns its workflow graph, queue, history, previews, and saved media; Werk
performs model discovery, routing, and inference.

## Existing ComfyUI compatibility

Werk already supports ComfyUI's built-in hosted OpenAI image node at
`POST /proxy/openai/images/generations`. These custom nodes are an additional,
Werk-native integration with model and available-task discovery, live parameter
schema discovery, Werk routing options, complete inference metadata, and
authenticated Werk URL outputs.

Werk does not implement ComfyUI's `/prompt`, `/history`, `/view`, WebSocket,
queue, custom-node, or workflow-graph protocols.

## Installation

### ComfyUI Manager (recommended)

The release package uses the immutable Comfy Registry ID `werk1112`. Once its
first Registry version has been published, search for **WERK1112 (Beta)** in ComfyUI
Manager and select **Install**. The equivalent Comfy CLI command is:

```bash
comfy node install werk1112
```

Manager installs the package dependencies in the correct ComfyUI Python
environment and handles future updates. Restart ComfyUI after installation.

Until the first Registry release, or for repository development, use the
manual method below.

### Manual development installation

Copy the package from a Werk1112 checkout into ComfyUI and install only its
small media-transfer dependencies in the Python environment used by ComfyUI:

```bash
cp -R utils/comfyUI /path/to/ComfyUI/custom_nodes/comfyui-werk1112
cd /path/to/ComfyUI
python -m pip install -r custom_nodes/comfyui-werk1112/requirements.txt
```

For repository development on Linux or macOS, use a symlink instead:

```bash
ln -s "$(pwd)/utils/comfyUI" \
  /path/to/ComfyUI/custom_nodes/comfyui-werk1112
```

PowerShell junction:

```powershell
New-Item -ItemType Junction `
  -Path "C:\path\to\ComfyUI\custom_nodes\comfyui-werk1112" `
  -Target (Resolve-Path ".\utils\comfyUI")
```

Restart ComfyUI. The nodes appear in the `WERK` categories. The package does
not pin, replace, or install Torch.

Do not use `pip install .` as the node installation method. ComfyUI must load
the complete node directory, including its frontend extension, from
`custom_nodes`; Registry/Manager installation or the directory copy/symlink
above provides that layout.

## Starting Werk

For a fresh API-key file:

```bash
werk auth api-key generate --name comfyUI
werk serve --image-model tiny-sd
```

`generate --name` does not currently append to an existing key file. If a key
file already exists, generate the ComfyUI key separately and copy its `[[keys]]`
block into `~/.config/werk1112/api-keys.toml`:

```bash
werk auth api-key generate --name comfyUI --path /tmp/comfyui-key.toml
```

For local unauthenticated development only:

```bash
werk serve --image-model tiny-sd --allow-unauthenticated
```

To provide default runtime-persistence policy for an unconnected **WERK
Prefill** policy socket, start Werk with:

```bash
werk serve --persistence
```

The policy and experimental defaults supplied by this setting apply only to
`/werk/v1/prefill`. Separately, when Werk starts local vLLM, the same setting
supplies vLLM's native APC default unless `WERK_VLLM_ARGS` explicitly enables
or disables it. Remote vLLM receives no generated launch argument. Neither
effect enables or disables the independent model/pipeline residency used by
normal image, video, audio or OpenAI-compatible `/v1` requests. Current named
state support is the experimental, functionally validated managed
llama-server path for an installed GGUF model, not a model-independent generic
KV format.

`--image-model` supplies the alias used by image compatibility endpoints; the
Werk-native image, vision, video, and audio model nodes discover installed
models and select one explicitly in the graph. They do not need server-wide
video/audio aliases.

The normal address is `http://127.0.0.1:11434`. Put the address and key into a
**WERK Connection** node, or set `WERK_BASE_URL` and `WERK_API_KEY` before
starting ComfyUI. Explicit widget values take precedence over environment
defaults.

### Preparing Wan2.2 for Werk video nodes

The Werk nodes call Werk's local media companion. They do not load checkpoints
from ComfyUI's `models` directories. Pull the official Diffusers repository
into Werk before starting the server:

```bash
werk pull Wan-AI/Wan2.2-TI2V-5B-Diffusers --name wan22-ti2v-5b
werk doctor --model wan22-ti2v-5b --task video-generation
werk doctor --model wan22-ti2v-5b --task image-to-video
werk serve
```

Use
[`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers),
not the native
[`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
layout. The latter has no Diffusers `model_index.json`, so Werk can catalog it
but the bundled local adapter rejects it as non-executable. ComfyUI's
[official native Wan2.2 workflow](https://docs.comfy.org/tutorials/video/wan/wan2_2)
is a separate path with model components installed directly under ComfyUI; it
is not required by these HTTP-backed Werk nodes.

## Nodes

- **WERK Connection** creates one reusable, credential-redacting connection.
  Click **Verify Connection** to perform a real request and show a visible
  success or error status directly on the node.
- **WERK Server Info** reports installed models and server capabilities.
- **WERK Runtime Info** performs strict Werk Protocol 1.0 discovery. It reports
  the active backend, negotiated limits, and every capability using the exact
  `supported`, `unsupported`, `unavailable`, `experimental`,
  `externally_managed`, or `metadata_only` status returned by Werk. Unlike the
  legacy discovery nodes, runtime-control nodes require `/werk/v1` and do not
  fall back when an older server lacks it.
- **WERK Persistence Policy** creates a typed `WERK_PERSISTENCE_POLICY` with
  `auto`, `ephemeral`, `memory`, or `disk` retention; `disabled`, `prefer`, or
  `required` reuse; an optional TTL; and pinning. A TTL widget value of zero
  omits `ttl_seconds` from the connected policy and therefore means no TTL; it
  does not mean immediate expiry. Because the top-level policy is present, a
  server-side granular TTL default does not replace it.
- **WERK Runtime States** lists the caller's visible prefix/runtime states with
  optional model, tier, page-size, and cursor filters. Every listed object is
  the complete public state summary for inspecting status, tier, size, age,
  expiry, pinning, backend, and reusability. The node enforces the page bound
  advertised by the connected server.
- **WERK State Control** pins, unpins, promotes, demotes, or evicts one explicit
  state. Dry-run is enabled by default. Promotion accepts only `ram`/`vram`,
  demotion only `ram`/`disk`, and other actions reject a target tier.
- **WERK State Prune** removes states through an explicit ID list, a constrained
  filter, or a separately confirmed `all` selector. It defaults to dry-run.
- **WERK Memory Status** reports host and accelerator capacity, availability,
  managed/reserved bytes, topology, and pressure without guessing unavailable
  telemetry.
- **WERK Runtime Experts** lists bounded pages of backend-reported MoE expert
  residency, size, hotness, pin state, and last-use telemetry. Model and tier
  filters are optional; cursors remain opaque.
- **WERK Expert Control** applies `prefetch`, `pin`, `unpin`, or `evict` to an
  explicit model and explicit expert IDs. It defaults to dry-run. Prefetch
  requires an explicit `vram` or `ram` target; other actions reject a target.
- **WERK Prefill** accepts text or role/content message JSON plus an optional
  persistence policy. It returns a `WERK_STATE_HANDOFF`, never a string or JSON
  token.
- **WERK Decode** is the only consumer of `WERK_STATE_HANDOFF`. It returns text,
  safe completion metadata, and an optional updated opaque handoff.
- **WERK Image Models** distinguishes declared image support from currently
  runtime probe-eligible image support. Probe eligibility is not a guarantee
  that a complete model pipeline will load successfully. Connect a
  **WERK Connection**, click **Refresh Models**, and select a model from the
  **available_model** dropdown.
- **WERK Image Parameters** reads the active model/runtime parameter schema.
- **WERK Vision Models** discovers only models that authoritatively declare
  `image-understanding`, and distinguishes declaration from current runtime
  probe eligibility.
- **WERK Vision Config** exposes the supported non-streaming chat controls:
  temperature, top-p, completion budget, seed, image detail, and stop strings.
  Per-request routing is deliberately absent because the current chat endpoint
  is routed by the `werk serve` configuration and ignores media routing fields.
- **WERK Vision Analyze** accepts a native ComfyUI `IMAGE` batch, preserves its
  order as PNG data URLs in one multimodal user message, and returns the
  assistant inspection plus sanitized completion metadata.
- **WERK Video Models** discovers video models per explicit task. Choose
  `video-generation` for T2V or `image-to-video` for I2V, refresh discovery,
  and select `preferred_model` when more than one eligible model exists.
- **WERK Video Parameters** reads the selected model/runtime schema for that
  same explicit video task.
- **WERK Routing Config** represents all current Werk routing overrides without
  turning inherited server, model, or profile defaults into explicit request
  values.
- **WERK Image Config** contains the common image controls and accepts the
  remaining model-specific image parameters as schema-driven JSON. It can
  consume a **WERK Routing Config**.
- **WERK Image Generate** is the compact generator. Its `model` socket
  is a real, required ComfyUI input and should be connected to the `model`
  output of **WERK Image Models**. It returns a ComfyUI `IMAGE` batch plus
  sanitized Werk metadata and IDs.
- **WERK Video Config** contains common dimensions, frame rate/count, sampling,
  seed, output-container, and temporal-VAE controls. Remaining schema-discovered
  video parameters are accepted as JSON, and routing stays a separate typed
  input.
- **WERK Video Generate** submits a persisted Werk video job, polls it, fetches
  each authenticated output, and returns native ComfyUI `VIDEO` values plus
  sanitized metadata, seed, job/result IDs, and output IDs. Its `model` socket
  must come from **WERK Video Models**. Connecting one `initial_image` changes
  the generated request to I2V.
- **WERK Audio Models** discovers models per exact audio task, including
  generation, transcription, detection, analysis, transformation, and
  embedding. Only models declared/runtime-available for the selected task are
  offered.
- **WERK Audio Parameters** reads the live task/model/runtime schema. Use it
  before adding model-specific fields to the JSON inputs. It is a read-only
  schema inspector: its JSON output is not an execution config and must not be
  connected to a generator's `model` input.
- **WERK Audio Config** covers `audio-generation`, `music-generation`, and
  `text-to-speech`. Zero sample rate/channels inherit model defaults; TTS seed
  `0` is also omitted so strict adapters without deterministic synthesis are
  not rejected. TTS language and speaking style are first-class optional
  inputs; Qwen3-TTS VoiceDesign receives `speaking_style` as `instruct`.
- **WERK Audio Generate** uses the specialized generation/speech endpoints and
  requires an active **WERK Audio Config**, then returns one or more native
  ComfyUI `AUDIO` dictionaries (`waveform` shaped `[B,C,T]` plus
  `sample_rate`). A bypassed or disconnected config fails validation instead
  of silently generating with defaults.
- **WERK Audio Process** accepts native source audio for `voice-conversion`,
  `stem-separation`, `audio-enhancement`, and `audio-editing`, then returns
  native `AUDIO`. Voice conversion may also receive `reference_audio`.
- **WERK Audio Analyze** accepts native audio for transcription/translation,
  detection, captioning/diarization/classification/understanding, and audio
  embedding. Its first output is a list of UTF-8 text or normalized JSON
  strings, matching the artifact MIME type.

The interactive verification and dropdown discovery run through a local
ComfyUI route. The browser sends the connection settings only to its own
ComfyUI backend; the backend contacts Werk. API keys are never included in the
route response.

### Runtime persistence, experts, and split prefill/decode

Use the runtime nodes in this shape:

```text
WERK Connection.connection ----------+--> WERK Runtime Info.connection
                                     +--> WERK Prefill.connection
                                     +--> WERK Decode.connection

WERK Persistence Policy.policy ----------> WERK Prefill.policy
WERK Prefill.handoff ---------------------> WERK Decode.handoff

WERK Connection.connection ----------+--> WERK Runtime Experts.connection
                                     +--> WERK Expert Control.connection
WERK Runtime Experts.expert_ids ----------> WERK Expert Control.expert_ids
```

The **WERK Persistence Policy** connection is optional. When it is unconnected,
**WERK Prefill** omits the request's top-level `policy`, allowing defaults from
`werk serve --persistence` and its granular persistence flags to apply. Without
those server flags, the normal protocol default is `auto`/`prefer`, no TTL and
not pinned. Connecting a policy makes that complete policy authoritative. In a
connected policy, the TTL widget's `0` value omits `ttl_seconds` and means no
TTL; it does not defer to `--persistence-ttl-seconds` and does not expire the
state immediately.

The **WERK Prefill** `allow_experimental` widget is also an explicit request
decision: both `true` and `false` override the server default. Enable it
deliberately for the current experimental managed llama-server path. Server
defaults never bypass the capability checks described below.

Prefill and decode are capability-gated. Experimental capabilities remain
disabled unless the node's explicit opt-in is enabled. The only preflight
exception is an `unavailable` Prefill/Handoff capability when that opt-in is
enabled: Prefill may send the request so the server can run its model-scoped
functional probe, and the server remains the authoritative second gate. Every
other non-operational status and operation fails closed with the server's
reason. The handoff is an in-memory, opaque workflow value and is deliberately
excluded from JSON/STRING metadata outputs, representations, and error text. It
may be one-time and short-lived, so a stale or reused value can correctly return
`expired_handoff`.

State pruning affects only runtime states selected by the request. It is not a
model, artifact, output, job, authentication, backend, or external-path cleanup
operation. Expert nodes require `runtime.experts.residency`: `supported` is
operational, `experimental` requires the node's explicit opt-in, and
`externally_managed` permits read-only expert telemetry but not control.
`unsupported`, `unavailable`, and `metadata_only` fail closed. Expert Control
never derives a selection from hotness or from the current page; the model and
all IDs must be supplied explicitly, and the server enforces its advertised ID
bound. No current production adapter advertises operational expert residency,
so these nodes currently report the truthful capability failure without
claiming that a backend action succeeded.

### Recommended image workflow

Use the nodes in this shape:

```text
WERK Connection.connection --------+--> WERK Image Models.connection
                                    +--> WERK Image Generate.connection

WERK Image Models.model ----------------> WERK Image Generate.model
WERK Routing Config.routing ------------> WERK Image Config.routing
WERK Image Config.config ---------------> WERK Image Generate.config
WERK Image Generate.images -------------> Preview Image.images
```

Connect the singular `model` output, not `available_models`. The latter is a
newline-separated diagnostic list. The generator deliberately has no model
widget or private model dropdown: its `model` input uses ComfyUI's
`forceInput`, so the selected model remains explicit in the graph.

`config` is optional. An unconnected config uses the Image Config defaults:
1024x1024, one image, batch size 1, 28 steps, guidance 7, seed 0, PNG, and an
embedded Base64 response. The default batch size is treated as inherited and
is not sent explicitly. For a reproducible and visible workflow, connecting an
explicit **WERK Image Config** is recommended.

### Recommended vision inspection workflow

Use Vision Analyze after rendering a page, slide, frame, or other generated
asset that needs visual QA:

```text
WERK Connection.connection --------+--> WERK Vision Models.connection
                                    +--> WERK Vision Analyze.connection

WERK Vision Models.model --------------> WERK Vision Analyze.model
Load Image.IMAGE -----------------------> WERK Vision Analyze.images
WERK Vision Config.config --------------> WERK Vision Analyze.config
```

One `IMAGE` tensor may contain a batch; every batch item becomes a separate
`image_url` content part in the original order. The prompt and all image parts
are sent together in one user message to `POST /v1/chat/completions`. An
optional system prompt can establish a reusable QA rubric. Vision Analyze
requires an active Vision Config so a bypassed configuration cannot silently
disappear from the queued request.

Model execution still depends on an installed model/backend pair that Werk
reports as runtime-available for `image-understanding`. Merely being able to
run the text side of Qwen3-VL, GLM-4V, Gemma, or another multimodal repository
does not make its vision path available. **WERK Vision Models** fails honestly
when the server declares a vision model but no compatible runtime passes the
probe.

### Recommended video workflows

Text-to-video uses this explicit chain:

```text
WERK Connection.connection --------+--> WERK Video Models.connection
                                    +--> WERK Video Generate.connection

WERK Video Models.model ----------------> WERK Video Generate.model
WERK Routing Config.routing ------------> WERK Video Config.routing
WERK Video Config.config ---------------> WERK Video Generate.config
WERK Video Generate.videos -------------> Save Video.video
```

Set **WERK Video Models.task** to `video-generation`. For image-to-video, set it
to `image-to-video` and add exactly one image:

```text
Load Image.IMAGE ------------------------> WERK Video Generate.initial_image
```

The selector task and presence of `initial_image` must agree. Task discovery
only filters eligible models; it does not add a hidden image or silently change
the generated request. **WERK Video Parameters** is an optional diagnostic
branch: connect the same connection and singular model output, then select the
same task to inspect its live schema.

As with image generation, connect `model`, not `available_models`. With
`require_available=true`, discovery filters to tasks accepted by the current
runtime probe. Multiple candidates require `preferred_model`; the node does
not choose one arbitrarily. Probe eligibility is still not a promise that the
first cold pipeline load will fit memory or support every explicit parameter.

For `wan22-ti2v-5b`, configure 1280x704, 121 frames, 24 FPS, 50 steps, and
guidance 5, and set `precision=bf16` in **WERK Routing Config**. The companion
keeps Wan's VAE in fp32. The official flow shift 5 is already stored in the
Diffusers repository's
[scheduler configuration](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/blob/main/scheduler/scheduler_config.json),
so leave `additional_video_parameters_json` empty. These values come from the
[official Wan2.2 configuration](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)
and [model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B). The generic
node defaults—832x480, 81 frames, 24 FPS, 30 steps, and guidance 6—are useful
for filling a portable request but are not a Wan2.2 quality preset.

Wan calls its 1280x704/704x1280 output 720P. Its native reference requires at
least 24 GB VRAM with offload and reports under nine minutes for a five-second
720P clip on a consumer GPU; the repository download is about 34.2 GB. The
official ComfyUI native workflow separately reports that its own native
offloading can fit the 5B model in 8 GB VRAM. Neither upstream figure guarantees
the memory or speed of Werk's Diffusers route. In **WERK Routing Config**, set
`allow_cpu_offload=enabled` only when the chosen accelerator and available host
RAM can support it, and use metadata to confirm the active pipeline and offload
state.

For a smaller T2V-only plumbing check, install
[`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
and use 832x480. The [official Wan2.1 model card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
reports 8.19 GB VRAM and about four minutes for five seconds of 480P video on an
RTX 4090 without quantization. This checks transport, job polling, download,
native `VIDEO` conversion, and saving, but not Wan2.2 I2V or 720P behavior.

### Recommended audio workflows

Generation and TTS use the typed config path:

```text
WERK Connection.connection --------+--> WERK Audio Models.connection
                                    +--> WERK Audio Generate.connection

WERK Audio Models.model ----------------> WERK Audio Generate.model
WERK Routing Config.routing ------------> WERK Audio Config.routing
WERK Audio Config.config ---------------> WERK Audio Generate.config
WERK Audio Generate.audio --------------> Preview Audio.audio
```

Select the same task on Models, Config, and Generate. Valid generation tasks
are `audio-generation`, `music-generation`, and `text-to-speech`. A TTS prompt
is the spoken text; TTS rejects a non-empty negative prompt. Config fields left
at their inherit values are omitted instead of forcing adapter-dependent
options. Do not bypass Audio Config: ComfyUI does not execute bypassed nodes,
so their widgets and additional JSON do not exist in the queued prompt. Audio
Generate requires the config link and rejects that state rather than silently
falling back to another request.

For a Qwen3-TTS VoiceDesign run equivalent to an explicit CUDA CLI request,
set Routing Config to `backend=auto`, `accelerator=cuda`, `precision=bf16`, and
`fallback_policy=none`. Set Audio Config to `task=text-to-speech`, the desired
non-zero seed, `output_format=wav`, `language=German`, and the complete voice
instruction in `speaking_style`. To reproduce a CLI seed, set ComfyUI's seed
control to `fixed`; `randomize` deliberately changes it after every run.

Audio-input workflows connect ComfyUI's **Load Audio** to either Process or
Analyze:

```text
Load Audio.AUDIO ------------------------> WERK Audio Process.source_audio
Load Audio.AUDIO ------------------------> WERK Audio Analyze.source_audio
```

For `voice-conversion`, a second **Load Audio** may connect to
`reference_audio`; it is transported with the distinct `reference_audio` role.
Other transform tasks reject that input. `audio-editing` and
`audio-understanding` require a non-empty prompt. Remaining task parameters
belong in `additional_audio_parameters_json`, using the live schema from
**WERK Audio Parameters**. Speech-to-text/translation keys normalize to the
`stt.*` namespace; other input-audio task keys normalize to `audio.*`.

All long audio operations are persisted jobs. The nodes poll the same terminal
states as video and issue a best-effort `DELETE /v1/jobs/{id}` when ComfyUI is
interrupted or the connection timeout expires. A persisted job is only a
request/status/result record: it does not persist a loaded model, media
pipeline, text KV cache or resumable computation. Nonterminal jobs are marked
failed after a Werk restart. Audio outputs are downloaded with authentication
and converted through PyAV. Source `AUDIO` is encoded as PCM16 WAV and embedded
in the generic job request.

Video generation is asynchronous at the Werk API boundary. The generator polls
states `queued`, `loading`, `running`, and `encoding`, and requests best-effort
job cancellation when ComfyUI interrupts or the connection timeout expires.
The **WERK Connection** default is 900 seconds; increase it for a slower local
run. The routing config's `inference_timeout_seconds` is a distinct Werk request
override and does not extend a shorter client connection timeout.

Ready API-prompt examples are provided for
[text-to-video](examples/werk_video_generation_api.json) and
[image-to-video](examples/werk_image_to_video_api.json), plus
[music generation](examples/werk_music_generation_api.json),
[text-to-speech](examples/werk_text_to_speech_api.json),
[audio understanding](examples/werk_audio_understanding_api.json), and
[voice conversion](examples/werk_voice_conversion_api.json), plus a
[vision inspection](examples/werk_vision_inspection_api.json) workflow for
missing controls, overflow, clipping, alignment, and spacing checks. They contain no
credential; read the [example assumptions](examples/README.md) before
submitting them to ComfyUI. Voice conversion demonstrates the prepared node
contract only; the bundled companion currently advertises no executable
generic adapter for it.

When these nodes call `werk serve`, generic Diffusers/Transformers media and
managed Qwen3-TTS use separate persistent, serialized execution workers. Each
worker owns a separate LRU with one resident model/pipeline by default. Health,
discovery, and estimation calls remain independent of the execution queues;
Werk caches bounded positive probe and estimate results in Rust so repeated
validated preflights avoid another one-shot Python import. Failures and
unavailable results are not cached.

The first generation or analysis is a cold load, and later runs with the same
model/runtime configuration should be substantially faster to start. Changing
prompt, seed, dimensions, steps, or count keeps the model warm. Changing model,
task adapter, device, dtype, offload/tiling settings, or LoRAs may reload it;
the previous entry in that worker is evicted before the new one is loaded at
the default cache size. Set
`WERK_MEDIA_PIPELINE_CACHE_SIZE=0` before starting Werk to disable pipeline
caching in both workers, or set a larger non-negative per-worker entry count
when system memory permits. If both worker types are used, each can retain up
to that count. Resident entries retain VRAM and/or RAM until eviction or Werk
shuts down.

Werk metadata exposes `model_cache_hit` and `model_load_seconds` to distinguish
warm and cold runs. A worker crash or inference timeout clears the resident
state, so the next run is cold. The resident transport never replays the same
`execute` frame; Werk's higher-level fallback policy may still try another
accepted runtime candidate. A legacy external media companion falls back to
one-shot execution when it does not support the persistent protocol.

These normal media nodes do not call `/werk/v1/prefill`; their resident model
caches work whether or not `werk serve --persistence` is present. **WERK Vision
Analyze** likewise calls `/v1/chat/completions`. A selected llama-server or
local vLLM process can keep model weights and its own backend cache resident;
remote vLLM owns that lifetime outside Werk. The embedded ONNX GenAI CPU
fallback has a bounded model/tokenizer LRU but a fresh generator per request.
An opaque external ONNX runner and MLX/MLX-VLM remain per-request subprocesses.
None of those normal nodes creates a named Werk runtime state. Use the explicit
**WERK Prefill** and **WERK Decode** nodes for that separate, capability-gated
path.

The Transformers compatibility and ONNX GenAI worker LRUs are controlled by
`WERK_TRANSFORMERS_MODEL_CACHE_SIZE` and
`WERK_ONNX_GENAI_MODEL_CACHE_SIZE`, respectively; each defaults to one exact
model entry, and `0` disables that cache without making the external fallback
paths resident.

### WERK Routing Config

The routing node covers all 17 fields in Werk's current routing schema:

| Node input | Meaning and inheritance |
| --- | --- |
| `backend` | Runtime backend; blank inherits |
| `accelerator` | Requested accelerator; blank inherits |
| `device` | Concrete device identifier; blank inherits |
| `precision` | Compute precision; blank inherits |
| `quantization` | Weight quantization; blank inherits |
| `profile` | Saved Werk parameter profile; blank inherits |
| `quality` | `inherit`, `draft`, `balanced`, `high`, or `maximum` |
| `performance_preference` | `inherit`, `quality`, `balanced`, `speed`, `latency`, `throughput`, or `memory` |
| `fallback_policy` | `inherit`, `none`, `backend`, or `degrade` |
| `parameter_policy` | `inherit`, `strict`, `warn`, or `permissive` |
| `allow_cpu_offload` | Tri-state: `inherit`, `enabled`, or `disabled` |
| `allow_sequential_offload` | Tri-state: `inherit`, `enabled`, or `disabled` |
| `allow_component_offload` | Tri-state: `inherit`, `enabled`, or `disabled` |
| `allow_disk_offload` | Tri-state: `inherit`, `enabled`, or `disabled` |
| `attention_backend` | Attention implementation; blank inherits |
| `compile` | Tri-state: `inherit`, `enabled`, or `disabled` |
| `inference_timeout_seconds` | `0` inherits; a positive value sends Werk's `timeout_seconds` override |

`inherit` and blank values are omitted from the request. In particular,
`inherit` is different from `disabled`: the latter sends an explicit JSON
`false`. The `config_json` output shows the exact normalized options and is
useful when diagnosing a workflow.

The connection timeout and a positive `inference_timeout_seconds` value have
no static ComfyUI upper bound. Their lower-bound semantics remain unchanged:
the HTTP connection timeout must be positive, while inference timeout `0`
inherits. Werk and the selected runtime remain authoritative for execution.

`additional_routing_parameters_json` is a forward-compatible escape hatch for
canonical `routing.*` parameters introduced by newer Werk versions. The 17
fields above must use their dedicated controls. Cross-namespace keys,
transport fields, credentials, and duplicate dedicated controls are rejected.

### WERK Image Config

The image config exposes the common controls directly:

| Node input | Werk request |
| --- | --- |
| `width`, `height` | `size: "WIDTHxHEIGHT"` |
| `count` | `n`; the normal, adapter-independent image count |
| `batch_size` | `parameters["image.batch_size"]` only when greater than 1 |
| `steps` | `parameters["image.steps"]` |
| `guidance` | `parameters["image.guidance"]` |
| `seed` | `parameters["image.seed"]` |
| `output_format` | `output_format` |
| `response_format` | `response_format` |
| `style` except `none` | OpenAI-compatible prompt style hint |
| `vae_tiling` | Tri-state `parameters["image.vae_tiling"]` |
| `vae_slicing` | Tri-state `parameters["image.vae_slicing"]` |

These inference controls have no universal upper limit in the ComfyUI node.
The node still enforces types, minima, finite floating-point values, the
count/batch rule below, and the signed-64-bit seed representation. The live
Werk task/model/runtime schema and selected backend are authoritative; a value
accepted by the portable node may still be rejected or exceed available
resources during planning or execution.

Use `count` for normal multi-image generation. `batch_size=1` is not included
in the request, so Werk and the selected adapter retain their resolved default.
A `batch_size` greater than 1 is an alternative, adapter-dependent way to
control how many images are produced and therefore requires `count=1`. If both
`count` and `batch_size` are greater than 1, the node rejects the config locally
instead of sending an ambiguous request. This also avoids the Diffusers
conflict caused by combining OpenAI-compatible `n` with an explicit image batch
size.

Werk's current image-generation schema contains 80 image parameter
descriptors. The common subset above has native widgets; every remaining
geometry, sampling, inpainting, conditioning, adapter, high-resolution, and
post-processing parameter is available through
`additional_image_parameters_json`. Discover the exact model defaults,
constraints, allowed values, and runtime support with **WERK Image Parameters**
or:

```bash
werk parameters MODEL --task image-generation --json
```

Additional image parameters may be unqualified, canonical, or grouped under
`image`. These examples are equivalent where they name the same fields:

```json
{
  "sampler": "euler",
  "image.scheduler": "karras",
  "image": {
    "guidance_rescale": 0.2,
    "high_resolution_fix": true
  }
}
```

List parameters accept either a normal JSON array or Werk's list-override
shape. For example, this appends a LoRA to inherited model/profile defaults:

```json
{
  "image.loras": {
    "operation": "add",
    "values": [
      {"model": "style.safetensors", "weight": 0.65}
    ]
  }
}
```

Supported list operations are `inherit`, `replace`, `add`, and `clear`.
Omitting a parameter is the normal way to inherit its resolved Werk value.
Routing/transport keys and duplicates of dedicated controls are rejected
instead of being silently overwritten. The `config_json` output displays the
fully normalized config before generation.

### FLUX.2 Klein on limited VRAM

FLUX.2 Klein may need offloading even at a modest output resolution because
model weights dominate the memory estimate. Start with this configuration:

- In **WERK Routing Config**, set `allow_cpu_offload` to `enabled`.
- Leave `allow_sequential_offload` and `allow_component_offload` at `inherit`.
- In **WERK Image Config**, start at 512x512, set guidance around 3.5, and set
  `vae_tiling` to `enabled`.

If model CPU offload still does not fit, set
`allow_sequential_offload` to `enabled`. Sequential offload takes precedence
over the other CPU/component modes and uses less accelerator memory, but is
substantially slower. Offloading requires a CUDA-style accelerator and enough
host RAM. A model being listed as runtime-available means a suitable adapter
was discovered; it does not guarantee that the full pipeline fits into VRAM
without routing overrides.

The node accepts embedded Base64 or authenticated URL responses. RGBA images
are composited over white, all images become float32 RGB `[B,H,W,C]` tensors,
and differently sized outputs fail rather than being resized.

### WERK Video Config

The video config exposes the portable request controls directly:

| Node input | Werk request |
| --- | --- |
| `width`, `height` | `size: "WIDTHxHEIGHT"` |
| `count` | `n` when `batch_size=1` |
| `batch_size` | `parameters["video.batch_size"]` only when greater than 1 |
| `frames` | `parameters["video.frames"]` |
| `fps` | `parameters["video.fps"]` |
| `steps` | `parameters["video.steps"]` |
| `guidance` | `parameters["video.guidance"]` |
| `seed` | `parameters["video.seed"]` |
| `output_format` | API `response_format`, either MP4 or GIF |
| `temporal_vae_tiling` | Tri-state `parameters["video.temporal_vae_tiling"]` |

`count` and `batch_size` are alternative video-count controls. The node
rejects a config when both are greater than 1 instead of relying on
adapter-specific precedence. The remaining inference controls have no static
ComfyUI upper bound; types, minima, finite floating-point values, and the
signed-64-bit seed representation are still enforced. The live task schema,
selected model/runtime, resource planner, and backend remain authoritative, so
inspect them before tuning:

```bash
werk parameters MODEL --task video-generation --json
werk parameters MODEL --task image-to-video --json
```

Use `additional_video_parameters_json` for schema-discovered controls without a
dedicated widget. Unqualified, canonical, and grouped spellings normalize into
the `video.*` namespace. Request, routing, and duplicate dedicated keys are
rejected. This escape hatch does not make an unsupported parameter executable:
the planner and concrete pipeline still validate it. In particular, do not
copy a native runner's sampling knobs into a Diffusers request merely because
their names look similar; prefer the selected repository's scheduler config
unless the live adapter schema explicitly accepts an override.

For I2V, the generator accepts exactly one ComfyUI `IMAGE`, converts it to an
inline RGB PNG, and sends it as `initial_image`. A batch with zero or multiple
images fails locally. Generated artifacts are fetched through authenticated
Werk output URLs, bounded by `WERK_MAX_VIDEO_BYTES`, and wrapped with
`comfy_api.latest.InputImpl.VideoFromFile`. Use a current ComfyUI release with
native `VIDEO` support and connect the result to its
[Save Video node](https://docs.comfy.org/built-in-nodes/SaveVideo).

### WERK Audio Config and task groups

Audio Config maps portable controls without overriding inherited runtime
choices unnecessarily:

| Node input | Werk request |
| --- | --- |
| `duration` | `audio.duration` for audio/music generation |
| `variations` | generation API `n` |
| non-zero `seed` | `audio.seed` or `tts.seed` |
| non-zero `sample_rate`, `channels` | task namespace parameter |
| `output_format` | API `response_format` (`wav`, `flac`, or `ogg`) |
| `instrumental` | tri-state `audio.instrumental` |
| non-empty `voice`, non-default `speed` | TTS request fields |
| non-empty `language` | `tts.language` |
| non-empty `speaking_style` | `tts.speaking_style`; Qwen3-TTS VoiceDesign `instruct` |

Audio duration, variation count, explicit sample rate/channel count, and TTS
speed have no static ComfyUI upper bound. Their minima, `0` inherit sentinels,
finite floating-point requirements, task-specific rules, enums, and the
signed-64-bit seed representation remain enforced. The live task/model/runtime
schema and concrete audio adapter decide which values are executable.

The native `AUDIO` socket contains waveform samples and sample rate, not the
downloaded artifact's container. Werk retains the requested WAV/FLAC/OGG file
under its managed output store and exposes its `output_id`; ComfyUI's built-in
**Preview Audio** may independently encode that waveform as a temporary FLAC.
That preview container does not change the Werk artifact or its samples.

The Models and Parameters selectors expose the Rust task taxonomy in this
order: generation (`audio-generation`, `music-generation`, `text-to-speech`),
transcription (`speech-to-text`, `speech-translation`), detection
(`audio-event-detection`, `voice-activity-detection`,
`speaker-identification`, `language-identification`,
`speech-emotion-recognition`), analysis (`audio-captioning`,
`speaker-diarization`, `audio-classification`, `audio-understanding`),
transformation (`voice-conversion`, `stem-separation`, `audio-enhancement`,
`audio-editing`), and `audio-embedding`. Discovery means the server declares
the task; a backend may still fail honestly at execution time if the selected
runtime cannot execute that particular model/task pair.

## Discovery and diagnostics

CLI equivalents:

```bash
werk inspect MODEL
werk parameters MODEL --task image-generation --json
werk parameters MODEL --task video-generation --json
werk parameters MODEL --task image-to-video --json
werk parameters MODEL --task music-generation --json
werk parameters MODEL --task speech-to-text --json
werk parameters MODEL --task audio-understanding --json
werk parameters MODEL --task voice-conversion --json
werk doctor --model MODEL --task image-generation
werk doctor --model MODEL --task video-generation
werk doctor --model MODEL --task image-to-video
```

The nodes call:

```text
GET /v1/models
GET /v1/capabilities
GET /v1/parameters?task=image-generation&model=MODEL&backend=auto
GET /v1/parameters?task=video-generation&model=MODEL&backend=auto
GET /v1/parameters?task=image-to-video&model=MODEL&backend=auto
GET /v1/parameters?task=AUDIO_TASK&model=MODEL&backend=auto
POST /v1/chat/completions
POST /v1/images/generations
POST /v1/videos/generations
POST /v1/audio/generations
POST /v1/audio/speech
POST /v1/jobs
GET /v1/jobs/{id}
DELETE /v1/jobs/{id}
GET /v1/outputs/{id}
GET /werk/v1/info
GET /werk/v1/capabilities
GET /werk/v1/states
POST /werk/v1/states/{id}/actions
POST /werk/v1/states/prune
GET /werk/v1/memory
GET /werk/v1/experts
POST /werk/v1/experts/actions
POST /werk/v1/prefill
POST /werk/v1/decode
```

## WSL and containers

`127.0.0.1` always refers to the environment in which ComfyUI runs:

- Windows ComfyUI calling Werk in WSL should use the WSL address reachable
  from Windows.
- WSL ComfyUI calling Windows Werk should use the Windows host address exposed
  to WSL.
- Docker ComfyUI commonly reaches a host Werk instance at
  `http://host.docker.internal:11434`; Linux Docker may require an explicit
  host-gateway mapping.
- Bind Werk with `--host 0.0.0.0` only when cross-environment access requires
  it. Restrict the firewall and keep API-key authentication enabled.

## Security

An API key typed directly into a node can be serialized into workflow JSON.
Prefer `WERK_API_KEY` for workflows that will be shared, and remove secrets
before export. Use HTTPS across untrusted networks. Never publicly expose
`--allow-unauthenticated`.

The client sends bearer credentials only to the exact Werk origin (matching
scheme, host, and effective port), rejects cross-origin redirects, bounds
downloads, and removes local filesystem paths from visible result metadata.
Werk URL outputs remain subject to Werk's output-retention policy.

The runtime-protocol transport independently enforces bounded JSON request and
response bodies, rejects every redirect, validates both the protocol envelope
and version, preserves typed error code/retry/request IDs, and redacts API keys
and opaque handoff values from errors. Runtime nodes never try legacy `/v1`
routes after `/werk/v1` fails.

Image decoding defaults to a 67,108,864-pixel allocation limit. Set
`WERK_MAX_IMAGE_PIXELS` before starting ComfyUI to choose another positive
limit. Video downloads default to a 536,870,912-byte (512 MiB) limit; set
`WERK_MAX_VIDEO_BYTES` to another positive byte count when a trusted workflow
needs larger artifacts. Audio downloads default to 268,435,456 bytes (256 MiB)
and use `WERK_MAX_AUDIO_BYTES`. Source audio has a separate aggregate
67,108,864-byte (64 MiB) PCM-WAV limit controlled by
`WERK_MAX_AUDIO_INPUT_BYTES`; it is enforced before Base64 allocation. The
smaller input default keeps Base64 plus JSON safely below Werk's default
128-MiB HTTP body limit, including the two-input voice-conversion path.
Vision input uses the same conservative 67,108,864-byte aggregate limit for
encoded PNG bytes, controlled by `WERK_MAX_VISION_INPUT_BYTES`. The existing
`WERK_MAX_IMAGE_PIXELS` limit applies to the aggregate vision batch as well.

## Tests

From the repository root:

```bash
python -m pytest -q utils/comfyUI/tests
```

Tests require no GPU, ComfyUI installation, model, or live Werk server.

## Publishing to the Comfy Registry

The node is validated and packaged by
`.github/workflows/comfyui-registry.yml` whenever this directory changes. A
push or pull request never publishes a release. Publication is an explicit
manual action and is allowed only from the repository's default branch.

Before the first publication:

1. Create or claim the `philipbodenbach` publisher in the Comfy Registry. If
   that publisher ID cannot be used, update `PublisherId` in `pyproject.toml`
   before publishing anything.
2. Create a Registry publishing API key for that publisher.
3. Add it to this GitHub repository as the Actions secret
   `REGISTRY_ACCESS_TOKEN`.
4. Confirm that the immutable node ID `werk1112` is available. Change
   `project.name` before the first publication if another ID is required.
5. Increment the semantic `project.version`, merge the change to the default
   branch, then run **ComfyUI Registry** from GitHub Actions. Optional release
   notes can be supplied through the workflow's `changelog` input.

Published Registry versions cannot be overwritten. Fixes require another
version increment. The workflow deliberately runs the Comfy CLI from this
directory so the Registry archive contains the node package rather than the
entire Werk1112 monorepo.

## License

These nodes are part of Werk1112 and use the repository-wide
[Elastic License 2.0](LICENSE). The adjacent `LICENSE` is the same authoritative
license text as the repository root and is included in standalone Comfy
Registry packages; it is not a separate license for the nodes.
