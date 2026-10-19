# Media CLI inputs, routing and parameters

This page defines the shared contract for `werk image`, `werk video` and
`werk audio`. Exact per-command syntax and every accepted option are listed in
the [image](cli-image.md), [video](cli-video.md) and [audio](cli-audio.md)
catalogues. The [canonical parameter schema](parameters-schema.md) lists types,
defaults, constraints and task applicability. [CLI semantics](cli.md) covers
`run`, model management, diagnostics and serving.

## Task selection

`MODEL` is an installed model ID. Importing a model, parsing an option or
recognizing a task does not establish backend support. Use
`werk doctor --model MODEL --task TASK` and
`werk parameters MODEL --task TASK --json --sources` to inspect the concrete
model. The [execution matrix](../media-inference.md#execution-support) distinguishes
implemented adapters from tasks that can only be represented or planned.

| Command | Selected task / required input |
| --- | --- |
| `image generate MODEL` | `image-generation`; requires a prompt. `--init-image` selects `image-editing`; adding `--mask` selects `image-inpainting`. A mask without an initial image fails. |
| `image edit MODEL --image SOURCE` | `image-editing`; with `--mask`, `image-inpainting`. Requires a prompt. |
| `image upscale MODEL --image SOURCE` | `image-upscaling`; optional prompt. |
| `video generate MODEL` | `video-generation`; `--initial-image` selects `image-to-video`. Requires a prompt. |
| `video animate MODEL --image SOURCE` | `image-to-video`; requires a prompt. |
| `video transform MODEL --video SOURCE` | `video-to-video`; requires a prompt. |
| `video upscale MODEL --video SOURCE` | `video-upscaling`. |
| `audio generate speech MODEL` | `text-to-speech`; requires text. |
| `audio generate music MODEL` | `music-generation`; requires a prompt or lyrics. |
| `audio generate sound MODEL` | `audio-generation`; requires a prompt or lyrics. |
| `audio transcribe MODEL --input SOURCE` | `speech-to-text`; `--task translate` selects `speech-translation`. |
| `audio translate MODEL --input SOURCE` | `speech-translation` (English translation); `--task transcribe` conflicts. |
| `audio detect event MODEL --input SOURCE` | `audio-event-detection`. |
| `audio detect voice MODEL --input SOURCE` | `voice-activity-detection`. |
| `audio detect speaker MODEL --input SOURCE` | `speaker-identification`. |
| `audio detect language MODEL --input SOURCE` | `language-identification`. |
| `audio detect emotion MODEL --input SOURCE` | `speech-emotion-recognition`. |
| `audio analyze caption MODEL --input SOURCE` | `audio-captioning`. |
| `audio analyze diarize MODEL --input SOURCE` | `speaker-diarization`. |
| `audio analyze classify MODEL --input SOURCE` | `audio-classification`. |
| `audio analyze understand MODEL --input SOURCE` | `audio-understanding`; requires a prompt. |
| `audio transform voice MODEL --input SOURCE` | `voice-conversion`; optional `--reference-audio`. |
| `audio transform separate MODEL --input SOURCE` | `stem-separation`. |
| `audio transform enhance MODEL --input SOURCE` | `audio-enhancement`. |
| `audio transform edit MODEL --input SOURCE` | `audio-editing`; requires a prompt. |
| `audio embed MODEL --input SOURCE` | `audio-embedding`. |

Other canonical tasks, such as image variation/outpainting, video extension,
video inpainting and frame interpolation, have no dedicated command name.
Express them with `werk run MODEL --task TASK` and canonical inputs/parameters
or `--request request.json`. Execution still requires an adapter for that task.
There is no implicit model installation.

The compatibility form `audio generate MODEL` infers music/audio generation
from the manifest. With `--source-audio`, it prefers song continuation when a
continuation start/duration is supplied and the model supports continuation;
otherwise it prefers supported song variation, then supported continuation.
The explicit `music` and `sound` forms keep their named task even with a source.
For an unambiguous continuation request, use `run --task song-continuation`.

## Text and media input

For a primary prompt/text, the first nonempty source wins: `--prompt` or
`--text`, then the matching `--prompt-file` or `--text-file`, then piped stdin,
then an interactive line if the task requires text. Files must be UTF-8; text
is trimmed. Missing required text fails. Optional prompts can consume piped
stdin but do not prompt interactively.

`--negative-prompt` and `--lyrics` use only their explicit value or corresponding
`--negative-prompt-file` / `--lyrics-file`; they do not independently consume
stdin. An explicitly empty optional value suppresses its file fallback.
For music/sound, supplied lyrics are kept as `audio.lyrics` and also supply
the required prompt when no prose prompt is supplied.

```bash
werk image generate image-model --prompt-file prompt.txt --output picture.png
werk audio generate speech voice-model --text-file speech.txt --output speech.wav
werk audio generate music music-model --lyrics-file lyrics.txt --output song.wav
```

Media source arguments are normalized as local paths, `http://`/`https://`
URLs or `data:MIME;base64,DATA` sources. Availability and supported formats
are checked by the runtime. A source flag is distinct from a UTF-8 text-file
flag and from a structured JSON-file flag. `--voice-reference` is a TTS
parameter handled by the selected runtime, not the primary speech input.

Canonical `run --input MODALITY[:ROLE]=SOURCE` is repeatable. Without a role,
the modality name is used as the role. Use the runtime's expected role for
audio, video, masks and conditioning inputs:

| Typed CLI input | Canonical modality / role |
| --- | --- |
| Image `--image` / `--init-image` | `image:image` |
| Image `--mask` | `image:mask_image` |
| Video `--image` / `--initial-image` | `image:initial_image` |
| Video `--video` | `video:source_video` |
| Audio `--input` / `--source-audio` | `audio:input_audio` |
| Audio `--reference-audio` | `audio:reference_audio` |
| Music `--instrumental-audio`, `--vocal-audio` | `audio:instrumental_audio`, `audio:vocal_audio` |

Reference lists and melody/rhythm/chord conditioning controls are normalized
parameters, rather than additional primary input roles in the typed CLI.

```bash
werk run image-model --task image-inpainting "Replace the background" \
  --input image:image=source.png --input image:mask_image=mask.png \
  --set image.steps=20 --output edited.png
werk run audio-model --task audio-captioning \
  --input audio:input_audio=recording.wav --json
```

## Resolution and support

Parameter resolution proceeds through system, task, family, model, runtime,
hardware/quality, saved profile, request and backend-adjustment layers. Omission
inherits. Schema defaults are starting values, not guarantees about the final
model request. Query `parameters MODEL --task TASK --sources --json` for
provenance and model constraints; `--example` prints a canonical request.

`--profile NAME` reads `WERK_HOME/profiles/NAME.json`. The file must contain a
JSON object mapping parameter paths to values, for example
`{"image.steps":20,"image.width":768,"image.height":768}`. A missing profile or
invalid name fails. Names are 1–128 ASCII letters/digits, hyphens, underscores
or dots, excluding `.` and `..`. There is no standalone profile-management command; create
the profile file yourself. Server deployment profiles instead use global
`--deployments` and a different configuration format.

| Routing option | Meaning |
| --- | --- |
| `--backend` | Global runtime selection. For media, `cpu`, `cuda`, `rocm` and `metal` become accelerator constraints; `auto` leaves backend selection open. Other explicit backends constrain the runtime family. oMLX rejects media tasks. |
| `--accelerator` | Explicit accelerator label. Takes priority over a backend-derived accelerator; conflicts with a different explicit non-auto `--device`. |
| `--device` | A global option primarily for Candle; in media it also contributes the requested device/accelerator. |
| `--precision`, `--quantization` | Requested compute precision and weight quantization; supported values depend on the model/runtime. |
| `--quality` | `draft`, `balanced`, `high`, `maximum`; schema default `balanced`. |
| `--performance-preference` | `quality`, `balanced`, `speed`, `latency`, `throughput`, `memory`; schema default `balanced`. |
| `--fallback-policy` | `none` prevents runtime retry; `backend` permits accepted backend alternatives; `degrade` additionally permits inherited memory-saving strategies. Schema default `backend`. |
| `--parameter-policy` | `strict` (default), `warn` or `permissive`; handling of explicitly requested unsupported parameters. |
| `--allow-cpu-offload`, `--allow-sequential-offload`, `--allow-component-offload`, `--allow-disk-offload` | Permit that memory strategy. Their `--no-allow-…` partners forbid it. Omission inherits policy; host-memory feasibility is still checked. Permission is not proof of implementation. |
| `--attention-backend` | Requested attention implementation; supported labels depend on the runtime. |
| `--compile`, `--no-compile` | Explicit runtime graph-compilation request or prohibition. |
| `--timeout` | Execution timeout in seconds; backend execution and cancellation support still apply. |
| `--verbose`, `--debug` | Timing/throughput statistics and resolved routing/backend diagnostics, respectively. |

Runtime parameter support is reported as `native`, `translated`, `emulated`,
`ignored`, `unsupported` or `model_dependent`. Under strict policy, explicitly
ignored/unsupported controls fail instead of silently becoming effective.
Warn policy records warnings; permissive policy can continue without applying
the control. Neither enables missing backend functionality. Many advanced
controls are model-dependent contracts; the generic schema does not invent
scheduler names, adapter formats or universal numeric scales for them.

Schema validation enforces types, finite numbers, minima and enumerated values.
Model maxima are descriptive planning hints rather than blanket numeric clamps;
specific context/safety/backend limits can still reject requests. See
[parameter and hardware guidance](../media-inference.md#effective-parameters-and-provenance).

## Boolean and list overrides

For paired flags such as `--image-vae-tiling` / `--no-image-vae-tiling`, omission
means inherit, the positive flag explicitly enables, and the negative flag
explicitly disables. Passing both is a parser error. The same applies to
offload, compilation, masking, refinement, video processing, music vocals,
normalization, TTS streaming and transcription options. Video uses
`--include-audio` / `--exclude-audio` as its positive/negative pair.

List item flags are repeatable. Structured lists expose a direct item form,
an inline JSON form and a JSON file form:

| Domain | List flag families |
| --- | --- |
| Image | `--image-control`, `--reference-image`, `--image-lora`, `--image-adapter`, each with `-json` and `-file` variants. |
| Video | `--video-control`, `--reference-image`, `--video-camera-keyframe`, `--video-prompt-keyframe`, `--video-guidance-schedule`, `--video-denoise-schedule`, `--video-adapter`, `--video-adapter-schedule`, each with `-json` and `-file` variants. |
| Music | `--music-instrument`, `--music-mix-control`, `--music-mastering-control`, with `-json` and `-file` variants; `--stem` with `--stems-json` / `--stems-file`. |

Other lists use repeated scalar flags, such as `--sigma`, `--genre`, `--mood`,
`--hotword`, `--suppress-token`, `--temperature-fallback` and
`--pronunciation-dictionary`. Their element types are shown in the option syntax
and schema. A list of paths is not automatically read as JSON.

For structured `SPEC` items (controls, LoRAs, adapters, keyframes, schedules,
instruments, mix/mastering controls), `key=value,key=value` becomes an object.
Keys are trimmed and hyphens become underscores. Values parse as JSON where
possible, otherwise as strings. Empty values and duplicate keys fail. A plain
item without `=` stays a string. There is no escaping for commas inside this
shorthand; use JSON/file forms for nested values or literal commas.

```bash
werk image generate image-model --prompt "A landscape" \
  --image-lora 'path=adapter.safetensors,weight=0.7'
werk image generate image-model --prompt "A landscape" \
  --image-lora-json '[{"path":"adapter.safetensors","weight":0.7}]'
werk image generate image-model --prompt "A landscape" \
  --image-lora-file loras.json
```

These examples demonstrate serialization. The selected backend defines the
supported fields and actual adapter execution. A `-json` argument contains
JSON itself; a `-file` argument names a UTF-8 JSON file.

Repeated direct items are collected first, then file data, then inline JSON.
Arrays concatenate in that order. A non-array JSON value replaces the existing
value and must still pass schema validation. Shell argument order does not
change this precedence. Prefer a single representation when possible.

At resolution time, a plain array replaces the inherited list. An object with
`operation` expresses explicit inheritance/merging:

| JSON value | Effect |
| --- | --- |
| `{"operation":"inherit"}` | Keep inherited items; no nonempty `values` allowed. |
| `{"operation":"replace","values":[...]}` | Replace inherited items. |
| `{"operation":"add","values":[...]}` | Append to inherited items; `append` is also accepted. |
| `{"operation":"clear"}` | Explicit empty list; no nonempty `values` allowed. |

Unknown operation fields fail. The `values` member, when supplied, must be an
array. Use `--set 'image.loras={"operation":"clear"}'`, for example, to clear
inherited LoRAs.

## Advanced parameter paths

`--set PATH=VALUE` is repeatable. Valid JSON retains its type (`20`, `0.5`,
`true`, `null`, lists, objects); other values are strings. Quote the entire
argument in the shell for spaces, JSON and special characters. Repeating the
same raw path uses the last value. Prefer explicit dotted canonical paths such
as `image.steps`, `video.frames`, `audio.duration`, `tts.speed` or `stt.language`.

Typed media commands collect ordinary flags and structured inputs before
applying raw overrides. `--set` replaces an identical raw key; do not combine
different spellings of the same normalized parameter or positive/negative
boolean representations. The fully resolved request is validated afterward.
Transport fields such as the model, primary prompt, input source and output
destination must use their dedicated arguments or a canonical request.

For `werk run`, `--set routing.precision=float16` changes the request's routing
object. On typed media commands, use dedicated routing flags such as
`--precision float16`; their raw `--set` values enter the parameter layer.
`--batch-size` also exists globally for llama.cpp: the media commands that
declare their own batch flag use media batching. A canonical
`--set image.batch_size=2` or `--set video.batch_size=1` is unambiguous.

## Outputs and streaming

The image/video and analysis commands use `--output-format`; music, speech
generation and stem separation use `--format`. Formats and encoders are
task/runtime-specific; see schema enums and the model's parameters. An output
extension is not a substitute for requesting a format.

`--output PATH` (alias `--output-path`) publishes to a chosen file or directory.
Without it, generated names are published directly under `WERK_HOME/outputs`.
An explicit destination must be outside that managed output store. Multiple
results, existing directories and nonexistent paths without an extension are
treated as directory destinations; a single output with a file destination
uses that filename. Parent directories are created. Existing output files are
not silently overwritten. Explicit publication copies all outputs before
removing temporary managed results; default publication moves them, rolling
back earlier moves if one fails. Typed commands print output paths and metadata;
`werk run … --json` exposes structured results.

Typed media commands complete their outputs before returning. TTS `--streaming`
is a model/runtime parameter, not the text CLI's `--stream`/NDJSON transport.
`run --stream` is rejected for media tasks. For background jobs, polling,
cancellation and server-managed output retention, use the
[HTTP job/output API](../api.md) and [media guide](../media-inference.md).
