# Example workflows

`werk_image_generation_workflow.json` is a ComfyUI UI workflow using the
classic workflow schema version `0.4`, which remains importable by ComfyUI
`0.29.0`. It includes node positions, widgets, typed links, Preview Image, and
Save Image nodes. The graph demonstrates the current explicit chain from
**WERK Image Models** through **WERK Routing Config** and **WERK Image Config**
into **WERK Image Generate**, whose `model` and `config` sockets are explicit
typed inputs in the graph.

`werk_image_generation_api.json` is the distinct API-prompt representation:
node IDs map to `{class_type, inputs}` objects and connected values use
`[origin_node_id, output_slot]`. It includes both `PreviewImage` and
`SaveImage` output nodes.

`werk_vision_inspection_api.json` is an API prompt for post-render visual QA.
It connects **Load Image** through **WERK Vision Models**, **WERK Vision
Config**, and **WERK Vision Analyze**, then asks for missing controls, clipped
or overflowing text, misaligned grids, overlaps, spacing, and hierarchy defects.
Replace its illustrative `qwen3-vl` alias with an installed model that Werk
reports as runtime-available for `image-understanding`, and upload or rename
`rendered-page.png` before submitting it.

`werk_video_generation_api.json` and `werk_image_to_video_api.json` are API
prompts for the new native `VIDEO` path. Both make the task and
`preferred_model=wan22-ti2v-5b` explicit, leave backend selection at Werk's
adapter-/registry-driven `auto` route, connect **WERK Video Config**, and finish
at ComfyUI's [Save Video node](https://docs.comfy.org/built-in-nodes/SaveVideo).
The I2V prompt additionally connects `LoadImage` to `initial_image`; upload
`station.png` to ComfyUI's input directory or change that filename before
submitting it.

The audio API prompts exercise each native path:

- `werk_music_generation_api.json` uses Audio Models, Audio Config, Audio
  Generate, and ComfyUI's native `PreviewAudio`. It automatically selects the
  model when exactly one executable `music-generation` model is installed and
  uses an explicit CUDA/FP16 route, fixed seed 1112, and an empty negative
  prompt compatible with the Transformers MusicGen adapter;
- `werk_text_to_speech_api.json` mirrors the documented Qwen3-TTS VoiceDesign
  CUDA smoke test: explicit BF16 routing, fixed seed 1112, German language,
  first-class speaking-style instruction, and WAV output;
- `werk_audio_understanding_api.json` connects `LoadAudio` to Audio Analyze and
  supplies the required understanding prompt; and
- `werk_voice_conversion_api.json` connects separate source and reference
  `LoadAudio` nodes to Audio Process, then previews the native `AUDIO` result.
  This last prompt documents the prepared interface: Werk's bundled generic
  companion currently reports voice conversion as unavailable, so it requires
  a compatible external media backend before it can execute.

Upload/change `example.wav`, `source.wav`, and `reference.wav` before submitting
the input-audio prompts. The music example leaves `preferred_model` empty and
therefore requires exactly one executable music model; set an installed alias
when multiple candidates exist. Replace the illustrative preferred aliases
`audio-understanding-model` and `voice-conversion-model` with installed aliases
that declare the selected task. The TTS prompt names the
installed Qwen3-TTS VoiceDesign repository directly; replace it with its local
alias if one was assigned during `werk pull`. Analysis is itself an output
node, so its text/JSON values and metadata are returned in ComfyUI history
without an additional third-party text node.

The examples assume:

- this directory is installed as `custom_nodes/comfyui-werk1112`;
- Werk is reachable at `http://127.0.0.1:11434` without authentication, or the
  connection input is replaced with a real key/environment-backed value;
- `black-forest-labs/FLUX.2-klein-4B` is installed, or the preferred model is
  changed to another discovered image model;
- the video examples' preferred alias was installed with
  `werk pull Wan-AI/Wan2.2-TI2V-5B-Diffusers --name wan22-ti2v-5b`, and current
  ComfyUI supplies native `VIDEO` and `SaveVideo` support.
- the audio examples use a current ComfyUI with native `AUDIO`, `LoadAudio`,
  and `PreviewAudio` support, and their illustrative aliases are replaced with
  discovered executable Werk audio models. The voice-conversion example is
  intentionally unavailable with the bundled companion until a reliable
  generic adapter exists.
- the vision example uses a current ComfyUI `IMAGE`, sends the image batch as
  bounded inline PNG data URLs, and requires a Werk vision runtime; text-only
  execution of the same model repository is insufficient.

No credential is included. The example explicitly enables CPU and component
offload for the FLUX model. Update the model, routing choices, and connection
before running.

The video examples enable CPU offload but leave component/sequential offload
and the backend inherited. Their 1280x704, 121-frame, 24 FPS, 50-step,
guidance-5 values are the official Wan2.2 TI2V settings from the
[Wan configuration](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)
and [model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), not lightweight
defaults. The official flow shift 5 is already part of the Diffusers
[scheduler configuration](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/blob/main/scheduler/scheduler_config.json)
and is therefore inherited. The required local layout is the official
[`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
repository; the similarly named base repository uses Wan's native layout and
is not executable through Werk's bundled Diffusers adapter.

The image example uses `count=1` and `batch_size=1`. The latter is not sent
explicitly. For multiple images, increase `count`, which controls the request's
`n` value. Alternatively, an adapter that supports `image.batch_size` may use a
`batch_size` greater than 1 while `count` remains 1. The node rejects workflows
where both values are greater than 1; keeping the two counting mechanisms
mutually exclusive avoids the corresponding Diffusers conflict. Video Config
enforces the same mutual-exclusion rule for its two video-count controls.
