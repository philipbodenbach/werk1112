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

`werk_video_generation_api.json` and `werk_image_to_video_api.json` are API
prompts for the new native `VIDEO` path. Both make the task and
`preferred_model=wan22-ti2v-5b` explicit, leave backend selection at Werk's
adapter-/registry-driven `auto` route, connect **WERK Video Config**, and finish
at ComfyUI's [Save Video node](https://docs.comfy.org/built-in-nodes/SaveVideo).
The I2V prompt additionally connects `LoadImage` to `initial_image`; upload
`station.png` to ComfyUI's input directory or change that filename before
submitting it.

The examples assume:

- this directory is installed as `custom_nodes/comfyui-werk1112`;
- Werk is reachable at `http://127.0.0.1:11434` without authentication, or the
  connection input is replaced with a real key/environment-backed value;
- `black-forest-labs/FLUX.2-klein-4B` is installed, or the preferred model is
  changed to another discovered image model;
- the video examples' preferred alias was installed with
  `werk pull Wan-AI/Wan2.2-TI2V-5B-Diffusers --name wan22-ti2v-5b`, and current
  ComfyUI supplies native `VIDEO` and `SaveVideo` support.

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
