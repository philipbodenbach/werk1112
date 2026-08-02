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

The examples assume:

- this directory is installed as `custom_nodes/comfyui-werk1112`;
- Werk is reachable at `http://127.0.0.1:11434` without authentication, or the
  connection input is replaced with a real key/environment-backed value;
- `black-forest-labs/FLUX.2-klein-4B` is installed, or the preferred model is
  changed to another discovered image model.

No credential is included. The example explicitly enables CPU and component
offload for the FLUX model. Update the model, routing choices, and connection
before running.

The example uses `count=1` and `batch_size=1`. The latter is not sent
explicitly. For multiple images, increase `count`, which controls the request's
`n` value. Alternatively, an adapter that supports `image.batch_size` may use a
`batch_size` greater than 1 while `count` remains 1. The node rejects workflows
where both values are greater than 1; keeping the two counting mechanisms
mutually exclusive avoids the corresponding Diffusers conflict.
