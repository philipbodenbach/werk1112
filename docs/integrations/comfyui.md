# ComfyUI integration choices

Werk1112 supports two distinct ComfyUI integrations:

1. ComfyUI's built-in hosted OpenAI image node can call Werk's compatibility
   proxy for text-to-image generation.
2. The Werk1112 custom-node package provides native Werk discovery, routing,
   image, video and audio workflows, plus versioned runtime persistence,
   memory and expert telemetry, dry-run-first state/expert maintenance, and
   split prefill/decode controls.

These choices are complementary. Neither turns `werk serve` into a ComfyUI
workflow server.

## Hosted OpenAI image node

Start Werk with an image model and a configured API key:

~~~bash
werk auth api-key generate --name comfyUI
werk serve --image-model IMAGE_MODEL
~~~

Start ComfyUI with Werk as the hosted API base:

~~~bash
python main.py --comfy-api-base http://127.0.0.1:11434
~~~

The base is the Werk server root, without `/v1`, because ComfyUI calls:

~~~text
POST /proxy/openai/images/generations
~~~

Set ComfyUI's `api_key_comfy_org` value to the generated Werk key. The hosted
node sends it as `X-API-Key`, which Werk accepts. Its generation request uses
the same handler as `POST /v1/images/generations` and receives the expected
`data[].b64_json` output.

The hosted integration is intentionally narrow:

- text-to-image generation is supported;
- the corresponding multipart image-edit proxy returns `501 Not Implemented`;
- image streaming and partial-image events are not supported;
- an OpenAI image alias resolves through `werk serve --image-model`;
- the node does not expose Werk's complete routing, parameter discovery, jobs or
  diagnostics surface.

When ComfyUI runs in Docker, WSL, a VM or another host, replace `127.0.0.1` with
an address reachable from that environment and bind Werk accordingly. Do not
expose the unauthenticated service to an untrusted network.

## Werk1112 custom nodes

Use the custom-node package when a workflow needs Werk-native behavior such as:

- installed-model and available-task discovery;
- live parameter-schema discovery;
- explicit backend, accelerator, precision and fallback routing;
- synchronous image generation;
- persisted video generation and polling;
- native ComfyUI `IMAGE`, `VIDEO` and `AUDIO` values;
- native `IMAGE` batches sent to `POST /v1/chat/completions` for visual QA and
  image understanding;
- audio generation, speech, transcription, analysis and transform jobs;
- authenticated output retrieval and sanitized inference metadata;
- strict `/werk/v1` capability and memory discovery;
- typed persistence policies and dry-run-first runtime-state maintenance;
- bounded expert-residency telemetry and explicit dry-run-first expert actions;
- opaque prefill/decode handoffs that never use a `STRING` or JSON socket.

The vision path uses **WERK Vision Models**, **WERK Vision Config**, and
**WERK Vision Analyze**. It is intended for inspecting rendered HTML, slides,
documents, generated images, and UI screenshots for missing elements,
overflow, clipping, alignment, spacing, and similar visual defects. Werk must
report the selected model/runtime as available for `image-understanding`;
text-only execution of a multimodal repository is not sufficient. See the
[ready API-prompt example](https://github.com/philipbodenbach/werk1112/blob/main/utils/comfyUI/examples/werk_vision_inspection_api.json).

Installation, node sockets, workflow examples, limits and troubleshooting are
maintained beside the package in the
[ComfyUI Werk1112 node documentation](https://github.com/philipbodenbach/werk1112/blob/main/utils/comfyUI/README.md). That page
is the source of truth for the custom nodes and is not duplicated here.

The custom nodes call Werk over HTTP. They do not load Werk models from
ComfyUI's own model directories, and they do not require Werk to implement a
Comfy workflow graph.

The runtime nodes are additive. Existing inference nodes retain their `/v1`
requests and compatibility with older Werk servers. Runtime nodes deliberately
fail closed if the versioned `/werk/v1` envelope, protocol version, capability
status, or typed response is missing or invalid. State prune defaults to dry
run and never removes models, artifacts, outputs, jobs, authentication data,
backend installations, or external output paths. Prefill returns an opaque
`WERK_STATE_HANDOFF` consumed only by Decode; it is excluded from visible JSON,
strings, representations, and error messages. Expert nodes are gated by
`runtime.experts.residency`; they expose all six capability statuses without
turning route presence into a support claim. The list permits read-only
`externally_managed` telemetry, while control requires `supported` or an
explicitly opted-in `experimental` capability. Current production adapters
remain truthfully unsupported.

The runtime-node client sends `Accept: application/json` and
`X-Werk-Protocol-Version: 1.0`. It requires the versioned JSON envelope and
checks an HTTP protocol-version response header against it when present. A
missing response header remains accepted for compatibility with an earlier 1.0
server; malformed, newer or contradictory version declarations fail closed.

## What Werk does not emulate

Werk does not implement ComfyUI's native:

- `/prompt`
- `/history`
- `/view`
- queue protocol
- WebSocket protocol
- custom-node execution protocol
- arbitrary workflow graph

Continue to run a real ComfyUI server for those facilities. ComfyUI remains
responsible for workflow execution, previews, history and saved workflow media;
Werk performs model discovery, inference routing and the requested model call.

For the hosted proxy contract and its authentication/error behavior, see the
[HTTP API reference](../api.md). For model layout and media runtime requirements,
see [Media inference](../media-inference.md). For the runtime nodes' server
contract and backend limits, see [Werk Protocol 1.0](../reference/werk-protocol-v1.md)
and the [runtime capability matrix](../concepts/runtime-persistence-and-memory.md#current-production-capability-matrix).
