# Tool calling across backends and modalities

Werk separates adapter support from model behavior. Every implemented chat or
vision adapter accepts OpenAI function tools and returns structured tool calls.
A model's ability to select useful tools and follow schemas is not inferred
from its name or architecture. Actual loading, format, modality and runtime
availability checks still apply. Context admission includes tool schemas, history
and a conservative allowance for protocol formatting; exact token counts remain
the runtime tokenizer’s responsibility.

| Adapter | Tool transport |
| --- | --- |
| llama.cpp server (CUDA, ROCm, Vulkan, Metal, CPU) | Native OpenAI tools, Jinja templates, tool history and streamed deltas |
| vLLM | Native local/remote transport; configure the runtime's parser and auto-tool-choice flags |
| oMLX | Native parser when available for the requested options; generic protocol otherwise |
| Candle, Burn, ONNX Runtime | Generic structured tool protocol |
| MLX, MLX-VLM, Transformers compatibility | Generic protocol, retaining supported image inputs |
| Legacy llama adapters | Generic protocol when the optional runtime is compiled and available |
| Image, video, audio, music and specialist media runtimes | Callable functions via the existing inference/job service |

Tools do not add image understanding to a text-only runtime or make an absent
runtime available. Vision-capable adapters preserve images alongside tools.
`--backend cuda` with a GGUF model can use native llama.cpp tool calling without
switching the model to vLLM. Use the normal Werk serve command and a context
large enough for the messages, tool definitions and answer budget.

## Generic protocol

Generators without native tool transport receive the schemas and complete
conversation, including prior call IDs and results, through an explicit
`werk-tool-call-v1` prompt contract. An assistant call is encoded as:

```text
<tool_call>{"name":"function_name","arguments":{"key":"value"}}</tool_call>
```

Werk validates the function name, JSON-object arguments, required/named choice
and parallel-call limit, then returns standard `tool_calls` with unique IDs.
Ordinary JSON prose is not a tool call. Malformed calls and unmet required choices
produce errors; truncated or filtered generations never publish an executable call.
No call is invented. Tool-enabled generic streaming buffers the
model output (bounded to 16 MiB) until it can be validated, then emits text/tool deltas and original
usage. Dropping the response stream drops the upstream receiver.

The generic protocol is not constrained JSON Schema decoding. It rejects
`strict: true`; omit it or use a native runtime supporting the requested strict
contract. It does not guarantee that every model follows the prompt. Normal
requests without tools retain the original generator behavior.

## Discover functions for every modality

`GET /v1/tools` returns an authenticated `werk.tools` object with an OpenAI
`tools` array. It contains a function per canonical task, using underscores in
the name (`image_generation`, `image_to_video`, `music_generation`,
`text_to_speech`, `speech_to_text`, `audio_understanding`, `voice_conversion`,
and all other tasks), plus `get_job` and `cancel_job`.

Use `GET /v1/tools?task=image-generation` to select one task plus job controls.
Supply only relevant definitions to a chat model so the catalog fits its context.

This catalog describes the API contract independently of installed models.
`GET /v1/capabilities` supplies installed models, readiness and schemas for their
available tasks. A model must still support the chosen inference task and have
an available compatible runtime when executed.

Media functions accept `model`, optional `prompt` and `negative_prompt`, typed
`inputs`, and `parameters` with canonical dotted paths. Their schemas are
built from the same descriptors as `/v1/parameters`; runtime selection is
available as `parameters["routing.backend"]` and `parameters["routing.device"]`.
Chat functions (`text_generation`, `image_understanding`) accept OpenAI chat
messages, including vision content parts and tool configuration. `image_understanding`
requires image input; editing tools advertise their required source and mask roles.

## Execute a selected function

The client explicitly posts an OpenAI function call object. Use the same API
key as for the rest of Werk:

```http
POST /v1/tools/call
Authorization: Bearer YOUR_WERK_API_KEY
Content-Type: application/json

{
  "id": "call_image_1",
  "type": "function",
  "function": {
    "name": "image_generation",
    "arguments": "{\"model\":\"YOUR_IMAGE_MODEL\",\"prompt\":\"A lighthouse at dusk\",\"parameters\":{\"image.width\":512,\"image.height\":512}}"
  }
}
```

Media calls return HTTP 202 with compact job information in `result`. Request
payloads and Base64 inputs are excluded from tool results to preserve chat context. The response's
`message` is a standard `role: "tool"` message preserving `tool_call_id`, ready
to append to the conversation. Call `get_job` with arguments `{"id":"JOB_ID"}`
to obtain the final status, errors and output URLs; `cancel_job` uses the same
arguments. Output bytes remain available through `/v1/outputs/{id}` under the
existing authentication and retention rules. Cancellation uses existing
cooperative job semantics and may not immediately stop a running model kernel.

Chat/vision functions return HTTP 200 with the chat completion in `result`.
Nested model-selected calls are returned to the client and are never executed
automatically. Existing media endpoints and `/v1/jobs` remain available.
