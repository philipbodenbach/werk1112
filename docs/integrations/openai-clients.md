# OpenAI-compatible clients

Werk1112 implements a focused OpenAI-compatible subset for model discovery,
chat completions and image generation. Other media, discovery, output and job
routes are OpenAI-inspired or Werk-native. The complete contract and current
gaps are documented in the [HTTP API reference](../api.md).

## Connection

Start Werk with authentication and, when useful, default chat and image models:

~~~bash
werk auth api-key generate
werk serve --model CHAT_MODEL --image-model IMAGE_MODEL
~~~

OpenAI clients use this base URL:

~~~text
http://127.0.0.1:11434/v1
~~~

Use the generated Werk key as the OpenAI API key. The normal authentication
header is:

~~~http
Authorization: Bearer sk-werk-example
~~~

Werk also accepts `X-API-Key`, although most OpenAI SDKs naturally send a bearer
token. Werk does not terminate TLS; put a trusted TLS reverse proxy in front of
the service before exposing it beyond a trusted host or network.

## Python SDK: model discovery and chat

~~~python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="sk-werk-replace-with-the-generated-value",
)

for model in client.models.list().data:
    print(model.id)

completion = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "user", "content": "Explain local inference routing briefly."}
    ],
    max_completion_tokens=128,
)
print(completion.choices[0].message.content)
~~~

Chat supports the documented subset of messages, streaming, temperature,
top-p, completion limits, stop strings and seed. Text and image content parts
can be used with a compatible vision model and runtime. Ordered
<code>image_url</code> parts accept data URLs or runtime-reachable URLs and an
optional <code>detail</code> hint. See
[Vision and visual quality assurance](vision.md) for a Python-independent curl
example, the supported backend matrix and visual-inspection guidance.

OpenAI function tools, named or string `tool_choice`, parallel-tool preference,
assistant `tool_calls` and `tool` result messages are supported when the
selected chat adapter is vLLM. This includes indexed partial tool-call deltas
in streaming responses. Werk forwards these values but does not execute tools
or configure vLLM's model-specific tool parser. Other adapters reject a
tool-requiring request with HTTP 400 and code `unsupported_tool_calling`; they
do not silently discard it. See the [chat request contract](../api.md#post-v1chatcompletions)
for a complete curl request and continuation shape.

Structured response formats, audio/video message content, log probabilities
and stream usage summaries are not implemented by this endpoint.

Unknown OpenAI chat fields are currently ignored during deserialization. A
request being accepted therefore does not prove that every supplied field was
applied. Use only the fields listed in the
[chat request contract](../api.md#post-v1chatcompletions).

## Python SDK: image generation

Werk defaults image responses to embedded Base64. This works with server-side
SDKs and avoids retaining a second managed output directory.

~~~python
import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="sk-werk-replace-with-the-generated-value",
)

result = client.images.generate(
    model="gpt-image-1",
    prompt="A tiny friendly red workshop robot repairing a wooden chair",
    size="1024x1024",
    n=1,
)

Path("robot-chair.png").write_bytes(
    base64.b64decode(result.data[0].b64_json)
)
~~~

`gpt-image-1` is a compatibility alias, not a downloaded OpenAI model. Image
model resolution follows this order:

1. an explicit installed Werk model ID wins;
2. an OpenAI alias uses `werk serve --image-model MODEL` when configured;
3. otherwise the default `--model` is used only if it declares image generation;
4. otherwise omission or an alias succeeds only when exactly one installed
   model declares image generation.

Recognized aliases include `dall-e-2`, `dall-e-3`,
`chatgpt-image-latest` and names beginning with `gpt-image-`.

Werk-aware clients may request `response_format: "url"`. The response then
contains an authenticated relative URL such as `/v1/outputs/OUTPUT_ID`, and the
managed output remains subject to Werk's retention policy. This is not a public
object-storage URL; the client must resolve it against the Werk server and send
the API key again.

## Compatibility boundary

The following routes use common OpenAI shapes:

- `GET /v1/models`
- `GET /v1/models/{id}`
- `POST /v1/chat/completions`
- `POST /v1/images/generations`

Image generation includes an additional `werk` diagnostics member. Image edits,
speech transcription and speech translation use JSON instead of OpenAI
multipart bodies. Werk currently has no OpenAI Responses, legacy Completions or
Embeddings endpoint. Video, generated audio, jobs, capabilities, parameters and
output downloads are Werk contracts and should not be assumed to match an
upstream OpenAI SDK.

For route-by-route fields, response schemas and status behavior, use
[docs/api.md](../api.md) as the canonical reference.
