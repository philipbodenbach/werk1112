# Open WebUI integration

Open WebUI can use Werk1112 as an OpenAI-compatible chat service and as an image
generation backend. For images, its OpenAI engine is the preferred integration.
Werk also exposes a deliberately small AUTOMATIC1111 compatibility surface for
clients that require that protocol.

## Start Werk

Generate a key and select a concrete local image model:

~~~bash
werk auth api-key generate
werk serve --host 0.0.0.0 --image-model IMAGE_MODEL
~~~

Bind to `0.0.0.0` only when another environment, such as a container, must reach
Werk. It exposes the port on every host interface; use host firewall rules and a
trusted network. When both applications run directly on the same host, keep the
default loopback binding instead:

~~~bash
werk serve --image-model IMAGE_MODEL
~~~

## Chat connection

Configure an OpenAI-compatible connection in Open WebUI with:

~~~text
Base URL: http://WERK_HOST:11434/v1
API key:  <generated Werk API key>
~~~

The `/v1` suffix is required for OpenAI clients. Installed Werk text models are
returned by `GET /v1/models`; whether an individual model can execute depends on
its declared task and an available backend.

Werk implements a chat-completions subset rather than the complete OpenAI API.
In particular, automatic tool-call orchestration and the Responses API are not
provided by `werk serve`.

## Recommended image engine: OpenAI

Configure Open WebUI's image generation settings or equivalent environment
variables as follows:

~~~text
ENABLE_IMAGE_GENERATION=true
IMAGE_GENERATION_ENGINE=openai
IMAGES_OPENAI_API_BASE_URL=http://WERK_HOST:11434/v1
IMAGES_OPENAI_API_KEY=<generated Werk API key>
IMAGE_GENERATION_MODEL=gpt-image-1
IMAGE_SIZE=1024x1024
~~~

Typical host values are:

- `127.0.0.1` when both processes run directly on the same host;
- `host.docker.internal` when Open WebUI runs in Docker and Werk runs on the
  host, where that hostname is provided by the container runtime;
- a deliberately reachable host or VM address when the applications run in
  separate environments.

`gpt-image-1` is an alias resolved to the model selected with
`werk serve --image-model IMAGE_MODEL`. An explicit installed Werk model ID can
be used instead. Werk returns embedded Base64 by default, so Open WebUI does not
need to make a second authenticated output request.

The current boundary is text-to-image generation. Open WebUI's OpenAI multipart
image-edit request is not supported by Werk's JSON image-edit endpoint. Image
streaming, partial image events and transparent-background enforcement are also
not implemented.

## Alternative image engine: AUTOMATIC1111

Use this only when the client must speak the A1111 protocol:

~~~text
ENABLE_IMAGE_GENERATION=true
IMAGE_GENERATION_ENGINE=automatic1111
AUTOMATIC1111_BASE_URL=http://WERK_HOST:11434
AUTOMATIC1111_API_AUTH=werk:<generated Werk API key>
~~~

The A1111 base URL is the server root and therefore does **not** include `/v1`.
Werk accepts Basic authentication with username `werk` for these compatibility
routes. It implements txt2img, model listing, options and coarse progress only;
it is not an A1111 extension/script runtime. See the
[AUTOMATIC1111 integration guide](automatic1111.md) for the precise subset.

## Browser and container notes

Open WebUI normally calls Werk from its server process, so browser CORS does not
participate. If a browser frontend calls Werk directly, start Werk with the
exact trusted origin, for example:

~~~bash
werk serve --image-model IMAGE_MODEL \
  --cors-origin http://127.0.0.1:3000
~~~

Wildcard, `null` and `file://` origins are rejected. CORS does not replace API
key authentication.

For the underlying request fields, output lifetime and compatibility limits,
see the [HTTP API reference](../api.md).
