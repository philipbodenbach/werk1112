# HTTP API reference and coverage

Werk1112 exposes an OpenAI-compatible subset together with Werk-native media,
discovery, output and job contracts. It also provides small compatibility
surfaces for ComfyUI's hosted OpenAI nodes and AUTOMATIC1111 clients.

Not every route is OpenAI wire-compatible. This page labels each contract
explicitly.

## Stability and source of truth

This document describes the current repository. Until a tested OpenAPI file is
added, the Axum router, request structs and API tests are the executable source
of truth.

Current surface:

- 21 unique paths
- 23 method/path operations
- JSON requests except raw output downloads
- server-sent events only for chat streaming
- persisted asynchronous jobs for video, generated audio and the native job API

## Base URL

The default server address is:

~~~text
http://127.0.0.1:11434
~~~

OpenAI-compatible clients normally configure:

~~~text
http://127.0.0.1:11434/v1
~~~

Generate a key and start the server:

~~~bash
werk auth api-key generate
export WERK_API_KEY="replace-with-generated-key"
werk serve --model CHAT_MODEL --image-model IMAGE_MODEL
~~~

Authentication is required by the CLI unless the server is explicitly started
with <code>--allow-unauthenticated</code>.

## Compatibility classes

| Class | Meaning |
| --- | --- |
| OpenAI-compatible subset | Common OpenAI request and response shapes are implemented, but only documented fields are supported. |
| OpenAI-inspired | The path or general purpose resembles OpenAI, but Werk adds or changes request/response behavior. |
| Werk-native | The route exposes Werk tasks, routing, estimates, plans, jobs or outputs directly. |
| Comfy alias | A compatibility path used by ComfyUI hosted nodes. |
| A1111 subset | A deliberately small AUTOMATIC1111-compatible surface. |

Do not infer compatibility with an entire upstream API from the path prefix.

## Complete route inventory

| Class | Method | Path | Success behavior |
| --- | --- | --- | --- |
| OpenAI-compatible | GET | <code>/v1/models</code> | Installed model list |
| OpenAI-compatible | GET | <code>/v1/models/{id}</code> | One model summary |
| OpenAI-compatible subset | POST | <code>/v1/chat/completions</code> | JSON completion or SSE stream |
| OpenAI-compatible extended | POST | <code>/v1/images/generations</code> | Synchronous Base64 or persisted Werk URL |
| Werk JSON | POST | <code>/v1/images/edits</code> | Synchronous JSON image edit/inpaint |
| Comfy alias | POST | <code>/proxy/openai/images/generations</code> | Same handler as image generation |
| Comfy alias | POST | <code>/proxy/openai/images/edits</code> | Authenticated 501; multipart is not implemented |
| Werk-native | POST | <code>/v1/videos/generations</code> | 202 persisted job |
| Werk-native | POST | <code>/v1/audio/generations</code> | 202 persisted job |
| OpenAI-inspired | POST | <code>/v1/audio/speech</code> | Raw audio or 202 job |
| OpenAI-inspired | POST | <code>/v1/audio/transcriptions</code> | Synchronous Werk JSON |
| OpenAI-inspired | POST | <code>/v1/audio/translations</code> | Synchronous Werk JSON |
| Werk-native | GET | <code>/v1/capabilities</code> | Model/task/runtime discovery |
| Werk-native | GET | <code>/v1/parameters</code> | Task/model parameter schema |
| Werk-native | GET | <code>/v1/outputs/{id}</code> | Persisted bytes |
| Werk-native | POST | <code>/v1/jobs</code> | 202 generic canonical task |
| Werk-native | GET | <code>/v1/jobs/{id}</code> | Persisted JobRecord |
| Werk-native | DELETE | <code>/v1/jobs/{id}</code> | Cooperative cancellation record |
| A1111 subset | POST | <code>/sdapi/v1/txt2img</code> | Synchronous embedded images |
| A1111 subset | GET | <code>/sdapi/v1/sd-models</code> | Installed image models |
| A1111 subset | GET | <code>/sdapi/v1/options</code> | Current compatibility model selection |
| A1111 subset | POST | <code>/sdapi/v1/options</code> | Update compatibility model selection |
| A1111 subset | GET | <code>/sdapi/v1/progress</code> | Coarse idle/active state |

## Authentication

All routes use the same configured API keys.

Normal Werk and OpenAI-style routes accept:

~~~http
Authorization: Bearer sk-werk-example
~~~

or:

~~~http
X-API-Key: sk-werk-example
~~~

A1111 routes additionally accept HTTP Basic authentication with username
<code>werk</code> and the API key as password.

Authentication properties:

- keys currently have equal permissions; there are no scopes or tenants;
- no built-in rate limiting is implemented;
- the server does not terminate TLS;
- remote deployments should use a TLS reverse proxy;
- browser access additionally depends on configured CORS origins.

## CORS

CORS is disabled when no browser origins are configured. When enabled, Werk
allows only the configured exact origins, methods GET/POST/DELETE, content and
authentication headers, and the OpenAI SDK request headers registered by the
router. The <code>x-werk-output-id</code> response header is exposed.

## Request body limits

The following JSON routes default to 128 MiB and can be configured up to
512 MiB with <code>WERK_API_BODY_LIMIT_BYTES</code>:

- <code>POST /v1/jobs</code>
- <code>POST /v1/audio/transcriptions</code>
- <code>POST /v1/audio/translations</code>

Other JSON routes retain Axum's smaller default, approximately 2 MiB. This is
especially relevant for Base64 image edits and large inline media.

Base64 expands binary data by roughly one third. Prefer server-local paths when
the client and server intentionally share a trusted filesystem.

## Common Werk media options

Every typed image, video and audio JSON request accepts its route-specific
fields plus these common fields:

| Field | Type | Purpose |
| --- | --- | --- |
| <code>routing</code> | object | Nested RoutingOverrides object |
| <code>backend</code> | string | Backend preference or <code>auto</code> |
| <code>accelerator</code> | string | Accelerator target such as CUDA, ROCm, MPS or CPU |
| <code>device</code> | string | Concrete device target where supported |
| <code>precision</code> | string | Requested dtype/precision |
| <code>quantization</code> | string | Requested quantization |
| <code>profile</code> | string | User/runtime profile |
| <code>quality</code> | string | Quality preference |
| <code>performance_preference</code> | string | quality, balanced, speed, latency, throughput or memory |
| <code>fallback_policy</code> | string | none, backend or degrade |
| <code>parameter_policy</code> | string | strict, warn or permissive |
| <code>allow_cpu_offload</code> | boolean | Permit model CPU offload |
| <code>allow_sequential_offload</code> | boolean | Permit sequential offload |
| <code>allow_component_offload</code> | boolean | Permit component offload |
| <code>allow_disk_offload</code> | boolean | Permit disk-backed offload |
| <code>attention_backend</code> | string | Requested attention implementation |
| <code>compile</code> | boolean | Request backend compilation |
| <code>timeout_seconds</code> | integer | Execution timeout |
| <code>parameters</code> | object | Canonical parameter overrides |
| <code>user</code> | string | Accepted OpenAI transport field; not a backend parameter |

Flat routing fields override the same values in the nested
<code>routing</code> object.

### Parameter normalization

Parameters may be supplied in several equivalent forms.

Nested:

~~~json
{
  "parameters": {
    "video": {
      "frames": 121,
      "fps": 24
    }
  }
}
~~~

Qualified:

~~~json
{
  "parameters": {
    "video.frames": 121,
    "video.fps": 24
  }
}
~~~

Route namespace shorthand:

~~~json
{
  "frames": 121,
  "fps": 24
}
~~~

An unqualified extra field is prefixed with the current task namespace. A field
already containing a dot remains qualified. A namespace object named
<code>image</code>, <code>video</code>, <code>audio</code>,
<code>tts</code> or <code>stt</code> is unfolded.

Precedence is:

1. nested routing values;
2. flat routing values;
3. values in <code>parameters</code>;
4. flat extra parameters;
5. explicit route transport fields such as <code>size</code>, <code>n</code>
   and <code>response_format</code>.

Under the default strict parameter policy, an explicit field unsupported by the
selected adapter rejects the request instead of being silently ignored.

## Media input forms

Convenience routes accept a media input as a string:

~~~json
"./input.wav"
~~~

a data URL:

~~~json
"data:audio/wav;base64,UklGR..."
~~~

or an object containing exactly one source:

~~~json
{"path": "/server/path/input.wav", "mime_type": "audio/wav"}
~~~

~~~json
{"base64": "UklGR...", "mime_type": "audio/wav"}
~~~

~~~json
{"url": "https://example.invalid/input.wav", "mime_type": "audio/wav"}
~~~

The transport schema accepts HTTP URLs, but the bundled offline media companion
does not fetch them. Server-local paths and Base64 are the currently executable
forms for that adapter.

The generic job route uses a tagged canonical input:

~~~json
{
  "modality": "audio",
  "role": "input_audio",
  "source": {
    "kind": "base64",
    "data": "UklGR..."
  },
  "mime_type": "audio/wav"
}
~~~

Source kinds are <code>path</code>, <code>url</code>, <code>base64</code> and
<code>text</code>. Required roles are task-specific.

## Task coverage

The generic job route parses every canonical task. Submission still requires
the installed model to declare that task. Later execution additionally requires
an accepted runtime and successful model load.

| Task | Dedicated route | Generic job | Generic companion status |
| --- | --- | --- | --- |
| text-generation | Chat completions | Syntactically accepted | Text runtime dependent |
| text-embedding | None | Syntactically accepted | Text runtime dependent; no OpenAI embeddings endpoint |
| image-understanding | Chat with image parts | Syntactically accepted | VLM/backend dependent |
| image-generation | Images generations | Yes | Executable for registered image pipelines |
| image-editing | Images edits | Yes | Executable for registered image pipelines |
| image-inpainting | Images edits with mask | Yes | Executable for registered image pipelines |
| image-variation | None | Yes | Executable for registered image pipelines |
| image-outpainting | None | Yes | Executable for registered image pipelines |
| image-upscaling | None | Yes | Executable for registered image pipelines |
| video-generation | Videos generations | Yes | Executable for registered video pipelines |
| image-to-video | Videos generations with initial image | Yes | Executable for registered video pipelines |
| video-to-video | None | Yes | Executable only when a registered adapter accepts the model |
| video-inpainting | None | Yes | Executable only when a registered adapter accepts the model |
| video-extension | None | Yes | Executable only when a registered adapter accepts the model |
| video-upscaling | None | Yes | Executable only when a registered adapter accepts the model |
| frame-interpolation | None | Yes | Executable only when a registered adapter accepts the model |
| audio-generation | Audio generations | Yes | Diffusers/Transformers model dependent |
| music-generation | Audio generations | Yes | Transformers/Diffusers model dependent |
| text-to-speech | Audio speech | Yes | Generic Transformers or architecture-specific adapter |
| speech-to-text | Audio transcriptions | Yes | ASR model dependent |
| speech-translation | Audio translations | Yes | Translation-capable ASR model dependent |
| audio event/voice/speaker/language/emotion detection | None | Yes | Whole-clip audio-classification model dependent |
| audio-classification | None | Yes | Audio-classification model dependent |
| audio-captioning | None | Yes | Any-to-any audio model dependent |
| audio-understanding | None | Yes | Any-to-any audio model dependent |
| audio-embedding | None | Yes | Registered processor/model dependent |
| song continuation/variation | None | Yes | Prepared; no generic adapter |
| speaker-diarization | None | Yes | Prepared; no generic adapter |
| voice-conversion | None | Yes | Prepared; no generic adapter |
| stem generation/separation | None | Yes | Prepared; no generic adapter |
| audio-enhancement | None | Yes | Prepared; no generic adapter |
| audio-editing | None | Yes | Prepared; no generic adapter |

“Generic job: yes” means the typed API contract exists. It does not promise an
execution adapter.

## Models

### GET /v1/models

Returns OpenAI-style model summaries:

~~~json
{
  "object": "list",
  "data": [
    {
      "id": "model-name",
      "object": "model",
      "created": 0,
      "owned_by": "local"
    }
  ]
}
~~~

It intentionally does not return the full Werk manifest. Use
<code>werk inspect MODEL</code> for local diagnostics.

### GET /v1/models/{id}

Returns the same summary shape for one installed ID, or 404.

## Chat completions

### POST /v1/chat/completions

Accepted top-level fields:

| Field | Required | Type | Behavior |
| --- | --- | --- | --- |
| <code>model</code> | Conditional | string | May be omitted when the server has a default chat model |
| <code>messages</code> | Yes | array | Chat messages |
| <code>stream</code> | No | boolean | Default false |
| <code>temperature</code> | No | number | Backend dependent |
| <code>top_p</code> | No | number | Backend dependent |
| <code>max_tokens</code> | No | integer | Legacy response budget |
| <code>max_completion_tokens</code> | No | integer | Wins over max_tokens |
| <code>stop</code> | No | string or string array | Added to model/template stops |
| <code>seed</code> | No | integer | Backend dependent |

The response-token default is 256 when neither field is present. Werk does not
silently clamp an explicit response budget; the selected model context and
backend remain authoritative and return an actionable error when they cannot
satisfy it.

Message content is either a string or an array of content parts. Supported
parts are text plus <code>image_url</code> or <code>input_image</code> for an
image-capable model/backend.

Example:

~~~bash
curl -fsS http://127.0.0.1:11434/v1/chat/completions \
  -H "Authorization: Bearer $WERK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Explain runtime routing briefly."}],
    "max_completion_tokens": 256
  }'
~~~

Not implemented:

- tools and tool calls
- structured <code>response_format</code>
- audio or video content
- choice count <code>n</code>
- log probabilities
- frequency and presence penalties
- stream usage summaries

Unknown chat fields are currently ignored by deserialization. Clients should
not interpret acceptance as support.

### Chat SSE

With <code>stream: true</code>, the response content type is
<code>text/event-stream</code>. The event sequence is:

1. assistant-role chunk;
2. one or more content chunks;
3. finish chunk;
4. literal <code>data: [DONE]</code>.

Output chunks are buffered text pieces rather than necessarily one token each.
If generation fails after the HTTP stream starts, the failure is emitted as an
error event inside the already successful HTTP response.

## Image generation

### POST /v1/images/generations

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| <code>model</code> | Conditional | string | May resolve from the configured/default single image model |
| <code>prompt</code> | Yes | string | Positive prompt |
| <code>negative_prompt</code> | No | string | Negative prompt |
| <code>n</code> | No | integer | Maps to image.num_images |
| <code>size</code> | No | string | WIDTHxHEIGHT or auto |
| <code>response_format</code> | No | string | b64_json/base64 default, or url |
| <code>output_format</code> | No | string | Image encoding requested from the adapter |
| <code>style</code> | No | string | vivid or natural; appended as a prompt hint |
| <code>background</code> | No | string | auto/opaque accepted; transparent rejected |
| <code>moderation</code> | No | string | auto/low accepted as transport compatibility only |
| <code>output_compression</code> | No | integer | Validated 0–100; not forwarded by the current adapter |
| <code>partial_images</code> | No | integer | Must be zero/omitted |
| <code>stream</code> | No | boolean | true is rejected |

The default response embeds Base64 and then cleans up the temporary result.
With <code>response_format: url</code>, the returned relative output URL is
authenticated and retained subject to output retention.

## Image editing and inpainting

### POST /v1/images/edits

This is a Werk JSON endpoint, not OpenAI multipart.

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| <code>model</code> | Yes | string | Installed image model |
| <code>prompt</code> | Yes | string | Edit prompt |
| <code>image</code> | Yes | media input | Source image |
| <code>mask</code> | No | media input | Presence selects image-inpainting |
| <code>negative_prompt</code> | No | string | Negative prompt |
| <code>n</code> | No | integer | Output count |
| <code>size</code> | No | string | WIDTHxHEIGHT |
| <code>response_format</code> | No | string | b64_json/base64 or url |

ComfyUI's hosted OpenAI multipart edit proxy reaches
<code>/proxy/openai/images/edits</code>, which currently returns 501 with
guidance to use this JSON route.

## Video generation

### POST /v1/videos/generations

Always returns 202 with a JobRecord.

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| <code>model</code> | Yes | string | Installed video model |
| <code>prompt</code> | Yes | string | Positive prompt |
| <code>initial_image</code> | No | media input | Alias: image; presence selects image-to-video |
| <code>negative_prompt</code> | No | string | Negative prompt |
| <code>n</code> | No | integer | Maps to video.num_videos |
| <code>size</code> | No | string | WIDTHxHEIGHT |
| <code>response_format</code> | No | string | Maps to video.output_format |

Example:

~~~bash
curl -fsS http://127.0.0.1:11434/v1/videos/generations \
  -H "Authorization: Bearer $WERK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wan22-ti2v-5b",
    "prompt": "A bird takes flight through warm forest sunbeams.",
    "size": "1280x704",
    "parameters": {
      "video.frames": 121,
      "video.fps": 24,
      "video.steps": 50,
      "video.guidance": 5
    },
    "precision": "bf16",
    "allow_cpu_offload": true
  }'
~~~

## Generated audio and music

### POST /v1/audio/generations

Always returns 202 with a JobRecord.

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| <code>model</code> | Yes | string | Installed audio model |
| <code>prompt</code> | Yes | string | Generation prompt |
| <code>negative_prompt</code> | No | string | Model dependent |
| <code>task</code> | No | string | audio-generation default or music-generation |
| <code>n</code> | No | integer | Maps to audio.variations |
| <code>response_format</code> | No | string | Maps to audio.output_format |

Other audio tasks are rejected by this convenience route and should use their
dedicated endpoint or <code>/v1/jobs</code>.

## Text to speech

### POST /v1/audio/speech

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| <code>model</code> | Yes | string | Installed TTS model |
| <code>input</code> | Yes | string | Alias: text |
| <code>voice</code> | No | string | Model/adapter dependent |
| <code>speed</code> | No | number | Model/adapter dependent |
| <code>response_format</code> | No | string | Maps to tts.output_format |
| <code>async</code> | No | boolean | Aliases: background, job |

By default Werk waits for generation, then returns the completed audio bytes
with:

- <code>Content-Type</code>
- <code>Content-Length</code>
- <code>x-werk-output-id</code>

This is completed-file transfer, not incremental speech generation. The
temporary result is deleted when the response stream ends or is abandoned, so
the output ID header is primarily diagnostic.

With <code>async: true</code>, the route returns a normal persisted JobRecord.

## Speech transcription and translation

### POST /v1/audio/transcriptions

### POST /v1/audio/translations

Both routes use Werk JSON rather than OpenAI multipart.

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| <code>model</code> | Yes | string | Installed ASR model |
| <code>file</code> | Yes | media input | Aliases: audio, input_audio |
| <code>prompt</code> | No | string | Maps to stt.initial_prompt |
| <code>language</code> | No | string | Source language hint |
| <code>temperature</code> | No | number | Model dependent |
| <code>response_format</code> | No | string | text/json/srt/vtt/tsv where supported |

Translation fixes <code>stt.operation</code> to <code>translate</code> and
requires a model that actually supports speech translation. A generic CTC ASR
model is not automatically translation-capable.

The synchronous response is the direct media envelope with text embedded in
<code>data[].text</code>. Its temporary output is cleaned after response
construction.

## Capabilities and parameters

### GET /v1/capabilities

Returns installed declarations and runtime probe information. Declared tasks
must be distinguished from available/executable tasks.

### GET /v1/parameters

Query parameters:

| Parameter | Required | Meaning |
| --- | --- | --- |
| <code>task</code> | Yes | Canonical task name |
| <code>model</code> | No | Installed model for constraints and runtime probing |
| <code>backend</code> | No | Filter runtime candidates; auto leaves all |

Example:

~~~bash
curl -fsS --get http://127.0.0.1:11434/v1/parameters \
  -H "Authorization: Bearer $WERK_API_KEY" \
  --data-urlencode "task=text-to-speech" \
  --data-urlencode "model=qwen3-tts-voice-design"
~~~

The response includes:

- canonical parameter descriptors;
- selected parameter-support information;
- runtime candidates;
- model-specific parameter constraints.

## Generic jobs

### POST /v1/jobs

Required body:

| Field | Required | Type |
| --- | --- | --- |
| <code>model</code> | Yes | string |
| <code>task</code> | Yes | canonical task string |
| <code>prompt</code> | Task-dependent | string |
| <code>negative_prompt</code> | No | string |
| <code>inputs</code> | Task-dependent | canonical input array |
| common Werk options | No | routing and parameters |

Example:

~~~bash
curl -fsS http://127.0.0.1:11434/v1/jobs \
  -H "Authorization: Bearer $WERK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ast-audioset",
    "task": "audio-classification",
    "inputs": [{
      "modality": "audio",
      "role": "input_audio",
      "source": {"kind": "path", "path": "/shared/field-recording.wav"},
      "mime_type": "audio/wav"
    }],
    "parameters": {"audio.top_k": 5}
  }'
~~~

A 202 response means the request resolved and was queued. It does not guarantee
that a runtime plan will exist, a dependency will import, the model will load
or accelerator memory will suffice.

### GET /v1/jobs/{id}

Returns the complete persisted JobRecord.

### DELETE /v1/jobs/{id}

Marks a nonterminal job cancelled and returns its JobRecord. It does not delete
the record or output files. Cancellation is cooperative: a third-party model
call already in progress may continue, but a later result is discarded.

### Job states

~~~text
queued
loading
running
encoding
completed
failed
cancelled
~~~

The current <code>encoding</code> state is a short lifecycle transition after
service execution, not an encoder progress meter.

Nonterminal jobs found after a server restart are marked failed. They are not
resumed.

## JobRecord schema

~~~json
{
  "id": "job-...",
  "status": "queued",
  "request": {
    "model": "model-name",
    "task": "video_generation",
    "prompt": "...",
    "negative_prompt": null,
    "inputs": [],
    "parameters": {},
    "routing": {}
  },
  "result": null,
  "error": null,
  "created_unix": 0,
  "updated_unix": 0
}
~~~

On completion, <code>result</code> contains the complete InferenceResult.

Task parsing accepts hyphenated and underscored input names. Serialized Rust
enum fields currently use snake_case. For example a request may send
<code>music-generation</code> while a JobRecord returns
<code>music_generation</code>. Clients should currently tolerate both.

## Direct media response

Synchronous image/edit/transcription/translation routes return:

~~~json
{
  "created": 0,
  "data": [
    {
      "id": "out-...-0",
      "url": "/v1/outputs/out-...-0",
      "mime_type": "image/png",
      "size_bytes": 123,
      "width": 1024,
      "height": 1024,
      "duration": null
    }
  ],
  "werk": {
    "id": "out-...",
    "task": "image_generation",
    "model": "model-name",
    "runtime": "media-companion-cuda",
    "outputs": [],
    "effective_request": {},
    "estimate": {},
    "plan": {},
    "backend_metadata": {},
    "timings": {},
    "warnings": [],
    "created_unix": 0
  }
}
~~~

Depending on route and response format, each data item contains one of:

- <code>b64_json</code>
- <code>url</code>
- <code>text</code>

The <code>werk</code> member intentionally exposes diagnostics including the
effective request, estimate, plan, runtime attempts and warnings. It is much
larger than a normal OpenAI response extension.

## InferenceResult and output metadata

An InferenceResult includes:

- result ID, task, model and selected runtime;
- output metadata;
- effective request with parameter source/provenance;
- workload estimate;
- complete execution plan and candidate decisions;
- backend metadata;
- phase and runtime-attempt timings;
- warnings and creation time.

Output metadata includes:

- output ID;
- task/model/runtime;
- server-side path;
- MIME type and byte size;
- optional width, height, duration and seed;
- effective parameters and backend metadata.

## Output download and retention

### GET /v1/outputs/{id}

The path takes an output ID, not a job ID or result ID. It returns the whole
file with its content type and length.

HTTP Range requests and object-storage redirects are not implemented.

Persisted output retention defaults to:

- 30 days
- 20 GiB total

Configure these with:

~~~text
WERK_OUTPUT_RETENTION_SECONDS
WERK_OUTPUT_MAX_BYTES
~~~

Retention is enforced before later inference. Oldest/expired output entries can
therefore disappear while an older completed JobRecord still retains stale
output metadata.

There is no output deletion endpoint today.

## Error response

Most Werk routes use:

~~~json
{
  "error": {
    "message": "human-readable explanation",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
~~~

Common statuses:

| Status | Typical meaning |
| --- | --- |
| 200 | Synchronous success, poll success or cancellation record |
| 202 | Job validated and queued |
| 400 | Invalid task/model/parameter, no accepted runtime, or execution/model-load failure |
| 401 | Missing or invalid credentials |
| 404 | Missing model, job or output |
| 413 | Request body over the route limit |
| 500 | Internal task, persistence or response construction failure |
| 501 | Unsupported Comfy multipart image edit proxy |

Current caveats:

- 401, 404 and 500 still use the generic
  <code>invalid_request_error</code> type;
- <code>code</code> is currently null;
- JSON extraction and body-limit failures may use Axum's response rather than
  the Werk envelope;
- model-load/backend errors are often reported as 400;
- A1111 compatibility routes use their own error shape.

## AUTOMATIC1111 subset

The supported compatibility routes are:

~~~text
POST /sdapi/v1/txt2img
GET  /sdapi/v1/sd-models
GET  /sdapi/v1/options
POST /sdapi/v1/options
GET  /sdapi/v1/progress
~~~

txt2img maps the useful core fields:

- prompt and negative_prompt
- seed, where -1 requests a generated seed
- batch_size multiplied by n_iter
- steps and cfg_scale
- width and height

Sampler, scheduler and ETA compatibility fields can be accepted with warnings
while Werk/runtime selection remains authoritative.

Explicitly unsupported features include high-resolution fix, face restoration,
seamless tiling, named styles, refiners, scripts/always-on scripts, image
saving by the A1111 server and nondefault subseed behavior.

Progress reports coarse idle/active state. It does not invent a percentage or
preview image when the backend has no progress callback.

## Security and privacy considerations

The current diagnostic contracts are intentionally transparent but can contain
sensitive information:

- direct responses include the effective request and absolute server output
  paths inside <code>werk</code>;
- JobRecord stores the complete original request;
- inline Base64 input is therefore persisted in job JSON;
- prompts, negative prompts and server-local input paths can be persisted;
- job records have no independent retention or deletion API;
- all API keys currently have equal access.

Do not expose Werk directly to an untrusted network. Use TLS termination,
access controls and a trusted reverse proxy. Prefer dedicated store roots for
different trust domains.

## Known API gaps

The following are not currently implemented:

- <code>POST /v1/embeddings</code>
- OpenAI Responses API
- legacy text Completions API
- multipart image edits, transcriptions and translations
- dedicated endpoints for video transforms
- dedicated endpoints for audio detection, analysis, transform and embedding
- list/delete APIs for jobs
- delete API for outputs
- health/readiness endpoint
- job webhooks, SSE or WebSocket events
- real backend progress percentages
- HTTP byte ranges
- idempotency-key contract
- rate limiting, scopes and tenant separation
- machine-readable OpenAPI

The generic job route provides typed access to many tasks in the meantime, but
it is not a substitute for every ergonomic or upstream-compatible endpoint.

## Planned contract hardening

The API documentation roadmap is:

1. add a checked OpenAPI file;
2. compare registered router operations against OpenAPI in CI;
3. normalize serialized task names;
4. separate a compact public response from optional verbose diagnostics;
5. redact or omit server paths and inline media from persisted/public records;
6. add explicit job/output retention and deletion behavior;
7. add architecture-specific API fixtures for image, video, TTS, ASR and
   generic analysis tasks.

Until then, clients should use capability/parameter discovery, tolerate the
documented task-name forms and treat 202 as queued validation rather than
execution success.
