# AUTOMATIC1111-compatible clients

Werk1112 implements the small AUTOMATIC1111 API subset needed by Open WebUI and
basic txt2img clients. It translates useful generation fields into Werk's
inference router; it does not run A1111 extensions, scripts or model code.

## Routes

~~~text
POST /sdapi/v1/txt2img
GET  /sdapi/v1/sd-models
GET  /sdapi/v1/options
POST /sdapi/v1/options
GET  /sdapi/v1/progress
~~~

Use the Werk server root, without `/v1`:

~~~text
http://127.0.0.1:11434
~~~

## Authentication

A1111 routes accept the normal Werk credentials:

- `Authorization: Bearer <Werk API key>`
- `X-API-Key: <Werk API key>`

They additionally accept HTTP Basic authentication with the exact username
`werk` and the Werk key as password:

~~~bash
curl -u "werk:$WERK_API_KEY" \
  http://127.0.0.1:11434/sdapi/v1/sd-models
~~~

## Generate an image

~~~bash
curl -fsS http://127.0.0.1:11434/sdapi/v1/txt2img \
  -u "werk:$WERK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A tiny red workshop robot repairing a wooden chair",
    "negative_prompt": "broken geometry",
    "width": 512,
    "height": 512,
    "steps": 8,
    "cfg_scale": 1,
    "seed": 0
  }' | jq -r '.images[0]' | base64 --decode > a1111-robot.png
~~~

The response follows the A1111 txt2img shape:

- `images` contains Base64 images when `send_images` is true;
- `parameters` contains the normalized compatibility request;
- `info` is a JSON-encoded string containing effective dimensions, steps,
  guidance, model, seeds and any `werk_warnings`.

Temporary Werk outputs are removed after the response is prepared. Setting
`send_images: false` therefore returns no retained output. `save_images: true`
is rejected rather than pretending that Werk wrote into an A1111 output tree.

## Field mapping

| A1111 field | Werk behavior |
| --- | --- |
| `prompt` | Positive image prompt |
| `negative_prompt` | Negative image prompt |
| `width`, `height` | Image dimensions |
| `steps` | Denoising steps |
| `cfg_scale` | Guidance |
| `seed` | Image seed; `-1` generates a random non-negative seed |
| `batch_size * n_iter` | Requested image count |
| `override_settings.sd_model_checkpoint` | Per-request installed Werk image model |

`sampler_name`, `sampler_index`, a non-automatic `scheduler`, and nonzero `eta`
may be accepted for protocol compatibility, but Werk's selected runtime remains
authoritative and reports a warning when it does not apply the requested value.

Meaningful unsupported behavior is rejected. This includes:

- high-resolution fix;
- face restoration;
- seamless tiling;
- named prompt styles;
- refiners;
- scripts and `alwayson_scripts`;
- nondefault subseed behavior and seed resizing;
- server-side image saving;
- `override_settings_restore_afterwards: false`;
- img2img and extension-specific endpoints.

Unknown extra fields are accepted only when their values are harmless defaults;
nondefault unsupported fields fail explicitly.

## Models and options

`GET /sdapi/v1/sd-models` lists installed models that declare
`image-generation`. Server filesystem paths and model hashes are not exposed.

`GET /sdapi/v1/options` returns the selected `sd_model_checkpoint` and
`samples_format: "png"`. Select a process-wide model with:

~~~bash
curl -fsS -X POST http://127.0.0.1:11434/sdapi/v1/options \
  -u "werk:$WERK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"sd_model_checkpoint":"IMAGE_MODEL"}'
~~~

The selected model must be installed and declare image generation. This
selection is process state and is not a durable model-store setting. A
per-request `override_settings.sd_model_checkpoint` takes precedence.

Without an explicit selection, Werk uses the configured `--image-model`, then
an image-capable default `--model`, or the sole installed image-generation
model. Multiple candidates require an explicit choice.

## Progress semantics

Werk serializes A1111 txt2img calls. While one is active,
`GET /sdapi/v1/progress` reports a coarse active state with `progress: 0.01` and
an explanation that step-level progress is unavailable. When idle it reports
zero. It does not fabricate an ETA, current preview image or sampling step.

The complete route inventory and common Werk security behavior are in the
[HTTP API reference](../api.md).
