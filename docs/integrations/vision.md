# Vision and visual quality assurance

Werk can send text and one or more images to a local vision-language model
through the normal chat contract. A practical use is the final inspection of a
rendered HTML page, PDF page or slide: the model sees the pixels that a source
code test cannot see and can report clipped text, missing controls, alignment
errors, overlap and inconsistent spacing.

Vision is a capability of the selected model and its model-specific processor,
not a feature supplied by vLLM. Werk routes the same multimodal request to an
eligible installed runtime. vLLM is one optional execution adapter; it is not
required when a compatible llama.cpp GGUF plus projector is available.

## Current execution routes

| Route | Model/layout requirements | Notes |
| --- | --- | --- |
| llama.cpp server | Compatible VLM GGUF plus exactly one manifest-listed multimodal projector GGUF whose filename contains `mmproj` or `projector` | Primary non-vLLM route. Werk starts the persistent server with `--mmproj`. The installed `llama-server` must advertise that option and support the concrete VLM. |
| vLLM | Transformers safetensors model with exact architecture metadata `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`, `qwen3_vl_moe`, `glm4v` or `glm4v_moe` | Optional local or configured remote route. Compatibility still depends on the installed vLLM release, model implementation and accelerator. |
| MLX-VLM | MLX or Hugging Face-style safetensors repository classified as `gemma4_unified` | Existing Apple-Silicon route through `mlx-vlm`. It accepts images but does not preserve the API `detail` hint or arbitrary text/image interleaving. |

The in-process Candle adapter remains text-only. Importing a Qwen3-VL or
GLM-4V repository, or finding a similarly named architecture in a dependency,
does not provide native Candle vision execution. Werk also does not claim a
general LM Studio adapter in this matrix.

The manifest must advertise the `image-understanding` task and image input.
Werk rejects an image request against a text-only manifest before treating a
runtime as eligible. Check classification and routing without performing a
large inference:

~~~bash
werk inspect VISUAL_QA_MODEL
werk doctor --model VISUAL_QA_MODEL --task image-understanding --debug
werk backend doctor --debug
~~~

For a multi-file GGUF repository, selecting the main model during `werk pull`
also retains the preferred recognized projector. Inspect the resulting
manifest before inference. An absent projector, more than one ambiguous
projector or an old `llama-server` without `--mmproj` produces an explicit
error instead of silently running text-only.

## CLI visual inspection

The one-shot `run` command accepts repeatable `--image` values. With automatic
routing:

~~~bash
werk --backend auto run VISUAL_QA_MODEL \
  "Inspect the rendered slide. Report missing or clipped text, overlapping elements, broken alignment, inconsistent margins, and controls that appear absent. Refer only to what is visible." \
  --image /absolute/path/to/rendered-slide.png \
  --max-tokens 768 \
  --debug
~~~

`werk chat VISUAL_QA_MODEL --image PATH` provides the same input in an
interactive session. An image path or URL may be repeated for multi-page or
before/after comparison. Werk reads local CLI paths and converts them to inline
data URLs before routing, so a remote runtime does not need access to the
client filesystem. Existing data URLs and HTTP(S) URLs remain URL inputs.

## HTTP request

Start the service with the VLM as its chat model. Backend selection is a server
configuration, not an OpenAI chat body field:

~~~bash
werk --backend auto serve --model VISUAL_QA_MODEL --verbose
~~~

`POST /v1/chat/completions` accepts ordered content parts. This Linux/macOS
example embeds a local PNG as a data URL:

~~~bash
SCREENSHOT_B64="$(base64 < rendered-slide.png | tr -d '\n')"

curl -fsS http://127.0.0.1:11434/v1/chat/completions \
  -H "Authorization: Bearer $WERK_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @- <<JSON
{
  "model": "VISUAL_QA_MODEL",
  "messages": [
    {
      "role": "system",
      "content": "You are a strict visual QA reviewer. Do not infer elements from source code; report only what is visible in the supplied render."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Inspect this slide for missing controls, clipped or overflowing text, overlap, grid misalignment, inconsistent spacing and insufficient contrast. Give each finding a severity and approximate screen location."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,${SCREENSHOT_B64}",
            "detail": "high"
          }
        },
        {
          "type": "text",
          "text": "If no defect is visible, say so explicitly and do not invent one."
        }
      ]
    }
  ],
  "max_completion_tokens": 768,
  "temperature": 0.1
}
JSON
~~~

Werk also accepts `type: "input_image"` as an input-part alias. Its image
source is still supplied in the part's `image_url` field, either as a URL
string or as an object with `url` and optional `detail`.

The API preserves the order of text and image parts. The llama.cpp and vLLM
adapters forward that multipart structure, including `detail`, to their chat
endpoint. `detail` is a processing hint, not a cross-model quality guarantee;
the selected runtime and processor decide its exact effect. The llama.cpp and
vLLM routes forward data and HTTP(S) URLs. MLX-VLM safely stages inline data
URLs as temporary native image files; HTTP(S) fetching remains the selected
runtime's responsibility. Client-local paths and `file:` URLs are not portable
API inputs and are rejected by the HTTP-backed llama.cpp and vLLM routes.

## Routing, persistence and caches

With `--backend auto`, the model manifest, image input, format, layout,
architecture, accelerator and successful runtime probe all participate in the
decision. A text-only runtime is not an eligible fallback for the same image
request. Use `--debug` to see every accepted or rejected candidate.

Vision requests remain request-aware so a previously selected text session
cannot pin them to the wrong adapter. This does not reload model weights for
every image: persistent llama.cpp and vLLM server processes remain persistent
and retain the caching behavior supplied by those runtimes. Werk does not
promise identical KV-cache, prefix-cache or image-embedding-cache semantics
across backends, and changing an image should be treated as a new multimodal
prompt.

To force a route for a reproducibility test, select it when starting Werk, for
example `werk --backend cuda serve ...` for a CUDA GGUF route or
`werk --backend vllm serve ...` for vLLM. The OpenAI-compatible chat request does not
accept Werk media `routing`, `backend` or `accelerator` fields; unknown chat
fields may be ignored.

## Size, fidelity and operational limits

`POST /v1/chat/completions` uses the configured API body limit. The default is
128 MiB and `WERK_API_BODY_LIMIT_BYTES` can raise it to at most 512 MiB. The
limit applies to the complete JSON body, and Base64 increases the encoded size
by roughly one third.

For visual QA:

- inspect the final browser/slide render rather than a thumbnail;
- retain enough resolution to make typography and one-pixel alignment visible;
- crop or tile unusually large canvases instead of asking the model to reason
  over illegible downscaling;
- use `detail: "high"` where the runtime implements the hint;
- keep deterministic conventional tests for structure and use vision as an
  additional rendered-output check, because a VLM finding is probabilistic;
- ask for locations and observable evidence, then map findings back to DOM,
  slide or document coordinates in the calling pipeline.

The configured request-body limit is a transport bound, not a model context or
vision-resolution guarantee. The model processor may impose stricter image
dimensions, pixel counts or token budgets. Audio and video message parts are
not supported by the chat endpoint.
