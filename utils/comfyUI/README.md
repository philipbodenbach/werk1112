# ComfyUI Werk1112 nodes

This package connects native ComfyUI nodes to `werk serve` over HTTP. ComfyUI
still owns its workflow graph, queue, history, previews, and saved images; Werk
performs model discovery, routing, and inference.

## Existing ComfyUI compatibility

Werk already supports ComfyUI's built-in hosted OpenAI image node at
`POST /proxy/openai/images/generations`. These custom nodes are an additional,
Werk-native integration with model and available-task discovery, live parameter
schema discovery, Werk routing options, complete inference metadata, and
authenticated Werk URL outputs.

Werk does not implement ComfyUI's `/prompt`, `/history`, `/view`, WebSocket,
queue, custom-node, or workflow-graph protocols.

## Installation

### ComfyUI Manager (recommended)

The release package uses the immutable Comfy Registry ID `werk1112`. Once its
first Registry version has been published, search for **WERK1112** in ComfyUI
Manager and select **Install**. The equivalent Comfy CLI command is:

```bash
comfy node install werk1112
```

Manager installs the package dependencies in the correct ComfyUI Python
environment and handles future updates. Restart ComfyUI after installation.

Until the first Registry release, or for repository development, use the
manual method below.

### Manual development installation

Copy the package from a Werk1112 checkout into ComfyUI and install only its
small image dependencies in the Python environment used by ComfyUI:

```bash
cp -R utils/comfyUI /path/to/ComfyUI/custom_nodes/comfyui-werk1112
cd /path/to/ComfyUI
python -m pip install -r custom_nodes/comfyui-werk1112/requirements.txt
```

For repository development on Linux or macOS, use a symlink instead:

```bash
ln -s "$(pwd)/utils/comfyUI" \
  /path/to/ComfyUI/custom_nodes/comfyui-werk1112
```

PowerShell junction:

```powershell
New-Item -ItemType Junction `
  -Path "C:\path\to\ComfyUI\custom_nodes\comfyui-werk1112" `
  -Target (Resolve-Path ".\utils\comfyUI")
```

Restart ComfyUI. The nodes appear in the `WERK` categories. The package does
not pin, replace, or install Torch.

Do not use `pip install .` as the node installation method. ComfyUI must load
the complete node directory, including its frontend extension, from
`custom_nodes`; Registry/Manager installation or the directory copy/symlink
above provides that layout.

## Starting Werk

For a fresh API-key file:

```bash
werk auth api-key generate --name comfyUI
werk serve --image-model tiny-sd
```

`generate --name` does not currently append to an existing key file. If a key
file already exists, generate the ComfyUI key separately and copy its `[[keys]]`
block into `~/.config/werk1112/api-keys.toml`:

```bash
werk auth api-key generate --name comfyUI --path /tmp/comfyui-key.toml
```

For local unauthenticated development only:

```bash
werk serve --image-model tiny-sd --allow-unauthenticated
```

The normal address is `http://127.0.0.1:11434`. Put the address and key into a
**WERK Connection** node, or set `WERK_BASE_URL` and `WERK_API_KEY` before
starting ComfyUI. Explicit widget values take precedence over environment
defaults.

## Nodes

- **WERK Connection** creates one reusable, credential-redacting connection.
  Click **Verify Connection** to perform a real request and show a visible
  success or error status directly on the node.
- **WERK Server Info** reports installed models and server capabilities.
- **WERK Image Models** distinguishes declared image support from currently
  runtime probe-eligible image support. Probe eligibility is not a guarantee
  that a complete model pipeline will load successfully. Connect a
  **WERK Connection**, click **Refresh Models**, and select a model from the
  **available_model** dropdown.
- **WERK Image Parameters** reads the active model/runtime parameter schema.
- **WERK Routing Config** represents all current Werk routing overrides without
  turning inherited server, model, or profile defaults into explicit request
  values.
- **WERK Image Config** contains the common image controls and accepts the
  remaining model-specific image parameters as schema-driven JSON. It can
  consume a **WERK Routing Config**.
- **WERK Image Generate** is the compact generator. Its `model` socket
  is a real, required ComfyUI input and should be connected to the `model`
  output of **WERK Image Models**. It returns a ComfyUI `IMAGE` batch plus
  sanitized Werk metadata and IDs.

The interactive verification and dropdown discovery run through a local
ComfyUI route. The browser sends the connection settings only to its own
ComfyUI backend; the backend contacts Werk. API keys are never included in the
route response.

### Recommended image workflow

Use the nodes in this shape:

```text
WERK Connection.connection --------+--> WERK Image Models.connection
                                    +--> WERK Image Generate.connection

WERK Image Models.model ----------------> WERK Image Generate.model
WERK Routing Config.routing ------------> WERK Image Config.routing
WERK Image Config.config ---------------> WERK Image Generate.config
WERK Image Generate.images -------------> Preview Image.images
```

Connect the singular `model` output, not `available_models`. The latter is a
newline-separated diagnostic list. The generator deliberately has no model
widget or private model dropdown: its `model` input uses ComfyUI's
`forceInput`, so the selected model remains explicit in the graph.

`config` is optional. An unconnected config uses the Image Config defaults:
1024x1024, one image, batch size 1, 28 steps, guidance 7, seed 0, PNG, and an
embedded Base64 response. The default batch size is treated as inherited and
is not sent explicitly. For a reproducible and visible workflow, connecting an
explicit **WERK Image Config** is recommended.

When these nodes call `werk serve`, the bundled media execution worker stays
running and serializes generation requests. Health, discovery, and estimation
preflights remain independent of that queue. Its Diffusers image/video cache holds one
fully configured pipeline by default: the first generation is a cold load, and
later generations with the same model/runtime configuration should be
substantially faster to start. Changing prompt, seed, dimensions, steps, or
count keeps the pipeline warm. Changing model, device, dtype, offload/tiling
settings, or LoRAs may reload it; the previous entry is evicted before the new
one is loaded at the default cache size. Set
`WERK_MEDIA_PIPELINE_CACHE_SIZE=0` before starting Werk to disable pipeline
caching, or set a larger non-negative entry count when system memory permits.
Resident entries retain VRAM and/or RAM until eviction or Werk shuts down.

Werk metadata exposes `model_cache_hit` and `model_load_seconds` to distinguish
warm and cold runs. A worker crash or inference timeout clears the resident
state, so the next run is cold. The resident transport never replays the same
`execute` frame; Werk's higher-level fallback policy may still try another
accepted runtime candidate. A legacy external media companion falls back to
one-shot execution when it does not support the persistent protocol.

### WERK Routing Config

The routing node covers all 17 fields in Werk's current routing schema:

| Node input | Meaning and inheritance |
| --- | --- |
| `backend` | Runtime backend; blank inherits |
| `accelerator` | Requested accelerator; blank inherits |
| `device` | Concrete device identifier; blank inherits |
| `precision` | Compute precision; blank inherits |
| `quantization` | Weight quantization; blank inherits |
| `profile` | Saved Werk parameter profile; blank inherits |
| `quality` | `inherit`, `draft`, `balanced`, `high`, or `maximum` |
| `performance_preference` | `inherit`, `quality`, `balanced`, `speed`, `latency`, `throughput`, or `memory` |
| `fallback_policy` | `inherit`, `none`, `backend`, or `degrade` |
| `parameter_policy` | `inherit`, `strict`, `warn`, or `permissive` |
| `allow_cpu_offload` | Tri-state: `inherit`, `enabled`, or `disabled` |
| `allow_sequential_offload` | Tri-state: `inherit`, `enabled`, or `disabled` |
| `allow_component_offload` | Tri-state: `inherit`, `enabled`, or `disabled` |
| `allow_disk_offload` | Tri-state: `inherit`, `enabled`, or `disabled` |
| `attention_backend` | Attention implementation; blank inherits |
| `compile` | Tri-state: `inherit`, `enabled`, or `disabled` |
| `inference_timeout_seconds` | `0` inherits; a positive value sends Werk's `timeout_seconds` override |

`inherit` and blank values are omitted from the request. In particular,
`inherit` is different from `disabled`: the latter sends an explicit JSON
`false`. The `config_json` output shows the exact normalized options and is
useful when diagnosing a workflow.

`additional_routing_parameters_json` is a forward-compatible escape hatch for
canonical `routing.*` parameters introduced by newer Werk versions. The 17
fields above must use their dedicated controls. Cross-namespace keys,
transport fields, credentials, and duplicate dedicated controls are rejected.

### WERK Image Config

The image config exposes the common controls directly:

| Node input | Werk request |
| --- | --- |
| `width`, `height` | `size: "WIDTHxHEIGHT"` |
| `count` | `n`; the normal, adapter-independent image count |
| `batch_size` | `parameters["image.batch_size"]` only when greater than 1 |
| `steps` | `parameters["image.steps"]` |
| `guidance` | `parameters["image.guidance"]` |
| `seed` | `parameters["image.seed"]` |
| `output_format` | `output_format` |
| `response_format` | `response_format` |
| `style` except `none` | OpenAI-compatible prompt style hint |
| `vae_tiling` | Tri-state `parameters["image.vae_tiling"]` |
| `vae_slicing` | Tri-state `parameters["image.vae_slicing"]` |

Use `count` for normal multi-image generation. `batch_size=1` is not included
in the request, so Werk and the selected adapter retain their resolved default.
A `batch_size` greater than 1 is an alternative, adapter-dependent way to
control how many images are produced and therefore requires `count=1`. If both
`count` and `batch_size` are greater than 1, the node rejects the config locally
instead of sending an ambiguous request. This also avoids the Diffusers
conflict caused by combining OpenAI-compatible `n` with an explicit image batch
size.

Werk's current image-generation schema contains 80 image parameter
descriptors. The common subset above has native widgets; every remaining
geometry, sampling, inpainting, conditioning, adapter, high-resolution, and
post-processing parameter is available through
`additional_image_parameters_json`. Discover the exact model defaults,
constraints, allowed values, and runtime support with **WERK Image Parameters**
or:

```bash
werk parameters MODEL --task image-generation --json
```

Additional image parameters may be unqualified, canonical, or grouped under
`image`. These examples are equivalent where they name the same fields:

```json
{
  "sampler": "euler",
  "image.scheduler": "karras",
  "image": {
    "guidance_rescale": 0.2,
    "high_resolution_fix": true
  }
}
```

List parameters accept either a normal JSON array or Werk's list-override
shape. For example, this appends a LoRA to inherited model/profile defaults:

```json
{
  "image.loras": {
    "operation": "add",
    "values": [
      {"model": "style.safetensors", "weight": 0.65}
    ]
  }
}
```

Supported list operations are `inherit`, `replace`, `add`, and `clear`.
Omitting a parameter is the normal way to inherit its resolved Werk value.
Routing/transport keys and duplicates of dedicated controls are rejected
instead of being silently overwritten. The `config_json` output displays the
fully normalized config before generation.

### FLUX.2 Klein on limited VRAM

FLUX.2 Klein may need offloading even at a modest output resolution because
model weights dominate the memory estimate. Start with this configuration:

- In **WERK Routing Config**, set `allow_cpu_offload` to `enabled`.
- Leave `allow_sequential_offload` and `allow_component_offload` at `inherit`.
- In **WERK Image Config**, start at 512x512, set guidance around 3.5, and set
  `vae_tiling` to `enabled`.

If model CPU offload still does not fit, set
`allow_sequential_offload` to `enabled`. Sequential offload takes precedence
over the other CPU/component modes and uses less accelerator memory, but is
substantially slower. Offloading requires a CUDA-style accelerator and enough
host RAM. A model being listed as runtime-available means a suitable adapter
was discovered; it does not guarantee that the full pipeline fits into VRAM
without routing overrides.

The node accepts embedded Base64 or authenticated URL responses. RGBA images
are composited over white, all images become float32 RGB `[B,H,W,C]` tensors,
and differently sized outputs fail rather than being resized.

## Discovery and diagnostics

CLI equivalents:

```bash
werk inspect MODEL
werk parameters MODEL --task image-generation --json
werk doctor --model MODEL --task image-generation
```

The nodes call:

```text
GET /v1/models
GET /v1/capabilities
GET /v1/parameters?task=image-generation&model=MODEL&backend=auto
POST /v1/images/generations
GET /v1/outputs/{id}
```

## WSL and containers

`127.0.0.1` always refers to the environment in which ComfyUI runs:

- Windows ComfyUI calling Werk in WSL should use the WSL address reachable
  from Windows.
- WSL ComfyUI calling Windows Werk should use the Windows host address exposed
  to WSL.
- Docker ComfyUI commonly reaches a host Werk instance at
  `http://host.docker.internal:11434`; Linux Docker may require an explicit
  host-gateway mapping.
- Bind Werk with `--host 0.0.0.0` only when cross-environment access requires
  it. Restrict the firewall and keep API-key authentication enabled.

## Security

An API key typed directly into a node can be serialized into workflow JSON.
Prefer `WERK_API_KEY` for workflows that will be shared, and remove secrets
before export. Use HTTPS across untrusted networks. Never publicly expose
`--allow-unauthenticated`.

The client sends bearer credentials only to the exact Werk origin (matching
scheme, host, and effective port), rejects cross-origin redirects, bounds
downloads, and removes local filesystem paths from visible result metadata.
Werk URL outputs remain subject to Werk's output-retention policy.

Image decoding defaults to a 67,108,864-pixel allocation limit. Set
`WERK_MAX_IMAGE_PIXELS` before starting ComfyUI to choose another positive
limit.

## Tests

From the repository root:

```bash
python -m pytest -q utils/comfyUI/tests
```

Tests require no GPU, ComfyUI installation, model, or live Werk server.

## Publishing to the Comfy Registry

The node is validated and packaged by
`.github/workflows/comfyui-registry.yml` whenever this directory changes. A
push or pull request never publishes a release. Publication is an explicit
manual action and is allowed only from the repository's default branch.

Before the first publication:

1. Create or claim the `philipbodenbach` publisher in the Comfy Registry. If
   that publisher ID cannot be used, update `PublisherId` in `pyproject.toml`
   before publishing anything.
2. Create a Registry publishing API key for that publisher.
3. Add it to this GitHub repository as the Actions secret
   `REGISTRY_ACCESS_TOKEN`.
4. Confirm that the immutable node ID `werk1112` is available. Change
   `project.name` before the first publication if another ID is required.
5. Increment the semantic `project.version`, merge the change to the default
   branch, then run **ComfyUI Registry** from GitHub Actions. Optional release
   notes can be supplied through the workflow's `changelog` input.

Published Registry versions cannot be overwritten. Fixes require another
version increment. The workflow deliberately runs the Comfy CLI from this
directory so the Registry archive contains the node package rather than the
entire Werk1112 monorepo.
