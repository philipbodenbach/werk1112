# Environment variables

This page lists environment variables read by Werk, its bundled helper, the
install scripts, or the ComfyUI integration. An explicit CLI option takes
precedence where the same setting is exposed as a flag. Paths should point to
the executable or directory described; Werk does not activate Python virtual
environments on behalf of a configured interpreter.

See [Backends, routing and platform support](../backends.md) for supported
runtime combinations and [Models, manifests and the store](../concepts/models-manifests-and-store.md)
for the managed directory layout.

## Core, serving and storage

| Variable | Meaning |
| --- | --- |
| `WERK_HOME` | Managed store root. A global `--model-home` value wins. Runtime-state data is stored beneath `runtime-state/v1` and its private namespace-HMAC key beneath `auth`. See the [store resolution order](../concepts/models-manifests-and-store.md#store-root). |
| `WERK_API_KEY` | Single bearer key for `werk serve`; also the default for `werk runtime` and the ComfyUI client. A configured serve key must not be empty. |
| `WERK_API_KEYS` | TOML key file for `werk serve`. The uninstall scripts also consult it when locating an optional key file to remove. |
| `WERK_API_BODY_LIMIT_BYTES` | Positive request-body limit for chat completions (including inline vision data), audio transcription/translation uploads and generic-job creation. Default: 128 MiB; maximum: 512 MiB. Invalid values use the default. |
| `WERK_OUTPUT_MAX_BYTES` | Output-store retention ceiling. Default: 20 GiB; invalid values use the default. Oldest outputs are removed first. |
| `WERK_OUTPUT_RETENTION_SECONDS` | Maximum output age. Default: 2,592,000 seconds (30 days); invalid values use the default. |
| `WERK_ACCELERATOR_MEMORY_BYTES` | Unsigned byte estimate overriding detected non-CPU accelerator memory for planning. |
| `WERK_MEDIA_ACCELERATOR` | Planner/device override: `cuda`, `rocm`/`hip`, `mps`/`metal`, `mlx`, or `cpu`; other values remain an opaque accelerator label. This is a planning declaration, not a hardware capability check. |

Hugging Face pull authentication is checked in this order: `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, the token saved by Werk, then the Hugging Face CLI
token below `HF_HOME` (or its normal home-directory default).

## Runtime persistence and memory policy

Runtime persistence follows the active **server's** `WERK_HOME`. Setting a
different store on a machine running `werk runtime` does not redirect the
remote server's state. The runtime CLI has an explicit `--url` option rather
than a `WERK_BASE_URL` default; `WERK_API_KEY` supplies its credential.

There is deliberately no environment variable for:

- a caller-supplied runtime-state directory or namespace;
- memory-pressure thresholds, reservation sizes or eviction policy;
- handoff contents or lifetime;
- cross-backend or cross-restart state reuse;
- enabling experimental capabilities process-wide.

`werk serve --persistence` and its granular persistence options are CLI-only
defaults for missing top-level `policy` and `allow_experimental` members on
`POST /werk/v1/prefill`. In addition, the effective reuse default controls the
native prefix-cache flag generated for a local Werk-started vLLM process;
explicit vLLM arguments win. Werk sends no generated flag to remote vLLM: its
model lifetime is externally managed and its APC configuration remains opaque
metadata. The options do not redirect ordinary `/v1` or media requests through
Prefill.

Independent same-process model/pipeline residency is automatic where supported
and is described in the
[execution-lifetime matrix](../concepts/runtime-persistence-and-memory.md#execution-lifetime-and-reuse-matrix).

The current memory thresholds, bounded state catalog, principal isolation and
dry-run-first maintenance rules are implementation safety policy. A backend
must report exact capability status and bounded memory requirements through its
adapter. See [Runtime persistence and memory architecture](../concepts/runtime-persistence-and-memory.md).

## Media and Qwen-TTS

| Variable | Meaning |
| --- | --- |
| `WERK_MEDIA_COMPANION` | Compatible standalone media-companion executable. This has priority over Python discovery. |
| `WERK_MEDIA_PYTHON` | Python interpreter used for the bundled media adapter when no standalone companion is configured. |
| `WERK_MEDIA_COMPANION_SCRIPT` | Explicit Python companion script. This is an advanced override of the adjacent/repository/embedded script discovery. |
| `WERK_MEDIA_PIPELINE_CACHE_SIZE` | Number of loaded models/pipelines retained independently by each resident execution worker: one generic media LRU and, when used, one separate managed-Qwen LRU. Default per worker: `1`; `0` disables both model caches without disabling either worker. |
| `WERK_MEDIA_DEBUG` | `1`, `true`, `yes`, or `on` enables traceback details for unexpected companion errors. |
| `WERK_QWEN_TTS_PYTHON` | Explicit Python interpreter containing the compatible `qwen-tts==0.1.1` installation. It is checked before Werk's managed Qwen-TTS environment. |

During companion requests Werk forces Hugging Face, Transformers, Diffusers
and Datasets offline modes, disables Hugging Face telemetry, and requests
local-only loading. Those injected variables are an execution boundary, not a
promise that an arbitrary third-party pipeline is offline-safe.

Werk also keeps bounded, process-local positive probe and estimate caches so a
successfully validated media configuration does not repeatedly pay the
one-shot preflight cost. These contain capability/estimate data, not model
weights or inference outputs; unavailable results and errors are not cached.
Their safety bound is internal and is not controlled by
`WERK_MEDIA_PIPELINE_CACHE_SIZE`.

## llama.cpp

The matching `werk serve` options override the resource-tuning variables.

| Variable | Meaning |
| --- | --- |
| `WERK_LLAMA_SERVER_CUDA`, `WERK_LLAMA_SERVER_ROCM`, `WERK_LLAMA_SERVER_VULKAN`, `WERK_LLAMA_SERVER_METAL`, `WERK_LLAMA_SERVER_CPU` | Mode-specific `llama-server` executable. |
| `WERK_LLAMA_SERVER` | Generic executable fallback after the mode-specific variable and before managed/PATH discovery. |
| `WERK_LLAMA_CTX` | Context size; `0` asks the model/runtime default where supported. |
| `WERK_LLAMA_BATCH`, `WERK_LLAMA_UBATCH` | Logical batch size and physical micro-batch size. |
| `WERK_LLAMA_GPU_LAYERS`, `WERK_LLAMA_MAIN_GPU` | GPU offload layer count and main GPU index. |
| `WERK_LLAMA_KV_CACHE_TYPE` | KV-cache type: `f16`, `f32`, or `q8-0`. |
| `WERK_LLAMA_FLASH_ATTN`, `WERK_LLAMA_KV_OFFLOAD` | Flash-attention and KV-cache offload switches. |
| `WERK_LLAMA_WARMUP_TOKENS` | Pre-warm token count; `0` disables token pre-warm. |
| `WERK_LLAMA_WARMUP` | Set to a false value (`0`, `false`, `no`, `off`) to disable warm-up on compatibility paths. |
| `WERK_LLAMA_THREADS`, `WERK_LLAMA_THREADS_BATCH` | Generation and batch thread counts. |
| `WERK_LLAMA_LOG` | Truthy value enables child-runtime logging. |
| `WERK_LLAMA_ARGS` | Additional whitespace-split arguments appended to `llama-server`. Advanced use: these can change runtime behavior outside Werk's typed validation. If the final arguments override the private single-slot state configuration, Werk leaves runtime-state capabilities unavailable. |

Managed CUDA builds additionally read `WERK_LLAMA_CUDA_COMPILER`,
`WERK_LLAMA_CUDA_HOST_COMPILER`, and `WERK_LLAMA_CUDA_ARCH`.

Managed ROCm llama.cpp builds read `WERK_LLAMA_ROCM_ARCH` before the upstream
`GPU_TARGETS` value and `WERK_LLAMA_HIP_NO_VMM` before the upstream
`GGML_HIP_NO_VMM` value. When the selected logical ROCm device on a detected
Strix Halo host is `gfx1151`, their defaults are `gfx1151` and enabled
respectively. Werk deliberately does not set llama.cpp's
`GGML_CUDA_ENABLE_UNIFIED_MEMORY` variable on Strix Halo: current upstream
gfx1151 behavior remains experimental, and llama.cpp treats the variable's
presence as enabled even when its value is `0`. An operator can still opt in
by setting the upstream variable explicitly. Werk does not set or recommend
`HSA_OVERRIDE_GFX_VERSION`; the ROCm stack and binary must support the real
`gfx1151` target.

## vLLM, ONNX, MLX and Transformers

| Variable | Meaning |
| --- | --- |
| `WERK_VLLM_HOST` + `WERK_VLLM_PORT` | Together explicitly select a remote vLLM endpoint before local discovery. Werk waits for its `/v1/models` readiness within the health timeout. |
| `WERK_VLLM_MODEL` | Served model ID for a remote vLLM endpoint. When omitted, Werk accepts an exact Werk-model-ID match or the endpoint's only advertised model; ambiguous endpoints fail instead of guessing. |
| `WERK_VLLM_PYTHON` | Python interpreter for a local vLLM installation. |
| `WERK_VLLM_ARGS` | Additional arguments for a Werk-started local vLLM server, parsed as a POSIX shell-word list and appended to its direct process argv. |
| `WERK_VLLM_HEALTH_TIMEOUT_SECONDS` | Positive cold-start readiness timeout for a Werk-supervised local server or explicitly configured remote endpoint. The default is 300 seconds, or 900 seconds on detected DGX Spark and AMD Strix Halo; this does not change request/inference timeouts. |
| `WERK_VLLM_LOG` | Truthy value enables child-runtime logging. |
| `WERK_VLLM_ACCELERATOR=rocm` or `WERK_VLLM_ROCM=1` | Declares that a remote vLLM endpoint is ROCm-backed. This does not verify the remote server. |
| `WERK_ONNX_RUNTIME_CUDA`, `WERK_ONNX_RUNTIME_ROCM`, `WERK_ONNX_RUNTIME_CPU` | Mode-specific `werk-onnx-runner` executable. |
| `WERK_ONNX_RUNTIME` | Generic ONNX runner fallback. |
| `WERK_ONNX_RUNTIME_BUNDLE_CUDA`, `WERK_ONNX_RUNTIME_BUNDLE_ROCM`, `WERK_ONNX_RUNTIME_BUNDLE_CPU` | Mode-specific local bundle used to provision a managed ONNX runner. |
| `WERK_ONNX_RUNTIME_BUNDLE` | Generic ONNX bundle fallback. |
| `WERK_ONNX_GENAI_PYTHON`, `WERK_ONNX_RUNTIME_PYTHON` | Python interpreter fallbacks for the CPU `onnxruntime_genai` path, checked in that order. |
| `WERK_ONNX_GENAI_MODEL_CACHE_SIZE` | Exact ONNX GenAI model/tokenizer entries retained by the Werk-owned resident CPU-fallback worker. Default: `1`; values are clamped to `0..8`; `0` disables this model cache. It does not affect an opaque external ONNX runner. |
| `WERK_ONNX_EXPORTER` | Executable used by `werk artifacts build` before `optimum-cli` or Python module discovery. |
| `WERK_MLX_PYTHON`, `WERK_MLX_MODULE`, `WERK_MLX_GENERATE` | MLX-LM interpreter, module (default `mlx_lm.generate`), and executable fallback. |
| `WERK_MLX_VLM_PYTHON`, `WERK_MLX_VLM_MODULE`, `WERK_MLX_VLM_GENERATE` | MLX-VLM equivalents; the default module is `mlx_vlm`. |
| `WERK_TRANSFORMERS_PYTHON` | Python interpreter containing PyTorch and Transformers for the compatibility backend. |
| `WERK_TRANSFORMERS_DEVICE` | Device override; `auto` chooses CUDA, then MPS, then CPU. |
| `WERK_TRANSFORMERS_DTYPE` | `auto`, `float32`/`fp32`/`f32`, `bfloat16`/`bf16`, or `float16`/`fp16`/`f16`/`half`. |
| `WERK_TRANSFORMERS_MODEL_CACHE_SIZE` | Exact model/tokenizer entries retained by the Werk-owned Transformers compatibility worker. Default: `1`; values are clamped to `0..8`; `0` disables model caching without disabling the worker. |

`WERK_VLLM_ARGS` is a list of arguments, not a shell command. Quoting and
backslash escaping follow POSIX shell-word rules, including quoted empty
arguments, but Werk never starts a shell and performs no command, environment,
tilde or glob expansion. Repeated flags and argument order are preserved. A
malformed list, a trailing unescaped backslash or a non-UTF-8 environment value
fails before a vLLM process is started.

Werk owns `--model`, `--host`, `--port` and `--served-model-name`; both
`--flag value` and `--flag=value` attempts to set one of these through
`WERK_VLLM_ARGS` are rejected. The variable applies only when Werk starts a
local vLLM process. A nonempty value together with `WERK_VLLM_HOST` and
`WERK_VLLM_PORT` is an error: configure launch flags on that remote server and
leave `WERK_VLLM_ARGS` unset or empty in the Werk process.

`WERK_VLLM_ARGS` can configure facilities owned by vLLM itself. Werk reports
`runtime.state.prefix_cache` as `externally_managed` with only the
`automatic_reuse` operation when at least one Werk-started local process is
active, no active process is remote, and every active process's effective
arguments contain the exact `--enable-prefix-caching` flag without
`--no-enable-prefix-caching`. If every active process explicitly disables APC,
the status is `unsupported`; remote, mixed or ambiguous evidence is
`metadata_only`; no active process is `unavailable`. None of these values turns
APC, KV offload, LMCache or expert residency into a named, persistable or
Werk-controlled state operation.

When `werk serve --persistence` supplies the local-vLLM APC default, Werk adds
`--enable-prefix-caching` only if `WERK_VLLM_ARGS` contains neither the enable
nor disable form. `--persistence-reuse disabled` instead supplies
`--no-enable-prefix-caching`. Werk checks the installed server's help for the
generated flag and fails before model-server spawn if that exact runtime does
not advertise it. User-supplied enable/disable arguments remain authoritative.

The Transformers and ONNX GenAI cache sizes control process-local model
residency only. Their generators, prompts and KV state remain request-local;
restarting Werk or a worker makes the next request cold. With a usable runtime,
a positive capacity permits `runtime.model_residency` to report `supported`;
capacity `0` is reported as `unsupported`, and a missing interpreter/runtime is
`unavailable`. The two variables do not affect the separate generic-media or
managed-Qwen caches.

No cache-size environment variable turns a one-shot execution path into a
resident one. The configured MLX/MLX-VLM commands and opaque external ONNX
runners remain per-request processes. Remote vLLM owns its own model lifetime
and receives neither these settings nor Werk's generated local APC argument.

## Installer and uninstaller

| Variable | Meaning |
| --- | --- |
| `WERK_REPO` | GitHub `owner/repository` used by the release installer. Default: `philipbodenbach/werk1112`. |
| `WERK_VERSION` | Release version (`latest`, `vX.Y.Z`, or `X.Y.Z`). Default: `latest`. |
| `WERK_INSTALL_DIR` | Binary install directory. Defaults to `$HOME/.local/bin` on POSIX and `%LOCALAPPDATA%\Programs\Werk1112\bin` on Windows. The uninstall scripts use the same override. |
| `WERK_ADD_TO_PATH` | Windows installer only: `1` permits adding the install directory to the user PATH. |

The uninstall scripts also use `WERK_HOME` and `WERK_API_KEYS` to locate
optional data. They prompt separately before deleting the model store or key
file.

## ComfyUI integration

These variables are read by the Python custom node, not by the Werk server:

| Variable | Default | Meaning |
| --- | --- | --- |
| `WERK_BASE_URL` | `http://127.0.0.1:11434` | Werk server URL. |
| `WERK_API_KEY` | empty | Default client bearer key. |
| `WERK_MAX_IMAGE_PIXELS` | 67,108,864 | Maximum decoded input image pixels. |
| `WERK_MAX_VIDEO_BYTES` | 536,870,912 | Maximum video bytes accepted by the node. |
| `WERK_MAX_AUDIO_BYTES` | 268,435,456 | Maximum returned/downloaded audio bytes. |
| `WERK_MAX_AUDIO_INPUT_BYTES` | 67,108,864 | Maximum input audio bytes before Base64 encoding. |
| `WERK_MAX_VISION_INPUT_BYTES` | 67,108,864 | Maximum aggregate PNG bytes embedded by the ComfyUI vision node before Base64 encoding. |

ComfyUI labels such as `WERK_CONNECTION`, `WERK_ROUTING_CONFIG`,
`WERK_IMAGE_CONFIG`, `WERK_VISION_CONFIG`, `WERK_VIDEO_CONFIG`,
`WERK_AUDIO_CONFIG`, `WERK_RUNTIME_INFO`, `WERK_PERSISTENCE_POLICY`, and
`WERK_STATE_HANDOFF` are socket type names, not environment variables.

## Recognized host and toolchain variables

Werk also observes standard host/toolchain variables when locating runtimes or
building managed backends:

- `PATH` for executable discovery;
- `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` for CUDA visibility;
- `ROCR_VISIBLE_DEVICES` for ROCm visibility;
- `CUDA_HOME`, `CUDA_PATH`, `CUDA_ROOT`, `CUDA_TOOLKIT_ROOT_DIR`,
  `CMAKE_CUDA_COMPILER`, `CUDA_NVCC`, and `CUDAARCHS` for CUDA toolchain
  discovery;
- `NCCL_HOME`, `NCCL_ROOT`, and `LD_LIBRARY_PATH` for NCCL/native-library
  checks;
- `WSL_DISTRO_NAME` and `WSL_INTEROP` for WSL detection.

`CUDA_COMPUTE_CAP` is also set by target release aliases where required; the
DGX Spark alias forces `121`. Custom source builds and the checked-in
target-specific `CC_*`/`CXX_*` settings are documented in the
[build guide](../development/build.md).

## Source of truth

- CLI-bound settings: [`src/cli.rs`](https://github.com/philipbodenbach/werk1112/blob/main/src/cli.rs)
- serving and retention: [`src/api/router.rs`](https://github.com/philipbodenbach/werk1112/blob/main/src/api/router.rs) and
  [`src/inference_service`](https://github.com/philipbodenbach/werk1112/tree/main/src/inference_service)
- backend discovery: [`src/backend`](https://github.com/philipbodenbach/werk1112/tree/main/src/backend)
- runtime persistence, memory and principal isolation: [`src/runtime_control`](https://github.com/philipbodenbach/werk1112/tree/main/src/runtime_control)
- versioned runtime DTOs and client: [`src/werk_protocol`](https://github.com/philipbodenbach/werk1112/tree/main/src/werk_protocol)
- media companion: [`src/media_companion.rs`](https://github.com/philipbodenbach/werk1112/blob/main/src/media_companion.rs) and
  [`runtime/werk_media_companion.py`](https://github.com/philipbodenbach/werk1112/blob/main/runtime/werk_media_companion.py)
- store and authentication: [`src/model_store.rs`](https://github.com/philipbodenbach/werk1112/blob/main/src/model_store.rs)
- installers: [`scripts`](https://github.com/philipbodenbach/werk1112/tree/main/scripts)
- ComfyUI defaults: [`utils/comfyUI/config.py`](https://github.com/philipbodenbach/werk1112/blob/main/utils/comfyUI/config.py)
