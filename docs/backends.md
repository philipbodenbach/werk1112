# Backends, routing and platform support

Werk1112 separates the stable inference contract from concrete execution
runtimes. A model can be installed and classified without every backend being
present, and one model may have several eligible runtime candidates.

This page documents the current implementation. “Installer exists” does not
mean “every model and accelerator combination is verified.”

Inference eligibility is separate from runtime-state control. The exact
prefix-state, persistence, memory, prefill/decode and expert-residency statuses
for each active adapter are in the
[runtime-control capability matrix](concepts/runtime-persistence-and-memory.md#current-production-capability-matrix).

## Three separate questions

Backend troubleshooting is easier when these questions are kept separate:

1. **Can Werk discover the runtime?**
2. **Can the runtime accept this model layout, architecture, task and explicit parameters?**
3. **Can it load and execute the model on this machine?**

A successful install answers only the first question. Model probing and
planning answer most of the second. The first cold inference is still the
definitive load and memory test.

## Runtime selection

For a typed request Werk:

1. loads the model manifest;
2. resolves task and routing parameters;
3. estimates accelerator and host-memory demand;
4. probes registered runtimes;
5. rejects candidates that do not match task, format, layout, architecture,
   accelerator or strict explicit parameters;
6. scores the remaining candidates;
7. executes the highest-scoring candidate;
8. optionally retries another already accepted candidate according to the
   fallback policy.

Use diagnostics before a large request:

~~~bash
werk inspect MODEL
werk doctor --model MODEL --task TASK --debug
werk backend list
werk backend doctor --debug
~~~

Qwen-TTS is currently an architecture adapter behind the media companion, not
a separately rendered runtime row. Its managed environment can be installed,
but <code>backend list</code> and <code>backend doctor</code> do not yet print a
complete Qwen-specific status block. A model-specific doctor run and companion
diagnostics report the missing or selected Qwen interpreter.

The media commands additionally expose the resolved request and complete
candidate decision:

~~~bash
werk audio generate speech TTS_MODEL \
  --text "Diagnostic test." \
  --backend auto --verbose --debug
~~~

## Automatic and explicit selection

The default backend value is <code>auto</code>. It does not mean that an
arbitrary installed runtime will be tried. Only candidates accepted for the
specific task and model participate.

A concrete backend increases the score of matching candidates. For typed media
requests it becomes a hard backend constraint when
<code>fallback_policy=none</code>. A concrete accelerator or device remains a
hard target constraint independently of backend retry.

For a strict reproducibility test use all relevant constraints:

~~~bash
werk video generate VIDEO_MODEL \
  --prompt "A short camera movement" \
  --backend auto \
  --accelerator cuda \
  --fallback-policy none \
  --precision bf16 \
  --verbose --debug
~~~

## vLLM launch arguments and tool calling

`WERK_VLLM_ARGS` supplies advanced arguments only to a vLLM process that Werk
starts locally. It uses POSIX shell-word quoting to construct a direct process
argument vector; it is not evaluated by a shell. Command substitution,
environment-variable expansion, tilde expansion and globbing therefore never
occur. Malformed quoting, a trailing unescaped backslash and non-UTF-8 values
fail before process creation.

For example, after importing or pulling a compatible model as
`qwen3-coder`, a local launch can be configured as follows. The global backend
option belongs before the `serve` subcommand:

~~~bash
export WERK_API_KEY="replace-with-generated-key"
export WERK_VLLM_ARGS="--quantization compressed-tensors --kv-cache-dtype fp8 --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}' --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-num-seqs 16"

werk --backend vllm serve --model qwen3-coder
~~~

The exact parser name and flags are examples for a compatible vLLM/model
combination, not Werk defaults. Verify them against the installed vLLM version
and the selected model. vLLM generally requires `--enable-auto-tool-choice`
when `tool_choice` is `auto`, together with a compatible tool-call parser. Werk
does not validate, replace or rewrite arbitrary parser names, and does not add
that enable flag automatically.

Conceptually, Werk's effective local child argv is one of these forms, with a
resolved model directory and an internally selected loopback port:

~~~text
$WERK_VLLM_PYTHON -m vllm.entrypoints.openai.api_server --model RESOLVED_MODEL_DIR --host 127.0.0.1 --port INTERNAL_PORT --served-model-name qwen3-coder [WERK_VLLM_ARGS...]
vllm serve RESOLVED_MODEL_DIR --host 127.0.0.1 --port INTERNAL_PORT --served-model-name qwen3-coder [WERK_VLLM_ARGS...]
~~~

Werk owns `--model`, `--host`, `--port` and `--served-model-name`. Supplying
any of them in separate or `--flag=value` form through `WERK_VLLM_ARGS` is an
error. Repeated non-reserved flags, JSON values, embedded spaces and quoted
empty arguments remain distinct argv elements and retain their order.

For an already running remote vLLM endpoint, configure these process arguments
where that server is launched. A nonempty `WERK_VLLM_ARGS` is rejected when
Werk uses `WERK_VLLM_HOST` and `WERK_VLLM_PORT`; it is never silently ignored.

`POST /v1/chat/completions` supports OpenAI function-tool requests through the
vLLM adapter, for both Werk-started and remote vLLM servers. Werk forwards the
tool definitions, `tool_choice`, `parallel_tool_calls`, assistant tool calls and
tool-result messages to vLLM without translating their contents. It likewise
preserves structured tool calls in normal and streaming responses. Werk does
not execute tools or select a vLLM tool parser for the operator.

Other production chat adapters explicitly reject a request that requires tool
calling with HTTP 400 and error code `unsupported_tool_calling`. Automatic
routing treats tool calling as a required backend capability and cannot send
such a request to an incompatible runtime. An explicit `--backend vllm` route
is strict and never falls back to a non-vLLM backend. Merely setting
`WERK_VLLM_ARGS` does not activate or prefer vLLM.

These guarantees cover Werk's argv construction and HTTP transport. Actual
tool-call quality, model support, vLLM-version compatibility, quantization,
speculative decoding and accelerator compatibility still require a live
runtime test before relying on a combination in production.

## Vision-language routing

An image attached to `werk run`, `werk chat` or
`POST /v1/chat/completions` changes runtime eligibility. The model manifest must
advertise image input and `image-understanding`; a text-only model is rejected
even if an installed backend can execute some other VLM.

| Runtime route | Current eligible model shape | Additional requirement |
| --- | --- | --- |
| Persistent llama.cpp server | Compatible VLM GGUF | Exactly one safe local projector GGUF listed in the manifest; its filename contains `mmproj` or `projector`, and `llama-server --help` advertises `--mmproj` |
| vLLM | Transformers safetensors with exact architecture `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`, `qwen3_vl_moe`, `glm4v` or `glm4v_moe` | Compatible installed vLLM version and model-specific processor; local or explicitly configured remote endpoint |
| MLX-VLM | MLX or safetensors `gemma4_unified` repository | Importable `mlx-vlm` environment on Apple Silicon |
| Candle | None | The in-process Candle adapter is currently text-only |

vLLM is optional and is not what makes a model visual. The vision encoder,
projector and preprocessing belong to the checkpoint/runtime implementation.
The primary non-vLLM path is llama.cpp plus the model's matching multimodal
projector. Model weights remain resident in persistent llama.cpp/vLLM server
processes, although cache details and image-embedding reuse remain
backend-specific.

Werk preserves ordered text/image parts and the image `detail` hint for the
llama.cpp and vLLM chat transports. The MLX-VLM subprocess currently receives
the prompt and image list but not arbitrary interleaving or `detail` semantics.
See [Vision and visual quality assurance](integrations/vision.md) for API and
CLI examples, body limits and the rendered HTML/slide inspection workflow.

## Fallback policies

| Policy | Candidate behavior | Quality or memory adjustment |
| --- | --- | --- |
| <code>none</code> | Execute only the selected runtime. A requested backend is a hard filter. | No inherited degradation. Explicitly requested offload still remains explicit. |
| <code>backend</code> | Default. Retry another already accepted runtime if execution or model loading fails. | No automatic inherited degradation. |
| <code>degrade</code> | Retry accepted runtimes and permit registered memory-saving adjustments. | May enable allowed offload or media tiling/windowing under memory pressure. |

Fallback never silently changes the model ID. Suggested lower resolutions,
shorter clips or smaller models are diagnostic recommendations, not automatic
request rewrites.

An unavailable architecture-specific runtime should produce:

- a rejected candidate and reason in debug output;
- an installation or configuration hint when Werk knows one;
- another accepted route only when policy and accelerator constraints permit it.

Media probes expose the same decision as a structured `task_readiness` value:
`available`, `fallback_available`, `installable`, `not_implemented`, or
`unavailable`. A concrete install command is shown only when the registered
adapter supplied that command. For example, a supported Qwen3-TTS VoiceDesign
model can recommend `werk backend install qwen-tts`; a task or model variant
without an implemented adapter explicitly says so and does not invent a pip or
Werk install command. `werk doctor --model MODEL --task TASK`, `--debug`,
`werk parameters MODEL --json`, and the HTTP discovery routes expose this same
status.

### Missing-backend negative smoke test

The repository includes a non-destructive test for the `installable` case:

~~~bash
./scripts/test-missing-media-backend.sh
~~~

It creates a metadata-only Qwen3-TTS VoiceDesign fixture and a temporary,
isolated model store, disables automatic backend provisioning, and verifies
both discovery and execution preflight. It never removes or changes the real
model store or an installed Qwen backend. To test a binary built from the
current checkout instead of the `werk` on `PATH`, select it explicitly:

~~~bash
WERK_BIN=./target/debug/werk ./scripts/test-missing-media-backend.sh
~~~

The relevant output is:

~~~text
Task readiness: installable
  Adapter: qwen3_tts_voice_design
  Required backend: qwen-tts
  Recommendation: werk backend install qwen-tts
...
Recommendation: run `werk backend install qwen-tts`; no compatible fallback was verified
PASS: missing managed backend was detected before inference and no output was created.
~~~

A missing required backend is a preflight failure, not a warning attached to a
successful inference. Werk reports `fallback_available` instead only when a
different runtime for the same model and task actually passed its probe and
the request policy permits that route.

## Parameter policy

The parameter policy is independent of backend fallback:

| Policy | Unsupported explicit parameter |
| --- | --- |
| <code>strict</code> | Reject the runtime/request. This is the default. |
| <code>warn</code> | Continue only where the resolver/adapter can safely do so and report a warning. |
| <code>permissive</code> | Allow the broadest adapter behavior, still subject to runtime validation. |

Use <code>strict</code> for production and compatibility testing. It prevents a
voice, sampler, offload or quality option from being silently ignored.

## Backend commands

The current command surface is:

~~~text
werk backend install TARGET
werk backend list
werk backend doctor [--debug]
~~~

The install targets are:

~~~text
llama-cuda
llama-rocm
llama-vulkan
llama-metal
llama-cpu
onnx-cuda
onnx-rocm
onnx-cpu
vllm
qwen-tts
~~~

There is currently no <code>werk backend uninstall</code> command. See
[Uninstall and cleanup](#uninstall-and-cleanup).

## What each managed installer does

| Target | Provisioning behavior | Main prerequisites | Validation |
| --- | --- | --- | --- |
| <code>llama-cuda</code> | Shallow-clones current llama.cpp and builds llama-server with CMake and GGML CUDA. | Git, CMake, C/C++ compiler, NVIDIA driver and CUDA toolkit. | llama-server help plus known CUDA initialization failures. |
| <code>llama-rocm</code> | Builds llama-server with GGML HIP. | Git, CMake, C/C++ compiler and compatible ROCm/HIP toolchain. | Executable help; a real HIP inference is not part of installation validation. |
| <code>llama-vulkan</code> | Builds llama-server with GGML Vulkan. | Git, CMake, C/C++ compiler and Vulkan development SDK. | Executable help; a real Vulkan inference is not part of installation validation. |
| <code>llama-metal</code> | Builds llama-server with GGML Metal. | macOS, Xcode command-line tools and CMake. | Rejected before build outside macOS; executable help after build. |
| <code>llama-cpu</code> | Builds the default CPU llama-server. | Git, CMake and a C/C++ compiler. | Executable help. |
| <code>onnx-cuda</code> | Copies an existing platform-specific Werk ONNX runner bundle. | A compatible bundled or explicitly configured runner. | Runner help only. |
| <code>onnx-rocm</code> | Copies an existing platform-specific Werk ONNX runner bundle. | A compatible bundled or explicitly configured runner. | Runner help only. |
| <code>onnx-cpu</code> | Copies an existing platform-specific Werk ONNX runner bundle. | A compatible bundled or explicitly configured runner. | Runner help only. |
| <code>vllm</code> | Creates an isolated virtual environment and installs vLLM with pip on eligible generic Linux hosts. On DGX Spark and AMD Strix Halo this target stops with platform-specific container/environment guidance instead of installing an unverified generic wheel. | Native Linux x86_64, Python/venv, pip, compatible PyTorch and accelerator stack. | Import/version and runtime health checks. |
| <code>qwen-tts</code> | Creates an isolated virtual environment and installs exactly qwen-tts 0.1.1. | Python 3.9+, venv, pip and platform-compatible PyTorch/audio dependencies. | Exact package version and Qwen3TTSModel import. |

Important limitations:

- llama.cpp provisioning follows the current upstream default branch rather
  than a Werk-pinned commit, so identical Werk versions can build different
  upstream revisions at different times;
- the ONNX installers do not download or build a runner today;
- ONNX installation verifies the executable, not the requested CUDA/ROCm
  execution provider;
- successful Python import does not prove that a large model fits or that the
  requested GPU kernel is available.

## Platform support matrix

The table distinguishes practical primary support from best-effort or
upstream-unconfirmed paths.

| Target | Native Linux x86_64 | AMD Strix Halo / `gfx1151` | Linux aarch64 / DGX Spark | WSL2 | Native Windows | macOS Apple Silicon |
| --- | --- | --- | --- | --- | --- | --- |
| Werk release binary | Generic x86_64 artifact | Backend-neutral Strix Halo x86_64 artifact | Spark-only arm64 `sm_121` artifact | Linux x86_64 artifact | x86_64 artifact | arm64 artifact |
| llama CPU | Supported build path | Supported build path | Supported build path | Supported build path | Supported build path | Supported build path |
| llama CUDA | Primary NVIDIA path | Not applicable | Primary GB10 path; build upstream natively | Best-effort Linux/CUDA path | Build path with CUDA toolchain | Not applicable |
| llama ROCm | Primary practical ROCm path | Primary HIP path; real `gfx1151` smoke required | Not applicable to GB10 | Not recommended | Not practically supported | Not applicable |
| llama Vulkan | Build path with Vulkan SDK | Implemented alternative; benchmark on target hardware | Not a primary Spark path | Best effort | Build path with Vulkan SDK | Not a primary path |
| llama Metal | Rejected | Rejected | Rejected | Rejected | Rejected | Supported build path |
| local vLLM | Eligible | Operator-provisioned ROCm environment only; generic managed pip install rejected | Native-Linux eligible; ARM64 package/model support remains upstream-dependent | Installer allowed, local execution currently rejected/cautioned | Rejected | Rejected |
| remote vLLM | Supported | Supported; declare ROCm | Supported | Supported | Supported | Supported |
| Qwen-TTS | CUDA is the primary documented path; CPU possible | ROCm/model dependent and hardware-unvalidated | Experimental/upstream-dependent | Experimental/upstream-unconfirmed | Experimental/upstream-unconfirmed | CPU/MPS experimental and upstream-unconfirmed |
| ONNX targets | Requires matching runner bundle | Requires matching ROCm runner bundle | Requires matching Linux aarch64 runner bundle | Requires matching Linux bundle | Requires matching Windows bundle | Requires matching macOS bundle |

Werk's release tooling produces profiles for generic Linux x86_64, AMD Strix
Halo x86_64, Linux aarch64/DGX Spark, Windows x86_64 and macOS arm64. Both
hardware profiles must be packaged and smoke-tested on their named host. The
Spark artifact uses a CUDA 13+ toolchain and targets GB10 compute capability
12.1 (`sm_121`); the Strix artifact remains backend-neutral and discovers ROCm
or Vulkan companion runtimes later. Windows arm64 and macOS x86_64 are not
current release targets.

### DGX Spark and Nemotron

Werk recognizes text-only Nemotron-H safetensors architectures and can route
them to vLLM. On Spark the recommended deployment is NVIDIA's compatible vLLM
container with Werk attached to its OpenAI endpoint. A separately provisioned
local vLLM interpreter can also be selected explicitly, but the managed generic
pip installer is deliberately disabled on Spark.

GGUF checkpoints can use a compatible llama.cpp server; the managed CUDA build
detects GB10 and selects the architecture-specific CMake target. Werk's vLLM
adapter remains text-only, so recognizing Nemotron-H does not imply support for
Nemotron Omni image, audio or video inputs. See the complete
[DGX Spark guide](integrations/dgx-spark.md).

### AMD Strix Halo

On Linux x86_64 Ryzen AI Max systems, Werk recognizes the Strix Halo CPU or
the `gfx1151` ROCm agent. GGUF can use an external llama.cpp ROCm/HIP or Vulkan
server. Supported safetensors text architectures, including eligible
Nemotron-H repositories, can use a separately provisioned ROCm vLLM
interpreter or endpoint. The generic managed vLLM pip install is deliberately
disabled for this profile so it cannot install a CUDA-oriented or otherwise
incompatible wheel.

Strix Halo uses physically shared CPU/GPU memory. Model estimates must not add
host RAM and GPU-visible capacity as independent pools, and CPU offload does
not create another physical tier. The integration and diagnostics are
implemented, but each runtime still requires a real `gfx1151` inference smoke
before being called hardware-validated. NVIDIA NVFP4 checkpoints are not
claimed as AMD-compatible. See the complete
[Strix Halo guide](integrations/strix-halo.md).

### WSL and vLLM

The current vLLM installer permits WSL and prints a warning, but local runtime
eligibility subsequently rejects WSL because vLLM can depend on GPU memory
features such as UVA and CUDA IPC. This is a known inconsistency. Prefer native
Linux or configure a remote vLLM endpoint.

## Qwen-TTS isolation

Qwen3-TTS does not use the generic Transformers text-to-audio pipeline. It uses
the qwen_tts package and its Qwen3TTSModel wrapper.

The package pins versions of shared libraries such as Transformers. Werk
therefore keeps it outside the general media-companion Python environment:

~~~text
WERK_HOME/
└── backends/
    └── qwen-tts/
        └── venv/
~~~

Install it explicitly:

~~~bash
werk backend install qwen-tts
werk backend doctor --debug
~~~

An externally managed compatible interpreter can be selected with:

~~~text
WERK_QWEN_TTS_PYTHON=/absolute/path/to/python
~~~

Discovery checks only the explicit interpreter and Werk's managed environment.
It deliberately does not select an arbitrary qwen-tts package from PATH.

### Qwen platform statement

Qwen does not publish a complete operating-system/accelerator support matrix.
Its documented reference examples use CUDA, BF16 and optionally
FlashAttention 2. The official CLI also accepts CPU, but this is not a
performance guarantee.

The qwen-tts 0.1.1 package is published as a platform-neutral Python wheel.
That describes the wheel, not the native PyTorch, audio or accelerator
dependencies. Until Werk has platform CI and real inference fixtures:

- Linux with NVIDIA CUDA is the primary documented target;
- CPU is expected to be much slower and remains model-dependent;
- Windows CUDA/CPU and WSL2 are experimental;
- macOS CPU/MPS is experimental;
- AMD ROCm is upstream-unconfirmed.

Upstream references:

- [Qwen3-TTS environment and CUDA examples](https://github.com/QwenLM/Qwen3-TTS#environment-setup)
- [qwen-tts package configuration](https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml)
- [qwen-tts 0.1.1 package files](https://pypi.org/project/qwen-tts/0.1.1/)

## Managed locations

The store root is selected in this order:

1. global <code>--model-home</code>;
2. <code>WERK_HOME</code>;
3. <code>XDG_DATA_HOME/werk1112</code>;
4. on Windows, <code>LOCALAPPDATA/werk1112</code> or the corresponding
   UserProfile fallback;
5. otherwise <code>HOME/.local/share/werk1112</code>.

Managed backend children are:

| Target | Child below the store root |
| --- | --- |
| llama targets | <code>backends/llama-cuda</code>, <code>llama-rocm</code>, <code>llama-vulkan</code>, <code>llama-metal</code> or <code>llama-cpu</code> |
| ONNX targets | <code>backends/onnxruntime-cuda</code>, <code>onnxruntime-rocm</code> or <code>onnxruntime-cpu</code> |
| vLLM | <code>backends/vllm</code> |
| Qwen-TTS | <code>backends/qwen-tts</code> |

Models, optimized model artifacts, outputs and jobs are separate siblings.
Removing one backend directory must not remove the store root.

## External runtime overrides

Managed installation is optional. Relevant explicit overrides include:

| Runtime | Override |
| --- | --- |
| llama.cpp | <code>WERK_LLAMA_SERVER_CUDA</code>, <code>WERK_LLAMA_SERVER_ROCM</code>, <code>WERK_LLAMA_SERVER_VULKAN</code>, <code>WERK_LLAMA_SERVER_METAL</code>, <code>WERK_LLAMA_SERVER_CPU</code> |
| ONNX | <code>WERK_ONNX_RUNTIME_*</code> for execution and <code>WERK_ONNX_RUNTIME_BUNDLE_*</code> for provisioning bundles |
| vLLM | <code>WERK_VLLM_PYTHON</code>, or remote <code>WERK_VLLM_HOST</code>, <code>WERK_VLLM_PORT</code> and optional <code>WERK_VLLM_MODEL</code> |
| Qwen-TTS | <code>WERK_QWEN_TTS_PYTHON</code> |
| general media companion | <code>WERK_MEDIA_PYTHON</code> or <code>WERK_MEDIA_COMPANION</code> |

Explicit paths remain the operator's responsibility and are not removed by
Werk.

## Uninstall and cleanup

There is no managed backend-uninstall subcommand in the current CLI:

~~~text
werk backend uninstall qwen-tts
# not implemented
~~~

The current safe manual procedure is:

1. stop Werk servers and active inference using the target;
2. determine the resolved store root from the same <code>--model-home</code> or
   <code>WERK_HOME</code> configuration used during installation;
3. select exactly one child listed in [Managed locations](#managed-locations);
4. remove that child with the operating system's normal file-management tools;
5. run <code>werk backend list</code> and <code>werk backend doctor --debug</code>.

Do not remove the complete store root. That would also target models, outputs,
jobs and shared artifacts. Manual backend removal is not recoverable except by
reinstallation.

For Qwen-TTS, removing only
<code>WERK_HOME/backends/qwen-tts</code> removes its isolated environment. It
does not remove Qwen model repositories stored under
<code>WERK_HOME/models</code>.

A future clean command should be idempotent, resolve and validate the target
inside the active store, stop a managed worker, support a dry run, remove only
target-owned state and never delete models or outputs.

## Known backend-management gaps

- no managed uninstall command
- backend list/doctor do not yet expose a dedicated Qwen-TTS status row
- backend CLI summary text historically referred only to llama.cpp
- no pinned llama.cpp revision or reproducible build receipt
- no automatic ONNX runner download
- no execution-provider validation during ONNX provisioning
- WSL vLLM install/runtime eligibility mismatch
- Qwen-TTS platform support is not yet verified by a multi-OS inference matrix
- no single machine-readable backend capability manifest

These gaps should remain visible in documentation and diagnostics until the
implementation changes.
