# Backends, routing and platform support

Werk1112 separates the stable inference contract from concrete execution
runtimes. A model can be installed and classified without every backend being
present, and one model may have several eligible runtime candidates.

This page documents the current implementation. “Installer exists” does not
mean “every model and accelerator combination is verified.”

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
| <code>vllm</code> | Creates an isolated virtual environment and installs vLLM with pip. | Native Linux, Python/venv, pip, compatible PyTorch and accelerator stack. | Import/version and runtime health checks. |
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

| Target | Native Linux | WSL2 | Native Windows | macOS Apple Silicon |
| --- | --- | --- | --- | --- |
| Werk release binary | x86_64 artifact | Linux artifact | x86_64 artifact | arm64 artifact |
| llama CPU | Supported build path | Supported build path | Supported build path | Supported build path |
| llama CUDA | Primary NVIDIA path | Best-effort Linux/CUDA path | Build path with CUDA toolchain | Not applicable |
| llama ROCm | Primary practical ROCm path | Not recommended | Not practically supported | Not applicable |
| llama Vulkan | Build path with Vulkan SDK | Best effort | Build path with Vulkan SDK | Not a primary path |
| llama Metal | Rejected | Rejected | Rejected | Supported build path |
| local vLLM | Eligible | Installer allowed, local execution currently rejected/cautioned | Rejected | Rejected |
| remote vLLM | Supported | Supported | Supported | Supported |
| Qwen-TTS | CUDA is the primary documented path; CPU possible | Experimental/upstream-unconfirmed | Experimental/upstream-unconfirmed | CPU/MPS experimental and upstream-unconfirmed |
| ONNX targets | Requires matching runner bundle | Requires matching Linux bundle | Requires matching Windows bundle | Requires matching macOS bundle |

Werk currently publishes normal end-user artifacts for Linux x86_64, Windows
x86_64 and macOS arm64. Linux arm64, Windows arm64 and macOS x86_64 are not
current release targets.

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
| vLLM | <code>WERK_VLLM_PYTHON</code>, or remote <code>WERK_VLLM_HOST</code> and <code>WERK_VLLM_PORT</code> |
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
