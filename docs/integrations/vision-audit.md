# Chat image-understanding audit — 2026-10-21

Scope: all 26 entries in `src/backend/mod.rs::RUNTIME_REGISTRY`, all 18
`BACKEND_INSTALL_TARGETS`, selectable legacy backends, hosted provider and
Werk-to-Werk forwarding. Registry membership is eligibility, not proof of a
particular model's execution. Image generation, pooling, audio and video are
separate tasks. See [setup commands and transport limits](vision.md).

## Existing path and changes

The OpenAI-compatible API already accepted ordered text/image parts and retained
them in llama-server and vLLM requests. llama-server already loaded a manifest
projector with `--mmproj`. HTTP streaming, tool-history transport, durable chat
serialization and WerkWorker's normalized inline attachments already existed.
This audit therefore does not add a second inference API or agent loop.

Corrected integration gaps:

- Main GGUF selection silently included an optional projector. Pull now asks or
  requires explicit opt-in, reports repository revision, candidate filenames and
  LFS sizes, and supports later projector-only setup.
- Filename sorting could choose across unrelated projector families; BF16 also
  incorrectly matched the F16 substring. Selection now separates families and
  precision, records an explicit manifest choice, and rejects contradictory
  GGUF identities before installing an association.
- vLLM's exact image architecture list omitted `qwen3_5` and `qwen3_5_moe`, the
  config identifiers used by the Qwen3.8 family. Similar names such as
  `qwen3_next` are not added.
- Image requests could enter ordinary automatic provisioning. Image routing now
  disables provisioning, and explicit disabled setup is enforced by adapters as
  well as routing. Runtime discovery remains read-only.
- Some direct SDK paths detected only the convenience `image_urls` list.
  Routing and text-only guards now also inspect structured message parts.
- The MLX-VLM CLI could flatten image history, interleaving and detail hints.
  Unrepresentable requests now fail explicitly. Local safetensors image paths
  check manifest-listed processor configuration; remote vLLM owns its assets.

Main changes are in `backend/vision_setup.rs`, `model_store/vision.rs`,
`model_store.rs`, `cli.rs`, `backend/{llama_server,vllm,external}.rs` and shared
request guards. WerkWorker changes add regression coverage for attachment
retention through compaction, storage and subsequent requests; its image
transport required no implementation rewrite.

## Complete working matrix

“Unsupported” means the Werk adapter has no usable chat-image path, even if
upstream supports some VLMs. “Not applicable” means this route implements a
different task. “Untested” never means proven. Hardware not present was not
simulated as a real inference success. Grouped rows explicitly name every
registry entry they cover.

| Backend / registry entries | Platform / accelerator | Format / architecture | Upstream image support | Werk integration | Components | Setup | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LlamaServerCuda | Linux/Windows NVIDIA CUDA; ARM64 needs matching toolchain | GGUF, concrete libmtmd architecture | Yes, model dependent | Ordered multipart HTTP, streaming, tool history | Compatible server + model-specific projector | `llama-cuda --vision`, per-model setup | Real Qwen27B CUDA image inference passed in normal, stream and tool-follow-up modes; deterministic transport/gating tests |
| Special llama CUDA NVFP4 / offload installers | NVIDIA, build-specific hardware limits | GGUF; NVFP4 needs its supported architecture and device | Depends on pinned fork/build | Same server adapter; no inherited success claim for a special text build | Validated special build plus mmproj support and matching assets | `llama-cuda-nvfp4` / `llama-cuda-offload --vision`; valid build reused, replacement staged | Build selection and validation code audited; these special profiles not live-tested |
| LlamaServerRocm | Linux/Windows where supported by installer, AMD HIP | GGUF | Model dependent | Same multipart server path | ROCm-compatible server + projector | `llama-rocm --vision` | AMD hardware absent; transport/routing tests only |
| LlamaServerVulkan | Linux/Windows, Vulkan device | GGUF | Model dependent | Same multipart server path | Vulkan server + projector | `llama-vulkan --vision` | Local build present; no second GPU model loaded because existing CUDA service occupies VRAM; real Vulkan inference untested |
| LlamaServerMetal | macOS Apple Silicon / Metal | GGUF | Model dependent | Same multipart server path | Metal server + projector | `llama-metal --vision` | Apple hardware absent; real inference untested |
| LlamaServerCpu | Host CPU | GGUF | Model dependent | Same multipart server path | CPU server + projector | `llama-cpu --vision` | Real Qwen27B CPU image inference passed after offline local-asset setup in an isolated store |
| Legacy LlamaFast / LlamaHighlevel (outside registry) | CPU; feature-gated CUDA/Vulkan legacy variants; legacy Metal/ROCm modes are unavailable | GGUF, legacy binding architectures | Current upstream server support does not apply to old bindings | **Unsupported**: text token evaluation, no mtmd image encoder | No enabling component in adapter | Cargo `llama-fast` / `llama-cpp`, legacy feature combinations; no vision wizard | Structured-image rejection code audited; no legacy model execution |
| VllmCuda | Native Linux NVIDIA; remote endpoint from other hosts | Safetensors; qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_vl_moe, qwen3_5, qwen3_5_moe, glm4v, glm4v_moe | Yes, depends on release and architecture | Ordered multipart HTTP, streaming, tool history | Compatible vLLM/PyTorch, config/tokenizer/processor + integrated vision weights | `vllm --vision`; model import/pull; or configured remote endpoint | Exact architecture, transport and routing tests; no local suitable safetensors VLM/runtime installed for live proof |
| VllmRocm | Supported native Linux AMD or explicitly identified remote ROCm endpoint | Same safetensors allowlist | Hardware/build dependent | Same multipart path | ROCm-compatible vLLM/PyTorch and model assets | Preprovisioned ROCm environment / endpoint; generic installer is not a universal ROCm wheel installer | AMD hardware absent; real inference untested |
| MlxVlm | Native arm64 macOS 14+ / Apple Silicon | MLX or safetensors, **gemma4_unified only** | mlx-vlm supports more models than Werk exposes | Restricted CLI image list; rejects detail/interleaving/history it cannot preserve | mlx-vlm + matching model/processor | `mlx-vlm --vision`; model setup | Simulated runner and rejection tests; no Apple hardware/live inference |
| Mlx | Apple Silicon | MLX/safetensors text models | mlx-lm is the language-model runner | **Unsupported**: text CLI | No integrated vision processor | `mlx`; no vision question, `--vision` rejected | Text-only gating and tool-adapter tests |
| Omlx | Apple Silicon, managed macOS 15+ | MLX/safetensors | Upstream oMLX has VLM support | **Unsupported in this Werk adapter**: explicit image rejection | Current Werk adapter exposes text/generic tools, not upstream VLM contract | `omlx`; no misleading vision install | Existing adapter image-rejection tests; live Apple inference untested |
| OnnxRuntimeCuda / OnnxRuntimeRocm / OnnxRuntimeCpu | Corresponding runtime bundle/EP and host support | ONNX; supported exports from safetensors | ONNX GenAI can support multimodal models | **Unsupported**: Werk external runner and Python fallback send text/tokens; no multimodal processor inputs | Text ONNX runner or onnxruntime-genai | `onnx-cuda`, `onnx-rocm`, `onnx-cpu`; no vision question | Runner transport tests for all 3 offered modes; direct structured-image guard. DirectML/CoreML/TensorRT/OpenVINO are not additional selectable chat-image EPs here |
| CandleCuda / CandleMetal / CandleCpu | CUDA feature / macOS Metal feature / CPU | Registry's safetensors and GGUF text architectures | Library has model-specific VLM implementations | **Unsupported**: this adapter implements text generation only | Compiled Candle text model | Cargo features `candle-cuda`, `metal`, CPU; no runtime installer | Shared capability/model tests and text-only guard; no image engine invented |
| BurnCuda / BurnCpu | Feature-gated CUDA / CPU | Safetensors phi3 | Framework building blocks do not imply adapter support | **Unsupported**: existing Phi3 text path | `burn-cuda` / `burn-cpu` | Cargo features, no runtime installer | Registry gating; direct image guard. Default-feature tests do not claim optional GPU compilation |
| TransformersCompat | Python, supported PyTorch device | Safetensors **chatglm** compatibility path | Transformers supports many multimodal classes | **Unsupported**: current compatibility worker is text-only | Text compatibility environment | `transformers`; no vision question | Text runner/tool roundtrip and image guard |
| TransformersPooling | CUDA/CPU | xlm-roberta, embedding_gemma2, laya | Task/model dependent | **Not applicable**: text embeddings/reranking/classification | Text-analysis companion | `text-analysis` | Registry/task classification tests |
| CandlePooling | CPU, optional Candle CUDA | Safetensors xlm-roberta | Task/model dependent | **Not applicable**: sequence classification/reranking | Embedded classifier | Cargo features | Registry/task tests |
| VllmPooling | CUDA | Safetensors xlm-roberta, embedding_gemma2 | Task/model dependent | **Not applicable**: text pooling | vLLM pooling worker | `vllm`; chat vision setup does not change pooling capabilities | Registry/task tests |
| MediaCompanionCuda / MediaCompanionRocm / MediaCompanionMetal / MediaCompanionCpu | Host/runtime-dependent CUDA/HIP/Metal/CPU | Media-specific layouts and architectures | Image generation/editing and audio/video capabilities differ by model | **Not applicable** to chat image-understanding | Media companion | `media`; no chat-vision question | Registry and media-task isolation tests |
| Qwen-TTS installer | Supported Python/PyTorch device | TTS assets | Audio task | **Not applicable** | Existing TTS environment | `qwen-tts`; no vision question | Installer coverage test; no speech inference required |
| Hosted ProviderBackend (outside registry) | Explicitly configured hosted Mistral service | `mistral-large-4`, provider metadata | Officially multimodal | Standard ordered OpenAI body forwarded; no local model installation | Existing provider authorization/configuration | Existing provider setup; local vision flag grants no cloud permission | Provider serialization/auth code audited and existing tests; no paid/cloud inference performed |
| WerkServerClient forwarding (outside registry) | Explicit remote Werk endpoint | Remote server's selected model | Determined by remote backend/model | Standard messages preserved; native protocol for server information | Endpoint authorization and remote readiness | Existing endpoint configuration | Transport tests, not proof of every remote backend |

On DGX Spark/GB10 and AMD Strix Halo the existing vLLM platform guards remain:
use the supported vendor runtime/container or explicitly provisioned Python;
no generic pip shortcut. WSL is not advertised as a managed local vLLM success.
No new cloud fallback or device fallback is added by this work.

## Versions and primary evidence

Checked against source and primary documentation on 2026-10-21:

- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) and
  [config](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json): a VLM
  using `Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`, with vision
  config. This is not Flash-Next. [Unsloth GGUF artifacts](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main)
  provide separate F16/BF16 projectors; sizes are read from repository metadata.
- [llama.cpp multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
  and [server contract](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
  Local source inspection showed `tools/server` links `mtmd`; root
  `LLAMA_BUILD_MTMD=OFF` controls standalone library build, not absence of
  multimodal server linkage. Existing server/tools flags were therefore kept.
  Special offload source is pinned to `907a73da9a149faa8c42ccde890f1d575586810f`;
  native NVFP4 source to `fc343a84bbd925b37dde3219de35ea0bed50d630` in the checkout.
- [vLLM model table](https://docs.vllm.ai/en/latest/models/supported_models/)
  includes Qwen3.5 conditional-generation dense/MoE. Werk's managed pip package
  is not version-pinned, so architecture eligibility cannot certify the
  installed version's model implementation or accelerator build.
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) and
  [mlx-lm](https://github.com/ml-explore/mlx-lm) are separate packages; Werk's
  installers are not pinned to a single package version. The Werk allowlist and
  CLI transport are narrower than upstream multimodal support.
- [oMLX v0.7.0](https://github.com/jundot/omlx/tree/v0.7.0) is the installer pin;
  upstream VLM support is not exposed by Werk's current text adapter.
- [ONNX GenAI](https://onnxruntime.ai/docs/genai/),
  [Candle](https://github.com/huggingface/candle), [Burn](https://github.com/tracel-ai/burn),
  and [Transformers multimodal chat](https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal)
  describe upstream facilities, not this adapter's readiness. Cargo uses Candle
  0.9.2, Burn 0.21.0, legacy llama bindings 0.3.2. The Transformers installer uses
  `>=5.19,<6`; ONNX support depends on selected external bundle/Python environment.
- [Mistral Large 4](https://docs.mistral.ai/models/mistral-large-4-0) documents
  vision. No cloud request was made as part of this audit.

## Verification and limits

`scripts/vision-smoke.py` creates a new two-color PNG for each request; neither
color is in the prompt. It checks exact left/right answers in non-streaming,
streaming (including DONE), and post-tool history requests. It only contacts an
explicitly configured existing server, never installs a runtime or model, and
does not print credentials or image data.

The existing CUDA service on RTX 3090 passed all three modes with local
Qwen3.8-27B UD-Q4_K_XL and its already installed BF16 projector. Server identity,
device and idle status were checked through `/werk/v1/info`, `/deployments` and
`/observability`. Its running process was not restarted; this is evidence for
real inference/transport, while the changed setup code is covered by separate
tests. No weights, projectors or runtime packages were downloaded.

A separate CPU smoke also passed with the changed CLI, an isolated temporary
store referencing the same existing weights, explicit offline image setup,
4096-token context, eight threads, and CPU projector processing. The random
blue/red image was correctly answered in 26.83 seconds after server startup.
That test server was stopped and its temporary store removed; the existing
CUDA service was left running.

Deterministic coverage includes setup yes/skip/cancel, noninteractive defaults,
all installer targets, conflicting CLI flags, ambiguous/wrong/missing projector
assets, cached-asset reuse, manifest failure preservation, runtime identity
changes, exact VLM architecture gating, MLX semantic rejection, API multipart
serialization/streaming/tool history, and Worker attachment persistence and
compaction. Readiness remains a prerequisite check and does not save a universal
“inference proven” bit from this one local machine.


Reproducible checks used for this audit:

```sh
cargo fmt --all --check
cargo check --locked --all-targets --no-default-features
cargo test --locked --no-default-features --lib -- --test-threads=1
python3 -B -m unittest discover -s src/backend -p 'test_*mlx*.py'
python3 -B -m unittest discover -s scripts/tests -p test_vision_setup.py
python3 scripts/generate-cli-reference.py target/debug/werk --check
# In the adjacent WerkWorker checkout:
cargo test --locked --test attachments --test context_policy --test werk
```

The aggregate CLI parser test uses a dedicated larger stack; callers need no
stack environment override. Serial Rust execution avoids shared file-lock/fork
interference observed with parallel execution in this environment. Tests retain
their original assertions. The full serial suite passed 1,323 tests with four
deliberately ignored; subsequent focused installer/asset checks also passed. Python MLX/oMLX probes ran 246 cases, with 34 platform/runtime-dependent
skips. Six real CLI/PTY setup tests (including cancellation after a new import) and
29 targeted Worker tests passed. The final vLLM-focused rerun passed 73 tests.
The Worker checks use captured/mock model requests, not a real Worker-controlled
Qwen agent run. The separate live API smokes prove image inference; they are not
an end-to-end claim about a Windows clipboard or a complete coding-agent task.

Standard OpenAI clients continue to use `/v1/chat/completions`; they do not
need to implement WerkProtocol. The native protocol supplies server/capacity
information and existing authorization, while model task readiness remains in
Werk's existing discovery/diagnostic surface. This change does not invent a
second permission system or silently authorize a hosted provider.
