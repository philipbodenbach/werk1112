# Runtime review, 2026-10-06

Werk baseline: `8d8ee73a002e8473818d1e86099e0a21b3b51d23`.
The revisions below were resolved through GitHub's commit API and their referenced
source files read on this date. README performance numbers are author reports;
none are Werk measurements. A native flag is evidence of an interface, not of
every model/quantization/parallelism combination.

| Runtime / reviewed revision | Models and hardware | Placement / offload | API, persistence, limits | Evidence and decision |
|---|---|---|---|---|
| llama.cpp `abeada335e2e78bd3fe63febafab7e900ce75810` | GGUF; build-specific CUDA/ROCm/Vulkan/CPU/etc. | Device list; none/layer/row/tensor modes; relative tensor split; CPU MoE and tensor overrides; separate KV controls | Native server, optional slot snapshots; no universal snapshot portability | [Argument implementation](https://github.com/ggml-org/llama.cpp/blob/abeada335e2e78bd3fe63febafab7e900ce75810/common/arg.cpp). Base integration; structured profiles initially expose single/layer and existing native CPU offload. Actual local hardware test uses the separately recorded installed build. |
| vLLM `3403e0f176efb90b0d92cc0ff1d350f092ed9e9f` | HF safetensors; supported CUDA/ROCm models and kernels depend on build | Local TP/PP/EP; CPU weights via UVA or asynchronous grouped prefetch; KV offload is separate | OpenAI transport; prefix caching does not imply named Werk KV snapshots | [Offload config](https://github.com/vllm-project/vllm/blob/3403e0f176efb90b0d92cc0ff1d350f092ed9e9f/vllm/config/offload.py). Base integration uses local MP and structured CPU-weight GiB. Prefetch/KV offload not exposed without combination validation; no local vLLM GPU run claimed. |
| GenerelSchwerz `907a73da9a149faa8c42ccde890f1d575586810f` | Fork-specific GGUF/CUDA MoE layouts | Per-expert-tensor slots, pinned staging, cache/prefetch and native grouped kernels; layer split permitted, tensor split incompatible | llama server transport and slot state; expert cache is not persistent weight residency | [Fork feature contract](https://github.com/GenerelSchwerz/llama.cpp/blob/907a73da9a149faa8c42ccde890f1d575586810f/docs/fork-features.md), Werk `cuda_offload.rs`. Keep existing optional pinned flavor; previous native boundary failure remains documented in `backends.md`. |
| Strata `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253` | Specialized Qwen-family native/canonical packs, tokenizer and profiles; NVIDIA/AMD paths have separate constraints | Layer stages, CPU expert pool, per-stage expert caches, optional helper GPUs and KV streaming | Python OpenAI/Anthropic server around native stdin/stdout protocol; specialized checkpoints; helper/speculative paths have additional numerical/configuration limits | Source-reviewed, not integrated or hardware-tested by this change. Detailed integration gaps below. |
| ik_llama.cpp `739839a5c922cf9663988912004912e9f0671d7d` | GGUF with additional quant formats; maintained CPU/CUDA emphasis | Graph splitting and CPU tensor/offload controls | llama-style server; graph + partial CPU offload has a documented correctness caveat | [README at pin](https://github.com/ikawrakow/ik_llama.cpp/blob/739839a5c922cf9663988912004912e9f0671d7d/README.md). No new adapter: graph correctness and format differences need independent qualification. |
| KTransformers/KT-Kernel `a5d7ad90479c9e8bdee491c16bfd563fa9d70262` | Specialized MoE CPU/GPU kernels, AMX/AVX512/AVX2, CPU quant formats | Heterogeneous expert computation and NUMA-aware memory; serving integrations | Serving stack/model conversion differs from llama and vLLM profiles | [Project contract](https://github.com/kvcache-ai/ktransformers/blob/a5d7ad90479c9e8bdee491c16bfd563fa9d70262/README.md). No additional adapter: existing CPU-MoE path addresses the initial requirement without another engine. |
| ExLlamaV3 `16a49792a3c93d8432d72e6c4bce800841566577`, TabbyAPI `2fd6cc76203a66e13042daf7d76e5898b21c1ad8` | Engine-specific quantization and supported HF architectures | TP/EP, CPU MoE; format/kernel support differs | Tabby OpenAI server; CPU offload/TP require appropriate shared memory | [Engine](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/README.md), [server](https://github.com/theroyallab/tabbyAPI/blob/2fd6cc76203a66e13042daf7d76e5898b21c1ad8/README.md). Not integrated. Tabby identifies its server as a hobby rather than production deployment target; its AGPL licensing also requires a separate distribution review. |

## Strata integration assessment

The current server is usable through more than a generic OpenAI label, but it is
not a drop-in llama-server or vLLM executable. The smallest suitable transport
would reuse Werk's OpenAI client with a dedicated process/configuration adapter,
plus the native `/v1/messages/count_tokens` route. Reusing llama's `/completion`
and `/apply-template` + `/tokenize` contract would be incorrect.

Concrete sources at the pin:

- [`serve/server.py`](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/serve/server.py):
  `gpu_list` (1771), `child_env` (1949), `engine_args` (1850),
  `StrataEngine` (533), count-token handler (4383), startup (4750).
- [`docs/MULTI_GPU.md`](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/docs/MULTI_GPU.md):
  stage-local caches, replicated state/buffers, pack restrictions, batching,
  WDDM pinning limits and opt-in numerical paths.
- [`docs/SECOND_GPU.md`](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/docs/SECOND_GPU.md):
  helper-GPU expert ownership and pinned result-transfer paths.

The unmodified server's configured GPU path accepts integer physical indices and
overwrites inherited CUDA visibility; configuration environment entries can then
overwrite it again. This cannot be passed through as a managed UUID reservation.
A future adapter must synthesize a restricted config, preserve the reserved UUID
mask, translate stage indices explicitly, and regression-test this against the
native server. Pack/tokenizer/profile files also need an exact registered artifact
identity; the existing GGUF manifest alone is insufficient. Required packs and
tokenizer artifacts for native validation were not provisioned in this task.

These are concrete **remaining integration work**, not proof that Strata is
fundamentally impossible to integrate. No Strata adapter, build, or performance
claim is shipped here. The smallest next step is a pinned config/visibility shim
with process/HTTP/token-count contract tests, followed by validation of an
explicitly configured native pack. Do not auto-select Strata for arbitrary GGUF.
Its MIT source and bundled third-party notices would need to accompany any future
copied components; this change copies no Strata code.

## Architecture decision

Use the existing generation adapters, per-key `RuntimeCache`, `ManagedChild`,
`ModelRuntimeIdentity`, API session caches and runtime-control adapters. Bind a
server-owned immutable plan to each instance, with independent native environment,
binary and admission lease. Extend the existing resource module for admission.
Keep state compatibility stricter than model alias equality. Unsupported native
movement remains unsupported; startup configuration is not live residency control.

The implementation and remaining acceptance gaps are tracked in
[multi-gpu-progress.md](multi-gpu-progress.md). This review is not a certification
of the upstream branches or the untested hardware combinations.
