# Native llama.cpp NVFP4 profile

This versioned patch integrates real Marlin CUDA kernels into the existing GGML CUDA backend. It does not import, link to, launch or require the vLLM inference runtime, Python or libtorch. WERK continues to own requests, tool calling and persistence through its existing llama.cpp server adapter.

- llama.cpp base: `fc343a84bbd925b37dde3219de35ea0bed50d630`.
- Marlin source: vllm-project/vllm commit `26f49e336a1498300a442a0d95911c11f5284500`, `csrc/libtorch_stable/quantization/marlin/` and `csrc/core/scalar_type.hpp`. The fork traces back to IST-DASLab/Marlin. The kernel headers retain upstream copyright notices; Apache-2.0 license is included both here and inside the patch. The adapter is covered by WERK's project license.

## Dispatch

The child-only `GGML_CUDA_NVFP4_KERNEL` accepts `auto` (default), `native`, `marlin`, or `ggml`. Auto first uses the upstream Blackwell native FP4 tensor-core path for supported devices, compiled architectures and operations, then Marlin, then upstream GGML. Native and Marlin overrides abort on unsupported NVFP4 GPU operations instead of silently changing activation format. CPU-offloaded tensors continue to use GGML CPU kernels. `ggml` preserves upstream dispatch, including upstream native FP4 where selected. Fusion guards keep upstream MMVQ fusion from bypassing the selected NVFP4 path.

Marlin supports NVIDIA SM80+ with corresponding compiled CUDA support. Dense `MUL_MAT` supports 2D matrices; `MUL_MAT_ID` supports normal expert routing with broadcast gate/up inputs and per-expert down inputs. Weights must be contiguous GGML NVFP4 and satisfy either K%128=N%64=0 or K%64=N%128=0. Higher-dimensional dense layouts fall back in auto and fail with a forced Marlin override. Expert routing also follows the upstream CUDA helper limits: fewer than 1024 selected experts, fewer than 2^22 tokens, and four bytes of shared-memory scratch per token within the device per-block opt-in limit.

The adapter converts F32 activations to BF16, expands unsigned E4M3 block scales exactly to BF16, and executes Marlin FP4 x BF16 tensor-core GEMM with FP32 accumulation/reduction and direct FP32 output. The normal llama graph applies tensor/expert `weight_scale_2` after this operation. No FP16 output is materialized before that scale, so large unscaled outputs do not overflow at 65504. BF16 input rounding remains part of the W4A16 numerical contract. GGUF subnormal block scales remain nonzero. Native Blackwell uses upstream W4A4 activation quantization.

This first integration repacks the currently executing dense matrix or active expert on the GPU on each operation, using GGML's scratch pool. It does not cache a second model-sized weight representation. MoE routing and empty-expert checks stay on the GPU; launches reuse one expert's scratch space. Repacking and per-expert launches add overhead; no throughput improvement is claimed.

## Build and validation

Apply `patch.diff` with `git apply --check` and `git apply` at the pinned source revision. Check `git apply --reverse --check` to recognize an already applied patch. Build the `llama-server`, `werk-fp4-probe`, `werk-marlin-smoke` and `werk-nvfp4-smoke` CMake targets with `GGML_CUDA=ON`. The two smoke targets run tiny synthetic tensors and never download a model.

`werk-fp4-probe` links the same `libggml-cuda` as the server and emits schema 1 JSON with kernel ABI `werk-nvfp4-v1`, pinned revision, actual compiled native support, and CUDA-visible device capabilities. The probe executes a tiny kernel per visible device and reports the actual loaded `compiled_arch` (86/120, or 0 on failure) and CUDA status; it does not infer binary compatibility merely from architecture ordering. Device enumeration respects `CUDA_VISIBLE_DEVICES`. WERK's installer separately binds the server, probe and CUDA library with a SHA-256 receipt.

`werk-marlin-smoke` compares actual Marlin kernels against an independently decoded FP32 CPU reference, including partial M tiles, batches larger than the SM tile grid, both K/N layouts, empty experts, E4M3 subnormal scales, K=8192 and values exceeding FP16's range. `werk-nvfp4-smoke` builds real GGML Dense/MoE graphs, including normal external scaling and both expert input layouts and strided tensors, to exercise dispatch and gather/scatter across repeated CUDA graph execution. Run it in separate processes with `GGML_CUDA_NVFP4_KERNEL=auto` and `marlin`.

RTX 3090 / SM86 is the available execution test device. Blackwell SM120/121 native dispatch cannot be runtime-tested on that device; compilation and reported capabilities must not be presented as a Blackwell benchmark.

A reproducible local build (replace CUDA path and architecture with the target device) is:

```sh
git clone https://github.com/ggml-org/llama.cpp.git /tmp/werk-nvfp4-check
git -C /tmp/werk-nvfp4-check checkout --detach fc343a84bbd925b37dde3219de35ea0bed50d630
git -C /tmp/werk-nvfp4-check apply /absolute/path/to/werk1112/runtime/llama-nvfp4/patch.diff
cmake -S /tmp/werk-nvfp4-check -B /tmp/werk-nvfp4-check/build \
  -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86-real -DLLAMA_OPENSSL=OFF
cmake --build /tmp/werk-nvfp4-check/build --target \
  llama-server werk-fp4-probe werk-marlin-smoke werk-nvfp4-smoke -j 4
/tmp/werk-nvfp4-check/build/bin/werk-fp4-probe
/tmp/werk-nvfp4-check/build/bin/werk-marlin-smoke
GGML_CUDA_NVFP4_KERNEL=auto /tmp/werk-nvfp4-check/build/bin/werk-nvfp4-smoke
GGML_CUDA_NVFP4_KERNEL=marlin /tmp/werk-nvfp4-check/build/bin/werk-nvfp4-smoke
```

Use a matching system compiler/linker/library environment. On hosts mixing Homebrew
and system GCC, `FindOpenMP` may otherwise select an incompatible `libgomp`; explicitly
set `OpenMP_gomp_LIBRARY` to the matching compiler's library if needed.
