# NVFP4, native Blackwell and Marlin

WERK runs NVFP4 GGUF weights through its existing llama.cpp CUDA adapter. WERK
continues to own the API, sessions and persistence. This profile does not start
vLLM, SGLang or a second inference service. The standalone Marlin CUDA kernels
are compiled into the GGML CUDA backend.

## Default selection

Without a kernel override, the native runtime selects per supported NVFP4
operation:

1. Native Blackwell FP4 tensor cores, when both the GPU and compiled CUDA
   architecture support them.
2. Marlin W4A16 on a supported GPU and tensor layout.
3. Upstream GGML dispatch for remaining layouts.

A marketing name alone is insufficient. An RTX 5090 or RTX PRO 6000 Blackwell
has different capabilities from an RTX 6000 Ada or RTX A6000. The capability
probe uses CUDA's visible devices and the compiled kernels. `CUDA_VISIBLE_DEVICES`
is honored. Native FP4 and Marlin W4A16 use different activation arithmetic;
WERK does not treat their persisted state as interchangeable.

For an installed NVFP4 GGUF, default `--backend auto` can build the matching
profile when a CUDA compiler and NVIDIA GPU are detected. It does so in a
separate versioned directory, without changing the default CUDA runtime for
other models. `--no-auto-install-backends` disables provisioning. Explicit
`WERK_LLAMA_SERVER_CUDA` / `WERK_LLAMA_SERVER` paths retain precedence and must
point to a compatible, validated build.

To install and select the profile explicitly:

```bash
werk backend install llama-cuda-nvfp4
```

The installer uses llama.cpp commit
`fc343a84bbd925b37dde3219de35ea0bed50d630`, plus WERK's versioned native patch.
Sources and build artifacts live below
`backends/llama-cuda/nvfp4-<revision>-<patch-hash>-<architecture-hash>/`. A GPU architecture change therefore gets its own build, so a former Ampere
installation does not keep a Blackwell GPU on a Marlin-only build. A successful build must
provide `llama-server`, `werk-fp4-probe` and a matching build receipt. WERK
checks the server, probe and adjacent CUDA library hashes before trusting the
reported kernel capabilities. Unchanged checks are cached.

The profile builds without the optional NCCL dependency by default. Set
`WERK_LLAMA_CUDA_NCCL=true` when NCCL is installed and required for your
multi-GPU setup. `CMAKE_BUILD_PARALLEL_LEVEL` controls build concurrency (the
profile defaults to at most four compiler jobs). On Linux, an implicit system
GCC toolchain is kept with its system linker/OpenMP when Homebrew shadows `ld`;
explicit compiler environment settings take precedence.

Normal CUDA installation is still available with `werk backend install
llama-cuda`. The separate `llama-cuda-offload` experimental fork has another
source pin; these two source profiles cannot currently be combined. Ordinary
upstream CPU expert offload remains available.

## Kernel overrides

```bash
# Default: native Blackwell first, then Marlin, then compatible GGML.
werk serve --model my-nvfp4-gguf

# Explicit W4A16 execution for supported NVFP4 CUDA operations.
werk --backend cuda --fp4-kernel marlin serve --model my-nvfp4-gguf

# Require native FP4 hardware and kernels.
werk --backend cuda --fp4-kernel native serve --model my-nvfp4-gguf

# Leave NVFP4 operation selection to upstream GGML.
werk --backend cuda --fp4-kernel ggml serve --model my-nvfp4-gguf
```

The environment equivalent is `WERK_FP4_KERNEL=auto|native|marlin|ggml`.
An explicit CLI value takes precedence. `ggml` can itself use native Blackwell
kernels; it bypasses WERK's native-first/Marlin dispatcher and permits an existing upstream runtime
without installing the Marlin profile. That runtime must itself support the
model's GGUF tensors. Explicit `native`
or `marlin` rejects unsupported devices/builds and unsupported tensor layouts,
instead of silently choosing another activation format. CPU-offloaded
operations remain CPU operations.

Build architecture overrides use `WERK_LLAMA_CUDA_ARCH` or `CUDAARCHS`.
Automatic detection includes all attached architectures and uses `120a-real`
for SM 12.0 and `121a-real` for SM 12.1. The CUDA toolkit must support the
requested architecture (at least CUDA 12.8 for SM 12.0; 12.9 for SM 12.1).

## Hugging Face NVFP4 checkpoints

HF NVFP4 safetensors are not directly loadable by llama.cpp. First register or
pull the complete checkpoint, then repack it into a separate GGUF model:

```bash
werk import /path/to/nvfp4-checkpoint --name source-nvfp4 --link
werk convert source-nvfp4 --to gguf --name my-nvfp4-gguf \
  --python /path/to/converter-venv/bin/python
werk serve --model my-nvfp4-gguf
```

`werk pull ORG/REPOSITORY` may be used instead of import. Conversion discovers
the installed NVFP4 profile's `convert_hf_to_gguf.py`; `--converter PATH` can
select another compatible script. Install that script's Python requirements
in the interpreter supplied with `--python`. Conversion does not install
Python packages, download missing files or overwrite an existing model.

The supported input serializers are ModelOpt and compressed-tensors with
NVFP4 metadata. Metadata detection includes nested text configuration, ModelOpt
sidecars and per-layer/mixed recipes. Filename hints do not authorize a
conversion. The converter itself must support the model architecture and the
checkpoint's exact recipe. WERK validates the resulting tensor descriptors and
payload bounds and requires actual GGUF NVFP4 tensors (type 40). It rejects a
silently dequantized or requantized replacement. Vision models additionally
require successful conversion of their multimodal projector.

Automatic routing does not install or select vLLM for an HF NVFP4 chat
checkpoint. It explains the required GGUF conversion. Existing explicit vLLM
selection remains a separate feature.

A repository label such as `Qwen3.8-Flash-Next-NVFP4` does not by itself prove
architecture, converter, memory or multimodal compatibility. In particular,
ModelOpt NVFP4, W4A16_NVFP4 and mixed recipes must not be confused. Native
Blackwell GGML execution quantizes activations to FP4; a source weight-only
recipe does not guarantee identical arithmetic on that path. Select Marlin
when W4A16 execution is required.

## Persistence and limits

The effective child environment, kernel policy, GPU/build capabilities and
CUDA patch identity enter runtime-state compatibility. Durable chat snapshots
also retain executable and runtime-library identity. A native/Marlin change
cannot reuse a snapshot produced with the other execution path. Existing WERK
slot save/restore and multimodal handling remain in the same adapter.

The Marlin adapter supports dense `MUL_MAT` and MoE `MUL_MAT_ID` for contiguous
NVFP4 weights, with `N % 64 == 0 && K % 128 == 0` or
`N % 128 == 0 && K % 64 == 0`. It does not claim support for arbitrary tensor
layouts, all model architectures, every Marlin quantization format or every
media pipeline. Unsupported operations use the documented automatic fallback
or fail under a forced kernel policy. Model weights remain NVFP4. Marlin uses BF16 activations and block scales,
FP32 reduction and direct FP32 output before the graph applies global weight
scales. The scale expansion preserves small values and avoids an intermediate
FP16 output overflow.

Repacking currently happens per operation, with scratch memory managed by the
GGML CUDA pool. This implementation makes no throughput improvement guarantee.
Native Blackwell execution requires validation on actual Blackwell hardware;
passing a capability or compile check is not a numerical hardware test.
