# AMD Strix Halo

Werk supports AMD Ryzen AI Max/Max+ systems (Strix Halo, `gfx1151`) as native
Linux x86_64 hosts. Strix Halo is a hardware profile, not a model format or a
Werk backend: Werk still routes each model to a compatible external ROCm,
Vulkan or Python runtime.

AMD's current
[ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
and
[RDNA 3.5 system guidance](https://rocm.docs.amd.com/en/develop/reference/system-optimization/rdna3-5.html)
are the source of truth for supported operating-system, kernel, ROCm, PyTorch
and precision combinations. Upstream runtime requirements change independently
of Werk releases.

## Support boundary

| Path | Werk status on one Strix Halo host |
| --- | --- |
| Werk CLI/server binary | Native backend-neutral `linux-x86_64-amd-strix-halo` archive |
| GGUF text models | llama.cpp through ROCm/HIP or Vulkan; concrete checkpoint and upstream build dependent |
| Compatible VLM GGUF plus projector | Image understanding through llama.cpp ROCm/HIP or Vulkan when `llama-server` advertises `--mmproj`; smoke-test the exact model on `gfx1151` |
| Text-only Nemotron-H safetensors | Eligible for a compatible vLLM ROCm server; real `gfx1151` validation remains required |
| Qwen2/2.5/3-VL or GLM4V safetensors | Eligible for typed image understanding through a compatible vLLM ROCm server; exact architecture/version and real-hardware validation required |
| Local vLLM | Use an operator-provisioned ROCm environment or container that explicitly supports `gfx1151` |
| Remote vLLM | Supported through its OpenAI endpoint; declare the ROCm accelerator explicitly |
| Image, video and audio | Werk media companion through an operator-provisioned ROCm PyTorch interpreter; model/package dependent |
| XDNA/NPU | No Werk inference adapter; the Strix Halo profile targets the RDNA 3.5 GPU |
| Native Windows vLLM | Not supported by upstream vLLM; Linux is Werk's primary Strix Halo target |
| Model fit estimates | Conservative: CPU and GPU use the same physical memory, not independent RAM and VRAM pools |

The integration and release tooling are implemented. Publishing the artifact
as hardware-validated requires the smoke gate below to pass on a real Ryzen AI
Max/`gfx1151` machine. Repository tests on another x86_64 host cannot validate
ROCm kernels, shared-memory limits, runtime performance or model fit.

## Install Werk on Strix Halo

The POSIX installer recognizes Linux x86_64 Strix Halo from specific host
signals, including an `AMD Ryzen AI Max` CPU/DMI identity, a matching Radeon
8050S/8060S/8040S identity, or a `gfx1151` ROCm agent. When the selected
release contains the platform archive, it downloads
`linux-x86_64-amd-strix-halo`; other Linux x86_64 systems continue to receive
the generic `linux-x86_64` archive. If an older selected release does not yet
contain the Strix-specific archive, the installer reports that fact and falls
back to its checksum-verified generic Linux x86_64 archive.

~~~bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/philipbodenbach/werk1112/main/scripts/install.sh)"
werk --version
~~~

Release builders produce the profile natively on Strix Halo:

~~~bash
cargo +stable build-linux-strix-halo
./scripts/package-release.sh linux-strix-halo
~~~

The feature bundle is intentionally backend-neutral. Building or installing
Werk does not install ROCm, Vulkan, llama.cpp, vLLM, PyTorch, model weights or
media packages.

## Unified-memory behavior

Strix Halo does not have an independent discrete-VRAM pool. AMD documents
`gfx1151` memory as GPU virtual mappings over physically shared system memory;
the firmware carve-out and the dynamic GTT/TTM mapping limit control how much
can be made GPU-accessible. See AMD's
[RDNA 3.5 memory guidance](https://rocm.docs.amd.com/en/develop/reference/system-optimization/rdna3-5.html).

Consequences for Werk workloads:

- do not add reported host RAM and reported GPU memory as though they were two
  independent capacities;
- CPU offload does not create a second physical memory tier;
- leave room for the operating system, Werk, the runtime, KV cache and media
  encoders instead of sizing from model weights alone;
- use the current AMD guidance to configure GTT/TTM and the required kernel,
  then reboot and re-run the hardware smoke checks.

Werk does not modify firmware, kernel or shared-memory limits.

## GGUF route: llama.cpp ROCm or Vulkan

AMD publishes
[validated llama.cpp binaries for Ubuntu 24.04 Strix and Strix Halo](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1.1/docs/advanced/advancedryz/linux/llm/llamacpp.html).
Werk can build the current upstream server with its managed ROCm target:

~~~bash
werk backend install llama-rocm
werk backend doctor --debug
werk --backend rocm chat MODEL --max-tokens 128 --debug
~~~

When the selected logical ROCm device on a detected Strix Halo host is
`gfx1151`, that managed build defaults CMake `GPU_TARGETS` to `gfx1151` and
enables `GGML_HIP_NO_VMM`. Explicit
`WERK_LLAMA_ROCM_ARCH`, `WERK_LLAMA_HIP_NO_VMM` or upstream build values take
precedence. Werk deliberately does not set llama.cpp's experimental
`GGML_CUDA_ENABLE_UNIFIED_MEMORY` runtime variable: current gfx1151 upstream
behavior has an open
[output-corruption report](https://github.com/ggml-org/llama.cpp/issues/26148),
and llama.cpp treats even a value of `0` as an enabled opt-in because only the
variable's presence is checked. Do not set
`HSA_OVERRIDE_GFX_VERSION` to masquerade as another GPU architecture: the
binary and ROCm stack must support the real `gfx1151` agent.

Alternatively, point Werk at AMD's compatible prebuilt `llama-server`:

~~~bash
export WERK_LLAMA_SERVER_ROCM=/absolute/path/to/llama-server
werk backend doctor --debug
werk --backend rocm chat MODEL --max-tokens 128 --debug
~~~

A Vulkan build can be selected separately:

~~~bash
export WERK_LLAMA_SERVER_VULKAN=/absolute/path/to/llama-server
werk --backend vulkan chat MODEL --max-tokens 128 --debug
~~~

ROCm and Vulkan are distinct explicit routes. Benchmark both with the same
model, context and sampling settings on the target machine; Werk does not
claim that one is universally faster.

For a small Nemotron plumbing check, the official
[NVIDIA Nemotron 3 Nano 4B GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF)
can use the same llama.cpp route when the selected server supports the
checkpoint:

~~~bash
werk pull nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF \
  --file NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf \
  --name nemotron-nano-4b

werk --backend rocm chat nemotron-nano-4b --max-tokens 128 --debug
~~~

## Safetensors route: vLLM ROCm

Upstream vLLM lists Ryzen AI Max/AI 300 (`gfx1151`/`gfx1150`) for Linux and
requires ROCm 7.0.2 or newer for that family. Follow the current
[vLLM ROCm installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html)
instead of assuming that an ordinary CUDA-oriented wheel is compatible.

For an already-running local container or remote server:

~~~bash
export WERK_VLLM_HOST=127.0.0.1
export WERK_VLLM_PORT=8000
export WERK_VLLM_ACCELERATOR=rocm
export WERK_VLLM_MODEL=served-model-id

curl -fsS http://127.0.0.1:8000/v1/models
werk backend doctor --debug
werk --backend rocm chat WERK_MODEL_ID --max-tokens 128 --debug
~~~

An operator-provisioned local interpreter can be selected instead:

~~~bash
export WERK_VLLM_PYTHON=/absolute/path/to/rocm-vllm/bin/python
werk backend doctor --debug
werk --backend rocm chat WERK_MODEL_ID --max-tokens 128 --debug
~~~

Text-only Nemotron-H repositories are eligible for this route when the exact
model is supported by the installed vLLM version. Eligibility is not a blanket
performance or memory-fit guarantee. NVIDIA NVFP4 checkpoints target NVIDIA
hardware and must not be assumed to work on AMD; use a checkpoint and precision
explicitly supported by the selected ROCm runtime. AMD's ROCm 7.2 Ryzen matrix
officially validates FP16; treat other precisions as version- and
model-dependent until confirmed on the host.

## Media companion

Create or select a Python environment from AMD's current ROCm/PyTorch
instructions, verify that it reports `gfx1151`, then give that exact interpreter
to Werk:

~~~bash
/absolute/path/to/rocm-media/bin/python -c \
  'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

export WERK_MEDIA_PYTHON=/absolute/path/to/rocm-media/bin/python
werk doctor --task image-generation
werk doctor --model MODEL --task TASK
~~~

This enables only adapters whose packages, model architecture and task are
compatible. It does not imply that every Diffusers or Transformers pipeline
has working `gfx1151` kernels. The XDNA NPU is not used by this path.

## Hardware release smoke gate

Run this gate on the exact release host before publishing the archive:

~~~bash
uname -m
grep -m1 -i 'model name' /proc/cpuinfo
rocminfo | grep -m1 gfx1151

cargo +stable build-linux-strix-halo
./scripts/package-release.sh linux-strix-halo
tar -tzf releases/werk1112-v<VERSION>-linux-x86_64-amd-strix-halo.tar.gz

werk backend doctor --debug
werk --backend rocm chat SMALL_GGUF_MODEL --max-tokens 32 --debug
~~~

For vLLM and media releases, additionally execute one real request through the
exact ROCm environment that will be documented. For a local interpreter, test
that exact path rather than an unrelated system Python:

~~~bash
"$WERK_VLLM_PYTHON" -c 'import vllm, torch; print(vllm.__version__, torch.version.hip, torch.cuda.get_device_name(0))'
"$WERK_MEDIA_PYTHON" -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))'
~~~

For a remote vLLM route, verify its configured `/v1/models` endpoint instead.
Record the host model, kernel, ROCm, PyTorch, runtime version, model
ID/quantization and peak shared-memory use. Until those requests pass,
describe the corresponding path as implemented but hardware validation
outstanding.

Common failures are intentionally separate:

- no `gfx1151` in `rocminfo`: fix the host ROCm/kernel installation before
  debugging Werk;
- wrong vLLM wheel or container: provision the ROCm build documented by
  upstream vLLM and verify `/v1/models` directly;
- wrong served ID: set `WERK_VLLM_MODEL` to the ID returned by `/v1/models`;
- model rejected by the runtime: choose a checkpoint, quantization and
  precision supported by that exact ROCm runtime;
- out of memory despite a large advertised capacity: inspect the shared
  GTT/TTM limit and reserve headroom rather than adding RAM and VRAM values;
- native Windows vLLM: use supported Linux or a separately hosted endpoint;
  Werk does not claim a native Windows vLLM path.
