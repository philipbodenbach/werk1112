# NVIDIA DGX Spark

Werk supports NVIDIA DGX Spark as a native Linux aarch64 host and as the
client/router in front of a Spark-compatible vLLM server. Spark is a hardware
target, not a special model format: the concrete checkpoint, quantization and
runtime still have to match.

The source of truth for container images and model-specific vLLM flags is
NVIDIA's current
[Nemotron Spark playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nemotron/README.md).
Those instructions change independently of Werk releases. Do not copy flags
from one Nemotron variant to another.

## Support boundary

| Path | Werk status on one DGX Spark |
| --- | --- |
| Werk CLI/server binary | Native `linux-aarch64-dgx-spark` archive, built for GB10 `sm_121` |
| Official Nemotron 4B GGUF | Text chat through a compatible llama.cpp server; model/runtime dependent |
| Text-only Nemotron-H safetensors | Text chat through local or remote vLLM |
| Qwen2/2.5/3-VL or GLM4V safetensors | Image understanding through a compatible local or remote vLLM version; exact Werk architecture allowlist applies |
| Compatible VLM GGUF plus projector | Image understanding through a llama.cpp server that advertises `--mmproj` |
| Spark-compatible vLLM container | Recommended integration route; Werk connects to its OpenAI endpoint |
| `werk backend install vllm` | Deliberately not offered on Spark; a generic pip wheel is not a verified CUDA 13/ARM64 deployment |
| Nemotron Omni image/audio/video inputs | Not eligible for Werk's current exact Qwen-VL/GLM4V vision allowlist; audio/video chat input remains unsupported |
| TensorRT-LLM or NIM | No native Werk adapter; an OpenAI-compatible vLLM endpoint is the implemented remote contract |
| Model fit estimates | Conservative only: Spark uses unified memory, not independent host-RAM and VRAM pools |

Werk never silently replaces the explicitly selected model. Backend fallback
may try another accepted runtime for the same installed model; it does not
switch from one Nemotron checkpoint to another.

The release archive and runtime path must be built and smoke-tested on real
GB10 hardware before publishing. The repository-level checks cannot validate
CUDA 13 linkage, Spark container compatibility, or model fit on their own.

## Install Werk on Spark

The POSIX installer recognizes Linux `aarch64`/`arm64` and downloads the
`linux-aarch64-dgx-spark` artifact only when the host identifies as DGX
Spark/GB10 and the selected release contains it:

~~~bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/philipbodenbach/werk1112/main/scripts/install.sh)"
werk --version
~~~

Release builders can produce the same target natively on Spark:

~~~bash
cargo +stable build-linux-aarch64
./scripts/package-release.sh linux-aarch64
~~~

See the [build guide](../development/build.md#dgx-spark--linux-aarch64-release-build)
for the CUDA/toolchain requirements. Cross-compiling the CUDA artifact on an
x86 host is not the supported release path.

## Recommended Nemotron-H route: Spark vLLM server

First install a supported text-only Nemotron-H repository in Werk. A short
local name keeps the managed path and served name predictable:

~~~bash
werk pull nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --name nemotron-nano-30b
werk inspect nemotron-nano-30b
~~~

Start the matching vLLM server using the current NVIDIA playbook or the
checkpoint's model card. When the server and Werk share one Spark, the vLLM
container can mount Werk's managed `models/nemotron-nano-30b/files` directory
read-only so the weights are stored only once. Set an explicit
`--served-model-name` in the vLLM command.

Verify the endpoint before involving Werk:

~~~bash
curl -fsS http://127.0.0.1:8000/v1/models
~~~

Then attach Werk to it:

~~~bash
export WERK_VLLM_HOST=127.0.0.1
export WERK_VLLM_PORT=8000
export WERK_VLLM_MODEL=nemotron-nano

werk backend doctor --debug
werk --backend vllm chat nemotron-nano-30b --max-tokens 256 --debug
~~~

`WERK_VLLM_MODEL` is the ID advertised by the server's `/v1/models` response.
It may differ from Werk's installed model ID. If it is omitted, Werk accepts an
exact matching ID or the endpoint's only advertised model; an ambiguous
multi-model endpoint is rejected instead of guessing.

The remote transport is plain HTTP. Keep it on loopback or a trusted private
network and apply network-layer authentication/TLS before exposing it beyond
that boundary. Werk's current remote-vLLM client does not add a vLLM API key.

Reasoning and tool parsers are server configuration. Werk reports a diagnostic
hint for Nemotron, but does not invent a parser because the correct parser and
flags depend on the exact checkpoint and vLLM image. The current Werk vLLM
adapter returns assistant `content`; it does not expose structured reasoning
or tool-call deltas. If a request exhausts its token budget in hidden reasoning
without assistant text, Werk reports that condition instead of returning an
empty successful completion.

## Explicit local vLLM environment

An operator-provisioned Spark-compatible Python environment can be selected
without the managed installer:

~~~bash
export WERK_VLLM_PYTHON=/absolute/path/to/spark-vllm/bin/python
export WERK_VLLM_ARGS='--trust-remote-code --gpu-memory-utilization 0.8'

werk backend doctor --debug
werk --backend vllm chat nemotron-nano-30b
~~~

The arguments above are illustrative, not Nemotron defaults. Use only the
flags required by the selected model card. Werk starts and supervises the local
vLLM server in this mode; a long cold start can be adjusted with
`WERK_VLLM_HEALTH_TIMEOUT_SECONDS`.

## Small GGUF smoke path

The official
[NVIDIA Nemotron 3 Nano 4B GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF)
is the smallest practical text smoke test. It does not require vLLM:

~~~bash
werk pull nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF \
  --file NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf \
  --name nemotron-nano-4b

werk --backend cuda chat nemotron-nano-4b --max-tokens 256 --debug
~~~

Execution still requires a llama.cpp server build that supports the checkpoint
and GB10. A managed `werk backend install llama-cuda` build detects Spark and
uses NVIDIA's documented CMake target `121a-real`. Override it only when an
upstream toolchain explicitly requires another spelling:

~~~bash
export WERK_LLAMA_CUDA_ARCH=121a-real
werk backend install llama-cuda
~~~

CUDA architecture selection for llama.cpp is separate from Werk's own
Rust/Candle release build.

## Diagnostics

Run these before reporting a routing failure:

~~~bash
uname -m
nvidia-smi
werk inspect MODEL
werk backend doctor --debug
werk --backend vllm chat MODEL --max-tokens 32 --debug
~~~

Common failures are intentionally actionable:

- missing Spark vLLM: Werk recommends the supported container/remote endpoint,
  not `werk backend install vllm`;
- wrong served name: inspect `/v1/models` and set `WERK_VLLM_MODEL`;
- unsupported architecture/layout: choose a text-only Nemotron-H safetensors or
  supported GGUF checkpoint;
- Nemotron Omni media input: the current typed vision adapter deliberately
  accepts only the documented Qwen-VL/GLM4V architecture set; use a supported
  checkpoint or its upstream deployment contract instead of relying on name
  similarity;
- cold-start timeout: increase `WERK_VLLM_HEALTH_TIMEOUT_SECONDS` instead of
  changing the inference timeout; Werk polls `/v1/models` for both a supervised
  local vLLM and an explicitly configured remote/container endpoint;
- memory pressure: lower the server's model length, concurrency or GPU-memory
  utilization according to the NVIDIA playbook. Spark's unified-memory
  reporting must not be interpreted as independent host and accelerator pools.
