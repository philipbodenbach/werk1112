# Single-host deployment profiles

Deployment profiles bind installed models to exact native executables and ordered
CUDA GPU UUIDs. They use Werk's existing HTTP gateway, worker caches, process
ownership, sessions, telemetry and runtime control. They are opt-in; ordinary
single-device, non-CUDA and remote-backend invocations retain their existing paths.

This implementation supports exclusive GPU groups. A llama.cpp instance can use
one device or layer splitting; vLLM profiles translate local TP, PP and compatible
EP to the multiprocess executor. Repeating an alias creates replicas on separate
device groups. `auto` conservatively chooses the smallest fitting single device;
it does not run a benchmark search or promise a multi-GPU speedup.

## Start with an installed backend and model

```bash
cargo build --locked --no-default-features --bin werk
target/debug/werk gpus
target/debug/werk list
target/debug/werk backend list
```

If needed, use the existing `werk backend install llama-cuda` or `werk backend
install vllm` installer. Profiles use absolute executable paths; they do not
change discovery pointers. A stock executable and the optional offload fork can
therefore coexist in different profiles. The offload installer still has its
historical discovery-pointer behavior for unprofiled invocations.

Copy an [example](../../examples/deployments/single.json), replacing its model ID,
binary path, build revision, GPU UUID and component budgets with local values:

```bash
target/debug/werk deployment-plan profiles.json
target/debug/werk --deployments profiles.json serve --model assistant \
  --api-key "$WERK_API_KEY"
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Authorization: Bearer $WERK_API_KEY" -H 'Content-Type: application/json' \
  -H 'x-werk-session-id: conversation-1' \
  -d '{"model":"assistant","messages":[{"role":"user","content":"What is 17 + 25?"}],"max_tokens":32}'
```

Omit `--model` to require an explicit alias in every request. Profiles load lazily
on first use. In profile mode, chat requests must name a configured alias or an
exact deployment ID; an installed model name cannot bypass placement. `/v1/models`
lists the aliases. Image/audio/video companion pipelines are outside this profile
manager and must be accounted for as external resource users.

## Configuration and units

JSON schema version 1 rejects unknown fields. Each profile contains:

| Field | Meaning |
|---|---|
| `id`, `alias`, `model` | Unique instance ID, client alias, exact installed model ID |
| `runtime` | `llama_cuda`, `llama_cuda_offload`, or `vllm_cuda` |
| `executable`, `build` | Absolute binary/CLI path and operator-recorded source/wheel revision; executable SHA-256 is verified before startup |
| `gpus` | Ordered full physical GPU UUIDs, restricted by inherited visibility |
| `strategy` | `single`, `layer`, `tensor`, `pipeline`, `expert`, or `auto` |
| `tensor_parallel`, `pipeline_parallel` | vLLM rank dimensions; product must equal device count |
| `split` | Positive llama layer-placement proportions, one per GPU; these are not TP degrees |
| `context`, `batch`, `parallel` | Tokens per sequence, prompt batching tokens, simultaneous request slots |
| `cpu_threads` | Native CPU helpers / OMP thread count per worker |
| `memory` | One component budget per device, in device order; all values in bytes |
| `host_bytes` | Shared host admission budget including CPU weights, pinned staging and runtime memory |
| `offload`, `kv_cpu` | Native mechanism and separate llama CPU KV placement |
| `native_args` | Small validated expert-option allowlist; arbitrary argv is not accepted |

Device components are `weights`, `replicated`, `kv`, `experts`, `compute`,
`overhead`, and `reserve`. They are declared estimates, not measurements. Account
for projectors, prefill buffers, context growth and replicated components. The
planner checks each device separately, never just the sum. Without explicit
llama ratios, resident weight estimates determine the ratios. vLLM uses the
smallest per-device budget fraction across ranks; heterogeneous groups can leave
capacity unused. TP attention/KV head divisibility and PP layer counts are
checked from local model config. Quantized vLLM multi-GPU profiles remain rejected
until their model/build combination has been validated.

Reservations are atomic within one Werk service and are owned by the native
worker, including startup and failure cleanup. GPU ownership is exclusive, RAM
budgets share a service-start envelope, and current free memory is checked again
before startup. Live allocations are already in free-memory measurements and are
not subtracted again. Other programs, including other Werk services, can still
allocate memory. Budgets are admission estimates; llama.cpp has no general hard
per-worker VRAM ceiling. vLLM receives the resolved memory utilization fraction.

The initial profile path targets discrete CUDA on Linux/WSL. Numeric CUDA masks
on multiple devices and MIG are rejected instead of guessed. UUID masks and their
order are supported; container visibility is intersected. Topology, NUMA and PCIe
information remain unknown where unavailable. P2P/NVLink availability is not
inferred from a GPU name or required for layer splitting. Structured ROCm/Vulkan,
unified-memory accounting, shared GPU admission and mixed-vendor sharding are
not implemented by profiles; existing backend paths remain available.

## Offload and state

| `offload.kind` | Execution semantics |
|---|---|
| `none` | Native GPU weight placement |
| `cpu_layers` with `gpu_layers` | llama static layer computation on CPU beyond the specified GPU layers |
| `cpu_experts` with `layers` | llama `--n-cpu-moe`; CPU computation of expert tensors in the first N layers |
| `cpu_weights` with `gib_per_gpu` | vLLM `--cpu-offload-gb`; GiB per GPU, not per service; model/version dependent |
| `expert_cache` with `slots`, `pinned_mib` | Pinned Werk offload fork only; expert slots per tensor, pinned MiB per process |
| `helper_gpu`, `ssd_streaming` | Explicitly rejected: no validated adapter translation |

The expert-cache flavor is limited to
`907a73da9a149faa8c42ccde890f1d575586810f`. Layer splitting is permitted; llama
tensor/row modes are not enabled in structured profiles. The existing fork's
native validation limitations still apply. A mmap page cache is not managed SSD
expert streaming. Weight placement, CPU computation, expert caching and KV
snapshots are distinct mechanisms. Profiles do not manufacture successful expert
promotion/eviction or dynamic budget changes; existing adapters report their
actual capabilities and perform Prepare/Execute/Commit only where supported.

Fingerprints include model identity, binary digest, declared build, ordered
devices, context, strategy, budgets and effective managed options. Device
free-memory samples do not affect identity. llama persisted-state compatibility
also includes the deployment fingerprint and child environment. Replicas do not
share opaque native state. Python package contents behind a vLLM launcher are not
content-hashed; use an immutable environment and update `build` when changing it.

Global `WERK_LLAMA_ARGS` and `WERK_VLLM_ARGS` are rejected in profile mode. Their
existing behavior remains available without profiles. Allowed native flags are
currently llama `--no-warmup`, `--jinja`, `--log-disable`, and vLLM
`--enforce-eager`, `--disable-log-requests`. Native capability probes reject a
missing flag before startup. Worker argv wins over native defaults; profile
launches turn llama automatic fitting off where supported, preserving explicit
placement. No ordinary inference request can change process environment/argv.

## Replicas, control and diagnostics

Use the same alias/model on profiles with different IDs/devices. Stateless
requests choose an available replica. `x-werk-session-id` deterministically binds
a conversation to one replica, scoped to the bearer token and alias. Preserve
the configured replica set during a conversation; changing it can change affinity.
There is no transparent KV migration or replay of a partially streamed request.
After worker death, a new complete request may recreate that same instance and
prefill again. Other groups remain alive.

The queue policy is deliberately fail-fast: capacity zero, HTTP 429 when all
slots are busy (or the affinity-bound replica is busy). Cancel/disconnect uses
the existing transport cancellation; retries are the client's responsibility.
Unix worker groups include vLLM child ranks in shutdown. No network executor or
multi-node discovery is added.

```bash
curl -H "Authorization: Bearer $WERK_API_KEY" \
  http://127.0.0.1:11434/werk/v1/deployments
curl -H "Authorization: Bearer $WERK_API_KEY" -H 'x-werk-deployment: assistant-0' \
  http://127.0.0.1:11434/werk/v1/capabilities
```

All existing WerkProtocol control routes require `x-werk-deployment` in profile
mode. This selects an exact control adapter; aliases cannot accidentally address
the last-used replica. The new deployment diagnosis includes inventory, plans,
reasons, component estimates, worker telemetry and slots. Existing `werk top` and
Prometheus telemetry aggregate workers and bounded deployment gauges; missing
GPU/transfer/expert measurements remain unknown.

The CLI equivalent is `werk runtime --deployment assistant-0 capabilities`
(also available for info, memory and state operations). `WerkProtocolClient`
provides a validated `with_deployment` builder for applications.

## Reproducible checks

```bash
RUST_MIN_STACK=8388608 cargo test --locked --lib --no-default-features -- --test-threads=1
python3 utils/multi_gpu/validate.py --config profiles.json \
  --out /tmp/werk-placement-result.json --repetitions 3 --tokens 32
python3 utils/multi_gpu/validate.py --config three-gpus.json \
  --required-gpus 3 --out /tmp/werk-three-gpu-result.json
```

The harness uses existing models only, starts each exact profile concurrently,
records plans, cold/warm HTTP TTFT and latency, native telemetry, sampled GPU
memory and raw outputs, then stops its service. Insufficient GPU count produces
a recorded skip. Cold means fresh worker; OS cache is not flushed. Peak GPU
memory includes unrelated processes. Compare identical checkpoints,
quantizations, prompts, contexts, sampling and token limits. Run tool/structured
output and session reference cases separately. Transport success and readable
text alone do not establish numerical equivalence. See the
[progress/acceptance record](../multi-gpu-progress.md) for actual results and gaps.
