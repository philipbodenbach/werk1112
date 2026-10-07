# Single-host deployment validation, 2026-10-06

Host: WSL2, NVIDIA RTX 3090, 24576 MiB, compute capability 8.6.
Native runtime: llama.cpp `ec928150501c2572fec05cb949061672bb424914`;
the executable digest and exact arguments/configuration are in the JSON plans.
Werk: working tree based on `8d8ee73a002e8473818d1e86099e0a21b3b51d23`;
results include the then-current change summary. No published release is implied.

Both cases use the **same local** `qwen3.8-27b-ud-q4_k_xl` GGUF checkpoint,
context 2048, batch 256, one sequence, eight CPU helper threads, temperature 0,
seed 42, output cap 16, and the same arithmetic prompt. The actual output was
`42` in all six requests, three completion tokens including terminal tokens.
The model was already installed; nothing was downloaded or converted.

| Case | Cold HTTP TTFT | Warm HTTP TTFT (two repetitions) | Total request latency (cold; warm) | Sampled GPU peak |
|---|---:|---|---|---:|
| GPU layers 999 | 28.106 s | 0.900 / 0.522 s | 28.282; 1.023 / 0.610 s | 20831 MiB |
| GPU layers 10, remaining layers CPU | 30.996 s | 2.062 / 1.918 s | 32.021; 3.128 / 2.771 s | 8081 MiB |

GPU peaks include unrelated host/display allocations, sampled approximately
every 200 ms. Native telemetry retains process RSS and native token timings;
HTTP TTFT includes startup, routing and transport. Cold means a new worker, not
an explicitly flushed OS cache. Warm requests recorded 25 cached prompt tokens.
The native expert cache was not enabled. No multi-GPU speedup, statistically
stable throughput, logit equivalence or broad quality claim follows from this
three-token smoke test. It verifies real placement/offload and transport.

Artifacts:

- [GPU results](single-gpu.json), [exact profile](single-profile.json).
- [CPU-layer results](cpu-layers.json), [exact profile](cpu-layers-profile.json).
- [Three-GPU case skipped](three-gpu-skipped.json): only one GPU available.
- [Final CPU-layer reference run](cpu-layers-quality.json): the final native-template
  path passed arithmetic, strict JSON schema (`{"result":42}`), a forced `add`
  tool call with arguments 17 and 25, and continuation after the tool result.
  Cold TTFT 34.665 s, warm 2.120 / 1.725 s; background GPU allocation differed
  from the earlier matched smoke cases, so do not combine these into a speedup.
- [Later all-GPU admission rejection](gpu-admission-rejected.json): external GPU
  occupancy rose and free VRAM fell below the declared budget. Werk rejected the
  plan; no external process was terminated and no silent CPU fallback occurred.

Reproduction on this host (adjust absolute paths on another host):

```bash
python3 utils/multi_gpu/validate.py \
  --config docs/benchmarks/2026-10-06-single-host-deployments/single-profile.json \
  --out /tmp/single.json --tokens 16 --repetitions 3
python3 utils/multi_gpu/validate.py \
  --config docs/benchmarks/2026-10-06-single-host-deployments/cpu-layers-profile.json \
  --out /tmp/cpu-layers.json --tokens 16 --repetitions 3
```

Add `--quality` to execute the JSON-schema/tool/continuation assertions as well.

Run the multi-device examples only after replacing UUIDs, model IDs, executables
and budgets. Required but unmeasured cases: equal/unequal two-GPU sharding,
independent models, a sharded plus independent model, concurrent GPU replicas,
MoE expert cache, long contexts/mixed prefill/decode, Linux versus WSL and P2P
variants. Software fixtures exercise placement, reservation and process behavior;
they are not substitutes for these hardware checks.
