# Local Qwen: native expert execution versus Werk offload

This is a diagnostic on the local **48 GiB** Mac, not a reproduction of the
reported GLM benchmark on a 128/256 GB Mac Studio. No full-model vanilla oMLX
tokens/s result was measured. The same installed MLX SwitchGLU implementation
used by oMLX's Qwen model was measured directly against Werk's expert adapter.

## Full-model Werk measurement

Model: `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`.
Its checkpoint contains about 4.981 GiB base weights, 63.281 GiB experts,
29.802 GiB PLE/N-gram tables and 0.836 GiB vision weights. Base plus experts
alone exceed this machine's physical RAM, even if PLE is offloaded. A native
fully resident comparison on this machine would therefore not be equivalent.

```sh
env -u WERK_OMLX_REASONING_EFFORT \
  WERK_OMLX_EXPERT_CACHE_MB=auto \
  WERK_OMLX_NGRAM_CACHE_MB=auto \
  WERK_OMLX_THINKING=0 \
  werk --backend omlx run \
  pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit \
  'Explain what a token is in three sentences.' \
  --max-tokens 64 --temperature 0 --verbose
```

Observed in `werk-run.log`:

| Metric | Result |
| --- | ---: |
| Prompt tokens | 22 |
| Generated tokens | 64 (length limit) |
| Reported prompt time / TTFT | 9.06 s |
| Reported generation time | 14.35 s |
| Generation rate | 4.46 tokens/s |
| Expert cache budget | 16,491 MiB |
| Expert cache misses | 8,198 |
| Expert cache evictions | 1,944 |
| Logical weight bytes read | 22,665,963,200 |
| Reader time | 11.321 s |
| Expert forward time | 21.712 s |
| Routing time | 3.455 s |

Adapter counters span prompt processing plus generation; they are not
decode-only. Reader/routing times are included in forward time and must not
be added to it. Logical reads include OS file-cache hits: this does not prove
22.7 GB of physical SSD traffic. The CLI reports load duration zero, so this
log is **not** a model-startup benchmark. Forward time accounts for about 93%
of the reported 23.44-second request duration.

## Isolated real expert layer

`expert_layer.py` loads layer 0's actual packed weights (1.318 GiB), preserving
checkpoint quantization. It compares installed native SwitchGLU with the actual
Werk streamed-expert adapter on identical BF16 inputs and ten fixed expert
routes. Both implementations are warmed first; 30 timed calls per implementation
alternate execution order. Outputs match with `atol=rtol=0.02`.

```sh
/opt/homebrew/Cellar/omlx/0.6.4/libexec/bin/python3.11 \
  docs/benchmarks/2026-09-19-omlx-comparison/expert_layer.py \
  /Users/philipbodenbach/.local/share/werk1112/models/pipenetwork-Qwen3.8-Flash-Next-MLX-mixed-4_8bit/files \
  --output docs/benchmarks/2026-09-19-omlx-comparison/expert-layer.json
```

| Warm execution | Median per call |
| --- | ---: |
| Native SwitchGLU | 0.305 ms |
| Werk, all selected experts cached | 0.927 ms |
| Ratio | 3.04x |
| Additional logical reads during warm measurement | 0 bytes |

This establishes adapter overhead independently of weight reads. It is a
single-layer, fixed-route microbenchmark, **not** full-model native throughput
and not a proof of a 3x or 9x end-to-end slowdown. The full Werk run separately
demonstrates substantial weight movement and cache churn. Both mechanisms can
contribute to the reported discrepancy; the original GLM/native versus
Qwen/Werk numbers still cannot quantify their individual contributions.

## Implemented changes and controlled before/after

Automatic Qwen/GLM selection now retains native expert modules if all experts
fit with base weights, auxiliary caches, OS headroom and workspace. Automatic
PLE tables also remain resident when the full combination fits. Positive cache
budgets still force bounded streaming. This branch was verified with Qwen/GLM
fixtures, including real private workers, streaming and prefix persistence;
the 99-GiB checkpoint cannot exercise that branch on this 48-GiB machine.

For offloaded single-token decode, every expert uses the same input directly.
The adapter concatenates the outputs of each leased group and scatters once per
group, instead of gathering the input and scattering output separately for each
expert. Initial zero-buffer evaluation and final unweighted reshape evaluation
are no longer separate synchronization points. Every group is still evaluated
before releasing its expert leases. Weight storage and cache limits are unchanged.

A trial that packed cached weights for native gather kernels on every token was
slower and was discarded. The retained offload optimization does **not** eliminate
all Python dispatch or use native batched expert kernels.

`expert-layer-optimized.json` compares old, new and native implementations in
the same process, with alternating order and identical real weights:

| Warm execution | Median |
| --- | ---: |
| Native | 0.300 ms |
| Old Werk | 0.939 ms |
| New Werk | 0.692 ms |

That is 26.2% less time (1.36x throughput) for this isolated expert call, with zero
warm weight reads. The remaining gap to native is 2.31x for this measurement.

`full_model_ab.py` then ran the old and newly installed release binaries in
before/after/after/before order, each in a fresh worker. Both used exactly
16,384 MiB expert cache, automatic N-gram cache, the same 22-token prompt,
temperature zero, thinking disabled and a 64-token output limit.

| Binary | Decode rates | Mean |
| --- | --- | ---: |
| Before | 4.48, 4.45 tokens/s | 4.465 tokens/s |
| After | 4.73, 4.74 tokens/s | 4.735 tokens/s |

The observed full-model improvement is **6.0%**. All four answers are identical,
and each run reads exactly 22,693,611,200 logical weight bytes. N-gram usage stays
within the same 64-MiB initial allocation. Raw logs and binary hashes are in
`full-model-ab/`. Two observations per binary establish a local result, not a
cross-hardware performance guarantee. This is a before/after Werk comparison,
not full-model vanilla oMLX throughput.

The resident-loader optimization addresses startup, not this repeated decode
work. Full-model native parity still requires an identical model on a machine
with sufficient RAM or a smaller common checkpoint. A future native gather path
for offload needs reusable packed storage with correct shared eviction accounting,
rather than copying the selected weights into a fresh pack at every token.
