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

The next optimization target is native batched expert execution over retained
weights, avoiding Python routing and per-expert dispatch, while keeping the
bounded offload cache. The resident-loader optimization addresses startup,
not this repeated decode work. Full-model native parity requires an identical
model on a machine with sufficient RAM or a smaller common checkpoint.
