# Text expert decode comparison — 2026-09-22

This change assembles single-token expert outputs directly in route order.
It removes the zero buffer and per-group gather/scatter work while preserving
the checkpoint quantization, activation, score precision and reduction order.
Each group's weight-dependent outputs are evaluated before its leases close.
Only small evaluated output rows remain for the final concatenation. Duplicate
routes are supported. Multi-token prefill uses the original implementation.

`WERK_OMLX_TEXT_DECODE=legacy` selects the original implementation;
`direct` (default) selects the candidate for the verified GLM/Qwen text offload adapters.
Changing the setting requires a fresh Werk process. The expert cache budget,
retention policy, read concurrency, memory guards and N-gram cache are unchanged.
DeepSeek and CUDA/llama.cpp do not use the new decoder.

`WERK_OMLX_GLM_PROFILE=1` adds cumulative per-layer counter snapshots to the
private expert status and verbose diagnostics. It retains one counter row per
routed layer, with no tensor references or per-call history. It is disabled by
default and was disabled for timing comparisons. These timings overlap and
must not be added; read bytes include OS-cache hits. Qwen does not use this
profiler. Its unweighted direct concat remains lazy, so the internal
`forward_seconds` boundary is not identical to legacy. Compare evaluated outer
timings or request decode rates instead.

## Real-weight layer benchmark

Apple Silicon, 48 GiB unified memory, installed oMLX 0.6.4 environment.
Each implementation holds a bounded pool of `2 × top_k` real experts from one
layer. Both use identical BF16 inputs, changing routes, native activations and
native score precision. Sixty alternating measurements follow twelve warmups.
No logical expert reads occur during the timed warm measurements.

| Model | Legacy median | Direct median | Layer speedup |
| --- | ---: | ---: | ---: |
| GLM | 0.779 ms | 0.758 ms | 2.68% |
| Qwen | 0.823 ms | 0.777 ms | 5.85% |

Both models passed 70 comparisons with bit-identical results and zero absolute
error. These are layer measurements, not predictions of full-model throughput.
Raw samples, quantization/runtime metadata and source hashes are in
[`glm-layer.json`](glm-layer.json) and [`qwen-layer.json`](qwen-layer.json).

```bash
/opt/homebrew/Cellar/omlx/0.6.4/libexec/bin/python3.11 \
  docs/benchmarks/2026-09-22-text-decode/compare_decode.py \
  /path/to/model/files --output /private/tmp/decode-layer.json
```

## Full-model serve comparison

`compare_serve.py` uses one release binary with explicit legacy/direct selection
in ABBA order. Every worker receives two independent identical streaming
requests (`first`, `repeat`), temperature zero, thinking disabled, 96 tokens,
22 GiB explicit expert ceiling, N-gram Auto and persistence disk/prefer.
Effective memory budgets, cached tokens, answer hashes and native phase timings
are retained. Models run sequentially; no status polling occurs during decode.
Existing user caches are not removed. Ten seconds between workers allow macOS
to reclaim the previous GPU allocation.

```bash
cargo build --release --locked --offline
python3 docs/benchmarks/2026-09-22-text-decode/compare_serve.py \
  --binary target/release/werk \
  --model pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit \
  --output /private/tmp/qwen-decode-serve
```

The initial comparison without a reclamation pause stopped at the second
worker's native load guard: the reported 28.04 GB resident estimate exceeded a temporary
23.89 GB dynamic ceiling. It never reached candidate decode and is excluded
from throughput conclusions. Memory guards were not relaxed.

Qwen completed all eight requests with the same 96-token response, identical
expert reads/misses and an unchanged effective 22 GiB expert ceiling. First
requests each had 8,899 expert misses; repeats had zero and reused 31 prompt
tokens. Both candidate repeat measurements exceeded both legacy measurements.

| Qwen request | Legacy mean tok/s | Direct mean tok/s | Change |
| --- | ---: | ---: | ---: |
| First | 6.09 | 6.26 | +2.86% |
| Repeat | 12.06 | 12.66 | +5.01% |

The common GLM/Qwen decoder is retained based on this comparison. This is
evidence for the measured workload, not a guarantee for every prompt, device
or concurrent workload. Full results are in [`qwen-serve.json`](qwen-serve.json).

GLM also completed all eight requests with identical answers and effective
budgets across modes. First-request expert counters match exactly. One direct
repeat has two fewer expert misses (6,720 versus 6,722), or 0.015 GiB fewer
logical reads; the other three repeats match. Its full-model result is
effectively unchanged: legacy repeats ranged from 3.677 to 3.787 tok/s,
while direct repeats reached 3.733 and 3.737 tok/s. The layer improvement does
not produce a measurable overall gain in this I/O-heavy workload. Each repeat
still issues about 49.23 GiB of logical reads. Do not interpret the tiny mean
difference as a confirmed GLM speedup.

| GLM request | Legacy mean tok/s | Direct mean tok/s | Change |
| --- | ---: | ---: | ---: |
| First | 3.368 | 3.358 | −0.30% |
| Repeat | 3.732 | 3.735 | +0.08% |

Full results are in [`glm-serve.json`](glm-serve.json). The optional GLM layer
profile is available for further targeted cache analysis; this change does not
alter retention policy or reserve additional cache memory.

## Correctness and isolation

- 82 targeted Rust tests passed, including embedded-worker bootstrap and
  architecture isolation, private cache lifecycle and allowlisted diagnostics.
- 31 Python/MLX tests passed, including exact decode parity with mixed 2/4/8-bit
  projections, FP16/BF16, duplicate/unsorted routes, forced eviction, prefill
  fallback, failed-read lease cleanup and cached/cold group overlap.
- Native tiny GLM/Qwen models match resident reference prefill and recurrent
  decode; private workers stream and reuse persisted prefixes. A real tiny GLM
  worker also exercises the optional layer profiler.
- Existing tokenizer fixtures were updated to the already established GLM
  thinking-template contract and its special tokens. The production thinking
  implementation was not changed.
- `cargo check --locked --offline` and the release build passed. No DeepSeek,
  shared expert cache implementation, CUDA or llama.cpp source was changed.

```bash
cargo test --locked --offline omlx --lib
env WERK_TEST_MLX_EXPERTS=1 \
  /opt/homebrew/Cellar/omlx/0.6.4/libexec/bin/python3.11 -m unittest \
  src.backend.test_omlx_decode src.backend.test_omlx_glm_profile \
  src.backend.test_omlx_text_offload -q
```
