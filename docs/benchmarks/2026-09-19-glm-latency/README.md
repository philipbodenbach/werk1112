# GLM latency on the local 48-GiB Mac

Checkpoint: `Vontra/GLM-5.3-Flash-MLX-oQ2-MTP`. This is not the mixed-4/8-bit
GLM checkpoint from the external 176-GB native-oMLX benchmark.

Metadata inventory measures 88.594 GiB of routed expert weights and 8.910 GiB
of base weights. Vision (1.050 GiB) and MTP (4.011 GiB) are not loaded by this
text offload path. Base plus experts already exceed physical RAM. Disabling
offload is therefore not an equivalent fully resident benchmark here.

## Reported slow request

The user's persisted `glm-perf` session restored four messages. Its next
request had 562 prompt tokens, of which 291 were cached and 271 evaluated.
It took 34.36 seconds to the first token and 133.09 seconds to generate 256
tokens (1.92 tokens/s), then hit the length limit without finishing its answer.

The 21.651-GiB expert cache recorded 39,818 misses and 36,862 evictions.
Logical reads totaled 313,141,493,760 bytes (291.636 GiB); measured reader
time was 88.602 seconds. Expert forward time was 165.006 seconds of the
reported 167.488-second request. Reader time (and 28.418 seconds of routing
time) is included in forward time, not additional to it. Routing includes
waiting for upstream MLX work, so it is not a pure Python-overhead measure.
Logical reads can hit the OS file cache and are not physical SSD traffic.

The installed chat template does not reference `enable_thinking`. It selects
`Reasoning Effort: Max` unless `reasoning_effort` is `low` or `high`, and always
opens `<think>`. Thus `WERK_OMLX_THINKING=0` did not disable this model's
reasoning. Unsetting reasoning effort was counterproductive for this latency
test. Low effort is still not a universal no-reasoning guarantee.

## Fresh-context low-effort request

`low-fresh.log` records the installed binary, automatic grouped expert offload,
automatic N-gram setting (not applicable to GLM), thinking false, reasoning
low, temperature zero and a 256-token limit. No saved session was used.

```sh
env WERK_OMLX_EXPERT_CACHE_MB=auto \
  WERK_OMLX_EXPERT_EXECUTION=grouped WERK_OMLX_NGRAM_CACHE_MB=auto \
  WERK_OMLX_THINKING=0 WERK_OMLX_REASONING_EFFORT=low \
  werk --backend omlx run Vontra/GLM-5.3-Flash-MLX-oQ2-MTP \
  'Explain what a token is in three sentences.' \
  --max-tokens 256 --temperature 0 --verbose
```

The model completed three sentences in 94 tokens: 11.83 seconds to first
token, 40.02 seconds decode, 2.35 tokens/s, 51.875 seconds reported request
duration. This is not a controlled speedup against the user's request:
history, cache budget and output length differ. Startup is not quantified
by this CLI log (`load duration: 0us`); do not equate its request duration
with total shell wall time. Reader time was still 27.006 seconds and logical
reads 90.095 GiB. Lower reasoning effort reduces wasted output; it does not
solve weight movement.

## Cached expert computation

`expert-layer.json` uses the existing `expert_layer.py` against actual GLM
layer 3 weights (2.109 GiB), BF16 inputs and eight fixed routes. Thirty
alternating warm calls gave medians of 0.453 ms native and 0.618 ms Werk,
a 1.36x ratio with zero warm logical reads. Numerical outputs passed the
benchmark's `atol=rtol=0.02` check. This is a single layer, not native
full-model throughput. It does not support extrapolating an 8x gain from
replacing the cached expert arithmetic.

## Same-worker repeat

`reproduce_warm.py` runs two identical requests in a single server, without
saved conversation history or Werk's additional disk-persistence helper.
Ordinary native caching remains active; counter differences distinguish
prefix reuse from retaining experts. Run sequentially with no other model
process, from the repository root:

```sh
WERK_FLASH_BIN="$(command -v werk)" python3 \
  docs/benchmarks/2026-09-19-glm-latency/reproduce_warm.py \
  latency --same-context --no-persistence
```

| Request | First visible text | Total request | Decode | Logical expert reads |
| --- | ---: | ---: | ---: | ---: |
| First | 11.637 s | 49.482 s | 2.48 tokens/s | 89.963 GiB |
| Same worker, repeated | 9.542 s | 46.236 s | 2.56 tokens/s | 77.952 GiB |

Both returned the identical complete 94-token answer, with zero cached prompt
tokens. These are request times after server readiness, excluding startup.
In the sampled decode windows, logical weight reads remained approximately
0.680 and 0.623 GiB per generated token, taking 0.204 and 0.190 seconds per
token respectively. Retaining the worker helps but does not remove cache churn.
Two observations establish this local diagnostic, not a universal speedup.

## Optimization assessment

- Set reasoning effort explicitly to `low` for this test and avoid reusing a
  session full of truncated reasoning when measuring a fresh prompt.
- Retain a worker using `chat` or `serve` to avoid rebuilding the expert cache
  on every `run`. Conversation persistence does not retain process RAM.
- The existing implementation already uses protected hot-expert retention,
  four parallel demand readers, grouped evaluation and shared attention-fusion
  storage. Recommending these as new fixes would be misleading.
- Further engineering should target avoiding or overlapping cache-miss reads:
  reusable bounded weight buffers and safe I/O/compute overlap are candidates,
  not measured improvements. Cache accounting, leases, native memory admission
  and numerical output need to remain correct.
- GLM MTP/speculative decoding is a larger unimplemented path. The filename
  does not mean it is active; its additional weights also compete with the
  expert cache. No speedup is established here.

No runtime behavior or model template was changed during this diagnosis.

## Implemented follow-up: overlap cached decode with missing-weight reads

Qwen/GLM mixed expert groups now launch bounded parallel reads first, evaluate
cached experts on the owning MLX executor, then consume the missing weights.
Both subsets scatter to the original route positions. All group members stay
leased, the existing staging/cache ceilings remain in force, and reader futures
are drained even when the cached computation fails. Prefill, fully cached/cold
groups, small-budget fallback and DeepSeek's separate executor retain their
existing execution order. No additional weights or speculative routes are read.

The updated reader timing counts foreground scheduling/wait, excluding work
overlapped on the executor. Therefore improvements must be established using
request/decode duration, not simply subtracting old/new `disk_read_seconds`.

The 41 offload tests pass with `WERK_TEST_MLX_EXPERTS=1`, including installed
native Qwen/GLM model comparisons. Added coverage checks that reads really can
remain pending during owner-thread work, owner failure drains readers, small
staging limits skip overlap, and split decode preserves duplicate/unsorted
routes and weighted outputs exactly. The release build succeeds.

`overlap_ab.py` compares saved before/after release binaries in ABBA order,
fresh workers, identical prompt, temperature zero, reasoning low and a 64-token
limit. It records binary hashes and full logs. The GLM run uses a fixed 22-GiB
expert cache and automatic N-gram setting (unused by GLM):

| GLM | Decode rates | Mean |
| --- | --- | ---: |
| Before | 2.47, 2.30 tokens/s | 2.385 tokens/s |
| Overlap | 2.63, 2.63 tokens/s | 2.630 tokens/s |

This is a 10.3% local decode improvement. Mean reported request duration falls
from 39.052 to 36.216 seconds (7.3%); first-token times remain around 12 seconds.
All four generated texts match and all runs read exactly 76,850,135,040 logical
weight bytes. These capped answers are comparisons of identical generated
prefixes, not a completed-answer quality test. The earlier low-effort diagnosis
above separately verifies completed answers. Two runs per binary do not establish
a cross-hardware guarantee. Raw data: `overlap-ab/`.

The same ABBA comparison for Qwen with a fixed 16-GiB expert cache yielded
4.62/4.54 tokens/s before and 4.86/4.79 tokens/s with overlap: means of
4.580 and 4.825 tokens/s (+5.3%). Generated texts match across all four runs;
logical weight reads remain exactly 22,693,611,200 bytes. Raw data:
`overlap-qwen-ab/`. These measurements use four reader threads.

```sh
python3 docs/benchmarks/2026-09-19-glm-latency/overlap_ab.py \
  --before /path/to/before/werk --after target/release/werk \
  --output /tmp/glm-overlap-ab
```

DeepSeek is technically eligible for the same scheduling approach: its separate
`_streamed_module` also waits for `prefetch_experts` before evaluating a group,
and already uses the shared bounded reader and leases. Its BF16-to-FP16
conversion must remain intact. No DeepSeek speedup is measured or enabled by
this follow-up; GLM results cannot be transferred numerically to it.

## Follow-up: resource-bounded reader count

The reader-only `reader_threads.py` sweep uses actual GLM tensor ranges and
4/8/16/16/8/4 threads. It does not control the OS file cache. The first four-thread
round has a 7.189-ms median, versus 2.131 ms on its repeat; treating this warm-up
as a thread-count gain would be misleading. Warm medians are approximately
2.0 ms with eight and 1.8 ms with sixteen readers. `reader-threads.json` retains
all timings, but only full-model comparisons support enabling a new default.

The candidate Qwen/GLM default is `min(16, max(1, os.cpu_count() or 4))`.
This machine has 14 logical CPUs. The thread pool starts threads on demand;
the number of pending tensor reads also limits useful concurrency. Staging
remains bounded at 128 MiB per expert group, independent of thread count.
This is a hardware-based upper bound, not load-aware or SSD-aware autotuning.
DeepSeek remains at four readers pending its own validation.

With overlap enabled in both binaries, `reader-scaling-ab/` repeats the GLM
ABBA comparison at the same 22-GiB expert budget:

| GLM reader count | Decode rates | Mean | Mean first token |
| --- | --- | ---: | ---: |
| 4 | 2.59, 2.61 tokens/s | 2.600 tokens/s | 11.995 s |
| 14 (automatic) | 2.90, 2.87 tokens/s | 2.885 tokens/s | 10.415 s |

That is 11.0% higher decode throughput and 13.2% lower first-token latency.
Mean whole-process wall time falls from 42.115 to 38.469 seconds (8.7%).
All generated text hashes match; each run reads 76,850,135,040 logical bytes.
The 42 offload tests pass, including CPU-count fallback/cap validation and
installed native model comparisons. These are two runs per configuration on
one machine, not a guarantee that every SSD benefits from more readers.

The Qwen reader-count ABBA check uses the same 16-GiB expert cache as above:
4.99/5.03 tokens/s with four readers versus 5.57/5.60 with fourteen. Means are
5.010 and 5.585 tokens/s (+11.5%); mean first-token time falls from 8.300 to
6.265 seconds. All four output hashes match and logical reads remain
22,693,611,200 bytes. N-gram usage stays within the same initial 64-MiB allocation;
the automatic upper ceiling can vary with available memory. Raw data:
`reader-scaling-qwen-ab/`. The resource-bounded reader count is retained for
Qwen/GLM based on these full-model checks, not just the warm reader microbenchmark.
