# Selective CPU expert preparation during model switches

The implementation uses the existing Linux CUDA llama-server startup path. It
prepares missing file-cache pages belonging to explicitly CPU-placed experts
before starting the unchanged upstream worker. It does not alter placement,
native arguments, or KV snapshot compatibility.

## Initial release comparison

Same candidate release executable, toggling only `WERK_LLAMA_PREFETCH=off|auto`.
Each group ran DeepSeek, Qwen immediately afterward, then Qwen again. All Qwen
calls restored the same frozen history and snapshot: 905 prompt tokens, 883
reported cached tokens, 22 evaluated tokens, and 77 generated tokens. All four
Qwen answers have the same SHA-256. Native arguments match after normalizing
only the per-process port and temporary slot directory. All native children
were gone after every command completed.

| Qwen run | Preparation, s | Prefill, s | First token including preparation, s | Generation, tokens/s | Werk + native reads, GiB |
|---|---:|---:|---:|---:|---:|
| After DeepSeek, off | 63.43 | 114.21 | 179.44 | 1.31 | 115.67 |
| After DeepSeek, auto | 53.63 | 12.08 | 67.67 | 5.34 | 73.48 |
| Direct repeat, off | 20.24 | 17.33 | 39.41 | 9.13 | 25.34 |
| Direct repeat, auto | 24.23 | 3.34 | 29.43 | 10.15 | 19.24 |

The auto switch checked 58.50 GiB of CPU expert ranges and populated 47.02 GiB
of missing pages in 32.51 seconds. That time is included above. The repeat
populated 0.13 GiB in 2.19 seconds. Preparation alone is not the acceptance
metric: the repeat has longer preparation but lower total first-token latency.

This is one sequential comparison, not a distribution of randomized trials.
No Linux or Windows caches were dropped. It establishes an observed improvement
for this switch workload, not a guaranteed cold-start time or permanent page
residency. Storage reads are sampled guest `/proc/<pid>/io` counters, not a
measurement of physical SSD traffic. Both Werk and the native child are counted
so moving file I/O into Werk cannot masquerade as eliminating it. The roughly
0.5-second sampling makes these counters approximate lower bounds. VM refault
counters are system-wide and may count the same page repeatedly.

The first DeepSeek baseline followed an earlier owned-buffer experiment; the
second followed mmap Qwen. Its preparation/prefill split therefore has different
initial cache conditions. Additionally, this initial candidate could not
inventory DeepSeek's I32 routing tensors and fell back to native loading in
0.08 seconds. Do not attribute DeepSeek's differing timings to expert prefetch.
Both Qwen switch measurements nevertheless immediately follow a completed
DeepSeek inference with the same model and native arguments.

## Metadata compatibility and final validation

The initial candidate reused Candle's full GGUF metadata/tensor reader. DeepSeek
has three I32 routing tables, a scalar type that this Candle reader rejects even
though the selected expert tensors use supported quantizations. The final
inventory retains Candle for metadata values and adds a bounded tensor-directory
reader with scalar I32 sizing. It reads no weights and performs no conversion.
This is general GGUF handling, without model-name branches or a dependency fork.

Final hardening also preserves shard identity across reopening, checks NUMA
environment overrides, and refreshes host/container memory limits between
chunks. Final build and workload results are recorded separately from the
initial comparison so the executable provenance is explicit.

The final release build completed successfully. Its three subsequent runs all
completed with prefetch active, no surviving native children, and native arguments
identical to the initial comparison. Both Qwen outputs also retained the exact
answer hash and 905/883/77 prompt/cached/output counts.

| Final validation | Preparation, s | First token including preparation, s | Generation, tokens/s | Werk + native reads, GiB |
|---|---:|---:|---:|---:|
| DeepSeek | 91.68 | 93.97 | 6.84 | 92.19 |
| Qwen after DeepSeek | 49.33 | 62.06 | 6.94 | 84.48 |
| Qwen direct repeat | 22.81 | 29.81 | 8.24 | 21.09 |

DeepSeek now successfully prepared 74.81 GiB of missing CPU expert pages in
55.63 seconds. Its native process then read approximately 17.01 GiB; the sampled
inference-window storage-read delta was zero. It resumed the previous test
session (76 prompt tokens, 65 cached, 72 output), so these timings are not a
matched comparison against the initial fresh DeepSeek prompts. Compilation
also occurred before this final sequence. Do not infer a DeepSeek speedup from
those unmatched runs or assign differences between the two auto sequences to
the metadata/hardening changes alone.

The Qwen switch prepared 58.41 GiB of missing pages in 26.95 seconds; its repeat
prepared 0.15 GiB in 1.30 seconds. Necessary cold I/O remains, and decode throughput
still varies. These measurements support improving model-switch latency, not
a claim of universally faster warm decoding or an 11-second cold load.

Validation: `cargo test --locked --offline --no-default-features
backend::llama_server` passed all **80 tests**. Seven local HTTP fixtures first
failed under the sandbox's socket restriction; the same suite passed outside
that restriction. Targeted `rustfmt --check` for the three new modules and
`git diff --check` pass. Repository-wide `cargo fmt --all -- --check` still
reports pre-existing formatting in `omlx.rs`, `omlx/tests.rs` and unrelated
`cli.rs` tests; these were not changed by this work.

## Rejected native loading alternative

An earlier four-run experiment retained upstream llama.cpp but explicitly used
`--load-mode none --no-host --no-repack`. Qwen preparation took 148.86 seconds
initially, 167.45 seconds after DeepSeek, and 135.10 seconds on a direct repeat.
Decode reached 10.03–12.29 tokens/s, but startup remained expensive. This mode
was not adopted as the default. These runs used fresh histories and are a
loading-policy exploration, not a matched KV comparison with the table above.

## Environment and artifacts

- WSL Linux, about 98 GiB guest RAM; NVIDIA RTX 3090 with 24 GiB VRAM.
- Models on the Linux filesystem; 24 generation and batch threads, context 4096.
- Official llama.cpp revision `ec928150501c2572fec05cb949061672bb424914`.
- Native arguments include `--n-cpu-moe 38 --reasoning off`; native warmup off.
- Standalone persistent streaming `run`, temperature 0, seed 17, maximum 256 tokens.
- Model names: `deepseek-v4-flash-q2-k-s` and `vumpt/Qwen3.8-Flash-Next-GGUF`.
- Isolated model-home with external model references and copied histories;
  user's installed executable, original models and sessions were not modified.

`initial-comparison.json` contains derived measurements; `initial-results.json`
contains raw completion diagnostics and answer hashes. `native-args-normalized.json`
records the argument comparison; `provenance.json` identifies the initial binary.
`owned-buffer-alternative.json` records the rejected native loading experiment.
The initial harness and summary script are included; their absolute paths are
specific to this test machine, not a portable benchmark installation.
`final-validation.json`, `final-results.json`, `final-native-args-normalized.json`
and the final harness/summary scripts record the final release validation. Its
binary and source hashes are included in `provenance.json`.

Linux API semantics: [mincore](https://man7.org/linux/man-pages/man2/mincore.2.html)
samples residency; [MADV_POPULATE_READ](https://man7.org/linux/man-pages/man2/madvise.2.html)
faults readable pages and reports errors without userspace tensor dereferences.
Neither operation pins pages against later reclamation.
