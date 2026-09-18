# Official llama.cpp CUDA update and DeepSeek diagnostic

The official CUDA runtime was updated from
`f113e02d5ab4c4910709d46e8d81af92ef945289` (2026-07-03) to
`ec928150501c2572fec05cb949061672bb424914` (2026-09-18). The new checkout is
unmodified upstream `ggml-org/llama.cpp`; no expert-cache fork or native patch
was used. Version output: `0.4.1-dev (build 1, commit ec92815)`.

## Installation

The new binary is under the normal managed CUDA directory:

```text
~/.local/share/werk1112/backends/llama-cuda/upstream-ec928150501c/build/bin/llama-server
```

`llama-server.path` now selects that binary. The regular `llama.cpp` checkout
was also advanced to the same revision so a subsequent normal backend install
does not rebuild the July source. The previous build remains at
`backends/llama-cuda/build/bin/llama-server`; its original pointer was saved as
`llama-server.path.before-ec928150501c`. No Werk source rebuild was needed.

Build: Release, CUDA 13.0.48, architecture 86, GNU 11.4, eight build jobs.
The build completed before inference measurements began.

## Measurement conditions

- Installed `werk`, RTX 3090 (24 GiB), Xeon E5-2673 v3, 48 logical CPUs.
- WSL2, 62.57 GiB guest RAM; model files on native ext4.
- Complete `ggml-org/DeepSeek-V4-Flash-GGUF` Q2_K_S, 91.82 GiB across two shards.
- Existing weights externally registered in an isolated temporary model home;
  real conversation history was not modified.
- `--cpu-moe --reasoning off`, context 4096, batch 2048, microbatch 512,
  F16 KV, native default 24 CPU threads, temperature 0, maximum eight output tokens.
- Native logging enabled at verbosity 4. No model-layout or performance flags
  were changed between the two process launches.
- CPU time, RSS, file-I/O counters, faults, and system memory/swap sampled each
  second; GPU metrics sampled every five seconds.
- The first process is not an OS-cache-cold measurement: previous use and import
  inventory reads may have populated caches. No system caches were dropped.

Prompts in the first process:

```text
Remember 37. Describe Rust in one sentence.
Which number? Then describe Rust.
```

After a clean restart with the same isolated persistent session:

```text
Which number? Then describe Rust.
```

## Results

| Metric | First request | Same-process follow-up | After restart |
| --- | ---: | ---: | ---: |
| Request time | 71.69 s | 54.02 s | 156.88 s |
| Time to first token | 45.85 s | 28.77 s | 138.94 s |
| Prompt tokens | 14 | 33 | 52 |
| Cached prompt tokens | 0 | 21 | 0 |
| Newly evaluated prompt tokens | 14 | 12 | 52 |
| Output tokens | 8 | 8 | 8 |
| Output rate reported by Werk | 0.31 tok/s | 0.32 tok/s | 0.45 tok/s |

All requests finished normally with `length`, as intended by the short token
limit. Both Werk processes exited successfully. The restarted process restored
four saved conversation messages and answered with the remembered number 37.
The request timings exclude model initialization and the native persistence
probe; whole-process elapsed times were 334.01 s and 344.51 s.

These are short functional/resource measurements, not a controlled throughput
comparison with the user's earlier 40–46-token responses or the different Qwen
model used in the Mac screenshot.

## CPU placement and storage traffic

Native model buffer allocations were 87,224.56 MiB CPU-mapped (85.18 GiB) and
6,795.51 MiB CUDA (6.64 GiB), plus separate KV and compute buffers. The CPU-side
model mapping exceeds guest RAM. This is static CPU MoE placement, not an
adaptive RAM/VRAM expert cache.

The first process accounted for 171.43 GiB of native-server storage reads and
735,976 major faults. The restarted process accounted for 177.76 GiB and 702,828
major faults. These totals include loading, warmup, capability probes and chats.
Two sampled windows within the first process's actual chat requests, excluding
startup/probes, still showed about 28.36 GiB and 21.63 GiB of additional reads.
Those windows consumed approximately 7.4 and 7.5 CPU cores on average, from a
configured pool of 24 threads. File-backed paging is a substantial contributor
to this workload; it is not merely a one-time load cost.

Linux `read_bytes` includes storage reads caused by mapped-file cache misses;
it does not measure Windows host physical SSD traffic. Major faults are not
synonymous with swap. System-wide swap-in was only 5.99 MiB / 49.32 MiB in the
two phases, while swap-out was 1,543.37 MiB / 2,332.08 MiB. Sampling can miss final
counter increments. GPU figures represent the whole device, including other
applications, so they should not be attributed exclusively to this process.

## Persistence limitation remains

Ordinary in-process prefix reuse works. Durable conversation text also works.
Werk's native KV persistence probe still rejects this DeepSeek route after the
official update:

```text
llama.cpp capability probe did not prove restored cache reuse
```

The native trace shows save/restore followed by an identical-prompt replay that
forces full prompt re-processing because cache data for rollback is unavailable.
Upstream DeepSeek's `seq_pos_min` exposes only the current compressed-state
boundary. Slot restore clears the server's prompt checkpoints. Thus replaying
the identical prompt, which needs at least one token re-evaluated for logits,
has different requirements from extending the restored prefix.

This run does not prove whether restored-prefix continuation can be reused.
The next targeted investigation should compare identical replay with an appended
continuation after the same verified save/erase/restore sequence. Werk's probe
must not simply accept save/restore success or disable reuse verification.
No product-code changes were made as part of this update and measurement.

Captured logs with sanitized paths and machine-readable results are alongside
this note; invalid UTF-8 in native diagnostic output is replaced for readability.
The original one-second resource samples and GPU CSV in `/tmp` were lost when
WSL was shut down to increase its memory limit. The
[98.22 GiB follow-up measurement](../2026-09-18-cuda-upstream-100gb/README.md)
preserves its samples and reproduction harness in the repository.
