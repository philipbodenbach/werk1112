# DeepSeek CUDA after increasing WSL memory

Raising the WSL memory limit from 62.57 GiB to 98.22 GiB substantially reduced
request latency and file-backed paging in the same short DeepSeek diagnostic.
The model, official llama.cpp revision, prompts, and inference settings are
unchanged from the [previous measurement](../2026-09-18-cuda-upstream/README.md).
No Werk product code was changed for this comparison.

## Conditions

- Windows host: 128 GB installed RAM. WSL configured with `memory=100GB`, then
  shut down and restarted; Linux reports 98.22 GiB total RAM and 25 GiB swap.
- RTX 3090, Xeon E5-2673 v3, 48 logical CPUs; native default 24 inference threads.
- Installed Werk; unmodified official llama.cpp
  `ec928150501c2572fec05cb949061672bb424914`, CUDA 13.0, architecture 86.
- Complete DeepSeek V4 Flash Q2_K_S, 91.82 GiB across two GGUF shards on WSL ext4.
- `--cpu-moe --reasoning off`, context 4096, batch 2048, microbatch 512,
  F16 KV, temperature 0, maximum eight output tokens, native verbosity 4.
- Isolated model registration and persistent chat session. The existing model
  manifest was reused without copying weights or hashing the entire model.
- Two requests in one process, then a third request after restarting Werk.
  Exact prompts and argv are in [report.json](report.json).

The CPU model buffer remains 87,224.56 MiB (85.18 GiB), with another
6,795.51 MiB on CUDA. CPU weights alone exceeded the old WSL RAM limit.
These are static MoE CPU placements, not dynamic expert residency or caching.

## Request results

All six compared responses contain eight output tokens and end with `length`.
Both new processes exited successfully without timing out.

| Metric | Request | 62.57 GiB WSL | 98.22 GiB WSL |
| --- | --- | ---: | ---: |
| Request duration | First | 71.69 s | 4.70 s |
| Request duration | Same-process follow-up | 54.02 s | 3.84 s |
| Request duration | After restart | 156.88 s | 4.78 s |
| Time to first token | First | 45.85 s | 2.65 s |
| Time to first token | Same-process follow-up | 28.77 s | 2.25 s |
| Time to first token | After restart | 138.94 s | 3.57 s |
| Werk output rate | First | 0.31 tok/s | 3.92 tok/s |
| Werk output rate | Same-process follow-up | 0.32 tok/s | 5.03 tok/s |
| Werk output rate | After restart | 0.45 tok/s | 6.61 tok/s |

Prompt counts also match: 14, 33, and 52 total; 0, 21, and 0 cached;
14, 12, and 52 newly evaluated. Output rates rose about 13–16 times in these
short samples. This is not a sustained generation benchmark. Werk divides all
eight output tokens by generation duration; native llama.cpp decode rates use
seven decode iterations, since the first token comes from prefill.

Request times exclude model startup and the persistence probe. Whole-process
times fell from 334.01 to 115.79 seconds for the initial two-request process,
and from 344.51 to 45.73 seconds for the restarted process. The first new launch
still spends about 96 seconds in native model loading before warmup/probing.

## Resource observations

Totals below include model loading, warmup, native persistence probes and chat.

| Metric | Process | 62.57 GiB WSL | 98.22 GiB WSL |
| --- | --- | ---: | ---: |
| Native-server Linux storage reads | First | 171.43 GiB | 94.77 GiB |
| Native-server Linux storage reads | Restarted | 177.76 GiB | 34.55 GiB |
| Native-server major faults | First | 735,976 | 3,186 |
| Native-server major faults | Restarted | 702,828 | 130 |
| System swap-in | First | 5.99 MiB | 0 MiB |
| System swap-in | Restarted | 49.32 MiB | 0 MiB |
| System swap-out | First | 1,543.37 MiB | 0.23 MiB |
| System swap-out | Restarted | 2,332.08 MiB | 0.68 MiB |

There are still additional file-backed reads during chat; the result does not
show that all paging has disappeared. Nevertheless, the much smaller fault
counts and faster identical requests support insufficient guest RAM as a major
cause of the previous slowdown. This comparison does not measure physical DIMM
bandwidth or establish the remaining performance ceiling.

`read_bytes` counts Linux-accounted storage reads, including mapped-file cache
misses; it does not measure physical Windows host SSD traffic. Major faults
are not synonymous with swap. System swap counters include other processes.
One-second sampling can miss final increments and short-lived processes. RSS
includes file-backed/shared mappings; summed RSS may count shared pages twice.
GPU CSV values cover the entire device, including other applications.

The WSL restart also changed OS caches and may have changed other GPU/process
activity. Neither run explicitly dropped caches, so this is a practical
before/after diagnostic, not an isolated laboratory RAM-bandwidth comparison.
The Mac screenshot used a different model and is not directly comparable.

## Persistence

Live prefix caching still works (21 of 33 prompt tokens reused). Restarting
restores four conversation messages and the model recalls the number 37.
Native KV snapshots remain disabled because the capability probe reports:

```text
llama.cpp capability probe did not prove restored cache reuse
```

The logs again show full prompt reprocessing after the probe's native
save/erase/restore sequence. More RAM does not resolve this probe result.
The [previous analysis](../2026-09-18-cuda-upstream/README.md#persistence-limitation-remains)
describes the distinction between identical-prompt replay and continuation of
a restored prefix. The latter has not yet been separately tested.

## Artifacts and reproduction

[report.json](report.json), both phase logs, one-second resource samples, and
[gpu.csv](gpu.csv) are preserved here with user paths and hostname sanitized.
Native diagnostic bytes that are not valid UTF-8 are replaced for readability.
The original temporary directory may disappear on a future WSL restart.

[reproduce.py](reproduce.py) preserves the executed harness with user-home paths
resolved through `Path.home()`. It assumes the installed Werk binary and the
existing `deepseek-v4-flash-q2-k-s` model registration under that home. It writes
only to a fresh `/tmp` output directory and does not drop caches:

```bash
python3 docs/benchmarks/2026-09-18-cuda-upstream-100gb/reproduce.py \
  --runtime "$HOME/.local/share/werk1112/backends/llama-cuda/upstream-ec928150501c/build/bin/llama-server" \
  --output-dir /tmp/werk-cuda-upstream-repeat
```
