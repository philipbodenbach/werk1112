After 330 seconds idle before any inference, the candidate reached its first
token in **2.412 seconds**, compared with **84.047 seconds** for the baseline.
Its first request took 11.026 seconds overall instead of 107.308 seconds. Both
started with zero resident model pages, used identical inference settings and
returned the same answer. These are one before/after run per binary on one
Qwen model, not repeated statistical measurements.

| Measurement | Baseline | Retained candidate |
|---|---:|---:|
| API ready | 78.666 s | 76.066 s |
| GGUF resident at ready | 77.891 GiB | 77.891 GiB |
| GGUF resident before first request | 3.425 GiB | 77.891 GiB |
| Werk model RSS before first request | 0.000 GiB | 58.476 GiB |
| First token | 84.047 s | 2.412 s |
| First request total | 107.308 s | 11.026 s |
| First request HTTP wall | 108.670 s | 11.365 s |
| First request prompt evaluation | 84.039 s | 2.404 s |
| First request decode | 23.261 s | 8.614 s |
| First request native storage reads | 59,811,942,400 B | 112,148,480 B |
| First request native major faults | 62,596 | 1,614 |
| Immediate repeat first token | 0.638 s | 0.656 s |
| Immediate repeat total | 6.508 s | 6.356 s |
| Immediate repeat decode | 5.870 s | 5.700 s |
| Immediate repeat native reads / major faults | 0 B / 0 | 0 B / 0 |
| Startup + first request total, excluding idle | 185.974 s | 87.092 s |
| Resident model bytes after shutdown | 0 | 0 |

The shorter first-request delay did not come from a longer measured startup:
the candidate's API was ready 2.60 seconds earlier. The sum of startup and
reported first-request time, excluding the deliberate idle interval, fell from
185.97 to 87.09 seconds.

In the baseline, full-file residency remained at 77.891 GiB through the sample
at +32.03 seconds after the ready snapshot, then fell to 4.323 GiB at +42.15
seconds. Native model RSS/PSS was only 4.323 GiB at ready and did not fall during
that first large cache loss. Before the first request, file residency and native
model RSS/PSS were both 3.425 GiB: 95.60% of the initially resident file pages had
gone. The worker stayed alive, and no model-cache release diagnostic appeared
before shutdown. Its first request incurred 55.70 GiB of native-process storage
reads and 62,596 major faults, after which residency recovered to 59.063 GiB.

All 33 ready/idle samples in the candidate run remained at exactly
83,635,294,208 resident model bytes (77.891 GiB). Werk's retained model mappings
accounted for 58.476 GiB RSS before the request. Its preparation diagnostic
reported 58.50 GiB mapped for the worker lifetime, reclaimable and not locked.
File residency includes cached pages outside either process's populated page
tables; model-mapping RSS measures that process's resident mappings. These
measurements should not be added together as independent memory allocations.

No cache-loss event comparable to the baseline was observed during the candidate
idle interval. The initiating host/kernel mechanism in the baseline was not
traced, so this pair of runs cannot prove that the candidate experienced and
resisted the same external reclaim event. Separate small-file
[kernel regression tests](../../../src/backend/llama_server/model_prefetch_linux.rs)
exercise `POSIX_FADV_DONTNEED`: prepared pages, including pages already cached
before preparation, remain resident while their populated mapping owner lives
and become releasable after it is dropped. That validates the mapping-lifetime
mechanism without promising residency under arbitrary memory pressure.

All four responses contained 20 prompt tokens and the same 78 output tokens,
ending with `stop`; their SHA-256 hashes match. Both first requests reused zero
prompt tokens, making their KV-cache state comparable. Both immediate repeats
reused 16 tokens, so comparisons between a first request and its repeat also
include KV reuse. The independent file-residency loss and storage-read/fault
counts establish a weight-cache problem that KV reuse alone cannot explain.
Output tokens divided by decode time gives 3.35 → 9.05 tokens/s for the first
request and 13.29 → 13.68 tokens/s for the repeat. Native llama.cpp logs exclude
the first output token from that numerator and report 3.31 → 8.94 and
13.12 → 13.51 tokens/s respectively.

The model was `vumpt/Qwen3.8-Flash-Next-GGUF`, file
`qwen3.8-flash-next-Q4_K_M.gguf` (119,150,722,112 bytes). Both runs selected upstream
llama.cpp `ec928150501c2572fec05cb949061672bb424914`, as recorded in the
[runtime provenance](../2026-09-21-cpu-expert-prefetch/provenance.json).
Linux reported 105,465,180,160 bytes of total memory. Binary SHA-256 values were:

- Installed baseline: `b80605866f0aadedb6481dc8732f2bf97d5401aab8bc7b4f1101e40b2faebd35`.
- Candidate `target/release/werk`: `263e61868a94e5cc07a236abcdbc46e4c8fb225d34329c0c067e3b9f5a627b3f`.

Both commands used CUDA, 24 generation and batch threads, a 4096-token context,
`--warmup-tokens 0`, and `serve --persistence`. Environment settings were
`WERK_LLAMA_ARGS='--n-cpu-moe 38 --reasoning off'`,
`WERK_MODEL_CACHE_RELEASE=on`, `WERK_LLAMA_PREFETCH=auto`, and
`WERK_LLAMA_LOG=1`. Native arguments contained `--no-warmup` and matched after
normalizing ephemeral ports and isolated storage paths. Werk arguments matched
apart from those paths/ports and the deliberate executable change. Both requests
used `Explain Rust ownership in three sentences.`, `max_tokens=256`,
`temperature=0`, `seed=17`, and `stream=false`.

Each harness verified that no Werk or llama-server process was running, then
started one worker with an isolated store referencing the existing weights.
After `GET /v1/models` succeeded, it waited 330.003 seconds in the baseline and
330.375 seconds in the candidate before the first inference. The identical
request followed without an intentional idle interval. Neither run used a WSL
restart, global cache clearing or synthetic warmup. Full-file `mincore` probes
used temporary `PROT_NONE`, `MAP_SHARED` mappings and did not read or fault model
pages back in. Samples occurred every ten seconds during idle; timestamps mark
snapshot start, before `smaps`/`mincore` collection completes.

Storage-read and fault counts are differences between the same native process's
before/after `/proc/io` and `/proc/stat` snapshots. Storage accounting is
process-wide rather than exact per-file attribution. First-token values come
from Werk's response timings; non-streaming HTTP wall time measures the complete
response. Reported load and warmup times were zero for all four requests.

SIGINT stopped each owned parent with exit code 130. No owned processes survived;
final `mincore` measurements found zero resident model pages, and weight-file
inode, size and modification time were unchanged. Both advisory cleanup logs
reported 110.97 GiB across one file, with zero shared, changed or failed files.
Actual zero residency was measured separately from accepted advice.

[baseline.json](baseline.json) and [retained.json](retained.json) contain normalized
commands, requests, all ready/idle observations, counters and cleanup results.
[comparison.json](comparison.json) records the matching-configuration checks,
answer hashes and timing differences. Raw-file hashes preserve provenance;
user paths, process IDs, addresses and ephemeral ports are omitted or normalized.
Reclaimable weight pages and worker lifetime are general mechanisms, so the
implementation follows the shared model lifecycle. Only this Qwen configuration
was measured; equal impact across models or backends is not established.
