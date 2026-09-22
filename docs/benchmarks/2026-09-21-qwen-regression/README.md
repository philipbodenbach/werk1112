# Qwen regression investigation — 2026-09-21

The large slowdown after switching from DeepSeek to Qwen was reproduced with
identical Qwen input, restored KV state, native arguments and output. The slow
run incurred substantial file reads and file-page refaults during both prompt
evaluation and generation. The previous model worker had already exited.

There is also a smaller latency difference associated with withdrawing Werk's
speculative snapshot-copy experiment: the restore phase grew approximately
1.1 seconds in the user's matched warm logs, consistent with preparing the copy
serially on the request path. That does not explain the multi-minute model-switch
stalls or slower decoding.

This investigation changes no inference implementation or installed binary.

## Controlled experiment

The comparison uses the installed `release-linux` Werk binary and a fresh
`release-linux` build of committed revision
`fbce8768b22bdb1067603e31947fd402d20336dd`. The current binary includes the subsequent
inspection, warmup and process-lifecycle changes. Both use the same installed
official llama.cpp CUDA executable at revision
`ec928150501c2572fec05cb949061672bb424914`.

The committed baseline is **not** the historical uncommitted speculative-copy
version that produced the user's 12.89-token/s measurement. Both binaries in
this controlled comparison prepare snapshot files serially. This distinction
limits which changes the comparison can isolate.

Hardware: WSL2 with approximately 98 GiB RAM, 24 physical / 48 logical CPU cores
and an RTX 3090 with 24 GiB VRAM. Model files reside in the Linux filesystem.
No compilation ran during inference. OS caches were not forcibly dropped.

The isolated model home contains external manifests pointing to the existing
model files. Original model files, chat histories and installation were left
unchanged. A primer established a compatible private snapshot, then the same
conversation and snapshot were restored before **every measured Qwen call**.
Each of those calls used 905 prompt tokens, restored 883, computed 22 new tokens,
and produced the identical 77-token answer. A new independent session was used
for each DeepSeek call. The primer is excluded from version comparisons; it
was not a verified OS-cache-cold measurement.

Qwen settings:

```bash
WERK_LLAMA_ARGS='--n-cpu-moe 38 --reasoning off' \
werk --backend cuda --threads 24 --threads-batch 24 --ctx-size 4096 \
  --warmup-tokens 0 \
  run vumpt/Qwen3.8-Flash-Next-GGUF \
  "Explain Rust ownership in three sentences." \
  --verbose --persistence --session qwen-startup-v2 --stream \
  --max-tokens 256 --temperature 0 --seed 17
```

The harness additionally selected its isolated `--model-home`, requested JSON
stream events, enabled existing native log forwarding, and pinned the native
executable. It sampled native child I/O, faults, RSS, system VM counters and
pressure approximately every 0.5 seconds, and GPU telemetry every 2 seconds.
Brief help/version processes were distinguished from the model-serving worker.

## Measurements

See [structured measurements](summary.json) and [completion records](results.json).
TTFT below includes preparation and snapshot restoration. Read totals are peak
sampled native-process `read_bytes`, so the last fraction of a second before
exit may be missed. This counter measures storage-layer reads attributed by the
Linux guest; it does not distinguish a Windows host-cache hit from a physical
SSD access. See the [Linux I/O counter documentation](https://man7.org/linux/man-pages/man5/proc_pid_io.5.html).

| Run | Version / condition | Prepare | Overall TTFT | Decode tok/s | Native reads |
| --- | --- | ---: | ---: | ---: | ---: |
| 01-current-warm | current | 31.61 s | 36.35 s | 10.69 | 17.590 GiB |
| 02-baseline-warm | baseline | 15.11 s | 19.87 s | 10.89 | 0.003 GiB |
| 03-baseline-warm | baseline | 10.24 s | 15.06 s | 12.03 | 0.003 GiB |
| 04-current-warm | current | 10.04 s | 14.71 s | 11.09 | 0.003 GiB |
| 05-current-deepseek | current | 90.23 s | 168.37 s | 0.74 | 159.221 GiB |
| 06-current-qwen-switch | current | 52.40 s | 154.59 s | 0.88 | 106.580 GiB |
| 07-current-qwen-repeat | current | 12.98 s | 19.68 s | 8.84 | 2.293 GiB |
| 08-baseline-deepseek | baseline | 81.13 s | 127.80 s | 1.09 | 137.926 GiB |
| 09-baseline-qwen-switch | baseline | 76.43 s | 189.02 s | 1.44 | 113.651 GiB |
| 10-baseline-qwen-repeat | baseline | 26.63 s | 46.16 s | 3.65 | 29.949 GiB |

The first repeated-Qwen row, `01-current-warm`, still read 17.59 GiB and must not
be treated as fully resident merely because the harness calls it "warm".
The two baseline warm calls achieved 10.89 and 12.03 token/s; current warm calls
achieved 10.69 and 11.09 token/s. The variation within the baseline exceeds the
small difference between the two pairs' averages. This sample does not establish
a small decoding regression or statistical equivalence; it does not reproduce
a multi-fold warm decoding regression between these binaries.

The committed baseline also reproduces the switch stall: `09` took 189.02
seconds to first output, read 113.65 GiB and generated 1.44 token/s. Its repeat
improved to 46.16 seconds and 3.65 token/s but still read 29.95 GiB, so even that
repeat was not fully resident. The current switch/repeat pair read 106.58/2.29
GiB. These differing residency states prevent assigning the exact difference
between the baseline and current switch pairs to code version.

All 11 calls completed successfully. All eight measured Qwen calls had identical
normalized native arguments and the same answer hash. No owned native worker
survived any call; no Werk or llama-server process remained after the experiment.

## Where the model-switch time goes

The current Qwen call immediately after DeepSeek (`06`) and its repeat (`07`)
used exactly the same restored context and answer:

| Phase | After DeepSeek: native reads | Repeat: native reads |
| --- | ---: | ---: |
| Before inference, including startup/restore | 28.49 GiB | 1.07 GiB |
| Prompt evaluation | 47.65 GiB | 0.82 GiB |
| Token generation | 30.40 GiB | 0.41 GiB |

Phase boundaries use nearby telemetry samples and are approximate. System-wide
file refault/reclaim counters during the slow prompt phase increased by about
47.17/48.07 GiB-equivalent; during generation, by 30.25/30.47 GiB-equivalent.
These are 4-KiB-page counter equivalents, not unique bytes or attribution to
specific model tensors. Combined with the native worker's read counters and
major faults, they provide strong evidence of file-page eviction and rereading.
Only about 432 KiB was swapped in during the slow Qwen call. Ordinary swap-in
does not account for its roughly 106.58 GiB of native file reads.

The preceding DeepSeek call read 159.22 GiB despite its files totaling about
91.82 GiB. Large refault counts and continued reads after readiness show that
this was more than a one-time initial load. The repeated Qwen call then needed
only 2.29 GiB. Successful KV reuse does not imply resident model weights: those
are separate caches.

A [read-only `mincore` observation](residency-switch-window.json) near the DeepSeek-to-Qwen transition found
approximately 58.09 GiB of DeepSeek's CPU-expert regions still in the Linux file
cache, compared with only 14.35 GiB of Qwen's 58.50-GiB CPU-expert regions. This
sample read GGUF headers but did not intentionally read or prefetch tensor data;
`mincore` queried residency without faulting the mapped weight pages in. Header
reads can still cause adjacent filesystem readahead. The two file scans were
sequential, not an atomic snapshot; they support competing file-cache residency
without identifying each subsequent eviction.

`MemAvailable` remained high because it includes reclaimable file cache, as
defined in the [kernel documentation](https://docs.kernel.org/filesystems/proc.html#meminfo). Nearly
all guest RAM was file cache and only roughly 0.5–0.7 GiB was physically free.
"Available" does not mean the next model's required pages are present, or that
reclaiming cached pages will be cheap. This is not evidence that the machine
needs 106 GiB of additional anonymous RAM or was swapping the entire model.

GPU allocations were similarly high in the slow call and repeat. The slow call
had lower utilization and frequently lower clocks, consistent with waiting for
CPU/I/O work. Temperature stayed below 45 C. These samples do not completely
exclude GPU memory pressure, host-side WSL effects or clock-related differences.

## Which Werk changes matter

The current effective native command matches the committed baseline after
normalizing ephemeral ports and slot directories: 38 CPU MoE layers, 24/24
threads, `-ngl 999`, context 4096, batch 2048, microbatch 512, one slot, flash
attention `auto`, KV offload, `--no-warmup`, reasoning disabled and identical
native cache options. Request generation, sampling, streaming and persisted
snapshot restore/save paths are unchanged between these two binaries.

Same arguments do not prove actual tensor placement is identical if native
automatic fitting reacts to changing free VRAM. The verbosity-3 logs do not
include a complete tensor allocation report.

The help/version/hash inspection jobs join before the model worker starts.
The signal-cleanup registry does not run an ongoing computation or hold a child
lock across inference. Both versions kill and wait for their child on normal
exit. The new signal handling fixes cleanup after interruption; it cannot keep
the previous model's file cache from competing with the next model after exit.

The historical speculative-copy experiment is a separate comparison:

| Prompt tokens | Speculative copy: native restore phase | Current serial path | Difference |
| --- | ---: | ---: | ---: |
| 513 | 0.070837 s | 1.202533 s | +1.131696 s |
| 611 | 0.073185 s | 1.216808 s | +1.143623 s |

These user runs share prompt sizes, 22 newly evaluated tokens and the same
77-token answer. Copying/checksumming was previously overlapped with startup;
it is now inside the measured restore phase. At 513 tokens native readiness
was virtually unchanged (7.593709 versus 7.597111 seconds). This accounts for
much of the small warm TTFT increase from about 12 to 13 seconds. The historical
speculative binary and identical snapshot were not replayed here, so the exact
amount attributable to the code change remains an estimate. That extra restore
work cannot account for slower token generation after restoration finishes.

The rollback was a conservative isolation step. Its suspected contribution to
the original cold-start slowdown was never established by a controlled test.

Relevant current Werk code:

- [`LlamaServerProcess::spawn`](../../../src/backend/llama_server.rs) waits for
  native readiness before file preparation and reports the phase durations.
- [`ChatPersistence::restore_inner`](../../../src/backend/llama_server/runtime_state/chat_persistence.rs)
  copies/checksums the snapshot before native slot restoration.
- [Runtime identity inspection](../../../src/backend/llama_server/runtime_state.rs)
  joins executable-inspection work before the native model load.
- [Owned child lifecycle](../../../src/backend/llama_process_lifecycle.rs)
  manages normal and signal-triggered shutdown.

## Native loading behavior at the installed revision

These findings come from the installed source tree at the revision above, not
an assumption about another llama.cpp release:

- `src/llama-model.cpp:1734` initializes mappings/prefetch before allocating
  buffers and copying GPU tensors.
- `src/llama-model-loader.cpp:1429` requests file-wide prefetch for the mmap path.
- `src/llama-mmap.cpp:480` uses Linux `MAP_POPULATE` only when there are no lazy
  regions; lines 500–503 advise `WILLNEED` for non-lazy regions.
- Qwen's approximately 32.78-GiB per-layer embedding tensor is lazy. Its file
  therefore does **not** take the unconditional whole-file `MAP_POPULATE` path.
  Roughly 78.18 GiB of non-lazy tensor regions remain, including the approximately
  58.50 GiB of expert regions selected by the 38-layer CPU override.
- CPU tensor data can remain directly file-mapped; subsequent access may fault
  missing pages back in. GPU source pages can also occupy the file cache after
  copying. Unmapping an address range is not a request to evict its file cache.
- Native automatic fitting sets `no_alloc` (`common/fit.cpp:57`); the corresponding
  early return before reading tensor data (`src/llama-model.cpp:1857`) does not
  represent another full model-weight load.

This explains a plausible mechanism for the measured residency churn. It does
not identify a single prefetch instruction as the sole cause or establish which
replacement loading policy would improve all models.

`--load-mode none` is not a demonstrated fast-start fix. A prior recorded native
trial already took **177.17 seconds** in worker startup and **189.18 seconds**
to its first token; see the [earlier startup experiment](../2026-09-20-local-startup/README.md).
It eagerly reads ordinary CPU tensors into owned buffers while lazy tensors
remain mapped. The subsequent user's trial timed out at Werk's 180-second
readiness limit. Sparse verbosity-3 logs alone do not establish a native deadlock.

## Historical evidence and limits

The same earlier user attachment that reached 11.79/12.89 token/s also began
with a 117.02-second TTFT and 4.16-token/s call. The latest attachment begins
at 95.13 seconds and 4.06 token/s, followed by 13.43/13.32/13.76-second TTFTs
and 10.62/10.30/11.03 token/s. Thus the severe first-call/repeat discrepancy
predates the latest lifecycle and warmup changes. The 12–13-token/s observations
are real, but comparing only those peaks with later switching calls confounds
file-cache state with version changes.

The experiment does not hold Windows host file-cache state, SSD behavior,
GPU clock state or free VRAM constant. The model-switch sequences were not
fully counterbalanced. Compare each switch with its own identical repeat;
do not attribute the exact difference between the two switch sequences solely
to Werk version. System VM counters are guest-wide, not per tensor.

The actionable next investigation is the existing native loading/file-residency
path, with separate measurements for first load, repeat and model switch.
Snapshot-copy overlap is a smaller, separately measurable optimization. A new
backend service, generic async conversion, larger timeout or a claimed dynamic
expert cache is not established as a remedy by these results.

Raw native output, arguments and telemetry are retained locally at
`/tmp/werk-qwen-exact-regression/`. Binary hashes and verification results are
recorded in [provenance](provenance.json). The original user attachments used for
the matched historical comparison are `acf599c0-fc8e-4298-b05d-e4a12f4da959`
and `cdb37318-2362-4866-904a-faaace42bbe3`.
