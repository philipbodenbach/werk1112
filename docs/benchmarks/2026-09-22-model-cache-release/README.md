# Model file-cache release after local worker shutdown

## Observed WSL memory retention and manual validation

After the model workers had exited, WSL reported roughly 93 GiB of Linux file
cache. This was reclaimable inside Linux, but the Windows host still accounted
for nearly 98 GiB in `VmmemWSL`. There was no surviving model worker or locked
memory allocation explaining that footprint.

A manual, file-scoped `POSIX_FADV_DONTNEED` operation on the Qwen weight file
produced the following observations without restarting WSL or globally flushing
the Linux cache:

| Measurement | Before | After |
|---|---:|---:|
| Windows `VmmemWSL` memory, bytes | 104,948,572,160 | 22,374,699,008 |
| Windows `VmmemWSL` PID | 67336 | 67336 |
| Linux cached memory, approximately | 93 GiB | 15 GiB |
| Linux free memory, approximately | 1.7 GiB | 79 GiB |

The unchanged VM process demonstrates that memory could return to Windows
without a VM restart on this machine. These are observed host/guest memory
changes from the manual validation, not the number of bytes promised by the
advice API and not yet a measurement of the integrated teardown path. The
operation did not delete or rewrite the model, conversations or KV snapshots.

## Shared implementation

The implementation follows local model ownership rather than a model name or
architecture. `WERK_MODEL_CACHE_RELEASE=auto` enables it on WSL; `on` also enables
it on other Linux systems, and `off` preserves ordinary OS caching. Non-Linux
platforms use a no-op facade and retain their native cache behavior.

- llama-server retains the selected GGUF shards and optional projector, after
  verifying that the actual native arguments select those paths.
- Embedded Candle, Burn and llama.cpp model owners retain guards alongside their
  weights. Normal destruction releases the weights before attempting advice.
- Owned vLLM workers and the shared Python companion path retain guards for
  their resolved model manifests or selected artifacts. This covers the local
  Transformers, ONNX and media integrations using that companion path.
- `run`, `chat` and `serve` use the same ownership lifecycle. A running chat or
  server retains its model between requests; release happens when the final
  owner or worker ends. Resident companion inventories remain leased until the
  worker ends, including models it may have internally unloaded earlier.
- Externally managed remote endpoints have no locally owned weight cache to
  release.

The guard keeps read-only regular-file descriptors and original file identities.
Shared `flock` leases protect other successfully leased Werk readers, including
readers configured with release disabled. Cleanup attempts a nonblocking
exclusive lease and revalidates the retained file identity before applying
`POSIX_FADV_DONTNEED`. Duplicate inodes are handled once; there is no recursive
directory eviction or system-wide cache operation.

Lease acquisition is bounded and optional. If setup fails, inference can proceed
without that owner's coordination or release guarantee. Other applications do
not participate in Werk's advisory leases. The kernel can retain mapped, dirty
or locked pages, so accepted advice is not a measurement of reclaimed memory.
Cleanup diagnostics report the file bytes for which advice was requested.

CLI interruption first stops and reaps registered owned subprocesses. A
preparation gate and cancellation flag prevent cleanup from racing active CPU
expert prefetch chunks; in-flight work finishes and unmaps before advice.
In-process model loading may not be instantly cancellable, and forced termination
cannot guarantee that embedded model mappings have been destroyed before
advisory cleanup. SIGKILL bypasses cleanup. Independently surviving subprocess
descendants are outside the direct-worker ownership guarantee.

This releases clean weight-file cache after model use; it does not introduce
expert swapping or alter inference/KV semantics. Saved prompt snapshots remain
available, but a later standalone run may need to read its weights again. Use
`WERK_MODEL_CACHE_RELEASE=off` when retaining those pages for repeated runs is
preferred.

## Automated validation

The final serial library run passed **968 tests**, with **62 filtered out**:

```bash
cargo test --no-default-features --offline --lib -- \
  --test-threads=1 --skip backend::omlx
```

Coverage includes file-content preservation, inode deduplication, competing
reader leases, partial setup cleanup, changed-file rejection, regular non-GGUF
assets, file-page residency after advice, and an isolated interrupted preparation
fixture. The interruption fixture verifies cancellation, mapping removal before
cleanup, blocked late child creation and signal exit.

The optional backend feature compilation also passed:

```bash
cargo check --no-default-features \
  --features llama-cpp,llama-fast,burn-cpu --offline
```

The earlier full parallel library run was **not entirely green**: 1,002 tests
passed and 28 failed. Twenty-seven oMLX tests failed through Linux's `E2BIG`
argument limit. Their unchanged launcher builds individual Python `-c` arguments
of 194,751 bytes for probing and 162,283 bytes for supervision; this host's
per-argument limit is 131,072 bytes. Those files and fixtures are unchanged from
baseline `fbce876`. This existing transport problem was not changed in the
cache-release work, and the final serial command excludes the whole oMLX group.

One unchanged chat-session lock test also failed in the parallel run, then passed
in the serial run. A brief inherited-file-lock window during concurrent process
creation is a plausible explanation, but was not independently reproduced.
The serial result must not be presented as a full-suite pass including oMLX.

## Integrated actual-model validation

The final `release-linux` binary passed five real CUDA lifecycle checks using
Qwen3.8-Flash-Next Q4_K_M and upstream llama-server `ec928150501c`. The three
`run` calls used the same model and session, without a model switch. The fixture
then ran a chat prompt followed by `/exit`, and one HTTP completion followed by
SIGINT on `serve`. All cases used 24 generation/batch threads, a 4096-token
context, disabled warmup and `--n-cpu-moe 38 --reasoning off`.

| Lifecycle | Total command time | Free guest RAM afterward | Guest file cache afterward | Qwen weight pages resident afterward | Surviving children |
|---|---:|---:|---:|---:|---:|
| Run 1 | 90.00 s | 91.24 GiB | 3.44 GiB | 0 | 0 |
| Run 2 | 85.92 s | 91.11 GiB | 3.44 GiB | 0 | 0 |
| Run 3 | 87.81 s | 91.11 GiB | 3.44 GiB | 0 | 0 |
| Chat `/exit` | 83.73 s | 90.92 GiB | 3.55 GiB | 0 | 0 |
| Serve SIGINT | 85.37 s | 90.99 GiB | 3.55 GiB | 0 | 0 |

Each command requested file advice for the 110.97-GiB weight file. Actual
residency was separately checked with `mincore`, without faulting pages back in;
zero pages remained in every case. Commands used an isolated store referencing
the existing weights, whose inode, length and modification time were unchanged.
Normal commands returned 0; the deliberately interrupted server returned 130.
No user model, conversation or installed Werk binary was overwritten.

Persistence survived: run 2 restored two saved messages and reused 25/46 prompt
tokens from its saved native snapshot; run 3 restored four messages and reused
53/74. Each completed run saved a new snapshot. Thus releasing weight pages did
not remove the durable KV state or prevent its actual reuse.

These deliberately short eight-output-token requests test cleanup, not sustained
generation throughput. Every run starts with zero resident weight pages, so
the approximately 84–90 seconds include another cold weight read. Reclaiming
the cache necessarily gives up the earlier fast warm standalone starts. Active
chat/server workers still retain their weights between requests.

Raw lifecycle measurements are in [lifecycle-results.json](lifecycle-results.json).
The harness incorrectly treated a PowerShell exit code of 1 for the missing
`vmmem` alias as failure and discarded valid `vmmemWSL` JSON. Its per-command
Windows entries are therefore unavailable, not zero. Separate direct queries
are recorded in [windows-observations.json](windows-observations.json): the same
VM process 67336 fell from 87,073,189,888 bytes (81.09 GiB) during validation to
39,987,392,512 bytes (37.24 GiB) after all commands, then to 9,566,896,128 bytes
(8.91 GiB) at a later post-test observation. Its private-memory count also fell
to 10,679,373,824 bytes (9.95 GiB). This confirms a substantial host working-set
reduction without restarting WSL, while also showing that Windows need not
immediately shrink the VM to the guest's anonymous-memory usage. Total host
reclaim timing remains controlled by Windows/WSL.
