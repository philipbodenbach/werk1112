# Concurrent local startup — 2026-09-20

**Historical experiment:** speculative library/snapshot preparation was removed
on 2026-09-21 after the reported first-call regression described below. The
shared help/version inspection and startup timing breakdown remain. These
measurements do not describe the resulting serial file-preparation path.

The existing llama.cpp startup path now shares a single help/version inspection,
executes independent inspection jobs concurrently, and overlaps library hashing
and private snapshot preparation with native model loading. It retains binary
postflight hashing, model/library namespace validation and native restore checks.
A prepared copy is discarded when compatibility changes. One file worker limits
competition with the native loader; no inference runs in that worker.

## Native CUDA validation

Qwen was run with the same installed official CUDA runtime, 24 threads, 24 batch
threads, context 4096, 38 CPU MoE layers, reasoning off and native warmup disabled.
Each invocation used the same prompt, temperature 0 and seed 17, and continued
the same isolated session. The user model files, installed Werk and user history
were not modified. The first run established the snapshot. The binaries were
then alternated; no compilation ran during these measurements.

| Run | Version | Through first token | Preparation | Native restore phase | Cached / total prompt |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Previous, first/cold | 130.010 s | 76.833 s | 0.000111 s | 0 / 20 |
| 2 | Optimized | 22.840 s | 18.827 s | 0.111 s | 97 / 119 |
| 3 | Previous, warm | 30.946 s | 16.635 s | 10.594 s | 197 / 219 |
| 4 | Optimized | 22.333 s | 18.591 s | 0.101 s | 295 / 317 |

All responses completed with `stop`, all subsequent processes reported restored
prefix reuse, and no synthetic cache probes ran. Cross-version namespace reuse
worked in both directions without a snapshot format migration.

The final optimized run separately reported 0.353 s of runtime inspection,
8.408 s of native readiness, 18.225 s of overlapped file preparation and 0.010 s
of final identity/cache validation. These durations are not additive. File
preparation was slower than native loading in this development build, so the
main thread waited for it. Moving copying/hashing out of the restore phase did
not eliminate that work; the overall first-token measurement includes it.

**Limits:** both binaries were development builds, where hashing is much slower
than in the user's release build. This is a small alternating test with growing
history, not a statistically controlled benchmark; do not compare the cold
first run directly to optimized warm runs or promise the same seconds saved in
release. The warm previous run took 30.946 s to the first token; optimized runs
on either side took 22.840 and 22.333 s. The installed release needs its own
measurement. Native decoding throughput is not the target of these changes.

## Regression checks

The focused suite covers single help/version inspection, changed executable
rejection, library additions and modifications, prepared-copy reuse, abandoned
copy cleanup, cold sessions, corrupted snapshots, actual restored hits versus
live-only hits, incomplete streams, and strict named-state capability probes.
See [raw native results](results.json) for full timings and response metadata.

## Follow-up: reported first-call regression, 2026-09-21

The user's installed release reported 117.019 s to the first token in the first
invocation, followed by 12.005 and 12.540 s. The previous supplied series began
at 19.844 s. On the slow invocation native readiness took 62.898 s and prompt
evaluation 51.839 s despite successful reuse of 393/415 prompt tokens.
Concurrent file preparation took 1.679 s; actual restore took 0.095 s.

These logs locate the delay but do not establish whether native file-cache
state, resource contention or another factor caused it. In particular, warm
alternating development runs were insufficient evidence to dismiss the user's
reported regression. Speculative library hashing and snapshot preparation were
therefore removed from native loading. The shared executable inspection remains;
its parallel jobs finish before the worker is spawned. The lazy, real-request
cache validation rule remains unchanged. This rollback is a conservative
regression isolation step, not a proven fix for all cold model loading delays.
No additional model run was used to claim a resolved first-call latency.
