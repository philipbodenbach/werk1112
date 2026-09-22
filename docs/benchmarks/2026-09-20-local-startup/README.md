# Local llama.cpp startup investigation — 2026-09-20

These measurements describe the earlier synthetic-probe implementation. The
subsequent lazy restore validation change removes that probe from local
`run`/`chat`; the timings below are historical, not measurements of that change.

The user's local-process traces separated previously hidden work:

| Model / invocation | Worker startup | Capability probe | Restore | Run to first token |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek Q2_K_S, first | 99.86 s | 129.08 s | 0.00 s | 244.22 s |
| DeepSeek, second | 37.91 s | 3.97 s | 0.19 s | 44.70 s |
| DeepSeek, third | 15.78 s | 4.21 s | 0.28 s | 22.82 s |
| Qwen Q4_K_M, first shown | 101.58 s | 112.95 s | 2.55 s | 247.97 s |
| Qwen, second | 35.63 s | 6.88 s | 1.92 s | 55.43 s |
| Qwen, third | 25.95 s | 4.06 s | 1.02 s | 34.72 s |

Qwen's first shown request already restored 295 cached tokens. Its slow start
therefore cannot be explained simply by a missing conversation or KV snapshot.
The capability probe also touched cold weights and performed an unnecessary
full replay before checking continuation.

## Implementation changes

- Chat probes now try verified token-prefix continuation immediately after
  save/erase/restore. The unchanged exact-replay path remains for named states
  and as a compatibility path when the tokenizer endpoint is unavailable.
- Successful probes have a bounded, private receipt in the existing session
  cache namespace. Reuse requires the same model/runtime/library/environment/
  argument fingerprint and probe schema. Restore failure invalidates it.
- Fresh upstream slots omit `n_prompt_tokens` before any generation task has
  run, even after restoring state. Restore now requires the exact acknowledged
  token/byte counts and an idle slot; a reported counter must match, but an
  absent counter is not treated as zero. This was found by the native restart
  test; the fake server now reproduces upstream's missing-counter behavior.
- `--warmup-tokens 0` maps to native `--no-warmup` when supported.
- Native logs remain captured. Fatal startup errors reach the CLI directly,
  without a second startup attempt under the conversation-only fallback.

## GLM diagnosis

Both GLM shards exist. The first GGUF header identifies `glm5next`. The installed
official CUDA runtime at revision `ec928150501c2572fec05cb949061672bb424914`
rejects it with `unknown model architecture: 'glm5next'`, including with zero
GPU-offloaded layers. The patched Werk reproduces that error in one startup
attempt. No model, installed runtime, or fork was changed.

The [upstream GLM implementation PR](https://github.com/ggml-org/llama.cpp/pull/27754)
was still open when checked. This is an architecture-support limitation, not a
persistence setting.

## Validation

Tests use separate model aliases and session directories in `/tmp`, referencing
the existing Qwen weights. The installed executable is unchanged; tests use a
local development build. Model/runtime settings are 24 threads, 24 batch threads,
context 4096, 38 CPU MoE layers, reasoning off, and native warmup disabled.
No OS caches were forcibly dropped, and the prompts/context lengths differ
from the user's traces; these runs do not establish a controlled speedup ratio.

### Native results

Three corrected local-process runs all completed and restored real prefix state:

| Run | Startup | Probe | Cached / total prompt | First delta incl. preparation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 67.091346s | 56.013813s | 55 / 80 | 148.518050s |
| 2 | 10.174381s | 0.000181s | 81 / 110 | 33.083448s |
| 3 | 10.290491s | 0.000279s | 112 / 134 | 32.689272s |

Runs 2 and 3 reused their verified capability receipt instead of evaluating a
probe. The reply to the memory check was `37`. These runs reused the test
session after the earlier missing-counter failure, so run 1 starts with existing
history and a 55-token snapshot; it is not a fresh-session benchmark.

An additional `--load-mode none` trial succeeded but took 177.17 seconds in
worker startup and 189.18 seconds to the first delta. Peak child RSS was about
59.7 GiB. The model-only prefill was fast (1.05 seconds), but moving the reads
before prefill did not provide a short startup. It is not selected as a default
or recommended for this user's short local runs. Other models/machines can differ.

All timings above use the development build. In particular, runtime-library and
snapshot hashing is substantially affected by Rust build optimization; these
values must not be presented as a controlled comparison with the user's release
binary. The concrete results are fewer probe evaluations, proven restored hits
after process replacement, and actionable startup errors. Cold model loading is
not eliminated.

Validation: 58 llama.cpp tests and 12 shared run tests (offline, no default
features), plus the native Qwen restarts and the single-attempt GLM failure test.
See [results.json](results.json) for raw completion timings and diagnostics.

## Follow-up: lazy validation on real requests

Local llama.cpp session startup no longer runs synthetic inference, including
when a receipt is missing or malformed. A cold session saves the completed user
state; a later process checks and restores it. Only a completed real request
with consistent positive cached-token usage immediately after a successful
restore records historical observed reuse. Live-only hits, zero/missing usage
and incomplete streams cannot establish disk reuse. Named state capability
probes are unchanged.

Regression validation: 59 llama.cpp tests and 12 shared `run` tests passed.
The request-path fixture asserts exactly one completion per user request and no
synthetic tokenization or slot erasure on healthy cold/warm paths. It also checks
ineffective restores followed by live hits and interrupted response streams.

Two native CUDA processes with a fresh, isolated Qwen session also passed:

| Invocation | Worker startup | Real prompt evaluation | Restore | Run to first token | Cached prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| New session | 65.650 s | 53.337 s | 0.000023 s | 127.010 s | 0 / 23 |
| Next process | 9.869 s | 8.227 s | 10.650 s | 36.770 s | 24 / 53 |

Both reported the synthetic probe as skipped. The first answered `OK`, saved
24 tokens and created no reuse receipt. The second answered `37`, reused all
24 restored tokens and only then recorded observed reuse. The existing model,
installed Werk and user sessions were not modified.

Removing the probe did not eliminate cold inference: the first real prefill now
took 53 seconds. This is consistent with moving cold-weight work from the probe
to the real request, but these traces do not isolate disk I/O from computation.
The development build also spends roughly 10 seconds on snapshot copying/hashing;
these timings are not a controlled comparison against the installed release
build or earlier prompts. This validates the rule and correctness, not a claimed
cold-start speedup. See [raw results](lazy-validation-results.json).
