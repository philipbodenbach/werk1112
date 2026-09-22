# Qwen snapshot continuation validation

Model: `vumpt/Qwen3.8-Flash-Next-GGUF`, `qwen4exp`, Q4_K_M.
Runtime: official llama.cpp `ec928150501c2572fec05cb949061672bb424914` on CUDA.
Settings: 38 CPU MoE layers, 24 generation/batch threads, context 4096,
greedy sampling with seed 17. No runtime fork or runtime update was used.

## Native snapshot test

The private capability prompt has 15 tokens. The continuation adds 9 tokens,
using explicit token IDs to preserve the original prefix. Each restore test
first erases the slot and reloads the original saved snapshot.

| Request | Cached tokens | Newly evaluated tokens |
| --- | ---: | ---: |
| Identical prompt after restore | 0 | 15 |
| Extended prompt after restore | 15 | 9 |
| Extended prompt after process restart and restore | 15 | 9 |

The identical-prompt probe incorrectly excluded this runtime from terminal
conversation persistence. It tests rewinding a hybrid state to recompute logits,
which is a stronger requirement than continuing that state with new tokens.

The comparison also checked generated output. A single whole-prompt prefill
produced different greedy output than split prefill. With the same split into
prefix and continuation, the live cache, restored snapshot, and freshly computed
reference produced exactly the same eight-token output. The post-restart output
matched that same split-prefill result. This establishes the tested restore path;
it does not claim general bitwise equivalence across different batch shapes.

See [native-results.json](native-results.json) for responses, token counters,
timings, and the numerical comparison. Cold pages and differing startup costs
make these timings unsuitable as a controlled throughput benchmark.

## Implementation and regression coverage

Terminal `run`/`chat` first tries the original exact-replay probe. On a cache miss,
it erases the recomputed slot, restores the original snapshot again, and tries
an exact token-prefix continuation. Success requires all saved prefix tokens to
be reported as reused, with additional tokens evaluated. It still cleans up the
private probe state on success and failure.

Named Werk Protocol states retain their exact-replay probe. Mock-server tests
cover ordinary caches, continuation-only caches, ineffective disk restores with
working live caches, snapshot corruption, and process replacement.

Validation command:

```bash
cargo test --offline --no-default-features backend::llama_server -- --test-threads=1
```

Result: 55 tests passed. The socket-based fixtures require local TCP permissions.

## End-to-end Werk test

Two separate processes used the patched Werk build, with the same official
llama.cpp executable and weights. The model was registered as an external binding
named `qwen-cache-test` in `/tmp/werk-qwen-cache-e2e`, keeping the normal model home
and user conversations untouched. Both requests used `--persistence --session
restart --max-tokens 12 --temperature 0 --seed 17 --verbose --json` and the CUDA
settings above.

| Run | Prompt | Answer | Prompt tokens | Cached | Evaluated |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | Remember the number 37. Reply only OK. | OK | 23 | 0 | 23 |
| 2, new process | Which number did I ask you to remember? Reply with the number only. | 37 | 53 | 24 | 29 |

Run 1 enabled native KV persistence and saved a 24-token snapshot. Run 2 restored
that snapshot, reported 24 actual prefix hits, and saved the extended 55-token
snapshot. Both returned exit status 0 and finish reason `stop`.
See [run-results.json](run-results.json) for complete stdout/stderr and wall times.
Model startup, the capability probe, and snapshot writes contribute overhead;
this verifies functionality and does not promise faster short one-shot calls.
