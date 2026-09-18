# DeepSeek V4 Flash CUDA placement and thread comparison

Moving five layers' routed expert weights to CUDA increased the combined
generation rate from 5.93 to 7.19 tokens/s in the fixed workload below. Reducing
the generation thread count from 24 to 16 or 12 did not help. A separate
persistent-chat check with the user's short Rust prompt produced 6.27–7.16
tokens/s (6.68 combined).

No Werk product code, installed runtime, model files, or user configuration was
changed. All runs use the existing official llama.cpp integration and installed
Werk binary. Test conversations live in isolated temporary model homes.

## Environment and workload

- Official llama.cpp `ec928150501c2572fec05cb949061672bb424914`, CUDA 13.0,
  compiled for architecture 86; RTX 3090 with 24 GiB VRAM.
- Xeon E5-2673 v3, 48 logical CPUs visible to WSL2, 98.22 GiB guest RAM.
- Existing DeepSeek V4 Flash Q2_K_S model on WSL ext4, 91.82 GiB across two
  shards, 43 model blocks with routed experts.
- Context 4096, batch 2048, microbatch 512, F16 KV, reasoning off;
  prompt evaluation uses 24 threads throughout.
- Each tuning variant runs two identical requests, temperature zero,
  `--no-history`, maximum 128 output tokens. All eight responses reached
  128 tokens and finished with `length`.
- Native verbosity 4 and one-second CPU/I/O/memory samples; whole-device GPU
  metrics every five seconds. Runs are sequential, with no concurrent
  llama-server detected before starting a variant.

Prompt:

```text
Explain how Rust ownership and borrowing work. Give a detailed practical example and discuss the tradeoffs.
```

Both requests have 23 prompt tokens. The second request reuses 19 tokens and
evaluates four again. The timing table reports generation, excluding startup
and prompt evaluation. Combined rate is total generated tokens divided by
total generation duration, not the arithmetic mean of the displayed rates.

## Results

| Placement | Generation threads | Request 1 | Request 2 | Combined |
| --- | ---: | ---: | ---: | ---: |
| `--cpu-moe` | 24 | 6.47 tok/s | 5.48 tok/s | 5.93 tok/s |
| `--cpu-moe` | 16 | 5.68 tok/s | 5.13 tok/s | 5.39 tok/s |
| `--cpu-moe` | 12 | 5.37 tok/s | 5.34 tok/s | 5.35 tok/s |
| `--n-cpu-moe 38` | 24 | 7.27 tok/s | 7.10 tok/s | 7.19 tok/s |

The placement change improves combined generation throughput by about 21%
against the 24-thread fixed-workload baseline. These are two samples per
configuration, not a confidence interval or guarantee for other prompts.
The first launch had substantial model-load I/O; subsequent launches benefited
from OS caching. No caches were explicitly dropped. The first baseline had
essentially no further weight reads during generation, so its loading time
must not be mistaken for slower decode performance.

## Placement semantics and memory

`--n-cpu-moe 38` keeps routed expert weights in blocks 0–37 on CPU, allowing
blocks 38–42 to follow Werk's existing GPU layer placement (`-ngl 999`). It
replaces `--cpu-moe`; the two flags must not be combined for this experiment.
The native parser appends tensor overrides, so adding the narrower flag after
the all-CPU override would not undo it.

The GGUF tensor inventory contains three Q2_K routed expert tensors per block,
together 1.96875 GiB per block. Five blocks therefore add 9.84375 GiB of GPU
weights. Native logs confirm CUDA model storage increasing from 6,795.51 MiB
to 16,875.51 MiB; the CUDA compute buffer remains 835.26 MiB. The reported
CPU-mapped buffer span remains 87,224.56 MiB, so this result must not be reported
as a measured reduction of that mapping by 9.84 GiB.

Observed whole-device VRAM peaked at 21,135 MiB in the tuning run and 21,214 MiB
in the chat check, leaving roughly 3.3 GiB on this 24-GiB GPU. These values
include other applications and may change with their activity or a larger
context/batch. GPU utilization likewise includes other applications.

This is static CPU/GPU placement, not dynamic expert caching or swapping.
Thread controls and extra native arguments already exist in Werk; no new
integration or fork is required.

## Persistent chat validation

The separate `moe38-chat` run uses `--persistence` with three repetitions of:

```text
Write me a sentence about the programming language rust.
```

It uses default sampling, a 128-token cap, and an isolated new conversation.

| Metric | Request 1 | Request 2 | Request 3 |
| --- | ---: | ---: | ---: |
| Output tokens | 38 | 45 | 41 |
| Output rate | 7.16 tok/s | 6.69 tok/s | 6.27 tok/s |
| Time to first token | 1.15 s | 1.28 s | 1.33 s |
| Request duration | 6.46 s | 8.01 s | 7.86 s |
| Cached / total prompt tokens | 0 / 14 | 51 / 65 | 109 / 123 |

All three responses were coherent in this smoke check and ended with `stop`.
This is not a numerical correctness/perplexity test of DeepSeek4 CUDA expert
kernels, and does not establish that every historical GPU-expert quality issue
is fixed. Earlier [upstream reports](https://github.com/ggml-org/llama.cpp/issues/25582)
must not be generalized into a demonstrated defect in this tested revision and
quantization either.

Conversation persistence and live prefix reuse work. Native KV persistence
still fails the existing restored-cache-reuse probe, as before. This tuning
does not resolve that independent limitation.

Equivalent interactive command for the installed model:

```bash
WERK_LLAMA_ARGS='--n-cpu-moe 38 --reasoning off' \
werk --backend cuda --threads 24 --threads-batch 24 --ctx-size 4096 \
  chat deepseek-v4-flash-q2-k-s \
  --verbose --persistence --session auto-test
```

## Reproduction and artifacts

The existing diagnostic harness was extended with bounded thread/placement
options and the two workloads above; it does not implement a separate inference
path. [reproduce.py](reproduce.py) uses the installed Werk and existing model
registration, refuses a concurrent llama-server, and requires a fresh output
directory under `/tmp`.

```bash
python3 docs/benchmarks/2026-09-19-cuda-tuning/reproduce.py \
  --runtime "$HOME/.local/share/werk1112/backends/llama-cuda/upstream-ec928150501c/build/bin/llama-server" \
  --output-dir /tmp/werk-cuda-placement-repeat \
  --tuning --generation-threads 24 --prompt-threads 24 --expert-cpu-layers 38
```

Omit `--expert-cpu-layers` for the all-CPU expert baseline. Change only
`--generation-threads` for the thread comparison. Use `--chat-smoke` instead of
`--tuning` to reproduce the persistent short-prompt check.

Each variant directory contains `report.json`, the native/Werk log, per-second
resource samples and GPU CSV. All five processes exited zero without timeout.
User paths and hostname are sanitized; invalid UTF-8 in native diagnostics is
replaced. Saved JSON/JSONL parsing and harness syntax were checked. Product
tests were not rerun because no product code was changed.
