# Qwen through a resident Werk server — 2026-09-20

Model: `vumpt/Qwen3.8-Flash-Next-GGUF`, Q4_K_M, architecture `qwen4exp`.
The installed GGUF declares 48 blocks, 512 experts and 10 active experts per token.
Runtime: official llama.cpp CUDA, revision
`ec928150501c2572fec05cb949061672bb424914`; no runtime changes or forks.
Hardware: RTX 3090 24 GiB, WSL2 with 98 GiB RAM, dual-socket host with 24 physical
cores / 48 logical processors.

One `werk serve` process used 24 generation threads, 24 batch threads,
context 4096, and `--n-cpu-moe 38 --reasoning off`. The model was preloaded with
`serve --model`. Four separate `werk run --server` processes used the same
persistent conversation, streaming NDJSON, temperature 0, seed 17 and maximum
128 output tokens. They authenticated to the server with a temporary test key.
An external binding under `/tmp/werk-qwen-cache-e2e` referenced the installed
weights; the normal conversation store was not used. The test server and its
worker were stopped afterwards.

## Results

Server startup to HTTP readiness took **93.68 seconds**, once before these calls.
The wall-clock first-delta measurements below start before each `run` process
launch; they include client startup, HTTP and prefill, but not the already
completed server startup. Native token rates use native decode duration.

| Run | Prompt | First visible delta | Prompt tokens | Cached | New prefill | Output tokens | Decode rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Remember 37; reply OK | 44.25 s | 23 | 0 | 23 | 2 | 2.57 tok/s |
| 2 | Which number? | 7.37 s | 53 | 24 | 29 | 3 | 3.35 tok/s |
| 3 | Explain Rust ownership in three sentences | 4.67 s | 77 | 55 | 22 | 94 | 7.50 tok/s |
| 4 | Explain Rust borrowing in three sentences | 1.71 s | 192 | 170 | 22 | 73 | 10.38 tok/s |

All processes returned success and `stop`; the first two answers were `OK` and
`37`. Prefix hits were reported by llama.cpp through Werk's API. No native disk
snapshots were uploaded/restored on this route. The last client's internal
first-delta measurement was 1.686 s versus 1.711 s measured outside the process.
Its prefill was 1.671 s, so almost all remaining first-token latency in this
particular warm request was prefill, not client-side cache copying.

The cold first prefill still took 43.70 s. Worker residency avoids repeating
startup, capability probes and snapshot restoration; it does not eliminate
cold-model effects. These varying prompts, growing history and warm-up effects
are **not** a controlled speedup comparison or proof of the hardware maximum.
The two very short answers are particularly poor throughput benchmarks.

GPU memory used after the first request was 23,766 MiB, leaving 560 MiB free
(including other GPU users; 2,307 MiB were already used before loading).
Reducing CPU placement below 38 would add expert weights to an already nearly
full GPU in this environment, so the provided command retains 38 and 24 threads.

See [results.json](results.json) for complete completion objects, native timing
and cache counts, and independent process wall-clock measurements.
