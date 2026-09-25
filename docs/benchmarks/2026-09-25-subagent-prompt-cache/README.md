# Interleaved agent prompt-cache regression

The installed CUDA llama.cpp revision `ec928150501c` was tested with
Qwen3.8-Flash-Next Q4_K_M, 38 CPU expert layers and one 4096-token slot.
Only synthetic reviewer prompts were used. The harness launches and reaps its
own native server on an ephemeral loopback port.

| Agent continuation | Cache disabled | RAM cache, 1024 MiB |
| --- | --- | --- |
| A after B | 0 / 410 cached tokens | 407 / 410 cached tokens |
| B after A | 0 / 410 cached tokens | 407 / 410 cached tokens |
| Explicit slot 0 after erase | 0 cached tokens | 0 cached tokens |

Both phases disable idle-slot caching and set slot similarity to zero. The
last request proves an explicitly selected, erased slot cannot silently load a
matching entry from the automatic RAM cache. Cold first requests remain cold.
The RAM cache test budget is smaller than Werk's 8192 MiB default; it is enough
for these short conversations. Timing values include warm-cache differences and
are not a general throughput benchmark. Results are in `results.json`.

Reproduce with the installed runtime and model paths on an idle CUDA device:

```sh
python3 docs/benchmarks/2026-09-25-subagent-prompt-cache/reproduce.py \
  --runtime /path/to/llama-server \
  --model /path/to/qwen3.8-flash-next-Q4_K_M.gguf
```

The profile uses 24 CPU threads and 38 CPU expert layers; adjust the native
arguments for a different machine. This checks native cache behavior directly;
Rust regression tests separately verify Werk's generated arguments, explicit
snapshot-slot requests, legacy compatibility and override validation.
