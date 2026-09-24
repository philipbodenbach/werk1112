# Werk observability

`werk top` is part of the normal Werk binary on macOS, Linux and Windows. No
separate Python collector or Grafana installation is needed for the terminal UI.
The palette matches the Werk logo: cyan, blue, indigo, violet and pink.

```sh
werk top
werk top --url http://127.0.0.1:11434 --interval-ms 2000
werk top --once --json
werk top --no-animation
werk top --demo
```

`--demo` is explicitly simulated and never contacts a server. `q`, Escape and
Ctrl-C leave the dashboard; Space freezes only the display, arrows select a
request, Tab/Enter opens details, `b` selects a native worker, and `a` toggles animation. The layout changes
for narrow/short terminals and handles live resizing. Updates run independently
of network polling. Connection failures show a disconnected state and retained
last-known values, and reconnect automatically. Graph gaps mean unavailable
samples, not zero throughput.

Use `WERK_API_KEY` for a remote/custom server. For literal loopback targets,
`werk top` also tries the first key in the normal local Werk API-key file.
Automatic local credentials are never sent to remote targets or redirects.
The URL is the server root, without `/v1`. `--once --json` supports scripts and
noninteractive terminals. Authentication errors never print the key.

## Server and data contract

The **server must also run a build containing observability**. Updating the
client does not add endpoints to an already-running older server. Restart that
server only when its existing work is finished. Neither dashboard launch nor
monitoring starts, stops, unloads or reconfigures inference workers.

* `GET /werk/v1/observability`: Werk Protocol 1.0 envelope, schema version 1;
  requires the usual `x-werk-protocol-version: 1.0` and authentication.
* `GET /metrics`: Prometheus text exposition 0.0.4; same API-key authentication,
  no Werk protocol header required.

These are **server-wide operational statistics**, visible to an authenticated
operator. No prompt text, generated text, tool arguments, API keys or document
contents are retained. Model names and opaque request sequence numbers are
visible. Counters reset on server/worker restart. Prometheus labels use backend,
model and worker identity; request IDs and user content are never labels.

The collector shares backend samples across clients for two seconds and permits
only one collection at a time. It does not hold inference/worker registry locks
during network I/O. Detailed request state is bounded to 128 active entries and
64 recent entries; aggregate request accounting continues beyond that limit.
The TUI retains at most 90 samples per graph. Backend metrics are read only from
already-running workers; unsupported values remain absent, not fabricated zeroes.

### Coverage and interpretation

* OpenAI and Anthropic chat routes share generation accounting, including normal
  and extended responses, streaming, tool calls, backend errors and stream cancellation.
  These count generation attempts after request validation, not every HTTP call.
  Token totals come from final backend usage, **not from counting SSE chunks**.
* These common request/timing metrics work across text backends, including
  llama.cpp/CUDA and vLLM. Standalone `werk run`, media jobs and explicit Werk
  prefill/decode routes are not included in these chat counters yet.
* Native **llama.cpp** workers (CUDA, Vulkan, Metal, ROCm and CPU) expose live
  decoded tokens, active slots, context occupancy and cached prompt tokens via
  `/slots`. Werk enables that endpoint on supported workers, including without
  persistence. The TUI shows process RSS, host memory and configured CPU/GPU
  layer placement; GPU memory is device-wide. llama.cpp does not expose oMLX
  expert-cache or disk-read counters, so its panels show context and placement.
* Native live counters also supplement **oMLX**: active/waiting requests,
  expert-cache occupancy/budgets/hits/misses/evictions, n-gram cache, offload
  reads and memory-guard budget reductions. Live worker telemetry is still absent
  for vLLM, Candle, Burn, ONNX, standalone MLX/MLX-VLM, Transformers and legacy
  embedded llama adapters; they provide common completed-request metrics only.
  Auto/preferred/runtime routing forwards telemetry from its concrete backends.
* `decode_tokens_per_second_estimate` uses native decoded-token deltas for
  llama.cpp slots with unchanged task IDs, excluding prefill and counter resets.
  oMLX uses decode-context growth between samples of a single active request
  with unchanged completion count. It is not an exact per-token trace; request/phase
  transitions and missing samples can make it unavailable. Completed request
  decode/prefill rates use the backend's reported token counts and timings.
* Expert hit ratio uses interval hit/miss deltas. No accesses means unavailable.
  Prefix reuse is separate, shown in each completed request's cached-token count.
* Expert read bytes include reads satisfied by the OS file cache. They do not
  measure physical SSD traffic or disk-space growth.
* Physical host/accelerator memory and managed accounting are different scopes.
  The memory panel uses host telemetry and native expert residency. It does not
  claim that expert-cache bytes equal total Metal/VRAM allocation. Linux host
  availability includes reclaimable cache (`MemAvailable`), not only free pages.

## Prometheus and Grafana

1. Put the existing Werk API key, by itself, in a protected `werk-api-key` file
   next to `prometheus.yml` (or adjust `credentials_file` to an absolute path).
   The file is ignored by Git. For an explicitly unauthenticated Werk server,
   remove the `authorization` block instead.
2. Start your Prometheus with `--config.file=prometheus.yml`. The example assumes
   Prometheus runs on the host; adjust the target when using containers or a
   remote host. Choose Prometheus retention for the desired history length.
3. Add that Prometheus instance as a Grafana data source.
4. Import `grafana-dashboard.json` and select the data source when prompted.

The dashboard includes the current estimated decode rate, request concurrency,
errors/cancellations, expert and n-gram caches, offload reads, host memory, budget
reductions, prefix-reuse totals and sample freshness. Empty panels mean the
selected backend does not supply that metric or no interval has been observed.
An idle backend must not look like a guaranteed 100% hit rate.

## Development and release toolchain

```sh
cargo test-observability
python3 utils/observability/validate.py
cargo build --locked --bin werk
python3 utils/observability/smoke.py
cargo run -- top --demo
```

The focused tests cover accounting/cancellation, API authentication, protocol
parity, bounded history, metric escaping, interval resets and responsive terminal
rendering using Ratatui's in-memory test backend. CI exercises macOS, Linux and
Windows without loading models. The normal release packaging scripts include
this directory as `observability/` next to the binary.
