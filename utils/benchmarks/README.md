# Chat benchmark

`chat.py` uses only the Python standard library. It exercises the same OpenAI-compatible streaming chat API used by Open WebUI and werkStation, including repeated prompts and conversations with the actual assistant responses in their history.

```sh
export OPENAI_MODEL=mlx-community/DeepSeek-V4-Flash-2bit-DQ
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
# Set OPENAI_API_KEY through your normal secret handling if authentication is enabled.
python3 utils/benchmarks/chat.py --output /tmp/werk-chat-before.json
```

Run the same command against the candidate server with a different output path. Keep the model, expert cache size, thinking mode, hardware load, and server sampling settings identical. The harness defaults to temperature 0, top-p 0.95, and seed 42; it leaves thinking controls to the server because those differ across backends. These tests assess implementation changes under controlled sampling; also review normal-temperature outputs before claiming a quality improvement.

Set `--temperature`, `--top-p`, and `--seed` to compare sampling profiles on the same fixtures:

```sh
python3 utils/benchmarks/chat.py --temperature 0.6 --top-p 0.95 --seed 42 \
  --output /tmp/werk-chat-temperature-06.json
python3 utils/benchmarks/chat.py --temperature 1 --top-p 0.95 --unseeded \
  --output /tmp/werk-chat-temperature-1-unseeded.json
```

Temperature must be finite and nonnegative; top-p must be finite, greater than 0, and at most 1. `--seed` accepts an integer. `--unseeded` omits the seed from requests and inherits the server's seed behavior; it cannot be combined with `--seed`. The report's `settings` records the controls actually sent, so `seed` is absent for unseeded runs. A profile that improves one model or fixture is not a universal quality fix.

Use `--case rust-english --case conversation` to select cases, `--repeats 3` to change the sample count, `--fixtures PATH` for other tasks, or `--api-key-env WERK_API_KEY` to read a different environment variable. Credentials are never included in the report. Reports contain prompts and answers, so use suitable output storage for private fixtures.

Results are written atomically after every completed request. Exit status 1 means an error, missing answer, token-limit truncation, failed automatic check, or interruption requires review. Language, factual quality, and repetition checks are explicitly marked for manual review. A passing JSON or arithmetic check is not a general quality assessment.

The report includes time to first visible text, total wall time, the actual server usage object, cached prompt tokens when supplied, finish reason, and full answer. Missing token/cache metrics remain `null`; SSE chunks are never counted as tokens. The optional client decode estimate is `(completion_tokens - 1) / (total_seconds - first_text_seconds)`. It includes transport/final-event overhead and assumes the first text chunk represents one token; compare backend decode timings when available.

By default, the estimate remains `null` unless the server explicitly reports `completion_tokens_details.reasoning_tokens: 0` and no reasoning is observed. Some servers or proxies hide reasoning deltas while including their tokens in `completion_tokens`, so `has_reasoning: false` does not establish that reasoning was disabled. Use `--estimate-decode-rate` only when you can establish that completion usage counts visible tokens alone; it permits the estimate when reasoning details are missing and records this assumption in `measurement_options`. Observed reasoning or a positive reported reasoning-token count still suppresses the estimate. For example, an answer containing only `12` with 54 completion tokens could include hidden reasoning and cannot establish visible-token throughput.

`first` and `repeat` identify sample order, not cold/warm cache guarantees. An already running worker may have cached the first prompt. Each repetition starts a fresh copy of the fixture conversation; each subsequent turn carries forward the sampled answer. Compare first-text latency, token-normalized decode estimates, answer quality, cache hits, and truncation separately rather than total duration alone.

`--timeout` sets the socket I/O timeout (default 180 seconds). `--deadline` bounds total streaming time when lines arrive (default 600 seconds); a blocked read can last until its I/O timeout. The server must support SSE plus `stream_options.include_usage` to return token counts; streams without usage still produce timing and answer results. `[DONE]` and a finish reason are required for a successful stream.

```sh
python3 -m unittest discover -s utils/benchmarks -p 'test_*.py'
```
