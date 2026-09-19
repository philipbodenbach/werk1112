# GLM explicit thinking-off validation

The local `Vontra/GLM-5.3-Flash-MLX-oQ2-MTP` template ignored
`enable_thinking=false`, always appended `<think>`, and defaulted to
`Reasoning Effort: Max`. Its rendered prompts with true and false were identical.
The text adapter now adds a conditional branch in memory: explicit false omits
the effort header and appends `<think></think>`. Default and true retain their
original rendered prompts. Checkpoint files are not modified.

The old `glm-perf` history and its KV cache were purged before validation.
Both runs used the following command (the second restored the first turn):

```sh
env -u WERK_OMLX_REASONING_EFFORT \
  WERK_OMLX_EXPERT_CACHE_MB=auto \
  WERK_OMLX_EXPERT_EXECUTION=grouped \
  WERK_OMLX_NGRAM_CACHE_MB=auto WERK_OMLX_THINKING=0 \
  werk --backend omlx run Vontra/GLM-5.3-Flash-MLX-oQ2-MTP \
  'Explain what a token is in three sentences.' \
  --session glm-perf --persistence --persistence-mode disk \
  --persistence-reuse prefer --max-tokens 256 --temperature 0 --verbose
```

| Metric | Fresh session | Restored follow-up |
|---|---:|---:|
| Total seconds | 41.33 | 42.39 |
| First token seconds | 8.90 | 7.47 |
| Generated tokens | 83 | 83 |
| Cached prompt tokens | 0 | 100 |
| Finish reason | stop | stop |

Both responses contain three complete sentences. Retokenizing the first
visible response with the checkpoint tokenizer gives exactly 83 tokens,
matching the reported generation count. The answer covers general computing
and security tokens; the prompt does not specify language-model tokens.

These are correctness checks, not controlled performance comparisons with the
earlier 256-token runs: history, prompt length, and generated text differ.
Logs: [fresh](glm-thinking-disabled.log),
[follow-up](glm-thinking-disabled-followup.log).

Validation: 80 oMLX Rust tests passed; 8 text-adapter Python tests passed,
6 optional tests skipped. The real checkpoint tokenizer also verified that
default rendering remained byte-identical and false closed the thinking span.
