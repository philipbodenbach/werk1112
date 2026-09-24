# Validation, 2026-09-24

Pinned llama.cpp `fc343a84bbd925b37dde3219de35ea0bed50d630`, CUDA 13.0.48, GNU 11.4.0, NVIDIA GeForce RTX 3090 (SM86). The test build remained isolated under `/tmp`; installed/running backends were not replaced.

- Built and linked `llama-server`, `werk-fp4-probe`, `werk-marlin-smoke`, and `werk-nvfp4-smoke`. Server `--version` reported the pinned commit.
- 17 raw Marlin CUDA numeric cases passed against independently decoded FP32 references with BF16-rounded inputs: M1/7/16/17/33/128, both tile configurations, M257/N1024 and MoE M521/N512, empty experts, partial tiles, tiny UE4M3 scales, K8192, scale448, and activation131072. Relative RMSE was below 3e-7; tiny scales and the two large positive range cases were exact.
- 29 real GGML graph cases passed: auto10, forcedMarlin9, ggml10. They covered dense/expert GEMM, gate/up broadcast, down inputs per expert, strided inputs and outputs, external scaling and four computes with changed inputs/expert IDs. CUDA graph warmup completed. The auto-only N96/K64 case exercised real upstream GGML fallback. GGML's Q8-activation path used its own appropriate numerical tolerance, not the BF16 Marlin tolerance.
- Forced native on SM86 and forced Marlin on an unsupported shape failed with the expected diagnostic instead of changing kernel/activation format silently.
- The probe executed a CUDA image and reported `compiled_arch:86`, `probe_cuda_error:0`, Marlin true and native false. With `CUDA_VISIBLE_DEVICES=-1`, its device list was empty.
- Compiled the final Marlin kernel as a mixed SM75/SM86/SM120a fatbin and the final GGML adapter/probe for SM120a. No Blackwell GPU was available for execution.
- Clean and reverse patch application checks and native `git diff --check` passed. Dynamic dependency inspection showed no libtorch or vLLM inference runtime.

This is kernel/adapter correctness validation on synthetic tensors. It is not a full model quality evaluation or a throughput benchmark. Per-operation repacking, supported tensor-layout limits and BF16 activation rounding remain documented in README.md.
