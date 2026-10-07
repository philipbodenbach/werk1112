# Single-host deployments: implementation record

Started 2026-10-06 on `feat/multi-gpu-support`, baseline
`8d8ee73a002e8473818d1e86099e0a21b3b51d23`. No applicable AGENTS.md found.
User changes to `.gitignore` and `.vscode/` are preserved. No push or release.

## Plan and evidence

1. Audit existing process/cache/control paths and pinned upstream code.
2. Add a serializable device inventory with explicit visibility resolution.
3. Resolve server-owned deployment profiles into immutable plans and reserve
   resources atomically before native startup.
4. Feed plans into the existing llama.cpp/vLLM launchers and lifecycle.
5. Validate native offload combinations and preserve truthful control capabilities.
6. Evaluate Strata's actual server/model contract; record integration gaps.
7. Route aliases/replicas through existing API sessions, with bounded admission.
8. Run software checks, available hardware cases, and document untested cases.

Existing integration points: `src/backend/runtime_cache.rs` already provides
per-key single-flight startup without a global inference lock;
`llama_process_lifecycle.rs` owns child cleanup;
`llama_server.rs`/`vllm.rs` own transport, state and worker caches;
`runtime_control` owns Prepare/Execute/Commit semantics. The existing
`inference_service/resources.rs` exposes a scalar accelerator capacity, not a
multi-device inventory or admission ledger. Extend these paths rather than
introducing another gateway or tensor engine.

Observed hardware: one NVIDIA RTX 3090, UUID
`GPU-b41bf955-b7e6-6f64-82b2-6b3cf08450e9`, PCI `00000000:03:00.0`,
24576 MiB total. No multi-GPU hardware claim is possible on this host.

Upstream revisions resolved from GitHub on 2026-10-06:

- llama.cpp: `abeada335e2e78bd3fe63febafab7e900ce75810`
- vLLM: `3403e0f176efb90b0d92cc0ff1d350f092ed9e9f`
- Strata: `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`
- Existing Werk CUDA offload pin: `907a73da9a149faa8c42ccde890f1d575586810f`

Status: core CUDA deployment implementation is present. The complete original
target is **not yet accepted**: native multi-GPU hardware, vLLM GPU execution and
the specialized Strata adapter still have outstanding evidence/work.

## Intermediate checks

- Device inventory and immutable profile plans implemented; CUDA numeric masks
  with multiple physical devices and MIG currently fail closed (UUIDs supported).
- Atomic exclusive GPU / shared RAM admission attached to existing process owners.
- Native llama.cpp layer placement and vLLM local MP TP/PP/EP arguments connected.
- Nine deployment tests passed (`cargo test --lib --no-default-features deployments::tests
  -- --test-threads=1`): unequal capacities, visibility, split validation, fingerprints,
  concurrent reservation and rollback. These are software tests, not GPU inference.
- API aliases, replica affinity and explicit control-instance selection connected;
  process/HTTP integration and hardware checks remain in progress.

## Acceptance record

| Stage | Implemented / software evidence | Hardware evidence / remaining scope |
|---|---|---|
| 1: inventory of code and engines | Existing caches/adapters/control retained; [pinned runtime review](multi-gpu-runtime-review.md) | No foreign performance result treated as a Werk measurement |
| 2: devices | CUDA UUID inventory, inherited masks/order, physical/logical distinction, memory, architecture and optional topology; tests include empty/unequal/hidden/reordered devices | RTX 3090/WSL measured; multi-device numeric masks and MIG explicitly rejected; structured ROCm/Vulkan inventory not implemented |
| 3: profiles and admission | Strict JSON; per-device component budgets; explicit/auto placement; atomic exclusive GPU + shared RAM admission; races, rollback and deterministic mixed-group dry-run tested | Operator estimates, not a complete architecture-specific memory estimator; unified memory and GPU sharing explicitly unsupported |
| 4: launchers | Existing llama/vLLM adapters receive immutable child-local plans; binary/version checks and cache identity; group cleanup; controlled concurrent processes, death/restart, retained dead session and failed startup tested | llama on one real GPU passed; no two-GPU or local vLLM execution available here |
| 5: offload | llama CPU layers/experts, pinned native expert-cache flavor, vLLM CPU weights; invalid combinations rejected; runtime-control remains native/unsupported as appropriate | Real CPU-layer offload passed; new multi-GPU MoE/cache combinations not verified; no helper-GPU or managed SSD streaming implementation |
| 6: specialist engines | Current Strata model/server/config contract reviewed; exact code gaps and next step recorded | **Not implemented**: dedicated pack identity and safe config/UUID translation plus native pack validation are still required |
| 7: routing/diagnosis | Aliases, separate profile sessions, deterministic replica affinity, bounded fail-fast admission, exact control-instance header and CLI flag; diagnosis + existing top/Prometheus aggregation; HTTP/control/replica tests passed | Real single-profile HTTP checked; concurrent GPU replica hardware unavailable; topology remains unknown where not measurable |
| 8: checks and measurements | Backend-neutral build; full lib test pass; process/HTTP tests; reproducible existing-model harness and seven config examples | [Recorded GPU/CPU-layer runs](benchmarks/2026-10-06-single-host-deployments/README.md); multi-GPU cases skipped, no broad throughput or logit-equivalence claim |

Commands and outcomes:

- `RUST_MIN_STACK=8388608 cargo test --locked --lib --no-default-features -- --test-threads=1`:
  **1251 passed, 4 ignored**, 162.13 s. The ordinary 2-MiB test-thread stack
  overflowed in the large existing CLI parser test; the larger test stack passes.
- `cargo check --locked --no-default-features --all-targets`: passed.
- `cargo build --locked --no-default-features --bin werk`: passed.
- `python3 utils/observability/validate.py`: 20 Grafana panels and Prometheus
  configuration validated.
- `cargo fmt --all -- --check`, `git diff --check`, Python compile checks: passed
  on the tested implementation; final command results are retained in `/tmp/werk-mgpu-*`.
- Subsequent targeted protocol-client tests verify the deployment header and
  header-injection rejection: **12 passed**. One additional real-socket test
  **passed** after discovering that the existing client's write-half-close could
  make Hyper cancel an asynchronous response; Content-Length already frames the
  request, so the client now waits without half-closing.
- Final native CPU-layer reference cases **passed** arithmetic, strict JSON
  schema, forced tool call and tool continuation. The later all-GPU repeat was
  correctly refused after unrelated GPU occupancy increased; both results are
  retained with the benchmark artifacts.

## Continuation points

1. Run the supplied same/unequal two-GPU, independent, mixed and replica configs
   on actual hardware; qualify vLLM TP/PP/EP with supported safetensors models.
2. Strata: register native pack/tokenizer/profile identities; synthesize restricted
   configuration that preserves UUID visibility; test the specialized HTTP/token
   contract and native numerical paths at the documented source pin. Current gaps
   do not establish that a future adapter is impossible.
3. Extend architecture-specific memory estimation and supported sharing/unified
   admission only with native allocation evidence. The current component values
   are explicit estimates and are not hard memory isolation.
4. Qualify quantized vLLM multi-GPU combinations, prefetch/KV offload, helper GPUs,
   live expert movement and SSD tiers individually before advertising them.
5. Collect longer matched-checkpoint quality/load studies, p95 latency,
   per-worker VRAM, pinned memory and transfer counters. The included arithmetic
   runs are functional smoke evidence, not a scaling study.

No driver/system settings changed. No push, merge or release. Existing user
changes in `.gitignore` and `.vscode/` remain outside this implementation.
