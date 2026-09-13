"""Header validation tests, plus opt-in real MLX numerical/eviction tests.

WERK_TEST_MLX_EXPERTS=1 runs the hardware tests in an oMLX Python environment.
The ordinary suite imports neither oMLX nor MLX and never reads model weights.
"""

import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("omlx_experts", Path(__file__).with_name("omlx_experts.py"))
experts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experts)


def config():
    return {"model_type": "deepseek_v4", "num_hidden_layers": 1,
            "n_routed_experts": 4, "hidden_size": 64, "moe_intermediate_size": 64,
            "num_experts_per_tok": 2, "quantization": {"bits": 2, "group_size": 32, "mode": "affine"}}


def write_checkpoint(path, *, transform_header=None, transform_config=None):
    cfg = config()
    if transform_config:
        transform_config(cfg)
    (path / "config.json").write_text(json.dumps(cfg))
    header = {}
    offset = 0
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("weight", "scales", "biases"):
            shape = [4, 64, 4 if suffix == "weight" else 2]
            dtype = "U32" if suffix == "weight" else "BF16"
            size = 4 * 64 * shape[-1] * (4 if suffix == "weight" else 2)
            header[f"model.layers.0.ffn.switch_mlp.{projection}.{suffix}"] = {
                "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
            offset += size
    header["model.norm.weight"] = {"dtype": "BF16", "shape": [64], "data_offsets": [offset, offset + 128]}
    offset += 128
    if transform_header:
        transform_header(header)
    raw = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def test_header_only_estimate_separates_expert_base_and_workspace(self):
        write_checkpoint(self.path)
        with patch.object(experts.os, "pread", side_effect=AssertionError("weights must not be read")):
            result = experts.inspect_model(self.path, 8192)
        self.assertEqual(result["expert_count"], 4)
        self.assertEqual(result["base_bytes"], 128)
        self.assertEqual(result["expert_bytes"], 18432)
        self.assertEqual(result["largest_expert_bytes"], 4608)
        self.assertEqual(result["resident_estimate_bytes"], 128 + 8192 + 1024**3)

    def test_rejects_cache_too_small_for_one_complete_expert(self):
        write_checkpoint(self.path)
        with self.assertRaisesRegex(ValueError, "at least one"):
            experts.inspect_model(self.path, 4607)

    def test_rejects_other_architecture_and_custom_model_code(self):
        for change in (lambda c: c.update(model_type="deepseek_v3"),
                       lambda c: c.update(model_file="custom.py")):
            write_checkpoint(self.path, transform_config=change)
            with self.assertRaisesRegex(ValueError, "native deepseek_v4"):
                experts.inspect_model(self.path, 8192)

    def test_rejects_custom_loader_that_would_bypass_streaming(self):
        write_checkpoint(self.path, transform_config=lambda c: c.update(quantization_config={"quant_method": "paroquant"}))
        with self.assertRaisesRegex(ValueError, "custom quantization loader"):
            experts.inspect_model(self.path, 8192)

    def test_rejects_embedded_speculative_drafter(self):
        write_checkpoint(self.path, transform_config=lambda c: c.update(dspark_block_size=8))
        with self.assertRaisesRegex(ValueError, "speculative drafters"):
            experts.inspect_model(self.path, 8192)

    def test_rejects_unrecognized_expert_layout(self):
        def change(header):
            key = next(iter(header))
            header[key.replace("switch_mlp.gate_proj", "experts.0.w1")] = header.pop(key)
        write_checkpoint(self.path, transform_header=change)
        with self.assertRaisesRegex(ValueError, "packed expert tensor"):
            experts.inspect_model(self.path, 8192)

    def test_rejects_overlapping_ranges(self):
        def change(header):
            names = list(header)
            header[names[2]]["data_offsets"] = header[names[1]]["data_offsets"]
        write_checkpoint(self.path, transform_header=change)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            experts.inspect_model(self.path, 8192)

    def test_rejects_out_of_file_range(self):
        def change(header):
            header["model.norm.weight"]["data_offsets"] = [10**9, 10**9 + 128]
        write_checkpoint(self.path, transform_header=change)
        with self.assertRaisesRegex(ValueError, "shard size"):
            experts.inspect_model(self.path, 8192)

    def test_rejects_geometry_and_quantization_mismatch(self):
        for change in (lambda c: c["quantization"].update(bits=3),
                       lambda c: c["quantization"].update(group_size=64)):
            write_checkpoint(self.path, transform_config=change)
            with self.assertRaises(ValueError):
                experts.inspect_model(self.path, 8192)

    def test_duplicate_json_tensor_names_are_rejected(self):
        write_checkpoint(self.path)
        path = self.path / "model.safetensors"
        raw = b'{"same":{},"same":{}}'
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            experts.inspect_model(self.path, 8192)

    def test_model_identity_and_tier_scoped_pagination(self):
        write_checkpoint(self.path)
        manager = experts.ExpertManager(experts.Checkpoint(self.path, 8192))
        class Model:
            pass
        model = Model()
        manager._model_ref = weakref.ref(model)
        page = manager.list_experts({"limit": 2, "allow_experimental": True})
        self.assertEqual(len(page["experts"]), 2)
        self.assertEqual(page["experts"][0]["tier"], "external")
        next_page = manager.list_experts({"limit": 2, "cursor": page["next_cursor"], "allow_experimental": True})
        self.assertEqual(next_page["experts"][0]["id"], "layer.0.expert.2")
        self.assertIsNone(next_page["next_cursor"])
        with self.assertRaisesRegex(ValueError, "filter"):
            manager.list_experts({"cursor": page["next_cursor"], "tier": "ram", "allow_experimental": True})
        with self.assertRaisesRegex(ValueError, "model_id"):
            manager.list_experts({"model_id": "other-model", "allow_experimental": True})
        del model
        with self.assertRaisesRegex(ValueError, "not loaded"):
            manager.list_experts({"allow_experimental": True})

    def test_checkpoint_fingerprint_changes_with_replaced_shard(self):
        write_checkpoint(self.path)
        first = experts.inspect_model(self.path, 8192)["fingerprint"]
        write_checkpoint(self.path)
        self.assertNotEqual(first, experts.inspect_model(self.path, 8192)["fingerprint"])


def memory_guard_fixture(cache_bytes=24 * 1024**3):
    """An isolated scheduler class with the validated 0.6.4 hook contracts.

    The fake owns admission/rejection, just as the native runtime does. These
    tests exercise Werk's integration boundaries without loading model weights.
    Native tracker and admission regressions below additionally exercise oMLX.
    """
    mib, gib = 1024**2, 1024**3
    checkpoint = SimpleNamespace(path=Path("/fixture/deepseek"), cache_bytes=cache_bytes,
                                 total_expert_bytes=86 * gib, expert_bytes={0: 8 * mib}, experts=11008)
    manager = experts.ExpertManager(checkpoint)

    class Model:
        pass

    class Scheduler:
        def __init__(self):
            self.model = Model()
            self.base_bytes = 4 * gib
            self.workspace = 64 * mib
            self.kv_bytes = 1 * mib
            self._memory_hard_limit_bytes = 36 * gib
            self._prefill_headroom_safety = 0.9
            self._prefill_memory_guard = True
            self.current_calls = []
            self.guard_calls = []
            self.adaptive_calls = []
            self.record_calls = []
            self.check_calls = []
            self.chunk_limit = 1024

        def _current_usage_bytes(self, *, refresh_mlx_active=True):
            self.current_calls.append(refresh_mlx_active)
            return self.base_bytes + manager._resident

        def _admission_limit_bytes(self):
            return 34 * gib

        def _prefill_abort_cap(self):
            return 32 * gib

        def _admission_estimate(self, *, num_prompt_tokens, cached_tokens, current):
            return SimpleNamespace(kv_exact=self.kv_bytes, transient=self.workspace,
                                   estimated=current + self.kv_bytes + self.workspace)

        def _admission_transient_bound(self, n_tokens, kv_len):
            return self.workspace

        def _predicted_chunk_transient(self, n_tokens, kv_len):
            return self.workspace

        def _adaptive_chunk_size(self, requested, *, request_id, loop_label, kv_len=0):
            self.adaptive_calls.append((requested, request_id, loop_label, kv_len))
            if self._current_usage_bytes() + self.workspace > self._prefill_abort_cap():
                return max(1, requested // 2)
            return requested

        def _guard_prefill_chunk(self, n_tokens, *, kv_len, progress, loop_label, request_id=None):
            self.guard_calls.append((n_tokens, kv_len, progress, loop_label, request_id))
            if self._prefill_memory_guard and self._current_usage_bytes() + self.workspace > self._prefill_abort_cap():
                raise RuntimeError("native prefill guard rejected unsafe peak")
            return min(n_tokens, self.chunk_limit)

        def _record_chunk_transient(self, n_tokens, pre_bytes, post_bytes, *, request_id,
                                    loop_label, kv_len=0, requested_step=None):
            self.record_calls.append((n_tokens, pre_bytes, post_bytes, request_id,
                                      loop_label, kv_len, requested_step))

        def _preflight_memory_check(self, request):
            self.check_calls.append(request)
            if self._prefill_memory_guard and self._current_usage_bytes() + self.workspace > self._prefill_abort_cap():
                return "native preflight rejected unsafe peak"
            return None

    class Engine:
        async def _preflight_or_raise_with_eviction(self, scheduler, *, num_prompt_tokens, request_id):
            return (scheduler, num_prompt_tokens, request_id)

    scheduler = Scheduler()
    manager._model_ref = weakref.ref(scheduler.model)
    guard = experts._ExpertMemoryGuard(manager, Scheduler, Engine)
    guard.install(Scheduler, Engine)
    return manager, scheduler, guard, Engine


class ExpertMemoryGuardTests(unittest.TestCase):
    def setUp(self):
        self.manager, self.scheduler, self.guard, self.engine = memory_guard_fixture()
        self.mib, self.gib = 1024**2, 1024**3

    def guard_chunk(self, n=49, request_id="first", label="external"):
        return self.scheduler._guard_prefill_chunk(
            n, kv_len=0, progress=0, loop_label=label, request_id=request_id)

    def test_adaptive_sizing_reclaims_cache_capacity_before_throttling(self):
        self.scheduler.base_bytes = 8 * self.gib
        self.scheduler.workspace = 5 * self.gib
        self.assertEqual(self.guard.original_adaptive(self.scheduler, 2048,
                         request_id="long-tools", loop_label="external"), 1024)
        self.assertEqual(self.scheduler._adaptive_chunk_size(2048,
                         request_id="long-tools", loop_label="external", kv_len=2048), 2048)
        self.assertLess(self.manager.effective_cache_bytes, 24 * self.gib)
        self.assertTrue(self.scheduler._prefill_memory_guard)
        self.assertEqual(self.scheduler.adaptive_calls[-1], (2048, "long-tools", "external", 2048))

    def test_adaptive_sizing_does_not_resize_an_unrelated_model(self):
        self.manager._model_ref = None
        self.scheduler.base_bytes = 31 * self.gib
        self.scheduler.workspace = 5 * self.gib
        self.assertEqual(self.scheduler._adaptive_chunk_size(2048,
                         request_id="other", loop_label="external"), 1024)
        self.assertEqual(self.manager.effective_cache_bytes, 24 * self.gib)

    def record_chunk(self, before, after, n=49, request_id="first", label="external"):
        self.scheduler._record_chunk_transient(
            n, before, after, request_id=request_id, loop_label=label,
            kv_len=50, requested_step=1024)

    def test_cold_49_token_chunk_removes_24_gib_residency_growth_only(self):
        self.assertEqual(self.guard_chunk(), 49)
        self.manager._resident = 24 * self.gib
        self.record_chunk(4 * self.gib, 28 * self.gib + 32 * self.mib)
        self.assertEqual(self.scheduler.record_calls[-1],
                         (49, 4 * self.gib, 4 * self.gib + 32 * self.mib,
                          "first", "external", 50, 1024))
        self.assertTrue(self.scheduler._prefill_memory_guard)
        self.assertEqual(len(self.scheduler.guard_calls), 1)
        self.assertEqual(self.manager.prefill_cache_growth_bytes, 24 * self.gib)
        self.assertEqual(self.manager.prefill_samples_corrected, 1)

    def test_empty_capacity_reserved_but_resident_weights_not_double_charged(self):
        self.assertEqual(self.scheduler._current_usage_bytes(refresh_mlx_active=False), 29 * self.gib)
        self.assertFalse(self.scheduler.current_calls[-1])
        self.manager._resident = 24 * self.gib
        self.assertEqual(self.scheduler._current_usage_bytes(), 29 * self.gib)
        self.assertEqual(self.manager.effective_cache_bytes, 24 * self.gib)

    def test_prepare_reduces_cache_for_context_then_recovers_when_headroom_returns(self):
        self.scheduler.workspace = 10 * self.gib
        request = SimpleNamespace(num_prompt_tokens=10000, cached_tokens=50)
        self.assertIsNone(self.scheduler._preflight_memory_check(request))
        # 32 GiB cap - 4 GiB base - 10 GiB transient - KV - 1 GiB reserve.
        self.assertEqual(self.manager.effective_cache_bytes, 17 * self.gib - self.mib)
        self.assertEqual(self.manager.budget_reductions, 1)
        self.assertEqual(self.scheduler.check_calls, [request])
        self.scheduler.workspace = 64 * self.mib
        self.assertIsNone(self.scheduler._preflight_memory_check(request))
        self.assertEqual(self.manager.effective_cache_bytes, 24 * self.gib)
        self.assertTrue(self.scheduler._prefill_memory_guard)

    def test_adapter_transient_reserve_reclaims_capacity_before_chunk_sizing(self):
        self.scheduler.workspace = 4 * self.gib
        self.manager.prefill_transient_reserve = lambda tokens, peak: peak
        self.assertEqual(self.scheduler._adaptive_chunk_size(
            2048, request_id="large-tools", loop_label="external"), 2048)
        self.assertEqual(self.manager.effective_cache_bytes, 19 * self.gib - 2 * self.mib)
        self.assertTrue(self.scheduler._prefill_memory_guard)

    def test_unknown_native_caps_preserve_current_budget_without_eviction(self):
        self.manager.effective_cache_bytes = 8 * self.gib
        self.manager._resident = 8 * self.gib
        self.scheduler._memory_hard_limit_bytes = 0
        self.scheduler._admission_limit_bytes = lambda: 0
        self.scheduler._prefill_abort_cap = lambda: 0
        with patch.object(self.manager, "_resize_cache") as resize:
            self.guard.prepare(self.scheduler, num_prompt_tokens=100, cached_tokens=50)
        resize.assert_not_called()
        self.assertEqual(self.manager.effective_cache_bytes, 8 * self.gib)
        self.assertEqual(self.manager._resident, 8 * self.gib)

    def test_eight_gib_configuration_stays_bounded_even_with_spare_headroom(self):
        manager, scheduler, guard, _ = memory_guard_fixture(cache_bytes=8 * self.gib)
        guard.prepare(scheduler, num_prompt_tokens=100, cached_tokens=50)
        self.assertEqual(manager.effective_cache_bytes, 8 * self.gib)
        self.assertEqual(scheduler._current_usage_bytes(), 13 * self.gib)
        scheduler.workspace = 40 * self.gib
        guard.prepare(scheduler, num_prompt_tokens=100000)
        self.assertEqual(manager.effective_cache_bytes, 8 * self.mib)
        scheduler.workspace = 64 * self.mib
        guard.prepare(scheduler, num_prompt_tokens=100)
        self.assertEqual(manager.effective_cache_bytes, 8 * self.gib)

    def test_real_workspace_pressure_remains_rejected_with_minimal_expert_cache(self):
        self.scheduler.workspace = 40 * self.gib
        request = SimpleNamespace(num_prompt_tokens=100000, cached_tokens=0)
        self.assertEqual(self.scheduler._preflight_memory_check(request),
                         "native preflight rejected unsafe peak")
        self.assertEqual(self.manager.effective_cache_bytes, 8 * self.mib)
        self.assertTrue(self.scheduler._prefill_memory_guard)
        with self.assertRaisesRegex(RuntimeError, "native prefill guard"):
            self.guard_chunk()

    def test_signed_eviction_delta_preserves_genuine_workspace_growth(self):
        self.manager._resident = 24 * self.gib
        self.guard_chunk()
        self.manager._resident = 16 * self.gib
        self.record_chunk(28 * self.gib, 20 * self.gib + 64 * self.mib)
        before, after = self.scheduler.record_calls[-1][1:3]
        self.assertEqual(after - before, 64 * self.mib)
        self.assertEqual(self.manager.prefill_cache_growth_bytes, 0)
        self.assertEqual(self.manager.prefill_samples_corrected, 1)

    def test_delayed_weight_release_is_not_native_transient_reallocation(self):
        self.guard_chunk()
        self.guard.deferred_reclaims[self.scheduler] = (8 * self.gib, 28 * self.gib, 0)
        self.record_chunk(28 * self.gib, 20 * self.gib)
        self.assertEqual(self.scheduler.record_calls[-1][1:3], (28 * self.gib, 28 * self.gib))
        self.assertEqual(self.guard.deferred_reclaims[self.scheduler][0], 0)

    def test_delayed_release_cannot_hide_simultaneous_active_memory_growth(self):
        self.scheduler._last_mlx_active_memory_bytes = 4 * self.gib
        self.guard_chunk()
        self.guard.deferred_reclaims[self.scheduler] = (8 * self.gib, 28 * self.gib, 0)
        self.scheduler._last_mlx_active_memory_bytes += 128 * self.mib
        self.record_chunk(28 * self.gib, 20 * self.gib)
        before, after = self.scheduler.record_calls[-1][1:3]
        self.assertEqual(after - before, 128 * self.mib)

    def test_releases_between_callbacks_consume_deferred_credit(self):
        self.guard.deferred_reclaims[self.scheduler] = (8 * self.gib, 12 * self.gib, 0)
        self.guard.prepare(self.scheduler, num_prompt_tokens=50)
        self.assertEqual(self.guard.deferred_reclaims[self.scheduler][0], 0)

    def test_prepare_records_only_evicted_weights_not_yet_physically_released(self):
        self.manager._resident = 24 * self.gib
        self.guard.original_current = lambda *args, **kwargs: 28 * self.gib
        with patch.object(self.manager, "_resize_cache", side_effect=lambda _: setattr(self.manager, "_resident", 8 * self.gib)):
            self.guard.prepare(self.scheduler, num_prompt_tokens=50)
        self.assertEqual(self.guard.deferred_reclaims[self.scheduler][0], 16 * self.gib)

    def test_unmatched_or_consumed_sample_cannot_hide_memory_growth(self):
        for mismatch in ({"request_id": "other"}, {"label": "chunked_step"}, {"n": 32}):
            with self.subTest(mismatch=mismatch):
                self.manager._resident = 0
                self.guard_chunk()
                self.manager._resident = 24 * self.gib
                self.record_chunk(4 * self.gib, 28 * self.gib, **mismatch)
                self.assertEqual(self.scheduler.record_calls[-1][1:3], (4 * self.gib, 28 * self.gib))
                self.record_chunk(4 * self.gib, 28 * self.gib)
                self.assertEqual(self.scheduler.record_calls[-1][1:3], (4 * self.gib, 28 * self.gib))
        self.assertEqual(self.manager.prefill_samples_corrected, 0)

    def test_sample_uses_native_throttled_chunk_length(self):
        self.scheduler.chunk_limit = 32
        self.assertEqual(self.guard_chunk(), 32)
        self.manager._resident = 24 * self.gib
        self.record_chunk(4 * self.gib, 28 * self.gib + self.mib, n=32)
        self.assertEqual(self.scheduler.record_calls[-1][2], 4 * self.gib + self.mib)

    def test_other_models_delegate_without_reservation_or_sample_correction(self):
        import asyncio
        other = type(self.scheduler)()
        self.assertEqual(other._current_usage_bytes(), 4 * self.gib)
        self.assertEqual(other._guard_prefill_chunk(49, kv_len=0, progress=0,
                                                  loop_label="external", request_id="other"), 49)
        other._record_chunk_transient(49, 4 * self.gib, 28 * self.gib,
                                      request_id="other", loop_label="external")
        self.assertEqual(other.record_calls[-1][1:3], (4 * self.gib, 28 * self.gib))
        result = asyncio.run(self.engine()._preflight_or_raise_with_eviction(
            other, num_prompt_tokens=50, request_id="other"))
        self.assertEqual(result, (other, 50, "other"))
        self.assertEqual(self.manager.prefill_samples_corrected, 0)

    def test_target_http_preflight_uses_owning_executor_and_preserves_native_preflight(self):
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        import threading
        owner = self.engine()
        threads = []
        original = self.guard.prepare
        def tracked_prepare(*args, **kwargs):
            threads.append(threading.current_thread().name)
            return original(*args, **kwargs)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="owning-mlx") as executor:
            owner._engine = SimpleNamespace(engine=SimpleNamespace(_mlx_executor=executor))
            with patch.object(self.guard, "prepare", side_effect=tracked_prepare):
                result = asyncio.run(owner._preflight_or_raise_with_eviction(
                    self.scheduler, num_prompt_tokens=50, request_id="first"))
        self.assertEqual(result, (self.scheduler, 50, "first"))
        self.assertEqual(len(threads), 1)
        self.assertTrue(threads[0].startswith("owning-mlx"))

    def test_target_http_preflight_fails_closed_without_owning_executor(self):
        import asyncio
        with self.assertRaisesRegex(ValueError, "owning MLX executor"):
            asyncio.run(self.engine()._preflight_or_raise_with_eviction(
                self.scheduler, num_prompt_tokens=50, request_id="first"))


class ExpertCacheResizeTests(unittest.TestCase):
    def setUp(self):
        self.manager, _, _, _ = memory_guard_fixture(cache_bytes=4 * 8 * 1024**2)
        self.size = 8 * 1024**2
        self.manager.checkpoint.experts = 4
        for expert in range(4):
            self.manager._cache[(0, expert)] = object()
        self.manager._resident = 4 * self.size

    def test_resize_preserves_pins_and_inflight_leases_and_evicts_only_lru(self):
        self.manager._pinned.add((0, 1))
        self.manager._leased.add((0, 2))
        with patch.object(self.manager, "_trim_allocator") as trim:
            self.manager._resize_cache(0)
        self.assertEqual(list(self.manager._cache), [(0, 1), (0, 2), (0, 3)])
        self.assertEqual(self.manager._pinned, {(0, 1)})
        self.assertEqual(self.manager._leased, {(0, 2)})
        self.assertEqual(self.manager._resident, 3 * self.size)
        self.assertEqual(self.manager.effective_cache_bytes, 3 * self.size)
        self.assertEqual(self.manager.cache_evictions, 1)
        trim.assert_called_once_with(force=True)

    def test_resize_is_bounded_by_one_expert_configured_maximum_and_model_size(self):
        with patch.object(self.manager, "_trim_allocator"):
            self.manager._resize_cache(-1)
        self.assertEqual(self.manager.effective_cache_bytes, self.size)
        self.assertEqual(len(self.manager._cache), 1)
        self.manager._resize_cache(100 * self.size)
        self.assertEqual(self.manager.effective_cache_bytes, 4 * self.size)
        self.manager.checkpoint.total_expert_bytes = 2 * self.size
        self.manager._resize_cache(100 * self.size)
        self.assertEqual(self.manager.effective_cache_bytes, 2 * self.size)


@unittest.skipUnless(os.environ.get("WERK_TEST_MLX_EXPERTS") == "1", "native oMLX tests are opt-in")
class NativePrefillMemoryAccountingTests(unittest.TestCase):
    def setUp(self):
        from omlx.engine.batched import BatchedEngine
        from omlx.prefill_transient_tracker import PrefillTransientTracker
        from omlx.scheduler import Scheduler
        self.gib, self.mib = 1024**3, 1024**2
        self.manager, fixture, _, _ = memory_guard_fixture()
        self.model = fixture.model
        # Construct scheduler bookkeeping only: no scheduler constructor, model
        # weights, MLX allocations, native worker, or inference are involved.
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.model = self.model
        self.scheduler._prefill_transient_tracker = PrefillTransientTracker("fixture")
        self.scheduler._prefill_min_chunk_tokens = 32
        self.scheduler._prefill_speed_priority = False
        self.scheduler._prefill_memory_guard = True
        self.scheduler._memory_hard_limit_bytes = 36 * self.gib
        self.scheduler.memory_monitor = SimpleNamespace(
            estimate_chunk_transient_bytes=lambda n, kv: 32 * self.mib,
            estimate_prompt_kv_bytes=lambda n: n * 1024,
            estimate_resident_kv_bytes=lambda n, chunk_tokens: n * 1024)
        self.scheduler._current_usage_bytes = lambda: 29 * self.gib
        self.scheduler._admission_limit_bytes = lambda: 34 * self.gib
        self.scheduler._preflight_safety_rejection = lambda **kwargs: None
        self.scheduler._raise_prefill_eviction_if_available = lambda **kwargs: None
        self.scheduler._format_rejection_message = lambda **kwargs: "unsafe native peak"
        self.guard = experts._ExpertMemoryGuard(self.manager, Scheduler, BatchedEngine)

    def test_actual_native_ewma_reproduces_second_turn_rejection_and_accepts_corrected_sample(self):
        scheduler = self.scheduler
        before, after = 4 * self.gib, 28 * self.gib + 32 * self.mib
        request = SimpleNamespace(num_prompt_tokens=100, cached_tokens=0, request_id="next")
        # Before the fix, the 49-token first chunk poisons the native learner
        # with persistent expert residency and charges >20 GiB on the next turn.
        self.guard.original_record(scheduler, 49, before, after,
                                   request_id="first", loop_label="external")
        poisoned = scheduler._admission_estimate(num_prompt_tokens=100, cached_tokens=0,
                                                 current=29 * self.gib)
        self.assertGreater(poisoned.transient, 20 * self.gib)
        self.assertIsNotNone(scheduler._preflight_memory_check(request))

        scheduler._prefill_transient_tracker.reset()
        self.guard.samples[scheduler] = ("first", "external", 49, 0)
        self.manager._resident = 24 * self.gib
        self.guard.record(scheduler, 49, before, after, request_id="first", loop_label="external")
        corrected = scheduler._admission_estimate(num_prompt_tokens=100, cached_tokens=0,
                                                  current=29 * self.gib)
        self.assertLess(corrected.transient, 64 * self.mib)
        self.assertIsNone(scheduler._preflight_memory_check(request))
        self.assertTrue(scheduler._prefill_memory_guard)

    def test_actual_native_reclaim_ledger_excludes_expert_eviction_but_keeps_pool_reclaim(self):
        self.guard.samples[self.scheduler] = ("evict", "external", 49, 24 * self.gib)
        self.manager._resident = 16 * self.gib
        self.guard.record(self.scheduler, 49, 28 * self.gib, 20 * self.gib - 32 * self.mib,
                          request_id="evict", loop_label="external")
        self.assertEqual(self.scheduler._prefill_transient_tracker.recent_reclaim_bytes, 32 * self.mib)
        self.assertEqual(self.scheduler._prefill_transient_tracker.samples, 0)

    def test_actual_native_reclaim_ledger_excludes_delayed_weight_release(self):
        scheduler = self.scheduler
        before, after = 28 * self.gib, 12 * self.gib - 32 * self.mib
        self.guard.original_record(scheduler, 2048, before, after,
                                   request_id="delayed", loop_label="external")
        self.assertGreater(scheduler._prefill_transient_tracker.recent_reclaim_bytes, 16 * self.gib)
        scheduler._prefill_transient_tracker.reset()
        self.guard.samples[scheduler] = ("delayed", "external", 2048, 0, None)
        self.guard.deferred_reclaims[scheduler] = (16 * self.gib, before, 0)
        self.guard.original_current = lambda _: after
        self.guard.record(scheduler, 2048, before, after,
                          request_id="delayed", loop_label="external")
        self.assertEqual(scheduler._prefill_transient_tracker.recent_reclaim_bytes, 32 * self.mib)

    def test_actual_native_guard_keeps_rejecting_genuine_large_transient(self):
        self.guard.samples[self.scheduler] = ("large", "external", 49, 0)
        self.manager._resident = 24 * self.gib
        self.guard.record(self.scheduler, 49, 4 * self.gib, 44 * self.gib,
                          request_id="large", loop_label="external")
        request = SimpleNamespace(num_prompt_tokens=100, cached_tokens=0, request_id="next")
        rejection = self.scheduler._preflight_memory_check(request)
        self.assertIsNotNone(rejection)
        self.assertGreater(rejection.estimated_bytes, rejection.limit_bytes)
        self.assertTrue(self.scheduler._prefill_memory_guard)


@unittest.skipUnless(os.environ.get("WERK_TEST_MLX_EXPERTS") == "1", "real MLX tests are opt-in")
class MlxStreamingTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        from omlx.patches.deepseek_v4.switch_layers import SwitchGLU
        from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
        apply_deepseek_v4_patch()
        from mlx_lm.models.deepseek_v4 import LimitedSwiGLU
        from mlx.utils import tree_flatten
        import mlx.nn as nn

        self.mx = mx
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        cfg = config()
        cfg["quantization"]["model.layers.0.ffn.switch_mlp.gate_proj"] = {
            "bits": 4, "group_size": 64, "mode": "affine"}
        (self.path / "config.json").write_text(json.dumps(cfg))
        self.baseline = SwitchGLU(64, 64, 4, activation=LimitedSwiGLU(10.0))
        nn.quantize(self.baseline, group_size=32, bits=2,
                    class_predicate=lambda path, module: {"bits": 4, "group_size": 64} if path == "gate_proj" else hasattr(module, "to_quantized"))
        tensors = {f"model.layers.0.ffn.switch_mlp.{key}": value for key, value in tree_flatten(self.baseline.parameters())}
        for name, value in list(tensors.items()):
            if name.endswith((".scales", ".biases")):
                tensors[name] = value.astype(mx.bfloat16)
        mx.save_safetensors(str(self.path / "model.safetensors"), tensors)
        # Match the installed sanitizer's conversion, including its BF16 rounding.
        self.baseline.load_weights([(key.split(".switch_mlp.", 1)[1], value.astype(mx.float16)
                                     if key.endswith((".scales", ".biases")) else value)
                                    for key, value in tensors.items()])
        self.baseline.eval()
        mx.eval(self.baseline.parameters())
        plan = experts.inspect_model(self.path, 1 << 20)
        checkpoint = experts.Checkpoint(self.path, plan["largest_expert_bytes"] * 2)
        self.manager = experts.ExpertManager(checkpoint)
        self.addCleanup(self.manager.deactivate)
        self.streamed = experts._streamed_module(self.manager, 0, self.baseline.activation)
        self.streamed.eval()
        import weakref
        self.manager._model_ref = weakref.ref(self.streamed)

    def compare(self, count, dtype=None):
        mx = self.mx
        x = mx.random.normal((1, count, 64)).astype(dtype or mx.bfloat16)
        routes = mx.array([[(i % 4), (i + 2) % 4] for i in range(count)], dtype=mx.uint32)[None]
        expected = self.baseline(x, routes)
        actual = self.streamed(x, routes)
        mx.eval(expected, actual)
        # qmm and gather_qmm may have different accumulation kernels; compare
        # numerical outputs, not just selected indices or tensor shapes.
        self.assertLess(float(mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))), 0.008)
        self.assertEqual(expected.shape, actual.shape)
        self.assertLessEqual(self.manager.status()["resident_cache_bytes"], self.manager.checkpoint.cache_bytes)
        from mlx.utils import tree_flatten
        self.assertFalse(tree_flatten(self.streamed.parameters()))

    def test_decode_matches_resident_quantized_experts(self):
        self.compare(1)

    def test_grouped_matches_serial_with_weighted_duplicate_routes(self):
        mx = self.mx
        routes = mx.array([[[0, 1, 0], [2, 3, 2]]], dtype=mx.uint32)
        x = mx.random.normal((1, 2, 64)).astype(mx.bfloat16)
        scores = mx.array([[[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]]])
        expected = (self.baseline(x, routes) * scores[..., None].astype(x.dtype)).sum(-2)
        outputs = []
        for mode in ("serial", "grouped"):
            self.manager.deactivate()
            self.manager.execution = mode
            result = self.streamed(x, routes, scores=scores, weighted_sum=True)
            mx.eval(result)
            outputs.append(result)
            self.assertFalse(self.manager._leased)
            self.assertLessEqual(self.manager._resident, self.manager.checkpoint.cache_bytes)
            self.assertLess(float(mx.max(mx.abs(expected.astype(mx.float32) - result.astype(mx.float32)))), 0.008)
        self.assertEqual(float(mx.max(mx.abs(outputs[0].astype(mx.float32) - outputs[1].astype(mx.float32)))), 0.0)

    def test_grouped_materialization_and_eval_reduce_syncs_without_more_reads(self):
        mx = self.mx
        x = mx.ones((1, 1, 64), dtype=mx.bfloat16)
        routes = mx.array([[[0, 1]]], dtype=mx.uint32)
        observed = {}
        for mode in ("serial", "grouped"):
            manager = experts.ExpertManager(self.manager.checkpoint, execution=mode)
            try:
                module = experts._streamed_module(manager, 0, self.baseline.activation)
                mx.eval(module(x, routes))
                observed[mode] = manager.status()
                self.assertFalse(manager._leased)
            finally:
                manager.deactivate()
        self.assertEqual(observed["serial"]["disk_bytes_read"], observed["grouped"]["disk_bytes_read"])
        self.assertEqual(observed["serial"]["tensor_materializations"], 18)
        self.assertEqual(observed["grouped"]["tensor_materializations"], 2)
        self.assertEqual(observed["serial"]["output_evaluations"], 2)
        self.assertEqual(observed["grouped"]["output_evaluations"], 1)

    def test_failed_group_read_releases_leases_for_later_requests(self):
        mx = self.mx
        self.manager.execution = "grouped"
        x = mx.ones((1, 1, 64), dtype=mx.bfloat16)
        routes = mx.array([[[0, 1]]], dtype=mx.uint32)
        with patch.object(self.manager, "_read", side_effect=OSError("failed read")):
            with self.assertRaises(OSError):
                self.streamed(x, routes)
        self.assertFalse(self.manager._leased)
        mx.eval(self.streamed(x, routes))
        self.assertFalse(self.manager._leased)

    def test_partial_group_acquisition_failure_releases_leases_and_recovers(self):
        mx = self.mx
        self.manager.execution = "grouped"
        x = mx.ones((1, 1, 64), dtype=mx.bfloat16)
        routes = mx.array([[[0, 1]]], dtype=mx.uint32)
        original_read = self.manager._read
        second_expert_reads = []

        def fail_during_second_expert(name, expert=None, pending=None):
            if expert == 1:
                second_expert_reads.append(name)
                if name.endswith("up_proj.scales"):
                    # Expert 0 is complete; expert 1 already has its gate
                    # tensors and up weight in the pending materialization.
                    self.assertEqual(set(self.manager._cache), {(0, 0)})
                    self.assertEqual(len(pending), 4)
                    self.assertEqual(self.manager._leased, {(0, 0), (0, 1)})
                    raise OSError("injected partial expert read failure")
            return original_read(name, expert, pending=pending)

        with patch.object(self.manager, "_read", side_effect=fail_during_second_expert):
            with self.assertRaisesRegex(OSError, "partial expert read"):
                self.streamed(x, routes)
        self.assertEqual(len(second_expert_reads), 5)
        self.assertEqual(set(self.manager._cache), {(0, 0)})
        self.assertEqual(self.manager._resident, self.manager.checkpoint.expert_bytes[0])
        self.assertFalse(self.manager._leased)

        # First force eviction of the completed expert from the failed group,
        # then retry that group so a partially loaded expert cannot go unnoticed.
        for next_routes in (mx.array([[[2, 3]]], dtype=mx.uint32), routes):
            expected = self.baseline(x, next_routes)
            actual = self.streamed(x, next_routes)
            mx.eval(expected, actual)
            self.assertLess(float(mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))), 0.008)
            self.assertFalse(self.manager._leased)
            self.assertLessEqual(self.manager._resident, self.manager.checkpoint.cache_bytes)

    def test_group_output_evaluation_failure_releases_leases_and_recovers(self):
        mx = self.mx
        self.manager.execution = "grouped"
        x = mx.ones((1, 1, 64), dtype=mx.bfloat16)
        routes = mx.array([[[0, 1]]], dtype=mx.uint32)
        # Materialize the weights before intercepting mx.eval, so the injected
        # failure targets the output graph rather than tensor loading.
        for key in ((0, 0), (0, 1)):
            self.manager._acquire(key)
        original_eval = mx.eval
        output_evaluations = []

        def fail_group_output(*values, **kwargs):
            if len(values) == 1 and values[0].shape == (2, 64):
                output_evaluations.append(bool(self.manager._leased))
                if self.manager._leased:
                    self.assertEqual(self.manager._leased, {(0, 0), (0, 1)})
                    raise RuntimeError("injected group output evaluation failure")
            return original_eval(*values, **kwargs)

        with patch.object(mx, "eval", side_effect=fail_group_output):
            with self.assertRaisesRegex(RuntimeError, "group output evaluation"):
                self.streamed(x, routes)
        # The first matching evaluation initializes zeros before leasing; the
        # second evaluates the completed expert graph while the group is leased.
        self.assertEqual(output_evaluations, [False, True])
        self.assertEqual(self.manager.output_evaluations, 0)
        self.assertFalse(self.manager._leased)

        for next_routes in (mx.array([[[2, 3]]], dtype=mx.uint32), routes):
            expected = self.baseline(x, next_routes)
            actual = self.streamed(x, next_routes)
            mx.eval(expected, actual)
            self.assertLess(float(mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))), 0.008)
            self.assertFalse(self.manager._leased)
            self.assertLessEqual(self.manager._resident, self.manager.checkpoint.cache_bytes)

    def test_float32_inputs_preserve_resident_numerics(self):
        self.compare(1, self.mx.float32)

    def test_prefill_matches_after_repeated_lru_evictions(self):
        self.compare(40)
        first_misses = self.manager.misses
        self.compare(40)
        self.assertGreater(self.manager.misses, first_misses)
        self.assertEqual(self.manager.status()["resident_experts"], 2)

    def test_selected_ranges_only_are_read_and_cache_hits_avoid_io(self):
        mx = self.mx
        x = mx.ones((1, 1, 64), dtype=mx.bfloat16)
        routes = mx.array([[[0, 1]]], dtype=mx.uint32)
        self.streamed(x, routes)
        self.assertEqual(self.manager.disk_bytes_read, self.manager.checkpoint.expert_bytes[0] * 2)
        before = self.manager.disk_bytes_read
        self.streamed(x, routes)
        self.assertEqual(self.manager.disk_bytes_read, before)
        self.assertEqual(self.manager.hits, 2)

    def test_pin_unpin_evict_and_dry_run_report_actual_residency(self):
        request = {"model_id": self.manager.model_id, "expert_ids": ["layer.0.expert.0"],
                   "action": "pin", "allow_experimental": True, "dry_run": True}
        result = self.manager.action(request)
        self.assertEqual(result["changed"], 1)
        self.assertEqual(self.manager.disk_bytes_read, 0)
        request["dry_run"] = False
        self.assertTrue(self.manager.action(request)["experts"][0]["pinned"])
        request["action"] = "evict"
        with self.assertRaisesRegex(ValueError, "pinned"):
            self.manager.action(request)
        request["action"] = "unpin"
        self.manager.action(request)
        request["action"] = "evict"
        self.assertEqual(self.manager.action(request)["experts"][0]["tier"], "external")
        self.assertEqual(self.manager.status()["resident_cache_bytes"], 0)

    def test_pin_cannot_consume_demand_loading_slot(self):
        request = {"model_id": self.manager.model_id, "expert_ids": ["layer.0.expert.0", "layer.0.expert.1"],
                   "action": "pin", "allow_experimental": True}
        with self.assertRaisesRegex(ValueError, "demand-loaded"):
            self.manager.action(request)

    def test_failed_pin_does_not_leave_unloaded_experts_pinned(self):
        request = {"model_id": self.manager.model_id, "expert_ids": ["layer.0.expert.0"],
                   "action": "pin", "allow_experimental": True}
        with patch.object(self.manager, "_read", side_effect=OSError("disk read failed")):
            with self.assertRaisesRegex(OSError, "disk read failed"):
                self.manager.action(request)
        self.assertFalse(self.manager._pinned)
        self.assertFalse(self.manager._cache)

    def test_complete_tiny_model_load_never_reads_packed_expert_tensors(self):
        import mlx.nn as nn
        from mlx_lm import utils
        from mlx_lm.models.deepseek_v4 import Model, ModelArgs
        from mlx.utils import tree_flatten
        mx = self.mx
        # A complete one-layer DeepSeek model exercises strict base-weight
        # loading, sanitization, tokenizer handoff, and module replacement.
        cfg = config()
        cfg.update(vocab_size=64, num_attention_heads=4, head_dim=64,
                   q_lora_rank=64, qk_rope_head_dim=32, o_groups=2,
                   o_lora_rank=32, num_hash_layers=0, compress_ratios=[0])
        (self.path / "config.json").write_text(json.dumps(cfg))
        baseline = Model(ModelArgs.from_dict(cfg))
        nn.quantize(baseline, bits=2, group_size=32,
                    class_predicate=lambda path, module: hasattr(module, "to_quantized")
                    and module.weight.shape[-1] % 32 == 0)
        tensors = dict(tree_flatten(baseline.parameters()))
        for name, value in list(tensors.items()):
            if ".switch_mlp." in name and name.endswith((".scales", ".biases")):
                tensors[name] = value.astype(mx.bfloat16)
        mx.save_safetensors(str(self.path / "model.safetensors"), tensors)
        baseline.load_weights(list(baseline.sanitize(tensors).items()))
        baseline.eval()
        manager = experts.ExpertManager(experts.Checkpoint(self.path, 16384))
        self.addCleanup(manager.deactivate)
        reads = []
        original_read = manager._read

        def tracked_read(name, expert=None):
            reads.append((name, expert))
            return original_read(name, expert)

        sentinel = object()
        with patch.object(manager, "_read", side_effect=tracked_read), patch.object(utils, "load_tokenizer", return_value=sentinel):
            model, tokenizer = experts._load_streamed_model(manager)
        self.assertIs(tokenizer, sentinel)
        self.assertTrue(manager.status()["active"])
        self.assertEqual(manager.disk_bytes_read, manager.checkpoint.base_bytes)
        self.assertTrue(all(".switch_mlp." not in name for name, _ in reads))
        self.assertTrue(all(".switch_mlp." not in name for name, _ in tree_flatten(model.parameters())))
        inputs = mx.array([[1, 2]], dtype=mx.uint32)
        expected, actual = baseline(inputs), model(inputs)
        mx.eval(expected, actual)
        self.assertLess(float(mx.max(mx.abs(expected - actual))), 0.02)




class AutomaticExpertBudgetTests(unittest.TestCase):
    def checkpoint(self, base_gib=4, experts_gib=86):
        return SimpleNamespace(base_bytes=base_gib * 1024**3,
                               total_expert_bytes=experts_gib * 1024**3,
                               expert_bytes={0: 8 * 1024**2})

    def test_48_gib_machine_uses_device_room_instead_of_fixed_24_gib(self):
        budget = experts.automatic_cache_budget(self.checkpoint(),
                    metal_limit=36 * 1024**3, available_memory=42 * 1024**3)
        self.assertGreater(budget, 24 * 1024**3)
        self.assertLess(budget + 4 * 1024**3 + experts._WORKSPACE_BYTES, 36 * 1024**3)

    def test_small_machine_keeps_partial_expert_residency(self):
        budget = experts.automatic_cache_budget(self.checkpoint(),
                    metal_limit=12 * 1024**3, available_memory=11 * 1024**3)
        self.assertGreater(budget, 8 * 1024**2)
        self.assertLess(budget, 8 * 1024**3)

    def test_current_memory_pressure_limits_budget(self):
        budget = experts.automatic_cache_budget(self.checkpoint(),
                    metal_limit=36 * 1024**3, available_memory=10 * 1024**3)
        self.assertLess(budget, 4 * 1024**3)

    def test_multi_terabyte_budget_is_bounded_by_model_without_24_gib_or_1_tib_cap(self):
        checkpoint = self.checkpoint(base_gib=64, experts_gib=2048)
        self.assertEqual(experts.automatic_cache_budget(checkpoint,
                    metal_limit=4096 * 1024**3, available_memory=4000 * 1024**3),
                    checkpoint.total_expert_bytes)

    def test_unknown_or_insufficient_room_never_means_unlimited(self):
        for metal, available in [(0, 20 * 1024**3), (None, 20 * 1024**3),
                                 (36 * 1024**3, 0), (36 * 1024**3, 5 * 1024**3)]:
            with self.assertRaises(ValueError):
                experts.automatic_cache_budget(self.checkpoint(),
                    metal_limit=metal, available_memory=available)

    def test_auto_metadata_does_not_read_weights_or_query_hardware(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            write_checkpoint(path)
            with patch.object(experts, '_configure_automatic_cache', side_effect=AssertionError('no hardware')):
                result = experts.inspect_model(path, 0)
            self.assertEqual(result['cache_budget_mode'], 'auto')
            self.assertEqual(result['cache_budget_bytes'], result['expert_bytes'])


if __name__ == "__main__":
    unittest.main()
