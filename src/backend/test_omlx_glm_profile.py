"""CPU-only contracts for optional GLM layer-counter profiling."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "werk_glm_profile", Path(__file__).with_name("omlx_glm_profile.py"))
profile_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile_module)
GlmLayerProfile = profile_module.GlmLayerProfile


def manager():
    return SimpleNamespace(
        hits=10, misses=20, cache_evictions=5,
        tensor_materializations=3, materialize_seconds=1.5, allocator_clears=2,
        reader=SimpleNamespace(logical_bytes=1000, calls=4, read_seconds=0.5),
        access=SimpleNamespace(load_seconds=0.25, forward_calls=3,
                               forward_seconds=2.0, routing_seconds=0.25,
                               output_evaluations=4),
    )


class GlmLayerProfileTests(unittest.TestCase):
    def test_disabled_never_inspects_the_manager_or_clock(self):
        class Forbidden:
            def __getattribute__(self, name):
                raise AssertionError("disabled profile inspected the manager")

        profile = GlmLayerProfile()
        with patch.object(profile_module.time, "perf_counter",
                          side_effect=AssertionError("clock read")):
            with profile.measure(0, Forbidden()):
                pass
        state = profile.snapshot()
        self.assertFalse(state["enabled"])
        self.assertEqual(state["layers"], [])
        self.assertEqual(state["totals"]["calls"], 0)

    def test_forward_deltas_are_attributed_and_accumulated_by_layer(self):
        model = manager()
        profile = GlmLayerProfile(enabled=True)
        with patch.object(profile_module.time, "perf_counter",
                          side_effect=[1.0, 2.5, 3.0, 3.5, 4.0, 4.25]):
            with profile.measure(7, model):
                model.hits += 4
                model.misses += 2
                model.cache_evictions += 1
                model.reader.logical_bytes += 8192
                model.reader.calls += 2
                model.reader.read_seconds += 0.5
                model.access.forward_calls += 1
                model.access.forward_seconds += 1.0
                model.access.routing_seconds += 0.125
                model.access.output_evaluations += 2
            with profile.measure(2, model):
                model.misses += 3
                model.reader.logical_bytes += 12288
            with profile.measure(7, model):
                model.hits += 1
        result = profile.snapshot()
        self.assertEqual([row["layer"] for row in result["layers"]], [2, 7])
        first, second = result["layers"]
        self.assertEqual(first["cache_hits"], 0)
        self.assertEqual(first["cache_misses"], 3)
        self.assertEqual(first["disk_bytes_read"], 12288)
        self.assertEqual(second["calls"], 2)
        self.assertEqual(second["cache_hits"], 5)
        self.assertEqual(second["cache_misses"], 2)
        self.assertEqual(second["cache_evictions"], 1)
        self.assertEqual(second["disk_bytes_read"], 8192)
        self.assertEqual(second["disk_read_calls"], 2)
        self.assertEqual(second["disk_read_seconds"], 0.5)
        self.assertEqual(second["forward_seconds"], 1.0)
        self.assertEqual(second["routing_seconds"], 0.125)
        self.assertEqual(second["output_evaluations"], 2)
        self.assertEqual(second["wall_seconds"], 1.75)
        self.assertEqual(result["totals"]["calls"], 3)
        self.assertEqual(result["totals"]["disk_bytes_read"], 20480)
        self.assertEqual(result["totals"]["wall_seconds"], 2.25)
        self.assertEqual(result["totals"]["invalid_samples"], 0)

    def test_failed_forward_keeps_partial_counters_and_original_exception(self):
        model = manager()
        profile = GlmLayerProfile(enabled=True)
        failure = RuntimeError("failed expert read")
        with self.assertRaises(RuntimeError) as caught:
            with profile.measure(4, model):
                model.misses += 1
                model.reader.logical_bytes += 16
                raise failure
        self.assertIs(caught.exception, failure)
        row = profile.snapshot()["layers"][0]
        self.assertEqual(row["calls"], 1)
        self.assertEqual(row["failures"], 1)
        self.assertEqual(row["cache_misses"], 1)
        self.assertEqual(row["disk_bytes_read"], 16)
        self.assertEqual(row["forward_calls"], 0)

    def test_reset_drops_negative_delta_and_next_call_uses_new_baseline(self):
        model = manager()
        profile = GlmLayerProfile(enabled=True)
        with profile.measure(1, model):
            model.hits = 0
            model.reader.logical_bytes = 0
            model.access.forward_seconds = 0.0
        with profile.measure(1, model):
            model.hits += 2
            model.reader.logical_bytes += 32
            model.access.forward_seconds += 0.5
        row = profile.snapshot()["layers"][0]
        self.assertEqual(row["counter_resets"], 3)
        self.assertEqual(row["cache_hits"], 2)
        self.assertEqual(row["disk_bytes_read"], 32)
        self.assertEqual(row["forward_seconds"], 0.5)

    def test_nonfinite_negative_and_unavailable_values_stay_json_safe(self):
        class BrokenReader:
            @property
            def logical_bytes(self):
                raise RuntimeError("counter unavailable")

        model = manager()
        profile = GlmLayerProfile(enabled=True)
        model.access.routing_seconds = float("nan")
        model.access.forward_seconds = float("inf")
        model.allocator_clears = -1
        model.reader = BrokenReader()
        with profile.measure(1, model):
            model.misses += 3
            model.hits = True
        result = profile.snapshot()
        row = result["layers"][0]
        self.assertEqual(row["cache_misses"], 3)
        self.assertEqual(row["cache_hits"], 0)
        self.assertEqual(row["routing_seconds"], 0.0)
        self.assertEqual(row["forward_seconds"], 0.0)
        self.assertEqual(row["allocator_clears"], 0)
        self.assertEqual(row["invalid_samples"], 7)
        json.dumps(result, allow_nan=False)

    def test_layer_bound_ignores_new_layers_and_preserves_existing_rows(self):
        model = manager()
        profile = GlmLayerProfile(enabled=True, max_layers=2)
        for layer in (1, 9, 15, -1, "invalid", True, 1):
            with profile.measure(layer, model):
                model.hits += 1
        result = profile.snapshot()
        self.assertEqual([row["layer"] for row in result["layers"]], [1, 9])
        self.assertEqual(result["ignored_calls"], 4)
        self.assertEqual(result["totals"]["cache_hits"], 3)
        self.assertEqual(result["totals"]["calls"], 3)

    def test_explicit_access_and_snapshot_are_independent(self):
        model = manager()
        explicit_access = manager().access
        profile = GlmLayerProfile(enabled=True)
        with profile.measure(2, model, explicit_access):
            explicit_access.forward_calls += 1
            model.access.forward_calls += 99
        snapshot = profile.snapshot()
        self.assertEqual(snapshot["layers"][0]["forward_calls"], 1)
        snapshot["layers"][0]["forward_calls"] = 900
        snapshot["totals"]["forward_calls"] = 900
        self.assertEqual(profile.snapshot()["layers"][0]["forward_calls"], 1)
        self.assertEqual(profile.snapshot()["totals"]["forward_calls"], 1)

    def test_summed_times_never_overflow_to_json_infinity(self):
        model = manager()
        profile = GlmLayerProfile(enabled=True)
        for _ in range(2):
            model.access.forward_seconds = 0.0
            with profile.measure(2, model):
                model.access.forward_seconds = 1.0e308
        snapshot = profile.snapshot()
        self.assertEqual(snapshot["totals"]["forward_seconds"],
                         profile_module.sys.float_info.max)
        json.dumps(snapshot, allow_nan=False)

    def test_invalid_configuration_is_rejected(self):
        for value in (0, -1, 1025, 1.5, True):
            with self.subTest(max_layers=value), self.assertRaises(ValueError):
                GlmLayerProfile(enabled=True, max_layers=value)
        with self.assertRaises(ValueError):
            GlmLayerProfile(enabled="false")


if __name__ == "__main__":
    unittest.main()
