"""Dependency-free contract checks; real CUDA smoke tests run separately."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("worker", Path(__file__).with_name("werk_text_analysis.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

class TextAnalysisTests(unittest.TestCase):
    def test_load_and_inference_timings_are_separate_on_cold_and_warm_requests(self):
        request = {"runtime": "transformers", "options": {"max_length": 512}}
        loaded = {"device": "cuda", "dtype": "torch.bfloat16"}
        with patch.object(worker, "_loaded", None), \
             patch.object(worker, "load", return_value=loaded) as load, \
             patch.object(worker, "infer", side_effect=[({}, []), ({}, [])]), \
             patch.object(worker.time, "monotonic", side_effect=[10., 12., 12.5, 13., 13., 13.2]):
            cold = worker.execute(request)["werk"]
            warm = worker.execute(request)["werk"]
            self.assertEqual(cold["load_seconds"], 2.)
            self.assertEqual(cold["inference_seconds"], 0.5)
            self.assertEqual(cold["total_seconds"], 2.5)
            self.assertFalse(cold["model_cache_hit"])
            self.assertEqual(warm["load_seconds"], 0.)
            self.assertAlmostEqual(warm["inference_seconds"], 0.2)
            self.assertTrue(warm["model_cache_hit"])
            load.assert_called_once()

    def test_input_errors_do_not_request_runtime_fallback(self):
        with patch.object(worker, "execute", side_effect=worker.InputError("split input")):
            self.assertEqual(worker.dispatch("execute", {}), {"ok": True, "invalid_input": "split input"})

    def test_runtime_errors_include_mitigation(self):
        with patch.object(worker, "execute", side_effect=RuntimeError("CUDA out of memory")):
            response = worker.dispatch("execute", {})
            self.assertFalse(response["ok"])
            self.assertIn("WERK_TEXT_PYTHON", response["error"]["message"])
            self.assertIn("CUDA out of memory", response["error"]["message"])

    def test_nonfinite_output_cannot_be_returned_as_success(self):
        for values in [[float("nan")], [[0.2, float("inf")]]]:
            with self.assertRaises(RuntimeError):
                worker.checked_numbers(values)

    def test_long_input_requires_explicit_truncation(self):
        tokenizer = lambda *args, **kwargs: {"input_ids": [[1] * 20]}
        with self.assertRaises(worker.InputError):
            worker.check_lengths(tokenizer, ["text"], {"max_length": 8})
        worker.check_lengths(tokenizer, ["text"], {"max_length": 8, "truncate": True})

if __name__ == "__main__":
    unittest.main()
