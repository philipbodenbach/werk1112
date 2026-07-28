import importlib.util
import tempfile
import unittest
from pathlib import Path


COMPANION_PATH = Path(__file__).with_name("werk_media_companion.py")
SPEC = importlib.util.spec_from_file_location("werk_media_companion", COMPANION_PATH)
COMPANION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPANION)


class Pipeline:
    def __init__(self, *, model_cpu=True, sequential=True):
        self.calls = []
        if not model_cpu:
            self.enable_model_cpu_offload = None
        if not sequential:
            self.enable_sequential_cpu_offload = None

    def set_progress_bar_config(self, **kwargs):
        self.calls.append(("progress", kwargs))

    def enable_model_cpu_offload(self):
        self.calls.append(("model_cpu",))

    def enable_sequential_cpu_offload(self):
        self.calls.append(("sequential_cpu",))

    def to(self, device):
        self.calls.append(("to", device))


class Guard:
    explicit = set()

    def __init__(self):
        self.rejected = []

    def reject(self, path, reason):
        self.rejected.append((path, reason))


def configure(pipeline, parameters, *, device="cuda", guard=None):
    guard = guard or Guard()
    metadata = COMPANION.configure_diffusers_pipeline(
        pipeline,
        device,
        "image_generation",
        parameters,
        [],
        guard,
    )
    return metadata, guard


class DiffusersOffloadMetadataTests(unittest.TestCase):
    def test_plain_device_move_reports_no_offload(self):
        pipeline = Pipeline()

        metadata, _guard = configure(pipeline, {})

        self.assertEqual(metadata, {"offload_mode": "none", "offload_request": "none"})
        self.assertIn(("to", "cuda"), pipeline.calls)

    def test_model_cpu_hook_reports_actual_mode(self):
        pipeline = Pipeline()

        metadata, _guard = configure(
            pipeline,
            {"_werk_enable_cpu_offload": True},
        )

        self.assertEqual(
            metadata,
            {"offload_mode": "model_cpu", "offload_request": "model_cpu"},
        )
        self.assertIn(("model_cpu",), pipeline.calls)
        self.assertNotIn(("to", "cuda"), pipeline.calls)

    def test_sequential_hook_reports_actual_mode(self):
        pipeline = Pipeline()

        metadata, _guard = configure(
            pipeline,
            {"_werk_enable_sequential_offload": True},
        )

        self.assertEqual(
            metadata,
            {"offload_mode": "sequential_cpu", "offload_request": "sequential_cpu"},
        )
        self.assertIn(("sequential_cpu",), pipeline.calls)

    def test_component_request_reports_model_cpu_hook_truthfully(self):
        pipeline = Pipeline()

        metadata, _guard = configure(
            pipeline,
            {"_werk_enable_component_offload": True},
        )

        self.assertEqual(
            metadata,
            {"offload_mode": "model_cpu", "offload_request": "component"},
        )
        self.assertIn(("model_cpu",), pipeline.calls)

    def test_non_cuda_offload_request_fails_instead_of_moving_pipeline(self):
        pipeline = Pipeline()

        with self.assertRaises(COMPANION.CompanionFailure) as failure:
            configure(
                pipeline,
                {"_werk_enable_component_offload": True},
                device="cpu",
            )

        self.assertEqual(failure.exception.code, "backend_configuration_failed")
        self.assertNotIn(("model_cpu",), pipeline.calls)
        self.assertNotIn(("to", "cpu"), pipeline.calls)

    def test_missing_hook_fails_instead_of_moving_pipeline_to_cuda(self):
        pipeline = Pipeline(model_cpu=False)

        with self.assertRaises(COMPANION.CompanionFailure) as failure:
            configure(
                pipeline,
                {"_werk_enable_component_offload": True},
            )

        self.assertEqual(failure.exception.code, "backend_configuration_failed")
        self.assertNotIn(("to", "cuda"), pipeline.calls)


class AdapterOffloadValidationTests(unittest.TestCase):
    def test_non_diffusers_adapters_reject_each_selected_offload_mode(self):
        cases = (
            ("_werk_enable_cpu_offload", "model_cpu"),
            ("_werk_enable_sequential_offload", "sequential_cpu"),
            ("_werk_enable_component_offload", "component"),
        )
        for adapter in ("transformers_audio", "transformers_tts", "transformers_asr"):
            for flag, expected_request in cases:
                with self.subTest(adapter=adapter, flag=flag):
                    with self.assertRaises(COMPANION.CompanionFailure) as failure:
                        COMPANION.validate_adapter_offload(adapter, {flag: True})
                    self.assertEqual(
                        failure.exception.code,
                        "backend_configuration_failed",
                    )
                    self.assertEqual(
                        failure.exception.detail["offload_request"],
                        expected_request,
                    )

    def test_non_diffusers_adapter_accepts_no_selected_offload(self):
        request = COMPANION.validate_adapter_offload(
            "transformers_asr",
            {"routing.allow_cpu_offload": True},
        )

        self.assertEqual(request, "none")

    def test_execute_rejects_selected_offload_before_loading_transformers(self):
        with tempfile.TemporaryDirectory() as model_path:
            with self.assertRaises(COMPANION.CompanionFailure) as failure:
                COMPANION.command_execute(
                    {
                        "model_path": model_path,
                        "task": "speech_to_text",
                        "effective_parameters": {
                            "_werk_enable_cpu_offload": True,
                        },
                    }
                )

        self.assertEqual(failure.exception.code, "backend_configuration_failed")
        self.assertEqual(failure.exception.detail["adapter"], "transformers_asr")


if __name__ == "__main__":
    unittest.main()
