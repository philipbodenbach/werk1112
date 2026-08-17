import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


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


class DiffusersCountParameterTests(unittest.TestCase):
    CASES = (
        ("image", "image_generation", "num_images", "num_images_per_prompt"),
        ("video", "video_generation", "num_videos", "num_videos_per_prompt"),
    )

    @staticmethod
    def guard(task, explicit):
        return COMPANION.ExplicitParameterGuard(
            {"explicit_parameters": explicit},
            task,
            "diffusers",
            {},
        )

    def test_explicit_batch_wins_over_resolved_task_count_default(self):
        for namespace, task, count_name, call_name in self.CASES:
            with self.subTest(namespace=namespace):
                batch_path = f"{namespace}.batch_size"
                result = COMPANION.diffusers_count_parameter(
                    namespace,
                    {count_name: 1, "batch_size": 3},
                    self.guard(task, [batch_path]),
                )

                self.assertEqual(result, (call_name, 3, batch_path))

    def test_task_count_wins_when_it_is_explicit_or_both_are_defaults(self):
        for namespace, task, count_name, call_name in self.CASES:
            count_path = f"{namespace}.{count_name}"
            for explicit in ([count_path], []):
                with self.subTest(namespace=namespace, explicit=explicit):
                    result = COMPANION.diffusers_count_parameter(
                        namespace,
                        {count_name: 2, "batch_size": 4},
                        self.guard(task, explicit),
                    )

                    self.assertEqual(result, (call_name, 2, count_path))

    def test_both_explicit_parameters_preserve_the_existing_conflict(self):
        for namespace, task, count_name, _call_name in self.CASES:
            count_path = f"{namespace}.{count_name}"
            batch_path = f"{namespace}.batch_size"
            with self.subTest(namespace=namespace):
                with self.assertRaises(COMPANION.CompanionFailure) as failure:
                    COMPANION.diffusers_count_parameter(
                        namespace,
                        {count_name: 2, "batch_size": 4},
                        self.guard(task, [count_path, batch_path]),
                    )

                self.assertEqual(failure.exception.code, "unsupported_parameter")
                self.assertEqual(
                    failure.exception.detail["parameters"],
                    [batch_path],
                )
                self.assertEqual(
                    failure.exception.detail["reasons"][batch_path],
                    f"it is overridden by explicit parameter '{count_path}'",
                )


class VideoBackendTests(unittest.TestCase):
    def test_null_schema_values_do_not_mask_video_estimate_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "model_index.json").write_text(
                json.dumps({"_class_name": "WanPipeline"}),
                encoding="utf-8",
            )

            estimate = COMPANION.command_estimate(
                {
                    "model_path": str(model),
                    "task": "image_to_video",
                    "effective_parameters": {
                        "video.width": 1216,
                        "video.height": 320,
                        "video.frames": 33,
                        "video.fps": 16.0,
                        "video.temporal_vae_tiling": True,
                        "video.window_size": None,
                        "video.bitrate": None,
                    },
                    "explicit_parameters": [
                        "video.width",
                        "video.height",
                        "video.frames",
                        "video.fps",
                        "video.temporal_vae_tiling",
                    ],
                }
            )

        self.assertIn(
            "1216x320, 33 frames at 16 fps, active window 33",
            estimate["assumptions"],
        )
        self.assertEqual(estimate["output_size_bytes"], 2_062_500)

    def test_adapter_accepts_diffusers_layout_and_rejects_native_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps({"architectures": ["ExampleVideoModel"]}),
                encoding="utf-8",
            )
            self.assertIsNone(
                COMPANION.execution_adapter(model, "video_generation")
            )

            (model / "model_index.json").write_text("{}", encoding="utf-8")
            self.assertIsNone(
                COMPANION.execution_adapter(model, "video_generation")
            )

            (model / "model_index.json").write_text(
                json.dumps({"_class_name": "ExampleVideoPipeline"}),
                encoding="utf-8",
            )
            self.assertEqual(
                COMPANION.execution_adapter(model, "video_generation"),
                "diffusers",
            )

    def test_probe_does_not_claim_image_pipeline_for_video(self):
        dependencies = {
            name: {"available": name in {"torch", "diffusers", "PIL", "av"}}
            for name in (
                "torch",
                "diffusers",
                "PIL",
                "av",
                "imageio",
                "imageio_ffmpeg",
                "ffmpeg",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "model_index.json").write_text(
                json.dumps({"_class_name": "StableDiffusionPipeline"}),
                encoding="utf-8",
            )
            with mock.patch.object(
                COMPANION,
                "dependency_snapshot",
                return_value=dependencies,
            ):
                result = COMPANION.command_probe_model(
                    {"model_path": str(model), "task": "video_generation"}
                )

        self.assertFalse(result["supported"])
        self.assertEqual(result["adapter"], "diffusers")
        self.assertTrue(
            any("does not advertise" in reason for reason in result["reasons"])
        )

    def test_video_dependencies_require_real_encoder_and_input_decoder(self):
        dependencies = {
            name: {"available": name in {"torch", "diffusers", "PIL", "imageio"}}
            for name in (
                "torch",
                "diffusers",
                "PIL",
                "av",
                "imageio",
                "imageio_ffmpeg",
                "ffmpeg",
            )
        }
        self.assertFalse(
            COMPANION.task_dependency_ready("video_generation", dependencies)[0]
        )

        dependencies["imageio_ffmpeg"]["available"] = True
        self.assertTrue(
            COMPANION.task_dependency_ready("video_generation", dependencies)[0]
        )
        self.assertTrue(
            COMPANION.task_dependency_ready("video_to_video", dependencies)[0]
        )

        dependencies["imageio"]["available"] = False
        self.assertFalse(
            COMPANION.task_dependency_ready("video_to_video", dependencies)[0]
        )
        dependencies["ffmpeg"]["available"] = True
        self.assertFalse(
            COMPANION.task_dependency_ready("video_to_video", dependencies)[0]
        )
        dependencies["av"]["available"] = True
        self.assertTrue(
            COMPANION.task_dependency_ready("video_to_video", dependencies)[0]
        )

    def test_task_registry_selects_related_image_to_video_pipeline(self):
        class WanImageToVideoPipeline:
            pass

        auto_pipeline = types.SimpleNamespace(
            AUTO_IMAGE2VIDEO_PIPELINES_MAPPING={
                "unrelated": object,
                "wan-i2v": WanImageToVideoPipeline,
            },
            _get_task_class=lambda *_args, **_kwargs: None,
            _get_model=lambda class_name: (
                "wan" if class_name == "WanPipeline" else None
            ),
        )
        diffusers = types.SimpleNamespace()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "model_index.json").write_text(
                json.dumps({"_class_name": "WanPipeline"}),
                encoding="utf-8",
            )
            with mock.patch.object(
                COMPANION.importlib,
                "import_module",
                return_value=auto_pipeline,
            ):
                selected = COMPANION.resolve_diffusers_video_pipeline_class(
                    diffusers,
                    model,
                    "image_to_video",
                )

        self.assertIs(selected, WanImageToVideoPipeline)

    def test_video_call_requests_portable_pil_frames(self):
        guard = COMPANION.ExplicitParameterGuard(
            {"explicit_parameters": []},
            "video_generation",
            "diffusers",
            {"prompt": "ocean"},
        )

        values, required, _seed, _paths = COMPANION.diffusers_call_values(
            "video_generation",
            {"prompt": "ocean"},
            {},
            object(),
            "cpu",
            guard,
        )

        self.assertEqual(values["output_type"], "pil")
        self.assertIn("prompt", required)

    @unittest.skipUnless(
        importlib.util.find_spec("PIL"),
        "Pillow is an optional media dependency",
    )
    def test_video_execution_reports_pipeline_and_output_metadata(self):
        from PIL import Image

        class FakeTorch:
            @staticmethod
            def inference_mode():
                return contextlib.nullcontext()

        class FakeVideoPipeline:
            def __call__(self, prompt, output_type=None):
                self.prompt = prompt
                self.output_type = output_type
                return types.SimpleNamespace(
                    frames=[
                        [Image.new("RGB", (10, 8), color) for color in (0, 127, 255)]
                    ]
                )

        pipeline = FakeVideoPipeline()
        entry = COMPANION.DiffusersPipelineEntry(
            pipeline,
            FakeTorch(),
            "cpu",
            "bf16",
            {"offload_mode": "none", "offload_request": "none"},
            COMPANION.DiffusersConfigurationOutcome(),
            0.25,
        )
        guard = COMPANION.ExplicitParameterGuard(
            {"explicit_parameters": []},
            "video_generation",
            "diffusers",
            {"prompt": "ocean", "frames": 3, "fps": 12.0},
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)

            def fake_export(_frames, path, _fps, _format):
                path.write_bytes(b"video")

            with mock.patch.object(
                COMPANION,
                "export_video",
                side_effect=fake_export,
            ):
                outputs, warnings, metadata = (
                    COMPANION.execute_prepared_diffusers_pipeline(
                        entry,
                        "cache-key",
                        False,
                        None,
                        entry.torch,
                        "cpu",
                        "bf16",
                        "video_generation",
                        {
                            "prompt": "ocean",
                            "frames": 3,
                            "fps": 12.0,
                            "output_format": "mp4",
                        },
                        {},
                        output_dir,
                        "test",
                        guard,
                    )
                )

        self.assertEqual(warnings, [])
        self.assertEqual(outputs[0]["mime_type"], "video/mp4")
        self.assertEqual(outputs[0]["width"], 10)
        self.assertEqual(outputs[0]["height"], 8)
        self.assertEqual(outputs[0]["duration"], 0.25)
        self.assertEqual(outputs[0]["metadata"], {"frames": 3, "fps": 12.0})
        self.assertEqual(metadata["pipeline_task"], "video_generation")
        self.assertEqual(metadata["pipeline_class"], "FakeVideoPipeline")
        self.assertEqual(metadata["dtype"], "bf16")

    def test_wan_vae_component_uses_fp32_without_changing_other_pipelines(self):
        torch = types.SimpleNamespace(float32="fp32")
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "model_index.json").write_text(
                json.dumps(
                    {
                        "_class_name": "WanPipeline",
                        "vae": ["diffusers", "AutoencoderKLWan"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                COMPANION.diffusers_load_dtype(model, torch, "bf16"),
                {"default": "bf16", "vae": "fp32"},
            )

            (model / "model_index.json").write_text(
                json.dumps(
                    {
                        "_class_name": "ExamplePipeline",
                        "vae": ["diffusers", "AutoencoderKL"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                COMPANION.diffusers_load_dtype(model, torch, "bf16"),
                "bf16",
            )

    def test_loader_passes_component_dtype_policy_to_diffusers(self):
        calls = []

        class Pipeline:
            @classmethod
            def from_pretrained(cls, model_path, **kwargs):
                calls.append((model_path, kwargs))
                return cls()

        torch = types.SimpleNamespace(float32="fp32")
        diffusers = types.SimpleNamespace(DiffusionPipeline=Pipeline)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "model_index.json").write_text(
                json.dumps(
                    {
                        "_class_name": "WanPipeline",
                        "vae": ["diffusers", "AutoencoderKLWan"],
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(COMPANION, "require_module", return_value=diffusers),
                mock.patch.object(
                    COMPANION,
                    "resolve_diffusers_video_pipeline_class",
                    return_value=None,
                ),
            ):
                COMPANION.load_diffusers_pipeline(
                    model,
                    "video_generation",
                    False,
                    torch,
                    "bf16",
                )

        self.assertEqual(
            calls[0][1]["torch_dtype"],
            {"default": "bf16", "vae": "fp32"},
        )

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"),
        "NumPy and Pillow are optional media dependencies",
    )
    def test_frame_batches_normalize_channels_last_and_channels_first(self):
        import numpy

        channels_last = numpy.zeros((2, 5, 8, 10, 3), dtype="uint8")
        batches = COMPANION.frame_batches(channels_last)
        self.assertEqual([len(batch) for batch in batches], [5, 5])
        self.assertEqual(COMPANION.ensure_pil_image(batches[0][0]).size, (10, 8))

        channels_first = numpy.zeros((2, 3, 5, 8, 10), dtype="float32")
        batches = COMPANION.frame_batches(channels_first)
        self.assertEqual([len(batch) for batch in batches], [5, 5])
        self.assertEqual(COMPANION.ensure_pil_image(batches[0][0]).size, (10, 8))

        short_bcfhw = numpy.zeros((1, 3, 1, 8, 10), dtype="float32")
        batches = COMPANION.frame_batches(short_bcfhw, expected_frames=1)
        self.assertEqual([len(batch) for batch in batches], [1])
        self.assertEqual(COMPANION.ensure_pil_image(batches[0][0]).size, (10, 8))

        short_fchw = numpy.zeros((1, 3, 8, 10), dtype="float32")
        batches = COMPANION.frame_batches(short_fchw, expected_frames=1)
        self.assertEqual([len(batch) for batch in batches], [1])
        self.assertEqual(COMPANION.ensure_pil_image(batches[0][0]).size, (10, 8))

    def test_ffmpeg_resolution_uses_imageio_ffmpeg_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "ffmpeg"
            executable.write_bytes(b"binary")
            module = types.SimpleNamespace(
                get_ffmpeg_exe=lambda: str(executable)
            )
            with (
                mock.patch.object(COMPANION.shutil, "which", return_value=None),
                mock.patch.object(
                    COMPANION.importlib,
                    "import_module",
                    return_value=module,
                ),
            ):
                self.assertEqual(
                    COMPANION.ffmpeg_executable(),
                    str(executable),
                )

    @unittest.skipUnless(
        importlib.util.find_spec("av")
        and importlib.util.find_spec("numpy")
        and importlib.util.find_spec("PIL"),
        "PyAV, NumPy and Pillow are optional video dependencies",
    )
    def test_pyav_encoder_writes_decodable_mp4(self):
        import av
        import numpy
        from PIL import Image

        frames = [
            Image.fromarray(numpy.full((16, 16, 3), value, dtype="uint8"))
            for value in (0, 127, 255)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "video.mp4"
            COMPANION.encode_video_with_pyav(frames, output, 12.0)
            with av.open(str(output)) as container:
                decoded = list(container.decode(video=0))
            loaded = COMPANION.load_video_frames(str(output), "source video")

            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(len(decoded), 3)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0].size, (16, 16))


class DiffusersPipelineCacheTests(unittest.TestCase):
    def test_lru_evicts_before_a_new_pipeline_is_loaded(self):
        cleaned = []
        cache = COMPANION.DiffusersPipelineCache(
            1,
            cleanup=lambda entry: cleaned.append(entry),
        )
        first = object()
        second = object()

        cache.prepare_for_load("first")
        cache.put("first", first)
        cache.prepare_for_load("second")

        self.assertEqual(cleaned, [first])
        self.assertEqual(len(cache), 0)
        cache.put("second", second)
        self.assertIs(cache.get("second"), second)

    def test_zero_size_disables_the_cache(self):
        cleaned = []
        cache = COMPANION.DiffusersPipelineCache(
            0,
            cleanup=lambda entry: cleaned.append(entry),
        )

        self.assertFalse(cache.put("model", object()))
        self.assertIsNone(cache.get("model"))
        self.assertEqual(cleaned, [])

    def test_cache_key_excludes_call_time_generation_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "model_index.json").write_text("{}", encoding="utf-8")
            lora = Path(directory) / "adapter.safetensors"
            lora.write_bytes(b"adapter")
            base = {
                "prompt": "first",
                "negative_prompt": "bad",
                "width": 512,
                "height": 512,
                "steps": 2,
                "guidance": 3.5,
                "seed": 1,
                "num_images": 1,
                "output_format": "png",
                "loras": [{"path": str(lora), "weight": 0.5}],
            }
            changed_call_values = dict(base)
            changed_call_values.update(
                prompt="second",
                negative_prompt="different",
                width=1024,
                height=768,
                steps=40,
                guidance=7.0,
                seed=99,
                num_images=4,
                output_format="jpeg",
            )

            first = COMPANION.diffusers_pipeline_cache_key(
                {"model": "example/model"},
                model,
                "image_generation",
                False,
                "cpu",
                "float32",
                base,
                COMPANION.normalized_lora_specs(base),
            )
            second = COMPANION.diffusers_pipeline_cache_key(
                {"model": "example/model"},
                model,
                "image_generation",
                False,
                "cpu",
                "float32",
                changed_call_values,
                COMPANION.normalized_lora_specs(changed_call_values),
            )
            different_lora = dict(base)
            different_lora["loras"] = [{"path": str(lora), "weight": 0.75}]
            third = COMPANION.diffusers_pipeline_cache_key(
                {"model": "example/model"},
                model,
                "image_generation",
                False,
                "cpu",
                "float32",
                different_lora,
                COMPANION.normalized_lora_specs(different_lora),
            )

            self.assertEqual(first, second)
            self.assertNotEqual(first, third)


class ResidentTransportTests(unittest.TestCase):
    def test_native_fd1_output_is_separated_from_protocol_stdout(self):
        code = f"""
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "werk_media_companion_child",
    {str(COMPANION_PATH)!r},
)
companion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(companion)
protocol, owned = companion.isolate_resident_protocol_output()
os.write(1, b"native-diagnostic\\n")
companion.write_json_line(protocol, {{"ok": True}})
if owned:
    protocol.close()
"""

        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
        )

        self.assertEqual(completed.stdout, b'{"ok":true}\n')
        self.assertIn(b"native-diagnostic\n", completed.stderr)

    def test_multiple_requests_share_runtime_and_shutdown_cleanly(self):
        requests = "\n".join(
            json.dumps(value)
            for value in (
                {
                    "transport_version": 1,
                    "request_id": "one",
                    "operation": "health",
                    "payload": {},
                },
                {
                    "transport_version": 1,
                    "request_id": "two",
                    "operation": "health",
                    "payload": {},
                },
                {
                    "transport_version": 1,
                    "request_id": "stop",
                    "operation": "shutdown",
                    "payload": {},
                },
            )
        )
        output = io.StringIO()
        runtime = COMPANION.CompanionRuntime(0)
        seen_runtimes = []

        def fake_dispatch(operation, payload, runtime=None):
            seen_runtimes.append(runtime)
            return {"status": operation, "payload": payload}

        with mock.patch.object(COMPANION, "dispatch", side_effect=fake_dispatch):
            COMPANION.serve_loop(io.StringIO(requests), output, runtime=runtime)

        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([item["request_id"] for item in responses], ["one", "two", "stop"])
        self.assertTrue(all(item["transport_version"] == 1 for item in responses))
        self.assertTrue(all(item["ok"] for item in responses))
        self.assertEqual(seen_runtimes, [runtime, runtime])
        self.assertEqual(responses[-1]["status"], "shutting_down")

    def test_malformed_line_does_not_terminate_the_worker(self):
        shutdown = json.dumps(
            {
                "transport_version": 1,
                "request_id": 2,
                "operation": "shutdown",
                "payload": {},
            }
        )
        output = io.StringIO()

        COMPANION.serve_loop(
            io.StringIO("{not json}\n" + shutdown + "\n"),
            output,
            runtime=COMPANION.CompanionRuntime(0),
        )

        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(responses[0]["error"]["code"], "invalid_json")
        self.assertFalse(responses[0]["ok"])
        self.assertEqual(responses[1]["request_id"], 2)
        self.assertTrue(responses[1]["ok"])

    def test_process_control_exception_terminates_worker_and_closes_runtime(self):
        request = json.dumps(
            {
                "transport_version": 1,
                "request_id": "interrupt",
                "operation": "execute",
                "payload": {},
            }
        )
        runtime = mock.Mock()
        output = io.StringIO()

        with (
            mock.patch.object(COMPANION, "dispatch", side_effect=SystemExit(17)),
            self.assertRaises(SystemExit),
        ):
            COMPANION.serve_loop(
                io.StringIO(request + "\n"),
                output,
                runtime=runtime,
            )

        runtime.close.assert_called_once_with()
        self.assertEqual(output.getvalue(), "")


class FakeTorch:
    def inference_mode(self):
        return contextlib.nullcontext()


class FakeImagePipeline:
    def __init__(self):
        self.calls = []
        self.device = None

    def set_progress_bar_config(self, **_kwargs):
        pass

    def to(self, device):
        self.device = device

    def __call__(self, prompt):
        self.calls.append(prompt)
        return type("Result", (), {"images": [object()]})()


class ResidentDiffusersExecutionTests(unittest.TestCase):
    @staticmethod
    def guard(parameters, *, explicit=(), policy="strict"):
        payload = {
            "explicit_parameters": list(explicit),
            "parameter_policy": policy,
        }
        if "negative_prompt" in parameters:
            payload["negative_prompt"] = parameters["negative_prompt"]
        return COMPANION.ExplicitParameterGuard(
            payload,
            "image_generation",
            "diffusers",
            parameters,
        )

    def execute(self, runtime, model, parameters, guard):
        return COMPANION.execute_diffusers(
            {"model": "fake/model"},
            model,
            "image_generation",
            parameters,
            {},
            model,
            "result",
            guard,
            runtime=runtime,
        )

    def test_success_then_unsupported_negative_prompt_then_success_loads_once(self):
        runtime = COMPANION.CompanionRuntime(1)
        pipeline = FakeImagePipeline()
        loads = []
        fake_torch = FakeTorch()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)

            def fake_load(*_args, **_kwargs):
                loads.append(True)
                return pipeline

            patches = (
                mock.patch.object(COMPANION, "torch_runtime", return_value=(fake_torch, "cpu", "float32")),
                mock.patch.object(COMPANION, "load_diffusers_pipeline", side_effect=fake_load),
                mock.patch.object(COMPANION, "synchronize_torch_device"),
                mock.patch.object(COMPANION, "save_images", return_value=[{"path": "image.png"}]),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                _outputs, _warnings, first = self.execute(
                    runtime,
                    model,
                    {"prompt": "first"},
                    self.guard({"prompt": "first"}),
                )
                rejected = {"prompt": "second", "negative_prompt": "without mouth"}
                with self.assertRaises(COMPANION.CompanionFailure) as failure:
                    self.execute(runtime, model, rejected, self.guard(rejected))
                _outputs, _warnings, third = self.execute(
                    runtime,
                    model,
                    {"prompt": "third"},
                    self.guard({"prompt": "third"}),
                )

            self.assertEqual(failure.exception.code, "unsupported_parameter")
            self.assertEqual(len(loads), 1)
            self.assertFalse(first["model_cache_hit"])
            self.assertTrue(third["model_cache_hit"])
            self.assertEqual(pipeline.calls, ["first", "third"])
        runtime.close()

    def test_pipeline_call_failure_evicts_the_cached_entry(self):
        class FailingPipeline(FakeImagePipeline):
            def __call__(self, prompt):
                if prompt == "boom":
                    raise RuntimeError("boom")
                return super().__call__(prompt)

        runtime = COMPANION.CompanionRuntime(1)
        loads = []
        fake_torch = FakeTorch()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)

            def fake_load(*_args, **_kwargs):
                pipeline = FailingPipeline()
                loads.append(pipeline)
                return pipeline

            with (
                mock.patch.object(COMPANION, "torch_runtime", return_value=(fake_torch, "cpu", "float32")),
                mock.patch.object(COMPANION, "load_diffusers_pipeline", side_effect=fake_load),
                mock.patch.object(COMPANION, "synchronize_torch_device"),
                mock.patch.object(COMPANION, "save_images", return_value=[{"path": "image.png"}]),
            ):
                with self.assertRaises(COMPANION.CompanionFailure):
                    self.execute(
                        runtime,
                        model,
                        {"prompt": "boom"},
                        self.guard({"prompt": "boom"}),
                    )
                _outputs, _warnings, metadata = self.execute(
                    runtime,
                    model,
                    {"prompt": "works"},
                    self.guard({"prompt": "works"}),
                )

            self.assertEqual(len(loads), 2)
            self.assertFalse(metadata["model_cache_hit"])
        runtime.close()

    def test_cached_configuration_replays_strict_explicit_rejection(self):
        runtime = COMPANION.CompanionRuntime(1)
        pipeline = FakeImagePipeline()
        loads = []
        fake_torch = FakeTorch()
        parameters = {"prompt": "image", "vae_tiling": True}
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            with (
                mock.patch.object(COMPANION, "torch_runtime", return_value=(fake_torch, "cpu", "float32")),
                mock.patch.object(
                    COMPANION,
                    "load_diffusers_pipeline",
                    side_effect=lambda *_args, **_kwargs: loads.append(True) or pipeline,
                ),
                mock.patch.object(COMPANION, "synchronize_torch_device"),
                mock.patch.object(COMPANION, "save_images", return_value=[{"path": "image.png"}]),
            ):
                _outputs, warnings, _metadata = self.execute(
                    runtime,
                    model,
                    parameters,
                    self.guard(parameters),
                )
                with self.assertRaises(COMPANION.CompanionFailure) as failure:
                    self.execute(
                        runtime,
                        model,
                        parameters,
                        self.guard(parameters, explicit=["image.vae_tiling"]),
                    )

            self.assertTrue(any("resolved defaults" in warning for warning in warnings))
            self.assertEqual(failure.exception.code, "unsupported_parameter")
            self.assertEqual(len(loads), 1)
        runtime.close()

    def test_uncached_success_releases_pipeline_for_one_shot_and_cache_zero(self):
        fake_torch = FakeTorch()
        for runtime in (None, COMPANION.CompanionRuntime(0)):
            with self.subTest(runtime="one-shot" if runtime is None else "cache-zero"):
                pipeline = FakeImagePipeline()
                with tempfile.TemporaryDirectory() as directory:
                    model = Path(directory)
                    with (
                        mock.patch.object(COMPANION, "torch_runtime", return_value=(fake_torch, "cpu", "float32")),
                        mock.patch.object(COMPANION, "load_diffusers_pipeline", return_value=pipeline),
                        mock.patch.object(COMPANION, "synchronize_torch_device"),
                        mock.patch.object(COMPANION, "save_images", return_value=[{"path": "image.png"}]),
                        mock.patch.object(COMPANION, "cleanup_diffusers_pipeline_entry") as cleanup,
                    ):
                        self.execute(
                            runtime,
                            model,
                            {"prompt": "success"},
                            self.guard({"prompt": "success"}),
                        )

                    cleanup.assert_called_once()
                    self.assertIs(cleanup.call_args.args[0].pipeline, pipeline)
                if runtime is not None:
                    runtime.close()

    def test_cache_zero_releases_pipeline_after_parameter_rejection(self):
        runtime = COMPANION.CompanionRuntime(0)
        pipeline = FakeImagePipeline()
        fake_torch = FakeTorch()
        parameters = {"prompt": "image", "negative_prompt": "unsupported"}
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            with (
                mock.patch.object(COMPANION, "torch_runtime", return_value=(fake_torch, "cpu", "float32")),
                mock.patch.object(COMPANION, "load_diffusers_pipeline", return_value=pipeline),
                mock.patch.object(COMPANION, "synchronize_torch_device"),
                mock.patch.object(COMPANION, "cleanup_diffusers_pipeline_entry") as cleanup,
            ):
                with self.assertRaises(COMPANION.CompanionFailure) as failure:
                    self.execute(runtime, model, parameters, self.guard(parameters))

            self.assertEqual(failure.exception.code, "unsupported_parameter")
            cleanup.assert_called_once()
            self.assertIs(cleanup.call_args.args[0].pipeline, pipeline)
        runtime.close()

    def test_cache_zero_releases_pipeline_after_pipeline_call_failure(self):
        class FailingPipeline(FakeImagePipeline):
            def __call__(self, prompt):
                raise RuntimeError(prompt)

        runtime = COMPANION.CompanionRuntime(0)
        pipeline = FailingPipeline()
        fake_torch = FakeTorch()
        parameters = {"prompt": "boom"}
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            with (
                mock.patch.object(COMPANION, "torch_runtime", return_value=(fake_torch, "cpu", "float32")),
                mock.patch.object(COMPANION, "load_diffusers_pipeline", return_value=pipeline),
                mock.patch.object(COMPANION, "synchronize_torch_device"),
                mock.patch.object(COMPANION, "cleanup_diffusers_pipeline_entry") as cleanup,
            ):
                with self.assertRaises(COMPANION.CompanionFailure) as failure:
                    self.execute(runtime, model, parameters, self.guard(parameters))

            self.assertEqual(failure.exception.code, "execution_failed")
            cleanup.assert_called_once()
            self.assertIs(cleanup.call_args.args[0].pipeline, pipeline)
        runtime.close()

    def test_cached_success_is_not_cleaned_until_runtime_closes(self):
        runtime = COMPANION.CompanionRuntime(1)
        pipeline = FakeImagePipeline()
        fake_torch = FakeTorch()
        parameters = {"prompt": "resident"}
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            with (
                mock.patch.object(COMPANION, "torch_runtime", return_value=(fake_torch, "cpu", "float32")),
                mock.patch.object(COMPANION, "load_diffusers_pipeline", return_value=pipeline),
                mock.patch.object(COMPANION, "synchronize_torch_device"),
                mock.patch.object(COMPANION, "save_images", return_value=[{"path": "image.png"}]),
                mock.patch.object(COMPANION, "cleanup_diffusers_pipeline_entry") as cleanup,
            ):
                runtime.pipeline_cache._cleanup = cleanup
                self.execute(runtime, model, parameters, self.guard(parameters))
                cleanup.assert_not_called()
                runtime.close()
                cleanup.assert_called_once()
                self.assertIs(cleanup.call_args.args[0].pipeline, pipeline)


if __name__ == "__main__":
    unittest.main()
