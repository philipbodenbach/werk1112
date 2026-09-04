import contextlib
import importlib.util
import io
import json
import os
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


class TorchAcceleratorSelectionTests(unittest.TestCase):
    @staticmethod
    def fake_torch(*, hip=None, cuda=None, architecture="gfx1151", available=True):
        class FakeCuda:
            @staticmethod
            def is_available():
                return available

            @staticmethod
            def device_count():
                return 1 if available else 0

            @staticmethod
            def get_device_name(_index):
                return "AMD Radeon 8060S" if hip else "NVIDIA GPU"

            @staticmethod
            def get_device_properties(_index):
                return types.SimpleNamespace(gcnArchName=architecture)

        return types.SimpleNamespace(
            __version__="2.9.1",
            version=types.SimpleNamespace(hip=hip, cuda=cuda),
            cuda=FakeCuda(),
            backends=types.SimpleNamespace(mps=None),
            float16="float16",
            bfloat16="bfloat16",
            float32="float32",
        )

    def test_explicit_rocm_requires_a_hip_pytorch_build(self):
        torch = self.fake_torch(hip=None, cuda="13.0")
        with mock.patch.object(COMPANION, "require_module", return_value=torch):
            with self.assertRaises(COMPANION.CompanionFailure) as failure:
                COMPANION.torch_runtime({"accelerator": "rocm"})

        self.assertEqual(failure.exception.code, "accelerator_unavailable")
        self.assertIn("not ROCm-capable", failure.exception.message)
        self.assertIn("CUDA build", failure.exception.detail)

    def test_explicit_cuda_rejects_a_rocm_pytorch_build(self):
        torch = self.fake_torch(hip="7.2.1", cuda=None)
        with mock.patch.object(COMPANION, "require_module", return_value=torch):
            with self.assertRaises(COMPANION.CompanionFailure) as failure:
                COMPANION.torch_runtime({"accelerator": "cuda"})

        self.assertEqual(failure.exception.code, "accelerator_unavailable")
        self.assertIn("use accelerator=rocm", failure.exception.message)
        self.assertIn("7.2.1", failure.exception.detail)

    def test_strix_halo_rocm_auto_precision_is_fp16(self):
        torch = self.fake_torch(hip="7.2.1", cuda=None)
        with mock.patch.object(COMPANION, "require_module", return_value=torch):
            selected_torch, device, dtype = COMPANION.torch_runtime(
                {"accelerator": "rocm", "precision": "auto"}
            )

        self.assertIs(selected_torch, torch)
        self.assertEqual(device, "cuda")
        self.assertEqual(dtype, "float16")

    def test_strix_halo_rocm_keeps_explicit_bf16_without_an_artificial_limit(self):
        torch = self.fake_torch(hip="7.2.1", cuda=None)
        with mock.patch.object(COMPANION, "require_module", return_value=torch):
            _selected_torch, device, dtype = COMPANION.torch_runtime(
                {"accelerator": "rocm", "precision": "bf16"}
            )

        self.assertEqual(device, "cuda")
        self.assertEqual(dtype, "bfloat16")

    def test_health_snapshot_identifies_strix_halo_as_rocm_not_cuda(self):
        torch = self.fake_torch(hip="7.2.1", cuda=None)
        with mock.patch.object(
            COMPANION.importlib,
            "import_module",
            side_effect=lambda name: torch if name == "torch" else None,
        ):
            snapshot = COMPANION.accelerator_snapshot()

        self.assertTrue(snapshot["rocm"]["available"])
        self.assertFalse(snapshot["cuda"]["available"])
        self.assertEqual(snapshot["rocm"]["version"], "7.2.1")
        self.assertIn("gfx1151", snapshot["rocm"]["detail"])
        self.assertIn("FP16", snapshot["rocm"]["detail"])
        self.assertIn("uses ROCM, not CUDA", snapshot["cuda"]["detail"])


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
            (Path(model_path) / "config.json").write_text(
                json.dumps({"model_type": "whisper"}),
                encoding="utf-8",
            )
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


class AudioRuntimeTests(unittest.TestCase):
    class RuntimeTorch:
        def __init__(self):
            self.seed = None
            self.cuda = types.SimpleNamespace(
                manual_seed_all=lambda value: setattr(self, "cuda_seed", value)
            )

        def inference_mode(self):
            return contextlib.nullcontext()

        def manual_seed(self, value):
            self.seed = value

    @staticmethod
    def guard(task, adapter, parameters, *, explicit=(), policy="strict"):
        return COMPANION.ExplicitParameterGuard(
            {
                "explicit_parameters": list(explicit),
                "parameter_policy": policy,
            },
            task,
            adapter,
            parameters,
        )

    def test_estimate_uses_canonical_audio_variations_and_legacy_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps({"model_type": "fixture"}),
                encoding="utf-8",
            )
            base = {
                "model_path": str(model),
                "task": "audio_generation",
                "effective_parameters": {
                    "audio.duration": 2,
                    "audio.sample_rate": 8_000,
                    "audio.channels": 1,
                    "audio.bit_depth": 16,
                },
            }
            canonical = dict(base)
            canonical["effective_parameters"] = {
                **base["effective_parameters"],
                "audio.variations": 3,
            }
            legacy = dict(base)
            legacy["effective_parameters"] = {
                **base["effective_parameters"],
                "num_variations": 2,
            }

            canonical_estimate = COMPANION.command_estimate(canonical)
            legacy_estimate = COMPANION.command_estimate(legacy)

        self.assertEqual(canonical_estimate["output_size_bytes"], 96_000)
        self.assertEqual(legacy_estimate["output_size_bytes"], 64_000)

    def test_pipeline_sample_rate_finds_nested_audio_processor_feature_extractor(self):
        feature_extractor = types.SimpleNamespace(sampling_rate=24_000)
        audio_processor = types.SimpleNamespace(
            feature_extractor=feature_extractor,
        )
        processor = types.SimpleNamespace(audio_processor=audio_processor)
        pipeline = types.SimpleNamespace(processor=processor)

        self.assertEqual(COMPANION.pipeline_sample_rate(pipeline), 24_000)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is an optional audio dependency",
    )
    def test_audio_normalization_preserves_more_than_thirty_two_channels(self):
        import numpy

        audio = numpy.linspace(-1.0, 1.0, 40 * 64, dtype="float32").reshape(40, 64)

        normalized, interleaved, channels, frames = (
            COMPANION.normalized_audio_array(audio)
        )

        self.assertEqual(normalized.shape, (64, 40))
        numpy.testing.assert_array_equal(normalized, audio.T)
        self.assertEqual(channels, 40)
        self.assertEqual(frames, 64)
        self.assertEqual(interleaved.size, 64 * 40)

    def test_text_to_audio_loader_does_not_duplicate_local_files_only(self):
        calls = []

        def fake_pipeline(
            *,
            task,
            model,
            device,
            trust_remote_code,
            model_kwargs=None,
            **hub_kwargs,
        ):
            # Mirrors the relevant Transformers factory behavior: it owns a
            # hub-level local_files_only value and separately expands caller
            # model kwargs into the same from_pretrained call.
            owned_hub_kwargs = {
                "local_files_only": hub_kwargs.get("local_files_only", False),
            }

            def from_pretrained(**kwargs):
                calls.append(kwargs)

            from_pretrained(**owned_hub_kwargs, **(model_kwargs or {}))
            calls.append(
                {
                    "task": task,
                    "model": model,
                    "device": device,
                    "trust_remote_code": trust_remote_code,
                }
            )
            return object()

        fake_transformers = types.SimpleNamespace(pipeline=fake_pipeline)
        runtime_values = (self.RuntimeTorch(), "cpu", "float32")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            COMPANION,
            "require_module",
            return_value=fake_transformers,
        ):
            pipeline, _torch, device, dtype = COMPANION.load_transformers_pipeline(
                Path(directory),
                "text-to-audio",
                {},
                runtime_values=runtime_values,
            )

        self.assertIsNotNone(pipeline)
        self.assertEqual(device, "cpu")
        self.assertEqual(dtype, "float32")
        self.assertEqual(calls[0], {"local_files_only": False})
        self.assertEqual(calls[1]["task"], "text-to-audio")
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is an optional audio dependency",
    )
    def test_transformers_tts_writes_pipeline_audio(self):
        import numpy

        class TextToSpeechPipeline:
            def __init__(self):
                self.text = None

            def __call__(self, text):
                self.text = text
                return {
                    "audio": numpy.linspace(-0.25, 0.25, 16, dtype="float32"),
                    "sampling_rate": 16_000,
                }

        pipeline = TextToSpeechPipeline()
        torch = self.RuntimeTorch()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            COMPANION,
            "load_transformers_pipeline",
            return_value=(pipeline, torch, "cpu", "float32"),
        ):
            outputs, warnings, metadata = COMPANION.execute_tts(
                Path(directory),
                "text_to_speech",
                {
                    "text": "Hello from Werk.",
                    "output_format": "wav",
                    "seed": 23,
                },
                Path(directory),
                "test",
            )

        self.assertEqual(pipeline.text, "Hello from Werk.")
        self.assertEqual(outputs[0]["mime_type"], "audio/wav")
        self.assertEqual(outputs[0]["metadata"]["sample_rate"], 16_000)
        self.assertEqual(warnings, [])
        self.assertEqual(metadata["pipeline_task"], "text-to-audio")
        self.assertEqual(torch.seed, 23)
        self.assertEqual(metadata["seed"], 23)
        self.assertIn(
            "tts.seed",
            COMPANION.supported_explicit_parameters(
                "text_to_speech",
                "transformers_tts",
            ),
        )
        self.assertIn(
            "tts.seed->torch.manual_seed",
            metadata["translated_parameters"],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is an optional audio dependency",
    )
    def test_transformers_generation_applies_duration_and_splits_variations(self):
        import numpy

        class GenerationPipeline:
            def __init__(self):
                self.model = types.SimpleNamespace(
                    config=types.SimpleNamespace(
                        model_type="musicgen",
                        audio_encoder=types.SimpleNamespace(frame_rate=50),
                    )
                )
                self.calls = []

            def __call__(self, prompts, generate_kwargs=None):
                self.calls.append((prompts, generate_kwargs))
                return [
                    {
                        "audio": numpy.zeros(16, dtype="float32"),
                        "sampling_rate": 8_000,
                    },
                    {
                        "audio": numpy.full(16, 0.5, dtype="float32"),
                        "sampling_rate": 8_000,
                    },
                ]

        pipeline = GenerationPipeline()
        torch = self.RuntimeTorch()
        parameters = {
            "prompt": "short drum loop",
            "duration": 2.0,
            "variations": 2,
            "output_format": "wav",
            "seed": 17,
            "temperature": 0.0,
        }
        guard = self.guard(
            "music_generation",
            "transformers_audio",
            parameters,
            explicit=("audio.duration", "audio.variations"),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            COMPANION,
            "load_transformers_pipeline",
            return_value=(pipeline, torch, "cpu", "float32"),
        ):
            outputs, warnings, metadata = COMPANION.execute_audio_generation(
                Path(directory),
                "music_generation",
                parameters,
                Path(directory),
                "test",
                guard,
            )

        prompts, generate_kwargs = pipeline.calls[0]
        self.assertEqual(prompts, ["short drum loop", "short drum loop"])
        self.assertEqual(generate_kwargs["max_new_tokens"], 100)
        self.assertFalse(generate_kwargs["do_sample"])
        self.assertEqual(torch.seed, 17)
        self.assertEqual(len(outputs), 2)
        self.assertEqual([item["metadata"]["channels"] for item in outputs], [1, 1])
        self.assertEqual(warnings, [])
        self.assertIn(
            "audio.duration->generate_kwargs.max_new_tokens",
            metadata["translated_parameters"],
        )
        self.assertIn(
            "audio.variations->batched prompts",
            metadata["translated_parameters"],
        )
        self.assertEqual(
            metadata["duration_control"],
            {
                "requested_seconds": 2.0,
                "audio_tokens": 100,
                "hard_limit_applied": False,
                "model_default_seconds": 30.0,
                "exceeds_model_default": False,
            },
        )

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is an optional audio dependency",
    )
    def test_transformers_variations_do_not_treat_stereo_as_a_batch(self):
        import numpy

        class StereoPipeline:
            sampling_rate = 8_000
            model = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    model_type="audiogen",
                    audio_encoder=types.SimpleNamespace(frame_rate=50),
                )
            )

            def __call__(self, _prompts):
                return {
                    "audio": numpy.zeros((2, 16), dtype="float32"),
                    "sampling_rate": self.sampling_rate,
                }

        pipeline = StereoPipeline()
        torch = self.RuntimeTorch()
        parameters = {
            "prompt": "stereo ambience",
            "variations": 2,
            "output_format": "wav",
        }
        guard = self.guard(
            "audio_generation",
            "transformers_audio",
            parameters,
            explicit=("audio.variations",),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            COMPANION,
            "load_transformers_pipeline",
            return_value=(pipeline, torch, "cpu", "float32"),
        ):
            with self.assertRaises(COMPANION.CompanionFailure) as failure:
                COMPANION.execute_audio_generation(
                    Path(directory),
                    "audio_generation",
                    parameters,
                    Path(directory),
                    "test",
                    guard,
                )

        self.assertEqual(failure.exception.code, "backend_error")

    def test_musicgen_forwards_explicit_duration_above_sixty_seconds(self):
        pipeline = types.SimpleNamespace(
            model=types.SimpleNamespace(
                config=types.SimpleNamespace(
                    model_type="musicgen",
                    audio_encoder=types.SimpleNamespace(frame_rate=50),
                )
            )
        )
        parameters = {"duration": 120.0}
        guard = self.guard(
            "music_generation",
            "transformers_audio",
            parameters,
            explicit=("audio.duration",),
        )

        warnings = []
        tokens = COMPANION.transformer_duration_tokens(
            pipeline,
            parameters,
            guard,
            warnings,
        )

        self.assertEqual(tokens, 6000)
        self.assertEqual(guard.unsupported, {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("exceeds MusicGen's 30-second model default", warnings[0])
        self.assertIn("does not impose a hard duration limit", warnings[0])
        self.assertEqual(
            COMPANION.transformer_audio_duration_metadata(
                pipeline,
                parameters,
                tokens,
            ),
            {
                "requested_seconds": 120.0,
                "audio_tokens": 6000,
                "hard_limit_applied": False,
                "model_default_seconds": 30.0,
                "exceeds_model_default": True,
            },
        )

    def test_transformers_duration_still_requires_a_positive_finite_value(self):
        pipeline = types.SimpleNamespace(
            model=types.SimpleNamespace(
                config=types.SimpleNamespace(
                    model_type="musicgen",
                    audio_encoder=types.SimpleNamespace(frame_rate=50),
                )
            )
        )
        for duration in (0, -1, float("inf"), float("nan")):
            parameters = {"duration": duration}
            guard = self.guard(
                "music_generation",
                "transformers_audio",
                parameters,
                explicit=("audio.duration",),
            )
            with self.subTest(duration=duration), self.assertRaises(
                COMPANION.CompanionFailure
            ) as failure:
                COMPANION.transformer_duration_tokens(
                    pipeline,
                    parameters,
                    guard,
                    [],
                )

            self.assertEqual(failure.exception.code, "invalid_parameter")

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is an optional audio dependency",
    )
    def test_diffusers_audio_maps_stable_audio_duration_and_variations(self):
        import numpy

        class StableAudioPipeline:
            sampling_rate = 8_000

            def __init__(self):
                self.call = None

            def __call__(
                self,
                prompt,
                audio_end_in_s=None,
                num_waveforms_per_prompt=None,
                output_type=None,
            ):
                self.call = {
                    "prompt": prompt,
                    "audio_end_in_s": audio_end_in_s,
                    "num_waveforms_per_prompt": num_waveforms_per_prompt,
                    "output_type": output_type,
                }
                return types.SimpleNamespace(
                    audios=numpy.stack(
                        [
                            numpy.zeros(12, dtype="float32"),
                            numpy.ones(12, dtype="float32") * 0.25,
                        ]
                    )
                )

        pipeline = StableAudioPipeline()
        torch = self.RuntimeTorch()
        entry = COMPANION.DiffusersPipelineEntry(
            pipeline,
            torch,
            "cpu",
            "float32",
            {"offload_mode": "none", "offload_request": "none"},
            None,
            0.1,
        )
        parameters = {
            "prompt": "ocean ambience",
            "duration": 1.5,
            "variations": 2,
            "output_format": "wav",
        }
        guard = self.guard(
            "audio_generation",
            "diffusers_audio",
            parameters,
            explicit=("audio.duration", "audio.variations"),
        )
        with tempfile.TemporaryDirectory() as directory:
            outputs, warnings, metadata = (
                COMPANION.execute_prepared_diffusers_audio(
                    entry,
                    "cache-key",
                    False,
                    None,
                    torch,
                    "cpu",
                    "float32",
                    "audio_generation",
                    parameters,
                    Path(directory),
                    "test",
                    guard,
                )
            )

        self.assertEqual(pipeline.call["audio_end_in_s"], 1.5)
        self.assertEqual(pipeline.call["num_waveforms_per_prompt"], 2)
        self.assertEqual(pipeline.call["output_type"], "np")
        self.assertEqual(len(outputs), 2)
        self.assertEqual(warnings, [])
        self.assertIn(
            "audio.duration->audio_end_in_s",
            metadata["translated_parameters"],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is an optional audio dependency",
    )
    def test_speech_translation_uses_local_waveform_and_translate_generation(self):
        import numpy

        class WhisperPipeline:
            type = "seq2seq_whisper"
            feature_extractor = types.SimpleNamespace(sampling_rate=16_000)
            model = types.SimpleNamespace(can_generate=lambda: True)

            def __init__(self):
                self.call = None

            def __call__(self, audio, generate_kwargs=None):
                self.call = (audio, generate_kwargs)
                return {"text": "translated speech"}

        pipeline = WhisperPipeline()
        torch = self.RuntimeTorch()
        parameters = {
            "output_format": "json",
            "language": "de",
            "beam_size": 3,
        }
        guard = self.guard(
            "speech_translation",
            "transformers_asr",
            parameters,
            explicit=("stt.language", "stt.beam_size"),
        )
        decoded = {
            "raw": numpy.zeros(32, dtype="float32"),
            "sampling_rate": 16_000,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "speech.wav"
            source.write_bytes(b"fixture")
            with (
                mock.patch.object(
                    COMPANION,
                    "load_transformers_pipeline",
                    return_value=(pipeline, torch, "cpu", "float32"),
                ),
                mock.patch.object(
                    COMPANION,
                    "decode_local_audio",
                    return_value=decoded,
                ) as decoder,
            ):
                outputs, warnings, metadata = COMPANION.execute_asr(
                    Path(directory),
                    "speech_translation",
                    parameters,
                    {"audio": str(source)},
                    Path(directory),
                    "test",
                    guard,
                )

        decoder.assert_called_once_with(source.resolve(), 16_000)
        self.assertIs(pipeline.call[0], decoded)
        self.assertEqual(pipeline.call[1]["task"], "translate")
        self.assertEqual(pipeline.call[1]["language"], "de")
        self.assertEqual(pipeline.call[1]["num_beams"], 3)
        self.assertEqual(outputs[0]["mime_type"], "application/json")
        self.assertEqual(warnings, [])
        self.assertIn(
            "stt.operation->generate_kwargs.task",
            metadata["translated_parameters"],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is an optional audio dependency",
    )
    def test_classification_decodes_locally_and_writes_sorted_json(self):
        import numpy

        class ClassificationPipeline:
            feature_extractor = types.SimpleNamespace(sampling_rate=16_000)

            def __init__(self):
                self.call = None

            def __call__(self, audio, top_k=None):
                self.call = (audio, top_k)
                return [
                    {"label": "silence", "score": 0.1},
                    {"label": "speech", "score": 0.9},
                ]

        pipeline = ClassificationPipeline()
        torch = self.RuntimeTorch()
        parameters = {
            # The shared schema's materialized audio default must not produce
            # a WAV for a structured analysis task.
            "output_format": "wav",
            "top_k": 2,
        }
        guard = self.guard(
            "voice_activity_detection",
            "transformers_audio_classification",
            parameters,
            explicit=("audio.top_k",),
        )
        decoded = {
            "raw": numpy.zeros(24, dtype="float32"),
            "sampling_rate": 16_000,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "speech.wav"
            source.write_bytes(b"fixture")
            with (
                mock.patch.object(
                    COMPANION,
                    "load_transformers_pipeline",
                    return_value=(pipeline, torch, "cpu", "float32"),
                ),
                mock.patch.object(
                    COMPANION,
                    "decode_local_audio",
                    return_value=decoded,
                ) as decoder,
            ):
                outputs, warnings, metadata = (
                    COMPANION.execute_audio_classification(
                        Path(directory),
                        "voice_activity_detection",
                        parameters,
                        {"audio": str(source)},
                        Path(directory),
                        "test",
                        guard,
                    )
                )
                contents = json.loads(Path(outputs[0]["path"]).read_text("utf-8"))

        decoder.assert_called_once_with(source.resolve(), 16_000)
        self.assertIs(pipeline.call[0], decoded)
        self.assertEqual(pipeline.call[1], 2)
        self.assertEqual(
            [item["label"] for item in contents["labels"]],
            ["speech", "silence"],
        )
        self.assertEqual(outputs[0]["mime_type"], "application/json")
        self.assertEqual(warnings, [])
        self.assertEqual(metadata["pipeline_task"], "audio-classification")

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is an optional audio dependency",
    )
    def test_any_to_any_audio_controls_are_forwarded_as_generation_kwargs(self):
        import numpy

        class AudioTextPipeline:
            feature_extractor = types.SimpleNamespace(sampling_rate=16_000)

            def __init__(self):
                self.call = None

            def __call__(
                self,
                prompt,
                audio=None,
                max_new_tokens=None,
                generate_kwargs=None,
                return_full_text=None,
            ):
                self.call = {
                    "prompt": prompt,
                    "audio": audio,
                    "max_new_tokens": max_new_tokens,
                    "generate_kwargs": generate_kwargs,
                    "return_full_text": return_full_text,
                }
                return [{"generated_text": "A dog barks twice."}]

        pipeline = AudioTextPipeline()
        torch = self.RuntimeTorch()
        parameters = {
            "prompt": "Describe the sound.",
            "max_new_tokens": 12,
            "temperature": 0.7,
            "top_k": 8,
            "top_p": 0.9,
            "output_format": "json",
        }
        guard = self.guard(
            "audio_understanding",
            "transformers_audio_text",
            parameters,
        )
        decoded = {
            "raw": numpy.zeros(24, dtype="float32"),
            "sampling_rate": 16_000,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sound.wav"
            source.write_bytes(b"fixture")
            with (
                mock.patch.object(
                    COMPANION,
                    "load_transformers_pipeline",
                    return_value=(pipeline, torch, "cpu", "float32"),
                ),
                mock.patch.object(
                    COMPANION,
                    "decoded_audio_for_pipeline",
                    return_value=decoded,
                ),
            ):
                outputs, _warnings, metadata = COMPANION.execute_audio_text(
                    Path(directory),
                    "audio_understanding",
                    parameters,
                    {"audio": str(source)},
                    Path(directory),
                    "test",
                    guard,
                )

        self.assertIs(pipeline.call["audio"], decoded["raw"])
        self.assertEqual(pipeline.call["max_new_tokens"], 12)
        self.assertFalse(pipeline.call["return_full_text"])
        self.assertEqual(
            pipeline.call["generate_kwargs"],
            {
                "temperature": 0.7,
                "top_k": 8,
                "top_p": 0.9,
                "do_sample": True,
            },
        )
        self.assertEqual(outputs[0]["mime_type"], "application/json")
        self.assertEqual(metadata["text"], "A dog barks twice.")

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"),
        "NumPy and PyTorch are optional audio dependencies",
    )
    def test_embedding_accepts_multidimensional_tensor_without_bool_coercion(self):
        import numpy
        import torch

        class Processor:
            feature_extractor = types.SimpleNamespace(sampling_rate=16_000)

            def __call__(self, *, audio, sampling_rate=None, return_tensors=None):
                self.audio = audio
                self.call = (sampling_rate, return_tensors)
                return {"input_values": torch.zeros((1, 8))}

        class EmbeddingModel:
            def __call__(self, **_inputs):
                return types.SimpleNamespace(
                    audio_embeds=torch.tensor([[3.0, 4.0]])
                )

        processor = Processor()
        model = EmbeddingModel()
        parameters = {
            "normalize": True,
            "pooling": "mean",
            "output_format": "wav",
        }
        guard = self.guard(
            "audio_embedding",
            "transformers_audio_embedding",
            parameters,
        )
        decoded = {
            "raw": numpy.zeros(24, dtype="float32"),
            "sampling_rate": 16_000,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sound.wav"
            source.write_bytes(b"fixture")
            with (
                mock.patch.object(
                    COMPANION,
                    "load_transformers_audio_embedder",
                    return_value=(processor, model, torch, "cpu", torch.float32),
                ),
                mock.patch.object(
                    COMPANION,
                    "decoded_audio_for_pipeline",
                    return_value=decoded,
                ),
            ):
                outputs, warnings, metadata = COMPANION.execute_audio_embedding(
                    Path(directory),
                    "audio_embedding",
                    parameters,
                    {"audio": str(source)},
                    Path(directory),
                    "test",
                    guard,
                )
                contents = json.loads(Path(outputs[0]["path"]).read_text("utf-8"))

        self.assertEqual(processor.call, (16_000, "pt"))
        self.assertIs(processor.audio, decoded["raw"])
        self.assertAlmostEqual(contents["embedding"][0], 0.6, places=6)
        self.assertAlmostEqual(contents["embedding"][1], 0.8, places=6)
        self.assertEqual(contents["dimensions"], 2)
        self.assertEqual(outputs[0]["mime_type"], "application/json")
        self.assertEqual(warnings, [])
        self.assertEqual(metadata["embedding_dimensions"], 2)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"),
        "NumPy and PyTorch are optional audio dependencies",
    )
    def test_numpy_audio_converts_torch_bfloat16_on_host(self):
        import numpy
        import torch

        result = COMPANION.numpy_audio(
            torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
        )

        self.assertEqual(result.dtype, numpy.dtype("float32"))
        numpy.testing.assert_allclose(result, [0.25, -0.5])

    def test_probe_recognizes_audio_classification_architecture(self):
        dependencies = {
            name: {"available": name in {"torch", "transformers", "numpy", "soundfile"}}
            for name in ("torch", "transformers", "numpy", "soundfile", "ffmpeg")
        }
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "wav2vec2",
                        "architectures": ["Wav2Vec2ForSequenceClassification"],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                COMPANION,
                "dependency_snapshot",
                return_value=dependencies,
            ):
                result = COMPANION.command_probe_model(
                    {
                        "model_path": str(model),
                        "task": "speaker_identification",
                    }
                )

        self.assertTrue(result["supported"])
        self.assertEqual(result["adapter"], "transformers_audio_classification")
        self.assertIn("audio_classification", result["probe"]["tasks"])

    def test_probe_does_not_claim_ctc_model_for_speech_translation(self):
        dependencies = {
            name: {"available": name in {"torch", "transformers", "numpy", "soundfile"}}
            for name in ("torch", "transformers", "numpy", "soundfile", "ffmpeg")
        }
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "wav2vec2",
                        "architectures": ["Wav2Vec2ForCTC"],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                COMPANION,
                "dependency_snapshot",
                return_value=dependencies,
            ):
                result = COMPANION.command_probe_model(
                    {
                        "model_path": str(model),
                        "task": "speech_translation",
                    }
                )

        self.assertFalse(result["supported"])
        self.assertIn("speech_to_text", result["probe"]["tasks"])
        self.assertNotIn("speech_translation", result["probe"]["tasks"])

    def test_probe_recognizes_declared_asr_translation_support(self):
        dependencies = {
            name: {"available": name in {"torch", "transformers", "numpy", "soundfile"}}
            for name in ("torch", "transformers", "numpy", "soundfile", "ffmpeg")
        }
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "whisper",
                        "architectures": ["WhisperForConditionalGeneration"],
                    }
                ),
                encoding="utf-8",
            )
            (model / "generation_config.json").write_text(
                json.dumps(
                    {
                        "task_to_id": {
                            "transcribe": 1,
                            "translate": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                COMPANION,
                "dependency_snapshot",
                return_value=dependencies,
            ):
                result = COMPANION.command_probe_model(
                    {
                        "model_path": str(model),
                        "task": "speech_translation",
                    }
                )

        self.assertTrue(result["supported"])
        self.assertIn("speech_translation", result["probe"]["tasks"])

    def test_probe_recognizes_gemma3n_audio_config_without_hub_metadata(self):
        dependencies = {
            name: {
                "available": name
                in {"torch", "transformers", "numpy", "soundfile"}
            }
            for name in (
                "torch",
                "transformers",
                "numpy",
                "soundfile",
                "ffmpeg",
                "librosa",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "gemma3n",
                        "architectures": ["Gemma3nForConditionalGeneration"],
                        "audio_config": {"model_type": "gemma3n_audio"},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                COMPANION,
                "dependency_snapshot",
                return_value=dependencies,
            ):
                result = COMPANION.command_probe_model(
                    {
                        "model_path": str(model),
                        "task": "audio_understanding",
                    }
                )

        self.assertTrue(result["supported"])
        self.assertEqual(result["adapter"], "transformers_audio_text")
        self.assertIn("audio_understanding", result["probe"]["tasks"])

    def test_command_execute_forwards_resident_runtime_to_every_adapter(self):
        cases = (
            ("image_generation", "diffusers", "execute_diffusers"),
            ("audio_generation", "diffusers_audio", "execute_diffusers_audio"),
            ("audio_generation", "transformers_audio", "execute_audio_generation"),
            ("text_to_speech", "transformers_tts", "execute_tts"),
            (
                "text_to_speech",
                COMPANION.QWEN3_TTS_ADAPTER,
                "execute_qwen3_tts_voice_design",
            ),
            ("audio_transcription", "transformers_asr", "execute_asr"),
            (
                "audio_classification",
                "transformers_audio_classification",
                "execute_audio_classification",
            ),
            ("audio_captioning", "transformers_audio_text", "execute_audio_text"),
            (
                "audio_embedding",
                "transformers_audio_embedding",
                "execute_audio_embedding",
            ),
        )
        runtime = object()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            for task, adapter, executor_name in cases:
                with (
                    self.subTest(task=task, adapter=adapter),
                    mock.patch.object(
                        COMPANION,
                        "execution_adapter",
                        return_value=adapter,
                    ),
                    mock.patch.object(
                        COMPANION,
                        executor_name,
                        return_value=([], [], {"runtime": adapter}),
                    ) as executor,
                ):
                    result = COMPANION.command_execute(
                        {
                            "model_path": str(model),
                            "task": task,
                            "output_dir": str(root / "outputs"),
                        },
                        runtime=runtime,
                    )

                executor.assert_called_once()
                self.assertIs(executor.call_args.kwargs["runtime"], runtime)
                self.assertEqual(result["task"], task)


class ResidentTransformersAudioCacheTests(unittest.TestCase):
    class RuntimeTorch:
        def inference_mode(self):
            return contextlib.nullcontext()

    @staticmethod
    def guard(task="audio_classification"):
        return COMPANION.ExplicitParameterGuard(
            {"explicit_parameters": []},
            task,
            "transformers_audio_classification",
            {},
        )

    def test_shared_limit_evicts_diffusers_before_transformers_load(self):
        runtime = COMPANION.CompanionRuntime(1)
        torch = self.RuntimeTorch()
        diffusers_pipeline = object()
        diffusers_entry = COMPANION.DiffusersPipelineEntry(
            diffusers_pipeline,
            torch,
            "cpu",
            "float32",
            {"offload_mode": "none", "offload_request": "none"},
            None,
            0.1,
        )
        runtime.pipeline_cache.put(("diffusers-fixture",), diffusers_entry)
        observed = []

        def load_pipeline(*_args, **_kwargs):
            observed.append(diffusers_entry.pipeline is None)
            return object(), torch, "cpu", "float32"

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(
                    COMPANION,
                    "torch_runtime",
                    return_value=(torch, "cpu", "float32"),
                ),
                mock.patch.object(
                    COMPANION,
                    "load_transformers_pipeline",
                    side_effect=load_pipeline,
                ),
            ):
                entry, _key, hit, resident = (
                    COMPANION.prepare_transformers_pipeline(
                        runtime,
                        Path(directory),
                        "transformers_audio_classification",
                        "audio-classification",
                        {},
                    )
                )

        self.assertEqual(observed, [True])
        self.assertFalse(hit)
        self.assertTrue(resident)
        self.assertEqual(len(runtime.pipeline_cache), 1)
        self.assertIsNotNone(entry.pipeline)
        runtime.close()
        self.assertIsNone(entry.pipeline)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is an optional audio dependency",
    )
    def test_repeated_classification_reuses_pipeline_and_reports_cache_hit(self):
        import numpy

        class ClassificationPipeline:
            feature_extractor = types.SimpleNamespace(sampling_rate=16_000)

            def __init__(self):
                self.calls = 0

            def __call__(self, _audio, top_k=None):
                self.calls += 1
                return [{"label": "speech", "score": 0.9}][:top_k]

        runtime = COMPANION.CompanionRuntime(1)
        torch = self.RuntimeTorch()
        pipeline = ClassificationPipeline()
        decoded = {
            "raw": numpy.zeros(16, dtype="float32"),
            "sampling_rate": 16_000,
        }
        parameters = {"output_format": "json", "top_k": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"model_type": "wav2vec2"}),
                encoding="utf-8",
            )
            source = root / "audio.wav"
            source.write_bytes(b"fixture")
            with (
                mock.patch.object(
                    COMPANION,
                    "torch_runtime",
                    return_value=(torch, "cpu", "float32"),
                ),
                mock.patch.object(
                    COMPANION,
                    "load_transformers_pipeline",
                    return_value=(pipeline, torch, "cpu", "float32"),
                ) as loader,
                mock.patch.object(
                    COMPANION,
                    "decode_local_audio",
                    return_value=decoded,
                ),
            ):
                first = COMPANION.execute_audio_classification(
                    model,
                    "audio_classification",
                    parameters,
                    {"audio": str(source)},
                    root,
                    "first",
                    self.guard(),
                    runtime=runtime,
                )[2]
                second = COMPANION.execute_audio_classification(
                    model,
                    "audio_classification",
                    parameters,
                    {"audio": str(source)},
                    root,
                    "second",
                    self.guard(),
                    runtime=runtime,
                )[2]

        loader.assert_called_once()
        self.assertEqual(pipeline.calls, 2)
        self.assertFalse(first["model_cache_hit"])
        self.assertGreaterEqual(first["model_load_seconds"], 0.0)
        self.assertTrue(second["model_cache_hit"])
        self.assertEqual(second["model_load_seconds"], 0.0)
        cached_entry = next(iter(runtime.pipeline_cache._entries.values()))
        runtime.close()
        self.assertIsNone(cached_entry.pipeline)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is an optional audio dependency",
    )
    def test_pipeline_failure_evicts_transformers_entry_and_reloads(self):
        import numpy

        class ClassificationPipeline:
            feature_extractor = types.SimpleNamespace(sampling_rate=16_000)

            def __init__(self):
                self.fail = True

            def __call__(self, _audio, top_k=None):
                if self.fail:
                    raise RuntimeError("broken inference")
                return [{"label": "speech", "score": 0.9}][:top_k]

        runtime = COMPANION.CompanionRuntime(1)
        torch = self.RuntimeTorch()
        pipeline = ClassificationPipeline()
        decoded = {
            "raw": numpy.zeros(16, dtype="float32"),
            "sampling_rate": 16_000,
        }
        parameters = {"output_format": "json", "top_k": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"model_type": "wav2vec2"}),
                encoding="utf-8",
            )
            source = root / "audio.wav"
            source.write_bytes(b"fixture")
            with (
                mock.patch.object(
                    COMPANION,
                    "torch_runtime",
                    return_value=(torch, "cpu", "float32"),
                ),
                mock.patch.object(
                    COMPANION,
                    "load_transformers_pipeline",
                    return_value=(pipeline, torch, "cpu", "float32"),
                ) as loader,
                mock.patch.object(
                    COMPANION,
                    "decode_local_audio",
                    return_value=decoded,
                ),
            ):
                with self.assertRaises(COMPANION.CompanionFailure) as failure:
                    COMPANION.execute_audio_classification(
                        model,
                        "audio_classification",
                        parameters,
                        {"audio": str(source)},
                        root,
                        "failure",
                        self.guard(),
                        runtime=runtime,
                    )
                self.assertEqual(len(runtime.pipeline_cache), 0)
                pipeline.fail = False
                metadata = COMPANION.execute_audio_classification(
                    model,
                    "audio_classification",
                    parameters,
                    {"audio": str(source)},
                    root,
                    "success",
                    self.guard(),
                    runtime=runtime,
                )[2]

        self.assertEqual(failure.exception.code, "execution_failed")
        self.assertEqual(loader.call_count, 2)
        self.assertFalse(metadata["model_cache_hit"])
        runtime.close()

    def test_embedding_processor_and_model_share_one_cached_entry(self):
        runtime = COMPANION.CompanionRuntime(1)
        torch = self.RuntimeTorch()
        processor = object()
        model = object()
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(
                    COMPANION,
                    "torch_runtime",
                    return_value=(torch, "cpu", "float32"),
                ),
                mock.patch.object(
                    COMPANION,
                    "load_transformers_audio_embedder",
                    return_value=(processor, model, torch, "cpu", "float32"),
                ) as loader,
            ):
                first, first_key, first_hit, _resident = (
                    COMPANION.prepare_transformers_audio_embedder(
                        runtime,
                        Path(directory),
                        {},
                    )
                )
                second, second_key, second_hit, _resident = (
                    COMPANION.prepare_transformers_audio_embedder(
                        runtime,
                        Path(directory),
                        {},
                    )
                )

        loader.assert_called_once()
        self.assertIs(first, second)
        self.assertEqual(first_key, second_key)
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertIs(first.processor, processor)
        self.assertIs(first.model, model)
        runtime.close()
        self.assertIsNone(first.processor)
        self.assertIsNone(first.model)


class Qwen3TTSVoiceDesignTests(unittest.TestCase):
    class RuntimeTorch:
        def __init__(self):
            self.seed = None
            self.cuda_seed = None
            self.cuda = types.SimpleNamespace(
                manual_seed_all=lambda value: setattr(self, "cuda_seed", value),
                current_device=lambda: 0,
            )

        def inference_mode(self):
            return contextlib.nullcontext()

        def manual_seed(self, value):
            self.seed = value

    @staticmethod
    def write_model(root, variant="voice_design"):
        (root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_tts",
                    "tts_model_type": variant,
                    "architectures": ["Qwen3TTSForConditionalGeneration"],
                }
            ),
            encoding="utf-8",
        )
        speech_tokenizer = root / "speech_tokenizer"
        speech_tokenizer.mkdir()
        (speech_tokenizer / "model.safetensors").write_bytes(b"tokenizer")
        (root / "model.safetensors").write_bytes(b"model")

    @staticmethod
    def dependencies(*, qwen_tts):
        return {
            name: {
                "available": name in {"torch", "numpy", "transformers"}
                or (name == "qwen_tts" and qwen_tts)
            }
            for name in (
                "torch",
                "numpy",
                "transformers",
                "qwen_tts",
                "diffusers",
                "PIL",
                "soundfile",
                "imageio",
                "imageio_ffmpeg",
                "av",
                "ffmpeg",
            )
        }

    def test_capabilities_advertise_tts_from_isolated_qwen_environment(self):
        dependencies = self.dependencies(qwen_tts=True)
        dependencies["transformers"]["available"] = False
        with mock.patch.object(
            COMPANION,
            "dependency_snapshot",
            return_value=dependencies,
        ):
            result = COMPANION.command_capabilities({})

        tts = next(
            item
            for item in result["capabilities"]
            if item["task"] == "text_to_speech"
        )
        self.assertTrue(tts["available"])
        self.assertEqual(tts["runtime"], "qwen3-tts-or-transformers")

    def test_dependency_status_rejects_installed_but_broken_qwen_package(self):
        with (
            mock.patch.object(
                COMPANION,
                "module_status",
                return_value={"available": True, "version": "0.1.1", "detail": None},
            ),
            mock.patch.object(
                COMPANION.importlib,
                "import_module",
                side_effect=RuntimeError("transformers mismatch"),
            ),
        ):
            status = COMPANION.importable_module_status(
                "qwen_tts",
                "qwen-tts",
                "Qwen3TTSModel",
            )

        self.assertFalse(status["available"])
        self.assertIn("transformers mismatch", status["detail"])

    def test_probe_recognizes_voice_design_and_requires_qwen_tts(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            self.write_model(model)
            with mock.patch.object(
                COMPANION,
                "dependency_snapshot",
                return_value=self.dependencies(qwen_tts=False),
            ):
                missing = COMPANION.command_probe_model(
                    {"model_path": str(model), "task": "text_to_speech"}
                )
            with mock.patch.object(
                COMPANION,
                "dependency_snapshot",
                return_value=self.dependencies(qwen_tts=True),
            ):
                ready = COMPANION.command_probe_model(
                    {"model_path": str(model), "task": "text_to_speech"}
                )

        self.assertEqual(missing["adapter"], COMPANION.QWEN3_TTS_ADAPTER)
        self.assertFalse(missing["supported"])
        self.assertTrue(any("qwen-tts" in reason for reason in missing["reasons"]))
        self.assertIn("werk backend install qwen-tts", missing["detail"])
        self.assertIn("WERK_QWEN_TTS_PYTHON", missing["detail"])
        self.assertEqual(missing["required_backend"], "qwen-tts")
        self.assertEqual(
            missing["install_command"],
            "werk backend install qwen-tts",
        )
        self.assertFalse(missing["backend_available"])
        self.assertFalse(missing["fallback_possible"])
        self.assertTrue(missing["architecture_adapter_supported"])
        self.assertEqual(missing["missing_dependencies"], ["qwen_tts"])
        self.assertEqual(
            missing["readiness"],
            {
                "status": "installable",
                "detail": missing["detail"],
                "adapter": COMPANION.QWEN3_TTS_ADAPTER,
                "required_backend": "qwen-tts",
                "install_command": "werk backend install qwen-tts",
                "fallback_backend": None,
                "missing_dependencies": ["qwen_tts"],
                "missing_dependency_groups": [],
            },
        )
        self.assertTrue(ready["supported"])
        self.assertEqual(ready["required_backend"], "qwen-tts")
        self.assertIsNone(ready["install_command"])
        self.assertTrue(ready["backend_available"])
        self.assertFalse(ready["fallback_possible"])
        self.assertTrue(ready["architecture_adapter_supported"])
        self.assertEqual(ready["missing_dependencies"], [])
        self.assertIn("text_to_speech", ready["probe"]["tasks"])
        self.assertIn("speech_tokenizer", ready["probe"]["components"])
        self.assertEqual(ready["probe"]["model_variant"], "voice_design")
        self.assertEqual(
            ready["readiness"],
            {
                "status": "available",
                "detail": ready["detail"],
                "adapter": COMPANION.QWEN3_TTS_ADAPTER,
                "required_backend": "qwen-tts",
                "install_command": None,
                "fallback_backend": None,
                "missing_dependencies": [],
                "missing_dependency_groups": [],
            },
        )

    def test_other_qwen_tts_variants_do_not_fall_through_to_transformers(self):
        for variant in ("custom_voice", "base"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                model = Path(directory)
                self.write_model(model, variant=variant)

                for task in (
                    "text_to_speech",
                    "audio_generation",
                    "audio_classification",
                ):
                    self.assertIsNone(COMPANION.execution_adapter(model, task))

                with mock.patch.object(
                    COMPANION,
                    "dependency_snapshot",
                    return_value=self.dependencies(qwen_tts=False),
                ):
                    result = COMPANION.command_probe_model(
                        {"model_path": str(model), "task": "text_to_speech"}
                    )

            self.assertFalse(result["supported"])
            self.assertEqual(result["required_backend"], "qwen-tts")
            self.assertFalse(result["backend_available"])
            self.assertFalse(result["fallback_possible"])
            self.assertFalse(result["architecture_adapter_supported"])
            self.assertIsNone(result["install_command"])
            self.assertEqual(result["readiness"]["status"], "not_implemented")
            self.assertIsNone(result["readiness"]["adapter"])
            self.assertEqual(
                result["readiness"]["required_backend"],
                "qwen-tts",
            )
            self.assertIsNone(result["readiness"]["install_command"])
            self.assertIsNone(result["readiness"]["fallback_backend"])
            self.assertEqual(
                result["readiness"]["missing_dependencies"],
                ["qwen_tts"],
            )
            self.assertEqual(
                result["readiness"]["missing_dependency_groups"],
                [],
            )

    def test_generic_transformers_probe_has_no_architecture_install_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "speecht5",
                        "architectures": ["SpeechT5ForTextToSpeech"],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                COMPANION,
                "dependency_snapshot",
                return_value=self.dependencies(qwen_tts=False),
            ):
                result = COMPANION.command_probe_model(
                    {"model_path": str(model), "task": "text_to_speech"}
                )

        self.assertTrue(result["supported"])
        self.assertEqual(result["adapter"], "transformers_tts")
        self.assertIsNone(result["required_backend"])
        self.assertIsNone(result["install_command"])
        self.assertIsNone(result["backend_available"])
        self.assertIsNone(result["fallback_possible"])
        self.assertIsNone(result["architecture_adapter_supported"])
        self.assertEqual(result["missing_dependencies"], [])
        self.assertEqual(result["readiness"]["status"], "available")
        self.assertEqual(result["readiness"]["adapter"], "transformers_tts")
        self.assertIsNone(result["readiness"]["required_backend"])
        self.assertIsNone(result["readiness"]["install_command"])
        self.assertIsNone(result["readiness"]["fallback_backend"])
        self.assertEqual(result["readiness"]["missing_dependencies"], [])
        self.assertEqual(result["readiness"]["missing_dependency_groups"], [])

    def test_generic_missing_dependencies_are_unavailable_not_installable(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "speecht5",
                        "architectures": ["SpeechT5ForTextToSpeech"],
                    }
                ),
                encoding="utf-8",
            )
            dependencies = self.dependencies(qwen_tts=False)
            dependencies["transformers"]["available"] = False
            with mock.patch.object(
                COMPANION,
                "dependency_snapshot",
                return_value=dependencies,
            ):
                result = COMPANION.command_probe_model(
                    {"model_path": str(model), "task": "text_to_speech"}
                )

        self.assertFalse(result["supported"])
        self.assertEqual(result["adapter"], "transformers_tts")
        self.assertEqual(result["readiness"]["status"], "unavailable")
        self.assertEqual(
            result["readiness"]["missing_dependencies"],
            ["transformers"],
        )
        self.assertIsNone(result["readiness"]["required_backend"])
        self.assertIsNone(result["readiness"]["install_command"])
        self.assertIsNone(result["readiness"]["fallback_backend"])
        self.assertEqual(result["readiness"]["missing_dependency_groups"], [])

    def test_generic_alternative_dependencies_preserve_any_of_semantics(self):
        dependencies = self.dependencies(qwen_tts=False)
        dependencies["transformers"]["available"] = False

        mandatory, groups = COMPANION.generic_dependency_readiness(
            "audio_generation",
            dependencies,
        )

        self.assertEqual(mandatory, [])
        self.assertEqual(
            groups,
            [
                {
                    "purpose": "audio_generation_framework",
                    "any_of": [
                        {"all_of": ["diffusers"]},
                        {"all_of": ["transformers"]},
                    ],
                }
            ],
        )

        mandatory, groups = COMPANION.generic_dependency_readiness(
            "speech_to_text",
            dependencies,
        )
        self.assertEqual(mandatory, ["transformers"])
        self.assertEqual(
            groups,
            [
                {
                    "purpose": "audio_decoder",
                    "any_of": [
                        {"all_of": ["soundfile"]},
                        {"all_of": ["ffmpeg"]},
                    ],
                }
            ],
        )

    def test_audio_model_readiness_requires_its_selected_framework(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "musicgen",
                        "architectures": ["MusicgenForConditionalGeneration"],
                    }
                ),
                encoding="utf-8",
            )
            dependencies = self.dependencies(qwen_tts=False)
            dependencies["diffusers"]["available"] = True
            dependencies["transformers"]["available"] = False
            with mock.patch.object(
                COMPANION,
                "dependency_snapshot",
                return_value=dependencies,
            ):
                result = COMPANION.command_probe_model(
                    {"model_path": str(model), "task": "music_generation"}
                )

        self.assertEqual(result["adapter"], "transformers_audio")
        self.assertFalse(result["supported"])
        self.assertEqual(result["readiness"]["status"], "unavailable")
        self.assertEqual(
            result["readiness"]["missing_dependencies"],
            ["transformers"],
        )
        self.assertEqual(result["readiness"]["missing_dependency_groups"], [])

    def test_video_decoder_represents_compound_alternative_routes(self):
        dependencies = self.dependencies(qwen_tts=False)
        dependencies["diffusers"]["available"] = True
        dependencies["PIL"]["available"] = True

        mandatory, groups = COMPANION.generic_dependency_readiness(
            "video_to_video",
            dependencies,
        )

        self.assertEqual(mandatory, [])
        self.assertEqual(
            groups,
            [
                {
                    "purpose": "video_encoder",
                    "any_of": [
                        {"all_of": ["av"]},
                        {"all_of": ["ffmpeg"]},
                        {"all_of": ["imageio_ffmpeg"]},
                    ],
                },
                {
                    "purpose": "video_decoder",
                    "any_of": [
                        {"all_of": ["av"]},
                        {"all_of": ["imageio", "imageio_ffmpeg"]},
                    ],
                },
            ],
        )

        dependencies["imageio"]["available"] = True
        dependencies["imageio_ffmpeg"]["available"] = True
        mandatory, groups = COMPANION.generic_dependency_readiness(
            "video_to_video",
            dependencies,
        )
        self.assertEqual(mandatory, [])
        self.assertEqual(groups, [])

    def test_declared_unsupported_tasks_are_not_implemented(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps({"model_type": "fixture"}),
                encoding="utf-8",
            )
            dependencies = self.dependencies(qwen_tts=False)
            for task in sorted(COMPANION.DECLARED_UNSUPPORTED_TASKS):
                with self.subTest(task=task), mock.patch.object(
                    COMPANION,
                    "dependency_snapshot",
                    return_value=dependencies,
                ):
                    result = COMPANION.command_probe_model(
                        {"model_path": str(model), "task": task}
                    )

                self.assertFalse(result["supported"])
                self.assertEqual(
                    result["readiness"]["status"],
                    "not_implemented",
                )
                self.assertIsNone(result["readiness"]["adapter"])
                self.assertIsNone(result["readiness"]["required_backend"])
                self.assertIsNone(result["readiness"]["install_command"])
                self.assertIsNone(result["readiness"]["fallback_backend"])
                self.assertEqual(
                    result["readiness"]["missing_dependencies"],
                    [],
                )
                self.assertEqual(
                    result["readiness"]["missing_dependency_groups"],
                    [],
                )

    def test_estimate_remains_available_but_reports_missing_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            self.write_model(model)
            with mock.patch.object(
                COMPANION,
                "dependency_snapshot",
                return_value=self.dependencies(qwen_tts=False),
            ):
                estimate = COMPANION.command_estimate(
                    {
                        "model_path": str(model),
                        "task": "text_to_speech",
                        "effective_parameters": {
                            "tts.language": "de",
                            "tts.speaking_style": "warm and precise",
                            "tts.output_format": "wav",
                        },
                        "explicit_parameters": [
                            "tts.language",
                            "tts.speaking_style",
                            "tts.output_format",
                        ],
                    }
                )

        self.assertTrue(any("bundled speech tokenizer" in item for item in estimate["assumptions"]))
        self.assertTrue(any("qwen-tts" in item for item in estimate["warnings"]))
        self.assertIn("tts.language", estimate["parameter_support"]["explicit_parameters"])

    def test_loader_uses_only_local_model_path_and_official_api(self):
        calls = []

        class FakeModel:
            def generate_voice_design(self, **_kwargs):
                return [], 24_000

        class FakeModelClass:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.append((path, kwargs))
                return FakeModel()

        fake_qwen = types.SimpleNamespace(Qwen3TTSModel=FakeModelClass)
        torch = self.RuntimeTorch()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            self.write_model(model)
            with mock.patch.object(
                COMPANION,
                "require_module",
                return_value=fake_qwen,
            ):
                loaded, loaded_torch, device, dtype = (
                    COMPANION.load_qwen3_tts_voice_design(
                        model,
                        {},
                        runtime_values=(torch, "cuda", "bfloat16"),
                    )
                )

        self.assertIsInstance(loaded, FakeModel)
        self.assertIs(loaded_torch, torch)
        self.assertEqual(device, "cuda")
        self.assertEqual(dtype, "bfloat16")
        self.assertEqual(calls, [(str(model), {"device_map": "cuda:0", "dtype": "bfloat16"})])

    def test_resident_runtime_reuses_qwen_model_and_reports_cache_hit(self):
        model = object()
        torch = self.RuntimeTorch()
        runtime = COMPANION.CompanionRuntime(1)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            self.write_model(model_path)
            with (
                mock.patch.object(
                    COMPANION,
                    "torch_runtime",
                    return_value=(torch, "cpu", "bfloat16"),
                ),
                mock.patch.object(
                    COMPANION,
                    "load_qwen3_tts_voice_design",
                    return_value=(model, torch, "cpu", "bfloat16"),
                ) as loader,
            ):
                first, first_key, first_hit, first_resident = (
                    COMPANION.prepare_qwen3_tts_voice_design(
                        runtime,
                        model_path,
                        {},
                    )
                )
                second, second_key, second_hit, second_resident = (
                    COMPANION.prepare_qwen3_tts_voice_design(
                        runtime,
                        model_path,
                        {},
                    )
                )

        loader.assert_called_once()
        self.assertIs(first, second)
        self.assertEqual(first_key, second_key)
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertTrue(first_resident)
        self.assertTrue(second_resident)
        self.assertIs(second.model, model)
        runtime.close()
        self.assertIsNone(second.model)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is an optional audio dependency",
    )
    def test_execute_maps_german_style_seed_and_writes_wav(self):
        import numpy

        class FakeModel:
            def __init__(self):
                self.calls = []

            def generate_voice_design(self, **kwargs):
                self.calls.append(kwargs)
                return [numpy.linspace(-0.2, 0.2, 24, dtype="float32")], 24_000

        model = FakeModel()
        torch = self.RuntimeTorch()
        entry = COMPANION.TransformersAudioEntry(
            torch,
            "cpu",
            "bfloat16",
            0.25,
            adapter=COMPANION.QWEN3_TTS_ADAPTER,
            pipeline_task="voice-design",
            model=model,
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            COMPANION,
            "prepare_qwen3_tts_voice_design",
            return_value=(entry, ("qwen",), False, False),
        ):
            outputs, warnings, metadata = (
                COMPANION.execute_qwen3_tts_voice_design(
                    Path(directory),
                    "text_to_speech",
                    {
                        "text": "Werk elf zwölf ist bereit.",
                        "language": "de-DE",
                        "speaking_style": "Warm, ruhig und präzise.",
                        "seed": 1112,
                        "output_format": "wav",
                    },
                    Path(directory),
                    "hq",
                )
            )
            output_path = Path(outputs[0]["path"])
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 44)

        self.assertEqual(
            model.calls,
            [
                {
                    "text": "Werk elf zwölf ist bereit.",
                    "language": "German",
                    "instruct": "Warm, ruhig und präzise.",
                    "non_streaming_mode": True,
                }
            ],
        )
        self.assertEqual(warnings, [])
        self.assertEqual(torch.seed, 1112)
        self.assertEqual(torch.cuda_seed, 1112)
        self.assertEqual(metadata["runtime"], "qwen-tts")
        self.assertEqual(metadata["language"], "German")
        self.assertIn("tts.speaking_style->instruct", metadata["translated_parameters"])
        self.assertIn("tts.seed->torch.manual_seed", metadata["translated_parameters"])
        self.assertEqual(outputs[0]["metadata"]["sample_rate"], 24_000)

    def test_supported_parameters_are_voice_design_specific(self):
        supported = COMPANION.supported_explicit_parameters(
            "text_to_speech",
            COMPANION.QWEN3_TTS_ADAPTER,
        )

        self.assertIn("tts.language", supported)
        self.assertIn("tts.speaking_style", supported)
        self.assertIn("tts.seed", supported)
        self.assertIn("tts.output_format", supported)
        self.assertNotIn("tts.voice", supported)

    def test_command_execute_dispatches_qwen_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            self.write_model(model)
            with mock.patch.object(
                COMPANION,
                "execute_qwen3_tts_voice_design",
                return_value=([], [], {"runtime": "qwen-tts"}),
            ) as executor:
                result = COMPANION.command_execute(
                    {
                        "model_path": str(model),
                        "task": "text_to_speech",
                        "prompt": "Werk elf zwölf ist bereit.",
                        "effective_parameters": {
                            "tts.speaking_style": "warm and precise",
                        },
                        "explicit_parameters": ["tts.speaking_style"],
                        "output_dir": str(root / "outputs"),
                    }
                )
                persisted = Path(result["metadata"]["metadata_path"]).read_text(
                    encoding="utf-8"
                )

        executor.assert_called_once()
        self.assertEqual(result["task"], "text_to_speech")
        self.assertEqual(result["metadata"]["backend"]["runtime"], "qwen-tts")
        effective = result["metadata"]["effective_parameters"]
        self.assertNotIn("text", effective)
        self.assertNotIn("prompt", effective)
        self.assertNotIn("speaking_style", effective)
        self.assertNotIn("tts.speaking_style", effective)
        self.assertNotIn("Werk elf zwölf ist bereit.", persisted)
        self.assertNotIn("warm and precise", persisted)


if __name__ == "__main__":
    unittest.main()
