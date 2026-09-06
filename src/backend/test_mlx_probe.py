"""Small installed-runtime simulations; no MLX installation or model weights."""

import contextlib
import importlib.util
import io
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("werk_mlx_probe", Path(__file__).with_name("mlx_probe.py"))
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)


def fixture_load(path):
    return load_model(path)


def fixture_overridden_load(path):
    return load_model(path, get_model_classes=custom_resolver)


def fake_runtime(architectures=("llama",), modes=("affine",)):
    class Model:
        def __init__(self, *args):
            raise AssertionError("probe must never construct a model")

    class ModelArgs:
        @classmethod
        def from_dict(cls, config):
            if config.get("hidden_size") == "broken":
                raise ValueError("hidden_size must be an integer")
            return config

    def resolve(config):
        architecture = {"mistral": "llama"}.get(config["model_type"], config["model_type"])
        if architecture not in architectures:
            raise ValueError(f"Model type {architecture} not supported.")
        return Model, ModelArgs

    def load_model(path, get_model_classes=resolve):
        # An executable miniature of the installed loader dispatch, deliberately
        # using a path whose read fails if preflight ever calls this function.
        config = json.loads(path.read_text())
        if model_file := config.get("model_file"):
            spec = importlib.util.spec_from_file_location("custom_model", path / model_file)
            arch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(arch)

        def _quantize(quantization):
            def class_predicate(p, m):
                if p in config["quantization"]:
                    return config["quantization"][p]
                return False

            nn.quantize(None, group_size=quantization["group_size"],
                        bits=quantization["bits"], mode=quantization.get("mode", "affine"),
                        class_predicate=class_predicate)

        if quantization := config.get("quantization"):
            _quantize(quantization)

    load = types.FunctionType(fixture_load.__code__, {"load_model": load_model})

    def quantize(array, group_size, bits, mode="affine"):
        if mode not in modes:
            raise ValueError(f"unsupported quantization mode {mode}")
        if mode in ("mxfp4", "mxfp8") and (group_size, bits) != (32, 4 if mode == "mxfp4" else 8):
            raise ValueError("invalid quantization layout")
        return (array,)

    def zeros(shape):
        if shape[0] != 1 or shape[1] > 1024:
            raise AssertionError("unbounded allocation")
        return shape

    def nn_quantize(model, group_size, bits, mode="affine", class_predicate=None):
        def _maybe_quantize(path, module):
            parameters = class_predicate(path, module)
            if isinstance(parameters, bool):
                return module.to_quantized(group_size=group_size, bits=bits, mode=mode)
            elif isinstance(parameters, dict):
                return module.to_quantized(**parameters)

        return _maybe_quantize("layer", model)

    return {
        "mlx_lm": types.SimpleNamespace(__version__="simulated-1"),
        "mlx_lm.generate": types.SimpleNamespace(load=load),
        "mlx_lm.utils": types.SimpleNamespace(load=load, load_model=load_model),
        "mlx.core": types.SimpleNamespace(
            __version__="simulated-1", quantize=quantize, zeros=zeros,
            eval=lambda _: None, metal=types.SimpleNamespace(is_available=lambda: True),
        ),
        "mlx.nn": types.SimpleNamespace(quantize=nn_quantize),
    }


class MlxMetadataProbeTests(unittest.TestCase):
    def run_probe(self, config, runtime=None, **payload):
        with patch.dict(sys.modules, runtime or fake_runtime()):
            return probe_module.probe({"config": config, **payload}, "mlx_lm.generate")

    def test_installed_runtime_resolves_architecture_alias_without_name_allowlist(self):
        self.assertIn("mlx-lm", self.run_probe({"model_type": "mistral"}))
        self.assertIn("mlx-lm", self.run_probe({"model_type": "new_arch"}, fake_runtime(("new_arch",))))

    def test_deepseek_mixed_mxfp_fixture_rejects_importable_unsupported_architecture(self):
        config = {
            "model_type": "deepseek_v4", "hidden_size": 4096,
            "quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4",
                             "model.layers.0.mlp": {"group_size": 32, "bits": 8, "mode": "mxfp8"}},
        }
        with self.assertRaisesRegex(ValueError, "architecture 'deepseek_v4'.*unsupported.*mlx-lm"):
            self.run_probe(config)
        # This is a simulated compatible installed implementation, not a claim
        # that any released runtime supports the real DeepSeek model.
        self.assertIn("mlx-lm", self.run_probe(config, fake_runtime(("deepseek_v4",), ("mxfp4", "mxfp8"))))
        with self.assertRaisesRegex(ValueError, "incompatible runtime quantization mxfp8"):
            self.run_probe(config, fake_runtime(("deepseek_v4",), ("mxfp4",)))
        runtime = fake_runtime(("deepseek_v4",), ("mxfp4", "mxfp8"))
        runtime["mlx.nn"].quantize = lambda model, group_size, bits, mode="affine": None
        with self.assertRaisesRegex(ValueError, "per-layer parameter dictionaries"):
            self.run_probe(config, runtime)

    def test_probe_results_are_not_reused_across_models_or_runtime_versions(self):
        self.run_probe({"model_type": "llama"})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.run_probe({"model_type": "deepseek_v4"})
        self.run_probe({"model_type": "deepseek_v4"}, fake_runtime(("deepseek_v4",)))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.run_probe({"model_type": "deepseek_v4"})

    def test_damaged_metadata_is_distinct_from_missing_architecture(self):
        for config in ({}, {"model_type": None}, [], None):
            with self.assertRaisesRegex(ValueError, "damaged model metadata"):
                self.run_probe(config)
        with self.assertRaisesRegex(ValueError, "damaged or incompatible model metadata"):
            self.run_probe({"model_type": "llama", "hidden_size": "broken"})

    def test_model_repository_code_is_not_executed(self):
        with self.assertRaisesRegex(ValueError, "without executing model repository code"):
            self.run_probe({"model_type": "llama", "model_file": "evil.py"})
        # auto_map is not passed to transformers or any repository-code loader.
        self.run_probe({"model_type": "llama", "auto_map": {"AutoConfig": "evil.Config"}})

    def test_selected_custom_module_must_expose_its_actual_loader(self):
        runtime = fake_runtime(("selected_only",))
        runtime["custom_generator"] = runtime["mlx_lm.generate"]
        runtime["mlx_lm.generate"] = fake_runtime()["mlx_lm.generate"]
        with patch.dict(sys.modules, runtime):
            self.assertIn("custom_generator", probe_module.probe({"config": {"model_type": "selected_only"}}, "custom_generator"))
            with self.assertRaisesRegex(ValueError, "cannot verify architecture resolution"):
                probe_module.probe({"config": {"model_type": "selected_only"}}, "mlx_lm.generate")
        runtime["custom_generator"] = types.SimpleNamespace()
        with patch.dict(sys.modules, runtime), self.assertRaisesRegex(ValueError, "cannot verify architecture resolution"):
            probe_module.probe({"config": {"model_type": "llama"}}, "custom_generator")

    def test_custom_loader_wrapper_cannot_certify_its_overridden_resolver(self):
        runtime = fake_runtime()
        loader = runtime["mlx_lm.utils"].load_model
        custom_resolver = lambda config: (_ for _ in ()).throw(ValueError("actual resolver rejects"))
        wrapper = types.FunctionType(fixture_overridden_load.__code__,
                                     {"load_model": loader, "custom_resolver": custom_resolver})
        runtime["mlx_lm.generate"].load = wrapper
        with self.assertRaisesRegex(ValueError, "custom load wrappers are unverified"):
            self.run_probe({"model_type": "llama"}, runtime)
        runtime["mlx_lm.utils"].load = wrapper
        with self.assertRaisesRegex(ValueError, "overrides or hides its model-class resolver"):
            self.run_probe({"model_type": "llama"}, runtime)

    def test_comments_do_not_establish_executable_loader_capabilities(self):
        def comments_only(path):
            # model_file quantization class_predicate dict mode= spec_from_file_location exec_module
            return None

        with self.assertRaisesRegex(ValueError, "cannot verify loader forwarding"):
            probe_module.verify_quantization_loader(comments_only, True, ["mxfp4"])
        with self.assertRaisesRegex(ValueError, "per-layer parameter dictionaries"):
            probe_module.verify_nn_dictionary_dispatch(comments_only)
        runtime = fake_runtime(("gemma4",))
        resolver = inspect.signature(runtime["mlx_lm.utils"].load_model).parameters["get_model_classes"].default

        def comments_only_model_file(path, get_model_classes=resolver):
            # model_file spec_from_file_location exec_module
            return None

        runtime["mlx_lm.utils"].load_model = comments_only_model_file
        runtime["mlx_lm.utils"].load.__globals__["load_model"] = comments_only_model_file
        with self.assertRaisesRegex(ValueError, "compatibility model_file is not supported"):
            self.run_probe({"model_type": "gemma4", "model_file": "werk_gemma4_unified_compat.py"},
                           runtime, werk_gemma4_compat=True)

    def test_quantization_dispatch_dictionaries_reach_the_layer(self):
        runtime = fake_runtime()
        expected = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        layer = types.SimpleNamespace(to_quantized=lambda **kwargs: kwargs)
        actual = runtime["mlx.nn"].quantize(layer, 64, 4, class_predicate=lambda path, module: expected)
        self.assertEqual(actual, expected)

    def test_metal_device_is_checked_in_selected_runtime(self):
        runtime = fake_runtime()
        runtime["mlx.core"].metal.is_available = lambda: False
        with self.assertRaisesRegex(ValueError, "Metal device is unavailable"):
            self.run_probe({"model_type": "llama"}, runtime)

    def test_legacy_quantization_and_invalid_layouts_are_not_claimed_verified(self):
        with self.assertRaisesRegex(ValueError, "cannot verify legacy"):
            self.run_probe({"model_type": "llama", "quantization_config": {"quant_method": "gptq"}})
        with self.assertRaisesRegex(ValueError, "damaged model metadata"):
            self.run_probe({"model_type": "llama", "quantization": {"bits": 4}})
        with self.assertRaisesRegex(ValueError, "bounded probe limit"):
            self.run_probe({"model_type": "llama", "quantization": {"bits": 4, "group_size": 8192}})

    def test_gemma4_werk_config_checks_installed_gemma_class(self):
        config = {"model_type": "gemma4", "model_file": "werk_gemma4_unified_compat.py"}
        self.run_probe(config, fake_runtime(("gemma4",)), werk_gemma4_compat=True)
        with self.assertRaisesRegex(ValueError, "architecture 'gemma4'.*unsupported"):
            self.run_probe(config, werk_gemma4_compat=True)

    def test_console_launcher_is_inspected_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mlx_lm.generate"
            path.write_text("#!/python\nimport sys\nfrom selected_generator import main\nif __name__ == '__main__':\n    sys.exit(main())\n")
            self.assertEqual(probe_module.launcher_module(path), "selected_generator")
            path.write_text("#!/python\nimport re\nimport sys\nfrom selected_generator import main\nif __name__ == '__main__':\n    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n    sys.exit(main())\n")
            self.assertEqual(probe_module.launcher_module(path), "selected_generator")
            path.write_text(path.read_text() + "raise AssertionError('must not execute')\n")
            with self.assertRaisesRegex(ValueError, "cannot verify custom MLX launcher"):
                probe_module.launcher_module(path)

    def test_launcher_imports_use_its_directory_like_real_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mlx_lm.generate"
            path.write_text("#!/python\nimport sys\nfrom adjacent_generator import main\nif __name__ == '__main__':\n    sys.exit(main())\n")
            (Path(directory) / "adjacent_generator.py").write_text("from mlx_lm.generate import load\n")
            original_path = sys.path[:]
            try:
                with patch.dict(sys.modules, fake_runtime()):
                    result = probe_module.probe({"config": {"model_type": "llama"}}, str(path), launcher=True)
                self.assertIn("adjacent_generator", result)
            finally:
                sys.path[:] = original_path
                sys.modules.pop("adjacent_generator", None)

    def test_json_protocol_returns_explicit_error(self):
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["probe", "module", "mlx_lm.generate"]), patch.object(sys, "stdin", io.StringIO('{"config": {}}')), contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit):
                probe_module.main()
        self.assertFalse(json.loads(stdout.getvalue())["ok"])


if __name__ == "__main__":
    unittest.main()
