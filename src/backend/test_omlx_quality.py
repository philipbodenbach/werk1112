"""Opt-in end-to-end numerical checks using a tiny installed DeepSeek V4.

Run with WERK_TEST_MLX_QUALITY=1 in the oMLX Python environment. These tests
construct random, small models; they never load the user's model weights or
contact an inference server. They check implementation fidelity, not language
quality of a trained checkpoint.
"""

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import weakref


def local_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.environ.get("WERK_TEST_MLX_QUALITY") == "1", "real MLX quality checks are opt-in")
class ModelDifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
        apply_deepseek_v4_patch()
        from mlx_lm.models.deepseek_v4 import Model, ModelArgs
        cls.mx, cls.nn, cls.tree_flatten = mx, nn, staticmethod(tree_flatten)
        cls.Model, cls.ModelArgs = Model, ModelArgs
        cls.experts = local_module("omlx_experts")
        cls.persistence = local_module("omlx_persistence")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mx.random.seed(113)

    def models(self, dtype):
        mx, nn = self.mx, self.nn
        args = self.ModelArgs(
            vocab_size=128, hidden_size=64, intermediate_size=64,
            moe_intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
            n_routed_experts=8, num_experts_per_tok=6, num_hash_layers=1,
            q_lora_rank=32, qk_rope_head_dim=16, head_dim=32,
            hc_mult=4, hc_sinkhorn_iters=5, sliding_window=16,
            o_groups=2, o_lora_rank=32, index_n_heads=2, index_head_dim=32,
            index_topk=8, compress_ratios=[0, 4, 0],
            use_native_ratio128_attention=False,
        )
        native = self.Model(args)
        for layer in native.layers:
            gate = layer.ffn.gate
            gate.weight = (mx.random.normal(gate.weight.shape) * 0.08).astype(dtype)
            if gate.hash:
                # Exercise real input-ID hash routing with distinct routes.
                gate.tid2eid = (mx.arange(args.vocab_size)[:, None] + mx.arange(6)[None]) % 8
        native.load_weights([(key, value.astype(dtype) if mx.issubdtype(value.dtype, mx.floating)
                              and native.cast_predicate(key) else value)
                             for key, value in self.tree_flatten(native.parameters())])

        def quantized(path, module):
            if ".switch_mlp." not in path or not hasattr(module, "to_quantized"):
                return False
            return {"bits": 4, "group_size": 64} if path.endswith("gate_proj") else True

        nn.quantize(native, group_size=32, bits=2, class_predicate=quantized)
        tensors = dict(self.tree_flatten(native.parameters()))
        # Real checkpoint quantization metadata is BF16; the installed model
        # sanitizer converts affine expert metadata to FP16 before inference.
        for key, value in list(tensors.items()):
            if ".switch_mlp." in key and key.endswith((".scales", ".biases")):
                tensors[key] = value.astype(mx.bfloat16)
        cfg = dataclasses.asdict(args)
        cfg["quantization"] = {"bits": 2, "group_size": 32, "mode": "affine"}
        for layer in range(3):
            cfg["quantization"][f"model.layers.{layer}.ffn.switch_mlp.gate_proj"] = {
                "bits": 4, "group_size": 64, "mode": "affine"}
        model_path = self.root / "model"
        model_path.mkdir()
        (model_path / "config.json").write_text(json.dumps(cfg))
        mx.save_safetensors(str(model_path / "model.safetensors"), tensors)
        weights = native.sanitize(tensors)
        native.load_weights(list(weights.items()), strict=True)
        native.eval()
        mx.eval(native.parameters())

        plan = self.experts.inspect_model(model_path, 1 << 20)
        # Force repeated eviction across layers and across selected experts.
        checkpoint = self.experts.Checkpoint(model_path, plan["largest_expert_bytes"] * 2)
        manager = self.experts.ExpertManager(checkpoint)
        self.addCleanup(manager.deactivate)
        streamed = self.Model(args)
        for index, layer in enumerate(streamed.layers):
            layer.ffn.switch_mlp = self.experts._streamed_module(manager, index, layer.ffn.switch_mlp.activation)
        streamed.load_weights([(key, value) for key, value in weights.items() if ".switch_mlp." not in key], strict=True)
        streamed.eval()
        mx.eval(streamed.parameters())
        manager._model_ref = weakref.ref(streamed)
        return native, streamed, manager, model_path

    def close_logits(self, expected, actual, *, tolerance=0.035):
        mx = self.mx
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.all(mx.isfinite(actual))))
        expected, actual = expected.astype(mx.float32), actual.astype(mx.float32)
        maximum = float(mx.max(mx.abs(expected - actual)))
        self.assertLessEqual(maximum, tolerance, f"logit max difference {maximum}")
        cosine = (expected * actual).sum(-1) / mx.sqrt((expected * expected).sum(-1) * (actual * actual).sum(-1))
        self.assertGreater(float(mx.min(cosine)), 0.999)

    def compare_model(self, dtype):
        mx = self.mx
        native, streamed, manager, _ = self.models(dtype)
        native_cache, streamed_cache = native.make_cache(), streamed.make_cache()
        self.assertEqual(type(native_cache[1].caches[1]).__name__, "PoolingCache")
        self.assertTrue(native.layers[0].ffn.gate.hash)
        self.assertFalse(native.layers[1].ffn.gate.hash)
        # Prefill, decode, and a suffix spanning pooling and rotating boundaries.
        for tokens in ([3, 14, 9, 26, 5, 10, 23, 42, 8, 11, 12, 13, 7],
                       [41], [18], [27], [36], [45], [54], [63], [72],
                       [4, 19, 22, 31, 44, 59, 66]):
            inputs = mx.array([tokens], dtype=mx.int32)
            self.close_logits(native(inputs, cache=native_cache), streamed(inputs, cache=streamed_cache))
        self.assertGreater(manager.misses, 24)
        self.assertLessEqual(manager.status()["resident_cache_bytes"], manager.checkpoint.cache_bytes)

    def test_hash_routing_pooling_and_multistep_decode_bfloat16(self):
        self.compare_model(self.mx.bfloat16)

    def test_hash_routing_pooling_and_multistep_decode_float16(self):
        self.compare_model(self.mx.float16)

    def test_exact_prefix_store_preserves_cold_decode_and_restored_logits(self):
        from omlx.cache.paged_cache import PagedCacheManager
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
        from omlx.cache.prefix_cache import BlockAwarePrefixCache
        from omlx.scheduler import Scheduler

        mx = self.mx
        _, model, _, model_path = self.models(mx.bfloat16)
        persistence = self.persistence.PrefixPersistence(model_path, self.root / "prefixes", "a" * 64)
        ssd = PagedSSDCacheManager(cache_dir=persistence.native_directory,
            max_size_bytes=16 * 1024 * 1024, expected_model_name="tiny-quality",
            expected_num_layers=3, expected_block_size=2048)
        self.addCleanup(ssd.close)
        paged = PagedCacheManager(block_size=2048, max_blocks=16, model_name="tiny-quality", initial_blocks=4)
        paged.set_paged_ssd_cache_manager(ssd)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.block_aware_cache = BlockAwarePrefixCache(model, paged, ssd)
        scheduler.model = model
        scheduler.model_name = "tiny-quality"
        scheduler._stream = mx.default_stream(mx.default_device())
        scheduler._bypass_hot_cache_under_pressure = lambda: False
        scheduler._release_paged_cache_for_request = lambda request_id: None
        persistence._scheduler = weakref.ref(scheduler)
        persistence.max_prefix_tokens = 2048
        prefix = [3, 14, 9, 26, 5, 10, 23, 42, 8, 11, 12, 13, 7, 41, 18, 27, 36, 45, 54]
        suffix = [4, 19, 22, 31, 44]
        reference, stored = model.make_cache(), model.make_cache()
        inputs = mx.array([prefix], dtype=mx.int32)
        self.close_logits(model(inputs, cache=reference), model(inputs, cache=stored), tolerance=0.0)
        request = SimpleNamespace(request_id="cold", prompt_token_ids=prefix + suffix,
                                  cached_tokens=0, block_table=None)
        self.assertTrue(persistence.store(scheduler, request, prefix, stored))
        self.assertEqual(request.cached_tokens, 0)
        restored_request = SimpleNamespace(request_id="branch", prompt_token_ids=prefix + suffix,
                                           cached_tokens=0, block_table=None)
        self.assertTrue(persistence.restore(scheduler, restored_request))
        self.assertEqual(restored_request.cached_tokens, len(prefix))
        for token in suffix:
            inputs = mx.array([[token]], dtype=mx.int32)
            expected = model(inputs, cache=reference)
            self.close_logits(expected, model(inputs, cache=stored), tolerance=0.0)
            self.close_logits(expected, model(inputs, cache=restored_request.prompt_cache), tolerance=0.0)


if __name__ == "__main__":
    unittest.main()
