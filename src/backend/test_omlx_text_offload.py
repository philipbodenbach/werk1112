"""Installed native model vs offloaded model, including recurrent follow-up tokens."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omlx_text_offload as adapter


class NamespaceTests(unittest.TestCase):
    def test_quantization_modules_and_weights_share_native_namespace(self):
        self.assertEqual(adapter._canonical_name("lm_head"), "language_model.lm_head")
        self.assertEqual(adapter._canonical_name("lm_head.weight"), "language_model.lm_head.weight")
        self.assertEqual(adapter._canonical_name("model.norm"), "language_model.model.norm")
        with self.assertRaises(ValueError): adapter._canonical_name("unverified.module")


class NgramAutoBudgetTests(unittest.TestCase):
    def test_auto_selects_native_only_when_resident_weights_fit(self):
        from test_omlx_offload import write_fixture
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_fixture(path)
            cp = adapter.TextCheckpoint(path, 0)
            cp.select_resident_experts(8 * adapter.GiB, 8 * adapter.GiB)
            self.assertFalse(cp.experts_enabled)
            self.assertFalse(cp.ngrams_enabled)
            self.assertEqual(cp.cache_bytes, 0)
            self.assertEqual(cp.base_bytes, cp.dense_bytes + cp.total_expert_bytes + cp.ngram_storage_bytes)
            manager = adapter.TextExpertManager(cp)
            manager._resize_cache(8 * adapter.GiB)
            self.assertEqual(manager.effective_cache_bytes, 0)
            cp = adapter.TextCheckpoint(path, 0)
            cp.select_resident_experts(8 * adapter.GiB, 2 * adapter.GiB)
            self.assertTrue(cp.experts_enabled)
            cp = adapter.TextCheckpoint(path, 1024**2)
            cp.select_resident_experts(8 * adapter.GiB, 8 * adapter.GiB)
            self.assertTrue(cp.experts_enabled)
            self.assertEqual(cp.cache_bytes, 1024**2)

    def test_text_prefill_reserve_tracks_native_peak_and_vocabulary(self):
        manager = object.__new__(adapter.TextExpertManager)
        manager.checkpoint = SimpleNamespace(config={"text_config": {"vocab_size": 248320}})
        self.assertEqual(manager.prefill_transient_reserve(2048, 1024**3), 8 * 2048 * 248320)
        self.assertEqual(manager.prefill_transient_reserve(2048, 8 * 1024**3), 8 * 1024**3)
        self.assertLess(manager.prefill_transient_reserve(1, 1024), 2 * 1024**2)

    def test_hardware_ceiling_reserves_experts_and_scales_beyond_one_gib(self):
        gib = 1024**3
        options = dict(metal_limit=37*gib, available_memory=35*gib, base_bytes=5*gib,
                       expert_minimum=8*1024**2, minimum_row_bytes=1024,
                       maximum_row_cache_bytes=64*gib)
        budget = adapter.automatic_ngram_budget(**options)
        self.assertGreater(budget, gib)
        self.assertLess(budget, 4*gib)
        large = adapter.automatic_ngram_budget(**{**options, 'metal_limit': 2048*gib,
                                                  'available_memory': 2048*gib})
        self.assertEqual(large, 64*gib)
        self.assertLess(adapter.automatic_ngram_budget(**{**options, 'available_memory': 12*gib}), budget)
        for override in ({'available_memory': 0}, {'metal_limit': 0}, {'available_memory': 6*gib}):
            with self.assertRaises(ValueError):
                adapter.automatic_ngram_budget(**{**options, **override})

    def manager(self, automatic=True):
        from test_omlx_offload import write_fixture
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        write_fixture(path)
        cp = adapter.TextCheckpoint(path, 1024**2, None if automatic else 1024)
        row = cp.largest_ngram_row_bytes
        cp.ngram_initial_cache_bytes = row
        cp.ngram_budget_bytes = row * 4
        return adapter.TextExpertManager(cp), row

    def test_auto_grows_only_after_demand_eviction_and_recovers_from_pressure(self):
        manager, row = self.manager()
        expert = max(manager.checkpoint.expert_bytes.values())
        room = 1024**3
        self.assertEqual(manager.resize_auxiliary_cache(room), row)
        for key in range(2):
            with manager.ngram_cache.acquire('ple', key, row, lambda: object()):
                pass
        self.assertEqual(manager.resize_auxiliary_cache(room), row * 2)
        self.assertEqual(manager.resize_auxiliary_cache(room), row * 2)
        with manager.ngram_cache.acquire('ple', 2, row, lambda: object()):
            pass
        self.assertEqual(manager.resize_auxiliary_cache(expert + row), row)
        self.assertEqual(manager.resize_auxiliary_cache(room), row * 2)
        self.assertEqual(manager.resize_auxiliary_cache(expert + row - 1), row)
        self.assertEqual(manager.resize_auxiliary_cache(-1024**3), row)
        self.assertEqual(manager.status()['ngram_cache_budget_mode'], 'auto')

    def test_explicit_budget_does_not_become_automatic(self):
        manager, row = self.manager(automatic=False)
        self.assertEqual(manager.resize_auxiliary_cache(1024**3), row * 4)
        for key in range(8):
            with manager.ngram_cache.acquire('ple', key, row, lambda: object()):
                pass
        self.assertEqual(manager.resize_auxiliary_cache(1024**3), row * 4)
        self.assertEqual(manager.status()['ngram_cache_budget_mode'], 'explicit')


def restart_prefix(test, model, model_path, state, tokens, directory):
    """Exercise the installed SSD codec after closing and reopening its index."""
    import mlx.core as mx
    import omlx_persistence as persistence
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache
    from omlx.scheduler import Scheduler
    manager = persistence.PrefixPersistence(model_path, directory, "d" * 64)

    def native_cache():
        ssd = PagedSSDCacheManager(cache_dir=manager.native_directory, max_size_bytes=16 * 1024 * 1024,
            expected_model_name="model", expected_num_layers=len(model.layers), expected_block_size=2048)
        paged = PagedCacheManager(block_size=2048, max_blocks=16, model_name="model", initial_blocks=4)
        paged.set_paged_ssd_cache_manager(ssd)
        return BlockAwarePrefixCache(model, paged, ssd), ssd

    cache, ssd = native_cache()
    test.addCleanup(ssd.close)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    scheduler.config = SimpleNamespace(model_path=str(model_path), model_name="model")
    scheduler.block_aware_cache = cache
    scheduler._stream = mx.default_stream(mx.default_device())
    scheduler._bypass_hot_cache_under_pressure = lambda: False
    scheduler._release_paged_cache_for_request = lambda _: None
    manager.attach(scheduler)
    test.assertTrue(manager.status()["active"], manager.status())
    request = SimpleNamespace(request_id="store", prompt_token_ids=[*tokens, 21])
    test.assertTrue(manager.store(scheduler, request, tokens, state), manager.status())
    ssd.close()
    scheduler.block_aware_cache, fresh_ssd = native_cache()
    test.addCleanup(fresh_ssd.close)
    fresh = persistence.PrefixPersistence(model_path, directory, "d" * 64)
    fresh.attach(scheduler)
    request.cached_tokens = 0
    request.block_table = None
    test.assertTrue(fresh.restore(scheduler, request), fresh.status())
    test.assertEqual(request.remaining_tokens, [21])
    test.assertEqual(request.cached_tokens, len(tokens))
    return request.prompt_cache


def tiny_config(architecture):
    common = dict(model_type=architecture + "_text", vocab_size=128, hidden_size=64,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                  moe_intermediate_size=64, num_experts_per_tok=2,
                  max_position_embeddings=128, rms_norm_eps=1e-6, eos_token_id=0,
                  tie_word_embeddings=False)
    if architecture == "qwen4_exp":
        common.update(num_experts=4, shared_expert_intermediate_size=64,
                      linear_num_value_heads=2, linear_num_key_heads=1,
                      linear_key_head_dim=32, linear_value_head_dim=32,
                      linear_conv_kernel_dim=4, head_dim=32, hc_count=2,
                      hc_lowrank=32, layer_types=["linear_attention", "full_attention"],
                      indexer_head_dim=32, indexer_n_heads=2, indexer_budget=8,
                      ple_layer_ids=[1], ple_embed_dim=64, heads_per_ngram=1,
                      ngram_size=2, ngram_vocab_size_base=17,
                      make_ngram_vocab_size_divisible_by=32, split_ngram_parts=2,
                      mtp_num_hidden_layers=0, rope_parameters={"type":"default",
                      "mrope_section":[1,1,2],"rope_theta":10000.0,"partial_rotary_factor":0.25})
        vision = dict(model_type=architecture, depth=1, hidden_size=32,
                      intermediate_size=64, num_heads=2, out_hidden_size=64,
                      patch_size=2, temporal_patch_size=1, spatial_merge_size=1)
    else:
        common.update(intermediate_size=64, n_shared_experts=1, n_routed_experts=4,
                      routed_scaling_factor=1.0, kv_lora_rank=32, q_lora_rank=32,
                      qk_rope_head_dim=16, v_head_dim=32, qk_nope_head_dim=32,
                      first_k_dense_replace=1, index_topk=4, index_head_dim=32,
                      index_n_heads=2, layer_types=["linear_attention", "full_attention"],
                      mlp_layer_types=["dense", "sparse"], hc_mult=4, eos_token_id=[0],
                      linear_attn_config={"num_heads":1,"head_dim":64,"short_conv_kernel_size":4})
        vision = None
    return dict(model_type=architecture, text_config=common, vision_config=vision,
                quantization={"bits":4,"group_size":32,"mode":"affine"})


@unittest.skipUnless(os.getenv("WERK_TEST_MLX_EXPERTS") == "1", "native Metal opt-in")
class NativeTextLoaderTests(unittest.TestCase):
    def test_resident_shards_preserve_dtypes_and_reject_changed_checkpoint(self):
        import mlx.core as mx
        from omlx_offload import signature

        with tempfile.TemporaryDirectory() as directory:
            selected = {}
            expected = {}
            for index, dtype in enumerate((mx.bfloat16, mx.uint32)):
                path = Path(directory) / f"model-{index}.safetensors"
                name = f"model.layers.{index}.weight"
                value = mx.array([[1, 2], [3, 4]], dtype=dtype)
                mx.save_safetensors(str(path), {name: value, "vision_tower.excluded": mx.ones((8,))})
                selected[name] = SimpleNamespace(path=path, signature=signature(path.stat()), shape=(2, 2))
                expected[adapter._canonical_name(name, "qwen4_exp")] = value
            weights = adapter._load_resident_weights(selected, "qwen4_exp")
            self.assertEqual(set(weights), set(expected))
            mx.eval(weights)
            adapter._check_resident_shards(selected)
            for name, value in weights.items():
                self.assertEqual(value.dtype, expected[name].dtype)
                self.assertTrue(mx.array_equal(value, expected[name]).item())
            with path.open("ab") as file:
                file.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                adapter._check_resident_shards(selected)
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                adapter._load_resident_weights(selected, "qwen4_exp")

    def test_qwen_tool_probe_matches_native_tokenizer_and_typed_parser(self):
        import omlx_probe as probe
        from mlx_lm import utils
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        from mlx_lm.tool_parsers import qwen3_coder as parser

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            adapter.native_classes(tiny_config("qwen4_exp"), path)
            template = "{{ '<tool_call>\\n<function=' }}{{ tools | tojson }}"
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
                unk_token="[UNK]", chat_template=template)
            tokenizer.save_pretrained(path)
            metadata = json.loads((path / "tokenizer_config.json").read_text())
            config = {"model_type": "qwen4_exp"}
            self.assertTrue(probe.supports_tools(config, metadata, path, utils))
            loaded = utils.load_tokenizer(path, {"local_files_only": True})
            self.assertIs(loaded.tool_parser, parser.parse_tool_call)
            self.assertEqual(loaded.tool_call_start, "<tool_call>")
            tools = [{"type": "function", "function": {"name": "add", "parameters": {
                "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}}]
            parsed = loaded.tool_parser("<function=add>\n<parameter=a>2</parameter>\n<parameter=b>3</parameter>\n</function>", tools)
            self.assertEqual(parsed, {"name": "add", "arguments": {"a": 2, "b": 3}})
            for override in ({"tool_parser_type": None}, {"tool_parser_type": "json_tools"},
                             {"chat_template_type": "deepseek_v4"}):
                self.assertFalse(probe.supports_tools(config, {**metadata, **override}, path, utils))
            (path / "chat_template.jinja").write_text("{{ messages }}")
            self.assertFalse(probe.supports_tools(config, metadata, path, utils))
            (path / "chat_template.jinja").write_text(template)
            with patch.object(parser, "parse_tool_call", lambda *_: {}):
                self.assertFalse(probe.supports_tools(config, metadata, path, utils))
            with patch.object(utils, "_load_tokenizer", lambda *_: None):
                self.assertFalse(probe.supports_tools(config, metadata, path, utils))
            self.assertFalse(probe.supports_tools({"model_type": "glm5_next"}, metadata, path, utils))

    def test_glm_tool_probe_matches_native_tokenizer_and_typed_parser(self):
        import omlx_probe as probe
        from mlx_lm import utils
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        from mlx_lm.tool_parsers import glm47 as parser

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            adapter.native_classes(tiny_config("glm5_next"), path)
            template = "{{ '<tool_call>lookup<arg_key>query</arg_key><arg_value>x</arg_value></tool_call>' }}{{ tools | tojson }}"
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
                unk_token="[UNK]", chat_template=template)
            tokenizer.save_pretrained(path)
            metadata = json.loads((path / "tokenizer_config.json").read_text())
            config = {"model_type": "glm5_next"}
            self.assertTrue(probe.supports_tools(config, metadata, path, utils))
            loaded = utils.load_tokenizer(path, {"local_files_only": True})
            self.assertIs(loaded.tool_parser, parser.parse_tool_call)
            self.assertEqual((loaded.tool_call_start, loaded.tool_call_end), ("<tool_call>", "</tool_call>"))
            tools = [{"type": "function", "function": {"name": "lookup", "parameters": {
                "type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"},
                "enabled": {"type": "boolean"}, "filters": {"type": "object"}}}}}]
            parsed = loaded.tool_parser('lookup<arg_key>query</arg_key><arg_value>001</arg_value>'
                '<arg_key>limit</arg_key><arg_value>2</arg_value>'
                '<arg_key>enabled</arg_key><arg_value>false</arg_value>'
                '<arg_key>filters</arg_key><arg_value>{"label":"Grün"}</arg_value>', tools)
            self.assertEqual(parsed, {"name": "lookup", "arguments": {
                "query": "001", "limit": 2, "enabled": False, "filters": {"label": "Grün"}}})
            self.assertEqual(loaded.tool_parser("ping", []), {"name": "ping", "arguments": {}})
            for override in ({"tool_parser_type": None}, {"tool_parser_type": "qwen3_coder"},
                             {"chat_template_type": "deepseek_v4"}):
                self.assertFalse(probe.supports_tools(config, {**metadata, **override}, path, utils))
            (path / "chat_template.jinja").write_text("{{ messages }}")
            self.assertFalse(probe.supports_tools(config, metadata, path, utils))
            (path / "chat_template.jinja").write_text(template)
            with patch.object(parser, "parse_tool_call", lambda *_: {}):
                self.assertFalse(probe.supports_tools(config, metadata, path, utils))
            with patch.object(parser, "_deserialize", lambda value: value):
                self.assertFalse(probe.supports_tools(config, metadata, path, utils))
            with patch.object(utils, "_load_tokenizer", lambda *_: None):
                self.assertFalse(probe.supports_tools(config, metadata, path, utils))
            (path / "chat_templates").mkdir()
            self.assertFalse(probe.supports_tools(config, metadata, path, utils))

    def test_glm_fused_projections_release_duplicate_storage_and_preserve_fallback(self):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np

        for quantized in (False, True):
            with self.subTest(quantized=quantized), tempfile.TemporaryDirectory() as directory:
                _, args = adapter.native_classes(tiny_config("glm5_next"), Path(directory))
                from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention, linear_forward
                attention = Glm5NextLinearAttention(args.text_config)
                attention.set_dtype(mx.bfloat16)
                if quantized:
                    nn.quantize(attention, group_size=32, bits=4)
                attention.eval()
                x = mx.ones((1, 2, args.text_config.hidden_size), dtype=mx.bfloat16)
                expected = attention._fused_in_proj(x)
                mx.eval(expected)
                expected = [np.array(value.astype(mx.float32)) for value in expected]
                fields = [attention._fw]
                if quantized:
                    fields.extend((attention._fs, attention._fb))
                fused_bytes = sum(array.nbytes for array in fields)
                before = mx.get_active_memory()
                adapter.share_attention_fusion(attention)
                after = mx.get_active_memory()
                # Assert actual MLX allocation savings, not summed view sizes.
                self.assertGreaterEqual(before - after, fused_bytes)
                actual = attention._fused_in_proj(x)
                mx.eval(actual)
                for left, right in zip(expected, actual):
                    np.testing.assert_array_equal(left, np.array(right.astype(mx.float32)))
                modules = [attention.q_proj, attention.k_proj, attention.v_proj,
                           attention.forget_gate.f_a_proj, attention.g_a_proj, attention.b_proj]
                fallback = [linear_forward(module, x) for module in modules]
                mx.eval(fallback)
                for left, right in zip(expected, fallback):
                    np.testing.assert_allclose(left, np.array(right.astype(mx.float32)), atol=0.02, rtol=0.02)

    def test_installed_models_match_offloaded_prefill_and_decode(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        import numpy as np

        for architecture, with_vision in (("qwen4_exp", True), ("glm5_next", False), ("glm5_next", True)):
            with self.subTest(architecture=architecture, with_vision=with_vision), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                config = tiny_config(architecture)
                if architecture == "glm5_next" and with_vision:
                    config["vision_config"] = dict(
                        model_type=architecture, depth=1, hidden_size=32,
                        intermediate_size=64, num_heads=2, patch_size=2,
                        out_hidden_size=64, projection_intermediate_size=64,
                        spatial_merge_size=1, temporal_patch_size=1, image_size=8)
                    for projection in ("f_a_proj", "f_b_proj"):
                        config["quantization"][f"language_model.model.layers.0.self_attn.{projection}"] = {
                            "bits": 8, "group_size": 32, "mode": "affine"}
                config["quantization"]["vision_tower.excluded_projection"] = {"bits": 8, "group_size": 64}
                config["quantization"]["language_model.mtp.excluded_projection"] = {"bits": 2, "group_size": 64}
                (path / "config.json").write_text(json.dumps(config))
                model_class, args = adapter.native_classes(config, path)
                if architecture == "qwen4_exp":
                    from mlx_vlm.models.qwen4_exp import language
                    language.configure_ple_runtime(path, mode="resident")
                    language.configure_mtp_runtime(path, enabled=False)
                mx.random.seed(91)
                reference = model_class(args)
                for name in ("vision_tower", "vision_model"):
                    if name in reference:
                        setattr(reference, name, None)
                reference.set_dtype(mx.bfloat16)
                def quantize(path, module):
                    if architecture == "glm5_next" and with_vision and path.endswith(("f_a_proj", "f_b_proj")):
                        return {"group_size": 32, "bits": 8}
                    return (hasattr(module, "to_quantized")
                            and hasattr(module, "weight")
                            and module.weight.shape[-1] % 32 == 0
                            and not path.endswith("mlp.gate"))
                nn.quantize(reference, group_size=32, bits=4, class_predicate=quantize)
                reference.eval()
                weights = dict(tree_flatten(reference.parameters()))
                mx.eval(weights)
                saved_weights = weights
                if architecture == "glm5_next" and with_vision:
                    # This real checkpoint stores affine forget-gate triplets
                    # in the flat HF namespace, including Q8 overrides.
                    saved_weights = {name.replace(".self_attn.forget_gate.f_", ".self_attn.f_"): value
                                     for name, value in weights.items()}
                mx.save_safetensors(str(path / "model.safetensors"), saved_weights)
                reference.load_weights(list(reference.sanitize(dict(weights)).items()), strict=True)
                modes = [(16384, 1024), (16384, 0), (None, 1024), (None, 0), (0, None)] if architecture == "qwen4_exp" else [(16384, None), (None, 0), (0, None)]
                for expert_budget, ngram_budget in modes:
                    with self.subTest(expert_budget=expert_budget, ngram_budget=ngram_budget):
                        cp = adapter.TextCheckpoint(path, expert_budget, ngram_budget)
                        if expert_budget == 0:
                            cp.configure_auto()
                            self.assertFalse(cp.experts_enabled)
                        manager = adapter.TextExpertManager(cp)
                        with patch("mlx_lm.utils.load_tokenizer", return_value=object()):
                            model, _ = adapter.load_text_model(manager)
                        if architecture == "glm5_next":
                            attention = model.core.language_model.model.layers[0].self_attn
                            # Uniform quantization fuses; mixed Q8/Q4 retains
                            # the native fallback. Both must be ready before
                            # prefill admission and remain numerically equal.
                            self.assertEqual(attention._fused_ready, not with_vision)
                            self.assertEqual(cp.attention_fusion_bytes > 0, not with_vision)
                        else:
                            self.assertEqual(cp.attention_fusion_bytes, 0)
                        left_cache = reference.language_model.make_cache()
                        right_cache = model.make_cache()
                        all_tokens = []
                        for tokens in ([1, 2, 0, 3], [4], [5], [0], [7], [9], [11], [13], [15], [17], [19]):
                            all_tokens.extend(tokens)
                            inputs = mx.array([tokens])
                            left = reference.language_model(inputs, cache=left_cache).logits
                            right = model(inputs, cache=right_cache)
                            mx.eval(left, right)
                            self.assertTrue(bool(mx.all(mx.isfinite(left)).item()), f"native reference is nonfinite: {architecture}, {tokens}")
                            self.assertTrue(bool(mx.all(mx.isfinite(right)).item()), f"offloaded model is nonfinite: {architecture}, {tokens}")
                            np.testing.assert_allclose(np.array(right.astype(mx.float32)),
                                                       np.array(left.astype(mx.float32)), atol=0.02, rtol=0.02)
                        self.assertTrue(manager.status()["active"])
                        if cp.experts_enabled:
                            self.assertEqual(manager.status()["cache_policy"], "segmented_lru")
                            rows = manager.list_experts({"allow_experimental":True})["experts"]
                            self.assertEqual(len(rows), len(cp.expert_bytes) * cp.experts)
                        self.assertLessEqual(manager.ngram_cache.used, max(1, cp.ngram_budget_bytes))
                        with tempfile.TemporaryDirectory() as cache_directory:
                            restored = restart_prefix(self, model, path, right_cache, all_tokens, Path(cache_directory))
                            for token in (21, 23, 0, 25):
                                inputs = mx.array([[token]])
                                left = model(inputs, cache=right_cache)
                                right = model(inputs, cache=restored)
                                mx.eval(left, right)
                                np.testing.assert_array_equal(np.array(left.astype(mx.float32)), np.array(right.astype(mx.float32)))
                        manager.deactivate()
                        del model

    def test_real_private_workers_stream_text_and_reuse_native_prefixes(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        source = Path(__file__).resolve().parent
        bootstrap = "import sys, importlib\nsys.path.insert(0, " + repr(str(source)) + ")\n"
        for name in ("omlx_experts", "omlx_offload", "omlx_offload_runtime", "omlx_text_offload", "omlx_persistence"):
            bootstrap += f"sys.modules['_werk_{name}'] = importlib.import_module('{name}')\n"
        bootstrap += "from omlx_supervisor import main\nmain()\n"
        launcher = Path(sys.executable).parent / "omlx"
        self.assertTrue(launcher.is_file())
        for architecture, expert_mode in (("qwen4_exp", "16384"), ("glm5_next", "16384"),
                                          ("qwen4_exp", "0"), ("glm5_next", "0")):
            with self.subTest(architecture=architecture, expert_mode=expert_mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "model"
                path.mkdir()
                config = tiny_config(architecture)
                (path / "config.json").write_text(json.dumps(config))
                model_class, args = adapter.native_classes(config, path)
                if architecture == "qwen4_exp":
                    from mlx_vlm.models.qwen4_exp import language
                    language.configure_ple_runtime(path, mode="resident")
                    language.configure_mtp_runtime(path, enabled=False)
                model = model_class(args)
                for name in ("vision_tower", "vision_model"):
                    if name in model: model[name] = None
                model.set_dtype(mx.bfloat16)
                nn.quantize(model, group_size=32, bits=4, class_predicate=lambda p, m:
                            hasattr(m, "to_quantized") and m.weight.shape[-1] % 32 == 0 and not p.endswith("mlp.gate"))
                mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
                del model
                mx.clear_cache()
                tokenizer = Tokenizer(models.WordLevel({"<eos>": 0, "<unk>": 1, **{f"word{i}": i for i in range(2, 128)}}, unk_token="<unk>"))
                tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
                fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>", unk_token="<unk>")
                fast.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
                fast.save_pretrained(path)
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                environment = dict(os.environ, HF_HOME=str(root / "hf"), WERK_OMLX_EXPERT_MODEL_DIR=str(path), WERK_OMLX_EXPERT_CACHE_BYTES=expert_mode,
                                   WERK_OMLX_PERSISTENCE_DIR=str(root / "prefix"), WERK_OMLX_PERSISTENCE_MODEL_DIR=str(path))
                if architecture == "qwen4_exp": environment["WERK_OMLX_NGRAM_CACHE_BYTES"] = "1024"
                else: environment.pop("WERK_OMLX_NGRAM_CACHE_BYTES", None)
                with (root / "worker.log").open("w+") as log:
                    child = subprocess.Popen([sys.executable, "-I", "-c", bootstrap, str(launcher), "serve",
                        "--model-dir", str(path), "--base-path", str(root / "worker"), "--host", "127.0.0.1",
                        "--port", str(port), "--api-key", "fixture-secret", "--paged-ssd-cache-dir", str(root / "prefix"),
                        "--paged-ssd-cache-max-size", "32MB"], env=environment, stdin=subprocess.PIPE,
                        stdout=log, stderr=log, start_new_session=True)
                    def request(endpoint, body=None, stream=False):
                        req = urllib.request.Request(f"http://127.0.0.1:{port}" + endpoint,
                            data=None if body is None else json.dumps(body).encode(),
                            headers={"Authorization": "Bearer fixture-secret", "Content-Type": "application/json"})
                        with urllib.request.urlopen(req, timeout=60) as response:
                            return response.read().decode() if stream else json.load(response)
                    try:
                        deadline = time.monotonic() + 45
                        while True:
                            try:
                                discovered = request("/v1/models")
                                break
                            except OSError:
                                if child.poll() is not None or time.monotonic() >= deadline:
                                    raise RuntimeError("private worker did not become ready")
                                time.sleep(0.1)
                        name = next(item["id"] for item in discovered["data"] if item["id"] == path.name)
                        request(f"/v1/models/{name}/load", {})
                        status = request("/werk/experts/status")
                        self.assertTrue(status["active"])
                        self.assertEqual(status["experts_offloaded"], expert_mode != "0")
                        for prompt in ("word2 word3 word4 word5", "word2 word3 word4 word5 word6"):
                            stream = request("/v1/chat/completions", {"model": name, "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 4, "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}, stream=True)
                            self.assertIn("data: [DONE]", stream)
                            events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: {")]
                            self.assertTrue(any(event.get("choices") for event in events))
                            self.assertFalse(any("error" in event for event in events), events)
                        persistence = request("/werk/persistence/status")
                        self.assertTrue(persistence["active"], persistence)
                        self.assertGreater(persistence["stores"], 0, persistence)
                    except BaseException:
                        log.flush(); log.seek(0)
                        print(log.read()[-16000:])
                        raise
                    finally:
                        child.stdin.close()
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait(timeout=5)



if __name__ == "__main__":
    unittest.main()
