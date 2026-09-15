"""Native prefix hooks: portable contract tests and opt-in real SSD codec test."""

from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref


SPEC = importlib.util.spec_from_file_location("omlx_persistence", Path(__file__).with_name("omlx_persistence.py"))
persistence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(persistence)
KVCache = type("KVCache", (), {})


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"llama"}')
        self.manager = persistence.PrefixPersistence(self.model, self.root / "cache", "a" * 64)
        self.cache = SimpleNamespace(restore_exact_prefix=Mock(return_value=[KVCache()]))
        self.scheduler = SimpleNamespace(block_aware_cache=self.cache,
            _bypass_hot_cache_under_pressure=lambda: False,
            _release_paged_cache_for_request=Mock())
        # SimpleNamespace lacks weakref support; tests use a plain subclass.
        class Scheduler:
            pass
        scheduler = Scheduler()
        scheduler.__dict__.update(vars(self.scheduler))
        self.scheduler = scheduler
        self.manager._scheduler = weakref.ref(scheduler)
        self.manager.max_prefix_tokens = 2048

    def request(self, tokens):
        return SimpleNamespace(request_id="request", prompt_token_ids=tokens,
                               prompt_cache=None, cached_tokens=0, remaining_tokens=tokens,
                               block_table=None, shared_prefix_blocks=0)

    def test_token_digest_is_ordered_and_rejects_non_token_types(self):
        self.assertNotEqual(persistence._token_digest([1, 2]), persistence._token_digest([2, 1]))
        for invalid in ([True], [-1], [2**32], [1.5], ["1"]):
            with self.assertRaises(ValueError):
                persistence._token_digest(invalid)

    def test_side_index_is_private_and_contains_only_token_count_and_digest(self):
        self.manager._commit_index([12345, 98765])
        data = json.loads(self.manager.index_path.read_text())
        self.assertEqual(data["entries"], [{"tokens": 2, "digest": persistence._token_digest([12345, 98765])}])
        self.assertNotIn("12345", self.manager.index_path.read_text())
        self.assertEqual(stat.S_IMODE(self.manager.index_path.stat().st_mode), 0o600)
        fresh = persistence.PrefixPersistence(self.model, self.root / "cache", "a" * 64)
        self.assertEqual(fresh._entries, self.manager._entries)

    def test_index_is_bounded_and_refreshes_existing_prefix(self):
        with patch.object(persistence, "_MAX_ENTRIES", 2):
            for tokens in ([1], [2], [1], [3]):
                self.manager._commit_index(tokens)
        self.assertEqual([row["digest"] for row in self.manager._entries],
                         [persistence._token_digest([3]), persistence._token_digest([1])])

    def test_corrupt_or_wrong_fingerprint_index_is_ignored(self):
        for text in ("not json", json.dumps({"format": persistence.FORMAT, "fingerprint": "b" * 64, "entries": []})):
            self.manager.index_path.write_text(text)
            fresh = persistence.PrefixPersistence(self.model, self.root / "cache", "a" * 64)
            self.assertEqual(fresh._entries, [])

    def test_fingerprint_changes_when_tokenizer_or_checkpoint_changes(self):
        versions = {"omlx": "0.6.4", "mlx": "0.32.2", "mlx-lm": "0.31.3"}
        first = persistence._fingerprint(self.model, versions)
        (self.model / "tokenizer.json").write_text("tokenizer configuration")
        second = persistence._fingerprint(self.model, versions)
        self.assertNotEqual(first, second)
        (self.model / "model.safetensors").write_bytes(b"weights")
        self.assertNotEqual(second, persistence._fingerprint(self.model, versions))

    def test_restore_leaves_real_kickoff_token_and_prefers_longest_exact_prefix(self):
        self.manager._commit_index([1, 2])
        self.manager._commit_index([1, 2, 3, 4])
        request = self.request([1, 2, 3, 4, 5])
        self.assertTrue(self.manager.restore(self.scheduler, request))
        self.assertEqual(request.cached_tokens, 4)
        self.assertEqual(request.remaining_tokens, [5])
        self.assertEqual(self.manager.restored_tokens, 4)
        self.assertEqual(self.cache.restore_exact_prefix.call_args.args[1], [1, 2, 3, 4])

    def test_exact_same_length_never_replays_last_token(self):
        self.manager._commit_index([1, 2, 3])
        request = self.request([1, 2, 3])
        self.assertFalse(self.manager.restore(self.scheduler, request))
        self.cache.restore_exact_prefix.assert_not_called()
        self.assertEqual(request.remaining_tokens, [1, 2, 3])

    def test_divergent_tokens_do_not_restore_cache(self):
        self.manager._commit_index([1, 2])
        self.assertFalse(self.manager.restore(self.scheduler, self.request([1, 99, 3])))
        self.cache.restore_exact_prefix.assert_not_called()

    def test_interleaved_chats_restore_only_their_matching_prefix(self):
        first_prefix = [1, 2, 10, 11]
        second_prefix = [1, 2, 20, 21]
        snapshots = {tuple(tokens): tuple(tokens) for tokens in (first_prefix, second_prefix)}

        def restore_native(_request_id, tokens, **_options):
            cache = KVCache()
            cache.tokens = snapshots[tuple(tokens)]
            return [cache]

        self.cache.restore_exact_prefix.side_effect = restore_native
        self.manager._commit_index(first_prefix)
        first = self.request(first_prefix + [12])
        first.request_id = "chat-a-first"
        self.assertTrue(self.manager.restore(self.scheduler, first))

        # Another conversation becomes the newest index entry in the worker.
        self.manager._commit_index(second_prefix)
        second = self.request(second_prefix + [22])
        second.request_id = "chat-b"
        self.assertTrue(self.manager.restore(self.scheduler, second))
        resumed = self.request(first_prefix + [12, 13])
        resumed.request_id = "chat-a-follow-up"
        self.assertTrue(self.manager.restore(self.scheduler, resumed))

        self.assertEqual(first.prompt_cache[0].tokens, tuple(first_prefix))
        self.assertEqual(second.prompt_cache[0].tokens, tuple(second_prefix))
        self.assertEqual(resumed.prompt_cache[0].tokens, tuple(first_prefix))
        self.assertEqual(resumed.cached_tokens, len(first_prefix))
        self.assertEqual(resumed.remaining_tokens, [12, 13])
        self.assertEqual([call.args[1] for call in self.cache.restore_exact_prefix.call_args_list],
                         [first_prefix, second_prefix, first_prefix])
        divergent = self.request([1, 2, 30, 31, 32])
        self.assertFalse(self.manager.restore(self.scheduler, divergent))
        self.assertEqual(self.cache.restore_exact_prefix.call_count, 3)
        self.assertIsNone(divergent.prompt_cache)

    def test_each_request_restores_its_own_mutable_native_cache(self):
        prefix = [1, 2, 3]
        self.manager._commit_index(prefix)

        def restore_native(_request_id, tokens, **_options):
            cache = KVCache()
            cache.tokens = list(tokens)
            return [cache]

        self.cache.restore_exact_prefix.side_effect = restore_native
        first = self.request(prefix + [10])
        second = self.request(prefix + [20])
        first.request_id = "first"
        second.request_id = "second"
        self.assertTrue(self.manager.restore(self.scheduler, first))
        self.assertTrue(self.manager.restore(self.scheduler, second))
        self.assertIsNot(first.prompt_cache, second.prompt_cache)
        self.assertIsNot(first.prompt_cache[0], second.prompt_cache[0])
        first.prompt_cache[0].tokens.append(10)
        first.remaining_tokens.append(11)
        self.assertEqual(second.prompt_cache[0].tokens, prefix)
        self.assertEqual(second.remaining_tokens, [20])
        native_ids = [call.args[0] for call in self.cache.restore_exact_prefix.call_args_list]
        self.assertEqual(len(set(native_ids)), 2)

    def test_unsupported_cache_attachment_disables_only_the_extension(self):
        self.manager._commit_index([1, 2])
        self.scheduler.config = SimpleNamespace(model_path=str(self.model), model_name="model")
        self.scheduler.model = object()
        fake_cache = ModuleType("mlx_lm.models.cache")
        fake_cache.make_prompt_cache = Mock(return_value=[object()])
        modules = {"mlx_lm": ModuleType("mlx_lm"), "mlx_lm.models": ModuleType("mlx_lm.models"),
                   "mlx_lm.models.cache": fake_cache}
        with patch.dict(sys.modules, modules):
            self.manager.attach(self.scheduler)

        status = self.manager.status()
        self.assertFalse(status["active"])
        self.assertEqual(status["max_prefix_tokens"], 0)
        self.assertIn("not verified", status["reason"])
        request = self.request([1, 2, 3])
        normal_cache = [KVCache()]
        normal_table = object()
        request.prompt_cache = normal_cache
        request.block_table = normal_table
        request.cached_tokens = 1
        request.remaining_tokens = [2, 3]
        self.assertFalse(self.manager.restore(self.scheduler, request))
        self.assertFalse(self.manager.store(self.scheduler, request, [1, 2], normal_cache))
        self.assertIs(request.prompt_cache, normal_cache)
        self.assertIs(request.block_table, normal_table)
        self.assertEqual(request.cached_tokens, 1)
        self.assertEqual(request.remaining_tokens, [2, 3])
        self.cache.restore_exact_prefix.assert_not_called()
        self.scheduler._release_paged_cache_for_request.assert_not_called()

    def test_native_eviction_or_restore_failure_keeps_normal_prefill(self):
        self.manager._commit_index([1, 2])
        for result in (None, RuntimeError("native cache unavailable")):
            self.cache.restore_exact_prefix = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
            request = self.request([1, 2, 3])
            self.assertFalse(self.manager.restore(self.scheduler, request))
            self.assertIsNone(request.prompt_cache)
            self.assertEqual(request.remaining_tokens, [1, 2, 3])

    def test_longer_existing_cache_and_multimodal_requests_remain_unchanged(self):
        self.manager._commit_index([1, 2])
        request = self.request([1, 2, 3, 4])
        request.cached_tokens = 3
        self.assertFalse(self.manager.restore(self.scheduler, request))
        request.cached_tokens = 0
        request.vlm_extra_keys_for_cache = ["image digest"]
        self.assertFalse(self.manager.restore(self.scheduler, request))
        self.cache.restore_exact_prefix.assert_not_called()

    def test_replacing_shorter_paged_hit_releases_its_ownership(self):
        self.manager._commit_index([1, 2, 3])
        request = self.request([1, 2, 3, 4])
        request.cached_tokens = 1
        request.block_table = object()
        self.assertTrue(self.manager.restore(self.scheduler, request))
        self.scheduler._release_paged_cache_for_request.assert_called_once_with("request")
        self.assertIsNone(request.block_table)

    def test_durable_requires_committed_native_file_not_just_index_metadata(self):
        digest = b"native hash"
        cache_file = self.root / "block.safetensors"
        self.cache._exact_prefix_hashes = lambda tokens: [(digest, len(tokens))]
        index = Mock()
        index.get.return_value = SimpleNamespace(file_path=cache_file)
        self.cache.paged_ssd_cache = SimpleNamespace(_pending_write_hashes_lock=threading.Lock(),
            _pending_write_hashes=set(), _index=index)
        self.assertFalse(self.manager._durable(self.cache, [1]))
        cache_file.write_bytes(b"native tensor data")
        self.assertTrue(self.manager._durable(self.cache, [1]))
        self.cache.paged_ssd_cache._pending_write_hashes.add(digest)
        with patch.object(persistence, "_COMMIT_TIMEOUT", 0):
            self.assertFalse(self.manager._durable(self.cache, [1]))

    def test_failed_or_uncommitted_store_does_not_publish_prefix(self):
        fake_omlx = ModuleType("omlx")
        fake_scheduler = ModuleType("omlx.scheduler")
        fake_scheduler._safe_sync_stream = lambda stream: None
        fake_mlx = ModuleType("mlx")
        fake_core = ModuleType("mlx.core")
        fake_core.stream = lambda stream: nullcontext()
        self.scheduler._stream = None
        self.scheduler._extract_cache_states = lambda cache: ([{"state": "native"}], None)
        self.cache.store_exact_prefix = Mock(return_value=SimpleNamespace(num_tokens=2))
        modules = {"omlx": fake_omlx, "omlx.scheduler": fake_scheduler,
                   "mlx": fake_mlx, "mlx.core": fake_core}
        with patch.dict(sys.modules, modules), patch.object(self.manager, "_durable", return_value=False):
            self.assertFalse(self.manager.store(self.scheduler, self.request([1, 2, 3]), [1, 2], [KVCache()]))
        self.assertFalse(self.manager.index_path.exists())
        self.assertEqual(self.manager.stores, 0)

    def test_installed_hooks_keep_normal_outputs_and_bind_only_selected_model(self):
        owner = self
        class Scheduler:
            def __init__(self, model, tokenizer, config=None, stream=None):
                self.config = config
                self.model = model
                self._stream = stream
                self.block_aware_cache = SimpleNamespace(
                    block_size=2048, paged_ssd_cache=object(),
                    store_exact_prefix=Mock(), restore_exact_prefix=Mock(return_value=[KVCache()]))
                self._prefix_cache_prepared = set()
                self.uid_to_request_id = {7: "request"}
                self.running = {}
                self._bypass_hot_cache_under_pressure = lambda: False

            def _prepare_prefix_cache_for_request(self, request):
                request.remaining_tokens = request.prompt_token_ids
                self._prefix_cache_prepared.add(request.request_id)
                return "normal prepare result"

            def _do_external_prefill(self, request, tokens, existing_cache, vlm_embeds=None):
                return [KVCache()], tokens[-1:]

            def _finalize_chunked_prefill_cache_for_insert(self, request, prompt_cache):
                return "normal finalize result"

            def _process_batch_responses(self, responses):
                self.running.clear()
                return ["streamed output"], {"request"}

        fake_omlx = ModuleType("omlx")
        fake_scheduler = ModuleType("omlx.scheduler")
        fake_scheduler.Scheduler = Scheduler
        fake_cache = ModuleType("mlx_lm.models.cache")
        fake_cache.make_prompt_cache = lambda model: [KVCache()]
        modules = {"omlx": fake_omlx, "omlx.scheduler": fake_scheduler,
                   "mlx_lm": ModuleType("mlx_lm"), "mlx_lm.models": ModuleType("mlx_lm.models"),
                   "mlx_lm.models.cache": fake_cache}
        with patch.dict(sys.modules, modules), patch.object(persistence, "_MANAGER", None), patch.object(persistence.importlib.metadata, "version", return_value="0.6.4"):
            manager = persistence.install(self.model, self.root / "hook-cache")
            config = SimpleNamespace(model_path=str(self.model), model_name="physical-id", paged_ssd_cache_dir="original")
            scheduler = Scheduler(object(), object(), config=config)
            self.assertEqual(config.paged_ssd_cache_dir, "original")
            self.assertEqual(scheduler.config.paged_ssd_cache_dir, str(manager.native_directory))
            self.assertTrue(manager.status()["active"])
            self.assertEqual(manager.status()["model_id"], "physical-id")
            manager._commit_index([1, 2, 3, 4])
            request = owner.request([1, 2, 3, 4, 5])
            self.assertEqual(scheduler._prepare_prefix_cache_for_request(request), "normal prepare result")
            self.assertEqual(request.remaining_tokens, [5])
            # Repeat preparation must not restore again or reset N-1 state.
            scheduler._prepare_prefix_cache_for_request(request)
            self.assertEqual(manager.restores, 1)
            with patch.object(manager, "store", return_value=True) as store:
                cache, last = scheduler._do_external_prefill(request, [1, 2, 3, 4, 5], None)
                self.assertEqual(last, [5])
                self.assertEqual(store.call_args.args[2], [1, 2, 3, 4])
                self.assertEqual(scheduler._finalize_chunked_prefill_cache_for_insert(request, cache), "normal finalize result")
                scheduler.running["request"] = request
                response = SimpleNamespace(uid=7, prompt_cache=cache, all_tokens=[1, 2, 3, 4, 5, 6])
                self.assertEqual(scheduler._process_batch_responses([response]), (["streamed output"], {"request"}))
                self.assertEqual(store.call_args.args[2], [1, 2, 3, 4, 5, 6])
            unrelated_config = SimpleNamespace(model_path=str(self.root / "other-model"), model_name="other", paged_ssd_cache_dir="other-cache")
            unrelated = Scheduler(object(), object(), unrelated_config)
            self.assertIs(unrelated.config, unrelated_config)
            self.assertEqual(unrelated.config.paged_ssd_cache_dir, "other-cache")
            unrelated_request = owner.request([1, 2, 3, 4, 5])
            unrelated._prepare_prefix_cache_for_request(unrelated_request)
            self.assertEqual(unrelated_request.cached_tokens, 0)
            unrelated.block_aware_cache.restore_exact_prefix.assert_not_called()


@unittest.skipUnless(os.environ.get("WERK_TEST_MLX_PERSISTENCE") == "1", "real MLX cache codec test is opt-in")
class NativeCodecTests(unittest.TestCase):
    def test_short_deepseek_cache_round_trips_through_ssd_after_recreation(self):
        import mlx.core as mx
        from omlx.patches.deepseek_v4 import apply_pooling_cache_support
        apply_pooling_cache_support()
        from mlx_lm.models.cache import CacheList, PoolingCache, RotatingKVCache
        from omlx.cache.paged_cache import PagedCacheManager
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
        from omlx.cache.prefix_cache import BlockAwarePrefixCache
        from omlx.scheduler import Scheduler

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "model"
            model_path.mkdir()
            manager = persistence.PrefixPersistence(model_path, root / "cache", "c" * 64)
            model = SimpleNamespace(layers=[object()])

            def native_cache():
                ssd = PagedSSDCacheManager(cache_dir=manager.native_directory, max_size_bytes=16 * 1024 * 1024,
                    expected_model_name="model", expected_num_layers=1, expected_block_size=2048)
                paged = PagedCacheManager(block_size=2048, max_blocks=16, model_name="model", initial_blocks=4)
                paged.set_paged_ssd_cache_manager(ssd)
                return BlockAwarePrefixCache(model, paged, ssd), ssd

            cache, ssd = native_cache()
            self.addCleanup(ssd.close)
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.block_aware_cache = cache
            scheduler.model = model
            scheduler._stream = mx.default_stream(mx.default_device())
            scheduler._bypass_hot_cache_under_pressure = lambda: False
            scheduler._release_paged_cache_for_request = lambda request_id: None
            manager._scheduler = weakref.ref(scheduler)
            manager.max_prefix_tokens = 2048
            rotating = RotatingKVCache(max_size=128)
            keys = mx.arange(5 * 8).reshape(1, 1, 5, 8).astype(mx.float16)
            rotating.update_and_fetch(keys, keys + 100)
            pooling = PoolingCache(4)
            raw_kv = mx.arange(5 * 8).reshape(1, 5, 8).astype(mx.float16)
            raw_gate = mx.ones((1, 5, 8), dtype=mx.float16)
            pooling.accumulate_windows(raw_kv, raw_gate, 0)
            pooling.update_and_fetch(mx.ones((1, 1, 8), dtype=mx.float16))
            pooling.prev_win_kv = raw_kv[:, :4, :][:, None]
            pooling.prev_win_gate = raw_gate[:, :4, :][:, None]
            state = [CacheList(rotating, pooling)]
            request = SimpleNamespace(request_id="short", prompt_token_ids=[1, 2, 3, 4, 5, 6])
            self.assertTrue(manager.store(scheduler, request, [1, 2, 3, 4, 5], state))
            ssd.close()
            cache2, ssd2 = native_cache()
            self.addCleanup(ssd2.close)
            scheduler.block_aware_cache = cache2
            fresh = persistence.PrefixPersistence(model_path, root / "cache", "c" * 64)
            fresh._scheduler = weakref.ref(scheduler)
            fresh.max_prefix_tokens = 2048
            request.cached_tokens = 0
            request.block_table = None
            observed_types = []
            original_supported = persistence._supported_cache_tree
            def inspect_supported(tree):
                observed_types.append([type(item).__name__ for item in tree])
                return original_supported(tree)
            with patch.object(persistence, "_supported_cache_tree", side_effect=inspect_supported):
                restored_ok = fresh.restore(scheduler, request)
            self.assertTrue(restored_ok, observed_types)
            self.assertEqual(request.remaining_tokens, [6])
            restored = request.prompt_cache[0]
            self.assertEqual(restored.caches[0].offset, 5)
            self.assertTrue(bool(mx.all(restored.caches[0].keys == rotating.keys)))
            self.assertEqual(restored.caches[1].ratio, 4)
            self.assertEqual(restored.caches[1].remainder, 1)
            self.assertTrue(bool(mx.all(restored.caches[1].prev_win_kv == pooling.prev_win_kv)))
            self.assertTrue(bool(mx.all(restored.caches[1].buf_kv[:, :1] == pooling.buf_kv[:, :1])))
            # A shared server can restore the same prefix for different active
            # chats. Native codec output must retain separate mutable ownership.
            other = SimpleNamespace(request_id="other-chat", prompt_token_ids=[1, 2, 3, 4, 5, 7],
                                    cached_tokens=0, block_table=None)
            self.assertTrue(fresh.restore(scheduler, other))
            self.assertIsNot(request.prompt_cache, other.prompt_cache)
            self.assertIsNot(restored, other.prompt_cache[0])
            self.assertIsNot(restored.caches[0], other.prompt_cache[0].caches[0])
            self.assertIsNot(restored.caches[1], other.prompt_cache[0].caches[1])
            restored.caches[0].offset += 1
            restored.caches[1].prev_win_kv = mx.zeros_like(restored.caches[1].prev_win_kv)
            self.assertEqual(other.prompt_cache[0].caches[0].offset, 5)
            self.assertTrue(bool(mx.all(other.prompt_cache[0].caches[1].prev_win_kv == pooling.prev_win_kv)))
            self.assertEqual(other.remaining_tokens, [7])


if __name__ == "__main__":
    unittest.main()
