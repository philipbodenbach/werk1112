"""Optional exact-prefix KV persistence inside a private Werk oMLX worker.

oMLX retains its scheduler, sampler and streaming protocol. This extension uses
its native typed SSD cache to retain short exact prefixes that ordinary paged
matching omits. Only token counts and digests are recorded in the side index.
"""

import copy
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import re
import struct
import tempfile
import threading
import time
import uuid
import weakref


FORMAT = "omlx-exact-prefix-v1"
_MAX_ENTRIES = 128
_MAX_INDEX_BYTES = 65536
_MAX_EXACT_TOKENS = 8192
_COMMIT_TIMEOUT = 5.0
_ALLOWED_CACHE_TYPES = {"KVCache", "RotatingKVCache", "PrefillReadyRotatingKVCache", "PoolingCache", "CacheList"}
_LOG = logging.getLogger("werk.omlx.persistence")
_MANAGER = None


def _token_digest(tokens):
    if not isinstance(tokens, (list, tuple)) or any(type(token) is not int or not 0 <= token < 2**32 for token in tokens):
        raise ValueError("invalid exact-prefix token IDs")
    digest = hashlib.sha256(b"werk-omlx-prefix-v1\0")
    for token in tokens:
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def _fingerprint(model_path, runtime_versions):
    """Bind native KV to the checkpoint, tokenizer and installed runtime."""
    root = Path(model_path).resolve(strict=True)
    digest = hashlib.sha256(FORMAT.encode())
    digest.update(json.dumps(runtime_versions, sort_keys=True).encode())
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            stat = path.stat()
            digest.update(path.name.encode())
            digest.update(repr((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)).encode())
        elif path.name in {"config.json", "tokenizer.json", "tokenizer_config.json",
                           "special_tokens_map.json", "tokenizer.model", "vocab.json", "merges.txt"}:
            digest.update(path.name.encode())
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def _supported_cache_tree(cache):
    if not isinstance(cache, (list, tuple)) or not cache:
        return False
    for layer in cache:
        if type(layer).__name__ not in _ALLOWED_CACHE_TYPES:
            return False
        if type(layer).__name__ == "CacheList" and not _supported_cache_tree(layer.caches):
            return False
    return True


def _eligible_request(request):
    # Visual embeddings and speculative sparsification are not represented by
    # token IDs alone. Normal oMLX continues handling those requests.
    return (getattr(request, "specprefill_indices", None) is None
            and getattr(request, "vlm_inputs_embeds", None) is None
            and not getattr(request, "vlm_extra_keys_for_cache", None)
            and not getattr(request, "vlm_extra_key_ranges_for_cache", None))


class PrefixPersistence:
    def __init__(self, model_path, cache_directory, fingerprint):
        self.model_path = Path(model_path).resolve(strict=True)
        self.model_id = self.model_path.name
        self.fingerprint = fingerprint
        self.directory = Path(cache_directory).resolve() / fingerprint
        self.native_directory = self.directory / "native"
        self.index_path = self.directory / "werk-prefixes.json"
        self._scheduler = None
        self._lock = threading.RLock()
        self._entries = []
        self.stores = self.restores = self.restored_tokens = self.failures = 0
        self.max_prefix_tokens = 0
        self.reason = "model scheduler is not loaded"
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink() or self.native_directory.is_symlink():
            raise ValueError("native persistence directories must not be symlinks")
        self.native_directory.mkdir(exist_ok=True, mode=0o700)
        self._load_index()

    def matches(self, config):
        path = getattr(config, "model_path", None)
        return bool(path) and Path(path).resolve() == self.model_path

    def attach(self, scheduler):
        if not self.matches(scheduler.config):
            return
        self._scheduler = None
        self.max_prefix_tokens = 0
        self.model_id = scheduler.config.model_name or self.model_path.name
        cache = getattr(scheduler, "block_aware_cache", None)
        try:
            from mlx_lm.models.cache import make_prompt_cache
            tree = make_prompt_cache(scheduler.model)
            if not _supported_cache_tree(tree):
                self.reason = "model cache type is not verified for native exact-prefix persistence"
                return
        except Exception:
            self.reason = "model cache type could not be verified"
            return
        if cache is None or getattr(cache, "paged_ssd_cache", None) is None:
            self.reason = "native SSD prefix cache is unavailable"
            return
        if not callable(getattr(cache, "store_exact_prefix", None)) or not callable(getattr(cache, "restore_exact_prefix", None)):
            self.reason = "native exact-prefix cache API is unavailable"
            return
        self.max_prefix_tokens = min(int(cache.block_size), _MAX_EXACT_TOKENS)
        self._scheduler = weakref.ref(scheduler)
        self.reason = None

    def status(self):
        with self._lock:
            active = self._scheduler is not None and self._scheduler() is not None
            return {"installed": True, "active": active, "format": FORMAT,
                    "model_id": self.model_id, "max_prefix_tokens": self.max_prefix_tokens,
                    "stores": self.stores, "restores": self.restores,
                    "restored_tokens": self.restored_tokens, "failures": self.failures,
                    "indexed_prefixes": len(self._entries), "reason": self.reason}

    def _load_index(self):
        try:
            if self.index_path.is_symlink():
                return
            with self.index_path.open("rb") as source:
                raw = source.read(_MAX_INDEX_BYTES + 1)
            if len(raw) > _MAX_INDEX_BYTES:
                return
            value = json.loads(raw)
            if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("fingerprint") != self.fingerprint:
                return
            entries = value.get("entries")
            if not isinstance(entries, list) or len(entries) > _MAX_ENTRIES:
                return
            result = []
            for row in entries:
                if not isinstance(row, dict) or set(row) != {"tokens", "digest"}:
                    return
                if type(row["tokens"]) is not int or not 1 <= row["tokens"] <= _MAX_EXACT_TOKENS:
                    return
                if not isinstance(row["digest"], str) or re.fullmatch("[0-9a-f]{64}", row["digest"]) is None:
                    return
                if row not in result:
                    result.append(row)
            self._entries = result
        except (OSError, ValueError, TypeError):
            # Missing/corrupt cache metadata costs recomputation, never a wrong
            # prefix or a failed conversation restore.
            return

    def _commit_index(self, tokens):
        entry = {"tokens": len(tokens), "digest": _token_digest(tokens)}
        with self._lock:
            entries = [entry] + [row for row in self._entries if row != entry]
            entries = entries[:_MAX_ENTRIES]
            value = {"format": FORMAT, "fingerprint": self.fingerprint, "entries": entries}
            descriptor, temporary = tempfile.mkstemp(prefix=".werk-prefixes-", dir=self.directory)
            try:
                with os.fdopen(descriptor, "w") as output:
                    json.dump(value, output, separators=(",", ":"))
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.index_path)
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                self._entries = entries
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def _is_target(self, scheduler):
        return self._scheduler is not None and self._scheduler() is scheduler

    def _safe_request(self, scheduler, request):
        return (self._is_target(scheduler) and _eligible_request(request)
                and getattr(scheduler, "_vlm_mtp_drafter", None) is None
                and getattr(scheduler, "_specprefill_draft_model", None) is None)

    def _durable(self, prefix_cache, tokens):
        ssd = prefix_cache.paged_ssd_cache
        hashes = [item[0] for item in prefix_cache._exact_prefix_hashes(tokens)]
        deadline = time.monotonic() + _COMMIT_TIMEOUT
        while True:
            with ssd._pending_write_hashes_lock:
                pending = any(block_hash in ssd._pending_write_hashes for block_hash in hashes)
            if not pending:
                for block_hash in hashes:
                    metadata = ssd._index.get(block_hash)
                    if metadata is None or not metadata.file_path.is_file():
                        return False
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def store(self, scheduler, request, tokens, cache):
        if not self._safe_request(scheduler, request) or not tokens or not cache:
            return False
        if len(tokens) > self.max_prefix_tokens or not _supported_cache_tree(cache):
            return False
        try:
            _token_digest(tokens)
            from omlx.scheduler import _safe_sync_stream
            import mlx.core as mx
            _safe_sync_stream(scheduler._stream)
            with mx.stream(scheduler._stream):
                extracted, model_config = scheduler._extract_cache_states(cache)
                if not extracted:
                    return False
                table = scheduler.block_aware_cache.store_exact_prefix(
                    "werk-prefix-store-" + uuid.uuid4().hex, list(tokens), extracted,
                    model_cache_config=model_config,
                )
            if table is None or table.num_tokens != len(tokens) or not self._durable(scheduler.block_aware_cache, tokens):
                self.failures += 1
                return False
            self._commit_index(tokens)
            self.stores += 1
            return True
        except Exception:
            self.failures += 1
            _LOG.debug("Exact-prefix cache store failed; ordinary inference continues")
            return False

    def restore(self, scheduler, request):
        if not self._safe_request(scheduler, request):
            return False
        tokens = request.prompt_token_ids
        if not isinstance(tokens, list) or len(tokens) < 2:
            return False
        with self._lock:
            candidates = sorted(self._entries, key=lambda row: row["tokens"], reverse=True)
        for entry in candidates:
            length = entry["tokens"]
            # Always leave a real prompt token for generation kickoff. Never
            # trim Pooling/rotating state from N to N-1 or replay token N.
            if not 0 < length < len(tokens) or length <= getattr(request, "cached_tokens", 0):
                continue
            if length > self.max_prefix_tokens or _token_digest(tokens[:length]) != entry["digest"]:
                continue
            try:
                cache = scheduler.block_aware_cache.restore_exact_prefix(
                    "werk-prefix-restore-" + uuid.uuid4().hex, tokens[:length],
                    promote_to_hot_cache=not scheduler._bypass_hot_cache_under_pressure(),
                )
                if cache is None or not _supported_cache_tree(cache):
                    continue
                # Hooked after ordinary preparation, so release its shorter
                # shared block references before replacing the cache state.
                if getattr(request, "block_table", None) is not None:
                    scheduler._release_paged_cache_for_request(request.request_id)
                request.prompt_cache = cache
                request.block_table = None
                request.shared_prefix_blocks = 0
                request.cached_tokens = length
                request.remaining_tokens = tokens[length:]
                self.restores += 1
                self.restored_tokens += length
                return True
            except Exception:
                self.failures += 1
                _LOG.debug("Exact-prefix cache restore missed; ordinary prefill continues")
        return False


def install(model_path, cache_directory):
    """Patch only the selected model in this explicitly persistent worker."""
    global _MANAGER
    if _MANAGER is not None:
        raise ValueError("native prefix persistence is already installed")
    versions = {name: importlib.metadata.version(name) for name in ("omlx", "mlx", "mlx-lm")}
    if versions["omlx"] != "0.6.4":
        raise ValueError("native prefix persistence currently requires oMLX 0.6.4")
    from omlx.scheduler import Scheduler
    original_init = Scheduler.__init__
    original_prepare = Scheduler._prepare_prefix_cache_for_request
    original_prefill = Scheduler._do_external_prefill
    original_finalize = Scheduler._finalize_chunked_prefill_cache_for_insert
    original_responses = Scheduler._process_batch_responses
    signatures = ((original_init, ("self", "model", "tokenizer", "config", "stream")),
                  (original_prepare, ("self", "request")),
                  (original_prefill, ("self", "request", "tokens", "existing_cache", "vlm_embeds")),
                  (original_finalize, ("self", "request", "prompt_cache")),
                  (original_responses, ("self", "responses")))
    if any(tuple(inspect.signature(function).parameters) != expected for function, expected in signatures):
        raise ValueError("unsupported oMLX scheduler API for native prefix persistence")
    manager = PrefixPersistence(model_path, cache_directory, _fingerprint(model_path, versions))

    def initialize(scheduler, model, tokenizer, config=None, stream=None):
        if config is not None and manager.matches(config):
            config = copy.copy(config)
            config.paged_ssd_cache_dir = str(manager.native_directory)
            config.hot_cache_write_through = True
        original_init(scheduler, model, tokenizer, config, stream)
        manager.attach(scheduler)

    def prepare(scheduler, request):
        already_prepared = request.request_id in scheduler._prefix_cache_prepared
        result = original_prepare(scheduler, request)
        if not already_prepared:
            manager.restore(scheduler, request)
        return result

    def prefill(scheduler, request, tokens, existing_cache, vlm_embeds=None):
        cache, last_token = original_prefill(scheduler, request, tokens, existing_cache, vlm_embeds)
        if vlm_embeds is None and len(last_token) == 1:
            manager.store(scheduler, request, request.prompt_token_ids[:-1], cache)
        return cache, last_token

    def finalize(scheduler, request, prompt_cache):
        result = original_finalize(scheduler, request, prompt_cache)
        manager.store(scheduler, request, request.prompt_token_ids[:-1], prompt_cache)
        return result

    def responses(scheduler, response_list):
        # Capture before the normal finalizer clears request ownership. Native
        # BatchGenerator.all_tokens explicitly identifies tokens in this KV.
        for response in response_list:
            raw_cache = getattr(response, "prompt_cache", None)
            all_tokens = getattr(response, "all_tokens", None)
            request_id = scheduler.uid_to_request_id.get(response.uid)
            request = scheduler.running.get(request_id)
            if request is not None and raw_cache is not None and all_tokens is not None:
                manager.store(scheduler, request, list(all_tokens), raw_cache)
        return original_responses(scheduler, response_list)

    Scheduler.__init__ = initialize
    Scheduler._prepare_prefix_cache_for_request = prepare
    Scheduler._do_external_prefill = prefill
    Scheduler._finalize_chunked_prefill_cache_for_insert = finalize
    Scheduler._process_batch_responses = responses
    _MANAGER = manager
    return manager
