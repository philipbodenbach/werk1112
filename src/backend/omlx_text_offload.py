"""Private text-only adapters for installed oMLX Qwen/GLM implementations.

The checkpoint's Python file is never imported. Native model mathematics and
sanitizers remain authoritative; only routed weights and PLE row storage are
replaced. Large-model acceptance additionally requires a real inference run.
"""

from contextlib import contextmanager
import copy
import importlib
import importlib.metadata
import inspect
from pathlib import Path
import weakref

try:
    from _werk_omlx_offload import Inventory, RangeReader, SharedCache, GiB, MiB, integer
    from _werk_omlx_offload_runtime import WeightAccess, array_from_tensor, streamed_experts, streamed_embedding
    from _werk_omlx_experts import ExpertManager, _ExpertMemoryGuard, _WORKSPACE_BYTES, _ALLOCATOR_CACHE_BYTES, automatic_cache_budget
except ImportError:  # Direct repository tests.
    from omlx_offload import Inventory, RangeReader, SharedCache, GiB, MiB, integer
    from omlx_offload_runtime import WeightAccess, array_from_tensor, streamed_experts, streamed_embedding
    from omlx_experts import ExpertManager, _ExpertMemoryGuard, _WORKSPACE_BYTES, _ALLOCATOR_CACHE_BYTES, automatic_cache_budget


ARCHITECTURES = {"qwen4_exp", "glm5_next"}
_MANAGER = None


def automatic_ngram_budget(*, metal_limit, available_memory, base_bytes,
                           expert_minimum, minimum_row_bytes, maximum_row_cache_bytes):
    """A device ceiling, not a reservation: growth follows actual row evictions.

    With streamed experts, at most one eighth of shared cache room goes to
    N-gram rows. The expert minimum is always protected. Without expert offload
    the native expert weights belong in base_bytes and rows may use all room.
    """
    for name, value in locals().copy().items():
        integer(value, name)
    if (not metal_limit or not available_memory or not minimum_row_bytes
            or maximum_row_cache_bytes < minimum_row_bytes):
        raise ValueError("N-gram Auto requires valid memory telemetry and row geometry")
    room = min(metal_limit * 9 // 10, max(0, available_memory - 2 * GiB))
    room -= base_bytes + _WORKSPACE_BYTES + _ALLOCATOR_CACHE_BYTES
    if room < expert_minimum + minimum_row_bytes:
        raise ValueError("not enough memory for base weights, workspace, one N-gram row and one expert")
    share = max(minimum_row_bytes, room // 8) if expert_minimum else room
    return min(maximum_row_cache_bytes, room - expert_minimum, share)


def native_classes(config, model_path):
    """Resolve installed classes and validate arguments without making a model."""
    architecture = config.get("model_type")
    if architecture not in ARCHITECTURES:
        raise ValueError("no native text offload adapter for this architecture")
    if importlib.metadata.version("omlx") != "0.6.4":
        raise ValueError("Qwen/GLM text offload requires oMLX 0.6.4")
    patch = importlib.import_module(f"omlx.patches.mlx_vlm_{architecture}_compat")
    getattr(patch, f"apply_mlx_vlm_{architecture}_compat_patch")()
    module = importlib.import_module(f"mlx_vlm.models.{architecture}")
    for cls in (module.Model, module.ModelConfig):
        origin = importlib.import_module(cls.__module__).__file__
        if not origin or Path(origin).resolve().is_relative_to(Path(model_path).resolve()):
            raise ValueError("model repository classes are not an installed text adapter")
    values = copy.deepcopy(config)
    if architecture == "glm5_next":
        config_module = importlib.import_module("mlx_vlm.models.glm5_next.config")
        values["text_config"] = config_module.TextConfig.from_dict(values["text_config"])
        values["vision_config"] = config_module.VisionConfig.from_dict(values.get("vision_config"))
    args = module.ModelConfig.from_dict(values)
    return module.Model, args


class TextCheckpoint:
    def __init__(self, path, expert_bytes, ngram_bytes=None):
        self.inventory = Inventory(path)
        inv = self.inventory
        self.path, self.config = inv.path, inv.config
        self.fingerprint = inv.fingerprint
        self.layers, self.experts = inv.layers, inv.experts
        self.hidden, self.intermediate, self.top_k = inv.hidden, inv.intermediate, inv.top_k
        self.expert_bytes = dict(inv.layer_bytes)
        self.total_expert_bytes = sum(inv.layer_bytes.values()) * inv.experts
        if expert_bytes is not None and (type(expert_bytes) is not int or expert_bytes < 0):
            raise ValueError("expert cache requires nonnegative bytes; zero means Auto internally")
        self.experts_enabled = expert_bytes is not None
        self.cache_budget_mode = "disabled" if expert_bytes is None else "auto" if expert_bytes == 0 else "explicit"
        self.cache_bytes = (expert_bytes or self.total_expert_bytes) if self.experts_enabled else 0
        if self.experts_enabled and self.cache_bytes < max(self.expert_bytes.values()):
            raise ValueError("expert cache must fit one complete routed expert")
        self.ngram_storage_bytes = sum(t.size for t in inv.categories["ple"].values())
        if ngram_bytes is not None and (type(ngram_bytes) is not int or ngram_bytes < 0):
            raise ValueError("N-gram cache budget must be nonnegative bytes")
        if ngram_bytes and not inv.ple_tables:
            raise ValueError("this checkpoint has no supported N-gram tables")
        self.ngrams_enabled = bool(inv.ple_tables) and ngram_bytes != 0
        self.ngram_budget_mode = "disabled" if not self.ngrams_enabled else "auto" if ngram_bytes is None else "explicit"
        rows = [(end - start, 2 * sum(
            inv.tensors[prefix + "." + suffix].size // inv.tensors[prefix + "." + suffix].shape[0]
            for suffix in ("weight", "scales", "biases")) + 4 * dims)
            for dims, specs in inv.ple_tables.values() for start, end, prefix, _ in specs]
        self.largest_ngram_row_bytes = max((size for _, size in rows), default=0)
        # Cache accounting includes decoded rows and conversion staging, which
        # may exceed the packed checkpoint's byte count.
        self.maximum_ngram_cache_bytes = sum(count * size for count, size in rows)
        self.ngram_budget_bytes = (min(ngram_bytes or self.maximum_ngram_cache_bytes,
                                      self.maximum_ngram_cache_bytes) if self.ngrams_enabled else 0)
        if self.ngrams_enabled and self.ngram_budget_bytes < self.largest_ngram_row_bytes:
            raise ValueError("N-gram cache must fit one complete row including dequantization")
        self.ngram_initial_cache_bytes = (min(self.ngram_budget_bytes, max(64 * MiB, self.largest_ngram_row_bytes))
                                          if self.ngram_budget_mode == "auto" else self.ngram_budget_bytes)
        self.dense_bytes = sum(t.size for t in inv.categories["base"].values())
        self.base_bytes = (self.dense_bytes + self.ngram_initial_cache_bytes
                           + (0 if self.ngrams_enabled else self.ngram_storage_bytes)
                           + (0 if self.experts_enabled else self.total_expert_bytes))
        self.resident_estimate_bytes = self.base_bytes + min(self.cache_bytes, self.total_expert_bytes) + _WORKSPACE_BYTES
        self.projections = {key: q for key, (_, q) in inv.projections.items()}

    def summary(self):
        return {
            "architecture": self.inventory.architecture, "fingerprint": self.fingerprint,
            "base_bytes": self.dense_bytes, "expert_bytes": self.total_expert_bytes,
            "checkpoint_bytes": self.inventory.summary()["checkpoint_bytes"],
            "cache_budget_bytes": self.cache_bytes, "cache_budget_mode": self.cache_budget_mode,
            "workspace_bytes": _WORKSPACE_BYTES, "resident_estimate_bytes": self.resident_estimate_bytes,
            "expert_count": len(self.expert_bytes) * self.experts,
            "largest_expert_bytes": max(self.expert_bytes.values()),
            "ngram_storage_bytes": self.ngram_storage_bytes,
            "ngram_maximum_cache_bytes": self.maximum_ngram_cache_bytes,
            "ngram_cache_budget_bytes": self.ngram_budget_bytes,
            "ngram_cache_budget_mode": self.ngram_budget_mode,
            "ngram_initial_cache_bytes": self.ngram_initial_cache_bytes,
            "experts_offloaded": self.experts_enabled,
            "ngram_offload": "supported" if self.ngrams_enabled else "disabled" if self.inventory.ple_tables else "not_applicable",
            "loader": "installed_native_text_port", "mtp_enabled": False,
        }

    def configure_auto(self):
        if self.cache_budget_mode == "auto" or self.ngram_budget_mode == "auto":
            from omlx.process_memory_enforcer import get_effective_metal_cap_bytes
            import psutil
            metal_limit = get_effective_metal_cap_bytes()
            available_memory = psutil.virtual_memory().available
            if self.ngram_budget_mode == "auto":
                self.ngram_budget_bytes = automatic_ngram_budget(
                    metal_limit=metal_limit, available_memory=available_memory,
                    base_bytes=self.base_bytes - self.ngram_initial_cache_bytes,
                    expert_minimum=max(self.expert_bytes.values()) if self.experts_enabled else 0,
                    minimum_row_bytes=self.largest_ngram_row_bytes,
                    maximum_row_cache_bytes=self.maximum_ngram_cache_bytes)
                initial = min(self.ngram_initial_cache_bytes, self.ngram_budget_bytes)
                self.base_bytes += initial - self.ngram_initial_cache_bytes
                self.ngram_initial_cache_bytes = initial
        if self.cache_budget_mode == "auto":
            self.cache_bytes = automatic_cache_budget(
                self, metal_limit=metal_limit, available_memory=available_memory)
        self.resident_estimate_bytes = self.base_bytes + min(self.cache_bytes, self.total_expert_bytes) + _WORKSPACE_BYTES


class TextExpertManager(ExpertManager):
    def __init__(self, checkpoint, execution=None):
        super().__init__(checkpoint, execution=execution)
        self.reader = RangeReader()
        self.ngram_cache = SharedCache(max(1, checkpoint.ngram_initial_cache_bytes))
        self._ngram_auto_target = checkpoint.ngram_initial_cache_bytes
        self._ngram_last_evictions = 0
        self.access = WeightAccess(checkpoint.inventory, self.ngram_cache, self.reader)
        # The expert protocol and the forward pass share exactly this cache.
        self.access.expert = self.expert
        self.access.expert_groups = lambda layer, indices: (
            [index for _, index in group] for group in self._groups(layer, indices))
        self.access.prefetch_experts = self.prefetch_experts

    @contextmanager
    def prefetch_experts(self, layer, indices):
        with self._lock:
            keys = {(layer, int(index)) for index in indices}
            prior_leases = set(self._leased)
            self._leased.update(keys)
            try:
                tensors = [self.checkpoint.inventory.tensors[
                    self._projection_prefix(layer, projection) + '.' + suffix].rows(index)
                    for _, index in sorted(keys) if (layer, index) not in self._cache
                    for projection in ('gate_proj', 'up_proj', 'down_proj')
                    for suffix in ('weight', 'scales', 'biases')]
                # Raw read-ahead plus independent row copies fit in the
                # existing workspace, including small expert-cache budgets.
                with self.reader.prefetch(tensors, max_bytes=_WORKSPACE_BYTES // 8):
                    yield
            finally:
                self._leased.difference_update(keys - prior_leases)

    def _projection_prefix(self, layer, projection):
        return self.checkpoint.inventory.projections[layer, projection][0]

    def _read(self, name, expert=None, pending=None):
        tensor = self.checkpoint.inventory.tensors[name]
        if expert is not None:
            tensor = tensor.rows(expert)
        value = array_from_tensor(self.reader, tensor, evaluate=pending is None)
        if pending is not None:
            pending.append((value, None))
        self.disk_bytes_read = self.reader.logical_bytes
        return value[0] if expert is not None else value

    @contextmanager
    def expert(self, layer, index):
        key = self._key(f"layer.{layer}.expert.{index}")
        with self._lock:
            self._leased.add(key)
            try:
                yield self._acquire(key)
            finally:
                self._leased.discard(key)
                self._trim_allocator()

    def status(self):
        result = super().status()
        result["ngram_cache"] = self.ngram_cache.snapshot()
        result["ngram_requested_rows"] = self.access.ple_requested_rows
        result["ngram_unique_rows"] = self.access.ple_unique_rows
        result["ngram_resident_cache_bytes"] = self.ngram_cache.used
        result["ngram_effective_cache_budget_bytes"] = self.ngram_cache.budget if self.checkpoint.ngrams_enabled else 0
        for key in ("hits", "misses", "evictions"):
            result["ngram_cache_" + key] = result["ngram_cache"]["namespaces"].get("ple", {}).get(key, 0)
        result["forward_calls"] = self.access.forward_calls
        result["forward_seconds"] = self.access.forward_seconds
        result["routing_seconds"] = self.access.routing_seconds
        result["output_evaluations"] = self.access.output_evaluations
        result["disk_read_seconds"] = self.reader.read_seconds
        result["disk_bytes_read"] = self.reader.logical_bytes
        result["last_prefill_admission"] = getattr(self, "last_prefill_admission", None)
        return result

    def auxiliary_cache_usage(self):
        if not self.checkpoint.ngram_budget_bytes:
            return 0, 0
        return self.ngram_cache.used, self.ngram_cache.budget

    def _resize_cache(self, budget):
        if self.checkpoint.experts_enabled:
            super()._resize_cache(budget)

    def list_experts(self, filters=None):
        if not self.checkpoint.experts_enabled:
            raise ValueError("expert offload is disabled for this worker")
        return super().list_experts(filters)

    def action(self, request):
        if not self.checkpoint.experts_enabled:
            raise ValueError("expert offload is disabled for this worker")
        return super().action(request)

    def prefill_transient_reserve(self, n_tokens, native_peak):
        # Native endpoint samples miss intermediate peaks in these hybrid
        # text models. Keep another predicted transient outside the weight
        # caches, with room for two FP32 vocabulary-sized buffers at minimum.
        # This is reclaimable cache capacity, not an allocation or guard bypass.
        # A 34-tool Qwen request otherwise exceeded the native hard watermark
        # mid-prefill despite passing the chunk-boundary checks.
        vocab = self.checkpoint.config["text_config"]["vocab_size"]
        return max(int(native_peak), 8 * n_tokens * vocab)

    def resize_auxiliary_cache(self, available):
        cp = self.checkpoint
        if not cp.ngram_budget_bytes:
            return 0
        # Leave one demand expert in addition to any pins. One PLE row must
        # also fit; the native guard rejects requests below that minimum.
        protected = sum(cp.expert_bytes[layer] for layer, _ in self._pinned | self._leased)
        expert_minimum = protected + max(cp.expert_bytes.values()) if cp.experts_enabled else 0
        largest_row = cp.largest_ngram_row_bytes
        # A requested chunk may not fit yet. Preserve the minimal working set
        # and let the following native adaptive/guard calls shrink or reject
        # the chunk; raising here would prevent that native fallback entirely.
        target = cp.ngram_budget_bytes
        if cp.ngram_budget_mode == "auto":
            with self.ngram_cache.lock:
                evictions = self.ngram_cache.stats.get("ple", {}).get("evictions", 0)
                if (evictions > self._ngram_last_evictions
                        and self.ngram_cache.used * 4 >= self.ngram_cache.budget * 3):
                    self._ngram_auto_target = min(cp.ngram_budget_bytes, 2 * self._ngram_auto_target)
                self._ngram_last_evictions = evictions
            target = self._ngram_auto_target
        limit = max(largest_row, min(target, int(available) - expert_minimum))
        self.ngram_cache.resize(limit)
        if cp.ngram_budget_mode == "auto":
            # Reclamation is not demand: do not grow because pressure itself
            # evicted rows at the previous boundary.
            self._ngram_last_evictions = self.ngram_cache.stats.get("ple", {}).get("evictions", 0)
        return limit

    def deactivate(self):
        super().deactivate()
        self.reader.close()
        if hasattr(self, "ngram_cache"):
            with self.ngram_cache.lock:
                self.ngram_cache.entries.clear()
                self.ngram_cache.used = 0
                self.ngram_cache.stats.clear()


def _canonical_name(name, architecture=None):
    # Native GLM nests the forget-gate projections. Its sanitizer remaps only
    # their weight, so affine metadata and per-module quantization must follow
    # the same namespace before sanitizing or constructing quantized modules.
    if architecture == "glm5_next":
        for projection in ("f_a_proj", "f_b_proj"):
            source = ".self_attn." + projection
            if name.endswith(source) or source + "." in name:
                name = name.replace(source, ".self_attn.forget_gate." + projection)
    if name.startswith("language_model."):
        return name
    if name == "lm_head" or name.startswith(("model.", "lm_head.")):
        return "language_model." + name
    raise ValueError(f"unverified native text tensor name: {name}")


def load_text_model(manager, tokenizer_config=None, **kwargs):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import utils

    if set(kwargs) - {"lazy", "return_config", "trust_remote_code"}:
        raise ValueError("native text offload does not accept adapters or model overrides")
    cp = manager.checkpoint
    current = Inventory(cp.path)
    if current.fingerprint != cp.fingerprint:
        raise ValueError("checkpoint changed after offload preflight")
    if manager.status()["active"]:
        raise ValueError("offloaded text model is already loaded")
    model_class, args = native_classes(cp.config, cp.path)
    if current.architecture == "qwen4_exp":
        language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
        # Constructor arrays stay lazy and are replaced before any evaluation.
        language.configure_ple_runtime(cp.path, mode="resident")
        language.configure_mtp_runtime(cp.path, enabled=False)
    root = model_class(args)
    try:
        for attribute in ("vision_tower", "vision_model"):
            if attribute in root:
                # Module.__setattr__ removes the registered child and stores
                # an ordinary None attribute. A dictionary None is invisible
                # to MLX attribute lookup and breaks native GLM sanitize.
                setattr(root, attribute, None)
        for layer in sorted(cp.expert_bytes) if cp.experts_enabled else ():
            block = root.language_model.model.layers[layer]
            mlp = block.mlp
            activation = mlp.switch_mlp.activation
            mlp.switch_mlp = streamed_experts(manager.access, layer, activation, _WORKSPACE_BYTES)
            if hasattr(block, "compile_ffn"):
                # Python routing and bounded disk I/O cannot be traced into an
                # MLX compiled graph. The arithmetic itself stays native.
                block.compile_ffn = False
        for layer in current.ple_tables if cp.ngrams_enabled else ():
            root.language_model.model.layers[layer].ple.ple_embedding.ngram_embedding = streamed_embedding(manager.access, layer)
        selected = dict(current.categories["base"])
        if not cp.experts_enabled:
            selected.update(current.categories["experts"])
        if not cp.ngrams_enabled:
            selected.update(current.categories["ple"])
        weights = {_canonical_name(name, current.architecture): manager._read(name) for name in selected}
        # Retain installed normalization/layout logic, including Qwen's
        # direct-gamma checkpoint conversion and GLM's FP32 routing parameters.
        weights = root.sanitize(weights)
        # Quantize only projections whose tensors are actually being loaded.
        # Vision/MTP overrides may use unrelated namespaces and must not leak
        # back into the text loader after those weights have been excluded.
        source_quant = {_canonical_name(name.removesuffix(".scales"), current.architecture):
                        current.quantization(name.removesuffix(".scales"))
                        for name in selected if name.endswith(".scales")}
        default_quant = current.quantization("__default__")

        def quantized(path, module):
            if not hasattr(module, "to_quantized") or path + ".scales" not in weights:
                return False
            return source_quant.get(path, default_quant)

        nn.quantize(root, **default_quant, class_predicate=quantized)
        root.eval()
        # Avoid native whole-table concatenation when resident PLE was selected;
        # its additional full copy is not part of the bounded loader contract.
        nn.Module.load_weights(root, list(weights.items()), strict=True)
        mx.eval(root.parameters())
        del weights
        mx.clear_cache()

        class TextModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.core = root
                self.model_type = current.architecture
                self.args = root.language_model.args

            @property
            def layers(self):
                return self.core.language_model.layers

            def make_cache(self):
                return self.core.language_model.make_cache()

            def __call__(self, inputs, cache=None, input_embeddings=None):
                return self.core.language_model(inputs, cache=cache, inputs_embeds=input_embeddings).logits

        model = TextModel()
        tokenizer = utils.load_tokenizer(cp.path, tokenizer_config,
                                         eos_token_ids=cp.config.get("eos_token_id", cp.config["text_config"].get("eos_token_id")))
        manager._model_ref = weakref.ref(model)
        weakref.finalize(model, manager.deactivate)
        return (model, tokenizer, cp.config) if kwargs.get("return_config") else (model, tokenizer)
    except BaseException:
        manager.deactivate()
        raise


def inspect_model(path, expert_bytes, ngram_bytes=None):
    checkpoint = TextCheckpoint(path, expert_bytes, ngram_bytes)
    native_classes(checkpoint.config, checkpoint.path)
    return checkpoint.summary()


def install(path, expert_bytes, ngram_bytes=None):
    """Bind exact-path text loading, discovery and native memory admission."""
    global _MANAGER
    if _MANAGER is not None:
        raise ValueError("native text offload is already installed")
    checkpoint = TextCheckpoint(path, expert_bytes, ngram_bytes)
    native_classes(checkpoint.config, checkpoint.path)
    checkpoint.configure_auto()
    from omlx.utils import model_loading
    from omlx import model_discovery
    from omlx.engine_pool import EnginePool
    from omlx.scheduler import Scheduler
    from omlx.engine.batched import BatchedEngine
    original_load = model_loading.lm_load_compat
    original_size = EnginePool._entry_runtime_resident_size
    original_detect = model_discovery.detect_model_type
    if tuple(inspect.signature(original_size).parameters) != ("self", "entry", "runtime_settings", "base_size"):
        raise ValueError("unsupported oMLX memory admission contract")
    manager = TextExpertManager(checkpoint)
    guard = _ExpertMemoryGuard(manager, Scheduler, BatchedEngine)

    def matches(value):
        return Path(value).resolve() == checkpoint.path

    def load(path_or_repo, **kwargs):
        if matches(path_or_repo):
            return load_text_model(manager, **kwargs)
        return original_load(path_or_repo, **kwargs)

    def detect(model_path):
        # This private worker exposes only the verified text adapter for its
        # exact checkpoint. Other model discoveries retain the native result.
        return "llm" if matches(model_path) else original_detect(model_path)

    def resident_size(pool, entry, runtime_settings, *, base_size=None):
        if not matches(entry.model_path):
            return original_size(pool, entry, runtime_settings, base_size=base_size)
        if pool._distributed_deployment_for_entry(entry) is not None:
            raise ValueError("native text offload cannot use distributed loading")
        manager.model_id = entry.model_id
        return checkpoint.resident_estimate_bytes

    model_loading.lm_load_compat = load
    model_discovery.detect_model_type = detect
    EnginePool._entry_runtime_resident_size = resident_size
    guard.install(Scheduler, BatchedEngine)
    _MANAGER = manager
    return manager
