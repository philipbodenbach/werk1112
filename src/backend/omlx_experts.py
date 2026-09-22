"""Private, opt-in oMLX worker support for streaming packed DeepSeek V4 experts.

The source is embedded in Werk, never installed over the user's runtime. Header
inspection uses only the standard library. The execution path reads individual
expert byte ranges and keeps a bounded cache; no full expert tensor is loaded.
"""

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import struct
import threading
import time
import weakref


_EXPERT_KEY = re.compile(
    r"^model\.layers\.(\d+)\.ffn\.switch_mlp\."
    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
_DTYPES = {"U32": 4, "I32": 4, "F32": 4, "F16": 2, "BF16": 2,
           "I64": 8, "U64": 8, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1}
_WORKSPACE_BYTES = 1024 * 1024 * 1024
_ALLOCATOR_CACHE_BYTES = 64 * 1024 * 1024
_ADMISSION_HEADROOM_BYTES = 64 * 1024 * 1024
_EXPERT_GROUP_SIZE = 8
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_MANAGER = None


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class Checkpoint:
    def __init__(self, model_path, cache_bytes):
        self.path = Path(model_path).resolve(strict=True)
        self.cache_budget_mode = "auto" if type(cache_bytes) is int and cache_bytes == 0 else "explicit"
        self.cache_bytes = 0 if self.cache_budget_mode == "auto" else _positive_int(cache_bytes, "expert cache bytes")
        with (self.path / "config.json").open("rb") as config_file:
            config_data = config_file.read(4 * 1024 * 1024 + 1)
        if len(config_data) > 4 * 1024 * 1024:
            raise ValueError("expert streaming config.json exceeds 4 MiB")
        self.config = json.loads(config_data, object_pairs_hook=_unique_object)
        c = self.config
        if not isinstance(c, dict):
            raise ValueError("expert streaming config.json must contain an object")
        if c.get("model_type") != "deepseek_v4" or c.get("model_file"):
            raise ValueError("expert streaming requires a native deepseek_v4 checkpoint")
        self.layers = _positive_int(c.get("num_hidden_layers"), "num_hidden_layers")
        self.experts = _positive_int(c.get("n_routed_experts"), "n_routed_experts")
        self.hidden = _positive_int(c.get("hidden_size"), "hidden_size")
        self.intermediate = _positive_int(c.get("moe_intermediate_size"), "moe_intermediate_size")
        self.top_k = _positive_int(c.get("num_experts_per_tok"), "num_experts_per_tok")
        if self.layers > 256 or self.experts > 4096 or self.layers * self.experts > 262144:
            raise ValueError("expert checkpoint geometry exceeds supported limits")
        if self.top_k > self.experts or c.get("quantize_activations", False):
            raise ValueError("unsupported expert routing or activation quantization")
        quant = c.get("quantization")
        if not isinstance(quant, dict) or quant.get("mode", "affine") != "affine":
            raise ValueError("expert streaming currently requires affine MLX quantization")
        custom_quant = c.get("quantization_config")
        if custom_quant is not None and (not isinstance(custom_quant, dict) or custom_quant.get("quant_method")):
            raise ValueError("expert streaming cannot use a custom quantization loader")
        if c.get("dspark_block_size") or c.get("n_mtp_layers") or c.get("dspark_target_layer_ids"):
            raise ValueError("expert streaming does not support embedded speculative drafters")
        self.tensors = {}
        self.files = {}
        fingerprint = hashlib.sha256(config_data)
        files = sorted(self.path.glob("model*.safetensors"))
        if not files or len(files) > 1024:
            raise ValueError("expert streaming requires local model*.safetensors shards")
        for path in files:
            with path.open("rb") as stream:
                signature = _signature(os.fstat(stream.fileno()))
                size = signature[2]
                raw_length = stream.read(8)
                if len(raw_length) != 8:
                    raise ValueError(f"truncated safetensors header: {path.name}")
                length = struct.unpack("<Q", raw_length)[0]
                if not 2 <= length <= min(_MAX_HEADER_BYTES, size - 8):
                    raise ValueError(f"invalid safetensors header length: {path.name}")
                raw_header = stream.read(length)
            header = json.loads(raw_header, object_pairs_hook=_unique_object)
            if not isinstance(header, dict):
                raise ValueError(f"invalid safetensors header: {path.name}")
            fingerprint.update(path.name.encode())
            fingerprint.update(raw_header)
            fingerprint.update(repr(signature).encode())
            self.files[str(path)] = signature
            ranges = []
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                if name in self.tensors or not isinstance(info, dict):
                    raise ValueError(f"duplicate or invalid tensor: {name}")
                dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
                if dtype not in _DTYPES or not isinstance(shape, list) or any(
                    type(dim) is not int or dim < 0 for dim in shape
                ):
                    raise ValueError(f"unsupported tensor dtype/shape: {name}")
                if not isinstance(offsets, list) or len(offsets) != 2 or any(
                    type(offset) is not int or offset < 0 for offset in offsets
                ):
                    raise ValueError(f"invalid tensor offsets: {name}")
                start, end = offsets
                nbytes = math.prod(shape) * _DTYPES[dtype]
                if end - start != nbytes or end + 8 + length > size:
                    raise ValueError(f"tensor range does not match shape or shard size: {name}")
                ranges.append((start, end))
                self.tensors[name] = {
                    "path": str(path), "offset": start + 8 + length,
                    "bytes": nbytes, "dtype": dtype, "shape": tuple(shape),
                }
            ranges.sort()
            if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
                raise ValueError(f"overlapping tensor data in {path.name}")
        self.fingerprint = fingerprint.hexdigest()
        self.projections = {}
        self.expert_bytes = {}
        self.expert_names = set()
        for layer in range(self.layers):
            layer_bytes = 0
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"model.layers.{layer}.ffn.switch_mlp.{projection}"
                q = dict(quant)
                override = quant.get(prefix)
                if override is not None:
                    if not isinstance(override, dict):
                        raise ValueError(f"invalid per-projection quantization: {prefix}")
                    q.update(override)
                bits = q.get("bits")
                group_size = q.get("group_size")
                if type(bits) is not int or type(group_size) is not int or bits not in (2, 4, 8) or group_size not in (32, 64, 128) or q.get("mode", "affine") != "affine":
                    raise ValueError(f"unsupported expert quantization: {prefix}")
                input_dims, output_dims = self.hidden, self.intermediate
                if projection == "down_proj":
                    input_dims, output_dims = output_dims, input_dims
                if input_dims % group_size or input_dims * bits % 32:
                    raise ValueError(f"invalid expert quantization geometry: {prefix}")
                for suffix in ("weight", "scales", "biases"):
                    name = f"{prefix}.{suffix}"
                    info = self.tensors.get(name)
                    width = input_dims * bits // 32 if suffix == "weight" else input_dims // group_size
                    allowed_dtype = ("U32",) if suffix == "weight" else ("F16", "BF16")
                    if info is None or info["shape"] != (self.experts, output_dims, width) or info["dtype"] not in allowed_dtype:
                        raise ValueError(f"missing or unsupported packed expert tensor: {name}")
                    self.expert_names.add(name)
                    layer_bytes += info["bytes"] // self.experts
                self.projections[(layer, projection)] = {"bits": bits, "group_size": group_size, "mode": "affine"}
            self.expert_bytes[layer] = layer_bytes
        for name in self.tensors:
            if (".switch_mlp." in name or ".ffn.experts." in name) and name not in self.expert_names:
                raise ValueError(f"unrecognized expert tensor layout: {name}")
            if name.startswith("mtp.") or not name.startswith(("model.", "lm_head.")):
                raise ValueError(f"streaming requires sanitized MLX tensor names: {name}")
        if self.cache_budget_mode == "auto":
            # Header-only inspection never queries hardware or loads weights.
            # Resolve the actual budget in the worker immediately before loading.
            self.cache_bytes = sum(self.expert_bytes.values()) * self.experts
        if self.cache_bytes < max(self.expert_bytes.values()):
            raise ValueError("expert cache must fit at least one complete routed expert")
        self.base_bytes = sum(v["bytes"] for k, v in self.tensors.items() if k not in self.expert_names)
        self.total_expert_bytes = sum(self.expert_bytes.values()) * self.experts
        self.resident_estimate_bytes = self.base_bytes + min(self.cache_bytes, self.total_expert_bytes) + _WORKSPACE_BYTES

    def summary(self):
        return {
            "architecture": "deepseek_v4", "fingerprint": self.fingerprint,
            "base_bytes": self.base_bytes, "expert_bytes": self.total_expert_bytes,
            "checkpoint_bytes": self.base_bytes + self.total_expert_bytes,
            "cache_budget_bytes": self.cache_bytes, "cache_budget_mode": self.cache_budget_mode,
            "workspace_bytes": _WORKSPACE_BYTES,
            "resident_estimate_bytes": self.resident_estimate_bytes,
            "expert_count": self.layers * self.experts,
            "largest_expert_bytes": max(self.expert_bytes.values()),
        }


def automatic_cache_budget(checkpoint, *, metal_limit, available_memory):
    """Bound retained experts by this device, current free memory and model size.

    Native admission further reserves actual KV/SDPA needs before each prefill.
    Unknown hardware telemetry must never become an unlimited allocation.
    """
    metal_limit = _positive_int(metal_limit, "Metal working-set limit")
    available_memory = _positive_int(available_memory, "available system memory")
    reserve = _WORKSPACE_BYTES + _ALLOCATOR_CACHE_BYTES
    room = min(int(metal_limit * 0.90), max(0, available_memory - 2 * 1024**3))
    budget = min(checkpoint.total_expert_bytes, room - checkpoint.base_bytes - reserve)
    minimum = max(checkpoint.expert_bytes.values())
    if budget < minimum:
        raise ValueError("not enough available memory for base weights, workspace and one routed expert")
    return budget


def _configure_automatic_cache(checkpoint):
    from omlx.process_memory_enforcer import get_effective_metal_cap_bytes
    import psutil

    checkpoint.cache_bytes = automatic_cache_budget(
        checkpoint, metal_limit=get_effective_metal_cap_bytes(),
        available_memory=psutil.virtual_memory().available)
    checkpoint.resident_estimate_bytes = checkpoint.base_bytes + checkpoint.cache_bytes + _WORKSPACE_BYTES


def inspect_model(model_path, cache_bytes):
    """Validate packed tensor layouts and estimate memory without reading weights."""
    return Checkpoint(model_path, cache_bytes).summary()


class ExpertManager:
    def __init__(self, checkpoint, execution=None):
        self.checkpoint = checkpoint
        self.execution = execution or os.environ.get("WERK_OMLX_EXPERT_EXECUTION", "grouped")
        if self.execution not in ("serial", "grouped"):
            raise ValueError("expert execution must be serial or grouped")
        self.model_id = checkpoint.path.name
        self._cache = OrderedDict()
        self._retention = None
        self._pinned = set()
        self._leased = set()
        self._usage = {}
        self._lock = threading.RLock()
        self._reader = None
        self._tensor_specs = {}
        self._resident = 0
        self.effective_cache_bytes = min(checkpoint.cache_bytes, checkpoint.total_expert_bytes)
        if getattr(checkpoint, "config", {}).get("model_type") == "deepseek_v4":
            # Imported lazily: the embedded worker registers the shared helper
            # after this module, before any manager is constructed.
            try:
                from _werk_omlx_offload import ExpertRetention
            except ImportError:
                from omlx_offload import ExpertRetention
            self._retention = ExpertRetention(self.effective_cache_bytes)
        self.budget_reductions = 0
        self.prefill_cache_growth_bytes = 0
        self.prefill_samples_corrected = 0
        self._model_ref = None
        self.hits = self.misses = self.disk_bytes_read = 0
        self.cache_evictions = self.allocator_clears = self.tensor_materializations = 0
        self.forward_calls = self.output_evaluations = 0
        self.disk_read_seconds = self.materialize_seconds = 0.0
        self.forward_seconds = self.routing_seconds = 0.0
        self.allocator_cache_bytes = 0

    def status(self):
        with self._lock:
            result = self.checkpoint.summary()
            result.update({"active": self._model_ref is not None and self._model_ref() is not None,
                           "model_id": self.model_id, "resident_cache_bytes": self._resident,
                           "effective_cache_budget_bytes": self.effective_cache_bytes,
                           "budget_reductions": self.budget_reductions,
                           "prefill_cache_growth_bytes": self.prefill_cache_growth_bytes,
                           "prefill_samples_corrected": self.prefill_samples_corrected,
                           "last_decode_admission": getattr(self, "last_decode_admission", None),
                           "resident_experts": len(self._cache), "cache_hits": self.hits,
                           "cache_misses": self.misses, "disk_bytes_read": self.disk_bytes_read,
                           "execution": self.execution, "cache_evictions": self.cache_evictions,
                           "cache_policy": "segmented_lru" if self._retention else "lru",
                           "allocator_clears": self.allocator_clears,
                           "allocator_cache_bytes": self.allocator_cache_bytes,
                           "allocator_cache_limit_bytes": _ALLOCATOR_CACHE_BYTES,
                           "tensor_materializations": self.tensor_materializations,
                           "disk_read_seconds": self.disk_read_seconds,
                           "materialize_seconds": self.materialize_seconds,
                           "forward_calls": self.forward_calls, "forward_seconds": self.forward_seconds,
                           "routing_seconds": self.routing_seconds,
                           "output_evaluations": self.output_evaluations})
            return result

    def deactivate(self):
        with self._lock:
            self._cache.clear()
            if self._retention:
                self._retention.clear()
            self._pinned.clear()
            self._leased.clear()
            self._resident = 0
            self._model_ref = None
            if self._reader is not None:
                self._reader.close()
                self._reader = None

    def _materialize(self, pending):
        import mlx.core as mx

        if not pending:
            return
        started = time.perf_counter()
        # Keep the source bytearrays alive until all numpy-backed arrays and
        # BF16 conversions have materialized, including failed evaluations.
        mx.eval(*(value for value, raw in pending))
        self.tensor_materializations += 1
        self.materialize_seconds += time.perf_counter() - started
        pending.clear()

    def _resize_cache(self, budget):
        """Change the working set on the MLX executor, preserving pins/leases."""
        with self._lock:
            protected = self._pinned | self._leased
            reserved = sum(self.checkpoint.expert_bytes[key[0]] for key in protected)
            protected_counts = {}
            for layer, _ in protected:
                protected_counts[layer] = protected_counts.get(layer, 0) + 1
            demand = max((size for layer, size in self.checkpoint.expert_bytes.items()
                          if protected_counts.get(layer, 0) < self.checkpoint.experts), default=0)
            # Pin actions promise that demand routing retains room for at least
            # one additional expert. Shrinking must preserve that promise.
            minimum = reserved + demand
            ceiling = min(self.checkpoint.cache_bytes, self.checkpoint.total_expert_bytes)
            budget = max(minimum, min(ceiling, int(budget)))
            if budget < self.effective_cache_bytes:
                self.budget_reductions += 1
            self.effective_cache_bytes = budget
            if self._retention:
                self._retention.resize(budget)
            evicted = False
            while self._resident > budget:
                victim = self._cache_victim(protected)
                if victim is None:
                    break
                self._cache.pop(victim)
                if self._retention:
                    self._retention.discard(victim)
                self._resident -= self.checkpoint.expert_bytes[victim[0]]
                self.cache_evictions += 1
                evicted = True
            if evicted:
                self._trim_allocator(force=True)

    def _trim_allocator(self, force=False):
        import mlx.core as mx

        self.allocator_cache_bytes = mx.get_cache_memory()
        if force or self.allocator_cache_bytes > _ALLOCATOR_CACHE_BYTES:
            mx.clear_cache()
            self.allocator_clears += 1
            self.allocator_cache_bytes = mx.get_cache_memory()

    def _range_reader(self):
        if self._reader is None:
            try:
                from _werk_omlx_offload import RangeReader
            except ImportError:
                from omlx_offload import RangeReader
            self._reader = RangeReader(prefetch_workers=4 if self.execution == "grouped" else 0)
        return self._reader

    def _tensor(self, name, expert=None):
        tensor = self._tensor_specs.get(name)
        if tensor is None:
            try:
                from _werk_omlx_offload import Tensor
            except ImportError:
                from omlx_offload import Tensor
            info = self.checkpoint.tensors[name]
            tensor = Tensor(Path(info["path"]), info["offset"], info["bytes"],
                            info["shape"], info["dtype"], self.checkpoint.files[info["path"]])
            self._tensor_specs[name] = tensor
        if expert is not None:
            if type(expert) is not int or not 0 <= expert < tensor.shape[0]:
                raise ValueError("expert index out of bounds")
            tensor = tensor.rows(expert)
        return tensor

    @contextmanager
    def prefetch_experts(self, layer, indices):
        # The caller leases this entire group before any read or acquisition.
        # Only demanded, missing rows are staged; CPU workers never call MLX.
        if self.execution != "grouped":
            yield
            return
        tensors = [self._tensor(self._projection_prefix(layer, projection) + "." + suffix, int(expert))
                   for expert in indices if (layer, int(expert)) not in self._cache
                   for projection in ("gate_proj", "up_proj", "down_proj")
                   for suffix in ("weight", "scales", "biases")]
        reader = self._range_reader()
        before_bytes, before_seconds = reader.logical_bytes, reader.read_seconds
        try:
            with reader.prefetch(tensors, max_bytes=_WORKSPACE_BYTES // 8):
                # Account only prefetch here. Acquisitions in the context count
                # their own reads (including the oversized-group fallback).
                self.disk_bytes_read += reader.logical_bytes - before_bytes
                self.disk_read_seconds += reader.read_seconds - before_seconds
                before_bytes = before_seconds = None
                yield
        finally:
            if before_bytes is not None:
                self.disk_bytes_read += reader.logical_bytes - before_bytes
                self.disk_read_seconds += reader.read_seconds - before_seconds

    def _read(self, name, expert=None, pending=None):
        import mlx.core as mx
        import numpy as np

        info = self.checkpoint.tensors[name]
        tensor = self._tensor(name, expert)
        shape = tensor.shape[1:] if expert is not None else tensor.shape
        reader = self._range_reader()
        before_bytes, before_seconds = reader.logical_bytes, reader.read_seconds
        try:
            raw = reader.read(tensor)
        finally:
            self.disk_bytes_read += reader.logical_bytes - before_bytes
            self.disk_read_seconds += reader.read_seconds - before_seconds
        dtypes = {"U32": "<u4", "I32": "<i4", "F32": "<f4", "F16": "<f2", "BF16": "<u2",
                  "I64": "<i8", "U64": "<u8", "I16": "<i2", "U16": "<u2", "I8": "i1", "U8": "u1", "BOOL": "?"}
        value = mx.array(np.frombuffer(raw, dtype=dtypes[info["dtype"]]).reshape(shape))
        if info["dtype"] == "BF16":
            value = value.view(mx.bfloat16)
            # Matches oMLX DeepSeek V4 sanitize: affine expert metadata is f16.
            if expert is not None and name.endswith((".scales", ".biases")):
                value = value.astype(mx.float16)
        if pending is None:
            self._materialize([(value, raw)])
        else:
            pending.append((value, raw))
        return value

    def _cache_victim(self, excluded):
        if self._retention:
            return self._retention.victim(excluded)
        return next((key for key in self._cache if key not in excluded), None)

    def _acquire(self, key, record=True):
        import mlx.core as mx

        layer, expert = key
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
        else:
            size = self.checkpoint.expert_bytes[layer]
            while self._resident + size > self.effective_cache_bytes:
                victim = self._cache_victim(self._pinned | self._leased)
                if victim is None:
                    raise ValueError("expert cache is full of pinned experts; unpin experts or raise its budget")
                self._cache.pop(victim)
                if self._retention:
                    self._retention.discard(victim)
                self._resident -= self.checkpoint.expert_bytes[victim[0]]
                self.cache_evictions += 1
                # Small recyclable allocations serve the next expert load.
                # This allowance is part of the existing workspace reservation.
                self._trim_allocator(force=self.execution == "serial")
            values = {}
            pending = [] if self.execution == "grouped" else None
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = self._projection_prefix(layer, projection)
                values[projection] = tuple(self._read(f"{prefix}.{suffix}", expert, pending=pending)
                                           for suffix in ("weight", "scales", "biases"))
            self._materialize(pending)
            self._cache[key] = values
            self._resident += size
            self.misses += 1
        if self._retention:
            self._retention.touch(key, self.checkpoint.expert_bytes[layer])
        if record:
            previous = self._usage.get(key, (0, None))
            self._usage[key] = (previous[0] + 1, int(time.time() * 1000))
        return self._cache[key]

    def _projection_prefix(self, layer, projection):
        return f"model.layers.{layer}.ffn.switch_mlp.{projection}"

    def _groups(self, layer, experts):
        """Group routes without exceeding the cache, including persistent pins."""
        group = []
        reserved = sum(self.checkpoint.expert_bytes[key[0]] for key in self._pinned)
        used = reserved
        limit = 1 if self.execution == "serial" else _EXPERT_GROUP_SIZE
        for expert in experts:
            key = (layer, int(expert))
            size = 0 if key in self._pinned else self.checkpoint.expert_bytes[layer]
            if group and (len(group) == limit or used + size > self.effective_cache_bytes):
                yield group
                group, used = [], reserved
            if used + size > self.effective_cache_bytes:
                raise ValueError("expert cache is full of pinned experts; unpin experts or raise its budget")
            group.append(key)
            used += size
        if group:
            yield group

    def _key(self, expert_id):
        match = re.fullmatch(r"layer\.(\d+)\.expert\.(\d+)", str(expert_id))
        if match is None:
            raise ValueError(f"invalid expert id: {expert_id}")
        key = tuple(map(int, match.groups()))
        if key[0] not in self.checkpoint.expert_bytes or key[1] >= self.checkpoint.experts:
            raise ValueError(f"unknown expert id: {expert_id}")
        return key

    def _summary(self, key):
        use, last = self._usage.get(key, (0, None))
        return {"id": f"layer.{key[0]}.expert.{key[1]}", "model_id": self.model_id,
                "tier": "ram" if key in self._cache else "external",
                "bytes": self.checkpoint.expert_bytes[key[0]], "hotness": float(use),
                "pinned": key in self._pinned, "last_used_unix_ms": last}

    def list_experts(self, filters=None):
        filters = filters or {}
        if not isinstance(filters, dict) or set(filters) - {"model_id", "tier", "limit", "cursor", "allow_experimental"}:
            raise ValueError("invalid expert list filter")
        if filters.get("allow_experimental") is not True:
            raise ValueError("expert streaming lists require allow_experimental=true")
        if not self.status()["active"]:
            raise ValueError("streamed model is not loaded")
        if filters.get("model_id") not in (None, self.model_id):
            raise ValueError("model_id does not identify this worker's streamed model")
        limit = filters.get("limit")
        if limit is None:
            limit = 100
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("expert page limit must be between 1 and 1000")
        tier = filters.get("tier")
        if tier not in (None, "ram", "external", "vram"):
            raise ValueError("unknown expert tier")
        cursor = filters.get("cursor")
        offset = 0
        if cursor:
            prefix = self.checkpoint.fingerprint[:16] + ":" + str(tier) + ":"
            if not str(cursor).startswith(prefix):
                raise ValueError("expert cursor does not match this checkpoint and filter")
            try:
                offset = int(str(cursor)[len(prefix):])
            except ValueError as error:
                raise ValueError("invalid expert cursor") from error
        routed_layers = sorted(self.checkpoint.expert_bytes)
        total = len(routed_layers) * self.checkpoint.experts
        if not 0 <= offset <= total:
            raise ValueError("invalid expert cursor offset")
        with self._lock:
            rows = []
            index = offset
            while index < total and len(rows) < limit:
                layer_index, expert_index = divmod(index, self.checkpoint.experts)
                key = (routed_layers[layer_index], expert_index)
                row = self._summary(key)
                if tier is None or row["tier"] == tier:
                    rows.append(row)
                index += 1
            return {"experts": rows, "next_cursor": None if index == total else
                    self.checkpoint.fingerprint[:16] + ":" + str(tier) + ":" + str(index)}

    def action(self, request):
        if not isinstance(request, dict) or set(request) - {"model_id", "expert_ids", "action", "target_tier", "dry_run", "allow_experimental"}:
            raise ValueError("invalid expert action request")
        if request.get("model_id") != self.model_id:
            raise ValueError("model_id does not identify this worker's streamed model")
        if request.get("allow_experimental") is not True:
            raise ValueError("expert streaming actions require allow_experimental=true")
        action = request.get("action")
        if action not in ("prefetch", "pin", "unpin", "evict"):
            raise ValueError("unknown expert action")
        target = request.get("target_tier")
        if action == "prefetch" and target != "ram":
            raise ValueError("prefetch requires target_tier=ram on Apple unified memory")
        if action != "prefetch" and target is not None:
            raise ValueError("target_tier is only valid for prefetch")
        ids = request.get("expert_ids")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 4096 or any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("actions require 1..4096 unique expert ids")
        keys = [self._key(item) for item in ids]
        dry_run = request.get("dry_run", False)
        if type(dry_run) is not bool:
            raise ValueError("dry_run must be a boolean")
        with self._lock:
            if not self.status()["active"]:
                raise ValueError("streamed model is not loaded")
            if action == "evict" and any(key in self._pinned for key in keys):
                raise ValueError("cannot evict pinned experts; unpin them first")
            if action in ("pin", "prefetch"):
                required = self._pinned.union(keys)
                if sum(self.checkpoint.expert_bytes[key[0]] for key in required) > self.effective_cache_bytes:
                    raise ValueError("requested experts and pins exceed the expert cache budget")
                # Reserve room for one demand-loaded expert so pinning can never
                # disable the model's next legitimate routing decision.
                if action == "pin" and len(required) < len(self.checkpoint.expert_bytes) * self.checkpoint.experts:
                    if sum(self.checkpoint.expert_bytes[key[0]] for key in required) + max(self.checkpoint.expert_bytes.values()) > self.effective_cache_bytes:
                        raise ValueError("pins must leave room for at least one demand-loaded expert")
            before = [self._summary(key) for key in keys]
            if not dry_run:
                if action in ("pin", "prefetch"):
                    prior_pins = self._pinned.copy()
                    self._pinned.update(keys)
                    try:
                        for key in keys:
                            self._acquire(key, record=False)
                    except BaseException:
                        self._pinned = prior_pins
                        raise
                    finally:
                        if action == "prefetch":
                            self._pinned = prior_pins
                elif action == "unpin":
                    self._pinned.difference_update(keys)
                else:
                    for key in keys:
                        if key in self._cache:
                            self._cache.pop(key)
                            if self._retention:
                                self._retention.discard(key)
                            self._resident -= self.checkpoint.expert_bytes[key[0]]
                    import mlx.core as mx
                    mx.clear_cache()
            after = [self._summary(key) for key in keys]
            if dry_run:
                changed = sum((action == "pin" and not row["pinned"]) or
                              (action == "unpin" and row["pinned"]) or
                              (action == "prefetch" and row["tier"] == "external") or
                              (action == "evict" and row["tier"] == "ram") for row in before)
            else:
                changed = sum(a != b for a, b in zip(before, after))
            return {"experts": after, "changed": changed, "dry_run": dry_run}


def _streamed_module(manager, layer, activation):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    class StreamedSwitchGLU(nn.Module):
        def __init__(self):
            super().__init__()
            # The manager is a plain object, not a parameter subtree. Cached
            # arrays never participate in model.parameters()/materialization.
            self._expert_manager = manager
            self.activation = activation

        def __call__(self, x, indices, scores=None, weighted_sum=False):
            with manager._lock:
                started = time.perf_counter()
                mx.eval(x, indices)
                routes = np.array(indices).reshape(-1, indices.shape[-1])
                manager.routing_seconds += time.perf_counter() - started
                flat_x = x.reshape(-1, x.shape[-1])
                if routes.shape[0] != flat_x.shape[0] or np.any(routes < 0) or np.any(routes >= manager.checkpoint.experts):
                    raise ValueError("invalid expert routing shape or index")
                # A bounded prefill chunk is required by the worker's scheduler.
                output_bytes = routes.size * x.shape[-1] * x.itemsize
                if output_bytes > _WORKSPACE_BYTES // 2:
                    raise ValueError("expert prefill batch exceeds streaming workspace; reduce prefill chunk size")
                output = mx.zeros((routes.size, x.shape[-1]), dtype=x.dtype)
                mx.eval(output)
                for group in manager._groups(layer, np.unique(routes)):
                    manager._leased.update(group)
                    acquired = []
                    try:
                        with manager.prefetch_experts(layer, [key[1] for key in group]):
                            for key in group:
                                rows, slots = np.nonzero(routes == key[1])
                                acquired.append((manager._acquire(key), rows, slots))
                        # Output storage uses at most half the workspace; this
                        # separate allowance bounds the grouped activation DAG.
                        bytes_per_row = len(group) * max(x.shape[-1], manager.checkpoint.intermediate) * 4 * 4
                        if bytes_per_row > _WORKSPACE_BYTES // 4:
                            raise ValueError("expert activation exceeds streaming workspace")
                        chunk_size = min(128, (_WORKSPACE_BYTES // 4) // bytes_per_row)
                        for start in range(0, max(len(rows) for _, rows, _ in acquired), chunk_size):
                            for weights, rows, slots in acquired:
                                batch_rows = rows[start:start + chunk_size]
                                if not len(batch_rows):
                                    continue
                                selected = flat_x[mx.array(batch_rows)]
                                if selected.dtype == mx.bfloat16:
                                    selected = selected.astype(mx.float16)

                                def project(name, inputs):
                                    w, scales, biases = weights[name]
                                    q = manager.checkpoint.projections[(layer, name)]
                                    return mx.quantized_matmul(inputs, w, scales, biases, transpose=True, **q)

                                up = project("up_proj", selected)
                                gate = project("gate_proj", selected)
                                hidden = self.activation(up, gate)
                                value = project("down_proj", hidden).astype(x.dtype)
                                positions = batch_rows * routes.shape[1] + slots[start:start + chunk_size]
                                output = output.at[mx.array(positions)].add(value)
                                del selected, up, gate, hidden, value
                            # Every expert in this lazy graph stays leased until
                            # its output has completed, even with a tiny cache.
                            mx.eval(output)
                            manager.output_evaluations += 1
                        del weights
                    finally:
                        # On an exception the unreturned graph is discarded;
                        # no subsequent acquisition can observe a partial result.
                        acquired.clear()
                        manager._leased.difference_update(group)
                    if manager.execution == "grouped":
                        manager._trim_allocator()
                output = output.reshape(*indices.shape, x.shape[-1])
                if weighted_sum and scores is not None:
                    output = (output * scores[..., None].astype(output.dtype)).sum(-2)
                    mx.eval(output)
                manager.forward_calls += 1
                manager.forward_seconds += time.perf_counter() - started
                return output

    return StreamedSwitchGLU()


def _load_streamed_model(manager, tokenizer_config=None, **kwargs):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import utils

    if kwargs.get("adapter_path") or kwargs.get("model_config") or kwargs.get("revision"):
        raise ValueError("expert streaming does not support adapters, config overrides, or remote revisions")
    allowed = {"adapter_path", "model_config", "revision", "lazy", "return_config", "trust_remote_code"}
    if set(kwargs) - allowed:
        raise ValueError("unsupported expert streaming loader arguments")
    checkpoint = manager.checkpoint
    # Recheck before loading; admission was made for this exact geometry.
    current = Checkpoint(checkpoint.path, checkpoint.cache_bytes)
    if current.fingerprint != checkpoint.fingerprint:
        raise ValueError("checkpoint changed after expert streaming preflight")
    if manager.status()["active"]:
        raise ValueError("streamed model is already loaded in this worker")
    manager.deactivate()
    config = dict(checkpoint.config)
    from omlx.patches.deepseek_v4.utils_patch import _native_ratio128_attention_enabled
    config["use_native_ratio128_attention"] = bool(config.get("use_native_ratio128_attention", True)) and _native_ratio128_attention_enabled(config)
    model_class, args_class = utils._get_classes(config=config)
    model = model_class(args_class.from_dict(config))
    try:
        layers = model.model.layers
        if len(layers) != checkpoint.layers:
            raise ValueError("installed DeepSeek model layer layout does not match checkpoint")
        for layer_index, layer in enumerate(layers):
            switch = layer.ffn.switch_mlp
            layer.ffn.switch_mlp = _streamed_module(manager, layer_index, switch.activation)
        # Drop the constructor's lazy expert arrays without ever evaluating them.
        del switch
        weights = {name: manager._read(name) for name in checkpoint.tensors if name not in checkpoint.expert_names}
        weights = model.sanitize(weights)
        quant = config["quantization"]

        def quantized(path, module):
            if not hasattr(module, "to_quantized") or f"{path}.scales" not in weights:
                return False
            return quant.get(path, True)

        nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"],
                    mode=quant.get("mode", "affine"), class_predicate=quantized)
        model.eval()
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        del weights
        mx.clear_cache()
        tokenizer = utils.load_tokenizer(checkpoint.path, tokenizer_config,
                                         eos_token_ids=config.get("eos_token_id"))
        manager._model_ref = weakref.ref(model)
        weakref.finalize(model, manager.deactivate)
        if kwargs.get("return_config", False):
            return model, tokenizer, config
        return model, tokenizer
    except BaseException:
        manager.deactivate()
        raise


def get_manager():
    return _MANAGER


class _ExpertMemoryGuard:
    """Account streamed weight residency separately from native KV/SDPA costs.

    oMLX 0.6.4 learns chunk transients from process-footprint growth. Filling
    our persistent weight cache is not a repeatable per-token transient. Keep
    that growth out of the learner, but reserve all remaining cache capacity
    in admission and chunk checks, including the first cold request.
    """

    def __init__(self, manager, scheduler_class, engine_class):
        self.manager = manager
        self.samples = weakref.WeakKeyDictionary()
        self.deferred_reclaims = weakref.WeakKeyDictionary()
        self.original_current = scheduler_class._current_usage_bytes
        self.original_guard = scheduler_class._guard_prefill_chunk
        self.original_adaptive = scheduler_class._adaptive_chunk_size
        self.original_record = scheduler_class._record_chunk_transient
        self.original_check = scheduler_class._preflight_memory_check
        self.original_responses = scheduler_class._process_batch_responses
        self.original_preflight = engine_class._preflight_or_raise_with_eviction
        contracts = (
            (self.original_current, ("self", "refresh_mlx_active")),
            (self.original_guard, ("self", "n_tokens", "kv_len", "progress", "loop_label", "request_id")),
            (self.original_adaptive, ("self", "requested", "request_id", "loop_label", "kv_len")),
            (self.original_record, ("self", "n_tokens", "pre_bytes", "post_bytes", "request_id", "loop_label", "kv_len", "requested_step")),
            (self.original_check, ("self", "request")),
            (self.original_responses, ("self", "responses")),
            (self.original_preflight, ("self", "scheduler", "num_prompt_tokens", "request_id")),
        )
        if any(tuple(inspect.signature(method).parameters) != signature
               for method, signature in contracts):
            raise ValueError("unsupported oMLX prefill accounting API for expert streaming")

    def matches(self, scheduler):
        model = self.manager._model_ref() if self.manager._model_ref is not None else None
        return model is not None and getattr(scheduler, "model", None) is model

    def cache_totals(self):
        extra = getattr(self.manager, "auxiliary_cache_usage", None)
        resident, capacity = extra() if extra is not None else (0, 0)
        return self.manager._resident + resident, self.manager.effective_cache_bytes + capacity

    def current(self, scheduler, *, refresh_mlx_active=True):
        current = self.original_current(scheduler, refresh_mlx_active=refresh_mlx_active)
        if self.matches(scheduler):
            with self.manager._lock:
                # Resident weights are already in native current usage. Only
                # unused capacity is extra; never subtract weights from usage.
                resident, capacity = self.cache_totals()
                current += max(0, capacity - resident)
                current += _WORKSPACE_BYTES
        return current

    def prepare(self, scheduler, *, num_prompt_tokens, cached_tokens=0, chunk=None,
                phase="prefill"):
        """Reclaim only on the MLX executor, before native guards see usage."""
        if not self.matches(scheduler):
            return
        with self.manager._lock:
            current = self.original_current(scheduler)
            resident_before, _ = self.cache_totals()
            credit, previous_usage, previous_resident = self.deferred_reclaims.get(
                scheduler, (0, current, resident_before))
            # Consume releases that landed between scheduler callbacks. Account
            # for intervening weight growth so it cannot hide a delayed release.
            credit = max(0, credit - max(0, previous_usage + resident_before - previous_resident - current))
            estimate = scheduler._admission_estimate(
                num_prompt_tokens=num_prompt_tokens, cached_tokens=cached_tokens, current=current)
            peak = estimate.kv_exact + estimate.transient if estimate is not None else 0
            if chunk is not None:
                n_tokens, kv_len = chunk
                peak = max(peak, scheduler._admission_transient_bound(n_tokens, kv_len),
                           scheduler._predicted_chunk_transient(n_tokens, kv_len))
                reserve = getattr(self.manager, "prefill_transient_reserve", None)
                if reserve is not None:
                    peak += reserve(n_tokens, peak)
            hard = scheduler._memory_hard_limit_bytes
            caps = [scheduler._admission_limit_bytes(), scheduler._prefill_abort_cap(),
                    int(hard * scheduler._prefill_headroom_safety)]
            caps = [cap for cap in caps if cap > 0]
            if not caps:
                # An unknown ceiling must not be treated as unlimited room.
                return
            resident, _ = self.cache_totals()
            non_expert = max(0, current - resident)
            # Native preflight re-samples process usage after this executor
            # callback. Filling the predicted cap exactly makes even a small
            # intervening allocation reject a feasible prompt. Leave a bounded
            # margin outside current()'s workspace reservation; adding it to
            # both calculations would cancel it and leave no real headroom.
            available = (min(caps) - non_expert - int(peak) - _WORKSPACE_BYTES
                         - _ADMISSION_HEADROOM_BYTES)
            resize_auxiliary = getattr(self.manager, "resize_auxiliary_cache", None)
            if resize_auxiliary is not None:
                available -= resize_auxiliary(available)
            self.manager._resize_cache(available)
            admission = {
                "chunk_tokens": chunk[0] if chunk else None,
                "cached_tokens": cached_tokens,
                "current_bytes": current,
                "non_weight_bytes": non_expert,
                "predicted_peak_bytes": int(peak),
                "ceiling_bytes": min(caps),
                "headroom_bytes": _ADMISSION_HEADROOM_BYTES,
                "effective_expert_bytes": self.manager.effective_cache_bytes,
            }
            setattr(self.manager, "last_" + phase + "_admission", admission)
            # Refresh executor telemetry after actual eviction. Early HTTP
            # preflight subsequently reads this sample without touching MLX.
            refreshed = self.original_current(scheduler)
            resident_after = self.cache_totals()[0]
            deferred = max(0, resident_before - resident_after - max(0, current - refreshed))
            self.deferred_reclaims[scheduler] = (credit + deferred, refreshed, resident_after)

    def responses(self, scheduler, responses):
        result = self.original_responses(scheduler, responses)
        # This callback runs on the owning MLX executor, after prefill locals
        # have been released and the first token has been evaluated. Revisit
        # the budget throughout decode: a prefill-sized reserve must not keep
        # evicting experts for the rest of a single-request generation.
        # Interleaved prefills/batches retain their existing native accounting.
        if (self.matches(scheduler) and not scheduler.prefilling
                and not scheduler.waiting and len(scheduler.running) == 1):
            request = next(iter(scheduler.running.values()))
            if request.num_output_tokens > 0:
                kv_len = request.num_prompt_tokens + request.num_output_tokens
                self.prepare(scheduler, num_prompt_tokens=kv_len + 1,
                             cached_tokens=kv_len, chunk=(1, kv_len), phase="decode")
        return result

    def adaptive(self, scheduler, requested, *, request_id, loop_label, kv_len=0):
        # Native adaptive sizing runs BEFORE the final chunk guard. Reclaim
        # weight caches here, or that earlier gate throttles against their old
        # residency and causes extra passes over the offloaded checkpoint.
        if self.matches(scheduler) and requested > 0:
            self.prepare(scheduler, num_prompt_tokens=kv_len + requested + 1,
                         cached_tokens=kv_len, chunk=(requested, kv_len))
        return self.original_adaptive(scheduler, requested, request_id=request_id,
                                      loop_label=loop_label, kv_len=kv_len)

    def guard(self, scheduler, n_tokens, *, kv_len, progress, loop_label, request_id=None):
        if self.matches(scheduler):
            self.samples.pop(scheduler, None)
            self.prepare(scheduler, num_prompt_tokens=kv_len + n_tokens + 1,
                         cached_tokens=kv_len, chunk=(n_tokens, kv_len))
        n = self.original_guard(scheduler, n_tokens, kv_len=kv_len, progress=progress,
                                loop_label=loop_label, request_id=request_id)
        if self.matches(scheduler):
            self.samples[scheduler] = (request_id, loop_label, n, self.cache_totals()[0],
                                       getattr(scheduler, "_last_mlx_active_memory_bytes", None))
        return n

    def record(self, scheduler, n_tokens, pre_bytes, post_bytes, *, request_id,
               loop_label, kv_len=0, requested_step=None):
        sample = self.samples.pop(scheduler, None)
        if self.matches(scheduler) and sample is not None and sample[:3] == (request_id, loop_label, n_tokens):
            change = self.cache_totals()[0] - sample[3]
            # Remove residency changes in both directions. Deliberate expert
            # eviction must not train the native pool-reallocation ledger.
            post_bytes -= change
            credit = self.deferred_reclaims.get(scheduler, (0, 0, 0))[0]
            if credit:
                # Physical footprint may fall a chunk AFTER weight eviction.
                # Do not teach the native allocator tracker to reserve those
                # deliberately removed weights again as transient workspace.
                released = min(credit, max(0, pre_bytes - post_bytes))
                post_bytes += released
                current = self.original_current(scheduler)
                active = getattr(scheduler, "_last_mlx_active_memory_bytes", None)
                if sample[4] is not None and active is not None:
                    # A simultaneous real MLX growth must remain visible even
                    # when delayed physical release outweighs it.
                    post_bytes = max(post_bytes, pre_bytes + active - sample[4] - change)
                self.deferred_reclaims[scheduler] = (credit - released, current, self.cache_totals()[0])
            self.manager.prefill_cache_growth_bytes += max(0, change)
            self.manager.prefill_samples_corrected += 1
        return self.original_record(scheduler, n_tokens, pre_bytes, post_bytes,
                                    request_id=request_id, loop_label=loop_label,
                                    kv_len=kv_len, requested_step=requested_step)

    def check(self, scheduler, request):
        self.prepare(scheduler, num_prompt_tokens=request.num_prompt_tokens,
                     cached_tokens=request.cached_tokens or 0)
        return self.original_check(scheduler, request)

    async def preflight(self, engine, scheduler, *, num_prompt_tokens, request_id):
        if self.matches(scheduler):
            import asyncio
            from functools import partial

            executor = getattr(getattr(getattr(engine, "_engine", None), "engine", None),
                               "_mlx_executor", None)
            if executor is None:
                raise ValueError("expert memory admission requires the owning MLX executor")
            await asyncio.get_running_loop().run_in_executor(
                executor, partial(self.prepare, scheduler, num_prompt_tokens=num_prompt_tokens))
        return await self.original_preflight(engine, scheduler,
                                             num_prompt_tokens=num_prompt_tokens,
                                             request_id=request_id)

    def install(self, scheduler_class, engine_class):
        # Class attributes need ordinary functions (a bound helper method would
        # consume the scheduler/engine argument). Other models delegate intact.
        owner = self
        def current(scheduler, *, refresh_mlx_active=True):
            return owner.current(scheduler, refresh_mlx_active=refresh_mlx_active)
        def guard(scheduler, n_tokens, **kwargs):
            return owner.guard(scheduler, n_tokens, **kwargs)
        def adaptive(scheduler, requested, **kwargs):
            return owner.adaptive(scheduler, requested, **kwargs)
        def record(scheduler, n_tokens, pre_bytes, post_bytes, **kwargs):
            return owner.record(scheduler, n_tokens, pre_bytes, post_bytes, **kwargs)
        def check(scheduler, request):
            return owner.check(scheduler, request)
        def responses(self, responses):
            # Preserve the native signature for the persistence wrapper.
            return owner.responses(self, responses)
        async def preflight(engine, scheduler, **kwargs):
            return await owner.preflight(engine, scheduler, **kwargs)
        scheduler_class._current_usage_bytes = current
        scheduler_class._guard_prefill_chunk = guard
        scheduler_class._adaptive_chunk_size = adaptive
        scheduler_class._record_chunk_transient = record
        scheduler_class._preflight_memory_check = check
        scheduler_class._process_batch_responses = responses
        engine_class._preflight_or_raise_with_eviction = preflight


def install(model_path, cache_bytes):
    """Install only in a private Werk worker, before the console entry point."""
    global _MANAGER
    if _MANAGER is not None:
        raise ValueError("expert streaming is already configured for this worker")
    # The admission and loading hooks depend on private runtime contracts. New
    # versions need validation; ordinary oMLX execution has no such restriction.
    if importlib.metadata.version("omlx") != "0.6.4":
        raise ValueError("experimental expert streaming currently requires oMLX 0.6.4")
    from omlx.utils import model_loading
    from omlx.engine_pool import EnginePool
    from omlx.scheduler import Scheduler
    from omlx.engine.batched import BatchedEngine

    original_load = model_loading.lm_load_compat
    original_size = EnginePool._entry_runtime_resident_size
    if tuple(inspect.signature(original_size).parameters) != ("self", "entry", "runtime_settings", "base_size"):
        raise ValueError("unsupported oMLX memory admission API for expert streaming")
    checkpoint = Checkpoint(model_path, cache_bytes)
    if checkpoint.cache_budget_mode == "auto":
        _configure_automatic_cache(checkpoint)
    manager = ExpertManager(checkpoint)
    memory_guard = _ExpertMemoryGuard(manager, Scheduler, BatchedEngine)

    def matches(path):
        return Path(path).resolve() == checkpoint.path

    def streamed_load(path_or_repo, **kwargs):
        if not matches(path_or_repo):
            return original_load(path_or_repo, **kwargs)
        # oMLX's engine invokes this after its architecture/tokenizer patches.
        return _load_streamed_model(manager, **kwargs)

    def resident_size(pool, entry, runtime_settings, *, base_size=None):
        if not matches(entry.model_path):
            return original_size(pool, entry, runtime_settings, base_size=base_size)
        if pool._distributed_deployment_for_entry(entry) is not None:
            raise ValueError("expert streaming cannot be combined with distributed loading")
        manager.model_id = entry.model_id
        return checkpoint.resident_estimate_bytes

    model_loading.lm_load_compat = streamed_load
    EnginePool._entry_runtime_resident_size = resident_size
    memory_guard.install(Scheduler, BatchedEngine)
    _MANAGER = manager
    return manager
