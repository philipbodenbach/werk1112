"""Opt-in, bounded GLM expert-layer telemetry without MLX dependencies.

The caller must measure one serialized expert forward at a time, under the
owning manager's lock. These are deltas of existing counters, not GPU timers;
disk bytes include reads satisfied by the OS file cache. This module neither
wraps shared implementations nor changes cache admission or model arithmetic.
"""

from contextlib import contextmanager
import math
import sys
import threading
import time


_MAX_LAYERS = 1024
_MAX_COUNT = (1 << 63) - 1
_MAX_SECONDS = sys.float_info.max
_COUNTERS = (
    ("cache_hits", "manager", "hits", int),
    ("cache_misses", "manager", "misses", int),
    ("cache_evictions", "manager", "cache_evictions", int),
    ("disk_bytes_read", "reader", "logical_bytes", int),
    ("disk_read_calls", "reader", "calls", int),
    ("disk_read_seconds", "reader", "read_seconds", float),
    ("tensor_materializations", "manager", "tensor_materializations", int),
    ("materialize_seconds", "manager", "materialize_seconds", float),
    ("allocator_clears", "manager", "allocator_clears", int),
    ("load_seconds", "access", "load_seconds", float),
    ("forward_calls", "access", "forward_calls", int),
    ("forward_seconds", "access", "forward_seconds", float),
    ("routing_seconds", "access", "routing_seconds", float),
    ("output_evaluations", "access", "output_evaluations", int),
)


def _attribute(obj, name):
    try:
        return getattr(obj, name, None)
    except Exception:
        # Optional instrumentation must not fail a forward because a counter
        # disappeared or a runtime implemented it as an unavailable property.
        return None


def _number(value, kind):
    if type(value) not in (int, float) or value < 0:
        return None
    if kind is int:
        return min(value, _MAX_COUNT) if type(value) is int else None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _add(current, increment):
    ceiling = _MAX_SECONDS if type(current) is float else _MAX_COUNT
    return min(ceiling, current + increment)


def _empty():
    row = {name: kind(0) for name, _, _, kind in _COUNTERS}
    row.update(calls=0, failures=0, wall_seconds=0.0,
               counter_resets=0, invalid_samples=0)
    return row


class GlmLayerProfile:
    """Collect cumulative per-layer deltas only when explicitly enabled.

    ``measure(layer, manager, access=None)`` reads ``manager.access`` and
    ``manager.reader`` by default. Existing rows remain measurable after the
    layer bound is reached; new/invalid layer IDs only increment ignored_calls.
    No model, manager, tensors, or per-call history are retained by the profile.
    """

    def __init__(self, enabled=False, max_layers=256):
        if type(enabled) is not bool:
            raise ValueError("GLM profiling enabled must be a boolean")
        if type(max_layers) is not int or not 1 <= max_layers <= _MAX_LAYERS:
            raise ValueError("GLM profiling max_layers must be within 1..1024")
        self.enabled = enabled
        self.max_layers = max_layers
        self._layers = {}
        self._totals = _empty()
        self._ignored_calls = 0
        self._lock = threading.Lock()

    @staticmethod
    def _sample(manager, access):
        if access is None:
            access = _attribute(manager, "access")
        sources = {"manager": manager, "access": access,
                   "reader": _attribute(manager, "reader")}
        return {
            name: _number(_attribute(sources[source], attribute), kind)
            for name, source, attribute, kind in _COUNTERS
        }

    @contextmanager
    def measure(self, layer, manager, access=None):
        if not self.enabled:
            yield
            return
        with self._lock:
            valid = type(layer) is int and 0 <= layer <= _MAX_COUNT
            if valid and layer not in self._layers:
                if len(self._layers) < self.max_layers:
                    self._layers[layer] = _empty()
                else:
                    valid = False
            if not valid:
                self._ignored_calls = _add(self._ignored_calls, 1)
        if not valid:
            yield
            return

        before = self._sample(manager, access)
        started = time.perf_counter()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = _number(time.perf_counter() - started, float)
            after = self._sample(manager, access)
            delta = _empty()
            delta.update(calls=1, failures=int(failed), wall_seconds=elapsed or 0.0)
            for name, _, _, _ in _COUNTERS:
                if before[name] is None or after[name] is None:
                    delta["invalid_samples"] += 1
                elif after[name] < before[name]:
                    # Reset or replacement during a forward: attributing the
                    # new total to this layer would create an invented delta.
                    delta["counter_resets"] += 1
                else:
                    delta[name] = after[name] - before[name]
            with self._lock:
                for row in (self._layers[layer], self._totals):
                    for name, increment in delta.items():
                        row[name] = _add(row[name], increment)

    def snapshot(self):
        """Return detached, deterministic values accepted by strict JSON."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "max_layers": self.max_layers,
                "ignored_calls": self._ignored_calls,
                "totals": dict(self._totals),
                "layers": [dict(layer=layer, **values)
                           for layer, values in sorted(self._layers.items())],
            }
