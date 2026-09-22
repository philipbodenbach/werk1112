#!/usr/bin/env python3
"""Compare two decode implementations using a bounded pool of real expert rows.

Uses one GLM/Qwen layer, BF16 inputs, native activations, and separate warm
TextExpertManager caches. No model, attention, router, tokenizer, KV cache,
server or full-model generation is started. Run without competing inference,
using the installed oMLX Python. Results do not establish tokens/s or a
full-model performance guarantee; missing-expert I/O is deliberately excluded.

Example:
  python compare_decode.py /path/to/model --output /tmp/decode.json
  python compare_decode.py /path/to/model --max-experts 8 --output /tmp/small.json
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/backend"))


def positive_integer(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def source_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def package_versions():
    result = {}
    for package in ("omlx", "mlx", "mlx-lm", "mlx-vlm", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def cache_stats(manager):
    return {
        "cache_hits": manager.hits,
        "cache_misses": manager.misses,
        "cache_evictions": manager.cache_evictions,
        "resident_cache_bytes": manager._resident,
        "effective_cache_budget_bytes": manager.effective_cache_bytes,
        "logical_read_bytes": manager.reader.logical_bytes,
        "read_calls": manager.reader.calls,
        "output_evaluations": manager.access.output_evaluations,
        "forward_calls": manager.access.forward_calls,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, help="default: first routed layer")
    parser.add_argument("--max-experts", type=positive_integer,
                        help="pool size cap; default 2*top_k; must fit top_k distinct experts")
    parser.add_argument("--cache-mib", type=positive_integer, default=256,
                        help="fixed budget for EACH of the two caches (default: 256)")
    parser.add_argument("--iterations", type=positive_integer, default=60)
    parser.add_argument("--warmup", type=positive_integer, default=12)
    parser.add_argument("--cases", type=positive_integer, default=8)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--unweighted", action="store_true",
                        help="measure raw expert outputs instead of native-style score reduction")
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np
    import omlx_decode as candidate_runtime
    import omlx_offload_runtime as legacy_runtime
    import omlx_text_offload as adapter

    cp = adapter.TextCheckpoint(args.model, args.cache_mib * 1024**2, 0)
    inv = cp.inventory
    layer = min(cp.expert_bytes) if args.layer is None else args.layer
    if layer not in cp.expert_bytes:
        parser.error("--layer must identify a routed expert layer")
    pool_size = min(inv.experts, args.max_experts or 2 * inv.top_k)
    if pool_size < inv.top_k:
        parser.error("--max-experts must be at least the model's top_k")
    pool_bytes = pool_size * cp.expert_bytes[layer]
    if pool_bytes > cp.cache_bytes:
        parser.error("selected expert pool exceeds --cache-mib; reduce --max-experts or raise the budget")

    # Resolve the installed native configuration/activation without constructing
    # the model or any random dense/expert projection weights.
    _, native_args = adapter.native_classes(cp.config, cp.path)
    if inv.architecture == "glm5_next":
        language = importlib.import_module("mlx_vlm.models.glm5_next.language")
        activation = language.Glm5NextClampedSwiGLU(native_args.text_config.swiglu_limit)
        score_dtype = mx.float32
        reduction = "inside_expert_module"
    elif inv.architecture == "qwen4_exp":
        from mlx_lm.models.switch_layers import SwiGLU
        activation = SwiGLU()
        score_dtype = mx.bfloat16
        reduction = "outside_expert_module"
    else:
        parser.error("only the installed GLM and Qwen text adapters are supported")
    if args.unweighted:
        reduction = "none"

    rng = np.random.default_rng(args.seed)
    pool = sorted(int(expert) for expert in rng.choice(inv.experts, pool_size, replace=False))
    cases = []
    # Timed cases use distinct, changing routes as real top-k selection does.
    # Duplicate routes are additional untimed correctness probes below.
    for index in range(args.cases):
        routes = rng.choice(pool, inv.top_k, replace=False)
        if index % 2 == 0:
            routes.sort()
        x = mx.array(rng.standard_normal((1, 1, inv.hidden)).astype(np.float32)).astype(mx.bfloat16)
        scores = rng.uniform(.05, 1.0, (1, 1, inv.top_k)).astype(np.float32)
        scores /= scores.sum(axis=-1, keepdims=True)
        indices = mx.array(routes.reshape(1, 1, inv.top_k).astype(np.uint32))
        weights = mx.array(scores).astype(score_dtype)
        mx.eval(x, indices, weights)
        cases.append((x, indices, weights))
    probes = list(cases)
    if inv.top_k > 1:
        x, indices, weights = cases[0]
        duplicate = np.array(indices)
        duplicate[..., -1] = duplicate[..., 0]
        probes.append((x, mx.array(duplicate), weights))
        probes.append((x, mx.full(indices.shape, pool[0], dtype=mx.uint32), weights))

    managers = {}
    modules = {}
    factories = {"legacy": legacy_runtime.streamed_experts,
                 "candidate": candidate_runtime.streamed_decode_experts}
    correctness = {"comparisons": 0, "array_equal": True, "bitwise_equal": True,
                   "allclose_atol_0_rtol_0": True, "max_absolute_error": 0.0}

    def compare(left, right):
        # Evaluate and copy OUTSIDE all timing windows. BF16 is converted to
        # float32 for portable NumPy comparisons; raw bits are checked separately.
        mx.eval(left, right)
        lhs, rhs = np.array(left.astype(mx.float32)), np.array(right.astype(mx.float32))
        equal = bool(np.array_equal(lhs, rhs))
        close = bool(np.allclose(lhs, rhs, atol=0.0, rtol=0.0, equal_nan=False))
        bits_dtype = mx.uint16 if left.itemsize == 2 else mx.uint32
        bitwise = left.dtype == right.dtype and bool(np.array_equal(
            np.array(left.view(bits_dtype)), np.array(right.view(bits_dtype))))
        error = float(np.max(np.abs(lhs - rhs)))
        correctness["comparisons"] += 1
        correctness["array_equal"] &= equal
        correctness["bitwise_equal"] &= bitwise
        correctness["allclose_atol_0_rtol_0"] &= close
        correctness["max_absolute_error"] = max(correctness["max_absolute_error"], error)

    def call(module, case):
        x, indices, scores = case
        started = time.perf_counter_ns()
        if reduction == "inside_expert_module":
            value = module(x, indices, scores=scores, weighted_sum=True)
        else:
            value = module(x, indices)
            if reduction == "outside_expert_module":
                value = (value * scores[..., None]).sum(axis=-2)
        mx.eval(value)
        return (time.perf_counter_ns() - started) / 1_000_000, value

    try:
        for label, factory in factories.items():
            manager = adapter.TextExpertManager(cp, execution="grouped")
            managers[label] = manager
            modules[label] = factory(manager.access, layer, activation, adapter._WORKSPACE_BYTES)
            # Warm exactly the selected pool; no other expert/model rows load.
            for expert in pool:
                with manager.expert(layer, expert):
                    pass
        loaded = {label: cache_stats(manager) for label, manager in managers.items()}
        for case in probes:
            compare(call(modules["legacy"], case)[1], call(modules["candidate"], case)[1])
        labels = list(modules)
        for iteration in range(args.warmup):
            for label in labels[::1 if iteration % 2 else -1]:
                call(modules[label], cases[iteration % len(cases)])
        before = {label: cache_stats(manager) for label, manager in managers.items()}
        samples = {label: [] for label in labels}
        paired = []
        for iteration in range(args.iterations):
            order = labels[::1 if iteration % 2 else -1]
            case = cases[iteration % len(cases)]
            observed = {}
            for label in order:
                elapsed, value = call(modules[label], case)
                samples[label].append(elapsed)
                observed[label] = value
            compare(observed["legacy"], observed["candidate"])
            paired.append({"iteration": iteration, "case": iteration % len(cases),
                           "order": order, "legacy_ms": samples["legacy"][-1],
                           "candidate_ms": samples["candidate"][-1]})
        after = {label: cache_stats(manager) for label, manager in managers.items()}
        read_deltas = {label: after[label]["logical_read_bytes"] - before[label]["logical_read_bytes"]
                       for label in labels}
        stats = {
            label: {"median_ms": statistics.median(values), "min_ms": min(values),
                    "p10_ms": float(np.percentile(values, 10)),
                    "p90_ms": float(np.percentile(values, 90)), "max_ms": max(values)}
            for label, values in samples.items()
        }
        source_paths = [Path(__file__), Path(legacy_runtime.__file__), Path(candidate_runtime.__file__),
                        Path(adapter.__file__), ROOT / "src/backend/omlx_experts.py",
                        ROOT / "src/backend/omlx_offload.py", Path(inspect.getfile(type(activation)))]
        report = {
            "scope": "one real expert layer, warm bounded expert pools, BF16 single-token decode",
            "limits": ["not full-model tokens/s or model-quality evidence",
                       "no missing-expert I/O, prefill, attention, router, KV, or server overhead",
                       "fixed small route/input pool; not a production routing distribution",
                       "both implementations retain their own expert arrays during alternating samples"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv, "python": sys.version, "platform": platform.platform(),
            "packages": package_versions(), "sources": [source_record(path) for path in source_paths],
            "model": str(cp.path), "checkpoint_fingerprint": cp.fingerprint,
            "architecture": inv.architecture, "layer": layer, "hidden": inv.hidden,
            "intermediate": inv.intermediate, "top_k": inv.top_k, "expert_pool": pool,
            "pool_bytes_per_implementation": pool_bytes, "cache_bytes_per_implementation": cp.cache_bytes,
            "activation": type(activation).__module__ + "." + type(activation).__qualname__,
            "activation_limit": getattr(activation, "limit", None), "input_dtype": "bfloat16",
            "score_dtype": str(score_dtype), "reduction": reduction, "seed": args.seed,
            "timed_routes": [np.array(case[1]).reshape(-1).tolist() for case in cases],
            "warmup": args.warmup, "iterations": args.iterations, "correctness": correctness,
            "stats": stats, "candidate_speedup": stats["legacy"]["median_ms"] / stats["candidate"]["median_ms"],
            "paired_speedup_median": statistics.median(
                pair["legacy_ms"] / pair["candidate_ms"] for pair in paired),
            "warm_logical_read_bytes": read_deltas, "cache_after_pool_load": loaded,
            "cache_before_samples": before, "cache_after_samples": after,
            "samples_ms": samples, "paired_samples": paired,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: report[key] for key in
                         ("scope", "architecture", "layer", "top_k", "pool_bytes_per_implementation",
                          "correctness", "stats", "candidate_speedup", "warm_logical_read_bytes")}, indent=2))
        if not correctness["allclose_atol_0_rtol_0"] or any(read_deltas.values()):
            raise SystemExit("comparison failed: outputs differ or timed samples reloaded expert rows")
    finally:
        for manager in managers.values():
            manager.deactivate()


if __name__ == "__main__":
    main()
