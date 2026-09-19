#!/usr/bin/env python3
"""Compare installed native and Werk expert execution on identical real weights.

One layer fits on smaller Macs. This is NOT a whole-model tokens/s benchmark.
Run sequentially, without another inference process, using the oMLX Python.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/backend'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-runtime', type=Path)
    args = parser.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU
    import omlx_text_offload as adapter
    from omlx_offload_runtime import streamed_experts

    cp = adapter.TextCheckpoint(args.model, 2 * 1024**3, 0)
    inv = cp.inventory
    layer = min(cp.expert_bytes)
    selected = {}
    names = {}
    for projection in ('gate_proj', 'up_proj', 'down_proj'):
        prefix, _ = inv.projections[layer, projection]
        for suffix in ('weight', 'scales', 'biases'):
            name = prefix + '.' + suffix
            selected[name] = inv.tensors[name]
            names[adapter._canonical_name(name, inv.architecture)] = projection + '.' + suffix
    native = SwitchGLU(inv.hidden, inv.intermediate, inv.experts)
    nn.quantize(native, **inv.quantization('__default__'),
                class_predicate=lambda path, module: inv.projections[layer, path][1]
                if path in ('gate_proj', 'up_proj', 'down_proj') else False)
    start = time.perf_counter()
    weights = adapter._load_resident_weights(selected, inv.architecture)
    native.load_weights([(names[name], value) for name, value in weights.items()])
    native.eval()
    mx.eval(native.parameters())
    adapter._check_resident_shards(selected)
    load_seconds = time.perf_counter() - start
    del weights
    manager = adapter.TextExpertManager(cp)
    streamed = streamed_experts(manager.access, layer, native.activation, adapter._WORKSPACE_BYTES)
    mx.random.seed(42)
    x = mx.random.normal((1, 1, inv.hidden)).astype(mx.bfloat16)
    indices = mx.array([[list(range(inv.top_k))]], dtype=mx.uint32)
    mx.eval(x, indices)

    def call(module):
        start = time.perf_counter()
        value = module(x, indices)
        mx.eval(value)
        return time.perf_counter() - start, value

    native_first, reference = call(native)
    streamed_first, actual = call(streamed)
    import numpy as np
    np.testing.assert_allclose(np.array(actual.astype(mx.float32)),
                               np.array(reference.astype(mx.float32)), atol=.02, rtol=.02)
    before = manager.reader.logical_bytes
    samples = {'native': [], 'werk_cached': []}
    implementations = [('native', native), ('werk_cached', streamed)]
    reference_manager = None
    if args.reference_runtime:
        import importlib.util
        spec = importlib.util.spec_from_file_location('reference_runtime', args.reference_runtime)
        reference_runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_runtime)
        reference_manager = adapter.TextExpertManager(cp)
        old = reference_runtime.streamed_experts(reference_manager.access, layer, native.activation, adapter._WORKSPACE_BYTES)
        _, old_value = call(old)
        np.testing.assert_allclose(np.array(old_value.astype(mx.float32)),
                                   np.array(reference.astype(mx.float32)), atol=.02, rtol=.02)
        implementations.append(('werk_before', old))
        samples['werk_before'] = []
    # Alternate order to reduce thermal/order bias; identical ready inputs.
    for iteration in range(30):
        order = implementations
        for label, module in order[::1 if iteration % 2 else -1]:
            duration, value = call(module)
            samples[label].append(duration * 1000)
    report = {
        'scope': 'one real expert layer; fixed routes; not full-model throughput',
        'model': str(args.model), 'layer': layer, 'top_k': inv.top_k,
        'resident_layer_GiB': sum(t.size for t in selected.values()) / 1024**3,
        'native_load_seconds': load_seconds,
        'first_call_seconds': {'native': native_first, 'werk': streamed_first},
        'warm_median_ms': {k: statistics.median(v) for k, v in samples.items()},
        'warm_logical_read_bytes': manager.reader.logical_bytes - before,
        'outputs_allclose_atol_rtol_002': True, 'samples_ms': samples,
        'cache': manager.status(),
    }
    report['warm_slowdown'] = report['warm_median_ms']['werk_cached'] / report['warm_median_ms']['native']
    if reference_manager:
        report['speedup_vs_before'] = report['warm_median_ms']['werk_before'] / report['warm_median_ms']['werk_cached']
        reference_manager.deactivate()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('samples_ms', 'cache')}, indent=2))
    manager.deactivate()


if __name__ == '__main__':
    main()
