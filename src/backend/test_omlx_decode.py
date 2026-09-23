"""Numerical/lifetime coverage for text decode; no full model is loaded."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@unittest.skipUnless(os.getenv('WERK_TEST_MLX_EXPERTS') == '1', 'native Metal test opt-in')
class DecodeTests(unittest.TestCase):
    def test_quantized_decode_matches_reference_with_eviction_duplicates_and_prefill(self):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        import omlx_decode
        import omlx_offload as offload
        import omlx_offload_runtime as runtime
        from test_omlx_offload import config, weights_layout

        mx.random.seed(81)
        for architecture in ('glm5_next', 'qwen4_exp'):
            for dtype in (mx.float16, mx.bfloat16):
                with self.subTest(architecture=architecture, dtype=dtype), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    cfg = config(architecture)
                    layer = 1 if architecture == 'glm5_next' else 0
                    prefix = 'language_model.' if architecture == 'glm5_next' else ''
                    for name, bits in (('up_proj', 2), ('gate_proj', 4), ('down_proj', 8)):
                        cfg['quantization'][f'{prefix}model.layers.{layer}.mlp.switch_mlp.{name}'] = {
                            'bits': bits, 'group_size': 32, 'mode': 'affine'}
                    weights = {}
                    for name, shape, bits in weights_layout(cfg):
                        packed = mx.quantize(mx.random.normal(shape).astype(dtype) * .1,
                                             group_size=32, bits=bits)
                        weights.update({name + '.' + suffix: value
                                        for suffix, value in zip(('weight', 'scales', 'biases'), packed)})
                    mx.save_safetensors(str(path / 'model.safetensors'), weights)
                    (path / 'config.json').write_text(json.dumps(cfg))
                    inv = offload.Inventory(path)
                    # Force eviction between single-expert groups.
                    caches = [offload.SharedCache(2 * max(inv.layer_bytes.values()) + 2048) for _ in range(2)]
                    readers = [offload.RangeReader() for _ in range(2)]
                    accesses = [runtime.WeightAccess(inv, cache, reader) for cache, reader in zip(caches, readers)]
                    activation = lambda up, gate: up * nn.silu(gate)
                    baseline = runtime.streamed_experts(accesses[0], layer, activation)
                    candidate = omlx_decode.streamed_decode_experts(accesses[1], layer, activation)
                    try:
                        for rows in (1, 4, 1):
                            x = mx.random.normal((1, rows, 64)).astype(dtype)
                            indices = mx.array([[[3, 1, 3]] * rows], dtype=mx.uint32)
                            scores = mx.array([[[.2, .3, .5]] * rows], dtype=dtype)
                            for weighted in (False, True):
                                expected = baseline(x, indices, scores, weighted)
                                actual = candidate(x, indices, scores, weighted)
                                mx.eval(expected, actual)
                                np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),
                                                              np.array(expected.astype(mx.float32)))
                                self.assertLessEqual(caches[1].used, caches[1].budget)
                                self.assertFalse(caches[1].leases)
                        self.assertGreater(caches[1].snapshot()['namespaces']['experts']['evictions'], 0)

                        # Exercise overlap with cached/cold parts in the same
                        # group. Reference handles the identical split.
                        for cache in caches:
                            cache.resize(2 * max(inv.layer_bytes.values()) * inv.experts + 2048)
                        for access in accesses:
                            access.expert_groups = lambda layer, experts: [list(experts)]

                            @contextmanager
                            def prefetch(layer, group, ready):
                                ready([3])
                                yield

                            access.prefetch_ready_experts = prefetch
                        expected = baseline(x, indices, scores, True)
                        actual = candidate(x, indices, scores, True)
                        mx.eval(expected, actual)
                        np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),
                                                      np.array(expected.astype(mx.float32)))
                        self.assertFalse(caches[1].leases)
                    finally:
                        for reader in readers:
                            reader.close()

    def test_failure_releases_leases_and_invalid_routes_are_rejected(self):
        import mlx.core as mx
        import omlx_decode
        from types import SimpleNamespace
        leased = set()

        @contextmanager
        def expert(layer, index):
            leased.add(index)
            try:
                raise RuntimeError('read failed')
                yield
            finally:
                leased.remove(index)

        access = SimpleNamespace(
            inventory=SimpleNamespace(experts=4, projections={(0, name): ('unused', {})
                for name in ('up_proj', 'gate_proj', 'down_proj')}),
            expert=expert, routing_seconds=0, output_evaluations=0, forward_calls=0, forward_seconds=0)
        candidate = omlx_decode.streamed_decode_experts(access, 0, lambda a, b: a * b)
        x = mx.ones((1, 1, 64))
        with self.assertRaisesRegex(RuntimeError, 'read failed'):
            candidate(x, mx.array([[[1]]]))
        self.assertFalse(leased)
        for route in (-1, 4):
            with self.assertRaisesRegex(ValueError, 'invalid expert routes'):
                candidate(x, mx.array([[[route]]]))
        with self.assertRaisesRegex(ValueError, 'matching scores'):
            candidate(x, mx.array([[[1]]]), weighted_sum=True)


if __name__ == '__main__':
    unittest.main()
