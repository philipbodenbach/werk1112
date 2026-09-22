"""Single-token text expert execution; existing grouped prefill stays intact.

Only small, evaluated expert outputs outlive the group's weight leases. Weights
are never stacked, copied into a second cache, or retained by a compiled graph.
"""
from contextlib import ExitStack
import time


def streamed_decode_experts(access, layer, activation, workspace_bytes=1024**3):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    try:
        from _werk_omlx_offload_runtime import streamed_experts
    except ImportError:
        from omlx_offload_runtime import streamed_experts

    legacy = streamed_experts(access, layer, activation, workspace_bytes)
    quantization = {name: access.inventory.projections[layer, name][1]
                    for name in ('up_proj', 'gate_proj', 'down_proj')}

    class DecodeExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = activation
            self._access = access
            self._prefill = legacy

        def __call__(self, x, indices, scores=None, weighted_sum=False):
            if x.size != x.shape[-1]:
                return self._prefill(x, indices, scores, weighted_sum)
            started = time.perf_counter()
            mx.eval(x, indices)
            routes = np.array(indices).reshape(-1, indices.shape[-1])
            access.routing_seconds += time.perf_counter() - started
            if (routes.shape[0] != 1 or routes.dtype.kind not in 'iu'
                    or np.any(routes < 0) or np.any(routes >= access.inventory.experts)):
                raise ValueError('invalid expert routes')
            if routes.size * x.shape[-1] * x.itemsize > workspace_bytes // 2:
                raise ValueError('expert output exceeds workspace')
            if weighted_sum and (scores is None or scores.shape != indices.shape):
                raise ValueError('weighted experts require matching scores')
            flat = x.reshape(1, x.shape[-1])
            unique = np.unique(routes)
            groups = (access.expert_groups(layer, unique) if hasattr(access, 'expert_groups')
                      else ([int(index)] for index in unique))
            outputs = {}
            for group in groups:
                with ExitStack() as leases:
                    def evaluate(experts):
                        pending = []
                        for expert in experts:
                            expert = int(expert)
                            weights = leases.enter_context(access.expert(layer, expert))

                            def project(name, value):
                                w, scales, biases = weights[name]
                                return mx.quantized_matmul(
                                    value, w, scales, biases, transpose=True,
                                    **quantization[name])

                            up = project('up_proj', flat)
                            gate = project('gate_proj', flat)
                            value = project('down_proj', self.activation(up, gate)).astype(x.dtype)
                            outputs[expert] = value
                            pending.append(value)
                        if pending:
                            # Finish weight-dependent GPU work before the leases
                            # close. Only one small output row per expert remains.
                            mx.eval(*pending)
                            access.output_evaluations += 1

                    if hasattr(access, 'prefetch_ready_experts'):
                        leases.enter_context(access.prefetch_ready_experts(layer, group, ready=evaluate))
                    elif hasattr(access, 'prefetch_experts'):
                        leases.enter_context(access.prefetch_experts(layer, group))
                    evaluate(expert for expert in group if int(expert) not in outputs)

            # Restore original route order, including duplicate experts, with
            # no zero buffer, per-group gather, or scatter-add kernel.
            output = mx.concatenate([outputs[int(expert)] for expert in routes[0]], axis=0)
            output = output.reshape(*indices.shape, x.shape[-1])
            if weighted_sum:
                output = (output * scores[..., None]).sum(-2).astype(x.dtype)
                mx.eval(output)
            access.forward_calls += 1
            access.forward_seconds += time.perf_counter() - started
            return output

    return DecodeExperts()
