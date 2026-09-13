"""Numerically testable Qwen/GLM offload operations; no loader auto-activation."""
from contextlib import ExitStack
import bisect
import time


def array_from_tensor(reader, tensor, *, evaluate=True):
    import mlx.core as mx
    import numpy as np
    dtypes={'U32':'<u4','I32':'<i4','F32':'<f4','F16':'<f2','BF16':'<u2',
            'I64':'<i8','U64':'<u8','I16':'<i2','U16':'<u2','I8':'i1','U8':'u1','BOOL':'?'}
    raw=reader.read(tensor)
    value=mx.array(np.frombuffer(raw,dtype=dtypes[tensor.dtype]).reshape(tensor.shape))
    if tensor.dtype=='BF16': value=value.view(mx.bfloat16)
    if evaluate:
        mx.eval(value)
    return value


class WeightAccess:
    def __init__(self, inventory, cache, reader):
        self.inventory=inventory
        self.cache=cache
        self.reader=reader
        self.load_seconds=0.0
        self.forward_seconds=0.0
        self.forward_calls=0
        self.output_evaluations=0
        self.routing_seconds=0.0
        self.ple_requested_rows=0
        self.ple_unique_rows=0

    def _read_triplet(self,prefix,row):
        started=time.perf_counter()
        values=tuple(array_from_tensor(self.reader,self.inventory.tensors[prefix+'.'+suffix].rows(row))[0]
                     for suffix in ('weight','scales','biases'))
        self.load_seconds+=time.perf_counter()-started
        return values

    def expert(self,layer,index):
        if type(index) is not int or not 0<=index<self.inventory.experts:
            raise ValueError('expert out of range')
        def load():
            return {projection:self._read_triplet(self.inventory.projections[layer,projection][0],index)
                    for projection in ('gate_proj','up_proj','down_proj')}
        # Reserve packed weights plus their conversion staging. Conservative
        # initially: optimize this allowance only after measured peak accounting.
        size=2*self.inventory.layer_bytes[layer]
        return self.cache.acquire('experts',(layer,index),size,load)

    def ple_row(self,layer,index):
        dims,specs=self.inventory.ple_tables[layer]
        if type(index) is not int or index<0 or index>=specs[-1][1]:
            raise ValueError('PLE index out of range')
        shard=bisect.bisect_right([entry[1] for entry in specs],index)
        start,_,prefix,q=specs[shard]
        packed=sum(self.inventory.tensors[prefix+'.'+suffix].size//self.inventory.tensors[prefix+'.'+suffix].shape[0]
                   for suffix in ('weight','scales','biases'))
        def load():
            import mlx.core as mx
            w,s,b=self._read_triplet(prefix,index-start)
            value=mx.dequantize(w[None],s[None],b[None],**q)[0]
            mx.eval(value)
            return value
        return self.cache.acquire('ple',(layer,index),2*packed+4*dims,load)

    def snapshot(self):
        return {**self.cache.snapshot(),'logical_bytes_read':self.reader.logical_bytes,
                'read_calls':self.reader.calls,'load_seconds':self.load_seconds,
                'forward_seconds':self.forward_seconds,'forward_calls':self.forward_calls,
                'ple_requested_rows':self.ple_requested_rows,'ple_unique_rows':self.ple_unique_rows}


def streamed_experts(access,layer,activation,workspace_bytes=1024**3):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    class StreamedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation=activation
            self._access=access

        def __call__(self,x,indices,scores=None,weighted_sum=False):
            started=time.perf_counter()
            mx.eval(x,indices)
            routes=np.array(indices).reshape(-1,indices.shape[-1])
            access.routing_seconds+=time.perf_counter()-started
            flat=x.reshape(-1,x.shape[-1])
            if routes.shape[0]!=flat.shape[0] or routes.dtype.kind not in 'iu' or np.any(routes<0) or np.any(routes>=access.inventory.experts):
                raise ValueError('invalid expert routes')
            output_size=routes.size*x.shape[-1]*x.itemsize
            if output_size>workspace_bytes//2: raise ValueError('expert output exceeds workspace')
            output=mx.zeros((routes.size,x.shape[-1]),dtype=x.dtype)
            mx.eval(output)
            # Native activation/dtype and routing order stay unchanged. The
            # owner's grouping includes pins and falls back to one expert for
            # small budgets. Complete the graph before releasing any group.
            unique=np.unique(routes)
            groups=access.expert_groups(layer,unique) if hasattr(access,'expert_groups') else ([int(i)] for i in unique)
            for group in groups:
                chunk=max(1,min(128,workspace_bytes//max(1,16*len(group)*(access.inventory.hidden+access.inventory.intermediate))))
                with ExitStack() as leases:
                    if hasattr(access, 'prefetch_experts'):
                        leases.enter_context(access.prefetch_experts(layer, group))
                    acquired=[]
                    for expert in group:
                        rows,slots=np.nonzero(routes==expert)
                        weights=leases.enter_context(access.expert(layer,int(expert)))
                        acquired.append((weights,rows,slots))
                    for offset in range(0,max(len(rows) for _,rows,_ in acquired),chunk):
                        for weights,rows,slots in acquired:
                            r=rows[offset:offset+chunk];s=slots[offset:offset+chunk]
                            if not len(r):continue
                            selected=flat[mx.array(r)]
                            def project(name,value):
                                w,scales,biases=weights[name]
                                q=access.inventory.projections[layer,name][1]
                                return mx.quantized_matmul(value,w,scales,biases,transpose=True,**q)
                            up=project('up_proj',selected);gate=project('gate_proj',selected)
                            value=project('down_proj',self.activation(up,gate)).astype(x.dtype)
                            output=output.at[mx.array(r*routes.shape[1]+s)].add(value)
                        mx.eval(output)
                        access.output_evaluations+=1
                    del weights, acquired
            output=output.reshape(*indices.shape,x.shape[-1])
            if weighted_sum:
                if scores is None or scores.shape!=indices.shape: raise ValueError('weighted experts require matching scores')
                output=(output*scores[...,None]).sum(-2).astype(x.dtype)
            mx.eval(output)
            access.forward_calls+=1;access.forward_seconds+=time.perf_counter()-started
            return output
    return StreamedExperts()


def streamed_embedding(access,layer):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    class StreamedEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self._access=access
            self.weight_scale=mx.ones((1,),dtype=mx.bfloat16)

        def __call__(self,indices):
            mx.eval(indices)
            host=np.array(indices)
            if host.dtype.kind not in 'iu': raise ValueError('PLE indices must be integers')
            dims,specs=access.inventory.ple_tables[layer]
            if host.size and (host.min()<0 or host.max()>=specs[-1][1]): raise ValueError('PLE index out of range')
            if host.size*dims*2>64*1024**2: raise ValueError('PLE output exceeds bounded workspace')
            unique,inverse=np.unique(host.reshape(-1),return_inverse=True)
            access.ple_requested_rows+=host.size;access.ple_unique_rows+=len(unique)
            result=mx.zeros((host.size,dims),dtype=mx.bfloat16)
            mx.eval(result)
            for slot,index in enumerate(unique):
                with access.ple_row(layer,int(index)) as row:
                    positions=np.flatnonzero(inverse==slot)
                    result=result.at[mx.array(positions)].add(row.astype(mx.bfloat16)*self.weight_scale)
                    # Complete all dependent work before permitting eviction.
                    mx.eval(result)
            return result.reshape(*indices.shape,dims)
    return StreamedEmbedding()
