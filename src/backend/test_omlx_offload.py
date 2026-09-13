"""Qwen/GLM header, shared-budget and opt-in native numerical tests."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch


def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result

offload=module('omlx_offload')
runtime=module('omlx_offload_runtime')


def config(arch='qwen4_exp'):
    text={'num_hidden_layers':2,'num_experts':4,'n_routed_experts':4,'hidden_size':64,
          'moe_intermediate_size':64,'num_experts_per_tok':2,'first_k_dense_replace':1,
          'ple_layer_ids':[1],'heads_per_ngram':1,'ngram_size':2,'ple_embed_dim':64,'split_ngram_parts':2}
    return {'model_type':arch,'text_config':text,'quantization':{'bits':4,'group_size':32,'mode':'affine'}}


def weights_layout(cfg):
    prefix='' if cfg['model_type']=='qwen4_exp' else 'language_model.'
    start=0 if not prefix else 1
    for layer in range(start,2):
        for proj in ('gate_proj','up_proj','down_proj'):
            name=f'{prefix}model.layers.{layer}.mlp.switch_mlp.{proj}'
            q=cfg['quantization'].get(name,cfg['quantization'])
            yield name,(4,64,64),q['bits']
    if not prefix:
        for shard in range(2): yield f'model.layers.0.ple.ple_embedding.ngram_embedding.shard_{shard}',(5+shard,64),4


def write_fixture(path,arch='qwen4_exp',override=None):
    cfg=config(arch)
    if override: override(cfg)
    header={};data=bytearray()
    for prefix,shape,bits in weights_layout(cfg):
        for suffix in ('weight','scales','biases'):
            dims=(*shape[:-1],shape[-1]*bits//32 if suffix=='weight' else shape[-1]//32)
            size=1
            for dim in dims: size*=dim
            dtype='U32' if suffix=='weight' else 'F16';size*=4 if suffix=='weight' else 2
            header[prefix+'.'+suffix]={'shape':list(dims),'dtype':dtype,'data_offsets':[len(data),len(data)+size]}
            data.extend(bytes(size))
    header['model.norm.weight']={'shape':[64],'dtype':'F16','data_offsets':[len(data),len(data)+128]};data.extend(bytes(128))
    (path/'config.json').write_text(json.dumps(cfg))
    raw=json.dumps(header).encode();(path/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+data)
    return cfg


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)

    def test_two_architectures_header_only_inventory(self):
        for arch in ('qwen4_exp','glm5_next'):
            write_fixture(self.path,arch)
            with patch.object(os,'pread',side_effect=AssertionError('no payload reads')):
                inv=offload.Inventory(self.path)
            summary=inv.summary()
            self.assertEqual(sum(summary['bytes'].values()),sum(t.size for t in inv.tensors.values()))
            self.assertEqual(summary['routed_layers'],2 if arch=='qwen4_exp' else 1)
            self.assertEqual(summary['bytes']['base'],128)
            self.assertEqual(bool(inv.ple_tables),arch=='qwen4_exp')
            self.assertFalse(summary['execution_verified'])

    def test_per_projection_quantization_and_mtp_are_not_global(self):
        key='language_model.model.layers.1.mlp.switch_mlp.gate_proj'
        write_fixture(self.path,'glm5_next',lambda c:c['quantization'].update({key:{'bits':8,'group_size':32}}))
        inv=offload.Inventory(self.path)
        self.assertEqual(inv.projections[1,'gate_proj'][1]['bits'],8)
        self.assertEqual(inv.projections[1,'up_proj'][1]['bits'],4)

    def test_partial_download_rejected(self):
        write_fixture(self.path)
        shard=self.path/'model.safetensors'
        with shard.open('r+b') as stream: stream.truncate(shard.stat().st_size-1)
        with self.assertRaisesRegex(ValueError,'incomplete'): offload.Inventory(self.path)

    def test_quantization_alias_matches_native_tensor_namespace(self):
        key='model.layers.1.mlp.switch_mlp.gate_proj'
        write_fixture(self.path,'glm5_next',lambda c:c['quantization'].update({key:{'bits':4,'group_size':32}}))
        inv=offload.Inventory(self.path)
        inv.config['quantization'][key]['bits']=8
        self.assertEqual(inv.quantization('language_model.'+key)['bits'],8)
        inv.config['quantization']['language_model.'+key]={'bits':2,'group_size':32}
        with self.assertRaisesRegex(ValueError,'conflicting'):inv.quantization(key)

    def test_undeclared_ple_tensors_are_rejected(self):
        write_fixture(self.path,override=lambda c:c['text_config'].update(ple_layer_ids=[]))
        with self.assertRaisesRegex(ValueError,'unexpected PLE'):offload.Inventory(self.path)

    def test_index_must_match_shards(self):
        write_fixture(self.path)
        (self.path/'model.safetensors.index.json').write_text('{"weight_map":{"missing":"model.safetensors"}}')
        with self.assertRaisesRegex(ValueError,'index'): offload.Inventory(self.path)

    def test_deepseek_not_silently_accepted_by_new_adapter(self):
        write_fixture(self.path)
        cfg=config();cfg['model_type']='deepseek_v4';(self.path/'config.json').write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'requires qwen'): offload.Inventory(self.path)

    def test_missing_ple_shards_rejected(self):
        write_fixture(self.path,override=lambda c:c['text_config'].update(split_ngram_parts=3))
        with self.assertRaisesRegex(ValueError,'incomplete PLE'): offload.Inventory(self.path)

    def test_partial_rows_and_replacement_invalidate_reader(self):
        write_fixture(self.path);inv=offload.Inventory(self.path)
        tensor=next(iter(inv.tensors.values()));reader=offload.RangeReader()
        row=tensor.rows(2)
        self.assertEqual(len(reader.read(row)),tensor.size//4)
        self.assertEqual(reader.logical_bytes,tensor.size//4)
        with self.assertRaisesRegex(ValueError,'bounds'):tensor.rows(4)
        temporary=self.path/'replacement';temporary.write_bytes(tensor.path.read_bytes());temporary.replace(tensor.path)
        with self.assertRaisesRegex(ValueError,'changed'):reader.read(row)


class RangeReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def tensor(self, name, data):
        path = self.path / name
        path.write_bytes(b'header-padding' + data)
        return offload.Tensor(path, 14, len(data), (len(data),), 'U8', offload.signature(path.stat()))

    def test_owned_reads_reuse_handles_and_survive_lru_close(self):
        first = self.tensor('one', bytes(range(256)) * 4)
        second = self.tensor('two', b'other contents')
        reader = offload.RangeReader(max_open_files=1)
        self.addCleanup(reader.close)
        with patch.object(os, 'open', wraps=os.open) as opened:
            content = reader.read(first)
            self.assertEqual(reader.read(first), content)
            self.assertEqual(opened.call_count, 1)
            self.assertEqual(reader.read(second), b'other contents')
            # The OS may reuse the descriptor number; the old buffer still
            # belongs to its caller, independently of the cached file handle.
            self.assertEqual(content, bytes(range(256)) * 4)
            self.assertEqual(reader.read(first), content)
            self.assertEqual(opened.call_count, 3)
        fd = next(iter(reader._fds.values()))
        reader.close()
        reader.close()
        with self.assertRaises(OSError): os.fstat(fd)
        with self.assertRaisesRegex(ValueError, 'closed'): reader.read(first)

    def test_short_reads_preserve_offsets_and_content(self):
        data = bytes(range(256)) * 4
        tensor = self.tensor('short', data)
        reader = offload.RangeReader()
        self.addCleanup(reader.close)
        original = os.pread
        with patch.object(os, 'pread', side_effect=lambda fd, size, offset: original(fd, min(size, 17), offset)):
            self.assertEqual(reader.read(tensor), data)
        self.assertEqual(reader.logical_bytes, len(data))

    def test_adjacent_experts_share_a_read_without_changing_bytes_or_lifetimes(self):
        expected = [bytes([index]) * 256 for index in range(4)]
        data = b''.join(expected)
        tensor = self.tensor('adjacent', data)
        tensor = offload.Tensor(tensor.path, tensor.offset, tensor.size, (4, 256), 'U8', tensor.signature)
        rows = [tensor.rows(i) for i in range(4)]
        reader = offload.RangeReader()
        self.addCleanup(reader.close)
        with reader.prefetch(rows):
            result = [reader.read(row) for row in rows]
        self.assertEqual(result, expected)
        self.assertEqual(reader.calls, 1)
        self.assertEqual(reader.logical_bytes, len(data))
        self.assertIsNone(reader._prefetched)
        reader.close()
        self.assertEqual(b''.join(result), data)

    def test_sparse_ranges_and_small_staging_budget_do_not_overread(self):
        tensor = self.tensor('sparse', bytes(range(256)) * 4)
        tensor = offload.Tensor(tensor.path, tensor.offset, tensor.size, (4, 256), 'U8', tensor.signature)
        for indices, limit in [([0, 2], 1024), ([0, 1], 1)]:
            reader = offload.RangeReader()
            self.addCleanup(reader.close)
            rows = [tensor.rows(i) for i in indices]
            with reader.prefetch(rows, max_bytes=limit):
                self.assertEqual([reader.read(row) for row in rows], [bytes(range(256))] * 2)
            self.assertEqual(reader.calls, 2)
            self.assertEqual(reader.logical_bytes, 512)

    def test_prefetch_failure_clears_temporary_state_for_retry(self):
        tensor = self.tensor('retry', b'a' * 1024)
        tensor = offload.Tensor(tensor.path, tensor.offset, tensor.size, (4, 256), 'U8', tensor.signature)
        reader = offload.RangeReader()
        self.addCleanup(reader.close)
        rows = [tensor.rows(0), tensor.rows(1)]
        with patch.object(os, 'pread', side_effect=OSError('injected read failure')):
            with self.assertRaisesRegex(OSError, 'injected'):
                with reader.prefetch(rows): pass
        self.assertIsNone(reader._prefetched)
        with reader.prefetch(rows): self.assertEqual(reader.read(rows[0]), b'a' * 256)

    def test_mutation_during_cached_read_is_rejected(self):
        tensor = self.tensor('mutated', b'a' * 1024)
        reader = offload.RangeReader()
        self.addCleanup(reader.close)
        reader.read(tensor)
        original = os.pread
        def mutate(fd, size, offset):
            result = original(fd, size, offset)
            with tensor.path.open('ab') as output: output.write(b'changed')
            return result
        with patch.object(os, 'pread', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during'): reader.read(tensor)


class SharedBudgetTests(unittest.TestCase):
    def test_joint_lru_preserves_active_expert_while_ple_evicts(self):
        cache=offload.SharedCache(10)
        with cache.acquire('experts',1,6,lambda:'expert'):
            with cache.acquire('ple',1,4,lambda:'row'): pass
            with cache.acquire('ple',2,4,lambda:'other'): pass
            self.assertEqual(cache.snapshot()['resident_bytes'],10)
            self.assertEqual(cache.snapshot()['namespaces']['ple']['evictions'],1)
            with self.assertRaisesRegex(ValueError,'leases'):
                with cache.acquire('experts',2,7,lambda:'blocked'):pass

    def test_namespace_caps_keep_manual_expert_limit(self):
        cache=offload.SharedCache(20,{'experts':8})
        with cache.acquire('experts',1,6,lambda:1):pass
        with cache.acquire('ple',1,6,lambda:2):pass
        with cache.acquire('experts',2,6,lambda:3):pass
        self.assertEqual(cache.snapshot()['resident_bytes'],12)
        self.assertEqual(cache.snapshot()['namespaces']['ple']['evictions'],0)

    def test_nested_leases_exception_and_shrink(self):
        cache=offload.SharedCache(10)
        with cache.acquire('experts',1,6,lambda:object()) as a:
            with cache.acquire('experts',1,6,lambda:self.fail('cache hit reloaded')) as b:self.assertIs(a,b)
            with self.assertRaisesRegex(ValueError,'leases'):cache.resize(5)
        cache.resize(5);self.assertEqual(cache.used,0)
        def broken():raise RuntimeError('failed I/O')
        with self.assertRaises(RuntimeError):
            with cache.acquire('ple',1,4,broken):pass
        self.assertEqual(cache.used,0)
        self.assertFalse(cache.pending)

    def test_load_reservation_cannot_be_spent_twice(self):
        cache=offload.SharedCache(10)
        def load():
            with self.assertRaisesRegex(ValueError,'leases'):cache.resize(4)
            with self.assertRaisesRegex(ValueError,'leases'):
                with cache.acquire('ple',1,6,lambda:1):pass
            return 2
        with cache.acquire('experts',1,6,load):pass
        self.assertEqual(cache.used,6)

    def test_auto_scales_and_reserves_state_memory(self):
        args=dict(metal_limit=36*offload.GiB,available_memory=40*offload.GiB,base_bytes=5*offload.GiB,
                  state_bytes=2*offload.GiB,workspace_bytes=offload.GiB,expert_bytes=70*offload.GiB,
                  ple_bytes=30*offload.GiB,minimum_expert_bytes=8*offload.MiB)
        value=offload.automatic_budget(**args)
        self.assertLess(value,28*offload.GiB)
        self.assertEqual(offload.automatic_budget(**{**args,'state_bytes':3*offload.GiB}),value-offload.GiB)
        self.assertEqual(offload.automatic_budget(**{**args,'metal_limit':4*1024*offload.GiB,'available_memory':4*1024*offload.GiB}),100*offload.GiB)
        with self.assertRaisesRegex(ValueError,'exceed'):offload.automatic_budget(**{**args,'available_memory':4*offload.GiB})


@unittest.skipUnless(os.getenv('WERK_TEST_MLX_EXPERTS')=='1','native Metal test opt-in')
class NativeOffloadTests(unittest.TestCase):
    def test_streaming_experts_and_ple_match_resident_quantized_reference(self):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        mx.random.seed(73)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);cfg=config();weights={}
            for prefix,shape,bits in weights_layout(cfg):
                values=mx.random.normal(shape).astype(mx.float16)*0.1
                packed=mx.quantize(values,group_size=32,bits=bits)
                for suffix,value in zip(('weight','scales','biases'),packed):weights[prefix+'.'+suffix]=value
            weights['model.norm.weight']=mx.ones((64,),dtype=mx.float16)
            mx.save_safetensors(str(path/'model.safetensors'),weights)
            (path/'config.json').write_text(json.dumps(cfg))
            inv=offload.Inventory(path)
            # Only one resident expert fits: every route change must evict.
            cache=offload.SharedCache(2*max(inv.layer_bytes.values())+2048)
            access=runtime.WeightAccess(inv,cache,offload.RangeReader())
            activation=lambda up,gate:up*nn.silu(gate)
            streamed=runtime.streamed_experts(access,0,activation)
            x=mx.random.normal((1,4,64)).astype(mx.float16)
            indices=mx.array([[[0,1],[2,3],[1,0],[3,2]]])
            for _ in range(2):
                actual=streamed(x,indices)
                expected=[]
                for row in range(4):
                    outputs=[]
                    for expert in indices[0,row].tolist():
                        def project(name,value):
                            prefix=f'model.layers.0.mlp.switch_mlp.{name}'
                            w,s,b=(weights[prefix+'.'+suffix][expert] for suffix in ('weight','scales','biases'))
                            return mx.quantized_matmul(value,w,s,b,transpose=True,bits=4,group_size=32)
                        outputs.append(project('down_proj',activation(project('up_proj',x[0,row:row+1]),project('gate_proj',x[0,row:row+1])))[0])
                    expected.append(mx.stack(outputs))
                expected=mx.stack(expected)[None];mx.eval(actual,expected)
                np.testing.assert_allclose(np.array(actual),np.array(expected),atol=1e-4,rtol=1e-3)
            embedding=runtime.streamed_embedding(access,0)
            ids=mx.array([[0,5,10,5],[4,1,8,0]])
            actual=embedding(ids)
            tables=[]
            for shard in range(2):
                prefix=f'model.layers.0.ple.ple_embedding.ngram_embedding.shard_{shard}'
                tables.append(mx.dequantize(*(weights[prefix+'.'+suffix] for suffix in ('weight','scales','biases')),bits=4,group_size=32))
            expected=mx.concatenate(tables)[ids].astype(mx.bfloat16);mx.eval(actual,expected)
            np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),np.array(expected.astype(mx.float32)))
            self.assertGreater(cache.snapshot()['namespaces']['experts']['evictions'],0)
            self.assertEqual(access.ple_requested_rows,8);self.assertEqual(access.ple_unique_rows,6)
            self.assertLessEqual(cache.used,cache.budget)


if __name__=='__main__':unittest.main()
