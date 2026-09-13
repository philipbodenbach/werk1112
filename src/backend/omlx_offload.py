"""Architecture-specific offload inventory and bounded shared residency.

No MLX import, model construction or model-repository code execution is allowed
in this module. DeepSeek keeps its existing adapter until integration parity is
verified. Byte counts include quantization metadata, not parameter estimates.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import threading
import time
import weakref

MiB = 1024**2
GiB = 1024**3
DTYPES = {'U32': 4, 'I32': 4, 'F32': 4, 'F16': 2, 'BF16': 2,
          'I64': 8, 'U64': 8, 'I16': 2, 'U16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
EXPERT = re.compile(r'^(?:language_model\.)?model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
PLE = re.compile(r'^(?:language_model\.)?model\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.(?:shard_(\d+)|shards\.(\d+))\.(weight|scales|biases)$')


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path, limit=4*MiB):
    with Path(path).open('rb') as source:
        raw = source.read(limit+1)
    if len(raw) > limit:
        raise ValueError(f'metadata exceeds {limit} bytes')
    value = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError('metadata must be an object')
    return value, raw


def signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True)
class Tensor:
    path: Path
    offset: int
    size: int
    shape: tuple
    dtype: str
    signature: tuple

    def rows(self, start, count=1):
        integer(start, 'row start')
        integer(count, 'row count', 1)
        if not self.shape or start + count > self.shape[0]:
            raise ValueError('tensor row range out of bounds')
        stride = self.size // self.shape[0]
        return Tensor(self.path, self.offset + start*stride, count*stride,
                      (count, *self.shape[1:]), self.dtype, self.signature)


class Inventory:
    """Inspect complete shards only; never evaluate or read their tensor payloads."""
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.config, raw = read_json(self.path/'config.json')
        self.architecture = self.config.get('model_type')
        if self.architecture not in ('qwen4_exp', 'glm5_next'):
            raise ValueError('offload inventory requires qwen4_exp or glm5_next')
        text = self.config.get('text_config')
        if not isinstance(text, dict):
            raise ValueError('nested text_config is required')
        self.layers = integer(text.get('num_hidden_layers'), 'layers', 1)
        self.experts = integer(text.get('num_experts') if self.architecture == 'qwen4_exp'
                               else text.get('n_routed_experts'), 'experts', 1)
        self.hidden = integer(text.get('hidden_size'), 'hidden size', 1)
        self.intermediate = integer(text.get('moe_intermediate_size'), 'MoE width', 1)
        self.top_k = integer(text.get('num_experts_per_tok'), 'active experts', 1)
        if self.layers > 256 or self.experts > 4096 or self.top_k > self.experts:
            raise ValueError('unsupported offload geometry')
        self.tensors = {}
        digest = hashlib.sha256(raw)
        shards = sorted(self.path.glob('model*.safetensors'))
        if not shards or len(shards) > 1024:
            raise ValueError('complete local model*.safetensors shards required')
        for shard in shards:
            if not shard.resolve().is_relative_to(self.path):
                raise ValueError('shard escapes checkpoint directory')
            with shard.open('rb') as source:
                stat = os.fstat(source.fileno())
                header_length = source.read(8)
                if len(header_length) != 8:
                    raise ValueError('incomplete safetensors header')
                length = struct.unpack('<Q', header_length)[0]
                if not 2 <= length <= min(64*MiB, stat.st_size-8):
                    raise ValueError('invalid safetensors header size')
                header_raw = source.read(length)
            header = json.loads(header_raw, object_pairs_hook=unique_object)
            if not isinstance(header, dict):
                raise ValueError('invalid safetensors header')
            digest.update(shard.name.encode()); digest.update(header_raw)
            digest.update(repr(signature(stat)).encode())
            ranges = []
            for name, info in header.items():
                if name == '__metadata__':
                    continue
                if name in self.tensors or not isinstance(info, dict):
                    raise ValueError('duplicate or invalid tensor')
                shape, dtype, offsets = info.get('shape'), info.get('dtype'), info.get('data_offsets')
                if not isinstance(shape, list) or any(type(v) is not int or v <= 0 for v in shape) or dtype not in DTYPES:
                    raise ValueError(f'unsupported shape/dtype: {name}')
                if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int or v < 0 for v in offsets):
                    raise ValueError('invalid tensor offsets')
                start, end = offsets
                size = math.prod(shape)*DTYPES[dtype]
                if end-start != size or end > stat.st_size-8-length:
                    raise ValueError(f'incomplete or invalid tensor: {name}')
                ranges.append((start,end))
                self.tensors[name] = Tensor(shard, start+8+length, size, tuple(shape), dtype, signature(stat))
            ranges.sort()
            if any(a[1] > b[0] for a,b in zip(ranges,ranges[1:])):
                raise ValueError('overlapping tensor ranges')
        index = self.path/'model.safetensors.index.json'
        if index.exists():
            metadata, raw = read_json(index, 16*MiB)
            mapping = metadata.get('weight_map')
            if not isinstance(mapping, dict) or set(mapping) != set(self.tensors):
                raise ValueError('index does not match complete shard tensor set')
            for name, filename in mapping.items():
                if filename != self.tensors[name].path.name:
                    raise ValueError('index points to a wrong or missing shard')
            digest.update(raw)
        self.fingerprint = digest.hexdigest()
        self.categories = {key: {} for key in ('base','experts','ple','vision','mtp')}
        for name, tensor in self.tensors.items():
            if re.search(r'(^|\.)mtp\.', name):
                category = 'mtp'
            elif EXPERT.fullmatch(name):
                category = 'experts'
            elif PLE.fullmatch(name):
                category = 'ple'
            elif name.startswith(('vision_tower.', 'vision_model.', 'visual.')):
                category = 'vision'
            else:
                if '.switch_mlp.' in name or '.experts.' in name or '.ngram_embedding.' in name and not name.endswith('weight_scale'):
                    raise ValueError(f'unrecognized offload tensor: {name}')
                category = 'base'
            self.categories[category][name] = tensor
        self.projections = {}
        self.layer_bytes = {}
        start = 0 if self.architecture == 'qwen4_exp' else integer(text.get('first_k_dense_replace'), 'dense layers')
        if start >= self.layers:
            raise ValueError('checkpoint has no routed layers')
        prefixes = {EXPERT.fullmatch(n).group(1): n.split('.mlp.switch_mlp.')[0]
                    for n in self.categories['experts']}
        if set(map(int,prefixes)) != set(range(start,self.layers)):
            raise ValueError('missing or unexpected routed layer')
        for layer in range(start,self.layers):
            size = 0
            for projection in ('gate_proj','up_proj','down_proj'):
                prefix = prefixes[str(layer)] + '.mlp.switch_mlp.' + projection
                inputs, outputs = (self.intermediate,self.hidden) if projection == 'down_proj' else (self.hidden,self.intermediate)
                q = self.quantization(prefix)
                bits, group = q['bits'], q['group_size']
                if inputs % group or inputs*bits % 32:
                    raise ValueError('invalid packed expert geometry')
                for suffix in ('weight','scales','biases'):
                    tensor = self.tensors.get(prefix+'.'+suffix)
                    width = inputs*bits//32 if suffix == 'weight' else inputs//group
                    if tensor is None or tensor.shape != (self.experts,outputs,width) or tensor.dtype not in (('U32',) if suffix=='weight' else ('F16','BF16')):
                        raise ValueError(f'invalid packed expert: {prefix}.{suffix}')
                    size += tensor.size//self.experts
                self.projections[layer,projection] = (prefix,q)
            self.layer_bytes[layer] = size
        if self.architecture == 'glm5_next' and self.categories['ple']:
            raise ValueError('GLM has no verified PLE layout')
        self.ple_tables = {}
        layers = text.get('ple_layer_ids', [])
        if self.architecture == 'qwen4_exp':
            if not isinstance(layers, list) or any(type(layer) is not int for layer in layers) or len(set(layers)) != len(layers):
                raise ValueError('invalid PLE layer IDs')
        if self.architecture == 'qwen4_exp' and layers:
            heads = integer(text.get('heads_per_ngram'), 'PLE heads', 1)
            grams = integer(text.get('ngram_size'), 'ngram size', 2)-1
            width = integer(text.get('ple_embed_dim'), 'PLE width', 1)
            if width % (heads*grams): raise ValueError('invalid PLE head width')
            dims = width//(heads*grams)
            shards = integer(text.get('split_ngram_parts'), 'PLE shards', 1)
            for number in layers:
                layer = integer(number, 'PLE layer', 1)-1
                if layer >= self.layers: raise ValueError('PLE layer out of range')
                selected = {}
                for name in self.categories['ple']:
                    match = PLE.fullmatch(name)
                    if int(match[1]) == layer and match[4] == 'weight':
                        shard = int(match[2] or match[3])
                        if shard in selected: raise ValueError('duplicate PLE shard alias')
                        selected[shard] = name[:-len('.weight')]
                if set(selected) != set(range(shards)): raise ValueError('incomplete PLE table')
                specs=[]
                offset=0
                for shard in range(shards):
                    prefix=selected[shard];q=self.quantization(prefix)
                    if dims % q['group_size'] or dims*q['bits'] % 32:
                        raise ValueError('invalid PLE quantization geometry')
                    w=self.tensors[prefix+'.weight']
                    if len(w.shape)!=2 or w.dtype!='U32' or w.shape[1]!=dims*q['bits']//32:
                        raise ValueError('invalid packed PLE weight')
                    for suffix in ('scales','biases'):
                        t=self.tensors.get(prefix+'.'+suffix)
                        if t is None or t.shape!=(w.shape[0],dims//q['group_size']) or t.dtype not in ('F16','BF16'):
                            raise ValueError('invalid PLE scale/bias geometry')
                    specs.append((offset,offset+w.shape[0],prefix,q));offset+=w.shape[0]
                self.ple_tables[layer]=(dims,specs)
        present={int(PLE.fullmatch(n)[1]) for n in self.categories['ple']}
        if present!=set(self.ple_tables): raise ValueError('unexpected PLE layer')

    def quantization(self, prefix):
        q = self.config.get('quantization')
        if not isinstance(q,dict):
            raise ValueError('affine quantization metadata required')
        bare = prefix.removeprefix('language_model.')
        aliases = (bare, 'language_model.' + bare)
        overrides = [q[name] for name in aliases if name in q]
        if len(overrides) == 2 and overrides[0] != overrides[1]:
            raise ValueError('conflicting quantization namespace aliases')
        override = overrides[0] if overrides else {}
        if not isinstance(override,dict):
            raise ValueError('invalid quantization override')
        result = {key: override.get(key,q.get(key,default)) for key,default in
                  (('bits',None),('group_size',None),('mode','affine'))}
        if type(result['bits']) is not int or result['bits'] not in (2,4,8) or type(result['group_size']) is not int or result['group_size'] not in (32,64,128) or result['mode']!='affine':
            raise ValueError('unsupported affine quantization')
        return result

    def summary(self):
        sizes = {key:sum(t.size for t in values.values()) for key,values in self.categories.items()}
        return {'architecture':self.architecture, 'fingerprint':self.fingerprint,
                'bytes':sizes, 'checkpoint_bytes':sum(sizes.values()),
                'largest_expert_bytes':max(self.layer_bytes.values()),
                'routed_layers':len(self.layer_bytes), 'experts_per_layer':self.experts,
                'repository_code_required':bool(self.config.get('model_file')),
                'execution_verified':False}


def automatic_budget(*, metal_limit, available_memory, base_bytes, state_bytes,
                     workspace_bytes, expert_bytes, ple_bytes, minimum_expert_bytes):
    """One combined weight-cache allowance; no double spending by PLE/experts."""
    for name,value in locals().copy().items():
        integer(value,name)
    if not metal_limit or not minimum_expert_bytes or expert_bytes < minimum_expert_bytes:
        raise ValueError('invalid memory limits or expert inventory')
    # Integer arithmetic also stays exact on multi-TiB hosts.
    usable = min(metal_limit*9//10, max(0,available_memory-2*GiB))
    cache = min(expert_bytes+ple_bytes, usable-base_bytes-state_bytes-workspace_bytes)
    if cache < minimum_expert_bytes:
        raise ValueError('base, states, workspace and one expert exceed available memory')
    return cache


class SharedCache:
    """Bounded synchronous cache with namespace caps and reference-counted leases.

    Callers account retained arrays AND in-flight staging in the entry size.
    GPU callers must evaluate dependent work before leaving a lease. Loads are
    serialized, preventing two concurrent misses from spending the same bytes.
    """
    def __init__(self, budget, limits=None):
        self.budget = integer(budget,'cache budget',1)
        self.limits = dict(limits or {})
        for value in self.limits.values(): integer(value,'namespace budget')
        self.entries = OrderedDict()
        self.leases = {}
        self.pending = {}
        self.used = 0
        self.stats = {}
        self.lock = threading.RLock()

    def _stat(self, namespace):
        return self.stats.setdefault(namespace, {'hits':0,'misses':0,'evictions':0,'bytes':0})

    def _evict_for(self, namespace, size):
        stats = self._stat(namespace)
        cap = self.limits.get(namespace,self.budget)
        if size > min(cap,self.budget): raise ValueError('entry exceeds cache budget')
        while self.used+size > self.budget or stats['bytes']+size > cap:
            own = stats['bytes']+size > cap
            victim = next((key for key in self.entries if not self.leases.get(key)
                           and (not own or key[0]==namespace)),None)
            if victim is None: raise ValueError('active cache leases prevent admission')
            _,count = self.entries.pop(victim)
            previous = self._stat(victim[0]);previous['bytes']-=count;previous['evictions']+=1
            self.used-=count

    @contextmanager
    def acquire(self, namespace, key, size, loader):
        integer(size,'entry bytes',1)
        identity = (namespace,key)
        with self.lock:
            if identity in self.pending:
                raise ValueError('recursive load of the same cache entry')
            stats = self._stat(namespace)
            if identity not in self.entries:
                self._evict_for(namespace,size)
                # Reserve before loader execution, including recursive acquisitions.
                self.used+=size;stats['bytes']+=size
                self.pending[identity]=size
                try: value = loader()
                except BaseException:
                    self.used-=size;stats['bytes']-=size
                    raise
                finally:
                    del self.pending[identity]
                self.entries[identity]=(value,size);stats['misses']+=1
            else:
                value,old_size=self.entries[identity]
                if old_size != size: raise ValueError('cache key reused with different byte size')
                stats['hits']+=1;self.entries.move_to_end(identity)
            self.leases[identity]=self.leases.get(identity,0)+1
        try: yield value
        finally:
            with self.lock:
                self.leases[identity]-=1
                if not self.leases[identity]: del self.leases[identity]

    def resize(self, budget):
        integer(budget,'cache budget',1)
        with self.lock:
            protected=sum(self.entries[key][1] for key in self.leases)+sum(self.pending.values())
            if budget < protected: raise ValueError('new budget cannot evict active leases')
            self.budget=budget
            while self.used>budget:
                victim=next(key for key in self.entries if not self.leases.get(key))
                _,size=self.entries.pop(victim)
                stats=self._stat(victim[0]);stats['bytes']-=size;stats['evictions']+=1
                self.used-=size

    def snapshot(self):
        with self.lock:
            return {'budget_bytes':self.budget,'resident_bytes':self.used,
                    'namespaces':{key:dict(value) for key,value in self.stats.items()}}


class RangeReader:
    def __init__(self, max_open_files=16):
        self.max_open_files = integer(max_open_files, 'open file limit', 1)
        self._fds = OrderedDict()
        self._lock = threading.RLock()
        self._closed = False
        self._prefetched = None
        self._finalizer = weakref.finalize(self, self._close_files, self._fds)
        self.logical_bytes=0
        self.calls=0
        self.read_seconds=0.0

    @staticmethod
    def _close_files(fds):
        while fds:
            _, fd = fds.popitem()
            os.close(fd)

    def close(self):
        with self._lock:
            self._closed = True
            self._finalizer()

    def _file(self, path):
        if self._closed:
            raise ValueError('checkpoint reader is closed')
        fd = self._fds.get(path)
        if fd is None:
            if len(self._fds) >= self.max_open_files:
                _, old = self._fds.popitem(last=False)
                os.close(old)
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        self._fds.move_to_end(path)
        return fd

    def read(self, tensor):
        started=time.perf_counter()
        try:
            with self._lock:
                if self._prefetched is not None and tensor in self._prefetched:
                    return self._prefetched.pop(tensor)
                fd = self._file(tensor.path)
                if signature(os.fstat(fd)) != tensor.signature or signature(os.stat(tensor.path)) != tensor.signature:
                    raise ValueError('checkpoint changed after inspection')
                # Almost every expert tensor fits in one read. Return that
                # owned buffer directly instead of allocating a second full
                # tensor and copying through a bytearray slice. NumPy/MLX keep
                # the buffer alive for their own conversion as before.
                raw = os.pread(fd, min(8*MiB, tensor.size), tensor.offset)
                self.logical_bytes += len(raw); self.calls += 1
                if len(raw) != tensor.size:
                    if not raw and tensor.size:
                        raise ValueError('checkpoint truncated while reading')
                    first = raw
                    raw = bytearray(tensor.size)
                    raw[:len(first)] = first
                    position = len(first)
                    while position < tensor.size:
                        chunk = os.pread(fd, min(8*MiB, tensor.size-position), tensor.offset+position)
                        if not chunk: raise ValueError('checkpoint truncated while reading')
                        raw[position:position+len(chunk)] = chunk
                        position += len(chunk)
                        self.logical_bytes += len(chunk); self.calls += 1
                if signature(os.fstat(fd)) != tensor.signature:
                    raise ValueError('checkpoint changed during reading')
                return raw
        finally:
            self.read_seconds+=time.perf_counter()-started

    @contextmanager
    def prefetch(self, tensors, max_bytes=256*MiB):
        """Read adjacent needed ranges together, with bounded temporary storage.

        Each returned row owns its bytes. A cached MLX expert therefore cannot
        keep an entire group's backing allocation alive after sibling eviction.
        Non-adjacent ranges retain the normal demand-read path.
        """
        tensors = sorted(set(tensors), key=lambda t: (str(t.path), t.offset))
        if sum(t.size for t in tensors) > max_bytes:
            yield
            return
        with self._lock:
            if self._prefetched is not None:
                raise ValueError('nested checkpoint prefetch is unsupported')
            self._prefetched = {}
            try:
                runs = []
                for tensor in tensors:
                    if (runs and runs[-1][-1].path == tensor.path
                            and runs[-1][-1].signature == tensor.signature
                            and runs[-1][-1].offset + runs[-1][-1].size == tensor.offset):
                        runs[-1].append(tensor)
                    else:
                        runs.append([tensor])
                for run in runs:
                    if len(run) < 2:
                        continue
                    size = sum(t.size for t in run)
                    combined = Tensor(run[0].path, run[0].offset, size, (size,), 'U8', run[0].signature)
                    raw = self.read(combined)
                    copy_started = time.perf_counter()
                    offset = 0
                    for tensor in run:
                        self._prefetched[tensor] = bytes(raw[offset:offset+tensor.size])
                        offset += tensor.size
                    self.read_seconds += time.perf_counter() - copy_started
                yield
            finally:
                self._prefetched = None
