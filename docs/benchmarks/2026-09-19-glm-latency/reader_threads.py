#!/usr/bin/env python3
"""Reader-only thread sweep; OS cache is uncontrolled, not full-model throughput."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/backend'))
from omlx_offload import Inventory, RangeReader
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('model', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
inv = Inventory(a.model)
rng = random.Random(20260919)
groups = []
for layer in list(inv.layer_bytes)[::2]:
    group = []
    for expert in rng.sample(range(inv.experts), 3):
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            prefix = inv.projections[layer, projection][0]
            group.extend(inv.tensors[prefix + '.' + suffix].rows(expert)
                         for suffix in ('weight', 'scales', 'biases'))
    groups.append(group)
rows = []
# Same work in reversed order. No OS cache flushing or model load.
for workers in (4, 8, 16, 16, 8, 4):
    reader = RangeReader(prefetch_workers=4)
    # Override only the benchmark pool; production bounds remain unchanged.
    reader._read_pool = ThreadPoolExecutor(max_workers=workers)
    samples = []
    try:
        for group in groups:
            start = time.perf_counter()
            with reader.prefetch(group, max_bytes=128*1024**2):
                for tensor in group:
                    data = reader.read(tensor)
                    if len(data) != tensor.size:
                        raise RuntimeError('short read')
            samples.append(time.perf_counter() - start)
        row = dict(workers=workers, total_seconds=sum(samples),
                   median_group_ms=statistics.median(samples)*1000,
                   logical_bytes=reader.logical_bytes, samples_seconds=samples)
        rows.append(row)
        a.output.write_text(json.dumps({'scope': __doc__, 'runs': rows}, indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k != 'samples_seconds'}), flush=True)
    finally:
        reader.close()
