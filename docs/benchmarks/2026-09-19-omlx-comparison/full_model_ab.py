#!/usr/bin/env python3
"""Sequential before/after comparison with equal cache budgets and fresh workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('WERK_OMLX_', 'OMLX_'))}
    env.update(WERK_OMLX_EXPERT_CACHE_MB='16384', WERK_OMLX_NGRAM_CACHE_MB='auto',
               WERK_OMLX_THINKING='0', WERK_OMLX_EXPERT_EXECUTION='grouped')
    rows = []
    for index, label in enumerate(('before', 'after', 'after', 'before')):
        processes = subprocess.check_output(['ps', '-axo', 'comm='], text=True)
        if any(Path(line.strip()).name == 'omlx-server' for line in processes.splitlines()):
            raise RuntimeError('An oMLX server is running; refusing concurrent model loads')
        binary = getattr(args, label).resolve()
        command = [str(binary), '--backend', 'omlx', 'run',
                   'pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit',
                   'Explain what a token is in three sentences.',
                   '--max-tokens', '64', '--temperature', '0', '--verbose']
        log_path = args.output / f'{index}-{label}.log'
        started = time.perf_counter()
        with log_path.open('w') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                result = child.wait(timeout=300)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        elapsed = time.perf_counter() - started
        text = log_path.read_text()
        if result:
            raise RuntimeError(f'Run failed: {log_path}')
        def metric(name):
            match = re.search(r'^' + re.escape(name) + r':\s+([\d.]+)', text, re.M)
            if not match:
                raise RuntimeError(f'Missing {name} in {log_path}')
            return float(match[1])
        stats = re.search(r'oMLX experts .*?: (\{.*\})', text)
        row = dict(label=label, wall_seconds=elapsed, eval_rate=metric('eval rate'),
                   eval_count=metric('eval count'), prompt_tokens=metric('prompt total count'),
                   binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   expert_stats=json.loads(stats[1]) if stats else None,
                   log=str(log_path))
        rows.append(row)
        (args.output / 'report.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'expert_stats'}), flush=True)
        time.sleep(1)


if __name__ == '__main__':
    main()
