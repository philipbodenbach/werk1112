#!/usr/bin/env python3
"""Isolated, bounded official llama.cpp/DeepSeek diagnostic; no product writes.

Reports, model registration and chat state live under --output-dir in /tmp.
The weights are externally registered without copy or symlink. CPU/I/O counters
are Linux observations, not host Windows physical disk measurements. File-backed
reads, major faults and swap are reported separately. No cache is dropped.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time

WERK = str(Path.home() / '.cargo/bin/werk')
MODEL = str(Path.home() / '.local/share/werk1112/models/deepseek-v4-flash-q2-k-s/files')
TICKS = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')
PROMPTS = [
    'Remember 37. Describe Rust in one sentence.\nWhich number? Then describe Rust.\n/exit\n',
    'Which number? Then describe Rust.\n/exit\n',
]


def read_kv(path):
    result = {}
    try:
        for line in Path(path).read_text().splitlines():
            parts = line.replace(':', '').split()
            if len(parts) >= 2:
                try:
                    result[parts[0]] = int(parts[1])
                except ValueError:
                    pass
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return result


def system_sample():
    mem = read_kv('/proc/meminfo')
    vm = read_kv('/proc/vmstat')
    return {
        'mem_kib': {k: mem.get(k) for k in ('MemTotal', 'MemAvailable', 'MemFree', 'Cached', 'Buffers', 'SwapTotal', 'SwapFree', 'Dirty', 'Writeback')},
        'vm_pages': {k: vm.get(k) for k in ('pgfault', 'pgmajfault', 'pswpin', 'pswpout', 'pgpgin', 'pgpgout')},
    }


def process_sample(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        right = stat.rfind(')')
        fields = stat[right + 2:].split()
        io = read_kv(f'/proc/{pid}/io')
        return {
            'pid': pid, 'comm': stat[stat.find('(') + 1:right],
            'state': fields[0], 'ppid': int(fields[1]), 'pgrp': int(fields[2]),
            'start_ticks': int(fields[19]),
            'cpu_seconds': (int(fields[11]) + int(fields[12])) / TICKS,
            'minor_faults': int(fields[7]), 'major_faults': int(fields[9]),
            'rss_bytes': int(fields[21]) * PAGE,
            'read_bytes': io.get('read_bytes'), 'write_bytes': io.get('write_bytes'),
            'rchar': io.get('rchar'), 'wchar': io.get('wchar'),
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return None


def tree_sample(leader):
    result = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            item = process_sample(int(entry.name))
            if item and item['pgrp'] == leader:
                result.append(item)
    return result


def stop_group(proc):
    # Even when Werk already exited, its native children could remain.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def parse_stats(text):
    labels = {
        'total duration': 'total_duration', 'first token': 'first_token',
        'prompt total count': 'prompt_total_tokens',
        'prompt cached count': 'prompt_cached_tokens',
        'prompt eval count': 'prompt_eval_tokens',
        'prompt eval duration': 'prompt_eval_duration',
        'prompt eval rate': 'prompt_eval_rate', 'eval count': 'generated_tokens',
        'eval duration': 'generation_duration', 'eval rate': 'generation_rate',
        'finish reason': 'finish_reason',
    }
    stats = {key: re.findall(r'^' + re.escape(label) + r':\s*(.+)$', text, re.M) for label, key in labels.items()}
    stats['persistence_lines'] = [line for line in text.splitlines() if any(s in line for s in ('native KV', 'persistence enabled', 'conversation persistence active'))]
    stats['native_snapshot_restored'] = bool(re.search(r'native KV snapshot restored: \d+ tokens', text))
    stats['native_snapshot_saved'] = bool(re.search(r'native KV snapshot saved: \d+ tokens', text))
    stats['native_kv_unavailable'] = 'native KV cache unavailable:' in text
    return stats


def summarize_samples(samples, wall):
    per_pid = {}
    peak_tree_rss = 0
    for sample in samples:
        peak_tree_rss = max(peak_tree_rss, sum(p['rss_bytes'] for p in sample['processes']))
        for p in sample['processes']:
            key = (p['pid'], p['start_ticks'])
            previous = per_pid.get(key)
            if previous is None:
                per_pid[key] = dict(p)
            else:
                for field in ('cpu_seconds', 'minor_faults', 'major_faults', 'rss_bytes', 'read_bytes', 'write_bytes', 'rchar', 'wchar'):
                    value = p.get(field)
                    if value is not None:
                        previous[field] = max(value, previous.get(field) or 0)
    procs = list(per_pid.values())
    native = [p for p in procs if 'llama-server' in p['comm']]
    total_cpu = sum(p['cpu_seconds'] for p in procs)
    first, last = samples[0]['system'], samples[-1]['system']
    vm_delta = {k: last['vm_pages'][k] - first['vm_pages'][k] for k in first['vm_pages'] if first['vm_pages'][k] is not None and last['vm_pages'][k] is not None}
    return {
        'observed_processes': procs,
        'observed_process_tree_cpu_seconds': total_cpu,
        'mean_cpu_percent_one_core_100': 100 * total_cpu / wall if wall else None,
        'peak_process_tree_rss_bytes': peak_tree_rss,
        'native_server_storage_read_bytes': sum(p['read_bytes'] or 0 for p in native),
        'native_server_read_syscall_chars': sum(p['rchar'] or 0 for p in native),
        'native_server_major_faults': sum(p['major_faults'] for p in native),
        'system_vm_counter_deltas': vm_delta,
        'system_swap_in_bytes': vm_delta.get('pswpin', 0) * PAGE,
        'system_swap_out_bytes': vm_delta.get('pswpout', 0) * PAGE,
        'system_mem_kib_start': first['mem_kib'], 'system_mem_kib_end': last['mem_kib'],
        'notes': [
            'Process counters are sampled once per second and may miss short-lived processes or final increments.',
            'RSS includes file-backed pages and shared mappings; summing processes can count shared pages more than once.',
            'read_bytes measures Linux-accounted storage reads including file-backed cache misses; it does not measure host physical SSD traffic.',
            'rchar measures read-like syscall characters; mmap page faults are not fully represented by rchar.',
            'Major faults can involve file-backed weights or swap and do not by themselves prove swap activity.',
            'pswpin/pswpout are system-wide swap-page counters; pgpgin/pgpgout use KiB despite sharing the vm counter object.',
        ],
    }


def run_phase(index, command, env, output, timeout):
    log_path = output / f'phase-{index}.log'
    sample_path = output / f'phase-{index}-samples.jsonl'
    done = threading.Event()
    failure = []
    started = time.monotonic()
    samples = [{'elapsed_seconds': 0.0, 'system': system_sample(), 'processes': [], 'log_bytes': 0}]
    proc = None
    timed_out = False
    with log_path.open('w') as log, sample_path.open('w') as sample_log:
        sample_log.write(json.dumps(samples[0]) + '\n')
        try:
            proc = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            def communicate():
                try:
                    proc.communicate(input=PROMPTS[index])
                except BaseException as error:
                    failure.append(repr(error))
                finally:
                    done.set()
            worker = threading.Thread(target=communicate, daemon=True)
            worker.start()
            next_progress = 0
            while not done.wait(1.0):
                elapsed = time.monotonic() - started
                sample = {'elapsed_seconds': elapsed, 'system': system_sample(), 'processes': tree_sample(proc.pid), 'log_bytes': log_path.stat().st_size}
                samples.append(sample)
                sample_log.write(json.dumps(sample) + '\n')
                sample_log.flush()
                if elapsed >= next_progress:
                    print(json.dumps({'event': 'progress', 'phase': index, 'elapsed_seconds': round(elapsed, 1), 'pids': [p['pid'] for p in sample['processes']], 'rss_gib': round(sum(p['rss_bytes'] for p in sample['processes']) / 2**30, 2), 'log_bytes': sample['log_bytes']}), flush=True)
                    next_progress = elapsed + 30
                if elapsed > timeout:
                    timed_out = True
                    stop_group(proc)
                    worker.join(timeout=10)
                    break
            samples.append({'elapsed_seconds': time.monotonic() - started, 'system': system_sample(), 'processes': tree_sample(proc.pid), 'log_bytes': log_path.stat().st_size})
            sample_log.write(json.dumps(samples[-1]) + '\n')
        finally:
            if proc is not None:
                stop_group(proc)
    wall = time.monotonic() - started
    text = log_path.read_text(errors='replace')
    report = {
        'phase': index, 'exit_code': proc.returncode if proc else None,
        'wall_seconds': wall, 'timed_out': timed_out, 'communicate_errors': failure,
        'log_path': str(log_path), 'sample_path': str(sample_path),
        'werk_stats': parse_stats(text), 'resources': summarize_samples(samples, wall),
    }
    (output / f'phase-{index}-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'event': 'phase_finished', 'phase': index, 'exit_code': report['exit_code'], 'wall_seconds': wall, 'timed_out': timed_out, 'werk_stats': report['werk_stats']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--phase-timeout', type=int, default=900)
    parser.add_argument('--phases', type=int, choices=(1, 2), default=2)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--no-gpu-samples', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(Path('/tmp')):
        parser.error('--output-dir must be inside /tmp')
    if not 1 <= args.phase_timeout <= 1800:
        parser.error('--phase-timeout must be between 1 and 1800 seconds')
    if not args.runtime.is_file():
        parser.error('--runtime must point to an existing executable')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('phase-*.log')):
        parser.error('output directory already contains phase logs; choose a fresh directory')
    home = output / 'model-home'
    home.mkdir(exist_ok=True)
    env = dict(os.environ)
    removed = []
    for key in list(env):
        if key.startswith('LLAMA_ARG_') or key.startswith('WERK_LLAMA_'):
            removed.append(key)
            del env[key]
    env['WERK_LLAMA_SERVER_CUDA'] = str(args.runtime.resolve())
    env['WERK_LLAMA_ARGS'] = '--cpu-moe --reasoning off --verbosity 4'
    env['WERK_LLAMA_LOG'] = '1'
    env['WERK_HOME'] = str(home)
    manifest = home / 'models' / 'deepseek-test' / 'manifest.json'
    if not manifest.exists():
        # Reuse the existing inventory, avoiding another 92-GiB checksum pass.
        import_source = str(Path(MODEL).parent)
        import_command = [WERK, '--model-home', str(home), 'import', import_source, '--link', '--name', 'deepseek-test']
        imported = subprocess.run(import_command, env=env, capture_output=True, text=True, timeout=60)
        (output / 'import.log').write_text(imported.stdout + imported.stderr)
        if imported.returncode:
            raise SystemExit('Import failed; see ' + str(output / 'import.log'))
    command = [WERK, '--model-home', str(home), '--no-auto-install-backends', '--backend', 'cuda', '--ctx-size', '4096', '--batch-size', '2048', '--ubatch-size', '512', '--kv-cache-type', 'f16', 'chat', 'deepseek-test', '--verbose', '--persistence', '--session', 'upstream-check', '--max-tokens', '8', '--temperature', '0']
    cpu_model = next((line.partition(':')[2].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown')
    version = subprocess.run([str(args.runtime.resolve()), '--version'], env=env, capture_output=True, text=True, timeout=15)
    metadata = {
        'command': command, 'model_source': MODEL, 'runtime': str(args.runtime.resolve()),
        'native_args': env['WERK_LLAMA_ARGS'], 'removed_inherited_environment_keys': removed,
        'linux_page_bytes': PAGE, 'sample_interval_seconds': 1,
        'phase_timeout_seconds': args.phase_timeout, 'cpu_thread_policy': 'native default; no threads override',
        'cpu_model': cpu_model, 'logical_cpus': os.cpu_count(), 'uname': list(os.uname()),
        'runtime_version': version.stdout + version.stderr,
        'runtime_version_exit_code': version.returncode,
        'prompts_by_phase': PROMPTS,
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if args.prepare_only:
        print(json.dumps({'event': 'prepared', 'metadata': metadata}), flush=True)
        return
    gpu = None
    reports = []
    with (output / 'gpu.csv').open('w') as gpu_log, (output / 'gpu-stderr.log').open('w') as gpu_err:
        if not args.no_gpu_samples and shutil.which('nvidia-smi'):
            gpu = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw', '--format=csv', '--loop=5'], stdout=gpu_log, stderr=gpu_err, start_new_session=True)
        try:
            for index in range(args.phases):
                report = run_phase(index, command, env, output, args.phase_timeout)
                reports.append(report)
                if report['timed_out'] or report['exit_code'] != 0 or report['communicate_errors']:
                    break
        finally:
            if gpu is not None:
                stop_group(gpu)
            (output / 'report.json').write_text(json.dumps({'metadata': metadata, 'phases': reports}, indent=2) + '\n')
    if len(reports) != args.phases or any(p['timed_out'] or p['exit_code'] != 0 for p in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
