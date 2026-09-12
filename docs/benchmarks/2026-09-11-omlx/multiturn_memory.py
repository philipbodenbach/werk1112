#!/usr/bin/env python3
"""Focused growing-conversation regression through Werk's public HTTP API.

Requires the already-running public Werk and exactly one private oMLX worker.
Does not start/stop servers or alter runtime settings. Authentication is read
from OPENAI_API_KEY, or the existing local WebUI configuration (read-only).
"""

import argparse
import datetime as dt
import http.client
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = Path(__file__).resolve().parents[3]
MODEL = 'mlx-community/DeepSeek-V4-Flash-2bit-DQ'
PROMPT = 'Write me a sentence about the programming language rust.'
SEED_ANSWER = ('Your sentence: Rust is a powerful systems programming language '
               'that combines memory safety with a focus on performance and reliability.')
spec = importlib.util.spec_from_file_location('chat_bench', REPO / 'utils/benchmarks/chat.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def public_key():
    if 'OPENAI_API_KEY' in os.environ:
        return os.environ['OPENAI_API_KEY']
    db = Path('/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/open_webui/data/webui.db')
    with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as connection:
        config = {key: json.loads(value) for key, value in connection.execute(
            "SELECT key,value FROM config WHERE key IN ('openai.api_base_urls','openai.api_keys')")}
    index = next(index for index, url in enumerate(config['openai.api_base_urls'])
                 if urllib.parse.urlparse(url).hostname in ('127.0.0.1', 'localhost')
                 and urllib.parse.urlparse(url).port == 11434)
    return config['openai.api_keys'][index]


def find_worker():
    found = []
    roots = Path.home() / '.local/share/werk1112/backends/omlx/workers'
    for path in roots.glob('*/settings.json'):
        try:
            settings = json.loads(path.read_text())
            url = 'http://127.0.0.1:' + str(settings['server']['port'])
            with urllib.request.urlopen(url + '/health', timeout=0.25) as response:
                if response.status == 200:
                    found.append((path.parent.name, url, settings['auth']['api_key']))
        except (OSError, ValueError, KeyError, urllib.error.URLError):
            continue
    if len(found) != 1:
        raise RuntimeError('Expected exactly one active private oMLX worker; found ' + str(len(found)))
    return found[0]


def command_output(command):
    return subprocess.run(command, capture_output=True, text=True, timeout=5).stdout.strip()


def numeric_delta(before, after, fields):
    return {field: after[field] - before[field] for field in fields
            if isinstance(before.get(field), (int, float))
            and isinstance(after.get(field), (int, float))}


def stream_request(key, payload, timeout):
    headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request('http://127.0.0.1:11434/v1/chat/completions',
                                     data=json.dumps(payload).encode(), headers=headers)
    started = time.perf_counter()
    status, body = None, None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            if response.headers.get_content_type() != 'text/event-stream':
                raise ValueError('Expected text/event-stream')
            result = bench.consume_stream(response, started, deadline_seconds=600)
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            status = exc.code
            with exc:
                raw = exc.read(262145)
            body = raw[:262144].decode('utf-8', errors='replace')
            try:
                body = json.loads(body)
            except ValueError:
                pass
        result = {
            'first_text_seconds': None, 'total_seconds': time.perf_counter() - started,
            'finish_reason': None, 'truncated': False, 'answer': '', 'usage': None,
            'received_done': False, 'has_reasoning': False, 'error': str(exc),
            **bench.summarize_usage({}, None, 0),
        }
    result.update({'http_status': status, 'http_error_body': body})
    return bench.redact(result, key) if key else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--turns', type=int, default=6)
    parser.add_argument('--max-tokens', type=int, default=96)
    parser.add_argument('--empty-history', action='store_true', help='Start with one message instead of reproducing the three-message WebUI continuation')
    args = parser.parse_args()
    if args.turns < 1 or args.max_tokens < 1:
        parser.error('turns and max-tokens must be positive')
    if not args.label or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for char in args.label):
        parser.error('label must use only letters, digits, underscores, or hyphens')
    output = args.output or Path('/tmp/werk-multiturn-memory-' + args.label + '.json')
    key = public_key()
    worker_name, worker_url, worker_key = find_worker()
    worker_port = str(urllib.parse.urlparse(worker_url).port)
    worker_pid = command_output(['lsof', '-n', '-P', '-t', '-iTCP:' + worker_port, '-sTCP:LISTEN'])
    if not worker_pid.isdigit():
        raise RuntimeError('Could not identify one private worker listener PID')

    def snapshot(path):
        try:
            request = urllib.request.Request(worker_url + path, headers={'Authorization': 'Bearer ' + worker_key})
            with urllib.request.urlopen(request, timeout=10) as response:
                value = json.load(response)
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
            value = {'diagnostic_error': str(exc)}
        return bench.redact(value, worker_key) if worker_key else value

    report = {
        'label': args.label, 'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'worker': worker_name, 'model': MODEL,
        'settings': {'temperature': 0, 'top_p': .95, 'seed': 42, 'thinking': False,
                     'max_tokens': args.max_tokens, 'turns': args.turns},
        'seed_messages': [] if args.empty_history else [
            {'role': 'user', 'content': PROMPT}, {'role': 'assistant', 'content': SEED_ANSWER}],
        'notes': ['Growing history replays actual answers, starting with 3, 5, 7... messages by default.',
                  'This script does not clear caches or restart workers; first does not imply cold.',
                  'Expert and native-prefix counters are worker intervals, potentially including other requests.',
                  'RSS is sampled once per second; it does not establish the exact Metal peak.',
                  'Decode estimates remain null unless usage explicitly establishes zero reasoning tokens.',
                  'English coherence and factual correctness require manual answer review.'],
        'system_before': {name: command_output(['sysctl', '-n', name])
                          for name in ('hw.memsize', 'machdep.cpu.brand_string', 'vm.swapusage')},
        'samples': [], 'rss_samples': [],
    }
    messages = list(report['seed_messages'])
    stop = threading.Event()
    def monitor():
        started = time.monotonic()
        while not stop.is_set():
            try:
                rss = command_output(['ps', '-p', worker_pid, '-o', 'rss='])
                if rss.isdigit():
                    report['rss_samples'].append({'elapsed_seconds': time.monotonic() - started,
                                                  'rss_bytes': int(rss) * 1024})
            except (OSError, subprocess.TimeoutExpired):
                pass
            stop.wait(1)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    failed = False
    bench.atomic_json(output, report)
    try:
        for turn in range(1, args.turns + 1):
            messages.append({'role': 'user', 'content': PROMPT})
            experts_before = snapshot('/werk/experts/status')
            prefix_before = snapshot('/werk/persistence/status')
            payload = {'model': MODEL, 'messages': messages, 'stream': True,
                       'stream_options': {'include_usage': True}, 'max_tokens': args.max_tokens,
                       'temperature': 0, 'top_p': .95, 'seed': 42,
                       'chat_template_kwargs': {'enable_thinking': False}}
            print(json.dumps({'turn': turn, 'messages': len(messages), 'event': 'started'}), flush=True)
            result = stream_request(key, payload, 240)
            experts_after = snapshot('/werk/experts/status')
            prefix_after = snapshot('/werk/persistence/status')
            result.update({'turn': turn, 'messages': len(messages), 'prompt': PROMPT,
                           'experts_before': experts_before, 'experts_after': experts_after,
                           'prefix_before': prefix_before, 'prefix_after': prefix_after,
                           'expert_interval': numeric_delta(experts_before, experts_after,
                               ('cache_hits', 'cache_misses', 'cache_evictions', 'disk_bytes_read',
                                'disk_read_seconds', 'forward_calls', 'forward_seconds')),
                           'prefix_interval': numeric_delta(prefix_before, prefix_after,
                               ('stores', 'restores', 'restored_tokens', 'failures')),
                           'swapusage_after': command_output(['sysctl', '-n', 'vm.swapusage'])})
            report['samples'].append(result)
            bench.atomic_json(output, report)
            print(json.dumps({field: result.get(field) for field in
                              ('turn', 'messages', 'first_text_seconds', 'total_seconds',
                               'completion_tokens', 'finish_reason', 'error', 'prefix_interval')},
                             ensure_ascii=False), flush=True)
            bad = bool(result['error'] or result['truncated'] or not result['answer'])
            failed = failed or bad
            if result['error'] or not result['answer']:
                break
            messages.append({'role': 'assistant', 'content': result['answer']})
    except KeyboardInterrupt:
        report['interrupted'] = True
        failed = True
    finally:
        stop.set()
        thread.join(timeout=6)
        report['worker_rss_bytes_max_sampled'] = max((sample['rss_bytes'] for sample in report['rss_samples']), default=None)
        report['system_after'] = {'swapusage': command_output(['sysctl', '-n', 'vm.swapusage']),
                                  'vm_stat': command_output(['vm_stat'])}
        report['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        report['needs_review'] = failed or len(report['samples']) != args.turns
        bench.atomic_json(output, report)
        print(json.dumps({'report': str(output), 'needs_review': report['needs_review'],
                          'worker_rss_bytes_max_sampled': report['worker_rss_bytes_max_sampled']}), flush=True)
    return 1 if report['needs_review'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
