"""Two public Flash SSE requests with 34 tools and configurable expert budget."""
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
MODEL = os.environ.get('WERK_FLASH_MODEL', 'pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit')
OUTPUT = Path(os.environ.get('WERK_FLASH_LARGE_TOOL_REPORT', '/tmp/werk-qwen-large-tools-auto.json'))
TOOL_COUNT = int(os.environ.get('WERK_FLASH_TOOL_COUNT', '34'))
if TOOL_COUNT not in (0, 34):
    raise ValueError('WERK_FLASH_TOOL_COUNT must be 0 (short diagnostic) or 34 (WebUI contract)')

TOOLS = [{'type': 'function', 'function': {
    'name': f'lookup_reference_{i:02d}',
    'description': 'Look up reference information in the selected collection. Use only when the user explicitly asks for external reference lookup. Return matching entries with their source identifiers and a concise explanation of the match.',
    'parameters': {'type': 'object', 'properties': {
        key: {'type': 'string', 'description': explanation}
        for key, explanation in [
            ('query', 'The exact text query to search for in the selected source.'),
            ('collection', 'The collection identifier selected by the user.'),
            ('language', 'The preferred language for the matching source entries.'),
            ('since', 'Optional lower date boundary in ISO format.'),
            ('until', 'Optional upper date boundary in ISO format.')]
    }, 'required': ['query']}}} for i in range(TOOL_COUNT)]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    monitor_stop = threading.Event()
    monitor_thread = None
    key = secrets.token_hex(32)
    url = f'http://127.0.0.1:{port}'
    expert_budget = os.environ.get('WERK_FLASH_EXPERT_CACHE_MB', 'auto')
    env = dict(os.environ, WERK_API_KEY=key, WERK_OMLX_EXPERT_CACHE_MB=expert_budget,
               WERK_OMLX_NGRAM_CACHE_MB='auto', WERK_OMLX_THINKING='0', WERK_OMLX_EXPERT_EXECUTION='grouped')
    report = {'model': MODEL, 'expert_cache_mb': expert_budget, 'ngram_cache_mb': 'auto', 'tool_count': len(TOOLS), 'tool_schema_bytes': len(json.dumps(TOOLS, separators=(',', ':')).encode()), 'samples': []}

    def request(messages, stream):
        payload = dict(model=MODEL, messages=messages, tools=TOOLS, tool_choice='auto',
                       stream=stream, temperature=0, max_tokens=256, stream_options={'include_usage': True}, werk={'omlx': {'ngram_cache_mb': 'auto'}})
        req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(payload).encode(),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=600) as response:
            if not stream:
                raw = json.load(response)
                choice = raw['choices'][0]
                answer, finish, done = choice['message'], choice['finish_reason'], None
            else:
                answer = {'role': 'assistant', 'content': ''}
                calls = {}
                finish, done = None, False
                usage = None
                first_text = None
                for raw in response:
                    line = raw.decode().strip()
                    if line == 'data: [DONE]':
                        done = True
                        break
                    if not line.startswith('data: '):
                        continue
                    event = json.loads(line[6:])
                    assert 'error' not in event, event
                    if event.get('usage'): usage = event['usage']
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        if delta.get('content') and first_text is None:
                            first_text = time.monotonic() - started
                        answer['content'] += delta.get('content') or ''
                        for call in delta.get('tool_calls', []):
                            target = calls.setdefault(call['index'], {'type': 'function', 'id': '',
                                'function': {'name': '', 'arguments': ''}})
                            target['id'] += call.get('id') or ''
                            for field in ('name', 'arguments'):
                                target['function'][field] += call.get('function', {}).get(field) or ''
                        finish = choice.get('finish_reason') or finish
                if calls:
                    answer['tool_calls'] = [calls[i] for i in sorted(calls)]
                assert done and finish, (done, finish)
        sample = dict(stream=stream, message=answer, finish_reason=finish, received_done=done,
                      seconds=time.monotonic() - started, usage=usage, first_text_seconds=first_text)
        report['samples'].append(sample)
        OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(sample), flush=True)
        return answer, finish

    with OUTPUT.with_suffix('.log').open('w') as log:
        child = subprocess.Popen([str(ROOT / 'target/release/werk'), '--backend', 'omlx', 'serve',
            '--model', MODEL, '--port', str(port), '--persistence', '--persistence-mode', 'disk', '--persistence-reuse', 'prefer', '--verbose'], env=env,
            stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 90
            while True:
                try:
                    req = urllib.request.Request(url + '/v1/models', headers={'Authorization': 'Bearer ' + key})
                    with urllib.request.urlopen(req, timeout=2):
                        break
                except OSError:
                    if child.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Test server did not start')
                    time.sleep(.2)
            worker_root = Path.home() / '.local/share/werk1112/backends/omlx/workers'
            settings_path = max(worker_root.glob('*/settings.json'), key=lambda p: p.stat().st_mtime)
            settings = json.loads(settings_path.read_text())
            def monitor():
                samples = []
                while not monitor_stop.wait(25):
                    req = urllib.request.Request('http://127.0.0.1:' + str(settings['server']['port']) + '/werk/experts/status',
                        headers={'Authorization': 'Bearer ' + settings['auth']['api_key']})
                    try:
                        with urllib.request.urlopen(req, timeout=5) as response:
                            status = json.load(response)
                        sample = {field: status.get(field) for field in (
                            'last_prefill_admission', 'forward_calls', 'disk_bytes_read',
                            'resident_cache_bytes', 'ngram_resident_cache_bytes',
                            'ngram_effective_cache_budget_bytes', 'prefill_samples_corrected')}
                        samples.append(sample)
                        OUTPUT.with_suffix('.status.json').write_text(json.dumps(samples, indent=2) + '\n')
                        print(json.dumps({'cache_progress': sample}), flush=True)
                    except OSError:
                        pass
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            message, finish = request([{'role': 'user', 'content':
                'Write me a single sentence about the programming language Rust.'}], True)
            assert 'rust' in message['content'].lower() and not message.get('tool_calls'), message
            message, finish = request([{'role': 'user', 'content': 'Explain Rust ownership in one short sentence.'}], True)
            assert message['content'] and not message.get('tool_calls'), message
            report['passed'] = True
            OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
        except BaseException as error:
            report['passed'] = False
            report['error'] = str(error)
            OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
            raise
        finally:
            monitor_stop.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=6)
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)

    phases = [json.loads(line.split('phases ', 1)[1])
              for line in OUTPUT.with_suffix('.log').read_text().splitlines()
              if line.startswith('[werk serve] phases ')]
    for sample, phase in zip(report['samples'], phases):
        sample['backend_phases'] = phase
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
