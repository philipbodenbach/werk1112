"""Public GLM tool round-trip and ordinary text with an attached tool catalog."""
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
MODEL = 'Vontra/GLM-5.3-Flash-MLX-oQ2-MTP'
OUTPUT = ROOT / 'docs/benchmarks/2026-09-12-flash-offload/glm-tools.json'
TOOLS = [{'type': 'function', 'function': {'name': 'add', 'description': 'Add two integers.',
    'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                   'required': ['a', 'b']}}}]


def main():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    key = secrets.token_hex(32)
    url = f'http://127.0.0.1:{port}'
    env = dict(os.environ, WERK_API_KEY=key, WERK_OMLX_EXPERT_CACHE_MB='auto',
               WERK_OMLX_NGRAM_CACHE_MB='auto', WERK_OMLX_THINKING='0', WERK_OMLX_REASONING_EFFORT='low')
    report = {'model': MODEL, 'expert_cache_mb': 'auto', 'ngram_cache_mb': 'auto', 'reasoning_effort': 'low', 'samples': []}

    def request(messages, stream):
        payload = dict(model=MODEL, messages=messages, tools=TOOLS, tool_choice='auto',
                       stream=stream, temperature=0, max_tokens=256)
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
                for raw in response:
                    line = raw.decode().strip()
                    if line == 'data: [DONE]':
                        done = True
                        break
                    if not line.startswith('data: '):
                        continue
                    event = json.loads(line[6:])
                    assert 'error' not in event, event
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
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
                      seconds=time.monotonic() - started)
        report['samples'].append(sample)
        OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(sample), flush=True)
        return answer, finish

    with OUTPUT.with_suffix('.log').open('w') as log:
        child = subprocess.Popen([str(ROOT / 'target/release/werk'), '--backend', 'omlx', 'serve',
            '--model', MODEL, '--port', str(port), '--persistence', '--verbose'], env=env,
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
            message, finish = request([{'role': 'user', 'content':
                'Write me a single sentence about the programming language Rust.'}], True)
            assert 'rust' in message['content'].lower() and not message.get('tool_calls'), message
            for stream in (False, True):
                messages = [{'role': 'user', 'content':
                    'Use the add tool to calculate 17 plus 25. You must call the tool; do not calculate it yourself.'}]
                message, finish = request(messages, stream)
                calls = message.get('tool_calls', [])
                assert finish == 'tool_calls' and len(calls) == 1, message
                call = calls[0]
                assert call['id'] and call['function']['name'] == 'add', call
                args = json.loads(call['function']['arguments'])
                assert args == {'a': 17, 'b': 25}, args
                messages += [message, {'role': 'tool', 'tool_call_id': call['id'], 'content': '42'}]
                message, finish = request(messages, stream)
                assert finish == 'stop' and '42' in message['content'] and not message.get('tool_calls'), message
            report['passed'] = True
            OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
        except BaseException as error:
            report['passed'] = False
            report['error'] = str(error)
            OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
            raise
        finally:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


if __name__ == '__main__':
    main()
