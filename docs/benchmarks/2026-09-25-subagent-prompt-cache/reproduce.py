import argparse, json, socket, subprocess, tempfile, time, urllib.request
from pathlib import Path
parser = argparse.ArgumentParser(description="Isolated llama.cpp agent-switch cache regression")
parser.add_argument("--runtime", required=True, help="Path to llama-server")
parser.add_argument("--model", required=True, help="Path to Qwen3.8-Flash-Next Q4_K_M GGUF")
options = parser.parse_args()
runtime, model = options.runtime, options.model

def probe(cache):
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    root = Path(tempfile.mkdtemp(prefix='werk-subagent-cache-'))
    url = f'http://127.0.0.1:{port}'
    args = [runtime, '-m', model, '--host', '127.0.0.1', '--port', str(port), '-c', '4096', '-b', '512', '-ub', '512', '-np', '1', '-ngl', '999', '--n-cpu-moe', '38', '-t', '24', '-tb', '24', '--no-warmup', '--slots', '--slot-save-path', str(root), '--cache-ram', str(cache), '--no-cache-idle-slots', '--slot-prompt-similarity', '0', '-fa', 'on', '--reasoning', 'off']
    def req(path, data=None):
        r = urllib.request.Request(url + path, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(r, timeout=180) as f:
            return json.load(f)
    with (root / 'native.log').open('wb') as log:
        child = subprocess.Popen(args, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 180
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f'native exited {child.returncode}: {root}/native.log')
                try:
                    req('/health')
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('native health timeout')
                    time.sleep(.25)
            alpha = 'You are the code reviewer. ' + 'Review the function for correctness and explain each branch. ' * 40
            beta = 'You are the design reviewer. ' + 'Review the page layout for spacing and readable colors. ' * 40
            results = []
            for name, prompt, pinned in [('A', alpha, False), ('B', beta, False), ('A-return', alpha + ' Continue carefully.', False), ('B-return', beta + ' Continue carefully.', False)]:
                body = {'prompt': prompt, 'n_predict': 1, 'temperature': 0, 'cache_prompt': True}
                value = req('/completion', body)
                results.append({'name':name, 'timings':value.get('timings'), 'tokens_evaluated':value.get('tokens_evaluated')})
                print(json.dumps({'cache_mib':cache, **results[-1]}), flush=True)
            # A native RAM entry must not rescue a deliberately erased pinned slot.
            req('/slots/0?action=erase', {})
            value = req('/completion', {'prompt': alpha + ' Continue carefully.', 'n_predict':1, 'temperature':0, 'cache_prompt':True, 'id_slot':0})
            pinned_cache = value.get('timings', {}).get('cache_n')
            print(json.dumps({'cache_mib':cache, 'name':'pinned-after-erase', 'timings':value.get('timings')}), flush=True)
            assert pinned_cache == 0, pinned_cache
            for result in results[2:]:
                cached = result['timings']['cache_n']
                assert (cached > 100) if cache else (cached == 0), result
            (root / 'results.json').write_text(json.dumps(results, indent=2))
            print(f'PASS cache={cache}; evidence={root}', flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()

for cache in [0, 1024]:
    probe(cache)
