#!/usr/bin/env python3
"""Sequential real CLI/public HTTP comparison; owns and stops every test process."""
import argparse, datetime, hashlib, importlib.util, json, os, pathlib, re, secrets, subprocess, time, urllib.request, uuid
ROOT=pathlib.Path(__file__).resolve().parents[3]
MODEL='mlx-community/DeepSeek-V4-Flash-2bit-DQ'
PROMPT='Hi, schreibe mir einen satz über die programmiersprache rust.'
spec=importlib.util.spec_from_file_location('sse',ROOT/'utils/benchmarks/chat.py')
sse=importlib.util.module_from_spec(spec);spec.loader.exec_module(sse)

def wait_workers():
    for _ in range(100):
        ps=subprocess.run(['ps','-axo','comm='],capture_output=True,text=True,check=True).stdout
        if not any(pathlib.Path(x).name=='omlx-server' for x in ps.splitlines()): return
        time.sleep(.1)
    raise RuntimeError('Another oMLX worker remains; refusing parallel model loads')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('budget',choices=['8192','auto']);ap.add_argument('--reuse-cli',action='store_true');a=ap.parse_args()
    wait_workers()
    out=pathlib.Path('/tmp/werk-parity-'+a.budget);out.mkdir(exist_ok=True)
    env=dict(os.environ,WERK_OMLX_THINKING='0',WERK_OMLX_EXPERT_EXECUTION='grouped')
    if a.budget=='auto': env.pop('WERK_OMLX_EXPERT_CACHE_MB',None)
    else: env['WERK_OMLX_EXPERT_CACHE_MB']=a.budget
    session='validation-parity-'+uuid.uuid4().hex
    exe=str(ROOT/'target/release/werk')
    print('CLI starting',a.budget,'fresh isolated conversation',flush=True)
    if a.reuse_cli:
        log=(out/'chat.log').read_text()
    else:
        cli=subprocess.Popen([exe,'--backend','omlx','chat',MODEL,'--persistence','--session',session,'--temperature','0','--top-p','0.95','--seed','42','--max-tokens','64','--verbose'],env=env,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            log,_=cli.communicate((PROMPT+'\n')*2+'/quit\n',timeout=600)
        finally:
            if cli.poll() is None: cli.terminate();cli.wait(timeout=15)
        (out/'chat.log').write_text(log)
        if cli.returncode: raise RuntimeError('CLI failed; inspect '+str(out/'chat.log'))
    answers=re.findall(r'assistant> (.*?)\n\nbackend:',log,re.S)
    answers=[answer.split('\nnote: response reached --max-tokens')[0].rstrip() for answer in answers]
    if len(answers)!=2: raise RuntimeError('Expected two CLI completions')
    print('CLI finished',a.budget,flush=True)
    wait_workers()
    key=secrets.token_hex(24);env['WERK_API_KEY']=key
    port=18434
    # Never terminate a user process occupying the selected test port.
    import socket
    with socket.socket() as sock: sock.bind(('127.0.0.1',port))
    handle=(out/'serve.log').open('w')
    server=subprocess.Popen([exe,'--backend','omlx','serve','--model',MODEL,'--port',str(port),'--verbose','--persistence'],env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
    rows=[]
    try:
        for _ in range(120):
            if server.poll() is not None: raise RuntimeError('Serve startup failed')
            try:
                req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/models',headers={'Authorization':'Bearer '+key})
                with urllib.request.urlopen(req,timeout=1) as response: json.load(response)
                break
            except OSError: time.sleep(1)
        else: raise RuntimeError('Serve startup timeout')
        messages=[]
        for turn in range(2):
            messages.append({'role':'user','content':PROMPT})
            payload={'model':MODEL,'messages':messages,'temperature':0,'top_p':.95,'seed':42,'max_tokens':64,'stream':True,'stream_options':{'include_usage':True}}
            req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            started=time.perf_counter()
            with urllib.request.urlopen(req,timeout=240) as response: row=sse.consume_stream(response,started)
            row['matches_cli']=row['answer'].strip()==answers[turn].strip()
            rows.append(row)
            if row['error'] or not row['received_done']: raise RuntimeError('Incomplete SSE')
            # Use the CLI answer so turn two tests exactly the same history even on a mismatch.
            messages.append({'role':'assistant','content':answers[turn]})
            print('Serve turn',turn+1,'seconds',round(row['total_seconds'],3),'matches CLI',row['matches_cli'],flush=True)
    finally:
        if server.poll() is None: server.terminate()
        server.wait(timeout=20);handle.close()
        wait_workers()
    report={'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'binary_sha256':hashlib.sha256(pathlib.Path(exe).read_bytes()).hexdigest(),'budget':a.budget,'model':MODEL,'temperature':0,'top_p':.95,'seed':42,'thinking':False,'max_tokens':64,'cli_log_reused':a.reuse_cli,'cli_answers':answers,'serve':rows}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('Report',out/'report.json','all owned processes stopped',flush=True)

if __name__=='__main__':main()
