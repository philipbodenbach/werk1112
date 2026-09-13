#!/usr/bin/env python3
"""Reproduce the public, short Flash offload diagnostic (not a quality benchmark)."""
import subprocess,os,time,json,sys,socket,urllib.request,urllib.error,statistics,secrets
from pathlib import Path
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'utils/benchmarks'));import chat
model,label,execution,budget=sys.argv[1:5];output=Path(os.environ.get('WERK_FLASH_REPORT_DIR','/tmp/werk-flash-reproduction'));output.mkdir(parents=True,exist_ok=True)
fixture=json.loads((Path(__file__).parent/'qwen-chat.json').read_text()); report={'model':model,'execution':execution,'expert_cache_mb':budget,'ngram_cache_mb':1024 if label.startswith('qwen') else None,'thinking':False,'temperature':0,'samples':[]}
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
key=secrets.token_hex(32)
max_tokens=int(os.environ.get('WERK_FLASH_MAX_TOKENS','16'));report['max_tokens']=max_tokens
environment=dict(os.environ,WERK_API_KEY=key,WERK_OMLX_EXPERT_CACHE_MB=budget,WERK_OMLX_EXPERT_EXECUTION=execution,WERK_OMLX_THINKING='0')
if label.startswith('qwen'):environment['WERK_OMLX_NGRAM_CACHE_MB']='1024'
else:environment.pop('WERK_OMLX_NGRAM_CACHE_MB',None)
command=[str(root/'target/release/werk'),'--backend','omlx','serve','--model',model,'--port',str(port),'--persistence','--verbose']
if label.startswith('glm'):command+=['--model-home',os.environ.get('WERK_FLASH_MODEL_HOME','/private/tmp/werk-glm-offload-validation')]
with (output/(label+'-serve.log')).open('w') as log:
 child=subprocess.Popen(command,cwd=root,env=environment,stdout=log,stderr=log)
 try:
  url=f'http://127.0.0.1:{port}';deadline=time.monotonic()+90
  while True:
   try:
    with urllib.request.urlopen(urllib.request.Request(url+'/v1/models',headers={'Authorization':'Bearer '+key}),timeout=2):pass
    break
   except OSError:
    if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('server not ready')
    time.sleep(.1)
  messages=[]
  for i,prompt in enumerate(fixture['prompts'][:int(os.environ.get('WERK_FLASH_PROMPT_LIMIT','11'))]):
   messages.append({'role':'user','content':prompt});payload={'model':model,'messages':messages,'temperature':0,'max_tokens':max_tokens,'stream':True,'stream_options':{'include_usage':True},'werk':{'omlx':{'thinking':False}}}
   sample=chat.request_chat(url+'/v1/chat/completions',key,payload,180,300)
   sample.update(turn=i+1,prompt=prompt,expected=fixture['results'][i]['expected']);sample['correct']=sample['answer'].strip().rstrip('.!')==sample['expected'];report['samples'].append(sample)
   chat.atomic_json(output/(label+'-serve.json'),report)
   print(i+1,repr(sample['answer']),round(sample['total_seconds'],2),sample.get('cached_prompt_tokens'),flush=True)
   if sample['error'] or not sample['received_done']:raise RuntimeError('incomplete stream: '+str(sample))
   messages.append({'role':'assistant','content':sample['answer']})
  report['all_correct']=all(s['correct'] for s in report['samples']);report['warm_median_seconds']=statistics.median(s['total_seconds'] for s in report['samples'][1:]);chat.atomic_json(output/(label+'-serve.json'),report)
  print('all_correct',report['all_correct'],'warm_median',report['warm_median_seconds'],flush=True)
  if os.environ.get('WERK_FLASH_CHECK_TOOLS') == '1':
   payload={'model':model,'messages':[{'role':'user','content':'Use the add tool to calculate 17 plus 25.'}],
            'temperature':0,'max_tokens':64,'stream':False,'tool_choice':'auto',
            'tools':[{'type':'function','function':{'name':'add','description':'Add two integers.',
                     'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}]}
   request=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(payload).encode(),
       headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
   try:
    with urllib.request.urlopen(request,timeout=180) as response:
     report['tool_probe']={'http_status':response.status,'response':json.load(response)}
   except urllib.error.HTTPError as error:
    report['tool_probe']={'http_status':error.code,'response':json.loads(error.read())}
   chat.atomic_json(output/(label+'-serve.json'),report)
   print('tool_probe',json.dumps(report['tool_probe']),flush=True)
 finally:
  child.terminate()
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
