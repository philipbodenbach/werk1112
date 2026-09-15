#!/usr/bin/env python3
"""GLM Auto memory / reasoning comparison; label and optional low|high|max.

Requires the installed checkpoint and target/release/werk. Labels containing
baseline run one prompt; other labels run three. Uses and closes a private
server. Reports checkpoint-cache counters, not physical SSD bandwidth.
"""
import os,sys,json,time,socket,secrets,subprocess,threading,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'utils/benchmarks'));import chat
out=ROOT/'docs/benchmarks/2026-09-12-flash-offload'/('glm-opt-'+(sys.argv[1] if len(sys.argv)>1 else 'baseline'))
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
key=secrets.token_hex(32);env=dict(os.environ,WERK_API_KEY=key,WERK_OMLX_EXPERT_CACHE_MB='auto',WERK_OMLX_EXPERT_EXECUTION='grouped',WERK_OMLX_THINKING='0')
env.pop('WERK_OMLX_NGRAM_CACHE_MB', None)
env.pop('WERK_OMLX_REASONING_EFFORT', None)
if len(sys.argv)>2:env['WERK_OMLX_REASONING_EFFORT']=sys.argv[2]
model='Vontra/GLM-5.3-Flash-MLX-oQ2-MTP';url=f'http://127.0.0.1:{port}'
report={'model':model,'expert_budget':'auto','reasoning_effort':env.get('WERK_OMLX_REASONING_EFFORT'),'samples':[],'status':[]};stop=threading.Event();monitor=None
awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
with out.with_suffix('.log').open('w') as log:
 child=subprocess.Popen([str(ROOT/'target/release/werk'),'--backend','omlx','serve','--model',model,'--port',str(port),'--persistence','--verbose'],env=env,stdout=log,stderr=log)
 try:
  deadline=time.monotonic()+90
  while True:
   try:
    with urllib.request.urlopen(urllib.request.Request(url+'/v1/models',headers={'Authorization':'Bearer '+key}),timeout=2):break
   except OSError:
    if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('server not ready')
    time.sleep(.2)
  settings_path=max((Path.home()/'.local/share/werk1112/backends/omlx/workers').glob('*/settings.json'),key=lambda p:p.stat().st_mtime);settings=json.loads(settings_path.read_text())
  def status():
   req=urllib.request.Request('http://127.0.0.1:'+str(settings['server']['port'])+'/werk/experts/status',headers={'Authorization':'Bearer '+settings['auth']['api_key']})
   with urllib.request.urlopen(req,timeout=3) as r:d=json.load(r)
   return {k:d.get(k) for k in ('attention_fusion_bytes','last_prefill_admission','forward_calls','disk_bytes_read','disk_read_seconds','resident_cache_bytes','effective_cache_budget_bytes')}
  def watch():
   while not stop.wait(1):
    try:report['status'].append(status())
    except OSError:pass
  monitor=threading.Thread(target=watch,daemon=True);monitor.start()
  messages=[]
  prompts=['Merke dir die Zahl 37. Antworte nur mit OK.'] if 'baseline' in str(out) else ['Merke dir die Zahl 37. Antworte nur mit OK.','Welche Zahl sollst du dir merken? Antworte nur mit der Zahl.','Was ist 2 plus 3? Nur die Zahl.']
  for prompt in prompts:
   messages.append({'role':'user','content':prompt})
   payload={'model':model,'messages':messages,'temperature':0,'max_tokens':256,'stream':True,'stream_options':{'include_usage':True}}
   result=chat.request_chat(url+'/v1/chat/completions',key,payload,180,300);report['samples'].append(result);result['correct']=result['answer'].strip().rstrip('.!')==['OK','37','5'][len(report['samples'])-1];print(json.dumps(result),flush=True)
   out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
   if result['error']:break
   messages.append({'role':'assistant','content':result['answer']})
  report['status'].append(status())
 finally:
  stop.set()
  if monitor:monitor.join(timeout=4)
  out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
  child.terminate()
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
  awake.terminate();awake.wait(timeout=5)
