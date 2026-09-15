#!/usr/bin/env python3
"""Three identical Rust prompts for Qwen or DeepSeek; optional label.

WERK_FLASH_FAMILY selects qwen (default) or deepseek. --fixed-history replays
the corresponding baseline answers. Qwen Ngram cache is fixed at 1024 MiB.

Uses temperature zero, Auto expert memory by default, thinking off and persistence.
Add --same-context after the label to repeat an identical request without history.
Use --no-persistence for ordinary native caching, --fixed-history to replay the
recorded conversation, and WERK_FLASH_EXPERT_CACHE_MB to fix the expert budget.
WERK_FLASH_BIN selects a separately built comparison executable.
Reports native counter snapshots; logical reads include the OS file cache.
"""
import os,sys,json,time,socket,secrets,subprocess,threading,urllib.request,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'utils/benchmarks'));import chat
family=os.environ.get('WERK_FLASH_FAMILY','qwen')
models={'qwen':('pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit',48),'deepseek':('mlx-community/DeepSeek-V4-Flash-2bit-DQ',43)}
model,layers=models[family]
out=ROOT/'docs/benchmarks/2026-09-12-flash-offload'/(family+'-decode-'+(sys.argv[1] if len(sys.argv)>1 else 'baseline'))
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
key=secrets.token_hex(32);env=dict(os.environ,WERK_API_KEY=key,WERK_OMLX_EXPERT_CACHE_MB=os.environ.get('WERK_FLASH_EXPERT_CACHE_MB','auto'),WERK_OMLX_EXPERT_EXECUTION='grouped',WERK_OMLX_THINKING='0')
env.pop('WERK_OMLX_NGRAM_CACHE_MB', None)
env.pop('WERK_OMLX_REASONING_EFFORT', None)
if family=='qwen':env['WERK_OMLX_NGRAM_CACHE_MB']='1024'
url=f'http://127.0.0.1:{port}'
same_context='--same-context' in sys.argv[2:]
persistence='--no-persistence' not in sys.argv[2:]
fixed_history='--fixed-history' in sys.argv[2:]
reference=json.loads((out.parent/(family+'-decode-baseline.json')).read_text()) if fixed_history else None
binary=os.environ.get('WERK_FLASH_BIN',str(ROOT/'target/release/werk'))
report={'model':model,'binary':binary,'binary_sha256':hashlib.sha256(Path(binary).read_bytes()).hexdigest(),'same_context':same_context,'persistence':persistence,'fixed_history':fixed_history,'expert_budget':env['WERK_OMLX_EXPERT_CACHE_MB'],'reasoning_effort':env.get('WERK_OMLX_REASONING_EFFORT'),'samples':[],'status':[]};stop=threading.Event();monitor=None
awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
with out.with_suffix('.log').open('w') as log:
 child=subprocess.Popen([binary,'--backend','omlx','serve','--model',model,'--port',str(port),*(['--persistence'] if persistence else []),'--verbose'],env=env,stdout=log,stderr=log)
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
   return {k:d.get(k) for k in ('cache_policy','attention_fusion_bytes','attention_shared_bytes','last_prefill_admission','last_decode_admission','cache_hits','cache_misses','cache_evictions','forward_calls','forward_seconds','routing_seconds','materialize_seconds','disk_bytes_read','disk_read_seconds','resident_cache_bytes','effective_cache_budget_bytes')}
  def watch():
   while not stop.wait(1):
    try:report['status'].append(status())
    except OSError:pass
  monitor=threading.Thread(target=watch,daemon=True);monitor.start()
  messages=[]
  prompts=['Explain ownership and borrowing in Rust in about 100 words.']*3
  for turn,prompt in enumerate(prompts):
   if same_context:messages=[]
   messages.append({'role':'user','content':prompt})
   payload={'model':model,'messages':messages,'temperature':0,'max_tokens':256,'stream':True,'stream_options':{'include_usage':True}}
   before=status();start_index=len(report['status'])
   result=chat.request_chat(url+'/v1/chat/completions',key,payload,300,600);result['before']=before;result['after']=status();result['status_range']=[start_index,len(report['status'])];report['samples'].append(result);result['correct']='rust' in result['answer'].lower() and result['finish_reason']=='stop';print(json.dumps(result),flush=True)
   if reference:result['matches_baseline']=result['answer']==reference['samples'][turn]['answer']
   out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
   if result['error']:break
   messages.append({'role':'assistant','content':reference['samples'][turn]['answer'] if reference else result['answer']})
  report['status'].append(status())
  log.flush()
  phases=[json.loads(line.split('phases ',1)[1]) for line in out.with_suffix('.log').read_text().splitlines() if line.startswith('[werk serve] phases ')]
  for sample,phase in zip(report['samples'],phases):
   sample['backend_phases']=phase
   before,after=sample['before'],sample['after'];hits=after['cache_hits']-before['cache_hits'];misses=after['cache_misses']-before['cache_misses']
   sample['expert_hit_rate']=hits/(hits+misses) if hits+misses else None
   sample['expert_read_GiB']=(after['disk_bytes_read']-before['disk_bytes_read'])/1024**3
   sample['decode_tokens_per_second']=sample['completion_tokens']/phase['decode_seconds'] if phase['decode_seconds'] else None
   start,end=sample['status_range']
   # Only snapshots after a decode callback for this request. Counter reads
   # can occur partway through a token; normalize by this model's routed layers.
   states=[s for s in report['status'][start:end]
           if s.get('last_decode_admission')!=sample['before'].get('last_decode_admission')
           and (s.get('last_decode_admission') or {}).get('cached_tokens',0)>sample['prompt_tokens']]
   if len(states)>1:
    first,last=states[0],states[-1];tokens=(last['forward_calls']-first['forward_calls'])/layers
    hits=last['cache_hits']-first['cache_hits'];misses=last['cache_misses']-first['cache_misses']
    if tokens>0 and hits+misses>0:
     sample['decode_window']={'token_equivalents':tokens,'expert_hit_rate':hits/(hits+misses),
       'read_GiB_per_token':(last['disk_bytes_read']-first['disk_bytes_read'])/1024**3/tokens,
       **{k+'_per_token':(last[k]-first[k])/tokens for k in ('disk_read_seconds','forward_seconds','routing_seconds')}}
 finally:
  stop.set()
  if monitor:monitor.join(timeout=4)
  out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
  child.terminate()
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
  awake.terminate();awake.wait(timeout=5)
