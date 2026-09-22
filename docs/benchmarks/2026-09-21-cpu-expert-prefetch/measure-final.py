import json, os, pathlib, shutil, subprocess, time, hashlib
R=pathlib.Path('/tmp/werk-prefetch-final-validation');OLD=pathlib.Path('/tmp/werk-qwen-exact-regression'); H=OLD/'store';S=OLD/'seed';U=pathlib.Path('/home/philipbodenbach/.local/share/werk1112')
Q='vumpt/Qwen3.8-Flash-Next-GGUF';D='deepseek-v4-flash-q2-k-s';K='96d4a7ab3571d69197e40742b4a619ed8fcdda2cc3af10c25a0ca7a4b54c9ae6'
env=os.environ.copy();env.update(WERK_LLAMA_ARGS='--n-cpu-moe 38 --reasoning off',WERK_LLAMA_LOG='1',WERK_LLAMA_SERVER_CUDA=str(U/'backends/llama-cuda/upstream-ec928150501c/build/bin/llama-server'))
B={'off':str(R/'werk-candidate'),'auto':str(R/'werk-candidate')};report=[]
def restore():
 for suffix in ['.json','.cache']:
  src=S/(K+suffix);dst=H/'chat-sessions'/(K+suffix)
  if dst.is_dir():shutil.rmtree(dst)
  if src.is_dir():shutil.copytree(src,dst)
  else:shutil.copy2(src,dst)
def run(label,version,model):
 if model==Q and S.exists():restore()
 folder=R/label;folder.mkdir(exist_ok=True)
 args=[B[version],'--model-home',str(H),'--backend','cuda','--threads','24','--threads-batch','24','--ctx-size','4096','--warmup-tokens','0','run',model,'Explain Rust ownership in three sentences.','--verbose','--persistence','--session','qwen-startup-v2' if model==Q else label,'--stream','--json','--max-tokens','256','--temperature','0','--seed','17']
 (folder/'args.json').write_text(json.dumps(args)); start=time.monotonic();seen={}
 print(json.dumps({'starting':label,'version':version,'model':model}),flush=True)
 with (folder/'stdout').open('w') as out,(folder/'stderr').open('w') as err,(folder/'samples.jsonl').open('w') as samples:
  p=subprocess.Popen(args,env=env,stdout=out,stderr=err);lastgpu=-99
  while p.poll() is None:
   t=time.monotonic()-start
   if t>420:p.terminate();p.wait(timeout=15);raise RuntimeError('benchmark safety timeout')
   row={'elapsed':t,'processes':[]}
   for key in ['meminfo','vmstat','pressure/io','pressure/memory']:
    row[key]=pathlib.Path('/proc',key).read_text()
   try:
    row['parent']={key:pathlib.Path('/proc',str(p.pid),key).read_text() for key in ['stat','status','io']}
   except OSError:pass
   for path in pathlib.Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
     if (path/'comm').read_text().strip()!='llama-server':continue
     stat=(path/'stat').read_text()
     if int(stat.split()[3])!=p.pid:continue
     item={key:(path/key).read_text() for key in ['stat','status','io','cmdline']}
     item['pid']=int(path.name); row['processes'].append(item);seen[int(path.name)]=True
     if not (folder/'native-args.txt').exists() and '--model' in item['cmdline']:(folder/'native-args.txt').write_text(item['cmdline'].replace('\0',' '))
    except OSError:pass
   if t-lastgpu>=2:
    g=subprocess.run(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
    row['gpu']=g.stdout.strip();lastgpu=t
   samples.write(json.dumps(row)+'\n');samples.flush();time.sleep(.5)
  code=p.returncode
 events=[]
 for line in (folder/'stdout').read_text().splitlines():
  try:events.append(json.loads(line))
  except ValueError:pass
 completion=next((e for e in reversed(events) if e.get('type')=='completion'),None)
 alive=[pid for pid in seen if pathlib.Path('/proc',str(pid)).exists()]
 result={'label':label,'version':version,'model':model,'exit':code,'wall_seconds':time.monotonic()-start,'children_after_exit':alive,'completion':completion}
 if completion:result['answer_sha256']=hashlib.sha256(completion['message']['content'].encode()).hexdigest()
 report.append(result);(R/'report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(result),flush=True)
 if code or not completion:raise RuntimeError('run failed; inspect '+str(folder))
 if alive:raise RuntimeError('native child survived normal exit')

for policy in ['auto']:
 env['WERK_LLAMA_PREFETCH']=policy
 run(policy+'-deepseek',policy,D)
 run(policy+'-qwen-switch',policy,Q)
 run(policy+'-qwen-repeat',policy,Q)
print('Completed prefetch comparison.',flush=True)
