import json,pathlib,re,statistics
R=pathlib.Path('/tmp/werk-prefetch-final-validation')
rows=json.loads((R/'report.json').read_text())
def vals(s):
 return {a[0].rstrip(':'):int(a[1]) for line in s.splitlines() if len(a:=line.split())>1 and a[1].isdigit()}
summary=[]
for row in rows:
 c=row.get('completion')
 if not c:continue
 t=c['timings'];ds=c.get('backend_diagnostics',[])
 d={'label':row['label'],'version':row['version'],'prompt':c['usage']['prompt_tokens'],'cached':t['cached_prompt_tokens'],'output':c['usage']['completion_tokens'],'prepare':t['load_seconds'],'prefill':t['prompt_seconds'],'decode':t['decode_seconds'],'tok_s':c['usage']['completion_tokens']/t['decode_seconds'],'answer_sha256':row['answer_sha256'],'children_after_exit':row['children_after_exit']}
 for key,needle in [('ttft','run first token including preparation: '),('restore','llama.cpp native KV restore duration: '),('ready','llama.cpp native readiness duration: ')]:
  v=next((s.split(needle)[1].split('s')[0] for s in ds if needle in s),None)
  if v:d[key]=float(v)
 rs=[json.loads(l) for l in (R/row['label']/'samples.jsonl').read_text().splitlines()]
 native=[(r,p) for r in rs for p in r['processes'] if '--model' in p['cmdline']]
 if native:
  d['read_gib']=max(vals(p['io'])['read_bytes'] for r,p in native)/2**30
  d['major_faults']=max(int(p['stat'].split()[11]) for r,p in native)
  d['peak_rss_gib']=max(int(p['stat'].split()[23])*4096 for r,p in native)/2**30
  d['min_available_gib']=min(vals(r['meminfo'])['MemAvailable'] for r in rs)/2**20
  first,last=vals(rs[0]['vmstat']),vals(rs[-1]['vmstat']);d['swapin_mib']=(last['pswpin']-first['pswpin'])*4096/2**20
  native_start= native[0][0]['elapsed']
  # Sample boundary is approximate: the first observed native process appears within one poll.
  log=(R/row['label']/'stderr').read_text()
  match=re.search(r'(\d+)\.(\d+)\.(\d+)\.(\d+).*launch_slot_.*processing task',log)
  if match:
   a,b,cc,dd=map(int,match.groups()); task_sec=a*60+b+cc/1000+dd/1e6
   task_t=native_start+task_sec
   r,p=min(native,key=lambda rp:abs(rp[0]['elapsed']-task_t))
   d['inference_read_gib_approx']=d['read_gib']-vals(p['io'])['read_bytes']/2**30
  gpu=[r['gpu'].split(',') for r in rs if r.get('gpu')]
  if gpu:
   d['gpu_peak_mib']=max(float(g[1]) for g in gpu)
   d['gpu_clock_sm_median']=statistics.median(float(g[3]) for g in gpu)
 d['combined_read_gib']=max((vals(r['parent']['io']).get('read_bytes',0) if 'parent' in r else 0)+sum(vals(p['io']).get('read_bytes',0) for p in r['processes'] if '--model' in p['cmdline']) for r in rs)/2**30
 d['prefetch']=next((v for v in ds if v.startswith('llama.cpp CPU expert prefetch:')),None)
 first,last=vals(rs[0]['vmstat']),vals(rs[-1]['vmstat'])
 d['file_refault_gib_equivalent']=(last['workingset_refault_file']-first['workingset_refault_file'])*4096/2**30
 summary.append(d)
(R/'summary.json').write_text(json.dumps(summary,indent=2))
for d in summary:
 print(' '.join(f'{k}={round(v,3) if isinstance(v,float) else v}' for k,v in d.items() if k!='answer_sha256'))
