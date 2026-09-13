#!/usr/bin/env python3
"""Reproduce the public, short Flash offload diagnostic (not a quality benchmark)."""
import subprocess,os,time,json,sys,threading
from pathlib import Path
model=sys.argv[1]; label=sys.argv[2]
max_tokens=int(os.environ.get('WERK_FLASH_MAX_TOKENS','16'))
root=Path(__file__).resolve().parents[3]
output=Path(os.environ.get('WERK_FLASH_REPORT_DIR','/tmp/werk-flash-reproduction'));output.mkdir(parents=True,exist_ok=True)
prompts=['Merke dir die Zahl 37. Antworte nur mit OK.','Welche Zahl sollst du dir merken? Antworte nur mit der Zahl.','Was ist 2 plus 3? Nur die Zahl.','What is 6 times 7? Answer with the number only.','Welche Zahl sollte gespeichert bleiben? Nur die Zahl.','Ist Berlin die Hauptstadt von Deutschland? Antworte nur mit Ja oder Nein.','Was ist 20 minus 8? Nur die Zahl.','Translate cat into German. One word only.','Nenne erneut unsere gespeicherte Zahl. Nur die Zahl.','What is 9 plus 4? Answer with the number only.','Wie lautet die anfangs gemerkte Zahl? Nur die Zahl.']
command=[str(root/'target/release/werk'),'--backend','omlx','chat',model,'--persistence','--session',os.environ.get('WERK_FLASH_SESSION','offload-validation-'+label+'-'+str(time.time_ns())),'--temperature','0','--max-tokens',str(max_tokens),'--verbose']
environment=dict(os.environ,WERK_OMLX_EXPERT_CACHE_MB=os.environ.get('WERK_FLASH_EXPERT_CACHE_MB','8192'),WERK_OMLX_THINKING='0',WERK_OMLX_EXPERT_EXECUTION='grouped')
if label=='qwen':environment['WERK_OMLX_NGRAM_CACHE_MB']='1024'
else:environment.pop('WERK_OMLX_NGRAM_CACHE_MB',None)
restart=len(sys.argv)>3 and sys.argv[3].startswith('restart')
if label.startswith('glm'):command+=['--model-home',os.environ.get('WERK_FLASH_MODEL_HOME','/private/tmp/werk-glm-offload-validation')]
if restart:prompts=['Welche Zahl habe ich dich am Anfang gebeten zu merken? Antworte nur mit der Zahl.']
comparison=len(sys.argv)>3 and sys.argv[3].startswith('compare-')
report_label=label+'-'+sys.argv[3].removeprefix('compare-') if comparison else label+'-'+sys.argv[3] if restart else label
if comparison:command[command.index('--session')+1]+='-'+sys.argv[3].removeprefix('compare-')
if comparison and sys.argv[3]=='compare-serial':environment['WERK_OMLX_EXPERT_EXECUTION']='serial'
if comparison and sys.argv[3]=='compare-auto':environment['WERK_OMLX_EXPERT_CACHE_MB']='auto'
started=time.monotonic()
with (output/(report_label+'-chat.log')).open('w') as log:
 child=subprocess.Popen(command,cwd=root,env=environment,stdin=subprocess.PIPE,stdout=log,stderr=log,text=True)
 def feed():
  try:
   child.stdin.write('\n'.join(prompts+['/quit'])+'\n');child.stdin.flush();child.stdin.close()
  except BrokenPipeError:pass
 threading.Thread(target=feed,daemon=True).start()
 try:code=child.wait(timeout=1200)
 except subprocess.TimeoutExpired:
  child.terminate();child.wait(timeout=15);code=124
log_text=(output/(report_label+'-chat.log')).read_text()
process_exit_code=code
completed_turns=sum(line.startswith('finish reason:') for line in log_text.splitlines())
generation_errors=[line for line in log_text.splitlines() if line.startswith('error: ')]
if code == 0 and (generation_errors or completed_turns != len(prompts)): code=1
report={'process_exit_code':process_exit_code,'completed_turns':completed_turns,'generation_errors':generation_errors,'model':model,'session':command[command.index('--session')+1],'execution':environment['WERK_OMLX_EXPERT_EXECUTION'],'expert_cache_mb':environment['WERK_OMLX_EXPERT_CACHE_MB'],'ngram_cache_mb':1024 if label=='qwen' else None,'thinking':False,'temperature':0,'max_tokens':max_tokens,'persistence':True,'prompts':prompts,'wall_seconds':time.monotonic()-started,'exit_code':code}
(output/(report_label+'-chat.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));print((output/(report_label+'-chat.log')).read_text()[-3500:]);sys.exit(code)
