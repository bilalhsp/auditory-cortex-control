"""Run from the repository root with existing scientific dependency paths/environment."""
import io,json,os,sqlite3,subprocess,threading,time,zipfile
import yaml
from pathlib import Path
from uuid import uuid4
from fastapi.testclient import TestClient
from backend.app import create_app
from backend.auth import hash_password
from backend.config import Settings
old=json.loads(Path('.scratch/web-migration/issue05-parity.json').read_text())
source_root=Path(old['root']);fit_id=old['attempts'][-1]['fit_id']
root=Path(yaml.safe_load(Path('auditory_cortex/config.yml').read_text())['cache_dir'])/'auditory-web-checks/issue08-real'/str(uuid4());root.mkdir(parents=True)
with sqlite3.connect(str(source_root/'sessions.sqlite3')) as source,sqlite3.connect(str(root/'sessions.sqlite3')) as target:
    source.backup(target);target.execute('DELETE FROM generations')
settings=Settings(root/'sessions.sqlite3','operator',hash_password('test-password'))
client=TestClient(create_app(settings));client.auth=('operator','test-password')
response=client.post('/api/v1/fits/'+fit_id+'/batches',json={'request_id':str(uuid4()),'gpu_count':1,'duration_seconds':1,'clip_count':1,'targets':[{'channel_id':'12.03','objective':objective} for objective in ['stretch','one-hot']]})
assert response.status_code==202,response.text
batch=response.json();env=dict(os.environ,AUDITORY_OPERATOR=settings.operator,AUDITORY_PASSWORD_HASH=settings.password_hash,AUDITORY_DATABASE_PATH=str(settings.database_path))
stop=threading.Event();memory=[]
def monitor():
    while not stop.is_set():
        try: memory.append(max(int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()))
        except Exception: pass
        stop.wait(2)
thread=threading.Thread(target=monitor);thread.start();start=time.monotonic()
try:
    with (root/'worker.log').open('w') as log:
        subprocess.run(['.venv/bin/python','-m','backend.worker','--once'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
finally: stop.set();thread.join()
result=client.get('/api/v1/batches/'+batch['id']).json();assert result['state']=='succeeded',(result,root)
download=client.get('/api/v1/batches/'+batch['id']+'/download');assert download.status_code==200,download.text
archive=zipfile.ZipFile(io.BytesIO(download.content));assert len(archive.namelist())==3
assert client.get('/api/v1/capabilities').json()['gpu']['available_count']==1
assert all(t['exit_code']==0 and t['clips'][0]['frames']==15392 for t in result['targets'])
assert result['targets'][0]['finished_at']<=result['targets'][1]['started_at']
report={'completed':True,'root':str(root),'batch_id':batch['id'],'gpu_count':1,'targets':2,'total_seconds':time.monotonic()-start,'peak_observed_gpu_memory_mib':max(memory) if memory else None,'sequential_wave_order_verified':True,'capacity_released':True,'physical_multi_gpu_benchmark_available':False,'latency_threshold':'User declined latency targets; report measurements only','zip_members':archive.namelist(),'target_results':[{'id':t['id'],'objective':t['objective'],'state':t['state'],'exit_code':t['exit_code'],'started_at':t['started_at'],'finished_at':t['finished_at']} for t in result['targets']]}
Path('.scratch/web-migration/issue08-benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
