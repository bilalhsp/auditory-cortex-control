"""Run from the repository root with the existing scientific environment/PYTHONPATH."""
import json,os,subprocess,time
import yaml
from pathlib import Path
from uuid import uuid4
from fastapi.testclient import TestClient
from backend.app import create_app
from backend.config import Settings
from backend.auth import hash_password
root=Path(yaml.safe_load(Path('auditory_cortex/config.yml').read_text())['cache_dir'])/'auditory-web-checks/issue09-real'/str(uuid4());root.mkdir(parents=True)
settings=Settings(root/'sessions.sqlite3','operator',hash_password('test-password'))
client=TestClient(create_app(settings));client.auth=('operator','test-password')
experiment=client.post('/api/v1/experiments',json={'gpu_count':1}).json();prefix='/api/v1/experiments/'+experiment['id']
record=next(row for row in client.get(prefix+'/recordings').json()['recordings'] if '2026-01-14a' in row['display_name'])
experiment=client.post(prefix+'/recording-selection',json={'recording_id':record['id']}).json()
env=dict(os.environ,AUDITORY_OPERATOR=settings.operator,AUDITORY_PASSWORD_HASH=settings.password_hash,AUDITORY_DATABASE_PATH=str(settings.database_path))
def submit():
    response=client.post(prefix+'/fits',json={'request_id':str(uuid4()),'recording_id':record['id'],'recording_revision':1,'seed':42});assert response.status_code==202,response.text;return response.json()
fit=submit();start=time.monotonic()
with (root/'complete.log').open('w') as log: subprocess.run(['.venv/bin/python','-m','backend.worker','--once'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
result=client.get('/api/v1/fits/'+fit['id']).json();assert result['state']=='succeeded', (result,root)
baseline=json.loads(Path('.scratch/web-migration/fit-baseline.json').read_text())['correlations']
delta=max(abs(row['correlation']-baseline[row['channel_id']]) for row in result['correlations']);assert delta==0.0,delta
complete_seconds=time.monotonic()-start
second=submit()
with (root/'cancel.log').open('w') as log:
    process=subprocess.Popen(['.venv/bin/python','-m','backend.worker','--once'],env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            current=client.get('/api/v1/fits/'+second['id']).json()
            if current['state']=='running' and current['stage']=='fitting': break
            assert current['state'] not in ['failed','succeeded'], (current,root)
            time.sleep(1)
        else: raise RuntimeError('Native fitting phase did not start')
        close=client.post('/api/v1/sessions/'+experiment['session_id']+'/close');assert close.status_code==200,close.text
        process.wait(timeout=30)
    finally:
        if process.poll() is None: process.terminate();process.wait(timeout=10)
final=client.get('/api/v1/fits/'+second['id']).json();assert final['state']=='cancelled',final
session=client.get('/api/v1/sessions/'+experiment['session_id']).json();assert session['session']['status']=='closed',session
assert client.get('/api/v1/capabilities').json()['gpu']['available_count']==1
assert client.get('/api/v1/fits/'+fit['id']).json()['state']=='succeeded'
child=json.loads((root/'fits'/second['id']/'process.json').read_text());assert child['exit_code']!=0,child
report={'completed':True,'root':str(root),'fit_max_absolute_baseline_delta':delta,'channels':len(baseline),'complete_fit_seconds':complete_seconds,'cancelled_during':'fitting','cancelled_fit_id':second['id'],'child_exit_code':child['exit_code'],'capacity_released':True,'session_closed':True,'previous_completed_fit_preserved':True}
Path('.scratch/web-migration/issue09-native.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
