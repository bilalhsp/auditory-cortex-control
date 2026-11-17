import json,os,sqlite3,subprocess,sys
from pathlib import Path
import yaml
from uuid import uuid4
from backend.auth import hash_password
old=json.loads(Path('.scratch/web-migration/issue09-native.json').read_text())
cache=Path(yaml.safe_load(Path('auditory_cortex/config.yml').read_text())['cache_dir'])
root=cache/'auditory-web-checks/issue11-archive'/str(uuid4());root.mkdir(parents=True)
with sqlite3.connect(str(Path(old['root'])/'sessions.sqlite3')) as source,sqlite3.connect(str(root/'sessions.sqlite3')) as target:source.backup(target)
archive=str(cache/'web-migration-builds/auditory-api.pyz')
script='''
import json,sys
from pathlib import Path
from fastapi.testclient import TestClient
from backend.app import create_app
from backend.config import Settings
body=json.load(sys.stdin)
client=TestClient(create_app(Settings(Path(body['db']),'operator',body['hash']),lambda:0));client.auth=('operator','test-password')
sessions=client.get('/api/v1/sessions');assert sessions.status_code==200,sessions.text
assert sessions.json()[0]['status']=='closed'
detail=client.get('/api/v1/sessions/'+sessions.json()[0]['id']);assert 'succeeded' in [job['state'] for job in detail.json()['jobs']]
assert body['private_root'] not in detail.text
experiment=client.post('/api/v1/experiments',json={'gpu_count':0}).json()
closed=client.post('/api/v1/sessions/'+experiment['session_id']+'/close').json()
assert closed['status']=='closed'
assert client.post('/api/v1/sessions/'+experiment['session_id']+'/close').json()==closed
assert 'torch' not in sys.modules
print('PASS: standalone archive recovers closed-session results with zero GPUs and no scientific imports; new session close is idempotent.')
'''
result=subprocess.run([str(Path('.venv/bin/python').absolute()),'-c',script],cwd='/tmp',env=dict(os.environ,PYTHONPATH=archive),input=json.dumps({'db':str(root/'sessions.sqlite3'),'hash':hash_password('test-password'),'private_root':old['root']}),text=True,capture_output=True,check=True)
print(result.stdout)
