import os,socket,subprocess,time,urllib.request
from pathlib import Path
from uuid import uuid4
import yaml
repo=Path.cwd();cache=Path(yaml.safe_load((repo/'auditory_cortex/config.yml').read_text())['cache_dir']);checks=cache/'auditory-web-checks';root=checks/'laptop-browser'/str(uuid4());root.mkdir(parents=True)
def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
api,front=port(),port()
(root/'fixture.py').write_text('''from fastapi import FastAPI
from backend.tests.test_recordings import RecordingApiTests
from backend.app import create_app
from dataclasses import replace
fixture=RecordingApiTests();fixture.setUpClass();fixture.setUp()
app=create_app(replace(fixture.settings,allowed_origins=(%r,)),lambda:1)
'''%('http://127.0.0.1:'+str(front)))
env=dict(os.environ,PYTHONPATH=str(repo)+':'+str(root),CHECK_API='http://127.0.0.1:'+str(api),CHECK_FRONT='http://127.0.0.1:'+str(front),AUDITORY_CHECK_ROOT=str(checks),PLAYWRIGHT_BROWSERS_PATH=str(checks/'browsers'))
processes=[]
try:
    with (root/'api.log').open('w') as a,(root/'front.log').open('w') as f:
        processes.append(subprocess.Popen([str(repo/'.venv/bin/python'),'-m','uvicorn','fixture:app','--host','127.0.0.1','--port',str(api)],env=env,stdout=a,stderr=subprocess.STDOUT))
        processes.append(subprocess.Popen([str(repo/'start-frontend.sh'),'--port',str(front),'--connection-command','bash connect-gilbreth.sh --login gilbreth-fe02 --node gilbreth-n6'],env=env,stdout=f,stderr=subprocess.STDOUT))
        for url in [env['CHECK_API']+'/healthz',env['CHECK_FRONT']]:
            for _ in range(120):
                try:urllib.request.urlopen(url,timeout=1).close();break
                except OSError:time.sleep(.2)
            else:raise RuntimeError('Server did not start')
        with urllib.request.urlopen(env['CHECK_FRONT']+'/config.js') as response:
            assert b'connect-gilbreth.sh --login gilbreth-fe02 --node gilbreth-n6' in response.read()
        subprocess.run(['node','.scratch/web-migration/connection-card-browser.mjs'],env=env,check=True,timeout=90)
finally:
    for p in processes:p.terminate()
    for p in processes:
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
