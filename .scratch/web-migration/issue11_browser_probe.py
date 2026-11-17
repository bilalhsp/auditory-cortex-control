import os,subprocess,time,urllib.request
from pathlib import Path
root=Path('/scratch/gilbreth/ahmedb/cache/auditory-web-checks')
(root/'gpu-count').write_text('3')
env=dict(os.environ, PYTHONPATH=str(Path.cwd())+':'+str(Path('.scratch/web-migration').absolute()),PLAYWRIGHT_BROWSERS_PATH=str(root/'browsers'))
processes=[]
try:
    with (root/'api.log').open('w') as api_log, (root/'frontend.log').open('w') as front_log:
        processes.append(subprocess.Popen(['.venv/bin/python','-m','uvicorn','issue11_browser_api:app','--host','127.0.0.1','--port','8765'],env=env,stdout=api_log,stderr=subprocess.STDOUT))
        processes.append(subprocess.Popen(['node','frontend/node_modules/vite/bin/vite.js','preview','frontend','--host','127.0.0.1','--port','5173','--strictPort'],cwd=Path.cwd(),env=env,stdout=front_log,stderr=subprocess.STDOUT))
        for url in ['http://127.0.0.1:8765/healthz','http://127.0.0.1:5173']:
            for attempt in range(100):
                try:
                    urllib.request.urlopen(url,timeout=1).close()
                    break
                except Exception:
                    time.sleep(.2)
            else: raise RuntimeError('Server did not start: '+url)
        subprocess.run(['node','.scratch/web-migration/issue11_browser.mjs'],env=env,check=True,timeout=180)
finally:
    for p in processes:
        p.terminate()
    for p in processes:
        try: p.wait(timeout=10)
        except subprocess.TimeoutExpired: p.kill()
