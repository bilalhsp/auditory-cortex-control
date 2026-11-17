"""Local ports, real native data, packaged API, and separately served built UI."""
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

import yaml
from backend.auth import hash_password


def main():
    cache = Path(yaml.safe_load(Path('auditory_cortex/config.yml').read_text())['cache_dir'])
    checks = cache / 'auditory-web-checks'
    root = checks / 'issue11-deployed' / str(uuid4())
    root.mkdir(parents=True)
    native = json.loads(Path('.scratch/web-migration/issue11-native.json').read_text())
    assert native['completed']
    assets = root / 'frontend'
    shutil.copytree('frontend/dist', assets)
    (assets / 'config.js').write_text('window.AUDITORY_CONFIG={apiBaseUrl:"http://127.0.0.1:8765"};\n')
    password = secrets.token_urlsafe(24)
    env = dict(os.environ, AUDITORY_OPERATOR='operator', AUDITORY_PASSWORD_HASH=hash_password(password),
               AUDITORY_DATABASE_PATH=str(Path(native['root']) / 'sessions.sqlite3'),
               AUDITORY_ALLOWED_ORIGINS='http://127.0.0.1:5173',
               AUDITORY_CHECK_ROOT=str(checks), AUDITORY_CHECK_PASSWORD=password,
               PLAYWRIGHT_BROWSERS_PATH=str(checks / 'browsers'))
    processes = []
    try:
        with (root / 'api.log').open('w') as api_log, (root / 'frontend.log').open('w') as frontend_log:
            processes.append(subprocess.Popen([sys.executable, str(cache / 'web-migration-builds/auditory-api.pyz'),
                'backend.app:create_app', '--factory', '--host', '127.0.0.1', '--port', '8765'],
                cwd=str(root), env=env, stdout=api_log, stderr=subprocess.STDOUT))
            processes.append(subprocess.Popen([sys.executable, '-m', 'http.server', '5173', '--bind', '127.0.0.1',
                '--directory', str(assets)], stdout=frontend_log, stderr=subprocess.STDOUT))
            for url in ['http://127.0.0.1:8765/healthz', 'http://127.0.0.1:5173/config.js']:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    assert all(process.poll() is None for process in processes), 'Local server exited'
                    try:
                        with urllib.request.urlopen(url, timeout=1) as response:
                            assert response.status == 200
                        break
                    except OSError:
                        time.sleep(.2)
                else:
                    raise RuntimeError('Local deployment did not start: ' + url)
            subprocess.run(['node', '.scratch/web-migration/issue11_deployed_browser.mjs'],
                           env=env, check=True, timeout=120)
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == '__main__':
    main()
