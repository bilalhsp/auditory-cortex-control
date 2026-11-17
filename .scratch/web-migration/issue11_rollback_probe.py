"""Isolated legacy rollback rehearsal; does not switch production traffic."""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

import yaml


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    cache = Path(yaml.safe_load(Path('auditory_cortex/config.yml').read_text())['cache_dir'])
    root = cache / 'auditory-web-checks/issue11-rollback' / str(uuid4())
    root.mkdir(parents=True)
    baseline = json.loads(Path('.scratch/web-migration/generation-baseline.json').read_text())
    audio = [Path(item['path']) for item in baseline['outputs']]
    fit = cache / 'web-migration-baseline/issue-01/fit-cache'
    artifacts = audio + list(fit.rglob('*.pkl'))
    assert len(audio) == 4 and artifacts, 'Retained baseline artifacts unavailable'
    before = {str(path): digest(path) for path in artifacts}
    report = {'completed': False, 'root': str(root), 'production_traffic_changed': False}
    started = time.monotonic()
    try:
        # Execute the actual retained launcher with no browser session, so it
        # cannot initialize the legacy UI's destructive dated output directory.
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        with (root / 'streamlit.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
                '--server.address', '127.0.0.1', '--server.port', str(port),
                '--server.headless', 'true', '--browser.gatherUsageStats', 'false'],
                stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError('Legacy launcher exited; inspect private streamlit.log')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:%d/_stcore/health' % port, timeout=1) as response:
                            assert response.status == 200
                        break
                    except OSError:
                        time.sleep(.2)
                else:
                    raise RuntimeError('Legacy launcher readiness timed out')
                report['legacy_launcher_health'] = 'passed'
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        # Exercise the same Frontend in an explicitly private work directory.
        from streamlit.testing.v1 import AppTest
        script = root / 'isolated_legacy.py'
        script.write_text('from auditory_cortex.optimal_stimulus.frontend import Frontend\n'
                          'Frontend(%r).render()\n' % str(root / 'legacy-work'))
        app = AppTest.from_file(str(script), default_timeout=90).run()
        assert not app.exception, [item.message for item in app.exception]
        assert app.title[0].value == 'Optimal Stimulus Generator'
        assert any(item.label == 'Stimulus type' for item in app.selectbox)
        report['isolated_legacy_ui'] = 'passed'
        after = {str(path): digest(path) for path in artifacts}
        assert after == before, 'Retained scientific artifacts changed'
        report['retained_artifact_sha256'] = after
        report['completed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report['seconds'] = time.monotonic() - started
        Path('.scratch/web-migration/issue11-rollback.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
