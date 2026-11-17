"""Run from repo root in the existing scientific environment; never production DB.

Compares real mVocs web fits for both recording sources with a newly measured
legacy reference. Large artifacts/logs remain in the configured cache. A failed
probe writes its failure and exits nonzero; it never authorizes production cutover.
"""
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

import yaml
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.auth import hash_password
from backend.config import Settings


def main():
    cache = Path(yaml.safe_load(Path('auditory_cortex/config.yml').read_text())['cache_dir'])
    root = cache / 'auditory-web-checks/issue11-native' / str(uuid4())
    root.mkdir(parents=True)
    report = {'completed': False, 'root': str(root), 'family': 'mVocs', 'fits': [],
              'latency_threshold': None, 'scope': 'native fit parity; generation separately gated'}
    destination = Path('.scratch/web-migration/issue11-native.json')
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        import cupy as cp
        from auditory_cortex.optimal_stimulus.generator import StimGenerator
        report['gpu'] = torch.cuda.get_device_name(0)
        before = time.monotonic()
        legacy = StimGenerator('ucdavisBAK', 'deepspeech2', 2, mVocs=True, bin_width=50, lag=200)
        report['initialization_seconds'] = time.monotonic() - before
        np.random.seed(42)
        torch.manual_seed(42)
        before = time.monotonic()
        _, correlations = legacy.fit_encoding_model(0, artifact_path=root / 'legacy-model.pkl')
        report['legacy_fit_seconds'] = time.monotonic() - before
        baseline = {str(key): float(value) for key, value in correlations.items()}
        report['legacy_correlations'] = baseline
        del legacy
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        settings = Settings(root / 'sessions.sqlite3', 'operator', hash_password('test-password'))
        client = TestClient(create_app(settings))
        client.auth = ('operator', 'test-password')
        env = dict(os.environ, AUDITORY_OPERATOR=settings.operator,
                   AUDITORY_PASSWORD_HASH=settings.password_hash,
                   AUDITORY_DATABASE_PATH=str(settings.database_path))
        for source in ['existing', 'upload']:
            response = client.post('/api/v1/experiments', json={'stimulus_family': 'mVocs', 'gpu_count': 1})
            assert response.status_code == 201, response.text
            experiment = response.json()
            prefix = '/api/v1/experiments/' + experiment['id']
            record = next(item for item in client.get(prefix + '/recordings').json()['recordings']
                          if '2026-01-14a' in item['display_name'])
            if source == 'existing':
                response = client.post(prefix + '/recording-selection', json={'recording_id': record['id']})
                assert response.status_code == 200, response.text
                experiment = response.json()
            else:
                recording = Path('/depot/jgmakin/data/auditory_cortex/datasets/ucdavisBAK/Data/psydwaze_2026-01-14a_boilermaker.mat')
                response = client.post(prefix + '/recording-upload', content=recording.read_bytes(), headers={
                    'Content-Type': 'application/octet-stream', 'X-Recording-Filename': recording.name})
                assert response.status_code == 201, response.text
                experiment = response.json()['experiment']
            response = client.post(prefix + '/fits', json={'request_id': str(uuid4()),
                'recording_id': experiment['recording']['id'],
                'recording_revision': experiment['recording_revision'], 'seed': 42})
            assert response.status_code == 202, response.text
            fit = response.json()
            before = time.monotonic()
            with (root / (source + '-worker.log')).open('w') as log:
                subprocess.run([sys.executable, '-m', 'backend.worker', '--once'], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            final = client.get('/api/v1/fits/' + fit['id']).json()
            assert final['state'] == 'succeeded', final
            actual = {row['channel_id']: row['correlation'] for row in final['correlations']}
            assert set(actual) == set(baseline), (actual, baseline)
            delta = max(abs(actual[key] - baseline[key]) for key in baseline)
            entry = {'source': source, 'fit_id': fit['id'], 'job_id': fit['job_id'],
                     'seconds': time.monotonic() - before, 'channels': len(actual),
                     'max_absolute_legacy_delta': delta, 'session_id': experiment['session_id']}
            report['fits'].append(entry)
            destination.write_text(json.dumps(report, indent=2) + '\n')
            assert delta == 0.0, entry
            print(json.dumps(entry), flush=True)
        report['completed'] = True
    except Exception as error:
        report['error'] = repr(error)
        traceback.print_exc()
    finally:
        report['total_seconds'] = time.monotonic() - started
        destination.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    if not report['completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
