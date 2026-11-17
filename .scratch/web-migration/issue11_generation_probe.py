"""Controlled real mVocs legacy/web audio comparison after issue11_native_probe.

Uses identical recorded initialization/RNG in one process, as approved in issue07.
It does not claim cross-process or cross-hardware determinism.
"""
import gc
import hashlib
import io
import json
import time
import traceback
import zipfile
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.auth import hash_password
from backend.config import Settings
from backend.generation_compute import scientific_generation
from backend.worker import run_worker


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import cupy as cp
    from auditory_cortex.diffusion.utils import seed_everything
    from auditory_cortex.optimal_stimulus.generator import Sampler
    from auditory_cortex.io_utils.io import read_dict

    fits = json.loads(Path('.scratch/web-migration/issue11-native.json').read_text())
    assert fits['completed'], 'Native mVocs fits must pass first'
    assert [fit['source'] for fit in fits['fits']] == ['existing', 'upload']
    root = Path(fits['root'])
    report = {'completed': False, 'family': 'mVocs', 'root': str(root), 'results': [],
              'acceptance': 'Approved controlled legacy/web initialization and RNG comparison'}
    destination = Path('.scratch/web-migration/issue11-generation.json')
    started = time.monotonic()
    try:
        parameters = read_dict(root / 'legacy-model.pkl')
        settings = Settings(root / 'sessions.sqlite3', 'operator', hash_password('test-password'))
        client = TestClient(create_app(settings))
        client.auth = ('operator', 'test-password')
        for objective, fit in zip(['stretch', 'one-hot'], fits['fits']):
            workspace = root / 'fits' / fit['fit_id']
            config = json.loads((workspace / 'trf_config.json').read_text())
            saved = read_dict(workspace / 'model.pkl')
            assert set(saved) == set(parameters), 'Fit parameter keys differ'
            deltas = []
            for key, expected_parameter in parameters.items():
                if expected_parameter is None or saved[key] is None:
                    assert expected_parameter is None and saved[key] is None, key
                    continue
                actual_parameter = cp.asnumpy(saved[key])
                expected_parameter = cp.asnumpy(expected_parameter)
                assert actual_parameter.shape == expected_parameter.shape, key
                deltas.append(float(np.max(abs(actual_parameter - expected_parameter))))
            parameter_delta = max(deltas)
            assert parameter_delta == 0.0, parameter_delta
            ready = client.get('/api/v1/fits/' + fit['fit_id']).json()
            channel = ready['best_channel']['channel_id']
            seed_everything(42)
            rng = {'cpu_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                   'cuda_sha256': hashlib.sha256(torch.cuda.get_rng_state(0).cpu().numpy().tobytes()).hexdigest()}
            before = time.monotonic()
            sampler = Sampler(torch.device('cuda:0'), model_name='deepspeech2', session_id=0,
                layer_id=2, bin_width=50, mVocs=True, lag=200, dataset_name='ucdavisBAK',
                unit_ids=config['unit_ids'], artifact_path=root / 'legacy-model.pkl')
            files = sampler.generate_and_save_stimuli(root / 'legacy-audio' / objective,
                unit_id=float(channel), task=objective, duration=1, n_stimuli=1, seed=42, stim_type='mvocs')
            legacy = Path(files[0]['path'])
            legacy_seconds = time.monotonic() - before
            del sampler
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()
            response = client.post('/api/v1/fits/' + fit['fit_id'] + '/generations', json={
                'request_id': str(uuid4()), 'channel_id': channel, 'objective': objective,
                'duration_seconds': 1, 'clip_count': 1, 'seed': 42})
            assert response.status_code == 202, response.text
            job = response.json()
            def compute(task, progress):
                progress('generating')
                scientific_generation(task.workspace)
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.empty_cache()
                return 0
            before = time.monotonic()
            run_worker(settings, once=True, generation_runner=compute)
            final = client.get('/api/v1/generations/' + job['id']).json()
            assert final['state'] == 'succeeded', final
            response = client.get('/api/v1/generations/' + job['id'] + '/download')
            assert response.status_code == 200, response.text
            archive = zipfile.ZipFile(io.BytesIO(response.content))
            assert archive.namelist() == ['clip-001.wav']
            data = archive.read('clip-001.wav')
            provenance = json.loads((root / 'generations' / job['id'] / 'compute-provenance.json').read_text())
            assert provenance['initialization_rng'] == rng
            actual, rate = sf.read(io.BytesIO(data))
            expected, _ = sf.read(legacy)
            entry = {'objective': objective, 'source': fit['source'], 'fit_id': fit['fit_id'],
                     'generation_id': job['id'], 'job_id': job['job_id'], 'channel_id': channel,
                     'legacy_seconds': legacy_seconds, 'web_seconds': time.monotonic() - before,
                     'max_fit_parameter_delta': parameter_delta, 'initialization_rng': rng,
                     'frames': len(actual), 'rate': rate, 'subtype': sf.info(io.BytesIO(data)).subtype,
                     'web_sha256': hashlib.sha256(data).hexdigest(), 'legacy_sha256': hashlib.sha256(legacy.read_bytes()).hexdigest(),
                     'max_absolute_pcm_difference': float(np.max(abs(actual - expected))),
                     'rms_pcm_difference': float(np.sqrt(np.mean((actual - expected)**2))),
                     'byte_identical': data == legacy.read_bytes(), 'exit_code': final['exit_code']}
            report['results'].append(entry)
            destination.write_text(json.dumps(report, indent=2) + '\n')
            assert (entry['frames'], entry['rate'], entry['subtype'], entry['exit_code']) == (15392, 16000, 'PCM_16', 0)
            assert entry['byte_identical'], entry
            print(json.dumps(entry), flush=True)
        assert len(report['results']) == 2
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
