"""Run real repeatability checks; keep scientific artifacts in configured cache."""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path[:0] = [str(Path.cwd()), '/home/ahmedb/projects/deepspeech.pytorch',
               '/home/ahmedb/projects/Wav2Letter', '/home/ahmedb/projects/utils_jgm',
               '/home/ahmedb/projects/pycochleagram']
import yaml

cache = Path(yaml.safe_load(Path('auditory_cortex/config.yml').read_text())['cache_dir'])
root = cache / 'web-migration-baseline' / 'issue-01'
report = {'completed': False, 'phase': 'load_fit', 'outputs': []}
started = time.monotonic()
try:
    import numpy as np
    import soundfile as sf
    import torch
    from auditory_cortex.optimal_stimulus.generator import Sampler
    import auditory_cortex.io_utils.io as io

    fit = json.loads((root / 'probe.json').read_text())
    assert fit['completed'], 'Fitting baseline must complete first'
    correlations = fit['fits'][-1]['correlations']
    channels = [float(channel) for channel in correlations]
    target = float(max(correlations, key=lambda key: correlations[key]))
    io.cache_dir = root / 'fit-cache'
    report['phase'] = 'initialize_sampler'
    print('INITIALIZING_SAMPLER', flush=True)
    sampler = Sampler(torch.device('cuda:0'), model_name='deepspeech2', session_id=0,
                      layer_id=2, bin_width=50, mVocs=False, lag=200,
                      dataset_name='ucdavisBAK', unit_ids=channels)
    report['initialization_seconds'] = time.monotonic() - started
    for objective in ('stretch', 'one-hot'):
        waves = []
        for repeat in range(2):
            report['phase'] = f'{objective}_{repeat}'
            before = time.monotonic()
            files = sampler.generate_and_save_stimuli(
                root / 'audio' / objective / str(repeat), unit_id=target,
                task=objective, duration=1, n_stimuli=1, seed=42, stim_type='timit')
            assert len(files) == 1
            path = Path(files[0]['path'])
            info = sf.info(path)
            waveform, rate = sf.read(path)
            assert rate == 16000 and info.subtype == 'PCM_16'
            assert np.isfinite(waveform).all() and np.max(np.abs(waveform)) <= 1
            waves.append(waveform)
            report['outputs'].append({'objective': objective, 'repeat': repeat,
                'path': str(path), 'frames': info.frames, 'rate': rate,
                'subtype': info.subtype, 'seconds': time.monotonic() - before})
            print('SAMPLE_COMPLETE', report['outputs'][-1], flush=True)
        assert waves[0].shape == waves[1].shape
        report.setdefault('repeatability', {})[objective] = {
            'max_absolute_pcm_difference': float(np.max(np.abs(waves[0] - waves[1]))),
            'rms_pcm_difference': float(np.sqrt(np.mean((waves[0] - waves[1])**2)))}
    report['completed'] = True
except Exception as exc:
    report['error'] = repr(exc)
    traceback.print_exc()
finally:
    report['total_seconds'] = time.monotonic() - started
    (root / 'generation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not report['completed']:
        sys.exit(1)
