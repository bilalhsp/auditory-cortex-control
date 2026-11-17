import sys, time, json, traceback
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), '/home/ahmedb/projects/deepspeech.pytorch', '/home/ahmedb/projects/Wav2Letter', '/home/ahmedb/projects/utils_jgm', '/home/ahmedb/projects/pycochleagram']
import yaml
cache = Path(yaml.safe_load(Path('auditory_cortex/config.yml').read_text())['cache_dir'])
out = cache / 'web-migration-baseline' / 'issue-01'
out.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
report = {'phase': 'import', 'completed': False}
try:
    import numpy as np
    import torch
    from auditory_cortex.optimal_stimulus.generator import StimGenerator, Sampler
    import auditory_cortex.io_utils.io as io
    from auditory_cortex.neural_data import create_neural_dataset
    for name in ('write_trf_parameters',):
        original = getattr(io, name)
        def isolated(*args, _original=original, **kwargs):
            previous = io.cache_dir
            io.cache_dir = out / 'fit-cache'
            try:
                return _original(*args, **kwargs)
            finally:
                io.cache_dir = previous
        setattr(io, name, isolated)
    report['phase'] = 'initialize'
    print('INITIALIZING', flush=True)
    gen = StimGenerator('ucdavisBAK', 'deepspeech2', 2, mVocs=False, bin_width=50, lag=200)
    report['initialization_seconds'] = time.monotonic() - started
    source = Path('/depot/jgmakin/data/auditory_cortex/datasets/ucdavisBAK/Data/psydwaze_2026-01-14a_boilermaker.mat')
    report['fits'] = []
    for kind, value in [('existing', 0), ('existing', 0), ('upload', str(source))]:
        report['phase'] = 'fit_' + kind
        np.random.seed(42)
        torch.manual_seed(42)
        before = time.monotonic()
        session, corr = gen.fit_encoding_model(value, kind)
        entry = {'source':kind, 'session':session, 'seconds':time.monotonic()-before, 'correlations':{str(k):float(v) for k,v in corr.items()}}
        report['fits'].append(entry)
        (out / 'probe.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print('FIT_COMPLETE', kind, entry['seconds'], flush=True)
    report['completed'] = True
except Exception as exc:
    report['error'] = repr(exc)
    traceback.print_exc()
finally:
    report['total_seconds'] = time.monotonic()-started
    (out / 'probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report['completed']:
        sys.exit(1)
