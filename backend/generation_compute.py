"""Isolated scientific generation process; only trusted server artifacts are loaded."""
import ctypes
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from backend.models import Generation
from backend.gpu import device_environment


@dataclass(frozen=True)
class GenerationTask:
    generation: Generation
    workspace: Path
    device_index: int = 0


def subprocess_generation(task: GenerationTask, progress: Callable[[str], None]) -> int:
    progress('generating')
    env = device_environment(task.device_index)
    with (task.workspace / 'diagnostics.log').open('a') as log:
        result = subprocess.run([sys.executable, '-m', 'backend.generation_compute', str(task.workspace.resolve()), str(os.getpid())],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def scientific_generation(workspace: Path) -> None:
    from backend.generations import verify_fit
    data = json.loads((workspace / 'input.json').read_text())
    generation = Generation.model_validate(data['generation'])
    fit_workspace = Path(data['fit_workspace'])
    if verify_fit(fit_workspace) != data['fit_manifest']:
        raise ValueError('Pinned fit changed')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable')
    torch.cuda.set_device(0)
    from auditory_cortex.optimal_stimulus.generator import Sampler, CONF_DIR
    from auditory_cortex import pretrained_dir
    from auditory_cortex.utils import load_dnn_config
    from backend.fits import file_sha256
    cfg = json.loads((fit_workspace / 'trf_config.json').read_text())
    fit_provenance = json.loads((fit_workspace / 'provenance.json').read_text())
    checkpoint = pretrained_dir / cfg['model_name'] / load_dnn_config(model_name=cfg['model_name'])['saved_checkpoint']
    if file_sha256(checkpoint) != fit_provenance['checkpoint_sha256']:
        raise ValueError('Encoding checkpoint changed since fitting')
    inputs = {'sampler_config': CONF_DIR / 'reps-config/sampler/daps.yaml',
              'audioldm_checkpoint': pretrained_dir / 'audioldm/audioldm-s-full.ckpt',
              'encoding_checkpoint': checkpoint}
    hashes = {name: file_sha256(path) for name, path in inputs.items()}
    sampler_configuration = inputs['sampler_config'].read_text()
    units = cfg['unit_ids']
    target = float(generation.channel_id)
    if target not in units:
        raise ValueError('Target does not belong to the fitted model')
    from auditory_cortex.diffusion.utils import seed_everything
    seed_everything(generation.seed)
    import hashlib
    initialization_rng = {
        'cpu_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        'cuda_sha256': hashlib.sha256(torch.cuda.get_rng_state(0).cpu().numpy().tobytes()).hexdigest(),
    }
    sampler = Sampler(torch.device('cuda:0'), model_name=cfg['model_name'], session_id=cfg['session_id'],
                      layer_id=cfg['layer_id'], bin_width=cfg['bin_width'], mVocs=cfg['mVocs'],
                      lag=cfg['lag'], dataset_name=cfg['dataset_name'], unit_ids=units,
                      artifact_path=fit_workspace / 'model.pkl')
    files = sampler.generate_and_save_stimuli(workspace / 'raw', unit_id=target,
                task=generation.objective, duration=generation.duration_seconds,
                n_stimuli=generation.clip_count, seed=generation.seed,
                stim_type='mvocs' if cfg['mVocs'] else 'timit')
    for index, item in enumerate(files):
        (workspace / ('clip-%03d.wav' % (index + 1))).write_bytes(Path(item['path']).read_bytes())
    if hashes != {name: file_sha256(path) for name, path in inputs.items()}:
        raise ValueError('Scientific inputs changed during generation')
    import platform
    from importlib.metadata import version
    provenance = {'fit_manifest': data['fit_manifest'], 'generation': generation.model_dump(),
                  'input_sha256': hashes, 'python': platform.python_version(),
                  'sampler_configuration': sampler_configuration,
                  'initialization_seed': generation.seed, 'initialization_rng': initialization_rng,
                  'packages': {name: version(name) for name in ('torch', 'cupy-cuda12x', 'numpy', 'scipy')},
                  'generator_code_sha256': file_sha256(CONF_DIR.parent / 'auditory_cortex/optimal_stimulus/generator.py')}
    (workspace / 'compute-provenance.json').write_text(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    # Linux parent-death signal protects the allocator lock: killing the owning
    # worker also kills its GPU child, including the startup race before prctl.
    expected_parent = int(sys.argv[2])
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != expected_parent:
        raise SystemExit('Generation worker parent exited')
    scientific_generation(Path(sys.argv[1]))
