"""Scientific compute boundary. Imported only by the dedicated fit worker."""
import gc
import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

from backend.fits import file_sha256
from backend.models import Fit


@dataclass(frozen=True)
class FitTask:
    fit: Fit
    workspace: Path
    cancelled: Callable[[], bool] = lambda: False
    allocator_fd: Optional[int] = None


@dataclass(frozen=True)
class FitOutput:
    correlations: Mapping[str, float]
    provenance: Dict


def scientific_fit(task: FitTask, progress: Callable[[str], None]) -> FitOutput:
    # PYTHONPATH must include the research repository and its existing local
    # scientific dependency checkouts. No model import belongs in the HTTP API.
    import auditory_cortex
    snapshot = json.loads((task.workspace / 'input.json').read_text())
    auditory_cortex.neural_data_dir = Path(snapshot['neural_data_dir'])
    import numpy as np
    import torch
    from torch.version import cuda
    import cupy as cp
    from auditory_cortex.neural_data import create_neural_dataset
    from auditory_cortex.optimal_stimulus.generator import StimGenerator
    from auditory_cortex.utils import load_dnn_config

    progress('loading')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for fitting')
    recording = task.workspace / snapshot['recording_file']
    metadata = task.workspace / snapshot['metadata_file']
    if file_sha256(recording) != snapshot['recording_sha256'] or file_sha256(metadata) != snapshot['metadata_sha256']:
        raise RuntimeError('Pinned recording inputs changed')
    cfg = task.fit.configuration
    uploaded = task.fit.recording.source == 'upload'
    dataset = create_neural_dataset(cfg.dataset_name, recording if uploaded else snapshot['native_session_id'],
                                    full_path=uploaded,
                                    recording_path=recording, metadata_config_path=metadata)
    checkpoint_name = load_dnn_config(model_name=cfg.model_name)['saved_checkpoint']
    checkpoint = auditory_cortex.pretrained_dir / cfg.model_name / checkpoint_name
    checkpoint_sha = file_sha256(checkpoint)
    generator = StimGenerator(cfg.dataset_name, cfg.model_name, cfg.layer_id,
                              mVocs=cfg.stimulus_family == 'mVocs', bin_width=cfg.bin_width_ms,
                              lag=cfg.receptive_field_ms, neural_dataset=dataset)
    try:
        np.random.seed(task.fit.seed)
        torch.manual_seed(task.fit.seed)
        progress('fitting')
        _, correlations = generator.fit_encoding_model(snapshot['native_session_id'], neural_dataset=dataset,
                                                       session_source='upload' if uploaded else 'existing',
                                                       artifact_path=task.workspace / 'model.pkl')
        progress('writing')
        generator.save_trf_config(task.workspace)
        if file_sha256(checkpoint) != checkpoint_sha:
            raise RuntimeError('Checkpoint changed during fitting')
        scientific_root = Path(auditory_cortex.__file__).resolve().parent
        code_digest = hashlib.sha256()
        for path in sorted(scientific_root.rglob('*.py')):
            code_digest.update(str(path.relative_to(scientific_root)).encode())
            code_digest.update(path.read_bytes())
        provenance = {
            'recording_sha256': snapshot['recording_sha256'], 'metadata_sha256': snapshot['metadata_sha256'],
            'checkpoint_sha256': checkpoint_sha, 'scientific_code_sha256': code_digest.hexdigest(),
            'python': platform.python_version(), 'packages': {name: version(name) for name in ['torch', 'cupy-cuda12x', 'numpy', 'scipy']},
            'cuda_build': cuda, 'seed': task.fit.seed, 'cross_validation_folds': 3,
        }
        return FitOutput({str(key): float(value) for key, value in correlations.items()}, provenance)
    finally:
        del generator, dataset
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()


def subprocess_fit(task: FitTask, progress: Callable[[str], None]) -> FitOutput:
    from backend.gpu import device_environment
    from backend.processes import run_compute
    (task.workspace / 'fit-task.json').write_text(task.fit.model_dump_json())
    last_stage = [None]
    def tick():
        path = task.workspace / 'progress.json'
        if path.exists():
            stage = json.loads(path.read_text())['stage']
            if stage != last_stage[0]:
                progress(stage);last_stage[0] = stage
    code = run_compute('backend.fit_compute', task.workspace, task.cancelled, device_environment(0), tick, lock_fd=task.allocator_fd)
    if code != 0: raise RuntimeError('Fit process exited unsuccessfully')
    output = json.loads((task.workspace / 'fit-output.json').read_text())
    return FitOutput(output['correlations'], output['provenance'])


if __name__ == '__main__':
    import sys
    from backend.processes import require_parent
    require_parent(int(sys.argv[2]))
    workspace = Path(sys.argv[1])
    task = FitTask(Fit.model_validate_json((workspace / 'fit-task.json').read_text()), workspace)
    def publish_progress(stage):
        path = workspace / 'progress.next'
        path.write_text(json.dumps({'stage': stage}))
        path.replace(workspace / 'progress.json')
    output = scientific_fit(task, publish_progress)
    (workspace / 'fit-output.json').write_text(json.dumps({'correlations': dict(output.correlations), 'provenance': output.provenance}))
