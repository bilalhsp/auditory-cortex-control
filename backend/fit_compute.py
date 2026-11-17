"""Scientific compute boundary. Imported only by the dedicated fit worker."""
import gc
import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Dict, Mapping

from backend.fits import file_sha256
from backend.models import Fit


@dataclass(frozen=True)
class FitTask:
    fit: Fit
    workspace: Path


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
    dataset = create_neural_dataset(cfg.dataset_name, snapshot['native_session_id'],
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
