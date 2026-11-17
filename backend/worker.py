"""Single-owner persistent fit queue. Run separately from the HTTP server."""
import argparse
import fcntl
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Callable

from backend.config import Settings
from backend.fit_compute import FitOutput, FitTask, scientific_fit
from backend.fits import file_sha256
from backend.models import ChannelCorrelation
from backend.generation_compute import GenerationTask, subprocess_generation
from backend.generations import validate_clips, verify_fit
from backend.store import ExperimentStore


FitRunner = Callable[[FitTask, Callable[[str], None]], FitOutput]
logger = logging.getLogger(__name__)


def run_next_fit(settings: Settings, runner: FitRunner = scientific_fit) -> bool:
    store = ExperimentStore(settings.database_path)
    claimed = store.claim_fit()
    if claimed is None:
        return False
    fit, workspace = claimed
    handler = None
    try:
        handler = logging.FileHandler(str(workspace / 'diagnostics.log'))
        logging.getLogger().addHandler(handler)
        def progress(stage: str) -> None:
            if stage not in {'loading', 'fitting', 'writing'}:
                raise ValueError('Unknown fit phase')
            fit.stage = stage
            store.update_fit(fit)
        output = runner(FitTask(fit, workspace), progress)
        rows = [ChannelCorrelation(channel_id=str(channel), correlation=float(value) if math.isfinite(float(value)) else None)
                for channel, value in output.correlations.items()]
        rows.sort(key=lambda item: (item.correlation is None, -(item.correlation or 0.0), item.channel_id))
        manifest = {}
        for filename in ['model.pkl', 'trf_config.json']:
            path = workspace / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError('Missing fit artifact')
            manifest[filename] = file_sha256(path)
        json.loads((workspace / 'trf_config.json').read_text())
        provenance = dict(output.provenance, fit_id=fit.id, job_id=fit.job_id,
                          configuration=fit.configuration.model_dump(), recording_id=fit.recording.id,
                          recording_revision=fit.recording_revision, seed=fit.seed)
        (workspace / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False))
        manifest['provenance.json'] = file_sha256(workspace / 'provenance.json')
        (workspace / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        fit.correlations = rows
        fit.best_channel = next((row for row in rows if row.correlation is not None), None)
        fit.artifacts_ready = True
        fit.state, fit.stage = 'succeeded', 'complete'
    except Exception:
        logger.exception('Fit job %s failed during %s', fit.job_id, fit.stage)
        fit.error = {'loading': 'Could not load the encoding model or recording.',
                     'fitting': 'Encoding model fitting failed.',
                     'writing': 'Could not save the fitted model artifacts.'}.get(fit.stage, 'Fit computation failed.')
        fit.state = 'failed'
        fit.artifacts_ready = False
        fit.correlations = []
        fit.best_channel = None
    finally:
        fit.finished_at = datetime.now(timezone.utc).isoformat()
        store.update_fit(fit)
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
    return True


def run_next_generation(settings: Settings, runner=subprocess_generation) -> bool:
    store = ExperimentStore(settings.database_path)
    claimed = store.claim_generation()
    if claimed is None:
        return False
    generation, workspace = claimed
    try:
        snapshot = json.loads((workspace / 'input.json').read_text())
        from pathlib import Path
        if verify_fit(Path(snapshot['fit_workspace'])) != snapshot['fit_manifest']:
            raise ValueError('Pinned fit changed')
        def progress(stage: str) -> None:
            if stage not in {'loading', 'generating', 'writing'}:
                raise ValueError('Unknown generation phase')
            generation.stage = stage
            store.update_generation(generation)
        generation.exit_code = runner(GenerationTask(generation, workspace), progress)
        if generation.exit_code != 0:
            raise RuntimeError('Generation process exited unsuccessfully')
        progress('writing')
        clips = validate_clips(generation, workspace)
        (workspace / 'manifest.json').write_text(json.dumps([clip.model_dump() for clip in clips], indent=2))
        provenance = {'generation': generation.model_dump(), 'fit_manifest': snapshot['fit_manifest'],
                      'exit_code': generation.exit_code}
        (workspace / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        generation.clips = clips
        generation.artifacts_ready = True
        generation.state, generation.stage = 'succeeded', 'complete'
    except Exception:
        generation.error = 'Generation failed. Check GPU/model availability and retry with a new request.'
        generation.state = 'failed'
        generation.artifacts_ready = False
        generation.clips = []
        try:
            with (workspace / 'diagnostics.log').open('a') as log:
                import traceback
                traceback.print_exc(file=log)
        except OSError:
            logger.exception('Could not write diagnostics for generation %s', generation.id)
    finally:
        generation.finished_at = datetime.now(timezone.utc).isoformat()
        store.update_generation(generation)
    return True


def run_worker(settings: Settings, once: bool = False, runner: FitRunner = scientific_fit, generation_runner=subprocess_generation) -> None:
    # Kernel-owned lock survives API restarts and releases only when this compute
    # process exits. A killed worker cannot leave a separate child using GPUs.
    lock_path = settings.database_path.resolve().with_suffix('.fit-worker.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('A fit worker already owns this database') from error
        ExperimentStore(settings.database_path).interrupt_running_fits()
        ExperimentStore(settings.database_path).interrupt_running_generations()
        while True:
            did_work = run_next_fit(settings, runner) or run_next_generation(settings, generation_runner)
            if once:
                return
            if not did_work:
                time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', help='Process one queued fit then exit')
    args = parser.parse_args()
    run_worker(Settings.from_environment(), once=args.once)
