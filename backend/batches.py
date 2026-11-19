"""Transactional batch submission and bounded workers with one slot per device."""
import io
import json
import logging
import traceback
import os
import queue
import shutil
import sqlite3
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.fits import FitRequestError
from backend.generations import ready_fit, download_zip
from backend.gpu import visibility_signature
from backend.models import Batch, BatchSubmission, Generation
from backend.store import ExperimentStore

logger = logging.getLogger(__name__)


def submit_batch(settings, store: ExperimentStore, fit_id: str, request: BatchSubmission, visible_count: int, retry_source=None, limit_to_experiment=False):
    command = request.model_dump(mode='json')
    previous = store.batch_for_request(fit_id, str(request.request_id))
    if previous is not None:
        if previous[1] != command:
            raise FitRequestError(409, 'Request identity was already used for different batch settings.')
        return previous[0]
    fit, fit_workspace, manifest = ready_fit(store, fit_id)
    if limit_to_experiment and (
            request.gpu_count > fit.configuration.gpu_count or len(request.targets) > request.gpu_count):
        raise FitRequestError(422, 'Select no more targets than GPUs, within the experiment GPU limit.')
    channels = {row.channel_id for row in fit.correlations if row.correlation is not None}
    if any(target.channel_id not in channels for target in request.targets):
        raise FitRequestError(422, 'Choose available channels from this fit.')
    now = datetime.now(timezone.utc).isoformat()
    targets = [Generation(id=str(uuid4()), job_id=str(uuid4()), fit_id=fit.id, experiment_id=fit.experiment_id,
                          configuration=fit.configuration, channel_id=target.channel_id, objective=target.objective,
                          duration_seconds=request.duration_seconds, clip_count=request.clip_count, seed=request.seed,
                          unused_gpu_count=0, created_at=now) for target in request.targets]
    used = min(request.gpu_count, len(targets))
    batch = Batch(id=str(uuid4()), job_id=str(uuid4()), fit_id=fit.id, experiment_id=fit.experiment_id,
                  configuration=fit.configuration, gpu_count=request.gpu_count, used_gpu_count=used,
                  unused_gpu_count=request.gpu_count-used, targets=targets, created_at=now)
    workspace = settings.database_path.parent / 'batches' / batch.id
    retained = False
    created = False
    try:
        workspace.parent.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(mode=0o700)
        created = True
        for target in targets:
            directory = workspace / target.id
            directory.mkdir()
            (directory / 'input.json').write_text(json.dumps({'generation': target.model_dump(),
                'fit_workspace': str(fit_workspace.resolve()), 'fit': fit.model_dump(), 'fit_manifest': manifest}, indent=2))
        accepted = store.enqueue_batch(batch, command, workspace, visible_count, visibility_signature(), retry_source)
        retained = accepted.id == batch.id
        return store.require_batch(accepted.id)[0]
    except ValueError as error:
        raise FitRequestError(409, str(error)) from error
    except (OSError, sqlite3.Error) as error:
        raise FitRequestError(503, 'Batch storage is unavailable. Restore storage and retry.') from error
    finally:
        if created and not retained:
            shutil.rmtree(str(workspace), ignore_errors=True)


def run_next_batch(settings, runner, gpu_count, allocator_fd=None) -> bool:
    from backend.worker import execute_generation
    store = ExperimentStore(settings.database_path)
    claimed = store.claim_batch()
    if claimed is None:
        return False
    batch, workspace, devices, visibility = claimed
    try:
        (workspace / 'diagnostics.log').write_text(json.dumps({'batch_id': batch.id, 'requested_gpus': batch.gpu_count, 'reserved_local_indices': devices}) + '\n')
        if (visibility_signature() != visibility or gpu_count() < batch.gpu_count
                or any(index >= gpu_count() for index in devices)):
            raise RuntimeError('GPU capacity or visibility changed before allocation')
        pending = queue.Queue()
        for target in batch.targets:
            pending.put(target)
        def slot(device):
            while True:
                try: target = pending.get_nowait()
                except queue.Empty: return
                if visibility_signature() != visibility or device >= gpu_count():
                    target.state, target.stage = 'failed', 'loading'
                    target.error = 'GPU capacity changed. Submit a new batch to retry.'
                    target.finished_at = datetime.now(timezone.utc).isoformat()
                    store.update_generation(target)
                else:
                    execute_generation(store, target, workspace / target.id, runner, device, allocator_fd)
        with ThreadPoolExecutor(max_workers=batch.used_gpu_count) as executor:
            list(executor.map(slot, devices[:batch.used_gpu_count]))
        batch.targets = store.require_batch(batch.id)[0].targets
        completed = sum(target.state == 'succeeded' for target in batch.targets)
        batch.state = 'succeeded' if completed == len(batch.targets) else 'partial' if completed else 'cancelled' if all(target.state == 'cancelled' for target in batch.targets) else 'failed'
        batch.artifacts_ready = completed > 0
        if batch.state != 'succeeded': batch.error = 'Some targets failed. Only completed targets are downloadable.'
    except Exception:
        try:
            with (workspace / 'diagnostics.log').open('a') as log:
                traceback.print_exc(file=log)
        except OSError:
            logger.exception('Could not write batch diagnostics for %s', batch.id)
        # All executor threads have exited before releasing this reservation.
        batch.targets = store.require_batch(batch.id)[0].targets
        for target in batch.targets:
            if target.state in ('queued', 'running'):
                target.state, target.stage = 'failed', 'loading'
                target.error = 'GPU allocation failed. Refresh capacity and submit a new batch.'
                target.finished_at = datetime.now(timezone.utc).isoformat()
                store.update_generation(target)
        batch.state = 'partial' if any(target.state == 'succeeded' for target in batch.targets) else 'failed'
        batch.artifacts_ready = any(target.state == 'succeeded' for target in batch.targets)
        batch.error = 'GPU allocation failed or capacity changed. Refresh capacity and submit a new batch.'
    finally:
        batch.finished_at = datetime.now(timezone.utc).isoformat()
        store.update_batch(batch)
    return True


def batch_manifest(batch: Batch):
    return {'batch_id': batch.id, 'state': batch.state, 'targets': [
        {'id': target.id, 'channel_id': target.channel_id, 'objective': target.objective,
         'state': target.state, 'clips': [clip.model_dump() for clip in target.clips] if target.state == 'succeeded' else []}
        for target in batch.targets]}


def build_batch_zip(batch: Batch, workspace: Path) -> Path:
    if batch.state not in ('succeeded', 'partial', 'interrupted', 'cancelled') or not batch.artifacts_ready:
        raise FitRequestError(409, 'Batch has no completed downloadable results.')
    handle, name = tempfile.mkstemp(prefix='download-', suffix='.zip', dir=str(workspace))
    os.close(handle)
    output = Path(name)
    try:
        with zipfile.ZipFile(str(output), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for target in batch.targets:
                if target.state != 'succeeded': continue
                with zipfile.ZipFile(io.BytesIO(download_zip(target, workspace / target.id))) as clips:
                    for filename in clips.namelist():
                        archive.writestr(target.id + '/' + filename, clips.read(filename))
            archive.writestr('manifest.json', json.dumps(batch_manifest(batch), indent=2))
        return output
    except Exception:
        output.unlink(missing_ok=True)
        raise
