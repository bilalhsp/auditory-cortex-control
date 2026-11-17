"""Immutable fit submission and private recording snapshots."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.config import Settings
from backend.models import Fit, FitSubmission, Experiment
from backend.recordings import RecordingCatalog, ResolvedRecording
from backend.store import ExperimentStore


class FitRequestError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def submit_fit(settings: Settings, store: ExperimentStore, catalog: RecordingCatalog,
               experiment: Experiment, request: FitSubmission, available_gpus: int, retry_source=None) -> Fit:
    previous = store.fit_for_request(experiment.id, str(request.request_id))
    if previous is not None:
        if (previous.recording.id, previous.recording_revision, previous.seed) != (str(request.recording_id), request.recording_revision, request.seed):
            raise FitRequestError(409, 'Request identity was already used for a different fit.')
        return previous
    if (experiment.recording is None or experiment.recording.id != str(request.recording_id)
            or experiment.recording_revision != request.recording_revision):
        raise FitRequestError(409, 'Recording selection changed. Reload and submit again.')
    if experiment.configuration.gpu_count < 1 or available_gpus < 1:
        raise FitRequestError(409, 'No GPU is available for fitting. Refresh capacity and retry.')
    uploaded = None
    if experiment.recording.source == 'upload':
        uploaded = store.get_upload(experiment.recording.id, experiment.id)
        if uploaded is None or not uploaded.path.is_file():
            raise FitRequestError(404, 'Uploaded recording is missing. Upload it again.')
        data_root, metadata_path = catalog.locations()
        resolved = ResolvedRecording(uploaded.recording, 0, uploaded.path, metadata_path, data_root)
    else:
        resolved = catalog.resolve(experiment.configuration.dataset_name, experiment.recording.id)
    if resolved is None:
        raise FitRequestError(404, 'Recording is missing or no longer available.')
    if experiment.configuration.stimulus_family not in resolved.recording.stimulus_families:
        raise FitRequestError(409, 'Recording is incompatible with this stimulus family.')
    fit = Fit(id=str(uuid4()), job_id=str(uuid4()), experiment_id=experiment.id,
              recording=resolved.recording, recording_revision=experiment.recording_revision,
              configuration=experiment.configuration, seed=request.seed,
              created_at=datetime.now(timezone.utc).isoformat())
    root = settings.fit_dir or settings.database_path.parent / 'fits'
    workspace = root / fit.id
    retained = False
    created = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(mode=0o700)
        created = True
        inputs = workspace / 'input'
        inputs.mkdir()
        recording = inputs / resolved.path.name
        metadata = inputs / 'sessions.yml'
        shutil.copyfile(str(resolved.path), str(recording))
        shutil.copyfile(str(resolved.metadata_path), str(metadata))
        recording_hash = file_sha256(recording)
        if uploaded is not None and (recording_hash != uploaded.sha256 or recording.stat().st_size != uploaded.size_bytes):
            raise FitRequestError(409, 'Uploaded recording changed after validation. Upload it again.')
        snapshot = {'native_session_id': resolved.session_id, 'recording_file': str(recording.relative_to(workspace)),
                    'metadata_file': 'input/sessions.yml', 'recording_sha256': recording_hash,
                    'metadata_sha256': file_sha256(metadata), 'neural_data_dir': str(resolved.data_root.resolve())}
        (workspace / 'input.json').write_text(json.dumps(snapshot, indent=2))
        accepted = store.enqueue_fit(fit, str(request.request_id), workspace, retry_source)
        retained = accepted.id == fit.id
        return accepted
    except ValueError as error:
        raise FitRequestError(409, str(error)) from error
    except OSError as error:
        raise FitRequestError(503, 'Fit storage is unavailable. Restore storage and retry.') from error
    finally:
        if created and not retained:
            shutil.rmtree(str(workspace))
