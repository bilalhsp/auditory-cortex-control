"""Generation submission, artifact verification, and bounded authenticated ZIPs."""
import io
import json
import logging
import shutil
import sqlite3
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.config import Settings
from backend.fits import FitRequestError, file_sha256
from backend.models import Generation, GenerationSubmission, GeneratedClip
from backend.store import ExperimentStore

# Native AudioLDM latent rounding: preserve output length rather than trimming.
NATIVE_FRAMES = {1: 15392, 2: 35872, 3: 51232, 4: 66592}
logger = logging.getLogger(__name__)


def verify_fit(workspace: Path) -> dict:
    manifest = json.loads((workspace / 'manifest.json').read_text())
    for name in ('model.pkl', 'trf_config.json', 'provenance.json'):
        path = workspace / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != manifest[name]:
            raise ValueError('Fit artifact is missing or changed')
    return manifest


def ready_fit(store: ExperimentStore, fit_id: str):
    fit = store.get_fit(fit_id)
    if fit is None:
        raise FitRequestError(404, 'Fit not found.')
    if fit.state != 'succeeded' or not fit.artifacts_ready:
        raise FitRequestError(409, 'A successfully completed fit is required.')
    experiment = store.get(fit.experiment_id)
    if (experiment is None or experiment.recording is None or
            (experiment.recording.id, experiment.recording_revision) != (fit.recording.id, fit.recording_revision)):
        raise FitRequestError(409, 'Fit belongs to an earlier recording selection. Fit the current recording first.')
    fit_workspace = store.fit_workspace(fit_id)
    try:
        if fit_workspace is None:
            raise ValueError('Missing fit storage')
        manifest = verify_fit(fit_workspace)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FitRequestError(409, 'Fitted model artifacts are missing or changed. Fit again before generating.') from error
    return fit, fit_workspace, manifest

def submit_generation(settings: Settings, store: ExperimentStore, fit_id: str,
                      request: GenerationSubmission, available_gpus: int, retry_source=None) -> Generation:
    previous = store.generation_for_request(fit_id, str(request.request_id))
    if previous is not None:
        if any(getattr(previous, key) != getattr(request, key) for key in request.context_fields()):
            raise FitRequestError(409, 'Request identity was already used for different generation settings.')
        return previous
    fit, fit_workspace, manifest = ready_fit(store, fit_id)
    if not any(row.channel_id == request.channel_id and row.correlation is not None for row in fit.correlations):
        raise FitRequestError(422, 'Choose an available channel from this fit.')
    if fit.configuration.gpu_count < 1 or available_gpus < 1:
        raise FitRequestError(409, 'No GPU is available. Refresh capacity and retry.')
    generation = Generation(id=str(uuid4()), job_id=str(uuid4()), fit_id=fit.id,
                            experiment_id=fit.experiment_id, configuration=fit.configuration,
                            unused_gpu_count=max(0, fit.configuration.gpu_count - 1),
                            created_at=datetime.now(timezone.utc).isoformat(),
                            **request.model_dump(exclude={'request_id'}))
    workspace = settings.database_path.parent / 'generations' / generation.id
    retained = False
    created = False
    try:
        workspace.parent.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(mode=0o700)
        created = True
        (workspace / 'input.json').write_text(json.dumps({'generation': generation.model_dump(),
            'fit_workspace': str(fit_workspace.resolve()), 'fit': fit.model_dump(), 'fit_manifest': manifest}, indent=2))
        accepted = store.enqueue_generation(generation, str(request.request_id), workspace, retry_source)
        retained = accepted.id == generation.id
        return accepted
    except ValueError as error:
        raise FitRequestError(409, str(error)) from error
    except (OSError, sqlite3.Error) as error:
        raise FitRequestError(503, 'Generation storage is unavailable. Restore storage and retry.') from error
    finally:
        if created and not retained:
            try:
                shutil.rmtree(str(workspace))
            except OSError:
                logger.exception('Could not remove incomplete generation %s', generation.id)


def validate_clips(generation: Generation, workspace: Path):
    paths = sorted(workspace.glob('clip-*.wav'))
    if len(paths) != generation.clip_count:
        raise ValueError('Incomplete generation output')
    clips = []
    for path in paths:
        if path.is_symlink() or path.parent != workspace:
            raise ValueError('Invalid output location')
        if path.stat().st_size > 2 * NATIVE_FRAMES[generation.duration_seconds] + 4096:
            raise ValueError('Generated WAV exceeds its expected size')
        data = path.read_bytes()
        with wave.open(io.BytesIO(data), 'rb') as audio:
            if (audio.getframerate() != 16000 or audio.getsampwidth() != 2 or audio.getnchannels() != 1
                    or audio.getcomptype() != 'NONE' or audio.getnframes() != NATIVE_FRAMES[generation.duration_seconds]
                    or len(audio.readframes(audio.getnframes())) != 2 * audio.getnframes()):
                raise ValueError('Invalid generated WAV format or duration')
        clips.append(GeneratedClip(filename=path.name, sha256=file_sha256(path), frames=NATIVE_FRAMES[generation.duration_seconds]))
    return clips


def download_zip(generation: Generation, workspace: Path) -> bytes:
    if generation.state != 'succeeded' or not generation.artifacts_ready:
        raise FitRequestError(409, 'Generation is not complete. No download is available.')
    try:
        clips = validate_clips(generation, workspace)
        if clips != generation.clips or json.loads((workspace / 'manifest.json').read_text()) != [clip.model_dump() for clip in clips]:
            raise ValueError('Generated artifacts changed')
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for clip in clips:
                archive.write(str(workspace / clip.filename), clip.filename)
        return stream.getvalue()
    except (OSError, ValueError, KeyError, wave.Error, EOFError) as error:
        raise FitRequestError(409, 'Generated samples are missing or changed. Create a new generation.') from error
