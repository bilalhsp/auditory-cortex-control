"""Independent web API; scientific models are never loaded at startup."""

import secrets
from datetime import datetime, timezone
from typing import Callable, Optional
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from backend.batches import submit_batch, build_batch_zip, batch_manifest
from backend.gpu import local_gpu_count

from backend.auth import password_matches
from backend.config import Settings
from backend.models import Experiment, ExperimentConfiguration, RecordingCatalogResult, RecordingSelection, UploadResult, Fit, FitSubmission, Generation, GenerationSubmission, Batch, BatchSubmission, GENERATION_OBJECTIVES, GENERATION_DURATIONS, GENERATION_CLIP_COUNTS
from backend.recordings import CatalogUnavailable, RecordingCatalog
from backend.store import ExperimentStore
from backend.uploads import UploadError, receive_upload
from backend.fits import FitRequestError, submit_fit
from backend.generations import submit_generation, download_zip


def create_app(
    settings: Optional[Settings] = None,
    gpu_count: Callable[[], int] = local_gpu_count,
) -> FastAPI:
    settings = settings or Settings.from_environment()
    if not settings.operator or not settings.password_hash:
        raise RuntimeError("Operator authentication must be configured")
    if settings.max_upload_bytes < 1:
        raise RuntimeError("Upload size limit must be positive")
    # Validate the configured hash once, failing closed on invalid configuration.
    password_matches("", settings.password_hash)
    store = ExperimentStore(settings.database_path)
    catalog = RecordingCatalog(settings)
    app = FastAPI(title="Auditory Cortex Control", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type", "X-Recording-Filename"],
    )
    security = HTTPBasic()

    def operator(credentials: HTTPBasicCredentials = Depends(security)) -> str:
        valid_password = password_matches(credentials.password, settings.password_hash)
        valid_name = secrets.compare_digest(credentials.username.encode(), settings.operator.encode())
        if not (valid_password and valid_name):
            raise HTTPException(401, "Invalid operator credentials", headers={"WWW-Authenticate": "Basic"})
        return credentials.username

    def capacity() -> dict:
        try:
            count = gpu_count()
            if count < 0:
                raise RuntimeError("Invalid GPU count")
            status = "ready" if count else "no_gpu"
        except (ImportError, RuntimeError, OSError):
            count, status = 0, "unavailable"
        return {
            "visible_count": count, "available_count": max(0, count - store.active_compute_count()),
            "observed_at": datetime.now(timezone.utc).isoformat(), "status": status,
        }

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/v1/capabilities", dependencies=[Depends(operator)])
    def capabilities() -> dict:
        return {
            "settings": {
                "stimulus_families": ["TIMIT", "mVocs"], "datasets": ["ucdavisBAK"],
                "models": ["deepspeech2"], "layers": [2],
                "bin_width_ms": 50, "receptive_field_ms": 200,
            },
            "gpu": capacity(),
            "upload": {"max_bytes": settings.max_upload_bytes, "formats": [".mat"]},
            "fit": {"supported_sources": ["existing", "upload"], "default_seed": 42},
            "batch": {"supported": True},
            "generation": {"objectives": GENERATION_OBJECTIVES, "durations": GENERATION_DURATIONS, "clip_counts": GENERATION_CLIP_COUNTS},
        }

    @app.post("/api/v1/experiments", response_model=Experiment, status_code=201,
              dependencies=[Depends(operator)])
    def create_experiment(configuration: ExperimentConfiguration) -> Experiment:
        current = capacity()
        count = current["available_count"]
        if configuration.gpu_count > count:
            raise HTTPException(409, {
                "message": "GPU capacity changed. Refresh capacity and choose again.",
                "available_count": count,
            })
        if count > 0 and configuration.gpu_count == 0:
            raise HTTPException(422, "Choose at least one available GPU")
        return store.create(configuration)

    @app.get("/api/v1/experiments/{identity}", response_model=Experiment,
             dependencies=[Depends(operator)])
    def read_experiment(identity: UUID) -> Experiment:
        experiment = store.get(str(identity))
        if experiment is None:
            raise HTTPException(404, "Experiment not found")
        return experiment

    def available_recordings(experiment: Experiment) -> RecordingCatalogResult:
        try:
            return catalog.list(experiment.configuration.dataset_name)
        except CatalogUnavailable as error:
            raise HTTPException(503, str(error)) from error

    @app.get("/api/v1/experiments/{identity}/recordings", response_model=RecordingCatalogResult,
             dependencies=[Depends(operator)])
    def recordings(identity: UUID) -> RecordingCatalogResult:
        experiment = read_experiment(identity)
        result = available_recordings(experiment)
        result.recordings = [recording for recording in result.recordings
                             if experiment.configuration.stimulus_family in recording.stimulus_families]
        return result

    @app.post("/api/v1/experiments/{identity}/recording-selection", response_model=Experiment,
              dependencies=[Depends(operator)])
    def select_recording(identity: UUID, selection: RecordingSelection) -> Experiment:
        experiment = read_experiment(identity)
        uploaded = store.get_upload(str(selection.recording_id), str(identity))
        if uploaded is not None:
            recording = uploaded.recording if uploaded.path.is_file() else None
        else:
            result = available_recordings(experiment)
            recording = next((item for item in result.recordings if item.id == str(selection.recording_id)), None)
        if recording is None:
            raise HTTPException(404, "Recording is missing or no longer available. Refresh the list.")
        if experiment.configuration.stimulus_family not in recording.stimulus_families:
            raise HTTPException(409, "Recording is incompatible with this experiment's stimulus family.")
        selected = store.select_recording(str(identity), recording)
        if selected is None:
            raise HTTPException(404, "Experiment not found")
        return selected

    @app.post("/api/v1/experiments/{identity}/recording-upload", response_model=UploadResult,
              status_code=201, dependencies=[Depends(operator)])
    async def upload_recording(identity: UUID, request: Request,
                               filename: str = Header(..., alias="X-Recording-Filename")) -> UploadResult:
        experiment = read_experiment(identity)
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/octet-stream":
            raise HTTPException(415, {"validation_status": "invalid", "message": "Send MAT recording bytes as application/octet-stream."})
        try:
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    byte_count = int(length)
                except ValueError as error:
                    raise UploadError(400, "Invalid upload length.") from error
                if byte_count < 0:
                    raise UploadError(400, "Invalid upload length.")
                if byte_count > settings.max_upload_bytes:
                    raise UploadError(413, "Recording exceeds the configured upload size limit.")
            return await receive_upload(request.stream(), filename, experiment, settings, store)
        except UploadError as error:
            raise HTTPException(error.status, {
                "validation_status": "invalid" if error.status < 500 else "error", "message": str(error),
            }) from error

    @app.post("/api/v1/experiments/{identity}/fits", response_model=Fit, status_code=202,
              dependencies=[Depends(operator)])
    def fit_recording(identity: UUID, submission: FitSubmission) -> Fit:
        experiment = read_experiment(identity)
        try:
            return submit_fit(settings, store, catalog, experiment, submission, capacity()["available_count"])
        except FitRequestError as error:
            raise HTTPException(error.status, str(error)) from error
        except CatalogUnavailable as error:
            raise HTTPException(503, str(error)) from error

    @app.get("/api/v1/fits/{identity}", response_model=Fit, dependencies=[Depends(operator)])
    def read_fit(identity: UUID) -> Fit:
        fit = store.get_fit(str(identity))
        if fit is None:
            raise HTTPException(404, "Fit not found")
        return fit

    @app.get("/api/v1/jobs/{identity}", dependencies=[Depends(operator)])
    def read_job(identity: UUID):
        fit = store.get_fit(str(identity), by_job=True)
        if fit is not None:
            return fit
        record = store.generation_record(str(identity), by_job=True)
        if record is not None: return record[0]
        batch = store.batch_record(str(identity), by_job=True)
        if batch is None: raise HTTPException(404, 'Job not found')
        return batch[0]

    @app.post('/api/v1/fits/{identity}/generations', response_model=Generation, status_code=202,
              dependencies=[Depends(operator)])
    def generate(identity: UUID, submission: GenerationSubmission) -> Generation:
        try:
            return submit_generation(settings, store, str(identity), submission, capacity()['available_count'])
        except FitRequestError as error:
            raise HTTPException(error.status, str(error)) from error

    @app.get('/api/v1/generations/{identity}', response_model=Generation, dependencies=[Depends(operator)])
    def read_generation(identity: UUID) -> Generation:
        record = store.generation_record(str(identity))
        if record is None:
            raise HTTPException(404, 'Generation not found')
        return record[0]

    @app.get('/api/v1/generations/{identity}/download', dependencies=[Depends(operator)])
    def download_generation(identity: UUID) -> Response:
        record = store.generation_record(str(identity))
        if record is None:
            raise HTTPException(404, 'Generation not found')
        try:
            data = download_zip(*record)
        except FitRequestError as error:
            raise HTTPException(error.status, str(error)) from error
        return Response(data, media_type='application/zip', headers={
            'Content-Disposition': 'attachment; filename="generation-' + str(identity) + '.zip"',
            'Cache-Control': 'no-store',
        })

    @app.post('/api/v1/fits/{identity}/batches', response_model=Batch, status_code=202, dependencies=[Depends(operator)])
    def create_batch(identity: UUID, submission: BatchSubmission) -> Batch:
        try:
            return submit_batch(settings, store, str(identity), submission, capacity()['visible_count'])
        except FitRequestError as error:
            raise HTTPException(error.status, str(error)) from error

    @app.get('/api/v1/batches/{identity}', response_model=Batch, dependencies=[Depends(operator)])
    def read_batch(identity: UUID) -> Batch:
        record = store.batch_record(str(identity))
        if record is None: raise HTTPException(404, 'Batch not found')
        return record[0]

    @app.get('/api/v1/batches/{identity}/manifest', dependencies=[Depends(operator)])
    def read_batch_manifest(identity: UUID):
        return batch_manifest(read_batch(identity))

    @app.get('/api/v1/batches/{identity}/download', dependencies=[Depends(operator)])
    def download_batch(identity: UUID):
        record = store.batch_record(str(identity))
        if record is None: raise HTTPException(404, 'Batch not found')
        try:
            path = build_batch_zip(*record)
        except FitRequestError as error:
            raise HTTPException(error.status, str(error)) from error
        except OSError as error:
            raise HTTPException(503, 'Download storage is unavailable. Retry after storage is restored.') from error
        return FileResponse(path, media_type='application/zip', filename='batch-' + str(identity) + '.zip',
                            headers={'Cache-Control': 'no-store'}, background=BackgroundTask(path.unlink, missing_ok=True))

    return app
