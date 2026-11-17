"""Independent web API; scientific models are never loaded at startup."""

import secrets
from datetime import datetime, timezone
from typing import Callable, Optional
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from backend.auth import password_matches
from backend.config import Settings
from backend.models import Experiment, ExperimentConfiguration
from backend.store import ExperimentStore


def local_gpu_count() -> int:
    # Lazy import respects CUDA_VISIBLE_DEVICES without loading any model.
    import torch
    return torch.cuda.device_count()


def create_app(
    settings: Optional[Settings] = None,
    gpu_count: Callable[[], int] = local_gpu_count,
) -> FastAPI:
    settings = settings or Settings.from_environment()
    if not settings.operator or not settings.password_hash:
        raise RuntimeError("Operator authentication must be configured")
    # Validate the configured hash once, failing closed on invalid configuration.
    password_matches("", settings.password_hash)
    store = ExperimentStore(settings.database_path)
    app = FastAPI(title="Auditory Cortex Control", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"],
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
            "visible_count": count, "available_count": count,
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

    return app
