import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml


@dataclass(frozen=True)
class Settings:
    database_path: Path
    operator: str
    password_hash: str
    allowed_origins: Tuple[str, ...] = ("http://localhost:5173",)
    neural_data_dir: Optional[Path] = None
    recording_metadata_path: Optional[Path] = None
    upload_dir: Optional[Path] = None
    max_upload_bytes: int = 50 * 1024 * 1024

    @classmethod
    def from_environment(cls) -> "Settings":
        operator = os.environ.get("AUDITORY_OPERATOR", "")
        password_hash = os.environ.get("AUDITORY_PASSWORD_HASH", "")
        if not operator or not password_hash:
            raise RuntimeError("Set AUDITORY_OPERATOR and AUDITORY_PASSWORD_HASH before starting the API")
        database = os.environ.get("AUDITORY_DATABASE_PATH")
        if database is None:
            config_path = Path(__file__).resolve().parents[1] / "auditory_cortex/config.yml"
            config = yaml.safe_load(config_path.read_text())
            database = str(Path(config["cache_dir"]) / "web/sessions.sqlite3")
        origins = tuple(
            origin.strip() for origin in os.environ.get(
                "AUDITORY_ALLOWED_ORIGINS", "http://localhost:5173"
            ).split(",") if origin.strip()
        )
        if "*" in origins:
            raise RuntimeError("Use explicit frontend origins, not a wildcard")
        data_dir = os.environ.get("AUDITORY_NEURAL_DATA_DIR")
        metadata_path = os.environ.get("AUDITORY_RECORDINGS_METADATA")
        upload_dir = os.environ.get("AUDITORY_UPLOAD_DIR")
        maximum = int(os.environ.get("AUDITORY_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
        if maximum < 1:
            raise RuntimeError("AUDITORY_MAX_UPLOAD_BYTES must be positive")
        return cls(Path(database), operator, password_hash, origins,
                   Path(data_dir) if data_dir else None,
                   Path(metadata_path) if metadata_path else None,
                   Path(upload_dir) if upload_dir else None, maximum)
