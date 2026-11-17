"""SQLite experiment persistence, independent of GPU and model availability."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from backend.models import Experiment, ExperimentConfiguration


class ExperimentStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(path)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS experiments "
                "(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, configuration TEXT NOT NULL)"
            )

    def create(self, configuration: ExperimentConfiguration) -> Experiment:
        experiment = Experiment(
            id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
            configuration=configuration,
        )
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("INSERT INTO experiments VALUES (?, ?, ?)", (
                experiment.id, experiment.created_at, configuration.model_dump_json(),
            ))
        return experiment

    def get(self, identity: str) -> Optional[Experiment]:
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute(
                "SELECT id, created_at, configuration FROM experiments WHERE id = ?", (identity,)
            ).fetchone()
        if row is None:
            return None
        return Experiment(id=row[0], created_at=row[1], configuration=json.loads(row[2]))
