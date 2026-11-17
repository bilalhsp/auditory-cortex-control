"""SQLite experiment persistence, independent of GPU and model availability."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from backend.models import Experiment, ExperimentConfiguration, Recording


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
            connection.execute(
                "CREATE TABLE IF NOT EXISTS experiment_recordings "
                "(experiment_id TEXT PRIMARY KEY REFERENCES experiments(id), "
                "recording TEXT NOT NULL, revision INTEGER NOT NULL)"
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
            return self._get(connection, identity)

    @staticmethod
    def _get(connection: sqlite3.Connection, identity: str) -> Optional[Experiment]:
        row = connection.execute(
            "SELECT e.id, e.created_at, e.configuration, r.recording, r.revision "
            "FROM experiments e LEFT JOIN experiment_recordings r ON r.experiment_id = e.id "
            "WHERE e.id = ?", (identity,)
        ).fetchone()
        if row is None:
            return None
        return Experiment(id=row[0], created_at=row[1], configuration=json.loads(row[2]),
                          recording=json.loads(row[3]) if row[3] else None,
                          recording_revision=row[4] or 0)

    def select_recording(self, identity: str, recording: Recording) -> Optional[Experiment]:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            experiment = self._get(connection, identity)
            if experiment is None:
                return None
            if experiment.recording and experiment.recording.id == recording.id:
                return experiment
            revision = experiment.recording_revision + 1
            connection.execute(
                "INSERT INTO experiment_recordings VALUES (?, ?, ?) "
                "ON CONFLICT(experiment_id) DO UPDATE SET recording=excluded.recording, revision=excluded.revision",
                (identity, recording.model_dump_json(), revision),
            )
            return experiment.model_copy(update={"recording": recording, "recording_revision": revision})
