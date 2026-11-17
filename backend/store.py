"""SQLite experiment persistence, independent of GPU and model availability."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

from backend.models import Experiment, ExperimentConfiguration, Recording, Fit, Generation, GenerationSubmission


@dataclass(frozen=True)
class StoredUpload:
    recording: Recording
    path: Path
    size_bytes: int
    sha256: str


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
            connection.execute(
                "CREATE TABLE IF NOT EXISTS uploaded_recordings "
                "(id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id), "
                "recording TEXT NOT NULL, path TEXT NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS fits (id TEXT PRIMARY KEY, job_id TEXT UNIQUE NOT NULL, "
                "experiment_id TEXT NOT NULL, request_id TEXT NOT NULL, state TEXT NOT NULL, "
                "data TEXT NOT NULL, workspace TEXT NOT NULL, UNIQUE(experiment_id, request_id))"
            )

            connection.execute(
                'CREATE TABLE IF NOT EXISTS generations (id TEXT PRIMARY KEY, job_id TEXT UNIQUE NOT NULL, '
                'fit_id TEXT NOT NULL, request_id TEXT NOT NULL, state TEXT NOT NULL, '
                'data TEXT NOT NULL, workspace TEXT NOT NULL, UNIQUE(fit_id, request_id))'
            )

    def fit_workspace(self, identity: str) -> Optional[Path]:
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute('SELECT workspace FROM fits WHERE id = ?', (identity,)).fetchone()
        return Path(row[0]) if row else None

    def generation_record(self, identity: str, by_job: bool = False):
        column = 'job_id' if by_job else 'id'
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute('SELECT data, workspace FROM generations WHERE ' + column + ' = ?', (identity,)).fetchone()
        return (Generation.model_validate_json(row[0]), Path(row[1])) if row else None

    def generation_for_request(self, fit_id: str, request_id: str) -> Optional[Generation]:
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute('SELECT data FROM generations WHERE fit_id = ? AND request_id = ?', (fit_id, request_id)).fetchone()
        return Generation.model_validate_json(row[0]) if row else None

    def enqueue_generation(self, generation: Generation, request_id: str, workspace: Path) -> Generation:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute('BEGIN IMMEDIATE')
            previous = connection.execute('SELECT data FROM generations WHERE fit_id = ? AND request_id = ?', (generation.fit_id, request_id)).fetchone()
            if previous:
                prior = Generation.model_validate_json(previous[0])
                if any(getattr(prior, name) != getattr(generation, name) for name in GenerationSubmission.context_fields()):
                    raise ValueError('Request identity was already used for different generation settings.')
                return prior
            experiment = self._get(connection, generation.experiment_id)
            fit_row = connection.execute('SELECT data FROM fits WHERE id = ?', (generation.fit_id,)).fetchone()
            fit = Fit.model_validate_json(fit_row[0]) if fit_row else None
            if (fit is None or fit.state != 'succeeded' or not fit.artifacts_ready or experiment is None
                    or experiment.recording is None or (experiment.recording.id, experiment.recording_revision) != (fit.recording.id, fit.recording_revision)):
                raise ValueError('Fit context changed. Select and fit the current recording again.')
            connection.execute('INSERT INTO generations VALUES (?, ?, ?, ?, ?, ?, ?)',
                               (generation.id, generation.job_id, generation.fit_id, request_id, generation.state, generation.model_dump_json(), str(workspace)))
        return generation

    def update_generation(self, generation: Generation) -> None:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute('UPDATE generations SET state = ?, data = ? WHERE id = ?',
                               (generation.state, generation.model_dump_json(), generation.id))

    def active_compute_count(self) -> int:
        with sqlite3.connect(str(self.path)) as connection:
            return sum(connection.execute('SELECT COUNT(*) FROM ' + table + " WHERE state = 'running'").fetchone()[0] for table in ('fits', 'generations'))

    def claim_generation(self):
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute('BEGIN IMMEDIATE')
            if any(connection.execute('SELECT 1 FROM ' + table + " WHERE state = 'running'").fetchone() for table in ('fits', 'generations')):
                return None
            row = connection.execute("SELECT data, workspace FROM generations WHERE state = 'queued' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                return None
            generation = Generation.model_validate_json(row[0])
            generation.state, generation.stage = 'running', 'loading'
            generation.started_at = datetime.now(timezone.utc).isoformat()
            connection.execute('UPDATE generations SET state = ?, data = ? WHERE id = ?', ('running', generation.model_dump_json(), generation.id))
            return generation, Path(row[1])

    def interrupt_running_generations(self) -> None:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute('BEGIN IMMEDIATE')
            for row in connection.execute("SELECT data FROM generations WHERE state = 'running'").fetchall():
                generation = Generation.model_validate_json(row[0])
                generation.state, generation.stage = 'interrupted', 'interrupted'
                generation.error = 'Compute worker restarted. Submit a new generation to retry.'
                generation.finished_at = datetime.now(timezone.utc).isoformat()
                connection.execute('UPDATE generations SET state = ?, data = ? WHERE id = ?', (generation.state, generation.model_dump_json(), generation.id))

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
            return self._select(connection, identity, recording)

    def _select(self, connection: sqlite3.Connection, identity: str, recording: Recording) -> Optional[Experiment]:
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

    def register_upload(self, identity: str, upload: StoredUpload) -> Optional[Experiment]:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._get(connection, identity) is None:
                return None
            connection.execute("INSERT INTO uploaded_recordings VALUES (?, ?, ?, ?, ?, ?)", (
                upload.recording.id, identity, upload.recording.model_dump_json(), str(upload.path),
                upload.size_bytes, upload.sha256,
            ))
            return self._select(connection, identity, upload.recording)

    def get_upload(self, identity: str, experiment_id: str) -> Optional[StoredUpload]:
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute(
                "SELECT recording, path, size_bytes, sha256 FROM uploaded_recordings "
                "WHERE id = ? AND experiment_id = ?", (identity, experiment_id),
            ).fetchone()
        if row is None:
            return None
        return StoredUpload(Recording.model_validate_json(row[0]), Path(row[1]), row[2], row[3])

    def get_fit(self, identity: str, by_job: bool = False) -> Optional[Fit]:
        column = "job_id" if by_job else "id"
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute("SELECT data FROM fits WHERE " + column + " = ?", (identity,)).fetchone()
        return Fit.model_validate_json(row[0]) if row else None

    def fit_for_request(self, experiment_id: str, request_id: str) -> Optional[Fit]:
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute("SELECT data FROM fits WHERE experiment_id = ? AND request_id = ?",
                                     (experiment_id, request_id)).fetchone()
        return Fit.model_validate_json(row[0]) if row else None

    def enqueue_fit(self, fit: Fit, request_id: str, workspace: Path) -> Fit:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT data FROM fits WHERE experiment_id = ? AND request_id = ?",
                                          (fit.experiment_id, request_id)).fetchone()
            if previous:
                prior = Fit.model_validate_json(previous[0])
                if (prior.recording.id, prior.recording_revision, prior.seed) != (fit.recording.id, fit.recording_revision, fit.seed):
                    raise ValueError("Request identity was already used for a different fit")
                return prior
            experiment = self._get(connection, fit.experiment_id)
            if (experiment is None or experiment.recording is None or
                    (experiment.recording.id, experiment.recording_revision) != (fit.recording.id, fit.recording_revision)):
                raise ValueError("Recording selection changed. Reload and submit again.")
            connection.execute("INSERT INTO fits VALUES (?, ?, ?, ?, ?, ?, ?)", (
                fit.id, fit.job_id, fit.experiment_id, request_id, fit.state, fit.model_dump_json(), str(workspace),
            ))
            return fit

    @staticmethod
    def _save_fit(connection: sqlite3.Connection, fit: Fit) -> None:
        connection.execute("UPDATE fits SET state = ?, data = ? WHERE id = ?",
                           (fit.state, fit.model_dump_json(), fit.id))

    def running_fit_count(self) -> int:
        with sqlite3.connect(str(self.path)) as connection:
            return connection.execute("SELECT COUNT(*) FROM fits WHERE state = 'running'").fetchone()[0]

    def claim_fit(self) -> Optional[Tuple[Fit, Path]]:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if any(connection.execute("SELECT 1 FROM " + table + " WHERE state = 'running'").fetchone() for table in ("fits", "generations")):
                return None
            row = connection.execute("SELECT data, workspace FROM fits WHERE state = 'queued' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                return None
            fit = Fit.model_validate_json(row[0])
            fit.state, fit.stage = "running", "loading"
            fit.started_at = datetime.now(timezone.utc).isoformat()
            self._save_fit(connection, fit)
            return fit, Path(row[1])

    def update_fit(self, fit: Fit) -> None:
        with sqlite3.connect(str(self.path)) as connection:
            self._save_fit(connection, fit)

    def interrupt_running_fits(self) -> None:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for row in connection.execute("SELECT data FROM fits WHERE state = 'running'").fetchall():
                fit = Fit.model_validate_json(row[0])
                fit.state, fit.stage = "interrupted", "interrupted"
                fit.error = "Fit worker restarted. Submit a new fit to retry."
                fit.finished_at = datetime.now(timezone.utc).isoformat()
                self._save_fit(connection, fit)
