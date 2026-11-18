"""SQLite experiment persistence, independent of GPU and model availability."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

from backend.lifecycle import initialize, ensure_open, expose, save_job, cancellation_requested, ensure_command_available, normalize_session_timezone
from backend.models import Experiment, ExperimentConfiguration, Recording, Fit, Generation, GenerationSubmission, Batch


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
            connection.execute("BEGIN IMMEDIATE")
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

            columns = {row[1] for row in connection.execute('PRAGMA table_info(generations)')}
            if 'batch_id' not in columns:
                connection.execute('ALTER TABLE generations ADD COLUMN batch_id TEXT')
            connection.execute(
                'CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, job_id TEXT UNIQUE NOT NULL, '
                'fit_id TEXT NOT NULL, request_id TEXT NOT NULL, state TEXT NOT NULL, gpu_count INTEGER NOT NULL, '
                'devices TEXT NOT NULL, visibility TEXT NOT NULL, data TEXT NOT NULL, workspace TEXT NOT NULL, '
                'request TEXT NOT NULL, UNIQUE(fit_id, request_id))'
            )

            initialize(connection)

    @staticmethod
    def _reserved_devices(connection):
        used = {0} if any(connection.execute('SELECT 1 FROM ' + table + " WHERE state = 'running'" + (' AND batch_id IS NULL' if table == 'generations' else '')).fetchone() for table in ('fits', 'generations')) else set()
        for row in connection.execute("SELECT devices FROM batches WHERE state IN ('queued', 'running')"):
            used.update(json.loads(row[0]))
        return used

    def batch_for_request(self, fit_id: str, request_id: str):
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute('SELECT id, request FROM batches WHERE fit_id = ? AND request_id = ?', (fit_id, request_id)).fetchone()
        return (self.require_batch(row[0])[0], json.loads(row[1])) if row else None

    def batch_record(self, identity: str, by_job: bool = False):
        column = 'job_id' if by_job else 'id'
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute('SELECT id, data, workspace FROM batches WHERE ' + column + ' = ?', (identity,)).fetchone()
            if row is None: return None
            targets = [expose(connection, Generation.model_validate_json(target[0])) for target in connection.execute('SELECT data FROM generations WHERE batch_id = ? ORDER BY rowid', (row[0],))]
            batch = expose(connection, Batch(**json.loads(row[1]), targets=targets))
        return batch, Path(row[2])

    def require_batch(self, identity: str) -> Tuple[Batch, Path]:
        record = self.batch_record(identity)
        if record is None:
            raise ValueError('Batch metadata is unavailable')
        return record

    def enqueue_batch(self, batch: Batch, request: dict, workspace: Path, visible_count: int, visibility, retry_source=None) -> Batch:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute('BEGIN IMMEDIATE')
            prior = connection.execute('SELECT id, request FROM batches WHERE fit_id = ? AND request_id = ?', (batch.fit_id, request['request_id'])).fetchone()
            if prior:
                if json.loads(prior[1]) != request: raise ValueError('Request identity was already used for different batch settings.')
                # The caller reloads after the transaction closes.
                return batch.model_copy(update={'id': prior[0]})
            ensure_command_available(connection, request['request_id'], retry_source)
            ensure_open(connection, batch.experiment_id)
            free = [index for index in range(visible_count) if index not in self._reserved_devices(connection)]
            if len(free) < batch.gpu_count:
                raise ValueError('GPU capacity changed. Refresh capacity and choose again; the requested count was not reduced.')
            experiment = self._get(connection, batch.experiment_id)
            fit_row = connection.execute('SELECT data FROM fits WHERE id = ?', (batch.fit_id,)).fetchone()
            fit = Fit.model_validate_json(fit_row[0]) if fit_row else None
            if (fit is None or fit.state != 'succeeded' or not fit.artifacts_ready or experiment is None or experiment.recording is None
                    or (experiment.recording.id, experiment.recording_revision) != (fit.recording.id, fit.recording_revision)):
                raise ValueError('Fit context changed. Fit the current recording first.')
            connection.execute('INSERT INTO batches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (batch.id, batch.job_id, batch.fit_id, request['request_id'], batch.state, batch.gpu_count,
                 json.dumps(free[:batch.gpu_count]), json.dumps(visibility), batch.model_dump_json(exclude={'targets'}), str(workspace), json.dumps(request)))
            if retry_source:
                connection.execute('INSERT INTO job_retries VALUES (?,?)', (batch.job_id,retry_source))
            for target in batch.targets:
                connection.execute('INSERT INTO generations (id,job_id,fit_id,request_id,state,data,workspace,batch_id) VALUES (?,?,?,?,?,?,?,?)',
                    (target.id, target.job_id, target.fit_id, target.id, target.state, target.model_dump_json(), str(workspace / target.id), batch.id))
        return batch

    def update_batch(self, batch: Batch):
        with sqlite3.connect(str(self.path)) as connection:
            save_job(connection, 'batches', batch)

    def claim_batch(self):
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute('BEGIN IMMEDIATE')
            if (connection.execute("SELECT 1 FROM fits WHERE state = 'running'").fetchone()
                    or connection.execute("SELECT 1 FROM generations WHERE state = 'running' AND batch_id IS NULL").fetchone()
                    or connection.execute("SELECT 1 FROM batches WHERE state = 'running'").fetchone()): return None
            row = connection.execute("SELECT id, workspace, devices, visibility FROM batches WHERE state = 'queued' ORDER BY rowid LIMIT 1").fetchone()
            if row is None: return None
            connection.execute("UPDATE batches SET state = 'running' WHERE id = ?", (row[0],))
        batch = self.require_batch(row[0])[0]
        batch.state = 'running'
        batch.started_at = datetime.now(timezone.utc).isoformat()
        self.update_batch(batch)
        return batch, Path(row[1]), json.loads(row[2]), json.loads(row[3])

    def interrupt_running_batches(self):
        with sqlite3.connect(str(self.path)) as connection:
            rows = connection.execute("SELECT id FROM batches WHERE state = 'running'").fetchall()
        for row in rows:
            batch = self.require_batch(row[0])[0]
            for target in batch.targets:
                if target.state in ('queued', 'running', 'cancelling'):
                    target.state, target.stage = 'interrupted', 'interrupted'
                    target.error = 'Compute worker restarted. Submit a new batch to retry.'
                    target.finished_at = datetime.now(timezone.utc).isoformat()
                    self.update_generation(target)
            batch.state = 'interrupted'
            batch.artifacts_ready = any(target.state == 'succeeded' for target in batch.targets)
            batch.finished_at = datetime.now(timezone.utc).isoformat()
            batch.error = ('Worker restarted after target results were saved. Download the completed results.'
                           if all(target.state == 'succeeded' for target in batch.targets)
                           else 'Compute worker restarted. Submit a new batch to retry.')
            self.update_batch(batch)

    def fit_workspace(self, identity: str) -> Optional[Path]:
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute('SELECT workspace FROM fits WHERE id = ?', (identity,)).fetchone()
        return Path(row[0]) if row else None

    def generation_record(self, identity: str, by_job: bool = False):
        column = 'job_id' if by_job else 'id'
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute('SELECT data, workspace FROM generations WHERE ' + column + ' = ?', (identity,)).fetchone()
            result = (expose(connection, Generation.model_validate_json(row[0])), Path(row[1])) if row else None
        return result

    def generation_for_request(self, fit_id: str, request_id: str) -> Optional[Generation]:
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute('SELECT data FROM generations WHERE fit_id = ? AND request_id = ?', (fit_id, request_id)).fetchone()
            result = expose(connection, Generation.model_validate_json(row[0])) if row else None
        return result

    def enqueue_generation(self, generation: Generation, request_id: str, workspace: Path, retry_source=None) -> Generation:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute('BEGIN IMMEDIATE')
            previous = connection.execute('SELECT data FROM generations WHERE fit_id = ? AND request_id = ?', (generation.fit_id, request_id)).fetchone()
            if previous:
                prior = Generation.model_validate_json(previous[0])
                if any(getattr(prior, name) != getattr(generation, name) for name in GenerationSubmission.context_fields()):
                    raise ValueError('Request identity was already used for different generation settings.')
                return prior
            ensure_command_available(connection, request_id, retry_source)
            ensure_open(connection, generation.experiment_id)
            experiment = self._get(connection, generation.experiment_id)
            fit_row = connection.execute('SELECT data FROM fits WHERE id = ?', (generation.fit_id,)).fetchone()
            fit = Fit.model_validate_json(fit_row[0]) if fit_row else None
            if (fit is None or fit.state != 'succeeded' or not fit.artifacts_ready or experiment is None
                    or experiment.recording is None or (experiment.recording.id, experiment.recording_revision) != (fit.recording.id, fit.recording_revision)):
                raise ValueError('Fit context changed. Select and fit the current recording again.')
            connection.execute('INSERT INTO generations (id,job_id,fit_id,request_id,state,data,workspace) VALUES (?, ?, ?, ?, ?, ?, ?)',
                               (generation.id, generation.job_id, generation.fit_id, request_id, generation.state, generation.model_dump_json(), str(workspace)))
            if retry_source:
                connection.execute('INSERT INTO job_retries VALUES (?,?)', (generation.job_id,retry_source))
        return generation

    def update_generation(self, generation: Generation) -> bool:
        with sqlite3.connect(str(self.path)) as connection:
            return save_job(connection, 'generations', generation)

    def active_compute_count(self) -> int:
        with sqlite3.connect(str(self.path)) as connection:
            return len(self._reserved_devices(connection))

    def claim_generation(self):
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute('BEGIN IMMEDIATE')
            if self._reserved_devices(connection):
                return None
            row = connection.execute("SELECT data, workspace FROM generations WHERE state = 'queued' AND batch_id IS NULL ORDER BY rowid LIMIT 1").fetchone()
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
            for row in connection.execute("SELECT data FROM generations WHERE state = 'running' AND batch_id IS NULL").fetchall():
                generation = Generation.model_validate_json(row[0])
                generation.state, generation.stage = 'interrupted', 'interrupted'
                generation.error = 'Compute worker restarted. Submit a new generation to retry.'
                generation.finished_at = datetime.now(timezone.utc).isoformat()
                save_job(connection, 'generations', generation)

    def create(self, configuration: ExperimentConfiguration, session_id: Optional[str] = None,
               session_timezone: Optional[str] = None) -> Experiment:
        experiment = Experiment(
            id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
            configuration=configuration, session_id=session_id or str(uuid4()),
        )
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute('BEGIN IMMEDIATE')
            if session_id is not None:
                row = connection.execute('SELECT status FROM application_sessions WHERE id=?', (session_id,)).fetchone()
                if row is None or row[0] != 'open': raise ValueError('Application session is closing or closed.')
            else:
                connection.execute(
                    "INSERT INTO application_sessions (id,created_at,status,timezone) VALUES (?,?,'open',?)",
                    (experiment.session_id, experiment.created_at, normalize_session_timezone(session_timezone)),
                )
            connection.execute('INSERT INTO session_experiments VALUES (?,?)', (experiment.id, experiment.session_id))
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
                          recording_revision=row[4] or 0, session_id=connection.execute('SELECT session_id FROM session_experiments WHERE experiment_id=?', (identity,)).fetchone()[0])

    def select_recording(self, identity: str, recording: Recording) -> Optional[Experiment]:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._select(connection, identity, recording)

    def _select(self, connection: sqlite3.Connection, identity: str, recording: Recording) -> Optional[Experiment]:
        experiment = self._get(connection, identity)
        if experiment is None:
            return None
        ensure_open(connection, identity)
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
            result = expose(connection, Fit.model_validate_json(row[0])) if row else None
        return result

    def fit_for_request(self, experiment_id: str, request_id: str) -> Optional[Fit]:
        with sqlite3.connect(str(self.path)) as connection:
            row = connection.execute("SELECT data FROM fits WHERE experiment_id = ? AND request_id = ?",
                                     (experiment_id, request_id)).fetchone()
            result = expose(connection, Fit.model_validate_json(row[0])) if row else None
        return result

    def enqueue_fit(self, fit: Fit, request_id: str, workspace: Path, retry_source=None) -> Fit:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT data FROM fits WHERE experiment_id = ? AND request_id = ?",
                                          (fit.experiment_id, request_id)).fetchone()
            if previous:
                prior = Fit.model_validate_json(previous[0])
                if (prior.recording.id, prior.recording_revision, prior.seed) != (fit.recording.id, fit.recording_revision, fit.seed):
                    raise ValueError("Request identity was already used for a different fit")
                return prior
            ensure_command_available(connection, request_id, retry_source)
            ensure_open(connection, fit.experiment_id)
            experiment = self._get(connection, fit.experiment_id)
            if (experiment is None or experiment.recording is None or
                    (experiment.recording.id, experiment.recording_revision) != (fit.recording.id, fit.recording_revision)):
                raise ValueError("Recording selection changed. Reload and submit again.")
            connection.execute("INSERT INTO fits VALUES (?, ?, ?, ?, ?, ?, ?)", (
                fit.id, fit.job_id, fit.experiment_id, request_id, fit.state, fit.model_dump_json(), str(workspace),
            ))
            if retry_source:
                connection.execute('INSERT INTO job_retries VALUES (?,?)', (fit.job_id,retry_source))
            return fit

    @staticmethod
    def _save_fit(connection: sqlite3.Connection, fit: Fit) -> None:
        save_job(connection, 'fits', fit)

    def running_fit_count(self) -> int:
        with sqlite3.connect(str(self.path)) as connection:
            return connection.execute("SELECT COUNT(*) FROM fits WHERE state = 'running'").fetchone()[0]

    def claim_fit(self) -> Optional[Tuple[Fit, Path]]:
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._reserved_devices(connection):
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
