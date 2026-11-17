"""Application-session history and durable, transactionally ordered job controls."""
import json
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid4, uuid5
from backend.models import ApplicationSession
if TYPE_CHECKING:
    from backend.store import ExperimentStore

TERMINAL = {'succeeded', 'partial', 'failed', 'interrupted', 'cancelled'}


def initialize(connection):
    connection.execute('CREATE TABLE IF NOT EXISTS application_sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, closed_at TEXT)')
    connection.execute('CREATE TABLE IF NOT EXISTS session_experiments (experiment_id TEXT PRIMARY KEY, session_id TEXT NOT NULL)')
    connection.execute('CREATE TABLE IF NOT EXISTS job_controls (job_id TEXT PRIMARY KEY, requested_at TEXT NOT NULL)')
    connection.execute('CREATE TABLE IF NOT EXISTS job_retries (job_id TEXT PRIMARY KEY, previous_job_id TEXT NOT NULL)')
    for identity, created in connection.execute('SELECT e.id,e.created_at FROM experiments e LEFT JOIN session_experiments s ON s.experiment_id=e.id WHERE s.experiment_id IS NULL').fetchall():
        session = str(uuid5(NAMESPACE_URL, 'auditory-cortex:application-session:' + identity))
        connection.execute("INSERT OR IGNORE INTO application_sessions VALUES (?,?,'open',NULL)", (session, created))
        connection.execute('INSERT INTO session_experiments VALUES (?,?)', (identity, session))


def ensure_open(connection, experiment_id):
    row = connection.execute('SELECT a.status FROM application_sessions a JOIN session_experiments s ON s.session_id=a.id WHERE s.experiment_id=?', (experiment_id,)).fetchone()
    if row is None or row[0] != 'open':
        raise ValueError('Application session is closing or closed. Start a new session.')


def cancellation_requested(connection, job_id):
    return connection.execute('SELECT 1 FROM job_controls WHERE job_id=?', (job_id,)).fetchone() is not None


def expose(connection, job):
    if job.state == 'running' and cancellation_requested(connection, job.job_id):
        return job.model_copy(update={'state': 'cancelling'})
    return job


def save_job(connection, table, job):
    if not connection.in_transaction: connection.execute('BEGIN IMMEDIATE')
    prior = connection.execute('SELECT state FROM ' + table + ' WHERE id=?', (job.id,)).fetchone()
    if prior is None or prior[0] in TERMINAL: return False
    if cancellation_requested(connection, job.job_id) and job.state in TERMINAL:
        job.state = 'cancelled'
        job.error = 'Stopped by the operator.'
        if table == 'batches': job.artifacts_ready = any(target.state == 'succeeded' for target in job.targets)
        else:
            job.stage = 'cancelled'
            job.artifacts_ready = False
            if table == 'fits': job.correlations, job.best_channel = [], None
            else: job.clips = []
    data = job.model_dump_json(exclude={'targets'}) if table == 'batches' else job.model_dump_json()
    connection.execute('UPDATE ' + table + ' SET state=?,data=? WHERE id=?', (job.state, data, job.id))
    return True


def _cancel(connection, table, identity):
    row = connection.execute('SELECT job_id,state,data FROM ' + table + ' WHERE id=?', (identity,)).fetchone()
    if row is None: raise KeyError('Job not found')
    if row[1] in TERMINAL: return
    connection.execute('INSERT OR IGNORE INTO job_controls VALUES (?,?)', (row[0], datetime.now(timezone.utc).isoformat()))
    if row[1] == 'queued':
        data = json.loads(row[2]);data.update(state='cancelled', finished_at=datetime.now(timezone.utc).isoformat(), error='Stopped by the operator.')
        if 'stage' in data: data['stage'] = 'cancelled'
        connection.execute('UPDATE ' + table + ' SET state=?,data=? WHERE id=?', ('cancelled', json.dumps(data), identity))
    if table == 'batches':
        for target in connection.execute('SELECT id FROM generations WHERE batch_id=?', (identity,)).fetchall():
            _cancel(connection, 'generations', target[0])


class Lifecycle:
    def __init__(self, store: 'ExperimentStore'):
        self.store = store

    def cancelled(self, job_id):
        with sqlite3.connect(str(self.store.path)) as connection:
            return cancellation_requested(connection, job_id)

    def sessions(self):
        with sqlite3.connect(str(self.store.path)) as connection:
            ids = [row[0] for row in connection.execute('SELECT id FROM application_sessions ORDER BY created_at DESC')]
        return [self.session(identity) for identity in ids]

    def session(self, identity):
        with sqlite3.connect(str(self.store.path)) as connection:
            row = connection.execute('SELECT id,created_at,status,closed_at FROM application_sessions WHERE id=?', (identity,)).fetchone()
            if row is None: return None
            if row[2] == 'closing':
                connection.execute('BEGIN IMMEDIATE')
                running = any(connection.execute('SELECT 1 FROM ' + table + ' j JOIN session_experiments s ON s.experiment_id=' + ('j.experiment_id' if table == 'fits' else '(SELECT experiment_id FROM fits WHERE id=j.fit_id)') + " WHERE s.session_id=? AND j.state='running'", (identity,)).fetchone() for table in ('fits','generations','batches'))
                if not running:
                    closed = datetime.now(timezone.utc).isoformat()
                    connection.execute("UPDATE application_sessions SET status='closed',closed_at=? WHERE id=? AND status='closing'", (closed,identity))
                    row = connection.execute('SELECT id,created_at,status,closed_at FROM application_sessions WHERE id=?', (identity,)).fetchone()
        return ApplicationSession(id=row[0], created_at=row[1], status=row[2], closed_at=row[3])

    def work(self, experiment_id):
        with sqlite3.connect(str(self.store.path)) as connection:
            fits = [row[0] for row in connection.execute('SELECT id FROM fits WHERE experiment_id=? ORDER BY rowid DESC', (experiment_id,))]
            generations = [row[0] for row in connection.execute('SELECT g.id FROM generations g JOIN fits f ON f.id=g.fit_id WHERE f.experiment_id=? AND g.batch_id IS NULL ORDER BY g.rowid DESC', (experiment_id,))]
            batches = [row[0] for row in connection.execute('SELECT b.id FROM batches b JOIN fits f ON f.id=b.fit_id WHERE f.experiment_id=? ORDER BY b.rowid DESC', (experiment_id,))]
        generation_records = [self.store.generation_record(identity) for identity in generations]
        return {'fits': [job for identity in fits if (job := self.store.get_fit(identity)) is not None],
                'generations': [record[0] for record in generation_records if record is not None],
                'batches': [self.store.require_batch(identity)[0] for identity in batches]}

    def detail(self, identity):
        session = self.session(identity)
        if session is None: raise KeyError('Application session not found')
        with sqlite3.connect(str(self.store.path)) as connection:
            ids = [row[0] for row in connection.execute('SELECT experiment_id FROM session_experiments WHERE session_id=? ORDER BY rowid DESC', (identity,))]
            retries = dict(connection.execute('SELECT job_id,previous_job_id FROM job_retries'))
        jobs = []
        for experiment in ids:
            for kind, operations in self.work(experiment).items():
                for operation in operations:
                    jobs.append({'kind': {'fits':'fit','generations':'generation','batches':'batch'}[kind], 'id':operation.id,
                                 'job_id':operation.job_id, 'state':operation.state, 'created_at':operation.created_at,
                                 'artifacts_ready':operation.artifacts_ready, 'retry_of_job_id':retries.get(operation.job_id)})
        jobs.sort(key=lambda job: job['created_at'], reverse=True)
        return {'session':session, 'experiments':[self.store.get(identity) for identity in ids], 'jobs':jobs}

    def cancel_job(self, job_id):
        with sqlite3.connect(str(self.store.path)) as connection:
            connection.execute('BEGIN IMMEDIATE')
            for table in ('fits','generations','batches'):
                row = connection.execute('SELECT id FROM ' + table + ' WHERE job_id=?', (job_id,)).fetchone()
                if row:
                    _cancel(connection, table, row[0]);return
        raise KeyError('Job not found')

    def close(self, identity):
        with sqlite3.connect(str(self.store.path)) as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT status FROM application_sessions WHERE id=?', (identity,)).fetchone()
            if row is None: raise KeyError('Application session not found')
            if row[0] == 'open': connection.execute("UPDATE application_sessions SET status='closing' WHERE id=?", (identity,))
            for table in ('fits','generations','batches'):
                experiment = 'j.experiment_id' if table == 'fits' else '(SELECT experiment_id FROM fits WHERE id=j.fit_id)'
                ids = connection.execute('SELECT j.id FROM ' + table + ' j JOIN session_experiments s ON s.experiment_id=' + experiment + ' WHERE s.session_id=?', (identity,)).fetchall()
                for job in ids: _cancel(connection, table, job[0])
        return self.session(identity)
