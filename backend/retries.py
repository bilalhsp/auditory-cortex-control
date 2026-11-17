"""Explicit retries create isolated attempts, preserving command idempotency."""
import json
import sqlite3
from backend.fits import FitRequestError, submit_fit
from backend.generations import submit_generation
from backend.batches import submit_batch
from backend.models import FitSubmission, GenerationSubmission, BatchSubmission


def retry_job(settings, store, catalog, job_id, request_id, capacity):
    fit = store.get_fit(job_id, by_job=True)
    generation = store.generation_record(job_id, by_job=True)
    batch = store.batch_record(job_id, by_job=True)
    operation = fit or (generation[0] if generation else batch[0] if batch else None)
    if operation is None: raise FitRequestError(404, 'Job not found.')
    if operation.state not in ('failed','interrupted','cancelled','partial'):
        raise FitRequestError(409, 'Only failed, interrupted, partial, or stopped work can be retried.')
    if fit:
        experiment = store.get(fit.experiment_id)
        if experiment is None: raise FitRequestError(404, 'Experiment not found.')
        result = submit_fit(settings, store, catalog, experiment, FitSubmission(request_id=request_id,
            recording_id=fit.recording.id, recording_revision=fit.recording_revision, seed=fit.seed), capacity['available_count'])
    elif generation:
        source = generation[0]
        result = submit_generation(settings, store, source.fit_id, GenerationSubmission(request_id=request_id,
            channel_id=source.channel_id, objective=source.objective, duration_seconds=source.duration_seconds,
            clip_count=source.clip_count, seed=source.seed), capacity['available_count'])
    else:
        assert batch is not None
        source = batch[0]
        with sqlite3.connect(str(store.path)) as connection:
            command = json.loads(connection.execute('SELECT request FROM batches WHERE id=?', (source.id,)).fetchone()[0])
        command['request_id'] = str(request_id)
        command['targets'] = [{'channel_id': target.channel_id, 'objective': target.objective}
                              for target in source.targets if target.state != 'succeeded']
        result = submit_batch(settings, store, source.fit_id, BatchSubmission(**command), capacity['visible_count'])
    if result.job_id == job_id:
        raise FitRequestError(409, 'A retry requires a new command identity.')
    try:
        with sqlite3.connect(str(store.path)) as connection:
            connection.execute('BEGIN IMMEDIATE')
            previous = connection.execute('SELECT previous_job_id FROM job_retries WHERE job_id=?', (result.job_id,)).fetchone()
            if previous and previous[0] != job_id:
                raise FitRequestError(409, 'Command identity was already used for another retry.')
            connection.execute('INSERT OR IGNORE INTO job_retries VALUES (?,?)', (result.job_id, job_id))
    except sqlite3.Error as error:
        raise FitRequestError(503, 'Retry metadata is unavailable. Retry with the same command identity.') from error
    return result
