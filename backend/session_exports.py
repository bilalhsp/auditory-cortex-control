"""Validated sample archives scoped to an application session."""

import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import pytz
from pytz.tzfile import build_tzinfo

from backend.fits import FitRequestError
from backend.generations import validated_download_clips


def session_basename(session):
    zone = session.timezone
    try:
        tz = pytz.timezone(zone)
    except pytz.UnknownTimeZoneError:
        try:
            with (Path('/usr/share/zoneinfo') / zone).open('rb') as source:
                tz = build_tzinfo(zone, source)
        except (OSError, ValueError):
            tz = pytz.timezone('America/Los_Angeles')
    return datetime.fromisoformat(session.created_at).astimezone(tz).strftime('session-%Y-%m-%d_%H-%M-%S')


def sample_archive(lifecycle, identity, job_id=None):
    detail = lifecycle.detail(identity)
    jobs = [job for job in detail['jobs'] if job['kind'] in ('generation', 'batch')
            and (job_id is None or job['job_id'] == job_id)]
    if job_id is not None and not jobs:
        raise KeyError('Sample run not found in this session')
    name = session_basename(detail['session'])
    if job_id:
        name += '-' + jobs[0]['kind'] + '-' + jobs[0]['id']

    handle, filename = tempfile.mkstemp(prefix='session-export-', suffix='.zip',
                                        dir=str(lifecycle.store.path.parent))
    os.close(handle)
    path = Path(filename)
    try:
        completed = 0
        unavailable = []
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for job in jobs:
                if not job['artifacts_ready']:
                    continue
                if job['kind'] == 'batch':
                    batch, workspace = lifecycle.store.require_batch(job['id'])
                    if batch.state not in ('succeeded', 'partial', 'interrupted', 'cancelled'):
                        continue
                    runs = [(target, workspace / target.id, '/' + target.id)
                            for target in batch.targets if target.state == 'succeeded']
                else:
                    record = lifecycle.store.generation_record(job['id'])
                    runs = [(*record, '')] if record else []
                for generation, directory, target_suffix in runs:
                    run_path = name + '/' + job['kind'] + '-' + job['id'] + target_suffix
                    try:
                        clips = validated_download_clips(generation, directory)
                    except FitRequestError:
                        unavailable.append(run_path.split('/', 1)[1])
                        continue
                    for clip in clips:
                        archive.write(directory / clip.filename, run_path + '/' + clip.filename)
                        completed += 1
            if completed and unavailable:
                archive.writestr(name + '/unavailable-results.json', json.dumps(unavailable, indent=2))
        if not completed:
            raise FitRequestError(409, 'No completed samples are available to download.')
        return path, name + '.zip'
    except BaseException:
        path.unlink(missing_ok=True)
        raise
