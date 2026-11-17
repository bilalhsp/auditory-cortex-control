"""Bounded recording ingestion. Private artifact paths never enter API models."""

import hashlib
import os
import re
import sqlite3
from pathlib import Path
from typing import AsyncIterable, List, Literal
from urllib.parse import unquote
from uuid import uuid4

from starlette.concurrency import run_in_threadpool

from backend.config import Settings
from backend.models import Experiment, Recording, UploadResult
from backend.recordings import RecordingCatalog
from backend.store import ExperimentStore, StoredUpload


class UploadError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


def validate_recording(path: Path, stimulus_family: str) -> List[Literal['TIMIT', 'mVocs']]:
    import numpy as np
    from scipy.io import loadmat, whosmat
    from scipy.io.matlab import MatReadError

    try:
        names = [name for name, _, _ in whosmat(str(path))]
        channels = [name for name in names if re.fullmatch(r'WM_\d+|BAK\d+', name)]
        if 'TrialStimData' not in names or not channels:
            raise ValueError('Missing dataset trial or neural channel fields')
        data = loadmat(str(path), variable_names=['TrialStimData'] + channels,
                       squeeze_me=True, struct_as_record=False)
        has_s1t = False
        for trial in np.atleast_1d(data['TrialStimData']).flat:
            stimuli = np.atleast_1d(trial.StimulusName)
            kinds = np.atleast_1d(trial.StimulusType)
            onsets = np.atleast_1d(trial.StimulusTimeOn)
            if (stimuli.ndim != 1 or kinds.ndim != 1 or onsets.ndim != 1
                    or not len(stimuli) or len(stimuli) != len(kinds) or len(stimuli) != len(onsets)
                    or onsets.dtype.kind not in 'iuf' or not np.isfinite(onsets).all()):
                raise ValueError('Invalid trial fields')
            if not set(str(kind) for kind in kinds).issubset({'S1T', 'S2'}):
                raise ValueError('Unsupported full-path trial types')
            has_s1t = has_s1t or bool(np.any(kinds == 'S1T'))
        if not has_s1t:
            raise ValueError('Uploaded recordings require S1T trials for the legacy full-path reader')
        spike_count = 0
        for name in channels:
            times = np.atleast_1d(data[name].times)
            if times.ndim != 1 or times.dtype.kind not in 'iuf' or not np.isfinite(times).all():
                raise ValueError('Invalid spike times')
            spike_count += times.size
            if name.startswith('WM_'):
                codes = np.atleast_1d(data[name].codes)
                if codes.ndim != 1 or len(codes) != len(times) or codes.dtype.kind not in 'iuf' or not np.isfinite(codes).all():
                    raise ValueError('Invalid spike codes')
        if not spike_count:
            raise ValueError('No neural spikes')
        families = RecordingCatalog.stimulus_families(path, 3)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, MatReadError, NotImplementedError) as error:
        raise UploadError(422, 'Invalid recording. Upload a readable MAT recording with S1T stimulus names, trial onsets, and WM/BAK neural spike data.') from error
    if stimulus_family not in families:
        raise UploadError(409, "Recording is incompatible with this experiment's stimulus family.")
    return families


async def receive_upload(stream: AsyncIterable[bytes], filename: str, experiment: Experiment,
                         settings: Settings, store: ExperimentStore) -> UploadResult:
    try:
        display_name = unquote(filename, encoding='utf-8', errors='strict').replace('\\', '/').rsplit('/', 1)[-1]
    except UnicodeError as error:
        raise UploadError(422, 'Invalid recording filename.') from error
    if not display_name or len(display_name) > 255 or any(ord(c) < 32 for c in display_name):
        raise UploadError(422, 'Invalid recording filename.')
    if Path(display_name).suffix.lower() != '.mat':
        raise UploadError(415, 'Only MAT recording files are supported. Model checkpoints are not accepted.')
    identity = str(uuid4())
    root = settings.upload_dir or settings.database_path.parent / 'uploads'
    directory = root / identity
    staged = directory / 'recording.part'
    # The legacy full-path parser derives session labels from underscore-separated
    # filename components. Keep that convention using a server-generated name.
    final = directory / ('upload_{}_recording.mat'.format(identity))
    published = False
    created = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        directory.mkdir(mode=0o700)
        created = True
        size = 0
        digest = hashlib.sha256()
        with staged.open('xb') as handle:
            async for chunk in stream:
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise UploadError(413, 'Recording exceeds the configured upload size limit.')
                digest.update(chunk)
                await run_in_threadpool(handle.write, chunk)
            await run_in_threadpool(handle.flush)
            await run_in_threadpool(os.fsync, handle.fileno())
        families = await run_in_threadpool(validate_recording, staged, experiment.configuration.stimulus_family)
        staged.rename(final)
        recording = Recording(id=identity, display_name=display_name, dataset_name=experiment.configuration.dataset_name,
                              source='upload', stimulus_families=families)
        selected = await run_in_threadpool(store.register_upload, experiment.id,
                                          StoredUpload(recording, final, size, digest.hexdigest()))
        if selected is None:
            raise UploadError(404, 'Experiment not found.')
        published = True
        return UploadResult(experiment=selected)
    except (OSError, sqlite3.Error) as error:
        raise UploadError(503, 'Recording storage is unavailable. Retry after the server storage is restored.') from error
    finally:
        if created and not published:
            for path in (staged, final):
                path.unlink(missing_ok=True)
            directory.rmdir()
