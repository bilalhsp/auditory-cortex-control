# Experiment setup API

Run from the repository root using the existing `.venv`. Do not create or upgrade
an environment or run `uv sync` for this slice. The HTTP server does not load
scientific models. The legacy Streamlit launcher remains available during migration.

Generate an operator password hash interactively:

```bash
uv run --no-sync python -m backend.auth
```

Set `AUDITORY_OPERATOR` and `AUDITORY_PASSWORD_HASH` in the server environment.
Use the entire printed hash literally (including dollar signs); never put the
password or hash in frontend configuration. There are no default credentials.
Then start the independently deployable API:

```bash
uv run --no-sync python -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000
```

The API uses HTTP Basic authentication with a salted PBKDF2-SHA256 password
hash. Production requires an HTTPS reverse proxy and authentication request-rate
limits. Keep the API's direct HTTP listener private to that proxy. Protect all
API endpoints; only `/healthz` is intentionally unauthenticated and reveals no
experiment or compute data. The frontend holds credentials in memory and drops
them on sign-out; reloading requires sign-in again. Stop/Close computation is a
separate, later workflow; signing out does not stop accepted work.

## Configuration

| Variable | Meaning |
| --- | --- |
| `AUDITORY_OPERATOR` | Required single-operator username |
| `AUDITORY_PASSWORD_HASH` | Required output of the interactive hash command |
| `AUDITORY_DATABASE_PATH` | SQLite file; default is `web/sessions.sqlite3` inside the repository's configured cache |
| `AUDITORY_ALLOWED_ORIGINS` | Comma-separated frontend origins; default `http://localhost:5173`; wildcards rejected |
| `AUDITORY_NEURAL_DATA_DIR` | Dataset root; defaults to `neural_data_dir` in the existing research configuration |
| `AUDITORY_RECORDINGS_METADATA` | Existing ucdavisBAK YAML session catalog; defaults to the research package's `ucdavisBAK_config.yml` |
| `AUDITORY_UPLOAD_DIR` | Persistent private upload directory; defaults to `uploads/` beside the SQLite database |
| `AUDITORY_MAX_UPLOAD_BYTES` | Positive byte limit; defaults to 50 MiB and is exposed in capabilities |
| `AUDITORY_FIT_DIR` | Persistent private fitted-model storage; defaults to `fits/` beside SQLite |
| `CUDA_VISIBLE_DEVICES` | Devices permitted to the backend; honored by torch detection |

When deploying the backend artifact without the research repository, explicitly set
`AUDITORY_DATABASE_PATH`. For recording discovery outside the repository, also
set `AUDITORY_NEURAL_DATA_DIR` and `AUDITORY_RECORDINGS_METADATA`; the archive
does not bundle research data or a duplicate session catalog.
Production origins must match the frontend's HTTPS
origin exactly. An HTTPS proxy may expose the frontend and API at one origin;
they remain independent build artifacts and processes.

Build the backend source artifact using the existing environment. This uses
the standard library, avoiding the existing setuptools/packaging mismatch and
creating no build environment:

```bash
uv run --no-sync python -m backend.build /path/to/artifacts/auditory-api.pyz
```

The archive contains the backend code, not Python dependencies or model weights.
It runs using the same existing Python environment with FastAPI, Uvicorn, and
PyYAML, NumPy, and SciPy installed. Exact current direct versions are recorded in
`requirements.txt`; this slice continues using the existing environment. Supply
the required environment variables, then launch it
from any working directory:

```bash
/path/to/existing/.venv/bin/python /path/to/artifacts/auditory-api.pyz backend.app:create_app --factory --host 127.0.0.1 --port 8000
```

## Contract

- `GET /healthz`: API liveness, independent of GPU/model availability.
- `GET /api/v1/capabilities`: authenticated supported settings and timestamped
  local GPU counts, with `ready`, `no_gpu`, or `unavailable` compute status.
- `POST /api/v1/experiments`: validated configuration with `gpu_count`; returns
  HTTP 201 and stable experiment identity, creation time, and effective settings.
- `GET /api/v1/experiments/{id}`: saved experiment; persists across API restart.
- `GET /api/v1/experiments/{id}/recordings`: compatible existing recordings,
  with opaque IDs, display names, and a count of unavailable catalog entries.
- `POST /api/v1/experiments/{id}/recording-selection`: attach an opaque
  `recording_id`; returns the experiment with its selected recording and revision.
- `POST /api/v1/experiments/{id}/recording-upload`: upload raw MAT bytes using
  `Content-Type: application/octet-stream` and a percent-encoded UTF-8
  `X-Recording-Filename` display name. Authentication and declared size checks
  run before consuming the body; streamed byte counting also enforces the limit
  when Content-Length is absent or inaccurate. Proxy request limits should align
  with the configured byte limit.
- `POST /api/v1/experiments/{id}/fits`: queue an existing or validated uploaded recording fit; send
  `request_id` (client UUID), `recording_id`, `recording_revision`, and optional
  `seed` (default 42). HTTP 202 returns immutable fit/job identities and context.
- `GET /api/v1/fits/{id}` and `GET /api/v1/jobs/{id}`: authenticated persisted
  lifecycle and ranked correlations. Responses contain no private filesystem paths.

Uploads validate synchronously before selection: the UI reports pending while
the request runs; HTTP 201 returns `validation_status: valid` and the updated
experiment with a stable recording identity. Invalid files return 415/422,
incompatible stimulus families 409, and oversize files 413, with structured
`validation_status: invalid` feedback. Invalid uploads are not registered,
their staged files are removed, and prior selections remain unchanged. Storage
failures return 503 with `validation_status: error`. This is not a durable
background-validation job.

Validation uses SciPy's MAT parser and the shared trial-family reader, never
pickle/model-checkpoint loading. Uploaded ucdavisBAK recordings must have the
legacy full-path reader's S1T trials (optional S2 targets), one-dimensional
stimulus names/types/onsets, Windows-style backslash stimulus paths, and finite WM/BAK spike times; WM codes must match
their spike times. Validation establishes format and stimulus-family
compatibility, not eventual fit quality or GPU readiness.

Each upload uses a private server-generated directory and canonical filename;
the supplied filename is reduced to a display basename only. The generated
filename preserves the legacy underscore-separated session-label convention.
SQLite retains artifact path, size, checksum, and experiment ownership privately;
public responses never serialize those paths. Publishing the validated artifact
precedes an atomic metadata/selection transaction. Replacing an upload increments
the recording revision and retains previous files. Older registered uploads may
be reselected by ID only within their original experiment. Put upload storage
on persistent filesystem storage; no automatic retention cleanup is implemented.
Once validation completes, a protected finalizer owns artifact publication,
metadata commit, and failure cleanup together. Cancelling the HTTP request cannot
delete an artifact while its commit is still running. Earlier interruptions
discard only the incomplete upload; completed earlier uploads remain untouched.

Discovery reuses the research dataset's YAML session catalog and reads only MAT
`TrialStimData`. It follows the legacy TIMIT/mvox folder and trial-type conventions,
excluding S2 target-only trials. It never constructs a generator or loads model
weights. Only catalogued files within the configured dataset's `Data` directory
are visible; browser contracts never accept a filesystem path. Missing or
unreadable files are skipped with a count; unavailable catalog/storage returns
503, distinct from a valid empty list. Selection rechecks availability and stimulus
compatibility (404 for unavailable recording, 409 for incompatible stimulus family).

Selections are persisted per experiment. Selecting the same recording is
idempotent; changing it increments `recording_revision` without deleting any
data or artifacts. Later fit/generation resources must pin experiment identity
and recording revision so a changed selection cannot silently retarget old work.
Fit construction must resolve the opaque recording ID back to the existing
catalog's native dataset session on the server; never derive a session ID or
filesystem location from the display name supplied to the browser.

Current supported settings are TIMIT/mVocs, ucdavisBAK, deepspeech2, layer 2,
50-ms bins, and 200-ms receptive field. Unknown settings/fields and non-integer
GPU counts return 422. Capacity is rechecked on creation; insufficient capacity
returns 409 without changing the request. Zero GPUs permits setup with count zero,
but does not enable CPU generation.

## Fit worker

Start one worker separately from the API, using the same SQLite file, fit storage,
existing environment, research configuration, and visible GPUs. Its Python path
must include this research repository and the existing local scientific dependency
checkouts (`deepspeech.pytorch`, `Wav2Letter`, `utils_jgm`, `pycochleagram`). Use
the paths already configured for the legacy scientific application; do not install
replacement packages. With those paths in `PYTHONPATH`:

```bash
uv run --no-sync python -m backend.worker
```

`--once` processes one queued fit and exits. The source archive also includes the
worker: put the archive and research/dependency paths on `PYTHONPATH`, then use the
existing Python executable with `-m backend.worker` from any directory. Deploy
the API and worker as separate supervised processes on the backend server, with
shared persistent SQLite/filesystem access. A kernel lock beside SQLite prevents
two fit workers from owning that database. This is a single-host queue, not a
multi-host scheduler. Never run another allocator against the dedicated GPUs.

Each fit uses one GPU and a newly constructed generator; pending fits run serially.
The API subtracts an active fit from available capacity. Saving an experiment
records the requested GPU count; multi-GPU generation allocation follows in issue
08. No CPU fit fallback is advertised.

Submission snapshots recording bytes and session metadata under a private UUID
workspace before committing the queued job. A recording revision check at commit
rejects a concurrently changed selection. Reusing a request UUID with identical
context returns its existing fit, including after a lost response; conflicting
reuse returns 409. New attempts require new request UUIDs. Unknown/missing
recordings return 404, stale/incompatible/no-GPU context 409, unavailable catalog
or storage 503.

Validated uploads use the same fit endpoint and lifecycle. The server resolves
the upload only within its original experiment and checks the pinned bytes against
the checksum and size registered during validation. Missing uploads require
re-upload (404); changed files are rejected (409) before a job is accepted. Browser
filenames never resolve paths. Unvalidated files have no selectable recording ID.
Upload fitting uses the legacy full-path parser and internal session zero, while
the fit UUID controls every model/configuration output path. Two uploads with the
same display name therefore retain distinct recording, fit, and artifact IDs.
Replacing a selection leaves earlier accepted fits and their pinned inputs intact.

The worker persists `queued` → `running` (loading/fitting/writing) → `succeeded`
or `failed`. It reuses three-fold TRF cross-validation and the original data
assembly. Success requires nonempty model/config files, provenance, and a checksum
manifest. Each workspace holds `model.pkl`, `trf_config.json`, `provenance.json`,
`manifest.json`, pinned inputs, and private `diagnostics.log`. Provenance records
effective configuration, recording/metadata/model hashes, scientific source hash,
Python/package/CUDA versions, seed, and operation identities. Keep these directories
private and persistent; they are never served as static content.

Correlations sort from largest to smallest; channel IDs remain strings preserving
signs and decimals. NaN/infinite correlations serialize as null (Unavailable),
after finite results; no finite values means no best-channel summary. Public
errors identify the failing phase without serializing scientific exception details.

Browser disconnect/sign-out and API restart leave accepted work intact. A worker
restart marks previously running fits `interrupted` without automatic retry;
queued work survives and resumes. Computation runs inside the lock-owning worker,
so killing it cannot leave a separate child using GPUs. Failed/interrupted attempts
retain diagnostics, never report ready artifacts, and require a new fit to retry.
History/reconnection UI and explicit Stop/Close follow in issue 09. This slice
does not enable generation from incomplete fits.

Run the complete current API suite with the existing environment:

```bash
uv run --no-sync python -m unittest discover -s backend/tests -v
```

The approved test boundary is HTTP with real SQLite/application services and a
controlled external GPU-count provider and compute runner. No scientific model
downloads or GPU computations are needed for these contract tests. Real GPU parity
evidence is recorded separately in `.scratch/web-migration/issue05-parity.json`
and `.scratch/web-migration/issue06-parity.json`.
