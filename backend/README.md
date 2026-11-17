# Experiment setup API

Run from the repository root using the existing `.venv`. Do not create or upgrade
an environment or run `uv sync` for this slice. Scientific imports and the
Streamlit launcher remain unchanged.

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
stimulus names/types/onsets, and finite WM/BAK spike times; WM codes must match
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

There are no compute jobs or reservations in this slice: all permitted devices
are available. Saving records a requested count, not an allocation. The generation
slice must connect reservation accounting before executing any jobs. Only use
one allocator process when that behavior is introduced.

Run the complete current API suite with the existing environment:

```bash
uv run --no-sync python -m unittest discover -s backend/tests -v
```

The approved test boundary is HTTP with real SQLite/application services and a
controlled external GPU-count provider. No scientific model downloads or GPU
computations are needed for these contract tests.
