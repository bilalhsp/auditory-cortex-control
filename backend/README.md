# Auditory cortex API and compute worker

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
Then start the API and one compute worker together in the GPU allocation:

```bash
./start-backend.sh
```

The launcher checks real CUDA access before starting, stops both services on
Ctrl+C, and stops the peer if either service exits. For API-only metadata/history
access or manual process management, use `./start-backend.sh --api-only` and start
`./start-worker.sh` separately when needed. Normal combined startup needs no extra
worker command. URL/port changes are described in `docs/deployment/local-launch.md`.

The API uses HTTP Basic authentication with a salted PBKDF2-SHA256 password
hash. Production requires an HTTPS reverse proxy and authentication request-rate
limits. Keep the API's direct HTTP listener private to that proxy. Protect all
API endpoints; only `/healthz` is intentionally unauthenticated and reveals no
experiment or compute data. The frontend holds credentials in memory and drops
them on sign-out; reloading requires sign-in again. Signing out does not stop accepted work; use the explicit Stop/Close actions.

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

Start one compute worker separately from the API, using the same SQLite file, fit storage,
existing environment, research configuration, and visible GPUs. Its Python path
must include this research repository and the existing local scientific dependency
checkouts (`deepspeech.pytorch`, `Wav2Letter`, `utils_jgm`, `pycochleagram`). Use
the paths already configured for the legacy scientific application; do not install
replacement packages. With those paths in `PYTHONPATH`:

```bash
uv run --no-sync python -m backend.worker
```

`--once` processes one queued fit or generation and exits. The source archive also includes the
worker: put the archive and research/dependency paths on `PYTHONPATH`, then use the
existing Python executable with `-m backend.worker` from any directory. Deploy
the API and worker as separate supervised processes on the backend server, with
shared persistent SQLite/filesystem access. A kernel lock beside SQLite prevents
two fit workers from owning that database. This is a single-host queue, not a
multi-host scheduler. Never run another allocator against the dedicated GPUs.

Each fit or single-target generation uses one GPU; pending operations run serially. Fits construct a fresh generator; generations launch an isolated sampler process.
The API subtracts an active fit or generation from available capacity. Saving an experiment
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

Browser disconnect/sign-out leave accepted work intact. An API-only restart
leaves a separately managed worker running; restarting the combined launcher
stops its worker and follows interrupted/manual-retry recovery. A worker
restart marks previously running fits `interrupted` without automatic retry;
queued work survives and resumes. Owned compute children retain the allocator lock and receive a Linux parent-death
signal. Cancellation waits for process exit before releasing reservations. Failed/interrupted attempts
retain diagnostics, never report ready artifacts, and require a new fit to retry.
History/reconnection UI and explicit Stop/Close are described below. This slice
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

## Single-target generation

After a fit succeeds, `POST /api/v1/fits/{id}/generations` accepts
`request_id` (UUID), `channel_id` (exact string from the fit), `objective`
(`stretch` or `one-hot`), `duration_seconds` (1–4, default 2), `clip_count`
(1/2/4/8/10, default 4), and `seed` (default 42). HTTP 202 returns a stable
generation/job identity and immutable configuration. GPU device IDs and paths
are never accepted from the browser. Request UUID reuse is idempotent only for
the same fit/settings; conflicting reuse returns 409.

Generation requires the current recording revision, a completed fit with verified
model/config/provenance checksums, a finite-correlation channel belonging to that
fit, and at least one available GPU. Stale/incomplete/missing-artifact fits return
409; unknown fits 404; invalid settings/channels 422. Accepted requests remain
pinned when the browser disconnects or the selected recording changes.

`GET /api/v1/generations/{id}` and the common `GET /api/v1/jobs/{id}` expose
persisted status. `GET /api/v1/generations/{id}/download` returns an authenticated
ZIP containing only this generation's validated clips. Unfinished, failed,
interrupted, missing, or changed output returns 409, never a partial ZIP.

The same single-owner worker handles fits and generation under one allocation
lock. Generation runs in a subprocess using logical device zero from the first
permitted `CUDA_VISIBLE_DEVICES` entry. Its exit code must be zero before model
output is considered. Linux parent-death signalling kills the GPU child if its
worker exits; worker restart marks interrupted operations for manual retry. Run
this worker on Linux with the existing scientific dependency paths/environment.
It reuses `Sampler` with the exact fitted pickle path and original unit ordering,
rather than looking up a shared native session-zero model. The optional explicit
artifact argument preserves the legacy sampler/CLI default behavior.

A single target uses one GPU. The saved requested count remains unchanged, with
`used_gpu_count: 1` and an explicit `unused_gpu_count`. Additional requested GPUs
remain idle for this target; batch scheduling follows in issue 08. There is no
CPU fallback, concurrent multi-target allocation, or automatic retry here.

Private UUID workspaces under `generations/` beside SQLite retain input context,
raw sampler output, checked clips, manifest, provenance, and diagnostics. Model,
checkpoint, configuration, and sampler identity are recorded privately. Public
errors identify failure without serializing exception text. Earlier generations
are retained, and no UI initialization deletes output. Session-history browsing
and explicit Stop/Close follow in issue 09.

Output validation requires the requested clip count, mono 16-kHz PCM16 WAVs,
complete payloads, correct native frame count, and checksums. Existing RMS
normalization/clipping is unchanged. Native latent rounding maps requested
1/2/3/4 seconds to 15,392/35,872/51,232/66,592 frames (0.962/2.242/3.202/4.162 s).
The UI reports actual duration instead of trimming scientific output. ZIP
membership excludes models, recordings, raw files, diagnostics, and provenance.

Generation seeds are applied before model initialization as well as sampling.
Private compute provenance records initialization seed and CPU/CUDA RNG hashes.
The historical issue 01 WAVs measured repeats within a single initialized sampler;
fresh legacy processes also differ from those WAVs. The operator approved
controlled legacy/web comparisons with identical initialization/RNG state as the
migration acceptance check. Historical evidence is retained separately; this is
not a claim of portable audio determinism across processes or GPU/runtime changes.

## Batches and GPU reservations

`POST /api/v1/fits/{id}/batches` accepts a client `request_id`, an explicit
`gpu_count`, a nonempty list of unique `{channel_id, objective}` targets, and
shared `duration_seconds`, `clip_count`, and `seed` settings with the same defaults
as single-target generation. The batch GPU count is an explicit resource choice;
scientific settings remain pinned to the completed fit. Target devices are never
accepted from the browser. Unknown/duplicate/unavailable targets or invalid
settings return 422; stale fits and insufficient capacity return 409.

SQLite serializes reservation admission under `BEGIN IMMEDIATE`. Submission
reserves the full requested count, including idle devices when there are fewer
targets than GPUs. The capacity endpoint subtracts queued/running reservations.
Concurrent requests cannot reserve the same logical device. No count is silently
reduced. A repeated identical request UUID returns its original batch even when
capacity is now occupied; conflicting reuse returns 409.

The single allocator processes one batch group at a time. Each used GPU owns one
worker slot; that slot takes the next pending target after its current target
exits. At most the selected count runs concurrently, with excess targets queued
in waves. Other accepted batch groups retain their disjoint reservations while
waiting. Standalone fits/generations wait behind reserved batches. This is a
single-host allocator with dedicated backend GPUs, not a remote-host scheduler.

Allocation rechecks actual local count and the runtime visibility signature.
Changed/insufficient capacity fails the attempt explicitly before launching;
visibility/device loss before later targets also fails those targets. Each child
receives exactly its reserved runtime device through `CUDA_VISIBLE_DEVICES` and
uses logical CUDA device zero inside that restricted process. Physical device
identifiers stay private. Every target reuses issue 07's sampler, artifact
verification, seeded initialization, exit-code checks, and WAV validation.

`GET /api/v1/batches/{id}` exposes overall and per-target outcomes, and the common
job endpoint accepts the batch job ID. Target IDs/job IDs also use the existing
generation/status/download endpoints. `GET /api/v1/batches/{id}/manifest` exposes
public target/clip metadata. All routes require operator authentication.

Overall states distinguish `succeeded`, `partial`, and `failed`. Validated output
from successful targets remains available when another target fails; unmanifested
or nonzero-exit target files stay private and are excluded. A finished batch with
completed targets can be downloaded from `/api/v1/batches/{id}/download` as a ZIP
with `manifest.json` plus `{target_id}/clip-NNN.wav` entries. This ZIP is built on
private disk and removed after the response; sample files remain untouched.
Changed/missing successful output rejects download rather than silently omitting
it. Per-target downloads can be used once that target succeeds.

Target UUID directories isolate inputs, output, provenance, and diagnostics.
Batch diagnostics record the batch identity and reserved logical slots privately.
Reservations release only after target workers exit and the batch reaches its
terminal outcome. Worker restart interrupts unfinished targets, preserves completed
targets and their downloadable files, and releases the interrupted reservation.
Retries are explicit new attempts. Session Stop/Close is available through the lifecycle API below.

Deploy the API and worker together. Stop the prior worker/API before installing
this backend version, then start the new API and worker against the same database
and files. Startup migrates metadata transactionally by adding the batch table and
nullable `generations.batch_id` column. Do not keep a pre-batch worker running
against the new API: its queue interpretation predates grouped targets. Completed
scientific files are preserved; stopping running work follows the documented
interrupted/manual-retry policy. Frontend static assets remain independently
deployable and use the advertised batch capability.

Real execution was benchmarked on the available single GPU with two targets in
waves; multi-GPU scheduling, admission, and device mapping are covered using
controlled external compute and the real subprocess launcher. See
`.scratch/web-migration/issue08-benchmark.json`. No latency threshold was imposed.

## Sessions, recovery, Stop, and retry

Application sessions have distinct stable IDs and group saved experiments/jobs.
Saving the first experiment creates an open session; `POST /experiments` can use
`?session_id={id}` to add a configuration to an existing open session. Startup
migrates legacy experiments into distinct persistent sessions without deleting
recordings, fitted models, or generated output.

Authenticated recovery routes:

- `GET /api/v1/sessions`: saved session identities, timestamps, and status.
- `GET /api/v1/sessions/{id}`: experiment configurations and job summaries,
  including retry relationships.
- `GET /api/v1/experiments/{id}/work`: persisted fits, standalone generations,
  and batches for workflow restoration.
- Existing fit/job/generation/batch endpoints remain the authoritative resource
  views. History and downloads remain available with zero GPUs.

Disconnect/sign-out leave accepted work running. API-only restarts preserve the
separate worker; restarting the combined backend stops both services. Reconnect
requires sign-in, then history and Resume setup recover stable identities and
current results. No compute is automatically resubmitted by page initialization.

`POST /api/v1/jobs/{job_id}/cancel` is idempotent. Queued jobs cancel immediately;
running jobs expose `cancelling` while retaining their running reservation. The
allocator consumes the durable request, terminates its owned process handle,
waits for exit (escalating to kill if necessary), and then commits `cancelled`.
A transaction orders cancellation against completion: committed terminal results
are immutable; a cancellation committed before final result publication wins.
Files from uncommitted/failed work remain private, not downloadable results.

`POST /api/v1/sessions/{id}/close` prevents new submissions, recording changes,
and uploads in that session and stops its unfinished jobs. Status remains
`closing` until running workers finish cancellation/reconciliation, then becomes
`closed`. Other sessions are unaffected. Completed target manifests/results
survive Stop/Close, including completed targets of a cancelled batch. Closed
sessions are read-only; start a new session for further work.

Both fitting and generation now run in owned subprocesses using the existing
scientific entry points/environment. Children inherit the kernel allocator lock,
so a replacement allocator cannot reconcile/release GPUs while an old child is
still alive. Linux parent-death signals handle owner exit/startup races. Worker
restart marks unfinished running attempts interrupted (or cancelled if a stop
request was already recorded), preserving completed targets and files. Queued
accepted commands survive restart. If the compute worker is offline, Stop remains
pending until it exits/restarts and reconciliation confirms completion; a lost
connection or timeout alone does not free an occupied reservation.

`POST /api/v1/jobs/{job_id}/retry` takes a new client `request_id` UUID. Only
failed/interrupted/stopped/partial work can be retried; repeats of that UUID
return the same new attempt. Fits retain recording/revision/seed; generations
retain the exact fit/target/settings. Batch retry contains only unfinished
targets and keeps the explicitly requested GPU count. Attempt IDs/output paths
are distinct, and history links each retry to its source job. Changed recording
context or a closed source session requires a new workflow instead of silently
retargeting an old command. No automatic scientific retries occur.

Declined/out-of-scope guarantees remain unchanged: no seamless checkpoint
continuation, automatic scientific retry, backup implementation, public
registration, or multi-user collaboration. Session lifecycle never deletes saved
scientific results. The operator pilot and application release are approved; see `docs/deployment/release-status.md`. Upgrade API and worker together using the existing coordinated-deployment
procedure; lifecycle metadata migration is transactional.

Retry command ownership is bound durably before enqueueing. The public retry UUID
selects one source job and a private enqueue identity; original command IDs cannot
be adopted as new attempts. Retry provenance commits with the new job. An
interrupted batch with every target already saved returns 409 for Retry and
remains downloadable. The session capability advertises these recovery routes.


## Access and retained data (issue 10)

One configured operator can access all application sessions. There are no tenant
accounts, sharing links, anonymous artifact URLs, or public registration. UUIDs
identify resources; they do not grant access. Every `/api/v1/` command, status,
catalog, history, manifest, upload, and download requires the configured Basic
credentials. Responses, including authentication errors, use `Cache-Control:
no-store`. CORS controls browser origins and does not replace authentication.

Deploy only frontend `dist/` as static content. Never expose the repository,
SQLite database (including WAL/SHM), configured cache, upload/fit directories,
model weights, worker workspaces, or diagnostic logs through a static-file alias
or directory listing. Downloads must pass through the authenticated API. Run the
API/worker as a dedicated OS user with access to private persistent data; give
that user ownership of storage and restrict directory access (for example umask
077). Keep secrets in the server environment, not Vite variables or build output.

Public deployment requires HTTPS for both frontend and API, an exact production
`AUDITORY_ALLOWED_ORIGINS`, and a reverse proxy with authentication request-rate
limits. Bind the upstream API to loopback or a private connection reachable only
by that proxy. Forward Authorization, avoid logging credentials or request bodies,
disable proxy caching for `/api/`, and route that prefix only to FastAPI. Configure
TLS/HSTS at the public proxy according to the deployment's certificate setup.
The repository cannot verify a future external proxy: before publishing, check
that HTTP redirects to HTTPS, direct upstream ports and data directories are
unreachable, and unauthenticated API/download requests return 401. Repeat with a
wrong password and wrong username. Health alone is intentionally public.

There is no automatic expiry, startup cleanup, session-close deletion, or deletion
API in this release. Keep metadata and all referenced recordings/models/samples
on persistent filesystem storage across deployments. Completed artifacts remain
available after closure and restart, including completed targets of partial or
cancelled batches. Incomplete/failed output remains private and is excluded from
downloads. Never sweep workspaces based on age or browser connection state:
accepted queued/running work and historical references must remain intact.
Rejected upload staging and per-request temporary download ZIPs are cleaned up;
those temporary files are distinct from retained scientific artifacts. Initialization
never deletes shared research results. Any future explicit deletion feature needs
its own reference/active-work policy; do not add an ad hoc cleanup cron job.

Backup implementation and backup/restore acceptance were explicitly excluded by
the operator. Restart/recovery tests verify persistence against the same database
and filesystem; they are not evidence of disaster recovery or a tested backup.
