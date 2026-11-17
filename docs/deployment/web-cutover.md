# Web release, local acceptance, and rollback

The approved initial deployment is one authenticated operator, one backend
allocator, SQLite metadata, private filesystem artifacts, and independently
built React assets. Use the existing `.venv` with `uv run --no-sync`; do not
recreate environments or install Gradio. Preserve the configured scientific cache.

## Release artifacts and configuration

Record the release commit and preserve its backend archive, frontend `dist/`,
server environment configuration, and existing scientific dependency revisions.
Build from the repository root:

```bash
npm run build --prefix frontend
uv run --no-sync python -m backend.build /scratch/gilbreth/ahmedb/cache/web-migration-builds/auditory-api.pyz
```

The archive includes source only; the existing environment, research repository,
local scientific dependency checkouts, model weights, and recordings are needed
by the compute worker. See `backend/README.md` for all configuration variables.
Use persistent database/upload/fit directories outside release directories.
Protect password hashes and filesystem paths as server configuration. Never place
credentials in frontend assets. Never serve the scientific cache as static files.

Deploy frontend assets independently. Set the served `config.js` API origin at
runtime; it needs no frontend rebuild. Configure the exact frontend origin in
`AUDITORY_ALLOWED_ORIGINS`. API and worker must be upgraded together and use the
same database, storage, environment, and GPU visibility. There must be only one
allocator for that database; its kernel lock also covers owned compute children.
A successful API health response does not imply a running worker or usable CUDA.

## Local URL-and-port acceptance

The operator approved local deployments for issue 11 verification. The reproducible
browser probe starts built frontend assets at `http://127.0.0.1:5173` and a
separate API at `http://127.0.0.1:8765`. It uses an isolated temporary SQLite
fixture and controlled compute, tests real HTTP/CORS/credentials and frontend
runtime API configuration, then shuts down both servers. Loopback HTTP is for
local acceptance only; public access requires the HTTPS/proxy boundary below.

```bash
uv run --no-sync python .scratch/web-migration/issue11_browser_probe.py
```

It requires the existing cached Playwright installation/browsers under the
configured cache (`auditory-web-checks/`). It does not install tooling. The probe
checks reconnect without duplicate commands, explicit retry, disconnect/recover,
saved downloads, session closure with partial saved targets, history after reload,
late-reply selection protection, and mobile layout. This is automated acceptance,
not operator release sign-off.

Run the HTTP release matrix for both TIMIT/mVocs and existing/uploaded recordings:

```bash
uv run --no-sync python -m unittest backend.tests.test_acceptance -v
```

The compute substitute verifies workflows, isolation, and persistence. Scientific
acceptance uses the native probes and saved comparisons under
`.scratch/web-migration/`; it must not be inferred from controlled output alone.
Record initialization, queue wait (`started_at - created_at`), compute duration,
GPU memory/device, job IDs, and input/configuration hashes. There is no approved
latency release threshold. Do not claim physical multi-GPU throughput from a
one-GPU server; controlled tests verify full-count admission and target waves.

Inspect private per-job workspaces using the saved fit/generation/batch ID.
`input.json`, manifests, provenance, diagnostics, and process exit records correlate
compute failures with HTTP job status. Do not publish private logs or expose them
as artifact downloads. A queued job with no worker, CUDA unavailable, and an
interrupted worker have distinct operational causes even when API health is OK.

## Operator pilot and release gate

Before switching production traffic, record an explicit operator pilot/sign-off
against the chosen release commit and measured scientific acceptance results:

1. Sign in, confirm visible/available GPU counts and maximum default, select a
   smaller count where hardware permits, and verify that it is retained.
2. For both families, select an existing recording and upload a recording, fit,
   inspect correlations/channel identities, generate both objectives repeatedly,
   run a batch, and download only the selected result's valid WAVs.
3. Disconnect during accepted work; reconnect and resume history. Stop/Close an
   active session, wait for workers to exit/GPU release, and download prior saved
   samples. Confirm a different session's results remain available.
4. Check failure/manual retry and zero-GPU history access. A failed or incomplete
   output must never be presented as successful or included in a download.
5. For public deployment, verify HTTPS redirects/certificates, proxy rate limits,
   exact allowed origins, rejected missing/wrong credentials on direct downloads,
   unreachable upstream ports, no API caching, and no static data/log access.

Automated local checks cannot stand in for the operator pilot or an external
proxy check. Keep each outstanding item visible; do not archive Streamlit or
announce production readiness while acceptance remains incomplete.

## Controlled cutover

Use a maintenance period: stop new submissions, inspect accepted work, and either
let it finish or explicitly stop it and wait for child exit. Verify no remaining
compute process owns the allocator before replacing API/worker. Closing a session
preserves its completed results but prevents new submissions in that session.
Start the matched API/worker release against the same persistent storage. Check
health, authenticated history and downloads, CUDA capacity, and a deliberately
submitted compute job. Then switch frontend assets/configuration or proxy routing
to the approved release and repeat access checks. Never run older workers beside
a newer API or move/delete scientific data during cutover.

## Rollback rehearsal and procedure

Keep the original Streamlit launcher, existing Python environment, dependency
checkouts, baseline artifacts, and legacy scientific CLI/notebook interfaces.
No calendar retirement window has been approved. The legacy UI clears its dated
output directory during initialization; rollback must use a fresh isolated work
directory, never a web artifact directory or prior dated legacy output directory.

```bash
# Use the scientific PYTHONPATH documented with the native probes.
uv run --no-sync python .scratch/web-migration/issue11_rollback_probe.py
```

The rehearsal starts the retained `streamlit_app.py` server on a private ephemeral
port and checks health without opening a session against shared output. It then
runs the same legacy Frontend via Streamlit's app runner with a fresh private cache
workspace and verifies the setup UI. Hashes of retained baseline models/audio
must remain unchanged. It does not claim a production proxy switch or a complete
legacy scientific rerun; scientific comparisons are separate evidence.

For an actual rollback, stop new web commands, drain or explicitly stop accepted
work, and wait for worker/child exit. Retain the web database and every referenced
artifact in place; do not down-migrate SQLite or overwrite it with an older copy.
Prepare the isolated legacy UI workspace as rehearsed, retain the same scientific
runtime, and redirect the authenticated proxy to that legacy UI only after its
local health/UI checks pass. Keep public authentication in front of Streamlit.
The legacy UI does not read web SQLite session history; preserved web results
remain available when the matched web release is restarted, not automatically
imported into legacy history. Recheck baseline artifacts and record the operator's
rollback decision. Returning to web uses the retained matched API/worker release
and the unchanged persistent storage.

Backup/restore is explicitly outside scope. Retaining runtime and artifacts is a
rollback strategy, not a tested disaster-recovery guarantee. Retire Streamlit only
after the operator separately approves retirement.
