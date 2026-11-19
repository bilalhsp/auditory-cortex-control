# Experiment setup frontend

Use Node 22.12+ or a newer supported Node version. From this directory:

```bash
npm ci
npm run dev
```

Vite serves the interface at `http://localhost:5173` and proxies `/api` to the
independent backend at `http://127.0.0.1:8000`. Configure backend authentication
as described in its README, then sign in through the interface.

```bash
npm run typecheck
npm run build
```

Deploy the resulting `dist/` directory with a static web server. Vite preview is
only a local build check, not the production web server.

## Independent backend deployment


Edit the served `config.js` to set `window.AUDITORY_CONFIG.apiBaseUrl` to the
backend's HTTPS origin, without a trailing slash. No frontend rebuild is needed.
Empty uses the current origin, suitable for a reverse proxy that forwards `/api`.
Configure the API's allowed frontend origins accordingly. Serve `config.js` with
cache revalidation so endpoint changes are picked up on reload. Never include
passwords, hashes, or other secrets in this public file or in the JavaScript build.

The GPU selector initially defaults to all available devices. Refresh preserves
an explicit valid choice. If capacity invalidates it, choose again; requests
are never silently reduced. With no GPUs, setup can still be saved, but GPU
generation remains unavailable. Saving a setup reserves no GPU and starts no job.

After saving an experiment, select a compatible existing recording and choose
"Use this recording" to persist the attachment. Refresh reloads the catalog;
missing/unreadable entries are reported separately from a valid empty list.

Changing stimulus family or GPU selection clears the current setup and recording
display. Previous experiments and selections remain persisted in the backend.
Changing the selected recording advances its revision without deleting prior
results. Late requests from an old experiment cannot replace the current setup.
The upload panel accepts MAT recordings up to the backend's advertised limit.
It shows upload/validation progress and server errors, and attaches only valid,
compatible recordings. Same-named uploads receive different identities; replacing
one retains its earlier artifact. Failed validation preserves the saved selection.
The file input resets after success, allowing the same filename to be uploaded
again. Older in-flight responses cannot overwrite a newer recording revision or
a different experiment. Upload availability is capability-driven so older API
deployments can continue supporting setup without the upload panel.

An existing or validated uploaded recording enables "Fit encoding model" when at least one GPU was
selected. A fit submission returns immediately; the panel polls its persisted
queued/loading/fitting/writing state and displays ranked channel correlations
and the best finite channel after successful artifact publication. Signed and
decimal channel identities are preserved; unavailable numeric values are labeled.
The backend requires its separate fit worker, described in `backend/README.md`.
Failed/interrupted fits expose a retry through a new fit attempt; recording or
experiment changes clear dependent UI without deleting earlier server artifacts.

Signing out drops in-memory credentials and leaves accepted work running.
Completed fits expose one target channel, stretch/one-hot objective, requested
1–4-second duration (default 2), and 1/2/4/8/10 clips (default 4). Generation
status polls independently; successful output exposes an authenticated ZIP
download without putting credentials in a URL. The actual native audio length is
displayed with the completed clip count. Unused requested GPUs are explained;
only one target runs in this slice. New generations retain earlier server output.
Session history lists saved periods and their fit/generation/batch outcomes after
sign-in. Resume setup restores saved configuration, recording context, and the
latest operation identities; active jobs resume polling without resubmission.
View exposes earlier correlations/results, including saved sample downloads.
Disconnect clears credentials while accepted work continues. Stop / close ends
the selected session, stops unfinished work, and retains completed results.
Individual jobs can be stopped; explicit retry creates a distinct attempt. Batch
retry reruns only unfinished targets. Closing/closed sessions show their status
and remain available in history. Closed periods are read-only; Start a new session
opens a new work period when its configuration is saved.

When the backend advertises batch support, completed fits expose a batch panel
with channel/objective checkboxes. Each selected pair becomes a separate target.
The batch GPU selector starts with the saved setup choice (whose initial default
is maximum available); refreshing preserves a valid smaller choice. An invalidated
choice is cleared for explicit reselection, never silently reduced. Capacity is
rechecked atomically on submission and again before worker allocation.

The panel shows per-target states, requested GPUs, parallel slots, and idle GPUs
when targets are fewer than selected devices. Partial batches offer a ZIP of
completed targets plus a public manifest; failed/incomplete target files are
excluded. New batches preserve earlier server results. A batch count can be
explicitly changed without changing the fit's scientific configuration.

Recovery controls use the advertised session capability. Older backends can keep
serving the original fit/generation workflow without recovery requests. Saved-job
views use job identities for polling; changing the selected session discards late
responses from the previous view. Retry eligibility comes from the server, so an
interrupted batch whose targets all completed offers its saved download instead
of an empty retry.


## Laptop SSH tunnel port discovery

`public/config.js` can provide `apiBaseUrls` for multiple local forwarded ports.
The default candidates match `connect-gilbreth.sh`: 28000, 28001, 28002, 18000,
and 8000. A nonempty `apiBaseUrl` overrides the list. Discovery checks health
without credentials during login, then pins the connection; it never retries
scientific commands against another endpoint. See
[the laptop connection guide](../docs/deployment/laptop-connection.md) for existing
Slurm allocation discovery, Windows Git Bash, CORS, and GitHub Pages base paths.


Run the laptop connection script in your terminal before signing in. The header
shows backend reachability, and SSH and startup progress are shown in the laptop
terminal. The browser checks the public backend identity before sending credentials.
