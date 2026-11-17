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
Fitting, generation, uploads, and session history remain subsequent slices.
Signing out drops in-memory credentials.
