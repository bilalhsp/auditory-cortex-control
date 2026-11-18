# Local launch commands

The frontend is already built. Run these from the repository root in separate
terminals, using the existing `.venv`; no environment creation or dependency
installation is performed. Each script also works when invoked by absolute path
from another directory.

```bash
./start-frontend.sh
```

Open http://127.0.0.1:5173. This serves `frontend/dist/` and dynamically supplies
`config.js` with API URL http://127.0.0.1:8000. It never rebuilds or edits the saved
build. Rebuild only after changing frontend source/dependencies.

In a working Slurm GPU allocation, on the same compute node, start the API and
one worker in separate terminals:

```bash
./start-backend.sh
```

```bash
./start-worker.sh
```

The backend automatically loads `<configured cache_dir>/web/local.env`, including
the username/password hash created earlier. If that file does not exist, the
first interactive backend launch prompts for a username and a password of at
least 12 characters, then stores the hash in a private mode-0600 file. Sign in
using the original password, not the hash. No password is included in frontend
assets. To change a password later, generate a new hash with
`.venv/bin/python -m backend.auth`, update `local.env`, and restart the API.

The worker is required for fitting and generation; without it accepted jobs wait
in the queue. It verifies actual CUDA access on every visible GPU before opening
the queue. If CUDA/MPS fails, fix the GPU allocation/runtime before restarting the
worker. It does not start or reconfigure an MPS daemon. GPU visibility comes from
the allocation environment; the scripts do not override `CUDA_VISIBLE_DEVICES`.

The API can serve metadata/history without a GPU, but normal compute deployment
requires the API and worker on the same allocated GPU host, sharing the same
SQLite database and artifact paths. Stop an older worker before starting another;
the existing allocator lock rejects duplicate workers. The frontend may run on a
separate host. These launchers do not request Slurm allocations or configure SSH.

## Your current alternate SSH ports

If Windows forwards local port 15173 to the frontend's port 5173 and local port
18000 to the API's port 8000, run:

```bash
./start-frontend.sh --api-url http://127.0.0.1:18000
```

```bash
AUDITORY_FRONTEND_ORIGIN=http://127.0.0.1:15173 ./start-backend.sh
```

```bash
./start-worker.sh
```

Open http://127.0.0.1:15173 in Windows. The API URL is the address reached by the
browser, not an address reached by the frontend server. The extra origin is
appended to the configured CORS list. If you move the API to a compute node,
forward the backend port to that same node. A tunnel to the login node will not
reach an API bound to the compute node's loopback address.

## Options and shutdown

- Frontend: `--port 5173`, `--host 127.0.0.1`, `--api-url URL`.
- Backend: Uvicorn options such as `--port 8001`; defaults to loopback port 8000.
- Worker: existing worker options, including `--once` for a single queue pass.
- `AUDITORY_ENV_FILE=/absolute/path/local.env` selects another trusted backend
  shell configuration. Both API and worker must select the same file.
- `AUDITORY_FRONTEND_ORIGIN=URL` appends an exact browser origin to backend CORS.
- Adjacent scientific dependency checkouts are automatically added to PYTHONPATH;
  existing custom PYTHONPATH entries are retained.

Keep the three terminals open. Finish or explicitly stop accepted work through
the UI before stopping services, then use Ctrl+C. Stopping the worker during
compute follows the existing interrupted/manual-retry behavior; completed saved
results remain. Public deployment still requires the HTTPS/access boundary in
[the cutover runbook](web-cutover.md).
