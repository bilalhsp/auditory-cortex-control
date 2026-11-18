# Connect a laptop to an existing Gilbreth GPU allocation

Run `connect-gilbreth.sh` on your laptop using Bash 4+, OpenSSH, and curl. On
Windows use **Git Bash** with your existing Windows SSH configuration. WSL also
works if the login alias and credentials are configured in WSL's SSH environment.
The script itself needs no local Python, Node, or application environment.

The matching updated project must exist on Gilbreth, including
`scripts/web/start_remote_backend.sh`. Configure the operator/password once on
Gilbreth using the existing backend launcher before using this connection script.

## Current node

Use your configured login-node SSH alias and your GitHub Pages origin:

```bash
bash connect-gilbreth.sh \
  --login gilbreth-fe02 \
  --node gilbreth-n6 \
  --frontend-origin https://YOUR_USERNAME.github.io
```

Replace `gilbreth-fe02` with the login alias that works in your laptop's SSH
configuration. The script uses that alias as the compute-node jump host. It
retains normal SSH authentication and host-key verification. Credentials are
never put in frontend configuration or copied to the laptop by this script.

`gilbreth-n6` matches the canonical Slurm node `gilbreth-n006`. The script reads
Slurm's current assignment rather than assuming a fixed compute-node address.
Future nodes can be selected with `--node`, or selected by allocation job ID:

```bash
bash connect-gilbreth.sh --login YOUR_LOGIN_ALIAS --job-id YOUR_JOB_ID \
  --frontend-origin https://YOUR_USERNAME.github.io
```

With exactly one running single-node GPU allocation, omit both selectors and the
script discovers it. With several allocations/nodes, it lists choices and stops
until you select one; it never picks an arbitrary training job. Use an allocation
dedicated to this backend. No new allocation is requested or cancelled.

## Startup and ports

The script:

1. Queries your running GPU allocations through the login node.
2. Resolves and revalidates the selected compute node and job ownership.
3. Starts the API/worker together using one `srun --jobid ... --overlap` step in
   that existing allocation. Scientific work runs on the compute node.
4. Opens a loopback-only SSH tunnel through your login alias to that node.
5. Tries laptop ports **28000, 28001, 28002, 18000, 8000**, in that order.
6. Checks backend health and the exact configured frontend CORS origin.

Only bind/listen failures cause a port fallback. An SSH authentication, host-key,
or connection failure stops with its diagnostic instead of cycling login attempts.
The final console message prints the chosen local backend URL. Keep the terminal
open while using the frontend. Browser local/loopback network permission may need
to be allowed for your GitHub Pages site.

You can override the candidate list, but keep it synchronized with frontend config:

```bash
bash connect-gilbreth.sh --login YOUR_LOGIN_ALIAS --node gilbreth-n6 \
  --frontend-origin https://YOUR_USERNAME.github.io --ports '28000 28001 28002'
```

Other options: `--repo /absolute/remote/project/path`, `--api-port 8000`, and
`--help`. Defaults can also be set with `GILBRETH_LOGIN_HOST`, `GILBRETH_REPO`,
`GILBRETH_FRONTEND_ORIGIN`, and `GILBRETH_LOCAL_PORTS` in your laptop shell.

## Frontend configuration and publishing

The updated `frontend/public/config.js` includes the matching candidate URLs in
`apiBaseUrls`. A nonempty `apiBaseUrl` overrides that list for a fixed endpoint.
At sign-in the frontend checks candidate health and the static application
identity **without credentials**, then
sends credentials only to the selected backend. After login every command, status
request, and download stays pinned to it. A 401 or later network failure never
moves commands or credentials automatically to another backend. If a reconnect
uses a different laptop port, Disconnect and sign in again; saved work is retained.

Update/restart the backend and build/republish the frontend once to include this
discovery feature. The basic `/healthz` response remains compatible; discovery
uses `/healthz?identify=true` to distinguish unrelated local services. Future
candidate-list edits can be made in the deployed runtime `config.js` without a
bundle rebuild. The local frontend launcher still supplies a fixed `apiBaseUrl`
and preserves its existing behavior.

For a repository-path GitHub Pages deployment, use the corresponding Vite base:

```bash
npm run build --prefix frontend -- --base /YOUR_REPOSITORY/
```

`index.html` now resolves `config.js` through `%BASE_URL%`, so it loads under the
same Pages subdirectory. Publish only `frontend/dist/`; the Bash launcher and
private server environment are not website assets. GitHub Pages publishing itself
is a separate workflow; this change does not push or publish the repository.

## Disconnect, reconnect, and stop

Ctrl+C closes only the laptop SSH tunnel. The backend runs detached under the
existing Slurm allocation, so a laptop disconnect does not terminate accepted
computation. Rerunning the script for the same job/node/API port reuses its locked
launcher. If its origin settings changed, restart that backend step with the new
origin; a reconnect cannot change the environment of an already-running process.

Remote logs and the step ID live under the configured cache's `web/launches/`.
The script prints both. Allocation expiration stops the backend; on a later
worker startup, unfinished attempts follow interrupted/manual-retry recovery.
SQLite remains the individual fit/generation queue; Slurm starts the service step,
not a separate Slurm job for each request.

To stop the backend deliberately, finish/stop active work first, then cancel only
the **backend step ID printed by the script** from Gilbreth:

```bash
scancel JOB_ID.STEP_ID
```

Do not replace that with the allocation ID alone. Before using a different
allocation with the same database, stop the old backend; the existing allocator
lock rejects duplicate workers. Existing manually started API/worker processes
should likewise be stopped before first switching to this launcher.
