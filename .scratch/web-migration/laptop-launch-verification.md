# Laptop-to-Gilbreth launcher verification

Request: connect from a laptop through an existing configured login SSH alias,
read the assigned GPU compute node, start the combined API/worker there, and
forward a usable local port from a frontend-preconfigured list. The operator
selected existing-allocation mode and identified the current node as gilbreth-n6.
No allocation submission, replacement, or automatic cancellation was requested.

## Implementation

- `connect-gilbreth.sh` runs with Bash 4+, OpenSSH, and curl, including Git Bash.
  It queries the current user's running GPU jobs, resolves Slurm node names,
  supports `--node`/`--job-id`, and refuses ambiguous selection. The n6 alias
  matches canonical n006. The login alias remains the SSH jump host.
- The remote helper revalidates job ownership/node membership, loads the private
  backend environment, and starts a detached locked `srun --jobid --overlap` step
  in that existing allocation. It does not perform GPU computation on the login
  node. Reconnecting to the same instance reuses its lock rather than creating
  another backend. The service remains running after the laptop tunnel closes.
- Actual SSH bind failures select the next loopback-only local port; unrelated
  authentication/host-key failures stop. Health and exact CORS checks run before
  declaring the connection usable. Only the owned tunnel is closed on Ctrl+C.
- The frontend runtime config contains matching candidate URLs. Discovery checks
  health plus static application identity without credentials, then pins all
  authenticated commands/downloads to one endpoint. A 401 or later network loss
  does not fail over. Explicit scalar `apiBaseUrl` remains compatible and wins.
- Basic `/healthz` remains unchanged. Optional `/healthz?identify=true` adds only
  the public service identity, preventing accidental login to unrelated services
  with a generic healthy response. No models, commands, or data become anonymous.
- `%BASE_URL%config.js` supports a GitHub Pages repository path. Bash files use LF
  checkout line endings. The guide covers publishing the changed frontend once,
  private credentials, browser local-network permission, logs, and explicit
  backend-step shutdown without cancelling the allocation.

## Checked

Six subprocess CLI acceptance tests pass with controlled OpenSSH/Slurm adapters:
no allocation; ambiguous allocation; node normalization/blocked-port fallback;
authentication failure without cycling ports; all ports blocked; wrong CORS origin.

Real Slurm verification used the current owned job 12405213 on gilbreth-n006 with
isolated metadata under configured cache:
`auditory-web-checks/laptop-remote/f089e268-6f3c-4b4e-9506-05a71cce035b`.
The remote helper started a real GPU-backed combined backend, served health, and
reused its instance on reconnect. Only its isolated test step 12405213.0 was
cancelled afterward. The allocation remained running; production data was untouched.

Chromium with a real FastAPI service and built frontend passed unauthenticated
candidate selection, rejection of unrelated health, actual login/save, pinned
endpoint after loss, no credential retry after 401, and actionable all-offline
state. The public discovery HTTP assertion failed before implementation (missing
service identity), then passed. Normal frontend and repository-path Pages builds
pass; the latter resolves `/auditory-cortex-control/config.js` correctly.

The existing browser harness required one persistent browser context for this
container's single-process Chromium setup; that harness correction is separate
from application behavior. Backend typechecks and the full backend regression
result are recorded at final completion. No scientific algorithms changed.

## Limits

Actual Windows port reservations, the laptop's SSH config, MFA/host-key prompts,
and a real published GitHub Pages site's local-network permission cannot be
exercised from this Linux workspace. CLI tests substitute those SSH boundary
conditions; the remote Slurm service and actual browser/API behavior were tested
separately. This work neither pushes to GitHub nor publishes a Pages site.
