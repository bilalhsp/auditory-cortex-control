# Application release status

**Accepted by the operator on 2026-11-18 (UTC).** The operator reported the pilot
successful and authorized application finalization. The React + TypeScript + Vite
frontend and FastAPI backend are the accepted application workflow for the current
local/SSH deployment. Migration issues 01–11 are covered by the release evidence.

Deployment hostnames, ports, origins, and public hosting may change independently.
No public URL, TLS proxy, or traffic routing was changed by this finalization.
Verify the existing HTTPS/access checklist before public exposure. Streamlit,
its environment, and compatible scientific artifacts remain available for rollback;
retirement requires separate explicit approval.

## Normal startup

The frontend must already be built. In separate terminals:

```bash
./start-frontend.sh
./start-backend.sh
```

The backend command starts and monitors the API and exactly one queue-owning
worker on the same allocated GPU host. It checks actual CUDA access first. If
one service exits, it stops the other and reports an error. Ctrl+C stops both.
There are no automatic service restarts or scientific retries. Accepted browser
work continues after browser disconnect; stopping the backend or losing its Slurm
allocation follows the interrupted/manual-retry policy.

For metadata/history access without a GPU, or externally managed services, use
`./start-backend.sh --api-only`; launch `./start-worker.sh` separately only in that
mode. The allocator lock prevents duplicate workers from owning the same queue.
Slurm allocates the node/GPUs; SQLite schedules individual application jobs. No
Slurm job is submitted per fit or generation.

## Changing deployment addresses

| Setting | How to change it |
| --- | --- |
| Browser-visible backend URL | `./start-frontend.sh --api-url https://api.example.org` |
| Frontend listening address/port | `./start-frontend.sh --host 127.0.0.1 --port 5173` |
| API listening address/port | `./start-backend.sh --host 127.0.0.1 --port 8000` |
| Browser origin permitted by CORS | `AUDITORY_FRONTEND_ORIGIN=https://bilalhsp.github.io ./start-backend.sh` |
| Operator credentials, persistent database/storage, multiple allowed origins | Private `<configured cache_dir>/web/local.env` |
| Alternative trusted backend configuration | `AUDITORY_ENV_FILE=/absolute/path/local.env` for the launcher |

The frontend launcher serves the runtime API URL without modifying the built
assets; changing it needs a launcher restart and browser reload, not a rebuild.
For another static host, edit its deployed `config.js` and configure the API's
allowed origins. A repository/subdirectory deployment may also require a matching
Vite base path and rebuild. Backend-only updates do not require a frontend rebuild
when the API remains compatible.

## Acceptance and operating references

- [Release verification](../../.scratch/web-migration/issue11-verification.md):
  automated acceptance, real-GPU scientific parity, and user-reported pilot sign-off.
- [Local launch](local-launch.md): credentials, existing environment, and SSH ports.
- [Cutover and rollback](web-cutover.md): deployment checks and retained legacy path.

Real TIMIT/mVocs fits preserve all 18 reference channel correlations exactly for
both recording sources. Both audio objectives match byte-for-byte under the
approved controlled initialization/RNG comparison. Historical cross-process
variability remains documented; portable determinism is not promised. Scientific
algorithms were not changed during finalization. Completed samples, metadata, and
models are retained across restarts and deployment-address changes.


Final release checks: **76 backend tests passed**, backend and frontend typechecks
passed, backend archive build passed, and independent Standards/Spec reviews each
reported zero findings. Real-GPU supervisor checks covered shutdown, API failure,
worker failure, and occupied-port cleanup. See the verification record for measured
scientific comparisons and the preserved limits of those claims.


GitHub Pages owner/origin: `bilalhsp` / `https://bilalhsp.github.io`. The sign-in
page provides a launcher-command copy action and a credential-free connection
check. See [laptop connection](laptop-connection.md) for the selected terminal-led
workflow. Operator credentials remain separate from the GitHub username.
