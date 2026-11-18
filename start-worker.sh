#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/web/common.sh"
load_backend_config
if ! "$WEB_PYTHON" - <<'PY'
import torch
count = torch.cuda.device_count()
if count < 1:
    raise SystemExit('No CUDA GPUs are visible to this worker.')
for device in range(count):
    torch.ones(1, device='cuda:%d' % device)
    torch.cuda.synchronize(device)
    print('CUDA ready: GPU %d (%s)' % (device, torch.cuda.get_device_name(device)), flush=True)
PY
then
    echo "GPU preflight failed. Run the worker inside a working Slurm GPU allocation; check CUDA/MPS access." >&2
    exit 1
fi
exec "$WEB_PYTHON" -m backend.worker "$@"
