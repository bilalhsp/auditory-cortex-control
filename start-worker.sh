#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/web/common.sh"
load_backend_config
"$WEB_PYTHON" -m backend.gpu
exec "$WEB_PYTHON" -m backend.worker "$@"
