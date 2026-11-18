#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/web/common.sh"
load_backend_config setup
exec "$WEB_PYTHON" -m backend.serve \
    --host "${AUDITORY_API_HOST:-127.0.0.1}" --port "${AUDITORY_API_PORT:-8000}" "$@"
