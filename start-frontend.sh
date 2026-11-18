#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/web/common.sh"
exec "$WEB_PYTHON" "$WEB_REPO_ROOT/scripts/web/serve_frontend.py" \
    --directory "$WEB_REPO_ROOT/frontend/dist" "$@"
