#!/usr/bin/env bash
# Build a locked runtime on scratch; activation is a separate, explicit step.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${AUDITORY_BOOTSTRAP_PYTHON:-$repo/.venv/bin/python}"
exec "$python" "$repo/scripts/web/setup_runtime.py" "$@"
