# 02: Configure an experiment and select backend GPUs

**What to build:** Run the React interface and FastAPI independently, configure a supported experiment, and choose the number of backend-server GPUs to use, defaulting to the maximum available.

**Blocked by:** 01: Resolve implementation gates and capture parity baseline.

**Status:** resolved

- [x] Frontend and backend build and run independently, with deployment-configurable API connectivity and reproducible uv-managed Python startup using the environment decision from 01.
- [x] The interface obtains supported settings from the backend and preserves TIMIT/mVocs, ucdavisBAK, deepspeech2, layer 2, fixed 50-ms bin width, and fixed 200-ms receptive field unless 01 explicitly expands scope.
- [x] Creating and reading an experiment returns stable identity and effective settings; invalid combinations fail clearly, and changing upstream configuration clears stale downstream selections.
- [x] Backend GPU detection respects permitted local devices and the availability definition from 01; return timestamped capacity without multiplying it by remote host counts.
- [x] The GPU-count selector defaults to maximum available, permits valid smaller integer selections, preserves valid explicit choices on refresh, and explains invalidated selections after capacity changes.
- [x] Zero-GPU capacity is displayed explicitly and disables GPU generation without inventing a CPU fallback; API/metadata readiness remains distinguishable from compute readiness.
- [x] Behavioral checks at the selected boundary cover zero/one/multiple GPU capacities, invalid settings/counts, stale snapshots, and independent frontend/API connectivity; client configuration contains no secrets.
- [x] Introduce only the minimal configuration and capability adapters needed for this slice; existing Streamlit and scientific entry points remain usable.

## Verification

All 11 HTTP contract tests, Python/TypeScript static checks, production frontend
build, independent backend archive smoke test, and actual browser interactions
passed. The issue 02 verification record captures scope, observations, and the
existing-environment packaging limitation. GPU allocation and scientific jobs
remain later slices; experiments currently store the requested count only.

Issues 03 and 04 are now unblocked.
