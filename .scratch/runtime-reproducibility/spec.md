# Reproducible Python and CUDA runtime

Status: implementing

The operator approved making the uv-managed runtime reproducible while preserving
Python 3.8.5 and existing scientific package versions. Package downloads, build
temporary files, kernel caches and isolated environments must use the repository's
configured scratch cache; home is nearly at quota. Preserve the existing shared
environment until isolated validation passes.

## Requirements

- Keep Python 3.8.5, PyTorch 2.0.1, CuPy 12.0.0 and existing scientific versions.
- Lock compatible CUDA 12 user-space components; do not depend on a cluster CUDA
  module or inherited CUDA library paths. The host driver remains a requirement.
- Preserve the legacy source revisions and recorded local source patches.
- Reproduce the scientific environment in isolation using locked dependencies.
- Validate actual PyTorch and CuPy GPU operations, kernel compilation and linear
  algebra, plus the existing controlled real-GPU fit/generation parity probes.
- Fail combined backend/worker startup before accepting work when either PyTorch
  or CuPy is unusable. Preserve API-only metadata access without CUDA.
- Provide repeatable setup/activation and deployment documentation. Do not change
  algorithms, frontend behavior, credentials, saved sessions, models or samples.
- Preserve accepted jobs; any deployment restart must occur after checking that
  work is finished or explicitly stopped.

## Decisions

Use the existing root uv.lock for Python and a separate locked CUDA 12 component
environment. Expose the latter as a toolkit layout to CuPy 12 and configure
launchers to load it. Separate NVIDIA wheel namespaces preserve the PyTorch CUDA
11 runtime. Recreate four scientific source checkouts at exact baseline revisions
and apply checksum-verified recorded patches. Build in fingerprinted scratch
directories; activation atomically repoints .venv while preserving its old target.

## Validation

Full backend and launcher acceptance suites, static type checking, isolated native
GPU preflight, legacy/web fit and controlled generation comparisons. Routine
acceptance remains at the authenticated API/service workflow and launcher CLI
seams; scientific comparisons use real GPU compute and controlled RNG state.
