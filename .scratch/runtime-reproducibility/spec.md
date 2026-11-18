# Reproducible Python and CUDA runtime

Status: resolved

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

## Answer

Implemented and activated the locked scratch runtime on Gilbreth. Python 3.8.5,
PyTorch 2.0.1+cu117, CuPy 12.0.0, and all 370 common installed distribution
versions are preserved. CUDA 12.6 libraries and source checkout provenance are
locked separately. Downloads, temporary builds and compiled-kernel caches use
scratch. The existing shared environment remains intact for rollback.

The 78 backend, 8 launcher and 3 post-activation launcher checks passed; Pyright
reported zero errors. Both real mVocs fit sources matched legacy across all 18
channels, and saved cross-environment correlations matched exactly. Both
controlled generation objectives produced byte-identical WAVs with zero PCM
delta. Clean-path validation confirmed kernels, linear solve, FFT and RNG without
cluster CUDA modules or libraries. These controlled comparisons do not claim
portable cross-process or cross-hardware determinism.

Standards review: zero remaining findings. Spec review: zero remaining findings.
Review fixes moved NVIDIA's separate driver cache to scratch and made source
publication atomic; the CLI regression test rejects the previous implementation
and passes the fixed version.

After confirming no accepted work was queued/running under a database write
lock, restarted only backend step 12405213.2 as 12405213.3. API health and GitHub
Pages CORS passed, and metadata states/counts were preserved. Existing saved
results and credentials were unchanged. Unused isolated validation environments
were removed; the active runtime and download cache remain on scratch.

See [setup instructions](../../docs/deployment/python-runtime.md) and
[validation evidence](validation.json). Detailed logs/artifacts remain under
`/scratch/gilbreth/ahmedb/cache/uv-validation/` and the referenced native fit root.
