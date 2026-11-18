# Reproducible GPU runtime

Run from a Slurm GPU allocation with `uv 0.13.1`, Git, and the existing Python
3.8.5 interpreter available:

```bash
./setup-runtime.sh --plan
./setup-runtime.sh
```

The default storage root comes from `cache_dir` in `auditory_cortex/config.yml`:
`/scratch/gilbreth/ahmedb/cache`. Downloads (`uv/`), temporary builds (`tmp/`),
Python downloads (`python/`, disabled during setup), and environments (`runtime/`)
stay there. Home-directory caches are rejected. Existing downloaded packages in
home are neither copied nor deleted. Scratch may be purged by the cluster; keep
lockfiles and source manifests in Git so the runtime can be recreated.
The project also sets uv's default cache to this scratch path for direct uv
commands; `UV_CACHE_DIR` can override it when moving to another machine.

Setup leaves the current `.venv` and running services untouched. It creates a
versioned runtime identified by the lockfiles, manifests, preserved source
patches, and setup script. It uses `uv sync --locked` for both environments:

- Python 3.8.5 with the existing root `uv.lock` scientific versions, including
  PyTorch 2.0.1 (CUDA 11.7 build) and CuPy 12.0.0.
- A separate, locked CUDA 12.6 component environment for CuPy. Keeping these
  NVIDIA wheels separate avoids overwriting PyTorch's CUDA 11 wheel headers.

The CUDA component versions match the working Gilbreth CUDA 12.6.0 toolkit.
The setup assembles its headers and libraries into a toolkit layout for CuPy 12.
Backend and worker launchers load its `CUDA_PATH`, `CUDA_HOME`, library path and
scratch kernel cache automatically. No CUDA module is required for this runtime.
Both `CUPY_CACHE_DIR` and the NVIDIA driver's separate `CUDA_CACHE_PATH` point to
scratch; neither compiled-kernel cache defaults to home.
Startup executes real PyTorch and CuPy kernels and a CuPy linear solve on every
visible GPU before accepting compute work. Failure prevents combined API/worker
startup rather than waiting for a fit to fail.

The four formerly adjacent scientific source checkouts are fetched at the
revisions in `runtime/scientific-sources.json`, with the exact recorded legacy
patches applied and checksum-verified. Their dependencies come from the root
lockfile. Existing adjacent working trees are preserved and are not imported by
the new runtime. The unrelated editable `predictor-corrector` distribution is
not required by the application's scientific workflow.

After validation, finish or stop accepted work in the UI and stop the old
backend. Then activate and launch:

```bash
./setup-runtime.sh --activate
./start-backend.sh
```

Activation atomically repoints an existing `.venv` symlink; it preserves its old
target and prints that path for rollback. It refuses to replace a real directory.
The frontend build, database, recordings, models, and saved samples are unchanged.
Do not use `uv sync` directly against the old shared `.venv`.

For direct diagnostic commands, load the same runtime environment first:

```bash
source .venv/runtime.env
uv run --no-sync python -m backend.gpu
```

On another machine, install/provide Python 3.8.5 and use an appropriate filesystem
cache path; `AUDITORY_BOOTSTRAP_PYTHON=/path/to/python3.8 ./setup-runtime.sh
--cache-dir /scratch/path` selects both explicitly. The interpreter must report
3.8.5. The source manifests and locks target Linux x86-64.

The NVIDIA driver and an allocated GPU remain host requirements; they cannot be
installed inside uv. CUDA 12.6 requires a compatible driver (the tested Gilbreth
driver supports it). Host C/C++ build tools are also required for locked source
packages such as KenLM. GPU models, drivers, native build tools and model/data
artifacts must be recorded when assessing scientific parity; package locking
does not promise bitwise equality across different hardware or RNG states.
