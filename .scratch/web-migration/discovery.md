# Web architecture migration: discovery and decision record

Status: discovery; no implementation authorized.

## Confirmed constraints

User-selected: React + TypeScript + Vite frontend; Python + FastAPI backend;
uv for Python environments; one monorepo with frontend/ and backend/;
frontend and backend independently deployable. Preserve existing functionality
and minimize unnecessary rewrites. Document decisions before implementation.

These choices do not yet settle compute placement, deployment hosts, user
identity, ownership, persistence, or runtime compatibility.

## Observed architecture

- streamlit_app.py initializes a session-state Frontend and adjusts native CUDA
  library loading using the active environment's lib directory.
- auditory_cortex/optimal_stimulus/frontend.py implements experiment setup,
  uploaded or existing recording selection, encoding-model fitting, ranked
  channel correlations, stimulus generation, and ZIP download.
- Active UI setup restricts dataset/model/layer to ucdavisBAK/deepspeech2/2;
  TIMIT and mVocs are selectable. Broader research-package capabilities are
  distinct from the UI's current scope.
- generator.py supplies reusable StimGenerator and Sampler implementations;
  encoding, data assembly, datasets, feature extraction, and diffusion are
  existing Python scientific logic.
- factory.py couples generator construction to Streamlit caching. The cached
  generator is mutable: fitting changes its recording and channel state.
- UI generation writes fixed temporary JSON config names, starts local Python
  subprocesses per selected target, and waits without checking return codes.
  Multi-node Schedular code exists but is commented out in the active UI path.
- Frontend initialization deletes an existing date-named output directory.
  Upload filenames, temporary configs, and fitted model identities require
  isolation before concurrent use. Uploaded recordings use internal session ID 0.
- Python metadata and uv.lock currently restrict Python to >=3.8.5,<3.9.
  Dependencies include scientific, notebook, UI, API, and CUDA packages together.
  Existing uv configuration is Linux x86_64 only.
- Package initialization loads site-specific paths and overrides results_dir
  with an absolute /depot path. Deployment is currently tied to its environment.

Evidence: streamlit_app.py; auditory_cortex/optimal_stimulus/{frontend,factory,
generator,sampling_worker,schedular}.py; auditory_cortex/__init__.py;
auditory_cortex/config.yml; pyproject.toml; uv.lock.

## Design tree: unresolved

First frontier:
- Preservation scope: active web workflow versus broader research workflows.
- Audience and trust boundary: operator, lab, or public service.
- Operational context: live recording loop versus offline analysis; latency needs.

Downstream decisions:
- Hosting and GPU placement; Slurm integration versus dedicated workers.
- Authentication, authorization, ownership, and concurrent use.
- Durable experiment/fit/generation identities and resource lifecycle.
- API operations, artifact transfer, job progress, failure, cancellation, retries.
- Metadata storage, file storage, retention, backup, and provenance.
- API/compute environment separation and Python/CUDA upgrade strategy.
- Migration sequence, scientific parity checks, rollout and rollback criteria.

## Candidate boundary (not accepted)

React owns presentation and draft form state. FastAPI owns validation, resource
identity, access checks, and job submission/status. Compute workers call existing
Python scientific code. Files and fitted models are referenced by stable IDs,
not browser-supplied filesystem paths. Fitting and generation may need durable
jobs independent of browser connections; execution mechanism remains open.

## Verification and limits

Static source inspection only. No model execution, environment installation,
GPU availability check, dependency resolution, or numerical parity verification.
No implementation changes have been made.
