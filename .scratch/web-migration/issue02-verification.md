# Issue 02 verification

- Complete API suite: 11 unittest tests pass at the approved HTTP boundary,
  using real SQLite and controlled GPU discovery. Covers authentication,
  persistence/restart, supported settings, invalid inputs, zero/one/multiple
  devices, stale capacity, CORS, and compute failures with readable metadata.
- Python static check: Pyright reports zero errors and warnings using the
  existing `.venv`. TypeScript strict checking and the Vite production build pass.
- Independent backend artifact: standard-library archive builds using uv with
  `--no-sync`; runs from outside the repository with explicit database settings;
  authenticated capability query detects the actual single local GPU.
- Real browser smoke: React and FastAPI run as separate processes/origins, with
  real API/storage and controlled compute. Verified maximum default, preserved
  explicit selection on refresh, saving, upstream invalidation, stale-capacity
  rejection, zero-GPU behavior, sign-out, and empty browser credential storage.
- Desktop and 390-pixel mobile layouts inspected; mobile has no horizontal
  overflow. Browser reported no page JavaScript errors. Headless cluster
  compatibility flags were required; browser tooling lives only in cache.
- Review correction verified with an intentionally delayed real save: sign-out
  stays disabled during a pending request, and login clears previous result state.
- Existing scientific code, Streamlit launcher, root Python project/lockfile,
  and shared Python environment are unchanged. No Gradio or new Python
  environment was added. New frontend dependencies are separately locked.

Backend wheel packaging was attempted but the existing setuptools/packaging
versions conflict. No upgrade was performed. The documented independent source
archive build avoids that tooling problem and was successfully executed.

This slice configures experiments only. Saving does not allocate GPUs or execute
jobs. Production hosting still requires operator credentials, HTTPS termination,
explicit allowed origins, and proxy request-rate limits as documented in the
backend README; no public deployment was performed.
