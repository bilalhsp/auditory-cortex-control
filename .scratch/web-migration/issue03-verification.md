# Issue 03 verification

Implemented authenticated recording discovery and selection through the existing
FastAPI seam, with real SQLite persistence and real MAT trial metadata in tests.

- Complete suite: 21 passing HTTP tests. Coverage includes stable opaque IDs, stimulus-family filtering, selection across
  restart, per-experiment isolation, idempotent reselection and revision changes,
  absent/incompatible recordings, valid empty catalogs, unavailable storage/config,
  corrupt files, target-only trials, authentication, arbitrary-path rejection,
  external symlink exclusion, and cyclic-symlink handling.
- Real configured data: all three ucdavisBAK recordings are listed for both TIMIT
  and mVocs, with zero unavailable entries. Neither Torch nor the scientific
  package initializer is loaded by discovery.
- Browser check: independent React/API processes with real services and synthetic
  MAT fixtures. Browse, attach/change, context filtering/reset, old-experiment
  isolation, missing/empty/error states, and delayed responses after changing
  context pass. Desktop and 390-pixel mobile layouts were captured; no horizontal
  overflow or page JavaScript errors were observed.
- Python and TypeScript checks and the frontend build pass. The backend archive
  runs outside the repository and discovers the real recordings using explicit
  dataset/catalog configuration.
- Review identified a cyclic-link failure outside the per-file error handler.
  A failing HTTP regression reproduced it; path-resolution failures are now
  normalized and counted per recording without hiding healthy entries.

The existing Python environment, root lockfile, scientific code, Streamlit
launcher, and research datasets/models are unchanged. NumPy/SciPy were already
installed; their versions are now documented as backend runtime requirements.
No scientific model was loaded, no GPU job was started, and no existing research
artifact was deleted.

Changed recording selections increment the saved revision. Actual fit/generation
resources remain later slices and must pin that revision and resolve native
session identities server-side. Public catalog responses never include raw
filesystem paths or accept them as selection inputs.
