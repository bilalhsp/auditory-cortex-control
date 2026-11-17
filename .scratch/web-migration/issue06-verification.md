# Issue 06 verification

The approved boundary is authenticated FastAPI HTTP with real application/SQLite
services and controlled external compute, plus separate real GPU parity runs.
The existing `.venv` and configured cache were used without dependency changes.

- First acceptance test failed with HTTP 409 (existing recordings only), then
  passed after uploads entered the shared workflow.
- Uploaded-fit acceptance tests pass: upload → fit → ranked correlations;
  same-named replacements with two queued fits and isolated session-zero models;
  invalid/missing/modified/foreign upload rejection; model loading/fitting failures
  without successful or ready output and without private error content.
- Final full backend suite: 45 tests pass after the review correction, including all existing-recording regressions.
- Backend Pyright: 0 errors; React TypeScript and Vite production build pass.
- Independently built backend archive, launched outside the repository with
  explicit configuration, validates a real upload, queues its fit, and reads the
  persisted job. The packaged worker CLI also starts outside the repository.
- Chromium with real API/SQLite and controlled worker passes uploaded fit queued,
  running, success, sorted signed/decimal channel IDs, unavailable values, failure,
  same-named replacement invalidation, and desktop/mobile layout, with no page errors.
  Screenshots remain under configured cache `auditory-web-checks/issue06-*`.
- Two separate real MAT uploads named `same.mat` were fitted through HTTP and the
  scientific worker on CUDA, using seed 42, TIMIT/deepspeech2 layer 2, 50-ms bins,
  200-ms lag, and unchanged three-fold cross-validation. Both fit all 18 channels
  with maximum absolute delta 0.0 against the issue 01 existing/upload baseline,
  satisfying exact equality in the unchanged runtime. Best channel: 12.03,
  correlation 0.2760693717950025. Native session-zero configuration remains private
  inside distinct fit UUID workspaces. First artifacts remain unchanged after the
  second fit. Evidence: [issue06-parity.json](issue06-parity.json).
- Scientific application code and algorithms are unchanged in this issue. The
  worker invokes the existing full-path reader for uploads and reuses all fitting,
  result sorting, persistence, provenance, and artifact publication from issue 05.
- Existing missing-SoX warning remains nonblocking for these successful fits.

Large recordings/model files and runtime diagnostics remain in the configured
scientific cache, outside the source repository. No new environment was created.

## Standards

Fresh independent review of `792f1c0` against fixed base
`7f62696733ac43ef0ab17237558572fe2783ffc8` found no documented violations or
substantive smell findings. Follow-up review of `51e2f08` also found none.

## Spec

One P2 finding: resolving uploaded-fit fallback configuration outside error
handling could produce a generic HTTP 500. Fixed in `51e2f08` by translating
configuration failures into an actionable `CatalogUnavailable` HTTP 503. A new
standalone-archive HTTP regression reproduced 500 before the fix and passes with
503 afterward; public errors contain no filesystem paths. The reviewer confirmed
resolution with no remaining spec findings. Fifteen targeted uploaded-fit/catalog
tests, Pyright, and the rebuilt backend artifact pass after the correction.

Review totals: Standards 0 findings; Spec 1 P2 finding resolved, 0 remaining.
