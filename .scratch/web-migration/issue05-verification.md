# Issue 05 verification

The approved seam remains authenticated FastAPI HTTP with real SQLite and a
controlled external compute runner; scientific parity runs separately on the
existing GPU environment. No environment was created or upgraded.

- Complete backend suite: 40 tests passed, including eight fit workflow tests.
  Coverage includes immutable submissions, idempotency/conflicting request IDs,
  stale selections, missing files, authentication, absent GPUs, running status,
  pinned recording bytes, finite/nonfinite ranking, missing artifacts, sanitized
  failures, distinct artifact retention, worker exclusivity, and interruption
  after terminating a real worker process.
- Python backend typecheck: 0 errors; changed scientific modules compile.
- React TypeScript check and Vite production build pass.
- Chromium using the real API/SQLite and a controlled external worker passes
  queued/running/success/failure workflows, signed/noninteger channel ranking,
  explicit unavailable values, and desktop/mobile layout without page errors.
  Screenshots remain in the configured cache under `auditory-web-checks/issue05-*`.
- Two independently submitted real CUDA fits of existing session 0 (TIMIT,
  deepspeech2 layer 2, 50-ms bins, 200-ms lag, seed 42) both match all 18 issue 01
  correlations exactly: maximum absolute delta 0.0, satisfying the issue 01
  unchanged-runtime exact-equality requirement.
  Best channel remains 12.03, correlation 0.2760693717950025. Their UUID workspaces
  and manifests are distinct; the first manifest remains unchanged after the
  second fit. Evidence: [issue05-parity.json](issue05-parity.json). Large scientific
  artifacts and diagnostics remain in the repository's configured cache.
- Legacy `scripts/daps_sampling.py --help` succeeds with the existing dependency
  checkouts. Existing fit/default dataset/save arguments retain their defaults;
  explicit web-only pinned-input and output-path arguments are optional. This is
  an import/argument compatibility check, not a full new generation parity run.
- Independently built backend archive, launched from `/tmp` with explicit
  configuration, discovers real recordings, selects one, queues a fit, and reads
  its persisted job. The packaged worker CLI also launches outside the repository.
- Existing environment emits its pre-existing missing-SoX warning on scientific
  import. Both real fitting runs succeed; no package changes were made.

The worker is intentionally single-host/single-owner and serializes fitting on
one GPU. API restart does not own compute recovery; worker restart interrupts
running attempts and preserves queued work. Full session history, explicit
Stop/Close, and multi-GPU generation belong to issues 08/09. Uploaded fitting is
issue 06; this slice accepts existing recordings only.

## Independent review

Reviewed implementation commit `af311b0` against fixed base
`6f24374e642ba9af84c9291342a99df5a0db565e` in two fresh parallel agents.

Standards: no substantive violations or actionable smell findings. The ticket's
missing `Type: task` metadata was added.

Spec: no actionable issue 05 violations or scope creep. Verification wording was
corrected to state the agreed unchanged-runtime exact-equality requirement; the
measured delta was already zero in both real runs.

Final review changes affect documentation only. `git diff --check` passes.
