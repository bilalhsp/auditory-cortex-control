# Issue 08 verification

Approved seam: authenticated FastAPI HTTP with real SQLite/services and controlled
external compute, plus separate real GPU execution. Existing `.venv` and configured
scientific cache were reused without environment creation or upgrades.

- First batch acceptance test failed with 404, then passed after batch submission
  and transactional reservations were implemented.
- Seven batch acceptance tests cover atomic reservation/idempotency, actual
  competing HTTP requests, two concurrent device slots with four targets, no
  same-device overlap, partial target failure with valid but withheld failed output,
  exact ZIP membership, release after success/failure, fewer targets than reserved
  GPUs, changed count/visibility before launch, invalid/duplicate targets, auth,
  and restart preserving completed targets while interrupting unfinished ones.
- The real subprocess launcher was exercised with controlled compute children
  under a runtime visibility list of opaque device IDs. Children received only
  their respective reserved device, not guessed remote/host identifiers.
- Complete backend suite: 58 tests pass. Existing standalone generation, fit,
  upload, recording, and authentication regressions pass after metadata migration.
- Standalone backend archive, launched from `/tmp` against an isolated metadata
  fixture, reserves a batch and reads its job without importing scientific models.
- Python Pyright: 0 errors; React TypeScript and production build pass. Backend
  source artifact builds successfully.
- Chromium with real API/SQLite and controlled compute passes initial maximum
  batch GPU choice, smaller choice retention, multiple channel/objective targets,
  running/partial outcomes, authenticated partial ZIP download, stale capacity
  rejection/explicit reselection, and mobile layout without page errors.
  Screenshots/ZIP remain in configured cache `auditory-web-checks/issue08-*`.
- Real single-GPU batch: stretch and one-hot targets, seed 42, nominal one second,
  one clip each. Both subprocesses exited zero, generated validated 15,392-frame
  WAVs, appeared in the batch ZIP with its manifest, and released capacity. Target
  timestamps confirm sequential waves on one GPU. Total time 570.180 seconds;
  peak observed memory 5,497 MiB (sampled via nvidia-smi). The user declined latency
  limits, so measurements are reported without a performance pass threshold.
  Evidence: [issue08-benchmark.json](issue08-benchmark.json).
- This server exposes one physical GPU. Physical multi-GPU throughput/memory was
  not measured. Multi-GPU concurrency, allocation, binding, visibility, and failure
  behavior are verified with controlled external compute; this limitation is
  explicit rather than represented as a hardware benchmark.

No scientific sampling algorithm was rewritten. Batch targets reuse issue 07's
scientific adapter and validation. API/worker upgrade instructions require a
coordinated stop/start so pre-batch workers do not consume grouped target rows.
Session history/Stop/Close remain issue 09.

## Standards

Fresh review of `cad7de2` against fixed base
`45b815a30186b96e989fc6d795becac80caf8324` found no documented violations. One
nonblocking duplication/recovery finding in frontend polling/download handling was
addressed in `aa38891` using shared helpers. A browser check injects a transient
503, confirms batch polling recovers and clears the error, and rechecks batch and
single-target downloads. Production build/typecheck pass. Follow-up Standards
review confirms no outstanding finding.

## Spec

Fresh independent review found no concrete spec violation or scope creep. The
review confirmed the reservation, queue, visibility, partial-result, and recovery
contracts. Queued batches survive restart consistently with existing queued-job
semantics; running batches become interrupted with completed targets preserved.
The physical multi-GPU benchmark limitation remains explicitly disclosed.

Review totals: Standards 1 heuristic addressed, 0 outstanding; Spec 0 findings.
