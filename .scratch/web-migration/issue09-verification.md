# Issue 09 verification

Approved seam: authenticated FastAPI HTTP with real SQLite/workflow services and
controlled external compute, plus native GPU verification. Existing `.venv` and
configured cache were reused without installing/upgrading Python dependencies.

- First recovery acceptance test failed with HTTP 404, then passed after session
  metadata, history, work recovery, and Stop/Close were added.
- Eleven lifecycle tests pass: API restart/reconnect; queued cancellation and closed
  session rejection; running cancellation retaining reservations until exit;
  terminal-success immutability; completed downloads after closure/restart;
  idempotent new retry attempts and source association; retrying only unfinished
  batch targets; scoped/authenticated close; actual owned child termination;
  cancelled-batch completed-target preservation; and inherited-lock protection
  against premature reallocation while a child survives its owner.
- Final complete backend suite: 69 tests pass, including review regressions for
  independent-attempt identity reuse and interrupted batches with all targets saved.
- Standalone archive reads closed-session results with zero GPUs and no scientific
  imports. Capability-gated frontend testing also confirms fitting/generation work
  without requesting recovery endpoints when the backend does not advertise them.
- Pyright and TypeScript checks pass; frontend production and backend archive
  builds pass. New metadata migrations preserve prior workflow fixtures.
- Chromium passes reload during accepted fitting without another fit submission,
  retry of a failed generation, Disconnect/relogin during that retry, resumed
  generation status/download, Stop/Close of a running batch with one saved target,
  cancelled-batch history/download after reload, and mobile layout without page
  errors. Active saved-job views poll to completion using job IDs; delayed replies
  cannot replace another session's view. Screenshots remain under configured cache `auditory-web-checks/issue09-*`.
- Real GPU verification: a fit through the new owned subprocess matches all 18
  baseline correlations exactly, maximum absolute delta 0.0 (149.009 seconds).
  A second real fit was stopped during the fitting phase: child exited -15,
  capacity returned, session closed, and the earlier completed fit remained
  available. Evidence: [issue09-native.json](issue09-native.json); reproduction
  probe: [issue09_native_probe.py](issue09_native_probe.py).

Cancellation wins only before terminal publication under the SQLite transaction.
A completed job cannot be undone by cancellation. Running work stays reserved
while stopping; inherited allocator ownership prevents a new allocator from
releasing/reusing devices before surviving compute children exit. Restarted
running attempts are interrupted for manual retry, with completed batch targets
preserved; accepted queued commands remain durable. No automatic scientific retry,
checkpoint continuation, backups, or multi-user scope was introduced.

## Standards

Reviewed implementation `4b3cc3f` against fixed base
`0bfc5580eda117cbf4f1c18351a281f48a0f4066` in a fresh independent agent.

- P1: saved-job polling supplied resource IDs to a job-ID endpoint. Fixed with an
  explicit polling identity; the browser regression observes an active saved
  generation becoming downloadable and checks recovery after reconnect.
- P2: a retry could relabel an independent existing command with identical inputs.
  Fixed with durable public-command/source ownership before enqueueing, a private
  enqueue identity, ownership guards, and atomic attempt provenance. API regression
  rejects reuse without altering either independent attempt.

Both were corrected in `60e0c0e`; follow-up review confirms resolution. Session
selection guards also prevent delayed replies from replacing another view.

## Spec

P2: a batch interrupted after all target commits had no unfinished targets, but
Retry attempted an empty submission. Fixed with an actionable 409, server-provided
retry eligibility, and a hidden empty retry action. A persisted crash-boundary
fixture verifies interruption, retained downloads, and the 409 response. Follow-up
review and the focused test confirm resolution. No scope creep was found.

Final deployment compatibility and completed-result guidance were verified in
`0ea9f06`. Standards: 2 findings resolved, 0 outstanding. Spec: 1 finding resolved,
0 outstanding.
