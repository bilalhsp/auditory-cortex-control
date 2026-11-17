# Issue 09 verification

Approved seam: authenticated FastAPI HTTP with real SQLite/workflow services and
controlled external compute, plus native GPU verification. Existing `.venv` and
configured cache were reused without installing/upgrading Python dependencies.

- First recovery acceptance test failed with HTTP 404, then passed after session
  metadata, history, work recovery, and Stop/Close were added.
- Nine lifecycle tests pass: API restart/reconnect; queued cancellation and closed
  session rejection; running cancellation retaining reservations until exit;
  terminal-success immutability; completed downloads after closure/restart;
  idempotent new retry attempts and source association; retrying only unfinished
  batch targets; scoped/authenticated close; actual owned child termination;
  cancelled-batch completed-target preservation; and inherited-lock protection
  against premature reallocation while a child survives its owner.
- Complete suite before the final lock-inheritance regression: 66 tests pass.
  The final complete suite is being rerun with that additional regression.
- Pyright and TypeScript checks pass; frontend production and backend archive
  builds pass. New metadata migrations preserve prior workflow fixtures.
- Chromium passes reload during accepted fitting without another fit submission,
  retry of a failed generation, Disconnect/relogin during that retry, resumed
  generation status/download, Stop/Close of a running batch with one saved target,
  cancelled-batch history/download after reload, and mobile layout without page
  errors. Screenshots remain under configured cache `auditory-web-checks/issue09-*`.
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

Independent Standards and Spec reviews will be recorded before resolution.
