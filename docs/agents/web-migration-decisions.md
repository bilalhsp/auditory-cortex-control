# Web migration decisions

This record captures user decisions made after the migration specification was
published. It supplements that specification; unresolved proposals are not
implicitly accepted.

## Accepted

- The application has a single operator. Multi-user collaboration is not an
  initial requirement. This does not itself decide network access controls.
- Accepted computation continues after an unexpected browser disconnect or
  closing the browser.
- An explicit Stop/Close session action terminates running work and releases
  its GPUs. It retains completed generated samples.
- The frontend allows downloading generated samples and browsing saved results
  from previous application sessions. Closing a session does not delete those
  results. Retain saved results until explicit deletion; automatic expiry is
  outside the initial scope.
- The primary test boundary is the FastAPI HTTP contract with real workflow
  services and controlled compute, supplemented by real-GPU scientific parity
  checks. The earlier browser-led primary seam was rejected.
- Backend-server GPU detection, visible capacity, selectable GPU count, and a
  maximum-available default remain confirmed requirements.

## Observable acceptance requirements

- Disconnect the browser during accepted work: computation continues; reconnect
  makes its status and available results accessible.
- Explicitly close an active session: running work stops and assigned GPUs are
  released. Already completed samples remain downloadable.
- Open session history: prior application sessions and their completed results
  can be selected. A recording session and an application session are distinct.
- A worker failure or incomplete output cannot be shown as successful results.
- Closing one session must not remove historical results from other sessions.

## Additional accepted requirements

- Preserve the active Streamlit workflow and existing Python scripts/notebooks;
  do not expand the web scope to other scientific capabilities initially.
- Serve the application through a public web URL with single-operator
  authentication. Do not expose GPU commands or research artifacts anonymously.
- Use SQLite for session/job metadata and the filesystem for recordings,
  fitted models, and generated samples. Backup implementation is outside scope.
- After API/compute restart, mark interrupted work and require manual retry;
  completed results remain accessible. Seamless restart resumption is not required.
- Stop cancels pending tasks and terminates running tasks. Keep fully saved
  samples; manual retries create distinct attempts.
- GPUs are exclusive to the backend. Reserve requested devices within the
  application, reject insufficient capacity without silently reducing it, and
  process excess targets in waves. Fewer targets may leave selected GPUs idle.
- No fitting/generation latency target is required; report measured runtime
  without using performance as a release threshold.
- Continue using the existing `.venv`; do not create or upgrade environments.
  Use the cache directory configured in the repository for scientific caches
  and isolated baseline artifacts. The frontend is entirely React; do not add
  Gradio. Its upstream AudioLDM dependency is intentionally excluded already.

## Implementation contracts derived from those decisions

- Preserve current Python imports and scientific artifact formats during the
  web migration. Keep legacy scripts/notebooks operational.
- Authenticate the single operator for all commands, session history, status,
  uploads, and downloads; production requires HTTPS. Public registration and
  multi-user collaboration are outside scope.
- Application session, recording, fit, generation, and retry attempt have
  distinct identities. New attempts never overwrite prior completed artifacts.
- Duplicate submission of the same command identity returns the existing job;
  an explicit retry creates a new attempt. Disable automatic scientific retries.
- Stop cancels queued work and requests termination of active worker processes;
  release GPU reservations after workers exit. Preserve completed manifest
  entries, exclude incomplete files from downloads, and show partial/cancelled
  status rather than success. Cancellation is idempotent and cannot undo a
  completed result.
- On startup, reconcile unfinished jobs as interrupted before allocating their
  devices to new work; surviving workers must be reconciled or stopped before
  releasing reservations. Do not promise checkpoint-based continuation.
- Backend-visible devices are the upper limit; available count subtracts its
  active reservations. GPUs are dedicated to this backend, so external workload
  scheduling is outside scope. Never infer local capacity from remote hosts.
- Serialize allocation within the single backend allocator. Queue target tasks
  within an accepted batch in waves; reject unavailable batch capacity clearly.
- Session history must survive API restart and explicit session closure.

## Release and verification gates

- Use the accepted FastAPI seam for new workflow tests and actual scientific
  entry points for baseline experiments. The setup API now supports authenticated
  capabilities and persisted experiment configuration; compute workflows remain
  subsequent slices.
- Pilot with the single operator only after workflow parity, authentication,
  artifact isolation, cancellation, recovery, and history acceptance checks pass.
- Retain the legacy launcher/environment and its compatible artifacts until the
  operator explicitly approves retirement. Do not impose an unapproved calendar
  rollback window or latency threshold.
- Dependency smoke checks and measured numerical tolerances must be recorded;
  available checkpoints or successful imports alone are not scientific parity.

Do not treat the next implementation ticket as unblocked until ticket 01's
remaining acceptance criteria are satisfied.
