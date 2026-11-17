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
  results. The retention limit and deletion policy are not yet decided.
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

## Open implementation gates

- Scope beyond the active UI and exact legacy import/artifact compatibility.
- Deployment host and network access controls, including scheduler constraints.
- Metadata/artifact storage, backup/restore, retention, and explicit deletion.
- API/worker restart recovery; unexpected browser loss is already settled.
- Retry, duplicate submission, cancellation races, and partial-output policy.
- Meaning of available GPU, external workloads, reservation and queue policy.
- Python/CUDA target and whether API and scientific compute share an environment.
- Representative parity fixtures, measured tolerances, latency targets, pilot
  criteria, and rollback window.

Do not treat the next implementation ticket as unblocked until ticket 01's
remaining acceptance criteria are satisfied.
