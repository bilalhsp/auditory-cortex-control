# 01: Resolve implementation gates and capture parity baseline

**What to build:** Resolve the specification’s open implementation choices and establish a reproducible legacy baseline so subsequent slices have explicit contracts and measurable acceptance targets.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

Type: research

- [ ] Record the preserved web feature inventory and whether broader scientific capabilities must be exposed; state CLI/notebook and artifact compatibility commitments.
- [ ] Obtain and record decisions on audience, concurrent use, authentication/access, hosting and compute placement, metadata/artifact storage, retention, and backup. Do not silently choose unresolved requirements.
- [ ] Define job durability, reconnect/restart behavior, cancellation, retries, duplicate submission, partial results, and failure semantics; explicitly exclude declined capabilities.
- [ ] Define available GPU semantics, external workload accounting, reservation behavior, contention handling, target scheduling, and behavior when targets are fewer than requested GPUs.
- [ ] Validate a target Python/CUDA/dependency arrangement using uv and scientific import/worker smoke checks; decide whether API and compute require separate environments and preserve the reproducible legacy runtime.
- [ ] Obtain a replacement primary test boundary: the earlier browser-led proposal was rejected. Record the selected boundary and fixtures without treating that rejected proposal as accepted. Boundary accepted: FastAPI HTTP contract with real workflow services and controlled compute, plus real-GPU parity; representative fixtures remain outstanding.
- [ ] Capture representative existing/uploaded recordings, effective settings, correlations, and audio outputs in isolated storage; establish numerical tolerances from measured repeatability.
- [ ] Record initialization, fitting, generation and queue latency targets, production acceptance criteria, pilot conditions, and rollback window; record substantive architectural trade-offs as ADRs when appropriate.

## Progress

User-confirmed decisions are recorded in the web migration decision record and
the session-lifecycle ADR. Single operator, browser-disconnect survival, explicit
stop with GPU release, retained downloads, previous-session history, and the
replacement primary testing boundary are accepted. Remaining unchecked criteria
are not complete; dependent issues remain blocked. No fitting or generation
baseline has been captured yet.
