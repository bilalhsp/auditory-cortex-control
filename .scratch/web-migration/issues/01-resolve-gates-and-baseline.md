# 01: Resolve implementation gates and capture parity baseline

**What to build:** Resolve the specification’s open implementation choices and establish a reproducible legacy baseline so subsequent slices have explicit contracts and measurable acceptance targets.

**Blocked by:** None (can start immediately).

**Status:** claimed

Type: research

- [x] Record the preserved web feature inventory and whether broader scientific capabilities must be exposed; state CLI/notebook and artifact compatibility commitments.
- [x] Obtain and record decisions on audience, concurrent use, authentication/access, hosting and compute placement, metadata/artifact storage, retention, and backup. Do not silently choose unresolved requirements.
- [x] Define job durability, reconnect/restart behavior, cancellation, retries, duplicate submission, partial results, and failure semantics; explicitly exclude declined capabilities.
- [x] Define available GPU semantics, external workload accounting, reservation behavior, contention handling, target scheduling, and behavior when targets are fewer than requested GPUs.
- [ ] Validate a target Python/CUDA/dependency arrangement using uv and scientific import/worker smoke checks; decide whether API and compute require separate environments and preserve the reproducible legacy runtime.
- [x] Obtain a replacement primary test boundary: the earlier browser-led proposal was rejected. Record the selected boundary and fixtures without treating that rejected proposal as accepted. Boundary accepted: FastAPI HTTP contract with real workflow services and controlled compute, plus real-GPU parity. A hashed representative recording and measured fit results cover existing/uploaded source fixtures.
- [ ] Capture representative existing/uploaded recordings, effective settings, correlations, and audio outputs in isolated storage; establish numerical tolerances from measured repeatability.
- [x] Record initialization, fitting, generation and queue latency targets, production acceptance criteria, pilot conditions, and rollback window; record substantive architectural trade-offs as ADRs when appropriate. User declined latency targets; retain the legacy runtime until explicit retirement approval rather than impose an unapproved calendar window.

## Progress

Accepted decisions are recorded in the migration decision record and ADRs:
single authenticated operator at a public web URL, SQLite metadata, filesystem
artifacts, browser-disconnect survival, explicit stop, history/downloads, manual
retry after restart, dedicated backend GPUs, existing environment/cache, and
FastAPI HTTP tests plus real-GPU parity. Backups and latency thresholds are
outside initial scope.

Real CUDA, CuPy, HTTP, parser, and lockfile checks have passed. Three seeded fits
for 18 units produced identical correlations across repetition and existing/
upload source paths. Real sample generation and audio repeatability are still
running; this ticket is not complete until those results are recorded.
