# 01: Resolve implementation gates and capture parity baseline

**What to build:** Resolve the specification’s open implementation choices and establish a reproducible legacy baseline so subsequent slices have explicit contracts and measurable acceptance targets.

**Blocked by:** None (can start immediately).

**Status:** resolved

Type: research

- [x] Record the preserved web feature inventory and whether broader scientific capabilities must be exposed; state CLI/notebook and artifact compatibility commitments.
- [x] Obtain and record decisions on audience, concurrent use, authentication/access, hosting and compute placement, metadata/artifact storage, retention, and backup. Do not silently choose unresolved requirements.
- [x] Define job durability, reconnect/restart behavior, cancellation, retries, duplicate submission, partial results, and failure semantics; explicitly exclude declined capabilities.
- [x] Define available GPU semantics, external workload accounting, reservation behavior, contention handling, target scheduling, and behavior when targets are fewer than requested GPUs.
- [x] Validate a target Python/CUDA/dependency arrangement using uv and scientific import/worker smoke checks; decide whether API and compute require separate environments and preserve the reproducible legacy runtime. Use the existing environment for separate API/worker processes; CUDA, CuPy, FastAPI, worker CLI, real fits, and real generation pass with recorded local dependency provenance.
- [x] Obtain a replacement primary test boundary: the earlier browser-led proposal was rejected. Record the selected boundary and fixtures without treating that rejected proposal as accepted. Boundary accepted: FastAPI HTTP contract with real workflow services and controlled compute, plus real-GPU parity. A hashed representative recording and measured fit results cover existing/uploaded source fixtures.
- [x] Capture representative existing/uploaded recordings, effective settings, correlations, and audio outputs in isolated storage; establish numerical tolerances from measured repeatability. Three fits match exactly across repetition/source paths; stretch and one-hot each produce byte-identical repeated WAVs at seed 42 in the unchanged runtime.
- [x] Record initialization, fitting, generation and queue latency targets, production acceptance criteria, pilot conditions, and rollback window; record substantive architectural trade-offs as ADRs when appropriate. User declined latency targets; retain the legacy runtime until explicit retirement approval rather than impose an unapproved calendar window.

## Progress

Accepted decisions are recorded in the migration decision record and ADRs:
single authenticated operator at a public web URL, SQLite metadata, filesystem
artifacts, browser-disconnect survival, explicit stop, history/downloads, manual
retry after restart, dedicated backend GPUs, existing environment/cache, and
FastAPI HTTP tests plus real-GPU parity. Backups and latency thresholds are
outside initial scope.

Real CUDA, CuPy, HTTP, parser, lockfile, and worker CLI checks passed. Three seeded
fits for 18 units produced identical correlations across repetition and existing/
upload source paths. Four real generated WAVs (two per objective) pass finite
value, sample-rate, and PCM-format checks. Repeated outputs are byte-identical
within each objective. The unchanged-stack baseline tolerance is exact equality;
different hardware/dependencies require fresh measurements.

All scientific artifacts remain under the configured cache, and the existing
environment and scientific application code were not modified. Source manifests,
measured summaries, reproduction probes, and limitations are recorded alongside
this ticket. Nominal one-second samples contain 15,392 frames at 16 kHz; this
native behavior is documented rather than silently changed. Issue 02 is now
unblocked. This resolves discovery/baseline work, not the web implementation.

## Acceptance update (issue 07)

2026-11-17: Fresh unchanged legacy generation also differs from the historical
WAVs. The user explicitly approved controlled legacy/web comparison with identical
recorded initialization and RNG state as the generation migration gate, retaining
the historical evidence and documenting fresh-process variability. This does not
relax fitting parity or introduce a numeric audio tolerance. See issue 07 comments
and its controlled comparison evidence.
