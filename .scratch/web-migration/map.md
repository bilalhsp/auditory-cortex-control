# Web migration

## Notes

Implementation tickets live in [issues/](issues/); the approved requirements are
in [spec.md](spec.md). Issues 01–10 are resolved.

## Decisions-so-far

- Preserve the existing Python environment and configured scientific cache.
- Serve one authenticated operator through React and a separate FastAPI API,
  with SQLite metadata and private filesystem scientific artifacts.
- Issue 05 implements a separate, single-owner fit worker. Accepted requests
  pin recording bytes, metadata, experiment settings, revision, and seed.
  One fit runs at a time on one GPU; worker restart interrupts running attempts
  for manual retry while retaining queued requests and earlier results.
- Ranked correlations preserve channel identities and explicit unavailable values.
  Two web-submitted GPU fits match all 18 legacy baseline values exactly.
- [Issue 05](issues/05-fit-existing-recording.md) is complete; see its
  [verification](issue05-verification.md).
- [Issue 06](issues/06-fit-uploaded-recording.md) adds validated uploads to that
  same workflow, preserving native session zero while isolating every fit artifact
  by UUID and checking upload integrity/ownership. Two same-named uploads match
  all 18 baseline correlations exactly and retain distinct artifacts; see
  [verification](issue06-verification.md).
- [Issue 07](issues/07-generate-one-target.md) provides single-target generation
  and authenticated WAV ZIP downloads with preserved artifacts and GPU requests.
  Both objectives match controlled legacy outputs exactly under the user-approved
  initialization/RNG comparison. Fresh legacy processes also vary from historical
  WAVs; retain that evidence instead of claiming cross-process determinism.
  See [verification](issue07-verification.md).
- [Issue 08](issues/08-generate-batches-with-gpu-allocation.md) adds transactional
  full-count reservations, per-device target waves, and partial-result ZIPs.
  Controlled multi-device scheduling/binding tests and a real one-GPU/two-target
  benchmark pass. Physical multi-GPU throughput is not claimed on this one-GPU
  server. See [verification](issue08-verification.md).
- [Issue 09](issues/09-recover-and-manage-jobs.md) adds persistent sessions,
  reconnect/history, explicit Stop/Close, owned child termination, and isolated
  retries. Committed results survive closure and restart; retries keep source
  ownership and skip completed batch targets. Inherited allocator locks prevent
  premature GPU reallocation after owner failure. See
  [verification](issue09-verification.md).

- [Issue 10](issues/10-enforce-access-and-data-lifecycle.md) verifies authentication
  across all API routes and retained downloads, adds no-store API responses, and
  documents private storage/public HTTPS proxy boundaries. Indefinite retention
  remains enforced; backups and multi-user sharing are explicitly excluded.
  All 73 backend tests pass; see [verification](issue10-verification.md).

- [Issue 11](issues/11-verify-parity-and-cut-over.md) automated local acceptance
  passes: all 74 HTTP/workflow tests, independently deployed production artifacts,
  real mVocs fits for both sources (18 correlations exactly equal), both mVocs
  objectives byte-identical under the approved controlled comparison, and isolated
  legacy rollback rehearsal. See [verification](issue11-verification.md) and the
  [release runbook](../../docs/deployment/web-cutover.md). Operator pilot/release
  sign-off remain pending; no production switch or Streamlit retirement occurred.

## Fog

Issue 11 awaits operator pilot and release sign-off. Local URL/port acceptance is
approved and passed. Public HTTPS/proxy validation is required before public
exposure. Streamlit remains available until separately approved retirement.
