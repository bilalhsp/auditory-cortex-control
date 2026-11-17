# Web migration

## Notes

Implementation tickets live in [issues/](issues/); the approved requirements are
in [spec.md](spec.md). Issues 01–08 are resolved.

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
  server. See [verification](issue08-verification.md). Issue 09 is next.

## Fog

Later slices implement history/Stop/reconnection (09), lifecycle hardening (10),
and final parity/cutover (11). No new user decisions block issue 09.
