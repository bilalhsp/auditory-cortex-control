# Web migration

## Notes

Implementation tickets live in [issues/](issues/); the approved requirements are
in [spec.md](spec.md). Issues 01–05 are resolved.

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
  [verification](issue05-verification.md). Issue 06 is the next ready issue.

## Fog

Later slices implement uploaded fitting (06), single-target generation/downloads
(07), multi-GPU batches (08), history/Stop/reconnection (09), lifecycle hardening
(10), and final parity/cutover (11). No new user decisions block issue 06.
