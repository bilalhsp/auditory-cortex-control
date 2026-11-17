# 04: Upload and validate a recording

**What to build:** Upload a recording from the web interface, receive validation feedback, and attach its unique identity to an experiment without colliding with other uploads.

**Blocked by:** 02: Configure an experiment and select backend GPUs.

**Status:** resolved

- [x] Upload returns a stable recording identity and clear pending/valid/invalid feedback according to the validation contract selected in 01.
- [x] Validate format, configured size limits, parseability, and dataset/stimulus compatibility; reject unsupported content without exposing internal paths.
- [x] Assign server-controlled isolated storage names; duplicate client filenames and unsafe path components cannot overwrite or escape recording storage.
- [x] Browser state references recording identities rather than physical paths; apply the access policy selected in 01.
- [x] Changing or replacing the selected upload invalidates stale dependent state while preserving prior artifacts under the selected lifecycle policy.
- [x] Verify valid, malformed, oversized, incompatible, and same-named uploads plus isolation through the approved boundary; use existing scientific parsers without accepting uploaded checkpoints as trusted objects.

## Verification

All 32 API tests, Python/TypeScript checks, builds, real-recording and standalone
archive checks, and browser upload workflows pass. The issue 04 verification
record covers cancellation safety and legacy-path compatibility corrections
identified during review. Validation is synchronous and bounded; UI pending
state ends with a valid attachment or structured invalid/error feedback.

Recording revisions invalidate later analysis context without deleting prior
uploads. Fitting/generation remain later slices. Issue 05 remains the next ready
issue; issue 06 also requires completion of 05.
