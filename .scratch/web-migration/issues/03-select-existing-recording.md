# 03: Select an existing recording

**What to build:** Browse compatible stored recording sessions and attach one to a configured experiment through the web interface without loading a mutable shared generator merely to list metadata.

**Blocked by:** 02: Configure an experiment and select backend GPUs.

**Status:** resolved

- [x] List compatible recordings for the experiment context with stable opaque identities and readable names, including useful empty/error states.
- [x] Selecting a recording attaches the intended dataset session to the experiment; incompatible or missing recordings receive actionable errors.
- [x] Browser/API contracts do not accept arbitrary server filesystem paths; apply the access and visibility policy decided in 01 from the first recording endpoint.
- [x] Independent workflows maintain isolated selections; a changed recording invalidates downstream fit/generation selections without deleting existing artifacts.
- [x] Use configurable data locations and existing dataset/metadata functionality while preserving legacy behavior.
- [x] Behavioral verification covers selection, absent/incompatible recordings, context changes, and workflow isolation through the approved test boundary.

## Verification

The complete 21-test HTTP suite, Python/TypeScript checks, frontend build, real
browser workflow, and independent backend archive checks pass. All three actual
configured recordings are discoverable for both stimulus families without
loading Torch or scientific models. The issue 03 verification record captures
details and the cyclic-symlink regression fixed during review.

Selections are isolated and revisioned; future fit/generation resources must pin
the selected revision and resolve native session identities on the server. No
existing research data or artifacts were modified or deleted. Issue 05 is now
unblocked; issue 04 remains ready independently.
