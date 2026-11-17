# 03: Select an existing recording

**What to build:** Browse compatible stored recording sessions and attach one to a configured experiment through the web interface without loading a mutable shared generator merely to list metadata.

**Blocked by:** 02: Configure an experiment and select backend GPUs.

**Status:** claimed

- [ ] List compatible recordings for the experiment context with stable opaque identities and readable names, including useful empty/error states.
- [ ] Selecting a recording attaches the intended dataset session to the experiment; incompatible or missing recordings receive actionable errors.
- [ ] Browser/API contracts do not accept arbitrary server filesystem paths; apply the access and visibility policy decided in 01 from the first recording endpoint.
- [ ] Independent workflows maintain isolated selections; a changed recording invalidates downstream fit/generation selections without deleting existing artifacts.
- [ ] Use configurable data locations and existing dataset/metadata functionality while preserving legacy behavior.
- [ ] Behavioral verification covers selection, absent/incompatible recordings, context changes, and workflow isolation through the approved test boundary.
