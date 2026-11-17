# 04: Upload and validate a recording

**What to build:** Upload a recording from the web interface, receive validation feedback, and attach its unique identity to an experiment without colliding with other uploads.

**Blocked by:** 02: Configure an experiment and select backend GPUs.

**Status:** claimed

- [ ] Upload returns a stable recording identity and clear pending/valid/invalid feedback according to the validation contract selected in 01.
- [ ] Validate format, configured size limits, parseability, and dataset/stimulus compatibility; reject unsupported content without exposing internal paths.
- [ ] Assign server-controlled isolated storage names; duplicate client filenames and unsafe path components cannot overwrite or escape recording storage.
- [ ] Browser state references recording identities rather than physical paths; apply the access policy selected in 01.
- [ ] Changing or replacing the selected upload invalidates stale dependent state while preserving prior artifacts under the selected lifecycle policy.
- [ ] Verify valid, malformed, oversized, incompatible, and same-named uploads plus isolation through the approved boundary; use existing scientific parsers without accepting uploaded checkpoints as trusted objects.
