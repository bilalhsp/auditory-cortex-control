# 11: Verify production parity and cut over

**What to build:** Validate the complete replacement in its intended deployment, pilot it with operators, and perform an approved cutover with a tested rollback path.

**Blocked by:** 06: Fit an uploaded recording, 08: Generate batches using the selected GPU count, 09: Recover and manage long-running work, 10: Enforce deployment access and data lifecycle policies.

**Status:** claimed

- [x] Run the agreed acceptance workflow for both stimulus families and recording sources, fitting/correlations, both objectives, repeated generation, selected GPU counts, and scoped downloads.
- [x] Compare scientific results against preserved baselines at measured tolerances and verify manifest completeness, audio format/duration, and configuration provenance.
- [x] Demonstrate independent frontend/backend builds and release/configuration changes, API compatibility, configured origins/credentials, data/model access, and separate API/compute readiness.
- [x] Validate contention/isolation, worker failures, chosen recovery/access guarantees, and operation-correlated diagnostics in the intended environment.
- [x] Measure initialization, fit, generation, queue delay, and resource use against 01 targets; record unresolved failures rather than claim readiness.
- [ ] Provide and rehearse rollback using retained legacy runtime/artifacts; pilot with intended operators and obtain release sign-off before production cutover.
- [ ] Archive Streamlit only after acceptance and the agreed rollback window; retain scoped CLI/notebook and scientific artifact compatibility.

## Comments

The operator approved local deployments over URL/port for acceptance. Automated
checks passed using built frontend assets on 127.0.0.1:5173 and a separate API on
127.0.0.1:8765. A second check uses the actual backend archive from outside the
repository, actual mVocs results, and a real served runtime config.js.

All 74 backend tests passed. Both mVocs recording-source fits match the unchanged
legacy reference across all 18 channels exactly; stretch and one-hot native
legacy/web audio are byte-identical under the previously approved controlled
initialization/RNG comparison. Prior TIMIT scientific evidence is retained.
Typechecks, builds, browser recovery/lifecycle checks, archive recovery, and
isolated legacy rollback rehearsal passed. Reviews have zero outstanding findings.

Rollback rehearsal is complete, but the sixth checkbox remains partial because
an operator pilot and explicit release sign-off have not occurred. No production
traffic switch or Streamlit archival was performed. The accepted policy requires
retaining the legacy launcher until retirement is separately approved; no calendar
rollback window or backup implementation is introduced. Public HTTPS/proxy checks
remain necessary before public exposure. See [verification](../issue11-verification.md)
and the [cutover runbook](../../../docs/deployment/web-cutover.md).


## Operator acceptance — 2026-11-18 UTC

The operator explicitly reported the pilot successful and authorized finalizing
the application. Future host/URL changes are deployment configuration work and
are not outstanding application acceptance questions. The exercised local/SSH
web workflow is accepted. Public hosting/TLS routing has not been changed here.

The earlier pending-pilot note is superseded by this approval. Streamlit and
scientific CLI/notebook compatibility remain retained for rollback under the
accepted retirement policy; no separate retirement approval was given. Final
launcher supervision/regression checks and release documentation are in progress.
