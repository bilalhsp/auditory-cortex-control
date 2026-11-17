# 11: Verify production parity and cut over

**What to build:** Validate the complete replacement in its intended deployment, pilot it with operators, and perform an approved cutover with a tested rollback path.

**Blocked by:** 06: Fit an uploaded recording, 08: Generate batches using the selected GPU count, 09: Recover and manage long-running work, 10: Enforce deployment access and data lifecycle policies.

**Status:** claimed

- [ ] Run the agreed acceptance workflow for both stimulus families and recording sources, fitting/correlations, both objectives, repeated generation, selected GPU counts, and scoped downloads.
- [ ] Compare scientific results against preserved baselines at measured tolerances and verify manifest completeness, audio format/duration, and configuration provenance.
- [ ] Demonstrate independent frontend/backend builds and release/configuration changes, API compatibility, configured origins/credentials, data/model access, and separate API/compute readiness.
- [ ] Validate contention/isolation, worker failures, chosen recovery/access guarantees, and operation-correlated diagnostics in the intended environment.
- [ ] Measure initialization, fit, generation, queue delay, and resource use against 01 targets; record unresolved failures rather than claim readiness.
- [ ] Provide and rehearse rollback using retained legacy runtime/artifacts; pilot with intended operators and obtain release sign-off before production cutover.
- [ ] Archive Streamlit only after acceptance and the agreed rollback window; retain scoped CLI/notebook and scientific artifact compatibility.
