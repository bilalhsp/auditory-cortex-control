# 06: Fit an uploaded recording

**What to build:** Fit an uploaded recording using the same web workflow and correlation view while preserving unique recording and model identities.

**Blocked by:** 04: Upload and validate a recording, 05: Fit an existing recording and inspect correlations.

**Status:** claimed

Type: task

- [ ] Validated uploaded recordings can enter the existing fit submission/status/result workflow with the same scientific settings and correlation display.
- [ ] The legacy internal session-zero convention cannot cause uploaded fitted models to overwrite or reload another recording’s fit.
- [ ] Validation failures prevent fitting; parsing, loading, and computation errors remain actionable and cannot yield a success state.
- [ ] Simultaneous or sequential uploaded recordings with identical client filenames retain distinct recording, fit, and artifact references.
- [ ] Existing and uploaded sources produce scientifically consistent fits for equivalent input according to agreed baseline tolerances, and applicable access rules cover both.
- [ ] Verify upload → fit → ranked correlations through the approved boundary without duplicating scientific algorithms.
