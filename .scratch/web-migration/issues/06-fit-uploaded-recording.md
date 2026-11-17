# 06: Fit an uploaded recording

**What to build:** Fit an uploaded recording using the same web workflow and correlation view while preserving unique recording and model identities.

**Blocked by:** 04: Upload and validate a recording, 05: Fit an existing recording and inspect correlations.

**Status:** resolved

Type: task

- [x] Validated uploaded recordings can enter the existing fit submission/status/result workflow with the same scientific settings and correlation display.
- [x] The legacy internal session-zero convention cannot cause uploaded fitted models to overwrite or reload another recording’s fit.
- [x] Validation failures prevent fitting; parsing, loading, and computation errors remain actionable and cannot yield a success state.
- [x] Simultaneous or sequential uploaded recordings with identical client filenames retain distinct recording, fit, and artifact references.
- [x] Existing and uploaded sources produce scientifically consistent fits for equivalent input according to agreed baseline tolerances, and applicable access rules cover both.
- [x] Verify upload → fit → ranked correlations through the approved boundary without duplicating scientific algorithms.

## Answer

Validated uploads now use the shared persistent fit endpoint, worker, status, and
React correlation panel. Upload ownership, validation checksum/size, immutable
recording revision, and pinned inputs are enforced before acceptance. Legacy
full-path/session-zero parsing is preserved while fit UUID workspaces isolate
all models, configurations, and provenance. Replacing an upload preserves older
accepted fits and artifacts.

All 45 backend tests, Python/TypeScript checks, builds, browser checks, and
standalone artifact checks pass. Two same-named real uploads each reproduce all
18 baseline correlations exactly; first artifacts remain unchanged after the
second fit. See [verification](../issue06-verification.md),
[parity evidence](../issue06-parity.json), and
[Decisions-so-far](../map.md#decisions-so-far).

Standards review found no violations. The Spec review's configuration-error
finding was corrected with a deployment-level HTTP regression and independently
confirmed resolved. No findings remain. Issue 07 is the next ready issue.
