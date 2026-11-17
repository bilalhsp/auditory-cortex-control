# 05: Fit an existing recording and inspect correlations

**What to build:** Fit an encoding model for an existing recording through the web interface, observe its outcome, and inspect ranked channel correlations backed by a uniquely identified fitted artifact.

**Blocked by:** 03: Select an existing recording.

**Status:** resolved

Type: task

- [x] A fit references an immutable experiment revision and recording identity; submission returns stable fit/job identities without holding HTTP open for computation.
- [x] Reuse existing TRF/data assembly logic and three-fold cross-validation. Prefactor Streamlit-dependent construction only as needed and isolate mutable fitting state rather than globally sharing it.
- [x] Persist and expose lifecycle/results according to 01, including useful loading/running/failure states and operation-correlated diagnostics without private recording content.
- [x] Save each fitted artifact/configuration under a unique operation identity and snapshot effective settings, recording/model identity, code/environment identity, and seed where applicable.
- [x] Display correlations in descending order with a best-channel summary; preserve signed/non-integer channel identities and represent unavailable numeric results explicitly.
- [x] Unknown recordings, incompatible context, model-loading failures, and fitting errors cannot report completion or unlock generation.
- [x] Compare representative fits against the 01 baseline at agreed tolerances; verify distinct fits cannot overwrite one another and preserve CLI/legacy compatibility.

## Answer

Implemented immutable fit submission, a separately supervised persistent fit
worker, unique private model/config/provenance storage, and a React fit-status and
ranked-correlation view. Reused the existing scientific pipeline with optional
pinned-input/output hooks while preserving legacy defaults.

All 40 API tests, backend/frontend typechecks, production builds, browser checks,
standalone backend artifact checks, and legacy CLI import checks pass. Two real
GPU fits match all 18 issue 01 correlations exactly and retain distinct artifacts.
See [verification](../issue05-verification.md) and
[Decisions-so-far](../map.md#decisions-so-far).

Independent Standards and Spec reviews found no substantive implementation
issues. Their documentation notes (ticket type and exact-equality wording) were
corrected. Issue 06 is now the lowest-numbered ready issue; issue 07 is also
unblocked. Generation, uploaded fitting, history, and Stop/Close remain later
slices.
