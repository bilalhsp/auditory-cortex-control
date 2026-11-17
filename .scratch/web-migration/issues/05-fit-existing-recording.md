# 05: Fit an existing recording and inspect correlations

**What to build:** Fit an encoding model for an existing recording through the web interface, observe its outcome, and inspect ranked channel correlations backed by a uniquely identified fitted artifact.

**Blocked by:** 03: Select an existing recording.

**Status:** claimed

- [ ] A fit references an immutable experiment revision and recording identity; submission returns stable fit/job identities without holding HTTP open for computation.
- [ ] Reuse existing TRF/data assembly logic and three-fold cross-validation. Prefactor Streamlit-dependent construction only as needed and isolate mutable fitting state rather than globally sharing it.
- [ ] Persist and expose lifecycle/results according to 01, including useful loading/running/failure states and operation-correlated diagnostics without private recording content.
- [ ] Save each fitted artifact/configuration under a unique operation identity and snapshot effective settings, recording/model identity, code/environment identity, and seed where applicable.
- [ ] Display correlations in descending order with a best-channel summary; preserve signed/non-integer channel identities and represent unavailable numeric results explicitly.
- [ ] Unknown recordings, incompatible context, model-loading failures, and fitting errors cannot report completion or unlock generation.
- [ ] Compare representative fits against the 01 baseline at agreed tolerances; verify distinct fits cannot overwrite one another and preserve CLI/legacy compatibility.
