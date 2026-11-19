# 03: Guide setup, recording selection, and fitting

**What to build:** Guide the operator through setup, Recording and Fit with Back and a persistent responsive summary.

**Blocked by:** 02

**Status:** resolved

- [x] Guide the operator through setup, Recording and Fit with Back and a persistent responsive summary.
- [x] Default to Upload recording and offer Choose existing recording in the same view.
- [x] Display meaningful fit stages and reduced-motion-compatible progress. Editing fit inputs after accepted work creates a separate experiment and requires a new fit.

[Parent specification](../spec.md)

## Answer

The Workbench now presents Setup, Recording, and Fit as separate steps with Back navigation and a responsive selection summary. Recording opens on Upload, with an adjacent option for an existing recording. Fit reports its latest stage and respects reduced-motion preferences.

Changed setup settings save a separate experiment. Changing the recording after fit acceptance attaches the recording to a separate experiment in the same session, preserving the original inputs and jobs. Start over opens Setup, and Back is disabled while the browser submits a fit.

Verified with the frontend build, focused API tests, and a Playwright run against an isolated FastAPI instance and real SQLite persistence. The browser exercised both recording sources, navigation, replacement after fit acceptance, reduced motion, and the mobile layout.
