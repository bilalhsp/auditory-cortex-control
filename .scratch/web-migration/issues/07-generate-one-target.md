# 07: Generate and download stimuli for one target

**What to build:** Generate stimuli for one selected channel/objective from a completed fit, observe its outcome, and download a validated ZIP of that generation’s WAV clips.

**Blocked by:** 05: Fit an existing recording and inspect correlations.

**Status:** resolved

Type: task

- [x] Select stretch or one-hot, a valid fitted channel, duration 1–4 seconds, and 1/2/4/8/10 clips; retain defaults of 2 seconds and 4 clips.
- [x] Generation references the exact immutable fit and effective configuration. Invalid/stale fits, unknown channels, empty targets, and invalid settings fail before launching work.
- [x] Reuse the existing sampler/worker and scientific behavior. Execute outside the HTTP request, bind a permitted local device server-side, and expose stable generation/job identities.
- [x] Preserve the GPU-count request from configuration; this slice demonstrates one target without silently changing that request. Explain unused devices under the scheduling policy from 01; full batch allocation is delivered in 08.
- [x] Check worker exit status and validate a manifest before reporting success; missing/partial output, model failure, or unavailable GPU produces the defined failure outcome.
- [x] Download only this generation’s intended artifacts as a ZIP; verify clip count, duration, 16-kHz PCM16 format, existing RMS normalization/clipping behavior, and access rules.
- [x] Additional generations reuse the fit without overwriting earlier outputs; creating/opening a frontend cannot delete previous results.
- [x] Verify both objectives against agreed scientific tolerances and exercise nonzero exit/missing-output failures using the approved test boundary.

## Comments

2026-11-17: User approved replacing historical-WAV equality as the generation
acceptance gate with controlled legacy/web comparisons using identical recorded
model initialization and RNG state. The unchanged legacy sampler also fails the
historical stretch hash in a fresh process (max PCM delta 1.15093994140625). Keep
the historical evidence and document cross-process variability. Both objectives
still require measured controlled comparison; do not invent a numerical tolerance.

## Answer

Implemented single-channel generation from immutable completed fits, supported
objectives/durations/counts and defaults, shared persistent worker scheduling,
isolated sampler execution, and authenticated validated ZIP downloads in React.
GPU requests remain unchanged; one target uses one GPU with unused capacity
explained. Repeated generations retain earlier files and operation identities.

All 51 backend tests, typechecks/builds, browser workflows, and standalone
artifact checks pass. Under the explicitly approved controlled initialization/RNG
comparison, both stretch and one-hot downloaded WAVs are byte-identical to legacy
outputs (maximum/RMS PCM difference 0.0). Historical fresh-process variability is
retained and documented; no portable determinism claim is made.

See [verification](../issue07-verification.md),
[controlled evidence](../issue07-parity.json), and
[Decisions-so-far](../map.md#decisions-so-far). Standards review's duplicated
choices were consolidated. Spec review's scientific acceptance gate is satisfied
under the user-approved revision. Issue 08 is the next ready issue.
