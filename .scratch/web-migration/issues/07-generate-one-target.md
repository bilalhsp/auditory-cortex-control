# 07: Generate and download stimuli for one target

**What to build:** Generate stimuli for one selected channel/objective from a completed fit, observe its outcome, and download a validated ZIP of that generation’s WAV clips.

**Blocked by:** 05: Fit an existing recording and inspect correlations.

**Status:** claimed

Type: task

- [ ] Select stretch or one-hot, a valid fitted channel, duration 1–4 seconds, and 1/2/4/8/10 clips; retain defaults of 2 seconds and 4 clips.
- [ ] Generation references the exact immutable fit and effective configuration. Invalid/stale fits, unknown channels, empty targets, and invalid settings fail before launching work.
- [ ] Reuse the existing sampler/worker and scientific behavior. Execute outside the HTTP request, bind a permitted local device server-side, and expose stable generation/job identities.
- [ ] Preserve the GPU-count request from configuration; this slice demonstrates one target without silently changing that request. Explain unused devices under the scheduling policy from 01; full batch allocation is delivered in 08.
- [ ] Check worker exit status and validate a manifest before reporting success; missing/partial output, model failure, or unavailable GPU produces the defined failure outcome.
- [ ] Download only this generation’s intended artifacts as a ZIP; verify clip count, duration, 16-kHz PCM16 format, existing RMS normalization/clipping behavior, and access rules.
- [ ] Additional generations reuse the fit without overwriting earlier outputs; creating/opening a frontend cannot delete previous results.
- [ ] Verify both objectives against agreed scientific tolerances and exercise nonzero exit/missing-output failures using the approved test boundary.
