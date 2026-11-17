# Issue 07 verification

Approved boundary: authenticated FastAPI HTTP with real services/SQLite and a
controlled external compute runner; scientific parity is checked separately on
the existing CUDA environment. No environment or dependencies were replaced.

- Six generation acceptance tests pass: default settings and idempotent IDs;
  authenticated job/status/download access; exact ZIP membership; artifact
  retention and corruption rejection; stale/missing/incomplete fit checks;
  invalid channels/settings; GPU absence; nonzero exit even with valid WAVs;
  missing/invalid output; preserved multi-GPU request with one active target;
  API restart while generation remains running.
- Complete backend suite: 51 tests pass. TypeScript/Pyright checks, frontend
  production build, and changed scientific module syntax checks pass.
- Chromium with the real API/SQLite and controlled compute passes completed fit
  → generation controls/defaults → running/success → authenticated ZIP download,
  actual native duration, failures hiding downloads, and mobile layout without
  page errors. Screenshots remain under configured cache `auditory-web-checks/issue07-*`.
- Built backend archive, imported from `/tmp`, can queue and inspect a generation
  against an isolated metadata fixture without scientific model imports.
- First real sampling attempt exposed a wrong configuration import after sampling;
  it exited nonzero and correctly remained failed with no download. The import
  was corrected. A subsequent real stretch job succeeded with valid output and
  a verified download, but its WAV differs from the historical issue 01 hash.
- All saved fit parameters (weights, alphas, X mean/std, y mean) match the legacy
  baseline exactly, maximum absolute element delta 0.0. Channel order, checkpoint,
  target 12.03, sampler settings, seed 42, format, and duration match.
- PCM difference for that real stretch output against the historical WAV:
  maximum absolute 0.844940185546875; RMS 0.13711356744158967. This does NOT pass
  issue 01's exact-equality requirement. No broader tolerance has been adopted.

Fresh unchanged legacy sampling also differs from the historical WAVs for both
objectives: maximum PCM differences 1.15093994140625 (stretch) and
0.981964111328125 (one-hot). The historical measurements remain in
[issue07-historical-comparison.json](issue07-historical-comparison.json).

The user explicitly approved controlled legacy/web comparisons with identical
recorded initialization and RNG state instead of requiring historical WAV hashes.
Both controlled comparisons now pass with byte-identical WAVs and maximum/RMS
PCM differences 0.0. Each output has 15,392 frames at 16 kHz PCM16, matching native
one-second rounding; exit codes are zero. CPU/CUDA initialization RNG hashes match.
See [issue07-parity.json](issue07-parity.json) and the reproducible
[controlled probe](issue07_controlled_probe.py).

The controlled scientific comparison runs the real default legacy Sampler and
real web scientific adapter sequentially in one process, seeding before each
model construction. Web generation submission, persistence, validation, status,
and ZIP download still cross the real HTTP/SQLite/worker boundary; the external
compute runner invokes the scientific adapter directly to control initialization.
The production isolated child path was checked separately with a successful real
stretch output and an actual nonzero failure withholding partial output. This
controlled result does not establish cross-process or cross-hardware equality.

Production initialization now uses the requested seed before constructing models,
as well as the existing per-sample seeding. Provenance records the initialization
seed and CPU/CUDA RNG hashes. No sampling algorithm, fitted parameter, RMS
normalization, clipping, or native duration was rewritten.

## Standards

Independent fresh-context review of `ccfaa2c` against fixed base
`4e6b36778054e2477ebd69ce27c861da9ed73b6d` found no documented violations.
One possible duplication smell concerned supported choices repeated in API,
validation, and React. Addressed in `35a3680`: shared backend definitions and
capability-driven frontend options. Typechecks, production build, and browser
workflow pass after this change.

## Spec

Independent review confirmed the implemented functional scope and flagged the
initially unmet requirement to verify both objectives against agreed scientific
tolerances. The user approved the controlled-comparison revision after fresh
legacy processes also failed historical equality. Both revised comparisons now
pass exactly; historical evidence is retained. Follow-up Spec review confirmed
the revised gate is satisfied with no further blocker. No scope creep was identified.

Review totals: Standards 1 heuristic addressed, 0 hard violations; Spec 1
initially incomplete acceptance requirement, now satisfied under the approved
revision. No implementation findings remain.
