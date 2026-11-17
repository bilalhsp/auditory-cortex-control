# Issue 01 scientific baseline

## Scope and storage

These experiments use the existing `.venv` and unchanged scientific algorithms.
All large scientific artifacts are under the configured cache directory, in
`web-migration-baseline/issue-01`. Recording and model files remain at their
configured locations. No new Python environment or Gradio UI is used.

The representative recording is existing session 0 of ucdavisBAK, also passed
through the full-path uploaded-recording parser. This verifies the two scientific
input paths, not an HTTP upload endpoint that has not been implemented yet.
Both TIMIT and mVocs parse successfully; this numerical baseline uses TIMIT.
Full workflow coverage for both stimulus families remains a migration acceptance
requirement, rather than a claim made by this representative experiment.

## Reproduction

Run the baseline probe from the repository root with the existing environment,
then run the generation probe after the fit report indicates completion. The
probes use existing local dependency checkouts; input hashes, revisions, effective
settings, and pre-existing dependency modifications are captured in
baseline-inputs.json and dependency-patches. They do not install packages.

The baseline probe redirects only newly fitted TRF artifact writes into its
isolated subdirectory of the configured cache. Feature caches use the repository's
normal configured cache location. It does not replace production fitted models.
The generation probe reads that isolated fit and writes separate result folders
for each objective and repeat.

## Passed checks

| Check | Observed result |
| --- | --- |
| torch GPU multiplication | 2 squared returns 4.0 |
| CuPy GPU operation | [0.0, 1.0, 4.0] |
| FastAPI HTTP probe | HTTP 200 with expected JSON using TestClient |
| SQLite import | Version 3.33.0 |
| Existing uv lock | Offline validation resolves 372 packages |
| Recording metadata | Three available ucdavisBAK sessions |
| Existing/uploaded parsers | Same session identity and unit list |
| TIMIT recording content | 267 unique and 10 repeated stimuli |
| mVocs recording content | 865 unique and 18 repeated stimuli |
| Seeded fits | Three successful fits with 18 finite unit correlations |
| Repeated existing fit | Maximum absolute correlation difference 0.0 |
| Existing versus uploaded fit | Maximum absolute correlation difference 0.0 |
| Worker CLI | Import and --help complete with exit status zero |

Measured fitting times were approximately 60.33, 57.28, and 56.33 seconds. Initial
construction including uncached feature extraction/persistence took approximately
332.93 seconds. These are observations, not release latency thresholds.

For this unchanged environment and fixed seed, the observed correlation baseline
is exact equality over the 18 recorded values. This is not evidence of exact
equality across hardware, dependency upgrades, or arbitrary recordings. Changes
to that runtime require a fresh repeatability measurement rather than an invented
portable numerical tolerance.

## Generation verification

Real DAPS generation is run with the unmodified 50-step sampler configuration,
seed 42, one nominal one-second sample, and the best-correlated unit (12.03),
twice for each of stretch and one-hot. Completion and repeatability are recorded
after all four samples pass finite-value, sample-rate, and PCM-format checks.

The first completed stretch output contains 15,392 frames at 16 kHz (0.962 seconds)
despite a nominal one-second request. Preserve and report this native output
behavior during migration; do not silently alter the sampler to force duration.

## Environment limitations

- The lockfile omits local scientific packages required by imports. Their pinned
  source revisions and pre-existing patches are part of this runtime's provenance.
- `uv pip check` reports AudioLDM's missing Gradio requirement, matching the
  explicit existing uv exclusion. The frontend is React; do not install Gradio
  to silence this check. Real generation, not import alone, validates that path.
- SoX is absent and emits an import warning. Existing recording parsing, fitting,
  and the completed sample used no SoX executable. Other untested scientific
  workflows may still need it.
- No new production API, authentication flow, cancellation, history, or restart
  handling is implemented by these baseline probes.
