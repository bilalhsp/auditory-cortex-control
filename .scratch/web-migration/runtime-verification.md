# Runtime verification for issue 01

Read-only checks during issue 01:

- uv reports version 0.13.1.
- Existing scientific environment reports Python 3.8.5.
- Earlier device inspection reported one NVIDIA A100-SXM4-40GB; this is a
  hardware observation, not proof of exclusive allocation or usable VRAM.
- Configured datasets, pretrained weights, and cache directories exist.
- A bounded 40-second combined scientific import probe produced no completion
  output before its timeout. It did not establish generator compatibility.
- A subsequent isolated torch import completed within its 15-second bound:
  torch 2.0.1+cu117, CUDA build 11.7, one visible GPU, exit status zero. This
  verifies torch import/device enumeration, not CUDA computation or the full
  scientific worker stack.
- No model fitting or sample generation has been run, and no representative
  parity fixtures or numerical tolerances have been captured.

The existing environment and dependency lockfile have not been modified.
Target runtime selection and scientific parity remain open acceptance criteria.

## Follow-up checks in the existing environment

The user requires continuing with the existing environment, without creating a
replacement or upgrading dependencies. Both API and scientific worker processes
will initially use this same environment; process lifetime remains independent
of the browser connection.

- torch 2.0.1+cu117: actual CUDA multiplication returns 4.0 for 2.0 squared.
- CuPy 12.0.0: actual GPU array calculation returns [0.0, 1.0, 4.0].
- FastAPI 0.111.0: TestClient receives HTTP 200 and the expected JSON body from
  a minimal probe endpoint. SQLite 3.33.0 imports successfully.
- `uv lock --check --offline` resolves the existing 372-package lockfile.
- `uv pip check` reports AudioLDM's missing Gradio dependency. This exclusion is
  explicit in the existing uv configuration; Gradio is not part of the new UI.
  Generator import succeeds without it; end-to-end sampler execution is a
  separate baseline check, not implied by that import.
- Scientific generator import needs the existing local DeepSpeech, Wav2Letter,
  utils_jgm, and pycochleagram checkouts on the process import path. Do not claim
  that the root lockfile alone captures these dependencies.
- Local dependency revisions, dirty-diff hashes, recording identity, and model
  checkpoint hashes are recorded in baseline-inputs.json alongside this file.
  Small dependency patches preserve the pre-existing local changes; no package
  was installed or upgraded during verification.
- Metadata lists three ucdavisBAK sessions. The representative recording parses
  through both existing-session and full-path upload paths with matching session
  identity and channel list. It contains 267 unique/10 repeated TIMIT stimuli
  and 865 unique/18 repeated mVocs stimuli.
- The first fit produced finite correlations for 18 units. Full repeatability
  and generation results are recorded separately once completed.
- An earlier cache write exhausted the home-directory quota. Only the temporary
  generated caches were removed; all subsequent scientific artifacts use the
  configured cache directory. Original fitted models remain untouched.
