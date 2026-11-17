# Issue 11 release verification

Status: automated local acceptance passed; awaiting operator pilot/release sign-off.
Production cutover and Streamlit retirement have not been performed.
The operator approved local URL-and-port deployment checks for this issue.
Accepted requirements are in `docs/agents/web-migration-decisions.md`; backups,
latency thresholds, automatic expiry, multi-user features, and a fixed rollback
calendar remain excluded.

## Acceptance coverage

| Requirement | Evidence | State |
| --- | --- | --- |
| Both families and both recording sources; fit/correlations; both objectives and repeated generation; scoped persisted downloads | `backend/tests/test_acceptance.py`: four combinations, three distinct generations each, closure and zero-GPU API restart | Controlled HTTP matrix passed |
| TIMIT fit scientific parity, both sources | `issue05-parity.json`, `issue06-parity.json`, `issue09-native.json` | 18 correlations match baseline exactly; max delta 0.0 |
| TIMIT audio scientific parity, both objectives | `issue07-parity.json`, approved controlled initialization/RNG comparison | Byte-identical, max/RMS PCM delta 0.0; historical cross-process variability retained |
| mVocs scientific parity | `issue11_native_probe.py` | Real fits passed: 18 correlations per source, max delta 0.0; Both real fits and both controlled real audio comparisons pass exactly |
| GPU count admission, bounded target waves, idle reservations, contention/isolation | Existing batch/subprocess/lifecycle HTTP tests; `issue08-benchmark.json` | Controlled multi-device checks and real single-GPU two-target batch passed |
| Browser reconnect, retry, disconnect/history/download, close with preserved partial samples, mobile | `issue11_browser_probe.py`, `issue11_browser_api.py`, `issue11_browser.mjs` | Passed against built frontend on 127.0.0.1:5173 and independent API on 127.0.0.1:8765 |
| Separate builds/runtime API configuration/CORS/credentials and zero-GPU metadata readiness | Frontend build; archive probe; browser route-injected runtime API configuration; API suite | Frontend and backend archive builds passed; standalone archive recovery/closure check passed |
| Worker interruption, surviving child ownership, Stop/release ordering, attempt isolation | `backend/tests/test_sessions.py`, prior real fit cancellation in `issue09-native.json` | Passed again in full regression |
| Authentication on all commands/status/history/downloads, safe temporary cleanup, retained artifacts | Issue 10 HTTP tests and upload/batch/lifecycle regression cases | Passed again in full regression |
| Rollback runtime and retained artifacts | `issue11_rollback_probe.py`, `issue11-rollback.json` | Legacy server health and isolated legacy UI passed; all five retained artifact hashes unchanged |
| Operator pilot/release sign-off, public proxy checks, actual cutover and retirement | `docs/deployment/web-cutover.md` checklist | Pending; no production traffic changed |

A second local deployment check (`issue11_deployed_probe.py` and
`issue11_deployed_browser.mjs`) launches the actual backend archive from outside
the repository on port 8765 and serves a copied frontend build on port 5173.
Its real served `config.js` points to that API; no request interception supplies
configuration. Browser sign-in and history show all 18 real mVocs correlations
from the native database. This passed. The probe is read-only and does not start
new scientific work. Fresh loopback-only credentials are ephemeral and not logged.

## Reproduction

Use the existing Python environment. For native and legacy UI probes set
`PYTHONPATH` to the repository plus the existing dependency checkouts:

```bash
export PYTHONPATH="$PWD:/home/ahmedb/projects/deepspeech.pytorch:/home/ahmedb/projects/Wav2Letter:/home/ahmedb/projects/utils_jgm:/home/ahmedb/projects/pycochleagram"
uv run --no-sync python .scratch/web-migration/issue11_native_probe.py
uv run --no-sync python .scratch/web-migration/issue11_generation_probe.py
uv run --no-sync python .scratch/web-migration/issue11_rollback_probe.py
uv run --no-sync python .scratch/web-migration/issue11_archive_probe.py
uv run --no-sync python .scratch/web-migration/issue11_browser_probe.py
uv run --no-sync python .scratch/web-migration/issue11_deployed_probe.py
```

All large artifacts/logs/screenshots remain under the configured cache. The
browser fixture uses controlled compute; the native fit probe invokes the real
worker subprocess after real HTTP submission. The archive probe launches from
`/tmp` with only the built archive on its import path, copies fixture metadata
using SQLite's consistent backup API, and confirms history with no torch imports.
That isolated fixture-copy mechanism is not a user backup/restore feature.

## Measurements and limits

Prior measured TIMIT initialization was 332.93 s; baseline fits were 56.33–60.33 s,
web subprocess fits approximately 133–149 s. The real two-target TIMIT batch took
570.18 s with peak observed GPU memory 5,497 MiB on an A100 40 GB. Queue wait and
target start/finish timestamps remain in the corresponding persisted job records.
No latency threshold was imposed. mVocs uncached initialization took 715.79 s, legacy fit 74.95 s, and web
existing/upload fits 176.00/174.39 s; see `issue11-native.json`. Physical multi-GPU throughput is unmeasured
on this one-GPU server; do not infer it from controlled allocation tests.

The retained legacy Frontend deletes its dated output folder at startup. The
rollback rehearsal deliberately uses a new isolated cache workspace rather than
opening that UI against prior outputs. Streamlit remains available; it has not
been archived. The actual public proxy traffic switch is not rehearsed locally.

## Regression and review

All **74 backend tests passed in 355.034 seconds**, including the four-way release
matrix. Backend Pyright reports zero errors, frontend TypeScript/Vite build and
backend archive build pass, and preserved probes pass syntax checks. The browser
probe passed again after switching its paths to the configured cache directory.
The archive probe now explicitly compares two closure responses before claiming
idempotency and passed again.

Review base: `768142820514a5e76ac6d168900eb8dcdeff8ab3`. Standards review found one
unsupported idempotency message, fixed with the repeated-close assertion. Spec
review of the added audio probe found nullable/CuPy parameter handling would fail
before sampling; fixed with matching keys/shapes, explicit paired-None handling,
and CPU conversion. Both were verification-probe fixes, not scientific changes.
The audio probe also requires exactly the expected existing/upload fit pair and
two completed objective comparisons before reporting completion.


Real mVocs fitting is now complete: both sources preserve all 18 legacy channel
correlations exactly, including best channel 12.03. Both real audio comparisons now pass with byte-identical legacy/web WAVs and
maximum/RMS PCM deltas 0.0; see `issue11-generation.json`. This is the approved
controlled initialization/RNG comparison, not cross-process determinism.


Resource observations during the mVocs run: 5,857 MiB GPU memory with 27% GPU
utilization during fitting, and 3,377 MiB with 38% utilization during sampling.
These are point observations, not measured peaks. The existing TIMIT batch report
records a sampled peak separately. Native fit queue waits were 0.983 s (existing)
and 0.873 s (upload).

## Standards

One unsupported idempotency claim was corrected by asserting repeated closure;
the reviewer confirmed resolution. Follow-up review of native-data deployment
probes found no actionable standards or cleanup findings. Zero outstanding.

## Spec

One scientific-probe parameter comparison bug (nullable values and explicit CuPy
conversion) was corrected; the reviewer confirmed resolution. Actual sampling
has now passed that comparison. Follow-up review of native-data local deployment
found no actionable mismatch. Zero outstanding implementation findings; operator
pilot and release sign-off remain explicit acceptance gates.


## Final automated acceptance

`issue11-generation.json` is complete. Both objectives preserve 15,392 mono frames
at 16 kHz PCM16 (native duration 0.962 s for nominal 1 s), exit code zero, exact
WAV bytes, and maximum/RMS PCM deltas 0.0. Saved fit parameter differences are
0.0, initialization RNG hashes match, and scoped ZIPs contain only clip-001.wav.
The stretch comparison uses the existing-recording fit and one-hot uses the
uploaded-recording fit; the controlled HTTP release matrix covers both objectives
and repeated generation for every family/source combination. No physical
multi-GPU throughput or cross-process audio determinism claim is made.

All automated checks are complete and passing. Remaining release requirements
are an operator pilot and explicit release sign-off, an actual production traffic
switch if desired, public HTTPS/proxy checks before public exposure, and separate
approval before retiring Streamlit. The user approved local URL-and-port checks;
those have passed and the local test servers were shut down. No production
traffic or persistent deployment was changed. The ticket remains open at this
human release gate; it is not recorded as a completed production cutover.


Native mVocs timing (legacy/web): stretch 188.08/204.39 s and one-hot
177.41/203.36 s. Web queue waits were 0.218/0.271 s respectively. Total controlled
audio comparison runtime was 778.21 s. These are observations, not release limits.
