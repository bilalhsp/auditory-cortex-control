# Issue 11 release verification

Status: local acceptance in progress; production cutover/retirement not approved.
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
| mVocs scientific parity | `issue11_native_probe.py` | Real legacy/web fit probe running; audio comparison pending |
| GPU count admission, bounded target waves, idle reservations, contention/isolation | Existing batch/subprocess/lifecycle HTTP tests; `issue08-benchmark.json` | Controlled multi-device checks and real single-GPU two-target batch passed |
| Browser reconnect, retry, disconnect/history/download, close with preserved partial samples, mobile | `issue11_browser_probe.py`, `issue11_browser_api.py`, `issue11_browser.mjs` | Passed against built frontend on 127.0.0.1:5173 and independent API on 127.0.0.1:8765 |
| Separate builds/runtime API configuration/CORS/credentials and zero-GPU metadata readiness | Frontend build; archive probe; browser route-injected runtime API configuration; API suite | Builds passed; archive recheck in progress |
| Worker interruption, surviving child ownership, Stop/release ordering, attempt isolation | `backend/tests/test_sessions.py`, prior real fit cancellation in `issue09-native.json` | Existing checks passed; full regression recheck pending |
| Authentication on all commands/status/history/downloads, safe temporary cleanup, retained artifacts | Issue 10 HTTP tests and upload/batch/lifecycle regression cases | Existing checks passed; full regression recheck pending |
| Rollback runtime and retained artifacts | `issue11_rollback_probe.py`, `issue11-rollback.json` | Legacy server health and isolated legacy UI passed; all five retained artifact hashes unchanged |
| Operator pilot/release sign-off, public proxy checks, actual cutover and retirement | `docs/deployment/web-cutover.md` checklist | Pending; no production traffic changed |

## Reproduction

Use the existing Python environment. For native and legacy UI probes set
`PYTHONPATH` to the repository plus the existing dependency checkouts:

```bash
export PYTHONPATH="$PWD:/home/ahmedb/projects/deepspeech.pytorch:/home/ahmedb/projects/Wav2Letter:/home/ahmedb/projects/utils_jgm:/home/ahmedb/projects/pycochleagram"
uv run --no-sync python .scratch/web-migration/issue11_native_probe.py
uv run --no-sync python .scratch/web-migration/issue11_rollback_probe.py
uv run --no-sync python .scratch/web-migration/issue11_archive_probe.py
uv run --no-sync python .scratch/web-migration/issue11_browser_probe.py
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
No latency threshold was imposed. mVocs initialization/fit measurements will be
recorded by the current native probe. Physical multi-GPU throughput is unmeasured
on this one-GPU server; do not infer it from controlled allocation tests.

The retained legacy Frontend deletes its dated output folder at startup. The
rollback rehearsal deliberately uses a new isolated cache workspace rather than
opening that UI against prior outputs. Streamlit remains available; it has not
been archived. The actual public proxy traffic switch is not rehearsed locally.
