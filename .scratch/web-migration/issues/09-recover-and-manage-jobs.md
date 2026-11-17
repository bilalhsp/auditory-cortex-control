# 09: Recover and manage long-running work

**What to build:** Deliver the reconnect, restart, retry, and cancellation behavior chosen in 01, making computation outcomes recoverable and resource handling predictable.

**Blocked by:** 05: Fit an existing recording and inspect correlations, 07: Generate and download stimuli for one target.

**Status:** resolved

Type: task

- [x] Implement only the job guarantees accepted in 01; explicitly document declined capabilities instead of silently broadening scope.
- [x] Where durability is accepted, accepted work is persisted before execution and browser reconnect recovers fit/generation status and results by stable identity.
- [x] Reconcile interrupted API/worker execution after restart without falsely declaring success, losing completed artifacts, or duplicating scientific work.
- [x] Where retries/idempotency are accepted, distinguish execution attempts, isolate attempt outputs, and avoid duplicate generation after ambiguous submissions.
- [x] Where cancellation is accepted, expose it through UI/API, define terminal-state races and partial outputs, terminate/reconcile workers, and release assigned resources.
- [x] Verify the selected guarantees against connection loss, API restart, worker interruption, repeated submission, and terminal transitions at the approved boundary; preserve baseline behavior for declined features.

## Answer

Implemented persistent application-session history, stable work recovery, Resume
setup, saved result views/downloads, individual Stop, and explicit Stop/Close.
Queued work cancels immediately; active work retains reservations until owned
compute children exit. Children inherit allocator ownership across owner failure.
Terminal-state transactions preserve completed results and prevent cancelled work
from publishing success. Closed sessions are read-only and retain their history.

Explicit retries create isolated, idempotent attempts with source ownership and
atomic provenance. Batch retry includes only unfinished targets. Fully saved
interrupted batches remain downloadable without offering an empty retry.

All 69 backend tests, typechecks, builds, standalone archive checks, and browser
flows pass. A real subprocess fit matches all 18 baseline correlations exactly;
a second native fit stopped during fitting exited -15, released capacity, and
preserved the previous result. See [verification](../issue09-verification.md),
[native evidence](../issue09-native.json), and
[Decisions-so-far](../map.md#decisions-so-far).

Standards and Spec findings were corrected and confirmed resolved. No automatic
scientific retries, seamless checkpoint continuation, backups, or multi-user scope
were added. Issue 10 is the next ready issue.
