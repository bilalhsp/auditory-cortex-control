# 09: Recover and manage long-running work

**What to build:** Deliver the reconnect, restart, retry, and cancellation behavior chosen in 01, making computation outcomes recoverable and resource handling predictable.

**Blocked by:** 05: Fit an existing recording and inspect correlations, 07: Generate and download stimuli for one target.

**Status:** claimed

Type: task

- [ ] Implement only the job guarantees accepted in 01; explicitly document declined capabilities instead of silently broadening scope.
- [ ] Where durability is accepted, accepted work is persisted before execution and browser reconnect recovers fit/generation status and results by stable identity.
- [ ] Reconcile interrupted API/worker execution after restart without falsely declaring success, losing completed artifacts, or duplicating scientific work.
- [ ] Where retries/idempotency are accepted, distinguish execution attempts, isolate attempt outputs, and avoid duplicate generation after ambiguous submissions.
- [ ] Where cancellation is accepted, expose it through UI/API, define terminal-state races and partial outputs, terminate/reconcile workers, and release assigned resources.
- [ ] Verify the selected guarantees against connection loss, API restart, worker interruption, repeated submission, and terminal transitions at the approved boundary; preserve baseline behavior for declined features.
