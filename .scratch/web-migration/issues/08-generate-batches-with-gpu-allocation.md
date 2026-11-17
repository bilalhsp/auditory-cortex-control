# 08: Generate batches using the selected GPU count

**What to build:** Generate multiple channel/objective targets while enforcing the user-selected GPU count, current availability, and explicit per-target outcomes.

**Blocked by:** 02: Configure an experiment and select backend GPUs, 07: Generate and download stimuli for one target.

**Status:** resolved

Type: task

- [x] Preserve the maximum-available default and valid smaller explicit requests when submitting a target batch; resolve actual device IDs on the backend.
- [x] Revalidate capacity at submission/allocation using the reservation and contention policy from 01; handle stale capacity explicitly without silent reduction or oversubscription.
- [x] Bound concurrent target workers to selected count and schedule excess targets according to 01; verify behavior when targets are fewer than GPUs. Do not imply single-target distributed sampling.
- [x] Honor runtime visibility and external allocation constraints; local workers never receive invalid device IDs inferred from remote host counts.
- [x] Expose per-target lifecycle/results, overall outcome, and manifest/download availability under the partial-completion policy from 01; release reservations on success and failure.
- [x] Isolate worker configuration/output locations and prevent concurrent filename races; associate logs with generation and target identities.
- [x] Verify one/multiple selected GPUs, more/fewer targets than GPUs, changed capacity, competing requests, failed targets, and resource release; benchmark real-GPU execution against agreed limits.

## Answer

Implemented authenticated target batches with atomic full-count GPU reservations,
allocation/visibility rechecks, one bounded worker slot per used device, explicit
per-target states, isolated outputs/logs, and partial-result ZIP manifests. The
React batch panel preserves the saved GPU choice, retains valid smaller choices,
and requires explicit reselection when capacity changes. Completed targets survive
failed siblings and worker interruption; terminal outcomes release reservations.

All 58 backend tests, typechecks, builds, standalone archive checks, and browser
workflows pass. A real one-GPU/two-target batch completed in 570.180 seconds with
5,497 MiB peak observed memory, validated both outputs, and released capacity.
Multi-GPU scheduling and binding are verified with controlled external compute
and the real subprocess launcher; this server exposes only one physical GPU, so
physical multi-GPU throughput remains unmeasured. No latency limit was imposed.

See [verification](../issue08-verification.md),
[benchmark](../issue08-benchmark.json), and
[Decisions-so-far](../map.md#decisions-so-far). Standards review's polling/download
duplication was addressed and rechecked; Spec review found no violations. Issue 09
is the next ready issue.
