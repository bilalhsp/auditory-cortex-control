# 08: Generate batches using the selected GPU count

**What to build:** Generate multiple channel/objective targets while enforcing the user-selected GPU count, current availability, and explicit per-target outcomes.

**Blocked by:** 02: Configure an experiment and select backend GPUs, 07: Generate and download stimuli for one target.

**Status:** claimed

Type: task

- [ ] Preserve the maximum-available default and valid smaller explicit requests when submitting a target batch; resolve actual device IDs on the backend.
- [ ] Revalidate capacity at submission/allocation using the reservation and contention policy from 01; handle stale capacity explicitly without silent reduction or oversubscription.
- [ ] Bound concurrent target workers to selected count and schedule excess targets according to 01; verify behavior when targets are fewer than GPUs. Do not imply single-target distributed sampling.
- [ ] Honor runtime visibility and external allocation constraints; local workers never receive invalid device IDs inferred from remote host counts.
- [ ] Expose per-target lifecycle/results, overall outcome, and manifest/download availability under the partial-completion policy from 01; release reservations on success and failure.
- [ ] Isolate worker configuration/output locations and prevent concurrent filename races; associate logs with generation and target identities.
- [ ] Verify one/multiple selected GPUs, more/fewer targets than GPUs, changed capacity, competing requests, failed targets, and resource release; benchmark real-GPU execution against agreed limits.
