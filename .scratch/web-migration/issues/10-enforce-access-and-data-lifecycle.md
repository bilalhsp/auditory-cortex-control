# 10: Enforce deployment access and data lifecycle policies

**What to build:** Complete and verify the access/sharing, retention, and backup policies chosen in 01 across both recording sources, fits, jobs, and downloads.

**Blocked by:** 06: Fit an uploaded recording, 07: Generate and download stimuli for one target.

**Status:** claimed

- [ ] Apply the audience/access decision from 01 consistently to recordings, experiments, fits, job status, generation results, and artifact downloads; do not introduce public signup unless approved.
- [ ] Verify unauthorized identity/resource combinations cannot access protected data, including direct artifact URLs; for single-operator deployments verify the selected deployment access boundary.
- [ ] Preserve any sharing rules and isolate resource ownership; verify core access enforcement introduced in preceding slices rather than postpone it until this ticket.
- [ ] Implement approved retention/cleanup behavior with protected active work and complete/partial artifact semantics; initialization never deletes shared results.
- [ ] Exercise the selected backup/restore procedure for metadata and artifacts and verify references remain usable after restoration.
- [ ] Verify both existing and uploaded recording workflows, rejected access, safe cleanup, and recovery under the approved policies; exclude multi-user features if explicitly declined.
