# 10: Enforce deployment access and data lifecycle policies

**What to build:** Complete and verify the access/sharing, retention, and backup policies chosen in 01 across both recording sources, fits, jobs, and downloads.

**Blocked by:** 06: Fit an uploaded recording, 07: Generate and download stimuli for one target.

**Status:** resolved

- [x] Apply the audience/access decision from 01 consistently to recordings, experiments, fits, job status, generation results, and artifact downloads; do not introduce public signup unless approved.
- [x] Verify unauthorized identity/resource combinations cannot access protected data, including direct artifact URLs; for single-operator deployments verify the selected deployment access boundary.
- [x] Preserve any sharing rules and isolate resource ownership; verify core access enforcement introduced in preceding slices rather than postpone it until this ticket.
- [x] Implement approved retention/cleanup behavior with protected active work and complete/partial artifact semantics; initialization never deletes shared results.
- [x] Exercise the selected backup/restore procedure for metadata and artifacts and verify references remain usable after restoration.
- [x] Verify both existing and uploaded recording workflows, rejected access, safe cleanup, and recovery under the approved policies; exclude multi-user features if explicitly declined.

## Comments

The accepted decisions in `docs/agents/web-migration-decisions.md` and ADR 0002
supersede the original backup/restore and multi-user/sharing proposals. Backup
implementation and restore acceptance are excluded. Retention means no automatic
expiry/deletion; session closure retains completed results. There is no approved
deletion workflow to implement. Existing staging/download temporary cleanup must
not delete accepted work or retained scientific artifacts.

## Answer

Verified the selected single-operator boundary across every API route, including
direct downloads, and added no-store protection for all API responses. Saved
sample downloads remain authenticated and unchanged after session closure and API
restart with zero GPUs. Existing tests verify uploaded-recording ownership,
rejected/cancelled staging cleanup, independent artifacts, protected active work,
and completed/partial result retention. No automatic expiry or deletion was added.

The backup/restore checklist is satisfied by the explicit accepted scope exclusion,
not by implementing or claiming a tested backup. Multi-user sharing is likewise
excluded. Public proxy setup remains a deployment gate documented in the backend
README; local checks do not claim a live HTTPS deployment.

All 73 backend tests passed (324.539 seconds); Pyright, React typecheck/production
build, backend archive build, and diff whitespace checks passed. Independent
Standards and Spec reviews: zero findings each. See
[verification](../issue10-verification.md). Issue 11 is next.
