# Issue 10 verification

The accepted policy in `docs/agents/web-migration-decisions.md` supersedes the
original ticket's multi-user/sharing and backup/restore proposals: one operator,
private authenticated resources, persistent filesystem artifacts, no automatic
expiry, and no backup implementation. No deletion endpoint, expiry scheduler,
backup system, or multi-user ownership model was introduced.

## Changes

- All `/api/` responses now carry `Cache-Control: no-store`, including rejected
  authentication and normal metadata/results. Existing authenticated downloads
  keep that header. Health remains public and reveals only liveness.
- HTTP acceptance checks enumerate every current API route and reject missing
  credentials, an incorrect password, and an incorrect username, including direct
  download and manifest routes. No static artifact or documentation route exists.
- A real persisted workflow with controlled compute creates a valid sample ZIP,
  closes its application session, and restarts the API with zero GPUs. The same
  ZIP remains available byte-for-byte to the operator; invalid credentials still
  receive 401.
- Deployment documentation specifies private upstream/storage, exact frontend
  origins, HTTPS and proxy authentication rate limits, secrets handling, no API
  caching, and no static exposure of research data or diagnostic files.

## Verification

The new cache acceptance test failed before the change (missing no-store), then
passed. The four new access tests pass using real FastAPI/SQLite/workflow services
and controlled compute. Backend Pyright reports zero errors; React TypeScript and
Vite production build pass. The independent backend archive builds using the
existing Python environment and configured cache directory.

Full backend regression: **73 tests passed in 324.539 seconds**. Existing cases cover both recording sources, foreign-upload
rejection, rejected/cancelled upload cleanup, isolated fitted artifacts, active
reservation protection, partial batch manifests/downloads, and restart/history
retention. Scientific arithmetic is unchanged; no additional GPU parity run is
required for this HTTP-header/documentation change.

## Limits

No external HTTPS proxy was provisioned or contacted. Its deployment checks must
be completed before public exposure using the documented checklist. Restart
against persistent storage is verified; backup/restore is explicitly outside the
approved scope and was not performed. No automatic artifact cleanup is enabled.

## Review

Pinned review base: `4d174129d62ab6260521980c1d8b937bf6393d03`.
Implementation: `7396980`. Independent fresh Standards and Spec reviewers each
reported zero actionable findings. The Spec reviewer independently reran all four
new access tests successfully. The ticket and map were resolved after the full regression suite passed.
