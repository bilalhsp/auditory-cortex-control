# 07: Download timestamped session and run archives

**What to build:** Export completed validated samples for a session or a run.

**Blocked by:** 06

**Status:** resolved

- [x] Export completed validated samples for a session or a run.
- [x] Use creation timezone for stable timestamped ZIP and folder names, with Pacific fallback for legacy sessions.
- [x] Use stable run suffixes and distinct run and target paths.

[Parent specification](../spec.md)

## Answer

Session and run exports use the session creation timezone, with Pacific fallback, and put validated samples under a timestamped archive folder. All-session exports include old single runs and new batches. Damaged outputs are omitted with a report when other validated samples remain. Browser and API checks verified both export scopes, isolation, and contents.
