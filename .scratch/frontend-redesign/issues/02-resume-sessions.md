# 02: Resume a session and start a new experiment

**What to build:** Resume open incomplete or active sessions across dates.

**Blocked by:** 01

**Status:** resolved

- [x] Resume open incomplete or active sessions across dates.
- [x] Show creation timestamps and recording or experiment context. Persist the browser creation timezone with Pacific fallback.
- [x] Start over creates a new experiment in the current session and preserves accepted work.

[Parent specification](../spec.md)

## Answer

Workbench groups resumable experiments under their session creation timestamp. A session can offer both its active work and its unfinished setups across dates. Start over saves a fresh experiment within the same session immediately, using refreshed GPU capacity and retaining prior experiments and results.

New sessions retain the browser creation timezone. Legacy sessions and invalid timezone values use Pacific time. Backends without the new Workbench capability continue to offer recovery through the existing History controls.

Verified the production build, authenticated session API tests, and browser resume after reload with real FastAPI and SQLite. The browser also verified two resumable experiments in one session, the stored creation timezone, and the layout at 320 pixels.
