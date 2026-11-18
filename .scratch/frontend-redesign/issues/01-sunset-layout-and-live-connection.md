# 01: Sunset Boulevard layout and live connection status

**What to build:** Use the approved palette and typography in a responsive, keyboard-accessible app shell.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Use the approved palette and typography in a responsive, keyboard-accessible app shell.
- [x] Show a labeled green/red connection indicator independent of job state. Detect outages and recovery without redirecting signed-in requests.
- [x] Place the existing sign-out action in a circular account disclosure.

[Parent specification](../spec.md)

## Answer

Applied the approved theme and replaced the sidebar and breadcrumb with a responsive header. The connection indicator checks the signed-in endpoint without credentials and recovers after an interruption. The account disclosure keeps the existing sign-out behavior until ticket 08.

The frontend production build and browser checks against isolated FastAPI passed. Browser coverage included connection loss and recovery, keyboard disclosure and Escape, sign-out, and a 320-pixel viewport. The review findings on stale connection status, GPU indicator ambiguity, contrast, and internal helper visibility were addressed.
