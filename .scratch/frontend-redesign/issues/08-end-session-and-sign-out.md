# 08: End the current session and sign out

**What to build:** End only the current Workbench session even when browsing History.

**Blocked by:** 02

**Status:** resolved

- [x] End only the current Workbench session even when browsing History.
- [x] Confirm before stopping unfinished work and retain completed results.
- [x] Sign out after close is accepted, stay signed in if it fails, and sign out directly if no session is current. Browser disconnect leaves accepted work running.

[Parent specification](../spec.md)

## Answer

The account menu closes only the current Workbench session before signing out. An atomic idle check requires confirmation for active work. Canceling confirmation or a failed close leaves the operator signed in. The browser verified these cases while viewing another session in History.
