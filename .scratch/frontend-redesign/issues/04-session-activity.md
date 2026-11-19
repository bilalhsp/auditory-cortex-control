# 04: Show session activity in the browser

**What to build:** Show timestamped session stages and safe actionable errors in a collapsible Activity panel.

**Blocked by:** 02

**Status:** resolved

- [x] Show timestamped session stages and safe actionable errors in a collapsible Activity panel.
- [x] Limit authenticated activity to the selected session and exclude raw diagnostics, credentials and server paths.

[Parent specification](../spec.md)

## Answer

The Workbench Activity panel polls only the current application session while expanded. It shows the latest 100 timestamped events using approved messages, including batch target stages and stop requests. Auth, cross-session isolation, and browser stages were checked.
