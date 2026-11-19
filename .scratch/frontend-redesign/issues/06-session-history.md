# 06: Browse Session History and view saved results

**What to build:** Show saved sessions from today and prior dates on a separate tab.

**Blocked by:** 02

**Status:** resolved

- [x] Show saved sessions from today and prior dates on a separate tab.
- [x] View legacy and new saved results without resuming or changing the current Workbench session. History works with zero visible GPUs.

[Parent specification](../spec.md)

## Answer

Session History is a separate read-only view. It displays sessions across dates and saved results without switching the current Workbench session or requiring GPUs. Browser checks exercised selecting a separate session during Workbench close and viewing saved results.
