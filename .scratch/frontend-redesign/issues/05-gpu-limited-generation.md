# 05: Generate samples with one GPU-limited action

**What to build:** Default to maximum visible GPUs, enforce current visibility and limit unique targets to the experiment GPU count in UI and API.

**Blocked by:** 03

**Status:** resolved

- [x] Default to maximum visible GPUs, enforce current visibility and limit unique targets to the experiment GPU count in UI and API.
- [x] Handle reduced or zero visibility without losing results.
- [x] Send all selected targets through Generate samples, show per-target and overall state, and offer Generate more and downloads on completion.

[Parent specification](../spec.md)

## Answer

Setup defaults to the backend’s visible GPU count. Generate samples submits distinct channel/objective targets through the bounded API. The API checks experiment limits and available devices before accepting new work. Lost responses can be replayed with the original command. Legacy batch submission retains its existing contract for older clients. Browser and API checks covered generation, limits, replay, and results.
