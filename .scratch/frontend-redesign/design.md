# Frontend redesign

Status: approved

## Requested direction

- Put backend connection status and the account menu in the top right.
- Give current work and session history separate views. Use readable, timestamped session labels. History provides access to saved results and downloads.
- Guide the operator through experiment setup, recording selection, fitting, generation, and results. Keep a summary of selections visible.
- Put Upload recording (default) and Choose existing recording in the same step.
- Show fit stages with activity feedback, and provide a terminal-style activity panel.
- Use one generation action, with selections capped by the GPU limit. Offer Generate more and Download results after completion.
- Use the session timestamp in downloaded ZIP and folder names. Provide a single End session and sign out action.

## Existing decisions

An application session groups work and results. A recording session is input data for a fit. The interface can call these “Session” and “Recording” respectively. Accepted work continues across browser disconnects. Explicitly closing a session stops its active work and preserves completed results, per ADR 0001.

## Workbench layout

Use Workbench and Session History tabs. Give the current step the main column, keep a selection summary on the right, and place a collapsible Activity panel beneath the workflow. On narrow screens, place the summary inline near the current step. Show connection state separately from job progress.

## Confirmed in round 1

- Resume open sessions with unfinished setup as well as sessions with queued or running work. A date boundary does not hide resumable work. History includes today's results and prior dates.
- Use the browser's system timezone for session names, with Pacific time as fallback. Label sessions by creation time and add recording or experiment context when available.
- Back reviews earlier selections. Start over opens a fresh setup and preserves accepted work. Provide one combined action to close the session and sign out. Confirm when work is active, stop unfinished work in the session being closed, and retain completed results. Browser disconnects continue to leave jobs running.
- The collapsible Activity panel shows session-specific timestamped stages and actionable safe error messages. Raw diagnostics stay on the server.
- Experiment GPU count defaults to the maximum visible count and cannot exceed it. Unique unit/objective selections are capped by the experiment GPU count. Generate samples submits all selected combinations together.
- Apply Sunset Boulevard to the approved workbench layout. Use #264653 for dark text, #e76f51 and #f4a261 for accents, #e9c46a for highlights, and a pale warm surface. Use DejaVu Serif Bold for headings and DejaVu Sans for interface text, with system fallbacks. Reserve the green/red LED and its visible text label for connection state. Keep the fit stage visible during processing, with motion that respects reduced-motion preferences.

## Confirmed in round 2

- End session and sign out applies only to the current Workbench session, including when the operator is browsing History. With no current session, the action signs out. Confirm active work before requesting closure. Wait for the backend to accept the close request before signing out, and keep the operator signed in with an error if the request fails.
- If a saved experiment's GPU count exceeds current visibility, preserve its results and require a valid count before submitting new work. Recheck visibility on the backend at submission. With zero visible GPUs, History and downloads remain available.
- Back allows review of earlier inputs. Changing fit inputs after accepted work creates a new experiment within the current session and requires a new fit. Start over also starts a fresh experiment within the current session. Keep previously accepted jobs and results associated with their original experiments.
- History can export all completed samples from a session, and provides an individual download for each completed run. Use the session creation timezone for stable ZIP and top-level folder names across devices. For legacy sessions without a stored timezone, use Pacific time. Use the session basename plus a run-specific suffix for individual exports.
- Use “Session” for the operator's work and “Recording” for the data being fitted. See the [glossary](../../GLOSSARY.md).
