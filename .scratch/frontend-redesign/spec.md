# Auditory Cortex Control frontend redesign

Status: ready-for-agent

## Problem Statement

The operator has to interpret ambiguous background-work messages, navigate between controls for the same workflow, and inspect server logs to distinguish lost connectivity from computation. Current work and saved results need clearer entry points. Generation offers separate actions for single and batch jobs, and downloaded archives are hard to identify by session.

## Solution

Provide a Sunset Boulevard Workbench that guides the operator from experiment setup through results, with a visible selection summary, independent connection and job status, and timestamped session activity. Provide a separate Session History for viewing and downloading completed results. Submit one or several targets with the same GPU-limited Generate samples action.

## User Stories

1. As an operator, I want a top-right green LED with a Connected label, so that I can see when the backend is reachable.
2. As an operator, I want a red LED with a Disconnected label when contact is lost, so that I can distinguish connectivity from job state.
3. As an operator, I want a circular account menu, so that I can find the session-ending action.
4. As an operator, I want to resume a session with queued or running work, so that I can return to it after disconnecting.
5. As an operator, I want to resume unfinished setup, so that reloading does not force me to begin again.
6. As an operator, I want to resume eligible work that started on a previous date, so that midnight does not hide it.
7. As an operator, I want recognizable session labels with creation date, time, and available recording or experiment context, so that I can identify my work.
8. As an operator, I want a separate Session History tab, so that I can find today's and prior dates' saved results.
9. As an operator, I want to view completed results without resuming a session, so that I can inspect previous work directly.
10. As an operator, I want a guided flow from setup to results, so that I know what to do next.
11. As an operator, I want to review earlier choices with Back, so that I can check my inputs.
12. As an operator, I want input changes that affect an accepted fit to create another experiment in the same session, so that the original work stays intact.
13. As an operator, I want Start over to begin a fresh experiment within the same session, so that I can try another setup without losing accepted work.
14. As an operator, I want Upload recording selected by default, so that I can add data immediately.
15. As an operator, I want an adjacent Choose existing recording tab, so that I can use stored data in the same workflow step.
16. As an operator, I want to see my selections throughout the workflow, so that I can verify them before submitting.
17. As an operator, I want fit stages and a visible activity cue, so that I can tell when the backend is fitting or writing results.
18. As an operator, I want activity text to remain useful with reduced motion enabled, so that I can follow progress comfortably.
19. As an operator, I want a collapsible session Activity panel, so that I can inspect progress without reading server files.
20. As an operator, I want timestamped activity and actionable error messages, so that I can diagnose delays and failures.
21. As an operator, I want experiment GPU selection to default to the maximum visible count, so that I can use the available capacity immediately.
22. As an operator, I want to select a smaller valid GPU count, so that I can choose how many devices my experiment uses.
23. As an operator, I want to see when a saved GPU count exceeds the current count, so that I can choose a valid count before submitting work.
24. As an operator, I want to see my selected target count and GPU limit, so that I know how many unique unit/objective combinations I can add.
25. As an operator, I want one Generate samples action, so that I can submit all selected targets together.
26. As an operator, I want to see overall generation progress and individual target states, so that I can tell which results are ready.
27. As an operator, I want Generate more after completion, so that I can reuse the fitted model.
28. As an operator, I want to download all completed samples from a session or a single completed run, so that I can choose the scope of an export.
29. As an operator, I want timestamped ZIP and folder names, so that I can recognize a session after extracting it.
30. As an operator, I want archive names to stay stable across devices, so that the same session can be found reliably.
31. As an operator, I want History and downloads to work without visible GPUs, so that I can retrieve existing results.
32. As an operator, I want a single End session and sign out action, so that I can leave without managing two separate controls.
33. As an operator, I want to confirm ending a session with active work, so that I can avoid stopping jobs accidentally.
34. As an operator, I want completed results retained when I end the session, so that I can download them later.
35. As an operator, I want errors closing the session to leave me signed in, so that I can retry.
36. As an operator, I want accepted work to continue if I lose the tunnel or close the browser, so that I can return when connected again.

## Implementation Decisions

- Keep React, TypeScript, and Vite with the existing FastAPI backend and persisted session records. An application session groups experiments and results. In UI copy, use Session for the operator's work and Recording for fit input.
- Make Workbench and Session History distinct tabs. List open sessions with incomplete setup or queued or running jobs under Resume without a date cutoff. History provides read-only results access for current and earlier dates. Viewing History does not change the current Workbench session.
- Use Setup, Recording, Fit, Generate, and Results as the ordered views. Keep the summary on the right on wide screens and inline near the active step on narrow screens. Keep Upload recording and Choose existing recording within Recording.
- Back permits review. When fit inputs change after work has been accepted, create a separate experiment in the current session and require a new fit. Start over also creates a fresh experiment in the current session. Preserve original job provenance, configurations, and results.
- Use Sunset Boulevard with a pale warm surface, deep #264653 text, #e76f51 and #f4a261 accents, and #e9c46a highlights. Use DejaVu Serif Bold for headings and DejaVu Sans for UI text with system fallbacks. Use readable contrast, visible keyboard focus, and reduced-motion support.
- Show transport state and job state separately. Never infer that a disconnected backend has failed or that its jobs have stopped. Show the latest known stage and mark it as such during an outage. Use clear pending, running, and terminal feedback and avoid inventing progress percentages.
- Expose authenticated, timestamped activity from the selected application session using public job stages and safe error explanations. Keep raw worker output, credentials, private server paths, and unrelated session data out of the response. Present events in a collapsible terminal-style panel.
- Default experiment GPU count to the maximum visible count. Enforce a positive count at or below backend visibility for compute submissions. Enforce one or more unique unit/objective pairs per generation request, at most the experiment GPU count. Apply these rules in the UI and on the server. Perform server validation before accepting new work so a stale browser cannot exceed the limit. If visibility drops below a saved count, explain the discrepancy and require a valid selection. Preserve stored results and keep History available with zero GPUs.
- Route one and multiple targets through a single Generate samples action. Show selected count against the limit before submission and show overall and target status afterward. Keep Generate more and Download results available as appropriate. Existing results from the previous single and batch flows remain readable and downloadable.
- Capture the browser's IANA timezone when the session is created. Normalize invalid or unavailable timezone values and missing legacy values to America/Los_Angeles. Keep the creation instant and chosen zone stable. Format readable session labels from them, with recording or experiment context when available. Use a safe timestamp basename such as session-2026-11-18_12-15-10 for both the session ZIP and its top-level folder. Use a stable run suffix for individual run exports. Aggregate completed samples across the session with distinct run and target paths to avoid file collisions. Include only completed, validated outputs in exports.
- End session and sign out applies only to the current Workbench session, including while browsing History. When it has unfinished accepted jobs, confirm cancellation and explain result retention. Canceling the confirmation keeps the session open. Submit closure, wait for backend acceptance, then sign out. If the request fails, keep the operator signed in and display the error. With no current Workbench session, sign out directly. Preserve the existing rule that browser disconnect alone leaves accepted jobs running.

## Testing Decisions

- Approved primary boundary is browser interaction through React and FastAPI with real session persistence and controlled compute. Use the existing acceptance approach with deterministic job progress and outcomes. Follow visible behavior, reloads, and downloads rather than React internals or database layout.
- Exercise the Workbench sequence, both recording sources, persistent summary, Back and Start over, reconnect after a tunnel interruption, resume across dates, History, and both scopes of download. Include saved legacy results, narrow layouts, keyboard operation, and reduced motion.
- Verify that confirming closure stops only the active Workbench session's unfinished work and that the operator is signed out only after acceptance. Cover canceling the dialog, a rejected close request, and closing while viewing another session in History.
- Use focused authenticated API checks for backend-enforced GPU and target limits, changing visibility, activity isolation and redaction, and ZIP naming and contents. Check fallback timezone and stable naming across device timezones. Existing session, access, batch, generation, recording, and release acceptance tests provide prior art.
- Verify scientifically valid fit and generation output through existing checks where compute integration is affected. Routine UI coverage uses controlled compute.

## Out of Scope

- Changes to fitting or generation algorithms, compute runtime packaging, Slurm, SSH tunneling, deployment architecture, and authentication model.
- Public access to raw server or worker logs, multi-operator support, and changing saved result retention policy.
- Replacing historical job records or requiring previously saved experiments to be recomputed to retrieve their results.

## Further Notes

The approved design notes capture the original decisions. The existing session lifecycle and single-operator persistence ADRs still apply. This spec changes the explicit account action while preserving the lifecycle of accepted work during connection loss.
