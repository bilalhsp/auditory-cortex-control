# Auditory cortex control: web migration specification

Status: ready-for-agent
Readiness: published for review; open decisions gate affected implementation. Implementation is not authorized in this documentation phase.

## Problem Statement

Researchers use Streamlit to configure experiments, select recordings, fit encoding models, inspect channel correlations, generate optimized audio, and download results. Presentation, mutable scientific state, storage, and process orchestration are coupled. Shared cached generators, temporary filenames, fitted-model identifiers, and destructive output initialization make concurrent operation unsafe. Site-specific paths and the combined scientific environment complicate deployment.

The scientific computation already exists. Modernizing its interface should preserve that investment and its numerical behavior while establishing maintainable frontend/backend boundaries.

## Solution

Use React + TypeScript + Vite for the frontend and Python + FastAPI for the backend, with uv managing Python environments. Keep a monorepo with separate frontend/ and backend/ directories and independently deployable frontend and backend.

Preserve the configure → upload/select recording → fit → inspect correlations → generate → download workflow. Detect available GPUs on the backend server, display their number in the interface, and allow users to choose a GPU count, defaulting to the maximum available.

Confirmed requirements and proposed technical contracts are distinguished below. Recommendations from the unfinished grilling session are not accepted decisions.

## User Stories

1. As a researcher, I want to configure an experiment, so that fitting and generation use my intended settings.
2. As a researcher, I want to select TIMIT or mVocs, so that I retain both existing stimulus families.
3. As a researcher, I want supported dataset, model, and layer options, so that I cannot choose an invalid combination.
4. As a researcher, I want effective bin width and receptive field displayed, so that I understand fixed scientific settings.
5. As a researcher, I want to upload a recording, so that I can analyze newly collected data.
6. As a researcher, I want to choose an existing recording, so that I can reuse stored sessions.
7. As a researcher, I want invalid recordings rejected clearly, so that I can correct inputs before expensive computation.
8. As a researcher, I want to fit an encoding model, so that generation can target recorded neural responses.
9. As a researcher, I want correlations ranked by channel, so that I can identify useful targets.
10. As a researcher, I want the best channel highlighted, so that I can quickly assess fit quality.
11. As a researcher, I want stretch and one-hot objectives, so that I retain existing generation behavior.
12. As a researcher, I want multiple channel/objective selections, so that I can generate a batch of targets.
13. As a researcher, I want durations of one through four seconds, so that I retain current duration choices.
14. As a researcher, I want one, two, four, eight, or ten clips per target, so that I retain current batch choices.
15. As a researcher, I want the backend's available GPU count shown, so that I know the available capacity.
16. As a researcher, I want to choose how many GPUs to use, so that I control resource consumption.
17. As a researcher, I want the GPU selector to default to the maximum available, so that full capacity requires no extra setup.
18. As a researcher, I want capacity changes explained, so that my resource request is not silently changed.
19. As a researcher, I want a clear no-GPU state, so that I understand when generation is unavailable.
20. As a researcher, I want failures distinguished from success, so that missing clips cannot masquerade as completed results.
21. As a researcher, I want a ZIP of generated WAV files, so that I can use them in my experiment.
22. As a researcher, I want additional batches from an existing fit, so that I avoid unnecessary refitting.
23. As a researcher, I want isolated recordings, fits, and outputs, so that another workflow cannot overwrite my work.
24. As a researcher, I want changed upstream settings to invalidate stale selections, so that I cannot accidentally generate from the wrong fit.
25. As an operator, I want independent frontend/backend releases, so that one component can change without rebuilding the other.
26. As an operator, I want environment-specific runtime configuration, so that deployment does not depend on a researcher's absolute paths.
27. As an operator, I want reproducible uv environments, so that the runtime can be rebuilt and validated.
28. As an operator, I want useful computation logs and failures, so that I can diagnose operational problems.
29. As a maintainer, I want existing scientific logic reused, so that the migration minimizes numerical changes.
30. As a maintainer, I want scientific parity checked before cutover, so that existing functionality is preserved.

Conditional stories, pending the audience, durability, and operational decisions:

31. As a researcher, I want accepted work to survive browser disconnection, so that closing a tab does not lose computation.
32. As a researcher, I want to reconnect to progress and results, so that long-running work remains accessible.
33. As an authorized user, I want access limited to permitted recordings and results, so that research data remains protected.
34. As a researcher, I want cancellation with clear partial-output semantics, so that I can release unused resources safely.
35. As an operator, I want retention and backup policies, so that storage growth and recovery are predictable.

## Implementation Decisions

### Confirmed requirements

- React + TypeScript + Vite frontend; Python + FastAPI backend; uv for Python environments.
- A monorepo with separate frontend and backend directories and independent deployment.
- Preserve existing application functionality while minimizing unnecessary rewrites.
- Detect available GPUs on the backend server, show the count, allow GPU-count selection, and default to the maximum available.
- Document and refine the architecture before implementation.

### Observed preservation baseline

- The active UI offers ucdavisBAK, deepspeech2, and layer 2. TIMIT and mVocs are selectable; bin width is fixed at 50 ms and receptive field at 200 ms. The untrained-model control is disabled.
- Recording sources are uploaded files and existing dataset sessions. Fitting performs three-fold TRF cross-validation and returns per-channel correlations, displayed in ranked order with a best-channel summary.
- Generation supports stretch and one-hot objectives, 1–4-second durations, and 1/2/4/8/10 clips per target. Defaults are 2 seconds and 4 clips.
- Output is RMS-normalized and clipped, then saved as 16-kHz PCM16 WAV. The UI provides a ZIP download, not audio playback.
- The active UI launches local worker subprocesses, one per target, and waits. Separate multi-node scheduling code exists but is inactive in this path.
- The reusable generator and sampler contain fitting and generation logic. Data assembly, neural datasets, feature extraction, encoding, and diffusion are Python scientific modules suitable for reuse.
- Current hazards include a globally cached mutable generator, shared temporary files, uploaded-fit identity zero, unchecked worker exit codes, and deletion of an existing date-named output directory on UI initialization.
- Broader research-package capabilities exist; their inclusion as web features is an unresolved scope decision.

### Proposed responsibility boundaries — not accepted architecture

| Component | Responsibility |
| --- | --- |
| Frontend | Forms, draft selections, capability/GPU display, validation feedback, correlations, operation status, downloads |
| FastAPI | Input validation, resource identities, capability queries, upload and command handling, status/results, eventual access enforcement |
| Workflow services | Coordinate recording registration, fitting, generation, artifacts, and operation state using existing scientific entry points |
| Compute execution | Run expensive loading, fitting, and generation outside HTTP request execution, retaining existing worker configuration interfaces where practical |
| Persistence | Operational metadata and artifact references, separate from scientific checkpoints, datasets, and audio files |

- Browser state is not authoritative for fitted models or completed operations.
- Separate Streamlit caching from reusable resource construction. Immutable model weights may be cached safely; mutable recording-specific fitting state must be isolated.
- Prefer adapting existing generator, sampler, and worker boundaries over rewriting numerical code. Physical relocation of the scientific package is not necessary to establish an API boundary; preserve existing imports and CLI workflows initially unless explicitly changed.
- Isolated compute execution can still run on the backend server. A separate process does not imply remote/cloud GPUs or a separate deployment product.

### Proposed resource model — pending review

| Resource | Required information and relationships |
| --- | --- |
| Experiment | Effective stimulus family, dataset, model, layer, bin width, receptive field, configuration revision |
| Recording | Opaque identity, uploaded/existing source, display name, dataset context, validation state, artifact reference |
| Fit | Immutable experiment revision and recording reference, fitted artifact, channel identities, correlations, completion state |
| Generation request | Immutable fit reference, channel/objective targets, duration, clip count, requested GPU count, seed, effective sampler settings |
| Job | Operation kind, resource reference, lifecycle state, timestamps, execution attempt, failure details, assigned resources, measurable progress |
| Artifact | Producing operation, media type, size, storage reference, integrity metadata, provenance |

- Distinguish recording sessions, browser sessions, fits, and execution jobs. Uploaded recordings must not share fitted-model storage keys merely because legacy computation uses session zero internally.
- Return correlations as records with explicit channel identities and numeric values, preserving signed/non-integer identifiers without JSON object-key coercion.
- Snapshot recording/model identity, effective configuration, random seed, code version, and environment identity. Preserve current sampler behavior until a separate scientific change is approved.
- Generation references an exact completed fit. Configuration or recording changes invalidate dependent frontend selections rather than silently changing that fit.

### Proposed HTTP operations — exact routes and schemas unselected

| Operation | Input | Observable result |
| --- | --- | --- |
| Read capabilities | None | Supported settings and timestamped GPU availability |
| Create/read experiment | Validated scientific settings | Identity and effective configuration |
| List recordings | Dataset/stimulus context | Compatible existing recordings |
| Upload recording | File and context | Identity and validation result, or validation job |
| Request fit | Experiment revision and recording identity | Fit identity and computation job identity |
| Read fit | Fit identity | State, channels, ranked correlations, artifact availability |
| Request generation | Fit, targets, duration, count, GPU count | Generation identity and computation job identity |
| Read job | Job identity | State, timestamps, meaningful progress, failure details |
| Read generation | Generation identity | Per-target outcome and output manifest |
| Download bundle | Permitted artifact identity | ZIP of that generation's intended outputs |

- Proposed commands return accepted responses with stable identities rather than holding an HTTP request open for computation. Distinguish invalid inputs, absent resources, unavailable capacity, and execution failures.
- Proposed initial progress transport is polling. SSE/WebSockets remain alternatives if established latency requirements justify them.
- Browser requests use opaque identities, never arbitrary filesystem paths or physical GPU device identifiers.
- Define API compatibility before independent releases. Frontend API configuration must be deployment-configurable and contain no server secrets.

### GPU behavior: confirmed requirement and proposed semantics

Confirmed behavior: detect available GPUs on the backend server, display their count, provide a count selector, and default to the maximum available.

Proposed details requiring review:

1. Distinguish GPUs visible/permitted to the backend runtime from GPUs currently allocatable to this application. Expose both counts with a timestamp when useful; default the selector to allocatable capacity. Hardware detection alone is not a reservation system.
2. Respect runtime GPU visibility and scheduler allocations. Do not multiply the local count by remote host count or expose cluster-wide inventory as local capacity.
3. Allow integer selections from 1 to the available maximum. With zero available GPUs, show an explicit unavailable state and disable GPU generation; no CPU fallback is assumed.
4. Initialize the selection to the maximum when capabilities first load. Preserve a valid explicit selection on refresh. If capacity drops below it, require correction rather than silently reducing the request.
5. Recheck capacity on submission and allocation. Prevent incompatible concurrent jobs from reserving the same exclusive GPU; determine how external workloads are accounted for.
6. Resolve actual device identities server-side. The browser requests a count, not device numbers.
7. Use selected count as a concurrent-worker limit; run excess selected targets in waves. If targets are fewer than selected GPUs, explain potential idle devices. Splitting a single target across multiple GPUs is not implied.
8. Rejecting versus queueing a temporarily unsatisfiable request remains open. Memory admission limits must be measured; a device count does not establish sufficient VRAM.

### Proposed computation lifecycle and storage behavior

- Suggested job states are queued, running, succeeded, and failed. Add cancellation states only if cancellation is accepted, with a defined point after which cancellation is no longer possible.
- Declare success only after checking worker exit status and validating an output manifest. Distinguish partial completion from total success and define partial-download policy.
- Give each operation isolated configuration and output locations. Publish complete artifacts atomically; avoid filename-count races, client-controlled storage names, and destructive initialization.
- Validate upload format, size, and dataset compatibility before expensive parsing where possible. Never load user-supplied serialized model objects as trusted checkpoints.
- If durable jobs are accepted, persist accepted state before execution, reconcile interrupted jobs after restart, and make retries explicit isolated attempts. An ambiguous HTTP failure must not automatically duplicate expensive computation.
- Artifact ownership/sharing, retry limits, cancellation, idempotency, retention, backups, and job recovery guarantees remain unresolved.

### Proposed deployment and environment approach

- Independently build frontend assets and the API runtime. A shared browser origin through a reverse proxy is compatible with independent deployment; separate origins require deliberate CORS and credential configuration.
- Keep heavy GPU model loading out of API startup where feasible. Whether API and compute use one environment or distinct uv environments is open.
- Existing Python metadata is constrained to 3.8 and Linux x86_64; dependencies combine notebooks, Streamlit, API, scientific packages, and CUDA libraries. Select target Python and native dependencies through compatibility validation rather than assume uv resolves CUDA compatibility.
- Derive direct dependencies from actual use, retain a reproducible legacy environment, and validate the scientific stack before removing old pins or native-library bootstrap behavior.
- Replace site-specific absolute paths and import-time overrides with explicit configuration for datasets, model weights, cache, metadata, and outputs.
- Expose API readiness separately from GPU readiness. Existing results and metadata should remain readable when no GPU is available.
- Log operation identity, attempt, duration, allocation, worker exit status, and useful errors without exposing recordings or credentials.
- Hosting, TLS termination, authentication provider, metadata database, file store, queue, quotas, backup, and monitoring products are unselected. Do not assume Redis, Kubernetes, or object storage is required.

### Proposed migration sequence and gates

1. Confirm preservation scope and inventory the active workflow. Capture representative recordings, settings, fit results, and outputs before changing orchestration.
2. Isolate recording/fit/generation identities and files. Extract workflow orchestration and reusable construction from Streamlit handlers while retaining scientific algorithms and legacy compatibility.
3. Establish the approved API/compute environment arrangement, configurable storage, GPU detection/allocation, and worker failure reporting.
4. Build the API contracts for capabilities, recordings, fits, generation, status, and downloads behind the approved primary test seam.
5. Build React workflow parity, including displayed capacity, maximum-default GPU selection, upstream invalidation, and actionable errors.
6. Compare old and new workflows using isolated fixtures and outputs; exercise failure handling, independent deployment, and resource isolation.
7. Pilot with intended users. Cut over only after agreed parity, latency, and operational criteria pass. Retain the legacy launcher/environment for an agreed rollback window, then archive Streamlit after sign-off.

## Testing Decisions

### Boundary selection remains open

The user rejected the initial primary browser → FastAPI → persisted-job/result acceptance seam. No replacement has been selected. Alternatives presented are:

- FastAPI HTTP contract with real application services and controlled compute.
- Python workflow service boundary, with separate thin API and UI checks.
- Existing scientific functions and worker CLI.

Select the primary boundary before committing to a test architecture. Prefer the highest approved seam and existing scientific entry points; avoid scattering redundant tests through internal helpers. Test observable outcomes, not implementation structure or mirrored algorithms.

### Required acceptance coverage; harness pending boundary selection

- Both recording sources, current settings, both objectives, ranked correlations, repeated generation from a fit, and ZIP downloads remain usable.
- Capacity cases of zero, one, and multiple GPUs; maximum default; valid smaller selections; invalid counts; stale snapshots; allocation conflicts; no oversubscription.
- Invalid recording/settings/channel/empty target selection produce useful failures; channel identities survive serialization.
- Concurrent or sequential workflows cannot overwrite or delete another fit/output. Opening the UI cannot remove existing results.
- Nonzero worker exit, missing/partial output, failed model loading, and GPU loss cannot produce false success.
- If durability is accepted: reconnect, API restart, worker interruption, retry, and repeated-submission behavior match specified guarantees.
- If multiple users are accepted: access rules cover recordings, fits, status, artifacts, and downloads, including contention.
- Compare old and new scientific outputs using identical fixtures, checkpoints, effective settings, and environments. Establish tolerances from measured repeatability; do not assume byte-identical GPU output or invent numerical thresholds.
- Verify WAV sample rate/format, clip count/duration, complete manifests, and ZIP membership separately from waveform similarity.
- Verify reproducible uv resolution/builds in the approved runtime and independently deployed frontend/API connectivity, configuration, and artifact access.

Prior art consists of scientific scripts/notebooks, existing fitting and sampling entry points, and the standalone worker CLI. Static inspection did not identify a dedicated web/API acceptance suite. Deprecated experiment scripts named as tests are not an established migration suite. No GPU runs, environment installation, dependency resolution, or scientific tests have been performed for this specification.

## Out of Scope

- Application implementation, infrastructure provisioning, dependency installation, and deployment during this documentation phase.
- Rewriting scientific algorithms, retraining models, changing objectives or normalization solely for the web migration.
- New audio playback, public signup, billing, research dashboards, or unrestricted model/dataset expansion without an explicit scope decision.
- Distributed training or splitting one target across GPUs as an automatic consequence of GPU-count selection.
- Selecting vendors, queues, identity providers, or databases before their requirements are resolved.

## Further Notes

### Unresolved decisions and implementation gates

| Decision | Status / recommendation |
| --- | --- |
| Scope beyond current UI | Open; recommend preserving existing CLI/notebook compatibility and migrating the exposed workflow first |
| Audience and concurrency | Open; authenticated shared lab was recommended, not accepted |
| Live experiment needs | Open; establish initialization, fitting, generation, and queue latency targets |
| Compute placement | Backend-server detection confirmed; dedicated server versus scheduler-managed backend, and any remote execution, remain open |
| Meaning of available GPU | Open; recommend permitted and currently allocatable devices with allocation-time revalidation |
| Target/GPU scheduling | Open; recommend bounded workers and waves for excess targets |
| Jobs, recovery, cancellation | Open; durable accepted work was recommended, not accepted |
| Storage and retention | Open; distinguish operational metadata from scientific artifacts |
| Authentication and sharing | Open; determine audience and data sensitivity before selecting technology |
| Python/CUDA compatibility | Open; validate separately from UI replacement |
| Primary test boundary | Initial browser-led proposal rejected; alternative pending |
| Cutover and rollback | Open; agree parity thresholds, pilot criteria, and rollback window |

The earlier discovery document is preserved alongside this specification. No ADRs were accepted during the unfinished grilling session, and no domain glossary terms were resolved; this specification does not fabricate either. Record substantive architectural trade-offs when decisions are made.

The ready-for-agent status is the requested local tracker classification, not evidence that open design gates are settled. Work dependent on those gates requires their resolution first. No implementation tickets are created by this synthesis.

## Comments

- Synthesized from the conversation and static source inspection.
- User confirmed backend GPU detection, visible count, GPU-count selection, and maximum-available default.
- User rejected the initial primary testing seam; replacement remains pending.
