import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { ApiError, discoverApiBaseUrl, makeApi } from "./api";
import type { Capabilities, Experiment } from "./api";
import RecordingSelector from "./RecordingSelector";
import RecordingUpload from "./RecordingUpload";
import FitPanel from "./FitPanel";
import SessionWorkbench from "./SessionWorkbench";
import SessionActivity from "./SessionActivity";
import SessionHistory from "./SessionHistory";
import BackendConnection from "./BackendConnection";
import WorkspaceHeader from "./WorkspaceHeader";

export default function App() {
  const [step, setStep] = useState<"Setup" | "Recording" | "Fit">("Setup");
  const [recordingSource, setRecordingSource] = useState<"upload" | "existing">("upload");
  const [submissionBusy, setSubmissionBusy] = useState(false);
  const [resumeCount, setResumeCount] = useState(0);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [api, setApi] = useState<ReturnType<typeof makeApi> | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [family, setFamily] = useState("");
  const [gpuCount, setGpuCount] = useState(0);
  const [explicitGpu, setExplicitGpu] = useState(false);
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (experiment?.session_id) setSessionId(experiment.session_id);
  }, [experiment?.session_id]);
  function reportError(cause: unknown) {
    setError(
      cause instanceof Error
        ? cause.message
        : "Could not connect to the server.",
    );
    if (cause instanceof ApiError && cause.status === 401) {
      setSessionId(null);
      setApi(null);
      setCaps(null);
      setExperiment(null);
    }
  }
  async function login(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const baseUrl = await discoverApiBaseUrl();
      const connection = makeApi(username, password, baseUrl);
      const result = await connection.capabilities();
      setApi(connection);
      setStep("Setup");
      setExperiment(null);
      setCaps(result);
      setFamily(result.settings.stimulus_families[0]);
      setGpuCount(result.gpu.available_count);
      setExplicitGpu(false);
      setPassword("");
    } catch (cause) {
      reportError(cause);
    } finally {
      setBusy(false);
    }
  }
  async function refresh(resetGpu = false) {
    if (!api) return;
    setBusy(true);
    setError("");
    try {
      const result = await api.capabilities();
      setCaps(result);
      if (resetGpu || !explicitGpu) {
        setGpuCount(result.gpu.available_count);
      }
    } catch (cause) {
      reportError(cause);
    } finally {
      setBusy(false);
    }
  }
  async function save(event: FormEvent) {
    event.preventDefault();
    if (setupUnchanged) {
      setStep("Recording");
      return;
    }
    await createExperiment(family, gpuCount);
  }
  async function createExperiment(nextFamily: string, nextGpuCount?: number) {
    if (!api || !caps || busy) return;
    setBusy(true);
    setError("");
    try {
      const current = nextGpuCount === undefined ? await api.capabilities() : caps;
      const count = nextGpuCount ?? current.gpu.available_count;
      setCaps(current);
      setExperiment(
        await api.createExperiment(
          {
            stimulus_family: nextFamily,
            dataset_name: caps.settings.datasets[0],
            model_name: caps.settings.models[0],
            layer_id: caps.settings.layers[0],
            bin_width_ms: caps.settings.bin_width_ms,
            receptive_field_ms: caps.settings.receptive_field_ms,
            gpu_count: count,
          },
          sessionId,
        ),
      );
      setRecordingSource("upload");
      setStep("Recording");
      setFamily(nextFamily);
      setGpuCount(count);
      return true;
    } catch (cause) {
      reportError(cause);
    } finally {
      setBusy(false);
    }
  }
  function logout() {
    setSessionId(null);
    setApi(null);
    setCaps(null);
    setExperiment(null);
    setPassword("");
    setError("");
  }
  function canFit(chosen: Experiment | null): boolean {
    return Boolean(chosen?.recording && caps?.fit?.supported_sources.includes(chosen.recording.source));
  }
  function resume(experiment: Experiment) {
    setStep(canFit(experiment) ? "Fit" : "Recording");
    setRecordingSource("upload");
    setResumeCount((count) => count + 1);
    setSessionId(experiment.session_id);
    setExperiment(experiment);
    setFamily(experiment.configuration.stimulus_family);
    setGpuCount(experiment.configuration.gpu_count);
    setExplicitGpu(true);
  }
  function closed(id: string) {
    setSessionId((current) => (current === id ? null : current));
    setExperiment((current) => (current?.session_id === id ? null : current));
  }
  async function newLegacySession() {
    setSessionId(null);
    setExperiment(null);
    setFamily(caps?.settings.stimulus_families[0] ?? "");
    setGpuCount(caps?.gpu.available_count ?? 0);
    setExplicitGpu(false);
    setStep("Setup");
    await refresh(true);
  }
  async function startOver() {
    if (!caps || !sessionId) return;
    if (await createExperiment(caps.settings.stimulus_families[0])) {
      setStep("Setup");
      setExplicitGpu(false);
    }
  }
  function recordingSelected(updated: Experiment) {
    setExperiment((current) =>
      current?.id === experiment?.id && current != null &&
      updated.session_id === current.session_id &&
      (current.id !== updated.id || updated.recording_revision >= current.recording_revision)
        ? updated
        : current,
    );
  }
  const activeStep = experiment ? step : "Setup";
  const navigatingLocked = busy || submissionBusy;
  const available = caps?.gpu.available_count ?? 0;
  const setupUnchanged = experiment?.configuration.stimulus_family === family &&
    experiment.configuration.gpu_count === gpuCount;
  const canChangeRecording = !experiment?.recording || caps?.fit?.preserve_fits_on_recording_change;
  const invalidGpu = gpuCount > available || (gpuCount === 0 && available > 0);

  return (
    <div className="shell">
      <div className="workspace">
        <WorkspaceHeader key={api ? "signed-in" : "signed-out"} signedIn={Boolean(api)} checkConnection={api?.checkConnection} username={username} busy={navigatingLocked} onSignOut={logout} />
        <main>
          {api && caps?.sessions?.supported && (
            <>
              {caps.sessions.resumable && <SessionWorkbench
                api={api}
                sessionId={sessionId}
                experimentId={experiment?.id}
                busy={navigatingLocked}
                onResume={resume}
                onStartOver={startOver}
              />}
              <SessionHistory
                api={api}
                currentExperiment={experiment}
                onClosed={closed}
                onResume={caps.sessions.resumable ? undefined : resume}
                onNew={caps.sessions.resumable ? undefined : newLegacySession}
              />
            </>
          )}
          <h1>
            {api ? ({ Setup: "Set up your experiment", Recording: "Choose a recording", Fit: "Fit the encoding model" })[activeStep] : "Connect to your research workspace"}
          </h1>
          {(!api || activeStep === "Setup") && <p className="intro">
            Configure an encoding model and the compute resources for your next
            experiment.
          </p>}
          {error && (
            <div className="notice error" role="alert">
              {error}
            </div>
          )}
          {!api || !caps ? (
            <section className="card login-card">
              <BackendConnection />
              <div className="section-heading">
                <span className="step">01</span>
                <div>
                  <h2>Operator sign in</h2>
                  <p>Connect to your research backend.</p>
                </div>
              </div>
              <form onSubmit={login}>
                <label>
                  Username
                  <input
                    autoComplete="username"
                    required
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                  />
                </label>
                <label>
                  Password
                  <input
                    type="password"
                    autoComplete="current-password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                </label>
                <button className="primary" disabled={busy}>
                  {busy ? "Connecting…" : "Sign in →"}
                </button>
              </form>
            </section>
          ) : (
            <>
              <ol className="workflow guided-steps" aria-label="Experiment workflow">
                {(["Setup", "Recording", "Fit"] as const).map((item) => (
                  <li key={item} aria-current={activeStep === item ? "step" : undefined}>{item}</li>
                ))}
              </ol>
              <div className="guided-layout">
                <aside className="selection-summary" aria-labelledby="selection-heading">
                  <h2 id="selection-heading">Your selections</h2>
                  <dl>
                    <div><dt>Stimulus family</dt><dd>{family}</dd></div>
                    <div><dt>Neural dataset</dt><dd>{experiment?.configuration.dataset_name ?? caps.settings.datasets[0]}</dd></div>
                    <div><dt>Encoding model</dt><dd>{experiment?.configuration.model_name ?? caps.settings.models[0]}</dd></div>
                    <div><dt>GPUs selected</dt><dd>{gpuCount}</dd></div>
                    <div><dt>Recording</dt><dd>{experiment?.recording?.display_name ?? "Not selected"}</dd></div>
                  </dl>
                </aside>
                <div className="guided-content">
                  {activeStep === "Setup" && <form onSubmit={save}>
                    <div className="setup-grid">
                      <section className="card">
                        <div className="section-heading">
                          <span className="step">01</span>
                          <div>
                            <h2>Experiment configuration</h2>
                            <p>Choose the stimulus family for your recording.</p>
                          </div>
                        </div>
                        <fieldset disabled={busy}>
                          <legend>Stimulus family</legend>
                          <div className="family-options">
                            {caps.settings.stimulus_families.map((option) => (
                              <label
                                className={
                                  family === option ? "family selected" : "family"
                                }
                                key={option}
                              >
                                <input
                                  type="radio"
                                  name="family"
                                  value={option}
                                  checked={family === option}
                                  onChange={() => {
                                    setFamily(option);
                                  }}
                                />
                                <strong>{option}</strong>
                                <small>
                                  {option === "TIMIT"
                                    ? "Speech stimuli"
                                    : "Vocalization stimuli"}
                                </small>
                              </label>
                            ))}
                          </div>
                          <div className="fields">
                            <label>
                              Neural dataset
                              <select value={caps.settings.datasets[0]} disabled>
                                {caps.settings.datasets.map((value) => (
                                  <option key={value}>{value}</option>
                                ))}
                              </select>
                            </label>
                            <label>
                              Encoding model
                              <select value={caps.settings.models[0]} disabled>
                                {caps.settings.models.map((value) => (
                                  <option key={value}>{value}</option>
                                ))}
                              </select>
                            </label>
                          </div>
                          <div className="fixed-settings">
                            <div>
                              <small>MODEL LAYER</small>
                              <strong>{caps.settings.layers[0]}</strong>
                            </div>
                            <div>
                              <small>BIN WIDTH</small>
                              <strong>
                                {caps.settings.bin_width_ms} <span>ms</span>
                              </strong>
                            </div>
                            <div>
                              <small>RECEPTIVE FIELD</small>
                              <strong>
                                {caps.settings.receptive_field_ms} <span>ms</span>
                              </strong>
                            </div>
                          </div>
                        </fieldset>
                      </section>
                      <section className="card">
                        <div className="section-heading">
                          <span className="step">02</span>
                          <div>
                            <h2>Compute resources</h2>
                            <p>Detected on your backend server.</p>
                          </div>
                        </div>
                        <div className="capacity">
                          <div>
                            {available
                              ? "Compute available"
                              : "Compute unavailable"}
                          </div>
                          <strong>
                            {available}
                            <small> / {caps.gpu.visible_count} GPUs</small>
                          </strong>
                        </div>
                        <label>
                          GPUs to use
                          <select
                            aria-describedby="gpu-help"
                            disabled={busy}
                            value={gpuCount}
                            onChange={(e) => {
                              setGpuCount(Number(e.target.value));
                              setExplicitGpu(true);
                            }}
                          >
                            {invalidGpu && (
                              <option value={gpuCount} disabled>
                                {gpuCount} — no longer available
                              </option>
                            )}
                            {available === 0 ? (
                              <option value={0}>No GPUs available</option>
                            ) : (
                              Array.from(
                                { length: available },
                                (_, i) => i + 1,
                              ).map((count) => (
                                <option key={count} value={count}>
                                  {count}
                                  {count === available
                                    ? " — maximum available"
                                    : ""}
                                </option>
                              ))
                            )}
                          </select>
                        </label>
                        <p id="gpu-help" className="help">
                          {available
                            ? "Defaults to all available GPUs. Choose fewer to limit resource use."
                            : "You can save your setup. GPU generation is unavailable until capacity returns."}
                        </p>
                        {invalidGpu && (
                          <div className="notice warning" role="alert">
                            Capacity changed. Select a currently available GPU count
                            to continue.
                          </div>
                        )}
                        {caps.gpu.status === "unavailable" && (
                          <div className="notice warning" role="status">
                            GPU detection is unavailable. The API is reachable;
                            refresh capacity to retry.
                          </div>
                        )}
                        <div className="capacity-footer">
                          <small>
                            Checked{" "}
                            {new Date(caps.gpu.observed_at).toLocaleTimeString()}
                          </small>
                          <button
                            type="button"
                            className="text-button"
                            onClick={() => void refresh()}
                            disabled={busy}
                          >
                            {busy ? "Checking…" : "↻ Refresh capacity"}
                          </button>
                        </div>
                      </section>
                    </div>
                    <div className="action-bar guided-actions">
                      <button className="primary" disabled={busy || (invalidGpu && !setupUnchanged)}>
                        {busy ? "Saving…" : "Continue to recording"}
                      </button>
                    </div>
                  </form>}
                  {activeStep === "Recording" && experiment && (
                    <section className="card" aria-label="Recording source">
                      {canChangeRecording && caps.upload && <div className="source-options" role="group" aria-label="Recording method">
                        <button type="button" disabled={submissionBusy} aria-pressed={recordingSource === "upload"}
                          onClick={() => setRecordingSource("upload")}>Upload recording</button>
                        <button type="button" disabled={submissionBusy} aria-pressed={recordingSource === "existing"}
                          onClick={() => setRecordingSource("existing")}>Choose existing recording</button>
                      </div>}
                      {!canChangeRecording ? (
                        <p className="help">This recording is already attached. Start another experiment to use a different recording on this backend.</p>
                      ) : recordingSource === "upload" && caps.upload ? (
                        <RecordingUpload key={experiment.id} api={api} experiment={experiment}
                          maxBytes={caps.upload.max_bytes} onSelected={recordingSelected} onBusy={setSubmissionBusy} />
                      ) : (
                        <RecordingSelector key={experiment.id} api={api} experiment={experiment}
                          onSelected={recordingSelected} onBusy={setSubmissionBusy} />
                      )}
                      <div className="action-bar">
                        <button type="button" className="text-button" disabled={submissionBusy} onClick={() => setStep("Setup")}>Back</button>
                        <button type="button" className="primary" disabled={!canFit(experiment) || submissionBusy}
                          onClick={() => setStep("Fit")}>Continue to fit</button>
                      </div>
                    </section>
                  )}
                  {activeStep === "Fit" && (
                    <button type="button" className="text-button guided-back" disabled={submissionBusy} onClick={() => setStep("Recording")}>Back</button>
                  )}
                  {activeStep === "Fit" && experiment && caps.fit && canFit(experiment) && (
                      <FitPanel
                        key={`fit-${experiment.id}-${experiment.recording_revision}-${resumeCount}`}
                        api={api}
                        experiment={experiment}
                        seed={caps.fit.default_seed}
                      onBusy={setSubmissionBusy}
                        generationOptions={caps.generation}
                        batchEnabled={Boolean(caps.batch?.supported)}
                        recoveryEnabled={Boolean(caps.sessions?.supported)}
                      />
                    )}
                </div>
              </div>
              {caps.sessions?.activity && sessionId && <SessionActivity key={sessionId} api={api} sessionId={sessionId} />}
            </>
          )}
        </main>
        <footer>
          Auditory Cortex Control <span>Experiment workspace</span>
        </footer>
      </div>
    </div>
  );
}
