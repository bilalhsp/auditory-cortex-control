import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { ApiError, discoverApiBaseUrl, makeApi } from "./api";
import type { Capabilities, Experiment } from "./api";
import RecordingSelector from "./RecordingSelector";
import RecordingUpload from "./RecordingUpload";
import FitPanel from "./FitPanel";
import SessionWorkbench from "./SessionWorkbench";
import SessionHistory from "./SessionHistory";
import BackendConnection from "./BackendConnection";
import WorkspaceHeader from "./WorkspaceHeader";

export default function App() {
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
  async function refresh() {
    if (!api) return;
    setBusy(true);
    setError("");
    try {
      const result = await api.capabilities();
      setCaps(result);
      if (!explicitGpu) {
        setGpuCount(result.gpu.available_count);
        if (result.gpu.available_count !== gpuCount) setExperiment(null);
      } else if (
        gpuCount > result.gpu.available_count ||
        (gpuCount === 0 && result.gpu.available_count > 0)
      ) {
        setExperiment(null);
      }
    } catch (cause) {
      reportError(cause);
    } finally {
      setBusy(false);
    }
  }
  async function save(event: FormEvent) {
    event.preventDefault();
    await createExperiment(family, gpuCount);
  }
  async function createExperiment(nextFamily: string, nextGpuCount: number) {
    if (!api || !caps || busy) return;
    setBusy(true);
    setError("");
    try {
      setExperiment(
        await api.createExperiment(
          {
            stimulus_family: nextFamily,
            dataset_name: caps.settings.datasets[0],
            model_name: caps.settings.models[0],
            layer_id: caps.settings.layers[0],
            bin_width_ms: caps.settings.bin_width_ms,
            receptive_field_ms: caps.settings.receptive_field_ms,
            gpu_count: nextGpuCount,
          },
          sessionId,
        ),
      );
      setFamily(nextFamily);
      setGpuCount(nextGpuCount);
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
  function resume(experiment: Experiment) {
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
  function newLegacySession() {
    setSessionId(null);
    setExperiment(null);
    setFamily(caps?.settings.stimulus_families[0] ?? "");
    setGpuCount(caps?.gpu.available_count ?? 0);
    setExplicitGpu(false);
    setError("");
  }
  async function startOver() {
    if (!caps || !sessionId) return;
    if (await createExperiment(caps.settings.stimulus_families[0], caps.gpu.available_count)) {
      setExplicitGpu(false);
    }
  }
  function recordingSelected(updated: Experiment) {
    setExperiment((current) =>
      current?.id === updated.id &&
      updated.recording_revision >= current.recording_revision
        ? updated
        : current,
    );
  }
  const available = caps?.gpu.available_count ?? 0;
  const invalidGpu = gpuCount > available || (gpuCount === 0 && available > 0);

  return (
    <div className="shell">
      <div className="workspace">
        <WorkspaceHeader key={api ? "signed-in" : "signed-out"} signedIn={Boolean(api)} checkConnection={api?.checkConnection} username={username} busy={busy} onSignOut={logout} />
        <main>
          {api && caps?.sessions?.supported && (
            <>
              {caps.sessions.resumable && <SessionWorkbench
                api={api}
                sessionId={sessionId}
                experimentId={experiment?.id}
                busy={busy}
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
            {api ? "Set up your experiment" : "Connect to your research workspace"}
          </h1>
          <p className="intro">
            Configure an encoding model and the compute resources for your next
            experiment.
          </p>
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
              <div className="workflow" aria-label="Experiment workflow">
                <span className={!experiment ? "current" : ""}>
                  1 <b>Configure</b>
                </span>
                <span className={experiment ? "current" : ""}>2 Recording</span>
                <span>3 Fit model</span>
                <span>4 Generate</span>
              </div>
              <form onSubmit={save}>
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
                                setExperiment(null);
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
                          setExperiment(null);
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
                        onClick={refresh}
                        disabled={busy}
                      >
                        {busy ? "Checking…" : "↻ Refresh capacity"}
                      </button>
                    </div>
                  </section>
                </div>
                <div className="action-bar">
                  <p>Settings are saved with a unique experiment identity.</p>
                  <button className="primary" disabled={busy || invalidGpu}>
                    {busy ? "Saving…" : "Save experiment →"}
                  </button>
                </div>
              </form>
              {experiment && (
                <section className="notice success" role="status">
                  <strong>Experiment configured</strong>
                  <p>
                    Your {experiment.configuration.stimulus_family} setup is
                    saved with {experiment.configuration.gpu_count} selected
                    GPUs.
                  </p>
                  <code>{experiment.id}</code>
                </section>
              )}
              {experiment && (
                <RecordingSelector
                  key={experiment.id}
                  api={api}
                  experiment={experiment}
                  onSelected={recordingSelected}
                />
              )}
              {experiment && caps.upload && (
                <RecordingUpload
                  key={`upload-${experiment.id}`}
                  api={api}
                  experiment={experiment}
                  maxBytes={caps.upload.max_bytes}
                  onSelected={recordingSelected}
                />
              )}
              <p className="footnote">
                GPU capacity is checked again when you save. Saving a setup does
                not reserve GPUs or start computation.
              </p>
              {experiment?.recording &&
                caps.fit?.supported_sources.includes(
                  experiment.recording.source,
                ) && (
                  <FitPanel
                    key={`fit-${experiment.id}-${experiment.recording_revision}-${resumeCount}`}
                    api={api}
                    experiment={experiment}
                    seed={caps.fit.default_seed}
                    generationOptions={caps.generation}
                    batchEnabled={Boolean(caps.batch?.supported)}
                    recoveryEnabled={Boolean(caps.sessions?.supported)}
                  />
                )}
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
