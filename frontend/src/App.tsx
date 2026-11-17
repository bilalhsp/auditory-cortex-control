import { useState } from "react";
import type { FormEvent } from "react";
import { ApiError, makeApi } from "./api";
import type { Capabilities, Experiment } from "./api";
import RecordingSelector from "./RecordingSelector";
import RecordingUpload from "./RecordingUpload";

export default function App() {
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

  function reportError(cause: unknown) {
    setError(
      cause instanceof Error
        ? cause.message
        : "Could not connect to the server.",
    );
    if (cause instanceof ApiError && cause.status === 401) {
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
      const connection = makeApi(username, password);
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
    if (!api || !caps) return;
    setBusy(true);
    setError("");
    setExperiment(null);
    try {
      setExperiment(
        await api.createExperiment({
          stimulus_family: family,
          dataset_name: caps.settings.datasets[0],
          model_name: caps.settings.models[0],
          layer_id: caps.settings.layers[0],
          bin_width_ms: caps.settings.bin_width_ms,
          receptive_field_ms: caps.settings.receptive_field_ms,
          gpu_count: gpuCount,
        }),
      );
    } catch (cause) {
      reportError(cause);
    } finally {
      setBusy(false);
    }
  }
  function logout() {
    setApi(null);
    setCaps(null);
    setExperiment(null);
    setPassword("");
    setError("");
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
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Auditory Cortex Control home">
          <span className="brand-mark">∿</span>
          <span>
            Auditory Cortex
            <br />
            <strong>Control</strong>
          </span>
        </a>
        <div className="nav-caption">WORKSPACE</div>
        <div className="nav-item active">◈ &nbsp; Experiment setup</div>
        <div className="sidebar-note">
          <span className="dot" /> Single-operator workspace
          <br />
          <small>Neural response · Optimal stimulus</small>
        </div>
      </aside>
      <div className="workspace">
        <header>
          <span>
            Research workspace <span className="slash">/</span>{" "}
            <strong>New experiment</strong>
          </span>
          {api && (
            <button className="text-button" onClick={logout} disabled={busy}>
              Sign out
            </button>
          )}
        </header>
        <main>
          <div className="eyebrow">AUDITORY CORTEX CONTROL</div>
          <h1>
            {api ? "Set up your experiment" : "Your research, connected."}
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
                        <span className={available ? "dot" : "dot muted"} />
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
