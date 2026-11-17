import { useEffect, useRef, useState } from "react";
import type { Experiment, Fit, makeApi } from "./api";

import GenerationPanel from "./GenerationPanel";

interface Props {
  generationEnabled?: boolean;
  api: ReturnType<typeof makeApi>;
  experiment: Experiment;
  seed: number;
}

export default function FitPanel({
  api,
  experiment,
  seed,
  generationEnabled,
}: Props) {
  const [fit, setFit] = useState<Fit | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const mounted = useRef(false);
  const requestId = useRef<string | null>(null);
  const active = fit?.state === "queued" || fit?.state === "running";

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (!fit || !active) return;
    let current = true;
    const timer = window.setInterval(() => {
      api
        .fit(fit.id)
        .then((result) => {
          if (current) {
            setFit(result);
            setError("");
          }
        })
        .catch(() => {
          if (current)
            setError(
              "Could not refresh fit status. Accepted work continues on the server; reconnect to refresh.",
            );
        });
    }, 1500);
    return () => {
      current = false;
      window.clearInterval(timer);
    };
  }, [api, fit?.id, fit?.state, active]);

  async function start() {
    if (!experiment.recording) return;
    requestId.current ??= crypto.randomUUID();
    setSubmitting(true);
    setError("");
    try {
      const result = await api.startFit(experiment.id, {
        request_id: requestId.current,
        recording_id: experiment.recording.id,
        recording_revision: experiment.recording_revision,
        seed,
      });
      if (mounted.current) {
        setFit(result);
        requestId.current = null;
      }
    } catch (cause) {
      if (mounted.current)
        setError(
          cause instanceof Error
            ? cause.message
            : "Could not submit fit. Retry submission to check its outcome.",
        );
    } finally {
      if (mounted.current) setSubmitting(false);
    }
  }

  const labels: Record<string, string> = {
    queued: "Waiting for fit worker",
    loading: "Loading models and features",
    fitting: "Fitting encoding model",
    writing: "Saving model and results",
    complete: "Fit complete",
    interrupted: "Fit interrupted",
  };
  return (
    <section className="card recording-card" aria-labelledby="fit-heading">
      <div className="section-heading">
        <span className="step">05</span>
        <div>
          <h2 id="fit-heading">Fit the encoding model</h2>
          <p>Three-fold cross-validation · Seed {seed} · One fitting GPU</p>
        </div>
      </div>
      <p className="help">
        {experiment.recording?.display_name}. Each fit keeps its own recording
        snapshot and model. Accepted work continues when this page closes.
      </p>
      {error && (
        <div className="notice error" role="alert">
          {error}
        </div>
      )}
      <button
        type="button"
        className="primary"
        disabled={
          submitting || active || experiment.configuration.gpu_count < 1
        }
        onClick={start}
      >
        {submitting
          ? "Submitting…"
          : active
            ? "Fit in progress…"
            : fit
              ? "Start a new fit →"
              : "Fit encoding model →"}
      </button>
      {experiment.configuration.gpu_count < 1 && (
        <p className="notice warning">
          Select at least one GPU in your setup to enable fitting.
        </p>
      )}
      {fit && (
        <div className="fit-status" role="status">
          <strong>
            {fit.state === "failed"
              ? "Fit failed"
              : (labels[fit.stage] ?? fit.state)}
          </strong>
          <p>
            {fit.error ??
              (fit.state === "queued"
                ? "Your request is saved. Waiting for the backend worker."
                : fit.state === "running"
                  ? "Computation is running independently of this browser."
                  : "")}
          </p>
          <small>Fit {fit.id}</small>
        </div>
      )}
      {fit?.state === "succeeded" && fit.artifacts_ready && (
        <div className="fit-results">
          <h3>Channel correlations</h3>
          {fit.best_channel ? (
            <div className="correlation-summary">
              <span>
                Best channel <strong>{fit.best_channel.channel_id}</strong>
              </span>
              <span>
                Correlation{" "}
                <strong>{fit.best_channel.correlation?.toFixed(3)}</strong>
              </span>
            </div>
          ) : (
            <p className="notice warning">
              No finite channel correlations are available for this fit.
            </p>
          )}
          <div className="correlations-scroll">
            <table className="fit-table">
              <thead>
                <tr>
                  <th scope="col">Channel</th>
                  <th scope="col">Correlation</th>
                </tr>
              </thead>
              <tbody>
                {fit.correlations.map((row) => (
                  <tr key={row.channel_id}>
                    <td>{row.channel_id}</td>
                    <td>
                      {row.correlation === null
                        ? "Unavailable"
                        : row.correlation.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {generationEnabled && (
            <GenerationPanel key={fit.id} api={api} fit={fit} />
          )}
        </div>
      )}
    </section>
  );
}
