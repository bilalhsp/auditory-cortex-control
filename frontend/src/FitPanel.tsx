import { useJobPolling } from "./useJobPolling";
import { useEffect, useRef, useState } from "react";
import type {
  Capabilities,
  Experiment,
  Fit,
  RecoveredWork,
  makeApi,
} from "./api";

import GenerationPanel from "./GenerationPanel";
import BatchPanel from "./BatchPanel";

interface Props {
  recoveryEnabled?: boolean;
  batchEnabled?: boolean;
  generationOptions?: Capabilities["generation"];
  api: ReturnType<typeof makeApi>;
  experiment: Experiment;
  seed: number;
}

function isActiveJob(job: { state: string }): boolean {
  return job.state === "queued" || job.state === "running";
}

function preferActive<T extends { state: string }>(jobs: T[]): T | undefined {
  return jobs.find(isActiveJob) ?? jobs[0];
}

export default function FitPanel({
  api,
  experiment,
  seed,
  generationOptions,
  batchEnabled,
  recoveryEnabled,
}: Props) {
  const [recovered, setRecovered] = useState<RecoveredWork | null>(null);
  const [recovering, setRecovering] = useState(true);
  const [fit, setFit] = useState<Fit | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const mounted = useRef(false);
  const requestId = useRef<string | null>(null);
  useEffect(() => {
    if (!recoveryEnabled) {
      setRecovering(false);
      return;
    }
    let current = true;
    api
      .recoverExperiment(experiment.id)
      .then((work) => {
        if (!current) return;
        setRecovered(work);
        const matchingFits = work.fits.filter(
          (item) => item.recording.id === experiment.recording?.id &&
            item.recording_revision === experiment.recording_revision,
        );
        const activeFitIds = new Set(
          [...work.batches, ...work.generations]
            .filter(isActiveJob)
            .map((item) => item.fit_id),
        );
        setFit((previous) => previous ??
          matchingFits.find(isActiveJob) ??
          matchingFits.find((item) => activeFitIds.has(item.id)) ??
          matchingFits[0] ?? null);
      })
      .catch(() => {
        if (current)
          setError(
            "Could not recover saved work. Open session history to refresh.",
          );
      })
      .finally(() => {
        if (current) setRecovering(false);
      });
    return () => {
      current = false;
    };
  }, [api, experiment.id, experiment.recording_revision, recoveryEnabled]);

  const active =
    fit?.state === "queued" ||
    fit?.state === "running" ||
    fit?.state === "cancelling";

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useJobPolling(
    fit,
    api.fit,
    setFit,
    setError,
    "Could not refresh fit status. Accepted work continues on the server; reconnect to refresh.",
  );

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
          recovering ||
          submitting ||
          active ||
          experiment.configuration.gpu_count < 1
        }
        onClick={start}
      >
        {recovering
          ? "Recovering saved work…"
          : submitting
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
            {fit.state === "cancelling"
              ? "Stopping fit"
              : fit.state === "cancelled"
                ? "Fit stopped"
                : fit.state === "failed"
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
          {batchEnabled && generationOptions && (
            <BatchPanel
              key={`batch-${fit.id}`}
              api={api}
              fit={fit}
              options={generationOptions}
              recovered={preferActive(recovered?.batches.filter((item) => item.fit_id === fit.id) ?? [])}
            />
          )}
          {generationOptions && (
            <GenerationPanel
              key={fit.id}
              api={api}
              fit={fit}
              options={generationOptions}
              recovered={preferActive(recovered?.generations.filter((item) => item.fit_id === fit.id) ?? [])}
            />
          )}
        </div>
      )}
    </section>
  );
}
