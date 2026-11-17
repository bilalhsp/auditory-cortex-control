import { useEffect, useRef, useState } from "react";
import type { Fit, Generation, makeApi } from "./api";

export default function GenerationPanel({
  api,
  fit,
}: {
  api: ReturnType<typeof makeApi>;
  fit: Fit;
}) {
  const [channel, setChannel] = useState(fit.best_channel?.channel_id ?? "");
  const [objective, setObjective] = useState<"stretch" | "one-hot">("stretch");
  const [duration, setDuration] = useState(2);
  const [count, setCount] = useState(4);
  const [job, setJob] = useState<Generation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const requestId = useRef<string | null>(null);
  const mounted = useRef(false);
  const active = job?.state === "queued" || job?.state === "running";
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    if (!job || !active) return;
    let current = true;
    const timer = window.setInterval(() => {
      api
        .generation(job.id)
        .then((result) => {
          if (current) {
            setJob(result);
            setError("");
          }
        })
        .catch(() => {
          if (current)
            setError(
              "Could not refresh generation status. Accepted work continues on the server.",
            );
        });
    }, 1500);
    return () => {
      current = false;
      window.clearInterval(timer);
    };
  }, [api, job?.id, job?.state, active]);
  async function start() {
    requestId.current ??= crypto.randomUUID();
    setBusy(true);
    setError("");
    try {
      const result = await api.startGeneration(fit.id, {
        request_id: requestId.current,
        channel_id: channel,
        objective,
        duration_seconds: duration,
        clip_count: count,
        seed: 42,
      });
      if (mounted.current) {
        setJob(result);
        requestId.current = null;
      }
    } catch (cause) {
      if (mounted.current)
        setError(
          cause instanceof Error
            ? cause.message
            : "Could not submit generation. Retry to check its outcome.",
        );
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  async function download() {
    if (!job) return;
    setBusy(true);
    setError("");
    try {
      const blob = await api.downloadGeneration(job.id);
      if (!mounted.current) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `generation-${job.id}.zip`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 2000);
    } catch (cause) {
      if (mounted.current)
        setError(cause instanceof Error ? cause.message : "Download failed.");
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  function changed() {
    requestId.current = null;
  }
  return (
    <section aria-labelledby="generation-heading" className="generation-panel">
      <h3 id="generation-heading">Generate stimuli</h3>
      <p className="help">
        One target uses one GPU. Your request keeps{" "}
        {fit.configuration.gpu_count} GPU(s);{" "}
        {Math.max(0, fit.configuration.gpu_count - 1)} remain unused with one
        target. Seed 42.
      </p>
      <fieldset disabled={busy || active}>
        <label>
          Target channel
          <select
            aria-label="Target channel"
            value={channel}
            onChange={(event) => {
              setChannel(event.target.value);
              changed();
            }}
          >
            {fit.correlations
              .filter((row) => row.correlation !== null)
              .map((row) => (
                <option key={row.channel_id}>{row.channel_id}</option>
              ))}
          </select>
        </label>
        <label>
          Objective
          <select
            aria-label="Objective"
            value={objective}
            onChange={(event) => {
              setObjective(event.target.value as "stretch" | "one-hot");
              changed();
            }}
          >
            <option value="stretch">Stretch</option>
            <option value="one-hot">One-hot</option>
          </select>
        </label>
        <label>
          Requested duration
          <select
            aria-label="Requested duration"
            value={duration}
            onChange={(event) => {
              setDuration(Number(event.target.value));
              changed();
            }}
          >
            {[1, 2, 3, 4].map((value) => (
              <option key={value} value={value}>
                {value} seconds
              </option>
            ))}
          </select>
        </label>
        <label>
          Clips
          <select
            aria-label="Clips"
            value={count}
            onChange={(event) => {
              setCount(Number(event.target.value));
              changed();
            }}
          >
            {[1, 2, 4, 8, 10].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="primary"
          disabled={!channel}
          onClick={start}
        >
          {active
            ? "Generation in progress…"
            : busy
              ? "Submitting…"
              : "Generate samples →"}
        </button>
      </fieldset>
      <p className="help">
        The sampler rounds durations to its native audio length. The download
        reports the actual clip length. Earlier generations remain saved on the
        server.
      </p>
      {error && (
        <p role="alert" className="notice error">
          {error}
        </p>
      )}
      {job && (
        <div role="status" className="fit-status">
          <strong>
            {job.state === "succeeded"
              ? "Generation complete"
              : job.state === "running"
                ? "Generating audio"
                : job.state === "queued"
                  ? "Waiting for generation worker"
                  : `Generation ${job.state}`}
          </strong>
          <p>{job.error ?? `Generation ${job.id}`}</p>
        </div>
      )}
      {job?.state === "succeeded" && job.artifacts_ready && (
        <>
          <p>
            {job.clips.length} clips ·{" "}
            {((job.clips[0]?.frames ?? 0) / 16000).toFixed(3)} seconds each · 16
            kHz PCM16
          </p>
          <button type="button" disabled={busy} onClick={download}>
            Download samples ZIP
          </button>
        </>
      )}
    </section>
  );
}
