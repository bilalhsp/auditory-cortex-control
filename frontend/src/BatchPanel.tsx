import { saveDownload } from "./download";
import { useJobPolling } from "./useJobPolling";
import { useEffect, useRef, useState } from "react";
import type { Batch, Capabilities, Fit, makeApi } from "./api";

export default function BatchPanel({
  api,
  fit,
  options,
  recovered,
}: {
  api: ReturnType<typeof makeApi>;
  fit: Fit;
  recovered?: Batch;
  options: NonNullable<Capabilities["generation"]>;
}) {
  const [channels, setChannels] = useState<string[]>(
    fit.best_channel ? [fit.best_channel.channel_id] : [],
  );
  const [objectives, setObjectives] = useState<string[]>(["stretch"]);
  const [duration, setDuration] = useState(2),
    [clips, setClips] = useState(4);
  const [gpus, setGpus] = useState<number | null>(fit.configuration.gpu_count);
  const [available, setAvailable] = useState<number | null>(null);
  const [batch, setBatch] = useState<Batch | null>(recovered ?? null);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const mounted = useRef(false),
    requestId = useRef<string | null>(null);
  const active =
    batch?.state === "queued" ||
    batch?.state === "running" ||
    batch?.state === "cancelling";
  async function refresh() {
    try {
      const capacity = await api.capabilities();
      if (!mounted.current) return;
      setAvailable(capacity.gpu.available_count);
      setGpus((current) =>
        current !== null &&
        current >= 1 &&
        current <= capacity.gpu.available_count
          ? current
          : null,
      );
    } catch {
      if (mounted.current)
        setError("Could not refresh GPU capacity. Retry before submitting.");
    }
  }
  useEffect(() => {
    mounted.current = true;
    void refresh();
    return () => {
      mounted.current = false;
    };
  }, [api]);
  useJobPolling(
    batch,
    api.batch,
    setBatch,
    setError,
    "Could not refresh batch status. Accepted work continues on the server.",
  );
  function toggle(
    value: string,
    selected: string[],
    update: (values: string[]) => void,
  ) {
    update(
      selected.includes(value)
        ? selected.filter((item) => item !== value)
        : [...selected, value],
    );
    requestId.current = null;
  }
  async function start() {
    if (gpus === null) return;
    requestId.current ??= crypto.randomUUID();
    setBusy(true);
    setError("");
    try {
      const result = await api.startBatch(fit.id, {
        request_id: requestId.current,
        gpu_count: gpus,
        duration_seconds: duration,
        clip_count: clips,
        seed: 42,
        targets: channels.flatMap((channel_id) =>
          objectives.map((objective) => ({ channel_id, objective })),
        ),
      });
      if (mounted.current) {
        setBatch(result);
        requestId.current = null;
      }
    } catch (cause) {
      if (mounted.current) {
        setError(
          cause instanceof Error ? cause.message : "Could not submit batch.",
        );
        await refresh();
      }
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  async function download() {
    if (!batch) return;
    setBusy(true);
    setError("");
    try {
      const blob = await api.downloadBatch(batch.id);
      if (!mounted.current) return;
      saveDownload(blob, `batch-${batch.id}.zip`);
    } catch (cause) {
      if (mounted.current)
        setError(cause instanceof Error ? cause.message : "Download failed.");
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  return (
    <section className="generation-panel" aria-labelledby="batch-heading">
      <h3 id="batch-heading">Generate a target batch</h3>
      <p className="help">
        Each channel/objective pair uses one GPU. Extra targets wait for a free
        selected GPU. Requested GPUs remain reserved until the batch finishes,
        including idle devices when there are fewer targets.
      </p>
      <fieldset disabled={active || busy}>
        <legend>Batch targets</legend>
        <div className="batch-channels">
          {fit.correlations
            .filter((row) => row.correlation !== null)
            .map((row) => (
              <label key={row.channel_id}>
                <input
                  type="checkbox"
                  aria-label={`Batch channel ${row.channel_id}`}
                  checked={channels.includes(row.channel_id)}
                  onChange={() => toggle(row.channel_id, channels, setChannels)}
                />
                {row.channel_id}
              </label>
            ))}
        </div>
        <div className="batch-channels">
          {options.objectives.map((value) => (
            <label key={value}>
              <input
                type="checkbox"
                aria-label={`Batch ${value}`}
                checked={objectives.includes(value)}
                onChange={() => toggle(value, objectives, setObjectives)}
              />
              {value}
            </label>
          ))}
        </div>
        <label>
          Batch GPUs
          <select
            aria-label="Batch GPUs"
            value={gpus ?? ""}
            onChange={(event) => {
              setGpus(event.target.value ? Number(event.target.value) : null);
              requestId.current = null;
            }}
          >
            <option value="">Choose available GPUs</option>
            {Array.from(
              { length: available ?? 0 },
              (_, index) => index + 1,
            ).map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <p className="help">
          Available at last check: {available ?? "checking…"}. The initial
          choice preserves your saved GPU selection; a smaller valid choice
          stays selected when refreshed.
        </p>
        <button type="button" onClick={refresh}>
          Refresh batch GPU capacity
        </button>
        <label>
          Batch duration
          <select
            aria-label="Batch duration"
            value={duration}
            onChange={(event) => {
              setDuration(Number(event.target.value));
              requestId.current = null;
            }}
          >
            {options.durations.map((value) => (
              <option key={value} value={value}>
                {value} seconds
              </option>
            ))}
          </select>
        </label>
        <label>
          Batch clips per target
          <select
            aria-label="Batch clips per target"
            value={clips}
            onChange={(event) => {
              setClips(Number(event.target.value));
              requestId.current = null;
            }}
          >
            {options.clip_counts.map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <p>{channels.length * objectives.length} targets · Seed 42</p>
        <button
          type="button"
          className="primary"
          disabled={
            gpus === null ||
            available === null ||
            gpus > available ||
            channels.length === 0 ||
            objectives.length === 0
          }
          onClick={start}
        >
          {active ? "Batch in progress…" : "Generate batch →"}
        </button>
      </fieldset>
      {error && (
        <p role="alert" className="notice error">
          {error}
        </p>
      )}
      {batch && (
        <div className="fit-status" role="status">
          <strong>Batch {batch.state}</strong>
          <p>
            {batch.gpu_count} GPUs requested · {batch.used_gpu_count} parallel
            target slots · {batch.unused_gpu_count} idle GPUs
          </p>
          <p>{batch.error ?? `Batch ${batch.id}`}</p>
          <div className="correlations-scroll">
            <table className="fit-table">
              <thead>
                <tr>
                  <th>Channel</th>
                  <th>Objective</th>
                  <th>Outcome</th>
                  <th>Clips</th>
                </tr>
              </thead>
              <tbody>
                {batch.targets.map((target) => (
                  <tr key={target.id}>
                    <td>{target.channel_id}</td>
                    <td>{target.objective}</td>
                    <td>{target.state}</td>
                    <td>{target.clips.length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {batch?.artifacts_ready && (
        <>
          <p className="help">
            The ZIP includes a manifest and only validated completed targets.
            Earlier batches remain saved.
          </p>
          <button type="button" disabled={busy} onClick={download}>
            {batch.state === "partial"
              ? "Download completed targets ZIP"
              : "Download batch ZIP"}
          </button>
        </>
      )}
    </section>
  );
}
