import { useEffect, useState } from "react";
import { ApiError } from "./api";
import type { Batch, Capabilities, Fit, makeApi } from "./api";
import { isActiveJobState, useJobPolling } from "./useJobPolling";
import { saveDownload } from "./download";

const runStatus: Record<Batch["state"], string> = {
  queued: "Waiting to generate",
  running: "Generating samples",
  cancelling: "Stopping generation",
  succeeded: "Samples ready",
  partial: "Some samples are ready",
  failed: "Generation failed",
  interrupted: "Generation interrupted",
  cancelled: "Generation stopped",
};

type Props = {
  api: ReturnType<typeof makeApi>;
  fit: Fit;
  options: NonNullable<Capabilities["generation"]>;
  recovered?: Batch;
  exportSessionId?: string;
  onBusy: (busy: boolean) => void;
  onPhase: (phase: "Generate" | "Results") => void;
};

export default function SampleGeneration({ api, fit, options, recovered, onBusy, exportSessionId, onPhase }: Props) {
  const [targets, setTargets] = useState<{ channel_id: string; objective: string }[]>([]);
  const [channel, setChannel] = useState(fit.best_channel?.channel_id ?? "");
  const [objective, setObjective] = useState(options.objectives[0]);
  const [duration, setDuration] = useState(options.durations[0]);
  const [clips, setClips] = useState(options.clip_counts[0]);
  const [capacity, setCapacity] = useState<Capabilities["gpu"]>();
  const [gpuChoice, setGpuChoice] = useState<number>();
  const [batch, setBatch] = useState<Batch | null>(recovered ?? null);
  const [composing, setComposing] = useState(!recovered);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [pollError, setPollError] = useState("");
  const [pending, setPending] = useState<Parameters<typeof api.generateSamples>[1]>();
  const active = isActiveJobState(batch?.state);
  const limit = Math.min(fit.configuration.gpu_count, capacity?.visible_count ?? 0);
  const reduced = Boolean(capacity && fit.configuration.gpu_count > capacity.visible_count);
  const gpuCount = reduced && (gpuChoice === undefined || gpuChoice > limit)
    ? 0 : Math.min(gpuChoice ?? fit.configuration.gpu_count, limit);
  const selected = targets.some(target => target.channel_id === channel && target.objective === objective);

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    async function refresh() {
      try {
        const { gpu } = await api.capabilities();
        if (!stopped) setCapacity(gpu);
      } catch {
        if (!stopped) setCapacity(undefined);
      }
      if (!stopped) timer = window.setTimeout(refresh, 4000);
    }
    void refresh();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [api]);

  useJobPolling(batch, api.batch, setBatch, setPollError, "Can’t refresh sample progress. Reconnect to see updates.");

  async function generate() {
    if (busy || active || (!pending && (targets.length < 1 || targets.length > gpuCount))) return;
    const command = pending ?? {
      request_id: crypto.randomUUID(), gpu_count: gpuCount, duration_seconds: duration,
      clip_count: clips, seed: 42, targets,
    };
    setPending(command);
    setBusy(true);
    onBusy(true);
    setError("");
    try {
      setBatch(await api.generateSamples(fit.id, command));
      setPending(undefined);
      setComposing(false);
      onPhase("Results");
    } catch (cause) {
      if (cause instanceof ApiError && cause.status < 500) setPending(undefined);
      setError(cause instanceof Error ? cause.message : "Could not confirm submission. Retry to check its outcome.");
    } finally {
      setBusy(false);
      onBusy(false);
    }
  }

  async function download() {
    if (!batch) return;
    try {
      if (exportSessionId) {
        const result = await api.exportSamples(exportSessionId, batch.job_id);
        saveDownload(result.blob, result.filename);
      } else {
        saveDownload(await api.downloadBatch(batch.id), `batch-${batch.id}.zip`);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Download failed.");
    }
  }

  return (
    <section className="sample-generation" aria-label="Sample generation controls">
      {(error || pollError) && <p role="alert" className="notice error">{error || pollError}</p>}
      {batch && <div className="fit-status">
        <p role="status"><strong className="sample-status">
          {runStatus[batch.state]}
          {active && !pollError && <span className="sample-pulse" aria-hidden="true"><span /><span /><span /></span>}
        </strong></p>
        {batch.error && <p>{batch.error}</p>}
        <ul>{batch.targets.map(target => <li key={target.id}>
          {target.channel_id} / {target.objective} · {target.state} · {target.clips.length} {target.clips.length === 1 ? "clip" : "clips"}
        </li>)}</ul>
        <div className="sample-actions">
        {batch.artifacts_ready && <button type="button" className="text-button" onClick={download}>Download results</button>}
        {!active && !composing && <button type="button" className="primary" onClick={() => {
          setComposing(true);
          onPhase("Generate");
          setTargets([]);
        }}>Generate more</button>}
        </div>
      </div>}
      {composing && !pending && <fieldset disabled={active || busy}>
        <legend>Choose targets</legend>
        <p className="help">
          {capacity ? `${capacity.visible_count} GPUs visible, ${capacity.available_count} free.` : "Checking GPU capacity…"}
          {" "}Your experiment selected {fit.configuration.gpu_count}.
        </p>
        {reduced && (
          <p role="status">GPU visibility decreased. Choose a count within the current limit to generate more.</p>
        )}
        <label>GPUs for this run
          <select disabled={!limit} value={gpuCount} onChange={event => setGpuChoice(Number(event.target.value))}>
            {!gpuCount && <option value={0}>{limit ? "Choose a GPU count" : "No GPUs available"}</option>}
            {Array.from({ length: limit }, (_, index) => index + 1).map(count => (
              <option key={count} disabled={count < targets.length}>{count}</option>
            ))}
          </select>
        </label>
        <div className="fields">
          <label>Channel
            <select value={channel} onChange={event => setChannel(event.target.value)}>
              {fit.correlations.filter(row => row.correlation !== null).map(row => (
                <option key={row.channel_id}>{row.channel_id}</option>
              ))}
            </select>
          </label>
          <label>Objective
            <select value={objective} onChange={event => setObjective(event.target.value)}>
              {options.objectives.map(item => <option key={item}>{item}</option>)}
            </select>
          </label>
        </div>
        <button type="button" className="text-button" disabled={!channel || selected || targets.length >= gpuCount}
          onClick={() => setTargets([...targets, { channel_id: channel, objective }])}>Add target</button>
        <ul>{targets.map((target, index) => <li key={JSON.stringify(target)}>
          {target.channel_id} / {target.objective}{" "}
          <button type="button" className="text-button" onClick={() => setTargets(targets.filter((_, i) => i !== index))}
            aria-label={`Remove ${target.channel_id} / ${target.objective}`}>Remove</button>
        </li>)}</ul>
        <p role="status">{targets.length} of {gpuCount} targets selected</p>
        <div className="fields">
          <label>Duration
            <select value={duration} onChange={event => setDuration(Number(event.target.value))}>
              {options.durations.map(value => <option key={value} value={value}>{value} seconds</option>)}
            </select>
          </label>
          <label>Clips per target
            <select value={clips} onChange={event => setClips(Number(event.target.value))}>
              {options.clip_counts.map(value => <option key={value}>{value}</option>)}
            </select>
          </label>
        </div>
        <button type="button" className="primary"
          disabled={!targets.length || targets.length > gpuCount || gpuCount > (capacity?.available_count ?? 0)}
          onClick={generate}>Generate samples</button>
      </fieldset>}
      {pending && <button type="button" className="primary" disabled={busy}
        onClick={generate}>{busy ? "Submitting…" : "Retry sample submission"}</button>}
    </section>
  );
}
