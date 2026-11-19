import { useEffect, useRef, useState } from "react";
import type { Batch, Capabilities, Fit, makeApi } from "./api";
import { isActiveJobState, useJobPolling } from "./useJobPolling";
import { saveDownload } from "./download";

type Props = {
  api: ReturnType<typeof makeApi>;
  fit: Fit;
  options: NonNullable<Capabilities["generation"]>;
  recovered?: Batch;
  exportSessionId?: string;
  onBusy: (busy: boolean) => void;
};

export default function SampleGeneration({ api, fit, options, recovered, onBusy, exportSessionId }: Props) {
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
  const requestId = useRef<string | null>(null);
  const active = isActiveJobState(batch?.state);
  const limit = Math.min(fit.configuration.gpu_count, capacity?.visible_count ?? 0);
  const gpuCount = Math.min(gpuChoice ?? fit.configuration.gpu_count, limit);
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

  function changeTargets(next: typeof targets) {
    setTargets(next);
    requestId.current = null;
  }

  async function generate() {
    if (busy || active || targets.length < 1 || targets.length > gpuCount) return;
    requestId.current ??= crypto.randomUUID();
    setBusy(true);
    onBusy(true);
    setError("");
    try {
      setBatch(await api.generateSamples(fit.id, {
        request_id: requestId.current, gpu_count: gpuCount, duration_seconds: duration,
        clip_count: clips, seed: 42, targets,
      }));
      requestId.current = null;
      setComposing(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not generate samples.");
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
    <section className="generation-panel" aria-labelledby="samples-heading">
      <h3 id="samples-heading">Generate samples</h3>
      {(error || pollError) && <p role="alert" className="notice error">{error || pollError}</p>}
      {batch && <div className="fit-status">
        <p role="status"><strong>Sample run {batch.state}</strong></p>
        {batch.error && <p>{batch.error}</p>}
        <ul>{batch.targets.map(target => <li key={target.id}>
          {target.channel_id} / {target.objective} · {target.state} · {target.clips.length} clips
        </li>)}</ul>
        {batch.artifacts_ready && <button type="button" className="text-button" onClick={download}>Download results</button>}
        {!active && !composing && <button type="button" className="primary" onClick={() => {
          setComposing(true);
          setTargets([]);
        }}>Generate more</button>}
      </div>}
      {composing && <fieldset disabled={active || busy}>
        <legend>Choose targets</legend>
        <p className="help">
          {capacity ? `${capacity.visible_count} GPUs visible, ${capacity.available_count} free.` : "Checking GPU capacity…"}
          {" "}Your experiment selected {fit.configuration.gpu_count}.
        </p>
        {capacity && fit.configuration.gpu_count > capacity.visible_count && (
          <p role="status">GPU visibility decreased. Choose a count within the current limit to generate more.</p>
        )}
        <label>GPUs for this run
          <select disabled={!limit} value={gpuCount} onChange={event => {
            setGpuChoice(Number(event.target.value));
            requestId.current = null;
          }}>
            {!limit && <option value={0}>No GPUs available</option>}
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
          onClick={() => changeTargets([...targets, { channel_id: channel, objective }])}>Add target</button>
        <ul>{targets.map((target, index) => <li key={JSON.stringify(target)}>
          {target.channel_id} / {target.objective}{" "}
          <button type="button" className="text-button" onClick={() => changeTargets(targets.filter((_, i) => i !== index))}
            aria-label={`Remove ${target.channel_id} / ${target.objective}`}>Remove</button>
        </li>)}</ul>
        <p role="status">{targets.length} of {gpuCount} targets selected</p>
        <div className="fields">
          <label>Duration
            <select value={duration} onChange={event => { setDuration(Number(event.target.value)); requestId.current = null; }}>
              {options.durations.map(value => <option key={value} value={value}>{value} seconds</option>)}
            </select>
          </label>
          <label>Clips per target
            <select value={clips} onChange={event => { setClips(Number(event.target.value)); requestId.current = null; }}>
              {options.clip_counts.map(value => <option key={value}>{value}</option>)}
            </select>
          </label>
        </div>
        <button type="button" className="primary"
          disabled={!targets.length || targets.length > gpuCount || gpuCount > (capacity?.available_count ?? 0)}
          onClick={generate}>Generate samples</button>
      </fieldset>}
    </section>
  );
}
