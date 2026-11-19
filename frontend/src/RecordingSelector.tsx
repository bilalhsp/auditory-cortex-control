import { useEffect, useRef, useState } from "react";
import { ApiError } from "./api";
import type { Experiment, RecordingCatalog, makeApi } from "./api";

interface Props {
  api: ReturnType<typeof makeApi>;
  experiment: Experiment;
  onSelected: (experiment: Experiment) => void;
  onBusy: (busy: boolean) => void;
}

export default function RecordingSelector({
  api,
  experiment,
  onSelected,
  onBusy,
}: Props) {
  const [catalog, setCatalog] = useState<RecordingCatalog | null>(null);
  const [candidate, setCandidate] = useState(
    experiment.recording?.source === "existing" ? experiment.recording.id : "",
  );
  const [refresh, setRefresh] = useState(0);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const mounted = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (experiment.recording?.source === "upload") setCandidate("");
  }, [experiment.recording?.id, experiment.recording?.source]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    api
      .recordings(experiment.id)
      .then((result) => {
        if (active) setCatalog(result);
      })
      .catch((cause) => {
        if (active) setError(message(cause));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [api, experiment.id, refresh]);

  function message(cause: unknown) {
    if (cause instanceof ApiError && cause.status === 401)
      return "Sign-in is no longer valid. Sign out and reconnect.";
    return cause instanceof Error
      ? cause.message
      : "Could not load recordings. Refresh to retry.";
  }

  async function select() {
    setSaving(true);
    onBusy(true);
    setError("");
    try {
      const updated = await api.selectRecording(experiment.id, candidate);
      if (mounted.current) onSelected(updated);
    } catch (cause) {
      if (mounted.current) setError(message(cause));
    } finally {
      if (mounted.current) setSaving(false);
      onBusy(false);
    }
  }

  const valid =
    catalog?.recordings.some((recording) => recording.id === candidate) ??
    false;
  return (
    <div>
      {error && (
        <div className="notice error" role="alert">
          {error}
        </div>
      )}
      {loading && (
        <p role="status" className="help">
          Loading compatible recordings…
        </p>
      )}
      {!loading && catalog && !catalog.recordings.length && !error && (
        <p className="notice warning" role="status">
          No compatible recordings are available for this experiment.
        </p>
      )}
      {!!catalog?.unavailable_count && (
        <p className="notice warning" role="status">
          {catalog.unavailable_count} configured recording(s) are missing or
          unreadable. Available recordings are shown below.
        </p>
      )}
      <label htmlFor="existing-recording">Existing recording</label>
      <select
        id="existing-recording"
        value={candidate}
        disabled={loading || saving || !catalog?.recordings.length}
        onChange={(event) => {
          setCandidate(event.target.value);
          setError("");
        }}
      >
        <option value="">Choose a recording</option>
        {candidate && !valid && (
          <option value={candidate} disabled>
            Previously selected recording is unavailable
          </option>
        )}
        {catalog?.recordings.map((recording) => (
          <option key={recording.id} value={recording.id}>
            {recording.display_name}
          </option>
        ))}
      </select>
      <div className="action-bar">
        <button
          type="button"
          className="text-button"
          disabled={loading || saving}
          onClick={() => setRefresh((value) => value + 1)}
        >
          ↻ Refresh recordings
        </button>
        <button
          type="button"
          className="primary"
          disabled={loading || saving || !valid || !!error}
          onClick={select}
        >
          {saving ? "Attaching…" : "Use this recording →"}
        </button>
      </div>
      {experiment.recording?.source === "existing" && (
        <div className="notice selected-recording" role="status">
          <strong>Recording attached</strong>
          <p>{experiment.recording.display_name}</p>
          <small>
            Changing this recording starts a new selection for later analysis.
            Existing results are retained.
          </small>
        </div>
      )}
    </div>
  );
}
