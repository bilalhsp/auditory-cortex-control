import { useEffect, useRef, useState } from "react";
import type { Experiment, makeApi } from "./api";

interface Props {
  api: ReturnType<typeof makeApi>;
  experiment: Experiment;
  maxBytes: number;
  onSelected: (experiment: Experiment) => void;
}

export default function RecordingUpload({
  api,
  experiment,
  maxBytes,
  onSelected,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const mounted = useRef(false);
  const unit = maxBytes < 1024 * 1024 ? "KB" : "MB";
  const maximum = `${(maxBytes / (unit === "KB" ? 1024 : 1024 * 1024)).toLocaleString(undefined, { maximumFractionDigits: 1 })} ${unit}`;

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    setFile(null);
    setError("");
    if (input.current) input.current.value = "";
  }, [experiment.recording?.id]);

  function choose(selected: File | undefined) {
    setFile(null);
    setError("");
    if (!selected) return;
    if (!selected.name.toLowerCase().endsWith(".mat")) {
      setError(
        "Choose a MAT recording file. Model checkpoints are not supported.",
      );
    } else if (selected.size > maxBytes) {
      setError(`Recording exceeds the ${maximum} upload limit.`);
    } else {
      setFile(selected);
    }
  }

  async function upload() {
    if (!file) return;
    setPending(true);
    setError("");
    try {
      const result = await api.uploadRecording(experiment.id, file);
      if (mounted.current) onSelected(result.experiment);
    } catch (cause) {
      if (mounted.current)
        setError(
          cause instanceof Error
            ? cause.message
            : "Upload failed. Retry your recording.",
        );
    } finally {
      if (mounted.current) setPending(false);
    }
  }

  return (
    <section className="card recording-card" aria-labelledby="upload-heading">
      <div className="section-heading">
        <span className="step">04</span>
        <div>
          <h2 id="upload-heading">Or upload a recording</h2>
          <p>
            Use a new {experiment.configuration.stimulus_family} recording for
            this experiment.
          </p>
        </div>
      </div>
      <label htmlFor="recording-upload">MAT recording file</label>
      <input
        id="recording-upload"
        ref={input}
        type="file"
        accept=".mat"
        disabled={pending}
        onChange={(event) => choose(event.target.files?.[0])}
        aria-describedby="upload-help"
      />
      <p id="upload-help" className="help">
        Up to {maximum}. S1T trial data with stimulus onsets and WM/BAK neural
        spikes.
      </p>
      {error && (
        <div className="notice error" role="alert">
          {error}
        </div>
      )}
      {pending && (
        <p role="status" className="help">
          Uploading and validating recording…
        </p>
      )}
      <div className="action-bar">
        <p className="help">
          Previous uploads and completed results are retained.
        </p>
        <button
          type="button"
          className="primary"
          disabled={!file || pending}
          onClick={upload}
        >
          {pending ? "Validating…" : "Upload and use recording →"}
        </button>
      </div>
      {experiment.recording?.source === "upload" && (
        <div className="notice selected-recording" role="status">
          <strong>Validated recording attached</strong>
          <p>{experiment.recording.display_name}</p>
          <small>
            Compatible with {experiment.configuration.stimulus_family}.
            Replacing it retains the previous file.
          </small>
        </div>
      )}
    </section>
  );
}
