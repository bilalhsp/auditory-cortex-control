import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ApplicationSession,
  Experiment,
  Job,
  SessionDetail,
  makeApi,
} from "./api";
import { useJobPolling } from "./useJobPolling";
import { saveDownload } from "./download";
import { sessionDate } from "./sessionDate";

export default function SessionHistory({
  api,
  currentExperiment,
  onClosed,
  onResume,
  onNew,
  readOnly = false,
  timestampedExports = false,
}: {
  api: ReturnType<typeof makeApi>;
  currentExperiment: Experiment | null;
  onClosed?: (sessionId: string) => void;
  readOnly?: boolean;
  timestampedExports?: boolean;
  onResume?: (experiment: Experiment) => void;
  onNew?: () => void;
}) {
  const [sessions, setSessions] = useState<ApplicationSession[]>([]),
    [selected, setSelected] = useState("");
  const [detail, setDetail] = useState<SessionDetail | null>(null),
    [view, setView] = useState<Job | null>(null);
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const selection = useRef("");
  function chooseSession(id: string) {
    selection.current = id;
    setSelected(id);
    setDetail(null);
    setView(null);
  }
  function showJob(job: Job, context: string) {
    if (mounted.current && selection.current === context) setView(job);
  }
  const mounted = useRef(false),
    retries = useRef(new Map<string, string>());
  useEffect(() => {
    mounted.current = true;
    api
      .sessions()
      .then((rows) => {
        if (mounted.current) {
          setSessions(rows);
          if (!selection.current)
            chooseSession(currentExperiment?.session_id ?? rows[0]?.id ?? "");
        }
      })
      .catch(() => {
        if (mounted.current) setError("Could not load session history.");
      });
    return () => {
      mounted.current = false;
    };
  }, [api]);
  useEffect(() => {
    if (currentExperiment?.session_id)
      chooseSession(currentExperiment.session_id);
  }, [currentExperiment?.session_id]);
  useEffect(() => {
    let current = true;
    async function load() {
      try {
        const rows = await api.sessions();
        const result = selected ? await api.session(selected) : null;
        if (current) {
          setSessions(rows);
          setDetail(result);
          if (result && result.session.status !== "open")
            onClosed?.(result.session.id);
        }
      } catch {
        if (current) setError("Could not refresh session history.");
      }
    }
    void load();
    const timer = window.setInterval(load, 2000);
    return () => {
      current = false;
      window.clearInterval(timer);
    };
  }, [api, selected]);
  const updateView = useCallback(
    (job: Job) => {
      if (selection.current === selected)
        setView((current) => (current?.job_id === job.job_id ? job : current));
    },
    [selected],
  );
  const pollingError = useCallback(
    (message: string) => {
      if (selection.current === selected) setError(message);
    },
    [selected],
  );
  useJobPolling(
    view,
    api.job,
    updateView,
    pollingError,
    "Could not refresh saved job. Accepted work continues on the server.",
    view?.job_id,
  );
  async function action(run: () => Promise<unknown>) {
    const context = selected;
    setBusy(true);
    setError("");
    try {
      await run();
      if (mounted.current && selection.current === context && context) {
        const result = await api.session(context);
        if (mounted.current && selection.current === context) setDetail(result);
      }
    } catch (cause) {
      if (mounted.current && selection.current === context)
        setError(cause instanceof Error ? cause.message : "Request failed.");
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  async function retry(id: string) {
    const key = retries.current.get(id) ?? crypto.randomUUID();
    retries.current.set(id, key);
    const context = selected;
    const job = await api.retryJob(id, key);
    retries.current.delete(id);
    showJob(job, context);
  }
  async function exportSamples(sessionId: string, jobId?: string) {
    const { blob, filename } = await api.exportSamples(sessionId, jobId);
    if (mounted.current) saveDownload(blob, filename);
  }
  async function download(job: Job) {
    if (timestampedExports) return exportSamples(selected, job.job_id);
    const blob =
      "targets" in job
        ? await api.downloadBatch(job.id)
        : await api.downloadGeneration(job.id);
    if (mounted.current)
      saveDownload(
        blob,
        `${"targets" in job ? "batch" : "generation"}-${job.id}.zip`,
      );
  }
  const open = detail?.session.status === "open";
  return (
    <section className="card" aria-labelledby="history-heading">
      {readOnly ? <h1 id="history-heading">Session History</h1> : <h2 id="history-heading">Session history</h2>}
      <p className="help">
        {readOnly ? "Choose a session to view its saved results." : "Browser disconnects leave accepted work running. Closing a session stops unfinished work and keeps saved results."}
      </p>
      <label>
        Saved session
        <select
          aria-label="Saved session"
          value={selected}
          onChange={(event) => chooseSession(event.target.value)}
        >
          <option value="">Choose a session</option>
          {sessions.map((session) => (
            <option key={session.id} value={session.id}>
              {sessionDate(session)} · {session.status}{" "}
              · {session.id.slice(0, 8)}
            </option>
          ))}
        </select>
      </label>
      {onNew && <button type="button" disabled={busy} onClick={onNew}>Start a new session</button>}
      {!sessions.length && !error && <p className="help">No saved sessions yet.</p>}
      {detail && (
        <>
          <p role="status">Session {detail.session.status}</p>
          {timestampedExports && detail.jobs.some(job => job.kind !== "fit" && job.artifacts_ready) && (
            <button type="button" className="primary" disabled={busy}
              onClick={() => action(() => exportSamples(detail.session.id))}>Download session results</button>
          )}
          {!readOnly && <button
            type="button"
            disabled={busy || !open}
            onClick={() =>
              action(async () => {
                await api.closeSession(detail.session.id);
                onClosed?.(detail.session.id);
              })
            }
          >
            Stop / close session
          </button>}
          {open && onResume && detail.experiments.map((experiment) => (
            <button key={experiment.id} type="button" disabled={busy} onClick={() => onResume(experiment)}>
              Resume setup {experiment.id.slice(0, 8)}
            </button>
          ))}
          <div className="correlations-scroll">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Operation</th>
                  <th>State</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {detail.jobs.map((job) => (
                  <tr key={job.job_id}>
                    <td>
                      {job.kind} {job.id.slice(0, 8)}
                      {job.retry_of_job_id && <small> · retry attempt</small>}
                    </td>
                    <td>{job.state}</td>
                    <td>
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() =>
                          action(async () => {
                            showJob(await api.job(job.job_id), selected);
                          })
                        }
                      >
                        View {job.id.slice(0, 8)}
                      </button>
                      {!readOnly && ["queued", "running", "cancelling"].includes(
                        job.state,
                      ) && (
                        <button
                          type="button"
                          disabled={busy || job.state === "cancelling"}
                          onClick={() =>
                            action(async () => {
                              showJob(
                                await api.cancelJob(job.job_id),
                                selected,
                              );
                            })
                          }
                        >
                          Stop job {job.id.slice(0, 8)}
                        </button>
                      )}
                      {!readOnly && open && job.retryable && (
                        <button
                          type="button"
                          disabled={
                            busy ||
                            detail.jobs.some(
                              (item) =>
                                item.retry_of_job_id === job.job_id &&
                                ["queued", "running", "cancelling"].includes(
                                  item.state,
                                ),
                            )
                          }
                          onClick={() => action(() => retry(job.job_id))}
                        >
                          Retry{" "}
                          {job.kind === "batch"
                            ? "unfinished targets"
                            : job.kind}{" "}
                          {job.id.slice(0, 8)}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {error && (
        <p role="alert" className="notice error">
          {error}
        </p>
      )}
      {view && (
        <div className="fit-status" aria-label="Saved result">
          <h3>
            Saved{" "}
            {"correlations" in view
              ? "fit"
              : "targets" in view
                ? "batch"
                : "generation"}{" "}
            · {view.state}
          </h3>
          <small>{view.id}</small>
          {view.error && <p>{view.error}</p>}
          {"correlations" in view ? (
            <>
              <p>
                Best channel: {view.best_channel?.channel_id ?? "Unavailable"}
              </p>
              <div className="correlations-scroll">
                <table className="history-table">
                  <tbody>
                    {view.correlations.map((row) => (
                      <tr key={row.channel_id}>
                        <td>{row.channel_id}</td>
                        <td>{row.correlation?.toFixed(3) ?? "Unavailable"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : "targets" in view ? (
            <ul>
              {view.targets.map((target) => (
                <li key={target.id}>
                  {target.channel_id} · {target.objective} · {target.state} ·{" "}
                  {target.clips.length} clips
                </li>
              ))}
            </ul>
          ) : (
            <p>
              {view.channel_id} · {view.objective} · {view.clips.length} clips
            </p>
          )}
          {!("correlations" in view) && view.artifacts_ready && (
            <button
              type="button"
              disabled={busy}
              onClick={() => action(() => download(view))}
            >
              Download saved samples ZIP
            </button>
          )}
        </div>
      )}
    </section>
  );
}
