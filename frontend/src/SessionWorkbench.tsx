import { useEffect, useState } from "react";
import type { Experiment, ResumableSession, makeApi } from "./api";
import { sessionDate } from "./sessionDate";

type Props = {
  api: ReturnType<typeof makeApi>;
  sessionId: string | null;
  experimentId?: string;
  busy: boolean;
  onResume: (experiment: Experiment) => void;
  onStartOver: () => void;
};

export default function SessionWorkbench({
  api, sessionId, experimentId, busy, onResume, onStartOver,
}: Props) {
  const [sessions, setSessions] = useState<ResumableSession[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    async function refresh() {
      try {
        const result = await api.resumableSessions();
        if (!stopped) {
          setSessions(result);
          setError(false);
        }
      } catch {
        if (!stopped) setError(true);
      }
      if (!stopped) timer = window.setTimeout(refresh, 4000);
    }
    void refresh();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [api, sessionId, experimentId]);

  const groups = new Map<string, ResumableSession[]>();
  for (const entry of sessions ?? []) {
    const group = groups.get(entry.session.id);
    if (group) group.push(entry);
    else groups.set(entry.session.id, [entry]);
  }

  return (
    <section className="card session-workbench" aria-labelledby="workbench-heading">
      <div className="workbench-heading">
        <h2 id="workbench-heading">Workbench</h2>
        {sessionId && <button type="button" className="text-button" disabled={busy || !experimentId}
          onClick={onStartOver}>Start over</button>}
      </div>
      <h3>Resume a session</h3>
      {error && <p className="notice error" role="alert">Could not refresh sessions. Reconnect to try again.</p>}
      {!sessions && !error && <p className="help">Loading sessions…</p>}
      {sessions?.length === 0 && !error && <p className="help">No sessions to resume. Set up an experiment below.</p>}
      {!!sessions?.length && (
        <ul className="resume-list">
          {[...groups.values()].map((entries) => {
            const { session } = entries[0];
            return (
              <li key={session.id} className="resume-session">
                <h4>{sessionDate(session)}</h4>
                <ul className="resume-list">
                  {entries.map(({ experiment, active_work }) => (
                    <li key={experiment.id} className="resume-row">
                      <div>
                        <span>{experiment.recording?.display_name ??
                          `${experiment.configuration.stimulus_family} / ${experiment.configuration.model_name}`}</span>
                        <small>{active_work ? "Work in progress" : "Setup in progress"}</small>
                      </div>
                      <button type="button" className="text-button" disabled={busy || error || experimentId === experiment.id}
                        onClick={() => onResume(experiment)}>
                        {experimentId === experiment.id ? "Current experiment" : "Resume"}
                      </button>
                    </li>
                  ))}
                </ul>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
