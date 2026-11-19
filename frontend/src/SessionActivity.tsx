import { useEffect, useState } from "react";
import type { makeApi } from "./api";

export default function SessionActivity({ api, sessionId }: {
  api: ReturnType<typeof makeApi>;
  sessionId: string;
}) {
  const [open, setOpen] = useState(false);
  const [events, setEvents] = useState<Awaited<ReturnType<typeof api.activity>>>();
  const [stale, setStale] = useState(false);

  useEffect(() => {
    if (!open) return;
    let stopped = false;
    let timer: number | undefined;
    async function refresh() {
      try {
        const items = await api.activity(sessionId);
        if (!stopped) {
          setEvents(items);
          setStale(false);
        }
      } catch {
        if (!stopped) setStale(true);
      }
      if (!stopped) timer = window.setTimeout(refresh, 2500);
    }
    void refresh();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [api, sessionId, open]);

  return (
    <details className="session-activity" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Activity</summary>
      {stale && <p role="status">Can’t refresh activity. Reconnect to see updates.</p>}
      {!events && !stale && <p>Loading activity…</p>}
      {events?.length === 0 && <p>No jobs have been submitted in this session.</p>}
      {!!events?.length && <ol aria-label="Latest session activity">
        {events.map(({ at, message }, index) => <li key={`${at}-${message}-${index}`}>
          <time dateTime={at}>{new Date(at).toLocaleString()}</time> {message}
        </li>)}
      </ol>}
    </details>
  );
}
