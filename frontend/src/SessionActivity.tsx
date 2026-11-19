import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { makeApi } from "./api";

export default function SessionActivity({ api, sessionId, visible }: {
  api: ReturnType<typeof makeApi>;
  sessionId: string;
  visible: boolean;
}) {
  const [open, setOpen] = useState(true);
  const [events, setEvents] = useState<Awaited<ReturnType<typeof api.activity>>>();
  const log = useRef<HTMLOListElement>(null);
  const followLatest = useRef(true);
  const [stale, setStale] = useState(false);

  useEffect(() => {
    if (!open || !visible) return;
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
  }, [api, sessionId, open, visible]);

  useLayoutEffect(() => {
    if (visible && open && followLatest.current && log.current) {
      log.current.scrollTop = log.current.scrollHeight;
    }
  }, [events, open, visible]);

  return (
    <details className="session-activity" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Activity</summary>
      {stale && <p role="status">Can’t refresh activity. Reconnect to see updates.</p>}
      {!events && !stale && <p>Loading activity…</p>}
      {events?.length === 0 && <p>No jobs have been submitted in this session.</p>}
      {!!events?.length && <ol ref={log} tabIndex={0} aria-label="Session activity, oldest first" onScroll={(event) => {
        const list = event.currentTarget;
        if (list.clientHeight) {
          followLatest.current = list.scrollHeight - list.scrollTop - list.clientHeight <= 8;
        }
      }}>
        {[...events].reverse().map(({ at, message }) => <li key={`${at}-${message}`}>
          <time dateTime={at}>{new Date(at).toLocaleString()}</time> {message}
        </li>)}
      </ol>}
    </details>
  );
}
