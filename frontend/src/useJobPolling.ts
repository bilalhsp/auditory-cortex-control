import { useEffect } from "react";

export function isActiveJobState(state?: string): boolean {
  return state === "queued" || state === "running" || state === "cancelling";
}

export function useJobPolling<T extends { id: string; state: string }>(
  operation: T | null,
  read: (id: string) => Promise<T>,
  update: (value: T) => void,
  reportError: (message: string) => void,
  failureMessage: string,
  pollIdentity?: string,
) {
  const id = pollIdentity ?? operation?.id;
  const state = operation?.state;
  useEffect(() => {
    if (!id || !isActiveJobState(state)) return;
    let current = true;
    const timer = window.setInterval(() => {
      read(id)
        .then((result) => {
          if (current) {
            update(result);
            reportError("");
          }
        })
        .catch(() => {
          if (current) reportError(failureMessage);
        });
    }, 1500);
    return () => {
      current = false;
      window.clearInterval(timer);
    };
  }, [id, state, read, update, reportError, failureMessage]);
}
