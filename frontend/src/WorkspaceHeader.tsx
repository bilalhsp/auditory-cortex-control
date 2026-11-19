import { useEffect, useRef, useState } from "react";
import { checkBackendConnection } from "./api";

type Props = {
  checkConnection?: () => Promise<boolean>;
  signedIn: boolean;
  username: string;
  busy: boolean;
  hasSession: boolean;
  onSignOut: () => void;
};

export default function WorkspaceHeader({ checkConnection, signedIn, username, busy, hasSession, onSignOut }: Props) {
  const [reachable, setReachable] = useState<boolean | null>(null);
  const account = useRef<HTMLDetailsElement>(null);
  const accountButton = useRef<HTMLElement>(null);

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    async function poll() {
      const connected = checkConnection
        ? await checkConnection()
        : await checkBackendConnection().then(() => true, () => false);
      if (stopped) return;
      setReachable(connected);
      timer = window.setTimeout(poll, 4000);
    }
    void poll();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [checkConnection]);

  const status = reachable === null ? "Checking connection" : reachable ? "Connected" : "Disconnected";

  return (
    <header className="workspace-header">
      <a className="brand" href={import.meta.env.BASE_URL}>
        <span className="brand-mark" aria-hidden="true">∿</span>
        <span>Auditory Cortex Control</span>
      </a>
      <div className="header-controls">
        <span role="status" className="backend-status">
          <span aria-hidden="true" className={`status-led ${reachable === null ? "" : status.toLowerCase()}`} />
          {status}
        </span>
        {signedIn && (
          <details className="account" ref={account} onKeyDown={(event) => {
            if (event.key === "Escape" && account.current?.open) {
              account.current.open = false;
              accountButton.current?.focus();
            }
          }}>
            <summary ref={accountButton} aria-label="Account">{username.trim().slice(0, 1).toUpperCase() || "?"}</summary>
            <div className="account-actions">
              <button type="button" disabled={busy} onClick={onSignOut}>{hasSession ? "End session and sign out" : "Sign out"}</button>
            </div>
          </details>
        )}
      </div>
    </header>
  );
}
