import { useState } from "react";
import { checkBackendConnection } from "./api";

const defaultCommand =
  "bash connect-gilbreth.sh --login gilbreth-fe02 --node gilbreth-n6 --frontend-origin https://bilalhsp.github.io";

export default function BackendConnection() {
  const command = window.AUDITORY_CONFIG?.connectionCommand ?? defaultCommand;
  const [checking, setChecking] = useState(false);
  const [reachable, setReachable] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [copyStatus, setCopyStatus] = useState("");

  async function copyCommand() {
    try {
      await navigator.clipboard.writeText(command);
      setCopyStatus("Command copied. Run it in Git Bash on your laptop.");
    } catch {
      setCopyStatus("Select and copy the command above, then run it in Git Bash.");
    }
  }
  async function check() {
    setChecking(true);
    setReachable(null);
    setError("");
    try {
      const base = await checkBackendConnection();
      setReachable(base || window.location.origin);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Backend connection check failed.");
    } finally {
      setChecking(false);
    }
  }
  return (
    <section className="connection-panel" aria-label="Backend connection">
      <h2>Connect to your backend</h2>
      <p className="help">
        Run the launcher in Git Bash on your laptop. Its terminal discovers the
        allocated node, starts the backend, and opens the SSH tunnel.
      </p>
      <pre><code>{command}</code></pre>
      <div className="connection-actions">
        <button type="button" onClick={copyCommand}>Copy launcher command</button>
        <button type="button" onClick={check} disabled={checking}>
          {checking ? "Checking backend…" : "Check backend connection"}
        </button>
      </div>
      {copyStatus && <p role="status" className="help">{copyStatus}</p>}
      <p className="help">
        After the terminal reports Connected, check here, then sign in below.
        This checks the forwarded application; node and service startup progress
        appear in the launcher terminal. Allow browser local-network access if prompted.
      </p>
      {reachable && <p role="status">Backend reachable: {reachable}</p>}
      {error && <p role="alert" className="notice error">{error}</p>}
    </section>
  );
}
