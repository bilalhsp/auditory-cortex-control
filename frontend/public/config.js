// Public runtime configuration: never include credentials here.
// A nonempty apiBaseUrl selects one fixed backend and overrides apiBaseUrls.
// Empty plus candidates discovers an available laptop SSH tunnel at sign-in.
// After sign-in the selected endpoint stays pinned; jobs never fail over silently.
window.AUDITORY_CONFIG = {
  apiBaseUrl: "",
  connectionCommand: "bash connect-gilbreth.sh --login gilbreth-fe02 --node gilbreth-n6 --frontend-origin https://bilalhsp.github.io",
  apiBaseUrls: [
    "http://127.0.0.1:28000",
    "http://127.0.0.1:28001",
    "http://127.0.0.1:28002",
    "http://127.0.0.1:18000",
    "http://127.0.0.1:8000",
  ],
};
