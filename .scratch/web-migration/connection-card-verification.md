# GitHub owner and connection card verification

Request: use GitHub username bilalhsp and provide a web connection action or
terminal connection to make the Gilbreth startup/tunnel workflow clearer.
The operator explicitly selected **Connect/check button with launcher command**
rather than installing a local helper for browser-triggered SSH.

## Changes

- Launcher defaults, fresh backend CORS configuration, deployment examples, and
  the public launcher command now use `https://bilalhsp.github.io`. The Pages
  repository URL is `https://bilalhsp.github.io/auditory-cortex-control/`.
  Research operator credentials and SSH username remain unchanged.
- The sign-in page now offers Copy launcher command and Check backend connection.
  The command runs in laptop Git Bash; SSH/MFA and node/service startup progress
  remain in that terminal. The check verifies the forwarded application's identity
  without credentials. It does not claim a browser page can execute SSH or prove
  allocation/worker startup independently of the terminal launcher.
- A failed recheck clears the prior success message. Sign-in, endpoint pinning,
  scientific commands, and downloads retain their existing behavior.
- An explicit connection check probes fixed URLs as well as candidate lists;
  fixed-URL/legacy sign-in behavior stays compatible. Public runtime config can
  override connectionCommand. The local frontend server supports the corresponding
  --connection-command option without modifying the built assets.
- The sidebar home link respects Vite's base path for the Pages repository URL.
  No local daemon, extension, browser shell execution, SSH secret collection, or
  new dependency was introduced.

## Verification

The browser check first failed against the old frontend because the requested
button did not exist. It passes against the new built frontend with real FastAPI
and isolated metadata: exact command copied to the clipboard, unauthenticated
fixed-URL health/identity check, offline failure clearing success, recovery,
normal authenticated sign-in, and no horizontal overflow at mobile width.
The actual runtime --connection-command option is also checked over HTTP.
Reproduction: PYTHONPATH="$PWD" .venv/bin/python
.scratch/web-migration/connection-card-browser.py (one command line).
The probe uses the existing cached Chromium/Playwright tools and configured cache.

Three targeted backend launch tests and seven CLI tests pass, including the
bilalhsp default CORS origin and laptop command origin. Backend Pyright reports
zero errors; frontend TypeScript and normal production build pass. The Pages
subdirectory build is checked separately. Final full-suite/review results are
recorded when complete.

## Scope

Public publishing and changing a running deployment are separate actions. The
updated frontend must be published and backend launcher restarted to use the new
UI/defaults. Windows SSH/MFA and an actual published site's local-network
permissions still require the operator's laptop, as in the prior launcher check.
No scientific algorithms, models, saved samples, or passwords were changed.
