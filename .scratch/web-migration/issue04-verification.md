# Issue 04 verification

- Complete current API suite: 32 passing HTTP/ASGI tests, including upload
  authentication, declared/chunked size limits, configured capabilities/CORS,
  format and neural/trial validation, both stimulus families, duplicate/unsafe
  filenames, retention, restart persistence, isolation, and storage failures.
- Invalid uploads do not replace the previous selection or leave staged files
  under normal writable-storage conditions. Valid uploads have independent UUID
  directories and server-generated names; private path/size/hash metadata stays
  out of public responses. Older uploads can be reselected within their experiment.
- Real recording smoke: the representative ucdavisBAK MAT recording validates
  for TIMIT and mVocs. Source data is read-only; new copies are isolated under
  the configured cache. The independent backend archive also validates and
  attaches the real recording when run outside the repository.
- Browser verification uses actual React/FastAPI services and MAT fixtures.
  Pending, valid, invalid, incompatible, and oversize feedback; same-name
  replacements; existing/upload switching; out-of-order revisions; context
  changes; and mobile layout all pass without page JavaScript errors.
- Python static checks report zero errors/warnings; TypeScript checking,
  production frontend build, and independent backend archive build pass.

Review identified and corrected two regressions:

1. A request cancelled while SQLite registration was blocked could delete an
   artifact before the worker committed its selection. A real HTTP/ASGI test
   holds an external SQLite write lock and cancels the request, without replacing
   application services. It reproduced the missing-file bug. Publication,
   registration, and failure cleanup now belong to a protected finalizer; the
   regression verifies committed metadata retains the file.
2. Forward-slash stimulus paths passed metadata discovery but were incompatible
   with the unchanged scientific upload reader. An HTTP regression reproduced
   acceptance; validation now requires its backslash path convention and rejects
   unsupported files before attachment rather than rewriting source data.

No new Python environment or dependencies were installed, no Gradio was added,
and the scientific package, root Python lockfile/configuration, and original
research recordings/models remain unchanged. No fitting/generation jobs were
started. Future uploaded-recording fitting must resolve the private registry
path by identity and scope fitted artifacts independently of legacy session zero.
