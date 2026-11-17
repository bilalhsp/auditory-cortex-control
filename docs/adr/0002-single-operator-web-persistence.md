# Authenticate one operator and separate metadata from scientific files

The replacement serves a public web URL for a single operator, so commands and
saved results require authentication even though multi-user collaboration is
outside scope. Use SQLite for application-session and job metadata and the
filesystem for recordings, models, and generated samples, preserving scientific
artifact compatibility instead of moving those artifacts into a database.
Completed results survive session closure and backend restart; interrupted work
requires manual retry, and backup implementation is outside the initial scope.
