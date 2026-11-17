# Separate browser connections from computation and saved results

For the single-operator application, accepted work continues when the browser
disconnects, while explicit Stop/Close session terminates running work and
releases its GPUs. Completed samples remain downloadable and discoverable in
previous-session history, so connection lifetime and session closure must not
control result deletion; storage technology, restart recovery, and retention
limits remain undecided.
