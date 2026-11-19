# Issue tracker: Local Markdown

Issues and specs live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`.
- Spec: `.scratch/<feature-slug>/spec.md`.
- One file per implementation ticket:
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`,
  numbered from 01. Never combine tickets into one file.
- Record state in a `Status:` line near the top.
- Append conversation under `## Comments`.

## Publishing and fetching

To publish, create the appropriate file and parent directories.
To fetch a ticket, read its referenced path. Resolve issue numbers
within the relevant feature directory.

## Wayfinding operations

- Map: `.scratch/<effort>/map.md`, containing Notes,
  Decisions-so-far, and Fog.
- Child ticket: `.scratch/<effort>/issues/NN-<slug>.md`.
- Record `Type: research`, `prototype`, `grilling`, or `task`.
- Record dependencies as `Blocked by: NN, NN`.
- An issue is unblocked when all listed dependencies are resolved.
- Select the lowest-numbered open, unblocked, unclaimed ticket.
- Claim it by saving `Status: claimed` before starting work.
- Resolve it by appending `## Answer`, setting `Status: resolved`,
  and adding a summary and link to the map's Decisions-so-far.
