# Domain docs

## Before exploring

Read the root `GLOSSARY.md` and ADRs in `docs/adr/` relevant
to the area being explored.

If these files do not exist, proceed silently. Do not suggest
creating them upfront. The domain-modeling skill creates them
when terms or decisions are resolved.

## Layout

This repository uses a single-context layout:

- `GLOSSARY.md`: shared domain vocabulary.
- `docs/adr/`: architectural decision records.

## Use the glossary's vocabulary

Use defined domain terms in issues, proposals, hypotheses,
and tests. Avoid synonyms the glossary explicitly excludes.

If a concept is missing, reconsider whether it belongs in the
project or note the gap for domain-modeling.

## Flag ADR conflicts

Explicitly identify any proposal that contradicts an existing
ADR, citing the ADR and explaining why it should be reconsidered.
