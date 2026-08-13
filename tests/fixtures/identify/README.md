# Fixture provenance

Every file in this directory is **synthetic**, authored for the identify
test cycle. None of it is captured from a real session, journal, store, or
agent.

The file names describe the **test role** of each fixture, not a data
source:

- `atlas-journal-structured.json` — exercises the structured-journal *shape*
  (named-field entries). It is not any agent's journal.
- `dogfood-journal-slice.jsonl` — exercises the line-delimited journal
  *format*. It is not a slice of dogfood data.
- `gold-units.jsonl` / `gold-source-map.json` — hand-authored expected
  outputs for the gold tests.

Tells that these are synthetic, visible in the bytes: placeholder
timestamps (`2026-01-01T00:00:00+00:00`), empty `session_id` and `branch`
fields, and generic invented engineering scenarios.

Rules for adding fixtures here:

1. Synthetic only. Never copy content from a real session, journal, or
   store, even "scrubbed" — author fresh content instead.
2. Zero or placeholder coordinates (timestamps, session ids, branches).
3. If a fixture must mirror the *shape* of a real defect, reproduce the
   shape with invented content and say so in the fixture's commit message.

Why this file exists: a public-branch audit flagged this directory because
the file names read as captured data. The audit's resolution confirmed
synthetic provenance from the authoring PR's own record; this README makes
that provenance legible at the directory itself so the names never trigger
an escalation again.
