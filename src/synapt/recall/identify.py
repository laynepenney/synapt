"""Structure-aware identify — recall consolidation step ① (Phase A, recall#868 wiring).

The generic flattened-cluster identify was NO-GO (Atlas config#481: source-support
91.5% < 98%, clean-atomic 40% << 85% — the 3B fabricates/promotes/drifts when asked
to re-find durable units in a flattened blob). The fix: `JournalEntry` (journal.py)
already carries SEPARATE structured fields — `focus`, `done`, `decisions`,
`next_steps` — and every durable gold unit lives in `done`/`decisions`. So DON'T
flatten and re-segment; read the structure directly.

identify is a single DETERMINISTIC step — the `prefilter` (NO LLM): collect
`entry.done[i]` + `entry.decisions[i]` with free attribution; `focus`/`next_steps` are
NEVER read. It does NOT atomize compound items — atomization is extract_batch's job.

Why there is no split step (a narrow-split sub-step was built, measured, and dropped):
  1. A 3B FREE split ("emit atomic facts one per line") word-salads — 24.6% at BF16, the
     upper-bound screen (split-fidelity gate 2026-07-13, Modal ap-nCK7lZHLtkD7Jrek8KDTY4).
  2. A deterministic clause-splitter is lexically safe (verbatim) but semantically
     OVER-splits (anaphoric fragments, initialisms) — reviewer-2 (Sentinel + Atlas).
  3. The DECISIVE measurement (Modal ap-y0V0GsYUHqY49KAiFF6pmI): extract_batch's STRUCTURED
     extraction atomizes compounds cleanly — 0/65 word-salad, 62/65 ok, 0 confab, 36/39 of
     the would-be-under-split cases atomized. So the split is REDUNDANT (Opus decided,
     path b): `prefilter` → extract_batch DIRECTLY on raw candidates; extract_batch atomizes.
The ~5% extract_batch drop-to-empty is the honest cost, arbitrated by the ≤-legacy dogfood.
Full evidence: config/design/results/{split-fidelity,extract-atomization}-probe-2026-07-13/.

Design-note config §RECONCILE: the prefilter's done/decisions-only choice is PRECISION-FIRST
and carries a KNOWN, MEASURED recall gap of ~7.3% — the ~10 config#481 gold units that live
in rich `focus` lines (concentrated in the dogfood-04 "summary-in-focus" outlier). Accepted
for v1 and NOT recovered by an LLM focus-classifier: that reintroduces the exact 3B
fabrication the NO-GO proved, and a false memory pollutes worse than a missing one omits. The
dogfood ≤-legacy measure is the real arbiter; a v2 recovery, if needed, is a DETERMINISTIC
heuristic, never an LLM classifier.

Boundary: OSS — recall consolidation is a core primitive.
"""

from __future__ import annotations

from dataclasses import dataclass

from synapt.recall.journal import JournalEntry

# The only journal fields that carry durable units (per config#481 gold: every unit
# sources from done[N] or decisions[N]). focus/next_steps are excluded structurally.
_DURABLE_FIELDS = ("done", "decisions")


@dataclass
class Candidate:
    """A pre-identified durable-unit candidate plus its structural attribution.

    ``attr`` = ``{session_id, entry_index, field, index}`` — every downstream envelope is
    traceable to the exact journal field it came from. ``entry_index`` (position in the
    cluster) is the within-cluster UNIQUENESS key; ``batch_unit_id`` namespaces it by
    cluster for global uniqueness.
    """

    text: str
    attr: dict


def prefilter(cluster: list[JournalEntry]) -> list[Candidate]:
    """1a — deterministic prefilter (NO LLM). Read the structured `done`/`decisions`
    of each entry as candidate units; NEVER read `focus`/`next_steps`.

    Attribution is free: ``entry_index`` (the entry's position in the cluster) +
    field name + list index, plus ``session_id`` for provenance. ``entry_index`` is
    the UNIQUENESS key, not ``session_id`` — journal entries frequently carry an empty
    or repeated ``session_id`` (an agent may write the journal several times per
    session, and older entries lack one), so a session-keyed id collides within a
    cluster (and would trip extract_batch's dup-id guard downstream). This deletes the
    entire junk / next-promoted / control-fabrication failure mass before any
    inference — structurally, not by a model. Empty/whitespace items are skipped.
    """
    candidates: list[Candidate] = []
    for entry_index, entry in enumerate(cluster):
        for fieldname in _DURABLE_FIELDS:
            items = getattr(entry, fieldname, None) or []
            for index, item in enumerate(items):
                if isinstance(item, str) and item.strip():
                    candidates.append(
                        Candidate(
                            text=item.strip(),
                            attr={
                                "session_id": entry.session_id,
                                "entry_index": entry_index,
                                "field": fieldname,
                                "index": index,
                            },
                        )
                    )
    return candidates


def batch_unit_id(cluster_id: str, candidate: Candidate) -> str:
    """The stable, GLOBALLY-unique id for a candidate's extract_batch ``BatchUnit`` (Phase B).

    Namespaced by ``cluster_id`` because ``entry_index`` only resets per cluster (Atlas,
    reviewer-2): a batch spanning clusters would collide on ``entry_index`` alone and trip
    extract_batch's dup-id guard. Form: ``{cluster_id}:{entry_index}:{field}:{index}``. The
    id rides into the extraction envelope as ``source_unit_id`` (out-of-band), so a
    merge/split/drop stays detectable back to the exact journal field.
    """
    attr = candidate.attr
    return f"{cluster_id}:{attr['entry_index']}:{attr['field']}:{attr['index']}"


def identify(cluster: list[JournalEntry]) -> list[Candidate]:
    """The identify step (Phase A) — a single deterministic pass, exactly the `prefilter`.
    It does NOT atomize compound items; extract_batch's structured extraction does that
    downstream (measured clean: config/design/results/extract-atomization-probe-2026-07-13).
    Kept as the named step-① entry point for the consolidation pipeline + Phase-B wiring."""
    return prefilter(cluster)
