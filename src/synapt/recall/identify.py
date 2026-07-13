"""Structure-aware identify — recall consolidation step ① (Phase A, recall#868 wiring).

The generic flattened-cluster identify was NO-GO (Atlas config#481: source-support
91.5% < 98%, clean-atomic 40% << 85% — the 3B fabricates/promotes/drifts when asked
to re-find durable units in a flattened blob). The fix: `JournalEntry` (journal.py)
already carries SEPARATE structured fields — `focus`, `done`, `decisions`,
`next_steps` — and every durable gold unit lives in `done`/`decisions`. So DON'T
flatten and re-segment; read the structure directly.

Two sub-steps; the model appears only in the second, narrowly:
  1a `prefilter` — deterministic (NO LLM): collect `entry.done[i]` + `entry.decisions[i]`
     with free attribution; `focus`/`next_steps` are NEVER read.
  1b `split` — narrow LLM call on COMPOUND candidates only (a `done` string carrying
     >1 fact); atomic candidates pass through untouched.

Design-note config 3030967 §RECONCILE (Opus, Layne-pending): the prefilter's
done/decisions-only choice is PRECISION-FIRST and carries a KNOWN, MEASURED recall
gap of ~7.3% — the ~10 config#481 gold units that live in rich `focus` lines
(concentrated in the dogfood-04 "summary-in-focus" outlier). This is accepted for
v1 and NOT recovered by an LLM focus-classifier: that reintroduces the exact 3B
fabrication the NO-GO proved, and a false memory pollutes worse than a missing one
omits. The dogfood ≤-legacy measure is the real arbiter; a v2 recovery, if needed,
is a DETERMINISTIC heuristic, never an LLM classifier.

Boundary: OSS — recall consolidation is a core primitive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from synapt.recall.journal import JournalEntry

# The only journal fields that carry durable units (per config#481 gold: every unit
# sources from done[N] or decisions[N]). focus/next_steps are excluded structurally.
_DURABLE_FIELDS = ("done", "decisions")


@dataclass
class Candidate:
    """A pre-identified durable-unit candidate plus its structural attribution.

    ``attr`` = ``{session_id, field, index}`` (+ ``split`` once a compound is split),
    so every downstream envelope is traceable to the exact journal field it came from.
    """

    text: str
    attr: dict
    split_index: int | None = None


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


# --- 1b — compound detection (deterministic pretest) + narrow LLM split -----------

# Conservative multi-clause markers. Start narrow (design-note "open impl detail":
# tune the false-negative rate in validation; a missed compound degrades gracefully
# to one under-split unit, unlike the generic path's fabrication).
_SEMICOLON = "; "
_MULTI_CLAUSE = re.compile(
    r",\s+(?:and|then|but|which|while|after|before)\s|\band\s+(?:also|then)\b",
    re.IGNORECASE,
)


def is_compound(text: str, *, length_threshold: int = 240) -> bool:
    """Conservative pretest: does this candidate plausibly carry >1 fact (→ split)?

    Flags a semicolon-joined list, a clear multi-clause conjunction, or an over-length
    item. Everything else passes through atomic with NO model call. Conservative-first
    by design — false negatives (missed compounds) degrade gracefully; the LLM surface
    stays as small as possible.
    """
    if _SEMICOLON in text:
        return True
    if len(text) > length_threshold:
        return True
    return bool(_MULTI_CLAUSE.search(text))


_SPLIT_PROMPT = (
    "Split this single journal note into separate atomic facts, one per line. "
    "Only split — do not add, infer, reword, or drop anything. "
    "If it is already a single fact, return it unchanged.\n\nNote: {text}"
)


def split_candidate(candidate: Candidate, infer: Callable[[str], str]) -> list[Candidate]:
    """1b — narrow LLM split of a COMPOUND candidate. Fidelity-first: only split, no
    add/infer/reword/drop. Split units inherit the parent's attribution (+ split_index).

    Non-compound candidates should not reach here (caller gates on ``is_compound``).
    A blank/degenerate model response falls back to the unsplit candidate — never drops.
    """
    completion = infer(_SPLIT_PROMPT.format(text=candidate.text))
    lines = [ln.strip() for ln in (completion or "").splitlines() if ln.strip()]
    if not lines:
        return [candidate]
    return [
        Candidate(text=line, attr=dict(candidate.attr), split_index=n)
        for n, line in enumerate(lines)
    ]


def identify(cluster: list[JournalEntry], infer: Callable[[str], str] | None = None) -> list[Candidate]:
    """Full structure-aware identify: prefilter → split compounds. ``infer`` is the
    narrow split model call; if omitted, compound candidates pass through unsplit
    (deterministic-only mode, for the prefilter gate / no-model contexts)."""
    units: list[Candidate] = []
    for candidate in prefilter(cluster):
        if infer is not None and is_compound(candidate.text):
            units.extend(split_candidate(candidate, infer))
        else:
            units.append(candidate)
    return units
