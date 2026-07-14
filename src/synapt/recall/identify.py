"""Structure-aware identify — recall consolidation step ① (Phase A, recall#868 wiring).

The generic flattened-cluster identify was NO-GO (Atlas config#481: source-support
91.5% < 98%, clean-atomic 40% << 85% — the 3B fabricates/promotes/drifts when asked
to re-find durable units in a flattened blob). The fix: `JournalEntry` (journal.py)
already carries SEPARATE structured fields — `focus`, `done`, `decisions`,
`next_steps` — and every durable gold unit lives in `done`/`decisions`. So DON'T
flatten and re-segment; read the structure directly.

Two DETERMINISTIC sub-steps — NO LLM anywhere in identify:
  1a `prefilter` — collect `entry.done[i]` + `entry.decisions[i]` with free attribution;
     `focus`/`next_steps` are NEVER read.
  1b `split_candidate` — a CONSERVATIVE deterministic clause-splitter partitions a compound
     item at high-precision fact boundaries (`; ` and sentence `. `, paren/abbrev-guarded);
     atomic candidates pass through untouched.

Why deterministic, not a narrow LLM call: the split-fidelity gate (2026-07-13, Modal
ap-nCK7lZHLtkD7Jrek8KDTY4) measured a 3B narrow-split on the 65 real compound candidates at
BF16 (the upper-bound screen; 4-bit is strictly worse) and it FAILED — 24.6% word-salad
(shatters a note into ~1-word-per-line), 44% seed-stable, plus real reword/drop/meaning-
errors. That is false-memory injection at the split step, which precision-first cannot
tolerate. A deterministic partition CANNOT confab, reword, drop, or reorder — every piece is
a verbatim, ordered, non-overlapping substring of the source; its only failure modes are
UNDER-split (graceful — one merged unit, recovered downstream by extract_batch) and
over-split, which the high-precision markers + a paren-depth guard minimize (0 fabricated
boundaries across the 65-case corpus). This is the design-note's own pre-planned fallback
(line 151), promoted to primary.

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
from dataclasses import dataclass

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


# --- 1b — deterministic conservative clause-splitter (recall, NO LLM) --------------
#
# Fidelity-first: partition a compound item into atomic units at HIGH-precision fact
# boundaries ONLY, so every piece is a verbatim, ordered, non-overlapping substring of the
# source. Prefer UNDER-split (leave merged; graceful — extract_batch re-extracts downstream)
# over OVER-split (a false boundary fabricates a non-fact — the precision-killer). Split
# markers: `; ` (semicolon) and a sentence boundary (`.`/`!`/`?` + whitespace + a fact-start
# char), BOTH only at paren/bracket depth 0, with an abbreviation guard. NOT bare commas,
# NOT bare " and ", NOT " + " (measured to over-split method chains + parenthetical lists).

# Common abbreviations whose trailing period is NOT a sentence boundary.
_ABBREVIATIONS = (
    "e.g", "i.e", "etc", "vs", "cf", " al", "approx",
    "dr", "mr", "ms", "mrs", "sr", "jr", "st", "fig", " no", "inc", "ltd", "corp",
)
# Sentence boundary: end punctuation + whitespace + a fact-start char (Capital / digit /
# quote / open-paren). Decimals, versions, and filenames ("0.5.0", "eval.py") lack the
# space after the dot and are therefore naturally safe.
_SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+(?=[A-Z0-9\"'(])")
_SEMICOLON = re.compile(r";\s+")


def _paren_depth(text: str, index: int) -> int:
    head = text[:index]
    return (head.count("(") - head.count(")")) + (head.count("[") - head.count("]"))


def _ends_with_abbreviation(text: str, dot_index: int) -> bool:
    tail = text[max(0, dot_index - 6):dot_index].lower()
    return any(tail.endswith(abbr) for abbr in _ABBREVIATIONS)


def _split_on(text: str, marker: re.Pattern, keep_left: int) -> list[str]:
    """Partition ``text`` at ``marker`` matches sitting at paren/bracket depth 0.
    ``keep_left`` is how many chars of the match stay on the left piece (1 keeps sentence
    punctuation, 0 drops the semicolon). Content is preserved; only the delimiter run and
    inter-piece whitespace are dropped, so pieces stay verbatim substrings of the source."""
    cuts: list[tuple[int, int]] = []
    for match in marker.finditer(text):
        if _paren_depth(text, match.start()) != 0:
            continue
        if keep_left and _ends_with_abbreviation(text, match.start()):
            continue
        cuts.append((match.start() + keep_left, match.end()))
    if not cuts:
        return [text]
    pieces, start = [], 0
    for cut, resume in cuts:
        piece = text[start:cut].strip()
        if piece:
            pieces.append(piece)
        start = resume
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces or [text]


def _split_markers(text: str) -> list[str]:
    """The conservative partition: semicolons, then sentence boundaries (both depth-0)."""
    pieces = [text]
    pieces = [p for chunk in pieces for p in _split_on(chunk, _SEMICOLON, 0)]
    pieces = [p for chunk in pieces for p in _split_on(chunk, _SENTENCE_BOUNDARY, 1)]
    return [p.strip() for p in pieces if p.strip()]


def is_compound(text: str) -> bool:
    """True iff the deterministic splitter partitions ``text`` into more than one unit —
    i.e. it carries a high-precision fact boundary. Advisory predicate; ``split_candidate``
    is safe to call on any candidate regardless (an atomic one returns unchanged)."""
    return len(_split_markers(text)) > 1


def split_candidate(candidate: Candidate) -> list[Candidate]:
    """1b — deterministic conservative split of a candidate into atomic units. Each unit is
    a verbatim, ordered, non-overlapping substring of the source and inherits the parent's
    attribution (+ ``split_index``). A candidate with no high-precision boundary returns
    unchanged (single-element list) — graceful under-split, never a fabricated unit."""
    pieces = _split_markers(candidate.text)
    if len(pieces) <= 1:
        return [candidate]
    return [
        Candidate(text=piece, attr=dict(candidate.attr), split_index=n)
        for n, piece in enumerate(pieces)
    ]


def identify(cluster: list[JournalEntry]) -> list[Candidate]:
    """Full structure-aware identify — fully deterministic (NO LLM): prefilter each entry's
    `done`/`decisions` into candidates, then conservatively split any compound candidate."""
    units: list[Candidate] = []
    for candidate in prefilter(cluster):
        units.extend(split_candidate(candidate))
    return units
