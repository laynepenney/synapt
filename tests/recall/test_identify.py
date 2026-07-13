"""Gold-test gate for structure-aware identify (recall#868 wiring, Phase A).

Executable spec against Atlas's frozen config#481 corpus (design/results/
identify-step-probe-2026-07-13, SHA-pinned fixtures copied into tests/fixtures/
identify/). The DETERMINISTIC prefilter carries most of the load and is validated
model-free here; the narrow SPLIT's fidelity is a separate Modal 3B gate (paced).

Reconciled target (design-note config 3030967 §RECONCILE, Opus, Layne-pending):
  124 gold source from done/decisions  → prefilter MUST capture (primary gate)
    3 gold are multi-field (focus AND done/decisions) → capturable via done/decisions
   10 gold are pure focus/next          → KNOWN, MEASURED recall gap (~7.3%), documented
The prefilter is done/decisions-only (precision-first); the ~10-unit gap is accepted
for v1, NOT recovered by an LLM classifier (would reintroduce the NO-GO fabrication).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from synapt.recall.consolidate import _read_all_entries, cluster_journal_entries
from synapt.recall.identify import Candidate, is_compound, prefilter

_FIX = Path(__file__).parent.parent / "fixtures" / "identify"
_GOLD = [json.loads(line) for line in (_FIX / "gold-units.jsonl").read_text().splitlines() if line.strip()]


def _classify_source(source: str) -> str:
    """Classify a gold unit's source-field attribution vs the done/decisions-only prefilter."""
    s = source.lower()
    focus_or_next = ("focus" in s) or ("next" in s)
    done_or_decisions = ("done" in s) or ("decision" in s)
    if focus_or_next and done_or_decisions:
        return "multi_field"        # a focus+done/decisions unit → capturable via done/decisions
    if focus_or_next:
        return "pure_focus_next"    # lives only in focus/next → documented recall gap
    return "done_decisions"         # prefilter-capturable


def _dogfood_clusters():
    """Reconstruct the 11 dogfood clusters exactly as consolidation does — the same
    loader + clustering the real path uses, on the SHA-pinned structured journal slice."""
    entries = _read_all_entries(_FIX / "dogfood-journal-slice.jsonl")
    rich = [e for e in entries if e.has_rich_content()]
    return cluster_journal_entries(rich)


# --- gold field-attribution: the reconciled target (model-free, the validation bulk) ---

def test_gold_field_attribution_matches_reconcile():
    counts = Counter(_classify_source(g["source"]) for g in _GOLD)
    assert len(_GOLD) == 137
    assert counts["done_decisions"] == 124   # primary gate: prefilter must capture these
    assert counts["multi_field"] == 3        # capturable via their done/decisions component
    assert counts["pure_focus_next"] == 10   # the accepted, documented recall gap


def test_documented_recall_gap_is_measured_and_small():
    counts = Counter(_classify_source(g["source"]) for g in _GOLD)
    gap = counts["pure_focus_next"] / len(_GOLD)
    # ~7.3%, concentrated in the dogfood-04 summary-in-focus outlier. Precision-first
    # accepts it for v1; the dogfood <=-legacy measure is the real arbiter (Phase B).
    assert 0.06 <= gap <= 0.08


# --- prefilter structural correctness on the real dogfood clusters (deterministic) ---

def test_prefilter_reads_only_done_decisions_never_focus_next():
    clusters = _dogfood_clusters()
    assert clusters, "expected dogfood clusters"
    focus_next_material = set()
    for cluster in clusters:
        for entry in cluster:
            if entry.focus.strip():
                focus_next_material.add(entry.focus.strip())
            for item in entry.next_steps:
                if item.strip():
                    focus_next_material.add(item.strip())

    for cluster in clusters:
        for cand in prefilter(cluster):
            assert cand.attr["field"] in ("done", "decisions")
            assert isinstance(cand.attr["index"], int)
            assert isinstance(cand.attr["entry_index"], int)  # uniqueness key
            assert cand.text.strip()
            # exclusion: no candidate is focus/next material
            assert cand.text not in focus_next_material


def test_prefilter_covers_every_done_decisions_item():
    """The prefilter candidate set is exactly the non-empty done/decisions items across
    the clusters — so it covers 100% of where the 124 done/decisions gold live."""
    clusters = _dogfood_clusters()
    for cluster in clusters:
        expected = 0
        for entry in cluster:
            expected += sum(1 for x in entry.done if isinstance(x, str) and x.strip())
            expected += sum(1 for x in entry.decisions if isinstance(x, str) and x.strip())
        cands = prefilter(cluster)
        assert len(cands) == expected
        # round-trip: every candidate's text is exactly its attributed structured item,
        # keyed on entry_index (session_id is NOT unique — often empty/repeated)
        for cand in cands:
            entry = cluster[cand.attr["entry_index"]]
            items = getattr(entry, cand.attr["field"])
            assert items[cand.attr["index"]].strip() == cand.text
        # attribution uniqueness — the Phase-B BatchUnit id must not collide even when
        # session_id is empty/repeated within the cluster (extract_batch dup-id guard)
        keys = [(c.attr["entry_index"], c.attr["field"], c.attr["index"]) for c in cands]
        assert len(keys) == len(set(keys))


def test_prefilter_is_deterministic_no_model():
    clusters = _dogfood_clusters()
    a = [(c.text, c.attr["field"], c.attr["index"]) for cl in clusters for c in prefilter(cl)]
    b = [(c.text, c.attr["field"], c.attr["index"]) for cl in clusters for c in prefilter(cl)]
    assert a == b and a, "prefilter must be pure/deterministic"


# --- compound-detection pretest (conservative-first; the split's own fidelity is Modal) ---

def test_is_compound_flags_semicolon_and_multiclause_passes_atomic():
    assert is_compound("Merged PR #12; filed issue #13; bumped version to 0.2.0")
    assert is_compound("x" * 300)  # over-length
    assert is_compound("Wrote the parser, and then verified it against the fixtures")
    # atomic single facts pass through (no model call)
    assert not is_compound("The recall package is open source.")
    assert not is_compound("Grip issue #763 was filed for the tomllib migration follow-up.")
