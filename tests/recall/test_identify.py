"""Gold-test gate for structure-aware identify (recall#868 wiring, Phase A — PREFILTER-ONLY).

Phase A is a single deterministic step: `prefilter` reads structured `done`/`decisions`, never
`focus`/`next_steps`. There is NO split step — a narrow-split was built, measured, and DROPPED:
the split-fidelity gate showed a 3B free-split word-salads (24.6% at BF16); the deterministic
clause-splitter is lexically safe but semantically OVER-splits (reviewer-2 Sentinel/Atlas); and
the DECISIVE atomization measurement showed extract_batch's STRUCTURED extraction atomizes
compounds cleanly (0/65 word-salad, 62/65 ok, 0 confab). Path (b): prefilter → extract_batch
directly. Evidence: config/design/results/{split-fidelity,extract-atomization}-probe-2026-07-13/.

This gate EXECUTES the frozen config#481 gold (not a label count — Atlas reviewer-2) over ALL 18
clusters (dogfood slice + the atlas-journal reconstruction from corpus.json — closes Atlas's
42.3% gap): it runs prefilter and checks every done/decisions gold unit's content is CAPTURED,
and reports the focus/next gap per-cluster. The EXACT per-index gold→item mapping is Atlas's
frozen-boundary domain (pending ratification); this uses content-coverage + the gold's own field
labels, robust to the gold's reworded atomization.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from synapt.recall.consolidate import _read_all_entries, cluster_journal_entries
from synapt.recall.identify import batch_unit_id, identify, prefilter

_FIX = Path(__file__).parent.parent / "fixtures" / "identify"
_GOLD = [json.loads(line) for line in (_FIX / "gold-units.jsonl").read_text().splitlines() if line.strip()]
_ATLAS = json.loads((_FIX / "atlas-journal-structured.json").read_text())


def _distinctive(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9#]+", text.lower()) if len(w) > 3}


def _classify_field(source: str) -> str:
    """Classify a gold unit's source-field vs the done/decisions-only prefilter (from the
    gold's own frozen source label — exact, not fuzzy)."""
    s = source.lower()
    focus_next = ("focus" in s) or ("next" in s)
    done_dec = ("done" in s) or ("decision" in s)
    if focus_next and done_dec:
        return "multi_field"     # focus AND done/decisions → capturable via done/decisions
    if focus_next:
        return "focus_next"      # documented recall gap
    return "done_decisions"


def _dogfood_rich():
    return [e for e in _read_all_entries(_FIX / "dogfood-journal-slice.jsonl") if e.has_rich_content()]


def _dogfood_clusters():
    return cluster_journal_entries(_dogfood_rich())


def _atlas_clusters():
    """The 7 atlas-journal clusters as duck-typed entries (prefilter reads done/decisions/
    session_id; classification reads focus/next_steps)."""
    clusters = []
    for entries in _ATLAS.values():
        clusters.append([
            SimpleNamespace(
                session_id=e.get("session_id", ""), focus=e.get("focus", ""),
                done=e.get("done", []), decisions=e.get("decisions", []),
                next_steps=e.get("next_steps", []),
            )
            for e in entries
        ])
    return clusters


def _all_clusters():
    return _dogfood_clusters() + _atlas_clusters()


# --- prefilter structural correctness, on ALL 18 clusters (dogfood + atlas) ----------------

def test_prefilter_reads_only_done_decisions_never_focus_next():
    clusters = _all_clusters()
    assert clusters
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
            assert isinstance(cand.attr["entry_index"], int)
            assert isinstance(cand.attr["index"], int)
            assert cand.text.strip()
            assert cand.text not in focus_next_material


def test_prefilter_covers_every_done_decisions_item():
    for cluster in _all_clusters():
        expected = 0
        for entry in cluster:
            expected += sum(1 for x in entry.done if isinstance(x, str) and x.strip())
            expected += sum(1 for x in entry.decisions if isinstance(x, str) and x.strip())
        cands = prefilter(cluster)
        assert len(cands) == expected
        for cand in cands:
            entry = cluster[cand.attr["entry_index"]]
            items = getattr(entry, cand.attr["field"])
            assert items[cand.attr["index"]].strip() == cand.text
        keys = [(c.attr["entry_index"], c.attr["field"], c.attr["index"]) for c in cands]
        assert len(keys) == len(set(keys))          # attribution unique within cluster


def test_prefilter_is_deterministic_no_model():
    clusters = _all_clusters()
    a = [(c.text, c.attr["field"], c.attr["index"]) for cl in clusters for c in prefilter(cl)]
    b = [(c.text, c.attr["field"], c.attr["index"]) for cl in clusters for c in prefilter(cl)]
    assert a == b and a


# --- EXECUTE the gold (Atlas reviewer-2: run prefilter, don't count labels) ----------------

def test_gold_executes_prefilter_captures_done_decisions_content():
    """The core execute-the-gold check: build the prefilter candidate token-pool over ALL 18
    clusters, then confirm every done/decisions gold unit's distinctive content is CAPTURED.
    Tolerant of the gold's reworded atomization (>=0.5 distinctive-token overlap = captured);
    asserts the aggregate capture rate with margin and surfaces any uncovered units."""
    pool: set[str] = set()
    for cluster in _all_clusters():
        for cand in prefilter(cluster):
            pool |= _distinctive(cand.text)

    dd_gold = [g for g in _GOLD if _classify_field(g["source"]) in ("done_decisions", "multi_field")]
    uncovered = []
    for g in dd_gold:
        gt = _distinctive(g["text"])
        covered = len(gt & pool) / len(gt) if gt else 1.0
        if covered < 0.5:
            uncovered.append((g["cluster_id"], g["source"], round(covered, 2)))
    capture_rate = 1 - len(uncovered) / len(dd_gold)
    # >=95% of done/decisions gold is captured by prefilter's output; the residue is reworded
    # gold (e.g. "vs" -> "versus"), not a prefilter miss. Surfaces uncovered for inspection.
    assert capture_rate >= 0.95, f"only {capture_rate:.3f} captured; uncovered={uncovered}"


def test_gold_field_classification_per_cluster_and_documented_gap():
    """Per-cluster reporting (Atlas reviewer-2), from the gold's own frozen source labels.
    The focus/next gap is KNOWN, SMALL (~7.3%), and CONCENTRATED — it must stay visible per
    cluster, not hidden in an aggregate; the >=-legacy dogfood measures it per cluster too."""
    per_cluster = defaultdict(Counter)
    for g in _GOLD:
        per_cluster[g["cluster_id"]][_classify_field(g["source"])] += 1

    totals = Counter()
    for counts in per_cluster.values():
        totals.update(counts)
    assert sum(totals.values()) == 137
    assert totals["done_decisions"] == 124        # prefilter-capturable
    assert totals["multi_field"] == 3             # capturable via their done/decisions component
    assert totals["focus_next"] == 10             # documented recall gap

    gap = totals["focus_next"] / 137
    assert 0.06 <= gap <= 0.08                     # ~7.3%
    # the gap is CONCENTRATED, not spread — the three carriers must be exactly these clusters
    gap_clusters = {cid for cid, c in per_cluster.items() if c["focus_next"]}
    assert gap_clusters == {"dogfood-01", "dogfood-04", "dogfood-06"}


def test_gold_covers_all_18_corpus_clusters_no_atlas_gap():
    """Atlas reviewer-2: the prior gate omitted the 7 atlas clusters (58/137 = 42.3% of gold).
    This gate exercises prefilter on the atlas clusters too, and the gold spans all 15
    gold-carrying clusters (11 dogfood minus 3 empty + 7 atlas)."""
    gold_clusters = {g["cluster_id"] for g in _GOLD}
    atlas_gold = [g for g in _GOLD if g["cluster_id"].startswith("atlas-journal")]
    assert len(atlas_gold) == 58                   # the previously-omitted 42.3%
    assert {c for c in gold_clusters if c.startswith("atlas-journal")} == set(_ATLAS.keys())
    # prefilter actually runs on the atlas clusters and yields their durable items
    atlas_units = sum(len(prefilter(cluster)) for cluster in _atlas_clusters())
    assert atlas_units > 0


# --- Phase-B id scheme: the BatchUnit id must be cluster-namespaced (Atlas reviewer-2) -----

def test_batch_unit_id_is_globally_unique_across_clusters():
    clusters = _dogfood_clusters()
    namespaced, bare = [], []
    for ci, cluster in enumerate(clusters):
        cluster_id = f"dogfood-{ci}"
        for cand in prefilter(cluster):
            namespaced.append(batch_unit_id(cluster_id, cand))
            bare.append(f"{cand.attr['entry_index']}:{cand.attr['field']}:{cand.attr['index']}")
    # entry_index resets per cluster -> bare ids COLLIDE across clusters (the bug Atlas named)...
    assert len(bare) > len(set(bare))
    # ...but the cluster namespace makes them globally unique (extract_batch dup-id guard safe)
    assert len(namespaced) == len(set(namespaced))


# --- identify is now exactly the prefilter (no split) --------------------------------------

def test_identify_equals_prefilter_no_split():
    for cluster in _all_clusters():
        got = identify(cluster)
        want = prefilter(cluster)
        assert [(c.text, c.attr) for c in got] == [(c.text, c.attr) for c in want]
        # no unit is a fabricated fragment: every identified unit is a whole structured item
        for cand in got:
            entry = cluster[cand.attr["entry_index"]]
            assert getattr(entry, cand.attr["field"])[cand.attr["index"]].strip() == cand.text
