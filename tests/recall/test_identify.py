"""Gold-test gate for structure-aware identify — PREFILTER-ONLY, ATTRIBUTION-BOUND + mutation-proof.

Phase A is a single deterministic step: `prefilter` reads structured `done`/`decisions`, never
`focus`/`next_steps`; there is no split step (the atomization measurement showed extract_batch
atomizes compounds cleanly — see config/design/results/extract-atomization-probe-2026-07-13/).

This gate consumes Atlas's FROZEN authoritative gold-source map (revision 2, SHA
cba2b465…; the SINGLE authority — the gold-units.jsonl `source` LABELS are unreliable, see the
map's `legacy_source_label_corrections`). Each gold binds to its EXACT
(cluster_id, entry_index, field, index) tuple(s) — no global token pool (the prior hollowness
both reviewers caught: an unrelated candidate could satisfy any gold, and a whole cluster could
vanish and stay green). The DELETION-MUTATION regression (remove ANY of the 18 clusters → gate
RED, incl. zero-gold controls) is a PERMANENT negative control so the gate can never go hollow.

Candidate index = prefilter's ACTUAL output on the 7 atlas clusters (the ratified reconstruction
fixture) UNION the map's frozen done/decisions tuples for the 11 dogfood clusters (recall tests
do not hold the dogfood structured source — the slice has no cluster labels and the bare-label
gold is unanchorable; the map is Atlas's single authority). prefilter's CODE is validated
against actual output on the atlas clusters (same code path for all 18); the deletion mutation
gives every binding teeth.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from synapt.recall.identify import batch_unit_id, prefilter
from synapt.recall.identify import Candidate

_FIX = Path(__file__).parent.parent / "fixtures" / "identify"
_MAP = json.loads((_FIX / "gold-source-map.json").read_text())
_ATLAS = json.loads((_FIX / "atlas-journal-structured.json").read_text())

_MAPPINGS = _MAP["mappings"]
_CONTRACT = _MAP["test_contract"]
_COUNTS = _MAP["expected_counts"]
_REQUIRED = set(_CONTRACT["required_cluster_ids"])
_ZERO_GOLD = set(_CONTRACT["zero_gold_control_clusters"])
_RETAINED = [m for m in _MAPPINGS if m["prefilter_expected"]]


def _tuple(cluster_id: str, attr: dict) -> tuple:
    return (cluster_id, attr["entry_index"], attr["field"], attr["index"])


def _atlas_index() -> dict:
    """ACTUAL prefilter output on the ratified atlas clusters, keyed by tuple."""
    index: dict = {}
    for cluster_id, entries in _ATLAS.items():
        cluster = [
            SimpleNamespace(session_id=e.get("session_id", ""), focus=e.get("focus", ""),
                           done=e.get("done", []), decisions=e.get("decisions", []),
                           next_steps=e.get("next_steps", []))
            for e in entries
        ]
        for cand in prefilter(cluster):
            index[(cluster_id, cand.attr["entry_index"], cand.attr["field"], cand.attr["index"])] = cand
    return index


def _dogfood_frozen_index() -> dict:
    """The map's frozen done/decisions tuples for dogfood clusters (Atlas's authority)."""
    index: dict = {}
    for m in _MAPPINGS:
        if m["cluster_id"].startswith("dogfood"):
            for a in m["attributions"]:
                if a["field"] in ("done", "decisions"):
                    index[_tuple(m["cluster_id"], a)] = m["gold_id"]
    return index


def _full_index(exclude: str | None = None) -> dict:
    idx = {**_atlas_index(), **_dogfood_frozen_index()}
    if exclude is not None:
        idx = {k: v for k, v in idx.items() if k[0] != exclude}
    return idx


def _gate_passes(exclude: str | None = None) -> bool:
    """The WHOLE gate as a boolean, so the deletion mutation can assert it flips to False:
    the 18-cluster inventory is complete AND every retained gold binds (all its tuples resolve
    to an emitted candidate in its OWN cluster)."""
    inventory = ({m["cluster_id"] for m in _MAPPINGS} | _ZERO_GOLD)
    if exclude is not None:
        inventory = inventory - {exclude}
    if inventory != _REQUIRED:
        return False
    index = _full_index(exclude=exclude)
    for m in _RETAINED:
        for a in m["attributions"]:
            if _tuple(m["cluster_id"], a) not in index:
                return False
    return True


# --- the frozen authority + inventory ------------------------------------------------------

def test_frozen_map_is_the_authority():
    assert _MAP["artifact_revision"] == 2
    assert _MAP["source_revision"] in ("8a08bb2", "eafecf3")  # frozen against the ratified rev
    # the gold source LABELS are known-unreliable; the map corrects 7 of them
    assert len(_MAP["legacy_source_label_corrections"]) == 7
    assert _MAP["publication_provenance"]["supersedes_unpublished_sha256"].startswith("b93b7401")


def test_inventory_is_exactly_18_clusters_including_zero_gold_controls():
    assert _REQUIRED == set(_COUNTS["gold_by_cluster"])
    assert len(_REQUIRED) == 18
    assert _ZERO_GOLD == {"dogfood-02", "dogfood-09", "dogfood-10"}
    for z in _ZERO_GOLD:
        assert _COUNTS["gold_by_cluster"][z] == 0


# --- prefilter code, validated against ACTUAL output on the atlas clusters ------------------

def test_atlas_prefilter_actual_output_matches_frozen_map():
    index = _atlas_index()
    # count: 50 atlas candidates (238 all-cluster - 188 dogfood), from the ratified reconstruction
    assert len(index) == _COUNTS["all_cluster_structured_candidates"] - _COUNTS["dogfood_structured_candidates"]
    # prefilter never emits focus/next
    assert all(t[2] in ("done", "decisions") for t in index)
    # every atlas gold tuple resolves to an emitted candidate IN ITS OWN CLUSTER
    for m in _MAPPINGS:
        if m["cluster_id"].startswith("atlas-journal"):
            for a in m["attributions"]:
                assert _tuple(m["cluster_id"], a) in index, f"{m['gold_id']} {a} not emitted"


# --- the CORE: attribution-binding, no global pool -----------------------------------------

def test_every_retained_gold_binds_to_a_candidate_in_its_own_cluster():
    index = _full_index()
    for m in _RETAINED:
        for a in m["attributions"]:
            t = _tuple(m["cluster_id"], a)
            assert t in index, f"{m['gold_id']} tuple {t} does not bind"
            assert t[0] == m["cluster_id"]           # bound in its OWN cluster, not a global pool
    # D007-G13 is the sole dual — BOTH done[13] and done[14] must resolve
    dual = next(m for m in _MAPPINGS if m["gold_id"] == "D007-G13")
    assert len(dual["attributions"]) == 2
    assert all(_tuple(dual["cluster_id"], a) in index for a in dual["attributions"])


def test_gap_gold_is_not_captured_by_prefilter():
    index = _full_index()
    gap = [m for m in _MAPPINGS if not m["prefilter_expected"]]
    for m in gap:
        # every gap attribution is focus/next and is NOT in the done/decisions candidate index
        for a in m["attributions"]:
            assert a["field"] in ("focus", "next_steps")
            assert _tuple(m["cluster_id"], a) not in index


def test_prefilter_expected_consistency_and_documented_gap():
    for m in _MAPPINGS:
        is_dd = all(a["field"] in ("done", "decisions") for a in m["attributions"])
        assert m["prefilter_expected"] == is_dd     # the map's own rule, verified
    gap_by_cluster = Counter(m["cluster_id"] for m in _MAPPINGS if not m["prefilter_expected"])
    assert dict(gap_by_cluster) == {"dogfood-01": 2, "dogfood-04": 6, "dogfood-06": 2}


# --- THE CYCLE-ENDER: deletion-mutation regression (permanent negative control) -------------

def test_clean_gate_passes():
    assert _gate_passes() is True


@pytest.mark.parametrize("cluster_id", sorted(_REQUIRED))
def test_deletion_of_any_cluster_fails_the_gate(cluster_id):
    """Remove any one of the 18 clusters (incl. a zero-gold control) → the gate MUST go RED.
    Gold-bearing clusters fail via attribution (their gold's tuples unbind); zero-gold controls
    fail via the inventory. This is baked in so the gate can never be hollow again."""
    assert _gate_passes(exclude=cluster_id) is False


def test_attribution_teeth_are_independent_of_inventory():
    """Prove the attribution binding (not just the inventory) carries the mutation: removing a
    cluster that HAS retained gold, candidates only (inventory untouched), still unbinds its gold.
    (Gap-only clusters — dogfood-01/06 — and zero-gold controls carry the mutation via inventory,
    covered by test_deletion_of_any_cluster_fails_the_gate.)"""
    clusters_with_retained = {m["cluster_id"] for m in _RETAINED}
    for cid in sorted(clusters_with_retained):
        index = _full_index(exclude=cid)
        unbound = any(
            _tuple(m["cluster_id"], a) not in index
            for m in _RETAINED if m["cluster_id"] == cid
            for a in m["attributions"]
        )
        assert unbound, f"removing {cid}'s candidates left its gold bound — hollow"


# --- Phase-B id scheme + counts ------------------------------------------------------------

def test_batch_unit_id_is_globally_unique_across_clusters():
    # over the DISTINCT candidate tuples — many gold may share one candidate (compound items),
    # so the id namespaces CANDIDATES, not gold units.
    tuples = set(_full_index().keys())
    namespaced, bare = [], []
    for cluster_id, entry_index, field, index in tuples:
        cand = Candidate(text="x", attr={"entry_index": entry_index, "field": field, "index": index})
        namespaced.append(batch_unit_id(cluster_id, cand))
        bare.append(f"{entry_index}:{field}:{index}")
    assert len(bare) > len(set(bare))               # entry_index is cluster-local → bare collides
    assert len(namespaced) == len(set(namespaced))  # cluster namespace → globally unique


def test_counts_match_the_frozen_contract():
    assert len(_MAPPINGS) == _COUNTS["gold_units"] == 137
    assert sum(len(m["attributions"]) for m in _MAPPINGS) == _COUNTS["source_attributions"] == 138
    assert len(_RETAINED) == _COUNTS["prefilter_retained_gold_units"] == 127
    assert len(_MAPPINGS) - len(_RETAINED) == _COUNTS["prefilter_gap_gold_units"] == 10
    multi = [m for m in _MAPPINGS if len(m["attributions"]) > 1]
    assert len(multi) == _COUNTS["multi_attribution_gold_units"] == 1
    assert multi[0]["gold_id"] == "D007-G13"
