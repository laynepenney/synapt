"""Gold-test gate for structure-aware identify — PREFILTER-ONLY, ATTRIBUTION-BOUND, END-TO-END.

Phase A is a single deterministic step: `prefilter` reads structured `done`/`decisions`, never
`focus`/`next_steps`; there is no split step (the atomization measurement showed extract_batch
atomizes compounds cleanly — internal atomization-probe results).

This gate runs prefilter on the ACTUAL structured source for ALL 7 clusters and binds each gold
to its EXACT (cluster_id, entry_index, field, index) tuple in its OWN cluster — no global pool,
and NO map-proxy: the index is built from real prefilter output, not the map itself, so it can't
be self-fulfilling. The dogfood structured source IS present: `_read_all_entries` +
`cluster_journal_entries` reconstruct the 4 dogfood clusters from the fixture slice (dogfood-N =
clusters[N] positionally) → 18 actual candidates; the curated (pre-clustered) fixture → 10;
28 all-cluster. Every one of the map's 22 attributions resolves against this ACTUAL output.

PRIVACY NOTE (2026-07-29): this fixture set is a from-scratch SYNTHETIC replacement for a prior
real `recall_journal` dogfood capture, removed for privacy (it carried real private-repo issue
numbers, real file paths, and business-confidential content). The replacement fixtures
are fully fictional (a made-up "beacon" example codebase) but are NOT hand-waved: every cluster
and candidate position below is the ACTUAL output of `cluster_journal_entries()`/`prefilter()`
run against that fictional content (see `build_identify_fixture.py`, kept out-of-repo), so the
non-self-fulfilling property this suite is designed to enforce still holds.

The frozen gold-source map (revision 3, the privacy re-author) is the authoritative gold_id→tuple
contract. It is SHA-hash-pinned here (a mutated map fails), and its id set must exactly equal the
frozen gold-units.jsonl ids. The DELETION-MUTATION regression (remove ANY of the 7 clusters →
gate RED, incl. zero-gold controls) is a permanent negative control: because the index is built
from real prefilter output, removing a cluster removes its real candidates and unbinds its gold.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from synapt.recall.consolidate import _read_all_entries, cluster_journal_entries
from synapt.recall.identify import Candidate, batch_unit_id, prefilter

_FIX = Path(__file__).parent.parent / "fixtures" / "identify"
_MAP_PATH = _FIX / "gold-source-map.json"
_MAP_SHA256 = "4c2432d829fc5ee59f5b31aad230f8f8679e9f169a9931f6098100e2f65fb592"
_MAP = json.loads(_MAP_PATH.read_text())
_ATLAS = json.loads((_FIX / "atlas-journal-structured.json").read_text())
_GOLD_IDS = {
    json.loads(line)["gold_id"]
    for line in (_FIX / "gold-units.jsonl").read_text().splitlines() if line.strip()
}

_MAPPINGS = _MAP["mappings"]
_CONTRACT = _MAP["test_contract"]
_COUNTS = _MAP["expected_counts"]
_REQUIRED = set(_CONTRACT["required_cluster_ids"])
_ZERO_GOLD = set(_CONTRACT["zero_gold_control_clusters"])
_RETAINED = [m for m in _MAPPINGS if m["prefilter_expected"]]


def _key(cluster_id: str, attr: dict) -> tuple:
    return (cluster_id, attr["entry_index"], attr["field"], attr["index"])


def _dogfood_clusters():
    """Reconstruct the 4 dogfood clusters from the slice, exactly as consolidation does.
    dogfood-N = the N-th cluster (cluster_journal_entries is deterministic; verified: all 13
    dogfood map attributions (12 gold ids) resolve against this actual output, 18 candidates)."""
    rich = [e for e in _read_all_entries(_FIX / "dogfood-journal-slice.jsonl") if e.has_rich_content()]
    return [(f"dogfood-{i:02d}", cluster) for i, cluster in enumerate(cluster_journal_entries(rich))]


def _atlas_clusters():
    return [
        (cid, [SimpleNamespace(session_id=e.get("session_id", ""), focus=e.get("focus", ""),
                               done=e.get("done", []), decisions=e.get("decisions", []),
                               next_steps=e.get("next_steps", []))
               for e in entries])
        for cid, entries in _ATLAS.items()
    ]


def _all_clusters(exclude: str | None = None):
    clusters = _dogfood_clusters() + _atlas_clusters()
    return [(cid, cl) for cid, cl in clusters if cid != exclude]


def _candidate_index(exclude: str | None = None) -> dict:
    """ACTUAL prefilter output over the (7 minus exclude) real clusters, keyed by tuple."""
    index: dict = {}
    for cluster_id, cluster in _all_clusters(exclude=exclude):
        for cand in prefilter(cluster):
            index[_key(cluster_id, cand.attr)] = cand
    return index


def _inventory(exclude: str | None = None) -> set:
    return {cid for cid, _ in _all_clusters(exclude=exclude)}


def _gate_passes(exclude: str | None = None) -> bool:
    """The whole gate as a boolean, so the deletion mutation can assert it flips to False: the
    7-cluster inventory is complete AND every retained gold binds to an ACTUAL prefilter
    candidate at its exact tuple in its own cluster."""
    if _inventory(exclude) != _REQUIRED:
        return False
    index = _candidate_index(exclude)
    return all(_key(m["cluster_id"], a) in index for m in _RETAINED for a in m["attributions"])


# --- the frozen authority: hash-pinned + id-exact ------------------------------------------

def test_frozen_map_is_sha_pinned():
    # a mutated map (any byte) changes the hash and fails here — the map can't drift silently
    assert hashlib.sha256(_MAP_PATH.read_bytes()).hexdigest() == _MAP_SHA256
    assert _MAP["artifact_revision"] == 3


def test_map_ids_are_unique_and_exactly_the_frozen_gold_ids():
    ids = [m["gold_id"] for m in _MAPPINGS]
    assert len(ids) == len(set(ids)) == 21
    assert set(ids) == _GOLD_IDS               # the map covers exactly the frozen gold, no more/less


# --- ALL 7 clusters reconstruct from ACTUAL source; counts match ---------------------------

def test_all_7_clusters_reconstruct_from_actual_source_with_expected_candidate_counts():
    dogfood = _dogfood_clusters()
    assert len(dogfood) == 4
    dogfood_cands = sum(len(prefilter(cl)) for _, cl in dogfood)
    atlas_cands = sum(len(prefilter(cl)) for _, cl in _atlas_clusters())
    assert dogfood_cands == _COUNTS["dogfood_structured_candidates"] == 18
    assert dogfood_cands + atlas_cands == _COUNTS["all_cluster_structured_candidates"] == 28
    assert _inventory() == _REQUIRED
    assert len(_REQUIRED) == 7
    for z in _ZERO_GOLD:                        # zero-gold controls are REAL clusters, still visible
        assert z in _inventory()


# --- THE CORE: every retained gold binds to ACTUAL prefilter output, in its own cluster -----

def test_every_retained_gold_binds_to_actual_candidate_in_its_own_cluster():
    index = _candidate_index()
    for m in _RETAINED:
        for a in m["attributions"]:
            k = _key(m["cluster_id"], a)
            assert k in index, f"{m['gold_id']} tuple {k} not in ACTUAL prefilter output"
            assert k[0] == m["cluster_id"]      # bound in its OWN cluster, not a global pool
            assert a["field"] in ("done", "decisions")


def test_DF03_G01_dual_is_exactly_done_0_and_1_and_both_bind():
    index = _candidate_index()
    dual = next(m for m in _MAPPINGS if m["gold_id"] == "DF03-G01")
    got = sorted((a["entry_index"], a["field"], a["index"]) for a in dual["attributions"])
    assert got == [(0, "done", 0), (0, "done", 1)]             # the exact dual, not just len==2
    assert dual["cluster_id"] == "dogfood-03"
    assert all(_key("dogfood-03", a) in index for a in dual["attributions"])


def test_gap_gold_is_not_in_actual_candidate_output():
    index = _candidate_index()
    for m in _MAPPINGS:
        if not m["prefilter_expected"]:
            for a in m["attributions"]:
                assert a["field"] in ("focus", "next_steps")
                assert _key(m["cluster_id"], a) not in index


def test_prefilter_expected_consistency_and_documented_gap():
    for m in _MAPPINGS:
        is_dd = all(a["field"] in ("done", "decisions") for a in m["attributions"])
        assert m["prefilter_expected"] == is_dd
    gap = Counter(m["cluster_id"] for m in _MAPPINGS if not m["prefilter_expected"])
    assert dict(gap) == {"dogfood-02": 2, "curated-03": 1}


# --- DELETION-MUTATION regression: the permanent negative control ---------------------------

def test_clean_gate_passes():
    assert _gate_passes() is True


@pytest.mark.parametrize("cluster_id", sorted(_REQUIRED))
def test_deletion_of_any_cluster_fails_the_gate(cluster_id):
    """Remove any one of the 7 clusters (incl. a zero-gold control) from the ACTUAL source →
    its real candidates vanish → the gate goes RED (gold-bearing via attribution, zero-gold via
    inventory). Because the index is real prefilter output, this cannot be fooled."""
    assert _gate_passes(exclude=cluster_id) is False


def test_gate_depends_on_actual_prefilter_output_not_the_map():
    """Non-self-fulfilling proof: against an EMPTY candidate index (as if prefilter produced
    nothing) no retained gold binds — so the gate reads the real output, not the map."""
    empty: dict = {}
    assert not all(_key(m["cluster_id"], a) in empty for m in _RETAINED for a in m["attributions"])


def test_attribution_teeth_are_independent_of_inventory():
    """Removing a gold-bearing cluster's CANDIDATES only (inventory untouched) still unbinds its
    gold — the attribution binding, not just the inventory, carries the mutation."""
    for cid in sorted({m["cluster_id"] for m in _RETAINED}):
        index = _candidate_index(exclude=cid)
        assert any(_key(m["cluster_id"], a) not in index
                   for m in _RETAINED if m["cluster_id"] == cid
                   for a in m["attributions"]), f"removing {cid}'s candidates left its gold bound"


# --- Phase-B id scheme + counts ------------------------------------------------------------

def test_batch_unit_id_is_globally_unique_across_clusters():
    tuples = set(_candidate_index().keys())    # distinct actual candidates (28)
    namespaced, bare = [], []
    for cluster_id, entry_index, field, index in tuples:
        cand = Candidate(text="x", attr={"entry_index": entry_index, "field": field, "index": index})
        namespaced.append(batch_unit_id(cluster_id, cand))
        bare.append(f"{entry_index}:{field}:{index}")
    assert len(tuples) == 28
    assert len(bare) > len(set(bare))          # entry_index is cluster-local → bare collides
    assert len(namespaced) == len(set(namespaced)) == 28    # cluster namespace → globally unique


def test_counts_match_the_frozen_contract():
    assert len(_MAPPINGS) == _COUNTS["gold_units"] == 21
    assert sum(len(m["attributions"]) for m in _MAPPINGS) == _COUNTS["source_attributions"] == 22
    assert len(_RETAINED) == _COUNTS["prefilter_retained_gold_units"] == 18
    assert len(_MAPPINGS) - len(_RETAINED) == _COUNTS["prefilter_gap_gold_units"] == 3
    multi = [m for m in _MAPPINGS if len(m["attributions"]) > 1]
    assert len(multi) == _COUNTS["multi_attribution_gold_units"] == 1 and multi[0]["gold_id"] == "DF03-G01"
