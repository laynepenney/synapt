"""B2 gate — the action-decision pass (create/corroborate/contradict against existing knowledge).

Reframed gate (Opus, 2026-07-14, ratified after the frontier ideal was found to be an END-STATE
node-set — 0 action/existing_id/contradiction fields, 0 corpus reversals — not a per-fact
action-gold): this is a MECHANISM gate, not a corpus-accuracy gate. It proves the pass fires
correctly against SYNTHETIC existing-knowledge + facts (fake infer, zero model dependency, same
technique as B1): create-when-new, corroborate-when-match, and — the one Opus asked for
explicitly — a couple of synthetic CONTRADICTION cases, proving the mechanism CAN fire even
though this corpus has no reversals to validate against. Corpus-scale contradiction accuracy is
explicitly DESCOPED for v1 (0 examples in the 185-node frontier ideal) — the capability is KEPT
and documented, not dropped, per Opus's Q3 ratification.

The anti-inflation / 40-merge-group corroboration accuracy claim is NOT tested here — that is
the Phase-C dogfood's job (B2+B3 end-state vs the 185-node ideal), which needs the real
pluggable model, not a fake infer. This file proves the WIRING is correct; the dogfood proves
the QUALITY is correct.
"""

from __future__ import annotations

import pytest

pytest.importorskip("synapt.extract.batch")

from synapt.recall.journal import JournalEntry
from synapt.recall.knowledge import KnowledgeNode
from synapt.recall.consolidate import (
    _decide_actions,
    _flatten_envelope_facts,
    _run_coro_blocking,
    _extract_cluster_units,
)


def _entry(session_id="s1", *, done=None, decisions=None) -> JournalEntry:
    return JournalEntry(
        timestamp="2026-07-13T10:00:00Z", session_id=session_id,
        done=list(done or []), decisions=list(decisions or []),
    )


def _node(content, category="fact", node_id=None) -> KnowledgeNode:
    return KnowledgeNode.create(content=content, category=category, node_id=node_id)


def _envelope_ok(source_unit_id: str, *, facts=None, decisions=None):
    """A minimal fake BatchUnitResult-shaped object (status='ok') carrying a SynaptExtraction
    envelope, without needing a real extract_batch round trip — _flatten_envelope_facts /
    _decide_actions only read .status/.extraction/.source_unit_id."""
    from types import SimpleNamespace
    return SimpleNamespace(
        source_unit_id=source_unit_id,
        status="ok",
        extraction={
            "facts": facts or [],
            "decisions": decisions or [],
        },
        reason=None,
    )


def _envelope_failed(source_unit_id: str, reason="unparseable"):
    from types import SimpleNamespace
    return SimpleNamespace(source_unit_id=source_unit_id, status="failed", extraction=None, reason=reason)


def _real_ok_envelopes(cluster, cluster_id, infer):
    """Run the REAL B1 front-half (prefilter -> extract_batch) to get genuine envelopes for
    tests that want realism over hand-built fixtures."""
    results = _run_coro_blocking(_extract_cluster_units(cluster, cluster_id, infer))
    return [r for r in results if r.status == "ok"]


def _ok_envelope_infer(request):
    import json
    return json.dumps({
        "extracted_at": "2026-07-13T10:00:00Z",
        "facts": [{"text": f"fact from: {request['prompt'][:20]}"}],
        "decisions": [],
        "temporal_refs": [],
    })


# --- _flatten_envelope_facts -----------------------------------------------------------------

def test_flatten_extracts_facts_and_decisions_with_category_tagging():
    env = _envelope_ok(
        "c:0:done:0",
        facts=[{"text": "recall ships extract_batch", "category": "fact"}],
        decisions=[{"text": "chose the URI form for produced_by"}],
    )
    flat = _flatten_envelope_facts([env])
    assert len(flat) == 2
    by_text = {f["text"]: f for f in flat}
    assert by_text["recall ships extract_batch"]["category"] == "fact"
    assert by_text["chose the URI form for produced_by"]["category"] == "decision"
    assert all(f["source_unit_id"] == "c:0:done:0" for f in flat)


def test_flatten_skips_failed_envelopes():
    envs = [_envelope_failed("c:0:done:0"), _envelope_ok("c:0:done:1", facts=[{"text": "kept"}])]
    flat = _flatten_envelope_facts(envs)
    assert [f["text"] for f in flat] == ["kept"]


def test_flatten_skips_empty_or_missing_text():
    env = _envelope_ok("c:0:done:0", facts=[{"text": ""}, {}, {"text": "real"}])
    flat = _flatten_envelope_facts([env])
    assert [f["text"] for f in flat] == ["real"]


def test_flatten_defaults_fact_category_when_absent():
    env = _envelope_ok("c:0:done:0", facts=[{"text": "no category given"}])
    flat = _flatten_envelope_facts([env])
    assert flat[0]["category"] == "fact"


def test_flatten_multiple_envelopes_preserve_order_and_attribution():
    envs = [
        _envelope_ok("c:0:done:0", facts=[{"text": "first"}]),
        _envelope_ok("c:0:done:1", facts=[{"text": "second"}]),
    ]
    flat = _flatten_envelope_facts(envs)
    assert [f["text"] for f in flat] == ["first", "second"]
    assert [f["source_unit_id"] for f in flat] == ["c:0:done:0", "c:0:done:1"]


# --- _decide_actions: MECHANISM tests (synthetic, fake infer) --------------------------------

def test_decide_actions_empty_facts_returns_empty():
    assert _decide_actions([], "clu", [], [], lambda req: "{}") == []


def test_decide_actions_create_when_new():
    cluster = [_entry(done=["a genuinely new fact"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "a genuinely new fact"}])]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "create", "existing_id": null}]}'

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert len(out) == 1
    assert out[0]["action"] == "create"
    assert out[0]["existing_id"] is None
    assert out[0]["content"] == "a genuinely new fact"
    assert out[0]["category"] == "fact"
    assert out[0]["source_turns"] == []  # journal-field attribution, not transcript-turn-shaped


def test_decide_actions_corroborate_when_match():
    existing = [_node("recall wired extract_batch", node_id="kn_abc123")]
    cluster = [_entry(done=["recall wired extract_batch into consolidation"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "recall wired extract_batch into consolidation"}])]

    def infer(request):
        assert "kn_abc123" in request["prompt"]  # existing knowledge must be in-context
        return '{"actions": [{"index": 0, "action": "corroborate", "existing_id": "kn_abc123"}]}'

    out = _decide_actions(cluster, "clu", envs, existing, infer)
    assert out[0]["action"] == "corroborate"
    assert out[0]["existing_id"] == "kn_abc123"


def test_decide_actions_contradict_synthetic_proves_mechanism_can_fire():
    """Opus's explicit ask: a synthetic case proving contradiction CAN fire, since the frontier
    corpus has 0 real reversals to validate against at scale (descoped v1, capability KEPT)."""
    existing = [_node("we chose Ministral as the production model", node_id="kn_old_decision")]
    cluster = [_entry(decisions=["reversed course: switched production model to Qwen3.5-4B"])]
    envs = [_envelope_ok(
        "clu:0:decisions:0", decisions=[{"text": "reversed course: switched production model to Qwen3.5-4B"}]
    )]

    def infer(request):
        return (
            '{"actions": [{"index": 0, "action": "contradict", "existing_id": "kn_old_decision", '
            '"contradiction_note": "production model switched from Ministral to Qwen3.5-4B"}]}'
        )

    out = _decide_actions(cluster, "clu", envs, existing, infer)
    assert len(out) == 1
    assert out[0]["action"] == "contradict"
    assert out[0]["existing_id"] == "kn_old_decision"
    assert out[0]["contradiction_note"] == "production model switched from Ministral to Qwen3.5-4B"
    assert out[0]["category"] == "decision"


def test_decide_actions_malformed_response_fails_closed_to_create():
    """Fail-closed, never dropped: an unparseable response degrades EVERY fact's action to
    'create' (mirrors the monolith's own existing_id-not-found -> create fallback) rather than
    losing the fact. Matches B1's never-silent-drop discipline at the action layer."""
    cluster = [_entry(done=["a", "b"])]
    envs = [
        _envelope_ok("clu:0:done:0", facts=[{"text": "a"}]),
        _envelope_ok("clu:0:done:1", facts=[{"text": "b"}]),
    ]
    out = _decide_actions(cluster, "clu", envs, [], lambda req: "not json at all")
    assert len(out) == 2  # count-invariant: nothing dropped
    assert all(o["action"] == "create" for o in out)
    assert all(o["existing_id"] is None for o in out)


def test_decide_actions_infer_exception_fails_closed_to_create():
    cluster = [_entry(done=["a"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "a"}])]

    def boom(request):
        raise RuntimeError("model backend down")

    out = _decide_actions(cluster, "clu", envs, [], boom)
    assert len(out) == 1
    assert out[0]["action"] == "create"


def test_decide_actions_partial_response_defaults_missing_indices_to_create():
    """The model may address only SOME facts (a batched call is a harder task than B1's
    per-unit calls) — index-based matching means an addressed fact keeps its real action and
    an unaddressed one degrades to create, rather than a global all-or-nothing failure."""
    cluster = [_entry(done=["addressed", "not addressed"])]
    envs = [
        _envelope_ok("clu:0:done:0", facts=[{"text": "addressed"}]),
        _envelope_ok("clu:0:done:1", facts=[{"text": "not addressed"}]),
    ]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "corroborate", "existing_id": "kn_1"}]}'

    out = _decide_actions(cluster, "clu", envs, [_node("x", node_id="kn_1")], infer)
    assert out[0]["action"] == "corroborate" and out[0]["existing_id"] == "kn_1"
    assert out[1]["action"] == "create" and out[1]["existing_id"] is None  # missing index -> create


def test_decide_actions_reordered_response_still_matches_by_index():
    cluster = [_entry(done=["zero", "one"])]
    envs = [
        _envelope_ok("clu:0:done:0", facts=[{"text": "zero"}]),
        _envelope_ok("clu:0:done:1", facts=[{"text": "one"}]),
    ]

    def infer(request):
        # model returns index 1 BEFORE index 0 — matching must not assume response order
        return (
            '{"actions": ['
            '{"index": 1, "action": "corroborate", "existing_id": "kn_one"}, '
            '{"index": 0, "action": "corroborate", "existing_id": "kn_zero"}'
            ']}'
        )

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["existing_id"] == "kn_zero"  # positional output order matches INPUT order
    assert out[1]["existing_id"] == "kn_one"


def test_decide_actions_invalid_action_value_defaults_to_create():
    cluster = [_entry(done=["a"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "a"}])]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "delete_everything", "existing_id": "kn_1"}]}'

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["action"] == "create"  # not in {create,corroborate,contradict} -> fail-closed


def test_decide_actions_count_invariant_across_multiple_facts():
    cluster = [_entry(done=["a", "b", "c"], decisions=["d"])]
    envs = [
        _envelope_ok("clu:0:done:0", facts=[{"text": "a"}]),
        _envelope_ok("clu:0:done:1", facts=[{"text": "b"}]),
        _envelope_ok("clu:0:done:2", facts=[{"text": "c"}]),
        _envelope_ok("clu:0:decisions:0", decisions=[{"text": "d"}]),
    ]
    out = _decide_actions(cluster, "clu", envs, [], lambda req: "{}")
    assert len(out) == 4  # N facts in -> N action-dicts out, always


def test_decide_actions_output_matches_apply_consolidation_result_shape():
    """Explicit contract-shape check: output keys are EXACTLY what
    _apply_consolidation_result's parsed['nodes'] item reads (consolidate.py ~1044-1076)."""
    cluster = [_entry(done=["a"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "a", "category": "fact"}])]
    out = _decide_actions(cluster, "clu", envs, [], lambda req: "{}")
    assert set(out[0].keys()) == {
        "action", "existing_id", "content", "category", "tags", "source_turns", "contradiction_note",
    }


def test_decide_actions_tags_pass_through_when_present():
    cluster = [_entry(done=["a"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "a"}])]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "create", "tags": ["premium", "boundary"]}]}'

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["tags"] == ["premium", "boundary"]


def test_decide_actions_tags_default_empty_when_absent_or_invalid():
    cluster = [_entry(done=["a", "b"])]
    envs = [
        _envelope_ok("clu:0:done:0", facts=[{"text": "a"}]),
        _envelope_ok("clu:0:done:1", facts=[{"text": "b"}]),
    ]

    def infer(request):
        return (
            '{"actions": ['
            '{"index": 0, "action": "create"}, '
            '{"index": 1, "action": "create", "tags": "not-a-list"}'
            ']}'
        )

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["tags"] == []
    assert out[1]["tags"] == []  # a non-list tags value is coerced to empty, not propagated raw


# --- integration with REAL B1 envelopes (not hand-built fixtures) ----------------------------

def test_decide_actions_with_real_b1_envelopes():
    cluster = [_entry(done=["a durable fact from real extract_batch"])]
    real_envs = _real_ok_envelopes(cluster, "clu", _ok_envelope_infer)
    assert len(real_envs) == 1  # confirms the real B1 path produced a usable envelope

    def action_infer(request):
        return '{"actions": [{"index": 0, "action": "create"}]}'

    out = _decide_actions(cluster, "clu", real_envs, [], action_infer)
    assert len(out) == 1
    assert out[0]["action"] == "create"
    assert out[0]["content"].startswith("fact from:")
