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

import json
import re

import pytest

pytest.importorskip("synapt.extract.batch")

from synapt.recall.journal import JournalEntry
from synapt.recall.knowledge import KnowledgeNode
from synapt.recall.consolidate import (
    MIN_RESPONSE_TOKENS,
    _decide_actions,
    _estimate_action_decision_budget,
    _flatten_envelope_facts,
    _map_temporal_refs_to_bounds,
    _normalize_for_dedup,
    _run_coro_blocking,
    _extract_cluster_units,
)


def _entry(session_id="s1", *, done=None, decisions=None, ts="2026-07-13T10:00:00Z") -> JournalEntry:
    return JournalEntry(
        timestamp=ts, session_id=session_id,
        done=list(done or []), decisions=list(decisions or []),
    )


def _node(content, category="fact", node_id=None) -> KnowledgeNode:
    return KnowledgeNode.create(content=content, category=category, node_id=node_id)


def _envelope_ok(source_unit_id: str, *, facts=None, decisions=None, temporal_refs=None):
    """A minimal fake BatchUnitResult-shaped object (status='ok') carrying a SynaptExtraction
    envelope, without needing a real extract_batch round trip — _flatten_envelope_facts /
    _decide_actions read .status/.extraction/.source_unit_id. ``temporal_refs`` is the
    extraction(unit)-level list extract_batch now emits (see consolidate.py's TEMPORAL — ROLE
    note): each ref carries {raw, resolved, resolved_end?, role, version} exactly as pinned from
    real extract_batch output. Defaults to [] so the many non-temporal fixtures below are
    unaffected. _flatten_envelope_facts maps these to per-fact valid_from/valid_until
    DETERMINISTICALLY by role — no LLM re-judgment."""
    from types import SimpleNamespace
    return SimpleNamespace(
        source_unit_id=source_unit_id,
        status="ok",
        extraction={
            "facts": facts or [],
            "decisions": decisions or [],
            "temporal_refs": temporal_refs or [],
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


# --- _map_temporal_refs_to_bounds: the deterministic role -> (valid_from, valid_until) map ----
# Direction now rides in extract's ROLE field (extract#31); recall maps it with ZERO LLM
# re-judgment. Every ref shape below matches real extract_batch output (pinned: {raw, resolved,
# resolved_end?, role, version}). The mapper is pure + fail-safe: unknown/absent role or
# unparseable date contributes nothing (honest None), never a crash.

def _ref(role, resolved=None, *, resolved_end=None, raw="some date"):
    """A temporal_ref shaped exactly like real extract_batch output (incl. the harmless
    ``version`` stamp the mapper must ignore)."""
    r = {"raw": raw, "role": role, "version": "1"}
    if resolved is not None:
        r["resolved"] = resolved
    if resolved_end is not None:
        r["resolved_end"] = resolved_end
    return r


def test_map_effective_role_sets_valid_from_only():
    assert _map_temporal_refs_to_bounds([_ref("effective", "2026-03-01")]) == ("2026-03-01", None)


def test_map_expiry_role_sets_valid_until_only():
    assert _map_temporal_refs_to_bounds([_ref("expiry", "2026-04-30")]) == (None, "2026-04-30")


def test_map_superseded_role_sets_valid_until():
    # superseded == the fact stopped being true when the newer one arrived -> an upper bound.
    assert _map_temporal_refs_to_bounds([_ref("superseded", "2026-05-15")]) == (None, "2026-05-15")


def test_map_range_role_sets_both_bounds():
    got = _map_temporal_refs_to_bounds([_ref("range", "2026-01-01", resolved_end="2026-06-30")])
    assert got == ("2026-01-01", "2026-06-30")


def test_map_point_role_defaults_to_valid_from():
    # a bare point-in-time with no start/end direction -> valid_from by spec default.
    assert _map_temporal_refs_to_bounds([_ref("point", "2026-07-04")]) == ("2026-07-04", None)


def test_map_combines_effective_and_expiry_across_refs():
    # THE multi-ref case: one unit's text yields both an effective and an expiry ref; the fact
    # gets BOTH bounds. _derive_temporal_bounds's combine, now driven by role not LLM output.
    refs = [_ref("effective", "2026-03-01"), _ref("expiry", "2026-04-30")]
    assert _map_temporal_refs_to_bounds(refs) == ("2026-03-01", "2026-04-30")


def test_map_absent_role_contributes_nothing():
    # a ref WITHOUT a role can't encode direction -> honest None (the old placeholder behavior,
    # now scoped to exactly the role-less case rather than being unconditional).
    assert _map_temporal_refs_to_bounds([{"raw": "April 30", "resolved": "2026-04-30", "version": "1"}]) == (None, None)


def test_map_unknown_role_contributes_nothing():
    assert _map_temporal_refs_to_bounds([_ref("whenever", "2026-04-30")]) == (None, None)


def test_map_missing_resolved_yields_none_bound():
    # role present but the model gave no resolved date -> no bound to assign, no crash.
    assert _map_temporal_refs_to_bounds([_ref("expiry", None)]) == (None, None)


def test_map_malformed_resolved_date_rejected():
    # _validate_iso_date guards: a non-ISO 'resolved' does not become a bound.
    assert _map_temporal_refs_to_bounds([_ref("expiry", "not-a-date")]) == (None, None)


def test_map_range_missing_resolved_end_sets_only_start():
    # defensive: validate.py enforces range->resolved_end, but if it's absent the start still maps.
    assert _map_temporal_refs_to_bounds([_ref("range", "2026-01-01")]) == ("2026-01-01", None)


def test_map_first_non_null_wins_per_bound_deterministic():
    # two effective refs -> the FIRST resolved wins (order-preserving, deterministic; no min/max
    # heuristic that could reorder under equal inputs).
    refs = [_ref("effective", "2026-03-01"), _ref("effective", "2026-09-09")]
    assert _map_temporal_refs_to_bounds(refs) == ("2026-03-01", None)


def test_map_empty_list_is_null_bounds():
    assert _map_temporal_refs_to_bounds([]) == (None, None)


def test_map_none_input_is_null_bounds():
    assert _map_temporal_refs_to_bounds(None) == (None, None)


def test_map_non_dict_ref_skipped_fail_safe():
    # a malformed (non-dict) element must not crash the whole cluster's consolidation.
    refs = ["garbage", None, _ref("expiry", "2026-04-30")]
    assert _map_temporal_refs_to_bounds(refs) == (None, "2026-04-30")


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


# --- _flatten_envelope_facts: temporal wiring (envelope role -> per-fact bounds) -------------

def test_flatten_attaches_role_mapped_bounds_to_facts():
    env = _envelope_ok(
        "c:0:done:0",
        facts=[{"text": "the API key expires April 30"}],
        temporal_refs=[_ref("expiry", "2025-04-30")],
    )
    flat = _flatten_envelope_facts([env])
    assert flat[0]["valid_from"] is None
    assert flat[0]["valid_until"] == "2025-04-30"


def test_flatten_without_temporal_refs_gives_null_bounds():
    # backward-compat: the common case (no temporal expression in the unit) stays null — the
    # honest placeholder, now the fallback rather than the unconditional rule.
    env = _envelope_ok("c:0:done:0", facts=[{"text": "no dates here"}])
    flat = _flatten_envelope_facts([env])
    assert flat[0]["valid_from"] is None and flat[0]["valid_until"] is None


def test_flatten_unit_temporal_refs_apply_to_all_facts_of_that_unit():
    # temporal_refs are unit-level (siblings of facts[] in the extraction, confirmed from real
    # extract_batch output); every fact flattened from that unit shares the unit's bounds.
    env = _envelope_ok(
        "c:0:done:0",
        facts=[{"text": "fact one"}, {"text": "fact two"}],
        temporal_refs=[_ref("effective", "2026-03-01")],
    )
    flat = _flatten_envelope_facts([env])
    assert all(f["valid_from"] == "2026-03-01" for f in flat)


def test_flatten_bounds_are_per_envelope_not_bled_across_units():
    # two units, only the first carries a temporal ref -> the second stays null (no cross-unit
    # bleed from the shared out[] accumulator).
    envs = [
        _envelope_ok("c:0:done:0", facts=[{"text": "dated"}], temporal_refs=[_ref("expiry", "2026-04-30")]),
        _envelope_ok("c:0:done:1", facts=[{"text": "undated"}]),
    ]
    flat = _flatten_envelope_facts(envs)
    by_text = {f["text"]: f for f in flat}
    assert by_text["dated"]["valid_until"] == "2026-04-30"
    assert by_text["undated"]["valid_until"] is None


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
    _apply_consolidation_result's parsed['nodes'] item reads (consolidate.py ~1044-1076),
    INCLUDING valid_from/valid_until (added in the review-fix round — reconcile has no
    fallback for valid_until, so it must be supplied when derivable, not omitted)."""
    cluster = [_entry(done=["a"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "a", "category": "fact"}])]
    out = _decide_actions(cluster, "clu", envs, [], lambda req: "{}")
    assert set(out[0].keys()) == {
        "action", "existing_id", "content", "category", "tags", "source_turns",
        "contradiction_note", "valid_from", "valid_until",
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


# =============================================================================================
# REVIEW-FIX REGRESSION SUITE (2026-07-14, Sentinel + Opus REQUEST-CHANGES)
# =============================================================================================

# --- Blocker #1: dense-cluster token-cap truncation → mass fail-closed-to-create -------------

def _truncating_action_infer_factory():
    """A fake infer SIMULATING a token-budget-limited model: builds the full action response
    for every candidate in the prompt, then truncates the "actions" array to whatever fits
    within the REQUESTED max_tokens (~4 chars/token, the same approximation the codebase
    already uses) — reproducing the exact causal mechanism the dense-cluster inflation bug
    depended on. Returns (infer, calls) so a test can inspect the actual requested budget."""
    calls: list[dict] = []

    def infer(request):
        calls.append(request)
        max_tokens = request.get("max_tokens") or MIN_RESPONSE_TOKENS
        budget_chars = max_tokens * 4
        n = len(re.findall(r"^\[(\d+)\]", request["prompt"], re.MULTILINE))
        kept: list[dict] = []
        for i in range(n):
            candidate = kept + [{"index": i, "action": "corroborate", "existing_id": f"kn_{i:04d}"}]
            if len(json.dumps({"actions": candidate})) > budget_chars:
                break
            kept = candidate
        return json.dumps({"actions": kept})

    return infer, calls


def test_truncating_fixture_is_faithful_flat_800_budget_truncates_dense_response():
    """Negative control for the fixture itself: with the OLD flat 800-token budget, a
    64-candidate response DOES truncate — proving the fixture genuinely simulates truncation,
    so the positive test below isn't vacuously passing because nothing ever truncates."""
    n = 64
    infer, _ = _truncating_action_infer_factory()
    request = {
        "prompt": "\n".join(f"[{i}] (fact) durable fact number {i}" for i in range(n)),
        "messages": [], "capabilities": [], "max_tokens": 800,
    }
    parsed = json.loads(infer(request))
    assert len(parsed["actions"]) < n


def test_decide_actions_dense_cluster_all_facts_addressed_with_scaled_budget():
    """THE dense-cluster inflation bug (Opus blocker #2 / Sentinel's original finding): B2's
    per-cluster call inherited B1's flat per-unit 800-token floor, so a response covering many
    candidates truncated partway and every unaddressed index fail-closed to "create" — the
    "inflation fix" silently didn't fix inflation on dense clusters. This proves the FIX
    (_estimate_action_decision_budget scaling with candidate count) threads a big-enough
    budget through, so a 64-candidate cluster's response is NOT truncated."""
    n = 64
    cluster = [_entry(done=[f"durable fact number {i}" for i in range(n)])]
    envs = [
        _envelope_ok(f"clu:0:done:{i}", facts=[{"text": f"durable fact number {i}"}])
        for i in range(n)
    ]
    infer, calls = _truncating_action_infer_factory()

    out = _decide_actions(cluster, "clu", envs, [], infer)

    assert len(out) == n  # count-invariant regardless of truncation
    assert len(calls) == 1
    requested_budget = calls[0]["max_tokens"]
    assert requested_budget > 800, "the scaled budget was NOT requested — still the flat floor"
    addressed = [o for o in out if o["action"] == "corroborate"]
    assert len(addressed) == n, (
        f"only {len(addressed)}/{n} facts were addressed by the model — "
        "the response was still truncated despite the scaled budget"
    )


def test_estimate_action_decision_budget_scales_with_fact_count():
    # A short, identical prompt for both n_facts values would let _estimate_response_budget's
    # OWN context-awareness (near CONTEXT_BUDGET for a tiny prompt) dominate both calls
    # equally, masking whether the fact-count term does anything. The scenario where it
    # actually MATTERS is a prompt heavy with existing-knowledge context — close enough to
    # consuming the assumed context budget that _estimate_response_budget alone floors at
    # MIN_RESPONSE_TOKENS regardless of fact count; the scaled term must grow past that floor.
    heavy_existing = "x" * 28000
    small = _estimate_action_decision_budget(heavy_existing, 1)
    dense = _estimate_action_decision_budget(heavy_existing, 64)
    assert dense > small
    assert dense > MIN_RESPONSE_TOKENS * 3  # a 64-candidate cluster needs substantially more


def test_estimate_action_decision_budget_never_below_the_monolith_context_estimate():
    # a huge prompt (lots of existing knowledge) with FEW facts should still get at least
    # what _estimate_response_budget itself would grant — the scaled-by-facts floor never
    # UNDERCUTS the monolith's own context-aware estimate.
    huge_prompt = "x" * 20000
    budget = _estimate_action_decision_budget(huge_prompt, 1)
    assert budget >= MIN_RESPONSE_TOKENS


# --- Blocker #2: fail-closed-to-create persisted duplicates ----------------------------------

def test_normalize_for_dedup_case_and_whitespace_insensitive():
    a = _normalize_for_dedup("Recall#875   Wired  Extract_Batch")
    b = _normalize_for_dedup("recall#875 wired extract_batch")
    assert a == b
    assert _normalize_for_dedup("  leading and trailing  ") == "leading and trailing"


def test_decide_actions_exact_match_dedup_converts_fail_closed_create_to_corroborate():
    existing = [_node("recall#875 wired extract_batch into consolidation", node_id="kn_dup01")]
    cluster = [_entry(done=["recall#875 wired extract_batch into consolidation"])]
    envs = [_envelope_ok(
        "clu:0:done:0", facts=[{"text": "recall#875 wired extract_batch into consolidation"}]
    )]
    # malformed response -> every fact fail-closes to create, INCLUDING this exact-duplicate one
    out = _decide_actions(cluster, "clu", envs, existing, lambda req: "not json")
    assert out[0]["action"] == "corroborate"
    assert out[0]["existing_id"] == "kn_dup01"


def test_decide_actions_exact_match_dedup_case_and_whitespace_insensitive():
    existing = [_node("Recall#875   Wired Extract_Batch", node_id="kn_dup02")]
    cluster = [_entry(done=["recall#875 wired extract_batch"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "recall#875 wired extract_batch"}])]
    out = _decide_actions(cluster, "clu", envs, existing, lambda req: "not json")
    assert out[0]["action"] == "corroborate"
    assert out[0]["existing_id"] == "kn_dup02"


def test_decide_actions_exact_match_dedup_does_not_fire_on_dissimilar_content():
    """Negative control: a genuinely NEW fact (no exact match) stays create+None — the dedup
    check must not over-match."""
    existing = [_node("recall#875 wired extract_batch into consolidation", node_id="kn_other")]
    cluster = [_entry(done=["a completely different fact about extract_batch tests"])]
    envs = [_envelope_ok(
        "clu:0:done:0", facts=[{"text": "a completely different fact about extract_batch tests"}]
    )]
    out = _decide_actions(cluster, "clu", envs, existing, lambda req: "not json")
    assert out[0]["action"] == "create"
    assert out[0]["existing_id"] is None


def test_decide_actions_exact_match_dedup_applies_to_model_chosen_create_too():
    """The dedup check applies UNIVERSALLY to any action that resolves to create — even when
    the MODEL explicitly (not fail-closed) chose create — since exact-duplicate content is
    never the right create, regardless of WHY the action ended up "create"."""
    existing = [_node("recall#875 wired extract_batch into consolidation", node_id="kn_explicit")]
    cluster = [_entry(done=["recall#875 wired extract_batch into consolidation"])]
    envs = [_envelope_ok(
        "clu:0:done:0", facts=[{"text": "recall#875 wired extract_batch into consolidation"}]
    )]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "create", "existing_id": null}]}'

    out = _decide_actions(cluster, "clu", envs, existing, infer)
    assert out[0]["action"] == "corroborate"
    assert out[0]["existing_id"] == "kn_explicit"


def test_decide_actions_exact_match_dedup_does_not_override_model_chosen_corroborate():
    """The dedup check only touches action=="create" — a model-chosen corroborate against a
    DIFFERENT existing node is never silently redirected to the exact-match one."""
    existing = [
        _node("recall#875 wired extract_batch into consolidation", node_id="kn_exact"),
        _node("a totally different node", node_id="kn_target"),
    ]
    cluster = [_entry(done=["recall#875 wired extract_batch into consolidation"])]
    envs = [_envelope_ok(
        "clu:0:done:0", facts=[{"text": "recall#875 wired extract_batch into consolidation"}]
    )]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "corroborate", "existing_id": "kn_target"}]}'

    out = _decide_actions(cluster, "clu", envs, existing, infer)
    assert out[0]["action"] == "corroborate"
    assert out[0]["existing_id"] == "kn_target"  # the model's own choice, NOT overridden


# --- TEMPORAL — ROLE MAPPING (2026-07-15, extract#31 landed the role field) ------------------
#
# The temporal loop is now closed the RIGHT way (config/design/extract-temporal-role-2026-07-14.
# md): extraction reads direction ONCE and emits a validity ROLE; recall maps role -> valid_from/
# valid_until DETERMINISTICALLY in _flatten_envelope_facts (see the _map_temporal_refs_to_bounds
# tests above). Two invariants survive from the held-placeholder era and are STRONGER now, not
# obsolete: (1) the action-decision LLM never supplies temporal — bounds come only from the
# envelope's role, so a model that emits valid_from/valid_until in its ACTION response is still
# ignored; (2) the action-decision PROMPT carries no temporal instructions. What changed: a fact
# whose unit carried a role-bearing temporal_ref now gets a REAL bound instead of unconditional
# null. Null is now the honest FALLBACK (no role / no resolved date), not the rule.

def test_decide_actions_maps_envelope_role_to_node_bounds():
    """The positive case: a genuinely new fact whose unit carried an expiry role lands on the
    node dict with valid_until set — deterministically, from the envelope, no LLM judgment."""
    cluster = [_entry(done=["the API key expires April 30"])]
    envs = [_envelope_ok(
        "clu:0:done:0",
        facts=[{"text": "the API key expires April 30"}],
        temporal_refs=[_ref("expiry", "2025-04-30")],
    )]
    out = _decide_actions(cluster, "clu", envs, [], lambda req: '{"actions": [{"index": 0, "action": "create"}]}')
    assert out[0]["valid_from"] is None
    assert out[0]["valid_until"] == "2025-04-30"


def test_decide_actions_ignores_action_response_temporal_values():
    """Bounds come ONLY from the envelope's role, never from the action-decision response. Even
    if the model emits valid_from/valid_until in its ACTION json (residual habit, or a prompt
    regression re-adding the instruction), _decide_actions must not read them — the envelope
    here has no temporal_refs, so the bounds stay null despite the model's attempt. Regression
    guard against a second temporal mechanism creeping back into the action pass."""
    cluster = [_entry(done=["we migrated to PostgreSQL in March 2026"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "we migrated to PostgreSQL in March 2026"}])]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "create", "valid_from": "2026-03-01", "valid_until": "2026-04-30"}]}'

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["valid_from"] is None  # NOT the model's "2026-03-01" — action response ignored
    assert out[0]["valid_until"] is None  # NOT the model's "2026-04-30"


def test_decide_actions_bounds_orthogonal_to_action():
    """A role-mapped bound rides through regardless of the resolved action (create/corroborate/
    contradict) — temporal is decided at extraction, independent of the action decision."""
    existing = [_node("some existing fact", node_id="kn_1")]
    cluster = [_entry(done=["a"], decisions=["reversed: b"])]
    envs = [
        _envelope_ok("clu:0:done:0", facts=[{"text": "a"}], temporal_refs=[_ref("effective", "2026-01-01")]),
        _envelope_ok("clu:0:decisions:0", decisions=[{"text": "reversed: b"}], temporal_refs=[_ref("expiry", "2026-02-02")]),
    ]

    def infer(request):
        return (
            '{"actions": ['
            '{"index": 0, "action": "corroborate", "existing_id": "kn_1"}, '
            '{"index": 1, "action": "contradict", "existing_id": "kn_1", "contradiction_note": "x"}'
            ']}'
        )

    out = _decide_actions(cluster, "clu", envs, existing, infer)
    assert out[0]["action"] == "corroborate" and out[0]["valid_from"] == "2026-01-01"
    assert out[1]["action"] == "contradict" and out[1]["valid_until"] == "2026-02-02"


def test_decide_actions_bound_survives_fail_closed_action():
    """DECOUPLING proof: a garbage action response fail-closes the ACTION to create, but the
    envelope-role bound still lands — because temporal comes from the envelope, not the action
    LLM. The fail-closed path fabricates nothing AND loses nothing that extraction already read."""
    cluster = [_entry(done=["a", "b"])]
    envs = [
        _envelope_ok("clu:0:done:0", facts=[{"text": "a"}], temporal_refs=[_ref("expiry", "2026-04-30")]),
        _envelope_ok("clu:0:done:1", facts=[{"text": "b"}]),  # no ref -> honest null fallback
    ]
    out = _decide_actions(cluster, "clu", envs, [], lambda req: "not json")
    assert all(o["action"] == "create" for o in out)  # action fail-closed
    assert out[0]["valid_until"] == "2026-04-30"       # ...but the bound survived
    assert out[1]["valid_until"] is None               # ...and the ref-less fact stays null


def test_build_action_decision_prompt_does_not_leak_temporal():
    """The action-decision prompt carries no temporal — decided at extraction (role), never
    re-judged by the action pass. STRONG guard: the facts fed here carry real valid_from/
    valid_until exactly as production facts now do (mapped by _flatten_envelope_facts), and
    NEITHER the keys NOR their values may reach the prompt — leaking them would re-expose
    temporal to the action LLM, the coupling the deterministic role-map exists to remove."""
    from synapt.recall.consolidate import _build_action_decision_prompt
    facts = [{"text": "a fact", "category": "fact", "valid_from": "2026-03-01", "valid_until": "2026-04-30"}]
    prompt = _build_action_decision_prompt(facts, [], [_entry(done=["a fact"])])
    assert "valid_from" not in prompt and "valid_until" not in prompt
    assert "2026-03-01" not in prompt and "2026-04-30" not in prompt


# --- CAPSTONE: Sentinel's wrong-year scenario, end-to-end through recall's consumption --------

def test_capstone_wrong_year_scenario_end_to_end_through_recall_consumption():
    """Sentinel's exact real-path scenario, end-to-end through recall's WHOLE consumption path:
    a 2025-sourced 'the API key expires April 30' becomes a node with valid_until='2025-04-30'
    (NOT 2026). Exercises the REAL extract_batch — whose coercion preserves role at base tier,
    the exact behavior extract#31 fixed — via _real_ok_envelopes (default _EXTRACT_CAPABILITIES,
    so this also proves recall actually REQUESTS temporal_refs now), then the deterministic
    mapper + decide_actions node build.

    Policy-compliant: the model output is a FAKE infer seam (never a live/Fable call); resolved
    is pinned to 2025-04-30 as what the source-date-anchored model WOULD emit. The model's actual
    year-resolution against the threaded source date is extract#31's contract, proven there — here
    we prove recall (a) requests the role, (b) preserves it through real coercion, (c) maps it
    deterministically to the node's valid_until."""
    cluster = [_entry(done=["the API key expires April 30"], ts="2025-03-01T09:00:00Z")]

    def infer(request):
        # a role-bearing Stage-1 extraction, resolved to the SOURCE year (2025), as a source-date-
        # anchored model would produce. The unit text + source date are in request["prompt"].
        return json.dumps({
            "extracted_at": "2025-03-01T09:00:00Z",
            "facts": [{"text": "the API key expires April 30", "category": "fact"}],
            "decisions": [],
            "temporal_refs": [{"raw": "April 30", "resolved": "2025-04-30", "role": "expiry"}],
        })

    ok = _real_ok_envelopes(cluster, "clu", infer)
    assert len(ok) == 1
    # real extract_batch preserved role at base tier (extract#31 in recall's default cap set):
    assert ok[0].extraction["temporal_refs"][0]["role"] == "expiry"
    assert ok[0].extraction["temporal_refs"][0]["resolved"] == "2025-04-30"

    # ...and recall maps it deterministically onto the reconcile-ready node dict:
    out = _decide_actions(cluster, "clu", ok, [], lambda req: '{"actions": [{"index": 0, "action": "create"}]}')
    assert out[0]["valid_until"] == "2025-04-30"  # the anchored year, NOT 2026
    assert out[0]["valid_from"] is None
