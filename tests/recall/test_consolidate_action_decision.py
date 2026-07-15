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
    _normalize_for_dedup,
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
    _decide_actions only read .status/.extraction/.source_unit_id. No temporal_refs field —
    B2 no longer requests or reads it (see consolidate.py's TEMPORAL — REVISED note); temporal
    bounds are judged by B2's own LLM pass directly from fact content."""
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


# --- Blocker #3 REVISED: temporal_refs-derivation ABANDONED, content-based LLM judgment ------
#
# Opus withdrew his approve on the ORIGINAL temporal_refs-derivation fix after fruit-checking a
# REAL extract_batch call (not a hand-built envelope): resolved_end/type/context are stripped
# by the base "temporal_refs" capability's schema (VERIFIED — a separate "temporal_classes"
# capability is required to unlock them, which was never requested), so resolved_end could
# NEVER arrive. WORSE, even a "point"-type ref's bare "resolved" is directionally AMBIGUOUS (no
# field says start-vs-expiry) — a semantic judgment only reading the fact's own sentence can
# make. ABANDONED that mechanism entirely; B2's OWN LLM pass now judges valid_from/valid_until
# per fact directly from content, reusing the monolith's proven instruction language. These
# tests prove the WIRING (does the parsed action item's valid_from/valid_until thread through
# correctly) — NOT whether a real model judges the RIGHT direction for a given sentence, which
# is a model-quality question the Phase-C dogfood measures, not a fake-infer unit test.

def test_build_action_decision_prompt_includes_temporal_instructions():
    """Sanity check that the prompt actually carries the new temporal instructions — not a
    full wording pin (too brittle), just confirms the instruction text made it into the
    rendered prompt."""
    from synapt.recall.consolidate import _build_action_decision_prompt
    facts = [{"text": "a fact", "category": "fact"}]
    prompt = _build_action_decision_prompt(facts, [], [_entry(done=["a fact"])])
    assert "valid_from" in prompt and "valid_until" in prompt
    assert "expires" in prompt.lower()  # the monolith's own expiry example, reused


def test_decide_actions_llm_supplies_valid_from():
    cluster = [_entry(done=["we migrated to PostgreSQL in March 2026"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "we migrated to PostgreSQL in March 2026"}])]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "create", "valid_from": "2026-03-01", "valid_until": null}]}'

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["valid_from"] == "2026-03-01"
    assert out[0]["valid_until"] is None


def test_decide_actions_llm_supplies_valid_until_for_an_expiry():
    """THE semantic-direction regression guard: an EXPIRY fact must thread as valid_until, not
    valid_from — the exact reversal risk Opus's withdrawal flagged. The wiring here just
    trusts whichever field the model populates; this confirms it threads the CORRECT field
    (valid_until) without accidentally cross-wiring it to valid_from."""
    cluster = [_entry(done=["the API key expires April 30"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "the API key expires April 30"}])]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "create", "valid_from": null, "valid_until": "2026-04-30"}]}'

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["valid_from"] is None
    assert out[0]["valid_until"] == "2026-04-30"  # NOT valid_from — direction preserved


def test_decide_actions_llm_supplies_both_bounds():
    cluster = [_entry(done=["the migration window ran march to april"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "the migration window ran march to april"}])]

    def infer(request):
        return (
            '{"actions": [{"index": 0, "action": "create", '
            '"valid_from": "2026-03-01", "valid_until": "2026-04-30"}]}'
        )

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["valid_from"] == "2026-03-01"
    assert out[0]["valid_until"] == "2026-04-30"


def test_decide_actions_no_temporal_signal_yields_none_bounds():
    """Negative control: most facts have no clear temporal boundary — the model (and reconcile
    downstream) treats null as correct, not a gap. Matches "null is better than guessing"."""
    cluster = [_entry(done=["a fact with no temporal signal"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "a fact with no temporal signal"}])]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "create", "valid_from": null, "valid_until": null}]}'

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["valid_from"] is None
    assert out[0]["valid_until"] is None


def test_decide_actions_temporal_fields_absent_from_response_yield_none():
    """The model may omit valid_from/valid_until entirely (not even null keys) — must not
    crash, defaults to None same as an explicit null."""
    cluster = [_entry(done=["a fact"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "a fact"}])]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "create"}]}'  # no temporal keys at all

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["valid_from"] is None
    assert out[0]["valid_until"] is None


def test_decide_actions_invalid_temporal_type_coerced_to_none():
    """A non-string valid_from/valid_until (e.g. a number or nested object) is coerced to
    None, not propagated raw — reconcile's _validate_iso_date would reject a malformed STRING
    gracefully, but a non-string should never reach it in the first place."""
    cluster = [_entry(done=["a fact"])]
    envs = [_envelope_ok("clu:0:done:0", facts=[{"text": "a fact"}])]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "create", "valid_from": 20260301, "valid_until": {}}]}'

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["valid_from"] is None
    assert out[0]["valid_until"] is None


def test_decide_actions_temporal_bounds_are_per_fact_not_shared():
    """Two facts in the SAME response, only one carrying a temporal signal — confirms bounds
    are genuinely PER-FACT (the structural fix for Sentinel's unit-level-bleed hazard: there
    is no unit-level derivation left to bleed from, since each fact gets its OWN judgment)."""
    cluster = [_entry(done=["a fact with a date", "a fact with no date"])]
    envs = [
        _envelope_ok("clu:0:done:0", facts=[{"text": "a fact with a date"}]),
        _envelope_ok("clu:0:done:1", facts=[{"text": "a fact with no date"}]),
    ]

    def infer(request):
        return (
            '{"actions": ['
            '{"index": 0, "action": "create", "valid_from": "2026-03-01", "valid_until": null}, '
            '{"index": 1, "action": "create", "valid_from": null, "valid_until": null}'
            ']}'
        )

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["valid_from"] == "2026-03-01"
    assert out[1]["valid_from"] is None  # NOT bled from index 0


def test_decide_actions_unaddressed_index_yields_none_temporal_not_fabricated():
    """An unaddressed index (fail-closed to create) must not fabricate a temporal bound — null
    is correct here, matching the monolith's own "null is better than guessing" instruction."""
    cluster = [_entry(done=["addressed", "not addressed"])]
    envs = [
        _envelope_ok("clu:0:done:0", facts=[{"text": "addressed"}]),
        _envelope_ok("clu:0:done:1", facts=[{"text": "not addressed"}]),
    ]

    def infer(request):
        return '{"actions": [{"index": 0, "action": "create", "valid_from": "2026-03-01"}]}'

    out = _decide_actions(cluster, "clu", envs, [], infer)
    assert out[0]["valid_from"] == "2026-03-01"
    assert out[1]["action"] == "create"  # fail-closed, per the existing discipline
    assert out[1]["valid_from"] is None and out[1]["valid_until"] is None  # not fabricated


def test_decide_actions_malformed_response_yields_none_temporal_for_every_fact():
    """An unparseable response fail-closes every fact's action to create AND leaves temporal
    bounds None for all of them — no fabrication anywhere in the fail-closed path."""
    cluster = [_entry(done=["a", "b"])]
    envs = [
        _envelope_ok("clu:0:done:0", facts=[{"text": "a"}]),
        _envelope_ok("clu:0:done:1", facts=[{"text": "b"}]),
    ]
    out = _decide_actions(cluster, "clu", envs, [], lambda req: "not json")
    assert all(o["action"] == "create" for o in out)
    assert all(o["valid_from"] is None and o["valid_until"] is None for o in out)
