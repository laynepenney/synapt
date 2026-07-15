"""B1 gate — the decomposed extract path front-half wired into consolidation.

Covers `_extract_cluster_units` (prefilter -> BatchUnits -> extract_batch -> envelopes) and
its scaffold helpers (the async->sync bridge, the pluggable infer seam, per-unit failure
logging). The unit under test injects extract's `infer` seam, so these run with a FAKE sync
infer — zero model dependency, exactly the point of the injected seam.

Contract anchors (verified against the landed extract_batch @ extract sprint-39):
  - COUNT-INVARIANT: one BatchUnitResult per prefilter Candidate; failed units are marked
    (status="failed"), never dropped.
  - ID SCHEME: source_unit_id == batch_unit_id(cluster_id, cand) == "{cluster_id}:{entry_index}:{field}:{index}".
  - produced_by MUST be a provider URI ("recall://consolidate"); the dotted form fails
    validate_extraction on every unit (schema_invalid) — pinned here as a regression guard.

B1 stops at envelopes: the envelope->node adapter, the action-decision pass, and the reconcile
feed are B2/B3, so nothing here creates knowledge nodes.
"""

from __future__ import annotations

import asyncio
import json

import pytest

# The decomposed path depends on extract_batch, which ships in extract sprint-39 (unpublished);
# the published synapt-extract (v0.5.0) lacks it. Locally we run against a LOCAL editable install
# (per the design note); in CI, skip cleanly until extract publishes with batch.
pytest.importorskip("synapt.extract.batch")

from synapt.recall.journal import JournalEntry
from synapt.recall.identify import identify, batch_unit_id
from synapt.recall.knowledge import KnowledgeNode, read_nodes
from synapt.recall.consolidate import (
    _EXTRACT_PRODUCED_BY,
    _extract_cluster_units,
    _make_recall_infer,
    _run_coro_blocking,
    _run_extract_path,
)

_EXTRACTED_AT = "2026-07-13T10:00:00Z"


def _entry(session_id="s1", *, done=None, decisions=None, focus="", next_steps=None,
           ts="2026-07-13T10:00:00Z") -> JournalEntry:
    return JournalEntry(
        timestamp=ts,
        session_id=session_id,
        focus=focus,
        done=list(done or []),
        decisions=list(decisions or []),
        next_steps=list(next_steps or []),
    )


def _ok_envelope(text: str) -> str:
    """A minimal stage-1 completion that finalizes VALID (facts non-empty, extracted_at present)."""
    return json.dumps({
        "extracted_at": _EXTRACTED_AT,
        "facts": [{"text": f"fact from: {text}"}],
        "decisions": [],
        "temporal_refs": [],
    })


def _ok_infer(request):
    return _ok_envelope(request["prompt"])


def _garbage_infer(request):
    return "this is not JSON at all"


def _mixed_infer_for(bad_markers):
    """A fake infer that returns garbage when the unit text contains any marker, else a valid envelope.
    Mirrors the spec test's technique: the unit text rides verbatim into request['prompt']."""
    def infer(request):
        prompt = request["prompt"]
        if any(marker in prompt for marker in bad_markers):
            return "not json"
        return _ok_envelope(prompt)
    return infer


# --- COUNT-INVARIANCE ----------------------------------------------------------------------

# --- source-date resolution anchor (BatchUnit.date, recall's half of the wrong-year fix) -----
# A relative date ("April 30") only resolves to the right YEAR against the fact's source date.
# recall threads each candidate's OWN journal-entry timestamp into BatchUnit.date, which
# extract_batch renders into the Stage-1 prompt ("Resolve relative dates using: <date>."). This
# is the recall-side seam of the fix Sentinel's real-path finding opened (a 2025-sourced "expires
# April 30" resolving to 2026 under the old, anchor-less path). We assert the date REACHES the
# prompt (the deterministic seam recall owns); whether the model then resolves correctly is
# extract#31's contract, proven there.

def _capturing_infer():
    """A fake infer that records every request prompt, then returns a valid envelope."""
    seen = []
    def infer(request):
        seen.append(request["prompt"])
        return _ok_envelope(request["prompt"])
    return infer, seen


def test_source_date_threaded_into_extraction_prompt():
    cluster = [_entry(done=["the API key expires April 30"], ts="2025-03-01T09:00:00Z")]
    infer, seen = _capturing_infer()
    _run_coro_blocking(_extract_cluster_units(cluster, "clu", infer))
    assert len(seen) == 1
    assert "2025-03-01" in seen[0]  # the source date rode into the Stage-1 prompt


def test_per_candidate_source_date_from_its_own_entry():
    # two entries with DIFFERENT dates -> each candidate's prompt carries ITS entry's date, not a
    # single cluster-wide date (entry_index maps each candidate back to its own JournalEntry).
    cluster = [
        _entry(session_id="s1", done=["first thing"], ts="2025-01-01T00:00:00Z"),
        _entry(session_id="s2", done=["second thing"], ts="2026-12-31T00:00:00Z"),
    ]
    infer, seen = _capturing_infer()
    _run_coro_blocking(_extract_cluster_units(cluster, "clu", infer))
    first = next(p for p in seen if "first thing" in p)
    second = next(p for p in seen if "second thing" in p)
    assert "2025-01-01" in first and "2026-12-31" not in first
    assert "2026-12-31" in second and "2025-01-01" not in second


def test_missing_source_timestamp_does_not_crash_and_omits_anchor():
    # fail-safe: an entry with no timestamp -> unit still built (count-invariant), the prompt
    # simply carries no resolution anchor (no crash, no fabricated date).
    cluster = [_entry(done=["undated fact"], ts="")]
    infer, seen = _capturing_infer()
    results = _run_coro_blocking(_extract_cluster_units(cluster, "clu", infer))
    assert len(results) == 1  # never dropped
    assert "Resolve relative dates using:" not in seen[0]  # no anchor line, no "None" literal


def test_count_invariant_one_result_per_candidate():
    cluster = [_entry(done=["a", "b", "c"], decisions=["d", "e"])]
    n_candidates = len(identify(cluster))
    assert n_candidates == 5  # 3 done + 2 decisions; focus/next never read
    results = _run_coro_blocking(_extract_cluster_units(cluster, "clu", _ok_infer))
    assert len(results) == n_candidates == 5


def test_count_invariant_holds_under_total_failure():
    cluster = [_entry(done=["a", "b"], decisions=["c"])]
    results = _run_coro_blocking(_extract_cluster_units(cluster, "clu", _garbage_infer))
    assert len(results) == 3  # nothing dropped even when every unit fails


# --- ID SCHEME -----------------------------------------------------------------------------

def test_result_ids_follow_cluster_namespaced_scheme():
    cluster = [_entry(done=["x"], decisions=["y"]), _entry(session_id="s2", done=["z"])]
    cluster_id = "cluster-42"
    results = _run_coro_blocking(_extract_cluster_units(cluster, cluster_id, _ok_infer))
    got = {r.source_unit_id for r in results}
    expected = {batch_unit_id(cluster_id, c) for c in identify(cluster)}
    assert got == expected
    # every id is cluster-namespaced and of the exact 4-part form
    for sid in got:
        assert sid.startswith(f"{cluster_id}:")
        assert len(sid.split(":")) == 4


def test_ids_are_unique_no_duplicate_id_valueerror():
    # same TEXT in two different fields must still produce distinct ids (field differs) — else
    # extract_batch raises ValueError on duplicate ids.
    cluster = [_entry(done=["identical text"], decisions=["identical text"])]
    results = _run_coro_blocking(_extract_cluster_units(cluster, "clu", _ok_infer))
    ids = [r.source_unit_id for r in results]
    assert len(ids) == len(set(ids)) == 2


# --- FAILED MARKERS (never dropped) --------------------------------------------------------

def test_failed_units_are_marked_not_dropped():
    cluster = [_entry(done=["a", "b"], decisions=["c"])]
    results = _run_coro_blocking(_extract_cluster_units(cluster, "clu", _garbage_infer))
    assert all(r.status == "failed" for r in results)
    assert all(r.reason == "unparseable" for r in results)
    assert all(r.extraction is None for r in results)


def test_mixed_ok_and_failed_per_unit():
    cluster = [_entry(done=["good one", "bad one"], decisions=["good two"])]
    results = _run_coro_blocking(
        _extract_cluster_units(cluster, "clu", _mixed_infer_for(["bad one"]))
    )
    by_status = {}
    for r in results:
        by_status.setdefault(r.status, []).append(r)
    assert len(by_status["ok"]) == 2
    assert len(by_status["failed"]) == 1
    assert by_status["failed"][0].reason == "unparseable"


# --- OK ENVELOPES + produced_by URI regression ---------------------------------------------

def test_ok_units_carry_finalized_extraction():
    cluster = [_entry(done=["ship it"])]
    results = _run_coro_blocking(_extract_cluster_units(cluster, "clu", _ok_infer))
    assert len(results) == 1
    env = results[0]
    assert env.status == "ok"
    assert isinstance(env.extraction, dict)
    assert env.extraction["facts"][0]["text"].startswith("fact from:")
    assert env.extraction["produced_by"] == _EXTRACT_PRODUCED_BY


def test_produced_by_is_a_provider_uri_not_the_dotted_form():
    # regression guard for the contract correction: the dotted form fails validate_extraction on
    # EVERY unit (schema_invalid). The default MUST be a scheme://identifier URI.
    assert "://" in _EXTRACT_PRODUCED_BY
    assert _EXTRACT_PRODUCED_BY == "recall://consolidate"
    cluster = [_entry(done=["a"])]
    ok = _run_coro_blocking(_extract_cluster_units(cluster, "clu", _ok_infer))
    assert ok[0].status == "ok"  # proves the default URI validates
    bad = _run_coro_blocking(
        _extract_cluster_units(cluster, "clu", _ok_infer, produced_by="recall.consolidate")
    )
    assert bad[0].status == "failed" and bad[0].reason == "schema_invalid"


# --- EMPTY CLUSTER -------------------------------------------------------------------------

def test_empty_cluster_yields_no_units():
    # only focus/next_steps → prefilter reads neither → no candidates → no extract call
    cluster = [_entry(focus="just a focus line", next_steps=["do a thing"])]
    results = _run_coro_blocking(_extract_cluster_units(cluster, "clu", _ok_infer))
    assert results == []


# --- ASYNC->SYNC BRIDGE (the CURRENT MCP running-loop path, not a future defense) -----------

def test_run_coro_blocking_from_sync_context():
    async def coro():
        return 7
    assert _run_coro_blocking(coro()) == 7


def test_run_coro_blocking_from_within_running_loop():
    # this IS the current MCP call-site's shape: FastMCP 1.27 runs the sync recall_consolidate
    # tool inline on the event-loop thread, so a loop is already running here — asyncio.run
    # would RuntimeError, and the guard offloads to a fresh thread with its own loop.
    async def outer():
        async def inner():
            return 99
        return _run_coro_blocking(inner())
    assert asyncio.run(outer()) == 99


def test_run_coro_blocking_propagates_exceptions_from_thread_path():
    async def outer():
        async def boom():
            raise ValueError("kaboom")
        return _run_coro_blocking(boom())
    with pytest.raises(ValueError, match="kaboom"):
        asyncio.run(outer())


# --- PLUGGABLE INFER SEAM ------------------------------------------------------------------

class _FakeClient:
    def __init__(self, completion="{}"):
        self.completion = completion
        self.calls = []

    def chat(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return self.completion


def test_make_recall_infer_threads_model_and_maps_messages():
    client = _FakeClient(completion="MODEL-OUT")
    infer = _make_recall_infer(client, "some-swappable-model")
    out = infer({"prompt": "P", "messages": [{"role": "user", "content": "hello"}], "capabilities": []})
    assert out == "MODEL-OUT"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "some-swappable-model"          # pluggable seam: model threaded through
    assert [m.content for m in call["messages"]] == ["hello"]


def test_make_recall_infer_falls_back_to_prompt_when_no_messages():
    client = _FakeClient()
    infer = _make_recall_infer(client, "m")
    infer({"prompt": "only-a-prompt", "messages": [], "capabilities": []})
    assert [m.content for m in client.calls[0]["messages"]] == ["only-a-prompt"]


def _run_extract(cluster, cluster_id, client, model, failures_path, tmp_path,
                  existing_nodes=None, **kwargs):
    """Test helper: call _run_extract_path with a fresh knowledge.jsonl path (matching the
    tests/recall/test_consolidate.py convention of Path(tmpdir) / "knowledge.jsonl")."""
    kn_path = tmp_path / "knowledge.jsonl"
    return _run_extract_path(
        cluster, cluster_id, client, model, failures_path,
        existing_nodes if existing_nodes is not None else [], kn_path, **kwargs,
    )


class _RoutingFakeClient:
    """A fake client returning DIFFERENT completions depending on which prompt it receives —
    needed once B1's extract call and B2's action-decision call share the same infer seam but
    need distinct canned responses. Routes on ACTION_DECISION_PROMPT's distinctive marker text
    ("New Facts (indexed)"), which never appears in extract's Stage-1 prompt."""

    def __init__(self, *, extract_completion, action_completion="{}"):
        self.extract_completion = extract_completion
        self.action_completion = action_completion
        self.calls = []

    def chat(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        content = messages[0].content if messages else ""
        if "New Facts (indexed)" in content:
            return self.action_completion
        return self.extract_completion


# --- B1 FLAG-BRANCH: envelope extraction + failure logging (feeds B2/B3 below) --------------

def test_run_extract_path_logs_failed_markers(tmp_path):
    cluster = [_entry(done=["a", "b"], decisions=["c"])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _FakeClient(completion="garbage-not-json")
    result = _run_extract(cluster, "clu", client, "m", failures_path, tmp_path)
    assert result is not None  # a fully-failed extract batch is still a PROCESSED cluster
    assert result.nodes_created == 0  # nothing to create — every unit failed extraction
    # every failed unit logged (never silent-dropped), one line each — WITH status, the
    # never-silent contract (Sentinel blocker 1: status was previously omitted).
    records = [json.loads(raw) for raw in failures_path.read_text().splitlines() if raw.strip()]
    assert len(records) == 3
    assert {rec["source_unit_id"] for rec in records} == {
        batch_unit_id("clu", c) for c in identify(cluster)
    }
    assert all(
        rec["status"] == "failed" and rec["reason"] == "unparseable" and rec["path"] == "extract_batch"
        for rec in records
    )


def test_run_extract_path_ok_units_produce_no_failure_log(tmp_path):
    cluster = [_entry(done=["a"])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    # extract succeeds; the action-decision call gets "{}" (no "actions" key) -> fail-closed to
    # create, which still exercises the full B1->B2->B3 path without asserting on node identity.
    client = _RoutingFakeClient(extract_completion=_ok_envelope("a"))
    result = _run_extract(cluster, "clu", client, "m", failures_path, tmp_path)
    assert result is not None
    assert not failures_path.exists() or failures_path.read_text().strip() == ""


# --- SILENT-DROP REGRESSION (Sentinel blocker 2): a marker-write failure must be VISIBLE ----

def test_log_extract_failure_returns_false_when_write_fails(tmp_path, monkeypatch):
    """If persisting the failure marker itself raises OSError, that must be reported (False),
    never swallowed as success — a swallowed write-failure is a SILENTLY lost failed unit."""
    from synapt.recall import consolidate as consolidate_mod

    cluster = [_entry(done=["a"])]
    envelope = _run_coro_blocking(
        _extract_cluster_units(cluster, "clu", _garbage_infer)
    )[0]
    assert envelope.status == "failed"

    def _boom_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(consolidate_mod, "open", _boom_open, raising=False)
    persisted = consolidate_mod._log_extract_failure(
        tmp_path / "consolidation_failures.jsonl", "clu", envelope
    )
    assert persisted is False  # NOT swallowed — the caller must see the loss


def test_run_extract_path_returns_none_when_a_marker_write_fails(tmp_path, monkeypatch):
    """THE silent-drop Sentinel caught: previously, an OSError in _log_extract_failure was
    swallowed and _run_extract_path still returned True, declaring the cluster processed while
    a failed unit vanished with no record. Must now propagate as a non-successful cluster
    (None) — never a ConsolidationResult that LOOKS like a clean, if empty, success."""
    from synapt.recall import consolidate as consolidate_mod

    cluster = [_entry(done=["a", "b"])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _FakeClient(completion="garbage-not-json")  # every unit fails -> every unit logs

    real_log = consolidate_mod._log_extract_failure
    calls = {"n": 0}

    def _flaky_log(path, cluster_id, envelope):
        calls["n"] += 1
        if calls["n"] == 1:
            return False  # simulate the first marker write failing
        return real_log(path, cluster_id, envelope)

    monkeypatch.setattr(consolidate_mod, "_log_extract_failure", _flaky_log)
    result = _run_extract(cluster, "clu", client, "m", failures_path, tmp_path)
    assert result is None  # the lost marker must NOT be reported as a clean success


# --- B3: the FULL B1->B2->B3 pipeline via _run_extract_path (reconcile, real nodes) ----------

def test_run_extract_path_creates_a_node_when_action_is_create(tmp_path):
    # fact text needs a specificity signal (snake_case "extract_batch") to clear
    # _apply_consolidation_result's EXISTING low-specificity filter — the same filter the
    # monolith path already contends with; this is not new B2/B3 behavior.
    fact = "recall#875 wired extract_batch into consolidation"
    cluster = [_entry(session_id="s1", done=[fact])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _RoutingFakeClient(
        extract_completion=_ok_envelope(fact),
        action_completion='{"actions": [{"index": 0, "action": "create"}]}',
    )
    kn_path = tmp_path / "knowledge.jsonl"
    result = _run_extract_path(cluster, "clu", client, "m", failures_path, [], kn_path)
    assert result is not None
    assert result.nodes_created == 1
    assert result.nodes_corroborated == 0
    persisted = read_nodes(kn_path)
    assert len(persisted) == 1
    assert persisted[0].source_turns == []  # journal-field attribution isn't turn-shaped (design note)
    assert persisted[0].source_sessions == ["s1"]  # cluster provenance still flows through


def test_run_extract_path_corroborates_against_existing_node(tmp_path):
    kn_path = tmp_path / "knowledge.jsonl"
    existing_node = KnowledgeNode.create(
        content="recall#875 wired extract_batch", category="fact", node_id="kn_abc123",
    )
    from synapt.recall.knowledge import append_node
    append_node(existing_node, kn_path)  # must be ON DISK — update_node reads/writes the file
    existing = [existing_node]

    fact = "recall#875 wired extract_batch into consolidation"
    cluster = [_entry(session_id="s2", done=[fact])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _RoutingFakeClient(
        extract_completion=_ok_envelope(fact),
        action_completion='{"actions": [{"index": 0, "action": "corroborate", "existing_id": "kn_abc123"}]}',
    )
    result = _run_extract_path(cluster, "clu", client, "m", failures_path, existing, kn_path)
    assert result is not None
    assert result.nodes_corroborated == 1
    assert result.nodes_created == 0  # corroborate must NOT create a duplicate node
    persisted = {n.id: n for n in read_nodes(kn_path)}
    assert len(persisted) == 1  # still just the ONE original node, updated in place
    assert "s2" in persisted["kn_abc123"].source_sessions  # the new session was added


def test_run_extract_path_contradicts_and_auto_applies_without_db(tmp_path):
    """No RecallDB passed -> _apply_consolidation_result's legacy auto-apply contradiction path
    (queues nothing, marks the old node contradicted, creates the reversing node directly)."""
    kn_path = tmp_path / "knowledge.jsonl"
    existing_node = KnowledgeNode.create(
        content="extract_batch: production model is Ministral-3B", category="decision", node_id="kn_old",
    )
    from synapt.recall.knowledge import append_node
    append_node(existing_node, kn_path)  # must be ON DISK — update_node reads/writes the file
    existing = [existing_node]

    fact = "extract_batch reversed course: production model switched to Qwen3.5-4B"
    cluster = [_entry(session_id="s1", decisions=[fact])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _RoutingFakeClient(
        extract_completion=_ok_envelope(fact),
        action_completion=(
            '{"actions": [{"index": 0, "action": "contradict", "existing_id": "kn_old", '
            '"contradiction_note": "model switched from Ministral to Qwen3.5-4B"}]}'
        ),
    )
    result = _run_extract_path(cluster, "clu", client, "m", failures_path, existing, kn_path)
    assert result is not None
    assert result.nodes_contradicted == 1
    assert result.nodes_created == 1  # legacy no-db path creates the reversing node directly
    persisted = {n.id: n for n in read_nodes(kn_path)}
    assert persisted["kn_old"].status == "contradicted"


# --- BLOCKER 2 fix (Sentinel, 2026-07-15): FULL B1->B2->B3 fruit for the temporal bound, all
# three actions, read from the PERSISTED node (not the _decide_actions dict — Opus's own P1
# probe stopped one layer short of this: "fruit-to-the-dict is not fruit-to-the-database").
# Mirrors Sentinel's exact reproduction technique (single-fact unit, one expiry ref, so B1fix's
# fan-out suppression does not apply and the bound genuinely flows).

def _ok_envelope_with_temporal(fact_text: str, *, role: str, resolved: str) -> str:
    """A single-fact, single-ref Stage-1 completion — deliberately ONE output so B1fix's
    fan-out suppression (>1 usable output -> null) does not mask the bound-flow this covers."""
    return json.dumps({
        "extracted_at": _EXTRACTED_AT,
        "facts": [{"text": fact_text, "category": "fact"}],
        "decisions": [],
        "temporal_refs": [{"raw": "temporal-expr", "resolved": resolved, "role": role}],
    })


def test_run_extract_path_create_persists_role_mapped_bound(tmp_path):
    """Pin the CREATE case as a permanent full-path regression guard — Sentinel's own fruit
    confirmed this already works; this test is the guard against it silently regressing."""
    fact = "the API key expires April 30 in the production_env config"  # specificity signal
    cluster = [_entry(session_id="s1", done=[fact])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _RoutingFakeClient(
        extract_completion=_ok_envelope_with_temporal(fact, role="expiry", resolved="2025-04-30"),
        action_completion='{"actions": [{"index": 0, "action": "create"}]}',
    )
    kn_path = tmp_path / "knowledge.jsonl"
    result = _run_extract_path(cluster, "clu", client, "m", failures_path, [], kn_path)
    assert result is not None
    assert result.nodes_created == 1
    persisted = read_nodes(kn_path)
    assert persisted[0].valid_until == "2025-04-30"  # the anchored bound, persisted


def test_run_extract_path_corroborate_persists_role_mapped_bound(tmp_path):
    """BLOCKER 2, corroborate sub-case: full path, real _decide_actions + real reconcile,
    persisted-node read. Before the fix: bound reaches _decide_actions's dict but reconcile's
    corroborate branch never reads it -> persisted node stays valid_until=None forever."""
    kn_path = tmp_path / "knowledge.jsonl"
    existing_node = KnowledgeNode.create(
        content="the API key expires soon in the production_env config",
        category="fact", node_id="kn_abc123",
    )
    from synapt.recall.knowledge import append_node
    append_node(existing_node, kn_path)
    existing = [existing_node]

    fact = "the API key expires April 30 in the production_env config"
    cluster = [_entry(session_id="s2", done=[fact])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _RoutingFakeClient(
        extract_completion=_ok_envelope_with_temporal(fact, role="expiry", resolved="2025-04-30"),
        action_completion='{"actions": [{"index": 0, "action": "corroborate", "existing_id": "kn_abc123"}]}',
    )
    result = _run_extract_path(cluster, "clu", client, "m", failures_path, existing, kn_path)
    assert result is not None
    assert result.nodes_corroborated == 1
    persisted = {n.id: n for n in read_nodes(kn_path)}
    assert persisted["kn_abc123"].valid_until == "2025-04-30"  # filled, was missing


def test_run_extract_path_contradict_legacy_persists_candidate_bound_on_replacement(tmp_path):
    """BLOCKER 2, contradict sub-case (legacy no-db path): full path, persisted-node read.
    Before the fix: the replacement node's valid_from is cluster-derived (ignores the candidate
    entirely) and valid_until is never set at all -- Sentinel's fruit: "replacement had
    valid_until=None"."""
    kn_path = tmp_path / "knowledge.jsonl"
    existing_node = KnowledgeNode.create(
        content="extract_batch: production model is Ministral-3B",
        category="decision", node_id="kn_old",
    )
    from synapt.recall.knowledge import append_node
    append_node(existing_node, kn_path)
    existing = [existing_node]

    fact = "the API key expires April 30 in the production_env config"
    cluster = [_entry(session_id="s1", decisions=[fact])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _RoutingFakeClient(
        extract_completion=_ok_envelope_with_temporal(fact, role="expiry", resolved="2025-04-30"),
        action_completion=(
            '{"actions": [{"index": 0, "action": "contradict", "existing_id": "kn_old", '
            '"contradiction_note": "reversed"}]}'
        ),
    )
    result = _run_extract_path(cluster, "clu", client, "m", failures_path, existing, kn_path)
    assert result is not None
    assert result.nodes_contradicted == 1
    assert result.nodes_created == 1
    persisted = {n.id: n for n in read_nodes(kn_path)}
    replacement = [n for n in persisted.values() if n.status == "active"][0]
    assert replacement.valid_until == "2025-04-30"  # candidate's bound, persisted


def test_run_extract_path_all_failed_extraction_yields_zero_result_not_none(tmp_path):
    """A fully-failed EXTRACT batch (every unit fails B1) is still a PROCESSED cluster — the
    same semantics as the monolith's own "no durable patterns" empty-nodes-list outcome. Only an
    infrastructure failure (exception, lost marker) returns None."""
    cluster = [_entry(done=["a"])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _FakeClient(completion="garbage-not-json")
    kn_path = tmp_path / "knowledge.jsonl"
    result = _run_extract_path(cluster, "clu", client, "m", failures_path, [], kn_path)
    assert result is not None
    assert result.nodes_created == 0
    assert result.nodes_corroborated == 0
    assert result.nodes_contradicted == 0


def test_run_extract_path_empty_cluster_returns_empty_result(tmp_path):
    cluster = [_entry(focus="just a focus line")]  # prefilter reads neither focus nor next_steps
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _FakeClient(completion="{}")
    kn_path = tmp_path / "knowledge.jsonl"
    result = _run_extract_path(cluster, "clu", client, "m", failures_path, [], kn_path)
    assert result is not None
    assert result.nodes_created == 0


# --- INTEGRATION: the real consolidate() flag-branch dispatch, full B1->B2->B3 --------------

def test_consolidate_flag_branch_creates_nodes_end_to_end(tmp_path, monkeypatch):
    """End-to-end through the real consolidate(): with SYNAPT_USE_EXTRACT on, the
    _process_cluster flag-branch fires, resolves its closure vars (model/failures_path/client/
    existing_nodes/kn_path), and now runs the FULL B1->B2->B3 pipeline — nodes ARE created
    (superseding the earlier B1-only "creates no nodes" assertion, which was true only before
    B2/B3 landed). Catches the dispatch/closure-scoping class the direct-call unit tests
    cannot see."""
    from synapt.recall.consolidate import consolidate
    from synapt.recall.journal import JournalEntry, append_entry, _journal_path

    jpath = _journal_path(tmp_path)
    for sid, ts, done in [
        ("s1", "2026-07-13T10:00:00Z", ["wired extract_batch into consolidate step three"]),
        ("s2", "2026-07-13T11:00:00Z", ["tested extract_batch count invariance in consolidate"]),
        ("s3", "2026-07-13T12:00:00Z", ["extract_batch consolidate wiring behind the flag"]),
    ]:
        append_entry(JournalEntry(timestamp=ts, session_id=sid, done=done), jpath)

    # fact text needs a specificity signal (snake_case "extract_batch") to clear
    # _apply_consolidation_result's EXISTING low-specificity filter — same filter the monolith
    # path already contends with, not new B2/B3 behavior.
    fake = _RoutingFakeClient(
        extract_completion=_ok_envelope("recall#875 wired extract_batch"),
        action_completion='{"actions": [{"index": 0, "action": "create"}]}',
    )
    monkeypatch.setattr(
        "synapt.recall.consolidate._get_consolidation_client", lambda *a, **k: fake
    )
    monkeypatch.setenv("SYNAPT_USE_EXTRACT", "1")

    result = consolidate(project_dir=tmp_path, force=True, min_entries=3)

    assert result.entries_processed == 3
    assert result.clusters_found >= 1
    assert result.nodes_created > 0            # B1->B2->B3: the pipeline now creates real nodes
    assert len(fake.calls) > 0                  # the extract path actually ran the model seam
    kn_path = tmp_path / ".synapt" / "recall" / "knowledge.jsonl"
    assert kn_path.exists() and kn_path.read_text().strip() != ""


# --- REVIEW-FIX: dense-cluster token budget reaches the REAL client (not just _decide_actions) --

def test_run_extract_path_dense_cluster_scaled_budget_reaches_the_real_client(tmp_path):
    """B3-level proof that the scaled action-decision budget (Opus/Sentinel blocker #2) reaches
    the ACTUAL client through _run_extract_path -> _make_recall_infer, not just the isolated
    _decide_actions unit tests (which inject infer directly, bypassing _make_recall_infer's
    per-request max_tokens reading entirely — a wiring gap those tests structurally cannot
    see, the same class of gap the flag-branch dispatch test exists to catch for B1)."""
    n = 40
    cluster = [_entry(session_id="s1", done=[f"recall#875 dense fact number {i} extract_batch" for i in range(n)])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _RoutingFakeClient(
        extract_completion=_ok_envelope("dense"),
        action_completion='{"actions": [{"index": 0, "action": "create"}]}',
    )
    kn_path = tmp_path / "knowledge.jsonl"
    result = _run_extract_path(cluster, "clu", client, "m", failures_path, [], kn_path)
    assert result is not None
    action_calls = [c for c in client.calls if "New Facts (indexed)" in c["messages"][0].content]
    assert len(action_calls) == 1
    requested_budget = action_calls[0]["kwargs"]["max_tokens"]
    assert requested_budget > 800, (
        f"action-decision call requested only {requested_budget} tokens for a {n}-candidate "
        "cluster — the scaled budget did not reach the real client through _run_extract_path"
    )
