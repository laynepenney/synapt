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


# --- FLAG-BRANCH BODY: envelopes + failure logging, NO nodes -------------------------------

def test_run_extract_path_logs_failed_markers_and_creates_no_nodes(tmp_path):
    cluster = [_entry(done=["a", "b"], decisions=["c"])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    client = _FakeClient(completion="garbage-not-json")
    ok = _run_extract_path(cluster, "clu", client, "m", failures_path)
    assert ok is True
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
    client = _FakeClient(completion=_ok_envelope("a"))
    ok = _run_extract_path(cluster, "clu", client, "m", failures_path)
    assert ok is True
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


def test_run_extract_path_returns_false_when_a_marker_write_fails(tmp_path, monkeypatch):
    """THE silent-drop Sentinel caught: previously, an OSError in _log_extract_failure was
    swallowed and _run_extract_path still returned True, declaring the cluster processed while
    a failed unit vanished with no record. Must now propagate as a non-successful cluster."""
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
    ok = _run_extract_path(cluster, "clu", client, "m", failures_path)
    assert ok is False  # the lost marker must NOT be reported as a clean success


# --- INTEGRATION: the real consolidate() flag-branch dispatch ------------------------------

def test_consolidate_flag_branch_dispatches_and_creates_no_nodes(tmp_path, monkeypatch):
    """End-to-end through the real consolidate(): with SYNAPT_USE_EXTRACT on, the
    _process_cluster flag-branch must fire, resolve its closure vars (model/failures_path/
    client), run the extract path, and create NO nodes (B1 stops at envelopes). Catches the
    dispatch/closure-scoping class the direct-call unit tests cannot see."""
    from synapt.recall.consolidate import consolidate
    from synapt.recall.journal import JournalEntry, append_entry, _journal_path

    jpath = _journal_path(tmp_path)
    for sid, ts, done in [
        ("s1", "2026-07-13T10:00:00Z", ["wired extract_batch into consolidate step three"]),
        ("s2", "2026-07-13T11:00:00Z", ["tested extract_batch count invariance in consolidate"]),
        ("s3", "2026-07-13T12:00:00Z", ["extract_batch consolidate wiring behind the flag"]),
    ]:
        append_entry(JournalEntry(timestamp=ts, session_id=sid, done=done), jpath)

    fake = _FakeClient(completion=_ok_envelope("durable"))
    monkeypatch.setattr(
        "synapt.recall.consolidate._get_consolidation_client", lambda *a, **k: fake
    )
    monkeypatch.setenv("SYNAPT_USE_EXTRACT", "1")

    result = consolidate(project_dir=tmp_path, force=True, min_entries=3)

    assert result.entries_processed == 3
    assert result.clusters_found >= 1
    assert result.nodes_created == 0          # B1 creates no nodes — envelopes only
    assert result.nodes_corroborated == 0
    assert len(fake.calls) > 0                 # the extract path actually ran the model seam
    # no knowledge file written
    assert not (tmp_path / ".synapt" / "recall" / "knowledge.jsonl").exists() or \
        (tmp_path / ".synapt" / "recall" / "knowledge.jsonl").read_text().strip() == ""
