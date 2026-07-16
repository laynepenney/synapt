"""TDD spec (premium#743, OSS seam half, part 2): tap point 3 — consolidation stages.

One consolidate_stage UsageEvent per stage per cluster (B1-B4), giving the
per-run cost decomposition dogfood analyses currently reconstruct by hand.
B4's decision-log provenance stays what it is (audit substrate) -- the meter
is volume/cost, not lineage; two records, two purposes, no overloading.

tokens_in/out/cached_tokens are null for these events (this op is not itself
a model call -- tap-1 already meters the individual infer calls inside each
stage; this is the stage's own wall-clock envelope). duration_ms is real.
"""

from __future__ import annotations

import json

from synapt.recall.journal import JournalEntry
from synapt.recall.usage import UsageEvent, clear_usage_sinks, register_usage_sink
import pytest


@pytest.fixture(autouse=True)
def _clean_usage_state():
    clear_usage_sinks()
    yield
    clear_usage_sinks()


def _entry(session_id="s1", *, done=None, decisions=None):
    return JournalEntry(
        timestamp="2026-07-16T10:00:00Z",
        session_id=session_id,
        done=list(done or []),
        decisions=list(decisions or []),
    )


def _ok_envelope(text: str) -> str:
    return json.dumps({
        "extracted_at": "2026-07-16T10:00:00Z",
        "facts": [{"text": f"fact from: {text}"}],
        "decisions": [],
        "temporal_refs": [],
    })


class _RoutingFakeClient:
    """Routes on ACTION_DECISION_PROMPT's marker ("New Facts (indexed)") vs.
    B4's rejoin marker ("groups") vs. plain extraction — same technique as the
    existing test_consolidate_extract.py fixtures."""

    def chat(self, *, model, messages, **kwargs):
        content = messages[0].content if messages else ""
        if "New Facts (indexed)" in content:
            return '{"actions": [{"index": 0, "action": "create"}]}'
        if "grouping and composing" in content:
            return '{"groups": [{"indices": [0], "content": "irrelevant, singleton"}]}'
        return _ok_envelope(content)


def test_run_extract_path_emits_one_consolidate_stage_event_per_stage(tmp_path):
    from synapt.recall.consolidate import _run_extract_path

    received: list[UsageEvent] = []
    register_usage_sink(received.append)

    fact = "recall#875 wired extract_batch into consolidation"
    cluster = [_entry(done=[fact])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    kn_path = tmp_path / "knowledge.jsonl"

    _run_extract_path(cluster, "clu", _RoutingFakeClient(), "m", failures_path, [], kn_path)

    stage_events = [e for e in received if e.op == "consolidate_stage"]
    assert [e.detail for e in stage_events] == ["B1", "B2", "B4", "B3"]
    for event in stage_events:
        assert event.tokens_in is None
        assert event.tokens_out is None
        assert event.cached_tokens is None
        assert event.duration_ms is not None
        assert event.duration_ms >= 0


def test_consolidate_stage_events_never_disrupt_the_pipeline_on_a_broken_sink(tmp_path):
    def bad_sink(_event):
        raise RuntimeError("sink is broken")

    register_usage_sink(bad_sink)

    from synapt.recall.consolidate import _run_extract_path

    fact = "recall#875 wired extract_batch into consolidation"
    cluster = [_entry(done=[fact])]
    failures_path = tmp_path / "consolidation_failures.jsonl"
    kn_path = tmp_path / "knowledge.jsonl"

    result = _run_extract_path(
        cluster, "clu", _RoutingFakeClient(), "m", failures_path, [], kn_path,
    )
    assert result is not None  # the pipeline must complete despite the broken sink
