"""TDD spec: OSS usage-event emission with an opaque correlation key.

OSS calls ``emit_usage_event`` at each tap point; zero registered sinks is the
default (no-op); an optional local JSONL debug sink exists for development.
``session_ref`` is an opaque correlation key supplied by the environment
(``SYNAPT_AGENT_ID`` when set, otherwise ``"unattributed"``). OSS never
interprets it. The schema has no identity-named field; that is a guarantee about
the schema, not about what a caller chooses to place in ``session_ref``.
"""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("synapt.recall.usage", reason="TDD contract: seam not implemented yet")

from synapt.recall.usage import (
    UsageEvent,
    clear_usage_sinks,
    disable_debug_sink,
    emit_usage_event,
    enable_debug_sink,
    register_usage_sink,
    unregister_usage_sink,
)


@pytest.fixture(autouse=True)
def _clean_usage_state():
    clear_usage_sinks()
    disable_debug_sink()
    yield
    clear_usage_sinks()
    disable_debug_sink()


# --- schema: no identity-named field ------------------------------------------


def test_usage_event_schema_has_no_identity_field():
    """The schema has no identity-named field.

    ``session_ref`` is an opaque correlation key supplied by the environment
    (``SYNAPT_AGENT_ID`` when set, otherwise ``"unattributed"``). OSS never
    interprets it. This is a guarantee about the schema, not about what a caller
    chooses to place in ``session_ref``.
    """
    fields = set(UsageEvent.__dataclass_fields__.keys())
    identity_shaped = {"agent_id", "org_id", "team_id", "user", "user_id", "identity"}
    assert fields.isdisjoint(identity_shaped)
    assert fields == {
        "ts", "session_ref", "op", "detail", "model",
        "tokens_in", "tokens_out", "cached_tokens", "duration_ms",
    }


def test_usage_event_carries_counts_never_money():
    """No cost, dollar, or price field exists or is computed at emission."""
    fields = set(UsageEvent.__dataclass_fields__.keys())
    money_shaped = {"cost", "cost_usd", "price", "dollars", "amount"}
    assert fields.isdisjoint(money_shaped)


def test_usage_event_rejects_an_unknown_op():
    with pytest.raises(ValueError, match="op"):
        UsageEvent(
            ts="2026-07-16T00:00:00Z", session_ref="s1", op="bogus_op", detail="x",
        )


@pytest.mark.parametrize(
    "op", ["infer", "mem_write", "mem_read", "mem_search", "consolidate_stage", "channel_post"],
)
def test_usage_event_accepts_every_designed_op(op):
    UsageEvent(ts="2026-07-16T00:00:00Z", session_ref="s1", op=op, detail="x")


# --- no-op default -------------------------------------------------------------


def test_emit_with_no_registered_sink_is_a_silent_no_op():
    event = UsageEvent(
        ts="2026-07-16T00:00:00Z", session_ref="s1", op="infer", detail="B2",
    )
    emit_usage_event(event)  # must not raise, must not require a sink


def test_a_raising_sink_never_disrupts_the_emitting_call():
    """A broken sink must never break the operation being metered — the seam's
    own never-disrupt rule, same shape as the existing dedup-decision logging."""
    def bad_sink(_event):
        raise RuntimeError("sink is broken")

    register_usage_sink(bad_sink)
    event = UsageEvent(
        ts="2026-07-16T00:00:00Z", session_ref="s1", op="infer", detail="B2",
    )
    emit_usage_event(event)  # must not raise


# --- registered sinks ------------------------------------------------------


def test_a_registered_sink_receives_the_emitted_event():
    received = []
    register_usage_sink(received.append)
    event = UsageEvent(
        ts="2026-07-16T00:00:00Z", session_ref="s1", op="infer", detail="B2",
        model="qwen3.5-4b", tokens_in=120, tokens_out=340,
    )
    emit_usage_event(event)
    assert received == [event]


def test_multiple_registered_sinks_all_receive_the_event():
    received_a, received_b = [], []
    register_usage_sink(received_a.append)
    register_usage_sink(received_b.append)
    event = UsageEvent(
        ts="2026-07-16T00:00:00Z", session_ref="s1", op="mem_write", detail="recall_save",
    )
    emit_usage_event(event)
    assert received_a == [event]
    assert received_b == [event]


def test_unregister_usage_sink_stops_delivery():
    received = []

    def receive(event):
        received.append(event)

    register_usage_sink(receive)
    unregister_usage_sink(receive)
    emit_usage_event(
        UsageEvent(ts="2026-07-16T00:00:00Z", session_ref="s1", op="infer", detail="B2"),
    )
    assert received == []


# --- optional local JSONL debug sink -----------------------------------------


def test_enabled_debug_sink_appends_one_jsonl_record_per_event(tmp_path):
    path = tmp_path / "usage_debug.jsonl"
    enable_debug_sink(path)
    event = UsageEvent(
        ts="2026-07-16T00:00:00Z", session_ref="s1", op="infer", detail="B2",
        model="qwen3.5-4b", tokens_in=120, tokens_out=340, cached_tokens=None,
        duration_ms=812,
    )
    emit_usage_event(event)
    emit_usage_event(event)

    lines = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(lines) == 2
    assert lines[0]["session_ref"] == "s1"
    assert lines[0]["op"] == "infer"
    assert lines[0]["tokens_in"] == 120
    assert lines[0]["cached_tokens"] is None


def test_debug_sink_write_failure_is_logged_not_raised(tmp_path, caplog):
    """Best-effort, same shape as _log_dedup_decision: a debug-sink write
    failure must never break the operation being metered."""
    caplog.set_level(logging.DEBUG)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    unwritable = blocker / "usage_debug.jsonl"
    enable_debug_sink(unwritable)

    emit_usage_event(
        UsageEvent(ts="2026-07-16T00:00:00Z", session_ref="s1", op="infer", detail="B2"),
    )  # must not raise
    messages = [record.getMessage().lower() for record in caplog.records]
    assert any("usage" in message and "sink" in message for message in messages)


def test_disable_debug_sink_stops_writes(tmp_path):
    path = tmp_path / "usage_debug.jsonl"
    enable_debug_sink(path)
    event = UsageEvent(
        ts="2026-07-16T00:00:00Z", session_ref="s1", op="infer", detail="B2",
    )
    emit_usage_event(event)
    before = path.read_text()
    disable_debug_sink()
    emit_usage_event(event)
    assert path.read_text() == before
