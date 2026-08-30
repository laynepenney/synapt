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
import threading

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


# --- SPEC ADDITIONS (Apollo, implementation range) ----------------------------
# Contract-read item 6, ratified before implementation: the witness that
# registers from another thread during emission is required, not optional.
#
# MEASURED GAP, before writing these: reverting emit_usage_event to iterate the
# LIVE registry instead of a snapshot left all 17 tests in this file GREEN.  The
# registry is process-global mutable state and recall emits from consolidation
# stages and from the native poller's own loop, so mutate-during-iterate is
# reachable rather than theoretical -- and in Python it does not raise.  It
# silently DELIVERS to a sink registered after the emit began, and silently
# SKIPS a sink when one is removed mid-iteration.  No exception, no symptom,
# which is why the fail-open rule cannot save us here.
#
# WHAT IS NOT WITNESSED HERE, disclosed rather than left for a reader to assume:
# deleting the LOCK from register_usage_sink leaves all of these GREEN, measured.
# On CPython list.append is atomic under the GIL, so the lock is not observable
# at this granularity.  It is NOT retained because append is unsafe elsewhere --
# under PEP 703 free-threading, list operations are internally locked and append
# stays thread-safe.  It is retained because emit_usage_event reads TWO module
# globals, _sinks and _debug_sink_path, and needs them as one consistent pair,
# which no per-object locking provides on any build.  That is REASONING, not a
# measurement: there is no witness for it here and none is claimed.  The SNAPSHOT
# is the part that carries weight on CPython, and the two tests below hold it.


def test_a_sink_registered_during_emission_does_not_receive_that_event():
    """The snapshot boundary, from another thread.

    Under a live-list iteration a sink appended while the loop is running is
    visited by that same loop, so it receives an event it was not registered
    for. Registration happens on a DIFFERENT thread on purpose: the defect this
    pins is a threading one, and a same-thread registration would also pass
    against an implementation that merely copied the list for re-entrancy.
    """
    late_deliveries = []
    registered = threading.Event()

    def late_sink(event):
        late_deliveries.append(event)

    def registers_from_another_thread(event):
        thread = threading.Thread(
            target=lambda: (register_usage_sink(late_sink), registered.set()),
        )
        thread.start()
        thread.join()

    register_usage_sink(registers_from_another_thread)
    emit_usage_event(
        UsageEvent(ts="2026-07-16T00:00:00Z", session_ref="s1", op="infer", detail="B2"),
    )

    assert registered.is_set(), "the registering thread must have run"
    assert late_deliveries == [], "a sink registered mid-emission must not receive that event"


def test_a_sink_removed_during_emission_does_not_silently_skip_the_next_sink():
    """The other direction, and the one that LOSES events.

    A first draft of this test removed a LATER sink and passed against a
    live-list implementation, i.e. it discriminated nothing -- which is worse
    than no test, because it reads as coverage. Removing element i while the
    loop sits at index i is what shifts the list under the cursor: the sink that
    moves into slot i is never visited. So the removal here is SELF-removal from
    index 0, and the assertion is that the sink at index 1 still receives.

    Measured, not reasoned: against a live-list emit this fails and against the
    snapshot it passes; the earlier later-sink form passed against both.
    """
    received_second = []
    removed = threading.Event()

    def second(event):
        received_second.append(event)

    def first(event):
        thread = threading.Thread(
            target=lambda: (unregister_usage_sink(first), removed.set()),
        )
        thread.start()
        thread.join()

    register_usage_sink(first)
    register_usage_sink(second)
    event = UsageEvent(
        ts="2026-07-16T00:00:00Z", session_ref="s1", op="infer", detail="B2",
    )
    emit_usage_event(event)

    assert removed.is_set()
    assert received_second == [event], "a self-removal mid-emission must not skip the next sink"


def test_concurrent_registration_during_emission_never_raises_and_never_loses_a_sink():
    """The lock itself, under contention rather than by inspection.

    Eight threads register while the main thread emits repeatedly. The
    assertions are that nothing raises -- emit_usage_event never raising is the
    never-disrupt rule, and a fail-open swallow would hide a registry
    corruption -- and that every registration is present afterwards, which a
    lost update under a racing append would violate.
    """
    sinks = [lambda event, index=index: None for index in range(8)]
    errors: list[BaseException] = []
    stop = threading.Event()

    def register_all():
        try:
            for sink in sinks:
                register_usage_sink(sink)
        except BaseException as exc:  # noqa: BLE001 - the assertion is that there are none
            errors.append(exc)

    threads = [threading.Thread(target=register_all) for _ in range(8)]
    for thread in threads:
        thread.start()
    event = UsageEvent(
        ts="2026-07-16T00:00:00Z", session_ref="s1", op="infer", detail="B2",
    )
    for _ in range(200):
        emit_usage_event(event)
    for thread in threads:
        thread.join()
    stop.set()

    assert errors == []
    from synapt.recall.usage import _sinks

    assert len(_sinks) == 8 * 8, "every registration must survive; a lost update means a racing append"


def test_the_debug_sink_lifecycle_is_independent_of_the_registered_sinks(tmp_path):
    """clear_usage_sinks must not disable the debug sink, and disabling the
    debug sink must not drop registered sinks. Two lifecycles that happen to
    share one lock are easy to accidentally fuse into one."""
    path = tmp_path / "usage_debug.jsonl"
    received = []
    register_usage_sink(received.append)
    enable_debug_sink(path)
    event = UsageEvent(
        ts="2026-07-16T00:00:00Z", session_ref="s1", op="infer", detail="B2",
    )

    clear_usage_sinks()
    emit_usage_event(event)
    assert received == [], "cleared sinks must not receive"
    assert len(path.read_text().splitlines()) == 1, "the debug sink must survive clear_usage_sinks"

    register_usage_sink(received.append)
    disable_debug_sink()
    emit_usage_event(event)
    assert received == [event], "disabling the debug sink must not drop registered sinks"
    assert len(path.read_text().splitlines()) == 1


def test_the_session_reference_is_read_at_call_time_not_at_import(monkeypatch):
    """current_session_ref must reflect the environment as it is WHEN CALLED.

    A module-level constant would be captured at import, before any runtime had
    a chance to set the variable, and would report "unattributed" forever with
    no test going red -- so this pins the read, not the value.
    """
    from synapt.recall.usage import UNATTRIBUTED, current_session_ref

    monkeypatch.delenv("SYNAPT_AGENT_ID", raising=False)
    assert current_session_ref() == UNATTRIBUTED

    monkeypatch.setenv("SYNAPT_AGENT_ID", "an-opaque-token")
    assert current_session_ref() == "an-opaque-token"

    monkeypatch.setenv("SYNAPT_AGENT_ID", "")
    assert current_session_ref() == UNATTRIBUTED, "empty must fall back, not become a falsy bucket"
