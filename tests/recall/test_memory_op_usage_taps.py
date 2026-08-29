"""TDD spec: OSS memory-operation taps with an opaque correlation key.

Tap point 2: recall_save -> mem_write, recall_search -> mem_search,
recall_quick -> mem_read. Each entry point emits exactly one UsageEvent per
call via the same emit_usage_event seam tap-1 already uses (see
test_consolidate_usage_tap.py) — no new registration mechanism, no identity
resolution at this layer either.

``session_ref`` is an opaque correlation key supplied by the environment
(``SYNAPT_AGENT_ID`` when set, otherwise ``"unattributed"``). OSS never
interprets it. The schema has no identity-named field; that is a guarantee about
the schema, not about what a caller chooses to place in ``session_ref``. The
event's ``detail`` remains the fixed public operation name and never copies that
reference.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("synapt.recall.usage", reason="TDD contract: seam not implemented yet")

from synapt.recall.usage import UsageEvent, clear_usage_sinks, register_usage_sink


@pytest.fixture(autouse=True)
def _clean_usage_state():
    clear_usage_sinks()
    yield
    clear_usage_sinks()


class _FakeEmbeddingProvider:
    def embed(self, texts):
        return [[0.01] * 384 for _ in texts]


def test_recall_save_emits_a_mem_write_event(tmp_path):
    from synapt.recall.server import recall_save

    received: list[UsageEvent] = []
    register_usage_sink(received.append)

    with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
         patch("synapt.recall.server.get_embedding_provider", return_value=_FakeEmbeddingProvider()), \
         patch("synapt.recall.server._invalidate_cache"):
        recall_save(content="Deploy previews expire after 7 days", category="workflow")

    mem_write_events = [e for e in received if e.op == "mem_write"]
    assert len(mem_write_events) == 1
    assert mem_write_events[0].detail == "recall_save"


def test_recall_search_emits_a_mem_search_event(tmp_path):
    from synapt.recall.server import recall_search

    received: list[UsageEvent] = []
    register_usage_sink(received.append)

    with patch("synapt.recall.server.Path.cwd", return_value=tmp_path):
        recall_search("deploy previews")

    mem_search_events = [e for e in received if e.op == "mem_search"]
    assert len(mem_search_events) == 1
    assert mem_search_events[0].detail == "recall_search"


def test_recall_quick_emits_a_mem_read_event(tmp_path):
    from synapt.recall.server import recall_quick

    received: list[UsageEvent] = []
    register_usage_sink(received.append)

    with patch("synapt.recall.server.Path.cwd", return_value=tmp_path):
        recall_quick("deploy previews")

    mem_read_events = [e for e in received if e.op == "mem_read"]
    assert len(mem_read_events) == 1
    assert mem_read_events[0].detail == "recall_quick"


def test_memory_op_details_do_not_copy_the_session_reference(tmp_path, monkeypatch):
    """Each tap keeps the opaque routing reference out of free-text ``detail``."""
    from synapt.recall.server import recall_quick, recall_save, recall_search

    marker = "identity-shaped-control-value"
    monkeypatch.setenv("SYNAPT_AGENT_ID", marker)
    received: list[UsageEvent] = []
    register_usage_sink(received.append)

    with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
         patch("synapt.recall.server.get_embedding_provider", return_value=_FakeEmbeddingProvider()), \
         patch("synapt.recall.server._invalidate_cache"):
        recall_save(content="A durable memory", category="workflow")
        recall_search("durable memory")
        recall_quick("durable memory")

    memory_events = [event for event in received if event.op.startswith("mem_")]
    assert [event.detail for event in memory_events] == [
        "recall_save", "recall_search", "recall_quick",
    ]
    assert all(marker not in event.detail for event in memory_events)
    assert all(event.session_ref == marker for event in memory_events)
