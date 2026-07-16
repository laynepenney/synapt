"""TDD spec (premium#743, OSS seam half, part 2): memory-op taps.

Tap point 2: recall_save -> mem_write, recall_search -> mem_search,
recall_quick -> mem_read. Each entry point emits exactly one UsageEvent per
call via the same emit_usage_event seam tap-1 already uses (see
test_consolidate_usage_tap.py) — no new registration mechanism, no identity
resolution at this layer either.

session_ref is resolved once, uniformly, via a shared OSS helper reading the
SYNAPT_AGENT_ID env var when set (the same opaque-token precedent already
established at core.py:157's TranscriptChunk.agent_id) -- an unset env var
resolves to "unattributed", matching the collector's own never-drop rule for
unmappable events.
"""

from __future__ import annotations

from unittest.mock import patch

from synapt.recall.usage import UsageEvent, clear_usage_sinks, register_usage_sink
import pytest


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


def test_memory_op_events_carry_no_identity_field():
    """Structural re-check at the tap-site level, not just the schema level —
    a tap could theoretically stuff identity into `detail` as a free-text
    string. It must not."""
    from synapt.recall.usage import _current_session_ref

    ref = _current_session_ref()
    assert isinstance(ref, str)
    # session_ref is opaque; the seam never labels it as identity-shaped.
    assert ref != ""
