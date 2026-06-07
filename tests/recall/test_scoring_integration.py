"""Integration tests for PR4f-A Phase 2: scoring strategy threading at
consolidate.py + enrich.py callsites.

Per config#339 Pattern 4 ratification + DI-threading-seam rule (per
feedback_xfail_removal_doc_sweep.md): paired reject/accept E2E threading tests
for the consolidate + enrich integration seam.

Phase 1 (recall#823 Phase 1) tested the Protocol + registry seam in isolation.
Phase 2 tests cover:
- score_cluster_chunks(JournalEntries) threads through get_active_strategy
- score_transcript_windows(windows) threads through get_active_strategy
- E2E: activate fake strategy → call helpers → verify fake invoked
- Backward-compat: with no plugin activated, default RecencyScoring is used
"""

from __future__ import annotations

import pytest

from synapt.recall.consolidate import score_cluster_chunks
from synapt.recall.enrich import score_transcript_windows
from synapt.recall.journal import JournalEntry
from synapt.recall.scoring import (
    ChunkScoringStrategy,
    DEFAULT_RECENT_TOKEN_WINDOW,
    RecencyScoring,
    ScoredChunk,
    ScoringInput,
    activate_scoring_strategy,
    register_scoring_strategy,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    reset_registry()
    yield
    reset_registry()


def _entry(focus: str, timestamp: str, session_id: str = "s1") -> JournalEntry:
    """Build a minimal JournalEntry for tests."""
    return JournalEntry(
        timestamp=timestamp,
        session_id=session_id,
        focus=focus,
    )


class _RecordingStrategy:
    """Fake strategy that records score() calls for threading verification."""

    name = "recording"
    window = 8

    def __init__(self):
        self.score_called_with: list[list[ScoringInput]] = []

    def score(self, inputs: list[ScoringInput]) -> list[ScoredChunk]:
        self.score_called_with.append(list(inputs))
        return [
            ScoredChunk(input=i, score=float(i.position), strategy_name=self.name)
            for i in inputs
        ]


# === consolidate.score_cluster_chunks threading ===


class TestScoreClusterChunksThreading:

    def test_empty_cluster_returns_empty(self):
        assert score_cluster_chunks([]) == []

    def test_default_strategy_uses_recency(self):
        """No plugin activated → default RecencyScoring is used."""
        cluster = [
            _entry("oldest", "2026-01-01T00:00:00Z"),
            _entry("middle", "2026-02-01T00:00:00Z"),
            _entry("newest", "2026-03-01T00:00:00Z"),
        ]
        scored = score_cluster_chunks(cluster)
        assert len(scored) == 3
        assert all(s.strategy_name == "recency" for s in scored)
        # All in window (default 16), scores are linear ramp 0.0 → 1.0
        assert scored[0].score == 0.0
        assert scored[-1].score == 1.0

    def test_active_strategy_threads_through(self):
        """Activating a custom strategy threads through score_cluster_chunks."""
        recorder = _RecordingStrategy()
        register_scoring_strategy("recording", recorder)
        activate_scoring_strategy("recording")

        cluster = [
            _entry("a", "2026-01-01T00:00:00Z"),
            _entry("b", "2026-02-01T00:00:00Z"),
        ]
        scored = score_cluster_chunks(cluster)

        assert len(recorder.score_called_with) == 1
        inputs = recorder.score_called_with[0]
        assert len(inputs) == 2
        assert inputs[0].content == "a"  # focus → content
        assert inputs[1].content == "b"  # focus → content
        assert all(s.strategy_name == "recording" for s in scored)

    def test_cluster_sorted_by_timestamp(self):
        """Unsorted cluster is ordered by timestamp before scoring (recency-by-time)."""
        cluster = [
            _entry("newest", "2026-03-01T00:00:00Z"),
            _entry("oldest", "2026-01-01T00:00:00Z"),
            _entry("middle", "2026-02-01T00:00:00Z"),
        ]
        scored = score_cluster_chunks(cluster)
        # Position 0 should be oldest, position 2 should be newest
        assert scored[0].input.content == "oldest"
        assert scored[1].input.content == "middle"
        assert scored[2].input.content == "newest"

    def test_metadata_includes_timestamp_and_session(self):
        """JournalEntry metadata is preserved in ScoringInput.metadata."""
        cluster = [_entry("c", "2026-01-01T00:00:00Z", session_id="sess-a")]
        scored = score_cluster_chunks(cluster)
        assert scored[0].input.metadata["timestamp"] == "2026-01-01T00:00:00Z"
        assert scored[0].input.metadata["session_id"] == "sess-a"


# === enrich.score_transcript_windows threading ===


class TestScoreTranscriptWindowsThreading:

    def test_empty_windows_returns_empty(self):
        assert score_transcript_windows([]) == []

    def test_default_strategy_uses_recency(self):
        windows = ["oldest", "middle", "newest"]
        scored = score_transcript_windows(windows)
        assert len(scored) == 3
        assert all(s.strategy_name == "recency" for s in scored)
        assert scored[0].score == 0.0
        assert scored[-1].score == 1.0

    def test_active_strategy_threads_through(self):
        recorder = _RecordingStrategy()
        register_scoring_strategy("recording", recorder)
        activate_scoring_strategy("recording")

        windows = ["w0", "w1", "w2"]
        scored = score_transcript_windows(windows)

        assert len(recorder.score_called_with) == 1
        inputs = recorder.score_called_with[0]
        assert [i.content for i in inputs] == ["w0", "w1", "w2"]
        assert [i.position for i in inputs] == [0, 1, 2]
        assert all(s.strategy_name == "recording" for s in scored)

    def test_position_preserves_input_order(self):
        """score_transcript_windows preserves caller-provided order; no resort."""
        windows = ["last", "first", "middle"]  # caller passes whatever order
        scored = score_transcript_windows(windows)
        # Positions match input order (no resort)
        assert scored[0].input.content == "last"
        assert scored[0].input.position == 0
        assert scored[1].input.content == "first"
        assert scored[1].input.position == 1


# === E2E DI-threading-seam pairs (Phase 2) ===


class TestPhase2EndToEndThreadingSeam:
    """DI-threading-seam reject/accept E2E pairs for Phase 2 integration helpers.

    Per feedback_xfail_removal_doc_sweep.md DI-threading-seam rule: when adding
    a substrate seam, paired reject/accept E2E tests verify the contract holds
    at the integration boundary, not just at the unit level.
    """

    def test_accept_register_activate_then_consolidate_uses_active(self):
        recorder = _RecordingStrategy()
        register_scoring_strategy("recording", recorder)
        activate_scoring_strategy("recording")

        cluster = [_entry(f"c{i}", f"2026-0{i+1}-01T00:00:00Z") for i in range(3)]
        scored = score_cluster_chunks(cluster)

        assert recorder.score_called_with, "recorder must have been invoked"
        # strategy_name reflects the strategy's self-identification (its .name attr),
        # not necessarily the registration key. Pattern 4 single-active-strategy
        # makes the two coincide in well-behaved plugins.
        assert all(s.strategy_name == recorder.name for s in scored)

    def test_accept_register_activate_then_enrich_uses_active(self):
        recorder = _RecordingStrategy()
        register_scoring_strategy("recording", recorder)
        activate_scoring_strategy("recording")

        windows = [f"window-{i}" for i in range(3)]
        scored = score_transcript_windows(windows)

        assert recorder.score_called_with, "recorder must have been invoked"
        assert all(s.strategy_name == recorder.name for s in scored)

    def test_reject_no_activation_falls_back_to_default(self):
        """Reject path: never-activated registry produces default RecencyScoring
        across BOTH integration helpers; default is reused across calls."""
        # Register but DO NOT activate
        recorder = _RecordingStrategy()
        register_scoring_strategy("rec", recorder)
        # NO activate_scoring_strategy call

        consolidate_result = score_cluster_chunks([
            _entry("a", "2026-01-01T00:00:00Z"),
        ])
        enrich_result = score_transcript_windows(["x"])

        # Recorder never invoked
        assert recorder.score_called_with == []
        # Both helpers used default recency strategy
        assert consolidate_result[0].strategy_name == "recency"
        assert enrich_result[0].strategy_name == "recency"

    def test_reject_activate_then_reset_falls_back(self):
        """Reject path: post-reset, helpers fall back to default even after prior
        activation."""
        recorder = _RecordingStrategy()
        register_scoring_strategy("rec", recorder)
        activate_scoring_strategy("rec")
        reset_registry()

        # After reset: no active strategy → default RecencyScoring
        result = score_cluster_chunks([_entry("a", "2026-01-01T00:00:00Z")])
        assert result[0].strategy_name == "recency"
        assert recorder.score_called_with == []
