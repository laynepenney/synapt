"""TDD contract for recall#837 informative recall_quick misses."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

from synapt.recall.server import recall_quick


@dataclass
class _NearMiss:
    topic: str
    score: float


@dataclass
class _QuickMissDiagnostics:
    sessions_searched: int
    oldest_session_started_at: str | None
    chunks_scanned: int
    best_score: float | None
    threshold: float
    near_misses: list[_NearMiss] = field(default_factory=list)
    reason: str = "below_threshold"

    def format_message(self) -> str:
        return "legacy diagnostic miss"


class _MissIndex:
    _embedding_status = "disabled"
    _embedding_reason = ""

    def __init__(self, diagnostics: _QuickMissDiagnostics) -> None:
        self._last_diagnostics = diagnostics
        self.lookup_kwargs: dict[str, object] | None = None

    def lookup(self, query: str, **kwargs: object) -> str:
        self.lookup_kwargs = kwargs
        return ""


def test_recall_quick_miss_returns_verified_absence_with_coverage_stats() -> None:
    index = _MissIndex(
        _QuickMissDiagnostics(
            sessions_searched=12,
            oldest_session_started_at="2026-05-01",
            chunks_scanned=384,
            best_score=0.08,
            threshold=0.20,
        )
    )

    with patch("synapt.recall.server._get_index", return_value=index):
        result = recall_quick("licenses proceeding")

    assert "No prior discussion found for 'licenses proceeding'" in result
    assert "searched 12 sessions back to 2026-05-01" in result
    assert "384 chunks scanned" in result
    assert "best score 0.08 below threshold 0.20" in result
    assert "Proceeding fresh is safe." in result
    assert "legacy diagnostic miss" not in result


def test_recall_quick_miss_includes_at_most_two_near_miss_topics_below_threshold() -> None:
    index = _MissIndex(
        _QuickMissDiagnostics(
            sessions_searched=7,
            oldest_session_started_at="2026-05-20",
            chunks_scanned=91,
            best_score=0.19,
            threshold=0.20,
            near_misses=[
                _NearMiss(topic="license boundary cleanup", score=0.19),
                _NearMiss(topic="grep intercept hook", score=0.17),
                _NearMiss(topic="old unrelated topic", score=0.05),
                _NearMiss(topic="at threshold should be a hit", score=0.20),
            ],
        )
    )

    with patch("synapt.recall.server._get_index", return_value=index):
        result = recall_quick("licenses proceeding")

    assert "Closest near misses:" in result
    assert '- "license boundary cleanup" at 0.19' in result
    assert '- "grep intercept hook" at 0.17' in result
    assert "old unrelated topic" not in result
    assert "at threshold should be a hit" not in result


def test_recall_quick_miss_threshold_boundary_does_not_report_equal_score_as_absence() -> None:
    index = _MissIndex(
        _QuickMissDiagnostics(
            sessions_searched=3,
            oldest_session_started_at="2026-06-01",
            chunks_scanned=44,
            best_score=0.20,
            threshold=0.20,
            reason="threshold_boundary",
        )
    )

    with patch("synapt.recall.server._get_index", return_value=index):
        result = recall_quick("threshold boundary")

    assert "No prior discussion found" not in result
    assert "best score 0.20 below threshold 0.20" not in result
    assert "threshold boundary" in result.lower()


def test_recall_quick_empty_corpus_is_not_a_verified_absence() -> None:
    index = _MissIndex(
        _QuickMissDiagnostics(
            sessions_searched=0,
            oldest_session_started_at=None,
            chunks_scanned=0,
            best_score=None,
            threshold=0.20,
            reason="empty_corpus",
        )
    )

    with patch("synapt.recall.server._get_index", return_value=index):
        result = recall_quick("anything")

    assert "No indexed recall corpus available for 'anything'" in result
    assert "0 sessions" in result
    assert "0 chunks scanned" in result
    assert "Proceeding fresh is safe." not in result
    assert "Verified absence unavailable" in result


def test_recall_quick_miss_uses_quick_threshold_from_lookup_call() -> None:
    index = _MissIndex(
        _QuickMissDiagnostics(
            sessions_searched=1,
            oldest_session_started_at="2026-06-12",
            chunks_scanned=5,
            best_score=0.04,
            threshold=0.20,
        )
    )

    with patch("synapt.recall.server._get_index", return_value=index):
        recall_quick("needle")

    assert index.lookup_kwargs is not None
    assert index.lookup_kwargs["threshold_ratio"] == 0.2
