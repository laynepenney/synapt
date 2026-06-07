"""TDD tests for synapt.recall.scoring (PR4f-A Phase 1).

Per config#339 Pattern 4 ratification:
- ChunkScoringStrategy Protocol contract
- Default RecencyScoring behavior
- Module-level registry: register_scoring_strategy + activate_scoring_strategy
- DI-threading-seam reject/accept E2E pairs (per
  feedback_xfail_removal_doc_sweep.md DI-threading-seam rule)
- window=16 default per Layne directive 2026-06-07 (keep empirical anchor until
  window=8 directly tested; Atlas research#7 tested 4/16/32/64/128 — no inflection)

Phase 2 (consolidate.py + enrich.py integration via get_active_strategy) covered
by integration tests in a follow-on commit.
"""

from __future__ import annotations

import pytest

from synapt.recall.scoring import (
    DEFAULT_RECENT_TOKEN_WINDOW,
    ChunkScoringStrategy,
    RecencyScoring,
    ScoredChunk,
    ScoringInput,
    activate_scoring_strategy,
    get_active_strategy,
    register_scoring_strategy,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test starts with a clean registry."""
    reset_registry()
    yield
    reset_registry()


# === DEFAULT_RECENT_TOKEN_WINDOW ===


class TestDefaultRecentTokenWindow:
    """Per Layne directive 2026-06-07: window=16 stays as locked default until
    window=8 directly tested. RC4 calibration smoke anchored at window=16."""

    def test_default_is_sixteen(self):
        assert DEFAULT_RECENT_TOKEN_WINDOW == 16


# === RecencyScoring Protocol contract ===


class TestRecencyScoringContract:

    def test_implements_protocol(self):
        assert isinstance(RecencyScoring(), ChunkScoringStrategy)

    def test_name_is_recency(self):
        assert RecencyScoring().name == "recency"

    def test_default_window_is_sixteen(self):
        assert RecencyScoring().window == DEFAULT_RECENT_TOKEN_WINDOW
        assert RecencyScoring().window == 16

    def test_custom_window_honored(self):
        assert RecencyScoring(window=16).window == 16

    def test_zero_window_rejected(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            RecencyScoring(window=0)

    def test_negative_window_rejected(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            RecencyScoring(window=-1)


# === RecencyScoring behavior ===


class TestRecencyScoringBehavior:

    def test_empty_inputs_returns_empty(self):
        result = RecencyScoring().score([])
        assert result == []

    def test_single_input_max_score(self):
        chunk = ScoringInput(content="lone", position=0)
        scored = RecencyScoring().score([chunk])
        assert len(scored) == 1
        assert scored[0].score == 1.0
        assert scored[0].strategy_name == "recency"
        assert scored[0].input is chunk

    def test_recency_ramp_all_in_window(self):
        """Newest chunk gets highest score; oldest in window gets lowest."""
        inputs = [ScoringInput(content=f"c{i}", position=i) for i in range(4)]
        scored = RecencyScoring(window=4).score(inputs)
        scores = [s.score for s in scored]
        assert scores[0] == 0.0
        assert scores[-1] == 1.0
        assert scores == sorted(scores), "scores must be monotonic in position"

    def test_chunks_outside_window_zero(self):
        """When more inputs than window, oldest chunks score 0.0."""
        inputs = [ScoringInput(content=f"c{i}", position=i) for i in range(10)]
        scored = RecencyScoring(window=3).score(inputs)
        for s in scored[:7]:
            assert s.score == 0.0
        assert scored[7].score == 0.0
        assert scored[8].score == 0.5
        assert scored[9].score == 1.0

    def test_strategy_name_on_each_scored_chunk(self):
        inputs = [
            ScoringInput(content="a", position=0),
            ScoringInput(content="b", position=1),
        ]
        scored = RecencyScoring().score(inputs)
        for s in scored:
            assert s.strategy_name == "recency"

    def test_deterministic(self):
        """Same input → same output (testability property)."""
        inputs = [ScoringInput(content=f"c{i}", position=i) for i in range(5)]
        scoring = RecencyScoring(window=3)
        first = scoring.score(inputs)
        second = scoring.score(inputs)
        assert first == second


# === Registry: register_scoring_strategy ===


class TestRegisterScoringStrategy:

    def test_register_valid_strategy(self):
        register_scoring_strategy("recency", RecencyScoring())

    def test_register_empty_name_rejected(self):
        with pytest.raises(ValueError, match="strategy name must be non-empty"):
            register_scoring_strategy("", RecencyScoring())

    def test_register_non_string_name_rejected(self):
        with pytest.raises(ValueError, match="strategy name must be non-empty"):
            register_scoring_strategy(None, RecencyScoring())  # type: ignore[arg-type]

    def test_register_duplicate_rejected(self):
        register_scoring_strategy("recency", RecencyScoring())
        with pytest.raises(ValueError, match="already registered"):
            register_scoring_strategy("recency", RecencyScoring())

    def test_register_non_protocol_rejected(self):
        class NotAStrategy:
            pass

        with pytest.raises(TypeError, match="does not implement"):
            register_scoring_strategy("fake", NotAStrategy())  # type: ignore[arg-type]


# === Registry: activate_scoring_strategy ===


class TestActivateScoringStrategy:

    def test_activate_registered(self):
        strategy = RecencyScoring(window=16)
        register_scoring_strategy("recency-wide", strategy)
        activate_scoring_strategy("recency-wide")
        active = get_active_strategy()
        assert active is strategy

    def test_activate_unregistered_rejected(self):
        with pytest.raises(KeyError, match="not registered"):
            activate_scoring_strategy("nonexistent")

    def test_activate_switches_active(self):
        s1 = RecencyScoring(window=4)
        s2 = RecencyScoring(window=16)
        register_scoring_strategy("narrow", s1)
        register_scoring_strategy("wide", s2)
        activate_scoring_strategy("narrow")
        assert get_active_strategy() is s1
        activate_scoring_strategy("wide")
        assert get_active_strategy() is s2


# === get_active_strategy fallback ===


class TestGetActiveStrategyFallback:

    def test_default_when_none_activated(self):
        """OSS-only operability: get_active_strategy returns default RecencyScoring
        even without plugin activation."""
        active = get_active_strategy()
        assert isinstance(active, RecencyScoring)
        assert active.name == "recency"
        assert active.window == DEFAULT_RECENT_TOKEN_WINDOW


# === DI-threading-seam tests (per feedback_xfail_removal_doc_sweep.md) ===


class TestDIThreadingSeamRejectAccept:
    """Reject + accept pairs for the register + activate seam.

    Per DI-threading-seam rule: the substrate-fix for adding a registry/activation
    seam needs paired reject/accept E2E threading tests. This class covers the
    seam contract; integration tests at consolidate.py + enrich.py callsites
    follow in Phase 2 of PR4f-A.
    """

    def test_reject_register_invalid_then_recover(self):
        """Reject path: invalid registration does not pollute registry."""
        with pytest.raises(ValueError):
            register_scoring_strategy("", RecencyScoring())
        with pytest.raises(KeyError):
            activate_scoring_strategy("recency")
        register_scoring_strategy("recency", RecencyScoring())
        activate_scoring_strategy("recency")
        assert get_active_strategy().name == "recency"

    def test_reject_activate_invalid_does_not_leak_state(self):
        """Reject path: failed activation does not corrupt prior active state."""
        register_scoring_strategy("recency", RecencyScoring(window=4))
        activate_scoring_strategy("recency")
        prior = get_active_strategy()

        with pytest.raises(KeyError):
            activate_scoring_strategy("nonexistent")

        assert get_active_strategy() is prior

    def test_accept_register_then_activate_then_score(self):
        """Accept path: full register → activate → use-via-get_active threading."""
        register_scoring_strategy("recency", RecencyScoring(window=2))
        activate_scoring_strategy("recency")
        active = get_active_strategy()
        inputs = [ScoringInput(content=f"c{i}", position=i) for i in range(3)]
        result = active.score(inputs)
        assert len(result) == 3
        assert all(s.strategy_name == "recency" for s in result)

    def test_isolate_registry_resets_state(self):
        """Reset hygiene: after reset, no strategies remain registered."""
        register_scoring_strategy("recency", RecencyScoring())
        activate_scoring_strategy("recency")
        reset_registry()
        with pytest.raises(KeyError, match="not registered"):
            activate_scoring_strategy("recency")
        active = get_active_strategy()
        assert isinstance(active, RecencyScoring)
