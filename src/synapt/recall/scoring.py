"""ChunkScoringStrategy: pluggable scoring for chunks in consolidate + enrich.

Per config#339 Pattern 4 ratification (Q1-Q4) + config#332 consolidation-primary
locus:

- **Protocol shape shared** across consolidate + enrich (Q2 ratified) — single
  `ChunkScoringStrategy` Protocol; OSS Reference vs Premium Implementation rule
  applies (OSS knows Protocol shape + provides default; premium implements
  differentiated strategies without OSS naming them).
- **Default RecencyScoring in OSS**; premium VornScoring via registry (Q1
  Pattern 4 strategy-resolver).
- **Plugin-time activation** (Q3 ratified): premium plugin module-level code
  calls `register_scoring_strategy` + `activate_scoring_strategy` at import; no
  per-call override (would re-introduce search-side API surface rejected per
  config#332 frame shift).
- **Single-active-strategy by construction** (Pattern 4 caveat; revisit Pattern 3
  hooks if multi-strategy composition becomes a need).
- **window=16 default** per Layne directive 2026-06-07: keep the
  empirically-anchored choice (RC4 calibration smoke anchor) until window=8 is
  directly tested. Atlas research#7 (2026-06-07) AXIS 3 sweep tested windows
  4/16/32/64/128 and found NO inflection — identical SEMU selections across all
  windows on this fixture (vorn-dilution thesis prediction did NOT replicate).
  config#339 originally recommended window=8 on theoretical
  "conservative-dilution-resilient zone" rationale; Layne corrected: don't move
  off the empirical anchor without empirical reason. window=16 stays as the
  locked default; window=8 would require direct empirical test (~$0.06 single
  Modal cell) before adoption. Selector-sensitive fixture lane (Atlas surfaced)
  is the bigger lever for any future window-sensitivity work.

Move 2 contract redesign: this module IS the redesigned contract for
config#330's rejected compression-as-search-param-with-strategy-enum frame.
Locus is consolidation + enrichment (substrate-reshape sites), not search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# Default recent-token window for recency-based scoring.
# Anchored at the existing RC4 calibration smoke baseline (window=16).
# Atlas research#7 (2026-06-07) AXIS 3 sweep found no inflection across
# windows 4/16/32/64/128 on the calibration fixture. Per Layne directive
# 2026-06-07 ("keep 16 until we test 8"), don't move off the empirical
# anchor without direct empirical reason. See module docstring.
DEFAULT_RECENT_TOKEN_WINDOW = 16


@dataclass(frozen=True)
class ScoringInput:
    """Generic chunk-like input for scoring strategies.

    Portable across consolidate (TranscriptChunk-derived) and enrich (transcript
    window-derived). Strategies receive a list of ScoringInputs ordered by
    temporal position (older to newer) and return ScoredChunks.

    `position` is the temporal index within the input list (0 = oldest,
    N-1 = newest). `metadata` is the extension point for strategy-specific
    context (e.g. vorn-attention requires per-chunk attention weights).
    """

    content: str
    position: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    """Result of strategy.score() — input + score + strategy attribution."""

    input: ScoringInput
    score: float
    strategy_name: str


@runtime_checkable
class ChunkScoringStrategy(Protocol):
    """Pluggable scoring strategy for chunks during consolidation + enrichment.

    Implementations:
    - OSS: `RecencyScoring` (default; temporal-position only)
    - Premium: `VornScoring` (vorn-attention-based; registered via plugin)
    - Premium: `AutoScoring` (adaptive selection; gated on AXIS 3 fixture lane)

    Per Pattern 4 single-active-strategy constraint: only ONE strategy is active
    at any time. Multi-strategy composition (Pattern 3 hooks) is a future
    consideration if needed.
    """

    @property
    def name(self) -> str:
        """Unique strategy identifier (e.g., 'recency', 'vorn', 'auto')."""
        ...

    @property
    def window(self) -> int:
        """Recent-token window for temporal/contextual scope of scoring."""
        ...

    def score(self, inputs: list[ScoringInput]) -> list[ScoredChunk]:
        """Score a list of ScoringInputs.

        Implementations should:
        - Return ScoredChunks in input order (preserve indices)
        - Set `strategy_name` to `self.name` on each ScoredChunk
        - Honor `window` for any windowed-context calculations
        - Be deterministic for the same input (testability)
        """
        ...


class RecencyScoring:
    """Default OSS scoring strategy: temporal-position-only recency.

    Score formula: linear ramp from 0.0 (oldest in window) to 1.0 (newest)
    within the recent window; chunks outside the window receive 0.0.

    This is the substrate-coherent OSS default per the OSS Reference vs Premium
    Implementation rule (2026-05-04): OSS owns the Protocol + a real, usable
    default; premium provides the differentiated implementation (VornScoring).
    """

    name = "recency"

    def __init__(self, window: int = DEFAULT_RECENT_TOKEN_WINDOW):
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window}")
        self._window = window

    @property
    def window(self) -> int:
        return self._window

    def score(self, inputs: list[ScoringInput]) -> list[ScoredChunk]:
        if not inputs:
            return []

        n = len(inputs)
        window_start = max(0, n - self._window)

        scored: list[ScoredChunk] = []
        for input_ in inputs:
            position = input_.position
            if position < window_start:
                score = 0.0
            else:
                in_window_pos = position - window_start
                window_size = n - window_start
                if window_size <= 1:
                    score = 1.0
                else:
                    score = in_window_pos / (window_size - 1)

            scored.append(
                ScoredChunk(
                    input=input_,
                    score=score,
                    strategy_name=self.name,
                )
            )

        return scored


# Module-level registry. Strategies register at plugin-import time; activated
# strategy is selected at plugin-time (Q3 ratified — not per-call).
_registered_strategies: dict[str, ChunkScoringStrategy] = {}
_active_strategy_name: str | None = None


def register_scoring_strategy(name: str, strategy: ChunkScoringStrategy) -> None:
    """Register a scoring strategy by name.

    Premium plugins call this at module-import time to make their strategies
    available. The registry is process-global; subsequent
    `activate_scoring_strategy` selects which registered strategy is active.

    Raises:
        ValueError: name is empty/non-string or already registered.
        TypeError: strategy does not implement the Protocol.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"strategy name must be non-empty string; got {name!r}")
    if name in _registered_strategies:
        raise ValueError(f"strategy {name!r} already registered")
    if not isinstance(strategy, ChunkScoringStrategy):
        raise TypeError(
            f"strategy {name!r} does not implement ChunkScoringStrategy Protocol"
        )
    _registered_strategies[name] = strategy


def activate_scoring_strategy(name: str) -> None:
    """Activate a registered strategy by name.

    Per Pattern 4 single-active-strategy constraint: subsequent calls to
    `get_active_strategy()` will return the strategy registered under `name`.

    Raises:
        KeyError: name is not registered.
    """
    if name not in _registered_strategies:
        raise KeyError(
            f"strategy {name!r} not registered; "
            f"available: {sorted(_registered_strategies)}"
        )
    global _active_strategy_name
    _active_strategy_name = name


def get_active_strategy() -> ChunkScoringStrategy:
    """Return the currently active strategy.

    Falls back to a default `RecencyScoring` if no strategy is explicitly
    activated. This preserves OSS-only operability without plugin activation.

    Note: the fallback returns a fresh `RecencyScoring()` each call to avoid
    mutable-state leakage; hot-path integration sites should cache the active
    strategy at the call boundary.
    """
    if _active_strategy_name is not None:
        return _registered_strategies[_active_strategy_name]
    return RecencyScoring()


def reset_registry() -> None:
    """Test-only: reset registry state. Not part of the public API.

    Used by test fixtures to isolate strategy-registry state across tests.
    """
    global _active_strategy_name
    _registered_strategies.clear()
    _active_strategy_name = None
