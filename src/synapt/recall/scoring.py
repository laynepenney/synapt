"""ChunkScoringStrategy: pluggable scoring for chunks in consolidate + enrich.

A single `ChunkScoringStrategy` Protocol is shared across consolidate and
enrich. OSS provides a default `RecencyScoring` implementation; a
downstream layer may register additional strategies via the registry in
this module. Only one strategy is active at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# Default recent-token window for recency-based scoring.
DEFAULT_RECENT_TOKEN_WINDOW = 16


@dataclass(frozen=True)
class ScoringInput:
    """Generic chunk-like input for scoring strategies.

    Portable across consolidate (TranscriptChunk-derived) and enrich (transcript
    window-derived). Strategies receive a list of ScoringInputs ordered by
    temporal position (older to newer) and return ScoredChunks.

    `position` is the temporal index within the input list (0 = oldest,
    N-1 = newest). `metadata` is the extension point for strategy-specific
    context (alternative strategies may require per-chunk metadata such
    as attention weights or precomputed scores).
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

    OSS provides `RecencyScoring` as the default (temporal-position only).
    A downstream layer may register additional strategies via the registry
    seam. Only one strategy is active at any time.
    """

    @property
    def name(self) -> str:
        """Unique strategy identifier. OSS provides 'recency' as the default."""
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


# Module-level registry. Strategies register at import time; the active
# strategy is selected separately (not per-call).
_registered_strategies: dict[str, ChunkScoringStrategy] = {}
_active_strategy_name: str | None = None


def register_scoring_strategy(name: str, strategy: ChunkScoringStrategy) -> None:
    """Register a scoring strategy by name.

    The registry is process-global; subsequent `activate_scoring_strategy`
    selects which registered strategy is active.

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

    Subsequent calls to `get_active_strategy()` return the strategy
    registered under `name`.

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


class ScoreContractViolation(TypeError):
    """Raised when a strategy's `score()` return value violates the contract.

    The Protocol provides static shape only; this runtime check validates
    the return value so a non-conforming strategy fails closed instead of
    silently corrupting downstream callers.
    """


def _validate_score_result(
    inputs: list[ScoringInput],
    result: object,
    strategy_name: str,
) -> list[ScoredChunk]:
    """Validate a strategy's score() return value against the runtime contract.

    The contract:
    - Return value is a list
    - Length equals the input length
    - Each item is a ScoredChunk
    - Each item.input is the corresponding inputs[i] (identity preserved)
    - Each item.score is numeric (int or float; not bool)
    - Each item.strategy_name equals the strategy's self-reported name

    Raises:
        ScoreContractViolation: any of the above conditions fail. The error
            includes which item / which check failed for diagnostic clarity.
    """
    if not isinstance(result, list):
        raise ScoreContractViolation(
            f"strategy {strategy_name!r} returned "
            f"{type(result).__name__}, expected list"
        )
    if len(result) != len(inputs):
        raise ScoreContractViolation(
            f"strategy {strategy_name!r} returned {len(result)} items, "
            f"expected {len(inputs)} (length must match inputs)"
        )
    for i, (item, expected_input) in enumerate(zip(result, inputs)):
        if not isinstance(item, ScoredChunk):
            raise ScoreContractViolation(
                f"strategy {strategy_name!r} item[{i}] is "
                f"{type(item).__name__}, expected ScoredChunk"
            )
        # bool is a subclass of int in Python; reject explicitly so a
        # strategy returning True/False as score does not silently pass.
        if isinstance(item.score, bool) or not isinstance(item.score, (int, float)):
            raise ScoreContractViolation(
                f"strategy {strategy_name!r} item[{i}].score is "
                f"{type(item.score).__name__}, expected numeric (int or float)"
            )
        if item.input is not expected_input:
            raise ScoreContractViolation(
                f"strategy {strategy_name!r} item[{i}].input does not preserve "
                f"input identity (must be the corresponding inputs[i])"
            )
        if item.strategy_name != strategy_name:
            raise ScoreContractViolation(
                f"strategy {strategy_name!r} item[{i}].strategy_name="
                f"{item.strategy_name!r}; must equal strategy.name="
                f"{strategy_name!r}"
            )
    return result


def score_with_validation(
    strategy: ChunkScoringStrategy,
    inputs: list[ScoringInput],
) -> list[ScoredChunk]:
    """Score `inputs` via `strategy` and validate the return contract.

    This is the canonical integration boundary for both `consolidate` and
    `enrich` to invoke scoring. Empty inputs short-circuit to an empty
    result without invoking the strategy (consistent with the strategies'
    own empty-input handling).

    Raises:
        ScoreContractViolation: strategy return value violates the contract;
            see `_validate_score_result` for the specific check that failed.
    """
    if not inputs:
        return []
    result = strategy.score(inputs)
    return _validate_score_result(inputs, result, strategy.name)


def reset_registry() -> None:
    """Test-only: reset registry state. Not part of the public API.

    Used by test fixtures to isolate strategy-registry state across tests.
    """
    global _active_strategy_name
    _registered_strategies.clear()
    _active_strategy_name = None
