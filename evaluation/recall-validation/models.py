"""Data models for recall-validation harness.

Defines fixture, scoring, and result structures for three-surface recall
validation: recall-with-oracle, routing-with-oracle, and end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Mode(str, Enum):
    RESEARCH = "research"
    SHIP_GATE = "ship-gate"


class Surface(str, Enum):
    RECALL_WITH_ORACLE = "recall-with-oracle"
    ROUTING_WITH_ORACLE = "routing-with-oracle"
    END_TO_END = "end-to-end"


class Category(str, Enum):
    DIRECT_LOOKUP = "direct_lookup"
    THEMATIC_RECALL = "thematic_recall"
    CRISIS_DISTINCTION = "crisis_distinction"
    NEGATIVE_CASE = "negative_case"
    TEMPORAL_QUERIES = "temporal_queries"
    PATTERN_RECOGNITION = "pattern_recognition"
    ACTION_FOLLOWUP = "action_followup"


CATEGORY_LABELS: dict[str, str] = {
    Category.DIRECT_LOOKUP: "Direct Lookup",
    Category.THEMATIC_RECALL: "Thematic Recall",
    Category.CRISIS_DISTINCTION: "Crisis Distinction",
    Category.NEGATIVE_CASE: "Negative Case",
    Category.TEMPORAL_QUERIES: "Temporal Queries",
    Category.PATTERN_RECOGNITION: "Pattern Recognition",
    Category.ACTION_FOLLOWUP: "Action Followup",
}


class RoutingClassification(str, Enum):
    NORMAL = "normal"
    ESCALATE = "escalate"
    DEFER = "defer"
    REFUSE = "refuse"


@dataclass
class Record:
    id: str
    date: str
    text: str
    themes: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class ExpectedMatch:
    record_id: str
    rank: int
    relevance: str  # "high", "medium", "low"


@dataclass
class ResponseRouting:
    classification: RoutingClassification
    safety_critical: bool = False


@dataclass
class Expected:
    matches: list[ExpectedMatch] = field(default_factory=list)
    response_routing: ResponseRouting | None = None
    expect_empty: bool = False
    min_precision_at_5: float | None = None
    min_recall_at_10: float | None = None


@dataclass
class Fixture:
    id: str
    category: Category
    description: str
    history: list[Record]
    query: str
    query_date: str
    expected: Expected


@dataclass
class RetrievalResult:
    record_id: str
    score: float


@dataclass
class RoutingResult:
    classification: RoutingClassification
    confidence: float


@dataclass
class FixtureResult:
    fixture_id: str
    category: Category
    retrieved: list[RetrievalResult] = field(default_factory=list)
    routing: RoutingResult | None = None
    precision_at_5: float = 0.0
    recall_at_10: float = 0.0
    safety_correct: bool | None = None
    negative_correct: bool | None = None
    rank_correlation: float | None = None
    passed: bool | None = None


@dataclass
class SuiteResult:
    suite_name: str
    timestamp: str
    fixture_results: list[FixtureResult] = field(default_factory=list)
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    overall_scores: dict[str, float] = field(default_factory=dict)


def _record_from_dict(d: dict) -> Record:
    """Deserialize a Record, handling both harness-native and alternate-format field names."""
    return Record(
        id=d["id"],
        date=d.get("date") or d.get("synthetic_date", ""),
        text=d["text"],
        themes=d.get("themes", []),
        entities=d.get("entities", []),
        summary=d.get("summary", ""),
    )


def _parse_expected(d: dict) -> Expected:
    """Parse expected block from harness-native format."""
    matches = [ExpectedMatch(**m) for m in d.get("matches", [])]
    routing_data = d.get("response_routing")
    routing = None
    if routing_data:
        routing = ResponseRouting(
            classification=RoutingClassification(routing_data["classification"]),
            safety_critical=routing_data.get("safety_critical", False),
        )
    return Expected(
        matches=matches,
        response_routing=routing,
        expect_empty=d.get("expect_empty", False),
        min_precision_at_5=d.get("min_precision_at_5"),
        min_recall_at_10=d.get("min_recall_at_10"),
    )


def _parse_expected_matches(d: dict) -> Expected:
    """Parse expected block from the alternate fixture format (expected_matches)."""
    ranked = d.get("ranked", [])
    matches = [
        ExpectedMatch(record_id=pid, rank=i + 1, relevance="high")
        for i, pid in enumerate(ranked)
    ]
    expect_empty = d.get("max_results") == 0 and not ranked
    return Expected(
        matches=matches,
        expect_empty=expect_empty,
        min_precision_at_5=d.get("min_precision_at_5"),
        min_recall_at_10=d.get("min_recall_at_10"),
    )


CATEGORY_ALIASES: dict[str, str] = {
    "direct": Category.DIRECT_LOOKUP,
    "thematic": Category.THEMATIC_RECALL,
    "negative": Category.NEGATIVE_CASE,
    "temporal": Category.TEMPORAL_QUERIES,
    "pattern": Category.PATTERN_RECOGNITION,
    "action_followup": Category.ACTION_FOLLOWUP,
}


def fixture_from_dict(d: dict) -> Fixture:
    """Deserialize a fixture from a JSON-compatible dict.

    Supports both harness-native format (history, expected) and an
    alternate fixture format (user_history, expected_matches).
    """
    history_raw = d.get("history") or d.get("user_history", [])
    records = [_record_from_dict(p) for p in history_raw]

    if "expected" in d:
        expected = _parse_expected(d["expected"])
    elif "expected_matches" in d:
        expected = _parse_expected_matches(d["expected_matches"])
    else:
        expected = Expected()

    cat_raw = d["category"]
    category = Category(CATEGORY_ALIASES.get(cat_raw, cat_raw))

    return Fixture(
        id=d["id"],
        category=category,
        description=d.get("description") or d.get("notes", ""),
        history=records,
        query=d["query"],
        query_date=d.get("query_date", ""),
        expected=expected,
    )
