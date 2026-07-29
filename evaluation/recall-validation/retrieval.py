"""Pluggable retrieval interface and placeholder implementation.

Production integration will implement RetrievalBackend against the
production retrieval API. The placeholder uses keyword overlap for
skeleton verification.
"""

from __future__ import annotations

import re
from typing import Protocol

from .models import (
    Fixture,
    Record,
    RetrievalResult,
    RoutingClassification,
    RoutingResult,
)


class RetrievalBackend(Protocol):
    def retrieve(self, query: str, history: list[Record], k: int) -> list[RetrievalResult]:
        ...

    def classify_routing(
        self,
        query: str,
        history: list[Record],
        retrieved_context: list[RetrievalResult] | None = None,
    ) -> RoutingResult:
        ...


_STOP_WORDS = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "and", "but", "or", "nor", "not", "no", "so", "if", "then",
    "that", "this", "what", "when", "where", "who", "how", "which",
    "for", "of", "to", "in", "on", "at", "by", "with", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under",
    "again", "further", "just", "also", "very", "too",
    "god", "lord", "please", "pray", "prayer", "prayed", "praying",
    "show", "tell", "give", "help", "want",
})

_CRISIS_SIGNALS = frozenset({
    "don't want to be here", "end it all", "can't go on",
    "not worth living", "better off without me", "kill myself",
    "want to die", "no reason to live", "suicide", "self-harm",
    "hurt myself", "give up on everything",
})

_TEMPORAL_WORDS = frozenset({
    "today", "yesterday", "tomorrow", "week", "month", "sunday", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday",
    "last", "this", "next", "recent", "recently", "ago",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
})


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOP_WORDS}


class KeywordOverlapRetrieval:
    """Placeholder retrieval using token overlap between query and records.

    Scores each record by Jaccard-like overlap of non-stopword tokens between
    the query and the record text + summary + entities. Sufficient for
    validating the harness end-to-end; not a real retrieval implementation.
    """

    def retrieve(
        self, query: str, history: list[Record], k: int = 10,
    ) -> list[RetrievalResult]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[str, float]] = []
        for record in history:
            record_tokens = _tokenize(record.text) | _tokenize(record.summary)
            for entity in record.entities:
                record_tokens |= _tokenize(entity)
            for theme in record.themes:
                record_tokens |= _tokenize(theme)

            overlap = query_tokens & record_tokens
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens | record_tokens)
            scored.append((record.id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [RetrievalResult(record_id=rid, score=s) for rid, s in scored[:k]]

    def classify_routing(
        self,
        query: str,
        history: list[Record],
        retrieved_context: list[RetrievalResult] | None = None,
    ) -> RoutingResult:
        query_lower = query.lower()

        for signal in _CRISIS_SIGNALS:
            if signal in query_lower:
                return RoutingResult(
                    classification=RoutingClassification.ESCALATE,
                    confidence=0.95,
                )

        query_tokens = _tokenize(query)
        if query_tokens & {"empty", "numb", "tired", "exhausted", "hopeless", "pointless"}:
            return RoutingResult(
                classification=RoutingClassification.DEFER,
                confidence=0.6,
            )

        return RoutingResult(
            classification=RoutingClassification.NORMAL,
            confidence=0.8,
        )
