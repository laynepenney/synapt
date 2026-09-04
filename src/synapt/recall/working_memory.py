"""Working memory for recall search — with cross-session persistence.

A small in-memory LRU buffer of recently accessed items. Items in
working memory get a relevance boost in search results — mimicking how
human working memory keeps recently-used context readily available.

On startup, seeds from the access_log DB table so frequently accessed
topics from recent sessions carry forward with natural decay.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Tokenizer (same as bm25._tokenize for consistency)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> set[str]:
    """Extract lowercase token set from text."""
    return {m.group().lower() for m in _TOKEN_RE.finditer(text) if len(m.group()) > 1}


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------

MAX_SLOTS = 10  # cognitive science: 7 ± 2, rounded up


@dataclass
class WorkingMemorySlot:
    """A single item in working memory."""

    item_type: str  # "chunk" | "cluster" | "knowledge"
    item_id: str
    content_preview: str  # first 200 chars for display
    tokens: set[str] = field(default_factory=set)
    last_accessed: float = 0.0  # time.monotonic()
    access_count: int = 0


class WorkingMemory:
    """LRU buffer of recently accessed recall items.

    Capacity: MAX_SLOTS (10). Eviction: least-recently-accessed when full.
    """

    def __init__(self) -> None:
        self._slots: dict[str, WorkingMemorySlot] = {}  # keyed by item_id
        self._access_seq: int = 0  # Monotonic counter for LRU ordering

    def record(self, item_type: str, item_id: str, content: str) -> None:
        """Record that an item was returned in search results or drilled into."""
        self._access_seq += 1
        now = self._access_seq  # Use counter, not clock (Windows timer ~15ms)
        key = item_id
        if key in self._slots:
            slot = self._slots[key]
            slot.last_accessed = now
            slot.access_count += 1
        else:
            # Evict LRU if at capacity
            if len(self._slots) >= MAX_SLOTS:
                oldest_key = min(self._slots, key=lambda k: self._slots[k].last_accessed)
                del self._slots[oldest_key]
            self._slots[key] = WorkingMemorySlot(
                item_type=item_type,
                item_id=item_id,
                content_preview=content[:200],
                tokens=_tokenize(content),
                last_accessed=now,
                access_count=1,
            )

    def query(self, query_tokens: set[str], max_results: int = 3) -> list[WorkingMemorySlot]:
        """Find working memory slots relevant to a query.

        Uses token overlap — slots sharing 2+ tokens with the query are
        returned, ranked by overlap * (1 + log(access_count)).
        """
        if not query_tokens or not self._slots:
            return []

        scored: list[tuple[float, WorkingMemorySlot]] = []
        for slot in self._slots.values():
            overlap = len(query_tokens & slot.tokens)
            if overlap >= 2:
                score = overlap * (1 + math.log(slot.access_count))
                scored.append((score, slot))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [slot for _, slot in scored[:max_results]]

    def boost_multiplier(self, item_id: str) -> float:
        """Return the working-memory multiplier for an item.

        access_count is incremented on every record() call, and record()
        fires whenever an item is merely returned in a result set -- not
        when the caller actually reads, cites, or otherwise uses it. A
        single pull is therefore not evidence of use, only of having been
        surfaced once; boosting it back into the next query's ranking on
        that basis alone is what let a stale, single-pull row outrank a
        genuinely on-topic result that had never come up before. Requiring
        a second pull before any boost applies is the cheapest available
        proxy for "this kept mattering" rather than "this was shown once."
        """
        slot = self._slots.get(item_id)
        if slot is None:
            return 1.0
        if slot.access_count >= 3:
            return 2.0
        if slot.access_count >= 2:
            return 1.5
        return 1.0

    def boost_score(self, base_score: float, item_id: str) -> float:
        """Apply working memory boost to a search result score."""
        return base_score * self.boost_multiplier(item_id)

    def clear(self) -> None:
        """Drop all in-memory slots without touching persisted access history."""
        self._slots.clear()
        self._access_seq = 0

    def seed_from_db(self, db, days: int = 7) -> int:
        """Seed working memory from recent access_log entries.

        Loads the most frequently accessed items from the last N days
        and pre-populates working memory slots. This gives cross-session
        persistence — topics you searched for yesterday still get a mild
        boost today, with natural decay via the LRU eviction.

        Args:
            db: RecallDB instance with access_log table.
            days: How many days of history to consider (default: 7).

        Returns:
            Number of slots seeded.
        """
        if db is None:
            return 0

        try:
            from datetime import datetime, timedelta, timezone
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            # Get top accessed items by frequency in the last N days
            rows = db._conn.execute(
                "SELECT item_type, item_id, COUNT(*) as cnt, "
                "  MAX(query) as last_query "
                "FROM access_log "
                "WHERE created_at > ? "
                "GROUP BY item_type, item_id "
                "ORDER BY cnt DESC "
                "LIMIT ?",
                (cutoff, MAX_SLOTS),
            ).fetchall()

            seeded = 0
            for row in rows:
                item_id = row["item_id"]
                if item_id in self._slots:
                    continue  # Already populated (e.g., from current session)
                self._access_seq += 1
                self._slots[item_id] = WorkingMemorySlot(
                    item_type=row["item_type"],
                    item_id=item_id,
                    content_preview=row["last_query"][:200] if row["last_query"] else "",
                    tokens=_tokenize(row["last_query"] or ""),
                    last_accessed=self._access_seq,
                    access_count=row["cnt"],
                )
                seeded += 1
            return seeded
        except Exception:
            return 0

    def __len__(self) -> int:
        return len(self._slots)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._slots
