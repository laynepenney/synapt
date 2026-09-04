"""Tests for the session-scoped working memory."""

import time

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.working_memory import WorkingMemory, WorkingMemorySlot, MAX_SLOTS


class TestWorkingMemory:
    """Unit tests for WorkingMemory."""

    def test_record_and_query(self):
        """Record an item, query with overlapping tokens."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "Fixed flock TOCTOU race condition in locking")

        results = wm.query({"flock", "race", "condition"})
        assert len(results) == 1
        assert results[0].item_id == "s1:t0"
        assert results[0].access_count == 1

    def test_query_requires_two_token_overlap(self):
        """Query needs 2+ shared tokens to match."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "Fixed flock TOCTOU race condition")

        # Only 1 token overlap → no match
        results = wm.query({"flock", "unrelated"})
        assert len(results) == 0

        # 2 tokens → match
        results = wm.query({"flock", "race"})
        assert len(results) == 1

    def test_repeated_access_increments_count(self):
        """Recording the same item again increments access_count."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "flock race fix")
        wm.record("chunk", "s1:t0", "flock race fix")
        wm.record("chunk", "s1:t0", "flock race fix")

        assert "s1:t0" in wm
        results = wm.query({"flock", "race"})
        assert results[0].access_count == 3

    def test_lru_eviction(self):
        """When at capacity, evicts least-recently-accessed slot."""
        wm = WorkingMemory()

        # Fill to capacity
        for i in range(MAX_SLOTS):
            wm.record("chunk", f"s1:t{i}", f"topic{i} content{i} extra{i}")

        assert len(wm) == MAX_SLOTS

        # Access the first item to make it recent
        wm.record("chunk", "s1:t0", "topic0 content0 extra0")

        # Add one more — should evict s1:t1 (LRU, not s1:t0 which was just accessed)
        wm.record("chunk", "s1:t99", "new topic99 content99 extra99")

        assert len(wm) == MAX_SLOTS
        assert "s1:t0" in wm   # kept (recently accessed)
        assert "s1:t99" in wm  # just added
        assert "s1:t1" not in wm  # evicted (LRU)

    def test_boost_score_not_in_memory(self):
        """Items not in working memory get no boost."""
        wm = WorkingMemory()
        assert wm.boost_score(5.0, "unknown") == 5.0

    def test_boost_score_single_pull_not_boosted(self):
        """A single pull is not evidence of use. record() fires on mere
        emission, so one prior pull must not earn a boost -- only being
        returned repeatedly does."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "something")

        assert wm.boost_score(4.0, "s1:t0") == 4.0  # unboosted after 1 pull

    def test_boost_score_second_pull_is_1_5x(self):
        """Control: an item genuinely pulled more than once still boosts --
        this fix narrows *when* a boost applies, it does not remove boosting."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "something")
        wm.record("chunk", "s1:t0", "something")

        assert wm.boost_score(4.0, "s1:t0") == 6.0  # 4.0 * 1.5

    def test_boost_score_frequent(self):
        """Items accessed 3+ times get 2.0x boost."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "something")
        wm.record("chunk", "s1:t0", "something")
        wm.record("chunk", "s1:t0", "something")

        assert wm.boost_score(4.0, "s1:t0") == 8.0  # 4.0 * 2.0

    def test_empty_query(self):
        """Empty query tokens return nothing."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "some content here")
        assert wm.query(set()) == []

    def test_query_ranking(self):
        """Results ranked by overlap * log(access_count)."""
        wm = WorkingMemory()
        wm.record("chunk", "s1:t0", "swift adapter training eval")
        wm.record("cluster", "clust-abc", "swift adapter patcher critic training")
        # Access cluster more to boost its score
        wm.record("cluster", "clust-abc", "swift adapter patcher critic training")

        results = wm.query({"swift", "adapter", "training"}, max_results=2)
        assert len(results) == 2
        # Cluster should rank higher (same overlap but higher access_count)
        assert results[0].item_id == "clust-abc"

    def test_len_and_contains(self):
        """__len__ and __contains__ work correctly."""
        wm = WorkingMemory()
        assert len(wm) == 0
        assert "s1:t0" not in wm

        wm.record("chunk", "s1:t0", "content")
        assert len(wm) == 1
        assert "s1:t0" in wm


def _chunk(id: str, user: str = "", assistant: str = "") -> TranscriptChunk:
    return TranscriptChunk(
        id=id,
        session_id="sess-1",
        timestamp="2026-03-20T09:00:00Z",
        turn_index=0,
        user_text=user,
        assistant_text=assistant,
    )


class TestWorkingMemoryResultFormatting:
    def test_format_results_does_not_tag_single_pull_as_boosted(self):
        """Reproduction: record() fires on mere emission, so a
        single prior pull of sess-1:t1 (never revisited, never acted on)
        must not carry a working-memory boost tag into the next query's
        results -- being surfaced once is not evidence it was used."""
        idx = TranscriptIndex([
            _chunk("sess-1:t0", user="Show the strongest result", assistant="Top unboosted result."),
            _chunk("sess-1:t1", user="What drink do I prefer?", assistant="You prefer tea."),
        ])
        idx._working_memory.record("chunk", "sess-1:t1", "You prefer tea.")

        result = idx._format_results([(0, 10.0), (1, 7.0)], max_tokens=2000)

        assert "boosted: working-memory" not in result

    def test_format_results_tags_working_memory_boost_after_second_pull(self):
        """Control: a row genuinely pulled a second time still gets boosted
        and labeled -- the fix narrows the threshold, it does not remove
        boosting for items that keep recurring."""
        idx = TranscriptIndex([
            _chunk("sess-1:t0", user="Show the strongest result", assistant="Top unboosted result."),
            _chunk("sess-1:t1", user="What drink do I prefer?", assistant="You prefer tea."),
        ])
        idx._working_memory.record("chunk", "sess-1:t1", "You prefer tea.")
        idx._working_memory.record("chunk", "sess-1:t1", "You prefer tea.")

        result = idx._format_results([(0, 10.0), (1, 7.0)], max_tokens=2000)

        assert "[boosted: working-memory 1.5x]" in result

    def test_format_results_tags_frequent_working_memory_boost(self):
        idx = TranscriptIndex([
            _chunk("sess-1:t0", user="Show the strongest result", assistant="Top unboosted result."),
            _chunk("sess-1:t1", user="What drink do I prefer?", assistant="You prefer tea."),
        ])
        idx._working_memory.record("chunk", "sess-1:t1", "You prefer tea.")
        idx._working_memory.record("chunk", "sess-1:t1", "You prefer tea.")
        idx._working_memory.record("chunk", "sess-1:t1", "You prefer tea.")

        result = idx._format_results([(0, 10.0), (1, 7.0)], max_tokens=2000)

        assert "[boosted: working-memory 2.0x]" in result
