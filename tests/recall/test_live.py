"""Tests for synapt.recall.live — live transcript search."""

from __future__ import annotations

import json
import math
import tempfile
import threading
import unittest

from _isolation_helpers import owned_store
from pathlib import Path
from unittest.mock import MagicMock, patch

from synapt.recall.journal import compact_journal, extract_session_id
from synapt.recall.live import (
    _LiveCache,
    _format_live_results,
    _get_live_chunks,
    _score_chunks,
    search_live_transcript,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_transcript(path: Path, session_id: str, turns: list[dict]) -> None:
    """Write a minimal Claude Code transcript JSONL to *path*."""
    with open(path, "w", encoding="utf-8") as f:
        # progress entry (session_id source)
        f.write(json.dumps({
            "type": "progress",
            "sessionId": session_id,
            "timestamp": "2026-03-01T10:00:00.000Z",
        }) + "\n")
        # turn entries
        for i, turn in enumerate(turns):
            f.write(json.dumps({
                "type": "user",
                "message": {"role": "user", "content": turn.get("user", "")},
                "sessionId": session_id,
                "uuid": f"user-{i}",
                "timestamp": f"2026-03-01T10:0{i}:00.000Z",
            }) + "\n")
            f.write(json.dumps({
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": turn.get("assistant", "")},
                    ],
                },
                "sessionId": session_id,
                "uuid": f"asst-{i}",
                "timestamp": f"2026-03-01T10:0{i}:30.000Z",
            }) + "\n")


def _make_chunk(text: str, turn_index: int = 0):
    """Return a minimal TranscriptChunk-like object for scoring tests."""
    from synapt.recall.core import TranscriptChunk
    return TranscriptChunk(
        id=f"test:t{turn_index}",
        session_id="test-session",
        timestamp="2026-03-01T10:00:00",
        turn_index=turn_index,
        user_text=text,
        assistant_text="",
        text=text,
    )


# ---------------------------------------------------------------------------
# extract_session_id
# ---------------------------------------------------------------------------

class TestExtractSessionId(unittest.TestCase):

    def test_extracts_from_progress_entry(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session.jsonl"
            _write_transcript(p, "abc123def456", [])
            self.assertEqual(extract_session_id(p), "abc123def456")

    def test_returns_empty_when_no_progress(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session.jsonl"
            p.write_text(json.dumps({"type": "other", "data": 1}) + "\n")
            self.assertEqual(extract_session_id(p), "")

    def test_returns_empty_on_missing_file(self):
        self.assertEqual(extract_session_id("/nonexistent/path.jsonl"), "")

    def test_handles_corrupt_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session.jsonl"
            p.write_text("not json\n" + json.dumps({
                "type": "progress", "sessionId": "real-id"
            }) + "\n")
            self.assertEqual(extract_session_id(p), "real-id")

    def test_returns_empty_when_session_id_is_blank(self):
        """Progress entry with sessionId: "" is falsy — extract_session_id skips it."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session.jsonl"
            p.write_text(json.dumps({"type": "progress", "sessionId": ""}) + "\n")
            self.assertEqual(extract_session_id(p), "")


# ---------------------------------------------------------------------------
# _score_chunks
# ---------------------------------------------------------------------------

class TestScoreChunks(unittest.TestCase):

    def test_exact_match_scores_positive(self):
        chunks = [_make_chunk("the swift adapter training pipeline")]
        from synapt.recall.bm25 import _tokenize
        scored = _score_chunks(chunks, _tokenize("swift adapter"))
        self.assertEqual(len(scored), 1)
        self.assertGreater(scored[0][1], 0)

    def test_no_match_returns_empty(self):
        chunks = [_make_chunk("docker compose for postgres")]
        from synapt.recall.bm25 import _tokenize
        scored = _score_chunks(chunks, _tokenize("swift adapter training"))
        self.assertEqual(scored, [])

    def test_multi_term_match_ranks_higher_than_single(self):
        multi = _make_chunk("swift adapter training pipeline", turn_index=0)
        single = _make_chunk("swift docker build", turn_index=1)
        from synapt.recall.bm25 import _tokenize
        scored = _score_chunks([multi, single], _tokenize("swift adapter training"))
        indices = [idx for idx, _ in scored]
        # multi-match should rank first
        self.assertEqual(indices[0], 0)

    def test_empty_query_returns_empty(self):
        chunks = [_make_chunk("some content")]
        scored = _score_chunks(chunks, [])
        self.assertEqual(scored, [])

    def test_empty_chunks_returns_empty(self):
        scored = _score_chunks([], ["swift"])
        self.assertEqual(scored, [])

    def test_sublinear_tf_applied(self):
        # Chunk with 10 occurrences of the term should score less than 10x
        # a chunk with 1 occurrence (sublinear TF dampens repetition)
        many = _make_chunk("swift " * 10, turn_index=0)
        one = _make_chunk("swift", turn_index=1)
        from synapt.recall.bm25 import _tokenize
        scored = dict(_score_chunks([many, one], _tokenize("swift")))
        ratio = scored[0] / scored[1]
        # Sublinear: ratio should be < 10 (specifically: (1+log(10))/(1+log(1)) ≈ 3.3)
        self.assertLess(ratio, 10)
        self.assertGreater(ratio, 1)  # many still scores higher


# ---------------------------------------------------------------------------
# _format_live_results
# ---------------------------------------------------------------------------

class TestFormatLiveResults(unittest.TestCase):

    def _make_chunks_and_scores(self, texts):
        chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]
        scored = [(i, float(i + 1)) for i in range(len(chunks))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return chunks, scored

    def test_includes_header(self):
        chunks, scored = self._make_chunks_and_scores(["hello world"])
        result = _format_live_results(chunks, scored, max_chunks=5, max_tokens=500)
        self.assertIn("Current session context:", result)

    def test_empty_scored_returns_empty(self):
        chunks = [_make_chunk("hello")]
        result = _format_live_results(chunks, [], max_chunks=5, max_tokens=500)
        self.assertEqual(result, "")

    def test_respects_max_chunks(self):
        chunks, scored = self._make_chunks_and_scores(["a", "b", "c", "d", "e"])
        result = _format_live_results(chunks, scored, max_chunks=2, max_tokens=5000)
        # Only 2 turn blocks should appear (plus header)
        self.assertEqual(result.count("--- [current session"), 2)

    def test_user_text_truncated(self):
        long_text = "word " * 200  # 1000 chars
        chunks, scored = self._make_chunks_and_scores([long_text])
        result = _format_live_results(chunks, scored, max_chunks=5, max_tokens=5000)
        # 500 char limit on user_text (+ "...")
        user_line = result.split("User: ")[1].split("\n")[0]
        self.assertLess(len(user_line), 510)
        self.assertTrue(user_line.endswith("..."))


# ---------------------------------------------------------------------------
# search_live_transcript
# ---------------------------------------------------------------------------

class TestSearchLiveTranscript(unittest.TestCase):

    def setUp(self):
        # Reset the module-level cache before each test, respecting the lock.
        import synapt.recall.live as live_mod
        with live_mod._cache_lock:
            live_mod._cache = _LiveCache()

    def test_returns_current_session_results(self):
        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "session-abc", [
                {"user": "implementing swift adapter training", "assistant": "OK"},
            ])

            with patch("synapt.recall.live.latest_transcript_path", return_value=str(transcript)):
                result = search_live_transcript("swift adapter", max_chunks=3)

        self.assertIn("Current session context:", result)

    def test_returns_empty_when_no_transcript(self):
        with patch("synapt.recall.live.latest_transcript_path", return_value=None):
            result = search_live_transcript("swift adapter")
        self.assertEqual(result, "")

    def test_skips_when_session_already_indexed(self):
        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "already-indexed-id", [
                {"user": "swift adapter work", "assistant": "done"},
            ])

            # Mock index with sessions dict; getattr(mock, "sessions") returns this dict.
            mock_index = MagicMock()
            mock_index.sessions = {"already-indexed-id": []}

            with patch("synapt.recall.live.latest_transcript_path", return_value=str(transcript)):
                result = search_live_transcript("swift", index=mock_index)

        self.assertEqual(result, "")

    def test_returns_empty_when_no_query_match(self):
        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "session-xyz", [
                {"user": "docker compose postgres setup", "assistant": "done"},
            ])

            with patch("synapt.recall.live.latest_transcript_path", return_value=str(transcript)):
                result = search_live_transcript("swift adapter training")

        self.assertEqual(result, "")

    def test_respects_max_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "session-qrs", [
                {"user": f"swift work turn {i}", "assistant": "ok"} for i in range(10)
            ])

            with patch("synapt.recall.live.latest_transcript_path", return_value=str(transcript)):
                result = search_live_transcript("swift", max_chunks=2)

        self.assertTrue(result, "Expected live search results for 'swift' with 10 matching turns")
        self.assertLessEqual(result.count("--- [current session"), 2)

    def test_min_score_suppresses_weak_match(self):
        """A chunk matching only 1/4 query terms should be suppressed by min_score=1.0."""
        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            # "swift" appears, but "adapter", "training", "pipeline" do not
            _write_transcript(transcript, "session-weak", [
                {"user": "swift docker setup config build", "assistant": "ok"},
            ])

            with patch("synapt.recall.live.latest_transcript_path", return_value=str(transcript)):
                # Default min_score=1.0: single weak match should not qualify
                result_default = search_live_transcript(
                    "swift adapter training pipeline", min_score=1.0
                )
                # min_score=0: same query should return a result
                result_zero = search_live_transcript(
                    "swift adapter training pipeline", min_score=0.0
                )

        self.assertEqual(result_default, "", "Weak match should be suppressed at min_score=1.0")
        # Guard against stemmer changes: verify "swift" is preserved before
        # asserting that the un-gated search returns a result.
        from synapt.recall.bm25 import _tokenize
        self.assertIn("swift", _tokenize("swift"),
                      "Stemmer now removes 'swift' — update the test document text")
        self.assertIn("Current session context:", result_zero)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestLiveCache(unittest.TestCase):

    def setUp(self):
        import synapt.recall.live as live_mod
        with live_mod._cache_lock:
            live_mod._cache = _LiveCache()

    def test_cache_avoids_reparse_on_same_size(self):
        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "cached-session", [
                {"user": "initial work", "assistant": "ok"},
            ])

            path = str(transcript)
            # Patch the source module (synapt.recall.core) because live.py
            # imports parse_transcript lazily inside _get_live_chunks.  If that
            # import is ever moved to the top of live.py, the correct target
            # becomes "synapt.recall.live.parse_transcript".
            with patch("synapt.recall.core.parse_transcript") as mock_parse:
                from synapt.recall.core import TranscriptChunk
                mock_parse.return_value = [_make_chunk("initial work")]

                _get_live_chunks(path)
                _get_live_chunks(path)  # same file size — should NOT re-parse

                self.assertEqual(mock_parse.call_count, 1)

    def test_cache_reparses_when_file_grows(self):
        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "growing-session", [
                {"user": "first turn", "assistant": "ok"},
            ])

            path = str(transcript)
            with patch("synapt.recall.core.parse_transcript") as mock_parse:
                mock_parse.return_value = [_make_chunk("first turn")]

                _get_live_chunks(path)

                # Append a new turn — file grows
                with open(transcript, "a") as f:
                    f.write(json.dumps({"type": "user", "message": {
                        "role": "user", "content": "second turn"
                    }}) + "\n")

                _get_live_chunks(path)  # bigger file — should re-parse

                self.assertEqual(mock_parse.call_count, 2)

    def test_concurrent_calls_parse_once(self):
        """Two threads racing on a cache miss should trigger exactly one parse.

        The start_barrier synchronizes thread entry so they genuinely race for
        _cache_lock.  The barrier must be OUTSIDE the lock — placing it inside
        the parse mock would deadlock because Thread 2 blocks on lock.acquire()
        and never reaches the barrier, leaving Thread 1 stuck waiting.

        Note: this test verifies parse-once semantics, not the race itself.
        On heavily-loaded or single-core CI the OS may run threads sequentially
        (Thread 1 acquires lock, parses, releases; Thread 2 finds cache warm).
        The assertion is still correct — parse is called once — but the actual
        simultaneous lock contention is not deterministically exercised.
        Accepted: forcing the race would require mocking the lock internals.
        """
        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "concurrent-session", [
                {"user": "concurrent work on the project", "assistant": "ok"},
            ])

            path = str(transcript)
            parse_call_count: list[int] = []
            # Both threads synchronize here (outside the lock) so they race
            # for _cache_lock simultaneously rather than running sequentially.
            start_barrier = threading.Barrier(2)

            def _run():
                start_barrier.wait()   # sync before lock acquisition
                _get_live_chunks(path)

            def _counting_parse(p):
                parse_call_count.append(1)
                return [_make_chunk("concurrent work")]

            with patch("synapt.recall.core.parse_transcript", side_effect=_counting_parse):
                threads = [threading.Thread(target=_run) for _ in range(2)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            # The lock guarantees that only one thread runs parse; the second
            # finds the cache warm when it enters the critical section.
            self.assertEqual(len(parse_call_count), 1, "parse_transcript called more than once")


# ---------------------------------------------------------------------------
# compact_journal edge cases
# ---------------------------------------------------------------------------

class TestCompactJournal(unittest.TestCase):

    def test_handles_empty_file(self):
        """compact_journal on a 0-byte journal returns 0 without error."""
        with tempfile.TemporaryDirectory() as d:
            journal = Path(d) / "journal.jsonl"
            journal.write_text("")  # empty file
            result = compact_journal(journal)
            self.assertEqual(result, 0)

    def test_returns_zero_when_no_duplicates(self):
        """compact_journal returns 0 when there is nothing to compact."""
        from synapt.recall.journal import append_entry, JournalEntry
        with tempfile.TemporaryDirectory() as d:
            journal = Path(d) / "journal.jsonl"
            entry = JournalEntry(
                timestamp="2026-03-01T10:00:00",
                session_id="unique-session",
                focus="some work",
            )
            append_entry(entry, journal)
            result = compact_journal(journal)
            self.assertEqual(result, 0)

    def test_compacts_duplicate_sessions(self):
        """compact_journal removes duplicates and keeps the richest entry."""
        from synapt.recall.journal import append_entry, JournalEntry
        with tempfile.TemporaryDirectory() as d:
            journal = Path(d) / "journal.jsonl"
            # Auto-extracted stub (less rich)
            stub = JournalEntry(
                timestamp="2026-03-01T10:00:00",
                session_id="dup-session",
                auto=True,
            )
            # Enriched entry (richer)
            rich = JournalEntry(
                timestamp="2026-03-01T11:00:00",
                session_id="dup-session",
                focus="detailed focus",
                done=["task 1"],
                auto=False,
            )
            append_entry(stub, journal)
            append_entry(rich, journal)
            removed = compact_journal(journal)
            self.assertEqual(removed, 1)
            # Verify the rich entry survived
            from synapt.recall.journal import read_entries
            entries = read_entries(journal, n=10)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].focus, "detailed focus")

    def test_rewrites_file_in_chronological_order(self):
        """compact_journal stores surviving entries in ascending timestamp order."""
        import json
        from synapt.recall.journal import append_entry, JournalEntry
        with tempfile.TemporaryDirectory() as d:
            journal = Path(d) / "journal.jsonl"
            # Write two sessions in reverse chronological order, each with a
            # duplicate, so compact_journal does actual work (removed > 0).
            newer = JournalEntry(
                timestamp="2026-03-02T12:00:00",
                session_id="newer-session",
                focus="newer work",
            )
            older = JournalEntry(
                timestamp="2026-03-01T08:00:00",
                session_id="older-session",
                focus="older work",
            )
            # Write newer first, then older (reverse order), then duplicates
            for entry in [newer, older,
                          JournalEntry(timestamp="2026-03-02T12:00:00",
                                       session_id="newer-session", auto=True),
                          JournalEntry(timestamp="2026-03-01T08:00:00",
                                       session_id="older-session", auto=True)]:
                append_entry(entry, journal)

            removed = compact_journal(journal)
            self.assertEqual(removed, 2)

            # File must be in ascending timestamp order
            lines = [l for l in journal.read_text().splitlines() if l.strip()]
            timestamps = [json.loads(l)["timestamp"] for l in lines]
            self.assertEqual(timestamps, sorted(timestamps),
                             "compact_journal must write entries chronologically")


# ---------------------------------------------------------------------------
# Integration: recall_search includes live results
# ---------------------------------------------------------------------------

class TestRecallSearchLiveIntegration(unittest.TestCase):

    def setUp(self):
        # Ref #967: recall_search resolves the recall data root implicitly, so
        # the "unavailable" and "setup" messages these tests assert on could
        # otherwise be produced against the operator's live store rather than
        # against the absence this test is constructing.
        self._store = owned_store()

    def tearDown(self):
        self._store.restore()

    def test_recall_search_combines_live_and_indexed(self):
        """recall_search should join 'Current session context:' and indexed results."""
        from synapt.recall.server import recall_search

        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "live-session", [
                {"user": "swift adapter training pipeline", "assistant": "done"},
            ])

            mock_index = MagicMock()
            mock_index.sessions = {}  # live session not yet indexed
            mock_index.lookup.return_value = "Past session context:\n--- some indexed result ---"
            mock_index._last_diagnostics = None

            with (
                patch("synapt.recall.server._get_index", return_value=mock_index),
                patch("synapt.recall.live.latest_transcript_path", return_value=str(transcript)),
            ):
                result = recall_search("swift adapter")

        # Both live (current) and indexed (past) results must appear
        self.assertIn("Current session context:", result, "Live result missing from combined output")
        self.assertIn("Past session context:", result, "Indexed result missing from combined output")
        # Results are joined with "\n\n"
        self.assertIn("\n\n", result, "Expected double-newline separator between live and indexed")
        # Live results are prepended before indexed results
        live_pos = result.index("Current session")
        past_pos = result.index("Past session")
        self.assertLess(live_pos, past_pos, "Live results should precede indexed results")

    def test_recall_search_no_live_only_indexed(self):
        """recall_search returns only indexed results when live transcript has no match."""
        from synapt.recall.server import recall_search

        mock_index = MagicMock()
        mock_index.sessions = {}
        mock_index.lookup.return_value = "Past session context:\n--- some indexed result ---"
        mock_index._last_diagnostics = None

        with (
            patch("synapt.recall.server._get_index", return_value=mock_index),
            patch("synapt.recall.live.latest_transcript_path", return_value=None),
        ):
            result = recall_search("swift adapter")

        self.assertNotIn("Current session context:", result)
        self.assertIn("Past session context:", result)

    def test_recall_search_live_only_when_no_index(self):
        """recall_search returns live results even when no index exists."""
        from synapt.recall.server import recall_search

        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "live-only", [
                {"user": "swift adapter training pipeline", "assistant": "done"},
            ])

            with (
                patch("synapt.recall.server._get_index", return_value=None),
                patch("synapt.recall.live.latest_transcript_path", return_value=str(transcript)),
            ):
                result = recall_search("swift adapter")

        self.assertIn("Current session context:", result)

    def test_recall_search_setup_message_when_nothing(self):
        """recall_search returns setup message when neither index nor live transcript exists."""
        from synapt.recall.server import recall_search

        with (
            patch("synapt.recall.server._get_index", return_value=None),
            patch("synapt.recall.live.latest_transcript_path", return_value=None),
        ):
            result = recall_search("swift adapter")

        self.assertIn("Run `synapt recall setup`", result)


    def test_before_filter_suppresses_live_results(self):
        """recall_search with before= should NOT include current-session context."""
        from synapt.recall.server import recall_search

        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "live-session", [
                {"user": "swift adapter training pipeline", "assistant": "done"},
            ])

            mock_index = MagicMock()
            mock_index.sessions = {}
            mock_index.lookup.return_value = "Past session context:\n--- historical result ---"
            mock_index._last_diagnostics = None

            with (
                patch("synapt.recall.server._get_index", return_value=mock_index),
                patch("synapt.recall.live.latest_transcript_path", return_value=str(transcript)),
            ):
                result = recall_search("swift adapter", before="2026-04-09")

        self.assertNotIn("Current session context:", result,
                         "Live results should be suppressed when before= is set")
        self.assertIn("Past session context:", result)

    def test_after_filter_still_includes_live_results(self):
        """recall_search with after= (but no before=) should still include live."""
        from synapt.recall.server import recall_search

        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "live-session", [
                {"user": "swift adapter training pipeline", "assistant": "done"},
            ])

            mock_index = MagicMock()
            mock_index.sessions = {}
            mock_index.lookup.return_value = "Past session context:\n--- indexed result ---"
            mock_index._last_diagnostics = None

            with (
                patch("synapt.recall.server._get_index", return_value=mock_index),
                patch("synapt.recall.live.latest_transcript_path", return_value=str(transcript)),
            ):
                result = recall_search("swift adapter", after="2026-04-01")

        self.assertIn("Current session context:", result,
                       "Live results should still appear with after= only")

    def test_before_filter_no_index_returns_unavailable(self):
        """When before= is set but no index exists, return clear error."""
        from synapt.recall.server import recall_search

        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "live-session", [
                {"user": "swift adapter training pipeline", "assistant": "done"},
            ])

            with (
                patch("synapt.recall.server._get_index", return_value=None),
                patch("synapt.recall.live.latest_transcript_path", return_value=str(transcript)),
            ):
                result = recall_search("swift adapter", before="2026-04-09")

        self.assertIn("Historical search unavailable", result,
                       "Should explain that historical search needs an index")
        self.assertNotIn("Current session context:", result,
                         "Should not fall back to live results for historical query")

    def test_after_filter_no_index_returns_unavailable(self):
        """When after= is set but no index exists, return clear error."""
        from synapt.recall.server import recall_search

        with (
            patch("synapt.recall.server._get_index", return_value=None),
            patch("synapt.recall.live.latest_transcript_path", return_value=None),
        ):
            result = recall_search("swift adapter", after="2026-04-01")

        self.assertIn("Historical search unavailable", result)


class TestRecallQuickStatusRouting(unittest.TestCase):

    def test_pending_query_uses_summary_depth(self):
        """Pending-work queries should let recall_quick surface journal entries."""
        from synapt.recall.server import recall_quick

        calls = []

        class MockIndex:
            _last_diagnostics = None
            _embedding_status = "disabled"

            def lookup(self, query, **kwargs):
                calls.append((query, kwargs))
                return "Past session context:\n--- journal result ---"

        with (
            patch("synapt.recall.server._get_index", return_value=MockIndex()),
            patch("synapt.recall.server.project_index_dir", return_value=None),
            patch(
                "synapt.recall.server._query_freshness_line",
                return_value="Freshness: NOT_BOUND test",
            ),
        ):
            result = recall_quick("what's pending")

        self.assertIn("journal result", result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["depth"], "summary")

    def test_non_status_query_keeps_concise_depth(self):
        """Ordinary quick checks should preserve concise mode."""
        from synapt.recall.server import recall_quick

        calls = []

        class MockIndex:
            _last_diagnostics = None
            _embedding_status = "disabled"

            def lookup(self, query, **kwargs):
                calls.append((query, kwargs))
                return "Past session context:\n--- concise result ---"

        with (
            patch("synapt.recall.server._get_index", return_value=MockIndex()),
            patch("synapt.recall.server.project_index_dir", return_value=None),
            patch(
                "synapt.recall.server._query_freshness_line",
                return_value="Freshness: NOT_BOUND test",
            ),
        ):
            result = recall_quick("what is the database port")

        self.assertIn("concise result", result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["depth"], "concise")


# ---------------------------------------------------------------------------
# PreCompact journal write
# ---------------------------------------------------------------------------

class TestPrecompactJournalWrite(unittest.TestCase):

    def _write_minimal_transcript(self, path: Path, session_id: str) -> None:
        _write_transcript(path, session_id, [
            {"user": "implement recall feature", "assistant": "done"},
        ])

    def test_writes_journal_on_first_call(self):
        from synapt.recall.cli import _precompact_journal_write

        with tempfile.TemporaryDirectory() as d:
            project = Path(d)
            transcript = Path(d) / "session.jsonl"
            self._write_minimal_transcript(transcript, "new-session-id")

            with (
                patch("synapt.recall.cli.latest_transcript_path", return_value=str(transcript)),
                patch("synapt.recall.cli.extract_session_id", return_value="new-session-id"),
                patch("synapt.recall.cli._read_all_session_ids", return_value=set()),
                patch("synapt.recall.cli.append_entry") as mock_append,
                patch("synapt.recall.cli.auto_extract_entry") as mock_extract,
                patch("synapt.recall.cli._journal_path", return_value=project / "journal.jsonl"),
            ):
                from synapt.recall.journal import JournalEntry
                mock_entry = MagicMock(spec=JournalEntry)
                mock_entry.has_content.return_value = True
                mock_extract.return_value = mock_entry

                _precompact_journal_write(project)

            mock_append.assert_called_once()

    def test_skips_duplicate_on_second_call(self):
        from synapt.recall.cli import _precompact_journal_write

        with tempfile.TemporaryDirectory() as d:
            project = Path(d)
            transcript = Path(d) / "session.jsonl"
            self._write_minimal_transcript(transcript, "existing-session-id")

            with (
                patch("synapt.recall.cli.latest_transcript_path", return_value=str(transcript)),
                patch("synapt.recall.cli.extract_session_id", return_value="existing-session-id"),
                patch("synapt.recall.cli._read_all_session_ids",
                      return_value={"existing-session-id"}),
                patch("synapt.recall.cli.append_entry") as mock_append,
                patch("synapt.recall.cli._journal_path", return_value=project / "journal.jsonl"),
            ):
                _precompact_journal_write(project)

            mock_append.assert_not_called()

    def test_skips_when_session_id_empty(self):
        """_precompact_journal_write exits early if extract_session_id returns ''."""
        from synapt.recall.cli import _precompact_journal_write

        with tempfile.TemporaryDirectory() as d:
            project = Path(d)
            transcript = Path(d) / "session.jsonl"
            self._write_minimal_transcript(transcript, "")

            with (
                patch("synapt.recall.cli.latest_transcript_path", return_value=str(transcript)),
                patch("synapt.recall.cli.extract_session_id", return_value=""),
                patch("synapt.recall.cli.append_entry") as mock_append,
                patch("synapt.recall.cli._journal_path", return_value=project / "journal.jsonl"),
            ):
                _precompact_journal_write(project)

            mock_append.assert_not_called()

    def test_post_extract_recheck_prevents_duplicate(self):
        """The second _read_all_session_ids call (after auto_extract_entry) fires
        when a concurrent SessionEnd writes the entry during the git subprocess
        window.  append_entry must NOT be called in this case."""
        from synapt.recall.cli import _precompact_journal_write

        with tempfile.TemporaryDirectory() as d:
            project = Path(d)
            transcript = Path(d) / "session.jsonl"
            self._write_minimal_transcript(transcript, "race-session-id")

            # First call returns set() (check 1 passes), second call returns the
            # id (simulating SessionEnd writing during auto_extract_entry).
            read_ids_results = [set(), {"race-session-id"}]

            with (
                patch("synapt.recall.cli.latest_transcript_path", return_value=str(transcript)),
                patch("synapt.recall.cli.extract_session_id", return_value="race-session-id"),
                patch("synapt.recall.cli._read_all_session_ids",
                      side_effect=read_ids_results),
                patch("synapt.recall.cli.append_entry") as mock_append,
                patch("synapt.recall.cli.auto_extract_entry") as mock_extract,
                patch("synapt.recall.cli._journal_path", return_value=project / "journal.jsonl"),
            ):
                from synapt.recall.journal import JournalEntry
                mock_entry = MagicMock(spec=JournalEntry)
                mock_entry.has_content.return_value = True
                mock_extract.return_value = mock_entry

                _precompact_journal_write(project)

            # The post-extract re-check caught the race — no duplicate written
            mock_append.assert_not_called()


# ---------------------------------------------------------------------------
# Additional coverage: exception paths and edge cases
# ---------------------------------------------------------------------------

class TestLiveEdgeCases(unittest.TestCase):

    def setUp(self):
        import synapt.recall.live as live_mod
        with live_mod._cache_lock:
            live_mod._cache = _LiveCache()

    def test_get_live_chunks_handles_parse_exception(self):
        """_get_live_chunks returns ('', []) when parse_transcript raises a
        non-OSError exception (e.g., UnicodeDecodeError from a malformed line)."""
        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "err-session", [
                {"user": "some work", "assistant": "ok"},
            ])

            with patch("synapt.recall.core.parse_transcript",
                       side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "bad byte")):
                session_id, chunks = _get_live_chunks(str(transcript))

        self.assertEqual(session_id, "")
        self.assertEqual(chunks, [])

    def test_search_live_transcript_warns_on_unexpected_error(self):
        """search_live_transcript logs a WARNING and returns '' when _get_live_chunks
        raises an unexpected exception (the outer except Exception guard)."""
        with tempfile.TemporaryDirectory() as d:
            transcript = Path(d) / "session.jsonl"
            _write_transcript(transcript, "explode-session", [
                {"user": "work", "assistant": "ok"},
            ])

            with (
                patch("synapt.recall.live.latest_transcript_path", return_value=str(transcript)),
                patch("synapt.recall.live._get_live_chunks",
                      side_effect=RuntimeError("unexpected internal error")),
            ):
                import logging
                with self.assertLogs("synapt.recall.live", level=logging.WARNING):
                    result = search_live_transcript("work")

        self.assertEqual(result, "")

    def test_format_live_results_zero_budget_emits_first_chunk(self):
        """Even with max_tokens=0 the first chunk is always included
        (the len(lines) > 1 guard is False on iteration 1)."""
        chunks, scored = (
            [_make_chunk("first chunk content", 0), _make_chunk("second chunk content", 1)],
            [(0, 2.0), (1, 1.0)],
        )
        result = _format_live_results(chunks, scored, max_chunks=5, max_tokens=0)
        self.assertIn("Current session context:", result)
        self.assertIn("first chunk", result)
        # Second chunk must be suppressed — budget exhausted after first
        self.assertNotIn("second chunk", result)


if __name__ == "__main__":
    unittest.main()
