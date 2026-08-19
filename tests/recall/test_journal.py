"""Tests for synapt.recall.journal — session journal storage and formatting."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.journal import (
    JournalEntry,
    _dedup_entries,
    append_entry,
    auto_extract_entry,
    compact_journal,
    format_entry_full,
    format_for_session_start,
    merge_carried_forward_next_steps,
    read_entries,
    read_latest,
    read_previous_meaningful,
)


class TestJournalEntry(unittest.TestCase):
    def test_round_trip(self):
        entry = JournalEntry(
            timestamp="2026-03-02T12:00:00+00:00",
            session_id="abc123",
            branch="main",
            focus="testing",
            done=["wrote tests"],
            decisions=["use JSONL"],
            next_steps=["deploy"],
            files_modified=["foo.py"],
            git_log=["abc1234 initial commit"],
        )
        d = entry.to_dict()
        restored = JournalEntry.from_dict(d)
        self.assertEqual(restored.timestamp, entry.timestamp)
        self.assertEqual(restored.focus, entry.focus)
        self.assertEqual(restored.done, entry.done)
        self.assertEqual(restored.next_steps, entry.next_steps)

    def test_from_dict_ignores_extra_keys(self):
        d = {"timestamp": "2026-03-02T12:00:00", "unknown_field": "ignored"}
        entry = JournalEntry.from_dict(d)
        self.assertEqual(entry.timestamp, "2026-03-02T12:00:00")


class TestJournalStorage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "journal.jsonl"

    def test_append_and_read_latest(self):
        e1 = JournalEntry(timestamp="2026-03-01T10:00:00", focus="first")
        e2 = JournalEntry(timestamp="2026-03-02T10:00:00", focus="second")
        append_entry(e1, self.path)
        append_entry(e2, self.path)

        latest = read_latest(self.path)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.focus, "second")

    def test_read_entries_ordering(self):
        for i in range(5):
            append_entry(
                JournalEntry(timestamp=f"2026-03-0{i+1}T10:00:00", focus=f"entry-{i}"),
                self.path,
            )
        entries = read_entries(self.path, n=3)
        self.assertEqual(len(entries), 3)
        # Most recent first
        self.assertEqual(entries[0].focus, "entry-4")
        self.assertEqual(entries[2].focus, "entry-2")

    def test_read_empty_file(self):
        self.assertEqual(read_entries(self.path, n=5), [])
        self.assertIsNone(read_latest(self.path))

    def test_read_nonexistent_file(self):
        fake = Path(self.tmpdir) / "nonexistent.jsonl"
        self.assertEqual(read_entries(fake), [])

    def test_append_creates_parent_dirs(self):
        nested = Path(self.tmpdir) / "a" / "b" / "journal.jsonl"
        entry = JournalEntry(timestamp="2026-03-02T10:00:00", focus="nested")
        append_entry(entry, nested)
        self.assertTrue(nested.exists())

    def test_corrupt_line_skipped(self):
        # Write a valid entry, then a corrupt line, then another valid entry
        with open(self.path, "w") as f:
            f.write(json.dumps({"timestamp": "2026-03-01T10:00:00", "focus": "good1"}) + "\n")
            f.write("not json\n")
            f.write(json.dumps({"timestamp": "2026-03-02T10:00:00", "focus": "good2"}) + "\n")
        entries = read_entries(self.path, n=5)
        self.assertEqual(len(entries), 2)


class TestJournalFormatting(unittest.TestCase):
    def test_format_for_session_start(self):
        entry = JournalEntry(
            timestamp="2026-03-02T12:30:00",
            focus="building journal",
            done=["wrote module"],
            next_steps=["add tests", "update hooks"],
        )
        text = format_for_session_start(entry)
        self.assertIn("building journal", text)
        self.assertIn("wrote module", text)
        self.assertIn("add tests", text)
        self.assertIn("update hooks", text)

    def test_format_empty_entry(self):
        entry = JournalEntry(timestamp="2026-03-02T12:00:00")
        text = format_for_session_start(entry)
        self.assertEqual(text, "")

    def test_format_entry_full(self):
        entry = JournalEntry(
            timestamp="2026-03-02T12:30:00",
            branch="feat/test",
            focus="testing",
            done=["task A"],
            decisions=["use JSON"],
            next_steps=["task B"],
            files_modified=["a.py", "b.py"],
            git_log=["abc1234 commit msg"],
        )
        text = format_entry_full(entry)
        self.assertIn("feat/test", text)
        self.assertIn("testing", text)
        self.assertIn("task A", text)
        self.assertIn("use JSON", text)
        self.assertIn("task B", text)
        self.assertIn("a.py", text)
        self.assertIn("abc1234", text)


class TestAutoExtract(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_extracts_files_from_transcript(self):
        transcript = Path(self.tmpdir) / "session.jsonl"
        transcript.write_text(
            json.dumps({"type": "progress", "sessionId": "abc-123"}) + "\n"
            + json.dumps({
                "type": "assistant",
                "message": {"content": [
                    {"name": "Edit", "input": {"file_path": f"{self.tmpdir}/src/foo.py"}},
                    {"name": "Write", "input": {"file_path": f"{self.tmpdir}/src/bar.py"}},
                ]},
            }) + "\n"
        )
        with patch("synapt.recall.journal._get_branch", return_value="main"), \
             patch("synapt.recall.journal._get_recent_commits", return_value=["abc commit"]):
            entry = auto_extract_entry(str(transcript), self.tmpdir)
        self.assertEqual(entry.session_id, "abc-123")
        self.assertEqual(entry.branch, "main")
        self.assertIn("src/foo.py", entry.files_modified)
        self.assertIn("src/bar.py", entry.files_modified)

    def test_extracts_without_transcript(self):
        with patch("synapt.recall.journal._get_branch", return_value="feat/x"), \
             patch("synapt.recall.journal._get_recent_commits", return_value=[]):
            entry = auto_extract_entry(None, self.tmpdir)
        self.assertEqual(entry.branch, "feat/x")
        self.assertEqual(entry.files_modified, [])
        self.assertEqual(entry.session_id, "")

    def test_skips_claude_internal_paths(self):
        transcript = Path(self.tmpdir) / "session.jsonl"
        transcript.write_text(
            json.dumps({
                "type": "assistant",
                "message": {"content": [
                    {"name": "Write", "input": {"file_path": f"{self.tmpdir}/.claude/plans/foo.md"}},
                    {"name": "Edit", "input": {"file_path": f"{self.tmpdir}/real.py"}},
                ]},
            }) + "\n"
        )
        with patch("synapt.recall.journal._get_branch", return_value=""), \
             patch("synapt.recall.journal._get_recent_commits", return_value=[]):
            entry = auto_extract_entry(str(transcript), self.tmpdir)
        self.assertEqual(entry.files_modified, ["real.py"])

    def test_handles_malformed_transcript(self):
        transcript = Path(self.tmpdir) / "session.jsonl"
        transcript.write_text("not json\n{bad\n")
        with patch("synapt.recall.journal._get_branch", return_value=""), \
             patch("synapt.recall.journal._get_recent_commits", return_value=[]):
            entry = auto_extract_entry(str(transcript), self.tmpdir)
        self.assertEqual(entry.files_modified, [])


class TestReadLatestMeaningful(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "journal.jsonl"

    def test_skips_auto_entries_to_find_meaningful(self):
        # Rich entry first, then two auto-extracted noise entries on top
        append_entry(JournalEntry(
            timestamp="2026-03-02T10:00:00", focus="real work", next_steps=["do X"],
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-03-02T11:00:00", files_modified=["a.py"],
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-03-02T12:00:00", files_modified=["b.py"],
        ), self.path)

        # Without meaningful=True, gets the latest (empty noise)
        latest = read_latest(self.path, meaningful=False)
        self.assertEqual(latest.files_modified, ["b.py"])
        self.assertEqual(latest.focus, "")

        # With meaningful=True, skips noise to find the rich entry
        meaningful = read_latest(self.path, meaningful=True)
        self.assertEqual(meaningful.focus, "real work")
        self.assertEqual(meaningful.next_steps, ["do X"])

    def test_meaningful_returns_none_when_all_empty(self):
        append_entry(JournalEntry(
            timestamp="2026-03-02T10:00:00", files_modified=["a.py"],
        ), self.path)
        self.assertIsNone(read_latest(self.path, meaningful=True))

    def test_meaningful_survives_many_empty_entries(self):
        """Meaningful entry is found even under 20+ empty auto-entries (#236)."""
        # Write one meaningful entry
        append_entry(JournalEntry(
            timestamp="2026-02-07T10:00:00", focus="deep work",
            done=["shipped feature"],
        ), self.path)
        # Pile 25 empty auto-entries on top (old lookback was 10)
        for i in range(25):
            append_entry(JournalEntry(
                timestamp=f"2026-03-03T{10+i//60:02d}:{i%60:02d}:00",
                files_modified=[f"file{i}.py"],
            ), self.path)

        result = read_latest(self.path, meaningful=True)
        self.assertIsNotNone(result, "Should find meaningful entry under 25 empty ones")
        self.assertEqual(result.focus, "deep work")


class TestHasContent(unittest.TestCase):
    def test_empty_entry_has_no_content(self):
        entry = JournalEntry(timestamp="2026-03-02T12:00:00")
        self.assertFalse(entry.has_content())

    def test_entry_with_focus_has_content(self):
        entry = JournalEntry(timestamp="2026-03-02T12:00:00", focus="testing")
        self.assertTrue(entry.has_content())

    def test_entry_with_files_has_content(self):
        entry = JournalEntry(timestamp="2026-03-02T12:00:00", files_modified=["a.py"])
        self.assertTrue(entry.has_content())


class TestDedupAndCompact(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "journal.jsonl"

    def test_read_entries_deduplicates_by_session_id(self):
        """Two entries with the same session_id: only the richest is returned."""
        append_entry(JournalEntry(
            timestamp="2026-03-01T10:00:00", session_id="sess-A",
            files_modified=["a.py"], auto=True,
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-03-01T11:00:00", session_id="sess-A",
            focus="enriched focus", done=["task done"], auto=False,
        ), self.path)
        entries = read_entries(self.path, n=5)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].focus, "enriched focus")

    def test_read_entries_sorted_by_timestamp(self):
        """Entries written out of order are returned newest-first."""
        # Write in non-chronological order (simulates synthesis appending old entries)
        append_entry(JournalEntry(
            timestamp="2026-03-03T10:00:00", session_id="sess-C", focus="newest",
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-02-01T10:00:00", session_id="sess-A", focus="oldest",
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-02-15T10:00:00", session_id="sess-B", focus="middle",
        ), self.path)
        entries = read_entries(self.path, n=5)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0].focus, "newest")
        self.assertEqual(entries[1].focus, "middle")
        self.assertEqual(entries[2].focus, "oldest")

    def test_read_entries_prefers_non_auto(self):
        """Non-auto entry wins over auto entry for the same session_id."""
        append_entry(JournalEntry(
            timestamp="2026-03-01T12:00:00", session_id="sess-X",
            focus="auto focus", files_modified=["x.py"], auto=True,
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-03-01T10:00:00", session_id="sess-X",
            focus="manual focus", done=["real work"], auto=False,
        ), self.path)
        entries = read_entries(self.path, n=5)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].focus, "manual focus")
        self.assertFalse(entries[0].auto)

    def test_read_entries_keeps_newer_current_next_steps_over_richer_prior_entry(self):
        """A new session write must not lose its current next step to an older,
        richer entry that happens to share a session id.

        Session IDs can be reused by a resumed runtime. The current entry is
        the continuity handoff, so recency must win before richness: otherwise
        a completed item remains visible and the next step just written drops.
        """
        append_entry(JournalEntry(
            timestamp="2026-08-13T10:00:00", session_id="resumed-session",
            focus="prior work", done=["completed item"], decisions=["kept"],
            next_steps=["prior unfinished"], auto=False,
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-08-13T11:00:00", session_id="resumed-session",
            focus="current work", next_steps=["current next step"], auto=False,
        ), self.path)

        latest = read_latest(self.path)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.next_steps, ["current next step"])
        self.assertNotIn("completed item", latest.done)

    def test_dedup_orders_mixed_timestamp_formats_chronologically(self):
        """A naive legacy timestamp is UTC, not a lexicographic timestamp."""
        append_entry(JournalEntry(
            timestamp="2026-08-13T12:00:00", session_id="resumed-session",
            focus="older naive entry", auto=False,
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-08-13T11:30:00-02:00", session_id="resumed-session",
            focus="newer aware entry", auto=False,
        ), self.path)

        latest = read_latest(self.path)

        self.assertIsNotNone(latest)
        self.assertEqual(latest.focus, "newer aware entry")

    def test_read_and_compact_order_mixed_timestamp_formats_chronologically(self):
        """All journal ordering uses the same timestamp interpretation."""
        append_entry(JournalEntry(
            timestamp="2026-08-13T12:00:00", session_id="naive",
            focus="older naive entry", auto=False,
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-08-13T11:30:00-02:00", session_id="aware",
            focus="newer aware entry", auto=False,
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-08-13T10:00:00+00:00", session_id="dedup",
            focus="discarded auto entry", auto=True,
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-08-13T10:30:00+00:00", session_id="dedup",
            focus="dedup winner", auto=False,
        ), self.path)

        self.assertEqual(
            [entry.focus for entry in read_entries(self.path, n=2)],
            ["newer aware entry", "older naive entry"],
        )

        self.assertEqual(compact_journal(self.path), 1)
        with open(self.path) as f:
            stored = [json.loads(line)["focus"] for line in f if line.strip()]
        self.assertEqual(
            stored,
            ["dedup winner", "older naive entry", "newer aware entry"],
        )

    def test_unparseable_timestamp_sorts_before_known_timestamps(self):
        """A malformed legacy timestamp cannot displace a dated entry."""
        append_entry(JournalEntry(
            timestamp="not-a-timestamp", session_id="unknown", focus="unknown time",
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-08-13T10:00:00+00:00", session_id="known", focus="known time",
        ), self.path)

        self.assertEqual(
            [entry.focus for entry in read_entries(self.path, n=2)],
            ["known time", "unknown time"],
        )

    def test_extreme_offset_timestamp_sorts_before_known_timestamps(self):
        """A parseable legacy value that overflows UTC conversion stays readable."""
        append_entry(JournalEntry(
            timestamp="0001-01-01T00:00:00+05:00", session_id="underflow",
            focus="underflow time",
        ), self.path)
        append_entry(JournalEntry(
            timestamp="9999-12-31T23:59:59-05:00", session_id="overflow",
            focus="overflow time",
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-08-13T10:00:00+00:00", session_id="known",
            focus="known time",
        ), self.path)

        entries = read_entries(self.path, n=3)

        self.assertEqual(entries[0].focus, "known time")
        self.assertEqual({entry.focus for entry in entries[1:]}, {"underflow time", "overflow time"})

    def test_compact_journal_removes_duplicates(self):
        """compact_journal deduplicates and sorts the file."""
        append_entry(JournalEntry(
            timestamp="2026-03-03T10:00:00", session_id="sess-B", focus="B",
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-03-01T10:00:00", session_id="sess-A", focus="A-auto",
            auto=True,
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-03-01T11:00:00", session_id="sess-A", focus="A-enriched",
            done=["task"], auto=False,
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-03-02T10:00:00", session_id="sess-A",
            files_modified=["x.py"], auto=True,
        ), self.path)

        removed = compact_journal(self.path)
        self.assertEqual(removed, 2)  # 4 entries -> 2 unique sessions

        # Re-read raw lines — file should be sorted chronologically
        with open(self.path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["session_id"], "sess-A")  # earlier
        self.assertEqual(lines[0]["focus"], "A-enriched")   # richest kept
        self.assertEqual(lines[1]["session_id"], "sess-B")  # later

    def test_compact_journal_noop_when_clean(self):
        """compact_journal returns 0 and doesn't rewrite when no dupes exist."""
        append_entry(JournalEntry(
            timestamp="2026-03-01T10:00:00", session_id="sess-A", focus="A",
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-03-02T10:00:00", session_id="sess-B", focus="B",
        ), self.path)

        mtime_before = self.path.stat().st_mtime
        removed = compact_journal(self.path)
        self.assertEqual(removed, 0)
        self.assertEqual(self.path.stat().st_mtime, mtime_before)

    def test_dedup_keeps_entries_without_session_id(self):
        """Entries without session_id are never deduped against each other."""
        entries = [
            JournalEntry(timestamp="2026-03-01T10:00:00", focus="first"),
            JournalEntry(timestamp="2026-03-02T10:00:00", focus="second"),
        ]
        result = _dedup_entries(entries)
        self.assertEqual(len(result), 2)

    def test_read_latest_with_out_of_order_file(self):
        """read_latest returns newest entry even when file is unordered (#244)."""
        # Simulate synthesis appending old entries at end of file
        append_entry(JournalEntry(
            timestamp="2026-03-03T10:00:00", session_id="sess-new", focus="today",
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-02-01T10:00:00", session_id="sess-old", focus="month ago",
        ), self.path)
        latest = read_latest(self.path, meaningful=False)
        self.assertEqual(latest.focus, "today")  # newest by timestamp, not last in file

    def test_compact_nonexistent_file(self):
        """compact_journal returns 0 for a missing file."""
        missing = Path(self.tmpdir) / "no-such-file.jsonl"
        self.assertEqual(compact_journal(missing), 0)

    def test_compact_preserves_no_sid_entries(self):
        """compact_journal keeps entries without session_id."""
        append_entry(JournalEntry(
            timestamp="2026-03-01T10:00:00", focus="no sid entry",
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-03-02T10:00:00", session_id="sess-A", focus="A1", auto=True,
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-03-02T11:00:00", session_id="sess-A", focus="A2",
        ), self.path)

        removed = compact_journal(self.path)
        self.assertEqual(removed, 1)  # one sess-A duplicate

        with open(self.path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 2)
        # Both the no-sid entry and the sess-A winner survive
        foci = {l["focus"] for l in lines}
        self.assertIn("no sid entry", foci)
        self.assertIn("A2", foci)

    def test_dedup_richness_tiebreaker_by_field_count(self):
        """When same-session writes tie on time, richer content breaks the tie."""
        entries = [
            JournalEntry(
                timestamp="2026-03-01T10:00:00", session_id="sess-A",
                focus="just focus", auto=True,
            ),
            JournalEntry(
                timestamp="2026-03-01T10:00:00", session_id="sess-A",
                focus="focus + done", done=["task"], auto=True,
            ),
        ]
        result = _dedup_entries(entries)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].focus, "focus + done")
        self.assertEqual(result[0].done, ["task"])


class TestNextStepCarryForward(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "journal.jsonl"

    def test_read_previous_meaningful_skips_current_session(self):
        append_entry(JournalEntry(
            timestamp="2026-03-01T10:00:00",
            session_id="prior",
            focus="prior session",
            next_steps=["follow up"],
        ), self.path)
        append_entry(JournalEntry(
            timestamp="2026-03-02T10:00:00",
            session_id="current",
            focus="current session",
            next_steps=["new task"],
        ), self.path)

        previous = read_previous_meaningful("current", self.path)
        self.assertIsNotNone(previous)
        self.assertEqual(previous.session_id, "prior")

    def test_merge_carries_forward_unresolved_prior_steps(self):
        previous = JournalEntry(
            timestamp="2026-03-01T10:00:00",
            next_steps=["ship docs", "follow up with team"],
        )

        merged = merge_carried_forward_next_steps(
            current_next_steps=["write tests"],
            current_done=["ship docs"],
            previous_entry=previous,
        )

        self.assertEqual(merged, ["write tests", "follow up with team"])

    def test_merge_deduplicates_existing_next_steps(self):
        previous = JournalEntry(
            timestamp="2026-03-01T10:00:00",
            next_steps=["Write tests", "follow up with team"],
        )

        merged = merge_carried_forward_next_steps(
            current_next_steps=["write tests", "close loop"],
            current_done=[],
            previous_entry=previous,
        )

        self.assertEqual(merged, ["write tests", "close loop", "follow up with team"])


class TestJournalWriteResponse(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "journal.jsonl"
        self.project = Path(self.tmpdir) / "project"
        self.project.mkdir()

    def test_write_response_separates_explicit_next_steps_from_carried_forward(self):
        """MCP write response must not present prior pending work as caller-written."""
        from synapt.recall.server import recall_journal

        append_entry(
            JournalEntry(
                timestamp="2026-05-06T10:00:00+00:00",
                session_id="prior-session",
                focus="prior work",
                next_steps=[
                    "retrieve the missing design docs",
                    "estimate effort for prompt profiles",
                ],
            ),
            self.path,
        )

        with patch("synapt.recall.journal._journal_path", return_value=self.path), \
             patch("synapt.recall.server.Path.cwd", return_value=self.project), \
             patch("synapt.recall.journal.latest_transcript_path", return_value=None), \
             patch("synapt.recall.journal.auto_extract_entry", return_value=JournalEntry(
                 timestamp="2026-05-07T10:00:00+00:00",
                 session_id="current-session",
             )):
            response = recall_journal(
                action="write",
                focus="current work",
                next_steps="write the crisis eval; update the B3 row",
            )

        next_section = response.split("### Next", 1)[1].split(
            "### Carried Forward Next Steps",
            1,
        )[0]
        self.assertIn("write the crisis eval", next_section)
        self.assertIn("update the B3 row", next_section)
        self.assertNotIn("retrieve the missing design docs", next_section)
        self.assertNotIn("estimate effort for prompt profiles", next_section)
        self.assertIn("### Carried Forward Next Steps", response)
        self.assertIn("retrieve the missing design docs", response)
        self.assertIn("estimate effort for prompt profiles", response)

        latest = read_latest(self.path)
        self.assertIsNotNone(latest)
        self.assertEqual(
            latest.next_steps,
            [
                "write the crisis eval",
                "update the B3 row",
                "retrieve the missing design docs",
                "estimate effort for prompt profiles",
            ],
        )


class TestPendingNextSteps(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.jpath = Path(self.tmp) / "journal.jsonl"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_empty_journal(self):
        from synapt.recall.journal import pending_next_steps
        self.assertEqual(pending_next_steps(self.jpath), [])

    def test_single_entry_all_pending(self):
        from synapt.recall.journal import pending_next_steps, append_entry, JournalEntry
        e = JournalEntry(
            timestamp="2026-04-01T10:00",
            next_steps=["task A", "task B", "task C"],
        )
        append_entry(e, self.jpath)
        self.assertEqual(pending_next_steps(self.jpath), ["task A", "task B", "task C"])

    def test_done_items_filtered_out(self):
        from synapt.recall.journal import pending_next_steps, append_entry, JournalEntry
        e1 = JournalEntry(
            timestamp="2026-04-01T10:00",
            next_steps=["task A", "task B", "task C"],
        )
        append_entry(e1, self.jpath)
        e2 = JournalEntry(
            timestamp="2026-04-02T10:00",
            done=["task B"],
        )
        append_entry(e2, self.jpath)
        self.assertEqual(pending_next_steps(self.jpath), ["task A", "task C"])

    def test_normalization_matching(self):
        from synapt.recall.journal import pending_next_steps, append_entry, JournalEntry
        e1 = JournalEntry(
            timestamp="2026-04-01T10:00",
            next_steps=["Fix the  Bug", "deploy to prod"],
        )
        append_entry(e1, self.jpath)
        e2 = JournalEntry(
            timestamp="2026-04-02T10:00",
            done=["fix the bug"],  # different case + whitespace
        )
        append_entry(e2, self.jpath)
        self.assertEqual(pending_next_steps(self.jpath), ["deploy to prod"])

    def test_scans_all_entries_not_just_latest(self):
        """Older next_steps not carried forward should still appear."""
        from synapt.recall.journal import pending_next_steps, append_entry, JournalEntry
        e1 = JournalEntry(
            timestamp="2026-04-01T10:00",
            next_steps=["old task"],
        )
        append_entry(e1, self.jpath)
        # Entry 2 has its own next_steps but didn't carry forward "old task"
        e2 = JournalEntry(
            timestamp="2026-04-02T10:00",
            next_steps=["new task"],
        )
        append_entry(e2, self.jpath)
        pending = pending_next_steps(self.jpath)
        self.assertIn("new task", pending)
        self.assertIn("old task", pending)

    def test_all_done_returns_empty(self):
        from synapt.recall.journal import pending_next_steps, append_entry, JournalEntry
        e1 = JournalEntry(
            timestamp="2026-04-01T10:00",
            next_steps=["task A"],
        )
        append_entry(e1, self.jpath)
        e2 = JournalEntry(
            timestamp="2026-04-02T10:00",
            done=["task A"],
        )
        append_entry(e2, self.jpath)
        self.assertEqual(pending_next_steps(self.jpath), [])


if __name__ == "__main__":
    unittest.main()
