"""Tests for journal → TranscriptChunk indexing (parse_journal_entries)."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

from synapt.recall.core import parse_journal_entries, TranscriptChunk, TranscriptIndex


def _write_journal(tmp_path, entries):
    """Write journal entries to a temporary journal.jsonl file."""
    path = tmp_path / "journal.jsonl"
    lines = [json.dumps(e) for e in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _expected_hash(session_id, timestamp):
    """Compute expected chunk ID hash suffix."""
    return hashlib.sha256(f"{session_id}:{timestamp}".encode()).hexdigest()[:8]


FULL_ENTRY = {
    "timestamp": "2026-03-02T14:11:00+00:00",
    "session_id": "abc12345-6789-0000-0000-000000000000",
    "branch": "main",
    "focus": "Journal and reminders feature",
    "done": ["Built journal.py", "Added CLI subcommands", "44 tests passing"],
    "decisions": ["JSONL for journal", "JSON for reminders"],
    "next_steps": ["Merge PR #191", "Test hooks end-to-end"],
    "files_modified": ["src/synapt/recall/journal.py", "src/synapt/recall/server.py"],
    "git_log": ["bba239f feat: session journal"],
}

FULL_ENTRY_HASH = _expected_hash(FULL_ENTRY["session_id"], FULL_ENTRY["timestamp"])


class TestParseJournalEntries:

    def test_basic_roundtrip(self, tmp_path):
        path = _write_journal(tmp_path, [FULL_ENTRY])
        chunks = parse_journal_entries(path)

        assert len(chunks) == 1
        c = chunks[0]
        assert isinstance(c, TranscriptChunk)
        assert c.turn_index == -1
        assert c.session_id == FULL_ENTRY["session_id"]
        assert c.timestamp == FULL_ENTRY["timestamp"]

    def test_ambient_agent_identity_does_not_attribute_journal(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SYNAPT_AGENT_ID", "current-runtime-agent")
        path = _write_journal(tmp_path, [FULL_ENTRY])

        chunks = parse_journal_entries(path)

        assert chunks[0].agent_id is None

    def test_user_text_from_focus(self, tmp_path):
        path = _write_journal(tmp_path, [FULL_ENTRY])
        chunks = parse_journal_entries(path)

        assert chunks[0].user_text == "Session focus: Journal and reminders feature"

    def test_empty_focus_gives_empty_user_text(self, tmp_path):
        entry = {**FULL_ENTRY, "focus": ""}
        path = _write_journal(tmp_path, [entry])
        chunks = parse_journal_entries(path)

        assert chunks[0].user_text == ""

    def test_assistant_text_has_done(self, tmp_path):
        path = _write_journal(tmp_path, [FULL_ENTRY])
        chunks = parse_journal_entries(path)

        assert "Done:" in chunks[0].assistant_text
        assert "Built journal.py" in chunks[0].assistant_text

    def test_assistant_text_has_decisions(self, tmp_path):
        path = _write_journal(tmp_path, [FULL_ENTRY])
        chunks = parse_journal_entries(path)

        assert "Decisions:" in chunks[0].assistant_text
        assert "JSONL for journal" in chunks[0].assistant_text

    def test_assistant_text_has_next_steps(self, tmp_path):
        path = _write_journal(tmp_path, [FULL_ENTRY])
        chunks = parse_journal_entries(path)

        assert "Next steps:" in chunks[0].assistant_text
        assert "Merge PR #191" in chunks[0].assistant_text

    def test_assistant_text_has_branch(self, tmp_path):
        path = _write_journal(tmp_path, [FULL_ENTRY])
        chunks = parse_journal_entries(path)

        assert "Branch: main" in chunks[0].assistant_text

    def test_chunk_id_content_based(self, tmp_path):
        """Chunk ID uses hash of session_id + timestamp, not line number."""
        path = _write_journal(tmp_path, [FULL_ENTRY])
        chunks = parse_journal_entries(path)

        assert chunks[0].id == f"abc12345:journal:{FULL_ENTRY_HASH}"

    def test_chunk_id_stable_across_file_edits(self, tmp_path):
        """Adding/removing lines shouldn't change existing entry IDs."""
        path = _write_journal(tmp_path, [FULL_ENTRY])
        id_before = parse_journal_entries(path)[0].id

        # Prepend another entry — in line-index scheme this would shift IDs
        entry2 = {**FULL_ENTRY, "timestamp": "2026-03-01T10:00:00+00:00",
                   "session_id": "zzz00000-0000-0000-0000-000000000000"}
        path = _write_journal(tmp_path, [entry2, FULL_ENTRY])
        chunks = parse_journal_entries(path)

        id_after = [c for c in chunks if c.session_id == FULL_ENTRY["session_id"]][0].id
        assert id_before == id_after

    def test_chunk_id_fallback_to_date(self, tmp_path):
        entry = {**FULL_ENTRY, "session_id": ""}
        path = _write_journal(tmp_path, [entry])
        chunks = parse_journal_entries(path)

        expected_hash = _expected_hash("", FULL_ENTRY["timestamp"])
        assert chunks[0].id == f"2026-03-02:journal:{expected_hash}"

    def test_session_id_fallback(self, tmp_path):
        entry = {**FULL_ENTRY, "session_id": ""}
        path = _write_journal(tmp_path, [entry])
        chunks = parse_journal_entries(path)

        assert chunks[0].session_id == f"journal-{FULL_ENTRY['timestamp']}"

    def test_files_touched_from_files_modified(self, tmp_path):
        path = _write_journal(tmp_path, [FULL_ENTRY])
        chunks = parse_journal_entries(path)

        assert chunks[0].files_touched == [
            "src/synapt/recall/journal.py",
            "src/synapt/recall/server.py",
        ]

    def test_searchable_text_contains_all_fields(self, tmp_path):
        path = _write_journal(tmp_path, [FULL_ENTRY])
        chunks = parse_journal_entries(path)

        text = chunks[0].text
        assert "Journal and reminders feature" in text
        assert "Built journal.py" in text
        assert "JSONL for journal" in text
        assert "Merge PR #191" in text
        assert "src/synapt/recall/journal.py" in text

    def test_multiple_entries(self, tmp_path):
        entry2 = {
            "timestamp": "2026-03-01T10:00:00+00:00",
            "session_id": "def67890-1234-0000-0000-000000000000",
            "focus": "Secret scrubbing",
            "done": ["Implemented scrub.py"],
            "decisions": [],
            "next_steps": [],
            "files_modified": [],
        }
        path = _write_journal(tmp_path, [FULL_ENTRY, entry2])
        chunks = parse_journal_entries(path)

        assert len(chunks) == 2
        assert chunks[0].id.startswith("abc12345:journal:")
        assert chunks[1].id.startswith("def67890:journal:")
        assert chunks[0].id != chunks[1].id

    def test_empty_file(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        path.write_text("", encoding="utf-8")
        chunks = parse_journal_entries(path)

        assert chunks == []

    def test_nonexistent_file(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        chunks = parse_journal_entries(path)

        assert chunks == []

    def test_malformed_json_lines_skipped(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        path.write_text(
            "not valid json\n" + json.dumps(FULL_ENTRY) + "\n",
            encoding="utf-8",
        )
        chunks = parse_journal_entries(path)

        assert len(chunks) == 1
        assert chunks[0].id == f"abc12345:journal:{FULL_ENTRY_HASH}"

    def test_entry_without_content_skipped(self, tmp_path):
        empty_entry = {
            "timestamp": "2026-03-01T10:00:00+00:00",
            "session_id": "empty000",
            "focus": "",
            "done": [],
            "decisions": [],
            "next_steps": [],
            "files_modified": [],
        }
        path = _write_journal(tmp_path, [empty_entry, FULL_ENTRY])
        chunks = parse_journal_entries(path)

        assert len(chunks) == 1
        assert chunks[0].id == f"abc12345:journal:{FULL_ENTRY_HASH}"

    def test_turn_index_sentinel(self, tmp_path):
        path = _write_journal(tmp_path, [FULL_ENTRY])
        chunks = parse_journal_entries(path)

        assert chunks[0].turn_index == -1

    def test_tools_used_empty(self, tmp_path):
        path = _write_journal(tmp_path, [FULL_ENTRY])
        chunks = parse_journal_entries(path)

        assert chunks[0].tools_used == []

    def test_same_session_different_timestamps_unique_ids(self, tmp_path):
        """Multiple entries from the same session with different timestamps get unique IDs."""
        entry1 = {**FULL_ENTRY, "timestamp": "2026-03-02T14:11:00+00:00"}
        entry2 = {**FULL_ENTRY, "timestamp": "2026-03-02T15:30:00+00:00"}
        path = _write_journal(tmp_path, [entry1, entry2])
        chunks = parse_journal_entries(path)

        assert len(chunks) == 2
        assert chunks[0].id != chunks[1].id


class TestJournalInIndex:
    """Integration tests for journal chunks inside TranscriptIndex."""

    def _make_transcript_chunk(self, session_id, turn_index, user_text):
        return TranscriptChunk(
            id=f"{session_id[:8]}:t{turn_index}",
            session_id=session_id,
            timestamp="2026-03-02T14:00:00+00:00",
            turn_index=turn_index,
            user_text=user_text,
            assistant_text="Some response.",
        )

    def test_list_sessions_prefers_transcript_first_message(self):
        """Journal focus should NOT override the real first user message."""
        session_id = FULL_ENTRY["session_id"]
        transcript_chunk = self._make_transcript_chunk(
            session_id, 0, "How do I fix the login bug?"
        )
        journal_chunk = TranscriptChunk(
            id=f"abc12345:journal:{FULL_ENTRY_HASH}",
            session_id=session_id,
            timestamp=FULL_ENTRY["timestamp"],
            turn_index=-1,
            user_text="Session focus: Login bug investigation",
            assistant_text="Done: Fixed login bug",
        )
        index = TranscriptIndex([journal_chunk, transcript_chunk])
        sessions = index.list_sessions()

        assert len(sessions) == 1
        assert sessions[0]["first_message"] == "How do I fix the login bug?"

    def test_list_sessions_falls_back_to_journal_if_no_transcript(self):
        """Journal-only session should still show something."""
        journal_chunk = TranscriptChunk(
            id=f"abc12345:journal:{FULL_ENTRY_HASH}",
            session_id=FULL_ENTRY["session_id"],
            timestamp=FULL_ENTRY["timestamp"],
            turn_index=-1,
            user_text="Session focus: Exploratory session",
            assistant_text="Done: Explored the codebase",
        )
        index = TranscriptIndex([journal_chunk])
        sessions = index.list_sessions()

        assert len(sessions) == 1
        assert "Exploratory session" in sessions[0]["first_message"]

    def test_turn_lookup_excludes_journal(self):
        """Journal chunks should not appear in _turn_lookup."""
        session_id = FULL_ENTRY["session_id"]
        journal_chunk = TranscriptChunk(
            id=f"abc12345:journal:{FULL_ENTRY_HASH}",
            session_id=session_id,
            timestamp=FULL_ENTRY["timestamp"],
            turn_index=-1,
            user_text="Session focus: Test",
            assistant_text="Done: Test",
        )
        index = TranscriptIndex([journal_chunk])

        assert (session_id, -1) not in index._turn_lookup

    def test_format_results_journal_header(self):
        """Journal chunks should display 'journal' not 'turn -1' in headers."""
        journal_chunk = TranscriptChunk(
            id=f"abc12345:journal:{FULL_ENTRY_HASH}",
            session_id=FULL_ENTRY["session_id"],
            timestamp=FULL_ENTRY["timestamp"],
            turn_index=-1,
            user_text="Session focus: Journal test",
            assistant_text="Decisions: Use FTS5 for search",
        )
        index = TranscriptIndex([journal_chunk])
        result = index.lookup("FTS5 search", max_chunks=1)

        assert "journal" in result
        assert "turn -1" not in result

    def test_sqlite_roundtrip_preserves_turn_index(self, tmp_path):
        """Journal chunks with turn_index=-1 survive SQLite save/load."""
        from synapt.recall.storage import RecallDB

        journal_chunk = TranscriptChunk(
            id=f"abc12345:journal:{FULL_ENTRY_HASH}",
            session_id=FULL_ENTRY["session_id"],
            timestamp=FULL_ENTRY["timestamp"],
            turn_index=-1,
            user_text="Session focus: SQLite test",
            assistant_text="Done: Verified round-trip",
            files_touched=["core.py"],
        )

        db = RecallDB(tmp_path / "test.db")
        db.save_chunks([journal_chunk])
        loaded = db.load_chunks()

        assert len(loaded) == 1
        assert loaded[0].turn_index == -1
        assert loaded[0].id == journal_chunk.id
        assert loaded[0].files_touched == ["core.py"]

    def test_list_sessions_turn_count_excludes_journal(self):
        """turn_count should only count transcript turns, not journal entries."""
        session_id = FULL_ENTRY["session_id"]
        transcript_chunk = self._make_transcript_chunk(
            session_id, 0, "Hello"
        )
        journal_chunk = TranscriptChunk(
            id=f"abc12345:journal:{FULL_ENTRY_HASH}",
            session_id=session_id,
            timestamp=FULL_ENTRY["timestamp"],
            turn_index=-1,
            user_text="Session focus: Test",
            assistant_text="Done: Test",
        )
        index = TranscriptIndex([journal_chunk, transcript_chunk])
        sessions = index.list_sessions()

        assert sessions[0]["turn_count"] == 1  # Only the transcript turn

    def test_status_query_surfaces_next_steps_in_summary_mode(self):
        """Pending-work queries should route to journal next_steps cleanly."""
        next_steps_chunk = TranscriptChunk(
            id="sessnext:journal:11111111",
            session_id="sess-next",
            timestamp="2026-03-02T15:00:00+00:00",
            turn_index=-1,
            user_text="Session focus: Finish sprint work",
            assistant_text="Next steps: Merge PR #416 - Check rollout status",
        )
        decision_chunk = TranscriptChunk(
            id="sessdone:journal:22222222",
            session_id="sess-done",
            timestamp="2026-03-01T10:00:00+00:00",
            turn_index=-1,
            user_text="Session focus: Prior architecture discussion",
            assistant_text="Decisions: Use JSONL for journaling",
        )
        index = TranscriptIndex([decision_chunk, next_steps_chunk], use_embeddings=False)

        result = index.lookup("what's pending", max_chunks=2, depth="summary")

        assert "Next steps:" in result
        assert "Merge PR #416" in result
        assert "Decisions: Use JSONL for journaling" not in result

    def test_status_query_prefers_latest_next_steps_journal(self):
        """Status queries should prioritize the newest pending journal."""
        now = datetime.now(timezone.utc)
        older = TranscriptChunk(
            id="sessold:journal:11111111",
            session_id="sess-old",
            timestamp=(now - timedelta(days=14)).isoformat(),
            turn_index=-1,
            user_text="Session focus: Earlier sprint work",
            assistant_text="Next steps: Finish old migration plan",
        )
        latest = TranscriptChunk(
            id="sessnew:journal:22222222",
            session_id="sess-new",
            timestamp=now.isoformat(),
            turn_index=-1,
            user_text="Session focus: Current sprint work",
            assistant_text="Next steps: Merge the current PR",
        )
        index = TranscriptIndex([older, latest], use_embeddings=False)

        result = index.lookup("what's pending", max_chunks=1, depth="summary")

        assert "Merge the current PR" in result
        assert "Finish old migration plan" in result
        assert result.index("Merge the current PR") < result.index("Finish old migration plan")
