"""Tests for auto-journal synthesis (Tier 1) and enrichment (Tier 3)."""

import hashlib
import json
from unittest.mock import patch, MagicMock

from synapt.recall.core import TranscriptChunk, TranscriptIndex, parse_journal_entries
from synapt.recall.journal import (
    JournalEntry, append_entry, synthesize_journal_stubs, _filter_project_files,
    auto_extract_entry,
)
from synapt.recall.enrich import _parse_llm_response, iter_enrichable_entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(session_id, turn_index, user_text, assistant_text="Response.",
                files_touched=None, timestamp="2026-03-02T14:00:00+00:00"):
    short = session_id[:8]
    return TranscriptChunk(
        id=f"{short}:t{turn_index}",
        session_id=session_id,
        timestamp=timestamp,
        turn_index=turn_index,
        user_text=user_text,
        assistant_text=assistant_text,
        files_touched=files_touched or [],
    )


def _write_journal(tmp_path, entries):
    """Write journal entries to a temporary journal.jsonl file."""
    path = tmp_path / "journal.jsonl"
    lines = [json.dumps(e.to_dict() if isinstance(e, JournalEntry) else e) for e in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _expected_hash(session_id, timestamp):
    return hashlib.sha256(f"{session_id}:{timestamp}".encode()).hexdigest()[:8]


SESSION_A = "aaaaaaaa-1111-0000-0000-000000000000"
SESSION_B = "bbbbbbbb-2222-0000-0000-000000000000"
SESSION_C = "cccccccc-3333-0000-0000-000000000000"


# ===========================================================================
# Tier 1: synthesize_journal_stubs
# ===========================================================================


class TestSynthesizeJournalStubs:

    def _make_sessions(self):
        """Build 3 sessions with 2 turns each."""
        return {
            SESSION_A: [
                _make_chunk(SESSION_A, 0, "Fix the login bug"),
                _make_chunk(SESSION_A, 1, "Thanks!", files_touched=["src/auth.py"]),
            ],
            SESSION_B: [
                _make_chunk(SESSION_B, 0, "Add dark mode to the app"),
                _make_chunk(SESSION_B, 1, "Looks good", files_touched=["src/theme.py", "src/app.css"]),
            ],
            SESSION_C: [
                _make_chunk(SESSION_C, 0, "Help me refactor the database layer"),
                _make_chunk(SESSION_C, 1, "Done", files_touched=["src/db.py"]),
            ],
        }

    def test_synthesize_creates_stubs(self, tmp_path):
        """Sessions without journal entries get auto-stubs."""
        sessions = self._make_sessions()
        # Pre-populate journal with entry for SESSION_A
        existing = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            session_id=SESSION_A,
            focus="Login bug fix",
        )
        journal_path = _write_journal(tmp_path, [existing])

        count = synthesize_journal_stubs(sessions, journal_path)

        assert count == 2  # SESSION_B and SESSION_C

        # Read back and verify
        with open(journal_path) as f:
            all_entries = [json.loads(line) for line in f if line.strip()]
        assert len(all_entries) == 3  # 1 existing + 2 new

    def test_synthesize_sets_auto_true(self, tmp_path):
        sessions = {SESSION_A: [_make_chunk(SESSION_A, 0, "Hello")]}
        journal_path = tmp_path / "journal.jsonl"

        synthesize_journal_stubs(sessions, journal_path)

        with open(journal_path) as f:
            entry = json.loads(f.readline())
        assert entry["auto"] is True
        assert entry["enriched"] is False

    def test_synthesize_focus_from_turn_0(self, tmp_path):
        sessions = {SESSION_A: [
            _make_chunk(SESSION_A, 1, "Follow-up"),
            _make_chunk(SESSION_A, 0, "Fix the login bug"),
        ]}
        journal_path = tmp_path / "journal.jsonl"

        synthesize_journal_stubs(sessions, journal_path)

        with open(journal_path) as f:
            entry = json.loads(f.readline())
        assert entry["focus"] == "Fix the login bug"

    def test_synthesize_truncates_focus(self, tmp_path):
        long_msg = "x" * 300
        sessions = {SESSION_A: [_make_chunk(SESSION_A, 0, long_msg)]}
        journal_path = tmp_path / "journal.jsonl"

        synthesize_journal_stubs(sessions, journal_path)

        with open(journal_path) as f:
            entry = json.loads(f.readline())
        assert len(entry["focus"]) == 200

    def test_synthesize_collects_files(self, tmp_path):
        sessions = {SESSION_A: [
            _make_chunk(SESSION_A, 0, "Hello", files_touched=["a.py", "b.py"]),
            _make_chunk(SESSION_A, 1, "Done", files_touched=["b.py", "c.py"]),
        ]}
        journal_path = tmp_path / "journal.jsonl"

        synthesize_journal_stubs(sessions, journal_path)

        with open(journal_path) as f:
            entry = json.loads(f.readline())
        assert sorted(entry["files_modified"]) == ["a.py", "b.py", "c.py"]

    def test_synthesize_skips_existing(self, tmp_path):
        sessions = {SESSION_A: [_make_chunk(SESSION_A, 0, "Hello")]}
        existing = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            session_id=SESSION_A,
            focus="Already exists",
        )
        journal_path = _write_journal(tmp_path, [existing])

        count = synthesize_journal_stubs(sessions, journal_path)

        assert count == 0

    def test_synthesize_idempotent(self, tmp_path):
        sessions = {SESSION_A: [_make_chunk(SESSION_A, 0, "Hello")]}
        journal_path = tmp_path / "journal.jsonl"

        count1 = synthesize_journal_stubs(sessions, journal_path)
        count2 = synthesize_journal_stubs(sessions, journal_path)

        assert count1 == 1
        assert count2 == 0  # Already exists after first run

    def test_synthesize_skips_journal_only_sessions(self, tmp_path):
        """Sessions with only journal chunks (turn_index=-1) are skipped."""
        sessions = {SESSION_A: [
            TranscriptChunk(
                id=f"aaaaaaaa:journal:abc123",
                session_id=SESSION_A,
                timestamp="2026-03-02T14:00:00+00:00",
                turn_index=-1,
                user_text="Session focus: test",
                assistant_text="Done: test",
            ),
        ]}
        journal_path = tmp_path / "journal.jsonl"

        count = synthesize_journal_stubs(sessions, journal_path)

        assert count == 0

    def test_synthesize_empty_journal(self, tmp_path):
        """Works correctly when journal.jsonl doesn't exist yet."""
        sessions = {SESSION_A: [_make_chunk(SESSION_A, 0, "Hello")]}
        journal_path = tmp_path / "journal.jsonl"

        count = synthesize_journal_stubs(sessions, journal_path)

        assert count == 1
        assert journal_path.exists()

    def test_synthesize_skips_empty_timestamps(self, tmp_path):
        """Sessions where all chunks have empty timestamps are skipped."""
        sessions = {SESSION_A: [
            _make_chunk(SESSION_A, 0, "Hello", timestamp=""),
            _make_chunk(SESSION_A, 1, "More", timestamp=""),
        ]}
        journal_path = tmp_path / "journal.jsonl"

        count = synthesize_journal_stubs(sessions, journal_path)

        assert count == 0


# ===========================================================================
# _filter_project_files
# ===========================================================================


class TestFilterProjectFiles:

    def test_keeps_relative_paths(self):
        files = ["src/main.py", "tests/test_foo.py"]
        assert _filter_project_files(files) == ["src/main.py", "tests/test_foo.py"]

    def test_filters_claude_internals(self):
        files = [
            "src/main.py",
            "/Users/me/.claude/plans/plan.md",
            "/Users/me/.claude/projects/foo/tool-results/abc.txt",
            "/Users/me/.claude/hooks/pre-commit.sh",
            "/Users/me/.claude/settings.json",
        ]
        assert _filter_project_files(files) == ["src/main.py"]

    def test_filters_tmp_paths(self):
        files = ["src/main.py", "/private/tmp/something.txt"]
        assert _filter_project_files(files) == ["src/main.py"]

    def test_converts_absolute_project_paths_to_relative(self):
        files = ["/home/user/project/src/main.py", "/home/user/project/tests/test.py"]
        result = _filter_project_files(files, project_root="/home/user/project")
        assert result == ["src/main.py", "tests/test.py"]

    def test_filters_other_project_paths(self):
        files = [
            "/home/user/project/src/main.py",
            "/home/user/other-project/src/lib.py",
        ]
        result = _filter_project_files(files, project_root="/home/user/project")
        assert result == ["src/main.py"]

    def test_deduplicates(self):
        files = ["a.py", "b.py", "a.py"]
        assert _filter_project_files(files) == ["a.py", "b.py"]

    def test_empty_and_blank_strings(self):
        files = ["", "  ", "a.py"]
        assert _filter_project_files(files) == ["a.py"]

    def test_accepts_set_input(self):
        files = {"b.py", "a.py"}
        assert _filter_project_files(files) == ["a.py", "b.py"]


class TestSynthesizeFiltersNoisePaths:

    def test_synthesize_filters_claude_paths(self, tmp_path):
        sessions = {SESSION_A: [
            _make_chunk(SESSION_A, 0, "Hello", files_touched=[
                "src/main.py",
                "/Users/me/.claude/plans/plan.md",
                "/private/tmp/result.txt",
            ]),
        ]}
        journal_path = tmp_path / "journal.jsonl"

        synthesize_journal_stubs(sessions, journal_path)

        with open(journal_path) as f:
            entry = json.loads(f.readline())
        assert entry["files_modified"] == ["src/main.py"]


# ===========================================================================
# parse_journal_entries dedup
# ===========================================================================


class TestParseJournalDedup:

    def test_prefers_manual_over_auto(self, tmp_path):
        auto_entry = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            session_id=SESSION_A,
            focus="Auto focus",
            auto=True,
        )
        manual_entry = JournalEntry(
            timestamp="2026-03-02T15:00:00+00:00",
            session_id=SESSION_A,
            focus="Manual focus",
            done=["Fixed the bug"],
        )
        journal_path = _write_journal(tmp_path, [auto_entry, manual_entry])
        chunks = parse_journal_entries(journal_path)

        assert len(chunks) == 1
        assert "Manual focus" in chunks[0].user_text

    def test_prefers_enriched_over_auto(self, tmp_path):
        auto_entry = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            session_id=SESSION_A,
            focus="Auto focus",
            auto=True,
        )
        enriched_entry = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            session_id=SESSION_A,
            focus="Enriched focus",
            done=["Completed refactor"],
            enriched=True,
        )
        journal_path = _write_journal(tmp_path, [auto_entry, enriched_entry])
        chunks = parse_journal_entries(journal_path)

        assert len(chunks) == 1
        assert "Enriched focus" in chunks[0].user_text

    def test_keeps_multiple_manual(self, tmp_path):
        """Multiple manual entries for the same session are all preserved."""
        entry1 = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            session_id=SESSION_A,
            focus="First write",
        )
        entry2 = JournalEntry(
            timestamp="2026-03-02T16:00:00+00:00",
            session_id=SESSION_A,
            focus="Second write",
            done=["More stuff"],
        )
        journal_path = _write_journal(tmp_path, [entry1, entry2])
        chunks = parse_journal_entries(journal_path)

        assert len(chunks) == 2

    def test_auto_field_backward_compat(self, tmp_path):
        """Old journal entries without auto/enriched fields default to False."""
        old_entry = {
            "timestamp": "2026-03-02T14:00:00+00:00",
            "session_id": SESSION_A,
            "focus": "Old entry",
            "done": ["Stuff"],
            "decisions": [],
            "next_steps": [],
            "files_modified": [],
            "git_log": [],
            # No "auto" or "enriched" keys
        }
        journal_path = tmp_path / "journal.jsonl"
        journal_path.write_text(json.dumps(old_entry) + "\n", encoding="utf-8")

        chunks = parse_journal_entries(journal_path)

        assert len(chunks) == 1
        assert "Old entry" in chunks[0].user_text

    def test_auto_only_keeps_most_recent(self, tmp_path):
        """When only auto-stubs exist, keep the most recent."""
        old_auto = JournalEntry(
            timestamp="2026-03-01T10:00:00+00:00",
            session_id=SESSION_A,
            focus="Old auto",
            auto=True,
        )
        new_auto = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            session_id=SESSION_A,
            focus="New auto",
            auto=True,
        )
        journal_path = _write_journal(tmp_path, [old_auto, new_auto])
        chunks = parse_journal_entries(journal_path)

        assert len(chunks) == 1
        assert "New auto" in chunks[0].user_text


# ===========================================================================
# Tier 3: LLM response parsing
# ===========================================================================


class TestParseLlmResponse:

    def test_valid_json(self):
        response = '{"focus": "Fix login", "done": ["Fixed auth"], "decisions": [], "next_steps": []}'
        result = _parse_llm_response(response)
        assert result["focus"] == "Fix login"
        assert result["done"] == ["Fixed auth"]

    def test_markdown_fenced(self):
        response = '```json\n{"focus": "Fix login", "done": ["Fixed"]}\n```'
        result = _parse_llm_response(response)
        assert result["focus"] == "Fix login"

    def test_embedded_json(self):
        response = 'Here is the result:\n{"focus": "Fix login", "done": []}\nDone!'
        result = _parse_llm_response(response)
        assert result["focus"] == "Fix login"

    def test_invalid_returns_none(self):
        assert _parse_llm_response("not json at all") is None
        assert _parse_llm_response("") is None
        assert _parse_llm_response("{broken json") is None


# ===========================================================================
# Tier 3: Enrichable entry iteration
# ===========================================================================


class TestIterEnrichableEntries:

    def test_yields_auto_stubs(self, tmp_path):
        # Auto stub with focus but no done/decisions/next_steps (eligible)
        auto_entry = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            session_id=SESSION_A,
            focus="Fix the login bug",
            files_modified=["src/auth.py"],
            auto=True,
        )
        manual_entry = JournalEntry(
            timestamp="2026-03-02T15:00:00+00:00",
            session_id=SESSION_B,
            focus="Manual",
            done=["Did stuff"],
        )
        enriched_entry = JournalEntry(
            timestamp="2026-03-02T16:00:00+00:00",
            session_id=SESSION_C,
            focus="Enriched",
            done=["More stuff"],
            enriched=True,
        )
        journal_path = _write_journal(tmp_path, [auto_entry, manual_entry, enriched_entry])

        enrichable = list(iter_enrichable_entries(journal_path))

        # Only auto_entry is enrichable (manual has done, enriched already enriched)
        assert len(enrichable) == 1
        assert enrichable[0].session_id == SESSION_A

    def test_skips_auto_with_structured_content(self, tmp_path):
        """Auto entries that already have done/decisions/next_steps aren't re-enriched."""
        entry = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            session_id=SESSION_A,
            focus="Auto focus",
            done=["Already has done"],
            auto=True,
        )
        journal_path = _write_journal(tmp_path, [entry])

        enrichable = list(iter_enrichable_entries(journal_path))
        assert len(enrichable) == 0

    def test_empty_journal(self, tmp_path):
        journal_path = tmp_path / "journal.jsonl"
        journal_path.write_text("", encoding="utf-8")
        assert list(iter_enrichable_entries(journal_path)) == []

    def test_nonexistent_journal(self, tmp_path):
        journal_path = tmp_path / "missing.jsonl"
        assert list(iter_enrichable_entries(journal_path)) == []


# ===========================================================================
# JournalEntry field tests
# ===========================================================================


class TestJournalEntryFields:

    def test_auto_enriched_round_trip(self):
        entry = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            session_id=SESSION_A,
            focus="Test",
            auto=True,
            enriched=True,
        )
        d = entry.to_dict()
        assert d["auto"] is True
        assert d["enriched"] is True

        restored = JournalEntry.from_dict(d)
        assert restored.auto is True
        assert restored.enriched is True

    def test_has_rich_content_auto_stub(self):
        """Auto-stub with only files_modified has no rich content."""
        entry = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            files_modified=["a.py"],
            auto=True,
        )
        assert entry.has_content() is True  # files_modified counts
        assert entry.has_rich_content() is False  # no focus/done/decisions/next

    def test_has_rich_content_with_focus(self):
        entry = JournalEntry(
            timestamp="2026-03-02T14:00:00+00:00",
            focus="Something",
            auto=True,
        )
        assert entry.has_rich_content() is True

    def test_default_auto_false(self):
        entry = JournalEntry(timestamp="2026-03-02T14:00:00+00:00")
        assert entry.auto is False
        assert entry.enriched is False


# ===========================================================================
# Integration: Transcript summary building
# ===========================================================================


class TestBuildTranscriptSummary:

    def test_builds_summary_from_transcript(self, tmp_path):
        """Verify _build_transcript_summary produces readable output."""
        from synapt.recall.enrich import _build_transcript_summary

        # Create a fake transcript file (matching core.py's expected format)
        transcript_data = [
            {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Fix the login bug"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "I'll look into the auth module."}]}},
        ]
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()
        transcript_file = transcript_dir / f"{SESSION_A}.jsonl"
        transcript_file.write_text(
            "\n".join(json.dumps(d) for d in transcript_data) + "\n",
            encoding="utf-8",
        )

        with patch("synapt.recall.enrich.all_worktree_archive_dirs", return_value=[transcript_dir]), \
             patch("synapt.recall.enrich.project_transcript_dir", return_value=None):
            summary = _build_transcript_summary(SESSION_A, tmp_path)

        assert "Fix the login bug" in summary
        assert "auth module" in summary

    def test_returns_empty_for_missing_transcript(self, tmp_path):
        from synapt.recall.enrich import _build_transcript_summary

        with patch("synapt.recall.enrich.all_worktree_archive_dirs", return_value=[]), \
             patch("synapt.recall.enrich.project_transcript_dir", return_value=None):
            summary = _build_transcript_summary("nonexistent-id", tmp_path)

        assert summary == ""


# ===========================================================================
# agent_id attribution on write
# ===========================================================================


class TestJournalAgentIdAttribution:
    """The journal agent_id field must carry a real agent identity or nothing —
    never the session-scoped fallback (s_xxxxxxxx) that channel._agent_id()
    returns for an unregistered session. A session id in agent_id is
    unattributable-by-agent while looking attributed. The session is already
    captured in session_id."""

    def test_session_scoped_fallback_is_not_stored_in_agent_id(
        self, tmp_path, monkeypatch
    ):
        # Pin the identity seam to the session-scoped fallback shape that
        # channel._agent_id() returns for an unregistered session. Doing it at
        # the seam (rather than via env) makes the witness deterministic: it
        # does not depend on the ambient gripspace/GRIPSPACE_ROOT the test runs
        # in, and it reds if the write site ever stores the raw fallback again.
        monkeypatch.setattr("synapt.recall.channel._resolve_griptree", lambda *a, **k: "gt")
        monkeypatch.setattr("synapt.recall.channel._agent_id", lambda *a, **k: "s_deadbeef")
        entry = auto_extract_entry(transcript_path=None, cwd=str(tmp_path))
        assert not entry.agent_id.startswith("s_"), (
            f"session-scoped fallback leaked into agent_id: {entry.agent_id!r}"
        )
        assert entry.agent_id == "", (
            f"expected empty agent_id when only a session fallback is available, "
            f"got {entry.agent_id!r}"
        )

    def test_registered_agent_id_is_preserved(self, tmp_path, monkeypatch):
        # Control: a real registered identity must NOT be dropped by the fix.
        monkeypatch.setattr("synapt.recall.channel._resolve_griptree", lambda *a, **k: "gt")
        monkeypatch.setattr("synapt.recall.channel._agent_id", lambda *a, **k: "apollo-001")
        entry = auto_extract_entry(transcript_path=None, cwd=str(tmp_path))
        assert entry.agent_id == "apollo-001", (
            f"registered agent id was not preserved: {entry.agent_id!r}"
        )
