"""Tests for synapt.recall.codex — Codex CLI transcript parsing."""

import json
import os
import tempfile
import unittest
from unittest import mock

from _isolation_helpers import owned_store
from pathlib import Path

from synapt.recall.codex import (
    parse_codex_transcript,
    list_codex_transcripts,
    archive_codex_transcripts,
    is_codex_transcript,
    _extract_file_paths,
)
from synapt.recall.core import build_index
from synapt.recall.codex import _has_buildable_transcripts
from synapt.recall.journal import auto_extract_entry, extract_session_id


def _write_codex_transcript(tmpdir: str, entries: list[dict], name: str = "rollout-test.jsonl") -> Path:
    """Write a Codex-format JSONL file and return its path."""
    path = Path(tmpdir) / name
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


class TestParseCodexTranscript(unittest.TestCase):
    """Test Codex transcript parsing into TranscriptChunks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_basic_user_assistant_turn(self):
        """A simple user→assistant conversation produces one chunk."""
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "test-session-001"}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": "what is 2+2?"}
             ]}},
            {"timestamp": "2026-03-01T10:00:02Z", "type": "response_item",
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": "2+2 is 4."}
             ]}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries)
        chunks = parse_codex_transcript(path)

        self.assertEqual(len(chunks), 1)
        self.assertIn("2+2", chunks[0].user_text)
        self.assertIn("4", chunks[0].assistant_text)
        self.assertEqual(chunks[0].session_id, "test-session-001")

    def test_multiple_turns(self):
        """Multiple user messages produce multiple chunks."""
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "multi-turn-session"}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": "first question"}
             ]}},
            {"timestamp": "2026-03-01T10:00:02Z", "type": "response_item",
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": "first answer"}
             ]}},
            {"timestamp": "2026-03-01T10:00:03Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": "second question"}
             ]}},
            {"timestamp": "2026-03-01T10:00:04Z", "type": "response_item",
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": "second answer"}
             ]}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries)
        chunks = parse_codex_transcript(path)

        self.assertEqual(len(chunks), 2)
        self.assertIn("first question", chunks[0].user_text)
        self.assertIn("second question", chunks[1].user_text)

    def test_tool_calls_detected(self):
        """Function calls are captured as tools_used."""
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "tool-session"}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": "list files"}
             ]}},
            {"timestamp": "2026-03-01T10:00:02Z", "type": "response_item",
             "payload": {"type": "function_call", "name": "exec_command",
                          "arguments": '{"cmd":"ls -la /tmp"}', "call_id": "call_1"}},
            {"timestamp": "2026-03-01T10:00:03Z", "type": "response_item",
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": "Here are the files."}
             ]}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries)
        chunks = parse_codex_transcript(path)

        self.assertEqual(len(chunks), 1)
        self.assertIn("exec_command", chunks[0].tools_used)
        self.assertIn("ls -la", chunks[0].tool_content)

    def test_custom_tool_families_are_captured_without_duplicate_agent_text(self):
        """Custom tool envelopes retain tool context and deduplicate agent text."""
        synthetic_file = "/workspace/synthetic.py"
        oversized_input = "{" + '"payload":"' + ("x" * 600) + '"}'
        legacy_oversized_args = "{" + '"payload":"' + ("y" * 600) + '"}'
        assistant_text = "Synthetic assistant response."
        unique_agent_text = "Synthetic event response."
        commentary_agent_text = "Synthetic commentary event."
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "custom-tool-session"}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": "inspect synthetic source"}
             ]}},
            {"timestamp": "2026-03-01T10:00:02Z", "type": "response_item",
             "payload": {"type": "custom_tool_call", "name": "shell_run",
                         "arguments": json.dumps({"cmd": f"cat {synthetic_file}"})}},
            {"timestamp": "2026-03-01T10:00:03Z", "type": "response_item",
             "payload": {"type": "custom_tool_call", "name": "large_input_tool",
                         "input": oversized_input}},
            {"timestamp": "2026-03-01T10:00:04Z", "type": "response_item",
             "payload": {"type": "custom_tool_call_output", "output": "ignored"}},
            {"timestamp": "2026-03-01T10:00:05Z", "type": "response_item",
             "payload": {"type": "function_call", "name": "legacy_large_tool",
                         "arguments": legacy_oversized_args}},
            # Agent-message delivery leads the response item in production.
            {"timestamp": "2026-03-01T10:00:06Z", "type": "event_msg",
             "payload": {"type": "agent_message", "phase": "final_answer",
                         "message": assistant_text}},
            {"timestamp": "2026-03-01T10:00:07Z", "type": "response_item",
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": assistant_text}
             ]}},
            {"timestamp": "2026-03-01T10:00:08Z", "type": "event_msg",
             "payload": {"type": "agent_message", "message": unique_agent_text}},
            {"timestamp": "2026-03-01T10:00:09Z", "type": "event_msg",
             "payload": {"type": "agent_message", "phase": "final_answer",
                         "message": assistant_text}},
            {"timestamp": "2026-03-01T10:00:10Z", "type": "event_msg",
             "payload": {"type": "agent_message", "phase": "commentary",
                         "message": commentary_agent_text}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries)

        chunks = parse_codex_transcript(path)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(
            chunk.tools_used,
            ["shell_run", "large_input_tool", "legacy_large_tool"],
        )
        self.assertIn(f"[shell_run] cat {synthetic_file}", chunk.tool_content)
        self.assertIn(synthetic_file, chunk.files_touched)
        self.assertIn("[large_input_tool]", chunk.tool_content)
        self.assertNotIn("[legacy_large_tool]", chunk.tool_content)
        self.assertEqual(chunk.assistant_text.count(assistant_text), 1)
        self.assertEqual(chunk.assistant_text.count(unique_agent_text), 1)
        self.assertNotIn(commentary_agent_text, chunk.assistant_text)
        self.assertEqual(chunk.commentary_text, commentary_agent_text)

    def test_custom_tool_summary_budget_keeps_all_tool_names(self):
        """Custom call summaries are bounded without dropping tool identities."""
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "custom-summary-budget"}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": "inspect synthetic tools"}
             ]}},
        ]
        for index in range(10):
            entries.append(
                {"timestamp": f"2026-03-01T10:00:{index + 2:02d}Z", "type": "response_item",
                 "payload": {"type": "custom_tool_call", "name": f"synthetic_tool_{index}",
                             "input": "x" * 300}},
            )
        path = _write_codex_transcript(self.tmpdir, entries)

        chunks = parse_codex_transcript(path)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.tools_used, [f"synthetic_tool_{index}" for index in range(10)])
        self.assertEqual(chunk.tool_content.count("[synthetic_tool_"), 4)
        self.assertIn("+6 more tool calls", chunk.tool_content)

    def test_skips_system_content(self):
        """Developer role and permissions/env context are filtered out."""
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "filter-session"}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "developer", "content": [
                 {"type": "input_text", "text": "You are Codex, a coding agent."}
             ]}},
            {"timestamp": "2026-03-01T10:00:02Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": "<permissions instructions>sandbox</permissions instructions>"},
                 {"type": "input_text", "text": "<environment_context>stuff</environment_context>"},
                 {"type": "input_text", "text": "actual user question"},
             ]}},
            {"timestamp": "2026-03-01T10:00:03Z", "type": "response_item",
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": "answer"}
             ]}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries)
        chunks = parse_codex_transcript(path)

        self.assertEqual(len(chunks), 1)
        self.assertNotIn("permissions", chunks[0].user_text)
        self.assertNotIn("environment_context", chunks[0].user_text)
        self.assertNotIn("Codex", chunks[0].user_text)
        self.assertIn("actual user question", chunks[0].user_text)

    def test_skips_commentary_phase(self):
        """Response-item commentary is segregated from primary assistant text."""
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "commentary-session"}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": "fix the bug"}
             ]}},
            {"timestamp": "2026-03-01T10:00:02Z", "type": "response_item",
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": "Looking at the code..."}
             ], "phase": "commentary"}},
            {"timestamp": "2026-03-01T10:00:03Z", "type": "response_item",
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": "Fixed the null pointer."}
             ]}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries)
        chunks = parse_codex_transcript(path)

        self.assertEqual(len(chunks), 1)
        self.assertNotIn("Looking at", chunks[0].assistant_text)
        self.assertIn("Fixed the null pointer", chunks[0].assistant_text)
        self.assertEqual(chunks[0].commentary_text, "Looking at the code...")
        restored = type(chunks[0]).from_dict(chunks[0].to_dict())
        self.assertEqual(restored.commentary_text, "Looking at the code...")

    def test_commentary_only_turn_does_not_create_a_chunk(self):
        """Commentary without a user or final response remains non-flushing."""
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "commentary-only-session"}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "assistant", "phase": "commentary", "content": [
                 {"type": "output_text", "text": "Synthetic commentary only."}
             ]}},
            {"timestamp": "2026-03-01T10:00:02Z", "type": "event_msg",
             "payload": {"type": "agent_message", "phase": "commentary",
                         "message": "Synthetic commentary event only."}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries)

        self.assertEqual(parse_codex_transcript(path), [])

    def test_commentary_text_is_capped(self):
        """Commentary storage has the same per-turn bound as assistant text."""
        commentary = "c" * 5001
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "commentary-cap-session"}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": "retain commentary safely"}
             ]}},
            {"timestamp": "2026-03-01T10:00:02Z", "type": "response_item",
             "payload": {"role": "assistant", "phase": "commentary", "content": [
                 {"type": "output_text", "text": commentary}
             ]}},
            {"timestamp": "2026-03-01T10:00:03Z", "type": "response_item",
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": "Synthetic final answer."}
             ]}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries)

        chunks = parse_codex_transcript(path)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].commentary_text), 5003)
        self.assertTrue(chunks[0].commentary_text.endswith("..."))

    def test_dedup_by_session_id(self):
        """Same session ID parsed twice returns empty on second call."""
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "dedup-session"}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": "hello"}
             ]}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries)
        seen = set()
        chunks1 = parse_codex_transcript(path, seen_uuids=seen)
        chunks2 = parse_codex_transcript(path, seen_uuids=seen)

        self.assertEqual(len(chunks1), 1)
        self.assertEqual(len(chunks2), 0)

    def test_empty_file(self):
        """Empty file returns no chunks."""
        path = Path(self.tmpdir) / "rollout-empty.jsonl"
        path.touch()
        chunks = parse_codex_transcript(path)
        self.assertEqual(chunks, [])

    def test_event_msg_user_message(self):
        """User messages via event_msg are also captured."""
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "event-session"}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "event_msg",
             "payload": {"type": "user_message", "message": "via event_msg"}},
            {"timestamp": "2026-03-01T10:00:02Z", "type": "response_item",
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": "got it"}
             ]}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries)
        chunks = parse_codex_transcript(path)

        self.assertEqual(len(chunks), 1)
        self.assertIn("via event_msg", chunks[0].user_text)

    def test_journal_helpers_support_codex_transcript(self):
        local_file = str(Path(self.tmpdir) / "example.py")
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "journal-codex-session", "cwd": self.tmpdir}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": f"inspect {local_file}"}
             ]}},
            {"timestamp": "2026-03-01T10:00:02Z", "type": "response_item",
             "payload": {"type": "function_call", "name": "exec_command",
                          "arguments": json.dumps({"cmd": f"sed -n 1,20p {local_file}"}), "call_id": "call_1"}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries, name="rollout-journal.jsonl")

        self.assertEqual(extract_session_id(path), "journal-codex-session")
        entry = auto_extract_entry(transcript_path=path, cwd=self.tmpdir)
        self.assertEqual(entry.session_id, "journal-codex-session")
        self.assertIn("example.py", entry.files_modified)


class TestListCodexTranscripts(unittest.TestCase):
    """Test transcript discovery."""

    def test_finds_rollout_files(self):
        tmpdir = tempfile.mkdtemp()
        sessions = Path(tmpdir) / "2026" / "03" / "01"
        sessions.mkdir(parents=True)
        (sessions / "rollout-test1.jsonl").touch()
        (sessions / "rollout-test2.jsonl").touch()
        (sessions / "other-file.jsonl").touch()  # Should not match

        found = list_codex_transcripts(Path(tmpdir))
        self.assertEqual(len(found), 2)
        self.assertTrue(all("rollout-" in p.name for p in found))

    def test_old_date_path_still_discovers_live_appended_rollout(self):
        """Discovery scans all rollout paths because a path date is start-order."""
        tmpdir = tempfile.mkdtemp()
        sessions_root = Path(tmpdir) / "sessions"
        old_date = sessions_root / "2001" / "01" / "01"
        old_date.mkdir(parents=True)
        path = _write_codex_transcript(
            str(old_date),
            [
                {"timestamp": "2001-01-01T10:00:00Z", "type": "session_meta",
                 "payload": {"id": "long-lived-session"}},
                {"timestamp": "2001-01-01T10:00:01Z", "type": "response_item",
                 "payload": {"role": "user", "content": [
                     {"type": "input_text", "text": "initial request"}
                 ]}},
            ],
            name="rollout-long-lived.jsonl",
        )

        self.assertEqual(list_codex_transcripts(sessions_root), [path])
        parse_codex_transcript(path)

        with path.open("a", encoding="utf-8") as transcript:
            transcript.write(json.dumps({
                "timestamp": "2026-08-07T10:00:00Z",
                "type": "response_item",
                "payload": {"role": "assistant", "content": [
                    {"type": "output_text", "text": "appended live response"}
                ]},
            }) + "\n")

        discovered = list_codex_transcripts(sessions_root)
        reparsed = parse_codex_transcript(discovered[0])
        self.assertEqual(discovered, [path])
        self.assertIn("appended live response", reparsed[0].assistant_text)

    def test_empty_dir(self):
        tmpdir = tempfile.mkdtemp()
        found = list_codex_transcripts(Path(tmpdir))
        self.assertEqual(found, [])

    def test_filters_to_project_scope(self):
        tmpdir = tempfile.mkdtemp()
        sessions = Path(tmpdir) / "2026" / "03" / "01"
        sessions.mkdir(parents=True)

        project_root = Path(tmpdir) / "project"
        project_root.mkdir()
        other_root = Path(tmpdir) / "other-project"
        other_root.mkdir()

        matching = _write_codex_transcript(
            str(sessions),
            [{"type": "session_meta", "payload": {"id": "match", "cwd": str(project_root / "subdir")}}],
            name="rollout-match.jsonl",
        )
        _write_codex_transcript(
            str(sessions),
            [{"type": "session_meta", "payload": {"id": "miss", "cwd": str(other_root)}}],
            name="rollout-miss.jsonl",
        )

        found = list_codex_transcripts(Path(tmpdir), project_dir=project_root)
        self.assertEqual(found, [matching])

    def test_archive_codex_transcripts_filters_to_project(self):
        tmpdir = tempfile.mkdtemp()
        sessions = Path(tmpdir) / "2026" / "03" / "01"
        sessions.mkdir(parents=True)

        project_root = Path(tmpdir) / "project"
        project_root.mkdir()
        archive_root = project_root / ".synapt" / "recall" / "worktrees" / "project" / "transcripts"
        archive_root.mkdir(parents=True)
        other_root = Path(tmpdir) / "other-project"
        other_root.mkdir()

        matching = _write_codex_transcript(
            str(sessions),
            [{"type": "session_meta", "payload": {"id": "match", "cwd": str(project_root)}}],
            name="rollout-match.jsonl",
        )
        _write_codex_transcript(
            str(sessions),
            [{"type": "session_meta", "payload": {"id": "miss", "cwd": str(other_root)}}],
            name="rollout-miss.jsonl",
        )

        copied = archive_codex_transcripts(project_root, sessions_dir=Path(tmpdir))
        self.assertEqual([p.name for p in copied], [matching.name])
        self.assertTrue((archive_root / matching.name).exists())

    def test_build_index_parses_codex_archived_file(self):
        tmpdir = tempfile.mkdtemp()
        entries = [
            {"timestamp": "2026-03-01T10:00:00Z", "type": "session_meta",
             "payload": {"id": "build-codex-session", "cwd": tmpdir}},
            {"timestamp": "2026-03-01T10:00:01Z", "type": "response_item",
             "payload": {"role": "user", "content": [
                 {"type": "input_text", "text": "question from codex"}
             ]}},
            {"timestamp": "2026-03-01T10:00:02Z", "type": "response_item",
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": "answer from codex"}
             ]}},
        ]
        path = _write_codex_transcript(tmpdir, entries, name="rollout-build.jsonl")

        self.assertTrue(is_codex_transcript(path))
        index = build_index(Path(tmpdir), use_embeddings=False)
        self.assertEqual(len(index.chunks), 1)
        self.assertIn("question from codex", index.chunks[0].user_text)


class TestExtractFilePaths(unittest.TestCase):
    """Test file path extraction from text."""

    def test_extracts_absolute_paths(self):
        paths = _extract_file_paths("editing /src/main.py and /tmp/test.js")
        self.assertIn("/src/main.py", paths)
        self.assertIn("/tmp/test.js", paths)

    def test_extracts_windows_absolute_paths(self):
        paths = _extract_file_paths(r'editing C:\repo\main.py and D:\tmp\test.js')
        self.assertIn(r"C:\repo\main.py", paths)
        self.assertIn(r"D:\tmp\test.js", paths)

    def test_no_false_positives_on_plain_text(self):
        paths = _extract_file_paths("hello world this is a test")
        self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()


class TestCodexOnlyProjectCanBootstrap(unittest.TestCase):
    """A project whose ONLY history is Codex sessions must be buildable.

    The build's "no transcripts found" pre-check counted live Claude transcript
    directories and archived transcripts, and nothing else -- so a codex-only
    project exited before ``archive_codex_transcripts`` ever ran, and the Codex
    sessions that WOULD have satisfied the build were never discovered.

    That is a pre-check disagreeing with the thing it gates: the build could
    have succeeded, and the guard said there was nothing to build. Passing
    ``--source <empty dir>`` routed around the guard and the rest of the path
    archived and indexed the session correctly, which is what proved the defect
    was the guard rather than the ingestion.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project = Path(self.tmpdir) / "project"
        self.project.mkdir()
        self.sessions = Path(self.tmpdir) / "sessions" / "2026" / "08" / "05"
        self.sessions.mkdir(parents=True)

    def test_a_matching_codex_session_counts_as_a_transcript(self):
        _write_codex_transcript(
            str(self.sessions),
            [{"type": "session_meta",
              "payload": {"id": "s1", "cwd": str(self.project / "sub")}}],
            name="rollout-match.jsonl",
        )
        self.assertTrue(
            _has_buildable_transcripts(self.project, sessions_dir=self.sessions),
            "a discoverable Codex session matching the project must satisfy the "
            "pre-check; without this the build refuses work it could do")

    def test_a_session_from_another_project_does_not_count(self):
        # The control. Without it, a pre-check that counted ANY rollout on disk
        # would pass this class while letting an unrelated project's sessions
        # authorise a build that then finds nothing.
        other = Path(self.tmpdir) / "other"
        other.mkdir()
        _write_codex_transcript(
            str(self.sessions),
            [{"type": "session_meta",
              "payload": {"id": "s2", "cwd": str(other / "sub")}}],
            name="rollout-other.jsonl",
        )
        self.assertFalse(
            _has_buildable_transcripts(self.project, sessions_dir=self.sessions))

    def test_no_sessions_at_all_still_reports_nothing_to_build(self):
        self.assertFalse(
            _has_buildable_transcripts(self.project, sessions_dir=self.sessions))


class TestTheBuildPreCheckReadsTheCodexArm(unittest.TestCase):
    """The predicate existing is not the fix -- the fix is that cmd_build READS it.

    A guard whose result nothing consumes does not exist, so this drives the real
    pre-check branch rather than poking the helper.
    """

    def setUp(self):
        # cmd_build resolves the recall data root implicitly, so without an
        # owned root this drives the pre-check against a live store (Ref #967).
        # unittest.TestCase cannot take the shared fixture, so it uses the
        # shared helper — this was a hand-rolled two-variable save/restore
        # until now, which is exactly the duplication owned_store exists to
        # remove. Consolidated: one restore path, not a per-class copy.
        #
        # recall#1125's cache configuration calls project_index_dir(project)
        # with an EXPLICIT project_dir (cmd_build's own project = Path.cwd()),
        # and an explicit project_dir suppresses every env override by design
        # — owned_store()'s SYNAPT_RECALL_ROOT never reaches it. Per
        # owned_store's own docstring ("a test genuinely asserting inference
        # should chdir to an owned directory so the inference still runs,
        # just from a starting point it owns"), chdir into the owned root so
        # even an explicit-project_dir resolution lands under it.
        self._store = owned_store()
        self._orig_cwd = os.getcwd()
        os.chdir(self._store.root)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._store.restore()

    def _run_precheck(self, has_codex: bool):
        """Drive cmd_build's pre-check with everything downstream stubbed."""
        import argparse
        from unittest import mock
        from synapt.recall import cli

        args = argparse.Namespace(
            source=None, hf=None, chatgpt_archive=None,
            no_embeddings=True, incremental=False,
        )
        fake_index = mock.Mock()
        fake_index.stats.return_value = {"chunk_count": 1, "session_count": 1}
        with mock.patch.object(cli, "project_transcript_dirs", return_value=[]), \
             mock.patch.object(cli, "all_worktree_archive_dirs", return_value=[]), \
             mock.patch.object(cli, "_check_legacy_index", return_value=None), \
             mock.patch.object(cli, "_archive_and_build", return_value=fake_index), \
             mock.patch("synapt.recall.codex._has_buildable_transcripts",
                        return_value=has_codex):
            try:
                cli.cmd_build(args)
            except SystemExit as exc:
                return int(exc.code or 0)
        return 0

    def test_codex_only_project_is_allowed_to_build(self):
        self.assertEqual(
            self._run_precheck(has_codex=True), 0,
            "the pre-check still refuses a codex-only project; the helper is "
            "not being consulted at the real call site")

    def test_a_project_with_nothing_at_all_still_exits(self):
        # The control. Without it, deleting the guard entirely would satisfy the
        # test above while letting an empty project proceed to a no-op build.
        self.assertEqual(self._run_precheck(has_codex=False), 1)


class TestSessionCwdCache(unittest.TestCase):
    """recall#1125: _session_cwd's opt-in, disk-backed (path,size,mtime) cache.

    Every test resets the module-global cache state directly (not through
    the public configure_cwd_cache(None), which leaves an active in-memory
    cache rather than disabling it) so no test's cache state can leak into
    another test in this file that calls _session_cwd unaware of caching.
    """

    def setUp(self):
        import synapt.recall.codex as codex_module

        self._codex_module = codex_module
        self._reset_cache_state()

    def tearDown(self):
        self._reset_cache_state()

    def _reset_cache_state(self):
        self._codex_module._cwd_cache = None
        self._codex_module._cwd_cache_dirty = False
        self._codex_module._cwd_cache_index_dir = None

    def test_disabled_by_default_reads_every_call(self):
        """No configure_cwd_cache call: behavior is unchanged from before
        the cache existed — every _session_cwd call re-reads the file."""
        tmpdir = tempfile.mkdtemp()
        path = _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "s", "cwd": tmpdir}}],
        )
        from synapt.recall.codex import _session_cwd

        with mock.patch(
            "synapt.recall.codex._session_cwd_uncached",
            wraps=self._codex_module._session_cwd_uncached,
        ) as spy:
            _session_cwd(path)
            _session_cwd(path)
        self.assertEqual(spy.call_count, 2)

    def test_configured_cache_avoids_reopening_unchanged_file(self):
        """The concrete recall#1125 proof: with the cache configured, a
        second call on an unchanged file does not re-scan it."""
        tmpdir = tempfile.mkdtemp()
        index_dir = Path(tempfile.mkdtemp())
        path = _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "s", "cwd": tmpdir}}],
        )
        from synapt.recall.codex import _session_cwd, configure_cwd_cache

        configure_cwd_cache(index_dir)
        with mock.patch(
            "synapt.recall.codex._session_cwd_uncached",
            wraps=self._codex_module._session_cwd_uncached,
        ) as spy:
            first = _session_cwd(path)
            second = _session_cwd(path)
        self.assertEqual(spy.call_count, 1, "second call re-scanned an unchanged file")
        self.assertEqual(first, second)
        self.assertEqual(first, Path(tmpdir).resolve())

    def test_cache_invalidates_when_file_changes(self):
        """A changed file (new mtime/size) must never be served a stale
        cwd — this is the control proving the cache isn't just short-circuiting."""
        tmpdir = tempfile.mkdtemp()
        other_dir = tempfile.mkdtemp()
        index_dir = Path(tempfile.mkdtemp())
        path = _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "s", "cwd": tmpdir}}],
        )
        from synapt.recall.codex import _session_cwd, configure_cwd_cache

        configure_cwd_cache(index_dir)
        first = _session_cwd(path)
        self.assertEqual(first, Path(tmpdir).resolve())

        # Rewrite with a different recorded cwd. Sleep-free: bump mtime
        # explicitly in case the filesystem's mtime resolution is coarse
        # enough that a fast rewrite could otherwise collide with the
        # cached (size, mtime) key.
        _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "s", "cwd": other_dir + "-x"}}],
            name=path.name,
        )
        new_stat = path.stat()
        os.utime(path, (new_stat.st_atime, new_stat.st_mtime + 5))

        second = _session_cwd(path)
        self.assertEqual(second, Path(other_dir + "-x").resolve())
        self.assertNotEqual(first, second)

    def test_cache_invalidates_on_mtime_alone_when_size_is_unchanged(self):
        """The size-only-key control: the test above rewrites to a LONGER
        cwd string, so a mutant comparing size alone (dropping mtime from
        the cache key) would still pass it — the rewrite changes both. This
        keeps the byte length of the recorded cwd identical (one character
        swapped) so only mtime distinguishes the two versions, and would
        fail against a size-only key even though it passes against the
        real (size, mtime) key.
        """
        tmpdir = tempfile.mkdtemp()
        index_dir = Path(tempfile.mkdtemp())
        first_cwd = str(tmpdir)
        last_char = first_cwd[-1]
        swapped_char = "9" if last_char != "9" else "8"
        second_cwd = first_cwd[:-1] + swapped_char
        self.assertEqual(len(first_cwd), len(second_cwd), "test setup must keep size identical")

        path = _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "s", "cwd": first_cwd}}],
        )
        from synapt.recall.codex import _session_cwd, configure_cwd_cache

        configure_cwd_cache(index_dir)
        first = _session_cwd(path)
        self.assertEqual(first, Path(first_cwd).resolve())

        pre_size = path.stat().st_size
        _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "s", "cwd": second_cwd}}],
            name=path.name,
        )
        post_stat = path.stat()
        self.assertEqual(pre_size, post_stat.st_size, "test setup must keep size identical")
        os.utime(path, (post_stat.st_atime, post_stat.st_mtime + 5))

        second = _session_cwd(path)
        self.assertEqual(second, Path(second_cwd).resolve())
        self.assertNotEqual(first, second)

    def test_cache_persists_across_reconfigure_reads_disk_without_rescanning(self):
        """Two bare `resume` invocations are two separate Python processes:
        the cache must survive via the on-disk file, not just in memory."""
        tmpdir = tempfile.mkdtemp()
        index_dir = Path(tempfile.mkdtemp())
        path = _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "s", "cwd": tmpdir}}],
        )
        from synapt.recall.codex import (
            _session_cwd,
            configure_cwd_cache,
            flush_cwd_cache,
        )

        # "Process 1": scan once, then flush to disk.
        configure_cwd_cache(index_dir)
        _session_cwd(path)
        flush_cwd_cache()
        self.assertTrue((index_dir / "codex_cwd_cache.json").exists())

        # "Process 2": fresh module state, reconfigure from the same
        # index_dir, and the second scan must come from the loaded cache.
        self._reset_cache_state()
        configure_cwd_cache(index_dir)
        with mock.patch(
            "synapt.recall.codex._session_cwd_uncached",
            wraps=self._codex_module._session_cwd_uncached,
        ) as spy:
            result = _session_cwd(path)
        self.assertEqual(spy.call_count, 0, "cross-process cache was not loaded from disk")
        self.assertEqual(result, Path(tmpdir).resolve())

    def test_missing_or_corrupt_cache_file_degrades_to_empty_cache(self):
        """A cache is a courtesy: a corrupt cache file must not fail resume."""
        index_dir = Path(tempfile.mkdtemp())
        (index_dir / "codex_cwd_cache.json").write_text("{not valid json")
        from synapt.recall.codex import configure_cwd_cache

        configure_cwd_cache(index_dir)  # must not raise
        self.assertEqual(self._codex_module._cwd_cache, {})

    def test_failed_flush_never_corrupts_the_existing_cache_file(self):
        """Stromus's R2 (recall-codex-cwd-cache-followon-v1, #dev m_c7c9f9a5):
        an in-place write (revert atomic-flush) passes every other test in
        this class, because none of them fail mid-write. This is the actual
        crash shape atomicity exists for: os.replace fails partway through
        flush, and the ORIGINAL file must be exactly as it was, with no
        leftover tmp file — an in-place write would have already truncated
        or partially overwritten the original by the time the failure could
        even be detected."""
        tmpdir = tempfile.mkdtemp()
        index_dir = Path(tempfile.mkdtemp())
        path = _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "s", "cwd": tmpdir}}],
        )
        from synapt.recall.codex import _session_cwd, configure_cwd_cache, flush_cwd_cache

        cache_path = index_dir / "codex_cwd_cache.json"
        original_bytes = b'{"sentinel-original-content": true}'
        cache_path.write_bytes(original_bytes)

        configure_cwd_cache(index_dir)  # a corrupt-looking seed degrades to {} in memory
        _session_cwd(path)  # a real miss: marks the in-memory cache dirty

        with mock.patch("os.replace", side_effect=OSError("simulated crash mid-flush")):
            flush_cwd_cache()

        self.assertEqual(
            cache_path.read_bytes(), original_bytes,
            "a failed flush must never corrupt the existing cache file",
        )
        leftover_tmp = list(index_dir.glob("codex_cwd_cache.json.tmp-*"))
        self.assertEqual(leftover_tmp, [], "a failed flush must not leave a tmp file behind")

    def test_flush_is_a_noop_when_nothing_changed(self):
        """No cache miss occurred (or the cache was never configured with a
        real index_dir) -> flush must not write a file at all."""
        index_dir = Path(tempfile.mkdtemp())
        from synapt.recall.codex import configure_cwd_cache, flush_cwd_cache

        configure_cwd_cache(index_dir)
        flush_cwd_cache()
        self.assertFalse((index_dir / "codex_cwd_cache.json").exists())

    def test_deleted_file_is_never_served_its_stale_cached_cwd(self):
        """Atlas's R1 note on recall-codex-cwd-cache-1125-v2: a cache entry
        for a since-deleted rollout must never be served on a later read.
        _session_cwd already stats the path FIRST and returns None on
        OSError before ever consulting the cache dict, so this is a
        regression witness for existing read-path behavior, not a new fix —
        it protects the read side from ever regressing to check the cache
        before the filesystem."""
        tmpdir = tempfile.mkdtemp()
        index_dir = Path(tempfile.mkdtemp())
        path = _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "s", "cwd": tmpdir}}],
        )
        from synapt.recall.codex import _session_cwd, configure_cwd_cache

        configure_cwd_cache(index_dir)
        first = _session_cwd(path)
        self.assertEqual(first, Path(tmpdir).resolve())

        path.unlink()
        second = _session_cwd(path)
        self.assertIsNone(second, "a deleted file must never be served its stale cached cwd")

    def test_flush_prunes_entries_for_removed_files(self):
        """A cache entry for a since-deleted rollout must not accumulate
        forever — it is dropped the next time something triggers a flush."""
        tmpdir = tempfile.mkdtemp()
        index_dir = Path(tempfile.mkdtemp())
        gone_path = _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "s", "cwd": tmpdir}}],
            name="rollout-will-be-removed.jsonl",
        )
        kept_path = _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "k", "cwd": tmpdir}}],
            name="rollout-kept.jsonl",
        )
        from synapt.recall.codex import _session_cwd, configure_cwd_cache, flush_cwd_cache

        configure_cwd_cache(index_dir)
        _session_cwd(gone_path)
        _session_cwd(kept_path)
        gone_path.unlink()

        # Deleting gone_path alone does not mark the cache dirty; a real
        # miss (kept_path re-read after being touched) is what triggers the
        # flush this test is checking prunes the stale entry too. Bump
        # mtime forward explicitly rather than os.utime(path, None) — "now"
        # can collide with the mtime already cached on a coarse filesystem.
        kept_stat = kept_path.stat()
        os.utime(kept_path, (kept_stat.st_atime, kept_stat.st_mtime + 5))
        _session_cwd(kept_path)
        flush_cwd_cache()

        saved = json.loads((index_dir / "codex_cwd_cache.json").read_text())
        self.assertNotIn(str(gone_path), saved)
        self.assertIn(str(kept_path), saved)

    def test_uncached_scan_is_bounded_on_a_malformed_file(self):
        """A file with more leading blank/garbage lines than the scan cap
        must not force reading an entire multi-thousand-line rollout."""
        from synapt.recall.codex import (
            _SESSION_CWD_SCAN_LINE_CAP,
            _session_cwd_uncached,
        )

        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "rollout-garbage.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for _ in range(_SESSION_CWD_SCAN_LINE_CAP + 10):
                f.write("not json at all\n")
            f.write(json.dumps({"type": "session_meta", "payload": {"cwd": tmpdir}}) + "\n")

        # The real session_meta line sits past the cap, so a correctly
        # bounded scan reports no cwd rather than finding it — the cap
        # trades a rare late-session_meta miss for a hard ceiling on read
        # cost, which is the point.
        self.assertIsNone(_session_cwd_uncached(path))

    def test_uncached_scan_still_finds_session_meta_within_the_cap(self):
        """The control for the cap test: a well-formed file (session_meta
        first, as Codex always writes it) is unaffected by the cap."""
        tmpdir = tempfile.mkdtemp()
        path = _write_codex_transcript(
            tmpdir,
            [{"type": "session_meta", "payload": {"id": "s", "cwd": tmpdir}}],
            name="rollout-wellformed.jsonl",
        )
        from synapt.recall.codex import _session_cwd_uncached

        self.assertEqual(_session_cwd_uncached(path), Path(tmpdir).resolve())


class TestCodexCwdCacheWiredIntoCommands(unittest.TestCase):
    """Stromus's R2 (recall-codex-cwd-cache-followon-v1, #dev m_c7c9f9a5):
    removing cmd_build's arming call passed every other test in the file —
    the cache existing is not the fix, the fix is that the commands which
    reach the Codex scan actually arm it. Same "predicate exists but nothing
    consumes it" shape as TestTheBuildPreCheckReadsTheCodexArm above, one
    call site over.
    """

    def setUp(self):
        self._store = owned_store()
        self._orig_cwd = os.getcwd()
        os.chdir(self._store.root)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._store.restore()

    def test_cmd_build_arms_the_cache_at_the_projects_index_dir(self):
        import argparse
        from synapt.recall import cli

        args = argparse.Namespace(
            source=None, hf=None, chatgpt_archive=None,
            no_embeddings=True, incremental=False,
        )
        fake_index = mock.Mock()
        fake_index.stats.return_value = {"chunk_count": 1, "session_count": 1}
        expected_index_dir = cli.project_index_dir(Path.cwd().resolve())
        with mock.patch.object(cli, "project_transcript_dirs", return_value=["dummy"]), \
             mock.patch.object(cli, "_check_legacy_index", return_value=None), \
             mock.patch.object(cli, "_archive_and_build", return_value=fake_index), \
             mock.patch("synapt.recall.codex.configure_cwd_cache") as spy:
            cli.cmd_build(args)
        spy.assert_called_once_with(expected_index_dir)

    def test_cmd_hook_precompact_arms_the_cache_at_the_projects_index_dir(self):
        import argparse
        import io
        import sys
        from synapt.recall import cli

        with mock.patch.object(sys, "stdin", io.StringIO("{}")), \
             mock.patch.object(cli, "project_transcript_dirs", return_value=["dummy"]), \
             mock.patch.object(cli, "_archive_and_build", return_value=None), \
             mock.patch.object(cli, "_sync_after_rebuild"), \
             mock.patch.object(cli, "_precompact_journal_write"), \
             mock.patch("synapt.recall.channel.channel_heartbeat"), \
             mock.patch("synapt.recall.codex.configure_cwd_cache") as spy:
            expected_index_dir = cli.project_index_dir(Path.cwd().resolve())
            cli.cmd_hook(argparse.Namespace(event="precompact"))
        spy.assert_called_once_with(expected_index_dir)
