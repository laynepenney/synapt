"""Tests for synapt.recall.codex — Codex CLI transcript parsing."""

import json
import tempfile
import unittest
from pathlib import Path

from synapt.recall.codex import (
    parse_codex_transcript,
    list_codex_transcripts,
    archive_codex_transcripts,
    is_codex_transcript,
    _extract_file_paths,
)
from synapt.recall.core import build_index
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
             "payload": {"role": "assistant", "content": [
                 {"type": "output_text", "text": assistant_text}
             ]}},
            {"timestamp": "2026-03-01T10:00:06Z", "type": "event_msg",
             "payload": {"type": "agent_message", "message": assistant_text}},
            {"timestamp": "2026-03-01T10:00:07Z", "type": "event_msg",
             "payload": {"type": "agent_message", "message": unique_agent_text}},
            {"timestamp": "2026-03-01T10:00:08Z", "type": "event_msg",
             "payload": {"type": "agent_message", "phase": "commentary",
                         "message": commentary_agent_text}},
        ]
        path = _write_codex_transcript(self.tmpdir, entries)

        chunks = parse_codex_transcript(path)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.tools_used, ["shell_run", "large_input_tool"])
        self.assertIn(f"[shell_run] cat {synthetic_file}", chunk.tool_content)
        self.assertIn(synthetic_file, chunk.files_touched)
        self.assertIn("[large_input_tool]", chunk.tool_content)
        self.assertEqual(chunk.assistant_text.count(assistant_text), 1)
        self.assertEqual(chunk.assistant_text.count(unique_agent_text), 1)
        self.assertNotIn(commentary_agent_text, chunk.assistant_text)

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
        """Commentary phase assistant messages are filtered out."""
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
