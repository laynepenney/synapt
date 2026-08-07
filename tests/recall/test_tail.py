"""Tests for tail_turns — bounded live-tail parsing into a TailView.

The contract: tail_turns(transcript_path, n=20) reads only a bounded
trailing window of a live transcript file (Claude Code jsonl or Codex
rollout), parses complete turns, and returns the last n oldest-to-newest,
with honesty metadata: read_at (timestamp of THIS read), source_path (the
file actually read), bytes_scanned (window actually consumed), and
truncated_head (True when the window cut into the file's head — the
normal case for a live session, never a warning). Zero parsed turns from
a nonempty window is a valid answer, not an error.
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from synapt.recall.tail import TailView, tail_turns


def _write_jsonl(tmpdir: str, name: str, entries: list) -> Path:
    path = Path(tmpdir) / name
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return path


def _claude_user(text: str, ts: str, uid: str) -> dict:
    return {
        "type": "user",
        "uuid": uid,
        "timestamp": ts,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _claude_assistant(text: str, ts: str, uid: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uid,
        "timestamp": ts,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _codex_meta(sid: str) -> dict:
    return {
        "timestamp": "2026-03-01T10:00:00Z",
        "type": "session_meta",
        "payload": {"id": sid},
    }


def _codex_user(text: str, ts: str) -> dict:
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {"role": "user", "content": [{"type": "input_text", "text": text}]},
    }


def _codex_assistant(text: str, ts: str, phase: str | None = None) -> dict:
    payload = {"role": "assistant", "content": [{"type": "output_text", "text": text}]}
    if phase is not None:
        payload["phase"] = phase
    return {"timestamp": ts, "type": "response_item", "payload": payload}


def _codex_event(msg_type: str, text: str, ts: str, phase: str | None = None) -> dict:
    payload = {"type": msg_type, "message": text}
    if phase is not None:
        payload["phase"] = phase
    return {"timestamp": ts, "type": "event_msg", "payload": payload}


class TestTailTurnsClaude(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_last_n_turns_oldest_to_newest(self):
        entries = []
        for i in range(6):
            entries.append(_claude_user(f"question {i}", f"2026-08-07T10:0{i}:00Z", f"u{i}"))
            entries.append(_claude_assistant(f"answer {i}", f"2026-08-07T10:0{i}:30Z", f"a{i}"))
        path = _write_jsonl(self.tmpdir, "session.jsonl", entries)

        view = tail_turns(path, n=4)

        self.assertIsInstance(view, TailView)
        self.assertEqual(
            [t.text for t in view.turns],
            ["question 4", "answer 4", "question 5", "answer 5"],
        )
        self.assertEqual(
            [t.speaker for t in view.turns],
            ["user", "assistant", "user", "assistant"],
        )
        self.assertTrue(all(t.when for t in view.turns))

    def test_noise_types_and_reminder_only_entries_skipped(self):
        entries = [
            _claude_user("real question", "2026-08-07T10:00:00Z", "u1"),
            {"type": "progress", "uuid": "p1", "timestamp": "x", "data": {}},
            {
                "type": "user",
                "uuid": "u2",
                "timestamp": "x",
                "message": {
                    "role": "user",
                    "content": "<system-reminder>ambient noise</system-reminder>",
                },
            },
            _claude_assistant("real answer", "2026-08-07T10:00:05Z", "a1"),
        ]
        path = _write_jsonl(self.tmpdir, "session.jsonl", entries)

        view = tail_turns(path)

        self.assertEqual([t.text for t in view.turns], ["real question", "real answer"])

    def test_bounded_window_sets_truncated_head(self):
        entries = []
        for i in range(200):
            entries.append(_claude_user("q" * 200, f"2026-08-07T09:00:{i % 60:02d}Z", f"u{i}"))
            entries.append(_claude_assistant("a" * 200, f"2026-08-07T09:01:{i % 60:02d}Z", f"a{i}"))
        entries.append(_claude_assistant("the newest answer", "2026-08-07T10:00:00Z", "afinal"))
        path = _write_jsonl(self.tmpdir, "session.jsonl", entries)
        self.assertGreater(path.stat().st_size, 4096)

        view = tail_turns(path, n=5, window_bytes=4096)

        self.assertTrue(view.truncated_head)
        self.assertLessEqual(view.bytes_scanned, 4096)
        self.assertGreater(len(view.turns), 0)
        self.assertEqual(view.turns[-1].text, "the newest answer")

    def test_whole_file_read_is_not_truncated(self):
        path = _write_jsonl(
            self.tmpdir,
            "session.jsonl",
            [_claude_user("only question", "2026-08-07T10:00:00Z", "u1")],
        )

        view = tail_turns(path)

        self.assertFalse(view.truncated_head)
        self.assertEqual(view.bytes_scanned, path.stat().st_size)

    def test_honest_empty_on_unparseable_window(self):
        path = Path(self.tmpdir) / "garbage.jsonl"
        path.write_text("this is not json\n" * 50, encoding="utf-8")

        view = tail_turns(path)

        self.assertEqual(view.turns, [])
        self.assertGreater(view.bytes_scanned, 0)

    def test_n_exceeding_available_returns_all(self):
        path = _write_jsonl(
            self.tmpdir,
            "session.jsonl",
            [
                _claude_user("q", "2026-08-07T10:00:00Z", "u1"),
                _claude_assistant("a", "2026-08-07T10:00:01Z", "a1"),
            ],
        )

        view = tail_turns(path, n=50)

        self.assertEqual(len(view.turns), 2)

    def test_n_zero_returns_no_turns(self):
        path = _write_jsonl(
            self.tmpdir,
            "session.jsonl",
            [_claude_user("q", "2026-08-07T10:00:00Z", "u1")],
        )

        view = tail_turns(path, n=0)

        self.assertEqual(view.turns, [])

    def test_read_at_and_source_path_state_the_observation(self):
        path = _write_jsonl(
            self.tmpdir,
            "session.jsonl",
            [_claude_user("q", "2026-08-07T10:00:00Z", "u1")],
        )

        view = tail_turns(path)

        self.assertEqual(view.source_path, str(path))
        datetime.fromisoformat(view.read_at)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            tail_turns(Path(self.tmpdir) / "does-not-exist.jsonl")


class TestTailTurnsCodex(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_response_items_parse_as_turns(self):
        entries = [
            _codex_meta("tail-codex"),
            _codex_user("codex question", "2026-03-01T10:00:01Z"),
            _codex_assistant("codex answer", "2026-03-01T10:00:02Z"),
        ]
        path = _write_jsonl(self.tmpdir, "rollout-tail.jsonl", entries)

        view = tail_turns(path)

        self.assertEqual(
            [(t.speaker, t.text) for t in view.turns],
            [("user", "codex question"), ("assistant", "codex answer")],
        )

    def test_agent_message_redelivery_is_deduplicated(self):
        entries = [
            _codex_meta("tail-dedup"),
            _codex_user("do the thing", "2026-03-01T10:00:01Z"),
            _codex_assistant("done the thing", "2026-03-01T10:00:02Z"),
            _codex_event("agent_message", "done the thing", "2026-03-01T10:00:03Z"),
        ]
        path = _write_jsonl(self.tmpdir, "rollout-dedup.jsonl", entries)

        view = tail_turns(path)

        assistant_turns = [t for t in view.turns if t.speaker == "assistant"]
        self.assertEqual(len(assistant_turns), 1)

    def test_event_only_user_message_is_captured(self):
        entries = [
            _codex_meta("tail-event-user"),
            _codex_event("user_message", "typed via event", "2026-03-01T10:00:01Z"),
            _codex_assistant("event answer", "2026-03-01T10:00:02Z"),
        ]
        path = _write_jsonl(self.tmpdir, "rollout-event.jsonl", entries)

        view = tail_turns(path)

        self.assertEqual(
            [(t.speaker, t.text) for t in view.turns],
            [("user", "typed via event"), ("assistant", "event answer")],
        )

    def test_commentary_phase_is_included_as_assistant(self):
        entries = [
            _codex_meta("tail-commentary"),
            _codex_user("work on it", "2026-03-01T10:00:01Z"),
            _codex_assistant("thinking aloud first", "2026-03-01T10:00:02Z", phase="commentary"),
            _codex_assistant("the final answer", "2026-03-01T10:00:03Z"),
        ]
        path = _write_jsonl(self.tmpdir, "rollout-commentary.jsonl", entries)

        view = tail_turns(path)

        self.assertEqual(
            [t.text for t in view.turns],
            ["work on it", "thinking aloud first", "the final answer"],
        )
        self.assertEqual(
            [t.speaker for t in view.turns], ["user", "assistant", "assistant"]
        )


if __name__ == "__main__":
    unittest.main()
