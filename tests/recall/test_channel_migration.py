"""Tests for Phase 1: channel migration from local to global store.

TDD — these tests are written before the implementation. They should
all fail until the migration code is implemented.

Design spec (internal), Phase 1
Migration path: local .synapt/recall/channels/ → ~/.synapt/channels/<org>/<project>/
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _make_message(
    timestamp: str,
    from_agent: str,
    body: str,
    channel: str = "dev",
    msg_type: str = "message",
    msg_id: str | None = None,
    from_display: str | None = None,
    attachments: list[str] | None = None,
) -> dict:
    """Create a channel message dict matching the JSONL format."""
    msg = {
        "timestamp": timestamp,
        "from_agent": from_agent,
        "channel": channel,
        "type": msg_type,
        "body": body,
    }
    if msg_id:
        msg["id"] = msg_id
    if from_display:
        msg["from_display"] = from_display
    if attachments:
        msg["attachments"] = attachments
    return msg


def _write_messages(path: Path, messages: list[dict]) -> None:
    """Write messages to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


def _read_messages(path: Path) -> list[dict]:
    """Read messages from a JSONL file."""
    if not path.exists():
        return []
    messages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def _setup_local_channels(local_dir: Path, messages: list[dict]) -> None:
    """Set up a local channel store with messages, mimicking .synapt/recall/channels/."""
    channels: dict[str, list[dict]] = {}
    for msg in messages:
        ch = msg.get("channel", "dev")
        channels.setdefault(ch, []).append(msg)
    for ch, msgs in channels.items():
        _write_messages(local_dir / f"{ch}.jsonl", msgs)


def _setup_local_cursors(local_dir: Path, cursors: dict[str, dict[str, str]]) -> None:
    """Set up local cursors in channels.db, mimicking the current cursor table."""
    db_path = local_dir / "channels.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cursors ("
        "agent_id TEXT NOT NULL, channel TEXT NOT NULL, "
        "last_read_at TEXT NOT NULL, PRIMARY KEY (agent_id, channel))"
    )
    for agent_id, channels in cursors.items():
        for channel, last_read in channels.items():
            conn.execute(
                "INSERT INTO cursors (agent_id, channel, last_read_at) VALUES (?, ?, ?)",
                (agent_id, channel, last_read),
            )
    conn.commit()
    conn.close()


class TestChannelMigration(unittest.TestCase):
    """TDD tests for local → global channel migration."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.local_dir = Path(self._tmpdir) / "project" / ".synapt" / "recall" / "channels"
        self.global_dir = Path(self._tmpdir) / "home" / ".synapt" / "channels"
        self.org_id = "synapt-dev"
        self.project_id = "gripspace"

    def _target_dir(self) -> Path:
        return self.global_dir / self.org_id / self.project_id

    def _import_migrate(self):
        """Import the migration function. Will fail until implemented."""
        from synapt.recall.channel import migrate_channels_to_global
        return migrate_channels_to_global

    def test_migration_preserves_all_messages(self):
        """All messages from local store appear in global store."""
        messages = [
            _make_message(f"2026-04-0{i}T10:00:00Z", f"agent-{i}", f"Message {i}")
            for i in range(1, 10)
        ]
        _setup_local_channels(self.local_dir, messages)

        migrate = self._import_migrate()
        migrate(
            local_dir=self.local_dir,
            global_dir=self.global_dir,
            org_id=self.org_id,
            project_id=self.project_id,
        )

        result = _read_messages(self._target_dir() / "dev.jsonl")
        self.assertEqual(len(result), 9)

    def test_migration_preserves_message_order(self):
        """Messages maintain chronological order after migration."""
        messages = [
            _make_message("2026-04-01T10:00:00Z", "agent-1", "First"),
            _make_message("2026-04-01T11:00:00Z", "agent-2", "Second"),
            _make_message("2026-04-01T12:00:00Z", "agent-3", "Third"),
        ]
        _setup_local_channels(self.local_dir, messages)

        migrate = self._import_migrate()
        migrate(
            local_dir=self.local_dir,
            global_dir=self.global_dir,
            org_id=self.org_id,
            project_id=self.project_id,
        )

        result = _read_messages(self._target_dir() / "dev.jsonl")
        timestamps = [m["timestamp"] for m in result]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(result[0]["body"], "First")
        self.assertEqual(result[2]["body"], "Third")

    def test_migration_preserves_message_fields(self):
        """All message fields are preserved: from_agent, body, type, attachments, mentions."""
        messages = [
            _make_message(
                "2026-04-01T10:00:00Z",
                "agent-1",
                "Hello @team",
                msg_type="directive",
                msg_id="m_abc123",
                from_display="Atlas",
                attachments=["file.py"],
            ),
        ]
        _setup_local_channels(self.local_dir, messages)

        migrate = self._import_migrate()
        migrate(
            local_dir=self.local_dir,
            global_dir=self.global_dir,
            org_id=self.org_id,
            project_id=self.project_id,
        )

        result = _read_messages(self._target_dir() / "dev.jsonl")
        self.assertEqual(len(result), 1)
        msg = result[0]
        self.assertEqual(msg["from_agent"], "agent-1")
        self.assertEqual(msg["body"], "Hello @team")
        self.assertEqual(msg["type"], "directive")
        self.assertEqual(msg["id"], "m_abc123")
        self.assertEqual(msg["from_display"], "Atlas")
        self.assertEqual(msg["attachments"], ["file.py"])

    def test_migration_creates_correct_directory_structure(self):
        """Global store has org/project directory structure."""
        messages = [_make_message("2026-04-01T10:00:00Z", "a", "test")]
        _setup_local_channels(self.local_dir, messages)

        migrate = self._import_migrate()
        migrate(
            local_dir=self.local_dir,
            global_dir=self.global_dir,
            org_id=self.org_id,
            project_id=self.project_id,
        )

        self.assertTrue(self._target_dir().is_dir())
        self.assertTrue((self._target_dir() / "dev.jsonl").exists())

    def test_migration_resets_cursors_to_global_state_db(self):
        """Local cursors migrate to _state.db with org/project scope."""
        messages = [_make_message("2026-04-01T10:00:00Z", "a", "test")]
        _setup_local_channels(self.local_dir, messages)
        _setup_local_cursors(self.local_dir, {
            "agent-1": {"dev": "2026-04-01T09:00:00Z"},
            "agent-2": {"dev": "2026-04-01T08:00:00Z"},
        })

        migrate = self._import_migrate()
        migrate(
            local_dir=self.local_dir,
            global_dir=self.global_dir,
            org_id=self.org_id,
            project_id=self.project_id,
        )

        state_db = self.global_dir / "_state.db"
        self.assertTrue(state_db.exists())

        conn = sqlite3.connect(str(state_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM cursors WHERE org_id = ? AND project_id = ?",
            (self.org_id, self.project_id),
        ).fetchall()
        conn.close()

        self.assertEqual(len(rows), 2)
        agents = {r["agent_id"]: r["cursor_value"] for r in rows}
        self.assertEqual(agents["agent-1"], "2026-04-01T09:00:00Z")
        self.assertEqual(agents["agent-2"], "2026-04-01T08:00:00Z")

    def test_migration_is_idempotent(self):
        """Running migration twice doesn't duplicate messages."""
        messages = [
            _make_message("2026-04-01T10:00:00Z", "a", "msg1"),
            _make_message("2026-04-01T11:00:00Z", "b", "msg2"),
        ]
        _setup_local_channels(self.local_dir, messages)

        migrate = self._import_migrate()
        migrate(
            local_dir=self.local_dir,
            global_dir=self.global_dir,
            org_id=self.org_id,
            project_id=self.project_id,
        )
        migrate(
            local_dir=self.local_dir,
            global_dir=self.global_dir,
            org_id=self.org_id,
            project_id=self.project_id,
        )

        result = _read_messages(self._target_dir() / "dev.jsonl")
        self.assertEqual(len(result), 2)

    def test_migration_handles_empty_channels(self):
        """Migration doesn't crash on empty JSONL files."""
        self.local_dir.mkdir(parents=True, exist_ok=True)
        (self.local_dir / "dev.jsonl").write_text("")

        migrate = self._import_migrate()
        # Should not raise
        migrate(
            local_dir=self.local_dir,
            global_dir=self.global_dir,
            org_id=self.org_id,
            project_id=self.project_id,
        )

    def test_migration_handles_multiple_channels(self):
        """All channels in local store are migrated, not just dev."""
        dev_msgs = [_make_message("2026-04-01T10:00:00Z", "a", "dev msg", channel="dev")]
        ops_msgs = [_make_message("2026-04-01T11:00:00Z", "b", "ops msg", channel="ops")]

        _write_messages(self.local_dir / "dev.jsonl", dev_msgs)
        _write_messages(self.local_dir / "ops.jsonl", ops_msgs)

        migrate = self._import_migrate()
        migrate(
            local_dir=self.local_dir,
            global_dir=self.global_dir,
            org_id=self.org_id,
            project_id=self.project_id,
        )

        dev_result = _read_messages(self._target_dir() / "dev.jsonl")
        ops_result = _read_messages(self._target_dir() / "ops.jsonl")
        self.assertEqual(len(dev_result), 1)
        self.assertEqual(len(ops_result), 1)
        self.assertEqual(dev_result[0]["body"], "dev msg")
        self.assertEqual(ops_result[0]["body"], "ops msg")

    def test_migration_large_volume(self):
        """Migration handles 5K+ messages without data loss."""
        messages = [
            _make_message(
                f"2026-{(i // 1440) + 1:02d}-{((i % 1440) // 60) + 1:02d}T{(i % 60):02d}:{(i // 60 % 60):02d}:{i % 60:02d}Z",
                f"agent-{i % 5}",
                f"Message {i}",
            )
            for i in range(5000)
        ]
        _setup_local_channels(self.local_dir, messages)

        migrate = self._import_migrate()
        migrate(
            local_dir=self.local_dir,
            global_dir=self.global_dir,
            org_id=self.org_id,
            project_id=self.project_id,
        )

        result = _read_messages(self._target_dir() / "dev.jsonl")
        self.assertEqual(len(result), 5000)

    def test_migration_derives_org_project_from_manifest(self):
        """Org and project should be parsed from gripspace manifest URL."""
        from synapt.recall.channel import _extract_org_from_url, _extract_repo_from_url

        # SSH format
        self.assertEqual(
            _extract_org_from_url("git@github.com:synapt-dev/gripspace.git"),
            "synapt-dev",
        )
        self.assertEqual(
            _extract_repo_from_url("git@github.com:synapt-dev/gripspace.git"),
            "gripspace",
        )
        # HTTPS format
        self.assertEqual(
            _extract_org_from_url("https://github.com/synapt-dev/gripspace.git"),
            "synapt-dev",
        )
        self.assertEqual(
            _extract_repo_from_url("https://github.com/synapt-dev/gripspace.git"),
            "gripspace",
        )

    def test_migration_skips_already_migrated(self):
        """Partial migration: only append messages newer than what's already in global store."""
        # Pre-existing messages in global store
        existing = [
            _make_message("2026-04-01T10:00:00Z", "a", "already migrated"),
        ]
        _write_messages(self._target_dir() / "dev.jsonl", existing)

        # Local store has the old message + a new one
        local_msgs = [
            _make_message("2026-04-01T10:00:00Z", "a", "already migrated"),
            _make_message("2026-04-02T10:00:00Z", "b", "new message"),
        ]
        _setup_local_channels(self.local_dir, local_msgs)

        migrate = self._import_migrate()
        migrate(
            local_dir=self.local_dir,
            global_dir=self.global_dir,
            org_id=self.org_id,
            project_id=self.project_id,
        )

        result = _read_messages(self._target_dir() / "dev.jsonl")
        # Should have 2 messages total, not 3 (no duplicate of "already migrated")
        self.assertEqual(len(result), 2)
        bodies = [m["body"] for m in result]
        self.assertIn("already migrated", bodies)
        self.assertIn("new message", bodies)


if __name__ == "__main__":
    unittest.main()
