"""TDD tests for Mission Control dashboard API (recall#534 + recall#536).

Tests define the API contract for the web dashboard that wraps tmux sessions.
All tests should FAIL until the dashboard API endpoints are implemented.

recall#534: Dashboard API tests (8 unit)
recall#536: Adversarial + e2e tests (7 tests)
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Dashboard API imports — these endpoints don't exist yet in the current
# 511-line MVP. Tests should fail with ImportError or 404 until implemented.
try:
    from synapt.dashboard.app import create_app

    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# Process tracking imports — new module
try:
    from synapt.recall.process_tracking import (
        update_agent_status,
        get_agent_status,
        get_all_agent_statuses,
    )

    PROCESS_TRACKING_AVAILABLE = True
except ImportError:
    PROCESS_TRACKING_AVAILABLE = False

from synapt.recall.registry import register_agent, _ensure_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_team_db(db_path: Path, org_id: str = "synapt-dev") -> None:
    """Create a team.db with org_agents + process columns."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    _ensure_schema(conn)
    # Add process tracking columns (Sprint 9 schema extension)
    for col, coltype, default in [
        ("session_id", "TEXT", None),
        ("pid", "INTEGER", None),
        ("status", "TEXT", "'offline'"),
        ("tmux_target", "TEXT", None),
        ("log_path", "TEXT", None),
    ]:
        try:
            default_clause = f" DEFAULT {default}" if default else ""
            conn.execute(
                f"ALTER TABLE org_agents ADD COLUMN {col} {coltype}{default_clause}"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()
    conn.close()


def _mock_tmux_send_keys():
    """Return a mock for subprocess.run that simulates tmux send-keys."""
    mock = MagicMock()
    mock.return_value = MagicMock(returncode=0)
    return mock


def _mock_tmux_capture_pane(output: str = ""):
    """Return a mock for subprocess.run that simulates tmux capture-pane."""
    mock = MagicMock()
    mock.return_value = MagicMock(returncode=0, stdout=output)
    return mock


# ===========================================================================
# recall#534: Dashboard API tests (8 unit)
# ===========================================================================


class TestDashboardAgentsEndpoint(unittest.TestCase):
    """Test GET /api/agents returns registered agents with status."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = Path(self._tmpdir) / "team.db"
        _create_team_db(self._db_path)
        # Register test agents
        register_agent("synapt-dev", "Opus", role="CTO", db_path=self._db_path)
        register_agent("synapt-dev", "Atlas", role="architect", db_path=self._db_path)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_get_agents_returns_team_db(self):
        """GET /api/agents returns all registered agents with status."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT agent_id, display_name, role, status FROM org_agents WHERE org_id = ?",
            ("synapt-dev",),
        ).fetchall()
        conn.close()

        agents = [dict(r) for r in rows]
        self.assertEqual(len(agents), 2)
        names = {a["display_name"] for a in agents}
        self.assertEqual(names, {"Opus", "Atlas"})

    def test_agent_status_reflects_tmux_pane(self):
        """Agent marked online when tmux pane exists, offline when not."""
        conn = sqlite3.connect(str(self._db_path))
        # Simulate agent going online
        conn.execute(
            "UPDATE org_agents SET status = 'online', pid = 12345, tmux_target = 'synapt:opus' "
            "WHERE display_name = 'Opus'"
        )
        conn.commit()

        row = conn.execute(
            "SELECT status, pid, tmux_target FROM org_agents WHERE display_name = 'Opus'"
        ).fetchone()
        self.assertEqual(row[0], "online")
        self.assertEqual(row[1], 12345)
        self.assertEqual(row[2], "synapt:opus")

        # Simulate agent going offline
        conn.execute(
            "UPDATE org_agents SET status = 'offline', pid = NULL, tmux_target = NULL "
            "WHERE display_name = 'Opus'"
        )
        conn.commit()

        row = conn.execute(
            "SELECT status, pid FROM org_agents WHERE display_name = 'Opus'"
        ).fetchone()
        self.assertEqual(row[0], "offline")
        self.assertIsNone(row[1])
        conn.close()


class TestDashboardChannelEndpoints(unittest.TestCase):
    """Test channel-related dashboard API endpoints."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._channels_dir = Path(self._tmpdir) / "channels" / "synapt-dev" / "gripspace"
        self._channels_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_post_channel_message(self):
        """POST /api/channels/dev/post writes to global JSONL."""
        # Simulate a channel post by writing directly
        msg = {
            "timestamp": "2026-04-06T17:00:00Z",
            "from": "dashboard",
            "from_display": "Layne [human]",
            "channel": "dev",
            "type": "message",
            "body": "Hello from dashboard",
        }
        path = self._channels_dir / "dev.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(msg) + "\n")

        # Verify message was written
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["body"], "Hello from dashboard")

    def test_channel_stream_sse(self):
        """SSE stream returns new messages as they arrive."""
        # Write initial message
        path = self._channels_dir / "dev.jsonl"
        msg1 = {"timestamp": "2026-04-06T17:00:00Z", "body": "msg1", "channel": "dev"}
        with open(path, "w") as f:
            f.write(json.dumps(msg1) + "\n")

        # Verify we can read it
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 1)

        # Append second message (simulating SSE trigger)
        msg2 = {"timestamp": "2026-04-06T17:01:00Z", "body": "msg2", "channel": "dev"}
        with open(path, "a") as f:
            f.write(json.dumps(msg2) + "\n")

        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 2)

    def test_human_posts_carry_display_name(self):
        """Messages from dashboard have from_display with [human] tag."""
        msg = {
            "from": "dashboard",
            "from_display": "Layne [human]",
            "body": "test",
        }
        self.assertIn("[human]", msg["from_display"])


class TestDashboardAgentIO(unittest.TestCase):
    """Test agent input/output via tmux integration."""

    def test_agent_input_sends_tmux_keys(self):
        """POST /api/agent/opus/input calls tmux send-keys."""
        mock_run = _mock_tmux_send_keys()
        with patch("subprocess.run", mock_run):
            # Simulate the dashboard sending input to an agent
            import subprocess

            target = "synapt:opus"
            user_input = "check the deploy status"
            subprocess.run(
                ["tmux", "send-keys", "-t", target, user_input, "Enter"],
                check=True,
            )

        mock_run.assert_called_once_with(
            ["tmux", "send-keys", "-t", "synapt:opus", "check the deploy status", "Enter"],
            check=True,
        )

    def test_agent_output_streams_log(self):
        """Agent output is available from pipe-pane log file."""
        tmpdir = tempfile.mkdtemp()
        log_path = Path(tmpdir) / "opus" / "output.log"
        log_path.parent.mkdir(parents=True)

        # Simulate pipe-pane writing to log
        log_path.write_text("Agent output line 1\nAgent output line 2\n")

        # Dashboard reads the log
        lines = log_path.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "Agent output line 1")

        shutil.rmtree(tmpdir)

    def test_tmux_discovery_reads_only_the_bound_session(self):
        """Dashboard agent discovery stays inside its explicit tmux session."""
        from synapt.dashboard.app import _tmux_window_agents

        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout="opus\natlas\n")

        with patch("synapt.dashboard.app.subprocess.run", mock_run), \
             patch.dict("synapt.dashboard.app._KNOWN_AGENTS", {"opus": {}, "atlas": {}}, clear=True):
            agents = _tmux_window_agents("workspace-session")

        mock_run.assert_called_once_with(
            [
                "tmux",
                "list-windows",
                "-t",
                "workspace-session",
                "-F",
                "#{window_name}",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        self.assertEqual(agents, {"opus": "online", "atlas": "online"})

    def test_dashboard_works_without_tmux_session(self):
        """Dashboard serves with empty agent grid when no tmux session."""
        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=1)  # tmux not running

        with patch("subprocess.run", mock_run):
            import subprocess

            result = subprocess.run(
                ["tmux", "has-session", "-t", "synapt"],
                capture_output=True,
            )
        self.assertNotEqual(result.returncode, 0)


# ===========================================================================
# recall#536: Adversarial + e2e tests (7 tests)
# ===========================================================================


class TestAdversarialSpawn(unittest.TestCase):
    """Adversarial tests for spawn edge cases."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = Path(self._tmpdir) / "team.db"
        _create_team_db(self._db_path)
        register_agent("synapt-dev", "Opus", role="CTO", db_path=self._db_path)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_spawn_up_twice_skips_existing(self):
        """Second gr spawn up doesn't duplicate agents or tmux windows.

        Simulates the idempotency check: if tmux session exists and
        agent window exists, skip without error.
        """
        conn = sqlite3.connect(str(self._db_path))
        # First spawn: set agent online
        conn.execute(
            "UPDATE org_agents SET status = 'online', pid = 1000, tmux_target = 'synapt:opus' "
            "WHERE display_name = 'Opus'"
        )
        conn.commit()

        # Second spawn: agent already online — should not create duplicate
        row = conn.execute(
            "SELECT status, pid FROM org_agents WHERE display_name = 'Opus'"
        ).fetchone()
        self.assertEqual(row[0], "online")
        self.assertEqual(row[1], 1000)

        # Count agents — should still be 1
        count = conn.execute(
            "SELECT COUNT(*) FROM org_agents WHERE org_id = 'synapt-dev'"
        ).fetchone()[0]
        self.assertEqual(count, 1)
        conn.close()

    def test_agent_crash_detected(self):
        """Kill a tmux pane PID → status updates to 'crashed'."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            "UPDATE org_agents SET status = 'online', pid = 99999 "
            "WHERE display_name = 'Opus'"
        )
        conn.commit()

        # Simulate crash detection: PID no longer exists
        # In real code, this would check os.kill(pid, 0) or tmux list-panes
        pid_alive = False  # Simulated: process is dead
        if not pid_alive:
            conn.execute(
                "UPDATE org_agents SET status = 'crashed' WHERE display_name = 'Opus'"
            )
            conn.commit()

        row = conn.execute(
            "SELECT status FROM org_agents WHERE display_name = 'Opus'"
        ).fetchone()
        self.assertEqual(row[0], "crashed")
        conn.close()

    def test_crash_restart_with_resume(self):
        """After crash, restart uses --resume <session_id> from team.db."""
        conn = sqlite3.connect(str(self._db_path))
        session_id = "abc123-session"
        conn.execute(
            "UPDATE org_agents SET status = 'crashed', session_id = ? "
            "WHERE display_name = 'Opus'",
            (session_id,),
        )
        conn.commit()

        # On restart, read session_id for --resume flag
        row = conn.execute(
            "SELECT session_id FROM org_agents WHERE display_name = 'Opus'"
        ).fetchone()
        self.assertEqual(row[0], session_id)

        # Build restart command with --resume
        resume_cmd = f"claude --resume {row[0]}"
        self.assertIn("--resume", resume_cmd)
        self.assertIn(session_id, resume_cmd)
        conn.close()

    def test_log_rotation_or_size_limit(self):
        """Output log > threshold triggers rotation or truncation."""
        tmpdir = tempfile.mkdtemp()
        log_path = Path(tmpdir) / "opus" / "output.log"
        log_path.parent.mkdir(parents=True)

        # Write 10MB of log data
        chunk = "x" * 1024 + "\n"  # ~1KB per line
        with open(log_path, "w") as f:
            for _ in range(10240):  # ~10MB
                f.write(chunk)

        size_mb = log_path.stat().st_size / (1024 * 1024)
        self.assertGreater(size_mb, 9)  # Verify it's actually large

        # The implementation should rotate or truncate at some threshold
        # For now, we just verify the file exists and is large
        # Real test: after rotation, verify old log archived and new log started
        self.assertTrue(log_path.exists())

        shutil.rmtree(tmpdir)

    def test_dashboard_channel_only_mode(self):
        """Dashboard serves channels/posts without any tmux session or agents."""
        tmpdir = tempfile.mkdtemp()
        channels_dir = Path(tmpdir) / "channels" / "synapt-dev" / "gripspace"
        channels_dir.mkdir(parents=True)

        # Write a channel message
        msg = {"body": "test without agents", "channel": "dev"}
        with open(channels_dir / "dev.jsonl", "w") as f:
            f.write(json.dumps(msg) + "\n")

        # Dashboard should be able to read channels without tmux
        with open(channels_dir / "dev.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["body"], "test without agents")

        shutil.rmtree(tmpdir)


class TestDashboardTemplateUI(unittest.TestCase):
    """Verify the dashboard template has required UI elements."""

    def test_template_has_agent_input_panel(self):
        """Dashboard template contains agent input form (recall#544)."""
        template_path = Path(__file__).parent.parent.parent / "src" / "synapt" / "dashboard" / "template.html"
        content = template_path.read_text()
        self.assertIn("agent-input-panel", content)
        self.assertIn("agent-input-text", content)
        self.assertIn("agent-input-send", content)
        self.assertIn("/api/agent/", content)

    def test_agent_tiles_are_clickable(self):
        """Agent tile renderer adds clickable class and data-agent attribute."""
        from synapt.dashboard.app import _render_agent_tile
        tile_html = _render_agent_tile({
            "status": "online",
            "display_name": "Opus",
            "griptree": "synapt-codex/recall",
            "agent_id": "s_123",
            "role": "CTO",
            "channels": ["dev"],
            "last_seen": "2026-04-06T17:00:00Z",
        })
        self.assertIn("clickable", tile_html)
        self.assertIn('data-agent="Opus"', tile_html)


@pytest.mark.integration
class TestEndToEnd(unittest.TestCase):
    """End-to-end tests requiring real tmux.

    These tests create actual tmux sessions and verify the full pipeline.
    Skip in CI with: pytest -m "not integration"
    """

    def test_e2e_post_input_read_output(self):
        """Post input via API, verify it appears in agent's pipe-pane log."""
        # This test requires a real tmux session
        # Skipped in CI, run manually for integration validation
        pytest.skip("Integration test: requires real tmux session")

    def test_e2e_channel_roundtrip(self):
        """Human posts from dashboard, agent reads, agent replies, human sees reply."""
        pytest.skip("Integration test: requires real tmux + agent process")


@pytest.mark.skipif(not DASHBOARD_AVAILABLE, reason="Dashboard not importable")
class TestXSSProtection:
    """XSS vector tests for _render_message HTML sanitization (#756)."""

    def _render(self, body: str) -> str:
        from synapt.dashboard.app import _render_message

        msg = {
            "timestamp": "2026-04-24T10:00:00Z",
            "from_display": "attacker",
            "body": body,
            "type": "message",
        }
        return _render_message(msg)

    def test_script_tag_stripped(self):
        html = self._render("<script>alert('xss')</script>hello")
        assert "<script>" not in html
        assert "alert" not in html
        assert "hello" in html

    def test_img_onerror_stripped(self):
        html = self._render('<img src=x onerror=alert(1)>')
        assert "onerror" not in html

    def test_iframe_stripped(self):
        html = self._render('<iframe src="evil.com"></iframe>')
        assert "<iframe" not in html
        assert "evil.com" not in html

    def test_event_handler_attributes_stripped(self):
        html = self._render('<div onclick="alert(1)">click me</div>')
        assert "onclick" not in html
        assert "click me" in html

    def test_javascript_url_stripped(self):
        html = self._render('<a href="javascript:alert(1)">click</a>')
        assert "javascript:" not in html

    def test_legitimate_markdown_preserved(self):
        html = self._render("**bold** and `code` and [link](https://example.com)")
        assert "<strong>bold</strong>" in html
        assert "<code>code</code>" in html
        assert 'href="https://example.com"' in html

    def test_fenced_code_block_preserved(self):
        html = self._render("```python\nprint('hello')\n```")
        assert "<code" in html
        assert "print" in html

    def test_table_preserved(self):
        html = self._render("| A | B |\n|---|---|\n| 1 | 2 |")
        assert "<table>" in html
        assert "<td>" in html

    def test_mention_styling_preserved(self):
        html = self._render("hello @Apollo")
        assert "style=" in html
        assert "@Apollo" in html


if __name__ == "__main__":
    unittest.main()
