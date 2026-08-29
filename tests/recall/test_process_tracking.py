"""Tests for recall#535: Process tracking in team.db.

TDD — tests written before implementation. They define the contract
for process state columns in org_agents and status transition functions.

Process tracking adds: session_id, pid, status, tmux_target, log_path
to the org_agents table, plus update_agent_status() and crash detection.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestProcessTracking(unittest.TestCase):
    """Process tracking columns and status transitions in team.db."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = Path(self._tmpdir) / "team.db"

    def _import_functions(self):
        """Import process tracking functions. Fails until implemented."""
        from synapt.recall.registry import (
            update_agent_status,
            get_agent_status,
            detect_crashed_agents,
            clear_agent_session,
        )
        return update_agent_status, get_agent_status, detect_crashed_agents, clear_agent_session

    def _register_agent(self, agent_id: str, display_name: str, org_id: str = "synapt-dev"):
        """Register a test agent directly in team.db."""
        from synapt.recall.registry import register_agent, _ensure_schema
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)
        conn.execute(
            "INSERT OR IGNORE INTO org_agents (agent_id, display_name, role, org_id, created_at) "
            "VALUES (?, ?, 'agent', ?, datetime('now'))",
            (agent_id, display_name, org_id),
        )
        conn.commit()
        conn.close()

    def test_update_status_sets_process_columns(self):
        """update_agent_status writes session_id, pid, status, tmux_target, log_path."""
        update_status, get_status, _, _ = self._import_functions()
        self._register_agent("opus-001", "Opus")

        update_status(
            db_path=self._db_path,
            agent_id="opus-001",
            status="running",
            session_id="sess-abc",
            pid=12345,
            tmux_target="synapt:opus",
            log_path="/tmp/opus-stdout.log",
        )

        status = get_status(db_path=self._db_path, agent_id="opus-001")
        self.assertEqual(status["status"], "running")
        self.assertEqual(status["session_id"], "sess-abc")
        self.assertEqual(status["pid"], 12345)
        self.assertEqual(status["tmux_target"], "synapt:opus")
        self.assertEqual(status["log_path"], "/tmp/opus-stdout.log")

    def test_status_transitions(self):
        """Status transitions: stopped → running → stopped."""
        update_status, get_status, _, _ = self._import_functions()
        self._register_agent("atlas-001", "Atlas")

        # Initial: no status
        status = get_status(db_path=self._db_path, agent_id="atlas-001")
        self.assertIsNone(status["status"])

        # Start
        update_status(db_path=self._db_path, agent_id="atlas-001",
                       status="running", pid=111)
        self.assertEqual(get_status(db_path=self._db_path, agent_id="atlas-001")["status"], "running")

        # Stop
        update_status(db_path=self._db_path, agent_id="atlas-001",
                       status="stopped")
        self.assertEqual(get_status(db_path=self._db_path, agent_id="atlas-001")["status"], "stopped")

    def test_detect_crashed_agents(self):
        """Agents with status='running' but dead PIDs are detected as crashed."""
        update_status, _, detect_crashed, _ = self._import_functions()
        self._register_agent("apollo-001", "Apollo")

        # Set running with a PID that definitely doesn't exist
        update_status(db_path=self._db_path, agent_id="apollo-001",
                       status="running", pid=999999999)

        crashed = detect_crashed(db_path=self._db_path)
        self.assertIn("apollo-001", [c["agent_id"] for c in crashed])

    def test_crash_sweep_reuses_query_only_windows_pid_probe(self):
        """Crash detection reaches the shared Windows-safe liveness probe."""
        from synapt.recall import registry, session_start

        self.assertIs(registry._pid_alive, session_start._pid_alive)
        update_status, _, detect_crashed, _ = self._import_functions()
        self._register_agent("agent-001", "Agent")
        update_status(
            db_path=self._db_path,
            agent_id="agent-001",
            status="running",
            pid=424242,
        )
        pid_calls = []

        def query_only_probe(pid):
            pid_calls.append(pid)
            return True

        with patch.object(session_start.sys, "platform", "win32"), \
             patch.object(
                 session_start.os,
                 "kill",
                 side_effect=AssertionError("os.kill must not run on Windows"),
             ), \
             patch.object(session_start, "_pid_alive_win32", query_only_probe):
            crashed = detect_crashed(db_path=self._db_path)

        self.assertEqual(crashed, [])
        self.assertEqual(pid_calls, [424242])

    def test_clear_session_preserves_agent_id(self):
        """Clearing a session resets process columns but keeps agent identity."""
        update_status, get_status, _, clear_session = self._import_functions()
        self._register_agent("sentinel-001", "Sentinel")

        update_status(db_path=self._db_path, agent_id="sentinel-001",
                       status="running", session_id="sess-xyz", pid=54321,
                       tmux_target="synapt:sentinel", log_path="/tmp/sentinel.log")

        clear_session(db_path=self._db_path, agent_id="sentinel-001")

        status = get_status(db_path=self._db_path, agent_id="sentinel-001")
        # Agent still exists
        self.assertIsNotNone(status)
        # Process columns cleared
        self.assertEqual(status["status"], "stopped")
        self.assertIsNone(status["pid"])
        self.assertIsNone(status["session_id"])


if __name__ == "__main__":
    unittest.main()
