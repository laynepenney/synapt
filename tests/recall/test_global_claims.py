"""Tests for Story #12: claims in global _state.db.

TDD — claims move from per-gripspace channels.db to global _state.db
with org/project scoping. First-writer-wins via INSERT OR IGNORE.

Design spec (internal)
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


class TestGlobalClaims(unittest.TestCase):
    """Claims in global _state.db with org/project scope."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.state_db = Path(self._tmpdir) / ".synapt" / "channels" / "_state.db"
        self.org_id = "synapt-dev"
        self.project_id = "gripspace"

    def _import_global_claim(self):
        """Import the global claim function. Fails until implemented."""
        from synapt.recall.channel import global_claim
        return global_claim

    def _import_global_unclaim(self):
        """Import the global unclaim function. Fails until implemented."""
        from synapt.recall.channel import global_unclaim
        return global_unclaim

    def _import_is_globally_claimed(self):
        """Import the global claim check function. Fails until implemented."""
        from synapt.recall.channel import is_globally_claimed
        return is_globally_claimed

    def test_claim_creates_entry_in_state_db(self):
        """Claiming a message writes to _state.db with org/project scope."""
        claim = self._import_global_claim()
        claim(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id=self.project_id,
            channel="dev",
            message_id="m_abc123",
            claimed_by="opus-001",
            display_name="Opus",
        )

        conn = sqlite3.connect(str(self.state_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM claims WHERE message_id = ?", ("m_abc123",)
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row["org_id"], self.org_id)
        self.assertEqual(row["project_id"], self.project_id)
        self.assertEqual(row["claimed_by"], "opus-001")

    def test_first_writer_wins(self):
        """Second claim on same message_id is rejected (INSERT OR IGNORE)."""
        claim = self._import_global_claim()

        result1 = claim(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id=self.project_id,
            channel="dev",
            message_id="m_race",
            claimed_by="atlas-001",
            display_name="Atlas",
        )

        result2 = claim(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id=self.project_id,
            channel="dev",
            message_id="m_race",
            claimed_by="apollo-001",
            display_name="Apollo",
        )

        # First claim succeeds
        self.assertTrue(result1)
        # Second claim fails (already claimed)
        self.assertFalse(result2)

        # Verify original claimer wins
        conn = sqlite3.connect(str(self.state_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT claimed_by FROM claims WHERE message_id = ?", ("m_race",)
        ).fetchone()
        conn.close()
        self.assertEqual(row["claimed_by"], "atlas-001")

    def test_claims_scoped_by_org_project(self):
        """Same message_id can be claimed in different projects."""
        claim = self._import_global_claim()

        claim(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id="project-a",
            channel="dev",
            message_id="m_same",
            claimed_by="atlas-001",
            display_name="Atlas",
        )

        result = claim(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id="project-b",
            channel="dev",
            message_id="m_same",
            claimed_by="apollo-001",
            display_name="Apollo",
        )

        # Different project → different scope → both succeed
        self.assertTrue(result)

    def test_unclaim_removes_entry(self):
        """Unclaiming removes the entry from _state.db."""
        claim = self._import_global_claim()
        unclaim = self._import_global_unclaim()

        claim(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id=self.project_id,
            channel="dev",
            message_id="m_unclaim",
            claimed_by="opus-001",
            display_name="Opus",
        )

        unclaim(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id=self.project_id,
            channel="dev",
            message_id="m_unclaim",
            claimed_by="opus-001",
        )

        conn = sqlite3.connect(str(self.state_db))
        row = conn.execute(
            "SELECT * FROM claims WHERE message_id = ?", ("m_unclaim",)
        ).fetchone()
        conn.close()
        self.assertIsNone(row)

    def test_unclaim_only_by_claimer(self):
        """Only the original claimer can unclaim."""
        claim = self._import_global_claim()
        unclaim = self._import_global_unclaim()

        claim(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id=self.project_id,
            channel="dev",
            message_id="m_owned",
            claimed_by="atlas-001",
            display_name="Atlas",
        )

        result = unclaim(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id=self.project_id,
            channel="dev",
            message_id="m_owned",
            claimed_by="apollo-001",  # Not the claimer
        )

        self.assertFalse(result)

        # Verify still claimed by Atlas
        is_claimed = self._import_is_globally_claimed()
        claimer = is_claimed(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id=self.project_id,
            channel="dev",
            message_id="m_owned",
        )
        self.assertEqual(claimer, "atlas-001")

    def test_concurrent_claims_race(self):
        """Concurrent claims on same message — exactly one wins."""
        claim = self._import_global_claim()
        results = {}
        barrier = threading.Barrier(2)

        def try_claim(agent_id, display_name):
            barrier.wait()
            results[agent_id] = claim(
                state_db=self.state_db,
                org_id=self.org_id,
                project_id=self.project_id,
                channel="dev",
                message_id="m_concurrent",
                claimed_by=agent_id,
                display_name=display_name,
            )

        t1 = threading.Thread(target=try_claim, args=("atlas-001", "Atlas"))
        t2 = threading.Thread(target=try_claim, args=("apollo-001", "Apollo"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one succeeds
        wins = sum(1 for v in results.values() if v)
        self.assertEqual(wins, 1)

    def test_is_globally_claimed_returns_claimer(self):
        """is_globally_claimed returns the agent_id of the claimer."""
        claim = self._import_global_claim()
        is_claimed = self._import_is_globally_claimed()

        claim(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id=self.project_id,
            channel="dev",
            message_id="m_check",
            claimed_by="sentinel-001",
            display_name="Sentinel",
        )

        result = is_claimed(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id=self.project_id,
            channel="dev",
            message_id="m_check",
        )
        self.assertEqual(result, "sentinel-001")

    def test_is_globally_claimed_returns_none_when_unclaimed(self):
        """is_globally_claimed returns None for unclaimed messages."""
        is_claimed = self._import_is_globally_claimed()

        result = is_claimed(
            state_db=self.state_db,
            org_id=self.org_id,
            project_id=self.project_id,
            channel="dev",
            message_id="m_nobody",
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
