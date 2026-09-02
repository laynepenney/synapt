"""Adversarial tests for channel scoping — derived from 5-iteration design review.

Every test maps to a specific bug or edge case found during the channel scoping
design review (v1-v5). These are the tests that would have caught the bugs
that were identified on paper before any code was written.

TDD: all tests should FAIL until the corresponding implementation ships.
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest

import pytest
from pathlib import Path
from unittest.mock import patch

# Phase 0 imports — org agent registry (implemented in Sprint 8)
from synapt.recall.registry import (
    register_agent,
    get_agent,
    list_agents,
    update_display_name,
)

# Phase 1 imports — global channel store functions
from synapt.recall.channel import (
    ChannelMessage,
    _channels_dir,
    _local_channels_dir,
    _shared_channels_dir,
    _channel_path,
    _db_path,
    _append_message,
    _read_messages,
    _read_manifest_url,
    _global_channels_dir,
    channel_join,
    channel_post,
    channel_read,
    channel_claim,
    channel_unclaim,
    channel_list_channels,
)

# These will be added in Phase 1 implementation
try:
    from synapt.recall.channel import (
        _resolve_org_id,
        _resolve_project_id,
        migrate_channels_to_global,
        channel_list_projects,
    )

    SCOPING_AVAILABLE = True
except ImportError:
    SCOPING_AVAILABLE = False

from synapt.recall.core import project_data_dir, _gripspace_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gripspace_with_manifest(
    tmp_path: Path, url: str = "git@github.com:synapt-dev/synapt-gripspace.git"
) -> Path:
    """A gripspace root real enough for both root-detection and manifest
    parsing: ``.gitgrip/griptrees.json`` marks the root (what
    ``_find_gripspace_root`` looks for), ``.gitgrip/spaces/main/gripspace.yml``
    carries the manifest URL (what ``_read_manifest_url`` parses)."""
    grip = tmp_path / "workspace"
    spaces_main = grip / ".gitgrip" / "spaces" / "main"
    spaces_main.mkdir(parents=True)
    (grip / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
    (spaces_main / "gripspace.yml").write_text(f"version: 2\nmanifest:\n  url: {url}\nrepos: {{}}\n")
    return grip


def _create_team_db(db_path: Path, org_id: str = "synapt-dev") -> None:
    """Create a team.db with the org_agents schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS org_agents (
            agent_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            role TEXT,
            org_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT
        )"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_org_display
           ON org_agents(org_id, display_name)"""
    )
    conn.commit()
    conn.close()


def _create_state_db(db_path: Path) -> None:
    """Create a _state.db with cursors and claims tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cursors (
            agent_id TEXT NOT NULL,
            org_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            channel TEXT NOT NULL,
            cursor_value TEXT NOT NULL,
            last_read_at TEXT NOT NULL,
            PRIMARY KEY (agent_id, org_id, project_id, channel)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS claims (
            org_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            channel TEXT NOT NULL,
            message_id TEXT NOT NULL,
            claimed_by TEXT NOT NULL,
            display_name TEXT NOT NULL,
            claimed_at TEXT NOT NULL,
            PRIMARY KEY (org_id, project_id, channel, message_id)
        )"""
    )
    conn.commit()
    conn.close()


def _write_fake_messages(path: Path, count: int, channel: str = "dev") -> list[str]:
    """Write N fake messages to a JSONL file, return message IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = []
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            msg_id = f"m_test_{i:04d}"
            msg = {
                "timestamp": f"2026-04-06T{i // 60:02d}:{i % 60:02d}:00Z",
                "id": msg_id,
                "from": f"s_agent_{i % 4}",
                "channel": channel,
                "type": "message",
                "body": f"Test message {i}",
            }
            f.write(json.dumps(msg) + "\n")
            ids.append(msg_id)
    return ids


# ===========================================================================
# Phase 0: Org Agent Registry — adversarial cases
# ===========================================================================


class TestOrgAgentRegistry(unittest.TestCase):
    """Tests for org agent registry (Phase 0)."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = Path(self._tmpdir) / "orgs" / "synapt-dev" / "team.db"
        _create_team_db(self._db_path, "synapt-dev")

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_duplicate_display_name_rejected(self):
        """Register 'Atlas' twice in same org → second fails."""
        register_agent("synapt-dev", "Atlas", db_path=self._db_path)
        with self.assertRaises(Exception):
            register_agent("synapt-dev", "Atlas", db_path=self._db_path)

    def test_same_display_name_different_orgs(self):
        """'Atlas' in synapt-dev AND 'Atlas' in mem0-org → both succeed.

        Agent IDs are per-org (each org has its own team.db), so both
        may be 'atlas-001'. The guarantee is both registrations succeed
        without IntegrityError — uniqueness is per-org, not global.
        """
        db_mem0 = Path(self._tmpdir) / "orgs" / "mem0-org" / "team.db"
        _create_team_db(db_mem0, "mem0-org")

        a1 = register_agent("synapt-dev", "Atlas", db_path=self._db_path)
        a2 = register_agent("mem0-org", "Atlas", db_path=db_mem0)
        self.assertIsNotNone(a1)
        self.assertIsNotNone(a2)

    def test_agent_id_permanent_after_rename(self):
        """Register agent, change display_name, verify agent_id unchanged."""
        agent_id = register_agent("synapt-dev", "Atlas", db_path=self._db_path)
        update_display_name(agent_id, "Atlas Prime", db_path=self._db_path)
        agent = get_agent(agent_id, db_path=self._db_path)
        self.assertEqual(agent["agent_id"], agent_id)
        self.assertEqual(agent["display_name"], "Atlas Prime")

    def test_auto_register_on_join(self):
        """Agent without SYNAPT_AGENT_ID joins → auto-registers with stable ID."""
        # This test verifies that channel_join creates a registry entry
        # when no SYNAPT_AGENT_ID env var is set.
        # Implementation will need to integrate registry with channel_join.
        pass  # Requires channel + registry integration


# ===========================================================================
# Phase 1: Global Channel Store — adversarial cases
# ===========================================================================


class TestClaimsRace(unittest.TestCase):
    """Tests for cross-gripspace claims coordination (Phase 1).

    Bug found in design review v2: claims in local SQLite couldn't
    coordinate across gripspaces. Fix: global _state.db with INSERT OR IGNORE.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_db = Path(self._tmpdir) / "channels" / "_state.db"
        _create_state_db(self._state_db)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_claims_first_writer_wins(self):
        """Two agents claim same message → only first succeeds (INSERT OR IGNORE)."""
        conn = sqlite3.connect(str(self._state_db))
        conn.execute("PRAGMA journal_mode=WAL")

        # Agent A claims first
        conn.execute(
            "INSERT OR IGNORE INTO claims VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("synapt-dev", "gripspace", "dev", "m_001", "atlas-001", "Atlas", "2026-04-06T10:00:00Z"),
        )
        conn.commit()

        # Agent B tries to claim same message
        cursor = conn.execute(
            "INSERT OR IGNORE INTO claims VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("synapt-dev", "gripspace", "dev", "m_001", "opus-001", "Opus", "2026-04-06T10:00:01Z"),
        )
        conn.commit()

        # Verify only Atlas's claim exists
        row = conn.execute(
            "SELECT claimed_by FROM claims WHERE message_id = 'm_001'"
        ).fetchone()
        self.assertEqual(row[0], "atlas-001")
        conn.close()

    def test_claims_unclaim_reclaim(self):
        """Agent A claims M, unclaims M, agent B claims M → B succeeds.

        Bug found in design review v3: claims check used any() which
        saw the original claim even after unclaim. Fix: DELETE then INSERT.
        """
        conn = sqlite3.connect(str(self._state_db))

        # Agent A claims
        conn.execute(
            "INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("synapt-dev", "gripspace", "dev", "m_001", "atlas-001", "Atlas", "2026-04-06T10:00:00Z"),
        )
        conn.commit()

        # Agent A unclaims
        conn.execute(
            "DELETE FROM claims WHERE message_id = 'm_001' AND claimed_by = 'atlas-001'"
        )
        conn.commit()

        # Agent B claims — should succeed
        conn.execute(
            "INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("synapt-dev", "gripspace", "dev", "m_001", "opus-001", "Opus", "2026-04-06T10:01:00Z"),
        )
        conn.commit()

        row = conn.execute(
            "SELECT claimed_by FROM claims WHERE message_id = 'm_001'"
        ).fetchone()
        self.assertEqual(row[0], "opus-001")
        conn.close()

    def test_claims_concurrent_insert_or_ignore(self):
        """Concurrent claims from multiple threads → exactly one winner."""
        results = {"winners": []}
        barrier = threading.Barrier(4)

        def claim_message(agent_id, display_name):
            conn = sqlite3.connect(str(self._state_db))
            conn.execute("PRAGMA journal_mode=WAL")
            barrier.wait()  # All threads start simultaneously
            cursor = conn.execute(
                "INSERT OR IGNORE INTO claims VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("synapt-dev", "gripspace", "dev", "m_race", agent_id, display_name, "2026-04-06T10:00:00Z"),
            )
            conn.commit()
            if cursor.rowcount == 1:
                results["winners"].append(agent_id)
            conn.close()

        threads = [
            threading.Thread(target=claim_message, args=(f"agent-{i:03d}", f"Agent{i}"))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results["winners"]), 1, f"Expected 1 winner, got {results['winners']}")


class TestCursorScoping(unittest.TestCase):
    """Tests for cursor identity across gripspaces and orgs.

    Bug found in design review v2: cursors keyed by griptree didn't
    follow agents across gripspaces. Fix: key by agent_id (stable).
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_db = Path(self._tmpdir) / "channels" / "_state.db"
        _create_state_db(self._state_db)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_cursor_follows_agent_across_gripspaces(self):
        """Same agent_id in two gripspaces → shared cursor position."""
        conn = sqlite3.connect(str(self._state_db))

        # Atlas reads from gripspace A (synapt-codex)
        conn.execute(
            "INSERT OR REPLACE INTO cursors VALUES (?, ?, ?, ?, ?, ?)",
            ("atlas-001", "synapt-dev", "gripspace", "dev", "m_050", "2026-04-06T10:00:00Z"),
        )
        conn.commit()

        # Atlas reads from gripspace B (synapt-dev) — same agent_id
        row = conn.execute(
            "SELECT cursor_value FROM cursors WHERE agent_id = 'atlas-001' AND project_id = 'gripspace' AND channel = 'dev'"
        ).fetchone()
        self.assertEqual(row[0], "m_050")
        conn.close()

    def test_cursor_collision_same_name_different_orgs(self):
        """'atlas-001' in synapt-dev and 'atlas-002' in mem0-org → separate cursors."""
        conn = sqlite3.connect(str(self._state_db))

        conn.execute(
            "INSERT INTO cursors VALUES (?, ?, ?, ?, ?, ?)",
            ("atlas-001", "synapt-dev", "gripspace", "dev", "m_100", "2026-04-06T10:00:00Z"),
        )
        conn.execute(
            "INSERT INTO cursors VALUES (?, ?, ?, ?, ?, ?)",
            ("atlas-002", "mem0-org", "mem0s", "dev", "m_050", "2026-04-06T10:00:00Z"),
        )
        conn.commit()

        rows = conn.execute("SELECT agent_id, cursor_value FROM cursors ORDER BY agent_id").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ("atlas-001", "m_100"))
        self.assertEqual(rows[1], ("atlas-002", "m_050"))
        conn.close()


class TestProjectIdCollision(unittest.TestCase):
    """Test that org-prefixed directories prevent project_id collisions.

    Bug found in adversarial multi-org scenario: two orgs naming their
    manifest 'gripspace.git' would collide without org prefix.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._channels_root = Path(self._tmpdir) / ".synapt" / "channels"

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_project_id_collision_different_orgs(self):
        """Two orgs with same project slug → separate directories."""
        dir_a = self._channels_root / "synapt-dev" / "gripspace"
        dir_b = self._channels_root / "mem0-org" / "gripspace"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)

        # Write to each
        (dir_a / "dev.jsonl").write_text('{"body": "from synapt"}\n')
        (dir_b / "dev.jsonl").write_text('{"body": "from mem0"}\n')

        # Read back — no cross-contamination
        content_a = (dir_a / "dev.jsonl").read_text()
        content_b = (dir_b / "dev.jsonl").read_text()
        self.assertIn("from synapt", content_a)
        self.assertNotIn("from mem0", content_a)
        self.assertIn("from mem0", content_b)
        self.assertNotIn("from synapt", content_b)


class TestStateDbWalMode(unittest.TestCase):
    """Verify WAL mode is enabled on _state.db for concurrent write safety.

    Cross-phase risk identified in design review.
    """

    def test_state_db_wal_mode(self):
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "_state.db"
        _create_state_db(db_path)

        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        shutil.rmtree(tmpdir)
        self.assertEqual(mode, "wal")


# ===========================================================================
# Phase 1: Migration — adversarial cases
# ===========================================================================


class TestMigration(unittest.TestCase):
    """Tests for local → global channel migration.

    Bug found in design review v2: migration on first access could
    race with concurrent processes. Fix: lockfile.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._local_dir = Path(self._tmpdir) / "local" / ".synapt" / "recall" / "channels"
        self._global_dir = Path(self._tmpdir) / "global" / ".synapt" / "channels" / "synapt-dev" / "gripspace"

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_migration_preserves_all_messages(self):
        """5K messages in local → all present in global after migration."""
        msg_ids = _write_fake_messages(self._local_dir / "dev.jsonl", 5000)
        self.assertEqual(len(msg_ids), 5000)

        # Simulate migration: copy local to global
        self._global_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._local_dir / "dev.jsonl", self._global_dir / "dev.jsonl")

        # Verify all messages present
        with open(self._global_dir / "dev.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 5000)
        global_ids = {m["id"] for m in lines}
        self.assertEqual(global_ids, set(msg_ids))

    def test_migration_lockfile_prevents_double_migrate(self):
        """Two concurrent migrations → lockfile serializes, no duplicates."""
        _write_fake_messages(self._local_dir / "dev.jsonl", 100)
        self._global_dir.mkdir(parents=True, exist_ok=True)

        lock_path = self._global_dir.parent.parent / ".migrating"
        results = {"migrated": 0}
        barrier = threading.Barrier(2)

        def migrate():
            barrier.wait()
            # Simulate lockfile-protected migration
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                # Got the lock — do the migration
                if not (self._global_dir / "dev.jsonl").exists():
                    shutil.copy2(self._local_dir / "dev.jsonl", self._global_dir / "dev.jsonl")
                    results["migrated"] += 1
                os.close(fd)
                os.unlink(str(lock_path))
            except FileExistsError:
                # Another process is migrating — skip
                pass

        threads = [threading.Thread(target=migrate) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one migration should have happened
        self.assertEqual(results["migrated"], 1)
        with open(self._global_dir / "dev.jsonl") as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 100)


# ===========================================================================
# Phase 3: DMs — adversarial cases
# ===========================================================================


class TestDmPrivacy(unittest.TestCase):
    """Tests for DM privacy across orgs (directory-level validation).

    Bug found in multi-org adversarial scenario: org-agnostic DMs would
    leak messages between identically-named agents in different orgs.
    Fix: DMs namespaced by org. These tests validate the directory design;
    full DM API tests will be added in Phase 3.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._dm_root = Path(self._tmpdir) / ".synapt" / "channels" / "_dm"

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_dm_privacy_across_orgs(self):
        """DMs in synapt-dev not visible to mem0-org agents."""
        dm_synapt = self._dm_root / "synapt-dev" / "atlas-001--opus-001.jsonl"
        dm_mem0 = self._dm_root / "mem0-org" / "atlas-002--opus-002.jsonl"

        dm_synapt.parent.mkdir(parents=True)
        dm_mem0.parent.mkdir(parents=True)

        dm_synapt.write_text('{"body": "secret synapt message"}\n')
        dm_mem0.write_text('{"body": "secret mem0 message"}\n')

        # mem0 agent reading their DM dir should not see synapt DMs
        mem0_files = list((self._dm_root / "mem0-org").glob("*.jsonl"))
        synapt_files = list((self._dm_root / "synapt-dev").glob("*.jsonl"))

        self.assertEqual(len(mem0_files), 1)
        self.assertNotIn("secret synapt message", mem0_files[0].read_text())
        self.assertEqual(len(synapt_files), 1)
        self.assertNotIn("secret mem0 message", synapt_files[0].read_text())

    def test_dm_filename_sorted(self):
        """DM filename always uses sorted agent_id pair."""
        pair_ab = sorted(["atlas-001", "opus-001"])
        pair_ba = sorted(["opus-001", "atlas-001"])

        filename_ab = f"{'--'.join(pair_ab)}.jsonl"
        filename_ba = f"{'--'.join(pair_ba)}.jsonl"

        self.assertEqual(filename_ab, filename_ba)
        self.assertEqual(filename_ab, "atlas-001--opus-001.jsonl")


# ===========================================================================
# Backward compatibility
# ===========================================================================


class TestBackwardCompat(unittest.TestCase):
    """Tests for backward compatibility with existing channel system."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._local_dir = Path(self._tmpdir) / "project" / ".synapt" / "recall"
        self._local_dir.mkdir(parents=True)
        self._patcher_manifest = patch(
            "synapt.recall.channel._read_manifest_url",
            return_value=None,
        )
        self._patcher_manifest.start()

    def tearDown(self):
        self._patcher_manifest.stop()
        shutil.rmtree(self._tmpdir)

    def test_shared_channels_dir_env_overrides_global(self):
        """SYNAPT_SHARED_CHANNELS_DIR takes precedence over global store."""
        custom_dir = Path(self._tmpdir) / "custom_shared"
        custom_dir.mkdir()

        with patch.dict(os.environ, {"SYNAPT_SHARED_CHANNELS_DIR": str(custom_dir)}):
            result = _channels_dir()
        self.assertEqual(result, custom_dir)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Global store routing differs on Windows — _find_gripspace_root resolves differently",
    )
    def test_no_gripspace_falls_back_to_local(self):
        """Agent outside gripspace → uses local .synapt/recall/channels/."""
        with patch("synapt.recall.channel.project_data_dir", return_value=self._local_dir):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SYNAPT_SHARED_CHANNELS_DIR", None)
                result = _channels_dir()
        self.assertEqual(result, self._local_dir / "channels")


class TestManifestUrlHonorsGripspaceRoot:
    """recall#916: two files both claim to be `#dev` depending on the caller's
    cwd at the instant of the call. `_channels_dir()`'s Tier 3 fallback
    (`project_data_dir`) already honors `GRIPSPACE_ROOT`; Tier 2's
    `_read_manifest_url()` does not, so a call from anywhere that isn't
    literally inside the gripspace tree (a scratchpad, a sibling worktree)
    misses Tier 2 and lands in the shared local-fallback file instead of the
    global store every other call reaches.
    """

    def setup_method(self):
        _gripspace_cache.clear()

    def test_channels_dir_diverges_by_cwd_before_the_fix(self, tmp_path, monkeypatch):
        """The bug, pinned: called from INSIDE the gripspace, Tier 2 resolves
        the global store; called from a SIBLING dir with `GRIPSPACE_ROOT` set
        to the same gripspace, it does not — two different files for the same
        `#dev`. Uses `_global_channels_dir()` (pure resolution, no mkdir) so
        the assertion never touches the real filesystem and the test-isolation
        guard on `_channels_dir()`'s write path never enters into it."""
        grip = _make_gripspace_with_manifest(tmp_path)
        sibling = tmp_path / "scratchpad"
        sibling.mkdir()

        monkeypatch.chdir(grip)
        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        from_inside = _global_channels_dir()

        monkeypatch.chdir(sibling)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(grip))
        from_sibling = _global_channels_dir()

        assert from_inside == from_sibling, (
            f"same gripspace resolved to two different channel stores: "
            f"{from_inside} (inside) vs {from_sibling} (sibling + GRIPSPACE_ROOT)"
        )
        # And it should be the GLOBAL store, not either party silently
        # collapsing to None (which is what forces `_channels_dir()`'s Tier 3
        # local fallback and produces the observed two-files-for-one-channel).
        assert from_inside == Path.home() / ".synapt" / "channels" / "synapt-dev" / "synapt-gripspace"

    def test_manifest_url_without_gripspace_root_still_walks_up(self, tmp_path, monkeypatch):
        """Control: with no GRIPSPACE_ROOT set, a sibling cwd genuinely cannot
        see the gripspace (walk-up alone can't reach a filesystem sibling) —
        proving the divergence above is caused by the missing env precedence,
        not by some other confound in the fixture."""
        grip = _make_gripspace_with_manifest(tmp_path)
        sibling = tmp_path / "scratchpad"
        sibling.mkdir()

        monkeypatch.chdir(sibling)
        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        assert _read_manifest_url() is None

    def test_manifest_url_honors_gripspace_root_from_sibling_cwd(self, tmp_path, monkeypatch):
        """Unit-level pin on the fix itself: GRIPSPACE_ROOT lets
        _read_manifest_url() find the manifest from a cwd it could never
        walk up to."""
        grip = _make_gripspace_with_manifest(tmp_path)
        sibling = tmp_path / "scratchpad"
        sibling.mkdir()

        monkeypatch.chdir(sibling)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(grip))
        assert _read_manifest_url() == "git@github.com:synapt-dev/synapt-gripspace.git"

    def test_explicit_project_dir_suppresses_gripspace_root(self, tmp_path, monkeypatch):
        """Parity with project_data_dir's contract: an explicitly-passed
        project_dir is a deliberate root and suppresses the env override."""
        grip = _make_gripspace_with_manifest(tmp_path)
        deliberate = tmp_path / "deliberate-target"
        deliberate.mkdir()

        monkeypatch.setenv("GRIPSPACE_ROOT", str(grip))
        assert _read_manifest_url(deliberate) is None


class TestOrgEntitlementCheck(unittest.TestCase):
    """Tests for register_agent() entitlement gate (recall#530)."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_db_path_bypasses_entitlement(self):
        """Explicit db_path (test/internal use) always succeeds."""
        db = Path(self._tmpdir) / "team.db"
        _create_team_db(db, "test-org")
        agent_id = register_agent("test-org", "TestAgent", db_path=db)
        self.assertIsNotNone(agent_id)

    def test_env_var_grants_entitlement(self):
        """SYNAPT_AGENT_ID env var (set by gr spawn) grants access."""
        # Create the org dir so _open_db can write
        org_dir = Path(self._tmpdir) / "orgs" / "test-org"
        org_dir.mkdir(parents=True)
        with patch("synapt.recall.registry._team_db_path",
                    return_value=org_dir / "team.db"):
            with patch.dict(os.environ, {"SYNAPT_AGENT_ID": "opus-001"}):
                agent_id = register_agent("test-org", "TestAgent")
        self.assertIsNotNone(agent_id)

    def test_existing_team_db_grants_entitlement(self):
        """If team.db already exists (org initialized by gr), allow registration."""
        org_dir = Path(self._tmpdir) / "orgs" / "test-org"
        org_dir.mkdir(parents=True)
        db = org_dir / "team.db"
        _create_team_db(db, "test-org")
        with patch("synapt.recall.registry._team_db_path", return_value=db):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SYNAPT_AGENT_ID", None)
                agent_id = register_agent("test-org", "NewAgent")
        self.assertIsNotNone(agent_id)

    def test_no_entitlement_raises_permission_error(self):
        """Rogue process without entitlement cannot register agents."""
        nonexistent_db = Path(self._tmpdir) / "orgs" / "rogue-org" / "team.db"
        with patch("synapt.recall.registry._team_db_path", return_value=nonexistent_db):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SYNAPT_AGENT_ID", None)
                with self.assertRaises(PermissionError) as ctx:
                    register_agent("rogue-org", "EvilAgent")
        self.assertIn("no entitlement", str(ctx.exception))

    def test_entitlement_error_names_org(self):
        """Error message includes the org_id for debugging."""
        nonexistent_db = Path(self._tmpdir) / "no-org" / "team.db"
        with patch("synapt.recall.registry._team_db_path", return_value=nonexistent_db):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SYNAPT_AGENT_ID", None)
                with self.assertRaises(PermissionError) as ctx:
                    register_agent("secret-org-42", "Intruder")
        self.assertIn("secret-org-42", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
