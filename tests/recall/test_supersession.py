"""Tests for knowledge supersession — Phase 8a of adaptive memory.

Covers:
- Schema: temporal columns (valid_from, valid_until, version, lineage_id)
- Schema: pending_contradictions table
- Storage: knowledge CRUD with temporal fields
- Storage: lineage queries
- Storage: contradiction queue (add, list, resolve)
- Core: include_historical threading
- Core: confidence-based lineage dedup
- Core: _format_knowledge_block with historical/temporal labels
- Server: recall_contradict tool, recall_search include_historical
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import synapt.recall.consolidate as consolidate
from synapt.recall.storage import RecallDB
from synapt.recall.core import TranscriptChunk, TranscriptIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path) -> RecallDB:
    return RecallDB(tmp_path / "test.db")


def _make_knowledge_node(
    node_id=None, content="test fact", category="workflow",
    confidence=0.7, status="active", lineage_id="",
    version=1, valid_from=None, valid_until=None,
    source_sessions=None, contradiction_note="",
    superseded_by="",
):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": node_id or uuid.uuid4().hex[:12],
        "content": content,
        "category": category,
        "confidence": confidence,
        "source_sessions": source_sessions or ["sess-1"],
        "created_at": now,
        "updated_at": now,
        "status": status,
        "superseded_by": superseded_by,
        "contradiction_note": contradiction_note,
        "tags": [],
        "valid_from": valid_from,
        "valid_until": valid_until,
        "version": version,
        "lineage_id": lineage_id,
    }


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemaSupersession:
    """Verify temporal columns and pending_contradictions table exist."""

    def test_knowledge_has_temporal_columns(self, tmp_path):
        db = _make_db(tmp_path)
        cols = {
            r[1] for r in db._conn.execute("PRAGMA table_info(knowledge)").fetchall()
        }
        assert "valid_from" in cols
        assert "valid_until" in cols
        assert "version" in cols
        assert "lineage_id" in cols

    def test_pending_contradictions_table_exists(self, tmp_path):
        db = _make_db(tmp_path)
        tables = {
            r[0] for r in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "pending_contradictions" in tables

    def test_pending_contradictions_columns(self, tmp_path):
        db = _make_db(tmp_path)
        cols = {
            r[1] for r in db._conn.execute(
                "PRAGMA table_info(pending_contradictions)"
            ).fetchall()
        }
        expected = {
            "id", "old_node_id", "new_content", "category", "reason",
            "source_sessions", "detected_at", "detected_by", "status",
            "resolved_at",
        }
        assert expected <= cols

    def test_pending_contradictions_has_temporal_columns(self, tmp_path):
        # BLOCKER 2 fix (Sentinel, 2026-07-15): the queued-contradiction payload must be able to
        # carry candidate bounds, or they are lost the moment a contradiction is queued.
        db = _make_db(tmp_path)
        cols = {
            r[1] for r in db._conn.execute(
                "PRAGMA table_info(pending_contradictions)"
            ).fetchall()
        }
        assert "valid_from" in cols
        assert "valid_until" in cols

    def test_migration_handles_partial_temporal_column_state(self, tmp_path):
        """Bug found by adversarial verification workflow (2026-07-15): a DB with ONE of the two
        temporal columns already present (e.g. from a process interrupted mid-migration — OOM
        kill, kill -9, forced container restart — landing between the two sequential ALTER TABLE
        calls) must NOT crash on reopen. The prior combined ``has_temporal`` AND-check gated BOTH
        ALTER statements together, so if only one column was missing it unconditionally tried to
        re-add the one that already existed -> sqlite3.OperationalError: duplicate column name
        -> a PERMANENT poison-pill (every future RecallDB(path) on that file crashes the same
        way). Fixed per-column, matching the existing idiom in _migrate_knowledge_table /
        _migrate_access_stats_table in this same file."""
        import sqlite3

        for missing_col, present_col in [("valid_until", "valid_from"), ("valid_from", "valid_until")]:
            db_path = tmp_path / f"partial_{present_col}.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "CREATE TABLE pending_contradictions ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT, old_node_id TEXT,"
                "  new_content TEXT NOT NULL, category TEXT NOT NULL DEFAULT '',"
                "  reason TEXT NOT NULL DEFAULT '', source_sessions TEXT NOT NULL DEFAULT '[]',"
                "  detected_at TEXT NOT NULL, detected_by TEXT NOT NULL DEFAULT 'co-retrieval',"
                "  status TEXT NOT NULL DEFAULT 'pending', resolved_at TEXT, claim_text TEXT,"
                f"  {present_col} TEXT"
                ")"
            )
            conn.commit()
            conn.close()

            db = RecallDB(db_path)  # must NOT raise
            cols = {
                r[1] for r in db._conn.execute(
                    "PRAGMA table_info(pending_contradictions)"
                ).fetchall()
            }
            assert missing_col in cols
            assert present_col in cols

    def test_migration_reopen_after_partial_state_is_idempotent(self, tmp_path):
        """The fix must also be idempotent — opening the same partially-migrated (now fully
        migrated after the first open) DB a second time must not raise either."""
        import sqlite3

        db_path = tmp_path / "partial_reopen.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE pending_contradictions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT, old_node_id TEXT,"
            "  new_content TEXT NOT NULL, category TEXT NOT NULL DEFAULT '',"
            "  reason TEXT NOT NULL DEFAULT '', source_sessions TEXT NOT NULL DEFAULT '[]',"
            "  detected_at TEXT NOT NULL, detected_by TEXT NOT NULL DEFAULT 'co-retrieval',"
            "  status TEXT NOT NULL DEFAULT 'pending', resolved_at TEXT, claim_text TEXT,"
            "  valid_from TEXT"
            ")"
        )
        conn.commit()
        conn.close()

        RecallDB(db_path)  # first open: migrates valid_until in
        RecallDB(db_path)  # second open: must no-op, not raise

    def test_migration_adds_temporal_columns_to_pending_contradictions(self, tmp_path):
        """Simulate an old DB whose pending_contradictions predates the temporal columns."""
        db_path = tmp_path / "old_pending.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE pending_contradictions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT, old_node_id TEXT,"
            "  new_content TEXT NOT NULL, category TEXT NOT NULL DEFAULT '',"
            "  reason TEXT NOT NULL DEFAULT '', source_sessions TEXT NOT NULL DEFAULT '[]',"
            "  detected_at TEXT NOT NULL, detected_by TEXT NOT NULL DEFAULT 'co-retrieval',"
            "  status TEXT NOT NULL DEFAULT 'pending', resolved_at TEXT, claim_text TEXT"
            ")"
        )
        conn.commit()
        conn.close()
        db = RecallDB(db_path)  # opening runs migrations
        cols = {
            r[1] for r in db._conn.execute(
                "PRAGMA table_info(pending_contradictions)"
            ).fetchall()
        }
        assert "valid_from" in cols
        assert "valid_until" in cols

    def test_migration_adds_temporal_columns(self, tmp_path):
        """Simulate an old DB missing temporal columns, then migrate."""
        db_path = tmp_path / "old.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE knowledge ("
            "  rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, content TEXT, "
            "  category TEXT, confidence REAL, source_sessions TEXT, "
            "  created_at TEXT, updated_at TEXT, status TEXT, "
            "  superseded_by TEXT, tags TEXT"
            ")"
        )
        conn.commit()
        conn.close()
        # Opening RecallDB runs migrations
        db = RecallDB(db_path)
        cols = {
            r[1] for r in db._conn.execute("PRAGMA table_info(knowledge)").fetchall()
        }
        assert "valid_from" in cols
        assert "lineage_id" in cols


# ---------------------------------------------------------------------------
# Knowledge CRUD with temporal fields
# ---------------------------------------------------------------------------

class TestKnowledgeTemporal:
    """Save and load knowledge nodes with temporal fields."""

    def test_save_and_load_with_temporal(self, tmp_path):
        db = _make_db(tmp_path)
        node = _make_knowledge_node(
            lineage_id="lineage-abc",
            version=2,
            valid_from="2026-01-01T00:00:00+00:00",
        )
        db.save_knowledge_nodes([node])
        loaded = db.load_knowledge_nodes()
        assert len(loaded) == 1
        assert loaded[0]["lineage_id"] == "lineage-abc"
        assert loaded[0]["version"] == 2
        assert loaded[0]["valid_from"] == "2026-01-01T00:00:00+00:00"
        assert loaded[0]["valid_until"] is None

    def test_upsert_updates_temporal(self, tmp_path):
        db = _make_db(tmp_path)
        node = _make_knowledge_node(node_id="n1", version=1)
        db.save_knowledge_nodes([node])

        node["version"] = 2
        node["valid_from"] = "2026-02-01T00:00:00+00:00"
        db.upsert_knowledge_node(node)

        loaded = db.load_knowledge_nodes()
        assert len(loaded) == 1
        assert loaded[0]["version"] == 2

    def test_default_version_is_1(self, tmp_path):
        db = _make_db(tmp_path)
        # Save without explicit version
        node = _make_knowledge_node()
        del node["version"]
        del node["lineage_id"]
        db.save_knowledge_nodes([node])
        loaded = db.load_knowledge_nodes()
        assert loaded[0]["version"] == 1
        assert loaded[0]["lineage_id"] == ""


# ---------------------------------------------------------------------------
# Lineage queries
# ---------------------------------------------------------------------------

class TestKnowledgeLineage:
    """Test knowledge_lineage() — fetch all versions of a fact."""

    def test_lineage_returns_ordered_versions(self, tmp_path):
        db = _make_db(tmp_path)
        lid = "lineage-xyz"
        v1 = _make_knowledge_node(
            node_id="v1", lineage_id=lid, version=1,
            content="old fact", status="contradicted",
            valid_until="2026-03-01T00:00:00+00:00",
        )
        v2 = _make_knowledge_node(
            node_id="v2", lineage_id=lid, version=2,
            content="new fact",
            valid_from="2026-03-01T00:00:00+00:00",
        )
        db.save_knowledge_nodes([v2, v1])  # Insert out of order

        lineage = db.knowledge_lineage(lid)
        assert len(lineage) == 2
        assert lineage[0]["version"] == 1
        assert lineage[1]["version"] == 2

    def test_lineage_empty_for_unknown_id(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.knowledge_lineage("nonexistent") == []

    def test_lineage_empty_for_blank_id(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.knowledge_lineage("") == []


# ---------------------------------------------------------------------------
# FTS search with include_historical
# ---------------------------------------------------------------------------

class TestKnowledgeFtsHistorical:
    """Test knowledge_fts_search with include_historical flag."""

    def test_default_excludes_contradicted(self, tmp_path):
        db = _make_db(tmp_path)
        active = _make_knowledge_node(
            node_id="a1", content="Python uses pytest for testing",
            status="active",
        )
        contradicted = _make_knowledge_node(
            node_id="c1", content="Python uses unittest for testing",
            status="contradicted",
        )
        db.save_knowledge_nodes([active, contradicted])

        results = db.knowledge_fts_search("Python testing")
        ids = {db._knowledge_dict_from_row(
            db._conn.execute("SELECT * FROM knowledge WHERE rowid = ?", (r,)).fetchone()
        )["id"] for r, _ in results}
        assert "a1" in ids
        assert "c1" not in ids

    def test_include_historical_returns_contradicted(self, tmp_path):
        db = _make_db(tmp_path)
        active = _make_knowledge_node(
            node_id="a1", content="Python uses pytest for testing",
            status="active",
        )
        contradicted = _make_knowledge_node(
            node_id="c1", content="Python uses unittest for testing",
            status="contradicted",
        )
        db.save_knowledge_nodes([active, contradicted])

        results = db.knowledge_fts_search("Python testing", include_historical=True)
        ids = {db._knowledge_dict_from_row(
            db._conn.execute("SELECT * FROM knowledge WHERE rowid = ?", (r,)).fetchone()
        )["id"] for r, _ in results}
        assert "a1" in ids
        assert "c1" in ids

    def test_default_excludes_contested(self, tmp_path):
        """Fix B contested-memory-lifecycle reframe (internal design spec, section 10.6,
        fixture d): a contested node is
        hidden from default search by the SAME status != 'active' gate that already hides
        contradicted/superseded/retracted nodes -- no new mechanism, the existing one just
        gets a new status value."""
        db = _make_db(tmp_path)
        active = _make_knowledge_node(
            node_id="a1", content="Python uses pytest for testing",
            status="active",
        )
        contested = _make_knowledge_node(
            node_id="k1", content="Python uses unittest for testing",
            status="contested", confidence=0.3,
        )
        db.save_knowledge_nodes([active, contested])

        results = db.knowledge_fts_search("Python testing")
        ids = {db._knowledge_dict_from_row(
            db._conn.execute("SELECT * FROM knowledge WHERE rowid = ?", (r,)).fetchone()
        )["id"] for r, _ in results}
        assert "a1" in ids
        assert "k1" not in ids

    def test_include_historical_returns_contested(self, tmp_path):
        """The include_historical=True escape hatch (already MCP-exposed, server.py's
        recall_search) surfaces contested nodes too -- "surfaced as disputed," not hidden
        forever."""
        db = _make_db(tmp_path)
        active = _make_knowledge_node(
            node_id="a1", content="Python uses pytest for testing",
            status="active",
        )
        contested = _make_knowledge_node(
            node_id="k1", content="Python uses unittest for testing",
            status="contested", confidence=0.3,
        )
        db.save_knowledge_nodes([active, contested])

        results = db.knowledge_fts_search("Python testing", include_historical=True)
        ids = {db._knowledge_dict_from_row(
            db._conn.execute("SELECT * FROM knowledge WHERE rowid = ?", (r,)).fetchone()
        )["id"] for r, _ in results}
        assert "a1" in ids
        assert "k1" in ids


# ---------------------------------------------------------------------------
# Pending contradictions
# ---------------------------------------------------------------------------

class TestPendingContradictions:
    """Test contradiction queue: add, list, resolve."""

    def test_add_and_list(self, tmp_path):
        db = _make_db(tmp_path)
        cid = db.add_pending_contradiction(
            old_node_id="n1",
            new_content="updated fact",
            category="workflow",
            reason="new evidence found",
            source_sessions=["sess-5"],
            detected_by="consolidation",
        )
        assert cid > 0

        pending = db.list_pending_contradictions()
        assert len(pending) == 1
        assert pending[0]["old_node_id"] == "n1"
        assert pending[0]["new_content"] == "updated fact"
        assert pending[0]["reason"] == "new evidence found"
        assert pending[0]["detected_by"] == "consolidation"
        assert pending[0]["source_sessions"] == ["sess-5"]

    def test_resolve_confirmed(self, tmp_path):
        db = _make_db(tmp_path)
        cid = db.add_pending_contradiction("n1", "new fact")
        ok = db.resolve_contradiction(cid, "confirmed")
        assert ok

        # No longer pending
        pending = db.list_pending_contradictions()
        assert len(pending) == 0

    def test_resolve_dismissed(self, tmp_path):
        db = _make_db(tmp_path)
        cid = db.add_pending_contradiction("n1", "new fact")
        ok = db.resolve_contradiction(cid, "dismissed")
        assert ok
        assert len(db.list_pending_contradictions()) == 0

    def test_resolve_invalid_status(self, tmp_path):
        db = _make_db(tmp_path)
        cid = db.add_pending_contradiction("n1", "new fact")
        ok = db.resolve_contradiction(cid, "invalid")
        assert not ok
        assert len(db.list_pending_contradictions()) == 1

    def test_resolve_nonexistent(self, tmp_path):
        db = _make_db(tmp_path)
        ok = db.resolve_contradiction(999, "confirmed")
        assert not ok

    def test_resolve_already_resolved(self, tmp_path):
        db = _make_db(tmp_path)
        cid = db.add_pending_contradiction("n1", "new fact")
        db.resolve_contradiction(cid, "confirmed")
        # Second resolve should fail
        ok = db.resolve_contradiction(cid, "dismissed")
        assert not ok

    def test_multiple_pending(self, tmp_path):
        db = _make_db(tmp_path)
        db.add_pending_contradiction("n1", "fact A")
        db.add_pending_contradiction("n2", "fact B")
        db.add_pending_contradiction("n3", "fact C")

        pending = db.list_pending_contradictions()
        assert len(pending) == 3

        # Resolve one
        db.resolve_contradiction(pending[0]["id"], "confirmed")
        assert len(db.list_pending_contradictions()) == 2


# ---------------------------------------------------------------------------
# Confidence-based lineage dedup
# ---------------------------------------------------------------------------

class TestConfidenceBasedDedup:
    """Test _dedup_knowledge_by_lineage in core.py."""

    def test_keeps_higher_confidence(self):
        nodes = [
            {"id": "v1", "lineage_id": "L1", "confidence": 0.5, "content": "old"},
            {"id": "v2", "lineage_id": "L1", "confidence": 0.8, "content": "new"},
        ]
        result = TranscriptIndex._dedup_knowledge_by_lineage(nodes)
        assert len(result) == 1
        assert result[0]["id"] == "v2"

    def test_no_lineage_passes_through(self):
        nodes = [
            {"id": "a", "lineage_id": "", "confidence": 0.5},
            {"id": "b", "confidence": 0.6},  # Missing lineage_id key
        ]
        result = TranscriptIndex._dedup_knowledge_by_lineage(nodes)
        assert len(result) == 2

    def test_different_lineages_kept(self):
        nodes = [
            {"id": "v1", "lineage_id": "L1", "confidence": 0.5},
            {"id": "v2", "lineage_id": "L2", "confidence": 0.8},
        ]
        result = TranscriptIndex._dedup_knowledge_by_lineage(nodes)
        assert len(result) == 2

    def test_mixed_lineage_and_no_lineage(self):
        nodes = [
            {"id": "a", "lineage_id": "", "confidence": 0.9},
            {"id": "v1", "lineage_id": "L1", "confidence": 0.3},
            {"id": "v2", "lineage_id": "L1", "confidence": 0.7},
            {"id": "b", "lineage_id": "", "confidence": 0.4},
        ]
        result = TranscriptIndex._dedup_knowledge_by_lineage(nodes)
        assert len(result) == 3
        ids = {n["id"] for n in result}
        assert ids == {"a", "v2", "b"}

    def test_equal_confidence_keeps_first(self):
        """When confidence is equal, the first node seen wins (strict >)."""
        nodes = [
            {"id": "v1", "lineage_id": "L1", "confidence": 0.5},
            {"id": "v2", "lineage_id": "L1", "confidence": 0.5},
        ]
        result = TranscriptIndex._dedup_knowledge_by_lineage(nodes)
        assert len(result) == 1
        # v2 is last with equal confidence, NOT higher, so v1 wins (> not >=)
        assert result[0]["id"] == "v1"

    def test_three_versions_picks_highest_confidence(self):
        """With 3+ versions, the highest confidence wins regardless of position."""
        nodes = [
            {"id": "v1", "lineage_id": "L1", "confidence": 0.3},
            {"id": "v2", "lineage_id": "L1", "confidence": 0.9},
            {"id": "v3", "lineage_id": "L1", "confidence": 0.6},
        ]
        result = TranscriptIndex._dedup_knowledge_by_lineage(nodes)
        assert len(result) == 1
        assert result[0]["id"] == "v2"


# ---------------------------------------------------------------------------
# Format knowledge block with temporal/historical labels
# ---------------------------------------------------------------------------

class TestFormatKnowledgeBlock:
    """Test _format_knowledge_block for Phase 8a enhancements."""

    def test_active_node_normal_format(self):
        node = _make_knowledge_node(confidence=0.8, category="architecture")
        block = TranscriptIndex._format_knowledge_block(node)
        assert "[knowledge" in block
        assert "architecture (high" in block
        assert "historical" not in block

    def test_contradicted_node_historical_label(self):
        node = _make_knowledge_node(
            status="contradicted",
            contradiction_note="replaced by newer approach",
        )
        block = TranscriptIndex._format_knowledge_block(node)
        assert "CONTRADICTED" in block
        assert "replaced by newer approach" in block

    def test_contested_node_label(self):
        """Fix B contested-memory-lifecycle reframe (internal design spec, section 10.6,
        fixture d)."""
        node = _make_knowledge_node(status="contested", confidence=0.3)
        block = TranscriptIndex._format_knowledge_block(node)
        assert "CONTESTED" in block

    def test_valid_from_only(self):
        node = _make_knowledge_node(valid_from="2026-01-15T00:00:00+00:00")
        block = TranscriptIndex._format_knowledge_block(node)
        assert "Current since 2026-01-15" in block

    def test_valid_range(self):
        node = _make_knowledge_node(
            valid_from="2026-01-01T00:00:00+00:00",
            valid_until="2026-03-01T00:00:00+00:00",
        )
        block = TranscriptIndex._format_knowledge_block(node)
        assert "Valid 2026-01-01 to 2026-03-01" in block

    def test_confidence_labels(self):
        high = TranscriptIndex._format_knowledge_block(
            _make_knowledge_node(confidence=0.8)
        )
        assert "(high" in high

        medium = TranscriptIndex._format_knowledge_block(
            _make_knowledge_node(confidence=0.5)
        )
        assert "(medium" in medium

        low = TranscriptIndex._format_knowledge_block(
            _make_knowledge_node(confidence=0.2)
        )
        assert "(low" in low


# ---------------------------------------------------------------------------
# Server: recall_contradict
# ---------------------------------------------------------------------------

class TestRecallContradict:
    """Test the recall_contradict MCP tool."""

    def test_list_empty(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}

        with patch("synapt.recall.server._get_index", return_value=index):
            result = recall_contradict(action="list")
        assert "No pending contradictions" in result

    def test_list_shows_pending(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        # Add a knowledge node and a contradiction
        node = _make_knowledge_node(node_id="old-1", content="use unittest")
        db.save_knowledge_nodes([node])
        db.add_pending_contradiction(
            "old-1", "use pytest instead",
            reason="pytest is now standard",
        )
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}

        with patch("synapt.recall.server._get_index", return_value=index):
            result = recall_contradict(action="list")
        assert "use pytest instead" in result
        assert "use unittest" in result

    def test_resolve_confirmed_supersedes(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        node = _make_knowledge_node(node_id="old-1", content="use unittest")
        db.save_knowledge_nodes([node])
        cid = db.add_pending_contradiction(
            "old-1", "use pytest instead",
            category="tooling",
            reason="pytest is standard",
        )
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="resolve",
                    contradiction_id=cid,
                    resolution="confirmed",
                )
        assert "confirmed" in result
        assert "superseded" in result

        # Verify old node is now contradicted with full metadata
        nodes = db.load_knowledge_nodes()
        old = [n for n in nodes if n["id"] == "old-1"]
        assert len(old) == 1
        assert old[0]["status"] == "contradicted"
        assert old[0]["valid_until"] is not None  # Timestamp set
        assert old[0]["contradiction_note"] == "pytest is standard"
        # Old node's lineage_id backfilled during bootstrap
        assert old[0]["lineage_id"] == "old-1"

        # Verify new node was created with lineage chain
        active = [n for n in nodes if n["status"] == "active"]
        assert len(active) == 1
        new_node = active[0]
        assert new_node["content"] == "use pytest instead"
        assert new_node["version"] == 2
        # Lineage bootstrapped from old node's id (old node had no lineage_id)
        assert new_node["lineage_id"] == "old-1"
        assert new_node["valid_from"] is not None
        # Old node's superseded_by points to new node
        assert old[0]["superseded_by"] == new_node["id"]

    def test_resolve_dismissed_keeps_old(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        node = _make_knowledge_node(node_id="old-1", content="use unittest")
        db.save_knowledge_nodes([node])
        cid = db.add_pending_contradiction("old-1", "use pytest instead")
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="resolve",
                    contradiction_id=cid,
                    resolution="dismissed",
                )
        assert "dismissed" in result

        # Old node unchanged
        nodes = db.load_knowledge_nodes(status="active")
        assert len(nodes) == 1
        assert nodes[0]["content"] == "use unittest"

    def test_resolve_missing_id(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}

        with patch("synapt.recall.server._get_index", return_value=index):
            result = recall_contradict(action="resolve")
        assert "required" in result

    def test_no_index(self):
        from synapt.recall.server import recall_contradict
        with patch("synapt.recall.server._get_index", return_value=None):
            result = recall_contradict(action="list")
        assert "No index" in result

    def test_chain_supersession_propagates_lineage(self, tmp_path):
        """v1→v2→v3 chain: lineage_id propagates through all versions."""
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        # v1 already has a lineage_id (from a prior supersession)
        v1 = _make_knowledge_node(
            node_id="v1", content="original fact",
            lineage_id="lineage-root", version=1,
        )
        db.save_knowledge_nodes([v1])
        cid = db.add_pending_contradiction(
            "v1", "updated fact v2",
            category="workflow",
            reason="new evidence",
        )
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                recall_contradict(
                    action="resolve", contradiction_id=cid,
                    resolution="confirmed",
                )

        nodes = db.load_knowledge_nodes()
        active = [n for n in nodes if n["status"] == "active"]
        assert len(active) == 1
        # Lineage carried forward from v1's existing lineage_id
        assert active[0]["lineage_id"] == "lineage-root"
        assert active[0]["version"] == 2

    def test_confirm_missing_old_node_no_crash(self, tmp_path):
        """Confirming a contradiction for a deleted old node does not crash."""
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        # Add contradiction referencing a node that doesn't exist
        cid = db.add_pending_contradiction("nonexistent", "new fact")
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="resolve", contradiction_id=cid,
                    resolution="confirmed",
                )
        # Should succeed (contradiction resolved) even though supersession
        # couldn't find the old node — it silently skips
        assert "confirmed" in result


# ---------------------------------------------------------------------------
# recall_contradict: Fix B contest resolution (candidate_wins/existing_wins/false_positive)
# ---------------------------------------------------------------------------

class TestRecallContradictContestResolution:
    """Fix B contested-memory-lifecycle reframe (internal design spec, section
    10.4/10.6 fixture c): the 3-way resolution vocabulary
    for a contested pair (new_node_id set on the pending row), distinct from the ordinary
    confirmed/dismissed path exercised by TestRecallContradict above. Same _make_index-style
    setup as TestRecallContradict; a contested pair is two ALREADY-PERSISTED nodes plus one
    pending_contradictions row carrying new_node_id, matching what the contest branch in
    _apply_consolidation_result actually produces (test_fix_b_1a_with_conflict_judge_and_
    source_unit_ids_resolves_via_contest in test_consolidate.py)."""

    def _make_contested_pair(self, tmp_path):
        db = _make_db(tmp_path)
        existing = _make_knowledge_node(
            node_id="existing-1", content="extract path does not share the legacy cache",
            status="contested", confidence=0.3, source_sessions=["s0"],
        )
        candidate = _make_knowledge_node(
            node_id="candidate-1", content="extract path writes to a distinct cache key",
            status="contested", confidence=0.3, source_sessions=["s0", "s1"],
        )
        db.save_knowledge_nodes([existing, candidate])
        cid = db.add_pending_contradiction(
            old_node_id="existing-1",
            new_content=candidate["content"],
            new_node_id="candidate-1",
            category="configuration",
            reason="Chronology: candidate (newer) conflicts with existing (older).",
            detected_by="b3-temporal-conflict-escalation",
        )
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}
        return db, index, cid

    def _make_contested_pair_with_jsonl(self, tmp_path, monkeypatch):
        """Same shape as _make_contested_pair, but ALSO writes both nodes into
        knowledge.jsonl at the REAL project-resolved path (not an ad-hoc DB-only fixture) --
        required to reproduce Sentinel's dual-store durability finding (PR#903
        issuecomment-5037168639): _apply_contest_resolution only ever wrote SQLite, so the
        next JSONL->SQLite sync (_sync_knowledge_to_db, consolidate.py) reads knowledge.jsonl
        as authoritative and reverts a valid resolution back to both-contested. monkeypatch.
        chdir pins Path.cwd() so _knowledge_path()'s bare (no-arg) default resolution inside
        _apply_contest_resolution lands on the SAME file this fixture writes."""
        from synapt.recall.knowledge import KnowledgeNode, append_node
        from synapt.recall.core import project_index_dir, project_data_dir

        monkeypatch.chdir(tmp_path)
        kn_path = project_data_dir(tmp_path) / "knowledge.jsonl"

        existing = KnowledgeNode.create(
            content="extract path does not share the legacy cache",
            category="configuration", source_sessions=["s0"], node_id="existing-1",
        )
        existing.status = "contested"
        existing.confidence = 0.3
        candidate = KnowledgeNode.create(
            content="extract path writes to a distinct cache key",
            category="configuration", source_sessions=["s0", "s1"], node_id="candidate-1",
        )
        candidate.status = "contested"
        candidate.confidence = 0.3
        append_node(existing, kn_path)
        append_node(candidate, kn_path)

        db_path = project_index_dir(tmp_path) / "recall.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = RecallDB(db_path)
        db.save_knowledge_nodes([existing.to_dict(), candidate.to_dict()])
        cid = db.add_pending_contradiction(
            old_node_id="existing-1",
            new_content=candidate.content,
            new_node_id="candidate-1",
            category="configuration",
            reason="Chronology: candidate (newer) conflicts with existing (older).",
            detected_by="b3-temporal-conflict-escalation",
        )
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}
        return db, index, cid, kn_path

    @pytest.mark.parametrize(
        "resolution,candidate_status,existing_status",
        [
            ("candidate_wins", "active", "superseded"),
            ("existing_wins", "stale", "active"),
            ("false_positive", "active", "active"),
        ],
    )
    def test_resolution_survives_the_next_jsonl_sqlite_sync(
        self, tmp_path, monkeypatch, resolution, candidate_status, existing_status,
    ):
        """Sentinel r2 BLOCKING (PR#903 issuecomment-5037168639): a valid resolution must
        survive the next REAL sync, not just look correct in SQLite until the next
        consolidation run silently reverts it. Reproduces the exact mechanism: knowledge.jsonl
        is the sync's authoritative source (_sync_knowledge_to_db does an unconditional
        INSERT OR REPLACE from JSONL into SQLite for every node present), so if resolution
        doesn't ALSO update JSONL, the sync blasts SQLite back to both-contested."""
        from synapt.recall.consolidate import _sync_knowledge_to_db
        from synapt.recall.server import recall_contradict

        db, index, cid, kn_path = self._make_contested_pair_with_jsonl(tmp_path, monkeypatch)

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                recall_contradict(
                    action="resolve", contradiction_id=cid, resolution=resolution,
                )

        # The REAL sync -- not a mock. If _apply_contest_resolution didn't persist to JSONL,
        # this overwrites SQLite's now-correct rows back to knowledge.jsonl's stale contested
        # state, because knowledge.jsonl still says both nodes are contested.
        _sync_knowledge_to_db(tmp_path, kn_path)

        candidate = db.get_knowledge_node("candidate-1")
        existing = db.get_knowledge_node("existing-1")
        assert candidate["status"] == candidate_status, (
            f"candidate reverted to {candidate['status']!r} after sync -- resolution "
            f"{resolution!r} did not survive"
        )
        assert existing["status"] == existing_status, (
            f"existing reverted to {existing['status']!r} after sync -- resolution "
            f"{resolution!r} did not survive"
        )
        # The queue entry must also stay resolved -- a revert-to-contested with an
        # already-cleared queue row would be WORSE than the original bug (stuck contested,
        # invisible, no way to re-resolve).
        assert db.list_pending_contradictions() == []

    def test_resolve_fails_loud_when_candidate_missing_from_sqlite_not_false_success(
        self, tmp_path,
    ):
        """recall#905 (0.17.0 blocker, Opus 2026-07-22, Part 2 -- CO-PRIMARY, not merely
        defense-in-depth): the dogfood-found false-success bug. A freshly-created contest's
        candidate node genuinely does not exist in SQLite until some sync has run (root
        cause, closed separately by Part 1). Before this fix, the caller marked the pending
        row resolved BEFORE attempting the node-level mutation, so a mutation that silently
        no-opped (candidate not found) still reported "resolved" -- false success masking
        real data corruption. This test proves Part 2 holds independent of Part 1: even with
        a node genuinely missing from SQLite for ANY reason, resolve must fail loud (an
        honest error, not a false "resolved" message) and leave the pending row untouched
        (still pending, retryable) rather than silently discarding the review."""
        from synapt.recall.server import recall_contradict

        db = _make_db(tmp_path)
        existing = _make_knowledge_node(
            node_id="existing-1", content="extract path does not share the legacy cache",
            status="contested", confidence=0.3,
        )
        # Candidate deliberately NEVER written to SQLite -- reproduces the exact "freshly
        # created, not yet synced" condition, without depending on a real judge/model.
        db.save_knowledge_nodes([existing])
        cid = db.add_pending_contradiction(
            old_node_id="existing-1",
            new_content="extract path writes to a distinct cache key",
            new_node_id="candidate-1",
            category="configuration",
            detected_by="b3-temporal-conflict-escalation",
        )
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="resolve", contradiction_id=cid, resolution="candidate_wins",
                )

        # "resolved:" (with the colon) is the exact success-message shape this handler uses
        # ("Contradiction #N resolved: <outcome>."); checking for that shape specifically,
        # not the bare substring "resolved" -- an honest failure message legitimately says
        # "could not be resolved", which must NOT trip this assertion.
        assert "resolved:" not in result.lower(), (
            f"FALSE SUCCESS: resolve reported success despite the candidate node not "
            f"existing -- got {result!r}"
        )
        assert result.startswith("Error"), f"expected an honest error message, got {result!r}"
        # The pending row must be UNCHANGED (still pending) -- retryable once the node is
        # actually available, not silently discarded.
        pending = db.list_pending_contradictions()
        assert len(pending) == 1, (
            "pending row was marked resolved despite the mutation failing -- the review is "
            "now lost, unretryable"
        )
        assert pending[0]["id"] == cid
        # And the existing node must be UNCHANGED too -- no partial mutation.
        assert db.get_knowledge_node("existing-1")["status"] == "contested"

    @pytest.mark.skipif(
        not consolidate._MLX_AVAILABLE, reason=consolidate._SKIP_REASON,
    )
    def test_real_contest_then_resolve_then_sync_end_to_end(self, tmp_path, monkeypatch):
        """recall#905 (0.17.0 blocker, Opus 2026-07-22) -- Part 1's primary reproduction, and
        the anti-masking lesson made permanent: this test goes through the REAL contest-
        creation path (_apply_consolidation_result's contest branch, the REAL production
        judge -- not a fake) and does NOT pre-seed either node into SQLite. That freshly-
        created state (candidate absent from SQLite until Part 1's fix upserts it at creation
        time) is the EXACT condition the recall#903 r2 unit tests masked by pre-seeding both
        nodes directly. Found by the post-merge dogfood pass on the real founding dogfood-
        06/07 cache-suffix pair; reproduced here permanently with the same real pair.

        Full real flow: create (real judge flags CONFLICT, both nodes contested, queued) ->
        resolve via recall_contradict (candidate_wins) -> the REAL _sync_knowledge_to_db ->
        assert the resolution survived and the queue is empty."""
        import os
        from unittest.mock import patch as _patch
        from synapt._models.mlx_client import MLXClient, MLXOptions
        from synapt.recall.consolidate import (
            DEFAULT_MODEL,
            _apply_consolidation_result,
            _local_conflict_judge,
            _make_recall_infer,
            _sync_knowledge_to_db,
        )
        import synapt.recall.consolidate as consolidate_mod
        from synapt.recall.core import project_data_dir, project_index_dir
        from synapt.recall.journal import JournalEntry
        from synapt.recall.knowledge import KnowledgeNode, append_node, read_nodes
        from synapt.recall.server import recall_contradict

        stale_content = "extract path does not share the legacy response_cache"
        stale_source = "986d09c3e8bb2ae5:0:decisions:5"
        current_content = (
            'writing extract-path results to response_cache under a distinct ":extract" '
            'key suffix'
        )
        current_source = "986d09c3e8bb2ae5:2:done:6"

        monkeypatch.chdir(tmp_path)
        kn_path = project_data_dir(tmp_path) / "knowledge.jsonl"
        db_path = project_index_dir(tmp_path) / "recall.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Seed ONLY the existing node, as a real prior consolidation + sync cycle would have
        # left it -- already active in both stores.
        existing = KnowledgeNode.create(
            content=stale_content, category="configuration", source_sessions=["s0"],
            source_unit_id=stale_source,
        )
        append_node(existing, kn_path)
        db = RecallDB(db_path)
        db.save_knowledge_nodes([existing.to_dict()])

        client = MLXClient(MLXOptions())
        infer = _make_recall_infer(client, DEFAULT_MODEL)
        conflict_judge = _local_conflict_judge(infer)

        original_fn = consolidate_mod._inline_embedding_dedup

        def _force_match(candidate, existing_nodes, threshold=0.80):
            return (existing_nodes[0], 0.8298) if existing_nodes else (None, 0.0)

        consolidate_mod._inline_embedding_dedup = _force_match
        try:
            parsed = {"nodes": [{
                "action": "create", "content": current_content, "category": "solution",
                "source_unit_id": current_source,
            }]}
            cluster = [JournalEntry(session_id="s1", timestamp="2026-07-13T10:00:00Z", focus="")]
            result = _apply_consolidation_result(
                parsed, [existing], cluster, kn_path, db=db, conflict_judge=conflict_judge,
            )
        finally:
            consolidate_mod._inline_embedding_dedup = original_fn

        assert result.nodes_contested == 1, "real judge did not flag the founding pair"
        pending = db.list_pending_contradictions()
        assert len(pending) == 1
        cid = pending[0]["id"]
        new_node_id = pending[0]["new_node_id"]
        old_node_id = pending[0]["old_node_id"]

        # Scenario-thinking check (Opus, 2026-07-22): Part 1 upserts both nodes into SQLite
        # immediately at creation time -- confirm that does NOT accidentally surface them as
        # active in default retrieval. Both nodes ARE now in SQLite (that's the fix), but
        # both must still be status="contested" and therefore excluded by the existing
        # status != 'active' FTS gate, exactly as they already were pre-fix when only JSONL
        # (not SQLite) had them.
        assert db.get_knowledge_node(new_node_id)["status"] == "contested"
        assert db.get_knowledge_node(old_node_id)["status"] == "contested"
        default_results = db.knowledge_fts_search("response_cache")
        default_ids = {
            db._knowledge_dict_from_row(
                db._conn.execute("SELECT * FROM knowledge WHERE rowid = ?", (r,)).fetchone()
            )["id"]
            for r, _ in default_results
        }
        assert new_node_id not in default_ids, (
            "contested candidate surfaced in DEFAULT search -- upserting to SQLite at "
            "creation time must not bypass the status != 'active' exclusion gate"
        )
        assert old_node_id not in default_ids, (
            "contested existing node surfaced in DEFAULT search -- same exclusion-gate "
            "regression risk"
        )
        historical_results = db.knowledge_fts_search("response_cache", include_historical=True)
        historical_ids = {
            db._knowledge_dict_from_row(
                db._conn.execute("SELECT * FROM knowledge WHERE rowid = ?", (r,)).fetchone()
            )["id"]
            for r, _ in historical_results
        }
        assert new_node_id in historical_ids and old_node_id in historical_ids, (
            "contested nodes must still be reachable via include_historical=True -- "
            "'excluded by default' is not the same as 'gone'"
        )

        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}
        with _patch("synapt.recall.server._get_index", return_value=index):
            with _patch("synapt.recall.server._invalidate_cache"):
                resolve_result = recall_contradict(
                    action="resolve", contradiction_id=cid, resolution="candidate_wins",
                )
        assert "resolved:" in resolve_result.lower(), resolve_result

        _sync_knowledge_to_db(tmp_path, kn_path)

        candidate = db.get_knowledge_node(new_node_id)
        existing_after = db.get_knowledge_node(old_node_id)
        assert candidate["status"] == "active", (
            f"resolution did not survive the real sync -- candidate reverted to "
            f"{candidate['status']!r}"
        )
        assert existing_after["status"] == "superseded"
        assert existing_after["superseded_by"] == new_node_id
        assert db.list_pending_contradictions() == []
        db.close()

    def test_contest_creation_upserts_candidate_to_sqlite_portable(self, tmp_path, monkeypatch):
        """recall#906 Sentinel r2b (0.17.0 blocker): the SOLE guard for Part 1 (contest-
        creation upserting both nodes into SQLite at creation time) was
        test_real_contest_then_resolve_then_sync_end_to_end -- real-MLX-only, skipped on
        every non-Apple-Silicon CI runner. Proven mutation-silent on CI: with ONLY Part 1's
        db.upsert_knowledge_node(contested_node.to_dict()) removed, every non-MLX test
        (including the existing judge-stubbed contest integration test in
        test_consolidate.py, which never asserts SQLite presence/exclusion) stayed green.

        This is the portable sibling: drives the REAL _apply_consolidation_result contest
        branch with a deterministic fake conflict_judge (the injected-seam pattern #903 r2
        already locked -- no MLX, no real model), does NOT pre-seed the candidate into
        SQLite, and asserts exactly what the real-model test's scenario-thinking check
        asserts: candidate genuinely in SQLite, status contested, excluded from default
        search, reachable via include_historical=True. Must go RED with Part 1's upsert
        removed -- verified live before this landed, per the day's rule (reproduce, don't
        assume)."""
        from synapt.recall.consolidate import _apply_consolidation_result
        from synapt.recall.core import project_data_dir, project_index_dir
        from synapt.recall.journal import JournalEntry
        from synapt.recall.knowledge import KnowledgeNode, append_node

        stale_content = "extract path does not share the legacy response_cache"
        stale_source = "986d09c3e8bb2ae5:0:decisions:5"
        current_content = (
            'writing extract-path results to response_cache under a distinct ":extract" '
            'key suffix'
        )
        current_source = "986d09c3e8bb2ae5:2:done:6"

        monkeypatch.chdir(tmp_path)
        kn_path = project_data_dir(tmp_path) / "knowledge.jsonl"
        db_path = project_index_dir(tmp_path) / "recall.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        existing = KnowledgeNode.create(
            content=stale_content, category="configuration", source_sessions=["s0"],
            source_unit_id=stale_source,
        )
        append_node(existing, kn_path)
        db = RecallDB(db_path)
        db.save_knowledge_nodes([existing.to_dict()])

        original_fn = consolidate._inline_embedding_dedup

        def _force_match(candidate, existing_nodes, threshold=0.80):
            return (existing_nodes[0], 0.8298) if existing_nodes else (None, 0.0)

        consolidate._inline_embedding_dedup = _force_match
        try:
            parsed = {"nodes": [{
                "action": "create", "content": current_content, "category": "solution",
                "source_unit_id": current_source,
            }]}
            cluster = [JournalEntry(session_id="s1", timestamp="2026-07-13T10:00:00Z", focus="")]
            result = _apply_consolidation_result(
                parsed, [existing], cluster, kn_path, db=db,
                conflict_judge=lambda candidate_text, existing_text: True,
            )
        finally:
            consolidate._inline_embedding_dedup = original_fn

        assert result.nodes_contested == 1
        pending = db.list_pending_contradictions()
        assert len(pending) == 1
        candidate_id = pending[0]["new_node_id"]

        # Part 1's own assertion: the candidate genuinely exists in SQLite right after
        # creation -- not deferred to some later sync.
        candidate_row = db.get_knowledge_node(candidate_id)
        assert candidate_row is not None, (
            "candidate node not found in SQLite immediately after contest-creation -- "
            "Part 1's db.upsert_knowledge_node did not run (or was removed)"
        )
        assert candidate_row["status"] == "contested"

        # Same exclusion property the real-model test's scenario-thinking check proves --
        # upserting to SQLite at creation must not bypass the status != 'active' gate.
        default_ids = {
            db._knowledge_dict_from_row(
                db._conn.execute("SELECT * FROM knowledge WHERE rowid = ?", (r,)).fetchone()
            )["id"]
            for r, _ in db.knowledge_fts_search("response_cache")
        }
        assert candidate_id not in default_ids
        historical_ids = {
            db._knowledge_dict_from_row(
                db._conn.execute("SELECT * FROM knowledge WHERE rowid = ?", (r,)).fetchone()
            )["id"]
            for r, _ in db.knowledge_fts_search("response_cache", include_historical=True)
        }
        assert candidate_id in historical_ids
        db.close()

    def test_candidate_wins_promotes_candidate_supersedes_existing(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db, index, cid = self._make_contested_pair(tmp_path)

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="resolve", contradiction_id=cid, resolution="candidate_wins",
                )
        assert "candidate wins" in result

        candidate = db.get_knowledge_node("candidate-1")
        existing = db.get_knowledge_node("existing-1")
        assert candidate["status"] == "active"
        assert candidate["confidence"] > 0.3  # recomputed, no longer capped
        assert existing["status"] == "superseded"
        assert existing["superseded_by"] == "candidate-1"
        # Queue entry resolved, no longer pending
        assert db.list_pending_contradictions() == []

    def test_existing_wins_restores_existing_retires_candidate(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db, index, cid = self._make_contested_pair(tmp_path)

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="resolve", contradiction_id=cid, resolution="existing_wins",
                )
        assert "existing node wins" in result

        candidate = db.get_knowledge_node("candidate-1")
        existing = db.get_knowledge_node("existing-1")
        assert existing["status"] == "active"
        assert existing["confidence"] > 0.3  # recomputed, no longer capped
        # Contest, don't discard — retired, not deleted
        assert candidate["status"] == "stale"
        assert db.list_pending_contradictions() == []

    def test_false_positive_restores_both_supersedes_neither(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db, index, cid = self._make_contested_pair(tmp_path)

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="resolve", contradiction_id=cid, resolution="false_positive",
                )
        assert "false positive" in result

        candidate = db.get_knowledge_node("candidate-1")
        existing = db.get_knowledge_node("existing-1")
        assert candidate["status"] == "active"
        assert existing["status"] == "active"
        assert candidate["confidence"] > 0.3
        assert existing["confidence"] > 0.3
        assert candidate["superseded_by"] == ""
        assert existing["superseded_by"] == ""
        assert db.list_pending_contradictions() == []

    def test_invalid_resolution_on_contest_row_rejected_no_mutation(self, tmp_path):
        """A contest row (new_node_id set) rejects the ORDINARY confirmed/dismissed
        vocabulary -- the two vocabularies are deliberately not interchangeable (section
        10.4). Nothing mutates and the row stays pending."""
        from synapt.recall.server import recall_contradict
        db, index, cid = self._make_contested_pair(tmp_path)

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="resolve", contradiction_id=cid, resolution="confirmed",
                )
        assert "Error" in result
        assert db.get_knowledge_node("candidate-1")["status"] == "contested"
        assert db.get_knowledge_node("existing-1")["status"] == "contested"
        assert len(db.list_pending_contradictions()) == 1

    def test_ordinary_contradiction_row_rejects_contest_vocabulary(self, tmp_path):
        """The converse of the test above: an ORDINARY row (new_node_id unset) is untouched
        by the new 3-way vocabulary -- passing 'candidate_wins' against a plain contradiction
        just fails resolve_contradiction's own validation, exactly as any other invalid
        resolution string would have before this reframe existed."""
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        node = _make_knowledge_node(node_id="old-1", content="use unittest")
        db.save_knowledge_nodes([node])
        cid = db.add_pending_contradiction("old-1", "use pytest instead")
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="resolve", contradiction_id=cid, resolution="candidate_wins",
                )
        assert "not found or already resolved" in result
        assert len(db.list_pending_contradictions()) == 1


# ---------------------------------------------------------------------------
# recall_contradict flag action (free-text claims) — #58
# ---------------------------------------------------------------------------

class TestRecallContradictFlag:
    """Test the 'flag' action for user-initiated contradictions."""

    def _make_index(self, db):
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}
        return index

    def test_flag_with_explicit_node_id(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        node = _make_knowledge_node(node_id="n1", content="deploy on Fridays")
        db.save_knowledge_nodes([node])
        index = self._make_index(db)

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="flag",
                    claim="we should never deploy on Fridays",
                    new_content="never deploy on Fridays",
                    old_node_id="n1",
                    reason="outage last Friday",
                )
        assert "flagged" in result
        assert "n1" in result
        pending = db.list_pending_contradictions()
        assert len(pending) == 1
        assert pending[0]["old_node_id"] == "n1"
        assert pending[0]["detected_by"] == "manual"
        assert pending[0]["claim_text"] == "we should never deploy on Fridays"

    def test_flag_free_text_no_matching_node(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        index = self._make_index(db)

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="flag",
                    claim="the API key expires monthly",
                )
        assert "free-text claim" in result
        pending = db.list_pending_contradictions()
        assert len(pending) == 1
        assert pending[0]["old_node_id"] is None
        assert pending[0]["new_content"] == "the API key expires monthly"

    def test_flag_requires_claim_or_content(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        index = self._make_index(db)

        with patch("synapt.recall.server._get_index", return_value=index):
            result = recall_contradict(action="flag")
        assert "required" in result.lower() or "error" in result.lower()

    def test_flag_invalid_node_id(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        index = self._make_index(db)

        with patch("synapt.recall.server._get_index", return_value=index):
            result = recall_contradict(
                action="flag",
                claim="something wrong",
                old_node_id="nonexistent",
            )
        assert "not found" in result

    def test_resolve_free_text_claim_creates_node(self, tmp_path):
        """Confirming a free-text claim (no old_node_id) creates a knowledge node."""
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        cid = db.add_pending_contradiction(
            old_node_id=None,
            new_content="API keys expire every 90 days",
            category="convention",
            detected_by="manual",
            claim_text="API key rotation policy",
        )
        index = self._make_index(db)

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="resolve",
                    contradiction_id=cid,
                    resolution="confirmed",
                )
        assert "confirmed" in result
        assert "knowledge node created" in result
        # Verify a knowledge node was created
        nodes = db.load_knowledge_nodes(status="active")
        assert len(nodes) == 1
        assert nodes[0]["content"] == "API keys expire every 90 days"

    def test_flag_fts_matches_existing_node(self, tmp_path):
        """When no old_node_id given, flag searches FTS and matches best node."""
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        node = _make_knowledge_node(node_id="k1", content="deploy every Tuesday at 3pm")
        db.save_knowledge_nodes([node])
        # Rebuild FTS so the node is searchable
        db._conn.execute(
            "INSERT INTO knowledge_fts(rowid, content, category, tags) "
            "SELECT rowid, content, category, tags FROM knowledge"
        )
        db._conn.commit()
        index = self._make_index(db)

        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="flag",
                    claim="deploy schedule changed to Thursday",
                )
        assert "flagged" in result
        assert "k1" in result
        pending = db.list_pending_contradictions()
        assert len(pending) == 1
        assert pending[0]["old_node_id"] == "k1"

    def test_list_shows_free_text_claims(self, tmp_path):
        from synapt.recall.server import recall_contradict
        db = _make_db(tmp_path)
        db.add_pending_contradiction(
            old_node_id=None,
            new_content="new policy",
            detected_by="manual",
            claim_text="policy changed last week",
        )
        index = self._make_index(db)

        with patch("synapt.recall.server._get_index", return_value=index):
            result = recall_contradict(action="list")
        assert "free-text claim" in result
        assert "policy changed last week" in result


# ---------------------------------------------------------------------------
# Server: recall_search include_historical
# ---------------------------------------------------------------------------

class TestRecallSearchHistorical:
    """Test that include_historical threads through to index.lookup()."""

    def test_include_historical_param_threaded(self, tmp_path):
        """Verify recall_search passes include_historical to index.lookup()."""
        from synapt.recall.server import recall_search

        calls = []

        class MockIndex:
            _db = None
            chunks = [TranscriptChunk(
                id="s1:t0", session_id="s1", turn_index=0,
                timestamp="2026-01-01",
                user_text="test", assistant_text="reply",
            )]
            sessions = {"s1": [0]}
            _last_diagnostics = None

            def lookup(self, query, **kwargs):
                calls.append(kwargs)
                return "result"

        mock_index = MockIndex()
        with patch("synapt.recall.server._get_index", return_value=mock_index):
            with patch("synapt.recall.live.search_live_transcript", return_value=""):
                recall_search("test query", include_historical=True)

        assert len(calls) == 1
        assert calls[0]["include_historical"] is True


# ---------------------------------------------------------------------------
# SessionStart hook: contradiction surfacing
# ---------------------------------------------------------------------------

class TestContradictionSessionStart:
    """Test format_contradictions_for_session_start."""

    def test_no_pending_returns_empty(self, tmp_path):
        from synapt.recall.server import format_contradictions_for_session_start
        # Create DB as recall.db so the function finds it
        db = RecallDB(tmp_path / "recall.db")

        with patch("synapt.recall.server.project_index_dir", return_value=tmp_path):
            result = format_contradictions_for_session_start()
        assert result == ""

    def test_pending_formatted_for_model(self, tmp_path):
        from synapt.recall.server import format_contradictions_for_session_start
        db = RecallDB(tmp_path / "recall.db")
        node = _make_knowledge_node(node_id="n1", content="old approach")
        db.save_knowledge_nodes([node])
        db.add_pending_contradiction(
            "n1", "new approach",
            reason="benchmarks improved",
            detected_by="consolidation",
        )
        db.close()

        with patch("synapt.recall.server.project_index_dir", return_value=tmp_path):
            result = format_contradictions_for_session_start()
        assert "Pending contradictions (1)" in result
        assert "ask the user to resolve" in result
        assert "old approach" in result
        assert "new approach" in result
        assert "benchmarks improved" in result
        assert "recall_contradict" in result

    def test_no_db_returns_empty(self, tmp_path):
        from synapt.recall.server import format_contradictions_for_session_start
        # Point at empty dir — no recall.db exists
        with patch("synapt.recall.server.project_index_dir", return_value=tmp_path):
            result = format_contradictions_for_session_start()
        assert result == ""


# ---------------------------------------------------------------------------
# KnowledgeNode dataclass temporal fields
# ---------------------------------------------------------------------------

class TestKnowledgeNodeDataclass:
    """Verify KnowledgeNode dataclass has temporal fields."""

    def test_new_fields_exist(self):
        from synapt.recall.knowledge import KnowledgeNode
        node = KnowledgeNode.create("test fact", "workflow")
        assert node.version == 1
        assert node.lineage_id == ""
        assert node.valid_from is None
        assert node.valid_until is None

    def test_to_dict_includes_temporal(self):
        from synapt.recall.knowledge import KnowledgeNode
        node = KnowledgeNode.create("test fact", "workflow")
        node.lineage_id = "L1"
        node.version = 3
        d = node.to_dict()
        assert d["lineage_id"] == "L1"
        assert d["version"] == 3

    def test_from_dict_with_temporal(self):
        from synapt.recall.knowledge import KnowledgeNode
        d = {
            "id": "abc", "content": "fact", "category": "workflow",
            "confidence": 0.7, "source_sessions": [], "created_at": "",
            "updated_at": "", "status": "active", "superseded_by": "",
            "contradiction_note": "", "tags": [],
            "valid_from": "2026-01-01", "valid_until": None,
            "version": 2, "lineage_id": "L1",
        }
        node = KnowledgeNode.from_dict(d)
        assert node.version == 2
        assert node.lineage_id == "L1"


# ---------------------------------------------------------------------------
# Phase 8b: Consolidation queues contradictions for user review
# ---------------------------------------------------------------------------

class TestConsolidationContradictQueuing:
    """Test that _apply_consolidation_result queues contradictions via DB."""

    def test_contradict_queued_when_db_provided(self, tmp_path):
        """With a DB, contradictions go to pending queue instead of auto-apply."""
        from synapt.recall.consolidate import _apply_consolidation_result
        from synapt.recall.knowledge import KnowledgeNode, append_node
        from synapt.recall.journal import JournalEntry

        kn_path = tmp_path / "knowledge.jsonl"
        db = _make_db(tmp_path)

        # Create existing node in JSONL and DB
        old_node = KnowledgeNode.create("use unittest", "tooling")
        append_node(old_node, kn_path)
        db.save_knowledge_nodes([old_node.to_dict()])

        parsed = {
            "nodes": [{
                "action": "contradict",
                "existing_id": old_node.id,
                "content": "use pytest instead",
                "category": "tooling",
                "contradiction_note": "pytest is standard now",
                "tags": [],
            }]
        }
        cluster = [
            JournalEntry(session_id="s1", timestamp="2026-03-01", focus="testing"),
        ]

        result = _apply_consolidation_result(
            parsed, [old_node], cluster, kn_path, db=db,
        )

        assert result.nodes_contradicted == 1
        # Node was NOT auto-applied (old node still active in JSONL)
        from synapt.recall.knowledge import read_nodes
        nodes = read_nodes(kn_path)
        assert all(n.status == "active" for n in nodes)
        # Contradiction was queued in DB
        pending = db.list_pending_contradictions()
        assert len(pending) == 1
        assert pending[0]["new_content"] == "use pytest instead"
        assert pending[0]["detected_by"] == "consolidation"

    def test_contradict_legacy_when_no_db(self, tmp_path):
        """Without a DB, contradictions auto-apply (legacy behavior)."""
        from synapt.recall.consolidate import _apply_consolidation_result
        from synapt.recall.knowledge import KnowledgeNode, append_node, read_nodes
        from synapt.recall.journal import JournalEntry

        kn_path = tmp_path / "knowledge.jsonl"
        old_node = KnowledgeNode.create("use unittest", "tooling")
        append_node(old_node, kn_path)

        parsed = {
            "nodes": [{
                "action": "contradict",
                "existing_id": old_node.id,
                "content": "use pytest instead",
                "category": "tooling",
                "contradiction_note": "pytest is standard",
                "tags": [],
            }]
        }
        cluster = [
            JournalEntry(session_id="s1", timestamp="2026-03-01", focus="testing"),
        ]

        result = _apply_consolidation_result(
            parsed, [old_node], cluster, kn_path, db=None,
        )

        assert result.nodes_contradicted == 1
        assert result.nodes_created == 1  # Legacy creates replacement inline
        nodes = read_nodes(kn_path)
        statuses = {n.status for n in nodes}
        assert "contradicted" in statuses
        assert "active" in statuses

    def test_decision_log_shows_queued_action(self, tmp_path):
        """Decision log records 'contradict-queued' when DB is used."""
        from synapt.recall.consolidate import _apply_consolidation_result
        from synapt.recall.knowledge import KnowledgeNode, append_node
        from synapt.recall.journal import JournalEntry

        kn_path = tmp_path / "knowledge.jsonl"
        decision_path = tmp_path / "decisions.jsonl"
        db = _make_db(tmp_path)

        old_node = KnowledgeNode.create("old approach", "workflow")
        append_node(old_node, kn_path)
        db.save_knowledge_nodes([old_node.to_dict()])

        parsed = {
            "nodes": [{
                "action": "contradict",
                "existing_id": old_node.id,
                "content": "new approach",
                "category": "workflow",
                "contradiction_note": "improved",
                "tags": [],
            }]
        }
        cluster = [
            JournalEntry(session_id="s1", timestamp="2026-03-01", focus="work"),
        ]

        _apply_consolidation_result(
            parsed, [old_node], cluster, kn_path,
            decision_log_path=decision_path, db=db,
        )

        decisions = []
        with open(decision_path) as f:
            for line in f:
                decisions.append(json.loads(line))
        assert len(decisions) == 1
        assert decisions[0]["action"] == "contradict-queued"

    def test_decision_log_shows_contradict_without_db(self, tmp_path):
        """Decision log records 'contradict' (not queued) when no DB."""
        from synapt.recall.consolidate import _apply_consolidation_result
        from synapt.recall.knowledge import KnowledgeNode, append_node
        from synapt.recall.journal import JournalEntry

        kn_path = tmp_path / "knowledge.jsonl"
        decision_path = tmp_path / "decisions.jsonl"

        old_node = KnowledgeNode.create("old approach", "workflow")
        append_node(old_node, kn_path)

        parsed = {
            "nodes": [{
                "action": "contradict",
                "existing_id": old_node.id,
                "content": "new approach",
                "category": "workflow",
                "contradiction_note": "improved",
                "tags": [],
            }]
        }
        cluster = [
            JournalEntry(session_id="s1", timestamp="2026-03-01", focus="work"),
        ]

        _apply_consolidation_result(
            parsed, [old_node], cluster, kn_path,
            decision_log_path=decision_path, db=None,
        )

        decisions = []
        with open(decision_path) as f:
            for line in f:
                decisions.append(json.loads(line))
        assert len(decisions) == 1
        assert decisions[0]["action"] == "contradict"

    def test_valid_from_set_on_created_nodes(self, tmp_path):
        """Newly created nodes during consolidation have valid_from set."""
        from synapt.recall.consolidate import _apply_consolidation_result
        from synapt.recall.knowledge import KnowledgeNode, read_nodes
        from synapt.recall.journal import JournalEntry

        kn_path = tmp_path / "knowledge.jsonl"

        parsed = {
            "nodes": [{
                "action": "create",
                "content": "always use A100 for 8B training",
                "category": "infrastructure",
                "confidence": 0.7,
                "tags": ["gpu"],
            }]
        }
        cluster = [
            JournalEntry(session_id="s1", timestamp="2026-03-01", focus="training"),
            JournalEntry(session_id="s2", timestamp="2026-03-02", focus="training"),
        ]

        _apply_consolidation_result(parsed, [], cluster, kn_path)

        nodes = read_nodes(kn_path)
        assert len(nodes) == 1
        assert nodes[0].valid_from is not None
        assert nodes[0].valid_from.startswith("2026-")

    def test_valid_from_set_on_legacy_contradict_replacement(self, tmp_path):
        """Legacy contradict path also sets valid_from on the replacement."""
        from synapt.recall.consolidate import _apply_consolidation_result
        from synapt.recall.knowledge import KnowledgeNode, append_node, read_nodes
        from synapt.recall.journal import JournalEntry

        kn_path = tmp_path / "knowledge.jsonl"
        old_node = KnowledgeNode.create("old fact", "workflow")
        append_node(old_node, kn_path)

        parsed = {
            "nodes": [{
                "action": "contradict",
                "existing_id": old_node.id,
                "content": "new fact",
                "category": "workflow",
                "contradiction_note": "updated",
                "tags": [],
            }]
        }
        cluster = [
            JournalEntry(session_id="s1", timestamp="2026-03-01", focus="work"),
        ]

        _apply_consolidation_result(
            parsed, [old_node], cluster, kn_path, db=None,
        )

        nodes = read_nodes(kn_path, status="active")
        assert len(nodes) == 1
        assert nodes[0].valid_from is not None

    def test_contradict_missing_target_falls_through_to_create(self, tmp_path):
        """When contradict references a nonexistent ID, the node is created instead."""
        from synapt.recall.consolidate import _apply_consolidation_result
        from synapt.recall.knowledge import read_nodes
        from synapt.recall.journal import JournalEntry

        kn_path = tmp_path / "knowledge.jsonl"

        parsed = {
            "nodes": [{
                "action": "contradict",
                "existing_id": "nonexistent-id",
                "content": "new approach to training",
                "category": "workflow",
                "contradiction_note": "old approach obsolete",
                "tags": ["training"],
            }]
        }
        cluster = [
            JournalEntry(session_id="s1", timestamp="2026-03-01", focus="training"),
            JournalEntry(session_id="s2", timestamp="2026-03-02", focus="training"),
        ]

        result = _apply_consolidation_result(
            parsed, [], cluster, kn_path, db=None,
        )

        # Should create the node (not silently drop it)
        assert result.nodes_created == 1
        assert result.nodes_contradicted == 0  # No target → no contradiction
        nodes = read_nodes(kn_path)
        assert len(nodes) == 1
        assert nodes[0].content == "new approach to training"
        assert nodes[0].valid_from is not None

    def test_queued_contradiction_carries_candidate_bounds(self, tmp_path):
        """BLOCKER 2 fix: the DB-queued contradict branch must pass the candidate's temporal
        bounds into add_pending_contradiction -- before this fix, they were dropped entirely
        (add_pending_contradiction was called with no temporal args at all)."""
        from synapt.recall.consolidate import _apply_consolidation_result
        from synapt.recall.knowledge import KnowledgeNode, append_node
        from synapt.recall.journal import JournalEntry

        kn_path = tmp_path / "knowledge.jsonl"
        db = _make_db(tmp_path)

        old_node = KnowledgeNode.create("API key policy unclear", "tooling")
        append_node(old_node, kn_path)
        db.save_knowledge_nodes([old_node.to_dict()])

        parsed = {
            "nodes": [{
                "action": "contradict",
                "existing_id": old_node.id,
                "content": "the API key expires 2025-04-30",
                "category": "tooling",
                "contradiction_note": "expiry clarified",
                "tags": [],
                "valid_from": None,
                "valid_until": "2025-04-30",
            }]
        }
        cluster = [JournalEntry(session_id="s1", timestamp="2026-03-01", focus="keys")]

        _apply_consolidation_result(parsed, [old_node], cluster, kn_path, db=db)

        pending = db.list_pending_contradictions()
        assert len(pending) == 1
        assert pending[0]["valid_from"] is None
        assert pending[0]["valid_until"] == "2025-04-30"  # candidate's bound, in the payload

    def test_confirm_carries_queued_bound_onto_materialized_node(self, tmp_path):
        """BLOCKER 2 fix, the FULL round trip: queue (with a bound) -> confirm -> materialize,
        through the REAL recall_contradict MCP tool (not a hand-duplicated resolve). Before this
        fix, _apply_supersession hardcoded valid_from=now/valid_until=None on confirm, so even a
        bound that survived the queue was lost the moment the contradiction was materialized --
        "fruit-to-the-dict is not fruit-to-the-database" one layer further than the queue row."""
        from synapt.recall.consolidate import _apply_consolidation_result
        from synapt.recall.knowledge import KnowledgeNode, append_node
        from synapt.recall.journal import JournalEntry
        from synapt.recall.server import recall_contradict

        kn_path = tmp_path / "knowledge.jsonl"
        db = _make_db(tmp_path)

        old_node = KnowledgeNode.create("API key policy unclear", "tooling")
        append_node(old_node, kn_path)
        db.save_knowledge_nodes([old_node.to_dict()])

        parsed = {
            "nodes": [{
                "action": "contradict",
                "existing_id": old_node.id,
                "content": "the API key expires 2025-04-30",
                "category": "tooling",
                "contradiction_note": "expiry clarified",
                "tags": [],
                "valid_from": None,
                "valid_until": "2025-04-30",
            }]
        }
        cluster = [JournalEntry(session_id="s1", timestamp="2026-03-01", focus="keys")]
        _apply_consolidation_result(parsed, [old_node], cluster, kn_path, db=db)
        cid = db.list_pending_contradictions()[0]["id"]

        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}
        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                result = recall_contradict(
                    action="resolve", contradiction_id=cid, resolution="confirmed",
                )
        assert "confirmed" in result

        nodes = db.load_knowledge_nodes()
        active = [n for n in nodes if n["status"] == "active"]
        assert len(active) == 1
        assert active[0]["content"] == "the API key expires 2025-04-30"
        assert active[0]["valid_until"] == "2025-04-30"  # candidate's bound survived to the DB

    def test_apply_supersession_rejects_malformed_bounds_defensively(self, tmp_path):
        """Bug found by adversarial verification workflow (2026-07-15): _apply_supersession is
        the ONE bound-consuming site in this feature with no _validate_iso_date guard of its own
        — every other site (consolidate.py's corroborate/contradict branches) validates before
        ever reaching a dict-update or DB-write. Feeding a list/dict straight through (bypassing
        the DB round trip, which today always validates upstream — this is a DEFENSE-IN-DEPTH
        gate, not a currently-reachable exploit) previously raised sqlite3.ProgrammingError AND
        left a non-atomic PARTIAL WRITE: the old node marked contradicted with superseded_by
        pointing at a replacement that was NEVER created (the second upsert crashed), with no
        retry path since resolve_contradiction's status flip to 'confirmed' already committed.
        Must now behave exactly like every other bound-consuming site: malformed input -> None,
        never a crash, never a corrupted half-applied supersession."""
        from synapt.recall.server import _apply_supersession

        for i, bad in enumerate([["2025-04-30"], {"a": 1}, 20250430, 20250430.0, "not-a-date", "   "]):
            iter_dir = tmp_path / f"iter{i}"
            iter_dir.mkdir()
            db = _make_db(iter_dir)  # a FRESH, isolated DB per iteration — no cross-iteration bleed
            old_node = _make_knowledge_node(node_id="old-1", content="old fact")
            db.save_knowledge_nodes([old_node])

            _apply_supersession(
                db, old_node_id=old_node["id"], new_content="new fact",
                category="tooling", reason="test", source_sessions=["s1"],
                valid_from=bad, valid_until=bad,
            )  # must NOT raise

            nodes = db.load_knowledge_nodes()
            active = [n for n in nodes if n["status"] == "active"]
            assert len(active) == 1  # the replacement WAS created — no partial write
            assert active[0]["valid_until"] is None  # malformed input never persists verbatim

    def test_apply_supersession_still_carries_a_valid_bound(self, tmp_path):
        # regression guard: the defensive validation must not break the real, valid-bound path.
        from synapt.recall.server import _apply_supersession

        db = _make_db(tmp_path)
        old_node = _make_knowledge_node(node_id="old-1", content="old fact")
        db.save_knowledge_nodes([old_node])

        _apply_supersession(
            db, old_node_id="old-1", new_content="new fact", category="tooling",
            reason="test", source_sessions=["s1"],
            valid_from=None, valid_until="2025-04-30",
        )

        active = [n for n in db.load_knowledge_nodes() if n["status"] == "active"]
        assert active[0]["valid_until"] == "2025-04-30"

    def test_confirm_falls_back_to_now_when_queued_bound_is_none(self, tmp_path):
        # regression guard: the EXISTING fallback (valid_from=now when nothing was queued) must
        # survive this fix unchanged, for contradictions that carry no temporal information.
        from synapt.recall.knowledge import KnowledgeNode
        from synapt.recall.server import recall_contradict

        db = _make_db(tmp_path)
        node = _make_knowledge_node(node_id="old-1", content="use unittest")
        db.save_knowledge_nodes([node])
        cid = db.add_pending_contradiction("old-1", "use pytest instead")

        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}
        with patch("synapt.recall.server._get_index", return_value=index):
            with patch("synapt.recall.server._invalidate_cache"):
                recall_contradict(action="resolve", contradiction_id=cid, resolution="confirmed")

        active = [n for n in db.load_knowledge_nodes() if n["status"] == "active"]
        assert active[0]["valid_from"] is not None  # fallback still fires
        assert active[0]["valid_until"] is None

    def test_queued_contradiction_source_sessions(self, tmp_path):
        """Queued contradictions include the cluster's session IDs."""
        from synapt.recall.consolidate import _apply_consolidation_result
        from synapt.recall.knowledge import KnowledgeNode, append_node
        from synapt.recall.journal import JournalEntry

        kn_path = tmp_path / "knowledge.jsonl"
        db = _make_db(tmp_path)

        old_node = KnowledgeNode.create("old", "workflow")
        append_node(old_node, kn_path)
        db.save_knowledge_nodes([old_node.to_dict()])

        parsed = {
            "nodes": [{
                "action": "contradict",
                "existing_id": old_node.id,
                "content": "new",
                "category": "workflow",
                "contradiction_note": "changed",
                "tags": [],
            }]
        }
        cluster = [
            JournalEntry(session_id="sess-A", timestamp="2026-03-01", focus="a"),
            JournalEntry(session_id="sess-B", timestamp="2026-03-02", focus="b"),
        ]

        _apply_consolidation_result(
            parsed, [old_node], cluster, kn_path, db=db,
        )

        pending = db.list_pending_contradictions()
        assert len(pending) == 1
        assert "sess-A" in pending[0]["source_sessions"]
        assert "sess-B" in pending[0]["source_sessions"]


# ---------------------------------------------------------------------------
# Phase 8c: Co-retrieval conflict detection
# ---------------------------------------------------------------------------

class TestCoRetrievalConflictDetection:
    """Test _detect_co_retrieval_conflicts in core.py."""

    def _make_index_with_db(self, tmp_path):
        """Create a minimal TranscriptIndex with a DB attached."""
        db = _make_db(tmp_path)
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = db
        index.chunks = []
        index.sessions = {}
        return index, db

    def test_detects_conflict_same_category_low_overlap(self, tmp_path):
        """Two active nodes, same category, divergent content → queued."""
        index, db = self._make_index_with_db(tmp_path)
        results = [
            _make_knowledge_node(
                node_id="n1", content="always use unittest for Python testing",
                category="tooling", confidence=0.7,
            ),
            _make_knowledge_node(
                node_id="n2", content="prefer pytest with fixtures and markers",
                category="tooling", confidence=0.8,
            ),
        ]

        index._detect_co_retrieval_conflicts(results)

        pending = db.list_pending_contradictions()
        assert len(pending) == 1
        assert pending[0]["detected_by"] == "co-retrieval"
        # Lower confidence node (n1) is the "old" one
        assert pending[0]["old_node_id"] == "n1"

    def test_no_conflict_different_categories(self, tmp_path):
        """Two active nodes in different categories → no conflict."""
        index, db = self._make_index_with_db(tmp_path)
        results = [
            _make_knowledge_node(
                node_id="n1", content="use A100 for training",
                category="infrastructure", confidence=0.7,
            ),
            _make_knowledge_node(
                node_id="n2", content="always review PRs before merge",
                category="workflow", confidence=0.8,
            ),
        ]

        index._detect_co_retrieval_conflicts(results)

        assert db.pending_contradiction_count() == 0

    def test_no_conflict_high_overlap(self, tmp_path):
        """Two similar nodes (Jaccard >= 0.3) → no conflict."""
        index, db = self._make_index_with_db(tmp_path)
        results = [
            _make_knowledge_node(
                node_id="n1", content="use pytest for unit testing",
                category="tooling", confidence=0.7,
            ),
            _make_knowledge_node(
                node_id="n2", content="use pytest for integration testing",
                category="tooling", confidence=0.8,
            ),
        ]

        index._detect_co_retrieval_conflicts(results)

        assert db.pending_contradiction_count() == 0

    def test_skips_low_confidence_nodes(self, tmp_path):
        """Nodes with confidence < 0.4 are ignored."""
        index, db = self._make_index_with_db(tmp_path)
        results = [
            _make_knowledge_node(
                node_id="n1", content="use unittest exclusively",
                category="tooling", confidence=0.2,
            ),
            _make_knowledge_node(
                node_id="n2", content="prefer pytest with markers",
                category="tooling", confidence=0.8,
            ),
        ]

        index._detect_co_retrieval_conflicts(results)

        assert db.pending_contradiction_count() == 0

    def test_skips_same_lineage(self, tmp_path):
        """Nodes sharing a lineage_id are versions, not conflicts."""
        index, db = self._make_index_with_db(tmp_path)
        results = [
            _make_knowledge_node(
                node_id="v1", content="use unittest for testing",
                category="tooling", confidence=0.5, lineage_id="L1",
            ),
            _make_knowledge_node(
                node_id="v2", content="prefer pytest with fixtures",
                category="tooling", confidence=0.8, lineage_id="L1",
            ),
        ]

        index._detect_co_retrieval_conflicts(results)

        assert db.pending_contradiction_count() == 0

    def test_skips_contradicted_nodes(self, tmp_path):
        """Non-active nodes are excluded from conflict detection."""
        index, db = self._make_index_with_db(tmp_path)
        results = [
            _make_knowledge_node(
                node_id="n1", content="use unittest exclusively",
                category="tooling", confidence=0.7, status="contradicted",
            ),
            _make_knowledge_node(
                node_id="n2", content="prefer pytest with markers",
                category="tooling", confidence=0.8,
            ),
        ]

        index._detect_co_retrieval_conflicts(results)

        assert db.pending_contradiction_count() == 0

    def test_dedup_skips_already_pending(self, tmp_path):
        """Does not queue a duplicate if node already has a pending contradiction."""
        index, db = self._make_index_with_db(tmp_path)
        # Pre-seed a pending contradiction for n1
        db.add_pending_contradiction("n1", "some other replacement")

        results = [
            _make_knowledge_node(
                node_id="n1", content="use unittest exclusively",
                category="tooling", confidence=0.7,
            ),
            _make_knowledge_node(
                node_id="n2", content="prefer pytest with fixtures",
                category="tooling", confidence=0.8,
            ),
        ]

        index._detect_co_retrieval_conflicts(results)

        # Still just the original one — no duplicate
        assert db.pending_contradiction_count() == 1

    def test_no_db_is_noop(self, tmp_path):
        """No DB attached → no crash, no action."""
        index = TranscriptIndex.__new__(TranscriptIndex)
        index._db = None
        index.chunks = []
        index.sessions = {}

        results = [
            _make_knowledge_node(node_id="n1", content="fact A", category="tooling"),
            _make_knowledge_node(node_id="n2", content="fact B", category="tooling"),
        ]

        # Should not raise
        index._detect_co_retrieval_conflicts(results)

    def test_single_result_is_noop(self, tmp_path):
        """A single result can't conflict with itself."""
        index, db = self._make_index_with_db(tmp_path)
        results = [
            _make_knowledge_node(node_id="n1", content="fact", category="tooling"),
        ]

        index._detect_co_retrieval_conflicts(results)

        assert db.pending_contradiction_count() == 0

    def test_returns_detected_pairs(self, tmp_path):
        """_detect_co_retrieval_conflicts returns (old, new) tuples."""
        index, db = self._make_index_with_db(tmp_path)
        results = [
            {"id": "n1", "content": "deploy on tuesday", "category": "workflow",
             "status": "active", "confidence": 0.6, "lineage_id": ""},
            {"id": "n2", "content": "never use feature flags", "category": "workflow",
             "status": "active", "confidence": 0.9, "lineage_id": ""},
        ]
        detected = index._detect_co_retrieval_conflicts(results)
        assert len(detected) == 1
        old, new = detected[0]
        assert old["id"] == "n1"  # Lower confidence
        assert new["id"] == "n2"

    def test_last_conflicts_cleared_between_searches(self, tmp_path):
        """_last_conflicts is reset at the start of each lookup call."""
        index, db = self._make_index_with_db(tmp_path)
        # Manually set stale conflicts
        index._last_conflicts = [({}, {})]
        # lookup() should clear it even if search returns nothing
        index.lookup("nonexistent query that matches nothing")
        assert index._last_conflicts == []

    def test_format_results_surfaces_conflict_notice(self, tmp_path):
        """Visible search output should surface detected knowledge conflicts."""
        db = _make_db(tmp_path)
        index = TranscriptIndex([], db=db)
        old = _make_knowledge_node(
            node_id="n1",
            content="always use unittest for Python testing",
            category="tooling",
            confidence=0.7,
        )
        new = _make_knowledge_node(
            node_id="n2",
            content="prefer pytest with fixtures and markers",
            category="tooling",
            confidence=0.8,
        )
        index._last_conflicts = [(old, new)]

        result = index._format_results(
            [],
            max_tokens=2000,
            knowledge_results=[old, new],
        )

        assert "Potential contradiction detected in retrieved knowledge" in result
        assert "recall_contradict(action=\"list\")" in result

    def test_has_pending_contradiction_for(self, tmp_path):
        """Test the storage dedup helper."""
        db = _make_db(tmp_path)
        assert not db.has_pending_contradiction_for("n1")
        db.add_pending_contradiction("n1", "new fact")
        assert db.has_pending_contradiction_for("n1")
        assert not db.has_pending_contradiction_for("n2")


class _FakeEmbeddingProvider:
    def embed_single(self, text: str) -> list[float]:
        return [0.01] * 384


class TestRecallSave:
    """Test the recall_save MCP tool."""

    def test_recall_save_creates_and_embeds_node(self, tmp_path):
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=_FakeEmbeddingProvider()), \
             patch("synapt.recall.server._invalidate_cache"):
            result = recall_save(
                content="Deploy previews expire after 7 days",
                category="workflow",
                confidence=0.9,
                tags=["preview", "deploy"],
                source_sessions=["sess-1"],
                source_turns=["sess-1:12"],
            )

        assert "Knowledge node saved:" in result
        assert "embedded for vector search" in result

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            nodes = db.load_knowledge_nodes(status="active")
            assert len(nodes) == 1
            assert nodes[0]["content"] == "Deploy previews expire after 7 days"
            assert nodes[0]["tags"] == ["preview", "deploy"]
            assert nodes[0]["source_sessions"] == ["sess-1"]
            assert nodes[0]["source_turns"] == ["sess-1:12"]
            emb_map = db.get_knowledge_embeddings_by_id()
            assert nodes[0]["id"] in emb_map
        finally:
            db.close()

    def test_recall_save_without_embeddings_still_saves_node(self, tmp_path):
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            result = recall_save(content="Use staging before production")

        assert "saved without embeddings" in result

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            nodes = db.load_knowledge_nodes(status="active")
            assert len(nodes) == 1
            assert nodes[0]["content"] == "Use staging before production"
            assert db.get_knowledge_embeddings_by_id() == {}
        finally:
            db.close()

    def test_recall_save_defaults_to_content_hash_upsert(self, tmp_path):
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            first = recall_save(content="Use staging before production")
            second = recall_save(content="Use staging before production")

        assert "Knowledge node saved:" in first
        assert "Knowledge node updated:" in second  # second save is an update

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            nodes = db.load_knowledge_nodes(status="active")
            assert len(nodes) == 1
            assert nodes[0]["content"] == "Use staging before production"
        finally:
            db.close()

    def test_recall_save_requires_content(self):
        from synapt.recall.server import recall_save

        assert "required" in recall_save(content="   ").lower()

    def test_recall_save_upserts_stable_node_id(self, tmp_path):
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            first = recall_save(
                content="Original memory",
                category="workflow",
                node_id="stable-node1",
            )
            second = recall_save(
                content="Updated memory",
                category="workflow",
                node_id="stable-node1",
            )

        assert "Knowledge node saved:" in first
        assert "Knowledge node updated:" in second

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            nodes = db.load_knowledge_nodes(status="active")
            assert len(nodes) == 1
            assert nodes[0]["id"] == "stable-node1"
            assert nodes[0]["content"] == "Updated memory"
        finally:
            db.close()


    def test_recall_save_update_bumps_version(self, tmp_path):
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            first = recall_save(
                content="LOCOMO score is 76%",
                category="benchmark",
                node_id="bench1",
            )
            second = recall_save(
                content="LOCOMO score is 72.4% (audited)",
                category="benchmark",
                node_id="bench1",
            )

        assert "saved" in first
        assert "updated" in second
        assert "v2" in second

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            node = db.get_knowledge_node("bench1")
            assert node["content"] == "LOCOMO score is 72.4% (audited)"
            assert node["version"] == 2
            assert node["lineage_id"] == "bench1"
        finally:
            db.close()

    def test_recall_save_retract(self, tmp_path):
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="Wrong fact", category="workflow", node_id="wrong1")
            result = recall_save(node_id="wrong1", retract=True)

        assert "retracted" in result.lower()

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            node = db.get_knowledge_node("wrong1")
            assert node["status"] == "retracted"
            assert node["valid_until"] is not None
            # Retracted nodes excluded from active search
            active = db.load_knowledge_nodes(status="active")
            assert all(n["id"] != "wrong1" for n in active)
        finally:
            db.close()

    def test_recall_save_retract_requires_node_id(self):
        from synapt.recall.server import recall_save

        result = recall_save(retract=True)
        assert "node_id is required" in result.lower()

    def test_recall_save_retract_nonexistent_node(self, tmp_path):
        from synapt.recall.server import recall_save

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            result = recall_save(node_id="nonexistent", retract=True)

        assert "not found" in result.lower()


class TestRecallSyncMemory:
    """Tests for MEMORY.md sync into shared recall knowledge."""

    def test_recall_sync_memory_is_idempotent_per_file(self, tmp_path):
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_sync_memory
        from synapt.recall.storage import RecallDB

        home = tmp_path / "home"
        project = tmp_path / "gripspace"
        memory_dir = home / ".claude" / "projects" / "proj-a" / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "policy.md").write_text(
            "---\n"
            "name: Ministral policy\n"
            "description: Use the local model first\n"
            "type: project\n"
            "---\n"
            "Prefer Ministral for quick local enrichment.\n",
            encoding="utf-8",
        )

        with patch("pathlib.Path.home", return_value=home), \
             patch("synapt.recall.server.Path.cwd", return_value=project), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            first = recall_sync_memory()
            second = recall_sync_memory()

        assert "1 synced" in first
        assert "1 synced" in second

        db = RecallDB(project_index_dir(project) / "recall.db")
        try:
            nodes = db.load_knowledge_nodes(status="active")
            assert len(nodes) == 1
            assert nodes[0]["content"].startswith("Ministral policy: Use the local model first")
            assert "memory.md" in nodes[0]["tags"]
        finally:
            db.close()
