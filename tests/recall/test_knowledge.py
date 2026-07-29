"""Tests for knowledge node CRUD, compaction, confidence, and FTS search."""

import json
import tempfile
import unittest
from pathlib import Path

from synapt.recall.knowledge import (
    KnowledgeNode,
    append_node,
    compact_knowledge,
    compute_confidence,
    format_knowledge_for_display,
    format_knowledge_for_session_start,
    read_nodes,
    update_node,
)


class TestKnowledgeNode(unittest.TestCase):
    def test_create_factory(self):
        node = KnowledgeNode.create(
            content="Always use A100 for training",
            category="infrastructure",
            source_sessions=["sess-A", "sess-B"],
            confidence=0.6,
            tags=["gpu", "training"],
        )
        self.assertEqual(len(node.id), 12)
        self.assertEqual(node.content, "Always use A100 for training")
        self.assertEqual(node.category, "infrastructure")
        self.assertEqual(node.confidence, 0.6)
        self.assertEqual(node.source_sessions, ["sess-A", "sess-B"])
        self.assertEqual(node.status, "active")
        self.assertTrue(node.created_at)
        self.assertTrue(node.updated_at)

    def test_create_clamps_confidence(self):
        node = KnowledgeNode.create(content="test", category="workflow", confidence=1.5)
        self.assertEqual(node.confidence, 1.0)

    def test_create_normalizes_invalid_category(self):
        node = KnowledgeNode.create(content="test", category="invalid-cat")
        self.assertEqual(node.category, "workflow")  # Falls back to default

    def test_create_accepts_preference_and_fact_categories(self):
        pref = KnowledgeNode.create(content="prefers dark roast", category="preference")
        self.assertEqual(pref.category, "preference")
        fact = KnowledgeNode.create(content="sister is Elena", category="fact")
        self.assertEqual(fact.category, "fact")

    def test_roundtrip_dict(self):
        node = KnowledgeNode.create(
            content="Test fact",
            category="debugging",
            tags=["test"],
        )
        d = node.to_dict()
        restored = KnowledgeNode.from_dict(d)
        self.assertEqual(restored.id, node.id)
        self.assertEqual(restored.content, node.content)
        self.assertEqual(restored.tags, node.tags)

    # --- source_unit_id (internal design spec, section 3): a small, additive,
    # backward-compatible field carrying the node's TRUE source
    # chronology -- distinct from created_at/updated_at, which are consolidation-run-time, not
    # event-time. Exists so Fix B's chronology escalation has a real value to compare on the
    # already-persisted side, not a proxy. ---

    def test_create_defaults_source_unit_id_to_none(self):
        """The overwhelming majority of callers (legacy path, collection pass) never pass
        this -- must default cleanly, never require it."""
        node = KnowledgeNode.create(content="test", category="workflow")
        self.assertIsNone(node.source_unit_id)

    def test_create_accepts_explicit_source_unit_id(self):
        node = KnowledgeNode.create(
            content="test", category="workflow",
            source_unit_id="986d09c3e8bb2ae5:2:done:6",
        )
        self.assertEqual(node.source_unit_id, "986d09c3e8bb2ae5:2:done:6")

    def test_roundtrip_dict_preserves_source_unit_id(self):
        node = KnowledgeNode.create(
            content="test", category="workflow",
            source_unit_id="986d09c3e8bb2ae5:2:done:6",
        )
        restored = KnowledgeNode.from_dict(node.to_dict())
        self.assertEqual(restored.source_unit_id, "986d09c3e8bb2ae5:2:done:6")

    def test_from_dict_on_a_pre_fix_row_without_the_key_defaults_to_none(self):
        """Backward compatibility, verified not assumed: a real pre-fix persisted row (a
        plain dict with no source_unit_id key at all, exactly what every JSONL row written
        before this change looks like) must load cleanly with source_unit_id=None, not
        KeyError or a silent wrong default."""
        pre_fix_row = {
            "id": "abc123def456",
            "content": "a fact persisted before this field existed",
            "category": "workflow",
            "confidence": 0.5,
        }
        restored = KnowledgeNode.from_dict(pre_fix_row)
        self.assertIsNone(restored.source_unit_id)


class TestKnowledgeCRUD(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "knowledge.jsonl"

    def test_node_create_and_read(self):
        node = KnowledgeNode.create(content="Fact 1", category="workflow")
        append_node(node, self.path)
        nodes = read_nodes(self.path)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].content, "Fact 1")

    def test_read_nodes_filtered_by_status(self):
        active = KnowledgeNode.create(content="Active", category="workflow")
        stale = KnowledgeNode.create(content="Stale", category="workflow")
        stale.status = "stale"
        append_node(active, self.path)
        append_node(stale, self.path)
        active_only = read_nodes(self.path, status="active")
        self.assertEqual(len(active_only), 1)
        self.assertEqual(active_only[0].content, "Active")

    def test_read_nodes_sorted_by_confidence(self):
        low = KnowledgeNode.create(content="Low", category="workflow", confidence=0.3)
        high = KnowledgeNode.create(content="High", category="workflow", confidence=0.9)
        append_node(low, self.path)
        append_node(high, self.path)
        nodes = read_nodes(self.path)
        self.assertEqual(nodes[0].content, "High")
        self.assertEqual(nodes[1].content, "Low")

    def test_node_update(self):
        node = KnowledgeNode.create(content="Original", category="workflow", confidence=0.5)
        append_node(node, self.path)
        updated = update_node(node.id, {"confidence": 0.8, "status": "stale"}, self.path)
        self.assertTrue(updated)
        nodes = read_nodes(self.path)
        self.assertEqual(len(nodes), 1)  # Deduped
        self.assertEqual(nodes[0].confidence, 0.8)
        self.assertEqual(nodes[0].status, "stale")

    def test_update_nonexistent_node(self):
        node = KnowledgeNode.create(content="X", category="workflow")
        append_node(node, self.path)
        result = update_node("nonexistent", {"confidence": 1.0}, self.path)
        self.assertFalse(result)

    def test_update_missing_file(self):
        missing = Path(self.tmpdir) / "nope.jsonl"
        result = update_node("abc", {"confidence": 1.0}, missing)
        self.assertFalse(result)

    def test_read_empty_file(self):
        nodes = read_nodes(Path(self.tmpdir) / "no-such-file.jsonl")
        self.assertEqual(nodes, [])


class TestCompactKnowledge(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "knowledge.jsonl"

    def test_compact_removes_duplicates(self):
        node = KnowledgeNode.create(content="Fact", category="workflow")
        append_node(node, self.path)
        # Simulate update (appends new version)
        update_node(node.id, {"confidence": 0.9}, self.path)
        # File now has 2 lines
        with open(self.path) as f:
            self.assertEqual(len(f.readlines()), 2)
        removed = compact_knowledge(self.path)
        self.assertEqual(removed, 1)
        with open(self.path) as f:
            self.assertEqual(len(f.readlines()), 1)

    def test_compact_noop_when_clean(self):
        node = KnowledgeNode.create(content="Fact", category="workflow")
        append_node(node, self.path)
        mtime_before = self.path.stat().st_mtime
        removed = compact_knowledge(self.path)
        self.assertEqual(removed, 0)
        self.assertEqual(self.path.stat().st_mtime, mtime_before)

    def test_compact_nonexistent_file(self):
        result = compact_knowledge(Path(self.tmpdir) / "nope.jsonl")
        self.assertEqual(result, 0)

    def test_compact_keeps_latest_version(self):
        node = KnowledgeNode.create(content="v1", category="workflow")
        node.updated_at = "2026-01-01T00:00:00"
        append_node(node, self.path)
        node2 = KnowledgeNode.from_dict(node.to_dict())
        node2.content = "v2"
        node2.updated_at = "2026-03-01T00:00:00"
        append_node(node2, self.path)
        compact_knowledge(self.path)
        nodes = read_nodes(self.path)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].content, "v2")


class TestConfidenceFormula(unittest.TestCase):
    def test_single_session(self):
        c = compute_confidence(1)
        self.assertAlmostEqual(c, 0.45, places=2)

    def test_two_sessions(self):
        c = compute_confidence(2)
        self.assertAlmostEqual(c, 0.60, places=2)

    def test_three_sessions(self):
        c = compute_confidence(3)
        self.assertAlmostEqual(c, 0.75, places=2)

    def test_five_plus_sessions_cap(self):
        c = compute_confidence(5)
        self.assertAlmostEqual(c, 0.9, places=2)
        c10 = compute_confidence(10)
        self.assertAlmostEqual(c10, 0.9, places=2)

    def test_monotonically_increases_with_sources(self):
        prev = 0
        for n in range(1, 10):
            c = compute_confidence(n)
            self.assertGreaterEqual(c, prev)
            prev = c
        # Specifically: 1 < 2 < 3 < 4 (before cap)
        self.assertLess(compute_confidence(1), compute_confidence(2))
        self.assertLess(compute_confidence(2), compute_confidence(3))
        self.assertLess(compute_confidence(3), compute_confidence(4))

    def test_decays_with_age(self):
        fresh = compute_confidence(3, age_days=0)
        old = compute_confidence(3, age_days=90)
        self.assertAlmostEqual(old, fresh * 0.5, places=2)

    def test_no_decay_at_zero_age(self):
        self.assertEqual(compute_confidence(2, 0.0), compute_confidence(2))


class TestFormatting(unittest.TestCase):
    def test_format_for_session_start_empty(self):
        result = format_knowledge_for_session_start([])
        self.assertEqual(result, "")

    def test_format_for_session_start_max_nodes(self):
        nodes = [
            KnowledgeNode.create(content=f"Fact {i}", category="workflow", confidence=0.9 - i * 0.1)
            for i in range(5)
        ]
        result = format_knowledge_for_session_start(nodes, max_nodes=3)
        self.assertIn("Knowledge:", result)
        self.assertEqual(result.count("- [workflow]"), 3)

    def test_format_for_session_start_filters_inactive(self):
        active = KnowledgeNode.create(content="Active", category="workflow", confidence=0.9)
        stale = KnowledgeNode.create(content="Stale", category="workflow", confidence=0.9)
        stale.status = "stale"
        result = format_knowledge_for_session_start([active, stale])
        self.assertIn("Active", result)
        self.assertNotIn("Stale", result)

    def test_format_for_display(self):
        node = KnowledgeNode.create(content="Test", category="debugging", confidence=0.8)
        result = format_knowledge_for_display([node])
        self.assertIn("[debugging]", result)
        self.assertIn("80%", result)

    def test_format_for_display_empty(self):
        result = format_knowledge_for_display([])
        self.assertIn("No knowledge nodes found", result)


class TestStorageKnowledge(unittest.TestCase):
    """Test SQLite storage for knowledge nodes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "recall.db"

    def test_knowledge_table_created(self):
        from synapt.recall.storage import RecallDB
        db = RecallDB(self.db_path)
        # Check table exists
        row = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge'"
        ).fetchone()
        self.assertIsNotNone(row)
        db.close()

    def test_knowledge_fts_table_created(self):
        from synapt.recall.storage import RecallDB
        db = RecallDB(self.db_path)
        row = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_fts'"
        ).fetchone()
        self.assertIsNotNone(row)
        db.close()

    def test_knowledge_upsert_and_load(self):
        from synapt.recall.storage import RecallDB
        db = RecallDB(self.db_path)
        node = {
            "id": "test123",
            "content": "Always use A100 for training",
            "category": "infrastructure",
            "confidence": 0.7,
            "source_sessions": ["sess-A"],
            "created_at": "2026-03-01T00:00:00",
            "updated_at": "2026-03-01T00:00:00",
            "status": "active",
            "superseded_by": "",
            "tags": ["gpu"],
        }
        db.upsert_knowledge_node(node)
        loaded = db.load_knowledge_nodes()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["content"], "Always use A100 for training")
        self.assertEqual(loaded[0]["tags"], ["gpu"])
        db.close()

    def test_knowledge_load_filtered_by_status(self):
        from synapt.recall.storage import RecallDB
        db = RecallDB(self.db_path)
        db.upsert_knowledge_node({
            "id": "a1", "content": "Active", "category": "workflow",
            "confidence": 0.5, "source_sessions": [], "created_at": "",
            "updated_at": "", "status": "active", "superseded_by": "", "tags": [],
        })
        db.upsert_knowledge_node({
            "id": "s1", "content": "Stale", "category": "workflow",
            "confidence": 0.5, "source_sessions": [], "created_at": "",
            "updated_at": "", "status": "stale", "superseded_by": "", "tags": [],
        })
        active = db.load_knowledge_nodes(status="active")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["content"], "Active")
        db.close()

    def test_knowledge_fts_search(self):
        from synapt.recall.storage import RecallDB
        db = RecallDB(self.db_path)
        db.upsert_knowledge_node({
            "id": "k1", "content": "Always use ollama for code generation",
            "category": "workflow", "confidence": 0.8,
            "source_sessions": ["s1", "s2"], "created_at": "2026-03-01T00:00:00",
            "updated_at": "2026-03-01T00:00:00", "status": "active",
            "superseded_by": "", "tags": ["ollama", "codegen"],
        })
        db.upsert_knowledge_node({
            "id": "k2", "content": "A10G for eval, A100 for training",
            "category": "infrastructure", "confidence": 0.7,
            "source_sessions": ["s1"], "created_at": "2026-03-01T00:00:00",
            "updated_at": "2026-03-01T00:00:00", "status": "active",
            "superseded_by": "", "tags": ["gpu"],
        })
        results = db.knowledge_fts_search("ollama code generation")
        self.assertGreater(len(results), 0)
        # First result should be the ollama node
        rowid = results[0][0]
        nodes = db.knowledge_by_rowid([rowid])
        self.assertIn("ollama", nodes[rowid]["content"])
        db.close()

    def test_knowledge_count(self):
        from synapt.recall.storage import RecallDB
        db = RecallDB(self.db_path)
        self.assertEqual(db.knowledge_count(), 0)
        db.upsert_knowledge_node({
            "id": "k1", "content": "Test", "category": "workflow",
            "confidence": 0.5, "source_sessions": [], "created_at": "",
            "updated_at": "", "status": "active", "superseded_by": "", "tags": [],
        })
        self.assertEqual(db.knowledge_count(), 1)
        db.close()

    def test_get_knowledge_rowid(self):
        from synapt.recall.storage import RecallDB
        db = RecallDB(self.db_path)
        db.upsert_knowledge_node({
            "id": "k1", "content": "Test", "category": "workflow",
            "confidence": 0.5, "source_sessions": [], "created_at": "",
            "updated_at": "", "status": "active", "superseded_by": "", "tags": [],
        })
        rowid = db.get_knowledge_rowid("k1")
        assert isinstance(rowid, int)
        assert rowid > 0
        assert db.get_knowledge_rowid("missing") is None
        db.close()

    def test_existing_db_gets_knowledge_table(self):
        """Opening an existing DB without knowledge table adds it without data loss."""
        import sqlite3
        # Create a minimal DB without knowledge table
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO metadata VALUES ('test', 'value')")
        conn.commit()
        conn.close()

        # Open with RecallDB — should add knowledge table
        from synapt.recall.storage import RecallDB
        db = RecallDB(self.db_path)
        # Metadata preserved
        val = db.get_metadata("test")
        self.assertEqual(val, "value")
        # Knowledge table exists
        self.assertEqual(db.knowledge_count(), 0)
        db.close()

    def test_migrate_knowledge_adds_contradiction_note(self):
        """Old knowledge table without contradiction_note gets the column added."""
        import sqlite3
        # Create a DB with a knowledge table missing contradiction_note
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL DEFAULT 0,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_name TEXT,
                file_path TEXT,
                timestamp TEXT NOT NULL DEFAULT '',
                token_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE knowledge (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.5,
                source_sessions TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                superseded_by TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]'
            );
        """)
        # Insert a node without contradiction_note
        conn.execute(
            "INSERT INTO knowledge (id, content, created_at, updated_at) "
            "VALUES ('old1', 'test fact', '2026-01-01', '2026-01-01')"
        )
        conn.commit()
        conn.close()

        # Open with RecallDB — migration should add the column
        from synapt.recall.storage import RecallDB
        db = RecallDB(self.db_path)
        # Can now save a node with contradiction_note
        db.upsert_knowledge_node({
            "id": "new1", "content": "New fact", "category": "workflow",
            "confidence": 0.5, "source_sessions": [], "created_at": "",
            "updated_at": "", "status": "active", "superseded_by": "",
            "contradiction_note": "replaces old approach", "tags": [],
        })
        loaded = db.load_knowledge_nodes()
        self.assertEqual(len(loaded), 2)
        # Old node preserved
        old = [n for n in loaded if n["id"] == "old1"][0]
        self.assertEqual(old["content"], "test fact")
        # New node has contradiction_note
        new = [n for n in loaded if n["id"] == "new1"][0]
        self.assertEqual(new["contradiction_note"], "replaces old approach")
        db.close()


class TestDedupKnowledgeNodes(unittest.TestCase):
    """Tests for dedup_knowledge_nodes() — Phase 3 of #333."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kn_path = Path(self.tmpdir) / "recall" / "knowledge.jsonl"
        self.kn_path.parent.mkdir(parents=True, exist_ok=True)
        # Patch project_data_dir to use our temp dir
        import synapt.recall.knowledge as kmod
        self._orig_knowledge_path = kmod._knowledge_path
        kmod._knowledge_path = lambda project_dir=None: self.kn_path
        # Also patch consolidate's _dedup_decisions_path
        import synapt.recall.consolidate as cmod
        self._orig_dedup_path = cmod._dedup_decisions_path
        cmod._dedup_decisions_path = lambda project_dir=None: (
            self.kn_path.parent / "dedup_decisions.jsonl"
        )

    def tearDown(self):
        import synapt.recall.knowledge as kmod
        import synapt.recall.consolidate as cmod
        kmod._knowledge_path = self._orig_knowledge_path
        cmod._dedup_decisions_path = self._orig_dedup_path

    def test_dedup_empty_noop(self):
        """No knowledge file → returns 0."""
        from synapt.recall.knowledge import dedup_knowledge_nodes
        result = dedup_knowledge_nodes(threshold=0.7)
        self.assertEqual(result, 0)

    def test_dedup_single_node_noop(self):
        """Only one active node → nothing to compare."""
        from synapt.recall.knowledge import dedup_knowledge_nodes
        node = KnowledgeNode.create(
            content="Always use A100 for training 8B models",
            category="infrastructure",
        )
        append_node(node, self.kn_path)
        result = dedup_knowledge_nodes(threshold=0.7)
        self.assertEqual(result, 0)

    def test_dedup_identical_content(self):
        """Identical content produces the same ID; _dedup_nodes keeps latest."""
        n1 = KnowledgeNode.create(
            content="Always use A100 for training 8B models",
            category="infrastructure",
            source_sessions=["sess-A"],
        )
        n1.updated_at = "2026-01-01T00:00:00"
        n2 = KnowledgeNode.create(
            content="Always use A100 for training 8B models",
            category="infrastructure",
            source_sessions=["sess-B"],
        )
        n2.updated_at = "2026-03-01T00:00:00"
        # Same content → same ID (content-hash)
        self.assertEqual(n1.id, n2.id)
        append_node(n1, self.kn_path)
        append_node(n2, self.kn_path)
        # _dedup_nodes keeps latest by updated_at, so only n2 survives
        active = read_nodes(self.kn_path, status="active")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].source_sessions, ["sess-B"])

    def test_dedup_markdown_formatting_diff(self):
        """Nodes differing only by markdown formatting should merge."""
        from synapt.recall.knowledge import dedup_knowledge_nodes
        n1 = KnowledgeNode.create(
            content="Train on Alfred eval set test on Batman never train and test on same benchmark",
            category="convention",
        )
        n1.updated_at = "2026-01-01T00:00:00"
        n2 = KnowledgeNode.create(
            content="Train on Alfred eval set, test on Batman — never train and test on the same benchmark",
            category="convention",
        )
        n2.updated_at = "2026-03-01T00:00:00"
        append_node(n1, self.kn_path)
        append_node(n2, self.kn_path)
        result = dedup_knowledge_nodes(threshold=0.7)
        self.assertEqual(result, 1)

    def test_dedup_below_threshold_kept(self):
        """Nodes with low similarity should both survive."""
        from synapt.recall.knowledge import dedup_knowledge_nodes
        n1 = KnowledgeNode.create(
            content="Always use A100 for training 8B models",
            category="infrastructure",
        )
        n2 = KnowledgeNode.create(
            content="Run scripts/verify_quality_curve.py before any training run",
            category="workflow",
        )
        append_node(n1, self.kn_path)
        append_node(n2, self.kn_path)
        result = dedup_knowledge_nodes(threshold=0.7)
        self.assertEqual(result, 0)
        active = read_nodes(self.kn_path, status="active")
        self.assertEqual(len(active), 2)

    def test_dedup_keeps_newest(self):
        """Survivor should be the node with the most recent updated_at."""
        from synapt.recall.knowledge import dedup_knowledge_nodes
        old = KnowledgeNode.create(
            content="Use A100 GPU for training large models always",
            category="infrastructure",
        )
        old.updated_at = "2026-01-01T00:00:00"
        new = KnowledgeNode.create(
            content="Use A100 GPU for training large models",
            category="infrastructure",
        )
        new.updated_at = "2026-03-09T00:00:00"
        append_node(old, self.kn_path)
        append_node(new, self.kn_path)
        result = dedup_knowledge_nodes(threshold=0.7)
        self.assertEqual(result, 1)
        active = read_nodes(self.kn_path, status="active")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].id, new.id)
        # Old node should be contradicted
        contradicted = read_nodes(self.kn_path, status="contradicted")
        self.assertEqual(len(contradicted), 1)
        self.assertEqual(contradicted[0].id, old.id)
        self.assertEqual(contradicted[0].superseded_by, new.id)

    def test_dedup_transfers_sessions(self):
        """Survivor should have source_sessions from both nodes."""
        from synapt.recall.knowledge import dedup_knowledge_nodes
        n1 = KnowledgeNode.create(
            content="Use A100 GPU for training large models",
            category="infrastructure",
            source_sessions=["sess-A", "sess-B"],
        )
        n1.updated_at = "2026-01-01T00:00:00"
        n2 = KnowledgeNode.create(
            content="Use A100 GPU for training large models always",
            category="infrastructure",
            source_sessions=["sess-C"],
        )
        n2.updated_at = "2026-03-01T00:00:00"
        append_node(n1, self.kn_path)
        append_node(n2, self.kn_path)
        result = dedup_knowledge_nodes(threshold=0.7)
        self.assertEqual(result, 1)
        active = read_nodes(self.kn_path, status="active")
        self.assertEqual(len(active), 1)
        # All sessions from both nodes present
        sessions = set(active[0].source_sessions)
        self.assertIn("sess-A", sessions)
        self.assertIn("sess-B", sessions)
        self.assertIn("sess-C", sessions)

    def test_dedup_logs_decisions(self):
        """Merge decisions should be logged to dedup_decisions.jsonl."""
        from synapt.recall.knowledge import dedup_knowledge_nodes
        n1 = KnowledgeNode.create(
            content="Always use A100 GPU for training 8B parameter models on the cluster",
            category="infrastructure",
        )
        n1.updated_at = "2026-01-01T00:00:00"
        n2 = KnowledgeNode.create(
            content="Always use A100 GPU for training 8B parameter models on cluster nodes",
            category="infrastructure",
        )
        n2.updated_at = "2026-03-01T00:00:00"
        # Different content → different IDs, but high Jaccard overlap
        self.assertNotEqual(n1.id, n2.id)
        append_node(n1, self.kn_path)
        append_node(n2, self.kn_path)
        dedup_knowledge_nodes(threshold=0.7)
        log_path = self.kn_path.parent / "dedup_decisions.jsonl"
        self.assertTrue(log_path.exists())
        with open(log_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["action"], "dedup-merge")
        self.assertTrue(
            entries[0]["source"].startswith("knowledge-dedup"),
            f"Expected source starting with 'knowledge-dedup', got {entries[0]['source']}",
        )


    def test_dedup_triple_duplicate_no_session_loss(self):
        """Three near-identical nodes should merge to one without losing sessions."""
        from synapt.recall.knowledge import dedup_knowledge_nodes
        n1 = KnowledgeNode.create(
            content="Use A100 GPU for training large models",
            category="infrastructure",
            source_sessions=["sess-A"],
        )
        n1.updated_at = "2026-01-01T00:00:00"
        n2 = KnowledgeNode.create(
            content="Use A100 GPU for training large models always",
            category="infrastructure",
            source_sessions=["sess-B"],
        )
        n2.updated_at = "2026-02-01T00:00:00"
        n3 = KnowledgeNode.create(
            content="Use A100 GPU for training large models on cloud",
            category="infrastructure",
            source_sessions=["sess-C"],
        )
        n3.updated_at = "2026-03-01T00:00:00"
        append_node(n1, self.kn_path)
        append_node(n2, self.kn_path)
        append_node(n3, self.kn_path)
        result = dedup_knowledge_nodes(threshold=0.7)
        self.assertEqual(result, 2)
        active = read_nodes(self.kn_path, status="active")
        self.assertEqual(len(active), 1)
        # Survivor must be the newest (n3)
        self.assertEqual(active[0].id, n3.id)
        # All sessions from all three nodes must be present
        sessions = set(active[0].source_sessions)
        self.assertIn("sess-A", sessions)
        self.assertIn("sess-B", sessions)
        self.assertIn("sess-C", sessions)


class TestColdStartKnowledgeSearch(unittest.TestCase):
    """Cold-start scenario: knowledge saved via recall_save but no transcript
    chunks exist yet. Verifies that lookup() and _concise_lookup() still
    surface knowledge nodes. Regression test for recall#592."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "recall.db"

    def _make_index_with_knowledge(self, nodes: list[dict]) -> "TranscriptIndex":
        """Create a TranscriptIndex with DB-backed knowledge but zero chunks."""
        from synapt.recall.core import TranscriptIndex
        from synapt.recall.storage import RecallDB

        db = RecallDB(self.db_path)
        db.save_knowledge_nodes(nodes)
        # No chunks saved -- simulates cold start
        index = TranscriptIndex([], db=db)
        return index

    def test_concise_lookup_surfaces_knowledge(self):
        """depth='concise' should return knowledge nodes even without chunks."""
        from synapt.recall.core import TranscriptIndex

        nodes = [
            {
                "id": "cold_test_01",
                "content": "The deploy pipeline uses Terraform with S3 backend",
                "category": "infrastructure",
                "status": "active",
                "confidence": 0.8,
                "source_sessions": ["sess-cold-1"],
            },
        ]
        index = self._make_index_with_knowledge(nodes)
        result = index.lookup(
            "deploy pipeline Terraform",
            max_chunks=5,
            max_tokens=500,
            depth="concise",
        )
        self.assertIn("Terraform", result)
        index._db.close()

    def test_full_lookup_surfaces_knowledge(self):
        """depth='full' should return knowledge nodes even without chunks."""
        from synapt.recall.core import TranscriptIndex

        nodes = [
            {
                "id": "cold_test_02",
                "content": "User prefers dark mode in all editors",
                "category": "preference",
                "status": "active",
                "confidence": 0.9,
                "source_sessions": ["sess-cold-2"],
            },
        ]
        index = self._make_index_with_knowledge(nodes)
        result = index.lookup(
            "dark mode editor preference",
            max_chunks=5,
            max_tokens=500,
            depth="full",
        )
        self.assertIn("dark mode", result)
        index._db.close()

    def test_empty_db_still_returns_empty(self):
        """No chunks AND no knowledge should still return empty string."""
        from synapt.recall.core import TranscriptIndex
        from synapt.recall.storage import RecallDB

        db = RecallDB(self.db_path)
        index = TranscriptIndex([], db=db)
        result = index.lookup("anything", max_chunks=5)
        self.assertEqual(result, "")
        db.close()

    def test_stale_knowledge_not_surfaced(self):
        """Stale knowledge nodes should not appear in cold-start search."""
        from synapt.recall.core import TranscriptIndex

        nodes = [
            {
                "id": "cold_stale_01",
                "content": "Old API key for staging environment",
                "category": "infrastructure",
                "status": "stale",
                "confidence": 0.5,
                "source_sessions": ["sess-stale"],
            },
        ]
        index = self._make_index_with_knowledge(nodes)
        result = index.lookup(
            "API key staging",
            max_chunks=5,
            max_tokens=500,
            depth="concise",
        )
        # Stale nodes should not appear (knowledge_count only counts active)
        self.assertEqual(result, "")
        index._db.close()

    def test_multiple_knowledge_nodes_ranked(self):
        """Multiple knowledge nodes compete by score in cold-start."""
        from synapt.recall.core import TranscriptIndex

        nodes = [
            {
                "id": "cold_multi_01",
                "content": "Database uses PostgreSQL 16 with pgvector extension",
                "category": "infrastructure",
                "status": "active",
                "confidence": 0.9,
                "source_sessions": ["sess-m1"],
            },
            {
                "id": "cold_multi_02",
                "content": "Redis cache layer sits in front of PostgreSQL",
                "category": "infrastructure",
                "status": "active",
                "confidence": 0.7,
                "source_sessions": ["sess-m2"],
            },
        ]
        index = self._make_index_with_knowledge(nodes)
        result = index.lookup(
            "PostgreSQL database",
            max_chunks=5,
            max_tokens=500,
            depth="concise",
        )
        self.assertIn("PostgreSQL", result)
        index._db.close()


class TestColdStartOnboarding(unittest.TestCase):
    """Verify that realistic onboarding knowledge nodes surface for the five
    cold-start query categories identified in recall#592.

    Each test seeds a fresh DB with onboarding nodes (no transcript chunks)
    and asserts that TranscriptIndex.lookup() returns content matching the
    expected keywords. This is the acceptance criteria for cold-start recall.
    """

    # Shared onboarding knowledge nodes — the minimum set a fresh agent needs.
    ONBOARDING_NODES = [
        {
            "id": "onboard_ceremony",
            "content": (
                "Sprint ceremony process: at sprint end, create a ceremony PR from "
                "sprint-N into main. All tests must be green, cargo fmt check passes. "
                "Squash merge only. PR title format: 'Sprint N ceremony'. "
                "Run full test suite before merging. Never merge with failing tests on main."
            ),
            "category": "workflow",
            "status": "active",
            "confidence": 0.95,
            "source_sessions": ["onboarding"],
        },
        {
            "id": "onboard_tdd",
            "content": (
                "Development follows strict TDD: write failing tests first, PR tests "
                "into the sprint branch, then implement until tests pass. Sprint branches "
                "allow failing tests. Main never has failures. Feature branches are named "
                "test/feature-name for tests and impl/feature-name for implementation."
            ),
            "category": "workflow",
            "status": "active",
            "confidence": 0.95,
            "source_sessions": ["onboarding"],
        },
        {
            "id": "onboard_tools",
            "content": (
                "Core tools: gr (gripspace orchestrator) for multi-repo git operations "
                "— gr status, gr sync, gr branch, gr commit, gr push, gr pr create. "
                "recall tools: recall_quick for fast knowledge checks, recall_search for "
                "full search, recall_channel for team communication, recall_journal for "
                "session notes. pytest for Python tests, cargo test for Rust."
            ),
            "category": "tools",
            "status": "active",
            "confidence": 0.95,
            "source_sessions": ["onboarding"],
        },
        {
            "id": "onboard_team",
            "content": (
                "Team roster: Layne (human, founder/CEO), Opus (AI, CTO/lead engineer, "
                "Claude Opus), Apollo (AI, Rust engineer, Claude Sonnet), Sentinel (AI, "
                "QA/review, Claude Opus), Atlas (AI, research/coordination, Claude Sonnet via Codex). "
                "Each agent owns their own worktree — never enter another agent's workspace."
            ),
            "category": "team",
            "status": "active",
            "confidence": 0.95,
            "source_sessions": ["onboarding"],
        },
        {
            "id": "onboard_repos",
            "content": (
                "Repository structure: synapt (public OSS — recall memory, MCP server, "
                "plugin system), a private extension package (eval, training, repair engine), "
                "gitgrip (multi-repo workspace orchestrator in Rust). Managed as a gripspace "
                "under synapt-global/."
            ),
            "category": "architecture",
            "status": "active",
            "confidence": 0.95,
            "source_sessions": ["onboarding"],
        },
    ]

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "recall.db"

    def _make_index(self) -> "TranscriptIndex":
        from synapt.recall.core import TranscriptIndex
        from synapt.recall.storage import RecallDB

        db = RecallDB(self.db_path)
        db.save_knowledge_nodes(self.ONBOARDING_NODES)
        return TranscriptIndex([], db=db)

    def _lookup(self, query: str) -> str:
        index = self._make_index()
        result = index.lookup(query, max_chunks=10, max_tokens=1000, depth="concise")
        index._db.close()
        return result

    def test_cold_start_ceremony(self):
        """Query about ceremony process must mention key steps."""
        result = self._lookup("how do we run sprint ceremony?")
        self.assertTrue(result, "Expected non-empty result for ceremony query")
        result_lower = result.lower()
        self.assertIn("ceremony", result_lower)
        self.assertIn("sprint", result_lower)
        # Must mention at least one of: tests green, main, merge
        self.assertTrue(
            any(kw in result_lower for kw in ["tests", "green", "main", "merge"]),
            f"Expected ceremony details in result: {result[:200]}",
        )

    def test_cold_start_tools(self):
        """Query about tools must mention gr commands and recall tools."""
        result = self._lookup("what tools do we use for development?")
        self.assertTrue(result, "Expected non-empty result for tools query")
        result_lower = result.lower()
        # Must mention gr
        self.assertTrue(
            any(kw in result_lower for kw in ["gr ", "gr status", "gripspace"]),
            f"Expected gr tool reference in result: {result[:200]}",
        )

    def test_cold_start_workflow(self):
        """Query about development workflow must mention TDD."""
        result = self._lookup("how do we develop code? TDD process")
        self.assertTrue(result, "Expected non-empty result for workflow query")
        result_lower = result.lower()
        self.assertTrue(
            any(kw in result_lower for kw in ["tdd", "failing tests", "test"]),
            f"Expected TDD reference in result: {result[:200]}",
        )

    def test_cold_start_team(self):
        """Query about team must mention agents and roles."""
        result = self._lookup("who is on the team?")
        self.assertTrue(result, "Expected non-empty result for team query")
        result_lower = result.lower()
        # Must mention at least Layne and Opus
        self.assertIn("layne", result_lower)
        self.assertTrue(
            any(name in result_lower for name in ["opus", "apollo", "sentinel", "atlas"]),
            f"Expected agent names in result: {result[:200]}",
        )

    def test_cold_start_repos(self):
        """Query about project structure must mention repos."""
        result = self._lookup("what repos are in this project?")
        self.assertTrue(result, "Expected non-empty result for repos query")
        result_lower = result.lower()
        self.assertTrue(
            any(kw in result_lower for kw in ["synapt", "gitgrip", "gripspace"]),
            f"Expected repo names in result: {result[:200]}",
        )

    def test_all_onboarding_nodes_indexed(self):
        """All 5 onboarding nodes must be stored and active."""
        from synapt.recall.storage import RecallDB

        db = RecallDB(self.db_path)
        db.save_knowledge_nodes(self.ONBOARDING_NODES)
        count = db.knowledge_count()
        self.assertEqual(count, 5, f"Expected 5 active nodes, got {count}")
        db.close()


if __name__ == "__main__":
    unittest.main()
