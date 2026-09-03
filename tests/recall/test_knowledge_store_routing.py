"""Consolidation/enrichment wrote knowledge nodes and metadata
into ``index_dir / "recall.db"`` unconditionally. On a sharded store
(``index.db`` present), ``ShardedRecallDB``'s primary path never reads
``recall.db`` -- so new knowledge silently landed in a file nobody
searches. Measured live on real stores before this fix: one store's
``knowledge_fts_search`` was returning ZERO results despite real
knowledge nodes existing.

Each test below sets up a directory with an empty ``index.db`` already
present (the sharded marker) -- never touching a live store -- and asserts
the write/read lands in ``index.db``, with a monolithic control confirming
the untouched, no-index.db case is unaffected.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from synapt.recall.knowledge import KnowledgeNode, append_node
from synapt.recall.storage import RecallDB


def _sharded_dir() -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "index.db").touch()
    RecallDB(d / "index.db").close()  # real schema, not just an empty file
    return d


def _monolithic_dir() -> Path:
    d = Path(tempfile.mkdtemp())  # no index.db at all
    RecallDB(d / "recall.db").close()  # matches a real monolithic store's own init
    return d


class TestSyncKnowledgeToDbRouting(unittest.TestCase):

    def _write_node_and_sync(self, project_dir: Path, index_dir: Path):
        from synapt.recall.consolidate import _sync_knowledge_to_db
        from synapt.recall.core import project_data_dir

        data_dir = project_data_dir(project_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        kn_path = data_dir / "knowledge.jsonl"
        node = KnowledgeNode.create(
            content="test node for store routing", category="configuration",
            source_sessions=["s0"], node_id="route-test-1",
        )
        append_node(node, kn_path)
        _sync_knowledge_to_db(project_dir, kn_path)

    def test_sharded_store_writes_knowledge_into_index_db_not_recall_db(self):
        index_dir = _sharded_dir()
        project_dir = index_dir.parent  # project_index_dir resolves .synapt/recall/index under project root; patch below
        import synapt.recall.core as core_mod
        from unittest.mock import patch

        import synapt.recall.consolidate as consolidate_mod
        with patch.object(consolidate_mod, "project_index_dir", return_value=index_dir):
            self._write_node_and_sync(project_dir, index_dir)

        db = RecallDB(index_dir / "index.db")
        self.assertIsNotNone(db.get_knowledge_node("route-test-1"))
        db.close()
        self.assertFalse(
            (index_dir / "recall.db").exists(),
            "sync must not create/write recall.db on a sharded store",
        )

    def test_monolithic_store_still_writes_knowledge_into_recall_db(self):
        """Control: unaffected case. No index.db at all -- the node must
        still land in recall.db, matching pre-fix behavior."""
        index_dir = _monolithic_dir()
        project_dir = index_dir.parent
        import synapt.recall.core as core_mod
        from unittest.mock import patch

        import synapt.recall.consolidate as consolidate_mod
        with patch.object(consolidate_mod, "project_index_dir", return_value=index_dir):
            self._write_node_and_sync(project_dir, index_dir)

        self.assertTrue((index_dir / "recall.db").exists())
        db = RecallDB(index_dir / "recall.db")
        self.assertIsNotNone(db.get_knowledge_node("route-test-1"))
        db.close()


class TestTimestampMetadataRouting(unittest.TestCase):
    """The two lower-stakes writers (_set_last_consolidation_ts,
    _set_last_enrichment_ts) have the identical unconditional-path bug."""

    def test_set_last_consolidation_ts_writes_into_index_db_on_a_sharded_store(self):
        from synapt.recall.consolidate import _set_last_consolidation_ts
        import synapt.recall.consolidate as consolidate_mod
        from unittest.mock import patch

        index_dir = _sharded_dir()
        project_dir = index_dir.parent
        with patch.object(consolidate_mod, "project_index_dir", return_value=index_dir):
            _set_last_consolidation_ts(project_dir)

        db = RecallDB(index_dir / "index.db")
        self.assertTrue(db.get_metadata("last_consolidation_ts"))
        db.close()
        self.assertFalse((index_dir / "recall.db").exists())

    def test_set_last_enrichment_ts_writes_into_index_db_on_a_sharded_store(self):
        from synapt.recall.enrich import _set_last_enrichment_ts
        import synapt.recall.core as core_mod
        from unittest.mock import patch

        index_dir = _sharded_dir()
        project_dir = index_dir.parent
        with patch.object(core_mod, "project_index_dir", return_value=index_dir):
            _set_last_enrichment_ts(project_dir)

        db = RecallDB(index_dir / "index.db")
        self.assertTrue(db.get_metadata("last_enrichment_ts"))
        db.close()
        self.assertFalse((index_dir / "recall.db").exists())


class TestServerCallSiteRouting(unittest.TestCase):
    """Two more unconverted recall.db call sites in server.py,
    found during review of this PR's original 8-site fix. format_contradictions_
    for_session_start is the urgent one -- this PR's own write-side change
    (routing contradiction-queuing to index.db) makes it a live regression:
    before this PR, its recall.db read agreed with the recall.db write, so
    contradictions surfaced despite both sides being on the wrong file; after
    this PR, the write moves and the read doesn't, so new contradictions on a
    sharded store go invisible at every session start. recall_save is the
    same bug class (RecallDB opened directly on recall.db, no is_sharded()
    check at all), not urgent, but the same helper applies."""

    def test_format_contradictions_for_session_start_sees_a_sharded_store_contradiction(self):
        from synapt.recall import server as server_mod
        from unittest.mock import patch

        index_dir = _sharded_dir()
        db = RecallDB(index_dir / "index.db")
        old_node = KnowledgeNode.create(
            content="old fact", category="configuration",
            source_sessions=["s0"], node_id="old-1",
        )
        db.save_knowledge_nodes([old_node.to_dict()])
        db.add_pending_contradiction(
            old_node_id="old-1", new_content="new conflicting fact",
            category="configuration", reason="test",
        )
        db.close()

        with patch.object(server_mod, "project_index_dir", return_value=index_dir):
            output = server_mod.format_contradictions_for_session_start()

        self.assertIn("Pending contradictions (1)", output)
        self.assertIn("new conflicting fact", output)
        self.assertFalse((index_dir / "recall.db").exists())

    def test_format_contradictions_for_session_start_monolithic_control(self):
        """Control: unaffected case, no index.db, must still read recall.db."""
        from synapt.recall import server as server_mod
        from unittest.mock import patch

        index_dir = _monolithic_dir()
        db = RecallDB(index_dir / "recall.db")
        old_node = KnowledgeNode.create(
            content="old fact", category="configuration",
            source_sessions=["s0"], node_id="old-1",
        )
        db.save_knowledge_nodes([old_node.to_dict()])
        db.add_pending_contradiction(
            old_node_id="old-1", new_content="new conflicting fact",
            category="configuration", reason="test",
        )
        db.close()

        with patch.object(server_mod, "project_index_dir", return_value=index_dir):
            output = server_mod.format_contradictions_for_session_start()

        self.assertIn("Pending contradictions (1)", output)
        self.assertIn("new conflicting fact", output)

    def test_recall_save_writes_into_index_db_on_a_sharded_store(self):
        from synapt.recall import server as server_mod
        from unittest.mock import patch
        import os

        index_dir = _sharded_dir()
        project_dir = index_dir.parent
        cwd_before = os.getcwd()
        os.chdir(project_dir)
        try:
            with patch.object(server_mod, "project_index_dir", return_value=index_dir):
                result = server_mod.recall_save(content="a durable fact", category="workflow")
        finally:
            os.chdir(cwd_before)

        self.assertNotIn("Error", result)
        db = RecallDB.open_readonly(index_dir / "index.db")
        nodes = db.load_knowledge_nodes(status=None)
        db.close()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["content"], "a durable fact")
        self.assertFalse((index_dir / "recall.db").exists())


if __name__ == "__main__":
    unittest.main()
