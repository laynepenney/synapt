"""Recovery, take two: the first repair script (diff recall.db against
index.db) was WRONG -- measured live, recall.db had drifted with its own
years of dead content (stale test fixtures, contradicted nodes never
pruned) independent of knowledge.jsonl, so diffing against it would have
copied hundreds of dead rows into production. knowledge.jsonl is the
actual source of truth. This script resyncs from it, and never opens
recall.db at all -- the test below proves a recall.db-only node is
never even considered, let alone copied."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.knowledge import KnowledgeNode, append_node
from synapt.recall.storage import RecallDB

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "resync-knowledge-from-jsonl.py"
_spec = importlib.util.spec_from_file_location("resync_knowledge_from_jsonl", _SCRIPT_PATH)
resync_mod = importlib.util.module_from_spec(_spec)
sys.modules["resync_knowledge_from_jsonl"] = resync_mod
_spec.loader.exec_module(resync_mod)


def _node(node_id: str, content: str, created_at: str) -> KnowledgeNode:
    n = KnowledgeNode.create(content=content, category="configuration", source_sessions=["s0"], node_id=node_id)
    n.created_at = created_at
    return n


def _fixture(*, sharded: bool, with_recall_db: bool = False):
    """A project dir with knowledge.jsonl at its real resolved path, and
    a store (sharded or monolithic) under index_dir. Patches
    project_data_dir/project_index_dir on the script module -- the real
    resolution logic (git worktree walk-up, gripspace root, env
    overrides) is exercised elsewhere; this test is about the sync
    logic, not path resolution."""
    project_dir = Path(tempfile.mkdtemp())
    data_dir = project_dir / ".synapt" / "recall"
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True)

    if sharded:
        RecallDB(index_dir / "index.db").close()
    else:
        RecallDB(index_dir / "recall.db").close()

    if with_recall_db and sharded:
        RecallDB(index_dir / "recall.db").close()

    patches = [
        patch.object(resync_mod, "project_data_dir", return_value=data_dir),
        patch.object(resync_mod, "project_index_dir", return_value=index_dir),
    ]
    return project_dir, data_dir, index_dir, patches


class TestFindMissingNodes(unittest.TestCase):

    def test_refuses_when_knowledge_jsonl_absent(self):
        project_dir, data_dir, index_dir, patches = _fixture(sharded=True)
        for p in patches:
            p.start()
        try:
            with self.assertRaises(FileNotFoundError):
                resync_mod.find_missing_nodes(project_dir)
        finally:
            for p in patches:
                p.stop()

    def test_refuses_when_live_store_absent(self):
        project_dir, data_dir, index_dir, patches = _fixture(sharded=True)
        append_node(_node("n1", "x", "2026-01-01T00:00:00Z"), data_dir / "knowledge.jsonl")
        (index_dir / "index.db").unlink()  # remove the live store the fixture created
        for p in patches:
            p.start()
        try:
            with self.assertRaises(FileNotFoundError):
                resync_mod.find_missing_nodes(project_dir)
        finally:
            for p in patches:
                p.stop()

    def test_finds_jsonl_nodes_absent_from_a_sharded_live_store(self):
        project_dir, data_dir, index_dir, patches = _fixture(sharded=True)
        append_node(_node("in-both", "present in index.db too", "2026-01-01T00:00:00Z"), data_dir / "knowledge.jsonl")
        append_node(_node("jsonl-only", "genuinely missing from the store", "2026-08-30T00:00:00Z"), data_dir / "knowledge.jsonl")

        idx = RecallDB(index_dir / "index.db")
        idx.save_knowledge_nodes([_node("in-both", "present in index.db too", "2026-01-01T00:00:00Z").to_dict()])
        idx.close()

        for p in patches:
            p.start()
        try:
            missing = resync_mod.find_missing_nodes(project_dir)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual({n["id"] for n in missing}, {"jsonl-only"})

    def test_a_recall_db_only_node_is_never_copied(self):
        """The exact defect the first repair script had: a node that
        exists ONLY in recall.db (not in knowledge.jsonl at all) must
        never be considered 'missing' here -- this script's only source
        is knowledge.jsonl, and it never opens recall.db."""
        project_dir, data_dir, index_dir, patches = _fixture(sharded=True, with_recall_db=True)
        append_node(_node("jsonl-node", "the only real source", "2026-08-30T00:00:00Z"), data_dir / "knowledge.jsonl")

        stale = RecallDB(index_dir / "recall.db")
        stale.save_knowledge_nodes([
            _node("recall-db-only", "years of drift never in jsonl", "2026-04-01T00:00:00Z").to_dict(),
        ])
        stale.close()

        for p in patches:
            p.start()
        try:
            missing = resync_mod.find_missing_nodes(project_dir)
        finally:
            for p in patches:
                p.stop()

        ids = {n["id"] for n in missing}
        self.assertIn("jsonl-node", ids)
        self.assertNotIn("recall-db-only", ids, "a node absent from knowledge.jsonl must never be copied, regardless of recall.db's content")

    def test_finds_jsonl_nodes_absent_from_a_monolithic_live_store(self):
        project_dir, data_dir, index_dir, patches = _fixture(sharded=False)
        append_node(_node("jsonl-only", "missing", "2026-08-30T00:00:00Z"), data_dir / "knowledge.jsonl")
        for p in patches:
            p.start()
        try:
            missing = resync_mod.find_missing_nodes(project_dir)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual({n["id"] for n in missing}, {"jsonl-only"})


class TestApplyResync(unittest.TestCase):

    def test_apply_writes_missing_nodes_into_the_live_store(self):
        project_dir, data_dir, index_dir, patches = _fixture(sharded=True)
        append_node(_node("jsonl-only", "missing", "2026-08-30T00:00:00Z"), data_dir / "knowledge.jsonl")
        for p in patches:
            p.start()
        try:
            missing = resync_mod.find_missing_nodes(project_dir)
            resync_mod.apply_resync(project_dir, missing)
        finally:
            for p in patches:
                p.stop()

        idx = RecallDB.open_readonly(index_dir / "index.db")
        node = idx.get_knowledge_node("jsonl-only")
        idx.close()
        self.assertIsNotNone(node)

    def test_apply_never_touches_recall_db(self):
        project_dir, data_dir, index_dir, patches = _fixture(sharded=True, with_recall_db=True)
        append_node(_node("jsonl-only", "missing", "2026-08-30T00:00:00Z"), data_dir / "knowledge.jsonl")
        before = (index_dir / "recall.db").stat().st_mtime
        for p in patches:
            p.start()
        try:
            missing = resync_mod.find_missing_nodes(project_dir)
            resync_mod.apply_resync(project_dir, missing)
        finally:
            for p in patches:
                p.stop()
        after = (index_dir / "recall.db").stat().st_mtime
        self.assertEqual(before, after)

    def test_resync_is_idempotent(self):
        project_dir, data_dir, index_dir, patches = _fixture(sharded=True)
        append_node(_node("jsonl-only", "missing", "2026-08-30T00:00:00Z"), data_dir / "knowledge.jsonl")
        for p in patches:
            p.start()
        try:
            missing1 = resync_mod.find_missing_nodes(project_dir)
            resync_mod.apply_resync(project_dir, missing1)
            missing2 = resync_mod.find_missing_nodes(project_dir)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(missing2, [])


class TestCLIEndToEnd(unittest.TestCase):
    """Real subprocess invocations of the actual script."""

    def _run(self, project_dir: Path, *extra_args: str, env_overrides: dict | None = None):
        import subprocess
        import os
        env = os.environ.copy()
        env["SYNAPT_RECALL_ROOT"] = str(project_dir)
        if env_overrides:
            env.update(env_overrides)
        return subprocess.run(
            [sys.executable, str(_SCRIPT_PATH), str(project_dir), *extra_args],
            capture_output=True, text=True, env=env,
        )

    def test_dry_run_via_real_project_dir_writes_nothing(self):
        """No patching here -- a real project_dir with a real .synapt/recall/
        layout, exercised through the actual CLI process."""
        project_dir = Path(tempfile.mkdtemp())
        data_dir = project_dir / ".synapt" / "recall"
        index_dir = data_dir / "index"
        index_dir.mkdir(parents=True)
        RecallDB(index_dir / "index.db").close()
        append_node(_node("jsonl-only", "missing", "2026-08-30T00:00:00Z"), data_dir / "knowledge.jsonl")

        before = (index_dir / "index.db").stat().st_size
        result = self._run(project_dir)
        after = (index_dir / "index.db").stat().st_size

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jsonl-only", result.stdout)
        self.assertIn("DRY RUN", result.stdout)
        self.assertEqual(before, after)

    def test_apply_via_real_project_dir_writes_and_verifies(self):
        project_dir = Path(tempfile.mkdtemp())
        data_dir = project_dir / ".synapt" / "recall"
        index_dir = data_dir / "index"
        index_dir.mkdir(parents=True)
        RecallDB(index_dir / "index.db").close()
        append_node(_node("jsonl-only", "missing", "2026-08-30T00:00:00Z"), data_dir / "knowledge.jsonl")

        result = self._run(project_dir, "--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("APPLIED", result.stdout)
        self.assertIn("0 nodes still missing", result.stdout)

    def test_refuses_cleanly_without_knowledge_jsonl(self):
        project_dir = Path(tempfile.mkdtemp())
        data_dir = project_dir / ".synapt" / "recall"
        index_dir = data_dir / "index"
        index_dir.mkdir(parents=True)
        RecallDB(index_dir / "index.db").close()
        result = self._run(project_dir)
        self.assertEqual(result.returncode, 2)
        self.assertIn("REFUSED", result.stderr)


if __name__ == "__main__":
    unittest.main()
