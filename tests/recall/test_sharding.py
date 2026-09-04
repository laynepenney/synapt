"""Tests for synapt.recall.sharding — size-based sharding utilities."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.sharding import (
    shard_name_for_index,
    list_shards,
    next_shard_index,
    is_sharded,
    live_store_path,
    estimate_split,
    split_monolithic_db,
    SHARD_CHUNK_THRESHOLD,
)


class TestShardNaming(unittest.TestCase):

    def test_sequential_names(self):
        self.assertEqual(shard_name_for_index(1), "data_001.db")
        self.assertEqual(shard_name_for_index(42), "data_042.db")
        self.assertEqual(shard_name_for_index(100), "data_100.db")

    def test_next_shard_index_empty(self):
        tmpdir = tempfile.mkdtemp()
        self.assertEqual(next_shard_index(Path(tmpdir)), 1)

    def test_next_shard_index_existing(self):
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "data_001.db").touch()
        (d / "data_002.db").touch()
        self.assertEqual(next_shard_index(d), 3)

    def test_next_shard_index_ignores_legacy_names(self):
        """Legacy data_YYYY_qN.db shards must not confuse the index parser."""
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "data_001.db").touch()
        (d / "data_2025_q1.db").touch()  # legacy shard
        self.assertEqual(next_shard_index(d), 2)


class TestListShards(unittest.TestCase):

    def test_finds_data_dbs(self):
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "data_001.db").touch()
        (d / "data_002.db").touch()
        (d / "index.db").touch()
        (d / "other.db").touch()

        shards = list_shards(d)
        self.assertEqual(len(shards), 2)
        self.assertEqual(shards[0].name, "data_001.db")
        self.assertEqual(shards[1].name, "data_002.db")

    def test_empty_dir(self):
        tmpdir = tempfile.mkdtemp()
        self.assertEqual(list_shards(Path(tmpdir)), [])


class TestIsSharded(unittest.TestCase):

    def test_monolithic(self):
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "recall.db").touch()
        self.assertFalse(is_sharded(d))

    def test_sharded(self):
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "index.db").touch()
        (d / "data_001.db").touch()
        self.assertTrue(is_sharded(d))


class TestLiveStorePath(unittest.TestCase):
    """Several callers (consolidate.py, knowledge.py, enrich.py)
    constructed ``index_dir / "recall.db"`` unconditionally, resolving to a
    file the primary ShardedRecallDB path never reads once index.db exists.
    live_store_path() is the one helper both cases should route through."""

    def test_monolithic_resolves_to_recall_db(self):
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "recall.db").touch()
        self.assertEqual(live_store_path(d), d / "recall.db")

    def test_sharded_resolves_to_index_db_even_when_recall_db_also_exists(self):
        """The exact production shape: a sharded store that ALSO still has
        a stale recall.db left over from an earlier migration. The live
        file must be index.db, not whichever file happens to exist."""
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "index.db").touch()
        (d / "recall.db").touch()
        self.assertEqual(live_store_path(d), d / "index.db")

    def test_neither_file_exists_yet_resolves_to_recall_db(self):
        """A brand-new project: is_sharded() is False (no index.db), so a
        fresh monolithic recall.db is where the first write should go —
        matches ShardedRecallDB.open()'s own "otherwise" branch."""
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        self.assertEqual(live_store_path(d), d / "recall.db")


class TestEstimateSplit(unittest.TestCase):

    def test_empty_db(self):
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        conn.close()
        self.assertEqual(estimate_split(db_path), {})

    def test_splits_by_threshold(self):
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE chunks (timestamp TEXT)")
        # Insert 25 chunks, threshold 10
        for i in range(25):
            conn.execute("INSERT INTO chunks VALUES (?)", (f"2025-01-{i+1:02d}T10:00:00Z",))
        conn.commit()
        conn.close()

        result = estimate_split(db_path, threshold=10)
        self.assertEqual(result["data_001.db"], 10)
        self.assertEqual(result["data_002.db"], 10)
        self.assertEqual(result["data_003.db"], 5)


class TestSplitMonolithicDb(unittest.TestCase):

    def _create_monolithic(self, tmpdir, num_chunks=25):
        from synapt.recall.storage import RecallDB
        db = RecallDB(Path(tmpdir) / "recall.db")
        db._conn.execute(
            "INSERT INTO knowledge (id, content, category, confidence, status, "
            "source_sessions, source_turns, source_offsets, created_at, updated_at, "
            "superseded_by, contradiction_note, tags, valid_from, valid_until, version, lineage_id) "
            "VALUES ('k1', 'test fact', 'workflow', 0.8, 'active', '[]', '[]', '[]', "
            "'2025-01-01', '2025-01-01', '', '', '', NULL, NULL, 1, '')"
        )
        for i in range(num_chunks):
            user_text = f"user chunk {i}"
            assistant_text = f"assistant unique_term_{i}"
            db._conn.execute(
                "INSERT INTO chunks (id, session_id, timestamp, turn_index, user_text, "
                "assistant_text, tools_used, files_touched, tool_content, transcript_path, "
                "byte_offset, byte_length) "
                f"VALUES ('c{i}', 's1', '2025-01-{(i % 28) + 1:02d}T10:00:00Z', {i}, "
                f"?, ?, '[]', '[]', '', '', 0, 0)",
                (user_text, assistant_text),
            )
        db._conn.commit()
        db.close()

    def test_split_creates_index_and_shards(self):
        tmpdir = tempfile.mkdtemp()
        self._create_monolithic(tmpdir, num_chunks=25)
        d = Path(tmpdir)

        plan = split_monolithic_db(d, threshold=10)

        self.assertIn("index.db", plan)
        self.assertTrue((d / "index.db").exists())
        self.assertTrue((d / "data_001.db").exists())
        self.assertTrue((d / "data_002.db").exists())
        self.assertTrue((d / "data_003.db").exists())
        self.assertEqual(plan["data_001.db"], 10)
        self.assertEqual(plan["data_002.db"], 10)
        self.assertEqual(plan["data_003.db"], 5)

    def test_split_preserves_knowledge_in_index(self):
        tmpdir = tempfile.mkdtemp()
        self._create_monolithic(tmpdir)
        d = Path(tmpdir)
        split_monolithic_db(d, threshold=10)

        conn = sqlite3.connect(str(d / "index.db"))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM knowledge WHERE id = 'k1'").fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["content"], "test fact")

    def test_shard_metadata_created(self):
        tmpdir = tempfile.mkdtemp()
        self._create_monolithic(tmpdir, num_chunks=25)
        d = Path(tmpdir)
        split_monolithic_db(d, threshold=10)

        conn = sqlite3.connect(str(d / "index.db"))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM shard_metadata ORDER BY shard_name"
        ).fetchall()
        conn.close()

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["shard_name"], "data_001.db")
        self.assertEqual(rows[0]["chunk_count"], 10)
        self.assertEqual(rows[2]["shard_name"], "data_003.db")
        self.assertEqual(rows[2]["is_active"], 1)  # Last shard is active

    def test_split_produces_queryable_shards(self):
        tmpdir = tempfile.mkdtemp()
        self._create_monolithic(tmpdir, num_chunks=25)
        d = Path(tmpdir)
        split_monolithic_db(d, threshold=10)

        db = ShardedRecallDB.open(d)
        try:
            results = db.fts_search("unique_term_17")
            self.assertEqual(len(results), 1)
        finally:
            db.close()

    def test_split_supports_real_transcript_index_search(self):
        from synapt.recall.core import TranscriptIndex

        tmpdir = tempfile.mkdtemp()
        self._create_monolithic(tmpdir, num_chunks=25)
        d = Path(tmpdir)
        split_monolithic_db(d, threshold=10)

        index = TranscriptIndex.load(d, use_embeddings=False)
        try:
            self.assertEqual(index._db.chunk_count(), 25)
            self.assertEqual(len(index._rowid_to_idx), 25)

            late_result = index.lookup("unique_term_17", max_chunks=3, max_tokens=500)
            early_result = index.lookup("unique_term_3", max_chunks=3, max_tokens=500)

            self.assertIn("unique_term_17", late_result)
            self.assertIn("unique_term_3", early_result)
        finally:
            if index._db is not None:
                index._db.close()

    def test_dry_run_doesnt_write(self):
        tmpdir = tempfile.mkdtemp()
        self._create_monolithic(tmpdir)
        d = Path(tmpdir)

        plan = split_monolithic_db(d, threshold=10, dry_run=True)
        self.assertIn("data_001.db", plan)
        self.assertFalse((d / "index.db").exists())

    def test_already_sharded_raises(self):
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "index.db").touch()
        (d / "recall.db").touch()
        with self.assertRaises(RuntimeError):
            split_monolithic_db(d)

    def test_no_recall_db_raises(self):
        tmpdir = tempfile.mkdtemp()
        with self.assertRaises(FileNotFoundError):
            split_monolithic_db(Path(tmpdir))

    def test_failure_cleans_up_temp_files(self):
        tmpdir = tempfile.mkdtemp()
        self._create_monolithic(tmpdir)
        d = Path(tmpdir)

        from unittest.mock import patch
        real_fn = shard_name_for_index
        call_count = [0]

        def failing_fn(idx):
            call_count[0] += 1
            if call_count[0] > 1:
                raise OSError("Simulated failure")
            return real_fn(idx)

        with patch("synapt.recall.sharding.shard_name_for_index", failing_fn):
            with self.assertRaises(OSError):
                split_monolithic_db(d, threshold=10)

        self.assertFalse((d / "index.db").exists())
        self.assertEqual(list(d.glob("data_*.db")), [])
        self.assertEqual(list(d.glob(".split_tmp_*")), [])


if __name__ == "__main__":
    unittest.main()
