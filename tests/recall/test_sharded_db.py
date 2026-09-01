"""Tests for ShardedRecallDB — tree-structured storage wrapper."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.resume import BoundedResumeIndex, build_resume_view
from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.storage import RecallDB


class TestShardedRecallDBMonolithic(unittest.TestCase):
    """Test ShardedRecallDB in monolithic mode (single recall.db)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_dir = Path(self.tmpdir)

    def test_open_creates_recall_db(self):
        db = ShardedRecallDB.open(self.index_dir)
        self.assertTrue(db.is_monolithic)
        self.assertEqual(db.shard_count, 0)
        db.close()

    def test_open_existing_recall_db(self):
        # Create a monolithic DB first
        RecallDB(self.index_dir / "recall.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        self.assertTrue(db.is_monolithic)
        db.close()

    def test_open_readonly_uses_query_only_connection(self):
        writer = RecallDB(self.index_dir / "recall.db")
        writer.close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        self.assertTrue(db.is_monolithic)
        with self.assertRaises(Exception):
            db._index._conn.execute("DELETE FROM chunks")
        db.close()

    def test_open_readonly_skips_schema_work(self):
        RecallDB(self.index_dir / "recall.db").close()

        with mock.patch.object(
            RecallDB,
            "_ensure_schema",
            side_effect=AssertionError("read-only open ran schema work"),
        ):
            db = ShardedRecallDB.open_readonly(self.index_dir)
            self.assertEqual(db.chunk_count(), 0)
            db.close()

    def test_knowledge_roundtrip(self):
        db = ShardedRecallDB.open(self.index_dir)
        node = {
            "id": "test-node",
            "content": "test fact",
            "category": "workflow",
            "confidence": 0.9,
            "source_sessions": [],
            "source_turns": [],
            "source_offsets": [],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "status": "active",
            "superseded_by": "",
            "contradiction_note": "",
            "tags": "",
            "valid_from": None,
            "valid_until": None,
            "version": 1,
            "lineage_id": "",
        }
        db.save_knowledge_nodes([node])
        nodes = db.load_knowledge_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["content"], "test fact")
        db.close()

    def test_pending_contradictions(self):
        db = ShardedRecallDB.open(self.index_dir)
        self.assertEqual(db.pending_contradiction_count(), 0)
        db.close()

    def test_passthrough_via_getattr(self):
        """Unhandled methods fall through to index DB."""
        db = ShardedRecallDB.open(self.index_dir)
        # load_manifest is explicitly delegated, but _path is on RecallDB
        self.assertIsNotNone(db._path)
        db.close()

    def test_close_is_safe(self):
        db = ShardedRecallDB.open(self.index_dir)
        db.close()
        # Double close shouldn't crash
        db.close()


class TestShardedRecallDBSharded(unittest.TestCase):
    """Test ShardedRecallDB with index.db + data shards."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_dir = Path(self.tmpdir)

    def _make_chunk(
        self,
        chunk_id: str,
        session_id: str,
        timestamp: str,
        text: str,
        agent_id: str | None = None,
    ) -> TranscriptChunk:
        return TranscriptChunk(
            id=chunk_id,
            session_id=session_id,
            timestamp=timestamp,
            turn_index=0,
            user_text=text,
            assistant_text="assistant",
            agent_id=agent_id,
        )

    def _create_two_shard_layout(self) -> ShardedRecallDB:
        RecallDB(self.index_dir / "index.db").close()
        first = RecallDB(self.index_dir / "data_001.db")
        second = RecallDB(self.index_dir / "data_002.db")
        first.save_chunks([
            self._make_chunk("s1:t0", "s1", "2026-01-01T00:00:00Z", "alpha memory"),
        ])
        second.save_chunks([
            self._make_chunk("s2:t0", "s2", "2026-01-02T00:00:00Z", "beta memory"),
        ])
        first.close()
        second.close()
        return ShardedRecallDB.open(self.index_dir)

    def test_open_detects_sharded_layout(self):
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()

        db = ShardedRecallDB.open(self.index_dir)
        self.assertFalse(db.is_monolithic)
        self.assertEqual(db.shard_count, 1)
        db.close()

    def test_open_readonly_detects_sharded_layout(self):
        self._create_two_shard_layout().close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        self.assertFalse(db.is_monolithic)
        self.assertEqual(db.shard_count, 2)
        self.assertEqual(db.chunk_count(), 2)
        db.close()

    def test_bounded_session_reads_merge_across_shards(self):
        RecallDB(self.index_dir / "index.db").close()
        first = RecallDB(self.index_dir / "data_001.db")
        second = RecallDB(self.index_dir / "data_002.db")
        first.save_chunks([
            self._make_chunk("shared:t0", "shared", "2026-01-01T00:00:00Z", "first"),
        ])
        second.save_chunks([
            TranscriptChunk(
                id="shared:t1",
                session_id="shared",
                timestamp="2026-01-02T00:00:00Z",
                turn_index=1,
                user_text="second",
                assistant_text="assistant",
            ),
        ])
        first.close()
        second.close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        try:
            self.assertIn("shared", db.session_activity())
            self.assertEqual(
                [chunk.id for chunk in db.load_session_chunks("shared")],
                ["shared:t0", "shared:t1"],
            )
        finally:
            db.close()

    def test_session_overview_unions_agent_identity_across_shards(self):
        RecallDB(self.index_dir / "index.db").close()
        first = RecallDB(self.index_dir / "data_001.db")
        second = RecallDB(self.index_dir / "data_002.db")
        first.save_chunks([
            self._make_chunk(
                "shared:t0",
                "shared",
                "2026-01-01T00:00:00Z",
                "claude runtime",
                agent_id="runtime-a",
            ),
        ])
        second.save_chunks([
            self._make_chunk(
                "shared:t1",
                "shared",
                "2026-01-02T00:00:00Z",
                "codex runtime",
                agent_id="runtime-b",
            ),
        ])
        first.close()
        second.close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        try:
            self.assertEqual(
                db.session_overview()["shared"]["agent_ids"],
                frozenset({"runtime-a", "runtime-b"}),
            )
        finally:
            db.close()

    def test_session_overview_unions_query_tail_agent_identity_with_base(self):
        RecallDB(self.index_dir / "index.db").close()
        shard = RecallDB(self.index_dir / "data_001.db")
        shard.save_chunks([
            self._make_chunk(
                "shared:t0",
                "shared",
                "2026-01-01T00:00:00Z",
                "base",
                agent_id="claude-agent",
            ),
        ])
        shard.close()
        index = RecallDB(self.index_dir / "index.db")
        index.replace_query_tail(
            source_key="runtime-source",
            session_id="shared",
            rewind_offset=1,
            chunks=[
                self._make_chunk(
                    "shared:t1",
                    "shared",
                    "2026-01-02T00:00:00Z",
                    "fresh overlay",
                    agent_id="codex-agent",
                )
            ],
            cursor={
                "transcript_path": "/runtime/shared.jsonl",
                "observed_complete_offset": 2,
                "rewind_offset": 1,
                "rewind_turn_index": 1,
                "source_size": 2,
                "source_mtime_ns": 2,
                "observed_prefix_sha256": "digest",
                "suppresses_base": False,
                "latest_projected_timestamp": "2026-01-02T00:00:00Z",
                "last_attempt_at": "2026-01-02T00:00:00Z",
                "last_success_at": "2026-01-02T00:00:00Z",
            },
        )
        index.close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        try:
            self.assertEqual(
                db.session_overview()["shared"]["agent_ids"],
                frozenset({"claude-agent", "codex-agent"}),
            )
        finally:
            db.close()

    def test_bounded_resume_prefers_newest_agent_session_from_query_tail(self):
        RecallDB(self.index_dir / "index.db").close()
        shard = RecallDB(self.index_dir / "data_001.db")
        shard.save_chunks([
            self._make_chunk(
                "old:t0",
                "old-session",
                "2026-01-01T00:00:00Z",
                "old base continuity",
                agent_id="stable-agent",
            ),
        ])
        shard.close()
        index_db = RecallDB(self.index_dir / "index.db")
        index_db.replace_query_tail(
            source_key="new-runtime-source",
            session_id="new-session",
            rewind_offset=0,
            chunks=[
                self._make_chunk(
                    "new:t0",
                    "new-session",
                    "2026-02-01T00:00:00Z",
                    "new query-tail continuity",
                    agent_id="stable-agent",
                )
            ],
            cursor={
                "transcript_path": "/runtime/new-session.jsonl",
                "observed_complete_offset": 1,
                "rewind_offset": 0,
                "rewind_turn_index": 0,
                "source_size": 1,
                "source_mtime_ns": 1,
                "observed_prefix_sha256": "digest",
                "suppresses_base": False,
                "latest_projected_timestamp": "2026-02-01T00:00:00Z",
                "last_attempt_at": "2026-02-01T00:00:00Z",
                "last_success_at": "2026-02-01T00:00:00Z",
            },
        )
        index_db.close()

        index = BoundedResumeIndex(ShardedRecallDB.open_readonly(self.index_dir))
        try:
            view = build_resume_view(
                index,
                agent_id="stable-agent",
                journal_path=None,
            )
        finally:
            index.close()

        self.assertEqual(view.session_id, "new-session")
        self.assertEqual(view.turns[0].user_text, "new query-tail continuity")

    def test_query_tail_journal_metadata_does_not_outrank_real_agent_activity(self):
        RecallDB(self.index_dir / "index.db").close()
        shard = RecallDB(self.index_dir / "data_001.db")
        shard.save_chunks([
            self._make_chunk(
                "real:t0",
                "real-session",
                "2026-01-01T00:00:00Z",
                "real continuity",
                agent_id="stable-agent",
            ),
        ])
        shard.close()
        index_db = RecallDB(self.index_dir / "index.db")
        journal = self._make_chunk(
            "journal:j0",
            "journal-only",
            "2026-02-01T00:00:00Z",
            "journal metadata",
            agent_id="stable-agent",
        )
        journal.turn_index = -1
        journal.assistant_text = ""
        index_db.replace_query_tail(
            source_key="journal-overlay",
            session_id="journal-only",
            rewind_offset=0,
            chunks=[journal],
            cursor={
                "transcript_path": "/runtime/journal.jsonl",
                "observed_complete_offset": 1,
                "rewind_offset": 0,
                "rewind_turn_index": -1,
                "source_size": 1,
                "source_mtime_ns": 1,
                "observed_prefix_sha256": "digest",
                "suppresses_base": False,
                "latest_projected_timestamp": "2026-02-01T00:00:00Z",
                "last_attempt_at": "2026-02-01T00:00:00Z",
                "last_success_at": "2026-02-01T00:00:00Z",
            },
        )
        index_db.close()

        index = BoundedResumeIndex(ShardedRecallDB.open_readonly(self.index_dir))
        try:
            view = build_resume_view(
                index, agent_id="stable-agent", journal_path=None
            )
        finally:
            index.close()

        self.assertEqual(view.session_id, "real-session")
        self.assertEqual(view.turns[0].user_text, "real continuity")

    def test_journal_in_a_later_shard_does_not_replace_real_activity(self):
        RecallDB(self.index_dir / "index.db").close()
        first = RecallDB(self.index_dir / "data_001.db")
        second = RecallDB(self.index_dir / "data_002.db")
        first.save_chunks([
            self._make_chunk("dead:t0", "dead", "2026-01-01T00:00:00Z", "old"),
            self._make_chunk("live:t0", "live", "2026-01-02T00:00:00Z", "new"),
        ])
        second.save_chunks([
            TranscriptChunk(
                id="dead:journal",
                session_id="dead",
                timestamp="2026-08-25T00:00:00Z",
                turn_index=-1,
                user_text="journal",
                assistant_text="",
            ),
        ])
        first.close()
        second.close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        try:
            activity = db.session_activity()
            self.assertGreater(activity["live"], activity["dead"])
        finally:
            db.close()

    def test_multiple_shards(self):
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        RecallDB(self.index_dir / "data_002.db").close()
        RecallDB(self.index_dir / "data_003.db").close()

        db = ShardedRecallDB.open(self.index_dir)
        self.assertEqual(db.shard_count, 3)
        db.close()

    def test_save_chunks_creates_fresh_shards(self):
        """Sharded save_chunks creates fresh shard(s) for the data."""
        from synapt.recall.core import TranscriptChunk
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        chunk = TranscriptChunk(
            id="test:t0", session_id="s1", timestamp="2025-04-15T10:00:00Z",
            turn_index=0, user_text="hello", assistant_text="hi",
        )
        db.save_chunks([chunk])
        # 1 chunk = 1 shard
        self.assertEqual(db.shard_count, 1)
        self.assertEqual(db.chunk_count(), 1)
        db.close()

    def test_save_chunks_empty_is_noop(self):
        """Saving empty list to sharded DB doesn't crash."""
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        db.save_chunks([])
        db.close()

    def test_chunk_count_spans_all_shards(self):
        db = self._create_two_shard_layout()
        self.assertEqual(db.chunk_count(), 2)
        db.close()

    def test_chunk_id_rowid_map_is_globally_unique(self):
        db = self._create_two_shard_layout()
        mapping = db.get_chunk_id_rowid_map()
        self.assertEqual(set(mapping.keys()), {"s1:t0", "s2:t0"})
        self.assertEqual(len(set(mapping.values())), 2)
        self.assertNotEqual(mapping["s1:t0"], mapping["s2:t0"])
        db.close()

    def test_fts_search_returns_shard_qualified_rowids(self):
        db = self._create_two_shard_layout()
        hits = db.fts_search("memory", limit=10)
        self.assertEqual(len(hits), 2)
        self.assertEqual(len({rowid for rowid, _ in hits}), 2)
        self.assertTrue(all(rowid > (1 << 32) for rowid, _ in hits))
        db.close()

    def test_fts_search_raw_returns_shard_qualified_rowids(self):
        db = self._create_two_shard_layout()
        hits = db.fts_search_raw("memory", limit=10)
        self.assertEqual(len(hits), 2)
        self.assertEqual(len({rowid for rowid, _ in hits}), 2)
        self.assertTrue(all(rowid > (1 << 32) for rowid, _ in hits))
        db.close()

    def test_get_all_embeddings_uses_shard_qualified_rowids(self):
        db = self._create_two_shard_layout()
        mapping = db.get_chunk_id_rowid_map()
        emb1 = [0.1] * 384
        emb2 = [0.2] * 384
        db.save_embeddings({
            mapping["s1:t0"]: emb1,
            mapping["s2:t0"]: emb2,
        })
        loaded = db.get_all_embeddings()
        self.assertEqual(set(loaded.keys()), set(mapping.values()))
        self.assertAlmostEqual(loaded[mapping["s1:t0"]][0], emb1[0], places=6)
        self.assertAlmostEqual(loaded[mapping["s2:t0"]][0], emb2[0], places=6)
        db.close()

    def test_content_hash_spans_all_shards_in_global_timestamp_order(self):
        db = self._create_two_shard_layout()
        import hashlib

        h = hashlib.sha256()
        h.update("s2:t0|beta memory|assistant|\n".encode())
        h.update("s1:t0|alpha memory|assistant|\n".encode())
        self.assertEqual(db.content_hash(), h.hexdigest()[:16])
        db.close()

    def test_transcript_index_load_can_search_sharded_chunks(self):
        self._create_two_shard_layout().close()
        index = TranscriptIndex.load(self.index_dir)
        result = index.lookup("beta", max_chunks=5, max_tokens=200)
        self.assertIn("beta memory", result)
        self.assertNotEqual(index._rowid_to_idx, {})
        self.assertIsNone(index._bm25)

    def test_load_chunk_by_rowid_uses_shard_qualified_ids(self):
        db = self._create_two_shard_layout()
        mapping = db.get_chunk_id_rowid_map()
        chunk = db.load_chunk_by_rowid(mapping["s2:t0"])
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.id, "s2:t0")
        loaded = db.load_chunks_by_rowids([mapping["s1:t0"], mapping["s2:t0"]])
        self.assertEqual(set(loaded.keys()), {mapping["s1:t0"], mapping["s2:t0"]})
        self.assertEqual(loaded[mapping["s1:t0"]].id, "s1:t0")
        self.assertEqual(loaded[mapping["s2:t0"]].id, "s2:t0")

    def test_sample_chunk_texts_reads_across_shards(self):
        db = self._create_two_shard_layout()
        samples = db.sample_chunk_texts(limit=10)
        joined = " ".join(samples)
        self.assertIn("alpha memory", joined)
        self.assertIn("beta memory", joined)
        db.close()

    def test_save_chunks_reshards_properly(self):
        """save_chunks deletes old shards and redistributes across fresh ones."""
        from synapt.recall.core import TranscriptChunk
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        RecallDB(self.index_dir / "data_002.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        self.assertEqual(db.shard_count, 2)

        # Seed shard 1 with old data
        old_chunk = TranscriptChunk(
            id="old:t0", session_id="s1", timestamp="2025-01-01T00:00:00Z",
            turn_index=0, user_text="old", assistant_text="stale",
        )
        db._data_dbs[0].save_chunks([old_chunk])

        # Now do a full rebuild save (like rescrub does)
        new_chunk = TranscriptChunk(
            id="new:t0", session_id="s2", timestamp="2025-06-01T00:00:00Z",
            turn_index=0, user_text="new", assistant_text="fresh",
        )
        db.save_chunks([new_chunk])

        # Old shard files should be deleted; 1 chunk = 1 fresh shard
        self.assertEqual(db.shard_count, 1)
        self.assertFalse(
            (self.index_dir / "data_002.db").exists(),
            "Old shard file should be deleted",
        )

        # Active shard should have the new chunk
        self.assertEqual(db.chunk_count(), 1)

        # FTS search should only find the new chunk
        results = db.fts_search("stale")
        self.assertEqual(len(results), 0, "Stale data should not appear in FTS")
        results = db.fts_search("fresh")
        self.assertEqual(len(results), 1)
        db.close()

    def test_save_chunks_no_unbounded_shard_growth(self):
        """Repeated rebuilds must not accumulate shard files (the sprint-13 bug)."""
        from synapt.recall.core import TranscriptChunk
        from synapt.recall.sharding import list_shards

        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        db = ShardedRecallDB.open(self.index_dir)

        chunks = [
            TranscriptChunk(
                id=f"c{i}:t0", session_id=f"s{i}",
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
                turn_index=0, user_text=f"chunk {i}", assistant_text="a",
            )
            for i in range(5)
        ]

        # Simulate 3 consecutive rebuilds (the bug created a new shard each time)
        for rebuild in range(3):
            db.save_chunks(chunks)

        shard_files = list_shards(self.index_dir)
        self.assertEqual(
            len(shard_files), 1,
            f"Expected 1 shard after rebuilds, got {len(shard_files)}: "
            f"{[p.name for p in shard_files]}",
        )
        self.assertEqual(db.chunk_count(), 5)
        db.close()

    def test_save_chunks_splits_at_threshold(self):
        """Chunks exceeding threshold are split across multiple shards."""
        from synapt.recall.core import TranscriptChunk
        from synapt.recall.sharding import list_shards
        import synapt.recall.sharding as _sharding_mod

        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        db = ShardedRecallDB.open(self.index_dir)

        chunks = [
            TranscriptChunk(
                id=f"c{i}:t0", session_id=f"s{i}",
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
                turn_index=0, user_text=f"chunk {i}", assistant_text="a",
            )
            for i in range(25)
        ]

        # Temporarily lower the threshold for testing
        original = _sharding_mod.SHARD_CHUNK_THRESHOLD
        _sharding_mod.SHARD_CHUNK_THRESHOLD = 10
        try:
            db.save_chunks(chunks)
        finally:
            _sharding_mod.SHARD_CHUNK_THRESHOLD = original

        # 25 chunks / 10 per shard = 3 shards
        self.assertEqual(db.shard_count, 3)
        self.assertEqual(db.chunk_count(), 25)

        shard_files = list_shards(self.index_dir)
        self.assertEqual(len(shard_files), 3)
        db.close()


if __name__ == "__main__":
    unittest.main()
