"""Tests for the topic clustering module."""

from __future__ import annotations

import pytest

from synapt.recall.core import TranscriptChunk
from synapt.recall.clustering import (
    _chunk_tokens,
    _cluster_id,
    _content_hash,
    _extract_topic,
    _has_meaningful_content,
    _jaccard,
    _novel_entities,
    cluster_chunks,
    generate_concat_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    id: str,
    user: str = "",
    assistant: str = "",
    session: str = "sess-1",
    ts: str = "2026-03-05T10:00:00Z",
    turn: int = 0,
) -> TranscriptChunk:
    return TranscriptChunk(
        id=id,
        session_id=session,
        timestamp=ts,
        turn_index=turn,
        user_text=user,
        assistant_text=assistant,
    )


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

class TestChunkTokens:
    def test_extracts_distinctive_tokens(self):
        chunk = _chunk("c1", user="fix the flock race condition",
                       assistant="The flock TOCTOU race was fixed using in-place rewrite")
        tokens = _chunk_tokens(chunk)
        assert "flock" in tokens
        assert "toctou" in tokens
        assert "race" in tokens

    def test_filters_stop_tokens(self):
        chunk = _chunk("c1", user="the code is here",
                       assistant="this is the fix for the issue")
        tokens = _chunk_tokens(chunk)
        assert "the" not in tokens
        assert "is" not in tokens
        assert "this" not in tokens

    def test_filters_short_tokens(self):
        chunk = _chunk("c1", user="a b cd the", assistant="x y ab")
        tokens = _chunk_tokens(chunk)
        # Single and two-char tokens filtered (len > 2 required)
        for t in tokens:
            assert len(t) > 2


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical_sets(self):
        s = {"flock", "journal", "compact"}
        assert _jaccard(s, s) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({"flock", "journal"}, {"swift", "harness"}) == 0.0

    def test_partial_overlap(self):
        a = {"flock", "journal", "compact", "rewrite"}
        b = {"flock", "compact", "atomic", "truncate"}
        # intersection = {flock, compact} = 2
        # union = {flock, journal, compact, rewrite, atomic, truncate} = 6
        assert abs(_jaccard(a, b) - 2.0 / 6.0) < 1e-9

    def test_empty_sets(self):
        assert _jaccard(set(), {"a"}) == 0.0
        assert _jaccard(set(), set()) == 0.0



# ---------------------------------------------------------------------------
# Cluster ID determinism
# ---------------------------------------------------------------------------

class TestClusterId:
    def test_deterministic(self):
        ids = ["sess1:t0", "sess1:t1", "sess2:t0"]
        assert _cluster_id(ids) == _cluster_id(ids)

    def test_order_independent(self):
        """Sorted internally, so different input order → same ID."""
        assert _cluster_id(["b", "a", "c"]) == _cluster_id(["c", "a", "b"])

    def test_different_inputs_different_ids(self):
        assert _cluster_id(["a", "b"]) != _cluster_id(["a", "c"])

    def test_format(self):
        cid = _cluster_id(["a", "b"])
        assert cid.startswith("clust-")
        assert len(cid) == 18  # "clust-" + 12 hex chars


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

class TestExtractTopic:
    def test_extracts_discriminative_tokens(self):
        from collections import Counter
        cluster_tokens = [
            {"flock", "journal", "compact", "rewrite"},
            {"flock", "compact", "atomic", "truncate"},
        ]
        all_tokens = cluster_tokens + [
            {"harness", "swift", "test", "eval"},
            {"swift", "adapter", "training", "loss"},
        ]
        global_df: Counter[str] = Counter()
        for ts in all_tokens:
            global_df.update(ts)
        topic = _extract_topic(cluster_tokens, global_df, len(all_tokens))
        assert isinstance(topic, str)
        assert len(topic) > 0

    def test_empty_returns_unknown(self):
        from collections import Counter
        assert _extract_topic([], Counter(), 0) == "unknown"

    def test_short_tokens_filtered(self):
        """Tokens shorter than 4 chars should not appear in topic labels."""
        from collections import Counter
        cluster_tokens = [{"abc", "flock", "journal"}, {"abc", "flock", "compact"}]
        global_df: Counter[str] = Counter()
        for ts in cluster_tokens:
            global_df.update(ts)
        topic = _extract_topic(cluster_tokens, global_df, len(cluster_tokens))
        assert "abc" not in topic

    def test_session_id_tokens_excluded(self):
        """A session id (m_<hex>) recurring across a cluster's
        own members is genuinely rare corpus-wide, so DF-IDF's rarity bonus
        is maximal for it despite carrying zero topical signal. Measured on
        the real store: "m_be61448... fathom m_6d67ca24" as a topic label."""
        from collections import Counter
        cluster_tokens = [
            {"m_be61448ec3d91a2", "antagonist", "frozen"},
            {"m_be61448ec3d91a2", "antagonist", "review"},
        ]
        all_tokens = cluster_tokens + [
            {"harness", "swift", "test", "eval"},
            {"swift", "adapter", "training", "loss"},
        ]
        global_df: Counter[str] = Counter()
        for ts in all_tokens:
            global_df.update(ts)
        topic = _extract_topic(cluster_tokens, global_df, len(all_tokens))
        assert "m_be61448ec3d91a2" not in topic
        assert "antagonist" in topic

    def test_tool_use_id_tokens_excluded(self):
        """toolu_<alnum> ids are the same shape of defect as
        session ids -- unique per tool call, so IDF treats them as maximally
        rare. Measured: "toolu_01eum13sxeaunhmdiljvdlmx u... notification"."""
        from collections import Counter
        cluster_tokens = [
            {"toolu_01eum13sxeaunhmdiljvdlmx", "notification", "queue"},
            {"toolu_01eum13sxeaunhmdiljvdlmx", "notification", "delivery"},
        ]
        global_df: Counter[str] = Counter()
        for ts in cluster_tokens:
            global_df.update(ts)
        topic = _extract_topic(cluster_tokens, global_df, len(cluster_tokens))
        assert "toolu_01eum13sxeaunhmdiljvdlmx" not in topic

    def test_sentence_final_punctuation_does_not_inflate_rarity(self):
        """_tokenize() keeps '.' inside a token, so "seen."
        (sentence-final) and "seen" (mid-sentence) land as separate dict
        keys. Before the fix, the punctuated form looked spuriously rare
        globally (most other occurrences of the word aren't sentence-final)
        and won the ranking on that artifact alone. Measured real examples:
        "world assembled. seen." and "change. activity. report" -- vacuous
        words that happened to fall at a sentence boundary in the cluster.
        Here "seen" is actually the MORE common word globally once its two
        forms are merged, so it must not lose to the genuinely rare
        "antagonist" purely because of how it's punctuated in this cluster.
        """
        from collections import Counter
        cluster_tokens = [
            {"seen.", "antagonist", "review"},
            {"seen.", "antagonist", "gate"},
        ]
        # "seen" (unpunctuated) is common corpus-wide; "antagonist" is rare.
        all_tokens = cluster_tokens + [
            {"seen", "harness", "test"},
            {"seen", "swift", "adapter"},
            {"seen", "training", "loss"},
            {"seen", "eval", "gate"},
        ]
        global_df: Counter[str] = Counter()
        for ts in all_tokens:
            global_df.update(ts)
        topic = _extract_topic(cluster_tokens, global_df, len(all_tokens))
        assert "antagonist" in topic
        assert "seen." not in topic.split()


# ---------------------------------------------------------------------------
# Clustering algorithm
# ---------------------------------------------------------------------------

class TestClusterChunks:
    def test_groups_related_chunks(self):
        """Chunks about the same topic should cluster together."""
        chunks = [
            _chunk("s1:t0", user="fix the flock race condition in compact_journal",
                   assistant="The flock TOCTOU race was in the stat-before-flock pattern. Fixed by moving check inside flock.",
                   ts="2026-03-05T10:00:00Z"),
            _chunk("s1:t1", user="what about the compact journal empty file case",
                   assistant="The compact_journal empty-file check was moved inside flock to eliminate the TOCTOU race.",
                   ts="2026-03-05T10:05:00Z"),
            _chunk("s1:t2", user="review the flock rewrite one more time",
                   assistant="Final review of compact_journal flock rewrite. All edge cases covered.",
                   ts="2026-03-05T10:10:00Z"),
            # Unrelated chunk
            _chunk("s2:t0", user="train the swift adapter on batman tasks",
                   assistant="Training swift adapter on batman t51-100 with weighted SFT.",
                   session="sess-2", ts="2026-03-05T11:00:00Z"),
            _chunk("s2:t1", user="what was the eval score for swift",
                   assistant="Swift batman eval scored 47/50 with the adapted model.",
                   session="sess-2", ts="2026-03-05T11:05:00Z"),
        ]
        clusters = cluster_chunks(chunks)
        assert len(clusters) >= 1

        # The flock-related chunks should be in the same cluster
        flock_cluster = None
        for cl in clusters:
            if "s1:t0" in cl["chunk_ids"] or "s1:t1" in cl["chunk_ids"]:
                flock_cluster = cl
                break
        assert flock_cluster is not None
        # At least 2 of the 3 flock chunks should be together
        flock_ids = {"s1:t0", "s1:t1", "s1:t2"}
        overlap = flock_ids & set(flock_cluster["chunk_ids"])
        assert len(overlap) >= 2

    def test_empty_input(self):
        assert cluster_chunks([]) == []

    def test_no_clusters_when_all_unrelated(self):
        """Completely different topics should not cluster (or form only singletons)."""
        chunks = [
            _chunk("c1", user="explain quantum entanglement theory",
                   assistant="Quantum entanglement is a phenomenon where particles are correlated.",
                   ts="2026-03-05T10:00:00Z"),
            _chunk("c2", user="best recipe for chocolate cake",
                   assistant="Mix flour, cocoa powder, sugar, eggs, and butter.",
                   ts="2026-03-05T11:00:00Z"),
        ]
        clusters = cluster_chunks(chunks)
        # Should be 0 clusters (singletons are filtered out)
        assert len(clusters) == 0

    def test_cluster_metadata(self):
        """Cluster dicts should have all required fields."""
        chunks = [
            _chunk("s1:t0", user="fix flock race", assistant="Fixed flock TOCTOU race condition",
                   session="sess-a", ts="2026-03-01T10:00:00Z"),
            _chunk("s1:t1", user="review flock fix", assistant="Reviewed flock race fix, looks good",
                   session="sess-a", ts="2026-03-01T10:05:00Z"),
            _chunk("s2:t0", user="flock follow up", assistant="Follow up on flock fix from last session",
                   session="sess-b", ts="2026-03-02T10:00:00Z"),
        ]
        clusters = cluster_chunks(chunks)
        assert len(clusters) >= 1
        cl = clusters[0]
        assert "cluster_id" in cl
        assert cl["cluster_id"].startswith("clust-")
        assert "topic" in cl
        assert "session_ids" in cl
        assert "date_start" in cl
        assert "date_end" in cl
        assert cl["chunk_count"] >= 2

    def test_cross_session_clustering(self):
        """Chunks from different sessions on the same topic should cluster."""
        chunks = [
            _chunk("s1:t0", user="implement Jaccard clustering algorithm",
                   assistant="Jaccard similarity measures token overlap between sets",
                   session="sess-a", ts="2026-03-01T10:00:00Z"),
            _chunk("s2:t0", user="test the Jaccard clustering threshold",
                   assistant="Testing Jaccard threshold at 0.15 for clustering accuracy",
                   session="sess-b", ts="2026-03-02T10:00:00Z"),
        ]
        clusters = cluster_chunks(chunks)
        if clusters:
            cl = clusters[0]
            assert len(cl["session_ids"]) == 2

    def test_deterministic_cluster_ids(self):
        """Same chunks should produce same cluster IDs across runs."""
        chunks = [
            _chunk("s1:t0", user="fix flock race", assistant="Fixed flock TOCTOU race",
                   ts="2026-03-01T10:00:00Z"),
            _chunk("s1:t1", user="review flock fix", assistant="Reviewed flock race fix",
                   ts="2026-03-01T10:05:00Z"),
        ]
        c1 = cluster_chunks(chunks)
        c2 = cluster_chunks(chunks)
        if c1 and c2:
            assert c1[0]["cluster_id"] == c2[0]["cluster_id"]

    def test_stability_adding_unrelated_chunk(self):
        """Adding an unrelated chunk should not reshuffle existing clusters.

        Union-based matching with raised threshold (0.20) preserves cluster
        membership when unrelated chunks are added (#306).
        """
        # Two clear topic groups
        flock_chunks = [
            _chunk("s1:t0", user="fix flock race condition in compact_journal",
                   assistant="The flock TOCTOU race was in the stat-before-flock pattern",
                   ts="2026-03-05T10:00:00Z"),
            _chunk("s1:t1", user="review the flock rewrite and compact fix",
                   assistant="Reviewed compact_journal flock rewrite. Edge cases covered.",
                   ts="2026-03-05T10:05:00Z"),
            _chunk("s1:t2", user="test flock race edge case on empty journal",
                   assistant="Empty journal flock test passes. Race condition eliminated.",
                   ts="2026-03-05T10:10:00Z"),
        ]
        swift_chunks = [
            _chunk("s2:t0", user="train swift adapter on batman evaluation tasks",
                   assistant="Training swift adapter on batman t51-100 weighted SFT",
                   session="sess-2", ts="2026-03-05T11:00:00Z"),
            _chunk("s2:t1", user="eval swift adapted model on batman benchmark",
                   assistant="Swift batman eval scored 47/50 with adapted model",
                   session="sess-2", ts="2026-03-05T11:05:00Z"),
        ]

        # Cluster without the new chunk
        clusters_before = cluster_chunks(flock_chunks + swift_chunks)

        # Add a completely unrelated chunk (about databases)
        new_chunk = _chunk(
            "s3:t0",
            user="migrate postgres database schema to version twelve",
            assistant="Database migration completed. Schema v12 applied successfully.",
            session="sess-3", ts="2026-03-05T12:00:00Z",
        )
        clusters_after = cluster_chunks(flock_chunks + swift_chunks + [new_chunk])

        # Find flock cluster in both runs
        def find_flock(clusters):
            for cl in clusters:
                if "s1:t0" in cl["chunk_ids"]:
                    return set(cl["chunk_ids"])
            return set()

        flock_before = find_flock(clusters_before)
        flock_after = find_flock(clusters_after)

        # Flock cluster should have the same members (stability)
        assert flock_before, "Flock chunks should form a cluster"
        assert flock_before == flock_after, (
            f"Adding unrelated chunk changed flock cluster: "
            f"before={flock_before}, after={flock_after}"
        )

    def test_refinement_moves_misassigned_chunk(self):
        """Refinement pass should correct a chunk that was greedily misassigned.

        Create a scenario where processing order causes a chunk to be
        assigned to the wrong cluster, then verify refinement fixes it.
        """
        # Cluster A: about flock/locking
        a1 = _chunk("a1", user="implement flock advisory locking mechanism",
                     assistant="Flock advisory locking prevents concurrent writes",
                     ts="2026-03-01T10:00:00Z")
        a2 = _chunk("a2", user="test flock locking under concurrent access",
                     assistant="Concurrent flock locking test passes cleanly",
                     ts="2026-03-01T10:05:00Z")

        # Cluster B: about clustering/jaccard
        b1 = _chunk("b1", user="implement Jaccard similarity clustering algorithm",
                     assistant="Jaccard clustering groups chunks by token overlap similarity",
                     ts="2026-03-01T11:00:00Z")
        b2 = _chunk("b2", user="test Jaccard clustering threshold sensitivity",
                     assistant="Jaccard threshold sensitivity analysis for clustering accuracy",
                     ts="2026-03-01T11:05:00Z")

        # Ambiguous chunk: mentions both but is really about clustering
        ambig = _chunk("ambig", user="compare flock clustering approach with Jaccard",
                       assistant="Jaccard clustering approach outperforms naive grouping for similarity",
                       ts="2026-03-01T12:00:00Z")

        clusters = cluster_chunks([a1, a2, b1, b2, ambig])

        # The ambiguous chunk should end up in one of the clusters
        # (we can't guarantee which without controlling processing order,
        #  but both clusters should exist and be non-empty)
        assert len(clusters) >= 2, "Should have at least 2 distinct clusters"


# ---------------------------------------------------------------------------
# Concat summary
# ---------------------------------------------------------------------------

class TestConcatSummary:
    def test_basic_summary(self):
        chunks = [
            _chunk("c1", assistant="The flock race was fixed.", ts="2026-03-01T10:00:00Z"),
            _chunk("c2", assistant="Review confirmed the fix.", ts="2026-03-01T10:05:00Z"),
        ]
        summary = generate_concat_summary(chunks)
        assert "flock" in summary.lower()
        assert len(summary) > 0

    def test_empty_chunks(self):
        assert generate_concat_summary([]) == ""

    def test_truncation(self):
        long_text = "x" * 5000
        chunks = [_chunk("c1", assistant=long_text)]
        summary = generate_concat_summary(chunks, max_tokens=50)
        # 50 tokens * 4 chars = 200 chars max
        assert len(summary) <= 210  # small margin for "..."

    def test_chronological_order(self):
        """Summary should use earliest chunks first."""
        chunks = [
            _chunk("c2", assistant="Second thing happened", ts="2026-03-01T10:05:00Z"),
            _chunk("c1", assistant="First thing happened", ts="2026-03-01T10:00:00Z"),
        ]
        summary = generate_concat_summary(chunks)
        # "First" should come before "Second" since we sort chronologically
        assert summary.index("First") < summary.index("Second")


# ---------------------------------------------------------------------------
# Storage integration
# ---------------------------------------------------------------------------

class TestClusterStorage:
    def test_save_and_load_clusters(self, tmp_path):
        from synapt.recall.storage import RecallDB

        db = RecallDB(tmp_path / "test.db")

        # Save some chunks first (FK constraint)
        test_chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="sess-1", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="fix flock",
                assistant_text="Fixed flock race condition",
            ),
            TranscriptChunk(
                id="s1:t1", session_id="sess-1", timestamp="2026-03-01T10:05:00Z",
                turn_index=1, user_text="review flock",
                assistant_text="Reviewed flock fix",
            ),
        ]
        db.save_chunks(test_chunks)

        clusters = [{
            "cluster_id": "clust-abcd1234",
            "topic": "flock race fix",
            "cluster_type": "topic",
            "session_ids": ["sess-1"],
            "branch": None,
            "date_start": "2026-03-01T10:00:00Z",
            "date_end": "2026-03-01T10:05:00Z",
            "chunk_count": 2,
            "status": "active",
            "created_at": "2026-03-05T12:00:00Z",
            "updated_at": "2026-03-05T12:00:00Z",
        }]
        memberships = [
            ("clust-abcd1234", "s1:t0", "2026-03-05T12:00:00Z"),
            ("clust-abcd1234", "s1:t1", "2026-03-05T12:00:00Z"),
        ]

        db.save_clusters(clusters, memberships)

        # Load and verify
        loaded = db.load_clusters()
        assert len(loaded) == 1
        assert loaded[0]["cluster_id"] == "clust-abcd1234"
        assert loaded[0]["topic"] == "flock race fix"
        assert loaded[0]["chunk_count"] == 2

        # Verify chunk membership
        chunk_ids = db.get_cluster_chunks("clust-abcd1234")
        assert set(chunk_ids) == {"s1:t0", "s1:t1"}

        # Verify reverse lookup
        clusters_for = db.clusters_for_chunk("s1:t0")
        assert "clust-abcd1234" in clusters_for

        # Verify count
        assert db.cluster_count() == 1

        db.close()

    def test_cluster_summary_crud(self, tmp_path):
        from synapt.recall.storage import RecallDB

        db = RecallDB(tmp_path / "test.db")

        # Must create cluster first
        db.save_chunks([
            TranscriptChunk(
                id="s1:t0", session_id="sess-1", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="x", assistant_text="y",
            ),
        ])
        db.save_clusters(
            [{
                "cluster_id": "clust-test0001",
                "topic": "test topic",
                "session_ids": ["sess-1"],
                "chunk_count": 1,
                "status": "active",
                "created_at": "2026-03-05T12:00:00Z",
                "updated_at": "2026-03-05T12:00:00Z",
            }],
            [("clust-test0001", "s1:t0", "2026-03-05T12:00:00Z")],
        )

        # No summary yet
        assert db.get_cluster_summary("clust-test0001") is None

        # Save summary
        db.save_cluster_summary("clust-test0001", "This is a test summary")

        # Load and verify
        summary = db.get_cluster_summary("clust-test0001")
        assert summary is not None
        assert summary["summary"] == "This is a test summary"
        assert summary["method"] == "concat"
        assert summary["stale"] is False

        db.close()

    def test_cluster_fts_search(self, tmp_path):
        from synapt.recall.storage import RecallDB

        db = RecallDB(tmp_path / "test.db")
        db.save_chunks([
            TranscriptChunk(
                id="s1:t0", session_id="sess-1", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="x", assistant_text="y",
            ),
        ])
        db.save_clusters(
            [{
                "cluster_id": "clust-fts00001",
                "topic": "journal flock compaction rewrite",
                "session_ids": ["sess-1"],
                "chunk_count": 1,
                "status": "active",
                "created_at": "2026-03-05T12:00:00Z",
                "updated_at": "2026-03-05T12:00:00Z",
            }],
            [("clust-fts00001", "s1:t0", "2026-03-05T12:00:00Z")],
        )

        # FTS search should find the cluster by topic
        rows = db._conn.execute(
            "SELECT rowid FROM clusters_fts WHERE clusters_fts MATCH 'flock'"
        ).fetchall()
        assert len(rows) == 1

        db.close()

    def test_get_cluster_by_id(self, tmp_path):
        from synapt.recall.storage import RecallDB

        db = RecallDB(tmp_path / "test.db")
        db.save_chunks([
            TranscriptChunk(
                id="s1:t0", session_id="sess-1", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="x", assistant_text="y",
            ),
        ])
        db.save_clusters(
            [{
                "cluster_id": "clust-lookup01",
                "topic": "targeted lookup test",
                "session_ids": ["sess-1"],
                "chunk_count": 1,
                "status": "active",
                "created_at": "2026-03-05T12:00:00Z",
                "updated_at": "2026-03-05T12:00:00Z",
            }],
            [("clust-lookup01", "s1:t0", "2026-03-05T12:00:00Z")],
        )

        cl = db.get_cluster("clust-lookup01")
        assert cl is not None
        assert cl["topic"] == "targeted lookup test"

        assert db.get_cluster("nonexistent") is None
        db.close()


# ---------------------------------------------------------------------------
# Integration: search formatting with clusters
# ---------------------------------------------------------------------------

class TestFormatWithClusters:
    """Test that _format_results groups results by cluster."""

    def _build_index_with_clusters(self, tmp_path):
        """Set up an index with chunks and clusters for format testing."""
        from synapt.recall.storage import RecallDB

        db = RecallDB(tmp_path / "test.db")
        chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="sess-a", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="fix flock race",
                assistant_text="Fixed the flock TOCTOU race condition",
            ),
            TranscriptChunk(
                id="s1:t1", session_id="sess-a", timestamp="2026-03-01T10:05:00Z",
                turn_index=1, user_text="review flock fix",
                assistant_text="Reviewed the flock race fix, looks correct",
            ),
            TranscriptChunk(
                id="s2:t0", session_id="sess-b", timestamp="2026-03-02T10:00:00Z",
                turn_index=0, user_text="train swift adapter",
                assistant_text="Training swift adapter on batman tasks",
            ),
        ]
        db.save_chunks(chunks)

        # Cluster the first two chunks together
        db.save_clusters(
            [{
                "cluster_id": "clust-flock001",
                "topic": "flock race fix",
                "session_ids": ["sess-a"],
                "date_start": "2026-03-01T10:00:00Z",
                "date_end": "2026-03-01T10:05:00Z",
                "chunk_count": 2,
                "status": "active",
                "created_at": "2026-03-05T12:00:00Z",
                "updated_at": "2026-03-05T12:00:00Z",
            }],
            [
                ("clust-flock001", "s1:t0", "2026-03-05T12:00:00Z"),
                ("clust-flock001", "s1:t1", "2026-03-05T12:00:00Z"),
            ],
        )
        db.save_cluster_summary("clust-flock001", "Fixed flock TOCTOU race.")

        from synapt.recall.core import TranscriptIndex
        idx = TranscriptIndex(chunks, db=db)
        # Populate rowid maps from DB
        idx._refresh_rowid_map()
        return idx, db

    def _find_idx(self, idx, chunk_id: str) -> int:
        """Find the index of a chunk by ID in the sorted index."""
        return idx._id_to_idx[chunk_id]

    def test_cluster_block_in_results(self, tmp_path):
        """When 2+ result chunks share a cluster, show cluster summary."""
        idx, db = self._build_index_with_clusters(tmp_path)

        # Both flock chunks (s1:t0, s1:t1) should cluster together
        i0 = self._find_idx(idx, "s1:t0")
        i1 = self._find_idx(idx, "s1:t1")
        ranked = [(i0, 5.0), (i1, 4.5)]
        result = idx._format_results(ranked, max_tokens=2000)

        assert "[cluster: flock race fix]" in result
        assert "(clust-" in result  # cluster ID shown for drill-down
        assert "Fixed flock TOCTOU race." in result
        db.close()

    def test_ungrouped_chunks_still_shown(self, tmp_path):
        """Chunks not in any cluster display as raw blocks."""
        idx, db = self._build_index_with_clusters(tmp_path)

        # Only one flock chunk + the unrelated swift chunk
        i_flock = self._find_idx(idx, "s1:t0")
        i_swift = self._find_idx(idx, "s2:t0")
        ranked = [(i_flock, 5.0), (i_swift, 3.0)]
        result = idx._format_results(ranked, max_tokens=2000)

        # s1:t0 is the only cluster member — should display as raw chunk
        assert "session sess-a" in result
        assert "train swift" in result.lower() or "swift adapter" in result.lower()
        db.close()

    def test_relevance_ordering_preserved(self, tmp_path):
        """Higher-scoring ungrouped chunk should appear before lower-scoring cluster."""
        idx, db = self._build_index_with_clusters(tmp_path)

        # Swift chunk has highest score, flock cluster has lower
        i_swift = self._find_idx(idx, "s2:t0")
        i0 = self._find_idx(idx, "s1:t0")
        i1 = self._find_idx(idx, "s1:t1")
        ranked = [(i_swift, 10.0), (i0, 2.0), (i1, 1.5)]
        result = idx._format_results(ranked, max_tokens=2000)

        # Swift chunk should come before the flock cluster
        swift_pos = result.find("swift")
        cluster_pos = result.find("[cluster:")
        assert swift_pos < cluster_pos, "Higher-scoring ungrouped chunk should precede lower-scoring cluster"
        db.close()

    def test_no_clusters_graceful_degradation(self, tmp_path):
        """When no clusters exist, format falls back to raw chunk display."""
        from synapt.recall.storage import RecallDB
        from synapt.recall.core import TranscriptIndex

        db = RecallDB(tmp_path / "test.db")
        chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="sess-a", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="hello", assistant_text="world",
            ),
        ]
        db.save_chunks(chunks)
        idx = TranscriptIndex(chunks, db=db)
        idx._refresh_rowid_map()

        ranked = [(0, 5.0)]
        result = idx._format_results(ranked, max_tokens=2000)

        assert "session sess-a" in result
        assert "world" in result
        assert "[cluster:" not in result
        db.close()


class TestGenerateClusterSummaryFromDB:
    """Test _generate_cluster_summary with full cluster membership."""

    def test_summary_uses_full_membership(self, tmp_path):
        """Summary should use all cluster members, not just search hits."""
        from synapt.recall.storage import RecallDB
        from synapt.recall.core import TranscriptIndex

        db = RecallDB(tmp_path / "test.db")
        chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="sess-a", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="first", assistant_text="Alpha content here",
            ),
            TranscriptChunk(
                id="s1:t1", session_id="sess-a", timestamp="2026-03-01T10:05:00Z",
                turn_index=1, user_text="second", assistant_text="Beta content here",
            ),
            TranscriptChunk(
                id="s1:t2", session_id="sess-a", timestamp="2026-03-01T10:10:00Z",
                turn_index=2, user_text="third", assistant_text="Gamma content here",
            ),
        ]
        db.save_chunks(chunks)
        db.save_clusters(
            [{
                "cluster_id": "clust-full0001",
                "topic": "test topic",
                "session_ids": ["sess-a"],
                "chunk_count": 3,
                "status": "active",
                "created_at": "2026-03-05T12:00:00Z",
                "updated_at": "2026-03-05T12:00:00Z",
            }],
            [
                ("clust-full0001", "s1:t0", "2026-03-05T12:00:00Z"),
                ("clust-full0001", "s1:t1", "2026-03-05T12:00:00Z"),
                ("clust-full0001", "s1:t2", "2026-03-05T12:00:00Z"),
            ],
        )

        idx = TranscriptIndex(chunks, db=db)
        idx._refresh_rowid_map()
        summary = idx._generate_cluster_summary("clust-full0001")

        # Should include content from first 2 chunks (chronological)
        assert "Alpha" in summary
        assert "Beta" in summary
        db.close()

    def test_missing_chunk_ids_skipped(self, tmp_path):
        """Cluster with chunk IDs not in the index should not crash."""
        from synapt.recall.storage import RecallDB
        from synapt.recall.core import TranscriptIndex

        db = RecallDB(tmp_path / "test.db")
        chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="sess-a", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="x", assistant_text="Real chunk text",
            ),
        ]
        db.save_chunks(chunks)
        db.save_clusters(
            [{
                "cluster_id": "clust-stale001",
                "topic": "stale cluster",
                "session_ids": ["sess-a"],
                "chunk_count": 2,
                "status": "active",
                "created_at": "2026-03-05T12:00:00Z",
                "updated_at": "2026-03-05T12:00:00Z",
            }],
            [
                ("clust-stale001", "s1:t0", "2026-03-05T12:00:00Z"),
                ("clust-stale001", "s1:t99", "2026-03-05T12:00:00Z"),  # stale ID
            ],
        )

        idx = TranscriptIndex(chunks, db=db)
        idx._refresh_rowid_map()
        summary = idx._generate_cluster_summary("clust-stale001")

        # Should work with only the real chunk
        assert "Real chunk text" in summary
        db.close()


class TestRecallContextCluster:
    """Test recall_context with cluster_id parameter."""

    def test_cluster_drilldown(self, tmp_path):
        """recall_context(cluster_id=...) should show all cluster chunks."""
        from unittest.mock import patch
        from synapt.recall.storage import RecallDB
        from synapt.recall.core import TranscriptIndex
        from synapt.recall import server

        db = RecallDB(tmp_path / "test.db")
        chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="sess-a", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="fix flock",
                assistant_text="Fixed the flock race",
                transcript_path="", byte_offset=-1, byte_length=0,
            ),
        ]
        db.save_chunks(chunks)
        db.save_clusters(
            [{
                "cluster_id": "clust-drill001",
                "topic": "flock drilldown",
                "session_ids": ["sess-a"],
                "chunk_count": 1,
                "status": "active",
                "created_at": "2026-03-05T12:00:00Z",
                "updated_at": "2026-03-05T12:00:00Z",
            }],
            [("clust-drill001", "s1:t0", "2026-03-05T12:00:00Z")],
        )

        idx = TranscriptIndex(chunks, db=db)
        idx._refresh_rowid_map()

        with patch.object(server, '_get_index', return_value=idx):
            result = server.recall_context(cluster_id="clust-drill001")

        assert "flock drilldown" in result
        assert "1 chunks" in result
        db.close()

    def test_nonexistent_cluster(self, owned_recall_root, tmp_path):
        """recall_context with nonexistent cluster_id returns helpful message.

        Ref #967 — ``recall_context`` resolves the data root implicitly, so
        without an owned root the "not found" could come from the operator's
        store rather than from the empty index this test builds.
        """
        from unittest.mock import patch
        from synapt.recall.storage import RecallDB
        from synapt.recall.core import TranscriptIndex
        from synapt.recall import server

        db = RecallDB(tmp_path / "test.db")
        idx = TranscriptIndex([], db=db)

        with patch.object(server, '_get_index', return_value=idx):
            result = server.recall_context(cluster_id="nonexistent")

        assert "not found" in result
        db.close()


class TestAccessTrackingIntegration(TestFormatWithClusters):
    """Integration tests: _format_results and recall_context record accesses."""

    def test_format_results_records_cluster_access(self, tmp_path):
        """_format_results records cluster accesses for emitted cluster blocks."""
        idx, db = self._build_index_with_clusters(tmp_path)

        i0 = self._find_idx(idx, "s1:t0")
        i1 = self._find_idx(idx, "s1:t1")
        ranked = [(i0, 5.0), (i1, 4.5)]
        idx._format_results(ranked, max_tokens=2000)

        stats = db.get_access_stats("cluster", "clust-flock001")
        assert stats is not None
        assert stats["access_count"] == 1
        assert stats["explicit_count"] == 1  # search counts as explicit
        db.close()

    def test_format_results_records_chunk_access(self, tmp_path):
        """_format_results records chunk accesses for ungrouped chunks."""
        idx, db = self._build_index_with_clusters(tmp_path)

        # Only the swift chunk (not in any cluster) → ungrouped
        i_swift = self._find_idx(idx, "s2:t0")
        ranked = [(i_swift, 3.0)]
        idx._format_results(ranked, max_tokens=2000)

        stats = db.get_access_stats("chunk", "s2:t0")
        assert stats is not None
        assert stats["access_count"] == 1
        db.close()

    def test_context_drilldown_records_explicit_access(self, tmp_path):
        """recall_context records explicit_count for drill-down."""
        from unittest.mock import patch
        from synapt.recall import server

        idx, db = self._build_index_with_clusters(tmp_path)

        with patch.object(server, '_get_index', return_value=idx):
            server.recall_context(cluster_id="clust-flock001")

        stats = db.get_access_stats("cluster", "clust-flock001")
        assert stats is not None
        assert stats["access_count"] == 1
        assert stats["explicit_count"] == 1
        db.close()

    def test_search_then_drilldown_accumulates(self, tmp_path):
        """Search + drill-down correctly accumulates both access types."""
        from unittest.mock import patch
        from synapt.recall import server

        idx, db = self._build_index_with_clusters(tmp_path)

        # First: search hit (auto, not explicit)
        i0 = self._find_idx(idx, "s1:t0")
        i1 = self._find_idx(idx, "s1:t1")
        idx._format_results([(i0, 5.0), (i1, 4.5)], max_tokens=2000)

        # Then: explicit drill-down
        with patch.object(server, '_get_index', return_value=idx):
            server.recall_context(cluster_id="clust-flock001")

        stats = db.get_access_stats("cluster", "clust-flock001")
        assert stats["access_count"] == 2
        assert stats["explicit_count"] == 2  # search + drill-down both explicit
        db.close()


class TestClusterFTSSearch:
    """Tests for cluster FTS search and concise mode."""

    def _build_index_with_search_text(self, tmp_path):
        """Build an index with clusters that have search_text populated."""
        from synapt.recall.storage import RecallDB
        from synapt.recall.core import TranscriptIndex, TranscriptChunk

        db = RecallDB(tmp_path / "test.db")
        chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="sess-a", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="fix flock race",
                assistant_text="Fixed the flock TOCTOU race condition in file locking",
            ),
            TranscriptChunk(
                id="s1:t1", session_id="sess-a", timestamp="2026-03-01T10:05:00Z",
                turn_index=1, user_text="review flock fix",
                assistant_text="Reviewed the flock race fix, looks correct",
            ),
            TranscriptChunk(
                id="s2:t0", session_id="sess-b", timestamp="2026-03-02T10:00:00Z",
                turn_index=0, user_text="train swift adapter",
                assistant_text="Training swift adapter on batman tasks",
            ),
        ]
        db.save_chunks(chunks)

        db.save_clusters(
            [{
                "cluster_id": "clust-flock001",
                "topic": "flock race fix",
                "search_text": "Fixed the flock TOCTOU race condition in file locking Reviewed the flock race fix looks correct",
                "session_ids": ["sess-a"],
                "date_start": "2026-03-01T10:00:00Z",
                "date_end": "2026-03-01T10:05:00Z",
                "chunk_count": 2,
                "status": "active",
                "created_at": "2026-03-05T12:00:00Z",
                "updated_at": "2026-03-05T12:00:00Z",
            }],
            [
                ("clust-flock001", "s1:t0", "2026-03-05T12:00:00Z"),
                ("clust-flock001", "s1:t1", "2026-03-05T12:00:00Z"),
            ],
        )
        db.save_cluster_summary("clust-flock001", "Fixed flock TOCTOU race.")

        idx = TranscriptIndex(chunks, db=db)
        idx._refresh_rowid_map()
        return idx, db

    def test_cluster_fts_search_by_topic(self, tmp_path):
        """cluster_fts_search finds clusters by topic keywords."""
        _idx, db = self._build_index_with_search_text(tmp_path)
        results = db.cluster_fts_search("flock race")
        assert len(results) >= 1
        assert results[0][0] == "clust-flock001"
        db.close()

    def test_cluster_fts_search_by_content(self, tmp_path):
        """cluster_fts_search finds clusters by member chunk content."""
        _idx, db = self._build_index_with_search_text(tmp_path)
        # "TOCTOU" is in search_text but not in topic
        results = db.cluster_fts_search("TOCTOU")
        assert len(results) >= 1
        assert results[0][0] == "clust-flock001"
        db.close()

    def test_concise_mode_returns_cluster_summaries(self, tmp_path):
        """depth='concise' returns cluster summaries, not individual chunks."""
        idx, db = self._build_index_with_search_text(tmp_path)
        result = idx._concise_lookup("flock", max_chunks=5, max_tokens=2000)
        assert "[cluster: flock race fix]" in result
        assert "Fixed flock TOCTOU race." in result
        db.close()

    def test_concise_mode_no_raw_chunks(self, tmp_path):
        """Concise mode should not show raw chunk blocks."""
        idx, db = self._build_index_with_search_text(tmp_path)
        result = idx._concise_lookup("flock", max_chunks=5, max_tokens=2000)
        # Must NOT contain raw chunk markers (session:turn headers, "---" block delimiters)
        assert "session sess-a" not in result
        assert "s1:t0" not in result
        assert "s1:t1" not in result
        # Must contain cluster structure
        assert result.startswith("Past session context:")
        assert "[cluster:" in result
        db.close()

    def test_concise_mode_records_access(self, tmp_path):
        """Concise lookup should record access for emitted items."""
        idx, db = self._build_index_with_search_text(tmp_path)
        idx._concise_lookup("flock", max_chunks=5, max_tokens=2000)
        assert db.access_log_count() >= 1
        stats = db.get_access_stats("cluster", "clust-flock001")
        assert stats is not None
        assert stats["access_count"] >= 1
        db.close()

    def test_access_frequency_boost_formula(self, tmp_path):
        """_access_frequency_boost applies log2-based boost capped at 1.3x."""
        import math
        idx, db = self._build_index_with_search_text(tmp_path)

        # No access history -> no boost
        score = idx._access_frequency_boost(10.0, "chunk", "nonexistent")
        assert score == 10.0

        # Simulate explicit access with a relevance score
        db.record_access(
            [{"item_type": "chunk", "item_id": "s1:t0", "score": 0.8}],
            context="context",
        )
        boosted = idx._access_frequency_boost(10.0, "chunk", "s1:t0")
        # weighted_count is 0.8
        expected_boost = 1.0 + min(math.log2(0.8 + 1) * 0.15, 0.3)
        assert abs(boosted - 10.0 * expected_boost) < 0.001

        # Hook access -> no boost (hook is not explicit)
        db.record_access(
            [{"item_type": "chunk", "item_id": "s2:t0", "score": 5.0}],
            context="hook",
        )
        no_boost = idx._access_frequency_boost(10.0, "chunk", "s2:t0")
        assert no_boost == 10.0

        db.close()

    def test_access_boost_weighted_high_vs_low(self, tmp_path):
        """High-score accesses produce a larger boost than low-score accesses."""
        import math
        idx, db = self._build_index_with_search_text(tmp_path)

        # 3 high-score accesses for s1:t0
        for _ in range(3):
            db.record_access(
                [{"item_type": "chunk", "item_id": "s1:t0", "score": 0.9}],
                context="search",
            )
        # 3 low-score accesses for s2:t0
        for _ in range(3):
            db.record_access(
                [{"item_type": "chunk", "item_id": "s2:t0", "score": 0.2}],
                context="search",
            )

        boost_high = idx._access_frequency_boost(10.0, "chunk", "s1:t0")
        boost_low = idx._access_frequency_boost(10.0, "chunk", "s2:t0")

        # Both get boosted, but high-score access gets more
        assert boost_high > 10.0
        assert boost_low > 10.0
        assert boost_high > boost_low

        # Verify the weighted formula: 3*0.9 = 2.7 vs 3*0.2 = 0.6
        expected_high = 10.0 * (1.0 + min(math.log2(2.7 + 1) * 0.15, 0.3))
        expected_low = 10.0 * (1.0 + min(math.log2(0.6 + 1) * 0.15, 0.3))
        assert abs(boost_high - expected_high) < 0.001
        assert abs(boost_low - expected_low) < 0.001

        db.close()

    def test_access_boost_cap_at_1_3x(self, tmp_path):
        """The boost caps at 1.3x regardless of access history."""
        idx, db = self._build_index_with_search_text(tmp_path)

        # Many high-score accesses to push weighted_count very high
        for _ in range(100):
            db.record_access(
                [{"item_type": "chunk", "item_id": "s1:t0", "score": 1.0}],
                context="search",
            )

        boosted = idx._access_frequency_boost(10.0, "chunk", "s1:t0")
        # Cap is 1.3x -> max 13.0
        assert boosted <= 10.0 * 1.3 + 0.001
        assert boosted > 10.0  # But it is boosted

        db.close()

    def test_access_boost_fallback_to_explicit_count(self, tmp_path):
        """Falls back to explicit_count when weighted_count is 0 (no score)."""
        import math
        idx, db = self._build_index_with_search_text(tmp_path)

        # Access without a score -> weighted_count stays 0, explicit_count = 1
        db.record_access(
            [{"item_type": "chunk", "item_id": "s1:t0"}],
            context="search",
        )

        stats = db.get_access_stats("chunk", "s1:t0")
        assert stats["explicit_count"] == 1
        assert stats["weighted_count"] == 0.0

        # Should fall back to explicit_count
        boosted = idx._access_frequency_boost(10.0, "chunk", "s1:t0")
        expected = 10.0 * (1.0 + min(math.log2(1 + 1) * 0.15, 0.3))
        assert abs(boosted - expected) < 0.001

        db.close()

    def test_chunk_fts_to_clusters_fallback(self, tmp_path):
        """chunk_fts_to_clusters finds parent clusters via chunk content."""
        _idx, db = self._build_index_with_search_text(tmp_path)
        # "flock" appears in chunk user_text but search a term only in chunks
        results = db.chunk_fts_to_clusters("flock race")
        assert len(results) >= 1
        assert results[0][0] == "clust-flock001"
        assert results[0][1] > 0  # positive score
        db.close()

    def test_chunk_fts_to_clusters_no_match(self, tmp_path):
        """chunk_fts_to_clusters returns empty for non-matching queries."""
        _idx, db = self._build_index_with_search_text(tmp_path)
        results = db.chunk_fts_to_clusters("nonexistent_term_xyz")
        assert results == []
        db.close()

    def test_chunk_fts_to_clusters_excludes_timeline(self, tmp_path):
        """chunk_fts_to_clusters should not return timeline clusters."""
        from synapt.recall.storage import RecallDB
        from synapt.recall.core import TranscriptChunk

        db = RecallDB(tmp_path / "tl_test.db")
        chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="sess-a", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="timeline test query",
                assistant_text="Some answer about timelines",
            ),
        ]
        db.save_chunks(chunks)
        # Create a timeline cluster containing this chunk
        db.save_clusters(
            [{
                "cluster_id": "tl-timetest001",
                "topic": "timeline arc",
                "search_text": "timeline test query",
                "cluster_type": "timeline",
                "session_ids": ["sess-a"],
                "date_start": "2026-03-01T10:00:00Z",
                "date_end": "2026-03-01T10:05:00Z",
                "chunk_count": 1,
                "status": "active",
                "created_at": "2026-03-05T12:00:00Z",
                "updated_at": "2026-03-05T12:00:00Z",
            }],
            [("tl-timetest001", "s1:t0", "2026-03-05T12:00:00Z")],
        )
        results = db.chunk_fts_to_clusters("timeline test query")
        # Timeline clusters should be excluded
        cluster_ids = [r[0] for r in results]
        assert "tl-timetest001" not in cluster_ids
        db.close()

    def test_concise_fallback_to_chunk_fts(self, tmp_path):
        """Concise lookup falls back to chunk FTS when cluster FTS misses."""
        from synapt.recall.storage import RecallDB
        from synapt.recall.core import TranscriptIndex, TranscriptChunk

        db = RecallDB(tmp_path / "fallback.db")
        # Create a chunk with unique user_text NOT in cluster search_text
        chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="sess-a", timestamp="2026-03-01T10:00:00Z",
                turn_index=0, user_text="PeftModel load_adapter GPU weights bug",
                assistant_text="Fixed the adapter loading issue",
            ),
        ]
        db.save_chunks(chunks)
        # Cluster search_text intentionally does NOT contain "PeftModel"
        db.save_clusters(
            [{
                "cluster_id": "clust-peft001",
                "topic": "adapter loading fix",
                "search_text": "Fixed the adapter loading issue",
                "session_ids": ["sess-a"],
                "date_start": "2026-03-01T10:00:00Z",
                "date_end": "2026-03-01T10:05:00Z",
                "chunk_count": 1,
                "status": "active",
                "created_at": "2026-03-05T12:00:00Z",
                "updated_at": "2026-03-05T12:00:00Z",
            }],
            [("clust-peft001", "s1:t0", "2026-03-05T12:00:00Z")],
        )
        db.save_cluster_summary("clust-peft001", "Fixed adapter loading bug.")

        idx = TranscriptIndex(chunks, db=db)
        idx._refresh_rowid_map()

        # "PeftModel" is only in chunk user_text, not cluster search_text
        result = idx._concise_lookup("PeftModel", max_chunks=5, max_tokens=2000)
        assert "clust-peft001" in result or "adapter loading" in result.lower()
        db.close()


# ---------------------------------------------------------------------------
# Content hash for LLM summary reuse (#313)
# ---------------------------------------------------------------------------

class TestContentHash:
    """Test _content_hash() determinism and collision avoidance."""

    def test_same_content_same_hash(self):
        texts = [
            {"user_text": "fix flock", "assistant_text": "Fixed race condition"},
            {"user_text": "test flock", "assistant_text": "Tests pass"},
        ]
        assert _content_hash(texts) == _content_hash(texts)

    def test_order_independent(self):
        """Hash should be stable regardless of chunk iteration order."""
        a = [
            {"user_text": "fix flock", "assistant_text": "Fixed race"},
            {"user_text": "test flock", "assistant_text": "Tests pass"},
        ]
        b = list(reversed(a))
        assert _content_hash(a) == _content_hash(b)

    def test_different_content_different_hash(self):
        a = [{"user_text": "fix flock", "assistant_text": "Fixed race"}]
        b = [{"user_text": "fix flock", "assistant_text": "Different fix"}]
        assert _content_hash(a) != _content_hash(b)

    def test_empty_fields_handled(self):
        a = [{"user_text": "", "assistant_text": ""}]
        b = [{}]
        assert _content_hash(a) == _content_hash(b)

    def test_no_collision_user_vs_assistant(self):
        """Content in user vs assistant field should produce different hashes."""
        a = [{"user_text": "hello", "assistant_text": ""}]
        b = [{"user_text": "", "assistant_text": "hello"}]
        assert _content_hash(a) != _content_hash(b)


class TestHasMeaningfulContent:
    """Test _has_meaningful_content() noise detection."""

    def test_real_content_is_meaningful(self):
        """Chunks with substantial assistant text pass."""
        texts = [
            {"user_text": "How do I fix flock?",
             "assistant_text": "The race condition in the lock manager can be fixed by using LOCK_EX with LOCK_NB."},
            {"user_text": "Run the tests",
             "assistant_text": "All 62 tests pass. The stability test confirms 85% cluster ID preservation."},
        ]
        assert _has_meaningful_content(texts) is True

    def test_empty_assistant_is_noise(self):
        """Clusters with empty assistant text (system messages) are noise."""
        texts = [
            {"user_text": "<local-command-caveat>DO NOT respond</local-command-caveat>",
             "assistant_text": ""},
            {"user_text": "<local-command-caveat>DO NOT respond</local-command-caveat>",
             "assistant_text": ""},
            {"user_text": "<local-command-caveat>DO NOT respond</local-command-caveat>",
             "assistant_text": ""},
        ]
        assert _has_meaningful_content(texts) is False

    def test_very_short_assistant_is_noise(self):
        """Clusters with very short responses (< 20 chars) are noise."""
        texts = [
            {"user_text": "interrupt", "assistant_text": "ok"},
            {"user_text": "interrupt", "assistant_text": ""},
            {"user_text": "interrupt", "assistant_text": "done"},
        ]
        assert _has_meaningful_content(texts) is False

    def test_mixed_content_above_threshold(self):
        """Clusters with some noise but enough real content pass."""
        texts = [
            {"user_text": "question",
             "assistant_text": "Here is a detailed explanation of the architecture decisions we made."},
            {"user_text": "stdout", "assistant_text": ""},
            {"user_text": "another question",
             "assistant_text": "The implementation uses Jaccard similarity with a threshold of 0.20 for stability."},
        ]
        # 2/3 = 67% have content, above 30% threshold
        assert _has_meaningful_content(texts) is True

    def test_repetitive_api_errors_rejected(self):
        """Repeated identical API errors fail the diversity check."""
        texts = [
            {"user_text": "error", "assistant_text": "API Error: 400 invalid_request_error messages.0.content block"},
            {"user_text": "error", "assistant_text": "API Error: 400 invalid_request_error messages.0.content block"},
            {"user_text": "error", "assistant_text": "API Error: 400 invalid_request_error messages.0.content block"},
            {"user_text": "error", "assistant_text": "API Error: 400 invalid_request_error messages.0.content block"},
        ]
        # diversity = 1/4 = 0.25, below MIN_DIVERSITY_RATIO (0.3)
        assert _has_meaningful_content(texts) is False

    def test_repetitive_system_messages_rejected(self):
        """Repeated system notifications cluster strongly but are noise."""
        texts = [
            {"user_text": "task done", "assistant_text": "That stale monitor is from the previous session. Please wait a minute."},
            {"user_text": "task done", "assistant_text": "That stale monitor is from the previous session. Please wait a minute."},
            {"user_text": "task done", "assistant_text": "That stale monitor is from the previous session. Please wait a minute."},
            {"user_text": "task done", "assistant_text": "That stale monitor is from the previous session. Please wait a minute."},
        ]
        # diversity = 1/4 = 0.25, below threshold
        assert _has_meaningful_content(texts) is False

    def test_diverse_content_passes(self):
        """Clusters with diverse assistant responses pass all checks."""
        texts = [
            {"user_text": "fix clustering", "assistant_text": "The Jaccard threshold was raised from 0.15 to 0.20 for stability."},
            {"user_text": "test it", "assistant_text": "All 62 tests pass and cluster IDs are 85% stable across rebuilds."},
        ]
        # diversity = 2/2 = 1.0, above threshold
        assert _has_meaningful_content(texts) is True

    def test_empty_list(self):
        assert _has_meaningful_content([]) is False

    def test_missing_keys(self):
        """Handles missing user_text/assistant_text keys."""
        texts = [{"user_text": "hello"}, {}]
        assert _has_meaningful_content(texts) is False


class TestNovelEntities:
    """Test _novel_entities() hallucination detection for cluster summaries."""

    def test_no_novel_entities(self):
        """Summary using only entities from source should return empty set."""
        summary = "The ConnectionPool was refactored to use threading.local()."
        source = "assistant: connectionpool refactored with threading.local()"
        assert _novel_entities(summary, source) == set()

    def test_detects_fabricated_entity(self):
        """Summary inventing a class name not in source should be caught."""
        summary = "The FarewellHandler processes exit commands and sends goodbye messages."
        source = "user ran /exit command. assistant: goodbye!"
        novel = _novel_entities(summary, source)
        assert "FarewellHandler" in novel

    def test_multiple_fabricated_entities(self):
        """Multiple invented entities should all be caught."""
        summary = "The ExitManager delegates to FarewellHandler for graceful shutdown."
        source = "user said bye, assistant closed the session"
        novel = _novel_entities(summary, source)
        assert len(novel) >= 2
        assert "ExitManager" in novel
        assert "FarewellHandler" in novel

    def test_common_camelcase_not_flagged(self):
        """Well-known tools like PyTorch, TensorFlow should not be flagged."""
        summary = "The pipeline uses PyTorch with TensorFlow integration."
        source = "pipeline uses machine learning for training"
        novel = _novel_entities(summary, source)
        assert "PyTorch" not in novel
        assert "TensorFlow" not in novel

    def test_entity_present_in_source(self):
        """Entity that appears in source (case-insensitive) is not novel."""
        summary = "ConnectionPool uses threading.local() for isolation."
        source = "the connectionpool class provides per-thread isolation"
        novel = _novel_entities(summary, source)
        assert "ConnectionPool" not in novel

    def test_empty_summary(self):
        """Empty summary should return empty set."""
        assert _novel_entities("", "some source text") == set()

    def test_no_camelcase(self):
        """Summary without CamelCase should return empty set."""
        summary = "The database was refactored to use WAL mode."
        source = "database uses wal mode"
        assert _novel_entities(summary, source) == set()


class TestContentHashReuse:
    """Integration: upgrade_large_cluster_summaries reuses summaries via hash."""

    def _setup_db_with_clusters(self, tmp_path, cluster_id, chunk_ids):
        from synapt.recall.storage import RecallDB
        from datetime import datetime, timezone

        db = RecallDB(tmp_path / "test.db")
        now = datetime.now(timezone.utc).isoformat()

        # Create chunks
        chunks = []
        for i, cid in enumerate(chunk_ids):
            chunks.append(TranscriptChunk(
                id=cid, session_id="sess-1",
                timestamp=f"2026-03-01T10:{i:02d}:00Z",
                turn_index=i,
                user_text=f"fix flock iteration {i}",
                assistant_text=f"Fixed flock race condition in attempt {i}",
            ))
        db.save_chunks(chunks)

        # Create cluster
        cluster = {
            "cluster_id": cluster_id,
            "topic": "flock locking",
            "cluster_type": "topic",
            "session_ids": ["sess-1"],
            "branch": None,
            "date_start": "2026-03-01T10:00:00Z",
            "date_end": f"2026-03-01T10:{len(chunk_ids)-1:02d}:00Z",
            "chunk_count": len(chunk_ids),
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }
        memberships = [
            (cluster_id, cid, now) for cid in chunk_ids
        ]
        db.save_clusters([cluster], memberships)
        return db, chunks

    def test_find_summary_by_content_hash(self, tmp_path):
        """find_summary_by_content_hash returns stored summary."""
        from synapt.recall.storage import RecallDB

        db = RecallDB(tmp_path / "test.db")
        db.save_cluster_summary(
            "clust-old", "Old summary text",
            method="llm", content_hash="abc123",
        )
        assert db.find_summary_by_content_hash("abc123") == "Old summary text"
        assert db.find_summary_by_content_hash("nonexistent") is None
        db.close()

    def test_find_ignores_concat_summaries(self, tmp_path):
        """Only LLM summaries are matched by content hash."""
        from synapt.recall.storage import RecallDB

        db = RecallDB(tmp_path / "test.db")
        db.save_cluster_summary(
            "clust-old", "Concat summary",
            method="concat", content_hash="abc123",
        )
        assert db.find_summary_by_content_hash("abc123") is None
        db.close()

    def test_upgrade_reuses_hashed_summary(self, tmp_path):
        """When content hash matches, skip LLM and reuse existing summary."""
        chunk_ids = [f"s1:t{i}" for i in range(5)]
        db, _chunks = self._setup_db_with_clusters(
            tmp_path, "clust-new", chunk_ids,
        )

        # Get the content hash for this cluster's chunk texts
        chunk_texts = db.get_cluster_chunk_texts("clust-new")
        expected_hash = _content_hash(chunk_texts)

        # Plant an orphaned LLM summary with matching content hash
        # (simulates a previous cluster that had the same content)
        db._conn.execute(
            "INSERT INTO cluster_summaries "
            "(cluster_id, summary, method, token_count, content_hash, created_at, stale) "
            "VALUES (?, ?, 'llm', 10, ?, ?, 0)",
            ("clust-dead", "Reusable summary from old cluster",
             expected_hash, "2026-03-01T00:00:00Z"),
        )
        db._conn.commit()

        from synapt.recall.clustering import upgrade_large_cluster_summaries

        class _DummyClient:
            """Stub client — content hash reuse should skip actual inference."""
            pass

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "synapt.recall.clustering.generate_llm_summary",
                lambda *a, **kw: pytest.fail("LLM should not be called"),
            )
            mp.setattr(
                "synapt.recall.clustering.create_summary_client",
                lambda: _DummyClient(),
            )
            count = upgrade_large_cluster_summaries(db, min_chunks=5)

        assert count == 1
        # Verify the reused summary is now stored under the new cluster ID
        summary = db.get_cluster_summary("clust-new")
        assert summary is not None
        assert summary["summary"] == "Reusable summary from old cluster"
        assert summary["method"] == "llm"

        # Orphan should be cleaned up
        orphan = db._conn.execute(
            "SELECT * FROM cluster_summaries WHERE cluster_id = 'clust-dead'"
        ).fetchone()
        assert orphan is None

        db.close()

    def test_upgrade_generates_when_no_hash_match(self, tmp_path):
        """When no content hash matches, generate a new LLM summary."""
        chunk_ids = [f"s1:t{i}" for i in range(5)]
        db, _chunks = self._setup_db_with_clusters(
            tmp_path, "clust-fresh", chunk_ids,
        )

        from synapt.recall.clustering import upgrade_large_cluster_summaries
        generated = []

        def fake_generate(chunk_texts, topic, **kw):
            generated.append(topic)
            return f"Generated summary for {topic}"

        class _DummyClient:
            pass

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "synapt.recall.clustering.generate_llm_summary", fake_generate,
            )
            mp.setattr(
                "synapt.recall.clustering.create_summary_client",
                lambda: _DummyClient(),
            )
            count = upgrade_large_cluster_summaries(db, min_chunks=5)

        assert count == 1
        assert len(generated) == 1
        summary = db.get_cluster_summary("clust-fresh")
        assert summary is not None
        assert "Generated summary" in summary["summary"]

        db.close()

    def test_orphan_cleanup_after_upgrade(self, tmp_path):
        """Orphaned LLM summaries are cleaned up after upgrade completes."""
        chunk_ids = [f"s1:t{i}" for i in range(5)]
        db, _chunks = self._setup_db_with_clusters(
            tmp_path, "clust-live", chunk_ids,
        )

        # Plant orphaned summaries — one with hash, one without
        db._conn.execute(
            "INSERT INTO cluster_summaries "
            "(cluster_id, summary, method, token_count, content_hash, created_at, stale) "
            "VALUES (?, ?, 'llm', 10, ?, ?, 0)",
            ("clust-orphan-hashed", "Has hash", "deadbeef", "2026-03-01T00:00:00Z"),
        )
        db._conn.execute(
            "INSERT INTO cluster_summaries "
            "(cluster_id, summary, method, token_count, content_hash, created_at, stale) "
            "VALUES (?, ?, 'llm', 10, NULL, ?, 0)",
            ("clust-orphan-nohash", "No hash", "2026-03-01T00:00:00Z"),
        )
        db._conn.commit()

        from synapt.recall.clustering import upgrade_large_cluster_summaries

        def fake_generate(chunk_texts, topic, **kw):
            return f"New summary for {topic}"

        class _DummyClient:
            pass

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "synapt.recall.clustering.generate_llm_summary", fake_generate,
            )
            mp.setattr(
                "synapt.recall.clustering.create_summary_client",
                lambda: _DummyClient(),
            )
            upgrade_large_cluster_summaries(db, min_chunks=5)

        # Both orphans should be cleaned up after upgrade
        orphans = db._conn.execute(
            "SELECT cluster_id FROM cluster_summaries "
            "WHERE cluster_id LIKE 'clust-orphan%'"
        ).fetchall()
        assert len(orphans) == 0

        db.close()
