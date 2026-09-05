"""TDD spec: recall#435 follow-on -- the bounded recluster maintenance op.

recall#435's ``skip_clustering`` build path (merged) saves transcript chunks
to FTS5 without clustering them, so they stay unclustered until something
else clusters them. This is that something else: ``synapt maintain
--recluster``, which clusters the oldest bounded batch of stale chunks and
reports what is left.

The load-bearing property under test is the memory bound: the op must never
reload the already-clustered corpus to add a small batch. Concretely, that
means it must call the ADDITIVE ``append_clusters`` (insert-only), never the
full-corpus-REPLACE ``save_clusters`` (deletes every existing topic cluster
first) -- using the wrong one would silently erase every already-clustered
chunk the first time the maintenance op ran on a real store. That distinction
is asserted directly, not just inferred from an end-to-end count.
"""

from __future__ import annotations

import os
from pathlib import Path

from conftest import assistant_entry, user_text_entry, write_jsonl


def _transcript(path: Path, *, turns: int = 2, prefix: str = "q") -> Path:
    entries = []
    for i in range(turns):
        entries.append(
            user_text_entry(f"{prefix} question {i}", uuid=f"{prefix}-u{i}",
                             ts=f"2026-03-01T10:{i:02d}:00Z")
        )
        entries.append(
            assistant_entry(text=f"{prefix} answer {i}", uuid=f"{prefix}-a{i}",
                             ts=f"2026-03-01T10:{i:02d}:30Z")
        )
    write_jsonl(path, entries)
    return path


def _build_store_with_stale_chunks(tmp_path, *, turns: int = 8, prefix: str = "recluster"):
    """A store built with skip_clustering=True: chunks indexed, none clustered."""
    from synapt.recall.cli import _archive_and_build

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _transcript(source / "s1.jsonl", turns=turns, prefix=prefix)

    _archive_and_build(project, source_dirs=[source], use_embeddings=False,
                        incremental=True, skip_clustering=True)
    return project


def _open_db(project):
    from synapt.recall.core import project_index_dir
    from synapt.recall.storage import RecallDB

    return RecallDB(project_index_dir(project) / "recall.db")


def test_stale_transcript_chunk_ids_finds_exactly_the_skipped_chunks(tmp_path):
    from synapt.recall.clustering import stale_transcript_chunk_ids

    project = _build_store_with_stale_chunks(tmp_path, turns=8)
    db = _open_db(project)
    try:
        stale = stale_transcript_chunk_ids(db)
        assert len(stale) == 8, f"all 8 transcript chunks should be stale: {stale}"
        assert db.cluster_count(cluster_type="topic") == 0
    finally:
        db.close()


def test_recluster_end_to_end_clusters_the_stale_batch(tmp_path):
    """The one e2e: build with skip_clustering, recluster, verify by fruit."""
    from synapt.recall.clustering import recluster_stale_chunks, stale_transcript_chunk_ids

    project = _build_store_with_stale_chunks(tmp_path, turns=8)
    db = _open_db(project)
    try:
        assert len(stale_transcript_chunk_ids(db)) == 8

        receipt = recluster_stale_chunks(db, batch_size=100)

        assert receipt["refused"] is False
        assert receipt["total_stale_at_start"] == 8
        assert receipt["still_stale"] == 0, receipt
        assert stale_transcript_chunk_ids(db) == []

        # A real cluster-scoped read, not just a count: the chunks are
        # actually reachable through cluster_chunks under a real cluster_id.
        rows = db._conn.execute(
            "SELECT cc.chunk_id FROM cluster_chunks cc "
            "JOIN clusters cl ON cl.cluster_id = cc.cluster_id "
            "WHERE cl.cluster_type = 'topic'"
        ).fetchall()
        assert len(rows) >= 1, "reclustered chunks must be reachable via cluster_chunks"
    finally:
        db.close()


def test_recluster_never_touches_already_clustered_chunks(tmp_path):
    """The critical regression: using save_clusters (full REPLACE) instead of
    append_clusters (additive) would wipe every pre-existing topic cluster
    the first time --recluster ran on a real store. Build a FULLY clustered
    store, hand-craft one stale chunk, recluster, and assert the original
    cluster survives untouched."""
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.clustering import recluster_stale_chunks, stale_transcript_chunk_ids
    from synapt.recall.core import project_index_dir

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _transcript(source / "s1.jsonl", turns=8, prefix="original")

    # Default build: clusters everything (skip_clustering defaults False).
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    db = _open_db(project)
    try:
        original_cluster_rows = db._conn.execute(
            "SELECT cluster_id, chunk_count FROM clusters WHERE cluster_type = 'topic'"
        ).fetchall()
        assert original_cluster_rows, "fixture must produce at least one real topic cluster"
        original_membership_count = db._conn.execute(
            "SELECT COUNT(*) FROM cluster_chunks cc "
            "JOIN clusters cl ON cl.cluster_id = cc.cluster_id "
            "WHERE cl.cluster_type = 'topic'"
        ).fetchone()[0]
        assert original_membership_count > 0
        assert stale_transcript_chunk_ids(db) == [], "fixture must start fully clustered"
    finally:
        db.close()

    # Add a second, separate transcript WITHOUT clustering it (simulates the
    # automatic cold_no_caller_refresh path leaving new chunks stale).
    _transcript(source / "s2.jsonl", turns=4, prefix="newstuff")
    _archive_and_build(project, source_dirs=[source], use_embeddings=False,
                        incremental=True, skip_clustering=True)

    db = _open_db(project)
    try:
        stale_before = stale_transcript_chunk_ids(db)
        assert len(stale_before) == 4, f"only the new transcript's chunks should be stale: {stale_before}"

        receipt = recluster_stale_chunks(db, batch_size=100)
        assert receipt["still_stale"] == 0, receipt

        # The ORIGINAL cluster(s) must be byte-for-byte present -- this is
        # what would break if recluster_stale_chunks called save_clusters
        # (full-corpus replace) instead of append_clusters (additive).
        surviving_rows = db._conn.execute(
            "SELECT cluster_id, chunk_count FROM clusters WHERE cluster_type = 'topic' "
            "ORDER BY cluster_id"
        ).fetchall()
        surviving_ids = {r[0] for r in surviving_rows}
        for cluster_id, chunk_count in original_cluster_rows:
            assert cluster_id in surviving_ids, (
                f"original cluster {cluster_id} was destroyed by recluster -- "
                "this is the save_clusters-vs-append_clusters regression"
            )
        surviving_membership_count = db._conn.execute(
            "SELECT COUNT(*) FROM cluster_chunks cc "
            "JOIN clusters cl ON cl.cluster_id = cc.cluster_id "
            "WHERE cl.cluster_type = 'topic'"
        ).fetchone()[0]
        assert surviving_membership_count >= original_membership_count, (
            "recluster must add memberships, never reduce below the pre-existing count"
        )
    finally:
        db.close()


def test_recluster_refuses_above_ceiling_and_names_the_drain_command(tmp_path):
    from synapt.recall.clustering import recluster_stale_chunks

    project = _build_store_with_stale_chunks(tmp_path, turns=8)
    db = _open_db(project)
    try:
        # Test-scale ceiling: catches a mutant that ignores the bound and
        # would otherwise attempt the whole stale set regardless of size.
        receipt = recluster_stale_chunks(db, batch_size=100, refuse_above=3)

        assert receipt["refused"] is True
        assert receipt["total_stale_at_start"] == 8
        assert receipt["still_stale"] == 8, "a refusal must cluster nothing"
        assert receipt["drain_command"], "a refusal must name an actionable next command"
        assert "--recluster" in receipt["drain_command"]
    finally:
        db.close()


def test_recluster_processes_oldest_batch_first_and_reports_backlog(tmp_path):
    from synapt.recall.clustering import recluster_stale_chunks, stale_transcript_chunk_ids

    project = _build_store_with_stale_chunks(tmp_path, turns=8)
    db = _open_db(project)
    try:
        stale_ids_before = stale_transcript_chunk_ids(db)
        oldest_half = set(stale_ids_before[:4])

        receipt = recluster_stale_chunks(db, batch_size=4)

        assert receipt["still_stale"] == 4, receipt
        assert receipt["drain_command"], "a nonzero backlog must name the drain command"

        remaining = set(stale_transcript_chunk_ids(db))
        assert remaining.isdisjoint(oldest_half), (
            "the OLDEST batch must be the one clustered, not an arbitrary subset"
        )
    finally:
        db.close()
