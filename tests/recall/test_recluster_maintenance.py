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


# _transcript's shared "question"/"answer" boilerplate made
# every "singleton" cluster with every OTHER singleton (measured: a
# 5-singleton fixture built with _transcript came back 100% clustered, not
# stuck at all). Each of these uses a fully distinct sentence template, not
# a shared phrase with a swapped-in topic word, so nothing overlaps between
# them beyond stopwords.
_DISJOINT_SINGLETON_TEXT = [
    ("Giraffes wander the savanna grasslands searching for acacia leaves",
     "Their long necks reach branches nothing else on the plain can touch"),
    ("Lighthouses warn sailors away from rocky treacherous coastlines",
     "Keepers once climbed spiral staircases to trim the flickering wick"),
]


def _disjoint_singleton(path: Path, *, index: int) -> Path:
    """A one-turn transcript whose vocabulary shares nothing with the other
    disjoint singleton or with the clusterable group below."""
    user_text, assistant_text = _DISJOINT_SINGLETON_TEXT[index]
    write_jsonl(path, [
        user_text_entry(user_text, uuid=f"s{index}-u", ts=f"2026-03-01T09:{index:02d}:00Z"),
        assistant_entry(text=assistant_text, uuid=f"s{index}-a",
                         ts=f"2026-03-01T09:{index:02d}:30Z"),
    ])
    return path


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


def test_recluster_ceiling_boundary_exact_value_proceeds_one_over_refuses(tmp_path):
    """Pins the '>' in the refuse_above comparison: a mutant weakening it to
    '>=' would refuse at the exact ceiling too, and every other test in this
    file uses stale counts far from any ceiling, so nothing else catches
    that one-token change."""
    from synapt.recall.clustering import recluster_stale_chunks

    project = _build_store_with_stale_chunks(tmp_path, turns=8)
    db = _open_db(project)
    try:
        # Exactly AT the ceiling: must proceed (docstring says "above", not
        # "at or above"). batch_size=100 so the whole set clusters in one go.
        receipt_at = recluster_stale_chunks(db, batch_size=100, refuse_above=8)
        assert receipt_at["refused"] is False, (
            "total_stale == refuse_above must proceed, not refuse"
        )
    finally:
        db.close()

    second = tmp_path / "second"
    second.mkdir()
    project2 = _build_store_with_stale_chunks(second, turns=8)
    db2 = _open_db(project2)
    try:
        # ONE past the ceiling: must refuse.
        receipt_over = recluster_stale_chunks(db2, batch_size=100, refuse_above=7)
        assert receipt_over["refused"] is True, (
            "total_stale == refuse_above + 1 must refuse"
        )
    finally:
        db2.close()


def test_recluster_drain_command_names_the_actual_computed_run_count(tmp_path):
    """Pins the ceil-division arithmetic itself: a mutant hardcoding
    runs=1 would pass every other test in this file, because every other
    scenario here happens to need exactly one more run."""
    from synapt.recall.clustering import recluster_stale_chunks

    # 12 chunks, same prefix/topic so they cluster together as one group
    # (already established by the turns=8 e2e above) -- batch_size=5 leaves
    # a deterministic remainder whose drain count is NOT 1.
    project = _build_store_with_stale_chunks(tmp_path, turns=12)
    db = _open_db(project)
    try:
        receipt = recluster_stale_chunks(db, batch_size=5)

        assert receipt["still_stale"] == 7, receipt
        # ceil(7 / 5) == 2 -- if this were hardcoded to 1, this assertion
        # is the only thing in the file that would catch it.
        assert "at least 2 more runs" in receipt["drain_command"], receipt
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


# the recall#435 op reselects the same stale chunks every run
# when they fail to reach MIN_CLUSTER_SIZE (measured on the real store:
# 98.9% batch overlap between two consecutive runs, only 22 of 2000 chunks
# actually clearing). recluster_attempts + _select_recluster_batch fix this
# by routing future batches around chunks already tried and failed.

def _build_store_with_two_singletons_and_a_cluster(tmp_path):
    """2 genuinely disjoint singletons (will never cluster with anything)
    plus 8 chunks that all share one topic (will cluster together in one
    shot). 10 stale chunks total.

    Deliberately does NOT assume anything about which chunks land in which
    position of stale_transcript_chunk_ids()'s own ordering -- measured
    while writing this test that chunk insertion order is not simply
    "files in the order passed" or "oldest entry timestamp first", so every
    assertion below identifies chunks by content (the id string), or uses a
    batch_size covering the WHOLE set so ordering cannot matter."""
    from synapt.recall.cli import _archive_and_build

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _disjoint_singleton(source / "singleton0.jsonl", index=0)
    _disjoint_singleton(source / "singleton1.jsonl", index=1)
    _transcript(source / "cluster.jsonl", turns=8, prefix="widget")

    _archive_and_build(project, source_dirs=[source], use_embeddings=False,
                        incremental=True, skip_clustering=True)
    return project


def _add_clusterable_chunks(project, tmp_path, *, name: str, turns: int, prefix: str):
    """Adds a new skip_clustering=True source, simulating more chunks
    arriving (e.g. from ongoing team activity) between recluster runs."""
    from synapt.recall.cli import _archive_and_build

    source = tmp_path / f"source_{name}"
    source.mkdir()
    _transcript(source / f"{name}.jsonl", turns=turns, prefix=prefix)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False,
                        incremental=True, skip_clustering=True)


def test_recluster_attempted_marker_routes_around_stuck_singletons(tmp_path):
    """After a singleton fails to cluster, a LATER run must not reselect it
    while fresh (never-tried) chunks exist -- this is the mutant's target
    (dropping the skip reds this assertion).

    _select_recluster_batch's fresh-first-then-fallback split is order-
    independent by construction (fresh always fills the batch before any
    fallback chunk is considered, regardless of how fresh/attempted ids are
    interleaved in the stale list) -- that property is exactly what this
    test relies on instead of any assumption about processing order."""
    from synapt.recall.clustering import recluster_stale_chunks, stale_transcript_chunk_ids

    project = _build_store_with_two_singletons_and_a_cluster(tmp_path)
    db = _open_db(project)
    try:
        stale_before = set(stale_transcript_chunk_ids(db))
        assert len(stale_before) == 10
        singleton_ids = {cid for cid in stale_before if "singleton" in cid}
        assert len(singleton_ids) == 2

        # Run 1 sees ALL 10 at once (batch_size=10): the 8 widget chunks
        # clear MIN_CLUSTER_SIZE together, the 2 disjoint singletons cannot
        # cluster with anything and must fail -- deterministic regardless
        # of ordering, since nothing is left out of this batch.
        receipt1 = recluster_stale_chunks(db, batch_size=10)
        assert receipt1["chunks_clustered"] == 8, receipt1
        attempted_after_1 = db.get_recluster_attempted_ids()
        assert attempted_after_1 == singleton_ids, (
            f"only the two chunks that actually FAILED should be marked, got {attempted_after_1}"
        )
        assert set(stale_transcript_chunk_ids(db)) == singleton_ids

        # More chunks arrive (a different topic), simulating ongoing
        # skip_clustering builds elsewhere on the store.
        _add_clusterable_chunks(project, tmp_path, name="gadgets", turns=4, prefix="gadget")
        stale_now = set(stale_transcript_chunk_ids(db))
        gadget_ids = stale_now - singleton_ids
        assert singleton_ids <= stale_now, "the two singletons must still be present and stale"
        assert len(gadget_ids) == 4, stale_now

        # Run 2, batch_size=4: exactly the 4 fresh gadgets exist. With the
        # skip working, the 2 already-attempted singletons must NOT fill
        # any of this batch even though 4 == batch_size leaves no numeric
        # need to reach for them.
        receipt2 = recluster_stale_chunks(db, batch_size=4)
        assert receipt2["fresh_in_batch"] == 4, receipt2
        assert receipt2["fallback_in_batch"] == 0, (
            f"the two stuck singletons must not be reselected this run: {receipt2}"
        )
        assert receipt2["chunks_clustered"] == 4, receipt2

        # Verify by fruit: only the singletons remain stale.
        assert set(stale_transcript_chunk_ids(db)) == singleton_ids
    finally:
        db.close()


def test_recluster_attempted_marker_falls_back_once_fresh_is_exhausted(tmp_path):
    """Once every never-tried chunk is gone, the op must fall back to
    already-attempted ones rather than declaring victory with a nonzero
    backlog it refuses to touch."""
    from synapt.recall.clustering import recluster_stale_chunks, stale_transcript_chunk_ids

    project = _build_store_with_two_singletons_and_a_cluster(tmp_path)
    db = _open_db(project)
    try:
        recluster_stale_chunks(db, batch_size=10)  # widgets cluster, singletons fail
        assert len(stale_transcript_chunk_ids(db)) == 2  # only the 2 singletons remain

        # No fresh chunks exist at all now -- every stale chunk is
        # already-attempted. The op must still process them (fall back)
        # rather than treat "0 fresh available" as "nothing to do".
        receipt2 = recluster_stale_chunks(db, batch_size=4)
        assert receipt2["fresh_in_batch"] == 0, receipt2
        assert receipt2["fallback_in_batch"] == 2, receipt2
    finally:
        db.close()
