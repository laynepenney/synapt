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


# recluster_stale_chunks() clusters a batch against ITSELF only,
# so a stale chunk that IS similar to an EXISTING cluster still gets no home
# unless enough of its own similar chunks happen to land in the SAME batch.
# merge_into_existing lets a stale chunk join an existing cluster (asymmetric
# containment of the cluster's persisted top-N signature in the chunk's own
# tokens -- cheap, no corpus reload) before self-batch clustering runs.

# A dozen distinct topic words, cycled across turns so each has real
# WITHIN-cluster document frequency (unlike a 2-3-word fixture, whose
# signature would be too small to ever clear MIN_SHARED_SIGNATURE_TOKENS --
# measured while writing this: the "original question/answer" fixture used
# elsewhere in this file produces a 3-token signature, below the floor).
_TOPIC_WORDS = [
    "kubernetes", "deployment", "container", "cluster", "pod", "service",
    "ingress", "namespace", "volume", "secret", "configmap", "replica",
]


def _topic_transcript(path: Path, *, turns: int = 8) -> Path:
    """A transcript whose turns share most of ``_TOPIC_WORDS``, so the
    resulting cluster's persisted signature carries a realistic number of
    genuinely high-document-frequency tokens."""
    entries = []
    for i in range(turns):
        words = " ".join(w for j, w in enumerate(_TOPIC_WORDS) if (j + i) % 3 != 0)
        entries.append(
            user_text_entry(f"question about {words}", uuid=f"topic-u{i}",
                             ts=f"2026-03-01T10:{i:02d}:00Z")
        )
        entries.append(
            assistant_entry(text=f"answer about {words}", uuid=f"topic-a{i}",
                             ts=f"2026-03-01T10:{i:02d}:30Z")
        )
    write_jsonl(path, entries)
    return path


def _similar_to_cluster_singleton(path: Path) -> Path:
    """One-turn transcript sharing every one of ``_TOPIC_WORDS`` with the
    topic cluster built by ``_topic_transcript`` (distinct uuids so a
    rebuild does not collide on identity)."""
    words = " ".join(_TOPIC_WORDS)
    write_jsonl(path, [
        user_text_entry(f"question about {words}", uuid="sim-u",
                         ts="2026-03-01T11:00:00Z"),
        assistant_entry(text=f"answer about {words}", uuid="sim-a",
                         ts="2026-03-01T11:00:30Z"),
    ])
    return path


def test_recluster_merge_into_existing_joins_a_similar_stale_chunk(tmp_path):
    """The one e2e: a stale chunk whose tokens CONTAIN an
    EXISTING cluster's persisted signature joins that cluster under its SAME
    cluster_id (membership added, chunk_count grows) instead of needing a
    same-batch partner to form a new one. A dissimilar stale chunk in the
    same batch does not merge and, alone, cannot reach MIN_CLUSTER_SIZE by
    self-batch clustering either -- it stays stale, which is correct: this
    lane widens WHERE a chunk can find a home, it does not lower the bar
    for what counts as a cluster."""
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.clustering import recluster_stale_chunks, stale_transcript_chunk_ids

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _topic_transcript(source / "topic.jsonl", turns=8)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    db = _open_db(project)
    try:
        existing = db._conn.execute(
            "SELECT cluster_id, chunk_count FROM clusters WHERE cluster_type = 'topic'"
        ).fetchall()
        assert len(existing) == 1, f"fixture must produce exactly one existing cluster: {existing}"
        existing_cluster_id, existing_count = existing[0]

        # save_clusters must have written a signature at build time -- this
        # is the other half of the wiring, checked directly rather than
        # only inferred from the merge succeeding below.
        signatures = db.load_cluster_token_signatures()
        assert existing_cluster_id in signatures, (
            f"a topic cluster must get a persisted signature at build time: {signatures.keys()}"
        )
        assert len(signatures[existing_cluster_id]) >= 8, (
            f"fixture's signature must clear MIN_SHARED_SIGNATURE_TOKENS to "
            f"be a meaningful test: {signatures[existing_cluster_id]}"
        )
    finally:
        db.close()

    _similar_to_cluster_singleton(source / "similar.jsonl")
    _disjoint_singleton(source / "dissimilar.jsonl", index=0)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False,
                        incremental=True, skip_clustering=True)

    db = _open_db(project)
    try:
        stale_before = stale_transcript_chunk_ids(db)
        assert len(stale_before) == 2, f"exactly the two new chunks should be stale: {stale_before}"

        receipt = recluster_stale_chunks(db, batch_size=100, merge_into_existing=True)

        assert receipt["merged_into_existing"] == 1, receipt
        assert receipt["still_stale"] == 1, receipt  # the dissimilar chunk, alone

        stale_after = stale_transcript_chunk_ids(db)
        assert len(stale_after) == 1
        merged_chunk_id = (set(stale_before) - set(stale_after)).pop()

        row = db._conn.execute(
            "SELECT cluster_id, chunk_count FROM clusters WHERE cluster_id = ?",
            (existing_cluster_id,),
        ).fetchone()
        assert row is not None, "the existing cluster must still exist under the SAME cluster_id"
        assert row[1] == existing_count + 1, (
            f"chunk_count must grow by exactly the one merged chunk: {row}"
        )

        member_ids = {
            r[0] for r in db._conn.execute(
                "SELECT chunk_id FROM cluster_chunks WHERE cluster_id = ?",
                (existing_cluster_id,),
            ).fetchall()
        }
        assert merged_chunk_id in member_ids, (
            f"the merged chunk must be a real member of the existing cluster: {member_ids}"
        )
    finally:
        db.close()


def test_recluster_merge_into_existing_dry_run_reports_without_writing(tmp_path):
    """dry_run=True computes the SAME numbers a real run
    would (verified by running the identical batch for real immediately
    after) while writing nothing -- the whole point being that the
    mandated hand-read of merged pairs never has to touch the live store,
    or even a full copy of it, to see what a merge run WOULD do."""
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.clustering import recluster_stale_chunks, stale_transcript_chunk_ids

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _topic_transcript(source / "topic.jsonl", turns=8)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    _similar_to_cluster_singleton(source / "similar.jsonl")
    _disjoint_singleton(source / "dissimilar.jsonl", index=0)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False,
                        incremental=True, skip_clustering=True)

    db = _open_db(project)
    try:
        existing_cluster_id, existing_count, existing_topic = db._conn.execute(
            "SELECT cluster_id, chunk_count, topic FROM clusters WHERE cluster_type = 'topic'"
        ).fetchone()
        stale_before = set(stale_transcript_chunk_ids(db))
        member_count_before = db._conn.execute(
            "SELECT COUNT(*) FROM cluster_chunks WHERE cluster_id = ?",
            (existing_cluster_id,),
        ).fetchone()[0]

        dry_receipt = recluster_stale_chunks(
            db, batch_size=100, merge_into_existing=True, dry_run=True,
        )

        assert dry_receipt["dry_run"] is True
        assert dry_receipt["merged_into_existing"] == 1, dry_receipt
        assert dry_receipt["still_stale"] == 1, dry_receipt  # the dissimilar chunk
        assert dry_receipt["merge_run_id"] is None, (
            "a dry run stamps nothing, so it must report no run id to undo"
        )
        assert len(dry_receipt["merge_samples"]) == 1, dry_receipt
        sample = dry_receipt["merge_samples"][0]
        assert sample["cluster_id"] == existing_cluster_id
        assert sample["cluster_topic"] == existing_topic, (
            "a hand-read needs the topic label without a second DB query"
        )

        # Verify by fruit, not by trusting the receipt: NOTHING moved.
        assert set(stale_transcript_chunk_ids(db)) == stale_before, (
            "dry_run must not change which chunks are stale"
        )
        row = db._conn.execute(
            "SELECT chunk_count FROM clusters WHERE cluster_id = ?",
            (existing_cluster_id,),
        ).fetchone()
        assert row[0] == existing_count, "dry_run must not change chunk_count"
        member_count_after = db._conn.execute(
            "SELECT COUNT(*) FROM cluster_chunks WHERE cluster_id = ?",
            (existing_cluster_id,),
        ).fetchone()[0]
        assert member_count_after == member_count_before, (
            "dry_run must not add any cluster_chunks row"
        )
        assert db.get_recluster_attempted_ids() == set(), (
            "dry_run must not mark an attempt either -- that changes future batch selection"
        )

        # The SAME batch, run for real right after, must land on the SAME numbers.
        real_receipt = recluster_stale_chunks(db, batch_size=100, merge_into_existing=True)
        assert real_receipt["merged_into_existing"] == dry_receipt["merged_into_existing"]
        assert real_receipt["chunks_clustered"] == dry_receipt["chunks_clustered"]
        assert real_receipt["still_stale"] == dry_receipt["still_stale"]
    finally:
        db.close()


def test_merge_run_id_is_stamped_and_distinct_across_separate_runs(tmp_path):
    """A membership row a MERGE writes carries that run's
    OWN run_id (never a shared added_at timestamp -- the fallback a
    live-store incident needed before this column existed), so an undo can
    be scoped to exactly one run without touching another's rows. Rows from
    ORDINARY self-batch clustering (formed at build time, never by a merge)
    carry no run_id at all -- there is nothing there to undo-by-run."""
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.clustering import backfill_cluster_signatures, recluster_stale_chunks

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _topic_transcript(source / "topic.jsonl", turns=8)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    db = _open_db(project)
    try:
        existing_cluster_id = db._conn.execute(
            "SELECT cluster_id FROM clusters WHERE cluster_type = 'topic'"
        ).fetchone()[0]
        original_run_ids = {
            r[0] for r in db._conn.execute(
                "SELECT run_id FROM cluster_chunks WHERE cluster_id = ?",
                (existing_cluster_id,),
            ).fetchall()
        }
        assert original_run_ids == {None}, (
            f"chunks clustered at build time must carry no run_id: {original_run_ids}"
        )
    finally:
        db.close()

    write_jsonl(source / "similar1.jsonl", [
        user_text_entry("question about " + " ".join(_TOPIC_WORDS), uuid="sim1-u",
                         ts="2026-03-01T11:01:00Z"),
        assistant_entry(text="answer about " + " ".join(_TOPIC_WORDS), uuid="sim1-a",
                         ts="2026-03-01T11:01:30Z"),
    ])
    _archive_and_build(project, source_dirs=[source], use_embeddings=False,
                        incremental=True, skip_clustering=True)

    db = _open_db(project)
    try:
        receipt1 = recluster_stale_chunks(db, batch_size=100, merge_into_existing=True)
        assert receipt1["merged_into_existing"] == 1, receipt1
        run_id_1 = receipt1["merge_run_id"]
        assert run_id_1, "a real merge run must report the run_id it stamped"

        # The merge just deleted this cluster's signature (self-healing
        # invalidation) -- restore it so a SECOND, later merge has
        # something to match against, same as a real operator's next
        # backfill would.
        backfill_cluster_signatures(db, batch_size=500)
    finally:
        db.close()

    write_jsonl(source / "similar2.jsonl", [
        user_text_entry("question about " + " ".join(_TOPIC_WORDS), uuid="sim2-u",
                         ts="2026-03-01T11:02:00Z"),
        assistant_entry(text="answer about " + " ".join(_TOPIC_WORDS), uuid="sim2-a",
                         ts="2026-03-01T11:02:30Z"),
    ])
    _archive_and_build(project, source_dirs=[source], use_embeddings=False,
                        incremental=True, skip_clustering=True)

    db = _open_db(project)
    try:
        receipt2 = recluster_stale_chunks(db, batch_size=100, merge_into_existing=True)
        assert receipt2["merged_into_existing"] == 1, receipt2
        run_id_2 = receipt2["merge_run_id"]
        assert run_id_2 and run_id_2 != run_id_1, (
            "two separate merge runs must be undo-able independently"
        )

        rows = dict(db._conn.execute(
            "SELECT chunk_id, run_id FROM cluster_chunks "
            "WHERE cluster_id = ? AND run_id IS NOT NULL",
            (existing_cluster_id,),
        ).fetchall())
        assert len(rows) == 2, rows
        assert set(rows.values()) == {run_id_1, run_id_2}, (
            f"exactly the two runs' own ids must be on their own rows: {rows}"
        )
    finally:
        db.close()


def test_backfill_cluster_signatures_fills_in_clusters_that_predate_the_table(tmp_path):
    """A cluster with no persisted signature yet (predates this table, or
    had one invalidated by ``merge_chunks_into_cluster``) gets one from
    ``backfill_cluster_signatures``, computed from its ACTUAL current
    members -- bounded per batch of clusters, never the whole corpus."""
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.clustering import backfill_cluster_signatures

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _topic_transcript(source / "topic.jsonl", turns=8)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    db = _open_db(project)
    try:
        cluster_id = db._conn.execute(
            "SELECT cluster_id FROM clusters WHERE cluster_type = 'topic'"
        ).fetchone()[0]
        assert cluster_id in db.load_cluster_token_signatures()

        # Simulate a cluster that predates the signature table.
        db._conn.execute(
            "DELETE FROM cluster_token_signatures WHERE cluster_id = ?", (cluster_id,)
        )
        db._conn.commit()
        assert cluster_id in db.active_topic_clusters_missing_signature()

        receipt = backfill_cluster_signatures(db, batch_size=500)
        assert receipt["clusters_signed"] == 1, receipt
        assert receipt["clusters_remaining"] == 0, receipt

        signatures = db.load_cluster_token_signatures()
        assert cluster_id in signatures
        assert len(signatures[cluster_id]) >= 8, signatures[cluster_id]
    finally:
        db.close()


def test_compute_boilerplate_stoplist_finds_tokens_pervasive_across_clusters():
    """A token in a small minority of signatures is real topic vocabulary;
    a token repeated across most of them (an injected boilerplate preamble,
    on the real store) cannot be evidence of any one topic. Pure function,
    no DB -- the incident this catches: real-store measurement found one
    token set shared across 35 clusters via a verbatim skill preamble."""
    from synapt.recall.clustering import compute_boilerplate_stoplist

    signatures = {
        "clust-a": {"kubernetes", "pod", "boilerplate_marker"},
        "clust-b": {"deployment", "service", "boilerplate_marker"},
        "clust-c": {"release", "pyproject", "boilerplate_marker"},
        "clust-d": {"boilerplate_marker"},
        "clust-e": {"giraffe", "savanna"},  # no boilerplate token at all
    }
    stoplist = compute_boilerplate_stoplist(signatures, min_fraction=0.20)
    tokens = [tok for tok, _frac in stoplist]

    assert "boilerplate_marker" in tokens, stoplist
    # boilerplate_marker is in 4/5 = 0.80, the highest fraction -- must sort first.
    assert tokens[0] == "boilerplate_marker", stoplist
    # Real topic vocabulary, each appearing in exactly one signature (0.20,
    # not strictly greater than the 0.20 floor), must NOT be flagged.
    for real_word in ("kubernetes", "pod", "deployment", "giraffe"):
        assert real_word not in tokens, stoplist


def test_compute_boilerplate_stoplist_empty_signatures_is_empty_not_a_crash():
    from synapt.recall.clustering import compute_boilerplate_stoplist

    assert compute_boilerplate_stoplist({}) == []


def _decoy_signatures(n: int, *, size: int = 64) -> dict[str, frozenset[str]]:
    """N cluster signatures with mutually disjoint, synthetic vocabulary --
    background population for tests that are NOT about distinctiveness
    weighting, so every token in the signature actually under test still
    gets document frequency 1 (present in only that one signature) and
    therefore a UNIFORM per-token IDF. Under uniform IDF, the weighted
    containment ratio in ``_match_existing_cluster`` reduces exactly to the
    old ``|shared| / |signature|`` count ratio (every token contributes the
    same weight), so these decoys let existing count/ratio-focused tests
    keep testing exactly what they tested before the distinctiveness
    weighting existed, just against a realistically-sized cluster
    population instead of a population of one or two."""
    return {
        f"decoy-{i}": frozenset(f"decoytok{i}_{j}" for j in range(size))
        for i in range(n)
    }


# Hand-read finding: recall's own context-echo user_text
# ("(context: User previously asked: X)", core.py's synthetic restatement
# for every sub-chunk past the first of a long assistant reply) is the
# ENTIRE user_text for those chunks, not a span inside a larger blob -- so
# two sequential sub-chunks of the SAME exchange share it verbatim, which
# can supply enough shared tokens to pass MIN_SHARED_SIGNATURE_TOKENS and
# CONTAINMENT_THRESHOLD on its own, regardless of what each chunk's
# assistant_text (its actual content) says.

def _echo_chunk(chunk_id: str, echoed_question: str, assistant_text: str):
    from synapt.recall.core import TranscriptChunk

    return TranscriptChunk(
        id=chunk_id,
        session_id="echo-session",
        timestamp="2026-03-01T12:00:00Z",
        turn_index=1,
        user_text=f"(context: User previously asked: {echoed_question})",
        assistant_text=assistant_text,
    )


def test_context_echo_alone_does_not_match_but_real_body_overlap_still_does():
    """One witness, both directions: the SAME rich echoed prefix wraps two
    chunks. The one whose ASSISTANT TEXT actually shares the cluster's topic
    matches; the one whose assistant text is genuinely unrelated must not
    match on the echo alone, even though the echo text is deliberately built
    to supply every one of the cluster's signature tokens if it were not
    stripped."""
    from synapt.recall.clustering import (
        MIN_SHARED_SIGNATURE_TOKENS,
        _chunk_tokens,
        _match_existing_cluster,
        _signature_cross_cluster_df,
    )

    signature = frozenset(_TOPIC_WORDS)
    assert len(signature) >= MIN_SHARED_SIGNATURE_TOKENS

    # The echoed question is a verbatim restatement of the topic words --
    # exactly what an unstripped echo would leak into BOTH sub-chunks below.
    rich_echo = "question about " + " ".join(_TOPIC_WORDS)

    on_topic = _echo_chunk(
        "echo-session:t2", rich_echo,
        "answer about " + " ".join(_TOPIC_WORDS),
    )
    off_topic = _echo_chunk(
        "echo-session:t3", rich_echo,
        _DISJOINT_SINGLETON_TEXT[0][1],  # giraffes/savanna, shares nothing
    )

    # Decoys give the topic words a uniform, non-degenerate IDF (see
    # _decoy_signatures) so this test still isolates the echo-stripping
    # behavior it's named for, not the distinctiveness weighting.
    cluster_signatures = {"clust-topic": signature, **_decoy_signatures(50)}
    signature_df = _signature_cross_cluster_df(cluster_signatures)

    on_topic_tokens = _chunk_tokens(on_topic)
    off_topic_tokens = _chunk_tokens(off_topic)

    # Sanity: if the echo were NOT stripped, the off-topic chunk's raw
    # tokens (echo + giraffe body) would still contain every topic word and
    # trivially clear the floor -- proving this test can actually fail.
    unstripped_off_topic = set(rich_echo.lower().split()) | off_topic_tokens
    assert len(signature & unstripped_off_topic) >= MIN_SHARED_SIGNATURE_TOKENS

    assert _match_existing_cluster(
        on_topic_tokens, cluster_signatures, signature_df,
    ) == "clust-topic", (
        "a chunk whose own assistant text shares the topic must still match"
    )
    assert _match_existing_cluster(
        off_topic_tokens, cluster_signatures, signature_df,
    ) is None, (
        "a chunk that shares ONLY the echoed context, not its own content, "
        "must not match -- the echo must be stripped before comparison"
    )


# Hand-read, second round: an editor's "User selected: <path>
# N→<code>" line-selection echo puts bare decimal line numbers into the
# token stream, and two entirely unrelated files that both happen to start
# a selection around the same line number then share a run of "distinctive"
# tokens that are really just consecutive integers. Measured: one real
# cluster signature was 64/64 bare numbers, no topical content at all.

def test_chunk_tokens_drops_bare_numbers_from_a_line_numbered_dump():
    """The general rule, not a shape-specific strip: a content token has at
    least one letter. Applied at _chunk_tokens, this also covers digit
    ranges and digit-arrow-digit shapes for free, since _tokenize already
    treats '-' and the arrow as separators -- "100-200" and "100→200"
    both arrive as separate bare-digit tokens already, not one joined one."""
    from synapt.recall.clustering import _chunk_tokens
    from synapt.recall.core import TranscriptChunk

    chunk = TranscriptChunk(
        id="numbered:t1", session_id="numbered-session",
        timestamp="2026-03-01T12:00:00Z", turn_index=1,
        user_text="",
        assistant_text=(
            "User selected: src/lib.rs\n"
            "100→fn migrate_gripspace(repos: Vec<String>) -> Result<()> {\n"
            "101→    let parsed = parsed_repos.push(repo);\n"
            "102-105→    Ok(())\n"
        ),
    )
    tokens = _chunk_tokens(chunk)
    bare_numbers = {t for t in tokens if t.isdigit()}
    assert bare_numbers == set(), f"bare line-number tokens must not survive: {bare_numbers}"
    assert "migrate_gripspace" in tokens or "parsed_repos.push" in tokens, (
        "real code identifiers from the same dump must still come through"
    )


def test_signature_of_bare_numbers_is_never_a_merge_target():
    """Direct test of the eligibility guard in _match_existing_cluster,
    not just the _chunk_tokens fix: a signature PERSISTED BEFORE the
    letter-bearing filter existed can still be sitting on disk as pure
    line-number tokens until its next backfill (measured: 5.0% of a real
    store's rebuilt signatures still carried 5+ bare-numeric tokens right
    after a full rebuild, one was 64/64). Hand-built here, not through
    _chunk_tokens, because _chunk_tokens' own fix already keeps a FRESH
    chunk from ever producing such tokens again -- going through the real
    pipeline would prove only the tokenizer fix, not this separate guard."""
    from synapt.recall.clustering import (
        MIN_SHARED_SIGNATURE_TOKENS,
        _match_existing_cluster,
        _signature_cross_cluster_df,
    )

    numeric_signature = frozenset(str(n) for n in range(100, 100 + MIN_SHARED_SIGNATURE_TOKENS + 20))
    # Shares EVERY signature token plus real, unrelated words -- old floor
    # (>= 8 shared) and old ratio (shared/len(signature) == 1.0) both pass.
    chunk_tokens = numeric_signature | {"giraffe", "savanna", "acacia"}

    cluster_signatures = {"clust-numeric": numeric_signature}
    assert _match_existing_cluster(
        chunk_tokens, cluster_signatures, _signature_cross_cluster_df(cluster_signatures),
    ) is None, "a signature with no letter-bearing tokens must never be a merge target"


def test_containment_ratio_is_enforced_independently_of_the_absolute_floor():
    """R2 finding: setting CONTAINMENT_THRESHOLD to 0.0 still passed every
    existing test in this file -- the 8-token absolute floor alone blocked
    every fixture pair on its own, so nothing exercised the RATIO gate's
    own discriminating power once the floor is already cleared. A 64-token,
    all-letter-bearing signature (so the letter-eligibility guard is not
    what's under test here): sharing exactly 8 tokens (0.125 -- clears the
    floor, below the 0.20 ratio) must not match; sharing 16 (0.25 -- clears
    both) must."""
    from synapt.recall.clustering import (
        CONTAINMENT_THRESHOLD,
        MIN_SHARED_SIGNATURE_TOKENS,
        _match_existing_cluster,
        _signature_cross_cluster_df,
    )

    signature = frozenset(f"topicword{i}" for i in range(64))
    assert len(signature) == 64

    # Decoys give every "topicword*" token document frequency 1 (present
    # only in the signature under test) and therefore a uniform per-token
    # IDF -- under uniform IDF the weighted containment ratio reduces
    # exactly to the plain count ratio this test is named for.
    cluster_signatures = {"clust-x": signature, **_decoy_signatures(50)}
    signature_df = _signature_cross_cluster_df(cluster_signatures)

    shared_at_floor = sorted(signature)[:MIN_SHARED_SIGNATURE_TOKENS]
    low_ratio_tokens = frozenset(shared_at_floor) | {"unrelated1", "unrelated2"}
    assert len(low_ratio_tokens & signature) == MIN_SHARED_SIGNATURE_TOKENS
    assert MIN_SHARED_SIGNATURE_TOKENS / len(signature) < CONTAINMENT_THRESHOLD, (
        "fixture assumption: 8/64 must sit BELOW the ratio threshold"
    )

    shared_above_ratio = sorted(signature)[: MIN_SHARED_SIGNATURE_TOKENS * 2]
    high_ratio_tokens = frozenset(shared_above_ratio) | {"unrelated1", "unrelated2"}
    assert len(high_ratio_tokens & signature) == MIN_SHARED_SIGNATURE_TOKENS * 2
    assert (MIN_SHARED_SIGNATURE_TOKENS * 2) / len(signature) >= CONTAINMENT_THRESHOLD, (
        "fixture assumption: 16/64 must sit AT OR ABOVE the ratio threshold"
    )

    assert _match_existing_cluster(
        low_ratio_tokens, cluster_signatures, signature_df,
    ) is None, (
        "8/64 clears the absolute floor but not the containment ratio -- must not match"
    )
    assert _match_existing_cluster(
        high_ratio_tokens, cluster_signatures, signature_df,
    ) == "clust-x", (
        "16/64 clears both the floor and the ratio -- must match"
    )


# 10-sample hand read of live merge_into_existing candidates (recall#435
# follow-on, tracked privately; candidate-side lane, unblocked once the
# signature tiebreak fix converged): 5 of 10 shared mostly vocabulary
# common across MANY other clusters'
# signatures -- a recurring harness/status prompt's own words -- and clear
# BOTH the count floor and the containment ratio on that shared vocabulary
# alone, despite having no real topical relationship to the cluster they'd
# join. The other 5 share genuinely rare, cluster-specific vocabulary. The
# two tests below reproduce each shape directly against _match_existing_cluster.

def test_common_cross_cluster_tokens_do_not_merge_despite_clearing_count_and_ratio():
    """A candidate that shares an ENTIRE signature made of tokens recurring
    across many OTHER clusters' signatures too (the shape of a harness/
    status-update prompt's own vocabulary) must not merge, even though the
    raw count (20 shared, well above the floor) and containment ratio
    (1.0, well above the threshold) both clear comfortably on their own --
    exactly the false-merge shape a count/ratio-only check cannot see."""
    from synapt.recall.clustering import (
        MIN_SHARED_SIGNATURE_TOKENS,
        _match_existing_cluster,
        _signature_cross_cluster_df,
    )

    boilerplate = frozenset(f"ritualword{i}" for i in range(20))
    assert len(boilerplate) >= MIN_SHARED_SIGNATURE_TOKENS
    target_signature = boilerplate

    # 30 OTHER clusters share the SAME boilerplate vocabulary -- a
    # recurring prompt appearing across many unrelated real-topic clusters
    # -- plus their own disjoint filler, so they aren't literal duplicate
    # signatures, just clusters that also happen to carry this vocabulary.
    other_clusters = {
        f"clust-other-{i}": boilerplate | frozenset(f"filler{i}_{j}" for j in range(10))
        for i in range(30)
    }
    cluster_signatures = {"clust-target": target_signature, **other_clusters}
    signature_df = _signature_cross_cluster_df(cluster_signatures)

    candidate_tokens = boilerplate | {"unrelated_word_a", "unrelated_word_b"}
    assert len(candidate_tokens & target_signature) == len(boilerplate), (
        "fixture assumption: the candidate shares the WHOLE target signature"
    )

    assert _match_existing_cluster(
        candidate_tokens, cluster_signatures, signature_df,
    ) is None, (
        "shared tokens that recur across many OTHER clusters' signatures "
        "carry no topical signal, however many are shared or how high the "
        "raw containment ratio reads"
    )


def test_genuinely_rare_shared_tokens_still_merge():
    """The other half of the same fixture shape: a candidate sharing
    vocabulary specific to ONLY the target cluster's signature (present in
    no other cluster) must still merge -- the distinctiveness check
    preserves exactly the matches it exists to keep, not just the ones it
    exists to block."""
    from synapt.recall.clustering import (
        _match_existing_cluster,
        _signature_cross_cluster_df,
    )

    distinctive = frozenset(f"specificterm{i}" for i in range(10))
    target_signature = distinctive
    decoys = _decoy_signatures(30)
    cluster_signatures = {"clust-target": target_signature, **decoys}
    signature_df = _signature_cross_cluster_df(cluster_signatures)

    candidate_tokens = distinctive | {"chunk_only_word"}

    assert _match_existing_cluster(
        candidate_tokens, cluster_signatures, signature_df,
    ) == "clust-target", (
        "tokens specific to a single cluster's signature are exactly the "
        "distinctiveness signal this check exists to preserve a match on"
    )


# Real-store hand-read finding: a cluster
# whose members are mostly one recurring HARNESS/CRON/RITUAL prompt (e.g. a
# nightly close-shop checklist, or a wake-loop's "check #dev channel"
# instruction) gets a persisted signature dominated by that prompt's own
# generic process vocabulary (branch, merge, PRs, ready...) -- present in
# nearly every member purely because the prompt repeats verbatim, not
# because it is this cluster's real topic. Measured on the real store: of
# 300 sampled signed clusters, 128 (42.7%) had at least 80% of their members
# share byte-identical user_text. The batch-derived boilerplate stoplist in
# ``recluster_stale_chunks`` cannot see this -- it is computed from the
# INCOMING BATCH, and this pollution lives in the PERSISTED CLUSTER side.

def _cluster_member_chunk(chunk_id: str, user_text: str, assistant_text: str):
    from synapt.recall.core import TranscriptChunk

    return TranscriptChunk(
        id=chunk_id,
        session_id="dup-session",
        timestamp="2026-03-01T12:00:00Z",
        turn_index=1,
        user_text=user_text,
        assistant_text=assistant_text,
    )


def test_strip_cluster_duplicate_user_text_drops_a_recurring_ritual_prompt():
    """Four members, three sharing one byte-identical ritual prompt as their
    user_text (75% >= the 50% majority floor) and one with its own distinct
    user_text. The three ritual-prompt members lose their user_text for
    signature purposes; the odd one out keeps its own."""
    from synapt.recall.clustering import _strip_cluster_duplicate_user_text

    ritual = "10PM CLOSE SHOP -- Stromus. Zero Agent-tool calls. DANGLING-WORK SWEEP."
    chunks = [
        _cluster_member_chunk("c1", ritual, "Close-shop routine starting. First the sweep."),
        _cluster_member_chunk("c2", ritual, "Executing the routine, token-preservation mode."),
        _cluster_member_chunk("c3", ritual, "Sweep clean, syncing the .msg now."),
        _cluster_member_chunk("c4", "what's the status on the release?", "Everything shipped."),
    ]

    stripped = _strip_cluster_duplicate_user_text(chunks)
    by_id = {c.id: c for c in stripped}

    for cid in ("c1", "c2", "c3"):
        assert by_id[cid].user_text == "", (
            f"{cid}'s user_text is the 75%-majority ritual prompt and must be dropped"
        )
        assert by_id[cid].assistant_text == chunks[["c1", "c2", "c3"].index(cid)].assistant_text, (
            "assistant_text (the actual varying content) must be untouched"
        )
    assert by_id["c4"].user_text == "what's the status on the release?", (
        "the one member with its OWN distinct user_text must be untouched"
    )


def test_strip_cluster_duplicate_user_text_leaves_small_clusters_alone():
    """Two members sharing an identical, genuinely topical user_text (not a
    generic ritual prompt) must NOT be stripped: below
    MIN_CLUSTER_SIZE_FOR_DUPLICATE_GUARD, two members sharing text is as
    likely to be real coincidental topical overlap as it is to be a
    recurring harness prompt, and this guard only has evidence once there
    are enough members to call something a majority."""
    from synapt.recall.clustering import (
        MIN_CLUSTER_SIZE_FOR_DUPLICATE_GUARD,
        _strip_cluster_duplicate_user_text,
    )

    shared = "please review PR #400"
    chunks = [
        _cluster_member_chunk("c1", shared, "LGTM, merged."),
        _cluster_member_chunk("c2", shared, "One nit, otherwise fine."),
    ]
    assert len(chunks) < MIN_CLUSTER_SIZE_FOR_DUPLICATE_GUARD, (
        "fixture assumption: below the minimum cluster size for this guard"
    )

    stripped = _strip_cluster_duplicate_user_text(chunks)
    by_id = {c.id: c for c in stripped}
    assert by_id["c1"].user_text == shared
    assert by_id["c2"].user_text == shared


def test_strip_cluster_duplicate_user_text_below_majority_floor_is_untouched():
    """Three members, no single user_text shared by a majority (each
    distinct) -- the ordinary, non-polluted case. Nothing is stripped."""
    from synapt.recall.clustering import _strip_cluster_duplicate_user_text

    chunks = [
        _cluster_member_chunk("c1", "question one", "answer one"),
        _cluster_member_chunk("c2", "question two", "answer two"),
        _cluster_member_chunk("c3", "question three", "answer three"),
    ]
    stripped = _strip_cluster_duplicate_user_text(chunks)
    for original, result in zip(chunks, stripped):
        assert result.user_text == original.user_text


def test_tokenize_cluster_members_signature_drops_ritual_prompt_vocabulary():
    """End-to-end through the tokenizer + signature builder: with the
    ritual-prompt user_text stripped, the resulting signature's top tokens
    come from the varying assistant_text (the real content), not from the
    prompt's own generic process vocabulary."""
    from synapt.recall.clustering import _cluster_signature_tokens, _tokenize_cluster_members

    ritual = ("10PM CLOSE SHOP -- Stromus rhythm. Zero Agent-tool calls. "
              "DANGLING-WORK SWEEP verified branch merge prs ready.")
    chunks = [
        _cluster_member_chunk("c1", ritual, "baseline running giraffe savanna experiment"),
        _cluster_member_chunk("c2", ritual, "giraffe savanna experiment baseline nearly done"),
        _cluster_member_chunk("c3", ritual, "savanna baseline experiment giraffe results in"),
        _cluster_member_chunk("c4", ritual, "giraffe experiment baseline savanna concluded"),
    ]

    token_sets = _tokenize_cluster_members(chunks)
    signature = _cluster_signature_tokens(token_sets)

    for ritual_only_word in ("sweep", "dangling", "prs", "branch", "merge"):
        assert ritual_only_word not in signature, (
            f"{ritual_only_word!r} comes only from the stripped ritual prompt "
            f"and must not reach the signature: {signature}"
        )
    for real_word in ("giraffe", "savanna", "baseline", "experi"):  # "experiment" stems to "experi"
        assert real_word in signature, (
            f"{real_word!r} is the cluster's real, varying content and must "
            f"survive into the signature: {signature}"
        )


# Second real-store finding from the same morning follow-on: a cluster's
# persisted signature computed BEFORE a tokenization fix landed (the
# context-echo strip, the ritual-prompt strip above) keeps carrying that
# pollution forever -- ``backfill_cluster_signatures`` only fills in
# ABSENT signatures, so an already-signed-but-polluted cluster is never
# revisited. ``redrive_cluster_signatures`` recomputes EVERY currently
# signed cluster from its actual current members with the CURRENT
# tokenizer, and only writes where the recomputed value actually differs
# -- itself a WRITE to the store, so it gets the same dry-run-with-a-count
# discipline as the merge lane.

def test_redrive_cluster_signatures_dry_run_reports_without_writing(tmp_path):
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.clustering import redrive_cluster_signatures

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _topic_transcript(source / "topic.jsonl", turns=8)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    db = _open_db(project)
    try:
        cluster_id = db._conn.execute(
            "SELECT cluster_id FROM clusters WHERE cluster_type = 'topic'"
        ).fetchone()[0]

        # Simulate a signature computed by a since-fixed, buggy mechanism:
        # inject a marker token that a fresh recomputation could not produce.
        polluted = ["polluted_marker_token", "another_stale_token"]
        db.save_cluster_token_signature(cluster_id, polluted, "2026-01-01T00:00:00Z")
        assert db.load_cluster_token_signatures()[cluster_id] == set(polluted)

        receipt = redrive_cluster_signatures(db, dry_run=True)
        assert receipt["dry_run"] is True
        assert receipt["clusters_checked"] >= 1, receipt
        assert receipt["clusters_changed"] >= 1, receipt
        assert cluster_id in receipt["changed_cluster_ids"], receipt

        # No write happened: the polluted signature is still exactly what
        # was stored, byte for byte -- verify by fruit, not by the flag.
        assert db.load_cluster_token_signatures()[cluster_id] == set(polluted), (
            "dry_run must not have written anything"
        )
    finally:
        db.close()


def test_redrive_cluster_signatures_writes_the_recomputed_signature(tmp_path):
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.clustering import redrive_cluster_signatures

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _topic_transcript(source / "topic.jsonl", turns=8)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    db = _open_db(project)
    try:
        cluster_id = db._conn.execute(
            "SELECT cluster_id FROM clusters WHERE cluster_type = 'topic'"
        ).fetchone()[0]
        polluted = ["polluted_marker_token"]
        db.save_cluster_token_signature(cluster_id, polluted, "2026-01-01T00:00:00Z")

        receipt = redrive_cluster_signatures(db, dry_run=False)
        assert receipt["dry_run"] is False
        assert cluster_id in receipt["changed_cluster_ids"], receipt

        rewritten = db.load_cluster_token_signatures()[cluster_id]
        assert "polluted_marker_token" not in rewritten, (
            "the stale marker must be gone after a real (non-dry) redrive"
        )
        assert len(rewritten) >= 8, rewritten
    finally:
        db.close()


def test_redrive_cluster_signatures_leaves_unchanged_signatures_alone(tmp_path):
    """A cluster whose stored signature already matches a fresh
    recomputation is reported as unchanged, not as changed -- the receipt
    must distinguish real drift from a no-op re-derivation."""
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.clustering import redrive_cluster_signatures

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _topic_transcript(source / "topic.jsonl", turns=8)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    db = _open_db(project)
    try:
        cluster_id = db._conn.execute(
            "SELECT cluster_id FROM clusters WHERE cluster_type = 'topic'"
        ).fetchone()[0]
        # Freshly built by the current tokenizer already -- redrive should
        # find nothing to change.
        before = db.load_cluster_token_signatures()[cluster_id]

        receipt = redrive_cluster_signatures(db, dry_run=False)
        assert cluster_id not in receipt["changed_cluster_ids"], receipt
        assert receipt["clusters_unchanged"] >= 1, receipt

        after = db.load_cluster_token_signatures()[cluster_id]
        assert after == before
    finally:
        db.close()


def test_redrive_cluster_signatures_refuses_above_ceiling(tmp_path):
    from synapt.recall.cli import _archive_and_build
    from synapt.recall.clustering import redrive_cluster_signatures

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _topic_transcript(source / "topic.jsonl", turns=8)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    db = _open_db(project)
    try:
        receipt = redrive_cluster_signatures(db, refuse_above=0, dry_run=True)
        assert receipt["refused"] is True, receipt
        assert receipt["clusters_checked"] == 0, receipt
    finally:
        db.close()


def test_recluster_merge_help_text_describes_the_shipped_mechanism():
    """Sentinel's carried finding (m_8a65239d): --recluster-merge's help
    text described the abandoned Jaccard-against-search_text mechanism
    (measured and rejected -- see recluster_stale_chunks' own docstring),
    not the containment-of-a-persisted-signature mechanism that actually
    shipped. Source-string regression, not a behavior test: this is
    documentation drift, not logic."""
    cli_source = (Path(__file__).parents[2] / "src" / "synapt" / "recall" / "cli.py").read_text()
    assert "Jaccard-matched against cluster search_text" not in cli_source, (
        "the abandoned mechanism's wording must not still be in the help text"
    )
    assert "containment of the cluster's persisted top-64" in cli_source, (
        "the help text must describe the actual shipped matching mechanism"
    )


# Real-store finding: a live redrive's second dry_run pass, in a fresh
# process, reported hundreds of clusters as "changed" with nothing between
# the two passes but a process restart. Root cause: _cluster_signature_tokens
# used Counter.most_common(top_n), whose tie-break at the cutoff falls back
# to Counter's insertion order -- which comes from iterating the input
# set[str] objects, and Python randomizes string-hash-derived set iteration
# order per process. A pytest run (one process, one hash seed) can never see
# this: the same input always produces the same output WITHIN a process.
# The only witness that can see it is two SEPARATE processes with two
# DIFFERENT hash seeds computing the same signature and comparing.

_SIGNATURE_TIEBREAK_SUBPROCESS_SCRIPT = """
import json
from synapt.recall.clustering import _cluster_signature_tokens

# Reproduces the real shape measured on the store: 34 tokens with count=2
# (unambiguous top ranks), 74 tokens with count=1 all tied at the count
# sitting exactly at the top_n=64 cutoff -- 30 of those 74 must be chosen
# arbitrarily unless the tie-break is deterministic.
hi = [f"hi{i}" for i in range(34)]
lo = [f"lo{i}" for i in range(74)]
set_a = set(hi) | set(lo[:37])
set_b = set(hi) | set(lo[37:])
signature = _cluster_signature_tokens([set_a, set_b], top_n=64)
print(json.dumps(signature))
"""


def _run_signature_tiebreak_subprocess(hashseed: str) -> list[str]:
    import json
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    result = subprocess.run(
        [sys.executable, "-c", _SIGNATURE_TIEBREAK_SUBPROCESS_SCRIPT],
        env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    return json.loads(result.stdout)


def test_cluster_signature_tiebreak_is_stable_across_process_hash_seeds():
    """The witness that a same-process mutation test cannot provide: compute
    the SAME tied-boundary signature in two separate subprocesses with two
    DIFFERENT PYTHONHASHSEED values and require byte-identical output. This
    is the actual property _cluster_signature_tokens must have (a pure
    function of the token multiset, not of that process's hash seed) --
    reverting the (-count, token) sort back to Counter.most_common(top_n)
    makes this test fail (verified by hand: seed 0 and seed 1 disagree on
    which lo-tokens fill the last several slots)."""
    sig_seed0 = _run_signature_tiebreak_subprocess("0")
    sig_seed1 = _run_signature_tiebreak_subprocess("1")

    assert sig_seed0 == sig_seed1, (
        "the signature must not depend on the process's hash seed -- "
        f"seed 0 gave {sig_seed0}, seed 1 gave {sig_seed1}"
    )
    assert len(sig_seed0) == 64, sig_seed0
    # The 34 unambiguous high-count tokens must always be present.
    assert all(f"hi{i}" in sig_seed0 for i in range(34)), sig_seed0
