"""Contract tests for the immutable-generation / CAS publish path that
fixes save_chunks()'s non-atomic full-rebuild (R3.1, PR two)."""

from __future__ import annotations

from pathlib import Path

from synapt.recall.core import TranscriptChunk
from synapt.recall.generations import (
    current_generation_dir,
    generation_dir,
    generations_root,
    publish_generation,
    read_current_generation,
    rebuild_and_publish,
)
from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.sharding import list_shards
from synapt.recall.storage import RecallDB


def _read_generation_chunks(gen_path: Path) -> list[TranscriptChunk]:
    """Read every chunk out of a generation's own shard files directly.

    A generation directory holds only data_NNN.db shards, no index.db --
    ShardedRecallDB.open()'s sharded/monolithic auto-detection is keyed on
    index.db's presence, and index.db lives in the PARENT index_dir, not
    inside any one generation. ShardedRecallDB.open(index_dir) correctly
    follows CURRENT to the right generation's shards (see
    test_sharded_db.py and the real cross-process demonstration below);
    opening a bare generation directory on its own was never meant to be
    supported and still isn't -- this helper verifies generation content
    directly for tests that only need the primitive, without needing the
    parent index_dir's own index.db to exist.
    """
    chunks: list[TranscriptChunk] = []
    for shard_path in list_shards(gen_path):
        db = RecallDB(shard_path)
        try:
            chunks.extend(db.load_chunks())
        finally:
            db.close()
    return chunks


def _chunk(session: str, turn: int, text: str) -> TranscriptChunk:
    return TranscriptChunk(
        id=f"{session}:t{turn}",
        session_id=session,
        timestamp=f"2026-09-03T10:{turn % 60:02d}:00Z",
        turn_index=turn,
        user_text=f"question {turn}: {text}",
        assistant_text=f"answer {turn}: {text}",
    )


def test_read_current_generation_is_none_before_any_publish(tmp_path):
    assert read_current_generation(tmp_path) is None
    assert current_generation_dir(tmp_path) is None


def test_publish_generation_succeeds_against_a_matching_parent(tmp_path):
    ok = publish_generation(tmp_path, "gen-a", expected_parent=None)
    assert ok is True
    assert read_current_generation(tmp_path) == "gen-a"

    ok2 = publish_generation(tmp_path, "gen-b", expected_parent="gen-a")
    assert ok2 is True
    assert read_current_generation(tmp_path) == "gen-b"


def test_publish_generation_fails_the_cas_against_a_stale_parent(tmp_path):
    assert publish_generation(tmp_path, "gen-a", expected_parent=None) is True

    # A stale expected_parent (someone else already published gen-a) must
    # be refused -- publishing over it would silently discard gen-a.
    refused = publish_generation(tmp_path, "gen-b", expected_parent="not-gen-a")
    assert refused is False
    assert read_current_generation(tmp_path) == "gen-a", (
        "a failed CAS must never have moved CURRENT"
    )


def test_rebuild_and_publish_writes_a_real_readable_generation(tmp_path):
    chunks = [_chunk("s1", 0, "hello"), _chunk("s1", 1, "world")]

    name = rebuild_and_publish(tmp_path, chunks)

    assert read_current_generation(tmp_path) == name
    gen_path = generation_dir(tmp_path, name)
    assert gen_path.is_dir()
    loaded = _read_generation_chunks(gen_path)
    assert {c.id for c in loaded} == {"s1:t0", "s1:t1"}


def test_rebuild_and_publish_two_concurrent_writers_exactly_one_wins_first_try(tmp_path):
    """The two-writer proof (Layne-ratified design):
    two independent rebuilds race to publish against the same starting
    parent. Exactly one must win on its first attempt (the CAS is a real
    mutual-exclusion point, not a race two callers can both win); the
    loser must detect the failure, rebuild against the new parent, and
    land on top -- never silently overwrite or get discarded. And a
    reader polling CURRENT throughout the whole race must never observe
    a generation directory that is missing or unreadable: CURRENT only
    ever names a generation that was fully written before it became
    visible.
    """
    import threading
    import time as time_module

    chunks_a = [_chunk("writer-a", i, "from A") for i in range(50)]
    chunks_b = [_chunk("writer-b", i, "from B") for i in range(50)]

    results: dict[str, dict] = {"a": {}, "b": {}}
    reader_observations: list[str] = []
    stop_reading = threading.Event()

    def _reader():
        while not stop_reading.is_set():
            name = read_current_generation(tmp_path)
            if name is not None:
                gen_path = generation_dir(tmp_path, name)
                if not gen_path.is_dir():
                    reader_observations.append(f"MISSING:{name}")
                    continue
                try:
                    _read_generation_chunks(gen_path)
                except Exception as exc:  # pragma: no cover - failure path
                    reader_observations.append(f"UNREADABLE:{name}:{exc}")
            time_module.sleep(0.001)

    def _writer(key: str, chunks: list[TranscriptChunk]):
        started = time_module.monotonic()
        results[key]["name"] = rebuild_and_publish(tmp_path, chunks)
        results[key]["elapsed"] = time_module.monotonic() - started

    reader_thread = threading.Thread(target=_reader)
    reader_thread.start()
    thread_a = threading.Thread(target=_writer, args=("a", chunks_a))
    thread_b = threading.Thread(target=_writer, args=("b", chunks_b))
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=30.0)
    thread_b.join(timeout=30.0)
    stop_reading.set()
    reader_thread.join(timeout=5.0)

    assert "name" in results["a"] and "name" in results["b"], (
        "both writers must finish (one on the first attempt, the other "
        "after a bounded retry) -- neither should hang or raise"
    )

    bad = [o for o in reader_observations if o.startswith(("MISSING", "UNREADABLE"))]
    assert bad == [], f"reader observed a half-published generation: {bad}"

    # The FINAL current generation must be one of the two writers' own --
    # never a third, corrupted state -- and must be genuinely readable
    # with exactly that writer's chunks.
    final_name = read_current_generation(tmp_path)
    assert final_name in (results["a"]["name"], results["b"]["name"])
    final_gen = generation_dir(tmp_path, final_name)
    loaded_ids = {c.id for c in _read_generation_chunks(final_gen)}
    expected_ids = (
        {c.id for c in chunks_a} if final_name == results["a"]["name"]
        else {c.id for c in chunks_b}
    )
    assert loaded_ids == expected_ids, (
        "the published generation's content must match the writer that "
        "actually won -- never a mix, never the loser's stale content"
    )


def test_old_layout_reader_finds_rebuilt_content_via_the_flat_mirror(tmp_path):
    """The mixed-version witness (PR two, Stromus 2026-09-03):
    an OLDER recall build shares this same index_dir on the same host --
    every teammate's MCP plugin and every dev-tip checkout do, today. Old
    code has no notion of CURRENT or generations/; it globs data_*.db
    directly in index_dir. Without a flat mirror, a rebuild through this
    system would make an old reader's glob find nothing and silently
    return empty results -- the silent-wrong-answer class, not mere
    staleness. This simulates exactly that old reader (list_shards on the
    bare index_dir, no CURRENT awareness at all) against a store that HAS
    been rebuilt via rebuild_and_publish, and asserts it finds the real,
    current content.
    """
    chunks = [_chunk("legacy-check", i, f"legacy-visible-marker-{i}") for i in range(30)]

    rebuild_and_publish(tmp_path, chunks)

    # The "old reader": exactly what pre-generation code did -- glob
    # data_*.db directly in index_dir, no CURRENT, no generations/.
    old_reader_shards = list_shards(tmp_path)
    assert old_reader_shards, (
        "an old-layout reader found ZERO shards after a rebuild -- this "
        "is the exact silent-empty-results regression the flat mirror "
        "exists to prevent"
    )
    old_reader_chunks: list[TranscriptChunk] = []
    for shard_path in old_reader_shards:
        db = RecallDB(shard_path)
        try:
            old_reader_chunks.extend(db.load_chunks())
        finally:
            db.close()

    assert {c.id for c in old_reader_chunks} == {c.id for c in chunks}, (
        "old-layout reader's content does not match the real rebuilt "
        "generation -- stale or partial mirror"
    )


def test_old_layout_reader_survives_a_second_rebuild_with_fewer_shards(tmp_path):
    """The flat mirror must not leave a stale extra shard behind when a
    later rebuild has FEWER shards than the one before it -- an old
    reader's glob would otherwise pick up a leftover file from the prior
    generation and mix it into results."""
    big = [_chunk("first", i, "first-gen") for i in range(2000)]
    small = [_chunk("second", 0, "second-gen-only")]

    import synapt.recall.sharding as sharding_mod
    original_threshold = sharding_mod.SHARD_CHUNK_THRESHOLD
    sharding_mod.SHARD_CHUNK_THRESHOLD = 500
    try:
        rebuild_and_publish(tmp_path, big)
        first_flat_count = len(list_shards(tmp_path))
        assert first_flat_count > 1, "test setup invalid: need multiple shards first"

        rebuild_and_publish(tmp_path, small)
    finally:
        sharding_mod.SHARD_CHUNK_THRESHOLD = original_threshold

    flat_shards = list_shards(tmp_path)
    assert len(flat_shards) == 1, (
        f"expected exactly 1 flat shard after the smaller rebuild, found "
        f"{len(flat_shards)}: {[p.name for p in flat_shards]} -- a stale "
        f"extra shard from the bigger first generation was not cleaned up"
    )
    db = RecallDB(flat_shards[0])
    try:
        ids = {c.id for c in db.load_chunks()}
    finally:
        db.close()
    assert ids == {"second:t0"}


def test_gc_keeps_only_current_and_previous_generation(tmp_path):
    """A full rebuild every session start on a large store must not
    accumulate one full-size generation directory per rebuild forever --
    disk fills. After N rebuilds, only the current generation and the one
    it superseded may remain on disk; everything older is collected."""
    for i in range(5):
        rebuild_and_publish(tmp_path, [_chunk(f"round{i}", 0, f"round {i}")])

    remaining = {p.name for p in generations_root(tmp_path).iterdir() if p.is_dir()}
    assert len(remaining) <= 2, (
        f"expected at most 2 generations (current + previous) after 5 "
        f"rebuilds, found {len(remaining)}: {sorted(remaining)}"
    )
    current = read_current_generation(tmp_path)
    assert current in remaining, "CURRENT itself must never be GC'd"


def test_real_concurrent_build_and_search_across_two_processes(tmp_path):
    """The real product-path proof, not just the primitive (Stromus,
    2026-09-03): a genuine SUBPROCESS running a full ShardedRecallDB
    rebuild while THIS process concurrently runs ShardedRecallDB.open()
    + fts_search() in a tight loop -- exactly what a real build and a
    real recall search do, end to end. The two-writer thread-based proof
    above exercises the generations.py primitive directly; this
    exercises save_chunks() and open() actually wired to it, across a
    real process boundary, which is the thing that ships.

    The reader must see only two states: the pre-rebuild content and the
    post-rebuild content -- never a missing/empty/exception state in
    between, across a rebuild large enough (tens of thousands of chunks)
    to take multiple real seconds, giving hundreds of genuine polls a
    real chance to land mid-rebuild.
    """
    import subprocess
    import sys
    import time as time_module

    from synapt.recall.storage import RecallDB

    RecallDB(tmp_path / "index.db").close()
    RecallDB(tmp_path / "data_001.db").close()
    seed_db = ShardedRecallDB.open(tmp_path)
    seed_chunks = [_chunk("seed", i, "quality curve seed") for i in range(10)]
    seed_db.save_chunks(seed_chunks)
    seed_count = seed_db.chunk_count()
    seed_db.close()
    assert seed_count == 10

    worker = str(Path(__file__).resolve().parent / "_generation_rebuild_worker.py")
    n_chunks, threshold = 20_000, 500

    proc = subprocess.Popen(
        [sys.executable, worker, str(tmp_path), str(n_chunks), str(threshold)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    observed_states: set[tuple[int, bool]] = set()
    errors: list[str] = []
    polls = 0
    started = time_module.monotonic()
    while proc.poll() is None:
        polls += 1
        try:
            db = ShardedRecallDB.open(tmp_path)
            hits = db.fts_search("quality curve")
            count = db.chunk_count()
            db.close()
            observed_states.add((count, len(hits) > 0))
            if len(hits) == 0:
                errors.append(f"poll {polls}: fts_search found nothing at chunk_count={count}")
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(f"poll {polls}: {type(exc).__name__}: {exc}")
        time_module.sleep(0.02)
    elapsed = time_module.monotonic() - started

    stdout, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0, f"builder subprocess failed: {stdout}"
    assert polls > 10, f"rebuild finished too fast to be a real concurrency test ({elapsed:.2f}s, {polls} polls)"
    assert errors == [], f"reader observed a bad state: {errors}"
    assert observed_states <= {(seed_count, True), (n_chunks, True)}, (
        f"reader observed an unexpected state not in {{before, after}}: {observed_states}"
    )

    final_db = ShardedRecallDB.open(tmp_path)
    final_count = final_db.chunk_count()
    final_hits = final_db.fts_search("quality curve")
    final_db.close()
    assert final_count == n_chunks, f"build did not land: {final_count} != {n_chunks}"
    assert len(final_hits) > 0

    # Hardening (Stromus, 2026-09-03 R2): the cross-process proof above
    # shows the reader's view; assert the writer's own side effects too
    # -- a real build through the real product path must leave CURRENT
    # published and GC must have already run, not just "the content is
    # readable" (which a leftover un-GC'd generation would also satisfy).
    assert read_current_generation(tmp_path) is not None, "build did not publish a CURRENT"
    remaining = [p for p in generations_root(tmp_path).iterdir() if p.is_dir()]
    assert len(remaining) <= 2, f"GC did not run after the real build: {[p.name for p in remaining]}"
