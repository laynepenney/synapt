"""Immutable generation directories + an atomically-swapped CURRENT
pointer, for the one write path that was never safe under concurrency:
a sharded index's full rebuild (``ShardedRecallDB.save_chunks``'s
data-shard-recreation branch), which today closes every shard
connection, deletes every shard file, then rebuilds from scratch with
no atomicity spanning those steps -- a reader mid-rebuild can see a
missing or partial shard set.

The fix, ratified by Layne: each rebuild writes a brand-new, complete,
self-contained generation of data shards under
``<index_dir>/generations/<name>/`` before it is ever visible to a
reader. Publishing is one atomic ``os.replace`` of a ``CURRENT``
pointer file, guarded by a compare-and-swap on the parent generation a
writer read before it started rebuilding: if some other writer has
already published since then, the CAS fails and the caller rebuilds
again against the new parent (bounded retry), rather than clobbering
newer work or racing to publish over it.

This module covers only the DATA SHARDS (``data_NNN.db``) that
``save_chunks`` currently deletes-and-recreates in place. ``index.db``
(knowledge, clusters, query_tail overlay, shard_metadata) is untouched
and stays outside the generation system for now -- see the module
docstring in ``sharded_db.py`` for the disclosed follow-up scope.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

GENERATIONS_DIRNAME = "generations"
CURRENT_FILENAME = "CURRENT"


def generations_root(index_dir: Path) -> Path:
    """Directory holding every generation, published or orphaned."""
    return index_dir / GENERATIONS_DIRNAME


def new_generation_name() -> str:
    """A fresh, sortable-by-time, collision-safe generation name."""
    return f"gen-{int(time.time())}-{uuid.uuid4().hex[:12]}"


def generation_dir(index_dir: Path, name: str) -> Path:
    return generations_root(index_dir) / name


def read_current_generation(index_dir: Path) -> str | None:
    """Read the CURRENT pointer. None if this index has never published
    a generation (a fresh index, or one not yet migrated onto this
    system -- both are indistinguishable to a reader, and both are
    correctly "no generation yet")."""
    current_path = index_dir / CURRENT_FILENAME
    try:
        content = current_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return content or None


def current_generation_dir(index_dir: Path) -> Path | None:
    """The directory a reader should open right now, or None if no
    generation has ever been published (caller falls back to whatever
    it did before generations existed)."""
    name = read_current_generation(index_dir)
    if name is None:
        return None
    return generation_dir(index_dir, name)


def _sync_flat_layout_for_legacy_readers(index_dir: Path, gen_path: Path) -> None:
    """Mirror the newly-published generation's shards into the flat
    ``index_dir/data_NNN.db`` location too, so a reader running an OLDER
    recall version -- one built before this system existed, which globs
    ``data_*.db`` directly in ``index_dir`` and has no notion of CURRENT
    -- keeps finding real, current shards instead of silently finding
    none and returning empty results (the silent-wrong-answer class,
    not a mere staleness class: multiple recall versions read the same
    store concurrently on this host today).

    A real COPY, not a hard link. Hard links were tried first (cheaper:
    same filesystem, near-instant, no duplicated bytes) and measured
    unsafe: even with the WAL fully checkpointed before closing the
    writer's connection (this module does that -- see rebuild_and_publish),
    two independent PATHS sharing one inode each get their OWN, separate
    ``-wal``/``-shm`` sidecar files under SQLite's WAL mode, which is a
    documented hazard for concurrent access to what SQLite believes are
    two different databases occupying the same bytes -- reproduced
    directly: a concurrent reader hit "disk I/O error" opening a
    hard-linked shard under real contention (the two-writer proof test,
    under full-suite load). A copy has no shared inode and no shared
    WAL-mode bookkeeping at all, so this hazard cannot occur; the cost is
    real I/O instead of a syscall, accepted for correctness.

    Each shard's swap into place is a full copy under a temp name, then
    one atomic ``os.replace`` over the real name, so an old reader's glob
    never observes a partially-written file at any single shard name.
    Across MULTIPLE shard names in one sync, an old reader's own glob
    (which was never atomic to begin with -- a rebuild-in-progress under
    the pre-generation code could already show a partial set) can still
    observe a brief mix of some updated and some still-stale names; this
    is strictly better than the alternative it replaces (returning
    nothing at all), and is the accepted, disclosed bound.

    Best-effort: a failure here (e.g. a disk-full mid-copy) is logged,
    not raised -- it only degrades an OLD reader back to pre-fix
    behavior, and must never fail the publish a NEW reader depends on.
    """
    import shutil as _shutil

    from synapt.recall.sharding import list_shards

    try:
        new_shards = list_shards(gen_path)
        new_names = {p.name for p in new_shards}

        for shard_path in new_shards:
            target = index_dir / shard_path.name
            tmp_copy = index_dir / f".{shard_path.name}.copy-tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            try:
                _shutil.copyfile(shard_path, tmp_copy)
                os.replace(tmp_copy, target)
                # SQLite's WAL header carries a salt that invalidates a
                # stale -wal against a base file it wasn't written
                # against, but don't rely on that alone: an old reader
                # may have left -wal/-shm sidecars here from a PRIOR
                # flat-mirror incarnation. Clear them so the next open at
                # this path starts clean against the file we just wrote.
                for suffix in ("-wal", "-shm"):
                    (index_dir / (shard_path.name + suffix)).unlink(missing_ok=True)
            except OSError:
                tmp_copy.unlink(missing_ok=True)
                raise

        # A smaller new generation must not leave a stale extra shard
        # behind for an old reader's glob to pick up.
        for old_path in list_shards(index_dir):
            if old_path.name not in new_names:
                for suffix in ("", "-wal", "-shm"):
                    p = old_path.parent / (old_path.name + suffix)
                    p.unlink(missing_ok=True)
    except OSError:
        logger.warning(
            "Failed to sync flat-layout shard mirror for legacy readers at %s",
            index_dir, exc_info=True,
        )


def publish_generation(
    index_dir: Path,
    new_name: str,
    *,
    expected_parent: str | None,
    lock_timeout: float = 30.0,
) -> bool:
    """Atomically make *new_name* the CURRENT generation -- but only if
    CURRENT is still *expected_parent* (the generation this writer read
    before it started building *new_name*).

    Returns True on success. Returns False if the CAS failed: some
    other writer already published a different generation since
    *expected_parent* was read, and the caller must rebuild against the
    new parent and retry rather than overwrite newer work.

    The compare-then-replace is not atomic on its own (two writers
    could both read the same old CURRENT before either publishes, and
    both believe they should win) -- the lock is what makes the whole
    read-compare-write one atomic step, which is the entire point of a
    CAS. Held across this fast operation AND the flat-layout mirror
    sync below, so a second writer's own publish can't interleave with
    this one's shard-by-shard mirror update.
    """
    from synapt.recall.cli import _acquire_build_lock, _release_build_lock

    lock_fd = _acquire_build_lock(index_dir, timeout=lock_timeout, name="generation.lock")
    if lock_fd is None:
        return False
    try:
        actual_parent = read_current_generation(index_dir)
        if actual_parent != expected_parent:
            return False

        current_path = index_dir / CURRENT_FILENAME
        tmp_path = index_dir / f".{CURRENT_FILENAME}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        tmp_path.write_text(new_name, encoding="utf-8")
        try:
            os.replace(tmp_path, current_path)  # atomic: same dir, same filesystem
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

        _sync_flat_layout_for_legacy_readers(index_dir, generation_dir(index_dir, new_name))
        return True
    finally:
        _release_build_lock(lock_fd)


def gc_old_generations(index_dir: Path, *, keep: set[str]) -> list[str]:
    """Delete every generation directory under *index_dir* NOT named in
    *keep*. Returns the names actually removed.

    Bound, not a guarantee: a reader that read CURRENT a moment before
    it changed needs that generation to still exist while it opens and
    uses it. Keeping both the current AND the immediately-previous
    generation (the caller's job -- pass both in *keep*) covers the
    realistic case (a reader opens, searches, and closes within one
    call, well inside the time between two rebuilds). A reader that
    stays open across MULTIPLE further rebuilds is not covered -- no
    call pattern in this codebase does that today, and it is the
    residual risk this bound accepts rather than pretends to close.
    """
    root = generations_root(index_dir)
    if not root.is_dir():
        return []
    removed = []
    for entry in root.iterdir():
        if entry.is_dir() and entry.name not in keep:
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
    return removed


def rebuild_and_publish(
    index_dir: Path,
    chunks: list,  # list["TranscriptChunk"], avoiding the import cycle
    *,
    max_retries: int = 3,
) -> str:
    """Build a fresh generation from *chunks* and publish it, retrying
    against a newer parent if another writer wins the race first.

    Returns the name of the generation that ended up CURRENT (this
    writer's own, on any attempt that wins).

    Raises RuntimeError if *max_retries* consecutive CAS failures occur
    -- under real contention this bounds how long one writer can be
    starved by a stream of competing publishes, rather than retrying
    forever.

    On a successful publish, also garbage-collects every generation
    except the new CURRENT and the one it superseded (see
    ``gc_old_generations``) -- a full rebuild every session start must
    not accumulate one full-size generation per rebuild forever.
    """
    from synapt.recall.sharding import (
        SHARD_CHUNK_THRESHOLD,
        shard_name_for_index,
    )
    from synapt.recall.storage import RecallDB

    generations_root(index_dir).mkdir(parents=True, exist_ok=True)

    attempt = 0
    while True:
        attempt += 1
        parent = read_current_generation(index_dir)
        new_name = new_generation_name()
        gen_path = generation_dir(index_dir, new_name)
        gen_path.mkdir(parents=True, exist_ok=False)

        sorted_chunks = sorted(chunks, key=lambda c: c.timestamp or "")
        shard_idx = 1
        for offset in range(0, len(sorted_chunks), SHARD_CHUNK_THRESHOLD):
            batch = sorted_chunks[offset:offset + SHARD_CHUNK_THRESHOLD]
            shard_path = gen_path / shard_name_for_index(shard_idx)
            shard_db = RecallDB(shard_path)
            shard_db.save_chunks(batch)
            # Force everything out of the WAL into the main file before
            # closing: RecallDB.close() does not checkpoint on its own,
            # and a shard file with pending WAL data is not a complete,
            # self-contained database -- reading it via a SEPARATE path
            # later (the flat legacy mirror below) would see incomplete
            # content, or worse, hit SQLite's documented WAL-mode hazard
            # of two independent -wal/-shm sidecars coordinating access
            # to the same underlying bytes. TRUNCATE also empties the WAL
            # file itself, so nothing is left to go stale.
            try:
                shard_db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                logger.warning("wal_checkpoint failed for %s", shard_path, exc_info=True)
            shard_db.close()
            shard_idx += 1

        if publish_generation(index_dir, new_name, expected_parent=parent):
            keep = {new_name} | ({parent} if parent else set())
            gc_old_generations(index_dir, keep=keep)
            return new_name

        # Lost the race: CURRENT moved since `parent` was read. Our
        # generation was never published, so no reader can see it --
        # safe to discard and retry against whatever is CURRENT now.
        shutil.rmtree(gen_path, ignore_errors=True)
        if attempt >= max_retries:
            raise RuntimeError(
                f"rebuild_and_publish: lost the CAS race {attempt} times in a "
                f"row against {index_dir} -- giving up rather than retrying forever"
            )
