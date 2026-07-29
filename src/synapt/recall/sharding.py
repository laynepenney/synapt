"""Sharded database utilities for tree-structured recall storage.

Splits the monolithic recall.db into:
  - index.db: lightweight routing index (knowledge, clusters, metadata,
    access tracking, shard metadata)
  - data_NNN.db: size-based chunk shards (chunks + FTS)

Shards are split by chunk count (default 10K chunks per shard).
Sequential naming (data_001, data_002, ...) with a shard_metadata
table in index.db for time-range queries and stats.

See issue #89 for design.
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARD_CHUNK_THRESHOLD = 10_000
"""Maximum chunks per data shard before splitting to a new one."""


# ---------------------------------------------------------------------------
# Shard naming and discovery
# ---------------------------------------------------------------------------

def shard_name_for_index(index: int) -> str:
    """Return the database filename for a shard index.

    >>> shard_name_for_index(1)
    'data_001.db'
    >>> shard_name_for_index(42)
    'data_042.db'
    """
    return f"data_{index:03d}.db"


_SEQUENTIAL_SHARD_RE = re.compile(r"^data_(\d{3,})\.db$")
"""Matches sequential shard names like data_001.db, data_042.db."""


def list_shards(index_dir: Path) -> list[Path]:
    """List all data shard DBs in the index directory, sorted by name.

    Lexicographic sort gives sequential order for ``data_NNN.db`` names.
    Also discovers legacy ``data_YYYY_qN.db`` shards for backward compat.
    """
    return sorted(index_dir.glob("data_*.db"))


def next_shard_index(index_dir: Path) -> int:
    """Return the next available shard index (1-based).

    Only considers sequential ``data_NNN.db`` shards — legacy
    ``data_YYYY_qN.db`` names are ignored to avoid misparsing.
    """
    max_idx = 0
    for p in index_dir.glob("data_*.db"):
        m = _SEQUENTIAL_SHARD_RE.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def is_sharded(index_dir: Path) -> bool:
    """Check if an index directory uses the sharded layout.

    Returns True if index.db exists (sharded), False if only recall.db
    exists (monolithic).
    """
    return (index_dir / "index.db").exists()


# ---------------------------------------------------------------------------
# Shard metadata schema (lives in index.db)
# ---------------------------------------------------------------------------

SHARD_METADATA_SQL = """\
CREATE TABLE IF NOT EXISTS shard_metadata (
    shard_name    TEXT PRIMARY KEY,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    min_timestamp TEXT,
    max_timestamp TEXT,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    is_active     INTEGER NOT NULL DEFAULT 0
);
"""


def _update_shard_metadata(
    index_conn: sqlite3.Connection,
    shard_path: Path,
    chunk_count: int,
    min_ts: str,
    max_ts: str,
    *,
    is_active: bool = False,
) -> None:
    """Insert or update shard metadata in index.db."""
    size = shard_path.stat().st_size if shard_path.exists() else 0
    index_conn.execute(
        "INSERT OR REPLACE INTO shard_metadata "
        "(shard_name, chunk_count, min_timestamp, max_timestamp, size_bytes, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (shard_path.name, chunk_count, min_ts, max_ts, size, int(is_active)),
    )


# ---------------------------------------------------------------------------
# Index DB tables — these stay in the lightweight always-loaded index.
# ---------------------------------------------------------------------------

_INDEX_TABLES = frozenset({
    "metadata",
    "knowledge",
    "knowledge_fts",
    "knowledge_fts_data",
    "knowledge_fts_idx",
    "knowledge_fts_docsize",
    "knowledge_fts_config",
    "clusters",
    "clusters_fts",
    "clusters_fts_data",
    "clusters_fts_idx",
    "clusters_fts_docsize",
    "clusters_fts_config",
    "cluster_summaries",
    "cluster_chunks",
    "access_log",
    "access_stats",
    "access_log_archive",
    "chunk_links",
    "pending_contradictions",
})

# Data tables — these get sharded by size.
# chunks_fts must co-locate with chunks (FTS5 content-sync triggers
# reference the same DB's chunks table).
_DATA_TABLES = frozenset({
    "chunks",
    "chunks_fts",
    "chunks_fts_data",
    "chunks_fts_idx",
    "chunks_fts_docsize",
    "chunks_fts_config",
})


# ---------------------------------------------------------------------------
# Split monolithic DB into size-based shards
# ---------------------------------------------------------------------------

def estimate_split(db_path: Path, threshold: int = SHARD_CHUNK_THRESHOLD) -> dict[str, int]:
    """Estimate how chunks would be distributed across size-based shards.

    Returns dict mapping shard name to chunk count.
    Returns empty dict if the DB has no chunks table.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if row is None:
            return {}
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total == 0:
            return {}

        plan: dict[str, int] = {"index.db": 0}
        shard_idx = 1
        remaining = total
        while remaining > 0:
            count = min(remaining, threshold)
            plan[shard_name_for_index(shard_idx)] = count
            remaining -= count
            shard_idx += 1
        return plan
    finally:
        conn.close()


def split_monolithic_db(
    index_dir: Path,
    threshold: int = SHARD_CHUNK_THRESHOLD,
    dry_run: bool = False,
) -> dict[str, int]:
    """Split a monolithic recall.db into index.db + size-based data shards.

    Creates:
      - index.db: knowledge, clusters, metadata, shard_metadata
      - data_001.db, data_002.db, ...: chunks grouped by size threshold

    Args:
        index_dir: Directory containing recall.db.
        threshold: Max chunks per shard (default 10,000).
        dry_run: If True, only returns the split plan without writing.

    Returns:
        Dict mapping shard name to chunk count (including "index.db": 0).

    Raises:
        FileNotFoundError: If recall.db doesn't exist.
        RuntimeError: If already sharded.
    """
    db_path = index_dir / "recall.db"
    if not db_path.exists():
        raise FileNotFoundError(f"No recall.db found at {index_dir}")
    if is_sharded(index_dir):
        raise RuntimeError(f"Already sharded at {index_dir}")

    if dry_run:
        return estimate_split(db_path, threshold)

    plan: dict[str, int] = {"index.db": 0}
    tmp_dir = Path(tempfile.mkdtemp(dir=index_dir, prefix=".split_tmp_"))

    src = sqlite3.connect(str(db_path))
    src.row_factory = sqlite3.Row
    try:
        # Step 1: Create index.db with non-chunk tables + shard metadata
        idx_path = tmp_dir / "index.db"
        idx = sqlite3.connect(str(idx_path))
        idx.execute("PRAGMA journal_mode=WAL")
        idx.execute("PRAGMA busy_timeout=5000")
        idx.execute(f"ATTACH DATABASE '{db_path}' AS src")
        try:
            for table_info in src.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall():
                tname = table_info["name"]
                tsql = table_info["sql"]
                if tname in _INDEX_TABLES and tsql and not tname.endswith(
                    ("_data", "_idx", "_docsize", "_config")
                ):
                    idx.execute(tsql)
                    idx.execute(f"INSERT INTO [{tname}] SELECT * FROM src.[{tname}]")
            for idx_info in src.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
            ).fetchall():
                try:
                    idx.execute(idx_info["sql"])
                except sqlite3.OperationalError:
                    pass
            # Create shard metadata table
            idx.execute(SHARD_METADATA_SQL)
            idx.commit()
        finally:
            idx.execute("DETACH DATABASE src")
            idx.close()

        # Step 2: Create data shards by chunk count (ordered by timestamp)
        col_names = [
            desc[0]
            for desc in src.execute("SELECT * FROM chunks LIMIT 0").description
        ]
        all_chunks = src.execute(
            "SELECT * FROM chunks ORDER BY timestamp"
        ).fetchall()

        chunks_sql = src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()["sql"]

        shard_idx = 1
        for offset in range(0, len(all_chunks), threshold):
            batch = all_chunks[offset : offset + threshold]
            s_name = shard_name_for_index(shard_idx)
            shard_path = tmp_dir / s_name
            shard = sqlite3.connect(str(shard_path))
            shard.execute("PRAGMA journal_mode=WAL")
            shard.execute("PRAGMA busy_timeout=5000")
            try:
                shard.execute(chunks_sql)
                placeholders = ",".join("?" for _ in col_names)
                for row in batch:
                    shard.execute(
                        f"INSERT INTO chunks ({','.join(col_names)}) VALUES ({placeholders})",
                        tuple(row[c] for c in col_names),
                    )
                shard.commit()
            finally:
                shard.close()

            # Finalize the shard through RecallDB so it has the same schema
            # guarantees as any normally-opened DB: FTS tables, triggers, and
            # rebuild for preexisting chunk rows.
            from synapt.recall.storage import RecallDB
            shard_db = RecallDB(shard_path)
            shard_db.close()

            # Record shard metadata
            min_ts = batch[0]["timestamp"] if batch else ""
            max_ts = batch[-1]["timestamp"] if batch else ""
            is_last = offset + threshold >= len(all_chunks)
            idx_conn = sqlite3.connect(str(idx_path))
            idx_conn.execute("PRAGMA busy_timeout=5000")
            try:
                idx_conn.execute(SHARD_METADATA_SQL)
                _update_shard_metadata(
                    idx_conn, shard_path, len(batch), min_ts, max_ts,
                    is_active=is_last,
                )
                idx_conn.commit()
            finally:
                idx_conn.close()

            plan[s_name] = len(batch)
            shard_idx += 1

        # Step 3: Validate total chunk count
        original_count = len(all_chunks)
        split_count = sum(v for k, v in plan.items() if k != "index.db")
        if split_count != original_count:
            raise RuntimeError(
                f"Chunk count mismatch: original={original_count}, split={split_count}"
            )

        # Step 4: Move validated files from temp dir to index dir
        for f in tmp_dir.iterdir():
            shutil.move(str(f), str(index_dir / f.name))
        tmp_dir.rmdir()

    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        src.close()

    return plan
