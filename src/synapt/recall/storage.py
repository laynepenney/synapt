"""SQLite storage backend for the synapt recall index.

Replaces the previous JSON/JSONL file-based storage with a single SQLite
database using WAL mode for concurrent read/write safety and FTS5 for
full-text search.

Schema:
    metadata  — key/value pairs (replaces manifest.json)
    chunks    — one row per transcript chunk (replaces chunks.jsonl)
    chunks_fts — FTS5 virtual table for full-text search over chunks
    knowledge — one row per knowledge node (durable cross-session facts)
    knowledge_fts — FTS5 virtual table for knowledge node search
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import struct
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synapt.recall.core import TranscriptChunk

logger = logging.getLogger("synapt.recall.storage")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384
_EMBEDDING_FMT = f"{EMBEDDING_DIM}f"
_EMBEDDING_BYTES = struct.calcsize(_EMBEDDING_FMT)

# FTS5 column weights for bm25():
#   user_text, assistant_text, tools_used, files_touched, tool_content, date_text
_FTS_WEIGHTS = "1.0, 1.5, 2.0, 2.0, 1.5, 3.0"


def query_tail_source_key(session_id: str, path: Path) -> str:
    """Bind a query-tail cursor to one concrete transcript generation."""
    stat = path.stat()
    identity = (
        f"{session_id}\0{path.resolve()}\0"
        f"{getattr(stat, 'st_dev', 0)}\0{getattr(stat, 'st_ino', 0)}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    user_text TEXT NOT NULL DEFAULT '',
    assistant_text TEXT NOT NULL DEFAULT '',
    tools_used TEXT NOT NULL DEFAULT '',
    files_touched TEXT NOT NULL DEFAULT '',
    tool_content TEXT NOT NULL DEFAULT '',
    date_text TEXT NOT NULL DEFAULT '',
    transcript_path TEXT NOT NULL DEFAULT '',
    byte_offset INTEGER NOT NULL DEFAULT -1,
    byte_length INTEGER NOT NULL DEFAULT 0,
    agent_id TEXT,
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_timestamp ON chunks(timestamp);

CREATE TABLE IF NOT EXISTS knowledge (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.5,
    source_sessions TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by TEXT NOT NULL DEFAULT '',
    contradiction_note TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    valid_from TEXT,            -- ISO 8601: when this became true (NULL = since first observation)
    valid_until TEXT,           -- ISO 8601: when this stopped being true (NULL = still current)
    version INTEGER NOT NULL DEFAULT 1,
    lineage_id TEXT NOT NULL DEFAULT '',  -- shared ID across versions of the same fact
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS pending_contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    old_node_id TEXT,                                    -- NULL for free-text claims
    new_content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    source_sessions TEXT NOT NULL DEFAULT '[]',
    detected_at TEXT NOT NULL,
    detected_by TEXT NOT NULL DEFAULT 'co-retrieval',  -- 'co-retrieval', 'consolidation', 'manual'
    status TEXT NOT NULL DEFAULT 'pending',             -- 'pending', 'confirmed', 'dismissed'
    resolved_at TEXT,
    claim_text TEXT,                                     -- free-text claim for manual contradictions
    valid_from TEXT,            -- ISO 8601: candidate's temporal bound, carried through to confirm
    valid_until TEXT,           -- ISO 8601: candidate's temporal bound, carried through to confirm
    new_node_id TEXT            -- Fix B (internal design spec, section 10.3): id of an
                                 -- ALREADY-MATERIALIZED
                                 -- contested candidate node, when the row came from the contest
                                 -- path rather than a text-only queued claim. NULL for every other
                                 -- caller (co-retrieval, consolidation contradict, manual flag) --
                                 -- resolution promotes this node instead of creating a new one from
                                 -- new_content, avoiding the orphan/duplicate _apply_supersession
                                 -- would otherwise create.
);

CREATE INDEX IF NOT EXISTS idx_knowledge_status ON knowledge(status);

CREATE TABLE IF NOT EXISTS clusters (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id  TEXT UNIQUE NOT NULL,
    topic       TEXT NOT NULL,
    search_text TEXT NOT NULL DEFAULT '',  -- topic + member chunk keywords for FTS
    cluster_type TEXT NOT NULL DEFAULT 'topic',
    session_ids TEXT NOT NULL DEFAULT '[]',
    branch      TEXT,
    date_start  TEXT,
    date_end    TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'active',
    durability  TEXT NOT NULL DEFAULT '',  -- durable|ephemeral|mixed (empty = unclassified)
    tags        TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_summaries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id   TEXT UNIQUE NOT NULL,
    summary      TEXT NOT NULL,
    method       TEXT NOT NULL DEFAULT 'concat',
    token_count  INTEGER,
    content_hash TEXT,
    created_at   TEXT NOT NULL,
    stale        INTEGER NOT NULL DEFAULT 0
);

-- Join table for cluster membership. No REFERENCES constraints because
-- PRAGMA foreign_keys is OFF (SQLite default) and clustering is fully
-- rebuilt on every recall build — orphans are impossible in practice.
--
-- run_id is NULL for ordinary self-batch clustering (the
-- table's original rows -- a full rebuild replaces them wholesale, so
-- there is nothing to undo by run) and set ONLY for rows written by
-- merge_chunks_into_cluster(). A live-store incident recovered a bad
-- merge's rows by their shared added_at TIMESTAMP because no run_id
-- existed to key on -- correct that time only because one write call
-- produces one timestamp for every row, which is not guaranteed in
-- general (clock resolution, or a future caller batching differently).
-- run_id is deliberate identity for exactly this recovery, not a
-- coincidence of implementation.
CREATE TABLE IF NOT EXISTS cluster_chunks (
    cluster_id  TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,
    added_at    TEXT NOT NULL,
    run_id      TEXT,
    PRIMARY KEY (cluster_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_cluster_chunks_chunk ON cluster_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_clusters_status ON clusters(status);

-- recall#435's recluster maintenance op reselects the same
-- stale chunks every run when they fail to reach MIN_CLUSTER_SIZE (measured
-- on the real store: 98.9% batch overlap between consecutive runs). This
-- table records which chunks a recluster run has already tried, so batch
-- selection can prefer never-tried chunks first. One row per chunk_id --
-- INSERT OR REPLACE on a repeat attempt, never deleted: a chunk that later
-- clusters is excluded from the stale query entirely and will never be
-- selected again, so its (now-inert) attempt row needs no cleanup.
CREATE TABLE IF NOT EXISTS recluster_attempts (
    chunk_id     TEXT PRIMARY KEY,
    attempted_at TEXT NOT NULL,
    run_id       TEXT NOT NULL
);

-- recall#435 follow-on: a bounded, persisted per-cluster token signature
-- (top N tokens by WITHIN-cluster document frequency), so a stale chunk can
-- be compared against what a cluster is ACTUALLY about instead of a
-- truncated raw-text search_text sample (measured: search_text averages
-- 121 chars, ~10-35 content tokens, versus a chunk's own 100-200 -- Jaccard
-- against that size mismatch dilutes real overlap below any sane
-- threshold). One row per cluster_id, INSERT OR REPLACE -- a cluster whose
-- membership changes gets a fresh signature computed from its current
-- members, never an incremental patch.
CREATE TABLE IF NOT EXISTS cluster_token_signatures (
    cluster_id  TEXT PRIMARY KEY,
    tokens      TEXT NOT NULL,  -- JSON list, top N by document frequency
    updated_at  TEXT NOT NULL
);

-- Access tracking tables for adaptive memory (Phase 2).
-- access_log is append-only: every returned search result or drill-down is
-- recorded.  access_stats is a materialized aggregate rebuilt from the log
-- during recall_build (or incrementally updated on each access).

CREATE TABLE IF NOT EXISTS access_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    item_type   TEXT NOT NULL,          -- 'chunk', 'cluster', 'knowledge'
    item_id     TEXT NOT NULL,          -- chunk id, cluster_id, or knowledge node id
    query       TEXT NOT NULL DEFAULT '',
    score       REAL NOT NULL DEFAULT 0.0,
    session_id  TEXT NOT NULL DEFAULT '',
    context     TEXT NOT NULL DEFAULT 'search',  -- 'search', 'context', 'hook'
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_access_log_item
    ON access_log(item_type, item_id);
CREATE INDEX IF NOT EXISTS idx_access_log_created
    ON access_log(created_at);

CREATE TABLE IF NOT EXISTS access_stats (
    item_type         TEXT NOT NULL,
    item_id           TEXT NOT NULL,
    access_count      INTEGER NOT NULL DEFAULT 0,  -- total accesses (search + context + hook)
    explicit_count    INTEGER NOT NULL DEFAULT 0,  -- user-initiated (search + context), not hook
    weighted_count    REAL NOT NULL DEFAULT 0.0,    -- relevance-weighted explicit access sum
    last_accessed     TEXT NOT NULL,
    first_accessed    TEXT NOT NULL,
    promotion_tier    TEXT NOT NULL DEFAULT 'raw',  -- raw|clustered|summarized|promoted|knowledge
    distinct_sessions INTEGER NOT NULL DEFAULT 0,
    distinct_queries  INTEGER NOT NULL DEFAULT 0,
    decay_score       REAL NOT NULL DEFAULT 1.0,   -- 0.0-1.0, recomputed during build
    PRIMARY KEY (item_type, item_id)
);

-- Cross-session threading: pre-computed nearest-neighbor links between
-- chunks in different sessions.  Built during recall build; used at query
-- time to expand results with related chunks from underrepresented sessions.
CREATE TABLE IF NOT EXISTS chunk_links (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    similarity REAL NOT NULL,
    PRIMARY KEY (source_id, target_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_links_source ON chunk_links(source_id);

CREATE TABLE IF NOT EXISTS access_log_archive (
    item_type    TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    date         TEXT NOT NULL,          -- ISO date (YYYY-MM-DD)
    access_count INTEGER NOT NULL,
    avg_score    REAL NOT NULL DEFAULT 0.0,
    queries      TEXT NOT NULL DEFAULT '[]',  -- JSON array of distinct queries
    PRIMARY KEY (item_type, item_id, date)
);
"""

# FTS5 table + sync triggers (created separately since IF NOT EXISTS
# doesn't work for virtual tables — we check first).
_FTS_TABLE_SQL = """\
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    user_text, assistant_text, tools_used, files_touched, tool_content, date_text,
    content=chunks,
    content_rowid=rowid,
    tokenize="porter unicode61 tokenchars '._+'"
);
"""

# Triggers are separated so they can be re-created independently
# (e.g., after a crash during save_chunks drops chunks_ad).
_FTS_TRIGGERS_SQL = """\
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, user_text, assistant_text, tools_used, files_touched, tool_content, date_text)
    VALUES (new.rowid, new.user_text, new.assistant_text, new.tools_used, new.files_touched, new.tool_content, new.date_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, user_text, assistant_text, tools_used, files_touched, tool_content, date_text)
    VALUES ('delete', old.rowid, old.user_text, old.assistant_text, old.tools_used, old.files_touched, old.tool_content, old.date_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF user_text, assistant_text, tools_used, files_touched, tool_content, date_text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, user_text, assistant_text, tools_used, files_touched, tool_content, date_text)
    VALUES ('delete', old.rowid, old.user_text, old.assistant_text, old.tools_used, old.files_touched, old.tool_content, old.date_text);
    INSERT INTO chunks_fts(rowid, user_text, assistant_text, tools_used, files_touched, tool_content, date_text)
    VALUES (new.rowid, new.user_text, new.assistant_text, new.tools_used, new.files_touched, new.tool_content, new.date_text);
END;
"""

_QUERY_TAIL_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS query_tail_chunks (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    source_key TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    user_text TEXT NOT NULL DEFAULT '',
    assistant_text TEXT NOT NULL DEFAULT '',
    tools_used TEXT NOT NULL DEFAULT '',
    files_touched TEXT NOT NULL DEFAULT '',
    tool_content TEXT NOT NULL DEFAULT '',
    date_text TEXT NOT NULL DEFAULT '',
    transcript_path TEXT NOT NULL DEFAULT '',
    byte_offset INTEGER NOT NULL,
    byte_length INTEGER NOT NULL DEFAULT 0,
    agent_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_query_tail_source_offset
    ON query_tail_chunks(source_key, byte_offset);
CREATE INDEX IF NOT EXISTS idx_query_tail_session
    ON query_tail_chunks(session_id);

CREATE TABLE IF NOT EXISTS query_tail_cursors (
    source_key TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    transcript_path TEXT NOT NULL,
    observed_complete_offset INTEGER NOT NULL,
    rewind_offset INTEGER NOT NULL,
    rewind_turn_index INTEGER NOT NULL,
    source_size INTEGER NOT NULL,
    source_mtime_ns INTEGER NOT NULL,
    observed_prefix_sha256 TEXT NOT NULL DEFAULT '',
    suppresses_base INTEGER NOT NULL DEFAULT 0,
    latest_projected_timestamp TEXT NOT NULL DEFAULT '',
    last_attempt_at TEXT NOT NULL,
    last_success_at TEXT NOT NULL
);
"""

_QUERY_TAIL_FTS_TABLE_SQL = """\
CREATE VIRTUAL TABLE query_tail_fts USING fts5(
    user_text, assistant_text, tools_used, files_touched, tool_content, date_text,
    content=query_tail_chunks,
    content_rowid=rowid,
    tokenize="porter unicode61 tokenchars '._+'"
);
"""

_QUERY_TAIL_FTS_TRIGGERS_SQL = """\
CREATE TRIGGER IF NOT EXISTS query_tail_ai AFTER INSERT ON query_tail_chunks BEGIN
    INSERT INTO query_tail_fts(rowid, user_text, assistant_text, tools_used, files_touched, tool_content, date_text)
    VALUES (new.rowid, new.user_text, new.assistant_text, new.tools_used, new.files_touched, new.tool_content, new.date_text);
END;

CREATE TRIGGER IF NOT EXISTS query_tail_ad AFTER DELETE ON query_tail_chunks BEGIN
    INSERT INTO query_tail_fts(query_tail_fts, rowid, user_text, assistant_text, tools_used, files_touched, tool_content, date_text)
    VALUES ('delete', old.rowid, old.user_text, old.assistant_text, old.tools_used, old.files_touched, old.tool_content, old.date_text);
END;

CREATE TRIGGER IF NOT EXISTS query_tail_au AFTER UPDATE OF user_text, assistant_text, tools_used, files_touched, tool_content, date_text ON query_tail_chunks BEGIN
    INSERT INTO query_tail_fts(query_tail_fts, rowid, user_text, assistant_text, tools_used, files_touched, tool_content, date_text)
    VALUES ('delete', old.rowid, old.user_text, old.assistant_text, old.tools_used, old.files_touched, old.tool_content, old.date_text);
    INSERT INTO query_tail_fts(rowid, user_text, assistant_text, tools_used, files_touched, tool_content, date_text)
    VALUES (new.rowid, new.user_text, new.assistant_text, new.tools_used, new.files_touched, new.tool_content, new.date_text);
END;
"""


# Knowledge FTS5 table + sync triggers (same pattern as chunks_fts)
_KNOWLEDGE_FTS_TABLE_SQL = """\
CREATE VIRTUAL TABLE knowledge_fts USING fts5(
    content, category, tags,
    content=knowledge,
    content_rowid=rowid,
    tokenize="porter unicode61 tokenchars '._+'"
);
"""

_KNOWLEDGE_FTS_TRIGGERS_SQL = """\
CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
    INSERT INTO knowledge_fts(rowid, content, category, tags)
    VALUES (new.rowid, new.content, new.category, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, category, tags)
    VALUES ('delete', old.rowid, old.content, old.category, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE OF content, category, tags ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, category, tags)
    VALUES ('delete', old.rowid, old.content, old.category, old.tags);
    INSERT INTO knowledge_fts(rowid, content, category, tags)
    VALUES (new.rowid, new.content, new.category, new.tags);
END;
"""

# FTS5 column weights for knowledge bm25(): content, category, tags
_KNOWLEDGE_FTS_WEIGHTS = "2.0, 1.0, 1.5"

# Cluster topic FTS5 table + sync triggers
# Note: content_rowid='id' (not 'rowid') because clusters.id is the
# explicit INTEGER PRIMARY KEY, not the implicit rowid alias used by
# chunks_fts and knowledge_fts (which use content_rowid=rowid).
_CLUSTERS_FTS_TABLE_SQL = """\
CREATE VIRTUAL TABLE clusters_fts USING fts5(
    topic, search_text,
    content='clusters',
    content_rowid='id',
    tokenize="porter unicode61 tokenchars '._+'"
);
"""

_CLUSTERS_FTS_TRIGGERS_SQL = """\
CREATE TRIGGER IF NOT EXISTS clusters_ai AFTER INSERT ON clusters BEGIN
    INSERT INTO clusters_fts(rowid, topic, search_text)
    VALUES (new.id, new.topic, new.search_text);
END;

CREATE TRIGGER IF NOT EXISTS clusters_ad AFTER DELETE ON clusters BEGIN
    INSERT INTO clusters_fts(clusters_fts, rowid, topic, search_text)
    VALUES ('delete', old.id, old.topic, old.search_text);
END;

CREATE TRIGGER IF NOT EXISTS clusters_au AFTER UPDATE OF topic, search_text ON clusters BEGIN
    INSERT INTO clusters_fts(clusters_fts, rowid, topic, search_text)
    VALUES ('delete', old.id, old.topic, old.search_text);
    INSERT INTO clusters_fts(rowid, topic, search_text)
    VALUES (new.id, new.topic, new.search_text);
END;
"""


def _normalize_ddl(sql: str) -> str:
    """Put a CREATE statement into the form ``sqlite_master`` stores it in.

    ``sqlite_master.sql`` holds the statement as executed and **without** the
    trailing statement terminator, so a module constant written as a runnable
    script — ending in ``);`` — can never compare equal to it no matter how the
    whitespace is handled.  Collapsing whitespace is not enough: ``strip()``
    removes the newline after the semicolon and leaves the semicolon itself.

    Both sides of a schema comparison go through this function so the two are
    in the same form.  Comparing a stored statement against a raw constant is
    the defect this exists to prevent.  All three detectors route through it;
    nothing in the code compels a fourth one to, so adding an FTS table means
    routing its detector here deliberately.
    """
    return " ".join(sql.strip().lower().split()).rstrip(";").rstrip()


_FTS5_KEYWORDS = frozenset({"and", "or", "not", "near"})


def _escape_fts_token(tok: str) -> str:
    """Quote a single FTS5 token if it contains special chars or is a keyword."""
    if any(ch in tok for ch in "._+") or tok in _FTS5_KEYWORDS:
        return f'"{tok}"'
    return tok

# Common stop words to strip from FTS queries. These are too frequent
# to be useful in full-text search and cause AND queries to fail when
# they don't co-occur with content words in the same chunk.
_FTS_STOP_WORDS = frozenset({
    "the", "is", "are", "was", "were", "be", "been", "am",
    "do", "does", "did", "has", "have", "had", "having",
    "an", "in", "on", "at", "to", "of", "for", "by", "it",
    "he", "she", "we", "my", "me", "no", "so", "if", "up",
    "go", "its", "but",
    "her", "him", "his", "our", "your", "you", "they", "them", "their",
    "can", "could", "would", "should", "will", "may", "might",
    "also", "just", "very", "much", "some", "any", "all", "than",
    # Question words — useful as intent signals but noisy in FTS
    "what", "when", "where", "which", "who", "whom", "whose",
    "why", "how", "that", "this", "with", "from", "about",
})


def _escape_fts_tokens(query: str) -> list[str]:
    """Tokenize and escape a user query for FTS5.

    Returns a list of escaped tokens.  Tokens containing dots are quoted
    (e.g. ``api_index.py`` → ``"api_index.py"``), as are FTS5 keywords.
    Single-character tokens and stop words are dropped to reduce noise.
    """
    tokens = [
        t for t in re.sub(r"[^a-zA-Z0-9_.+]", " ", query.lower()).split()
        if len(t) > 1 and t not in _FTS_STOP_WORDS
    ]
    return [_escape_fts_token(tok) for tok in tokens]


def _escape_fts_query(query: str, use_or: bool = False) -> str:
    """Escape a user query string for safe FTS5 matching.

    Tokenizes using the same regex as bm25._tokenize, then quotes any
    token containing a dot (e.g. ``api_index.py`` → ``"api_index.py"``)
    or matching an FTS5 keyword (AND, OR, NOT, NEAR).

    By default, tokens are joined with spaces (implicit AND in FTS5).
    Set *use_or* to join with OR — matches chunks containing ANY term.
    """
    tokens = _escape_fts_tokens(query)
    if not tokens:
        return ""
    joiner = " OR " if use_or else " "
    return joiner.join(tokens)


def _build_entity_anchored_query(
    entity_tokens: list[str],
    content_tokens: list[str],
) -> str:
    """Build an FTS5 query that requires entity terms AND any content term.

    Returns e.g. ``caroline AND (lgbtq OR support OR group)`` which finds
    chunks mentioning the entity with at least one content keyword.
    Returns empty string if either list is empty.
    """
    if not entity_tokens or not content_tokens:
        return ""
    entity_part = " ".join(_escape_fts_token(t) for t in entity_tokens)
    if len(content_tokens) == 1:
        content_part = _escape_fts_token(content_tokens[0])
    else:
        content_part = "(" + " OR ".join(_escape_fts_token(t) for t in content_tokens) + ")"
    return f"{entity_part} AND {content_part}"


# ---------------------------------------------------------------------------
# RecallDB
# ---------------------------------------------------------------------------


class RecallDB:
    """SQLite storage backend for the recall index.

    Opens (or creates) a database at *db_path* with WAL mode enabled for
    concurrent read/write safety.  Provides CRUD for chunks, FTS5 search,
    per-row embedding BLOBs, and key/value metadata.
    """

    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        self._ensure_schema()

    # -- connection --------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")  # 30s — builds can be slow
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def open_readonly(cls, db_path: Path | str, busy_timeout_ms: int = 2000) -> "RecallDB":
        """Open an EXISTING index for reads only.

        No schema DDL, no migrations, a short busy timeout, and SQLite's
        ``mode=ro`` so any write is an error rather than a contention.

        Why this exists: the session-start hook read pending contradictions
        through ``__init__``, whose ``_ensure_schema`` runs DDL and migrations
        under a 30s busy timeout. Behind a concurrent build that connect
        blocked for 13.9s while the query itself took 0.00s. A process that
        prints from existing state must never queue behind the writer.

        Raises ``sqlite3.OperationalError`` if the file does not exist: a
        read-only open never mints a store, and a missing index must read as
        "missing", not as "empty".
        """
        path = Path(db_path)
        if not path.exists():
            raise sqlite3.OperationalError(f"no index at {path}")
        uri = path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=busy_timeout_ms / 1000)
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        conn.execute("PRAGMA query_only=1")
        conn.row_factory = sqlite3.Row
        obj = cls.__new__(cls)
        obj._path = path
        obj._conn = conn
        return obj

    def _ensure_schema(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.executescript(_QUERY_TAIL_SCHEMA_SQL)
        # Migrate existing tables: add columns that may be missing
        query_tail_cursor_columns = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(query_tail_cursors)"
            ).fetchall()
        }
        if "suppresses_base" not in query_tail_cursor_columns:
            self._conn.execute(
                "ALTER TABLE query_tail_cursors "
                "ADD COLUMN suppresses_base INTEGER NOT NULL DEFAULT 0"
            )
        if "observed_prefix_sha256" not in query_tail_cursor_columns:
            self._conn.execute(
                "ALTER TABLE query_tail_cursors "
                "ADD COLUMN observed_prefix_sha256 TEXT NOT NULL DEFAULT ''"
            )
        self._migrate_chunks_table()
        # session_overview's GROUP BY session_id aggregates timestamp,
        # turn_index, transcript_path, and agent_id per row. idx_chunks_session
        # alone covers the GROUP BY key, but referencing any of those four
        # columns forces SQLite off the index and into a per-row table lookup
        # (measured: on an isolated shard copy, adding ANY one of them to the
        # SELECT list -- independent of which aggregate function touches it --
        # was the entire cost; julianday() and GROUP_CONCAT specifically were
        # not).
        #
        # This index covers session_id/timestamp/turn_index/agent_id -- NOT
        # transcript_path. That is deliberate, not an oversight: a 5-column
        # index covering all five was measured to cost ~22% more per INSERT
        # than no covering index at all, entirely from having one more
        # composite index on the table, not from the aggregate itself
        # (recall#1147 follow-on, tracked privately -- an interleaved,
        # two-batch-size write-cost measurement). transcript_path IS a
        # per-session constant (one source transcript file per session_id),
        # unlike agent_id (see session_overview()'s docstring) -- so instead
        # of covering it too, session_overview() fetches it via one lookup
        # PER SESSION rather than folding it into the per-CHUNK aggregate.
        # That drops the insert cost to ~18% while keeping the same read
        # benefit as the 5-column shape (measured, same follow-on).
        #
        # Created here, AFTER _migrate_chunks_table(), not in _SCHEMA_SQL's
        # executescript above: an older chunks table may not have
        # transcript_path/agent_id yet, and creating an index that
        # references them before the migration adds them fails outright.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_overview_covering "
            "ON chunks(session_id, timestamp, turn_index, agent_id)"
        )
        self._migrate_knowledge_table()
        self._migrate_clusters_table()
        self._migrate_access_stats_table()
        self._migrate_contradictions_table()
        self._migrate_chunk_links_table()
        self._migrate_cluster_chunks_table()
        # Check if FTS table exists (FTS5 virtual tables don't support
        # IF NOT EXISTS, so we check manually before creating)
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        if row is None:
            self._conn.executescript(_FTS_TABLE_SQL)
            self._conn.executescript(_FTS_TRIGGERS_SQL)
            chunk_count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if chunk_count > 0:
                # A DB may already contain chunk rows (e.g. split-created shards)
                # before chunks_fts exists. Rebuild immediately so first search
                # sees the existing content instead of an empty FTS index.
                self._conn.execute(
                    "INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')"
                )
                self._conn.commit()
        elif self._needs_fts_migration():
            # Tokenizer changed (e.g., porter added) — recreate FTS table
            # and rebuild from the content table so search keeps working.
            self._conn.execute("DROP TABLE IF EXISTS chunks_fts")
            for name in ("chunks_ai", "chunks_ad", "chunks_au"):
                self._conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            self._conn.executescript(_FTS_TABLE_SQL)
            self._conn.executescript(_FTS_TRIGGERS_SQL)
            # Repopulate FTS from existing chunks (external content rebuild)
            self._conn.execute(
                "INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')"
            )
            self._conn.commit()
        else:
            # Verify all three FTS sync triggers exist — they may be missing
            # after a crash during save_chunks (which drops/recreates chunks_ad)
            existing = {r[0] for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name IN ('chunks_ai', 'chunks_ad', 'chunks_au')"
            ).fetchall()}
            if len(existing) < 3:
                self._conn.executescript(_FTS_TRIGGERS_SQL)

        query_tail_fts = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='query_tail_fts'"
        ).fetchone()
        if query_tail_fts is None:
            self._conn.executescript(_QUERY_TAIL_FTS_TABLE_SQL)
            self._conn.executescript(_QUERY_TAIL_FTS_TRIGGERS_SQL)
            tail_count = self._conn.execute(
                "SELECT COUNT(*) FROM query_tail_chunks"
            ).fetchone()[0]
            if tail_count:
                self._conn.execute(
                    "INSERT INTO query_tail_fts(query_tail_fts) VALUES('rebuild')"
                )
                self._conn.commit()
        else:
            existing_tail_triggers = {
                row[0]
                for row in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger' "
                    "AND name IN ('query_tail_ai', 'query_tail_ad', 'query_tail_au')"
                ).fetchall()
            }
            if len(existing_tail_triggers) < 3:
                self._conn.executescript(_QUERY_TAIL_FTS_TRIGGERS_SQL)

        # Knowledge FTS table (same pattern: check existence, create if missing)
        krow = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_fts'"
        ).fetchone()
        if krow is None:
            self._conn.executescript(_KNOWLEDGE_FTS_TABLE_SQL)
            self._conn.executescript(_KNOWLEDGE_FTS_TRIGGERS_SQL)
        elif self._needs_knowledge_fts_migration():
            self._conn.execute("DROP TABLE IF EXISTS knowledge_fts")
            for name in ("knowledge_ai", "knowledge_ad", "knowledge_au"):
                self._conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            self._conn.executescript(_KNOWLEDGE_FTS_TABLE_SQL)
            self._conn.executescript(_KNOWLEDGE_FTS_TRIGGERS_SQL)
            self._conn.execute(
                "INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')"
            )
            self._conn.commit()
        else:
            # Verify triggers exist
            existing_kt = {r[0] for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name IN ('knowledge_ai', 'knowledge_ad', 'knowledge_au')"
            ).fetchall()}
            if len(existing_kt) < 3:
                self._conn.executescript(_KNOWLEDGE_FTS_TRIGGERS_SQL)

        # Clusters FTS table (same pattern)
        crow = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='clusters_fts'"
        ).fetchone()
        if crow is None:
            # Only create if the clusters table exists (it should — _SCHEMA_SQL creates it)
            ctbl = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='clusters'"
            ).fetchone()
            if ctbl is not None:
                self._conn.executescript(_CLUSTERS_FTS_TABLE_SQL)
                self._conn.executescript(_CLUSTERS_FTS_TRIGGERS_SQL)
        elif self._needs_clusters_fts_migration():
            # Schema changed (e.g., search_text column added) — recreate
            self._conn.execute("DROP TABLE IF EXISTS clusters_fts")
            for name in ("clusters_ai", "clusters_ad", "clusters_au"):
                self._conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            self._conn.executescript(_CLUSTERS_FTS_TABLE_SQL)
            self._conn.executescript(_CLUSTERS_FTS_TRIGGERS_SQL)
            self._conn.execute(
                "INSERT INTO clusters_fts(clusters_fts) VALUES('rebuild')"
            )
            self._conn.commit()
        else:
            existing_ct = {r[0] for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name IN ('clusters_ai', 'clusters_ad', 'clusters_au')"
            ).fetchall()}
            if len(existing_ct) < 3:
                self._conn.executescript(_CLUSTERS_FTS_TRIGGERS_SQL)

    def _migrate_chunks_table(self) -> None:
        """Add columns that may be missing from an older chunks table."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if row is None:
            return
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        new_cols = {
            "tool_content": "TEXT NOT NULL DEFAULT ''",
            "date_text": "TEXT NOT NULL DEFAULT ''",
            "transcript_path": "TEXT NOT NULL DEFAULT ''",
            "byte_offset": "INTEGER NOT NULL DEFAULT -1",
            "byte_length": "INTEGER NOT NULL DEFAULT 0",
            "agent_id": "TEXT",
        }
        for col_name, col_def in new_cols.items():
            if col_name not in cols:
                self._conn.execute(
                    f"ALTER TABLE chunks ADD COLUMN {col_name} {col_def}"
                )
        self._conn.commit()

    def _migrate_knowledge_table(self) -> None:
        """Add columns that may be missing from an older knowledge table."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge'"
        ).fetchone()
        if row is None:
            return  # Table doesn't exist yet; CREATE TABLE will handle it
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(knowledge)").fetchall()
        }
        migrations = [
            ("contradiction_note", "TEXT NOT NULL DEFAULT ''"),
            ("valid_from", "TEXT"),
            ("valid_until", "TEXT"),
            ("version", "INTEGER NOT NULL DEFAULT 1"),
            ("lineage_id", "TEXT NOT NULL DEFAULT ''"),
            ("embedding", "BLOB"),
            ("source_turns", "TEXT NOT NULL DEFAULT '[]'"),
            ("source_offsets", "TEXT NOT NULL DEFAULT '[]'"),
        ]
        for col_name, col_def in migrations:
            if col_name not in cols:
                self._conn.execute(
                    f"ALTER TABLE knowledge ADD COLUMN {col_name} {col_def}"
                )
        self._conn.commit()

    def _migrate_clusters_table(self) -> None:
        """Add columns that may be missing from older tables."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='clusters'"
        ).fetchone()
        if row is None:
            return
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(clusters)").fetchall()
        }
        if "search_text" not in cols:
            self._conn.execute(
                "ALTER TABLE clusters ADD COLUMN "
                "search_text TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()

        if "tags" not in cols:
            self._conn.execute(
                "ALTER TABLE clusters ADD COLUMN "
                "tags TEXT NOT NULL DEFAULT '[]'"
            )
            self._conn.commit()

        if "durability" not in cols:
            self._conn.execute(
                "ALTER TABLE clusters ADD COLUMN "
                "durability TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()

        # Add content_hash to cluster_summaries for LLM summary reuse
        row = self._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='cluster_summaries'"
        ).fetchone()
        if row is not None:
            sum_cols = {
                r[1]
                for r in self._conn.execute(
                    "PRAGMA table_info(cluster_summaries)"
                ).fetchall()
            }
            if "content_hash" not in sum_cols:
                self._conn.execute(
                    "ALTER TABLE cluster_summaries ADD COLUMN content_hash TEXT"
                )
                self._conn.commit()

    def _migrate_cluster_chunks_table(self) -> None:
        """Add ``run_id`` to a ``cluster_chunks`` table created before it
        existed. NULL default: pre-existing rows were not written by any
        merge run and have nothing to be keyed by."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cluster_chunks'"
        ).fetchone()
        if row is None:
            return
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(cluster_chunks)").fetchall()
        }
        if "run_id" not in cols:
            self._conn.execute("ALTER TABLE cluster_chunks ADD COLUMN run_id TEXT")
            self._conn.commit()

    def _migrate_access_stats_table(self) -> None:
        """Add promotion columns that may be missing from an older access_stats table."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='access_stats'"
        ).fetchone()
        if row is None:
            return
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(access_stats)").fetchall()
        }
        migrations = [
            ("promotion_tier", "TEXT NOT NULL DEFAULT 'raw'"),
            ("distinct_sessions", "INTEGER NOT NULL DEFAULT 0"),
            ("distinct_queries", "INTEGER NOT NULL DEFAULT 0"),
            ("decay_score", "REAL NOT NULL DEFAULT 1.0"),
            ("weighted_count", "REAL NOT NULL DEFAULT 0.0"),
        ]
        for col_name, col_def in migrations:
            if col_name not in cols:
                self._conn.execute(
                    f"ALTER TABLE access_stats ADD COLUMN {col_name} {col_def}"
                )
        # Always ensure the tier index exists (safe for fresh or migrated DBs)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_stats_tier "
            "ON access_stats(promotion_tier)"
        )
        self._conn.commit()

    def _migrate_contradictions_table(self) -> None:
        """Migrate pending_contradictions: add claim_text, make old_node_id nullable, add
        new_node_id (Fix B, internal design spec, section 10.10 -- Opus's five-check
        #2, installed/DB path requirement).

        SQLite doesn't support ALTER COLUMN, so we recreate the table when
        old_node_id has a NOT NULL constraint from the old schema.
        """
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pending_contradictions'"
        ).fetchone()
        if row is None:
            return

        # Check if old_node_id is NOT NULL (old schema)
        col_info = self._conn.execute("PRAGMA table_info(pending_contradictions)").fetchall()
        col_map = {r[1]: r for r in col_info}
        needs_recreate = col_map.get("old_node_id", (None,) * 6)[3] == 1  # notnull flag
        has_claim_text = "claim_text" in col_map
        has_temporal = "valid_from" in col_map and "valid_until" in col_map
        has_new_node_id = "new_node_id" in col_map

        if not needs_recreate and has_claim_text and has_temporal and has_new_node_id:
            return  # Already migrated

        if needs_recreate:
            # Recreate table: old_node_id NOT NULL → nullable, add claim_text + temporal +
            # new_node_id columns. new_node_id is baked into _pc_new directly rather than left
            # to the ALTER TABLE branch below -- this branch and that one are mutually
            # exclusive within a single call, so a DB old enough to need recreate would
            # otherwise not pick up new_node_id until some later re-open (spec section 10.10).
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS _pc_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    old_node_id TEXT,
                    new_content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    source_sessions TEXT NOT NULL DEFAULT '[]',
                    detected_at TEXT NOT NULL,
                    detected_by TEXT NOT NULL DEFAULT 'co-retrieval',
                    status TEXT NOT NULL DEFAULT 'pending',
                    resolved_at TEXT,
                    claim_text TEXT,
                    valid_from TEXT,
                    valid_until TEXT,
                    new_node_id TEXT
                );
                INSERT INTO _pc_new (id, old_node_id, new_content, category, reason,
                    source_sessions, detected_at, detected_by, status, resolved_at)
                SELECT id, old_node_id, new_content, category, reason,
                    source_sessions, detected_at, detected_by, status, resolved_at
                FROM pending_contradictions;
                DROP TABLE pending_contradictions;
                ALTER TABLE _pc_new RENAME TO pending_contradictions;
            """)
            self._conn.commit()
        else:
            if not has_claim_text:
                self._conn.execute(
                    "ALTER TABLE pending_contradictions ADD COLUMN claim_text TEXT"
                )
            # Per-column, NOT gated behind the combined has_temporal check (adversarial
            # verification finding, 2026-07-15): a process interrupted between two sequential
            # ALTER TABLE calls (OOM kill, kill -9, forced restart) can leave exactly one of the
            # two columns present. A combined check would then unconditionally re-add the column
            # that already exists -> sqlite3.OperationalError: duplicate column name -> a
            # PERMANENT poison-pill (every future open of that DB file crashes identically).
            # Matches the per-column idiom already used by _migrate_knowledge_table /
            # _migrate_access_stats_table in this same file. new_node_id follows the same
            # idiom for the same reason (spec section 10.10).
            if "valid_from" not in col_map:
                self._conn.execute(
                    "ALTER TABLE pending_contradictions ADD COLUMN valid_from TEXT"
                )
            if "valid_until" not in col_map:
                self._conn.execute(
                    "ALTER TABLE pending_contradictions ADD COLUMN valid_until TEXT"
                )
            if not has_new_node_id:
                self._conn.execute(
                    "ALTER TABLE pending_contradictions ADD COLUMN new_node_id TEXT"
                )
            self._conn.commit()

    def _migrate_chunk_links_table(self) -> None:
        """Migrate chunk_links from rowid-based to chunk-ID-based schema.

        Old schema: (source_rowid INTEGER, target_rowid INTEGER, similarity)
        New schema: (source_id TEXT, target_id TEXT, similarity)

        Chunk IDs are globally unique across shards, unlike rowids.
        """
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_links'"
        ).fetchone()
        if row is None:
            return
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(chunk_links)").fetchall()
        }
        if "source_id" in cols:
            return  # Already migrated

        if "source_rowid" in cols:
            # Recreate with new schema — existing links become stale
            # (rowids are meaningless after sharding) so we drop them.
            # They'll be rebuilt on next `recall build`.
            self._conn.executescript("""
                DROP TABLE IF EXISTS chunk_links;
                CREATE TABLE chunk_links (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    PRIMARY KEY (source_id, target_id)
                );
                CREATE INDEX IF NOT EXISTS idx_chunk_links_source ON chunk_links(source_id);
            """)
            self._conn.commit()

    def _needs_clusters_fts_migration(self) -> bool:
        """Check if clusters_fts schema differs from current definition."""
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='clusters_fts'"
        ).fetchone()
        if row is None:
            return False
        return _normalize_ddl(row[0] or "") != _normalize_ddl(_CLUSTERS_FTS_TABLE_SQL)

    def _needs_fts_migration(self) -> bool:
        """Check if the FTS table's definition differs from the current schema.

        Compares the full normalized DDL (whitespace-collapsed, lowercased)
        rather than checking for a single keyword.  This detects any change
        to the tokenizer config, column weights, or tokenchars.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        if row is None:
            return False
        return _normalize_ddl(row[0] or "") != _normalize_ddl(_FTS_TABLE_SQL)

    def _needs_knowledge_fts_migration(self) -> bool:
        """Check if the knowledge FTS definition differs from the current schema."""
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='knowledge_fts'"
        ).fetchone()
        if row is None:
            return False
        return _normalize_ddl(row[0] or "") != _normalize_ddl(_KNOWLEDGE_FTS_TABLE_SQL)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> RecallDB:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    @property
    def path(self) -> Path:
        return self._path

    @property
    def mtime(self) -> float:
        """File modification time of the database (for cache invalidation)."""
        try:
            return self._path.stat().st_mtime
        except OSError:
            return 0.0

    # -- chunks CRUD -------------------------------------------------------

    @staticmethod
    def _query_tail_chunk_from_row(row) -> TranscriptChunk:
        from synapt.recall.core import TranscriptChunk

        return TranscriptChunk(
            id=row["id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            turn_index=row["turn_index"],
            user_text=row["user_text"],
            assistant_text=row["assistant_text"],
            tools_used=json.loads(row["tools_used"]) if row["tools_used"] else [],
            files_touched=(
                json.loads(row["files_touched"]) if row["files_touched"] else []
            ),
            tool_content=row["tool_content"] or "",
            date_text=row["date_text"] or "",
            transcript_path=row["transcript_path"] or "",
            byte_offset=row["byte_offset"],
            byte_length=row["byte_length"],
            agent_id=row["agent_id"],
        )

    def load_query_tail_cursor(self, source_key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM query_tail_cursors WHERE source_key = ?",
            (source_key,),
        ).fetchone()
        return dict(row) if row is not None else None

    def load_query_tail_cursor_for_session(self, session_id: str) -> dict | None:
        """Load the newest overlay cursor bound to one transcript session."""
        row = self._conn.execute(
            "SELECT * FROM query_tail_cursors WHERE session_id = ? "
            "ORDER BY last_success_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def load_query_tail_cursors(self) -> list[dict]:
        """Load every committed query-tail cursor."""
        rows = self._conn.execute(
            "SELECT * FROM query_tail_cursors ORDER BY source_key"
        ).fetchall()
        return [dict(row) for row in rows]

    def query_tail_suppressed_sessions(self) -> set[str]:
        """Return sessions whose replacement overlay hides every base row."""
        rows = self._conn.execute(
            "SELECT DISTINCT session_id FROM query_tail_cursors "
            "WHERE suppresses_base = 1"
        ).fetchall()
        return {row["session_id"] for row in rows}

    def session_indexed_extent(self, session_id: str) -> dict | None:
        """Return the last searchable transcript turn stored in base chunks."""
        row = self._conn.execute(
            "SELECT byte_offset, byte_length, turn_index, timestamp, transcript_path "
            "FROM chunks WHERE session_id = ? AND turn_index >= 0 "
            "AND byte_offset >= 0 ORDER BY byte_offset DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "observed_complete_offset": row["byte_offset"] + row["byte_length"],
            "rewind_offset": row["byte_offset"],
            "rewind_turn_index": row["turn_index"],
            "latest_projected_timestamp": row["timestamp"] or "",
            "transcript_path": row["transcript_path"] or "",
        }

    def clear_query_tail(self, source_key: str) -> None:
        """Retire an overlay only after its coverage exists in the base index."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM query_tail_chunks WHERE source_key = ?",
                (source_key,),
            )
            self._conn.execute(
                "DELETE FROM query_tail_cursors WHERE source_key = ?",
                (source_key,),
            )

    def clear_query_tail_session(self, session_id: str) -> None:
        """Discard every prior source generation for one session."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM query_tail_chunks WHERE session_id = ?",
                (session_id,),
            )
            self._conn.execute(
                "DELETE FROM query_tail_cursors WHERE session_id = ?",
                (session_id,),
            )

    def retire_absorbed_query_tails(self) -> None:
        """Retire overlays absorbed by this monolithic base index."""
        for cursor in self.load_query_tail_cursors():
            path = Path(cursor["transcript_path"])
            try:
                current_key = query_tail_source_key(cursor["session_id"], path)
            except OSError:
                continue
            if current_key != cursor["source_key"]:
                continue
            extent = self.session_indexed_extent(cursor["session_id"])
            if extent is None:
                continue
            if (
                extent["observed_complete_offset"]
                < cursor["observed_complete_offset"]
                or extent.get("latest_projected_timestamp", "")
                != cursor.get("latest_projected_timestamp", "")
            ):
                continue
            self.clear_query_tail(cursor["source_key"])

    def replace_query_tail(
        self,
        *,
        source_key: str,
        session_id: str,
        rewind_offset: int,
        chunks: list[TranscriptChunk],
        cursor: dict,
    ) -> None:
        """Atomically replace one reparsed tail and advance its cursor."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM query_tail_chunks "
                "WHERE source_key = ? AND byte_offset >= ?",
                (source_key, rewind_offset),
            )
            for chunk in chunks:
                self._conn.execute(
                    "INSERT OR REPLACE INTO query_tail_chunks "
                    "(id, source_key, session_id, timestamp, turn_index, user_text, "
                    " assistant_text, tools_used, files_touched, tool_content, date_text, "
                    " transcript_path, byte_offset, byte_length, agent_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.id,
                        source_key,
                        session_id,
                        chunk.timestamp,
                        chunk.turn_index,
                        chunk.user_text,
                        chunk.assistant_text,
                        json.dumps(chunk.tools_used),
                        json.dumps(chunk.files_touched),
                        chunk.tool_content,
                        chunk.date_text,
                        chunk.transcript_path,
                        chunk.byte_offset,
                        chunk.byte_length,
                        chunk.agent_id,
                    ),
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO query_tail_cursors "
                "(source_key, session_id, transcript_path, observed_complete_offset, "
                " rewind_offset, rewind_turn_index, source_size, source_mtime_ns, "
                " observed_prefix_sha256, "
                " suppresses_base, latest_projected_timestamp, last_attempt_at, "
                " last_success_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_key,
                    session_id,
                    cursor["transcript_path"],
                    cursor["observed_complete_offset"],
                    cursor["rewind_offset"],
                    cursor["rewind_turn_index"],
                    cursor["source_size"],
                    cursor["source_mtime_ns"],
                    cursor["observed_prefix_sha256"],
                    int(cursor.get("suppresses_base", False)),
                    cursor.get("latest_projected_timestamp", ""),
                    cursor["last_attempt_at"],
                    cursor["last_success_at"],
                ),
            )

    def load_query_tail_chunks(self) -> list[TranscriptChunk]:
        rows = self._conn.execute(
            "SELECT * FROM query_tail_chunks ORDER BY rowid"
        ).fetchall()
        return [self._query_tail_chunk_from_row(row) for row in rows]

    def query_tail_chunk_refs(self) -> list[tuple[str, str]]:
        """Return lightweight ``(id, session_id)`` overlay identities."""
        return [
            (row["id"], row["session_id"])
            for row in self._conn.execute(
                "SELECT id, session_id FROM query_tail_chunks"
            ).fetchall()
        ]

    def query_tail_chunk_count(self) -> int:
        """Count query-tail chunks without hydrating their content."""
        row = self._conn.execute("SELECT COUNT(*) FROM query_tail_chunks").fetchone()
        return row[0] if row else 0

    def load_query_tail_chunk_by_rowid(self, rowid: int) -> TranscriptChunk | None:
        row = self._conn.execute(
            "SELECT * FROM query_tail_chunks WHERE rowid = ?",
            (rowid,),
        ).fetchone()
        return self._query_tail_chunk_from_row(row) if row is not None else None

    def load_query_tail_chunks_by_rowids(
        self, rowids: list[int],
    ) -> dict[int, TranscriptChunk]:
        if not rowids:
            return {}
        loaded: dict[int, TranscriptChunk] = {}
        for offset in range(0, len(rowids), 900):
            batch = rowids[offset:offset + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                "SELECT * FROM query_tail_chunks WHERE rowid IN (" + placeholders + ")",
                batch,
            ).fetchall()
            for row in rows:
                loaded[row["rowid"]] = self._query_tail_chunk_from_row(row)
        return loaded

    def query_tail_id_rowid_map(self) -> dict[str, int]:
        return {
            row["id"]: row["rowid"]
            for row in self._conn.execute(
                "SELECT rowid, id FROM query_tail_chunks"
            ).fetchall()
        }

    def query_tail_fts_search(
        self, query: str, limit: int = 100,
    ) -> list[tuple[int, float]]:
        fts_query = _escape_fts_query(query)
        if not fts_query:
            return []
        rows = self._conn.execute(
            f"SELECT query_tail_fts.rowid, "
            f"-bm25(query_tail_fts, {_FTS_WEIGHTS}) AS score "
            "FROM query_tail_fts WHERE query_tail_fts MATCH ? "
            "ORDER BY score DESC LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        if not rows and len(_escape_fts_tokens(query)) > 1:
            rows = self._conn.execute(
                f"SELECT query_tail_fts.rowid, "
                f"-bm25(query_tail_fts, {_FTS_WEIGHTS}) AS score "
                "FROM query_tail_fts WHERE query_tail_fts MATCH ? "
                "ORDER BY score DESC LIMIT ?",
                (_escape_fts_query(query, use_or=True), limit),
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def query_tail_fts_search_raw(
        self, fts_query: str, limit: int = 100,
    ) -> list[tuple[int, float]]:
        if not fts_query:
            return []
        rows = self._conn.execute(
            f"SELECT query_tail_fts.rowid, "
            f"-bm25(query_tail_fts, {_FTS_WEIGHTS}) AS score "
            "FROM query_tail_fts WHERE query_tail_fts MATCH ? "
            "ORDER BY score DESC LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def query_tail_fts_search_by_session(
        self, query: str, session_ids: list[str], limit: int = 100,
    ) -> list[tuple[int, float]]:
        fts_query = _escape_fts_query(query)
        if not fts_query or not session_ids:
            return []
        placeholders = ",".join("?" for _ in session_ids)
        rows = self._conn.execute(
            f"SELECT query_tail_fts.rowid, "
            f"-bm25(query_tail_fts, {_FTS_WEIGHTS}) AS score "
            "FROM query_tail_fts JOIN query_tail_chunks c "
            "ON c.rowid = query_tail_fts.rowid "
            "WHERE query_tail_fts MATCH ? "
            f"AND c.session_id IN ({placeholders}) "
            "ORDER BY score DESC LIMIT ?",
            (fts_query, *session_ids, limit),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def save_chunks(self, chunks: list[TranscriptChunk]) -> None:
        """Replace all chunks in the database and rebuild the FTS5 index.

        Preserves existing embeddings by matching on chunk ID.  If a chunk
        already exists with an embedding, the embedding is carried over to
        the new row.
        """
        cur = self._conn.cursor()

        # Snapshot existing embeddings keyed by chunk ID before wiping
        existing_embs: dict[str, bytes] = {}
        rows = cur.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        for r in rows:
            existing_embs[r[0]] = r[1]

        # Disable triggers during bulk delete to avoid per-row FTS deletes
        # (we do a single 'delete-all' instead, which is O(1) vs O(n))
        cur.execute("DROP TRIGGER IF EXISTS chunks_ad")
        cur.execute("DELETE FROM chunks")
        cur.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('delete-all')")
        # Re-create the delete trigger (use execute(), NOT executescript() —
        # executescript implicitly commits, breaking our transaction atomicity)
        cur.execute(
            "CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN "
            "  INSERT INTO chunks_fts(chunks_fts, rowid, user_text, assistant_text, "
            "    tools_used, files_touched, tool_content, date_text) "
            "  VALUES ('delete', old.rowid, old.user_text, old.assistant_text, "
            "    old.tools_used, old.files_touched, old.tool_content, old.date_text); "
            "END;"
        )

        # Insert in order (rowid = insertion order, 1-based).
        # Carry over embeddings by matching chunk IDs.
        total = len(chunks)
        log_interval = max(500, total // 10)  # Every 500 or 10%, whichever is larger
        t0 = time.monotonic()
        for i, chunk in enumerate(chunks, 1):
            emb_blob = existing_embs.get(chunk.id)
            cur.execute(
                "INSERT INTO chunks "
                "(id, session_id, timestamp, turn_index, user_text, assistant_text, "
                " tools_used, files_touched, tool_content, date_text, "
                " transcript_path, byte_offset, byte_length, agent_id, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk.id,
                    chunk.session_id,
                    chunk.timestamp,
                    chunk.turn_index,
                    chunk.user_text,
                    chunk.assistant_text,
                    json.dumps(chunk.tools_used),
                    json.dumps(chunk.files_touched),
                    chunk.tool_content,
                    chunk.date_text,
                    chunk.transcript_path,
                    chunk.byte_offset,
                    chunk.byte_length,
                    chunk.agent_id,
                    emb_blob,
                ),
            )
            if i % log_interval == 0 or i == total:
                now = time.monotonic()
                elapsed = now - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                logger.info(
                    "FTS5 index: %d/%d chunks (%.0f/s, %.0fs remaining)",
                    i, total, rate, eta,
                )
        self._conn.commit()
        elapsed = time.monotonic() - t0
        logger.info("FTS5 index: committed %d chunks in %.1fs", total, elapsed)

    def load_chunks(self) -> list[TranscriptChunk]:
        """Load all chunks from the database, ordered by rowid."""
        from synapt.recall.core import TranscriptChunk

        rows = self._conn.execute(
            "SELECT id, session_id, timestamp, turn_index, "
            "user_text, assistant_text, tools_used, files_touched, "
            "tool_content, date_text, transcript_path, byte_offset, byte_length, "
            "agent_id "
            "FROM chunks ORDER BY rowid"
        ).fetchall()

        chunks = []
        for r in rows:
            chunks.append(TranscriptChunk(
                id=r["id"],
                session_id=r["session_id"],
                timestamp=r["timestamp"],
                turn_index=r["turn_index"],
                user_text=r["user_text"],
                assistant_text=r["assistant_text"],
                tools_used=json.loads(r["tools_used"]) if r["tools_used"] else [],
                files_touched=json.loads(r["files_touched"]) if r["files_touched"] else [],
                tool_content=r["tool_content"] or "",
                date_text=r["date_text"] or "",
                transcript_path=r["transcript_path"] or "",
                byte_offset=r["byte_offset"] if r["byte_offset"] is not None else -1,
                byte_length=r["byte_length"] if r["byte_length"] is not None else 0,
                agent_id=r["agent_id"],
            ))
        return chunks

    def load_chunk_headers(self) -> list[TranscriptChunk]:
        """Load lightweight chunk metadata, ordered by rowid."""
        from synapt.recall.core import TranscriptChunk

        rows = self._conn.execute(
            "SELECT id, session_id, timestamp, turn_index "
            "FROM chunks ORDER BY rowid"
        ).fetchall()

        return [
            TranscriptChunk(
                id=r["id"],
                session_id=r["session_id"],
                timestamp=r["timestamp"],
                turn_index=r["turn_index"],
                user_text="",
                assistant_text="",
                tools_used=[],
                files_touched=[],
                tool_content="",
                transcript_path="",
                byte_offset=-1,
                byte_length=0,
            )
            for r in rows
        ]

    def load_chunk_by_rowid(self, rowid: int) -> TranscriptChunk | None:
        """Load one chunk by rowid."""
        from synapt.recall.core import TranscriptChunk

        r = self._conn.execute(
            "SELECT id, session_id, timestamp, turn_index, "
            "user_text, assistant_text, tools_used, files_touched, "
            "tool_content, date_text, transcript_path, byte_offset, byte_length "
            "FROM chunks WHERE rowid = ?",
            (rowid,),
        ).fetchone()
        if r is None:
            return None

        return TranscriptChunk(
            id=r["id"],
            session_id=r["session_id"],
            timestamp=r["timestamp"],
            turn_index=r["turn_index"],
            user_text=r["user_text"],
            assistant_text=r["assistant_text"],
            tools_used=json.loads(r["tools_used"]) if r["tools_used"] else [],
            files_touched=json.loads(r["files_touched"]) if r["files_touched"] else [],
            tool_content=r["tool_content"] or "",
            date_text=r["date_text"] or "",
            transcript_path=r["transcript_path"] or "",
            byte_offset=r["byte_offset"] if r["byte_offset"] is not None else -1,
            byte_length=r["byte_length"] if r["byte_length"] is not None else 0,
        )

    def load_chunks_by_rowids(self, rowids: list[int]) -> dict[int, TranscriptChunk]:
        """Load multiple chunks by rowid with bounded SQL query count."""
        if not rowids:
            return {}

        from synapt.recall.core import TranscriptChunk

        loaded: dict[int, TranscriptChunk] = {}
        # Stay below SQLite's commonly configured 999-variable limit.
        for offset in range(0, len(rowids), 900):
            batch = rowids[offset:offset + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                "SELECT rowid, id, session_id, timestamp, turn_index, "
                "user_text, assistant_text, tools_used, files_touched, "
                "tool_content, date_text, transcript_path, byte_offset, byte_length, "
                "agent_id FROM chunks WHERE rowid IN (" + placeholders + ")",
                batch,
            ).fetchall()
            for r in rows:
                loaded[r["rowid"]] = TranscriptChunk(
                    id=r["id"],
                    session_id=r["session_id"],
                    timestamp=r["timestamp"],
                    turn_index=r["turn_index"],
                    user_text=r["user_text"],
                    assistant_text=r["assistant_text"],
                    tools_used=json.loads(r["tools_used"]) if r["tools_used"] else [],
                    files_touched=(
                        json.loads(r["files_touched"]) if r["files_touched"] else []
                    ),
                    tool_content=r["tool_content"] or "",
                    date_text=r["date_text"] or "",
                    transcript_path=r["transcript_path"] or "",
                    byte_offset=r["byte_offset"] if r["byte_offset"] is not None else -1,
                    byte_length=r["byte_length"] if r["byte_length"] is not None else 0,
                    agent_id=r["agent_id"],
                )
        return loaded

    def session_overview(self) -> dict[str, dict]:
        """Return routing and listing metadata without materializing chunks.

        Split into two queries rather than one, deliberately: agent_id is
        NOT a per-session constant (a session can genuinely have multiple
        contributing agents -- resume.py and sharded_db.py both treat
        session_overview's ``agent_ids`` as a real multi-value set, unioned
        across chunks/shards, and resolve scope with ``agent_id in
        agent_ids``), so it stays a real ``GROUP_CONCAT(DISTINCT ...)``
        aggregate over every chunk. transcript_path IS a per-session
        constant (one source transcript file per session_id), so instead
        of folding it into the same per-chunk aggregate -- which would
        require covering it in the same index as the other four columns,
        at a measured ~22% insert-cost premium (recall#1147 follow-on,
        tracked privately) -- it is fetched via one lookup PER SESSION
        (a MIN(rowid) per session_id, restricted to rows that actually
        HAVE a transcript_path, matching the tolerance the old single-query
        MIN(NULLIF(transcript_path, '')) had for some rows being empty).
        Measured: this keeps the same read benefit as covering all five
        columns, at a lower per-insert cost (~18% vs ~22%), because the
        second query does N_SESSIONS lookups, not N_CHUNKS.
        """
        rows = self._conn.execute(
            "SELECT session_id, "
            "MIN(NULLIF(timestamp, '')) AS earliest_ts, "
            "MAX(NULLIF(timestamp, '')) AS latest_ts, "
            "SUM(CASE WHEN turn_index >= 0 THEN 1 ELSE 0 END) AS turn_count, "
            "SUM(CASE WHEN turn_index != -1 AND timestamp IS NOT NULL "
            "          AND timestamp != '' THEN 1 ELSE 0 END) AS activity_count, "
            "MAX(CASE WHEN turn_index != -1 THEN julianday(timestamp) END) AS activity_jd, "
            "MAX(CASE WHEN turn_index != -1 AND julianday(timestamp) IS NULL "
            "         THEN timestamp END) AS activity_raw, "
            "MAX(julianday(timestamp)) AS fallback_jd, "
            "MAX(CASE WHEN julianday(timestamp) IS NULL THEN timestamp END) AS fallback_raw "
            ", GROUP_CONCAT(DISTINCT CASE WHEN turn_index != -1 "
            "THEN NULLIF(agent_id, '') END) AS agent_ids "
            "FROM chunks GROUP BY session_id"
        ).fetchall()

        transcript_paths: dict[str, str] = {
            row["session_id"]: row["transcript_path"]
            for row in self._conn.execute(
                "SELECT session_id, transcript_path FROM chunks "
                "WHERE rowid IN ("
                "  SELECT MIN(rowid) FROM chunks "
                "  WHERE transcript_path != '' GROUP BY session_id"
                ")"
            ).fetchall()
        }

        result: dict[str, dict] = {}
        for row in rows:
            parsed = row["activity_jd"]
            raw = row["activity_raw"]
            if parsed is None and raw is None:
                parsed = row["fallback_jd"]
                raw = row["fallback_raw"]
            if parsed is not None:
                unix_seconds = (float(parsed) - 2440587.5) * 86400
                activity = (1, f"{unix_seconds:020.6f}")
            else:
                activity = (0, raw or "")
            result[row["session_id"]] = {
                "activity": activity,
                "earliest_ts": row["earliest_ts"] or "",
                "latest_ts": row["latest_ts"] or "",
                "turn_count": int(row["turn_count"] or 0),
                "has_real_activity": bool(row["activity_count"]),
                "transcript_path": transcript_paths.get(row["session_id"], ""),
                "agent_ids": frozenset(
                    part for part in (row["agent_ids"] or "").split(",") if part
                ),
            }
        return result

    def session_activity(self) -> dict[str, tuple[int, str]]:
        """Return each session's newest activity without materializing chunks."""
        return {
            session_id: overview["activity"]
            for session_id, overview in self.session_overview().items()
        }

    def load_session_chunks(self, session_id: str) -> list[TranscriptChunk]:
        """Load conversation chunks for one session, oldest turn first."""
        return self.load_session_chunks_many([session_id], include_journal=False).get(
            session_id, []
        )

    def load_session_chunks_many(
        self,
        session_ids: list[str],
        include_journal: bool = True,
    ) -> dict[str, list[TranscriptChunk]]:
        """Load full chunks for a bounded set of sessions."""
        from synapt.recall.core import TranscriptChunk

        grouped: dict[str, list[TranscriptChunk]] = {sid: [] for sid in session_ids}
        if not session_ids:
            return grouped
        for offset in range(0, len(session_ids), 900):
            batch = session_ids[offset:offset + 900]
            placeholders = ",".join("?" for _ in batch)
            journal_clause = "" if include_journal else " AND turn_index >= 0"
            rows = self._conn.execute(
                "SELECT id, session_id, timestamp, turn_index, "
                "user_text, assistant_text, tools_used, files_touched, "
                "tool_content, date_text, transcript_path, byte_offset, byte_length, "
                "agent_id FROM chunks WHERE session_id IN (" + placeholders + ")"
                + journal_clause + " ORDER BY session_id, turn_index",
                batch,
            ).fetchall()
            for r in rows:
                grouped.setdefault(r["session_id"], []).append(
                    TranscriptChunk(
                        id=r["id"],
                        session_id=r["session_id"],
                        timestamp=r["timestamp"],
                        turn_index=r["turn_index"],
                        user_text=r["user_text"],
                        assistant_text=r["assistant_text"],
                        tools_used=(
                            json.loads(r["tools_used"]) if r["tools_used"] else []
                        ),
                        files_touched=(
                            json.loads(r["files_touched"])
                            if r["files_touched"] else []
                        ),
                        tool_content=r["tool_content"] or "",
                        date_text=r["date_text"] or "",
                        transcript_path=r["transcript_path"] or "",
                        byte_offset=(
                            r["byte_offset"] if r["byte_offset"] is not None else -1
                        ),
                        byte_length=(
                            r["byte_length"] if r["byte_length"] is not None else 0
                        ),
                        agent_id=r["agent_id"],
                    )
                )
        return grouped

    def load_session_listing(self, session_ids: list[str]) -> dict[str, list[dict]]:
        """Load only the fields needed to render session summaries."""
        grouped: dict[str, list[dict]] = {sid: [] for sid in session_ids}
        for offset in range(0, len(session_ids), 900):
            batch = session_ids[offset:offset + 900]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                "SELECT id, session_id, turn_index, user_text, files_touched, "
                "transcript_path "
                "FROM chunks WHERE session_id IN (" + placeholders + ") "
                "ORDER BY session_id, turn_index",
                batch,
            ).fetchall()
            for row in rows:
                grouped.setdefault(row["session_id"], []).append({
                    "id": row["id"],
                    "turn_index": row["turn_index"],
                    "user_text": row["user_text"] or "",
                    "files_touched": (
                        json.loads(row["files_touched"])
                        if row["files_touched"] else []
                    ),
                    "transcript_path": row["transcript_path"] or "",
                })
        return grouped

    def chunk_count(self) -> int:
        """Number of chunks in the database."""
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def chunk_count_for_values(self, column: str, values: set[str]) -> int:
        """Count chunks matching a bounded set without materializing their text."""
        if column not in {"id", "session_id"}:
            raise ValueError(f"unsupported chunk count column: {column}")
        ordered = sorted(values)
        total = 0
        for offset in range(0, len(ordered), 900):
            batch = ordered[offset:offset + 900]
            placeholders = ",".join("?" for _ in batch)
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM chunks WHERE {column} IN ({placeholders})",
                batch,
            ).fetchone()
            total += row[0] if row else 0
        return total

    # ------------------------------------------------------------------
    # Cross-session chunk links
    # ------------------------------------------------------------------

    def save_chunk_links(self, links: list[tuple[str, str, float]]) -> None:
        """Replace all cross-session links.

        Each tuple is (source_id, target_id, similarity) where IDs are
        globally unique chunk IDs (e.g. "s001c00:t5"), not rowids.
        """
        self._conn.execute("DELETE FROM chunk_links")
        self._conn.executemany(
            "INSERT INTO chunk_links (source_id, target_id, similarity) "
            "VALUES (?, ?, ?)",
            links,
        )
        self._conn.commit()

    def get_cross_links_batch(
        self, source_ids: list[str], limit_per: int = 3,
    ) -> dict[str, list[tuple[str, float]]]:
        """Get cross-session neighbors for multiple chunks at once.

        Returns {source_id: [(target_id, similarity), ...]}.
        """
        if not source_ids:
            return {}
        placeholders = ",".join("?" * len(source_ids))
        rows = self._conn.execute(
            f"SELECT source_id, target_id, similarity "
            f"FROM chunk_links WHERE source_id IN ({placeholders}) "
            f"ORDER BY source_id, similarity DESC",
            source_ids,
        ).fetchall()
        result: dict[str, list[tuple[str, float]]] = {}
        for src, tgt, sim in rows:
            bucket = result.setdefault(src, [])
            if len(bucket) < limit_per:
                bucket.append((tgt, sim))
        return result

    def has_chunk_links(self) -> bool:
        """True if cross-session links have been built."""
        row = self._conn.execute(
            "SELECT 1 FROM chunk_links LIMIT 1"
        ).fetchone()
        return row is not None

    def chunk_link_count(self) -> int:
        """Number of cross-session links."""
        row = self._conn.execute("SELECT COUNT(*) FROM chunk_links").fetchone()
        return row[0] if row else 0

    def chunk_session_map(self) -> dict[int, str]:
        """Return {rowid: session_id} for all chunks."""
        rows = self._conn.execute(
            "SELECT rowid, session_id FROM chunks"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def chunk_id_map(self) -> dict[int, str]:
        """Return {rowid: chunk_id} for all chunks."""
        rows = self._conn.execute(
            "SELECT rowid, id FROM chunks"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def sample_chunk_texts(self, limit: int = 100) -> list[str]:
        """Return a sample of chunk texts for content profile detection.

        Samples evenly across the dataset (not just the first N) to get
        a representative view of the content type. Concatenates user_text,
        assistant_text, and files_touched for a holistic signal.
        """
        total = self.chunk_count()
        step = max(1, total // limit)
        rows = self._conn.execute(
            "SELECT user_text, assistant_text, files_touched "
            "FROM chunks WHERE rowid % ? = 0 LIMIT ?",
            (step, limit),
        ).fetchall()
        return [
            f"{r[0] or ''} {r[1] or ''} {r[2] or ''}" for r in rows
        ]

    # -- FTS5 search -------------------------------------------------------

    def fts_search(
        self,
        query: str,
        limit: int = 100,
    ) -> list[tuple[int, float]]:
        """Full-text search via FTS5 with BM25 scoring.

        Returns list of (rowid, score) tuples sorted by relevance
        (higher score = more relevant).  FTS5's ``bm25()`` returns
        negative values, so we negate them.

        Multi-term queries use AND first; if AND returns no results and
        the query has 2+ tokens, retries with OR so that chunks matching
        *any* term are surfaced rather than returning nothing.
        """
        fts_query = _escape_fts_query(query)
        if not fts_query:
            return []

        rows = self._conn.execute(
            f"SELECT chunks_fts.rowid, "
            f"  -bm25(chunks_fts, {_FTS_WEIGHTS}) AS score "
            f"FROM chunks_fts "
            f"WHERE chunks_fts MATCH ? "
            f"ORDER BY score DESC "
            f"LIMIT ?",
            (fts_query, limit),
        ).fetchall()

        # OR-fallback: if AND returned nothing and query has multiple tokens
        if not rows and len(_escape_fts_tokens(query)) > 1:
            fts_or = _escape_fts_query(query, use_or=True)
            rows = self._conn.execute(
                f"SELECT chunks_fts.rowid, "
                f"  -bm25(chunks_fts, {_FTS_WEIGHTS}) AS score "
                f"FROM chunks_fts "
                f"WHERE chunks_fts MATCH ? "
                f"ORDER BY score DESC "
                f"LIMIT ?",
                (fts_or, limit),
            ).fetchall()

        return [(r[0], r[1]) for r in rows]

    def fts_search_raw(
        self,
        fts_query: str,
        limit: int = 100,
    ) -> list[tuple[int, float]]:
        """Execute a pre-built FTS5 query (no escaping/tokenization).

        Used for custom query syntax like entity-anchored OR queries.
        Returns list of (rowid, score) tuples sorted by relevance.
        """
        if not fts_query:
            return []
        rows = self._conn.execute(
            f"SELECT chunks_fts.rowid, "
            f"  -bm25(chunks_fts, {_FTS_WEIGHTS}) AS score "
            f"FROM chunks_fts "
            f"WHERE chunks_fts MATCH ? "
            f"ORDER BY score DESC "
            f"LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def fts_search_by_session(
        self,
        query: str,
        session_ids: list[str],
        limit: int = 100,
    ) -> list[tuple[int, float]]:
        """FTS5 search restricted to specific sessions.

        Returns list of (rowid, score) tuples.
        """
        fts_query = _escape_fts_query(query)
        if not fts_query or not session_ids:
            return []

        placeholders = ",".join("?" for _ in session_ids)
        rows = self._conn.execute(
            f"SELECT chunks_fts.rowid, "
            f"  -bm25(chunks_fts, {_FTS_WEIGHTS}) AS score "
            f"FROM chunks_fts "
            f"JOIN chunks c ON c.rowid = chunks_fts.rowid "
            f"WHERE chunks_fts MATCH ? "
            f"  AND c.session_id IN ({placeholders}) "
            f"ORDER BY score DESC "
            f"LIMIT ?",
            (fts_query, *session_ids, limit),
        ).fetchall()

        # OR-fallback for session-scoped search
        if not rows and len(_escape_fts_tokens(query)) > 1:
            fts_or = _escape_fts_query(query, use_or=True)
            rows = self._conn.execute(
                f"SELECT chunks_fts.rowid, "
                f"  -bm25(chunks_fts, {_FTS_WEIGHTS}) AS score "
                f"FROM chunks_fts "
                f"JOIN chunks c ON c.rowid = chunks_fts.rowid "
                f"WHERE chunks_fts MATCH ? "
                f"  AND c.session_id IN ({placeholders}) "
                f"ORDER BY score DESC "
                f"LIMIT ?",
                (fts_or, *session_ids, limit),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    # -- embeddings --------------------------------------------------------

    def get_embeddings(self, rowids: list[int]) -> dict[int, list[float]]:
        """Fetch embeddings for specific rowids.

        Returns {rowid: [float, ...]} for rows that have non-NULL embeddings.
        """
        if not rowids:
            return {}
        result: dict[int, list[float]] = {}
        # Batch in groups of 500 to avoid SQLite variable limit
        for i in range(0, len(rowids), 500):
            batch = rowids[i : i + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"SELECT rowid, embedding FROM chunks "
                f"WHERE rowid IN ({placeholders}) AND embedding IS NOT NULL",
                batch,
            ).fetchall()
            for r in rows:
                try:
                    result[r[0]] = list(struct.unpack(_EMBEDDING_FMT, r[1]))
                except struct.error:
                    continue  # Skip corrupt/wrong-size BLOBs
        return result

    def save_embeddings(self, embeddings: dict[int, list[float]]) -> None:
        """Store embedding BLOBs for specific rowids."""
        cur = self._conn.cursor()
        for rowid, emb in embeddings.items():
            blob = struct.pack(_EMBEDDING_FMT, *emb)
            cur.execute("UPDATE chunks SET embedding = ? WHERE rowid = ?", (blob, rowid))
        self._conn.commit()

    def has_embeddings(self) -> bool:
        """Check if any chunk has an embedding stored."""
        row = self._conn.execute(
            "SELECT 1 FROM chunks WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        return row is not None

    def content_hash(self) -> str:
        """Compute a content hash directly from the DB.

        Streams rows in timestamp-descending order to match the ordering
        used by TranscriptIndex (sorted newest-first).  This ensures the
        hash is identical to the old _materialize_all_chunks() path.
        """
        import hashlib
        h = hashlib.sha256()
        cursor = self._conn.execute(
            "SELECT id, user_text, assistant_text, tool_content "
            "FROM chunks ORDER BY timestamp DESC, rowid DESC"
        )
        for row in cursor:
            h.update(
                f"{row[0]}|{row[1] or ''}|{row[2] or ''}|{row[3] or ''}\n"
                .encode()
            )
        return h.hexdigest()[:16]

    def get_all_embeddings(self) -> dict[int, list[float]]:
        """Load ALL chunk embeddings into memory for embedding-only search.

        Returns {rowid: [float, ...]} for every chunk with a stored embedding.
        For ~3500 chunks × 384 dims this is ~5MB — perfectly fine in memory.
        """
        result: dict[int, list[float]] = {}
        rows = self._conn.execute(
            "SELECT rowid, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        for r in rows:
            try:
                result[r[0]] = list(struct.unpack(_EMBEDDING_FMT, r[1]))
            except struct.error:
                continue
        return result

    def get_all_embeddings_numpy(self) -> "tuple[np.ndarray, list[int]]":
        """Load chunk embeddings directly into a numpy matrix.

        Returns ``(matrix, rowids)`` where *matrix* is an ``(N, 384)``
        float32 array and *rowids* is the corresponding list of chunk
        rowids.  Loads ~14× faster than :meth:`get_all_embeddings` for
        large indexes because it skips creating millions of Python float
        objects — each embedding row is decoded via :func:`numpy.frombuffer`
        straight into the contiguous array.
        """
        import numpy as np

        rows = self._conn.execute(
            "SELECT rowid, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32), []

        rowids = [r[0] for r in rows]
        matrix = np.empty((len(rows), EMBEDDING_DIM), dtype=np.float32)
        for i, r in enumerate(rows):
            matrix[i] = np.frombuffer(r[1], dtype=np.float32)
        return matrix, rowids

    def get_chunk_id_rowid_map(self) -> dict[str, int]:
        """Return ``{chunk_id: rowid}`` for all chunks.

        Used by :class:`TranscriptIndex` to build the bidirectional mapping
        between in-memory chunk indices and SQLite rowids.
        """
        rows = self._conn.execute("SELECT id, rowid FROM chunks").fetchall()
        return {r[0]: r[1] for r in rows}

    # -- knowledge ---------------------------------------------------------

    def save_knowledge_nodes(self, nodes: list[dict]) -> None:
        """Bulk upsert knowledge nodes (INSERT OR REPLACE)."""
        cur = self._conn.cursor()
        for node in nodes:
            cur.execute(
                "INSERT OR REPLACE INTO knowledge "
                "(id, content, category, confidence, source_sessions, "
                " created_at, updated_at, status, superseded_by, "
                " contradiction_note, tags, "
                " valid_from, valid_until, version, lineage_id, "
                " source_turns, source_offsets) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    node["id"],
                    node["content"],
                    node.get("category", ""),
                    node.get("confidence", 0.5),
                    json.dumps(node.get("source_sessions", [])),
                    node.get("created_at", ""),
                    node.get("updated_at", ""),
                    node.get("status", "active"),
                    node.get("superseded_by", ""),
                    node.get("contradiction_note", ""),
                    json.dumps(node.get("tags", [])),
                    node.get("valid_from"),
                    node.get("valid_until"),
                    node.get("version", 1),
                    node.get("lineage_id", ""),
                    json.dumps(node.get("source_turns", [])),
                    json.dumps(node.get("source_offsets", [])),
                ),
            )
        self._conn.commit()

    def upsert_knowledge_node(self, node: dict) -> None:
        """Insert or replace a single knowledge node."""
        self.save_knowledge_nodes([node])

    @staticmethod
    def _knowledge_dict_from_row(r) -> dict:
        """Convert a knowledge table row to a dict."""
        d = {
            "id": r["id"],
            "content": r["content"],
            "category": r["category"],
            "confidence": r["confidence"],
            "source_sessions": json.loads(r["source_sessions"]),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "status": r["status"],
            "superseded_by": r["superseded_by"],
            "contradiction_note": r["contradiction_note"],
            "tags": json.loads(r["tags"]),
        }
        # Phase 8+ columns — graceful fallback for old DBs mid-migration
        for col, default in [
            ("valid_from", None), ("valid_until", None),
            ("version", 1), ("lineage_id", ""),
        ]:
            try:
                d[col] = r[col]
            except (IndexError, KeyError):
                d[col] = default
        # source_turns / source_offsets — added after Phase 8
        try:
            d["source_turns"] = json.loads(r["source_turns"])
        except (IndexError, KeyError):
            d["source_turns"] = []
        try:
            d["source_offsets"] = json.loads(r["source_offsets"])
        except (IndexError, KeyError):
            d["source_offsets"] = []
        return d

    def load_knowledge_nodes(self, status: str | None = None) -> list[dict]:
        """Load knowledge nodes, optionally filtered by status."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM knowledge WHERE status = ? ORDER BY confidence DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM knowledge ORDER BY confidence DESC"
            ).fetchall()
        return [self._knowledge_dict_from_row(r) for r in rows]

    def knowledge_fts_search(
        self,
        query: str,
        limit: int = 20,
        include_historical: bool = False,
    ) -> list[tuple[int, float]]:
        """Full-text search over knowledge nodes via FTS5.

        Returns list of (rowid, score) sorted by relevance.
        If include_historical is True, also returns contradicted/superseded nodes.
        """
        fts_query = _escape_fts_query(query)
        if not fts_query:
            return []

        status_clause = "" if include_historical else "AND k.status = 'active' "
        sql = (
            f"SELECT knowledge_fts.rowid, "
            f"  -bm25(knowledge_fts, {_KNOWLEDGE_FTS_WEIGHTS}) AS score "
            f"FROM knowledge_fts "
            f"JOIN knowledge k ON k.rowid = knowledge_fts.rowid "
            f"WHERE knowledge_fts MATCH ? {status_clause}"
            f"ORDER BY score DESC "
            f"LIMIT ?"
        )
        rows = self._conn.execute(sql, (fts_query, limit)).fetchall()

        # OR-fallback: reuse same SQL with OR-joined query tokens
        if not rows and len(_escape_fts_tokens(query)) > 1:
            fts_or = _escape_fts_query(query, use_or=True)
            rows = self._conn.execute(sql, (fts_or, limit)).fetchall()

        return [(r[0], r[1]) for r in rows]

    def knowledge_lineage(self, lineage_id: str) -> list[dict]:
        """Load all knowledge nodes sharing a lineage, ordered by version."""
        if not lineage_id:
            return []
        rows = self._conn.execute(
            "SELECT * FROM knowledge WHERE lineage_id = ? ORDER BY version ASC",
            (lineage_id,),
        ).fetchall()
        return [self._knowledge_dict_from_row(r) for r in rows]

    # -- knowledge embeddings ----------------------------------------------

    def get_knowledge_embeddings(self) -> dict[int, list[float]]:
        """Load all knowledge node embeddings into memory.

        Returns {rowid: [float, ...]} for nodes with stored embeddings.
        """
        result: dict[int, list[float]] = {}
        rows = self._conn.execute(
            "SELECT rowid, embedding FROM knowledge "
            "WHERE embedding IS NOT NULL AND status = 'active'"
        ).fetchall()
        for r in rows:
            try:
                result[r[0]] = list(struct.unpack(_EMBEDDING_FMT, r[1]))
            except struct.error:
                continue
        return result

    def get_knowledge_embeddings_by_id(self) -> dict[str, list[float]]:
        """Load active knowledge node embeddings keyed by node ID.

        Returns {node_id: [float, ...]} for nodes with stored embeddings.
        """
        result: dict[str, list[float]] = {}
        rows = self._conn.execute(
            "SELECT id, embedding FROM knowledge "
            "WHERE embedding IS NOT NULL AND status = 'active'"
        ).fetchall()
        for r in rows:
            try:
                result[r[0]] = list(struct.unpack(_EMBEDDING_FMT, r[1]))
            except struct.error:
                continue
        return result

    def save_knowledge_embeddings(self, embeddings: dict[int, list[float]]) -> None:
        """Store embedding BLOBs for knowledge nodes by rowid."""
        cur = self._conn.cursor()
        for rowid, emb in embeddings.items():
            blob = struct.pack(_EMBEDDING_FMT, *emb)
            cur.execute(
                "UPDATE knowledge SET embedding = ? WHERE rowid = ?",
                (blob, rowid),
            )
        self._conn.commit()

    def get_knowledge_rowids_without_embeddings(self) -> list[tuple[int, str]]:
        """Return (rowid, content) for active knowledge nodes missing embeddings."""
        rows = self._conn.execute(
            "SELECT rowid, content FROM knowledge "
            "WHERE embedding IS NULL AND status = 'active'"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    # -- pending contradictions --------------------------------------------

    def add_pending_contradiction(
        self,
        old_node_id: str | None,
        new_content: str,
        category: str = "",
        reason: str = "",
        source_sessions: list[str] | None = None,
        detected_by: str = "co-retrieval",
        claim_text: str | None = None,
        valid_from: str | None = None,
        valid_until: str | None = None,
        new_node_id: str | None = None,
    ) -> int:
        """Queue a potential contradiction for user review.

        For system-detected contradictions, *old_node_id* references the
        existing knowledge node.  For user-initiated (free-text) claims,
        *old_node_id* may be ``None`` and *claim_text* contains the
        user's description of the conflicting information.

        *valid_from*/*valid_until* carry the candidate's temporal bounds (if
        any) so they survive to the eventual replacement node when this
        contradiction is confirmed — otherwise they would be lost the moment
        the contradiction is queued.

        *new_node_id* (Fix B, internal design spec, section 10.3): set ONLY by the
        contest path, where the candidate is
        already a real, persisted (contested-status) node at queue time — resolution promotes
        this node rather than creating a new one from *new_content*. ``None`` for every other
        caller (co-retrieval, consolidation contradict, manual flag), where *new_content* is
        still text-only until confirm-time materializes it.

        Returns the ID of the new pending contradiction.
        """
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO pending_contradictions "
            "(old_node_id, new_content, category, reason, "
            " source_sessions, detected_at, detected_by, claim_text, "
            " valid_from, valid_until, new_node_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                old_node_id, new_content, category, reason,
                json.dumps(source_sessions or []), now, detected_by,
                claim_text, valid_from, valid_until, new_node_id,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def list_pending_contradictions(self) -> list[dict]:
        """Return all pending contradictions awaiting user review."""
        rows = self._conn.execute(
            "SELECT * FROM pending_contradictions "
            "WHERE status = 'pending' ORDER BY detected_at DESC"
        ).fetchall()
        result = []
        for r in rows:
            keys = r.keys()
            result.append({
                "id": r["id"],
                "old_node_id": r["old_node_id"],
                "new_content": r["new_content"],
                "category": r["category"],
                "reason": r["reason"],
                "source_sessions": json.loads(r["source_sessions"]),
                "detected_at": r["detected_at"],
                "detected_by": r["detected_by"],
                "claim_text": r["claim_text"] if "claim_text" in keys else None,
                "valid_from": r["valid_from"] if "valid_from" in keys else None,
                "valid_until": r["valid_until"] if "valid_until" in keys else None,
                "new_node_id": r["new_node_id"] if "new_node_id" in keys else None,
            })
        return result

    def has_pending_contradiction_for(self, old_node_id: str) -> bool:
        """Check if a pending contradiction already exists for a node."""
        row = self._conn.execute(
            "SELECT 1 FROM pending_contradictions "
            "WHERE old_node_id = ? AND status = 'pending' LIMIT 1",
            (old_node_id,),
        ).fetchone()
        return row is not None

    def pending_contradiction_count(self) -> int:
        """Return count of pending contradictions (no deserialization)."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM pending_contradictions WHERE status = 'pending'"
        ).fetchone()[0]

    def get_knowledge_node(self, node_id: str) -> dict | None:
        """Fetch a single knowledge node by ID. Returns None if not found."""
        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._knowledge_dict_from_row(row)

    def get_knowledge_rowid(self, node_id: str) -> int | None:
        """Fetch the SQLite rowid for a knowledge node by ID."""
        row = self._conn.execute(
            "SELECT rowid FROM knowledge WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return int(row[0])

    def resolve_contradiction(
        self, contradiction_id: int, status: str = "confirmed",
    ) -> bool:
        """Mark a pending contradiction as confirmed or dismissed.

        Returns True if a row was updated.
        """
        if status not in ("confirmed", "dismissed"):
            return False
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "UPDATE pending_contradictions SET status = ?, resolved_at = ? "
            "WHERE id = ? AND status = 'pending'",
            (status, now, contradiction_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def knowledge_by_rowid(self, rowids: list[int]) -> dict[int, dict]:
        """Fetch knowledge nodes by rowid. Returns {rowid: node_dict}."""
        if not rowids:
            return {}
        result: dict[int, dict] = {}
        for i in range(0, len(rowids), 500):
            batch = rowids[i : i + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"SELECT rowid, * FROM knowledge WHERE rowid IN ({placeholders})",
                batch,
            ).fetchall()
            for r in rows:
                d = self._knowledge_dict_from_row(r)
                result[r["rowid"]] = d
        return result

    def knowledge_count(self) -> int:
        """Number of active knowledge nodes."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM knowledge WHERE status = 'active'"
        ).fetchone()
        return row[0] if row else 0

    # -- clusters ----------------------------------------------------------

    def save_clusters(
        self,
        clusters: list[dict],
        chunk_memberships: list[tuple[str, str, str]],
    ) -> None:
        """Replace all clusters and memberships (full rebuild).

        Args:
            clusters: List of cluster dicts with keys: cluster_id, topic,
                cluster_type, session_ids, branch, date_start, date_end,
                chunk_count, status, created_at, updated_at.
            chunk_memberships: List of (cluster_id, chunk_id, added_at) tuples.
        """
        cur = self._conn.cursor()

        # Clear topic-derived cluster data (preserve access-promoted singletons)
        cur.execute(
            "DELETE FROM cluster_chunks WHERE cluster_id IN "
            "(SELECT cluster_id FROM clusters WHERE cluster_type = 'topic')"
        )
        # Preserve LLM-generated summaries — they're expensive to regenerate
        # and remain valid when cluster_id is unchanged (deterministic ID).
        # Orphaned LLM summaries (cluster ID changed due to membership shift)
        # are cleaned up after new clusters are inserted (see below).
        cur.execute(
            "DELETE FROM cluster_summaries WHERE cluster_id IN "
            "(SELECT cluster_id FROM clusters WHERE cluster_type = 'topic') "
            "AND method != 'llm'"
        )
        cur.execute(
            "DELETE FROM cluster_token_signatures WHERE cluster_id IN "
            "(SELECT cluster_id FROM clusters WHERE cluster_type = 'topic')"
        )
        # Disable FTS trigger during bulk delete, then rebuild
        cur.execute("DROP TRIGGER IF EXISTS clusters_ad")
        cur.execute(
            "DELETE FROM clusters WHERE cluster_type = 'topic'"
        )
        # Wipe FTS for deleted clusters; will rebuild from all clusters at end
        cur.execute("INSERT INTO clusters_fts(clusters_fts) VALUES ('delete-all')")
        # Restore the delete trigger
        cur.execute(
            "CREATE TRIGGER IF NOT EXISTS clusters_ad AFTER DELETE ON clusters BEGIN "
            "  INSERT INTO clusters_fts(clusters_fts, rowid, topic, search_text) "
            "  VALUES ('delete', old.id, old.topic, old.search_text); "
            "END;"
        )

        for c in clusters:
            cur.execute(
                "INSERT INTO clusters "
                "(cluster_id, topic, search_text, cluster_type, session_ids, branch, "
                " date_start, date_end, chunk_count, status, tags, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    c["cluster_id"],
                    c["topic"],
                    c.get("search_text", ""),
                    c.get("cluster_type", "topic"),
                    json.dumps(c.get("session_ids", [])),
                    c.get("branch"),
                    c.get("date_start"),
                    c.get("date_end"),
                    c.get("chunk_count", 0),
                    c.get("status", "active"),
                    json.dumps(c.get("tags", [])),
                    c["created_at"],
                    c["updated_at"],
                ),
            )

        for cluster_id, chunk_id, added_at in chunk_memberships:
            cur.execute(
                "INSERT OR IGNORE INTO cluster_chunks "
                "(cluster_id, chunk_id, added_at) VALUES (?, ?, ?)",
                (cluster_id, chunk_id, added_at),
            )

        for c in clusters:
            sig = c.get("signature_tokens")
            if sig is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO cluster_token_signatures "
                    "(cluster_id, tokens, updated_at) VALUES (?, ?, ?)",
                    (c["cluster_id"], json.dumps(sig), c["updated_at"]),
                )

        # Keep orphaned LLM summaries temporarily — they carry content_hash
        # values that upgrade_large_cluster_summaries() can match against
        # to avoid regenerating identical summaries. Orphans without
        # a content_hash (legacy summaries) are cleaned up immediately.
        cur.execute(
            "DELETE FROM cluster_summaries "
            "WHERE method = 'llm' "
            "AND content_hash IS NULL "
            "AND cluster_id NOT IN (SELECT cluster_id FROM clusters)"
        )

        # Rebuild FTS from all clusters (topic + preserved access singletons)
        cur.execute("INSERT INTO clusters_fts(clusters_fts) VALUES ('rebuild')")
        self._conn.commit()

    def append_clusters(
        self,
        clusters: list[dict],
        chunk_memberships: list[tuple[str, str, str]],
    ) -> None:
        """Insert NEW clusters/memberships without touching existing ones.

        recall#435 follow-on (the recluster maintenance op): ``save_clusters``
        above is a full-corpus REPLACE -- it deletes every ``cluster_type =
        'topic'`` row before inserting, which is correct for a full build but
        would erase every already-clustered chunk if used to add a small
        incremental batch of newly-stale chunks. This is the additive
        counterpart: plain inserts, ``OR IGNORE`` so a rerun over the same
        (already-applied) batch is a no-op rather than a duplicate-key error.
        """
        cur = self._conn.cursor()
        for c in clusters:
            cur.execute(
                "INSERT OR IGNORE INTO clusters "
                "(cluster_id, topic, search_text, cluster_type, session_ids, branch, "
                " date_start, date_end, chunk_count, status, tags, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    c["cluster_id"],
                    c["topic"],
                    c.get("search_text", ""),
                    c.get("cluster_type", "topic"),
                    json.dumps(c.get("session_ids", [])),
                    c.get("branch"),
                    c.get("date_start"),
                    c.get("date_end"),
                    c.get("chunk_count", 0),
                    c.get("status", "active"),
                    json.dumps(c.get("tags", [])),
                    c["created_at"],
                    c["updated_at"],
                ),
            )
        for cluster_id, chunk_id, added_at in chunk_memberships:
            cur.execute(
                "INSERT OR IGNORE INTO cluster_chunks "
                "(cluster_id, chunk_id, added_at) VALUES (?, ?, ?)",
                (cluster_id, chunk_id, added_at),
            )
        for c in clusters:
            sig = c.get("signature_tokens")
            if sig is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO cluster_token_signatures "
                    "(cluster_id, tokens, updated_at) VALUES (?, ?, ?)",
                    (c["cluster_id"], json.dumps(sig), c["updated_at"]),
                )
        cur.execute("INSERT INTO clusters_fts(clusters_fts) VALUES ('rebuild')")
        self._conn.commit()

    def mark_recluster_attempted(self, chunk_ids: list[str], run_id: str) -> None:
        """Record that a recluster run tried these chunk ids.

        ``INSERT OR REPLACE`` -- a chunk attempted again on a later run just
        updates its timestamp/run_id, it does not accumulate history rows.
        """
        if not chunk_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO recluster_attempts (chunk_id, attempted_at, run_id) "
            "VALUES (?, ?, ?)",
            [(cid, now, run_id) for cid in chunk_ids],
        )
        self._conn.commit()

    def get_recluster_attempted_ids(self) -> set[str]:
        """Chunk ids a recluster run has already tried and failed on."""
        return {
            row[0]
            for row in self._conn.execute("SELECT chunk_id FROM recluster_attempts").fetchall()
        }

    def save_cluster_token_signature(
        self, cluster_id: str, tokens: list[str], updated_at: str,
    ) -> None:
        """Persist a cluster's bounded token signature, replacing any prior
        one -- a cluster's membership changing invalidates its old signature
        outright, there is no incremental patch that stays correct."""
        self._conn.execute(
            "INSERT OR REPLACE INTO cluster_token_signatures "
            "(cluster_id, tokens, updated_at) VALUES (?, ?, ?)",
            (cluster_id, json.dumps(tokens), updated_at),
        )
        self._conn.commit()

    def load_cluster_token_signatures(self) -> dict[str, set[str]]:
        """Every persisted cluster signature, keyed by ``cluster_id``.

        Cheap by construction: each signature is capped at a small fixed
        token count (see ``clustering.TOP_SIGNATURE_TOKENS``), so this is
        cluster-count * a constant, never chunk-count or corpus size.
        """
        return {
            row[0]: set(json.loads(row[1]))
            for row in self._conn.execute(
                "SELECT cluster_id, tokens FROM cluster_token_signatures"
            ).fetchall()
        }

    def cluster_ids_with_signature_oldest_first(self) -> list[str]:
        """Every cluster_id with a row in ``cluster_token_signatures``,
        oldest ``updated_at`` first -- selection order for
        ``clustering.redrive_cluster_signatures``'s bounded batches, so a
        drain across repeated calls makes forward progress rather than
        re-checking the same freshly-redriven clusters first."""
        return [
            row[0]
            for row in self._conn.execute(
                "SELECT cluster_id FROM cluster_token_signatures "
                "ORDER BY updated_at ASC"
            ).fetchall()
        ]

    def active_topic_clusters_missing_signature(self) -> list[str]:
        """Active topic cluster ids with no row in ``cluster_token_signatures``
        yet -- the backfill queue for clusters that predate this table."""
        return [
            row[0]
            for row in self._conn.execute(
                "SELECT cl.cluster_id FROM clusters cl "
                "LEFT JOIN cluster_token_signatures sig "
                "  ON sig.cluster_id = cl.cluster_id "
                "WHERE cl.cluster_type = 'topic' AND cl.status = 'active' "
                "  AND sig.cluster_id IS NULL"
            ).fetchall()
        ]

    def load_cluster_member_chunk_ids(self, cluster_ids: list[str]) -> dict[str, list[str]]:
        """Member chunk ids for each of the given clusters, keyed by
        ``cluster_id``. Bounded by the caller's own cluster batch, not the
        corpus -- used to compute a signature from ACTUAL current members."""
        if not cluster_ids:
            return {}
        placeholders = ",".join("?" * len(cluster_ids))
        result: dict[str, list[str]] = {cid: [] for cid in cluster_ids}
        for cluster_id, chunk_id in self._conn.execute(
            f"SELECT cluster_id, chunk_id FROM cluster_chunks "
            f"WHERE cluster_id IN ({placeholders})",
            cluster_ids,
        ).fetchall():
            result[cluster_id].append(chunk_id)
        return result

    def load_cluster_topics(self, cluster_ids: list[str]) -> dict[str, str]:
        """Topic label for each of the given clusters, keyed by
        ``cluster_id``. Bounded by the caller's own set -- used to label a
        merge sample for hand-reading without a caller having to query
        ``clusters`` directly."""
        if not cluster_ids:
            return {}
        placeholders = ",".join("?" * len(cluster_ids))
        return dict(
            self._conn.execute(
                f"SELECT cluster_id, topic FROM clusters WHERE cluster_id IN ({placeholders})",
                cluster_ids,
            ).fetchall()
        )

    def merge_chunks_into_cluster(
        self,
        cluster_id: str,
        chunk_ids: list[str],
        appended_text: str,
        added_at: str,
        run_id: str | None = None,
    ) -> None:
        """Add chunk(s) to an EXISTING cluster without changing its identity.

        Unlike ``append_clusters`` (new clusters formed from a self-batch),
        this grows a cluster that already exists and may already be
        referenced by its ``cluster_id`` elsewhere -- so the id, unlike a
        full rebuild's, does not get recomputed from the new membership set.
        ``chunk_count`` is recomputed from ``cluster_chunks`` itself rather
        than incremented, so a rerun over an already-merged chunk (``INSERT
        OR IGNORE``) cannot inflate the count.

        ``run_id``, if given, is stamped on every membership row this call
        writes (see ``cluster_chunks``'s schema comment) so a bad run can be
        undone by exactly its own rows, not by a shared ``added_at``
        timestamp -- which a live-store incident had to fall back to
        because this column did not exist yet, and a shared timestamp is
        only unambiguous by accident of one write call producing one clock
        reading for every row.

        Also deletes any persisted token signature for this cluster: a
        signature is a snapshot of a specific membership set, and this call
        just changed that set, so the old signature is now describing
        members the cluster no longer has in the shape it was signed. The
        delete re-enters the cluster into the backfill queue (a cluster with
        no signature row) rather than leaving a stale one silently drifting.
        """
        if not chunk_ids:
            return
        cur = self._conn.cursor()
        cur.executemany(
            "INSERT OR IGNORE INTO cluster_chunks (cluster_id, chunk_id, added_at, run_id) "
            "VALUES (?, ?, ?, ?)",
            [(cluster_id, cid, added_at, run_id) for cid in chunk_ids],
        )
        cur.execute(
            "UPDATE clusters SET "
            "  chunk_count = (SELECT COUNT(*) FROM cluster_chunks "
            "                 WHERE cluster_chunks.cluster_id = clusters.cluster_id), "
            "  search_text = substr(search_text || ' ' || ?, 1, 4000), "
            "  updated_at = ? "
            "WHERE cluster_id = ?",
            (appended_text, added_at, cluster_id),
        )
        cur.execute(
            "DELETE FROM cluster_token_signatures WHERE cluster_id = ?", (cluster_id,),
        )
        self._conn.commit()

    def load_clusters(self, status: str = "active") -> list[dict]:
        """Load clusters filtered by status."""
        rows = self._conn.execute(
            "SELECT * FROM clusters WHERE status = ? ORDER BY date_end DESC",
            (status,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "cluster_id": r["cluster_id"],
                "topic": r["topic"],
                "cluster_type": r["cluster_type"],
                "session_ids": json.loads(r["session_ids"]),
                "branch": r["branch"],
                "date_start": r["date_start"],
                "date_end": r["date_end"],
                "chunk_count": r["chunk_count"],
                "status": r["status"],
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def get_cluster_chunks(self, cluster_id: str) -> list[str]:
        """Return chunk IDs belonging to a cluster."""
        rows = self._conn.execute(
            "SELECT chunk_id FROM cluster_chunks WHERE cluster_id = ? "
            "ORDER BY added_at",
            (cluster_id,),
        ).fetchall()
        return [r["chunk_id"] for r in rows]

    def get_cluster_chunk_texts(self, cluster_id: str) -> list[dict]:
        """Return text content for all chunks in a cluster.

        Joins cluster_chunks with chunks to get the actual text.
        Ordered chronologically for coherent summary generation.
        """
        rows = self._conn.execute(
            "SELECT c.user_text, c.assistant_text, c.tools_used, c.files_touched "
            "FROM cluster_chunks cc "
            "JOIN chunks c ON cc.chunk_id = c.id "
            "WHERE cc.cluster_id = ? "
            "ORDER BY c.timestamp",
            (cluster_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_chunk_cluster_map(self) -> dict[str, str]:
        """Return {chunk_id: cluster_id} for all clustered chunks."""
        rows = self._conn.execute(
            "SELECT chunk_id, cluster_id FROM cluster_chunks"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_chunk_cluster_id(self, chunk_id: str) -> str | None:
        """Find the cluster a chunk belongs to, if any."""
        row = self._conn.execute(
            "SELECT cluster_id FROM cluster_chunks WHERE chunk_id = ? LIMIT 1",
            (chunk_id,),
        ).fetchone()
        return row["cluster_id"] if row else None

    def get_cluster_summary(self, cluster_id: str) -> dict | None:
        """Get the summary for a cluster, if it exists."""
        row = self._conn.execute(
            "SELECT * FROM cluster_summaries WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "cluster_id": row["cluster_id"],
            "summary": row["summary"],
            "method": row["method"],
            "token_count": row["token_count"],
            "created_at": row["created_at"],
            "stale": bool(row["stale"]),
        }

    def save_cluster_summary(
        self,
        cluster_id: str,
        summary: str,
        method: str = "concat",
        content_hash: str | None = None,
    ) -> None:
        """Upsert a cluster summary."""
        from datetime import datetime, timezone

        token_count = len(summary) // 4
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO cluster_summaries "
            "(cluster_id, summary, method, token_count, content_hash, "
            " created_at, stale) "
            "VALUES (?, ?, ?, ?, ?, ?, 0)",
            (cluster_id, summary, method, token_count, content_hash, now),
        )
        self._conn.commit()

    def find_summary_by_content_hash(
        self, content_hash: str,
    ) -> str | None:
        """Find an existing LLM summary whose content hash matches.

        Used to reuse expensive LLM summaries when a cluster's ID changes
        (membership shift) but the actual chunk content is unchanged.
        """
        row = self._conn.execute(
            "SELECT summary FROM cluster_summaries "
            "WHERE content_hash = ? AND method = 'llm' "
            "LIMIT 1",
            (content_hash,),
        ).fetchone()
        return row["summary"] if row else None

    def get_cluster(self, cluster_id: str) -> dict | None:
        """Load a single cluster by ID."""
        row = self._conn.execute(
            "SELECT * FROM clusters WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "cluster_id": row["cluster_id"],
            "topic": row["topic"],
            "cluster_type": row["cluster_type"],
            "session_ids": json.loads(row["session_ids"]),
            "branch": row["branch"],
            "date_start": row["date_start"],
            "date_end": row["date_end"],
            "chunk_count": row["chunk_count"],
            "status": row["status"],
            "durability": row["durability"] if "durability" in row.keys() else "",
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def classify_all_clusters(self) -> int:
        """Classify durability for all clusters that haven't been classified yet.

        Returns the number of clusters updated.
        """
        from synapt.recall.hybrid import classify_cluster_durability

        rows = self._conn.execute(
            "SELECT cluster_id, topic FROM clusters WHERE durability = ''"
        ).fetchall()
        if not rows:
            return 0

        updated = 0
        for row in rows:
            cluster_id = row["cluster_id"]
            topic = row["topic"]
            # Get summary if available
            summary_row = self._conn.execute(
                "SELECT summary FROM cluster_summaries WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchone()
            summary = summary_row["summary"] if summary_row else ""

            durability = classify_cluster_durability(topic, summary)
            self._conn.execute(
                "UPDATE clusters SET durability = ? WHERE cluster_id = ?",
                (durability, cluster_id),
            )
            updated += 1

        self._conn.commit()
        return updated

    def cluster_count(self, cluster_type: str | None = None) -> int:
        """Number of active clusters, optionally filtered by type."""
        if cluster_type:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM clusters "
                "WHERE status = 'active' AND cluster_type = ?",
                (cluster_type,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM clusters WHERE status = 'active'"
            ).fetchone()
        return row[0] if row else 0

    def load_timeline_clusters(
        self,
        after: str | None = None,
        before: str | None = None,
        branch: str | None = None,
    ) -> list[dict]:
        """Load timeline clusters with optional date/branch filters.

        Args:
            after: Only arcs ending after this date (ISO 8601).
            before: Only arcs starting before this date (ISO 8601).
            branch: Filter to arcs on this branch.
        """
        query = (
            "SELECT * FROM clusters "
            "WHERE cluster_type = 'timeline' AND status = 'active'"
        )
        params: list[str] = []
        if after:
            query += " AND date_end >= ?"
            params.append(after)
        if before:
            query += " AND date_start <= ?"
            params.append(before)
        if branch:
            query += " AND branch = ?"
            params.append(branch)
        query += " ORDER BY date_start"

        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "cluster_id": r["cluster_id"],
                "topic": r["topic"],
                "cluster_type": r["cluster_type"],
                "session_ids": json.loads(r["session_ids"]),
                "branch": r["branch"],
                "date_start": r["date_start"],
                "date_end": r["date_end"],
                "chunk_count": r["chunk_count"],
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "search_text": r["search_text"],
            }
            for r in rows
        ]

    def clusters_for_chunk(self, chunk_id: str) -> list[str]:
        """Return cluster IDs that contain a given chunk.

        Ordered deterministically by cluster_id. Phase 1 clustering
        produces non-overlapping clusters, but the schema allows
        multi-membership for future phases.
        """
        rows = self._conn.execute(
            "SELECT cluster_id FROM cluster_chunks WHERE chunk_id = ? "
            "ORDER BY cluster_id",
            (chunk_id,),
        ).fetchall()
        return [r["cluster_id"] for r in rows]

    def cluster_fts_search(
        self,
        query: str,
        limit: int = 20,
        include_archived: bool = False,
    ) -> list[tuple[str, float]]:
        """Search clusters via FTS5 on topic + search_text.

        Returns list of (cluster_id, score) sorted by relevance.
        If include_archived is True, also returns archived clusters.
        """
        escaped = _escape_fts_query(query, use_or=True)
        if not escaped:
            return []
        try:
            status_clause = "" if include_archived else "AND c.status = 'active' "
            rows = self._conn.execute(
                "SELECT c.cluster_id, bm25(clusters_fts, 2.0, 1.0) as score "
                "FROM clusters_fts "
                "JOIN clusters c ON c.id = clusters_fts.rowid "
                f"WHERE clusters_fts MATCH ? {status_clause}"
                "AND c.cluster_type != 'timeline' "
                "ORDER BY score ASC LIMIT ?",
                (escaped, limit),
            ).fetchall()
            return [(r["cluster_id"], -r["score"]) for r in rows]
        except Exception:
            return []

    def chunk_fts_to_clusters(
        self,
        query: str,
        limit: int = 50,
        include_archived: bool = False,
    ) -> list[tuple[str, float]]:
        """Search chunks via FTS5 and return parent cluster IDs with scores.

        Fallback for concise mode: when cluster FTS finds nothing, search
        raw chunks and map them to their parent clusters.  Deduplicates by
        cluster_id, keeping the highest score per cluster.

        Returns list of (cluster_id, score) sorted by score descending.
        """
        fts_query = _escape_fts_query(query, use_or=True)
        if not fts_query:
            return []
        status_clause = "" if include_archived else "AND cl.status = 'active' "
        try:
            # Step 1: FTS search for matching chunk rowids + scores.
            # bm25() must be called with the FTS table as the sole FROM
            # source — SQLite flattens subqueries so nesting doesn't help.
            fts_rows = self._conn.execute(
                f"SELECT chunks_fts.rowid, "
                f"  -bm25(chunks_fts, {_FTS_WEIGHTS}) AS score "
                f"FROM chunks_fts "
                f"WHERE chunks_fts MATCH ? "
                f"ORDER BY score DESC LIMIT ?",
                (fts_query, limit * 3),
            ).fetchall()
            if not fts_rows:
                return []

            # Step 2: Map chunk rowids → cluster_ids via JOINs.
            placeholders = ",".join("?" for _ in fts_rows)
            score_by_rowid = {r["rowid"]: r["score"] for r in fts_rows}
            rowids = list(score_by_rowid.keys())
            cluster_rows = self._conn.execute(
                f"SELECT c.rowid AS chunk_rowid, cl.cluster_id "
                f"FROM chunks c "
                f"JOIN cluster_chunks cc ON cc.chunk_id = c.id "
                f"JOIN clusters cl ON cl.cluster_id = cc.cluster_id "
                f"WHERE c.rowid IN ({placeholders}) "
                f"AND cl.cluster_type != 'timeline' {status_clause}",
                rowids,
            ).fetchall()

            # Step 3: Aggregate — keep max score per cluster.
            best: dict[str, float] = {}
            for r in cluster_rows:
                cid = r["cluster_id"]
                score = score_by_rowid[r["chunk_rowid"]]
                if cid not in best or score > best[cid]:
                    best[cid] = score
            result = sorted(best.items(), key=lambda x: x[1], reverse=True)
            return result[:limit]
        except Exception:
            return []

    # -- access tracking ---------------------------------------------------

    def record_access(
        self,
        items: list[dict],
        context: str = "search",
    ) -> None:
        """Record access events and update aggregate stats.

        Each item dict has: item_type, item_id, query (optional),
        score (optional), session_id (optional).

        Args:
            items: List of accessed items to record.
            context: Access context — 'search' (auto-returned results),
                     'context' (explicit drill-down), or 'hook' (session hook).
        """
        if not items:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            for item in items:
                item_type = item["item_type"]
                item_id = item["item_id"]
                self._conn.execute(
                    "INSERT INTO access_log "
                    "(item_type, item_id, query, score, session_id, context, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        item_type,
                        item_id,
                        item.get("query", ""),
                        item.get("score", 0.0),
                        item.get("session_id", ""),
                        context,
                        now,
                    ),
                )
                # Upsert access_stats: increment counts, update timestamps.
                # "search" and "context" are user-initiated (explicit);
                # "hook" is automated and does not count toward promotions.
                # weighted_count accumulates the relevance score for explicit
                # accesses — high-relevance results contribute more to the
                # frequency boost than low-relevance ones.
                is_explicit = 1 if context in ("search", "context") else 0
                score = item.get("score", 0.0) if is_explicit else 0.0
                self._conn.execute(
                    "INSERT INTO access_stats "
                    "(item_type, item_id, access_count, explicit_count, "
                    " weighted_count, first_accessed, last_accessed) "
                    "VALUES (?, ?, 1, ?, ?, ?, ?) "
                    "ON CONFLICT(item_type, item_id) DO UPDATE SET "
                    "access_count = access_count + 1, "
                    "explicit_count = explicit_count + excluded.explicit_count, "
                    "weighted_count = weighted_count + excluded.weighted_count, "
                    "last_accessed = excluded.last_accessed",
                    (item_type, item_id, is_explicit, score, now, now),
                )
                # Update distinct session/query counts from the log
                if is_explicit:
                    row = self._conn.execute(
                        "SELECT COUNT(DISTINCT session_id) AS ds, "
                        "       COUNT(DISTINCT query) AS dq "
                        "FROM access_log "
                        "WHERE item_type = ? AND item_id = ? "
                        "  AND context IN ('search', 'context') "
                        "  AND session_id != '' AND query != ''",
                        (item_type, item_id),
                    ).fetchone()
                    if row:
                        self._conn.execute(
                            "UPDATE access_stats "
                            "SET distinct_sessions = ?, distinct_queries = ? "
                            "WHERE item_type = ? AND item_id = ?",
                            (row["ds"], row["dq"], item_type, item_id),
                        )

    def get_access_stats(
        self,
        item_type: str,
        item_id: str,
    ) -> dict | None:
        """Get access stats for a specific item."""
        row = self._conn.execute(
            "SELECT * FROM access_stats WHERE item_type = ? AND item_id = ?",
            (item_type, item_id),
        ).fetchone()
        if row is None:
            return None
        result = {
            "item_type": row["item_type"],
            "item_id": row["item_id"],
            "access_count": row["access_count"],
            "explicit_count": row["explicit_count"],
            "last_accessed": row["last_accessed"],
            "first_accessed": row["first_accessed"],
        }
        # New columns (may not exist in un-migrated DBs)
        for col, default in [
            ("promotion_tier", "raw"),
            ("distinct_sessions", 0),
            ("distinct_queries", 0),
            ("decay_score", 1.0),
            ("weighted_count", 0.0),
        ]:
            try:
                result[col] = row[col]
            except (IndexError, KeyError):
                result[col] = default
        return result

    def access_summary(self) -> dict:
        """Aggregate access statistics for recall_stats display."""
        row = self._conn.execute(
            "SELECT COUNT(*) as total_events FROM access_log"
        ).fetchone()
        total_events = row["total_events"] if row else 0

        row = self._conn.execute(
            "SELECT COUNT(*) as tracked_items FROM access_stats"
        ).fetchone()
        tracked_items = row["tracked_items"] if row else 0

        row = self._conn.execute(
            "SELECT COUNT(*) as explicit FROM access_stats "
            "WHERE explicit_count > 0"
        ).fetchone()
        items_drilled = row["explicit"] if row else 0

        # Top 5 most-accessed items
        top_items = self._conn.execute(
            "SELECT item_type, item_id, access_count, explicit_count, "
            "last_accessed FROM access_stats "
            "ORDER BY access_count DESC LIMIT 5"
        ).fetchall()

        # Promotion tier distribution
        try:
            tier_rows = self._conn.execute(
                "SELECT promotion_tier, COUNT(*) as cnt FROM access_stats "
                "GROUP BY promotion_tier"
            ).fetchall()
            tier_dist = {r["promotion_tier"]: r["cnt"] for r in tier_rows}
        except Exception:
            tier_dist = {}

        return {
            "total_events": total_events,
            "tracked_items": tracked_items,
            "items_drilled_into": items_drilled,
            "promotion_tiers": tier_dist,
            "top_items": [
                {
                    "item_type": r["item_type"],
                    "item_id": r["item_id"],
                    "access_count": r["access_count"],
                    "explicit_count": r["explicit_count"],
                    "last_accessed": r["last_accessed"],
                }
                for r in top_items
            ],
        }

    def access_log_count(self) -> int:
        """Total number of access log entries."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM access_log"
        ).fetchone()
        return row["cnt"] if row else 0

    # -- promotion pipeline ------------------------------------------------

    def update_promotion_tier(
        self, item_type: str, item_id: str, tier: str,
    ) -> None:
        """Set the promotion tier for an item in access_stats."""
        self._conn.execute(
            "UPDATE access_stats SET promotion_tier = ? "
            "WHERE item_type = ? AND item_id = ?",
            (tier, item_type, item_id),
        )
        self._conn.commit()

    def items_at_tier(self, tier: str) -> list[dict]:
        """Return all access_stats rows at a given promotion tier.

        Ordered by explicit_count DESC so budget-capped operations
        promote the most-accessed items first.
        """
        rows = self._conn.execute(
            "SELECT * FROM access_stats WHERE promotion_tier = ? "
            "ORDER BY explicit_count DESC",
            (tier,),
        ).fetchall()
        results = []
        for row in rows:
            d = {
                "item_type": row["item_type"],
                "item_id": row["item_id"],
                "access_count": row["access_count"],
                "explicit_count": row["explicit_count"],
                "promotion_tier": row["promotion_tier"],
                "distinct_sessions": row["distinct_sessions"],
                "distinct_queries": row["distinct_queries"],
            }
            results.append(d)
        return results

    def create_singleton_cluster(
        self, chunk_id: str, topic: str = "",
    ) -> str | None:
        """Create a singleton cluster for a hot orphan chunk.

        Returns the new cluster_id, or None if chunk doesn't exist.
        The cluster is tagged with cluster_type='access' to distinguish
        it from Jaccard-derived topic clusters during rebuild.
        """
        import hashlib
        row = self._conn.execute(
            "SELECT id FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None

        cluster_id = "clust-" + hashlib.sha1(
            chunk_id.encode()
        ).hexdigest()[:12]

        # Check if already in a cluster
        existing = self._conn.execute(
            "SELECT cluster_id FROM cluster_chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if existing:
            return existing["cluster_id"]

        now = datetime.now(timezone.utc).isoformat()
        if not topic:
            # Extract a topic from the chunk's user_text
            cdata = self._conn.execute(
                "SELECT user_text FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            topic = (cdata["user_text"] or "")[:100] if cdata else ""

        self._conn.execute(
            "INSERT OR IGNORE INTO clusters "
            "(cluster_id, topic, cluster_type, session_ids, date_start, "
            " date_end, chunk_count, status, created_at, updated_at, search_text) "
            "VALUES (?, ?, 'access', '[]', ?, ?, 1, 'active', ?, ?, '')",
            (cluster_id, topic, now, now, now, now),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO cluster_chunks "
            "(cluster_id, chunk_id, added_at) VALUES (?, ?, ?)",
            (cluster_id, chunk_id, now),
        )
        self._conn.commit()
        return cluster_id

    # -- archival + compaction ---------------------------------------------

    def recompute_decay_scores(self, half_life_days: float = 30.0) -> int:
        """Recompute decay_score for all items in access_stats.

        Uses exponential decay based on time since last access:
          decay = 2^(-days_since_last_access / half_life)

        Returns the number of items updated.
        """
        now = datetime.now(timezone.utc)
        rows = self._conn.execute(
            "SELECT item_type, item_id, last_accessed FROM access_stats"
        ).fetchall()
        count = 0
        for row in rows:
            try:
                last = datetime.fromisoformat(row["last_accessed"].replace("Z", "+00:00"))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                days_since = (now - last).total_seconds() / 86400
                if half_life_days <= 0:
                    decay = 1.0
                else:
                    decay = min(1.0, 2.0 ** (-days_since / half_life_days))
            except (ValueError, TypeError, ZeroDivisionError):
                decay = 1.0
            self._conn.execute(
                "UPDATE access_stats SET decay_score = ? "
                "WHERE item_type = ? AND item_id = ?",
                (decay, row["item_type"], row["item_id"]),
            )
            count += 1
        self._conn.commit()
        return count

    def archive_cold_clusters(
        self, decay_threshold: float = 0.1, min_age_days: int = 90,
    ) -> list[str]:
        """Archive clusters with low decay and no recent access.

        Sets status='archived' on clusters whose access_stats show:
        - decay_score < threshold
        - last_accessed > min_age_days ago

        Knowledge nodes are never archived.
        Returns list of archived cluster_ids.
        """
        cutoff = datetime.now(timezone.utc)
        cutoff_iso = (cutoff - timedelta(days=min_age_days)).isoformat()

        rows = self._conn.execute(
            "SELECT item_id FROM access_stats "
            "WHERE item_type = 'cluster' "
            "  AND decay_score < ? "
            "  AND last_accessed < ?",
            (decay_threshold, cutoff_iso),
        ).fetchall()
        archived = []
        for row in rows:
            cluster_id = row["item_id"]
            self._conn.execute(
                "UPDATE clusters SET status = 'archived' "
                "WHERE cluster_id = ? AND status = 'active'",
                (cluster_id,),
            )
            changed = self._conn.execute("SELECT changes()").fetchone()[0]
            if changed > 0:
                archived.append(cluster_id)
        if archived:
            self._conn.commit()
        return archived

    def compact_access_log(self, retention_days: int = 90) -> int:
        """Roll up old access_log entries into daily aggregates.

        Entries older than retention_days are aggregated into
        access_log_archive (one row per item per day) and then deleted
        from access_log.

        Returns the number of log entries compacted.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=retention_days)
        ).strftime("%Y-%m-%d")

        # Count entries to compact
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM access_log "
            "WHERE DATE(created_at) < ?",
            (cutoff,),
        ).fetchone()
        count = row["cnt"] if row else 0
        if count == 0:
            return 0

        # Roll up into daily aggregates
        self._conn.execute(
            "INSERT OR REPLACE INTO access_log_archive "
            "(item_type, item_id, date, access_count, avg_score, queries) "
            "SELECT item_type, item_id, DATE(created_at), COUNT(*), "
            "       AVG(score), JSON_GROUP_ARRAY(DISTINCT query) "
            "FROM access_log "
            "WHERE DATE(created_at) < ? "
            "GROUP BY item_type, item_id, DATE(created_at)",
            (cutoff,),
        )
        # Delete compacted entries
        self._conn.execute(
            "DELETE FROM access_log WHERE DATE(created_at) < ?",
            (cutoff,),
        )
        self._conn.commit()
        return count

    def decay_distribution(self) -> dict[str, int]:
        """Return distribution of items across decay score buckets."""
        buckets = {"fresh": 0, "warm": 0, "cool": 0, "cold": 0}
        rows = self._conn.execute(
            "SELECT decay_score FROM access_stats"
        ).fetchall()
        for row in rows:
            score = row["decay_score"]
            if score > 0.8:
                buckets["fresh"] += 1
            elif score > 0.4:
                buckets["warm"] += 1
            elif score > 0.1:
                buckets["cool"] += 1
            else:
                buckets["cold"] += 1
        return buckets

    # -- metadata ----------------------------------------------------------

    def get_metadata(self, key: str) -> str | None:
        """Read a metadata value by key."""
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Write a metadata value (upsert)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def save_manifest(self, manifest: dict) -> None:
        """Store manifest data as individual metadata keys."""
        cur = self._conn.cursor()
        for k, v in manifest.items():
            cur.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (k, json.dumps(v) if not isinstance(v, str) else v),
            )
        self._conn.commit()

    def load_manifest(self) -> dict:
        """Reconstruct manifest dict from metadata keys."""
        rows = self._conn.execute("SELECT key, value FROM metadata").fetchall()
        manifest: dict = {}
        for r in rows:
            try:
                manifest[r["key"]] = json.loads(r["value"])
            except (json.JSONDecodeError, TypeError):
                manifest[r["key"]] = r["value"]
        return manifest
