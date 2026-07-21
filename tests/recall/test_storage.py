"""Tests for the SQLite storage backend (RecallDB)."""

import struct
import tempfile
from pathlib import Path

import pytest

from synapt.recall.core import TranscriptChunk
from synapt.recall.storage import (
    EMBEDDING_DIM,
    RecallDB,
    _escape_fts_query,
    _escape_fts_tokens,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "index" / "recall.db"


@pytest.fixture
def db(db_path):
    d = RecallDB(db_path)
    yield d
    d.close()


@pytest.fixture
def sample_chunks():
    return [
        TranscriptChunk(
            id="aaa11111:t0",
            session_id="session-aaa",
            timestamp="2026-03-01T10:00:00Z",
            turn_index=0,
            user_text="How do I fix the authentication bug?",
            assistant_text="The auth bug is in login.py line 42. You need to check the token expiry.",
            tools_used=["Read", "Edit"],
            files_touched=["src/auth/login.py", "src/auth/tokens.py"],
        ),
        TranscriptChunk(
            id="aaa11111:t1",
            session_id="session-aaa",
            timestamp="2026-03-01T10:05:00Z",
            turn_index=1,
            user_text="Can you also update the api_index.py file?",
            assistant_text="Updated api_index.py with the new endpoint configuration.",
            tools_used=["Read", "Write"],
            files_touched=["src/api_index.py"],
        ),
        TranscriptChunk(
            id="bbb22222:t0",
            session_id="session-bbb",
            timestamp="2026-03-02T14:00:00Z",
            turn_index=0,
            user_text="Run the swift test suite",
            assistant_text="Running swift test. All 50 tests passed.",
            tools_used=["Bash"],
            files_touched=["eval/swift/tasks.json"],
        ),
    ]


# -- Database creation ----------------------------------------------------

class TestDBCreation:
    def test_creates_db_file(self, db, db_path):
        assert db_path.exists()

    def test_creates_parent_dirs(self, db_path):
        nested = db_path.parent / "deep" / "nested" / "recall.db"
        d = RecallDB(nested)
        assert nested.exists()
        d.close()

    def test_wal_mode_enabled(self, db):
        row = db._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"

    def test_fts_table_exists(self, db):
        row = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        assert row is not None

    def test_idempotent_schema(self, db_path):
        """Opening the same DB twice doesn't crash or duplicate tables."""
        d1 = RecallDB(db_path)
        d1.close()
        d2 = RecallDB(db_path)
        d2.close()

    def test_trigger_recovery(self, db_path, sample_chunks):
        """_ensure_schema recreates missing triggers (crash recovery)."""
        d1 = RecallDB(db_path)
        d1.save_chunks(sample_chunks)
        # Simulate crash: drop a trigger manually
        d1._conn.execute("DROP TRIGGER IF EXISTS chunks_ad")
        d1._conn.commit()
        d1.close()
        # Re-opening should detect the missing trigger and recreate it
        d2 = RecallDB(db_path)
        # Verify all 3 triggers exist
        triggers = {r[0] for r in d2._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' "
            "AND name IN ('chunks_ai', 'chunks_ad', 'chunks_au')"
        ).fetchall()}
        assert triggers == {"chunks_ai", "chunks_ad", "chunks_au"}
        # FTS should still work after recovery
        results = d2.fts_search("authentication")
        assert len(results) > 0
        d2.close()

    def test_context_manager(self, tmp_path):
        """RecallDB works as a context manager."""
        db_path = tmp_path / "ctx.db"
        with RecallDB(db_path) as db:
            assert db.path == db_path
        # Connection closed after with-block (no error on re-open)
        with RecallDB(db_path) as db2:
            assert db2.chunk_count() == 0

    def test_update_trigger_syncs_fts(self, db, sample_chunks):
        """AFTER UPDATE trigger keeps FTS5 in sync when text columns change."""
        db.save_chunks(sample_chunks)
        # Verify search finds original text
        results = db.fts_search("authentication bug")
        assert len(results) > 0

        # Update a single row's text column (use unique chunk ID)
        db._conn.execute(
            "UPDATE chunks SET user_text = 'new topic about databases' WHERE id = 'aaa11111:t0'"
        )
        db._conn.commit()

        # FTS should reflect the update — "authentication" was only in user_text,
        # so the old term should no longer match that column (may still match
        # "auth" in assistant_text, so check the new term is findable)
        new_results = db.fts_search("databases")
        assert len(new_results) > 0


# -- Chunks CRUD ---------------------------------------------------------

class TestChunksCRUD:
    def test_save_load_roundtrip(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        loaded = db.load_chunks()
        assert len(loaded) == 3
        assert loaded[0].id == "aaa11111:t0"
        assert loaded[1].id == "aaa11111:t1"
        assert loaded[2].id == "bbb22222:t0"

    def test_preserves_all_fields(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        loaded = db.load_chunks()
        orig = sample_chunks[0]
        got = loaded[0]
        assert got.session_id == orig.session_id
        assert got.timestamp == orig.timestamp
        assert got.turn_index == orig.turn_index
        assert got.user_text == orig.user_text
        assert got.assistant_text == orig.assistant_text
        assert got.tools_used == orig.tools_used
        assert got.files_touched == orig.files_touched
        assert got.date_text == orig.date_text

    def test_load_chunk_by_rowid_and_headers(self, db, sample_chunks):
        db.save_chunks(sample_chunks)

        headers = db.load_chunk_headers()
        assert len(headers) == len(sample_chunks)
        assert headers[0].user_text == ""
        assert headers[0].assistant_text == ""

        loaded = db.load_chunk_by_rowid(1)
        assert loaded is not None
        assert loaded.id == sample_chunks[0].id
        assert loaded.user_text == sample_chunks[0].user_text

        batch = db.load_chunks_by_rowids([1, 2])
        assert set(batch) == {1, 2}
        assert batch[2].id == sample_chunks[1].id

    def test_save_replaces_existing(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        assert db.chunk_count() == 3
        db.save_chunks(sample_chunks[:1])
        assert db.chunk_count() == 1

    def test_chunk_count(self, db, sample_chunks):
        assert db.chunk_count() == 0
        db.save_chunks(sample_chunks)
        assert db.chunk_count() == 3

    def test_empty_chunks(self, db):
        db.save_chunks([])
        assert db.load_chunks() == []
        assert db.chunk_count() == 0

    def test_tools_files_as_json_arrays(self, db, sample_chunks):
        """tools_used and files_touched are stored as JSON arrays."""
        db.save_chunks(sample_chunks)
        row = db._conn.execute(
            "SELECT tools_used, files_touched FROM chunks WHERE id = ?",
            ("aaa11111:t0",),
        ).fetchone()
        assert row["tools_used"] == '["Read", "Edit"]'
        assert "login.py" in row["files_touched"]


# -- FTS5 search ---------------------------------------------------------

class TestFTSSearch:
    def test_basic_search(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        results = db.fts_search("authentication bug")
        assert len(results) > 0
        # First result should be the auth chunk
        rowid, score = results[0]
        assert score > 0

    def test_returns_positive_scores(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        results = db.fts_search("login")
        for _, score in results:
            assert score > 0

    def test_search_by_session(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        results = db.fts_search_by_session("test", ["session-bbb"])
        assert len(results) > 0
        # Verify all results are from session-bbb
        for rowid, _ in results:
            row = db._conn.execute(
                "SELECT session_id FROM chunks WHERE rowid = ?", (rowid,)
            ).fetchone()
            assert row["session_id"] == "session-bbb"

    def test_search_by_session_excludes_others(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        # "authentication" only appears in session-aaa
        results = db.fts_search_by_session("authentication", ["session-bbb"])
        assert len(results) == 0

    def test_empty_query_returns_empty(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        assert db.fts_search("") == []
        assert db.fts_search("a") == []  # single char filtered

    def test_no_results_for_absent_term(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        results = db.fts_search("nonexistent_term_xyz")
        assert results == []

    def test_search_tools(self, db, sample_chunks):
        """FTS5 indexes tool names."""
        db.save_chunks(sample_chunks)
        results = db.fts_search("Bash")
        assert len(results) > 0

    def test_search_tool_content(self, db):
        """FTS5 indexes tool_content so Bash commands and errors are searchable."""
        chunks = [
            TranscriptChunk(
                id="tc1:t0", session_id="session-tc1",
                timestamp="2026-03-01T10:00:00Z", turn_index=0,
                user_text="run the tests",
                assistant_text="Running tests.",
                tool_content="$ pytest tests/\nFAILED tests/test_auth.py - AssertionError",
            ),
        ]
        db.save_chunks(chunks)
        # Should find via tool_content (not in user_text or assistant_text)
        results = db.fts_search("AssertionError")
        assert len(results) > 0

    def test_search_date_text(self, db, sample_chunks):
        """FTS5 indexes expanded calendar text for month/year queries."""
        db.save_chunks(sample_chunks)
        results = db.fts_search("march 2026")
        assert len(results) > 0

    def test_search_files_with_dots(self, db, sample_chunks):
        """FTS5 with tokenchars '._' handles filenames with dots."""
        db.save_chunks(sample_chunks)
        results = db.fts_search("api_index.py")
        assert len(results) > 0

    def test_search_terms_with_plus(self, db):
        """FTS5 can search literal C++-style terms without collapsing to empty."""
        chunks = [
            TranscriptChunk(
                id="cpp11111:t0",
                session_id="session-cpp",
                timestamp="2026-03-02T10:00:00Z",
                turn_index=0,
                user_text="Sentinel joked about being the C++O and writing C#.",
                assistant_text="C++O means Chief Everything Officer.",
                tools_used=[],
                files_touched=[],
            ),
        ]
        db.save_chunks(chunks)
        assert len(db.fts_search("C++O")) > 0

    def test_limit_respected(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        results = db.fts_search("the", limit=1)
        assert len(results) <= 1

    def test_fts_synced_after_save(self, db, sample_chunks):
        """FTS5 triggers keep the index in sync."""
        db.save_chunks(sample_chunks[:1])
        assert len(db.fts_search("authentication")) > 0
        # Replace with different chunks
        db.save_chunks(sample_chunks[2:])
        assert len(db.fts_search("authentication")) == 0
        assert len(db.fts_search("swift")) > 0

    def test_fts_porter_stemming(self, db, sample_chunks):
        """FTS5 Porter tokenizer matches morphological variants."""
        db.save_chunks(sample_chunks)
        # sample_chunks contain "authentication" — Porter stems it,
        # so "authenticating" should also match via shared stem
        results_base = db.fts_search("authentication")
        results_variant = db.fts_search("authenticating")
        assert len(results_base) > 0
        assert len(results_variant) > 0
        # Both should find the same chunk
        assert results_base[0][0] == results_variant[0][0]


class TestFTSMigration:
    def test_migration_recreates_fts_with_porter(self, tmp_path):
        """Opening a DB with old tokenizer triggers FTS migration."""
        db_path = tmp_path / "recall.db"
        # Create DB with old tokenizer (without porter)
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp TEXT,
                turn_index INTEGER,
                user_text TEXT DEFAULT '',
                assistant_text TEXT DEFAULT '',
                tools_used TEXT DEFAULT '',
                files_touched TEXT DEFAULT '',
                embedding BLOB
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                user_text, assistant_text, tools_used, files_touched,
                content=chunks, content_rowid=rowid,
                tokenize="unicode61 tokenchars '._'"
            );
        """)
        conn.close()

        # Insert data BEFORE migration so we can verify it survives
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO chunks (id, session_id, timestamp, turn_index, "
            "user_text, assistant_text, tools_used, files_touched) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1:t0", "session-aaa", "2026-02-26T10:00:00Z", 0,
             "authentication bug", "Fixed the login issue", "Edit", "auth.py"),
        )
        conn.commit()
        conn.close()

        # Opening with RecallDB should trigger migration
        db = RecallDB(db_path)
        # Verify the FTS table now uses porter
        row = db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        assert "porter" in row[0].lower()
        assert "date_text" in row[0].lower()

        # Critical: verify existing data is searchable after migration
        results = db.fts_search("authentication")
        assert len(results) > 0, "FTS migration lost existing data"
        cols = {
            r[1]
            for r in db._conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        assert "date_text" in cols
        db.close()

    def test_migration_detects_tokenchars_change(self, tmp_path):
        """FTS migration detects tokenizer config changes beyond just porter (#225)."""
        db_path = tmp_path / "recall.db"
        # Create DB with porter but DIFFERENT tokenchars (missing '._')
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp TEXT,
                turn_index INTEGER,
                user_text TEXT DEFAULT '',
                assistant_text TEXT DEFAULT '',
                tools_used TEXT DEFAULT '',
                files_touched TEXT DEFAULT '',
                embedding BLOB
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                user_text, assistant_text, tools_used, files_touched,
                content=chunks, content_rowid=rowid,
                tokenize="porter unicode61"
            );
        """)
        conn.close()

        # Opening should detect the different tokenchars and migrate
        db = RecallDB(db_path)
        row = db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        assert "tokenchars" in row[0].lower(), "Migration should restore tokenchars config"
        db.close()

    def test_no_migration_when_schema_matches(self, tmp_path, sample_chunks):
        """No migration triggered when FTS schema already matches."""
        db_path = tmp_path / "recall.db"
        # Create a fresh DB (uses current schema)
        db = RecallDB(db_path)
        db.save_chunks(sample_chunks)
        db.close()

        # Re-open — should NOT trigger migration (no DROP/CREATE)
        db2 = RecallDB(db_path)
        results = db2.fts_search("authentication")
        assert len(results) > 0, "Data should survive re-open without migration"
        db2.close()


class TestPendingContradictionsNewNodeIdMigration:
    """Fix B (config/design/recall-b3-temporal-conflict-escalation-spec-2026-07-21.md section
    10.10 -- Opus's five-check #2, installed/DB path requirement): pending_contradictions.
    new_node_id is a genuine schema change and needs a real migration for already-installed
    DBs, not a bare CREATE TABLE IF NOT EXISTS assumption. No existing test covered
    _migrate_contradictions_table directly before this one -- this is the first, not a
    pattern deviation from TestFTSMigration above, which it mirrors."""

    def test_migration_adds_new_node_id_to_pre_existing_db(self, tmp_path):
        """Opening a DB with the post-claim_text-era schema (no new_node_id) triggers the
        migration and the column exists afterward."""
        db_path = tmp_path / "recall.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pending_contradictions (
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
                valid_until TEXT
            );
        """)
        conn.close()

        db = RecallDB(db_path)
        cols = {
            r[1] for r in db._conn.execute(
                "PRAGMA table_info(pending_contradictions)"
            ).fetchall()
        }
        assert "new_node_id" in cols
        db.close()

    def test_migration_preserves_pre_existing_row_with_null_new_node_id(self, tmp_path):
        """A row inserted BEFORE the migration survives it, with new_node_id defaulting to
        NULL -- backward-compat, not just column-presence."""
        db_path = tmp_path / "recall.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pending_contradictions (
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
                valid_until TEXT
            );
        """)
        conn.execute(
            "INSERT INTO pending_contradictions "
            "(old_node_id, new_content, source_sessions, detected_at) "
            "VALUES (?, ?, ?, ?)",
            ("old-1", "pre-migration content", "[]", "2026-07-01T00:00:00Z"),
        )
        conn.commit()
        conn.close()

        db = RecallDB(db_path)
        pending = db.list_pending_contradictions()
        assert len(pending) == 1
        assert pending[0]["new_content"] == "pre-migration content"
        assert pending[0]["new_node_id"] is None
        db.close()

    def test_new_node_id_roundtrips_after_migration(self, tmp_path):
        """After migration, a row inserted WITH a real new_node_id round-trips it -- the
        migration doesn't just add the column, add_pending_contradiction actually uses it."""
        db_path = tmp_path / "recall.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pending_contradictions (
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
                valid_until TEXT
            );
        """)
        conn.close()

        db = RecallDB(db_path)
        cid = db.add_pending_contradiction(
            old_node_id="old-1",
            new_content="contested candidate content",
            new_node_id="candidate-node-id-1",
            detected_by="b3-temporal-conflict-escalation",
        )
        pending = db.list_pending_contradictions()
        row = next(p for p in pending if p["id"] == cid)
        assert row["new_node_id"] == "candidate-node-id-1"
        db.close()


# -- FTS query escaping ---------------------------------------------------

class TestFTSEscaping:
    def test_basic_tokens(self):
        assert _escape_fts_query("hello world") == "hello world"

    def test_dot_tokens_quoted(self):
        assert _escape_fts_query("api_index.py") == '"api_index.py"'

    def test_mixed(self):
        result = _escape_fts_query("fix bug in api_index.py")
        assert '"api_index.py"' in result
        assert "fix" in result
        assert "bug" in result

    def test_empty_returns_empty(self):
        assert _escape_fts_query("") == ""

    def test_single_char_filtered(self):
        assert _escape_fts_query("a b c") == ""

    def test_special_chars_stripped(self):
        result = _escape_fts_query("hello! world? foo@bar")
        assert "hello" in result
        assert "world" in result

    def test_plus_tokens_preserved_and_quoted(self):
        result = _escape_fts_query("C++O")
        assert '"c++o"' in result

    def test_fts_keywords_quoted(self):
        """FTS5 reserved keywords (AND, OR, NOT, NEAR) are double-quoted."""
        result = _escape_fts_query("NOT working OR broken")
        assert '"not"' in result
        assert '"or"' in result
        assert "working" in result
        assert "broken" in result
        # Non-keywords should NOT be quoted
        assert '"working"' not in result
        assert '"broken"' not in result

    def test_fts_keyword_case_insensitive(self):
        """FTS5 keywords are matched case-insensitively."""
        result = _escape_fts_query("NEAR miss and bug")
        assert '"near"' in result
        assert '"and"' in result

    def test_or_mode(self):
        """use_or=True joins tokens with OR instead of implicit AND."""
        result = _escape_fts_query("recall bugs issues", use_or=True)
        assert " OR " in result
        assert "recall" in result
        assert "bugs" in result
        assert "issues" in result

    def test_or_mode_single_token(self):
        """Single-token query is identical in AND and OR mode."""
        and_result = _escape_fts_query("recall")
        or_result = _escape_fts_query("recall", use_or=True)
        assert and_result == or_result

    def test_escape_fts_tokens(self):
        """_escape_fts_tokens returns list of escaped tokens."""
        tokens = _escape_fts_tokens("fix bug in api_index.py")
        assert "fix" in tokens
        assert "bug" in tokens
        assert '"api_index.py"' in tokens
        # "in" is a stop word and should be filtered
        assert "in" not in tokens

    def test_escape_fts_tokens_preserves_plus(self):
        tokens = _escape_fts_tokens("C++O")
        assert '"c++o"' in tokens

    def test_stop_words_filtered(self):
        """Stop words are removed from FTS queries to improve AND precision."""
        tokens = _escape_fts_tokens(
            "When did Caroline go to the LGBTQ support group?"
        )
        # Content words preserved
        assert "caroline" in tokens
        assert "lgbtq" in tokens
        assert "support" in tokens
        assert "group" in tokens
        # Stop/question words removed
        assert "when" not in tokens
        assert "did" not in tokens
        assert "the" not in tokens
        assert "to" not in tokens
        assert "go" not in tokens


class TestEntityAnchoredQuery:
    """Tests for _build_entity_anchored_query (multi-hop retrieval)."""

    def test_basic_entity_anchored(self):
        from synapt.recall.storage import _build_entity_anchored_query
        result = _build_entity_anchored_query(
            ["caroline"], ["lgbtq", "support", "group"],
        )
        assert result == "caroline AND (lgbtq OR support OR group)"

    def test_single_content_token(self):
        from synapt.recall.storage import _build_entity_anchored_query
        result = _build_entity_anchored_query(["caroline"], ["lgbtq"])
        assert result == "caroline AND lgbtq"

    def test_empty_entity(self):
        from synapt.recall.storage import _build_entity_anchored_query
        assert _build_entity_anchored_query([], ["lgbtq"]) == ""

    def test_empty_content(self):
        from synapt.recall.storage import _build_entity_anchored_query
        assert _build_entity_anchored_query(["caroline"], []) == ""

    def test_multiple_entities(self):
        from synapt.recall.storage import _build_entity_anchored_query
        result = _build_entity_anchored_query(["john", "smith"], ["drums"])
        assert result == "john smith AND drums"

    def test_dot_token_quoted(self):
        from synapt.recall.storage import _build_entity_anchored_query
        result = _build_entity_anchored_query(["fix"], ["api_index.py"])
        assert '"api_index.py"' in result

    def test_entity_anchored_finds_partial_match(self, db):
        """Entity-anchored query finds chunks with entity + any content term."""
        chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="s1", timestamp="2026-03-01T10:00:00Z",
                turn_index=0,
                user_text="Caroline went to the queer support group meeting.",
                assistant_text="Good for her.",
                tools_used=[], files_touched=[],
            ),
            TranscriptChunk(
                id="s1:t1", session_id="s1", timestamp="2026-03-01T10:05:00Z",
                turn_index=1,
                user_text="John likes pizza",
                assistant_text="Cool.",
                tools_used=[], files_touched=[],
            ),
        ]
        db.save_chunks(chunks)

        from synapt.recall.storage import _build_entity_anchored_query, _FTS_WEIGHTS

        # AND query "caroline lgbtq support group" would fail because
        # "lgbtq" isn't in the chunk. Entity-anchored should work:
        # caroline AND (lgbtq OR support OR group) → matches "support" + "group"
        q = _build_entity_anchored_query(
            ["caroline"], ["lgbtq", "support", "group"],
        )
        rows = db._conn.execute(
            f"SELECT chunks_fts.rowid, -bm25(chunks_fts, {_FTS_WEIGHTS}) AS score "
            f"FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score DESC LIMIT 10",
            (q,),
        ).fetchall()
        assert len(rows) >= 1, "Entity-anchored query should find partial match"
        # The matching chunk should be about Caroline (not John)
        matched_rowids = {r[0] for r in rows}
        # Verify via content
        for rowid in matched_rowids:
            row = db._conn.execute(
                "SELECT user_text FROM chunks WHERE rowid = ?", (rowid,)
            ).fetchone()
            assert "caroline" in row[0].lower() or "support" in row[0].lower()


class TestFTSOrFallback:
    """Tests for the OR-fallback behavior when AND returns no results."""

    def test_or_fallback_on_no_and_matches(self, db):
        """Multi-term query with no AND match falls back to OR (#235)."""
        chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="s1", timestamp="2026-03-01T10:00:00Z",
                turn_index=0,
                user_text="How do I fix the authentication bug?",
                assistant_text="Check the token expiry in login.py.",
                tools_used=[], files_touched=[],
            ),
            TranscriptChunk(
                id="s1:t1", session_id="s1", timestamp="2026-03-01T10:05:00Z",
                turn_index=1,
                user_text="What about performance issues?",
                assistant_text="The database queries are slow.",
                tools_used=[], files_touched=[],
            ),
        ]
        db.save_chunks(chunks)

        # "authentication" is in chunk 0, "performance" is in chunk 1.
        # AND query: no single chunk has both → 0 results.
        # OR fallback: should find both chunks.
        results = db.fts_search("authentication performance")
        assert len(results) >= 1, "OR-fallback should find chunks matching any term"

    def test_and_preferred_when_matches_exist(self, db, sample_chunks):
        """When AND matches exist, OR-fallback is not needed."""
        db.save_chunks(sample_chunks)
        # "authentication bug" both appear in chunk 0
        results = db.fts_search("authentication bug")
        assert len(results) > 0

    def test_single_term_no_fallback(self, db, sample_chunks):
        """Single-term query doesn't trigger OR-fallback."""
        db.save_chunks(sample_chunks)
        results = db.fts_search("nonexistent_term_xyz")
        assert results == []

    def test_session_scoped_or_fallback(self, db):
        """OR-fallback also works for session-scoped searches."""
        chunks = [
            TranscriptChunk(
                id="s1:t0", session_id="s1", timestamp="2026-03-01T10:00:00Z",
                turn_index=0,
                user_text="Fix the authentication bug",
                assistant_text="Done.", tools_used=[], files_touched=[],
            ),
            TranscriptChunk(
                id="s1:t1", session_id="s1", timestamp="2026-03-01T10:05:00Z",
                turn_index=1,
                user_text="What about performance?",
                assistant_text="Optimized.", tools_used=[], files_touched=[],
            ),
            TranscriptChunk(
                id="s2:t0", session_id="s2", timestamp="2026-03-02T10:00:00Z",
                turn_index=0,
                user_text="Unrelated session about cooking",
                assistant_text="Pasta recipe.", tools_used=[], files_touched=[],
            ),
        ]
        db.save_chunks(chunks)

        # "authentication performance" AND → 0 matches in s1.
        # OR-fallback should find chunks in s1 only (not s2).
        results = db.fts_search_by_session("authentication performance", ["s1"])
        assert len(results) >= 1
        # Verify all results are from s1
        for rowid, _ in results:
            row = db._conn.execute(
                "SELECT session_id FROM chunks WHERE rowid = ?", (rowid,)
            ).fetchone()
            assert row["session_id"] == "s1"


# -- Embeddings -----------------------------------------------------------

class TestEmbeddings:
    def _make_embedding(self, seed: float = 1.0):
        """Create a deterministic test embedding."""
        return [seed * (i + 1) / EMBEDDING_DIM for i in range(EMBEDDING_DIM)]

    def test_save_load_roundtrip(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        emb1 = self._make_embedding(1.0)
        emb2 = self._make_embedding(2.0)
        db.save_embeddings({1: emb1, 2: emb2})

        result = db.get_embeddings([1, 2, 3])
        assert 1 in result
        assert 2 in result
        assert 3 not in result  # no embedding stored for rowid 3
        # Check values round-trip with float32 precision
        for i in range(EMBEDDING_DIM):
            assert abs(result[1][i] - emb1[i]) < 1e-6
            assert abs(result[2][i] - emb2[i]) < 1e-6

    def test_empty_rowids(self, db):
        assert db.get_embeddings([]) == {}

    def test_has_embeddings_false(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        assert db.has_embeddings() is False

    def test_has_embeddings_true(self, db, sample_chunks):
        db.save_chunks(sample_chunks)
        db.save_embeddings({1: self._make_embedding()})
        assert db.has_embeddings() is True

    def test_large_batch(self, db, sample_chunks):
        """get_embeddings handles batches >500 rowids."""
        db.save_chunks(sample_chunks)
        emb = self._make_embedding()
        db.save_embeddings({1: emb})
        # Request many rowids (most won't exist)
        rowids = list(range(1, 1000))
        result = db.get_embeddings(rowids)
        assert 1 in result


# -- Metadata --------------------------------------------------------------

class TestMetadata:
    def test_set_get(self, db):
        db.set_metadata("version", "1")
        assert db.get_metadata("version") == "1"

    def test_get_missing_returns_none(self, db):
        assert db.get_metadata("nonexistent") is None

    def test_upsert(self, db):
        db.set_metadata("key", "old")
        db.set_metadata("key", "new")
        assert db.get_metadata("key") == "new"

    def test_manifest_roundtrip(self, db):
        manifest = {
            "chunk_count": 42,
            "session_count": 5,
            "build_timestamp": "2026-03-02T10:00:00",
            "chunks_hash": "abc123",
            "sessions": {"s1": {"chunk_count": 10}},
        }
        db.save_manifest(manifest)
        loaded = db.load_manifest()
        assert loaded["chunk_count"] == 42
        assert loaded["session_count"] == 5
        assert loaded["build_timestamp"] == "2026-03-02T10:00:00"
        assert loaded["chunks_hash"] == "abc123"
        assert loaded["sessions"]["s1"]["chunk_count"] == 10

    def test_mtime(self, db, sample_chunks):
        mtime1 = db.mtime
        assert mtime1 > 0
        db.save_chunks(sample_chunks)
        # mtime should update after write
        mtime2 = db.mtime
        assert mtime2 >= mtime1


class TestAccessTracking:
    """Tests for access_log and access_stats tables."""

    def test_record_search_access(self, db):
        """Search accesses increment both access_count and explicit_count.

        Per design doc: search + context are user-initiated (explicit).
        Only hook accesses are non-explicit.
        """
        db.record_access(
            [{"item_type": "chunk", "item_id": "s1:t0", "score": 4.5}],
            context="search",
        )
        stats = db.get_access_stats("chunk", "s1:t0")
        assert stats is not None
        assert stats["access_count"] == 1
        assert stats["explicit_count"] == 1

    def test_record_context_access(self, db):
        """Context (drill-down) accesses increment both counts."""
        db.record_access(
            [{"item_type": "cluster", "item_id": "clust-abc12345"}],
            context="context",
        )
        stats = db.get_access_stats("cluster", "clust-abc12345")
        assert stats is not None
        assert stats["access_count"] == 1
        assert stats["explicit_count"] == 1

    def test_multiple_accesses_accumulate(self, db):
        """Repeated accesses increment counts correctly."""
        item = [{"item_type": "knowledge", "item_id": "K-001"}]
        db.record_access(item, context="search")
        db.record_access(item, context="search")
        db.record_access(item, context="context")

        stats = db.get_access_stats("knowledge", "K-001")
        assert stats["access_count"] == 3
        assert stats["explicit_count"] == 3  # all 3 are user-initiated

    def test_access_log_entries(self, db):
        """Access log records individual events with timestamps."""
        db.record_access(
            [
                {"item_type": "chunk", "item_id": "s1:t0", "score": 3.0},
                {"item_type": "chunk", "item_id": "s1:t1", "score": 2.5},
            ],
            context="search",
        )
        assert db.access_log_count() == 2

    def test_access_summary(self, db):
        """access_summary returns aggregate statistics."""
        db.record_access(
            [{"item_type": "chunk", "item_id": "s1:t0"}],
            context="search",
        )
        db.record_access(
            [{"item_type": "cluster", "item_id": "clust-abc"}],
            context="context",
        )

        summary = db.access_summary()
        assert summary["total_events"] == 2
        assert summary["tracked_items"] == 2
        assert summary["items_drilled_into"] == 2  # both search + context are explicit
        assert len(summary["top_items"]) == 2
        assert "promotion_tiers" in summary

    def test_get_access_stats_missing(self, db):
        """get_access_stats returns None for untracked items."""
        assert db.get_access_stats("chunk", "nonexistent") is None

    def test_empty_items_noop(self, db):
        """record_access with empty list does nothing."""
        db.record_access([], context="search")
        assert db.access_log_count() == 0

    def test_timestamps_set(self, db):
        """first_accessed and last_accessed are set correctly."""
        db.record_access(
            [{"item_type": "chunk", "item_id": "s1:t0"}],
            context="search",
        )
        stats = db.get_access_stats("chunk", "s1:t0")
        assert stats["first_accessed"] is not None
        assert stats["last_accessed"] is not None
        # first and last should be equal on first access
        assert stats["first_accessed"] == stats["last_accessed"]

    def test_weighted_count_accumulates_score(self, db):
        """weighted_count accumulates the relevance score from each access."""
        db.record_access(
            [{"item_type": "chunk", "item_id": "s1:t0", "score": 0.9}],
            context="search",
        )
        stats = db.get_access_stats("chunk", "s1:t0")
        assert abs(stats["weighted_count"] - 0.9) < 0.001

        # Second access with a lower score adds to the weighted count
        db.record_access(
            [{"item_type": "chunk", "item_id": "s1:t0", "score": 0.2}],
            context="search",
        )
        stats = db.get_access_stats("chunk", "s1:t0")
        assert abs(stats["weighted_count"] - 1.1) < 0.001
        assert stats["explicit_count"] == 2

    def test_weighted_count_zero_for_hook(self, db):
        """Hook accesses do not contribute to weighted_count."""
        db.record_access(
            [{"item_type": "chunk", "item_id": "s1:t0", "score": 5.0}],
            context="hook",
        )
        stats = db.get_access_stats("chunk", "s1:t0")
        assert stats["weighted_count"] == 0.0
        assert stats["explicit_count"] == 0

    def test_weighted_count_missing_score_defaults_zero(self, db):
        """Access without a score key contributes 0.0 to weighted_count."""
        db.record_access(
            [{"item_type": "chunk", "item_id": "s1:t0"}],
            context="search",
        )
        stats = db.get_access_stats("chunk", "s1:t0")
        assert stats["weighted_count"] == 0.0
        assert stats["explicit_count"] == 1

    def test_high_score_contributes_more_than_low_score(self, db):
        """A high-relevance access contributes more to weighted_count."""
        # Item A: 3 high-score accesses
        for _ in range(3):
            db.record_access(
                [{"item_type": "chunk", "item_id": "high:t0", "score": 0.9}],
                context="search",
            )
        # Item B: 3 low-score accesses
        for _ in range(3):
            db.record_access(
                [{"item_type": "chunk", "item_id": "low:t0", "score": 0.2}],
                context="search",
            )
        stats_high = db.get_access_stats("chunk", "high:t0")
        stats_low = db.get_access_stats("chunk", "low:t0")
        # Same explicit_count, but weighted_count differs
        assert stats_high["explicit_count"] == stats_low["explicit_count"] == 3
        assert stats_high["weighted_count"] > stats_low["weighted_count"]
        assert abs(stats_high["weighted_count"] - 2.7) < 0.001
        assert abs(stats_low["weighted_count"] - 0.6) < 0.001
