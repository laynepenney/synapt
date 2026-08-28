"""Schema-drift detection for the three FTS tables, and the branch it guards.

Every ``RecallDB`` open compares each FTS table's stored DDL against the
module constant that defines it.  Those comparisons decided *nothing* for as
long as they have existed: ``sqlite_master`` stores a statement without its
trailing terminator, the constants end in ``);``, so "differs" was true on
every open of every store.  Each open therefore dropped three FTS5 indexes,
dropped nine triggers, recreated all of them, and reissued three full
``rebuild`` commands over the whole corpus.

The consequence is a reliability one before it is a performance one.  A
rebuild takes SQLite's write lock, so *opening a store to read it* contended
with any concurrent writer, and under one the open failed rather than waited.

These tests pin both halves: that an unchanged schema is recognised as
unchanged, and that the branch which only runs when it is — the trigger
survival check — does what it claims.  That branch has never executed in
production; the fix starts running it on every open, everywhere, at once.
Each of the three tables gets its own witness pair rather than one standing
in for the others, because they are three independent code paths that merely
look alike.
"""

from __future__ import annotations

import sqlite3

import pytest

from synapt.recall.core import TranscriptChunk
from synapt.recall.storage import (
    RecallDB,
    _CLUSTERS_FTS_TABLE_SQL,
    _FTS_TABLE_SQL,
    _KNOWLEDGE_FTS_TABLE_SQL,
    _normalize_ddl,
)


# Table name -> (detector method, defining constant, trigger names)
FTS_TABLES = {
    "chunks_fts": (
        "_needs_fts_migration",
        _FTS_TABLE_SQL,
        ("chunks_ai", "chunks_ad", "chunks_au"),
    ),
    "knowledge_fts": (
        "_needs_knowledge_fts_migration",
        _KNOWLEDGE_FTS_TABLE_SQL,
        ("knowledge_ai", "knowledge_ad", "knowledge_au"),
    ),
    "clusters_fts": (
        "_needs_clusters_fts_migration",
        _CLUSTERS_FTS_TABLE_SQL,
        ("clusters_ai", "clusters_ad", "clusters_au"),
    ),
}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "index" / "recall.db"


def _triggers_present(path, names):
    """Which of ``names`` exist as triggers, read straight from the file."""
    conn = sqlite3.connect(path)
    try:
        placeholders = ",".join("?" * len(names))
        rows = conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='trigger' "
            f"AND name IN ({placeholders})",
            names,
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _sample_chunk(text="alpha bravo charlie"):
    return TranscriptChunk(
        id="fts11111:t0",
        session_id="session-fts",
        timestamp="2026-08-14T10:00:00Z",
        turn_index=0,
        user_text=text,
        assistant_text=text,
    )


# ---------------------------------------------------------------------------
# The detector itself
# ---------------------------------------------------------------------------


class TestSchemaDriftDetection:
    @pytest.mark.parametrize("table", sorted(FTS_TABLES))
    def test_unchanged_schema_reports_no_drift(self, db_path, table):
        """A store whose schema this code just wrote has not drifted from it.

        This is the whole defect in one assertion.  Before the fix every one
        of these returned True, so the migration branch fired on every open.
        """
        detector = FTS_TABLES[table][0]
        db = RecallDB(db_path)
        try:
            assert getattr(db, detector)() is False, (
                f"{table}: a freshly written schema reported as drifted"
            )
        finally:
            db.close()

    @pytest.mark.parametrize("table", sorted(FTS_TABLES))
    def test_real_drift_is_still_detected(self, db_path, table, monkeypatch):
        """The negative control: the fix must not be "always report no drift".

        A detector that never fires is exactly as broken as one that always
        fires — it would strand a real tokenizer change in a stale index.
        """
        import synapt.recall.storage as storage

        detector, constant, _ = FTS_TABLES[table]
        db = RecallDB(db_path)
        try:
            assert getattr(db, detector)() is False

            const_name = {
                "chunks_fts": "_FTS_TABLE_SQL",
                "knowledge_fts": "_KNOWLEDGE_FTS_TABLE_SQL",
                "clusters_fts": "_CLUSTERS_FTS_TABLE_SQL",
            }[table]
            altered = constant.replace("tokenchars '._+'", "tokenchars '._+-'")
            assert altered != constant, "fixture failed to alter the DDL"
            monkeypatch.setattr(storage, const_name, altered)

            assert getattr(db, detector)() is True, (
                f"{table}: a genuine tokenizer change was not detected"
            )
        finally:
            db.close()

    def test_normalizer_removes_only_the_terminator(self):
        """The mechanism, pinned directly.

        ``strip()`` removes the newline after the semicolon and leaves the
        semicolon, which is why whitespace collapsing alone never fixed this.
        """
        assert _normalize_ddl("CREATE TABLE x(a);") == "create table x(a)"
        assert _normalize_ddl("CREATE  TABLE\n  x(a)\n") == "create table x(a)"
        assert _normalize_ddl("CREATE TABLE x(a) ;\n") == "create table x(a)"
        # A semicolon that is not a terminator must survive.
        assert _normalize_ddl("CREATE TABLE x(a, b ';')") == "create table x(a, b ';')"

    @pytest.mark.parametrize("table", sorted(FTS_TABLES))
    def test_stored_ddl_and_constant_agree_after_normalization(self, db_path, table):
        """Compare the two forms directly, independent of the detector.

        If this fails the detector is not the thing that is wrong.
        """
        constant = FTS_TABLES[table][1]
        db = RecallDB(db_path)
        try:
            row = db._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            assert row is not None, f"{table} was never created"
            assert _normalize_ddl(row[0]) == _normalize_ddl(constant)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# The branch that only runs when there is no drift
# ---------------------------------------------------------------------------


class TestTriggerSurvivalBranch:
    """Witnesses for the ``else`` arm at each of the three call sites.

    Never executed in production before this fix, because the detector above
    always sent control down the migration arm instead.  Three separate arms,
    three separate witness pairs.
    """

    @pytest.mark.parametrize("table", sorted(FTS_TABLES))
    def test_missing_trigger_is_restored_on_open(self, db_path, table):
        """A trigger lost to a crash mid-``save_chunks`` is reinstalled."""
        triggers = FTS_TABLES[table][2]
        RecallDB(db_path).close()
        assert _triggers_present(db_path, triggers) == set(triggers)

        casualty = triggers[1]
        conn = sqlite3.connect(db_path)
        conn.execute(f"DROP TRIGGER {casualty}")
        conn.commit()
        conn.close()
        assert casualty not in _triggers_present(db_path, triggers), (
            "fixture did not actually remove the trigger"
        )

        RecallDB(db_path).close()

        assert _triggers_present(db_path, triggers) == set(triggers), (
            f"{table}: reopening did not restore {casualty}"
        )

    @pytest.mark.parametrize("table", sorted(FTS_TABLES))
    def test_all_triggers_present_is_a_clean_no_op(self, db_path, table):
        """With nothing missing and nothing drifted, the open changes nothing.

        Proving *absence* of a rebuild needs an observable a rebuild would
        destroy, so the fixture deliberately desyncs the FTS index from its
        content table and then restores the triggers.  A no-op leaves the
        desync in place.  A rebuild silently repairs it — so if this store is
        still desynced after reopening, the migration arm did not run.
        """
        triggers = FTS_TABLES[table][2]
        RecallDB(db_path).close()
        assert _triggers_present(db_path, triggers) == set(triggers)

        # Desync, without touching any content table so the fixture is the
        # same shape for all three: write an orphan entry straight into the index.
        # An external-content FTS5 table accepts it, and it has no backing
        # content row, so a rebuild erases it and a no-op preserves it.
        conn = sqlite3.connect(db_path)
        column = conn.execute(f"PRAGMA table_info({table})").fetchall()[0][1]
        conn.execute(
            f"INSERT INTO {table}(rowid, {column}) VALUES (?, ?)",
            (10**6, "zzorphanmarker"),
        )
        conn.commit()
        stale = conn.execute(
            f"SELECT count(*) FROM {table} WHERE {table} MATCH 'zzorphanmarker'"
        ).fetchone()[0]
        conn.close()
        assert stale == 1, f"{table}: fixture failed to create a orphaned index entry"

        db = RecallDB(db_path)
        try:
            assert getattr(db, FTS_TABLES[table][0])() is False
        finally:
            db.close()

        assert _triggers_present(db_path, triggers) == set(triggers)

        conn = sqlite3.connect(db_path)
        still_stale = conn.execute(
            f"SELECT count(*) FROM {table} WHERE {table} MATCH 'zzorphanmarker'"
        ).fetchone()[0]
        conn.close()
        assert still_stale == stale, (
            f"{table}: the orphaned index entry was erased, so a rebuild ran on a "
            "store that had not drifted"
        )

    def test_intact_store_keeps_its_content_across_reopen(self, db_path):
        """The plain case, stated because it is what users actually do."""
        db = RecallDB(db_path)
        db.save_chunks([_sample_chunk("delta echo foxtrot")])
        db.close()

        db = RecallDB(db_path)
        try:
            assert db._needs_fts_migration() is False
            hits = db._conn.execute(
                "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'echo'"
            ).fetchone()[0]
            assert hits == 1
        finally:
            db.close()
