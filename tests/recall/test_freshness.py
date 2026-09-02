"""A resume that cannot answer must say so, and name the surface it checked.

`synapt resume` printed `Session journal- · showing 0 of 0 turns` against a
store holding 153 real turns. The turns were parsed correctly and archived
correctly; the index had simply not been rebuilt since they landed. Nothing in
the output distinguished that from a session that genuinely has no turns, so
the reader was handed a confident empty and no way to test it.

These tests pin the two obligations that follow:

1. **A freshness verdict names its own surface.** `scanned` is not decoration.
   The cheap check compares the index against the archive; the deep check also
   compares the archive against the live sources. A verdict that omits which
   one it ran reads as covering both, and "fresh" then means two different
   things depending on who is reading.

2. **An empty view carries its provenance.** Empty-and-stale and
   empty-and-fresh are different answers and must not render identically.

The deep leg exists because the cheap shortcut for it does not work: comparing
directory mtimes catches a file being ADDED, and Codex appends to the session's
start-date file — so a directory whose live session grew all day looks
untouched. The obvious optimization is blind to precisely the case that
motivated this module.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from synapt.recall.freshness import IndexFreshness, check_index_freshness


# ---------------------------------------------------------------------------
# Fixture world — a store we fully control, no real transcripts
# ---------------------------------------------------------------------------


def _write_index(
    index_dir: Path,
    source_files: list[dict],
    build_ts: str = "2026-08-06T00:00:00",
    skipped_oversize: list[dict] | None = None,
) -> None:
    """Create a minimal index carrying only the metadata freshness reads."""
    index_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(index_dir / "recall.db")
    con.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('source_files', ?)",
                (json.dumps(source_files),))
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('build_timestamp', ?)", (build_ts,))
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('skipped_oversize', ?)",
                (json.dumps(skipped_oversize or []),))
    con.commit()
    con.close()


def _archive_file(archive_dir: Path, name: str, body: str) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / name
    path.write_text(body)
    return path


def _entry(path: Path) -> dict:
    st = path.stat()
    return {"name": path.name, "mtime": st.st_mtime, "size": st.st_size}


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    """A project whose store is laid out by the SAME helpers production uses.

    Deliberately not hardcoded. An earlier cut of this fixture invented
    ``.synapt/recall/transcripts`` while production archives live at
    ``.synapt/recall/worktrees/<name>/transcripts`` — the tests would have gone
    green against a layout no build ever writes.
    """
    project = tmp_path / "proj"
    project.mkdir()
    _index_dir(project).mkdir(parents=True, exist_ok=True)
    _archive_dir(project).mkdir(parents=True, exist_ok=True)
    return project


def _index_dir(project: Path) -> Path:
    from synapt.recall.core import project_index_dir

    return project_index_dir(project)


def _archive_dir(project: Path) -> Path:
    from synapt.recall.core import project_archive_dir

    return project_archive_dir(project)


# ---------------------------------------------------------------------------
# The cheap leg: index vs archive
# ---------------------------------------------------------------------------


def test_matching_archive_and_manifest_is_fresh(store):
    f = _archive_file(_archive_dir(store), "a.jsonl", "one")
    _write_index(_index_dir(store), [_entry(f)])

    result = check_index_freshness(store)

    assert isinstance(result, IndexFreshness)
    assert result.stale is False
    assert result.new_files == []
    assert result.changed_files == []


def test_archived_file_absent_from_the_manifest_is_stale(store):
    """The demonstrated defect: the rollout was archived, the index was not rebuilt."""
    _archive_file(_archive_dir(store), "rollout-live.jsonl", "turns")
    _write_index(_index_dir(store), [])

    result = check_index_freshness(store)

    assert result.stale is True
    assert "rollout-live.jsonl" in result.new_files
    assert result.changed_files == []


def test_grown_archive_file_is_stale(store):
    """A session that grew after the build — size differs from the manifest."""
    f = _archive_file(_archive_dir(store), "a.jsonl", "one")
    recorded = _entry(f)
    f.write_text("one plus considerably more")
    _write_index(_index_dir(store), [recorded])

    result = check_index_freshness(store)

    assert result.stale is True
    assert "a.jsonl" in result.changed_files
    assert result.new_files == []


def test_skipped_oversize_file_is_not_reported_as_new(store):
    """A file the build examined and rejected for size must be
    labelled distinctly, not folded into "new" -- which would falsely imply a
    plain rebuild resolves it, when the same build will skip it again."""
    archived = _archive_file(_archive_dir(store), "huge.jsonl", "x" * 100)
    _write_index(
        _index_dir(store),
        source_files=[],  # never recorded as known -- it was never parsed
        skipped_oversize=[{"name": "huge.jsonl", "size": archived.stat().st_size}],
    )

    result = check_index_freshness(store)

    assert "huge.jsonl" not in result.new_files
    assert len(result.skipped_oversize) == 1
    assert result.skipped_oversize[0]["name"] == "huge.jsonl"
    assert result.skipped_oversize[0]["size"] == archived.stat().st_size


def test_only_a_skipped_oversize_file_is_not_stale(store):
    """A store whose sole gap is a known, permanent, oversize exclusion is
    CURRENT with respect to everything it is able to index -- staleness means
    "a rebuild would help," which is not true for a file the ceiling refuses
    every time."""
    _archive_file(_archive_dir(store), "huge.jsonl", "x" * 100)
    _write_index(
        _index_dir(store),
        source_files=[],
        skipped_oversize=[{"name": "huge.jsonl", "size": 100}],
    )

    result = check_index_freshness(store)

    assert result.stale is False
    assert result.skipped_oversize != []


def test_verdict_names_the_surface_it_scanned(store):
    """`scanned` is the design's signature: an unlabelled verdict reads as covering both legs."""
    f = _archive_file(_archive_dir(store), "a.jsonl", "one")
    _write_index(_index_dir(store), [_entry(f)])

    shallow = check_index_freshness(store)
    deep = check_index_freshness(store, deep=True)

    assert shallow.scanned == "archive"
    assert deep.scanned == "archive+sources"
    assert shallow.scanned != deep.scanned


def test_remedy_is_a_paste_ready_command(store):
    _archive_file(_archive_dir(store), "a.jsonl", "one")
    _write_index(_index_dir(store), [])

    result = check_index_freshness(store)

    assert result.remedy, "a stale verdict without a remedy makes the reader guess"
    assert "recall build" in result.remedy


def test_build_timestamp_is_reported(store):
    f = _archive_file(_archive_dir(store), "a.jsonl", "one")
    _write_index(_index_dir(store), [_entry(f)], build_ts="2026-08-05T22:50:20")

    result = check_index_freshness(store)

    assert result.build_timestamp == "2026-08-05T22:50:20"


# ---------------------------------------------------------------------------
# Fail-safe: an unreadable answer is not a clean one
# ---------------------------------------------------------------------------


def test_missing_index_is_stale_not_an_exception(store):
    """No index at all is the most stale a store can be, and must not raise."""
    _archive_file(_archive_dir(store), "a.jsonl", "one")

    result = check_index_freshness(store)

    assert result.stale is True


def test_unreadable_manifest_is_stale_rather_than_fresh(store):
    """A verdict we could not compute must never render as 'fresh'."""
    _archive_file(_archive_dir(store), "a.jsonl", "one")
    index_dir = _index_dir(store)
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "recall.db").write_bytes(b"not a database")

    result = check_index_freshness(store)

    assert result.stale is True, "an uncomputable freshness must fail closed, not clean"


# ---------------------------------------------------------------------------
# The deep leg: archive vs live sources
# ---------------------------------------------------------------------------


def test_deep_scan_flags_a_live_source_newer_than_its_archive(store, monkeypatch):
    """The append case: the live session grew, the archive copy did not."""
    archived = _archive_file(_archive_dir(store), "rollout-x.jsonl", "short")
    _write_index(_index_dir(store), [_entry(archived)])

    live_dir = store.parent / "live"
    live_dir.mkdir()
    live = live_dir / "rollout-x.jsonl"
    live.write_text("short plus a whole day of appended turns")

    monkeypatch.setattr(
        "synapt.recall.freshness._live_source_files",
        lambda project_dir: [live],
    )

    shallow = check_index_freshness(store)
    deep = check_index_freshness(store, deep=True)

    assert shallow.stale is False, "the cheap leg cannot see this — that is why deep exists"
    assert deep.stale is True
    assert "rollout-x.jsonl" in deep.new_files + deep.changed_files


def test_deep_scan_agrees_when_archives_are_current(store, monkeypatch):
    archived = _archive_file(_archive_dir(store), "rollout-x.jsonl", "same bytes")
    _write_index(_index_dir(store), [_entry(archived)])

    live_dir = store.parent / "live"
    live_dir.mkdir()
    live = live_dir / "rollout-x.jsonl"
    live.write_text("same bytes")

    monkeypatch.setattr(
        "synapt.recall.freshness._live_source_files",
        lambda project_dir: [live],
    )

    deep = check_index_freshness(store, deep=True)

    assert deep.stale is False
    assert deep.scanned == "archive+sources"


# ---------------------------------------------------------------------------
# The rendering obligation: an empty view carries its provenance
#
# The line a user actually received was:
#
#   "No conversational turns in this session — every indexed chunk was
#    harness output or empty."
#
# That sentence names a CAUSE the code never established. There were no chunks
# for his session in the index at all; the index was stale. The renderer
# reported a property of the session when it had only observed a property of
# the index, and the reader has no way to tell those apart.
# ---------------------------------------------------------------------------


from synapt.recall.resume import ResumeView, format_resume  # noqa: E402


def _empty_view(freshness) -> ResumeView:
    return ResumeView(session_id="019faa41", turns=[], total_turns=0, freshness=freshness)


def _fresh() -> IndexFreshness:
    return IndexFreshness(
        stale=False, new_files=[], changed_files=[],
        build_timestamp="2026-08-06T11:00:00", scanned="archive",
        remedy="synapt recall build --no-embeddings",
    )


def _stale() -> IndexFreshness:
    return IndexFreshness(
        stale=True, new_files=["rollout-live.jsonl"], changed_files=[],
        build_timestamp="2026-08-05T22:50:20", scanned="archive",
        remedy="synapt recall build --no-embeddings",
    )


def test_empty_and_stale_does_not_blame_the_session(store):
    """The defect, pinned: a stale index must not be reported as an empty session."""
    out = format_resume(_empty_view(_stale()))

    assert "every indexed chunk was harness output" not in out, (
        "this asserts a cause the code did not establish"
    )
    assert "stale" in out.lower()
    assert "synapt recall build" in out


def test_empty_and_fresh_is_an_honest_empty(store):
    """When the index IS current, the empty verdict is load-bearing and says so."""
    out = format_resume(_empty_view(_fresh()))

    assert "synapt recall build" not in out, "nothing to remedy when the index is current"
    assert "no conversational turns" in out.lower()


def test_empty_and_stale_renders_differently_from_empty_and_fresh(store):
    """Two different answers must not print the same thing."""
    assert format_resume(_empty_view(_stale())) != format_resume(_empty_view(_fresh()))


def test_empty_with_unknown_freshness_claims_no_cause(store):
    """No freshness information is not evidence of a fresh index."""
    out = format_resume(_empty_view(None))

    assert "every indexed chunk was harness output" not in out


def test_a_stale_index_is_disclosed_even_when_turns_render(store):
    """Turns shown from a stale index may still be missing the newest ones."""
    from synapt.recall.resume import ResumeTurn

    view = ResumeView(
        session_id="019faa41",
        turns=[ResumeTurn(chunk_id="019faa41:t1", turn_index=1,
                          timestamp="2026-08-06T11:00:00Z", user_text="hello",
                          assistant_text="hi", tools_used=[], is_continuation=False)],
        total_turns=1,
        freshness=_stale(),
    )

    out = format_resume(view)

    assert "stale" in out.lower(), "a stale index is disclosed whether or not turns render"
    assert "hello" in out, "disclosure must not replace the content"


# ---------------------------------------------------------------------------
# The wiring itself, at the CLI boundary
#
# Reviewer-2's probe: severing `view = _attach_freshness(view, args)` in
# cmd_resume red NOTHING — 95 tests passed with the feature completely dead.
# Every test above exercises the seam or the renderer directly, so the one line
# that connects them was unwitnessed, and the whole feature sat one deletion
# from silent removal.
#
# That is the defect this change's own commit message records ("the tests were
# green while the feature was dead"), recurring one layer up. Finding it twice
# in one change is the argument for testing the CALL, not only the callee.
# ---------------------------------------------------------------------------


import contextlib  # noqa: E402
import io  # noqa: E402
import sqlite3 as _sqlite3  # noqa: E402
from argparse import Namespace  # noqa: E402

from synapt.recall.core import TranscriptChunk  # noqa: E402


def _run_resume_cli(project: Path):
    """Run cmd_resume against *project*, returning stdout."""
    from synapt.recall.cli import cmd_resume

    args = Namespace(
        index=str(_index_dir(project)), session=None, turns=10, project=project
    )
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            cmd_resume(args)
    except SystemExit:
        pass
    return out.getvalue()


def _real_index(project: Path) -> None:
    """Persist an index the way a build does, with one resumable turn."""
    from synapt.recall.storage import RecallDB

    chunk = TranscriptChunk(
        id="aaaaaaaa:t0", session_id="aaaaaaaa-0000-0000-0000-000000000000",
        timestamp="2026-08-06T10:00:00Z", turn_index=0,
        user_text="a real question", assistant_text="a real answer",
        tools_used=[], files_touched=[], tool_content="",
        transcript_path="", byte_offset=0, byte_length=0,
    )
    index_dir = _index_dir(project)
    index_dir.mkdir(parents=True, exist_ok=True)
    db = RecallDB(index_dir / "recall.db")
    db.save_chunks([chunk])
    db.close()


def _set_manifest(project: Path, source_files: list[dict]) -> None:
    con = _sqlite3.connect(_index_dir(project) / "recall.db")
    con.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('source_files', ?)",
                (json.dumps(source_files),))
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('build_timestamp', '2026-08-05T22:50:20')")
    con.commit()
    con.close()


def test_cmd_resume_prints_the_stale_banner(owned_recall_root, store):
    """The wiring, end to end: a stale store must say so in stdout."""
    _real_index(store)
    _archive_file(_archive_dir(store), "rollout-unindexed.jsonl", "turns nobody indexed")
    _set_manifest(store, [])          # archive holds a file the manifest does not

    out = _run_resume_cli(store)

    assert "STALE" in out.upper(), "cmd_resume did not disclose a stale index"
    assert "recall build" in out, "the banner must carry its remedy"


def test_cmd_resume_stays_quiet_when_the_index_is_current(owned_recall_root, store):
    """Control: the banner is conditional, not unconditional decoration."""
    _real_index(store)
    f = _archive_file(_archive_dir(store), "rollout-indexed.jsonl", "turns")
    _set_manifest(store, [_entry(f)])

    out = _run_resume_cli(store)

    assert "STALE" not in out.upper()
    assert "a real question" in out, "the turns must still render"


def test_cmd_resume_cold_trigger_is_deep_and_source_aware_and_carries_the_verdict(owned_recall_root, store, monkeypatch):
    """The cold no-caller trigger path, pinned at the call site rather than described.

    Fix-forward 1+2: on a no-caller resume the cold trigger now runs ONE
    SOURCE-AWARE (deep=True, source_dir=cwd) freshness probe, and its verdict is
    CARRIED into the render — so `_attach_freshness` does not scan again and
    cannot reclassify a known-stale index. This pins every half at the call site:
    the trigger fires only with no caller, it is deep and source-aware, and when
    it runs the attach leg is skipped. The with-caller cases are the controls that
    keep the original attach contract (cheap, plus deep on an empty view) intact.
    """
    import synapt.recall.freshness as fresh_mod

    calls: list[tuple[bool, object]] = []
    fresh = IndexFreshness(stale=False, build_timestamp="t", scanned="archive")

    def record(project_dir=None, *, index_dir=None, deep=False, source_dir=None):
        calls.append((deep, source_dir))
        return fresh

    monkeypatch.setattr(fresh_mod, "check_index_freshness", record)

    _real_index(store)
    _set_manifest(store, [])

    # Control A: a caller is present, so the cold trigger is skipped. A view with
    # turns pays for exactly one cheap attach leg, no deep, no source_dir.
    from synapt.recall.resume import CallerTranscript
    _caller = CallerTranscript(session_id="caller-0000", path=store / "caller.jsonl", mtime=0.0, size=0)
    monkeypatch.setattr("synapt.recall.resume.caller_transcripts", lambda project: [_caller])
    _run_resume_cli(store)
    assert calls == [(False, None)], "with a caller + turns: one cheap attach leg, no cold trigger, no deep"

    # Control B: a caller is present with an EMPTY view — the attach leg still runs
    # cheap then deep (this leg is unchanged; source_dir stays None on attach).
    calls.clear()
    monkeypatch.setattr(
        "synapt.recall.resume.build_resume_view",
        lambda index, **kw: ResumeView(session_id="aaaaaaaa", turns=[], total_turns=0),
    )
    _run_resume_cli(store)
    assert calls == [(False, None), (True, None)], "with a caller + empty view: attach runs cheap then deep"

    # No caller + empty view: the cold trigger runs ONE deep source-aware probe
    # (source_dir == the render's project/cwd), and because that verdict is carried
    # the attach leg does NOT scan again — exactly one call, deep, source-aware.
    calls.clear()
    monkeypatch.setattr("synapt.recall.resume.caller_transcripts", lambda project: [])
    _run_resume_cli(store)
    assert calls == [(True, store)], "no caller: one deep source-aware trigger probe, attach carries it (no re-scan)"

    # No caller + a view WITH turns: same single deep source-aware trigger, still
    # carried — a non-empty view does not add an attach scan on the cold path.
    calls.clear()
    from synapt.recall.resume import ResumeTurn
    monkeypatch.setattr(
        "synapt.recall.resume.build_resume_view",
        lambda index, **kw: ResumeView(
            session_id="aaaaaaaa",
            turns=[ResumeTurn(chunk_id="aaaaaaaa:t0", turn_index=0, timestamp="t",
                              user_text="q", assistant_text="a", tools_used=[], is_continuation=False)],
            total_turns=1,
        ),
    )
    _run_resume_cli(store)
    assert calls == [(True, store)], "no caller + turns: still one carried deep source-aware probe"


def test_cmd_resume_cold_refresh_discovery_oserror_degrades_no_lock_no_build(owned_recall_root, store, monkeypatch):
    """this change's R2 (Atlas) closure witness, driven through the real cmd_resume.

    On a cold, stale, no-caller resume the pre-load refresh walks the filesystem
    for the newest source, and that walk can raise OSError (denied root, racing
    unlink). It must degrade to the stale render — never abort the read, never
    acquire the lock, never build. Removing the discovery try/except re-escapes
    the OSError through cmd_resume (mutation-verified in
    test_cold_no_caller_refresh)."""
    _real_index(store)
    _set_manifest(store, [])

    lock_calls: list[int] = []
    build_calls: list[int] = []
    monkeypatch.setattr(
        "synapt.recall.freshness.check_index_freshness",
        lambda *a, **k: IndexFreshness(stale=True, build_timestamp="t", scanned="archive"),
    )
    monkeypatch.setattr("synapt.recall.resume.caller_transcripts", lambda project: [])

    def _raise(*a, **k):
        raise OSError("denied")

    monkeypatch.setattr("synapt.recall.cli._newest_source_file", _raise)
    monkeypatch.setattr("synapt.recall.cli._acquire_build_lock", lambda *a, **k: lock_calls.append(1))
    monkeypatch.setattr("synapt.recall.cli._archive_and_build_locked", lambda *a, **k: build_calls.append(1))

    out = _run_resume_cli(store)  # must NOT raise

    assert lock_calls == [], "discovery failed before the lock; nothing acquired"
    assert build_calls == [], "nothing built"
    assert "Session" in out, "the read still produced a resume render"


# ---------------------------------------------------------------------------
# Fix-forward 1+2 (this change's R1, Sentinel): the cold no-caller trigger must be
# SOURCE-AWARE and its verdict CARRIED. The measured defect was a live source
# newer than its archive (deep-stale, cheap-fresh) rendering an old tail with NO
# STALE: the cheap trigger never saw it and the cheap attach called the
# un-refreshed index fresh. These three witnesses drive the real cmd_resume.
# ---------------------------------------------------------------------------


def _write_turn(db_path: Path, cid: str, turn_index: int, q: str, a: str) -> None:
    from synapt.recall.storage import RecallDB

    db = RecallDB(db_path)
    db.save_chunks([TranscriptChunk(
        id=cid, session_id="aaaaaaaa-0000-0000-0000-000000000000",
        timestamp="2026-08-06T10:00:00Z", turn_index=turn_index,
        user_text=q, assistant_text=a, tools_used=[], files_touched=[],
        tool_content="", transcript_path="", byte_offset=0, byte_length=0,
    )])
    db.close()


def test_cmd_resume_cold_pristine_lock_held_still_renders_stale(owned_recall_root, store, monkeypatch):
    """Lock-held leg of the carry: a deep-stale/cheap-fresh index with the build
    lock HELD builds nothing, so the ONLY thing that can warn the reader is the
    carried source-aware verdict. Before the fix this rendered an old tail with no
    STALE (the measured refresh_calls=0 case)."""
    _real_index(store)
    f = _archive_file(_archive_dir(store), "a.jsonl", "one")
    _set_manifest(store, [_entry(f)])                       # cheap: archive == manifest -> FRESH

    monkeypatch.setattr("synapt.recall.resume.caller_transcripts", lambda project: [])
    # deep: a live source the archive has never seen; source scope == the render's cwd.
    # It must EXIST on disk — the deep leg stats each live file and skips OSErrors.
    live = store / "rollout-live-newer.jsonl"
    live.write_text("a live turn never archived")
    monkeypatch.setattr(
        "synapt.recall.freshness._live_source_files",
        lambda root: [live] if Path(root) == store else [],
    )
    # a real newest source so the refresh reaches the lock rather than no_source
    monkeypatch.setattr("synapt.recall.cli._newest_source_file", lambda p: live)
    monkeypatch.setattr("synapt.recall.cli._acquire_build_lock", lambda *a, **k: None)  # HELD
    build_calls: list[int] = []
    monkeypatch.setattr("synapt.recall.cli._archive_and_build_locked",
                        lambda *a, **k: build_calls.append(1))

    out = _run_resume_cli(store)

    assert build_calls == [], "lock held: nothing built"
    assert "STALE" in out.upper(), "a known-stale index with the lock held must still say STALE"
    assert "a real question" in out, "the old turn still renders under the banner"


def test_cmd_resume_cold_pristine_lock_error_still_renders_stale(owned_recall_root, store, monkeypatch):
    """Error leg of the carry: the same deep-stale/cheap-fresh index, but the lock
    ACQUIRE raises. The refresh degrades to `error`, builds nothing, and the
    carried verdict still renders STALE — a failure to refresh must never present
    as a fresh index."""
    _real_index(store)
    f = _archive_file(_archive_dir(store), "a.jsonl", "one")
    _set_manifest(store, [_entry(f)])

    monkeypatch.setattr("synapt.recall.resume.caller_transcripts", lambda project: [])
    live = store / "rollout-live-newer.jsonl"
    live.write_text("a live turn never archived")
    monkeypatch.setattr(
        "synapt.recall.freshness._live_source_files",
        lambda root: [live] if Path(root) == store else [],
    )
    monkeypatch.setattr("synapt.recall.cli._newest_source_file", lambda p: live)

    def _boom(*a, **k):
        raise OSError("lock parent is a file")

    monkeypatch.setattr("synapt.recall.cli._acquire_build_lock", _boom)
    build_calls: list[int] = []
    monkeypatch.setattr("synapt.recall.cli._archive_and_build_locked",
                        lambda *a, **k: build_calls.append(1))

    out = _run_resume_cli(store)

    assert build_calls == [], "acquire raised before the build; nothing built"
    assert "STALE" in out.upper(), "a refresh error must still surface the known-stale index"
    assert "a real question" in out, "the old turn still renders"


def test_cmd_resume_cold_ab_store_split_builds_into_A_and_renders_the_new_turn(owned_recall_root, tmp_path, monkeypatch):
    """The happy path across a store split. GRIPSPACE_ROOT store A owns the index
    the render loads; cwd B is a filesystem SIBLING owning a live source newer than
    A's archive. The source-aware trigger (source=B, archive=A) detects it, the
    refresh targets A (store=A, source=B, no B/.synapt), and the reloaded A renders
    the NEW turn.

    The heavy archive+build is replaced by a side-effect that writes the turn the
    build would produce into A's index, so the RELOAD+RENDER path is exercised for
    real (archiver/builder are covered by their own suites). Real fs, real lock.

    MUTATION deep->cheap: a cheap trigger reads A cheap-fresh, skips the refresh,
    the side-effect never runs, and only the OLD turn renders -> this test reds.
    """
    from argparse import Namespace

    a = tmp_path / "canonical"
    b = tmp_path / "agent-desk"
    a_index = a / ".synapt" / "recall" / "index"
    a_index.mkdir(parents=True)
    b.mkdir()

    # A: one OLD turn, and an archive+manifest that AGREE (cheap FRESH).
    _write_turn(a_index / "recall.db", "aaaaaaaa:t0", 0, "the OLD question", "old answer")
    arch = a / ".synapt" / "recall" / "worktrees" / "wt" / "transcripts"
    arch.mkdir(parents=True)
    af = arch / "a.jsonl"
    af.write_text("one")
    con = sqlite3.connect(a_index / "recall.db")
    con.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('source_files', ?)",
                (json.dumps([_entry(af)]),))
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('build_timestamp', 'ts-old')")
    con.commit()
    con.close()

    # B owns a live source A has never archived (deep-stale, source-scoped to B).
    # It must EXIST on disk — the deep leg stats each live file.
    b_live = b / "rollout-new.jsonl"
    b_live.write_text("a fresh session over on B")
    monkeypatch.setattr("synapt.recall.resume.caller_transcripts", lambda project: [])
    monkeypatch.setattr(
        "synapt.recall.freshness._live_source_files",
        lambda root: [b_live] if Path(root) == b else [],
    )
    monkeypatch.setattr("synapt.recall.cli._newest_source_file", lambda p: b_live)

    captured: dict[str, object] = {}

    def spy_build(project_dir, source_dirs, **kw):
        captured["store"] = Path(project_dir)
        captured["source"] = kw.get("source_dir")
        # write the turn the real incremental build would produce, INTO A.
        _write_turn(_index_dir(Path(project_dir)) / "recall.db",
                    "aaaaaaaa:t1", 1, "the FRESH question", "fresh answer")
        return None

    monkeypatch.setattr("synapt.recall.cli._archive_and_build_locked", spy_build)

    args = Namespace(index=str(a_index), session=None, turns=10, project=b)
    out = _io_run_resume(args)

    assert captured.get("store") == a.resolve(), "the build must target store A (GRIPSPACE_ROOT), resolved"
    assert captured.get("source") == b, "the SOURCE scope stays cwd B"
    assert not (b / ".synapt").exists(), "no cwd-derived secondary store"
    assert "the FRESH question" in out, "the reloaded A must render the newly-built turn"


def _io_run_resume(args) -> str:
    """Run cmd_resume against a hand-built Namespace (project/index may differ)."""
    import contextlib as _cl
    import io as _io

    from synapt.recall.cli import cmd_resume

    out = _io.StringIO()
    try:
        with _cl.redirect_stdout(out), _cl.redirect_stderr(_io.StringIO()):
            cmd_resume(args)
    except SystemExit:
        pass
    return out.getvalue()


# ---------------------------------------------------------------------------
# r2's findings. All three share one root: something downstream was described
# rather than driven, so a fixture supplied what reality does not.
# ---------------------------------------------------------------------------


def _run_real_cli(argv: list[str]) -> str:
    """Drive the REAL parser and dispatch — sys.argv through cli.main().

    The wiring test above built a Namespace by hand and passed `project=`,
    a field the resume subparser never produces (it defines only `session`,
    `--index` and `--turns`). The fixture invented reality, so the test could
    not see that freshness was resolving a different store than the render.
    Nothing short of the real parser catches that class.
    """
    import sys as _sys

    from synapt.recall import cli as _cli

    out = io.StringIO()
    old = _sys.argv
    _sys.argv = argv
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            _cli.main()
    except SystemExit:
        pass
    finally:
        _sys.argv = old
    return out.getvalue()


def test_f1_freshness_follows_the_index_the_render_loaded(store, tmp_path, monkeypatch):
    """F1: `--index` must bind freshness too, not just the render.

    `resume` has no `--project`, so freshness resolved the CWD while the render
    followed `--index`. Pointed at another project's index from a clean cwd,
    the stale banner vanished — a true stale index reported as fine.
    """
    _real_index(store)
    _archive_file(_archive_dir(store), "rollout-unindexed.jsonl", "turns nobody indexed")
    _set_manifest(store, [])

    # The cwd must be a store that is genuinely FRESH. Otherwise a
    # cwd-resolving implementation reports stale for its own reason -- finding
    # no index there -- and the test passes while proving nothing. Verified:
    # with an empty directory as cwd this test passed against the DEFECT.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _index_dir(elsewhere).mkdir(parents=True, exist_ok=True)
    _archive_dir(elsewhere).mkdir(parents=True, exist_ok=True)
    other = _archive_file(_archive_dir(elsewhere), "other.jsonl", "indexed")
    _write_index(_index_dir(elsewhere), [_entry(other)])
    assert check_index_freshness(elsewhere).stale is False, "control: cwd store must be fresh"

    monkeypatch.chdir(elsewhere)

    out = _run_real_cli(["synapt", "resume", "--index", str(_index_dir(store))])

    assert "STALE" in out.upper(), (
        "freshness followed the cwd (which is fresh), not the --index the render loaded"
    )


def test_f2_a_non_list_manifest_fails_closed(store):
    """F2: `source_files` JSON null raised TypeError, swallowed to freshness=None.

    The module contract says unreadable means STALE. Anything that is not a
    list of entries is unreadable, and must never arrive as 'not checked'.
    """
    _archive_file(_archive_dir(store), "a.jsonl", "one")
    index_dir = _index_dir(store)
    index_dir.mkdir(parents=True, exist_ok=True)
    con = _sqlite3.connect(index_dir / "recall.db")
    con.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('source_files', 'null')")
    con.commit()
    con.close()

    result = check_index_freshness(store)

    assert result.stale is True, "a null manifest must fail closed, not raise"


def test_f2_a_scalar_manifest_fails_closed(store):
    """The same contract for any non-list shape, not just null."""
    _archive_file(_archive_dir(store), "a.jsonl", "one")
    index_dir = _index_dir(store)
    index_dir.mkdir(parents=True, exist_ok=True)
    con = _sqlite3.connect(index_dir / "recall.db")
    con.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('source_files', '42')")
    con.commit()
    con.close()

    assert check_index_freshness(store).stale is True


def test_f3_a_grown_file_does_not_read_as_missing(store):
    """F3: 'not yet indexed' is wrong for a file the index already knows.

    A grown transcript and an unseen one need different fixes in the reader's
    head; calling both 'not in the index' sends one of them down the wrong path.
    """
    f = _archive_file(_archive_dir(store), "a.jsonl", "one")
    recorded = _entry(f)
    f.write_text("one plus considerably more")
    _write_index(_index_dir(store), [recorded])

    view = ResumeView(session_id="s", turns=[], total_turns=0,
                      freshness=check_index_freshness(store))
    out = format_resume(view)

    assert "not yet indexed" not in out.lower(), (
        "a changed file is not a missing one"
    )
    assert "grown" in out.lower() or "changed" in out.lower(), (
        "the banner must say which of the two happened"
    )


def test_f1b_deep_leg_enumerates_the_index_stores_live_files_not_the_cwds(store, tmp_path, monkeypatch):
    """The other half of F1: the DEEP leg must follow the bound index too.

    Binding the archive side alone left `_live_source_files` resolving
    independently. Under `--index A` from cwd B — which is the deep trigger's
    guaranteed path, since the trigger fires on cheap-fresh-and-empty — it
    compared A's archive against B's live files and appended them as unseen,
    producing a STALE verdict over A that names another project's files and a
    remedy no rebuild of A can satisfy.
    """
    f = _archive_file(_archive_dir(store), "a.jsonl", "one")
    _write_index(_index_dir(store), [_entry(f)])

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    monkeypatch.chdir(foreign)

    seen: list[Path] = []

    def live_for(project_dir):
        seen.append(project_dir)
        # Faithful to the real function: it answers about the root it is GIVEN.
        # A stub that returned the foreign file unconditionally could not tell a
        # bound call from an unbound one -- it would fail against the fix as
        # readily as against the defect, which is no test at all.
        if project_dir is not None and Path(project_dir) == foreign:
            return [foreign / "rollout-foreign.jsonl"]
        return []

    (foreign / "rollout-foreign.jsonl").write_text("a live session over here")
    monkeypatch.setattr("synapt.recall.freshness._live_source_files", live_for)

    result = check_index_freshness(index_dir=_index_dir(store), deep=True)

    assert "rollout-foreign.jsonl" not in result.new_files + result.changed_files, (
        "the deep leg enumerated the cwd's live files against the bound index's archive"
    )
    assert seen and seen[0] is not None, (
        "the deep leg passed project_dir=None, so enumeration resolved the cwd"
    )
    assert Path(seen[0]) == store, (
        f"the deep leg enumerated {seen[0]}, not the bound index's own root"
    )
