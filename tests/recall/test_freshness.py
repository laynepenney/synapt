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


def _write_index(index_dir: Path, source_files: list[dict], build_ts: str = "2026-08-06T00:00:00") -> None:
    """Create a minimal index carrying only the metadata freshness reads."""
    index_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(index_dir / "recall.db")
    con.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('source_files', ?)",
                (json.dumps(source_files),))
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('build_timestamp', ?)", (build_ts,))
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
