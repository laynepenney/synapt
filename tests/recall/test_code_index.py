"""Contract for the code symbol index (recall#940).

The index is an OSS primitive: it maps source files to the symbols they
declare. It knows nothing about sessions, transcripts or memory — those joins
live elsewhere, and a test here that reached for them would be a boundary
violation rather than a missing feature.

The load-bearing claim in the spec is the incremental one: *a full re-run on an
unchanged repo touches zero rows*. "Touches" is measured, not inferred —
``IndexStats.rows_written`` carries SQLite's own ``total_changes`` delta, so a
regression that silently rewrote every row on every run would turn this suite
red instead of merely being slower.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from synapt.recall.code_index import (
    file_outline,
    find_symbols,
    index_repo,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

PY_SRC = '''\
import os
from a.b import helper

CONST_VALUE = 1


def top_level(x):
    """A top-level function."""
    return x


class Widget:
    def render(self):
        return 1

    def resize(self):
        return 2
'''

TS_SRC = '''\
import {thing} from "mod";

export const RATE: number = 3;

export function compute(a: string): number {
  return 1;
}

export class Engine {
  start(): void {}
}

interface Shape { x: number }

type Alias = string;

enum Color { Red }
'''

JS_SRC = '''\
import {thing} from "mod";

const RATE = 3;

function compute(a) { return a; }

class Engine {
  start() {}
}
'''

RS_SRC = '''\
use std::fmt;

const LIMIT: u32 = 4;

pub fn compute(x: u32) -> u32 { x }

struct Engine { a: u32 }

enum Mode { Fast }

trait Runnable { fn run(&self); }

impl Engine {
    pub fn start(&self) {}
}
'''

GO_SRC = '''\
package main

import "fmt"

const Limit = 5

func Compute(a int) int { return a }

type Engine struct { A int }

type Runnable interface { Run() }

func (e Engine) Start() {}
'''


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small multi-language repo."""
    src = tmp_path / "repo"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "widget.py").write_text(PY_SRC)
    (src / "engine.ts").write_text(TS_SRC)
    (src / "engine.js").write_text(JS_SRC)
    (src / "engine.rs").write_text(RS_SRC)
    (src / "engine.go").write_text(GO_SRC)
    return src


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "code.db"


def _rows(db: Path, sql: str, args: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_schema_matches_the_spec(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")

    tables = {r[0] for r in _rows(db, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"code_files", "code_symbols", "code_imports"} <= tables

    indexes = {r[0] for r in _rows(db, "SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_symbols_name" in indexes
    assert "idx_symbols_file" in indexes

    cols = {r[1] for r in _rows(db, "PRAGMA table_info(code_files)")}
    assert {"repo", "path", "lang", "content_hash", "mtime", "indexed_at"} <= cols


def test_repo_path_pair_is_unique(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT repo, path FROM code_files LIMIT 1"
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO code_files (repo, path, lang, content_hash) VALUES (?,?,?,?)",
                (row[0], row[1], "python", "deadbeef"),
            )
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Extraction — the five MVP languages
# --------------------------------------------------------------------------


def _names(db: Path, kind: str | None = None, lang: str | None = None) -> set[str]:
    sql = (
        "SELECT s.name FROM code_symbols s JOIN code_files f ON f.id = s.file_id WHERE 1=1"
    )
    args: list = []
    if kind:
        sql += " AND s.kind = ?"
        args.append(kind)
    if lang:
        sql += " AND f.lang = ?"
        args.append(lang)
    return {r[0] for r in _rows(db, sql, tuple(args))}


def test_python_symbols(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    assert "top_level" in _names(db, lang="python")
    assert "Widget" in _names(db, kind="class", lang="python")
    assert {"render", "resize"} <= _names(db, lang="python")


def test_typescript_symbols_including_exported_ones(repo: Path, db: Path):
    """Most TS declarations are wrapped in `export_statement`.

    A walk that only inspected top-level children would find none of them and
    still report success, so this asserts the exported names specifically.
    """
    index_repo(repo, db, repo="demo")
    ts = _names(db, lang="typescript")
    assert {"compute", "Engine", "Shape", "Color", "RATE"} <= ts


def test_javascript_symbols(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    js = _names(db, lang="javascript")
    assert {"compute", "Engine", "RATE"} <= js


def test_rust_symbols(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    rs = _names(db, lang="rust")
    assert {"compute", "Engine", "Mode", "Runnable", "start", "LIMIT"} <= rs


def test_go_symbols(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    go = _names(db, lang="go")
    assert {"Compute", "Engine", "Runnable", "Start", "Limit"} <= go


def test_module_level_constants_are_recorded(repo: Path, db: Path):
    """`const` is in the kind vocabulary; Python must actually produce some."""
    index_repo(repo, db, repo="demo")
    assert "CONST_VALUE" in _names(db, kind="const", lang="python")
    assert "LIMIT" in _names(db, kind="const", lang="rust")
    assert "Limit" in _names(db, kind="const", lang="go")


def test_locals_are_not_recorded_as_constants(tmp_path: Path):
    """Otherwise a file's outline is buried under every function's locals."""
    src = tmp_path / "s"
    src.mkdir()
    (src / "m.py").write_text(
        "TOP_LEVEL = 1\n\n\ndef f():\n    inner_local = 2\n    return inner_local\n"
    )
    (src / "m.js").write_text(
        "const TOP = 1;\nfunction f(){ const innerLocal = 2; return innerLocal; }\n"
    )
    db = tmp_path / "s.db"
    index_repo(src, db, repo="demo")

    names = _names(db)
    assert {"TOP_LEVEL", "TOP"} <= names
    assert "inner_local" not in names
    assert "innerLocal" not in names


def test_methods_are_nested_under_their_class(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    rows = _rows(
        db,
        "SELECT c.name FROM code_symbols c "
        "JOIN code_symbols p ON p.id = c.parent_id "
        "WHERE p.name = 'Widget'",
    )
    assert {r[0] for r in rows} == {"render", "resize"}


def test_line_spans_are_recorded(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    row = _rows(
        db, "SELECT line_start, line_end FROM code_symbols WHERE name='top_level'"
    )[0]
    start, end = row
    assert start >= 1
    assert end >= start


def test_imports_are_recorded(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    mods = {r[0] for r in _rows(db, "SELECT module FROM code_imports")}
    assert any("os" == m or m.endswith("os") for m in mods)
    assert any("mod" in m for m in mods)


# --------------------------------------------------------------------------
# Incremental behaviour — the spec's load-bearing claim
# --------------------------------------------------------------------------


def test_unchanged_rerun_touches_zero_rows(repo: Path, db: Path):
    first = index_repo(repo, db, repo="demo")
    assert first.rows_written > 0, "control: the first run must write something"

    second = index_repo(repo, db, repo="demo")
    assert second.rows_written == 0
    assert second.files_indexed == 0
    assert second.files_skipped == first.files_indexed


def test_touching_mtime_without_changing_content_is_not_a_reindex(repo: Path, db: Path):
    """Change detection is by content hash, not mtime.

    Checkouts and formatters routinely bump mtime without changing bytes; an
    mtime-driven index would re-parse the world on every branch switch.
    """
    index_repo(repo, db, repo="demo")
    target = repo / "pkg" / "widget.py"
    target.touch()
    second = index_repo(repo, db, repo="demo")
    assert second.rows_written == 0


def test_changed_file_is_reindexed_and_stale_symbols_are_gone(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    (repo / "pkg" / "widget.py").write_text("def replaced():\n    return 1\n")

    third = index_repo(repo, db, repo="demo")
    assert third.files_indexed == 1
    names = _names(db, lang="python")
    assert "replaced" in names
    assert "top_level" not in names, "the old symbols must not survive a re-index"
    assert "Widget" not in names


def test_deleted_file_is_pruned_with_its_symbols_and_imports(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    (repo / "pkg" / "widget.py").unlink()

    stats = index_repo(repo, db, repo="demo")
    assert stats.files_pruned == 1
    assert _rows(db, "SELECT 1 FROM code_files WHERE path LIKE '%widget.py'") == []
    assert "top_level" not in _names(db)
    orphan_syms = _rows(
        db,
        "SELECT 1 FROM code_symbols WHERE file_id NOT IN (SELECT id FROM code_files)",
    )
    orphan_imps = _rows(
        db,
        "SELECT 1 FROM code_imports WHERE file_id NOT IN (SELECT id FROM code_files)",
    )
    assert orphan_syms == [] and orphan_imps == []


def test_pruning_is_scoped_to_the_repo_being_indexed(repo: Path, db: Path, tmp_path: Path):
    """Indexing repo A must not prune repo B's files."""
    other = tmp_path / "other"
    other.mkdir()
    (other / "b.py").write_text("def only_in_b():\n    return 1\n")

    index_repo(repo, db, repo="demo")
    index_repo(other, db, repo="other")

    index_repo(repo, db, repo="demo")
    assert "only_in_b" in _names(db)


# --------------------------------------------------------------------------
# find_symbols
# --------------------------------------------------------------------------


@pytest.fixture
def ranked(tmp_path: Path) -> tuple[Path, Path]:
    src = tmp_path / "r"
    src.mkdir()
    (src / "m.py").write_text(
        "def render():\n    pass\n\n\n"
        "def render_widget():\n    pass\n\n\n"
        "def prerender():\n    pass\n"
    )
    db = tmp_path / "r.db"
    index_repo(src, db, repo="demo")
    return src, db


def test_find_ranks_exact_then_prefix_then_substring(ranked):
    _, db = ranked
    hits = [h["name"] for h in find_symbols(db, "render")]
    assert hits[:3] == ["render", "render_widget", "prerender"]


def test_find_can_scope_by_kind(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    hits = find_symbols(db, "Widget", kind="class")
    assert hits and all(h["kind"] == "class" for h in hits)


def test_find_can_scope_by_repo(repo: Path, db: Path, tmp_path: Path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "b.py").write_text("def compute():\n    return 1\n")
    index_repo(repo, db, repo="demo")
    index_repo(other, db, repo="other")

    hits = find_symbols(db, "compute", repo="other")
    assert hits and all(h["repo"] == "other" for h in hits)


def test_find_returns_location_for_every_hit(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    for hit in find_symbols(db, "render"):
        assert hit["path"] and hit["line_start"] >= 1


def test_find_respects_limit(ranked):
    _, db = ranked
    assert len(find_symbols(db, "render", limit=2)) == 2


def test_find_with_no_match_returns_empty_not_error(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    assert find_symbols(db, "zzz_no_such_symbol") == []


# --------------------------------------------------------------------------
# file_outline
# --------------------------------------------------------------------------


def test_outline_is_ordered_and_nested(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    outline = file_outline(db, "demo", "pkg/widget.py")

    names = [n["name"] for n in outline]
    starts = [n["line_start"] for n in outline]
    assert starts == sorted(starts), "top-level entries must be in source order"
    assert "Widget" in names
    assert names.index("top_level") < names.index("Widget")

    widget = next(n for n in outline if n["name"] == "Widget")
    assert [c["name"] for c in widget["children"]] == ["render", "resize"]


def test_outline_of_unknown_file_is_empty(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    assert file_outline(db, "demo", "does/not/exist.py") == []


# --------------------------------------------------------------------------
# Robustness — inputs a real repo actually contains
# --------------------------------------------------------------------------


def test_undecodable_bytes_do_not_abort_the_run(tmp_path: Path):
    src = tmp_path / "s"
    src.mkdir()
    (src / "bad.py").write_bytes(b"\xff\xfe\x00def broken(:\n")
    (src / "good.py").write_text("def fine():\n    return 1\n")
    db = tmp_path / "s.db"

    index_repo(src, db, repo="demo")
    assert "fine" in _names(db)


def test_syntactically_broken_source_does_not_abort_the_run(tmp_path: Path):
    src = tmp_path / "s"
    src.mkdir()
    (src / "broken.py").write_text("def (((: unterminated\n")
    (src / "good.py").write_text("def fine():\n    return 1\n")
    db = tmp_path / "s.db"

    index_repo(src, db, repo="demo")
    assert "fine" in _names(db)


def test_vendor_directories_are_skipped(tmp_path: Path):
    src = tmp_path / "s"
    (src / "node_modules" / "dep").mkdir(parents=True)
    (src / ".git").mkdir()
    (src / "node_modules" / "dep" / "v.js").write_text("function vendored(){}")
    (src / "app.js").write_text("function mine(){}")
    db = tmp_path / "s.db"

    index_repo(src, db, repo="demo")
    names = _names(db)
    assert "mine" in names
    assert "vendored" not in names


def test_unknown_extensions_are_ignored(tmp_path: Path):
    src = tmp_path / "s"
    src.mkdir()
    (src / "notes.md").write_text("# not code\n")
    (src / "app.py").write_text("def mine():\n    return 1\n")
    db = tmp_path / "s.db"

    stats = index_repo(src, db, repo="demo")
    assert stats.files_indexed == 1


def test_empty_file_is_handled(tmp_path: Path):
    src = tmp_path / "s"
    src.mkdir()
    (src / "empty.py").write_text("")
    db = tmp_path / "s.db"
    index_repo(src, db, repo="demo")
    assert _rows(db, "SELECT 1 FROM code_files WHERE path='empty.py'")


def test_paths_are_stored_repo_relative_and_posix(repo: Path, db: Path):
    """Stored paths must not leak the indexing machine's absolute layout."""
    index_repo(repo, db, repo="demo")
    paths = [r[0] for r in _rows(db, "SELECT path FROM code_files")]
    assert "pkg/widget.py" in paths
    assert not any(p.startswith("/") or "\\" in p for p in paths)


def test_reindexing_after_a_move_does_not_leave_the_old_path(repo: Path, db: Path):
    index_repo(repo, db, repo="demo")
    (repo / "pkg" / "widget.py").rename(repo / "pkg" / "renamed.py")

    index_repo(repo, db, repo="demo")
    paths = [r[0] for r in _rows(db, "SELECT path FROM code_files")]
    assert "pkg/renamed.py" in paths
    assert "pkg/widget.py" not in paths


# --------------------------------------------------------------------------
# Boundary — the spec's MUST-NOT list, asserted rather than trusted
# --------------------------------------------------------------------------


def test_primitive_imports_nothing_from_premium_or_the_memory_layer():
    """The index must know nothing about sessions, transcripts or memory.

    Asserted over the parsed import graph rather than raw text: a text scan
    would flag the module's own boundary declaration (which correctly contains
    the word "premium") and would miss nothing in exchange. The AST walk also
    covers deferred imports inside functions, which a top-level-only check
    would let through.
    """
    import ast
    import importlib

    mod = importlib.import_module("synapt.recall.code_index")
    tree = ast.parse(Path(mod.__file__).read_text())

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = ("premium", "transcript", "session", "journal", "core", "channel")
    for name in imported:
        low = name.lower()
        assert not any(f in low for f in forbidden), (
            f"code_index must not import {name!r} — the primitive stays free of "
            "the memory layer so the join can live above it"
        )


# --------------------------------------------------------------------------
# Parser stack absent — a defect on the released 0.24.0, tracked privately.
#
# tree-sitter-language-pack is optional (the ``code-index`` extra; a plain
# ``pip install synapt`` does not carry it). Before this fix, a missing stack
# made _get_parser() return None once per file, and index_repo() turned each
# None into its own generic "<path>: could not be parsed" — indistinguishable
# from N real per-file parse failures, with nothing anywhere naming the actual
# cause. This suite had zero coverage of that path (the `except ImportError`
# branch in _get_parser carried a "pragma: no cover" saying so) before now.
# --------------------------------------------------------------------------


@pytest.fixture
def _reset_parser_stack_cache():
    from synapt.recall import code_index

    code_index._PARSER_STACK_AVAILABLE = None
    yield
    code_index._PARSER_STACK_AVAILABLE = None


def test_missing_parser_stack_reports_one_clear_error_not_one_per_file(
    repo: Path, db: Path, monkeypatch, _reset_parser_stack_cache
):
    """The bug, pinned: simulate tree-sitter-language-pack being absent (a
    fresh install without the ``code-index`` extra) via ``sys.modules``, and
    assert ONE clear, named signal for the whole run — not five generic
    "could not be parsed" lines, one per fixture file."""
    import sys

    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)

    stats = index_repo(repo, db, repo="demo")

    assert stats.parser_stack_missing is True
    assert stats.symbols == 0
    assert stats.imports == 0
    assert len(stats.errors) == 1, stats.errors
    assert "tree-sitter-language-pack" in stats.errors[0]
    assert "code-index" in stats.errors[0]


def test_parser_stack_present_leaves_the_flag_false_and_reports_no_errors(
    repo: Path, db: Path
):
    """Control: with the stack genuinely available (this test environment,
    same as ``pip install synapt[code-index]``), the new flag stays False and
    no aggregate error is added — the flag tracks absence, not a blanket
    always-on signal."""
    stats = index_repo(repo, db, repo="demo")

    assert stats.parser_stack_missing is False
    assert stats.errors == []
    assert stats.symbols > 0


def test_missing_parser_stack_does_not_poison_the_index_for_a_later_run(
    repo: Path, db: Path, _reset_parser_stack_cache
):
    """Regression witness for the review finding on this fix: a run with the
    parser stack missing must not write a code_files row per file at all.
    Before this correction, it did — with the file's REAL content_hash and
    empty symbols/imports, counted as indexed — so a LATER run with the stack
    installed saw an unchanged content_hash for every file and silently
    skipped them all as already-current, forever, until the file's own
    content changed. Two runs against the SAME db, same unchanged files on
    disk: stack missing, then stack present, must actually index for real on
    the second run.
    """
    import sys

    from synapt.recall import code_index

    had_key = "tree_sitter_language_pack" in sys.modules
    prior_value = sys.modules.get("tree_sitter_language_pack")
    try:
        # Run 1: stack missing.
        sys.modules["tree_sitter_language_pack"] = None
        code_index._PARSER_STACK_AVAILABLE = None

        stats1 = index_repo(repo, db, repo="demo")
        assert stats1.parser_stack_missing is True
        assert stats1.symbols == 0
        assert stats1.files_indexed == 0, (
            "a file the parser stack can't process must not be counted as "
            "indexed — that count is what a caller trusts as 'it worked'"
        )
        assert stats1.files_skipped == 0, (
            "not-yet-parsed is a different claim than unchanged-since-last-run"
        )

        # Run 2: stack present again (real module, no longer blocked), same
        # db, same unchanged files on disk.
        if had_key:
            sys.modules["tree_sitter_language_pack"] = prior_value
        else:
            del sys.modules["tree_sitter_language_pack"]
        code_index._PARSER_STACK_AVAILABLE = None

        stats2 = index_repo(repo, db, repo="demo")
        assert stats2.parser_stack_missing is False
        assert stats2.errors == []
        assert stats2.symbols > 0, (
            "run 2 must actually index the files run 1 couldn't parse — a "
            "poisoned content_hash from run 1 would read them as unchanged "
            "and skip them silently, forever"
        )
    finally:
        if had_key:
            sys.modules["tree_sitter_language_pack"] = prior_value
        else:
            sys.modules.pop("tree_sitter_language_pack", None)
