"""Symbol index over source files — an OSS primitive (recall#940).

Maps files to the symbols they declare, so a reader can ask *where is this
defined* and *what is in this file* without grepping. tree-sitter parses;
SQLite stores; nothing here knows about memory.

That last point is the design constraint, not an accident of scope. The index
records what the code SAYS. Reconstructing *why* it says it — joining a line
span to the conversations that produced it — is a separate concern that
composes on top of this module and never inside it. So this file imports no
memory layer, and a test asserts that over the parsed import graph rather than
trusting the reviewer to notice.

Two behaviours are worth knowing before reading the code:

**Change detection is by content hash, never mtime.** Branch switches and
formatters bump mtime on files whose bytes are identical; an mtime-driven index
would re-parse the world every time you changed branches.

**An unchanged file is skipped entirely — not re-stamped.** It would be natural
to refresh ``indexed_at`` on every pass, but that turns "nothing changed" into a
write per file and quietly makes the incremental path cost the same as a full
one. ``IndexStats.rows_written`` reports SQLite's own ``total_changes`` delta so
that claim is measurable instead of asserted.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "IndexStats",
    "SUPPORTED_LANGUAGES",
    "file_outline",
    "find_symbols",
    "index_repo",
]

# --------------------------------------------------------------------------
# Language configuration
# --------------------------------------------------------------------------

#: Extension -> tree-sitter language name.
EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
    ".go": "go",
}

SUPPORTED_LANGUAGES = ("python", "typescript", "javascript", "rust", "go")

#: ``tsx`` is a distinct tree-sitter grammar but the same language to a reader,
#: so it is parsed with its own parser and stored under ``typescript``.
LANG_ALIAS = {"tsx": "typescript"}

#: Node types that declare a symbol, mapped to the kind we record.
#:
#: These are the types that CARRY the name. In several grammars that is not the
#: type you would guess: a JS ``const`` is named on ``variable_declarator``
#: rather than ``lexical_declaration``, and Go names live on ``type_spec`` and
#: ``const_spec`` rather than on the enclosing declaration.
SYMBOL_NODES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
        "assignment": "const",
    },
    "javascript": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "variable_declarator": "const",
    },
    "typescript": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "abstract_class_declaration": "class",
        "method_definition": "method",
        "method_signature": "method",
        "variable_declarator": "const",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "interface",
        # No kind in the spec's vocabulary describes an impl block, but it is a
        # real container: recording it as a class is what makes its functions
        # come back as methods of something nameable.
        "impl_item": "class",
        "const_item": "const",
        "static_item": "const",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_spec": "type",
        "const_spec": "const",
    },
}

#: Nodes that introduce a dependency, and the field holding the module.
IMPORT_NODES: dict[str, dict[str, str | None]] = {
    "python": {"import_statement": None, "import_from_statement": "module_name"},
    "javascript": {"import_statement": "source"},
    "typescript": {"import_statement": "source"},
    "rust": {"use_declaration": "argument"},
    "go": {"import_spec": "path"},
}

#: Kinds that can own methods. A function declared inside one of these is
#: recorded as a method rather than a function, which is how "methods under
#: classes" stays true across five grammars without five special cases.
CONTAINER_KINDS = frozenset({"class", "struct", "interface", "enum"})

#: Node types recorded only at module level, never inside another symbol.
#:
#: Python's ``assignment`` and JS/TS's ``variable_declarator`` are the nodes that
#: name a constant — but they also name every local variable in every function
#: body. Recording those would bury a file's real outline under its locals, so
#: these count only when nothing encloses them. Rust and Go declare constants
#: with dedicated node types and need no such guard.
MODULE_LEVEL_ONLY = frozenset({"assignment", "variable_declarator"})

#: Directories never worth indexing. Skipped by name at any depth.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".tox", ".venv", "venv", "env",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "target", "dist", "build", ".next", ".cache",
    "site-packages", ".idea", ".vscode", "vendor",
})

#: Files larger than this are recorded but not parsed. A multi-megabyte
#: generated bundle costs far more to parse than its symbols are worth.
MAX_PARSE_BYTES = 2_000_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS code_files (
  id INTEGER PRIMARY KEY, repo TEXT NOT NULL, path TEXT NOT NULL,
  lang TEXT NOT NULL, content_hash TEXT NOT NULL, mtime REAL,
  indexed_at REAL, UNIQUE(repo, path)
);
CREATE TABLE IF NOT EXISTS code_symbols (
  id INTEGER PRIMARY KEY, file_id INTEGER REFERENCES code_files(id),
  name TEXT NOT NULL, kind TEXT NOT NULL,
  line_start INTEGER, line_end INTEGER,
  parent_id INTEGER REFERENCES code_symbols(id),
  signature TEXT
);
CREATE TABLE IF NOT EXISTS code_imports (
  id INTEGER PRIMARY KEY, file_id INTEGER REFERENCES code_files(id),
  module TEXT NOT NULL, line INTEGER
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON code_symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON code_symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_files_repo ON code_files(repo);
"""


@dataclass
class IndexStats:
    """What one ``index_repo`` pass did.

    ``rows_written`` is SQLite's ``total_changes`` delta for the pass, which is
    what makes the incremental guarantee checkable rather than merely claimed.
    """

    files_indexed: int = 0
    files_skipped: int = 0
    files_pruned: int = 0
    symbols: int = 0
    imports: int = 0
    rows_written: int = 0
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _get_parser(lang: str):
    """Return a parser, or ``None`` when the grammar is unavailable.

    Imported lazily so that importing this module — and the rest of recall —
    does not require the tree-sitter stack to be installed.
    """
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:  # pragma: no cover - exercised by environments without the extra
        return None
    try:
        return get_parser(lang)
    except Exception:  # pragma: no cover - unknown grammar name
        return None


def _node_name(node: Any, lang: str) -> str | None:
    """Best-effort declared name for a node."""
    named = node.child_by_field_name("name")
    if named is not None:
        return named.text.decode("utf-8", "replace")
    # A Rust impl block is named by the type it implements.
    if node.type == "impl_item":
        target = node.child_by_field_name("type")
        if target is not None:
            return target.text.decode("utf-8", "replace")
    # A Python assignment names its target on the left.
    if node.type == "assignment":
        target = node.child_by_field_name("left")
        if target is not None and target.type == "identifier":
            return target.text.decode("utf-8", "replace")
    return None


def _refine_kind(node: Any, lang: str, kind: str, parent_kind: str | None) -> str:
    """Sharpen a node's kind using context the node type alone cannot give."""
    if lang == "go" and node.type == "type_spec":
        for child in node.children:
            if child.type == "struct_type":
                return "struct"
            if child.type == "interface_type":
                return "interface"
        return "type"
    if kind == "function" and parent_kind in CONTAINER_KINDS:
        return "method"
    return kind


def _signature(node: Any, source: bytes) -> str:
    """The declaration's first line, trimmed — enough to identify it in a list."""
    start = node.start_byte
    end = min(node.end_byte, start + 400)
    text = source[start:end].decode("utf-8", "replace")
    first = text.split("\n", 1)[0].strip()
    return first[:300]


def _walk_symbols(
    node: Any,
    lang: str,
    source: bytes,
    parent_id: int | None,
    parent_kind: str | None,
    emit,
) -> None:
    """Record symbols depth-first, threading parent ids as we descend.

    Recording at any depth (rather than only at the top level) is deliberate:
    ``parent_id`` already expresses nesting, so a local helper shows up under
    the function that owns it instead of vanishing.
    """
    table = SYMBOL_NODES.get(lang, {})
    kind = table.get(node.type)

    next_parent_id, next_parent_kind = parent_id, parent_kind

    if kind is not None and node.type in MODULE_LEVEL_ONLY and parent_kind is not None:
        kind = None  # a local, not a declaration worth an outline entry

    if kind is not None:
        name = _node_name(node, lang)
        if name:
            kind = _refine_kind(node, lang, kind, parent_kind)
            new_id = emit(
                name,
                kind,
                node.start_point[0] + 1,
                node.end_point[0] + 1,
                _signature(node, source),
                parent_id,
            )
            next_parent_id, next_parent_kind = new_id, kind

    for child in node.children:
        _walk_symbols(child, lang, source, next_parent_id, next_parent_kind, emit)


def _extract_imports(root: Any, lang: str) -> list[tuple[str, int]]:
    """Collect (module, line) for every import in the tree."""
    wanted = IMPORT_NODES.get(lang, {})
    found: list[tuple[str, int]] = []

    def visit(node: Any) -> None:
        if node.type in wanted:
            field_name = wanted[node.type]
            module: str | None = None
            if field_name:
                target = node.child_by_field_name(field_name)
                if target is not None:
                    module = target.text.decode("utf-8", "replace")
            else:
                # Python's plain ``import a, b`` names its modules in children.
                for child in node.children:
                    if child.type in ("dotted_name", "aliased_import", "identifier"):
                        module = child.text.decode("utf-8", "replace")
                        break
            if module:
                found.append((module.strip("'\"` "), node.start_point[0] + 1))
        for child in node.children:
            visit(child)

    visit(root)
    return found


def _parse_file(path: Path, lang: str) -> tuple[list[tuple], list[tuple[str, int]]] | None:
    """Parse one file into (symbol rows, import rows).

    Returns ``None`` when the file cannot be parsed at all. A single unreadable
    or unparseable file must never abort a repo-wide pass — real trees contain
    generated files, fixtures of deliberately broken source, and binaries that
    happen to carry a source extension.
    """
    parser = _get_parser(lang)
    if parser is None:
        return None
    try:
        source = path.read_bytes()
    except OSError:
        return None
    if len(source) > MAX_PARSE_BYTES:
        return [], []
    try:
        tree = parser.parse(source)
    except Exception:
        return None

    collected: list[tuple] = []

    def emit(name, kind, line_start, line_end, signature, parent_id) -> int:
        # A local id, resolved to real row ids at insert time.
        local_id = len(collected)
        collected.append((local_id, parent_id, name, kind, line_start, line_end, signature))
        return local_id

    try:
        _walk_symbols(tree.root_node, lang, source, None, None, emit)
        imports = _extract_imports(tree.root_node, lang)
    except Exception:
        return None
    return collected, imports


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def _connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def _iter_source_files(root: Path) -> Iterator[Path]:
    """Yield indexable files, skipping vendor trees and symlinked directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS
            and not d.startswith(".")
            and not os.path.islink(os.path.join(dirpath, d))
        ]
        for filename in filenames:
            if Path(filename).suffix.lower() in EXT_TO_LANG:
                candidate = Path(dirpath) / filename
                if not candidate.is_symlink():
                    yield candidate


def _clear_file_children(conn: sqlite3.Connection, file_ids: Iterable[int]) -> None:
    """Remove symbols and imports belonging to the given files.

    Done explicitly because the schema declares ``REFERENCES`` without
    ``ON DELETE CASCADE`` — and SQLite does not enforce foreign keys by default
    anyway, so relying on the declaration would leave orphans behind.
    """
    ids = list(file_ids)
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM code_symbols WHERE file_id IN ({marks})", ids)
    conn.execute(f"DELETE FROM code_imports WHERE file_id IN ({marks})", ids)


def _store_file(
    conn: sqlite3.Connection,
    file_id: int,
    parsed: tuple[list[tuple], list[tuple[str, int]]],
) -> tuple[int, int]:
    symbols, imports = parsed
    local_to_real: dict[int, int] = {}

    # Insert in collection order so a parent always precedes its children.
    for local_id, parent_local, name, kind, line_start, line_end, signature in symbols:
        parent_real = local_to_real.get(parent_local) if parent_local is not None else None
        cur = conn.execute(
            "INSERT INTO code_symbols "
            "(file_id, name, kind, line_start, line_end, parent_id, signature) "
            "VALUES (?,?,?,?,?,?,?)",
            (file_id, name, kind, line_start, line_end, parent_real, signature),
        )
        local_to_real[local_id] = cur.lastrowid

    conn.executemany(
        "INSERT INTO code_imports (file_id, module, line) VALUES (?,?,?)",
        [(file_id, module, line) for module, line in imports],
    )
    return len(symbols), len(imports)


def index_repo(
    root: Path | str,
    db_path: Path | str,
    repo: str,
) -> IndexStats:
    """Index ``root`` into ``db_path`` under the name ``repo``.

    Incremental: files whose content hash is unchanged are skipped without any
    write, changed files are re-parsed with their old rows removed first, and
    files that have disappeared are pruned along with their symbols and
    imports. Pruning is scoped to ``repo`` so indexing one project never
    disturbs another sharing the database.
    """
    root = Path(root).resolve()
    stats = IndexStats()
    conn = _connect(db_path)
    changes_before = conn.total_changes

    try:
        existing: dict[str, tuple[int, str]] = {
            row[1]: (row[0], row[2])
            for row in conn.execute(
                "SELECT id, path, content_hash FROM code_files WHERE repo = ?", (repo,)
            )
        }
        seen: set[str] = set()

        for source_path in _iter_source_files(root):
            rel = source_path.relative_to(root).as_posix()
            seen.add(rel)

            suffix = source_path.suffix.lower()
            grammar = EXT_TO_LANG[suffix]
            lang = LANG_ALIAS.get(grammar, grammar)

            try:
                raw = source_path.read_bytes()
                mtime = source_path.stat().st_mtime
            except OSError as exc:
                stats.errors.append(f"{rel}: {exc}")
                continue

            content_hash = hashlib.sha256(raw).hexdigest()
            prior = existing.get(rel)

            if prior is not None and prior[1] == content_hash:
                # Unchanged. Deliberately no write at all, not even indexed_at.
                stats.files_skipped += 1
                continue

            parsed = _parse_file(source_path, grammar)
            if parsed is None:
                stats.errors.append(f"{rel}: could not be parsed")
                parsed = ([], [])

            now = time.time()
            if prior is None:
                cur = conn.execute(
                    "INSERT INTO code_files "
                    "(repo, path, lang, content_hash, mtime, indexed_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (repo, rel, lang, content_hash, mtime, now),
                )
                file_id = cur.lastrowid
            else:
                file_id = prior[0]
                _clear_file_children(conn, [file_id])
                conn.execute(
                    "UPDATE code_files SET lang=?, content_hash=?, mtime=?, indexed_at=? "
                    "WHERE id=?",
                    (lang, content_hash, mtime, now, file_id),
                )

            n_symbols, n_imports = _store_file(conn, file_id, parsed)
            stats.files_indexed += 1
            stats.symbols += n_symbols
            stats.imports += n_imports

        gone = [(path, fid) for path, (fid, _) in existing.items() if path not in seen]
        if gone:
            ids = [fid for _, fid in gone]
            _clear_file_children(conn, ids)
            marks = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM code_files WHERE id IN ({marks})", ids)
            stats.files_pruned = len(gone)

        conn.commit()
        stats.rows_written = conn.total_changes - changes_before
    finally:
        conn.close()

    return stats


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------


def find_symbols(
    db_path: Path | str,
    name: str,
    repo: str | None = None,
    kind: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Look up symbols by name, best match first.

    Ranking is exact, then prefix, then substring. No embeddings — a name
    lookup is a name lookup, and the retrieval-upgrade question is a separate
    decision that should not be pre-empted here.
    """
    if not name:
        return []

    escaped = name.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    sql = """
        SELECT f.repo, f.path, f.lang, s.name, s.kind,
               s.line_start, s.line_end, s.signature,
               CASE
                 WHEN s.name = ?      THEN 0
                 WHEN s.name LIKE ?   THEN 1
                 ELSE 2
               END AS rank
        FROM code_symbols s
        JOIN code_files f ON f.id = s.file_id
        WHERE s.name LIKE ? ESCAPE '!'
    """
    args: list = [name, f"{escaped}%", f"%{escaped}%"]

    if repo:
        sql += " AND f.repo = ?"
        args.append(repo)
    if kind:
        sql += " AND s.kind = ?"
        args.append(kind)

    sql += " ORDER BY rank, LENGTH(s.name), f.path, s.line_start LIMIT ?"
    args.append(limit)

    conn = _connect(db_path)
    try:
        rows = conn.execute(sql, args).fetchall()
    finally:
        conn.close()

    return [
        {
            "repo": r[0], "path": r[1], "lang": r[2], "name": r[3], "kind": r[4],
            "line_start": r[5], "line_end": r[6], "signature": r[7],
        }
        for r in rows
    ]


def file_outline(db_path: Path | str, repo: str, path: str) -> list[dict]:
    """Symbols of one file in source order, nested by parent.

    An unknown file yields an empty list rather than an error: "this file has
    no outline" and "this file is not indexed" are the same answer to a caller
    rendering a pane, and inventing a distinction would only add a failure mode.
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT s.id, s.parent_id, s.name, s.kind,
                   s.line_start, s.line_end, s.signature
            FROM code_symbols s
            JOIN code_files f ON f.id = s.file_id
            WHERE f.repo = ? AND f.path = ?
            ORDER BY s.line_start, s.id
            """,
            (repo, path),
        ).fetchall()
    finally:
        conn.close()

    nodes: dict[int, dict] = {}
    for sid, parent_id, sym_name, kind, line_start, line_end, signature in rows:
        nodes[sid] = {
            "name": sym_name, "kind": kind,
            "line_start": line_start, "line_end": line_end,
            "signature": signature, "children": [],
        }

    roots: list[dict] = []
    for sid, parent_id, *_ in rows:
        if parent_id is not None and parent_id in nodes:
            nodes[parent_id]["children"].append(nodes[sid])
        else:
            roots.append(nodes[sid])
    return roots
