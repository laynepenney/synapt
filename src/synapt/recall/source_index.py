"""Generic local source-unit indexing for authorized downstream providers.

This module deliberately does not resolve agent identity or choose private
source roots.  A downstream provider authorizes a :class:`SourceAdmission`,
opens the protected per-scope SQLite store, and supplies both operations as
callables.  Authorization therefore runs before the store opener or adapter.

The first implementation is lexical and Markdown-only.  It establishes the
distinct source-unit lifecycle and the composition seam without projecting
files into transcript, journal, or knowledge storage.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import stat
import unicodedata
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Protocol


_PARSER_VERSION = "markdown-v1"
_SUCCESS = "complete"
_FAILURE_STATES = {
    "cancelled",
    "permission_lost",
    "root_missing",
    "iterator_failed",
    "file_limit_exceeded",
    "scan_limit_exceeded",
    "parser_limit_exceeded",
    "unsupported",
    "unauthorized",
    "generation_changed",
}


@dataclass(frozen=True)
class SourceAdmission:
    """One immutable, downstream-authorized source admission."""

    source_id: str
    scope_capability: bytes
    root_handle: int
    root_handle_id: str
    admission_epoch: int
    policy_epoch: int
    allowed_extensions: tuple[str, ...] = (".md",)
    exclusion_policy_hash: str = ""
    parser_profile: str = _PARSER_VERSION
    path_disclosure: str = "hidden"

    def __post_init__(self) -> None:
        if not self.source_id or not self.scope_capability or not self.root_handle_id:
            raise ValueError("source admission requires opaque source, capability, and handle IDs")
        if self.admission_epoch < 0 or self.policy_epoch < 0:
            raise ValueError("source admission epochs must be non-negative")
        if self.path_disclosure not in {"hidden", "relative"}:
            raise ValueError("path_disclosure must be hidden or relative")


@dataclass(frozen=True)
class SourceLimits:
    file_bytes: int = 4 * 1024 * 1024
    scan_bytes: int = 16 * 1024 * 1024
    parser_units: int = 1_000

    def __post_init__(self) -> None:
        if min(self.file_bytes, self.scan_bytes, self.parser_units) < 1:
            raise ValueError("source limits must be positive")


@dataclass(frozen=True)
class SourceDocument:
    relative_path: str
    content: bytes
    sha256: str
    observed_at: str


@dataclass(frozen=True)
class ParsedSourceUnit:
    structural_address: str
    line_start: int
    line_end: int
    content: str


@dataclass(frozen=True)
class SourceSearchResult:
    content: str
    structural_address: str
    lifecycle: str
    revision_token: str
    observed_at: str
    relative_path: str | None = None
    source_kind: str = "memory_file"


@dataclass(frozen=True)
class SourceScanReceipt:
    scan_id: str
    state: str
    generation: int | None = None
    documents_seen: int | None = None
    units_published: int | None = None
    documents_reused: int | None = None


class SourceAdapter(Protocol):
    """Read already-admitted source bytes without performing identity lookup."""

    def enumerate(
        self, admission: SourceAdmission, limits: SourceLimits
    ) -> Iterable[SourceDocument]: ...


AuthorizationCheck = Callable[[SourceAdmission], bool]
StoreOpener = Callable[[], sqlite3.Connection]
MarkdownParser = Callable[[bytes], list[ParsedSourceUnit]]


class _ScanFailure(Exception):
    def __init__(self, state: str):
        if state not in _FAILURE_STATES:
            raise ValueError(f"unknown source scan state: {state}")
        self.state = state
        super().__init__(state)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _authorized(admission: SourceAdmission, check: AuthorizationCheck) -> bool:
    try:
        return bool(check(admission))
    except Exception:
        return False


def _normalize_component(component: str) -> str:
    if (
        not component
        or component in {".", ".."}
        or "\x00" in component
        or "/" in component
        or "\\" in component
    ):
        raise _ScanFailure("iterator_failed")
    return unicodedata.normalize("NFC", component)


def _eligible_leaf(name: str, admission: SourceAdmission) -> bool:
    lowered = name.casefold()
    if name.startswith("."):
        return False
    if lowered.endswith(("~", ".bak", ".backup", ".tmp", ".swp")):
        return False
    return any(lowered.endswith(ext.casefold()) for ext in admission.allowed_extensions)


class DescriptorSourceAdapter:
    """Unix descriptor-relative, no-follow local source adapter.

    Windows requires a native root-handle-relative implementation.  This
    adapter refuses there instead of weakening containment to path post-checks.
    """

    _READ_SIZE = 64 * 1024

    def enumerate(
        self, admission: SourceAdmission, limits: SourceLimits
    ) -> Iterable[SourceDocument]:
        if os.name == "nt" or not hasattr(os, "O_NOFOLLOW"):
            raise _ScanFailure("unsupported")
        try:
            root_before = os.fstat(admission.root_handle)
        except OSError as exc:
            raise _ScanFailure("root_missing") from exc
        if not stat.S_ISDIR(root_before.st_mode):
            raise _ScanFailure("root_missing")

        try:
            root_fd = os.dup(admission.root_handle)
        except OSError as exc:
            raise _ScanFailure("permission_lost") from exc

        documents: list[SourceDocument] = []
        collision_keys: set[str] = set()
        consumed = 0
        try:
            consumed = self._walk(
                root_fd,
                (),
                admission,
                limits,
                documents,
                collision_keys,
                consumed,
            )
            del consumed
            root_after = os.fstat(admission.root_handle)
            if (root_before.st_dev, root_before.st_ino) != (
                root_after.st_dev,
                root_after.st_ino,
            ):
                raise _ScanFailure("root_missing")
        except _ScanFailure:
            raise
        except PermissionError as exc:
            raise _ScanFailure("permission_lost") from exc
        except FileNotFoundError as exc:
            raise _ScanFailure("root_missing") from exc
        except OSError as exc:
            raise _ScanFailure("iterator_failed") from exc
        finally:
            os.close(root_fd)
        return documents

    def _walk(
        self,
        directory_fd: int,
        parents: tuple[str, ...],
        admission: SourceAdmission,
        limits: SourceLimits,
        documents: list[SourceDocument],
        collision_keys: set[str],
        consumed: int,
    ) -> int:
        for raw_name in sorted(os.listdir(directory_fd)):
            name = _normalize_component(raw_name)
            if name.startswith("."):
                continue
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode):
                continue
            if stat.S_ISDIR(before.st_mode):
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                child_fd = os.open(name, flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(child_fd)
                    if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                        raise _ScanFailure("iterator_failed")
                    consumed = self._walk(
                        child_fd,
                        parents + (name,),
                        admission,
                        limits,
                        documents,
                        collision_keys,
                        consumed,
                    )
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(before.st_mode) or not _eligible_leaf(name, admission):
                continue
            if before.st_nlink != 1:
                continue
            relative_path = "/".join(parents + (name,))
            collision_key = unicodedata.normalize("NFC", relative_path).casefold()
            if collision_key in collision_keys:
                raise _ScanFailure("iterator_failed")
            collision_keys.add(collision_key)
            content, read_count, observed_at = self._read_leaf(
                directory_fd, name, before, limits.file_bytes
            )
            consumed += read_count
            if consumed > limits.scan_bytes:
                raise _ScanFailure("scan_limit_exceeded")
            documents.append(
                SourceDocument(
                    relative_path=relative_path,
                    content=content,
                    sha256=hashlib.sha256(content).hexdigest(),
                    observed_at=observed_at,
                )
            )
        return consumed

    def _read_leaf(
        self, directory_fd: int, name: str, enumerated: os.stat_result, byte_cap: int
    ) -> tuple[bytes, int, str]:
        if enumerated.st_size > byte_cap:
            raise _ScanFailure("file_limit_exceeded")
        flags = os.O_RDONLY | os.O_NOFOLLOW
        fd = os.open(name, flags, dir_fd=directory_fd)
        try:
            opened = os.fstat(fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (enumerated.st_dev, enumerated.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise _ScanFailure("iterator_failed")
            chunks: list[bytes] = []
            consumed = 0
            while True:
                chunk = os.read(fd, min(self._READ_SIZE, byte_cap + 1 - consumed))
                if not chunk:
                    break
                chunks.append(chunk)
                consumed += len(chunk)
                if consumed > byte_cap:
                    raise _ScanFailure("file_limit_exceeded")
            after = os.fstat(fd)
            identity_before = (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                getattr(opened, "st_mtime_ns", int(opened.st_mtime * 1e9)),
                opened.st_nlink,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                getattr(after, "st_mtime_ns", int(after.st_mtime * 1e9)),
                after.st_nlink,
            )
            if identity_before != identity_after:
                raise _ScanFailure("iterator_failed")
            return b"".join(chunks), consumed, _now()
        finally:
            os.close(fd)


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse_markdown(content: bytes) -> list[ParsedSourceUnit]:
    """Split UTF-8 Markdown into deterministic heading-addressed units."""

    text = content.decode("utf-8")
    lines = text.splitlines()
    if not lines:
        return []
    starts: list[tuple[int, list[str], str]] = []
    heading_stack: list[str] = []
    preamble_start = 1
    for line_number, line in enumerate(lines, 1):
        match = _HEADING.match(line)
        if match is None:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(title)
        starts.append((line_number, list(heading_stack), line))
    units: list[ParsedSourceUnit] = []
    ordinal_by_address: dict[str, int] = {}
    if not starts:
        return [ParsedSourceUnit("document [1]", 1, len(lines), text)]
    if starts[0][0] > preamble_start:
        body = "\n".join(lines[: starts[0][0] - 1]).strip()
        if body:
            units.append(ParsedSourceUnit("preamble [1]", 1, starts[0][0] - 1, body))
    for index, (line_start, headings, _heading_line) in enumerate(starts):
        line_end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        body = "\n".join(lines[line_start - 1 : line_end]).strip()
        if not body:
            continue
        address = " > ".join(headings)
        ordinal_by_address[address] = ordinal_by_address.get(address, 0) + 1
        units.append(
            ParsedSourceUnit(
                f"{address} [{ordinal_by_address[address]}]",
                line_start,
                line_end,
                body,
            )
        )
    return units


_SOURCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_state (
    source_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL,
    root_handle_id TEXT NOT NULL,
    admission_epoch INTEGER NOT NULL,
    policy_epoch INTEGER NOT NULL,
    parser_profile TEXT NOT NULL,
    exclusion_policy_hash TEXT NOT NULL,
    revision_key BLOB NOT NULL,
    completed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_documents (
    document_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    collision_key TEXT NOT NULL,
    document_sha256 TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    stale_reason TEXT,
    revision_token TEXT NOT NULL,
    generation INTEGER NOT NULL,
    UNIQUE(source_id, collision_key)
);
CREATE TABLE IF NOT EXISTS source_units (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    structural_address TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    content TEXT NOT NULL,
    unit_sha256 TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    revision_token TEXT NOT NULL,
    generation INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS source_units_source ON source_units(source_id, lifecycle);
CREATE TABLE IF NOT EXISTS source_scan_receipts (
    scan_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    completed_at TEXT NOT NULL,
    documents_seen INTEGER NOT NULL,
    units_published INTEGER NOT NULL,
    documents_reused INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS source_units_fts USING fts5(
    unit_id UNINDEXED,
    content,
    structural_address,
    tokenize="porter unicode61 tokenchars '._+'"
);
"""


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(_SOURCE_SCHEMA)
    connection.commit()


def _snapshot_state(
    connection: sqlite3.Connection, admission: SourceAdmission
) -> tuple[int, bytes, dict[str, sqlite3.Row]]:
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT generation, revision_key FROM source_state WHERE source_id = ?",
        (admission.source_id,),
    ).fetchone()
    generation = int(row["generation"]) if row else 0
    revision_key = bytes(row["revision_key"]) if row else secrets.token_bytes(32)
    documents = {
        item["relative_path"]: item
        for item in connection.execute(
            "SELECT * FROM source_documents WHERE source_id = ? AND lifecycle = 'current'",
            (admission.source_id,),
        ).fetchall()
    }
    return generation, revision_key, documents


def _document_id(source_id: str, relative_path: str) -> str:
    return hashlib.sha256(f"{source_id}\x00{relative_path}".encode()).hexdigest()


def _revision_token(revision_key: bytes, document_sha256: str) -> str:
    return hmac.new(revision_key, document_sha256.encode(), hashlib.sha256).hexdigest()[:20]


def _unit_id(document_id: str, unit: ParsedSourceUnit, unit_sha256: str) -> str:
    authority = f"{document_id}\x00{unit.structural_address}\x00{unit_sha256}"
    return hashlib.sha256(authority.encode()).hexdigest()


def sync_source(
    admission: SourceAdmission,
    adapter: SourceAdapter,
    open_store: StoreOpener,
    authorize: AuthorizationCheck,
    *,
    limits: SourceLimits | None = None,
    parser: MarkdownParser = parse_markdown,
) -> SourceScanReceipt:
    """Scan and atomically publish one authorized source generation."""

    scan_id = f"scan_{uuid.uuid4().hex[:12]}"
    limits = limits or SourceLimits()
    if not _authorized(admission, authorize):
        return SourceScanReceipt(scan_id, "unauthorized")
    connection = open_store()
    try:
        _ensure_schema(connection)
        generation, revision_key, current_documents = _snapshot_state(connection, admission)
    finally:
        connection.close()

    try:
        observed_documents = list(adapter.enumerate(admission, limits))
    except _ScanFailure as exc:
        return SourceScanReceipt(scan_id, exc.state)
    except Exception:
        return SourceScanReceipt(scan_id, "iterator_failed")

    staged: dict[str, tuple[SourceDocument, list[ParsedSourceUnit]]] = {}
    reused = 0
    total_units = 0
    for document in observed_documents:
        existing = current_documents.get(document.relative_path)
        if (
            existing is not None
            and existing["document_sha256"] == document.sha256
            and existing["parser_version"] == admission.parser_profile
        ):
            reused += 1
            continue
        try:
            units = parser(document.content)
        except (UnicodeDecodeError, ValueError):
            return SourceScanReceipt(scan_id, "iterator_failed")
        total_units += len(units)
        if total_units > limits.parser_units:
            return SourceScanReceipt(scan_id, "parser_limit_exceeded")
        staged[document.relative_path] = (document, units)

    if not _authorized(admission, authorize):
        return SourceScanReceipt(scan_id, "unauthorized")

    connection = open_store()
    connection.row_factory = sqlite3.Row
    try:
        _ensure_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        live = connection.execute(
            "SELECT generation FROM source_state WHERE source_id = ?",
            (admission.source_id,),
        ).fetchone()
        live_generation = int(live["generation"]) if live else 0
        if live_generation != generation:
            connection.rollback()
            return SourceScanReceipt(scan_id, "generation_changed")
        next_generation = generation + 1
        indexed_at = _now()
        observed_paths = {document.relative_path for document in observed_documents}
        deleted_paths = set(current_documents) - observed_paths
        changed_paths = set(staged)
        for relative_path in deleted_paths | changed_paths:
            old = current_documents.get(relative_path)
            if old is None:
                continue
            unit_ids = [
                row["unit_id"]
                for row in connection.execute(
                    "SELECT unit_id FROM source_units WHERE document_id = ? AND lifecycle = 'current'",
                    (old["document_id"],),
                ).fetchall()
            ]
            for unit_id in unit_ids:
                connection.execute("DELETE FROM source_units_fts WHERE unit_id = ?", (unit_id,))
            connection.execute(
                "UPDATE source_units SET lifecycle = 'deleted', generation = ? "
                "WHERE document_id = ? AND lifecycle = 'current'",
                (next_generation, old["document_id"]),
            )
            reason = "source_deleted" if relative_path in deleted_paths else "replaced"
            connection.execute(
                "UPDATE source_documents SET lifecycle = 'deleted', stale_reason = ?, "
                "generation = ? WHERE document_id = ?",
                (reason, next_generation, old["document_id"]),
            )

        units_published = 0
        for relative_path, (document, units) in staged.items():
            document_id = _document_id(admission.source_id, relative_path)
            revision = _revision_token(revision_key, document.sha256)
            connection.execute(
                "INSERT OR REPLACE INTO source_documents "
                "(document_id, source_id, relative_path, collision_key, document_sha256, "
                "parser_version, observed_at, indexed_at, lifecycle, stale_reason, "
                "revision_token, generation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'current', NULL, ?, ?)",
                (
                    document_id,
                    admission.source_id,
                    relative_path,
                    unicodedata.normalize("NFC", relative_path).casefold(),
                    document.sha256,
                    admission.parser_profile,
                    document.observed_at,
                    indexed_at,
                    revision,
                    next_generation,
                ),
            )
            for unit in units:
                unit_sha256 = hashlib.sha256(unit.content.encode()).hexdigest()
                unit_id = _unit_id(document_id, unit, unit_sha256)
                connection.execute(
                    "INSERT INTO source_units "
                    "(unit_id, document_id, source_id, structural_address, line_start, line_end, "
                    "content, unit_sha256, observed_at, indexed_at, lifecycle, revision_token, generation) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'current', ?, ?)",
                    (
                        unit_id,
                        document_id,
                        admission.source_id,
                        unit.structural_address,
                        unit.line_start,
                        unit.line_end,
                        unit.content,
                        unit_sha256,
                        document.observed_at,
                        indexed_at,
                        revision,
                        next_generation,
                    ),
                )
                connection.execute(
                    "INSERT INTO source_units_fts(unit_id, content, structural_address) VALUES (?, ?, ?)",
                    (unit_id, unit.content, unit.structural_address),
                )
                units_published += 1
        connection.execute(
            "INSERT INTO source_state "
            "(source_id, generation, root_handle_id, admission_epoch, policy_epoch, "
            "parser_profile, exclusion_policy_hash, revision_key, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_id) DO UPDATE SET generation=excluded.generation, "
            "root_handle_id=excluded.root_handle_id, admission_epoch=excluded.admission_epoch, "
            "policy_epoch=excluded.policy_epoch, parser_profile=excluded.parser_profile, "
            "exclusion_policy_hash=excluded.exclusion_policy_hash, completed_at=excluded.completed_at",
            (
                admission.source_id,
                next_generation,
                admission.root_handle_id,
                admission.admission_epoch,
                admission.policy_epoch,
                admission.parser_profile,
                admission.exclusion_policy_hash,
                revision_key,
                indexed_at,
            ),
        )
        connection.execute(
            "INSERT INTO source_scan_receipts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                scan_id,
                admission.source_id,
                next_generation,
                indexed_at,
                len(observed_documents),
                units_published,
                reused,
            ),
        )
        connection.commit()
        return SourceScanReceipt(
            scan_id,
            _SUCCESS,
            generation=next_generation,
            documents_seen=len(observed_documents),
            units_published=units_published,
            documents_reused=reused,
        )
    except Exception:
        connection.rollback()
        return SourceScanReceipt(scan_id, "iterator_failed")
    finally:
        connection.close()


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[\w.+-]+", query, flags=re.UNICODE)
    return " ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def search_source(
    admission: SourceAdmission,
    open_store: StoreOpener,
    authorize: AuthorizationCheck,
    query: str,
    *,
    limit: int = 5,
) -> list[SourceSearchResult]:
    """Return bounded current lexical results after two authority checks."""

    if limit < 1 or not _authorized(admission, authorize):
        return []
    expression = _fts_query(query)
    if not expression:
        return []
    connection = open_store()
    connection.row_factory = sqlite3.Row
    try:
        _ensure_schema(connection)
        rows = connection.execute(
            "SELECT u.content, u.structural_address, u.lifecycle, u.revision_token, "
            "u.observed_at, d.relative_path "
            "FROM source_units_fts f "
            "JOIN source_units u ON u.unit_id = f.unit_id "
            "JOIN source_documents d ON d.document_id = u.document_id "
            "WHERE source_units_fts MATCH ? AND u.source_id = ? "
            "AND u.lifecycle = 'current' AND d.lifecycle = 'current' "
            "ORDER BY bm25(source_units_fts), u.unit_id LIMIT ?",
            (expression, admission.source_id, limit),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    if not _authorized(admission, authorize):
        return []
    disclose_path = admission.path_disclosure == "relative"
    return [
        SourceSearchResult(
            content=row["content"],
            structural_address=row["structural_address"],
            lifecycle=row["lifecycle"],
            revision_token=row["revision_token"],
            observed_at=row["observed_at"],
            relative_path=row["relative_path"] if disclose_path else None,
        )
        for row in rows
    ]


def render_source_results(results: Iterable[SourceSearchResult]) -> str:
    """Render authorized source results without internal IDs or raw hashes."""

    blocks: list[str] = []
    for result in results:
        address = result.structural_address
        if result.relative_path:
            address = f"{result.relative_path} · {address}"
        blocks.append(
            f"[source:{result.source_kind} · {result.lifecycle} · {address} · "
            f"revision {result.revision_token} · observed {result.observed_at}]\n"
            f"{result.content}"
        )
    return "\n\n".join(blocks)


def compose_source_results(base_result: str, source_results: Iterable[SourceSearchResult]) -> str:
    """Deterministically append source-unit results to an existing recall render."""

    source_render = render_source_results(source_results)
    return "\n\n".join(part for part in (base_result, source_render) if part)
