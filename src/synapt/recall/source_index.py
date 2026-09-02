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
import struct
import threading
import unicodedata
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from synapt.recall.embeddings import EmbeddingProvider
from synapt.recall.hybrid import weighted_rrf_merge


SOURCE_INDEX_SUPPORTED = hasattr(os, "O_DIRECTORY")
"""Whether this platform can run DescriptorSourceAdapter's fd-based
admission. False on Windows, where os.O_DIRECTORY does not exist -- the
single source of truth production code and tests both check, so a Windows
run degrades to a clear refusal instead of an AttributeError deep in the
walk."""

_PARSER_VERSION = "markdown-v1"

# Minimum cosine similarity for an embedding-only candidate (one with no
# lexical/BM25 overlap) to be admitted into hybrid search results. Measured
# 2026-09-02 against a real ~950-unit markdown corpus (all-MiniLM-L6-v2): ten
# grep-verified-absent queries (nonsense tokens + real terms from a different
# repo) scored a max cosine of 0.3876 against their nearest unit; floor is
# that maximum plus a 0.03 margin. This is corpus- and model-dependent, not a
# universal constant -- callers with a materially different corpus size or
# embedding model should re-measure rather than trust this default. Full
# measurement tracked privately, not linked here.
DEFAULT_SOURCE_SIMILARITY_FLOOR = 0.4176

# How many candidates each side (BM25, embedding) contributes to the merge
# pool before truncating to the caller's requested limit. Wider than `limit`
# so RRF has real ranking material to fuse from both sides.
_HYBRID_CANDIDATE_POOL = 20
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
            raise ValueError(
                "source admission requires opaque source, capability, and handle IDs"
            )
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
    similarity: float | None = None
    """Cosine similarity against the query, when the embedding path scored
    this unit for this query (regardless of whether it passed the floor or
    was surfaced via BM25 instead). ``None`` when hybrid search was not used,
    or when this unit has no stored embedding (e.g. synced before the flag
    was enabled). A caller sees the real number next to a BM25-only ``None``
    and judges, rather than trusting an internal admit/reject decision alone."""


@dataclass(frozen=True)
class SourceSearchRequest:
    """Neutral query contract passed to already-authorized source providers."""

    query: str
    limit: int
    after: str | None = None
    before: str | None = None
    include_historical: bool = False

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("source search limit must be positive")


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


class SourceSearchProvider(Protocol):
    """Return authorized, usefulness-gated results for one neutral request.

    The provider owns authorization before any source, store, cache, metrics,
    or candidate work.  Returning an empty iterable is a valid refusal or
    genuine-absence result.
    """

    def search(self, request: SourceSearchRequest) -> Iterable[SourceSearchResult]: ...


AuthorizationCheck = Callable[[SourceAdmission], bool]
StoreOpener = Callable[[], sqlite3.Connection]
MarkdownParser = Callable[[bytes], list[ParsedSourceUnit]]


_source_search_providers: dict[str, SourceSearchProvider] = {}
_source_search_provider_lock = threading.RLock()


def register_source_search_provider(name: str, provider: SourceSearchProvider) -> None:
    """Register one downstream source provider for normal recall composition."""

    normalized = name.strip()
    if not normalized or not callable(getattr(provider, "search", None)):
        raise ValueError("source provider requires a name and search callable")
    with _source_search_provider_lock:
        existing = _source_search_providers.get(normalized)
        if existing is not None and existing is not provider:
            raise ValueError(f"source provider already registered: {normalized}")
        _source_search_providers[normalized] = provider


def _clear_source_search_providers() -> None:
    """Reset the process-local provider registry for isolated tests."""

    with _source_search_provider_lock:
        _source_search_providers.clear()


def search_registered_sources(request: SourceSearchRequest) -> list[SourceSearchResult]:
    """Query a stable provider snapshot and fail closed on provider refusal."""

    with _source_search_provider_lock:
        providers = tuple(sorted(_source_search_providers.items()))
    results: list[SourceSearchResult] = []
    for _name, provider in providers:
        try:
            candidates = provider.search(request)
            for candidate in candidates:
                if not isinstance(candidate, SourceSearchResult):
                    continue
                allowed_lifecycles = (
                    {"current", "stale", "deleted"}
                    if request.include_historical
                    else {"current"}
                )
                if candidate.lifecycle not in allowed_lifecycles:
                    continue
                if not _source_result_in_window(candidate, request):
                    continue
                results.append(candidate)
                if len(results) >= request.limit:
                    return results
        except Exception:
            # Provider errors may contain private source details.  Refuse this
            # provider without surfacing or logging the exception text.
            continue
    return results


def _source_result_in_window(
    result: SourceSearchResult, request: SourceSearchRequest
) -> bool:
    """Apply the caller's date window even when a provider omits that filter."""

    if request.after is None and request.before is None:
        return True
    try:
        observed = datetime.fromisoformat(result.observed_at.replace("Z", "+00:00"))
        if observed.tzinfo is None:
            return False
        if request.after is not None:
            after = datetime.fromisoformat(request.after.replace("Z", "+00:00"))
            if after.tzinfo is None or observed < after:
                return False
        if request.before is not None:
            before = datetime.fromisoformat(request.before.replace("Z", "+00:00"))
            if before.tzinfo is None or observed >= before:
                return False
    except ValueError:
        return False
    return True


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
        if not SOURCE_INDEX_SUPPORTED or not hasattr(os, "O_NOFOLLOW"):
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
CREATE TABLE IF NOT EXISTS source_units_embeddings (
    unit_id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model TEXT NOT NULL
);
"""
# No source_id column on source_units_embeddings: every embedding query must
# join through source_units (and source_documents for lifecycle) to reach a
# row at all, which makes it structurally impossible to query embeddings
# without the same source_id/lifecycle scoping the FTS path already applies --
# the same authorization discipline #918's cross-org isolation witness
# (test_source_index_cross_org_isolation.py) already proved for search_source.


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
    return hmac.new(revision_key, document_sha256.encode(), hashlib.sha256).hexdigest()[
        :20
    ]


def _unit_id(document_id: str, unit: ParsedSourceUnit, unit_sha256: str) -> str:
    authority = f"{document_id}\x00{unit.structural_address}\x00{unit_sha256}"
    return hashlib.sha256(authority.encode()).hexdigest()


def _pack_vector(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_vector(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def sync_source(
    admission: SourceAdmission,
    adapter: SourceAdapter,
    open_store: StoreOpener,
    authorize: AuthorizationCheck,
    *,
    limits: SourceLimits | None = None,
    parser: MarkdownParser = parse_markdown,
    embed_provider: EmbeddingProvider | None = None,
) -> SourceScanReceipt:
    """Scan and atomically publish one authorized source generation.

    ``embed_provider`` is opt-in and defaults to ``None`` -- omitting it (the
    default for every existing caller) produces byte-for-byte the same
    behavior as before this parameter existed. When supplied, only the units
    actually being (re)published this generation are embedded (the same
    content-hash-scoped set that already skips unchanged documents), never
    the whole corpus -- the embedding step rides the existing incrementality
    rather than adding a second full-corpus pass.
    """

    scan_id = f"scan_{uuid.uuid4().hex[:12]}"
    limits = limits or SourceLimits()
    if not _authorized(admission, authorize):
        return SourceScanReceipt(scan_id, "unauthorized")
    connection = open_store()
    try:
        _ensure_schema(connection)
        generation, revision_key, current_documents = _snapshot_state(
            connection, admission
        )
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

    # Embedding runs outside any DB transaction/lock, same placement as
    # parsing above -- a provider call (network or model inference) must
    # never happen while holding the write lock. Only the units actually
    # staged this generation are embedded (already skips reused/unchanged
    # documents), so this rides the existing incrementality rather than
    # adding a second full-corpus pass.
    embeddings_by_unit_id: dict[str, list[float]] = {}
    if embed_provider is not None and staged:
        pending_ids: list[str] = []
        pending_texts: list[str] = []
        for relative_path, (document, units) in staged.items():
            document_id = _document_id(admission.source_id, relative_path)
            for unit in units:
                unit_sha256 = hashlib.sha256(unit.content.encode()).hexdigest()
                pending_ids.append(_unit_id(document_id, unit, unit_sha256))
                pending_texts.append(unit.content)
        if pending_texts:
            embeddings_by_unit_id = dict(
                zip(pending_ids, embed_provider.embed(pending_texts))
            )

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
                connection.execute(
                    "DELETE FROM source_units_fts WHERE unit_id = ?", (unit_id,)
                )
                # Always clean up, independent of whether embed_provider was
                # passed THIS call -- a unit deleted/replaced while the flag
                # happened to be off must not leave an orphaned embedding row
                # keyed on a unit_id that source_units no longer considers
                # current.
                connection.execute(
                    "DELETE FROM source_units_embeddings WHERE unit_id = ?", (unit_id,)
                )
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
                vector = embeddings_by_unit_id.get(unit_id)
                if vector is not None:
                    connection.execute(
                        "INSERT OR REPLACE INTO source_units_embeddings "
                        "(unit_id, vector, dim, model) VALUES (?, ?, ?, ?)",
                        (unit_id, _pack_vector(vector), len(vector), type(embed_provider).__name__),
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


_FTS_STOPWORDS = frozenset(
    """
    a about after again all am an and any are as at be because been before
    being below between both but by can could did do does doing don down
    during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just let me
    more most my myself no nor not now of off on once only or other our
    ours ourselves out over own same she should so some someone something
    such than that the their theirs them themselves then there these they
    this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours
    yourself yourselves ve re ll s t d m
    """.split()
)


def _fts_query(query: str) -> str:
    """Space-separated quoted tokens, which SQLite FTS5 MATCH treats as an
    AND of every token. English filler words are dropped first so a verbose
    natural-language question doesn't fail purely because "how", "does", or
    "the" happens not to co-occur with a document's real content words --
    the actual defect this stopword filter closes (a token-count/phrasing
    problem, independent of any semantic/synonym gap; tracked privately,
    measured on a real corpus before this change).

    Filtering never turns a real query into an unconstrained one: if every
    token happens to be a stopword (or the query is otherwise short), the
    unfiltered tokens are used instead of falling through to an empty match
    expression. AND semantics are preserved exactly -- this narrows WHICH
    tokens must co-occur, never loosens co-occurrence to OR. A query whose
    every (filtered) token is genuinely absent from the corpus still returns
    nothing: this is a structural property of AND, not a tuned threshold --
    verified empirically too, see the fix's PR body.
    """
    tokens = re.findall(r"[\w.+-]+", query, flags=re.UNICODE)
    content_tokens = [t for t in tokens if t.lower() not in _FTS_STOPWORDS]
    use = content_tokens if content_tokens else tokens
    return " ".join('"' + token.replace('"', '""') + '"' for token in use)


def _row_to_result(
    row: sqlite3.Row, disclose_path: bool, similarity: float | None = None
) -> SourceSearchResult:
    return SourceSearchResult(
        content=row["content"],
        structural_address=row["structural_address"],
        lifecycle=row["lifecycle"],
        revision_token=row["revision_token"],
        observed_at=row["observed_at"],
        relative_path=row["relative_path"] if disclose_path else None,
        similarity=similarity,
    )


def search_source(
    admission: SourceAdmission,
    open_store: StoreOpener,
    authorize: AuthorizationCheck,
    query: str,
    *,
    limit: int = 5,
    embed_provider: EmbeddingProvider | None = None,
    similarity_floor: float = DEFAULT_SOURCE_SIMILARITY_FLOOR,
) -> list[SourceSearchResult]:
    """Return bounded current results after two authority checks.

    ``embed_provider`` is opt-in and defaults to ``None`` -- every existing
    caller gets byte-for-byte the same lexical-only behavior as before this
    parameter existed. When supplied, BM25 and embedding-cosine candidate
    pools are fused via ``weighted_rrf_merge`` (the same fusion transcript
    search already uses). An embedding-only candidate (no BM25 overlap) is
    admitted only above ``similarity_floor``; a candidate found via BM25
    is never excluded by the floor, matching it or not. Every result whose
    unit has a stored embedding carries its cosine in ``similarity`` --
    whether or not that score cleared the floor -- so a caller can see a
    0.42 next to a 0.34 and judge, rather than trust an internal admit
    decision alone.
    """

    if limit < 1 or not _authorized(admission, authorize):
        return []

    if embed_provider is None:
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
        return [_row_to_result(row, disclose_path) for row in rows]

    # Hybrid path. Both queries below apply the identical source_id +
    # lifecycle='current' scoping the lexical-only path already proved safe
    # under a shared physical store (test_source_index_cross_org_isolation.py)
    # -- the embeddings table has no source_id column of its own precisely so
    # this join is the only way to reach a row.
    connection = open_store()
    connection.row_factory = sqlite3.Row
    try:
        _ensure_schema(connection)
        expression = _fts_query(query)
        bm25_rows: list[sqlite3.Row] = []
        if expression:
            try:
                bm25_rows = connection.execute(
                    "SELECT u.unit_id, u.content, u.structural_address, u.lifecycle, "
                    "u.revision_token, u.observed_at, d.relative_path "
                    "FROM source_units_fts f "
                    "JOIN source_units u ON u.unit_id = f.unit_id "
                    "JOIN source_documents d ON d.document_id = u.document_id "
                    "WHERE source_units_fts MATCH ? AND u.source_id = ? "
                    "AND u.lifecycle = 'current' AND d.lifecycle = 'current' "
                    "ORDER BY bm25(source_units_fts), u.unit_id LIMIT ?",
                    (expression, admission.source_id, _HYBRID_CANDIDATE_POOL),
                ).fetchall()
            except sqlite3.Error:
                bm25_rows = []

        embedding_rows = connection.execute(
            "SELECT e.unit_id, e.vector, e.dim, u.content, u.structural_address, "
            "u.lifecycle, u.revision_token, u.observed_at, d.relative_path "
            "FROM source_units_embeddings e "
            "JOIN source_units u ON u.unit_id = e.unit_id "
            "JOIN source_documents d ON d.document_id = u.document_id "
            "WHERE u.source_id = ? AND u.lifecycle = 'current' AND d.lifecycle = 'current'",
            (admission.source_id,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()

    if not _authorized(admission, authorize):
        return []

    row_by_unit_id: dict[str, sqlite3.Row] = {row["unit_id"]: row for row in bm25_rows}

    query_vector = embed_provider.embed_single(query)
    cosine_by_unit_id: dict[str, float] = {}
    for row in embedding_rows:
        vector = _unpack_vector(row["vector"], row["dim"])
        cosine_by_unit_id[row["unit_id"]] = _cosine(query_vector, vector)
        row_by_unit_id.setdefault(row["unit_id"], row)

    bm25_ranked = [(row["unit_id"], 0.0) for row in bm25_rows]
    emb_ranked = [
        (unit_id, score)
        for unit_id, score in sorted(
            cosine_by_unit_id.items(), key=lambda item: item[1], reverse=True
        )
        if score >= similarity_floor
    ][:_HYBRID_CANDIDATE_POOL]

    if not bm25_ranked and not emb_ranked:
        return []

    merged = weighted_rrf_merge(bm25_ranked, emb_ranked)[:limit]

    disclose_path = admission.path_disclosure == "relative"
    results: list[SourceSearchResult] = []
    for unit_id, _rrf_score in merged:
        row = row_by_unit_id.get(unit_id)
        if row is None:
            continue
        results.append(
            _row_to_result(row, disclose_path, similarity=cosine_by_unit_id.get(unit_id))
        )
    return results


@dataclass(frozen=True)
class IndexedSourceSearchProvider:
    """Bridge one admitted source index into ordinary recall composition."""

    admission: SourceAdmission
    open_store: StoreOpener
    authorize: AuthorizationCheck

    def search(self, request: SourceSearchRequest) -> list[SourceSearchResult]:
        return search_source(
            self.admission,
            self.open_store,
            self.authorize,
            request.query,
            limit=request.limit,
        )


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


def compose_source_results(
    base_result: str, source_results: Iterable[SourceSearchResult]
) -> str:
    """Deterministically append source-unit results to an existing recall render."""

    source_render = render_source_results(source_results)
    return "\n\n".join(part for part in (base_result, source_render) if part)
