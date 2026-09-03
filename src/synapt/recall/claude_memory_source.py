"""Admit and index an agent's own Claude Code memory directory as a
recall source (R3 "Memory Everywhere" first fruit: what an agent wrote
in one runtime becomes available to the same agent in another runtime).

Self-scoped only: no agent name, role, or org identity is resolved or
stored anywhere in this module. The admission names a DIRECTORY -- the
same store-resolution coordinate ``project_data_dir()`` already computes
for this process -- not an agent. Team/org authorization and visibility
policy are out of scope for this module and may be layered on by a
downstream consumer without changing this admission's contract.

This indexes the ORIGINAL files in place via ``source_index.py``'s
generic engine. It does not copy their content into knowledge nodes the
way ``server.recall_sync_memory`` does -- that is the bulk-load the R3
acceptance fruit exists to avoid.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from synapt.recall.core import (
    _find_gripspace_root,
    _resolve_project_root_override,
    project_data_dir,
    project_slug,
)
from synapt.recall.source_index import (
    SOURCE_INDEX_SUPPORTED,
    DescriptorSourceAdapter,
    IndexedSourceSearchProvider,
    SourceAdmission,
    SourceScanReceipt,
    register_source_search_provider,
    sync_source,
)

CLAUDE_MEMORY_SOURCE_ID = "claude_memory"
CLAUDE_MEMORY_SOURCE_KIND = "claude_memory"


def _resolve_gripspace_root(project_dir: Path | None) -> Path | None:
    """Same precedence ``project_data_dir()``/``channel._read_manifest_url()``
    already use: an explicit ``project_dir`` is a deliberate root and wins
    outright; otherwise ``GRIPSPACE_ROOT`` (or a persisted marker) wins over
    a cwd walk-up. None means no gripspace resolves at all -- there is
    nothing for this process to admit."""
    if project_dir is not None:
        return project_dir
    override = _resolve_project_root_override(None)
    if override is not None:
        return override
    return _find_gripspace_root(Path.cwd())


def _claude_memory_dir(gripspace_root: Path) -> Path:
    """The directory Claude Code itself writes this gripspace's memory
    files to: ``~/.claude/projects/<slug>/memory/``, where ``<slug>`` is
    Claude Code's own path-to-directory-name convention (``project_slug``,
    already used for transcript discovery)."""
    return Path.home() / ".claude" / "projects" / project_slug(gripspace_root) / "memory"


def admit_and_index_claude_memory(
    project_dir: Path | None = None,
) -> SourceScanReceipt | None:
    """Admit this agent's own Claude Code memory directory as a recall
    source and register it for search, if one exists.

    Returns None (no-op, no error) when: this platform can't run the
    descriptor adapter, no gripspace can be resolved, or the resolved
    gripspace has no ``memory/`` directory yet. A caller with something to
    log checks the return value; there is nothing to report on None.
    """
    if not SOURCE_INDEX_SUPPORTED:
        return None
    gripspace_root = _resolve_gripspace_root(project_dir)
    if gripspace_root is None:
        return None
    memory_dir = _claude_memory_dir(gripspace_root)
    if not memory_dir.is_dir():
        return None

    db_path = project_data_dir(gripspace_root) / "source_index" / "claude_memory.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    def open_store() -> sqlite3.Connection:
        return sqlite3.connect(db_path)

    def authorize(_admission: SourceAdmission) -> bool:
        # Self-scoped: this process's own memory directory, already gated
        # by filesystem permissions on the fd below. No cross-agent claim
        # is made or checked here.
        return True

    root_fd = os.open(memory_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = SourceAdmission(
            source_id=CLAUDE_MEMORY_SOURCE_ID,
            scope_capability=b"claude-memory-self",
            root_handle=root_fd,
            root_handle_id=f"claude-memory:{project_slug(gripspace_root)}",
            admission_epoch=1,
            policy_epoch=1,
            path_disclosure="relative",
            source_kind=CLAUDE_MEMORY_SOURCE_KIND,
        )
        receipt = sync_source(admission, DescriptorSourceAdapter(), open_store, authorize)
    finally:
        # Only sync_source's enumerate() step touches the fd; the
        # registered search provider below only ever queries the SQLite
        # index, so closing it here is safe.
        os.close(root_fd)

    if receipt.state == "complete":
        try:
            register_source_search_provider(
                CLAUDE_MEMORY_SOURCE_ID,
                IndexedSourceSearchProvider(admission, open_store, authorize),
            )
        except ValueError:
            # A second admission in this same process (a repeated call, not
            # a name clash with an unrelated caller -- this module is the
            # only one that ever registers CLAUDE_MEMORY_SOURCE_ID) collides
            # with the registry's overwrite guard even though the new
            # provider is functionally identical: search_source() always
            # reads live from the just-resynced SQLite file, never from
            # anything cached on the admission/provider object, so the
            # already-registered provider already reflects this sync.
            pass
    return receipt
