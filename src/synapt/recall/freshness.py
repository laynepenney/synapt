"""Is the index current, and which surface did we check to decide?

`synapt resume` printed `showing 0 of 0 turns` against a store holding 153 real
turns, because the index had not been rebuilt since they were archived. The
turns parsed correctly and archived correctly; only the index was behind. The
output said nothing about that, so a stale index and an empty session rendered
identically and the reader had no way to tell them apart.

This module answers the question that distinguishes them, in two legs:

- **archive** — the index manifest's recorded `(name, mtime, size)` per source
  file, against the files actually in the archive. This is the leg that catches
  the demonstrated defect.
- **archive+sources** — additionally compares the archive against the live
  transcript sources, catching a session that was never archived at all.

Measured on one real store: **~24 ms** for `archive` (14 archived files) and
**~1.2 s** for `archive+sources` (~1000 live files). Both scale with file
count, and the second is dominated not by `stat` but by opening each candidate
Codex session's first line to decide whether it belongs to this project — so a
figure quoted for "enumerate and stat" alone understates it by more than
double. Quote these with their scope or not at all.

`IndexFreshness.scanned` names which leg ran, and that is load-bearing rather
than informational: a verdict that does not state its surface reads as covering
both, so "fresh" would mean two different things depending on the reader.

The cheap shortcut for the second leg does not work, and it is worth saying so
here because it is the first thing anyone will reach for. Comparing *directory*
mtimes detects a file being ADDED — but Codex appends to the session's
start-date file, so a directory whose live session grew all day looks
untouched. The obvious optimization is blind to exactly the case that motivated
this module.

Both legs deliberately reuse the SAME directory helpers the build uses
(``all_worktree_archive_dirs``, ``project_index_dir``) rather than
reimplementing "where do transcripts live". A freshness check that answers a
narrower question than the build answers would report fresh while the build
would find work to do — the guard and the thing it gates disagreeing, with the
guard winning. That defect already happened once in this codebase, in the
Codex build pre-check, and its fix is recorded in
``codex._has_buildable_transcripts``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["IndexFreshness", "check_index_freshness"]


#: The command that rebuilds the index. Carried on the verdict so a stale
#: answer never leaves the reader to guess what would fix it.
REMEDY_COMMAND = "synapt recall build --no-embeddings"


@dataclass(frozen=True)
class IndexFreshness:
    """A freshness verdict, and the surface it was computed over.

    ``scanned`` is ``"archive"`` or ``"archive+sources"``. It is required, not
    optional: an unlabelled verdict is indistinguishable from one that checked
    everything, which is the failure this module exists to prevent.
    """

    stale: bool
    build_timestamp: str
    scanned: str
    new_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    remedy: str = REMEDY_COMMAND


def _live_source_files(project_dir: Path) -> list[Path]:
    """Live transcript files for *project_dir* — Claude projects and Codex sessions.

    Split out as its own function so the deep leg has a seam the spec can
    control without reaching into a real home directory.
    """
    from synapt.recall.codex import discover_codex_sessions, list_codex_transcripts
    from synapt.recall.core import project_transcript_dirs

    files: list[Path] = []
    for src in project_transcript_dirs(project_dir) or []:
        try:
            files.extend(sorted(src.glob("*.jsonl")))
        except OSError:
            continue
    codex_dir = discover_codex_sessions()
    if codex_dir is not None:
        files.extend(list_codex_transcripts(codex_dir, project_dir=project_dir))
    return files


def _archived_files(project_dir: Path) -> dict[str, tuple[float, int]]:
    """``{name: (mtime, size)}`` for every archived transcript the build reads."""
    from synapt.recall.core import all_worktree_archive_dirs, project_archive_dir

    dirs = list(all_worktree_archive_dirs(project_dir))
    own = project_archive_dir(project_dir)
    if own not in dirs:
        dirs.append(own)

    found: dict[str, tuple[float, int]] = {}
    for d in dirs:
        try:
            entries = sorted(d.glob("*.jsonl"))
        except OSError:
            continue
        for p in entries:
            try:
                st = p.stat()
            except OSError:
                continue
            found[p.name] = (st.st_mtime, st.st_size)
    return found


def _read_manifest(project_dir: Path) -> tuple[dict[str, tuple[float, int]], str] | None:
    """``({name: (mtime, size)}, build_timestamp)``, or ``None`` if unreadable.

    ``None`` is not "empty" — it means we could not compute an answer, and the
    caller must fail closed rather than report a clean index.
    """
    import json
    import sqlite3

    from synapt.recall.core import project_index_dir

    db = project_index_dir(project_dir) / "recall.db"
    if not db.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = dict(con.execute(
                "SELECT key, value FROM metadata WHERE key IN "
                "('source_files', 'build_timestamp')"
            ).fetchall())
        finally:
            con.close()
    except sqlite3.Error:
        return None

    try:
        entries = json.loads(rows.get("source_files") or "[]")
    except (TypeError, ValueError):
        return None

    known: dict[str, tuple[float, int]] = {}
    for e in entries:
        if isinstance(e, dict) and e.get("name"):
            known[e["name"]] = (e.get("mtime", 0), e.get("size", 0))
    return known, rows.get("build_timestamp") or ""


def check_index_freshness(project_dir: Path, *, deep: bool = False) -> IndexFreshness:
    """Return whether the index is behind its sources, and over which surface.

    ``deep=False`` compares the index manifest against the archive. ``deep=True``
    additionally compares the archive against the live transcript sources, which
    is the only leg that sees a session appended to but never re-archived.

    Costs are stated in the module docstring with the store they were measured
    on, because a bare millisecond figure invites being quoted against a
    different-sized store than the one that produced it.
    """
    scanned = "archive+sources" if deep else "archive"

    manifest = _read_manifest(project_dir)
    if manifest is None:
        # No index, or one we cannot read. Either way we have not established
        # that it is current, and "we could not tell" must never render as
        # "fresh" — the whole failure this module exists to prevent.
        return IndexFreshness(
            stale=True,
            build_timestamp="",
            scanned=scanned,
            new_files=sorted(_archived_files(project_dir)),
        )
    known, build_timestamp = manifest

    archived = _archived_files(project_dir)
    new = sorted(n for n in archived if n not in known)
    changed = sorted(n for n in archived if n in known and archived[n] != known[n])

    if deep:
        # Compare SIZE only, never mtime. Archiving copies bytes into a new
        # file, so the archived copy's mtime is the copy time and differs from
        # the source's by construction — an mtime comparison here would report
        # every archived transcript as changed, forever. Size is also exactly
        # what ``archive_codex_transcripts`` uses to decide whether to re-copy,
        # so this leg asks the same question the archiver answers.
        for live in _live_source_files(project_dir):
            try:
                live_size = live.stat().st_size
            except OSError:
                continue
            got = archived.get(live.name)
            if got is None:
                if live.name not in new:
                    new.append(live.name)
            elif live_size > got[1]:
                # Only GROWTH means the archive is behind. A source smaller
                # than its archived copy is a truncation the archiver
                # deliberately refuses to propagate.
                if live.name not in changed:
                    changed.append(live.name)
        new.sort()
        changed.sort()

    return IndexFreshness(
        stale=bool(new or changed),
        build_timestamp=build_timestamp,
        scanned=scanned,
        new_files=new,
        changed_files=changed,
    )
