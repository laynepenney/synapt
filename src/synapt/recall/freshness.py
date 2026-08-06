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

STATUS: SPEC ONLY. ``check_index_freshness`` is deliberately unimplemented -- it
returns a fixed always-fresh verdict so the contract is expressible and every
behavioural test fails on its own assertion rather than on an import error. A
red that names which obligation is unmet is worth more than one that says a
module is missing.

Both legs will reuse the SAME directory helpers the build uses
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
    raise NotImplementedError("spec only")


def check_index_freshness(project_dir: Path, *, deep: bool = False) -> IndexFreshness:
    """Return whether the index is behind its sources.

    SPEC ONLY — returns a fixed always-fresh verdict. Every behavioural
    assertion in ``tests/recall/test_freshness.py`` fails against this, by
    design, and each failure names the obligation it is waiting on.
    """
    return IndexFreshness(stale=False, build_timestamp="", scanned="archive")
