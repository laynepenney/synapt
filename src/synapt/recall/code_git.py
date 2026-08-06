"""Git history for code files — an OSS primitive, computed live.

Two questions, answered straight from the repository at call time: *which
commits shaped this file* (``file_history``) and *which commit wrote each
line of this range* (``blame_range``). Nothing is stored — no database, no
cache, no index. A downstream consumer that wants to join these facts with
anything else composes on top of this module, never inside it, so this file
imports no memory layer and a test asserts that over the parsed import
graph.

Two implementation notes worth knowing before reading the code:

**Fields travel on the unit separator, not ``|``.** Commit subjects and
author names routinely contain pipes; ``%x1f`` (ASCII unit separator) is a
byte git will not invent inside either field, so parsing is a plain split
with no quoting rules.

**Porcelain blame emits commit metadata once per commit, not once per
line.** A commit's ``author``/``author-time`` lines appear only the first
time that commit shows up in the output; later groups reference it by sha
alone. The parser therefore collects metadata into a per-commit table
during the pass and resolves spans from that table afterwards — resolving
inline would leave every repeated group without an author.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess

__all__ = ["BlameSpan", "FileCommit", "blame_range", "file_history"]

#: ASCII unit separator — a byte that cannot collide with subject/author text.
_FIELD_SEP = "\x1f"

_LOG_FORMAT = _FIELD_SEP.join(("%H", "%an", "%at", "%s"))


@dataclass(frozen=True)
class FileCommit:
    """One commit in a file's history."""

    sha: str
    author: str
    timestamp: int
    subject: str


@dataclass(frozen=True)
class BlameSpan:
    """A contiguous run of lines last written by the same commit.

    ``line_start``/``line_end`` are 1-based, inclusive, in file coordinates
    (not offsets into the requested range).
    """

    sha: str
    author: str
    timestamp: int
    line_start: int
    line_end: int


# A live-per-call primitive needs a ceiling: a git blocked on an index.lock or
# a pathological repository would otherwise hang the caller indefinitely.
_GIT_TIMEOUT_SECONDS = 30.0


def _run_git(repo: str | Path, args: list[str]) -> str:
    # LC_ALL=C pins git's message text: the parsers and error assertions match
    # English strings, and an unpinned locale is green on this machine and red
    # on a non-English one.
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise ValueError(
            f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS:g}s"
        ) from None
    if proc.returncode != 0:
        # The repository path stays out of the message: an error that reaches a
        # user-facing surface must not disclose local layout.
        raise ValueError(f"git {args[0]} failed: {proc.stderr.strip()}")
    return proc.stdout


def file_history(repo: str | Path, path: str, limit: int = 20) -> list[FileCommit]:
    """Commits that touched ``path``, newest first, following renames.

    A path with no history returns an empty list — an honest empty, not an
    error, because "this file has no recorded past" is a real answer.
    A directory that is not a git repository raises ``ValueError``.
    """
    out = _run_git(
        repo,
        [
            "log",
            "--follow",
            f"--format={_LOG_FORMAT}",
            "-n",
            str(limit),
            "--",
            str(path),
        ],
    )
    commits: list[FileCommit] = []
    for line in out.splitlines():
        if not line:
            continue
        sha, author, timestamp, subject = line.split(_FIELD_SEP, 3)
        commits.append(
            FileCommit(
                sha=sha,
                author=author,
                timestamp=int(timestamp),
                subject=subject,
            )
        )
    return commits


def _is_header(fields: list[str]) -> bool:
    """A porcelain header line: ``<40-hex sha> <orig-line> <final-line> [n]``."""
    return (
        len(fields) >= 3
        and len(fields[0]) == 40
        and all(c in "0123456789abcdef" for c in fields[0])
        and fields[1].isdigit()
        and fields[2].isdigit()
    )


def blame_range(
    repo: str | Path, path: str, line_start: int, line_end: int
) -> list[BlameSpan]:
    """Who last wrote lines ``line_start``..``line_end`` (1-based, inclusive).

    Contiguous lines from the same commit merge into one span. A range that
    runs past the end of the file, or a path git does not track, raises
    ``ValueError`` carrying git's own message — a stale range is a caller
    fact worth surfacing, not silently narrowing.
    """
    out = _run_git(
        repo,
        [
            "blame",
            "--porcelain",
            "-L",
            f"{line_start},{line_end}",
            "--",
            str(path),
        ],
    )

    meta: dict[str, dict[str, object]] = {}
    line_owners: list[tuple[int, str]] = []
    cur_sha: str | None = None
    cur_final: int | None = None

    for raw in out.splitlines():
        if raw.startswith("\t"):
            if cur_sha is not None and cur_final is not None:
                line_owners.append((cur_final, cur_sha))
            continue
        fields = raw.split(" ")
        if _is_header(fields):
            cur_sha = fields[0]
            cur_final = int(fields[2])
            meta.setdefault(cur_sha, {})
            continue
        if cur_sha is not None:
            key, _, value = raw.partition(" ")
            if key == "author":
                meta[cur_sha]["author"] = value
            elif key == "author-time":
                meta[cur_sha]["author-time"] = int(value)

    spans: list[BlameSpan] = []
    for final_line, sha in line_owners:
        if spans and spans[-1].sha == sha and spans[-1].line_end == final_line - 1:
            prev = spans.pop()
            spans.append(
                BlameSpan(
                    sha=sha,
                    author=prev.author,
                    timestamp=prev.timestamp,
                    line_start=prev.line_start,
                    line_end=final_line,
                )
            )
        else:
            commit_meta = meta.get(sha, {})
            spans.append(
                BlameSpan(
                    sha=sha,
                    author=str(commit_meta.get("author", "")),
                    timestamp=int(commit_meta.get("author-time", 0)),
                    line_start=final_line,
                    line_end=final_line,
                )
            )
    return spans
