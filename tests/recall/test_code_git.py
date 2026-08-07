"""Tests for the git layer — live history + blame primitives (code_git).

The module under test answers two questions straight from a repository,
computing live with no storage: which commits shaped a file, and which
commit wrote each line of a range. These tests build real throwaway git
repos with pinned identities and timestamps so every assertion is exact.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

from synapt.recall.code_git import BlameSpan, FileCommit, blame_range, file_history

BASE_TS = 1754000000


def _env(author: str = "Ada Dev", ts: int = BASE_TS) -> dict[str, str]:
    """Git env with pinned identity/time and host config isolated away."""
    return {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_AUTHOR_NAME": author,
        "GIT_AUTHOR_EMAIL": "author@example.test",
        "GIT_COMMITTER_NAME": "Committer",
        "GIT_COMMITTER_EMAIL": "committer@example.test",
        "GIT_AUTHOR_DATE": f"{ts} +0000",
        "GIT_COMMITTER_DATE": f"{ts} +0000",
    }


def _git(repo: Path, *args: str, author: str = "Ada Dev", ts: int = BASE_TS) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=_env(author=author, ts=ts),
    )
    assert proc.returncode == 0, f"fixture git {args} failed: {proc.stderr}"
    return proc.stdout


def _commit_file(
    repo: Path,
    name: str,
    content: str,
    subject: str,
    *,
    author: str = "Ada Dev",
    ts: int = BASE_TS,
) -> str:
    (repo / name).write_text(content)
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", subject, author=author, ts=ts)
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    return r


# ---------------------------------------------------------------------------
# file_history
# ---------------------------------------------------------------------------


def test_file_history_newest_first_with_exact_fields(repo: Path) -> None:
    sha1 = _commit_file(repo, "a.py", "one\n", "first", ts=BASE_TS)
    sha2 = _commit_file(repo, "a.py", "two\n", "second", ts=BASE_TS + 100)
    sha3 = _commit_file(repo, "a.py", "three\n", "third", ts=BASE_TS + 200)

    hist = file_history(repo, "a.py")

    assert [c.sha for c in hist] == [sha3, sha2, sha1]
    assert [c.subject for c in hist] == ["third", "second", "first"]
    assert [c.timestamp for c in hist] == [BASE_TS + 200, BASE_TS + 100, BASE_TS]
    assert all(c.author == "Ada Dev" for c in hist)
    assert all(len(c.sha) == 40 for c in hist)
    assert all(isinstance(c, FileCommit) for c in hist)


def test_file_history_follows_rename(repo: Path) -> None:
    _commit_file(repo, "old.py", "def f():\n    return 1\n", "create", ts=BASE_TS)
    _git(repo, "mv", "old.py", "new.py")
    _git(repo, "commit", "-q", "-m", "rename", ts=BASE_TS + 100)
    _commit_file(
        repo, "new.py", "def f():\n    return 2\n", "edit after rename", ts=BASE_TS + 200
    )

    hist = file_history(repo, "new.py")

    # Without --follow the pre-rename commit is invisible; with it all three show.
    assert [c.subject for c in hist] == ["edit after rename", "rename", "create"]


def test_file_history_respects_limit(repo: Path) -> None:
    for i in range(4):
        _commit_file(repo, "a.py", f"v{i}\n", f"commit {i}", ts=BASE_TS + i * 10)

    hist = file_history(repo, "a.py", limit=2)

    assert [c.subject for c in hist] == ["commit 3", "commit 2"]


def test_file_history_untracked_path_returns_empty(repo: Path) -> None:
    _commit_file(repo, "a.py", "one\n", "first")

    assert file_history(repo, "never-existed.py") == []


def test_file_history_not_a_repo_raises(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-repo"
    plain.mkdir()

    with pytest.raises(ValueError, match="not a git repository"):
        file_history(plain, "a.py")


def test_file_history_separator_hostile_metadata(repo: Path) -> None:
    """Subjects and author names containing ``|`` must round-trip exactly."""
    _commit_file(
        repo,
        "a.py",
        "x\n",
        "feat: add a|b [x|y] handling",
        author="Grace|Hopper",
    )

    (entry,) = file_history(repo, "a.py")

    assert entry.subject == "feat: add a|b [x|y] handling"
    assert entry.author == "Grace|Hopper"


# ---------------------------------------------------------------------------
# blame_range
# ---------------------------------------------------------------------------


@pytest.fixture()
def blamed_repo(repo: Path) -> dict[str, object]:
    """Three lines: commit A wrote 1-3, commit B rewrote line 2 only."""
    sha_a = _commit_file(
        repo, "b.py", "alpha\nbeta\ngamma\n", "layer A", author="Ada Dev", ts=BASE_TS
    )
    sha_b = _commit_file(
        repo,
        "b.py",
        "alpha\nBETA2\ngamma\n",
        "layer B",
        author="Barbara Liskov",
        ts=BASE_TS + 500,
    )
    return {"repo": repo, "a": sha_a, "b": sha_b}


def test_blame_range_clusters_contiguous_lines(blamed_repo: dict[str, object]) -> None:
    repo = blamed_repo["repo"]
    sha_a, sha_b = blamed_repo["a"], blamed_repo["b"]

    spans = blame_range(repo, "b.py", 1, 3)

    assert [(s.sha, s.line_start, s.line_end) for s in spans] == [
        (sha_a, 1, 1),
        (sha_b, 2, 2),
        (sha_a, 3, 3),
    ]
    # Metadata resolves on EVERY span, including the second appearance of A —
    # porcelain emits author metadata only the first time a commit appears,
    # so this dies if the parser doesn't carry commit metadata across groups.
    assert spans[0].author == "Ada Dev"
    assert spans[1].author == "Barbara Liskov"
    assert spans[2].author == "Ada Dev"
    assert spans[0].timestamp == BASE_TS
    assert spans[1].timestamp == BASE_TS + 500
    assert spans[2].timestamp == BASE_TS
    assert all(isinstance(s, BlameSpan) for s in spans)


def test_blame_subrange_keeps_file_coordinates(blamed_repo: dict[str, object]) -> None:
    """A -L sub-range reports file line numbers, not range-relative ones."""
    repo = blamed_repo["repo"]
    sha_a, sha_b = blamed_repo["a"], blamed_repo["b"]

    spans = blame_range(repo, "b.py", 2, 3)

    assert [(s.sha, s.line_start, s.line_end) for s in spans] == [
        (sha_b, 2, 2),
        (sha_a, 3, 3),
    ]


def test_blame_reports_final_line_numbers_after_insertion(repo: Path) -> None:
    """Spans carry FINAL (current-file) line numbers, not original ones.

    Porcelain headers hold both; after a prepend the two diverge for every
    shifted line, so a parser reading the wrong field fails here and only
    here — in the other fixtures the numbers happen to coincide.
    """
    sha_a = _commit_file(repo, "d.py", "alpha\nbeta\n", "original", ts=BASE_TS)
    sha_b = _commit_file(
        repo, "d.py", "NEW\nalpha\nbeta\n", "prepend", ts=BASE_TS + 50
    )

    spans = blame_range(repo, "d.py", 1, 3)

    assert [(s.sha, s.line_start, s.line_end) for s in spans] == [
        (sha_b, 1, 1),
        (sha_a, 2, 3),
    ]


def test_blame_merges_adjacent_lines_from_same_commit(repo: Path) -> None:
    sha = _commit_file(repo, "c.py", "l1\nl2\nl3\nl4\n", "all at once")

    spans = blame_range(repo, "c.py", 1, 4)

    assert [(s.sha, s.line_start, s.line_end) for s in spans] == [(sha, 1, 4)]


def test_blame_end_beyond_eof_clamps_to_real_lines(
    blamed_repo: dict[str, object]
) -> None:
    """git clamps an end past EOF; we expose that: real blame for real lines.

    A stale caller range (file shrank since it was recorded) still gets
    honest data for the lines that exist rather than an error or padding.
    """
    repo = blamed_repo["repo"]
    sha_a, sha_b = blamed_repo["a"], blamed_repo["b"]

    spans = blame_range(repo, "b.py", 2, 99)

    assert [(s.sha, s.line_start, s.line_end) for s in spans] == [
        (sha_b, 2, 2),
        (sha_a, 3, 3),
    ]


def test_blame_start_beyond_eof_raises(blamed_repo: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="has only"):
        blame_range(blamed_repo["repo"], "b.py", 50, 99)


def test_blame_untracked_path_raises(repo: Path) -> None:
    _commit_file(repo, "a.py", "x\n", "first")

    with pytest.raises(ValueError, match="no such path"):
        blame_range(repo, "ghost.py", 1, 1)


# ---------------------------------------------------------------------------
# boundary: the primitive stays a primitive
# ---------------------------------------------------------------------------


def test_module_imports_no_storage_and_no_memory_layer() -> None:
    """Computed live means computed live: no sqlite, no synapt-internal imports.

    Asserted over the parsed import graph rather than trusted to review,
    matching the convention the symbol index established.
    """
    import synapt.recall.code_git as code_git

    tree = ast.parse(Path(code_git.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)

    assert "sqlite3" not in imported
    assert not any(name == "synapt" or name.startswith("synapt.") for name in imported)
