"""Tests for GitGrip gripspace detection in recall path resolution."""

import json
import time
from pathlib import Path
from unittest.mock import patch

from synapt.recall.core import (
    _find_gripspace_root,
    _gripspace_cache,
    _GRIPSPACE_CACHE_TTL,
    project_data_dir,
    project_slug,
    project_transcript_dirs,
)


def _make_gripspace(tmp_path: Path) -> Path:
    """Create a minimal gripspace directory structure."""
    grip = tmp_path / "workspace"
    grip.mkdir()
    (grip / ".gitgrip").mkdir()
    # griptrees.json (plural) marks this as the gripspace root
    (grip / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
    return grip


def _make_gr2_workspace(tmp_path: Path) -> Path:
    """Create the marker shared by every gr2-managed workspace."""
    workspace = tmp_path / "gr2-workspace"
    (workspace / ".grip").mkdir(parents=True)
    return workspace


def _make_git_repo(parent: Path, name: str) -> Path:
    """Create a directory that looks like a git repo."""
    repo = parent / name
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


class TestFindGripspaceRoot:
    """Tests for _find_gripspace_root()."""

    def setup_method(self):
        _gripspace_cache.clear()

    def test_finds_gripspace_at_cwd(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        assert _find_gripspace_root(grip) == grip

    def test_finds_gripspace_from_child(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        child = grip / "some-repo"
        child.mkdir()
        assert _find_gripspace_root(child) == grip

    def test_finds_gripspace_from_nested_child(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        deep = grip / "repo" / "src" / "pkg"
        deep.mkdir(parents=True)
        assert _find_gripspace_root(deep) == grip

    def test_finds_gr2_workspace_from_spawned_unit_home(self, tmp_path):
        workspace = _make_gr2_workspace(tmp_path)
        unit_home = workspace / "units" / "u_one" / "home"
        unit_home.mkdir(parents=True)

        assert _find_gripspace_root(unit_home) == workspace

    def test_gr2_units_share_the_workspace_recall_root(self, tmp_path):
        workspace = _make_gr2_workspace(tmp_path)
        first = workspace / "units" / "u_one" / "home"
        second = workspace / "units" / "u_two" / "home"
        first.mkdir(parents=True)
        second.mkdir(parents=True)

        expected = workspace / ".synapt" / "recall"
        assert project_data_dir(first) == expected
        assert project_data_dir(second) == expected

    def test_returns_none_outside_gripspace(self, tmp_path):
        # No .gitgrip anywhere under tmp_path
        plain = tmp_path / "standalone"
        plain.mkdir()
        assert _find_gripspace_root(plain) is None

    def test_returns_none_for_standalone_git_repo(self, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        (repo / ".git").mkdir()
        assert _find_gripspace_root(repo) is None

    def test_linked_griptree_resolves_to_parent_gripspace(self, tmp_path):
        """A linked griptree (griptree.json singular) should resolve
        to the parent gripspace via git worktree pointers."""
        # Create main gripspace
        grip = _make_gripspace(tmp_path)
        main_repo = grip / "my-repo"
        main_repo.mkdir()
        git_dir = main_repo / ".git"
        git_dir.mkdir()
        worktrees_dir = git_dir / "worktrees" / "dev"
        worktrees_dir.mkdir(parents=True)

        # Create linked griptree (sibling of gripspace, NOT a child)
        griptree = tmp_path / "dev-tree"
        griptree.mkdir()
        (griptree / ".gitgrip").mkdir()
        # griptree.json (singular) = linked griptree, NOT a gripspace root
        (griptree / ".gitgrip" / "griptree.json").write_text(
            '{"branch": "dev", "path": "' + str(griptree) + '"}'
        )
        # Sub-repo with .git file pointing to main worktree
        linked_repo = griptree / "my-repo"
        linked_repo.mkdir()
        (linked_repo / ".git").write_text(
            f"gitdir: {worktrees_dir}\n"
        )

        result = _find_gripspace_root(griptree)
        assert result == grip

    def test_linked_griptree_data_dir_matches_parent(self, tmp_path):
        """project_data_dir from a linked griptree should match the parent gripspace."""
        grip = _make_gripspace(tmp_path)
        main_repo = grip / "my-repo"
        main_repo.mkdir()
        git_dir = main_repo / ".git"
        git_dir.mkdir()
        worktrees_dir = git_dir / "worktrees" / "dev"
        worktrees_dir.mkdir(parents=True)

        griptree = tmp_path / "dev-tree"
        griptree.mkdir()
        (griptree / ".gitgrip").mkdir()
        (griptree / ".gitgrip" / "griptree.json").write_text(
            '{"branch": "dev", "path": "' + str(griptree) + '"}'
        )
        linked_repo = griptree / "my-repo"
        linked_repo.mkdir()
        (linked_repo / ".git").write_text(
            f"gitdir: {worktrees_dir}\n"
        )

        result = project_data_dir(griptree)
        assert result == grip / ".synapt" / "recall"

    def test_linked_griptree_no_subrepos_returns_none(self, tmp_path):
        """A linked griptree with no sub-repos can't resolve — returns None."""
        griptree = tmp_path / "orphan-tree"
        griptree.mkdir()
        (griptree / ".gitgrip").mkdir()
        (griptree / ".gitgrip" / "griptree.json").write_text(
            '{"branch": "dev", "path": "' + str(griptree) + '"}'
        )
        # No sub-repos with .git files
        (griptree / "docs").mkdir()

        result = _find_gripspace_root(griptree)
        assert result is None

    def test_stops_at_home_directory(self, tmp_path):
        """Should not walk above $HOME."""
        # Put a .gitgrip ABOVE the fake $HOME
        fake_root = tmp_path / "root"
        fake_root.mkdir()
        (fake_root / ".gitgrip").mkdir()
        (fake_root / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')

        fake_home = fake_root / "home" / "user"
        fake_home.mkdir(parents=True)

        project = fake_home / "project"
        project.mkdir()

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            _gripspace_cache.clear()
            result = _find_gripspace_root(project)
        assert result is None

    def test_gripspace_at_home_is_found(self, tmp_path):
        """A gripspace AT $HOME itself should still be found."""
        fake_home = tmp_path / "home" / "user"
        fake_home.mkdir(parents=True)
        (fake_home / ".gitgrip").mkdir()
        (fake_home / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')

        project = fake_home / "project"
        project.mkdir()

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            _gripspace_cache.clear()
            result = _find_gripspace_root(project)
        assert result == fake_home

    def test_cache_hit(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        child = grip / "repo"
        child.mkdir()

        # First call populates cache
        result1 = _find_gripspace_root(child)
        assert result1 == grip

        # Second call should use cache (even if we remove griptrees.json)
        import shutil
        shutil.rmtree(grip / ".gitgrip")
        result2 = _find_gripspace_root(child)
        assert result2 == grip  # cached result

    def test_cache_miss_returns_none(self, tmp_path):
        plain = tmp_path / "no-grip"
        plain.mkdir()

        result = _find_gripspace_root(plain)
        assert result is None
        # Verify None is cached too (stored as (value, timestamp) tuple)
        assert str(plain) in _gripspace_cache
        value, ts = _gripspace_cache[str(plain)]
        assert value is None

    def test_cache_expires_after_ttl(self, tmp_path):
        """Cache entries expire after TTL, picking up new gripspaces."""
        plain = tmp_path / "workspace"
        plain.mkdir()

        # First call: no gripspace → caches None
        result1 = _find_gripspace_root(plain)
        assert result1 is None

        # Simulate TTL expiry by backdating the cache entry
        cache_key = str(plain.resolve())
        value, _ = _gripspace_cache[cache_key]
        _gripspace_cache[cache_key] = (value, time.monotonic() - _GRIPSPACE_CACHE_TTL - 1)

        # Now add .gitgrip/ with griptrees.json and call again — should find it
        (plain / ".gitgrip").mkdir()
        (plain / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
        result2 = _find_gripspace_root(plain)
        assert result2 == plain


class TestProjectDataDirGripspace:
    """Tests for gripspace resolution in project_data_dir()."""

    def setup_method(self):
        _gripspace_cache.clear()

    def test_gripspace_root_resolves_to_itself(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        result = project_data_dir(grip)
        assert result == grip / ".synapt" / "recall"

    def test_sub_repo_resolves_to_gripspace_root(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        repo = _make_git_repo(grip, "my-repo")
        result = project_data_dir(repo)
        assert result == grip / ".synapt" / "recall"

    def test_standalone_git_repo_unaffected(self, tmp_path):
        repo = tmp_path / "standalone"
        repo.mkdir()
        (repo / ".git").mkdir()
        result = project_data_dir(repo)
        assert result == repo / ".synapt" / "recall"

    def test_all_sub_repos_share_same_data_dir(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        repo_a = _make_git_repo(grip, "repo-a")
        repo_b = _make_git_repo(grip, "repo-b")

        result_root = project_data_dir(grip)
        result_a = project_data_dir(repo_a)
        result_b = project_data_dir(repo_b)

        assert result_root == result_a == result_b


class TestProjectTranscriptDirsGripspace:
    """Tests for gripspace-aware transcript discovery."""

    def setup_method(self):
        _gripspace_cache.clear()

    def test_discovers_sub_repo_transcripts(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        repo_a = _make_git_repo(grip, "repo-a")
        repo_b = _make_git_repo(grip, "repo-b")

        # Create fake Claude Code transcript dirs
        fake_home = tmp_path / "home"
        slug_a = str(repo_a).replace("\\", "/").replace("/", "-")
        slug_b = str(repo_b).replace("\\", "/").replace("/", "-")
        td_a = fake_home / ".claude" / "projects" / slug_a
        td_b = fake_home / ".claude" / "projects" / slug_b
        td_a.mkdir(parents=True)
        td_b.mkdir(parents=True)
        (td_a / "session1.jsonl").write_text("{}")
        (td_b / "session2.jsonl").write_text("{}")

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            dirs = project_transcript_dirs(grip)

        assert td_a in dirs
        assert td_b in dirs

    def test_deduplicates_cwd_and_sub_repo(self, tmp_path):
        """If CWD is a sub-repo, don't include it twice."""
        grip = _make_gripspace(tmp_path)
        repo = _make_git_repo(grip, "my-repo")

        fake_home = tmp_path / "home"
        slug = str(repo).replace("\\", "/").replace("/", "-")
        td = fake_home / ".claude" / "projects" / slug
        td.mkdir(parents=True)
        (td / "session.jsonl").write_text("{}")

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            dirs = project_transcript_dirs(repo)

        # Should appear exactly once
        assert dirs.count(td) == 1

    def test_skips_non_git_children(self, tmp_path):
        """Non-repo directories in gripspace are ignored."""
        grip = _make_gripspace(tmp_path)
        _make_git_repo(grip, "real-repo")
        (grip / "docs").mkdir()  # not a git repo
        (grip / "scripts").mkdir()  # not a git repo

        fake_home = tmp_path / "home"
        fake_home.mkdir(parents=True)

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            dirs = project_transcript_dirs(grip)

        # Should work without errors; only real-repo would be checked
        assert isinstance(dirs, list)

    def test_includes_gripspace_root_own_transcripts(self, tmp_path):
        """Gripspace root's own transcript dir is included via the CWD slug."""
        grip = _make_gripspace(tmp_path)
        repo = _make_git_repo(grip, "repo-a")

        fake_home = tmp_path / "home"
        # Create transcript dir for the gripspace root itself
        slug_root = str(grip).replace("\\", "/").replace("/", "-")
        td_root = fake_home / ".claude" / "projects" / slug_root
        td_root.mkdir(parents=True)
        (td_root / "session-root.jsonl").write_text("{}")

        # And for the sub-repo
        slug_repo = str(repo).replace("\\", "/").replace("/", "-")
        td_repo = fake_home / ".claude" / "projects" / slug_repo
        td_repo.mkdir(parents=True)
        (td_repo / "session-repo.jsonl").write_text("{}")

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            dirs = project_transcript_dirs(grip)

        assert td_root in dirs
        assert td_repo in dirs

    def test_discovers_linked_worktree_transcripts_from_gripspace_root(self, tmp_path):
        """Gripspace-root discovery should include linked repos under .worktrees/*."""
        grip = _make_gripspace(tmp_path)
        repo = _make_git_repo(grip, "repo-a")

        worktrees_root = grip / ".worktrees"
        worktrees_root.mkdir()
        linked = worktrees_root / "atlas-repo-a"
        linked.mkdir()
        (linked / ".git").write_text(f"gitdir: {repo / '.git' / 'worktrees' / 'atlas'}\n")

        fake_home = tmp_path / "home"
        slug_linked = str(linked).replace("\\", "/").replace("/", "-")
        td_linked = fake_home / ".claude" / "projects" / slug_linked
        td_linked.mkdir(parents=True)
        (td_linked / "session-linked.jsonl").write_text("{}")

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            dirs = project_transcript_dirs(grip)

        assert td_linked in dirs

    def test_discovers_declared_sibling_griptree_transcripts(self, tmp_path):
        """Gripspace-root discovery should include griptrees declared in griptrees.json."""
        grip = _make_gripspace(tmp_path)

        sibling = tmp_path / "sibling-tree"
        sibling.mkdir()
        (grip / ".gitgrip" / "griptrees.json").write_text(
            json.dumps({"griptrees": {"sibling-tree": {"path": str(sibling)}}})
        )

        fake_home = tmp_path / "home"
        slug_sibling = str(sibling).replace("\\", "/").replace("/", "-")
        td_sibling = fake_home / ".claude" / "projects" / slug_sibling
        td_sibling.mkdir(parents=True)
        (td_sibling / "session-sibling.jsonl").write_text("{}")

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            dirs = project_transcript_dirs(grip)

        assert td_sibling in dirs

    def test_standalone_repo_returns_empty_when_no_transcripts(self, tmp_path):
        """Standalone git repo outside any gripspace."""
        repo = tmp_path / "standalone"
        repo.mkdir()
        (repo / ".git").mkdir()

        fake_home = tmp_path / "home"
        fake_home.mkdir(parents=True)

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            dirs = project_transcript_dirs(repo)

        assert dirs == []


class TestGripspaceEdgeCases:
    """Edge cases for gripspace detection."""

    def setup_method(self):
        _gripspace_cache.clear()

    def test_path_outside_home_does_not_crash(self, tmp_path):
        """Paths outside $HOME don't walk to filesystem root."""
        # tmp_path is typically /tmp/... which is outside $HOME
        project = tmp_path / "outside-home"
        project.mkdir()

        # Fake $HOME that is NOT a parent of tmp_path
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            result = _find_gripspace_root(project)
        # Should return None without errors — walks up to / but finds nothing
        assert result is None

    def test_iterdir_permission_error_handled(self, tmp_path):
        """OSError during iterdir() in gripspace doesn't crash discovery."""
        grip = _make_gripspace(tmp_path)

        fake_home = tmp_path / "home"
        fake_home.mkdir(parents=True)

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            # Mock iterdir to raise PermissionError
            with patch.object(type(grip), "iterdir", side_effect=PermissionError("denied")):
                dirs = project_transcript_dirs(grip)

        # Should return empty list, not crash
        assert isinstance(dirs, list)
