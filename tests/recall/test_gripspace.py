"""Tests for GitGrip gripspace detection in recall path resolution."""

import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from synapt.recall.core import (
    _find_gripspace_root,
    _gripspace_cache,
    _GRIPSPACE_CACHE_TTL,
    project_data_dir,
    project_slug,
    project_worktree_dir,
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

    def test_finds_standalone_clone_gripspace_via_manifest_marker(self, tmp_path):
        """new-agent-gripspace.sh creates a standalone clone: no
        griptree.json, no griptrees.json, no .grip -- only
        .gitgrip/spaces/main/gripspace.yml. Before this fix, such a
        gripspace was invisible to _find_gripspace_root and every
        recall/channel write fell through to $HOME. Same family as
        recall#974 and recall#936."""
        grip = tmp_path / "standalone-space"
        grip.mkdir()
        spaces_main = grip / ".gitgrip" / "spaces" / "main"
        spaces_main.mkdir(parents=True)
        (spaces_main / "gripspace.yml").write_text("version: 2\nrepos: {}\n")

        assert _find_gripspace_root(grip) == grip

    def test_finds_standalone_clone_gripspace_from_child(self, tmp_path):
        grip = tmp_path / "standalone-space"
        spaces_main = grip / ".gitgrip" / "spaces" / "main"
        spaces_main.mkdir(parents=True)
        (spaces_main / "gripspace.yml").write_text("version: 2\nrepos: {}\n")
        child = grip / "synapt"
        child.mkdir()

        assert _find_gripspace_root(child) == grip

    def test_gitgrip_dir_without_the_manifest_marker_still_returns_none(self, tmp_path):
        """A bare .gitgrip/ with none of the three markers (griptree.json,
        griptrees.json, spaces/main/gripspace.yml) must not match --
        pins the marker to the exact path, not the directory's mere
        existence."""
        repo = tmp_path / "myrepo"
        (repo / ".gitgrip").mkdir(parents=True)
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

    def _make_member_carrying_both_markers(self, tmp_path: Path) -> tuple[Path, Path]:
        """A directory that is BOTH a gr2 workspace and a declared member.

        This combination is not hypothetical: a workspace acquires the gr2
        marker when it is managed by gr2, and keeps its membership marker
        because it is still a griptree of a larger gripspace. Every existing
        test in this class builds one marker or the other, which is why the
        suite could not see the ordering between them.
        """
        grip = _make_gripspace(tmp_path)
        main_repo = grip / "my-repo"
        main_repo.mkdir()
        git_dir = main_repo / ".git"
        git_dir.mkdir()
        worktrees_dir = git_dir / "worktrees" / "dev"
        worktrees_dir.mkdir(parents=True)

        member = tmp_path / "dev-tree"
        member.mkdir()
        (member / ".gitgrip").mkdir()
        (member / ".gitgrip" / "griptree.json").write_text(
            '{"branch": "dev", "path": "' + str(member) + '"}'
        )
        linked_repo = member / "my-repo"
        linked_repo.mkdir()
        (linked_repo / ".git").write_text(f"gitdir: {worktrees_dir}\n")
        # ...and it is also a gr2-managed workspace.
        (member / ".grip").mkdir()
        return grip, member

    def test_membership_beats_locality(self, tmp_path):
        """Both markers present: resolve to the whole, not to the part.

        The two markers answer different questions. ".grip" asks "am I a
        workspace containing units?"; the griptree marker asks "whose larger
        whole am I part of?". Store resolution needs the second, because a
        member's data belongs to the workspace it is a member of — resolving
        to itself strands that data where no other surface in the workspace
        looks, and reports nothing.
        """
        grip, member = self._make_member_carrying_both_markers(tmp_path)

        assert _find_gripspace_root(member) == grip

    def test_membership_beats_locality_at_the_data_dir(self, tmp_path):
        """The same claim at the level the caller actually consumes.

        Asserting the resolver alone would leave the reader-visible effect
        unproven, and the reader-visible effect is the whole point: this is
        the path a store lands on.
        """
        grip, member = self._make_member_carrying_both_markers(tmp_path)

        assert project_data_dir(member) == grip / ".synapt" / "recall"

    def test_gr2_locality_still_applies_without_a_membership_claim(self, tmp_path):
        """The control for the two above: locality is narrowed, not removed.

        A gr2 workspace with no membership marker must still resolve to
        itself. Without this, the two tests above would also pass if the
        ".grip" branch had simply been deleted.
        """
        workspace = _make_gr2_workspace(tmp_path)
        unit_home = workspace / "units" / "u_one" / "home"
        unit_home.mkdir(parents=True)

        assert _find_gripspace_root(workspace) == workspace
        assert _find_gripspace_root(unit_home) == workspace

    def _make_unresolvable_member(
        self, parent: Path, with_grip: bool, name: str = "member"
    ) -> Path:
        """A directory declaring membership whose parent cannot be resolved.

        It has sub-directories but none carries a `.git` *file*, so the
        worktree-pointer walk that resolves a griptree back to its parent
        finds nothing to follow and reports no parent.
        """
        member = parent / name
        member.mkdir()
        (member / ".gitgrip").mkdir()
        (member / ".gitgrip" / "griptree.json").write_text('{"branch": "dev"}')
        (member / "docs").mkdir()
        if with_grip:
            (member / ".grip").mkdir()
        return member

    def test_unresolvable_membership_falls_back_to_locality(self, tmp_path):
        """THE REGRESSION WITNESS — consulting membership first must not strand.

        Membership is asserted here and cannot be verified. Before the marker
        order changed, such a directory short-circuited on `.grip` and
        resolved to itself; consulting membership first would instead resolve
        it to nothing, and its sub-directories would each land on a store of
        their own. That fragments a store that previously cohered, which is
        the exact failure this function is being changed to prevent.

        So the fall-through is not a nicety: without it this change is a
        regression for this population. Pinned so no later edit can quietly
        re-break it.
        """
        member = self._make_unresolvable_member(tmp_path, with_grip=True)
        deep = member / "docs" / "deep"
        deep.mkdir(parents=True)

        assert _find_gripspace_root(member) == member
        assert _find_gripspace_root(deep) == member
        assert project_data_dir(deep) == member / ".synapt" / "recall"

    def test_unresolvable_membership_without_locality_does_not_walk_upward(
        self, tmp_path
    ):
        """The third case, pinned deliberately UNCHANGED rather than improved.

        Membership asserted, unverifiable, and no locality evidence either.
        There IS a gripspace root above it, so continuing the walk would find
        a better answer than `None` — and that is precisely why this asserts
        `None`. A real improvement here belongs to the change that can bring
        its own evidence for it. Cause 1's claim is that every directory
        either improves or is unchanged and nothing else moves; the moment it
        also improves an unrelated case, that claim stops being checkable.
        """
        grip = _make_gripspace(tmp_path)
        member = self._make_unresolvable_member(grip, with_grip=False)

        assert _find_gripspace_root(member) is None

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


def test_worktree_bucket_is_stable_below_a_gr2_workspace(tmp_path):
    """A subdirectory must read the workspace's journal, not mint a slice.

    The ``.grip`` marker deliberately holds the data root constant.  Before
    recall#974's fix, only the bucket changed from ``workspace`` to ``server``.
    """
    workspace = _make_gr2_workspace(tmp_path)
    nested = workspace / "server"
    nested.mkdir()

    root_bucket = project_worktree_dir(workspace)
    nested_bucket = project_worktree_dir(nested)

    assert root_bucket == nested_bucket
    assert root_bucket == workspace / ".synapt" / "recall" / "worktrees" / "gr2-workspace"


def test_gr2_workspace_boundary_beats_an_enclosing_git_root(tmp_path):
    """A workspace marker owns its namespace even inside another checkout."""
    (tmp_path / ".git").mkdir()
    workspace = _make_gr2_workspace(tmp_path)
    nested = workspace / "server"
    nested.mkdir()

    assert project_worktree_dir(nested) == (
        workspace / ".synapt" / "recall" / "worktrees" / "gr2-workspace"
    )


def test_worktree_bucket_uses_the_repo_root_beneath_a_gripspace(tmp_path):
    """Constituent repos share the store but retain distinct stable buckets."""
    grip = _make_gripspace(tmp_path)
    repo = _make_git_repo(grip, "repo-a")
    nested = repo / "src" / "synapt"
    nested.mkdir(parents=True)

    repo_bucket = project_worktree_dir(repo)
    nested_bucket = project_worktree_dir(nested)

    assert repo_bucket == nested_bucket
    assert repo_bucket == grip / ".synapt" / "recall" / "worktrees" / "repo-a"


def test_worktree_bucket_is_stable_below_a_linked_griptree(tmp_path):
    """A linked griptree root is its own bucket even without a root .git."""
    grip = _make_gripspace(tmp_path)
    main_repo = _make_git_repo(grip, "repo-a")
    worktree_dir = main_repo / ".git" / "worktrees" / "dev"
    worktree_dir.mkdir(parents=True)

    griptree = tmp_path / "dev-tree"
    griptree.mkdir()
    (griptree / ".gitgrip").mkdir()
    (griptree / ".gitgrip" / "griptree.json").write_text("{}")
    linked_repo = griptree / "repo-a"
    linked_repo.mkdir()
    (linked_repo / ".git").write_text(f"gitdir: {worktree_dir}\n")
    nested = griptree / "docs"
    nested.mkdir()

    root_bucket = project_worktree_dir(griptree)
    nested_bucket = project_worktree_dir(nested)

    assert root_bucket == nested_bucket
    assert root_bucket == grip / ".synapt" / "recall" / "worktrees" / "dev-tree"


def test_worktree_bucket_uses_a_linked_repo_file_as_its_root(tmp_path):
    """A linked repository's .git file is a root marker, not a directory."""
    grip = _make_gripspace(tmp_path)
    main_repo = _make_git_repo(grip, "repo-a")
    worktree_dir = main_repo / ".git" / "worktrees" / "dev"
    worktree_dir.mkdir(parents=True)

    griptree = tmp_path / "dev-tree"
    griptree.mkdir()
    (griptree / ".gitgrip").mkdir()
    (griptree / ".gitgrip" / "griptree.json").write_text("{}")
    linked_repo = griptree / "repo-a"
    linked_repo.mkdir()
    (linked_repo / ".git").write_text(f"gitdir: {worktree_dir}\n")
    nested = linked_repo / "src" / "synapt"
    nested.mkdir(parents=True)

    root_bucket = project_worktree_dir(linked_repo)
    nested_bucket = project_worktree_dir(nested)

    assert root_bucket == nested_bucket
    assert root_bucket == grip / ".synapt" / "recall" / "worktrees" / "repo-a"


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


# ---------------------------------------------------------------------------
# Explicit root override — SYNAPT_RECALL_ROOT / SYNAPT_RECALL_WORKTREE
#
# A CLI invoked from inside one workspace can be told to use another
# workspace's store. Path inference cannot discover that relationship when the
# two roots are filesystem SIBLINGS (recall#936): walking up from a caller's
# tree can never arrive at a root that does not contain it. So the override is
# EXPLICIT, and only consulted when the caller passes no project_dir.
# ---------------------------------------------------------------------------


def test_env_root_wins_over_gripspace_inference(tmp_path, monkeypatch):
    grip = _make_gripspace(tmp_path)
    repo = _make_git_repo(grip, "repo")
    shared = tmp_path / "shared-workspace"
    shared.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(shared))
    assert project_data_dir(None) == shared / ".synapt" / "recall"


def test_explicit_arg_beats_env_root(tmp_path, monkeypatch):
    """A programmatic caller that names a root is more explicit than the
    environment: tests pass tmp dirs, the server passes resolved dirs, and an
    env var that hijacked those would fail far from its cause."""
    shared = tmp_path / "shared"
    shared.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(shared))
    assert project_data_dir(other) == other / ".synapt" / "recall"


def test_env_root_that_does_not_exist_raises(tmp_path, monkeypatch):
    """A typo'd override must fail loudly, never silently mint a fresh store.

    Silently creating an empty store under a mistyped root is the worst
    failure available here: every read then reports an empty history that
    looks exactly like a real answer (the recall#936 presentation).
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(tmp_path / "typo"))
    with pytest.raises(ValueError, match="SYNAPT_RECALL_ROOT"):
        project_data_dir(None)


def test_empty_env_root_is_ignored(tmp_path, monkeypatch):
    """`VAR=` shell artifacts mean unset, not "use the empty path".

    Run from INSIDE a gripspace, where the two behaviours differ:
    ``Path("").resolve()`` is the cwd, so in a bare tmp dir empty-as-set and
    empty-as-unset produce the same path and the test cannot fail (caught by
    reviewer mutation, 2026-08-06). Inference must win — the gripspace root,
    not the cwd.
    """
    grip = _make_gripspace(tmp_path)
    repo = _make_git_repo(grip, "repo")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", "")
    assert project_data_dir(None) == grip / ".synapt" / "recall"


def test_env_worktree_names_the_namespace(tmp_path, monkeypatch):
    """Store location and worktree identity are SEPARATE overrides.

    Redirecting only the root would file the caller's per-worktree data under
    its cwd basename inside the shared store — potentially another workspace's
    namespace. That is cross-attribution (one workspace's journal presented
    under another's name), not sharing.
    """
    shared = tmp_path / "shared"
    shared.mkdir()
    caller = tmp_path / "elsewhere" / "repo"
    caller.mkdir(parents=True)
    monkeypatch.chdir(caller)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(shared))
    monkeypatch.setenv("SYNAPT_RECALL_WORKTREE", "unit-b")
    assert project_worktree_dir(None) == (
        shared / ".synapt" / "recall" / "worktrees" / "unit-b"
    )


def test_env_worktree_unset_falls_back_to_cwd_basename(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    caller = tmp_path / "elsewhere" / "repo"
    caller.mkdir(parents=True)
    monkeypatch.chdir(caller)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(shared))
    assert project_worktree_dir(None) == (
        shared / ".synapt" / "recall" / "worktrees" / "repo"
    )


def test_no_override_keeps_inference_unchanged(tmp_path, monkeypatch):
    """Control: with neither variable set, resolution is the pre-seam behavior."""
    monkeypatch.chdir(tmp_path)
    assert project_data_dir(None) == tmp_path / ".synapt" / "recall"


def test_explicit_arg_beats_env_worktree(tmp_path, monkeypatch):
    """The precedence pinned for ROOT holds for WORKTREE too (reviewer
    mutation survived without this: the guard existed, unwitnessed)."""
    explicit = tmp_path / "explicit-dir"
    explicit.mkdir()
    monkeypatch.setenv("SYNAPT_RECALL_WORKTREE", "hijack")
    wt = project_worktree_dir(explicit)
    assert wt.name == "explicit-dir"


def test_env_root_with_legacy_store_migrates(tmp_path, monkeypatch):
    """An overridden root may be a PRE-RENAME workspace (reviewer finding,
    ran with a control on the inference path): the override must flow through
    the same legacy migration, or real history sits stranded in .synapse
    while resume reads an empty .synapt that looks like an empty history.
    """
    shared = tmp_path / "old-workspace"
    legacy = shared / ".synapse" / "recall"
    legacy.mkdir(parents=True)
    (legacy / "marker.txt").write_text("real history")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(shared))

    resolved = project_data_dir(None)
    assert resolved == shared / ".synapt" / "recall"
    assert (resolved / "marker.txt").read_text() == "real history"
    assert not (shared / ".synapse").exists()


def test_env_worktree_rejects_path_components(tmp_path, monkeypatch):
    """The namespace label is one path component, never a path. ".." would
    relocate per-worktree files to the store root."""
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(shared))
    for bad in ("..", ".", "a/b", "a\\b"):
        monkeypatch.setenv("SYNAPT_RECALL_WORKTREE", bad)
        with pytest.raises(ValueError, match="SYNAPT_RECALL_WORKTREE"):
            project_worktree_dir(None)


class TestGripspaceRootEnv:
    """recall honors GRIPSPACE_ROOT (exported by the workspace tool's
    find_workspace_root) so a fresh gripspace whose agent worktree is a
    filesystem SIBLING of the workspace root resolves to the SHARED store
    instead of silently minting one at cwd or $HOME. Walking up from a sibling
    can never reach the workspace, which is the live bug: a fresh session sees
    'no index', an empty shared channel, and no journal entries.

    Ordering: explicit SYNAPT_RECALL_ROOT > GRIPSPACE_ROOT > walk-up > $HOME
    NAMED ERROR. Both env roots are consulted only in the None-branch ('resolve
    like every recall verb'); an explicitly-passed project_dir suppresses them,
    preserving the deliberate export/import --path contract (cli.py:1583).
    """

    def setup_method(self):
        _gripspace_cache.clear()

    def test_gripspace_root_env_resolves_the_shared_store(self, tmp_path, monkeypatch):
        grip = _make_gr2_workspace(tmp_path)  # the workspace root, has .grip
        sibling = tmp_path / "agent-worktree"  # sibling of grip, NOT inside it
        sibling.mkdir()
        monkeypatch.chdir(sibling)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(grip))
        assert project_data_dir() == grip / ".synapt" / "recall"

    def test_walk_up_alone_cannot_reach_a_sibling_workspace(self, tmp_path, monkeypatch):
        # The bug, pinned: with no GRIPSPACE_ROOT, a sibling worktree does NOT
        # resolve to the workspace (walk-up misses it).
        grip = _make_gr2_workspace(tmp_path)
        sibling = tmp_path / "agent-worktree"
        sibling.mkdir()
        monkeypatch.chdir(sibling)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        assert project_data_dir() != grip / ".synapt" / "recall"

    def test_explicit_recall_root_beats_gripspace_root(self, tmp_path, monkeypatch):
        grip = _make_gr2_workspace(tmp_path)
        store = tmp_path / "explicit-store"
        store.mkdir()
        monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(store))
        monkeypatch.setenv("GRIPSPACE_ROOT", str(grip))
        assert project_data_dir() == store / ".synapt" / "recall"

    def test_refuses_nonexistent_gripspace_root(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(tmp_path / "does-not-exist"))
        with pytest.raises(ValueError, match="GRIPSPACE_ROOT"):
            project_data_dir()

    def test_home_is_never_a_store_root(self, tmp_path, monkeypatch):
        # No env roots and inference lands on $HOME -> NAMED ERROR, never a mint.
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        monkeypatch.chdir(fake_home)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        with pytest.raises(ValueError, match="(?i)home"):
            project_data_dir()

    def test_explicit_project_dir_suppresses_gripspace_root(self, tmp_path, monkeypatch):
        # The deliberate export/import --path contract: an explicitly-passed
        # project_dir suppresses the env override (cli.py:1583).
        grip = _make_gr2_workspace(tmp_path)
        deliberate = tmp_path / "deliberate-target"
        deliberate.mkdir()
        monkeypatch.setenv("GRIPSPACE_ROOT", str(grip))
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        assert project_data_dir(deliberate) == deliberate / ".synapt" / "recall"


def _make_named_gripspace(tmp_path: Path, name: str) -> Path:
    """A second, independently self-resolvable gripspace (its own real
    .gitgrip/griptrees.json), distinct from _make_gripspace's default -- for
    tests that need TWO real gripspaces (the agent's own worktree and the
    shared canonical root) rather than one gripspace plus an unmarked dir.
    """
    grip = tmp_path / name
    (grip / ".gitgrip").mkdir(parents=True)
    (grip / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
    return grip


class TestGripspaceRootMarkerPersistence:
    """recall#936, narrowed: GRIPSPACE_ROOT binds an env-carrying process (the
    MCP server, always gr-spawned) to a shared coordinate, but a bare CLI
    invocation from a shell that never inherited that env var self-resolves
    to its OWN worktree's gripspace instead -- correct in isolation, but a
    real divergence when the two are different real, independently
    self-resolvable gripspaces (not the unreachable-sibling shape
    TestGripspaceRootEnv covers, where walk-up cannot even reach the target).

    Fix direction: an env-bound call PERSISTS the resolved shared root to a
    marker file inside the CALLER's own self-resolved gripspace
    (.synapt/gripspace-root); a later env-less call in the same gripspace
    reads that marker and converges on the same coordinate -- "computed once
    and shared" rather than "bound explicitly at launch" (GRIPSPACE_ROOT
    itself already covers the latter).
    """

    def setup_method(self):
        _gripspace_cache.clear()

    def test_env_bound_call_persists_a_marker_in_the_callers_own_gripspace(
        self, tmp_path, monkeypatch
    ):
        agent_ws = _make_named_gripspace(tmp_path, "agent-worktree")
        shared_ws = _make_named_gripspace(tmp_path, "shared-canonical")
        monkeypatch.chdir(agent_ws)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(shared_ws))
        assert project_data_dir() == shared_ws / ".synapt" / "recall"
        marker = agent_ws / ".synapt" / "gripspace-root"
        assert marker.is_file()
        assert Path(marker.read_text().strip()) == shared_ws

    def test_bare_cli_with_no_env_converges_via_the_persisted_marker(
        self, tmp_path, monkeypatch
    ):
        agent_ws = _make_named_gripspace(tmp_path, "agent-worktree")
        shared_ws = _make_named_gripspace(tmp_path, "shared-canonical")
        monkeypatch.chdir(agent_ws)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        # Earlier call, as the MCP server would make it: env-bound, persists
        # the marker as a side effect.
        monkeypatch.setenv("GRIPSPACE_ROOT", str(shared_ws))
        project_data_dir()
        _gripspace_cache.clear()
        # Later call, as a bare `synapt` CLI invocation would make it: same
        # cwd, same gripspace, but NO env var in this shell.
        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        assert project_data_dir() == shared_ws / ".synapt" / "recall"

    def test_no_marker_and_no_env_falls_back_to_self_resolution_unchanged(
        self, tmp_path, monkeypatch
    ):
        agent_ws = _make_named_gripspace(tmp_path, "agent-worktree")
        monkeypatch.chdir(agent_ws)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        # No env-bound call has ever run here, so no marker exists -- must
        # behave exactly as before the fix: self-resolve to the agent's own
        # gripspace, and never invent a marker file out of nothing.
        assert project_data_dir() == agent_ws / ".synapt" / "recall"
        assert not (agent_ws / ".synapt" / "gripspace-root").exists()

    def test_marker_pointing_at_a_no_longer_gripspace_is_ignored_not_followed(
        self, tmp_path, monkeypatch
    ):
        """A moved coordinator tree or a deleted gripspace must not redirect
        a worktree forever. The marker is a CACHED coordinate, not a live
        pointer -- if its recorded target no longer carries its own
        gripspace marker (.gitgrip/.grip), it is stale and must be ignored,
        falling through to self-resolution exactly as if no marker existed.
        """
        agent_ws = _make_named_gripspace(tmp_path, "agent-worktree")
        shared_ws = _make_named_gripspace(tmp_path, "shared-canonical")
        monkeypatch.chdir(agent_ws)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(shared_ws))
        project_data_dir()  # persists the marker pointing at shared_ws
        _gripspace_cache.clear()

        # The coordinator tree is gone: its gripspace marker is removed
        # (the directory itself may or may not still exist -- either way it
        # is no longer a gripspace).
        shutil.rmtree(shared_ws / ".gitgrip")

        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        assert project_data_dir() == agent_ws / ".synapt" / "recall"

    def test_marker_pointing_at_a_deleted_directory_is_ignored_not_followed(
        self, tmp_path, monkeypatch
    ):
        agent_ws = _make_named_gripspace(tmp_path, "agent-worktree")
        shared_ws = _make_named_gripspace(tmp_path, "shared-canonical")
        monkeypatch.chdir(agent_ws)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(shared_ws))
        project_data_dir()
        _gripspace_cache.clear()

        shutil.rmtree(shared_ws)

        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        assert project_data_dir() == agent_ws / ".synapt" / "recall"

    def test_valid_marker_still_used_when_target_still_a_real_gripspace(
        self, tmp_path, monkeypatch
    ):
        """Control for the two staleness tests above: an unmutated marker
        must still be followed. Without this, a mutation that always
        ignores the marker would pass the staleness tests trivially."""
        agent_ws = _make_named_gripspace(tmp_path, "agent-worktree")
        shared_ws = _make_named_gripspace(tmp_path, "shared-canonical")
        monkeypatch.chdir(agent_ws)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(shared_ws))
        project_data_dir()
        _gripspace_cache.clear()

        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        assert project_data_dir() == shared_ws / ".synapt" / "recall"


class TestGripspaceRootSourceDisclosure:
    """One word naming which of env:GRIPSPACE_ROOT / env:SYNAPT_RECALL_ROOT /
    marker:<path> / walk-up chose the coordinate -- so a reader of the
    existing store-disclosure lines can see that a marker (rather than a
    live env var) picked the root, per the same visibility discipline as
    the orphaned-store line and the provenance line.
    """

    def setup_method(self):
        _gripspace_cache.clear()

    def test_source_is_gripspace_root_env_when_set(self, tmp_path, monkeypatch):
        from synapt.recall.core import describe_root_source

        shared_ws = _make_named_gripspace(tmp_path, "shared")
        monkeypatch.chdir(shared_ws)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(shared_ws))
        assert describe_root_source() == "env:GRIPSPACE_ROOT"

    def test_source_is_synapt_recall_root_env_when_set(self, tmp_path, monkeypatch):
        from synapt.recall.core import describe_root_source

        store = tmp_path / "store"
        store.mkdir()
        monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(store))
        assert describe_root_source() == "env:SYNAPT_RECALL_ROOT"

    def test_source_is_marker_path_when_converging_via_a_persisted_marker(
        self, tmp_path, monkeypatch
    ):
        from synapt.recall.core import describe_root_source

        agent_ws = _make_named_gripspace(tmp_path, "agent-worktree")
        shared_ws = _make_named_gripspace(tmp_path, "shared-canonical")
        monkeypatch.chdir(agent_ws)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(shared_ws))
        project_data_dir()
        _gripspace_cache.clear()

        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        assert describe_root_source() == f"marker:{shared_ws}"

    def test_source_is_walk_up_with_no_env_and_no_marker(self, tmp_path, monkeypatch):
        from synapt.recall.core import describe_root_source

        agent_ws = _make_named_gripspace(tmp_path, "agent-worktree")
        monkeypatch.chdir(agent_ws)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        assert describe_root_source() == "walk-up"

    def test_source_names_the_ignored_stale_marker(self, tmp_path, monkeypatch):
        """A stale marker is not silently indistinguishable from no marker
        at all -- the source string names what was ignored and why."""
        from synapt.recall.core import describe_root_source

        agent_ws = _make_named_gripspace(tmp_path, "agent-worktree")
        shared_ws = _make_named_gripspace(tmp_path, "shared-canonical")
        monkeypatch.chdir(agent_ws)
        monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(shared_ws))
        project_data_dir()
        _gripspace_cache.clear()
        shutil.rmtree(shared_ws / ".gitgrip")

        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        source = describe_root_source()
        assert source.startswith("walk-up")
        assert str(shared_ws) in source
        assert "stale" in source.lower()
