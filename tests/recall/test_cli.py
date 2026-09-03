"""Tests for CLI: auto-discovery, hook installer, argument parsing, setup."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from synapt.recall.core import project_slug, project_index_dir, project_archive_dir, project_worktree_dir, project_transcript_dir
from synapt.recall.cli import (
    discover_transcript_dirs,
    _ensure_gitignore,
    _install_global_hooks,
    cmd_journal,
    cmd_rebuild,
    cmd_split,
    cmd_sync,
    main,
)


def _write_codex_transcript(path: Path, cwd: Path) -> None:
    """Write a minimal Codex-format transcript."""
    entries = [
        {
            "timestamp": "2026-03-01T10:00:00Z",
            "type": "session_meta",
            "payload": {"id": "codex-session-001", "cwd": str(cwd)},
        },
        {
            "timestamp": "2026-03-01T10:00:01Z",
            "type": "response_item",
            "payload": {
                "role": "user",
                "content": [{"type": "input_text", "text": "How does Codex indexing work?"}],
            },
        },
        {
            "timestamp": "2026-03-01T10:00:02Z",
            "type": "response_item",
            "payload": {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "It should index project-scoped Codex transcripts."}],
            },
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_discover_transcript_dirs_finds_dirs(tmp_path):
    """discover_transcript_dirs finds directories with .jsonl files."""
    projects = tmp_path / ".claude" / "projects"
    proj_a = projects / "-Users-test-proj-a"
    proj_a.mkdir(parents=True)
    (proj_a / "session1.jsonl").write_text("{}")

    proj_b = projects / "-Users-test-proj-b"
    proj_b.mkdir(parents=True)
    (proj_b / "session2.jsonl").write_text("{}")

    # Empty dir — should not be discovered
    proj_c = projects / "-Users-test-proj-c"
    proj_c.mkdir(parents=True)

    with patch("synapt.recall.cli.Path.home", return_value=tmp_path):
        dirs = discover_transcript_dirs()

    assert len(dirs) == 2
    assert proj_a in dirs
    assert proj_b in dirs


def test_discover_transcript_dirs_no_claude_dir(tmp_path):
    """Returns empty when ~/.claude/projects doesn't exist."""
    with patch("synapt.recall.cli.Path.home", return_value=tmp_path):
        dirs = discover_transcript_dirs()
    assert dirs == []


def test_install_global_hooks_creates_entries(tmp_path):
    """_install_global_hooks registers all hooks in ~/.claude/settings.json."""
    with patch("synapt.recall.cli.Path.home", return_value=tmp_path):
        installed = _install_global_hooks()

    assert installed == 3  # SessionStart, SessionEnd, PreCompact

    settings_path = tmp_path / ".claude" / "settings.json"
    assert settings_path.exists()
    settings = json.loads(settings_path.read_text())
    assert "SessionStart" in settings["hooks"]
    assert "SessionEnd" in settings["hooks"]
    assert "PreCompact" in settings["hooks"]

    session_start = settings["hooks"]["SessionStart"][0]
    assert session_start["matcher"] == "startup|resume|clear|fork"
    assert session_start["hooks"][0]["command"] == "synapt recall hook session-start"

    # Verify PreCompact uses the synapt recall hook command (not a shell script)
    matchers = settings["hooks"]["PreCompact"]
    inner_hooks = matchers[0]["hooks"]
    assert inner_hooks[0]["command"] == "synapt recall hook precompact"
    assert inner_hooks[0]["timeout"] == 300
    session_end = settings["hooks"]["SessionEnd"][0]["hooks"][0]
    assert session_end["command"] == "synapt recall checkpoint --event-json -"
    assert session_end["timeout"] == 3


def test_install_global_hooks_idempotent(tmp_path):
    """Running install twice doesn't duplicate the hook entries."""
    with patch("synapt.recall.cli.Path.home", return_value=tmp_path):
        _install_global_hooks()
        installed = _install_global_hooks()

    assert installed == 0  # all already registered

    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    for event in ("SessionStart", "SessionEnd", "PreCompact"):
        matchers = settings["hooks"][event]
        assert len(matchers) == 1
        assert len(matchers[0]["hooks"]) == 1


def test_install_global_hooks_preserves_existing_settings(tmp_path):
    """Existing settings.json content is preserved."""
    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.json").write_text(json.dumps({
        "permissions": {"allow": ["Bash(git:*)"]},
        "hooks": {"PostSave": [{"matcher": "", "hooks": [{"command": "echo done"}]}]}
    }))

    with patch("synapt.recall.cli.Path.home", return_value=tmp_path):
        _install_global_hooks()

    settings = json.loads((settings_dir / "settings.json").read_text())
    assert settings["permissions"]["allow"] == ["Bash(git:*)"]
    assert "PostSave" in settings["hooks"]
    assert "PreCompact" in settings["hooks"]


def test_install_global_hooks_replaces_unbounded_session_end(tmp_path):
    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.json").write_text(json.dumps({
        "hooks": {
            "SessionEnd": [{
                "matcher": "",
                "hooks": [{
                    "type": "command",
                    "command": "synapt recall hook session-end",
                    "timeout": 60,
                }],
            }],
        },
    }))

    with patch("synapt.recall.cli.Path.home", return_value=tmp_path):
        _install_global_hooks()

    hooks = json.loads((settings_dir / "settings.json").read_text())["hooks"]["SessionEnd"]
    commands = [hook["command"] for matcher in hooks for hook in matcher["hooks"]]
    assert commands == ["synapt recall checkpoint --event-json -"]


def test_install_global_hooks_moves_session_start_off_catch_all(tmp_path):
    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.json").write_text(json.dumps({
        "hooks": {
            "SessionStart": [{
                "matcher": "",
                "hooks": [{
                    "type": "command",
                    "command": "synapt recall hook session-start",
                    "timeout": 60,
                }],
            }],
        },
    }))

    with patch("synapt.recall.cli.Path.home", return_value=tmp_path):
        _install_global_hooks()

    matchers = json.loads(
        (settings_dir / "settings.json").read_text(),
    )["hooks"]["SessionStart"]
    commands = [
        (matcher["matcher"], hook["command"])
        for matcher in matchers
        for hook in matcher["hooks"]
    ]
    assert commands == [
        ("startup|resume|clear|fork", "synapt recall hook session-start"),
    ]


def test_install_global_hooks_persists_duplicate_catch_all_cleanup(tmp_path):
    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir(parents=True)
    command = {
        "type": "command",
        "command": "synapt recall hook session-start",
        "timeout": 60,
    }
    (settings_dir / "settings.json").write_text(json.dumps({
        "hooks": {
            "SessionStart": [
                {"matcher": "", "hooks": [command]},
                {
                    "matcher": "startup|resume|clear|fork",
                    "hooks": [command],
                },
            ],
            "SessionEnd": [{
                "matcher": "",
                "hooks": [{
                    "type": "command",
                    "command": "synapt recall checkpoint --event-json -",
                    "timeout": 3,
                }],
            }],
            "PreCompact": [{
                "matcher": "",
                "hooks": [{
                    "type": "command",
                    "command": "synapt recall hook precompact",
                    "timeout": 300,
                }],
            }],
        },
    }))

    with patch("synapt.recall.cli.Path.home", return_value=tmp_path):
        assert _install_global_hooks() == 0

    matchers = json.loads(
        (settings_dir / "settings.json").read_text(),
    )["hooks"]["SessionStart"]
    commands = [
        (matcher["matcher"], hook["command"])
        for matcher in matchers
        for hook in matcher["hooks"]
    ]
    assert commands == [
        ("startup|resume|clear|fork", "synapt recall hook session-start"),
    ]


# ---------------------------------------------------------------------------
# Tests: project slug utilities
# ---------------------------------------------------------------------------

def test_project_slug_converts_path(tmp_path):
    """project_slug converts absolute paths to Claude Code's slug format."""
    test_dir = tmp_path / "Development" / "synapse"
    test_dir.mkdir(parents=True)
    slug = project_slug(test_dir)
    expected = str(test_dir.resolve()).replace("\\", "/").replace("/", "-")
    assert slug == expected


def test_project_slug_defaults_to_cwd(tmp_path):
    """project_slug uses cwd when no argument given."""
    with patch("synapt.recall.core.Path.cwd", return_value=tmp_path):
        slug = project_slug()
    assert slug == str(tmp_path).replace("\\", "/").replace("/", "-")


def test_project_index_dir_returns_in_project_path(tmp_path):
    """project_index_dir returns <project>/.synapt/recall/index/."""
    project = tmp_path / "myproject"
    project.mkdir()
    idx_dir = project_index_dir(project)
    assert idx_dir == project / ".synapt" / "recall" / "index"


def test_project_archive_dir_returns_in_project_path(tmp_path):
    """project_archive_dir returns <project>/.synapt/recall/worktrees/<name>/transcripts/."""
    project = tmp_path / "myproject"
    project.mkdir()
    archive_dir = project_archive_dir(project)
    assert archive_dir == project / ".synapt" / "recall" / "worktrees" / project.name / "transcripts"


def test_project_transcript_dir_finds_transcripts(tmp_path):
    """project_transcript_dir finds the matching Claude Code project dir."""
    myproject = tmp_path / "myproject"
    myproject.mkdir()
    slug = str(myproject.resolve()).replace("\\", "/").replace("/", "-")
    proj_dir = tmp_path / ".claude" / "projects" / slug
    proj_dir.mkdir(parents=True)
    (proj_dir / "session.jsonl").write_text("{}")

    with patch("synapt.recall.core.Path.home", return_value=tmp_path):
        result = project_transcript_dir(myproject)
    assert result == proj_dir


def test_project_transcript_dir_returns_none_when_missing(tmp_path):
    """project_transcript_dir returns None when no transcripts exist."""
    nonexistent = tmp_path / "nonexistent"
    nonexistent.mkdir()
    with patch("synapt.recall.core.Path.home", return_value=tmp_path):
        result = project_transcript_dir(nonexistent)
    assert result is None


# ---------------------------------------------------------------------------
# Tests: setup command
# ---------------------------------------------------------------------------

def test_cmd_setup_orchestrates_all_steps(tmp_path):
    """setup archives transcripts, builds index, registers MCP, and installs hook."""
    from conftest import user_text_entry, assistant_entry, write_jsonl

    # Create a fake project with transcripts
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    slug = str(project_dir).replace("\\", "/").replace("/", "-")

    transcript_dir = tmp_path / ".claude" / "projects" / slug
    transcript_dir.mkdir(parents=True)

    # Write a valid transcript using conftest helpers
    entries = [
        user_text_entry("What is the quality curve?", uuid="u1"),
        assistant_entry(text="The quality curve weights Cat3 examples.", uuid="a1"),
    ]
    write_jsonl(transcript_dir / "session.jsonl", entries)

    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

    with patch.dict("os.environ", {"CODEX_HOME": str(tmp_path / ".codex")}), \
         patch("synapt.recall.core.Path.home", return_value=tmp_path), \
         patch("synapt.recall.core.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.cli.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.cli.subprocess.run", mock_run), \
         patch("synapt.recall.cli.shutil.which", return_value="/usr/bin/claude"):
        from synapt.recall.cli import cmd_setup
        import argparse
        args = argparse.Namespace(
            no_embeddings=True,
            no_hook=False,
            global_scope=False,
            sync=None,
        )
        cmd_setup(args)

    # Verify index was built (now in-project, SQLite)
    index_dir = project_index_dir(project_dir)
    assert (index_dir / "recall.db").exists()

    # Verify transcripts were archived
    archive_dir = project_archive_dir(project_dir)
    assert archive_dir.exists()
    assert (archive_dir / "session.jsonl").exists()

    # Verify MCP registration was called (filter out git worktree discovery calls)
    mcp_calls = [c for c in mock_run.call_args_list if "claude" in c[0][0]]
    assert len(mcp_calls) == 1
    call_args = mcp_calls[0][0][0]
    assert "claude" in call_args
    assert "mcp" in call_args
    assert "synapt" in call_args

    # Verify global hooks were registered
    global_settings = tmp_path / ".claude" / "settings.json"
    assert global_settings.exists()
    import json
    settings = json.loads(global_settings.read_text())
    hook_cmds = []
    for event_hooks in settings.get("hooks", {}).values():
        for matcher in event_hooks:
            for h in matcher.get("hooks", []):
                hook_cmds.append(h.get("command", ""))
    assert "synapt recall hook session-start" in hook_cmds
    assert "synapt recall checkpoint --event-json -" in hook_cmds
    assert "synapt recall hook precompact" in hook_cmds

    # Verify .gitignore was updated
    gitignore = project_dir / ".gitignore"
    assert gitignore.exists()
    assert ".synapt/" in gitignore.read_text()

    # Verify Codex skill was deployed
    skill_path = tmp_path / ".codex" / "skills" / "dev-loop" / "SKILL.md"
    assert skill_path.exists()
    assert "Do not use this skill as a passive channel watcher." in skill_path.read_text()


def test_cmd_setup_with_codex_only_transcripts(tmp_path):
    """setup succeeds for a Codex-only project with no Claude transcripts."""
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()

    codex_sessions = tmp_path / ".codex" / "sessions" / "2026" / "03" / "01"
    _write_codex_transcript(codex_sessions / "rollout-test.jsonl", project_dir)

    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

    with patch.dict("os.environ", {"CODEX_HOME": str(tmp_path / ".codex")}), \
         patch("synapt.recall.core.Path.home", return_value=tmp_path), \
         patch("synapt.recall.cli.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.journal.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.codex.Path.home", return_value=tmp_path), \
         patch("synapt.recall.cli.subprocess.run", mock_run), \
         patch("synapt.recall.cli.shutil.which", return_value="/usr/bin/claude"):
        from synapt.recall.cli import cmd_setup
        args = argparse.Namespace(
            no_embeddings=True,
            no_hook=False,
            global_scope=False,
            sync=None,
        )
        cmd_setup(args)

    index_dir = project_index_dir(project_dir)
    assert (index_dir / "recall.db").exists()
    archive_dir = project_archive_dir(project_dir)
    assert (archive_dir / "rollout-test.jsonl").exists()
    assert (tmp_path / ".codex" / "skills" / "dev-loop" / "SKILL.md").exists()


def test_cmd_setup_no_transcripts_still_registers_mcp(tmp_path):
    """setup with zero transcripts skips indexing but still registers MCP, hooks, gitignore."""
    project_dir = tmp_path / "freshproject"
    project_dir.mkdir()

    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

    with patch.dict("os.environ", {"CODEX_HOME": str(tmp_path / ".codex")}), \
         patch("synapt.recall.core.Path.home", return_value=tmp_path), \
         patch("synapt.recall.core.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.cli.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.cli.subprocess.run", mock_run), \
         patch("synapt.recall.cli.shutil.which", return_value="/usr/bin/claude"):
        from synapt.recall.cli import cmd_setup
        args = argparse.Namespace(
            no_embeddings=True,
            no_hook=False,
            global_scope=False,
            sync=None,
        )
        cmd_setup(args)

    # No index should exist (nothing to index)
    index_dir = project_index_dir(project_dir)
    assert not (index_dir / "recall.db").exists()

    # MCP registration was still called
    mcp_calls = [c for c in mock_run.call_args_list if "claude" in c[0][0]]
    assert len(mcp_calls) == 1
    call_args = mcp_calls[0][0][0]
    assert "mcp" in call_args

    # Global hooks were still registered
    global_settings = tmp_path / ".claude" / "settings.json"
    assert global_settings.exists()
    settings = json.loads(global_settings.read_text())
    hook_cmds = []
    for event_hooks in settings.get("hooks", {}).values():
        for matcher in event_hooks:
            for h in matcher.get("hooks", []):
                hook_cmds.append(h.get("command", ""))
    assert "synapt recall hook session-start" in hook_cmds

    # .gitignore was still created
    gitignore = project_dir / ".gitignore"
    assert gitignore.exists()
    assert ".synapt/" in gitignore.read_text()

    # Codex skill was still deployed
    skill_path = tmp_path / ".codex" / "skills" / "dev-loop" / "SKILL.md"
    assert skill_path.exists()


def test_cmd_setup_narrates_every_announced_step(tmp_path, capsys):
    """The banner announces total_steps=5 (hooks enabled); every step 1..5 must be narrated.

    The .gitignore step ran silently -- the banner said 5 steps, only 4
    ever printed "[setup] Step N/5: ...", so a user counting along saw
    the completion summary appear right after "4/5" with no "5/5" anywhere.
    """
    project_dir = tmp_path / "freshproject"
    project_dir.mkdir()

    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

    with patch.dict("os.environ", {"CODEX_HOME": str(tmp_path / ".codex")}), \
         patch("synapt.recall.core.Path.home", return_value=tmp_path), \
         patch("synapt.recall.core.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.cli.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.cli.subprocess.run", mock_run), \
         patch("synapt.recall.cli.shutil.which", return_value="/usr/bin/claude"):
        from synapt.recall.cli import cmd_setup
        args = argparse.Namespace(
            no_embeddings=True,
            no_hook=False,
            global_scope=False,
            sync=None,
        )
        cmd_setup(args)

    out = capsys.readouterr().out
    for n in range(1, 6):
        assert f"Step {n}/5:" in out, f"missing narration for step {n}/5 in setup output"


def test_ensure_gitignore_creates_file(tmp_path):
    """_ensure_gitignore creates .gitignore with .synapt/ entry if missing."""
    _ensure_gitignore(tmp_path)
    content = (tmp_path / ".gitignore").read_text()
    assert ".synapt/" in content


def test_ensure_gitignore_appends_to_existing(tmp_path):
    """_ensure_gitignore appends .synapt/ to existing .gitignore."""
    (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")
    _ensure_gitignore(tmp_path)
    content = (tmp_path / ".gitignore").read_text()
    assert "*.pyc" in content
    assert ".synapt/" in content


def test_ensure_gitignore_idempotent(tmp_path):
    """_ensure_gitignore doesn't duplicate the .synapt/ entry."""
    (tmp_path / ".gitignore").write_text(".synapt/\n")
    _ensure_gitignore(tmp_path)
    content = (tmp_path / ".gitignore").read_text()
    assert content.count(".synapt/") == 1


def test_ensure_gitignore_migrates_old_entry(tmp_path):
    """_ensure_gitignore replaces .synapse-recall/ with .synapt/ in existing .gitignore."""
    (tmp_path / ".gitignore").write_text("*.pyc\n.synapse-recall/\n__pycache__/\n")
    _ensure_gitignore(tmp_path)
    content = (tmp_path / ".gitignore").read_text()
    assert ".synapt/" in content
    assert ".synapse-recall/" not in content
    assert "*.pyc" in content


def test_ensure_gitignore_migrates_synapse_entry(tmp_path):
    """_ensure_gitignore replaces .synapse/ with .synapt/ in existing .gitignore."""
    (tmp_path / ".gitignore").write_text("*.pyc\n.synapse/\n__pycache__/\n")
    _ensure_gitignore(tmp_path)
    content = (tmp_path / ".gitignore").read_text()
    assert ".synapt/" in content
    assert ".synapse/" not in content
    assert "*.pyc" in content


# ---------------------------------------------------------------------------
# Tests: cmd_rebuild
# ---------------------------------------------------------------------------

def test_cmd_rebuild_archives_and_builds(tmp_path):
    """cmd_rebuild archives transcripts and builds an index."""
    from conftest import user_text_entry, assistant_entry, write_jsonl

    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    slug = str(project_dir).replace("\\", "/").replace("/", "-")

    transcript_dir = tmp_path / ".claude" / "projects" / slug
    transcript_dir.mkdir(parents=True)

    entries = [
        user_text_entry("How does archiving work?", uuid="u1"),
        assistant_entry(text="Transcripts are copied to .synapse-recall/transcripts/.", uuid="a1"),
    ]
    write_jsonl(transcript_dir / "session.jsonl", entries)

    with patch("synapt.recall.core.Path.home", return_value=tmp_path), \
         patch("synapt.recall.core.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.cli.Path.cwd", return_value=project_dir):
        args = argparse.Namespace(out=None, sync=False)
        cmd_rebuild(args)

    # Index built in-project (SQLite)
    index_dir = project_index_dir(project_dir)
    assert (index_dir / "recall.db").exists()

    # Transcripts archived
    archive_dir = project_archive_dir(project_dir)
    assert (archive_dir / "session.jsonl").exists()


def test_cmd_rebuild_with_sync_skips_when_no_config(tmp_path):
    """cmd_rebuild --sync does nothing when no sync config exists."""
    from conftest import user_text_entry, assistant_entry, write_jsonl

    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    slug = str(project_dir).replace("\\", "/").replace("/", "-")

    transcript_dir = tmp_path / ".claude" / "projects" / slug
    transcript_dir.mkdir(parents=True)

    entries = [
        user_text_entry("Test sync skip", uuid="u1"),
        assistant_entry(text="No config means no sync.", uuid="a1"),
    ]
    write_jsonl(transcript_dir / "session.jsonl", entries)

    with patch("synapt.recall.core.Path.home", return_value=tmp_path), \
         patch("synapt.recall.core.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.cli.Path.cwd", return_value=project_dir):
        args = argparse.Namespace(out=None, sync=True)
        # Should not raise — gracefully skips sync when no config
        cmd_rebuild(args)

    index_dir = project_index_dir(project_dir)
    assert (index_dir / "recall.db").exists()


# ---------------------------------------------------------------------------
# Tests: cmd_sync
# ---------------------------------------------------------------------------

def test_cmd_sync_errors_without_config(tmp_path, capsys):
    """cmd_sync exits with error when no sync target is configured."""
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()

    with patch("synapt.recall.cli.Path.cwd", return_value=project_dir):
        args = argparse.Namespace(direction="both", repo=None)
        try:
            cmd_sync(args)
            assert False, "Should have called sys.exit"
        except SystemExit as e:
            assert e.code == 1

    captured = capsys.readouterr()
    assert "no sync target" in captured.err


def test_cmd_split_prints_experimental_warning(tmp_path, capsys):
    """split warns that sharded mode is still experimental."""
    with patch("synapt.recall.cli.project_index_dir", return_value=tmp_path), \
         patch("synapt.recall.sharding.split_monolithic_db", return_value={
             "index.db": 0,
             "data_001.db": 3,
         }):
        cmd_split(argparse.Namespace(dry_run=False))

    captured = capsys.readouterr()
    assert "experimental" in captured.err.lower()
    assert "Split complete: 3 chunks across 1 quarterly shard(s)" in captured.out


# ---------------------------------------------------------------------------
# Tests: recall_setup MCP tool
# ---------------------------------------------------------------------------

def test_recall_setup_mcp_tool(tmp_path):
    """recall_setup builds index, installs hook, and updates gitignore."""
    from conftest import user_text_entry, assistant_entry, write_jsonl
    from synapt.recall.server import recall_setup

    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    slug = str(project_dir).replace("\\", "/").replace("/", "-")

    transcript_dir = tmp_path / ".claude" / "projects" / slug
    transcript_dir.mkdir(parents=True)

    entries = [
        user_text_entry("How does recall_setup work?", uuid="u1"),
        assistant_entry(text="It delegates to shared helpers.", uuid="a1"),
    ]
    write_jsonl(transcript_dir / "session.jsonl", entries)

    with patch("synapt.recall.core.Path.home", return_value=tmp_path), \
         patch("synapt.recall.core.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.cli.Path.cwd", return_value=project_dir):
        result = recall_setup(no_hook=False)

    assert "Setup complete" in result
    assert "chunks from" in result
    assert "ensured in .gitignore" in result
    assert "global hook" in result.lower()
    assert "Index size:" in result

    # Index built (SQLite)
    index_dir = project_index_dir(project_dir)
    assert (index_dir / "recall.db").exists()

    # Transcripts archived
    archive_dir = project_archive_dir(project_dir)
    assert (archive_dir / "session.jsonl").exists()

    # Global hooks installed (in ~/.claude/settings.json, not project-level)
    assert "hook" in result.lower()

    # Gitignore updated
    gitignore = project_dir / ".gitignore"
    assert ".synapt/" in gitignore.read_text()


def test_recall_setup_no_hook(tmp_path):
    """recall_setup with no_hook=True skips hook installation."""
    from conftest import user_text_entry, assistant_entry, write_jsonl
    from synapt.recall.server import recall_setup

    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    slug = str(project_dir).replace("\\", "/").replace("/", "-")

    transcript_dir = tmp_path / ".claude" / "projects" / slug
    transcript_dir.mkdir(parents=True)

    entries = [
        user_text_entry("Test no-hook mode", uuid="u1"),
        assistant_entry(text="Hook should be skipped.", uuid="a1"),
    ]
    write_jsonl(transcript_dir / "session.jsonl", entries)

    with patch("synapt.recall.core.Path.home", return_value=tmp_path), \
         patch("synapt.recall.core.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.cli.Path.cwd", return_value=project_dir):
        result = recall_setup(no_hook=True)

    assert "Setup complete" in result
    assert "skipped" in result.lower()


def test_recall_setup_no_transcripts(tmp_path):
    """recall_setup still completes hooks and gitignore when no transcripts exist."""
    from synapt.recall.server import recall_setup

    project_dir = tmp_path / "myproject"
    project_dir.mkdir()

    with patch("synapt.recall.core.Path.home", return_value=tmp_path), \
         patch("synapt.recall.core.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.cli.Path.cwd", return_value=project_dir):
        result = recall_setup()

    assert "Setup complete" in result
    assert "No transcripts found yet" in result
    assert ".gitignore" in result

    global_settings = tmp_path / ".claude" / "settings.json"
    assert global_settings.exists()

    gitignore = project_dir / ".gitignore"
    assert gitignore.exists()
    assert ".synapt/" in gitignore.read_text()


def test_recall_setup_mcp_tool_with_codex_only_transcripts(tmp_path):
    """recall_setup succeeds for a Codex-only project."""
    from synapt.recall.server import recall_setup

    project_dir = tmp_path / "myproject"
    project_dir.mkdir()

    codex_sessions = tmp_path / ".codex" / "sessions" / "2026" / "03" / "01"
    _write_codex_transcript(codex_sessions / "rollout-test.jsonl", project_dir)

    with patch("synapt.recall.core.Path.home", return_value=tmp_path), \
         patch("synapt.recall.cli.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.server.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.journal.Path.cwd", return_value=project_dir), \
         patch("synapt.recall.codex.Path.home", return_value=tmp_path):
        result = recall_setup(no_hook=True)

    assert "Setup complete" in result
    assert "chunks from" in result
    archive_dir = project_archive_dir(project_dir)
    assert (archive_dir / "rollout-test.jsonl").exists()


def test_main_dispatches_export_command(tmp_path):
    """CLI main dispatches the export command to cmd_export."""
    with patch.object(sys, "argv", ["synapt", "export", "backup.synapt-archive"]), \
         patch("synapt.recall.cli.cmd_export") as mock_export:
        main()
    mock_export.assert_called_once()
    assert mock_export.call_args[0][0].output == "backup.synapt-archive"


def test_main_dispatches_import_command(tmp_path):
    """CLI main dispatches the import command to cmd_import."""
    with patch.object(sys, "argv", ["synapt", "import", "backup.synapt-archive", "--merge"]), \
         patch("synapt.recall.cli.cmd_import") as mock_import:
        main()
    mock_import.assert_called_once()
    args = mock_import.call_args[0][0]
    assert args.archive == "backup.synapt-archive"
    assert args.merge is True


def test_recall_export_tool_formats_success(tmp_path):
    """recall_export MCP tool returns a concise summary."""
    from synapt.recall.server import recall_export

    archive_path = tmp_path / "project.synapt-archive"
    with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
         patch("synapt.recall.archive.export_recall_archive",
               return_value=(archive_path, {"chunk_count": 3, "knowledge_count": 2, "worktree_count": 1})):
        result = recall_export()

    assert str(archive_path) in result
    assert "chunks: 3" in result
    assert "knowledge: 2" in result


def test_recall_import_tool_formats_success(tmp_path):
    """recall_import MCP tool returns a concise summary."""
    from synapt.recall.server import recall_import

    archive_path = tmp_path / "project.synapt-archive"
    with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
         patch("synapt.recall.archive.import_recall_archive",
               return_value={"mode": "merge", "chunk_count": 5, "knowledge_count": 4}):
        result = recall_import(str(archive_path), mode="merge")

    assert str(archive_path) in result
    assert "mode: merge" in result
    assert "chunks: 5" in result


# ---------------------------------------------------------------------------
# Tests: _extract_session_id and _catchup_archive_and_journal
# ---------------------------------------------------------------------------


def test_extract_session_id(tmp_path):
    """extract_session_id reads session ID from transcript progress events."""
    from synapt.recall.journal import extract_session_id

    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        '{"type": "user", "message": {"content": "hello"}}\n'
        '{"type": "progress", "sessionId": "abc-123-def"}\n'
        '{"type": "assistant", "message": {"content": "hi"}}\n'
    )
    assert extract_session_id(transcript) == "abc-123-def"


def test_extract_session_id_missing(tmp_path):
    """extract_session_id returns empty string when no progress event."""
    from synapt.recall.journal import extract_session_id

    transcript = tmp_path / "session.jsonl"
    transcript.write_text('{"type": "user"}\n')
    assert extract_session_id(transcript) == ""


def test_catchup_writes_journal_for_unjournaled_session(tmp_path):
    """_catchup_archive_and_journal writes journal entry for un-journaled sessions."""
    from synapt.recall.cli import _catchup_archive_and_journal
    from synapt.recall.journal import _journal_path, read_entries

    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    # Create a transcript with a session ID and a file modification
    transcript = source / "session1.jsonl"
    import json as _json
    file_path = str(project / "src/main.py")
    assistant_line = _json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Edit", "input": {"file_path": file_path}}
    ]}})
    transcript.write_text(
        '{"type": "progress", "sessionId": "test-session-001"}\n'
        '{"type": "user", "message": {"content": "fix the bug"}}\n'
        + assistant_line + '\n'
    )

    with patch("synapt.recall.core.Path.cwd", return_value=project):
        _catchup_archive_and_journal(project, source)

    # Should have archived the transcript
    archive_dir = project_archive_dir(project)
    assert (archive_dir / "session1.jsonl").exists()

    # Should have written a journal entry
    journal_path = project_worktree_dir(project) / "journal.jsonl"
    assert journal_path.exists()
    entries = read_entries(journal_path, n=1)
    assert len(entries) == 1
    assert entries[0].session_id == "test-session-001"


def test_catchup_skips_already_journaled_session(tmp_path):
    """_catchup_archive_and_journal skips sessions that already have journal entries."""
    from synapt.recall.cli import _catchup_archive_and_journal
    from synapt.recall.journal import JournalEntry, append_entry, _journal_path, read_entries

    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    # Create transcript
    transcript = source / "session1.jsonl"
    transcript.write_text(
        '{"type": "progress", "sessionId": "already-done-001"}\n'
        '{"type": "user", "message": {"content": "hello"}}\n'
    )

    # Pre-populate journal with an entry for this session
    journal_path = project_worktree_dir(project) / "journal.jsonl"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    existing = JournalEntry(
        timestamp="2026-03-03T10:00:00+00:00",
        session_id="already-done-001",
        focus="Previous work",
    )
    append_entry(existing, journal_path)

    with patch("synapt.recall.core.Path.cwd", return_value=project):
        _catchup_archive_and_journal(project, source)

    # Should still have only 1 journal entry (no duplicate)
    entries = read_entries(journal_path, n=10)
    assert len(entries) == 1
    assert entries[0].focus == "Previous work"


def test_catchup_writes_journal_for_codex_session(tmp_path):
    """_catchup_archive_and_journal handles Codex transcripts from the archive source."""
    from synapt.recall.cli import _catchup_archive_and_journal
    from synapt.recall.journal import read_entries

    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    transcript = source / "rollout-test.jsonl"
    local_file = project / "src" / "main.py"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "timestamp": "2026-03-01T10:00:00Z",
            "type": "session_meta",
            "payload": {"id": "codex-session-001", "cwd": str(project)},
        },
        {
            "timestamp": "2026-03-01T10:00:01Z",
            "type": "response_item",
            "payload": {
                "role": "user",
                "content": [{"type": "input_text", "text": f"inspect {local_file}"}],
            },
        },
        {
            "timestamp": "2026-03-01T10:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": f"sed -n 1,20p {local_file}"}),
                "call_id": "call_1",
            },
        },
    ]
    with open(transcript, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    with patch("synapt.recall.core.Path.cwd", return_value=project):
        _catchup_archive_and_journal(project, source)

    archive_dir = project_archive_dir(project)
    assert (archive_dir / "rollout-test.jsonl").exists()

    journal_path = project_worktree_dir(project) / "journal.jsonl"
    entries = read_entries(journal_path, n=1)
    assert len(entries) == 1
    assert entries[0].session_id == "codex-session-001"


def test_cmd_journal_write_carries_forward_unresolved_next_steps(tmp_path, capsys):
    """Journal write merges prior unresolved next_steps into the new entry."""
    from synapt.recall.journal import JournalEntry, append_entry, read_latest

    project = tmp_path / "project"
    project.mkdir()
    journal_path = project_worktree_dir(project) / "journal.jsonl"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    append_entry(
        JournalEntry(
            timestamp="2026-03-31T10:00:00+00:00",
            session_id="prior-session",
            focus="previous work",
            next_steps=["ship docs", "follow up with team"],
        ),
        journal_path,
    )

    args = argparse.Namespace(
        read=False,
        list=False,
        show=None,
        write=True,
        focus="current work",
        done="ship docs",
        decisions=None,
        next="close loop",
    )

    with patch("synapt.recall.cli.Path.cwd", return_value=project), \
         patch("synapt.recall.journal.latest_transcript_path", return_value=None), \
         patch("synapt.recall.journal.auto_extract_entry", return_value=JournalEntry(
             timestamp="2026-04-01T12:00:00+00:00",
             session_id="current-session",
         )):
        cmd_journal(args)

    latest = read_latest(journal_path)
    assert latest is not None
    # recall#984: the carried step carries the date of the entry it came from
    assert latest.next_steps[0] == "close loop"
    assert len(latest.next_steps) == 2
    assert latest.next_steps[1].startswith("follow up with team [carried since ")

    captured = capsys.readouterr()
    assert "Journal entry written" in captured.err


# ---------------------------------------------------------------------------
# Channel CLI flag tests
# ---------------------------------------------------------------------------


def test_channel_cli_dispatches_read_with_defaults():
    """CLI dispatches channel read with default args."""
    with patch.object(sys, "argv", ["synapt", "channel", "read"]), \
         patch("synapt.recall.cli.cmd_channel") as mock:
        main()
    mock.assert_called_once()
    args = mock.call_args[0][0]
    assert args.action == "read"
    assert args.channel == "dev"
    assert args.limit == 20
    assert args.detail is None


def test_channel_cli_read_with_detail_flag():
    """CLI passes --detail to channel read."""
    with patch.object(sys, "argv", ["synapt", "channel", "read", "dev", "--detail", "low"]), \
         patch("synapt.recall.cli.cmd_channel") as mock:
        main()
    args = mock.call_args[0][0]
    assert args.action == "read"
    assert args.detail == "low"


def test_channel_cli_join_with_name_flag():
    """CLI passes --name to channel join."""
    with patch.object(sys, "argv", ["synapt", "channel", "join", "dev", "--name", "Opus"]), \
         patch("synapt.recall.cli.cmd_channel") as mock:
        main()
    args = mock.call_args[0][0]
    assert args.action == "join"
    assert args.name == "Opus"


def test_channel_cli_join_name_wired_to_display_name(tmp_path):
    """--name on join passes through as display_name kwarg."""
    with patch.object(sys, "argv", ["synapt", "channel", "join", "dev", "--name", "TestAgent"]), \
         patch("synapt.recall.channel.channel_join", return_value="Joined #dev as TestAgent") as mock_join:
        from synapt.recall.cli import main
        main()
    mock_join.assert_called_once()
    assert mock_join.call_args[1].get("display_name") == "TestAgent"


def test_channel_cli_read_detail_wired_to_channel_read(tmp_path):
    """--detail on read passes through to channel_read."""
    with patch.object(sys, "argv", ["synapt", "channel", "read", "dev", "--detail", "min"]), \
         patch("synapt.recall.channel.channel_read", return_value="## #dev") as mock_read:
        from synapt.recall.cli import main
        main()
    mock_read.assert_called_once()
    assert mock_read.call_args[1].get("detail") == "min"


def test_channel_cli_post_without_name():
    """CLI dispatches channel post without --name (no display_name kwarg)."""
    with patch.object(sys, "argv", ["synapt", "channel", "post", "dev", "hello"]), \
         patch("synapt.recall.cli.cmd_channel") as mock:
        main()
    args = mock.call_args[0][0]
    assert args.action == "post"
    assert args.message == "hello"
    assert args.name is None


# ---------------------------------------------------------------------------
# migrate-channels CLI tests
# ---------------------------------------------------------------------------


def test_migrate_channels_cli_dispatches_correctly():
    """CLI dispatches 'migrate' subcommand to cmd_migrate_channels."""
    with patch.object(sys, "argv", ["synapt", "migrate"]), \
         patch("synapt.recall.cli.cmd_migrate_channels") as mock:
        main()
    mock.assert_called_once()


def test_migrate_channels_cli_flags():
    """CLI passes --org, --project, --project-dir to cmd_migrate_channels."""
    with patch.object(
        sys, "argv",
        ["synapt", "migrate", "--org", "myorg", "--project", "myproject", "--project-dir", "/tmp/foo"],
    ), patch("synapt.recall.cli.cmd_migrate_channels") as mock:
        main()
    args = mock.call_args[0][0]
    assert args.org == "myorg"
    assert args.project == "myproject"
    assert args.project_dir == "/tmp/foo"


def test_cmd_migrate_channels_calls_migrate_function(tmp_path):
    """cmd_migrate_channels calls migrate_channels_to_global with resolved paths."""
    import argparse
    from synapt.recall.cli import cmd_migrate_channels

    # Write a channel file to the local channels dir
    local_channels = tmp_path / ".synapt" / "recall" / "channels"
    local_channels.mkdir(parents=True)
    (local_channels / "dev.jsonl").write_text(
        '{"timestamp": "2026-01-01T00:00:00", "from_agent": "apollo", "body": "hi"}\n'
    )

    args = argparse.Namespace(
        project_dir=str(tmp_path),
        org="test-org",
        project="test-project",
    )

    global_dir = tmp_path / "global"
    with patch("synapt.recall.cli._get_global_channels_dir", return_value=global_dir):
        cmd_migrate_channels(args)

    migrated = global_dir / "test-org" / "test-project" / "dev.jsonl"
    assert migrated.exists(), "Channel file should be in global store after migration"
    msgs = [json.loads(line) for line in migrated.read_text().splitlines()]
    assert msgs[0]["from_agent"] == "apollo"


def test_cmd_migrate_channels_missing_org_exits(tmp_path):
    """cmd_migrate_channels exits with error if org cannot be resolved."""
    import argparse
    from synapt.recall.cli import cmd_migrate_channels

    args = argparse.Namespace(project_dir=str(tmp_path), org=None, project=None)
    with patch("synapt.recall.cli._resolve_org_id_for_cli", return_value=None), \
         pytest.raises(SystemExit):
        cmd_migrate_channels(args)
