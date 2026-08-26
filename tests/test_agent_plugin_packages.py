"""Contract tests for the Codex and Claude Code plugin packages."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_COMMAND = "synapt recall hook session-start"
CHECKPOINT_COMMAND = "synapt recall checkpoint --event-json -"


def _hooks(runtime: str) -> dict:
    path = REPO_ROOT / f"{runtime}-plugin" / "hooks" / "hooks.json"
    return json.loads(path.read_text())


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _session_start_command(runtime: str) -> dict:
    hooks = _hooks(runtime)["hooks"]
    assert set(hooks) == {"SessionStart", "SessionEnd"}
    groups = hooks["SessionStart"]
    assert len(groups) == 1
    assert groups[0]["matcher"] == "startup|resume|clear|fork"
    commands = groups[0]["hooks"]
    assert len(commands) == 1
    return commands[0]


def _session_end_command(runtime: str) -> dict:
    groups = _hooks(runtime)["hooks"]["SessionEnd"]
    assert len(groups) == 1
    commands = groups[0]["hooks"]
    assert len(commands) == 1
    return commands[0]


def test_codex_plugin_loads_bounded_session_context() -> None:
    command = _session_start_command("codex")

    assert command["type"] == "command"
    assert command["command"] == HOOK_COMMAND
    assert command["timeout"] == 30
    assert command["additionalContextLimit"] == 3000


def test_claude_plugin_loads_bounded_session_context() -> None:
    command = _session_start_command("claude")

    assert command["type"] == "command"
    assert command["command"] == HOOK_COMMAND
    assert command["timeout"] == 30
    assert "additionalContextLimit" not in command


def test_plugins_capture_bounded_session_end_checkpoint() -> None:
    for runtime in ("codex", "claude"):
        command = _session_end_command(runtime)
        assert command["type"] == "command"
        assert command["command"] == CHECKPOINT_COMMAND
        assert command["timeout"] == 3


def test_claude_marketplace_distributes_repo_plugin() -> None:
    marketplace = _json(REPO_ROOT / ".claude-plugin" / "marketplace.json")

    assert marketplace["name"] == "synapt-plugins"
    plugins = marketplace["plugins"]
    assert len(plugins) == 1
    assert plugins[0]["name"] == "synapt-recall"
    assert plugins[0]["source"] == "./claude-plugin"


def test_claude_project_settings_enable_recall_marketplace() -> None:
    settings = _json(REPO_ROOT / "claude-plugin" / "project-settings.json")

    assert settings["enabledPlugins"] == {
        "synapt-recall@synapt-plugins": True,
    }
    marketplace = settings["extraKnownMarketplaces"]["synapt-plugins"]
    assert marketplace["source"] == {
        "source": "github",
        "repo": "synapt-dev/recall",
    }
    assert marketplace["autoUpdate"] is True
