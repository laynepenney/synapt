"""TDD contract for recall#836 grep-intercept hook."""

from __future__ import annotations

import importlib
import time
from typing import Any


def _grep_intercept():
    return importlib.import_module("synapt.integrations.grep_intercept")


def _bash(command: str) -> dict[str, Any]:
    return {"tool_name": "Bash", "input": {"command": command}}


def _grep_tool(pattern: str, *, path: str = ".") -> dict[str, Any]:
    return {"tool_name": "Grep", "input": {"pattern": pattern, "path": path}}


def test_extracts_patterns_from_grep_and_rg_shapes() -> None:
    mod = _grep_intercept()

    cases = [
        (_bash('rg -n "memory leak" src tests'), "memory leak"),
        (_bash("rg --fixed-strings 'SYNAPT_SHARED_CHANNELS_DIR' src"), "SYNAPT_SHARED_CHANNELS_DIR"),
        (_bash("grep -R --line-number 'modal_execution_requested' active-compression"), "modal_execution_requested"),
        (_bash("grep -- 'literal-leading-dash' README.md"), "literal-leading-dash"),
        (_grep_tool("recall_quick verified absence", path="src"), "recall_quick verified absence"),
    ]

    for tool_call, expected in cases:
        assert mod.extract_grep_pattern(tool_call) == expected


def test_non_grep_commands_are_ignored() -> None:
    mod = _grep_intercept()

    assert mod.extract_grep_pattern(_bash("git status --short")) is None
    assert mod.extract_grep_pattern(_bash("python -m pytest tests/recall -q")) is None
    assert mod.extract_grep_pattern({"tool_name": "Read", "input": {"file_path": "README.md"}}) is None


def test_annotation_format_for_positive_recall_hit() -> None:
    mod = _grep_intercept()
    config = mod.GrepInterceptConfig(enabled=True, timeout_ms=150)
    tool_result = "src/app.py:10: memory leak fixed here"

    def recall_quick(query: str) -> str:
        assert query == "memory leak"
        return "\n".join([
            "Past session context:",
            "Session: alpha",
            "Session: beta",
            "Session: gamma",
        ])

    annotated = mod.annotate_tool_result(
        _bash('rg "memory leak" src'),
        tool_result,
        config=config,
        recall_quick=recall_quick,
    )

    assert annotated.startswith(
        'recall: 3 related conversations (recall_search "memory leak" for detail)\n'
    )
    assert annotated.endswith(tool_result)


def test_miss_or_unavailable_recall_is_silent_noop() -> None:
    mod = _grep_intercept()
    config = mod.GrepInterceptConfig(enabled=True, timeout_ms=150)
    tool_result = "grep output remains untouched"

    miss = mod.annotate_tool_result(
        _bash('rg "unseen topic" .'),
        tool_result,
        config=config,
        recall_quick=lambda _query: "No prior discussion found for 'unseen topic'. Proceeding fresh is safe.",
    )
    assert miss == tool_result

    def unavailable(_query: str) -> str:
        raise RuntimeError("recall index unavailable")

    failure = mod.annotate_tool_result(
        _bash('rg "unseen topic" .'),
        tool_result,
        config=config,
        recall_quick=unavailable,
    )
    assert failure == tool_result


def test_timeout_never_blocks_the_original_grep_result() -> None:
    mod = _grep_intercept()
    config = mod.GrepInterceptConfig(enabled=True, timeout_ms=25)
    tool_result = "src/app.py:10: needle"

    def slow_recall(_query: str) -> str:
        time.sleep(0.30)
        return "Past session context:\nSession: too-late"

    started = time.perf_counter()
    annotated = mod.annotate_tool_result(
        _bash('rg "needle" src'),
        tool_result,
        config=config,
        recall_quick=slow_recall,
    )
    elapsed = time.perf_counter() - started

    assert annotated == tool_result
    assert elapsed < 0.150


def test_opt_in_config_disabled_never_calls_recall_quick() -> None:
    mod = _grep_intercept()
    config = mod.GrepInterceptConfig(enabled=False, timeout_ms=150)
    calls: list[str] = []

    def recall_quick(query: str) -> str:
        calls.append(query)
        return "Past session context:\nSession: should-not-run"

    result = mod.annotate_tool_result(
        _grep_tool("feature flag"),
        "src/config.py:1: feature flag",
        config=config,
        recall_quick=recall_quick,
    )

    assert result == "src/config.py:1: feature flag"
    assert calls == []


def test_claude_pretooluse_settings_snippet_is_opt_in_and_bounded() -> None:
    mod = _grep_intercept()

    snippet = mod.claude_pretooluse_settings_snippet(enabled=True, timeout_ms=150)

    assert "hooks" in snippet
    assert "PreToolUse" in snippet["hooks"]
    matcher = snippet["hooks"]["PreToolUse"][0]
    assert matcher["matcher"] == "Bash|Grep"
    hook = matcher["hooks"][0]
    assert hook["command"] == "synapt recall grep-intercept"
    assert hook["timeout"] <= 1
