"""Optional recall context for grep-shaped Claude Code tool calls.

The hook is advisory.  It must never delay or replace the tool result when
recall is disabled, unavailable, empty, or slower than its configured budget.
"""

from __future__ import annotations

import json
import queue
import re
import shlex
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


RecallQuick = Callable[[str], str]

_RECALL_BLOCK = re.compile(
    r"^--- \[(?:cluster:|knowledge #|\d{4}-\d{2}-\d{2} session )",
    re.MULTILINE,
)


@dataclass(frozen=True)
class GrepInterceptConfig:
    """Runtime controls for the opt-in grep interception hook."""

    enabled: bool = False
    timeout_ms: int = 150


def extract_grep_pattern(tool_call: Mapping[str, Any]) -> str | None:
    """Return the search pattern from a Grep tool or grep-shaped Bash call."""
    tool_name = tool_call.get("tool_name")
    tool_input = tool_call.get("tool_input")
    if not isinstance(tool_input, Mapping):
        return None

    if tool_name == "Grep":
        pattern = tool_input.get("pattern")
        return pattern if isinstance(pattern, str) and pattern else None

    if tool_name != "Bash":
        return None
    command = tool_input.get("command")
    if not isinstance(command, str):
        return None
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    if not words or words[0] not in {"grep", "rg"}:
        return None

    index = 1
    while index < len(words):
        word = words[index]
        if word == "--":
            index += 1
            break
        if not word.startswith("-") or word == "-":
            break
        index += 1
    if index >= len(words):
        return None
    return words[index]


def count_related_conversations(recall_output: str) -> int:
    """Count result blocks emitted by recall_quick, excluding absence prose."""
    return len(_RECALL_BLOCK.findall(recall_output))


def _default_recall_quick(query: str) -> str:
    from synapt.recall.server import recall_quick

    return recall_quick(query)


def _bounded_recall(
    query: str,
    *,
    timeout_ms: int,
    recall_quick: RecallQuick,
) -> str | None:
    """Run recall within a wall-clock budget without waiting for a late result."""
    result_queue: queue.Queue[str | Exception] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result_queue.put_nowait(recall_quick(query))
        except Exception as exc:
            result_queue.put_nowait(exc)

    worker = threading.Thread(target=invoke, daemon=True, name="grep-intercept-recall")
    worker.start()
    try:
        result = result_queue.get(timeout=max(0, timeout_ms) / 1000)
    except queue.Empty:
        return None
    if isinstance(result, Exception):
        return None
    return result


def build_pretooluse_context(
    tool_call: Mapping[str, Any],
    *,
    config: GrepInterceptConfig,
    recall_quick: RecallQuick = _default_recall_quick,
) -> str | None:
    """Build the advisory one-line context for a PreToolUse hook response."""
    if not config.enabled:
        return None
    pattern = extract_grep_pattern(tool_call)
    if pattern is None:
        return None
    recall_output = _bounded_recall(
        pattern,
        timeout_ms=config.timeout_ms,
        recall_quick=recall_quick,
    )
    if recall_output is None:
        return None
    count = count_related_conversations(recall_output)
    if count == 0:
        return None
    query = json.dumps(pattern, ensure_ascii=False)
    return f"recall: {count} related conversations (recall_search {query} for detail)"


def build_pretooluse_output(
    tool_call: Mapping[str, Any],
    *,
    config: GrepInterceptConfig,
    recall_quick: RecallQuick = _default_recall_quick,
) -> dict[str, Any] | None:
    """Return the Claude Code PreToolUse envelope, or no output on a miss."""
    context = build_pretooluse_context(
        tool_call,
        config=config,
        recall_quick=recall_quick,
    )
    if context is None:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": context,
        }
    }


def annotate_tool_result(
    tool_call: Mapping[str, Any],
    tool_result: str,
    *,
    config: GrepInterceptConfig,
    recall_quick: RecallQuick = _default_recall_quick,
) -> str:
    """Prepend advisory recall context while preserving the exact tool result."""
    context = build_pretooluse_context(
        tool_call,
        config=config,
        recall_quick=recall_quick,
    )
    return tool_result if context is None else f"{context}\n{tool_result}"


def claude_pretooluse_settings_snippet(
    *,
    enabled: bool,
    timeout_ms: int = 150,
) -> dict[str, Any]:
    """Return an opt-in Claude Code settings fragment for the hook."""
    if not enabled:
        return {"hooks": {"PreToolUse": []}}
    timeout_seconds = min(1.0, max(0.001, timeout_ms / 1000))
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash|Grep",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "synapt recall grep-intercept",
                            "timeout": timeout_seconds,
                        }
                    ],
                }
            ]
        }
    }
