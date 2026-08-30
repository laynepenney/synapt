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
    r"^--- \[(?:cluster:|knowledge #|\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})? session )",
    re.MULTILINE,
)

_FLAG_OPTIONS = {
    "grep": {
        "--basic-regexp", "--extended-regexp", "--fixed-strings", "--ignore-case",
        "--invert-match", "--line-number", "--no-filename", "--only-matching",
        "--quiet", "--recursive", "--with-filename", "--word-regexp",
        "-E", "-F", "-H", "-I", "-L", "-R", "-c", "-h", "-i", "-l",
        "-n", "-o", "-q", "-r", "-s", "-v", "-w", "-x",
    },
    "rg": {
        "--case-sensitive", "--column", "--count", "--files-with-matches",
        "--fixed-strings", "--hidden", "--ignore-case", "--invert-match",
        "--line-number", "--no-heading", "--no-ignore", "--only-matching",
        "--smart-case", "--stats", "--text", "--trim", "--type-list",
        "--unrestricted", "--word-regexp", "-F", "-H", "-I", "-L", "-N",
        "-S", "-U", "-c", "-h", "-i", "-l", "-n", "-o", "-p", "-s",
        "-u", "-v", "-w", "-x",
    },
}

_VALUE_OPTIONS = {
    "grep": {
        "--after-context", "--before-context", "--binary-files", "--context",
        "--devices", "--directories", "--exclude", "--exclude-dir",
        "--exclude-from", "--include", "--label", "--max-count",
        "-A", "-B", "-C", "-m",
    },
    "rg": {
        "--after-context", "--before-context", "--context", "--encoding",
        "--engine", "--glob", "--iglob", "--max-columns",
        "--max-count", "--max-depth", "--max-filesize", "--path-separator",
        "--pre", "--pre-glob", "--regex-size-limit", "--replace", "--sort",
        "--sortr", "--type", "--type-add", "--type-not", "-A", "-B", "-C",
        "-E", "-M", "-T", "-g", "-j", "-m", "-r", "-t",
    },
}

_PATTERN_OPTIONS = {
    "grep": {"--regexp", "-e"},
    "rg": {"--regexp", "-e"},
}

_PATTERN_FILE_OPTIONS = {
    "grep": {"--file", "-f"},
    "rg": {"--file", "-f"},
}


@dataclass(frozen=True)
class GrepInterceptConfig:
    """Runtime controls for the opt-in grep interception hook."""

    enabled: bool = False
    timeout_ms: int = 500


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

    command_name = words[0]
    flag_options = _FLAG_OPTIONS[command_name]
    value_options = _VALUE_OPTIONS[command_name]
    pattern_options = _PATTERN_OPTIONS[command_name]
    pattern_file_options = _PATTERN_FILE_OPTIONS[command_name]
    short_flags = {option[1:] for option in flag_options if re.fullmatch(r"-[A-Za-z]", option)}

    index = 1
    pattern_file_seen = False
    while index < len(words):
        word = words[index]
        if word == "--":
            index += 1
            break
        if not word.startswith("-") or word == "-":
            break

        option, separator, inline_value = word.partition("=")
        if option in pattern_options:
            if separator:
                return inline_value or None
            index += 1
            return words[index] if index < len(words) and words[index] else None
        if option in pattern_file_options:
            pattern_file_seen = True
            index += 1 if separator else 2
            continue
        if option in value_options:
            index += 1 if separator else 2
            continue
        if option in flag_options:
            index += 1
            continue

        if word.startswith("--"):
            return None

        short_option = word[:2]
        attached_value = word[2:]
        if short_option in pattern_options and attached_value:
            return attached_value
        if short_option in pattern_file_options and attached_value:
            pattern_file_seen = True
            index += 1
            continue
        if short_option in value_options and attached_value:
            index += 1
            continue
        if len(word) > 2 and all(character in short_flags for character in word[1:]):
            index += 1
            continue
        return None
    if index >= len(words):
        return None
    if pattern_file_seen:
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
    timeout_ms: int = 500,
) -> dict[str, Any]:
    """Return an opt-in Claude Code settings fragment for the hook."""
    if not enabled:
        return {"hooks": {"PreToolUse": []}}
    inner_timeout_ms = min(500, max(1, timeout_ms))
    # The command process must start, import the CLI, and decode stdin before the
    # inner recall budget begins. Keep those two budgets distinct.
    startup_headroom_seconds = 0.5
    timeout_seconds = inner_timeout_ms / 1000 + startup_headroom_seconds
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash|Grep",
                    "hooks": [
                        {
                            "type": "command",
                            "command": (
                                "synapt recall grep-intercept "
                                f"--timeout-ms {inner_timeout_ms}"
                            ),
                            "timeout": timeout_seconds,
                        }
                    ],
                }
            ]
        }
    }
