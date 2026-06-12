"""recall#836 grep-intercept hook.

Agents reach for grep/rg at impulse time because deterministic tools beat
probabilistic ones. This module meets the impulse instead of fighting it: it
recognises grep-shaped tool calls, runs recall_quick on the search pattern, and
surfaces a one-line pointer to related conversations. Two entry shapes:

- ``build_pretooluse_context`` -- advisory context BEFORE the grep runs (the
  Claude Code PreToolUse hook point; there is no tool result yet).
- ``annotate_tool_result`` -- a pure post-result helper that prefixes the same
  pointer onto a result the agent already has.

Both share one rule set: opt-in only, silent no-op on a miss / unavailable
recall / timeout, and the original grep result is never blocked or altered on
failure. The <150ms budget is enforced with a bounded wait so a slow recall can
never delay the grep.

Premium boundary: OSS (retrieval integration; no identity / org semantics).
"""

from __future__ import annotations

import math
import shlex
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional


# Grep-family executables we recognise as "the agent is searching".
_GREP_EXECUTABLES = frozenset({"grep", "egrep", "fgrep", "rg", "ripgrep"})

# recall_quick renders each related conversation as a block header line of the
# form ``--- ... ---`` under a leading ``Past session context:`` header. This is
# the ONE seam coupling the hook to recall_quick's output format: if that format
# changes, count_related_conversations' test fails (one test, not two features).
_RECALL_HIT_HEADER = "Past session context:"
_BLOCK_HEADER = "---"


@dataclass
class GrepInterceptConfig:
    """Opt-in configuration for the grep-intercept hook.

    enabled defaults False: the hook does nothing until a workspace opts in.
    timeout_ms bounds how long recall_quick may run before the hook gives up and
    leaves the grep result untouched.
    """

    enabled: bool = False
    timeout_ms: int = 150


def extract_grep_pattern(tool_call: dict[str, Any]) -> Optional[str]:
    """Return the search pattern for a grep-shaped tool call, else None.

    Handles the native Grep tool (``input.pattern``) and Bash commands invoking
    a grep-family executable. Pattern is the first positional argument after the
    flags, honouring ``--`` as end-of-options. Non-grep commands, pipelines that
    do not start with grep/rg, and other tools return None (a safe no-op).

    Best-effort by design: value-taking flags (e.g. ``-A 3``) are not modelled,
    so a mis-extraction degrades to a likely recall miss, which is itself a
    silent no-op. Never raises.
    """
    if not isinstance(tool_call, dict):
        return None
    tool_name = tool_call.get("tool_name")
    tool_input = tool_call.get("input") or {}

    if tool_name == "Grep":
        pattern = tool_input.get("pattern")
        return pattern if isinstance(pattern, str) and pattern else None

    if tool_name != "Bash":
        return None

    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return None

    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens:
        return None

    executable = tokens[0].rsplit("/", 1)[-1]
    if executable not in _GREP_EXECUTABLES:
        return None

    args = tokens[1:]
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            # End of options: the next token is the pattern.
            return args[index + 1] if index + 1 < len(args) else None
        if token.startswith("-"):
            index += 1
            continue
        # First positional after the flags is the pattern.
        return token
    return None


def count_related_conversations(recall_output: str) -> int:
    """Count related conversations in recall_quick output.

    A hit renders block headers (``--- ... ---``) under a ``Past session
    context:`` header, one per related conversation (cluster / knowledge / raw /
    source). Every informative absence from recall#837 (verified absence,
    empty-corpus / "Verified absence unavailable") and any generic no-result
    output carries no such header and therefore counts as zero, so the hook
    never annotates on top of an absence.
    """
    if not recall_output or _RECALL_HIT_HEADER not in recall_output:
        return 0
    count = 0
    for line in recall_output.splitlines():
        stripped = line.strip()
        if stripped.startswith(_BLOCK_HEADER) and stripped.endswith(_BLOCK_HEADER) and len(stripped) > len(_BLOCK_HEADER):
            count += 1
    return count


def _recall_pointer(
    tool_call: dict[str, Any],
    config: GrepInterceptConfig,
    recall_quick: Callable[[str], str],
) -> Optional[str]:
    """Shared seam: the one-line recall pointer for a grep-shaped tool call, or
    None when the hook should stay silent (disabled / not grep / timeout /
    unavailable / no related conversations).
    """
    if not config.enabled:
        return None
    pattern = extract_grep_pattern(tool_call)
    if pattern is None:
        return None

    output, failed = _run_bounded(recall_quick, pattern, config.timeout_ms)
    if failed or output is None:
        return None

    count = count_related_conversations(output)
    if count <= 0:
        return None

    return f'recall: {count} related conversations (recall_search "{pattern}" for detail)'


def build_pretooluse_context(
    tool_call: dict[str, Any],
    *,
    config: GrepInterceptConfig,
    recall_quick: Callable[[str], str],
) -> Optional[str]:
    """PreToolUse advisory context for a grep-shaped tool call, or None.

    Fires before the grep runs, so there is no tool result. Returns only the
    one-line recall pointer (no result body) for injection as additional
    context, or None when the hook should stay silent.
    """
    return _recall_pointer(tool_call, config, recall_quick)


def annotate_tool_result(
    tool_call: dict[str, Any],
    tool_result: str,
    *,
    config: GrepInterceptConfig,
    recall_quick: Callable[[str], str],
) -> str:
    """Pure post-result helper: prefix the recall pointer onto an existing grep
    result. Returns the result untouched on any non-hit condition (disabled, not
    grep, miss, unavailable, timeout), so the grep result is never blocked.
    """
    pointer = _recall_pointer(tool_call, config, recall_quick)
    if pointer is None:
        return tool_result
    return f"{pointer}\n{tool_result}"


def _run_bounded(
    fn: Callable[[str], str],
    arg: str,
    timeout_ms: int,
) -> tuple[Optional[str], bool]:
    """Run fn(arg) under a wall-clock bound. Returns (value, failed).

    The work runs on a daemon thread joined with the timeout, so a slow or
    hanging recall_quick can never delay the caller beyond timeout_ms. On
    timeout the daemon thread is abandoned (it cannot be killed; it is harmless
    and reclaimed at interpreter exit). Any exception from fn is swallowed and
    reported as failed, so recall being unavailable is a silent no-op.
    """
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["value"] = fn(arg)
        except BaseException:  # noqa: BLE001 - silent no-op is the contract
            box["failed"] = True

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(max(0.0, timeout_ms / 1000.0))

    if thread.is_alive():
        return None, True
    if box.get("failed"):
        return None, True
    return box.get("value"), False


def claude_pretooluse_settings_snippet(
    enabled: bool = True,
    timeout_ms: int = 150,
) -> dict[str, Any]:
    """Return a Claude Code settings.json snippet wiring the PreToolUse hook.

    The hook matches Bash and Grep tool calls and invokes the
    ``synapt recall grep-intercept`` command. The outer hook timeout is bounded
    to whole seconds (>=1) covering the internal timeout_ms recall budget. When
    disabled, the snippet wires no matcher so applying it is a no-op.
    """
    if not enabled:
        return {"hooks": {"PreToolUse": []}}

    outer_timeout_s = max(1, math.ceil(timeout_ms / 1000))
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash|Grep",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "synapt recall grep-intercept",
                            "timeout": outer_timeout_s,
                        }
                    ],
                }
            ]
        }
    }
