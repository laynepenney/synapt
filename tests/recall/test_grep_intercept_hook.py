"""TDD contract for recall#836 grep-intercept hook."""

from __future__ import annotations

import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _grep_intercept():
    return importlib.import_module("synapt.integrations.grep_intercept")


def _bash(command: str) -> dict[str, Any]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": command},
    }


def _grep_tool(pattern: str, *, path: str = ".") -> dict[str, Any]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Grep",
        "tool_input": {"pattern": pattern, "path": path},
    }


def test_extracts_patterns_from_grep_and_rg_shapes() -> None:
    mod = _grep_intercept()

    cases = [
        (_bash('rg -n "memory leak" src tests'), "memory leak"),
        (_bash("rg --fixed-strings 'SYNAPT_SHARED_CHANNELS_DIR' src"), "SYNAPT_SHARED_CHANNELS_DIR"),
        (_bash("grep -R --line-number 'worker_ready' runtime-logs"), "worker_ready"),
        (_bash("grep -- 'literal-leading-dash' README.md"), "literal-leading-dash"),
        (_bash("rg -g '*.py' needle src"), "needle"),
        (_bash("rg --glob '*.py' needle src"), "needle"),
        (_bash("rg --replace replacement needle src"), "needle"),
        (_bash("rg -t py needle src"), "needle"),
        (_bash("grep --regexp=needle src/file"), "needle"),
        (_bash("grep -A 3 needle src/file"), "needle"),
        (_grep_tool("recall_quick verified absence", path="src"), "recall_quick verified absence"),
    ]

    for tool_call, expected in cases:
        assert mod.extract_grep_pattern(tool_call) == expected


def test_non_grep_commands_are_ignored() -> None:
    mod = _grep_intercept()

    assert mod.extract_grep_pattern(_bash("git status --short")) is None
    assert mod.extract_grep_pattern(_bash("python -m pytest tests/recall -q")) is None
    assert mod.extract_grep_pattern(_bash("rg --future-option value needle")) is None
    assert (
        mod.extract_grep_pattern(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Read",
                "tool_input": {"file_path": "README.md"},
            }
        )
        is None
    )


def test_pattern_file_options_refuse_a_path_without_an_explicit_pattern() -> None:
    mod = _grep_intercept()

    pattern_file_only = [
        _bash("grep -f patterns.txt src"),
        _bash("grep -fpatterns.txt src"),
        _bash("grep --file patterns.txt src"),
        _bash("grep --file=patterns.txt src"),
        _bash("rg -f patterns.txt src"),
        _bash("rg -fpatterns.txt src"),
        _bash("rg --file patterns.txt src"),
        _bash("rg --file=patterns.txt src"),
    ]

    for tool_call in pattern_file_only:
        assert mod.extract_grep_pattern(tool_call) is None
    assert mod.extract_grep_pattern(_bash("grep -f patterns.txt -e needle src")) == "needle"


def test_annotation_format_for_positive_recall_hit() -> None:
    mod = _grep_intercept()
    config = mod.GrepInterceptConfig(enabled=True, timeout_ms=150)
    tool_result = "src/app.py:10: memory leak fixed here"

    def recall_quick(query: str) -> str:
        assert query == "memory leak"
        return "\n".join([
            "Past session context:",
            "--- [cluster: memory leak triage] 2026-06-01, 4 chunks (clust-alpha) ---",
            "Memory leak investigation context.",
            "--- [knowledge #42] debugging (high, today) ---",
            "Memory leak root cause was retained callbacks.",
            "--- [2026-06-02 session beta1234] assistant turn ---",
            "Patch landed in src/app.py.",
        ])

    annotated = mod.annotate_tool_result(
        _bash('rg "memory leak" src'),
        tool_result,
        config=config,
        recall_quick=recall_quick,
    )

    assert annotated == (
        'recall: 3 related conversations (recall_search "memory leak" for detail)\n'
        + tool_result
    )


def test_hit_discriminator_uses_real_recall_quick_block_shape() -> None:
    mod = _grep_intercept()
    recall_hit = "\n".join([
        "Past session context:",
        "--- [cluster: memory leak triage] 2026-06-01, 4 chunks (clust-alpha) ---",
        "Cluster summary.",
        "--- [knowledge #42] debugging (high, today) ---",
        "Knowledge content.",
        "--- [2026-06-02 08:15 session beta1234] assistant turn ---",
        "Raw chunk content.",
    ])

    assert mod.count_related_conversations(recall_hit) == 3


def test_hit_discriminator_treats_informative_absences_as_zero() -> None:
    mod = _grep_intercept()

    assert (
        mod.count_related_conversations(
            "No prior discussion found for 'licenses proceeding'. Proceeding fresh is safe."
        )
        == 0
    )
    assert (
        mod.count_related_conversations(
            "No indexed recall corpus available for 'anything' "
            "(0 sessions, 0 chunks scanned). Verified absence unavailable. "
            "The index is empty."
        )
        == 0
    )
    assert mod.count_related_conversations("No results found.") == 0


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
    # Small 5ms internal budget, far under the 150ms product ceiling. slow_recall
    # always exceeds it, so the bounded join always times out to a no-op; the
    # small budget maximizes headroom for CI scheduling jitter (a 25ms budget ran
    # 0.153s on a loaded macOS 3.10 runner) so the wall-clock stays comfortably
    # under the <150ms product assertion the contract requires.
    config = mod.GrepInterceptConfig(enabled=True, timeout_ms=5)
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


def test_pretooluse_context_is_advisory_and_does_not_require_tool_result() -> None:
    mod = _grep_intercept()
    config = mod.GrepInterceptConfig(enabled=True, timeout_ms=150)

    def recall_quick(query: str) -> str:
        assert query == "memory leak"
        return "\n".join([
            "Past session context:",
            "--- [cluster: memory leak triage] 2026-06-01, 4 chunks (clust-alpha) ---",
            "Memory leak investigation context.",
        ])

    context = mod.build_pretooluse_context(
        _bash('rg "memory leak" src'),
        config=config,
        recall_quick=recall_quick,
    )

    assert context == 'recall: 1 related conversations (recall_search "memory leak" for detail)'


def test_pretooluse_output_uses_current_claude_hook_envelope() -> None:
    mod = _grep_intercept()
    config = mod.GrepInterceptConfig(enabled=True, timeout_ms=150)

    output = mod.build_pretooluse_output(
        _bash('rg "memory leak" src'),
        config=config,
        recall_quick=lambda _query: "\n".join(
            [
                "Past session context:",
                "--- [cluster: memory leak triage] 2026-06-01, 4 chunks (clust-alpha) ---",
                "Cluster summary.",
            ]
        ),
    )

    assert output == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": (
                'recall: 1 related conversations '
                '(recall_search "memory leak" for detail)'
            ),
        }
    }


def test_cli_command_is_registered_and_silent_on_miss(tmp_path) -> None:
    payload = _bash('rg "a query with no local corpus" src')
    env = os.environ.copy()
    env["SYNAPT_RECALL_ROOT"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")

    result = subprocess.run(
        [sys.executable, "-m", "synapt.cli", "recall", "grep-intercept"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=3,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_cli_positive_hit_finishes_inside_published_hook_budget(tmp_path) -> None:
    from synapt.recall.core import TranscriptChunk, TranscriptIndex
    from synapt.recall.storage import RecallDB

    mod = _grep_intercept()
    store_root = tmp_path / "store"
    index_dir = store_root / ".synapt" / "recall" / "index"
    index_dir.mkdir(parents=True)
    db = RecallDB(index_dir / "recall.db")
    chunk = TranscriptChunk(
        id="session-hit:t0",
        session_id="session-hit",
        timestamp="2026-08-30T08:00:00+00:00",
        turn_index=0,
        user_text="grep_intercept.py bounded startup witness",
        assistant_text="the recall context survives process startup headroom",
    )
    index = TranscriptIndex([chunk], use_embeddings=False, cache_dir=index_dir, db=db)
    index.save(index_dir)

    payload = _bash('rg "grep_intercept.py" src')
    env = os.environ.copy()
    env["SYNAPT_RECALL_ROOT"] = str(store_root)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    snippet = mod.claude_pretooluse_settings_snippet(enabled=True)
    hook = snippet["hooks"]["PreToolUse"][0]["hooks"][0]
    published_command = shlex.split(hook["command"])
    assert published_command[:2] == ["synapt", "recall"]
    outer_timeout = snippet["hooks"]["PreToolUse"][0]["hooks"][0]["timeout"]

    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "synapt.cli", "recall", *published_command[2:]],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=outer_timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started

    assert result.returncode == 0
    assert result.stderr == ""
    output = json.loads(result.stdout)
    context = output["hookSpecificOutput"]["additionalContext"]
    assert context == (
        'recall: 1 related conversations '
        '(recall_search "grep_intercept.py" for detail)'
    )
    assert elapsed < outer_timeout
    print(
        f"positive-cli: bytes={len(result.stdout.encode())} "
        f"outer={outer_timeout:.3f}s"
    )


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

    disabled = mod.claude_pretooluse_settings_snippet(enabled=False, timeout_ms=150)
    snippet = mod.claude_pretooluse_settings_snippet(enabled=True, timeout_ms=150)

    assert disabled == {"hooks": {"PreToolUse": []}}
    assert "hooks" in snippet
    assert "PreToolUse" in snippet["hooks"]
    matcher = snippet["hooks"]["PreToolUse"][0]
    assert matcher["matcher"] == "Bash|Grep"
    hook = matcher["hooks"][0]
    assert hook["command"] == "synapt recall grep-intercept --timeout-ms 150"
    assert hook["timeout"] == 0.650
