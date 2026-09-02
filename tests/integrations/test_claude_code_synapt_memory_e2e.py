"""Red specs for Claude Code + SynaptMemoryTool end-to-end integration."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("anthropic", reason="requires the optional anthropic integration")

from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)

from synapt.recall.archive import archive_transcripts
from synapt.recall.core import build_index, project_archive_dir, project_index_dir
from synapt.recall.server import recall_search


def _load_synapt_memory_tool():
    try:
        module = importlib.import_module("synapt.integrations.anthropic")
    except ModuleNotFoundError:
        pytest.fail(
            "Expected backend module `synapt.integrations.anthropic` with "
            "`SynaptMemoryTool`, but it does not exist yet."
        )
    except Exception as exc:  # pragma: no cover - exercised once backend exists
        pytest.fail(f"Importing `synapt.integrations.anthropic` failed: {exc!r}")

    try:
        return module.SynaptMemoryTool
    except AttributeError:
        pytest.fail(
            "Expected `synapt.integrations.anthropic.SynaptMemoryTool`, "
            "but the symbol is missing."
        )


def _new_tool():
    cls = _load_synapt_memory_tool()
    return cls()


def _user_entry(text: str, *, uuid: str, ts: str) -> dict[str, object]:
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": "claude-session-1",
        "message": {"role": "user", "content": text},
    }


def _assistant_tool_use(
    *,
    uuid: str,
    ts: str,
    tool_use_id: str,
    command: str,
    path: str,
    extra_input: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"command": command, "path": path}
    if extra_input:
        payload.update(extra_input)
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": "claude-session-1",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": "memory",
                    "input": payload,
                }
            ],
        },
    }


def _assistant_text(text: str, *, uuid: str, ts: str) -> dict[str, object]:
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": "claude-session-1",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }


def _tool_result(result: str, *, uuid: str, ts: str, tool_use_id: str) -> dict[str, object]:
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": "claude-session-1",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                }
            ],
        },
    }


def _write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def _reset_recall_search_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    import synapt.recall.server as recall_server

    monkeypatch.setattr(recall_server, "_cached_index", None, raising=False)
    monkeypatch.setattr(recall_server, "_cached_mtime", None, raising=False)
    monkeypatch.setattr(recall_server, "_cached_dir", None, raising=False)
    monkeypatch.setattr(recall_server, "_cached_has_embeddings", False, raising=False)


def _exercise_memory_session(project: Path, source_dir: Path) -> tuple[object, Path]:
    tool = _new_tool()
    path = "/memories/notes/deploy.md"

    create_result = tool.execute(
        BetaMemoryTool20250818CreateCommand(
            command="create",
            path=path,
            file_text="deployment uses blue-green strategy",
        )
    )
    replace_result = tool.execute(
        BetaMemoryTool20250818StrReplaceCommand(
            command="str_replace",
            path=path,
            old_str="blue-green",
            new_str="canary",
        )
    )
    view_result = tool.execute(
        BetaMemoryTool20250818ViewCommand(command="view", path=path)
    )

    transcript = source_dir / "claude-session-1.jsonl"
    _write_jsonl(
        transcript,
        [
            _user_entry(
                "Remember our deployment strategy, update it, then show it back to me.",
                uuid="u1",
                ts="2026-04-24T10:00:00Z",
            ),
            _assistant_tool_use(
                uuid="a1",
                ts="2026-04-24T10:00:01Z",
                tool_use_id="toolu_create",
                command="create",
                path=path,
                extra_input={"file_text": "deployment uses blue-green strategy"},
            ),
            _tool_result(
                create_result,
                uuid="tr1",
                ts="2026-04-24T10:00:02Z",
                tool_use_id="toolu_create",
            ),
            _assistant_tool_use(
                uuid="a2",
                ts="2026-04-24T10:00:03Z",
                tool_use_id="toolu_replace",
                command="str_replace",
                path=path,
                extra_input={"old_str": "blue-green", "new_str": "canary"},
            ),
            _tool_result(
                replace_result,
                uuid="tr2",
                ts="2026-04-24T10:00:04Z",
                tool_use_id="toolu_replace",
            ),
            _assistant_tool_use(
                uuid="a3",
                ts="2026-04-24T10:00:05Z",
                tool_use_id="toolu_view",
                command="view",
                path=path,
            ),
            _tool_result(
                view_result,
                uuid="tr3",
                ts="2026-04-24T10:00:06Z",
                tool_use_id="toolu_view",
            ),
            _assistant_text(
                "Deployment uses canary strategy.",
                uuid="a4",
                ts="2026-04-24T10:00:07Z",
            ),
        ],
    )

    archive_transcripts(project, source_dir)
    index_dir = project_index_dir(project)
    index = build_index(
        project_archive_dir(project),
        use_embeddings=False,
        cache_dir=index_dir,
    )
    index.save(index_dir)
    return tool, index_dir


@pytest.mark.xfail(reason="Spec/impl mismatch: indexer records raw result text, not structured ops — reconcile in Sprint 30")
def test_claude_memory_session_indexes_command_level_operation_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source_dir = tmp_path / "claude-source"
    source_dir.mkdir()
    monkeypatch.chdir(project)
    _reset_recall_search_cache(monkeypatch)

    _, _ = _exercise_memory_session(project, source_dir)
    index = build_index(
        project_archive_dir(project),
        use_embeddings=False,
        cache_dir=project_index_dir(project),
    )

    assert len(index.chunks) == 1
    chunk = index.chunks[0]
    assert chunk.tools_used == ["memory"]
    assert "memory.create /memories/notes/deploy.md" in chunk.tool_content
    assert "memory.str_replace /memories/notes/deploy.md" in chunk.tool_content
    assert "blue-green" in chunk.tool_content
    assert "canary" in chunk.tool_content
    assert "memory.view /memories/notes/deploy.md" in chunk.tool_content
    assert "1\tdeployment uses canary strategy" in chunk.tool_content


@pytest.mark.xfail(reason="Spec/impl mismatch: search result doesn't include virtual path — reconcile in Sprint 30")
def test_claude_memory_writes_flow_into_recall_search_with_memory_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source_dir = tmp_path / "claude-source"
    source_dir.mkdir()
    monkeypatch.chdir(project)
    _reset_recall_search_cache(monkeypatch)

    _tool, _ = _exercise_memory_session(project, source_dir)

    search_result = recall_search(
        "deployment canary strategy",
        max_chunks=5,
        max_tokens=800,
    )

    assert "deployment uses canary strategy" in search_result
    assert "/memories/notes/deploy.md" in search_result
    assert "str_replace" in search_result
