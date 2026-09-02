"""Red specs for the Anthropic memory backend."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

pytest.importorskip("anthropic", reason="requires the optional anthropic integration")

from anthropic.lib.tools._beta_builtin_memory_tool import BetaAbstractMemoryTool
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)


def _load_synapt_memory_tool():
    try:
        module = importlib.import_module("synapt.integrations.anthropic")
    except ModuleNotFoundError as exc:
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


def _new_tool(project_root: Path):
    cls = _load_synapt_memory_tool()
    try:
        return cls(project_root=project_root)
    except TypeError as exc:
        pytest.fail(
            "SynaptMemoryTool should be constructible as "
            "`SynaptMemoryTool(project_root=Path(...))`."
        )


def _assert_line_numbered_view(result: str, expected: list[tuple[int, str]]) -> None:
    for line_no, text in expected:
        pattern = rf"(?m)^\s*{line_no}(?:\t|:)\s*{re.escape(text)}$"
        assert re.search(pattern, result), result


def test_synapt_memory_tool_matches_anthropic_memory_tool_contract(tmp_path: Path) -> None:
    tool = _new_tool(tmp_path)

    assert isinstance(tool, BetaAbstractMemoryTool)
    assert tool.to_dict() == {"type": "memory_20250818", "name": "memory"}



def test_create_and_view_expose_a_virtual_memories_filesystem(tmp_path: Path) -> None:
    tool = _new_tool(tmp_path)

    create_result = tool.execute(
        BetaMemoryTool20250818CreateCommand(
            command="create",
            path="/memories/progress/todo.txt",
            file_text="first line\nsecond line\n",
        )
    )
    assert create_result == "File created successfully at: /memories/progress/todo.txt"

    directory_result = tool.execute(
        BetaMemoryTool20250818ViewCommand(command="view", path="/memories")
    )
    assert "/memories/progress/todo.txt" in directory_result
    assert "/memories" in directory_result

    file_result = tool.execute(
        BetaMemoryTool20250818ViewCommand(
            command="view",
            path="/memories/progress/todo.txt",
        )
    )
    assert "Here's the content of /memories/progress/todo.txt" in file_result
    _assert_line_numbered_view(
        file_result,
        [(1, "first line"), (2, "second line")],
    )

    range_result = tool.execute(
        BetaMemoryTool20250818ViewCommand(
            command="view",
            path="/memories/progress/todo.txt",
            view_range=[2, -1],
        )
    )
    assert "Here's the content of /memories/progress/todo.txt" in range_result
    assert not re.search(r"(?m)^\s*1(?:\t|:)", range_result), range_result
    _assert_line_numbered_view(range_result, [(2, "second line")])



def test_str_replace_matches_filesystem_tool_style_and_returns_numbered_snippet(
    tmp_path: Path,
) -> None:
    tool = _new_tool(tmp_path)
    tool.execute(
        BetaMemoryTool20250818CreateCommand(
            command="create",
            path="/memories/preferences.txt",
            file_text="favorite drink: tea\nfavorite snack: pears\n",
        )
    )

    result = tool.execute(
        BetaMemoryTool20250818StrReplaceCommand(
            command="str_replace",
            path="/memories/preferences.txt",
            old_str="favorite drink: tea",
            new_str="favorite drink: coffee",
        )
    )

    assert "The memory file has been edited." in result
    _assert_line_numbered_view(result, [(1, "favorite drink: coffee")])



def test_insert_matches_filesystem_tool_success_message(tmp_path: Path) -> None:
    tool = _new_tool(tmp_path)
    tool.execute(
        BetaMemoryTool20250818CreateCommand(
            command="create",
            path="/memories/todo.txt",
            file_text="- first\n- third\n",
        )
    )

    result = tool.execute(
        BetaMemoryTool20250818InsertCommand(
            command="insert",
            path="/memories/todo.txt",
            insert_line=1,
            insert_text="- second\n",
        )
    )

    assert result == "The file /memories/todo.txt has been edited."
    file_result = tool.execute(
        BetaMemoryTool20250818ViewCommand(command="view", path="/memories/todo.txt")
    )
    _assert_line_numbered_view(
        file_result,
        [(1, "- first"), (2, "- second"), (3, "- third")],
    )



def test_delete_and_rename_match_documented_success_messages(tmp_path: Path) -> None:
    tool = _new_tool(tmp_path)
    tool.execute(
        BetaMemoryTool20250818CreateCommand(
            command="create",
            path="/memories/draft.txt",
            file_text="draft\n",
        )
    )

    rename_result = tool.execute(
        BetaMemoryTool20250818RenameCommand(
            command="rename",
            old_path="/memories/draft.txt",
            new_path="/memories/final.txt",
        )
    )
    assert rename_result == (
        "Successfully renamed /memories/draft.txt to /memories/final.txt"
    )

    delete_result = tool.execute(
        BetaMemoryTool20250818DeleteCommand(
            command="delete",
            path="/memories/final.txt",
        )
    )
    assert delete_result == "Successfully deleted /memories/final.txt"



def test_path_traversal_is_rejected_by_the_virtual_facade(tmp_path: Path) -> None:
    tool = _new_tool(tmp_path)

    with pytest.raises(Exception, match="traversal"):
        tool.execute(
            BetaMemoryTool20250818ViewCommand(
                command="view",
                path="/memories/../../outside.txt",
            )
        )

    assert not (tmp_path.parent / "outside.txt").exists()



def test_write_operations_schedule_async_enrichment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _new_tool(tmp_path)
    scheduled: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_schedule(*args: object, **kwargs: object) -> None:
        scheduled.append((args, kwargs))

    monkeypatch.setattr(tool, "_schedule_enrichment", fake_schedule, raising=False)

    tool.execute(
        BetaMemoryTool20250818CreateCommand(
            command="create",
            path="/memories/notes.txt",
            file_text="alpha\nbeta\n",
        )
    )
    tool.execute(
        BetaMemoryTool20250818StrReplaceCommand(
            command="str_replace",
            path="/memories/notes.txt",
            old_str="beta",
            new_str="gamma",
        )
    )
    tool.execute(
        BetaMemoryTool20250818InsertCommand(
            command="insert",
            path="/memories/notes.txt",
            insert_line=2,
            insert_text="delta\n",
        )
    )

    assert len(scheduled) == 3
    assert all("/memories/notes.txt" in repr(call) for call in scheduled)
