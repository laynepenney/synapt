"""Tests for the Anthropic Memory Tool backend (SynaptMemoryTool)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("anthropic")

from anthropic.types.beta import (
    BetaMemoryTool20250818ViewCommand,
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
)


@pytest.fixture
def tool():
    with patch("synapt.integrations.anthropic._RecallBridge") as MockBridge:
        bridge_instance = MockBridge.return_value
        bridge_instance.save.return_value = "saved"
        bridge_instance.retract.return_value = "retracted"
        bridge_instance.search.return_value = "No results found."
        bridge_instance.load_knowledge_files.return_value = None

        from synapt.integrations.anthropic import SynaptMemoryTool
        t = SynaptMemoryTool()
        t._core._bridge = bridge_instance
        yield t


def _view(path, view_range=None):
    return BetaMemoryTool20250818ViewCommand(command="view", path=path, view_range=view_range)


def _create(path, file_text):
    return BetaMemoryTool20250818CreateCommand(command="create", path=path, file_text=file_text)


def _str_replace(path, old_str, new_str):
    return BetaMemoryTool20250818StrReplaceCommand(
        command="str_replace", path=path, old_str=old_str, new_str=new_str
    )


def _insert(path, insert_line, insert_text):
    return BetaMemoryTool20250818InsertCommand(
        command="insert", path=path, insert_line=insert_line, insert_text=insert_text
    )


def _delete(path):
    return BetaMemoryTool20250818DeleteCommand(command="delete", path=path)


def _rename(old_path, new_path):
    return BetaMemoryTool20250818RenameCommand(command="rename", old_path=old_path, new_path=new_path)


class TestViewCommand:

    def test_view_root_lists_directories(self, tool):
        result = tool.view(_view("/memories"))
        assert "knowledge/" in result
        assert "sessions/" in result
        assert "notes/" in result

    def test_view_empty_directory(self, tool):
        result = tool.view(_view("/memories/knowledge"))
        assert "empty directory" in result

    def test_view_nonexistent_file(self, tool):
        with pytest.raises(Exception, match="does not exist"):
            tool.view(_view("/memories/knowledge/nope.md"))

    def test_view_file_with_line_numbers(self, tool):
        tool.create(_create("/memories/notes/deploy.md", "line one\nline two\nline three"))
        result = tool.view(_view("/memories/notes/deploy.md"))
        assert "1\tline one" in result
        assert "2\tline two" in result
        assert "3\tline three" in result

    def test_view_file_with_range(self, tool):
        tool.create(_create("/memories/notes/multi.md", "a\nb\nc\nd\ne"))
        result = tool.view(_view("/memories/notes/multi.md", view_range=[2, 4]))
        assert "2\tb" in result
        assert "3\tc" in result
        assert "4\td" in result
        assert "1\ta" not in result
        assert "5\te" not in result

    def test_view_directory_listing(self, tool):
        tool.create(_create("/memories/notes/a.md", "aaa"))
        tool.create(_create("/memories/notes/b.md", "bbb"))
        result = tool.view(_view("/memories/notes"))
        assert "a.md" in result
        assert "b.md" in result

    def test_view_no_search_augmentation(self, tool):
        tool._core._bridge.search.return_value = "Relevant context: deploy uses blue-green"
        tool.create(_create("/memories/notes/deploy.md", "blue-green deploys"))
        result = tool.view(_view("/memories/notes/deploy.md"))
        assert "Related recall context" not in result


class TestCreateCommand:

    def test_create_new_file(self, tool):
        result = tool.create(_create("/memories/notes/test.md", "hello world"))
        assert "created successfully" in result

    def test_create_duplicate_raises(self, tool):
        tool.create(_create("/memories/notes/dup.md", "first"))
        with pytest.raises(Exception, match="already exists"):
            tool.create(_create("/memories/notes/dup.md", "second"))

    def test_create_triggers_recall_save(self, tool):
        tool.create(_create("/memories/knowledge/fact.md", "Python 3.13 is latest"))
        tool._core._bridge.save.assert_called()

    def test_create_relative_path(self, tool):
        result = tool.create(_create("notes/rel.md", "relative path"))
        assert "created successfully" in result
        view_result = tool.view(_view("/memories/notes/rel.md"))
        assert "relative path" in view_result


class TestStrReplaceCommand:

    def test_replace_succeeds(self, tool):
        tool.create(_create("/memories/notes/edit.md", "old text here"))
        result = tool.str_replace(_str_replace("/memories/notes/edit.md", "old text", "new text"))
        assert "The memory file has been edited." in result
        view = tool.view(_view("/memories/notes/edit.md"))
        assert "new text" in view
        assert "old text" not in view

    def test_replace_not_found_raises(self, tool):
        tool.create(_create("/memories/notes/nf.md", "some content"))
        with pytest.raises(Exception, match="not found"):
            tool.str_replace(_str_replace("/memories/notes/nf.md", "missing", "new"))

    def test_replace_ambiguous_raises(self, tool):
        tool.create(_create("/memories/notes/amb.md", "aaa aaa"))
        with pytest.raises(Exception, match="2 times"):
            tool.str_replace(_str_replace("/memories/notes/amb.md", "aaa", "bbb"))

    def test_replace_nonexistent_file_raises(self, tool):
        with pytest.raises(Exception, match="does not exist"):
            tool.str_replace(_str_replace("/memories/notes/nope.md", "a", "b"))


class TestInsertCommand:

    def test_insert_at_beginning(self, tool):
        tool.create(_create("/memories/notes/ins.md", "line 1\nline 2"))
        result = tool.insert(_insert("/memories/notes/ins.md", 0, "new first"))
        assert "has been edited." in result
        view = tool.view(_view("/memories/notes/ins.md"))
        assert "1\tnew first" in view

    def test_insert_at_end(self, tool):
        tool.create(_create("/memories/notes/ins2.md", "line 1\nline 2"))
        tool.insert(_insert("/memories/notes/ins2.md", 2, "line 3"))
        view = tool.view(_view("/memories/notes/ins2.md"))
        assert "3\tline 3" in view

    def test_insert_out_of_range_raises(self, tool):
        tool.create(_create("/memories/notes/ins3.md", "one"))
        with pytest.raises(Exception, match="out of range"):
            tool.insert(_insert("/memories/notes/ins3.md", 100, "nope"))

    def test_insert_nonexistent_file_raises(self, tool):
        with pytest.raises(Exception, match="does not exist"):
            tool.insert(_insert("/memories/notes/nope.md", 0, "text"))


class TestDeleteCommand:

    def test_delete_file(self, tool):
        tool.create(_create("/memories/notes/del.md", "to delete"))
        result = tool.delete(_delete("/memories/notes/del.md"))
        assert "deleted" in result
        with pytest.raises(Exception, match="does not exist"):
            tool.view(_view("/memories/notes/del.md"))

    def test_delete_nonexistent_raises(self, tool):
        with pytest.raises(Exception, match="does not exist"):
            tool.delete(_delete("/memories/notes/nope.md"))

    def test_delete_directory(self, tool):
        tool.create(_create("/memories/notes/sub/a.md", "aaa"))
        tool.create(_create("/memories/notes/sub/b.md", "bbb"))
        result = tool.delete(_delete("/memories/notes/sub"))
        assert "deleted" in result
        assert "2 file(s)" in result

    def test_delete_triggers_recall_retract(self, tool):
        tool.create(_create("/memories/notes/ret.md", "retract me"))
        tool.delete(_delete("/memories/notes/ret.md"))
        tool._core._bridge.retract.assert_called()


class TestRenameCommand:

    def test_rename_succeeds(self, tool):
        tool.create(_create("/memories/notes/old.md", "content"))
        result = tool.rename(_rename("/memories/notes/old.md", "/memories/notes/new.md"))
        assert "Successfully renamed" in result
        with pytest.raises(Exception, match="does not exist"):
            tool.view(_view("/memories/notes/old.md"))
        view_new = tool.view(_view("/memories/notes/new.md"))
        assert "content" in view_new

    def test_rename_source_missing_raises(self, tool):
        with pytest.raises(Exception, match="does not exist"):
            tool.rename(_rename("/memories/notes/nope.md", "/memories/notes/new.md"))

    def test_rename_dest_exists_raises(self, tool):
        tool.create(_create("/memories/notes/a.md", "aaa"))
        tool.create(_create("/memories/notes/b.md", "bbb"))
        with pytest.raises(Exception, match="already exists"):
            tool.rename(_rename("/memories/notes/a.md", "/memories/notes/b.md"))


class TestGetContext:

    def test_returns_search_results(self, tool):
        tool._core._bridge.search.return_value = "deploy uses blue-green strategy"
        result = tool.get_context("deployment")
        assert "blue-green" in result

    def test_returns_empty_on_no_results(self, tool):
        tool._core._bridge.search.return_value = "No results found."
        result = tool.get_context("nonexistent topic")
        assert result == ""

    def test_returns_empty_on_error(self, tool):
        tool._core._bridge.search.side_effect = Exception("boom")
        result = tool.get_context("anything")
        assert result == ""


class TestExecuteDispatch:

    def test_execute_routes_view(self, tool):
        result = tool.execute(_view("/memories"))
        assert "knowledge/" in result

    def test_execute_routes_create(self, tool):
        result = tool.execute(_create("/memories/notes/exec.md", "via execute"))
        assert "created successfully" in result


class TestClearAll:

    def test_clear_all_memory(self, tool):
        tool.create(_create("/memories/notes/a.md", "aaa"))
        tool.create(_create("/memories/notes/b.md", "bbb"))
        result = tool.clear_all_memory()
        assert "cleared" in result
        with pytest.raises(Exception, match="does not exist"):
            tool.view(_view("/memories/notes/a.md"))


class TestPathNormalization:

    def test_relative_path_normalized(self, tool):
        tool.create(_create("knowledge/fact.md", "fact"))
        result = tool.view(_view("/memories/knowledge/fact.md"))
        assert "fact" in result

    def test_bare_filename(self, tool):
        tool.create(_create("test.md", "bare"))
        result = tool.view(_view("/memories/test.md"))
        assert "bare" in result


class TestPathTraversal:

    def test_traversal_raises_on_view(self, tool):
        with pytest.raises(Exception, match="traversal"):
            tool.view(_view("/memories/../../outside.txt"))

    def test_traversal_raises_on_create(self, tool):
        with pytest.raises(Exception, match="traversal"):
            tool.create(_create("/memories/../../../etc/passwd", "hack"))


class TestImportability:

    def test_sync_importable(self):
        from synapt.integrations.anthropic import SynaptMemoryTool
        assert SynaptMemoryTool is not None

    def test_async_importable(self):
        from synapt.integrations.anthropic import SynaptAsyncMemoryTool
        assert SynaptAsyncMemoryTool is not None

    def test_to_dict(self, tool):
        d = tool.to_dict()
        assert d["type"] == "memory_20250818"
        assert d["name"] == "memory"


class TestRecallBridgeLoadKnowledgeFilesRouting:
    """load_knowledge_files hardcoded project_index_dir(project) / "recall.db"
    with no is_sharded fallback: on a sharded store (only index.db present,
    the file consolidation writes to since the routing fix) db_path.exists()
    was always False, so this silently returned with the cache empty --
    every existing knowledge node invisible to the memory-tool view command,
    with no error. Real temp dirs and a real _RecallBridge, not the mocked
    fixture the rest of this file uses -- the mock can't witness which path
    RecallDB was constructed with resolving wrong."""

    def test_seeds_the_cache_from_a_sharded_store(self, tmp_path):
        from synapt.integrations.anthropic import _FileCache, _RecallBridge
        from synapt.recall.knowledge import KnowledgeNode
        from synapt.recall.storage import RecallDB

        project = tmp_path / "proj"
        index_dir = project / ".synapt" / "recall" / "index"
        index_dir.mkdir(parents=True)
        db = RecallDB(index_dir / "index.db")  # the sharded marker
        db.save_knowledge_nodes([
            KnowledgeNode.create(
                content="the launch runbook lives in docs/runbook.md",
                category="workflow",
                source_sessions=["s0"],
                node_id="n1",
            ).to_dict(),
        ])
        db.close()
        assert not (index_dir / "recall.db").exists()

        bridge = _RecallBridge(project_root=project)
        cache = _FileCache()
        bridge.load_knowledge_files(cache)

        assert any("runbook" in v for v in cache._files.values())

    def test_seeds_the_cache_from_a_monolithic_store_control(self, tmp_path):
        """Control: the untouched, no-index.db case is unaffected."""
        from synapt.integrations.anthropic import _FileCache, _RecallBridge
        from synapt.recall.knowledge import KnowledgeNode
        from synapt.recall.storage import RecallDB

        project = tmp_path / "proj"
        index_dir = project / ".synapt" / "recall" / "index"
        index_dir.mkdir(parents=True)
        db = RecallDB(index_dir / "recall.db")  # no index.db at all
        db.save_knowledge_nodes([
            KnowledgeNode.create(
                content="the launch runbook lives in docs/runbook.md",
                category="workflow",
                source_sessions=["s0"],
                node_id="n1",
            ).to_dict(),
        ])
        db.close()

        bridge = _RecallBridge(project_root=project)
        cache = _FileCache()
        bridge.load_knowledge_files(cache)

        assert any("runbook" in v for v in cache._files.values())
