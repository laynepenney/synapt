"""R3 "Memory Everywhere" first fruit: admit and index an agent's own
Claude Code memory directory as a recall source, and answer a query
through it without bulk-loading the files into knowledge nodes.

TDD: these tests fail until claude_memory_source.py ships.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synapt.recall.core import _gripspace_cache, project_data_dir
from synapt.recall.source_index import (
    SOURCE_INDEX_SUPPORTED,
    SourceSearchRequest,
    _clear_source_search_providers,
    search_registered_sources,
)

pytestmark = pytest.mark.skipif(
    not SOURCE_INDEX_SUPPORTED,
    reason="descriptor adapter is POSIX-only: os.O_DIRECTORY unavailable",
)


def _make_gripspace(root: Path, name: str = "gripspace") -> Path:
    grip = root / name
    (grip / ".gitgrip").mkdir(parents=True)
    (grip / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
    return grip


def _write_memory_files(home: Path, gripspace_root: Path) -> Path:
    """Lay down a MEMORY.md plus one linked topic file at the exact path
    Claude Code itself would use for this gripspace root, under a fake
    ``home``. No markdown-link-following code is exercised or needed:
    DescriptorSourceAdapter already walks every .md file under the
    admitted root."""
    from synapt.recall.core import project_slug

    slug = project_slug(gripspace_root)
    memory_dir = home / ".claude" / "projects" / slug / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "MEMORY.md").write_text(
        "# MEMORY\n\n- [Vorn-mat lane](vorn_mat_lane.md) — publication boundary notes\n",
        encoding="utf-8",
    )
    # The query phrase deliberately does not end the unit's content: a
    # measured source_index tokenizer quirk (tokenchars + porter glue
    # trailing punctuation onto the LAST token of a unit's content, so a
    # bare query for that literal final word silently fails to match --
    # filed separately, out of scope for this fix) means a fixture whose
    # matching phrase is content-final gives a false negative unrelated to
    # anything this module does.
    (memory_dir / "vorn_mat_lane.md").write_text(
        "---\nname: vorn-mat-lane\n---\n\n"
        "Phase 3 active-eviction work stays private until Layne reratifies "
        "scope, see the tracker for the current status.\n",
        encoding="utf-8",
    )
    return memory_dir


@pytest.fixture(autouse=True)
def _isolated_registry_and_cache():
    _clear_source_search_providers()
    _gripspace_cache.clear()
    yield
    _clear_source_search_providers()
    _gripspace_cache.clear()


class TestAdmitAndIndexClaudeMemory:
    def test_no_gripspace_returns_none_without_error(self, tmp_path, monkeypatch):
        from synapt.recall.claude_memory_source import admit_and_index_claude_memory

        outside = tmp_path / "not-a-gripspace"
        outside.mkdir()
        monkeypatch.chdir(outside)
        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)

        assert admit_and_index_claude_memory() is None

    def test_gripspace_with_no_memory_dir_returns_none_without_error(
        self, tmp_path, monkeypatch
    ):
        from synapt.recall.claude_memory_source import admit_and_index_claude_memory

        grip = _make_gripspace(tmp_path)
        monkeypatch.chdir(grip)
        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "empty-home")

        assert admit_and_index_claude_memory() is None

    def test_indexes_memory_dir_and_answers_query_through_registered_search(
        self, tmp_path, monkeypatch
    ):
        """The thinnest slice: one admitted directory (MEMORY.md + one linked
        topic file), synced, registered, and found through the SAME bridge
        recall_search already composes over (search_registered_sources) --
        not the low-level search_source, since that composition is the part
        with zero real production callers today."""
        from synapt.recall.claude_memory_source import admit_and_index_claude_memory

        grip = _make_gripspace(tmp_path)
        home = tmp_path / "home"
        _write_memory_files(home, grip)
        monkeypatch.chdir(grip)
        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        monkeypatch.setattr(Path, "home", lambda: home)

        receipt = admit_and_index_claude_memory()
        assert receipt is not None
        assert receipt.state == "complete"
        assert receipt.documents_seen == 2

        results = search_registered_sources(
            SourceSearchRequest(query="active-eviction reratifies scope", limit=5)
        )
        assert len(results) == 1
        result = results[0]
        assert result.source_kind == "claude_memory"
        assert result.relative_path == "vorn_mat_lane.md"
        assert "reratifies" in result.content

    def test_cwd_outside_gripspace_with_gripspace_root_env_finds_same_file(
        self, tmp_path, monkeypatch
    ):
        """The CWD-independence claim: a caller whose process cwd is nowhere
        near the gripspace, with only GRIPSPACE_ROOT pointing back at it,
        must resolve and index the identical memory directory -- this is the
        exact shape that bit the channel store before today's fix."""
        from synapt.recall.claude_memory_source import admit_and_index_claude_memory

        grip = _make_gripspace(tmp_path)
        home = tmp_path / "home"
        _write_memory_files(home, grip)
        scratchpad = tmp_path / "scratchpad"
        scratchpad.mkdir()
        monkeypatch.chdir(scratchpad)
        monkeypatch.setenv("GRIPSPACE_ROOT", str(grip))
        monkeypatch.setattr(Path, "home", lambda: home)

        receipt = admit_and_index_claude_memory()
        assert receipt is not None
        assert receipt.state == "complete"
        assert receipt.documents_seen == 2

        results = search_registered_sources(
            SourceSearchRequest(query="active-eviction reratifies scope", limit=5)
        )
        assert len(results) == 1
        assert results[0].relative_path == "vorn_mat_lane.md"

        # And the store landed under the GRIPSPACE_ROOT coordinate, not the
        # scratchpad cwd -- same guarantee project_data_dir already gives
        # every other recall store.
        db_path = project_data_dir(grip) / "source_index" / "claude_memory.db"
        assert db_path.exists()

    def test_second_call_reuses_unchanged_documents(self, tmp_path, monkeypatch):
        from synapt.recall.claude_memory_source import admit_and_index_claude_memory

        grip = _make_gripspace(tmp_path)
        home = tmp_path / "home"
        _write_memory_files(home, grip)
        monkeypatch.chdir(grip)
        monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
        monkeypatch.setattr(Path, "home", lambda: home)

        first = admit_and_index_claude_memory()
        second = admit_and_index_claude_memory()
        assert first.state == "complete"
        assert second.state == "complete"
        assert second.documents_reused == 2


class TestSyncClaudeMemorySourceOnStartup:
    """server._sync_claude_memory_source_on_startup(): the eager-at-startup
    wiring point, exercised in isolation from a real FastMCP instance and
    a real memory directory (register_tools() itself must NOT trigger this
    -- several tests call register_tools() directly against whatever
    memory directory happens to exist on the machine running the suite)."""

    def test_no_receipt_logs_nothing(self, monkeypatch, capsys):
        from synapt.recall import server

        monkeypatch.setattr(
            "synapt.recall.claude_memory_source.admit_and_index_claude_memory",
            lambda: None,
        )
        server._sync_claude_memory_source_on_startup()
        assert capsys.readouterr().err == ""

    def test_receipt_logs_one_line_naming_state_files_and_generation(
        self, monkeypatch, capsys
    ):
        from synapt.recall import server
        from synapt.recall.source_index import SourceScanReceipt

        monkeypatch.setattr(
            "synapt.recall.claude_memory_source.admit_and_index_claude_memory",
            lambda: SourceScanReceipt(
                scan_id="scan_test", state="complete", generation=1,
                documents_seen=2, units_published=2, documents_reused=0,
            ),
        )
        server._sync_claude_memory_source_on_startup()
        err = capsys.readouterr().err
        assert "[claude_memory]" in err
        assert "complete" in err
        assert "2 file(s)" in err
        assert "generation 1" in err

    def test_admission_exception_never_raises_or_blocks_startup(
        self, monkeypatch, capsys
    ):
        from synapt.recall import server

        def _boom():
            raise RuntimeError("disk exploded")

        monkeypatch.setattr(
            "synapt.recall.claude_memory_source.admit_and_index_claude_memory", _boom
        )
        server._sync_claude_memory_source_on_startup()  # must not raise
        assert capsys.readouterr().err == ""

    def test_register_tools_alone_does_not_trigger_a_real_memory_scan(
        self, monkeypatch
    ):
        """The regression this class exists to prevent: register_tools()
        is called directly by several other test files against a real
        FastMCP instance, and must never reach a real filesystem scan."""
        import synapt.recall.server as server

        called = []
        monkeypatch.setattr(
            "synapt.recall.claude_memory_source.admit_and_index_claude_memory",
            lambda: called.append(True),
        )

        class _FakeMCP:
            def tool(self):
                return lambda fn: fn

        server.register_tools(_FakeMCP())
        assert called == []
