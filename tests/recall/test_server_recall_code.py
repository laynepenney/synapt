"""recall_code as an MCP tool and a CLI verb.

The composition itself (code_search.recall_code) is covered by
test_code_search.py; these tests pin the WIRING: the tool exists on the
server, it indexes the named repo by content hash before searching, it says
so when no symbol matched, and `synapt code` prints the same text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import synapt.recall.server as recall_server
from synapt.recall import code_index

pytestmark = pytest.mark.skipif(
    not code_index._parser_stack_available(),
    reason="tree-sitter-language-pack not installed",
)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A tiny repo whose data dir resolves under itself, with memory search stubbed."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    root = tmp_path / "tinyrepo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "refresh.py").write_text(
        "def cold_no_caller_refresh(index_dir):\n    return index_dir\n"
    )
    calls: list[str] = []

    def fake_search(query, **kwargs):
        calls.append(query)
        return "No results found."

    monkeypatch.setattr(
        "synapt.recall.code_search.recall_server.recall_search", fake_search
    )
    return root, calls


def test_tool_is_registered():
    names: list[str] = []

    class FakeMCP:
        def tool(self):
            def deco(fn):
                names.append(fn.__name__)
                return fn

            return deco

    recall_server.register_tools(FakeMCP())
    assert "recall_code" in names, names


def test_symbol_hit_carries_span_and_memory(repo):
    root, calls = repo
    out = recall_server.recall_code("cold no-caller refresh", repo_root=str(root))
    assert "function cold_no_caller_refresh" in out
    assert "pkg/refresh.py:1-2" in out
    assert "[exact on" in out
    assert "What the team said:" in out
    assert calls == ["cold no-caller refresh"]
    assert "memories only" not in out


def test_no_symbol_says_so(repo):
    root, _ = repo
    out = recall_server.recall_code("why is config push and resolve", repo_root=str(root))
    assert "No code symbol matched; memories only." in out
    assert "What the team said:" in out


def test_second_call_reparses_nothing_and_edit_reindexes(repo):
    root, _ = repo
    first = recall_server.recall_code("cold_no_caller_refresh", repo_root=str(root))
    assert "1 files re-parsed" in first
    second = recall_server.recall_code("cold_no_caller_refresh", repo_root=str(root))
    assert "0 files re-parsed, 1 unchanged" in second
    (root / "pkg" / "refresh.py").write_text("def warm_refresh(index_dir):\n    return 1\n")
    third = recall_server.recall_code("cold_no_caller_refresh", repo_root=str(root))
    assert "1 files re-parsed" in third
    assert "No code symbol matched" in third, third


def test_missing_root_is_an_error_line(tmp_path):
    out = recall_server.recall_code("anything", repo_root=str(tmp_path / "nope"))
    assert out.startswith("Repo root not found:")


def test_cli_code_verb_prints_the_same_text(repo, monkeypatch, capsys):
    root, _ = repo
    from synapt.recall import cli

    monkeypatch.setattr(sys, "argv", ["synapt", "code", "cold no-caller refresh", "--repo-root", str(root)])
    cli.main()
    printed = capsys.readouterr().out
    assert "function cold_no_caller_refresh" in printed
    assert "What the team said:" in printed
