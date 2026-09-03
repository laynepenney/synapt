"""``recall_save`` must reject an unrecognized ``category`` loudly, not coerce it.

Found during a stranger-onboarding dogfood: ``recall_save(category="note")``
silently saved the node with category "workflow" instead -- "note" is not
one of the eleven values in ``VALID_CATEGORIES``
(``synapt.recall.knowledge``), and the tool's own docstring ("workflow,
tooling, decision, etc.") gives no hint the set is closed. The success
message reported "workflow" back as if that were the caller's own choice.

Same class of defect as the MCP unknown-ARGUMENT-name fix in
``test_server_arg_validation.py`` (recall_journal accepted a misnamed field
and silently dropped its content) -- here it is a recognized argument NAME
with an unrecognized VALUE, one level deeper, and that fix does not reach
it: FastMCP's schema validation only checks which keys exist, never what a
string-typed value contains.

The fix stays inside ``recall_save`` (server.py), not
``KnowledgeNode.create()``'s existing silent-fallback line: several internal
consolidation call sites (``consolidate.py``) pass an LLM-derived category
that must keep degrading gracefully rather than crashing an unattended
background job, so the create()-level coercion for THOSE callers is
deliberately kept. The MCP tool boundary is where a human is present to
read and correct an error, so that is where the loud rejection belongs.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from synapt.recall.knowledge import VALID_CATEGORIES
from synapt.recall.server import ValidatingFastMCP, register_tools


def _call_tool(name: str, **kwargs):
    mcp = ValidatingFastMCP(f"test-synapt-recall-save-category-{name}")
    register_tools(mcp)
    content_blocks, _structured = asyncio.run(mcp.call_tool(name, kwargs))
    return content_blocks[0].text


def test_unrecognized_category_is_a_clean_error_naming_the_valid_set(tmp_path):
    with patch("synapt.recall.server.Path.cwd", return_value=tmp_path):
        text = _call_tool("recall_save", content="a fact worth keeping", category="note")
    assert "Error" in text
    assert "note" in text
    # A caller correcting the mistake needs the real set, not a vague hint.
    for valid in sorted(VALID_CATEGORIES):
        assert valid in text


def test_unrecognized_category_writes_nothing(tmp_path):
    """The rejected call must not silently save under a different category --
    the whole point is that a caller's mistake should be visible, not quietly
    absorbed into a node they never asked for."""
    with patch("synapt.recall.server.Path.cwd", return_value=tmp_path):
        _call_tool("recall_save", content="unique-marker-mkc7f2", category="note")
        search_text = _call_tool("recall_search", query="unique-marker-mkc7f2")
    assert "unique-marker-mkc7f2" not in search_text


def test_recognized_category_is_unaffected(tmp_path):
    with patch("synapt.recall.server.Path.cwd", return_value=tmp_path):
        text = _call_tool("recall_save", content="a real preference", category="preference")
    assert "preference" in text
    assert "Error" not in text
