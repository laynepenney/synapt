"""Reject unknown MCP tool-call arguments instead of silently dropping them.

A caller who misnames a keyword field (``accomplishments`` instead of
``done`` on ``recall_journal``) got a success response with the content
silently missing -- no warning, no error, the field just vanished.

The tool FUNCTION itself never sees the mismatched field: FastMCP builds a
per-tool Pydantic model from the function's signature and validates the raw
call arguments against THAT model before ever invoking the function, and its
default is to silently drop an unrecognized field -- so no amount of
validation inside a tool function itself could ever catch this.

The fix is ``ValidatingFastMCP`` (in ``synapt.recall.server``): a FastMCP
subclass whose overridden ``call_tool`` checks the raw arguments against
each tool's own published schema (``list_tools()`` / ``Tool.inputSchema``,
both public MCP wire-protocol surface) before delegating to the real
dispatch. It has to be the class *constructed*, not applied to an
already-built ``FastMCP`` instance afterward -- ``FastMCP.__init__`` binds
``self.call_tool`` as the wire handler before ``register_tools()`` ever sees
the instance, so the tests below build a ``ValidatingFastMCP`` directly
rather than patching one post-construction.
"""

from __future__ import annotations

import asyncio

import pytest

from synapt.recall.server import ValidatingFastMCP, register_tools


def _sample_tool(action: str = "read", focus: str | None = None, done: str | None = None) -> str:
    """A minimal stand-in with the same optional-keyword shape as recall_journal.

    Registered through the SAME register_tools() call as every real synapt
    tool, so it proves the fix applies at the server level -- not just to
    one hand-picked function -- without needing a real store.
    """
    return f"action={action} focus={focus} done={done}"


def _unnamed_tool(text: str) -> str:
    """A tool this fix's issue never named, to prove global coverage rather
    than a per-tool allowlist."""
    return f"text={text}"


def _build_server() -> ValidatingFastMCP:
    mcp = ValidatingFastMCP("test-synapt-arg-validation")
    register_tools(mcp)
    mcp.add_tool(_sample_tool)
    mcp.add_tool(_unnamed_tool)
    return mcp


def test_unknown_argument_is_rejected_with_the_field_named():
    mcp = _build_server()
    with pytest.raises(Exception) as exc_info:
        asyncio.run(mcp.call_tool(
            "_sample_tool",
            {"action": "write", "accomplishments": "did the thing", "focus": "x"},
        ))
    # The error must name the OFFENDING field, not just say "something went
    # wrong" -- a caller correcting a typo needs the field name.
    assert "accomplishments" in str(exc_info.value)


def test_known_arguments_are_unaffected():
    mcp = _build_server()
    content_blocks, _structured = asyncio.run(
        mcp.call_tool("_sample_tool", {"action": "write", "done": "did the thing"})
    )
    text = content_blocks[0].text
    assert "action=write" in text
    assert "done=did the thing" in text


def test_a_tool_never_named_by_this_fix_is_covered_too():
    """The fix is a property of the SERVER, not a per-tool allowlist -- a
    tool nobody thought to check gets the same protection."""
    mcp = _build_server()
    with pytest.raises(Exception) as exc_info:
        asyncio.run(mcp.call_tool(
            "_unnamed_tool", {"text": "hi", "extra_junk": 1},
        ))
    assert "extra_junk" in str(exc_info.value)

    content_blocks, _structured = asyncio.run(
        mcp.call_tool("_unnamed_tool", {"text": "hi"})
    )
    assert content_blocks[0].text == "text=hi"


def test_recall_journal_itself_rejects_the_exact_misname(tmp_path):
    """End-to-end through the real registration path, not the stand-in --
    proves the fix reaches recall_journal specifically, isolated so nothing
    touches a real store."""
    from _isolation_helpers import owned_store

    store = owned_store()
    try:
        mcp = ValidatingFastMCP("test-synapt-journal")
        register_tools(mcp)
        with pytest.raises(Exception) as exc_info:
            asyncio.run(mcp.call_tool(
                "recall_journal",
                {
                    "action": "write",
                    "focus": "session focus",
                    "accomplishments": "did the thing",
                    "decisions": "",
                    "next_steps": "",
                },
            ))
        assert "accomplishments" in str(exc_info.value)
    finally:
        store.restore()


def test_recall_channel_and_recall_remind_are_covered_too():
    """Whether the same silent acceptance applies to other synapt MCP tools
    with optional fields, naming recall_channel and recall_remind
    specifically. The fix is at the server-construction level (whichever
    ValidatingFastMCP instance the caller builds), not a per-tool wrapper,
    so both are covered by construction -- this closes the question rather
    than leaving it open."""
    mcp = ValidatingFastMCP("test-synapt-other-tools")
    register_tools(mcp)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(mcp.call_tool(
            "recall_remind", {"action": "list", "bogus_field": "x"},
        ))
    assert "bogus_field" in str(exc_info.value)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(mcp.call_tool(
            "recall_channel", {"action": "read", "bogus_field": "x"},
        ))
    assert "bogus_field" in str(exc_info.value)


def test_a_plain_fastmcp_instance_is_unprotected():
    """Documents the catch named in ValidatingFastMCP's docstring: the
    protection is a property of which class gets CONSTRUCTED, not something
    register_tools() can retrofit onto an already-built plain FastMCP."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("test-synapt-unprotected")
    register_tools(mcp)
    content_blocks, _structured = asyncio.run(
        mcp.call_tool(
            "recall_journal",
            {"action": "write", "accomplishments": "silently dropped"},
        )
    )
    # No exception: this is the ORIGINAL bug, preserved here as a control so
    # a future FastMCP upgrade that changes this default trips a visible
    # failure here rather than in ValidatingFastMCP's tests.
    assert content_blocks[0].text is not None
