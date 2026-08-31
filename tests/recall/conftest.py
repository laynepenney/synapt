"""Shared fixtures for synapt recall tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synapt.recall.core import TranscriptChunk


# ---------------------------------------------------------------------------
# Synthetic Claude Code transcript entries
# ---------------------------------------------------------------------------

def user_text_entry(text: str, uuid: str = "u1", ts: str = "2026-02-28T10:00:00Z"):
    """A real user message with string content."""
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": "test-session-001",
        "message": {"role": "user", "content": text},
    }


def user_text_list_entry(text: str, uuid: str = "u2", ts: str = "2026-02-28T10:01:00Z"):
    """A real user message with list content containing a text block."""
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": "test-session-001",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    }


def user_tool_result_entry(
    content: str = "file contents here...",
    uuid: str = "u3",
    ts: str = "2026-02-28T10:02:00Z",
):
    """A user entry that is only a tool_result (NOT a real user message)."""
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": "test-session-001",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_123",
                    "content": content,
                }
            ],
        },
    }


def assistant_entry(
    text: str = "",
    tool_name: str = "",
    file_path: str = "",
    tool_input: dict | None = None,
    tool_use_id: str = "toolu_test001",
    uuid: str = "a1",
    ts: str = "2026-02-28T10:00:30Z",
):
    """An assistant message with optional text, tool use, and thinking."""
    content = []
    content.append({"type": "thinking", "thinking": "internal reasoning..."})
    if text:
        content.append({"type": "text", "text": text})
    if tool_name:
        inp = tool_input or {}
        if file_path and "file_path" not in inp:
            inp["file_path"] = file_path
        content.append({
            "type": "tool_use",
            "id": tool_use_id,
            "name": tool_name,
            "input": inp,
        })
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": "test-session-001",
        "message": {"role": "assistant", "content": content},
    }


def tool_result_entry(
    tool_use_id: str = "toolu_test001",
    result: str = "",
    uuid: str = "tr1",
    ts: str = "2026-02-28T10:00:31Z",
):
    """A user message containing a tool_result block."""
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": "test-session-001",
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


def progress_entry(uuid: str = "p1"):
    return {"type": "progress", "uuid": uuid, "data": "..."}


def system_entry(uuid: str = "s1"):
    return {"type": "system", "uuid": uuid, "message": {"role": "system", "content": "..."}}


def write_jsonl(path: Path, entries: list[dict]):
    # Use newline="" to prevent \r\n on Windows — byte offsets must match binary reads
    with open(path, "w", encoding="utf-8", newline="") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def chatgpt_message(
    role: str,
    text: str = "",
    *,
    content_type: str = "text",
    create_time: float = 1769000000.0,
    recipient: str = "all",
    author_name: str | None = None,
):
    content = {"content_type": content_type}
    if content_type == "code":
        content["text"] = text
    elif content_type == "multimodal_text":
        content["parts"] = [{"content_type": "audio_transcription", "text": text}]
    elif content_type == "user_editable_context":
        content["user_profile"] = text
    else:
        content["parts"] = [text]

    author = {"role": role}
    if author_name:
        author["name"] = author_name

    return {
        "id": f"msg-{role}-{abs(hash((role, text, create_time))) % 100000}",
        "author": author,
        "content": content,
        "create_time": create_time,
        "update_time": create_time,
        "recipient": recipient,
        "status": "finished_successfully",
    }


def make_test_chunks() -> list[TranscriptChunk]:
    """Create a set of test chunks for index testing."""
    return [
        TranscriptChunk(
            id="sess1:t0", session_id="session-aaa",
            timestamp="2026-02-26T10:00:00Z", turn_index=0,
            user_text="how does the quality curve work?",
            assistant_text="The quality curve uses a Hermite spline with three zones.",
        ),
        TranscriptChunk(
            id="sess1:t1", session_id="session-aaa",
            timestamp="2026-02-26T10:05:00Z", turn_index=1,
            user_text="what about Cat3 weighting?",
            assistant_text="Cat3 examples get weighted by output closeness. Zone 2 is a plateau.",
            tools_used=["Read"],
            files_touched=["scripts/verify_quality_curve.py"],
        ),
        TranscriptChunk(
            id="sess2:t0", session_id="session-bbb",
            timestamp="2026-02-28T14:00:00Z", turn_index=0,
            user_text="fix the harness bug in swift tests",
            assistant_text="The harness had two bugs: XCTAssertNil crashed on correct nil values.",
            tools_used=["Edit", "Bash"],
            files_touched=["src/graph/swift_errors.py"],
        ),
        TranscriptChunk(
            id="sess2:t1", session_id="session-bbb",
            timestamp="2026-02-28T14:05:00Z", turn_index=1,
            user_text="run the eval again",
            assistant_text="Running eval on Batman t51-100 with the fixed harness.",
            tools_used=["Bash"],
        ),
    ]


@pytest.fixture(autouse=True)
def _isolate_recall_root_env(monkeypatch):
    """Recall tests measure path INFERENCE unless a test sets the override
    itself. An ambient SYNAPT_RECALL_ROOT — or now GRIPSPACE_ROOT, which gr
    exports on every spawn and recall consults in the same None-branch — from
    the invoking shell would redirect every cwd-based fixture into a live store:
    green in CI, where nothing sets it, and red or silently store-polluting on
    any agent machine where gr set it. The green-where-checked inversion,
    prevented at birth. A test that means to exercise an env root sets it back.
    """
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)


@pytest.fixture
def owned_recall_root(tmp_path, monkeypatch) -> Path:
    """Point implicit recall data resolution at a directory this test owns.

    Ref #967. The autouse fixture above strips an *ambient* override so these
    tests measure inference rather than a value the shell happened to set —
    that intent is right and is preserved. This fixture is the other half it
    already anticipated: "unless a test sets the override itself." Requesting
    it is a test declaring that it needs a store, not that it is measuring
    where one is inferred from.

    Use this when the test needs recall data to exist somewhere. When the test
    is genuinely asserting *inference* behaviour, prefer ``monkeypatch.chdir``
    to an owned directory instead, so the inference still runs — just from a
    starting point the test owns rather than from the operator's checkout.

    The directory is created before it is handed over, because the override
    refuses a root that does not exist: silently minting a fresh store under a
    mistyped path would present an empty history as a real answer.
    """
    root = tmp_path / "recall-root"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(root))
    monkeypatch.setenv("SYNAPT_RECALL_WORKTREE", "pytest-owned")
    return root
