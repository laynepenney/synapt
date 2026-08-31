"""Contract: a message the operator queues mid-turn is a user turn.

Typing while a turn is running does not produce a ``type: "user"`` line.
Claude Code writes a ``queue-operation`` line -- which the parser skips as
noise, correctly, since it carries no structured text -- plus an
``attachment`` line whose ``attachment.prompt`` holds what the human said.
Before this contract the parser read neither, so the operator's own words
never reached ``user_text``.

A query for something the operator said mid-turn could return the
assistant's later paraphrase and never the message itself.

``origin.kind`` is the safety boundary, so it also defines the cohort:
``human`` is the operator, ``peer`` is another agent's queued message, and
entries with no ``origin`` key at all are machine task-notifications.
Presence of an ``origin`` cannot distinguish authorship, since both
``human`` and ``peer`` have one; only ``origin.kind == "human"`` is
accepted.

Deliberately no corpus counts here.  An earlier version of this docstring
carried them, they were measured over the wrong cohort, and they went
stale twice more while the rest of the change was corrected -- a test
file is the last place anyone re-reads for a number.  The measured
figures live in the changelog and in the timestamped gate evidence,
where they carry the moment they belong to.

A duplicate cost exists and is small: a few queued messages also appear
verbatim as an ordinary turn, so reading the attachment indexes those
texts twice.  The cause is undetermined.
"""

from __future__ import annotations

import json

from synapt.recall.core import (
    _extract_user_text,
    _is_real_user_message,
    _queued_human_prompt,
)


def _queued(prompt, *, kind: str | None = "human", mode: str = "prompt") -> dict:
    """The shape Claude Code actually writes, trimmed to the read fields.

    ``kind=None`` omits the ``origin`` key entirely, which is how the runtime
    writes task notifications -- not ``origin: {"kind": ...}`` with some other
    value.  A fixture that invents an origin for them would test a shape that
    never occurs.
    """
    attachment = {
        "type": "queued_command",
        "commandMode": mode,
        "prompt": prompt,
        "timestamp": "2026-08-30T12:41:00.000Z",
    }
    if kind is not None:
        attachment["origin"] = {"kind": kind}
    return {
        "type": "attachment",
        "timestamp": "2026-08-30T12:41:00.000Z",
        "uuid": "e18f6172-e34a-4f00-9c11-000000000001",
        "attachment": attachment,
    }


def _plain(text: str) -> dict:
    return {
        "type": "user",
        "timestamp": "2026-08-30T12:40:00.000Z",
        "message": {"role": "user", "content": text},
    }


def test_a_queued_human_message_is_a_user_turn():
    entry = _queued("check the parser before the alpha work, please")
    assert _is_real_user_message(entry) is True
    assert _extract_user_text(entry) == "check the parser before the alpha work, please"


def test_a_plain_user_turn_still_reads_as_before():
    """The control. If this ever fails, the test above proves nothing: it
    would be passing in a world where every entry reads as a user turn."""
    entry = _plain("ordinary message typed between turns")
    assert _is_real_user_message(entry) is True
    assert _extract_user_text(entry) == "ordinary message typed between turns"


def test_both_forms_land_in_one_transcript_read():
    """The two forms together, which is the shape a real session has and the
    single-form fixtures cannot discriminate."""
    entries = [_plain("first, typed between turns"), _queued("second, typed during one")]
    texts = [_extract_user_text(e) for e in entries if _is_real_user_message(e)]
    assert texts == ["first, typed between turns", "second, typed during one"]


def test_a_queued_entry_from_a_PEER_agent_is_not_an_operator_turn():
    """``origin.kind`` is load-bearing, and ``peer`` is a real observed cohort:
    another agent's queued message. Indexing it would attribute an agent's
    words to the operator -- this contract's own failure, sign-flipped.

    Named explicitly so nobody later widens the guard to "has an origin".
    """
    entry = _queued("a message queued by another agent, not by a person", kind="peer")
    assert _queued_human_prompt(entry) is None
    assert _is_real_user_message(entry) is False


def test_a_task_notification_is_not_an_operator_turn():
    """The third cohort, and the one that produced a false public claim.

    The runtime writes background-task events as queued_command attachments
    with NO ``origin`` key and ``commandMode == "task-notification"``.  They
    are numerous -- 158 of 510 on the workspace this was measured on -- and
    their prompts are NOT empty.  An earlier draft counted them as things the
    operator said and then explained the gap by calling them empty text; both
    halves were wrong.  The guard already excluded them, which is why the code
    was right while the measurement was not.
    """
    entry = _queued(
        "Background command \"run the suite\" completed (exit code 0)",
        kind=None,
        mode="task-notification",
    )
    assert entry["attachment"].get("origin") is None, "the runtime omits the key entirely"
    assert _queued_human_prompt(entry) is None
    assert _is_real_user_message(entry) is False


def test_a_content_block_list_prompt_is_read():
    """``prompt`` is sometimes a list of content blocks rather than a string.

    This shape does not occur at all in the corpus the fix was first measured
    on, so no amount of care with that corpus could have surfaced it; it was
    found by a second reader measuring a different workspace, where 28 human
    queued prompts had it and were being dropped silently.  A shape your own
    data cannot produce is invisible to your own data.
    """
    entry = _queued([{"type": "text", "text": "first block"},
                     {"type": "text", "text": "second block"}])
    assert _queued_human_prompt(entry) == "first block\nsecond block"
    assert _is_real_user_message(entry) is True
    assert _extract_user_text(entry) == "first block\nsecond block"


def test_a_block_list_with_no_text_blocks_creates_no_turn():
    """The negative control for the branch above: reading the list form must
    not widen into "any list is a turn"."""
    entry = _queued([{"type": "image", "source": {}}])
    assert _queued_human_prompt(entry) is None
    assert _is_real_user_message(entry) is False


def test_only_text_blocks_are_read_from_a_block_list():
    """``block.get("type") == "text"`` must be load-bearing, and the fixture
    above cannot show that it is.

    An image block carries no ``text`` key, so dropping the type check leaves
    its behaviour identical -- measured: that mutation survived with the image
    fixture as the only negative. The discriminating case is a NON-TEXT block
    that DOES carry text. Without the type check its content would be indexed
    as the operator's own words, which is this contract's core failure wearing
    a different hat.
    """
    entry = _queued([
        {"type": "thinking", "text": "not something the operator said"},
        {"type": "text", "text": "what they actually typed"},
    ])
    assert _queued_human_prompt(entry) == "what they actually typed"


def test_an_empty_queued_prompt_creates_no_phantom_turn():
    """Matches the existing rule that a user entry with no text after
    stripping is not a turn: an empty prompt must not open one either."""
    assert _queued_human_prompt(_queued("   ")) is None
    assert _is_real_user_message(_queued("   ")) is False


def test_a_non_queued_attachment_is_untouched():
    """Attachments are a broad category. Only the queued_command form is a
    user turn; widening past that would index file contents as speech.

    The fixture carries a human ``origin`` and a ``prompt`` DELIBERATELY, so
    ``attachment.type`` is the only guard that can reject it.  The obvious
    fixture -- a bare ``file_content`` attachment -- is rejected by the later
    ``origin`` check instead, so it passes whether or not the type check
    exists and pins nothing.  Measured: with that fixture, mutating the type
    check to ``is None`` left all seven tests green.
    """
    entry = {
        "type": "attachment",
        "attachment": {
            "type": "file_content",
            "origin": {"kind": "human"},
            "prompt": "README.md contents that are not speech",
        },
    }
    assert _queued_human_prompt(entry) is None
    assert _is_real_user_message(entry) is False


def test_queue_operation_lines_remain_skipped():
    """The sibling line for the same event. It carries the raw text with no
    structure, so reading BOTH would double every queued message."""
    from synapt.recall.core import SKIP_TYPES

    assert "queue-operation" in SKIP_TYPES

# --- the transcript RENDERER, a separate consumer with its own wiring -------
#
# `_read_raw_turn` reads `message.content` directly and does not inherit the
# `_is_real_user_message` / `_extract_user_text` fix.  It needed its own branch,
# and therefore needs its own witness: a reviewer removed that four-line branch
# on a reconstructed head and every other test in this file stayed green.  A
# consumer with no witness is indistinguishable from a consumer that does not
# exist.


def _chunk_over(tmp_path, lines):
    """A production-shaped chunk: a real JSONL file, real byte offsets."""
    from synapt.recall.core import TranscriptChunk

    path = tmp_path / "session.jsonl"
    blob = "".join(json.dumps(line) + "\n" for line in lines).encode("utf-8")
    path.write_bytes(blob)
    return TranscriptChunk(
        id="c1",
        session_id="s1",
        timestamp="2026-08-30T12:41:00.000Z",
        turn_index=0,
        user_text="",
        assistant_text="",
        commentary_text="",
        tools_used=[],
        files_touched=[],
        tool_content="",
        date_text="",
        transcript_path=str(path),
        byte_offset=0,
        byte_length=len(blob),
    )


def test_the_renderer_emits_a_queued_human_message_as_a_user_turn(tmp_path):
    """Drives `_read_raw_turn` itself, over bytes on disk.

    The ordinary turn is the control: if the renderer broke entirely, both
    would vanish and this would still fail for the wrong reason.
    """
    from synapt.recall.core import TranscriptIndex

    chunk = _chunk_over(
        tmp_path,
        [_plain("typed between turns"), _queued("typed during one")],
    )
    rendered = TranscriptIndex._read_raw_turn(chunk)
    assert "typed between turns" in rendered, "control: the ordinary turn must render"
    assert "typed during one" in rendered, "the queued turn must reach the renderer"


def test_the_renderer_does_not_emit_a_task_notification(tmp_path):
    """The renderer must apply the same authorship boundary as the parser.
    Rendering a machine notification as `[User]` would show the operator
    words they never wrote."""
    from synapt.recall.core import TranscriptIndex

    chunk = _chunk_over(
        tmp_path,
        [_queued("Background command completed (exit code 0)", kind=None,
                 mode="task-notification")],
    )
    rendered = TranscriptIndex._read_raw_turn(chunk)
    assert "Background command completed" not in rendered
