"""Bounded live-tail parsing for transcript files.

``tail_turns`` reads only the trailing window of a live transcript —
Claude Code session jsonl or Codex CLI rollout jsonl — and returns the
last *n* utterances with honesty metadata. It states only what it
observed (``read_at``, ``source_path``, ``bytes_scanned``) and never a
freshness claim it did not establish; index-freshness labeling belongs
to the consumer.

Codex note: rollout files append to the session START-date file, so
callers resolve the live file by session id, never by a date-derived
guess. Path dates are start-order, not recency-order.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from synapt.recall.core import (
    SKIP_TYPES,
    _extract_assistant_content,
    _extract_user_text,
    _is_real_user_message,
)
from synapt.recall.scrub import scrub_text

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_BYTES = 256 * 1024


@dataclass
class TailTurn:
    """One utterance in a live tail: who said it, what, and when."""

    speaker: str
    text: str
    when: str


@dataclass
class TailView:
    """The result of one bounded tail read.

    ``truncated_head`` is True when the window started mid-file — the
    normal case for a live session, never a warning. ``turns`` may be
    empty even when ``bytes_scanned`` > 0: zero parsed turns from a
    nonempty window is an answer, not an error.
    """

    turns: list[TailTurn]
    read_at: str
    source_path: str
    bytes_scanned: int
    truncated_head: bool


def tail_turns(
    transcript_path: Path,
    n: int = 20,
    window_bytes: int = _DEFAULT_WINDOW_BYTES,
) -> TailView:
    """Parse the trailing window of a transcript into its last *n* turns.

    Reads at most ``window_bytes`` from the end of the file, drops the
    leading partial line when the window starts mid-line (a window that
    starts exactly on a line boundary keeps its complete first line),
    classifies each remaining line by shape (Claude Code entry or Codex
    rollout entry), and returns the last *n* utterances oldest-to-newest.

    Every producer routes through one guarded append point: a duplicate
    delivery of the same speaker+text inside the window is dropped no
    matter which entry kind produced it or in which order it arrived.

    A missing file raises ``FileNotFoundError`` — the path is the
    caller's claim, and this function does not soften a claim it can
    disprove. Unparseable content never raises: lines that are not JSON,
    or JSON of no known shape, are skipped.
    """
    path = Path(transcript_path)
    size = path.stat().st_size
    start = max(0, size - window_bytes)
    head_byte = b"\n"
    with path.open("rb") as f:
        if start > 0:
            # One byte of evidence before the window: a newline there means
            # the window starts exactly on a line boundary and its first
            # line is complete, not partial — dropping it would deny a turn
            # the window actually contained.
            f.seek(start - 1)
            head_byte = f.read(1)
        raw = f.read()

    bytes_scanned = len(raw)
    truncated_head = start > 0

    lines = raw.decode("utf-8", errors="replace").splitlines()
    if truncated_head and lines and head_byte != b"\n":
        lines = lines[1:]

    turns: list[TailTurn] = []

    def _append(speaker: str, text: str, when: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        if any(t.speaker == speaker and t.text == stripped for t in turns):
            return
        turns.append(TailTurn(speaker=speaker, text=stripped, when=when))

    def _scrubbed(text: str) -> str:
        try:
            return scrub_text(text)
        except Exception:
            logger.debug("scrub_text failed on tail text, using raw", exc_info=True)
            return text

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue

        entry_type = entry.get("type", "")
        when = entry.get("timestamp", "")

        # --- Claude Code session shapes ---
        if entry_type in SKIP_TYPES:
            continue
        if _is_real_user_message(entry):
            _append("user", _scrubbed(_extract_user_text(entry)), when)
            continue
        if entry_type == "assistant":
            text, _tools, _files, _tool_uses = _extract_assistant_content(entry)
            if text:
                _append("assistant", _scrubbed(text), when)
            continue

        # --- Codex rollout shapes ---
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        if entry_type == "response_item":
            role = payload.get("role", "")
            content = payload.get("content")
            texts: list[str] = []
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") not in ("input_text", "output_text"):
                        continue
                    block_text = block.get("text", "")
                    if block_text:
                        texts.append(block_text)
            joined = "\n".join(texts)
            if role == "user":
                _append("user", joined, when)
            elif role == "assistant":
                _append("assistant", joined, when)
        elif entry_type == "event_msg":
            msg_type = payload.get("type", "")
            if msg_type == "user_message":
                _append("user", payload.get("message", ""), when)
            elif msg_type == "agent_message":
                _append("assistant", payload.get("message", ""), when)

    return TailView(
        turns=turns[-n:] if n > 0 else [],
        read_at=datetime.now(timezone.utc).isoformat(),
        source_path=str(path),
        bytes_scanned=bytes_scanned,
        truncated_head=truncated_head,
    )
