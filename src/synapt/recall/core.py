"""synapt.recall — persistent conversational memory for Claude Code sessions.

Parses Claude Code transcript archives (.jsonl), chunks them into semantic
turn-level units, indexes them with BM25 (+optional embeddings), and enables
time-ordered retrieval of past session context.

Usage:
    from synapt.recall import parse_transcript, TranscriptIndex

    chunks = parse_transcript(Path("~/.claude/projects/.../session.jsonl"))
    index = TranscriptIndex(chunks)
    context = index.lookup("quality curve weighting", max_chunks=3)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from synapt.recall.scrub import scrub_text

logger = logging.getLogger("synapt.recall")

from synapt.recall.bm25 import BM25, _tokenize
from synapt.recall.hybrid import augment_query_for_intent, extract_entities
from synapt.recall.storage import RecallDB
from synapt.recall.sharded_db import ShardedRecallDB

# Multiplier for knowledge nodes whose content matches query entities
ENTITY_BOOST = 1.5

# Category-intent alignment map: which knowledge node categories are most
# relevant for each query intent. Used to boost aligned nodes by 1.5×.
_CAT_INTENT_MAP: dict[str, set[str]] = {
    "decision": {"decision", "lesson-learned"},
    "debug": {"debugging", "lesson-learned"},
    "factual": {"fact", "infrastructure", "convention"},
    "procedural": {"workflow", "convention"},
    "aggregation": {"fact", "preference"},
}


def format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string (KB or MB)."""
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    return f"{size_bytes / 1_000:.1f} KB"


def atomic_json_write(data: dict | list, path: Path, indent: int = 2) -> None:
    """Write JSON to *path* atomically via temp-file + rename."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _build_date_text(timestamp: str) -> str:
    """Expand an ISO-ish timestamp into date terms useful for FTS/BM25."""
    if not timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    day_num = str(dt.day)
    day_padded = dt.strftime("%d")
    month_full = dt.strftime("%B")
    month_abbrev = dt.strftime("%b")
    weekday = dt.strftime("%A")
    year = dt.strftime("%Y")
    month_num = dt.strftime("%m")
    iso_date = dt.strftime("%Y-%m-%d")
    return " ".join(
        [
            year,
            f"{year}-{month_num}",
            iso_date,
            month_full,
            month_abbrev,
            day_num,
            day_padded,
            weekday,
            f"{month_full} {year}",
            f"{month_abbrev} {year}",
            f"{weekday} {month_full} {day_num} {year}",
        ]
    )


# Near-duplicate Jaccard threshold for result deduplication.
# 0.70 catches near-identical chunks whose formatted blocks diverge
# slightly due to metadata (context lines, turn headers).
# Lower values (0.6) aggressively remove "similar" chunks that contain
# different critical details — this caused a -4pp regression on both
# LOCOMO and CodeMemo benchmarks (see #459).
_DEDUP_JACCARD_THRESHOLD = 0.70


def _env_flag(name: str) -> bool:
    """Return True when an ablation env flag is enabled."""
    value = os.environ.get(name, "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _dedup_limit(max_chunks: int) -> int:
    """Extra candidates to pass through formatting to compensate for dedup."""
    return max_chunks + max(5, max_chunks // 3)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TranscriptChunk:
    """One turn in a Claude Code conversation."""

    id: str  # "{session_short}:t{turn_index}"
    session_id: str  # Full UUID
    timestamp: str  # ISO 8601
    turn_index: int  # 0-based within session
    user_text: str  # Cleaned user message
    assistant_text: str  # Concatenated assistant text blocks
    commentary_text: str = ""  # Intermediate assistant commentary, not searchable
    tools_used: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    tool_content: str = ""  # Summarized tool inputs + results
    date_text: str = ""  # Timestamp expanded into FTS-friendly calendar text
    transcript_path: str = ""  # Path to source transcript file
    byte_offset: int = -1  # Byte position where this turn starts in raw JSONL
    byte_length: int = 0  # Total bytes of raw JSONL entries for this turn
    text: str = ""  # Combined searchable text (built at init)
    agent_id: str | None = None  # Agent that created this chunk (from SYNAPT_AGENT_ID)

    def __post_init__(self):
        if self.agent_id is None:
            self.agent_id = os.environ.get("SYNAPT_AGENT_ID")
        if not self.date_text:
            self.date_text = _build_date_text(self.timestamp)
        if not self.text:
            self.text = self._build_text()

    def _build_text(self) -> str:
        parts = []
        if self.user_text:
            parts.append(self.user_text)
        if self.assistant_text:
            parts.append(self.assistant_text)
        if self.tool_content:
            parts.append(self.tool_content)
        if self.tools_used:
            parts.append(" ".join(self.tools_used))
        if self.files_touched:
            parts.append(" ".join(self.files_touched))
        if self.date_text:
            parts.append(self.date_text)
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "turn_index": self.turn_index,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "commentary_text": self.commentary_text,
            "tools_used": self.tools_used,
            "files_touched": self.files_touched,
            "tool_content": self.tool_content,
            "date_text": self.date_text,
            "transcript_path": self.transcript_path,
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TranscriptChunk:
        return cls(
            id=d["id"],
            session_id=d["session_id"],
            timestamp=d["timestamp"],
            turn_index=d["turn_index"],
            user_text=d["user_text"],
            assistant_text=d["assistant_text"],
            commentary_text=d.get("commentary_text", ""),
            tools_used=d.get("tools_used", []),
            files_touched=d.get("files_touched", []),
            tool_content=d.get("tool_content", ""),
            date_text=d.get("date_text", ""),
            transcript_path=d.get("transcript_path", ""),
            byte_offset=d.get("byte_offset", -1),
            byte_length=d.get("byte_length", 0),
        )


@dataclass
class SearchDiagnostics:
    """Diagnostic info collected when a search returns no results."""

    total_chunks: int = 0
    total_sessions: int = 0
    candidates_found: int = 0        # Chunks with score > 0 (before threshold)
    search_mode: str = ""            # fts_global, bm25_global, fts_progressive, bm25_progressive
    sessions_searched: int = 0       # Progressive mode only
    date_filter_active: bool = False
    embeddings_available: bool = True  # False when search ran without embeddings
    semantic_search_used: bool = False
    reason: str = ""                 # empty_index, empty_query, no_matches
    oldest_indexed_at: str | None = None

    def format_message(self) -> str:
        """Format diagnostic information as a user-facing message."""
        if self.reason == "empty_index":
            return "No results found — index is empty. Run `recall_setup` or `recall_build` first."

        if self.reason == "empty_query":
            return "No results found — query produced no search tokens."

        if self.reason == "no_matches":
            lines = ["No results found — no chunks matched your query terms."]
            lines.append(f"  Index: {self.total_chunks} chunks across {self.total_sessions} sessions")
            if not self.embeddings_available:
                lines.append("  Warning: semantic search unavailable — using keyword matching only.")
                lines.append("  Broad or abstract queries may not match. Try exact terms from conversations.")
            if self.date_filter_active:
                lines.append("  Note: date filter was active — try widening the date range")
            lines.append("  Try: broader terms, different keywords, or check `recall_stats`")
            return "\n".join(lines)

        return "No results found."


@dataclass
class SearchResultSummary:
    """Summary of what a successful search emitted."""

    intent: str = ""
    max_knowledge: int | None = None
    selected_blocks: int = 0
    chunk_blocks: int = 0
    knowledge_blocks: int = 0
    cluster_blocks: int = 0


def _oldest_indexed_date(chunks: list[TranscriptChunk]) -> str | None:
    """Return the oldest valid chunk timestamp as a UTC calendar date."""
    dates: list[str] = []
    for chunk in chunks:
        if not chunk.timestamp:
            continue
        try:
            timestamp = datetime.fromisoformat(
                chunk.timestamp.replace("Z", "+00:00")
            )
        except (TypeError, ValueError):
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        dates.append(timestamp.astimezone(timezone.utc).date().isoformat())
    return min(dates, default=None)


# ---------------------------------------------------------------------------
# Transcript parser
# ---------------------------------------------------------------------------

SKIP_TYPES = {"progress", "file-history-snapshot", "queue-operation", "system"}

# Patterns that indicate an error in Bash output (used to give errors 2x budget).
_ERROR_PATTERNS = re.compile(
    r"(?:^|\n)(?:Error|Traceback|FAILED|FATAL|panic:|error\[E)"
    r"|(?:exit code [1-9]|Exit code [1-9]|returned non-zero)"
    r"|(?:ModuleNotFoundError|ImportError|SyntaxError|NameError"
    r"|TypeError|ValueError|KeyError|AttributeError|FileNotFoundError)",
)


def _summarize_tool_result(
    tool_name: str,
    tool_input: dict,
    result_text: str,
) -> str:
    """Summarize a tool result into a compact searchable string.

    Different tools get different budgets — Bash errors are high-value,
    Read file contents are low-value (file exists in repo).
    """
    if not result_text:
        return ""

    name = tool_name.lower()
    # Strip common MCP prefixes (lowered for routing, original for display)
    short = name.rsplit("__", 1)[-1] if "__" in name else name
    display_name = tool_name.rsplit("__", 1)[-1] if "__" in tool_name else tool_name

    if short == "bash":
        cmd = tool_input.get("command", "")
        is_error = bool(_ERROR_PATTERNS.search(result_text))
        cap = 1200 if is_error else 600
        output = result_text[:cap]
        if len(result_text) > cap:
            output += "..."
        return f"$ {cmd}\n{output}" if cmd else output

    if short == "read":
        path = tool_input.get("file_path", "")
        n_lines = result_text.count("\n") + 1
        return f"Read {path} ({n_lines} lines)"

    if short == "write":
        path = tool_input.get("file_path", "")
        return f"Wrote {path}"

    if short == "edit":
        path = tool_input.get("file_path", "")
        old = tool_input.get("old_string", "")[:60]
        new = tool_input.get("new_string", "")[:60]
        if old or new:
            return f"Edited {path}: {old!r} -> {new!r}"
        return f"Edited {path}"

    if short == "grep":
        pattern = tool_input.get("pattern", "")
        return f'Grep "{pattern}": {result_text[:200]}'

    if short == "glob":
        pattern = tool_input.get("pattern", "")
        n_matches = result_text.count("\n") + 1 if result_text.strip() else 0
        return f'Glob "{pattern}": {n_matches} files'

    if short == "agent":
        return f"Agent: {result_text[:200]}"

    # Default: use display name (prefix-stripped, original casing) + truncated result
    return f"{display_name}: {result_text[:200]}"


def _summarize_tool_input(tool_name: str, tool_input: dict) -> str:
    """Extract searchable content from an assistant's tool_use input.

    Captures commands, patterns, and queries — content that helps
    answer "what did I do?" but isn't captured in files_touched.
    """
    name = tool_name.lower()
    short = name.rsplit("__", 1)[-1] if "__" in name else name

    if short == "bash":
        cmd = tool_input.get("command", "")
        desc = tool_input.get("description", "")
        if desc:
            return f"$ {cmd}  # {desc}"
        return f"$ {cmd}" if cmd else ""

    if short == "edit":
        path = tool_input.get("file_path", "")
        old = tool_input.get("old_string", "")[:60]
        new = tool_input.get("new_string", "")[:60]
        if old or new:
            return f"Edit {path}: {old!r} -> {new!r}"
        return ""

    if short == "grep":
        return f'Grep "{tool_input.get("pattern", "")}"'

    if short in ("websearch", "web_search"):
        return f'Search: {tool_input.get("query", "")}'

    return ""

# Phrases that signal a user decision (lowercase matching).
_DECISION_PHRASES = (
    "let's do", "let's go with", "let's use", "let's try",
    "go with option", "go with approach",
    "i'll go with", "i choose", "i picked",
    "approve the plan", "looks good, proceed",
    "yes, do that", "yes, go ahead", "yes, let's",
    "option a", "option b", "option c",
)

# Compiled word-boundary regex for each phrase (prevents substring
# false positives like "option actually" matching "option a").
_DECISION_RE = re.compile(
    "|".join(r"\b" + re.escape(p) + r"\b" for p in _DECISION_PHRASES)
)


def _extract_tool_result_text(entry: dict) -> str:
    """Extract text content from tool_result blocks in a user message.

    AskUserQuestion responses appear as tool_result entries.  The user's
    selected option is in the ``content`` field of the tool_result block.
    """
    content = entry.get("message", {}).get("content")
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        result = block.get("content", "")
        if isinstance(result, str) and result.strip():
            parts.append(result.strip())
        elif isinstance(result, list):
            for sub in result:
                if isinstance(sub, dict) and sub.get("type") == "text":
                    text = sub.get("text", "").strip()
                    if text:
                        parts.append(text)
    return "\n".join(parts)


def _detect_decision_markers(user_text: str, tools_used: list[str]) -> list[str]:
    """Detect decision signals and return virtual tool markers for search boosting.

    Injected into ``tools_used`` so they get indexed with 2x weight in FTS5
    and boosted in BM25 — no schema changes needed.

    Markers use spaces (not underscores) so FTS5's ``tokenchars '._'`` setting
    doesn't merge them into unsearchable single tokens.  E.g. "decision point"
    becomes two tokens ("decision", "point") searchable individually.

    Uses a set internally to avoid duplicates when multiple signals fire
    (e.g., both AskUserQuestion and ExitPlanMode in one turn).
    """
    markers: set[str] = set()
    text_lower = user_text.lower()

    # Signal 1: AskUserQuestion tool was used (structured choice)
    if "AskUserQuestion" in tools_used:
        markers.add("decision point")
        markers.add("user choice")

    # Signal 2: ExitPlanMode (plan approved)
    if "ExitPlanMode" in tools_used:
        markers.add("decision point")
        markers.add("plan approved")

    # Signal 3: Explicit choice keywords in user text (word-boundary match)
    if not markers and _DECISION_RE.search(text_lower):
        markers.add("decision point")

    return sorted(markers)


def _joined_text_blocks(blocks: list) -> str:
    """Join the text of a content-block list, ignoring every other block type.

    Same shape ``message.content`` uses, read the same way.  A non-text block
    can still carry a ``text`` key, so filtering on ``type`` is load-bearing
    and not merely tidy: without it a thinking block's contents would be
    indexed as something the operator said.
    """
    parts = [
        blk.get("text", "").strip()
        for blk in blocks
        if isinstance(blk, dict) and blk.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part)


def _queued_human_prompt(entry: dict) -> str | None:
    """The operator's text if ``entry`` is a message they queued mid-turn.

    A human who types while a turn is running does not produce a ``type:
    "user"`` line.  Claude Code writes a ``queue-operation`` line (which is in
    :data:`SKIP_TYPES`) and an ``attachment`` line carrying the text, so the
    words never reach ``user_text`` and the operator's own question is absent
    from the index that exists to answer it.

    Measured 2026-08-30T13:44Z on one workspace, and every count here is a
    property of that moment -- these files are appended to continuously, and
    that workspace's ordinary-turn total read 1071, 1074, 1075 and 1076 across
    a single afternoon.  Of 510 ``queued_command`` attachments: 352
    ``origin.kind == "human"`` against 1075 ordinary user turns, so about a
    quarter of what the operator said was unindexed, and 349 of the 352 appear
    nowhere else in the file.

    ``origin.kind`` is the whole safety boundary and there are at least three
    cohorts, so it must also define the measurement cohort:

    * ``human``   -- the operator typing.  The only one that is a user turn.
    * ``peer``    -- another agent's queued message.  Excluded, deliberately.
    * no ``origin`` key at all -- ``commandMode == "task-notification"``,
      machine background-task events.  158 of the 510 here, and NONE of them
      empty; an earlier draft of this docstring called them empty text and was
      simply wrong about what they are.

    Widening past ``human`` would attribute an agent's words, or a runtime
    notification, to a person who did not write them.

    ``prompt`` arrives as a string or as a list of content blocks; both are
    read.  Only the list form was absent from the workspace this was first
    measured on, which is exactly why it was missed: a shape your own corpus
    cannot produce is invisible to any amount of care with your own corpus.

    Returns ``None`` for every other entry, so callers can use it as the
    queued-form branch without widening what counts as a user turn.
    """
    if entry.get("type") != "attachment":
        return None
    attachment = entry.get("attachment")
    if not isinstance(attachment, dict):
        return None
    if attachment.get("type") != "queued_command":
        return None
    origin = attachment.get("origin")
    if not isinstance(origin, dict) or origin.get("kind") != "human":
        return None
    prompt = attachment.get("prompt")
    if isinstance(prompt, list):
        prompt = _joined_text_blocks(prompt)
    if not isinstance(prompt, str):
        return None
    return prompt.strip() or None


def _is_real_user_message(entry: dict) -> bool:
    """True if entry is a user turn that starts a new conversation turn.

    Real user messages have string content or contain at least one text block
    *after* system artifacts are stripped. Messages containing only
    ``<system-reminder>`` blocks (no actual user text) are treated as
    non-messages so they don't create phantom empty turns in the index.
    """
    from synapt.recall.scrub import strip_system_artifacts

    queued = _queued_human_prompt(entry)
    if queued is not None:
        return bool(strip_system_artifacts(queued))
    if entry.get("type") != "user":
        return False
    msg = entry.get("message", {})
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return bool(strip_system_artifacts(content.strip()))
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "text"
            and strip_system_artifacts(b.get("text", "").strip())
            for b in content
        )
    return False


def _extract_user_text(entry: dict) -> str:
    """Extract plain text from a user message entry.

    Strips Claude Code system artifacts (``<system-reminder>``,
    ``<local-command-caveat>``, etc.) so they never reach the index,
    journal stubs, or timeline arcs.
    """
    from synapt.recall.scrub import strip_system_artifacts

    queued = _queued_human_prompt(entry)
    if queued is not None:
        return strip_system_artifacts(queued)
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return strip_system_artifacts(content.strip())
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    parts.append(text)
        return strip_system_artifacts("\n".join(parts))
    return ""


def _extract_assistant_content(
    entry: dict,
) -> tuple[str, list[str], list[str], list[tuple[str, str, dict]]]:
    """Extract text, tool names, file paths, and tool invocations.

    Returns (text, tools_used, files_touched, tool_uses) where tool_uses
    is a list of (tool_use_id, tool_name, tool_input) tuples for matching
    with subsequent tool_result entries.
    """
    content = entry.get("message", {}).get("content")
    if not isinstance(content, list):
        return "", [], [], []

    texts = []
    tools = []
    files = []
    tool_uses: list[tuple[str, str, dict]] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "").strip()
            if text:
                texts.append(text)
        elif btype == "tool_use":
            name = block.get("name", "")
            if name:
                tools.append(name)
            inp = block.get("input", {})
            tool_use_id = block.get("id", "")
            if tool_use_id and isinstance(inp, dict):
                tool_uses.append((tool_use_id, name, inp))
            if isinstance(inp, dict):
                for key in ("file_path", "path", "command", "pattern"):
                    val = inp.get(key)
                    if val and isinstance(val, str):
                        if key in ("file_path", "path") or (
                            key == "pattern" and "/" in val
                        ):
                            files.append(val)
                        break
        # Skip 'thinking' blocks — internal reasoning, not useful for RAG

    return "\n".join(texts), tools, files, tool_uses


def _short_sid(session_id: str) -> str:
    """Shorten a session ID for display.

    UUID-style IDs (32+ hex chars) have unique 8-char prefixes.
    Shorter IDs (e.g. "session_001") are kept in full to avoid
    collisions where all sessions truncate to the same prefix.
    """
    return session_id[:8] if len(session_id) >= 16 else session_id


def _extract_snippet(text: str, query: str, context_lines: int = 1) -> str:
    """Extract the most query-relevant sentence from *text* with surrounding context.

    Inspired by grep's ``-C`` flag: scores each sentence by token overlap with
    the query, returns the best match plus *context_lines* sentences on each side.

    Falls back to the first 200 characters if no sentence scores above zero.
    """
    if not text or not query:
        return text[:200] if text else ""

    # Split into sentences. Use period/exclamation/question + space, or newlines.
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[:200]

    if len(sentences) == 1:
        return sentences[0]

    # Score each sentence by token overlap with the query
    query_tokens = set(_tokenize(query))
    best_idx = 0
    best_score = -1

    for i, sentence in enumerate(sentences):
        sent_tokens = set(_tokenize(sentence))
        overlap = len(query_tokens & sent_tokens)
        if overlap > best_score:
            best_score = overlap
            best_idx = i

    # No overlap at all — return first N chars as fallback
    if best_score <= 0:
        return text[:200] + ("..." if len(text) > 200 else "")

    # Extract the window: best sentence +/- context_lines
    start = max(0, best_idx - context_lines)
    end = min(len(sentences), best_idx + context_lines + 1)
    snippet = " ".join(sentences[start:end])

    # Add ellipsis markers when we're not at the boundaries
    if start > 0:
        snippet = "..." + snippet
    if end < len(sentences):
        snippet = snippet + "..."

    return snippet


def _parse_chunk_timestamp(timestamp: str) -> datetime | None:
    """Parse ISO-ish chunk timestamps into aware UTC datetimes."""
    if not timestamp:
        return None
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _knowledge_overlaps_temporal_window(
    node: dict,
    after: str | None = None,
    before: str | None = None,
) -> tuple[bool, bool]:
    """Return whether a knowledge node overlaps a query window.

    Returns ``(overlaps, has_temporal_bounds)``. Nodes without parseable
    ``valid_from``/``valid_until`` remain eligible as timeless fallback
    context, but do not receive a temporal overlap boost.
    """
    valid_from = _parse_chunk_timestamp(node.get("valid_from", ""))
    valid_until = _parse_chunk_timestamp(node.get("valid_until", ""))
    if valid_from is None and valid_until is None:
        return True, False

    query_after = _parse_chunk_timestamp(after or "")
    query_before = _parse_chunk_timestamp(before or "")

    if query_before is not None and valid_from is not None and valid_from >= query_before:
        return False, True
    if query_after is not None and valid_until is not None and valid_until < query_after:
        return False, True
    return True, True


def _format_timestamp_display(timestamp: str, intent: str = "") -> str:
    """Render chunk timestamps in a query-appropriate display format."""
    if not timestamp:
        return "unknown"

    dt = _parse_chunk_timestamp(timestamp)
    if dt is None:
        return timestamp

    if intent == "temporal":
        hour = dt.strftime("%I").lstrip("0") or "12"
        return (
            f"{dt.strftime('%A')}, {dt.strftime('%B')} {dt.day}, "
            f"{dt.year}, {hour}:{dt.strftime('%M')} {dt.strftime('%p')}"
        )

    return dt.strftime("%Y-%m-%d %H:%M")


def parse_transcript(
    path: Path,
    seen_uuids: set[str] | None = None,
    subchunk_min_text: int | None = None,
) -> list[TranscriptChunk]:
    """Parse a Claude Code transcript JSONL file into semantic chunks.

    Each chunk represents one user->assistant turn, or a sub-chunk of a turn
    when the turn spans multiple tool-use cycles with distinct topics.

    Large turns (>1200 chars) with multiple tool cycles are split at tool
    boundaries — where the assistant processes tool output and begins a new
    action.  This prevents topical dilution in embeddings (e.g., coverage
    numbers buried in a lint-fix chunk).

    Args:
        path: Path to a .jsonl transcript file.
        seen_uuids: Set of already-seen (session_id, uuid) pairs for dedup.
                    Mutated in-place to add new entries.

    Returns:
        List of TranscriptChunk objects.
    """
    if seen_uuids is None:
        seen_uuids = set()

    chunks: list[TranscriptChunk] = []
    session_id = path.stem  # UUID from filename
    transcript_path = str(path)

    # Accumulator for current turn — segments track tool-boundary splits
    current_user_text = ""
    current_assistant_texts: list[str] = []
    current_tools: list[str] = []
    current_files: list[str] = []
    current_timestamp = ""
    current_tool_summaries: list[str] = []
    # Maps tool_use_id → (tool_name, tool_input) for matching results
    current_tool_use_map: dict[str, tuple[str, dict]] = {}
    turn_index = 0
    turn_start_offset = 0
    current_offset = 0

    # --- Sub-chunk segment tracking ---
    # Each segment represents one coherent work unit within a turn.
    # A new segment starts when assistant text follows a tool_result,
    # indicating the assistant processed output and shifted topics.
    _Seg = lambda: {"texts": [], "tools": [], "files": [], "summaries": []}
    current_segments: list[dict] = [_Seg()]
    saw_tool_result = False

    def _flush_turn(end_offset: int | None = None):
        nonlocal turn_index, current_segments, saw_tool_result
        if not current_user_text and not current_assistant_texts:
            return
        _end = end_offset if end_offset is not None else current_offset

        short_id = _short_sid(session_id)

        # Decide whether to sub-chunk: need ≥2 non-empty segments
        # and enough total text to justify splitting.
        nonempty_segs = [s for s in current_segments
                         if s["texts"] or s["summaries"]]

        total_text = sum(
            sum(len(t) for t in s["texts"]) +
            sum(len(t) for t in s["summaries"])
            for s in nonempty_segs
        )

        if subchunk_min_text is not None:
            _sub_chunk_min_text = subchunk_min_text
        else:
            _sub_chunk_min_text = int(os.environ.get("SYNAPT_SUBCHUNK_MIN_TEXT", "1200"))
        if len(nonempty_segs) >= 2 and total_text > _sub_chunk_min_text and _sub_chunk_min_text > 0:
            # --- Sub-chunk mode ---
            for seg_i, seg in enumerate(nonempty_segs):
                a_text = "\n".join(seg["texts"]).strip()
                if len(a_text) > 5000:
                    a_text = a_text[:5000] + "..."

                seg_tools = list(dict.fromkeys(seg["tools"]))
                seg_files = list(dict.fromkeys(seg["files"]))

                tc = "\n".join(s for s in seg["summaries"] if s)
                if tc:
                    try:
                        tc = scrub_text(tc)
                    except Exception:
                        logger.debug("scrub_text failed on tool content, using raw", exc_info=True)
                if len(tc) > 3000:
                    tc = tc[:3000] + "..."

                # First sub-chunk gets the full user question;
                # subsequent ones get a brief context prefix.
                if seg_i == 0:
                    u_text = (current_user_text[:1500] + "..."
                              if len(current_user_text) > 1500
                              else current_user_text)
                else:
                    abbrev = current_user_text[:100]
                    if len(current_user_text) > 100:
                        abbrev += "..."
                    u_text = f"(context: User previously asked: {abbrev})"

                # Decision markers only on first sub-chunk
                if seg_i == 0:
                    dm = _detect_decision_markers(current_user_text, seg_tools)
                    if dm:
                        seg_tools.extend(dm)

                chunk = TranscriptChunk(
                    id=f"{short_id}:t{turn_index}",
                    session_id=session_id,
                    timestamp=current_timestamp,
                    turn_index=turn_index,
                    user_text=u_text,
                    assistant_text=a_text,
                    tools_used=seg_tools,
                    files_touched=seg_files,
                    tool_content=tc,
                    transcript_path=transcript_path,
                    byte_offset=turn_start_offset,
                    byte_length=_end - turn_start_offset,
                )
                chunks.append(chunk)
                turn_index += 1
        else:
            # --- Single-chunk mode (original behavior) ---
            assistant_text = "\n".join(current_assistant_texts).strip()
            if len(assistant_text) > 5000:
                assistant_text = assistant_text[:5000] + "..."
            tools_deduped = list(dict.fromkeys(current_tools))
            decision_markers = _detect_decision_markers(current_user_text, tools_deduped)
            if decision_markers:
                tools_deduped.extend(decision_markers)
            tool_content = "\n".join(s for s in current_tool_summaries if s)
            if tool_content:
                try:
                    tool_content = scrub_text(tool_content)
                except Exception:
                    logger.debug("scrub_text failed on tool content, using raw", exc_info=True)
            if len(tool_content) > 3000:
                tool_content = tool_content[:3000] + "..."
            chunk = TranscriptChunk(
                id=f"{short_id}:t{turn_index}",
                session_id=session_id,
                timestamp=current_timestamp,
                turn_index=turn_index,
                user_text=(current_user_text[:1500] + "..."
                          if len(current_user_text) > 1500
                          else current_user_text),
                assistant_text=assistant_text,
                tools_used=tools_deduped,
                files_touched=list(dict.fromkeys(current_files)),
                tool_content=tool_content,
                transcript_path=transcript_path,
                byte_offset=turn_start_offset,
                byte_length=_end - turn_start_offset,
            )
            chunks.append(chunk)
            turn_index += 1

        # Reset segment tracking
        current_segments = [_Seg()]
        saw_tool_result = False

    with open(path, encoding="utf-8", newline="") as f:
        for raw_line in f:
            line_start = current_offset
            current_offset += len(raw_line.encode("utf-8"))
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")

            # Skip noise types
            if entry_type in SKIP_TYPES:
                continue

            # Dedup: skip already-seen entries (cumulative snapshot overlap)
            uuid = entry.get("uuid", "")
            if uuid:
                dedup_key = (session_id, uuid)
                if dedup_key in seen_uuids:
                    continue
                seen_uuids.add(dedup_key)

            if _is_real_user_message(entry):
                # Flush previous turn — end boundary is this line's start
                # (so the previous turn doesn't include this new user message)
                _flush_turn(end_offset=line_start)
                turn_start_offset = line_start
                raw_user = _extract_user_text(entry)
                try:
                    current_user_text = scrub_text(raw_user)
                except Exception:
                    logger.debug("scrub_text failed on user text, using raw", exc_info=True)
                    current_user_text = raw_user
                current_assistant_texts = []
                current_tools = []
                current_files = []
                current_tool_summaries = []
                current_tool_use_map = {}
                current_timestamp = entry.get("timestamp", "")
                current_segments = [_Seg()]
                saw_tool_result = False

            elif entry_type == "user":
                # Tool-result-only entry — capture AskUserQuestion responses
                # AND summarize all tool results for search indexing.
                if "AskUserQuestion" in current_tools:
                    result_text = _extract_tool_result_text(entry)
                    if result_text:
                        current_assistant_texts.append(f"User selected: {result_text}")
                        current_segments[-1]["texts"].append(f"User selected: {result_text}")
                # Summarize tool results
                content = entry.get("message", {}).get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_result":
                            continue
                        tool_use_id = block.get("tool_use_id", "")
                        result = block.get("content", "")
                        if isinstance(result, list):
                            # Extract text from nested blocks
                            result = "\n".join(
                                sub.get("text", "")
                                for sub in result
                                if isinstance(sub, dict) and sub.get("type") == "text"
                            )
                        if not isinstance(result, str):
                            continue
                        # Skip tool error results — infrastructure noise, not session context
                        if "<tool_use_error>" in result:
                            continue
                        tool_name, tool_input = current_tool_use_map.get(
                            tool_use_id, ("", {})
                        )
                        if tool_name:
                            summary = _summarize_tool_result(
                                tool_name, tool_input, result,
                            )
                            if summary:
                                current_tool_summaries.append(summary)
                                current_segments[-1]["summaries"].append(summary)
                saw_tool_result = True

            elif entry_type == "assistant":
                text, tools, files, tool_uses = _extract_assistant_content(entry)
                if text:
                    try:
                        text = scrub_text(text)
                    except Exception:
                        logger.debug("scrub_text failed on assistant text, using raw", exc_info=True)
                    # Start a new segment when assistant text follows a tool
                    # result — the assistant has processed output and is likely
                    # shifting to a new action or topic.
                    if saw_tool_result and text.strip():
                        current_segments.append(_Seg())
                        saw_tool_result = False
                    current_assistant_texts.append(text)
                    current_segments[-1]["texts"].append(text)
                current_tools.extend(tools)
                current_files.extend(files)
                current_segments[-1]["tools"].extend(tools)
                current_segments[-1]["files"].extend(files)
                # Track tool invocations for matching with results
                for tu_id, tu_name, tu_input in tool_uses:
                    current_tool_use_map[tu_id] = (tu_name, tu_input)
                    input_summary = _summarize_tool_input(tu_name, tu_input)
                    if input_summary:
                        current_tool_summaries.append(input_summary)
                        current_segments[-1]["summaries"].append(input_summary)

    # Flush last turn
    _flush_turn()

    return chunks


# ---------------------------------------------------------------------------
# Session recency
# ---------------------------------------------------------------------------

#: Chunks with this turn index are journal entries rather than conversation.
JOURNAL_TURN_INDEX = -1


def normalize_timestamp(value: str) -> tuple[int, str]:
    """Return a sort key that compares timestamps by instant, not spelling.

    Timestamps reach the index in more than one ISO-8601 spelling — most end in
    ``Z``, a few carry an explicit ``+00:00`` offset, and some carry fractional
    seconds. Compared as raw text those spellings interleave wrongly: ``'Z'``
    (0x5A) outranks the ``'.'`` (0x2E) that opens a fraction, so
    ``03:00:00.500000+00:00`` sorts BELOW ``03:00:00Z`` despite being half a
    second later.

    Parsed values sort ahead of unparseable ones so that a malformed timestamp
    degrades to "oldest" rather than winning by accident. On the unparseable
    branch the raw string is the second element, so malformed values still order
    deterministically among themselves; two *parsed* values at the same instant
    tie outright and fall to the caller's sort stability, which is correct —
    they name the same moment, and inventing a winner between them would be
    ordering by spelling again.
    """
    if not value:
        return (0, "")
    try:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (1, f"{parsed.timestamp():020.6f}")
    except (ValueError, TypeError):
        return (0, value)


def _session_activity_key(chunks: list) -> tuple[int, str]:
    """Newest ACTIVITY in a session, ignoring when journals were written."""
    activity = [
        c.timestamp for c in chunks
        if getattr(c, "turn_index", 0) != JOURNAL_TURN_INDEX and c.timestamp
    ]
    if not activity:
        # Journal-only session: fall back so it still orders somewhere.
        activity = [c.timestamp for c in chunks if c.timestamp]
    if not activity:
        return (0, "")
    return max(normalize_timestamp(t) for t in activity)


# ---------------------------------------------------------------------------
# TranscriptIndex
# ---------------------------------------------------------------------------

class TranscriptIndex:
    """Searchable index over transcript chunks.

    Supports BM25 keyword search and optional embedding-based semantic search.
    Chunks are organized by session and sorted newest-first for progressive
    time-ordered retrieval.
    """

    def __init__(
        self,
        chunks: list[TranscriptChunk],
        use_embeddings: bool = False,
        cache_dir: Path | None = None,
        db: RecallDB | ShardedRecallDB | None = None,
        lazy_chunks: bool = False,
    ):
        # Sort by timestamp descending (most recent first)
        self.chunks = sorted(chunks, key=lambda c: c.timestamp, reverse=True)
        self._oldest_indexed_at = _oldest_indexed_date(self.chunks)
        self._lazy_chunks = lazy_chunks

        # Group by session for progressive search
        self.sessions: dict[str, list[TranscriptChunk]] = {}
        for chunk in self.chunks:
            self.sessions.setdefault(chunk.session_id, []).append(chunk)

        # Session order: most recent first, by latest ACTIVITY in the session.
        #
        # Journal chunks (turn_index -1) are excluded from the comparison: their
        # timestamp is when the journal was WRITTEN, and indexing stamps
        # auto-synthesised ones at build time. Including them lets a session
        # that has been dead for weeks acquire a timestamp of "now" and sort
        # above a live one — which is how a 2026-07-22 session won a
        # "newest session" contract on 2026-08-06 (recall#935).
        #
        # A session with nothing BUT journal chunks falls back to those, so it
        # still orders somewhere rather than disappearing.
        self._session_order = sorted(
            self.sessions.keys(),
            key=lambda sid: _session_activity_key(self.sessions[sid]),
            reverse=True,
        )

        # Turn index lookup: (session_id, turn_index) -> chunk for O(1) preceding turn
        # Exclude journal chunks (turn_index=-1) to avoid collisions
        self._turn_lookup: dict[tuple[str, int], TranscriptChunk] = {
            (c.session_id, c.turn_index): c for c in self.chunks
            if c.turn_index >= 0
        }

        # Search diagnostics (populated on empty results for caller inspection)
        self._last_diagnostics: SearchDiagnostics | None = None
        self._last_search_summary: SearchResultSummary | None = None
        self._last_conflicts: list[tuple[dict, dict]] = []
        self._last_knowledge_semantic_used = False

        # Query result cache — avoids re-computing BM25/embedding/RRF for
        # identical queries within the same index lifetime. LRU with max 32 entries.
        self._query_cache: dict[
            tuple,
            tuple[str, SearchResultSummary | None, SearchDiagnostics | None],
        ] = {}
        self._query_cache_max = 32

        # Working memory — seeded from recent access_log for cross-session persistence
        from synapt.recall.working_memory import WorkingMemory
        self._working_memory = WorkingMemory()
        if db is not None:
            self._working_memory.seed_from_db(db)

        # Current session ID (set by MCP server for access tracking)
        self._current_session_id: str = ""

        # Chunk ID -> index mapping (for rowid conversion)
        self._id_to_idx: dict[str, int] = {
            c.id: i for i, c in enumerate(self.chunks)
        }

        # SQLite backend (None = legacy BM25-only mode)
        self._db: RecallDB | None = db
        self._rowid_to_idx: dict[int, int] = {}
        self._idx_to_rowid: dict[int, int] = {}

        if db is not None and db.chunk_count() > 0:
            self._refresh_rowid_map()

        # Content profile — adapts filters and retrieval to content type
        from synapt.recall.content_profile import (
            adaptive_params,
            detect_content_profile,
            forced_content_profile,
        )
        if os.environ.get("SYNAPT_DISABLE_CONTENT_PROFILE"):
            from synapt.recall.content_profile import ContentProfile
            self.content_profile = ContentProfile(
                _type="code", file_refs=0, personal_refs=0, total_chunks=len(self.chunks)
            )
        else:
            forced = forced_content_profile(total_chunks=len(self.chunks))
            if forced is not None:
                self.content_profile = forced
            elif self._lazy_chunks and self._db is not None and self._idx_to_rowid:
                sample_size = min(len(self.chunks), 200)
                sample_rowids = [self._idx_to_rowid[i] for i in range(sample_size)]
                sample_chunks = [
                    chunk for _, chunk in sorted(
                        self._db.load_chunks_by_rowids(sample_rowids).items()
                    )
                ]
                self.content_profile = detect_content_profile(sample_chunks)
                self.content_profile.total_chunks = len(self.chunks)
            else:
                self.content_profile = detect_content_profile(self.chunks)
        self._adaptive = adaptive_params(self.content_profile)
        if self.content_profile.content_type != "code":
            logger.info(
                "Content profile: %s (file_refs=%d, personal=%d, chunks=%d)",
                self.content_profile.content_type,
                self.content_profile.file_refs,
                self.content_profile.personal_refs,
                self.content_profile.total_chunks,
            )

        if self._rowid_to_idx:
            # FTS5 is ready — skip in-memory BM25
            self._bm25 = None
        else:
            # Build in-memory BM25 index (no DB or DB not yet populated)
            self._bm25 = BM25()
            corpus = []
            for chunk in self.chunks:
                tokens = _tokenize(chunk.text)
                # Boost tools and files by repeating them
                boost_text = " ".join(chunk.tools_used + chunk.files_touched)
                tokens.extend(_tokenize(boost_text))
                corpus.append(tokens)
            self._bm25.index(corpus)

        # Optional embedding index
        self._embeddings: list[list[float]] | None = None
        self._embed_provider = None
        # Pre-loaded embeddings for hybrid search — numpy-backed for memory
        # efficiency.  Loaded lazily on first search to avoid blocking init.
        self._all_embeddings: dict[int, list[float]] = {}
        self._knowledge_embeddings: dict[int, list[float]] = {}
        self._emb_matrix: "np.ndarray | None" = None
        self._emb_rowids: list[int] = []
        self._kn_emb_matrix: "np.ndarray | None" = None
        self._kn_emb_rowids: list[int] = []
        self._embeddings_loaded: bool = False
        self._cache_dir = cache_dir
        # Track embedding status for user-facing messages
        self._embedding_status: str = "disabled"  # disabled | active | unavailable
        self._embedding_reason: str = ""
        if use_embeddings:
            try:
                has_chunk_embeddings = self._db.has_embeddings() if self._db is not None else True
                # Only build embeddings if storage is available.
                if self._db is None or self._idx_to_rowid:
                    self._load_or_build_embeddings(cache_dir)

                # If the index has no chunk embeddings yet, keep CLI/server
                # search on the fast BM25 path instead of paying model load for
                # knowledge-only semantic lookup. A later build can populate
                # embeddings in the DB, but this TranscriptIndex instance stays
                # BM25-only until the next fresh load.
                if not has_chunk_embeddings:
                    self._embedding_reason = (
                        "Chunk embeddings are not available for this index yet; "
                        "using BM25-only search."
                    )
                else:
                    from synapt.recall.embeddings import get_embedding_provider
                    provider = get_embedding_provider()
                    if provider:
                        self._embed_provider = provider
                        self._embedding_status = "active"
                        # Embedding data is loaded lazily via
                        # _ensure_embeddings_loaded() on first search call.
                    else:
                        self._embedding_status = "unavailable"
                        self._embedding_reason = (
                            "No embedding provider found. "
                            "Install sentence-transformers for semantic search: "
                            "pip install sentence-transformers"
                        )
            except Exception as e:
                logger.warning("Embeddings unavailable: %s", e)
                self._embedding_status = "unavailable"
                self._embedding_reason = str(e)

        # Cross-encoder reranking (Phase 2)
        from synapt.recall.reranker import is_reranker_enabled
        self._use_reranker = is_reranker_enabled()
        if self._use_reranker:
            logger.info("Cross-encoder reranking enabled")

    def _open_background_db(self):
        """Open a fresh DB handle for background embedding work."""
        if self._cache_dir is None:
            return self._db
        return ShardedRecallDB.open(self._cache_dir)

    def _build_embeddings_background(self, content_hash: str) -> None:
        """Build chunk + knowledge embeddings in a background thread."""
        db = None
        try:
            from synapt.recall.embeddings import get_embedding_provider

            db = self._open_background_db()
            if db is None:
                return
            provider = get_embedding_provider()
            if provider is None:
                return

            texts = [c.text[:500] for c in self._materialize_all_chunks()]
            all_embs: list[list[float]] = []
            for i in range(0, len(texts), 64):
                batch = texts[i:i + 64]
                all_embs.extend(provider.embed(batch))

            emb_mapping: dict[int, list[float]] = {}
            for i, emb in enumerate(all_embs):
                rowid = self._idx_to_rowid.get(i)
                if rowid is not None:
                    emb_mapping[rowid] = emb
            if emb_mapping:
                db.save_embeddings(emb_mapping)
                db.set_metadata("embedding_hash", content_hash)
                logger.info(
                    "Background embedding build complete: %d chunks",
                    len(emb_mapping),
                )

            self._build_knowledge_embeddings(db=db, provider=provider)
        except Exception as e:
            logger.warning("Background embedding build failed: %s", e)
        finally:
            if db is not None and db is not self._db:
                with contextlib.suppress(Exception):
                    db.close()

    def _build_knowledge_embeddings(self, db=None, provider=None) -> None:
        """Build embeddings for knowledge nodes that don't have them yet."""
        db = db or self._db
        provider = provider or self._embed_provider
        if not db or not provider:
            return
        try:
            missing = db.get_knowledge_rowids_without_embeddings()
            if not missing:
                return
            texts = [content[:500] for _, content in missing]
            rowids = [rowid for rowid, _ in missing]
            all_embs: list[list[float]] = []
            for i in range(0, len(texts), 64):
                batch = texts[i:i + 64]
                all_embs.extend(provider.embed(batch))
            emb_mapping = dict(zip(rowids, all_embs))
            if emb_mapping:
                db.save_knowledge_embeddings(emb_mapping)
                logger.info(
                    "Built embeddings for %d knowledge nodes", len(emb_mapping),
                )
        except Exception as e:
            logger.warning("Knowledge embedding build failed: %s", e)

    def _get_chunk(self, idx: int) -> TranscriptChunk:
        """Return a chunk, hydrating it from the DB on demand when needed."""
        chunk = self.chunks[idx]
        if not self._lazy_chunks or self._db is None:
            return chunk
        if chunk.user_text or chunk.assistant_text or chunk.tool_content or chunk.transcript_path:
            return chunk
        rowid = self._idx_to_rowid.get(idx)
        if rowid is None:
            return chunk
        loaded = self._db.load_chunk_by_rowid(rowid)
        if loaded is None:
            return chunk
        self.chunks[idx] = loaded
        session_chunks = self.sessions.get(loaded.session_id, [])
        for pos, existing in enumerate(session_chunks):
            if existing.id == loaded.id:
                session_chunks[pos] = loaded
                break
        if loaded.turn_index >= 0:
            self._turn_lookup[(loaded.session_id, loaded.turn_index)] = loaded
        return loaded

    def _materialize_all_chunks(self) -> list[TranscriptChunk]:
        """Force hydration of all chunks when a full-scan path needs raw text."""
        if not self._lazy_chunks:
            return self.chunks
        for i in range(len(self.chunks)):
            self._get_chunk(i)
        return self.chunks

    def _ensure_embeddings_loaded(self) -> None:
        """Load embedding data on first access (lazy initialization).

        Populates ``_emb_matrix`` / ``_emb_rowids`` from the DB using
        numpy-backed bulk loading, avoiding millions of Python float
        objects.  Also populates the legacy ``_all_embeddings`` dict
        for any code paths that still reference it, and loads knowledge
        embeddings.

        Skips loading if embeddings were already injected (e.g. by tests).
        """
        if self._embeddings_loaded:
            return
        self._embeddings_loaded = True
        # Respect manually injected embeddings (used in integration tests)
        if self._all_embeddings:
            return
        if self._db is None:
            return
        try:
            import numpy as np
            matrix, rowids = self._db.get_all_embeddings_numpy()
            self._emb_matrix = matrix
            self._emb_rowids = rowids
            # Build legacy dict from numpy for backward-compat paths
            self._all_embeddings = {
                rowids[i]: matrix[i].tolist()
                for i in range(len(rowids))
            } if len(rowids) > 0 else {}
        except Exception:
            # Fallback to the dict loader if numpy is unavailable
            self._all_embeddings = self._db.get_all_embeddings()
            self._emb_matrix = None
            self._emb_rowids = []
        self._knowledge_embeddings = self._db.get_knowledge_embeddings()

    def _rerank_candidates(
        self,
        query: str,
        candidates: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Apply cross-encoder reranking if enabled."""
        if os.environ.get("SYNAPT_DISABLE_RERANKER"):
            return candidates
        if not self._use_reranker:
            return candidates
        try:
            from synapt.recall.reranker import rerank
            return rerank(query, candidates, self.chunks)
        except Exception as e:
            logger.warning("Reranking failed, using original order: %s", e)
            return candidates

    # ------------------------------------------------------------------
    # Cross-session threading
    # ------------------------------------------------------------------

    # Build-time constants
    CROSS_LINK_MAX_PER_CHUNK = 3    # top-K neighbors per chunk
    CROSS_LINK_MIN_SIM = 0.35      # minimum cosine similarity for a link
    # Query-time constants
    CROSS_LINK_DISCOUNT = 0.7      # score multiplier for expanded results
    CROSS_LINK_MAX_EXPAND = int(os.environ.get("SYNAPT_CROSS_LINK_MAX_EXPAND", "3"))
    # CROSS_LINK_DIVERSITY and SEGMENT_* removed — dual-view RRF was
    # hurting non-cross-session queries.  Link expansion alone is simpler
    # and doesn't disrupt the natural search ordering.

    def build_cross_session_links(self) -> int:
        """Compute and store cross-session nearest-neighbor links.

        For each chunk, find its top-K most similar chunks from OTHER sessions
        using cosine similarity on pre-computed embeddings. Links are stored in
        the SQLite chunk_links table.

        Returns the number of links created.
        """
        if _env_flag("SYNAPT_DISABLE_CROSS_LINKS"):
            return 0
        self._ensure_embeddings_loaded()
        if not self._db or not self._all_embeddings:
            return 0
        import numpy as np

        rowids = self._emb_rowids if self._emb_matrix is not None else list(self._all_embeddings.keys())
        if len(rowids) < 2:
            return 0

        # Build session map and chunk ID map: rowid -> session_id, rowid -> chunk_id
        session_map = self._db.chunk_session_map()
        id_map = self._db.chunk_id_map()
        sessions = [session_map.get(r, "") for r in rowids]

        # Use pre-built numpy matrix if available, else build from dict
        if self._emb_matrix is not None:
            matrix = self._emb_matrix.copy()
        else:
            matrix = np.array(
                [self._all_embeddings[r] for r in rowids], dtype=np.float32,
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = matrix / norms

        # Full pairwise cosine similarity
        sim_matrix = normed @ normed.T

        # Zero out self-similarity and same-session pairs
        for i in range(len(rowids)):
            sim_matrix[i, i] = -1.0
            for j in range(i + 1, len(rowids)):
                if sessions[i] == sessions[j]:
                    sim_matrix[i, j] = -1.0
                    sim_matrix[j, i] = -1.0

        links: list[tuple[str, str, float]] = []
        k = self.CROSS_LINK_MAX_PER_CHUNK
        min_sim = self.CROSS_LINK_MIN_SIM

        for i in range(len(rowids)):
            row = sim_matrix[i]
            # Get top-K indices
            if len(rowids) > k:
                top_idx = np.argpartition(row, -k)[-k:]
            else:
                top_idx = np.arange(len(rowids))
            for j in top_idx:
                if row[j] >= min_sim:
                    src_id = id_map.get(rowids[i], str(rowids[i]))
                    tgt_id = id_map.get(rowids[int(j)], str(rowids[int(j)]))
                    links.append((src_id, tgt_id, float(row[j])))

        self._db.save_chunk_links(links)
        logger.info("Built %d cross-session links for %d chunks", len(links), len(rowids))
        return len(links)

    def _expand_cross_session(
        self,
        candidates: list[tuple[int, float]],
        max_chunks: int,
    ) -> list[tuple[int, float]]:
        """Expand results via pre-computed cross-session similarity links.

        Follows the top-K nearest-neighbor links from the best results to
        inject related chunks — including from already-represented sessions.
        This addresses the "right session, wrong chunk" problem where
        evidence is buried in a chunk whose primary topic differs from
        the query (e.g., coverage numbers in a lint-fix chunk).

        Preserves the original search ordering — only appends linked chunks
        after their source position.
        """
        if _env_flag("SYNAPT_DISABLE_CROSS_LINKS"):
            return candidates
        if not self._db:
            return candidates
        if len(candidates) < 3:
            return candidates

        # --- Cross-link expansion only ---
        # Don't re-rank candidates (dual-view RRF hurt non-cross-session
        # queries by disrupting the natural search order).  Instead, follow
        # pre-computed similarity links from top results to inject related
        # chunks — even from already-represented sessions, because the
        # problem is often having the RIGHT session but WRONG chunk.
        if not self._db.has_chunk_links():
            return candidates

        result = list(candidates)
        seen = {idx for idx, _ in result}

        # Map top result indices to chunk IDs for link lookup
        source_ids = []
        idx_to_chunk_id: dict[int, str] = {}
        chunk_id_to_idx: dict[str, int] = {}
        for idx, _ in result[:5]:
            if idx < len(self.chunks):
                cid = self._get_chunk(idx).id
                source_ids.append(cid)
                idx_to_chunk_id[idx] = cid
                chunk_id_to_idx[cid] = idx

        if not source_ids:
            return candidates

        # Build reverse map for all chunks (for target lookup)
        if not chunk_id_to_idx:
            return candidates
        for i, chunk in enumerate(self.chunks):
            if chunk.id not in chunk_id_to_idx:
                chunk_id_to_idx[chunk.id] = i

        neighbors = self._db.get_cross_links_batch(source_ids)
        expansions = 0
        for src_id in source_ids:
            src_idx = chunk_id_to_idx.get(src_id)
            if src_idx is None:
                continue
            src_score = next(
                (s for i, s in result if i == src_idx), 0.0,
            )
            for tgt_id, sim in neighbors.get(src_id, []):
                tgt_idx = chunk_id_to_idx.get(tgt_id)
                if tgt_idx is None or tgt_idx in seen:
                    continue
                exp_score = src_score * sim * self.CROSS_LINK_DISCOUNT
                # Insert after source position
                src_pos = next(
                    (i for i, (ci, _) in enumerate(result) if ci == src_idx),
                    len(result),
                )
                result.insert(src_pos + 1, (tgt_idx, exp_score))
                seen.add(tgt_idx)
                expansions += 1
                if expansions >= self.CROSS_LINK_MAX_EXPAND:
                    break
            if expansions >= self.CROSS_LINK_MAX_EXPAND:
                break

        return result

    def _refresh_rowid_map(self) -> None:
        """Build rowid <-> chunk-index mappings from the database."""
        if not self._db:
            return
        id_to_rowid = self._db.get_chunk_id_rowid_map()
        self._rowid_to_idx = {}
        self._idx_to_rowid = {}
        for i, chunk in enumerate(self.chunks):
            rowid = id_to_rowid.get(chunk.id)
            if rowid is not None:
                self._rowid_to_idx[rowid] = i
                self._idx_to_rowid[i] = rowid

    def _load_or_build_embeddings(self, cache_dir: Path | None):
        """Build or load cached embeddings for all chunks."""
        if self._db is not None:
            return self._load_or_build_embeddings_db()

        # Legacy file-based cache
        cache_path = cache_dir / "transcript_embeddings.json" if cache_dir else None
        content_hash = self._content_hash()

        # Try cache first
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("hash") == content_hash:
                    self._embeddings = cached["embeddings"]
                    return
            except (json.JSONDecodeError, KeyError):
                pass

        # Build embeddings
        texts = [c.text[:500] for c in self._materialize_all_chunks()]  # cap text length
        try:
            all_embs = []
            for i in range(0, len(texts), 64):
                batch = texts[i:i + 64]
                all_embs.extend(self._embed_provider.embed(batch))
            self._embeddings = all_embs

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"hash": content_hash, "embeddings": all_embs}, f)
        except Exception as e:
            logger.warning("Embedding build failed: %s", e)
            self._embeddings = None

    def _load_or_build_embeddings_db(self):
        """Build or load embeddings using the SQLite backend.

        When embeddings are stale (content hash mismatch), the rebuild is
        deferred to a background thread so search returns immediately with
        BM25-only results. The next search after the build finishes will
        pick up the fresh embeddings via _ensure_embeddings_loaded().
        """
        if not self._db.has_embeddings():
            return

        content_hash = self._content_hash()
        stored_hash = self._db.get_metadata("embedding_hash")

        if stored_hash == content_hash and self._db.has_embeddings():
            return  # Embeddings are current; fetch on demand during lookup

        # Stale embeddings: rebuild in background to avoid blocking search.
        # Search will use BM25-only until the build completes.
        logger.info(
            "Embedding hash mismatch (stored=%s, current=%s). "
            "Rebuilding in background...",
            (stored_hash or "none")[:8], content_hash[:8],
        )
        import threading
        self._embedding_build_thread = threading.Thread(
            target=self._build_embeddings_background,
            args=(content_hash,),
            daemon=True,
            name="synapt-embedding-build",
        )
        self._embedding_build_thread.start()

    def _content_hash(self) -> str:
        """Hash of chunk IDs + text content for embedding cache invalidation.

        Uses the DB-level content_hash() when available to avoid
        materializing all chunks into Python objects (#435).
        """
        if self._db is not None:
            return self._db.content_hash()
        h = hashlib.sha256()
        for c in self._materialize_all_chunks():
            h.update(
                f"{c.id}|{c.user_text}|{c.assistant_text}|{c.tool_content}\n"
                .encode()
            )
        return h.hexdigest()[:16]

    def _filter_by_date(
        self,
        after: str | None = None,
        before: str | None = None,
    ) -> set[int] | None:
        """Return indices of chunks within the date range, or None for all.

        Uses ISO 8601 string comparison — works because the format is
        lexicographically ordered. Accepts date-only ("2026-02-28") or
        datetime ("2026-02-28T10:00:00Z").
        """
        if not after and not before:
            return None
        valid = set()
        for i, chunk in enumerate(self.chunks):
            ts = chunk.timestamp
            if not ts:
                continue
            if after and ts < after:
                continue
            if before and ts >= before:
                continue
            valid.add(i)
        return valid

    def _apply_recency_decay(
        self,
        scores: list[float],
        half_life: float = 30.0,
        now: datetime | None = None,
    ) -> list[float]:
        """Multiply scores by exp(-age_days * ln2 / half_life) for time decay.

        Args:
            scores: Scores per chunk, same order as self.chunks.
            half_life: Days for score to decay to ~50%. Default 30.
            now: Reference time. Defaults to UTC now.

        Returns:
            New list of decayed scores.
        """
        if half_life <= 0:
            return list(scores)
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        decay_rate = math.log(2) / half_life
        result = list(scores)
        for i, chunk in enumerate(self.chunks):
            if result[i] == 0.0 or not chunk.timestamp:
                continue
            try:
                ts = datetime.fromisoformat(chunk.timestamp.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
                result[i] *= math.exp(-decay_rate * age_days)
            except (ValueError, TypeError):
                pass
        return result

    def _decay_candidates(
        self,
        candidates: list[tuple[int, float]],
        half_life: float,
        now: datetime | None = None,
    ) -> list[tuple[int, float]]:
        """Apply recency decay to (chunk_idx, score) candidate pairs.

        Same formula as _apply_recency_decay but for sparse candidate lists.
        """
        if half_life <= 0 or not candidates:
            return candidates
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        decay_rate = math.log(2) / half_life
        for j, (idx, score) in enumerate(candidates):
            chunk = self._get_chunk(idx)
            if not chunk.timestamp:
                continue
            try:
                ts = datetime.fromisoformat(
                    chunk.timestamp.replace("Z", "+00:00")
                )
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
                candidates[j] = (idx, score * math.exp(-decay_rate * age_days))
            except (ValueError, TypeError):
                pass
        return candidates

    def _status_recency_multiplier(
        self,
        timestamp: str,
        now: datetime | None = None,
        half_life: float = 3.0,
    ) -> float:
        """Extra recency preference for status-style journal chunks.

        Status queries are asking for *current* open work. The general recency
        decay already helps, but old journal entries that happen to contain
        ``Next steps:`` can still compete too well with the latest carry-forward
        journal. This multiplier makes recent pending-work journals win more
        decisively without changing other intents.
        """
        if not timestamp:
            return 1.0
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
        except (ValueError, TypeError):
            return 1.0

        decay_rate = math.log(2) / half_life
        freshness = math.exp(-decay_rate * age_days)
        return 1.0 + (1.5 * freshness)

    def _apply_threshold_with_diagnostics(
        self,
        candidates: list[tuple[int, float]],
        threshold_ratio: float,
        search_mode: str,
        date_filter_active: bool = False,
        sessions_searched: int = 0,
        reference_candidates: list[tuple[int, float]] | None = None,
        embeddings_available: bool = True,
        semantic_search_used: bool = False,
    ) -> list[tuple[int, float]]:
        """Apply threshold filtering and record diagnostics if no candidates exist.

        Expects *candidates* sorted by score descending (highest first), with
        only positive scores (pre-filtered by caller).  Sets
        ``self._last_diagnostics`` when candidates is empty (``no_matches``).
        ``reference_candidates`` lets callers threshold on the pre-boost score
        domain while still ranking on the current candidate scores.
        """
        if not candidates:
            self._last_diagnostics = SearchDiagnostics(
                total_chunks=len(self.chunks),
                total_sessions=len(self.sessions),
                candidates_found=0,
                search_mode=search_mode,
                date_filter_active=date_filter_active,
                sessions_searched=sessions_searched,
                embeddings_available=embeddings_available,
                semantic_search_used=semantic_search_used,
                reason="no_matches",
                oldest_indexed_at=self._oldest_indexed_at,
            )
            return []

        if threshold_ratio > 0:
            reference = reference_candidates or candidates
            cutoff = max(s for _, s in reference) * threshold_ratio
            ref_scores = {idx: score for idx, score in reference}
            filtered = [
                (i, s) for i, s in candidates
                if ref_scores.get(i, s) >= cutoff
            ]
        else:
            filtered = candidates

        return filtered

    def _get_preceding_turn(self, chunk: TranscriptChunk) -> TranscriptChunk | None:
        """Find the chunk with turn_index - 1 in the same session (O(1) lookup)."""
        if chunk.turn_index == 0:
            return None
        return self._turn_lookup.get((chunk.session_id, chunk.turn_index - 1))

    def lookup(
        self,
        query: str,
        max_chunks: int = 5,
        max_tokens: int = 500,
        max_sessions: int | None = None,
        after: str | None = None,
        before: str | None = None,
        half_life: float | None = None,
        threshold_ratio: float = 0.2,
        depth: str = "full",
        include_archived: bool = False,
        include_historical: bool = False,
        now: datetime | None = None,
        knowledge_boost: float | None = None,
        max_knowledge: int | None = None,
        min_confidence: float = 0.0,
        context: int = 0,
        agent_id: str | None = None,
    ) -> str:
        """Search transcripts and return formatted context string.

        Grep-style parameters:
            min_confidence: Filter knowledge nodes below this confidence (0.0-1.0).
            context: Include N surrounding chunks per match (like grep -C N).

        Args:
            query: Search query (natural language or keywords).
            max_chunks: Maximum number of chunks to return.
            max_tokens: Approximate token budget (~4 chars/token).
            max_sessions: If set, only search N most recent sessions
                         (progressive mode). None = search all (global mode).
            after: Only include chunks with timestamp >= this ISO 8601 string.
            before: Only include chunks with timestamp < this ISO 8601 string.
            half_life: Days for recency decay to reach ~50%. 0 disables.
                None = use intent-based default (or 30.0 as fallback).
            threshold_ratio: Drop results below this fraction of top score. 0 disables.
            depth: "full" (default) = knowledge + journal + transcript.
                   "summary" = knowledge + journal only (no raw transcripts).
            include_archived: If True, include archived clusters in concise mode.
                In full mode, individual chunks are always searchable regardless.
            include_historical: If True, include superseded/contradicted knowledge
                nodes in results, clearly labeled as historical.
            now: Reference time for recency decay. Defaults to UTC now.
                Set to a conversation's last timestamp to enable meaningful
                recency bias on historical data.
            knowledge_boost: Override for knowledge node boost multiplier.
                None = use intent-classified default (typically 2.0).
            max_knowledge: Maximum number of knowledge blocks to include in
                the context. None = no cap (all qualifying nodes compete).
                Set to e.g. 5 to prevent knowledge from crowding out raw
                conversation chunks.

        Returns:
            Formatted string ready for prompt injection, or empty string.
            When empty, ``self._last_diagnostics`` explains why.
        """
        self._last_diagnostics = None
        self._last_search_summary = None
        self._last_conflicts: list[tuple[dict, dict]] = []

        # Allow knowledge-only search even when no transcript chunks exist
        # (cold-start: recall_save was called but no conversations ingested)
        has_knowledge = self._db and self._db.knowledge_count() > 0
        if not self.chunks and not has_knowledge:
            self._last_diagnostics = SearchDiagnostics(
                total_chunks=0,
                total_sessions=0,
                reason="empty_index",
                oldest_indexed_at=None,
            )
            return ""

        # Check query cache — skip if max_tokens=0 (diagnostics-only mode)
        cache_key = (
            query, max_chunks, max_tokens, max_sessions, after, before,
            half_life, threshold_ratio, depth, include_archived,
            include_historical, now, knowledge_boost, max_knowledge,
            agent_id,
        )
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            cached_result, cached_summary, cached_diagnostics = cached
            self._last_search_summary = (
                replace(cached_summary) if cached_summary is not None else None
            )
            self._last_diagnostics = (
                replace(cached_diagnostics)
                if cached_diagnostics is not None
                else None
            )
            return cached_result

        # Query intent classification — adjusts search parameters based on
        # the type of information being sought (recency, embedding weight,
        # knowledge boost). Only overrides half_life when caller didn't
        # provide an explicit value (half_life is None).
        emb_weight = 1.0
        kb_default = 2.0
        intent = ""
        caller_set_half_life = half_life is not None
        if half_life is None:
            half_life = 60.0
        try:
            from synapt.recall.hybrid import (
                classify_query_intent, intent_search_params,
                extract_temporal_range,
            )
            # Ablation toggle: disable intent classification
            if os.environ.get("SYNAPT_DISABLE_INTENT"):
                intent = "general"
            else:
                intent = classify_query_intent(query)
            params = intent_search_params(intent)
            emb_weight = params.get("emb_weight", emb_weight)
            kb_default = params.get("knowledge_boost", kb_default)
            if not caller_set_half_life:
                half_life = params.get("half_life", half_life)
            # Intent-based max_knowledge (only if caller didn't set it)
            if max_knowledge is None and "max_knowledge" in params:
                max_knowledge = params["max_knowledge"]

            # Auto-extract date range from temporal expressions in query
            # (only when caller didn't provide explicit date filters)
            if (
                after is None
                and before is None
                and not _env_flag("SYNAPT_DISABLE_TEMPORAL_EXTRACTION")
            ):
                extracted_after, extracted_before = extract_temporal_range(query, now=now)
                if extracted_after is not None:
                    after = extracted_after
                    before = extracted_before
                    logger.debug(
                        "Auto-extracted date range: after=%s, before=%s",
                        after, before,
                    )
        except Exception:
            logger.debug("Intent classification failed, using defaults", exc_info=True)

        # Apply content-profile adjustments (after intent, before caller overrides)
        if hasattr(self, "_adaptive"):
            _ap = self._adaptive
            # Raise max_knowledge floor for personal/mixed content
            if max_knowledge is None and _ap.max_knowledge_default is not None:
                max_knowledge = _ap.max_knowledge_default
            elif max_knowledge is not None and _ap.max_knowledge_default is not None:
                max_knowledge = max(max_knowledge, _ap.max_knowledge_default)
            kb_default = kb_default + _ap.knowledge_boost_adjust

        # Apply caller override for knowledge boost (after intent classification)
        _knowledge_boost = knowledge_boost if knowledge_boost is not None else kb_default

        retrieval_query = augment_query_for_intent(query, intent) if intent else query

        query_tokens = _tokenize(retrieval_query)
        if not query_tokens:
            self._last_diagnostics = SearchDiagnostics(
                total_chunks=len(self.chunks),
                total_sessions=len(self.sessions),
                reason="empty_query",
                oldest_indexed_at=self._oldest_indexed_at,
            )
            return ""

        date_filter = self._filter_by_date(after, before)

        if max_sessions is not None:
            result = self._progressive_lookup(
                retrieval_query, query_tokens, max_chunks, max_tokens, max_sessions,
                date_filter, after, before, half_life, threshold_ratio, depth,
                include_historical,
                emb_weight=emb_weight, knowledge_boost=_knowledge_boost,
                now=now, max_knowledge=max_knowledge,
                intent=intent, agent_id=agent_id,
            )
        else:
            result = self._global_lookup(
                retrieval_query, query_tokens, max_chunks, max_tokens, date_filter,
                after, before,
                half_life, threshold_ratio, depth, include_archived,
                include_historical,
                emb_weight=emb_weight, knowledge_boost=_knowledge_boost,
                now=now, max_knowledge=max_knowledge,
                intent=intent, min_confidence=min_confidence, context=context,
                agent_id=agent_id,
            )

        # Cache the result (LRU eviction when full)
        if len(self._query_cache) >= self._query_cache_max:
            # Evict oldest entry
            oldest = next(iter(self._query_cache))
            del self._query_cache[oldest]
        cached_summary = (
            replace(self._last_search_summary)
            if self._last_search_summary is not None
            else None
        )
        cached_diagnostics = (
            replace(self._last_diagnostics)
            if self._last_diagnostics is not None
            else None
        )
        self._query_cache[cache_key] = (
            result,
            cached_summary,
            cached_diagnostics,
        )
        return result

    def _global_lookup(
        self,
        query: str,
        query_tokens: list[str],
        max_chunks: int,
        max_tokens: int,
        date_filter: set[int] | None = None,
        after: str | None = None,
        before: str | None = None,
        half_life: float = 30.0,
        threshold_ratio: float = 0.2,
        depth: str = "full",
        include_archived: bool = False,
        include_historical: bool = False,
        emb_weight: float = 1.0,
        knowledge_boost: float = 2.0,
        now: datetime | None = None,
        max_knowledge: int | None = None,
        intent: str = "",
        min_confidence: float = 0.0,
        context: int = 0,
        agent_id: str | None = None,
    ) -> str:
        """Score all chunks globally, return top-K."""
        if self._db and (self._rowid_to_idx or self._db.knowledge_count() > 0):
            return self._global_lookup_fts(
                query, max_chunks, max_tokens, date_filter,
                after, before,
                half_life, threshold_ratio, depth, include_archived,
                include_historical,
                emb_weight=emb_weight, knowledge_boost=knowledge_boost,
                now=now, max_knowledge=max_knowledge, intent=intent,
                min_confidence=min_confidence, context=context,
                agent_id=agent_id,
            )
        return self._global_lookup_bm25(
            query, query_tokens, max_chunks, max_tokens, date_filter,
            half_life, threshold_ratio, depth,
            emb_weight=emb_weight, now=now, agent_id=agent_id,
        )

    def _global_lookup_fts(
        self,
        query: str,
        max_chunks: int,
        max_tokens: int,
        date_filter: set[int] | None,
        after: str | None,
        before: str | None,
        half_life: float,
        threshold_ratio: float,
        depth: str = "full",
        include_archived: bool = False,
        include_historical: bool = False,
        emb_weight: float = 1.0,
        knowledge_boost: float = 2.0,
        now: datetime | None = None,
        max_knowledge: int | None = None,
        intent: str = "",
        min_confidence: float = 0.0,
        context: int = 0,
        agent_id: str | None = None,
    ) -> str:
        """Global lookup using FTS5 (SQLite backend)."""
        # Extract entities once and share across knowledge + FTS search
        query_entities = [] if _env_flag("SYNAPT_DISABLE_ENTITY_COLLECTION") else extract_entities(query)

        # Search knowledge nodes first (ranked higher via boost)
        knowledge_results = self._search_knowledge(
            query, max_chunks, include_historical=include_historical,
            knowledge_boost=knowledge_boost, emb_weight=emb_weight,
            query_entities=query_entities, intent=intent,
            after=after, before=before,
        )
        semantic_search_used = self._last_knowledge_semantic_used

        # Apply min_confidence filter (grep-style noise reduction)
        if min_confidence > 0 and knowledge_results:
            knowledge_results = [
                n for n in knowledge_results
                if n.get("confidence", 0) >= min_confidence
            ]

        # Concise mode: search clusters directly, return only summaries
        if depth == "concise":
            return self._concise_lookup(
                query, max_chunks, max_tokens, knowledge_results,
                include_archived,
                semantic_search_used=semantic_search_used,
            )

        fts_results = self._db.fts_search(query, limit=max_chunks * 10)

        # Entity-anchored supplementary search: when the query mentions
        # entities, run two additional searches:
        # 1. Entity AND (content1 OR content2 OR ...) — finds entity
        #    chunks matching any content keyword. This is the key middle
        #    ground between strict AND (misses synonym/paraphrase) and
        #    pure entity search (too broad). Critical for multi-hop
        #    queries like "When did Caroline go to the LGBTQ support group?"
        #    where the AND query fails if the chunk uses "queer" instead.
        # 2. Entity-only search — catches chunks where entities co-occur
        #    but content words don't appear at all.
        #
        # For aggregation intent, entity-only search is the primary signal
        # since content terms ("activities", "partake") are too abstract to
        # match concrete answers ("plays guitar", "goes hiking"). Tier
        # discounts are reduced to near-parity with Tier 1.
        is_aggregation = intent == "aggregation"
        if query_entities and not _env_flag("SYNAPT_DISABLE_ENTITY_COLLECTION"):
            from synapt.recall.storage import (
                _build_entity_anchored_query,
                _escape_fts_tokens,
                _FTS_WEIGHTS,
            )
            entity_tokens = sorted(query_entities)  # deterministic ordering
            all_tokens = _escape_fts_tokens(query)
            content_tokens = [t for t in all_tokens if t.lower().strip('"') not in query_entities]
            seen_rowids = {r for r, _ in fts_results}

            # Tier 2: Entity-anchored OR (entity AND (content1 OR content2 ...))
            # Discount: 0.95 — slightly below tier 1 (full AND) which has
            # stronger co-occurrence signal. Aggregation: 0.98 (near-parity).
            t2_discount = 0.98 if is_aggregation else 0.95
            if content_tokens:
                anchored_q = _build_entity_anchored_query(entity_tokens, content_tokens)
                if anchored_q:
                    try:
                        anchored_rows = self._db.fts_search_raw(
                            anchored_q, limit=max_chunks * 8,
                        )
                        for rowid, score in anchored_rows:
                            if rowid not in seen_rowids:
                                fts_results.append((rowid, score * t2_discount))
                                seen_rowids.add(rowid)
                    except Exception:
                        logger.debug(
                            "Entity-anchored FTS query failed: %s",
                            anchored_q, exc_info=True,
                        )

            # Tier 2.5: Per-entity search for multi-entity queries.
            # When the query mentions 2+ entities (e.g., "What do Jon and
            # Gina have in common?"), searching "jon AND gina" only finds
            # chunks where both co-occur. But the answer often requires
            # combining facts from individual entity chunks (Jon's hobbies
            # from one session + Gina's hobbies from another).
            # Aggregation: 0.95 discount (from 0.90) since per-entity is
            # the primary gathering mechanism.
            t25_discount = 0.95 if is_aggregation else 0.90
            if len(entity_tokens) >= 2 and content_tokens:
                for ent in entity_tokens:
                    per_ent_q = _build_entity_anchored_query([ent], content_tokens)
                    if per_ent_q:
                        try:
                            per_ent_rows = self._db.fts_search_raw(
                                per_ent_q, limit=max_chunks * 4,
                            )
                            for rowid, score in per_ent_rows:
                                if rowid not in seen_rowids:
                                    fts_results.append((rowid, score * t25_discount))
                                    seen_rowids.add(rowid)
                        except Exception:
                            logger.debug(
                                "Per-entity FTS query failed: %s",
                                per_ent_q, exc_info=True,
                            )

            # Tier 3: Entity-only search (broadest)
            # Discount: 0.85 — weakest signal, just entity mention.
            # For multi-entity queries, search each entity individually
            # to avoid requiring co-occurrence.
            # Aggregation: 0.95 discount and larger limits — entity-only
            # is the primary retrieval mechanism for gathering scattered
            # facts about a person/topic across many sessions.
            t3_discount_multi = 0.95 if is_aggregation else 0.80
            t3_discount_single = 0.95 if is_aggregation else 0.85
            t3_limit_multi = max_chunks * 6 if is_aggregation else max_chunks * 3
            t3_limit_single = max_chunks * 8 if is_aggregation else max_chunks * 5
            if len(entity_tokens) >= 2:
                for ent in entity_tokens:
                    ent_fts = self._db.fts_search(ent, limit=t3_limit_multi)
                    for rowid, score in ent_fts:
                        if rowid not in seen_rowids:
                            fts_results.append((rowid, score * t3_discount_multi))
                            seen_rowids.add(rowid)
            else:
                entity_query = " ".join(entity_tokens)
                entity_fts = self._db.fts_search(entity_query, limit=t3_limit_single)
                for rowid, score in entity_fts:
                    if rowid not in seen_rowids:
                        fts_results.append((rowid, score * t3_discount_single))
                        seen_rowids.add(rowid)

        # Convert rowids to chunk indices, applying date, depth, and agent filters
        candidates: list[tuple[int, float]] = []
        _agent_filter = agent_id not in (None, "*")
        for rowid, score in fts_results:
            idx = self._rowid_to_idx.get(rowid)
            if idx is None:
                continue
            if date_filter is not None and idx not in date_filter:
                continue
            chunk = self._get_chunk(idx)
            # In summary mode, only include journal chunks (turn_index == -1)
            if depth == "summary" and chunk.turn_index >= 0:
                continue
            # Agent scoping: when agent_id is set, only return chunks owned
            # by this agent or legacy chunks (agent_id=None)
            if _agent_filter and chunk.agent_id is not None and chunk.agent_id != agent_id:
                continue
            candidates.append((idx, score))

        reference_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        # Recency decay — applied BEFORE RRF intentionally.
        # RRF is rank-based (not score-based), so decaying scores here biases
        # the *ranks* that RRF sees. Both BM25 and embedding lists are decayed
        # symmetrically, so recent results rank higher in both lists and RRF
        # naturally promotes them. This differs from pure RRF (score-agnostic)
        # but produces better results for session recall where recency matters.
        candidates = self._decay_candidates(candidates, half_life, now=now)

        # Hybrid search: RRF fusion between BM25/FTS and embedding similarity.
        # Unlike the old additive boost (score + sim * 3.0), RRF can surface
        # results that BM25 missed entirely — critical for paraphrased queries.
        self._ensure_embeddings_loaded()
        if self._embed_provider and self._all_embeddings:
            try:
                from synapt.recall.hybrid import (
                    embedding_search, embedding_search_numpy,
                    weighted_rrf_merge,
                )
                q_emb = self._embed_provider.embed_single(query)

                # BM25/FTS ranked list (idx, score)
                bm25_ranked = sorted(candidates, key=lambda x: x[1], reverse=True)
                bm25_ranked_ref = reference_candidates

                # Embedding ranked list (rowid, sim) → convert to (idx, sim)
                if self._emb_matrix is not None:
                    emb_raw = embedding_search_numpy(
                        q_emb, self._emb_matrix, self._emb_rowids,
                        limit=max_chunks * 10,
                    )
                else:
                    emb_raw = embedding_search(
                        q_emb, self._all_embeddings, limit=max_chunks * 10,
                    )
                semantic_search_used = True
                emb_ranked = []
                emb_ranked_ref = []
                for rowid, sim in emb_raw:
                    idx = self._rowid_to_idx.get(rowid)
                    if idx is None:
                        continue
                    if date_filter is not None and idx not in date_filter:
                        continue
                    emb_chunk = self._get_chunk(idx)
                    if depth == "summary" and emb_chunk.turn_index >= 0:
                        continue
                    if _agent_filter and emb_chunk.agent_id is not None and emb_chunk.agent_id != agent_id:
                        continue
                    emb_ranked_ref.append((idx, sim))
                    emb_ranked.append((idx, sim))
                emb_ranked = self._decay_candidates(emb_ranked, half_life, now=now)

                # Weighted RRF merge — emb_weight from intent classification
                _bm25_floor = int(os.environ.get("SYNAPT_BM25_FLOOR", "0"))
                reference_candidates = weighted_rrf_merge(
                    bm25_ranked_ref, emb_ranked_ref, emb_weight=emb_weight,
                    bm25_floor=_bm25_floor,
                )
                merged = weighted_rrf_merge(
                    bm25_ranked, emb_ranked, emb_weight=emb_weight,
                    bm25_floor=_bm25_floor,
                )
                candidates = merged
            except Exception as e:
                logger.warning("Hybrid search failed, using BM25 only: %s", e)
                candidates.sort(key=lambda x: x[1], reverse=True)
        else:
            logger.debug("Semantic search unavailable — using keyword matching only")
            candidates.sort(key=lambda x: x[1], reverse=True)

        top = [(i, s) for i, s in candidates[:max_chunks * 2] if s > 0]

        has_emb = bool(self._embed_provider and self._all_embeddings)
        top = self._apply_threshold_with_diagnostics(
            top, threshold_ratio, "fts_global",
            date_filter_active=date_filter is not None,
            reference_candidates=reference_candidates,
            embeddings_available=has_emb,
            semantic_search_used=semantic_search_used,
        )

        # Cross-encoder reranking (Phase 2): rerank after threshold so
        # threshold operates on RRF scores (correct domain), not cross-encoder
        # logits which can be negative and break ratio-based filtering.
        top = self._rerank_candidates(query, top)

        # Cross-session expansion: follow pre-computed links to ensure
        # results span multiple sessions when initial retrieval clusters.
        top = self._expand_cross_session(top, max_chunks)

        # Pass extra candidates to _format_results to compensate for
        # near-duplicate filtering — the token budget is the real limiter.
        # Context expansion (grep -C): for each matched chunk, also include
        # surrounding chunks from the same session.
        if context > 0:
            expanded = []
            seen = set()
            for idx, score in top:
                if idx >= len(self.chunks):
                    continue
                chunk = self._get_chunk(idx)
                sid = chunk.session_id
                ti = chunk.turn_index
                # Add surrounding turns from same session
                for offset in range(-context, context + 1):
                    target_ti = ti + offset
                    if target_ti < 0:
                        continue
                    # Find chunk with matching session_id and turn_index
                    for ci, c in enumerate(self.chunks):
                        if c.session_id == sid and c.turn_index == target_ti and ci not in seen:
                            # Context chunks get the matched chunk's score (discounted)
                            ctx_score = score if offset == 0 else score * 0.5
                            expanded.append((ci, ctx_score))
                            seen.add(ci)
                            break
            top = expanded

        dedup_headroom = _dedup_limit(max_chunks)
        return self._format_results(
            top[:dedup_headroom], max_tokens,
            knowledge_results=knowledge_results,
            query=query,
            max_knowledge=max_knowledge,
            intent=intent,
            agent_id=agent_id,
        )

    def _concise_lookup(
        self,
        query: str,
        max_chunks: int,
        max_tokens: int,
        knowledge_results: list[dict] | None = None,
        include_archived: bool = False,
        semantic_search_used: bool = False,
    ) -> str:
        """Search clusters directly and return only summaries.

        Used by depth="concise" — shows high-level topic overview without
        individual chunk details. Each cluster is rendered as a compact block.
        """
        cluster_hits = self._db.cluster_fts_search(
            query, limit=max_chunks * 3, include_archived=include_archived,
        )
        # Fallback: if cluster FTS found nothing, search chunks and map to
        # parent clusters so concise mode has the same coverage as full mode.
        if not cluster_hits and self._db:
            cluster_hits = self._db.chunk_fts_to_clusters(
                query, limit=max_chunks * 3, include_archived=include_archived,
            )
        if not cluster_hits and not knowledge_results:
            self._last_diagnostics = SearchDiagnostics(
                total_chunks=len(self.chunks),
                total_sessions=len(self.sessions),
                candidates_found=0,
                search_mode="concise",
                embeddings_available=bool(self._embed_provider),
                semantic_search_used=semantic_search_used,
                reason="no_matches",
                oldest_indexed_at=self._oldest_indexed_at,
            )
            return ""

        wm = self._working_memory
        lines = ["Past session context:"]
        token_count = 0
        access_items: list[dict] = []

        # Knowledge nodes first
        for node in (knowledge_results or []):
            block = self._format_knowledge_block(node)
            block_tokens = len(block) // 4
            if token_count + block_tokens > max_tokens and len(lines) > 1:
                break
            lines.append(block)
            token_count += block_tokens
            item_id = node.get("id", "")
            access_items.append({
                "item_type": "knowledge",
                "item_id": item_id,
                "score": node.get("score", 0.0),
            })
            wm.record("knowledge", item_id, node.get("content", ""))

        # Re-rank cluster hits by durability: durable clusters are more
        # valuable in concise mode (high-level context). Ephemeral clusters
        # (debugging narration, navigation) get score discounts.
        _durability_mult = {"durable": 1.0, "mixed": 0.8, "ephemeral": 0.5}
        ranked_hits = []
        for cluster_id, score in cluster_hits:
            info = self._db.get_cluster(cluster_id)
            if info is None:
                continue
            dur = info.get("durability", "")
            mult = _durability_mult.get(dur, 0.7)  # unclassified = 0.7
            ranked_hits.append((cluster_id, score * mult, info))
        ranked_hits.sort(key=lambda x: x[1], reverse=True)

        # Cluster summaries — pass query for snippet extraction
        for cluster_id, score, info in ranked_hits:
            block = self._format_cluster_block(cluster_id, info, query=query)
            block_tokens = len(block) // 4
            if token_count + block_tokens > max_tokens and len(lines) > 1:
                break
            lines.append(block)
            token_count += block_tokens
            access_items.append({
                "item_type": "cluster",
                "item_id": cluster_id,
                "score": score,
            })
            wm.record("cluster", cluster_id, info.get("topic", ""))

        # Record access (fire-and-forget)
        for item in access_items:
            if query:
                item["query"] = query
            if self._current_session_id:
                item["session_id"] = self._current_session_id
        if access_items and self._db:
            try:
                self._db.record_access(access_items, context="search")
            except Exception:
                pass

            try:
                from synapt.recall.promotion import (
                    check_promotions, execute_cheap_promotions,
                )
                for item in access_items:
                    actions = check_promotions(
                        self._db, item["item_type"], item["item_id"],
                    )
                    if actions:
                        execute_cheap_promotions(
                            self._db, item["item_type"],
                            item["item_id"], actions,
                        )
            except Exception:
                pass

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def _global_lookup_bm25(
        self,
        query: str,
        query_tokens: list[str],
        max_chunks: int,
        max_tokens: int,
        date_filter: set[int] | None,
        half_life: float,
        threshold_ratio: float,
        depth: str = "full",
        emb_weight: float = 1.0,
        now: datetime | None = None,
        agent_id: str | None = None,
    ) -> str:
        """Global lookup using in-memory BM25 (legacy fallback)."""
        # BM25 path has no cluster FTS — concise mode requires FTS5
        if depth == "concise":
            self._last_diagnostics = SearchDiagnostics(
                total_chunks=len(self.chunks),
                total_sessions=len(self.sessions),
                candidates_found=0,
                search_mode="concise",
                embeddings_available=False,
                reason="no_matches",
                oldest_indexed_at=self._oldest_indexed_at,
            )
            return ""

        scores = self._bm25.score(query_tokens)

        # In summary mode, zero out non-journal chunks
        if depth == "summary":
            for i, chunk in enumerate(self.chunks):
                if chunk.turn_index >= 0:
                    scores[i] = 0.0

        if date_filter is not None:
            for i in range(len(scores)):
                if i not in date_filter:
                    scores[i] = 0.0

        # Agent scoping: zero out chunks not owned by this agent
        if agent_id not in (None, "*"):
            for i, chunk in enumerate(self.chunks):
                if chunk.agent_id is not None and chunk.agent_id != agent_id:
                    scores[i] = 0.0

        reference_ranked = sorted(
            [(i, s) for i, s in enumerate(scores) if s > 0],
            key=lambda x: x[1], reverse=True,
        )

        if half_life > 0:
            scores = self._apply_recency_decay(scores, half_life=half_life, now=now)

        # Hybrid search: RRF fusion (same as FTS path but using in-memory BM25)
        semantic_search_used = False
        if self._embeddings and self._embed_provider:
            try:
                from synapt.recall.hybrid import (
                    embedding_search, weighted_rrf_merge,
                )
                q_emb = self._embed_provider.embed_single(query)

                # BM25 ranked list
                bm25_ranked = sorted(
                    [(i, s) for i, s in enumerate(scores) if s > 0],
                    key=lambda x: x[1], reverse=True,
                )
                bm25_ranked_ref = reference_ranked

                # Embedding ranked list (using in-memory embeddings)
                emb_dict = {i: emb for i, emb in enumerate(self._embeddings)}
                emb_raw = embedding_search(q_emb, emb_dict, limit=max_chunks * 10)
                semantic_search_used = True
                emb_ranked = [
                    (i, s) for i, s in emb_raw
                    if date_filter is None or i in date_filter
                ]
                emb_ranked_ref = list(emb_ranked)
                if depth == "summary":
                    emb_ranked = [
                        (i, s) for i, s in emb_ranked
                        if self._get_chunk(i).turn_index < 0
                    ]
                    emb_ranked_ref = [
                        (i, s) for i, s in emb_ranked_ref
                        if self._get_chunk(i).turn_index < 0
                    ]

                reference_ranked = weighted_rrf_merge(
                    bm25_ranked_ref, emb_ranked_ref, emb_weight=emb_weight,
                )
                merged = weighted_rrf_merge(
                    bm25_ranked, emb_ranked, emb_weight=emb_weight,
                )
                top = [(i, s) for i, s in merged[:max_chunks * 2] if s > 0]
            except Exception as e:
                logger.warning("Hybrid search failed, using BM25 only: %s", e)
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                top = [(i, s) for i, s in ranked[:max_chunks * 2] if s > 0]
        else:
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top = [(i, s) for i, s in ranked[:max_chunks * 2] if s > 0]

        has_emb = bool(self._embeddings and self._embed_provider)
        top = self._apply_threshold_with_diagnostics(
            top, threshold_ratio, "bm25_global",
            date_filter_active=date_filter is not None,
            reference_candidates=reference_ranked,
            embeddings_available=has_emb,
            semantic_search_used=semantic_search_used,
        )

        # Cross-encoder reranking (Phase 2): after threshold (see fts_global)
        top = self._rerank_candidates(query, top)
        top = self._expand_cross_session(top, max_chunks)

        dedup_headroom = _dedup_limit(max_chunks)
        return self._format_results(top[:dedup_headroom], max_tokens, query=query, agent_id=agent_id)

    def _progressive_lookup(
        self,
        query: str,
        query_tokens: list[str],
        max_chunks: int,
        max_tokens: int,
        max_sessions: int,
        date_filter: set[int] | None = None,
        after: str | None = None,
        before: str | None = None,
        half_life: float = 30.0,
        threshold_ratio: float = 0.2,
        depth: str = "full",
        include_historical: bool = False,
        emb_weight: float = 1.0,
        knowledge_boost: float = 2.0,
        now: datetime | None = None,
        max_knowledge: int | None = None,
        intent: str = "",
        agent_id: str | None = None,
    ) -> str:
        """Search sessions newest-first, stop early if enough hits found."""
        # Knowledge results are always searched globally (not per-session)
        knowledge_results = (
            self._search_knowledge(
                query, max_chunks, include_historical=include_historical,
                knowledge_boost=knowledge_boost, emb_weight=emb_weight,
                intent=intent, after=after, before=before,
            )
            if self._db else []
        )
        knowledge_semantic_search_used = self._last_knowledge_semantic_used

        if self._db and self._rowid_to_idx:
            return self._progressive_lookup_fts(
                query, max_chunks, max_tokens, max_sessions,
                date_filter, half_life, threshold_ratio,
                knowledge_results=knowledge_results,
                emb_weight=emb_weight,
                depth=depth,
                now=now,
                max_knowledge=max_knowledge,
                intent=intent,
                agent_id=agent_id,
                semantic_search_used=knowledge_semantic_search_used,
            )
        return self._progressive_lookup_bm25(
            query, query_tokens, max_chunks, max_tokens, max_sessions,
            date_filter, half_life, threshold_ratio,
            depth=depth,
            now=now,
            agent_id=agent_id,
        )

    def _progressive_lookup_fts(
        self,
        query: str,
        max_chunks: int,
        max_tokens: int,
        max_sessions: int,
        date_filter: set[int] | None,
        half_life: float,
        threshold_ratio: float,
        knowledge_results: list[dict] | None = None,
        depth: str = "full",
        emb_weight: float = 1.0,
        now: datetime | None = None,
        max_knowledge: int | None = None,
        intent: str = "",
        agent_id: str | None = None,
        semantic_search_used: bool = False,
    ) -> str:
        """Progressive session search using FTS5 (SQLite backend)."""
        hits: list[tuple[int, float]] = []
        sessions_searched = 0
        BM25_THRESHOLD = 2.0
        _agent_filter = agent_id not in (None, "*")

        for session_id in self._session_order:
            if sessions_searched >= max_sessions:
                break

            fts_results = self._db.fts_search_by_session(
                query, [session_id], limit=max_chunks * 5,
            )
            session_hits: list[tuple[int, float]] = []
            for rowid, score in fts_results:
                idx = self._rowid_to_idx.get(rowid)
                if idx is None:
                    continue
                if date_filter is not None and idx not in date_filter:
                    continue
                chunk = self._get_chunk(idx)
                # In summary mode, skip raw transcript chunks (keep journal only)
                if depth == "summary" and chunk.turn_index >= 0:
                    continue
                if _agent_filter and chunk.agent_id is not None and chunk.agent_id != agent_id:
                    continue
                session_hits.append((idx, score))

            if not session_hits:
                continue

            hits.extend(session_hits)
            sessions_searched += 1

            high_quality = sum(1 for _, s in hits if s > BM25_THRESHOLD)
            if high_quality >= 3 and len(hits) >= max_chunks:
                break

        reference_hits = sorted(hits, key=lambda x: x[1], reverse=True)

        # Recency decay — applied BEFORE RRF intentionally (see _global_lookup_fts
        # for rationale). Both BM25 and embedding lists are decayed symmetrically
        # so RRF ranks reflect recency bias consistently.
        hits = self._decay_candidates(hits, half_life, now=now)

        # Hybrid: merge FTS progressive hits with global embedding search.
        # Progressive FTS is session-scoped, but embeddings intentionally
        # search globally — semantic recall should be broader than keyword
        # recall, surfacing paraphrased matches from any session even when
        # max_sessions limits the BM25 search window.
        self._ensure_embeddings_loaded()
        if self._embed_provider and self._all_embeddings:
            try:
                from synapt.recall.hybrid import (
                    embedding_search, embedding_search_numpy,
                    weighted_rrf_merge,
                )
                q_emb = self._embed_provider.embed_single(query)
                if self._emb_matrix is not None:
                    emb_raw = embedding_search_numpy(
                        q_emb, self._emb_matrix, self._emb_rowids,
                        limit=max_chunks * 5,
                    )
                else:
                    emb_raw = embedding_search(
                        q_emb, self._all_embeddings, limit=max_chunks * 5,
                    )
                semantic_search_used = True
                emb_ranked = []
                emb_ranked_ref = []
                for rowid, sim in emb_raw:
                    idx = self._rowid_to_idx.get(rowid)
                    if idx is None:
                        continue
                    if date_filter is not None and idx not in date_filter:
                        continue
                    emb_chunk = self._get_chunk(idx)
                    if depth == "summary" and emb_chunk.turn_index >= 0:
                        continue
                    if _agent_filter and emb_chunk.agent_id is not None and emb_chunk.agent_id != agent_id:
                        continue
                    emb_ranked_ref.append((idx, sim))
                    emb_ranked.append((idx, sim))
                emb_ranked = self._decay_candidates(emb_ranked, half_life, now=now)

                bm25_ranked = sorted(hits, key=lambda x: x[1], reverse=True)
                reference_hits = weighted_rrf_merge(
                    reference_hits, emb_ranked_ref, emb_weight=emb_weight,
                )
                merged = weighted_rrf_merge(
                    bm25_ranked, emb_ranked, emb_weight=emb_weight,
                )
                hits = merged
            except Exception as e:
                logger.warning("Progressive hybrid failed: %s", e)
                hits.sort(key=lambda x: x[1], reverse=True)
        else:
            hits.sort(key=lambda x: x[1], reverse=True)

        # Bound candidates before expensive operations
        hits = hits[:max_chunks * 2]

        hits = self._apply_threshold_with_diagnostics(
            hits, threshold_ratio, "fts_progressive",
            date_filter_active=date_filter is not None,
            sessions_searched=sessions_searched,
            reference_candidates=reference_hits,
            semantic_search_used=semantic_search_used,
        )

        # Cross-encoder reranking (Phase 2): after threshold (see fts_global)
        hits = self._rerank_candidates(query, hits)
        hits = self._expand_cross_session(hits, max_chunks)

        return self._format_results(
            hits[:max_chunks], max_tokens,
            knowledge_results=knowledge_results,
            query=query,
            max_knowledge=max_knowledge,
            intent=intent,
            agent_id=agent_id,
        )

    def _progressive_lookup_bm25(
        self,
        query: str,
        query_tokens: list[str],
        max_chunks: int,
        max_tokens: int,
        max_sessions: int,
        date_filter: set[int] | None,
        half_life: float,
        threshold_ratio: float,
        depth: str = "full",
        now: datetime | None = None,
        agent_id: str | None = None,
    ) -> str:
        """Progressive session search using in-memory BM25 (legacy fallback)."""
        bm25_scores = self._bm25.score(query_tokens)

        # In summary mode, zero out raw transcript chunks (keep journal only)
        if depth == "summary":
            for i, chunk in enumerate(self.chunks):
                if chunk.turn_index >= 0:
                    bm25_scores[i] = 0.0

        # Agent scoping: zero out chunks not owned by this agent
        if agent_id not in (None, "*"):
            for i, chunk in enumerate(self.chunks):
                if chunk.agent_id is not None and chunk.agent_id != agent_id:
                    bm25_scores[i] = 0.0

        reference_scores = list(bm25_scores)

        if half_life > 0:
            bm25_scores = self._apply_recency_decay(
                bm25_scores, half_life=half_life, now=now,
            )

        session_chunk_indices: dict[str, list[int]] = {}
        for i, chunk in enumerate(self.chunks):
            if date_filter is not None and i not in date_filter:
                continue
            session_chunk_indices.setdefault(chunk.session_id, []).append(i)

        hits: list[tuple[int, float]] = []
        sessions_searched = 0
        BM25_THRESHOLD = 2.0

        for session_id in self._session_order:
            if sessions_searched >= max_sessions:
                break
            indices = session_chunk_indices.get(session_id, [])
            if not indices:
                continue
            session_hits = [(i, bm25_scores[i]) for i in indices if bm25_scores[i] > 0]
            session_hits.sort(key=lambda x: x[1], reverse=True)
            hits.extend(session_hits)
            sessions_searched += 1

            high_quality = sum(1 for _, s in hits if s > BM25_THRESHOLD)
            if high_quality >= 3 and len(hits) >= max_chunks:
                break

        reference_hits = [(i, reference_scores[i]) for i, _ in hits]
        reference_hits.sort(key=lambda x: x[1], reverse=True)
        hits.sort(key=lambda x: x[1], reverse=True)

        # Bound candidates before expensive operations
        hits = hits[:max_chunks * 2]

        hits = self._apply_threshold_with_diagnostics(
            hits, threshold_ratio, "bm25_progressive",
            date_filter_active=date_filter is not None,
            sessions_searched=sessions_searched,
            reference_candidates=reference_hits,
        )

        # Cross-encoder reranking (Phase 2): after threshold (see fts_global)
        hits = self._rerank_candidates(query, hits)
        hits = self._expand_cross_session(hits, max_chunks)

        return self._format_results(
            hits[:max_chunks], max_tokens,
            query=query,
            agent_id=agent_id,
        )

    def _search_knowledge(
        self,
        query: str,
        max_results: int = 5,
        include_historical: bool = False,
        knowledge_boost: float = 2.0,
        emb_weight: float = 1.0,
        query_entities: set[str] | None = None,
        intent: str = "",
        after: str | None = None,
        before: str | None = None,
    ) -> list[dict]:
        """Search knowledge nodes via FTS5 + embedding hybrid.

        Returns a list of knowledge node dicts with an added 'score' key.
        Knowledge nodes are boosted by ``knowledge_boost`` vs chunk scores.
        If include_historical is True, also returns contradicted/superseded nodes.
        """
        self._last_knowledge_semantic_used = False
        if not self._db:
            return []
        try:
            # Aggregation queries gather facts scattered across many sessions,
            # so widen the search net to surface more knowledge nodes.
            search_mult = 4 if intent == "aggregation" else 2
            fts_hits = self._db.knowledge_fts_search(
                query, limit=max_results * search_mult,
                include_historical=include_historical,
            )

            # Aggregation entity-only search: for queries like "What activities
            # does Melanie do?", content words ("activities", "do") are too
            # abstract to match knowledge nodes ("plays guitar"). Search by
            # entity name alone to gather all facts about the person/topic.
            if intent == "aggregation" and query_entities:
                seen_kn = {r for r, _ in fts_hits}
                for ent in sorted(query_entities):
                    ent_hits = self._db.knowledge_fts_search(
                        ent, limit=max_results * 3,
                        include_historical=include_historical,
                    )
                    for rowid, score in ent_hits:
                        if rowid not in seen_kn:
                            fts_hits.append((rowid, score * 0.90))
                            seen_kn.add(rowid)

            # Hybrid: also search knowledge embeddings
            self._ensure_embeddings_loaded()
            emb_hits: list[tuple[int, float]] = []
            if self._embed_provider and self._knowledge_embeddings:
                try:
                    from synapt.recall.hybrid import embedding_search
                    q_emb = self._embed_provider.embed_single(query)
                    emb_hits = embedding_search(
                        q_emb, self._knowledge_embeddings,
                        limit=max_results * search_mult,
                    )
                    self._last_knowledge_semantic_used = True
                except Exception:
                    pass

            # Merge FTS and embedding results via RRF
            if fts_hits and emb_hits:
                from synapt.recall.hybrid import weighted_rrf_merge
                merged = weighted_rrf_merge(
                    fts_hits, emb_hits, emb_weight=emb_weight,
                )
            elif fts_hits:
                merged = fts_hits
            elif emb_hits:
                merged = emb_hits
            else:
                return []

            all_rowids = [r for r, _ in merged]
            # Use RRF-fused scores (not raw FTS/embedding scores) for ranking
            score_map = dict(merged)

            nodes_by_rowid = self._db.knowledge_by_rowid(all_rowids)
            # Coverage gate: require ≥1 distinct query token to appear in the
            # node content, OR a strong embedding match (sim > 0.4). This lets
            # semantic matches through even when BM25 tokens don't overlap.
            # Knowledge nodes are already distilled facts, so a single token
            # overlap (e.g. a person's name) is a meaningful signal.
            query_tokens = set(_tokenize(query))
            min_matches = max(1, round(len(query_tokens) * 0.2))
            emb_rowids = {r for r, s in emb_hits if s > 0.4}

            # Entity extraction: boost knowledge nodes mentioning query entities
            if query_entities is None:
                query_entities = extract_entities(query)

            # Filter expired knowledge nodes (valid_until in the past)
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            results = []
            seen_ids: set[str] = set()
            for rowid in all_rowids:
                node = nodes_by_rowid.get(rowid)
                if not node:
                    continue
                node_id = node.get("id", str(rowid))
                if node_id in seen_ids:
                    continue
                # Skip expired nodes unless include_historical is set
                valid_until = node.get("valid_until")
                if (
                    valid_until
                    and not include_historical
                    and not _env_flag("SYNAPT_DISABLE_KNOWLEDGE_EXPIRY")
                    and not (intent == "temporal" and (after or before))
                ):
                    try:
                        if valid_until[:10] < now_iso:
                            continue  # Expired — skip
                    except (TypeError, IndexError):
                        pass  # Invalid date format — don't filter
                temporal_overlap = True
                has_temporal_bounds = False
                if intent == "temporal" and (after or before):
                    temporal_overlap, has_temporal_bounds = (
                        _knowledge_overlaps_temporal_window(node, after, before)
                    )
                    if not temporal_overlap:
                        continue
                node_content = node.get("content", "")
                node_tokens = set(_tokenize(node_content))
                token_overlap = len(query_tokens & node_tokens) >= min_matches
                strong_emb = rowid in emb_rowids
                if token_overlap or strong_emb:
                    conf = node.get("confidence", 0.5)
                    if not isinstance(conf, (int, float)):
                        conf = 0.5
                    # Confidence gate: low-confidence knowledge (<0.4) gets
                    # no boost to avoid polluting results with weak facts.
                    # Above the gate, scale by confidence AND specificity.
                    # Reuses consolidation's _lacks_specificity which checks
                    # file paths, versions, CLI flags, CamelCase/snake_case,
                    # PR/issue refs, and proper nouns. Short-but-specific
                    # facts like "Pin croniter to v1.3.8" get full boost.
                    if conf < 0.4:
                        effective_boost = 1.0
                    else:
                        if _env_flag("SYNAPT_DISABLE_SPECIFICITY_SCORING"):
                            specificity = 1.0
                        else:
                            from synapt.recall.consolidate import _lacks_specificity
                            if _lacks_specificity(node_content, threshold=200):
                                specificity = 0.5  # abstract/generic — reduced boost
                            else:
                                specificity = 1.0  # specific (paths, versions, names)
                        effective_boost = 1.0 + conf * specificity * knowledge_boost
                    # Entity boost: prefer knowledge about the specific
                    # entities mentioned in the query (e.g. person names).
                    if query_entities and not _env_flag("SYNAPT_DISABLE_ENTITY_COLLECTION"):
                        if any(e in node_content.lower() for e in query_entities):
                            effective_boost *= ENTITY_BOOST
                    # Category-intent alignment: boost knowledge nodes whose
                    # category matches the query intent. This ensures that
                    # decision queries surface decision nodes (not generic
                    # tooling/architecture), debug queries surface debugging
                    # nodes, etc.
                    node_cat = node.get("category", "")
                    aligned_cats = _CAT_INTENT_MAP.get(intent, set())
                    if node_cat in aligned_cats:
                        effective_boost *= 1.5
                    if (
                        intent == "temporal"
                        and temporal_overlap
                        and has_temporal_bounds
                    ):
                        effective_boost *= 1.2
                    raw_score = score_map.get(rowid, 0.0)
                    node["base_score"] = raw_score
                    node["score"] = raw_score * effective_boost
                    results.append(node)
                    seen_ids.add(node_id)

            results.sort(key=lambda n: n.get("score", 0.0), reverse=True)

            # Confidence-based dedup: when multiple nodes share a lineage
            # (i.e. one superseded another), keep the highest-confidence one
            # as primary and demote others. This serves as the automatic
            # fallback when no user has resolved the contradiction.
            if include_historical:
                results = self._dedup_knowledge_by_lineage(results)

            # Co-retrieval conflict detection (Phase 8c): when two active
            # nodes in the same category have divergent content, queue a
            # pending contradiction for user review. Has its own try/except
            # internally — never disrupts search results.
            self._last_conflicts = self._detect_co_retrieval_conflicts(results)

            return results
        except Exception:
            return []

    @staticmethod
    def _dedup_knowledge_by_lineage(nodes: list[dict]) -> list[dict]:
        """Within each lineage, keep only the highest-confidence node.

        Nodes without a lineage_id pass through unchanged. When two nodes
        share a lineage, the lower-confidence one is dropped — this is the
        confidence-based fallback for unresolved contradictions.
        """
        lineage_best: dict[str, dict] = {}
        no_lineage: list[dict] = []
        for node in nodes:
            lid = node.get("lineage_id", "")
            if not lid:
                no_lineage.append(node)
                continue
            existing = lineage_best.get(lid)
            if existing is None or node.get("confidence", 0) > existing.get("confidence", 0):
                lineage_best[lid] = node
        return no_lineage + list(lineage_best.values())

    def _detect_co_retrieval_conflicts(self, results: list[dict]) -> list[tuple[dict, dict]]:
        """Detect conflicting knowledge nodes in the same result set.

        When two active nodes share a category but have low keyword overlap
        (Jaccard < 0.3), they may contradict each other. Queues a pending
        contradiction for user review. Best-effort: never fails the search.

        Returns list of (old_node, new_node) pairs that were flagged, so
        callers can surface warnings in search results.
        """
        detected: list[tuple[dict, dict]] = []
        if not self._db or len(results) < 2:
            return detected
        # Only consider active, confident nodes
        active = [
            n for n in results
            if n.get("status", "active") == "active"
            and n.get("confidence", 0) >= 0.4
        ]
        if len(active) < 2:
            return detected
        try:
            # Pre-compute token sets once per node (avoids redundant
            # _tokenize calls in the pairwise loop)
            token_cache: dict[str, set[str]] = {}
            for node in active:
                nid = node.get("id", "")
                token_cache[nid] = set(_tokenize(node.get("content", "")))

            # Load all pending old_node_ids in one query
            pending_ids = {
                p["old_node_id"]
                for p in self._db.list_pending_contradictions()
            }

            # Group by category
            by_cat: dict[str, list[dict]] = {}
            for node in active:
                by_cat.setdefault(node.get("category", ""), []).append(node)

            for cat_nodes in by_cat.values():
                if len(cat_nodes) < 2:
                    continue
                for i in range(len(cat_nodes)):
                    for j in range(i + 1, len(cat_nodes)):
                        a, b = cat_nodes[i], cat_nodes[j]
                        # Skip nodes already in the same lineage
                        lid_a = a.get("lineage_id", "")
                        lid_b = b.get("lineage_id", "")
                        if lid_a and lid_a == lid_b:
                            continue
                        # Compute keyword divergence
                        id_a = a.get("id", "")
                        id_b = b.get("id", "")
                        kw_a = token_cache.get(id_a, set())
                        kw_b = token_cache.get(id_b, set())
                        if not kw_a or not kw_b:
                            continue
                        jaccard = len(kw_a & kw_b) / len(kw_a | kw_b)
                        if jaccard >= 0.3:
                            continue  # Similar enough — not a conflict
                        # Check dedup: skip if already pending for either node
                        if id_a in pending_ids or id_b in pending_ids:
                            continue
                        # Queue the lower-confidence node as the "old" one
                        if a.get("confidence", 0) >= b.get("confidence", 0):
                            old, new = b, a
                        else:
                            old, new = a, b
                        self._db.add_pending_contradiction(
                            old_node_id=old["id"],
                            new_content=new.get("content", ""),
                            category=old.get("category", ""),
                            reason=f"Co-retrieved with conflicting node [{new['id']}]",
                            detected_by="co-retrieval",
                        )
                        detected.append((old, new))
                        pending_ids.add(old["id"])  # Prevent further dups this batch
                        logger.debug(
                            "Co-retrieval conflict: [%s] vs [%s] (jaccard=%.2f)",
                            old["id"], new["id"], jaccard,
                        )
        except Exception:
            pass  # Best-effort — never disrupt search
        return detected

    @staticmethod
    def _format_knowledge_block(node: dict) -> str:
        """Format a single knowledge node for search results."""
        conf = node.get("confidence", 0.5)
        conf_label = "high" if conf >= 0.7 else "medium" if conf >= 0.4 else "low"
        category = node.get("category", "unknown")
        content = node.get("content", "")
        sources = node.get("source_sessions", [])
        updated = (node.get("updated_at", "") or "")[:10]
        status = node.get("status", "active")

        # Freshness label from updated_at
        freshness = ""
        if updated and len(updated) >= 10:
            try:
                updated_dt = datetime.fromisoformat(updated)
                now = datetime.now(timezone.utc)
                if updated_dt.tzinfo is None:
                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
                days_ago = (now - updated_dt).days
                if days_ago == 0:
                    freshness = "today"
                elif days_ago == 1:
                    freshness = "yesterday"
                elif days_ago < 7:
                    freshness = f"{days_ago}d ago"
                elif days_ago < 30:
                    freshness = f"{days_ago // 7}w ago"
                else:
                    freshness = updated
            except (ValueError, TypeError):
                freshness = updated

        node_id = node.get("id", "")
        id_tag = f" #{node_id}" if node_id else ""

        # Historical/superseded/contradicted/retracted/contested labels
        if status == "contradicted":
            header = f"--- [knowledge{id_tag}, CONTRADICTED] {category} ---"
        elif status == "superseded":
            header = f"--- [knowledge{id_tag}, SUPERSEDED] {category} ---"
        elif status == "retracted":
            header = f"--- [knowledge{id_tag}, RETRACTED] {category} ---"
        elif status == "contested":
            # Fix B contested-memory-lifecycle reframe (internal design spec, section 10.6,
            # fixture d): shown only when
            # explicitly surfaced via include_historical=True -- hidden from default search by
            # the status != 'active' FTS gate (storage.py's knowledge_fts_search).
            header = f"--- [knowledge{id_tag}, CONTESTED] {category} ---"
        else:
            header = f"--- [knowledge{id_tag}] {category} ({conf_label}, {freshness or updated}) ---"

        # Temporal metadata with source attribution
        if sources:
            source_refs = ", ".join(s[:8] for s in sources[-5:])
            if len(sources) > 5:
                source_refs += f" +{len(sources) - 5} more"
            meta_parts = [f"Sources: {source_refs}"]
        else:
            meta_parts = []
        valid_from = node.get("valid_from")
        valid_until = node.get("valid_until")
        if valid_from and valid_until:
            meta_parts.append(f"Valid {valid_from[:10]} to {valid_until[:10]}")
        elif valid_from:
            meta_parts.append(f"Current since {valid_from[:10]}")

        # Supersession note — make it prominent
        note = node.get("contradiction_note", "")
        if status in ("contradicted", "superseded") and note:
            meta_parts.append(f"Note: {note}")

        meta_line = f"[{'. '.join(meta_parts)}]" if meta_parts else ""
        return f"{header}\n{content}\n{meta_line}" if meta_line else f"{header}\n{content}"

    def _format_conflict_notice(self) -> str:
        """Render a short user-facing notice for co-retrieval contradictions."""
        if not self._last_conflicts:
            return ""

        previews: list[str] = []
        for old, new in self._last_conflicts[:2]:
            old_text = (old.get("content", "") or old.get("id", "unknown"))[:50]
            new_text = (new.get("content", "") or new.get("id", "unknown"))[:50]
            previews.append(f'"{old_text}" vs "{new_text}"')

        summary = "; ".join(previews)
        extra = ""
        if len(self._last_conflicts) > 2:
            extra = f" (+{len(self._last_conflicts) - 2} more)"
        return (
            "[Potential contradiction detected in retrieved knowledge: "
            f"{summary}{extra}. Review with "
            "`recall_contradict(action=\"list\")`.]"
        )

    @staticmethod
    def _annotate_boosted_block(block: str, label: str) -> str:
        """Append a boost label to the first line of a rendered result block."""
        if not block:
            return block
        lines = block.splitlines()
        if not lines:
            return block
        lines[0] = f"{lines[0]} [{label}]"
        return "\n".join(lines)
    def _format_results(
        self,
        ranked: list[tuple[int, float]],
        max_tokens: int,
        knowledge_results: list[dict] | None = None,
        query: str = "",
        max_knowledge: int | None = None,
        intent: str = "",
        agent_id: str | None = None,
    ) -> str:
        """Format ranked chunk indices into a context string.

        When clusters exist in the DB, chunks sharing a cluster are grouped
        and displayed as a single cluster summary block — saving token budget
        for more diverse results. Chunks not in any cluster (or sole
        representatives of a cluster) display as raw chunk blocks.
        """
        if not ranked and not knowledge_results:
            return ""

        lines = ["Past session context:"]
        token_count = 0

        # Build cluster grouping and interleave by relevance score
        if os.environ.get("SYNAPT_DISABLE_CLUSTERS"):
            cluster_groups, ungrouped = {}, ranked
        else:
            cluster_groups, ungrouped = self._group_by_cluster(ranked)

        # Build a unified emit list sorted by relevance.
        # Knowledge nodes, clusters, and raw chunks all compete on score.
        # Each item: (score, block_text, item_type, item_id, sort_ts) for
        # access tracking plus display-only chronological ordering.
        emit_items: list[tuple[float, str, str, str, str]] = []

        # Include knowledge nodes in the unified ranking instead of
        # unconditionally prepending them (#284). They compete fairly
        # with chunks/clusters — irrelevant knowledge nodes drop below
        # high-quality transcript chunks.
        # Track which chunk indices are already in ranked results
        ranked_indices = {idx for idx, _ in ranked}

        # Limit source expansion to prevent knowledge nodes from crowding out
        # regular ranked chunks. Each expanded knowledge node adds source chunks
        # that compete in the token budget — too many displace relevant results.
        # When max_knowledge is set, only expand sources for those top-N nodes.
        # When uncapped, limit to the top 10 nodes for source expansion.
        knowledge_to_expand = knowledge_results or []
        if knowledge_to_expand:
            sorted_kn = sorted(
                knowledge_to_expand, key=lambda n: n.get("score", 0.0), reverse=True
            )
            expand_limit = max_knowledge if max_knowledge is not None else 10
            knowledge_to_expand = sorted_kn[:expand_limit]

        # Only format knowledge blocks that could actually be emitted
        # (avoids wasting _format_knowledge_block calls on nodes past the cap).
        kn_sorted = sorted(
            (knowledge_results or []),
            key=lambda n: n.get("score", 0.0), reverse=True,
        )
        for node in kn_sorted:
            block = self._format_knowledge_block(node)
            node_score = node.get("score", 0.0)
            sort_ts = node.get("valid_from") or node.get("valid_until") or ""
            emit_items.append((node_score, block, "knowledge", node.get("id", ""), sort_ts))

        # Source expansion only for the top knowledge nodes that will be emitted.
        # Use the unboosted base score (from FTS5/embedding) for source chunk
        # ranking — knowledge_boost should only elevate the knowledge block
        # itself, not inflate source chunk scores above regular ranked chunks.

        # Pre-build lookup indexes for O(1) source expansion:
        #   (session_id, turn_index) → [chunk_indices]
        #   session_id → [chunk_indices]
        _chunk_by_turn: dict[tuple[str, int], list[int]] = {}
        _chunks_by_session: dict[str, list[int]] = {}
        for i, chunk in enumerate(self.chunks):
            key = (chunk.session_id, chunk.turn_index)
            _chunk_by_turn.setdefault(key, []).append(i)
            _chunks_by_session.setdefault(chunk.session_id, []).append(i)

        max_per_node = 1 if max_knowledge is not None else 3
        for node in knowledge_to_expand:
            node_score = node.get("score", 0.0)
            base_score = node.get("base_score", node_score)
            source_offsets = node.get("source_offsets", [])
            source_sessions = node.get("source_sessions", [])
            if node_score <= 0:
                continue
            if not source_offsets and not source_sessions:
                continue

            # Offset-based source expansion: emit compact snippets
            if source_offsets:
                snippets_emitted = 0
                for offset in source_offsets[:max_per_node]:
                    sid = offset.get("s", "")
                    tidx = offset.get("t", -1)
                    begin = offset.get("b", 0)
                    end = offset.get("e", 0)
                    key = (sid, tidx)
                    indices = _chunk_by_turn.get(key, [])
                    if not indices:
                        continue
                    i = indices[0]
                    if i in ranked_indices:
                        continue
                    chunk = self._get_chunk(i)
                    # Agent scoping: skip source chunks from other agents
                    if agent_id not in (None, "*") and chunk.agent_id is not None and chunk.agent_id != agent_id:
                        continue
                    full_text = (chunk.user_text or "") + " " + (chunk.assistant_text or "")
                    snippet = full_text[begin:end].strip()
                    if not snippet:
                        continue
                    # Compact snippet block with source attribution
                    fpath = offset.get("f", "")
                    if fpath:
                        fname = Path(fpath).name
                        source_label = f"{fname} turn {tidx}"
                    else:
                        source_label = f"{_short_sid(sid)} turn {tidx}"
                    block = (
                        f"--- [source: {source_label}] ---\n"
                        f"{snippet}"
                    )
                    source_score = base_score * 0.6
                    emit_items.append((source_score, block, "chunk", chunk.id, chunk.timestamp))
                    ranked_indices.add(i)
                    snippets_emitted += 1
                if snippets_emitted > 0:
                    continue  # Skip keyword fallback

            # Keyword fallback: query-adaptive matching against source sessions
            if source_sessions and query:
                source_candidates: list[tuple[int, float]] = []
                query_toks = set(_tokenize(query))
                node_toks = set(_tokenize(node.get("content", "")))
                match_toks = query_toks | node_toks
                for sid in source_sessions:
                    for i in _chunks_by_session.get(sid, []):
                        if i in ranked_indices:
                            continue
                        chunk = self._get_chunk(i)
                        if chunk.turn_index < 0:
                            continue
                        # Agent scoping: skip source chunks from other agents
                        if agent_id not in (None, "*") and chunk.agent_id is not None and chunk.agent_id != agent_id:
                            continue
                        chunk_toks = set(_tokenize(chunk.text))
                        overlap = len(chunk_toks & match_toks)
                        if overlap < 2:
                            continue
                        source_candidates.append((i, overlap))
                source_candidates.sort(key=lambda x: x[1], reverse=True)
                for i, overlap in source_candidates[:max_per_node]:
                    source_score = base_score * 0.6
                    cblock = self._format_chunk_block(i, intent=intent, query=query)
                    emit_items.append(
                        (source_score, cblock, "chunk", self._get_chunk(i).id, self._get_chunk(i).timestamp)
                    )
                    ranked_indices.add(i)

        for cluster_id, (cluster_info, member_indices, max_score) in cluster_groups.items():
            block = self._format_cluster_block(cluster_id, cluster_info, query=query)
            sort_ts = cluster_info.get("date_start", "") or cluster_info.get("date_end", "")
            emit_items.append((max_score, block, "cluster", cluster_id, sort_ts))

        # Track which chunk IDs are sub-chunks (fragments of a split turn).
        # Used below for per-session sub-chunk capping.
        _subchunk_ids: set[str] = set()

        for idx, score in ungrouped:
            chunk = self._get_chunk(idx)
            if chunk.user_text.startswith("(context:"):
                _subchunk_ids.add(chunk.id)
            # Journal decision boost: journal chunks with "Decisions:" content
            # rank higher for decision-intent queries. This surfaces the raw
            # rationale instead of distilled knowledge nodes.
            if chunk.turn_index < 0:
                if intent == "decision" and "Decisions:" in chunk.assistant_text:
                    score *= 1.3
                elif intent == "status" and "Next steps:" in chunk.assistant_text:
                    score *= 1.5
                    score *= self._status_recency_multiplier(chunk.timestamp)
            block = self._format_chunk_block(idx, intent=intent, query=query)
            emit_items.append((score, block, "chunk", chunk.id, chunk.timestamp))

        # Apply adaptive score boosts before final sort:
        # 1. Working memory: items seen recently this session (1.5x / 2.0x)
        # 2. Access frequency: items drilled into in past sessions (up to 1.3x)
        #
        # IMPORTANT: boosts are capped so they can never promote a result above
        # the top unboosted score. This prevents working memory contamination
        # from previous queries from overriding relevance — boosts act as a
        # tiebreaker between similarly-scored results, not a rank override.
        wm = self._working_memory
        _disable_boosts = os.environ.get("SYNAPT_DISABLE_BOOSTS")
        if not _disable_boosts and emit_items:
            top_unboosted = max(s for s, _, _, _, _ in emit_items)
            boost_ceiling = top_unboosted * 0.95  # never exceed 95% of top score
            for i, (score, block, item_type, item_id, sort_ts) in enumerate(emit_items):
                wm_multiplier = wm.boost_multiplier(item_id)
                boosted = wm.boost_score(score, item_id)
                boosted = self._access_frequency_boost(boosted, item_type, item_id)
                if boosted != score:
                    boosted = min(boosted, max(score, boost_ceiling))
                    if boosted != score and wm_multiplier > 1.0:
                        block = self._annotate_boosted_block(
                            block,
                            f"boosted: working-memory {wm_multiplier:.1f}x",
                        )
                    emit_items[i] = (boosted, block, item_type, item_id, sort_ts)

        # Sort by score descending (highest relevance first)
        emit_items.sort(key=lambda x: x[0], reverse=True)

        # Deduplication: skip chunks that are near-duplicates of already-
        # emitted items (Jaccard similarity > 0.6 on token sets). This frees
        # token budget for diverse results, especially on multi-hop queries.
        from synapt.recall.clustering import _jaccard

        emitted_token_sets: list[set[str]] = []
        knowledge_emitted = 0

        # Per-session cap: prevent one session (especially after sub-chunk
        # splitting) from flooding the output and crowding out other sessions.
        # Sub-chunks get a tighter cap (2) since they're fragments of one turn
        # and shouldn't monopolize a session's budget over full evidence turns.
        _cap_override = os.environ.get("SYNAPT_MAX_PER_SESSION")
        if _cap_override:
            _MAX_PER_SESSION = int(_cap_override)
            _MAX_SUBCHUNKS_PER_SESSION = int(os.environ.get("SYNAPT_MAX_SUBCHUNKS_PER_SESSION", str(max(1, int(_cap_override) // 2))))
        else:
            _chunk_sessions = set()
            for _, _, itype, iid, _ in emit_items:
                if itype == "chunk":
                    sid = iid.split(":")[0] if ":" in iid else ""
                    if sid:
                        _chunk_sessions.add(sid)
            _n_sessions = max(1, len(_chunk_sessions))
            _budget = sum(1 for _, _, t, _, _ in emit_items if t == "chunk")
            _MAX_PER_SESSION = max(2, _budget // _n_sessions)
            _MAX_SUBCHUNKS_PER_SESSION = max(1, _MAX_PER_SESSION // 2)
        session_emit_counts: dict[str, int] = {}
        session_subchunk_counts: dict[str, int] = {}
        selected_blocks: list[tuple[str, int, str]] = []

        emitted_access: list[dict] = []
        for score, block, item_type, item_id, sort_ts in emit_items:
            # Cap knowledge blocks to prevent them from crowding out raw chunks
            if item_type == "knowledge" and max_knowledge is not None:
                if knowledge_emitted >= max_knowledge:
                    continue
            # Per-session cap for chunks — two-tier:
            # 1. Sub-chunks capped at _MAX_SUBCHUNKS_PER_SESSION (fragments
            #    of one split turn shouldn't crowd out full evidence turns)
            # 2. Total chunks capped at _MAX_PER_SESSION
            if item_type == "chunk":
                sid = item_id.split(":")[0] if ":" in item_id else ""
                if sid:
                    is_sub = item_id in _subchunk_ids
                    if is_sub and session_subchunk_counts.get(sid, 0) >= _MAX_SUBCHUNKS_PER_SESSION:
                        continue
                    if session_emit_counts.get(sid, 0) >= _MAX_PER_SESSION:
                        continue
            block_tokens_set = set(_tokenize(block))
            dedup_threshold = getattr(
                getattr(self, "_adaptive", None), "dedup_jaccard",
                _DEDUP_JACCARD_THRESHOLD,
            )
            dedup_override = os.environ.get("SYNAPT_DEDUP_JACCARD")
            if dedup_override:
                try:
                    dedup_threshold = float(dedup_override)
                except ValueError:
                    pass
            if not os.environ.get("SYNAPT_DISABLE_DEDUP"):
                if block_tokens_set and emitted_token_sets:
                    if any(_jaccard(block_tokens_set, prev) > dedup_threshold
                           for prev in emitted_token_sets):
                        continue  # Skip near-duplicate
            block_tokens = len(block) // 4
            if token_count + block_tokens > max_tokens and selected_blocks:
                break
            selected_blocks.append((sort_ts, len(selected_blocks), block))
            token_count += block_tokens
            emitted_token_sets.append(block_tokens_set)
            # Track session counts (total + sub-chunk separately)
            if item_type == "chunk":
                sid = item_id.split(":")[0] if ":" in item_id else ""
                if sid:
                    session_emit_counts[sid] = session_emit_counts.get(sid, 0) + 1
                    if item_id in _subchunk_ids:
                        session_subchunk_counts[sid] = session_subchunk_counts.get(sid, 0) + 1
            if item_type == "knowledge":
                knowledge_emitted += 1
            access_entry: dict = {
                "item_type": item_type,
                "item_id": item_id,
                "score": score,
            }
            emitted_access.append(access_entry)

        if intent == "temporal":
            selected_blocks.sort(key=lambda x: (x[0] == "", x[0], x[1]))
        lines.extend(block for _, _, block in selected_blocks)

        # Record access for all emitted items (fire-and-forget)
        all_access = emitted_access
        for item in all_access:
            if query:
                item["query"] = query
            if self._current_session_id:
                item["session_id"] = self._current_session_id
        if all_access and self._db:
            try:
                self._db.record_access(all_access, context="search")
            except Exception:
                pass  # Never fail a search due to access tracking

            # Check promotions for emitted items (cheap actions only)
            try:
                from synapt.recall.promotion import (
                    check_promotions, execute_cheap_promotions,
                )
                for item in all_access:
                    actions = check_promotions(
                        self._db, item["item_type"], item["item_id"],
                    )
                    if actions:
                        execute_cheap_promotions(
                            self._db, item["item_type"],
                            item["item_id"], actions,
                        )
            except Exception:
                pass  # Never fail a search due to promotions

        # Populate working memory with emitted items
        # Build a lookup for knowledge node content
        knowledge_content = {
            node.get("id", ""): node.get("content", "")
            for node in (knowledge_results or [])
        }
        for item in emitted_access:
            content = ""
            if item["item_type"] == "chunk":
                idx = self._id_to_idx.get(item["item_id"])
                if idx is not None:
                    content = self._get_chunk(idx).assistant_text
            elif item["item_type"] == "knowledge":
                content = knowledge_content.get(item["item_id"], "")
            wm.record(item["item_type"], item["item_id"], content)

        conflict_notice = self._format_conflict_notice()
        if conflict_notice:
            lines.append(conflict_notice)

        if len(lines) <= 1:
            self._last_search_summary = SearchResultSummary(
                intent=intent,
                max_knowledge=max_knowledge,
                selected_blocks=0,
                chunk_blocks=0,
                knowledge_blocks=0,
                cluster_blocks=0,
            )
            return ""
        self._last_search_summary = SearchResultSummary(
            intent=intent,
            max_knowledge=max_knowledge,
            selected_blocks=len(emitted_access),
            chunk_blocks=sum(1 for item in emitted_access if item["item_type"] == "chunk"),
            knowledge_blocks=sum(1 for item in emitted_access if item["item_type"] == "knowledge"),
            cluster_blocks=sum(1 for item in emitted_access if item["item_type"] == "cluster"),
        )
        return "\n".join(lines)

    def _access_frequency_boost(
        self, score: float, item_type: str, item_id: str,
    ) -> float:
        """Apply a frequency boost based on persistent access history.

        Uses weighted_count (relevance-weighted sum of explicit accesses)
        as the signal. Each access contributes its unboosted relevance
        score instead of a flat +1, so high-relevance accesses count more
        than low-relevance ones. Falls back to explicit_count for old DBs
        that lack the weighted_count column.

        Boost: 1 + min(log2(count + 1) * 0.15, 0.3) -> capped at 1.3x.
        This is intentionally mild -- access history should nudge ranking,
        not dominate it. BM25 relevance remains the primary signal.
        """
        if not self._db:
            return score
        stats = self._db.get_access_stats(item_type, item_id)
        if stats is None or stats["explicit_count"] == 0:
            return score
        # Prefer weighted_count; fall back to explicit_count for un-migrated DBs
        count = stats.get("weighted_count", 0.0)
        if count <= 0:
            count = float(stats["explicit_count"])
        boost = 1.0 + min(math.log2(count + 1) * 0.15, 0.3)
        return score * boost

    def _group_by_cluster(
        self,
        ranked: list[tuple[int, float]],
    ) -> tuple[dict[str, tuple[dict, list[int], float]], list[tuple[int, float]]]:
        """Partition ranked results into cluster groups and ungrouped chunks.

        Returns:
            (cluster_groups, ungrouped) where cluster_groups maps
            cluster_id → (cluster_info_dict, [chunk_indices], max_score) and
            ungrouped is the list of (idx, score) not in any multi-member group.
        """
        if not self._db or self._db.cluster_count() == 0:
            return {}, ranked

        # Map each result chunk to its primary cluster and track scores.
        # Invariant: Phase 1 clustering produces non-overlapping clusters
        # (each chunk belongs to at most one). We use the first cluster_id
        # returned, with deterministic ORDER BY in clusters_for_chunk().
        cluster_members: dict[str, list[int]] = {}
        cluster_scores: dict[str, float] = {}

        for idx, score in ranked:
            chunk = self._get_chunk(idx)
            clusters = self._db.clusters_for_chunk(chunk.id)
            if clusters:
                cid = clusters[0]
                cluster_members.setdefault(cid, []).append(idx)
                cluster_scores[cid] = max(cluster_scores.get(cid, 0.0), score)

        # Only group clusters with 2+ result members (singletons stay raw)
        multi_member_cids = [cid for cid, m in cluster_members.items() if len(m) >= 2]
        cluster_groups: dict[str, tuple[dict, list[int], float]] = {}
        grouped_indices: set[int] = set()

        if multi_member_cids:
            for cid in multi_member_cids:
                cluster_info = self._db.get_cluster(cid)
                if cluster_info:
                    cluster_groups[cid] = (
                        cluster_info,
                        cluster_members[cid],
                        cluster_scores[cid],
                    )
                    grouped_indices.update(cluster_members[cid])

        ungrouped = [(idx, s) for idx, s in ranked if idx not in grouped_indices]
        return cluster_groups, ungrouped

    def _format_cluster_block(
        self,
        cluster_id: str,
        cluster_info: dict,
        query: str = "",
    ) -> str:
        """Format a cluster as a summary block.

        When *query* is provided, the summary text is snippet-extracted to show
        only the most query-relevant sentence plus surrounding context, rather
        than the full (often noisy) cluster summary.
        """
        topic = cluster_info.get("topic", "unknown")
        date_start = (cluster_info.get("date_start") or "")[:10]
        date_end = (cluster_info.get("date_end") or "")[:10]
        chunk_count = cluster_info.get("chunk_count", 0)
        session_ids = cluster_info.get("session_ids", [])

        # Date range display
        if date_start and date_end and date_start != date_end:
            date_display = f"{date_start} \u2013 {date_end}"
        elif date_start:
            date_display = date_start
        else:
            date_display = "unknown"

        header = f"--- [cluster: {topic}] {date_display}, {chunk_count} chunks ({cluster_id}) ---"
        parts = [header]

        # Try to get a stored summary
        summary = None
        if self._db:
            stored = self._db.get_cluster_summary(cluster_id)
            if stored and not stored["stale"]:
                summary = stored["summary"]

        if not summary:
            # Generate concat summary from FULL cluster membership (not just
            # search result hits) so the summary is representative of the
            # whole cluster. Read-only: don't cache during search — summaries
            # are cached during recall_build or recall_enrich.
            summary = self._generate_cluster_summary(cluster_id)

        if summary:
            # Snippet extraction: when a query is provided, pull the most
            # relevant sentence + context instead of the full summary.
            if query:
                summary = _extract_snippet(summary, query, context_lines=1)
            parts.append(summary)

        # Session references
        if session_ids:
            sess_display = ", ".join(s[:8] for s in session_ids[:4])
            if len(session_ids) > 4:
                sess_display += f" +{len(session_ids) - 4} more"
            parts.append(f"[Sessions: {sess_display}]")

        return "\n".join(parts)

    def _generate_cluster_summary(self, cluster_id: str) -> str:
        """Generate a concat summary from all chunks in a cluster."""
        if not self._db:
            return ""
        chunk_ids = self._db.get_cluster_chunks(cluster_id)
        if not chunk_ids:
            return ""
        # Look up chunks by ID
        member_chunks = []
        for cid in chunk_ids:
            idx = self._id_to_idx.get(cid)
            if idx is not None:
                member_chunks.append(self._get_chunk(idx))
        if not member_chunks:
            return ""
        from synapt.recall.clustering import generate_concat_summary
        return generate_concat_summary(member_chunks, max_tokens=200)

    @staticmethod
    def _find_query_span(
        query: str, text: str, margin: int = 40,
    ) -> tuple[int, int] | None:
        """Find the best-matching sentence span in *text* for *query*.

        Splits *text* into sentences, scores each by token overlap with
        *query*, then returns ``(begin, end)`` covering the best contiguous
        window of 1-3 sentences (plus *margin* chars of context).

        Returns ``None`` if no sentence has ≥2 overlapping tokens.
        """
        if not query or not text:
            return None

        sentence_spans: list[tuple[int, int, str]] = []
        for m in re.finditer(r'[^.!?\n]*[.!?\n]+(?:\s|$)|[^.!?\n]+$', text):
            sentence_spans.append((m.start(), m.end(), m.group()))
        if not sentence_spans:
            return None

        query_toks = set(_tokenize(query))
        if not query_toks:
            return None

        scored: list[tuple[int, int, int]] = []
        for begin, end, sent in sentence_spans:
            sent_toks = set(_tokenize(sent))
            overlap = len(query_toks & sent_toks)
            scored.append((begin, end, overlap))

        best_score = 0
        best_begin = 0
        best_end = 0
        for window in range(1, min(4, len(scored) + 1)):
            for i in range(len(scored) - window + 1):
                total = sum(scored[j][2] for j in range(i, i + window))
                if total > best_score:
                    best_score = total
                    best_begin = scored[i][0]
                    best_end = scored[i + window - 1][1]

        if best_score < 2:
            return None

        begin = max(0, best_begin - margin)
        end = min(len(text), best_end + margin)
        return (begin, end)

    def _format_chunk_block(self, idx: int, intent: str = "", query: str = "") -> str:
        """Format a single chunk as a display block.

        When *query* is provided, attempts to extract the best-matching
        sentence span and emits a focused snippet instead of the full turn.
        Falls back to the full turn if no good span is found.
        """
        chunk = self._get_chunk(idx)
        ts_display = _format_timestamp_display(chunk.timestamp, intent=intent)
        turn_label = "journal" if chunk.turn_index == -1 else f"turn {chunk.turn_index}"
        # Freshness label for chunks — same style as knowledge nodes
        freshness = ""
        if chunk.timestamp and len(chunk.timestamp) >= 10:
            try:
                chunk_dt = _parse_chunk_timestamp(chunk.timestamp)
                if chunk_dt is None:
                    raise ValueError("unparseable timestamp")
                days_ago = (datetime.now(timezone.utc) - chunk_dt).days
                if days_ago == 0:
                    freshness = ", today"
                elif days_ago == 1:
                    freshness = ", yesterday"
                elif days_ago < 7:
                    freshness = f", {days_ago}d ago"
                elif days_ago < 30:
                    freshness = f", {days_ago // 7}w ago"
            except (ValueError, TypeError):
                pass
        header = f"--- [{ts_display} session {_short_sid(chunk.session_id)}] {turn_label}{freshness} ---"

        # Query-aware snippet extraction (#340).  Configurable via
        # SYNAPT_SNIPPET_CONTEXT (#344):
        #   -1 or unset → no snippeting (default — full chunk)
        #    0           → tight snippets (40-char margin, original #340)
        #    N           → N chars of context margin on each side
        snippet_ctx = os.environ.get("SYNAPT_SNIPPET_CONTEXT")
        if snippet_ctx is None and os.environ.get("SYNAPT_ENABLE_SNIPPETS"):
            snippet_ctx = "0"  # backward compat for legacy env var
        if snippet_ctx is None:
            snippet_ctx = "-1"
        try:
            snippet_margin = int(snippet_ctx)
        except ValueError:
            snippet_margin = -1
        if query and snippet_margin >= 0:
            margin = snippet_margin if snippet_margin > 0 else 40
            asst = chunk.assistant_text or ""
            user = chunk.user_text or ""
            # Try assistant text first — that's the evidence source
            if len(asst) > 200:
                span = self._find_query_span(query, asst, margin=margin)
                if span:
                    begin, end = span
                    snippet = asst[begin:end].strip()
                    prefix = "..." if begin > 0 else ""
                    suffix = "..." if end < len(asst) else ""
                    return f"{header}\nAssistant: {prefix}{snippet}{suffix}"
            # Fall back to user text only for journal entries or user-heavy turns
            if len(user) > 300 and chunk.turn_index < 0:
                span = self._find_query_span(query, user, margin=margin)
                if span:
                    begin, end = span
                    snippet = user[begin:end].strip()
                    prefix = "..." if begin > 0 else ""
                    suffix = "..." if end < len(user) else ""
                    return f"{header}\n{prefix}{snippet}{suffix}"

        parts = [header]
        # Sub-chunks already embed context in their user_text — skip
        # the preceding turn context to save tokens.
        is_subchunk = chunk.user_text.startswith("(context:")
        if not is_subchunk:
            prev = self._get_preceding_turn(chunk)
            if prev and prev.user_text:
                ctx = prev.user_text[:200]
                if len(prev.user_text) > 200:
                    ctx += "..."
                parts.append(f"  (context: User previously asked: {ctx})")
        if chunk.user_text:
            ut = chunk.user_text[:500]
            if len(chunk.user_text) > 500:
                ut += "..."
            parts.append(f"User: {ut}")
        if chunk.assistant_text:
            asst = chunk.assistant_text[:1500]
            if len(chunk.assistant_text) > 1500:
                asst += "..."
            parts.append(f"Assistant: {asst}")
        if chunk.files_touched and not is_subchunk:
            parts.append(f"[Files: {', '.join(chunk.files_touched[:5])}]")
        # Skip tool_content for sub-chunks — assistant_text already
        # summarizes the action, and tool details are noise.
        if chunk.tool_content and not is_subchunk:
            tc = chunk.tool_content[:400]
            if len(chunk.tool_content) > 400:
                tc += "..."
            parts.append(f"[Tools: {tc}]")

        return "\n".join(parts)

    # -------------------------------------------------------------------
    # Drill-down: full raw context for a turn
    # -------------------------------------------------------------------

    def read_turn_context(self, chunk_id: str) -> str:
        """Read full raw transcript content for a turn via byte offsets.

        Seeks to the stored byte_offset in the source transcript file and
        reads byte_length bytes, then extracts and formats all user messages,
        assistant text, tool uses with full inputs, and tool results.

        Returns formatted text with no cap — this is the detail view.
        """
        # Find chunk by ID (O(1) via index map)
        idx = self._id_to_idx.get(chunk_id)
        chunk = self._get_chunk(idx) if idx is not None else None
        if chunk is None and self._db:
            # Try DB lookup
            row = self._db._conn.execute(
                "SELECT transcript_path, byte_offset, byte_length, "
                "session_id, turn_index, timestamp "
                "FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if row:
                chunk = TranscriptChunk(
                    id=chunk_id,
                    session_id=row[3],
                    timestamp=row[5] or "",
                    turn_index=row[4],
                    user_text="",
                    assistant_text="",
                    transcript_path=row[0] or "",
                    byte_offset=row[1] if row[1] is not None else -1,
                    byte_length=row[2] if row[2] is not None else 0,
                )
        if chunk is None:
            return f"Chunk {chunk_id} not found."
        if not chunk.transcript_path or chunk.byte_offset < 0:
            return f"No transcript offset recorded for {chunk_id}."
        try:
            return self._read_raw_turn(chunk)
        except Exception as exc:
            return f"Error reading transcript: {exc}"

    @staticmethod
    def _read_raw_turn(chunk: TranscriptChunk) -> str:
        """Read and format raw JSONL entries for a turn."""
        path = Path(chunk.transcript_path)
        if not path.exists():
            return f"Transcript file not found: {path}"

        with open(path, "rb") as f:
            f.seek(chunk.byte_offset)
            raw = f.read(chunk.byte_length).decode("utf-8", errors="replace")

        parts = []
        ts = chunk.timestamp[:19] if chunk.timestamp else "unknown"
        parts.append(
            f"=== Full context: session {_short_sid(chunk.session_id)} "
            f"turn {chunk.turn_index} [{ts}] ==="
        )

        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = entry.get("type", "")
            if etype in SKIP_TYPES:
                continue
            queued_prompt = _queued_human_prompt(entry)
            if queued_prompt is not None:
                parts.append(f"\n[User]\n{queued_prompt}")
                continue
            content = entry.get("message", {}).get("content")
            if etype == "user" and isinstance(content, str) and content.strip():
                parts.append(f"\n[User]\n{content.strip()}")
            elif etype == "user" and isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        parts.append(f"\n[User]\n{block.get('text', '')}")
                    elif block.get("type") == "tool_result":
                        tu_id = block.get("tool_use_id", "?")
                        result = block.get("content", "")
                        if isinstance(result, list):
                            result = "\n".join(
                                s.get("text", "")
                                for s in result
                                if isinstance(s, dict) and s.get("type") == "text"
                            )
                        parts.append(f"\n[Tool Result for {tu_id}]\n{result}")
            elif etype == "assistant" and isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        parts.append(f"\n[Assistant]\n{block.get('text', '')}")
                    elif block.get("type") == "tool_use":
                        name = block.get("name", "?")
                        inp = json.dumps(block.get("input", {}), indent=2)
                        parts.append(f"\n[Tool Use: {name}]\n{inp}")

        return "\n".join(parts)

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        """Save index to disk for fast reload."""
        directory.mkdir(parents=True, exist_ok=True)

        # Build manifest data (shared by both paths)
        sessions_info = {}
        for sid, session_chunks in self.sessions.items():
            timestamps = [c.timestamp for c in session_chunks if c.timestamp]
            sessions_info[sid] = {
                "chunk_count": len(session_chunks),
                "min_ts": min(timestamps) if timestamps else "",
                "max_ts": max(timestamps) if timestamps else "",
            }

        manifest = {
            "chunk_count": len(self.chunks),
            "session_count": len(self.sessions),
            "sessions": sessions_info,
            "build_timestamp": datetime.now().isoformat(),
        }

        if self._db is not None:
            # SQLite path
            self._db.save_chunks(self._materialize_all_chunks())
            self._db.save_manifest(manifest)
            self._refresh_rowid_map()
            return

        # Legacy file-based path
        chunks_path = directory / "chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in self._materialize_all_chunks():
                f.write(json.dumps(chunk.to_dict()) + "\n")

        h = hashlib.sha256()
        with open(chunks_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)

        manifest["chunks_hash"] = h.hexdigest()[:16]
        atomic_json_write(manifest, directory / "manifest.json")

    @classmethod
    def load(cls, directory: Path, use_embeddings: bool = False) -> TranscriptIndex:
        """Load a saved index from disk.

        Prefers recall.db (SQLite) if present.  Falls back to chunks.jsonl
        with automatic migration to SQLite.  Malformed JSON lines are
        skipped during migration.
        """
        db_path = directory / "recall.db"
        chunks_path = directory / "chunks.jsonl"

        # Prefer ShardedRecallDB (auto-detects monolithic recall.db or sharded layout)
        from synapt.recall.sharding import is_sharded
        if db_path.exists() or is_sharded(directory):
            db = None
            try:
                db = ShardedRecallDB.open(directory)
                chunks = db.load_chunk_headers()
                return cls(
                    chunks,
                    use_embeddings=use_embeddings,
                    cache_dir=directory,
                    db=db,
                    lazy_chunks=True,
                )
            except (sqlite3.DatabaseError, OSError) as exc:
                logger.warning("Corrupt recall DB, rebuilding: %s", exc)
                if db is not None:
                    with contextlib.suppress(Exception):
                        db.close()
                if not is_sharded(directory):
                    with contextlib.suppress(OSError):
                        db_path.unlink()
                # Fall through to JSONL path or empty index

        # Legacy JSONL — load and auto-migrate to SQLite
        if chunks_path.exists():
            chunks = []
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunks.append(TranscriptChunk.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue  # Skip malformed lines

            # Migrate to SQLite
            db = RecallDB(db_path)
            index = cls(chunks, use_embeddings=use_embeddings, cache_dir=directory, db=db)
            index.save(directory)

            # Migrate embeddings from legacy JSON if present
            emb_path = directory / "transcript_embeddings.json"
            if emb_path.exists() and index._idx_to_rowid:
                try:
                    with open(emb_path, encoding="utf-8") as f:
                        cached = json.load(f)
                    old_embs = cached.get("embeddings", [])
                    old_hash = cached.get("hash")
                    if old_embs and old_hash == index._content_hash():
                        emb_mapping: dict[int, list[float]] = {}
                        for i, emb in enumerate(old_embs):
                            rowid = index._idx_to_rowid.get(i)
                            if rowid is not None:
                                emb_mapping[rowid] = emb
                        if emb_mapping:
                            db.save_embeddings(emb_mapping)
                            db.set_metadata("embedding_hash", index._content_hash())
                except (json.JSONDecodeError, KeyError, OSError):
                    pass

            # Clean up legacy files
            for old_file in [chunks_path, emb_path, directory / "manifest.json"]:
                with contextlib.suppress(OSError):
                    old_file.unlink()

            return index

        return cls([])

    # -------------------------------------------------------------------
    # File lookup
    # -------------------------------------------------------------------

    def lookup_files(
        self,
        pattern: str,
        max_chunks: int = 10,
        max_tokens: int = 500,
        after: str | None = None,
        before: str | None = None,
    ) -> str:
        """Find chunks that touched files matching the given pattern.

        Supports partial path matching: "repair.py" matches
        "src/graph/repair.py". Also matches basename-only.

        Results sorted newest-first (no BM25 scoring).
        """
        if not self.chunks or not pattern:
            return ""

        date_filter = self._filter_by_date(after, before)
        pattern_lower = pattern.lower()

        hits: list[tuple[int, float]] = []
        for i in range(len(self.chunks)):
            chunk = self._get_chunk(i)
            if date_filter is not None and i not in date_filter:
                continue
            if not chunk.files_touched:
                continue
            for fp in chunk.files_touched:
                fp_lower = fp.lower()
                if pattern_lower in fp_lower or fp_lower.endswith("/" + pattern_lower):
                    hits.append((i, 1.0))
                    break

        if not hits:
            return ""

        # Sort newest-first by timestamp
        hits.sort(key=lambda x: self._get_chunk(x[0]).timestamp, reverse=True)
        return self._format_results(hits[:max_chunks], max_tokens, query=pattern)

    # -------------------------------------------------------------------
    # Session browsing
    # -------------------------------------------------------------------

    def list_sessions(
        self,
        max_sessions: int = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> list[dict]:
        """Return recent sessions with summary info, newest-first."""
        candidates = []
        for session_id in self._session_order:
            chunks = self.sessions[session_id]

            timestamps = [c.timestamp for c in chunks if c.timestamp]
            if not timestamps:
                continue

            latest_ts = max(timestamps)
            earliest_ts = min(timestamps)

            if after and latest_ts < after:
                continue
            if before and earliest_ts >= before:
                continue

            transcript_chunks = [c for c in chunks if c.turn_index >= 0]
            candidates.append((
                session_id,
                chunks,
                transcript_chunks,
                earliest_ts,
            ))
            if len(candidates) >= max_sessions:
                break

        hydrated_by_id: dict[str, TranscriptChunk] = {}
        if self._lazy_chunks and self._db is not None:
            rowids = []
            for _, chunks, _, _ in candidates:
                for chunk in chunks:
                    idx = self._id_to_idx.get(chunk.id)
                    rowid = self._idx_to_rowid.get(idx) if idx is not None else None
                    if rowid is not None:
                        rowids.append(rowid)
            hydrated_by_id = {
                chunk.id: chunk
                for chunk in self._db.load_chunks_by_rowids(rowids).values()
            }
        else:
            hydrated_by_id = {
                chunk.id: chunk
                for _, chunks, _, _ in candidates
                for chunk in chunks
            }

        results = []
        for session_id, chunks, transcript_chunks, earliest_ts in candidates:
            # First user message as summary (skip journal chunks)
            sorted_chunks = sorted(transcript_chunks or chunks, key=lambda c: c.turn_index)
            first_msg = ""
            for c in sorted_chunks:
                full = hydrated_by_id.get(c.id, c)
                if full.user_text:
                    first_msg = full.user_text[:120]
                    if len(full.user_text) > 120:
                        first_msg += "..."
                    break

            all_files = set()
            for c in chunks:
                full = hydrated_by_id.get(c.id, c)
                all_files.update(full.files_touched)

            results.append({
                "session_id": session_id,
                "date": earliest_ts[:10],
                "turn_count": len(transcript_chunks),
                "first_message": first_msg,
                "files_count": len(all_files),
            })

        return results

    def session_tail(self, session_id: str) -> list[TranscriptChunk]:
        """Return one session's conversation turns, hydrated, oldest-first.

        Hydration is the point of this method. :meth:`load` returns *headers
        only* (``lazy_chunks=True``), so reading ``user_text`` off
        ``self.sessions[sid]`` yields empty strings rather than raising — a
        caller that skips hydration reports an empty session instead of failing.

        Chunks with a negative ``turn_index`` are excluded: that value is the
        sentinel for synthesized journal content, not conversation.
        """
        turns = [
            self._get_chunk(self._id_to_idx[header.id])
            for header in self.sessions.get(session_id, [])
            if header.turn_index >= 0
        ]
        turns.sort(key=lambda c: c.turn_index)
        return turns

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics about the index."""
        if not self.chunks:
            return {"chunk_count": 0, "session_count": 0}

        timestamps = [c.timestamp for c in self.chunks if c.timestamp]
        return {
            "chunk_count": len(self.chunks),
            "session_count": len(self.sessions),
            "date_range": {
                "earliest": min(timestamps) if timestamps else "",
                "latest": max(timestamps) if timestamps else "",
            },
            "avg_chunks_per_session": len(self.chunks) / max(len(self.sessions), 1),
            "total_tools_used": len(set(
                t for c in self._materialize_all_chunks() for t in c.tools_used
            )),
            "total_files_touched": len(set(
                f for c in self._materialize_all_chunks() for f in c.files_touched
            )),
            "embeddings_active": (
                self._embeddings is not None
                or (self._db is not None and self._db.has_embeddings())
            ),
            "embedding_provider": (
                type(self._embed_provider).__name__
                if self._embed_provider else None
            ),
            "storage_backend": "sqlite" if self._db else "memory",
            "knowledge_count": (
                self._db.knowledge_count() if self._db else 0
            ),
        }


# ---------------------------------------------------------------------------
# Project path utilities
# ---------------------------------------------------------------------------


def _git_main_worktree_root(path: Path) -> Path | None:
    """If *path* is inside a git worktree, return the main worktree's root.

    Uses ``git rev-parse --git-common-dir`` which returns the shared ``.git/``
    directory for all worktrees.  The main worktree root is the parent of that
    directory.  Returns *None* if *path* is not in a git repo or is already the
    main worktree (i.e. ``.git`` is a directory, not a gitdir pointer file).
    """
    git_path = path / ".git"
    # Fast path: if .git is a real directory, this is the main worktree already
    if git_path.is_dir():
        return None
    # If .git is a file, this is a linked worktree
    if not git_path.is_file():
        return None
    try:
        common_dir = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, cwd=path, timeout=5,
        )
        if common_dir.returncode != 0:
            return None
        # --git-common-dir returns e.g. /Users/me/Development/rd/.git
        # The main worktree root is its parent
        return Path(common_dir.stdout.strip()).parent
    except (OSError, subprocess.TimeoutExpired):
        return None


_gripspace_cache: dict[str, tuple[Path | None, float]] = {}
_GRIPSPACE_CACHE_TTL = 60.0  # seconds


def _find_gripspace_root(path: Path) -> Path | None:
    """Walk up from *path* to find the gr1 or gr2 workspace root.

    Returns the gripspace root path, or *None* if not inside a gripspace.
    Analogous to ``_git_main_worktree_root`` but for multi-repo gripspaces.

    A gr2 workspace is marked by ``.grip/``. For gr1, distinguishes between a
    **gripspace root** (has ``.gitgrip/griptrees.json``) and a **linked
    griptree** (has ``.gitgrip/griptree.json`` — singular).
    When a linked griptree is found, resolves back to the gripspace root via
    the git worktree pointer in any sub-repo.

    Stops at ``$HOME`` to avoid matching stray ``.gitgrip/`` directories
    above the user's project hierarchy.  Results are cached per resolved path
    with a 60-second TTL so long-running processes (MCP server) pick up
    gripspace changes without restart.
    """
    import time

    current = path.resolve()
    cache_key = str(current)
    cached = _gripspace_cache.get(cache_key)
    if cached is not None:
        value, ts = cached
        if time.monotonic() - ts < _GRIPSPACE_CACHE_TTL:
            return value

    home = Path.home().resolve()
    while current != current.parent:
        gitgrip = current / ".gitgrip"
        if gitgrip.is_dir():
            # Linked griptree has griptree.json (singular) — always resolve
            # to parent gripspace, even if griptrees.json also exists (which
            # happens when `gr` clones the full directory structure).
            if (gitgrip / "griptree.json").exists():
                root = _resolve_griptree_parent(current)
                if root is not None:
                    _gripspace_cache[cache_key] = (root, time.monotonic())
                    return root
                # Membership is ASSERTED here but could not be VERIFIED. Use
                # the strongest signal that actually was verified: ".grip" is
                # positive evidence this directory is a workspace, so fall
                # through to it rather than reporting nothing.
                #
                # Without this, reordering the two markers would make things
                # WORSE for exactly this population: such a directory used to
                # short-circuit on ".grip" and resolve to itself, and would
                # instead resolve to nothing — fragmenting a store that
                # previously cohered, which is the failure this function is
                # being changed to prevent.
                if (current / ".grip").is_dir():
                    _gripspace_cache[cache_key] = (current, time.monotonic())
                    return current
                # No membership we can verify and no locality evidence either.
                # Return None EXACTLY as before, and deliberately do NOT keep
                # walking upward. Continuing the walk might well find a better
                # answer, but it would be a NEW behaviour for a case this
                # change is not about, and it would break the property that
                # makes this change checkable: every directory either improves
                # or is unchanged, and nothing else moves.
                _gripspace_cache[cache_key] = (None, time.monotonic())
                return None
            # Gripspace root has only griptrees.json (plural), no singular
            if (gitgrip / "griptrees.json").exists():
                _gripspace_cache[cache_key] = (current, time.monotonic())
                return current

        # MEMBERSHIP BEATS LOCALITY. gr2 owns one .grip/ namespace at the
        # workspace root, and spawned units live beneath it, so recognizing
        # that marker makes their shared index and knowledge land at the
        # containing workspace instead of at each unit cwd. That is still
        # true, and it is why this check exists.
        #
        # It is consulted AFTER the gr1 membership markers above, because the
        # two answer different questions: ".grip" asks "am I a workspace
        # containing units?", while .gitgrip/griptree.json asks "whose larger
        # whole am I a part of?". A directory can hold both, and store
        # resolution needs the second — a member's data belongs to the
        # workspace it is a member of. Checked first, ".grip" would shadow the
        # membership claim entirely: the directory resolves to itself, its
        # data lands where no other surface in that workspace looks, and
        # nothing anywhere reports a problem.
        if (current / ".grip").is_dir():
            _gripspace_cache[cache_key] = (current, time.monotonic())
            return current
        # Don't walk above $HOME
        if current == home:
            break
        current = current.parent
    _gripspace_cache[cache_key] = (None, time.monotonic())
    return None


def _resolve_griptree_parent(griptree_path: Path) -> Path | None:
    """Resolve a linked griptree back to its parent gripspace root.

    Uses the git worktree pointer in any sub-repo to find the main worktree,
    then walks up from there to find the gripspace root.
    """
    # Find any sub-repo with a .git file (linked worktree pointer)
    try:
        children = list(griptree_path.iterdir())
    except OSError:
        return None
    for child in children:
        git_path = child / ".git"
        if git_path.is_file():
            # .git file contains: "gitdir: /path/to/main/.git/worktrees/<name>/"
            try:
                content = git_path.read_text(encoding="utf-8").strip()
                if content.startswith("gitdir:"):
                    gitdir = Path(content.split(":", 1)[1].strip())
                    # gitdir is <main-repo>/.git/worktrees/<name>/
                    # Walk up to find .git/, then parent is the main repo root
                    main_repo = gitdir
                    while main_repo.name != ".git" and main_repo != main_repo.parent:
                        main_repo = main_repo.parent
                    if main_repo.name == ".git":
                        main_repo_root = main_repo.parent
                        # Walk up from main repo to find gripspace root
                        candidate = main_repo_root
                        home = Path.home().resolve()
                        while candidate != candidate.parent:
                            if (candidate / ".gitgrip" / "griptrees.json").exists():
                                return candidate
                            if candidate == home:
                                break
                            candidate = candidate.parent
            except OSError:
                continue
    return None


def project_slug(project_dir: Path | None = None) -> str:
    """Convert a project path to Claude Code's directory slug format.

    Claude Code stores per-project data at ~/.claude/projects/<slug>/
    where the slug is the absolute path with ``/`` replaced by ``-``.

    Examples:
        /Users/me/Development/synapse → -Users-me-Development-synapse
    """
    p = (project_dir or Path.cwd()).resolve()
    # Use PurePosixPath to normalise Windows backslashes before slugifying.
    # On Unix this is a no-op; on Windows it converts C:\Users\... → C:/Users/...
    return str(p).replace("\\", "/").replace("/", "-")


def _worktree_name(project_dir: Path | None = None) -> str:
    """Return the stable worktree or workspace name for per-worktree data.

    For the main worktree at ``/Users/me/Development/rd``, returns ``rd``.
    For a linked worktree at ``/Users/me/Development/poe``, returns ``poe``.

    ``SYNAPT_RECALL_WORKTREE`` overrides the basename when set and no explicit
    *project_dir* was passed. It exists as the companion to
    ``SYNAPT_RECALL_ROOT``: redirecting only the store root would file this
    caller's per-worktree data under its cwd basename inside the shared store —
    potentially another workspace's namespace, which is cross-attribution
    rather than sharing.
    """
    if project_dir is None:
        env_name = os.environ.get("SYNAPT_RECALL_WORKTREE")
        if env_name:
            if "/" in env_name or "\\" in env_name or env_name in (".", ".."):
                raise ValueError(
                    f"SYNAPT_RECALL_WORKTREE must be a bare namespace label, "
                    f"got {env_name!r}. A path component here would relocate "
                    f"per-worktree files outside worktrees/."
                )
            return env_name

    current = (project_dir or Path.cwd()).resolve()
    grip_root = _find_gripspace_root(current)
    candidate = current
    while candidate != candidate.parent:
        # A .git marker identifies the root of either a main checkout or a
        # linked worktree. It is a directory in the former and a file in the
        # latter. Keep walking only until the nearest such root so a
        # constituent repository retains its own bucket inside a shared
        # gripspace store.
        if (candidate / ".git").exists():
            return candidate.name
        if (candidate / ".gitgrip" / "griptree.json").exists():
            return candidate.name
        if candidate == grip_root:
            break
        candidate = candidate.parent

    # A non-git directory inside a gripspace still belongs to the workspace.
    # This mirrors data-root resolution and prevents a cwd subdirectory from
    # becoming a second, silently partial journal bucket.
    if grip_root is not None:
        return grip_root.name
    return current.name


# ---------------------------------------------------------------------------
# Data-root policy seam
# ---------------------------------------------------------------------------
# A consumer may install a callable consulted with (operation, path) when a
# recall data root is resolved, before any directory is created or migrated.
# With no policy installed this is a no-op and behaviour is unchanged.

_data_root_policy = None


def set_data_root_policy(policy):
    """Install a recall data-root policy; returns the previous one."""
    global _data_root_policy
    previous = _data_root_policy
    _data_root_policy = policy
    return previous


def _guard_data_root(operation: str, path: Path) -> Path:
    """Consult the installed policy, then return *path* unchanged."""
    policy = _data_root_policy
    if policy is not None:
        policy(operation, path)
    return path


def project_data_dir(project_dir: Path | None = None) -> Path:
    """Return the root synapt recall data directory.

    ALL recall data lives under ``<root>/.synapt/recall/``:
    shared data (index, knowledge) at the root, and per-worktree data
    (transcripts, journal) under ``worktrees/<name>/``.

    Root resolution priority:
      0. ``SYNAPT_RECALL_ROOT`` environment variable — an explicit workspace
         root for multi-workspace setups, consulted only when *project_dir*
         is not passed. The directory must already exist: silently minting a
         fresh store under a mistyped root would present as an empty history
         that looks exactly like a real answer. Inference below cannot replace
         this: two workspaces that share a store may be filesystem SIBLINGS,
         and walking up from inside one can never arrive at the other.
      1. Git worktree → main worktree root
      2. GitGrip gripspace → gripspace root (all constituent repos share
         one index; each sub-repo gets its own ``worktrees/<name>/`` subdir)
      3. CWD as fallback

    Auto-migrates from two legacy locations (under the explicit override
    too — an overridden root may be a pre-rename workspace):
      1. ``.synapse/recall/``  → ``.synapt/recall/``
      2. ``.synapse-recall/``  → ``.synapt/recall/``
    """
    env_resolved: Path | None = None
    if project_dir is None:
        env_root = os.environ.get("SYNAPT_RECALL_ROOT")
        if env_root:
            env_resolved = Path(env_root).expanduser().resolve()
            if not env_resolved.is_dir():
                raise ValueError(
                    f"SYNAPT_RECALL_ROOT points at a directory that does not "
                    f"exist: {env_resolved}. Refusing to mint a fresh store "
                    f"under a mistyped root — unset the variable or create "
                    f"the workspace first."
                )

    if env_resolved is not None:
        # No early return: the override selects the ROOT and then flows
        # through the same legacy-migration tail as an inferred root. An
        # override pointed at a pre-rename workspace would otherwise resolve
        # to a .synapt path that does not exist while the real history sits
        # in .synapse — an empty store presenting as an empty history.
        root = env_resolved
    else:
        root = (project_dir or Path.cwd()).resolve()

        # Priority 1: git worktree → resolve to main worktree root
        main_root = _git_main_worktree_root(root)
        if main_root is not None:
            root = main_root

        # Priority 2: GitGrip gripspace → resolve to gripspace root
        # If CWD (or resolved root) is inside a gripspace, prefer the
        # gripspace root so all constituent repos share one recall index.
        grip_root = _find_gripspace_root(root)
        if grip_root is not None:
            root = grip_root

    new_dir = root / ".synapt" / "recall"

    # Guarded before the legacy-migration tail below, which renames real
    # directories on disk. A consumer may install a policy that refuses a
    # resolved data root; with none installed this is a no-op.
    _guard_data_root("project_data_dir", new_dir)

    if not new_dir.exists():
        # Tier 1: .synapse/recall/ (intermediate rename era)
        mid_dir = root / ".synapse" / "recall"
        if mid_dir.is_dir():
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            mid_dir.rename(new_dir)
            # Clean up empty .synapse/ parent if nothing else is inside
            synapse_parent = root / ".synapse"
            if synapse_parent.is_dir() and not any(synapse_parent.iterdir()):
                synapse_parent.rmdir()
        else:
            # Tier 2: .synapse-recall/ (legacy flat location)
            old_dir = root / ".synapse-recall"
            if old_dir.is_dir():
                new_dir.parent.mkdir(parents=True, exist_ok=True)
                old_dir.rename(new_dir)

    return new_dir


def project_worktree_dir(project_dir: Path | None = None) -> Path:
    """Return the per-worktree data directory inside the shared ``.synapt/``.

    Lives at ``<main>/.synapt/recall/worktrees/<worktree-name>/``.
    Each worktree gets its own subdirectory for transcripts and journal.

    **Auto-migrates** from the legacy flat layout on first access:
    moves ``transcripts/`` and ``journal.jsonl`` from the data root into
    ``worktrees/<main-name>/`` for the main worktree.
    """
    wt_name = _worktree_name(project_dir)
    data_dir = project_data_dir(project_dir)
    wt_dir = data_dir / "worktrees" / wt_name

    # Auto-migrate legacy flat layout → worktrees/<name>/
    # Only applies to the main worktree (linked worktrees never had flat data).
    # Race-safe: if two processes migrate concurrently, the loser's rename()
    # raises FileNotFoundError (source already moved) — we catch and continue.
    if not wt_dir.exists():
        legacy_transcripts = data_dir / "transcripts"
        legacy_journal = data_dir / "journal.jsonl"
        has_legacy = legacy_transcripts.is_dir() or legacy_journal.exists()

        if has_legacy:
            wt_dir.mkdir(parents=True, exist_ok=True)
            try:
                if legacy_transcripts.is_dir() and not legacy_transcripts.is_symlink():
                    legacy_transcripts.rename(wt_dir / "transcripts")
                    legacy_transcripts.symlink_to(wt_dir / "transcripts")
            except (FileNotFoundError, FileExistsError, OSError):
                pass  # Another process already migrated or symlinked
            try:
                if legacy_journal.exists() and not legacy_journal.is_symlink():
                    legacy_journal.rename(wt_dir / "journal.jsonl")
            except (FileNotFoundError, FileExistsError, OSError):
                pass  # Another process already migrated

    return wt_dir


def project_index_dir(project_dir: Path | None = None) -> Path:
    """Return the shared index directory.

    Index, clusters, and knowledge are shared across all worktrees.
    Lives at ``<main>/.synapt/recall/index/``.
    """
    return project_data_dir(project_dir) / "index"


def project_archive_dir(project_dir: Path | None = None) -> Path:
    """Return the per-worktree transcript archive directory.

    Lives at ``<main>/.synapt/recall/worktrees/<name>/transcripts/``.
    """
    return project_worktree_dir(project_dir) / "transcripts"


def project_transcript_dir(project_dir: Path | None = None) -> Path | None:
    """Return the Claude Code transcript directory for a project.

    Returns None if the directory doesn't exist or has no transcripts.
    """
    slug = project_slug(project_dir)
    d = Path.home() / ".claude" / "projects" / slug
    if d.is_dir() and any(d.glob("*.jsonl")):
        return d
    return None


def project_transcript_dirs(project_dir: Path | None = None) -> list[Path]:
    """Return all Claude Code transcript directories for a project.

    When working in a git worktree, Claude Code creates transcripts under
    a slug derived from the *worktree* path.  This function returns transcript
    directories for both the main worktree and the current worktree (if
    different), so that ``recall build`` archives sessions from all worktrees.

    In a gripspace, also discovers transcripts from:
    - direct child repos (directories with ``.git``)
    - linked worktrees under ``.worktrees/*`` (also directories with ``.git``)
    - declared griptrees from ``.gitgrip/griptrees.json`` even when they live
      outside the gripspace root directory
    """
    actual_dir = (project_dir or Path.cwd()).resolve()
    dirs: list[Path] = []
    seen_slugs: set[str] = set()

    def _maybe_add_transcript_dir(path: Path) -> None:
        slug = project_slug(path)
        if slug in seen_slugs:
            return
        seen_slugs.add(slug)
        transcript_dir = Path.home() / ".claude" / "projects" / slug
        if transcript_dir.is_dir() and any(transcript_dir.glob("*.jsonl")):
            dirs.append(transcript_dir)

    # Always include the actual CWD's transcript dir
    _maybe_add_transcript_dir(actual_dir)

    # If in a linked worktree, also include the main worktree's transcript dir
    main_root = _git_main_worktree_root(actual_dir)
    if main_root is not None:
        _maybe_add_transcript_dir(main_root)

    # If in a gripspace, discover transcripts from all constituent repos
    # plus linked worktrees under .worktrees/* and declared griptrees from
    # .gitgrip/griptrees.json. Active agent sessions often run from sibling
    # griptrees like synapt-global rather than from direct child repos.
    grip_root = _find_gripspace_root(actual_dir)
    if grip_root is not None:
        griptrees_json = grip_root / ".gitgrip" / "griptrees.json"
        if griptrees_json.is_file():
            try:
                data = json.loads(griptrees_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            for meta in data.get("griptrees", {}).values():
                path_str = meta.get("path")
                if isinstance(path_str, str) and path_str:
                    _maybe_add_transcript_dir(Path(path_str))
        try:
            children = sorted(grip_root.iterdir())
        except OSError:
            children = []
        for child in children:
            if child.is_dir() and (child / ".git").exists():
                _maybe_add_transcript_dir(child)
            elif child.is_dir() and child.name == ".worktrees":
                try:
                    worktrees = sorted(child.iterdir())
                except OSError:
                    worktrees = []
                for wt in worktrees:
                    if wt.is_dir() and (wt / ".git").exists():
                        _maybe_add_transcript_dir(wt)

    return dirs


def all_worktree_archive_dirs(project_dir: Path | None = None) -> list[Path]:
    """Return archive directories from all worktrees.

    Since all worktree data lives under ``<main>/.synapt/recall/worktrees/``,
    this simply globs for ``worktrees/*/transcripts/`` directories that
    contain archived transcripts.  No ``git worktree list`` needed.
    """
    data_dir = project_data_dir(project_dir)
    worktrees_root = data_dir / "worktrees"
    if not worktrees_root.is_dir():
        return []

    dirs: list[Path] = []
    for wt_dir in sorted(worktrees_root.iterdir()):
        archive = wt_dir / "transcripts"
        if archive.is_dir() and any(archive.glob("*.jsonl")):
            dirs.append(archive)
    return dirs


# ---------------------------------------------------------------------------
# Journal → chunk converter
# ---------------------------------------------------------------------------


def _journal_entry_to_chunk(entry, scrub_text) -> TranscriptChunk:
    """Convert a single JournalEntry to a TranscriptChunk."""
    user_text = f"Session focus: {entry.focus}" if entry.focus else ""

    parts: list[str] = []
    if entry.done:
        parts.append("Done: " + " - ".join(entry.done))
    if entry.decisions:
        parts.append("Decisions: " + " - ".join(entry.decisions))
    if entry.next_steps:
        parts.append("Next steps: " + " - ".join(entry.next_steps))
    if entry.branch:
        parts.append(f"Branch: {entry.branch}")
    assistant_text = "\n".join(parts)

    # Scrub secrets from journal content (matches transcript parser behavior)
    try:
        user_text = scrub_text(user_text)
        assistant_text = scrub_text(assistant_text)
    except Exception:
        pass  # Use unscrubbed text rather than losing the entry

    # Stable content-based ID: hash of session_id + timestamp
    session_short = _short_sid(entry.session_id) if entry.session_id else entry.timestamp[:10]
    ts_hash = hashlib.sha256(
        f"{entry.session_id}:{entry.timestamp}".encode()
    ).hexdigest()[:8]
    chunk_id = f"{session_short}:journal:{ts_hash}"
    session_id = entry.session_id or f"journal-{entry.timestamp}"

    return TranscriptChunk(
        id=chunk_id,
        session_id=session_id,
        timestamp=entry.timestamp,
        turn_index=-1,
        user_text=user_text,
        assistant_text=assistant_text,
        files_touched=list(entry.files_modified),
    )


def parse_journal_entries(journal_path: Path) -> list[TranscriptChunk]:
    """Convert journal entries into searchable TranscriptChunks.

    Each JournalEntry becomes one chunk with turn_index=-1 as a sentinel
    to distinguish journal chunks from regular transcript turns.

    When multiple entries share a session_id, non-auto entries (manual or
    enriched) take priority over auto-stubs. This supports the append-only
    pattern where enriched entries are appended rather than replacing stubs.

    Args:
        journal_path: Path to journal.jsonl file.

    Returns:
        List of TranscriptChunks, one per selected journal entry.
    """
    from synapt.recall.journal import JournalEntry
    from synapt.recall.scrub import scrub_text

    if not journal_path.exists():
        return []

    # Phase 1: Read all entries, grouped by session_id
    entries_by_session: dict[str, list] = {}
    with open(journal_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = JournalEntry.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if not entry.has_content():
                continue
            key = entry.session_id or entry.timestamp
            entries_by_session.setdefault(key, []).append(entry)

    # Phase 2: For each session, drop auto-stubs if a better entry exists
    selected = []
    for _sid, entries in entries_by_session.items():
        non_auto = [e for e in entries if not e.auto]
        if non_auto:
            selected.extend(non_auto)
        else:
            # Only auto-stubs; keep the most recent one
            selected.append(max(entries, key=lambda e: e.timestamp))

    # Phase 3: Convert selected entries to TranscriptChunks
    return [_journal_entry_to_chunk(entry, scrub_text) for entry in selected]


# ---------------------------------------------------------------------------
# Multi-file index builder
# ---------------------------------------------------------------------------

def build_index(
    source_dir: Path,
    use_embeddings: bool = False,
    cache_dir: Path | None = None,
    incremental_manifest: dict | None = None,
    db: RecallDB | None = None,
    subchunk_min_text: int | None = None,
) -> TranscriptIndex:
    """Build a TranscriptIndex from a directory of .jsonl transcript files.

    Args:
        source_dir: Directory containing .jsonl transcript files.
        use_embeddings: Whether to compute embeddings.
        cache_dir: Directory for caching embeddings.
        incremental_manifest: If provided, skip files already indexed
                             (by checking source_files in manifest).

    Returns:
        TranscriptIndex over all parsed chunks.
    """
    jsonl_files = sorted(source_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[synapt] No .jsonl files found in {source_dir}")
        return TranscriptIndex([])

    forced_subchunk = os.environ.get("SYNAPT_FORCE_SUBCHUNK_MIN_TEXT")
    if subchunk_min_text is None and forced_subchunk:
        try:
            subchunk_min_text = int(forced_subchunk)
        except ValueError:
            pass

    # Filter for incremental builds — skip files whose mtime AND size match
    # Keyed on (source dir, name), never the basename alone: `source_files` is
    # ONE flat list spanning every build source, so two worktrees archiving the
    # same session name yield two entries. Under a basename key the second
    # overwrites the first, and the loser's stamp can never match its own file —
    # so it re-parses on every incremental build, forever, with no error.
    #
    # The key uses the source dir's BASENAME, which is unique only because the
    # build sources are siblings under one root. A future multi-root source
    # list can collide here the same way basenames collide today, and would
    # need the fuller path. Impossible by topology now; stated so that whoever
    # adds the second root meets this instead of rediscovering it.
    already_indexed: dict[tuple[str, str], tuple[float, int]] = {}
    # Manifests written before entries carried "dir" keep the old flat key.
    # They cannot distinguish colliding basenames — that is the defect — but a
    # legacy manifest is better than forcing one full rebuild on upgrade, and
    # the next build rewrites the manifest in the scoped form.
    legacy_indexed: dict[str, tuple[float, int]] = {}
    if incremental_manifest:
        for src in incremental_manifest.get("source_files", []):
            stamp = (src.get("mtime", 0), src.get("size", 0))
            src_dir = src.get("dir")
            if src_dir is None:
                legacy_indexed[src["name"]] = stamp
            else:
                already_indexed[(src_dir, src["name"])] = stamp

    from synapt.recall.codex import is_codex_transcript, parse_codex_transcript

    all_chunks: list[TranscriptChunk] = []
    seen_uuids: set[str] = set()
    skipped = 0
    parsed_files: list[Path] = []  # Track which files were actually parsed

    for filepath in jsonl_files:
        stamp = already_indexed.get((filepath.parent.name, filepath.name))
        if stamp is None:
            stamp = legacy_indexed.get(filepath.name)
        if stamp is not None:
            stored_mtime, stored_size = stamp
            stat = filepath.stat()
            if stat.st_mtime == stored_mtime and stat.st_size == stored_size:
                skipped += 1
                continue
        try:
            if is_codex_transcript(filepath):
                chunks = parse_codex_transcript(filepath, seen_uuids=seen_uuids)
            else:
                chunks = parse_transcript(filepath, seen_uuids=seen_uuids,
                                          subchunk_min_text=subchunk_min_text)
            all_chunks.extend(chunks)
            parsed_files.append(filepath)
            print(f"  {filepath.name}: {len(chunks)} turns")
        except Exception as e:
            print(f"  {filepath.name}: ERROR - {e}")

    if skipped:
        print(f"[synapt] Skipped {skipped} already-indexed files")
    print(f"[synapt] Total: {len(all_chunks)} chunks from {len(jsonl_files) - skipped} files")

    # Auto-detect content profile and re-parse if sub-chunking should differ.
    # Personal content needs full turns (subchunk=0); code benefits from splitting.
    # Only re-parses when caller didn't explicitly set subchunk_min_text.
    if subchunk_min_text is None and all_chunks:
        from synapt.recall.content_profile import (
            adaptive_params,
            detect_content_profile,
            forced_content_profile,
        )
        profile = forced_content_profile(total_chunks=len(all_chunks)) or detect_content_profile(all_chunks)
        profile_threshold = adaptive_params(profile).subchunk_min_text
        default_threshold = int(os.environ.get("SYNAPT_SUBCHUNK_MIN_TEXT", "1200"))
        if profile_threshold != default_threshold:
            has_subchunks = any(c.user_text.startswith("(context:") for c in all_chunks)
            if has_subchunks and profile_threshold == 0:
                # Personal content was sub-chunked — re-parse without splitting
                logger.info("Content profile %s: re-parsing without sub-chunking",
                            profile.content_type)
                all_chunks = []
                seen_uuids_reparse: set[str] = set()
                for filepath in parsed_files:
                    try:
                        if is_codex_transcript(filepath):
                            chunks = parse_codex_transcript(filepath, seen_uuids=seen_uuids_reparse)
                        else:
                            chunks = parse_transcript(filepath, seen_uuids=seen_uuids_reparse,
                                                      subchunk_min_text=0)
                        all_chunks.extend(chunks)
                    except Exception:
                        pass
                print(f"[synapt] Re-parsed: {len(all_chunks)} chunks (sub-chunking disabled)")

    return TranscriptIndex(all_chunks, use_embeddings=use_embeddings, cache_dir=cache_dir, db=db)
