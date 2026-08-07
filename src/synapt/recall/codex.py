"""Parse Codex CLI transcripts into TranscriptChunks.

Codex CLI stores sessions at ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl.
The format differs from Claude Code:
  - session_meta entry has session ID and cwd
  - response_item entries with role: user/developer/assistant
  - function_call / function_call_output and custom_tool_call /
    custom_tool_call_output for tool use
  - Content blocks use input_text/output_text types

This module converts Codex transcripts into the same TranscriptChunk format
used for Claude Code, enabling cross-editor memory via synapt recall.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import re

from synapt.recall.core import TranscriptChunk, _short_sid, project_archive_dir

# Simple file path regex — matches common Unix and Windows absolute file paths
_FILE_RE = re.compile(
    r'(?:^|[\s"\'`])'
    r'('
    r'(?:/[^\s"\'`:,)]+?\.\w{1,10})'
    r'|'
    r'(?:[A-Za-z]:\\[^\n\r\t"\'`]+?\.\w{1,10})'
    r')'
    r'(?=[\s"\'`:,)]|$)'
)


def _extract_file_paths(text: str) -> list[str]:
    """Extract likely file paths from text."""
    return list(dict.fromkeys(_FILE_RE.findall(text)))


def discover_codex_sessions() -> Path | None:
    """Return the Codex sessions directory if it exists."""
    codex_dir = Path.home() / ".codex" / "sessions"
    if codex_dir.is_dir():
        return codex_dir
    return None


def _project_roots(project_dir: Path | None = None) -> list[Path]:
    """Return project roots whose Codex sessions should be indexed."""
    from synapt.recall.core import _find_gripspace_root, _git_main_worktree_root

    actual_dir = (project_dir or Path.cwd()).resolve()
    roots: list[Path] = []

    def _add_root(path: Path | None) -> None:
        if path is None:
            return
        resolved = path.resolve()
        if resolved not in roots:
            roots.append(resolved)

    _add_root(actual_dir)
    _add_root(_git_main_worktree_root(actual_dir))

    grip_root = _find_gripspace_root(actual_dir)
    if grip_root is not None:
        # Match Claude transcript discovery semantics: in a gripspace we treat
        # direct child repos as part of the same shared recall surface.
        try:
            children = sorted(grip_root.iterdir())
        except OSError:
            children = []
        for child in children:
            if child.is_dir() and (child / ".git").exists():
                _add_root(child)

    return roots


def _session_cwd(path: Path) -> Path | None:
    """Return the session cwd recorded in a Codex transcript, if present."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "session_meta":
                    return None
                cwd = entry.get("payload", {}).get("cwd", "")
                if not cwd:
                    return None
                try:
                    return Path(cwd).resolve()
                except OSError:
                    return None
    except OSError:
        return None
    return None


def is_codex_transcript(path: Path) -> bool:
    """True when *path* looks like a Codex CLI transcript."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                return entry.get("type") == "session_meta"
    except OSError:
        return False
    return False


def _matches_project(path: Path, project_dir: Path | None = None) -> bool:
    """True when the Codex transcript belongs to the current project scope."""
    session_cwd = _session_cwd(path)
    if session_cwd is None:
        return False

    for root in _project_roots(project_dir):
        if session_cwd == root or root in session_cwd.parents:
            return True
    return False


def list_codex_transcripts(
    sessions_dir: Path | None = None,
    project_dir: Path | None = None,
) -> list[Path]:
    """List project-relevant Codex transcript JSONL files, sorted by name."""
    if sessions_dir is None:
        sessions_dir = discover_codex_sessions()
    if sessions_dir is None:
        return []

    files = sorted(sessions_dir.rglob("rollout-*.jsonl"))
    if project_dir is None:
        return files
    return [path for path in files if _matches_project(path, project_dir)]


def archive_codex_transcripts(
    project_dir: Path,
    sessions_dir: Path | None = None,
) -> list[Path]:
    """Copy project-relevant Codex transcripts into the project archive.

    Uses the same size-based semantics as Claude transcript archiving:
    unchanged files are skipped, grown files overwrite the archive copy,
    and shrunken sources do not replace larger archived copies.
    """
    archive_dir = project_archive_dir(project_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for src_file in list_codex_transcripts(sessions_dir, project_dir=project_dir):
        dst_file = archive_dir / src_file.name
        src_size = src_file.stat().st_size
        if dst_file.exists():
            dst_size = dst_file.stat().st_size
            if src_size == dst_size:
                continue
            if src_size < dst_size:
                continue
        dst_file.write_bytes(src_file.read_bytes())
        copied.append(dst_file)

    return copied


def parse_codex_transcript(
    path: Path,
    seen_uuids: set[str] | None = None,
) -> list[TranscriptChunk]:
    """Parse a Codex CLI transcript into TranscriptChunks.

    Maps the Codex JSONL format to the same chunk structure used for
    Claude Code transcripts, enabling unified search and recall.

    Args:
        path: Path to a rollout-*.jsonl file.
        seen_uuids: Set of already-seen session IDs for dedup.

    Returns:
        List of TranscriptChunk objects.
    """
    if seen_uuids is None:
        seen_uuids = set()

    chunks: list[TranscriptChunk] = []
    session_id = ""
    transcript_path = str(path)

    # Accumulator for current turn
    current_user_text = ""
    current_assistant_texts: list[str] = []
    current_tools: list[str] = []
    current_files: list[str] = []
    current_timestamp = ""
    current_tool_summaries: list[str] = []
    turn_index = 0
    turn_start_offset = 0
    current_offset = 0

    def _record_tool_call(payload: dict) -> None:
        """Record either supported Codex tool-call envelope."""
        tool_name = payload.get("name", "unknown")
        current_tools.append(tool_name)
        args = payload.get("arguments", payload.get("input", ""))
        if not isinstance(args, str):
            return

        summary = args[:200]
        if len(args) < 500:
            try:
                args_parsed = json.loads(args)
                cmd = args_parsed.get("cmd", "")
                if cmd:
                    summary = cmd
                    current_files.extend(_extract_file_paths(cmd))
            except (json.JSONDecodeError, TypeError):
                pass
        current_tool_summaries.append(f"[{tool_name}] {summary}")

    def _flush_turn(end_offset: int = 0):
        nonlocal turn_index
        if not current_user_text and not current_assistant_texts:
            return

        short_id = _short_sid(session_id) if session_id else path.stem[:8]
        chunk_id = f"{short_id}:t{turn_index}"

        # Dedup
        if chunk_id in seen_uuids:
            return
        seen_uuids.add(chunk_id)

        assistant_text = "\n".join(current_assistant_texts).strip()
        if len(assistant_text) > 5000:
            assistant_text = assistant_text[:5000] + "..."

        tool_content = "\n".join(current_tool_summaries).strip()
        if len(tool_content) > 3000:
            tool_content = tool_content[:3000] + "..."

        # Extract files from all text
        all_text = current_user_text + " " + assistant_text + " " + tool_content
        files = list(dict.fromkeys(current_files + _extract_file_paths(all_text)))

        chunk = TranscriptChunk(
            id=chunk_id,
            session_id=session_id or path.stem,
            timestamp=current_timestamp,
            turn_index=turn_index,
            user_text=current_user_text.strip(),
            assistant_text=assistant_text,
            tools_used=list(dict.fromkeys(current_tools)),
            files_touched=files[:20],  # Cap to avoid bloat
            tool_content=tool_content,
            transcript_path=transcript_path,
            byte_offset=turn_start_offset,
            byte_length=end_offset - turn_start_offset,
        )
        chunks.append(chunk)
        turn_index += 1

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line_bytes = len(line.encode("utf-8"))
                line = line.strip()
                if not line:
                    current_offset += line_bytes
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    current_offset += line_bytes
                    continue

                entry_type = entry.get("type", "")
                timestamp = entry.get("timestamp", "")
                payload = entry.get("payload", {})

                if entry_type == "session_meta":
                    # Extract session ID from metadata
                    session_id = payload.get("id", path.stem)
                    if session_id in seen_uuids:
                        return []  # Already indexed
                    seen_uuids.add(session_id)

                elif entry_type == "response_item":
                    payload_type = payload.get("type", "")
                    role = payload.get("role", "")
                    content_blocks = payload.get("content", [])
                    phase = payload.get("phase", "")

                    # Handle function calls first (no role field)
                    if payload_type in {"function_call", "custom_tool_call"}:
                        _record_tool_call(payload)
                        current_offset += line_bytes
                        continue

                    if payload_type in {"function_call_output", "custom_tool_call_output"}:
                        # Tool output is deliberately handled as a no-op: call metadata
                        # above is enough for transcript recall, while output can be noisy.
                        current_offset += line_bytes
                        continue

                    if role == "user":
                        # New user turn — flush previous
                        _flush_turn(current_offset)
                        current_user_text = ""
                        current_assistant_texts = []
                        current_tools = []
                        current_files = []
                        current_tool_summaries = []
                        current_timestamp = timestamp
                        turn_start_offset = current_offset

                        for block in content_blocks:
                            text = block.get("text", "")
                            if block.get("type") == "input_text" and text:
                                # Skip system-level content (permissions, env context)
                                if text.startswith("<permissions") or text.startswith("<environment"):
                                    continue
                                # Skip AGENTS.md injections
                                if text.startswith("# AGENTS.md"):
                                    continue
                                current_user_text += text + "\n"

                    elif role == "assistant":
                        # Assistant response text
                        for block in content_blocks:
                            text = block.get("text", "")
                            if block.get("type") == "output_text" and text:
                                # Skip commentary phase — it's intermediate thinking
                                if phase == "commentary":
                                    continue
                                current_assistant_texts.append(text)

                    elif role == "developer":
                        # Developer role = system prompts — skip (too noisy)
                        pass

                elif entry_type == "event_msg":
                    msg_type = payload.get("type", "")
                    if msg_type == "user_message":
                        # Sometimes user text comes via event_msg instead of response_item
                        text = payload.get("message", "")
                        if text and not current_user_text:
                            _flush_turn(current_offset)
                            current_user_text = text
                            current_assistant_texts = []
                            current_tools = []
                            current_files = []
                            current_tool_summaries = []
                            current_timestamp = timestamp
                            turn_start_offset = current_offset
                    elif msg_type == "agent_message":
                        text = payload.get("message", "")
                        if text and (
                            not current_assistant_texts
                            or current_assistant_texts[-1] != text
                        ):
                            current_assistant_texts.append(text)

                current_offset += line_bytes

        # Flush final turn
        _flush_turn(current_offset)

    except OSError:
        pass

    return chunks
