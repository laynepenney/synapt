"""Parse Codex CLI transcripts into TranscriptChunks.

Codex CLI stores sessions at ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl.
The path date is the session start date, not its most recent activity: a live
session can append to the same rollout file for days. Determine recency and
liveness from content mtime or offsets, never the path date. Path sorting is
start-order, not recency-order.
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

_CUSTOM_TOOL_SUMMARY_LIMIT = 160
_CUSTOM_TOOL_SUMMARY_COUNT = 4


def _extract_file_paths(text: str) -> list[str]:
    """Extract likely file paths from text."""
    return list(dict.fromkeys(_FILE_RE.findall(text)))


def discover_codex_sessions() -> Path | None:
    """Return the Codex sessions directory if it exists."""
    codex_dir = Path.home() / ".codex" / "sessions"
    if codex_dir.is_dir():
        return codex_dir
    return None


_CWD_CACHE_FILENAME = "codex_cwd_cache.json"

# Process-wide, opt-in cache for _session_cwd's (path -> cwd) lookups, keyed
# by (size, mtime) so a changed file is never served a stale answer. None
# means the cache is not configured: _session_cwd behaves exactly as before,
# so every existing caller that never opts in sees no behavior change.
#
# ~/.codex/sessions is a single machine-wide directory (recall#1125):
# resume's cold path opens and line-scans every rollout file in it on every
# invocation, unscoped to gripspace or store size, because _session_cwd has
# no memory across the many call sites that need a transcript's cwd
# (caller_transcripts' own loop, and every list_codex_transcripts caller via
# _matches_project). Caching inside _session_cwd itself, rather than at any
# one call site, means every caller benefits without its own signature
# changing — the fix is scoped to the function whose repeated cost is the
# actual problem.
_cwd_cache: dict[str, dict] | None = None
_cwd_cache_dirty = False
_cwd_cache_index_dir: Path | None = None


def configure_cwd_cache(index_dir: Path | None) -> None:
    """Point the process-wide session-cwd cache at *index_dir*'s cache file,
    loading it if present. Call once near the start of a command that will
    scan Codex sessions; pair with ``flush_cwd_cache()`` before exit.

    A missing, unreadable, or corrupt cache file degrades to an empty cache
    rather than failing the caller — the cache is a courtesy, never a
    dependency to unlock resume/build.
    """
    global _cwd_cache, _cwd_cache_dirty, _cwd_cache_index_dir
    _cwd_cache_index_dir = index_dir
    _cwd_cache_dirty = False
    if index_dir is None:
        _cwd_cache = {}
        return
    try:
        _cwd_cache = json.loads((index_dir / _CWD_CACHE_FILENAME).read_text())
    except (OSError, json.JSONDecodeError):
        _cwd_cache = {}


def flush_cwd_cache() -> None:
    """Persist the process-wide session-cwd cache if it changed. Best-effort:
    a read-only or vanished index_dir must not turn a cache write failure
    into a resume/build failure.

    Prunes entries for files that no longer exist before writing: a removed
    Codex rollout never grows the cache back to the cost it exists to avoid
    (a stat() failure already refuses to serve a stale entry on read — see
    _session_cwd — so this is a cleanup at write time, not a correctness fix
    at read time).

    Written atomically (temp file in the same directory, then os.replace):
    build, rebuild, rescrub, hook, setup, and resume can all flush this same
    cache file, so a crash or a second concurrent flush mid-write must never
    leave a half-written, unparseable cache file for the next reader —
    configure_cwd_cache already degrades a corrupt file to an empty cache,
    but a corrupt file discards every entry it held, not just the ones that
    changed. os.replace is atomic on both POSIX and Windows.
    """
    global _cwd_cache_dirty
    if _cwd_cache_index_dir is None or not _cwd_cache_dirty or _cwd_cache is None:
        return
    for key in [k for k in _cwd_cache if not Path(k).exists()]:
        del _cwd_cache[key]
    cache_path = _cwd_cache_index_dir / _CWD_CACHE_FILENAME
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(_cwd_cache))
        os.replace(tmp_path, cache_path)
        _cwd_cache_dirty = False
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


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
    """Return the session cwd recorded in a Codex transcript, if present.

    Consults the process-wide cache configured via ``configure_cwd_cache``
    when one is active, keyed by (size, mtime) so a file that changed since
    it was cached is never served a stale answer. With no cache configured
    (the default), this reads the file every call, exactly as before.
    """
    global _cwd_cache_dirty

    cache = _cwd_cache
    stat_result = None
    if cache is not None:
        try:
            stat_result = path.stat()
        except OSError:
            return None
        key = str(path)
        cached = cache.get(key)
        if (
            cached is not None
            and cached.get("size") == stat_result.st_size
            and cached.get("mtime") == stat_result.st_mtime
        ):
            cwd = cached.get("cwd")
            return Path(cwd) if cwd else None

    cwd = _session_cwd_uncached(path)

    if cache is not None:
        if stat_result is None:
            try:
                stat_result = path.stat()
            except OSError:
                return cwd
        cache[str(path)] = {
            "size": stat_result.st_size,
            "mtime": stat_result.st_mtime,
            "cwd": str(cwd) if cwd else None,
        }
        _cwd_cache_dirty = True

    return cwd


_SESSION_CWD_SCAN_LINE_CAP = 50


def _session_cwd_uncached(path: Path) -> Path | None:
    """The real scan _session_cwd caches: open *path* and read its recorded cwd.

    A well-formed Codex rollout records session_meta as its first entry, so
    the loop normally reads one line. Bounded to
    _SESSION_CWD_SCAN_LINE_CAP non-empty lines so a malformed or corrupt
    file (leading blank lines, unparseable garbage) cannot force an
    unbounded scan of an otherwise-multi-thousand-turn rollout on a cache
    miss (recall#1125).
    """
    try:
        with open(path, encoding="utf-8") as f:
            scanned = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                scanned += 1
                if scanned > _SESSION_CWD_SCAN_LINE_CAP:
                    return None
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
    """List project-relevant Codex transcript JSONL files, sorted by name.

    Scan every rollout path. Its YYYY/MM/DD directories record session start
    date, while live sessions append to that file. Date filtering or path-based
    recency would hide an active long-lived session.
    """
    if sessions_dir is None:
        sessions_dir = discover_codex_sessions()
    if sessions_dir is None:
        return []

    files = sorted(sessions_dir.rglob("rollout-*.jsonl"))
    if project_dir is None:
        return files
    return [path for path in files if _matches_project(path, project_dir)]


def _has_buildable_transcripts(
    project_dir: Path,
    sessions_dir: Path | None = None,
) -> bool:
    """True when at least one discoverable Codex session belongs to *project_dir*.

    Exists so the build's "no transcripts found" pre-check can ask the SAME
    question the ingestion step will answer, rather than a narrower one.

    The pre-check previously counted live Claude transcript directories and
    archived transcripts only. A project whose entire history is Codex sessions
    therefore failed the guard and exited before ``archive_codex_transcripts``
    ran -- so the sessions that would have satisfied the build were never
    discovered. The guard and the thing it gates disagreed, and the guard won.

    Deliberately reuses ``list_codex_transcripts`` rather than reimplementing
    discovery. A second copy of "which sessions belong to this project" would
    be free to drift from the one that actually does the archiving, and a
    pre-check that drifts from its subject is how this defect existed at all.
    """
    return bool(list_codex_transcripts(sessions_dir, project_dir=project_dir))


def archive_codex_transcripts(
    project_dir: Path,
    sessions_dir: Path | None = None,
    store_dir: Path | None = None,
) -> list[Path]:
    """Copy project-relevant Codex transcripts into the project archive.

    Uses the same size-based semantics as Claude transcript archiving:
    unchanged files are skipped, grown files overwrite the archive copy,
    and shrunken sources do not replace larger archived copies.

    *project_dir* is the SOURCE scope — it filters which Codex rollouts belong
    to this project. *store_dir*, when given, is the STORE scope the archive is
    written into; it defaults to *project_dir* so existing callers are
    unchanged. The two differ only when a build must write into a store that is
    resolved separately from the source (this change's R2 (Atlas): a cold no-caller
    resume on a spawned desk archives cwd's rollouts INTO the GRIPSPACE_ROOT
    store, never a cwd-derived secondary one).
    """
    archive_dir = project_archive_dir(store_dir if store_dir is not None else project_dir)
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
    *,
    start_offset: int = 0,
    stop_offset: int | None = None,
    turn_index_start: int = 0,
    session_id_override: str | None = None,
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
    session_id = session_id_override or ""
    transcript_path = str(path)

    # Accumulator for current turn
    current_user_text = ""
    current_assistant_texts: list[str] = []
    current_commentary_texts: list[str] = []
    current_tools: list[str] = []
    current_files: list[str] = []
    current_timestamp = ""
    current_tool_summaries: list[str] = []
    current_custom_tool_summaries = 0
    current_custom_tool_summaries_omitted = 0
    turn_index = turn_index_start
    turn_start_offset = start_offset
    current_offset = start_offset

    def _record_custom_tool_call(payload: dict) -> None:
        """Record a custom Codex tool-call envelope with an input summary."""
        nonlocal current_custom_tool_summaries, current_custom_tool_summaries_omitted
        tool_name = payload.get("name", "unknown")
        current_tools.append(tool_name)
        args = payload.get("arguments", payload.get("input", ""))
        if not isinstance(args, str):
            return

        summary = args[:_CUSTOM_TOOL_SUMMARY_LIMIT]
        if len(args) < 500:
            try:
                args_parsed = json.loads(args)
                cmd = args_parsed.get("cmd", "")
                if cmd:
                    summary = cmd[:_CUSTOM_TOOL_SUMMARY_LIMIT]
                    current_files.extend(_extract_file_paths(cmd))
            except (json.JSONDecodeError, TypeError):
                pass
        if current_custom_tool_summaries < _CUSTOM_TOOL_SUMMARY_COUNT:
            current_tool_summaries.append(f"[{tool_name}] {summary}")
            current_custom_tool_summaries += 1
        else:
            current_custom_tool_summaries_omitted += 1

    def _append_assistant_text(text: str) -> None:
        """Retain each assistant message once, regardless of its producer."""
        if text and text not in current_assistant_texts:
            current_assistant_texts.append(text)

    def _append_commentary_text(text: str) -> None:
        """Retain each commentary message outside the primary retrieval text."""
        if text and text not in current_commentary_texts:
            current_commentary_texts.append(text)

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
        commentary_text = "\n".join(current_commentary_texts).strip()
        if len(commentary_text) > 5000:
            commentary_text = commentary_text[:5000] + "..."

        tool_summary_lines = list(current_tool_summaries)
        if current_custom_tool_summaries_omitted:
            tool_summary_lines.append(
                f"+{current_custom_tool_summaries_omitted} more tool calls"
            )
        tool_content = "\n".join(tool_summary_lines).strip()
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
            commentary_text=commentary_text,
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
            if start_offset:
                f.seek(start_offset)
            for line in f:
                line_bytes = len(line.encode("utf-8"))
                if stop_offset is not None and current_offset + line_bytes > stop_offset:
                    break
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
                    session_id = session_id_override or payload.get("id", path.stem)
                    if session_id in seen_uuids:
                        return []  # Already indexed
                    seen_uuids.add(session_id)

                elif entry_type == "response_item":
                    payload_type = payload.get("type", "")
                    role = payload.get("role", "")
                    content_blocks = payload.get("content", [])
                    phase = payload.get("phase", "")

                    # Handle function calls first (no role field)
                    if payload_type == "function_call":
                        # Keep legacy function-call summary semantics unchanged.
                        tool_name = payload.get("name", "unknown")
                        current_tools.append(tool_name)
                        args = payload.get("arguments", "")
                        if isinstance(args, str) and len(args) < 500:
                            try:
                                args_parsed = json.loads(args)
                                cmd = args_parsed.get("cmd", "")
                                if cmd:
                                    current_tool_summaries.append(f"[{tool_name}] {cmd}")
                                    current_files.extend(_extract_file_paths(cmd))
                            except (json.JSONDecodeError, TypeError):
                                current_tool_summaries.append(f"[{tool_name}] {args[:200]}")
                        current_offset += line_bytes
                        continue

                    if payload_type == "custom_tool_call":
                        _record_custom_tool_call(payload)
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
                        current_commentary_texts = []
                        current_tools = []
                        current_files = []
                        current_tool_summaries = []
                        current_custom_tool_summaries = 0
                        current_custom_tool_summaries_omitted = 0
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
                                    _append_commentary_text(text)
                                    continue
                                _append_assistant_text(text)

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
                            current_commentary_texts = []
                            current_tools = []
                            current_files = []
                            current_tool_summaries = []
                            current_custom_tool_summaries = 0
                            current_custom_tool_summaries_omitted = 0
                            current_timestamp = timestamp
                            turn_start_offset = current_offset
                    elif msg_type == "agent_message":
                        text = payload.get("message", "")
                        if payload.get("phase") == "commentary":
                            _append_commentary_text(text)
                        else:
                            _append_assistant_text(text)

                current_offset += line_bytes

        # Flush final turn
        _flush_turn(current_offset)

    except OSError:
        pass

    return chunks
