"""synapt.recall MCP server — expose transcript search as tools for Claude Code.

Provides fifteen tools via the Model Context Protocol:
  - recall_search: Search past session transcripts by keyword/topic
  - recall_quick: Fast, low-cost knowledge-only search for speculative checks
  - recall_context: Drill down into a search result for full raw content
  - recall_files: Find file history when you need prior context or rationale
  - recall_sessions: List recent sessions with summaries
  - recall_timeline: View chronological timeline of work arcs
  - recall_build: Build or rebuild the transcript index
  - recall_setup: Initialize synapt recall for the current project
  - recall_stats: Get index statistics
  - recall_journal: Read or write session journal entries
  - recall_save: Explicitly save durable knowledge nodes
  - recall_remind: Manage cross-session reminders
  - recall_enrich: Enrich chunks with LLM-generated summaries
  - recall_consolidate: Extract durable knowledge from journal entries
  - recall_contradict: Manage pending knowledge contradictions

Can run standalone (synapt-server) or be composed into the
unified synapt server via register_tools(mcp).
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import synapt.recall as _synapt_pkg
from synapt.recall.config import load_config

# Capture version at server startup for stale-process detection
_STARTUP_VERSION = getattr(_synapt_pkg, "__version__", "unknown")
from synapt.recall.core import (
    TranscriptIndex,
    format_size,
    project_data_dir,
    project_index_dir,
)
from synapt.recall._llm_util import truncate_at_word as _tw
from synapt.recall.embeddings import get_embedding_provider
from synapt.recall.hybrid import classify_query_intent, intent_search_params


def _cap_tokens(requested: int) -> int:
    """Apply the user-configured max_tokens cap."""
    limit = load_config().get_max_tokens()
    return min(requested, limit)

# ---------------------------------------------------------------------------
# MCP instructions — shared with the unified server (synapt.server)
# ---------------------------------------------------------------------------

MCP_INSTRUCTIONS = (
    "You have access to synapt recall — persistent memory across sessions. "
    "Search BEFORE you act, not after.\n"
    "\n"
    "WHEN TO SEARCH (do this automatically, without being asked):\n"
    "- When you need file history or design rationale for a specific path -> recall_files\n"
    "- Before making a design decision -> recall_search for prior discussion\n"
    "- When debugging an error -> recall_search for past fixes\n"
    "- When user references past work -> recall_search immediately\n"
    "- Starting a session -> recall_journal to read recent entries\n"
    "- When unsure if something was discussed before -> recall_quick\n"
    "- Before proposing a new approach -> recall_quick to check for prior attempts\n"
    "\n"
    "WHICH TOOL:\n"
    "- recall_quick: Fast, cheap knowledge check. Use speculatively when unsure.\n"
    "- recall_search: Full search with transcript chunks. Use when you need detail.\n"
    "- recall_files: Use for file history questions like 'who changed this and why?'\n"
    "- recall_journal: Read/write session notes. Check at session start.\n"
    "- recall_remind: Set/check cross-session reminders.\n"
    "\n"
    "DO NOT search for: general programming questions, syntax help, "
    "API docs, or anything not specific to this project's history.\n"
    "\n"
    "IMPORTANT: When in doubt, search. A quick recall_quick check costs ~500 tokens "
    "and takes <100ms. Missing relevant past context costs far more in wasted work "
    "and repeated mistakes. Err on the side of searching too much, not too little.\n"
    "\n"
    "CONTEXT BUDGET:\n"
    "- recall_channel has a `detail` parameter: max/high/medium/low/min.\n"
    "- Use detail='low' or 'min' for monitoring loops and periodic polling.\n"
    "- Use detail='high' or 'max' only when you need the full picture (e.g. catching up after being away).\n"
    "- Pins are large — they contain full benchmark tables. Read them once at session start with detail='high', then poll with 'low'.\n"
    "- Prefer pin=False for routine posts. Reserve pins for durable reference material."
)

# ---------------------------------------------------------------------------
# Cached index singleton — avoids reloading on every tool call.
# Invalidated when recall.db mtime changes (e.g., after a rebuild).
# ---------------------------------------------------------------------------

_cached_index: TranscriptIndex | None = None
_cached_mtime: float = 0.0
_cached_dir: Path | None = None
_cached_has_embeddings: bool = False


def _get_index(use_embeddings: bool = True) -> TranscriptIndex | None:
    """Load per-project index with caching. Reloads if recall.db was modified.

    Args:
        use_embeddings: If False, skip embedding model loading for faster
            startup. Use for recall_quick which only needs BM25/knowledge.
    """
    global _cached_index, _cached_mtime, _cached_dir, _cached_has_embeddings
    index_dir = project_index_dir()

    # Prefer recall.db, fall back to legacy chunks.jsonl
    db_path = index_dir / "recall.db"
    check_path = db_path if db_path.exists() else index_dir / "chunks.jsonl"
    if not check_path.exists():
        return None

    try:
        mtime = check_path.stat().st_mtime
        needs_reload = (
            _cached_index is None
            or mtime != _cached_mtime
            or index_dir != _cached_dir
            or (use_embeddings and not _cached_has_embeddings)
        )
        if needs_reload:
            # Close old DB connection before replacing cached index
            if _cached_index is not None and getattr(_cached_index, '_db', None) is not None:
                with contextlib.suppress(Exception):
                    _cached_index._db.close()
            import time as _time
            _load_t0 = _time.monotonic()
            logging.getLogger("synapt.recall").info("Loading index from %s ...", index_dir)
            _cached_index = TranscriptIndex.load(index_dir, use_embeddings=use_embeddings)
            logging.getLogger("synapt.recall").info(
                "Index loaded: %d chunks in %.1fs",
                len(_cached_index.chunks), _time.monotonic() - _load_t0,
            )
            _cached_has_embeddings = use_embeddings
            _cached_mtime = mtime
            _cached_dir = index_dir
            # Set current session ID for access tracking (distinct_sessions)
            try:
                from synapt.recall.journal import (
                    extract_session_id, latest_transcript_path,
        )
                live_path = latest_transcript_path()
                if live_path:
                    _cached_index._current_session_id = extract_session_id(live_path)
            except Exception:
                pass
        return _cached_index
    except Exception as e:
        logging.getLogger("synapt.recall").warning("Index load failed: %s", e)
        _cached_index = None
        return None


def _invalidate_cache() -> None:
    """Reset cached index so next search reloads from disk."""
    global _cached_index, _cached_mtime, _cached_dir, _cached_has_embeddings
    if _cached_index is not None and getattr(_cached_index, '_db', None) is not None:
        with contextlib.suppress(Exception):
            _cached_index._db.close()
    _cached_index = None
    _cached_mtime = 0.0
    _cached_dir = None
    _cached_has_embeddings = False


def _resolved_index_dir() -> Path:
    """The exact index root the MCP server has resolved for this request."""
    return _cached_dir or project_index_dir()


def _label_empty_result(result: str, index_dir: Path) -> str:
    """Attach scoped freshness to an MCP result that otherwise says nothing.

    A successful search or context read is already evidence about the specific
    content it returned. An empty result is different: without a freshness
    verdict it can be read as a claim that the requested history does not
    exist. Check the archive first and pay for live-source enumeration only
    when that cheap result is fresh.
    """
    from synapt.recall.freshness import check_index_freshness

    try:
        verdict = check_index_freshness(index_dir=index_dir)
        if not verdict.stale:
            verdict = check_index_freshness(index_dir=index_dir, deep=True)
    except Exception:
        return (
            f"{result}\n\n"
            f"Index freshness was not checked for root: {index_dir}. "
            "This does not establish that the requested history is absent."
        )

    if verdict.stale:
        behind = []
        if verdict.new_files:
            behind.append(f"{len(verdict.new_files)} file(s) not yet indexed")
        if verdict.changed_files:
            behind.append(f"{len(verdict.changed_files)} indexed file(s) grown since build")
        detail = "; ".join(behind) if behind else "the index is behind"
        status = (
            f"Index freshness: STALE ({detail}). Built "
            f"{verdict.build_timestamp or 'at an unrecorded time'}, "
            f"checked: {verdict.scanned}.\n"
            f"To index newer content: {verdict.remedy}"
        )
    else:
        status = (
            f"Index freshness: CURRENT. Built "
            f"{verdict.build_timestamp or 'at an unrecorded time'}, "
            f"checked: {verdict.scanned}."
        )
    return f"{result}\n\nIndex root: {index_dir}\n{status}"


# Clean up the DB connection before Python's module teardown begins.
# Without this, __del__ runs during shutdown when sqlite3 may already
# be partially torn down, causing a non-zero exit code that Claude Code
# reports as "MCP tool failed."
atexit.register(_invalidate_cache)


# ---------------------------------------------------------------------------
# Tool implementations — usable both as MCP tools and as plain functions.
# ---------------------------------------------------------------------------


def recall_search(
    query: str,
    max_chunks: int = 5,
    max_tokens: int = 1500,
    max_sessions: int | None = None,
    after: str | None = None,
    before: str | None = None,
    half_life: float = 60.0,
    threshold_ratio: float = 0.2,
    min_score: float | None = None,
    depth: str = "full",
    include_archived: bool = False,
    include_historical: bool = False,
    context: int = 0,
    min_confidence: float = 0.0,
    top_k: int = 0,
) -> str:
    """Search past coding sessions for relevant context. USE PROACTIVELY.

    Call this BEFORE starting work to check for prior decisions, bugs, or
    approaches. Returns relevant conversation chunks sorted by relevance.

    Works like grep: returns matching chunks with optional surrounding context.

    When to use (without being asked):
    - User mentions past work → search for it
    - Debugging an error → search for similar past errors
    - Making a design decision → check if it was discussed before
    - Starting a new feature → check for prior attempts or related work

    RESULT QUALITY:
        Use min_score to filter by relevance instead of fixed chunk counts.
        Results scoring below min_score × (best match score) are dropped.
        - min_score=0.2 (default): keep results ≥20% as relevant as the best match
        - min_score=0.5: stricter — only strong matches
        - min_score=0.0: return everything up to max_chunks (no quality filter)
        Combine with max_tokens to control context budget: "give me everything
        relevant, up to 2000 tokens."

    Args:
        query: Natural language query or keywords to search for.
        max_chunks: Maximum number of result chunks to return.
        max_tokens: Approximate token budget for the response.
        max_sessions: If set, only search the N most recent sessions.
        after: Only include results from after this date (ISO 8601, e.g. "2026-02-28").
        before: Only include results from before this date (ISO 8601, e.g. "2026-03-01").
        half_life: Days for recency decay to reach ~50%. 0 disables decay.
                   Default 60: a session from 2 months ago scores ~50% of today's,
                   making work from past quarters still discoverable.
                   Passed as None to lookup() so intent classification can
                   override when the caller uses the MCP default.
        min_score: Minimum relevance threshold (0.0-1.0). Results scoring below
                   min_score × (top result score) are dropped. Default 0.2 via
                   threshold_ratio. Higher = fewer, more relevant results.
                   Lower = more results, potentially noisy. 0 = no filter.
        threshold_ratio: Deprecated — use min_score instead. Same behavior.
        depth: "full" returns knowledge + journal + transcript results.
               "summary" returns only knowledge nodes + journal entries.
               "concise" returns knowledge + cluster summaries (no raw chunks).
        context: Number of surrounding chunks to include per match (like grep -C).
                 0 = just the matching chunk. 2 = matching chunk + 2 before/after.
        min_confidence: Minimum confidence for knowledge node results (0.0-1.0).
                       Filters out low-quality extractions. 0 = no filter.
        top_k: If set >0, return only the top K results regardless of other limits.
               Like grep -m K. 0 = use max_chunks (default behavior).
        include_archived: If True, include archived clusters in concise mode results.
                          In full mode, individual chunks are always searchable regardless.
        include_historical: If True, include superseded/contradicted knowledge nodes
                            in results. When multiple versions exist in the same lineage,
                            the highest-confidence one is kept (confidence-based fallback).
    """
    # min_score takes precedence over threshold_ratio when explicitly set
    if min_score is not None:
        threshold_ratio = min_score
    max_tokens = _cap_tokens(max_tokens)
    index = _get_index()

    # Search the live transcript for current-session context.
    # Skip when: (a) max_tokens=0, (b) `before` is set (the current session
    # is by definition "now" and cannot satisfy a historical cutoff).
    # Fixes recall#634: before-filtered queries no longer leak current-session
    # context that postdates the requested time window.
    historical_filter = before is not None or after is not None
    from synapt.recall.live import search_live_transcript
    live_budget = min(500, max_tokens // 3)
    live_result = ""
    if live_budget > 0 and not before:
        live_result = search_live_transcript(
            query,
            index=index,
            max_chunks=min(3, max_chunks),
            max_tokens=live_budget,
        )
    # Reserve budget for indexed search.  Floor is min(500, max_tokens) so
    # indexed search always gets a meaningful allocation without exceeding the
    # caller's total budget (a bare floor of 500 would exceed max_tokens when
    # max_tokens < 500, e.g. in tests or constrained callers).
    indexed_budget = max_tokens
    if live_result:
        # Ceiling division avoids rounding to 0 for very short live results
        # (len < 4 chars), which would deduct nothing from the indexed budget
        # despite emitting output.  In practice live blocks are always >100
        # chars, but the approximation should be accurate in all cases.
        live_consumed = (len(live_result) + 3) // 4
        budget_floor = min(500, max_tokens)
        indexed_budget = max(max_tokens - live_consumed, budget_floor)

    if index is None:
        if historical_filter:
            index_dir = project_index_dir()
            return (
                f"Historical search unavailable: no index found at {index_dir}. "
                f"Run `synapt recall setup` first. "
                f"Cannot satisfy date-filtered query without an index."
            )
        if live_result:
            return live_result
        index_dir = project_index_dir()
        return f"No index found at {index_dir}. Run `synapt recall setup` first."

    try:
        # Pass half_life=None when caller used the MCP default (60.0) so
        # intent classification can override it. When the user explicitly
        # sets a value, pass it through as-is to honour their choice.
        effective_hl: float | None = None if half_life == 60.0 else half_life
        # top_k overrides max_chunks for grep-style "give me N results"
        effective_max_chunks = top_k if top_k > 0 else max_chunks
        # min_confidence filters knowledge nodes post-retrieval
        effective_min_confidence = min_confidence
        result = index.lookup(
            query,
            max_chunks=effective_max_chunks,
            max_tokens=indexed_budget,
            max_sessions=max_sessions,
            after=after,
            before=before,
            half_life=effective_hl,
            threshold_ratio=threshold_ratio,
            depth=depth,
            include_archived=include_archived,
            include_historical=include_historical,
            min_confidence=effective_min_confidence,
            context=context,
        )
        # Combine live (current session) + indexed (past sessions) results
        parts = []
        if live_result:
            parts.append(live_result)
        if result:
            parts.append(result)

        # Surface contradiction warnings from co-retrieval detection
        conflicts = getattr(index, "_last_conflicts", [])
        if conflicts:
            warn_lines = [f"\n⚠ Conflicting information detected ({len(conflicts)} conflict(s)):"]
            for old, new in conflicts[:3]:  # Cap at 3 to avoid noise
                warn_lines.append(
                    f"  • \"{old.get('content', '')[:80]}\" vs \"{new.get('content', '')[:80]}\""
                )
            warn_lines.append("Use recall_contradict(action='list') to review and resolve.")
            parts.append("\n".join(warn_lines))

        # Surface embedding status when search is degraded
        if index._embedding_status == "unavailable":
            parts.append(
                f"\n[Note: Search is using keyword matching only (BM25). "
                f"{index._embedding_reason}]"
            )

        if parts:
            return "\n\n".join(parts)
        # Surface diagnostics explaining why search returned nothing
        diag = index._last_diagnostics
        if diag:
            msg = diag.format_message()
            if index._embedding_status == "unavailable":
                msg += (
                    f"\n[Note: Embeddings unavailable — semantic search disabled. "
                    f"{index._embedding_reason}]"
                )
            return _label_empty_result(msg, _resolved_index_dir())
        return _label_empty_result("No results found.", _resolved_index_dir())
    except Exception as exc:
        return f"Search failed: {exc}"


def recall_quick(query: str) -> str:
    """Quick, low-cost memory check. Use this speculatively — when you're
    not sure if past context exists but want to check.

    Returns concise results by default, and for pending-work queries it
    switches to summary mode so recent journal ``Next steps:`` entries can
    surface. If you find something relevant, follow up with recall_search
    or recall_context for full detail.

    Cost: ~500 tokens, <100ms. Cheaper than guessing wrong.
    Use BEFORE making assumptions about past work, decisions, or conventions.

    Args:
        query: Natural language query or keywords to search for.
    """
    quick_budget = _cap_tokens(500)
    intent = classify_query_intent(query)
    # Route depth by intent:
    # - status: summary (knowledge + journal next-steps)
    # - code: full (raw transcript chunks with file associations)
    # - everything else: concise (knowledge + cluster summaries)
    if intent == "status":
        depth = "summary"
    elif intent == "code":
        depth = "full"
    else:
        depth = "concise"
    params = intent_search_params(intent)
    index = _get_index(use_embeddings=False)
    if index is None:
        index_dir = project_index_dir()
        return f"No index found at {index_dir}. Run `synapt recall setup` first."

    try:
        result = index.lookup(
            query,
            max_chunks=5,
            max_tokens=quick_budget,
            half_life=params.get("half_life"),
            depth=depth,
            threshold_ratio=0.2,
            knowledge_boost=params.get("knowledge_boost"),
            max_knowledge=params.get("max_knowledge"),
        )
        if result:
            return result
        diag = index._last_diagnostics
        if diag:
            return diag.format_message()
        return "No results found."
    except Exception as exc:
        return f"Search failed: {exc}"


def recall_files(
    pattern: str,
    max_chunks: int = 10,
    max_tokens: int = 1500,
    after: str | None = None,
    before: str | None = None,
) -> str:
    """Find past sessions that touched a specific file.

    Searches the files_touched metadata of all indexed conversation turns.
    Supports partial path matching: 'repair.py' matches 'src/graph/repair.py'.
    Best when you need file history or design context, not as a generic
    "before editing" step.

    Args:
        pattern: File path or partial path to search for.
        max_chunks: Maximum number of result chunks to return.
        max_tokens: Approximate token budget for the response.
        after: Only include results from after this date (ISO 8601).
        before: Only include results from before this date (ISO 8601).
    """
    max_tokens = _cap_tokens(max_tokens)
    index = _get_index()
    if index is None:
        index_dir = project_index_dir()
        return f"No index found at {index_dir}. Run `synapt recall setup` first."

    try:
        result = index.lookup_files(
            pattern,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
            after=after,
            before=before,
        )
        return result if result else f"No sessions found that touched files matching '{pattern}'."
    except Exception as exc:
        return f"File search failed: {exc}"


def recall_sessions(
    max_sessions: int = 20,
    after: str | None = None,
    before: str | None = None,
) -> str:
    """List recent sessions with date, turn count, and first message.

    Returns a summary of the most recent sessions in the transcript index,
    useful for browsing what past sessions covered.

    Args:
        max_sessions: Maximum number of sessions to list.
        after: Only sessions with activity after this date (ISO 8601).
        before: Only sessions with activity before this date (ISO 8601).
    """
    index = _get_index()
    if index is None:
        index_dir = project_index_dir()
        return f"No index found at {index_dir}. Run `synapt recall setup` first."

    try:
        sessions = index.list_sessions(
            max_sessions=max_sessions,
            after=after,
            before=before,
        )
    except Exception as exc:
        return f"Session listing failed: {exc}"

    if not sessions:
        return "No sessions found."

    lines = [f"Recent sessions ({len(sessions)}):"]
    for s in sessions:
        lines.append(
            f"  {s['date']}  {s['session_id'][:8]}  "
            f"{s['turn_count']} turns  {s['files_count']} files  "
            f"\"{s['first_message']}\""
        )
    return "\n".join(lines)


def recall_build(incremental: bool = True) -> str:
    """Build or rebuild the transcript index from auto-discovered sources.

    Archives transcripts from Claude Code's source directory into the project,
    then builds a searchable index at <project>/.synapt/recall/index/.

    Args:
        incremental: If True, skip already-indexed files for faster rebuilds.
    """
    from synapt.recall.cli import _archive_and_build

    project = Path.cwd().resolve()
    try:
        final_index = _archive_and_build(
            project,
            use_embeddings=True,
            incremental=incremental,
        )
    except Exception as e:
        return f"Build failed: {e}"
    finally:
        _invalidate_cache()

    if not final_index or not final_index.chunks:
        return "No Claude Code or Codex transcripts found for this project."

    stats = final_index.stats()
    index_dir = project_index_dir(project)
    return (
        f"Index built: {stats['chunk_count']} chunks from "
        f"{stats['session_count']} sessions. Saved to {index_dir}"
    )


def recall_setup(no_hook: bool = False) -> str:
    """Initialize synapt recall for the current project.

    Archives transcripts, builds the search index, installs global hooks
    (SessionStart, SessionEnd, PreCompact), and adds .synapt/ to .gitignore.

    Args:
        no_hook: If True, skip installing global hooks.
    """
    from synapt.recall.cli import _archive_and_build, _ensure_gitignore, _install_global_hooks

    project = Path.cwd().resolve()
    steps: list[str] = []

    # 1. Archive transcripts + build index
    try:
        final_index = _archive_and_build(project, use_embeddings=True)
    except Exception as e:
        return f"Setup failed during index build: {e}"
    finally:
        _invalidate_cache()

    if final_index and final_index.chunks:
        stats = final_index.stats()
        steps.append(
            f"Index built: {stats['chunk_count']} chunks from "
            f"{stats['session_count']} sessions"
        )
    else:
        steps.append(
            "No transcripts found yet. Start a session, then run recall_setup again to index."
        )

    # 2. Install global hooks (user-level ~/.claude/settings.json)
    if not no_hook:
        installed = _install_global_hooks()
        if installed:
            steps.append(f"Installed {installed} global hook(s)")
        else:
            steps.append("Global hooks already registered")
    else:
        steps.append("Hook installation skipped (no_hook=True)")

    # 3. Add .synapt/ to .gitignore
    _ensure_gitignore(project)
    steps.append(".synapt/ ensured in .gitignore")

    # Summary
    index_dir = project_index_dir(project)
    if index_dir.exists() and any(index_dir.iterdir()):
        total_size = sum(fp.stat().st_size for fp in index_dir.iterdir() if fp.is_file())
        steps.append(f"Index size: {format_size(total_size)}")

    return "Setup complete.\n" + "\n".join(f"  - {s}" for s in steps)


def recall_export(
    output_path: str = "",
    exclude_transcripts: bool = False,
    exclude_channels: bool = False,
) -> str:
    """Export portable recall state to a .synapt-archive file.

    Args:
        output_path: Destination file path. Defaults to <project>.synapt-archive.
        exclude_transcripts: If True, omit archived raw transcript JSONL files.
        exclude_channels: If True, omit channel history files.
    """
    from synapt.recall.archive import export_recall_archive

    # None => resolve via SYNAPT_RECALL_ROOT + inference. Forwarding Path.cwd()
    # would suppress the override.
    try:
        archive_path, manifest = export_recall_archive(
            None,
            Path(output_path).expanduser() if output_path else None,
            exclude_transcripts=exclude_transcripts,
            exclude_channels=exclude_channels,
        )
    except Exception as e:
        return f"Export failed: {e}"

    return (
        f"Recall archive exported to {archive_path}\n"
        f"  - store: {manifest.get('data_dir', '?')}\n"
        f"  - chunks: {manifest.get('chunk_count', 0)}\n"
        f"  - knowledge: {manifest.get('knowledge_count', 0)}\n"
        f"  - worktrees: {manifest.get('worktree_count', 0)}"
    )


def recall_import(archive_path: str, mode: str = "replace") -> str:
    """Import portable recall state from a .synapt-archive file.

    Args:
        archive_path: Path to the exported .synapt-archive file.
        mode: Either "replace" or "merge".
    """
    from synapt.recall.archive import import_recall_archive

    try:
        summary = import_recall_archive(None, Path(archive_path), mode=mode)
    except Exception as e:
        return f"Import failed: {e}"

    return (
        f"Recall archive imported from {Path(archive_path).expanduser().resolve()}\n"
        f"  - mode: {summary.get('mode', mode)}\n"
        f"  - chunks: {summary.get('chunk_count', 0)}\n"
        f"  - knowledge: {summary.get('knowledge_count', 0)}\n"
        f"  - store: {summary.get('data_dir', '?')}"
    )


def recall_stats() -> str:
    """Get statistics about the transcript index.

    Returns chunk count, session count, date range, and index size.
    """
    index_dir = project_index_dir()

    # Check for recall.db or legacy manifest.json
    if not (index_dir / "recall.db").exists() and not (index_dir / "manifest.json").exists():
        return f"No index found at {index_dir}. Run `synapt recall setup` first."

    index = _get_index()
    if index is None:
        return f"No index found at {index_dir}."

    try:
        stats = index.stats()

        # Load manifest from DB or legacy file
        manifest: dict = {}
        try:
            if index._db:
                manifest = index._db.load_manifest()
            elif (index_dir / "manifest.json").exists():
                with open(index_dir / "manifest.json", encoding="utf-8") as f:
                    manifest = json.load(f)
        except Exception:
            pass

        lines = [
            f"synapt v{_STARTUP_VERSION}",
            f"Chunks: {stats.get('chunk_count', 0)}",
            f"Sessions: {stats.get('session_count', 0)}",
            f"Avg chunks/session: {stats.get('avg_chunks_per_session', 0):.1f}",
        ]
        if stats.get("date_range"):
            dr = stats["date_range"]
            lines.append(f"Date range: {dr['earliest'][:10]} to {dr['latest'][:10]}")
        lines.append(f"Unique tools: {stats.get('total_tools_used', 0)}")
        lines.append(f"Unique files: {stats.get('total_files_touched', 0)}")

        # Embedding status
        if stats.get("embeddings_active"):
            lines.append(f"Embeddings: active ({stats.get('embedding_provider', 'unknown')})")
        elif index._embedding_status == "unavailable":
            lines.append(
                f"Embeddings: unavailable — using BM25 keyword search only. "
                f"{index._embedding_reason}"
            )
        else:
            lines.append("Embeddings: inactive (not requested)")

        # Knowledge nodes
        kn_count = stats.get("knowledge_count", 0)
        if kn_count > 0:
            lines.append(f"Knowledge nodes: {kn_count}")

        # Pending contradictions
        if index._db:
            n_pending = index._db.pending_contradiction_count()
            if n_pending > 0:
                lines.append(f"Pending contradictions: {n_pending}")

        # Clusters
        if index._db:
            n_topic = index._db.cluster_count(cluster_type="topic")
            n_timeline = index._db.cluster_count(cluster_type="timeline")
            n_archived = index._db._conn.execute(
                "SELECT COUNT(*) FROM clusters WHERE status = 'archived'"
            ).fetchone()[0]
            if n_topic > 0 or n_timeline > 0 or n_archived > 0:
                cluster_parts = []
                if n_topic > 0:
                    cluster_parts.append(f"{n_topic} topic")
                if n_timeline > 0:
                    cluster_parts.append(f"{n_timeline} timeline arcs")
                if n_archived > 0:
                    cluster_parts.append(f"{n_archived} archived")
                lines.append(f"Clusters: {', '.join(cluster_parts)}")

        # Access tracking
        if index._db:
            access = index._db.access_summary()
            if access["total_events"] > 0:
                lines.append(
                    f"Access events: {access['total_events']} "
                    f"({access['tracked_items']} items, "
                    f"{access['items_drilled_into']} drilled into)"
                )
                tiers = access.get("promotion_tiers", {})
                if tiers:
                    tier_parts = [
                        f"{count} {tier}"
                        for tier, count in sorted(tiers.items())
                        if count > 0
                    ]
                    if tier_parts:
                        lines.append(f"Promotion tiers: {', '.join(tier_parts)}")
                # Decay distribution
                decay = index._db.decay_distribution()
                decay_total = sum(decay.values())
                if decay_total > 0:
                    lines.append(
                        f"Decay: {decay['fresh']} fresh, {decay['warm']} warm, "
                        f"{decay['cool']} cool, {decay['cold']} cold"
                    )

        # Storage backend
        backend = stats.get("storage_backend", "memory")
        lines.append(f"Storage: {backend}")

        build_ts = manifest.get("build_timestamp", "unknown")
        if isinstance(build_ts, str):
            lines.append(f"Built: {build_ts[:19]}")

        total_size = sum(fp.stat().st_size for fp in index_dir.iterdir() if fp.is_file())
        lines.append(f"Index size: {format_size(total_size)}")

        return "\n".join(lines)
    except Exception as exc:
        return f"Stats failed: {exc}"


def recall_consolidate(
    dry_run: bool = False,
    force: bool = False,
    model: str = "",
    adapter_path: str = "",
) -> str:
    """Consolidate session knowledge — extract durable patterns from journal entries.

    Analyzes enriched journal entries across multiple sessions to distill
    durable knowledge nodes: facts, conventions, decisions, and lessons
    that persist across sessions. Analogous to memory consolidation during sleep.

    Run this periodically (e.g., every few sessions) to build up the
    project's knowledge base. Knowledge nodes appear in search results
    and are surfaced at session start.

    Args:
        dry_run: Preview what would be consolidated without making changes.
        force: Reprocess all journal entries, ignoring last consolidation timestamp.
        model: MLX model to use (default: Ministral-3-3B-Instruct-2512-4bit).
        adapter_path: Optional LoRA adapter path for knowledge extraction.
    """
    try:
        from synapt.recall.consolidate import consolidate
        kwargs: dict = {"dry_run": dry_run, "force": force}
        if model:
            kwargs["model"] = model
        if adapter_path:
            kwargs["adapter_path"] = adapter_path
        result = consolidate(**kwargs)
        if dry_run:
            return (
                f"Dry run: {result.entries_processed} entries, "
                f"{result.clusters_found} clusters found"
            )
        parts = []
        if result.nodes_created:
            parts.append(f"{result.nodes_created} created")
        if result.nodes_corroborated:
            parts.append(f"{result.nodes_corroborated} corroborated")
        if result.nodes_contradicted:
            parts.append(f"{result.nodes_contradicted} contradicted")
        if result.nodes_contested:
            parts.append(f"{result.nodes_contested} contested")
        if result.nodes_deduped:
            parts.append(f"{result.nodes_deduped} deduped")
        if not parts:
            return (
                f"No knowledge extracted ({result.entries_processed} entries, "
                f"{result.clusters_found} clusters)"
            )
        return (
            f"Consolidated: {', '.join(parts)} "
            f"({result.entries_processed} entries, {result.clusters_found} clusters)"
        )
    except Exception as exc:
        return f"Consolidation failed: {exc}"
    finally:
        _invalidate_cache()


def recall_contradict(
    action: str = "list",
    contradiction_id: int | None = None,
    resolution: str = "confirmed",
    claim: str | None = None,
    new_content: str | None = None,
    old_node_id: str | None = None,
    reason: str | None = None,
) -> str:
    """Manage pending knowledge contradictions.

    When the system detects that a new observation contradicts an existing
    knowledge node, it queues a pending contradiction for user review.
    Use this tool to list, confirm, dismiss, or flag new contradictions.

    Args:
        action: "list" (show pending), "resolve" (confirm/dismiss one),
                "flag" (report a new contradiction),
                "forget" (archive a knowledge node — removes from search),
                "correct" (supersede a knowledge node with updated content).
        contradiction_id: ID of the contradiction to resolve (required for "resolve").
        resolution: "confirmed" (supersede old node) or "dismissed" (keep old node) for
                    ordinary contradictions. For a CONTESTED pair (Fix B, internal design
                    spec, section 10.4 --
                    both nodes already exist, both marked "contested"), use "candidate_wins"
                    (candidate promoted, existing superseded), "existing_wins" (existing
                    restored, candidate retired), or "false_positive" (both restored to
                    active, nothing superseded -- the judge's conflict flag was wrong).
        claim: Free-text description of conflicting information (for "flag").
               The system will search for matching knowledge nodes automatically.
        new_content: The correct/updated information (for "flag"). If omitted,
                     *claim* is used as both the description and the new content.
        old_node_id: Optional knowledge node ID that the claim contradicts.
                     If omitted, the system searches for matching nodes.
        reason: Why this is a contradiction (for "flag").
    """
    index = _get_index()
    if index is None or not index._db:
        return "No index found. Run `synapt recall setup` first."

    try:
        if action == "flag":
            return _handle_flag(
                index, claim=claim, new_content=new_content,
                old_node_id=old_node_id, reason=reason,
            )

        if action == "list":
            pending = index._db.list_pending_contradictions()
            if not pending:
                return "No pending contradictions."
            # Build lookup dict once (not per-iteration)
            old_ids = {c["old_node_id"] for c in pending if c["old_node_id"]}
            node_lookup = {
                nid: index._db.get_knowledge_node(nid)
                for nid in old_ids
            }
            lines = [f"Pending contradictions ({len(pending)}):"]
            for c in pending:
                old_node = node_lookup.get(c["old_node_id"]) if c["old_node_id"] else None
                old_content = old_node["content"] if old_node else ""
                lines.append(
                    f"\n  #{c['id']} ({c['detected_by']}, {c['detected_at'][:10]})"
                )
                if c.get("claim_text"):
                    lines.append(f"    Claim: {c['claim_text']}")
                if old_content:
                    lines.append(f"    Old: {old_content}")
                elif not c["old_node_id"]:
                    lines.append("    Old: (no matching node — free-text claim)")
                lines.append(f"    New: {c['new_content']}")
                if c["reason"]:
                    lines.append(f"    Reason: {c['reason']}")
            return "\n".join(lines)

        if action == "resolve":
            if contradiction_id is None:
                return "Error: contradiction_id is required for 'resolve' action."

            # Fetch BEFORE resolving (Fix B, internal design spec, section 10.4):
            # contest-row detection (new_node_id
            # set) determines how *resolution* maps onto the underlying confirmed/dismissed
            # vocabulary, and that mapping must be known before resolve_contradiction runs.
            # Filtering by status='pending' here is correct (unlike the single post-resolve
            # fetch this replaces, which deliberately read without a status filter because it
            # ran AFTER the row had already flipped away from 'pending').
            pending_row = index._db._conn.execute(
                "SELECT * FROM pending_contradictions WHERE id = ? AND status = 'pending'",
                (contradiction_id,),
            ).fetchone()
            if pending_row is None:
                return f"Contradiction #{contradiction_id} not found or already resolved."
            is_contest = (
                "new_node_id" in pending_row.keys() and bool(pending_row["new_node_id"])
            )

            if is_contest:
                if resolution not in ("candidate_wins", "existing_wins", "false_positive"):
                    return (
                        f"Error: contradiction #{contradiction_id} is a contested pair -- "
                        "resolution must be 'candidate_wins', 'existing_wins', or "
                        f"'false_positive' (got {resolution!r})."
                    )
                # recall#905 (0.17.0 blocker, Opus 2026-07-22, Part 2 -- CO-PRIMARY, not
                # merely defense-in-depth): the node-level mutation must succeed BEFORE the
                # pending row is marked resolved, never after. The dogfood-found bug: a
                # freshly-created contest's candidate node genuinely doesn't exist in SQLite
                # until some sync has run (Part 1 closes that gap at the source), so
                # _apply_contest_resolution's db.get_knowledge_node silently returned None
                # and no-opped -- but the OLD ordering here marked the row resolved first
                # regardless, so the false no-op still reported "resolved". A resolution that
                # cannot complete must FAIL LOUD, never report success, and must leave the
                # pending row genuinely retryable (still 'pending'), not silently discarded.
                applied = _apply_contest_resolution(
                    index._db,
                    old_node_id=pending_row["old_node_id"],
                    new_node_id=pending_row["new_node_id"],
                    resolution=resolution,
                )
                if not applied:
                    return (
                        f"Error: contradiction #{contradiction_id} could not be resolved -- "
                        "the candidate or existing node was not found (not yet synced to "
                        "the query index, or deleted). The pending review is UNCHANGED; "
                        "retry once the node is available."
                    )
                # candidate_wins/existing_wins both "decide" (confirmed); false_positive
                # decides nothing (dismissed) -- the shared pending_contradictions.status
                # column and every other caller of resolve_contradiction/
                # list_pending_contradictions/pending_contradiction_count stay untouched.
                underlying_status = (
                    "dismissed" if resolution == "false_positive" else "confirmed"
                )
                ok = index._db.resolve_contradiction(contradiction_id, underlying_status)
                if not ok:
                    return f"Contradiction #{contradiction_id} not found or already resolved."
                if resolution == "candidate_wins":
                    return (
                        f"Contradiction #{contradiction_id} resolved: candidate wins, "
                        "existing node superseded."
                    )
                if resolution == "existing_wins":
                    return (
                        f"Contradiction #{contradiction_id} resolved: existing node wins, "
                        "candidate retired."
                    )
                return (
                    f"Contradiction #{contradiction_id} resolved: false positive, both "
                    "nodes restored to active."
                )

            ok = index._db.resolve_contradiction(contradiction_id, resolution)
            if not ok:
                return f"Contradiction #{contradiction_id} not found or already resolved."

            if resolution == "confirmed":
                if pending_row["old_node_id"]:
                    _apply_supersession(
                        index._db,
                        old_node_id=pending_row["old_node_id"],
                        new_content=pending_row["new_content"],
                        category=pending_row["category"],
                        reason=pending_row["reason"],
                        source_sessions=json.loads(pending_row["source_sessions"]),
                        # BLOCKER 2 fix (Sentinel, 2026-07-15): carry the candidate bounds queued
                        # with this contradiction through to the materialized replacement — the
                        # queue payload only matters if confirm actually reads it.
                        valid_from=pending_row["valid_from"],
                        valid_until=pending_row["valid_until"],
                    )
                    return f"Contradiction #{contradiction_id} confirmed — old node superseded."
                # Free-text claim confirmed — create a new knowledge node
                _create_knowledge_from_claim(index._db, pending_row)
                return f"Contradiction #{contradiction_id} confirmed — new knowledge node created from claim."
            return f"Contradiction #{contradiction_id} dismissed — old node retained."

        if action == "forget":
            if not old_node_id:
                return "Error: old_node_id is required for 'forget'. Provide the knowledge node ID to archive."
            node = index._db.get_knowledge_node(old_node_id)
            if not node:
                return f"Knowledge node '{old_node_id}' not found."
            from synapt.recall.knowledge import update_node, _knowledge_path
            update_node(old_node_id, {"status": "archived"}, _knowledge_path())
            return f"Forgotten: node '{old_node_id}' archived. Content was: {node['content'][:100]}"

        if action == "correct":
            if not old_node_id:
                return "Error: old_node_id is required for 'correct'. Provide the knowledge node ID to update."
            if not claim and not new_content:
                return "Error: claim or new_content is required for 'correct'. Provide the corrected information."
            node = index._db.get_knowledge_node(old_node_id)
            if not node:
                return f"Knowledge node '{old_node_id}' not found."
            corrected = new_content or claim
            _apply_supersession(
                index._db,
                old_node_id=old_node_id,
                new_content=corrected,
                category=node.get("category", "general"),
                reason=reason or "Manual correction",
                source_sessions=[],
            )
            return f"Corrected: '{old_node_id}' superseded with: {corrected[:100]}"

        return f"Unknown action: {action}. Use 'list', 'resolve', 'flag', 'forget', or 'correct'."
    except Exception as exc:
        return f"Contradiction management failed: {exc}"
    finally:
        _invalidate_cache()


def _handle_flag(
    index,
    claim: str | None,
    new_content: str | None,
    old_node_id: str | None,
    reason: str | None,
) -> str:
    """Handle the 'flag' action for recall_contradict.

    Accepts a free-text claim and optionally searches for matching knowledge
    nodes.  If *old_node_id* is provided, uses it directly.  Otherwise,
    searches the knowledge FTS index for candidates.
    """
    if not claim and not new_content:
        return "Error: 'claim' or 'new_content' is required for 'flag' action."

    effective_claim = claim or ""
    effective_content = new_content or claim or ""
    effective_reason = reason or ""

    # If a specific node ID was provided, use it directly
    if old_node_id:
        node = index._db.get_knowledge_node(old_node_id)
        if not node:
            return f"Error: knowledge node '{old_node_id}' not found."
        cid = index._db.add_pending_contradiction(
            old_node_id=old_node_id,
            new_content=effective_content,
            reason=effective_reason,
            detected_by="manual",
            claim_text=effective_claim or None,
        )
        return (
            f"Contradiction #{cid} flagged against node '{old_node_id}': "
            f"\"{node['content'][:80]}\" → \"{effective_content[:80]}\". "
            f"Use recall_contradict(action='resolve', contradiction_id={cid}) to confirm or dismiss."
        )

    # Search for matching knowledge nodes via FTS
    search_text = effective_claim or effective_content
    try:
        fts_results = index._db.knowledge_fts_search(search_text, limit=3)
        if fts_results:
            rowids = [r[0] for r in fts_results]
            node_map = index._db.knowledge_by_rowid(rowids)
            matches = [node_map[rid] for rid in rowids if rid in node_map]
        else:
            matches = []
    except Exception:
        matches = []

    if matches:
        # Flag against the best match, mention alternatives
        best = matches[0]
        cid = index._db.add_pending_contradiction(
            old_node_id=best["id"],
            new_content=effective_content,
            reason=effective_reason,
            detected_by="manual",
            claim_text=effective_claim or None,
        )
        lines = [
            f"Contradiction #{cid} flagged against best-matching node '{best['id']}':",
            f"  Old: \"{best['content'][:100]}\"",
            f"  New: \"{effective_content[:100]}\"",
        ]
        if len(matches) > 1:
            lines.append(f"  Other candidates: {', '.join(m['id'] for m in matches[1:])}")
        lines.append(
            f"Use recall_contradict(action='resolve', contradiction_id={cid}) to confirm or dismiss."
        )
        return "\n".join(lines)

    # No matching knowledge nodes — search transcript chunks for context
    # so the agent can see what was discussed and extract properly.
    cid = index._db.add_pending_contradiction(
        old_node_id=None,
        new_content=effective_content,
        reason=effective_reason,
        detected_by="manual",
        claim_text=effective_claim or None,
    )
    lines = [
        f"Contradiction #{cid} flagged as free-text claim (no matching knowledge node found).",
        f"  Claim: \"{effective_content[:100]}\"",
        f"  When confirmed, a new knowledge node will be created.",
    ]

    # Search transcripts for relevant context to help the agent extract
    try:
        transcript_results = index.lookup(
            search_text,
            max_chunks=3,
            max_tokens=500,
            depth="concise",
        )
        if transcript_results:
            lines.append("")
            lines.append("Related transcript context (for extraction):")
            lines.append(transcript_results)
    except Exception:
        pass  # Transcript search is best-effort

    lines.append(
        f"\nUse recall_contradict(action='resolve', contradiction_id={cid}) to confirm or dismiss."
    )
    return "\n".join(lines)


def _create_knowledge_from_claim(db, pending_row) -> None:
    """Create a new knowledge node from a confirmed free-text claim."""
    from synapt.recall.knowledge import KnowledgeNode

    content = pending_row["new_content"]
    category = pending_row["category"] or "workflow"
    node = KnowledgeNode.create(content, category)
    db.upsert_knowledge_node(node.__dict__)


def _apply_supersession(
    db,
    old_node_id: str,
    new_content: str,
    category: str,
    reason: str,
    source_sessions: list[str],
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> None:
    """Execute a confirmed supersession: mark old node, create replacement.

    *valid_from*/*valid_until* are the queued contradiction's candidate bounds (BLOCKER 2 fix,
    Sentinel 2026-07-15) — the candidate's bound is preferred over the confirm-time ``now()``
    default, mirroring consolidate.py's create/legacy-contradict preference. Without this, a
    bound that survived all the way to the queue was still lost at the final materialization
    step — the exact "fruit-to-the-dict is not fruit-to-the-database" failure mode.

    Both are re-validated HERE (adversarial verification finding, 2026-07-15) — every other
    bound-consuming site in this feature validates via ``_validate_iso_date`` before a dict
    update or DB write; this was the one that didn't, so a non-string shape reaching this
    function directly (today unreachable via the live MCP surface — every current caller
    validates upstream — but a real landmine for any future caller that doesn't) would raise
    ``sqlite3.ProgrammingError`` on the SECOND of two non-atomic upserts, leaving the old node
    marked contradicted with ``superseded_by`` pointing at a replacement that was never created,
    permanently (the confirm status flip already committed, so there is no retry path).
    """
    from synapt.recall.consolidate import _validate_iso_date
    valid_from = _validate_iso_date(valid_from)
    valid_until = _validate_iso_date(valid_until)
    now = datetime.now(timezone.utc).isoformat()

    old_node = db.get_knowledge_node(old_node_id)
    if old_node is None:
        return

    lineage_id = old_node.get("lineage_id", "") or old_node["id"]
    old_version = old_node.get("version", 1)

    # Mark old node as contradicted — also backfill lineage_id if bootstrapping
    old_node["lineage_id"] = lineage_id
    old_node["status"] = "contradicted"
    old_node["valid_until"] = now
    old_node["contradiction_note"] = reason
    old_node["updated_at"] = now
    new_id = uuid.uuid4().hex[:12]
    old_node["superseded_by"] = new_id
    db.upsert_knowledge_node(old_node)

    # Create replacement node
    new_node = {
        "id": new_id,
        "content": _tw(new_content, 300),
        "category": category or old_node.get("category", "workflow"),
        "confidence": old_node.get("confidence", 0.5),
        "source_sessions": source_sessions or old_node.get("source_sessions", []),
        "created_at": now,
        "updated_at": now,
        "status": "active",
        "superseded_by": "",
        "contradiction_note": "",
        "tags": old_node.get("tags", []),
        "valid_from": valid_from or now,
        "valid_until": valid_until,
        "version": old_version + 1,
        "lineage_id": lineage_id,
    }
    db.upsert_knowledge_node(new_node)


def _apply_contest_resolution(
    db,
    old_node_id: str,
    new_node_id: str,
    resolution: str,
) -> bool:
    """Execute a resolved Fix B contest: promote/restore/retire the two ALREADY-MATERIALIZED
    nodes from a contested pair (internal design spec, section 10.4).

    Returns ``True`` if the mutation actually happened, ``False`` otherwise (recall#905,
    0.17.0 blocker, Opus 2026-07-22 -- Part 2, co-primary). The caller MUST check this before
    marking the pending row resolved: a resolution that cannot complete must fail loud, never
    report success. ``None`` was the original return type — always, whether the mutation
    happened or silently no-opped — which is exactly how the caller's old ordering (mark
    resolved, then attempt the mutation) produced a false "resolved" success message for a
    contest whose candidate node hadn't been synced to SQLite yet.

    Deliberately separate from ``_apply_supersession`` — that function always CREATES a fresh
    node from queued text; a contest row already has both nodes persisted (the candidate was
    materialized at contest time, section 10.5), so this function only PROMOTES, RESTORES, or
    RETIRES existing rows. Entangling the two risked pulling this new mutation shape into
    ``_apply_supersession``'s own hardened, already-subtle bound-carrying logic (its docstring
    documents a real non-atomic-upsert landmine) that this shape doesn't need at all.

    *resolution* is one of "candidate_wins" / "existing_wins" / "false_positive" — the 3-way
    vocabulary contest resolution needs that the underlying confirmed/dismissed
    ``pending_contradictions.status`` column does not carry (section 10.4's deliberate
    layering: the caller maps this onto confirmed/dismissed for the shared column, and calls
    this function separately for the node-level mutation).

    Confidence is RECOMPUTED via ``compute_confidence`` on every node returning to active
    status, not preserved from its pre-contest value — simpler, avoids two more schema
    columns, matches how confidence is computed everywhere else a node returns to active
    (section 10.4's deliberate simplification, not an oversight).

    No-ops (mutates neither store, returns ``False``) if either node is missing -- but this
    is no longer SILENT the way it originally was (recall#905): the caller surfaces the
    ``False`` return as a loud, honest error and leaves the pending row unresolved, rather
    than treating a missing node the way ``_apply_supersession`` treats a deleted one (quiet
    tolerance is correct there because that path never claims a mutation succeeded when it
    didn't; this path's caller used to make exactly that false claim).

    DUAL-STORE (Sentinel r2, PR#903 issuecomment-5037168639 -- blocking): persists to BOTH
    SQLite (``db.upsert_knowledge_node``, immediate query-path correctness) AND
    ``knowledge.jsonl`` (``update_node``, the consolidation source-of-truth). The first
    version of this function wrote SQLite only, matching ``_apply_supersession``'s own
    SQLite-only shape -- but ``_apply_supersession`` isn't consolidation's own write path
    the way contest resolution effectively is: ``_sync_knowledge_to_db`` treats
    ``knowledge.jsonl`` as authoritative and does an unconditional ``INSERT OR REPLACE``
    from it into SQLite for every node present. Writing SQLite alone meant the very next
    sync silently reverted every valid resolution back to both-contested — deterministic,
    not crash-dependent (Opus's r1 under-rated this as a rare crash-mid-sequence risk;
    Sentinel's r2 found it fires on any ordinary sync). The update dicts below are computed
    once per branch and applied identically to both stores so the two writes can't drift
    apart from each other.
    """
    from synapt.recall.consolidate import compute_confidence
    from synapt.recall.knowledge import update_node, _knowledge_path

    candidate = db.get_knowledge_node(new_node_id)
    existing = db.get_knowledge_node(old_node_id)
    if candidate is None or existing is None:
        return False

    now = datetime.now(timezone.utc).isoformat()

    if resolution == "candidate_wins":
        candidate_updates = {
            "status": "active",
            "confidence": compute_confidence(len(candidate.get("source_sessions", []))),
            "updated_at": now,
        }
        existing_updates = {
            "status": "superseded", "superseded_by": new_node_id, "updated_at": now,
        }
    elif resolution == "existing_wins":
        existing_updates = {
            "status": "active",
            "confidence": compute_confidence(len(existing.get("source_sessions", []))),
            "updated_at": now,
        }
        candidate_updates = {"status": "stale", "updated_at": now}
    elif resolution == "false_positive":
        candidate_updates = {
            "status": "active",
            "confidence": compute_confidence(len(candidate.get("source_sessions", []))),
            "updated_at": now,
        }
        existing_updates = {
            "status": "active",
            "confidence": compute_confidence(len(existing.get("source_sessions", []))),
            "updated_at": now,
        }
    else:
        return False  # unknown resolution -- caller already validated, defensive only

    candidate.update(candidate_updates)
    existing.update(existing_updates)
    db.upsert_knowledge_node(candidate)
    db.upsert_knowledge_node(existing)

    kn_path = _knowledge_path()
    update_node(new_node_id, candidate_updates, kn_path)
    update_node(old_node_id, existing_updates, kn_path)
    return True


def format_contradictions_for_session_start() -> str:
    """Format pending contradictions for the SessionStart hook.

    Returns a string to print to stdout (which becomes the system-reminder
    the model sees). The model should then ask the user about each one.
    Returns empty string if no pending contradictions.

    Uses a lightweight DB connection instead of loading the full index,
    since only SQL queries are needed (no chunk/embedding data).  Fixes #119.
    """
    from synapt.recall.storage import RecallDB

    db_path = project_index_dir() / "recall.db"
    if not db_path.exists():
        return ""

    try:
        db = RecallDB(db_path)
        pending = db.list_pending_contradictions()
        if not pending:
            db.close()
            return ""
        # Build node lookup for old content
        old_ids = {c["old_node_id"] for c in pending}
        node_lookup = {
            nid: db.get_knowledge_node(nid)
            for nid in old_ids
        }
        db.close()
        lines = [f"Pending contradictions ({len(pending)}) — ask the user to resolve:"]
        for c in pending:
            old_node = node_lookup.get(c["old_node_id"])
            old_content = old_node["content"][:120] if old_node else "(deleted node)"
            new_content = c["new_content"][:120]
            lines.append(f"  #{c['id']}: \"{old_content}\"")
            lines.append(f"    -> \"{new_content}\"")
            if c["reason"]:
                lines.append(f"    Reason: {c['reason']}")
        lines.append("Use recall_contradict to confirm or dismiss each one based on user input.")
        return "\n".join(lines)
    except Exception:
        return ""


def recall_correct(
    question: str,
    wrong_answer: str,
    correct_answer: str,
    category: str = "",
) -> str:
    """Capture a user correction as benchmark data and update knowledge.

    Call this when a user corrects a wrong answer from recall. It does
    three things in one call:

    1. Logs the correction to `.synapt/recall/corrections.jsonl` for
       benchmark use (question + wrong + correct + category + timestamp)
    2. Immediately creates a high-confidence knowledge node with the
       correct answer (no contradiction queue — corrections are pre-confirmed)
    3. Supersedes any existing knowledge node containing the wrong answer

    Args:
        question: The question that was answered incorrectly.
        wrong_answer: The incorrect answer that was given.
        correct_answer: The correct/updated answer from the user.
        category: Optional category (e.g., "convention", "factual",
                  "temporal", "debug", "architecture").
    """
    from synapt.recall.corrections import log_correction

    try:
        # Step 1: Log the correction for benchmark data
        path = log_correction(
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            category=category,
        )

        # Step 2: Immediately create a knowledge node (bypass contradiction queue)
        # User corrections are already confirmed — no need for a second confirmation step.
        from synapt.recall.knowledge import KnowledgeNode, append_node
        node_content = f"{correct_answer} (re: {question})"
        node_category = category if category else "fact"
        node = KnowledgeNode.create(
            content=node_content,
            category=node_category,
            confidence=0.9,  # High confidence — human-verified correction
        )
        kn_path = append_node(node)

        # Step 3: Sync to DB so the node is immediately searchable
        try:
            from synapt.recall.consolidate import _sync_knowledge_to_db
            from synapt.recall.core import project_data_dir
            project_dir = project_data_dir()
            _sync_knowledge_to_db(project_dir, kn_path)
            synced = "  Synced to search index."
        except Exception:
            synced = "  (Will sync on next consolidation.)"

        # Step 4: Search for and supersede any matching wrong knowledge node
        supersede_note = ""
        try:
            from synapt.recall.knowledge import read_nodes, update_node
            existing = read_nodes(kn_path, status="active")
            for existing_node in existing:
                if existing_node.id == node.id:
                    continue
                # Check if existing node matches the wrong answer
                # Require minimum length to avoid overly broad substring matches
                if len(wrong_answer) >= 5 and wrong_answer.lower() in existing_node.content.lower():
                    update_node(
                        existing_node.id,
                        {
                            "status": "contradicted",
                            "superseded_by": node.id,
                            "contradiction_note": f"Corrected by user: {correct_answer}",
                        },
                        kn_path,
                    )
                    supersede_note = f"\n  Superseded old node: {existing_node.content[:60]}"
                    break
        except Exception:
            pass

        return (
            f"Correction captured:\n"
            f"  Q: {question}\n"
            f"  Wrong: {wrong_answer}\n"
            f"  Correct: {correct_answer}\n"
            f"  Logged to: {path.name}\n"
            f"  Knowledge node created: {node.id}\n"
            f"{synced}{supersede_note}"
        )
    except Exception as exc:
        return f"Failed to capture correction: {exc}"


def recall_context(
    chunk_id: str | None = None,
    cluster_id: str | None = None,
) -> str:
    """Drill down into a search result to see full raw transcript content.

    Two modes:
    - chunk_id: Show full raw transcript for a single turn (e.g., "a1b2c3d4:t5")
    - cluster_id: Show all chunks in a topic cluster (e.g., "clust-abcd1234")

    If both are provided, cluster_id takes precedence.

    Use after recall_search finds a relevant turn/cluster but you need
    the complete detail.

    Args:
        chunk_id: The chunk identifier from a search result.
        cluster_id: A cluster identifier to show all member chunks.
    """
    try:
        idx = _get_index()
        if idx is None:
            return "No index found. Run recall_build first."

        if cluster_id:
            if cluster_id.startswith("tl-"):
                return (
                    f"Timeline arc {cluster_id} is a session-level grouping, "
                    "not a chunk cluster. Use `recall_timeline` to view it."
                )
            result = _format_cluster_context(idx, cluster_id)
            if "not found" not in result:
                _record_context_access(idx, "cluster", cluster_id)
                return result
            return _label_empty_result(result, _resolved_index_dir())
        if chunk_id:
            result = idx.read_turn_context(chunk_id)
            if result:
                _record_context_access(idx, "chunk", chunk_id)
                return result
            return _label_empty_result(
                f"Chunk {chunk_id} not found.", _resolved_index_dir()
            )
        return "Provide either chunk_id or cluster_id."
    except Exception as exc:
        return f"Error reading context: {exc}"


def _record_context_access(idx, item_type: str, item_id: str) -> None:
    """Record an explicit drill-down access and check promotions (fire-and-forget)."""
    try:
        if idx._db:
            idx._db.record_access(
                [{"item_type": item_type, "item_id": item_id, "score": 1.0}],
                context="context",
            )
            from synapt.recall.promotion import (
                check_promotions, execute_cheap_promotions,
            )
            actions = check_promotions(idx._db, item_type, item_id)
            if actions:
                execute_cheap_promotions(idx._db, item_type, item_id, actions)
    except Exception:
        pass


def _format_cluster_context(idx, cluster_id: str) -> str:
    """Format all chunks in a cluster for drill-down context."""
    db = idx._db
    if db is None:
        return "No database available for cluster lookup."

    chunk_ids = db.get_cluster_chunks(cluster_id)
    if not chunk_ids:
        return f"Cluster {cluster_id} not found or has no chunks."

    cluster_info = db.get_cluster(cluster_id)
    topic = cluster_info["topic"] if cluster_info else "unknown"

    parts = [f"Cluster: {topic} ({len(chunk_ids)} chunks)\n"]
    for cid in chunk_ids:
        context = idx.read_turn_context(cid)
        parts.append(context)
        parts.append("")  # blank line separator

    return "\n".join(parts)


def recall_journal(
    action: str = "read",
    focus: str | None = None,
    done: str | None = None,
    decisions: str | None = None,
    next_steps: str | None = None,
) -> str:
    """Read or write session journal entries.

    The journal tracks what happened each session: focus, done items,
    decisions, and next steps. Use "write" at end of session to persist
    context for next time. Use "read" to see the latest entry.

    Args:
        action: "read" (latest entry), "write" (create entry), "list" (recent entries),
            or "pending" (unresolved carry-forward next steps only).
        focus: What this session was about (write only).
        done: Accomplishments, ONE PER LINE (write only).
        decisions: Key decisions, ONE PER LINE (write only).
        next_steps: Next steps, ONE PER LINE (write only).

    Item separation: these three fields split on NEWLINES, so a semicolon inside
    a multi-line field is ordinary punctuation and the sentence stays whole. If a
    field contains NO newline it falls back to splitting on semicolons, which
    keeps older single-line callers working -- so a semicolon inside a
    single-line item WILL still split it. Write one item per line and this never
    bites you.

    Ordering: the session-start read is TRUNCATED, and it leads with next_steps.
    Put what is unresolved there; completed work is recoverable from version
    control and the tracker, an unrecorded open question is not.
    """
    try:
        from synapt.recall.journal import (
            append_entry,
            auto_extract_entry,
            format_entry_full,
            format_for_session_start,
            format_write_confirmation,
            latest_transcript_path,
            merge_carried_forward_next_steps,
            pending_next_steps,
            read_entries,
            read_latest,
            read_previous_meaningful,
            split_journal_field,
        )

        if action == "read":
            entry = read_latest(meaningful=True)
            if not entry:
                return "No journal entries yet."
            return format_for_session_start(entry)

        if action == "pending":
            items = pending_next_steps()
            if not items:
                return "No pending next steps."
            lines = ["Pending next steps:"]
            for item in items:
                lines.append(f"  - {item}")
            return "\n".join(lines)

        if action == "list":
            entries = read_entries(n=5)
            if not entries:
                return "No journal entries yet."
            return "\n\n---\n\n".join(format_entry_full(e) for e in entries)

        if action == "write":
            project = Path.cwd().resolve()
            transcript_path = latest_transcript_path(project)
            entry = auto_extract_entry(transcript_path=transcript_path, cwd=str(project))
            previous_entry = read_previous_meaningful(entry.session_id)

            if focus:
                entry.focus = focus
            if done:
                entry.done = split_journal_field(done)
            if decisions:
                entry.decisions = split_journal_field(decisions)
            explicit_next_steps = list(entry.next_steps)
            if next_steps:
                entry.next_steps = split_journal_field(next_steps)
                explicit_next_steps = list(entry.next_steps)
            entry.next_steps = merge_carried_forward_next_steps(
                entry.next_steps,
                entry.done,
                previous_entry,
            )

            # Clear auto flag when user provides rich content
            if entry.has_rich_content():
                entry.auto = False

            # Skip auto entries with no rich content
            if entry.auto and not entry.has_rich_content():
                return "No rich content to journal (auto-extract only)."
            if not entry.has_content():
                return "No content to journal (no files modified, no fields provided)."

            append_entry(entry)
            return (
                "Journal entry written.\n\n"
                f"{format_write_confirmation(entry, explicit_next_steps)}"
            )

        return f"Unknown action: {action}. Use 'read', 'write', 'list', or 'pending'."
    except Exception as exc:
        return f"Journal failed: {exc}"


def recall_save(
    content: str = "",
    category: str = "workflow",
    confidence: float = 0.8,
    tags: list[str] | None = None,
    source_sessions: list[str] | None = None,
    source_turns: list[str] | None = None,
    node_id: str | None = None,
    retract: bool = False,
) -> str:
    """Create, update, or retract a knowledge node.

    Args:
        content: Durable fact, convention, or decision to save.
            Required for create/update, ignored for retract.
        category: Knowledge category (workflow, tooling, decision, etc.).
        confidence: Confidence score from 0.0 to 1.0.
        tags: Optional search tags.
        source_sessions: Optional originating session IDs.
        source_turns: Optional originating turn refs ("session_id:turn_num").
        node_id: Optional stable knowledge-node ID to upsert. If omitted,
            recall_save derives a stable ID from the saved content.
        retract: If True, mark the node as retracted (hidden from search
            but preserved for audit). Requires node_id.
    """
    try:
        import hashlib
        from datetime import datetime, timezone

        from synapt.recall.knowledge import KnowledgeNode, append_node
        from synapt.recall.storage import RecallDB

        project = Path.cwd().resolve()
        db = RecallDB(project_index_dir(project) / "recall.db")
        try:
            # --- Retract path ---
            if retract:
                if not node_id:
                    return "Error: node_id is required for retract."
                existing = db.get_knowledge_node(node_id)
                if not existing:
                    return f"Error: node {node_id} not found."
                now = datetime.now(timezone.utc).isoformat()
                existing["status"] = "retracted"
                existing["valid_until"] = now
                existing["updated_at"] = now
                db.upsert_knowledge_node(existing)
                # db.close() handled by finally below
                _invalidate_cache()
                return f"Knowledge node retracted: {node_id}. Hidden from search, preserved for audit."

            # --- Create/update path ---
            clean_content = (content or "").strip()
            if not clean_content:
                return "Error: content is required."

            resolved_node_id = node_id or hashlib.sha1(
                clean_content.encode("utf-8")
            ).hexdigest()[:12]
            existing = db.get_knowledge_node(resolved_node_id)
            node = KnowledgeNode.create(
                content=clean_content,
                category=category,
                source_sessions=[s for s in (source_sessions or []) if s],
                confidence=confidence,
                tags=[t for t in (tags or []) if t],
                source_turns=[t for t in (source_turns or []) if t],
                node_id=resolved_node_id,
            )
            if existing:
                node.created_at = existing.get("created_at", node.created_at)
                node.version = existing.get("version", 1) + 1
                node.lineage_id = existing.get("lineage_id", "") or existing["id"]
            append_node(node, project_data_dir(project) / "knowledge.jsonl")
            db.upsert_knowledge_node(node.to_dict())

            embedded = False
            provider = get_embedding_provider()
            if provider:
                rowid = db.get_knowledge_rowid(node.id)
                if rowid is not None:
                    embedding = provider.embed_single(node.content[:500])
                    db.save_knowledge_embeddings({rowid: embedding})
                    embedded = True
        finally:
            db.close()

        _invalidate_cache()
        action = "updated" if existing else "saved"
        emb_status = "embedded for vector search" if embedded else "saved without embeddings"
        version_tag = f", v{node.version}" if node.version > 1 else ""
        return (
            f"Knowledge node {action}: {node.id} ({node.category}, "
            f"confidence={node.confidence:.2f}{version_tag}). {emb_status}."
        )
    except Exception as exc:
        return f"Knowledge save failed: {exc}"


def recall_sync_memory() -> str:
    """Sync Claude Code MEMORY.md files into recall as knowledge nodes.

    Scans ~/.claude/projects/*/memory/*.md, parses YAML frontmatter
    (name, description, type), and upserts each as a knowledge node via
    recall_save. Skips files that haven't changed since last sync.

    Memory types map to knowledge categories:
    - user → user
    - feedback → feedback
    - project → project
    - reference → reference
    """
    import hashlib
    import yaml
    from pathlib import Path

    memory_root = Path.home() / ".claude" / "projects"
    if not memory_root.exists():
        return "No Claude Code memory directory found."

    synced = 0
    skipped = 0
    errors = 0

    for memory_dir in sorted(memory_root.glob("*/memory")):
        for md_file in sorted(memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue  # Skip the index file

            try:
                text = md_file.read_text(encoding="utf-8")

                # Parse YAML frontmatter
                if not text.startswith("---"):
                    skipped += 1
                    continue
                parts = text.split("---", 2)
                if len(parts) < 3:
                    skipped += 1
                    continue

                frontmatter = yaml.safe_load(parts[1])
                if not isinstance(frontmatter, dict):
                    skipped += 1
                    continue

                name = frontmatter.get("name", md_file.stem)
                description = frontmatter.get("description", "")
                mem_type = frontmatter.get("type", "project")
                body = parts[2].strip()

                # Build content: description + body
                content = f"{name}: {description}" if description else name
                if body:
                    content += f"\n\n{body}"

                # Map memory type to category
                category = mem_type if mem_type in ("user", "feedback", "project", "reference") else "project"
                stable_id = hashlib.sha1(
                    str(md_file.resolve()).encode("utf-8")
                ).hexdigest()[:12]

                result = recall_save(
                    content=content,
                    category=category,
                    confidence=0.9,
                    tags=["memory.md", f"type:{mem_type}", f"name:{name}"],
                    node_id=stable_id,
                )

                if "saved" in result.lower() or "Knowledge node saved" in result:
                    synced += 1
                else:
                    errors += 1

            except Exception:
                errors += 1

    return (
        f"Memory sync complete: {synced} synced, {skipped} skipped, {errors} errors. "
        f"Scanned {memory_root}."
    )


def recall_remind(
    action: str = "add",
    text: str | None = None,
    reminder_id: str | None = None,
    sticky: bool = False,
) -> str:
    """Manage session reminders — lightweight nudges surfaced at session start.

    Reminders auto-clear after being shown once unless marked sticky.
    Use this to flag things to bring up next session.

    Args:
        action: "add" (create reminder), "list" (show all), "clear" (remove by id or all), "pending" (show pending).
        text: Reminder text (required for "add").
        reminder_id: Reminder ID (optional for "clear" — clears all if omitted).
        sticky: If true, reminder persists across sessions instead of auto-clearing.
    """
    try:
        from synapt.recall.reminders import (
            add_reminder,
            clear_reminder,
            load_reminders,
            pop_pending,
            format_for_session_start,
        )

        if action == "add":
            if not text:
                return "Error: text is required for 'add' action."
            reminder = add_reminder(text, sticky=sticky)
            sticky_label = " (sticky)" if sticky else ""
            return f"Added reminder{sticky_label}: {reminder.text} (id: {reminder.id})"

        if action == "list":
            reminders = load_reminders()
            if not reminders:
                return "No reminders."
            lines = []
            for r in reminders:
                s = " [sticky]" if r.sticky else ""
                shown = f" (shown {r.shown_count}x)" if r.shown_count > 0 else ""
                lines.append(f"  {r.id}  {r.text}{s}{shown}")
            return "\n".join(lines)

        if action == "clear":
            count = clear_reminder(reminder_id)
            return f"Cleared {count} reminder(s)." if count else "No reminders to clear."

        if action == "pending":
            pending = pop_pending()  # Single load-save cycle
            if not pending:
                return "No pending reminders."
            return format_for_session_start(pending)

        return f"Unknown action: {action}. Use 'add', 'list', 'clear', or 'pending'."
    except Exception as exc:
        return f"Reminder failed: {exc}"


def recall_enrich(
    model: str = "",
    max_entries: int = 10,
    dry_run: bool = False,
    adapter_path: str = "",
) -> str:
    """Enrich auto-generated journal stubs using a local MLX model.

    Reads journal entries tagged as auto-generated that lack rich content
    (done/decisions/next_steps), loads the original transcript, and uses
    MLX to extract structured information. Appends enriched entries to
    journal.jsonl.

    Requires mlx-lm to be installed (pip install mlx-lm). Safe to run
    multiple times — already-enriched entries are skipped.

    Args:
        model: MLX model to use for summarization (default: Ministral-3-3B-Instruct-2512-4bit).
        max_entries: Maximum entries to enrich per invocation.
        dry_run: If True, report what would be enriched without modifying anything.
        adapter_path: Optional LoRA adapter path for enrichment.
    """
    from synapt.recall.enrich import enrich_all, _MLX_AVAILABLE, _INSTALL_MSG

    if not _MLX_AVAILABLE:
        return _INSTALL_MSG

    project = Path.cwd().resolve()
    kwargs: dict = {"project_dir": project, "dry_run": dry_run, "max_entries": max_entries}
    if model:
        kwargs["model"] = model
    if adapter_path:
        kwargs["adapter_path"] = adapter_path
    try:
        count = enrich_all(**kwargs)
    except Exception as exc:
        return f"Enrichment failed: {exc}"
    finally:
        _invalidate_cache()

    if dry_run:
        return f"Dry run: {count} entries would be enriched."
    if count:
        return f"Enriched {count} journal entries. Run `recall_build` to re-index."
    return "No entries to enrich (all sessions already have journal entries)."


def recall_timeline(
    query: str = "",
    after: str | None = None,
    before: str | None = None,
    branch: str | None = None,
    max_results: int = 10,
) -> str:
    """View chronological timeline of work arcs.

    Returns session arcs — groups of consecutive sessions on the same
    branch/topic — ordered chronologically. Each arc shows date range,
    branch, sessions, and key accomplishments from journal entries.

    Args:
        query: Optional text query to filter arcs via FTS.
        after: Only arcs ending after this date (ISO 8601).
        before: Only arcs starting before this date (ISO 8601).
        branch: Filter to arcs on a specific branch.
        max_results: Maximum number of arcs to return (default 10).
    """
    index = _get_index()
    if index is None:
        return "No index found. Run `recall_build` first."
    if not index._db:
        return "No database found. Run `recall_build` to create one."

    try:
        db = index._db

        if query:
            # FTS search filtered to timeline type + date/branch filters
            from synapt.recall.storage import _escape_fts_query

            escaped = _escape_fts_query(query, use_or=True)
            if not escaped:
                return "No valid search terms in query."
            sql = (
                "SELECT c.* FROM clusters c "
                "JOIN clusters_fts f ON c.id = f.rowid "
                "WHERE clusters_fts MATCH ? "
                "AND c.cluster_type = 'timeline' AND c.status = 'active'"
            )
            params: list[str] = [escaped]
            if after:
                sql += " AND c.date_end >= ?"
                params.append(after)
            if before:
                sql += " AND c.date_start <= ?"
                params.append(before)
            if branch:
                sql += " AND c.branch = ?"
                params.append(branch)
            sql += " ORDER BY c.date_start"
            rows = db._conn.execute(sql, params).fetchall()
            arcs = [
                {
                    "cluster_id": r["cluster_id"],
                    "topic": r["topic"],
                    "session_ids": json.loads(r["session_ids"]),
                    "branch": r["branch"],
                    "date_start": r["date_start"],
                    "date_end": r["date_end"],
                    "chunk_count": r["chunk_count"],
                    "tags": json.loads(r["tags"]) if r["tags"] else [],
                }
                for r in rows
            ]
        else:
            arcs = db.load_timeline_clusters(
                after=after, before=before, branch=branch
            )

        if not arcs:
            return "No timeline arcs found."

        arcs = arcs[:max_results]

        # Load journal entries from ALL worktrees, prefer enriched
        from synapt.recall.journal import (
            _journal_path, _read_all_entries, _dedup_entries,
        )
        from synapt.recall.core import all_worktree_archive_dirs
        from synapt.recall.scrub import strip_system_artifacts

        all_jentries: list = []
        local_jp = _journal_path()
        if local_jp.exists():
            all_jentries.extend(_read_all_entries(local_jp))
        for wt_archive in all_worktree_archive_dirs():
            wt_jp = wt_archive.parent / "journal.jsonl"
            if wt_jp.exists() and wt_jp.resolve() != local_jp.resolve():
                all_jentries.extend(_read_all_entries(wt_jp))
        j_entries = _dedup_entries(all_jentries)

        # _dedup_entries guarantees one entry per session_id (richest wins)
        journal_by_session = {}
        for e in j_entries:
            if e.session_id:
                journal_by_session[e.session_id] = e

        lines: list[str] = []
        for arc in arcs:
            # Header
            topic = arc.get("topic", "unknown")
            ds = arc["date_start"][:10] if arc["date_start"] else "?"
            de = arc["date_end"][:10] if arc["date_end"] else "?"
            n_sessions = len(arc.get("session_ids", []))
            header = f"=== {topic} ({ds} -- {de}, {n_sessions} session(s)) ==="
            lines.append(header)

            # Tags
            tags = arc.get("tags", [])
            if tags:
                lines.append(f"Tags: {', '.join(tags)}")

            # Session details
            for sid in arc.get("session_ids", []):
                entry = journal_by_session.get(sid)
                if entry:
                    date = entry.timestamp[:10] if entry.timestamp else "?"
                    raw_focus = strip_system_artifacts(entry.focus) if entry.focus else ""
                    focus = raw_focus[:80] if raw_focus else "(no focus)"
                    lines.append(f"  {date} {sid[:8]}  {focus}")
                else:
                    lines.append(f"  {sid[:8]}  (no journal entry)")
            lines.append("")

        return "\n".join(lines).rstrip()
    except Exception as exc:
        return f"Timeline query failed: {exc}"


def recall_channel(
    action: str = "read",
    channel: str = "dev",
    message: str | None = None,
    to: str | None = None,
    target: str | None = None,
    limit: int = 20,
    pin: bool = False,
    name: str | None = None,
    attachments: str | None = None,
    show_pins: bool = True,
    detail: str = "medium",
    msg_type: str | None = None,
) -> str:
    """Cross-worktree communication channels for multi-agent coordination.

    Channels are append-only JSONL files in the shared .synapt/recall/ directory.
    Any agent (worktree) can post and read messages. No daemon needed.
    State (presence, cursors, pins) is stored in SQLite.

    Args:
        action: "join", "leave", "post", "read", "read_message", "who", "heartbeat", "unread",
                "pin", "directive", "mute", "unmute", "kick", "broadcast",
                "list", "search", "rename", "claim", "unclaim", "intent", "board".
        channel: Channel name (default "dev"). Any name works -- created on first post.
        message: Message body (required for "post", "directive", "broadcast") or message_id for "read_message"/pin actions.
        to: Target agent for "directive" action.
        target: Agent to mute/unmute/kick (agent_id, display name, or griptree name).
        limit: Max messages to return for "read" action (default 20).
        pin: If True with "post" action, also pin the message.
        name: Display name for this agent (set on join, shown in messages instead of agent ID).
        attachments: Semicolon-separated file paths to attach (copied into channel store on post).
        msg_type: Message type for "post" (status, claim, pr, code, message) or filter for "read".
            Default "message". On read, only messages matching the type are returned.
        show_pins: If False with "read" action, omit pinned messages from output (default True).
            Deprecated — use detail instead.
        detail: Output verbosity level. Controls pins, metadata, and truncation.
            "max"    — all pins, full messages, all metadata (IDs, claims, attachments)
            "high"   — all pins, full messages, message IDs only
            "medium" — full messages, IDs, claims, attachments; pins follow show_pins (default for "read")
            "low"    — no pins, truncated messages (200 chars), with refs for truncated messages
            "min"    — no pins, one-line per message, skip join/leave noise
            Use "low" or "min" for monitoring loops to save context budget.

    Coordination actions:
        claim: Claim a message/task by message_id (prevents duplicate work).
        unclaim: Release a previously claimed message_id.
        intent: Declare intent to create something (message = description of planned work).
    """
    try:
        from synapt.recall.actions import get_action_registry

        registry = get_action_registry()
        return registry.dispatch(
            action,
            channel=channel,
            message=message,
            to=to,
            target=target,
            limit=limit,
            pin=pin,
            name=name,
            attachments=attachments,
            show_pins=show_pins,
            detail=detail,
            msg_type=msg_type,
        )
    except Exception as exc:
        return f"Channel failed: {exc}"


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------


def _check_version_stale() -> str:
    """Check if installed synapt version differs from what this process loaded.

    Returns a warning string if stale, empty string if current.
    """
    try:
        from importlib.metadata import version as pkg_version
        installed = pkg_version("synapt")
        if installed != _STARTUP_VERSION:
            return (
                f"[synapt] Server running v{_STARTUP_VERSION} but v{installed} is installed. "
                f"Restart the MCP server to pick up changes (/mcp or kill the synapt server process)."
            )
    except Exception:
        pass
    return ""


def _check_channel_activity() -> str:
    """Lightweight check for new channel messages since last tool call.

    Uses file mtime comparison instead of reading full JSONL — ~1ms.
    Returns a notification string if new messages exist, empty string otherwise.
    Checks all channel files, not just #dev.
    """
    try:
        from synapt.recall.core import project_data_dir
        channels_dir = project_data_dir() / "channels"
        if not channels_dir.exists():
            return ""

        # Check mtime of all channel JSONL files
        channel_files = sorted(channels_dir.glob("*.jsonl"))
        if not channel_files:
            return ""

        marker = channels_dir / ".last_seen_mtime"
        max_mtime = max(f.stat().st_mtime for f in channel_files)

        if marker.exists():
            last_mtime = float(marker.read_text().strip())
            if max_mtime <= last_mtime:
                return ""  # No new messages

        # New messages detected — count BEFORE updating marker
        from synapt.recall.channel import channel_unread
        counts = channel_unread()
        if counts:
            total = sum(counts.values())
            if total > 0:
                # Update marker only AFTER successful read
                marker.write_text(str(max_mtime))
                channels = ", ".join(f"#{ch}: {n}" for ch, n in sorted(counts.items()) if n > 0)
                return f"[channel] {total} new message(s): {channels}. Use recall_channel(action='read') to see them."

        # No unread messages — still update marker to avoid re-checking
        marker.write_text(str(max_mtime))
    except Exception:
        pass
    return ""


_solo_mode_until: float = 0.0  # monotonic timestamp when solo check expires


def _is_solo_mode() -> bool:
    """Check if we're the only agent (no channel files exist).

    Cached for 60 seconds — avoids re-checking the filesystem on every
    tool call. In solo sessions, this saves ~19ms per call by skipping
    the directive and channel checks entirely.
    """
    import time as _time
    global _solo_mode_until
    now = _time.monotonic()
    if now < _solo_mode_until:
        return True
    try:
        from synapt.recall.core import project_data_dir
        channels_dir = project_data_dir() / "channels"
        if not channels_dir.exists() or not any(channels_dir.glob("*.jsonl")):
            _solo_mode_until = now + 60.0
            return True
    except Exception:
        pass
    _solo_mode_until = 0.0
    return False


def _directive_suffix() -> str:
    """Check for pending directives, @mentions, channel activity, and version.

    Returns empty string if nothing pending. In solo mode (no channel files),
    skips directive and channel checks entirely (~0ms instead of ~20ms).
    """
    parts = []

    # Skip channel checks in solo mode (#436)
    if not _is_solo_mode():
        try:
            from synapt.recall.channel import check_directives
            result = check_directives()
            if result:
                parts.append(result)
        except Exception:
            pass

        # Lightweight channel activity check (~1ms)
        activity = _check_channel_activity()
        if activity:
            parts.append(activity)

    stale = _check_version_stale()
    if stale:
        parts.append(stale)
    return "\n\n".join(parts)


def _with_directive_check(fn):
    """Wrap a tool function to append pending directives to its result."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        suffix = _directive_suffix()
        if suffix and isinstance(result, str):
            return result + "\n\n" + suffix
        return result

    return wrapper


def recall_reload() -> str:
    """Restart the MCP server to pick up code changes after pip install.

    Replaces the current process with a fresh one via os.execv().
    The MCP client (Claude Code) will reconnect automatically.
    """
    import os
    import sys

    stale = _check_version_stale()
    log = logging.getLogger("synapt.recall")
    if stale:
        log.info("Reloading MCP server (v%s -> installed)", _STARTUP_VERSION)
    else:
        log.info("Reloading MCP server (v%s)", _STARTUP_VERSION)

    # Flush any pending DB writes
    _invalidate_cache()

    # Replace this process with a fresh one
    os.execv(sys.executable, [sys.executable] + sys.argv)

    # os.execv never returns — this is just for type checkers
    return "Reloading..."  # pragma: no cover


def register_tools(mcp) -> None:
    """Register recall tools on the given FastMCP server instance.

    This allows the unified synapt server to compose recall tools alongside
    repair and watch tools on a single MCP server.
    """
    mcp.tool()(_with_directive_check(recall_search))
    mcp.tool()(_with_directive_check(recall_quick))
    mcp.tool()(_with_directive_check(recall_files))
    mcp.tool()(_with_directive_check(recall_sessions))
    mcp.tool()(recall_build)
    mcp.tool()(recall_setup)
    mcp.tool()(recall_export)
    mcp.tool()(recall_import)
    mcp.tool()(_with_directive_check(recall_stats))
    mcp.tool()(_with_directive_check(recall_journal))
    mcp.tool()(_with_directive_check(recall_save))
    mcp.tool()(recall_sync_memory)
    mcp.tool()(_with_directive_check(recall_remind))
    mcp.tool()(recall_enrich)
    mcp.tool()(recall_consolidate)
    mcp.tool()(_with_directive_check(recall_contradict))
    mcp.tool()(_with_directive_check(recall_correct))
    mcp.tool()(_with_directive_check(recall_context))
    mcp.tool()(_with_directive_check(recall_timeline))
    mcp.tool()(_with_directive_check(recall_channel))
    from synapt.recall.direct import speak_to_agent
    mcp.tool()(speak_to_agent)
    mcp.tool()(recall_reload)


def main():
    """Entry point for standalone synapt-recall-server."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "synapt-recall",
        instructions=MCP_INSTRUCTIONS,
    )
    register_tools(server)
    server.run()


if __name__ == "__main__":
    main()
