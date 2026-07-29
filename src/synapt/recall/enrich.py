"""Tier 3 journal enrichment — LLM-powered summarization of auto-stubs.

Reads auto-generated journal stubs, loads the corresponding transcript,
sends a summarization prompt to a local MLX model, and appends enriched
entries to journal.jsonl.

Requires mlx-lm (pip install mlx-lm). Degrades gracefully if not installed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from synapt.recall.journal import (
    JournalEntry,
    append_entry,
    _journal_path,
)
from synapt.recall.core import (
    parse_transcript,
    project_archive_dir,
    project_transcript_dir,
    all_worktree_archive_dirs,
)
from synapt.recall.scrub import scrub_text
from synapt.recall._llm_util import truncate_at_word as _tw

logger = logging.getLogger("synapt.recall.enrich")

from synapt.recall._mlx import MLX_AVAILABLE as _MLX_AVAILABLE, INSTALL_MSG as _INSTALL_MSG  # noqa: F401
if _MLX_AVAILABLE:
    from synapt._models.mlx_client import MLXClient, MLXOptions
    from synapt._models.base import Message

from synapt.recall._model_router import DEFAULT_DECODER_MODEL as DEFAULT_MODEL

# Mapping of text header labels to enrichment dict keys
_ENRICHMENT_FIELD_MAP = {
    "done": "done",
    "decisions": "decisions",
    "next_steps": "next_steps",
    "next steps": "next_steps",
}
MAX_TRANSCRIPT_CHARS = 6000  # ~1.5K tokens — fits in 3B context budget
MULTI_WINDOW_OVERLAP = int(os.environ.get("SYNAPT_MW_OVERLAP", "0"))  # default 0; set via env for ablation


def _find_transcript(session_id: str, project_dir: Path) -> Path | None:
    """Locate the transcript file for a session ID across all worktrees."""
    # Check all worktree archives first
    for archive in all_worktree_archive_dirs(project_dir):
        path = archive / f"{session_id}.jsonl"
        if path.exists():
            return path
    # Fall back to live transcripts
    transcript_dir = project_transcript_dir(project_dir)
    if transcript_dir:
        path = Path(transcript_dir) / f"{session_id}.jsonl"
        if path.exists():
            return path
    return None


def _build_summary_from_chunks(
    chunks: list,
    max_chars: int = MAX_TRANSCRIPT_CHARS,
) -> str:
    """Build a condensed text summary from a list of TranscriptChunks."""
    from synapt.recall.scrub import strip_system_artifacts

    parts: list[str] = []
    for chunk in chunks:
        turn = f"[Turn {chunk.turn_index}]"
        user_text = strip_system_artifacts(chunk.user_text)[:500] if chunk.user_text else ""
        if user_text:
            parts.append(f"{turn} User: {user_text}")
        if chunk.assistant_text:
            parts.append(f"{turn} Assistant: {chunk.assistant_text[:500]}")
        if chunk.tools_used:
            parts.append(f"{turn} Tools: {', '.join(chunk.tools_used[:10])}")
        if chunk.files_touched:
            parts.append(f"{turn} Files: {', '.join(chunk.files_touched[:10])}")
        # Include tool result content (truncated) — captures config values,
        # URLs, command outputs that pure tool names miss.
        if getattr(chunk, "tool_content", ""):
            parts.append(f"{turn} Output: {chunk.tool_content[:300]}")

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


def _split_windows(
    text: str,
    window_size: int = MAX_TRANSCRIPT_CHARS,
    overlap: int = MULTI_WINDOW_OVERLAP,
) -> list[str]:
    """Split text into overlapping windows for multi-window enrichment.

    Returns a list of text windows. If text fits in one window, returns [text].
    """
    if len(text) <= window_size:
        return [text]

    windows = []
    start = 0
    step = window_size - overlap
    while start < len(text):
        end = min(start + window_size, len(text))
        windows.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return windows


def _dedup_facts(items: list[str], threshold: float = 0.8) -> list[str]:
    """Deduplicate a list of fact strings using fuzzy matching.

    Keeps the first occurrence of each near-duplicate group.
    """
    from difflib import SequenceMatcher

    unique: list[str] = []
    for item in items:
        is_dup = False
        for existing in unique:
            if SequenceMatcher(None, item.lower(), existing.lower()).ratio() > threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(item)
    return unique


def _multi_window_enabled() -> bool:
    """Check if multi-window enrichment is enabled via env var."""
    return os.environ.get("SYNAPT_MULTI_WINDOW_ENRICH", "0") == "1"


def _build_transcript_summary(session_id: str, project_dir: Path) -> str:
    """Build a condensed text representation of a transcript for the LLM."""
    path = _find_transcript(session_id, project_dir)
    if not path:
        return ""

    chunks = parse_transcript(path)
    if not chunks:
        return ""

    return _build_summary_from_chunks(chunks)


def _build_full_transcript_text(session_id: str, project_dir: Path) -> str:
    """Build full (non-truncated) transcript text for multi-window enrichment."""
    path = _find_transcript(session_id, project_dir)
    if not path:
        return ""

    chunks = parse_transcript(path)
    if not chunks:
        return ""

    # Use a very large limit to avoid truncation
    return _build_summary_from_chunks(chunks, max_chars=10_000_000)


ENRICHMENT_PROMPT = """\
You are summarizing a session transcript. Extract structured information about what happened.

Session date: {session_date}

Given the following transcript, produce a JSON object with these fields:
- "focus": A one-sentence description of the session's main topic or goal (max 200 chars)
- "done": A list of {done_limit} concrete things accomplished, discussed, or learned (each max 100 chars)
- "decisions": A list of {decisions_limit} key decisions made or preferences stated (each max 100 chars)
- "next_steps": A list of {next_steps_limit} items to follow up on or remember (each max 100 chars)

Rules:
- Include specific names, dates, places, numbers, and details — NOT general themes
- Resolve ALL relative time references to absolute dates using the session date above (e.g. "yesterday" → the actual date, "last week" → the actual date range)
- Capture people's names, their relationships, possessions, hobbies, and stated preferences
- BAD: "discussed travel plans" — GOOD: "plans trip to Florida in June with sister Elena"
- BAD: "talked about hobbies" — GOOD: "signed up for pottery class on July 2nd"
- BAD: "fixed a bug yesterday" — GOOD: "fixed auth bug on March 14"
- NEVER output template variables like {{yesterday}}, {{last_friday}}, or {{today}} — always write the actual resolved date
{personal_rules}- If the session was short or trivial, use fewer items
- Output ONLY valid JSON, no markdown fences, no explanation

Transcript:
{transcript}

JSON:"""

# Additional extraction rules injected only for personal-content sessions.
_PERSONAL_RULES = """\
- Capture opinions, emotional reactions, hypothetical plans, social relationships, and recurring topics
- BAD: "talked about feelings" — GOOD: "Sarah is frustrated with her job at the bank and considering applying to grad school"
- Capture possessions, pets, addresses, workplace details, and family relationships
"""


# Content-profile-aware fact limits for enrichment output clamping.
# Personal conversations contain many more extractable facts than code sessions.
_FACT_LIMITS = {
    "code":     {"done": 5, "decisions": 3, "next_steps": 3},
    "mixed":    {"done": 5, "decisions": 3, "next_steps": 3},
    "personal": {"done": 15, "decisions": 5, "next_steps": 5},
}


def _get_fact_limits(content_type: str) -> dict:
    """Return fact limits for the given content type."""
    return _FACT_LIMITS.get(content_type, _FACT_LIMITS["mixed"])


def _parse_llm_response(response: str) -> dict | None:
    """Parse the LLM's JSON response, handling common formatting issues.

    Falls back to extracting structured text (Focus:/Done:/Decisions:)
    when the model ignores JSON instructions.
    """
    from synapt.recall._llm_util import parse_llm_json
    result = parse_llm_json(response)
    if result is not None:
        return result
    # Fallback: parse structured text output from small models
    return _parse_enrichment_text(response)


def _parse_enrichment_text(text: str) -> dict | None:
    """Extract enrichment fields from structured text output.

    Handles patterns like::

        Focus: The session discussed X and Y
        Done:
        - Thing 1
        - Thing 2
        Decisions:
        - Decided to do Z

    Also handles: all-caps headers (``FOCUS:``), numbered lists
    (``1. Thing``), inline items (``Done: Thing 1, Thing 2``),
    and blank lines between headers and items.
    """
    result: dict = {}

    # Extract focus (single line after "Focus:")
    focus_match = re.search(r"focus:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if focus_match:
        result["focus"] = focus_match.group(1).strip().strip('"')

    # Extract list fields (done, decisions, next_steps)
    for label, key in _ENRICHMENT_FIELD_MAP.items():
        # Block format: header + bulleted/numbered items (optional blank lines)
        block_pat = (
            rf"{re.escape(label)}:\s*\n"
            r"\s*((?:(?:\s*[-*•·]\s*.+|\s*\d+[.)]\s*.+)\n?)+)"
        )
        match = re.search(block_pat, text, re.IGNORECASE)
        if match:
            items = re.findall(r"(?:[-*•·]|\d+[.)])\s*(.+)", match.group(1))
            result[key] = [
                item.strip().strip('"') for item in items if item.strip()
            ]
            continue

        # Inline format: "Done: Thing 1, Thing 2" (items on same line)
        inline_pat = rf"{re.escape(label)}:\s*(.+?)(?:\n\n|\n(?=[A-Z])|$)"
        inline = re.search(inline_pat, text, re.IGNORECASE)
        if inline:
            val = inline.group(1).strip()
            if not val:
                continue
            if "," in val or ";" in val:
                items = re.split(r"[,;]\s*", val)
                parsed = [
                    i.strip().strip('"') for i in items
                    if i.strip() and len(i.strip()) > 2
                ]
                if parsed:
                    result[key] = parsed
            elif len(val) > 5:
                result[key] = [val.strip('"')]

    if result.get("focus") or result.get("done") or result.get("decisions"):
        return result
    return None


def iter_enrichable_entries(
    journal_path: Path | None = None,
) -> Iterator[JournalEntry]:
    """Yield journal entries eligible for enrichment.

    Eligible: auto=True AND no rich content (no done/decisions/next_steps).
    """
    path = journal_path or _journal_path()
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = JournalEntry.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if entry.auto and not entry.enriched and not (
                entry.done or entry.decisions or entry.next_steps
            ):
                yield entry


def _detect_content_type(transcript_text: str) -> str:
    """Detect content type from transcript text for enrichment tuning.

    Uses lightweight heuristics (not full ContentProfile) since we only
    have the summary text, not raw chunks.
    """
    from synapt.recall.content_profile import ContentProfile

    # Count code vs personal signals in the transcript text
    profile = ContentProfile(total_chunks=1)
    if re.search(r'(?:\.py|\.js|\.ts|\.rs|\.go|\.java|\.cpp|\.h|\.c)\b', transcript_text):
        profile.file_refs = 1
    if re.search(r'(?:Tool|Execute|Read|Write|Edit|Bash|Grep|Glob):', transcript_text):
        profile.tool_uses = 1
    if '```' in transcript_text:
        profile.code_fences = 1
    if re.search(r'(?:function|class|def |import |from |const |let |var )\b', transcript_text):
        profile.tech_idents = 1
    if re.search(r'\b(?:feel|friend|family|love|happy|sad|angry|dinner|weekend|vacation|birthday)\b',
                 transcript_text, re.IGNORECASE):
        profile.personal_refs = 1

    return profile.content_type


def _enrich_single_window(
    transcript_text: str,
    session_date: str,
    content_type: str,
    limits: dict,
    client: "MLXClient",
    model: str,
    adapter_path: str = "",
) -> dict | None:
    """Run enrichment on a single transcript window. Returns parsed dict or None."""
    prompt = ENRICHMENT_PROMPT.format(
        transcript=transcript_text,
        session_date=session_date,
        done_limit=f"1-{limits['done']}",
        decisions_limit=f"0-{limits['decisions']}",
        next_steps_limit=f"0-{limits['next_steps']}",
        personal_rules=_PERSONAL_RULES if content_type == "personal" else "",
    )

    try:
        response = client.chat(
            model=model,
            messages=[Message(role="user", content=prompt)],
            temperature=0.1,
            adapter_path=adapter_path or None,
        )
    except Exception as exc:
        logger.warning("LLM inference failed: %s", exc)
        return None

    return _parse_llm_response(response)


def _merge_enrichment_results(results: list[dict]) -> dict:
    """Merge enrichment results from multiple windows with dedup."""
    merged: dict = {"focus": "", "done": [], "decisions": [], "next_steps": []}

    for r in results:
        if not merged["focus"] and r.get("focus"):
            merged["focus"] = r["focus"]
        for key in ("done", "decisions", "next_steps"):
            merged[key].extend(r.get(key, []))

    # Dedup each list
    for key in ("done", "decisions", "next_steps"):
        merged[key] = _dedup_facts([str(item) for item in merged[key] if item])

    return merged


def score_transcript_windows(windows: list[str]):
    """Score transcript windows via the active scoring strategy.

    Per internal design ratification: enrichment-time scoring is a cheap
    precompute layer (scores stored on chunks
    at ingest) that the consolidation primary site consumes. This helper exposes
    the `ChunkScoringStrategy` seam at the canonical enrichment scoring point.

    OSS default (RecencyScoring): scores by position in input order (older →
    newer). Premium plugins may register alternative strategies via the
    registry seam; OSS does not name premium-side implementations.

    Backward-compat: with no plugin activated, returned scores reflect linear
    recency ramp. This helper is the integration seam for plugin-registered
    strategies, not a replacement for existing window-splitting + multi-window
    enrichment logic.

    Args:
        windows: transcript windows as strings (e.g. output of `_split_windows`),
            in temporal order (oldest → newest).

    Returns:
        List of ScoredChunks in input order.
    """
    from synapt.recall.scoring import (
        ScoringInput,
        get_active_strategy,
        score_with_validation,
    )

    if not windows:
        return []

    inputs = [
        ScoringInput(content=w, position=i, metadata={})
        for i, w in enumerate(windows)
    ]
    return score_with_validation(get_active_strategy(), inputs)


def enrich_entry(
    entry: JournalEntry,
    project_dir: Path,
    model: str = DEFAULT_MODEL,
    client: "MLXClient | None" = None,
    adapter_path: str = "",
    transcript_text: str = "",
) -> JournalEntry | None:
    """Enrich a single auto-journal stub using LLM inference.

    Uses the model router to select the best backend (decoder-only preferred
    for JSON schema compliance). Falls back to MLX decoder if router unavailable.

    When SYNAPT_MULTI_WINDOW_ENRICH=1 and content is personal, runs enrichment
    on overlapping 6K-char windows to avoid truncation of long conversations.

    Args:
        client: Reusable model client instance. Created via router if not provided.
        adapter_path: Optional LoRA adapter (MLX decoder-only).
        transcript_text: Pre-built transcript summary. If provided, skips
            transcript lookup (used by segment enrichment).

    Returns the enriched entry, or None if enrichment failed.
    """
    if not transcript_text:
        transcript_text = _build_transcript_summary(entry.session_id, project_dir)
    if not transcript_text:
        logger.warning("No transcript found for session %s", entry.session_id)
        return None

    if client is None:
        # Try router first (prefers encoder-decoder for speed)
        from synapt.recall._model_router import get_client, RecallTask
        client = get_client(RecallTask.ENRICH, max_tokens=800)
        if client is None:
            # All backends unavailable
            if not _MLX_AVAILABLE:
                return None
            client = MLXClient(MLXOptions(max_tokens=800))

    # Detect content type for profile-aware limits
    content_type = _detect_content_type(transcript_text)
    limits = _get_fact_limits(content_type)

    # Derive human-readable session date for relative date resolution
    session_date = "unknown"
    if entry.timestamp:
        try:
            from datetime import datetime as _dt, timezone as _tz
            _ts = entry.timestamp.replace("Z", "+00:00")
            _parsed = _dt.fromisoformat(_ts)
            session_date = _parsed.strftime("%A, %B %d, %Y")
        except (ValueError, AttributeError):
            pass

    # Multi-window enrichment for personal content when enabled
    use_multi_window = (
        _multi_window_enabled()
        and content_type == "personal"
        and len(transcript_text) > MAX_TRANSCRIPT_CHARS
    )

    if use_multi_window:
        # Get full (non-truncated) transcript text
        full_text = _build_full_transcript_text(entry.session_id, project_dir)
        if not full_text:
            full_text = transcript_text

        windows = _split_windows(full_text)
        logger.info(
            "Multi-window enrichment: %d windows for session %s (%d chars)",
            len(windows), entry.session_id, len(full_text),
        )

        window_results = []
        for i, window in enumerate(windows):
            parsed = _enrich_single_window(
                window, session_date, content_type, limits,
                client, model, adapter_path,
            )
            if parsed:
                window_results.append(parsed)
            else:
                logger.warning(
                    "Window %d/%d failed for session %s",
                    i + 1, len(windows), entry.session_id,
                )

        if not window_results:
            return None

        parsed = _merge_enrichment_results(window_results)
    else:
        parsed = _enrich_single_window(
            transcript_text, session_date, content_type, limits,
            client, model, adapter_path,
        )

    if not parsed:
        logger.warning(
            "Failed to parse enrichment for session %s", entry.session_id,
        )
        return None

    # Build enriched entry, preserving original metadata.
    # Scrub LLM output — the model may echo secrets from the transcript.
    def _s(text: str) -> str:
        try:
            return scrub_text(text)
        except Exception:
            return text  # Use unscrubbed text rather than aborting the batch

    return JournalEntry(
        timestamp=entry.timestamp,
        session_id=entry.session_id,
        branch=entry.branch,
        focus=_s(_tw(str(parsed.get("focus", entry.focus)), 200)),
        done=[_s(_tw(str(d), 100)) for d in parsed.get("done", []) if d][:limits["done"]],
        decisions=[_s(_tw(str(d), 100)) for d in parsed.get("decisions", []) if d][:limits["decisions"]],
        next_steps=[_s(_tw(str(d), 100)) for d in parsed.get("next_steps", []) if d][:limits["next_steps"]],
        files_modified=entry.files_modified,
        git_log=entry.git_log,
        auto=False,
        enriched=True,
    )


def _has_conversation(transcript_path: Path) -> bool:
    """Return True if transcript has actual conversation content (not just snapshots)."""
    import json as _json
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = _json.loads(line)
                    if d.get("type") in ("user", "assistant"):
                        return True
                except (ValueError, _json.JSONDecodeError):
                    continue
    except OSError:
        return False
    return False


def _backfill_stubs(
    project_dir: Path,
    journal_path: Path,
    journaled_sessions: set[str],
) -> int:
    """Create auto-stubs for archived transcripts missing journal entries.

    Only creates stubs for transcripts with actual conversation content
    (skips snapshot-only files that have no user/assistant turns).

    Returns the number of stubs created.
    """
    from datetime import datetime, timezone

    archive = project_archive_dir(project_dir)
    if not archive.exists():
        return 0

    count = 0
    for transcript in sorted(archive.glob("*.jsonl")):
        session_id = transcript.stem
        if session_id in journaled_sessions:
            continue
        if not _has_conversation(transcript):
            continue
        stub = JournalEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            branch="",
            auto=True,
            enriched=False,
        )
        append_entry(stub, journal_path)
        journaled_sessions.add(session_id)
        count += 1

    if count:
        logger.info("Backfilled %d auto-stubs from archived transcripts", count)
    return count


@dataclass
class TranscriptSegment:
    """A time-delimited segment of a larger transcript."""

    session_id: str  # Original session UUID
    segment_index: int  # 0-based within session
    segment_id: str  # "{session_id}:s{segment_index}"
    start_timestamp: str  # ISO 8601 of first chunk
    end_timestamp: str  # ISO 8601 of last chunk
    branch: str  # Most common branch in segment
    chunks: list  # TranscriptChunks in this segment
    files_touched: list[str]  # Unique files across all chunks


def _segment_transcript(
    transcript_path: Path,
    gap_minutes: int = 60,
) -> list[TranscriptSegment]:
    """Split a transcript into time-based segments.

    Groups chunks by time gaps — a gap of *gap_minutes* or more between
    consecutive chunks starts a new segment. Segments with no user text
    (e.g. snapshot-only regions) are dropped.

    Args:
        transcript_path: Path to a .jsonl transcript file.
        gap_minutes: Minimum gap in minutes to start a new segment.

    Returns list of TranscriptSegment, sorted chronologically.
    """
    from datetime import datetime, timedelta

    chunks = parse_transcript(transcript_path)
    if not chunks:
        return []

    # Filter to chunks with timestamps
    dated = [c for c in chunks if c.timestamp]
    if not dated:
        return []

    # Sort by timestamp
    dated.sort(key=lambda c: c.timestamp)
    gap = timedelta(minutes=gap_minutes)

    # Group into segments by time gaps
    groups: list[list] = [[dated[0]]]
    for chunk in dated[1:]:
        try:
            prev_ts = datetime.fromisoformat(groups[-1][-1].timestamp.replace("Z", "+00:00"))
            curr_ts = datetime.fromisoformat(chunk.timestamp.replace("Z", "+00:00"))
            if (curr_ts - prev_ts) >= gap:
                groups.append([chunk])
            else:
                groups[-1].append(chunk)
        except (ValueError, TypeError):
            groups[-1].append(chunk)

    # Build segments, dropping empty ones
    session_id = dated[0].session_id
    segments: list[TranscriptSegment] = []
    for i, group in enumerate(groups):
        # Skip segments with no user content
        if not any(c.user_text for c in group):
            continue

        # Use git branch from chunk metadata if available, else ""
        branch = ""

        # Collect unique files
        all_files: list[str] = []
        seen: set[str] = set()
        for c in group:
            for f in c.files_touched:
                if f not in seen:
                    all_files.append(f)
                    seen.add(f)

        seg_idx = len(segments)
        segments.append(TranscriptSegment(
            session_id=session_id,
            segment_index=seg_idx,
            segment_id=f"{session_id}:s{seg_idx}",
            start_timestamp=group[0].timestamp,
            end_timestamp=group[-1].timestamp,
            branch=branch,
            chunks=group,
            files_touched=all_files,
        ))

    return segments


def enrich_transcript_segments(
    transcript_path: Path,
    project_dir: Path,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    max_entries: int = 0,
    adapter_path: str = "",
    gap_minutes: int = 60,
) -> int:
    """Segment a large transcript and enrich each segment into a journal entry.

    Used for bootstrapping a project's journal from existing large transcripts
    that contain many logical sessions in a single file.

    Args:
        transcript_path: Path to the .jsonl transcript file.
        project_dir: Project root directory.
        gap_minutes: Minimum gap in minutes between segments (default: 60).

    Returns number of segments enriched.
    """
    if not _MLX_AVAILABLE and not dry_run:
        print(_INSTALL_MSG)
        return 0

    journal_path = _journal_path(project_dir)

    # Check which segment IDs already exist in journal
    existing_sids: set[str] = set()
    if journal_path.exists():
        with open(journal_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    sid = d.get("session_id", "")
                    if sid:
                        existing_sids.add(sid)
                except (json.JSONDecodeError, TypeError):
                    continue

    segments = _segment_transcript(transcript_path, gap_minutes=gap_minutes)
    if not segments:
        print("[init] No segments found in transcript.")
        return 0

    print(f"[init] Found {len(segments)} segments in transcript.")

    client = None
    if not dry_run and _MLX_AVAILABLE:
        client = MLXClient(MLXOptions(max_tokens=800))

    count = 0
    for seg in segments:
        if seg.segment_id in existing_sids:
            print(f"  Skipping {seg.segment_id[:20]} (already journaled)")
            continue

        summary = _build_summary_from_chunks(seg.chunks)
        if not summary:
            continue

        ts_raw = seg.start_timestamp
        # Truncate ISO 8601 to date+HH:MM; keep free-text timestamps intact.
        if len(ts_raw) > 16 and ts_raw[4:5] == "-" and ts_raw[10:11] == "T":
            ts_short = ts_raw[:16]
        else:
            ts_short = ts_raw

        if dry_run:
            n_turns = len(seg.chunks)
            user_preview = ""
            for c in seg.chunks:
                if c.user_text:
                    user_preview = c.user_text[:60]
                    break
            print(f"  Segment {seg.segment_index}: {ts_short} — {n_turns} turns — {user_preview}...")
            count += 1
        else:
            stub = JournalEntry(
                timestamp=seg.start_timestamp,
                session_id=seg.segment_id,
                branch=seg.branch,
                files_modified=seg.files_touched[:20],
                auto=True,
                enriched=False,
            )
            enriched = enrich_entry(
                stub, project_dir, model,
                client=client,
                adapter_path=adapter_path,
                transcript_text=summary,
            )
            if enriched:
                append_entry(enriched, journal_path)
                print(f"  Segment {seg.segment_index}: {ts_short} — {enriched.focus[:70]}")
                existing_sids.add(seg.segment_id)
                count += 1
            else:
                print(f"  Segment {seg.segment_index}: {ts_short} — (enrichment failed)")

        if max_entries and count >= max_entries:
            break

    return count


def enrich_all(
    project_dir: Path | None = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    max_entries: int = 0,
    adapter_path: str = "",
) -> int:
    """Enrich all eligible auto-journal stubs.

    Args:
        project_dir: Project root. Default: cwd.
        model: MLX model to use for summarization.
        dry_run: If True, print what would be enriched without doing it.
        max_entries: Max entries to enrich (0 = unlimited).
        adapter_path: Optional LoRA adapter for enrichment.

    Returns:
        Number of entries enriched.
    """
    # Use router to find best available backend
    from synapt.recall._model_router import get_client, RecallTask
    client = get_client(RecallTask.ENRICH, max_tokens=800)
    if client is None:
        if not _MLX_AVAILABLE:
            print(_INSTALL_MSG)
            return 0
        client = MLXClient(MLXOptions(max_tokens=800))

    project_dir = (project_dir or Path.cwd()).resolve()
    journal_path = _journal_path(project_dir)

    # Collect all journaled session_ids and non-auto session_ids
    all_journaled: set[str] = set()
    existing_non_auto: set[str] = set()
    if journal_path.exists():
        with open(journal_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    sid = d.get("session_id", "")
                    if sid:
                        all_journaled.add(sid)
                    if not d.get("auto", False) and sid:
                        existing_non_auto.add(sid)
                except (json.JSONDecodeError, TypeError):
                    continue

    # Backfill: create auto-stubs for archived transcripts with no journal entry
    _backfill_stubs(project_dir, journal_path, all_journaled)

    count = 0
    for entry in iter_enrichable_entries(journal_path):
        if entry.session_id in existing_non_auto:
            continue  # Already has a manual/enriched entry
        # Dedup relies on existing_non_auto: successfully enriched entries
        # get a non-auto entry appended, so they're skipped on retry.
        # Failed entries stay as auto stubs and are retried next run.

        if dry_run:
            focus_preview = entry.focus[:80] if entry.focus else "(no focus)"
            print(f"  Would enrich: {entry.session_id[:8]} — {focus_preview}")
            count += 1
        else:
            enriched = enrich_entry(entry, project_dir, model, client=client, adapter_path=adapter_path)
            if enriched:
                append_entry(enriched, journal_path)
                print(f"  Enriched: {entry.session_id[:8]} — {enriched.focus[:80]}")
                existing_non_auto.add(entry.session_id)
                count += 1
            else:
                print(f"  Skipped: {entry.session_id[:8]} (enrichment failed)")

        if max_entries and count >= max_entries:
            break

    # Record when enrichment last ran (observability only — not used for filtering)
    if count > 0 and not dry_run:
        _set_last_enrichment_ts(project_dir)

    return count


def _set_last_enrichment_ts(project_dir: Path) -> None:
    """Record when enrichment last ran (observability/debugging only).

    Not used for filtering — dedup relies on existing_non_auto set in
    enrich_all(), which correctly retries failed entries.
    """
    try:
        from datetime import datetime, timezone
        from synapt.recall.core import project_index_dir
        from synapt.recall.storage import RecallDB
        db_path = project_index_dir(project_dir) / "recall.db"
        if not db_path.exists():
            return
        db = RecallDB(db_path)
        db.set_metadata("last_enrichment_ts", datetime.now(timezone.utc).isoformat())
        db.close()
    except Exception as exc:
        logger.debug("Failed to save enrichment timestamp: %s", exc)
