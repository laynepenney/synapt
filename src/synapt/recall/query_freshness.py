"""Bounded current-session indexing before a recall query."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from synapt.recall.cli import _acquire_build_lock, _release_build_lock
from synapt.recall.resume import CallerTranscript, caller_transcripts
from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.storage import query_tail_source_key


class QueryFreshnessState(str, Enum):
    CURRENT = "CURRENT"
    RECENT_GAP = "RECENT_GAP"
    REFRESHED = "REFRESHED"
    PARTIAL = "PARTIAL"
    BUSY = "BUSY"
    NOT_BOUND = "NOT_BOUND"
    ERROR = "ERROR"


@dataclass(frozen=True)
class QueryFreshnessPolicy:
    age_threshold_seconds: float = 10 * 60
    byte_trigger: int = 4 * 1024 * 1024
    step_bytes: int = 4 * 1024 * 1024
    byte_cap: int = 32 * 1024 * 1024
    wall_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.age_threshold_seconds < 0 or self.byte_trigger < 0:
            raise ValueError("query freshness triggers must be non-negative")
        if self.step_bytes <= 0 or self.byte_cap <= 0 or self.wall_seconds <= 0:
            raise ValueError("query freshness bounds must be positive")


@dataclass(frozen=True)
class QueryFreshnessResult:
    state: QueryFreshnessState
    session_id: str = ""
    source_path: Path | None = None
    source_key: str = ""
    observed_complete_offset: int | None = None
    live_bytes: int | None = None
    indexed_now_bytes: int = 0
    indexed_now_chunks: int = 0
    index_changed: bool = False
    remaining_bytes: int | None = None
    wall_seconds: float = 0.0
    cut_short: bool = False
    reason: str = ""


def _source_key(source: CallerTranscript) -> str:
    return query_tail_source_key(source.session_id, source.path)


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _gap_age_seconds(source: CallerTranscript, indexed_timestamp: str) -> float:
    """Measure temporal distance from indexed coverage to the live source."""
    latest = _parse_timestamp(source.latest_timestamp)
    indexed = _parse_timestamp(indexed_timestamp)
    if latest is None or indexed is None:
        return float("inf")
    return max(0.0, (latest - indexed).total_seconds())


def _latest_projected_timestamp(chunks, default: str = "") -> str:  # noqa: ANN001
    """Choose the chronologically latest projection across mixed ISO offsets."""
    candidates = [default] if default else []
    candidates.extend(chunk.timestamp for chunk in chunks if chunk.timestamp)
    parsed = [
        (timestamp, instant)
        for timestamp in candidates
        if (instant := _parse_timestamp(timestamp)) is not None
    ]
    if not parsed:
        return default
    return max(parsed, key=lambda item: item[1])[0]


def _complete_end(path: Path, start: int, proposed_end: int, hard_end: int) -> int:
    """Find a complete-record boundary near the step target, within the cap."""
    if proposed_end <= start:
        return start
    with path.open("rb") as stream:
        stream.seek(start)
        data = stream.read(proposed_end - start)
        newline = data.rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        if proposed_end >= hard_end:
            return start
        extension = stream.read(hard_end - proposed_end)
    newline = extension.find(b"\n")
    return start if newline < 0 else proposed_end + newline + 1


def _prefix_digest(path: Path, end: int):  # noqa: ANN201
    """Hash exactly the bytes whose projection the cursor claims to cover."""
    digest = hashlib.sha256()
    remaining = end
    with path.open("rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise OSError("source_shorter_than_cursor")
            digest.update(block)
            remaining -= len(block)
    return digest


def _parse_slice(
    source: CallerTranscript,
    *,
    start: int,
    end: int,
    turn_index: int,
):
    with source.path.open("rb") as stream:
        stream.seek(start)
        for raw_line in stream.read(end - start).splitlines():
            if not raw_line.strip():
                continue
            try:
                json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("malformed_complete_record") from exc

    if source.path.name.startswith("rollout-"):
        from synapt.recall.codex import parse_codex_transcript

        return parse_codex_transcript(
            source.path,
            start_offset=start,
            stop_offset=end,
            turn_index_start=turn_index,
            session_id_override=source.session_id,
        )

    from synapt.recall.core import parse_transcript

    return parse_transcript(
        source.path,
        start_offset=start,
        stop_offset=end,
        turn_index_start=turn_index,
        session_id_override=source.session_id,
    )


def refresh_current_session(
    index_dir: Path,
    caller_root: Path | None = None,
    *,
    policy: QueryFreshnessPolicy | None = None,
    now: datetime | None = None,
) -> QueryFreshnessResult:
    """Refresh only the caller's newest transcript under explicit bounds."""
    if policy is None:
        from synapt.recall.config import load_config

        configured = load_config().get_query_freshness()
        policy = QueryFreshnessPolicy(
            age_threshold_seconds=configured["age_threshold_seconds"],
            byte_trigger=int(configured["byte_trigger"]),
            step_bytes=int(configured["step_bytes"]),
            byte_cap=int(configured["byte_cap"]),
            wall_seconds=configured["wall_seconds"],
        )
    now = now or datetime.now(timezone.utc)
    started = time.monotonic()
    sources = caller_transcripts(caller_root or Path.cwd())
    if not sources:
        return QueryFreshnessResult(
            state=QueryFreshnessState.NOT_BOUND,
            reason="no_caller_transcript",
        )

    source = sources[0]
    try:
        key = _source_key(source)
        source_stat = source.path.stat()
        live_size = source_stat.st_size
    except OSError as exc:
        return QueryFreshnessResult(
            state=QueryFreshnessState.ERROR,
            session_id=source.session_id,
            source_path=source.path,
            reason=f"source_stat:{type(exc).__name__}",
        )

    lock_fd = _acquire_build_lock(index_dir.parent, timeout=0)
    if lock_fd is None:
        return QueryFreshnessResult(
            state=QueryFreshnessState.BUSY,
            session_id=source.session_id,
            source_path=source.path,
            source_key=key,
            live_bytes=live_size,
            cut_short=True,
            reason="build_lock",
        )

    db = None
    try:
        db = ShardedRecallDB.open(index_dir)
        cursor = db.load_query_tail_cursor(key)
        prior_session_cursor = db.load_query_tail_cursor_for_session(source.session_id)
        base_extent = db.session_indexed_extent(source.session_id)
        cursor_proven = False
        prefix_digest = None
        if cursor is not None:
            claimed_hash = cursor.get("observed_prefix_sha256", "")
            claimed_offset = int(cursor["observed_complete_offset"])
            cursor_proven = bool(claimed_hash) and claimed_offset <= live_size
            stat_matches_cursor = (
                int(cursor.get("source_size", -1)) == live_size
                and int(cursor.get("source_mtime_ns", -1))
                == source_stat.st_mtime_ns
            )
            if cursor_proven and not stat_matches_cursor:
                prefix_digest = _prefix_digest(source.path, claimed_offset)
                cursor_proven = prefix_digest.hexdigest() == claimed_hash
        starting_extent = cursor if cursor_proven else {}
        observed = int(starting_extent.get("observed_complete_offset", 0))
        rewind = int(starting_extent.get("rewind_offset", 0))
        rewind_turn = int(starting_extent.get("rewind_turn_index", 0))
        base_requires_suppression = base_extent is not None
        suppresses_base = bool(cursor and cursor.get("suppresses_base", False))
        source_replaced = cursor is not None and not cursor_proven
        source_replaced = source_replaced or (
            prior_session_cursor is not None
            and prior_session_cursor["source_key"] != key
        )
        if source_replaced:
            db.clear_query_tail_session(source.session_id)
            cursor = None
            observed = 0
            rewind = 0
            rewind_turn = 0
            starting_extent = {}
            suppresses_base = False

        if prefix_digest is None and not observed:
            prefix_digest = hashlib.sha256()

        index_changed = False
        if (
            live_size == 0
            and base_requires_suppression
            and not suppresses_base
        ):
            stamp = now.astimezone(timezone.utc).isoformat()
            db.replace_query_tail(
                source_key=key,
                session_id=source.session_id,
                rewind_offset=0,
                chunks=[],
                cursor={
                    "transcript_path": str(source.path),
                    "observed_complete_offset": 0,
                    "rewind_offset": 0,
                    "rewind_turn_index": 0,
                    "source_size": 0,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                    "observed_prefix_sha256": prefix_digest.hexdigest(),
                    "suppresses_base": True,
                    "latest_projected_timestamp": "",
                    "last_attempt_at": stamp,
                    "last_success_at": stamp,
                },
            )
            suppresses_base = True
            index_changed = True

        if observed >= live_size:
            return QueryFreshnessResult(
                state=QueryFreshnessState.CURRENT,
                session_id=source.session_id,
                source_path=source.path,
                source_key=key,
                observed_complete_offset=observed,
                live_bytes=live_size,
                index_changed=index_changed,
                remaining_bytes=0,
                wall_seconds=time.monotonic() - started,
            )

        gap = live_size - observed
        if (
            gap < policy.byte_trigger
            and _gap_age_seconds(
                source,
                str(starting_extent.get("latest_projected_timestamp", "")),
            )
            < policy.age_threshold_seconds
        ):
            return QueryFreshnessResult(
                state=QueryFreshnessState.RECENT_GAP,
                session_id=source.session_id,
                source_path=source.path,
                source_key=key,
                observed_complete_offset=observed,
                live_bytes=live_size,
                remaining_bytes=gap,
                wall_seconds=time.monotonic() - started,
                reason="below_threshold",
            )

        if prefix_digest is None:
            prefix_digest = _prefix_digest(source.path, observed)

        indexed_bytes = 0
        indexed_chunks = 0
        reason = ""
        while observed < live_size:
            if indexed_bytes >= policy.byte_cap:
                reason = "byte_cap"
                break
            if time.monotonic() - started >= policy.wall_seconds:
                reason = "wall_cap"
                break
            remaining_cap = min(
                policy.byte_cap - indexed_bytes,
                live_size - observed,
            )
            allowance = min(policy.step_bytes, remaining_cap)
            hard_end = observed + remaining_cap
            end = _complete_end(
                source.path,
                observed,
                observed + allowance,
                hard_end,
            )
            if end <= observed:
                reason = (
                    "record_exceeds_byte_cap"
                    if hard_end < live_size
                    else "incomplete_record"
                )
                break

            chunks = _parse_slice(
                source,
                start=rewind,
                end=end,
                turn_index=rewind_turn,
            )
            next_rewind = chunks[-1].byte_offset if chunks else end
            next_turn = chunks[-1].turn_index if chunks else rewind_turn
            projected = _latest_projected_timestamp(
                chunks,
                str(starting_extent.get("latest_projected_timestamp", "")),
            )
            stamp = now.astimezone(timezone.utc).isoformat()
            step_suppresses_base = suppresses_base or (
                base_requires_suppression and end >= live_size
            )
            with source.path.open("rb") as stream:
                stream.seek(observed)
                prefix_digest.update(stream.read(end - observed))
            db.replace_query_tail(
                source_key=key,
                session_id=source.session_id,
                rewind_offset=rewind,
                chunks=chunks,
                cursor={
                    "transcript_path": str(source.path),
                    "observed_complete_offset": end,
                    "rewind_offset": next_rewind,
                    "rewind_turn_index": next_turn,
                    "source_size": live_size,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                    "observed_prefix_sha256": prefix_digest.hexdigest(),
                    "suppresses_base": step_suppresses_base,
                    "latest_projected_timestamp": projected,
                    "last_attempt_at": stamp,
                    "last_success_at": stamp,
                },
            )
            indexed_bytes += end - observed
            indexed_chunks += len(chunks)
            index_changed = True
            observed = end
            rewind = next_rewind
            rewind_turn = next_turn
            cursor = db.load_query_tail_cursor(key)
            suppresses_base = step_suppresses_base
            starting_extent = cursor or starting_extent

        remaining = live_size - observed
        complete = remaining == 0
        return QueryFreshnessResult(
            state=(
                QueryFreshnessState.REFRESHED
                if complete
                else QueryFreshnessState.PARTIAL
            ),
            session_id=source.session_id,
            source_path=source.path,
            source_key=key,
            observed_complete_offset=observed,
            live_bytes=live_size,
            indexed_now_bytes=indexed_bytes,
            indexed_now_chunks=indexed_chunks,
            index_changed=index_changed,
            remaining_bytes=remaining,
            wall_seconds=time.monotonic() - started,
            cut_short=not complete,
            reason=reason,
        )
    except Exception as exc:
        known_observed = locals().get("observed")
        known_indexed_bytes = locals().get("indexed_bytes", 0)
        known_indexed_chunks = locals().get("indexed_chunks", 0)
        return QueryFreshnessResult(
            state=QueryFreshnessState.ERROR,
            session_id=source.session_id,
            source_path=source.path,
            source_key=key,
            observed_complete_offset=known_observed,
            live_bytes=live_size,
            indexed_now_bytes=known_indexed_bytes,
            indexed_now_chunks=known_indexed_chunks,
            index_changed=bool(locals().get("index_changed", False)),
            remaining_bytes=(
                live_size - known_observed
                if isinstance(known_observed, int)
                else None
            ),
            wall_seconds=time.monotonic() - started,
            cut_short=True,
            reason=f"{type(exc).__name__}:{exc}",
        )
    finally:
        if db is not None:
            db.close()
        _release_build_lock(lock_fd)


def format_query_freshness(result: QueryFreshnessResult) -> str:
    """Render one stable, caller-scoped freshness line."""
    parts = [f"Freshness: {result.state.value}"]
    if result.session_id:
        parts.append(f"session={result.session_id[:8]}")
    parts.extend(
        [
            "indexed_through=" + (
                str(result.observed_complete_offset)
                if result.observed_complete_offset is not None
                else "unknown"
            ),
            "live_bytes=" + (
                str(result.live_bytes)
                if result.live_bytes is not None
                else "unknown"
            ),
            (
                f"indexed_now={result.indexed_now_bytes}B/"
                f"{result.indexed_now_chunks}chunks"
            ),
            f"wall={result.wall_seconds:.3f}s",
            "remaining=" + (
                f"{result.remaining_bytes}B"
                if result.remaining_bytes is not None
                else "unknown"
            ),
            f"cut_short={'true' if result.cut_short else 'false'}",
        ]
    )
    if result.reason:
        parts.append(f"reason={result.reason}")
    return " ".join(parts)
