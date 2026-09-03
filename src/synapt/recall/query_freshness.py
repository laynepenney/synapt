"""Bounded current-session indexing before a recall query."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from synapt.recall.cli import (
    _acquire_build_lock,
    _release_build_lock,
    build_lock_has_waiter,
)
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


def index_oversize_source(
    index_dir: Path,
    source: CallerTranscript,
    *,
    policy: QueryFreshnessPolicy | None = None,
    now: datetime | None = None,
) -> QueryFreshnessResult:
    """Chunk-index one oversize transcript under a byte/wall budget.

    R3.1 (data growth): the same budgeted-chunk mechanism
    ``refresh_current_session`` already uses for a caller's live session
    tail, generalized to an arbitrary file too large for ``build_index``
    to parse whole. Progress is written into the query_tail overlay
    (``replace_query_tail``), the same tables ``ShardedRecallDB.fts_search``
    already merges into every search -- so a search from another process
    sees content from the first chunk onward, without touching the base
    shards ``save_chunks`` rewrites (that path is not this function's
    concern; it never runs here).

    Deliberately NOT a caller-transcript / recency check: a file this
    function is asked to index is oversize by definition (the caller has
    already applied that ceiling), so there is no "not stale enough to
    bother" case the way a live session has. Every call either makes
    progress or returns why it made none.

    Resumable by construction: the persisted cursor's mtime/size stamp and
    prefix hash (the same drift check ``refresh_current_session`` uses) mean
    a rewrite of the source file is detected and restarts the cursor at
    zero rather than silently corrupting the offset it hands back.
    """
    if policy is None:
        policy = QueryFreshnessPolicy(byte_trigger=0)
    now = now or datetime.now(timezone.utc)
    started = time.monotonic()
    key = _source_key(source)

    try:
        source_stat = source.path.stat()
        live_size = source_stat.st_size
    except OSError as exc:
        return QueryFreshnessResult(
            state=QueryFreshnessState.ERROR,
            session_id=source.session_id,
            source_path=source.path,
            reason=f"source_stat:{type(exc).__name__}",
        )

    # The build lock is acquired PER CHUNK below, not once for the whole
    # call: reading the cursor needs no lock at all (readers never lock,
    # per the design this reuses from refresh_current_session), and each
    # WRITE reloads the cursor fresh immediately after, so releasing
    # between chunks costs nothing but lets a foreground build or
    # `recall_build` interleave instead of waiting out this call's entire
    # wall budget (measured: without this, a multi-minute catchup call
    # held the lock the whole span).
    db = None
    try:
        db = ShardedRecallDB.open(index_dir)
        cursor = db.load_query_tail_cursor(key)
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
        if cursor is not None and not cursor_proven:
            # Source was rewritten under us: the offset this cursor claims
            # no longer describes this file's bytes. Restart at zero rather
            # than resume against content that has moved.
            db.clear_query_tail(key)
            observed = 0
            rewind = 0
            rewind_turn = 0
            starting_extent = {}

        if prefix_digest is None and not observed:
            prefix_digest = hashlib.sha256()

        if observed >= live_size:
            return QueryFreshnessResult(
                state=QueryFreshnessState.CURRENT,
                session_id=source.session_id,
                source_path=source.path,
                source_key=key,
                observed_complete_offset=observed,
                live_bytes=live_size,
                remaining_bytes=0,
                wall_seconds=time.monotonic() - started,
            )

        if prefix_digest is None:
            prefix_digest = _prefix_digest(source.path, observed)

        indexed_bytes = 0
        indexed_chunks = 0
        index_changed = False
        reason = ""
        while observed < live_size:
            if indexed_bytes >= policy.byte_cap:
                reason = "byte_cap"
                break
            if time.monotonic() - started >= policy.wall_seconds:
                reason = "wall_cap"
                break

            lock_fd = _acquire_build_lock(index_dir.parent, timeout=0)
            if lock_fd is None:
                reason = "build_lock"
                break
            try:
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
                    # A record not fitting in what's LEFT of this call's cap
                    # (indexed_bytes > 0: we already made progress, we just
                    # ran low) is the ordinary byte_cap stop, one iteration
                    # later than the top-of-loop check catches it. Only a
                    # record that does not fit with the FULL cap available
                    # (indexed_bytes == 0, the very first attempt this call)
                    # is a genuine single-record-exceeds-the-cap condition.
                    if indexed_bytes == 0:
                        reason = (
                            "record_exceeds_byte_cap"
                            if hard_end < live_size
                            else "incomplete_record"
                        )
                    else:
                        reason = "byte_cap"
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
                        "suppresses_base": False,
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
                starting_extent = db.load_query_tail_cursor(key) or starting_extent
            finally:
                _release_build_lock(lock_fd)

            if build_lock_has_waiter(index_dir.parent):
                # Releasing and immediately re-acquiring in the same tight
                # loop wins the race against a poller almost every time (the
                # gap between release and re-acquire is microseconds; a
                # waiter polling every 0.5s rarely lands inside it) --
                # measured: a real _acquire_build_lock(timeout=15) waiter
                # never got in during a 200 MB chunked call. Checking the
                # waiting marker right after release and stopping here lets
                # a genuine waiter (a foreground build, recall_build) win
                # the very next attempt instead of losing to this call's own
                # remaining wall budget.
                reason = "build_lock_yield"
                break

        remaining = live_size - observed
        complete = remaining == 0
        if indexed_bytes == 0 and reason == "build_lock":
            # No chunk in this call ever got the lock -- the honest state
            # is the same BUSY this function returned before any resume
            # logic ran, just now carrying the resumed offset too (a prior
            # call's progress is real information, not a reason to hide it
            # behind a bare BUSY).
            return QueryFreshnessResult(
                state=QueryFreshnessState.BUSY,
                session_id=source.session_id,
                source_path=source.path,
                source_key=key,
                observed_complete_offset=observed,
                live_bytes=live_size,
                remaining_bytes=remaining,
                wall_seconds=time.monotonic() - started,
                cut_short=True,
                reason=reason,
            )
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


def oversize_transcripts(
    transcript_dirs: list[Path], ceiling: int | None = None
) -> list[Path]:
    """List .jsonl files over the oversize ceiling across the given dirs.

    Mirrors ``build_index``'s own ``stat.st_size > ceiling`` check exactly
    (same env-overridable default) so this function and the build's own
    skip decision can never disagree about which files are oversize.
    """
    if ceiling is None:
        from synapt.recall.core import _max_transcript_file_bytes

        ceiling, _warning = _max_transcript_file_bytes()
    found: list[Path] = []
    for transcript_dir in transcript_dirs:
        for filepath in sorted(transcript_dir.glob("*.jsonl")):
            try:
                if filepath.stat().st_size > ceiling:
                    found.append(filepath)
            except OSError:
                continue
    return found


#: Wall-governed, not byte-governed: the catchup path's whole point is to
#: make as much progress as the wall budget allows, so a byte_cap anywhere
#: near QueryFreshnessPolicy's live-session-tail default (32 MiB) silently
#: throttles convergence to a fraction of what the wall budget below would
#: otherwise finish (measured: with the class default, 33.5 MB per call
#: regardless of a 20s wall budget -- 112 session starts to finish a
#: 3.5 GB file that a lifted cap finishes in one).
OVERSIZE_CATCHUP_BYTE_CAP = 10**12

#: Env-overridable: the fix-forward target is finishing the motivating
#: multi-GB file in one detached catchup run. Safe to be this generous
#: because the build lock is released and re-acquired per chunk inside
#: index_oversize_source, not held for the whole span -- a foreground
#: build or `recall_build` interleaves rather than waiting out 10 minutes.
DEFAULT_OVERSIZE_CATCHUP_WALL_SECONDS = 600.0


def _oversize_catchup_wall_seconds() -> tuple[float, str | None]:
    from synapt.recall.core import _int_env_override

    value, warning = _int_env_override(
        "SYNAPT_OVERSIZE_CATCHUP_WALL_SECONDS",
        int(DEFAULT_OVERSIZE_CATCHUP_WALL_SECONDS),
    )
    return float(value), warning


def catchup_oversize_transcripts(
    project_dir: Path,
    index_dir: Path,
    *,
    overall_wall_seconds: float | None = None,
    per_call_policy: QueryFreshnessPolicy | None = None,
    ceiling: int | None = None,
    byte_cap: int | None = None,
) -> list[QueryFreshnessResult]:
    """Make budgeted progress on every oversize transcript, one catchup step.

    R3.1 (data growth): the ``cmd_catchup`` step this drives from. Spends up
    to ``overall_wall_seconds`` total across however many oversize files
    exist, so one session-start catchup invocation makes SOME progress on
    each rather than exhausting its whole budget on the first file found
    and starving the rest. A file already fully caught up
    (``remaining_bytes == 0``) costs one cheap stat + cursor read and moves
    on. Never raises: a per-file error is captured in that file's own
    ``QueryFreshnessResult`` (state=ERROR) so one bad file cannot stop
    progress on the others.

    Wall-governed, not byte-governed (see ``OVERSIZE_CATCHUP_BYTE_CAP``):
    each file's per-call policy lifts byte_cap so only the wall budget
    below bounds progress, unless the caller passes its own
    ``per_call_policy`` explicitly.
    """
    from synapt.recall.core import project_transcript_dirs

    if overall_wall_seconds is None:
        overall_wall_seconds, warning = _oversize_catchup_wall_seconds()
        if warning:
            print(f"[synapt] {warning}")

    started = time.monotonic()
    results: list[QueryFreshnessResult] = []
    transcript_dirs = project_transcript_dirs(project_dir)
    for filepath in oversize_transcripts(transcript_dirs, ceiling=ceiling):
        remaining_budget = overall_wall_seconds - (time.monotonic() - started)
        if remaining_budget <= 0:
            break
        try:
            stat = filepath.stat()
        except OSError as exc:
            results.append(
                QueryFreshnessResult(
                    state=QueryFreshnessState.ERROR,
                    source_path=filepath,
                    reason=f"source_stat:{type(exc).__name__}",
                )
            )
            continue
        source = CallerTranscript(
            session_id=filepath.stem,
            path=filepath,
            mtime=stat.st_mtime,
            size=stat.st_size,
        )
        policy = per_call_policy or QueryFreshnessPolicy(
            age_threshold_seconds=0,
            byte_trigger=0,
            byte_cap=byte_cap if byte_cap is not None else OVERSIZE_CATCHUP_BYTE_CAP,
            wall_seconds=min(remaining_budget, overall_wall_seconds),
        )
        results.append(index_oversize_source(index_dir, source, policy=policy))
    return results


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
