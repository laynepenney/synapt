"""Session-tail reconstruction — the surface behind ``synapt resume``.

recall#927. When a session ends cleanly it writes a journal entry. When it ends
*uncleanly* — a crash, exhausted quota, a dead terminal — nothing is written, and
the next session has no way to ask *"what were the last things that happened, and
what did they intend?"* Relevance-ranked search cannot answer it: a cold query
like "where did we leave off" has no notion of recency, so it surfaces
mid-session chunks. This module answers it by position instead of by relevance.

Every design choice here is shaped by one property of the problem: a wrong answer
is *invisible*. A resume view that quietly drops the last turn, or pairs this
session's tail with a different session's intent, looks exactly like a correct
one. So the rules below are deliberately asymmetric — they prefer showing
something slightly noisy over hiding something real, and they label an inferred
pairing rather than presenting it as a proven one.

Runtime independence is structural at the rendered-tail boundary: anything the
parsers produce resumes identically. Caller freshness additionally reads the
last top-level timestamp from the caller-scoped live JSONL source. It does not
interpret runtime-specific event payloads.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

from synapt.recall.freshness import IndexFreshness
from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.journal import JournalEntry, read_entries
from synapt.recall.sharded_db import ShardedRecallDB

DEFAULT_TURNS = 10

# Harness-emitted control blocks that survive scrub.py. scrub strips
# <system-reminder>, <local-command-caveat>, <available-deferred-tools> and
# <env>; the slash-command family below is *not* covered there, so this is new
# vocabulary rather than a second copy of an existing list.
_HARNESS_TAGS = (
    "command-name",
    "command-message",
    "command-args",
    "local-command-stdout",
    "local-command-stderr",
)
_HARNESS_BLOCK_RE = re.compile(
    rf"<({'|'.join(_HARNESS_TAGS)})(?:\s[^>]*)?>.*?</\1>",
    re.DOTALL,
)
# Unclosed/truncated variants, matching the fallback scrub.py already uses.
_HARNESS_OPEN_RE = re.compile(
    rf"<({'|'.join(_HARNESS_TAGS)})(?:\s[^>]*)?>.*",
    re.DOTALL,
)

# Emitted by the runtime when a compacted session resumes. Not authored by
# anyone in the conversation.
_CONTINUATION_PREAMBLE = (
    "This session is being continued from a previous conversation"
)

# When one user question produces a long reply, the chunker splits it into
# segments; segment 0 keeps the real question and later segments get this
# synthetic restatement as their entire user text (see core.py). Rendering it
# as user speech would report a question nobody asked twice.
_CONTEXT_PREFIX = "(context: User previously asked:"


class ResumeError(Exception):
    """Raised when the requested session cannot be resolved."""


@dataclass
class ResumeTurn:
    """One conversation turn, traceable back to the chunk it came from."""

    chunk_id: str
    turn_index: int
    timestamp: str
    user_text: str
    assistant_text: str
    tools_used: list[str] = field(default_factory=list)
    is_continuation: bool = False


@dataclass
class ResumeView:
    """Everything ``synapt resume`` renders, with provenance attached.

    ``journal_provenance`` is ``"bound"`` (the entry names this session),
    ``"inferred"`` (the entry names no session and this is the newest one), or
    ``None``. The distinction is load-bearing: an inferred pairing that renders
    identically to a proven one is a fabrication.
    """

    session_id: str
    turns: list[ResumeTurn] = field(default_factory=list)
    journal: JournalEntry | None = None
    journal_provenance: str | None = None
    total_turns: int = 0
    excluded_count: int = 0
    omitted_between: int = 0
    # Whether the index this view was read from is current, and over which
    # surface that was decided. ``None`` means NOT CHECKED -- which is not the
    # same as fresh, and the renderer must not treat it as such.
    freshness: "IndexFreshness | None" = None
    selection_scope: str = "explicit"
    source_label: str = "unknown"
    caller_unindexed: list["CallerTranscript"] = field(default_factory=list)
    caller_partial: "CallerExtentGap | None" = None
    # Set by the CLI when the selected session has no journal or checkpoint
    # covering its last activity. None means NOT CHECKED, not clean.
    unclean_end: "UncleanEnd | None" = None


@dataclass(frozen=True)
class CallerTranscript:
    session_id: str
    path: Path
    mtime: float
    size: int
    latest_timestamp: str = ""


@dataclass(frozen=True)
class CallerExtentGap:
    source: CallerTranscript
    indexed_latest: str
    live_latest: str


# A journal written this close to the session's last activity counts as its
# handoff. Fifteen minutes covers an EOD "journal, then a few closing turns".
UNCLEAN_END_GRACE_SECONDS = 15 * 60


@dataclass(frozen=True)
class UncleanEnd:
    """The newest previous session has no handoff of any kind.

    A crash, kill, or forced shutdown runs no SessionEnd, so no checkpoint is
    written, and nobody journals a session they did not know was ending. The
    wake then shows whatever checkpoint IS on disk, which may belong to another
    session entirely, and nothing says that hours of work have no record.

    A handoff is SESSION-BOUND evidence (Atlas, r1 on the first version of
    this): a journal certifies only the session it names. Reduced to a
    timestamp, any later journal from any later session would erase the crash
    verdict for good. So ``last_authored_journal`` is the newest journal that
    could be this session's (bound to it, or legacy sessionless), and
    ``foreign_journal`` is the newest one that belongs to someone else, kept so
    the wake can say why it does not count. ``gap_seconds`` is the time from
    that own-or-sessionless journal to the last activity, ``None`` when there
    is none before the activity. ``checkpoint_session`` names the session the
    on-disk checkpoint belongs to when it is not this one.

    Known limit: only the NEWEST previous transcript is judged. A short clean
    session after a crash (a subagent, a `claude -p` probe) becomes the newest
    and hides the crash, and with one checkpoint slot an older session that
    ended cleanly is indistinguishable from one that crashed once its
    checkpoint is overwritten. Per-session checkpoints keyed by id remove that
    limit; until then the wake judges one session and says which.
    """

    session_id: str
    transcript_path: Path
    last_activity: str
    last_authored_journal: str | None
    gap_seconds: float | None
    checkpoint_session: str | None
    foreign_journal: str | None = None


def _latest_event_timestamp(path: Path, *, block_size: int = 64 * 1024) -> str:
    """Read backward until a JSONL event with a top-level timestamp appears."""
    with path.open("rb") as stream:
        position = stream.seek(0, 2)
        suffix = b""
        while position:
            start = max(0, position - block_size)
            stream.seek(start)
            data = stream.read(position - start) + suffix
            lines = data.splitlines()
            suffix = lines.pop(0) if start and lines else b""
            for line in reversed(lines):
                try:
                    event = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                timestamp = event.get("timestamp") if isinstance(event, dict) else None
                if isinstance(timestamp, str) and timestamp:
                    return timestamp
            position = start
        if suffix:
            try:
                event = json.loads(suffix)
            except (UnicodeDecodeError, json.JSONDecodeError):
                event = None
            timestamp = event.get("timestamp") if isinstance(event, dict) else None
            if isinstance(timestamp, str) and timestamp:
                return timestamp
    return ""


def _timestamp_epoch(value: str) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def caller_transcripts(project_dir: Path | None = None) -> list[CallerTranscript]:
    """Discover transcripts rooted at the caller, not the shared gripspace."""
    from synapt.recall.codex import _session_cwd, discover_codex_sessions
    from synapt.recall.core import project_transcript_dir
    from synapt.recall.journal import extract_session_id

    caller = (project_dir or Path.cwd()).resolve()
    paths: list[Path] = []
    claude_root = project_transcript_dir(caller)
    if claude_root is not None:
        paths.extend(sorted(claude_root.glob("*.jsonl")))
    codex_root = discover_codex_sessions()
    if codex_root is not None:
        for path in sorted(codex_root.rglob("rollout-*.jsonl")):
            if _session_cwd(path) == caller:
                paths.append(path)

    found: list[CallerTranscript] = []
    for path in paths:
        try:
            stat = path.stat()
            session_id = extract_session_id(str(path))
            latest_timestamp = _latest_event_timestamp(path)
        except OSError:
            continue
        if session_id:
            found.append(
                CallerTranscript(
                    session_id,
                    path,
                    stat.st_mtime,
                    stat.st_size,
                    latest_timestamp,
                )
            )
    return sorted(found, key=lambda item: item.mtime, reverse=True)


def _journal_covers(entry: JournalEntry, session_id: str, last_epoch: float, grace: float) -> bool | None:
    """True: covers. False: cannot (bound elsewhere). None: not a cover, but not foreign."""
    jt = _timestamp_epoch(entry.timestamp)
    if entry.session_id and entry.session_id != session_id:
        return False
    if entry.session_id == session_id:
        # Its own handoff: written near the end, or any time after it.
        return jt >= last_epoch - grace or None
    # Legacy sessionless entry: only inside the symmetric window around the end.
    return abs(jt - last_epoch) <= grace or None


def detect_unclean_end(
    sources: list[CallerTranscript],
    *,
    checkpoint: dict | None,
    authored_journals: list[JournalEntry],
    exclude_session_id: str | None = None,
    grace_seconds: float = UNCLEAN_END_GRACE_SECONDS,
) -> UncleanEnd | None:
    """Judge whether the newest previous transcript was handed off. Pure.

    ``exclude_session_id`` is the session that is starting: at SessionStart it
    is the newest transcript on disk and would otherwise report itself. The
    exclusion is by IDENTITY alone and is a CALL-SITE guarantee; a caller that
    cannot name its own session must not publish this verdict. Recency is
    deliberately not treated as liveness evidence (Atlas, r1): a crash
    followed by a fast restart sits inside any recency window, and hiding a
    real crash costs more than a visible self-report from a caller whose id
    failed to match, which the reader can see and correct.
    """
    candidates = [
        item for item in sources
        if item.latest_timestamp and item.session_id != exclude_session_id
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda item: _timestamp_epoch(item.latest_timestamp))
    last_epoch = _timestamp_epoch(newest.latest_timestamp)

    checkpoint_session = None
    if checkpoint:
        checkpoint_session = str(checkpoint.get("session_id") or "") or None
        if checkpoint_session == newest.session_id:
            return None

    own_or_sessionless: list[JournalEntry] = []
    foreign: list[JournalEntry] = []
    for entry in authored_journals:
        verdict = _journal_covers(entry, newest.session_id, last_epoch, grace_seconds)
        if verdict is True:
            return None
        (foreign if verdict is False else own_or_sessionless).append(entry)

    latest_own = max(own_or_sessionless, key=lambda e: _timestamp_epoch(e.timestamp), default=None)
    latest_foreign = max(foreign, key=lambda e: _timestamp_epoch(e.timestamp), default=None)
    gap: float | None = None
    if latest_own is not None:
        own_epoch = _timestamp_epoch(latest_own.timestamp)
        if own_epoch <= last_epoch:
            gap = last_epoch - own_epoch

    return UncleanEnd(
        session_id=newest.session_id,
        transcript_path=newest.path,
        last_activity=newest.latest_timestamp,
        last_authored_journal=latest_own.timestamp if latest_own else None,
        gap_seconds=gap,
        checkpoint_session=checkpoint_session,
        foreign_journal=latest_foreign.timestamp if latest_foreign else None,
    )


def authored_journals(journal_path: Path | None) -> list[JournalEntry]:
    """Non-auto entries with content. Auto stubs are not handoffs."""
    if journal_path is None or not journal_path.exists():
        return []
    from synapt.recall.journal import _read_all_entries

    return [
        entry for entry in _read_all_entries(journal_path)
        if not entry.auto and entry.has_content()
    ]


def gather_unclean_end(
    project: Path,
    *,
    exclude_session_id: str | None = None,
    authored: list[JournalEntry] | None = None,
    journal_path: Path | None = None,
) -> UncleanEnd | None:
    """The I/O half of the detector: caller transcripts, checkpoint, journal.

    Bounded: transcript discovery reads one tail block per file and the
    checkpoint is one small JSON file. Never raises; a failure to judge is
    reported as no finding, which is the same as today's behaviour.
    """
    try:
        from synapt.checkpoint import read_checkpoint

        if authored is None:
            authored = authored_journals(journal_path)
        return detect_unclean_end(
            caller_transcripts(project),
            checkpoint=read_checkpoint(project),
            authored_journals=authored,
            exclude_session_id=exclude_session_id,
        )
    except Exception:
        return None


def _gap_phrase(seconds: float | None) -> str:
    if seconds is None:
        return "no authored journal at all"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    return f"{hours}h{minutes:02d}m"


def format_unclean_end(found: UncleanEnd, tail: dict | None) -> str:
    """The wake block. Header first, then the facts, then this session's own tail."""
    short = found.session_id[:8]
    if found.last_authored_journal and found.gap_seconds is not None:
        journal_line = (
            f"Last authored journal that could be this session's: {found.last_authored_journal}; "
            f"{_gap_phrase(found.gap_seconds)} of work after it has no journal."
        )
    elif found.last_authored_journal:
        journal_line = (
            f"The latest journal that could be this session's ({found.last_authored_journal}) "
            "is later than its last activity and outside the handoff window."
        )
    else:
        journal_line = "No authored journal exists for this session."
    if found.foreign_journal:
        later = (
            "later "
            if _timestamp_epoch(found.foreign_journal) > _timestamp_epoch(found.last_activity)
            else ""
        )
        journal_line += (
            f" A {later}journal at {found.foreign_journal} belongs to a different session"
            " and does not count."
        )
    lines = [
        f"UNCLEAN END — session {short} ended without a handoff",
        f"Last activity {found.last_activity}. {journal_line}",
    ]
    if found.checkpoint_session:
        lines.append(
            f"No SessionEnd checkpoint for this session; checkpoint.json holds session "
            f"{found.checkpoint_session[:8]}, which is NOT this one's bridge."
        )
    else:
        lines.append("No SessionEnd checkpoint for this session (a crash runs no SessionEnd).")
    lines.append(f"Bridge: the tail below, then #dev since {found.last_authored_journal or 'the start of that session'}.")
    if tail:
        user = tail.get("last_user_text") or "unavailable in bounded transcript tail"
        assistant = tail.get("last_assistant_text") or "unavailable in bounded transcript tail"
        files = tail.get("files_touched") or []
        lines.append(
            "RECOVERED TAIL (raw, bounded; not an authored journal; "
            f"status {tail.get('parse_status', 'unavailable')}"
            f"{' truncated' if tail.get('truncated') else ''}):"
        )
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {assistant}")
        lines.append("Files: " + (", ".join(str(f) for f in files[:8]) or "none observed"))
    lines.append(f"Full tail: synapt resume {found.session_id}")
    return "\n".join(lines)


def _source_label(path: str) -> str:
    if not path:
        return "unknown"
    parts = Path(path).parts
    try:
        pos = parts.index("worktrees")
    except ValueError:
        pos = -1
    if pos >= 0 and len(parts) > pos + 2 and parts[pos + 2] == "transcripts":
        return f"worktree:{parts[pos + 1]}"

    # Claude stores one directory per project below ``.claude/projects``. The
    # directory name is already the runtime's project slug, so printing the
    # absolute parent adds a home-directory prefix without adding identity.
    try:
        project_pos = parts.index("projects")
    except ValueError:
        project_pos = -1
    if project_pos >= 0 and len(parts) > project_pos + 1:
        return f"project:{parts[project_pos + 1]}"

    parent_name = Path(path).parent.name
    return f"source:{parent_name}" if parent_name else "source:unknown"


class BoundedResumeIndex:
    """The small TranscriptIndex surface resume needs over a read-only store."""

    def __init__(self, db: ShardedRecallDB):
        self._db = db
        self._overview = db.session_overview()
        self.sessions = {session_id: [] for session_id in self._overview}
        self._session_order = sorted(
            self._overview,
            key=lambda session_id: self._overview[session_id]["activity"],
            reverse=True,
        )

    def session_tail(self, session_id: str) -> list[TranscriptChunk]:
        return self._db.load_session_chunks(session_id)

    def list_sessions(
        self,
        max_sessions: int = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> list[dict]:
        """Return recent session summaries while hydrating only candidates."""
        session_ids = []
        for session_id in self._session_order:
            overview = self._overview[session_id]
            earliest_ts = overview["earliest_ts"]
            latest_ts = overview["latest_ts"]
            if not earliest_ts or not latest_ts:
                continue
            if after and latest_ts < after:
                continue
            if before and earliest_ts >= before:
                continue
            session_ids.append(session_id)
            if len(session_ids) >= max_sessions:
                break

        hydrated = self._db.load_session_listing(session_ids)
        results = []
        for session_id in session_ids:
            overview = self._overview[session_id]
            chunks = hydrated.get(session_id, [])
            transcript_chunks = [
                chunk for chunk in chunks if chunk["turn_index"] >= 0
            ]
            first_message = ""
            for chunk in sorted(
                transcript_chunks or chunks,
                key=lambda item: item["turn_index"],
            ):
                if chunk["user_text"]:
                    first_message = chunk["user_text"][:120]
                    if len(chunk["user_text"]) > 120:
                        first_message += "..."
                    break
            files = {
                file_path
                for chunk in chunks
                for file_path in chunk["files_touched"]
            }
            results.append(
                {
                    "session_id": session_id,
                    "date": overview["earliest_ts"][:10],
                    "turn_count": overview["turn_count"],
                    "first_message": first_message,
                    "files_count": len(files),
                    "source_root": _source_label(next(
                        (chunk["transcript_path"] for chunk in chunks if chunk.get("transcript_path")),
                        overview.get("transcript_path", ""),
                    )),
                }
            )
        return results

    def close(self) -> None:
        self._db.close()


def load_resume_index(directory: Path) -> TranscriptIndex | BoundedResumeIndex:
    """Load only session routing metadata until one session is selected."""
    from synapt.recall.sharding import is_sharded

    if (directory / "recall.db").exists() or is_sharded(directory):
        return BoundedResumeIndex(ShardedRecallDB.open_readonly(directory))
    return TranscriptIndex.load(directory, use_embeddings=False)


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------


def is_harness_authored(chunk: TranscriptChunk) -> bool:
    """True when a chunk was written by the runtime rather than a participant.

    The rule is a **conjunction**: the user text must be *entirely* a harness
    control block **and** nothing must have responded to it. Both halves are
    required because each one alone has a known false-reject:

    * The marker alone would reject a turn where someone *discussed* a command,
      quoting the tag in prose.
    * The emptiness alone would reject a final user message that never got a
      reply — which is precisely what a dropped baton looks like, and the single
      most load-bearing turn this feature exists to surface.

    Exclusion therefore requires positive identification. Absence of evidence
    keeps the turn, so the failure mode is a visible noisy line rather than an
    invisible deletion. A slash command a person actually typed (``/compact
    <directive>``) is kept for the same reason — only its *echo* is harness text.
    """
    if chunk.assistant_text.strip() or chunk.tools_used:
        return False

    text = chunk.user_text.strip()
    if not text:
        # Content-free, but that is emptiness rather than authorship. Keeping
        # the two reasons distinct is what lets each be witnessed separately.
        return False

    if text.startswith(_CONTINUATION_PREAMBLE):
        return True

    residue = _HARNESS_OPEN_RE.sub("", _HARNESS_BLOCK_RE.sub("", text)).strip()
    if residue == text:
        return False  # no marker found — nothing positively identified
    return not residue


def _is_content_free(chunk: TranscriptChunk) -> bool:
    """True when a chunk carries nothing a reader could use."""
    return not (
        chunk.user_text.strip()
        or chunk.assistant_text.strip()
        or chunk.tools_used
    )


def _is_continuation(chunk: TranscriptChunk) -> bool:
    return chunk.user_text.lstrip().startswith(_CONTEXT_PREFIX)


#: Channel chunks are grouped under pseudo-session ids with this prefix.
_CHANNEL_PREFIX = "channel_"


def is_channel_session(session_id: str) -> bool:
    """True when an id names a channel rather than a working session."""
    return session_id.startswith(_CHANNEL_PREFIX)


def _carries_intent(entry: JournalEntry) -> bool:
    """True when an entry records what someone MEANT, not just what was touched.

    Two things are deliberately not intent:

    * **A file list alone.** ``files_modified`` records what was touched, never
      why. An entry carrying only that is a record of activity, and presenting
      it as the session's journal tells a returning reader nothing they can act
      on — while looking exactly like a real answer.
    * **A focus made only of harness markup.** The auto-extractor takes the
      session's first user message as the focus, and after a ``/clear`` that
      message is the runtime's own control block. The same residue rule used to
      filter harness turns applies here, so the two stay consistent.
    """
    if entry.done or entry.decisions or entry.next_steps:
        return True
    focus = (entry.focus or "").strip()
    if not focus:
        return False
    residue = _HARNESS_OPEN_RE.sub("", _HARNESS_BLOCK_RE.sub("", focus)).strip()
    if residue == focus:
        return True  # no marker found — ordinary prose
    return bool(residue)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def resolve_session(
    index: TranscriptIndex,
    session_id: str | None,
    caller_session_ids: set[str] | None = None,
    agent_id: str | None = None,
) -> str:
    """Resolve a session id, accepting an exact id or a unique prefix.

    Prefixes are supported because ``recall sessions`` prints eight characters —
    the id a reader has in hand is usually a prefix, so rejecting it would make
    the two commands unusable together.

    An unresolvable id raises rather than falling back to the newest session.
    Silently resuming a *different* session is the worst failure available here:
    nothing in the output would look wrong.

    The default skips channel pseudo-sessions. Channels share the session
    namespace but are not sessions, and #dev is written to constantly — so the
    newest thing in the index is very often a channel. Naming one explicitly
    still works, because the failure this guards against is a silent wrong
    default, not a reader who asked for a channel on purpose. (recall#935: this
    was invisible until the ordering defect above it was fixed, and repairing
    ordering alone would have moved the default from a stale session to #dev.)
    """
    order = index._session_order
    if not order:
        raise ResumeError("No sessions indexed yet. Nothing to resume.")

    if not session_id:
        if agent_id:
            overview = getattr(index, "_overview", {})
            for sid in order:
                if is_channel_session(sid):
                    continue
                if overview:
                    agent_ids = overview.get(sid, {}).get("agent_ids", ())
                else:
                    agent_ids = {
                        chunk.agent_id
                        for chunk in index.sessions.get(sid, ())
                        if chunk.agent_id
                    }
                if agent_id in agent_ids:
                    return sid
        if caller_session_ids:
            for sid in order:
                if sid in caller_session_ids and not is_channel_session(sid):
                    return sid
        for sid in order:
            if not is_channel_session(sid):
                return sid
        raise ResumeError(
            "Only channels are indexed — no session to resume. "
            "Name a channel explicitly to read its tail."
        )

    if session_id in index.sessions:
        return session_id

    matches = [sid for sid in order if sid.startswith(session_id)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        listed = ", ".join(matches[:8])
        raise ResumeError(
            f"Session prefix '{session_id}' is ambiguous — matches: {listed}"
        )
    raise ResumeError(f"No session matching '{session_id}' in this index.")


def _select_journal(
    session_id: str,
    journal_path: Path | None,
    is_newest: bool,
) -> tuple[JournalEntry | None, str | None]:
    """Choose the journal entry that belongs with this session's tail.

    Three states, deliberately kept distinct:

    * The entry names this session — bound, shown as proven.
    * The entry names a *different* session — positive contradiction, never
      shown. Pairing one session's tail with another's intent would read as
      authoritative while being wrong.
    * The entry names no session at all — cannot be contradicted, so it is shown
      with its provenance disclosed, and only when resuming the newest session.

    The third case is not hypothetical: ``auto_extract_entry`` leaves
    ``session_id`` empty whenever no transcript was discoverable, which is common
    enough that an exact-match-only rule would fail to bind in the ordinary case.
    """
    if journal_path is None:
        return None, None

    entries = read_entries(journal_path, n=50)

    # An id match that carries no intent is a stub. It must not END the search:
    # the entry a reader actually needs is often the unbound rich one below, and
    # letting a stub win on a technicality is how a real journal with 188
    # next-steps went unseen behind a file list (recall#937). So this scans every
    # id match for one that carries intent before giving up on binding at all.
    for entry in entries:
        if entry.session_id and entry.session_id == session_id and _carries_intent(entry):
            return entry, "bound"

    if not is_newest:
        return None, None

    for entry in entries:
        if not entry.session_id and _carries_intent(entry):
            return entry, "inferred"

    return None, None


def build_resume_view(
    index: TranscriptIndex,
    session_id: str | None = None,
    limit: int = DEFAULT_TURNS,
    journal_path: Path | None = None,
    caller_sources: list[CallerTranscript] | None = None,
    agent_id: str | None = None,
) -> ResumeView:
    """Assemble the tail of a session, newest last.

    ``journal_path`` of ``None`` means *do not read a journal* rather than *use
    the default*. Callers name the path they want, so this function performs no
    implicit I/O and tests cannot accidentally read a real journal.
    """
    if limit < 1:
        raise ResumeError(f"--turns must be at least 1 (got {limit}).")

    caller_ids = {item.session_id for item in caller_sources or []}
    resolved = resolve_session(
        index,
        session_id,
        caller_ids if session_id is None else None,
        agent_id=agent_id if session_id is None else None,
    )
    if session_id is not None:
        selection_scope = "explicit"
    elif agent_id:
        overview = getattr(index, "_overview", {})
        if overview:
            selected_agent_ids = overview.get(resolved, {}).get("agent_ids", ())
        else:
            selected_agent_ids = {
                chunk.agent_id
                for chunk in index.sessions.get(resolved, ())
                if chunk.agent_id
            }
        selection_scope = "agent" if agent_id in selected_agent_ids else (
            "caller" if resolved in caller_ids else "store"
        )
    else:
        selection_scope = "caller" if resolved in caller_ids else "store"
    is_newest = (
        session_id is None and selection_scope == "caller"
    ) or (bool(index._session_order) and index._session_order[0] == resolved)

    chunks = index.session_tail(resolved)
    kept = [
        c for c in chunks
        if not _is_content_free(c) and not is_harness_authored(c)
    ]

    window = kept[-limit:]

    # Anchor the window to a question. In an agentic session one user message
    # commonly produces many segments, so a tail of N chunks can land entirely
    # inside a single reply and show no user message at all — verified against a
    # real 324-chunk session, where every turn in the default window was a
    # continuation. The reader is left asking what was even being answered.
    #
    # So when the window opens mid-reply, the question it continues is prepended.
    # Exactly one turn, never a range: the cost is bounded no matter how many
    # segments the reply ran to, and the gap is disclosed rather than glossed.
    omitted_between = 0
    if window and _is_continuation(window[0]):
        start = kept.index(window[0])
        for back in range(start - 1, -1, -1):
            if not _is_continuation(kept[back]):
                omitted_between = start - back - 1
                window = [kept[back]] + window
                break

    turns = [
        ResumeTurn(
            chunk_id=c.id,
            turn_index=c.turn_index,
            timestamp=c.timestamp,
            user_text="" if _is_continuation(c) else c.user_text,
            assistant_text=c.assistant_text,
            tools_used=list(c.tools_used),
            is_continuation=_is_continuation(c),
        )
        for c in window
    ]

    journal, provenance = _select_journal(resolved, journal_path, is_newest)

    source_path = next((c.transcript_path for c in chunks if c.transcript_path), "")
    source_label = _source_label(source_path)
    selected_latest = max(
        (chunk.timestamp for chunk in chunks if chunk.timestamp),
        key=_timestamp_epoch,
        default="",
    )
    selected_epoch = _timestamp_epoch(selected_latest)
    unindexed = [
        item for item in (caller_sources or [])
        if item.session_id not in index.sessions and item.mtime > selected_epoch
    ]
    selected_source = next(
        (item for item in (caller_sources or []) if item.session_id == resolved),
        None,
    )
    caller_partial = None
    if (
        selected_source is not None
        and selected_source.latest_timestamp
        and _timestamp_epoch(selected_source.latest_timestamp) > selected_epoch
    ):
        caller_partial = CallerExtentGap(
            selected_source,
            selected_latest or "no searchable transcript endpoint",
            selected_source.latest_timestamp,
        )

    return ResumeView(
        session_id=resolved,
        turns=turns,
        journal=journal,
        journal_provenance=provenance,
        total_turns=len(kept),
        excluded_count=len(chunks) - len(kept),
        omitted_between=omitted_between,
        selection_scope=selection_scope,
        source_label=source_label,
        caller_unindexed=unindexed,
        caller_partial=caller_partial,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _clip(text: str, max_chars: int) -> str:
    """Shorten text, always saying so. Silent truncation misreports the content."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n    … [truncated, {len(text) - max_chars} more chars]"


def _format_journal(view: ResumeView) -> list[str]:
    entry = view.journal
    if entry is None:
        return []

    if view.journal_provenance == "bound" and getattr(entry, "auto", False):
        # `auto` means the extractor wrote this, not the session. The id still
        # binds it to the right session, so the entry is worth showing — but
        # claiming authorship it does not have would misrepresent how much a
        # reader should trust it.
        header = "JOURNAL (auto-extracted for this session, not written by it)"
    elif view.journal_provenance == "bound":
        header = "JOURNAL (written by this session)"
    else:
        header = (
            "JOURNAL (provenance: inferred — the entry records no session id, "
            "so it may belong to a different session)"
        )

    lines = ["", header]
    if entry.focus:
        lines.append(f"  Focus: {entry.focus}")
    for label, items in (
        ("Done", entry.done),
        ("Decisions", entry.decisions),
        ("Next steps", entry.next_steps),
    ):
        if items:
            lines.append(f"  {label}:")
            lines.extend(f"    - {item}" for item in items)
    return lines




def _describe_behind(f) -> str:
    """Say WHICH way the index is behind — unseen files and grown ones differ.

    Calling a grown transcript "not yet indexed" is wrong and sends the reader
    looking for a file the index already knows about. The two conditions have
    the same remedy but not the same meaning, and a reader debugging one should
    not be told the other.
    """
    parts = []
    if f.new_files:
        n = len(f.new_files)
        parts.append(f"{n} file{'' if n == 1 else 's'} not yet indexed")
    if f.changed_files:
        n = len(f.changed_files)
        parts.append(f"{n} indexed file{'' if n == 1 else 's'} grown since the build")
    return " and ".join(parts) if parts else "behind"


def _format_freshness(view: ResumeView) -> list[str]:
    """Disclose a stale index whether or not turns rendered.

    Turns shown from a stale index are still real; they may simply be missing
    the newest ones. Silence here would let a partial answer read as complete.
    """
    f = view.freshness
    if f is None or not f.stale:
        return []
    detail = f" ({_describe_behind(f)})" if (f.new_files or f.changed_files) else ""
    return [
        "",
        f"⚠ The index is STALE{detail} — built {f.build_timestamp or 'at an unrecorded time'}, "
        f"checked against: {f.scanned}.",
        f"  Newer turns may exist that this view cannot see. To index them:  {f.remedy}",
    ]


def _format_empty(view: ResumeView) -> list[str]:
    """Render an empty view WITH its provenance.

    The original text here read "every indexed chunk was harness output or
    empty" unconditionally. That names a CAUSE the renderer had not
    established: when the index is stale the session may have no chunks in the
    index at all, and this function would report a property of the SESSION
    having observed only a property of the INDEX. The three branches below are
    three different answers and must never print the same words.
    """
    f = view.freshness
    if f is not None and f.stale:
        return [
            "No turns found — but the index is STALE, so this is not an answer "
            "about the session.",
            f"  {_describe_behind(f)} "
            f"(built {f.build_timestamp or 'at an unrecorded time'}, checked: {f.scanned}).",
            f"  Index them, then ask again:  {f.remedy}",
        ]
    if f is None:
        return [
            "No conversational turns found in the index for this session.",
            "  Index freshness was not checked, so this does not establish that "
            "the session is empty.",
        ]
    return [
        "No conversational turns in this session — every indexed chunk was "
        "harness output or empty.",
        f"  The index is current (checked: {f.scanned}), so this is an answer "
        f"about the session, not about the index.",
    ]


def format_resume(view: ResumeView, max_chars: int = 600) -> str:
    """Render a resume view for a terminal, oldest turn first."""
    date = view.turns[0].timestamp[:10] if view.turns else ""
    shown = len(view.turns)

    # Channel ids are shown in full. Truncated to eight characters every one of
    # them renders as "channel_", so the header names a channel without saying
    # WHICH — and `channel_dev` and `channel_dm--atlas--opus` become
    # indistinguishable at exactly the moment the reader needs to tell them
    # apart. Session UUIDs stay short: eight characters identify those.
    label = (
        view.session_id if is_channel_session(view.session_id)
        else view.session_id[:8]
    )
    header = f"Session {label}"
    if date:
        header += f" · {date}"
    header += f" · showing {shown} of {view.total_turns} turns"
    if view.excluded_count:
        header += f" ({view.excluded_count} harness turns filtered)"

    if view.selection_scope == "agent":
        header += " · agent identity"
    elif view.selection_scope == "store":
        header += f" · store fallback from {view.source_label}"
    if view.caller_unindexed:
        newest = view.caller_unindexed[0]
        stamp = datetime.fromtimestamp(newest.mtime, timezone.utc).isoformat(timespec="seconds")
        header = (
            f"⚠ CALLER SOURCE STALE: {newest.session_id[:8]} "
            f"({stamp}, {newest.size} bytes) is newer and unindexed | " + header
        )
    if view.caller_partial:
        gap = view.caller_partial
        header = (
            f"⚠ CALLER SOURCE PARTIAL: {gap.source.session_id[:8]} "
            f"indexed through {gap.indexed_latest}, live through {gap.live_latest} | "
            "run `synapt recall build --no-embeddings` to refresh | " + header
        )
    if view.unclean_end:
        found = view.unclean_end
        header = (
            f"⚠ UNCLEAN END: last activity {found.last_activity}, "
            + (f"{_gap_phrase(found.gap_seconds)} after the last journal that could be its own, "
               if found.gap_seconds is not None else "no journal of its own covers it, ")
            + "no SessionEnd checkpoint for this session | " + header
        )

    lines = [header]
    lines.extend(_format_freshness(view))
    lines.extend(_format_journal(view))

    if not view.turns:
        lines.append("")
        lines.extend(_format_empty(view))
        return "\n".join(lines)

    for position, turn in enumerate(view.turns):
        lines.append("")
        # The anchored question sits before a gap. Saying so keeps the tail from
        # reading as contiguous when it is not.
        if position == 1 and view.omitted_between:
            lines.append(f"     … {view.omitted_between} intermediate turns omitted …")
            lines.append("")
        lines.append(f"── {turn.chunk_id} · {turn.timestamp}")
        # Two independent claims, deliberately not chained with elif: this
        # branch prints the continuation MARKER, and suppression of the
        # synthetic restatement is the view's job alone. Chaining them made the
        # earlier one mask the later, so whichever ran first absorbed the other's
        # witness and a mutation of either survived (found 2026-08-05).
        if turn.is_continuation:
            lines.append("  (continues the previous question)")
        if turn.user_text:
            lines.append(f"  YOU: {_clip(turn.user_text, max_chars)}")
        if turn.assistant_text:
            lines.append(f"  ASSISTANT: {_clip(turn.assistant_text, max_chars)}")
        elif not turn.is_continuation:
            lines.append("  ASSISTANT: (no reply — the session ended here)")
        if turn.tools_used:
            lines.append(f"  tools: {', '.join(turn.tools_used)}")

    return "\n".join(lines)
