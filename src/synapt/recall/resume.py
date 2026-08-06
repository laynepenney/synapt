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

Runtime independence is structural, not special-cased: this module reads chunks
and never a transcript, so anything the parsers produce — Claude Code sessions
via ``parse_transcript``, Codex CLI rollouts via ``parse_codex_transcript`` —
resumes identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from synapt.recall.freshness import IndexFreshness
from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.journal import JournalEntry, read_entries

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


def resolve_session(index: TranscriptIndex, session_id: str | None) -> str:
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
) -> ResumeView:
    """Assemble the tail of a session, newest last.

    ``journal_path`` of ``None`` means *do not read a journal* rather than *use
    the default*. Callers name the path they want, so this function performs no
    implicit I/O and tests cannot accidentally read a real journal.
    """
    if limit < 1:
        raise ResumeError(f"--turns must be at least 1 (got {limit}).")

    resolved = resolve_session(index, session_id)
    is_newest = bool(index._session_order) and index._session_order[0] == resolved

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

    return ResumeView(
        session_id=resolved,
        turns=turns,
        journal=journal,
        journal_provenance=provenance,
        total_turns=len(kept),
        excluded_count=len(chunks) - len(kept),
        omitted_between=omitted_between,
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




def _format_freshness(view: ResumeView) -> list[str]:
    """Disclose a stale index whether or not turns rendered.

    Turns shown from a stale index are still real; they may simply be missing
    the newest ones. Silence here would let a partial answer read as complete.
    """
    f = view.freshness
    if f is None or not f.stale:
        return []
    behind = len(f.new_files) + len(f.changed_files)
    noun = "file" if behind == 1 else "files"
    detail = f" ({behind} {noun} not yet indexed)" if behind else ""
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
        behind = len(f.new_files) + len(f.changed_files)
        noun = "file" if behind == 1 else "files"
        return [
            f"No turns found — but the index is STALE, so this is not an answer "
            f"about the session.",
            f"  {behind} archived {noun} {'is' if behind == 1 else 'are'} not in the index "
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
