"""Session journal — structured session logging with auto-extraction.

Storage: .synapt/recall/worktrees/<name>/journal.jsonl (per-worktree, append-only).
Each entry records what was done, key decisions, and next steps.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from synapt.recall.core import project_worktree_dir, project_index_dir, project_transcript_dir
from synapt.recall._llm_util import truncate_at_word as _tw

# Paths that are always noise in journal file lists (checked after normalising to /)
_NOISE_PATH_SEGMENTS = ("/.claude/", "/private/tmp/")


def _norm(p: str) -> str:
    """Normalise path separators to forward slash for cross-platform comparison."""
    return p.replace("\\", "/")


# --- Field collapse ----------------------------------------------------
#
# When a tool-call parameter's closing tag is dropped, the emitter consumes
# everything downstream into that parameter's value -- the following parameters
# *including their tags* and the invoke terminator.  The call still succeeds,
# so the trailing fields simply arrive empty and the entry reads as partially
# filled rather than malformed.  Nothing goes red anywhere.
#
# These literals cannot occur in ordinary prose, which makes the check
# deterministic rather than heuristic.  The alternative -- "warn when one field
# is long and the next is empty" -- false-positives on a legitimately empty
# field, and a check that cries wolf gets ignored.
COLLAPSE_SIGNATURES: tuple[str, ...] = (
    "</focus>",
    "</done>",
    "</decisions>",
    "</next_steps>",
    "</invoke>",
    "<parameter name=",
    # Shape C (real store, 2026-08-26): every closing tag dropped, so sibling
    # fields arrive as BARE OPENING tags inside the unclosed value.  Exact,
    # case-sensitive field names only; "<donee>" or "<Done>" do not match.
    "<focus>",
    "<done>",
    "<decisions>",
    "<next_steps>",
)

# The journal fields a collapse can carry, in schema order.
_TEXT_FIELDS = ("focus", "done", "decisions", "next_steps")

_HEAD_BOUNDARY = re.compile(r"</(?:focus|done|decisions|next_steps)>")

# Both shapes found in real stored data:
#   <parameter name="next_steps">VALUE</next_steps>
#   <next_steps>VALUE</next_steps>
_SEGMENT = re.compile(
    r'<parameter\s+name="(?P<pname>[a-z_]+)"\s*>(?P<pval>.*?)</(?P=pname)>'
    r"|<(?P<tag>[a-z_]+)>(?P<tval>.*?)</(?P=tag)>",
    re.DOTALL,
)

# Shape C: an opening field tag with no closer; the value runs to the next
# opening field tag or the end of the text.
_BARE_OPENER = re.compile(
    r"<(?P<name>focus|done|decisions|next_steps)>(?P<val>.*?)(?=<(?:focus|done|decisions|next_steps)>|\Z)",
    re.DOTALL,
)


class JournalFieldCollapse(ValueError):
    """A journal field carries tool-call markup from an unclosed parameter."""


def is_collapsed(text: object) -> bool:
    """True if *text* carries a tool-call parameter tag."""
    return isinstance(text, str) and any(sig in text for sig in COLLAPSE_SIGNATURES)


def recover_collapsed(text: str) -> tuple[str, dict[str, list[str]]]:
    """Split a collapsed value back into its parts.

    Returns ``(head, swallowed)`` where *head* is the value the field was
    actually meant to carry and *swallowed* maps field name to the values that
    were consumed into it.  Clean text recovers to itself with no swallowed
    fields, so this is safe to call unconditionally.

    Recovery rather than truncation is deliberate: the swallowed text is real
    work somebody wrote.  Cutting at the tag would satisfy a "no markup
    remains" check while silently discarding it.
    """
    if not is_collapsed(text):
        return text, {}

    boundary = _HEAD_BOUNDARY.search(text)
    if boundary:
        head, remainder = text[: boundary.start()], text[boundary.end():]
    else:
        # No named closing tag -- only a stray "</invoke>" or an opening
        # parameter tag.  Cut at the first signature we can find.
        cut = min(text.find(s) for s in COLLAPSE_SIGNATURES if s in text)
        head, remainder = text[:cut], text[cut:]

    swallowed: dict[str, list[str]] = {}
    for match in _SEGMENT.finditer(remainder):
        name = match.group("pname") or match.group("tag")
        value = match.group("pval") if match.group("pname") else match.group("tval")
        if name not in _TEXT_FIELDS:
            continue
        value = (value or "").strip()
        if value:
            swallowed.setdefault(name, []).append(value)
    if not swallowed:
        # Shape C: bare openers, no closers.  Each field runs from its opening
        # tag to the next opening tag (or the end).
        for match in _BARE_OPENER.finditer(remainder):
            value = match.group("val").strip()
            if value:
                swallowed.setdefault(match.group("name"), []).append(value)
    return head.strip(), swallowed


def _entry_collapses(entry: JournalEntry) -> list[tuple[str, str]]:
    """Return ``(field_name, value)`` for every collapsed value on *entry*."""
    found: list[tuple[str, str]] = []
    for name in _TEXT_FIELDS:
        raw = getattr(entry, name, None)
        values = [raw] if isinstance(raw, str) else (raw or [])
        for value in values:
            if is_collapsed(value):
                found.append((name, value))
    return found


def split_journal_field(text: str) -> list[str]:
    """Split a journal field into items, preferring newlines over semicolons.

    Semicolons are ordinary punctuation in the prose these fields carry, so
    splitting on them silently fragments sentences and strips their subject --
    an entry that reads as terse notes rather than as damage. Newlines are how
    the fields are actually written, one item per line.

    Newline-first keeps every existing single-line caller working unchanged:
    with no newline present the semicolon behaviour is exactly what it was.
    """
    if "\n" in text:
        parts = text.split("\n")
    else:
        parts = text.split(";")
    return [item.strip() for item in parts if item.strip()]


def _served(values: list[str]) -> list[str]:
    """Drop collapsed values from anything about to be displayed.

    A collapsed value stays in the store -- it is the resolution marker that
    breaks the carry-forward loop -- but it is never shown to anyone.
    """
    return [v for v in values if not is_collapsed(v)]


def _filter_project_files(
    files: list[str] | set[str],
    project_root: str | None = None,
) -> list[str]:
    """Filter file paths to only include project-relevant files.

    Removes:
    - ~/.claude/ internals (plans, tool results, hooks, settings)
    - /private/tmp/ temp files
    - Absolute paths outside the project root (other projects)

    Relative paths are kept as-is (already project-scoped).
    Absolute paths under project_root are converted to relative.
    """
    if not project_root:
        project_root = str(Path.cwd())
    # Normalise to forward slashes for cross-platform prefix matching
    prefix = _norm(project_root).rstrip("/") + "/"

    result: set[str] = set()
    for fp in files:
        if not fp or fp.isspace():
            continue
        norm_fp = _norm(fp)
        # Skip known noise paths
        if any(seg in norm_fp for seg in _NOISE_PATH_SEGMENTS):
            continue
        # Relative paths are already project-scoped (no leading / or drive letter)
        if not os.path.isabs(fp):
            result.add(fp)
            continue
        # Absolute paths: keep only if under project root
        if norm_fp.startswith(prefix):
            result.add(norm_fp[len(prefix):])
        # else: absolute path outside project — skip
    return sorted(result)


def _journal_path(project_dir: Path | None = None) -> Path:
    """Return path to journal.jsonl for this worktree.

    Journal is per-worktree — each worktree tracks its own session history.
    Lives at ``<main>/.synapt/recall/worktrees/<name>/journal.jsonl``.
    """
    return project_worktree_dir(project_dir) / "journal.jsonl"


@dataclass
class JournalEntry:
    """One session's journal entry."""

    timestamp: str  # ISO 8601
    session_id: str = ""
    branch: str = ""
    focus: str = ""  # What this session was about
    done: list[str] = field(default_factory=list)  # What got accomplished
    decisions: list[str] = field(default_factory=list)  # Key decisions made
    next_steps: list[str] = field(default_factory=list)  # What to do next session
    files_modified: list[str] = field(default_factory=list)
    git_log: list[str] = field(default_factory=list)  # Recent commits
    auto: bool = False       # True if synthesized at build time
    enriched: bool = False   # True if LLM-enriched
    griptree: str = ""       # Agent's griptree identity (e.g., "synapt/synapt")
    agent_id: str = ""       # Agent's session-scoped ID (e.g., "s_a1b2c3d4")
    repair: bool = False     # True if written by repair_journal

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> JournalEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def has_content(self) -> bool:
        """True if the entry has any user-provided or auto-extracted content."""
        return bool(
            self.focus or self.done or self.decisions or self.next_steps
            or self.files_modified
        )

    def has_rich_content(self) -> bool:
        """True if entry has semantic content beyond just file paths."""
        return bool(self.focus or self.done or self.decisions or self.next_steps)


def append_entry(
    entry: JournalEntry,
    path: Path | None = None,
    allow_collapsed: bool = False,
) -> Path:
    """Append a journal entry to the JSONL file.

    Uses exclusive file locking to prevent interleaved writes
    when multiple processes append concurrently (e.g., background enrich
    + SessionEnd hook).

    Raises :class:`JournalFieldCollapse` if any text field carries tool-call
    markup.  This is the single write point, so no caller can bypass it.
    Refusing beats accepting: a collapsed value that lands in ``next_steps``
    can never be matched by a ``done`` item, so carry-forward re-injects it
    every session and it becomes immortal.  The caller still holds the content
    and can resend it correctly; the store cannot un-replicate it later.

    *allow_collapsed* exists for :func:`repair_journal`, whose corrective entry
    must reference the original text verbatim to mark it resolved.
    """
    if not allow_collapsed:
        collapses = _entry_collapses(entry)
        if collapses:
            field, value = collapses[0]
            sig = next(s for s in COLLAPSE_SIGNATURES if s in value)
            raise JournalFieldCollapse(
                f"journal field {field!r} carries tool-call markup {sig!r} — an "
                f"unclosed parameter swallowed the fields after it, so the "
                f"trailing fields are empty rather than missing. Resend with "
                f"shorter field values."
            )
    path = path or _journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    from synapt.recall._filelock import lock_exclusive
    with open(path, "a", encoding="utf-8") as f:
        lock_exclusive(f)
        f.write(json.dumps(entry.to_dict()) + "\n")
        f.flush()
        # lock released on close
    return path


def read_latest(path: Path | None = None, meaningful: bool = False) -> JournalEntry | None:
    """Read the most recent journal entry.

    If *meaningful* is True, skip auto-extracted entries that have no
    focus/done/decisions/next — these are noise from the SessionEnd hook.
    """
    if not meaningful:
        entries = read_entries(path, n=1)
        return entries[0] if entries else None
    # Already deduped+sorted newest-first, so first with rich content wins
    for entry in read_entries(path, n=50):
        if entry.focus or entry.done or entry.decisions or entry.next_steps:
            return entry
    return None


def read_previous_meaningful(
    current_session_id: str = "",
    path: Path | None = None,
) -> JournalEntry | None:
    """Read the most recent meaningful entry from a prior session.

    If *current_session_id* is provided, entries from the same session are
    skipped so repeated writes do not carry forward their own next steps.
    """
    for entry in read_entries(path, n=50):
        if not entry.has_rich_content():
            continue
        if current_session_id and entry.session_id == current_session_id:
            continue
        return entry
    return None


def read_entries(path: Path | None = None, n: int = 5) -> list[JournalEntry]:
    """Read the last N journal entries (most recent first).

    Deduplicates by session_id (keeps the newest manual/auto entry; richness
    breaks an exact-timestamp tie)
    and sorts by timestamp descending. Does NOT assume the file is
    chronologically ordered.
    """
    path = path or _journal_path()
    if not path.exists():
        return []
    raw = _read_all_entries(path)
    deduped = _dedup_entries(raw)
    deduped.sort(key=lambda e: _timestamp_order(e.timestamp), reverse=True)
    return deduped[:n]


def _read_all_entries(path: Path) -> list[JournalEntry]:
    """Read every entry from a journal JSONL file."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(JournalEntry.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return entries


def _entry_richness(entry: JournalEntry) -> tuple:
    """Score an entry for dedup ranking.

    Manual entries still beat auto-extracted stubs, but within either class the
    newest write wins before field count. A resumed runtime can reuse a session
    id: letting an older, richer entry win there retains completed work and
    hides the current entry's next steps, which are the continuity handoff.

    Returns a tuple that sorts higher for the retained entry:
    (not auto, normalized timestamp, rich field count).
    """
    rich_count = sum(bool(f) for f in (entry.focus, entry.done, entry.decisions, entry.next_steps))
    return (not entry.auto, _timestamp_order(entry.timestamp), rich_count)


def _timestamp_order(timestamp: str) -> datetime:
    """Return a deterministic chronological ordering key for a journal timestamp.

    Offset-aware values are normalized to UTC. Legacy offset-naive values are
    interpreted as UTC because their original timezone is not recoverable from
    disk. An unparseable legacy value sorts before any parseable timestamp so
    it cannot displace a known newer journal entry.
    """
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.astimezone(timezone.utc)
    except OverflowError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _dedup_entries(entries: list[JournalEntry]) -> list[JournalEntry]:
    """Keep the highest-priority entry per session_id.

    Priority: non-auto > auto, then newest timestamp, then most rich fields
    (focus/done/decisions/next_steps).
    Entries without a session_id are kept as-is.
    """
    best: dict[str, JournalEntry] = {}
    no_sid: list[JournalEntry] = []
    for entry in entries:
        if not entry.session_id:
            no_sid.append(entry)
            continue
        existing = best.get(entry.session_id)
        if existing is None or _entry_richness(entry) > _entry_richness(existing):
            best[entry.session_id] = entry
    return list(best.values()) + no_sid


def compact_journal(path: Path | None = None) -> int:
    """Physically dedup and sort journal.jsonl.

    Reads all entries, deduplicates by session_id (keeps newest within the
    manual/auto class, with richness as an exact-timestamp tie-breaker),
    sorts chronologically, and rewrites the file in-place.

    Uses an exclusive flock on the journal file itself (not a temp file)
    so concurrent append_entry calls block for the entire read-dedup-rewrite
    cycle and cannot sneak in entries that would be silently dropped.

    Note: entries with session_id="" are kept as-is and are never
    deduplicated against each other — only entries sharing a non-empty
    session_id are collapsed.

    Returns the number of duplicate entries removed.
    """
    path = path or _journal_path()
    if not path.exists():
        return 0
    # Open in "r+" (read-write, no truncate) so we can hold LOCK_EX on the
    # exact file descriptor we will rewrite.  All concurrent append_entry
    # callers flock the same path and will block until we close this fd.
    from synapt.recall._filelock import lock_exclusive
    with open(path, "r+", encoding="utf-8") as f:
        lock_exclusive(f)
        # Check for empty file AFTER acquiring the lock.  stat() before open()
        # is a TOCTOU race: a concurrent append_entry could write between the
        # stat() and the flock(), causing us to miss newly written entries.
        # seek(0, 2) in text mode is CPython/POSIX-specific (delegates to the
        # underlying binary fd's seek, well-defined on POSIX).  Not guaranteed
        # by the Python language spec but reliable on all supported platforms
        # (macOS/Linux, CPython ≥ 3.9).
        if f.seek(0, 2) == 0:  # seek to EOF; position 0 means file is empty
            return 0
        f.seek(0)  # reset to start for reading
        entries = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(JournalEntry.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        deduped = _dedup_entries(entries)
        removed = len(entries) - len(deduped)
        if removed == 0:
            return 0
        deduped.sort(key=lambda e: _timestamp_order(e.timestamp))
        f.seek(0)
        f.truncate(0)  # explicit arg: truncate to zero bytes regardless of buffer position
        for entry in deduped:
            f.write(json.dumps(entry.to_dict()) + "\n")
        f.flush()
    return removed


def format_for_session_start(entry: JournalEntry) -> str:
    """Format a journal entry's key info for hook output at session start."""
    lines = []
    ts = entry.timestamp[:16]  # Trim to minute precision

    # Collapsed values stay in the store as resolution markers but are never
    # displayed — see _served().
    if entry.focus and not is_collapsed(entry.focus):
        lines.append(f"Last session ({ts}): {entry.focus}")

    # Open threads FIRST. This read is BOUNDED, so ordering is not taste here --
    # whatever leads consumes the window. Completed work is recoverable from git
    # and the board; an unrecorded open question is recoverable from nowhere.
    for label, items in (
        ("Next steps:", _served(entry.next_steps)),
        ("Decisions:", _served(entry.decisions)),
        ("Done:", _served(entry.done)),
    ):
        if items:
            lines.append(label)
            lines.extend(f"  - {item}" for item in items)

    return "\n".join(lines) if lines else ""


def format_entry_full(entry: JournalEntry) -> str:
    """Format a journal entry for display (journal --show)."""
    lines = [f"## {entry.timestamp[:16]}"]
    if entry.branch:
        lines.append(f"**Branch:** {entry.branch}")
    if entry.focus and not is_collapsed(entry.focus):
        lines.append(f"**Focus:** {entry.focus}")
    # Same order as the session-start read. This surface is unbounded, so the
    # ordering is not forced here -- but two surfaces that teach different
    # priorities are their own defect, and a human reading top-down also stops.
    for heading, items in (
        ("\n### Next", _served(entry.next_steps)),
        ("\n### Decisions", _served(entry.decisions)),
        ("\n### Done", _served(entry.done)),
    ):
        if items:
            lines.append(heading)
            lines.extend(f"- {item}" for item in items)
    if entry.files_modified:
        lines.append(f"\n### Files ({len(entry.files_modified)})")
        for f in entry.files_modified[:15]:
            lines.append(f"- {f}")
        if len(entry.files_modified) > 15:
            lines.append(f"  ... and {len(entry.files_modified) - 15} more")
    if entry.git_log:
        lines.append("\n### Commits")
        for c in entry.git_log:
            lines.append(f"- {c}")
    return "\n".join(lines)


def format_write_confirmation(
    entry: JournalEntry,
    explicit_next_steps: list[str] | None = None,
    report: "CarryReport | None" = None,
) -> str:
    """Format a write response without conflating carried-forward work.

    Journal storage keeps unresolved prior next steps on the new entry so they
    remain visible next session.  The write response needs a clearer UX: steps
    supplied by the caller stay under ``Next``; unresolved prior steps are
    rendered under a separate carry-forward heading that, when a ``report`` is
    given, states what the filter did (carried / retired by done / withheld)
    and how to retire a carried step.
    """
    explicit_steps = [
        step.strip()
        for step in (explicit_next_steps or [])
        if step and step.strip()
    ]
    explicit_keys = {_step_key(step) for step in explicit_steps}
    if not explicit_keys:
        display_next: list[str] = []
        carried = [step for step in entry.next_steps if step and step.strip()]
    else:
        display_next = []
        carried = []
        for step in entry.next_steps:
            if not step or not step.strip():
                continue
            if _step_key(step) in explicit_keys:
                display_next.append(step)
            else:
                carried.append(step)

    text = format_entry_full(replace(entry, next_steps=display_next))
    parts = [text]
    if report is not None:
        # Always, zeros included: a filter that retired something and a filter
        # that had nothing to do must not render identically (recall#984).
        parts.append(f"\nCarry-forward: {report.carried} carried, "
                     f"{report.retired_by_done} retired by done, {report.withheld} withheld.")
    if carried:
        parts.append("\n### Carried Forward Next Steps")
        parts.extend(f"- {step}" for step in carried)
        parts.append("To retire a carried step next time, list its exact text under done.")
    return "\n".join(part for part in parts if part)


# recall#984: a carried step carries its age, the carry is bounded, and the bound
# announces itself.  Retirement stays EXACT on purpose: a near-miss must never
# retire (a wrongly retired step is gone; a stale one is at least visible).
CARRY_LIMIT = 20
WITHHELD_PREFIX = "[carry bound]"
_STAMP_RE = re.compile(r"\s*\[carried since (\d{4}-\d{2}-\d{2})\]\s*$")


def strip_carry_stamp(step: str) -> tuple[str, str | None]:
    """Split ``"text [carried since YYYY-MM-DD]"`` into ``(text, date)``.

    Returns ``(step, None)`` when there is no stamp.
    """
    m = _STAMP_RE.search(step)
    if not m:
        return step.strip(), None
    return step[: m.start()].strip(), m.group(1)


def is_withheld_marker(step: str) -> bool:
    """True for the synthetic line that announces a bounded carry."""
    return step.strip().startswith(WITHHELD_PREFIX)


def _step_key(step: str) -> str:
    """Normalize a next-step string for exact matching (the age stamp is ignored,
    so ``done`` can name a carried step with or without its stamp; a leading
    ``"- "`` bullet is also ignored, since the tool's own carry-forward
    response renders each carried step as ``f"- {step}"`` -- recall#984:
    copying that displayed line verbatim, exactly as the tool's own
    instruction says to, must not silently fail to retire because of a
    formatting artifact the tool itself introduced)."""
    bare, _ = strip_carry_stamp(step)
    bare = bare.strip()
    if bare.startswith("- "):
        bare = bare[2:]
    return " ".join(bare.split()).casefold()


@dataclass
class CarryReport:
    """What the carry-forward filter did, so a filter that removed nothing can
    say so instead of looking identical to one with nothing to remove."""

    carried: int = 0
    retired_by_done: int = 0
    withheld: int = 0
    oldest_since: str | None = None


def merge_carried_forward_with_report(
    current_next_steps: list[str],
    current_done: list[str],
    previous_entry: JournalEntry | None,
) -> tuple[list[str], CarryReport]:
    """Merge unresolved prior-session next steps into the current entry.

    Carries forward prior next steps unless the current entry already includes
    them or marks them done (exact text, stamp ignored).  New next steps stay
    first; carried items append in the previous entry's order, each stamped
    ``[carried since YYYY-MM-DD]`` with the date of the entry it was first
    written in (an existing stamp is preserved, never refreshed).  At most
    ``CARRY_LIMIT`` carried items are kept; the rest are withheld and a final
    marker line says how many and the oldest date, so the truncation is visible.
    """
    report = CarryReport()
    merged = [
        step.strip()
        for step in current_next_steps
        if step and step.strip() and not is_collapsed(step) and not is_withheld_marker(step)
    ]
    seen = {_step_key(step) for step in merged}
    # Matching still uses the RAW done list: a collapsed value recorded there
    # by repair_journal is exactly what marks the original step resolved.
    done = {_step_key(item) for item in current_done if item and item.strip()}

    if not previous_entry or not previous_entry.next_steps:
        return merged, report

    origin_date = (previous_entry.timestamp or "")[:10]
    carried: list[tuple[str, str]] = []  # (stamped text, since)
    for step in previous_entry.next_steps:
        clean = step.strip()
        if not clean or is_withheld_marker(clean):
            # The marker describes the PREVIOUS carry; it is regenerated below.
            continue
        if is_collapsed(clean):
            # The replication vector: a malformed step can never appear in a
            # done list, so without this it rides forward every session.
            continue
        key = _step_key(clean)
        if key in seen:
            continue
        if key in done:
            report.retired_by_done += 1
            continue
        bare, since = strip_carry_stamp(clean)
        since = since or origin_date
        stamped = f"{bare} [carried since {since}]" if since else bare
        carried.append((stamped, since))
        seen.add(key)

    kept, withheld = carried[:CARRY_LIMIT], carried[CARRY_LIMIT:]
    merged.extend(text for text, _ in kept)
    report.carried = len(kept)
    report.withheld = len(withheld)
    dates = sorted(d for _, d in carried if d)
    report.oldest_since = dates[0] if dates else None
    if withheld:
        oldest_withheld = min((d for _, d in withheld if d), default=None)
        marker = f"{WITHHELD_PREFIX} {len(withheld)} carried steps withheld"
        if oldest_withheld:
            marker += f"; oldest since {oldest_withheld}"
        marker += "; recall_journal action=pending lists every unresolved step"
        merged.append(marker)
    return merged, report


def merge_carried_forward_next_steps(
    current_next_steps: list[str],
    current_done: list[str],
    previous_entry: JournalEntry | None,
) -> list[str]:
    """Back-compatible wrapper: the merged list only (see the ``_with_report`` form)."""
    merged, _ = merge_carried_forward_with_report(current_next_steps, current_done, previous_entry)
    return merged


def pending_next_steps(path: Path | None = None) -> list[str]:
    """Return unresolved next_steps from recent journal entries.

    Scans all next_steps across recent entries and returns those that
    have not appeared in any entry's ``done`` list.  Deduplicates by
    normalized key, preserving the most recent wording.
    """
    entries = read_entries(path, n=20)
    if not entries:
        return []

    # Collect all done items from all recent entries
    all_done = set()
    for entry in entries:
        for item in entry.done:
            if item and item.strip():
                all_done.add(_step_key(item))

    # Collect all next_steps across entries (newest first), dedup by key
    seen: set[str] = set()
    pending: list[str] = []
    for entry in entries:
        for step in entry.next_steps:
            if not step or not step.strip():
                continue
            if is_collapsed(step):
                continue  # never serve tool-call markup
            if is_withheld_marker(step):
                continue  # the bound's own announcement is not a step
            key = _step_key(step)
            if key in all_done or key in seen:
                continue
            pending.append(step)
            seen.add(key)

    return pending


def repair_journal(path: Path | None = None, dry_run: bool = False) -> dict:
    """Recover content swallowed by collapsed fields, append-only.

    Appends ONE corrective entry that carries the recovered text in its proper
    fields and lists every collapsed value verbatim under ``done``.  That
    marking is not bookkeeping: ``done`` is the store's own "this is resolved"
    grammar, and it is what stops carry-forward from re-injecting a malformed
    step next session.  Nothing is rewritten or deleted — the original lines
    stay exactly as written.

    Returns a report; with *dry_run* the report is produced and nothing is
    written.
    """
    path = Path(path) if path else _journal_path()
    report = {
        "path": str(path),
        "root": "",
        "line_count": 0,
        "total_entries": 0,
        "contaminated_entries": 0,
        "repaired_entries": 0,
        "recovered_fields": {},
        "dry_run": dry_run,
    }
    if not path.exists():
        return report

    # Raw line count is reported alongside the parsed entry count on purpose.
    # Unparseable lines are skipped silently, so a store whose line count
    # exceeds its entry count has lost records that no other number reveals.
    with open(path, encoding="utf-8") as f:
        report["line_count"] = sum(1 for line in f if line.strip())

    entries = _read_all_entries(path)
    report["total_entries"] = len(entries)

    # Everything a previous repair pass already accounted for.
    already = {
        _step_key(value)
        for entry in entries
        if entry.repair
        for value in entry.done
    }

    collapsed_values: list[str] = []
    recovered: dict[str, list[str]] = {}
    contaminated = 0
    for entry in entries:
        if entry.repair:
            continue
        hits = _entry_collapses(entry)
        if not hits:
            continue
        contaminated += 1
        for field, value in hits:
            if _step_key(value) in already:
                continue
            collapsed_values.append(value)
            head, swallowed = recover_collapsed(value)
            if head:
                recovered.setdefault(field, []).append(head)
            for name, values in swallowed.items():
                recovered.setdefault(name, []).extend(values)

    report["contaminated_entries"] = contaminated
    report["repaired_entries"] = len(collapsed_values)
    report["recovered_fields"] = {k: len(v) for k, v in recovered.items()}
    if dry_run or not collapsed_values:
        return report

    # The recovered text must itself be clean — a repair pass that injected
    # what the guard refuses would be laundering the contamination.
    def clean(name: str) -> list[str]:
        return [v for v in recovered.get(name, []) if not is_collapsed(v)]

    # Parsing a field is not recovering it (Atlas, r2 on v1 of shape C): the
    # report counted a swallowed <done> that this entry then never carried.
    # Every recovered field lands on the corrective entry's OWN surface for
    # that field. ``done`` keeps the contaminated values FIRST and verbatim --
    # they are the loop-breaking marker -- and the clean recovered done values
    # follow them. ``focus`` is a scalar, so recovered focus text is appended
    # to the repair's own focus rather than replacing it.
    focus = "Journal field-collapse repair"
    recovered_focus = clean("focus")
    if recovered_focus:
        focus += " (recovered focus: " + " | ".join(recovered_focus) + ")"
    corrective = JournalEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        focus=focus,
        done=collapsed_values + clean("done"),
        decisions=clean("decisions"),
        next_steps=clean("next_steps"),
        repair=True,
    )
    append_entry(corrective, path, allow_collapsed=True)
    return report


def format_repair_report(report: dict) -> str:
    """Render a repair report — always naming the store it examined.

    The empty case is the one that needs the path most.  A bare "nothing to
    repair" is indistinguishable from having examined the wrong store, and
    that is not hypothetical: the command resolves its data root from the
    working directory, and a desk whose writes land under a different root
    gets a clean, confident, false report.  Naming the path turns an
    unfalsifiable claim into an auditable one.

    Line count is reported next to entry count because they can disagree:
    unparseable lines are skipped silently, so ``lines > entries`` is the only
    signal that records were dropped.
    """
    path = report.get("path") or "<no store>"
    root = report.get("root")
    where = f"{path}" if not root else f"{path}  [root {root}]"
    lines = report.get("line_count", 0)
    total = report.get("total_entries", 0)

    if not total:
        return f"{where}: 0 entries ({lines} lines) — nothing examined"

    counted = f"{total} entries / {lines} lines"
    if lines != total:
        counted += f" — {lines - total} unparseable line(s) skipped"

    bad = report.get("contaminated_entries", 0)
    if not bad:
        return f"{where}: {counted}, 0 contaminated"

    verb = "would recover" if report.get("dry_run") else "recovered"
    fields = report.get("recovered_fields") or {}
    return (
        f"{where}: {counted}, {bad} contaminated, "
        f"{verb} {report.get('repaired_entries', 0)} value(s) {fields}"
    ).rstrip()


def sweep_stores(root: Path | str, dry_run: bool = False) -> list[dict]:
    """Repair every per-worktree journal under *root*, naming each one.

    Returns one report per store.  A root holding no stores still returns a
    single report carrying the root, because "swept and found nothing" and
    "swept the wrong root" are otherwise the same sentence.
    """
    root = Path(root)
    reports: list[dict] = []
    for store in sorted((root / "worktrees").glob("*/journal.jsonl")):
        report = repair_journal(store, dry_run=dry_run)
        report["root"] = str(root)
        reports.append(report)
    if not reports:
        reports.append({
            "path": str(root / "worktrees"),
            "root": str(root),
            "line_count": 0,
            "total_entries": 0,
            "contaminated_entries": 0,
            "repaired_entries": 0,
            "recovered_fields": {},
            "dry_run": dry_run,
        })
    return reports


# How far extract_session_id will look before giving up. Modern transcripts
# carry the id on line 1; legacy ones within the first few dozen lines. The
# bound is generous for that and still O(1) in transcript size.
SESSION_ID_SCAN_MAX_LINES = 2000
SESSION_ID_SCAN_MAX_BYTES = 8_000_000


def _session_id_from_record(d: object) -> str:
    """The session id one transcript line carries, in every format met so far.

    * Claude Code, current: any record with a ``sessionId`` string — the
      first is a ``custom-title`` on line 1.
    * Claude Code, legacy: ``progress`` records (also ``sessionId``).
    * Codex archives: ``session_meta.payload.id``.
    """
    if not isinstance(d, dict):
        return ""
    if d.get("type") == "session_meta":
        payload = d.get("payload")
        sid = payload.get("id", "") if isinstance(payload, dict) else ""
        return sid if isinstance(sid, str) else ""
    sid = d.get("sessionId")
    return sid if isinstance(sid, str) else ""


def extract_session_id(
    path: Path | str,
    max_lines: int = SESSION_ID_SCAN_MAX_LINES,
    max_bytes: int = SESSION_ID_SCAN_MAX_BYTES,
) -> str:
    """Read the session id from a transcript, scanning a BOUNDED prefix.

    Returns "" when no line in the first *max_lines* lines / *max_bytes*
    bytes carries an id, or the file cannot be read.

    Why bounded, and why any-line: this used to accept only ``progress`` and
    ``session_meta`` records. Modern transcripts have neither — the id sits on
    line 1 in a ``custom-title`` record — so on 111 of 160 archived files the
    scan read the ENTIRE file (1.4 GB in one case) at every session start,
    returned "", never journaled the session, and did it all again next
    start. That is where the session-start hook's 60s timeout went. A miss
    that stops at the bound is the only kind that cannot become that.
    """
    seen_bytes = 0
    try:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, start=1):
                if n > max_lines:
                    break
                seen_bytes += len(line)
                if seen_bytes > max_bytes:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                sid = _session_id_from_record(d)
                if sid:
                    return sid
    except OSError:
        pass
    return ""


def _transcript_sort_key(f: Path) -> tuple:
    """Sort key for transcript JSONL files: (mtime, name) descending.

    Wraps stat() in a try/except to guard against a TOCTOU race where a file
    is deleted between glob() and stat() — which would raise FileNotFoundError
    inside sorted().  Deleted files fall to the end of the sort (mtime=0.0).
    The filename tiebreaker makes the sort deterministic when two files share
    the same 1-second mtime granularity.
    """
    try:
        return (f.stat().st_mtime, f.name)
    except OSError:
        return (0.0, f.name)


def latest_transcript_path(project: Path | None = None) -> str | None:
    """Find the most recently modified transcript file for the project."""
    project = (project or Path.cwd()).resolve()
    candidates: list[Path] = []

    transcript_dir = project_transcript_dir(project)
    if transcript_dir:
        candidates.extend(transcript_dir.glob("*.jsonl"))

    try:
        from synapt.recall.codex import discover_codex_sessions, list_codex_transcripts
        codex_dir = discover_codex_sessions()
        if codex_dir:
            candidates.extend(list_codex_transcripts(codex_dir, project_dir=project))
    except Exception:
        pass

    if not candidates:
        return None

    jsonl_files = sorted(candidates, key=_transcript_sort_key, reverse=True)
    return str(jsonl_files[0]) if jsonl_files else None


def _get_branch(cwd: str) -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _get_recent_commits(cwd: str, n: int = 5) -> list[str]:
    """Get last N git commits as one-liners."""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "log", "--oneline", f"-{n}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except Exception:
        return []


def auto_extract_entry(
    transcript_path: str | Path | None = None,
    cwd: str | None = None,
) -> JournalEntry:
    """Build a journal entry by extracting context from the environment.

    Populates branch, files_modified, and git_log automatically.
    The focus/done/decisions/next fields are left empty for the model
    (or user) to fill via CLI flags or MCP tool.
    """
    cwd = cwd or str(Path.cwd())
    now = datetime.now(timezone.utc)

    session_id = ""
    files_modified: list[str] = []

    # Parse transcript if available
    if transcript_path and Path(transcript_path).exists():
        files_set: set[str] = set()
        try:
            from synapt.recall.codex import is_codex_transcript, parse_codex_transcript
            is_codex = is_codex_transcript(Path(transcript_path))
        except Exception:
            is_codex = False

        if is_codex:
            chunks = parse_codex_transcript(Path(transcript_path))
            if chunks:
                session_id = chunks[0].session_id
            for chunk in chunks:
                files_set.update(chunk.files_touched)
        else:
            with open(transcript_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not session_id:
                        session_id = _session_id_from_record(d)
                    if d.get("type") == "assistant":
                        for block in d.get("message", {}).get("content", []) or []:
                            if not isinstance(block, dict):
                                continue
                            fp = block.get("input", {}).get("file_path", "") if isinstance(block.get("input"), dict) else ""
                            if fp:
                                files_set.add(fp)
        files_modified = _filter_project_files(files_set, project_root=cwd)

    # Capture agent identity if channel system is available
    griptree = ""
    agent_id = ""
    try:
        from synapt.recall.channel import _resolve_griptree, _agent_id
        griptree = _resolve_griptree()
        agent_id = _agent_id()
    except Exception:
        pass  # Channel module may not be available

    return JournalEntry(
        timestamp=now.isoformat(),
        session_id=session_id,
        branch=_get_branch(cwd),
        files_modified=files_modified,
        git_log=_get_recent_commits(cwd),
        auto=True,  # Auto-extracted; cleared if user adds rich content
        griptree=griptree,
        agent_id=agent_id,
    )


def _read_all_session_ids(journal_path: Path) -> set[str]:
    """Read all session_ids from journal.jsonl."""
    ids: set[str] = set()
    if not journal_path.exists():
        return ids
    with open(journal_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                sid = d.get("session_id", "")
                if sid:
                    ids.add(sid)
            except (json.JSONDecodeError, TypeError):
                continue
    return ids


def synthesize_journal_stubs(
    sessions: dict,
    journal_path: Path | None = None,
    project_root: str | None = None,
) -> int:
    """Synthesize lightweight journal entries for sessions without one.

    Scans the session grouping for sessions that have no corresponding
    journal entry. For each, creates a stub with:
      - focus: first user message (turn 0)
      - files_modified: union of files_touched across all turns
      - auto=True tag for later LLM enrichment

    Args:
        sessions: dict mapping session_id → list of TranscriptChunks.
        journal_path: Path to journal.jsonl. Default: project journal path.
        project_root: Project root for filtering file paths. Default: cwd.

    Returns:
        Number of stubs synthesized.
    """
    from synapt.recall.scrub import strip_system_artifacts

    journal_path = journal_path or _journal_path()
    existing_ids = _read_all_session_ids(journal_path)

    count = 0
    for session_id, chunks in sorted(sessions.items()):
        if session_id in existing_ids:
            continue

        # Only consider transcript chunks (turn_index >= 0)
        transcript_chunks = [c for c in chunks if c.turn_index >= 0]
        if not transcript_chunks:
            continue

        # Find turn 0 for focus (strip system artifacts for already-indexed data)
        sorted_chunks = sorted(transcript_chunks, key=lambda c: c.turn_index)
        focus = ""
        for chunk in sorted_chunks:
            msg = strip_system_artifacts(chunk.user_text.strip())
            if msg:
                focus = _tw(msg, 200)
                break

        # Collect all files touched, filtering noise paths
        files_set: set[str] = set()
        for c in transcript_chunks:
            files_set.update(c.files_touched)
        files_modified = _filter_project_files(files_set, project_root=project_root)

        # Use earliest timestamp
        timestamp = min(
            (c.timestamp for c in transcript_chunks if c.timestamp),
            default="",
        )
        if not timestamp:
            continue

        entry = JournalEntry(
            timestamp=timestamp,
            session_id=session_id,
            focus=focus,
            files_modified=files_modified,
            auto=True,
        )
        if entry.has_content():
            append_entry(entry, journal_path)
            count += 1

    return count
