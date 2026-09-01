"""The session-start wake: what a fresh session sees first, and the hook's
own run log.

Two facts shape everything here (measured 2026-08-10 and 2026-08-25):

* The harness previews roughly the first 2 KB of a SessionStart hook's
  stdout and saves the rest to a file it does not tell the model to read.
  So the FIRST 2 KB is the whole briefing for most sessions, and whatever
  comes first consumes it.
* A hook killed at its timeout emits NOTHING, and nothing anywhere says so.
  The fresh session starts with no continuity context and no signal that
  any was missing — the failure that looks exactly like "nothing to say".

So the wake is rendered inside a fixed byte budget with a head line first,
the full text is written to disk and pointed at, and every run is recorded
started-then-completed so the NEXT run can report a predecessor that never
finished.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from synapt.recall.core import project_data_dir, project_worktree_dir

# Total bytes the hook may print. The harness preview is ~2 KB; the rest is
# saved to a file. 24 KB keeps the saved file useful without letting one
# source (a 50 KB journal entry, a 50 KB reminder hoard) crowd out the others.
WAKE_BUDGET_BYTES = 24_000

# What the harness shows inline. The head line and the start of the open
# threads must fit here.
PREVIEW_BYTES = 2048

HOOK_RUN_LOG_MAX_RECORDS = 100

# Per-source byte caps. They sum to less than WAKE_BUDGET_BYTES by
# construction, so the budget holds without a second pass; the final guard in
# render_wake exists for the day someone adds a source and forgets this table.
_CAP_UNCLEAN_END = 2_500
_CAP_CHECKPOINT = 2_500
_CAP_COMPACTION = 4_000
_CAP_JOURNAL_LATEST = 6_000
_CAP_JOURNAL_OLDER = 1_500
_CAP_BRANCH = 1_200
_CAP_OPEN_PR = 400
_CAP_CHANNEL = 4_000
_CAP_DIRECTIVES = 3_000
_CAP_KNOWLEDGE = 1_500
_CAP_REMINDER_ITEMS = 8
_CAP_CONTRADICTIONS = 1_500
_CAP_OTHER = 1_000


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def hook_run_log_path() -> Path:
    """Append-only record of hook runs, shared across worktrees of a project."""
    return project_data_dir() / "hook-runs.jsonl"


def wake_file_path(project: Path | None = None) -> Path:
    """Where the FULL (unclipped) wake text is written, per worktree."""
    return project_worktree_dir(project) / "wake" / "latest.md"


# ---------------------------------------------------------------------------
# run log
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


@contextmanager
def _log_lock(path: Path):
    """Exclusive lock on a sidecar file for every write to the run log.

    Simultaneous session starts are the production case (the detached
    catchup is single-flight for exactly that reason), and the first form of
    this log was an unlocked read-modify-replace through one shared ``.tmp``
    path: Atlas's r2 probe kept 2 of 32 records from 16 concurrent runs. The
    critical section here is one append or one trim — microseconds — so a
    blocking lock cannot make the hook slow, and it makes record loss
    structurally impossible rather than unlikely.
    """
    from synapt.recall._filelock import lock_exclusive, unlock

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a+") as lf:
        lock_exclusive(lf)
        try:
            yield
        finally:
            unlock(lf)


def _append_record(path: Path, record: dict) -> None:
    """Append ONE record as ONE write under the lock. Never rewrites the file."""
    line = json.dumps(record) + "\n"
    with _log_lock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def _trim_records(path: Path, keep: int) -> None:
    """Drop the oldest records once the file has grown well past *keep*.

    Runs under the same lock as appends, and rewrites through a tmp path that
    is unique to this process, so a concurrent appender can neither lose its
    line nor race another trimmer on the rename.
    """
    with _log_lock(path):
        records = _read_records(path)
        if len(records) <= keep:
            return
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in records[-keep:]), encoding="utf-8")
        os.replace(tmp, path)


# A started record older than this with no completion is dead regardless of
# what the pid probe says: pids are reused, and a hook that has run for
# fifteen minutes has been killed by the harness long ago.
STALE_RUN_SECONDS = 15 * 60


def _pid_alive_win32(pid: int) -> bool:
    """Process existence on Windows WITHOUT os.kill.

    ``os.kill(pid, sig)`` on Windows calls ``TerminateProcess`` for every
    signal except CTRL_C_EVENT / CTRL_BREAK_EVENT — including ``0``. The
    Unix liveness idiom would kill the sibling hook this probe exists to
    avoid misreporting (Atlas, r2 on v2). So: open a query-only handle and
    ask for the exit code; ``STILL_ACTIVE`` (259) means running.
    """
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    k32 = _win32_kernel32()
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


def _win32_kernel32():
    """kernel32 with the three prototypes DECLARED.

    ctypes assumes every foreign function returns C ``int`` and passes bare
    Python ints as C ``int`` unless told otherwise. ``OpenProcess`` returns a
    ``HANDLE``, which is ``PVOID``: on 64-bit Windows an undeclared call
    narrows it to 32 bits before ``GetExitCodeProcess`` and ``CloseHandle``
    consume it (Atlas, r2 on v3). Declaring ``argtypes``/``restype`` is what
    makes the native contract representable; a fake handle of 7 in a test
    cannot show the difference, which is why the witness for this asserts the
    declarations themselves and passes a handle wider than 32 bits through.
    """
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetExitCodeProcess.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    return k32


def _pid_alive(pid: object) -> bool:
    """Is there a live process with this pid? Non-destructive on every platform."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            return _pid_alive_win32(pid)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def _started_age_seconds(record: dict) -> float | None:
    raw = record.get("started")
    if not isinstance(raw, str):
        return None
    try:
        started = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - started).total_seconds()


def _run_is_dead(record: dict, completed_ids: set) -> bool:
    """A started record with no completion of its own, whose process is gone
    OR whose start is older than STALE_RUN_SECONDS (pid reuse cannot rescue
    a fifteen-minute-old hook)."""
    if record.get("status") != "started" or record.get("run_id") in completed_ids:
        return False
    age = _started_age_seconds(record)
    if age is not None and age > STALE_RUN_SECONDS:
        return True
    return not _pid_alive(record.get("pid"))


class HookRun:
    """Record one hook run as ``started`` then ``completed``, append-only.

    Each run carries a unique ``run_id``; ``completed`` binds to its own
    ``started`` by that id, never by position, because concurrent runs
    interleave. ``begin()`` reports a previous run of the same event as
    unfinished only when it has no ``completed`` of its own AND its process
    is gone — a sibling hook still running is not a death, and reporting it
    as one would make every concurrent start accuse the other.
    """

    def __init__(self, event: str, source: str, log_path: Path | None = None):
        self.event = event
        self.source = source
        self.log_path = log_path or hook_run_log_path()
        self.run_id = uuid.uuid4().hex
        self.phases: dict[str, float] = {}
        self.previous: dict | None = None
        self._t0 = time.monotonic()

    def begin(self) -> str | None:
        try:
            records = _read_records(self.log_path)
        except OSError:
            records = []
        mine = [r for r in records if r.get("event") == self.event and r.get("run_id") != self.run_id]
        completed_ids = {r.get("run_id") for r in mine if r.get("status") == "completed"}
        dead = [r for r in mine if _run_is_dead(r, completed_ids)]
        finished_or_alive = [r for r in mine if r.get("status") == "completed" or (r.get("status") == "started" and r not in dead)]
        self.previous = finished_or_alive[-1] if finished_or_alive else (mine[-1] if mine else None)
        warning = None
        if dead:
            d = dead[-1]
            warning = (
                f"WARNING: previous {self.event} hook did not finish "
                f"(started {d.get('started', '?')}, pid {d.get('pid', '?')}, run {d.get('run_id', '?')})"
            )
            self.previous = d
        try:
            _append_record(self.log_path, {
                "event": self.event,
                "source": self.source,
                "status": "started",
                "started": _now_iso(),
                "pid": os.getpid(),
                "run_id": self.run_id,
            })
        except OSError:
            pass  # never fail a session start over its own diary
        return warning

    @contextmanager
    def phase(self, name: str):
        t = time.monotonic()
        try:
            yield
        finally:
            self.phases[name] = round(time.monotonic() - t, 3)

    def finish(self, output_bytes: int) -> None:
        try:
            _append_record(self.log_path, {
                "event": self.event,
                "source": self.source,
                "status": "completed",
                "finished": _now_iso(),
                "pid": os.getpid(),
                "run_id": self.run_id,
                "total_s": round(time.monotonic() - self._t0, 3),
                "phases": dict(self.phases),
                "output_bytes": int(output_bytes),
            })
            # Trim only once the file is well past the bound, so the common
            # path is one append and nothing else.
            if self.log_path.stat().st_size > HOOK_RUN_LOG_MAX_RECORDS * 150:
                _trim_records(self.log_path, HOOK_RUN_LOG_MAX_RECORDS)
        except OSError:
            pass

    def health(self) -> str:
        """One phrase about the previous run, for the head line."""
        p = self.previous
        if not p:
            return "first run"
        if p.get("status") == "started":
            return "previous run DID NOT FINISH" if not _pid_alive(p.get("pid")) else "a sibling run is in progress"
        return f"previous run ok {p.get('total_s', '?')}s"


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _first_line(block: str) -> str:
    for line in block.splitlines():
        if line.strip():
            return line
    return ""


def _kind(block: str) -> str:
    head = _first_line(block)
    if head.startswith("UNCLEAN END"):
        return "unclean_end"
    if head.startswith("LAST CHECKPOINT"):
        return "checkpoint"
    if head.startswith("LAST COMPACTION SUMMARY") \
            or head.startswith("AGENT COMPACTION DIRECTIVE"):
        return "compaction"
    if head.startswith("Journal read: "):
        return "journal_coverage"
    if head.startswith("Last session") or head.startswith("Next steps:") \
            or head.startswith("Decisions:") or head.startswith("Done:"):
        return "journal"
    if head.startswith("Branch context"):
        return "branch"
    if head.startswith("Open PR"):
        return "open_pr"
    if head.startswith("Channel:"):
        return "channel_counts"
    if head.startswith("Recent #dev"):
        return "channel"
    if head.startswith("Pending directives"):
        return "directives"
    if head.startswith("Knowledge:"):
        return "knowledge"
    if head.startswith("Reminders"):
        return "reminders"
    if head.startswith("Pending contradictions"):
        return "contradictions"
    return "other"


def _clip_bytes(block: str, cap: int) -> tuple[str, int]:
    """Keep whole lines up to *cap* bytes. Returns (text, lines_withheld)."""
    if len(block.encode("utf-8")) <= cap:
        return block, 0
    kept: list[str] = []
    used = 0
    lines = block.splitlines()
    for line in lines:
        n = len(line.encode("utf-8")) + 1
        if used + n > cap:
            break
        kept.append(line)
        used += n
    withheld = len(lines) - len(kept)
    if not kept:  # a single oversized line: keep a prefix so the kind survives
        kept = [lines[0].encode("utf-8")[:max(0, cap - 4)].decode("utf-8", errors="ignore") + " …"]
        withheld = max(0, len(lines) - 1)
    kept.append(f"  … (+{withheld} line(s) withheld)")
    return "\n".join(kept), withheld


def _clip_reminders(block: str, max_items: int) -> tuple[str, int, int]:
    """Reminders are counted, not dumped. Returns (text, total, withheld)."""
    lines = block.splitlines()
    items = [l for l in lines if l.startswith("  - ")]
    total = len(items)
    if total <= max_items:
        return block, total, 0
    sticky = sum(1 for l in items if "[sticky]" in l)
    head = f"Reminders ({total}; {sticky} sticky; showing {max_items}):"
    return "\n".join([head, *items[:max_items], f"  … (+{total - max_items} withheld)"]), total, total - max_items


def _counts_phrase(blocks: dict[str, list[str]], reminder_total: int) -> str:
    parts: list[str] = []
    for b in blocks.get("journal_coverage", []):
        parts.append(_first_line(b))
    for b in blocks.get("channel_counts", []):
        parts.append(_first_line(b).removeprefix("Channel: ").strip())
    for b in blocks.get("contradictions", []):
        m = re.search(r"Pending contradictions \((\d+)", b)
        if m:
            parts.append(f"{m.group(1)} contradictions")
    if reminder_total:
        parts.append(f"{reminder_total} reminders")
    for b in blocks.get("directives", []):
        n = sum(1 for l in b.splitlines()[1:] if l.strip())
        if n:
            parts.append(f"{n} directive line(s)")
    return " · ".join(parts)


def render_wake(
    lines: list[str],
    *,
    source: str,
    run: HookRun | None = None,
    warning: str | None = None,
    banners: list[str] | None = None,
    budget: int = WAKE_BUDGET_BYTES,
    full_path: Path | None = None,
) -> str:
    """Render startup context inside *budget* bytes and write the full text to disk.

    Order is the point: head line, warnings, an unclean-end block when the
    previous session left no handoff, the latest journal entry (open
    threads first — see journal.format_for_session_start), channel state,
    directives, then everything else clipped per source, then a footer that
    says how many bytes were shown, how many withheld, and where the rest is.
    """
    # Ambient (None): the full-text pointer follows the STORE of the journal it
    # renders -- the wake resolves its journal ambiently (workspace-aware via
    # GRIPSPACE_ROOT), so its pointer must land in the same store, not the cwd.
    # A pointer written to the cwd store while the content is workspace-resolved
    # is the split-store bug in miniature (Stromus ruling 2026-08-31). Callers
    # that need a specific path still pass `full_path` explicitly.
    full_path = full_path or wake_file_path()
    blocks: dict[str, list[str]] = {}
    for block in lines:
        if not block or not block.strip():
            continue
        blocks.setdefault(_kind(block), []).append(block)

    # --- full text on disk, unclipped -------------------------------------
    full_parts = [f"# synapt wake · source={source} · {_now_iso()}"]
    if warning:
        full_parts.append(warning)
    full_parts.extend(banners or [])
    full_parts.extend(b for b in lines if b and b.strip())
    full_text = "\n\n".join(full_parts) + "\n"
    full_bytes = len(full_text.encode("utf-8"))
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = full_path.with_suffix(".tmp")
        tmp.write_text(full_text, encoding="utf-8")
        os.replace(tmp, full_path)
        pointer = str(full_path)
    except OSError as exc:
        pointer = f"(could not write {full_path}: {exc})"

    # --- clipped body -------------------------------------------------------
    body: list[str] = []
    withheld_lines = 0

    def take(kind: str, cap: int | None, *, first_cap: int | None = None) -> None:
        nonlocal withheld_lines
        for i, b in enumerate(blocks.pop(kind, [])):
            c = first_cap if (i == 0 and first_cap is not None) else cap
            if c is None:
                body.append(b)
                continue
            text, w = _clip_bytes(b, c)
            body.append(text)
            withheld_lines += w

    reminder_total = 0
    reminder_blocks = blocks.pop("reminders", [])
    reminder_texts: list[str] = []
    for b in reminder_blocks:
        text, total, w = _clip_reminders(b, _CAP_REMINDER_ITEMS)
        reminder_total += total
        withheld_lines += w
        reminder_texts.append(text)

    counts = _counts_phrase(blocks, reminder_total)
    blocks.pop("journal_coverage", None)

    # A previous session that ended without a handoff changes what the reader
    # does first, so it renders before everything, including the journal it
    # would otherwise be read as continuous with.
    take("unclean_end", _CAP_UNCLEAN_END)
    take("checkpoint", _CAP_CHECKPOINT)
    take("compaction", _CAP_COMPACTION)
    take("branch", _CAP_BRANCH)
    take("open_pr", _CAP_OPEN_PR)
    take("journal", _CAP_JOURNAL_OLDER, first_cap=_CAP_JOURNAL_LATEST)
    take("channel_counts", None)
    take("channel", _CAP_CHANNEL)
    take("directives", _CAP_DIRECTIVES)
    take("knowledge", _CAP_KNOWLEDGE)
    body.extend(reminder_texts)
    take("contradictions", _CAP_CONTRADICTIONS)
    take("other", _CAP_OTHER)

    head_bits = [f"synapt wake · source={source}"]
    if warning:
        head_bits.append(warning)
    elif run is not None:
        head_bits.append(run.health())
    if counts:
        head_bits.append(counts)
    head_bits.append(f"full context {full_bytes:,} B → {pointer}")
    head = " · ".join(head_bits)

    parts = [head, *(banners or []), *body]
    text = "\n\n".join(p for p in parts if p)

    # --- final guard: the budget holds even if a new source skips the table --
    footer_room = 200
    encoded = text.encode("utf-8")
    if len(encoded) > budget - footer_room:
        cut = encoded[: budget - footer_room].decode("utf-8", errors="ignore")
        cut = cut[: cut.rfind("\n")] if "\n" in cut else cut
        text = cut + "\n  … (truncated to budget)"
        withheld_lines += 1

    shown = len(text.encode("utf-8"))
    withheld_bytes = max(0, full_bytes - shown)
    if withheld_lines or withheld_bytes > 0 and shown < full_bytes:
        footer = f"--- shown {shown:,} of {full_bytes:,} B; {withheld_bytes:,} B withheld → read {pointer}"
    else:
        footer = f"--- {full_bytes:,} B, complete"
    return text + "\n\n" + footer + "\n"
