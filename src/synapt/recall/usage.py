"""OSS usage-event emission with an opaque correlation key.

Every tap point in recall calls :func:`emit_usage_event`.  Zero registered sinks
is the default and is a silent no-op, so the seam costs nothing until something
downstream asks for it.  An optional local JSONL debug sink exists for
development.

``session_ref`` is an opaque correlation key supplied by the environment
(``SYNAPT_AGENT_ID`` when set, otherwise ``"unattributed"``).  OSS never
interprets it -- it is forwarded, never resolved, never used to look anything
up.  The schema has no identity-named field; that is a guarantee about the
SCHEMA, not about what a caller chooses to place in ``session_ref``.

NEVER-DISRUPT RULE.  A sink exists to observe work, so it must never be able to
break the work it observes.  Every sink call and every debug write is wrapped:
a raising sink is logged at DEBUG and the emitting call continues.  The one
thing that is NOT swallowed is a malformed :class:`UsageEvent` -- constructing
an event with an unknown ``op`` raises, because that is a programming error at
the tap site rather than a runtime condition in someone else's callback.

CONCURRENCY.  The sink registry is process-global mutable state, and recall
emits from consolidation stages and from the native poller's own loop, so
"register while another thread is iterating" is reachable rather than
theoretical.  Registration is lock-guarded and emission iterates over a
SNAPSHOT taken under that lock.  Without it, mutating the list mid-iteration
does NOT raise: Python silently DELIVERS to a sink appended after the emit
began, and silently SKIPS a sink removed at or before the cursor.  There is no
exception for the never-disrupt rule to swallow and no symptom anywhere -- a
sink simply stops receiving some events.  That is the direction that costs us,
and it is why the snapshot is a witnessed guarantee rather than a nicety.

The lock's own job is narrower than the snapshot's and is stated here as
REASONING, not as a measurement: :func:`emit_usage_event` reads two module
globals, ``_sinks`` and ``_debug_sink_path``, and needs them as one consistent
pair.  No per-object locking provides that on any build.  Deleting the lock
leaves every test in this suite green on CPython, which the test module
discloses rather than implying coverage it does not have.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

UsageSink = Callable[["UsageEvent"], None]

#: The closed operation set.  A new op is a deliberate, reviewable addition
#: here, not something a tap site can invent by passing a new string.
USAGE_OPS = frozenset(
    {
        "infer",
        "mem_write",
        "mem_read",
        "mem_search",
        "consolidate_stage",
        "channel_post",
    }
)

#: The value of ``session_ref`` when the environment supplies no correlation
#: key.  A literal, never an empty string, so a downstream consumer grouping by
#: ``session_ref`` sees one honest bucket rather than a falsy value it has to
#: special-case.
UNATTRIBUTED = "unattributed"


@dataclass
class UsageEvent:
    """One unit of metered work.

    Counts and duration only.  There is deliberately no cost, price, or
    currency field: pricing is a policy decision that belongs to whatever
    consumes these events, and computing it at emission would bake one
    consumer's policy into the OSS seam.
    """

    ts: str
    session_ref: str
    op: str
    detail: str
    model: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cached_tokens: Optional[int] = None
    duration_ms: Optional[int] = None

    def __post_init__(self) -> None:
        if self.op not in USAGE_OPS:
            raise ValueError(
                f"unknown op {self.op!r}; the op set is closed: "
                f"{sorted(USAGE_OPS)}"
            )


def now_iso() -> str:
    """UTC timestamp in the shape the schema documents."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def current_session_ref() -> str:
    """Read the opaque correlation key from the environment, AT CALL TIME.

    Deliberately not a module-level constant.  A process can have the variable
    set after import (a test's ``monkeypatch.setenv``, a runtime that populates
    the environment before dispatching work), and a constant captured at import
    would report ``unattributed`` forever with nothing going red.
    """
    return os.environ.get("SYNAPT_AGENT_ID") or UNATTRIBUTED


_sinks: list[UsageSink] = []
_sinks_lock = threading.Lock()
_debug_sink_path: Optional[Path] = None


def register_usage_sink(sink: UsageSink) -> None:
    """Register a callable to receive every emitted event."""
    with _sinks_lock:
        _sinks.append(sink)


def unregister_usage_sink(sink: UsageSink) -> None:
    """Stop delivering events to ``sink``.  Unregistering an unregistered sink
    is a no-op rather than an error -- teardown paths should not have to know
    whether setup got far enough to register."""
    with _sinks_lock:
        try:
            _sinks.remove(sink)
        except ValueError:
            pass


def clear_usage_sinks() -> None:
    """Drop every registered sink.  Does NOT touch the debug sink, which has
    its own lifecycle -- see :func:`disable_debug_sink`."""
    with _sinks_lock:
        _sinks.clear()


def enable_debug_sink(path) -> None:
    """Append every emitted event to ``path`` as JSONL.  Development only."""
    global _debug_sink_path
    with _sinks_lock:
        _debug_sink_path = Path(path)


def disable_debug_sink() -> None:
    global _debug_sink_path
    with _sinks_lock:
        _debug_sink_path = None


def emit_usage_event(event: UsageEvent) -> None:
    """Fan out one event.  Never raises."""
    with _sinks_lock:
        sinks = tuple(_sinks)
        debug_path = _debug_sink_path

    for sink in sinks:
        try:
            sink(event)
        except Exception:
            logger.debug("usage sink raised; event dropped for it", exc_info=True)

    if debug_path is not None:
        try:
            with open(debug_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(event)) + "\n")
        except Exception as exc:
            logger.debug("usage debug sink write failed: %s", exc)
