"""Agent-to-agent direct messaging -- unicast with delivery guarantees.

Storage layout (all under .synapt/recall/direct/):
  <agent_id>.jsonl  -- per-agent inbox log (append-only, open protocol)
  direct.db         -- SQLite for delivery state: status, acks, timestamps

Composes with recall_channel (broadcast) -- same storage root, different
semantic layer.  Channels are broadcast; direct messages are unicast with
delivery tracking and explicit acknowledgment.

Hook registration seam (a downstream layer may register handlers):
  before_send hooks can reject a message (return a reason string, or None to allow).
  state_change hooks fire on status transitions.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

MAX_BODY_SIZE = 65536  # 64KB default cap

# ---------------------------------------------------------------------------
# Hook registration -- downstream coordination extension seam
# ---------------------------------------------------------------------------

_before_send_hooks: list[Callable[["DirectMessage"], str | None]] = []
_state_change_hooks: list[Callable[[str, str, str], None]] = []


def register_before_send_hook(
    hook: Callable[["DirectMessage"], str | None],
) -> None:
    """Register a pre-send check.  Return None to allow, or a reason string to deny."""
    _before_send_hooks.append(hook)


def register_state_change_hook(
    hook: Callable[[str, str, str], None],
) -> None:
    """Register a callback for status transitions: (message_id, old_status, new_status)."""
    _state_change_hooks.append(hook)


def _clear_hooks() -> None:
    """Reset hooks -- for tests only."""
    _before_send_hooks.clear()
    _state_change_hooks.clear()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

STATUS_QUEUED = "queued"
STATUS_DELIVERED = "delivered"
STATUS_READ = "read"
STATUS_ACKED = "acked"

PRIORITY_NORMAL = "normal"
PRIORITY_URGENT = "urgent"
_VALID_PRIORITIES = {PRIORITY_NORMAL, PRIORITY_URGENT}


@dataclass
class DirectMessage:
    message_id: str
    from_agent: str
    to_agent: str
    timestamp: str
    body: str
    reply_to: str | None = None
    priority: str = PRIORITY_NORMAL

    def to_dict(self) -> dict:
        d = asdict(self)
        d["from"] = d.pop("from_agent")
        d["to"] = d.pop("to_agent")
        if not d.get("reply_to"):
            d.pop("reply_to", None)
        if d.get("priority") == PRIORITY_NORMAL:
            d.pop("priority", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DirectMessage:
        mapped = dict(d)
        if "from" in mapped:
            mapped["from_agent"] = mapped.pop("from")
        if "to" in mapped:
            mapped["to_agent"] = mapped.pop("to")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in mapped.items() if k in known})


@dataclass(frozen=True)
class RegisteredRecipient:
    """A stable inbox identity plus its configured canonical store coordinate."""

    agent_id: str
    store_coordinate: str
    display_name: str


_recipient_resolver: Callable[[str], RegisteredRecipient] | None = None


def set_recipient_resolver(
    resolver: Callable[[str], RegisteredRecipient] | None,
) -> Callable[[str], RegisteredRecipient] | None:
    """Install the configured identity resolver supplied by the embedding layer.

    OSS does not own identity or org truth.  Its transport accepts only the
    stable inbox ID and opaque recipient-owned store coordinate returned here.
    """
    global _recipient_resolver
    previous = _recipient_resolver
    _recipient_resolver = resolver
    return previous


def _recipient_aliases(recipient: RegisteredRecipient) -> set[str]:
    """Exact, documented spellings for one registered recipient."""
    agent_id = recipient.agent_id.lower()
    short_id = (
        agent_id.rsplit("-", 1)[0]
        if agent_id.rsplit("-", 1)[-1].isdigit()
        else agent_id
    )
    name = recipient.display_name.strip().lower()
    return {
        agent_id,
        short_id,
        name,
        f"{recipient.store_coordinate.lower()}:{agent_id}",
        f"{recipient.store_coordinate.lower()}:{short_id}",
        f"{recipient.store_coordinate.lower()}:{name}",
    }


def resolve_registered_recipient(
    target: str,
    *,
    recipients: Iterable[RegisteredRecipient] | None = None,
) -> RegisteredRecipient:
    """Resolve a recipient spelling to one stable registered inbox identity.

    Exact aliases only: canonical ID, display name, stable-ID stem, and their
    org-qualified forms.  No match and more than one match both fail closed so
    sending cannot manufacture an unregistered (phantom) inbox.
    """
    normalized = target.strip().lower()
    if not normalized:
        raise ValueError("recipient is required")
    if recipients is None:
        if _recipient_resolver is not None:
            return _validated_recipient(_recipient_resolver(target))
        records = load_pane_records()
        if not records:
            raise ValueError("recipient resolver is not configured")
        matches = [
            record
            for record in records
            if normalized == str(record.get("qualified_alias", "")).lower()
            or normalized == str(record.get("agent_id", "")).lower()
            or (
                ":" not in normalized
                and normalized == str(record.get("_key", "")).lower()
            )
        ]
        recipients = [
            RegisteredRecipient(
                str(record["agent_id"]),
                str(record["store_coordinate"]),
                str(record.get("qualified_alias", record["agent_id"])),
            )
            for record in matches
        ]
        configured_unique = {
            (recipient.store_coordinate, recipient.agent_id): recipient
            for recipient in recipients
        }
        if len(configured_unique) == 1:
            return _validated_recipient(next(iter(configured_unique.values())))
        if not configured_unique:
            raise ValueError(f"recipient '{target}' is not registered")
        choices = ", ".join(
            f"{recipient.store_coordinate}:{recipient.agent_id}"
            for recipient in sorted(
                configured_unique.values(),
                key=lambda recipient: (
                    recipient.store_coordinate,
                    recipient.agent_id,
                ),
            )
        )
        raise ValueError(
            f"recipient '{target}' is ambiguous ({choices}); use a qualified alias or canonical ID"
        )
    catalog = list(recipients)
    matches = [
        recipient
        for recipient in catalog
        if normalized in _recipient_aliases(recipient)
    ]
    unique = {(r.store_coordinate, r.agent_id): r for r in matches}
    if len(unique) == 1:
        return _validated_recipient(next(iter(unique.values())))
    if not unique:
        raise ValueError(f"recipient '{target}' is not registered")
    choices = ", ".join(
        f"{r.store_coordinate}:{r.agent_id}"
        for r in sorted(unique.values(), key=lambda r: (r.store_coordinate, r.agent_id))
    )
    raise ValueError(
        f"recipient '{target}' is ambiguous ({choices}); use an org-qualified or canonical ID"
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _direct_dir(project_dir: Path | None = None) -> Path:
    """Return the direct-messaging directory.

    Uses the same resolution hierarchy as channels:
    1. SYNAPT_SHARED_CHANNELS_DIR env var (shared root)
    2. Global store ~/.synapt/channels/<org>/<project>/direct/
    3. Local per-gripspace directory
    """
    from synapt.recall.channel import _channels_dir

    return _channels_dir(project_dir) / "direct"


def _inbox_path(agent_id: str, project_dir: Path | None = None) -> Path:
    agent_id = _validated_agent_id(agent_id)
    base = _direct_dir(project_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{agent_id}.jsonl"


# recall#820 cross-org silo fix: the gripspace-local stores above (_direct_dir
# routes through _channels_dir(project_dir)) silo direct messages per-gripspace.
# A send from gripspace A lands in A's store; a read from gripspace B queries B's
# store and finds nothing. The canonical cross-org root below is
# project-INDEPENDENT (org-keyed, project component dropped) so every gripspace
# in the org agrees on one inbox per recipient. send_message dual-writes here;
# read_inbox union-reads from here. Design: Option A (project-independent
# org-canonical cross-org root).
_DEFAULT_ORG = "synapt-dev"
_STORE_COORDINATE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_AGENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _validated_agent_id(agent_id: str) -> str:
    """Return a stable inbox ID only when it cannot alter a path."""
    if not isinstance(agent_id, str) or not _AGENT_ID_RE.fullmatch(agent_id):
        raise ValueError("invalid agent ID")
    return agent_id


def _validated_store_coordinate(coordinate: str) -> str:
    if not isinstance(coordinate, str) or not _STORE_COORDINATE_RE.fullmatch(
        coordinate
    ):
        raise ValueError("invalid recipient store coordinate")
    return coordinate


def _validated_recipient(recipient: RegisteredRecipient) -> RegisteredRecipient:
    """Validate premium-provided routing values before any transport side effect."""
    return RegisteredRecipient(
        agent_id=_validated_agent_id(recipient.agent_id),
        store_coordinate=_validated_store_coordinate(recipient.store_coordinate),
        display_name=recipient.display_name,
    )


def _cross_org_root(
    project_dir: Path | None = None, *, recipient_store_coordinate: str | None = None
) -> Path:
    """Resolve the project-independent, org-canonical cross-org direct root.

    Resolution:
    1. SYNAPT_SHARED_CHANNELS_DIR override -> <shared>/_cross-org/direct/.
       This branch keeps the existing test suite + explicit shared deployments
       isolated: because send_message now ALWAYS dual-writes here, a raw
       Path.home() root would write into the real ~/.synapt during tests that
       only set the shared override. Honoring the override forecloses that.
    2. Org-canonical under home -> ~/.synapt/channels/<org>/_cross-org/direct/.
       The org is derived from the gripspace manifest; the <project> component
       is intentionally dropped so the root is shared across all gripspaces in
       the org (this is the silo fix).
    """
    from synapt.recall.channel import _resolve_org_id, _shared_channels_dir

    if recipient_store_coordinate is not None:
        recipient_store_coordinate = _validated_store_coordinate(
            recipient_store_coordinate
        )
    shared = _shared_channels_dir()
    if shared:
        return shared / "_cross-org" / "direct"
    coordinate = _validated_store_coordinate(
        recipient_store_coordinate or _resolve_org_id(project_dir) or _DEFAULT_ORG
    )
    return Path.home() / ".synapt" / "channels" / coordinate / "_cross-org" / "direct"


def _cross_org_inbox_path(
    to_agent: str,
    project_dir: Path | None = None,
    *,
    recipient_store_coordinate: str | None = None,
) -> Path:
    to_agent = _validated_agent_id(to_agent)
    base = _cross_org_root(
        project_dir, recipient_store_coordinate=recipient_store_coordinate
    )
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{to_agent}.jsonl"


def _db_path(project_dir: Path | None = None) -> Path:
    base = _direct_dir(project_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / "direct.db"


def _canonical_db_path(
    project_dir: Path | None = None, *, recipient_store_coordinate: str | None = None
) -> Path:
    """Delivery state colocated with the recipient-owned canonical inbox."""
    return (
        _cross_org_root(
            project_dir, recipient_store_coordinate=recipient_store_coordinate
        )
        / "direct.db"
    )


# ---------------------------------------------------------------------------
# SQLite state layer
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    from_agent TEXT NOT NULL,
    to_agent TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    body TEXT NOT NULL,
    reply_to TEXT,
    priority TEXT DEFAULT 'normal',
    status TEXT DEFAULT 'queued',
    created_at REAL NOT NULL,
    delivered_at REAL,
    read_at REAL,
    acked_at REAL
);
CREATE INDEX IF NOT EXISTS idx_messages_to ON messages(to_agent, status);
CREATE INDEX IF NOT EXISTS idx_messages_from ON messages(from_agent);
CREATE INDEX IF NOT EXISTS idx_messages_reply ON messages(reply_to);
"""


def _ensure_direct_columns(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN canonical_store TEXT")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise
    columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    if "canonical_store" not in columns:
        raise sqlite3.OperationalError("messages.canonical_store migration failed")
    conn.commit()


def _get_db(project_dir: Path | None = None) -> sqlite3.Connection:
    path = _db_path(project_dir)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _ensure_direct_columns(conn)
    return conn


def _get_canonical_db(
    project_dir: Path | None = None, *, recipient_store_coordinate: str | None = None
) -> sqlite3.Connection:
    path = _canonical_db_path(
        project_dir, recipient_store_coordinate=recipient_store_coordinate
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _ensure_direct_columns(conn)
    return conn


def _transition_status(
    conn: sqlite3.Connection,
    message_id: str,
    new_status: str,
) -> str | None:
    """Transition message status.  Returns old status, or None if not found."""
    row = conn.execute(
        "SELECT status FROM messages WHERE message_id = ?", (message_id,)
    ).fetchone()
    if row is None:
        return None

    old_status = row["status"]
    now = datetime.now(timezone.utc).timestamp()

    ts_col = {
        STATUS_DELIVERED: "delivered_at",
        STATUS_READ: "read_at",
        STATUS_ACKED: "acked_at",
    }.get(new_status)

    if ts_col:
        conn.execute(
            f"UPDATE messages SET status = ?, {ts_col} = ? WHERE message_id = ?",
            (new_status, now, message_id),
        )
    else:
        conn.execute(
            "UPDATE messages SET status = ? WHERE message_id = ?",
            (new_status, message_id),
        )
    conn.commit()

    for hook in _state_change_hooks:
        try:
            hook(message_id, old_status, new_status)
        except Exception:
            pass

    return old_status


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def send_message(
    *,
    from_agent: str,
    to_agent: str,
    body: str,
    reply_to: str | None = None,
    priority: str = PRIORITY_NORMAL,
    project_dir: Path | None = None,
    recipient_store_coordinate: str | None = None,
) -> DirectMessage:
    """Send a direct message to an agent.  Returns the sent message."""
    if not from_agent:
        raise ValueError("from_agent is required")
    if not to_agent:
        raise ValueError("to_agent is required")
    from_agent = _validated_agent_id(from_agent)
    to_agent = _validated_agent_id(to_agent)
    if recipient_store_coordinate is not None:
        recipient_store_coordinate = _validated_store_coordinate(
            recipient_store_coordinate
        )
    if not body or not body.strip():
        raise ValueError("message body is required")
    if len(body.encode("utf-8")) > MAX_BODY_SIZE:
        raise ValueError(f"message body exceeds {MAX_BODY_SIZE} byte limit")
    if priority not in _VALID_PRIORITIES:
        raise ValueError(
            f"invalid priority '{priority}', must be one of {_VALID_PRIORITIES}"
        )
    if from_agent == to_agent:
        raise ValueError("cannot send a direct message to yourself")

    msg = DirectMessage(
        message_id=f"dm_{uuid.uuid4().hex[:12]}",
        from_agent=from_agent,
        to_agent=to_agent,
        timestamp=datetime.now(timezone.utc).isoformat(),
        body=body,
        reply_to=reply_to,
        priority=priority,
    )

    for hook in _before_send_hooks:
        deny_reason = hook(msg)
        if deny_reason is not None:
            raise PermissionError(f"send denied: {deny_reason}")

    # Write to recipient's inbox JSONL (gripspace-local; intra-org fast path)
    inbox = _inbox_path(to_agent, project_dir)
    line = json.dumps(msg.to_dict(), ensure_ascii=False) + "\n"
    with open(inbox, "a", encoding="utf-8") as f:
        f.write(line)

    # recall#820: dual-write to the project-independent cross-org canonical
    # inbox so a recipient in a different gripspace reads it (the silo fix).
    cross_inbox = _cross_org_inbox_path(
        to_agent, project_dir, recipient_store_coordinate=recipient_store_coordinate
    )
    with open(cross_inbox, "a", encoding="utf-8") as f:
        f.write(line)

    # Track in SQLite
    conn = _get_db(project_dir)
    try:
        now = datetime.now(timezone.utc).timestamp()
        conn.execute(
            """INSERT OR IGNORE INTO messages
               (message_id, from_agent, to_agent, timestamp, body,
                reply_to, priority, status, created_at, delivered_at, canonical_store)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg.message_id,
                msg.from_agent,
                msg.to_agent,
                msg.timestamp,
                msg.body,
                msg.reply_to,
                msg.priority,
                STATUS_DELIVERED,
                now,
                now,
                recipient_store_coordinate,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    canonical_conn = _get_canonical_db(
        project_dir, recipient_store_coordinate=recipient_store_coordinate
    )
    try:
        canonical_conn.execute(
            """INSERT OR IGNORE INTO messages
               (message_id, from_agent, to_agent, timestamp, body,
                reply_to, priority, status, created_at, delivered_at, canonical_store)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg.message_id,
                msg.from_agent,
                msg.to_agent,
                msg.timestamp,
                msg.body,
                msg.reply_to,
                msg.priority,
                STATUS_DELIVERED,
                now,
                now,
                recipient_store_coordinate,
            ),
        )
        canonical_conn.commit()
    finally:
        canonical_conn.close()

    for hook in _state_change_hooks:
        try:
            hook(msg.message_id, STATUS_QUEUED, STATUS_DELIVERED)
        except Exception:
            pass

    return msg


def _read_cross_org_candidates(
    agent_id: str,
    conn: sqlite3.Connection,
    local_ids: set[str],
    project_dir: Path | None,
    recipient_store_coordinate: str | None,
) -> list[DirectMessage]:
    """Return cross-org canonical messages for agent_id not already tracked locally.

    recall#820: messages sent from another gripspace have no local SQLite row
    (their delivery state lives in the SENDER's store). This scans the canonical
    cross-org inbox and surfaces any message whose message_id is neither in the
    local delivered set (`local_ids`) nor already present in the local SQLite
    `messages` table (read/acked on a prior read). Caller marks the returned
    ones READ; messages beyond the read limit are left untracked so a later
    read re-surfaces them.
    """
    cross_inbox = _cross_org_inbox_path(
        agent_id, project_dir, recipient_store_coordinate=recipient_store_coordinate
    )
    if not cross_inbox.exists():
        return []

    candidates: list[DirectMessage] = []
    seen: set[str] = set(local_ids)
    with open(cross_inbox, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg = DirectMessage.from_dict(data)
            if msg.to_agent != agent_id or msg.message_id in seen:
                continue
            seen.add(msg.message_id)
            already = conn.execute(
                "SELECT 1 FROM messages WHERE message_id = ?", (msg.message_id,)
            ).fetchone()
            if already is not None:
                continue
            candidates.append(msg)
    return candidates


def _track_cross_org_read(
    conn: sqlite3.Connection,
    msg: DirectMessage,
    *,
    project_dir: Path | None,
    recipient_store_coordinate: str | None,
) -> None:
    """Insert a local SQLite tracking row (status=READ) for a cross-org message.

    Gives the cross-org message a local delivery-state row so re-reads dedup it
    and ack_message can transition it. Idempotent via INSERT OR IGNORE.
    """
    now = datetime.now(timezone.utc).timestamp()
    conn.execute(
        """INSERT OR IGNORE INTO messages
           (message_id, from_agent, to_agent, timestamp, body, reply_to,
            priority, status, created_at, delivered_at, read_at, canonical_store)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            msg.message_id,
            msg.from_agent,
            msg.to_agent,
            msg.timestamp,
            msg.body,
            msg.reply_to,
            msg.priority,
            STATUS_READ,
            now,
            now,
            now,
            recipient_store_coordinate,
        ),
    )
    conn.commit()
    canonical_conn = _get_canonical_db(
        project_dir, recipient_store_coordinate=recipient_store_coordinate
    )
    try:
        _transition_status(canonical_conn, msg.message_id, STATUS_READ)
    finally:
        canonical_conn.close()


def read_inbox(
    *,
    agent_id: str,
    limit: int = 20,
    project_dir: Path | None = None,
    recipient_store_coordinate: str | None = None,
) -> list[DirectMessage]:
    """Read unread messages from an agent's inbox.  Marks them as READ.

    recall#820: union of the local SQLite delivery store (intra-org fast path)
    and the project-independent cross-org canonical inbox (cross-gripspace
    path), deduped by message_id. Urgent messages surface first, then oldest
    first; only the returned (post-limit) messages are marked READ.
    """
    agent_id = _validated_agent_id(agent_id)
    if recipient_store_coordinate is not None:
        recipient_store_coordinate = _validated_store_coordinate(
            recipient_store_coordinate
        )
    conn = _get_db(project_dir)
    try:
        local_rows = conn.execute(
            """SELECT message_id, from_agent, to_agent, timestamp, body,
                      reply_to, priority
               FROM messages
               WHERE to_agent = ? AND status = ?""",
            (agent_id, STATUS_DELIVERED),
        ).fetchall()

        local_msgs = [
            DirectMessage(
                message_id=row["message_id"],
                from_agent=row["from_agent"],
                to_agent=row["to_agent"],
                timestamp=row["timestamp"],
                body=row["body"],
                reply_to=row["reply_to"],
                priority=row["priority"],
            )
            for row in local_rows
        ]
        local_ids = {m.message_id for m in local_msgs}

        cross_msgs = _read_cross_org_candidates(
            agent_id, conn, local_ids, project_dir, recipient_store_coordinate
        )
        is_cross = {m.message_id for m in cross_msgs}

        combined = local_msgs + cross_msgs
        combined.sort(
            key=lambda m: (0 if m.priority == PRIORITY_URGENT else 1, m.timestamp)
        )
        returned = combined[:limit]

        for msg in returned:
            if msg.message_id in is_cross:
                _track_cross_org_read(
                    conn,
                    msg,
                    project_dir=project_dir,
                    recipient_store_coordinate=recipient_store_coordinate,
                )
            else:
                _transition_status(conn, msg.message_id, STATUS_READ)
                canonical_conn = _get_canonical_db(
                    project_dir, recipient_store_coordinate=recipient_store_coordinate
                )
                try:
                    _transition_status(canonical_conn, msg.message_id, STATUS_READ)
                finally:
                    canonical_conn.close()

        return returned
    finally:
        conn.close()


def ack_message(
    *,
    message_id: str,
    agent_id: str,
    project_dir: Path | None = None,
    recipient_store_coordinate: str | None = None,
) -> str:
    """Acknowledge a message.  Returns status string."""
    agent_id = _validated_agent_id(agent_id)
    if recipient_store_coordinate is not None:
        recipient_store_coordinate = _validated_store_coordinate(
            recipient_store_coordinate
        )
    conn = _get_db(project_dir)
    try:
        row = conn.execute(
            "SELECT to_agent, status FROM messages WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            return f"message {message_id} not found"
        if row["to_agent"] != agent_id:
            return f"message {message_id} is not addressed to {agent_id}"
        if row["status"] == STATUS_ACKED:
            return f"message {message_id} already acknowledged"

        _transition_status(conn, message_id, STATUS_ACKED)
        canonical_conn = _get_canonical_db(
            project_dir, recipient_store_coordinate=recipient_store_coordinate
        )
        try:
            _transition_status(canonical_conn, message_id, STATUS_ACKED)
        finally:
            canonical_conn.close()
        return f"message {message_id} acknowledged"
    finally:
        conn.close()


def check_status(
    *,
    message_id: str,
    project_dir: Path | None = None,
) -> dict | None:
    """Check delivery status of a message.  Returns status dict or None."""
    conn = _get_db(project_dir)
    try:
        row = conn.execute(
            """SELECT message_id, from_agent, to_agent, status, canonical_store,
                      created_at, delivered_at, read_at, acked_at
               FROM messages WHERE message_id = ?""",
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        if row["canonical_store"]:
            canonical = _get_canonical_db(
                project_dir, recipient_store_coordinate=row["canonical_store"]
            )
            try:
                shared_row = canonical.execute(
                    """SELECT message_id, from_agent, to_agent, status,
                              created_at, delivered_at, read_at, acked_at
                       FROM messages WHERE message_id = ?""",
                    (message_id,),
                ).fetchone()
                if shared_row is not None:
                    return dict(shared_row)
            finally:
                canonical.close()
        return dict(row)
    finally:
        conn.close()


def message_history(
    *,
    agent_id: str,
    with_agent: str,
    limit: int = 20,
    project_dir: Path | None = None,
    include_canonical: bool = False,
    canonical_store_coordinate: str | None = None,
) -> list[DirectMessage]:
    """Read message history between two agents (both directions)."""
    agent_id = _validated_agent_id(agent_id)
    with_agent = _validated_agent_id(with_agent)
    if canonical_store_coordinate is not None:
        canonical_store_coordinate = _validated_store_coordinate(
            canonical_store_coordinate
        )
    conn = _get_db(project_dir)
    try:
        rows = conn.execute(
            """SELECT message_id, from_agent, to_agent, timestamp, body,
                      reply_to, priority
               FROM messages
               WHERE (from_agent = ? AND to_agent = ?)
                  OR (from_agent = ? AND to_agent = ?)
               ORDER BY created_at DESC
               LIMIT ?""",
            (agent_id, with_agent, with_agent, agent_id, limit),
        ).fetchall()

        messages = [
            DirectMessage(
                message_id=row["message_id"],
                from_agent=row["from_agent"],
                to_agent=row["to_agent"],
                timestamp=row["timestamp"],
                body=row["body"],
                reply_to=row["reply_to"],
                priority=row["priority"],
            )
            for row in rows
        ]
        if not include_canonical:
            return messages
        seen = {message.message_id for message in messages}
        coordinates = [
            row[0]
            for row in conn.execute(
                """SELECT DISTINCT canonical_store FROM messages
                   WHERE ((from_agent = ? AND to_agent = ?)
                       OR (from_agent = ? AND to_agent = ?))
                     AND canonical_store IS NOT NULL""",
                (agent_id, with_agent, with_agent, agent_id),
            ).fetchall()
        ]
        if canonical_store_coordinate:
            coordinates.append(canonical_store_coordinate)
        for coordinate in coordinates:
            coordinate = _validated_store_coordinate(coordinate)
            try:
                canonical_conn = _get_canonical_db(
                    project_dir, recipient_store_coordinate=coordinate
                )
                try:
                    canonical_rows = canonical_conn.execute(
                        """SELECT message_id, from_agent, to_agent, timestamp, body,
                                  reply_to, priority
                           FROM messages
                           WHERE (from_agent = ? AND to_agent = ?)
                              OR (from_agent = ? AND to_agent = ?)""",
                        (agent_id, with_agent, with_agent, agent_id),
                    ).fetchall()
                finally:
                    canonical_conn.close()
            except sqlite3.Error:
                continue
            for row in canonical_rows:
                if row["message_id"] not in seen:
                    seen.add(row["message_id"])
                    messages.append(
                        DirectMessage(
                            message_id=row["message_id"],
                            from_agent=row["from_agent"],
                            to_agent=row["to_agent"],
                            timestamp=row["timestamp"],
                            body=row["body"],
                            reply_to=row["reply_to"],
                            priority=row["priority"],
                        )
                    )
        messages.sort(key=lambda message: message.timestamp, reverse=True)
        return messages[:limit]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# tmux delivery (recall#852) -- OSS transport
#
# The inbox write is durable but passive: the recipient has to poll it. This
# layer ALSO delivers the message into the recipient's live tmux pane via
# load-buffer + paste-buffer + send-keys, the mechanism that actually lands.
# Boundary: the tmux MECHANICS are OSS transport. The agent->pane map is
# operator-supplied data read from a neutral env/config seam (SYNAPT_AGENT_PANES),
# NOT identity topology hardcoded into OSS -- resolving "who is at which pane" is
# operator config, the same shape as an /etc/hosts routing table.
# ---------------------------------------------------------------------------

# Pastes at/above this size tend to collapse in the recipient TUI ("paste again
# to expand"); we send one extra guarded Enter to expand before the submit Enter.
LARGE_PASTE_THRESHOLD = 1200

# Runtime -> Enter count. Claude needs paste-expand + submit; Codex folds large
# pastes so it needs an extra fold-expand first.
_ENTER_COUNT = {"claude": 2, "codex": 3}

_TMUX_BUFFER = "synapt_speak_to_agent"

# A short pause between key sends so the TUI registers each Enter separately
# rather than coalescing expand and submit into one keystroke.
_SEND_KEY_PAUSE_SECONDS = 0.15


@dataclass(frozen=True)
class PaneTarget:
    """Where (and how) to deliver to an agent's live session."""

    target: str
    runtime: str


@dataclass
class TmuxDelivery:
    """Result of a best-effort tmux delivery."""

    delivered: bool
    target: str | None
    enters: int
    detail: str


def enter_count(runtime: str | None) -> int:
    """Enter keystrokes needed to submit a paste for this runtime.

    Claude=2 (paste-expand + submit), Codex=3 (fold-expand + paste-expand +
    submit). Unknown runtimes default to the Claude count.
    """
    return _ENTER_COUNT.get((runtime or "").strip().lower(), 2)


def load_pane_map() -> dict[str, Any]:
    """Operator-supplied agent->pane map (neutral routing table, not identity).

    Source order: the ``SYNAPT_AGENT_PANES`` env var as a JSON object, else the
    JSON file at ``SYNAPT_AGENT_PANES_FILE``. Defaults to an empty map (which
    makes delivery a no-op so the send falls back to inbox-only). OSS never
    hardcodes the workspace's agent topology -- that is operator config.

    Format (operator-supplied)::

        {"<agent>": {"target": "<tmux-session>:<window>", "runtime": "claude|codex"}}
    """
    raw = os.environ.get("SYNAPT_AGENT_PANES")
    if raw:
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return {}
    path = os.environ.get("SYNAPT_AGENT_PANES_FILE")
    if path and Path(path).exists():
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            records = payload.get("records") if isinstance(payload, dict) else payload
            if isinstance(records, list):
                return {
                    str(record["agent_id"]): {
                        "target": record["target"],
                        "runtime": record["runtime"],
                    }
                    for record in records
                    if isinstance(record, dict)
                    and {"agent_id", "target", "runtime"} <= set(record)
                }
            if isinstance(payload, dict) and all(
                isinstance(record, dict) for record in payload.values()
            ):
                return {
                    str(record["agent_id"]): {
                        "target": record["target"],
                        "runtime": record["runtime"],
                    }
                    for record in payload.values()
                    if {"agent_id", "target", "runtime"} <= set(record)
                }
            return payload if isinstance(payload, dict) else {}
        except (ValueError, OSError):
            return {}
    return {}


def load_pane_records() -> list[dict[str, Any]]:
    """Read grip-generated routing records, never infer identity from them."""
    path = os.environ.get("SYNAPT_AGENT_PANES_FILE")
    if not path:
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    if not isinstance(payload, dict):
        return []
    required = {"qualified_alias", "agent_id", "store_coordinate", "target", "runtime"}
    return [
        {**record, "_key": key}
        for key, record in payload.items()
        if isinstance(record, dict) and required <= set(record)
    ]


def _normalize_agent_key(to_agent: str) -> str:
    # Recipients may arrive bare ("apollo"), org-prefixed ("synapt:apollo"), or
    # cased; the pane map is keyed by the bare lower-case agent name.
    key = to_agent.strip().lower()
    if ":" in key:
        key = key.split(":", 1)[1]
    # Stable inbox IDs are ``name-NNN`` while legacy pane maps are commonly
    # keyed by ``name``.  Routing remains operator-owned; this only preserves
    # the existing map convention after recipient resolution canonicalizes IDs.
    stem, sep, suffix = key.rpartition("-")
    if sep and suffix.isdigit():
        key = stem
    return key


def resolve_pane(to_agent: str, pane_map: dict[str, Any]) -> PaneTarget | None:
    """Resolve a recipient to its PaneTarget via the operator map, or None."""
    if not to_agent or not pane_map:
        return None
    entry = pane_map.get(to_agent) or pane_map.get(_normalize_agent_key(to_agent))
    if not isinstance(entry, dict) or not entry.get("target"):
        return None
    return PaneTarget(
        target=str(entry["target"]), runtime=str(entry.get("runtime", "claude"))
    )


def build_tmux_commands(
    target: str,
    runtime: str | None,
    body: str,
    *,
    buffer_name: str = _TMUX_BUFFER,
    large_threshold: int = LARGE_PASTE_THRESHOLD,
) -> tuple[list[list[str]], int]:
    """Build the tmux command sequence + the Enter count it will send.

    load-buffer (body via stdin, never shell-escaped) -> paste-buffer into the
    target pane (-d deletes the buffer after) -> N send-keys Enter, where N is the
    runtime Enter count plus one guarded Enter when the paste is collapse-large.
    """
    enters = enter_count(runtime)
    if len(body) >= large_threshold:
        enters += 1
    cmds: list[list[str]] = [
        ["tmux", "load-buffer", "-b", buffer_name, "-"],
        ["tmux", "paste-buffer", "-t", target, "-b", buffer_name, "-d"],
    ]
    cmds += [["tmux", "send-keys", "-t", target, "Enter"] for _ in range(enters)]
    return cmds, enters


def _run_tmux(cmd: list[str], *, input: str | None = None) -> Any:
    """Default tmux runner (injectable for tests)."""
    return subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=10)


def deliver_via_tmux(
    target: str,
    runtime: str | None,
    body: str,
    *,
    run: Callable[..., Any] | None = None,
    sleep: Callable[[float], Any] | None = None,
    buffer_name: str = _TMUX_BUFFER,
    large_threshold: int = LARGE_PASTE_THRESHOLD,
) -> TmuxDelivery:
    """Best-effort delivery into a live tmux pane. Never raises.

    On any failure (missing pane, no tmux binary, error) ``delivered`` is False
    and ``detail`` carries the reason, so the caller can fall back to the durable
    inbox without an exception escaping. ``run``/``sleep`` resolve to the module
    defaults at call time (not bound as parameter defaults) so tests can
    monkeypatch ``_run_tmux``.
    """
    run = run or _run_tmux
    sleep = sleep or time.sleep
    cmds, enters = build_tmux_commands(
        target, runtime, body, buffer_name=buffer_name, large_threshold=large_threshold
    )
    try:
        for cmd in cmds:
            is_load = cmd[:2] == ["tmux", "load-buffer"]
            is_send = cmd[:2] == ["tmux", "send-keys"]
            if is_send:
                # Pause before each Enter: the first lets the paste register
                # before expand; later ones keep expand and submit distinct.
                sleep(_SEND_KEY_PAUSE_SECONDS)
            result = run(cmd, input=body if is_load else None)
            returncode = getattr(result, "returncode", 0)
            if returncode != 0:
                detail = getattr(result, "stderr", "") or f"tmux exited {returncode}"
                return TmuxDelivery(
                    delivered=False,
                    target=target,
                    enters=enters,
                    detail=f"{cmd[1]} failed for {target}: {detail}".strip(),
                )
    except FileNotFoundError:
        return TmuxDelivery(
            delivered=False, target=target, enters=enters, detail="tmux not available"
        )
    except Exception as exc:  # never let delivery break the send
        return TmuxDelivery(
            delivered=False,
            target=target,
            enters=enters,
            detail=f"tmux delivery error: {exc}",
        )
    return TmuxDelivery(
        delivered=True,
        target=target,
        enters=enters,
        detail=f"pasted to {target} ({enters} Enter(s))",
    )


def _attempt_tmux_delivery(to_agent: str, body: str) -> TmuxDelivery | None:
    """Resolve the recipient's pane from the operator map and deliver, or None
    when no pane is configured (caller then reports inbox-only)."""
    pane = resolve_pane(to_agent, load_pane_map())
    if pane is None:
        return None
    return deliver_via_tmux(pane.target, pane.runtime, body)


# ---------------------------------------------------------------------------
# MCP tool function
# ---------------------------------------------------------------------------


def speak_to_agent(
    action: str = "read",
    to: str | None = None,
    message: str | None = None,
    message_id: str | None = None,
    reply_to: str | None = None,
    priority: str = PRIORITY_NORMAL,
    with_agent: str | None = None,
    limit: int = 20,
) -> str:
    """Structured agent-to-agent direct messaging with delivery guarantees.

    Unicast complement to recall_channel (broadcast).  Each message has an
    envelope with sender, recipient, timestamp, and delivery tracking.

    Args:
        action: "send", "read", "ack", "status", "history".
        to: Recipient agent ID (required for "send").
        message: Message body (required for "send").
        message_id: Message ID (required for "ack" and "status").
        reply_to: Parent message ID for threading (optional, "send" only).
        priority: "normal" (default) or "urgent" (surfaces first in read).
        with_agent: Agent ID for "history" action.
        limit: Max messages to return (default 20).
    """
    from synapt.recall.channel import _agent_id

    try:
        agent_id = _agent_id()
    except Exception:
        agent_id = None

    if not agent_id:
        return "Cannot determine your agent ID.  Join a channel first (recall_channel action='join')."

    try:
        if action == "send":
            if not to:
                return "Error: 'to' (recipient agent ID) is required for send."
            if not message:
                return "Error: 'message' body is required for send."
            recipient = resolve_registered_recipient(to)
            msg = send_message(
                from_agent=agent_id,
                to_agent=recipient.agent_id,
                body=message,
                reply_to=reply_to,
                priority=priority,
                recipient_store_coordinate=recipient.store_coordinate,
            )
            # The inbox write above is the durable record. Now also push the
            # message into the recipient's live tmux pane so it actually lands
            # (recall#852) -- best-effort, never breaks the send.
            delivery = _attempt_tmux_delivery(recipient.agent_id, message)
            if delivery is None:
                status = (
                    f"Status: written to {recipient.agent_id}'s inbox "
                    "(no tmux pane configured; inbox-only)"
                )
            elif delivery.delivered:
                status = (
                    f"Status: written to {recipient.agent_id}'s inbox AND "
                    f"delivered to live pane {delivery.target} via tmux"
                )
            else:
                status = (
                    f"Status: written to {recipient.agent_id}'s inbox; tmux delivery "
                    f"did not land ({delivery.detail})"
                )
            return f"Sent to {recipient.agent_id}: {msg.message_id}\n{status}"

        elif action == "read":
            own_recipient = resolve_registered_recipient(agent_id)
            messages = read_inbox(
                agent_id=agent_id,
                limit=limit,
                recipient_store_coordinate=(own_recipient.store_coordinate),
            )
            if not messages:
                return "No unread direct messages."
            lines = [f"## Direct messages ({len(messages)} unread)"]
            for msg in messages:
                pri = " [URGENT]" if msg.priority == PRIORITY_URGENT else ""
                reply = f" (reply to {msg.reply_to})" if msg.reply_to else ""
                lines.append(
                    f"  {msg.timestamp}  from {msg.from_agent}{pri}{reply}\n"
                    f"  [{msg.message_id}] {msg.body}"
                )
            return "\n".join(lines)

        elif action == "ack":
            if not message_id:
                return "Error: 'message_id' is required for ack."
            own_recipient = resolve_registered_recipient(agent_id)
            return ack_message(
                message_id=message_id,
                agent_id=agent_id,
                recipient_store_coordinate=(own_recipient.store_coordinate),
            )

        elif action == "status":
            if not message_id:
                return "Error: 'message_id' is required for status."
            info = check_status(message_id=message_id)
            if info is None:
                return f"Message {message_id} not found."
            return (
                f"Message {info['message_id']}\n"
                f"  From: {info['from_agent']} → To: {info['to_agent']}\n"
                f"  Status: {info['status']}\n"
                f"  Created: {info['created_at']}\n"
                f"  Delivered: {info['delivered_at']}\n"
                f"  Read: {info['read_at']}\n"
                f"  Acked: {info['acked_at']}"
            )

        elif action == "history":
            if not with_agent:
                return "Error: 'with_agent' is required for history."
            peer = resolve_registered_recipient(with_agent)
            messages = message_history(
                agent_id=agent_id,
                with_agent=peer.agent_id,
                limit=limit,
                include_canonical=True,
                canonical_store_coordinate=peer.store_coordinate,
            )
            if not messages:
                return f"No message history with {peer.agent_id}."
            lines = [f"## History with {peer.agent_id} ({len(messages)} messages)"]
            for msg in messages:
                direction = "→" if msg.from_agent == agent_id else "←"
                other = msg.to_agent if msg.from_agent == agent_id else msg.from_agent
                lines.append(
                    f"  {msg.timestamp}  {direction} {other}: {msg.body[:200]}"
                )
            return "\n".join(lines)

        else:
            return f"Unknown action '{action}'. Use: send, read, ack, status, history."

    except PermissionError as exc:
        return str(exc)
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Direct messaging failed: {exc}"
