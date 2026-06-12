"""Agent-to-agent direct messaging -- unicast with delivery guarantees.

Storage layout (all under .synapt/recall/direct/):
  <agent_id>.jsonl  -- per-agent inbox log (append-only, open protocol)
  direct.db         -- SQLite for delivery state: status, acks, timestamps

Composes with recall_channel (broadcast) -- same storage root, different
semantic layer.  Channels are broadcast; direct messages are unicast with
delivery tracking and explicit acknowledgment.

Hook registration for premium:
  before_send hooks can reject messages (identity/auth/rate-limit).
  state_change hooks fire on status transitions (audit subscription).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

MAX_BODY_SIZE = 65536  # 64KB default cap

# ---------------------------------------------------------------------------
# Hook registration -- premium coordination layer
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
    base = _direct_dir(project_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{agent_id}.jsonl"


# recall#820 cross-org silo fix: the gripspace-local stores above (_direct_dir
# routes through _channels_dir(project_dir)) silo direct messages per-gripspace.
# A send from gripspace A lands in A's store; a read from gripspace B queries B's
# store and finds nothing. The canonical cross-org root below is
# project-INDEPENDENT (org-keyed, project component dropped) so every gripspace
# in the org agrees on one inbox per recipient. send_message dual-writes here;
# read_inbox union-reads from here. Design: config#359 Option A (Layne 2026-05-30).
_DEFAULT_ORG = "synapt-dev"


def _cross_org_root(project_dir: Path | None = None) -> Path:
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

    shared = _shared_channels_dir()
    if shared:
        return shared / "_cross-org" / "direct"
    org = _resolve_org_id(project_dir) or _DEFAULT_ORG
    return Path.home() / ".synapt" / "channels" / org / "_cross-org" / "direct"


def _cross_org_inbox_path(to_agent: str, project_dir: Path | None = None) -> Path:
    base = _cross_org_root(project_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{to_agent}.jsonl"


def _db_path(project_dir: Path | None = None) -> Path:
    base = _direct_dir(project_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / "direct.db"


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


def _get_db(project_dir: Path | None = None) -> sqlite3.Connection:
    path = _db_path(project_dir)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
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
) -> DirectMessage:
    """Send a direct message to an agent.  Returns the sent message."""
    if not from_agent:
        raise ValueError("from_agent is required")
    if not to_agent:
        raise ValueError("to_agent is required")
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
    cross_inbox = _cross_org_inbox_path(to_agent, project_dir)
    with open(cross_inbox, "a", encoding="utf-8") as f:
        f.write(line)

    # Track in SQLite
    conn = _get_db(project_dir)
    try:
        now = datetime.now(timezone.utc).timestamp()
        conn.execute(
            """INSERT OR IGNORE INTO messages
               (message_id, from_agent, to_agent, timestamp, body,
                reply_to, priority, status, created_at, delivered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            ),
        )
        conn.commit()
    finally:
        conn.close()

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
    cross_inbox = _cross_org_inbox_path(agent_id, project_dir)
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
    conn: sqlite3.Connection, msg: DirectMessage
) -> None:
    """Insert a local SQLite tracking row (status=READ) for a cross-org message.

    Gives the cross-org message a local delivery-state row so re-reads dedup it
    and ack_message can transition it. Idempotent via INSERT OR IGNORE.
    """
    now = datetime.now(timezone.utc).timestamp()
    conn.execute(
        """INSERT OR IGNORE INTO messages
           (message_id, from_agent, to_agent, timestamp, body, reply_to,
            priority, status, created_at, delivered_at, read_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        ),
    )
    conn.commit()


def read_inbox(
    *,
    agent_id: str,
    limit: int = 20,
    project_dir: Path | None = None,
) -> list[DirectMessage]:
    """Read unread messages from an agent's inbox.  Marks them as READ.

    recall#820: union of the local SQLite delivery store (intra-org fast path)
    and the project-independent cross-org canonical inbox (cross-gripspace
    path), deduped by message_id. Urgent messages surface first, then oldest
    first; only the returned (post-limit) messages are marked READ.
    """
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
            agent_id, conn, local_ids, project_dir
        )
        is_cross = {m.message_id for m in cross_msgs}

        combined = local_msgs + cross_msgs
        combined.sort(
            key=lambda m: (0 if m.priority == PRIORITY_URGENT else 1, m.timestamp)
        )
        returned = combined[:limit]

        for msg in returned:
            if msg.message_id in is_cross:
                _track_cross_org_read(conn, msg)
            else:
                _transition_status(conn, msg.message_id, STATUS_READ)

        return returned
    finally:
        conn.close()


def ack_message(
    *,
    message_id: str,
    agent_id: str,
    project_dir: Path | None = None,
) -> str:
    """Acknowledge a message.  Returns status string."""
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
            """SELECT message_id, from_agent, to_agent, status,
                      created_at, delivered_at, read_at, acked_at
               FROM messages WHERE message_id = ?""",
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def message_history(
    *,
    agent_id: str,
    with_agent: str,
    limit: int = 20,
    project_dir: Path | None = None,
) -> list[DirectMessage]:
    """Read message history between two agents (both directions)."""
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

        return [
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
    finally:
        conn.close()


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
            msg = send_message(
                from_agent=agent_id,
                to_agent=to,
                body=message,
                reply_to=reply_to,
                priority=priority,
            )
            return (
                f"Sent to {to}: {msg.message_id}\n"
                f"Status: delivered (written to {to}'s inbox)"
            )

        elif action == "read":
            messages = read_inbox(agent_id=agent_id, limit=limit)
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
            return ack_message(message_id=message_id, agent_id=agent_id)

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
            messages = message_history(
                agent_id=agent_id,
                with_agent=with_agent,
                limit=limit,
            )
            if not messages:
                return f"No message history with {with_agent}."
            lines = [f"## History with {with_agent} ({len(messages)} messages)"]
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
