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
import os
import sqlite3
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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

    # Write to recipient's inbox JSONL
    inbox = _inbox_path(to_agent, project_dir)
    with open(inbox, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")

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


def read_inbox(
    *,
    agent_id: str,
    limit: int = 20,
    project_dir: Path | None = None,
) -> list[DirectMessage]:
    """Read unread messages from an agent's inbox.  Marks them as READ."""
    conn = _get_db(project_dir)
    try:
        rows = conn.execute(
            """SELECT message_id, from_agent, to_agent, timestamp, body,
                      reply_to, priority
               FROM messages
               WHERE to_agent = ? AND status = ?
               ORDER BY
                   CASE WHEN priority = 'urgent' THEN 0 ELSE 1 END,
                   created_at ASC
               LIMIT ?""",
            (agent_id, STATUS_DELIVERED, limit),
        ).fetchall()

        messages = []
        for row in rows:
            msg = DirectMessage(
                message_id=row["message_id"],
                from_agent=row["from_agent"],
                to_agent=row["to_agent"],
                timestamp=row["timestamp"],
                body=row["body"],
                reply_to=row["reply_to"],
                priority=row["priority"],
            )
            messages.append(msg)
            _transition_status(conn, msg.message_id, STATUS_READ)

        return messages
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
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def _normalize_agent_key(to_agent: str) -> str:
    # Recipients may arrive bare ("apollo"), org-prefixed ("synapt:apollo"), or
    # cased; the pane map is keyed by the bare lower-case agent name.
    key = to_agent.strip().lower()
    if ":" in key:
        key = key.split(":", 1)[1]
    return key


def resolve_pane(to_agent: str, pane_map: dict[str, Any]) -> PaneTarget | None:
    """Resolve a recipient to its PaneTarget via the operator map, or None."""
    if not to_agent or not pane_map:
        return None
    entry = pane_map.get(to_agent) or pane_map.get(_normalize_agent_key(to_agent))
    if not isinstance(entry, dict) or not entry.get("target"):
        return None
    return PaneTarget(target=str(entry["target"]), runtime=str(entry.get("runtime", "claude")))


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
    cmds, enters = build_tmux_commands(target, runtime, body, buffer_name=buffer_name, large_threshold=large_threshold)
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
                return TmuxDelivery(delivered=False, target=target, enters=enters, detail=f"{cmd[1]} failed for {target}: {detail}".strip())
    except FileNotFoundError:
        return TmuxDelivery(delivered=False, target=target, enters=enters, detail="tmux not available")
    except Exception as exc:  # never let delivery break the send
        return TmuxDelivery(delivered=False, target=target, enters=enters, detail=f"tmux delivery error: {exc}")
    return TmuxDelivery(delivered=True, target=target, enters=enters, detail=f"pasted to {target} ({enters} Enter(s))")


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
            msg = send_message(
                from_agent=agent_id,
                to_agent=to,
                body=message,
                reply_to=reply_to,
                priority=priority,
            )
            # The inbox write above is the durable record. Now also push the
            # message into the recipient's live tmux pane so it actually lands
            # (recall#852) -- best-effort, never breaks the send.
            delivery = _attempt_tmux_delivery(to, message)
            if delivery is None:
                status = f"Status: written to {to}'s inbox (no tmux pane configured; inbox-only)"
            elif delivery.delivered:
                status = f"Status: written to {to}'s inbox AND delivered to live pane {delivery.target} via tmux"
            else:
                status = f"Status: written to {to}'s inbox; tmux delivery did not land ({delivery.detail})"
            return f"Sent to {to}: {msg.message_id}\n{status}"

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
