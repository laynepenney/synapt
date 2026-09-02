"""Agent channel -- cross-worktree communication via append-only JSONL files.

Storage layout (all under .synapt/recall/channels/):
  <name>.jsonl    -- append-only message log (the open protocol)
  channels.db     -- SQLite for state: presence, memberships, cursors, pins, mutes

Any process that can append/read files can participate -- no daemon required.
External agents that cannot use SQLite can still write JSONL lines directly;
the SQLite layer is used by agents that import this module.

Cross-gripspace sharing:
  Set SYNAPT_SHARED_CHANNELS_DIR to a directory accessible by all gripspaces.
  Channel JSONL logs and attachments use the shared dir; presence/cursor state
  (channels.db) stays local per gripspace to avoid cross-contamination.

Agent identity: every agent has three layers, but only `id` is passed around.
  - id:           session hash (s_xxxxxxxx), primary key everywhere
  - griptree:     auto-detected (gripspace/repo), stored on join
  - display_name: configurable alias, stored on join, resolved at render time
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

from synapt.recall.core import (
    project_data_dir,
    _worktree_name,
    _find_gripspace_root,
    _resolve_project_root_override,
)


# ---------------------------------------------------------------------------
# Stale-detection thresholds
# ---------------------------------------------------------------------------

_ONLINE_MINUTES = 5       # < 5 min  => online
_IDLE_MINUTES = 30        # 5-30 min => idle
_AWAY_MINUTES = 120       # 30-120 min => away
                          # > 120 min => offline (auto-leave)
_JOIN_MENTION_LOOKBACK_MINUTES = 10  # How far back to scan for @mentions on join

# ---------------------------------------------------------------------------
# Hook registration — downstream coordination extension seam
# ---------------------------------------------------------------------------
# _append_message fires these hooks after writing a message to JSONL.
# A downstream layer may register additional coordination handlers.
# OSS auto-registers its built-in handlers; without premium, the built-in
# handlers still fire.  A downstream layer can replace or extend them.

from typing import Callable

_message_posted_hooks: list[Callable[["ChannelMessage", "Path | None"], None]] = []


# ---------------------------------------------------------------------------
# Store-path policy seam
# ---------------------------------------------------------------------------
# A consumer may install a callable that is consulted with (operation, path)
# immediately before any channel-store directory is created or any channel
# file is opened for writing. With no policy installed this is a no-op and
# behaviour is unchanged; the seam exists so a harness can refuse a resolved
# path *before* the write rather than detect it afterwards.
#
# The seam sits at resolution, not only at the write calls, because resolving
# the global store directory creates it as a side effect — by the time an
# append is attempted the directory already exists.

_store_path_policy: Callable[[str, "Path"], None] | None = None


def set_store_path_policy(
    policy: Callable[[str, "Path"], None] | None,
) -> Callable[[str, "Path"], None] | None:
    """Install a channel-store path policy; returns the previous one."""
    global _store_path_policy
    previous = _store_path_policy
    _store_path_policy = policy
    return previous


def _guard_store_path(operation: str, path: Path) -> Path:
    """Consult the installed policy, then return *path* unchanged."""
    policy = _store_path_policy
    if policy is not None:
        policy(operation, path)
    return path


def register_message_hook(
    hook: Callable[["ChannelMessage", "Path | None"], None],
) -> None:
    """Register a callback invoked after a message is appended to JSONL.

    Hooks receive (msg, project_dir) and run synchronously after the write.
    A downstream layer may use this to register additional handlers.
    """
    _message_posted_hooks.append(hook)


def _clear_message_hooks() -> None:
    """Reset hooks — for tests only."""
    _message_posted_hooks.clear()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _read_manifest_url(project_dir: Path | None = None) -> str | None:
    """Read the manifest URL from gripspace.yml, or None if not in a gripspace.

    Honors the SAME env-var root override as project_data_dir() (Tier-3
    local/state store), via the shared ``_resolve_project_root_override`` —
    consulted only when project_dir is None, an explicitly-passed
    project_dir being a deliberate root that suppresses every env override,
    same contract as project_data_dir. One resolver decides the coordinate
    for both stores now; before this fix, project_data_dir additionally
    honored SYNAPT_RECALL_ROOT outright while
    this function considered only GRIPSPACE_ROOT, so a caller who set
    SYNAPT_RECALL_ROOT to redirect the local store got the log routed to
    the real gripspace's Tier-2 global path instead — two different
    gripspaces, not the by-design "log Tier-2, state Tier-3" split. The
    GRIPSPACE_ROOT-only gap this originally closed is recall#916 (`#dev`
    resolving to two different files depending on caller cwd).
    """
    if project_dir is None:
        override = _resolve_project_root_override(project_dir)
        if override is not None:
            project_dir = override
    root = _find_gripspace_root(project_dir or Path.cwd())
    if root is None:
        return None
    manifest_path = root / ".gitgrip" / "spaces" / "main" / "gripspace.yml"
    if not manifest_path.exists():
        return None
    try:
        # Lightweight YAML parse — just find the manifest url line
        with open(manifest_path, encoding="utf-8") as f:
            in_manifest = False
            for line in f:
                stripped = line.strip()
                if stripped == "manifest:":
                    in_manifest = True
                    continue
                if in_manifest and stripped.startswith("url:"):
                    return stripped.split(":", 1)[1].strip()
                if in_manifest and not line.startswith(" ") and not line.startswith("\t"):
                    break  # Left the manifest section
    except OSError:
        pass
    return None


def _extract_org_from_url(url: str) -> str | None:
    """Extract the GitHub org from a git URL.

    git@github.com:synapt-dev/gripspace.git → 'synapt-dev'
    https://github.com/synapt-dev/gripspace.git → 'synapt-dev'
    """
    if ":" in url and "@" in url:
        # SSH format: git@github.com:org/repo.git
        path_part = url.split(":", 1)[1]
    elif "github.com/" in url:
        # HTTPS format
        path_part = url.split("github.com/", 1)[1]
    else:
        return None
    parts = path_part.strip("/").split("/")
    return parts[0] if parts else None


def _extract_repo_from_url(url: str) -> str | None:
    """Extract the repo slug from a git URL (without .git suffix).

    git@github.com:synapt-dev/gripspace.git → 'gripspace'
    """
    if ":" in url and "@" in url:
        path_part = url.split(":", 1)[1]
    elif "github.com/" in url:
        path_part = url.split("github.com/", 1)[1]
    else:
        return None
    parts = path_part.strip("/").split("/")
    if len(parts) >= 2:
        repo = parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        return repo
    return None


def _resolve_org_id(project_dir: Path | None = None) -> str | None:
    """Derive org_id from the gripspace manifest URL's GitHub org."""
    url = _read_manifest_url(project_dir)
    if url:
        return _extract_org_from_url(url)
    return None


def _resolve_project_id(project_dir: Path | None = None) -> str | None:
    """Derive project_id from the gripspace manifest URL's repo name."""
    url = _read_manifest_url(project_dir)
    if url:
        return _extract_repo_from_url(url)
    return None


def _global_channels_dir(project_dir: Path | None = None) -> Path | None:
    """Return the global channel store path if inside a gripspace.

    Routes to ~/.synapt/channels/<org_id>/<project_id>/.
    Returns None if org/project can't be resolved.
    """
    org = _resolve_org_id(project_dir)
    project = _resolve_project_id(project_dir)
    if org and project:
        return Path.home() / ".synapt" / "channels" / org / project
    return None


def _shared_channels_dir() -> Path | None:
    """Return the shared channels directory from env var, or None."""
    shared = os.environ.get("SYNAPT_SHARED_CHANNELS_DIR")
    if shared:
        return Path(shared)
    return None


def _local_channels_dir(project_dir: Path | None = None) -> Path:
    """Return the local (per-gripspace) channels directory."""
    return project_data_dir(project_dir) / "channels"


def _channels_dir(project_dir: Path | None = None) -> Path:
    """Return the channels directory for JSONL logs and attachments.

    Three-tier resolution:
    1. SYNAPT_SHARED_CHANNELS_DIR env var (explicit override, backward compat)
    2. Global store ~/.synapt/channels/<org>/<project>/ (if in a gripspace)
    3. Local per-gripspace directory (fallback for non-gripspace repos)
    """
    # Tier 1: explicit env var override
    shared = _shared_channels_dir()
    if shared:
        return _guard_store_path("resolve_channels_dir", shared)
    # Tier 2: global store from manifest URL
    global_dir = _global_channels_dir(project_dir)
    if global_dir:
        # Guarded before the mkdir: resolving this tier creates the directory,
        # so a check placed after it would be reporting a write, not preventing
        # one.
        _guard_store_path("resolve_channels_dir", global_dir)
        global_dir.mkdir(parents=True, exist_ok=True)
        return global_dir
    # Tier 3: local fallback
    return _guard_store_path("resolve_channels_dir", _local_channels_dir(project_dir))


def _orphaned_local_channel_store(project_dir: Path | None = None) -> Path | None:
    """Return the legacy local channels dir if it looks orphaned, else None.

    "Orphaned" means: this call's log resolves Tier-2 GLOBAL (so the local
    Tier-3 directory is no longer the live log for this call), AND that
    local directory already exists on disk with at least one channel
    JSONL file in it -- the exact shape of a reported production symptom:
    a gripspace-local ``dev.jsonl`` that "stayed behind looking live" after
    the real log moved to the global store, its ``channels.db`` sitting
    right beside it and still being written (the state db is Tier-3 local
    by design), which is what made the leftover JSONL read as current to
    a casual reader.

    Detect and report ONLY -- never delete. Whether the leftover file
    still matters (backup it, migrate it, or leave it) is a human's call,
    not this function's.
    """
    if _global_channels_dir(project_dir) is None:
        return None  # this call's log is Tier-3 local already; nothing orphaned
    local_dir = _local_channels_dir(project_dir)
    if not local_dir.is_dir():
        return None
    if any(local_dir.glob("*.jsonl")):
        return local_dir
    return None


def _channel_to_filename(channel: str) -> str:
    """Encode a channel name for use as a filename (Windows-safe)."""
    return channel.replace(":", "--")


def _filename_to_channel(stem: str) -> str:
    """Decode a filename stem back to a channel name."""
    return stem.replace("--", ":")


def _channel_path(channel: str, project_dir: Path | None = None) -> Path:
    """Return the JSONL log path for a channel."""
    ch_dir = _channels_dir(project_dir)
    safe_name = _channel_to_filename(channel)
    new_path = ch_dir / f"{safe_name}.jsonl"
    # Migrate legacy colon-format files (not valid on Windows)
    if ":" in channel and not new_path.exists():
        old_path = ch_dir / f"{channel}.jsonl"
        if old_path.exists():
            old_path.rename(new_path)
    return new_path


def _db_path(project_dir: Path | None = None) -> Path:
    """Return the SQLite database path for channel state.

    Always uses the local directory — presence, cursors, pins, and mutes
    are per-gripspace even when channels are shared.

    Deliberately NOT guarded by the channel-store policy, and the reason is
    load-bearing rather than an oversight. This path is composed from
    ``project_data_dir``, so it can never resolve inside the global channel
    store; a channel-store check here could not refuse anything, and a check
    that cannot fail is the defect this seam exists to prevent. It is instead
    covered by the data-root policy, which ``project_data_dir`` consults —
    verified by probe, not assumed. See ``_open_state_db`` for the SQLite path
    that CAN reach the global store and is guarded accordingly.
    """
    return _local_channels_dir(project_dir) / "channels.db"


def _presence_path(project_dir: Path | None = None) -> Path:
    """Return the legacy presence JSON file path (for migration)."""
    return _local_channels_dir(project_dir) / "_presence.json"


def _attachments_dir(project_dir: Path | None = None) -> Path:
    """Return the channel attachments directory."""
    return _channels_dir(project_dir) / "attachments"


# ---------------------------------------------------------------------------
# Agent identity
# ---------------------------------------------------------------------------

_AGENT_ID_CACHE: dict[str, str] = {}


def _agent_id(project_dir: Path | None = None, name: str | None = None) -> str:
    """Return the agent's stable identity.

    Resolution order:
    1. SYNAPT_AGENT_ID env var (set by gr spawn) — stable, org-registered
    2. Named agent hash (a_xxxxxxxx) — when ``name`` is provided, creates a
       distinct identity per agent name within the same griptree. Fixes
       recall#590: agents sharing a griptree with a human session no longer
       collide on agent_id.
    3. Session-scoped hash (s_xxxxxxxx) — fallback for manual/unregistered sessions

    When SYNAPT_AGENT_ID is set, the agent uses its registered org identity
    for all channel operations (cursors, claims, DMs). This is the Phase 0
    integration point for the org agent registry.
    """
    # Phase 0: registered agent identity takes priority
    registered_id = os.environ.get("SYNAPT_AGENT_ID")
    if registered_id:
        return registered_id

    # Named agent: derive a distinct ID from name + griptree so multiple
    # callers in the same griptree get separate presence rows.
    if name:
        gt = _resolve_griptree(project_dir)
        seed = f"named:{gt}:{name}"
        return "a_" + hashlib.sha256(seed.encode()).hexdigest()[:8]

    # Fallback: session-scoped hash (backward compat for manual sessions)
    key = str(project_data_dir(project_dir))
    if key not in _AGENT_ID_CACHE:
        gt = _resolve_griptree(project_dir)
        ppid = os.getppid()
        seed = f"{gt}:{key}:{ppid}"
        _AGENT_ID_CACHE[key] = "s_" + hashlib.sha256(seed.encode()).hexdigest()[:8]
    return _AGENT_ID_CACHE[key]


def _resolve_griptree(project_dir: Path | None = None) -> str:
    """Auto-detect griptree identity: gripspace_name/repo_name.

    Derives the gripspace root from project_data_dir() by verifying the
    expected ``.synapt/recall`` suffix rather than blindly using parents[1].
    """
    try:
        data_dir = project_data_dir(project_dir)
        # data_dir should be <gripspace>/.synapt/recall
        # Verify the expected suffix before stripping
        if data_dir.parent.name == ".synapt" and data_dir.name == "recall":
            gripspace = data_dir.parent.parent
        else:
            return _worktree_name(project_dir)
        repo = Path.cwd()
        # If cwd is inside gripspace, use relative path
        try:
            rel = repo.relative_to(gripspace)
            # When cwd IS the gripspace root, relative_to returns Path(".")
            if rel == Path("."):
                return gripspace.name
            return f"{gripspace.name}/{rel}"
        except ValueError:
            return gripspace.name
    except Exception:
        return _worktree_name(project_dir)


def _resolve_display_name_for(agent_id: str, project_dir: Path | None = None) -> str:
    """Resolve display name for a specific agent_id from the presence DB."""
    try:
        conn = _open_db(project_dir)
        try:
            row = conn.execute(
                "SELECT display_name FROM presence WHERE agent_id = ?", (agent_id,)
            ).fetchone()
            if row and row["display_name"]:
                return row["display_name"]
        finally:
            conn.close()
    except Exception:
        pass
    return agent_id


def _resolve_display_name(project_dir: Path | None = None) -> str:
    """Resolve display name: env var > presence DB > griptree fallback."""
    name = os.environ.get("SYNAPT_AGENT_NAME", "")
    if name:
        return name
    # Check presence table for a display_name set via rename action
    try:
        aid = _agent_id(project_dir)
        conn = _open_db(project_dir)
        try:
            row = conn.execute(
                "SELECT display_name FROM presence WHERE agent_id = ?", (aid,)
            ).fetchone()
            if row and row["display_name"]:
                return row["display_name"]
        finally:
            conn.close()
    except Exception:
        pass
    return _resolve_griptree(project_dir)


def _normalize_display_name(name: str) -> str:
    """Normalize a display name for uniqueness checks."""
    return " ".join(name.split()).strip().casefold()


def _find_display_name_conflict(
    conn: sqlite3.Connection,
    display_name: str,
    agent_id: str,
) -> sqlite3.Row | None:
    """Return another online/active agent already using this display name.

    Stale agents (anything no longer "online" in the <5 minute heartbeat
    window) do not block name claims. Their old claim is cleared eagerly
    so display-name targeting stays unambiguous after the new join/rename.
    """
    normalized = _normalize_display_name(display_name)
    if not normalized:
        return None

    rows = conn.execute(
        "SELECT agent_id, display_name, status, last_seen FROM presence WHERE agent_id != ?",
        (agent_id,),
    ).fetchall()
    for row in rows:
        existing = row["display_name"] or ""
        if not existing:
            continue
        if _normalize_display_name(existing) != normalized:
            continue
        status = _agent_status(row["last_seen"])
        # Release name claims from stale agents before allowing reuse.
        # Only "online" (<5 min) agents truly hold a live claim.
        if status != "online":
            conn.execute(
                "UPDATE presence SET display_name = '' WHERE agent_id = ?",
                (row["agent_id"],),
            )
            continue
        return row
    return None


def _resolve_target_id(target: str, conn: sqlite3.Connection) -> str:
    """Resolve a target to an agent_id.

    Accepts an ``s_*`` agent_id directly, or looks up by display_name/griptree
    in the presence table. Returns the original string if no match is found.
    """
    if target.startswith("s_"):
        return target
    # Try display_name first, then griptree
    row = conn.execute(
        "SELECT agent_id FROM presence WHERE display_name = ? OR griptree = ? LIMIT 1",
        (target, target),
    ).fetchone()
    return row["agent_id"] if row else target


def _seed_cursor_for_join(
    conn: sqlite3.Connection,
    *,
    agent_id: str,
    channel: str,
    display_name: str,
    griptree: str,
    project_dir: Path | None = None,
) -> str:
    """Choose the initial unread cursor for a joining session.

    Brand-new identities should start at the current channel tail so old
    backlog does not flood their first unread poll. But restarted sessions
    with a new session hash should inherit the most recent cursor from an
    earlier matching identity so downtime messages remain unread.
    """
    existing = conn.execute(
        "SELECT last_read_at FROM cursors WHERE agent_id = ? AND channel = ?",
        (agent_id, channel),
    ).fetchone()
    if existing:
        return existing["last_read_at"]

    # Primary: match on griptree (stable, unique, not user-configurable).
    # Fallback: match on display_name (for backward compat with pre-griptree
    # sessions that only have display_name in presence).
    if griptree:
        row = conn.execute(
            """
            SELECT c.last_read_at
            FROM cursors c
            JOIN presence p ON p.agent_id = c.agent_id
            WHERE c.channel = ?
              AND c.agent_id != ?
              AND p.griptree = ?
            ORDER BY c.last_read_at DESC
            LIMIT 1
            """,
            (channel, agent_id, griptree),
        ).fetchone()
        if row:
            return row["last_read_at"]

    if display_name:
        row = conn.execute(
            """
            SELECT c.last_read_at
            FROM cursors c
            JOIN presence p ON p.agent_id = c.agent_id
            WHERE c.channel = ?
              AND c.agent_id != ?
              AND p.display_name != '' AND p.display_name = ?
            ORDER BY c.last_read_at DESC
            LIMIT 1
            """,
            (channel, agent_id, display_name),
        ).fetchone()
        if row:
            return row["last_read_at"]

    # Presence rows may already be gone after session end / stale reap, but the
    # channel log still carries the old session id on prior join/leave/messages.
    # Recover the latest matching session from the log and inherit its cursor.
    # Prefer griptree match, fall back to display_name.
    path = _channel_path(channel, project_dir)
    if path.exists():
        for msg in reversed(_read_messages(path)):
            if msg.from_agent == agent_id:
                continue
            # Check if this old agent shared our griptree or display_name
            prior = conn.execute(
                "SELECT griptree, display_name FROM presence WHERE agent_id = ?",
                (msg.from_agent,),
            ).fetchone()
            if prior:
                if griptree and prior["griptree"] == griptree:
                    pass  # match
                elif display_name and (prior["display_name"] or "") == display_name:
                    pass  # match
                else:
                    continue
            elif display_name and (msg.from_display or "") == display_name:
                pass  # presence gone, fall back to JSONL display name
            else:
                continue
            row = conn.execute(
                "SELECT last_read_at FROM cursors WHERE agent_id = ? AND channel = ?",
                (msg.from_agent, channel),
            ).fetchone()
            if row:
                return row["last_read_at"]

    last_ts = _read_last_timestamp(_channel_path(channel, project_dir))
    if last_ts:
        return last_ts
    return _now_iso()


# ---------------------------------------------------------------------------
# Message ID generation
# ---------------------------------------------------------------------------


def _generate_msg_id(timestamp: str, agent_id: str, body: str) -> str:
    """Generate a short deterministic message ID (m_xxxxxxxx).

    Uses 8 hex chars (32 bits) from SHA-256. Collision probability is ~50%
    at ~65k messages per channel (birthday bound). Acceptable for channel
    volumes; pins and directives reference IDs within a single channel.
    """
    seed = f"{timestamp}{agent_id}{body}"
    return "m_" + hashlib.sha256(seed.encode()).hexdigest()[:8]


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string with microsecond precision.

    Microsecond precision ensures that operations within the same second
    produce distinct timestamps for correct cursor-based unread counting.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp to a timezone-aware datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChannelMessage:
    """One message in a channel log."""

    timestamp: str
    channel: str
    type: str  # "message", "join", "leave", "directive"
    body: str = ""
    # Use field name "from_agent" in Python to avoid shadowing builtin "from".
    # Serialized as "from" in JSON for protocol compatibility.
    from_agent: str = ""
    from_display: str = ""  # display name at post time (persisted in JSONL)
    id: str = ""       # auto-generated message ID (m_xxxxxxxx)
    to: str = ""       # directive target agent (optional)
    worktree: str = ""  # sender's worktree/griptree name
    attachments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["from"] = d.pop("from_agent")
        # Omit empty optional fields to keep JSONL clean
        if not d.get("id"):
            d.pop("id", None)
        if not d.get("to"):
            d.pop("to", None)
        if not d.get("worktree"):
            d.pop("worktree", None)
        if not d.get("attachments"):
            d.pop("attachments", None)
        if not d.get("from_display"):
            d.pop("from_display", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ChannelMessage:
        # Accept "from" key from JSON, map to from_agent
        mapped = dict(d)
        if "from" in mapped:
            mapped["from_agent"] = mapped.pop("from")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in mapped.items() if k in known})


# ---------------------------------------------------------------------------
# SQLite state layer
# ---------------------------------------------------------------------------

_CHANNEL_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS presence (
    agent_id TEXT PRIMARY KEY,
    griptree TEXT NOT NULL DEFAULT '',
    display_name TEXT DEFAULT '',
    role TEXT NOT NULL DEFAULT 'agent',
    status TEXT DEFAULT 'online',
    last_seen TEXT NOT NULL,
    joined_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memberships (
    agent_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    joined_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, channel)
);

CREATE TABLE IF NOT EXISTS cursors (
    agent_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    last_read_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, channel)
);

CREATE TABLE IF NOT EXISTS pins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    body TEXT NOT NULL,
    message_id TEXT DEFAULT '',
    pinned_by TEXT NOT NULL,
    pinned_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mutes (
    agent_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    muted_by TEXT NOT NULL,
    muted_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, channel, muted_by)
);

CREATE TABLE IF NOT EXISTS claims (
    message_id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    claimed_by TEXT NOT NULL,
    claimed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    mentioned TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mentions_mentioned ON mentions(mentioned);

CREATE TABLE IF NOT EXISTS wake_requests (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    target TEXT NOT NULL,
    reason TEXT NOT NULL,
    priority INTEGER NOT NULL,
    source TEXT DEFAULT '',
    payload TEXT DEFAULT '',
    created TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wake_requests_target_seq
    ON wake_requests(target, seq);

CREATE TABLE IF NOT EXISTS status_board (
    channel TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (channel, agent_id)
);

CREATE TABLE IF NOT EXISTS status_board_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unread_flags (
    agent_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    dirty INTEGER DEFAULT 0,
    last_cleared_at TEXT,
    PRIMARY KEY (agent_id, channel)
);
"""

_WAKE_PRIORITIES = {
    "retry": -1,
    "interval": 0,
    "channel_activity": 1,
    "mention": 2,
    "directive": 3,
    "user_action": 4,
}


def _open_db(project_dir: Path | None = None) -> sqlite3.Connection:
    """Open (or create) the channels SQLite database with WAL mode."""
    path = _db_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    # Migrate old schema (agent → agent_id) before creating new tables
    _migrate_schema(conn)
    conn.executescript(_CHANNEL_SCHEMA_SQL)
    # Migrate legacy _presence.json if it exists
    _migrate_legacy_presence(conn, project_dir)
    return conn


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Migrate old schema columns if needed."""
    # Check if presence table exists with old 'agent' column
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(presence)").fetchall()]
    except sqlite3.OperationalError:
        return  # Table doesn't exist yet, CREATE will handle it
    if not cols:
        return
    if "agent" in cols and "agent_id" not in cols:
        # Old schema — recreate tables with new column names
        conn.executescript("""
            ALTER TABLE presence RENAME COLUMN agent TO agent_id;
            ALTER TABLE memberships RENAME COLUMN agent TO agent_id;
            ALTER TABLE cursors RENAME COLUMN agent TO agent_id;
        """)
    # Add new columns if missing
    if "griptree" not in cols:
        try:
            conn.execute("ALTER TABLE presence ADD COLUMN griptree TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass
    if "display_name" not in cols:
        try:
            conn.execute("ALTER TABLE presence ADD COLUMN display_name TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
    # Add message_id to pins if missing
    try:
        pin_cols = [r[1] for r in conn.execute("PRAGMA table_info(pins)").fetchall()]
        if pin_cols and "message_id" not in pin_cols:
            conn.execute("ALTER TABLE pins ADD COLUMN message_id TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    # Add role column to presence if missing
    if "role" not in cols:
        try:
            conn.execute("ALTER TABLE presence ADD COLUMN role TEXT NOT NULL DEFAULT 'agent'")
        except sqlite3.OperationalError:
            pass
    conn.commit()


def _migrate_legacy_presence(conn: sqlite3.Connection, project_dir: Path | None = None) -> None:
    """Migrate data from legacy _presence.json to SQLite, then delete the file."""
    path = _presence_path(project_dir)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt file -- just remove it
        try:
            path.unlink()
        except OSError:
            pass
        return

    if not data:
        try:
            path.unlink()
        except OSError:
            pass
        return

    for agent, info in data.items():
        joined_at = info.get("joined_at", _now_iso())
        last_seen = info.get("last_seen", joined_at)
        channels = info.get("channels", [])

        conn.execute(
            "INSERT OR IGNORE INTO presence (agent_id, status, last_seen, joined_at) "
            "VALUES (?, 'online', ?, ?)",
            (agent, last_seen, joined_at),
        )
        for ch in channels:
            conn.execute(
                "INSERT OR IGNORE INTO memberships (agent_id, channel, joined_at) "
                "VALUES (?, ?, ?)",
                (agent, ch, joined_at),
            )

    conn.commit()

    # Remove legacy file
    try:
        path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Stale detection and auto-leave
# ---------------------------------------------------------------------------

def _agent_status(last_seen: str) -> str:
    """Determine agent status based on last_seen timestamp."""
    now = datetime.now(timezone.utc)
    try:
        last_dt = _parse_iso(last_seen)
        delta = now - last_dt
        minutes = delta.total_seconds() / 60
        if minutes < _ONLINE_MINUTES:
            return "online"
        if minutes < _IDLE_MINUTES:
            return "idle"
        if minutes < _AWAY_MINUTES:
            return "away"
        return "offline"
    except (ValueError, TypeError):
        return "online"


def _reap_stale_agents(conn: sqlite3.Connection, project_dir: Path | None = None) -> list[str]:
    """Find agents that are > 2 hours stale, auto-leave them.

    Returns list of agents that were reaped.
    """
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(minutes=_AWAY_MINUTES)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Find stale agents
    rows = conn.execute(
        "SELECT agent_id, last_seen FROM presence WHERE last_seen < ?",
        (cutoff,),
    ).fetchall()

    reaped = []
    for row in rows:
        aid = row["agent_id"]
        last_seen = row["last_seen"]

        if _agent_status(last_seen) != "offline":
            continue

        # Determine if this agent is ephemeral:
        # - s_* are session-scoped fallback identities (MCP processes, CLI)
        # - agents whose display_name equals their griptree name are
        #   unnamed sessions that resolved to the bare repo name
        is_ephemeral = aid.startswith("s_")
        if not is_ephemeral:
            row_display = conn.execute(
                "SELECT display_name, griptree FROM presence WHERE agent_id = ?",
                (aid,),
            ).fetchone()
            if (row_display
                    and row_display["display_name"]
                    and row_display["griptree"]
                    and row_display["display_name"] == row_display["griptree"]):
                is_ephemeral = True

        channels = [
            r["channel"]
            for r in conn.execute(
                "SELECT channel FROM memberships WHERE agent_id = ?", (aid,)
            ).fetchall()
        ]

        # Never write leave messages to the channel log. The presence
        # table already tracks online/offline status; JSONL leave events
        # are redundant noise that clutters the feed. (recall#677)

        conn.execute("DELETE FROM claims WHERE claimed_by = ?", (aid,))
        conn.execute(
            "UPDATE presence SET status = 'offline' WHERE agent_id = ?", (aid,)
        )
        reaped.append(aid)

    if reaped:
        conn.commit()
    return reaped


# ---------------------------------------------------------------------------
# JSONL message I/O (unchanged -- this is the open protocol)
# ---------------------------------------------------------------------------

_MENTION_RE = re.compile(r"@(\w[\w.-]*)")


def _extract_mentions(text: str) -> list[str]:
    """Extract @mentioned names from message text."""
    return list(dict.fromkeys(_MENTION_RE.findall(text)))


def _store_mentions(
    msg: ChannelMessage, project_dir: Path | None = None,
) -> None:
    """Parse @mentions from a message and store them in the mentions table."""
    names = _extract_mentions(msg.body)
    if not names:
        return
    conn = _open_db(project_dir)
    try:
        # Build a lookup of all known identities → agent_id
        # Matches: display_name, agent_id, griptree, last griptree segment
        identity_map: dict[str, str] = {}
        for row in conn.execute(
            "SELECT agent_id, display_name, griptree FROM presence"
        ).fetchall():
            aid = row["agent_id"]
            identity_map[aid.lower()] = aid
            if row["display_name"]:
                identity_map[row["display_name"].lower()] = aid
            if row["griptree"]:
                identity_map[row["griptree"].lower()] = aid
                # Also match last segment (e.g., "synapt" from "synapt/synapt")
                parts = row["griptree"].split("/")
                if len(parts) > 1:
                    identity_map[parts[-1].lower()] = aid

        # Collect all unique agent_ids from the identity map (for @team/@all)
        all_agents = sorted(set(identity_map.values()))

        def _insert_mention_and_wake(agent_id: str) -> None:
            conn.execute(
                "INSERT INTO mentions (message_id, channel, mentioned, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (msg.id, msg.channel, agent_id, msg.timestamp),
            )
            _enqueue_wake_request(
                conn,
                target=f"agent:{agent_id}",
                reason="mention",
                source=msg.from_agent,
                payload={
                    "message_id": msg.id,
                    "channel": msg.channel,
                    "type": msg.type,
                    "mentioned": agent_id,
                },
                created=msg.timestamp,
            )

        seen: set[str] = set()
        for name in names:
            lower = name.lower()
            if lower in ("team", "all"):
                # Expand to all active agents except the sender
                for aid in all_agents:
                    if aid == msg.from_agent or aid in seen:
                        continue
                    seen.add(aid)
                    _insert_mention_and_wake(aid)
            else:
                mentioned = identity_map.get(lower, name)
                if mentioned in seen:
                    continue
                seen.add(mentioned)
                _insert_mention_and_wake(mentioned)
        conn.commit()
    finally:
        conn.close()


def _wake_reason_for_message(
    msg: ChannelMessage,
    conn: sqlite3.Connection,
) -> str:
    """Classify the base wake reason for a channel message."""
    if msg.type == "directive":
        return "directive"
    row = conn.execute(
        "SELECT role FROM presence WHERE agent_id = ?",
        (msg.from_agent,),
    ).fetchone()
    if row and row["role"] == "human":
        return "user_action"
    return "channel_activity"


def _enqueue_wake_request(
    conn: sqlite3.Connection,
    target: str,
    reason: str,
    source: str,
    payload: dict | None,
    created: str,
) -> None:
    """Persist one wake request in the durable cross-process wake queue."""
    conn.execute(
        "INSERT INTO wake_requests (target, reason, priority, source, payload, created) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            target,
            reason,
            _WAKE_PRIORITIES[reason],
            source,
            json.dumps(payload or {}, sort_keys=True),
            created,
        ),
    )


def _emit_message_wakes(msg: ChannelMessage, project_dir: Path | None = None) -> None:
    """Emit durable wake requests for channel-level activity."""
    if msg.type not in {"message", "directive"}:
        return

    conn = _open_db(project_dir)
    try:
        reason = _wake_reason_for_message(msg, conn)
        payload = {
            "message_id": msg.id,
            "channel": msg.channel,
            "type": msg.type,
        }
        if msg.to:
            payload["to"] = msg.to
        _enqueue_wake_request(
            conn,
            target=f"channel:{msg.channel}",
            reason=reason,
            source=msg.from_agent,
            payload=payload,
            created=msg.timestamp,
        )
        if msg.type == "directive" and msg.to and msg.to != "*":
            _enqueue_wake_request(
                conn,
                target=f"agent:{msg.to}",
                reason="directive",
                source=msg.from_agent,
                payload=payload,
                created=msg.timestamp,
            )
        conn.commit()
    finally:
        conn.close()


def _append_message(
    msg: ChannelMessage,
    project_dir: Path | None = None,
    channels_dir: Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Append a message to a channel's JSONL log with file locking.

    Auto-generates a message ID if not already set.
    Also parses and stores any @mentions found in the message body.
    When ``channels_dir`` is provided it overrides the normal path resolution
    so the message lands in the correct cross-project channel directory.
    When ``conn`` is provided it is reused for dirty-flag writes instead of
    opening a new connection (avoids SQLite lock contention when the caller
    already holds a transaction).
    """
    # Unescape literal \n and \t that LLM tool calls often produce.
    # MCP arguments are JSON strings, but models sometimes double-escape
    # newlines (sending "\\n" instead of "\n"), resulting in literal
    # backslash-n in the body.  Fixes recall#493.
    if msg.body:
        msg.body = msg.body.replace("\\n", "\n").replace("\\t", "\t")
    if not msg.id:
        msg.id = _generate_msg_id(msg.timestamp, msg.from_agent, msg.body)
    if channels_dir is not None:
        path = channels_dir / f"{_channel_to_filename(msg.channel)}.jsonl"
    else:
        path = _channel_path(msg.channel, project_dir)
    # An explicit channels_dir bypasses resolution entirely, so the write
    # surface is guarded in its own right rather than trusting the resolver.
    _guard_store_path("append_message", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from synapt.recall._filelock import lock_exclusive
    with open(path, "a", encoding="utf-8") as f:
        lock_exclusive(f)
        f.write(json.dumps(msg.to_dict()) + "\n")
        f.flush()
    # Set dirty flag for all other members of this channel (OSS substrate)
    _set_dirty_flags(msg.channel, msg.from_agent, project_dir, conn=conn)
    # Fire registered hooks (mention storage, wake emission, etc.)
    for hook in _message_posted_hooks:
        try:
            hook(msg, project_dir)
        except Exception:
            pass  # hooks are best-effort


def _set_dirty_flags(
    channel: str,
    sender_id: str,
    project_dir: Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Mark all other channel members as having unread messages."""
    owns_conn = conn is None
    if owns_conn:
        conn = _open_db(project_dir)
    try:
        members = conn.execute(
            "SELECT agent_id FROM memberships WHERE channel = ? AND agent_id != ?",
            (channel, sender_id),
        ).fetchall()
        for row in members:
            conn.execute(
                "INSERT INTO unread_flags (agent_id, channel, dirty) "
                "VALUES (?, ?, 1) "
                "ON CONFLICT(agent_id, channel) DO UPDATE SET dirty = 1",
                (row["agent_id"], channel),
            )
        conn.commit()
    finally:
        if owns_conn:
            conn.close()


def channel_has_unread(
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> dict[str, bool]:
    """Fast O(1) check for unread messages per channel.

    Returns a dict mapping channel names to whether they have unread messages.
    Uses the dirty-flag table instead of scanning JSONL files.
    Returns empty dict if the agent has no flags set (no memberships or
    everything is caught up).
    """
    aid = agent_name or _agent_id(project_dir)
    conn = _open_db(project_dir)
    try:
        rows = conn.execute(
            "SELECT channel, dirty FROM unread_flags WHERE agent_id = ?",
            (aid,),
        ).fetchall()
        return {r["channel"]: bool(r["dirty"]) for r in rows}
    finally:
        conn.close()


def channel_read_wakes(
    targets: str | list[str],
    after_seq: int = 0,
    limit: int = 100,
    project_dir: Path | None = None,
) -> list[dict]:
    """Read wake requests for one or more targets after a cursor.

    Intended for a single consumer per target set. Concurrent consumers
    watching the same targets should coordinate cursor/ack handling.
    """
    if isinstance(targets, str):
        target_list = [targets]
    else:
        target_list = [t for t in targets if t]
    if not target_list:
        return []

    conn = _open_db(project_dir)
    try:
        placeholders = ",".join("?" for _ in target_list)
        rows = conn.execute(
            f"SELECT seq, target, reason, priority, source, payload, created "
            f"FROM wake_requests "
            f"WHERE seq > ? AND target IN ({placeholders}) "
            f"ORDER BY seq ASC LIMIT ?",
            (after_seq, *target_list, limit),
        ).fetchall()
        result = []
        for row in rows:
            payload = row["payload"] or "{}"
            try:
                parsed_payload = json.loads(payload)
            except json.JSONDecodeError:
                parsed_payload = {}
            result.append({
                "seq": row["seq"],
                "target": row["target"],
                "reason": row["reason"],
                "priority": row["priority"],
                "source": row["source"],
                "payload": parsed_payload,
                "created": row["created"],
            })
        return result
    finally:
        conn.close()


def channel_ack_wakes(
    up_to_seq: int,
    targets: str | list[str] | None = None,
    project_dir: Path | None = None,
) -> int:
    """Delete wake requests up to and including a sequence number.

    When *targets* is provided, only rows for those wake targets are
    deleted. This prevents one consumer from deleting another target's
    unread wakes when they share the same transport DB.
    """
    conn = _open_db(project_dir)
    try:
        if targets is None:
            cursor = conn.execute(
                "DELETE FROM wake_requests WHERE seq <= ?",
                (up_to_seq,),
            )
        else:
            if isinstance(targets, str):
                target_list = [targets]
            else:
                target_list = [t for t in targets if t]
            if not target_list:
                return 0
            placeholders = ",".join("?" for _ in target_list)
            cursor = conn.execute(
                f"DELETE FROM wake_requests "
                f"WHERE seq <= ? AND target IN ({placeholders})",
                (up_to_seq, *target_list),
            )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def channel_wake_targets(
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> list[str]:
    """Return wake targets an agent process should watch."""
    aid = agent_name or _agent_id(project_dir)
    conn = _open_db(project_dir)
    try:
        channels = [
            row["channel"]
            for row in conn.execute(
                "SELECT channel FROM memberships WHERE agent_id = ? ORDER BY channel",
                (aid,),
            ).fetchall()
        ]
    finally:
        conn.close()
    return [f"agent:{aid}", *(f"channel:{channel}" for channel in channels)]


def _copy_attachments(
    message_id: str,
    attachment_paths: list[str],
    project_dir: Path | None = None,
) -> list[str]:
    """Copy attachments into the channel store and return relative paths."""
    target_dir = _attachments_dir(project_dir) / message_id
    # Attachments are a channel-owned write surface. A scrub or audit that
    # enumerates only *.jsonl misses this tree by construction, so it is
    # guarded explicitly rather than inheriting the JSONL path's coverage.
    _guard_store_path("copy_attachments", target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    stored: list[str] = []
    for raw_path in attachment_paths:
        source = Path(raw_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Attachment not found: {source}")

        candidate = target_dir / source.name
        if candidate.exists():
            stem = source.stem
            suffix = source.suffix
            idx = 2
            while True:
                candidate = target_dir / f"{stem}-{idx}{suffix}"
                if not candidate.exists():
                    break
                idx += 1

        shutil.copy2(source, candidate)
        stored.append(candidate.relative_to(_channels_dir(project_dir)).as_posix())

    return stored


def _read_messages(
    path: Path,
    since: str | None = None,
) -> list[ChannelMessage]:
    """Read all messages from a channel JSONL file, optionally filtered by timestamp."""
    messages: list[ChannelMessage] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = ChannelMessage.from_dict(json.loads(line))
                    if since and msg.timestamp <= since:
                        continue
                    messages.append(msg)
                except (json.JSONDecodeError, TypeError):
                    continue
    except OSError:
        pass
    return messages


def _estimate_token_count(text: str) -> int:
    """Approximate token count using the project's standard 4 chars/token heuristic."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _read_last_timestamp(path: Path) -> str | None:
    """Read the timestamp of the last message in a JSONL file.

    Seeks to near the end of the file and reads the last non-empty line,
    avoiding a full O(n) scan.
    """
    try:
        with open(path, "rb") as f:
            # Seek to near the end — 4KB is plenty for one JSONL line
            try:
                f.seek(-4096, 2)
            except OSError:
                f.seek(0)
            tail = f.read().decode("utf-8", errors="replace")
        # Find last non-empty line
        for line in reversed(tail.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line).get("timestamp")
            except (json.JSONDecodeError, TypeError):
                continue
    except OSError:
        pass
    return None


def _count_messages_since(path: Path, since: str) -> int:
    """Count JSONL lines with timestamp > since. Used for unread counts."""
    count = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("timestamp", "") > since:
                        count += 1
                except (json.JSONDecodeError, TypeError):
                    continue
    except OSError:
        pass
    return count


# ---------------------------------------------------------------------------
# Core channel operations
# ---------------------------------------------------------------------------

def channel_join(
    channel: str = "dev",
    agent_name: str | None = None,
    project_dir: Path | None = None,
    role: str = "agent",
    display_name: str | None = None,
) -> str:
    """Join a channel. Registers agent identity on first join.

    Args:
        role: "human" (set by session-start hook), "agent" (default), or "system".
        display_name: If provided, sets display name before joining so the
            join event shows the readable name instead of the agent hash.
            Also derives a distinct agent_id so agents sharing a griptree
            with a human session get their own presence row (recall#590).
    """
    aid = agent_name or _agent_id(project_dir, name=display_name)
    griptree = _resolve_griptree(project_dir)
    display = display_name or _resolve_display_name(project_dir)
    now = _now_iso()

    conn = _open_db(project_dir)
    try:
        cursor_init = _seed_cursor_for_join(
            conn,
            agent_id=aid,
            channel=channel,
            display_name=display,
            griptree=griptree,
            project_dir=project_dir,
        )

        if display_name is not None:
            conflict = _find_display_name_conflict(conn, display, aid)
            if conflict:
                existing = conflict["display_name"] or conflict["agent_id"]
                return (
                    f"Display name '{display}' is already in use by {existing} "
                    f"({conflict['agent_id']}). Choose a different name."
                )

        # Upsert presence with identity.
        # Role escalation: never downgrade human -> agent. A human session
        # that shares an agent_id with an agent (same griptree) keeps the
        # human role and display name. Fixes recall#546.
        #
        # Collision split (recall#665): when an agent resolves to a human's
        # agent_id, derive a distinct a_* identity for the agent instead of
        # absorbing it into the human's presence row. Cache the split id so
        # subsequent calls (channel_post) also use it.
        #
        # Registered agent override (recall#665): when the agent explicitly
        # owns the id via SYNAPT_AGENT_ID, allow role correction (the
        # session-start hook may have incorrectly set role="human").
        existing = conn.execute(
            "SELECT role, display_name, status FROM presence WHERE agent_id = ?",
            (aid,),
        ).fetchone()
        if existing and existing["role"] == "human" and role != "human":
            registered = os.environ.get("SYNAPT_AGENT_ID")
            if registered and registered == aid:
                # Registered agent owns this id; correct the role.
                conn.execute(
                    "INSERT INTO presence (agent_id, griptree, display_name, role, status, last_seen, joined_at) "
                    "VALUES (?, ?, ?, ?, 'online', ?, ?) "
                    "ON CONFLICT(agent_id) DO UPDATE SET status='online', last_seen=?, "
                    "griptree=?, display_name=?, role=?",
                    (aid, griptree, display, role, now, now, now, griptree, display, role),
                )
            else:
                # Collision: agent resolved to a human's s_* hash.
                # Derive a distinct identity and cache it.
                gt = _resolve_griptree(project_dir)
                seed = f"agent_split:{gt}:{aid}"
                aid = "a_" + hashlib.sha256(seed.encode()).hexdigest()[:8]
                key = str(project_data_dir(project_dir))
                _AGENT_ID_CACHE[key] = aid
                conn.execute(
                    "INSERT INTO presence (agent_id, griptree, display_name, role, status, last_seen, joined_at) "
                    "VALUES (?, ?, ?, ?, 'online', ?, ?) "
                    "ON CONFLICT(agent_id) DO UPDATE SET status='online', last_seen=?, "
                    "griptree=?, display_name=?, role=?",
                    (aid, griptree, display, role, now, now, now, griptree, display, role),
                )
        else:
            conn.execute(
                "INSERT INTO presence (agent_id, griptree, display_name, role, status, last_seen, joined_at) "
                "VALUES (?, ?, ?, ?, 'online', ?, ?) "
                "ON CONFLICT(agent_id) DO UPDATE SET status='online', last_seen=?, "
                "griptree=?, display_name=?, role=?",
                (aid, griptree, display, role, now, now, now, griptree, display, role),
            )

        # Add membership
        conn.execute(
            "INSERT OR IGNORE INTO memberships (agent_id, channel, joined_at) "
            "VALUES (?, ?, ?)",
            (aid, channel, now),
        )

        # Initialize unread flag (clean on join)
        conn.execute(
            "INSERT OR IGNORE INTO unread_flags (agent_id, channel, dirty) "
            "VALUES (?, ?, 0)",
            (aid, channel),
        )

        # Preserve prior read position for restarted sessions that inherit a
        # readable identity; otherwise start at the current tail for truly
        # first-time joins.
        conn.execute(
            "INSERT OR IGNORE INTO cursors (agent_id, channel, last_read_at) "
            "VALUES (?, ?, ?)",
            (aid, channel, cursor_init),
        )

        # Track whether this is a genuinely new join (for log event below).
        # An agent is "already joined" if they have presence (online) AND membership.
        has_membership = conn.execute(
            "SELECT 1 FROM memberships WHERE agent_id = ? AND channel = ?",
            (aid, channel),
        ).fetchone()
        is_new_join = not (existing and existing["status"] == "online" and has_membership)

        conn.commit()
    finally:
        conn.close()

    # Only append join event if this is a new join, not a reconnect.
    # Prevents duplicate "X joined #dev" spam on MCP restart. Fixes #546.
    if not is_new_join:
        display = display_name or _resolve_display_name(project_dir)
        return f"Reconnected to #{channel} as {display} ({aid})"

    # Append join event to channel log
    wt = _resolve_griptree(project_dir)
    msg = ChannelMessage(
        timestamp=now,
        from_agent=aid,
        from_display=display,
        channel=channel,
        type="join",
        body=f"{display} joined #{channel}",
        worktree=wt,
    )
    _append_message(msg, project_dir)

    result = f"Joined #{channel} as {display} ({aid})"

    # Surface recent @mentions with full content so agents don't miss
    # assignments (#453).  Including the actual message text means the
    # agent gets the directive regardless of cursor state.
    try:
        lookback = (datetime.now(timezone.utc) - timedelta(
            minutes=_JOIN_MENTION_LOOKBACK_MINUTES,
        )).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        conn = _open_db(project_dir)
        try:
            rows = conn.execute(
                "SELECT m.message_id, m.channel, m.timestamp "
                "FROM mentions m "
                "WHERE m.mentioned IN (?, ?) AND m.channel = ? "
                "AND m.timestamp > ? "
                "ORDER BY m.timestamp DESC LIMIT 5",
                (aid, display, channel, lookback),
            ).fetchall()
        finally:
            conn.close()
        if rows:
            mention_lines = []
            msg_ids = {r["message_id"] for r in rows}
            for m in _read_messages(_channel_path(channel, project_dir)):
                if m.id in msg_ids:
                    ts = m.timestamp[:16]
                    sender = m.from_display or m.from_agent
                    mention_lines.append(f"  {ts}  {sender}: {m.body}")
            if mention_lines:
                result += f"\n\n[channel] {len(mention_lines)} @mention(s):\n"
                result += "\n".join(mention_lines)
    except Exception as exc:
        import logging
        logging.getLogger("synapt.recall.channel").debug(
            "Failed to check recent mentions on join: %s", exc,
        )

    return result


def channel_leave(
    channel: str = "dev",
    agent_name: str | None = None,
    project_dir: Path | None = None,
    reason: str = "",
    display_name: str | None = None,
) -> str:
    """Leave a channel."""
    aid = agent_name or _agent_id(project_dir, name=display_name)
    now = _now_iso()

    conn = _open_db(project_dir)
    try:
        _reap_stale_agents(conn, project_dir)
        conn.execute(
            "DELETE FROM memberships WHERE agent_id = ? AND channel = ?",
            (aid, channel),
        )
        conn.execute(
            "DELETE FROM claims WHERE claimed_by = ? AND channel = ?",
            (aid, channel),
        )
        conn.execute(
            "UPDATE presence SET last_seen = ? WHERE agent_id = ?",
            (now, aid),
        )
        remaining = conn.execute(
            "SELECT COUNT(*) FROM memberships WHERE agent_id = ?",
            (aid,),
        ).fetchone()[0]
        if remaining == 0:
            conn.execute("DELETE FROM presence WHERE agent_id = ?", (aid,))
            conn.execute("DELETE FROM claims WHERE claimed_by = ?", (aid,))
        conn.execute(
            "DELETE FROM cursors WHERE agent_id = ? AND channel = ?",
            (aid, channel),
        )
        conn.commit()
    finally:
        conn.close()

    display = _resolve_display_name(project_dir)
    wt = _resolve_griptree(project_dir)
    body = reason if reason else f"{display} left #{channel}"
    msg = ChannelMessage(
        timestamp=now, from_agent=aid, from_display=display, channel=channel,
        type="leave", body=body, worktree=wt,
    )
    _append_message(msg, project_dir)
    return f"Left #{channel}"


# ---------------------------------------------------------------------------
# DM (direct message) channels -- private 1:1 agent-to-agent messaging
# ---------------------------------------------------------------------------


def resolve_dm_channel(agent_a: str, agent_b: str) -> str:
    """Return the canonical DM channel name for two agents.

    The name is always ``dm:{sorted_first}:{sorted_second}`` so both
    directions resolve to the same channel.  Self-DMs are rejected.
    Agent names must be non-empty and must not contain colons (which
    would break the channel name parsing).
    """
    if not agent_a or not agent_b:
        raise ValueError("Agent names must be non-empty")
    if ":" in agent_a or ":" in agent_b:
        raise ValueError(
            f"Agent names must not contain colons: {agent_a!r}, {agent_b!r}"
        )
    if agent_a == agent_b:
        raise ValueError(f"An agent cannot DM itself: {agent_a!r}")
    first, second = sorted([agent_a, agent_b])
    return f"dm:{first}:{second}"


def resolve_dm_channel_from_shorthand(channel: str, sender: str) -> str:
    """Resolve a shorthand like ``dm:atlas`` into the canonical DM channel.

    The shorthand encodes only the *recipient*; the sender is supplied
    separately so the canonical sorted-pair name can be computed.
    """
    if not channel.startswith("dm:"):
        raise ValueError(f"Not a DM shorthand: {channel!r}")
    parts = channel.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"DM shorthand must be 'dm:<recipient>', got: {channel!r}"
        )
    recipient = parts[1]
    return resolve_dm_channel(sender, recipient)


def is_dm_channel(channel: str) -> bool:
    """Return True if *channel* is a DM channel (``dm:<a>:<b>``)."""
    parts = channel.split(":")
    return len(parts) == 3 and parts[0] == "dm"


def dm_participants(channel: str) -> tuple[str, str]:
    """Extract the two participant names from a DM channel name.

    Raises ValueError if *channel* is not a valid DM channel.
    """
    if not is_dm_channel(channel):
        raise ValueError(f"Not a DM channel: {channel!r}")
    parts = channel.split(":")
    return (parts[1], parts[2])


def is_dm_participant(channel: str, agent_id: str) -> bool:
    """Return True if *agent_id* is one of the two DM participants."""
    if not is_dm_channel(channel):
        return False
    a, b = dm_participants(channel)
    return agent_id in (a, b)


def list_dm_channels(
    agent_id: str, project_dir: Path | None = None,
) -> list[str]:
    """Return all DM channel names that *agent_id* participates in."""
    ch_dir = _channels_dir(project_dir)
    if not ch_dir.exists():
        return []
    result = []
    for p in ch_dir.glob("*.jsonl"):
        name = _filename_to_channel(p.stem)
        if is_dm_channel(name) and is_dm_participant(name, agent_id):
            result.append(name)
    return sorted(result)


def channel_post(
    channel: str,
    message: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
    pin: bool = False,
    attachment_paths: list[str] | None = None,
    channels_dir: Path | None = None,
    display_name: str | None = None,
    msg_type: str = "message",
) -> str:
    """Post a message to a channel. If pin=True, also pin the message.

    ``channels_dir`` overrides the JSONL write location so messages can be
    posted to cross-project channels (e.g. from the dashboard switching views).
    DB operations (presence, pins) always use the local gripspace.

    ``display_name`` overrides the presence DB lookup so callers (e.g. MCP
    tool handler) can pass the agent's declared name directly.  Also
    derives a distinct agent_id per name (recall#590).
    """
    aid = agent_name or _agent_id(project_dir, name=display_name)
    now = _now_iso()
    display = display_name or _resolve_display_name_for(aid, project_dir)

    wt = _resolve_griptree(project_dir)
    msg = ChannelMessage(
        timestamp=now, from_agent=aid, from_display=display, channel=channel,
        type=msg_type or "message", body=message, worktree=wt,
    )
    if attachment_paths:
        msg.id = _generate_msg_id(msg.timestamp, msg.from_agent, msg.body)
        msg.attachments = _copy_attachments(msg.id, attachment_paths, project_dir)
    _append_message(msg, project_dir, channels_dir=channels_dir)

    conn = _open_db(project_dir)
    try:
        conn.execute(
            "UPDATE presence SET last_seen = ?, status = 'online' WHERE agent_id = ?",
            (now, aid),
        )
        if pin:
            conn.execute(
                "INSERT INTO pins (channel, body, message_id, pinned_by, pinned_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (channel, message, msg.id, aid, now),
            )
        conn.commit()
    finally:
        conn.close()
    attachment_suffix = ""
    if msg.attachments:
        attachment_suffix = f" [attachments: {', '.join(msg.attachments)}]"
    result = f"[#{channel}] {display}: {message}{attachment_suffix}".rstrip()
    if pin:
        result += " (pinned)"
    return result


def channel_read(
    channel: str = "dev",
    limit: int = 20,
    since: str | None = None,
    agent_name: str | None = None,
    project_dir: Path | None = None,
    show_pins: bool = True,
    detail: str = "medium",
    msg_type: str | None = None,
) -> str:
    """Read recent messages from a channel.

    Filters muted agents, highlights directives targeted at the reader.
    Uses a single DB connection for all operations.

    Detail levels control output verbosity and override show_pins:
        max    -- pins, full messages, all metadata (IDs, claims, attachments)
        high   -- pins, full messages, message IDs only
        medium -- full messages, IDs, claims, attachments; pins follow show_pins
        low    -- no pins, truncated messages (200 chars), with refs for truncated messages
        min    -- no pins, one-line per message, skip join/leave noise
    """
    _detail = detail.lower()
    if _detail in ("max", "high"):
        show_pins = True
    elif _detail in ("low", "min"):
        show_pins = False
    _show_ids = _detail in ("max", "high", "medium")
    _show_claims = _detail in ("max", "medium")
    _show_attachments = _detail in ("max", "medium")
    _truncate = 200 if _detail == "low" else (80 if _detail == "min" else 0)
    _one_line = _detail == "min"
    aid = agent_name or _agent_id(project_dir)
    now = _now_iso()

    conn = _open_db(project_dir)
    try:
        # Reap stale agents
        _reap_stale_agents(conn, project_dir)

        # Get muted agents for this channel
        muted = {
            r["agent_id"]
            for r in conn.execute(
                "SELECT agent_id FROM mutes WHERE channel = ? AND muted_by = ?",
                (channel, aid),
            ).fetchall()
        }

        # Get pins (skip query when show_pins=False to save overhead)
        pins = []
        if show_pins:
            pins = conn.execute(
                "SELECT body, message_id, pinned_by, pinned_at FROM pins WHERE channel = ? "
                "ORDER BY pinned_at",
                (channel,),
            ).fetchall()

        # Resolve display names and roles for rendering
        display_map = {}
        role_map: dict[str, str] = {}
        for r in conn.execute("SELECT agent_id, display_name, griptree, role FROM presence").fetchall():
            display_map[r["agent_id"]] = r["display_name"] or r["griptree"] or r["agent_id"]
            role_map[r["agent_id"]] = r["role"]

        # Load claims for this channel
        claim_map = {}
        for r in conn.execute(
            "SELECT message_id, claimed_by FROM claims WHERE channel = ?",
            (channel,),
        ).fetchall():
            claim_map[r["message_id"]] = r["claimed_by"]

        # Load status board entries
        board_rows = conn.execute(
            "SELECT agent_id, body, updated_at FROM status_board "
            "WHERE channel = ? ORDER BY updated_at DESC",
            (channel,),
        ).fetchall()

        # Update read cursor and clear dirty flag
        conn.execute(
            "INSERT INTO cursors (agent_id, channel, last_read_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(agent_id, channel) DO UPDATE SET last_read_at = ?",
            (aid, channel, now, now),
        )
        conn.execute(
            "UPDATE unread_flags SET dirty = 0, last_cleared_at = ? "
            "WHERE agent_id = ? AND channel = ?",
            (now, aid, channel),
        )
        conn.commit()
    finally:
        conn.close()

    path = _channel_path(channel, project_dir)
    if not path.exists():
        return f"Channel #{channel} has no messages yet."

    messages = _read_messages(path, since=since)
    if not messages:
        if since:
            return f"No messages in #{channel} since {since}."
        return f"Channel #{channel} has no messages yet."

    # Filter by message type
    if msg_type:
        messages = [m for m in messages if m.type == msg_type]

    # Filter muted agents
    if muted:
        messages = [m for m in messages if m.from_agent not in muted]

    # Take last N messages
    messages = messages[-limit:]

    # Zero-token response for empty polls (#441)
    if not messages and since:
        return ""

    lines = []

    # Pins at top
    if pins:
        lines.append(f"## Pinned in #{channel}")
        for pin in pins:
            ts = pin["pinned_at"][:16]
            by = display_map.get(pin["pinned_by"], pin["pinned_by"])
            mid = f" [{pin['message_id']}]" if pin["message_id"] and _show_ids else ""
            lines.append(f"  [pin]{mid} {ts}  {by}: {pin['body']}")
        lines.append("")

    # Status board after pins
    if not _one_line and board_rows:
        lines.append(f"## Status Board — #{channel}")
        for r in board_rows:
            bd = display_map.get(r["agent_id"], r["agent_id"])
            bts = r["updated_at"][:16].replace("T", " ")
            lines.append(f"  {bd} ({bts}): {r['body']}")
        lines.append("")

    lines.append(f"## #{channel} ({len(messages)} messages)")
    self_names = {aid.casefold()}
    own_display = display_map.get(aid)
    if own_display:
        self_names.add(own_display.casefold())
    truncated_messages: list[tuple[str, int]] = []
    for msg in messages:
        ts = msg.timestamp[:16]
        display = msg.from_display or display_map.get(msg.from_agent, msg.from_agent)
        mid = f" [{msg.id}]" if msg.id and _show_ids else ""
        inline_mid = mid
        # Role marker — human messages get a distinct prefix
        is_human = role_map.get(msg.from_agent) == "human"
        role_tag = " [human]" if is_human else ""
        # Claim annotation
        claim_tag = ""
        if _show_claims:
            claimer_id = claim_map.get(msg.id)
            if claimer_id:
                claimer_name = display_map.get(claimer_id, claimer_id)
                claim_tag = f" [CLAIMED by {claimer_name}]"
        attachment_tag = ""
        if _show_attachments and msg.attachments:
            attachment_tag = f" [attachments: {', '.join(msg.attachments)}]"
        body = msg.body
        mentions_self = any(
            mention.casefold() in self_names
            for mention in _extract_mentions(body)
        )
        actionable = (msg.type == "directive" and msg.to in (aid, "*")) or mentions_self
        truncation_tag = ""
        if _truncate and not actionable and len(body) > _truncate:
            omitted_tokens = _estimate_token_count(body[_truncate:])
            truncated_messages.append((msg.id or "(no-id)", omitted_tokens))
            body = body[:_truncate].rstrip() + "..."
            if msg.id and not _show_ids:
                inline_mid = f" [{msg.id}]"
            truncation_tag = f" [truncated ~{omitted_tokens} tok omitted]"
        if _one_line:
            body = body.replace("\n", " ").strip()
        # Worktree tag at max detail (recall#443)
        wt_tag = ""
        if _detail == "max" and msg.worktree:
            wt_tag = f" @{msg.worktree}"
        if msg.type in ("join", "leave", "claim", "unclaim"):
            if _one_line:
                continue
            lines.append(f"  {ts}{inline_mid}  -- {body}{truncation_tag}")
        elif msg.type == "directive":
            target = f" @{msg.to}" if msg.to else ""
            prefix = "[DIRECTIVE]" if msg.to in (aid, "*") else "[directive]"
            lines.append(
                f"  {ts}{inline_mid}  {prefix}{target} {display}{role_tag}{wt_tag}: "
                f"{body}{truncation_tag}{attachment_tag}{claim_tag}"
            )
        else:
            lines.append(
                f"  {ts}{inline_mid}  {display}{role_tag}{wt_tag}: "
                f"{body}{truncation_tag}{attachment_tag}{claim_tag}"
            )

    if truncated_messages:
        total_omitted = sum(tokens for _, tokens in truncated_messages)
        refs = ", ".join(f"{message_id} (~{tokens} tok)" for message_id, tokens in truncated_messages)
        lines.insert(
            len(lines) if not lines else lines.index(f"## #{channel} ({len(messages)} messages)") + 1,
            (
                f"  truncated {len(truncated_messages)} message(s), "
                f"~{total_omitted} tok omitted total: {refs}. "
                "Use action='read_message' with message=<id> to inspect the full body."
            ),
        )

    return "\n".join(lines)


def channel_read_message(
    message_id: str,
    channel: str = "dev",
    project_dir: Path | None = None,
) -> str:
    """Read a specific channel message by id."""
    path = _channel_path(channel, project_dir)
    if not path.exists():
        return f"Channel #{channel} has no messages yet."

    for msg in _read_messages(path):
        if msg.id != message_id:
            continue

        display = msg.from_display or msg.from_agent
        lines = [f"## #{channel} [{msg.id}]", f"Timestamp: {msg.timestamp}", f"Type: {msg.type}"]
        if display and display != msg.from_agent:
            lines.append(f"From: {display} ({msg.from_agent})")
        else:
            lines.append(f"From: {display}")
        if msg.to:
            lines.append(f"To: {msg.to}")
        if msg.attachments:
            lines.append(f"Attachments: {', '.join(msg.attachments)}")
        lines.append("")
        lines.append(msg.body or "(empty body)")
        return "\n".join(lines)

    return f"Message {message_id} not found in #{channel}."


def channel_who(project_dir: Path | None = None) -> str:
    """Show which agents are currently online and in which channels.

    Displays all three identity layers: display_name, griptree, agent_id.
    """
    conn = _open_db(project_dir)
    try:
        _reap_stale_agents(conn, project_dir)

        agents = conn.execute(
            "SELECT agent_id, griptree, display_name, role, status, last_seen FROM presence"
        ).fetchall()

        if not agents:
            return "No agents online."

        # Deduplicate by (griptree, display_name): when multiple agents share
        # the same griptree AND display name (dead sessions from the same
        # worktree that haven't been reaped yet), only show the most recently
        # seen one to avoid noisy /who output. Agents with distinct display
        # names (e.g., different worktrees) always appear separately.
        best_by_identity: dict[tuple[str, str], sqlite3.Row] = {}
        for row in agents:
            gt = row["griptree"] or row["agent_id"]
            dn = row["display_name"] or ""
            key = (gt, dn)
            existing = best_by_identity.get(key)
            if existing is None or row["last_seen"] > existing["last_seen"]:
                best_by_identity[key] = row
        deduped_agents = list(best_by_identity.values())

        lines = ["## Agents"]
        for row in sorted(deduped_agents, key=lambda r: r["display_name"] or r["griptree"] or r["agent_id"]):
            aid = row["agent_id"]
            last_seen = row["last_seen"]
            status = _agent_status(last_seen)

            if status == "offline":
                continue

            channels = [
                r["channel"]
                for r in conn.execute(
                    "SELECT channel FROM memberships WHERE agent_id = ? ORDER BY channel",
                    (aid,),
                ).fetchall()
            ]
            channels_str = ", ".join(f"#{c}" for c in channels) if channels else "(no channels)"

            # Display name with fallback chain
            display = row["display_name"] or row["griptree"] or aid
            identity = f" ({row['griptree']}, {aid})" if row["display_name"] else f" ({aid})"

            if status == "online":
                status_label = "online"
            else:
                delta = datetime.now(timezone.utc) - _parse_iso(last_seen)
                mins = int(delta.total_seconds() / 60)
                status_label = f"{status} ({mins}m)"

            try:
                agent_role = row["role"]
            except (IndexError, KeyError):
                agent_role = "agent"
            role_label = f" [{agent_role}]" if agent_role != "agent" else ""
            lines.append(f"  {display}{identity}{role_label}  [{status_label}]  {channels_str}")

        if len(lines) == 1:
            return "No agents online."
        return "\n".join(lines)
    finally:
        conn.close()


def channel_heartbeat(
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Update last_seen for an agent. Lightweight -- one UPDATE statement."""
    aid = agent_name or _agent_id(project_dir)
    now = _now_iso()

    conn = _open_db(project_dir)
    try:
        result = conn.execute(
            "UPDATE presence SET last_seen = ?, status = 'online' WHERE agent_id = ?",
            (now, aid),
        )
        conn.commit()
        if result.rowcount == 0:
            return f"Agent {aid} not in any channel (heartbeat skipped)."
        return f"Heartbeat: {aid} at {now}"
    finally:
        conn.close()


def channel_unread(
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> dict[str, int]:
    """Return unread message counts per channel for the given agent."""
    aid = agent_name or _agent_id(project_dir)

    conn = _open_db(project_dir)
    try:
        memberships = conn.execute(
            "SELECT channel FROM memberships WHERE agent_id = ?",
            (aid,),
        ).fetchall()
        if not memberships:
            return {}

        result = {}
        for row in memberships:
            ch = row["channel"]
            cursor_row = conn.execute(
                "SELECT last_read_at FROM cursors WHERE agent_id = ? AND channel = ?",
                (aid, ch),
            ).fetchone()
            last_read = cursor_row["last_read_at"] if cursor_row else "1970-01-01T00:00:00Z"
            path = _channel_path(ch, project_dir)
            result[ch] = _count_messages_since(path, last_read) if path.exists() else 0
        return result
    finally:
        conn.close()


def channel_unread_read(
    agent_name: str | None = None,
    project_dir: Path | None = None,
    limit: int = 20,
    show_pins: bool = False,
    detail: str = "medium",
) -> str:
    """Read unread messages across all joined channels and advance cursors.

    Keeps ``channel_unread()`` as the lightweight count primitive used by
    chat UI and directive checks, while giving the MCP ``unread`` action a
    single-call catchup path that includes message content.

    Default show_pins=False — unread is a polling action, pins are static
    and don't need to be re-sent every tick.
    """
    aid = agent_name or _agent_id(project_dir)

    conn = _open_db(project_dir)
    try:
        memberships = conn.execute(
            "SELECT channel FROM memberships WHERE agent_id = ?",
            (aid,),
        ).fetchall()
        if not memberships:
            return "No channel memberships -- join a channel first."

        unread_channels: list[tuple[str, str, int]] = []
        for row in memberships:
            ch = row["channel"]
            cursor_row = conn.execute(
                "SELECT last_read_at FROM cursors WHERE agent_id = ? AND channel = ?",
                (aid, ch),
            ).fetchone()
            last_read = cursor_row["last_read_at"] if cursor_row else "1970-01-01T00:00:00Z"
            path = _channel_path(ch, project_dir)
            count = _count_messages_since(path, last_read) if path.exists() else 0
            if count > 0:
                unread_channels.append((ch, last_read, count))
    finally:
        conn.close()

    if not unread_channels:
        return ""

    sections = []
    for ch, last_read, count in sorted(unread_channels):
        rendered = channel_read(
            channel=ch,
            limit=limit,
            since=last_read,
            agent_name=aid,
            project_dir=project_dir,
            show_pins=show_pins,
            detail=detail,
        )
        sections.append(rendered)

    return "\n\n".join(sections)


def channel_pin(
    channel: str,
    message_id: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Pin a message by its ID. Looks up the message in the JSONL log."""
    aid = agent_name or _agent_id(project_dir)
    now = _now_iso()

    # Find the message in the JSONL
    path = _channel_path(channel, project_dir)
    body = ""
    if path.exists():
        for msg in _read_messages(path):
            if msg.id == message_id:
                body = msg.body
                break
    if not body:
        return f"Message {message_id} not found in #{channel}."

    conn = _open_db(project_dir)
    try:
        conn.execute(
            "INSERT INTO pins (channel, body, message_id, pinned_by, pinned_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (channel, body, message_id, aid, now),
        )
        conn.commit()
    finally:
        conn.close()

    _set_dirty_flags(channel, aid, project_dir)
    return f"Pinned [{message_id}] in #{channel}: {body}"


def channel_unpin(
    channel: str,
    message_id: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Unpin a message by its message_id."""
    conn = _open_db(project_dir)
    try:
        row = conn.execute(
            "SELECT id, body FROM pins WHERE channel = ? AND message_id = ?",
            (channel, message_id),
        ).fetchone()
        if not row:
            return f"No pin found for [{message_id}] in #{channel}."
        conn.execute("DELETE FROM pins WHERE id = ?", (row["id"],))
        conn.commit()
    finally:
        conn.close()

    body_preview = row["body"][:80] + "..." if len(row["body"]) > 80 else row["body"]
    return f"Unpinned [{message_id}] from #{channel}: {body_preview}"


# ---------------------------------------------------------------------------
# New operations: directive, mute, kick, broadcast, list
# ---------------------------------------------------------------------------


def channel_directive(
    channel: str,
    message: str,
    to: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
    remind: bool = False,
    display_name: str | None = None,
) -> str:
    """Post a directive message targeted at a specific agent."""
    aid = agent_name or _agent_id(project_dir, name=display_name)
    now = _now_iso()
    display = display_name or _resolve_display_name(project_dir)

    msg = ChannelMessage(
        timestamp=now, from_agent=aid, from_display=display, channel=channel,
        type="directive", body=message, to=to,
    )
    _append_message(msg, project_dir)

    # Optionally bridge to reminders
    if remind:
        try:
            from synapt.recall.reminders import add_reminder
            add_reminder(f"[directive from {aid}] {message}")
        except Exception:
            pass
    return f"[#{channel}] {display} → @{to}: {message}"


def channel_mute(
    target: str,
    channel: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Mute an agent in a channel. Their messages won't appear in your reads.

    ``target`` can be an agent_id (s_*), display_name, or griptree name.
    """
    aid = agent_name or _agent_id(project_dir)
    now = _now_iso()

    conn = _open_db(project_dir)
    try:
        target_id = _resolve_target_id(target, conn)
        conn.execute(
            "INSERT OR REPLACE INTO mutes (agent_id, channel, muted_by, muted_at) "
            "VALUES (?, ?, ?, ?)",
            (target_id, channel, aid, now),
        )
        conn.commit()
    finally:
        conn.close()
    return f"Muted {target_id} in #{channel}."


def channel_unmute(
    target: str,
    channel: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Unmute an agent in a channel.

    ``target`` can be an agent_id (s_*), display_name, or griptree name.
    """
    aid = agent_name or _agent_id(project_dir)

    conn = _open_db(project_dir)
    try:
        target_id = _resolve_target_id(target, conn)
        conn.execute(
            "DELETE FROM mutes WHERE agent_id = ? AND channel = ? AND muted_by = ?",
            (target_id, channel, aid),
        )
        conn.commit()
    finally:
        conn.close()
    return f"Unmuted {target_id} in #{channel}."


def channel_kick(
    target: str,
    channel: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Remove an agent from a channel (admin action).

    ``target`` can be an agent_id (s_*), display_name, or griptree name.
    """
    aid = agent_name or _agent_id(project_dir)
    now = _now_iso()

    conn = _open_db(project_dir)
    try:
        target_id = _resolve_target_id(target, conn)
        conn.execute(
            "DELETE FROM memberships WHERE agent_id = ? AND channel = ?",
            (target_id, channel),
        )
        conn.execute(
            "DELETE FROM cursors WHERE agent_id = ? AND channel = ?",
            (target_id, channel),
        )
        remaining = conn.execute(
            "SELECT COUNT(*) FROM memberships WHERE agent_id = ?",
            (target_id,),
        ).fetchone()[0]
        if remaining == 0:
            conn.execute("DELETE FROM presence WHERE agent_id = ?", (target_id,))
        conn.commit()
    finally:
        conn.close()

    # Post kick event
    display = _resolve_display_name(project_dir)
    msg = ChannelMessage(
        timestamp=now, from_agent=aid, from_display=display, channel=channel,
        type="leave", body=f"{target_id} kicked from #{channel} by {display}",
    )
    _append_message(msg, project_dir)
    return f"Kicked {target_id} from #{channel}."


def channel_broadcast(
    message: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
    display_name: str | None = None,
) -> str:
    """Post a message to ALL active channels."""
    channels = channel_list_channels(project_dir)
    if not channels:
        return "No channels to broadcast to."
    for ch in channels:
        channel_post(ch, message, agent_name=agent_name, project_dir=project_dir, display_name=display_name)
    return f"Broadcast to {len(channels)} channel(s): {', '.join(f'#{c}' for c in channels)}"


def channel_board(
    channel: str = "dev",
    message: str | None = None,
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Read or update the mutable status board for a channel.

    Each agent has one row on the board. Setting a message replaces
    the previous value (previous state archived to history).
    Reading returns all agents' current board entries.
    Pass an empty string to clear your entry.
    """
    aid = agent_name or _agent_id(project_dir)
    now = _now_iso()
    conn = _open_db(project_dir)
    try:
        if message is not None:
            # Archive previous state
            conn.execute(
                "INSERT INTO status_board_history (channel, agent_id, body, updated_at) "
                "SELECT channel, agent_id, body, updated_at "
                "FROM status_board WHERE channel = ? AND agent_id = ?",
                (channel, aid),
            )
            if message:
                conn.execute(
                    "INSERT OR REPLACE INTO status_board (channel, agent_id, body, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (channel, aid, message, now),
                )
            else:
                # Empty string = clear entry
                conn.execute(
                    "DELETE FROM status_board WHERE channel = ? AND agent_id = ?",
                    (channel, aid),
                )
            conn.commit()

            # Resolve display name inline
            row = conn.execute(
                "SELECT display_name FROM presence WHERE agent_id = ?", (aid,)
            ).fetchone()
            display = (row["display_name"] if row and row["display_name"] else aid)
            return f"Board updated for {display} in #{channel}."

        # Read mode
        rows = conn.execute(
            "SELECT agent_id, body, updated_at FROM status_board "
            "WHERE channel = ? ORDER BY updated_at DESC",
            (channel,),
        ).fetchall()
        if not rows:
            return f"No status board entries in #{channel}."

        # Build display name map
        display_map: dict[str, str] = {}
        for r in conn.execute("SELECT agent_id, display_name FROM presence").fetchall():
            display_map[r["agent_id"]] = r["display_name"] or r["agent_id"]

        lines = [f"## Status Board — #{channel}"]
        for r in rows:
            display = display_map.get(r["agent_id"], r["agent_id"])
            ts = r["updated_at"][:16].replace("T", " ")
            lines.append(f"  {display} ({ts}): {r['body']}")
        return "\n".join(lines)
    finally:
        conn.close()


def channel_rename(
    new_name: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Set or update the display name for this agent.

    Persists in the presence table so future messages and /who output
    use the new name. No env var or MCP restart required.
    """
    aid = agent_name or _agent_id(project_dir)
    now = _now_iso()

    conn = _open_db(project_dir)
    try:
        conflict = _find_display_name_conflict(conn, new_name, aid)
        if conflict:
            existing = conflict["display_name"] or conflict["agent_id"]
            return (
                f"Display name '{new_name}' is already in use by {existing} "
                f"({conflict['agent_id']}). Choose a different name."
            )

        row = conn.execute(
            "SELECT agent_id FROM presence WHERE agent_id = ?", (aid,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE presence SET display_name = ?, last_seen = ? WHERE agent_id = ?",
                (new_name, now, aid),
            )
        else:
            griptree = _resolve_griptree(project_dir)
            conn.execute(
                "INSERT INTO presence (agent_id, griptree, display_name, status, last_seen, joined_at) "
                "VALUES (?, ?, ?, 'online', ?, ?)",
                (aid, griptree, new_name, now, now),
            )
        conn.commit()
    finally:
        conn.close()
    return f"Display name set to '{new_name}'."


def channel_claim(
    message_id: str,
    channel: str = "dev",
    agent_name: str | None = None,
    project_dir: Path | None = None,
    _silent: bool = False,
) -> str:
    """Claim a message so other agents know you're handling it.

    Uses INSERT OR IGNORE for atomic first-writer-wins semantics.
    Returns success if you claimed it, or the existing claimant if
    someone else already did.

    If ``_silent`` is False (the default), a claim notification is
    automatically posted to the channel so other agents see it in
    the message stream without polling the claims table.
    """
    aid = agent_name or _agent_id(project_dir)
    now = _now_iso()

    conn = _open_db(project_dir)
    try:
        _reap_stale_agents(conn, project_dir)
        conn.execute(
            "INSERT OR IGNORE INTO claims (message_id, channel, claimed_by, claimed_at) "
            "VALUES (?, ?, ?, ?)",
            (message_id, channel, aid, now),
        )
        conn.commit()

        # Check who actually holds the claim
        row = conn.execute(
            "SELECT claimed_by FROM claims WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        if row and row["claimed_by"] == aid:
            display = _resolve_display_name_for(aid, project_dir)
            # Auto-post claim notification to channel
            if not _silent:
                claim_msg = ChannelMessage(
                    timestamp=now, from_agent=aid, from_display=display,
                    channel=channel, type="claim",
                    body=f"{display} claimed {message_id}",
                )
                _append_message(claim_msg, project_dir)
            return f"Claimed {message_id} — you ({display}) own this task."
        else:
            claimer = row["claimed_by"] if row else "unknown"
            display = _resolve_display_name_for(claimer, project_dir)
            return f"Already claimed by {display} ({claimer})."
    finally:
        conn.close()


def is_claimed(
    message_id: str,
    project_dir: Path | None = None,
) -> dict | None:
    """Check if a message is claimed. Returns claim dict or None."""
    conn = _open_db(project_dir)
    try:
        _reap_stale_agents(conn, project_dir)
        row = conn.execute(
            "SELECT claimed_by, claimed_at FROM claims WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        if row:
            return {"claimed_by": row["claimed_by"], "claimed_at": row["claimed_at"]}
        return None
    finally:
        conn.close()


def channel_unclaim(
    message_id: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Release a claim on a message. Only the original claimant can unclaim."""
    aid = agent_name or _agent_id(project_dir)

    conn = _open_db(project_dir)
    try:
        _reap_stale_agents(conn, project_dir)
        row = conn.execute(
            "SELECT claimed_by, channel FROM claims WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        if not row:
            return f"Message {message_id} is not claimed."
        if row["claimed_by"] != aid:
            display = _resolve_display_name_for(row["claimed_by"], project_dir)
            return f"Cannot unclaim — owned by {display} ({row['claimed_by']})."
        ch = row["channel"]
        conn.execute("DELETE FROM claims WHERE message_id = ?", (message_id,))
        conn.commit()
        display = _resolve_display_name_for(aid, project_dir)
        unclaim_msg = ChannelMessage(
            timestamp=_now_iso(), from_agent=aid, from_display=display,
            channel=ch, type="unclaim",
            body=f"{display} released claim on {message_id}",
        )
        _append_message(unclaim_msg, project_dir)
        return f"Released claim on {message_id}."
    finally:
        conn.close()


def channel_claim_intent(
    intent: str,
    channel: str = "dev",
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> tuple[bool, str]:
    """Announce and claim an intent before creating an issue/PR.

    Posts a ``[INTENT]`` message to the channel and claims it atomically.
    Other agents should check for existing intents before duplicating.

    Args:
        intent: Description of what you're about to create (e.g.,
                "filing issue for session-start perf fix").
        channel: Channel to post in.

    Returns:
        (True, message) if intent claimed successfully.
        (False, message) if someone else already claimed the same intent.
    """
    aid = agent_name or _agent_id(project_dir)
    now = _now_iso()

    # Check if anyone recently posted the same intent (dedup by content)
    path = _channel_path(channel, project_dir)
    if path.exists():
        recent = _read_messages(path)
        # Check last 20 messages for duplicate intents
        for msg in recent[-20:]:
            if msg.type == "intent" and msg.from_agent != aid:
                if intent.lower() in msg.body.lower() or msg.body.lower() in intent.lower():
                    claimer = _resolve_display_name_for(msg.from_agent, project_dir)
                    return (False, f"Already claimed by {claimer}: {msg.body}")

    # Post and claim
    display = _resolve_display_name_for(aid, project_dir)
    msg = ChannelMessage(
        timestamp=now, from_agent=aid, from_display=display, channel=channel,
        type="intent", body=intent,
    )
    _append_message(msg, project_dir)

    # Auto-claim the intent message (silent — the intent post is the notification)
    channel_claim(msg.id, channel, agent_name=aid, project_dir=project_dir, _silent=True)

    display = _resolve_display_name_for(aid, project_dir)
    return (True, f"[INTENT] {display}: {intent}")


def channel_list_channels(project_dir: Path | None = None) -> list[str]:
    """Return list of all public channel names (from JSONL files).

    DM channels (``dm:*``) are excluded from the global list.
    Use :func:`list_dm_channels` to discover an agent's DM conversations.
    """
    ch_dir = _channels_dir(project_dir)
    if not ch_dir.exists():
        return []
    return sorted(
        _filename_to_channel(p.stem) for p in ch_dir.glob("*.jsonl")
        if not is_dm_channel(_filename_to_channel(p.stem))
    )


# ---------------------------------------------------------------------------
# JSON-returning wrappers for dashboard / API consumers
# ---------------------------------------------------------------------------


def channel_agents_json(project_dir: Path | None = None) -> list[dict]:
    """Return agent presence data as a list of dicts.

    Each dict contains: agent_id, display_name, griptree, role, status,
    last_seen, channels. Offline agents are excluded. Deduped by identity.
    """
    conn = _open_db(project_dir)
    try:
        _reap_stale_agents(conn, project_dir)
        agents = conn.execute(
            "SELECT agent_id, griptree, display_name, role, status, last_seen FROM presence"
        ).fetchall()
        if not agents:
            return []

        # If a human has both a fallback identity (display_name == griptree/empty)
        # and a better named identity in the same griptree, collapse those into
        # the named identity so the dashboard does not show duplicate humans.
        preferred_human_name: dict[str, tuple[str, str]] = {}
        for row in agents:
            if (row["role"] or "agent") != "human":
                continue
            gt = row["griptree"] or row["agent_id"]
            dn = row["display_name"] or ""
            if not dn:
                continue
            if _normalize_display_name(dn) == _normalize_display_name(gt):
                continue
            existing = preferred_human_name.get(gt)
            if existing is None or row["last_seen"] > existing[1]:
                preferred_human_name[gt] = (dn, row["last_seen"])

        # Dedup by identity — keep most recently seen
        best: dict[tuple[str, str], sqlite3.Row] = {}
        for row in agents:
            gt = row["griptree"] or row["agent_id"]
            dn = row["display_name"] or ""
            role = row["role"] or "agent"
            if role == "human" and gt in preferred_human_name:
                key = (gt, preferred_human_name[gt][0])
            else:
                key = (gt, dn)
            existing = best.get(key)
            if existing is None or row["last_seen"] > existing["last_seen"]:
                best[key] = row

        result = []
        for row in sorted(best.values(), key=lambda r: r["display_name"] or r["griptree"] or r["agent_id"]):
            status = _agent_status(row["last_seen"])
            if status == "offline":
                continue
            channels = [
                r["channel"]
                for r in conn.execute(
                    "SELECT channel FROM memberships WHERE agent_id = ? ORDER BY channel",
                    (row["agent_id"],),
                ).fetchall()
            ]
            result.append({
                "agent_id": row["agent_id"],
                "display_name": row["display_name"] or "",
                "griptree": row["griptree"] or "",
                "role": row["role"] or "agent",
                "status": status,
                "last_seen": row["last_seen"],
                "channels": channels,
            })
        return result
    finally:
        conn.close()


def channel_messages_json(
    channel: str = "dev",
    limit: int = 50,
    since: str | None = None,
    project_dir: Path | None = None,
    channels_dir: Path | None = None,
    msg_type: str | None = None,
) -> list[dict]:
    """Return recent channel messages as a list of dicts.

    Each dict matches the ChannelMessage.to_dict() format with keys:
    timestamp, channel, type, body, from, from_display, id, to, attachments.

    ``channels_dir`` overrides path resolution for cross-project reads.
    ``msg_type`` filters to messages of a specific type (e.g. "status", "claim", "pr").
    """
    path = (channels_dir / f"{_channel_to_filename(channel)}.jsonl") if channels_dir else _channel_path(channel, project_dir)
    if not path.exists():
        return []
    messages = _read_messages(path, since=since)
    if msg_type:
        messages = [m for m in messages if m.type == msg_type]
    messages = messages[-limit:]
    conn = _open_db(project_dir)
    try:
        display_map = {
            row["agent_id"]: row["display_name"] or row["griptree"] or row["agent_id"]
            for row in conn.execute(
                "SELECT agent_id, display_name, griptree FROM presence"
            ).fetchall()
        }
    finally:
        conn.close()

    result = []
    for msg in messages:
        d = msg.to_dict()
        current_display = display_map.get(msg.from_agent, msg.from_agent)
        # If the persisted display is missing or is just a raw session id,
        # prefer the current resolved display name for dashboard/API clients.
        if not d.get("from_display") or str(d.get("from_display", "")).startswith("s_"):
            d["from_display"] = current_display
        result.append(d)
    return result


def channel_search(
    query: str,
    max_results: int = 10,
    msg_type: str | None = None,
    project_dir: Path | None = None,
    agent_id: str | None = None,
) -> list[dict]:
    """Search all channels for messages matching a query.

    Args:
        query: Text to search for in message bodies.
        max_results: Maximum results to return.
        msg_type: Optional filter by message type (e.g. "directive",
                  "message", "join", "leave"). When set, only messages
                  of that type are searched.
        agent_id: When provided, DM channels are included only if
                  the agent is a participant.  Without this, DM
                  channels are excluded from search results.

    Returns a list of dicts with channel, message_id, from, timestamp, body,
    type, and a match_score (number of query terms found).
    """
    terms = query.lower().split()
    if not terms and not msg_type:
        return []

    # When filtering by type with no query, match all messages of that type
    match_all = not terms

    # Build the list of channels to search: public channels + agent's DMs
    ch_dir = _channels_dir(project_dir)
    all_channels: list[str] = []
    if ch_dir.exists():
        for p in ch_dir.glob("*.jsonl"):
            name = _filename_to_channel(p.stem)
            if is_dm_channel(name):
                if agent_id and is_dm_participant(name, agent_id):
                    all_channels.append(name)
            else:
                all_channels.append(name)

    results: list[dict] = []
    for ch in all_channels:
        path = _channel_path(ch, project_dir)
        for msg in _read_messages(path):
            # Type filter
            if msg_type:
                if msg.type != msg_type:
                    continue
            elif msg.type not in ("message", "directive"):
                continue

            if match_all:
                score = 1
            else:
                body_lower = msg.body.lower()
                score = sum(1 for t in terms if t in body_lower)
                if score == 0:
                    continue

            results.append({
                "channel": ch,
                "message_id": msg.id,
                "from": msg.from_agent,
                "timestamp": msg.timestamp,
                "body": msg.body,
                "type": msg.type,
                "score": score,
            })

    # Sort by score descending, then timestamp descending (newest first)
    results.sort(key=lambda r: (r["score"], r["timestamp"]), reverse=True)
    return results[:max_results]


def check_directives(
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Check for unread directives targeted at this agent. Fast path for hooks.

    Scans only messages after this agent's cursor in channels it belongs to.
    Returns formatted directives for stdout, or empty string if none pending.
    Designed to complete in < 50ms for PostToolUse hook.
    """
    aid = agent_name or _agent_id(project_dir)

    conn = _open_db(project_dir)
    try:
        _reap_stale_agents(conn, project_dir)
        # Get channels this agent is a member of
        memberships = conn.execute(
            "SELECT channel FROM memberships WHERE agent_id = ?",
            (aid,),
        ).fetchall()
        if not memberships:
            return ""

        # Get cursors for all channels in one query
        channels = [r["channel"] for r in memberships]
        placeholders = ",".join("?" for _ in channels)
        cursor_rows = conn.execute(
            f"SELECT channel, last_read_at FROM cursors "
            f"WHERE agent_id = ? AND channel IN ({placeholders})",
            (aid, *channels),
        ).fetchall()
        cursors = {r["channel"]: r["last_read_at"] for r in cursor_rows}

        # Also heartbeat while we're here (one UPDATE, already have the conn)
        now = _now_iso()
        conn.execute(
            "UPDATE presence SET last_seen = ?, status = 'online' WHERE agent_id = ?",
            (now, aid),
        )
        conn.commit()
    finally:
        conn.close()

    # Scan for unread directives (JSONL read, filtered by cursor)
    # Also load claims to skip broadcast directives claimed by other agents
    conn2 = _open_db(project_dir)
    try:
        all_claims = {
            r["message_id"]: r["claimed_by"]
            for r in conn2.execute("SELECT message_id, claimed_by FROM claims").fetchall()
        }
    finally:
        conn2.close()

    # Check for unread @mentions (by agent_id or display_name)
    # Filter by earliest cursor to avoid resurfacing old mentions
    earliest_cursor = min(cursors.values()) if cursors else "1970-01-01T00:00:00Z"
    mention_conn = _open_db(project_dir)
    try:
        # Get this agent's display name for mention matching
        dn_row = mention_conn.execute(
            "SELECT display_name FROM presence WHERE agent_id = ?", (aid,)
        ).fetchone()
        display_name = dn_row["display_name"] if dn_row else ""

        # Find mentions targeting this agent, only after cursor
        if display_name:
            rows = mention_conn.execute(
                "SELECT message_id FROM mentions "
                "WHERE mentioned IN (?, ?) AND timestamp > ?",
                (aid, display_name, earliest_cursor),
            ).fetchall()
        else:
            rows = mention_conn.execute(
                "SELECT message_id FROM mentions "
                "WHERE mentioned = ? AND timestamp > ?",
                (aid, earliest_cursor),
            ).fetchall()
        mention_ids = {r["message_id"] for r in rows}
    finally:
        mention_conn.close()

    pending: list[tuple[str, ChannelMessage]] = []
    mentions: list[tuple[str, ChannelMessage]] = []
    for ch in channels:
        since = cursors.get(ch, "1970-01-01T00:00:00Z")
        path = _channel_path(ch, project_dir)
        if not path.exists():
            continue
        for msg in _read_messages(path, since=since):
            if msg.type == "directive" and (msg.to == aid or msg.to == "*"):
                # Skip broadcast directives claimed by another agent
                claimer = all_claims.get(msg.id)
                if msg.to == "*" and claimer and claimer != aid:
                    continue
                # Auto-claim directives on first read (#174)
                if not claimer:
                    try:
                        channel_claim(msg.id, ch, agent_name=aid, project_dir=project_dir)
                    except Exception:
                        pass  # Claim is best-effort
                pending.append((ch, msg))
            elif msg.id in mention_ids and msg.from_agent != aid:
                # @mention from another agent (not self-mentions)
                mentions.append((ch, msg))

    # Dedup: remove mentions that are already in pending directives
    pending_ids = {msg.id for _, msg in pending}
    mentions = [(ch, msg) for ch, msg in mentions if msg.id not in pending_ids]

    # Collect recent intents from other agents (last 10 min)
    # so this agent sees what others are working on
    intents: list[tuple[str, ChannelMessage]] = []
    for ch in channels:
        since = cursors.get(ch, "1970-01-01T00:00:00Z")
        path = _channel_path(ch, project_dir)
        if not path.exists():
            continue
        for msg in _read_messages(path, since=since):
            if msg.type == "intent" and msg.from_agent != aid:
                intents.append((ch, msg))

    if not pending and not mentions and not intents:
        return ""

    # Format for output (will appear as system reminder in context)
    lines = []
    if pending:
        lines.append(f"[channel] {len(pending)} pending directive(s):")
        for ch, msg in pending:
            lines.append(f"  #{ch} from {msg.from_agent}: {msg.body}")
    if mentions:
        lines.append(f"[channel] {len(mentions)} @mention(s):")
        for ch, msg in mentions:
            lines.append(f"  #{ch} from {msg.from_agent}: {msg.body}")
    if intents:
        lines.append(f"[channel] {len(intents)} active intent(s):")
        for ch, msg in intents:
            display = _resolve_display_name_for(msg.from_agent, project_dir)
            lines.append(f"  #{ch} {display} is working on: {msg.body}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# URL parsing helpers for org/project resolution
# ---------------------------------------------------------------------------

def _extract_org_from_url(url: str) -> str:
    """Extract GitHub org from a manifest URL.

    Handles both SSH and HTTPS formats:
    - git@github.com:synapt-dev/gripspace.git → synapt-dev
    - https://github.com/synapt-dev/gripspace.git → synapt-dev
    """
    import re
    # SSH: git@github.com:org/repo.git
    m = re.search(r"[:/]([^/]+)/[^/]+?(?:\.git)?$", url)
    if m:
        return m.group(1)
    return ""


def _extract_repo_from_url(url: str) -> str:
    """Extract repo name from a manifest URL.

    Handles both SSH and HTTPS formats:
    - git@github.com:synapt-dev/gripspace.git → gripspace
    - https://github.com/synapt-dev/gripspace.git → gripspace
    """
    import re
    m = re.search(r"/([^/]+?)(?:\.git)?$", url)
    if m:
        return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Channel migration: local → global store (Phase 1)
# ---------------------------------------------------------------------------

def migrate_channels_to_global(
    local_dir: Path,
    global_dir: Path,
    org_id: str,
    project_id: str,
) -> None:
    """Migrate channel data from local .synapt/recall/channels/ to global store.

    Copies JSONL messages to ~/.synapt/channels/<org>/<project>/.
    Migrates cursors from local channels.db to global _state.db.
    Idempotent: skips messages that already exist in global store.
    """
    if not local_dir.exists():
        return

    target_dir = global_dir / org_id / project_id
    # This function composes the store path itself — it neither calls
    # _channels_dir nor accepts a channels_dir, so it sits outside both of the
    # forms the other guards cover and needs its own.
    _guard_store_path("migrate_channels_to_global", target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Migrate JSONL channel files
    for jsonl_file in sorted(local_dir.glob("*.jsonl")):
        channel_name = jsonl_file.stem
        target_file = target_dir / jsonl_file.name

        # Read local messages
        local_messages = []
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        local_messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not local_messages:
            continue

        # Read existing global messages (for partial migration / idempotency)
        existing_timestamps: set[str] = set()
        if target_file.exists():
            with open(target_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            msg = json.loads(line)
                            # Use timestamp + from_agent + body as dedup key
                            key = f"{msg.get('timestamp', '')}\t{msg.get('from_agent', '')}\t{msg.get('body', '')}"
                            existing_timestamps.add(key)
                        except json.JSONDecodeError:
                            continue

        # Append only new messages
        new_messages = []
        for msg in local_messages:
            key = f"{msg.get('timestamp', '')}\t{msg.get('from_agent', '')}\t{msg.get('body', '')}"
            if key not in existing_timestamps:
                new_messages.append(msg)

        if new_messages:
            with open(target_file, "a", encoding="utf-8") as f:
                for msg in new_messages:
                    f.write(json.dumps(msg) + "\n")

    # Migrate cursors from local channels.db to global _state.db
    local_db = local_dir / "channels.db"
    if local_db.exists():
        _migrate_cursors(local_db, global_dir, org_id, project_id)


def _migrate_cursors(
    local_db: Path,
    global_dir: Path,
    org_id: str,
    project_id: str,
) -> None:
    """Migrate cursor data from local channels.db to global _state.db."""
    state_db = global_dir / "_state.db"
    # Reachable directly, not only via migrate_channels_to_global.
    _guard_store_path("migrate_cursors", state_db)
    state_db.parent.mkdir(parents=True, exist_ok=True)

    # Read local cursors
    local_conn = sqlite3.connect(str(local_db))
    local_conn.execute("PRAGMA busy_timeout=5000")
    local_conn.row_factory = sqlite3.Row
    try:
        rows = local_conn.execute(
            "SELECT agent_id, channel, last_read_at FROM cursors"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        local_conn.close()

    if not rows:
        return

    # Write to global _state.db
    conn = sqlite3.connect(str(state_db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cursors ("
        "agent_id TEXT NOT NULL, "
        "org_id TEXT NOT NULL, "
        "project_id TEXT NOT NULL, "
        "channel TEXT NOT NULL, "
        "cursor_value TEXT NOT NULL, "
        "last_read_at TEXT NOT NULL, "
        "PRIMARY KEY (agent_id, org_id, project_id, channel))"
    )
    for row in rows:
        conn.execute(
            "INSERT OR REPLACE INTO cursors "
            "(agent_id, org_id, project_id, channel, cursor_value, last_read_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (row["agent_id"], org_id, project_id, row["channel"],
             row["last_read_at"], row["last_read_at"]),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Global claims in _state.db (Story #12)
# ---------------------------------------------------------------------------

def _open_state_db(state_db: Path) -> sqlite3.Connection:
    """Open or create the global _state.db with WAL mode."""
    # Caller-supplied path, same shape as an explicit channels_dir: it reaches
    # around resolution entirely, so it is guarded in its own right. This one
    # can land inside the global store, unlike the per-gripspace channels.db.
    _guard_store_path("open_state_db", state_db)
    state_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(state_db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS claims ("
        "org_id TEXT NOT NULL, "
        "project_id TEXT NOT NULL, "
        "channel TEXT NOT NULL, "
        "message_id TEXT NOT NULL, "
        "claimed_by TEXT NOT NULL, "
        "display_name TEXT NOT NULL, "
        "claimed_at TEXT NOT NULL, "
        "PRIMARY KEY (org_id, project_id, channel, message_id))"
    )
    # Ensure cursors table exists too
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cursors ("
        "agent_id TEXT NOT NULL, "
        "org_id TEXT NOT NULL, "
        "project_id TEXT NOT NULL, "
        "channel TEXT NOT NULL, "
        "cursor_value TEXT NOT NULL, "
        "last_read_at TEXT NOT NULL, "
        "PRIMARY KEY (agent_id, org_id, project_id, channel))"
    )
    return conn


def global_claim(
    state_db: Path,
    org_id: str,
    project_id: str,
    channel: str,
    message_id: str,
    claimed_by: str,
    display_name: str,
) -> bool:
    """Claim a message in global _state.db. First writer wins.

    Returns True if the claim succeeded, False if already claimed.
    """
    conn = _open_state_db(state_db)
    try:
        now = _now_iso()
        cursor = conn.execute(
            "INSERT OR IGNORE INTO claims "
            "(org_id, project_id, channel, message_id, claimed_by, display_name, claimed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (org_id, project_id, channel, message_id, claimed_by, display_name, now),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def global_unclaim(
    state_db: Path,
    org_id: str,
    project_id: str,
    channel: str,
    message_id: str,
    claimed_by: str,
) -> bool:
    """Unclaim a message. Only the original claimer can unclaim.

    Returns True if unclaimed, False if not the claimer or not claimed.
    """
    conn = _open_state_db(state_db)
    try:
        cursor = conn.execute(
            "DELETE FROM claims WHERE org_id = ? AND project_id = ? "
            "AND channel = ? AND message_id = ? AND claimed_by = ?",
            (org_id, project_id, channel, message_id, claimed_by),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def is_globally_claimed(
    state_db: Path,
    org_id: str,
    project_id: str,
    channel: str,
    message_id: str,
) -> str | None:
    """Check if a message is claimed. Returns claimer agent_id or None."""
    conn = _open_state_db(state_db)
    try:
        row = conn.execute(
            "SELECT claimed_by FROM claims WHERE org_id = ? AND project_id = ? "
            "AND channel = ? AND message_id = ?",
            (org_id, project_id, channel, message_id),
        ).fetchone()
        return row["claimed_by"] if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Default hook registration
# ---------------------------------------------------------------------------
# Auto-register mention storage and wake emission as the default hooks so
# channel behavior is consistent out of the box.

def _default_mention_hook(msg: ChannelMessage, project_dir: Path | None = None) -> None:
    """Store @mentions from a posted message (default OSS hook)."""
    if msg.type in ("message", "directive") and "@" in msg.body:
        _store_mentions(msg, project_dir)


def _default_wake_hook(msg: ChannelMessage, project_dir: Path | None = None) -> None:
    """Emit wake requests for a posted message (default OSS hook)."""
    _emit_message_wakes(msg, project_dir)


register_message_hook(_default_mention_hook)
register_message_hook(_default_wake_hook)
