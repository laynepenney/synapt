"""Org agent registry — stable identity for agents across sessions.

Every agent in an org gets a unique, permanent agent_id (e.g., "opus-001").
Display names are human-readable and unique per org but can be changed.

Storage: org_agents table in ~/.synapt/orgs/<org_id>/team.db.
Resolution: SYNAPT_AGENT_ID env var → presence DB lookup → auto-register.

Part of Phase 0 of the channel scoping design (5-iteration adversarial review).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synapt.recall.session_start import _pid_alive


def _team_db_path(org_id: str) -> Path:
    """Return the path to an org's team.db."""
    return Path.home() / ".synapt" / "orgs" / org_id / "team.db"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the org_agents table if it doesn't exist."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS org_agents (
            agent_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            role TEXT,
            org_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT,
            session_id TEXT,
            pid INTEGER,
            status TEXT,
            tmux_target TEXT,
            log_path TEXT
        )"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_org_display
           ON org_agents(org_id, display_name)"""
    )
    # Add process columns to existing tables (no-op if already present)
    for col, typ in [
        ("session_id", "TEXT"),
        ("pid", "INTEGER"),
        ("status", "TEXT"),
        ("tmux_target", "TEXT"),
        ("log_path", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE org_agents ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()


def _generate_agent_id(display_name: str, conn: sqlite3.Connection, org_id: str) -> str:
    """Generate a unique agent_id in the format 'name-NNN'.

    Finds the next available number for the given name prefix.
    """
    slug = display_name.lower().replace(" ", "-")
    rows = conn.execute(
        "SELECT agent_id FROM org_agents WHERE agent_id LIKE ?",
        (f"{slug}-%",),
    ).fetchall()
    existing_nums = []
    for (aid,) in rows:
        parts = aid.rsplit("-", 1)
        if len(parts) == 2:
            try:
                existing_nums.append(int(parts[1]))
            except ValueError:
                pass
    next_num = max(existing_nums, default=0) + 1
    return f"{slug}-{next_num:03d}"


def _open_db(org_id: str, db_path: Path | None = None) -> sqlite3.Connection:
    """Open team.db for the given org, ensuring schema exists."""
    path = db_path or _team_db_path(org_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _check_org_entitlement(org_id: str, db_path: Path | None = None) -> None:
    """Verify the caller is entitled to register agents in this org.

    Entitlement is established by any of:
    - db_path explicitly passed (test/internal use, caller owns the path)
    - SYNAPT_AGENT_ID env var is set (process was spawned by gr spawn)
    - org directory already contains a team.db (org was initialized by gr)

    Raises PermissionError if none of these conditions hold.
    Security: recall#530.
    """
    if db_path is not None:
        return
    if os.environ.get("SYNAPT_AGENT_ID"):
        return
    team_db = _team_db_path(org_id)
    if team_db.exists():
        return
    raise PermissionError(
        f"Cannot register agent in org '{org_id}': no entitlement. "
        f"Use `gr spawn` to create agents, or initialize the org first."
    )


def register_agent(
    org_id: str,
    display_name: str,
    role: str | None = None,
    db_path: Path | None = None,
) -> str:
    """Register a new agent in the org. Returns the assigned agent_id.

    Raises sqlite3.IntegrityError if display_name is already taken in this org.
    Raises PermissionError if the caller lacks org entitlement (recall#530).
    """
    _check_org_entitlement(org_id, db_path)
    conn = _open_db(org_id, db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        agent_id = _generate_agent_id(display_name, conn, org_id)
        conn.execute(
            "INSERT INTO org_agents (agent_id, display_name, role, org_id, created_at, last_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (agent_id, display_name, role, org_id, now, now),
        )
        conn.commit()
        return agent_id
    finally:
        conn.close()


def get_agent(agent_id: str, org_id: str | None = None, db_path: Path | None = None) -> dict[str, Any] | None:
    """Look up an agent by agent_id. Returns dict or None if not found."""
    if org_id is None and db_path is None:
        raise ValueError("Either org_id or db_path must be provided")
    conn = _open_db(org_id or "", db_path)
    try:
        row = conn.execute(
            "SELECT * FROM org_agents WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_agent_by_name(
    org_id: str, display_name: str, db_path: Path | None = None
) -> dict[str, Any] | None:
    """Look up an agent by org + display_name. Returns dict or None."""
    conn = _open_db(org_id, db_path)
    try:
        row = conn.execute(
            "SELECT * FROM org_agents WHERE org_id = ? AND display_name = ?",
            (org_id, display_name),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_agents(org_id: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    """List all agents in an org."""
    conn = _open_db(org_id, db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM org_agents WHERE org_id = ? ORDER BY created_at",
            (org_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_display_name(
    agent_id: str,
    new_display_name: str,
    org_id: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Update an agent's display name. Raises IntegrityError if name is taken."""
    if org_id is None and db_path is None:
        raise ValueError("Either org_id or db_path must be provided")
    conn = _open_db(org_id or "", db_path)
    try:
        conn.execute(
            "UPDATE org_agents SET display_name = ? WHERE agent_id = ?",
            (new_display_name, agent_id),
        )
        conn.commit()
    finally:
        conn.close()


def touch_last_seen(agent_id: str, org_id: str, db_path: Path | None = None) -> None:
    """Update the last_seen_at timestamp for an agent."""
    conn = _open_db(org_id, db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE org_agents SET last_seen_at = ? WHERE agent_id = ?",
            (now, agent_id),
        )
        conn.commit()
    finally:
        conn.close()


def resolve_agent_id(org_id: str, db_path: Path | None = None) -> str | None:
    """Resolve the current agent's ID.

    Priority:
    1. SYNAPT_AGENT_ID env var (set by gr spawn)
    2. None (caller should auto-register)
    """
    env_id = os.environ.get("SYNAPT_AGENT_ID")
    if env_id:
        return env_id
    return None


# ---------------------------------------------------------------------------
# Process tracking (recall#538)
# ---------------------------------------------------------------------------

def update_agent_status(
    db_path: Path,
    agent_id: str,
    status: str,
    session_id: str | None = None,
    pid: int | None = None,
    tmux_target: str | None = None,
    log_path: str | None = None,
) -> None:
    """Update process tracking columns for an agent."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    try:
        conn.execute(
            "UPDATE org_agents SET status = ?, session_id = ?, pid = ?, "
            "tmux_target = ?, log_path = ?, last_seen_at = datetime('now') "
            "WHERE agent_id = ?",
            (status, session_id, pid, tmux_target, log_path, agent_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_agent_status(db_path: Path, agent_id: str) -> dict[str, Any] | None:
    """Return process tracking info for an agent."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    try:
        row = conn.execute(
            "SELECT agent_id, display_name, status, session_id, pid, "
            "tmux_target, log_path FROM org_agents WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def detect_crashed_agents(db_path: Path) -> list[dict[str, Any]]:
    """Find agents with status='running' but dead PIDs."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    try:
        rows = conn.execute(
            "SELECT agent_id, pid, session_id FROM org_agents "
            "WHERE status = 'running' AND pid IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    crashed = []
    for row in rows:
        pid = row["pid"]
        if not _pid_alive(pid):
            crashed.append(dict(row))
    return crashed


def clear_agent_session(db_path: Path, agent_id: str) -> None:
    """Reset process columns while preserving agent identity."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    try:
        conn.execute(
            "UPDATE org_agents SET status = 'stopped', session_id = NULL, "
            "pid = NULL, tmux_target = NULL, log_path = NULL "
            "WHERE agent_id = ?",
            (agent_id,),
        )
        conn.commit()
    finally:
        conn.close()
