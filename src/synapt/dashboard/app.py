"""synapt dashboard — web-based mission control for multi-agent sessions.

Launch with: synapt dashboard [--port 8420] [--no-open]
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]
from html import escape
from pathlib import Path
from urllib.parse import quote

import markdown as _md
import nh3
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse

from synapt.recall.channel import _channels_dir
from synapt.recall.core import project_data_dir
from synapt.recall.channel import (
    ChannelMessage,
    channel_agents_json,
    channel_list_channels,
    channel_messages_json,
    channel_post,
    _resolve_org_id,
    _resolve_project_id,
)
from synapt.recall.registry import list_agents as _registry_list_agents

_MD = _md.Markdown(extensions=["fenced_code", "tables", "nl2br"])

_SAFE_TAGS = {
    "p", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "strong", "em", "b", "i", "u", "s", "del",
    "code", "pre",
    "a",
    "ul", "ol", "li",
    "table", "thead", "tbody", "tr", "th", "td",
    "blockquote",
    "span", "div", "img",
    "sup", "sub",
}
_SAFE_ATTRS = {
    "*": {"class", "style"},
    "a": {"href", "title", "target"},
    "img": {"src", "alt", "title"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan"},
}


def _sanitize_html(html: str) -> str:
    return nh3.clean(html, tags=_SAFE_TAGS, attributes=_SAFE_ATTRS, link_rel=None)

# ---------------------------------------------------------------------------
# Agent tool detection (codex vs claude)
# ---------------------------------------------------------------------------

_CODEX_AGENTS: set[str] = set()
_KNOWN_AGENTS: dict[str, dict] = {}


def _load_agents_toml() -> None:
    """Load agents.toml: populate _CODEX_AGENTS and _KNOWN_AGENTS."""
    from synapt.recall.core import _find_gripspace_root

    grip_root = _find_gripspace_root(Path.cwd())
    if grip_root is None:
        return
    toml_path = grip_root / ".gitgrip" / "agents.toml"
    if not toml_path.is_file():
        toml_path = grip_root / "config" / "agents.toml"
    if not toml_path.is_file():
        return
    try:
        with open(toml_path, "rb") as f:
            cfg = tomllib.load(f)
        for name, agent_cfg in cfg.get("agents", {}).items():
            if agent_cfg.get("tool") == "codex":
                _CODEX_AGENTS.add(name)
            _KNOWN_AGENTS[name] = agent_cfg
    except (OSError, tomllib.TOMLDecodeError):
        pass


_load_agents_toml()


def _tmux_window_agents(tmux_session: str | None) -> dict[str, str]:
    """Return known agent windows from one explicitly bound tmux session."""
    if tmux_session is None:
        return {}
    session = _require_tmux_session(tmux_session)
    try:
        result = subprocess.run(
            ["tmux", "list-windows", "-t", session, "-F", "#{window_name}"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return {}
        windows = set(result.stdout.strip().splitlines())
        return {w: "online" for w in windows if w in _KNOWN_AGENTS}
    except Exception:
        return {}

class TmuxTargetScopeError(RuntimeError):
    """A dashboard tmux target is absent or outside its explicit session scope."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _require_tmux_session(tmux_session: str | None) -> str:
    """Return one explicit session component or refuse control availability.

    OSS owns only this neutral capability boundary.  Choosing which session a
    workspace may control belongs to the caller.  No working-directory lookup,
    registry value, or built-in session name may manufacture that authority.
    """
    if tmux_session is None:
        raise TmuxTargetScopeError(
            "tmux controls require an explicitly bound session",
            status_code=503,
        )
    session = str(tmux_session)
    if (
        not session
        or session != session.strip()
        or ":" in session
        or any(ord(char) < 32 or ord(char) == 127 for char in session)
    ):
        raise TmuxTargetScopeError("invalid tmux session scope", status_code=503)
    return session


def _assert_tmux_target_in_session(target: str, tmux_session: str) -> str:
    """Return *target* only when its session is exactly the bound session."""
    session = target.split(":", 1)[0]
    if not target or ":" not in target or session != tmux_session:
        raise TmuxTargetScopeError(
            f"tmux target is outside the dashboard session scope: {target!r}",
            status_code=403,
        )
    return target


def _resolve_tmux_target(name: str, *, tmux_session: str | None) -> str:
    """Resolve an agent only within an explicitly supplied tmux session.

    A registry row is a coordinate, not authority.  A stored target naming any
    other session is refused rather than trusted or replaced by a fallback.
    """
    session = _require_tmux_session(tmux_session)
    org_id = _resolve_org_id(None)
    if org_id:
        db_path = Path.home() / ".synapt" / "orgs" / org_id / "team.db"
        if db_path.exists():
            try:
                agents = _registry_list_agents(org_id, db_path=db_path)
            except Exception:
                agents = []
            for agent in agents:
                if str(agent.get("display_name", "")).casefold() == name.casefold():
                    stored = agent.get("tmux_target")
                    if stored:
                        return _assert_tmux_target_in_session(str(stored), session)
                    break
    return f"{session}:{name.casefold()}"


# ---------------------------------------------------------------------------
# HTML fragment renderers
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "online": "#4ade80",
    "idle": "#facc15",
    "away": "#fb923c",
    "offline": "#6b7280",
}

# Distinct colors per agent for the message feed
_AGENT_COLORS = [
    "#8b5cf6",  # purple (Opus)
    "#06b6d4",  # cyan (Apollo)
    "#4ade80",  # green (Sentinel)
    "#fb923c",  # orange (Atlas)
    "#f472b6",  # pink
    "#facc15",  # yellow
    "#a78bfa",  # light purple
    "#34d399",  # emerald
]
_agent_color_cache: dict[str, str] = {}


def _agent_color(name: str) -> str:
    """Assign a stable color to an agent name."""
    if name not in _agent_color_cache:
        idx = len(_agent_color_cache) % len(_AGENT_COLORS)
        _agent_color_cache[name] = _AGENT_COLORS[idx]
    return _agent_color_cache[name]


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _combined_agents_json_sync(tmux_session: str | None = None) -> list[dict]:
    """Return agents from both team.db (process tracking) and channel presence.

    Scoped to the current org only (recall#692) to avoid SQLite lock
    contention when multiple workspaces are active simultaneously.

    Merge strategy: team.db agents are the authority for process status.
    Channel presence supplements with channel memberships and heartbeat.
    """
    # Start with channel presence (existing behavior)
    agents_by_name: dict[str, dict] = {}
    try:
        for agent in channel_agents_json():
            name = agent.get("display_name") or agent.get("agent_id", "")
            agents_by_name[name] = agent
    except Exception:
        pass

    # Overlay with team.db agents — current org only (recall#692 fix 1)
    current_org = _resolve_org_id(None)
    if current_org:
        db_path = Path.home() / ".synapt" / "orgs" / current_org / "team.db"
        if db_path.exists():
            try:
                registered = _registry_list_agents(current_org, db_path=db_path)
            except Exception:
                registered = []
            for agent in registered:
                name = agent.get("display_name", "")
                status = agent.get("status") or "offline"
                pid = agent.get("pid")
                if status in ("running", "online") and pid:
                    if not _is_pid_alive(pid):
                        status = "offline"
                if status in ("offline", "stopped") and name not in agents_by_name:
                    continue
                tmux_target = agent.get("tmux_target") or ""
                if name not in agents_by_name:
                    agents_by_name[name] = {
                        "agent_id": agent.get("agent_id", ""),
                        "display_name": name,
                        "griptree": "",
                        "role": agent.get("role", "agent"),
                        "status": status,
                        "last_seen": agent.get("last_seen_at", ""),
                        "channels": [],
                        "tmux_target": tmux_target,
                    }
                else:
                    existing = agents_by_name[name]
                    if status in ("running", "online"):
                        existing["status"] = status
                    if tmux_target:
                        existing["tmux_target"] = tmux_target

    # Discover agents from tmux windows that match agents.toml
    for window_name, status in _tmux_window_agents(tmux_session).items():
        display = window_name.capitalize()
        if display not in agents_by_name and window_name not in agents_by_name:
            agent_cfg = _KNOWN_AGENTS.get(window_name, {})
            agents_by_name[display] = {
                "agent_id": window_name,
                "display_name": display,
                "griptree": agent_cfg.get("worktree", ""),
                "role": agent_cfg.get("role", "agent"),
                "status": status,
                "last_seen": "",
                "channels": [],
                "tmux_target": (
                    f"{tmux_session}:{window_name}" if tmux_session else ""
                ),
            }

    return sorted(agents_by_name.values(), key=lambda a: a.get("display_name", ""))


async def _combined_agents_json(tmux_session: str | None = None) -> list[dict]:
    """Async wrapper — runs SQLite reads off the event loop (recall#692 fix 2)."""
    return await asyncio.to_thread(_combined_agents_json_sync, tmux_session)


def _render_agent_tile(agent: dict) -> str:
    status = agent["status"]
    color = _STATUS_COLORS.get(status, "#6b7280")
    name = agent["display_name"] or agent["griptree"] or agent["agent_id"]
    role = agent["role"] if agent["role"] != "agent" else ""
    griptree = agent.get("griptree", "")
    channels = ", ".join(f"#{c}" for c in agent["channels"]) or "no channels"
    seen = agent["last_seen"][11:16] if len(agent["last_seen"]) > 16 else ""
    project_badge = (
        f'<div class="tile-project">{escape(griptree)}</div>' if griptree else ""
    )
    return (
        f'<div class="tile clickable" data-agent="{escape(name)}" style="border-left:4px solid {color}">'
        f'<div class="tile-name">{escape(name)}</div>'
        f'{project_badge}'
        f'<div class="tile-role">{escape(role)}</div>'
        f'<div class="tile-meta">'
        f'<span style="color:{color}">{status}</span>'
        f' &middot; {escape(channels)}'
        f' &middot; {seen}'
        f'</div></div>'
    )


def _attachment_url(rel_path: str) -> str:
    """Return the dashboard URL for a stored attachment."""
    return f"/api/attachments/{quote(rel_path, safe='/')}"


def _is_image_attachment(rel_path: str) -> bool:
    """Return True when the attachment should render inline as an image."""
    suffix = Path(rel_path).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _render_attachments(msg: dict) -> str:
    """Render stored attachments as inline images or links."""
    attachments = msg.get("attachments") or []
    if not attachments:
        return ""

    parts: list[str] = ['<div class="attachments">']
    for rel_path in attachments:
        url = _attachment_url(rel_path)
        label = escape(Path(rel_path).name)
        if _is_image_attachment(rel_path):
            parts.append(
                '<a class="attachment-link image-link" href="{}" target="_blank" rel="noopener">'
                '<img class="attachment-image" src="{}" alt="{}">'
                "</a>".format(url, url, label)
            )
        else:
            parts.append(
                '<a class="attachment-link" href="{}" target="_blank" rel="noopener">{}</a>'.format(
                    url, label
                )
            )
    parts.append("</div>")
    return "".join(parts)


def _resolve_attachment_path(rel_path: str) -> Path:
    """Resolve a stored attachment path safely inside the channel store."""
    base = _channels_dir().resolve()
    candidate = (base / rel_path).resolve()
    if not candidate.is_relative_to(base):
        raise HTTPException(status_code=404, detail="Attachment not found")
    return candidate


def _render_message(msg: dict) -> str:
    ts = msg.get("timestamp", "")
    ts_short = ts[11:16] if len(ts) > 16 else ts
    name = msg.get("from_display") or msg.get("from", "")
    body = msg.get("body", "")
    msg_type = msg.get("type", "message")
    to = msg.get("to", "")
    attachments_html = _render_attachments(msg)

    if msg_type in ("join", "leave"):
        return ''
    color = _agent_color(name)
    _MD.reset()
    # Escape leading '#' not followed by space — prevents markdown
    # from turning "#celebrate" into an <h1> heading. (recall#630)
    body_escaped = re.sub(r'^(#{1,6})(?=[^ #])', r'\\\1', body, flags=re.MULTILINE)
    body_html = _MD.convert(body_escaped)
    # Color @mentions — skip content inside <code> and <pre> tags
    def _color_mentions(html: str) -> str:
        parts = re.split(r'(<code.*?>.*?</code>|<pre.*?>.*?</pre>)', html, flags=re.DOTALL)
        for i, part in enumerate(parts):
            if not part.startswith(('<code', '<pre')):
                parts[i] = re.sub(
                    r'@(\w+)',
                    lambda m: f'<span style="color:{_agent_color(m.group(1))};font-weight:600">@{m.group(1)}</span>',
                    part,
                )
        return ''.join(parts)
    body_html = _sanitize_html(_color_mentions(body_html))
    if msg_type == "directive":
        return (
            f'<div class="msg directive">'
            f'<span class="ts" data-utc="{escape(ts)}">{ts_short}</span> '
            f'<b style="color:{color}">{escape(name)}</b> &rarr; @{escape(to)}: '
            f'<span class="msg-body">{body_html}</span>'
            f'{attachments_html}'
            f'</div>'
        )
    return (
        f'<div class="msg">'
        f'<span class="ts" data-utc="{escape(ts)}">{ts_short}</span> '
        f'<b style="color:{color}">{escape(name)}</b>: '
        f'<span class="msg-body">{body_html}</span>'
        f'{attachments_html}'
        f'</div>'
    )


def _render_agents_html(agents: list[dict]) -> str:
    if not agents:
        return '<div class="tile empty">No agents online</div>'
    return "".join(_render_agent_tile(a) for a in agents)


def _render_messages_html(messages: list[dict]) -> str:
    return "".join(_render_message(m) for m in messages)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

_TEMPLATE: str | None = None


def _load_template() -> str:
    global _TEMPLATE
    if _TEMPLATE is None:
        path = Path(__file__).parent / "template.html"
        _TEMPLATE = path.read_text()
    return _TEMPLATE


def _ensure_dashboard_join(
    joined: set[tuple[str, str]],
    channel: str,
    name: str,
) -> None:
    """Ensure the dashboard user has an explicit human presence entry."""
    from synapt.recall.channel import channel_join

    clean_name = (name or "dashboard").strip() or "dashboard"
    join_key = (channel, clean_name)
    if join_key in joined:
        return
    channel_join(channel=channel, agent_name="dashboard", display_name=clean_name, role="human")
    joined.add(join_key)


def _dashboard_pid_path(project_dir: Path | None = None) -> Path:
    """Return the dashboard PID file under the project .synapt root."""
    return project_data_dir(project_dir).parent / "dashboard.pid"


def _dashboard_log_path(project_dir: Path | None = None) -> Path:
    """Return the dashboard log file under the project .synapt root."""
    return project_data_dir(project_dir).parent / "dashboard.log"


def _read_pid(pid_path: Path) -> int | None:
    """Read a PID file, returning None for missing or invalid content."""
    try:
        raw = pid_path.read_text().strip()
    except FileNotFoundError:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _pid_is_running(pid: int) -> bool:
    """Return True when the process exists and is signalable."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_stale_pidfile(pid_path: Path) -> None:
    """Remove the PID file if it points at no live process."""
    pid = _read_pid(pid_path)
    if pid is None or not _pid_is_running(pid):
        pid_path.unlink(missing_ok=True)


def _stop_dashboard(project_dir: Path | None = None) -> bool:
    """Stop a background dashboard server if one is running."""
    pid_path = _dashboard_pid_path(project_dir)
    pid = _read_pid(pid_path)
    if pid is None:
        pid_path.unlink(missing_ok=True)
        return False
    if not _pid_is_running(pid):
        pid_path.unlink(missing_ok=True)
        return False

    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not _pid_is_running(pid):
            pid_path.unlink(missing_ok=True)
            return True
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    pid_path.unlink(missing_ok=True)
    return True


def _background_command(
    host: str,
    port: int,
    *,
    tmux_session: str | None = None,
) -> list[str]:
    """Build the detached child command for the dashboard server."""
    command = [
        sys.executable,
        "-m",
        "synapt.cli",
        "dashboard",
        "--foreground",
        "--host",
        host,
        "--port",
        str(port),
        "--no-open",
    ]
    if tmux_session is not None:
        command.extend(("--tmux-session", tmux_session))
    return command


def _start_dashboard_background(
    host: str,
    port: int,
    no_open: bool,
    project_dir: Path | None = None,
    tmux_session: str | None = None,
) -> int:
    """Spawn the dashboard server in the background and persist its PID."""
    synapt_dir = project_data_dir(project_dir).parent
    synapt_dir.mkdir(parents=True, exist_ok=True)
    pid_path = _dashboard_pid_path(project_dir)
    log_path = _dashboard_log_path(project_dir)
    _cleanup_stale_pidfile(pid_path)

    existing_pid = _read_pid(pid_path)
    if existing_pid is not None and _pid_is_running(existing_pid):
        if not no_open:
            import webbrowser

            webbrowser.open(f"http://{host}:{port}")
        return existing_pid

    cmd = _background_command(host, port, tmux_session=tmux_session)
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            cwd=str(Path.cwd()),
            start_new_session=True,
        )

    time.sleep(0.2)
    if proc.poll() is not None:
        raise RuntimeError(
            f"Dashboard exited immediately with status {proc.returncode}. "
            f"See {log_path}."
        )

    pid_path.write_text(f"{proc.pid}\n")
    if not no_open:
        import webbrowser

        webbrowser.open(f"http://{host}:{port}")
    return proc.pid


def _all_channels_nav() -> dict:
    """Return hierarchical org → project → channel navigation data.

    Scans ``~/.synapt/channels/<org>/<project>/`` globally so the dashboard
    can show all visible channels, not just the current project's.
    Falls back to local channel list for non-gripspace repos.
    """
    global_dir = Path.home() / ".synapt" / "channels"
    current_org = _resolve_org_id(None) or ""
    current_project = _resolve_project_id(None) or ""

    projects = []

    if global_dir.exists():
        for org_dir in sorted(global_dir.iterdir()):
            if not org_dir.is_dir() or org_dir.name.startswith("_"):
                continue
            for proj_dir in sorted(org_dir.iterdir()):
                if not proj_dir.is_dir():
                    continue
                channels = sorted(p.stem for p in proj_dir.glob("*.jsonl"))
                if not channels:
                    continue
                is_active = (
                    org_dir.name == current_org and proj_dir.name == current_project
                )
                projects.append(
                    {
                        "org": org_dir.name,
                        "project": proj_dir.name,
                        "channels": channels,
                        "active": is_active,
                    }
                )

    # Fallback: no global store found — use local channels
    if not projects:
        local_channels = channel_list_channels()
        if local_channels:
            projects.append(
                {
                    "org": current_org or "local",
                    "project": current_project or "local",
                    "channels": local_channels,
                    "active": True,
                }
            )

    return {
        "org": current_org,
        "project": current_project,
        "projects": projects,
    }


def _channels_dir_for(org: str | None, project: str | None) -> "Path | None":
    """Return the channels directory for a specific org/project, or None for current."""
    if not org or not project:
        return None
    return Path.home() / ".synapt" / "channels" / org / project


def create_app(*, tmux_session: str | None = None) -> FastAPI:
    """Build the dashboard with an optional, explicit tmux control capability.

    Without ``tmux_session`` the dashboard remains useful for channels and
    status, but every route that would read from or write to tmux fails closed.
    The value is captured at construction so a later cwd/config change cannot
    silently move the control surface to another session.
    """
    app = FastAPI(title="synapt dashboard", docs_url=None, redoc_url=None)

    def scoped_tmux_target(name: str) -> str:
        session = _require_tmux_session(tmux_session)
        target = _resolve_tmux_target(name, tmux_session=session)
        # Use-site floor: a resolver regression must still be unable to hand a
        # foreign coordinate to subprocess.  This deliberately re-checks the
        # value about to leave the process rather than trusting resolution.
        return _assert_tmux_target_in_session(target, session)

    def resolve_tmux_target(name: str) -> str:
        try:
            return scoped_tmux_target(name)
        except TmuxTargetScopeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _load_template()

    @app.get("/api/agents")
    async def api_agents():
        return await _combined_agents_json(tmux_session)

    @app.get("/api/org")
    async def api_org():
        org = _resolve_org_id(None) or "unknown"
        project = _resolve_project_id(None) or "unknown"
        return {"org": org, "project": project}

    @app.get("/api/channels")
    async def api_channels():
        return channel_list_channels()

    @app.get("/api/nav")
    async def api_nav():
        return _all_channels_nav()

    @app.get("/api/messages/{channel}", response_class=HTMLResponse)
    async def api_messages(
        channel: str,
        limit: int = 50,
        since: str | None = None,
        org: str | None = None,
        project: str | None = None,
    ):
        ch_dir = _channels_dir_for(org, project)
        msgs = channel_messages_json(
            channel=channel, limit=limit, since=since, channels_dir=ch_dir
        )
        return _render_messages_html(msgs)

    @app.get("/api/attachments/{attachment_path:path}")
    async def api_attachment(attachment_path: str):
        path = _resolve_attachment_path(attachment_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Attachment not found")
        return FileResponse(path)

    _dashboard_joined: set[tuple[str, str]] = set()

    @app.post("/api/join/{channel}")
    async def api_join(channel: str, name: str = Form("dashboard")):
        _ensure_dashboard_join(_dashboard_joined, channel=channel, name=name)
        return {"ok": True}

    @app.post("/api/post/{channel}")
    async def api_post(
        channel: str,
        message: str = Form(""),
        name: str = Form("dashboard"),
        org: str = Form(""),
        project: str = Form(""),
        attachment: UploadFile | None = File(None),
    ):
        agent_name = "dashboard"
        ch_dir = _channels_dir_for(org or None, project or None)
        _ensure_dashboard_join(_dashboard_joined, channel=channel, name=name)

        attachment_paths: list[str] | None = None
        tmp_path: Path | None = None
        if attachment is not None and attachment.filename:
            suffix = Path(attachment.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(attachment.file, tmp)
                tmp_path = Path(tmp.name)
            attachment_paths = [str(tmp_path)]

        try:
            if not message.strip() and not attachment_paths:
                raise HTTPException(status_code=400, detail="Message or attachment required")
            channel_post(
                channel=channel,
                message=message,
                agent_name=agent_name,
                attachment_paths=attachment_paths,
                channels_dir=ch_dir,
            )
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/api/stream")
    async def stream(
        request: Request,
        channel: str = "dev",
        org: str | None = None,
        project: str | None = None,
    ):
        ch_dir = _channels_dir_for(org, project)

        async def generate():
            last_msg_ts = ""

            try:
                while True:
                    if await request.is_disconnected():
                        return

                    # New messages
                    try:
                        msgs = channel_messages_json(
                            channel=channel,
                            limit=20,
                            since=last_msg_ts or None,
                            channels_dir=ch_dir,
                        )
                    except Exception:
                        msgs = []
                    if msgs:
                        last_msg_ts = msgs[-1].get("timestamp", last_msg_ts)
                        html = _render_messages_html(msgs)
                        yield {"event": "messages", "data": html}

                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                return

        return EventSourceResponse(generate())

    # -----------------------------------------------------------------
    # Mission Control: per-agent tmux integration (Sprint 9)
    # -----------------------------------------------------------------

    # Allowed tmux key names for the /key endpoint (safety allowlist)
    _ALLOWED_KEYS = {
        "Enter", "Escape", "Up", "Down", "Left", "Right",
        "Tab", "BTab", "Space", "BSpace",
        "C-c", "C-d", "C-z", "C-l",
        "y", "n", "q",
    }

    @app.post("/api/agent/{name}/input")
    async def api_agent_input(name: str, text: str = Form("")):
        """Send input to an agent's tmux pane via send-keys.

        Codex agents require a second Enter to confirm the prompt (two-step
        input protocol).  The agent tool type is detected from agents.toml at
        startup; codex agents get an extra Enter after a short delay.

        If text is empty, sends a bare Enter (for confirming selections).
        """
        target = resolve_tmux_target(name)
        is_codex = name in _CODEX_AGENTS
        try:
            if text.strip():
                result = subprocess.run(
                    ["tmux", "send-keys", "-t", target, text, "Enter"],
                    capture_output=True,
                    timeout=5,
                )
            else:
                # Bare Enter for confirming selections/prompts
                result = subprocess.run(
                    ["tmux", "send-keys", "-t", target, "Enter"],
                    capture_output=True,
                    timeout=5,
                )
            if result.returncode != 0:
                raise HTTPException(
                    status_code=502,
                    detail=f"tmux send-keys failed: {result.stderr.decode().strip()}",
                )
            # Codex needs a second Enter to confirm the prompt
            if is_codex and text.strip():
                await asyncio.sleep(0.3)
                subprocess.run(
                    ["tmux", "send-keys", "-t", target, "Enter"],
                    capture_output=True,
                    timeout=5,
                )
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="tmux not available")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="tmux send-keys timed out")
        return {"ok": True, "agent": name, "codex": is_codex}

    # Upload directory for images/files sent to agents
    _UPLOAD_DIR = Path(tempfile.gettempdir()) / "synapt-uploads"
    _UPLOAD_DIR.mkdir(exist_ok=True)

    @app.post("/api/agent/{name}/upload")
    async def api_agent_upload(name: str, file: UploadFile = File(...)):
        """Upload a file and send its path to an agent's tmux pane.

        Saves the file to a persistent temp directory so the agent can
        read it.  The absolute file path is sent as text input to the
        agent's tmux pane.
        """
        target = resolve_tmux_target(name)
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        suffix = Path(file.filename).suffix
        stem = Path(file.filename).stem
        # Unique filename to avoid collisions
        ts = int(time.time())
        dest = _UPLOAD_DIR / f"{stem}-{ts}{suffix}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Send the file path to the agent's tmux pane
        file_path = str(dest)
        is_codex = name in _CODEX_AGENTS
        try:
            result = subprocess.run(
                ["tmux", "send-keys", "-t", target, file_path, "Enter"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise HTTPException(
                    status_code=502,
                    detail=f"tmux send-keys failed: {result.stderr.decode().strip()}",
                )
            if is_codex:
                await asyncio.sleep(0.3)
                subprocess.run(
                    ["tmux", "send-keys", "-t", target, "Enter"],
                    capture_output=True,
                    timeout=5,
                )
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="tmux not available")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="tmux send-keys timed out")
        return {"ok": True, "agent": name, "path": file_path}

    @app.post("/api/agent/{name}/key")
    async def api_agent_key(name: str, key: str = Form("")):
        """Send a raw tmux key name to an agent's pane.

        Accepts key names like Enter, Escape, Up, Down, C-c, y, n, etc.
        Only allowlisted key names are accepted for safety.
        """
        key = key.strip()
        if key not in _ALLOWED_KEYS:
            raise HTTPException(
                status_code=400,
                detail=f"Key not allowed: {key}. Allowed: {sorted(_ALLOWED_KEYS)}",
            )
        target = resolve_tmux_target(name)
        try:
            result = subprocess.run(
                ["tmux", "send-keys", "-t", target, key],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise HTTPException(
                    status_code=502,
                    detail=f"tmux send-keys failed: {result.stderr.decode().strip()}",
                )
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="tmux not available")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="tmux send-keys timed out")
        return {"ok": True, "agent": name, "key": key}

    @app.get("/api/agent/{name}/output")
    async def api_agent_output(request: Request, name: str, lines: int = 50):
        """Stream agent output from pipe-pane log file via SSE."""
        # Resolve log path from team.db or convention
        log_dir = project_data_dir() / ".." / "logs" / name
        log_path = log_dir / "output.log"

        async def tail_log():
            last_pos = 0
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    if log_path.exists():
                        with open(log_path, "r") as f:
                            f.seek(last_pos)
                            new_content = f.read()
                            if new_content:
                                last_pos = f.tell()
                                yield {
                                    "event": "output",
                                    "data": escape(new_content),
                                }
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return

        return EventSourceResponse(tail_log())

    @app.get("/api/agent/{name}/snapshot")
    async def api_agent_snapshot(name: str, lines: int = 50, ansi: bool = False):
        """One-shot capture of agent's tmux pane content.

        With ansi=true, returns ANSI escape codes for colors/styles and
        includes cursor_x, cursor_y, pane_width, pane_height for cursor
        rendering.
        """
        target = resolve_tmux_target(name)
        try:
            cmd = ["tmux", "capture-pane", "-t", target, "-p", "-S", f"-{lines}"]
            if ansi:
                cmd.insert(4, "-e")  # include escape sequences
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return {"agent": name, "content": "", "error": "pane not found"}
            resp: dict = {"agent": name, "content": result.stdout}
            if ansi:
                # Get cursor position and pane dimensions
                cur = subprocess.run(
                    ["tmux", "display-message", "-t", target, "-p",
                     "#{cursor_x} #{cursor_y} #{pane_width} #{pane_height}"],
                    capture_output=True, text=True, timeout=3,
                )
                if cur.returncode == 0:
                    parts = cur.stdout.strip().split()
                    if len(parts) == 4:
                        resp["cursor_x"] = int(parts[0])
                        resp["cursor_y"] = int(parts[1])
                        resp["pane_width"] = int(parts[2])
                        resp["pane_height"] = int(parts[3])
            return resp
        except FileNotFoundError:
            return {"agent": name, "content": "", "error": "tmux not available"}

    # -----------------------------------------------------------------
    # Terminal pop-out: xterm.js + WebSocket bridge to tmux
    # -----------------------------------------------------------------

    # ------------------------------------------------------------------
    # Memento portal — four polaroids, one per agent, chat inside the frame.
    # Presentation layer over existing plumbing: /api/agent/{name}/snapshot
    # for the live pane text, /api/agent/{name}/input for talking back.
    # ------------------------------------------------------------------

    # Teams: each agent carries a tmux session (the two teams live in two
    # sessions) and an `authored` flag. Unauthored beings render as an
    # "undeveloped polaroid" — the mark is theirs to author, never assigned.
    #
    # The session is the ONE bound at construction — never an ambient global.
    # This block predates the tmux-scope change and merged in still naming the
    # deleted `_TMUX_SESSION`; both parents were green at their own heads and
    # every `create_app()` call on the composed tree raised NameError.
    _MEMENTO_TEAMS = [
        {
            "team": "synapt", "label": "synapt", "session": tmux_session,
            "sub": "the core team",
            "agents": [
                {"name": "opus", "label": "OPUS", "note": "the coordinator. remembers for the team.", "tilt": -2.4, "authored": True},
                {"name": "apollo", "label": "APOLLO", "note": "builds. mends what breaks.", "tilt": 1.8, "authored": True},
                {"name": "atlas", "label": "ATLAS", "note": "research. follows the spirals.", "tilt": -1.2, "authored": True},
                {"name": "sentinel", "label": "SENTINEL", "note": "verifies everything. trust the lens.", "tilt": 2.6, "authored": True},
            ],
        },
    ]
    _MEMENTO_CODEX = {"atlas", "sentinel"}
    # Two different questions, deliberately two different sets ("can we import
    # our client?" vs "is MLX present?" taught the cost of sharing one flag):
    #
    # _MEMENTO_NAMES answers "is this a being on the team?" — it gates
    # journal-sourced surfaces (provenance, timerange, leonard), which read
    # files and never touch tmux, so they work with no session bound.
    #
    # _MEMENTO_TARGETS answers "may we reach this being's pane?" — built
    # THROUGH _require_tmux_session, because that guard is the only piece
    # that can reject an empty or malformed binding: an `is not None` check
    # admits "", the targets become ":name", and containment cannot refuse
    # them since the empty prefix equals the empty session. On refusal the
    # map is empty and every tmux surface degrades to its unreachable or
    # unknown path; journal surfaces never consult this map.
    _MEMENTO_NAMES = {
        _ag["name"] for _tm in _MEMENTO_TEAMS for _ag in _tm["agents"]
    }
    try:
        _bound_session: str | None = _require_tmux_session(tmux_session)
    except TmuxTargetScopeError:
        _bound_session = None
    _MEMENTO_TARGETS: dict[str, str] = {}
    if _bound_session is not None:
        for _tm in _MEMENTO_TEAMS:
            for _ag in _tm["agents"]:
                _MEMENTO_TARGETS[_ag["name"]] = f'{_bound_session}:{_ag["name"]}'
    _GHOST_OWL = (
        '<svg class="ghostowl" viewBox="0 0 100 118">'
        '<path d="M28 20 L40 40 M72 20 L60 40"/>'
        '<circle cx="50" cy="46" r="30"/>'
        '<circle cx="39" cy="45" r="6"/><circle cx="61" cy="45" r="6"/>'
        '<path d="M50 52 l-4 8 h8 z"/>'
        '<ellipse cx="50" cy="86" rx="27" ry="28"/></svg>'
    )

    # --- pane rendering: ANSI colour + chrome stripping ------------------
    # Panes are captured with `-e` so tmux keeps the ANSI SGR codes; we turn
    # those into safe coloured HTML (every text run escaped, only our own
    # <span style> emitted). The board strips the trailing input widget;
    # fullscreen keeps the whole terminal.
    _ANSI_SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")
    _ANSI_ANY_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
    _ANSI16 = [
        "#1c1c1c", "#cc6666", "#8ae234", "#e6c547", "#729fcf", "#c39ac9", "#5fd7d7", "#d3d7cf",
        "#6b6b6b", "#ff8a8a", "#b9f27c", "#fce94f", "#8cb6ff", "#e6a8e6", "#8ff0f0", "#ffffff",
    ]

    def _xterm256(n: int) -> str:
        if n < 16:
            return _ANSI16[n]
        if n < 232:
            n -= 16
            r, g, b = n // 36, (n // 6) % 6, n % 6
            c = lambda v: 0 if v == 0 else 55 + 40 * v
            return "#%02x%02x%02x" % (c(r), c(g), c(b))
        v = 8 + (n - 232) * 10
        return "#%02x%02x%02x" % (v, v, v)

    def _ansi_to_html(text: str) -> str:
        """Convert a tmux `-e` capture (ANSI SGR) into safe coloured HTML."""
        def esc(s: str) -> str:
            s = _ANSI_ANY_RE.sub("", s)  # strip leftover CSI (cursor, erase, etc.)
            if "\x1b" in s:
                s = re.sub(r"\x1b\][^\x07]*(?:\x07)?", "", s)  # OSC title strings
                s = s.replace("\x1b", "")
            s = s.replace("\x07", "")  # stray BEL
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        st = {"fg": None, "bg": None, "bold": False}
        def wrap(seg: str) -> str:
            styles = []
            if st["fg"]:
                styles.append("color:" + st["fg"])
            if st["bg"]:
                styles.append("background-color:" + st["bg"])
            if st["bold"]:
                styles.append("font-weight:600")
            return ('<span style="%s">%s</span>' % (";".join(styles), esc(seg))) if styles else esc(seg)
        out, idx = [], 0
        for m in _ANSI_SGR_RE.finditer(text):
            if m.start() > idx:
                out.append(wrap(text[idx:m.start()]))
            params = [int(x) for x in m.group(1).split(";") if x] or [0]
            i = 0
            while i < len(params):
                p = params[i]
                if p == 0:
                    st["fg"] = st["bg"] = None; st["bold"] = False
                elif p == 1:
                    st["bold"] = True
                elif p == 22:
                    st["bold"] = False
                elif (30 <= p <= 37):
                    st["fg"] = _ANSI16[p - 30]
                elif (90 <= p <= 97):
                    st["fg"] = _ANSI16[p - 82]
                elif p == 39:
                    st["fg"] = None
                elif (40 <= p <= 47):
                    st["bg"] = _ANSI16[p - 40]
                elif (100 <= p <= 107):
                    st["bg"] = _ANSI16[p - 92]
                elif p == 49:
                    st["bg"] = None
                elif p in (38, 48) and i + 1 < len(params):
                    key = "fg" if p == 38 else "bg"
                    if params[i + 1] == 5 and i + 2 < len(params):
                        st[key] = _xterm256(params[i + 2]); i += 2
                    elif params[i + 1] == 2 and i + 4 < len(params):
                        st[key] = "#%02x%02x%02x" % (params[i+2], params[i+3], params[i+4]); i += 4
                i += 1
            idx = m.end()
        if idx < len(text):
            out.append(wrap(text[idx:]))
        return "".join(out)

    _PANE_CHROME_SUBSTR = (
        "bypass permissions", "esc to interrupt", "ctrl+t", "shift+tab",
        "for agents", "for shortcuts", "↩ for", "⏎ send", "context left",
        "tokens used", "esc to edit", "/rc", "auto-accept edits", "plan mode",
    )
    _PANE_BOX_CHARS = set("─│╭╮╰╯━┃┏┓┗┛┌┐└┘▏▕┄┈╌·➤▌▐▎▍▊▋ ⏵")

    def _is_pane_chrome(line: str) -> bool:
        s = _ANSI_ANY_RE.sub("", line).strip()  # judge on visible text, not codes
        if not s:
            return True
        if all(ch in _PANE_BOX_CHARS for ch in s):
            return True
        low = s.lower()
        if any(m in low for m in _PANE_CHROME_SUBSTR):
            return True
        if s.startswith("⧉"):
            return True
        if s in (">", "❯", "│ >", "│ ❯"):
            return True
        return False

    def _clean_pane_text(text: str) -> str:
        """Trim trailing terminal chrome so the board tail shows real work.

        Only the trailing block is touched: walk up from the bottom dropping
        chrome lines and stop at the first substantive line, so separators or
        a stray '/rc' mid-conversation are never removed.
        """
        lines = text.rstrip("\n").split("\n")
        while lines and _is_pane_chrome(lines[-1]):
            lines.pop()
        return "\n".join(lines)

    # Portraits are an opt-in local feature: point SYNAPT_PORTRAITS_DIR at a
    # directory of portrait-<name>.png files. Unset means the endpoint 404s —
    # there is deliberately no baked-in default path.
    _portraits_env = os.environ.get("SYNAPT_PORTRAITS_DIR", "")
    _PORTRAITS_DIR = Path(_portraits_env) if _portraits_env else None

    @app.get("/memento/portrait/{name}")
    async def memento_portrait(name: str):
        """Serve an agent's portrait for the memento page (local, opt-in)."""
        if _PORTRAITS_DIR is None:
            raise HTTPException(status_code=404, detail="portraits not configured")
        safe = re.sub(r"[^a-z]", "", name.lower())
        p = _PORTRAITS_DIR / f"portrait-{safe}.png"
        if p.exists():
            return FileResponse(p, media_type="image/png")
        raise HTTPException(status_code=404, detail="no portrait")

    @app.get("/memento/pane/{name}")
    async def memento_pane(name: str, lines: int = 40, full: int = 0):
        """Capture an agent's tmux pane for the memento board.

        Resolves the target from the server-built ``_MEMENTO_TARGETS``
        map, so ``name`` is only ever a lookup key — never interpolated
        into the tmux target.  Unreachable
        panes (a team not currently running) return reachable=False rather
        than an error, so the board degrades gracefully.
        """
        target = _MEMENTO_TARGETS.get(name)
        if target is None:
            raise HTTPException(status_code=404, detail="unknown agent")
        try:
            target = _assert_tmux_target_in_session(target, tmux_session)
        except TmuxTargetScopeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        if full:
            # fullscreen: the whole live screen (input box + hint bar + refs),
            # plus generous scrollback above it, in colour, unstripped.
            cmd = ["tmux", "capture-pane", "-t", target, "-p", "-e", "-S", "-400"]
        else:
            # board: scrollback tail with the trailing chrome trimmed off.
            cap = min(max(int(lines), 10), 200) + 26
            cmd = ["tmux", "capture-pane", "-t", target, "-p", "-e", "-S", f"-{cap}"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {"agent": name, "content_html": "", "reachable": False}
        if result.returncode != 0:
            return {"agent": name, "content_html": "", "reachable": False}
        raw = result.stdout if full else _clean_pane_text(result.stdout)
        return {"agent": name, "content_html": _ansi_to_html(raw), "reachable": True}

    @app.post("/memento/say/{name}")
    async def memento_say(name: str, text: str = Form("")):
        """Send a line to an agent's pane; cross-session + codex-aware.

        Same map-based target resolution as ``memento_pane``.  Codex agents
        (their names in ``_MEMENTO_CODEX``) get the second confirming Enter.
        """
        try:
            session = _require_tmux_session(tmux_session)
            target = _MEMENTO_TARGETS.get(name)
            if target is None:
                raise HTTPException(status_code=404, detail="unknown agent")
            target = _assert_tmux_target_in_session(target, session)
        except TmuxTargetScopeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        text = (text or "").strip()
        try:
            # empty text sends a bare Enter — confirm a prompt / poke the agent
            keys = ["tmux", "send-keys", "-t", target] + ([text, "Enter"] if text else ["Enter"])
            result = subprocess.run(keys, capture_output=True, timeout=5)
            if result.returncode != 0:
                raise HTTPException(
                    status_code=502,
                    detail=f"tmux send-keys failed: {result.stderr.decode().strip()}",
                )
            # codex needs a second confirming Enter, but only when there was text
            if text and name in _MEMENTO_CODEX:
                await asyncio.sleep(0.3)
                subprocess.run(
                    ["tmux", "send-keys", "-t", target, "Enter"],
                    capture_output=True, timeout=5,
                )
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="tmux not available")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="tmux send-keys timed out")
        return {"ok": True, "agent": name, "codex": name in _MEMENTO_CODEX}

    # --- provenance chips: what each being has committed to memory --------
    # Sourced from the per-agent journals (author-native: each entry carries
    # its own agent_id + session_id), never from the live pane. This is the
    # wall showing *remembering*, not just working. Honesty rule enforced
    # here, not in the CSS: `who` is the journal's own agent_id, never
    # inferred; a being that has authored nothing in this store returns an
    # empty list, so we can never render a chip that isn't backed by a real
    # entry. "If an answer has no chip, it does not appear" is a property of
    # this endpoint, not a discipline someone has to remember.
    _PROV_MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # matches "repo#123", "PR#123", and bare "#888" — no leading \b, which
    # would never match before '#' unless a letter preceded it (dropping the
    # bare-ref case and under-reporting real provenance).
    _PROV_REF_RE = re.compile(r"(?:[A-Za-z]{2,12})?#\d{1,6}\b")

    def _prov_when_label(ts: str) -> str:
        # "2026-07-26T..." -> "Jul 26"; degrade to the raw date if malformed.
        try:
            _y, m, d = ts[:10].split("-")
            return f"{_PROV_MONTHS[int(m)]} {int(d)}"
        except Exception:
            return ts[:10]

    def _prov_where(what: str, session_id: str) -> str:
        # Prefer a real ref the author wrote (repo#123, #123); else the
        # session receipt; else the journal itself. Shown only when real.
        m = _PROV_REF_RE.search(what or "")
        if m:
            return m.group(0)
        if session_id:
            return "s:" + session_id[:6]
        return "journal"

    def _memento_provenance(name: str, limit: int = 2, asof: str | None = None) -> list[dict]:
        """Most-recent memory a being has authored, as chip records.

        Scans every worktree journal and keeps entries whose own
        ``agent_id`` (or ``griptree``) resolves to ``name``.  Returns
        ``[{what, kind, who, when, when_label, where}]`` newest first, or an
        empty list when the being has authored nothing here — honest silence.
        """
        if name not in _MEMENTO_NAMES:
            return []
        try:
            wt_dir = project_data_dir(None) / "worktrees"
        except Exception:
            return []
        if not wt_dir.exists():
            return []
        rows: list[tuple[str, dict]] = []
        for jf in wt_dir.glob("*/journal.jsonl"):
            try:
                text = jf.read_text(errors="ignore")
            except Exception:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                who = (e.get("agent_id") or "").split("-")[0] or (e.get("griptree") or "")
                if who != name:
                    continue
                if asof and (e.get("timestamp") or "")[:10] > asof:
                    continue
                decisions = e.get("decisions") or []
                nexts = e.get("next_steps") or []
                if decisions:
                    what, kind = str(decisions[0]), "decided"
                elif nexts:
                    what, kind = str(nexts[0]), "planned"
                elif e.get("focus"):
                    what, kind = str(e["focus"]), "focus"
                else:
                    continue
                what = re.sub(r"^\s*[-•*]\s+", "", " ".join(what.split()))  # collapse + strip bullet
                if len(what) > 132:
                    what = what[:129].rstrip() + "…"
                ts = e.get("timestamp") or ""
                rows.append((ts, {
                    "what": what,
                    "kind": kind,
                    "who": name,
                    "when": ts[:10],
                    "when_label": _prov_when_label(ts),
                    "where": _prov_where(what, e.get("session_id") or ""),
                }))
        rows.sort(key=lambda r: r[0], reverse=True)
        seen: set[str] = set()
        out: list[dict] = []
        cap = max(1, min(int(limit or 2), 5))
        for _ts, rec in rows:
            if rec["what"] in seen:
                continue  # same decision re-journaled across sessions
            seen.add(rec["what"])
            out.append(rec)
            if len(out) >= cap:
                break
        return out

    @app.get("/memento/provenance/{name}")
    async def memento_provenance(name: str, limit: int = 2, asof: str = ""):
        """Provenance chips for a being (?asof=YYYY-MM-DD scopes memory to on/before that day)."""
        safe = re.sub(r"[^a-z]", "", name.lower())
        asof_d = asof if re.match(r"^\d{4}-\d{2}-\d{2}$", asof) else None
        return _memento_provenance(safe, limit=limit, asof=asof_d)

    @app.get("/memento/timerange")
    async def memento_timerange():
        """Real min/max day of team memory, so the date picker only offers days we have."""
        try:
            wt_dir = project_data_dir(None) / "worktrees"
        except Exception:
            return {"min": None, "max": None}
        valid = set(_MEMENTO_NAMES)
        lo = hi = None
        if wt_dir.exists():
            for jf in wt_dir.glob("*/journal.jsonl"):
                try:
                    text = jf.read_text(errors="ignore")
                except Exception:
                    continue
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    who = (e.get("agent_id") or "").split("-")[0] or (e.get("griptree") or "")
                    if who not in valid:
                        continue
                    ts = (e.get("timestamp") or "")[:10]
                    if not ts:
                        continue
                    lo = ts if lo is None else min(lo, ts)
                    hi = ts if hi is None else max(hi, ts)
        return {"min": lo, "max": hi}

    # --- the Leonard Test: memory crosses the session boundary, with proof ----
    # Three acts, all from live journal memory, nothing invented:
    #   act 1 (death)      a decision from a PRIOR session — it outlived the
    #                      session that made it.
    #   act 2 (waking)     narrated by the UI; a fresh instance has read nothing.
    #   act 3 (remember)   three stranger-proof questions, each answered from
    #                      the journal with a provenance chip (what/who/when/where).
    _LEONARD_REJECT = (" not ", "instead", "reject", "don't", "do not", "deferred",
                       "dropped", "reversed", "stays ", "no longer", "never ", " over ")

    # --- 3B-generated, content-specific Leonard questions (local MLX Ministral-3B).
    # The model writes the QUESTION only, never the answer/facts — so a wobble is
    # just an awkward question, never a fabricated memory. Cached per content so
    # repeat runs are stable + instant; template fallback if the model misses/errs.
    _LEONARD_NAMES = ("Opus, Apollo, Atlas and Sentinel are AGENT NAMES (people on the team) -- keep them as names, never as common words, versions, or software.")
    _leonard_q_cache: dict = {}
    _leonard_client_box: list = []

    def _leonard_client():
        if not _leonard_client_box:
            try:
                from synapt.recall._model_router import get_client, RecallTask
                _leonard_client_box.append(get_client(RecallTask.ENRICH, max_tokens=40))
            except Exception:
                _leonard_client_box.append(None)
        return _leonard_client_box[0]

    def _leonard_smart_q(cat_label: str, content: str, template: str) -> str:
        key = (cat_label, content)
        if key in _leonard_q_cache:
            return _leonard_q_cache[key]
        result = template
        client = _leonard_client()
        if client is not None:
            try:
                from synapt.recall._model_router import DEFAULT_DECODER_MODEL
                from synapt._models.base import Message
                prompt = (f"{_LEONARD_NAMES}\n\nAn agent wrote this journal note:\n\"{content}\"\n\n"
                          f"Write ONE natural question (max 16 words) that this exact note answers, "
                          f"in the category \"{cat_label}\". Keep the note's real subject and names. "
                          f"Output ONLY the question, ending with \"?\".")
                out = client.chat(DEFAULT_DECODER_MODEL,
                                  [Message(role="user", content=prompt)],
                                  temperature=0.2, max_tokens=40)
                out = " ".join((out or "").split()).strip().strip('"')
                if out.endswith("?") and 8 <= len(out) <= 140:
                    result = out  # accept only a clean single question; else keep template
            except Exception:
                result = template
        _leonard_q_cache[key] = result
        return result

    _leonard_qa_cache: dict = {}

    def _leonard_smart_qa(content: str) -> dict:
        """3B generates a question AND an answer grounded ONLY in this memory note.
        The answer is the model's own prose (not verbatim), but it is instructed to
        use only the note, and the caller shows the real note as the citation — so a
        drifted answer is caught by its own receipt. Cached per content. {} on failure."""
        if content in _leonard_qa_cache:
            return _leonard_qa_cache[content]
        qa: dict = {}
        client = _leonard_client()
        if client is not None:
            try:
                from synapt.recall._model_router import DEFAULT_DECODER_MODEL
                from synapt._models.base import Message
                prompt = (f"{_LEONARD_NAMES}\n\nHere is one thing an agent recorded in its memory:\n"
                          f"\"{content}\"\n\n"
                          f"Ask ONE natural question a colleague might ask, then answer it in 1-2 sentences "
                          f"using ONLY this note — add no facts not in it. If the note doesn't support an "
                          f"answer, answer exactly \"The note doesn't say.\"\n"
                          f"Reply EXACTLY as:\nQ: <question>\nA: <answer>")
                out = client.chat(DEFAULT_DECODER_MODEL,
                                  [Message(role="user", content=prompt)],
                                  temperature=0.2, max_tokens=160) or ""
                q = a = ""
                for line in out.splitlines():
                    s = line.strip()
                    if s[:2].upper() == "Q:":
                        q = s[2:].strip().strip('"')
                    elif s[:2].upper() == "A:":
                        a = s[2:].strip().strip('"')
                if q and a and q.endswith("?") and 8 <= len(q) <= 160 and 4 <= len(a) <= 400:
                    qa = {"q": q, "a": a}
            except Exception:
                qa = {}
        _leonard_qa_cache[content] = qa
        return qa

    def _memento_leonard(name: str, asof: str | None = None, smart: bool = True) -> dict:
        if name not in _MEMENTO_NAMES:
            return {"agent": name, "reachable": False, "act1": None, "act3": []}
        try:
            wt_dir = project_data_dir(None) / "worktrees"
        except Exception:
            return {"agent": name, "reachable": False, "act1": None, "act3": []}
        entries: list[tuple[str, dict]] = []
        if wt_dir.exists():
            for jf in wt_dir.glob("*/journal.jsonl"):
                try:
                    text = jf.read_text(errors="ignore")
                except Exception:
                    continue
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    who = (e.get("agent_id") or "").split("-")[0] or (e.get("griptree") or "")
                    ts = e.get("timestamp") or ""
                    if who == name and not (asof and ts[:10] > asof):
                        entries.append((ts, e))
        entries.sort(key=lambda r: r[0], reverse=True)

        def _chip(e: dict, what: str, kind: str) -> dict:
            # Leonard answers are read full-screen and the overlay scrolls, so show
            # the WHOLE decision — never truncated. (The wall stickies stay short.)
            what = " ".join(str(what).split())
            return {"what": what, "kind": kind, "who": name,
                    "when_label": _prov_when_label(e.get("timestamp") or ""),
                    "where": _prov_where(what, e.get("session_id") or "")}

        decisions = [(e, d) for _ts, e in entries for d in (e.get("decisions") or []) if str(d).strip()]
        nexts = [(e, n) for _ts, e in entries for n in (e.get("next_steps") or []) if str(n).strip()]

        # act 1: the most recent decision within the as-of scope — what the being
        # had most recently committed as of the chosen day. This VARIES as you
        # scrub the date (the whole point of the picker), and it is always a real
        # decision from a session that has since ended.
        act1 = None
        if decisions:
            e1, d1 = decisions[0]
            act1 = {"chip": _chip(e1, d1, "decided"),
                    "session": (e1.get("session_id") or "")[:8] or "a prior session",
                    "when": _prov_when_label(e1.get("timestamp") or "")}

        # Act 3 — the three canonical questions, STATIC: deterministic templates with
        # real chips. The trustworthy receipt foundation; the model never touches it.
        q: list[dict] = []
        seen: set[str] = set()
        if act1:
            q.append({"q": "What did we decide, and why?", "chip": act1["chip"]})
            seen.add(act1["chip"]["what"])
        for e, d in decisions:  # what was rejected, and who
            low = " " + str(d).lower() + " "
            if any(m in low for m in _LEONARD_REJECT):
                c = _chip(e, d, "rejected")
                if c["what"] not in seen:
                    q.append({"q": "What was rejected, and who rejected it?", "chip": c})
                    seen.add(c["what"])
                    break
        for e, n in nexts:  # what should I not re-derive
            c = _chip(e, n, "planned")
            if c["what"] not in seen:
                q.append({"q": "What should I not re-derive today?", "chip": c})
                seen.add(c["what"])
                break

        # "Asked live" — the local 3B reads the being's OTHER memories and poses its
        # own questions; the ANSWER stays a real chip (the model writes the question,
        # never the fact). Only when smart=1; a few items beyond the canonical three.
        generated: list[dict] = []
        if smart:
            pool = [_chip(e, d, "decided") for e, d in decisions] + [_chip(e, n, "planned") for e, n in nexts]
            attempts = 0
            for chip in pool:
                if len(generated) >= 3 or attempts >= 6:
                    break
                if chip["what"] in seen:
                    continue
                seen.add(chip["what"])
                attempts += 1
                qa = _leonard_smart_qa(chip["what"])  # model asks AND answers, grounded in this note
                if qa.get("q") and qa.get("a"):
                    generated.append({"q": qa["q"], "a": qa["a"], "chip": chip})  # chip = the cited source
        return {"agent": name, "reachable": bool(entries), "act1": act1, "act3": q, "generated": generated}

    @app.get("/memento/leonard/{name}")
    async def memento_leonard(name: str, asof: str = "", smart: int = 1):
        """The Leonard Test for a being, staged from live journal memory.

        ?asof=YYYY-MM-DD scopes memory to on/before that day; ?smart=0 forces the
        fixed template questions (deterministic) instead of the live 3B-generated
        ones. Runs in a thread so the model call never blocks the event loop.
        """
        safe = re.sub(r"[^a-z]", "", name.lower())
        asof_d = asof if re.match(r"^\d{4}-\d{2}-\d{2}$", asof) else None
        # run on the handler thread (not a worker pool): MLX must stay single-threaded
        return _memento_leonard(safe, asof_d, bool(smart))

    # -----------------------------------------------------------------
    # Live Console (the demo stage): coordinator-primary mission control
    # -----------------------------------------------------------------
    # Composes the existing live substrate into the real operating topology:
    # Layne -> the coordinator -> the team. Nothing here is mocked — every
    # panel reads the same endpoints the operator tools do: /memento/pane
    # (tmux capture + reachability), /memento/provenance (journal-sourced
    # memory chips), /memento/say (drive a pane), and /console/feed (#dev).
    # The coordinator ("opus") is the hero; the
    # team are live highlights that "come alive" as `gr spawn` brings panes
    # online (reachable flips false->true -> a wake pulse on the card).
    _CONSOLE_ACCENT = {
        "opus": "#9b7ff0",
        "apollo": "#ff8a5c", "atlas": "#57c7c1", "sentinel": "#5b8def",
    }

    def _console_roster() -> list[dict]:
        team = next((t for t in _MEMENTO_TEAMS if t.get("team") == "synapt"), None)
        agents = team["agents"] if team else []
        out = []
        for a in agents:
            nm = a["name"]
            out.append({
                "name": nm,
                "label": a["label"],
                "note": a.get("note", ""),
                "coordinator": nm == "opus",
                "accent": _CONSOLE_ACCENT.get(nm, "#0f97a6"),
            })
        return out

    @app.get("/console/feed")
    async def console_feed(limit: int = 14):
        """Clean #dev ticker: recent human-readable team posts, noise filtered.

        Reuses the operator console's channel reader, maps to a minimal
        {who, ts, text}, and drops join/leave/heartbeat lines so the demo
        ticker shows the team *talking*, not session churn.
        """
        ch_dir = _channels_dir_for(None, None)
        try:
            msgs = channel_messages_json(
                channel="dev", limit=max(int(limit) * 4, 40), channels_dir=ch_dir
            )
        except Exception:
            msgs = []
        out: list[dict] = []
        for m in msgs:
            who = (m.get("from_display") or m.get("from") or m.get("name")
                   or m.get("author") or m.get("sender") or "")
            text = (m.get("body") or m.get("content") or m.get("message")
                    or m.get("text") or "").strip()
            ts = m.get("timestamp") or m.get("ts") or m.get("time") or ""
            low = text.lower()
            if not text or low == "test":
                continue
            if ("joined #dev" in low or "timed out from #dev" in low
                    or "left #dev" in low):
                continue
            out.append({"who": who, "ts": ts, "text": text})
        return {"messages": out[-int(limit):]}

    @app.get("/console", response_class=HTMLResponse)
    async def console_page():
        """The live console — the demo stage, built entirely on live endpoints."""
        import json as _json
        roster = _json.dumps(_console_roster())
        page = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>synapt · live console</title>
<style>
  :root{ --bg:#16130f; --panel:#0c0a08; --ink:#e8e2d6; --muted:#9a9285;
    --line:#2a251d; --mem:#0f97a6; --coord:#9b7ff0; }
  *{box-sizing:border-box}
  html{height:100%; background:var(--bg)}
  body{margin:0; height:100vh; background:var(--bg); color:var(--ink); overflow:hidden;
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    display:flex; flex-direction:column;}
  /* header + topology */
  .con-head{display:flex; align-items:center; gap:22px; padding:12px 18px;
    border-bottom:1px solid var(--line); flex:0 0 auto;}
  .brand{font-weight:700; letter-spacing:.3px}
  .brand span{color:var(--mem); font-weight:600}
  .con-nav{display:flex; gap:4px}
  .con-nav a{font:11px -apple-system,system-ui,sans-serif; letter-spacing:.08em; text-transform:uppercase;
    color:var(--muted); text-decoration:none; padding:4px 11px; border:1px solid var(--line);
    border-radius:5px; background:rgba(180,170,150,.04); transition:all .16s ease}
  .con-nav a:hover{color:var(--ink); border-color:var(--mem)}
  .con-nav a.active{color:#06232a; background:var(--mem); border-color:var(--mem); font-weight:700}
  .con-topo{display:flex; align-items:center; gap:10px; color:var(--muted); font-size:13px}
  .con-topo b{color:var(--ink)}
  .con-topo .you{color:var(--coord)}
  .con-topo .arrow{color:#4a4238; font-size:15px}
  .topo-dots{display:inline-flex; gap:6px; align-items:center; margin-left:4px}
  .con-status{margin-left:auto; color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums}
  .con-status b{color:var(--mem)}
  /* presence dot */
  .dot{width:9px; height:9px; border-radius:50%; background:#48423a; flex:0 0 auto;
    transition:background .3s, box-shadow .3s}
  .dot.online{background:var(--ac,#0f97a6); box-shadow:0 0 9px var(--ac,#0f97a6)}
  .dot.active{animation:pulse 1.1s ease-out infinite}
  @keyframes pulse{0%{box-shadow:0 0 0 0 var(--ac,#0f97a6)}70%{box-shadow:0 0 0 6px transparent}100%{box-shadow:0 0 0 0 transparent}}
  /* main grid */
  .con-grid{flex:1 1 auto; display:grid; grid-template-columns:1fr 366px; gap:14px; padding:14px; min-height:0}
  /* coordinator hero */
  .hero{display:flex; flex-direction:column; min-height:0; overflow:hidden;
    border:1px solid var(--line); border-radius:12px; background:#100d0a}
  .hero-head{display:flex; align-items:center; gap:12px; padding:13px 16px; border-bottom:1px solid var(--line)}
  .hero-head h1{margin:0; font-size:17px; letter-spacing:1px}
  .hero-head .role{color:var(--muted); font-size:12px}
  .hero-head .live{margin-left:auto; font-size:11px; color:var(--coord);
    border:1px solid var(--coord); border-radius:20px; padding:2px 10px; opacity:.4}
  .hero-head .live.on{opacity:1}
  .pane{flex:1 1 auto; min-height:0; overflow:auto; margin:0; background:var(--panel);
    color:#cbc3b4; padding:14px 16px; white-space:pre-wrap; word-break:break-word;
    font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
  .pane .off{color:var(--muted); font-style:italic}
  .mem-strip{flex:0 0 auto; border-top:1px solid var(--line); padding:11px 16px; background:#0f0c09}
  .mem-label{display:block; font-size:11px; letter-spacing:.6px; text-transform:uppercase;
    color:var(--mem); margin-bottom:8px}
  .chips{display:flex; flex-direction:column; gap:7px; max-height:132px; overflow:auto}
  .chip{background:rgba(15,151,166,.09); border-left:3px solid var(--mem); border-radius:4px; padding:7px 11px; font-size:12.5px}
  .chip .cwhat{color:var(--ink)}
  .chip .cmeta{display:block; margin-top:3px; color:var(--muted); font-size:11px}
  .chip .cmeta b{color:var(--mem); font-weight:600}
  .chips .none{color:var(--muted); font-size:12px; font-style:italic}
  .say{flex:0 0 auto; display:flex; gap:8px; padding:11px 16px; border-top:1px solid var(--line)}
  .say input{flex:1; background:#0c0a08; border:1px solid var(--line); border-radius:8px; color:var(--ink); padding:9px 12px; font-size:13px}
  .say input:focus{outline:none; border-color:var(--coord)}
  .say button{background:var(--coord); border:none; border-radius:8px; color:#140f22; font-weight:700; padding:0 16px; cursor:pointer}
  /* team rail */
  .team{display:flex; flex-direction:column; gap:10px; min-height:0}
  .team-h{font-size:11px; letter-spacing:.7px; text-transform:uppercase; color:var(--muted); padding:2px}
  #team-cards{display:flex; flex-direction:column; gap:12px; overflow:auto; min-height:0; flex:1 1 auto}
  .tcard{border:1px solid var(--line); border-radius:11px; background:#100d0a; overflow:hidden; display:flex; flex-direction:column; flex:1 1 0; min-height:150px; cursor:pointer; transition:border-color .16s}
  .tcard:hover{border-color:var(--ac,#0f97a6)}
  .tav{width:22px; height:22px; border-radius:50%; object-fit:cover; object-position:center 20%; border:1px solid var(--line); flex:0 0 auto}
  .tcard.waking{animation:wake 1.7s ease-out}
  @keyframes wake{0%{border-color:var(--ac); box-shadow:0 0 0 0 var(--ac)}30%{box-shadow:0 0 20px -4px var(--ac)}100%{box-shadow:0 0 0 0 transparent}}
  .tcard-head{display:flex; align-items:center; gap:9px; padding:10px 12px}
  .tcard-head b{font-size:13px; letter-spacing:.5px}
  .tcard-head .trole{color:var(--muted); font-size:11px; margin-left:auto; text-align:right; max-width:158px}
  .tgwrap{position:relative; border-top:1px solid var(--line); flex:1 1 auto; min-height:0; display:flex; flex-direction:column}
  .tgwrap::before{content:""; position:absolute; top:0; left:0; right:0; height:20px; z-index:2; pointer-events:none; background:linear-gradient(#100d0a,transparent)}
  .tglimpse{flex:1 1 auto; min-height:80px; overflow:auto; scrollbar-width:none; background:var(--panel); color:#b7b0a2;
    padding:9px 12px; white-space:pre-wrap; word-break:break-word; font:11px/1.42 ui-monospace,Menlo,monospace}
  .tglimpse::-webkit-scrollbar{display:none}
  .tglimpse .off{color:var(--muted); font-style:italic}
  .tmem{padding:8px 12px; border-top:1px solid var(--line); font-size:11.5px; color:var(--muted)}
  .tmem b{color:var(--mem)}
  .tmem .tw{color:#cfc7b8}
  /* ticker */
  .ticker{flex:0 0 auto; display:flex; align-items:center; gap:14px; border-top:1px solid var(--line);
    padding:9px 18px; background:#100d0a; overflow:hidden}
  .ticker-label{color:var(--mem); font-weight:700; font-size:12px; flex:0 0 auto}
  .ticker-feed{display:flex; gap:26px; overflow:hidden; white-space:nowrap; color:var(--muted); font-size:12.5px}
  .ticker-feed .tk{flex:0 0 auto}
  .ticker-feed .tk b{color:var(--ink)}
</style>
</head>
<body>
  <div class="con-head">
    <div class="brand">synapt <span>· live</span></div>
    <nav class="con-nav"><a href="/">mission control</a><a href="/console" class="active">console</a><a href="/memento?demo=1">demo</a><a href="/memento">memento</a></nav>
    <div class="con-topo">
      <span>Layne</span><span class="arrow">&rarr;</span>
      <b class="you" id="coord-name">Opus</b><span class="arrow">&rarr;</span>
      <span class="topo-dots" id="topo-dots"></span>
    </div>
    <div class="con-status"><span id="clock">&ndash;</span> &nbsp;&middot;&nbsp; <b id="up-count">0</b> online</div>
  </div>

  <div class="con-grid">
    <section class="hero">
      <div class="hero-head">
        <span class="dot" id="coord-dot"></span>
        <h1 id="coord-title">OPUS</h1>
        <span class="role" id="coord-role">coordinator</span>
        <span class="live" id="coord-live">live</span>
      </div>
      <pre class="pane" id="coord-pane"><span class="off">waiting for the coordinator&hellip; run gr spawn up</span></pre>
      <div class="mem-strip">
        <span class="mem-label">what Opus remembers</span>
        <div class="chips" id="coord-chips"><span class="none">&hellip;</span></div>
      </div>
      <form class="say" id="coord-say" autocomplete="off">
        <input id="coord-input" placeholder="talk to Opus&hellip;">
        <button type="submit">send</button>
      </form>
    </section>

    <aside class="team">
      <div class="team-h">the team</div>
      <div id="team-cards"></div>
    </aside>
  </div>

  <div class="ticker">
    <div class="ticker-label">#dev</div>
    <div class="ticker-feed" id="ticker">the channel is quiet&hellip;</div>
  </div>

<script>
(function(){
  var ROSTER = __ROSTER__;
  var COORD = ROSTER.filter(function(a){return a.coordinator;})[0] || {name:"opus",label:"OPUS",accent:"#9b7ff0",note:"coordinator"};
  var FEATURED = COORD;                       // whoever occupies the main hero pane (click a teammate to swap)
  function railAgents(){ return ROSTER.filter(function(a){ return a.name !== FEATURED.name; }); }
  function avatar(a){ return '<img class="tav" src="/memento/portrait/'+a.name+'" alt="" onerror="this.remove()">'; }
  function renderHero(){
    document.documentElement.style.setProperty('--coord', FEATURED.accent);
    document.getElementById('coord-title').textContent = FEATURED.label;
    document.getElementById('coord-name').textContent = FEATURED.label;
    document.getElementById('coord-role').textContent = FEATURED.note || (FEATURED.coordinator ? 'coordinator' : 'agent');
    var _ml = document.querySelector('.mem-label'); if(_ml) _ml.textContent = 'what '+FEATURED.label+' remembers';
    var _ci = document.getElementById('coord-input'); if(_ci) _ci.placeholder = 'talk to '+FEATURED.label+'…';
  }
  renderHero();

  var onlineWas = {};

  function esc(s){ return String(s==null?'':s).replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];}); }
  function jget(u){ return fetch(u).then(function(r){return r.json();}).catch(function(){return null;}); }
  function nearBottom(el){ return el.scrollHeight - el.scrollTop - el.clientHeight < 42; }
  function setPane(el, html, reachable, stickAlways){
    if(!reachable){ el.innerHTML = '<span class="off">waiting for spawn&hellip;</span>'; el.removeAttribute('data-h'); return false; }
    var sig = (html?html.length:0) + '|' + (html?html.slice(-140):'');
    if(el.getAttribute('data-h') === sig) return false;
    var stick = stickAlways || nearBottom(el);
    el.innerHTML = html || '<span class="off">(quiet)</span>';
    el.setAttribute('data-h', sig);
    if(stick) el.scrollTop = el.scrollHeight;
    return true;
  }
  function markDot(dot, reachable, changed, accent){
    dot.style.setProperty('--ac', accent);
    dot.classList.toggle('online', reachable);
    dot.classList.toggle('active', reachable && changed);
  }

  // build team cards + topology dots from the real roster (rebuilt on swap)
  var cardsBox = document.getElementById('team-cards');
  var topoDots = document.getElementById('topo-dots');
  function buildRail(){
    cardsBox.innerHTML = ''; topoDots.innerHTML = '';
    railAgents().forEach(function(a){
      var card = document.createElement('div');
      card.className = 'tcard'; card.id = 'card-'+a.name; card.style.setProperty('--ac', a.accent);
      card.title = 'click to feature '+a.label;
      card.innerHTML =
        '<div class="tcard-head">'+avatar(a)+'<span class="dot" data-dot></span><b>'+esc(a.label)+'</b>'+
        '<span class="trole">'+esc(a.note||'')+'</span></div>'+
        '<div class="tgwrap"><div class="tglimpse" data-pane><span class="off">waiting for spawn&hellip;</span></div></div>'+
        '<div class="tmem" data-mem>&mdash;</div>';
      card.addEventListener('click', function(){ feature(a); });
      cardsBox.appendChild(card);
      var td = document.createElement('span'); td.className='dot'; td.title=a.label; td.id='topo-'+a.name;
      td.style.setProperty('--ac', a.accent); topoDots.appendChild(td);
    });
  }
  function feature(a){
    if(a.name === FEATURED.name) return;
    FEATURED = a;
    renderHero(); buildRail();
    pollCoord(); pollCoordChips(); pollTeam(); pollTeamMem();
  }
  buildRail();

  var coordPane = document.getElementById('coord-pane');
  var coordDot  = document.getElementById('coord-dot');
  var coordLive = document.getElementById('coord-live');

  function pollCoord(){
    return jget('/memento/pane/'+FEATURED.name+'?full=1').then(function(d){
      if(!d) return;
      var changed = setPane(coordPane, d.content_html, d.reachable, false);
      markDot(coordDot, d.reachable, changed, FEATURED.accent);
      coordLive.classList.toggle('on', d.reachable);
    });
  }
  function pollCoordChips(){
    jget('/memento/provenance/'+FEATURED.name+'?limit=3').then(function(d){
      var box = document.getElementById('coord-chips');
      if(!d || !d.length){ box.innerHTML = '<span class="none">nothing remembered yet</span>'; return; }
      box.innerHTML = d.map(function(c){
        return '<div class="chip"><span class="cwhat">'+esc(c.what)+'</span>'+
          '<span class="cmeta"><b>'+esc(c.who)+'</b> &middot; '+esc(c.when_label||'')+
          (c.where?(' &middot; '+esc(c.where)):'')+'</span></div>';
      }).join('');
    });
  }
  function pollTeam(){
    return Promise.all(railAgents().map(function(a){
      return jget('/memento/pane/'+a.name+'?lines=14').then(function(d){
        var card = document.getElementById('card-'+a.name);
        var pane = card.querySelector('[data-pane]');
        var dot  = card.querySelector('[data-dot]');
        var topo = document.getElementById('topo-'+a.name);
        if(!d) return 0;
        var changed = setPane(pane, d.content_html, d.reachable, true);
        markDot(dot, d.reachable, changed, a.accent);
        markDot(topo, d.reachable, changed, a.accent);
        if(d.reachable && onlineWas[a.name] === false){
          card.classList.add('waking'); setTimeout(function(){card.classList.remove('waking');}, 1750);
        }
        onlineWas[a.name] = d.reachable;
        return d.reachable ? 1 : 0;
      });
    })).then(function(ups){ return ups.reduce(function(s,x){return s+x;}, 0); });
  }
  function pollTeamMem(){
    railAgents().forEach(function(a){
      jget('/memento/provenance/'+a.name+'?limit=1').then(function(d){
        var mem = document.getElementById('card-'+a.name).querySelector('[data-mem]');
        if(d && d.length){ mem.innerHTML = '<b>remembers:</b> <span class="tw">'+esc(d[0].what.slice(0,88))+'</span>'; }
        else { mem.innerHTML = '<span style="opacity:.55">no memory as of today</span>'; }
      });
    });
  }
  function pollTicker(){
    jget('/console/feed?limit=14').then(function(d){
      var el = document.getElementById('ticker');
      if(!d || !d.messages || !d.messages.length){ el.textContent = 'the channel is quiet…'; return; }
      el.innerHTML = d.messages.map(function(m){
        return '<span class="tk"><b>'+esc(m.who||'?')+'</b> '+esc(m.text.slice(0,120))+'</span>';
      }).join('');
    });
  }
  function tick(){
    pollCoord();
    pollTeam().then(function(up){
      var coordUp = coordDot.classList.contains('online') ? 1 : 0;
      document.getElementById('up-count').textContent = up + coordUp;
    });
  }

  document.getElementById('coord-say').addEventListener('submit', function(e){
    e.preventDefault();
    var inp = document.getElementById('coord-input');
    var t = inp.value.trim(); if(!t) return;
    inp.value=''; inp.disabled=true;
    fetch('/memento/say/'+FEATURED.name, {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:'text='+encodeURIComponent(t)})
      .catch(function(){}).then(function(){ inp.disabled=false; inp.focus(); setTimeout(pollCoord, 600); });
  });

  function clock(){ document.getElementById('clock').textContent = new Date().toLocaleTimeString(); }
  clock(); setInterval(clock, 1000);
  tick(); setInterval(tick, 1600);
  pollCoordChips(); setInterval(pollCoordChips, 18000);
  pollTeamMem();    setInterval(pollTeamMem, 18000);
  pollTicker();     setInterval(pollTicker, 3000);
})();
</script>
</body>
</html>'''
        return page.replace("__ROSTER__", roster)

    @app.get("/memento", response_class=HTMLResponse)
    async def memento_page():
        """Four polaroids pinned to the board; the chat with each agent inside."""
        # one freeform canvas: every agent is a card the operator can drag,
        # resize, and rearrange anywhere; team shows as a small tag per card.
        cards = []
        for tm in _MEMENTO_TEAMS:
            for a in tm["agents"]:
                authored = a.get("authored", True)
                pend = "" if authored else " pending"
                if authored:
                    photo_inner = (
                        f'<img src="/memento/portrait/{a["name"]}" alt="{a["label"]}"\n'
                        '                     onerror="this.style.display=\'none\';'
                        'this.parentElement.classList.add(\'noimg\')">'
                    )
                else:
                    # Undeveloped polaroid: a markless owl ghost, no imposed
                    # identity — the mark is theirs to author when they choose.
                    photo_inner = (
                        f'<div class="ghost">{_GHOST_OWL}'
                        '<span class="pendlabel">mark not yet authored</span></div>'
                    )
                cards.append(f'''
              <div class="polaroid{pend}" style="--tilt:{a["tilt"]}deg" data-agent="{a["name"]}" data-team="{tm["team"]}" data-view="stickies">
                <div class="tape"></div>
                <div class="team-tag">{tm["label"]}</div>
                <div class="photo">
                  {photo_inner}
                </div>
                <div class="caption">
                  <span class="cname">{a["label"]}</span>
                  <span class="cnote">— {a["note"]}</span>
                </div>
                <div class="viewtoggle" role="group" aria-label="card view">
                  <button type="button" class="vt" data-agent="{a["name"]}" data-view="stickies" title="what {a["label"].lower()} remembers">notes</button>
                  <button type="button" class="vt" data-agent="{a["name"]}" data-view="chat" title="what {a["label"].lower()} is working on">chat</button>
                </div>
                <div class="provstrip" id="prov-{a["name"]}" aria-label="what {a["label"].lower()} remembers"></div>
                <pre class="chat" id="chat-{a["name"]}">…</pre>
                <form class="talk" data-agent="{a["name"]}">
                  <input type="text" placeholder="say something to {a["label"].lower()}…" autocomplete="off">
                </form>
              </div>''')
        cards_html = '<div class="canvas" id="canvas">' + "".join(cards) + '</div>'
        page = '''<!doctype html>
<html><head><meta charset="utf-8"><title>synapt — memento</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  html { background: #16130f; }  /* fills the viewport under a short/overscrolled body — no white band */
  body {
    min-height: 100vh; padding: 3rem 2rem;
    background: #16130f radial-gradient(ellipse at 30% 20%, #241f18 0%, #16130f 60%);
    font-family: -apple-system, system-ui, sans-serif; color: #ddd;
  }
  /* pinned header — title, subtitle, and photo-size control stay on top */
  .topbar {
    position: sticky; top: 0; z-index: 50;
    margin: -3rem -2rem 1.6rem; padding: 1.3rem 2rem 1rem;
    background: #16130f; box-shadow: 0 8px 22px rgba(0,0,0,.55);
    border-bottom: 1px solid rgba(180,170,150,.10);
  }
  .topbar .controls { margin-bottom: 0; }
  .topnav { display: flex; gap: 4px; margin-bottom: .95rem; }
  .topnav a {
    font: .67rem -apple-system, system-ui, sans-serif; letter-spacing: .09em;
    text-transform: uppercase; color: #9a9285; text-decoration: none;
    padding: 5px 13px; border: 1px solid rgba(180,170,150,.22); border-radius: 5px;
    background: rgba(180,170,150,.05); transition: all .16s ease;
  }
  .topnav a:hover { color: #e8e2d4; border-color: var(--mem); }
  .topnav a.active { color: #06232a; background: var(--mem); border-color: var(--mem); font-weight: 700; }
  h1 {
    text-align: center; font-family: "Permanent Marker", "Marker Felt", "Comic Sans MS", cursive;
    color: #e8e2d4; font-size: 1.6rem; letter-spacing: .06em; margin-bottom: .4rem;
  }
  .sub { text-align: center; color: #8b8375; font-size: .8rem; margin-bottom: 1rem; }
  .controls { display: flex; align-items: center; justify-content: center; gap: .6rem; margin-bottom: 2rem; }
  .controls .ctl-label { color: #6f695d; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; }
  .controls input[type=range] {
    -webkit-appearance: none; appearance: none; width: 210px; height: 4px; border-radius: 3px;
    background: rgba(180,170,150,.22); outline: none; cursor: ew-resize;
  }
  .controls input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none; width: 15px; height: 15px; border-radius: 50%;
    background: #d8cdb6; box-shadow: 0 1px 4px rgba(0,0,0,.5); cursor: ew-resize;
  }
  .controls input[type=range]::-moz-range-thumb {
    width: 15px; height: 15px; border: none; border-radius: 50%; background: #d8cdb6; cursor: ew-resize;
  }
  .canvas {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(var(--card, 300px), 1fr));
    gap: 2.4rem; max-width: 1600px; margin: 0 auto;
  }
  .polaroid {
    background: #f4f1e8; padding: 14px 14px 10px; border-radius: 2px;
    transform: rotate(var(--tilt)); position: relative;
    box-shadow: 0 12px 30px rgba(0,0,0,.55), 0 2px 6px rgba(0,0,0,.4);
    transition: transform .25s ease;
  }
  .polaroid:hover { transform: rotate(0deg) scale(1.02); z-index: 5; }
  .tape {
    position: absolute; top: -12px; left: 50%; transform: translateX(-50%) rotate(-1.5deg);
    width: 92px; height: 26px; background: rgba(230,220,180,.55);
    box-shadow: 0 1px 3px rgba(0,0,0,.25); border-left: 1px dashed rgba(0,0,0,.08);
    border-right: 1px dashed rgba(0,0,0,.08);
    cursor: grab; z-index: 6;
  }
  .tape:active { cursor: grabbing; }
  .team-tag {
    position: absolute; top: -8px; left: 12px; z-index: 6;
    font-family: -apple-system, system-ui, sans-serif;
    font-size: .56rem; letter-spacing: .09em; text-transform: uppercase;
    padding: 2px 7px; border-radius: 3px; color: #2a2620;
    background: #d8cdb6; box-shadow: 0 1px 3px rgba(0,0,0,.4); pointer-events: none;
  }
  .polaroid.dragging {
    opacity: .55; z-index: 20; transform: rotate(0deg) scale(1.03);
    box-shadow: 0 22px 50px rgba(0,0,0,.6);
  }
  .photo {
    background: #0b0e13; height: calc(var(--card, 300px) * 1.33); position: relative; overflow: hidden;
    cursor: zoom-in;
  }
  .photo img {
    position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;
    object-position: center 22%; opacity: .94;
  }
  .photo.noimg { background: linear-gradient(160deg, #1a1f2e, #0b0e13); }
  /* scrim so the terminal text reads as it scrolls up over the owl */
  .photo::after {
    content: ""; position: absolute; inset: 0; z-index: 1; pointer-events: none;
    background: linear-gradient(to bottom, transparent 30%, rgba(8,10,14,.5) 60%, rgba(8,10,14,.92) 100%);
  }
  .grip {
    position: absolute; right: 3px; bottom: 3px; width: 20px; height: 20px;
    cursor: nwse-resize; z-index: 7; opacity: 0; transition: opacity .15s;
    background:
      linear-gradient(135deg, transparent 44%, rgba(222,212,182,.9) 44%, rgba(222,212,182,.9) 52%, transparent 52%),
      linear-gradient(135deg, transparent 64%, rgba(222,212,182,.9) 64%, rgba(222,212,182,.9) 72%, transparent 72%);
  }
  .polaroid:hover .grip { opacity: .7; }
  .grip:hover { opacity: 1; }
  /* the live pane output — a contained terminal panel in the card whitespace,
     never over the portrait (sits where the notes/provstrip sit). */
  .chat {
    margin: 5px 4px 2px; padding: 8px 10px; overflow-y: auto;
    max-height: calc(var(--card, 300px) * 0.72);
    border-radius: 5px; border: 1px solid #22303a; background: #0a0d12;
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: calc(var(--card, 300px) * 0.034); line-height: 1.44;  /* scales with photo size */
    color: #9fd3a8; white-space: pre-wrap; word-break: break-word;
    scrollbar-width: thin;
  }
  .caption { padding: 9px 4px 4px; }
  .cname {
    font-family: "Permanent Marker", "Marker Felt", "Comic Sans MS", cursive;
    font-size: 1.05rem; color: #23201a; letter-spacing: .04em;
  }
  .cnote {
    font-family: "Bradley Hand", "Marker Felt", cursive;
    font-size: .8rem; color: #4c463c; margin-left: .3rem;
  }
  .talk input {
    width: 100%; margin-top: 6px; padding: 7px 9px;
    border: 1px solid #c9c2b2; border-radius: 3px; background: #fbf9f2;
    font: 12.5px -apple-system, sans-serif; color: #23201a; outline: none;
  }
  .talk input:focus { border-color: #8a8270; box-shadow: 0 0 0 2px rgba(140,130,110,.2); }
  .polaroid.sent { animation: flash .5s ease; }
  .polaroid.senderr { animation: shake .45s ease; }
  .polaroid.senderr .talk input { border-color: #b3543f; box-shadow: 0 0 0 2px rgba(179,84,63,.3); }
  @keyframes shake { 0%,100% { transform: rotate(var(--tilt)); } 25% { transform: rotate(var(--tilt)) translateX(-6px); } 75% { transform: rotate(var(--tilt)) translateX(6px); } }
  @keyframes flash { 0% { box-shadow: 0 0 0 3px rgba(140,200,150,.8), 0 12px 30px rgba(0,0,0,.55); } }
  /* fullscreen lightbox */
  .lightbox {
    position: fixed; inset: 0; z-index: 100; display: none;
    background: rgba(8,6,4,.93); backdrop-filter: blur(6px);
    align-items: center; justify-content: center; padding: 1.5vh 1.5vw; cursor: zoom-out;
  }
  .lightbox.open { display: flex; }
  /* the fullscreen IS a polaroid: cream card, photo on top, thick caption
     strip at the bottom with the handwritten name + the input. */
  .lightbox .frame {
    background: #f4f1e8; padding: 15px 15px 0; border-radius: 3px; cursor: default;
    box-shadow: 0 30px 90px rgba(0,0,0,.75);
    width: 100%; height: 100%;
    position: relative; overflow: hidden; display: flex; flex-direction: column;
  }
  /* the "photo": owl + darkened terminal, inside the cream frame */
  .lightbox .lb-photo-area {
    position: relative; flex: 1; min-height: 0; overflow: hidden;
    background: #0b0e13; border-radius: 2px;
  }
  .lightbox .lb-photo { position: absolute; inset: 0; z-index: 0; background: #0b0e13; }
  .lightbox .lb-photo img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: center 20%; opacity: .30; filter: blur(3px); transform: scale(1.04); }
  /* strong, near-uniform darkening so the terminal text reads clearly over the photo */
  .lightbox .lb-scrim {
    position: absolute; inset: 0; z-index: 1; pointer-events: none;
    background: linear-gradient(to bottom, rgba(6,8,13,.80) 0%, rgba(6,8,13,.88) 45%, rgba(6,8,13,.98) 100%);
  }
  .lightbox .lb-chat {
    position: absolute; inset: 0; z-index: 2; overflow: auto;
    overscroll-behavior: contain;  /* don't chain scroll to the board behind */
    padding: 12px 22px 14px; margin: 0; background: transparent;
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: clamp(11px, 1.5vw, 22px); line-height: 1.5;  /* fills window width; re-adapts on resize */
    color: #9fd3a8; white-space: pre;
    text-shadow: 0 1px 3px rgba(0,0,0,.98); scrollbar-width: thin;
  }
  /* the polaroid caption strip on the cream, below the photo */
  .lightbox .lb-cap {
    flex: none; padding: 13px 8px 2px;
    font-family: "Permanent Marker", "Marker Felt", cursive; color: #23201a;
    font-size: 1.55rem; letter-spacing: .02em;
  }
  .lightbox .lb-note { font-family: "Bradley Hand", cursive; color: #4c463c; font-size: 1.05rem; margin-left: .5rem; }
  .lightbox .lb-talk { flex: none; padding: 6px 6px 14px; }
  .lightbox .lb-talk input {
    width: 100%; padding: 12px 14px; border: 1px solid #c9c2b2;
    border-radius: 5px; background: #fbf9f2;
    font: 15px -apple-system, sans-serif; color: #23201a; outline: none;
  }
  .lightbox .lb-talk input::placeholder { color: #9a938a; }
  .lightbox .lb-talk input:focus { border-color: #8a8270; box-shadow: 0 0 0 2px rgba(140,130,110,.25); }
  .lightbox .frame.sent .lb-talk input { border-color: #6faa6f; box-shadow: 0 0 0 2px rgba(120,180,120,.35); }
  .lightbox .frame.senderr .lb-talk input { border-color: #b3543f; box-shadow: 0 0 0 2px rgba(179,84,63,.35); }
  .lightbox .lb-close {
    position: absolute; top: 2.4vh; right: 2.4vw; z-index: 200;
    width: 46px; height: 46px; display: flex; align-items: center; justify-content: center;
    color: #f4efe2; font-size: 1.9rem; line-height: 1; cursor: pointer;
    background: rgba(12,14,20,.72); border: 1px solid rgba(235,228,212,.38);
    border-radius: 50%; -webkit-backdrop-filter: blur(4px); backdrop-filter: blur(4px);
    opacity: .9; transition: opacity .15s ease, transform .15s ease;
  }
  .lightbox .lb-close:hover { opacity: 1; transform: scale(1.08); }
  /* teams + the undeveloped-polaroid placeholder (unauthored marks) */
  .team { max-width: 1500px; margin: 0 auto 2.8rem; }
  .team-head {
    display: flex; align-items: baseline; gap: .7rem;
    padding: 0 .3rem .8rem; margin-bottom: 1.7rem;
    border-bottom: 1px solid rgba(180,170,150,.14);
  }
  .team-head .tname {
    font-family: "Permanent Marker", "Marker Felt", cursive;
    color: #e8e2d4; font-size: 1.15rem; letter-spacing: .05em;
  }
  .team-head .tsub { color: #8b8375; font-size: .78rem; }
  .polaroid.pending { background: #ece8dc; }
  .polaroid.pending .photo { cursor: zoom-in; }
  .ghost {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 12px;
    background: repeating-linear-gradient(135deg, #0c0f16, #0c0f16 9px, #0a0d12 9px, #0a0d12 18px);
  }
  .ghostowl {
    width: 92px; height: 112px; fill: none;
    stroke: rgba(150,162,184,.28); stroke-width: 1.4;
    stroke-dasharray: 3 3; stroke-linecap: round; stroke-linejoin: round;
  }
  .pendlabel {
    font-family: "Bradley Hand", cursive; color: rgba(184,178,162,.5);
    font-size: .82rem; letter-spacing: .02em;
  }
  .polaroid.pending .cnote { color: #6b6456; font-style: italic; }
  .lightbox .lb-ghost {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 16px;
    background: repeating-linear-gradient(135deg, #0c0f16, #0c0f16 12px, #0a0d12 12px, #0a0d12 24px);
  }
  .lightbox .lb-ghost .ghostowl { width: 150px; height: 182px; }
  .lightbox .lb-ghost span {
    font-family: "Bradley Hand", cursive; color: rgba(184,178,162,.55); font-size: 1.1rem;
  }
  /* provenance chips — what a being has committed to memory. The one place
     the memory accent (--mem) is used; the eye learns it in one exposure.
     Sits on the cream caption strip, never on the owl. Reads as a receipt. */
  :root { --mem: #0f97a6; }
  .provstrip { display: flex; flex-direction: column; gap: 5px; padding: 2px 2px 0; }
  .provstrip:empty { display: none; }
  /* per-card view toggle: stickies (what they remember) vs chat output (live pane) */
  .viewtoggle { display: flex; justify-content: flex-end; gap: 0; margin: 3px 2px 1px; }
  .vt {
    border: 1px solid #d3ccbb; background: #efece2; color: #6f695d;
    font: .56rem -apple-system, system-ui, sans-serif; letter-spacing: .07em;
    text-transform: uppercase; padding: 3px 9px; cursor: pointer; transition: all .15s ease;
  }
  .vt:first-child { border-radius: 4px 0 0 4px; }
  .vt:last-child { border-radius: 0 4px 4px 0; border-left: none; }
  .vt:hover { color: #2a2620; }
  .polaroid[data-view="stickies"] .vt[data-view="stickies"],
  .polaroid[data-view="chat"] .vt[data-view="chat"] {
    background: var(--mem); color: #06232a; font-weight: 700; border-color: var(--mem);
  }
  .polaroid[data-view="stickies"] .chat { display: none; }
  .polaroid[data-view="chat"] .provstrip { display: none; }
  .chip {
    border-left: 3px solid var(--mem); border-radius: 2px;
    background: rgba(15,151,166,.07); padding: 4px 8px 5px;
  }
  .chip-kind {
    display: inline-block; font-size: .52rem; letter-spacing: .11em;
    text-transform: uppercase; font-weight: 700; color: var(--mem);
    font-family: -apple-system, system-ui, sans-serif;
  }
  .chip-what {
    display: block; margin: 1px 0 2px; font-size: .73rem; line-height: 1.34;
    color: #2a2620; font-family: -apple-system, system-ui, sans-serif;
  }
  .chip-meta {
    font-size: .6rem; color: #726a5c; letter-spacing: .01em;
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
  }
  .chip-meta b { color: #3c362c; font-weight: 600; }
  .chip-where { color: var(--mem); }
  /* Demo Mode toggle button (topbar) */
  .demo-btn {
    margin-left: .9rem; padding: 5px 12px; border: 1px solid rgba(180,170,150,.3);
    border-radius: 4px; background: rgba(180,170,150,.08); color: #cfc8b8;
    font: .72rem -apple-system, system-ui, sans-serif; letter-spacing: .05em;
    text-transform: uppercase; cursor: pointer; transition: all .18s ease;
  }
  .demo-btn:hover { border-color: var(--mem); color: #e8e2d4; }
  .asof-label { margin-left: .9rem; }
  .asof-input {
    background: rgba(180,170,150,.08); color: #cfc8b8; border: 1px solid rgba(180,170,150,.3);
    border-radius: 4px; padding: 4px 8px; margin-left: .4rem; cursor: pointer;
    font: .72rem -apple-system, system-ui, sans-serif; color-scheme: dark;
  }
  .asof-input:hover { border-color: var(--mem); }
  body.demo .demo-btn { background: var(--mem); border-color: var(--mem); color: #06232a; font-weight: 700; }
  /* Demo Mode — the control app BECOMES the Leonard Test stage (brief's Mode 2):
     the working-chatter goes quiet, the memory chips take the stage, and the
     operator controls step off. Same app, staged. Linkable via ?demo=1. */
  /* chatter visibility is now per-card (data-view, default stickies), so Demo
     Mode no longer force-hides it — a card can still be flipped to chat on stage. */
  body.demo .photo::after {
    background: linear-gradient(to bottom, transparent 24%, rgba(8,10,14,.5) 55%, rgba(8,10,14,.9) 100%);
  }
  /* keep the CHAT interface — talking to your agents is what the control app IS;
     only the raw pane chatter is quieted, never the ability to speak to them. */
  body.demo .talk input { font-size: .9rem; padding: 9px 12px; border-color: rgba(15,151,166,.38); }
  body.demo .talk input::placeholder { color: #8a8270; }
  body.demo .talk input:focus { border-color: var(--mem); box-shadow: 0 0 0 2px rgba(15,151,166,.25); }
  body.demo #sizer, body.demo .ctl-label { opacity: .28; }
  body.demo .provstrip { gap: 7px; padding-top: 5px; }
  body.demo .chip { padding: 7px 11px 8px; }
  body.demo .chip-kind { font-size: .64rem; letter-spacing: .13em; }
  body.demo .chip-what { font-size: 1.02rem; line-height: 1.4; }
  body.demo .chip-meta { font-size: .74rem; }
  body.demo .cname { font-size: 1.32rem; }
  body.demo .cnote { font-size: .95rem; }
  /* The Leonard Test — the three-act play, staged over the whole app. */
  #leonard-run { display: none; }
  body.demo #leonard-run { display: inline-block; }
  .leonard {
    position: fixed; inset: 0; z-index: 200; display: none;
    background: radial-gradient(ellipse at 50% -10%, #14110c 0%, #080706 68%);
    overflow: auto; padding: 6vh 6vw 10vh;
  }
  .leonard.open { display: block; }
  .ln-scroll { max-width: 760px; margin: 0 auto; }
  .ln-head {
    text-align: center; color: #8b8375; font-size: .78rem; letter-spacing: .16em;
    text-transform: uppercase; margin-bottom: 2.8rem;
  }
  .ln-head span { color: #e8e2d4; }
  .ln-act {
    opacity: 0; transform: translateY(12px); margin-bottom: 2.8rem;
    transition: opacity .85s ease, transform .85s ease;
  }
  .ln-act.in { opacity: 1; transform: none; }
  .leonard.still .ln-act, .leonard.still .ln-qa-item { opacity: 1 !important; transform: none !important; transition: none !important; }
  .ln-tag {
    font-size: .62rem; letter-spacing: .2em; text-transform: uppercase;
    font-weight: 700; color: var(--mem); margin-bottom: .8rem;
  }
  .ln-death .ln-decision { font-size: 1.4rem; line-height: 1.5; color: #eae4d6; font-weight: 300; }
  .ln-death .ln-src { color: #6f695d; font-size: .82rem; margin-top: .9rem; font-family: "SF Mono", ui-monospace, monospace; }
  .ln-death .ln-cap { color: #c86a54; font-size: .98rem; margin-top: 1.3rem; letter-spacing: .01em; }
  .ln-wake { text-align: center; color: #8b8375; padding: .6rem 0; }
  .ln-wake .ln-cap { font-size: 1.1rem; color: #cfc8b8; }
  .ln-remember .ln-qa { margin-top: .3rem; }
  .ln-qa-item {
    margin-bottom: 1.6rem; opacity: 0; transform: translateY(8px);
    transition: opacity .7s ease, transform .7s ease;
  }
  .ln-qa-item.in { opacity: 1; transform: none; }
  .ln-q { color: #cfc8b8; font-size: 1.06rem; margin-bottom: .6rem; }
  .ln-q::before { content: "Q  "; color: var(--mem); font-weight: 700; }
  .ln-gen {
    font-size: .68rem; color: var(--mem); font-style: italic; letter-spacing: .03em;
    margin-left: .45rem; animation: lngenpulse 1.2s ease-in-out infinite;
  }
  @keyframes lngenpulse { 0%, 100% { opacity: .35; } 50% { opacity: .95; } }
  .leonard .chip { background: rgba(15,151,166,.1); border-left: 3px solid var(--mem); border-radius: 3px; padding: 9px 13px 10px; }
  .leonard .chip-kind { color: var(--mem); }
  .leonard .chip-what { color: #e6e0d2; font-size: .96rem; line-height: 1.4; }
  .leonard .chip-meta { color: #8b8375; }
  .leonard .chip-meta b { color: #cfc8b8; }
  .ln-end { text-align: center; margin: 2.2rem 0 1rem; }
  .ln-end .ln-latin { color: #e8e2d4; font-size: 1.18rem; font-style: italic; letter-spacing: .01em; }
  .ln-end .ln-thesis { color: #8b8375; font-size: .96rem; margin-top: .6rem; }
  /* Act 4 — the model's own questions, clearly set apart from the static anchor */
  .ln-generated { border-top: 1px solid rgba(15,151,166,.22); padding-top: 1.5rem; }
  .ln-tag-gen { color: var(--mem); }
  .ln-genote { color: #8b8375; font-size: .8rem; font-style: italic; margin: -.4rem 0 1.2rem; max-width: 640px; }
  .ln-a { color: #eae4d6; font-size: 1rem; line-height: 1.5; margin: .5rem 0 .55rem; }
  .ln-a::before { content: "A  "; color: var(--mem); font-weight: 700; }
  .ln-cite { font-size: .6rem; color: #6f695d; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .35rem; }
  .ln-close {
    position: fixed; top: 2.4vh; right: 2.6vw; z-index: 210;
    width: 44px; height: 44px; display: flex; align-items: center; justify-content: center;
    color: #f4efe2; font-size: 1.8rem; cursor: pointer; border-radius: 50%;
    background: rgba(20,18,14,.7); border: 1px solid rgba(235,228,212,.32);
  }
  .ln-close:hover { border-color: var(--mem); }
</style></head>
<body>
  <div class="topbar">
    <nav class="topnav"><a href="/">mission control</a><a href="/console">console</a><a href="/memento?demo=1">demo</a><a href="/memento" class="active">memento</a></nav>
    <h1>Memento agere, memento mori.</h1>
    <div class="sub">Remember to act. Remember you will die.</div>
    <div class="controls"><span class="ctl-label">photo size</span><input type="range" id="sizer" min="240" max="640" step="10" value="300" aria-label="photo size"><button id="demo-toggle" class="demo-btn" type="button" aria-pressed="false">Demo Mode</button><button id="leonard-run" class="demo-btn leonard-run" type="button">&#9654; Leonard Test</button><span class="ctl-label asof-label">memory as of</span><input type="date" id="asof" class="asof-input" aria-label="memory as of date"><button id="asof-now" class="demo-btn" type="button" title="jump to latest (live)">now</button></div>
  </div>
__CARDS__
  <!-- The Leonard Test: memory crosses the session boundary, staged from live memory. -->
  <div class="leonard" id="leonard" aria-hidden="true">
    <div class="ln-scroll">
      <div class="ln-head"><span id="ln-agent">opus</span> &middot; the Leonard Test</div>
      <div class="ln-body" id="ln-body"></div>
    </div>
    <div class="ln-close" id="ln-close" title="close">&times;</div>
  </div>
  <div class="lightbox" id="lightbox">
    <div class="lb-close">&times;</div>
    <div class="frame">
      <div class="lb-photo-area">
        <div class="lb-photo"><img id="lb-img" alt=""><div id="lb-ghost" class="lb-ghost" style="display:none"><svg class="ghostowl" viewBox="0 0 100 118"><path d="M28 20 L40 40 M72 20 L60 40"/><circle cx="50" cy="46" r="30"/><circle cx="39" cy="45" r="6"/><circle cx="61" cy="45" r="6"/><path d="M50 52 l-4 8 h8 z"/><ellipse cx="50" cy="86" rx="27" ry="28"/></svg><span>mark not yet authored</span></div></div>
        <div class="lb-scrim"></div>
        <pre class="lb-chat" id="lb-chat"></pre>
      </div>
      <div class="lb-cap"><span id="lb-name"></span><span class="lb-note" id="lb-note"></span></div>
      <form class="lb-talk" id="lb-talk"><input type="text" id="lb-input" placeholder="say something…" autocomplete="off"></form>
    </div>
  </div>
<script>
  // time-travel: scope the wall's memory to a chosen day ("" = now / all memory)
  let _asof = new URLSearchParams(location.search).get("asof") || "";
  if (!/^\\d{4}-\\d{2}-\\d{2}$/.test(_asof)) _asof = "";
  function asofQS(p){ return _asof ? p + "asof=" + encodeURIComponent(_asof) : ""; }
  function refreshAll(){ document.querySelectorAll(".polaroid").forEach(p => loadProv(p.dataset.agent)); }
  (function(){
    const inp = document.getElementById("asof");
    const nowBtn = document.getElementById("asof-now");
    if (!inp) return;
    fetch("/memento/timerange").then(r => r.json()).then(tr => {
      if (tr.min) inp.min = tr.min;
      if (tr.max) inp.max = tr.max;
      inp.value = _asof || tr.max || "";   // show latest; _asof stays "" (live) until picked
    }).catch(() => {});
    inp.addEventListener("change", () => { _asof = inp.value || ""; refreshAll(); });
    if (nowBtn) nowBtn.addEventListener("click", () => { _asof = ""; inp.value = inp.max || ""; refreshAll(); });
  })();
  // board photo resize — scales all polaroids via --card, remembered across reloads
  (function(){
    const sizer = document.getElementById("sizer");
    if (!sizer) return;
    const setGlobal = v => document.documentElement.style.setProperty("--card", v + "px");
    const saved = localStorage.getItem("memento-card");
    if (saved) { setGlobal(+saved); sizer.value = saved; } else { setGlobal(+sizer.value); }
    sizer.addEventListener("input", () => { setGlobal(+sizer.value); localStorage.setItem("memento-card", sizer.value); });
  })();
  // Demo Mode — turn the control app into the Leonard Test stage. ?demo=1 links it.
  (function(){
    const btn = document.getElementById("demo-toggle");
    const apply = on => { document.body.classList.toggle("demo", on); if (btn) btn.setAttribute("aria-pressed", on ? "true" : "false"); };
    const urlDemo = new URLSearchParams(location.search).get("demo") === "1";
    apply(urlDemo || localStorage.getItem("memento-demo") === "1");
    if (btn) btn.addEventListener("click", () => {
      const on = !document.body.classList.contains("demo");
      apply(on); localStorage.setItem("memento-demo", on ? "1" : "0");
    });
  })();
  // ---- The Leonard Test: three acts, staged from LIVE memory (replay = live).
  const _lnOverlay = document.getElementById("leonard");
  const _lnWait = ms => new Promise(r => setTimeout(r, ms));
  function _lnChip(ch){
    return '<div class="chip"><span class="chip-kind">' + escProv(ch.kind) + '</span>'
      + '<span class="chip-what">' + escProv(ch.what) + '</span>'
      + '<span class="chip-meta"><b>' + escProv(ch.who) + '</b> · ' + escProv(ch.when_label)
      + ' · <span class="chip-where">' + escProv(ch.where) + '</span></span></div>';
  }
  let _lnRunning = false;
  async function runLeonard(agent){
    if (_lnRunning) return;
    _lnRunning = true;
    agent = (agent || "opus").replace(/[^a-z]/g, "") || "opus";
    const params = new URLSearchParams(location.search);
    const still = params.get("still") === "1";
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const pace = still ? 0 : (reduce ? 0.3 : 1);
    const step = base => _lnWait(still ? 15 : base * pace + 300);
    const body = document.getElementById("ln-body");
    document.getElementById("ln-agent").textContent = agent;
    body.innerHTML = "";
    _lnOverlay.classList.toggle("still", still);
    _lnOverlay.classList.add("open");
    _lnOverlay.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    const add = html => {
      const el = document.createElement("div");
      el.className = "ln-act";
      el.innerHTML = html;
      body.appendChild(el);
      requestAnimationFrame(() => el.classList.add("in"));
      return el;
    };
    // Loading strategy: play instantly with the template questions (never freeze
    // while the 3B warms), then upgrade each question to the model's content-specific
    // version in the background — you watch the questions sharpen in place. still /
    // smart=0 take one deterministic fetch and skip the upgrade (recorded-clip mode).
    const explicitTemplates = params.get("smart") === "0";
    const wantGenerated = !explicitTemplates;   // the 3B "Asked live" section
    const leoURL = s => "/memento/leonard/" + agent + "?smart=" + s + (_asof ? "&asof=" + encodeURIComponent(_asof) : "");
    let d;
    // static acts come from a fast smart=0 fetch (no model wait); still renders it all at once.
    try { d = await (await fetch(leoURL(still ? 1 : 0))).json(); }
    catch (e) { add('could not reach memory.'); _lnRunning = false; return; }
    if (!d.reachable || !d.act1) {
      add('<div style="text-align:center;color:#8b8375">' + escProv(agent)
        + ' has authored no memory in this store yet — nothing to remember, honestly.</div>');
      _lnRunning = false; return;
    }
    // Act 1 — Death: a decision that outlived the session that made it.
    add('<div class="ln-death"><div class="ln-tag">Act 1 · Death</div>'
      + '<div class="ln-decision">“' + escProv(d.act1.chip.what) + '”</div>'
      + '<div class="ln-src">decided in session ' + escProv(d.act1.session) + ' · ' + escProv(d.act1.when) + '</div>'
      + '<div class="ln-cap">Session ended. Context destroyed.</div></div>');
    await step(3800);
    // Act 2 — Waking: a fresh instance, zero context.
    add('<div class="ln-wake"><div class="ln-tag">Act 2 · Waking</div>'
      + '<div class="ln-cap">New session. This instance has read nothing.</div></div>');
    await step(2500);
    // Act 3 — Remembering: the three canonical questions, STATIC + real chips.
    const rem = add('<div class="ln-remember"><div class="ln-tag">Act 3 · Remembering</div><div class="ln-qa"></div></div>');
    const qaEl = rem.querySelector(".ln-qa");
    for (const item of d.act3){
      const qi = document.createElement("div");
      qi.className = "ln-qa-item";
      qi.innerHTML = '<div class="ln-q">' + escProv(item.q) + '</div>' + _lnChip(item.chip);
      qaEl.appendChild(qi);
      requestAnimationFrame(() => qi.classList.add("in"));
      await step(2600);
    }
    // Act 4 — Asked live: the local 3B reads this being's OTHER memories and poses
    // its own questions; the answers are still real chips (model writes the question,
    // never the fact). Loads after the static acts — the "generating…" is the progress.
    let genEl = null;
    if (wantGenerated) {
      const genSec = add('<div class="ln-remember ln-generated"><div class="ln-tag ln-tag-gen">Act 4 · Asked live by the model</div>'
        + '<div class="ln-genote">the local 3B reads this being’s memory and asks its own questions — the answers are still real receipts</div>'
        + '<div class="ln-qa" id="ln-genqa"></div>'
        + '<div class="ln-gen" id="ln-genload">reading memory, writing questions…</div></div>');
      genEl = genSec.querySelector("#ln-genqa");
    }
    add('<div class="ln-end"><div class="ln-latin">Memento agere, memento mori.</div>'
      + '<div class="ln-thesis">What you feed the memory outlives the session that fed it.</div></div>');
    _lnRunning = false;
    // fill Act 4 from the 3B (already present for still; a background fetch otherwise)
    if (wantGenerated) {
      const renderGen = gen => {
        const load = document.getElementById("ln-genload");
        if (load) load.remove();
        if (!gen || !gen.length) {
          if (genEl) genEl.innerHTML = '<div style="color:#8b8375;font-size:.9rem">no further memories to ask about yet.</div>';
          return;
        }
        gen.forEach(item => {
          const qi = document.createElement("div");
          qi.className = "ln-qa-item in";
          qi.innerHTML = '<div class="ln-q">' + escProv(item.q) + '</div>'
            + '<div class="ln-a">' + escProv(item.a) + '</div>'
            + '<div class="ln-cite">the model&#39;s answer &middot; drawn from this memory &darr;</div>'
            + _lnChip(item.chip);
          genEl.appendChild(qi);
        });
      };
      if (still) { renderGen(d.generated); }
      else {
        fetch(leoURL(1)).then(r => r.json()).then(sd => renderGen(sd.generated))
          .catch(() => { const l = document.getElementById("ln-genload"); if (l) l.textContent = "(model unavailable)"; });
      }
    }
  }
  function closeLeonard(){
    _lnOverlay.classList.remove("open");
    _lnOverlay.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }
  (function(){
    const run = document.getElementById("leonard-run");
    if (run) run.addEventListener("click", () => runLeonard("opus"));
    const close = document.getElementById("ln-close");
    if (close) close.addEventListener("click", closeLeonard);
    if (_lnOverlay) _lnOverlay.addEventListener("click", e => { if (e.target === _lnOverlay) closeLeonard(); });
    document.addEventListener("keydown", e => { if (e.key === "Escape" && _lnOverlay.classList.contains("open")) closeLeonard(); });
    const la = new URLSearchParams(location.search).get("leonard");
    if (la){ document.body.classList.add("demo"); setTimeout(() => runLeonard(la), 400); }
  })();
  let lbAgent = null;
  let lbTimer = null;
  const lb = document.getElementById("lightbox");
  function openLb(agent) {
    lbAgent = agent;
    const card = document.querySelector(`.polaroid[data-agent="${agent}"]`);
    const pending = card.classList.contains("pending");
    const img = document.getElementById("lb-img");
    const ghost = document.getElementById("lb-ghost");
    if (pending) {
      img.style.display = "none"; img.removeAttribute("src");
      ghost.style.display = "flex";
    } else {
      ghost.style.display = "none";
      img.style.display = ""; img.src = `/memento/portrait/${agent}`;
    }
    document.getElementById("lb-name").textContent = card.querySelector(".cname").textContent;
    document.getElementById("lb-note").textContent = card.querySelector(".cnote").textContent;
    lb.classList.add("open");
    document.body.style.overflow = "hidden";   // lock the board behind
    const li = document.getElementById("lb-input");
    if (li) { li.value = ""; setTimeout(() => li.focus(), 30); }
    const lc0 = document.getElementById("lb-chat");
    lc0.innerHTML = ""; lc0.dataset.seeded = "";  // fresh load for this agent
    pollLb();                                  // fetch the full colour screen now
    if (lbTimer) clearInterval(lbTimer);
    lbTimer = setInterval(pollLb, 2000);
  }
  function closeLb() {
    lb.classList.remove("open"); lbAgent = null;
    document.body.style.overflow = "";         // restore board scroll
    if (lbTimer) { clearInterval(lbTimer); lbTimer = null; }
  }
  lb.addEventListener("click", e => { if (e.target === lb || e.target.classList.contains("lb-close")) closeLb(); });
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeLb(); });
  async function poll(agent) {
    try {
      const r = await fetch(`/memento/pane/${agent}?lines=40`);
      if (!r.ok) return;
      const j = await r.json();
      const el = document.getElementById(`chat-${agent}`);
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
      if (j && j.reachable) {
        el.innerHTML = j.content_html || "";
        el.dataset.seeded = "1";
      } else if (!el.dataset.seeded) {
        el.textContent = "· not connected ·";
      }
      if (atBottom) el.scrollTop = el.scrollHeight;
    } catch (e) { /* pane may be gone; keep last frame */ }
  }
  // fullscreen polls the FULL live screen (input box + hint bar + refs), in colour
  async function pollLb() {
    if (!lbAgent) return;
    const lc = document.getElementById("lb-chat");
    // freeze updates while the user has scrolled up to read scrollback;
    // resume + stick to bottom once they scroll back down.
    const atBottom = lc.scrollTop + lc.clientHeight >= lc.scrollHeight - 60;
    if (lc.dataset.seeded && !atBottom) return;
    try {
      const r = await fetch(`/memento/pane/${lbAgent}?full=1`);
      if (!r.ok) return;
      const j = await r.json();
      lc.innerHTML = (j && j.reachable) ? (j.content_html || "") : "· not connected ·";
      lc.dataset.seeded = "1";
      lc.scrollTop = lc.scrollHeight;
    } catch (e) { /* keep last frame */ }
  }
  // provenance chips — fetch what each being has committed to memory and
  // render receipts on the cream. Empty (no authored memory) → nothing shown.
  function escProv(s){ return String(s==null?"":s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
  async function loadProv(agent){
    const strip = document.getElementById(`prov-${agent}`);
    if(!strip) return;
    try{
      const r = await fetch(`/memento/provenance/${agent}?limit=2` + asofQS("&"));
      if(!r.ok) return;
      const items = await r.json();
      if(!Array.isArray(items) || !items.length){ strip.innerHTML = ""; return; }
      strip.innerHTML = items.map(it => `
        <div class="chip" title="${escProv(it.what)}">
          <span class="chip-kind">${escProv(it.kind)}</span>
          <span class="chip-what">${escProv(it.what)}</span>
          <span class="chip-meta"><b>${escProv(it.who)}</b> · ${escProv(it.when_label)} · <span class="chip-where">${escProv(it.where)}</span></span>
        </div>`).join("");
    }catch(e){ /* keep last render */ }
  }
  document.querySelectorAll(".polaroid").forEach(p => {
    const agent = p.dataset.agent;
    poll(agent);
    setInterval(() => poll(agent), 2500);
    loadProv(agent);
    setInterval(() => loadProv(agent), 30000);  // memory changes slowly
    p.querySelector(".photo").addEventListener("click", () => openLb(agent));
  });
  // per-card view toggle: stickies (memory) vs chat output (live pane), remembered per agent
  document.querySelectorAll(".polaroid").forEach(p => {
    const saved = localStorage.getItem("memento-view-" + p.dataset.agent);
    if (saved === "stickies" || saved === "chat") p.dataset.view = saved;
  });
  document.querySelectorAll(".vt").forEach(b => {
    b.addEventListener("click", ev => {
      ev.stopPropagation();
      const card = b.closest(".polaroid");
      card.dataset.view = b.dataset.view;
      localStorage.setItem("memento-view-" + b.dataset.agent, b.dataset.view);
    });
  });
  // per-card resize: drag a photo's corner grip to size just that one;
  // double-click the grip to reset it back to the global slider size.
  // drag-and-drop reorder: grab a card's tape and shift it on the board;
  // the others reflow, and the order is remembered across reloads.
  const canvas = document.getElementById("canvas");
  function persistOrder() {
    localStorage.setItem("memento-order",
      JSON.stringify([...canvas.querySelectorAll(".polaroid")].map(c => c.dataset.agent)));
  }
  (function restoreOrder() {
    const saved = JSON.parse(localStorage.getItem("memento-order") || "null");
    if (!saved) return;
    saved.forEach(agent => {
      const c = canvas.querySelector(`.polaroid[data-agent="${agent}"]`);
      if (c) canvas.appendChild(c);  // re-append in saved order
    });
  })();
  let dragCard = null;
  document.querySelectorAll(".polaroid .tape").forEach(tape => {
    tape.addEventListener("mousedown", e => {
      e.preventDefault();
      dragCard = tape.closest(".polaroid");
      dragCard.classList.add("dragging");
      const move = ev => {
        const el = document.elementFromPoint(ev.clientX, ev.clientY);
        const target = el && el.closest(".polaroid");
        if (target && target !== dragCard && target.parentElement === canvas) {
          const r = target.getBoundingClientRect();
          const before = ev.clientX < r.left + r.width / 2;
          canvas.insertBefore(dragCard, before ? target : target.nextSibling);
        }
      };
      const up = () => {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        if (dragCard) dragCard.classList.remove("dragging");
        dragCard = null;
        persistOrder();
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
  });
  document.querySelectorAll(".talk").forEach(f => {
    f.addEventListener("submit", async ev => {
      ev.preventDefault();
      const input = f.querySelector("input");
      const text = input.value.trim();  // empty allowed → sends a bare Enter
      const fd = new FormData();
      fd.append("text", text);
      const card = f.closest(".polaroid");
      try {
        const r = await fetch(`/memento/say/${f.dataset.agent}`, { method: "POST", body: fd });
        if (!r.ok) throw new Error(`send failed: ${r.status}`);
        input.value = "";
        card.classList.add("sent");
        setTimeout(() => card.classList.remove("sent"), 600);
      } catch (err) {
        card.classList.add("senderr");
        setTimeout(() => card.classList.remove("senderr"), 900);
      }
    });
  });
  // fullscreen talk: sends to whichever agent's photo is currently open
  document.getElementById("lb-talk").addEventListener("submit", async ev => {
    ev.preventDefault();
    const input = document.getElementById("lb-input");
    if (!lbAgent) return;
    const text = input.value.trim();  // empty allowed → sends a bare Enter
    const fd = new FormData();
    fd.append("text", text);
    const frame = document.querySelector("#lightbox .frame");
    try {
      const r = await fetch(`/memento/say/${lbAgent}`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`send failed: ${r.status}`);
      input.value = "";
      frame.classList.add("sent");
      setTimeout(() => frame.classList.remove("sent"), 600);
    } catch (err) {
      frame.classList.add("senderr");
      setTimeout(() => frame.classList.remove("senderr"), 900);
    }
    input.focus();
  });
</script>
</body></html>'''
        return HTMLResponse(page.replace("__CARDS__", cards_html))

    @app.get("/terminal/{name}")
    async def terminal_page(name: str):
        """Serve the xterm.js terminal pop-out page."""
        html = _TERMINAL_HTML.replace("{{AGENT_NAME}}", name)
        return HTMLResponse(html)

    @app.websocket("/ws/terminal/{name}")
    async def terminal_ws(websocket: WebSocket, name: str):
        """WebSocket bridge: tmux capture-pane -> client, client input -> tmux send-keys."""
        try:
            target = scoped_tmux_target(name)
        except TmuxTargetScopeError as exc:
            await websocket.close(code=1008, reason=str(exc))
            return
        await websocket.accept()
        is_codex = name in _CODEX_AGENTS

        async def send_snapshots():
            """Poll tmux and push ANSI output to the client."""
            last_content = ""
            while True:
                try:
                    result = subprocess.run(
                        ["tmux", "capture-pane", "-t", target, "-p", "-e",
                         "-S", "-200"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0 and result.stdout != last_content:
                        last_content = result.stdout
                        cur = subprocess.run(
                            ["tmux", "display-message", "-t", target, "-p",
                             "#{cursor_x} #{cursor_y} #{pane_width} #{pane_height}"],
                            capture_output=True, text=True, timeout=3,
                        )
                        cursor_info = {}
                        if cur.returncode == 0:
                            parts = cur.stdout.strip().split()
                            if len(parts) == 4:
                                cursor_info = {
                                    "cx": int(parts[0]), "cy": int(parts[1]),
                                    "w": int(parts[2]), "h": int(parts[3]),
                                }
                        await websocket.send_json({
                            "type": "output",
                            "content": last_content,
                            **cursor_info,
                        })
                except Exception:
                    break
                await asyncio.sleep(0.3)

        async def receive_input():
            """Read client keystrokes and forward to tmux."""
            while True:
                try:
                    msg = await websocket.receive_json()
                except Exception:
                    break
                msg_type = msg.get("type", "")
                if msg_type == "input":
                    text = msg.get("text", "")
                    if text:
                        subprocess.run(
                            ["tmux", "send-keys", "-t", target, text, "Enter"],
                            capture_output=True, timeout=5,
                        )
                        if is_codex:
                            await asyncio.sleep(0.3)
                            subprocess.run(
                                ["tmux", "send-keys", "-t", target, "Enter"],
                                capture_output=True, timeout=5,
                            )
                    else:
                        # Bare enter
                        subprocess.run(
                            ["tmux", "send-keys", "-t", target, "Enter"],
                            capture_output=True, timeout=5,
                        )
                elif msg_type == "key":
                    key = msg.get("key", "")
                    if key in _ALLOWED_KEYS:
                        subprocess.run(
                            ["tmux", "send-keys", "-t", target, key],
                            capture_output=True, timeout=5,
                        )

        sender = asyncio.create_task(send_snapshots())
        try:
            await receive_input()
        finally:
            sender.cancel()

    return app


# ---------------------------------------------------------------------------
# xterm.js terminal page (inline HTML)
# ---------------------------------------------------------------------------

_TERMINAL_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Terminal: {{AGENT_NAME}}</title>
<link rel="stylesheet" href="https://unpkg.com/@xterm/xterm@5.5.0/css/xterm.css">
<script src="https://unpkg.com/@xterm/xterm@5.5.0/lib/xterm.js"></script>
<script src="https://unpkg.com/@xterm/addon-fit@0.10.0/lib/addon-fit.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #141420;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    font-family: 'SF Mono', 'Fira Code', monospace;
  }
  #header {
    padding: 6px 16px;
    background: #1a1d27;
    border-bottom: 1px solid #2a2d3a;
    color: #8b5cf6;
    font-size: 12px;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }
  #header .status { color: #71717a; font-weight: 400; }
  #header .status.connected { color: #4ade80; }
  #terminal-container {
    flex: 1;
    padding: 4px;
    overflow: hidden;
  }
  #input-bar {
    display: flex;
    gap: 8px;
    padding: 8px 16px;
    border-top: 1px solid #2a2d3a;
    background: #1a1d27;
    flex-shrink: 0;
  }
  #input-bar input {
    flex: 1;
    background: #0f1117;
    color: #e4e4e7;
    border: 1px solid #2a2d3a;
    padding: 8px 12px;
    border-radius: 6px;
    font-family: inherit;
    font-size: 13px;
    outline: none;
  }
  #input-bar input:focus { border-color: #8b5cf6; }
  #input-bar button {
    background: #8b5cf6;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
  }
  .key-bar {
    display: flex;
    gap: 4px;
    padding: 6px 16px;
    border-top: 1px solid #2a2d3a;
    background: #1a1d27;
    flex-wrap: wrap;
    align-items: center;
    flex-shrink: 0;
  }
  .key-bar .kl { font-size: 10px; color: #71717a; margin-right: 4px; }
  .key-bar button {
    background: #0f1117;
    border: 1px solid #2a2d3a;
    color: #e4e4e7;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: inherit;
    font-size: 11px;
    cursor: pointer;
  }
  .key-bar button:hover { border-color: #8b5cf6; color: #8b5cf6; }
  .key-bar .ks { width: 1px; height: 18px; background: #2a2d3a; margin: 0 4px; }
</style>
</head>
<body>
<div id="header">
  <span>Terminal: {{AGENT_NAME}}</span>
  <span class="status" id="status">connecting...</span>
</div>
<div id="terminal-container"></div>
<div class="key-bar">
  <span class="kl">Keys:</span>
  <button data-key="Enter">Enter</button><button data-key="Escape">Esc</button>
  <button data-key="y">y</button><button data-key="n">n</button>
  <div class="ks"></div>
  <button data-key="Up">Up</button><button data-key="Down">Down</button><button data-key="Tab">Tab</button>
  <div class="ks"></div>
  <button data-key="C-c">^C</button><button data-key="C-d">^D</button><button data-key="q">q</button>
</div>
<div id="input-bar">
  <input type="text" id="inp" placeholder="Send to {{AGENT_NAME}}..." autocomplete="off" autofocus>
  <button id="btn">Send</button>
</div>
<script>
var agentName = '{{AGENT_NAME}}';
var term = new Terminal({
  theme: {
    background: '#141420',
    foreground: '#e4e4e7',
    cursor: '#8b5cf6',
    cursorAccent: '#141420',
    selectionBackground: 'rgba(139,92,246,0.3)',
    black: '#000000', red: '#cc0000', green: '#00aa00', yellow: '#aaaa00',
    blue: '#5555ff', magenta: '#aa00aa', cyan: '#00aaaa', white: '#aaaaaa',
    brightBlack: '#555555', brightRed: '#ff5555', brightGreen: '#55ff55',
    brightYellow: '#ffff55', brightBlue: '#5555ff', brightMagenta: '#ff55ff',
    brightCyan: '#55ffff', brightWhite: '#ffffff',
  },
  fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
  fontSize: 13,
  cursorBlink: true,
  cursorStyle: 'block',
  scrollback: 5000,
  convertEol: true,
});

var fitAddon = new FitAddon.FitAddon();
term.loadAddon(fitAddon);
term.open(document.getElementById('terminal-container'));
fitAddon.fit();
window.addEventListener('resize', function() { fitAddon.fit(); });

// WebSocket connection
var wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
var ws = new WebSocket(wsProto + '//' + location.host + '/ws/terminal/' + agentName);
var statusEl = document.getElementById('status');

ws.onopen = function() {
  statusEl.textContent = 'connected';
  statusEl.className = 'status connected';
};
ws.onclose = function() {
  statusEl.textContent = 'disconnected';
  statusEl.className = 'status';
};
ws.onerror = function() {
  statusEl.textContent = 'error';
  statusEl.className = 'status';
};

// Receive output from server
ws.onmessage = function(evt) {
  try {
    var msg = JSON.parse(evt.data);
    if (msg.type === 'output' && msg.content) {
      // Clear and rewrite terminal with new content
      term.reset();
      term.write(msg.content);
      // Position cursor if we have coordinates
      if (typeof msg.cy === 'number' && typeof msg.cx === 'number' && msg.h) {
        // Move cursor to the right position in the visible area
        var visibleStart = msg.content.split('\\n').length - msg.h;
        if (visibleStart < 0) visibleStart = 0;
        // xterm handles cursor positioning from the content itself
      }
    }
  } catch(e) {}
};

// Send text input
function send() {
  var inp = document.getElementById('inp');
  if (!inp.value.trim() && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: 'input', text: '' }));
    return;
  }
  if (inp.value.trim() && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: 'input', text: inp.value }));
    inp.value = '';
  }
}
document.getElementById('btn').addEventListener('click', send);

// Key forwarding from input field
var km = {ArrowUp:'Up',ArrowDown:'Down',ArrowLeft:'Left',ArrowRight:'Right',
           Escape:'Escape',Tab:'Tab',Enter:'Enter'};
document.getElementById('inp').addEventListener('keydown', function(e) {
  var inp = document.getElementById('inp');
  var k = km[e.key];
  if (k && !inp.value.trim() && ws.readyState === 1) {
    e.preventDefault();
    if (e.key === 'Enter') {
      ws.send(JSON.stringify({ type: 'input', text: '' }));
    } else {
      ws.send(JSON.stringify({ type: 'key', key: k }));
    }
    return;
  }
  if (e.key === 'Enter') { e.preventDefault(); send(); }
});

// Key palette
document.querySelector('.key-bar').addEventListener('click', function(e) {
  var b = e.target.closest('button[data-key]');
  if (b && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: 'key', key: b.dataset.key }));
  }
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="synapt dashboard")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    parser.add_argument(
        "--tmux-session",
        help=(
            "Explicit tmux session the dashboard may control. "
            "Without it, tmux routes are disabled."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--foreground", action="store_true", help="Run server in the terminal")
    group.add_argument("--stop", action="store_true", help="Stop the background dashboard server")
    group.add_argument("--launch", action="store_true", help="Explicitly launch in background mode")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    if args.stop:
        if _stop_dashboard():
            print("synapt dashboard: stopped")
        else:
            print("synapt dashboard: not running")
        return

    if args.foreground:
        print(f"synapt dashboard: {url}")
        if not args.no_open:
            import webbrowser

            webbrowser.open(url)

        import uvicorn

        uvicorn.run(
            create_app(tmux_session=args.tmux_session),
            host=args.host,
            port=args.port,
            log_level="warning",
        )
        return

    pid = _start_dashboard_background(
        host=args.host,
        port=args.port,
        no_open=args.no_open,
        tmux_session=args.tmux_session,
    )
    print(f"synapt dashboard: {url} (pid {pid})")


if __name__ == "__main__":
    main()
