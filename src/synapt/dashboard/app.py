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


def _tmux_window_agents() -> dict[str, str]:
    """Return {window_name: status} for agent windows in the tmux session."""
    try:
        result = subprocess.run(
            ["tmux", "list-windows", "-a", "-F", "#{window_name}"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return {}
        windows = set(result.stdout.strip().splitlines())
        return {w: "online" for w in windows if w in _KNOWN_AGENTS}
    except Exception:
        return {}

_TMUX_SESSION: str = "synapt"


def _load_tmux_session() -> None:
    """Load tmux session name from spawn config."""
    global _TMUX_SESSION
    from synapt.recall.core import _find_gripspace_root

    grip_root = _find_gripspace_root(Path.cwd())
    if grip_root is None:
        return
    for candidate in [
        grip_root / ".gitgrip" / "spawn.toml",
        grip_root / "config" / "spawn.toml",
    ]:
        if candidate.is_file():
            try:
                with open(candidate, "rb") as f:
                    cfg = tomllib.load(f)
                name = cfg.get("spawn", {}).get("session_name", "")
                if name:
                    _TMUX_SESSION = name
                return
            except (OSError, tomllib.TOMLDecodeError):
                pass


_load_tmux_session()


def _resolve_tmux_target(name: str) -> str:
    """Resolve a qualified tmux target for an agent (recall#692 fix 3).

    Checks cached agent data for a stored tmux_target. Falls back to
    {session}:{name} to avoid ambiguity when multiple sessions exist.
    """
    org_id = _resolve_org_id(None)
    if org_id:
        db_path = Path.home() / ".synapt" / "orgs" / org_id / "team.db"
        if db_path.exists():
            try:
                agents = _registry_list_agents(org_id, db_path=db_path)
                for agent in agents:
                    if agent.get("display_name") == name:
                        stored = agent.get("tmux_target")
                        if stored:
                            return stored
                        break
            except Exception:
                pass
    return f"{_TMUX_SESSION}:{name.lower()}"


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


def _combined_agents_json_sync() -> list[dict]:
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
    for window_name, status in _tmux_window_agents().items():
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
                "tmux_target": f"{_TMUX_SESSION}:{window_name}",
            }

    return sorted(agents_by_name.values(), key=lambda a: a.get("display_name", ""))


async def _combined_agents_json() -> list[dict]:
    """Async wrapper — runs SQLite reads off the event loop (recall#692 fix 2)."""
    return await asyncio.to_thread(_combined_agents_json_sync)


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


def _background_command(host: str, port: int) -> list[str]:
    """Build the detached child command for the dashboard server."""
    return [
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


def _start_dashboard_background(
    host: str,
    port: int,
    no_open: bool,
    project_dir: Path | None = None,
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

    cmd = _background_command(host, port)
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


def create_app() -> FastAPI:
    app = FastAPI(title="synapt dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _load_template()

    @app.get("/api/agents")
    async def api_agents():
        return await _combined_agents_json()

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
        target = _resolve_tmux_target(name)
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
        target = _resolve_tmux_target(name)
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
        target = _resolve_tmux_target(name)
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
        target = _resolve_tmux_target(name)
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
    _MEMENTO_TEAMS = [
        {
            "team": "synapt", "label": "synapt", "session": _TMUX_SESSION,
            "sub": "the core team",
            "agents": [
                {"name": "opus", "label": "OPUS", "note": "the coordinator. remembers for the team.", "tilt": -2.4, "authored": True},
                {"name": "apollo", "label": "APOLLO", "note": "builds. mends what breaks.", "tilt": 1.8, "authored": True},
                {"name": "atlas", "label": "ATLAS", "note": "research. follows the spirals.", "tilt": -1.2, "authored": True},
                {"name": "sentinel", "label": "SENTINEL", "note": "verifies everything. trust the lens.", "tilt": 2.6, "authored": True},
            ],
        },
        {
            "team": "conversa", "label": "conversa engagement", "session": "conversa",
            "sub": "synapt agents staffed on the Conversa product",
            "agents": [
                {"name": "anchor", "label": "ANCHOR", "note": "conversa lead · awaiting their mark", "tilt": -1.6, "authored": False},
                {"name": "helm", "label": "HELM", "note": "CTO · awaiting their mark", "tilt": 2.0, "authored": False},
                {"name": "lumen", "label": "LUMEN", "note": "CXO · awaiting their mark", "tilt": -2.4, "authored": False},
                {"name": "forge", "label": "FORGE", "note": "safety · awaiting their mark", "tilt": 1.6, "authored": False},
            ],
        },
    ]
    _MEMENTO_CODEX = {"atlas", "sentinel", "forge", "lumen"}
    _MEMENTO_TARGETS: dict[str, str] = {}
    for _tm in _MEMENTO_TEAMS:
        for _ag in _tm["agents"]:
            _MEMENTO_TARGETS[_ag["name"]] = f'{_tm["session"]}:{_ag["name"]}'
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

    _PORTRAITS_DIR = Path(
        os.environ.get(
            "SYNAPT_PORTRAITS_DIR",
            str(Path.home() / "Development/synapt/config/design/owl-canon/portraits"),
        )
    )

    @app.get("/memento/portrait/{name}")
    async def memento_portrait(name: str):
        """Serve an agent's canon portrait for the memento page (local demo)."""
        safe = re.sub(r"[^a-z]", "", name.lower())
        p = _PORTRAITS_DIR / f"portrait-{safe}.png"
        if p.exists():
            return FileResponse(p, media_type="image/png")
        raise HTTPException(status_code=404, detail="no portrait")

    @app.get("/memento/pane/{name}")
    async def memento_pane(name: str, lines: int = 40, full: int = 0):
        """Capture an agent's tmux pane for the memento board.

        Resolves the target cross-session (synapt|conversa) from the
        server-built ``_MEMENTO_TARGETS`` map, so ``name`` is only ever a
        lookup key — never interpolated into the tmux target.  Unreachable
        panes (a team not currently running) return reachable=False rather
        than an error, so the board degrades gracefully.
        """
        target = _MEMENTO_TARGETS.get(name)
        if target is None:
            raise HTTPException(status_code=404, detail="unknown agent")
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
        target = _MEMENTO_TARGETS.get(name)
        if target is None:
            raise HTTPException(status_code=404, detail="unknown agent")
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
    # matches "grip#790", "PR#123", and bare "#888" — no leading \b, which
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
        # Prefer a real ref the author wrote (grip#790, #123); else the
        # session receipt; else the journal itself. Shown only when real.
        m = _PROV_REF_RE.search(what or "")
        if m:
            return m.group(0)
        if session_id:
            return "s:" + session_id[:6]
        return "journal"

    def _memento_provenance(name: str, limit: int = 2) -> list[dict]:
        """Most-recent memory a being has authored, as chip records.

        Scans every worktree journal and keeps entries whose own
        ``agent_id`` (or ``griptree``) resolves to ``name``.  Returns
        ``[{what, kind, who, when, when_label, where}]`` newest first, or an
        empty list when the being has authored nothing here — honest silence.
        """
        if name not in _MEMENTO_TARGETS:
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
                what = " ".join(what.split())  # collapse whitespace/newlines
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
    async def memento_provenance(name: str, limit: int = 2):
        """Provenance chips for a being: what it has committed to memory."""
        safe = re.sub(r"[^a-z]", "", name.lower())
        return _memento_provenance(safe, limit=limit)

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
              <div class="polaroid{pend}" style="--tilt:{a["tilt"]}deg" data-agent="{a["name"]}" data-team="{tm["team"]}">
                <div class="tape"></div>
                <div class="team-tag">{tm["label"]}</div>
                <div class="photo">
                  {photo_inner}
                  <pre class="chat" id="chat-{a["name"]}">…</pre>
                </div>
                <div class="caption">
                  <span class="cname">{a["label"]}</span>
                  <span class="cnote">— {a["note"]}</span>
                </div>
                <div class="provstrip" id="prov-{a["name"]}" aria-label="what {a["label"].lower()} remembers"></div>
                <form class="talk" data-agent="{a["name"]}">
                  <input type="text" placeholder="say something to {a["label"].lower()}…" autocomplete="off">
                </form>
              </div>''')
        cards_html = '<div class="canvas" id="canvas">' + "".join(cards) + '</div>'
        page = '''<!doctype html>
<html><head><meta charset="utf-8"><title>synapt — memento</title>
<style>
  * { box-sizing: border-box; margin: 0; }
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
  .polaroid[data-team="conversa"] .team-tag { background: #b7c4d6; }
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
  .chat {
    position: absolute; inset: 0; z-index: 2; overflow-y: auto;
    padding: calc(var(--card, 300px) * 0.62) 10px 8px; margin: 0;
    font-family: "SF Mono", ui-monospace, Menlo, monospace;
    font-size: calc(var(--card, 300px) * 0.032); line-height: 1.4;  /* scales with photo size */
    color: #9fd3a8; white-space: pre-wrap; word-break: break-word;
    background: transparent; text-shadow: 0 1px 3px rgba(0,0,0,.96);
    /* text is solid where it rests low, and fades into the owl as it scrolls up */
    -webkit-mask-image: linear-gradient(to bottom, transparent 0, #000 30%, #000 100%);
    mask-image: linear-gradient(to bottom, transparent 0, #000 30%, #000 100%);
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
  .lightbox .lb-photo img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: center 20%; opacity: .82; }
  /* OG-style darkening gradient so the chat reads over the owl */
  .lightbox .lb-scrim {
    position: absolute; inset: 0; z-index: 1; pointer-events: none;
    background: linear-gradient(to bottom, rgba(8,10,15,.52) 0%, rgba(8,10,15,.80) 45%, rgba(8,10,15,.97) 100%);
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
</style></head>
<body>
  <div class="topbar">
    <h1>Memento agere, memento mori.</h1>
    <div class="sub">Remember to act. Remember you will die.</div>
    <div class="controls"><span class="ctl-label">photo size</span><input type="range" id="sizer" min="240" max="640" step="10" value="300" aria-label="photo size"></div>
  </div>
__CARDS__
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
  // board photo resize — scales all polaroids via --card, remembered across reloads
  (function(){
    const sizer = document.getElementById("sizer");
    if (!sizer) return;
    const setGlobal = v => document.documentElement.style.setProperty("--card", v + "px");
    const saved = localStorage.getItem("memento-card");
    if (saved) { setGlobal(+saved); sizer.value = saved; } else { setGlobal(+sizer.value); }
    sizer.addEventListener("input", () => { setGlobal(+sizer.value); localStorage.setItem("memento-card", sizer.value); });
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
      const r = await fetch(`/memento/provenance/${agent}?limit=2`);
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
        await websocket.accept()
        target = _resolve_tmux_target(name)
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

        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
        return

    pid = _start_dashboard_background(
        host=args.host,
        port=args.port,
        no_open=args.no_open,
    )
    print(f"synapt dashboard: {url} (pid {pid})")


if __name__ == "__main__":
    main()
