"""The dashboard may control only a tmux session explicitly bound at app creation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import synapt.dashboard.app as dashboard


def _registry_world(monkeypatch, tmp_path: Path, *, target: str) -> None:
    db_path = tmp_path / ".synapt" / "orgs" / "fixture-org" / "team.db"
    db_path.parent.mkdir(parents=True)
    db_path.touch()
    monkeypatch.setattr(dashboard.Path, "home", classmethod(lambda _cls: tmp_path))
    monkeypatch.setattr(dashboard, "_resolve_org_id", lambda _value: "fixture-org")
    monkeypatch.setattr(
        dashboard,
        "_registry_list_agents",
        lambda _org, *, db_path: [
            {"display_name": "Alpha", "tmux_target": target}
        ],
    )


def _must_not_run(calls: list[list[str]]):
    def refuse(args, **_kwargs):
        calls.append(list(args))
        raise AssertionError(f"tmux escaped the dashboard scope: {args!r}")

    return refuse


def test_missing_scope_refuses_input_before_tmux(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(dashboard.subprocess, "run", _must_not_run(calls))
    client = TestClient(dashboard.create_app())

    response = client.post("/api/agent/alpha/input", data={"text": "hello"})

    assert response.status_code == 503
    assert calls == []


def test_missing_scope_does_not_read_tmux_for_agent_status(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(dashboard, "channel_agents_json", lambda: [])
    monkeypatch.setattr(dashboard, "_resolve_org_id", lambda _value: None)
    monkeypatch.setattr(dashboard.subprocess, "run", _must_not_run(calls))
    client = TestClient(dashboard.create_app())

    response = client.get("/api/agents")

    assert response.status_code == 200
    assert calls == []


def test_agent_status_reads_only_the_bound_tmux_session(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(dashboard, "channel_agents_json", lambda: [])
    monkeypatch.setattr(dashboard, "_resolve_org_id", lambda _value: None)
    monkeypatch.setattr(dashboard, "_KNOWN_AGENTS", {"alpha": {}})

    def list_bound_session(args, **_kwargs):
        calls.append(list(args))
        return SimpleNamespace(returncode=0, stdout="alpha\n")

    monkeypatch.setattr(dashboard.subprocess, "run", list_bound_session)
    client = TestClient(dashboard.create_app(tmux_session="workspace-session"))

    response = client.get("/api/agents")

    assert response.status_code == 200
    assert calls == [
        ["tmux", "list-windows", "-t", "workspace-session", "-F", "#{window_name}"]
    ]
    assert response.json() == [
        {
            "agent_id": "alpha",
            "display_name": "Alpha",
            "griptree": "",
            "role": "agent",
            "status": "online",
            "last_seen": "",
            "channels": [],
            "tmux_target": "workspace-session:alpha",
        }
    ]


def test_foreign_registry_target_refuses_input_before_tmux(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    _registry_world(monkeypatch, tmp_path, target="foreign-session:alpha")
    monkeypatch.setattr(dashboard.subprocess, "run", _must_not_run(calls))
    client = TestClient(dashboard.create_app(tmux_session="workspace-session"))

    response = client.post("/api/agent/alpha/input", data={"text": "hello"})

    assert response.status_code == 403
    assert calls == []


def test_route_rechecks_a_broken_resolver_before_tmux(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        dashboard,
        "_resolve_tmux_target",
        lambda _name, *, tmux_session: "foreign-session:alpha",
    )
    monkeypatch.setattr(dashboard.subprocess, "run", _must_not_run(calls))
    client = TestClient(dashboard.create_app(tmux_session="workspace-session"))

    response = client.post("/api/agent/alpha/input", data={"text": "hello"})

    assert response.status_code == 403
    assert calls == []


def test_matching_registry_target_reaches_tmux(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    _registry_world(monkeypatch, tmp_path, target="workspace-session:alpha")

    def succeed(args, **_kwargs):
        calls.append(list(args))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(dashboard.subprocess, "run", succeed)
    client = TestClient(dashboard.create_app(tmux_session="workspace-session"))

    response = client.post("/api/agent/alpha/input", data={"text": "hello"})

    assert response.status_code == 200
    assert calls == [
        ["tmux", "send-keys", "-t", "workspace-session:alpha", "hello", "Enter"]
    ]


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/api/agent/alpha/key", {"data": {"key": "C-c"}}),
        ("post", "/api/agent/alpha/upload", {"files": {"file": ("x.txt", b"x")}}),
        ("get", "/api/agent/alpha/snapshot", {}),
    ],
)
def test_every_http_tmux_surface_refuses_a_foreign_target_before_tmux(
    monkeypatch, tmp_path, method, path, kwargs
):
    calls: list[list[str]] = []
    _registry_world(monkeypatch, tmp_path, target="foreign-session:alpha")
    monkeypatch.setattr(dashboard.subprocess, "run", _must_not_run(calls))
    client = TestClient(dashboard.create_app(tmux_session="workspace-session"))

    response = getattr(client, method)(path, **kwargs)

    assert response.status_code == 403
    assert calls == []


def test_websocket_refuses_a_foreign_target_before_tmux(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    _registry_world(monkeypatch, tmp_path, target="foreign-session:alpha")
    monkeypatch.setattr(dashboard.subprocess, "run", _must_not_run(calls))
    client = TestClient(dashboard.create_app(tmux_session="workspace-session"))

    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect("/ws/terminal/alpha"):
            pass

    assert refused.value.code == 1008
    assert calls == []


def test_background_child_receives_the_explicit_scope():
    command = dashboard._background_command(
        "127.0.0.1", 8420, tmux_session="workspace-session"
    )

    assert command[-2:] == ["--tmux-session", "workspace-session"]
