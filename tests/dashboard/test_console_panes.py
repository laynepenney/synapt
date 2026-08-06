"""The /console pane switcher may browse only the tmux session bound at app creation.

The switcher generalizes the console's fixed agent roster to the bound
session's REAL topology: every window and pane, enumerated live, any of them
viewable in the hero pane. That widening is exactly why its scope discipline
must be tighter than the roster's, not looser:

- Enumeration and capture both require the explicit session (503 otherwise),
  the same contract recall#923 established for every other tmux surface.
- Capture targets are WINDOW.PANE COORDINATES validated by membership in a
  fresh enumeration of the bound session. Raw tmux ``%id`` handles are
  rejected at the shape gate: a ``%id`` resolves across sessions, so
  accepting one would be an escape hatch around the very scope this
  dashboard binds. The coordinates the client sends never reach ``-t``;
  the target is rebuilt from re-parsed integers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import synapt.dashboard.app as dashboard


SESSION = "workspace-session"

PANES_FMT = (
    "#{window_index}\t#{pane_index}\t#{window_name}"
    "\t#{pane_active}\t#{pane_current_command}\t#{pane_title}"
)

LIST_PANES_ARGV = ["tmux", "list-panes", "-s", "-t", SESSION, "-F", PANES_FMT]

# One agent window (a REAL roster name — the roster is a create_app closure
# local, so it cannot be monkeypatched; "opus" is the stable production
# name), one plain window, one multi-pane window — with a tab hiding inside
# the last title to prove the parser splits at most five times and lets the
# title keep its bytes.
PANES_FIXTURE = (
    "1\t0\topus\t1\tnode\topus pane\n"
    "2\t0\tzsh\t0\tzsh\t\n"
    "5\t0\tdashboard\t0\tpython\tserver\n"
    "5\t2\tdashboard\t1\ttail\tlogs\there\n"
)


def _must_not_run(calls: list[list[str]]):
    def refuse(args, **_kwargs):
        calls.append(list(args))
        raise AssertionError(f"tmux escaped the dashboard scope: {args!r}")

    return refuse


def _enumeration_only(calls: list[list[str]]):
    """list-panes answers with the fixture; anything else is an escape."""

    def run(args, **_kwargs):
        calls.append(list(args))
        if args[1] == "list-panes":
            return SimpleNamespace(returncode=0, stdout=PANES_FIXTURE, stderr="")
        raise AssertionError(f"tmux escaped past enumeration: {args!r}")

    return run


def _enumeration_then_capture(calls: list[list[str]], capture_stdout: str):
    def run(args, **_kwargs):
        calls.append(list(args))
        if args[1] == "list-panes":
            return SimpleNamespace(returncode=0, stdout=PANES_FIXTURE, stderr="")
        if args[1] == "capture-pane":
            return SimpleNamespace(returncode=0, stdout=capture_stdout, stderr="")
        raise AssertionError(f"unexpected tmux verb: {args!r}")

    return run


# ---- refusal before tmux ----------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/console/panes"),
        ("get", "/console/pane/1.0"),
    ],
)
def test_missing_scope_refuses_console_surfaces_before_tmux(monkeypatch, method, path):
    calls: list[list[str]] = []
    monkeypatch.setattr(dashboard.subprocess, "run", _must_not_run(calls))
    client = TestClient(dashboard.create_app())

    response = getattr(client, method)(path)

    assert response.status_code == 503
    assert calls == []


# ---- enumeration ------------------------------------------------------------


def test_panes_lists_only_the_bound_session(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(dashboard.subprocess, "run", _enumeration_only(calls))
    client = TestClient(dashboard.create_app(tmux_session=SESSION))

    response = client.get("/console/panes")

    assert response.status_code == 200
    assert calls == [LIST_PANES_ARGV]
    body = response.json()
    assert body["session"] == SESSION
    assert body["reachable"] is True
    panes = body["panes"]
    assert [p["id"] for p in panes] == ["1.0", "2.0", "5.0", "5.2"]

    agent_pane = panes[0]
    assert agent_pane["window_name"] == "opus"
    assert agent_pane["agent"] == "opus"
    assert agent_pane["active"] is True
    assert agent_pane["command"] == "node"
    assert isinstance(agent_pane["accent"], str) and agent_pane["accent"].startswith("#")

    plain = panes[1]
    assert plain["agent"] is None
    assert plain["active"] is False

    # split("\t", 5): the title keeps its own tab rather than spawning a column
    assert panes[3]["title"] == "logs\there"
    assert panes[3]["window"] == 5 and panes[3]["pane"] == 2


def test_panes_reports_unreachable_when_tmux_fails(monkeypatch):
    def fail(args, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="no server")

    monkeypatch.setattr(dashboard.subprocess, "run", fail)
    client = TestClient(dashboard.create_app(tmux_session=SESSION))

    response = client.get("/console/panes")

    assert response.status_code == 200
    assert response.json() == {"session": SESSION, "reachable": False, "panes": []}


# ---- capture: the shape gate runs before any subprocess ---------------------


@pytest.mark.parametrize(
    "bad_id",
    [
        "%37",        # tmux pane handle — resolves ACROSS sessions; rejecting it
                      # at the shape gate is the scope boundary, not pedantry
        "1.2.3",
        "a.b",
        "1",
        "1..2",
        ".5",
        "5.",
        "-1.0",
        "1.0 ",
    ],
)
def test_capture_rejects_malformed_pane_ids_before_tmux(monkeypatch, bad_id):
    calls: list[list[str]] = []
    monkeypatch.setattr(dashboard.subprocess, "run", _must_not_run(calls))
    client = TestClient(dashboard.create_app(tmux_session=SESSION))

    response = client.get(f"/console/pane/{bad_id}")

    assert response.status_code == 404
    assert calls == []


def test_capture_refuses_a_pane_outside_the_enumeration(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(dashboard.subprocess, "run", _enumeration_only(calls))
    client = TestClient(dashboard.create_app(tmux_session=SESSION))

    response = client.get("/console/pane/9.9")

    assert response.status_code == 404
    # enumeration ran; capture never did — membership is checked first
    assert calls == [LIST_PANES_ARGV]


def test_capture_happy_path_targets_the_bound_session_by_rebuilt_coordinates(
    monkeypatch,
):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        _enumeration_then_capture(calls, "plain line\n\x1b[31mred line\x1b[0m\n"),
    )
    client = TestClient(dashboard.create_app(tmux_session=SESSION))

    response = client.get("/console/pane/5.2?lines=120")

    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[0] == LIST_PANES_ARGV
    capture = calls[1]
    assert capture[:2] == ["tmux", "capture-pane"]
    assert capture[capture.index("-t") + 1] == f"{SESSION}:5.2"
    assert "-e" in capture  # SGR kept so the converter can colour it
    assert "-120" in "".join(capture)  # the lines clamp reached the argv

    body = response.json()
    assert body["id"] == "5.2"
    assert body["reachable"] is True
    assert body["window_name"] == "dashboard"
    assert "\x1b" not in body["content_html"]
    assert "red line" in body["content_html"]
    assert "<span" in body["content_html"]


def test_capture_reports_unreachable_when_tmux_fails(monkeypatch):
    calls: list[list[str]] = []

    def enumerate_then_fail(args, **_kwargs):
        calls.append(list(args))
        if args[1] == "list-panes":
            return SimpleNamespace(returncode=0, stdout=PANES_FIXTURE, stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="gone")

    monkeypatch.setattr(dashboard.subprocess, "run", enumerate_then_fail)
    client = TestClient(dashboard.create_app(tmux_session=SESSION))

    response = client.get("/console/pane/1.0")

    assert response.status_code == 200
    body = response.json()
    assert body["reachable"] is False
    assert body["content_html"] == ""


# ---- the page carries the switcher ------------------------------------------


def test_console_page_carries_the_switcher_strip():
    client = TestClient(dashboard.create_app(tmux_session=SESSION))

    response = client.get("/console")

    assert response.status_code == 200
    page = response.text
    assert 'id="pane-strip"' in page
    assert "/console/panes" in page
    assert "/console/pane/" in page
