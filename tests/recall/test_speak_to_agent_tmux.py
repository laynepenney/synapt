"""Contract for speak_to_agent tmux delivery (recall#852).

speak_to_agent used to write a passive inbox the recipient had to poll. This locks
the fix: on send it ALSO resolves the recipient to a tmux pane and delivers via
load-buffer + paste-buffer + send-keys, with a runtime-aware Enter count
(Claude=2, Codex=3) and an extra guarded Enter for large pastes that collapse.

Boundary discipline (locked here): the tmux MECHANICS are OSS transport; the
agent->pane map is operator-supplied data read from a neutral env/config seam, NOT
hardcoded identity topology. resolve_pane on an empty map returns None -> the send
falls back to inbox-only, never raising.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from synapt.recall.direct import (
    LARGE_PASTE_THRESHOLD,
    PaneTarget,
    RegisteredRecipient,
    TmuxDelivery,
    build_tmux_commands,
    deliver_via_tmux,
    enter_count,
    load_pane_map,
    resolve_pane,
    speak_to_agent,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAPT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SYNAPT_SHARED_CHANNELS_DIR", str(tmp_path / "channels"))
    (tmp_path / "channels" / "direct").mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("SYNAPT_AGENT_PANES", raising=False)
    monkeypatch.delenv("SYNAPT_AGENT_PANES_FILE", raising=False)
    yield


# ── runtime-aware Enter count ──────────────────────────────────────


def test_enter_count_claude_is_two():
    assert enter_count("claude") == 2


def test_enter_count_codex_is_three():
    assert enter_count("codex") == 3


def test_enter_count_unknown_runtime_defaults_to_two():
    assert enter_count("something-else") == 2
    assert enter_count(None) == 2


# ── pane map (neutral operator-supplied seam) ──────────────────────


def test_load_pane_map_from_env_json(monkeypatch):
    monkeypatch.setenv(
        "SYNAPT_AGENT_PANES",
        json.dumps({"apollo": {"target": "synapt:3", "runtime": "claude"}}),
    )
    pane_map = load_pane_map()
    assert pane_map["apollo"]["target"] == "synapt:3"


def test_load_pane_map_defaults_empty_when_unset():
    assert load_pane_map() == {}


def test_resolve_pane_exact_and_normalized():
    pane_map = {
        "apollo": {"target": "synapt:3", "runtime": "claude"},
        "atlas": {"target": "synapt:4", "runtime": "codex"},
    }
    assert resolve_pane("apollo", pane_map) == PaneTarget(
        target="synapt:3", runtime="claude"
    )
    # case-insensitive + org-prefixed recipient both normalize to the bare agent key
    assert resolve_pane("Apollo", pane_map) == PaneTarget(
        target="synapt:3", runtime="claude"
    )
    assert resolve_pane("synapt:atlas", pane_map) == PaneTarget(
        target="synapt:4", runtime="codex"
    )


def test_resolve_pane_missing_returns_none():
    assert (
        resolve_pane("ghost", {"apollo": {"target": "synapt:3", "runtime": "claude"}})
        is None
    )
    assert resolve_pane("apollo", {}) is None


# ── tmux command sequence ──────────────────────────────────────────


def test_build_tmux_commands_sequence_for_claude():
    cmds, enters = build_tmux_commands("synapt:3", "claude", "hi", buffer_name="b")
    assert enters == 2
    assert cmds[0] == ["tmux", "load-buffer", "-b", "b", "-"]  # body piped via stdin
    assert cmds[1] == ["tmux", "paste-buffer", "-t", "synapt:3", "-b", "b", "-d"]
    send_keys = [c for c in cmds if c[:2] == ["tmux", "send-keys"]]
    assert send_keys == [["tmux", "send-keys", "-t", "synapt:3", "Enter"]] * 2


def test_build_tmux_commands_codex_gets_three_enters():
    cmds, enters = build_tmux_commands("synapt:4", "codex", "hi", buffer_name="b")
    assert enters == 3
    assert sum(1 for c in cmds if c[:2] == ["tmux", "send-keys"]) == 3


def test_build_tmux_commands_large_paste_adds_one_extra_enter():
    big = "x" * (LARGE_PASTE_THRESHOLD + 1)
    _cmds, enters_claude = build_tmux_commands(
        "synapt:3", "claude", big, buffer_name="b"
    )
    _cmds2, enters_codex = build_tmux_commands(
        "synapt:4", "codex", big, buffer_name="b"
    )
    assert enters_claude == 3  # 2 + 1 guarded for collapse
    assert enters_codex == 4  # 3 + 1


# ── delivery via injected runner ───────────────────────────────────


def _recorder():
    calls = []

    def run(cmd, *, input=None):
        calls.append((cmd, input))
        return SimpleNamespace(returncode=0, stderr="")

    return run, calls


def test_deliver_via_tmux_pipes_body_and_runs_full_sequence():
    run, calls = _recorder()
    sleeps = []
    result = deliver_via_tmux(
        "synapt:3",
        "claude",
        "hello world",
        run=run,
        sleep=sleeps.append,
        buffer_name="b",
    )
    assert isinstance(result, TmuxDelivery)
    assert result.delivered is True
    assert result.target == "synapt:3"
    assert result.enters == 2
    # body is piped to load-buffer's stdin, not shell-escaped into argv
    load = [c for c in calls if c[0][:2] == ["tmux", "load-buffer"]][0]
    assert load[1] == "hello world"
    # paste then the two Enters actually ran
    assert sum(1 for c, _ in calls if c[:2] == ["tmux", "send-keys"]) == 2
    assert sleeps  # guarded sleeps between expand/submit


def test_deliver_via_tmux_reports_failure_gracefully():
    def run(cmd, *, input=None):
        if cmd[:2] == ["tmux", "paste-buffer"]:
            return SimpleNamespace(returncode=1, stderr="can't find pane: synapt:9")
        return SimpleNamespace(returncode=0, stderr="")

    result = deliver_via_tmux(
        "synapt:9", "claude", "hi", run=run, sleep=lambda *_: None, buffer_name="b"
    )
    assert result.delivered is False
    assert "synapt:9" in result.detail


def test_deliver_via_tmux_survives_missing_tmux_binary():
    def run(cmd, *, input=None):
        raise FileNotFoundError("tmux")

    result = deliver_via_tmux(
        "synapt:3", "claude", "hi", run=run, sleep=lambda *_: None, buffer_name="b"
    )
    assert result.delivered is False  # no raise


# ── speak_to_agent integration: inbox ALWAYS, tmux when pane resolves ──


def test_speak_to_agent_send_delivers_via_tmux_and_keeps_inbox(monkeypatch):
    monkeypatch.setattr(
        "synapt.recall.channel._agent_id", lambda: "opus", raising=False
    )
    monkeypatch.setattr(
        "synapt.recall.direct.resolve_registered_recipient",
        lambda target: RegisteredRecipient("apollo-001", "synapt-dev", "Apollo"),
    )
    monkeypatch.setenv(
        "SYNAPT_AGENT_PANES",
        json.dumps({"apollo": {"target": "synapt:3", "runtime": "claude"}}),
    )
    run, calls = _recorder()
    monkeypatch.setattr("synapt.recall.direct._run_tmux", run)

    out = speak_to_agent(action="send", to="apollo", message="ping")

    # durable inbox write still happened (belt + suspenders)
    from synapt.recall.direct import read_inbox

    assert read_inbox(agent_id="apollo-001")  # canonical inbox has an unread message
    # AND tmux delivery was attempted to apollo's pane
    assert any(
        c[0][:2] == ["tmux", "paste-buffer"] and "synapt:3" in c[0] for c in calls
    )
    assert "synapt:3" in out or "tmux" in out.lower()


def test_speak_to_agent_send_inbox_only_when_no_pane(monkeypatch):
    monkeypatch.setattr(
        "synapt.recall.channel._agent_id", lambda: "opus", raising=False
    )
    monkeypatch.setattr(
        "synapt.recall.direct.resolve_registered_recipient",
        lambda target: RegisteredRecipient("apollo-001", "synapt-dev", "Apollo"),
    )
    # no SYNAPT_AGENT_PANES -> empty map -> no pane resolves
    called = []
    monkeypatch.setattr(
        "synapt.recall.direct._run_tmux", lambda cmd, **k: called.append(cmd)
    )

    out = speak_to_agent(action="send", to="apollo", message="ping")

    from synapt.recall.direct import read_inbox

    assert read_inbox(agent_id="apollo-001")  # canonical inbox still written
    assert called == []  # no tmux attempted
    assert "inbox" in out.lower()
