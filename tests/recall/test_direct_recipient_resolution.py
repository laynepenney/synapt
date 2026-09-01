"""Canonical recipient resolution for cross-runtime direct messaging."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from synapt.recall.direct import (
    RegisteredRecipient,
    ack_message,
    check_status,
    message_history,
    read_inbox,
    resolve_registered_recipient,
    send_message,
    speak_to_agent,
)
import synapt.recall.direct as direct


@pytest.fixture(autouse=True)
def _isolated_direct_store(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAPT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SYNAPT_SHARED_CHANNELS_DIR", str(tmp_path / "channels"))
    monkeypatch.setenv(
        "SYNAPT_AGENT_PANES",
        json.dumps({"anchor": {"target": "conversa:anchor", "runtime": "codex"}}),
    )
    monkeypatch.setattr("synapt.recall.channel._agent_id", lambda: "apollo-001")


@pytest.fixture
def one_anchor():
    return [RegisteredRecipient("anchor-001", "conversa", "Anchor")]


@pytest.mark.parametrize("spelling", ["anchor", "conversa:anchor", "anchor-001"])
def test_aliases_resolve_to_one_registered_stable_inbox(
    monkeypatch, one_anchor, spelling
):
    monkeypatch.setattr(
        "synapt.recall.direct._recipient_resolver",
        lambda target: resolve_registered_recipient(target, recipients=one_anchor),
    )
    calls: list[list[str]] = []

    def run(cmd, *, input=None):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("synapt.recall.direct._run_tmux", run)
    sent = speak_to_agent(action="send", to=spelling, message=f"through {spelling}")

    assert "Sent to anchor-001" in sent
    unread = read_inbox(agent_id="anchor-001")
    assert [message.to_agent for message in unread] == ["anchor-001"]
    assert any(
        cmd[:2] == ["tmux", "paste-buffer"] and "conversa:anchor" in cmd
        for cmd in calls
    )


def test_unknown_alias_is_rejected_before_a_phantom_inbox_is_created(
    monkeypatch, tmp_path, one_anchor
):
    monkeypatch.setattr(
        "synapt.recall.direct._recipient_resolver",
        lambda target: resolve_registered_recipient(target, recipients=one_anchor),
    )

    result = speak_to_agent(action="send", to="ghost", message="do not persist")

    assert "not registered" in result
    assert not (tmp_path / "channels" / "direct" / "ghost.jsonl").exists()
    assert not (
        tmp_path / "channels" / "_cross-org" / "direct" / "ghost.jsonl"
    ).exists()


def test_ambiguous_bare_alias_is_rejected_and_qualified_alias_narrows():
    recipients = [
        RegisteredRecipient("anchor-001", "conversa", "Anchor"),
        RegisteredRecipient("anchor-001", "other-org", "Anchor"),
    ]

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_registered_recipient("anchor", recipients=recipients)

    assert (
        resolve_registered_recipient("conversa:anchor", recipients=recipients)
        == recipients[0]
    )


def test_resolver_uses_the_injected_catalog_not_the_pane_map(monkeypatch):
    recipient = RegisteredRecipient("anchor-001", "conversa", "Anchor")
    monkeypatch.setattr(
        "synapt.recall.direct._recipient_resolver", lambda target: recipient
    )
    assert resolve_registered_recipient("conversa:anchor") == RegisteredRecipient(
        "anchor-001", "conversa", "Anchor"
    )


def test_grip_flat_record_resolves_alias_and_routes_with_its_live_runtime(
    tmp_path, monkeypatch
):
    record = {
        "gripspace": "conversa",
        "qualified_alias": "conversa:anchor",
        "agent_id": "anchor-001",
        "store_coordinate": "conversa",
        "target": "conversa:anchor",
        "runtime": "codex",
    }
    path = tmp_path / "agent-panes.json"
    path.write_text(json.dumps({"conversa:anchor": record, "anchor": record}))
    monkeypatch.setattr("synapt.recall.direct._recipient_resolver", None)
    monkeypatch.delenv("SYNAPT_AGENT_PANES", raising=False)
    monkeypatch.setenv("SYNAPT_AGENT_PANES_FILE", str(path))

    assert resolve_registered_recipient("anchor").agent_id == "anchor-001"
    from synapt.recall.direct import load_pane_map

    assert load_pane_map()["anchor-001"]["runtime"] == "codex"


def test_fresh_child_processes_use_generated_file_for_send_and_read(
    tmp_path, monkeypatch
):
    def record(alias, agent_id):
        return {
            "gripspace": "synapt",
            "qualified_alias": f"synapt:{alias}",
            "agent_id": agent_id,
            "store_coordinate": "synapt",
            "target": f"synapt:{alias}",
            "runtime": "codex",
        }

    routing = tmp_path / "agent-panes.json"
    routing.write_text(
        json.dumps(
            {
                "apollo": record("apollo", "apollo-001"),
                "anchor": record("anchor", "anchor-001"),
            }
        )
    )
    shared = tmp_path / "channels"
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        "SYNAPT_AGENT_PANES_FILE": str(routing),
        "SYNAPT_SHARED_CHANNELS_DIR": str(shared),
    }
    sender = subprocess.run(
        [
            sys.executable,
            "-c",
            "from synapt.recall.direct import speak_to_agent; print(speak_to_agent(action='send', to='anchor', message='fresh child'))",
        ],
        env={**env, "SYNAPT_AGENT_ID": "apollo-001"},
        text=True,
        capture_output=True,
        check=True,
    )
    reader = subprocess.run(
        [
            sys.executable,
            "-c",
            "from synapt.recall.direct import speak_to_agent; print(speak_to_agent(action='read'))",
        ],
        env={**env, "SYNAPT_AGENT_ID": "anchor-001"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Sent to anchor-001" in sender.stdout
    assert "fresh child" in reader.stdout


def test_fresh_child_uses_one_generated_record_when_inline_panes_conflict(tmp_path):
    """Configured identity cannot combine with an inline tmux route."""
    generated = {
        "gripspace": "synapt",
        "qualified_alias": "synapt:stromus",
        "agent_id": "stromus-001",
        "store_coordinate": "synapt",
        "target": "generated-stromus:0",
        "runtime": "codex",
    }
    routing = tmp_path / "agent-panes.json"
    routing.write_text(json.dumps({"synapt:stromus": generated, "stromus": generated}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        "SYNAPT_AGENT_ID": "apollo-001",
        "SYNAPT_AGENT_PANES_FILE": str(routing),
        "SYNAPT_AGENT_PANES": json.dumps(
            {"stromus": {"target": "inline-stromus:0", "runtime": "claude"}}
        ),
        "SYNAPT_SHARED_CHANNELS_DIR": str(tmp_path / "channels"),
    }
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from synapt.recall.direct import load_pane_map, resolve_pane, resolve_registered_recipient, speak_to_agent; r = resolve_registered_recipient('stromus'); p = resolve_pane(r.agent_id, load_pane_map()); print(f'{r.agent_id}|{r.store_coordinate}|{p.target}|{p.runtime}'); print(speak_to_agent(action='send', to='stromus', message='co-configured source'))",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    lines = child.stdout.splitlines()
    assert lines[0] == "stromus-001|synapt|generated-stromus:0|codex"
    assert lines[1].startswith("Sent to stromus-001:")


@pytest.mark.parametrize("coordinate", ["../escape", "/tmp", "a/b", ".", "a..b"])
def test_invalid_store_coordinate_refuses_before_store_access(coordinate):
    with pytest.raises(ValueError, match="invalid recipient store coordinate"):
        send_message(
            from_agent="anchor-001",
            to_agent="stromus-001",
            body="blocked",
            recipient_store_coordinate=coordinate,
        )


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        (
            "recipient store coordinate",
            {"to_agent": "stromus-001", "recipient_store_coordinate": "../escape"},
        ),
        ("agent ID", {"to_agent": "../escape", "recipient_store_coordinate": "synapt"}),
    ],
)
def test_send_rejects_path_fields_before_hooks_or_storage(
    tmp_path, monkeypatch, field, kwargs
):
    """Neither malformed path field may reach an inbox, DB, canonical root, or hook."""
    calls: list[str] = []

    def forbidden(name):
        def _call(*args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"unexpected side effect: {name}")

        return _call

    monkeypatch.setattr(direct, "_inbox_path", forbidden("inbox"))
    monkeypatch.setattr(direct, "_cross_org_inbox_path", forbidden("canonical inbox"))
    monkeypatch.setattr(direct, "_get_db", forbidden("local db"))
    monkeypatch.setattr(direct, "_get_canonical_db", forbidden("canonical db"))
    direct.register_before_send_hook(forbidden("before send hook"))
    try:
        with pytest.raises(ValueError, match=f"invalid {field}"):
            send_message(from_agent="anchor-001", body="blocked", **kwargs)
    finally:
        direct._clear_hooks()

    assert calls == []
    assert not (tmp_path / "channels").exists()


@pytest.mark.parametrize(
    ("operation", "kwargs", "error"),
    [
        (
            read_inbox,
            {"agent_id": "../escape", "recipient_store_coordinate": "synapt"},
            "invalid agent ID",
        ),
        (
            read_inbox,
            {"agent_id": "stromus-001", "recipient_store_coordinate": "../escape"},
            "invalid recipient store coordinate",
        ),
        (
            ack_message,
            {
                "message_id": "dm_test",
                "agent_id": "../escape",
                "recipient_store_coordinate": "synapt",
            },
            "invalid agent ID",
        ),
        (
            ack_message,
            {
                "message_id": "dm_test",
                "agent_id": "stromus-001",
                "recipient_store_coordinate": "../escape",
            },
            "invalid recipient store coordinate",
        ),
        (
            message_history,
            {
                "agent_id": "../escape",
                "with_agent": "anchor-001",
                "include_canonical": True,
            },
            "invalid agent ID",
        ),
        (
            message_history,
            {
                "agent_id": "stromus-001",
                "with_agent": "../escape",
                "include_canonical": True,
            },
            "invalid agent ID",
        ),
        (
            message_history,
            {
                "agent_id": "stromus-001",
                "with_agent": "anchor-001",
                "include_canonical": True,
                "canonical_store_coordinate": "../escape",
            },
            "invalid recipient store coordinate",
        ),
    ],
)
def test_read_ack_and_history_reject_path_fields_before_storage(
    monkeypatch, operation, kwargs, error
):
    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("storage")
        raise AssertionError("unexpected storage access")

    monkeypatch.setattr(direct, "_get_db", forbidden)
    monkeypatch.setattr(direct, "_get_canonical_db", forbidden)

    with pytest.raises(ValueError, match=error):
        operation(**kwargs)
    assert calls == []


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("agent_id", "../escape", "invalid agent ID"),
        ("store_coordinate", "../escape", "invalid recipient store coordinate"),
    ],
)
def test_configured_file_traversal_rejects_before_a_send_can_touch_storage(
    tmp_path, monkeypatch, field, value, error
):
    record = {
        "gripspace": "conversa",
        "qualified_alias": "conversa:anchor",
        "agent_id": "anchor-001",
        "store_coordinate": "conversa",
        "target": "conversa:anchor",
        "runtime": "codex",
    }
    record[field] = value
    routing = tmp_path / "agent-panes.json"
    routing.write_text(json.dumps({"anchor": record}))
    monkeypatch.setattr(direct, "_recipient_resolver", None)
    monkeypatch.delenv("SYNAPT_AGENT_PANES", raising=False)
    monkeypatch.setenv("SYNAPT_AGENT_PANES_FILE", str(routing))

    result = speak_to_agent(action="send", to="anchor", message="blocked")

    assert result == f"Error: {error}"
    assert not (tmp_path / "channels").exists()


def test_two_gripspaces_converge_on_recipient_owned_state_and_history(
    tmp_path, monkeypatch
):
    """Anchor's send and Stromus's read/ack share one recipient-owned record."""
    monkeypatch.delenv("SYNAPT_SHARED_CHANNELS_DIR", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("synapt.recall.direct.Path.home", classmethod(lambda cls: home))
    anchor_space = tmp_path / "conversa-gripspace"
    stromus_space = tmp_path / "synapt-gripspace"
    anchor_space.mkdir()
    stromus_space.mkdir()

    message = send_message(
        from_agent="anchor-001",
        to_agent="stromus-001",
        body="cross-runtime handoff",
        project_dir=anchor_space,
        recipient_store_coordinate="synapt",
    )

    fresh_history = message_history(
        agent_id="anchor-001",
        with_agent="stromus-001",
        project_dir=anchor_space,
        include_canonical=True,
        canonical_store_coordinate="synapt",
    )
    assert [item.message_id for item in fresh_history] == [message.message_id]

    received = read_inbox(
        agent_id="stromus-001",
        project_dir=stromus_space,
        recipient_store_coordinate="synapt",
    )
    assert [item.message_id for item in received] == [message.message_id]
    assert "acknowledged" in ack_message(
        message_id=message.message_id,
        agent_id="stromus-001",
        project_dir=stromus_space,
        recipient_store_coordinate="synapt",
    )
    assert (
        check_status(message_id=message.message_id, project_dir=anchor_space)["status"]
        == "acked"
    )

    anchor_history = message_history(
        agent_id="anchor-001",
        with_agent="stromus-001",
        project_dir=anchor_space,
        include_canonical=True,
    )
    stromus_history = message_history(
        agent_id="stromus-001",
        with_agent="anchor-001",
        project_dir=stromus_space,
        include_canonical=True,
    )
    assert [item.message_id for item in anchor_history] == [message.message_id]
    assert [item.message_id for item in stromus_history] == [message.message_id]
