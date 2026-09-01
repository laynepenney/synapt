"""Canonical recipient resolution for cross-runtime direct messaging."""

from __future__ import annotations

import json
from types import SimpleNamespace

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
