"""Cross-org direct-messaging silo fix (recall#820).

The bug: ``speak_to_agent`` reports "Delivered" on a cross-org send but the
recipient sees nothing on their inbox-read. Root cause: both the JSONL inbox
and the SQLite delivery-state store live under the gripspace-local
``_channels_dir(project_dir)``. A send from gripspace A lands in A's store; a
read from gripspace B queries B's store and finds nothing. Three months of
cross-org direct messages may be sitting in siloed inboxes the recipient never
reads.

The fix (Option A): a project-independent, org-canonical cross-org root that
``send_message`` dual-writes to and ``read_inbox`` union-reads from, deduping by
``message_id``.

These tests reproduce the silo by NOT setting ``SYNAPT_SHARED_CHANNELS_DIR``
(so local stores resolve per-``project_dir`` via the Tier-3 path) and isolating
the home-anchored cross-org root via ``monkeypatch``. This is the silo the
existing ``test_direct.py`` suite cannot reproduce: its autouse fixture sets
``SYNAPT_SHARED_CHANNELS_DIR``, which collapses every ``project_dir`` to one
shared store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synapt.recall.direct import (
    STATUS_ACKED,
    _clear_hooks,
    ack_message,
    check_status,
    read_inbox,
    send_message,
)


@pytest.fixture(autouse=True)
def _cross_org_isolation(tmp_path, monkeypatch):
    """Isolate the cross-org root without collapsing per-gripspace local stores.

    No ``SYNAPT_SHARED_CHANNELS_DIR``: local stores resolve per-``project_dir``
    (Tier 3), which reproduces the cross-gripspace silo. The cross-org root is
    home-anchored, so redirect ``Path.home`` into tmp to keep the test from
    touching the real ``~/.synapt``.
    """
    monkeypatch.delenv("SYNAPT_SHARED_CHANNELS_DIR", raising=False)
    monkeypatch.setenv("SYNAPT_DATA_DIR", str(tmp_path))
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    _clear_hooks()
    yield


@pytest.fixture
def gripspaces(tmp_path):
    """Two distinct gripspace project dirs -> two distinct local stores."""
    a = tmp_path / "gripspace_a"
    b = tmp_path / "gripspace_b"
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)
    return a, b


class TestCrossOrgPathResolution:
    def test_inbox_path_is_project_independent(self) -> None:
        # Lazy import: the symbol does not exist until the fix lands (RED).
        from synapt.recall.direct import _cross_org_inbox_path

        p_a = _cross_org_inbox_path("agent-b", project_dir=Path("/tmp/gs_a"))
        p_b = _cross_org_inbox_path("agent-b", project_dir=Path("/tmp/gs_b"))
        # Same recipient -> same canonical inbox regardless of sender gripspace.
        assert p_a == p_b

    def test_inbox_path_is_recipient_keyed_under_cross_org_root(self) -> None:
        from synapt.recall.direct import _cross_org_inbox_path

        p = _cross_org_inbox_path("agent-b")
        assert p.name == "agent-b.jsonl"
        assert p.parent.name == "direct"
        assert p.parent.parent.name == "_cross-org"


class TestCrossOrgDelivery:
    def test_cross_org_roundtrip(self, gripspaces) -> None:
        """The core silo fix: send from gripspace A, read from gripspace B."""
        gs_a, gs_b = gripspaces
        send_message(
            from_agent="agent-a",
            to_agent="agent-b",
            body="cross-org message",
            project_dir=gs_a,
        )
        inbox = read_inbox(agent_id="agent-b", project_dir=gs_b)
        assert len(inbox) == 1
        assert inbox[0].body == "cross-org message"
        assert inbox[0].from_agent == "agent-a"

    def test_cross_org_jsonl_written_at_canonical_path(self, gripspaces) -> None:
        from synapt.recall.direct import _cross_org_inbox_path

        gs_a, _ = gripspaces
        send_message(
            from_agent="agent-a",
            to_agent="agent-b",
            body="canonical write",
            project_dir=gs_a,
        )
        canonical = _cross_org_inbox_path("agent-b", project_dir=gs_a)
        assert canonical.exists()
        data = json.loads(canonical.read_text().strip().split("\n")[0])
        assert data["from"] == "agent-a"
        assert data["to"] == "agent-b"
        assert data["body"] == "canonical write"

    def test_cross_org_read_marks_read(self, gripspaces) -> None:
        """A second cross-org read returns empty (message tracked READ locally)."""
        gs_a, gs_b = gripspaces
        send_message(
            from_agent="agent-a",
            to_agent="agent-b",
            body="read once",
            project_dir=gs_a,
        )
        first = read_inbox(agent_id="agent-b", project_dir=gs_b)
        assert len(first) == 1
        second = read_inbox(agent_id="agent-b", project_dir=gs_b)
        assert len(second) == 0

    def test_cross_org_ack_after_read(self, gripspaces) -> None:
        """Ack works cross-org once the message has been read (local tracking row)."""
        gs_a, gs_b = gripspaces
        msg = send_message(
            from_agent="agent-a",
            to_agent="agent-b",
            body="ack me",
            project_dir=gs_a,
        )
        read_inbox(agent_id="agent-b", project_dir=gs_b)
        result = ack_message(
            message_id=msg.message_id, agent_id="agent-b", project_dir=gs_b
        )
        assert "acknowledged" in result
        status = check_status(message_id=msg.message_id, project_dir=gs_b)
        assert status is not None
        assert status["status"] == STATUS_ACKED

    def test_no_duplicate_when_sender_reads_own_gripspace(self, gripspaces) -> None:
        """Same-gripspace read dedups: the message lives in BOTH the local SQLite
        row and the cross-org JSONL, but read_inbox returns it exactly once."""
        gs_a, _ = gripspaces
        send_message(
            from_agent="agent-a",
            to_agent="agent-b",
            body="no dupes",
            project_dir=gs_a,
        )
        inbox = read_inbox(agent_id="agent-b", project_dir=gs_a)
        assert len(inbox) == 1

    def test_cross_org_dedup_across_repeated_reads(self, gripspaces) -> None:
        """Sending twice yields two messages; reading does not multiply them."""
        gs_a, gs_b = gripspaces
        send_message(
            from_agent="agent-a", to_agent="agent-b", body="m1", project_dir=gs_a
        )
        send_message(
            from_agent="agent-a", to_agent="agent-b", body="m2", project_dir=gs_a
        )
        inbox = read_inbox(agent_id="agent-b", project_dir=gs_b)
        bodies = sorted(m.body for m in inbox)
        assert bodies == ["m1", "m2"]

    def test_cross_org_not_resurfaced_on_second_read(self, gripspaces) -> None:
        """A cross-org message read once is not re-surfaced on a later read.

        The first read writes a local SQLite tracking row (status=READ); a
        second read must find the canonical message already tracked and skip
        it. The local DELIVERED-only query cannot see that READ row, so the
        only thing preventing re-surface is the message_id-present check in
        ``_read_cross_org_candidates``. Without it the message returns on every
        read. (The sibling ``dedup_across_repeated_reads`` test reads only once
        and does not exercise this path.)
        """
        gs_a, gs_b = gripspaces
        send_message(
            from_agent="agent-a", to_agent="agent-b", body="once", project_dir=gs_a
        )
        first = read_inbox(agent_id="agent-b", project_dir=gs_b)
        assert [m.body for m in first] == ["once"]
        second = read_inbox(agent_id="agent-b", project_dir=gs_b)
        assert second == []


class TestIntraOrgBackwardCompat:
    def test_intra_org_send_read_unchanged(self, gripspaces) -> None:
        """Same project_dir send+read still works (no regression from dual-write)."""
        gs_a, _ = gripspaces
        send_message(
            from_agent="agent-c",
            to_agent="agent-b",
            body="intra-org still works",
            project_dir=gs_a,
        )
        inbox = read_inbox(agent_id="agent-b", project_dir=gs_a)
        assert len(inbox) == 1
        assert inbox[0].body == "intra-org still works"
