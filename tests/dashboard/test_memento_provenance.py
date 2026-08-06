"""Honesty properties of the memento provenance chips + the Leonard Test.

The whole thesis is *persistence with provenance* — so the one thing these
endpoints must never do is fabricate a `who` or invent a chip for a being
that has authored nothing. These tests pin exactly that, against the real
routes, by pointing ``project_data_dir`` at a temp journal store via cwd.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from synapt.dashboard.app import create_app


def _write_journal(worktrees: Path, dirname: str, entries: list[dict]) -> None:
    d = worktrees / dirname
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "journal.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    # the /memento helpers resolve project_data_dir(None) -> cwd each request,
    # so chdir'ing into a bare temp root points them at our fixture store.
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app())


def test_provenance_who_is_never_fabricated(tmp_path, monkeypatch):
    wt = tmp_path / ".synapt" / "recall" / "worktrees"
    _write_journal(wt, "wt-a", [
        {"timestamp": "2026-07-01T10:00:00+00:00", "session_id": "sess1234abcd",
         "agent_id": "opus-001", "griptree": "synapt",
         "decisions": ["Ship the widget (repo#1)"], "next_steps": ["do the thing"]},
    ])
    _write_journal(wt, "wt-b", [
        {"timestamp": "2026-07-02T10:00:00+00:00", "session_id": "sess5678efgh",
         "agent_id": "apollo-001", "griptree": "synapt-dev",
         "decisions": ["Apollo decided something else entirely"], "next_steps": []},
    ])
    c = _client(tmp_path, monkeypatch)
    items = c.get("/memento/provenance/opus?limit=5").json()

    assert items, "opus authored a decision, expected chips"
    # who is copied from the entry's own agent_id, never inferred or cross-attributed
    assert all(it["who"] == "opus" for it in items)
    # apollo's decision must never surface under opus
    assert all("Apollo decided" not in it["what"] for it in items)
    # where prefers a real ref the author actually wrote
    assert any(it["where"] == "repo#1" for it in items)


def test_provenance_empty_is_honest(tmp_path, monkeypatch):
    (tmp_path / ".synapt" / "recall" / "worktrees").mkdir(parents=True)
    c = _client(tmp_path, monkeypatch)
    # a being that has authored nothing gets NOTHING, never an invented chip
    assert c.get("/memento/provenance/nemo").json() == []


def test_provenance_strips_leading_bullet(tmp_path, monkeypatch):
    wt = tmp_path / ".synapt" / "recall" / "worktrees"
    _write_journal(wt, "wt-a", [
        {"timestamp": "2026-07-01T10:00:00+00:00", "session_id": "s1",
         "agent_id": "opus-001", "decisions": ["- A decision written as a bullet"]},
    ])
    c = _client(tmp_path, monkeypatch)
    items = c.get("/memento/provenance/opus").json()
    assert items and not items[0]["what"].startswith("-")


def test_leonard_reachable_false_when_no_memory(tmp_path, monkeypatch):
    (tmp_path / ".synapt" / "recall" / "worktrees").mkdir(parents=True)
    c = _client(tmp_path, monkeypatch)
    d = c.get("/memento/leonard/nemo").json()
    assert d["reachable"] is False
    assert d["act1"] is None
    assert d["act3"] == []


def test_leonard_three_acts_from_real_memory(tmp_path, monkeypatch):
    wt = tmp_path / ".synapt" / "recall" / "worktrees"
    _write_journal(wt, "wt-a", [
        # an older, now-dead session — act 1's decision must outlive it
        {"timestamp": "2026-05-13T10:00:00+00:00", "session_id": "olddead1xyz",
         "agent_id": "opus-001", "decisions": ["Coin a name for the framework"], "next_steps": []},
        # a newer session with a rejection and a next-step
        {"timestamp": "2026-07-26T10:00:00+00:00", "session_id": "newsess1xyz",
         "agent_id": "opus-001",
         "decisions": ["Keep the gizmo blue not green"],
         "next_steps": ["Collect the sign-offs"]},
    ])
    c = _client(tmp_path, monkeypatch)
    # smart=0 pins the deterministic template questions (the 3B path is live/nondeterministic
    # and would load the model); this test verifies structure + attribution, not phrasing.
    d = c.get("/memento/leonard/opus?smart=0").json()

    assert d["reachable"] is True
    # act 1 pulls from the PRIOR (older) session, not the newest one
    assert d["act1"] is not None
    assert d["act1"]["session"].startswith("olddead1")
    assert "Coin a name" in d["act1"]["chip"]["what"]
    # act 3 answers, every one attributed to opus, with a rejection question present
    assert d["act3"], "expected remembering Q&A"
    assert all(item["chip"]["who"] == "opus" for item in d["act3"])
    assert any("rejected" in item["q"].lower() for item in d["act3"])


def test_leonard_asof_scopes_and_is_honest(tmp_path, monkeypatch):
    wt = tmp_path / ".synapt" / "recall" / "worktrees"
    _write_journal(wt, "wt-a", [
        {"timestamp": "2026-06-15T10:00:00+00:00", "session_id": "s1",
         "agent_id": "opus-001", "decisions": ["A decision made on June 15"], "next_steps": []},
    ])
    c = _client(tmp_path, monkeypatch)
    # as of before the only entry: nothing remembered yet — honest, never invented
    before = c.get("/memento/leonard/opus?asof=2026-06-01&smart=0").json()
    assert before["reachable"] is False
    assert before["act1"] is None
    # as of after: the memory is present and correctly scoped
    after = c.get("/memento/leonard/opus?asof=2026-07-01&smart=0").json()
    assert after["reachable"] is True
    assert "June 15" in after["act1"]["chip"]["what"]
