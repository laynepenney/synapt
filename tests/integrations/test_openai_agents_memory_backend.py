"""Red specs for the OpenAI Agents session backend."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

pytest.importorskip("agents", reason="requires the optional openai-agents integration")

from agents.memory.session import Session
from agents.memory.session_settings import SessionSettings


def _load_synapt_session():
    try:
        module = importlib.import_module("synapt.integrations.openai_agents")
    except ModuleNotFoundError:
        pytest.fail(
            "Expected backend module `synapt.integrations.openai_agents` with "
            "`SynaptSession`, but it does not exist yet."
        )
    except Exception as exc:  # pragma: no cover - exercised once backend exists
        pytest.fail(f"Importing `synapt.integrations.openai_agents` failed: {exc!r}")

    try:
        return module.SynaptSession
    except AttributeError:
        pytest.fail(
            "Expected `synapt.integrations.openai_agents.SynaptSession`, "
            "but the symbol is missing."
        )


def _new_session(
    tmp_path: Path,
    session_id: str = "session-1",
    *,
    session_settings: SessionSettings | None = None,
):
    cls = _load_synapt_session()
    try:
        return cls(
            session_id=session_id,
            db_path=tmp_path / "synapt-openai-session.db",
            session_settings=session_settings,
        )
    except TypeError:
        pytest.fail(
            "SynaptSession should accept the SQLiteSession-style constructor "
            "`(session_id, db_path=..., session_settings=...)`."
        )


def test_synapt_session_satisfies_the_openai_session_protocol(tmp_path: Path) -> None:
    session = _new_session(tmp_path)

    assert isinstance(session, Session)
    assert session.session_id == "session-1"


def test_synapt_session_round_trips_items_in_chronological_order(tmp_path: Path) -> None:
    session = _new_session(tmp_path)
    items = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]

    asyncio.run(session.add_items(items))
    stored = asyncio.run(session.get_items())

    assert stored == items


def test_synapt_session_honors_explicit_and_default_limits(tmp_path: Path) -> None:
    session = _new_session(
        tmp_path,
        session_settings=SessionSettings(limit=2),
    )
    items = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]

    asyncio.run(session.add_items(items))

    assert asyncio.run(session.get_items()) == items[-2:]
    assert asyncio.run(session.get_items(limit=1)) == items[-1:]


def test_synapt_session_pop_item_matches_sqlite_session_lifo_behavior(tmp_path: Path) -> None:
    session = _new_session(tmp_path)
    items = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]

    asyncio.run(session.add_items(items))

    popped = asyncio.run(session.pop_item())
    remaining = asyncio.run(session.get_items())

    assert popped == items[-1]
    assert remaining == items[:-1]


def test_synapt_session_clear_session_removes_all_history(tmp_path: Path) -> None:
    session = _new_session(tmp_path)
    asyncio.run(session.add_items([{"role": "user", "content": "ephemeral"}]))

    asyncio.run(session.clear_session())

    assert asyncio.run(session.get_items()) == []
    assert asyncio.run(session.pop_item()) is None


def test_synapt_session_persists_across_instances_like_sqlite_session(tmp_path: Path) -> None:
    first = _new_session(tmp_path, session_id="shared")
    asyncio.run(first.add_items([{"role": "user", "content": "persist me"}]))

    second = _new_session(tmp_path, session_id="shared")
    restored = asyncio.run(second.get_items())

    assert restored == [{"role": "user", "content": "persist me"}]
