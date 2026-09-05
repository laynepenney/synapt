from __future__ import annotations

import os

import pytest

crewai_memory = pytest.importorskip("crewai.memory")
crewai_types = pytest.importorskip("crewai.memory.types")

CrewAIMemory = crewai_memory.Memory
MemoryMatch = crewai_types.MemoryMatch
MemoryRecord = crewai_types.MemoryRecord

# crewai's Memory defaults BOTH its vector embedder and its remember()/recall()
# classification path to a real LLM call (crewai>=1.4.0's unified_memory.py
# unconditionally builds an LLM instance inside _encode_batch, regardless of
# whether categories/importance are passed explicitly to remember(), and
# recall()'s default depth="deep" needs the LLM too for its RecallFlow) -- so
# there is no public kwarg path to make this suite hermetic without wiring a
# fake LLM into a test meant to exercise our real adapter logic. Skip cleanly
# rather than fail loudly when no provider credential is configured, matching
# the pattern used for the MLX-guarded tests elsewhere in this suite.
pytestmark = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    reason="crewai's Memory needs a real LLM + embedder credential "
    "(OPENAI_API_KEY or ANTHROPIC_API_KEY) for its default remember()/recall() "
    "path; no credential is configured here.",
)


def _adapter_cls():
    from synapt.integrations.crewai import SynaptMemory

    return SynaptMemory


def test_crewai_adapter_instantiates_and_matches_memory_interface(tmp_path):
    SynaptMemory = _adapter_cls()

    memory = SynaptMemory(
        path=tmp_path / "crewai.jsonl",
        root_scope="/crew/session-1",
    )

    assert isinstance(memory, CrewAIMemory)
    assert memory.root_scope == "/crew/session-1"
    assert memory.recall("anything", limit=3) == []


def test_crewai_adapter_remember_and_recall_round_trip(tmp_path):
    SynaptMemory = _adapter_cls()

    memory = SynaptMemory(
        path=tmp_path / "crewai.jsonl",
    )

    record = memory.remember(
        "The customer prefers JSON payloads and terse summaries.",
        metadata={"kind": "preference"},
    )
    matches = memory.recall("What output format does the customer prefer?", limit=3)

    assert isinstance(record, MemoryRecord)
    assert isinstance(matches, list)
    assert matches
    assert isinstance(matches[0], MemoryMatch)
    assert matches[0].record.content == "The customer prefers JSON payloads and terse summaries."


def test_crewai_adapter_persists_and_isolates_by_scope(tmp_path):
    SynaptMemory = _adapter_cls()
    shared_path = tmp_path / "crewai.jsonl"

    first = SynaptMemory(
        path=shared_path,
        root_scope="/crew/session-1",
    )
    first.remember("Escalations for Acme should go to Dana first.")

    reloaded = SynaptMemory(
        path=shared_path,
        root_scope="/crew/session-1",
    )
    other_scope = SynaptMemory(
        path=shared_path,
        root_scope="/crew/session-2",
    )

    same_scope_matches = reloaded.recall("Who handles Acme escalations?", limit=3)
    other_scope_matches = other_scope.recall("Who handles Acme escalations?", limit=3)

    assert same_scope_matches
    assert same_scope_matches[0].record.content == "Escalations for Acme should go to Dana first."
    assert other_scope_matches == []


def test_crewai_adapter_reset_clears_only_own_scope(tmp_path):
    SynaptMemory = _adapter_cls()
    shared_path = tmp_path / "crewai.jsonl"

    first = SynaptMemory(
        path=shared_path,
        root_scope="/crew/alpha",
    )
    second = SynaptMemory(
        path=shared_path,
        root_scope="/crew/beta",
    )

    first.remember("Alpha crew owns incident response.")
    second.remember("Beta crew owns roadmap planning.")

    first.reset()

    assert first.recall("incident response", limit=3) == []
    second_matches = second.recall("roadmap planning", limit=3)
    assert second_matches
    assert second_matches[0].record.content == "Beta crew owns roadmap planning."
