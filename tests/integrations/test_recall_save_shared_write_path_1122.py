"""recall#1122: recall_save (server.py) and SynaptMemoryService._save_to_recall
(google_adk.py) both write through the shared save_knowledge_node helper
(knowledge.py) instead of each reimplementing the append_node +
upsert_knowledge_node pair. These tests guard the two guarantees that made
the extraction safe: the two callers agree on what gets persisted, and the
ADK integration's ambient default (project_root omitted) actually resolves
to the ambient store rather than silently landing nowhere -- the coverage
gap named in R1 on recall#1097 (nothing exercised SynaptMemoryService()
constructed without project_root against a real write).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("google.adk", reason="google-adk not installed")

from synapt.recall.knowledge import read_nodes
from synapt.recall.sharding import live_store_path
from synapt.recall.storage import RecallDB


def _read_db_node(index_dir: Path, node_id: str) -> dict | None:
    db = RecallDB(live_store_path(index_dir))
    try:
        return db.get_knowledge_node(node_id)
    finally:
        db.close()


def test_recall_save_and_the_adk_integration_persist_the_same_record_for_one_node(
    tmp_path, monkeypatch
):
    from synapt.integrations.google_adk import SynaptMemoryService, _node_id

    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)

    app_name, user_id, author = "app", "u1", "tester"
    text = "shared node content"
    content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    node_id = _node_id(app_name, user_id, content_hash)
    tags = ["google-adk", f"app:{app_name}", f"user:{user_id}", f"author:{author}"]

    # Path A: the ADK integration's real call, isolated to its own root.
    adk_root = tmp_path / "adk"
    svc = SynaptMemoryService(project_root=adk_root)
    svc._save_to_recall(app_name=app_name, user_id=user_id, text=text, author=author)

    # Path B: recall_save, given the SAME logical inputs the ADK path derives
    # (same content, category, tags, node_id), isolated to a DIFFERENT root
    # via SYNAPT_RECALL_ROOT so the two writes cannot collide or mask a bug
    # in either one's project-scoping.
    save_root = tmp_path / "save"
    save_root.mkdir()
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(save_root))
    from synapt.recall import server

    result = server.recall_save(
        content=text, category="workflow", tags=tags, node_id=node_id
    )
    assert "Error" not in result

    adk_knowledge = adk_root / ".synapt" / "recall" / "knowledge.jsonl"
    save_knowledge = save_root / ".synapt" / "recall" / "knowledge.jsonl"
    adk_jsonl = {n.id: n for n in read_nodes(adk_knowledge)}
    save_jsonl = {n.id: n for n in read_nodes(save_knowledge)}
    assert node_id in adk_jsonl and node_id in save_jsonl

    for field in ("id", "content", "category", "confidence"):
        assert getattr(adk_jsonl[node_id], field) == getattr(save_jsonl[node_id], field), field
    assert sorted(adk_jsonl[node_id].tags) == sorted(save_jsonl[node_id].tags)

    adk_index = adk_root / ".synapt" / "recall" / "index"
    save_index = save_root / ".synapt" / "recall" / "index"
    adk_db_node = _read_db_node(adk_index, node_id)
    save_db_node = _read_db_node(save_index, node_id)
    assert adk_db_node is not None and save_db_node is not None
    for field in ("id", "content", "category", "confidence"):
        assert adk_db_node[field] == save_db_node[field], field
    assert sorted(adk_db_node["tags"]) == sorted(save_db_node["tags"])


def test_adk_integration_with_no_project_root_writes_to_the_ambient_store(
    tmp_path, monkeypatch
):
    """Closes the coverage gap named in R1 on recall#1097 (m_5256f163): the
    module's own docstring shows SynaptMemoryService() -- no project_root --
    as the example usage, and nothing exercised that the resulting write
    actually lands in the AMBIENT store rather than some other or no
    location at all."""
    from synapt.integrations.google_adk import SynaptMemoryService

    ambient_root = tmp_path / "ambient"
    ambient_root.mkdir()
    monkeypatch.chdir(ambient_root)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(ambient_root))
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)

    svc = SynaptMemoryService()  # no project_root: the documented default usage
    assert svc._project_root is None

    svc._save_to_recall(
        app_name="app", user_id="u1", text="ambient default probe", author="tester"
    )

    knowledge_path = ambient_root / ".synapt" / "recall" / "knowledge.jsonl"
    assert knowledge_path.exists(), (
        "SynaptMemoryService() with no project_root did not write to the "
        "ambient store (SYNAPT_RECALL_ROOT) -- the documented default usage "
        "shape produced no observable write anywhere"
    )
    nodes = read_nodes(knowledge_path)
    assert len(nodes) == 1
    assert nodes[0].content == "ambient default probe"
