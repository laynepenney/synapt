"""MCP empty results state what index surface they observed.

Search and context are allowed to return no content. They are not allowed to
make that absence read as a statement about history until they identify the
resolved index root and the freshness surface that was checked.
"""

from __future__ import annotations

from types import SimpleNamespace

from synapt.recall import freshness, server
from synapt.recall.freshness import IndexFreshness


def _verdict(*, stale: bool, scanned: str) -> IndexFreshness:
    return IndexFreshness(
        stale=stale,
        build_timestamp="2026-08-07T12:00:00Z",
        scanned=scanned,
        new_files=["rollout-live.jsonl"] if stale else [],
        remedy="synapt recall build --no-embeddings",
    )


def _empty_index() -> SimpleNamespace:
    return SimpleNamespace(
        _embedding_status="available",
        _last_diagnostics=None,
        _last_conflicts=[],
        lookup=lambda *args, **kwargs: "",
        read_turn_context=lambda chunk_id: "",
        _db=SimpleNamespace(get_cluster_chunks=lambda cluster_id: []),
    )


def _wire_empty_index(monkeypatch, tmp_path):
    index_dir = tmp_path / "resolved" / ".synapt" / "recall" / "index"
    index_dir.mkdir(parents=True)
    monkeypatch.setattr(server, "_get_index", lambda **kwargs: _empty_index())
    monkeypatch.setattr(server, "project_index_dir", lambda: index_dir)
    monkeypatch.setattr("synapt.recall.live.search_live_transcript", lambda *args, **kwargs: "")
    return index_dir


def test_empty_search_escalates_to_deep_and_labels_the_resolved_root(monkeypatch, tmp_path):
    index_dir = _wire_empty_index(monkeypatch, tmp_path)
    calls: list[bool] = []

    def check(*args, **kwargs):
        calls.append(kwargs.get("deep", False))
        return _verdict(stale=kwargs.get("deep", False), scanned=(
            "archive+sources" if kwargs.get("deep", False) else "archive"
        ))

    monkeypatch.setattr(freshness, "check_index_freshness", check)

    result = server.recall_search("absent synthetic fact")

    assert calls == [False, True]
    assert "STALE" in result
    assert "archive+sources" in result
    assert "2026-08-07T12:00:00Z" in result
    assert str(index_dir) in result
    assert "synapt recall build --no-embeddings" in result


def test_missing_chunk_and_cluster_context_label_the_same_stale_root(monkeypatch, tmp_path):
    index_dir = _wire_empty_index(monkeypatch, tmp_path)
    monkeypatch.setattr(
        freshness,
        "check_index_freshness",
        lambda *args, **kwargs: _verdict(stale=True, scanned="archive"),
    )

    chunk = server.recall_context(chunk_id="missing:t0")
    cluster = server.recall_context(cluster_id="missing-cluster")

    for result in (chunk, cluster):
        assert "STALE" in result
        assert "archive" in result
        assert str(index_dir) in result


def test_cheap_stale_does_not_claim_live_source_coverage(monkeypatch, tmp_path):
    _wire_empty_index(monkeypatch, tmp_path)
    calls: list[bool] = []

    def check(*args, **kwargs):
        calls.append(kwargs.get("deep", False))
        return _verdict(stale=True, scanned="archive")

    monkeypatch.setattr(freshness, "check_index_freshness", check)

    result = server.recall_search("absent synthetic fact")

    assert calls == [False]
    assert "checked: archive" in result
    assert "archive+sources" not in result


def test_fresh_empty_runs_deep_before_claiming_an_honest_empty(monkeypatch, tmp_path):
    _wire_empty_index(monkeypatch, tmp_path)
    calls: list[bool] = []

    def check(*args, **kwargs):
        deep = kwargs.get("deep", False)
        calls.append(deep)
        return _verdict(stale=False, scanned="archive+sources" if deep else "archive")

    monkeypatch.setattr(freshness, "check_index_freshness", check)

    result = server.recall_search("absent synthetic fact")

    assert calls == [False, True]
    assert "No results found" in result
    assert "CURRENT" in result
    assert "archive+sources" in result
    assert "synapt recall build --no-embeddings" not in result
