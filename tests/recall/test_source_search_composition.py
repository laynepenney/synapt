from __future__ import annotations

import os
import sqlite3
from types import SimpleNamespace

import pytest

from synapt.recall import server
from synapt.recall.source_index import (
    SOURCE_INDEX_SUPPORTED,
    DescriptorSourceAdapter,
    IndexedSourceSearchProvider,
    SourceAdmission,
    SourceSearchRequest,
    SourceSearchResult,
    _clear_source_search_providers,
    register_source_search_provider,
    search_registered_sources,
    sync_source,
)


@pytest.fixture(autouse=True)
def _isolated_source_provider_registry():
    _clear_source_search_providers()
    yield
    _clear_source_search_providers()


def _result() -> SourceSearchResult:
    return SourceSearchResult(
        content="# Decision\n\nScope refraction keeps projection boundaries visible.",
        structural_address="Decision [1]",
        lifecycle="current",
        revision_token="opaque-revision",
        observed_at="2026-09-01T12:00:00+00:00",
    )


def _index(rendered: str):
    return SimpleNamespace(
        sessions={},
        lookup=lambda *args, **kwargs: rendered,
        _last_diagnostics=None,
        _last_conflicts=[],
        _embedding_status="available",
        _embedding_reason="",
    )


def _silence_live_and_freshness(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(
        "synapt.recall.live.search_live_transcript", lambda *args, **kwargs: ""
    )
    monkeypatch.setattr(server, "project_index_dir", lambda: tmp_path / "index")
    monkeypatch.setattr(
        server, "_query_freshness_line", lambda _path: "Freshness: TEST"
    )


def test_registered_source_result_flows_through_ordinary_recall_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    requests: list[SourceSearchRequest] = []

    class Provider:
        def search(self, request: SourceSearchRequest):
            requests.append(request)
            return [_result()]

    register_source_search_provider("authorized-source", Provider())
    monkeypatch.setattr(server, "_get_index", lambda: _index("Past session context"))
    _silence_live_and_freshness(monkeypatch, tmp_path)

    rendered = server.recall_search(
        "projection",
        max_chunks=2,
        max_tokens=600,
        after="2026-08-01T00:00:00+00:00",
        before="2026-10-01T00:00:00+00:00",
    )

    assert requests == [
        SourceSearchRequest(
            query="projection",
            limit=2,
            after="2026-08-01T00:00:00+00:00",
            before="2026-10-01T00:00:00+00:00",
        )
    ]
    assert rendered.index("Past session context") < rendered.index(
        "[source:memory_file"
    )
    assert "Scope refraction" in rendered
    assert "opaque-revision" in rendered


@pytest.mark.skipif(
    not SOURCE_INDEX_SUPPORTED,
    reason="descriptor adapter is POSIX-only: os.O_DIRECTORY unavailable",
)
def test_indexed_markdown_flows_through_ordinary_recall_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    (root / "decision.md").write_text(
        "# Decision\n\nThe amber lattice preserves the deployment boundary.\n",
        encoding="utf-8",
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    admission = SourceAdmission(
        source_id="source-opaque-e2e",
        scope_capability=b"representative-capability",
        root_handle=root_fd,
        root_handle_id="root-handle-e2e",
        admission_epoch=1,
        policy_epoch=1,
        path_disclosure="relative",
    )
    db_path = tmp_path / "source.db"

    def open_store() -> sqlite3.Connection:
        return sqlite3.connect(db_path)

    def authorize(candidate: SourceAdmission) -> bool:
        return candidate.scope_capability == b"representative-capability"

    try:
        receipt = sync_source(
            admission,
            DescriptorSourceAdapter(),
            open_store,
            authorize,
        )
        register_source_search_provider(
            "indexed-repository",
            IndexedSourceSearchProvider(admission, open_store, authorize),
        )
        monkeypatch.setattr(
            server, "_get_index", lambda: _index("Past session context")
        )
        _silence_live_and_freshness(monkeypatch, tmp_path)

        rendered = server.recall_search(
            "amber lattice",
            max_chunks=5,
            max_tokens=600,
        )
        monkeypatch.setattr(server, "_get_index", lambda: _index(""))
        absent_rendered = server.recall_search(
            "violet turbine",
            max_chunks=5,
            max_tokens=600,
        )
    finally:
        os.close(root_fd)

    assert receipt.state == "complete"
    assert receipt.documents_seen == 1
    assert receipt.units_published == 1
    assert rendered.index("Past session context") < rendered.index(
        "[source:memory_file"
    )
    assert "decision.md · Decision [1]" in rendered
    assert "amber lattice preserves the deployment boundary" in rendered.lower()
    assert "No results found." in absent_rendered
    assert "[source:" not in absent_rendered
    assert "amber lattice" not in absent_rendered.lower()


def test_registered_source_can_answer_when_transcript_index_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class Provider:
        def search(self, request: SourceSearchRequest):
            return [_result()]

    register_source_search_provider("authorized-source", Provider())
    monkeypatch.setattr(server, "_get_index", lambda: None)
    _silence_live_and_freshness(monkeypatch, tmp_path)

    rendered = server.recall_search("projection")

    assert "Scope refraction" in rendered
    assert "Run `synapt recall setup`" not in rendered


def test_source_provider_failure_does_not_discard_existing_recall_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class FailingProvider:
        def search(self, request: SourceSearchRequest):
            raise RuntimeError("private provider detail must not escape")

    register_source_search_provider("failing-source", FailingProvider())
    monkeypatch.setattr(server, "_get_index", lambda: _index("Past session context"))
    _silence_live_and_freshness(monkeypatch, tmp_path)

    rendered = server.recall_search("projection")

    assert "Past session context" in rendered
    assert "private provider detail" not in rendered


def test_compositor_enforces_lifecycle_and_date_window_when_provider_does_not() -> None:
    current = _result()
    stale = SourceSearchResult(
        content="stale source content",
        structural_address="Decision [2]",
        lifecycle="stale",
        revision_token="stale-revision",
        observed_at="2026-09-01T12:00:00+00:00",
    )
    too_old = SourceSearchResult(
        content="out of window content",
        structural_address="Decision [3]",
        lifecycle="current",
        revision_token="old-revision",
        observed_at="2026-07-01T12:00:00+00:00",
    )

    class Provider:
        def search(self, request: SourceSearchRequest):
            return [stale, too_old, current]

    register_source_search_provider("unfiltered-source", Provider())

    results = search_registered_sources(
        SourceSearchRequest(
            query="projection",
            limit=5,
            after="2026-08-01T00:00:00+00:00",
            before="2026-10-01T00:00:00+00:00",
        )
    )

    assert results == [current]


def test_explicit_historical_mode_can_return_stale_source_result() -> None:
    stale = SourceSearchResult(
        content="stale source content",
        structural_address="Decision [2]",
        lifecycle="stale",
        revision_token="stale-revision",
        observed_at="2026-09-01T12:00:00+00:00",
    )

    class Provider:
        def search(self, request: SourceSearchRequest):
            return [stale]

    register_source_search_provider("historical-source", Provider())

    results = search_registered_sources(
        SourceSearchRequest(
            query="projection",
            limit=5,
            include_historical=True,
        )
    )

    assert results == [stale]


def test_zero_token_search_never_invokes_registered_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calls: list[SourceSearchRequest] = []

    class Provider:
        def search(self, request: SourceSearchRequest):
            calls.append(request)
            return [_result()]

    register_source_search_provider("must-not-run", Provider())
    monkeypatch.setattr(server, "_get_index", lambda: _index(""))
    _silence_live_and_freshness(monkeypatch, tmp_path)

    server.recall_search("projection", max_tokens=0)

    assert calls == []
