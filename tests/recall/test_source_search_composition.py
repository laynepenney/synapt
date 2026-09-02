from __future__ import annotations

from types import SimpleNamespace

import pytest

from synapt.recall import server
from synapt.recall.source_index import (
    SourceSearchRequest,
    SourceSearchResult,
    _clear_source_search_providers,
    register_source_search_provider,
    search_registered_sources,
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
