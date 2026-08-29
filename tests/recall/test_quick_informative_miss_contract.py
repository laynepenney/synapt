"""TDD contract for recall#837 informative ``recall_quick`` misses."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.server import recall_quick
from synapt.recall.storage import RecallDB


@dataclass
class _QuickMissDiagnostics:
    total_sessions: int
    total_chunks: int
    oldest_indexed_at: str | None
    embeddings_available: bool = False
    reason: str = "no_matches"

    def format_message(self) -> str:
        return "legacy diagnostic miss"


class _MissIndex:
    _embedding_status = "disabled"
    _embedding_reason = ""

    def __init__(self, diagnostics: _QuickMissDiagnostics) -> None:
        self._last_diagnostics = diagnostics
        self.lookup_kwargs: dict[str, object] | None = None

    def lookup(self, query: str, **kwargs: object) -> str:
        self.lookup_kwargs = kwargs
        return ""


def _index_with_one_chunk(tmp_path) -> TranscriptIndex:
    db = RecallDB(tmp_path / "recall.db")
    chunk = TranscriptChunk(
        id="sess1:t0",
        session_id="sess1",
        timestamp="2026-05-01T09:00:00+00:00",
        turn_index=0,
        user_text="we discussed kubernetes deployment manifests at length",
        assistant_text="here is the helm chart for the kubernetes cluster",
    )
    return TranscriptIndex([chunk], use_embeddings=False, cache_dir=tmp_path, db=db)


def test_quick_miss_returns_scoped_absence_with_corpus_coverage() -> None:
    index = _MissIndex(
        _QuickMissDiagnostics(
            total_sessions=12,
            total_chunks=384,
            oldest_indexed_at="2026-05-01",
        )
    )

    with patch("synapt.recall.server._get_index", return_value=index):
        result = recall_quick("licenses proceeding")

    assert "No prior keyword match found for 'licenses proceeding'" in result
    assert "searched 12 sessions across 384 indexed chunks" in result
    assert "indexed back to 2026-05-01" in result
    assert "semantic search was not used" in result
    assert "Proceeding fresh is reasonable after this keyword check." in result
    assert "legacy diagnostic miss" not in result


def test_empty_corpus_does_not_claim_a_verified_absence() -> None:
    index = _MissIndex(
        _QuickMissDiagnostics(
            total_sessions=0,
            total_chunks=0,
            oldest_indexed_at=None,
            reason="empty_index",
        )
    )

    with patch("synapt.recall.server._get_index", return_value=index):
        result = recall_quick("anything")

    assert "No indexed recall corpus available for 'anything'" in result
    assert "0 sessions" in result
    assert "0 indexed chunks" in result
    assert "Verified absence unavailable" in result
    assert "Proceeding fresh is reasonable" not in result


def test_quick_miss_uses_the_threshold_passed_to_lookup() -> None:
    index = _MissIndex(
        _QuickMissDiagnostics(
            total_sessions=1,
            total_chunks=5,
            oldest_indexed_at="2026-06-12",
        )
    )

    with patch("synapt.recall.server._get_index", return_value=index):
        recall_quick("needle")

    assert index.lookup_kwargs is not None
    assert index.lookup_kwargs["threshold_ratio"] == 0.2


def test_real_lookup_no_match_records_coverage_diagnostics(tmp_path) -> None:
    index = _index_with_one_chunk(tmp_path)

    result = index.lookup(
        "xylophone marsupial zzzznonexistent",
        depth="summary",
        threshold_ratio=0.2,
    )

    assert result == ""
    diagnostics = index._last_diagnostics
    assert diagnostics is not None
    assert diagnostics.reason == "no_matches"
    assert diagnostics.total_sessions == 1
    assert diagnostics.total_chunks == 1
    assert diagnostics.oldest_indexed_at == "2026-05-01"


def test_real_lookup_drives_quick_verified_absence(tmp_path) -> None:
    index = _index_with_one_chunk(tmp_path)

    with patch("synapt.recall.server._get_index", return_value=index):
        result = recall_quick("xylophone marsupial zzzznonexistent")

    assert "No prior keyword match found for 'xylophone marsupial zzzznonexistent'" in result
    assert "searched 1 session across 1 indexed chunk" in result
    assert "indexed back to 2026-05-01" in result
    assert "semantic search was not used" in result
    assert "Proceeding fresh is reasonable after this keyword check." in result
    assert "No results found." not in result


def test_real_empty_index_drives_unverified_empty_corpus_message(tmp_path) -> None:
    db = RecallDB(tmp_path / "recall.db")
    index = TranscriptIndex([], use_embeddings=False, cache_dir=tmp_path, db=db)

    with patch("synapt.recall.server._get_index", return_value=index):
        result = recall_quick("anything at all")

    assert "No indexed recall corpus available for 'anything at all'" in result
    assert "Verified absence unavailable" in result
    assert "Proceeding fresh is reasonable" not in result


def test_recall_search_diagnostic_format_remains_legacy_compatible(tmp_path) -> None:
    db = RecallDB(tmp_path / "recall.db")
    index = TranscriptIndex([], use_embeddings=False, cache_dir=tmp_path, db=db)

    index.lookup("anything", threshold_ratio=0.2)

    diagnostics = index._last_diagnostics
    assert diagnostics is not None
    assert diagnostics.reason == "empty_index"
    assert "index is empty" in diagnostics.format_message().lower()
