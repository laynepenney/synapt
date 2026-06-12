"""Producer-side coverage for recall#837.

test_quick_informative_misses.py pins the consumer (recall_quick) against a
mocked _get_index. These tests close the producer/consumer gap: the REAL
index.lookup() must populate SearchDiagnostics with coverage so recall_quick
renders informative misses for real searches, not only under the mock.

The producer keeps emitting its legacy reasons (empty_index / no_matches) so
recall_search's format_message path is unchanged; recall_quick maps those onto
the informative-miss vocabulary (empty_corpus / below_threshold).
"""

from __future__ import annotations

from unittest.mock import patch

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.storage import RecallDB


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


def test_real_lookup_no_match_populates_below_threshold_coverage(tmp_path) -> None:
    index = _index_with_one_chunk(tmp_path)
    # Tokens that cannot match the single kubernetes chunk.
    result = index.lookup("xylophone marsupial zzzznonexistent", threshold_ratio=0.2)
    assert result == ""
    diag = index._last_diagnostics
    assert diag is not None
    assert diag.reason == "no_matches"  # recall_quick maps this to below_threshold
    assert diag.chunks_scanned == 1
    assert diag.best_score == 0.0
    assert diag.threshold == 0.2
    assert diag.oldest_session_started_at == "2026-05-01"


def test_real_lookup_drives_recall_quick_verified_absence(tmp_path) -> None:
    index = _index_with_one_chunk(tmp_path)
    with patch("synapt.recall.server._get_index", return_value=index):
        from synapt.recall.server import recall_quick

        out = recall_quick("xylophone marsupial zzzznonexistent")
    assert "No prior discussion found for 'xylophone marsupial zzzznonexistent'" in out
    assert "1 chunks scanned" in out
    assert "best score 0.00 below threshold 0.20" in out
    assert "back to 2026-05-01" in out
    assert "Proceeding fresh is safe." in out
    assert "No results found." not in out


def test_real_empty_index_drives_empty_corpus(tmp_path) -> None:
    db = RecallDB(tmp_path / "recall.db")
    index = TranscriptIndex([], use_embeddings=False, cache_dir=tmp_path, db=db)
    with patch("synapt.recall.server._get_index", return_value=index):
        from synapt.recall.server import recall_quick

        out = recall_quick("anything at all")
    assert "No indexed recall corpus available for 'anything at all'" in out
    assert "Verified absence unavailable" in out
    assert "Proceeding fresh is safe." not in out


def test_empty_index_reason_unchanged_for_format_message(tmp_path) -> None:
    # recall_search still relies on SearchDiagnostics.format_message with the
    # legacy reasons; the producer must not have changed that contract.
    db = RecallDB(tmp_path / "recall.db")
    index = TranscriptIndex([], use_embeddings=False, cache_dir=tmp_path, db=db)
    index.lookup("anything", threshold_ratio=0.2)
    diag = index._last_diagnostics
    assert diag is not None
    assert diag.reason == "empty_index"
    assert "index is empty" in diag.format_message().lower()
