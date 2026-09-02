from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from synapt.recall import source_index
from synapt.recall.source_index import (
    SOURCE_INDEX_SUPPORTED,
    DescriptorSourceAdapter,
    SourceAdmission,
    SourceLimits,
    compose_source_results,
    parse_markdown,
    render_source_results,
    search_source,
    sync_source,
)


pytestmark = pytest.mark.skipif(
    not SOURCE_INDEX_SUPPORTED,
    reason="descriptor adapter is POSIX-only: os.O_DIRECTORY unavailable",
)


def _admission(root_fd: int, *, disclosure: str = "hidden") -> SourceAdmission:
    return SourceAdmission(
        source_id="source-opaque-1",
        scope_capability=b"capability",
        root_handle=root_fd,
        root_handle_id="root-handle-opaque-1",
        admission_epoch=1,
        policy_epoch=1,
        path_disclosure=disclosure,
    )


def _opener(path: Path, calls: list[str] | None = None):
    def open_store() -> sqlite3.Connection:
        if calls is not None:
            calls.append("open")
        return sqlite3.connect(path)

    return open_store


def test_memory_file_flows_through_distinct_source_index_and_render(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    (root / "index.md").write_text(
        "# Decisions\n\nScope refraction keeps projection boundaries visible.\n",
        encoding="utf-8",
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = _admission(root_fd)
        db_path = tmp_path / "private-source.db"
        receipt = sync_source(
            admission,
            DescriptorSourceAdapter(),
            _opener(db_path),
            lambda candidate: candidate.scope_capability == b"capability",
        )
        assert receipt.state == "complete"
        assert receipt.documents_seen == 1
        assert receipt.units_published == 1

        results = search_source(
            admission,
            _opener(db_path),
            lambda candidate: candidate.scope_capability == b"capability",
            "scope refraction",
        )
    finally:
        os.close(root_fd)

    assert len(results) == 1
    result = results[0]
    assert result.source_kind == "memory_file"
    assert result.structural_address == "Decisions [1]"
    assert result.lifecycle == "current"
    assert result.relative_path is None
    assert "scope refraction" in result.content.lower()
    rendered = render_source_results(results)
    assert "[source:memory_file · current · Decisions [1] · revision " in rendered
    assert str(root) not in rendered
    assert "document_sha256" not in rendered
    assert compose_source_results("conversation result", results).startswith(
        "conversation result\n\n[source:"
    )

    connection = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        connection.close()
    assert "source_units" in tables
    assert not ({"chunks", "knowledge", "query_tail_chunks"} & tables)


def test_unchanged_second_scan_reuses_document_without_parsing(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    (root / "rule.md").write_text(
        "# Rule\n\nPrefer the narrow seam.\n", encoding="utf-8"
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    parser_calls: list[bytes] = []

    def counting_parser(content: bytes):
        parser_calls.append(content)
        return parse_markdown(content)

    try:
        admission = _admission(root_fd)
        opener = _opener(tmp_path / "private-source.db")
        first = sync_source(
            admission,
            DescriptorSourceAdapter(),
            opener,
            lambda _candidate: True,
            parser=counting_parser,
        )
        second = sync_source(
            admission,
            DescriptorSourceAdapter(),
            opener,
            lambda _candidate: True,
            parser=counting_parser,
        )
    finally:
        os.close(root_fd)

    assert first.state == second.state == "complete"
    assert len(parser_calls) == 1
    assert second.documents_reused == 1
    assert second.units_published == 0


def test_unauthorized_calls_open_nothing_and_return_no_corpus_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    store_calls: list[str] = []
    try:
        admission = _admission(root_fd)
        opener = _opener(tmp_path / "must-not-exist.db", store_calls)
        receipt = sync_source(
            admission,
            DescriptorSourceAdapter(),
            opener,
            lambda _candidate: False,
        )
        results = search_source(
            admission,
            opener,
            lambda _candidate: False,
            "anything",
        )
    finally:
        os.close(root_fd)

    assert receipt.state == "unauthorized"
    assert receipt.generation is None
    assert receipt.documents_seen is None
    assert results == []
    assert store_calls == []
    assert not (tmp_path / "must-not-exist.db").exists()


def test_enumerate_refuses_cleanly_when_platform_unsupported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SOURCE_INDEX_SUPPORTED False must raise a clear _ScanFailure("unsupported")
    before any os.O_DIRECTORY use -- not let its absence surface as a raw
    AttributeError deep in the walk, which is the actual failure this
    platform ever produces on Windows.

    Setup happens with the real O_DIRECTORY still present (this test only
    runs where it exists, per the module skip); the attribute is deleted
    only after the admission fd is built, so removing the guard below (as a
    manual mutation check) makes enumerate() reach the deleted attribute and
    raise a genuine AttributeError -- proving this test is bound to the
    guard, not merely to the derived flag."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "child").mkdir()
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(source_index, "SOURCE_INDEX_SUPPORTED", False)
        monkeypatch.delattr(os, "O_DIRECTORY", raising=False)
        admission = _admission(root_fd)
        with pytest.raises(source_index._ScanFailure) as exc_info:
            list(DescriptorSourceAdapter().enumerate(admission, SourceLimits()))
        assert exc_info.value.state == "unsupported"
    finally:
        os.close(root_fd)
