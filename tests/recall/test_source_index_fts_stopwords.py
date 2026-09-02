from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from synapt.recall.source_index import (
    SOURCE_INDEX_SUPPORTED,
    DescriptorSourceAdapter,
    SourceAdmission,
    _fts_query,
    search_source,
    sync_source,
)

pytestmark = pytest.mark.skipif(
    not SOURCE_INDEX_SUPPORTED,
    reason="descriptor adapter is POSIX-only: os.O_DIRECTORY unavailable",
)

CAP = b"fts-stopword-capability"


def _admission(source_id: str, root_fd: int) -> SourceAdmission:
    return SourceAdmission(
        source_id=source_id,
        scope_capability=CAP,
        root_handle=root_fd,
        root_handle_id=f"root-{source_id}",
        admission_epoch=1,
        policy_epoch=1,
        path_disclosure="relative",
    )


def _opener(path: Path):
    def open_store() -> sqlite3.Connection:
        return sqlite3.connect(path)

    return open_store


def _authorize_only(capability: bytes):
    return lambda candidate: candidate.scope_capability == capability


def test_fts_query_drops_common_english_filler_words() -> None:
    expr = _fts_query("why does the function catch exceptions but the other one does not")
    for stopword in ('"why"', '"does"', '"the"', '"but"', '"not"'):
        assert stopword not in expr, f"{stopword} should have been filtered"
    assert '"function"' in expr
    assert '"exceptions"' in expr


def test_fts_query_falls_back_to_unfiltered_when_every_token_is_a_stopword() -> None:
    # "of" and "the" alone are both stopwords -- filtering must not produce
    # an empty (unconstrained-match-nothing) expression.
    expr = _fts_query("of the")
    assert expr == '"of" "the"'


def test_fts_query_preserves_and_semantics_not_or() -> None:
    """Fewer, better tokens -- never a loosened OR. Space-separated quoted
    terms is still an implicit FTS5 AND; this just changes which tokens are
    required, not the boolean operator between them."""
    expr = _fts_query("widget factory")
    assert expr == '"widget" "factory"'
    assert " OR " not in expr


def test_verbose_query_with_fillers_now_finds_real_content(tmp_path: Path) -> None:
    """The actual defect (tracked privately, measured on a real corpus before
    this fix): a verbose natural-language question fails purely because filler
    words don't co-occur with a document's real content words, independent
    of any semantic/synonym gap. Distinctive content words verified present
    verbatim in the fixture; only fillers wrap them."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "doc.md").write_text(
        "# Working first\n\n"
        "Build the thinnest end-to-end implementation before any hardening work.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "store.db"
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = _admission("src-a", fd)
        receipt = sync_source(admission, DescriptorSourceAdapter(), _opener(db_path), _authorize_only(CAP))
        assert receipt.state == "complete"

        query = "should we build the thinnest end-to-end implementation before any hardening"
        results = search_source(admission, _opener(db_path), _authorize_only(CAP), query)
        assert len(results) == 1
        assert results[0].relative_path == "doc.md"
    finally:
        os.close(fd)


def test_negative_control_absent_content_words_still_return_nothing(tmp_path: Path) -> None:
    """Stopword filtering narrows which tokens must co-occur; it never loosens
    co-occurrence into OR. A query whose real (non-filler) content is
    genuinely absent from the corpus must still return a clear empty result,
    matching #918's own negative-control acceptance criterion."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "doc.md").write_text(
        "# Working first\n\n"
        "Build the thinnest end-to-end implementation before any hardening.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "store.db"
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = _admission("src-a", fd)
        sync_source(admission, DescriptorSourceAdapter(), _opener(db_path), _authorize_only(CAP))

        query = "should we discuss sourdough bread starter feeding schedules today"
        results = search_source(admission, _opener(db_path), _authorize_only(CAP), query)
        assert results == []
    finally:
        os.close(fd)


def test_short_exact_query_unaffected(tmp_path: Path) -> None:
    """Regression check: a short, already-working query (no filler-word
    problem to begin with) behaves identically to before this fix."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "doc.md").write_text("# Bubble up pattern\n\nActively notify your coordinator.\n", encoding="utf-8")
    db_path = tmp_path / "store.db"
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = _admission("src-a", fd)
        sync_source(admission, DescriptorSourceAdapter(), _opener(db_path), _authorize_only(CAP))

        results = search_source(admission, _opener(db_path), _authorize_only(CAP), "bubble up pattern")
        assert len(results) == 1
        assert results[0].relative_path == "doc.md"
    finally:
        os.close(fd)
