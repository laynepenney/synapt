from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from synapt.recall.source_index import (
    SOURCE_INDEX_SUPPORTED,
    DescriptorSourceAdapter,
    SourceAdmission,
    search_source,
    sync_source,
)


pytestmark = pytest.mark.skipif(
    not SOURCE_INDEX_SUPPORTED,
    reason="descriptor adapter is POSIX-only: os.O_DIRECTORY unavailable",
)


def _admission(source_id: str, capability: bytes, root_fd: int) -> SourceAdmission:
    return SourceAdmission(
        source_id=source_id,
        scope_capability=capability,
        root_handle=root_fd,
        root_handle_id=f"root-{source_id}",
        admission_epoch=1,
        policy_epoch=1,
    )


def _opener(path: Path):
    def open_store() -> sqlite3.Connection:
        return sqlite3.connect(path)

    return open_store


def _authorize_only(capability: bytes):
    return lambda candidate: candidate.scope_capability == capability


def test_two_real_orgs_sharing_one_store_cannot_see_each_others_units(
    tmp_path: Path,
) -> None:
    """A shared physical store is the exact failure mode this codebase has hit
    before (store resolution resolving two identities into one directory).
    Both orgs use the SAME sqlite file and the SAME on-disk filename, so
    isolation depends entirely on the source_id-scoped query, not on
    filesystem separation."""

    synapt_root = tmp_path / "synapt-org-root"
    conversa_root = tmp_path / "conversa-org-root"
    synapt_root.mkdir()
    conversa_root.mkdir()
    (synapt_root / "decision.md").write_text(
        "# Decision\n\nThe host-org renewal terms stay unlisted.\n",
        encoding="utf-8",
    )
    (conversa_root / "decision.md").write_text(
        "# Decision\n\nThe conversa client roster stays unlisted.\n",
        encoding="utf-8",
    )

    shared_db = tmp_path / "shared-host-store.db"  # ONE file, both orgs

    synapt_fd = os.open(synapt_root, os.O_RDONLY | os.O_DIRECTORY)
    conversa_fd = os.open(conversa_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        synapt_admission = _admission("org-synapt", b"synapt-capability", synapt_fd)
        conversa_admission = _admission(
            "org-conversa", b"conversa-capability", conversa_fd
        )

        for admission, capability in (
            (synapt_admission, b"synapt-capability"),
            (conversa_admission, b"conversa-capability"),
        ):
            receipt = sync_source(
                admission,
                DescriptorSourceAdapter(),
                _opener(shared_db),
                _authorize_only(capability),
            )
            assert receipt.state == "complete"
            assert receipt.documents_seen == 1

        # Witness that cross-contamination is a REAL possibility being averted,
        # not a vacuous test: the shared store genuinely holds both orgs' rows
        # under the identical relative_path "decision.md".
        raw = sqlite3.connect(shared_db)
        try:
            rows = raw.execute(
                "SELECT source_id, relative_path FROM source_documents "
                "ORDER BY source_id"
            ).fetchall()
        finally:
            raw.close()
        assert rows == [
            ("org-conversa", "decision.md"),
            ("org-synapt", "decision.md"),
        ]

        synapt_results = search_source(
            synapt_admission,
            _opener(shared_db),
            _authorize_only(b"synapt-capability"),
            "roster",  # a term that exists ONLY in the conversa document
        )
        conversa_results = search_source(
            conversa_admission,
            _opener(shared_db),
            _authorize_only(b"conversa-capability"),
            "renewal",  # a term that exists ONLY in the synapt document
        )
    finally:
        os.close(synapt_fd)
        os.close(conversa_fd)

    assert synapt_results == []
    assert conversa_results == []

    synapt_own = search_source(
        synapt_admission,
        _opener(shared_db),
        _authorize_only(b"synapt-capability"),
        "renewal",
    )
    conversa_own = search_source(
        conversa_admission,
        _opener(shared_db),
        _authorize_only(b"conversa-capability"),
        "roster",
    )
    assert len(synapt_own) == 1
    assert "renewal" in synapt_own[0].content.lower()
    assert len(conversa_own) == 1
    assert "roster" in conversa_own[0].content.lower()

    # An admission carrying the WRONG capability for its own source_id is
    # refused outright -- not merely scoped away, refused before any query.
    wrong_capability_results = search_source(
        synapt_admission,
        _opener(shared_db),
        _authorize_only(b"conversa-capability"),
        "renewal",
    )
    assert wrong_capability_results == []
