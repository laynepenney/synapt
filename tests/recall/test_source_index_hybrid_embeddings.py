from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from synapt.recall.source_index import (
    SOURCE_INDEX_SUPPORTED,
    DescriptorSourceAdapter,
    SourceAdmission,
    SourceLimits,
    search_source,
    sync_source,
)

pytestmark = pytest.mark.skipif(
    not SOURCE_INDEX_SUPPORTED,
    reason="descriptor adapter is POSIX-only: os.O_DIRECTORY unavailable",
)

CAP = b"hybrid-spike-capability"

TARGET_CONTENT = (
    "Widget assembly requires careful alignment of the primary gear housing "
    "components.\n"
)
DECOY_CONTENT = (
    "Quarterly financial projections indicate moderate revenue growth across "
    "all regions.\n"
)
PARAPHRASE_QUERY = "how do mechanical parts get properly lined up during construction"


class _FakeProvider:
    """Deterministic, dependency-free provider: exact-text-keyed vectors with
    a distinguishable default, so cosine relationships are hand-computable
    and tests never need the real (heavy) local model."""

    dim = 2

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors.get(t, [0.0, 0.0]) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        return self._vectors.get(text, [0.0, 0.0])


class _CountingProvider:
    """Wraps a real mapping but records every embed() call's text batch, so
    incrementality (only staged/changed units embedded) can be asserted."""

    dim = 2

    def __init__(self):
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[0.5, 0.5] for _ in texts]

    def embed_single(self, text: str) -> list[float]:
        return [0.5, 0.5]


def _admission(source_id: str, root_fd: int, path_disclosure: str = "relative") -> SourceAdmission:
    return SourceAdmission(
        source_id=source_id,
        scope_capability=CAP,
        root_handle=root_fd,
        root_handle_id=f"root-{source_id}",
        admission_epoch=1,
        policy_epoch=1,
        path_disclosure=path_disclosure,
    )


def _opener(path: Path):
    def open_store() -> sqlite3.Connection:
        return sqlite3.connect(path)

    return open_store


def _authorize_only(capability: bytes):
    return lambda candidate: candidate.scope_capability == capability


def _two_doc_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    (root / "target.md").write_text(TARGET_CONTENT, encoding="utf-8")
    (root / "decoy.md").write_text(DECOY_CONTENT, encoding="utf-8")
    return root


# query=[0.8, 0.6] (unit norm). target=[1, 0] -> cos 0.8. decoy=[0, 1] -> cos 0.6.
_QUERY_VECTOR = [0.8, 0.6]
_PARAPHRASE_VECTORS = {
    PARAPHRASE_QUERY: _QUERY_VECTOR,
    TARGET_CONTENT: [1.0, 0.0],
    DECOY_CONTENT: [0.0, 1.0],
}


def test_flag_off_sync_and_search_unchanged(tmp_path: Path) -> None:
    """Omitting embed_provider (every pre-existing caller) touches the new
    table not at all and returns identical results to before this feature."""
    root = _two_doc_root(tmp_path)
    db_path = tmp_path / "store.db"
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = _admission("src-a", fd)
        receipt = sync_source(admission, DescriptorSourceAdapter(), _opener(db_path), _authorize_only(CAP))
        assert receipt.state == "complete"
        assert receipt.units_published == 2

        conn = sqlite3.connect(db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM source_units_embeddings").fetchone()[0]
        finally:
            conn.close()
        assert count == 0

        # A paraphrase query with zero lexical overlap must miss, exactly as
        # it did before this feature existed.
        results = search_source(admission, _opener(db_path), _authorize_only(CAP), PARAPHRASE_QUERY)
        assert results == []
    finally:
        os.close(fd)


def test_sync_embeds_only_staged_units_incrementally(tmp_path: Path) -> None:
    root = _two_doc_root(tmp_path)
    db_path = tmp_path / "store.db"
    provider = _CountingProvider()
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = _admission("src-a", fd)
        receipt = sync_source(
            admission, DescriptorSourceAdapter(), _opener(db_path), _authorize_only(CAP),
            embed_provider=provider,
        )
        assert receipt.state == "complete"
        assert len(provider.embed_calls) == 1
        assert len(provider.embed_calls[0]) == 2  # both docs, first sync

        conn = sqlite3.connect(db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM source_units_embeddings").fetchone()[0]
        finally:
            conn.close()
        assert count == 2

        # Touch only target.md. A second sync must embed only its unit(s),
        # not the whole corpus again -- the incrementality free-ride #918
        # already provides via content-hash skip.
        (root / "target.md").write_text(TARGET_CONTENT + "\nAn added line.\n", encoding="utf-8")
        receipt2 = sync_source(
            admission, DescriptorSourceAdapter(), _opener(db_path), _authorize_only(CAP),
            embed_provider=provider,
        )
        assert receipt2.state == "complete"
        assert receipt2.documents_reused == 1  # decoy.md untouched, reused
        assert len(provider.embed_calls) == 2
        assert len(provider.embed_calls[1]) == 1  # only target.md's unit(s)
    finally:
        os.close(fd)


def test_search_hybrid_finds_paraphrase_bm25_misses(tmp_path: Path) -> None:
    root = _two_doc_root(tmp_path)
    db_path = tmp_path / "store.db"
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = _admission("src-a", fd)
        sync_source(
            admission, DescriptorSourceAdapter(), _opener(db_path), _authorize_only(CAP),
            embed_provider=_FakeProvider(_PARAPHRASE_VECTORS),
        )

        # Confirm the premise: BM25-only genuinely misses (zero lexical overlap).
        bm25_only = search_source(admission, _opener(db_path), _authorize_only(CAP), PARAPHRASE_QUERY)
        assert bm25_only == []

        hybrid = search_source(
            admission, _opener(db_path), _authorize_only(CAP), PARAPHRASE_QUERY,
            embed_provider=_FakeProvider(_PARAPHRASE_VECTORS), similarity_floor=0.7,
        )
        assert len(hybrid) == 1
        assert hybrid[0].relative_path == "target.md"
        assert hybrid[0].similarity == pytest.approx(0.8)
    finally:
        os.close(fd)


def test_similarity_floor_excludes_embedding_only_below_floor(tmp_path: Path) -> None:
    """decoy.md's cosine (0.6) is real but below a 0.7 floor and has no BM25
    overlap either -- it must not appear in results at all."""
    root = _two_doc_root(tmp_path)
    db_path = tmp_path / "store.db"
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = _admission("src-a", fd)
        sync_source(
            admission, DescriptorSourceAdapter(), _opener(db_path), _authorize_only(CAP),
            embed_provider=_FakeProvider(_PARAPHRASE_VECTORS),
        )
        results = search_source(
            admission, _opener(db_path), _authorize_only(CAP), PARAPHRASE_QUERY,
            embed_provider=_FakeProvider(_PARAPHRASE_VECTORS), similarity_floor=0.7,
        )
        relative_paths = {r.relative_path for r in results}
        assert "decoy.md" not in relative_paths
    finally:
        os.close(fd)


def test_negative_control_absent_query_returns_empty_above_floor(tmp_path: Path) -> None:
    """#918's own acceptance criterion: a query for something genuinely
    absent returns a clear empty result under the DEFAULT (measured) floor,
    not a plausible-looking near-match."""
    root = _two_doc_root(tmp_path)
    db_path = tmp_path / "store.db"
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = _admission("src-a", fd)
        # Both docs deliberately far from the absent query in this fake
        # embedding space (0.6 cosine, well under the measured 0.4176-margin
        # floor is NOT the point here -- the point is a floor exists and a
        # near-miss below it is excluded, mirrored with the real numbers in
        # the design note's section 3).
        vectors = {
            "content nobody asked about": [0.0, 0.0],
            TARGET_CONTENT: [1.0, 0.0],
            DECOY_CONTENT: [0.0, 1.0],
        }
        sync_source(
            admission, DescriptorSourceAdapter(), _opener(db_path), _authorize_only(CAP),
            embed_provider=_FakeProvider(vectors),
        )
        results = search_source(
            admission, _opener(db_path), _authorize_only(CAP),
            "content nobody asked about",
            embed_provider=_FakeProvider(vectors),
        )
        assert results == []
    finally:
        os.close(fd)


def test_wrong_capability_refused_with_hybrid_search_on(tmp_path: Path) -> None:
    root = _two_doc_root(tmp_path)
    db_path = tmp_path / "store.db"
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = _admission("src-a", fd)
        sync_source(
            admission, DescriptorSourceAdapter(), _opener(db_path), _authorize_only(CAP),
            embed_provider=_FakeProvider(_PARAPHRASE_VECTORS),
        )
        wrong_cap_results = search_source(
            admission, _opener(db_path), _authorize_only(b"a-different-capability"),
            PARAPHRASE_QUERY, embed_provider=_FakeProvider(_PARAPHRASE_VECTORS),
            similarity_floor=0.7,
        )
        assert wrong_cap_results == []
    finally:
        os.close(fd)


def test_cross_org_isolation_holds_with_hybrid_search_on(tmp_path: Path) -> None:
    """Extends the merged R3 negative control (test_source_index_cross_org_
    isolation.py) with embed_provider supplied on both sync and search --
    the new embeddings table joins through source_units/source_documents for
    every query, same as BM25, so it must never leak across a shared store."""
    synapt_root = tmp_path / "synapt-org-root"
    conversa_root = tmp_path / "conversa-org-root"
    synapt_root.mkdir()
    conversa_root.mkdir()
    (synapt_root / "decision.md").write_text(
        "The host-org renewal terms stay unlisted.\n", encoding="utf-8"
    )
    (conversa_root / "decision.md").write_text(
        "The conversa client roster stays unlisted.\n", encoding="utf-8"
    )
    shared_db = tmp_path / "shared-host-store.db"

    # Deliberately give both orgs' documents an IDENTICAL vector, so any leak
    # would show up as the wrong org's content surfacing via the embedding
    # channel even without any lexical overlap.
    vectors = {
        "The host-org renewal terms stay unlisted.\n": [1.0, 0.0],
        "The conversa client roster stays unlisted.\n": [1.0, 0.0],
        "renewal terms": [1.0, 0.0],
    }

    synapt_fd = os.open(synapt_root, os.O_RDONLY | os.O_DIRECTORY)
    conversa_fd = os.open(conversa_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        synapt_admission = SourceAdmission(
            source_id="org-synapt", scope_capability=b"synapt-capability",
            root_handle=synapt_fd, root_handle_id="root-synapt",
            admission_epoch=1, policy_epoch=1, path_disclosure="relative",
        )
        conversa_admission = SourceAdmission(
            source_id="org-conversa", scope_capability=b"conversa-capability",
            root_handle=conversa_fd, root_handle_id="root-conversa",
            admission_epoch=1, policy_epoch=1, path_disclosure="relative",
        )

        for admission, capability in (
            (synapt_admission, b"synapt-capability"),
            (conversa_admission, b"conversa-capability"),
        ):
            receipt = sync_source(
                admission, DescriptorSourceAdapter(), _opener(shared_db),
                _authorize_only(capability), embed_provider=_FakeProvider(vectors),
            )
            assert receipt.state == "complete"

        # conversa's admission querying with an embedding near BOTH orgs'
        # (identical) vectors must still only ever see its own org's row.
        conversa_results = search_source(
            conversa_admission, _opener(shared_db), _authorize_only(b"conversa-capability"),
            "renewal terms", embed_provider=_FakeProvider(vectors), similarity_floor=0.5,
        )
        synapt_results = search_source(
            synapt_admission, _opener(shared_db), _authorize_only(b"synapt-capability"),
            "renewal terms", embed_provider=_FakeProvider(vectors), similarity_floor=0.5,
        )
    finally:
        os.close(synapt_fd)
        os.close(conversa_fd)

    assert len(conversa_results) == 1
    assert "roster" in conversa_results[0].content.lower()
    assert len(synapt_results) == 1
    assert "renewal" in synapt_results[0].content.lower()
