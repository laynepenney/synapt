"""Contract for recall_code (recall#903-adjacent, R3.2 first cut).

recall_code composes find_symbols (code_index.py) with recall_search
(server.py) and, when one is registered, an optional per-hit annotator
discovered via the synapt.annotators entry-point group (same discovery
shape as synapt.backends in _model_router.py -- see
test_model_router.py::TestBackendRegistry::test_entry_point_discovery for
the established pattern this suite mirrors). This suite is hermetic: a
small real repo is indexed into a throwaway DB via index_repo (never this
gripspace's own code), and recall_search / entry-point discovery are
monkeypatched so no test depends on transcript history or on what happens
to be installed on whichever desk runs it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import synapt.recall.code_search as code_search
from synapt.recall.code_index import index_repo
from synapt.recall.code_search import (
    _identifier_tokens,
    _match_kind,
    recall_code,
)


REPO_NAME = "fixture-repo"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "code.db"


@pytest.fixture(autouse=True)
def _reset_annotator_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_annotator caches its result process-wide, same as
    _model_router's _backends_loaded -- reset per test so discovery tests
    and ordinary tests don't leak into each other regardless of run order."""
    monkeypatch.setattr(code_search, "_annotator_loaded", False)
    monkeypatch.setattr(code_search, "_annotator", None)


def _no_memory_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the code-search half: recall_search always reports no hit,
    so `has_memory_hit` and `memories` are deterministic regardless of
    whatever transcript state exists on the machine running the test."""
    monkeypatch.setattr(
        "synapt.recall.code_search.recall_server.recall_search",
        lambda query, max_chunks=3: "No results found.",
    )


def test_exact_name_query_returns_the_named_symbol(tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_memory_hit(monkeypatch)
    src = tmp_path / "repo"
    src.mkdir()
    (src / "widget.py").write_text(
        "def widget_factory(x):\n    return x\n"
    )
    index_repo(src, db, REPO_NAME)

    result = recall_code(
        "widget_factory", db_path=str(db), repo=REPO_NAME, repo_root=str(src)
    )

    assert result["has_code_hit"] is True
    assert result["has_memory_hit"] is False
    assert [s["name"] for s in result["symbols"]] == ["widget_factory"]
    assert result["symbols"][0]["match_kind"] == "exact"
    assert result["symbols"][0]["matched_token"] == "widget_factory"


def test_filler_words_are_dropped_before_querying(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stopwords never reach find_symbols as candidate tokens -- the crowding
    defect's root cause (measured 2026-09-02): 'catch' from a natural-
    language question substring-matched unrelated cmd_catchup-family
    symbols purely because it was fed to find_symbols at all."""
    tokens = _identifier_tokens(
        "why does the function catch exceptions but the other does not"
    )
    for stopword in ("why", "does", "the", "but", "not", "catch"):
        assert stopword not in tokens, f"{stopword!r} should have been filtered"
    assert "function" in tokens
    assert "exceptions" in tokens


def test_match_kind_classifies_exact_prefix_substring() -> None:
    assert _match_kind("widget", "widget") == "exact"
    assert _match_kind("widget_factory", "widget") == "prefix"
    assert _match_kind("legacy_widget_factory", "widget") == "substring"


def test_global_rank_sort_survives_a_token_whose_own_noise_fills_the_budget(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Witness for the crowding fix (2026-09-02, Stromus's structural-change
    ruling). Under the prior per-token-arrival order, a token processed
    FIRST whose own substring hits alone reached max_symbols crowded out a
    LATER token's exact hit, purely by append-then-truncate-at-the-end
    ordering -- stopword filtering alone cannot prevent this, since the
    crowding token here ("helper") is a real, meaningful word, not filler.

    Fixture: four functions sharing the substring "helper" (never an exact
    or prefix match for the query token "helper" itself... actually they
    ARE prefix matches, which still outranks nothing here since the real
    assertion is exact-beats-everything) plus one function named exactly
    "target_symbol". max_symbols=3 makes the old order's crowding
    deterministic: "helper" alone returns >=3 hits before "target_symbol"
    (the second query token) is ever processed."""
    _no_memory_hit(monkeypatch)
    src = tmp_path / "repo"
    src.mkdir()
    (src / "helpers.py").write_text(
        "def helper_one():\n    pass\n\n\n"
        "def helper_two():\n    pass\n\n\n"
        "def helper_three():\n    pass\n\n\n"
        "def helper_four():\n    pass\n\n\n"
        "def target_symbol():\n    pass\n"
    )
    index_repo(src, db, REPO_NAME)

    result = recall_code(
        "helper target_symbol",
        db_path=str(db),
        repo=REPO_NAME,
        repo_root=str(src),
        max_symbols=3,
    )

    names = [s["name"] for s in result["symbols"]]
    assert "target_symbol" in names, (
        f"exact hit crowded out by substring noise -- got {names}"
    )
    assert result["symbols"][0]["name"] == "target_symbol", (
        "the exact hit must sort first regardless of which token found it first"
    )
    assert result["symbols"][0]["match_kind"] == "exact"


def test_annotation_present_when_an_annotator_is_registered(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registers a fake annotator through the synapt.annotators entry-point
    seam -- the same discovery mechanism synapt.backends uses -- rather than
    depending on any specific downstream package actually being installed."""
    _no_memory_hit(monkeypatch)
    src = tmp_path / "repo"
    src.mkdir()
    (src / "widget.py").write_text("def widget_factory(x):\n    return x\n")
    index_repo(src, db, REPO_NAME)

    calls: list[tuple] = []

    def _fake_annotator(repo_root, path, line_start, line_end):
        calls.append((repo_root, path, line_start, line_end))
        return [{"excerpt": "a fake annotation"}]

    class ControlledEntryPoint:
        name = "fixture-annotator"

        @staticmethod
        def load():
            return _fake_annotator

    def controlled_entry_points(*, group):
        assert group == "synapt.annotators"
        return [ControlledEntryPoint()]

    monkeypatch.setattr(
        code_search.importlib.metadata, "entry_points", controlled_entry_points
    )

    result = recall_code(
        "widget_factory", db_path=str(db), repo=REPO_NAME, repo_root=str(src)
    )

    assert len(calls) == 1
    hit = result["symbols"][0]
    assert hit["annotation"] == [{"excerpt": "a fake annotation"}]
    assert "annotation_error" not in hit


def test_annotation_absent_without_error_when_reset_annotator_cache_registered(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No entry point registered under synapt.annotators -- the default
    (unmodified) OSS state, since entry_points() is not otherwise
    monkeypatched by this test. Symbols must come back with no "annotation"
    field and no "annotation_error" -- silence, not an error."""
    _no_memory_hit(monkeypatch)
    src = tmp_path / "repo"
    src.mkdir()
    (src / "widget.py").write_text("def widget_factory(x):\n    return x\n")
    index_repo(src, db, REPO_NAME)

    result = recall_code(
        "widget_factory", db_path=str(db), repo=REPO_NAME, repo_root=str(src)
    )

    assert result["has_code_hit"] is True
    hit = result["symbols"][0]
    assert "annotation" not in hit
    assert "annotation_error" not in hit


def test_annotation_error_is_captured_not_raised(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered annotator that raises degrades to an annotation_error
    field, since annotation is enrichment and never the answer -- mirrors
    the broken-entry-point tolerance test_model_router.py has for
    synapt.backends, but for the per-call failure rather than the load
    failure."""
    _no_memory_hit(monkeypatch)
    src = tmp_path / "repo"
    src.mkdir()
    (src / "widget.py").write_text("def widget_factory(x):\n    return x\n")
    index_repo(src, db, REPO_NAME)

    def _broken_annotator(repo_root, path, line_start, line_end):
        raise RuntimeError("annotator backend unreachable")

    class ControlledEntryPoint:
        name = "fixture-broken-annotator"

        @staticmethod
        def load():
            return _broken_annotator

    monkeypatch.setattr(
        code_search.importlib.metadata,
        "entry_points",
        lambda *, group: [ControlledEntryPoint()],
    )

    result = recall_code(
        "widget_factory", db_path=str(db), repo=REPO_NAME, repo_root=str(src)
    )

    hit = result["symbols"][0]
    assert "annotation" not in hit
    assert hit["annotation_error"] == "annotator backend unreachable"


def test_broken_annotator_entry_point_does_not_break_search(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry point that fails to *load* (not merely fails when called)
    must not break recall_code -- mirrors
    test_model_router.py::TestBackendRegistry::test_broken_entry_point_does_not_break_routing
    for this seam."""
    _no_memory_hit(monkeypatch)
    src = tmp_path / "repo"
    src.mkdir()
    (src / "widget.py").write_text("def widget_factory(x):\n    return x\n")
    index_repo(src, db, REPO_NAME)

    class BrokenEntryPoint:
        name = "broken"

        @staticmethod
        def load():
            raise ModuleNotFoundError("optional annotator package is unavailable")

    monkeypatch.setattr(
        code_search.importlib.metadata,
        "entry_points",
        lambda *, group: [BrokenEntryPoint()],
    )

    result = recall_code(
        "widget_factory", db_path=str(db), repo=REPO_NAME, repo_root=str(src)
    )

    assert result["has_code_hit"] is True
    hit = result["symbols"][0]
    assert "annotation" not in hit
    assert "annotation_error" not in hit
    assert code_search._annotator_loaded is True
    assert code_search._annotator is None


def test_memory_hit_reflects_recall_search(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "synapt.recall.code_search.recall_server.recall_search",
        lambda query, max_chunks=3: "Some past session decided X.",
    )
    src = tmp_path / "repo"
    src.mkdir()
    (src / "widget.py").write_text("def widget_factory(x):\n    return x\n")
    index_repo(src, db, REPO_NAME)

    result = recall_code(
        "widget_factory", db_path=str(db), repo=REPO_NAME, repo_root=str(src)
    )

    assert result["has_memory_hit"] is True
    assert result["memories"] == "Some past session decided X."


def test_no_hit_anywhere_is_honest_not_silent(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_memory_hit(monkeypatch)
    src = tmp_path / "repo"
    src.mkdir()
    (src / "widget.py").write_text("def widget_factory(x):\n    return x\n")
    index_repo(src, db, REPO_NAME)

    result = recall_code(
        "nothing_matches_anything_here",
        db_path=str(db),
        repo=REPO_NAME,
        repo_root=str(src),
    )

    assert result["has_code_hit"] is False
    assert result["symbols"] == []
    assert result["has_memory_hit"] is False


def test_coverage_outranks_a_single_exact_hit_on_a_generic_word(tmp_path, monkeypatch, _reset_annotator_cache) -> None:
    """Dogfood 2026-09-06: "where is the build lock acquired" returned four
    fixtures named ``build`` as exact hits and never ``_acquire_build_lock``.
    A symbol containing more of the question's words ranks first, the
    duplicate-key upgrade keeps the best match kind, and a test path never
    outranks a production path at equal coverage."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "lock.py").write_text(
        "def _acquire_build_lock(root):\n    return root\n\n"
        "def build_lock_has_waiter(root):\n    return False\n"
    )
    (repo / "tests" / "test_build.py").write_text(
        "def build():\n    return 1\n\n"
        "def blocked_build():\n    return 2\n"
    )
    db = tmp_path / "code.db"
    index_repo(repo, db, repo="repo")
    monkeypatch.setattr(
        "synapt.recall.code_search.recall_server.recall_search",
        lambda q, **k: "No results found.",
    )
    result = recall_code(
        "where is the build lock acquired",
        db_path=str(db), repo="repo", repo_root=str(repo), max_symbols=4,
    )
    names = [s["name"] for s in result["symbols"]]
    assert names[0] == "_acquire_build_lock", names
    top = result["symbols"][0]
    assert top["token_coverage"] == 3, top   # build, lock, acquire(d)
    assert top["is_test"] is False
    # the generic exact hit is still returned, just not first
    assert "build" in names
    assert names.index("build_lock_has_waiter") < names.index("build")


def test_duplicate_key_keeps_the_best_match_kind(tmp_path, monkeypatch, _reset_annotator_cache) -> None:
    """"cold no-caller refresh" reaches ``cold_no_caller_refresh`` first by
    the raw token "cold" (prefix) and then by the underscore-joined form
    (exact); the kept hit must carry the exact kind, not the first arrival."""
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "r.py").write_text("def cold_no_caller_refresh(d):\n    return d\n")
    db = tmp_path / "code.db"
    index_repo(repo, db, repo="repo")
    monkeypatch.setattr(
        "synapt.recall.code_search.recall_server.recall_search",
        lambda q, **k: "No results found.",
    )
    result = recall_code(
        "cold no-caller refresh", db_path=str(db), repo="repo", repo_root=str(repo)
    )
    assert [s["name"] for s in result["symbols"]] == ["cold_no_caller_refresh"]
    assert result["symbols"][0]["match_kind"] == "exact"
    assert result["symbols"][0]["matched_token"] == "cold_no_caller_refresh"


def test_production_path_outranks_test_path_on_a_genuine_tie(tmp_path, monkeypatch, _reset_annotator_cache) -> None:
    """Sentinel's R1 mutant (2026-09-06): forcing _is_test_path to False left
    every test green, so "production before test" was a claim with no
    witness. Two symbols with the SAME name, same match kind, same coverage,
    one under tests/: the production one must come first, and flipping the
    demotion off must red this test."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_widget.py").write_text("def widget_factory():\n    return 1\n")
    (repo / "src" / "widget.py").write_text("def widget_factory():\n    return 2\n")
    db = tmp_path / "code.db"
    index_repo(repo, db, repo="repo")
    monkeypatch.setattr(
        "synapt.recall.code_search.recall_server.recall_search",
        lambda q, **k: "No results found.",
    )
    result = recall_code("widget_factory", db_path=str(db), repo="repo", repo_root=str(repo), max_symbols=2)
    paths = [s["path"] for s in result["symbols"]]
    assert paths == ["src/widget.py", "tests/test_widget.py"], paths
    assert [s["is_test"] for s in result["symbols"]] == [False, True]
