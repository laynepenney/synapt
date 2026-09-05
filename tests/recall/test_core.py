"""Tests for synapt recall: parser, indexing, retrieval, persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from synapt.recall import (
    TranscriptChunk,
    TranscriptIndex,
    _extract_assistant_content,
    _extract_user_text,
    _is_real_user_message,
    parse_transcript,
)
from synapt.recall.core import (
    _detect_decision_markers,
    _extract_tool_result_text,
    _summarize_tool_result,
    _summarize_tool_input,
)

from conftest import (
    assistant_entry,
    make_test_chunks,
    progress_entry,
    system_entry,
    tool_result_entry,
    user_text_entry,
    user_text_list_entry,
    user_tool_result_entry,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Tests: _is_real_user_message
# ---------------------------------------------------------------------------

def test_is_real_user_message_string_content():
    assert _is_real_user_message(user_text_entry("hello")) is True


def test_is_real_user_message_text_block():
    assert _is_real_user_message(user_text_list_entry("hello")) is True


def test_is_real_user_message_tool_result_only():
    assert _is_real_user_message(user_tool_result_entry()) is False


def test_is_real_user_message_empty_string():
    assert _is_real_user_message(user_text_entry("")) is False
    assert _is_real_user_message(user_text_entry("   ")) is False


def test_is_real_user_message_wrong_type():
    entry = assistant_entry(text="hello")
    assert _is_real_user_message(entry) is False
    assert _is_real_user_message(progress_entry()) is False


# ---------------------------------------------------------------------------
# Tests: _extract_user_text
# ---------------------------------------------------------------------------

def test_extract_user_text_string():
    entry = user_text_entry("how do we fix the harness?")
    assert _extract_user_text(entry) == "how do we fix the harness?"


def test_extract_user_text_list():
    entry = user_text_list_entry("what about embeddings?")
    assert _extract_user_text(entry) == "what about embeddings?"


# ---------------------------------------------------------------------------
# Tests: _extract_assistant_content
# ---------------------------------------------------------------------------

def test_extract_assistant_text():
    entry = assistant_entry(text="I'll fix the bug now.")
    text, tools, files, tool_uses = _extract_assistant_content(entry)
    assert text == "I'll fix the bug now."
    assert tools == []
    assert files == []
    assert tool_uses == []


def test_extract_assistant_tool_use():
    entry = assistant_entry(
        text="Let me read the file.",
        tool_name="Read",
        file_path="/src/graph/api_index.py",
    )
    text, tools, files, tool_uses = _extract_assistant_content(entry)
    assert "Let me read the file." in text
    assert "Read" in tools
    assert "/src/graph/api_index.py" in files


def test_extract_assistant_skips_thinking():
    entry = assistant_entry(text="visible text")
    text, _, _, _ = _extract_assistant_content(entry)
    assert "internal reasoning" not in text
    assert text == "visible text"


# ---------------------------------------------------------------------------
# Tests: parse_transcript
# ---------------------------------------------------------------------------

def test_parse_transcript_basic():
    entries = [
        progress_entry("p1"),
        user_text_entry("what is the quality curve?", uuid="u1", ts="2026-02-28T10:00:00Z"),
        assistant_entry(text="The quality curve weights Cat3 examples.", uuid="a1", ts="2026-02-28T10:00:30Z"),
        progress_entry("p2"),
        user_text_entry("show me the code", uuid="u2", ts="2026-02-28T10:01:00Z"),
        assistant_entry(
            text="Here's the implementation.",
            tool_name="Read",
            file_path="/scripts/verify_quality_curve.py",
            uuid="a2",
            ts="2026-02-28T10:01:30Z",
        ),
    ]

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        path = Path(f.name)

    try:
        chunks = parse_transcript(path)
        assert len(chunks) == 2
        assert chunks[0].turn_index == 0
        assert "quality curve" in chunks[0].user_text
        assert "Cat3" in chunks[0].assistant_text
        assert chunks[1].turn_index == 1
        assert "show me the code" in chunks[1].user_text
        assert "Read" in chunks[1].tools_used
        assert "/scripts/verify_quality_curve.py" in chunks[1].files_touched
    finally:
        path.unlink()


def test_parse_transcript_filters_noise():
    entries = [
        progress_entry("p1"),
        system_entry("s1"),
        {"type": "file-history-snapshot", "uuid": "fh1", "snapshot": {}},
        {"type": "queue-operation", "uuid": "q1"},
        user_text_entry("hello", uuid="u1"),
        assistant_entry(text="hi there", uuid="a1"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        path = Path(f.name)

    try:
        chunks = parse_transcript(path)
        assert len(chunks) == 1
        assert chunks[0].user_text == "hello"
    finally:
        path.unlink()


def test_parse_transcript_tool_result_not_new_turn():
    entries = [
        user_text_entry("read the file", uuid="u1", ts="2026-02-28T10:00:00Z"),
        assistant_entry(tool_name="Read", file_path="/foo.py", uuid="a1"),
        user_tool_result_entry(uuid="u2"),
        assistant_entry(text="The file contains...", uuid="a2"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        path = Path(f.name)

    try:
        chunks = parse_transcript(path)
        assert len(chunks) == 1
        assert "The file contains" in chunks[0].assistant_text
        # BUG-2 fix: Read tool_results should NOT get "User selected:" prefix
        assert "User selected" not in chunks[0].assistant_text
        assert "Read" in chunks[0].tools_used
    finally:
        path.unlink()


def test_parse_tool_result_only_captured_for_ask_user_question():
    """Only AskUserQuestion tool_results get 'User selected:' prefix."""
    entries = [
        user_text_entry("which approach?", uuid="u1"),
        assistant_entry(
            text="Two options",
            tool_name="AskUserQuestion",
            uuid="a1",
        ),
        user_tool_result_entry(content="Option A", uuid="u2"),
        user_text_entry("now read a file", uuid="u3", ts="2026-02-28T10:05:00Z"),
        assistant_entry(
            text="Reading the file",
            tool_name="Read",
            file_path="/foo.py",
            uuid="a2",
            ts="2026-02-28T10:05:30Z",
        ),
        user_tool_result_entry(content="file contents here...", uuid="u4", ts="2026-02-28T10:06:00Z"),
        assistant_entry(text="The file has 100 lines", uuid="a3", ts="2026-02-28T10:06:30Z"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "session.jsonl"
        write_jsonl(path, entries)
        chunks = parse_transcript(path)

    # First turn: AskUserQuestion → tool_result captured
    assert "User selected" in chunks[0].assistant_text
    assert "Option A" in chunks[0].assistant_text
    # Second turn: Read → tool_result NOT captured
    assert "User selected" not in chunks[1].assistant_text
    assert "file contents" not in chunks[1].assistant_text


# ---------------------------------------------------------------------------
# Tests: decision point detection
# ---------------------------------------------------------------------------

def test_detect_decision_ask_user_question():
    """AskUserQuestion in tools_used triggers decision markers."""
    markers = _detect_decision_markers("pick one", ["Read", "AskUserQuestion"])
    assert "decision point" in markers
    assert "user choice" in markers


def test_detect_decision_exit_plan_mode():
    """ExitPlanMode in tools_used triggers plan_approved marker."""
    markers = _detect_decision_markers("", ["ExitPlanMode"])
    assert "decision point" in markers
    assert "plan approved" in markers


def test_detect_decision_keywords():
    """Explicit choice phrases in user text trigger decision_point."""
    markers = _detect_decision_markers("let's do the quality curve training", [])
    assert "decision point" in markers

    markers = _detect_decision_markers("I'll go with option A", [])
    assert "decision point" in markers


def test_detect_decision_no_duplicate_markers():
    """Both AskUserQuestion and ExitPlanMode in one turn don't duplicate decision_point."""
    markers = _detect_decision_markers("approve it", ["AskUserQuestion", "ExitPlanMode"])
    assert markers.count("decision point") == 1
    assert "user choice" in markers
    assert "plan approved" in markers


def test_detect_decision_no_false_positives():
    """Normal messages don't trigger decision markers."""
    markers = _detect_decision_markers("how does the quality curve work?", [])
    assert markers == []

    markers = _detect_decision_markers("fix the bug", ["Read", "Edit"])
    assert markers == []


def test_parse_captures_tool_result_text():
    """AskUserQuestion response text is captured in the chunk."""
    entries = [
        user_text_entry("which approach?", uuid="u1"),
        assistant_entry(
            text="I suggest two options",
            tool_name="AskUserQuestion",
            uuid="a1",
        ),
        user_tool_result_entry(content="Option A: use the cache", uuid="u2"),
        user_text_entry("great, continue", uuid="u3", ts="2026-02-28T10:05:00Z"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "session.jsonl"
        write_jsonl(path, entries)
        chunks = parse_transcript(path)

    assert len(chunks) == 2
    # First chunk should have the AskUserQuestion response text
    assert "Option A" in chunks[0].assistant_text
    assert "User selected" in chunks[0].assistant_text
    # First chunk should have decision markers
    assert "decision point" in chunks[0].tools_used
    assert "user choice" in chunks[0].tools_used


def test_parse_decision_keyword_in_user_text():
    """User choice keywords inject decision_point into tools_used."""
    entries = [
        user_text_entry("let's go with the sqlite approach", uuid="u1"),
        assistant_entry(text="OK, implementing sqlite.", uuid="a1"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "session.jsonl"
        write_jsonl(path, entries)
        chunks = parse_transcript(path)

    assert len(chunks) == 1
    assert "decision point" in chunks[0].tools_used


def test_decision_markers_boost_search():
    """Chunks with decision markers score higher for decision-related queries."""
    chunks = [
        TranscriptChunk(
            id="s1:t0", session_id="session-aaa",
            timestamp="2026-02-26T10:00:00Z", turn_index=0,
            user_text="let's use redis for caching",
            assistant_text="OK, switching to redis.",
            tools_used=["decision point"],
        ),
        TranscriptChunk(
            id="s1:t1", session_id="session-aaa",
            timestamp="2026-02-26T10:05:00Z", turn_index=1,
            user_text="how does caching invalidation work?",
            assistant_text="Cache invalidation uses TTL-based expiry.",
        ),
    ]
    index = TranscriptIndex(chunks)
    # "decision" should match the space-separated "decision point" marker
    # in chunk 0 (not just match via shared terms in both chunks)
    result = index.lookup("decision", max_chunks=2)
    assert result != ""
    assert "redis" in result.lower()


def test_decision_phrase_word_boundary():
    """Decision phrases use word-boundary matching, not substring (#224)."""
    from synapt.recall.core import _detect_decision_markers

    # Should match: exact phrase at word boundary
    assert _detect_decision_markers("let's go with sqlite", [])
    assert _detect_decision_markers("I said option a is best", [])
    assert _detect_decision_markers("yes, let's do it", [])

    # Should NOT match: substring false positives
    assert not _detect_decision_markers("the option actually is wrong", [])
    assert not _detect_decision_markers("we need options and choices", [])
    assert not _detect_decision_markers("optional feature flag", [])


def test_extract_tool_result_text():
    """_extract_tool_result_text captures content from tool_result blocks."""
    entry = user_tool_result_entry(content="Selected: Use Redis for caching")
    text = _extract_tool_result_text(entry)
    assert "Selected: Use Redis" in text


def test_extract_tool_result_text_empty():
    """Regular user messages have no tool_result text."""
    entry = user_text_entry("hello world")
    text = _extract_tool_result_text(entry)
    assert text == ""


def test_extract_tool_result_text_nested_list():
    """_extract_tool_result_text handles nested list content format."""
    entry = {
        "type": "user",
        "uuid": "u99",
        "timestamp": "2026-02-28T10:00:00Z",
        "sessionId": "test-session-001",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_456",
                    "content": [
                        {"type": "text", "text": "Option B: use PostgreSQL"},
                        {"type": "text", "text": "with connection pooling"},
                    ],
                }
            ],
        },
    }
    text = _extract_tool_result_text(entry)
    assert "Option B" in text
    assert "connection pooling" in text


# ---------------------------------------------------------------------------
# Tests: deduplication
# ---------------------------------------------------------------------------

def test_parse_transcript_deduplication():
    entries = [
        user_text_entry("first question", uuid="u1", ts="2026-02-28T10:00:00Z"),
        assistant_entry(text="first answer", uuid="a1", ts="2026-02-28T10:00:30Z"),
        user_text_entry("second question", uuid="u2", ts="2026-02-28T10:01:00Z"),
        assistant_entry(text="second answer", uuid="a2", ts="2026-02-28T10:01:30Z"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        path = tmpdir / "test-session.jsonl"

        write_jsonl(path, entries[:2])
        seen = set()
        chunks1 = parse_transcript(path, seen_uuids=seen)
        assert len(chunks1) == 1

        write_jsonl(path, entries)
        chunks2 = parse_transcript(path, seen_uuids=seen)
        assert len(chunks2) == 1
        assert "second question" in chunks2[0].user_text


# ---------------------------------------------------------------------------
# Tests: TranscriptIndex.lookup
# ---------------------------------------------------------------------------

def test_lookup_global_bm25():
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup("quality curve", max_chunks=3)
    assert "quality curve" in result.lower() or "hermite" in result.lower()
    assert "Past session context:" in result


def test_lookup_progressive():
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup("harness bug", max_chunks=2, max_sessions=1)
    assert "harness" in result.lower() or "XCTAssertNil" in result


def test_lookup_no_results():
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup("kubernetes deployment yaml")
    assert result == ""


def test_lookup_token_budget():
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup("quality curve", max_chunks=10, max_tokens=50)
    assert len(result) < 400


def test_format_results_token_budget_stops_after_first_oversized_block(monkeypatch):
    """The first useful block may exceed the approximation; later blocks may not."""
    monkeypatch.setenv("SYNAPT_DISABLE_DEDUP", "1")
    chunks = [
        TranscriptChunk(
            id="alpha:t0", session_id="alpha-session",
            timestamp="2026-08-27T10:00:00Z", turn_index=0,
            user_text="deployment alpha witness",
            assistant_text="alpha evidence " * 60,
        ),
        TranscriptChunk(
            id="beta:t0", session_id="beta-session",
            timestamp="2026-08-27T11:00:00Z", turn_index=0,
            user_text="deployment beta witness",
            assistant_text="beta evidence " * 60,
        ),
    ]
    index = TranscriptIndex(chunks)

    small = index._format_results([(0, 2.0), (1, 1.0)], max_tokens=10)
    large = index._format_results([(0, 2.0), (1, 1.0)], max_tokens=10_000)

    assert ("alpha evidence" in small) != ("beta evidence" in small)
    assert "alpha evidence" in large
    assert "beta evidence" in large


def _make_token_budget_ladder_chunks():
    return [
        TranscriptChunk(
            id=f"s{i}:t0", session_id=f"session-{i}",
            timestamp=f"2026-08-{10 + i:02d}T12:00:00Z", turn_index=0,
            user_text=f"deployment witness {i}",
            assistant_text="deployment evidence " + f"unique{i} " * 45,
        )
        for i in range(10)
    ]


def test_lookup_max_tokens_ladder_is_monotone_and_bounded(monkeypatch):
    """Budget changes move payload size without exceeding one-block tolerance."""
    monkeypatch.setenv("SYNAPT_DISABLE_CLUSTERS", "1")
    monkeypatch.setenv("SYNAPT_DISABLE_DEDUP", "1")
    monkeypatch.setenv("SYNAPT_DISABLE_BOOSTS", "1")
    chunks = _make_token_budget_ladder_chunks()
    budgets = (10, 500, 2400, 100000)
    results = [
        TranscriptIndex(chunks).lookup(
            "deployment", max_chunks=5, max_tokens=budget,
            threshold_ratio=0, half_life=0,
        )
        for budget in budgets
    ]
    sizes = [len(result) for result in results]

    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[2]

    # The formatter intentionally returns one useful block even when that
    # block alone exceeds the approximate budget. That makes the explicit
    # tolerance the largest first block plus the fixed header, and no more.
    index = TranscriptIndex(chunks)
    first_block_tokens = max(
        len(index._format_chunk_block(i, query="deployment")) // 4
        for i in range(len(index.chunks))
    )
    header_tokens = len("Past session context:") // 4 + 1
    for budget, result in zip(budgets, results):
        assert len(result) // 4 <= budget + first_block_tokens + header_tokens


def test_lookup_max_chunks_ladder_control_is_unchanged(monkeypatch):
    """The parameter control from #993 continues to move independently."""
    monkeypatch.setenv("SYNAPT_DISABLE_CLUSTERS", "1")
    monkeypatch.setenv("SYNAPT_DISABLE_DEDUP", "1")
    monkeypatch.setenv("SYNAPT_DISABLE_BOOSTS", "1")
    index = TranscriptIndex(_make_token_budget_ladder_chunks())
    sizes = [
        len(index.lookup(
            "deployment", max_chunks=max_chunks, max_tokens=100000,
            threshold_ratio=0, half_life=0,
        ))
        for max_chunks in (1, 3, 5, 10)
    ]
    assert sizes == [979, 2895, 4811, 4811]


def test_lookup_empty_index():
    index = TranscriptIndex([])
    assert index.lookup("anything") == ""


# ---------------------------------------------------------------------------
# Tests: search diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_empty_index():
    """Empty index sets diagnostics with reason 'empty_index'."""
    index = TranscriptIndex([])
    index.lookup("anything")
    diag = index._last_diagnostics
    assert diag is not None
    assert diag.reason == "empty_index"
    assert "empty" in diag.format_message().lower()


def test_diagnostics_no_matches():
    """Query with no keyword overlap sets 'no_matches' diagnostics."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    index.lookup("kubernetes deployment yaml")
    diag = index._last_diagnostics
    assert diag is not None
    assert diag.reason == "no_matches"
    assert diag.total_chunks == len(chunks)
    assert diag.total_sessions > 0
    msg = diag.format_message()
    assert "no chunks matched" in msg.lower()
    assert "recall_stats" in msg


def test_diagnostics_threshold_still_returns_top():
    """Threshold filtering keeps the top scorer (cutoff is relative to top score)."""
    chunks = [
        TranscriptChunk(
            id="s1:t0", session_id="session-aaa",
            timestamp="2026-02-26T10:00:00Z", turn_index=0,
            user_text="zebra migration patterns in africa",
            assistant_text="Zebra herds migrate seasonally. Zebra zebra zebra.",
        ),
        TranscriptChunk(
            id="s1:t1", session_id="session-aaa",
            timestamp="2026-02-26T10:05:00Z", turn_index=1,
            user_text="what about the zebra library?",
            assistant_text="It handles HTTP routing.",
        ),
    ]
    index = TranscriptIndex(chunks)
    # Even with threshold_ratio=0.99, the top scorer always survives
    # because cutoff = top_score * 0.99 <= top_score.
    result = index.lookup("zebra", max_chunks=5, threshold_ratio=0.99)
    assert result != ""
    # Diagnostics should be None (search succeeded)
    assert index._last_diagnostics is None


def test_diagnostics_cleared_on_success():
    """Successful search clears diagnostics to None."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    # First: trigger diagnostics
    index.lookup("kubernetes yaml")
    assert index._last_diagnostics is not None
    # Second: successful search should clear them
    result = index.lookup("quality curve")
    assert result != ""
    assert index._last_diagnostics is None


def test_diagnostics_date_filter_noted():
    """When date filter is active and no results, diagnostics note it."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    # Search for something that exists but filter to a date range where it doesn't
    result = index.lookup("quality curve", after="2030-01-01")
    assert result == ""
    diag = index._last_diagnostics
    assert diag is not None
    assert diag.date_filter_active


def test_diagnostics_progressive_sessions_searched():
    """Progressive mode records how many sessions were searched."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    index.lookup("nonexistent term xyz", max_sessions=2)
    diag = index._last_diagnostics
    assert diag is not None
    assert diag.reason == "no_matches"
    assert "progressive" in diag.search_mode


# ---------------------------------------------------------------------------
# Tests: date filtering
# ---------------------------------------------------------------------------

def test_lookup_after_excludes_old():
    """after='2026-02-28' should exclude Feb 26 chunks."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup("quality curve", after="2026-02-28")
    # "quality curve" only appears in session-aaa (Feb 26) — should be excluded
    assert result == ""


def test_lookup_before_excludes_new():
    """before='2026-02-27' should exclude Feb 28 chunks."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup("harness bug", before="2026-02-27")
    # "harness bug" only appears in session-bbb (Feb 28) — should be excluded
    assert result == ""


def test_lookup_date_range():
    """after + before together scopes to a date range."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    # Range covers only Feb 28
    result = index.lookup("harness", after="2026-02-28", before="2026-03-01")
    assert "harness" in result.lower()


def test_lookup_date_filter_with_datetime():
    """Full datetime strings work for filtering."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    # After 14:03 on Feb 28 — should only get the t1 chunk (14:05), not t0 (14:00)
    result = index.lookup("eval", after="2026-02-28T14:03:00Z")
    assert "eval" in result.lower()
    # The harness bug chunk (14:00) should be excluded
    assert "XCTAssertNil" not in result


def test_lookup_no_date_filter_returns_all():
    """Without date params, all chunks are searchable (backwards compat)."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup("quality curve")
    assert result != ""
    result2 = index.lookup("harness bug")
    assert result2 != ""


def test_lookup_progressive_with_date_filter():
    """Date filtering works in progressive (max_sessions) mode."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    # "harness bug" is in session-bbb (Feb 28) — should be excluded by before
    result = index.lookup("harness bug", max_sessions=5, before="2026-02-27")
    assert result == ""
    # Same query without date filter should find it
    result2 = index.lookup("harness bug", max_sessions=5)
    assert result2 != ""


def test_lookup_empty_date_range_returns_nothing():
    """An inverted date range (after > before) returns no results."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup("quality curve", after="2026-03-01", before="2026-02-01")
    assert result == ""


# ---------------------------------------------------------------------------
# Tests: persistence (save/load round-trip)
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    """Legacy save (no DB) + load triggers auto-migration to SQLite."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)

    save_dir = tmp_path / "test-index"
    index.save(save_dir)

    # Legacy save writes JSONL + manifest
    assert (save_dir / "chunks.jsonl").exists()
    assert (save_dir / "manifest.json").exists()

    # load() auto-migrates to SQLite
    loaded = TranscriptIndex.load(save_dir)
    assert len(loaded.chunks) == len(chunks)

    # After migration: recall.db exists, legacy files cleaned up
    assert (save_dir / "recall.db").exists()
    assert not (save_dir / "chunks.jsonl").exists()
    assert not (save_dir / "manifest.json").exists()

    # Search still works (now via FTS5)
    result = loaded.lookup("quality curve")
    assert "quality curve" in result.lower() or "hermite" in result.lower()


def test_save_load_manifest(tmp_path):
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)

    save_dir = tmp_path / "test-index"
    index.save(save_dir)

    with open(save_dir / "manifest.json") as f:
        manifest = json.load(f)

    assert manifest["chunk_count"] == 4
    assert manifest["session_count"] == 2
    assert "build_timestamp" in manifest
    assert "chunks_hash" in manifest


def test_save_load_sqlite_roundtrip(tmp_path):
    """Save with RecallDB, load from recall.db — no legacy files involved."""
    from synapt.recall.storage import RecallDB

    chunks = make_test_chunks()

    save_dir = tmp_path / "test-index"
    save_dir.mkdir(parents=True)
    db = RecallDB(save_dir / "recall.db")
    index = TranscriptIndex(chunks, db=db)
    index.save(save_dir)

    # Only recall.db, no legacy files
    assert (save_dir / "recall.db").exists()
    assert not (save_dir / "chunks.jsonl").exists()
    assert not (save_dir / "manifest.json").exists()

    loaded = TranscriptIndex.load(save_dir)
    assert len(loaded.chunks) == len(chunks)

    result = loaded.lookup("quality curve")
    assert "quality curve" in result.lower() or "hermite" in result.lower()
    db.close()


def test_load_sqlite_uses_lazy_chunk_headers(tmp_path):
    """SQLite loads should defer full chunk hydration until needed."""
    from synapt.recall.storage import RecallDB

    chunks = make_test_chunks()

    save_dir = tmp_path / "test-index"
    save_dir.mkdir(parents=True)
    db = RecallDB(save_dir / "recall.db")
    index = TranscriptIndex(chunks, db=db)
    index.save(save_dir)

    loaded = TranscriptIndex.load(save_dir)
    assert loaded._lazy_chunks is True
    assert any(not c.user_text and not c.assistant_text for c in loaded.chunks)

    result = loaded.lookup("quality curve")
    assert "quality curve" in result.lower() or "hermite" in result.lower()
    db.close()


def test_lookup_via_fts5():
    """TranscriptIndex with a RecallDB uses FTS5, not BM25 fallback."""
    from synapt.recall.storage import RecallDB

    chunks = make_test_chunks()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "recall.db"
        db = RecallDB(db_path)
        db.save_chunks(chunks)

        index = TranscriptIndex(chunks, db=db)

        # Verify FTS5 is active (BM25 should be None)
        assert index._bm25 is None

        # Global search
        result = index.lookup("quality curve", max_chunks=3)
        assert "quality curve" in result.lower() or "hermite" in result.lower()

        # Progressive search (max_sessions limits)
        result2 = index.lookup("harness bug", max_chunks=2, max_sessions=1)
        assert "harness" in result2.lower()

        # File search
        result3 = index.lookup_files("verify_quality_curve.py")
        assert result3 != ""

        # Date filtering
        result4 = index.lookup("quality curve", after="2026-02-28")
        assert result4 == ""  # quality curve is in Feb 26 session

        db.close()


def test_save_preserves_embeddings():
    """save_chunks preserves existing embeddings by matching chunk IDs."""
    from synapt.recall.storage import RecallDB

    chunks = make_test_chunks()

    with tempfile.TemporaryDirectory() as tmpdir:
        db = RecallDB(Path(tmpdir) / "recall.db")
        db.save_chunks(chunks)

        # Store a fake embedding for the first chunk
        rowids = list(db.get_chunk_id_rowid_map().values())
        fake_emb = [0.1] * 384
        db.save_embeddings({rowids[0]: fake_emb})
        assert db.has_embeddings()

        # Re-save the same chunks — embeddings should survive
        db.save_chunks(chunks)
        assert db.has_embeddings()

        new_rowids = list(db.get_chunk_id_rowid_map().values())
        embs = db.get_embeddings(new_rowids)
        assert len(embs) == 1  # Only the one we stored
        # Verify the actual embedding values survived (float32 precision)
        preserved = list(embs.values())[0]
        assert len(preserved) == 384
        for i in range(384):
            assert abs(preserved[i] - fake_emb[i]) < 1e-6
        db.close()


# ---------------------------------------------------------------------------
# Tests: stats
# ---------------------------------------------------------------------------

def test_stats():
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    stats = index.stats()

    assert stats["chunk_count"] == 4
    assert stats["session_count"] == 2
    assert stats["date_range"]["earliest"] == "2026-02-26T10:00:00Z"
    assert stats["date_range"]["latest"] == "2026-02-28T14:05:00Z"
    assert stats["total_tools_used"] >= 2
    assert stats["total_files_touched"] >= 1


def test_stats_empty():
    index = TranscriptIndex([])
    stats = index.stats()
    assert stats["chunk_count"] == 0
    assert stats["session_count"] == 0


# ---------------------------------------------------------------------------
# Tests: embedding visibility (Improvement 5)
# ---------------------------------------------------------------------------

def test_stats_includes_embedding_status():
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    stats = index.stats()
    assert "embeddings_active" in stats
    assert stats["embeddings_active"] is False
    assert stats["embedding_provider"] is None


# ---------------------------------------------------------------------------
# Tests: recency decay (Improvement 1)
# ---------------------------------------------------------------------------

def test_recency_decay_favors_newer():
    """Chunks with the same BM25 relevance should rank newer-first with decay."""
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    old_ts = (now - timedelta(days=60)).isoformat()
    new_ts = (now - timedelta(days=1)).isoformat()

    chunks = [
        TranscriptChunk(
            id="old:t0", session_id="session-old",
            timestamp=old_ts, turn_index=0,
            user_text="quality curve hermite spline zones",
            assistant_text="The quality curve uses zones.",
        ),
        TranscriptChunk(
            id="new:t0", session_id="session-new",
            timestamp=new_ts, turn_index=0,
            user_text="quality curve hermite spline zones",
            assistant_text="Updated quality curve uses zones.",
        ),
    ]
    index = TranscriptIndex(chunks)
    # Use a fixed "now" via internal method to verify ordering
    scores = index._bm25.score(
        __import__("synapt.recall.bm25", fromlist=["_tokenize"])._tokenize("quality curve")
    )
    decayed = index._apply_recency_decay(scores, half_life=30.0, now=now)
    # Find which index corresponds to the newer chunk
    new_idx = next(i for i, c in enumerate(index.chunks) if c.id == "new:t0")
    old_idx = next(i for i, c in enumerate(index.chunks) if c.id == "old:t0")
    assert decayed[new_idx] > decayed[old_idx]


def test_recency_decay_zero_disables():
    """half_life=0 returns scores unchanged."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    scores = [1.0, 2.0, 3.0, 4.0]
    result = index._apply_recency_decay(scores, half_life=0)
    assert result == [1.0, 2.0, 3.0, 4.0]


def test_recency_decay_backwards_compatible():
    """Default half_life=30 doesn't break existing lookups that find results."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup("quality curve")
    assert result != ""


# ---------------------------------------------------------------------------
# Tests: relevance threshold (Improvement 3)
# ---------------------------------------------------------------------------

def test_threshold_filters_low_scores():
    """Results with scores far below the top score should be filtered."""
    chunks = [
        TranscriptChunk(
            id="strong:t0", session_id="sess-strong-aaa",
            timestamp="2026-02-28T10:00:00Z", turn_index=0,
            user_text="quality curve hermite spline zone weighting",
            assistant_text="The quality curve uses hermite spline with three zones and weighting.",
        ),
        TranscriptChunk(
            id="weak:t0", session_id="sess-weak-bbb000",
            timestamp="2026-02-28T11:00:00Z", turn_index=0,
            user_text="the weather is nice today",
            assistant_text="Indeed the weather is pleasant.",
        ),
    ]
    index = TranscriptIndex(chunks)

    # With high threshold, weak match gets filtered
    result_strict = index.lookup(
        "quality curve hermite spline", max_chunks=10, threshold_ratio=0.5
    )
    assert "sess-str" in result_strict
    assert "sess-wea" not in result_strict


def test_threshold_zero_disables():
    """threshold_ratio=0 disables filtering."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup("quality curve", threshold_ratio=0.0)
    assert result != ""


def test_threshold_can_use_preboost_reference_scores():
    """Thresholding can preserve hits based on their pre-boost scores."""
    index = TranscriptIndex(make_test_chunks())

    filtered = index._apply_threshold_with_diagnostics(
        candidates=[(0, 10.0), (1, 0.5)],
        reference_candidates=[(0, 10.0), (1, 3.0)],
        threshold_ratio=0.2,
        search_mode="bm25_global",
    )

    assert filtered == [(0, 10.0), (1, 0.5)]


def test_status_threshold_uses_preboost_scores_for_filtering():
    """Status recency bias should not suppress otherwise-relevant transcript hits."""
    from datetime import datetime, timezone

    now = datetime(2026, 4, 4, tzinfo=timezone.utc)
    chunks = [
        TranscriptChunk(
            id="journal:t0", session_id="sess-journal",
            timestamp=now.isoformat(), turn_index=-1,
            user_text="Session focus: deploy work",
            assistant_text="Next steps: Deploy the api service and confirm rollout",
        ),
        TranscriptChunk(
            id="transcript:t0", session_id="sess-transcript",
            timestamp="2026-01-04T00:00:00+00:00", turn_index=0,
            user_text="How do we deploy the api service safely?",
            assistant_text="Apollo wrote the deploy checklist and rollback plan for the api service.",
        ),
    ]
    index = TranscriptIndex(chunks, use_embeddings=False)
    index._db = None
    index._rowid_to_idx = {}
    index._bm25.score = lambda _tokens: [10.0, 6.0]
    index._apply_recency_decay = lambda scores, half_life=0.0, now=None: [10.0, 0.5]
    index._rerank_candidates = lambda _query, ranked: ranked
    index._expand_cross_session = lambda ranked, _max_chunks: ranked
    captured = {}

    def _capture_format(ranked, *_args, **_kwargs):
        captured["ranked"] = ranked
        return "captured"

    index._format_results = _capture_format

    result = index.lookup(
        "what's pending deploy api service",
        max_chunks=5,
        threshold_ratio=0.5,
        now=now,
    )

    assert result == "captured"
    assert [idx for idx, _ in captured["ranked"]] == [0, 1]


# ---------------------------------------------------------------------------
# Tests: chunk context window (Improvement 2)
# ---------------------------------------------------------------------------

def test_format_includes_preceding_context():
    """Results for non-first turns include preceding turn's user text."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    # "Cat3 weighting" is in session-aaa:t1 (turn_index=1)
    result = index.lookup("Cat3 weighting", max_chunks=5, max_tokens=2000)
    assert "previously asked" in result.lower()


def test_format_no_context_for_first_turn():
    """First turns (turn_index=0) should not have preceding context."""
    # Create a single turn-0 chunk so it's the only result
    chunks = [
        TranscriptChunk(
            id="solo:t0", session_id="session-solo",
            timestamp="2026-02-28T10:00:00Z", turn_index=0,
            user_text="quality curve hermite spline",
            assistant_text="The quality curve uses a Hermite spline.",
        ),
    ]
    index = TranscriptIndex(chunks)
    result = index.lookup("quality curve hermite", max_chunks=1)
    assert "previously asked" not in result.lower()


# ---------------------------------------------------------------------------
# Tests: file lookup (Improvement 4)
# ---------------------------------------------------------------------------

def test_lookup_files_partial_match():
    """lookup_files matches partial file paths."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup_files("verify_quality_curve.py")
    assert result != ""
    assert "quality" in result.lower() or "cat3" in result.lower()


def test_lookup_files_substring_match():
    """lookup_files matches substring within full path."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup_files("swift_errors")
    assert result != ""
    assert "harness" in result.lower() or "swift" in result.lower()


def test_lookup_files_no_match():
    """lookup_files returns empty string when no files match."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    result = index.lookup_files("nonexistent_file.rs")
    assert result == ""


def test_lookup_files_with_date_filter():
    """lookup_files respects date filtering."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    # swift_errors.py is in session-bbb (Feb 28)
    result = index.lookup_files("swift_errors", before="2026-02-27")
    assert result == ""
    result2 = index.lookup_files("swift_errors", after="2026-02-28")
    assert result2 != ""


# ---------------------------------------------------------------------------
# Tests: session browsing (Improvement 6)
# ---------------------------------------------------------------------------

def test_list_sessions_returns_all():
    """list_sessions returns session summaries ordered newest-first."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    sessions = index.list_sessions()
    assert len(sessions) == 2
    # Newest first: session-bbb (Feb 28) before session-aaa (Feb 26)
    assert sessions[0]["session_id"] == "session-bbb"
    assert sessions[1]["session_id"] == "session-aaa"
    assert sessions[0]["turn_count"] == 2
    assert sessions[0]["date"] == "2026-02-28"


def test_list_sessions_with_max():
    """list_sessions respects max_sessions limit."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    sessions = index.list_sessions(max_sessions=1)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "session-bbb"


def test_list_sessions_with_date_filter():
    """list_sessions respects date filtering."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    sessions = index.list_sessions(after="2026-02-28")
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "session-bbb"


def test_list_sessions_includes_first_message():
    """list_sessions includes the first user message as summary."""
    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)
    sessions = index.list_sessions()
    # session-bbb first msg: "fix the harness bug in swift tests"
    assert "harness" in sessions[0]["first_message"].lower()


def test_list_sessions_empty_index():
    """list_sessions returns empty list for empty index."""
    index = TranscriptIndex([])
    assert index.list_sessions() == []


def test_list_sessions_batch_hydrates_candidate_sessions(tmp_path, monkeypatch):
    """Lazy session listing must not issue one hydration query per header."""
    from unittest.mock import Mock

    from synapt.recall.storage import RecallDB

    directory = tmp_path / "index"
    db = RecallDB(directory / "recall.db")
    try:
        db.save_chunks(make_test_chunks())
    finally:
        db.close()

    index = TranscriptIndex.load(directory, use_embeddings=False)
    batch = Mock(wraps=index._db.load_chunks_by_rowids)
    monkeypatch.setattr(index._db, "load_chunks_by_rowids", batch)
    monkeypatch.setattr(
        index,
        "_get_chunk",
        lambda _idx: (_ for _ in ()).throw(
            AssertionError("session listing hydrated one chunk at a time")
        ),
    )

    sessions = index.list_sessions()

    assert len(sessions) == 2
    batch.assert_called_once()
    index._db.close()


def test_materialize_all_chunks_batch_hydrates_instead_of_one_query_per_chunk(
    tmp_path, monkeypatch
):
    """recall#435: _materialize_all_chunks looped _get_chunk(i) one at a time,
    issuing one single-row DB query per chunk (335,524 calls / 40.1s cumtime
    measured against a 167,762-chunk store, the dominant frame in `stats`'s
    cold-start cost). load_chunks_by_rowids already exists and is already used
    by list_sessions (test above) -- this wires the same batch path into
    _materialize_all_chunks instead of inventing new plumbing."""
    from unittest.mock import Mock

    from synapt.recall.storage import RecallDB

    directory = tmp_path / "index"
    db = RecallDB(directory / "recall.db")
    try:
        db.save_chunks(make_test_chunks())
    finally:
        db.close()

    index = TranscriptIndex.load(directory, use_embeddings=False)
    batch = Mock(wraps=index._db.load_chunks_by_rowids)
    monkeypatch.setattr(index._db, "load_chunks_by_rowids", batch)
    monkeypatch.setattr(
        index,
        "_get_chunk",
        lambda _idx: (_ for _ in ()).throw(
            AssertionError("materialize_all_chunks hydrated one chunk at a time")
        ),
    )

    chunks = index._materialize_all_chunks()

    assert len(chunks) == len(make_test_chunks())
    assert all(c.user_text or c.assistant_text or c.tool_content for c in chunks)
    batch.assert_called_once()

    # A batch-loaded chunk must refresh the SAME two side-indexes the old
    # per-chunk _get_chunk path refreshed: index.sessions (the per-session
    # chunk list search_and other paths read) and _turn_lookup (turn-indexed
    # access). Refreshing only self.chunks and leaving these pointing at the
    # pre-hydration stub is silent -- nothing in the assertions above would
    # catch it, since len()/user_text/batch-call-count are all unaffected by
    # a side-index left stale. Identity (`is`), not equality: a stub and its
    # hydrated replacement can compare equal on shared fields while being
    # two different objects, so only `is` proves the refresh actually ran.
    for idx, chunk in enumerate(chunks):
        session_chunks = index.sessions[chunk.session_id]
        matching = [c for c in session_chunks if c.id == chunk.id]
        assert len(matching) == 1
        assert matching[0] is chunk, (
            f"index.sessions[{chunk.session_id!r}] still holds the pre-hydration "
            f"chunk for {chunk.id!r}, not the batch-loaded one"
        )
        if chunk.turn_index >= 0:
            assert index._turn_lookup[(chunk.session_id, chunk.turn_index)] is chunk, (
                f"_turn_lookup[{(chunk.session_id, chunk.turn_index)!r}] still holds "
                f"the pre-hydration chunk, not the batch-loaded one"
            )

    index._db.close()


def test_materialize_all_chunks_updates_session_list_in_constant_time_per_chunk(
    tmp_path, monkeypatch
):
    """recall#435 follow-on: the batch-hydration fix (above) still updated
    session_chunks via `for pos, existing in enumerate(session_chunks): if
    existing.id == loaded.id`, a linear scan run ONCE PER HYDRATED CHUNK --
    O(session_length) per chunk, O(session_length^2) total for one large
    session. Measured as the new dominant frame (22.3s tottime) once the DB
    N+1 from the batch fix was resolved.

    Wall-clock timing assertions are flaky under system load (see
    test_oversize_catchup's own 2s-deadline flake, diagnosed the same day
    this test was written) -- so this counts id-COMPARISONS directly via a
    string subclass, rather than asserting a time bound. A single 400-chunk
    session: O(n) touches on the order of n comparisons; the un-fixed
    O(n^2) scan touches up to n^2/2 = 80,000. The bound below (1,500) is
    comfortably above n for a linear fix and comfortably below n^2/2 for
    the un-fixed scan, so it discriminates cleanly either way."""
    from unittest.mock import Mock

    from synapt.recall.storage import RecallDB

    class CountingStr(str):
        """A str whose __eq__ increments a shared counter -- lets the test
        count exactly how many id-comparisons _materialize_all_chunks makes
        without depending on wall-clock time."""

        def __eq__(self, other):
            CountingStr.comparisons += 1
            return str.__eq__(self, other)

        def __hash__(self):
            return str.__hash__(self)

        comparisons = 0

    n = 400
    chunks = [
        TranscriptChunk(
            id=f"big:{i}",
            session_id="big-session",
            timestamp=f"2026-03-01T{(i // 60) % 24:02d}:{i % 60:02d}:00Z",
            turn_index=i,
            user_text=f"turn {i} question",
            assistant_text=f"turn {i} answer",
        )
        for i in range(n)
    ]

    directory = tmp_path / "index"
    db = RecallDB(directory / "recall.db")
    try:
        db.save_chunks(chunks)
    finally:
        db.close()

    index = TranscriptIndex.load(directory, use_embeddings=False)

    # Relabel every stub chunk's id with the counting subclass AFTER load, so
    # the counter measures only _materialize_all_chunks's own comparisons,
    # not construction/loading overhead. The DB-loaded replacement chunks
    # need the SAME treatment: a chunk loaded from load_chunks_by_rowids has
    # a plain str id (built fresh from the DB row), so once it replaces a
    # stub in session_chunks, a naive relabel-only-the-stubs approach stops
    # seeing comparisons at that position after its first replacement --
    # undercounting a real scan that still runs. Wrap the batch loader's
    # return value to relabel loaded chunks' ids too, so every comparison
    # for the life of the call is visible to the counter.
    real_batch_load = index._db.load_chunks_by_rowids

    def counting_batch_load(rowids):
        result = real_batch_load(rowids)
        for chunk in result.values():
            chunk.id = CountingStr(chunk.id)
        return result

    batch = Mock(wraps=counting_batch_load)
    monkeypatch.setattr(index._db, "load_chunks_by_rowids", batch)

    for c in index.chunks:
        c.id = CountingStr(c.id)
    CountingStr.comparisons = 0

    materialized = index._materialize_all_chunks()

    assert len(materialized) == n
    assert all(c.user_text for c in materialized)
    assert CountingStr.comparisons < 1500, (
        f"{CountingStr.comparisons} id-comparisons for {n} chunks in one "
        f"session -- looks like an O(session_length) scan per hydrated "
        f"chunk, not the O(1) position lookup this test guards"
    )
    index._db.close()


# ---------------------------------------------------------------------------
# Tests: build_index incremental change detection
# ---------------------------------------------------------------------------

def test_build_index_reparses_changed_files():
    """Incremental build re-parses files whose mtime or size changed."""
    import os
    import time
    from synapt.recall.core import build_index

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write a transcript with 2 turns
        entries = [
            user_text_entry("first question", uuid="u1", ts="2026-03-01T10:00:00Z"),
            assistant_entry(text="first answer", uuid="a1", ts="2026-03-01T10:00:30Z"),
            user_text_entry("second question", uuid="u2", ts="2026-03-01T10:01:00Z"),
            assistant_entry(text="second answer", uuid="a2", ts="2026-03-01T10:01:30Z"),
        ]
        transcript = tmpdir / "test-session.jsonl"
        write_jsonl(transcript, entries)

        # First build — no manifest, should parse everything
        index1 = build_index(transcript.parent)
        assert len(index1.chunks) == 2

        # Capture manifest-style source_files (simulates what cli.py saves)
        old_manifest = {
            "source_files": [{
                "name": transcript.name,
                "mtime": os.path.getmtime(transcript),
                "size": transcript.stat().st_size,
            }]
        }

        # Ensure mtime changes (HFS+ has 1s granularity)
        time.sleep(1.1)

        # Append a third turn
        with open(transcript, "a", encoding="utf-8") as f:
            f.write(json.dumps(user_text_entry(
                "third question", uuid="u3", ts="2026-03-01T10:02:00Z",
            )) + "\n")
            f.write(json.dumps(assistant_entry(
                text="third answer", uuid="a3", ts="2026-03-01T10:02:30Z",
            )) + "\n")

        # Incremental build with old manifest — should detect change and re-parse
        index2 = build_index(transcript.parent, incremental_manifest=old_manifest)
        assert len(index2.chunks) == 3, (
            f"Expected 3 chunks (file grew), got {len(index2.chunks)}"
        )


def test_build_index_skips_unchanged_files():
    """Incremental build skips files whose mtime and size match the manifest."""
    import os
    from synapt.recall.core import build_index

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        entries = [
            user_text_entry("hello", uuid="u1", ts="2026-03-01T10:00:00Z"),
            assistant_entry(text="hi", uuid="a1", ts="2026-03-01T10:00:30Z"),
        ]
        transcript = tmpdir / "test-session.jsonl"
        write_jsonl(transcript, entries)

        # Build once to get the manifest data
        index1 = build_index(transcript.parent)
        assert len(index1.chunks) == 1

        manifest = {
            "source_files": [{
                "name": transcript.name,
                "mtime": os.path.getmtime(transcript),
                "size": transcript.stat().st_size,
            }]
        }

        # Incremental build with matching manifest — should skip, return 0 chunks
        index2 = build_index(transcript.parent, incremental_manifest=manifest)
        assert len(index2.chunks) == 0, (
            f"Expected 0 chunks (file unchanged), got {len(index2.chunks)}"
        )


def test_build_index_skips_oversize_file_and_reports_it():
    """One pathological transcript must not hold up the rest of a build, and
    the skip must be visible rather than silently dropped.

    Before this: no per-file size cap existed anywhere in the builder, so a
    single runaway transcript in a source directory would be handed straight
    to parse_transcript() with no way to opt out short of never building that
    store again.
    """
    from synapt.recall.core import build_index

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        normal = tmpdir / "normal-session.jsonl"
        write_jsonl(normal, [
            user_text_entry("hello", uuid="u1", ts="2026-03-01T10:00:00Z"),
            assistant_entry(text="hi", uuid="a1", ts="2026-03-01T10:00:30Z"),
        ])

        oversize = tmpdir / "oversize-session.jsonl"
        write_jsonl(oversize, [
            user_text_entry("this one is too big", uuid="u2", ts="2026-03-01T11:00:00Z"),
            assistant_entry(text="ok", uuid="a2", ts="2026-03-01T11:00:30Z"),
        ])
        oversize_size = oversize.stat().st_size
        ceiling = oversize_size - 1  # strictly below the oversize file's size

        index = build_index(tmpdir, max_file_bytes=ceiling)

        # The build completes and the NORMAL file is still parsed — one
        # pathological file must not take the rest of the build down with it.
        assert len(index.chunks) == 1, (
            f"Expected the normal file's 1 chunk despite the oversize sibling, "
            f"got {len(index.chunks)}"
        )
        assert index.chunks[0].session_id == "normal-session"

        # The skip is REPORTED, not silent: visible on the returned index
        # with the file's real size, distinct from the ordinary
        # already-indexed skip path (which would make it look identical to a
        # file that WAS successfully searched).
        assert hasattr(index, "skipped_oversize"), (
            "build_index() must expose which files it skipped for size, "
            "not just drop them"
        )
        assert len(index.skipped_oversize) == 1
        skipped = index.skipped_oversize[0]
        assert skipped["name"] == "oversize-session.jsonl"
        assert skipped["size"] == oversize_size


def test_build_index_default_ceiling_does_not_affect_normal_files():
    """The default ceiling must not touch any realistically-sized transcript."""
    from synapt.recall.core import build_index

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        transcript = tmpdir / "normal-session.jsonl"
        write_jsonl(transcript, [
            user_text_entry("hello", uuid="u1", ts="2026-03-01T10:00:00Z"),
            assistant_entry(text="hi", uuid="a1", ts="2026-03-01T10:00:30Z"),
        ])

        index = build_index(tmpdir)  # no explicit ceiling — production default

        assert len(index.chunks) == 1
        assert index.skipped_oversize == []


def test_parse_transcript_skips_oversize_line_and_reports_it():
    """One pathological JSONL LINE inside an otherwise-small file must not be
    parsed at all -- a whole-file byte ceiling cannot see this coming, since
    the file stays small while one line inside it is huge. Measured: a real
    giant single line drives peak RSS to ~3.5-3.7x the file's own size, far
    worse than normal content's ~0.44x."""
    from synapt.recall.core import parse_transcript

    with tempfile.TemporaryDirectory() as tmpdir:
        transcript = Path(tmpdir) / "session.jsonl"
        huge_content = "x" * 5000
        write_jsonl(transcript, [
            user_text_entry("hello", uuid="u1", ts="2026-03-01T10:00:00Z"),
            assistant_entry(text="hi", uuid="a1", ts="2026-03-01T10:00:30Z"),
            user_tool_result_entry(content=huge_content, uuid="tr1", ts="2026-03-01T10:01:00Z"),
            user_text_entry("after", uuid="u2", ts="2026-03-01T10:02:00Z"),
            assistant_entry(text="ok", uuid="a2", ts="2026-03-01T10:02:30Z"),
        ])

        skipped_lines: list[dict] = []
        chunks = parse_transcript(
            transcript, max_line_bytes=1000, skipped_lines=skipped_lines,
        )

        assert len(skipped_lines) == 1, "the oversize line must be reported, not silently dropped"
        assert skipped_lines[0]["size"] > 1000

        # The huge blob never entered any chunk field -- skipped, not merely
        # truncated after the fact (which would still show a 3000-char prefix).
        joined = " ".join(
            (c.user_text or "") + (c.assistant_text or "") + (c.tool_content or "")
            for c in chunks
        )
        assert "x" * 500 not in joined

        # The surrounding ordinary turns still parsed -- one pathological
        # line does not take the rest of the file down with it.
        assert any(c.user_text == "hello" for c in chunks)
        assert any(c.user_text == "after" for c in chunks)


def test_parse_transcript_default_line_ceiling_does_not_affect_normal_lines():
    """The default line ceiling must not touch any realistically-sized turn."""
    from synapt.recall.core import parse_transcript

    with tempfile.TemporaryDirectory() as tmpdir:
        transcript = Path(tmpdir) / "session.jsonl"
        write_jsonl(transcript, [
            user_text_entry("hello", uuid="u1", ts="2026-03-01T10:00:00Z"),
            assistant_entry(text="hi", uuid="a1", ts="2026-03-01T10:00:30Z"),
        ])

        skipped_lines: list[dict] = []
        chunks = parse_transcript(transcript, skipped_lines=skipped_lines)  # no explicit ceiling

        assert len(chunks) == 1
        assert skipped_lines == []


def test_max_transcript_file_bytes_malformed_env_warns_and_falls_back(monkeypatch):
    """A bad SYNAPT_MAX_TRANSCRIPT_FILE_BYTES must not silently become the
    default with no trace -- the build still needs a working ceiling, but the
    operator needs to know their override was ignored."""
    from synapt.recall.core import _max_transcript_file_bytes, DEFAULT_MAX_TRANSCRIPT_FILE_BYTES

    monkeypatch.setenv("SYNAPT_MAX_TRANSCRIPT_FILE_BYTES", "not-a-number")
    value, warning = _max_transcript_file_bytes()

    assert value == DEFAULT_MAX_TRANSCRIPT_FILE_BYTES
    assert warning is not None
    assert "not-a-number" in warning


def test_max_transcript_file_bytes_valid_env_no_warning(monkeypatch):
    from synapt.recall.core import _max_transcript_file_bytes

    monkeypatch.setenv("SYNAPT_MAX_TRANSCRIPT_FILE_BYTES", "12345")
    value, warning = _max_transcript_file_bytes()

    assert value == 12345
    assert warning is None


def test_max_line_bytes_malformed_env_warns_and_falls_back(monkeypatch):
    from synapt.recall.core import _max_line_bytes, DEFAULT_MAX_LINE_BYTES

    monkeypatch.setenv("SYNAPT_MAX_TRANSCRIPT_LINE_BYTES", "garbage")
    value, warning = _max_line_bytes()

    assert value == DEFAULT_MAX_LINE_BYTES
    assert warning is not None
    assert "garbage" in warning


def test_build_index_surfaces_malformed_ceiling_env_as_config_warning(monkeypatch):
    """The malformed-override warning must reach the build's returned index,
    not just stdout -- the same visibility contract as skipped_oversize."""
    from synapt.recall.core import build_index

    monkeypatch.setenv("SYNAPT_MAX_TRANSCRIPT_FILE_BYTES", "not-a-number")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        write_jsonl(tmpdir / "session.jsonl", [
            user_text_entry("hello", uuid="u1", ts="2026-03-01T10:00:00Z"),
            assistant_entry(text="hi", uuid="a1", ts="2026-03-01T10:00:30Z"),
        ])

        index = build_index(tmpdir)

        assert hasattr(index, "config_warnings")
        assert any("not-a-number" in w for w in index.config_warnings)


def test_build_index_surfaces_skipped_lines_on_the_returned_index():
    """A line-level skip must reach the returned index, not just a print
    statement -- the earlier version of this fix built
    skipped_lines as a local that only ever reached stdout, so an in-process
    caller (the MCP build path) could never see it. Same always-present
    contract as skipped_oversize/config_warnings: empty when nothing was
    skipped, so a caller never needs a hasattr check."""
    from synapt.recall.core import build_index

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        transcript = tmpdir / "session.jsonl"
        huge_content = "x" * 5000
        write_jsonl(transcript, [
            user_text_entry("hello", uuid="u1", ts="2026-03-01T10:00:00Z"),
            assistant_entry(text="hi", uuid="a1", ts="2026-03-01T10:00:30Z"),
            user_tool_result_entry(content=huge_content, uuid="tr1", ts="2026-03-01T10:01:00Z"),
        ])

        index = build_index(tmpdir, max_line_bytes=1000)

        assert hasattr(index, "skipped_lines")
        assert len(index.skipped_lines) == 1
        assert index.skipped_lines[0]["size"] > 1000
        assert index.skipped_lines[0]["session_id"] == "session"

        # The file's other lines still parsed -- a line skip is not a file skip.
        assert any(c.user_text == "hello" for c in index.chunks)


def test_build_index_default_skipped_lines_is_empty_when_nothing_skipped():
    from synapt.recall.core import build_index

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        write_jsonl(tmpdir / "session.jsonl", [
            user_text_entry("hello", uuid="u1", ts="2026-03-01T10:00:00Z"),
            assistant_entry(text="hi", uuid="a1", ts="2026-03-01T10:00:30Z"),
        ])

        index = build_index(tmpdir)

        assert index.skipped_lines == []


# ---------------------------------------------------------------------------
# Tests: _summarize_tool_result
# ---------------------------------------------------------------------------

def test_summarize_bash_command_and_output():
    summary = _summarize_tool_result(
        "Bash", {"command": "pytest tests/"}, "12 passed, 3 failed\n"
    )
    assert summary.startswith("$ pytest tests/")
    assert "12 passed" in summary


def test_summarize_bash_error_gets_more_budget():
    long_error = "Traceback (most recent call last):\n" + "x" * 2000
    summary = _summarize_tool_result(
        "Bash", {"command": "python run.py"}, long_error
    )
    # Error output should get 1200 char budget instead of 600
    assert len(summary) > 600
    assert "Traceback" in summary


def test_summarize_read_is_compact():
    content = "\n".join(f"line {i}" for i in range(100))
    summary = _summarize_tool_result(
        "Read", {"file_path": "/src/auth.py"}, content
    )
    assert "Read /src/auth.py" in summary
    assert "100 lines" in summary
    assert len(summary) < 100  # Very compact


def test_summarize_write_is_compact():
    summary = _summarize_tool_result(
        "Write", {"file_path": "/src/new_file.py"}, "success"
    )
    assert summary == "Wrote /src/new_file.py"


def test_summarize_edit_shows_changes():
    summary = _summarize_tool_result(
        "Edit",
        {"file_path": "/src/auth.py", "old_string": "def login():", "new_string": "def login(user):"},
        "success",
    )
    assert "Edited /src/auth.py" in summary
    assert "login()" in summary or "login" in summary


def test_summarize_grep_shows_pattern_and_results():
    summary = _summarize_tool_result(
        "Grep", {"pattern": "TODO"}, "src/main.py:10: # TODO fix this"
    )
    assert 'Grep "TODO"' in summary
    assert "src/main.py" in summary


def test_summarize_glob_shows_count():
    summary = _summarize_tool_result(
        "Glob", {"pattern": "**/*.py"}, "a.py\nb.py\nc.py"
    )
    assert 'Glob "**/*.py"' in summary
    assert "3 files" in summary


def test_summarize_agent_truncates():
    summary = _summarize_tool_result(
        "Agent", {}, "The agent found that the file uses a factory pattern."
    )
    assert summary.startswith("Agent:")
    assert "factory pattern" in summary


def test_summarize_default_truncates():
    summary = _summarize_tool_result(
        "CustomTool", {}, "x" * 500
    )
    assert summary.startswith("CustomTool:")
    assert len(summary) <= 220  # tool name + ": " + 200 chars


def test_summarize_empty_result():
    assert _summarize_tool_result("Bash", {"command": "ls"}, "") == ""


def test_summarize_mcp_tool_strips_prefix():
    summary = _summarize_tool_result(
        "mcp__synapt__recall_search", {}, "some results"
    )
    assert summary.startswith("recall_search:")


# ---------------------------------------------------------------------------
# Tests: _summarize_tool_input
# ---------------------------------------------------------------------------

def test_summarize_input_bash():
    summary = _summarize_tool_input("Bash", {"command": "git status", "description": "Show status"})
    assert "$ git status" in summary
    assert "Show status" in summary


def test_summarize_input_edit():
    summary = _summarize_tool_input("Edit", {
        "file_path": "/x.py", "old_string": "old code", "new_string": "new code",
    })
    assert "Edit /x.py" in summary


def test_summarize_input_grep():
    summary = _summarize_tool_input("Grep", {"pattern": "import os"})
    assert 'Grep "import os"' in summary


def test_summarize_input_websearch():
    summary = _summarize_tool_input("WebSearch", {"query": "python async best practices"})
    assert "Search: python async" in summary


def test_summarize_input_unknown_returns_empty():
    assert _summarize_tool_input("Read", {"file_path": "/x.py"}) == ""


# ---------------------------------------------------------------------------
# Tests: tool_content in parser
# ---------------------------------------------------------------------------

def test_parse_captures_tool_content():
    """Parser captures summarized tool results in tool_content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript = Path(tmpdir) / "test-session-001.jsonl"

        entries = [
            user_text_entry("Run the tests"),
            assistant_entry(
                text="Running tests now.",
                tool_name="Bash",
                tool_input={"command": "pytest tests/", "description": "Run tests"},
                tool_use_id="toolu_001",
            ),
            tool_result_entry(
                tool_use_id="toolu_001",
                result="FAILED tests/test_auth.py::test_login - AssertionError",
            ),
        ]
        write_jsonl(transcript, entries)

        chunks = parse_transcript(transcript)
        assert len(chunks) == 1
        chunk = chunks[0]
        # tool_content should contain the command and error output
        assert "$ pytest tests/" in chunk.tool_content
        assert "FAILED" in chunk.tool_content or "AssertionError" in chunk.tool_content
        # tool_content should also be in the combined searchable text
        assert "pytest" in chunk.text


def test_parse_tool_content_cap():
    """tool_content is capped at 3000 chars."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript = Path(tmpdir) / "test-session-001.jsonl"

        # Generate many tool results to exceed cap
        entries = [user_text_entry("Do everything")]
        for i in range(20):
            entries.append(assistant_entry(
                text="",
                tool_name="Bash",
                tool_input={"command": f"echo {'x' * 200}"},
                tool_use_id=f"toolu_{i:03d}",
                uuid=f"a{i}",
            ))
            entries.append(tool_result_entry(
                tool_use_id=f"toolu_{i:03d}",
                result="x" * 300,
                uuid=f"tr{i}",
            ))
        write_jsonl(transcript, entries)

        chunks = parse_transcript(transcript)
        assert len(chunks) == 1
        assert len(chunks[0].tool_content) <= 3004  # 3000 + "..."
        assert chunks[0].tool_content.endswith("...")


def test_parse_records_byte_offsets():
    """Parser records byte offsets into the transcript file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript = Path(tmpdir) / "test-session-001.jsonl"

        entries = [
            user_text_entry("first question"),
            assistant_entry(text="first answer"),
            user_text_entry("second question", uuid="u2"),
            assistant_entry(text="second answer", uuid="a2"),
        ]
        write_jsonl(transcript, entries)

        chunks = parse_transcript(transcript)
        assert len(chunks) == 2
        # First chunk should start at offset 0
        assert chunks[0].byte_offset == 0
        assert chunks[0].byte_length > 0
        # Second chunk should start after the first
        assert chunks[1].byte_offset > chunks[0].byte_offset
        assert chunks[1].byte_length > 0
        # Transcript path should be set
        assert chunks[0].transcript_path == str(transcript)
        # Offsets should not overlap
        assert chunks[1].byte_offset >= chunks[0].byte_offset + chunks[0].byte_length


def test_parse_raised_caps():
    """user_text cap is 1500, assistant_text cap is 5000."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript = Path(tmpdir) / "test-session-001.jsonl"

        long_user = "x" * 2000
        long_assistant = "y" * 6000
        entries = [
            user_text_entry(long_user),
            assistant_entry(text=long_assistant),
        ]
        write_jsonl(transcript, entries)

        chunks = parse_transcript(transcript)
        assert len(chunks) == 1
        assert len(chunks[0].user_text) == 1503  # 1500 + "..."
        assert chunks[0].user_text.endswith("...")
        assert len(chunks[0].assistant_text) == 5003  # 5000 + "..."
        assert chunks[0].assistant_text.endswith("...")


# ---------------------------------------------------------------------------
# Tests: drill-down (read_turn_context)
# ---------------------------------------------------------------------------

def test_read_turn_context(tmp_path):
    """read_turn_context reads raw transcript content for a chunk."""
    transcript = tmp_path / "test-session-001.jsonl"

    entries = [
        user_text_entry("How do I fix the auth bug?"),
        assistant_entry(
            text="Let me check the code.",
            tool_name="Read",
            tool_input={"file_path": "/src/auth.py"},
            tool_use_id="toolu_read1",
        ),
        tool_result_entry(
            tool_use_id="toolu_read1",
            result="def login():\n    pass",
        ),
    ]
    write_jsonl(transcript, entries)

    chunks = parse_transcript(transcript)
    assert len(chunks) == 1

    index = TranscriptIndex(chunks)
    context = index.read_turn_context(chunks[0].id)
    assert "auth bug" in context
    assert "Let me check the code" in context
    assert "def login():" in context  # Full tool result content


def test_read_turn_context_missing_chunk():
    """read_turn_context returns error for unknown chunk ID."""
    index = TranscriptIndex([])
    result = index.read_turn_context("nonexistent:t0")
    assert "not found" in result


def test_read_turn_context_multibyte_utf8(tmp_path):
    """Drill-down reads exact bytes, not characters, so multi-byte UTF-8 stays within turn bounds."""
    transcript = tmp_path / "test-session-001.jsonl"

    # Write entries with ensure_ascii=False so emoji stay as raw UTF-8
    entries = [
        {"type": "user", "uuid": "u1", "timestamp": "2026-01-01",
         "sessionId": "test-session-001",
         "message": {"role": "user", "content": "Fix the bug \U0001f41b\U0001f41b\U0001f41b"}},
        {"type": "assistant", "uuid": "a1", "timestamp": "2026-01-01",
         "sessionId": "test-session-001",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "Done \u2705"}]}},
        {"type": "user", "uuid": "u2", "timestamp": "2026-01-02",
         "sessionId": "test-session-001",
         "message": {"role": "user", "content": "NEXT TURN sentinel"}},
        {"type": "assistant", "uuid": "a2", "timestamp": "2026-01-02",
         "sessionId": "test-session-001",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "second answer"}]}},
    ]
    with open(transcript, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    chunks = parse_transcript(transcript)
    assert len(chunks) == 2

    index = TranscriptIndex(chunks)
    context = index.read_turn_context(chunks[0].id)
    # First turn should contain the emoji but NOT the sentinel from turn 2
    assert "\U0001f41b" in context
    assert "NEXT TURN sentinel" not in context


def test_read_turn_context_deleted_file():
    """Drill-down returns friendly error when transcript file has been deleted."""
    chunk = TranscriptChunk(
        id="abc:t0", session_id="abc", timestamp="2026-01-01",
        turn_index=0, user_text="hi", assistant_text="hello",
        transcript_path="/nonexistent/path/to/deleted.jsonl",
        byte_offset=0, byte_length=100,
    )
    index = TranscriptIndex([chunk])
    result = index.read_turn_context("abc:t0")
    assert "not found" in result.lower()


def test_read_turn_context_no_offset():
    """Drill-down returns friendly error for legacy chunks with no byte offset."""
    chunk = TranscriptChunk(
        id="old:t0", session_id="old", timestamp="2026-01-01",
        turn_index=0, user_text="hi", assistant_text="hello",
        byte_offset=-1,
    )
    index = TranscriptIndex([chunk])
    result = index.read_turn_context("old:t0")
    assert "no transcript offset" in result.lower()


# ---------------------------------------------------------------------------
# Tests: TranscriptChunk new fields
# ---------------------------------------------------------------------------

def test_chunk_to_dict_includes_new_fields():
    chunk = TranscriptChunk(
        id="abc:t0", session_id="abc", timestamp="2026-01-01",
        turn_index=0, user_text="hi", assistant_text="hello",
        tool_content="$ ls\nfile1.py",
        transcript_path="/tmp/abc.jsonl",
        byte_offset=100, byte_length=500,
    )
    d = chunk.to_dict()
    assert d["tool_content"] == "$ ls\nfile1.py"
    assert d["transcript_path"] == "/tmp/abc.jsonl"
    assert d["byte_offset"] == 100
    assert d["byte_length"] == 500


def test_chunk_from_dict_handles_missing_new_fields():
    """Old serialized chunks without new fields get defaults."""
    d = {
        "id": "abc:t0", "session_id": "abc", "timestamp": "2026-01-01",
        "turn_index": 0, "user_text": "hi", "assistant_text": "hello",
        "tools_used": [], "files_touched": [],
    }
    chunk = TranscriptChunk.from_dict(d)
    assert chunk.tool_content == ""
    assert chunk.transcript_path == ""
    assert chunk.byte_offset == -1
    assert chunk.byte_length == 0


def test_chunk_build_text_includes_tool_content():
    chunk = TranscriptChunk(
        id="abc:t0", session_id="abc", timestamp="2026-01-01",
        turn_index=0, user_text="hi", assistant_text="hello",
        tool_content="$ pytest\n5 passed",
    )
    assert "pytest" in chunk.text
    assert "5 passed" in chunk.text


# ---------------------------------------------------------------------------
# Tests: tool_use_error filtering (#274)
# ---------------------------------------------------------------------------

def test_parse_filters_tool_use_error_from_tool_content():
    """tool_use_error messages are not stored in tool_content — they are noise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript = Path(tmpdir) / "test-session-001.jsonl"

        entries = [
            user_text_entry("create a task"),
            assistant_entry(
                text="Creating task now.",
                tool_name="TaskCreate",
                tool_input={"title": "My task"},
                tool_use_id="toolu_001",
            ),
            tool_result_entry(
                tool_use_id="toolu_001",
                result="<tool_use_error>Sibling tool call errored</tool_use_error>",
            ),
        ]
        write_jsonl(transcript, entries)

        chunks = parse_transcript(transcript)
        assert len(chunks) == 1
        assert "<tool_use_error>" not in chunks[0].tool_content
        assert "Sibling tool call errored" not in chunks[0].tool_content


def test_parse_keeps_successful_tool_result_alongside_error():
    """Filtering errors does not drop successful tool results in the same turn."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript = Path(tmpdir) / "test-session-001.jsonl"

        entries = [
            user_text_entry("run tests and check status"),
            assistant_entry(
                text="Running.",
                tool_name="Bash",
                tool_input={"command": "pytest tests/"},
                tool_use_id="toolu_bash",
            ),
            # Successful result — should be kept
            tool_result_entry(
                tool_use_id="toolu_bash",
                result="5 passed in 0.3s",
                uuid="tr_bash",
            ),
            assistant_entry(
                text="",
                tool_name="TaskCreate",
                tool_input={"title": "Follow-up"},
                tool_use_id="toolu_task",
                uuid="a2",
            ),
            # Error result — should be filtered
            tool_result_entry(
                tool_use_id="toolu_task",
                result="<tool_use_error>Sibling tool call errored</tool_use_error>",
                uuid="tr_task",
            ),
        ]
        write_jsonl(transcript, entries)

        chunks = parse_transcript(transcript)
        assert len(chunks) == 1
        tc = chunks[0].tool_content
        assert "5 passed" in tc
        assert "<tool_use_error>" not in tc


# ---------------------------------------------------------------------------
# Tests: knowledge node coverage gate (#284)
# ---------------------------------------------------------------------------

def test_search_knowledge_filters_weak_zero_token_matches():
    """Knowledge nodes with no query token overlap are filtered out."""
    from synapt.recall.core import TranscriptIndex

    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)

    # Simulate two knowledge nodes: one with strong coverage, one with none.
    strong_node = {
        "content": "compact journal flock lock race TOCTOU",  # 5 query tokens present
        "category": "debugging",
        "confidence": 0.8,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
    }
    unrelated_node = {
        "content": "always deploy containers promptly",  # no query tokens match
        "category": "workflow",
        "confidence": 0.8,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
    }

    from unittest.mock import patch, MagicMock
    mock_db = MagicMock()
    mock_db.knowledge_fts_search.return_value = [(1, 3.0), (2, 2.8)]
    mock_db.knowledge_by_rowid.return_value = {1: strong_node, 2: unrelated_node}
    index._db = mock_db

    # Query: "compact journal flock TOCTOU race fix" — 6 distinctive tokens
    # min_matches = max(1, round(6 * 0.2)) = max(1, 1) = 1
    # strong_node matches 5 tokens → kept
    # unrelated_node matches 0 tokens → filtered
    results = index._search_knowledge("compact journal flock TOCTOU race fix")
    assert len(results) == 1
    assert results[0]["content"] == strong_node["content"]


def test_search_knowledge_short_query_single_token_match_passes():
    """Short queries (2-3 tokens) allow single-token matches through.

    Knowledge nodes are distilled facts — a single matching token (e.g.
    a person's name) is a meaningful signal for short queries.
    """
    from synapt.recall.core import TranscriptIndex
    from unittest.mock import MagicMock

    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)

    exact_match = {
        "content": "use swift repair training loops",  # both query tokens present
        "category": "architecture",
        "confidence": 0.9,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
    }
    partial_match = {
        "content": "run swift builds in CI",  # only "swift" present, not "repair"
        "category": "tooling",
        "confidence": 0.8,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
    }

    mock_db = MagicMock()
    mock_db.knowledge_fts_search.return_value = [(1, 3.0), (2, 2.9)]
    mock_db.knowledge_by_rowid.return_value = {1: exact_match, 2: partial_match}
    index._db = mock_db

    # Query: "swift repair" — 2 tokens, min_matches = max(1, round(2*0.2)) = 1
    # Both nodes match >= 1 token, both pass the coverage gate.
    # exact_match has higher FTS score (3.0 vs 2.9) so ranks first.
    results = index._search_knowledge("swift repair")
    assert len(results) == 2
    assert results[0]["content"] == exact_match["content"]


def test_search_knowledge_entity_boost():
    """Knowledge nodes mentioning query entities get a 1.5x score boost."""
    from synapt.recall.core import TranscriptIndex
    from unittest.mock import MagicMock

    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)

    # Two nodes, both match the query tokens, but only one mentions "Caroline"
    about_caroline = {
        "id": "node-1",
        "content": "Caroline drinks dark roast coffee daily",
        "category": "preference",
        "confidence": 0.7,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
    }
    about_melanie = {
        "id": "node-2",
        "content": "Melanie drinks green tea and coffee",
        "category": "preference",
        "confidence": 0.7,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
    }

    mock_db = MagicMock()
    # Same FTS scores — without entity boost they'd rank by order
    mock_db.knowledge_fts_search.return_value = [(1, 3.0), (2, 3.0)]
    mock_db.knowledge_by_rowid.return_value = {1: about_caroline, 2: about_melanie}
    index._db = mock_db

    # Query mentions "Caroline" — node about Caroline should rank higher
    results = index._search_knowledge("What coffee does Caroline drink?")
    assert len(results) == 2
    caroline_node = [r for r in results if "Caroline" in r["content"]][0]
    melanie_node = [r for r in results if "Melanie" in r["content"]][0]
    assert caroline_node["score"] > melanie_node["score"]


def test_search_knowledge_entity_boost_can_be_disabled(monkeypatch):
    """Entity boost ablation flag should preserve base ordering."""
    from synapt.recall.core import TranscriptIndex
    from unittest.mock import MagicMock

    monkeypatch.setenv("SYNAPT_DISABLE_ENTITY_COLLECTION", "1")

    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)

    about_caroline = {
        "id": "node-1",
        "content": "Caroline drinks dark roast coffee daily",
        "category": "preference",
        "confidence": 0.7,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
    }
    about_melanie = {
        "id": "node-2",
        "content": "Melanie drinks green tea and coffee",
        "category": "preference",
        "confidence": 0.7,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
    }

    mock_db = MagicMock()
    # Melanie appears first in FTS order; without entity boost she should stay first.
    mock_db.knowledge_fts_search.return_value = [(2, 3.0), (1, 3.0)]
    mock_db.knowledge_by_rowid.return_value = {1: about_caroline, 2: about_melanie}
    index._db = mock_db

    results = index._search_knowledge("What coffee does Caroline drink?")
    assert len(results) == 2
    assert results[0]["content"] == about_melanie["content"]


def test_search_knowledge_expiry_can_be_disabled(monkeypatch):
    """Expired nodes remain searchable when knowledge-expiry ablation is enabled."""
    from synapt.recall.core import TranscriptIndex
    from unittest.mock import MagicMock

    monkeypatch.setenv("SYNAPT_DISABLE_KNOWLEDGE_EXPIRY", "1")

    chunks = make_test_chunks()
    index = TranscriptIndex(chunks)

    expired = {
        "id": "node-1",
        "content": "Caroline switched to dark roast coffee",
        "category": "preference",
        "confidence": 0.8,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
        "valid_until": "2026-01-01",
    }
    active = {
        "id": "node-2",
        "content": "Caroline still likes oat milk",
        "category": "preference",
        "confidence": 0.8,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
        "valid_until": None,
    }

    mock_db = MagicMock()
    mock_db.knowledge_fts_search.return_value = [(1, 3.0), (2, 2.9)]
    mock_db.knowledge_by_rowid.return_value = {1: expired, 2: active}
    index._db = mock_db

    results = index._search_knowledge("What coffee does Caroline like?")
    assert len(results) == 2
    assert any(r["content"] == expired["content"] for r in results)


def test_search_knowledge_skips_specificity_for_personal_profile():
    """Personal content should not be penalized for lacking code-style specificity."""
    from types import SimpleNamespace
    from synapt.recall.core import TranscriptIndex
    from unittest.mock import MagicMock

    index = TranscriptIndex(make_test_chunks())
    index._adaptive = SimpleNamespace(content_type="personal")

    abstract = {
        "id": "node-1",
        "content": "Caroline is interested in psychology and memory research",
        "category": "preference",
        "confidence": 0.8,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
    }
    code_specific = {
        "id": "node-2",
        "content": "Caroline documented psychology notes in docs/research.md after PR #25",
        "category": "preference",
        "confidence": 0.8,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
    }

    mock_db = MagicMock()
    # Same lexical score; without a specificity penalty the first FTS hit should stay first.
    mock_db.knowledge_fts_search.return_value = [(1, 3.0), (2, 3.0)]
    mock_db.knowledge_by_rowid.return_value = {1: abstract, 2: code_specific}
    index._db = mock_db

    results = index._search_knowledge("What is Caroline interested in?")

    assert len(results) == 2
    assert results[0]["content"] == abstract["content"]
    assert results[0]["score"] >= results[1]["score"]


def test_search_knowledge_temporal_filters_non_overlapping_windows():
    """Temporal queries should drop knowledge nodes outside the query window."""
    from synapt.recall.core import TranscriptIndex
    from unittest.mock import MagicMock

    index = TranscriptIndex(make_test_chunks())

    overlapping = {
        "id": "node-1",
        "content": "Caroline was using the old deploy flow in early March 2026",
        "category": "fact",
        "confidence": 0.8,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
        "valid_from": "2026-03-01",
        "valid_until": "2026-03-10",
    }
    outside_window = {
        "id": "node-2",
        "content": "Caroline switched to the new deploy flow in April 2026",
        "category": "fact",
        "confidence": 0.8,
        "source_sessions": ["sess2"],
        "updated_at": "2026-04-02",
        "valid_from": "2026-04-01",
        "valid_until": "2026-04-30",
    }

    mock_db = MagicMock()
    mock_db.knowledge_fts_search.return_value = [(2, 3.0), (1, 2.9)]
    mock_db.knowledge_by_rowid.return_value = {1: overlapping, 2: outside_window}
    index._db = mock_db

    results = index._search_knowledge(
        "What deploy flow was Caroline using in early March?",
        intent="temporal",
        after="2026-03-01",
        before="2026-03-15",
    )

    assert len(results) == 1
    assert results[0]["content"] == overlapping["content"]


def test_search_knowledge_temporal_overlap_boosts_bounded_nodes():
    """Overlapping temporal nodes should outrank timeless fallback nodes."""
    from synapt.recall.core import TranscriptIndex
    from unittest.mock import MagicMock

    index = TranscriptIndex(make_test_chunks())

    timeless = {
        "id": "node-1",
        "content": "Caroline deployed through the old pipeline",
        "category": "fact",
        "confidence": 0.8,
        "source_sessions": ["sess1"],
        "updated_at": "2026-03-05",
    }
    overlapping = {
        "id": "node-2",
        "content": "Caroline deployed through the old pipeline in March 2026",
        "category": "fact",
        "confidence": 0.8,
        "source_sessions": ["sess2"],
        "updated_at": "2026-03-05",
        "valid_from": "2026-03-01",
        "valid_until": "2026-03-31",
    }

    mock_db = MagicMock()
    mock_db.knowledge_fts_search.return_value = [(1, 3.0), (2, 3.0)]
    mock_db.knowledge_by_rowid.return_value = {1: timeless, 2: overlapping}
    index._db = mock_db

    results = index._search_knowledge(
        "Which pipeline was Caroline using in March?",
        intent="temporal",
        after="2026-03-01",
        before="2026-04-01",
    )

    assert len(results) == 2
    assert results[0]["content"] == overlapping["content"]
    assert results[0]["score"] > results[1]["score"]


def test_expand_cross_session_can_be_disabled(monkeypatch):
    """Cross-link ablation flag should skip expansion entirely."""
    from unittest.mock import MagicMock

    monkeypatch.setenv("SYNAPT_DISABLE_CROSS_LINKS", "1")

    index = TranscriptIndex(make_test_chunks())
    index._db = MagicMock()
    candidates = [(0, 1.0), (1, 0.9), (2, 0.8)]

    expanded = index._expand_cross_session(candidates, max_chunks=5)

    assert expanded == candidates
    index._db.has_chunk_links.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: date format in search results (#268, #275)
# ---------------------------------------------------------------------------

def test_format_results_shows_more_assistant_content():
    """Search results show up to 1500 chars of assistant text, not 500."""
    from synapt.recall.core import TranscriptIndex

    long_answer = "A" * 1200 + " CONCLUSION HERE"
    chunks = [TranscriptChunk(
        id="abc:t0", session_id="abcdef12-0000-0000-0000-000000000000",
        timestamp="2026-03-05T14:23:47Z", turn_index=0,
        user_text="what is the answer",
        assistant_text=long_answer,
    )]
    index = TranscriptIndex(chunks)
    result = index.lookup("answer conclusion", max_chunks=1, max_tokens=2000)

    assert "CONCLUSION HERE" in result  # visible under new 1500-char cap, not old 500


def test_format_results_shows_more_tool_content():
    """Search results show up to 400 chars of tool content, not 200."""
    from synapt.recall.core import TranscriptIndex

    long_tool = "$ pytest\n" + "RESULT_LINE\n" * 30  # well over 200 chars
    chunks = [TranscriptChunk(
        id="abc:t0", session_id="abcdef12-0000-0000-0000-000000000000",
        timestamp="2026-03-05T14:23:47Z", turn_index=0,
        user_text="run tests",
        assistant_text="Running.",
        tool_content=long_tool,
    )]
    index = TranscriptIndex(chunks)
    result = index.lookup("run tests pytest", max_chunks=1, max_tokens=2000)

    # Count how many RESULT_LINEs appear — old cap (200) shows ~17, new cap (400) ~35
    shown_lines = result.count("RESULT_LINE")
    assert shown_lines > 20  # meaningfully more than the old 200-char cap allowed


def test_format_results_shows_more_user_text():
    """Search results show up to 500 chars of user text, not 300."""
    from synapt.recall.core import TranscriptIndex

    long_question = "Q" * 400 + " IMPORTANT DETAIL"
    chunks = [TranscriptChunk(
        id="abc:t0", session_id="abcdef12-0000-0000-0000-000000000000",
        timestamp="2026-03-05T14:23:47Z", turn_index=0,
        user_text=long_question,
        assistant_text="The answer.",
    )]
    index = TranscriptIndex(chunks)
    result = index.lookup("important detail question", max_chunks=1, max_tokens=2000)

    assert "IMPORTANT DETAIL" in result  # visible under new 500-char cap, not old 300


def test_format_results_shows_wider_context_preview():
    """Context preview shows up to 200 chars, not 80."""
    from synapt.recall.core import TranscriptIndex

    prev_question = "A" * 100 + " CONTEXT END"
    chunks = [
        TranscriptChunk(
            id="abc:t0", session_id="abcdef12-0000-0000-0000-000000000000",
            timestamp="2026-03-05T14:00:00Z", turn_index=0,
            user_text=prev_question,
            assistant_text="First answer.",
        ),
        TranscriptChunk(
            id="abc:t1", session_id="abcdef12-0000-0000-0000-000000000000",
            timestamp="2026-03-05T14:05:00Z", turn_index=1,
            user_text="follow up question here",
            assistant_text="Second answer with details.",
        ),
    ]
    index = TranscriptIndex(chunks)
    result = index.lookup("follow up question", max_chunks=1, max_tokens=2000)

    assert "CONTEXT END" in result  # visible in context preview at 112 chars, past old 80


def test_format_results_date_drops_T_and_seconds():
    """Search result headers show 'YYYY-MM-DD HH:MM' not ISO 'YYYY-MM-DDTHH:MM:SS'."""
    from synapt.recall.core import TranscriptIndex

    chunks = [TranscriptChunk(
        id="abc:t0", session_id="abcdef12-0000-0000-0000-000000000000",
        timestamp="2026-03-05T14:23:47Z", turn_index=0,
        user_text="test query content here",
        assistant_text="The answer is here.",
    )]
    index = TranscriptIndex(chunks)
    result = index.lookup("test query content", max_chunks=1, max_tokens=500)

    assert "T14:23:47" not in result   # old ISO format gone
    assert "2026-03-05 14:23" in result  # new readable format present


def test_format_results_journal_chunk_uses_readable_date():
    """Journal chunks also use the readable date format."""
    from synapt.recall.core import TranscriptIndex, TranscriptChunk

    journal_chunk = TranscriptChunk(
        id="abc:journal:deadbeef", session_id="abcdef12-0000-0000-0000-000000000000",
        timestamp="2026-03-04T09:05:30Z", turn_index=-1,
        user_text="Session focus: add live search",
        assistant_text="Done: implemented live.py",
    )
    index = TranscriptIndex([journal_chunk])
    result = index.lookup("live search session focus", max_chunks=1, max_tokens=500)

    assert "T09:05:30" not in result
    assert "2026-03-04 09:05" in result
    assert "journal" in result


def test_format_results_temporal_uses_readable_dates_and_chronological_display():
    """Temporal intent shows fuller dates and orders emitted chunks chronologically."""
    chunks = [
        TranscriptChunk(
            id="newer:t0", session_id="11111111-1111-1111-1111-111111111111",
            timestamp="2026-03-06T16:20:00Z", turn_index=0,
            user_text="when did we finish the deployment",
            assistant_text="We wrapped it up on Friday afternoon.",
        ),
        TranscriptChunk(
            id="older:t0", session_id="22222222-2222-2222-2222-222222222222",
            timestamp="2026-03-04T09:05:00Z", turn_index=0,
            user_text="when did we start the deployment",
            assistant_text="We kicked it off on Wednesday morning.",
        ),
    ]
    index = TranscriptIndex(chunks)

    result = index._format_results(
        ranked=[(0, 1.0), (1, 0.9)],
        max_tokens=2000,
        intent="temporal",
    )

    older_header = "--- [Wednesday, March 4, 2026, 9:05 AM session 22222222] turn 0"
    newer_header = "--- [Friday, March 6, 2026, 4:20 PM session 11111111] turn 0"
    assert older_header in result
    assert newer_header in result
    assert result.index(older_header) < result.index(newer_header)


def test_format_results_deduplicates_near_identical_chunks():
    """Near-duplicate chunks are filtered out to save token budget."""
    from synapt.recall.core import TranscriptIndex

    # Three chunks with nearly identical content — only the first should appear
    chunks = [
        TranscriptChunk(
            id=f"abc:t{i}", session_id="abcdef12-0000-0000-0000-000000000000",
            timestamp=f"2026-03-05T14:0{i}:00Z", turn_index=i,
            user_text="How do I deploy the app?",
            assistant_text=f"To deploy, run deploy.sh then verify. Variant {i}.",
        )
        for i in range(3)
    ]
    index = TranscriptIndex(chunks)
    result = index.lookup("deploy app", max_chunks=10, max_tokens=5000)

    # Only 1 of the 3 near-duplicates should appear
    deploy_count = result.count("To deploy, run deploy.sh")
    assert deploy_count == 1, f"Expected 1 unique chunk, got {deploy_count}"


def test_format_results_keeps_diverse_chunks():
    """Chunks with different content are all kept despite sharing a query match."""
    from synapt.recall.core import TranscriptIndex

    chunks = [
        TranscriptChunk(
            id="abc:t0", session_id="abcdef12-0000-0000-0000-000000000000",
            timestamp="2026-03-05T14:00:00Z", turn_index=0,
            user_text="How do I deploy?",
            assistant_text="Run deploy.sh with the production flag.",
        ),
        TranscriptChunk(
            id="abc:t1", session_id="abcdef12-0000-0000-0000-000000000000",
            timestamp="2026-03-05T14:05:00Z", turn_index=1,
            user_text="What about rollback?",
            assistant_text="Use rollback.sh to revert to the previous version.",
        ),
    ]
    index = TranscriptIndex(chunks)
    result = index.lookup("deploy rollback", max_chunks=10, max_tokens=5000)

    # Both diverse chunks should appear
    assert "deploy.sh" in result
    assert "rollback.sh" in result


def test_format_results_respects_dedup_threshold_override(monkeypatch):
    """Ablation env override should allow stricter or looser dedup."""
    from synapt.recall.core import TranscriptIndex

    chunks = [
        TranscriptChunk(
            id=f"abc:t{i}", session_id="abcdef12-0000-0000-0000-000000000000",
            timestamp=f"2026-03-05T14:0{i}:00Z", turn_index=i,
            user_text="How do I deploy the app?",
            assistant_text=f"To deploy, run deploy.sh then verify. Variant {i}.",
        )
        for i in range(3)
    ]

    monkeypatch.setenv("SYNAPT_DEDUP_JACCARD", "1.1")
    try:
        index = TranscriptIndex(chunks)
        result = index.lookup("deploy app", max_chunks=10, max_tokens=5000)
    finally:
        monkeypatch.delenv("SYNAPT_DEDUP_JACCARD", raising=False)

    deploy_count = result.count("To deploy, run deploy.sh")
    assert deploy_count == 3, f"Expected override to keep all 3 chunks, got {deploy_count}"


# ---------------------------------------------------------------------------
# Tests: _find_query_span (query-aware snippet extraction)
# ---------------------------------------------------------------------------

def test_find_query_span_basic():
    """_find_query_span finds the sentence matching the query."""
    text = (
        "The weather was nice today. "
        "We deployed the application to production using Docker containers. "
        "Then we had lunch."
    )
    span = TranscriptIndex._find_query_span("deploy application Docker", text)
    assert span is not None
    begin, end = span
    snippet = text[begin:end]
    assert "deployed the application" in snippet
    # Should NOT include the unrelated sentences in the tight span
    # (margin may include a few chars of context, but core match is deploy)


def test_find_query_span_no_match():
    """Returns None when query has no overlap with text."""
    text = "The weather was nice today. We had lunch."
    span = TranscriptIndex._find_query_span("kubernetes orchestration", text)
    assert span is None


def test_find_query_span_empty_inputs():
    """Returns None for empty query or text."""
    assert TranscriptIndex._find_query_span("", "some text") is None
    assert TranscriptIndex._find_query_span("query", "") is None


def test_find_query_span_short_text():
    """Returns None when text is a single short sentence with < 2 overlapping tokens."""
    text = "Hello world."
    span = TranscriptIndex._find_query_span("goodbye universe", text)
    assert span is None


def test_find_query_span_multi_sentence_window():
    """Finds a contiguous multi-sentence window when query spans sentences."""
    text = (
        "First we set up the database schema. "
        "Then we ran the migration scripts. "
        "The migration completed successfully with zero errors. "
        "Finally we cleaned up temp files."
    )
    span = TranscriptIndex._find_query_span("migration scripts database schema", text)
    assert span is not None
    begin, end = span
    snippet = text[begin:end]
    # Should cover the migration/database region
    assert "migration" in snippet


def test_format_chunk_block_with_query_snippets(monkeypatch):
    """_format_chunk_block emits focused snippet when SYNAPT_SNIPPET_CONTEXT=1."""
    monkeypatch.setenv("SYNAPT_SNIPPET_CONTEXT", "1")
    chunks = [
        TranscriptChunk(
            id="s001:t0",
            session_id="session-001",
            timestamp="2026-03-24T12:00:00Z",
            turn_index=0,
            user_text="Can you help me set up the deployment pipeline?",
            assistant_text=(
                "Sure! First, you need to configure your CI/CD pipeline. "
                "Create a Dockerfile in the project root. "
                "Then add a GitHub Actions workflow file. "
                "The workflow should build the image, run tests, "
                "and push to your container registry. "
                "After that, set up ArgoCD for continuous deployment. "
                "ArgoCD will watch your registry and auto-deploy new images. "
                "Make sure to configure health checks and rollback policies. "
                "You should also set up monitoring with Prometheus and Grafana. "
                "Finally, add alerting rules for critical metrics."
            ),
            tools_used=[],
            files_touched=[],
            tool_content="",
            date_text="",
            transcript_path="",
            byte_offset=0,
            byte_length=0,
        ),
    ]
    index = TranscriptIndex(chunks)
    # With query — should get focused snippet
    block_with_query = index._format_chunk_block(0, query="ArgoCD continuous deployment")
    # Without query — should get full turn
    block_without_query = index._format_chunk_block(0)
    # The snippet version should be shorter (focused on ArgoCD section)
    assert len(block_with_query) < len(block_without_query)
    assert "ArgoCD" in block_with_query
    # Full version should have the full content
    assert "User:" in block_without_query
    assert "Assistant:" in block_without_query


@pytest.mark.xfail(strict=True, reason="pre-existing: snippet extraction clips 'adoption' from context window")
def test_format_chunk_block_prefers_assistant_over_user(monkeypatch):
    """Snippet extraction prefers assistant text over user question.

    Regression test for Atlas's review: concatenated user+assistant scoring
    would anchor on the user question and clip the actual answer evidence.
    """
    monkeypatch.setenv("SYNAPT_SNIPPET_CONTEXT", "1")
    chunks = [
        TranscriptChunk(
            id="s001:t0",
            session_id="session-001",
            timestamp="2026-03-24T00:00:00Z",
            turn_index=0,
            user_text="What did Caroline say about adoption at therapy last week?",
            assistant_text=(
                "She paused for a while before answering and talked through "
                "several unrelated details about the week. "
                "She mentioned that the commute to the office had been terrible "
                "because of the construction on Highway 101. "
                "She also talked about a new recipe she tried for dinner. "
                "The weather had been unusually cold for this time of year. "
                "She described a movie she watched over the weekend in detail. "
                "Later she explained that counseling had made the decision feel "
                "less overwhelming and that she now had a concrete plan to "
                "pursue adoption. "
                "She also said the therapy session reduced her anxiety about "
                "the whole process significantly. "
                "After that she shifted topics to discuss her garden project. "
                "She planted tomatoes and basil in the raised beds last Sunday. "
                "The neighborhood association meeting was also brought up. "
                "She mentioned they discussed parking regulations at length. "
                "Finally she said she was looking forward to the weekend trip."
            ),
            tools_used=[],
            files_touched=[],
            tool_content="",
            date_text="",
            transcript_path="",
            byte_offset=0,
            byte_length=0,
        ),
    ]
    index = TranscriptIndex(chunks)
    block = index._format_chunk_block(0, query="Caroline adoption therapy last week")
    # Must contain the actual answer evidence from assistant, not just the question
    assert "adoption" in block
    assert "concrete plan" in block or "counseling" in block
    # Should be a snippet (shorter than full turn)
    full_block = index._format_chunk_block(0)
    assert len(block) < len(full_block)


def test_format_chunk_block_query_no_match_falls_back():
    """When query doesn't match well, falls back to full turn format."""
    chunks = [
        TranscriptChunk(
            id="s001:t0",
            session_id="session-001",
            timestamp="2026-03-24T12:00:00Z",
            turn_index=0,
            user_text="Hello",
            assistant_text="Hi there! How can I help you today?",
            tools_used=[],
            files_touched=[],
            tool_content="",
            date_text="",
            transcript_path="",
            byte_offset=0,
            byte_length=0,
        ),
    ]
    index = TranscriptIndex(chunks)
    # Short text — won't snippet (< 300 chars)
    block = index._format_chunk_block(0, query="kubernetes orchestration")
    assert "User: Hello" in block
    assert "Assistant: Hi there" in block


def test_snippet_disabled_by_default():
    """Without SYNAPT_SNIPPET_CONTEXT, snippeting is off (full chunk)."""
    chunks = [
        TranscriptChunk(
            id="s001:t0",
            session_id="session-001",
            timestamp="2026-03-24T12:00:00Z",
            turn_index=0,
            user_text="Tell me about Docker",
            assistant_text=(
                "Docker is a containerization platform. "
                "It lets you package applications into containers. "
                "Containers are lightweight and portable. "
                "You can deploy them anywhere. "
                "Docker uses images as templates. "
                "Images are built from Dockerfiles. "
                "The Docker Hub hosts public images. "
                "Kubernetes can orchestrate Docker containers at scale."
            ),
            tools_used=[],
            files_touched=[],
            tool_content="",
            date_text="",
            transcript_path="",
            byte_offset=0,
            byte_length=0,
        ),
    ]
    index = TranscriptIndex(chunks)
    block = index._format_chunk_block(0, query="Kubernetes orchestrate containers")
    # Full block should include all content, not just the Kubernetes sentence
    assert "Docker is a containerization platform" in block
    assert "Kubernetes" in block


def test_snippet_wider_margin(monkeypatch):
    """SYNAPT_SNIPPET_CONTEXT=200 gives wider context around match."""
    monkeypatch.setenv("SYNAPT_SNIPPET_CONTEXT", "200")
    chunks = [
        TranscriptChunk(
            id="s001:t0",
            session_id="session-001",
            timestamp="2026-03-24T12:00:00Z",
            turn_index=0,
            user_text="Tell me about deployment",
            assistant_text=(
                "A" * 100 + ". "
                "We deployed using ArgoCD for continuous delivery. "
                "B" * 100 + ". "
                "C" * 100 + "."
            ),
            tools_used=[],
            files_touched=[],
            tool_content="",
            date_text="",
            transcript_path="",
            byte_offset=0,
            byte_length=0,
        ),
    ]
    index = TranscriptIndex(chunks)
    block = index._format_chunk_block(0, query="ArgoCD continuous delivery deployed")
    assert "ArgoCD" in block
    # With 200-char margin, should include surrounding context
    assert "..." in block  # Still a snippet, not full text


def test_snippet_tight_context_zero(monkeypatch):
    """SYNAPT_SNIPPET_CONTEXT=0 enables tight snippets (40-char margin)."""
    monkeypatch.setenv("SYNAPT_SNIPPET_CONTEXT", "0")
    chunks = [
        TranscriptChunk(
            id="s001:t0",
            session_id="session-001",
            timestamp="2026-03-24T12:00:00Z",
            turn_index=0,
            user_text="Tell me about Docker",
            assistant_text=(
                "Docker is a containerization platform. "
                "It lets you package applications into containers. "
                "Containers are lightweight and portable. "
                "You can deploy them anywhere. "
                "Docker uses images as templates. "
                "Images are built from Dockerfiles. "
                "The Docker Hub hosts public images. "
                "Kubernetes can orchestrate Docker containers at scale."
            ),
            tools_used=[],
            files_touched=[],
            tool_content="",
            date_text="",
            transcript_path="",
            byte_offset=0,
            byte_length=0,
        ),
    ]
    index = TranscriptIndex(chunks)
    block = index._format_chunk_block(0, query="Kubernetes orchestrate containers")
    # Tight snippet — should be shorter than full block
    full_block = index._format_chunk_block(0)
    assert len(block) < len(full_block)
    assert "Kubernetes" in block


def test_snippet_invalid_env_value_disables(monkeypatch):
    """Invalid SYNAPT_SNIPPET_CONTEXT value falls back to no snippeting."""
    monkeypatch.setenv("SYNAPT_SNIPPET_CONTEXT", "not_a_number")
    chunks = [
        TranscriptChunk(
            id="s001:t0",
            session_id="session-001",
            timestamp="2026-03-24T12:00:00Z",
            turn_index=0,
            user_text="Hello",
            assistant_text=(
                "Docker is a containerization platform. "
                "Kubernetes can orchestrate Docker containers at scale. "
                "Both are essential for modern deployments. "
                "CI/CD pipelines automate the build and deploy process."
            ),
            tools_used=[],
            files_touched=[],
            tool_content="",
            date_text="",
            transcript_path="",
            byte_offset=0,
            byte_length=0,
        ),
    ]
    index = TranscriptIndex(chunks)
    block = index._format_chunk_block(0, query="Kubernetes orchestrate")
    # Invalid value → no snippeting → full block with User: and Assistant:
    assert "User: Hello" in block
    assert "Docker is a containerization" in block


def test_snippet_legacy_enable_snippets_compat(monkeypatch):
    """SYNAPT_ENABLE_SNIPPETS=1 still works for backward compat."""
    monkeypatch.setenv("SYNAPT_ENABLE_SNIPPETS", "1")
    monkeypatch.delenv("SYNAPT_SNIPPET_CONTEXT", raising=False)
    chunks = [
        TranscriptChunk(
            id="s001:t0",
            session_id="session-001",
            timestamp="2026-03-24T12:00:00Z",
            turn_index=0,
            user_text="Tell me about Docker",
            assistant_text=(
                "Docker is a containerization platform. "
                "It lets you package applications into containers. "
                "Containers are lightweight and portable. "
                "You can deploy them anywhere. "
                "Docker uses images as templates. "
                "Images are built from Dockerfiles. "
                "The Docker Hub hosts public images. "
                "Kubernetes can orchestrate Docker containers at scale."
            ),
            tools_used=[],
            files_touched=[],
            tool_content="",
            date_text="",
            transcript_path="",
            byte_offset=0,
            byte_length=0,
        ),
    ]
    index = TranscriptIndex(chunks)
    block = index._format_chunk_block(0, query="Kubernetes orchestrate containers")
    full_block = index._format_chunk_block(0)
    # Legacy env var should enable snippets (tight mode)
    assert len(block) < len(full_block)
