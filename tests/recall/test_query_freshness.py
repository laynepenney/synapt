"""Contract tests for bounded query-time current-session freshness."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from synapt.recall.core import TranscriptChunk
from synapt.recall.query_freshness import (
    QueryFreshnessPolicy,
    QueryFreshnessState,
    format_query_freshness,
    refresh_current_session,
)
from synapt.recall.resume import CallerTranscript
from synapt.recall.resume import BoundedResumeIndex
from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.storage import RecallDB


SESSION = "aaaaaaaa-0000-0000-0000-000000000000"


def _line(kind: str, text: str, timestamp: str, uuid: str) -> str:
    if kind == "user":
        payload = {
            "type": "user",
            "message": {"role": "user", "content": text},
            "sessionId": SESSION,
            "uuid": uuid,
            "timestamp": timestamp,
        }
    else:
        payload = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
            "sessionId": SESSION,
            "uuid": uuid,
            "timestamp": timestamp,
        }
    return json.dumps(payload) + "\n"


def _write_turn(path: Path, user: str, assistant: str, minute: int) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            _line("user", user, f"2026-08-30T10:{minute:02d}:00Z", f"u-{minute}")
        )
        stream.write(
            _line(
                "assistant",
                assistant,
                f"2026-08-30T10:{minute:02d}:30Z",
                f"a-{minute}",
            )
        )


def _codex_entry(timestamp: str, role: str, text: str) -> str:
    kind = "input_text" if role == "user" else "output_text"
    return (
        json.dumps(
            {
                "timestamp": timestamp,
                "type": "response_item",
                "payload": {
                    "role": role,
                    "content": [{"type": kind, "text": text}],
                },
            }
        )
        + "\n"
    )


def _source(path: Path, latest: str = "2026-08-30T10:20:30Z") -> CallerTranscript:
    stat = path.stat()
    return CallerTranscript(SESSION, path, stat.st_mtime, stat.st_size, latest)


def _policy(**kwargs) -> QueryFreshnessPolicy:
    values = {
        "age_threshold_seconds": 0,
        "byte_trigger": 1,
        "step_bytes": 1024 * 1024,
        "byte_cap": 32 * 1024 * 1024,
        "wall_seconds": 5,
    }
    values.update(kwargs)
    return QueryFreshnessPolicy(**values)


def test_refresh_uses_the_explicit_caller_root_without_a_store_fallback(
    tmp_path, monkeypatch
):
    caller_root = tmp_path / "caller"
    caller_root.mkdir()
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    roots: list[Path] = []

    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: roots.append(root) or [],
    )

    result = refresh_current_session(index_dir, caller_root, policy=_policy())

    assert roots == [caller_root]
    assert result.state is QueryFreshnessState.NOT_BOUND
    assert result.reason == "no_caller_transcript"


def test_stale_tail_commits_before_the_searchable_read(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "query freshness witness", "indexed now", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    result = refresh_current_session(index_dir, tmp_path, policy=_policy())

    assert result.state is QueryFreshnessState.REFRESHED
    assert result.indexed_now_bytes == transcript.stat().st_size
    assert result.remaining_bytes == 0
    db = ShardedRecallDB.open(index_dir)
    try:
        chunks = db.load_chunks()
        assert any("query freshness witness" in chunk.user_text for chunk in chunks)
    finally:
        db.close()
    from synapt.recall.core import TranscriptIndex

    index = TranscriptIndex.load(index_dir, use_embeddings=False)
    try:
        rendered = index.lookup(
            "query freshness witness",
            max_chunks=5,
            max_tokens=500,
            threshold_ratio=0,
            half_life=0,
        )
    finally:
        index._db.close()
    assert "query freshness witness" in rendered


def test_below_threshold_gap_is_labelled_without_a_write(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "indexed overlay", "overlay answer", 0)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript, "2026-08-30T10:05:30Z")],
    )
    first = refresh_current_session(index_dir, tmp_path, policy=_policy())
    _write_turn(transcript, "recent gap", "not far enough ahead", 5)
    policy = _policy(age_threshold_seconds=600, byte_trigger=10**9)

    result = refresh_current_session(
        index_dir,
        tmp_path,
        policy=policy,
        now=datetime(2026, 8, 30, 10, 5, tzinfo=timezone.utc),
    )

    assert first.state is QueryFreshnessState.REFRESHED
    assert result.state is QueryFreshnessState.RECENT_GAP
    assert result.indexed_now_bytes == 0
    assert 0 < result.remaining_bytes < transcript.stat().st_size


def test_age_trigger_measures_ahead_of_index_not_age_of_live_source(
    tmp_path, monkeypatch
):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "old indexed overlay", "base answer", 0)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript, "2026-08-30T10:20:30Z")],
    )
    first = refresh_current_session(index_dir, tmp_path, policy=_policy())
    _write_turn(transcript, "small but stale gap", "must refresh", 20)

    result = refresh_current_session(
        index_dir,
        tmp_path,
        policy=_policy(age_threshold_seconds=600, byte_trigger=10**9),
        now=datetime(2026, 8, 30, 10, 20, 31, tzinfo=timezone.utc),
    )

    assert first.state is QueryFreshnessState.REFRESHED
    assert result.state is QueryFreshnessState.REFRESHED
    assert result.indexed_now_bytes > 0


def test_projected_timestamp_orders_mixed_offsets_by_instant(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    with transcript.open("w", encoding="utf-8") as stream:
        stream.write(_line("user", "later instant", "2026-08-30T23:00:00-05:00", "u-1"))
        stream.write(
            _line("assistant", "later answer", "2026-08-30T23:01:00-05:00", "a-1")
        )
        stream.write(
            _line("user", "earlier instant", "2026-08-31T01:00:00+14:00", "u-2")
        )
        stream.write(
            _line("assistant", "earlier answer", "2026-08-31T01:01:00+14:00", "a-2")
        )
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    result = refresh_current_session(index_dir, tmp_path, policy=_policy())
    db = RecallDB(index_dir / "recall.db")
    try:
        cursor = db.load_query_tail_cursor(result.source_key)
    finally:
        db.close()

    assert cursor is not None
    assert cursor["latest_projected_timestamp"] == "2026-08-30T23:00:00-05:00"


def test_byte_trigger_refreshes_even_when_temporal_gap_is_recent(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "indexed overlay", "base answer", 0)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript, "2026-08-30T10:01:30Z")],
    )
    first = refresh_current_session(index_dir, tmp_path, policy=_policy())
    _write_turn(transcript, "large recent gap " + "x" * 600, "refresh", 1)

    result = refresh_current_session(
        index_dir,
        tmp_path,
        policy=_policy(age_threshold_seconds=600, byte_trigger=100),
    )

    assert first.state is QueryFreshnessState.REFRESHED
    assert result.state is QueryFreshnessState.REFRESHED
    assert result.indexed_now_bytes > 100


def test_growing_open_turn_replaces_the_earlier_overlay_row(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    with transcript.open("w", encoding="utf-8") as stream:
        stream.write(_line("user", "open turn", "2026-08-30T10:20:00Z", "u-open"))
        stream.write(
            _line("assistant", "first answer", "2026-08-30T10:20:10Z", "a-one")
        )
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    first = refresh_current_session(index_dir, tmp_path, policy=_policy())
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(
            _line("assistant", "later answer", "2026-08-30T10:21:00Z", "a-two")
        )
    second = refresh_current_session(index_dir, tmp_path, policy=_policy())

    db = ShardedRecallDB.open(index_dir)
    try:
        chunks = [chunk for chunk in db.load_chunks() if chunk.session_id == SESSION]
    finally:
        db.close()
    assert first.state is QueryFreshnessState.REFRESHED
    assert second.state is QueryFreshnessState.REFRESHED
    assert len(chunks) == 1
    assert "first answer" in chunks[0].assistant_text
    assert "later answer" in chunks[0].assistant_text


def test_incomplete_final_record_does_not_advance_the_cursor(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "complete", "complete", 20)
    complete_size = transcript.stat().st_size
    with transcript.open("ab") as stream:
        stream.write(b'{"type":"user","message":')
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    result = refresh_current_session(index_dir, tmp_path, policy=_policy())
    db = RecallDB(index_dir / "recall.db")
    try:
        cursor = db.load_query_tail_cursor(result.source_key)
    finally:
        db.close()

    assert cursor is not None
    assert cursor["observed_complete_offset"] == complete_size
    assert result.remaining_bytes == transcript.stat().st_size - complete_size
    assert result.state is QueryFreshnessState.PARTIAL


def test_complete_record_larger_than_step_advances_and_becomes_current(
    tmp_path, monkeypatch
):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "large " + "x" * 1000, "complete", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    first = refresh_current_session(
        index_dir,
        tmp_path,
        policy=_policy(step_bytes=128, byte_cap=4096),
    )
    second = refresh_current_session(
        index_dir,
        tmp_path,
        policy=_policy(step_bytes=128, byte_cap=4096),
    )

    assert transcript.stat().st_size > 128
    assert first.state is QueryFreshnessState.REFRESHED
    assert first.observed_complete_offset == transcript.stat().st_size
    assert first.indexed_now_bytes == transcript.stat().st_size
    assert second.state is QueryFreshnessState.CURRENT
    assert second.indexed_now_bytes == 0


def test_empty_replacement_suppresses_stale_base_but_empty_without_base_is_clean(
    tmp_path, monkeypatch
):
    from synapt.recall.core import parse_transcript

    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "stale base", "must disappear", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    db = RecallDB(index_dir / "recall.db")
    db.save_chunks(parse_transcript(transcript))
    db.close()
    transcript.write_bytes(b"")
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    replaced = refresh_current_session(index_dir, tmp_path, policy=_policy())
    db = ShardedRecallDB.open(index_dir)
    try:
        replaced_cursor = db.load_query_tail_cursor(replaced.source_key)
        replaced_chunks = db.load_chunks()
    finally:
        db.close()

    clean_index = tmp_path / "clean-index"
    clean_index.mkdir()
    clean = refresh_current_session(clean_index, tmp_path, policy=_policy())
    db = ShardedRecallDB.open(clean_index)
    try:
        clean_cursor = db.load_query_tail_cursor(clean.source_key)
    finally:
        db.close()

    assert replaced.state is QueryFreshnessState.CURRENT
    assert replaced.index_changed is True
    assert replaced_cursor is not None
    assert replaced_cursor["suppresses_base"] == 1
    assert replaced_chunks == []
    assert clean.state is QueryFreshnessState.CURRENT
    assert clean.index_changed is False
    assert clean_cursor is None


def test_incomplete_first_record_keeps_prior_base_visible_and_reports_partial(
    tmp_path, monkeypatch
):
    from synapt.recall.core import parse_transcript

    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "prior base", "still visible", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    db = RecallDB(index_dir / "recall.db")
    db.save_chunks(parse_transcript(transcript))
    db.close()
    transcript.write_bytes(b'{"type":"user","message":')
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    result = refresh_current_session(index_dir, tmp_path, policy=_policy())
    db = ShardedRecallDB.open(index_dir)
    try:
        chunks = db.load_chunks()
        cursor = db.load_query_tail_cursor(result.source_key)
    finally:
        db.close()

    assert result.state is QueryFreshnessState.PARTIAL
    assert result.reason == "incomplete_record"
    assert result.observed_complete_offset == 0
    assert result.cut_short is True
    assert [chunk.user_text for chunk in chunks] == ["prior base"]
    assert cursor is None


def test_malformed_complete_record_does_not_advance_the_cursor(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "valid before malformed", "valid", 20)
    with transcript.open("ab") as stream:
        stream.write(b'{"type":"user",bad-json}\n')
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    result = refresh_current_session(index_dir, tmp_path, policy=_policy())

    db = RecallDB(index_dir / "recall.db")
    try:
        assert db.load_query_tail_chunks() == []
        assert db.load_query_tail_cursor(result.source_key) is None
    finally:
        db.close()
    assert result.state is QueryFreshnessState.ERROR
    assert result.reason == "ValueError:malformed_complete_record"
    assert result.observed_complete_offset == 0
    assert result.remaining_bytes == transcript.stat().st_size
    assert result.cut_short is True
    assert "indexed_through=0" in format_query_freshness(result)
    assert f"remaining={transcript.stat().st_size}B" in format_query_freshness(result)


def test_error_after_a_durable_step_reports_that_coverage(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "durable before error", "kept", 20)
    durable_size = transcript.stat().st_size
    with transcript.open("ab") as stream:
        stream.write(b'{"type":"user",bad-json}\n')
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    result = refresh_current_session(
        index_dir,
        tmp_path,
        policy=_policy(step_bytes=durable_size),
    )

    assert result.state is QueryFreshnessState.ERROR
    assert result.observed_complete_offset == durable_size
    assert result.indexed_now_bytes == durable_size
    assert result.remaining_bytes == transcript.stat().st_size - durable_size
    assert result.cut_short is True


def test_held_build_lock_returns_busy_without_parsing(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "busy witness", "must not write", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    timeouts: list[float] = []

    def held_lock(data_dir, timeout=0):
        timeouts.append(timeout)
        return None

    monkeypatch.setattr("synapt.recall.query_freshness._acquire_build_lock", held_lock)

    result = refresh_current_session(index_dir, tmp_path, policy=_policy())

    assert result.state is QueryFreshnessState.BUSY
    assert result.indexed_now_bytes == 0
    assert result.observed_complete_offset is None
    assert result.remaining_bytes is None
    assert "indexed_through=unknown" in format_query_freshness(result)
    assert "remaining=unknown" in format_query_freshness(result)
    assert result.cut_short is True
    assert not (index_dir / "recall.db").exists()
    assert timeouts == [0]


def test_busy_with_existing_coverage_marks_that_coverage_unknown(
    tmp_path, monkeypatch
):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "existing coverage", "durable", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    first = refresh_current_session(index_dir, tmp_path, policy=_policy())
    monkeypatch.setattr(
        "synapt.recall.query_freshness._acquire_build_lock",
        lambda data_dir, timeout=0: None,
    )

    busy = refresh_current_session(index_dir, tmp_path, policy=_policy())

    assert first.observed_complete_offset == transcript.stat().st_size
    assert busy.state is QueryFreshnessState.BUSY
    assert busy.observed_complete_offset is None
    assert busy.remaining_bytes is None
    assert busy.cut_short is True


def test_byte_cap_is_truthful_and_retryable(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    for minute in range(20, 24):
        _write_turn(transcript, f"turn {minute} " + "x" * 400, "answer", minute)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    cap = transcript.stat().st_size // 2

    first = refresh_current_session(
        index_dir,
        tmp_path,
        policy=_policy(step_bytes=cap, byte_cap=cap),
    )
    second = refresh_current_session(index_dir, tmp_path, policy=_policy())

    assert first.state is QueryFreshnessState.PARTIAL
    assert first.cut_short is True
    assert first.remaining_bytes > 0
    assert second.state is QueryFreshnessState.REFRESHED
    assert second.remaining_bytes == 0


def test_freshness_formatter_names_scope_and_work(tmp_path):
    from synapt.recall.query_freshness import QueryFreshnessResult

    result = QueryFreshnessResult(
        state=QueryFreshnessState.PARTIAL,
        session_id=SESSION,
        source_path=tmp_path / "session.jsonl",
        source_key="source-key",
        observed_complete_offset=200,
        live_bytes=260,
        indexed_now_bytes=100,
        indexed_now_chunks=3,
        remaining_bytes=60,
        wall_seconds=0.25,
        cut_short=True,
        reason="byte_cap",
    )

    line = format_query_freshness(result)

    assert line.startswith("Freshness: PARTIAL")
    assert SESSION[:8] in line
    assert "indexed_now=100B/3chunks" in line
    assert "remaining=60B" in line
    assert "cut_short=true" in line


def test_sharded_layout_reads_overlay_without_rewriting_base(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index_db = RecallDB(index_dir / "index.db")
    index_db.close()
    shard = RecallDB(index_dir / "data_0001.db")
    base = TranscriptChunk(
        id="base:t0",
        session_id="base",
        timestamp="2026-08-29T00:00:00Z",
        turn_index=0,
        user_text="base corpus marker",
        assistant_text="",
    )
    shard.save_chunks([base])
    shard.close()
    shard_before = (index_dir / "data_0001.db").read_bytes()

    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "overlay corpus marker", "searchable", 20)
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    refresh_current_session(index_dir, tmp_path, policy=_policy())

    db = ShardedRecallDB.open(index_dir)
    try:
        texts = [chunk.user_text for chunk in db.load_chunks()]
    finally:
        db.close()
    assert "base corpus marker" in texts
    assert "overlay corpus marker" in texts
    assert (index_dir / "data_0001.db").read_bytes() == shard_before


def test_unproven_base_extent_reparses_from_zero_and_suppresses_base(
    tmp_path, monkeypatch
):
    from synapt.recall.core import parse_transcript

    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "already in base", "base answer", 20)
    base_size = transcript.stat().st_size
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    base_chunks = parse_transcript(transcript)
    db = RecallDB(index_dir / "recall.db")
    db.save_chunks(base_chunks)
    db.close()
    _write_turn(transcript, "new query tail", "tail answer", 21)
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    result = refresh_current_session(index_dir, tmp_path, policy=_policy())

    assert result.state is QueryFreshnessState.REFRESHED
    assert result.indexed_now_bytes == transcript.stat().st_size
    db = ShardedRecallDB.open(index_dir)
    try:
        texts = [chunk.user_text for chunk in db.load_chunks()]
    finally:
        db.close()
    assert texts == ["already in base", "new query tail"]
    assert base_size < result.indexed_now_bytes


def test_partial_reparse_keeps_prior_base_visible_until_atomic_completion(
    tmp_path, monkeypatch
):
    from synapt.recall.core import parse_transcript

    transcript = tmp_path / f"{SESSION}.jsonl"
    for minute in range(20, 24):
        _write_turn(transcript, f"base turn {minute} " + "x" * 200, "answer", minute)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    db = RecallDB(index_dir / "recall.db")
    db.save_chunks(parse_transcript(transcript))
    db.close()
    transcript_bytes = transcript.read_bytes()
    first_newline = transcript_bytes.find(b"\n")
    first_turn_end = transcript_bytes.find(b"\n", first_newline + 1) + 1
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    initial = refresh_current_session(index_dir, tmp_path, policy=_policy())
    db = ShardedRecallDB.open(index_dir)
    try:
        db.clear_query_tail(initial.source_key)
        assert db.load_query_tail_cursor(initial.source_key) is None
    finally:
        db.close()

    partial = refresh_current_session(
        index_dir,
        tmp_path,
        policy=_policy(step_bytes=first_turn_end, byte_cap=first_turn_end),
    )
    db = ShardedRecallDB.open(index_dir)
    try:
        partial_texts = [chunk.user_text for chunk in db.load_chunks()]
        partial_cursor = db.load_query_tail_cursor(partial.source_key)
    finally:
        db.close()

    complete = refresh_current_session(index_dir, tmp_path, policy=_policy())
    db = ShardedRecallDB.open(index_dir)
    try:
        complete_texts = [chunk.user_text for chunk in db.load_chunks()]
        complete_cursor = db.load_query_tail_cursor(complete.source_key)
    finally:
        db.close()

    assert initial.state is QueryFreshnessState.REFRESHED
    assert partial.state is QueryFreshnessState.PARTIAL
    assert partial_cursor is not None
    assert partial_cursor["suppresses_base"] == 0
    assert len(partial_texts) == 4
    assert complete.state is QueryFreshnessState.REFRESHED
    assert complete_cursor is not None
    assert complete_cursor["suppresses_base"] == 1
    assert sorted(complete_texts) == sorted(partial_texts)


def test_codex_tail_reparses_an_unproven_base_under_the_overlay_contract(
    tmp_path, monkeypatch
):
    from synapt.recall.codex import parse_codex_transcript

    transcript = tmp_path / "rollout-query-freshness.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-30T10:20:00Z",
                "type": "session_meta",
                "payload": {"id": SESSION},
            }
        )
        + "\n"
        + _codex_entry("2026-08-30T10:20:01Z", "user", "codex base turn")
        + _codex_entry("2026-08-30T10:20:02Z", "assistant", "base answer")
    )
    base_size = transcript.stat().st_size
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    db = RecallDB(index_dir / "recall.db")
    db.save_chunks(parse_codex_transcript(transcript))
    db.close()
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(_codex_entry("2026-08-30T10:21:01Z", "user", "codex tail turn"))
        stream.write(_codex_entry("2026-08-30T10:21:02Z", "assistant", "tail answer"))
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    result = refresh_current_session(index_dir, tmp_path, policy=_policy())

    db = ShardedRecallDB.open(index_dir)
    try:
        chunks = db.load_chunks()
    finally:
        db.close()
    assert result.state is QueryFreshnessState.REFRESHED
    assert result.indexed_now_bytes == transcript.stat().st_size
    assert [chunk.user_text for chunk in chunks] == [
        "codex base turn",
        "codex tail turn",
    ]
    assert base_size < result.indexed_now_bytes


def test_same_session_in_a_different_path_never_seeds_from_the_old_base(
    tmp_path, monkeypatch
):
    from synapt.recall.core import parse_transcript

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old_path = old_dir / f"{SESSION}.jsonl"
    new_path = new_dir / f"{SESSION}.jsonl"
    _write_turn(old_path, "old directory truth", "old", 20)
    _write_turn(new_path, "new directory truth", "new", 20)
    assert old_path.stat().st_size == new_path.stat().st_size
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    db = RecallDB(index_dir / "recall.db")
    db.save_chunks(parse_transcript(old_path))
    db.close()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(new_path)],
    )

    result = refresh_current_session(index_dir, tmp_path, policy=_policy())

    db = ShardedRecallDB.open(index_dir)
    try:
        texts = [chunk.user_text for chunk in db.load_chunks()]
    finally:
        db.close()
    assert result.state is QueryFreshnessState.REFRESHED
    assert texts == ["new directory truth"]


def test_same_path_same_length_replacement_invalidates_the_cursor(
    tmp_path, monkeypatch
):
    import synapt.recall.query_freshness as freshness

    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "first generation", "old", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    first = refresh_current_session(index_dir, tmp_path, policy=_policy())
    old_size = transcript.stat().st_size
    old_mtime_ns = transcript.stat().st_mtime_ns
    old_text = transcript.read_text()
    transcript.write_text(old_text.replace("first generation", "other generation"))
    replaced_stat = transcript.stat()
    os.utime(
        transcript,
        ns=(replaced_stat.st_atime_ns, old_mtime_ns + 1),
    )
    assert transcript.stat().st_size == old_size
    assert transcript.stat().st_mtime_ns != old_mtime_ns

    digest_calls = 0
    real_prefix_digest = freshness._prefix_digest

    def counted_prefix_digest(path, end):  # noqa: ANN001, ANN202
        nonlocal digest_calls
        digest_calls += 1
        return real_prefix_digest(path, end)

    monkeypatch.setattr(freshness, "_prefix_digest", counted_prefix_digest)

    second = refresh_current_session(index_dir, tmp_path, policy=_policy())

    db = ShardedRecallDB.open(index_dir)
    try:
        texts = [chunk.user_text for chunk in db.load_chunks()]
    finally:
        db.close()
    assert first.state is QueryFreshnessState.REFRESHED
    assert second.state is QueryFreshnessState.REFRESHED
    assert digest_calls == 1
    assert texts == ["other generation"]


def test_unchanged_current_cursor_does_not_rehash_the_observed_prefix(
    tmp_path, monkeypatch
):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "already current", "answer", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    first = refresh_current_session(index_dir, tmp_path, policy=_policy())
    monkeypatch.setattr(
        "synapt.recall.query_freshness._prefix_digest",
        lambda path, end: (_ for _ in ()).throw(AssertionError("rehash")),
    )

    second = refresh_current_session(index_dir, tmp_path, policy=_policy())

    assert first.state is QueryFreshnessState.REFRESHED
    assert second.state is QueryFreshnessState.CURRENT


def test_chunk_count_uses_composed_sql_semantics_without_loading_corpus(tmp_path):
    def chunk(chunk_id: str, session_id: str, turn_index: int) -> TranscriptChunk:
        return TranscriptChunk(
            id=chunk_id,
            session_id=session_id,
            timestamp="",
            turn_index=turn_index,
            user_text=chunk_id,
            assistant_text="",
        )

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    db = RecallDB(index_dir / "recall.db")
    db.save_chunks([
        chunk("old:t0", "old", 0),
        chunk("old:t1", "old", 1),
        chunk("live:t0", "live", 0),
        chunk("live:t1", "live", 1),
    ])
    cursor = {
        "transcript_path": str(tmp_path / "source.jsonl"),
        "observed_complete_offset": 1,
        "rewind_offset": 0,
        "rewind_turn_index": 0,
        "source_size": 1,
        "source_mtime_ns": 1,
        "observed_prefix_sha256": "digest",
        "latest_projected_timestamp": "",
        "last_attempt_at": "2026-08-30T00:00:00+00:00",
        "last_success_at": "2026-08-30T00:00:00+00:00",
    }
    db.replace_query_tail(
        source_key="old-source",
        session_id="old",
        rewind_offset=0,
        chunks=[chunk("old:new", "old", 0)],
        cursor={**cursor, "suppresses_base": True},
    )
    db.replace_query_tail(
        source_key="live-source",
        session_id="live",
        rewind_offset=0,
        chunks=[chunk("live:t1", "live", 1)],
        cursor={**cursor, "suppresses_base": False},
    )
    db.close()

    sharded = ShardedRecallDB.open(index_dir)
    try:
        expected = len(sharded.load_chunks())
        sharded.load_chunks = lambda: (_ for _ in ()).throw(AssertionError("loaded"))
        measured = sharded.chunk_count()
    finally:
        sharded.close()

    assert expected == 3
    assert measured == expected


def test_source_shrink_suppresses_stale_base_rows_until_rebuilt(tmp_path, monkeypatch):
    from synapt.recall.core import parse_transcript

    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "replacement survives", "new truth", 20)
    first_turn = transcript.read_text()
    _write_turn(transcript, "stale base secret", "must disappear", 21)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    db = RecallDB(index_dir / "recall.db")
    db.save_chunks(parse_transcript(transcript))
    db.close()
    transcript.write_text(first_turn)
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )

    result = refresh_current_session(index_dir, tmp_path, policy=_policy())

    assert result.state is QueryFreshnessState.REFRESHED
    db = ShardedRecallDB.open(index_dir)
    try:
        chunks = db.load_chunks()
        stale_hits = db.fts_search("stale base secret", limit=10)
        stale_embeddings = db.get_all_embeddings()
        cursor = db.load_query_tail_cursor(result.source_key)
    finally:
        db.close()
    assert [chunk.user_text for chunk in chunks] == ["replacement survives"]
    assert stale_hits == []
    assert stale_embeddings == {}
    assert cursor is not None
    assert cursor["suppresses_base"] == 1


def test_base_rebuild_retires_overlay_only_after_matching_coverage(
    tmp_path, monkeypatch
):
    from synapt.recall.core import TranscriptIndex, parse_transcript

    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "base turn", "base answer", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    db = RecallDB(index_dir / "recall.db")
    db.save_chunks(parse_transcript(transcript))
    db.close()
    _write_turn(transcript, "overlay turn", "overlay answer", 21)
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    refreshed = refresh_current_session(index_dir, tmp_path, policy=_policy())
    assert refreshed.state is QueryFreshnessState.REFRESHED

    db = ShardedRecallDB.open(index_dir)
    db.retire_absorbed_query_tails()
    assert db.load_query_tail_cursor(refreshed.source_key) is not None
    assert db.load_query_tail_chunks()

    rebuilt = TranscriptIndex(
        parse_transcript(transcript),
        use_embeddings=False,
        cache_dir=index_dir,
        db=db,
    )
    rebuilt.save(index_dir)

    db = RecallDB(index_dir / "recall.db")
    try:
        cursor = db.load_query_tail_cursor(refreshed.source_key)
        overlay = db.load_query_tail_chunks()
    finally:
        db.close()
    current = refresh_current_session(index_dir, tmp_path, policy=_policy())
    assert current.state is QueryFreshnessState.REFRESHED
    assert cursor is None
    assert overlay == []


def test_overlay_only_session_hydrates_bounded_resume_listing(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "overlay listing witness", "visible", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    refresh_current_session(index_dir, tmp_path, policy=_policy())

    index = BoundedResumeIndex(ShardedRecallDB.open(index_dir))
    try:
        sessions = index.list_sessions()
    finally:
        index.close()

    assert len(sessions) == 1
    assert sessions[0]["session_id"] == SESSION
    assert sessions[0]["first_message"] == "overlay listing witness"
    assert sessions[0]["turn_count"] == 1


def test_cursor_and_overlay_rollback_together_on_commit_failure(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "first durable tail", "first answer", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    first = refresh_current_session(index_dir, tmp_path, policy=_policy())
    db = RecallDB(index_dir / "recall.db")
    old_cursor = db.load_query_tail_cursor(first.source_key)
    old_chunks = db.load_query_tail_chunks()
    db._conn.execute(
        "CREATE TRIGGER fail_query_tail_cursor BEFORE INSERT "
        "ON query_tail_cursors BEGIN SELECT RAISE(ABORT, 'cursor fault'); END"
    )
    db._conn.commit()
    db.close()
    _write_turn(transcript, "must roll back", "second answer", 21)

    failed = refresh_current_session(index_dir, tmp_path, policy=_policy())

    db = RecallDB(index_dir / "recall.db")
    try:
        new_cursor = db.load_query_tail_cursor(first.source_key)
        new_chunks = db.load_query_tail_chunks()
    finally:
        db.close()
    assert failed.state is QueryFreshnessState.ERROR
    assert old_cursor == new_cursor
    assert old_chunks == new_chunks


def test_wall_cap_prevents_starting_another_atomic_step(tmp_path, monkeypatch):
    transcript = tmp_path / f"{SESSION}.jsonl"
    _write_turn(transcript, "first bounded step " + "x" * 300, "one", 20)
    first_size = transcript.stat().st_size
    _write_turn(transcript, "second bounded step " + "y" * 300, "two", 21)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    monkeypatch.setattr(
        "synapt.recall.query_freshness._acquire_build_lock",
        lambda data_dir, timeout=0: object(),
    )
    monkeypatch.setattr(
        "synapt.recall.query_freshness._release_build_lock", lambda fd: None
    )
    times = iter([0.0, 0.0, 6.0, 6.0])
    monkeypatch.setattr(
        "synapt.recall.query_freshness.time.monotonic", lambda: next(times)
    )

    result = refresh_current_session(
        index_dir,
        tmp_path,
        policy=_policy(step_bytes=first_size, wall_seconds=5),
    )

    assert result.state is QueryFreshnessState.PARTIAL
    assert result.reason == "wall_cap"
    assert result.indexed_now_bytes == first_size
    assert result.remaining_bytes == transcript.stat().st_size - first_size


def test_search_and_quick_label_success_not_only_empty_results(monkeypatch, tmp_path):
    from synapt.recall import server

    fake_index = SimpleNamespace(
        lookup=lambda *args, **kwargs: "SEARCHABLE RESULT",
        _embedding_status="available",
        _embedding_reason="",
        _last_conflicts=[],
        _last_diagnostics=None,
        sessions={},
    )
    calls: list[Path] = []
    events: list[str] = []
    monkeypatch.setattr(server, "project_index_dir", lambda: tmp_path)
    monkeypatch.setattr(
        server,
        "_get_index",
        lambda **kwargs: events.append("read") or fake_index,
    )
    monkeypatch.setattr(
        server,
        "_query_freshness_line",
        lambda index_dir: (
            calls.append(index_dir)
            or events.append("refresh")
            or "Freshness: CURRENT witness"
        ),
    )
    monkeypatch.setattr(
        "synapt.recall.live.search_live_transcript", lambda *args, **kwargs: ""
    )

    search = server.recall_search("query freshness witness")
    quick = server.recall_quick("query freshness witness")

    assert search.endswith("Freshness: CURRENT witness")
    assert quick.endswith("Freshness: CURRENT witness")
    assert calls == [tmp_path, tmp_path]
    assert events == ["refresh", "read", "refresh", "read"]


def test_successful_refresh_invalidates_the_server_cache(monkeypatch, tmp_path):
    from synapt.recall import server
    from synapt.recall.query_freshness import QueryFreshnessResult

    invalidations: list[str] = []
    monkeypatch.setattr(
        "synapt.recall.query_freshness.refresh_current_session",
        lambda index_dir, caller_root: QueryFreshnessResult(
            state=QueryFreshnessState.REFRESHED,
            session_id=SESSION,
            indexed_now_bytes=10,
            index_changed=True,
        ),
    )
    monkeypatch.setattr(
        server, "_invalidate_cache", lambda: invalidations.append("invalidated")
    )

    line = server._query_freshness_line(tmp_path)

    assert line.startswith("Freshness: REFRESHED")
    assert invalidations == ["invalidated"]
