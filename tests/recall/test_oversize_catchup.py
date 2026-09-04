"""Contract tests for R3.1 (data growth): chunked/resumable oversize-transcript
indexing via the query_tail overlay, and the cmd_catchup wiring that drives it."""

from __future__ import annotations

import json
from pathlib import Path

from synapt.recall.core import parse_transcript
from synapt.recall.query_freshness import (
    QueryFreshnessPolicy,
    QueryFreshnessState,
    catchup_oversize_transcripts,
    index_oversize_source,
    oversize_transcripts,
)
from synapt.recall.resume import CallerTranscript
from synapt.recall.sharded_db import ShardedRecallDB


def _line(kind: str, text: str, idx: int) -> str:
    ts = f"2026-09-03T10:{idx % 60:02d}:00Z"
    if kind == "user":
        payload = {
            "type": "user",
            "message": {"role": "user", "content": text},
            "sessionId": "oversize-session",
            "uuid": f"u-{idx}",
            "timestamp": ts,
        }
    else:
        payload = {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
            "sessionId": "oversize-session",
            "uuid": f"a-{idx}",
            "timestamp": ts,
        }
    return json.dumps(payload) + "\n"


def _write_synthetic_transcript(path: Path, turns: int, filler_len: int = 200) -> int:
    filler = "x" * filler_len
    with path.open("w", encoding="utf-8") as stream:
        for i in range(turns):
            stream.write(_line("user", f"turn {i} question {filler}", i * 2))
            stream.write(_line("assistant", f"turn {i} answer {filler}", i * 2 + 1))
    return path.stat().st_size


def _source(path: Path, session_id: str = "oversize-session") -> CallerTranscript:
    stat = path.stat()
    return CallerTranscript(session_id, path, stat.st_mtime, stat.st_size, "")


def _tight_policy(**overrides) -> QueryFreshnessPolicy:
    values = {
        "age_threshold_seconds": 0,
        "byte_trigger": 0,
        "step_bytes": 4096,
        "byte_cap": 8192,
        "wall_seconds": 5.0,
    }
    values.update(overrides)
    return QueryFreshnessPolicy(**values)


def test_index_oversize_source_stops_at_byte_cap_with_partial_state(tmp_path):
    transcript = tmp_path / "oversize-session.jsonl"
    size = _write_synthetic_transcript(transcript, turns=200)
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    result = index_oversize_source(index_dir, _source(transcript), policy=_tight_policy())

    assert result.state is QueryFreshnessState.PARTIAL
    assert result.cut_short is True
    assert result.observed_complete_offset is not None
    assert 0 < result.observed_complete_offset < size
    assert result.remaining_bytes == size - result.observed_complete_offset
    # Either reason is a legitimate, safe stop under a tight byte_cap: the
    # loop-top check catching it cleanly, or the next record not fitting in
    # what's left of the cap. Both leave a valid, resumable partial state --
    # the contract under test is "stopped safely with real progress made",
    # not which of the two equally-safe stop points fired.
    assert result.reason in {"byte_cap", "record_exceeds_byte_cap"}

    db = ShardedRecallDB.open(index_dir)
    try:
        hits = db.query_tail_fts_search("turn 0 question", limit=5)
        assert len(hits) >= 1, "first-chunk content must be searchable after a partial call"
    finally:
        db.close()


def test_index_oversize_source_resumes_across_calls_to_completion(tmp_path):
    transcript = tmp_path / "oversize-session.jsonl"
    size = _write_synthetic_transcript(transcript, turns=200)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    source = _source(transcript)
    policy = _tight_policy()

    rounds = 0
    result = index_oversize_source(index_dir, source, policy=policy)
    rounds += 1
    while result.remaining_bytes and result.remaining_bytes > 0:
        result = index_oversize_source(index_dir, source, policy=policy)
        rounds += 1
        assert rounds < 500, "did not converge -- resumption is not making progress"

    assert result.state is QueryFreshnessState.REFRESHED
    assert result.observed_complete_offset == size
    assert result.remaining_bytes == 0
    assert rounds > 1, "the whole point of the test is that it took more than one call"

    one_shot = parse_transcript(transcript)
    db = ShardedRecallDB.open(index_dir)
    try:
        assert db.query_tail_chunk_count() == len(one_shot)
        hits = db.query_tail_fts_search("turn 199 answer", limit=5)
        assert len(hits) >= 1, "last-chunk content must be searchable once complete"
    finally:
        db.close()


def test_index_oversize_source_is_a_noop_once_complete(tmp_path):
    """A completed file costs one cheap call, not a re-index."""
    transcript = tmp_path / "oversize-session.jsonl"
    _write_synthetic_transcript(transcript, turns=5)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    source = _source(transcript)
    generous_policy = _tight_policy(byte_cap=10 * 1024 * 1024, wall_seconds=10.0)

    first = index_oversize_source(index_dir, source, policy=generous_policy)
    assert first.state is QueryFreshnessState.REFRESHED

    second = index_oversize_source(index_dir, source, policy=generous_policy)
    assert second.state is QueryFreshnessState.CURRENT
    assert second.remaining_bytes == 0
    assert second.indexed_now_bytes == 0
    assert second.indexed_now_chunks == 0


def test_index_oversize_source_detects_rewrite_and_restarts_from_zero(tmp_path):
    """Hardening requirement from R3.1 (data growth): a rewrite of the source file
    mid-progress must not silently resume against the new bytes at the old
    offset -- it must be detected (mtime/size drift) and restart at zero.

    Deliberately checks by CONTENT, not just offset arithmetic: the original
    file's turn 0 carries a marker that appears nowhere in the rewritten
    file. A correct restart re-parses the rewritten file's actual start (so
    the rewritten file's own marker becomes searchable) and drops the old
    overlay entirely (so the original marker stops being searchable) -- an
    incorrect "resume at the old offset against new bytes" would produce
    neither of those cleanly.
    """
    transcript = tmp_path / "oversize-session.jsonl"
    with transcript.open("w", encoding="utf-8") as stream:
        stream.write(_line("user", "MARKER_ORIGINAL_ONLY " + "x" * 200, 0))
        stream.write(_line("assistant", "original answer " + "x" * 200, 1))
        for i in range(1, 100):
            stream.write(_line("user", f"orig turn {i} " + "x" * 200, i * 2))
            stream.write(_line("assistant", f"orig turn {i} " + "x" * 200, i * 2 + 1))
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    source = _source(transcript)
    policy = _tight_policy()

    partial = index_oversize_source(index_dir, source, policy=policy)
    assert partial.state is QueryFreshnessState.PARTIAL

    db = ShardedRecallDB.open(index_dir)
    try:
        assert len(db.query_tail_fts_search("MARKER_ORIGINAL_ONLY", limit=5)) >= 1, (
            "test setup invalid: the original marker must be captured by the "
            "first partial call for this test to prove anything about restart"
        )
    finally:
        db.close()

    # Rewrite the file with DIFFERENT content -- a marker that never appears
    # in the original -- and force the mtime to actually differ, since the
    # drift check is mtime+size (a same-second rewrite on a coarse filesystem
    # clock would otherwise slip past the fast-path stat comparison).
    import os
    import time as _time

    with transcript.open("w", encoding="utf-8") as stream:
        stream.write(_line("user", "MARKER_REWRITTEN_ONLY " + "y" * 200, 0))
        stream.write(_line("assistant", "rewritten answer " + "y" * 200, 1))
        for i in range(1, 100):
            stream.write(_line("user", f"new turn {i} " + "y" * 200, i * 2))
            stream.write(_line("assistant", f"new turn {i} " + "y" * 200, i * 2 + 1))
    os.utime(transcript, (_time.time() + 5, _time.time() + 5))
    new_stat = transcript.stat()

    resumed = index_oversize_source(index_dir, source, policy=policy)
    assert resumed.state is QueryFreshnessState.PARTIAL
    assert resumed.observed_complete_offset is not None
    assert resumed.observed_complete_offset <= new_stat.st_size

    db = ShardedRecallDB.open(index_dir)
    try:
        stale_hits = db.query_tail_fts_search("MARKER_ORIGINAL_ONLY", limit=5)
        assert len(stale_hits) == 0, (
            "stale content from before the rewrite is still indexed -- "
            "the restart-from-zero did not actually clear the old overlay"
        )
        fresh_hits = db.query_tail_fts_search("MARKER_REWRITTEN_ONLY", limit=5)
        assert len(fresh_hits) >= 1, (
            "the rewritten file's own start is not searchable -- restart did "
            "not actually re-parse from byte 0 of the new content"
        )
        cursor = db.load_query_tail_cursor(_source_key_for_test(source))
        assert cursor is not None
        assert int(cursor["source_size"]) == new_stat.st_size
        assert int(cursor["source_mtime_ns"]) == new_stat.st_mtime_ns
    finally:
        db.close()


def _source_key_for_test(source: CallerTranscript) -> str:
    from synapt.recall.storage import query_tail_source_key

    return query_tail_source_key(source.session_id, source.path)


def test_oversize_transcripts_lists_only_files_over_the_ceiling(tmp_path):
    small = tmp_path / "small-session.jsonl"
    small.write_text("x" * 100)
    big = tmp_path / "big-session.jsonl"
    big.write_text("x" * 10_000)

    found = oversize_transcripts([tmp_path], ceiling=1000)

    assert found == [big]


def test_catchup_oversize_transcripts_progresses_multiple_files_within_one_budget(tmp_path):
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    first = transcripts_dir / "session-a.jsonl"
    second = transcripts_dir / "session-b.jsonl"
    _write_synthetic_transcript(first, turns=200)
    _write_synthetic_transcript(second, turns=200)
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    from unittest.mock import patch

    with patch(
        "synapt.recall.core.project_transcript_dirs",
        return_value=[transcripts_dir],
    ):
        results = catchup_oversize_transcripts(
            tmp_path,
            index_dir,
            overall_wall_seconds=20.0,
            per_call_policy=_tight_policy(),
            ceiling=1000,  # the synthetic fixtures are tens of KB, not GB
        )

    assert len(results) == 2
    seen_paths = {r.source_path for r in results}
    assert seen_paths == {first, second}
    for r in results:
        assert r.state in (
            QueryFreshnessState.PARTIAL,
            QueryFreshnessState.REFRESHED,
        )
        assert r.observed_complete_offset is not None and r.observed_complete_offset > 0


def test_cmd_catchup_calls_oversize_catchup_between_build_and_enrich(tmp_path, monkeypatch):
    """Mutation-provable regression pin: removing the wiring in cmd_catchup
    must turn this test red."""
    import argparse
    from unittest.mock import MagicMock, patch

    from synapt.recall import cli as cli_module

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude" / "projects").mkdir(parents=True, exist_ok=True)

    order: list[str] = []

    with patch.object(cli_module, "_acquire_build_lock", return_value=1), \
         patch.object(cli_module, "_release_build_lock"), \
         patch.object(cli_module, "project_transcript_dirs", return_value=[tmp_path]), \
         patch.object(cli_module, "_catchup_archive_and_journal"), \
         patch("synapt.recall.journal.compact_journal", return_value=0), \
         patch.object(
             cli_module.subprocess, "run",
             side_effect=lambda cmd, **kw: order.append(
                 "enrich" if "enrich" in cmd else "build"
             ),
         ), \
         patch(
             "synapt.recall.query_freshness.catchup_oversize_transcripts",
             side_effect=lambda *a, **kw: order.append("oversize_catchup") or [],
         ):
        cli_module.cmd_catchup(argparse.Namespace(no_build=False))

    assert order == ["build", "oversize_catchup", "enrich"], (
        "oversize catchup must run after the build subprocess exits "
        "(build.lock is only free then) and before enrich"
    )


def test_index_oversize_source_stuck_on_a_record_exceeding_the_budget_stays_safe(tmp_path):
    """A single line bigger than the whole per-call budget cannot be chunked
    within one call. This is a real, disclosed limitation (per-line ceiling
    inside a chunk is not yet enforced here the way build_index's
    max_line_bytes enforces it for the foreground path) -- the contract this
    test pins is narrower: the call must stay SAFE and OBSERVABLE when stuck,
    never silently corrupt state or claim false progress."""
    transcript = tmp_path / "oversize-session.jsonl"
    with transcript.open("w", encoding="utf-8") as stream:
        # One line far bigger than the tiny byte_cap below.
        stream.write(_line("user", "z" * 20000, 0))
        stream.write(_line("assistant", "short answer", 1))
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    tiny_policy = _tight_policy(step_bytes=1024, byte_cap=1024, wall_seconds=5.0)

    result = index_oversize_source(index_dir, _source(transcript), policy=tiny_policy)

    assert result.state is QueryFreshnessState.PARTIAL
    assert result.observed_complete_offset == 0, "no progress is the honest outcome here"
    assert result.indexed_now_bytes == 0
    assert result.indexed_now_chunks == 0
    assert result.reason in {"record_exceeds_byte_cap", "incomplete_record"}
    # Calling again must not crash or corrupt -- it repeats the same
    # observable, honest non-progress rather than raising.
    again = index_oversize_source(index_dir, _source(transcript), policy=tiny_policy)
    assert again.state is QueryFreshnessState.PARTIAL
    assert again.observed_complete_offset == 0


def test_catchup_path_completes_a_100mb_file_within_one_call(tmp_path):
    """The catchup path must be wall-governed, not throttled by the
    class-default 32 MiB byte_cap the way the original wiring silently
    was. A 100 MB file under a generous wall budget and the catchup
    path's own (lifted) default byte_cap must converge in the single
    call catchup_oversize_transcripts makes."""
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    big = transcripts_dir / "session-big.jsonl"
    # ~100 MB: measured ~777 bytes/turn pair at filler_len=200 with JSON
    # framing; 150,000 turns clears 100 MB with margin.
    _write_synthetic_transcript(big, turns=150_000)
    size = big.stat().st_size
    assert size > 100 * 1024 * 1024, f"fixture too small: {size} bytes"

    index_dir = tmp_path / "index"
    index_dir.mkdir()

    from unittest.mock import patch

    with patch(
        "synapt.recall.core.project_transcript_dirs",
        return_value=[transcripts_dir],
    ):
        results = catchup_oversize_transcripts(
            tmp_path,
            index_dir,
            overall_wall_seconds=60.0,
            ceiling=1000,  # the fixture is real MB, not GB; lower the bar to exercise it
        )

    assert len(results) == 1
    result = results[0]
    assert result.state is QueryFreshnessState.REFRESHED, (
        f"did not converge in one call: state={result.state.value} "
        f"reason={result.reason!r} remaining={result.remaining_bytes}"
    )
    assert result.observed_complete_offset == size
    assert result.remaining_bytes == 0

    db = ShardedRecallDB.open(index_dir)
    try:
        one_shot = parse_transcript(big)
        assert db.query_tail_chunk_count() == len(one_shot)
    finally:
        db.close()


def test_a_waiting_builder_gets_the_lock_within_2s_of_a_chunked_catchup_call(tmp_path):
    """Releasing and immediately re-acquiring the build lock in the same
    tight loop starves a real waiter (measured:
    a timeout=15 waiter never got in during a 200 MB chunked call, since the
    release-to-reacquire gap is microseconds and a 0.5s poller almost never
    lands inside it). The chunk loop must notice a waiting marker after each
    release and YIELD -- stopping its own call early -- so a foreground
    build gets the lock within one or two lock-cycle intervals instead of
    waiting out the whole catchup call.

    Measured behavior with the fix (not what a naive "waiter squeezes into
    idle gaps while the indexer keeps churning" model would predict): the
    indexer notices the waiter and stops ITSELF almost immediately (~0.15s
    into an 8s call, having made real partial progress), rather than running
    the full span. The waiter's bound (<2s) is what's asserted; how the
    indexer's own call ends (yielded early vs happened to finish) is
    asserted separately, from the indexer's own reported reason.
    """
    import threading
    import time as time_module

    from synapt.recall.cli import _acquire_build_lock, _release_build_lock

    transcript = tmp_path / "oversize-session.jsonl"
    # Measured (single-threaded, this policy): 20,000 turns takes ~7-8s end
    # to end -- enough margin over the 2s assertion below that ordinary
    # thread-scheduling and machine-load noise cannot make this borderline,
    # and enough chunks (~5800) that the loop is nowhere near finishing when
    # the waiter registers a few hundred ms in.
    _write_synthetic_transcript(transcript, turns=20_000)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    data_dir = index_dir.parent

    policy = QueryFreshnessPolicy(
        age_threshold_seconds=0,
        byte_trigger=0,
        step_bytes=4096,  # small: thousands of lock-cycle iterations
        byte_cap=10**9,  # effectively unbounded: byte_cap must not be the stop reason
        wall_seconds=60.0,  # generous: wall_cap must not be the stop reason either
    )

    waiter_result: dict = {}
    indexer_result: dict = {}

    def _waiter():
        waiter_result["start"] = time_module.monotonic()
        fd = _acquire_build_lock(data_dir, timeout=15.0)
        waiter_result["acquired_at"] = time_module.monotonic()
        waiter_result["fd"] = fd
        if fd is not None:
            _release_build_lock(fd)

    def _run_indexer():
        started = time_module.monotonic()
        indexer_result["value"] = index_oversize_source(index_dir, _source(transcript), policy=policy)
        indexer_result["elapsed"] = time_module.monotonic() - started

    indexer_thread = threading.Thread(target=_run_indexer)
    indexer_thread.start()
    time_module.sleep(0.15)  # let the indexer actually start cycling the lock
    waiter_thread = threading.Thread(target=_waiter)
    waiter_thread.start()
    waiter_thread.join(timeout=17.0)
    indexer_thread.join(timeout=35.0)

    assert "acquired_at" in waiter_result, "waiter never returned from _acquire_build_lock"
    assert waiter_result["fd"] is not None, "waiter timed out without acquiring the lock"
    assert "value" in indexer_result, "indexer thread never finished"
    wait_duration = waiter_result["acquired_at"] - waiter_result["start"]
    assert wait_duration < 2.0, (
        f"waiting builder took {wait_duration:.2f}s to get the lock "
        f"(should be well under the yield-check's own poll interval; "
        f"without the fix this is close to the full ~8s call duration)"
    )

    result = indexer_result["value"]
    assert result.state in (QueryFreshnessState.PARTIAL, QueryFreshnessState.REFRESHED)
    # A fixture this size, under an 8s-to-complete policy, stopping in well
    # under 2s can only mean it yielded early -- completing naturally that
    # fast would falsify the very setup this test relies on.
    assert result.state is QueryFreshnessState.PARTIAL, (
        f"expected the call to be cut short by the waiter, not complete "
        f"naturally (indexed {result.indexed_now_chunks} chunks in "
        f"{indexer_result['elapsed']:.2f}s)"
    )
    assert result.reason == "build_lock_yield", (
        f"stopped for the wrong reason: {result.reason!r} "
        f"(expected the waiter-detection yield, not byte_cap/wall_cap/build_lock)"
    )
    assert result.remaining_bytes and result.remaining_bytes > 0
    assert indexer_result["elapsed"] < 3.0, (
        f"indexer took {indexer_result['elapsed']:.2f}s to yield -- "
        f"too slow to count as noticing the waiter promptly"
    )
