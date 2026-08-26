from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from synapt.recall.compaction import (
    _CLAUDE_PREFIX,
    compaction_index_ready,
    extract_compaction_summaries,
    format_compaction_summary,
    latest_compaction_summary,
    update_compaction_summary_index,
)


def _jsonl(path: Path, *records: dict) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_extracts_claude_runtime_summary_after_boundary(tmp_path):
    path = tmp_path / "claude-session.jsonl"
    _jsonl(
        path,
        {"type": "system", "subtype": "compact_boundary", "timestamp": "2026-01-01T00:00:00Z"},
        {
            "type": "user",
            "sessionId": "session-exact",
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {"content": (
                "This session is being continued from a previous conversation that ran out of "
                "context. The summary below covers the earlier portion of the conversation.\n\n"
                "Summary:\nThe exact compacted handoff."
            )},
        },
    )

    summaries = extract_compaction_summaries(path)

    assert summaries == [{
        "runtime": "claude",
        "session_id": "session-exact",
        "timestamp": "2026-01-01T00:00:01Z",
        "source_path": str(path),
        "summary": "The exact compacted handoff.",
        "status": "available",
        "truncated": False,
    }]


def test_codex_encrypted_summary_is_explicitly_unavailable(tmp_path):
    path = tmp_path / "codex-session.jsonl"
    _jsonl(
        path,
        {"type": "session_meta", "payload": {"id": "codex-exact"}},
        {
            "type": "compacted",
            "timestamp": "2026-01-02T00:00:00Z",
            "payload": {"replacement_history": [{"type": "compaction", "encrypted_content": "opaque"}]},
        },
    )

    summary = extract_compaction_summaries(path)[0]

    assert summary["session_id"] == "codex-exact"
    assert summary["summary"] is None
    assert summary["status"] == "encrypted-unavailable"
    assert "not available" in format_compaction_summary(summary)


def test_claude_summary_is_scrubbed_before_sidecar_persistence(tmp_path):
    path = tmp_path / "secret.jsonl"
    _jsonl(
        path,
        {"type": "system", "subtype": "compact_boundary"},
        {
            "type": "user",
            "message": {"content": (
                "This session is being continued from a previous conversation that ran out of "
                "context. The summary below covers the earlier portion of the conversation.\n\n"
                "Summary:\nToken: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
            )},
        },
    )

    summary = extract_compaction_summaries(path)[0]["summary"]

    assert "ghp_" not in summary


def test_scrubber_failure_cannot_persist_plaintext_summary(tmp_path):
    source = tmp_path / "transcripts"
    source.mkdir()
    transcript = source / "secret.jsonl"
    _jsonl(
        transcript,
        {"type": "system", "subtype": "compact_boundary"},
        {
            "type": "user",
            "message": {"content": f"{_CLAUDE_PREFIX}\n\nSummary:\nplaintext-secret"},
        },
    )
    sidecar = tmp_path / "compaction-summaries.json"

    with patch("synapt.recall.compaction.scrub_text", side_effect=RuntimeError("scrub failed")), \
         patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar), \
         pytest.raises(RuntimeError, match="scrub failed"):
        update_compaction_summary_index([source], project=tmp_path)

    assert not sidecar.exists()


def test_ordinary_user_turn_consumes_claude_boundary_provenance(tmp_path):
    path = tmp_path / "claude-session.jsonl"
    _jsonl(
        path,
        {"type": "system", "subtype": "compact_boundary"},
        {"type": "attachment", "content": "runtime metadata may intervene"},
        {"type": "user", "message": {"content": "ordinary user turn"}},
        {
            "type": "user",
            "message": {"content": f"{_CLAUDE_PREFIX}\n\nSummary:\nuser imitation"},
        },
    )

    assert extract_compaction_summaries(path) == []


def test_malformed_record_invalidates_claude_boundary_provenance(tmp_path):
    path = tmp_path / "claude-session.jsonl"
    path.write_text(
        json.dumps({"type": "system", "subtype": "compact_boundary"})
        + "\n{malformed\n"
        + json.dumps({
            "type": "user",
            "message": {"content": f"{_CLAUDE_PREFIX}\n\nSummary:\nuser imitation"},
        })
        + "\n",
        encoding="utf-8",
    )

    assert extract_compaction_summaries(path) == []


def test_blank_separator_does_not_consume_claude_boundary(tmp_path):
    path = tmp_path / "claude-session.jsonl"
    path.write_text(
        json.dumps({"type": "system", "subtype": "compact_boundary"})
        + "\n\n"
        + json.dumps({
            "type": "user",
            "message": {"content": f"{_CLAUDE_PREFIX}\n\nSummary:\nruntime handoff"},
        })
        + "\n",
        encoding="utf-8",
    )

    summaries = extract_compaction_summaries(path)
    assert [item["summary"] for item in summaries] == ["runtime handoff"]


def test_incremental_update_preserves_unchanged_summaries(tmp_path):
    source = tmp_path / "transcripts"
    source.mkdir()
    changed = source / "changed.jsonl"
    _jsonl(changed, {"type": "compacted", "timestamp": "2026-01-03T00:00:00Z"})
    sidecar = tmp_path / "compaction-summaries.json"
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "summaries": [{
            "runtime": "claude", "session_id": "old", "timestamp": "2025-01-01T00:00:00Z",
            "source_path": "/worktrees/other/transcripts/unchanged.jsonl", "worktree": "other",
            "summary": "Keep me", "status": "available",
            "truncated": False,
        }],
    }), encoding="utf-8")

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        update_compaction_summary_index([source], project=tmp_path, previous_manifest={"source_files": []})
        latest = latest_compaction_summary(tmp_path)

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert len(data["summaries"]) == 2
    assert latest["runtime"] == "codex"
    assert latest["worktree"] == tmp_path.name


def test_latest_summary_is_scoped_to_current_worktree(tmp_path):
    sidecar = tmp_path / "compaction-summaries.json"
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "summaries": [
            {"worktree": "other", "timestamp": "2026-02-02", "summary": "wrong"},
            {"worktree": tmp_path.name, "timestamp": "2026-02-01", "summary": "right"},
        ],
    }), encoding="utf-8")

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        latest = latest_compaction_summary(tmp_path)

    assert latest["summary"] == "right"


def test_index_keeps_latest_summary_per_worktree(tmp_path):
    source_a = tmp_path / "a" / "transcripts"
    source_b = tmp_path / "b" / "transcripts"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)
    _jsonl(source_a / "old.jsonl", {"type": "compacted", "timestamp": "2026-01-01T00:00:00Z"})
    _jsonl(source_a / "new.jsonl", {"type": "compacted", "timestamp": "2026-01-03T00:00:00Z"})
    _jsonl(source_b / "only.jsonl", {"type": "compacted", "timestamp": "2026-01-02T00:00:00Z"})
    sidecar = tmp_path / "compaction-summaries.json"

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        update_compaction_summary_index([source_a, source_b], project=tmp_path)
        assert compaction_index_ready(tmp_path) is True

    summaries = json.loads(sidecar.read_text(encoding="utf-8"))["summaries"]
    assert [(item["worktree"], item["timestamp"]) for item in summaries] == [
        ("a", "2026-01-03T00:00:00Z"),
        ("b", "2026-01-02T00:00:00Z"),
    ]


@pytest.mark.parametrize("payload", [[], None, "wrong-shape", 1])
def test_non_object_sidecar_is_not_ready_or_readable(tmp_path, payload):
    sidecar = tmp_path / "compaction-summaries.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        assert compaction_index_ready(tmp_path) is False
        assert latest_compaction_summary(tmp_path) is None


@pytest.mark.parametrize(
    "invalid_record",
    [
        None,
        "wrong-shape",
        {},
        {"worktree": " "},
        {"worktree": "nested/name"},
        {"source_path": "/tmp/not-transcripts.jsonl"},
        {"source_path": "../transcripts/session.jsonl"},
    ],
)
def test_sidecar_with_unusable_summary_record_is_not_ready(
    tmp_path, invalid_record,
):
    sidecar = tmp_path / "compaction-summaries.json"
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "summaries": [{"worktree": "valid-shape"}, invalid_record],
    }), encoding="utf-8")

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        assert compaction_index_ready(tmp_path) is False


def test_legacy_source_path_supplies_resolvable_worktree_identity(tmp_path):
    sidecar = tmp_path / "compaction-summaries.json"
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "summaries": [{
            "source_path": str(tmp_path / "legacy" / "transcripts" / "session.jsonl"),
        }],
    }), encoding="utf-8")

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        assert compaction_index_ready(tmp_path) is True


def test_rebuild_drops_invalid_object_record_instead_of_normalizing_it(tmp_path):
    source = tmp_path / "transcripts"
    source.mkdir()
    sidecar = tmp_path / "compaction-summaries.json"
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "summaries": [{}],
    }), encoding="utf-8")

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        assert compaction_index_ready(tmp_path) is False
        update_compaction_summary_index(
            [source], project=tmp_path, previous_manifest={"source_files": []},
        )
        assert compaction_index_ready(tmp_path) is True

    assert json.loads(sidecar.read_text(encoding="utf-8"))["summaries"] == []


def test_rebuild_normalizes_padded_identity_before_deduplication(tmp_path):
    source = tmp_path / "transcripts"
    source.mkdir()
    sidecar = tmp_path / "compaction-summaries.json"
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "summaries": [
            {"worktree": " project ", "timestamp": "2026-01-02", "summary": "new"},
            {"worktree": "project", "timestamp": "2026-01-01", "summary": "old"},
        ],
    }), encoding="utf-8")

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        update_compaction_summary_index(
            [source], project=tmp_path, previous_manifest={"source_files": []},
        )

    summaries = json.loads(sidecar.read_text(encoding="utf-8"))["summaries"]
    assert [(item["worktree"], item["summary"]) for item in summaries] == [
        ("project", "new"),
    ]


@pytest.mark.parametrize("schema_version", [None, False, True, 0, 1.0, 2, "1"])
def test_wrong_schema_is_not_read_or_preserved(tmp_path, schema_version):
    source = tmp_path / "transcripts"
    source.mkdir()
    sidecar = tmp_path / "compaction-summaries.json"
    sidecar.write_text(json.dumps({
        "schema_version": schema_version,
        "summaries": [{
            "worktree": tmp_path.name,
            "timestamp": "2026-01-01",
            "summary": "must not surface",
        }],
    }), encoding="utf-8")

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        assert compaction_index_ready(tmp_path) is False
        assert latest_compaction_summary(tmp_path) is None
        update_compaction_summary_index(
            [source], project=tmp_path, previous_manifest={"source_files": []},
        )

    repaired = json.loads(sidecar.read_text(encoding="utf-8"))
    assert repaired == {"schema_version": 1, "summaries": []}


def _manifest_for(*paths: Path) -> dict:
    return {
        "source_files": [
            {
                "name": path.name,
                "dir": path.parent.name,
                "source_path": str(path),
                "mtime": path.stat().st_mtime,
                "size": path.stat().st_size,
            }
            for path in paths
        ],
    }


def test_changed_winner_falls_back_to_older_summary(tmp_path):
    source = tmp_path / "a" / "transcripts"
    source.mkdir(parents=True)
    old = source / "old.jsonl"
    winner = source / "winner.jsonl"
    _jsonl(old, {"type": "compacted", "timestamp": "2026-01-01T00:00:00Z"})
    _jsonl(winner, {"type": "compacted", "timestamp": "2026-01-03T00:00:00Z"})
    sidecar = tmp_path / "compaction-summaries.json"

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        update_compaction_summary_index([source], project=tmp_path)
        previous = _manifest_for(old, winner)
        _jsonl(winner, {"type": "user", "message": {"content": "ordinary turn"}})
        update_compaction_summary_index([source], project=tmp_path, previous_manifest=previous)

    summaries = json.loads(sidecar.read_text(encoding="utf-8"))["summaries"]
    assert [(item["source_path"], item["timestamp"]) for item in summaries] == [
        (str(old), "2026-01-01T00:00:00Z"),
    ]


def test_deleted_winner_falls_back_to_older_summary(tmp_path):
    source = tmp_path / "a" / "transcripts"
    source.mkdir(parents=True)
    old = source / "old.jsonl"
    winner = source / "winner.jsonl"
    _jsonl(old, {"type": "compacted", "timestamp": "2026-01-01T00:00:00Z"})
    _jsonl(winner, {"type": "compacted", "timestamp": "2026-01-03T00:00:00Z"})
    sidecar = tmp_path / "compaction-summaries.json"

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        update_compaction_summary_index([source], project=tmp_path)
        previous = _manifest_for(old, winner)
        winner.unlink()
        update_compaction_summary_index([source], project=tmp_path, previous_manifest=previous)

    summaries = json.loads(sidecar.read_text(encoding="utf-8"))["summaries"]
    assert [(item["source_path"], item["timestamp"]) for item in summaries] == [
        (str(old), "2026-01-01T00:00:00Z"),
    ]


@pytest.mark.parametrize("mutation", ["delete", "remove-summary"])
def test_changed_or_deleted_legacy_winner_uses_inferred_worktree_for_fallback(
    tmp_path, mutation,
):
    source = tmp_path / "legacy" / "transcripts"
    source.mkdir(parents=True)
    old = source / "old.jsonl"
    winner = source / "winner.jsonl"
    _jsonl(old, {"type": "compacted", "timestamp": "2026-01-01T00:00:00Z"})
    _jsonl(winner, {"type": "compacted", "timestamp": "2026-01-03T00:00:00Z"})
    sidecar = tmp_path / "compaction-summaries.json"
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "summaries": [{
            "runtime": "codex",
            "session_id": "winner",
            "timestamp": "2026-01-03T00:00:00Z",
            "source_path": str(winner),
            "summary": None,
            "status": "encrypted-unavailable",
            "truncated": False,
        }],
    }), encoding="utf-8")
    previous = _manifest_for(old, winner)
    if mutation == "delete":
        winner.unlink()
    else:
        _jsonl(winner, {"type": "user", "message": {"content": "ordinary turn"}})

    with patch("synapt.recall.compaction.compaction_index_path", return_value=sidecar):
        update_compaction_summary_index(
            [source], project=tmp_path, previous_manifest=previous,
        )

    summaries = json.loads(sidecar.read_text(encoding="utf-8"))["summaries"]
    assert [(item["source_path"], item["worktree"]) for item in summaries] == [
        (str(old), "legacy"),
    ]
