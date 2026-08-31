from __future__ import annotations

import json
import os
import subprocess
import sys
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import pytest

from synapt.checkpoint import (
    CHECKPOINT_BYTES,
    EVENT_BYTES,
    TAIL_BYTES,
    capture_checkpoint,
    checkpoint_path,
    format_checkpoint,
    is_newer_than,
    main,
    read_checkpoint,
    write_checkpoint,
)


def _record(role: str, text: str) -> bytes:
    return (json.dumps({
        "type": "response_item",
        "timestamp": "2026-01-01T00:00:00Z",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
        },
    }) + "\n").encode()


def _payload(path: Path, cwd: Path) -> dict:
    return {
        "session_id": "session-exact",
        "transcript_path": str(path),
        "cwd": str(cwd),
        "reason": "other",
        "runtime": "codex",
        "hook_event_name": "SessionEnd",
    }


def test_sparse_multi_gigabyte_transcript_reads_only_tail(tmp_path):
    transcript = tmp_path / "sparse.jsonl"
    with transcript.open("wb") as stream:
        stream.seek(4 * 1024**3)
        stream.write(b"\n" + _record("user", "last user") + _record("assistant", "last answer"))

    result = capture_checkpoint(_payload(transcript, tmp_path))

    assert result["transcript_size"] > 4 * 1024**3
    assert result["tail_bytes_read"] == TAIL_BYTES
    assert result["last_user_text"] == "last user"
    assert result["last_assistant_text"] == "last answer"
    assert result["truncated"] is True


def test_oversized_jsonl_line_does_not_hide_parseable_tail(tmp_path):
    transcript = tmp_path / "oversized.jsonl"
    transcript.write_bytes(b"x" * (TAIL_BYTES * 2) + b"\n" + _record("user", "bounded tail"))

    result = capture_checkpoint(_payload(transcript, tmp_path))

    assert result["tail_bytes_read"] == TAIL_BYTES
    assert result["last_user_text"] == "bounded tail"
    assert result["last_assistant_text"] is None
    assert result["parse_status"] == "partial"


def test_malformed_trailing_json_preserves_last_parseable_turns(tmp_path):
    transcript = tmp_path / "malformed.jsonl"
    transcript.write_bytes(_record("user", "question") + _record("assistant", "answer") + b'{"broken":')

    result = capture_checkpoint(_payload(transcript, tmp_path))

    assert result["last_user_text"] == "question"
    assert result["last_assistant_text"] == "answer"
    assert result["parse_status"] == "partial"


def test_missing_transcript_is_explicitly_unavailable(tmp_path):
    transcript = tmp_path / "missing.jsonl"
    result = capture_checkpoint(_payload(transcript, tmp_path))
    assert result["transcript_path"] == str(transcript)
    assert result["tail_bytes_read"] == 0
    assert result["parse_status"] == "unavailable"
    assert "unavailable in bounded transcript tail" in format_checkpoint(result)


def test_claude_compaction_handoff_is_not_mislabeled_as_user_text(tmp_path):
    transcript = tmp_path / "claude.jsonl"
    transcript.write_text(json.dumps({
        "type": "user",
        "message": {"content": (
            "This session is being continued from a previous conversation that ran out of "
            "context. The summary below covers the earlier portion of the conversation.\n\n"
            "Summary:\nRuntime-authored handoff"
        )},
    }) + "\n", encoding="utf-8")

    result = capture_checkpoint(_payload(transcript, tmp_path))

    assert result["last_user_text"] is None
    assert result["parse_status"] == "unavailable"


def test_checkpoint_uses_the_complete_shared_secret_scrubber(tmp_path):
    secrets = [
        "ak-abcdefghijklmnopqrst",
        "pypi-" + ("a" * 50),
        "Authorization: Bearer abcdefghijklmnop",
        "PASSWORD=supersecret1",
        "eyJabcdefghij.abcdefghijk.abcdefghijkl",
        "postgres://user:secret@db.example/test",
    ]
    transcript = tmp_path / "secrets.jsonl"
    transcript.write_bytes(_record("user", "\n".join(secrets)))

    result = capture_checkpoint(_payload(transcript, tmp_path))
    captured = result["last_user_text"]

    assert captured.count("[REDACTED:") == len(secrets)
    assert all(secret not in captured for secret in secrets)


def test_payload_paths_are_used_exactly_without_discovery(tmp_path):
    exact = tmp_path / "chosen.jsonl"
    decoy = tmp_path / "newer-decoy.jsonl"
    exact.write_bytes(_record("user", "chosen"))
    decoy.write_bytes(_record("user", "decoy"))
    os.utime(decoy, (exact.stat().st_atime + 100, exact.stat().st_mtime + 100))

    result = capture_checkpoint(_payload(exact, tmp_path))

    assert result["transcript_path"] == str(exact)
    assert result["cwd"] == str(tmp_path)
    assert result["session_id"] == "session-exact"
    assert result["last_user_text"] == "chosen"


def test_files_touched_are_bounded_and_include_structured_codex_paths(tmp_path):
    transcript = tmp_path / "files.jsonl"
    tool = {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "apply_patch",
            "arguments": json.dumps({
                "file_path": "/workspace/exact.py",
                "patch": "*** Begin Patch\n*** Update File: src/changed.py\n*** End Patch",
            }),
        },
    }
    transcript.write_text(json.dumps(tool) + "\n", encoding="utf-8")

    result = capture_checkpoint(_payload(transcript, tmp_path))

    assert result["files_touched"] == ["/workspace/exact.py", "src/changed.py"]


def test_checkpoint_path_does_not_run_legacy_store_migration(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    legacy = tmp_path / ".synapse-recall"
    legacy.mkdir()
    (legacy / "journal.jsonl").write_text("legacy\n", encoding="utf-8")

    resolved = checkpoint_path(tmp_path)

    assert resolved == tmp_path / ".synapt" / "recall" / "worktrees" / tmp_path.name / "checkpoint.json"
    assert legacy.is_dir()
    assert not (tmp_path / ".synapt").exists()


def test_linked_worktree_path_uses_git_pointer_without_running_git(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    main = tmp_path / "main"
    linked = tmp_path / "feature"
    gitdir = main / ".git" / "worktrees" / "feature"
    gitdir.mkdir(parents=True)
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")

    assert checkpoint_path(linked) == (
        main / ".synapt" / "recall" / "worktrees" / "feature" / "checkpoint.json"
    )


def test_nested_cwd_resolves_shared_gripspace_worktree(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    gripspace = tmp_path / "space"
    repository = gripspace / "recall"
    nested = repository / "tests" / "recall"
    (gripspace / ".gitgrip").mkdir(parents=True)
    (gripspace / ".gitgrip" / "griptrees.json").write_text("{}\n", encoding="utf-8")
    (repository / ".git").mkdir(parents=True)
    nested.mkdir(parents=True)

    assert checkpoint_path(nested) == (
        gripspace / ".synapt" / "recall" / "worktrees" / "recall" / "checkpoint.json"
    )


def test_nested_cwd_resolves_standalone_repository(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    repository = tmp_path / "project"
    nested = repository / "src" / "package"
    (repository / ".git").mkdir(parents=True)
    nested.mkdir(parents=True)

    assert checkpoint_path(nested) == (
        repository / ".synapt" / "recall" / "worktrees" / "project" / "checkpoint.json"
    )


def test_explicit_standalone_root_has_stable_namespace_from_nested_cwd(
    tmp_path, monkeypatch,
):
    repository = tmp_path / "project"
    nested = repository / "src" / "package"
    (repository / ".git").mkdir(parents=True)
    nested.mkdir(parents=True)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(repository))
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)

    expected = (
        repository / ".synapt" / "recall" / "worktrees" / "project" / "checkpoint.json"
    )
    assert checkpoint_path(nested) == expected
    assert checkpoint_path(repository) == expected


def test_explicit_standalone_root_ignores_ambient_gripspace_classification(
    tmp_path, monkeypatch,
):
    repository = tmp_path / "project"
    nested = repository / "src" / "package"
    ambient_gripspace = tmp_path / "ambient-space"
    (repository / ".git").mkdir(parents=True)
    nested.mkdir(parents=True)
    ambient_gripspace.mkdir()
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(repository))
    monkeypatch.setenv("GRIPSPACE_ROOT", str(ambient_gripspace))
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)

    expected = (
        repository / ".synapt" / "recall" / "worktrees" / "project" / "checkpoint.json"
    )
    assert checkpoint_path(nested) == expected
    assert checkpoint_path(repository) == expected


def test_linked_griptree_resolves_parent_gripspace_store(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    main_space = tmp_path / "main-space"
    main_repo = main_space / "project"
    linked_space = tmp_path / "feature-space"
    linked_repo = linked_space / "project"
    nested = linked_repo / "src"
    gitdir = main_repo / ".git" / "worktrees" / "feature"
    gitdir.mkdir(parents=True)
    (main_space / ".gitgrip").mkdir()
    (main_space / ".gitgrip" / "griptrees.json").write_text("{}\n", encoding="utf-8")
    (linked_space / ".gitgrip").mkdir(parents=True)
    (linked_space / ".gitgrip" / "griptree.json").write_text("{}\n", encoding="utf-8")
    linked_repo.mkdir()
    (linked_repo / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    nested.mkdir()

    assert checkpoint_path(nested) == (
        main_space
        / ".synapt"
        / "recall"
        / "worktrees"
        / "project"
        / "checkpoint.json"
    )

    other_main = main_space / "other" / ".git" / "worktrees" / "feature"
    other_main.mkdir(parents=True)
    other_linked = linked_space / "other"
    other_linked.mkdir()
    (other_linked / ".git").write_text(
        f"gitdir: {other_main}\n", encoding="utf-8",
    )
    assert checkpoint_path(other_linked) == (
        main_space
        / ".synapt"
        / "recall"
        / "worktrees"
        / "other"
        / "checkpoint.json"
    )

    member_expected = (
        main_space
        / ".synapt"
        / "recall"
        / "worktrees"
        / "feature-space"
        / "checkpoint.json"
    )
    assert checkpoint_path(linked_space) == member_expected
    docs = linked_space / "docs"
    docs.mkdir()
    assert checkpoint_path(docs) == member_expected

    for index in range(256):
        (linked_space / f"unrelated-{index}.txt").touch()
    assert checkpoint_path(linked_space) == member_expected


def test_checkpoint_import_does_not_load_recall_or_child_process_stack():
    source_root = Path(__file__).parents[2] / "src"
    script = """
import sys
import synapt.checkpoint
for forbidden in (
    'sqlite3', 'subprocess', 'synapt.recall.core', 'synapt.recall.storage',
    'synapt.recall.bm25', 'synapt.recall.hybrid',
):
    assert forbidden not in sys.modules, forbidden
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_atomic_replace_failure_preserves_previous_checkpoint(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_record("user", "new"))
    destination = tmp_path / "checkpoint.json"
    destination.write_text('{"old": true}\n', encoding="utf-8")

    with patch("synapt.checkpoint.checkpoint_path", return_value=destination), \
         patch("synapt.checkpoint.os.replace", side_effect=OSError("unwritable")):
        with pytest.raises(OSError, match="unwritable"):
            write_checkpoint(_payload(transcript, tmp_path))

    assert json.loads(destination.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob(".checkpoint.*.tmp")) == []


def test_writer_never_emits_a_checkpoint_the_reader_rejects(tmp_path):
    transcript = tmp_path / "escaped-paths.jsonl"
    records = []
    for index in range(32):
        records.append({
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "arguments": json.dumps({
                    "file_path": f"/tmp/{index}-" + ("\x01" * 1000) + ".txt",
                }),
            },
        })
    transcript.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    destination = tmp_path / "checkpoint.json"

    with patch("synapt.checkpoint.checkpoint_path", return_value=destination):
        write_checkpoint(_payload(transcript, tmp_path))
        recovered = read_checkpoint(tmp_path)

    assert destination.stat().st_size <= CHECKPOINT_BYTES
    assert recovered is not None
    assert recovered["files_touched"]


@pytest.mark.parametrize(
    "schema_version", [None, False, True, 0, 1.0, 2, "1"],
)
def test_reader_rejects_wrong_schema_domain(tmp_path, schema_version):
    destination = tmp_path / "checkpoint.json"
    destination.write_text(json.dumps({
        "schema_version": schema_version,
        "captured_at": "2026-01-01T00:00:00Z",
        "last_user_text": "must not surface",
    }), encoding="utf-8")

    with patch("synapt.checkpoint.checkpoint_path", return_value=destination):
        assert read_checkpoint(tmp_path) is None


def test_peak_allocation_is_independent_of_transcript_size(tmp_path):
    small = tmp_path / "small.jsonl"
    huge = tmp_path / "huge.jsonl"
    small.write_bytes(b"x" * (TAIL_BYTES * 2) + b"\n" + _record("user", "small"))
    with huge.open("wb") as stream:
        stream.seek(8 * 1024**3)
        stream.write(b"\n" + _record("user", "huge"))

    peaks = []
    for path in (small, huge):
        tracemalloc.start()
        capture_checkpoint(_payload(path, tmp_path))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)

    assert max(peaks) < 4 * 1024 * 1024
    assert abs(peaks[0] - peaks[1]) < 1024 * 1024


def test_checkpoint_only_surfaces_after_newest_authored_journal():
    checkpoint = {"captured_at": "2026-01-02T00:00:00+00:00"}
    assert is_newer_than(checkpoint, "2026-01-01T00:00:00+00:00") is True
    assert is_newer_than(checkpoint, "2026-01-03T00:00:00+00:00") is False


def test_event_input_is_bounded(tmp_path, monkeypatch, capsys):
    event = tmp_path / "event.json"
    event.write_bytes(b"{" + b"x" * EVENT_BYTES + b"}")
    assert main(["--event-json", str(event)]) == 1
    assert "exceeds" in capsys.readouterr().err


def _claude_line(role: str, text: str) -> bytes:
    content = text if role == "user" else [{"type": "text", "text": text}]
    return (json.dumps({
        "type": role,
        "timestamp": "2026-08-31T12:06:07Z",
        "message": {"role": role, "content": content},
    }) + "\n").encode()


def test_harness_user_lines_do_not_replace_the_operators_last_words(tmp_path):
    """A background-task notice is written as a user turn. It is not the user."""
    transcript = tmp_path / "claude.jsonl"
    transcript.write_bytes(
        _claude_line("user", "can you show me the herdr changes as html")
        + _claude_line("assistant", "working on it")
        + _claude_line("user", "<task-notification>\n<task-id>b1tvbumcj</task-id>\n"
                               "<status>completed</status>\n</task-notification>")
    )
    result = capture_checkpoint(_payload(transcript, tmp_path))
    assert result["last_user_text"] == "can you show me the herdr changes as html"
    assert result["last_assistant_text"] == "working on it"


def test_a_real_trailing_question_still_wins(tmp_path):
    """Control for the test above: a plain user line after the reply IS the last word,
    and prose that merely mentions a tag keeps its author."""
    transcript = tmp_path / "claude.jsonl"
    transcript.write_bytes(
        _claude_line("user", "first")
        + _claude_line("assistant", "reply")
        + _claude_line("user", "why did the <task-notification> arrive twice?")
    )
    result = capture_checkpoint(_payload(transcript, tmp_path))
    assert result["last_user_text"] == "why did the <task-notification> arrive twice?"
