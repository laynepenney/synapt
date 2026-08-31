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
    SCHEMA_VERSION,
    TAIL_BYTES,
    capture_checkpoint,
    checkpoint_path,
    format_checkpoint,
    is_newer_than,
    main,
    periodic_checkpoint_path,
    read_checkpoint,
    read_periodic_checkpoint,
    write_checkpoint,
    write_periodic_checkpoint,
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


# ---------------------------------------------------------------------------
# Periodic (per-session) checkpoint — tracked privately, deliberately not linked.
#
# The SessionEnd checkpoint above has ONE slot per worktree: any session's
# write overwrites whatever the last one left, which is why a crashed
# session's wake can show a DIFFERENT session's checkpoint entirely (recall
# unclean-end, `UncleanEnd.checkpoint_session`). A periodic mid-session
# checkpoint makes that worse unless it is keyed by session id -- otherwise a
# five-minute-old periodic tick from an unrelated live session would shadow
# the one crash recovery actually needs. These functions are strictly
# ADDITIVE: `checkpoint_path` / `write_checkpoint` / `read_checkpoint` (the
# SessionEnd path) are asserted unchanged by every test above and must stay
# that way -- resume.py's consumption of periodic checkpoints is a follow-up
# PR, scoped out here specifically to avoid the seam with the in-flight
# GRIPSPACE_ROOT resolver work touching the same caller code.
# ---------------------------------------------------------------------------


def test_periodic_checkpoint_path_is_session_keyed_and_distinct_from_legacy_slot(tmp_path, monkeypatch):
    # Interim per-test isolation (the shared process-environment fixture will supply this;
    # until it lands, these three lines are the established idiom this file
    # already uses elsewhere -- not a new mechanism). Ambient SYNAPT_RECALL_ROOT
    # or GRIPSPACE_ROOT makes checkpoint_path() IGNORE tmp_path entirely and
    # resolve to the real shared store; measured live in this session.
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    legacy = checkpoint_path(tmp_path)
    a = periodic_checkpoint_path(tmp_path, "session-aaaa")
    b = periodic_checkpoint_path(tmp_path, "session-bbbb")

    assert a != legacy
    assert b != legacy
    assert a != b
    assert a.parent == b.parent == legacy.parent / "checkpoints"


def test_periodic_checkpoint_rejects_a_session_id_that_is_not_a_bare_label(tmp_path, monkeypatch):
    # Interim per-test isolation (the shared process-environment fixture will supply this;
    # until it lands, these three lines are the established idiom this file
    # already uses elsewhere -- not a new mechanism). Ambient SYNAPT_RECALL_ROOT
    # or GRIPSPACE_ROOT makes checkpoint_path() IGNORE tmp_path entirely and
    # resolve to the real shared store; measured live in this session.
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    overlong = "a" * 300
    for bad in ("", ".", "..", "a/b", "a\\b", "a\x00b", "a\x01b", "a\x1fb", overlong):
        with pytest.raises(ValueError):
            periodic_checkpoint_path(tmp_path, bad)


@pytest.mark.parametrize(
    "bad", ["Session-AAAA", "SESSION-1234", "session-ABCD", "sessionAaaa"],
)
def test_periodic_checkpoint_rejects_a_session_id_containing_uppercase_ascii(tmp_path, bad, monkeypatch):
    """Atlas r1, checkpoint-v3 P1: on this macOS filesystem, "session-aaaa" and
    "Session-AAAA" are accepted labels that resolve to the SAME inode -- the
    direct bare-label spelling cannot promise noncollision by default. The
    filename key must be injective, so the accepted alphabet is constrained
    to canonical lowercase rather than left to case-fold silently."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    with pytest.raises(ValueError):
        periodic_checkpoint_path(tmp_path, bad)


def test_periodic_checkpoint_case_distinct_session_ids_never_both_resolve(tmp_path, monkeypatch):
    """The collision witness itself: because a mixed-case spelling is refused
    outright rather than silently folded, only the canonical lowercase
    spelling of a given session ever reaches the filesystem, so two spellings
    that a case-insensitive filesystem would otherwise treat as one file can
    never both be written."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    canonical = periodic_checkpoint_path(tmp_path, "session-aaaa")
    assert canonical is not None
    with pytest.raises(ValueError):
        periodic_checkpoint_path(tmp_path, "Session-AAAA")


@pytest.mark.parametrize(
    "bad",
    [
        "session-\u00c4",  # LATIN CAPITAL LETTER A WITH DIAERESIS
        "session-\u00e4",  # LATIN SMALL LETTER A WITH DIAERESIS -- case-folds to the above
        "session-caf\u00e9",  # NFC: e-with-acute as ONE composed codepoint (U+00E9)
        "session-cafe\u0301",  # NFD: plain "e" (U+0065) + COMBINING ACUTE ACCENT (U+0301)
    ],
    ids=["upper-a-diaeresis", "lower-a-diaeresis", "cafe-nfc", "cafe-nfd"],
)
def test_periodic_checkpoint_rejects_non_ascii_session_ids(tmp_path, bad, monkeypatch):
    """Atlas r1, checkpoint-v4 P1: rejecting uppercase ASCII closes the ASCII
    case class but not the general filesystem-key-injectivity contract.
    Measured on this macOS filesystem: upper/lower a-diaeresis case-fold to
    one inode, and the NFC/NFD spellings of the same rendered "cafe" name
    normalize to one inode -- writing the second changed the first in both
    pairs. "Two sessions never collide" requires an explicitly portable ASCII
    grammar (nothing left to case-fold or normalize), not a widening list of
    excluded Unicode classes. Written as \\uXXXX escapes rather than literal
    source characters: a literal NFD spelling here survived this very file's
    own on-disk write path silently renormalized to NFC, which would have
    made the intended discriminating pair into two duplicate cases -- found
    and fixed before freeze."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    with pytest.raises(ValueError):
        periodic_checkpoint_path(tmp_path, bad)


@pytest.mark.parametrize("bad", ["session_aaaa", "session aaaa", "session.aaaa", "session+aaaa"])
def test_periodic_checkpoint_session_id_grammar_is_ascii_lowercase_alnum_and_hyphen_only(
    tmp_path, bad, monkeypatch
):
    """Pins the accepted alphabet as exactly ASCII [a-z0-9-] rather than
    "not uppercase, not obviously non-ASCII": an implementation that merely
    special-cases the two collision classes above (uppercase, non-ASCII)
    without adopting an explicit allow-list would still accept these.

    Atlas r1, checkpoint-v5 P1: v5 dropped the `@pytest.mark.parametrize`
    decorator during a slice-and-splice edit (found while inserting the
    non-ASCII pair above), so this witness errored on a missing `bad`
    fixture before it ever called the validator -- a test that cannot run
    proves nothing. Restored."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    with pytest.raises(ValueError):
        periodic_checkpoint_path(tmp_path, bad)


def test_periodic_checkpoint_pins_session_id_length_to_a_255_byte_filename_component(
    tmp_path, monkeypatch
):
    """Atlas r1, checkpoint-v5 P1: the existing overlong witness uses 300
    characters, so an implementation accepting anything under 300 satisfies
    it while still raising ENAMETOOLONG at actual write time for ids between
    251 and 299 characters. Filename shape is `<session_id>.json` (matching
    the legacy `checkpoint.json` convention, and pinned explicitly here since
    no other test names it); the standard 255-byte NAME_MAX means the
    session-id component itself must be <= 250 ASCII bytes."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    at_limit = "a" * 250
    destination = periodic_checkpoint_path(tmp_path, at_limit)
    assert destination.name == f"{at_limit}.json"
    assert len(destination.name.encode("utf-8")) == 255

    with pytest.raises(ValueError):
        periodic_checkpoint_path(tmp_path, "a" * 251)


@pytest.mark.parametrize(
    "reserved",
    ["con", "prn", "aux", "nul", "com1", "com2", "com9", "lpt1", "lpt2", "lpt9"],
)
def test_periodic_checkpoint_rejects_windows_reserved_device_basenames(tmp_path, reserved, monkeypatch):
    """Atlas r1, checkpoint-v5 P1: CON, PRN, AUX, NUL, COM1-9, and LPT1-9 are
    reserved device names on Windows regardless of extension or case --
    "con.json" is unopenable/aliased there exactly like bare "con". The
    [a-z0-9-]+ grammar accepts these lowercase spellings outright since they
    contain no character it excludes. This schema claims cross-platform
    portability, so a session id that exactly names a reserved device is
    refused rather than silently producing an unusable path on that
    platform."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    with pytest.raises(ValueError):
        periodic_checkpoint_path(tmp_path, reserved)


def test_periodic_checkpoint_reserved_device_rejection_does_not_reject_substrings(tmp_path, monkeypatch):
    """Only the bare reserved name is unusable on Windows (Atlas's own
    adversarial probe on v5: "con"/"session-con" resolved to distinct,
    valid, non-colliding files on macOS) -- a session id that merely
    contains one as a substring is an ordinary, safe filename and must stay
    accepted."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    assert periodic_checkpoint_path(tmp_path, "session-con") is not None


def test_periodic_checkpoint_write_requires_session_id_in_payload(tmp_path, monkeypatch):
    # Interim per-test isolation (the shared process-environment fixture will supply this;
    # until it lands, these three lines are the established idiom this file
    # already uses elsewhere -- not a new mechanism). Ambient SYNAPT_RECALL_ROOT
    # or GRIPSPACE_ROOT makes checkpoint_path() IGNORE tmp_path entirely and
    # resolve to the real shared store; measured live in this session.
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_record("user", "hello"))
    payload = _payload(transcript, tmp_path)
    payload["session_id"] = ""

    with pytest.raises(ValueError):
        write_periodic_checkpoint(payload)


def test_periodic_checkpoint_write_never_shadows_a_different_sessions_checkpoint(tmp_path, monkeypatch):
    # Interim per-test isolation (the shared process-environment fixture will supply this;
    # until it lands, these three lines are the established idiom this file
    # already uses elsewhere -- not a new mechanism). Ambient SYNAPT_RECALL_ROOT
    # or GRIPSPACE_ROOT makes checkpoint_path() IGNORE tmp_path entirely and
    # resolve to the real shared store; measured live in this session.
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    transcript_a = tmp_path / "a.jsonl"
    transcript_a.write_bytes(_record("user", "session A's last words"))
    payload_a = _payload(transcript_a, tmp_path)
    payload_a["session_id"] = "session-aaaa"

    transcript_b = tmp_path / "b.jsonl"
    transcript_b.write_bytes(_record("user", "session B's last words"))
    payload_b = _payload(transcript_b, tmp_path)
    payload_b["session_id"] = "session-bbbb"

    write_periodic_checkpoint(payload_a)
    write_periodic_checkpoint(payload_b)

    recovered_a = read_periodic_checkpoint(tmp_path, "session-aaaa")
    recovered_b = read_periodic_checkpoint(tmp_path, "session-bbbb")

    assert recovered_a["last_user_text"] == "session A's last words"
    assert recovered_b["last_user_text"] == "session B's last words"


def test_periodic_checkpoint_second_write_replaces_its_own_sessions_prior_checkpoint(tmp_path, monkeypatch):
    # Interim per-test isolation (the shared process-environment fixture will supply this;
    # until it lands, these three lines are the established idiom this file
    # already uses elsewhere -- not a new mechanism). Ambient SYNAPT_RECALL_ROOT
    # or GRIPSPACE_ROOT makes checkpoint_path() IGNORE tmp_path entirely and
    # resolve to the real shared store; measured live in this session.
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    transcript = tmp_path / "session.jsonl"
    session_id = "session-cccc"

    transcript.write_bytes(_record("user", "first tick"))
    payload = _payload(transcript, tmp_path)
    payload["session_id"] = session_id
    write_periodic_checkpoint(payload)

    transcript.write_bytes(_record("user", "second tick"))
    write_periodic_checkpoint(payload)

    files = list(periodic_checkpoint_path(tmp_path, session_id).parent.glob(f"{session_id}*"))
    recovered = read_periodic_checkpoint(tmp_path, session_id)

    assert len(files) == 1
    assert recovered["last_user_text"] == "second tick"


# ---------------------------------------------------------------------------
# Four discriminators a minimal, non-atomic, unbounded, cross-session-blind
# implementation would still pass the tests above. Required before this spec
# freezes -- a reviewer's mutation run confirmed the tests up to this point
# are satisfiable by exactly that shape.
# ---------------------------------------------------------------------------


def test_periodic_checkpoint_atomic_replace_failure_preserves_prior_same_session_checkpoint(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    session_id = "session-mmmm"
    transcript = tmp_path / "session.jsonl"

    transcript.write_bytes(_record("user", "first tick, must survive"))
    payload = _payload(transcript, tmp_path)
    payload["session_id"] = session_id
    write_periodic_checkpoint(payload)

    transcript.write_bytes(_record("user", "second tick, must not land"))
    with patch("synapt.checkpoint.os.replace", side_effect=OSError("unwritable")):
        with pytest.raises(OSError, match="unwritable"):
            write_periodic_checkpoint(payload)

    recovered = read_periodic_checkpoint(tmp_path, session_id)
    destination = periodic_checkpoint_path(tmp_path, session_id)
    leftover_tmp = list(destination.parent.glob(".checkpoint.*.tmp"))

    assert recovered["last_user_text"] == "first tick, must survive"
    assert leftover_tmp == []


def test_periodic_checkpoint_write_fsyncs_before_replace(tmp_path, monkeypatch):
    """Atlas r1, checkpoint-v3 P1: a complete periodic write using temp write,
    flush, and os.replace -- but no os.fsync -- passed all 22 prior periodic
    tests. The legacy SessionEnd path fsyncs before replace (checkpoint.py:437);
    periodic exists specifically to survive an unclean stop mid-session, so
    the same durability seam is a requirement, not an implementation detail.
    """
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    session_id = "session-fsync"
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_record("user", "must be durable"))
    payload = _payload(transcript, tmp_path)
    payload["session_id"] = session_id

    calls: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def _tracked_fsync(fd):
        calls.append("fsync")
        return real_fsync(fd)

    def _tracked_replace(src, dst):
        calls.append("replace")
        return real_replace(src, dst)

    with patch("synapt.checkpoint.os.fsync", side_effect=_tracked_fsync) as mock_fsync, \
         patch("synapt.checkpoint.os.replace", side_effect=_tracked_replace):
        write_periodic_checkpoint(payload)

    assert mock_fsync.called, "periodic write must fsync, same durability seam as write_checkpoint"
    assert calls == ["fsync", "replace"], f"fsync must complete before replace, got {calls}"


def test_periodic_checkpoint_realistic_large_transcript_stays_within_bound(tmp_path, monkeypatch):
    """Regression coverage over a realistic large input. NOT a proof of write-side
    enforcement -- see the note on the two tests below. capture_checkpoint's own
    per-field caps (PATH_BYTES=1024 x FILES_LIMIT=32 with escape inflation, plus
    two TEXT_BYTES=4096 text fields) top out well under half of CHECKPOINT_BYTES
    for any input this fixture shape can produce, so a writer with NO bound
    enforcement at all would still pass this one. Kept as realistic-input coverage;
    the mutation-proof witnesses are below.
    """
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    session_id = "session-nnnn"
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
    payload = _payload(transcript, tmp_path)
    payload["session_id"] = session_id

    write_periodic_checkpoint(payload)
    destination = periodic_checkpoint_path(tmp_path, session_id)

    assert destination.stat().st_size <= CHECKPOINT_BYTES
    assert read_periodic_checkpoint(tmp_path, session_id) is not None


def test_periodic_checkpoint_reader_rejects_oversized_file_regardless_of_writer(tmp_path, monkeypatch):
    """Read-side bound, independent of what any writer would ever produce: a file
    that somehow grew past the limit on disk must still be refused."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    session_id = "session-nnnn"
    destination = periodic_checkpoint_path(tmp_path, session_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"x" * (CHECKPOINT_BYTES + 1))

    assert read_periodic_checkpoint(tmp_path, session_id) is None


# ---------------------------------------------------------------------------
# Mutation-proof write-bound witnesses (Sentinel r2, v2 review, P1). A realistic
# transcript-derived checkpoint cannot exceed CHECKPOINT_BYTES under
# capture_checkpoint's own field caps, which means the two tests above are
# satisfiable by a writer with NO byte-bound enforcement at all -- the write
# path's own truncate-or-refuse contract needs a fixture that is oversized
# BEFORE the write function's bound check runs, independent of what a real
# transcript could produce. Patching capture_checkpoint's return value is the
# established idiom this file already uses for isolating one step of the
# pipeline (e.g. patching checkpoint_path to isolate path resolution from
# capture logic).
# ---------------------------------------------------------------------------


def _oversized_but_trimmable_checkpoint(tmp_path, session_id: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "session-end",  # capture_checkpoint always stamps this; the
                                   # writer under test must relabel it "periodic"
        "captured_at": "2026-01-01T00:00:00Z",
        "runtime": "codex",
        "session_id": session_id,
        "reason": "other",
        "cwd": str(tmp_path),
        "transcript_path": "irrelevant.jsonl",
        "hook_event_name": "SessionEnd",
        "last_user_text": "u" * 3000,
        "last_assistant_text": "a" * 3000,
        # Deliberately far past FILES_LIMIT/PATH_BYTES if the writer imposed
        # no bound at all: serializes to well over CHECKPOINT_BYTES on its own.
        "files_touched": [f"/deliberately/oversized/path/{i}" * 50 for i in range(2000)],
        "tail_bytes_read": 0,
        "transcript_size": 0,
        "truncated": False,
        "parse_status": "ok",
    }


def test_periodic_checkpoint_write_trims_an_oversized_but_trimmable_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    session_id = "session-oversized-a"
    transcript = tmp_path / "irrelevant.jsonl"
    transcript.write_bytes(b"")
    payload = _payload(transcript, tmp_path)
    payload["session_id"] = session_id
    oversized = _oversized_but_trimmable_checkpoint(tmp_path, session_id)

    # A no-bound "minimal survivor" writer -- capture, stamp source, encode,
    # write, no truncate-or-refuse step -- would write this straight through
    # and produce a file far past CHECKPOINT_BYTES. Assert the actual writer
    # does not.
    with patch("synapt.checkpoint.capture_checkpoint", return_value=oversized):
        write_periodic_checkpoint(payload)

    destination = periodic_checkpoint_path(tmp_path, session_id)
    recovered = read_periodic_checkpoint(tmp_path, session_id)

    assert destination.stat().st_size <= CHECKPOINT_BYTES
    assert recovered is not None
    assert recovered["source"] == "periodic"


def test_periodic_checkpoint_write_refuses_an_untrimmable_checkpoint(tmp_path, monkeypatch):
    """Text fields alone past the ceiling, files_touched already empty: there is
    nothing left to trim, so the writer must refuse rather than write a
    checkpoint the reader will reject anyway (same policy as write_checkpoint's
    existing SessionEnd path -- see test_atomic_replace_failure_preserves_
    previous_checkpoint's sibling, the legacy CHECKPOINT_BYTES ValueError)."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    session_id = "session-oversized-b"
    transcript = tmp_path / "irrelevant.jsonl"
    transcript.write_bytes(b"")
    payload = _payload(transcript, tmp_path)
    payload["session_id"] = session_id
    untrimmable = _oversized_but_trimmable_checkpoint(tmp_path, session_id)
    untrimmable["files_touched"] = []
    untrimmable["last_user_text"] = "u" * (CHECKPOINT_BYTES + 1000)

    with patch("synapt.checkpoint.capture_checkpoint", return_value=untrimmable):
        with pytest.raises(ValueError, match=str(CHECKPOINT_BYTES)):
            write_periodic_checkpoint(payload)

    assert read_periodic_checkpoint(tmp_path, session_id) is None
    assert list(periodic_checkpoint_path(tmp_path, session_id).parent.glob(".checkpoint.*.tmp")) == []


def test_periodic_checkpoint_reader_rejects_session_id_mismatch_between_filename_and_content(
    tmp_path, monkeypatch
):
    """A file at session A's path whose own content claims session B is not A's checkpoint."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    destination = periodic_checkpoint_path(tmp_path, "session-oooo")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "source": "periodic",
        "captured_at": "2026-01-01T00:00:00Z",
        "session_id": "session-pppp",
        "last_user_text": "must not surface under the wrong session",
    }), encoding="utf-8")

    assert read_periodic_checkpoint(tmp_path, "session-oooo") is None


def test_periodic_checkpoint_reader_rejects_a_sessionend_labelled_file(tmp_path, monkeypatch):
    """A file that looks like a SessionEnd checkpoint is not a periodic one, wherever it sits."""
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    session_id = "session-qqqq"
    destination = periodic_checkpoint_path(tmp_path, session_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "source": "session-end",
        "captured_at": "2026-01-01T00:00:00Z",
        "session_id": session_id,
        "last_user_text": "must not surface as a periodic checkpoint",
    }), encoding="utf-8")

    assert read_periodic_checkpoint(tmp_path, session_id) is None


def test_periodic_checkpoint_is_labelled_periodic_not_session_end(tmp_path, monkeypatch):
    # Interim per-test isolation (the shared process-environment fixture will supply this;
    # until it lands, these three lines are the established idiom this file
    # already uses elsewhere -- not a new mechanism). Ambient SYNAPT_RECALL_ROOT
    # or GRIPSPACE_ROOT makes checkpoint_path() IGNORE tmp_path entirely and
    # resolve to the real shared store; measured live in this session.
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_record("user", "hi"))
    payload = _payload(transcript, tmp_path)
    payload["session_id"] = "session-dddd"

    write_periodic_checkpoint(payload)
    recovered = read_periodic_checkpoint(tmp_path, "session-dddd")

    assert recovered["source"] == "periodic"


@pytest.mark.parametrize(
    "schema_version", [None, False, True, 0, 1.0, 2, "1"],
)
def test_periodic_checkpoint_reader_rejects_wrong_schema_domain(tmp_path, schema_version, monkeypatch):
    # Interim per-test isolation (the shared process-environment fixture will supply this;
    # until it lands, these three lines are the established idiom this file
    # already uses elsewhere -- not a new mechanism). Ambient SYNAPT_RECALL_ROOT
    # or GRIPSPACE_ROOT makes checkpoint_path() IGNORE tmp_path entirely and
    # resolve to the real shared store; measured live in this session.
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    session_id = "session-eeee"
    destination = periodic_checkpoint_path(tmp_path, session_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Otherwise a fully valid periodic record (Atlas r1, checkpoint-v3 P1): a
    # fixture missing source/session_id gets rejected by cross-field binding
    # regardless of whether schema_version is checked at all, so a reader that
    # ignores schema_version entirely still passed all 7 parameters here.
    destination.write_text(json.dumps({
        "schema_version": schema_version,
        "source": "periodic",
        "captured_at": "2026-01-01T00:00:00Z",
        "session_id": session_id,
        "last_user_text": "must not surface",
    }), encoding="utf-8")

    assert read_periodic_checkpoint(tmp_path, session_id) is None


def test_periodic_checkpoint_reader_returns_none_when_nothing_written_yet(tmp_path, monkeypatch):
    # Interim per-test isolation (the shared process-environment fixture will supply this;
    # until it lands, these three lines are the established idiom this file
    # already uses elsewhere -- not a new mechanism). Ambient SYNAPT_RECALL_ROOT
    # or GRIPSPACE_ROOT makes checkpoint_path() IGNORE tmp_path entirely and
    # resolve to the real shared store; measured live in this session.
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    assert read_periodic_checkpoint(tmp_path, "session-never-written") is None


def test_legacy_sessionend_checkpoint_is_unaffected_by_periodic_checkpoints_existing(tmp_path, monkeypatch):
    # Interim per-test isolation (the shared process-environment fixture will supply this;
    # until it lands, these three lines are the established idiom this file
    # already uses elsewhere -- not a new mechanism). Ambient SYNAPT_RECALL_ROOT
    # or GRIPSPACE_ROOT makes checkpoint_path() IGNORE tmp_path entirely and
    # resolve to the real shared store; measured live in this session.
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)
    """Regression: writing periodic checkpoints must not change the SessionEnd path."""
    transcript = tmp_path / "periodic.jsonl"
    transcript.write_bytes(_record("user", "periodic write"))
    periodic_payload = _payload(transcript, tmp_path)
    periodic_payload["session_id"] = "session-ffff"
    write_periodic_checkpoint(periodic_payload)

    end_transcript = tmp_path / "session-end.jsonl"
    end_transcript.write_bytes(_record("user", "session-end write"))
    write_checkpoint(_payload(end_transcript, tmp_path))

    recovered = read_checkpoint(tmp_path)
    assert recovered["last_user_text"] == "session-end write"
    assert recovered["source"] == "session-end"


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
