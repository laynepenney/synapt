"""TDD spec: `synapt recall build` must be idempotent.

Contract under test
-------------------
A second consecutive build over an unchanged store must do ~zero work, run
fast, and SAY that it skipped. Today it does not: measured on a frozen
4-transcript corpus (39MB, 1,709 chunks) with mtimes pinned, a run that parsed
ZERO files still took 40.9s. The build has exactly one change-detector
(``core.build_index``, mtime+size) and it guards the cheapest stage; every
expensive stage downstream is unconditional.

Why the controls in here are not optional
-----------------------------------------
The cheapest way to make every "did it skip?" assertion pass is to skip
unconditionally, which would turn a slow build into a broken one. So each skip
assertion is paired with a NEGATIVE control that changes exactly one input and
demands the build notice. A suite that only proves skipping is a suite that
cannot fail the worst available implementation.

That pairing is not hypothetical caution — the harness that produced the
measurements above shipped a bad control first: it touched an mtime against a
freshness check keyed on SIZE, so it was aimed at a quantity the code does not
read, and it "passed" while proving nothing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from conftest import assistant_entry, user_text_entry, write_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transcript(path: Path, *, turns: int = 2, prefix: str = "q") -> Path:
    """Write a small but genuinely parseable Claude Code transcript."""
    entries = []
    for i in range(turns):
        entries.append(
            user_text_entry(
                f"{prefix} question {i}",
                uuid=f"{prefix}-u{i}",
                ts=f"2026-03-01T10:{i:02d}:00Z",
            )
        )
        entries.append(
            assistant_entry(
                text=f"{prefix} answer {i}",
                uuid=f"{prefix}-a{i}",
                ts=f"2026-03-01T10:{i:02d}:30Z",
            )
        )
    write_jsonl(path, entries)
    return path


def _set_mtime(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def _manifest_entry(path: Path, *, source_dir: Path | None = None) -> dict:
    st = path.stat()
    entry = {"name": path.name, "mtime": st.st_mtime, "size": st.st_size}
    if source_dir is not None:
        entry["dir"] = source_dir.name
    return entry


# ===========================================================================
# A. Parse-skip keying: the detector must be scoped by SOURCE DIR
# ===========================================================================
#
# cli.py writes ONE flat `source_files` list spanning every worktree archive
# dir, while core.build_index keys `already_indexed` on the BASENAME alone, so
# same-named files from different worktrees overwrite each other and the loser
# can never match. Verified on the live manifest: 18 entries, 14 distinct
# basenames, 4 Codex rollout files shadowed (identical size, mtime differing by
# ~107s) and re-parsing on every incremental build, permanently.

def test_same_basename_in_two_source_dirs_both_skip(tmp_path):
    """Two worktrees archiving the same session name must BOTH skip.

    This is the live defect in miniature: identical size, differing mtime.
    """
    from synapt.recall.core import build_index

    dir_a = tmp_path / "worktree-a"
    dir_b = tmp_path / "worktree-b"
    dir_a.mkdir()
    dir_b.mkdir()

    name = "rollout-2026-03-20T15-11-29-019d0ca8.jsonl"
    file_a = _transcript(dir_a / name, prefix="a")
    file_b = _transcript(dir_b / name, prefix="a")  # same bytes, same size
    assert file_a.stat().st_size == file_b.stat().st_size, "fixture must collide on size"

    # Distinct mtimes, exactly like the live manifest's ~107s spread.
    _set_mtime(file_a, 1785988016.0)
    _set_mtime(file_b, 1785988123.0)

    manifest = {
        "source_files": [
            _manifest_entry(file_a, source_dir=dir_a),
            _manifest_entry(file_b, source_dir=dir_b),
        ]
    }

    idx_a = build_index(dir_a, incremental_manifest=manifest)
    idx_b = build_index(dir_b, incremental_manifest=manifest)

    assert idx_a.chunks == [], f"dir_a re-parsed despite an exact manifest match: {len(idx_a.chunks)} chunks"
    assert idx_b.chunks == [], f"dir_b re-parsed despite an exact manifest match: {len(idx_b.chunks)} chunks"


def test_changed_file_still_reparses_under_dir_scoped_keys(tmp_path):
    """NEGATIVE CONTROL for the test above.

    Dir-scoping must not be implemented by making everything match. Change one
    file's content and it must come back.
    """
    from synapt.recall.core import build_index

    dir_a = tmp_path / "worktree-a"
    dir_b = tmp_path / "worktree-b"
    dir_a.mkdir()
    dir_b.mkdir()

    name = "shared-session.jsonl"
    file_a = _transcript(dir_a / name, prefix="a")
    file_b = _transcript(dir_b / name, prefix="a")
    _set_mtime(file_a, 1785988016.0)
    _set_mtime(file_b, 1785988123.0)

    manifest = {
        "source_files": [
            _manifest_entry(file_a, source_dir=dir_a),
            _manifest_entry(file_b, source_dir=dir_b),
        ]
    }

    # Grow dir_a's copy only.
    _transcript(file_a, turns=5, prefix="a")

    idx_a = build_index(dir_a, incremental_manifest=manifest)
    idx_b = build_index(dir_b, incremental_manifest=manifest)

    assert idx_a.chunks, "dir_a changed and must be re-parsed"
    assert idx_b.chunks == [], "dir_b did not change and must still skip"


def test_legacy_manifest_without_dir_field_still_skips(tmp_path):
    """Backward compatibility: manifests written before dir-scoping.

    An existing store's manifest has no `dir` key. Upgrading must not force a
    one-time full rebuild of every project in the wild.
    """
    from synapt.recall.core import build_index

    src = tmp_path / "archive"
    src.mkdir()
    f = _transcript(src / "legacy-session.jsonl")
    _set_mtime(f, 1785988016.0)

    legacy = {"source_files": [{"name": f.name, "mtime": f.stat().st_mtime, "size": f.stat().st_size}]}

    idx = build_index(src, incremental_manifest=legacy)
    assert idx.chunks == [], "a legacy (dir-less) manifest entry must still be honoured"


# ===========================================================================
# B. Archive freshness: size OR mtime
# ===========================================================================
#
# archive.py:704 skips when `src_size == dst_size`, so a same-size content edit
# is never re-archived and therefore never re-indexed.

def test_archive_refreshes_on_same_size_newer_mtime(tmp_path):
    from synapt.recall.archive import archive_transcripts
    from synapt.recall.core import project_archive_dir

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    f = source / "session.jsonl"
    f.write_text('{"a": "0000000000"}\n')
    archive_transcripts(project, source)

    archived = project_archive_dir(project) / "session.jsonl"
    assert archived.exists(), "fixture failed: first archive did not land"
    original_size = archived.stat().st_size

    # Same byte count, different content, newer mtime.
    f.write_text('{"a": "1111111111"}\n')
    assert f.stat().st_size == original_size, "fixture must hold size constant"
    _set_mtime(f, time.time() + 10)

    archive_transcripts(project, source)
    assert archived.read_text() == f.read_text(), (
        "same-size content edit was never re-archived, so it can never be re-indexed"
    )


def test_archive_skips_when_size_and_mtime_both_match(tmp_path):
    """NEGATIVE CONTROL: freshness must not degrade into copy-always."""
    from synapt.recall.archive import archive_transcripts

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _transcript(source / "session.jsonl")

    archive_transcripts(project, source)
    second = archive_transcripts(project, source)
    assert second == [], f"unchanged source was re-archived: {second}"


def test_archive_still_preserves_larger_archive_when_source_shrinks(tmp_path):
    """Regression guard on existing behavior.

    A `/clear` truncates the live transcript. The archive holds the longer
    history and must not be overwritten by the shorter one.
    """
    from synapt.recall.archive import archive_transcripts
    from synapt.recall.core import project_archive_dir

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    f = _transcript(source / "session.jsonl", turns=6)
    archive_transcripts(project, source)
    archived = project_archive_dir(project) / "session.jsonl"
    long_size = archived.stat().st_size

    _transcript(f, turns=1)  # truncated by /clear
    _set_mtime(f, time.time() + 10)
    archive_transcripts(project, source)

    assert archived.stat().st_size == long_size, "shorter source overwrote a longer archive"


# ===========================================================================
# C. The nothing-changed signal
# ===========================================================================
#
# `incremental` currently gates exactly one call site. A real no-op needs a
# signal covering EVERY build input: transcripts, channels and journals.

def test_identical_inputs_are_a_noop(tmp_path):
    from synapt.recall.build_delta import compute_input_signature, is_noop

    src = tmp_path / "archive"
    src.mkdir()
    _transcript(src / "s1.jsonl")
    channels = tmp_path / "channels"
    channels.mkdir()
    (channels / "dev.jsonl").write_text('{"id":"m1","body":"hello"}\n')
    journal = tmp_path / "journal.jsonl"
    journal.write_text('{"session_id":"s1","focus":"x"}\n')

    first = compute_input_signature([src], channels, [journal])
    second = compute_input_signature([src], channels, [journal])

    assert is_noop(first, second) is True


def test_no_previous_signature_is_never_a_noop(tmp_path):
    """A first build, or a build after a corrupt manifest, must do the work."""
    from synapt.recall.build_delta import compute_input_signature, is_noop

    src = tmp_path / "archive"
    src.mkdir()
    _transcript(src / "s1.jsonl")

    assert is_noop(None, compute_input_signature([src], None, [])) is False


@pytest.mark.parametrize("mutate", ["transcript", "channel", "journal"])
def test_any_changed_input_defeats_the_noop(tmp_path, mutate):
    """NEGATIVE CONTROLS, one per input class.

    These are the tests that stop "make it fast" from becoming "make it lie".
    A signal that only watches transcripts would let a new channel message or a
    fresh journal entry go unindexed while the build cheerfully reports it is
    up to date — which is worse than being slow, because it is silent.
    """
    from synapt.recall.build_delta import compute_input_signature, is_noop

    src = tmp_path / "archive"
    src.mkdir()
    t = _transcript(src / "s1.jsonl")
    channels = tmp_path / "channels"
    channels.mkdir()
    ch = channels / "dev.jsonl"
    ch.write_text('{"id":"m1","body":"hello"}\n')
    journal = tmp_path / "journal.jsonl"
    journal.write_text('{"session_id":"s1","focus":"x"}\n')

    before = compute_input_signature([src], channels, [journal])

    if mutate == "transcript":
        _transcript(t, turns=5)
    elif mutate == "channel":
        with ch.open("a") as fh:
            fh.write('{"id":"m2","body":"a new message"}\n')
    else:
        with journal.open("a") as fh:
            fh.write('{"session_id":"s2","focus":"y"}\n')

    after = compute_input_signature([src], channels, [journal])
    assert is_noop(before, after) is False, f"a changed {mutate} must defeat the no-op"


def test_new_file_appearing_defeats_the_noop(tmp_path):
    """A signature that only hashes known files misses arrivals."""
    from synapt.recall.build_delta import compute_input_signature, is_noop

    src = tmp_path / "archive"
    src.mkdir()
    _transcript(src / "s1.jsonl")
    before = compute_input_signature([src], None, [])

    _transcript(src / "s2.jsonl", prefix="b")
    after = compute_input_signature([src], None, [])

    assert is_noop(before, after) is False, "a newly arrived transcript must defeat the no-op"


def test_signature_survives_a_manifest_round_trip(tmp_path):
    """The signal is only useful if it persists between processes."""
    from synapt.recall.build_delta import (
        compute_input_signature,
        is_noop,
        signature_from_manifest,
        signature_to_manifest,
    )

    src = tmp_path / "archive"
    src.mkdir()
    _transcript(src / "s1.jsonl")
    sig = compute_input_signature([src], None, [])

    # Through JSON, because that is how it reaches SQLite metadata.
    restored = signature_from_manifest(json.loads(json.dumps(signature_to_manifest(sig))))

    assert restored is not None
    assert is_noop(restored, compute_input_signature([src], None, [])) is True


def test_signature_absent_from_manifest_returns_none(tmp_path):
    from synapt.recall.build_delta import signature_from_manifest

    assert signature_from_manifest({"source_files": []}) is None


@pytest.mark.parametrize(
    "label,payload",
    [
        # `bool` subclasses `int`, so an isinstance guard admits both, and the
        # dataclass field declared `int` ends up holding True.
        ("files is True", {"version": 1, "digest": "a" * 64, "files": True}),
        ("files is False", {"version": 1, "digest": "a" * 64, "files": False}),
        # Nothing bounded the sign. A negative count is not a count.
        ("files is -1", {"version": 1, "digest": "a" * 64, "files": -1}),
        ("files is a float", {"version": 1, "digest": "a" * 64, "files": 3.0}),
        ("files is a digit string", {"version": 1, "digest": "a" * 64, "files": "3"}),
        # A digest is what sha256().hexdigest() produces, not any string.
        ("digest is too short", {"version": 1, "digest": "x", "files": 3}),
        ("digest is non-hex", {"version": 1, "digest": "z" * 64, "files": 3}),
        ("digest is uppercase", {"version": 1, "digest": "A" * 64, "files": 3}),
        ("digest is 63 chars", {"version": 1, "digest": "a" * 63, "files": 3}),
        ("digest is 65 chars", {"version": 1, "digest": "a" * 65, "files": 3}),
        ("digest is padded", {"version": 1, "digest": " " + "a" * 64, "files": 3}),
        # Found by closing the CLASS rather than the three reported instances:
        # the version guard is a bare `!=`, and True == 1 and 1.0 == 1.
        ("version is True", {"version": True, "digest": "a" * 64, "files": 3}),
        ("version is 1.0", {"version": 1.0, "digest": "a" * 64, "files": 3}),
    ],
)
def test_a_malformed_signature_payload_resolves_to_WORK(label, payload):
    """Rejecting is only half of it: the refusal must reach a BUILD.

    `signature_from_manifest` returning None is an internal fact. What the
    module promises is that no usable prior state means the work happens, so
    each case asserts BOTH halves -- the parse refuses, AND `is_noop` turns
    that refusal into False. A guard that returned None while some caller
    treated None as "unchanged" would satisfy the first assertion and lose the
    property the first assertion exists to protect.

    Written after the original validator accepted every payload here while its
    own docstring said it rejected malformed ones.
    """
    from synapt.recall.build_delta import (
        InputSignature,
        is_noop,
        signature_from_manifest,
    )

    recovered = signature_from_manifest({"input_signature": payload})
    assert recovered is None, f"malformed payload was accepted: {label}"

    current = InputSignature(digest="b" * 64, file_count=5)
    assert is_noop(recovered, current) is False, (
        f"refusal did not resolve to work: {label}"
    )


def test_a_well_formed_payload_is_still_accepted():
    """The control for the rejection cases above.

    Thirteen tests asserting None would all pass against a validator that
    rejects everything, including real signatures -- a guard that refuses
    universally is exactly as broken as one that accepts universally, and it
    fails in the direction that looks safe. This is the case that would go red
    if the new checks were tightened past correctness.
    """
    from synapt.recall.build_delta import (
        InputSignature,
        is_noop,
        signature_from_manifest,
    )

    good = {"version": 1, "digest": "a" * 64, "files": 3}
    recovered = signature_from_manifest({"input_signature": good})

    assert recovered is not None, "a well-formed payload must survive"
    assert recovered.digest == "a" * 64
    assert recovered.file_count == 3
    assert type(recovered.file_count) is int
    assert is_noop(recovered, InputSignature(digest="a" * 64, file_count=3)) is True


# ===========================================================================
# D. The build must never grind LLM summaries
# ===========================================================================
#
# `upgrade_large_cluster_summaries` is an unconditional backlog grinder:
# max_upgrades=5 per build, loading flan-t5-base and making ~11 huggingface.co
# round-trips every run. Measured at 37.7s of a 40.9s do-nothing build (92%).
# On a real 11.6k-chunk store measured during this work, 970 clusters were
# pending — 194 further builds just to drain the backlog, while newly arriving
# chunks refill it. It belongs on an explicit maintenance command, not on the
# build path.

def test_build_never_calls_the_summary_grinder(tmp_path, monkeypatch):
    import synapt.recall.clustering as clustering
    from synapt.recall.cli import _archive_and_build

    calls: list[dict] = []

    def _spy(db, min_chunks=5, max_upgrades=5):
        calls.append({"min_chunks": min_chunks, "max_upgrades": max_upgrades})
        return 0

    monkeypatch.setattr(clustering, "upgrade_large_cluster_summaries", _spy)

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _transcript(source / "s1.jsonl", turns=8)

    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    assert calls == [], f"build invoked the summary grinder: {calls}"


# ===========================================================================
# E. `synapt maintain` — the grinder's new, explicit home
# ===========================================================================

def test_maintain_subcommand_is_registered():
    """`make_parser()` does not exist yet — the parser is built inline inside
    `main()`, which is why no CLI default is currently testable at all.
    Extracting it is part of this change, not incidental refactoring."""
    from synapt.recall.cli import make_parser

    actions = [a for a in make_parser()._actions if hasattr(a, "choices") and a.choices]
    commands = set()
    for a in actions:
        commands.update(a.choices or {})
    assert "maintain" in commands, f"no `maintain` subcommand; got {sorted(commands)}"


def test_maintain_parser_accepts_limit():
    from synapt.recall.cli import make_parser

    args = make_parser().parse_args(["maintain", "--limit", "7"])
    assert args.command == "maintain"
    assert args.limit == 7


def test_maintain_has_a_default_limit():
    from synapt.recall.cli import make_parser

    args = make_parser().parse_args(["maintain"])
    assert isinstance(args.limit, int) and args.limit > 0, (
        "maintain must be bounded by default; an unbounded grind is what we just removed"
    )


def test_maintain_passes_the_limit_through_and_reports_backlog(tmp_path, monkeypatch, capsys):
    """The backlog must stay visible. Draining it silently would be the
    regression this change is meant to avoid."""
    import synapt.recall.clustering as clustering
    from synapt.recall.cli import cmd_maintain, make_parser

    seen: list[int] = []

    def _spy(db, min_chunks=5, max_upgrades=5):
        seen.append(max_upgrades)
        return 3

    monkeypatch.setattr(clustering, "upgrade_large_cluster_summaries", _spy)
    monkeypatch.chdir(tmp_path)

    project = tmp_path
    source = tmp_path / "source"
    source.mkdir()
    _transcript(source / "s1.jsonl", turns=8)

    from synapt.recall.cli import _archive_and_build
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=False)

    args = make_parser().parse_args(["maintain", "--limit", "3"])
    cmd_maintain(args)

    out = capsys.readouterr().out.lower()
    assert seen == [3], f"maintain did not pass --limit through: {seen}"
    assert "remaining" in out, "maintain must report the remaining backlog, not drain it silently"


# ===========================================================================
# F. A no-op run must SAY it is a no-op
# ===========================================================================
#
# A fast build and a broken build look identical from the outside unless the
# build states which stages it skipped and why.

def test_second_build_reports_that_it_skipped(tmp_path, capsys):
    from synapt.recall.cli import _archive_and_build

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _transcript(source / "s1.jsonl", turns=8)

    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    capsys.readouterr()

    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    out = capsys.readouterr().out.lower()

    assert "up to date" in out, f"a no-op build said nothing about being a no-op:\n{out}"
    assert "skip" in out, f"a no-op build did not report skipped stages:\n{out}"


def test_second_build_after_a_change_does_not_claim_to_be_up_to_date(tmp_path, capsys):
    """NEGATIVE CONTROL for the message itself.

    The skip line is load-bearing: an operator reads it and stops looking. It
    must never appear on a run that had work to do.
    """
    from synapt.recall.cli import _archive_and_build

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    t = _transcript(source / "s1.jsonl", turns=8)

    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    capsys.readouterr()

    _transcript(t, turns=20)
    _set_mtime(t, time.time() + 10)

    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    out = capsys.readouterr().out.lower()

    assert "up to date" not in out, f"build claimed 'up to date' after real work:\n{out}"


def test_noop_build_still_returns_a_usable_index(tmp_path):
    """Skipping stages must not degrade the return contract.

    Callers (MCP recall_build, the hooks, setup) read `.stats()` off the
    result. A no-op that returns None would turn a fast path into a crash.
    """
    from synapt.recall.cli import _archive_and_build

    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    _transcript(source / "s1.jsonl", turns=8)

    first = _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    second = _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)

    assert second is not None, "no-op build returned None; callers dereference this"
    assert second.stats()["chunk_count"] == first.stats()["chunk_count"]


# ===========================================================================
# G. Default flip — incremental by default, --full the explicit opt-in
# ===========================================================================
#
# Lands as its own commit. The CLI defaults `--incremental` to False while MCP
# recall_build already defaults it True: same operation, opposite defaults
# depending on the surface you reach through.

def test_build_defaults_to_incremental():
    from synapt.recall.cli import make_parser

    args = make_parser().parse_args(["build"])
    assert getattr(args, "full", False) is False
    assert args.incremental is True, "bare `synapt build` must not rewrite everything"


def test_full_flag_forces_a_full_rebuild():
    from synapt.recall.cli import make_parser

    args = make_parser().parse_args(["build", "--full"])
    assert args.incremental is False, "--full must be the explicit opt-out from incremental"


def test_incremental_flag_still_accepted_for_compatibility():
    """Scripts and hooks in the wild pass --incremental explicitly."""
    from synapt.recall.cli import make_parser

    args = make_parser().parse_args(["build", "--incremental"])
    assert args.incremental is True


def test_cli_and_mcp_defaults_agree():
    """The divergence that made this defect hard to see from either side."""
    import inspect

    from synapt.recall.cli import make_parser
    from synapt.recall.server import recall_build

    cli_default = make_parser().parse_args(["build"]).incremental
    mcp_default = inspect.signature(recall_build).parameters["incremental"].default
    assert cli_default == mcp_default, (
        f"CLI default ({cli_default}) and MCP default ({mcp_default}) disagree"
    )
