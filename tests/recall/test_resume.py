"""Tests for synapt.recall.resume — the session-tail surface behind `synapt resume`.

recall#927. The feature answers "what were the last things the previous session
did and intended?" after an unclean stop, so nearly every test here is about a
*silent* failure: an answer that is confidently wrong reads exactly like an
answer that is right.

Three hazards get first-class witnesses rather than footnotes:

1. ``TranscriptIndex.load`` returns HEADERS ONLY. Code that reads ``user_text``
   straight off ``index.sessions[sid]`` gets empty strings and reports "no
   meaningful turns" without erroring. ``TestLazyHydration`` proves the hazard
   is real (control) before proving the code avoids it.
2. The noise filter can delete the single most load-bearing turn in the output —
   a final user message with no reply, which is what a dropped baton looks like.
   ``TestHarnessNoiseDiscriminator`` pins both halves of the conjunction that
   prevents it.
3. Journal binding can pair one session's tail with another session's intent.
   ``TestJournalBinding`` pins the three provenance states separately.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from _isolation_helpers import owned_store
from argparse import Namespace
from pathlib import Path
from unittest import mock

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.journal import JournalEntry, append_entry
from synapt.recall.resume import (
    CallerTranscript,
    UncleanEnd,
    detect_unclean_end,
    ResumeError,
    build_resume_view,
    caller_transcripts,
    format_resume,
    is_harness_authored,
    load_resume_index,
    resolve_session,
    _latest_event_timestamp,
    _source_label,
    _timestamp_epoch,
)

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
SESSION_B = "bbbbbbbb-5555-6666-7777-888888888888"

CONTINUATION_PREAMBLE = (
    "This session is being continued from a previous conversation that ran out "
    "of context. The summary below covers the earlier portion."
)


def _chunk(
    session_id: str,
    turn_index: int,
    user_text: str = "",
    assistant_text: str = "",
    tools_used: list[str] | None = None,
    timestamp: str = "2026-08-05T10:00:00Z",
    tool_content: str = "",
    transcript_path: str = "",
    agent_id: str | None = None,
) -> TranscriptChunk:
    """Build a chunk the way the parsers do (short-id prefix, ``:t<n>`` suffix)."""
    return TranscriptChunk(
        id=f"{session_id[:8]}:t{turn_index}",
        session_id=session_id,
        timestamp=timestamp,
        turn_index=turn_index,
        user_text=user_text,
        assistant_text=assistant_text,
        tools_used=list(tools_used or []),
        tool_content=tool_content,
        transcript_path=transcript_path,
        agent_id=agent_id,
    )


def _index(chunks: list[TranscriptChunk]) -> TranscriptIndex:
    return TranscriptIndex(chunks, use_embeddings=False)


def _save_sqlite_index(chunks: list[TranscriptChunk], directory: Path) -> None:
    """Persist an index the way a real build does — SQLite, not chunks.jsonl.

    This distinction is the whole point of ``TestLazyHydration``. Saving an
    index that has no ``RecallDB`` attached writes only ``chunks.jsonl``, and
    loading that migrates EAGERLY (``lazy_chunks=False``), so the headers arrive
    already carrying their text. A fixture built that way cannot tell a
    hydrating implementation from a broken one — verified 2026-08-05, when the
    control below caught exactly that.
    """
    from synapt.recall.storage import RecallDB

    db = RecallDB(directory / "recall.db")
    try:
        TranscriptIndex(chunks, use_embeddings=False, db=db).save(directory)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Session selection
# ---------------------------------------------------------------------------


class TestSessionSelection(unittest.TestCase):
    """Which session gets resumed, and what happens when the answer is unclear.

    The load-bearing case is the unknown id. Falling back to the newest session
    would hand a reader another session's tail while they believe it is theirs —
    strictly worse than an error, because nothing about the output would look
    wrong.
    """

    def setUp(self):
        # B is newer than A, so "newest" and "named A" are distinguishable.
        self.index = _index([
            _chunk(SESSION_A, 0, "old question", "old answer",
                   timestamp="2026-08-01T10:00:00Z"),
            _chunk(SESSION_B, 0, "new question", "new answer",
                   timestamp="2026-08-05T10:00:00Z"),
        ])

    def test_default_resolves_to_newest_session(self):
        self.assertEqual(resolve_session(self.index, None), SESSION_B)

    def test_default_prefers_newest_caller_session_over_newer_foreign_session(self):
        self.assertEqual(resolve_session(self.index, None, {SESSION_A}), SESSION_A)

    def test_default_falls_back_store_wide_only_when_caller_has_no_indexed_session(self):
        self.assertEqual(resolve_session(self.index, None, {"not-indexed"}), SESSION_B)

    def test_legacy_unattributed_chunks_do_not_acquire_resuming_agent_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with mock.patch.dict(os.environ, {}, clear=True):
                old = _chunk(
                    SESSION_A,
                    0,
                    "caller continuity",
                    "answer",
                    timestamp="2026-01-01T00:00:00Z",
                    agent_id=None,
                ).to_dict()
                newer = _chunk(
                    SESSION_B,
                    0,
                    "foreign continuity",
                    "answer",
                    timestamp="2026-02-01T00:00:00Z",
                    agent_id=None,
                ).to_dict()
            # The legacy format had no authorship field at all.
            old.pop("agent_id", None)
            newer.pop("agent_id", None)
            (directory / "chunks.jsonl").write_text(
                json.dumps(old) + "\n" + json.dumps(newer) + "\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ, {"SYNAPT_AGENT_ID": "stable-agent"}, clear=True
            ):
                index = load_resume_index(directory)
                view = build_resume_view(
                    index,
                    caller_sources=[
                        CallerTranscript(
                            SESSION_A, Path("/old/cwd.jsonl"), 1.0, 1
                        )
                    ],
                    agent_id="stable-agent",
                    journal_path=None,
                )

            self.assertEqual(view.session_id, SESSION_A)
            self.assertEqual(view.selection_scope, "caller")

    def test_agent_identity_outranks_runtime_cwd_and_store_recency(self):
        index = _index([
            _chunk(
                SESSION_A,
                0,
                "same agent on the previous runtime",
                "continuing",
                timestamp="2026-08-31T10:00:00Z",
                agent_id="stable-agent",
            ),
            _chunk(
                SESSION_B,
                0,
                "newer foreign runtime cwd",
                "not the same agent",
                timestamp="2026-08-31T11:00:00Z",
                agent_id="foreign-agent",
            ),
        ])

        view = build_resume_view(
            index,
            caller_sources=[
                CallerTranscript(SESSION_B, Path("/codex/current.jsonl"), 1.0, 1)
            ],
            agent_id="stable-agent",
            journal_path=None,
        )

        self.assertEqual(view.session_id, SESSION_A)
        self.assertEqual(view.selection_scope, "agent")
        self.assertIn("same agent on the previous runtime", format_resume(view))
        self.assertIn("agent identity", format_resume(view).splitlines()[0])

    def test_explicit_session_selection_outranks_agent_identity(self):
        index = _index([
            _chunk(SESSION_A, 0, "agent default", "a", agent_id="stable-agent"),
            _chunk(SESSION_B, 0, "explicit target", "b", agent_id="foreign-agent"),
        ])

        view = build_resume_view(
            index,
            session_id=SESSION_B[:8],
            agent_id="stable-agent",
            journal_path=None,
        )

        self.assertEqual(view.session_id, SESSION_B)
        self.assertEqual(view.selection_scope, "explicit")
        self.assertIn("explicit target", format_resume(view))

    def test_unknown_agent_identity_falls_back_to_runtime_cwd(self):
        self.assertEqual(
            resolve_session(
                self.index,
                None,
                caller_session_ids={SESSION_A},
                agent_id="new-agent-with-no-history",
            ),
            SESSION_A,
        )

    def test_caller_unindexed_newer_transcript_is_named_on_first_line(self):
        source = CallerTranscript(
            session_id="cccccccc-1111-2222-3333-444444444444",
            path=Path("/source/cccc.jsonl"),
            mtime=2_000_000_000.0,
            size=14_000_000,
        )
        view = build_resume_view(
            self.index,
            caller_sources=[source],
            journal_path=None,
        )
        first = format_resume(view).splitlines()[0]
        self.assertIn("CALLER SOURCE STALE", first)
        self.assertIn("cccccccc", first)
        self.assertIn("14000000 bytes", first)

    def test_caller_present_but_live_extent_newer_is_named_on_first_line(self):
        source = CallerTranscript(
            session_id=SESSION_A,
            path=Path("/caller/a.jsonl"),
            mtime=2_000_000_000.0,
            size=14_000_000,
            latest_timestamp="2026-08-05T12:00:00Z",
        )
        view = build_resume_view(
            self.index,
            caller_sources=[source],
            journal_path=None,
        )

        first = format_resume(view).splitlines()[0]

        self.assertIn("CALLER SOURCE PARTIAL", first)
        self.assertIn(SESSION_A[:8], first)
        self.assertIn("2026-08-01T10:00:00Z", first)
        self.assertIn("2026-08-05T12:00:00Z", first)
        self.assertIn("synapt recall build --no-embeddings", first)

    def test_caller_matching_live_and_indexed_extents_has_no_partial_warning(self):
        source = CallerTranscript(
            session_id=SESSION_A,
            path=Path("/caller/a.jsonl"),
            mtime=2_000_000_000.0,
            size=14_000_000,
            latest_timestamp="2026-08-01T10:00:00Z",
        )
        first = format_resume(
            build_resume_view(
                self.index,
                caller_sources=[source],
                journal_path=None,
            )
        ).splitlines()[0]

        self.assertNotIn("CALLER SOURCE PARTIAL", first)

    def test_mixed_iso_offsets_compare_by_instant_not_spelling(self):
        index = _index([
            _chunk(
                SESSION_A,
                0,
                "earlier instant",
                "answer",
                timestamp="2026-08-01T12:00:00+02:00",
            ),
            _chunk(
                SESSION_A,
                1,
                "later instant",
                "answer",
                timestamp="2026-08-01T11:00:00Z",
            ),
        ])
        source = CallerTranscript(
            session_id=SESSION_A,
            path=Path("/caller/a.jsonl"),
            mtime=2_000_000_000.0,
            size=14_000_000,
            latest_timestamp="2026-08-01T10:30:00Z",
        )

        first = format_resume(
            build_resume_view(index, caller_sources=[source], journal_path=None)
        ).splitlines()[0]

        self.assertNotIn("CALLER SOURCE PARTIAL", first)

    def test_partial_warning_preserves_mixed_offset_spellings(self):
        indexed = "2026-08-01T12:00:00+01:00"
        live = "2026-08-01T12:30:00+01:00"
        index = _index([
            _chunk(
                SESSION_A,
                0,
                "indexed question",
                "indexed answer",
                timestamp=indexed,
            )
        ])
        source = CallerTranscript(
            session_id=SESSION_A,
            path=Path("/caller/a.jsonl"),
            mtime=2_000_000_000.0,
            size=14_000_000,
            latest_timestamp=live,
        )

        first = format_resume(
            build_resume_view(index, caller_sources=[source], journal_path=None)
        ).splitlines()[0]

        self.assertIn("CALLER SOURCE PARTIAL", first)
        self.assertIn(f"indexed through {indexed}", first)
        self.assertIn(f"live through {live}", first)

    def test_sidecar_only_session_names_missing_searchable_endpoint(self):
        index = _index([
            _chunk(
                SESSION_A,
                -2,
                "sidecar",
                "projection",
                timestamp="2026-08-01T12:00:00Z",
            )
        ])
        source = CallerTranscript(
            session_id=SESSION_A,
            path=Path("/caller/a.jsonl"),
            mtime=2_000_000_000.0,
            size=14_000_000,
            latest_timestamp="2026-08-01T11:00:00Z",
        )

        first = format_resume(
            build_resume_view(index, caller_sources=[source], journal_path=None)
        ).splitlines()[0]

        self.assertIn("CALLER SOURCE PARTIAL", first)
        self.assertIn("indexed through no searchable transcript endpoint", first)
        self.assertIn("live through 2026-08-01T11:00:00Z", first)

    def test_sidecar_projection_timestamp_cannot_hide_partial_transcript(self):
        index = _index([
            _chunk(
                SESSION_A,
                0,
                "indexed question",
                "indexed answer",
                timestamp="2026-08-01T10:00:00Z",
            ),
            _chunk(
                SESSION_A,
                -2,
                "sidecar",
                "projection",
                timestamp="2026-08-01T12:00:00Z",
            ),
        ])
        source = CallerTranscript(
            session_id=SESSION_A,
            path=Path("/caller/a.jsonl"),
            mtime=2_000_000_000.0,
            size=14_000_000,
            latest_timestamp="2026-08-01T11:00:00Z",
        )

        first = format_resume(
            build_resume_view(index, caller_sources=[source], journal_path=None)
        ).splitlines()[0]

        self.assertIn("CALLER SOURCE PARTIAL", first)
        self.assertIn("indexed through 2026-08-01T10:00:00Z", first)

    def test_foreign_live_extent_is_not_a_caller_partial_warning(self):
        source = CallerTranscript(
            session_id=SESSION_B,
            path=Path("/foreign/b.jsonl"),
            mtime=2_000_000_000.0,
            size=14_000_000,
            latest_timestamp="2026-08-06T12:00:00Z",
        )
        first = format_resume(
            build_resume_view(
                self.index,
                session_id=SESSION_A,
                caller_sources=[source],
                journal_path=None,
            )
        ).splitlines()[0]

        self.assertNotIn("CALLER SOURCE PARTIAL", first)

    def test_latest_event_timestamp_reads_backward_over_large_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "session.jsonl"
            transcript.write_text(
                json.dumps({"timestamp": "2026-08-01T10:00:00Z"})
                + "\n"
                + json.dumps(
                    {
                        "timestamp": "2026-08-05T12:00:00Z",
                        "payload": "x" * 70_000,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _latest_event_timestamp(transcript),
                "2026-08-05T12:00:00Z",
            )

    def test_store_fallback_names_selected_worktree_on_first_line(self):
        index = _index([
            _chunk(
                SESSION_B,
                0,
                "q",
                "a",
                transcript_path="/repo/.synapt/recall/worktrees/foreign/transcripts/b.jsonl",
            )
        ])
        first = format_resume(
            build_resume_view(index, caller_sources=[], journal_path=None)
        ).splitlines()[0]
        self.assertIn("store fallback from worktree:foreign", first)

    def test_caller_discovery_excludes_codex_sessions_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            caller = root / "caller"
            caller.mkdir()
            codex = root / "codex"
            codex.mkdir()
            own = codex / "rollout-own.jsonl"
            foreign = codex / "rollout-foreign.jsonl"
            own.write_text("{}\n")
            foreign.write_text("{}\n")

            def cwd_for(path):
                return caller.resolve() if path == own else (root / "foreign").resolve()

            def sid_for(path):
                return "own-session" if Path(path) == own else "foreign-session"

            with (
                mock.patch("synapt.recall.core.project_transcript_dir", return_value=None),
                mock.patch("synapt.recall.codex.discover_codex_sessions", return_value=codex),
                mock.patch("synapt.recall.codex._session_cwd", side_effect=cwd_for),
                mock.patch("synapt.recall.journal.extract_session_id", side_effect=sid_for),
            ):
                found = caller_transcripts(caller)

            self.assertEqual([item.session_id for item in found], ["own-session"])

    def test_caller_discovery_skips_source_removed_after_stat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            caller = root / "caller"
            caller.mkdir()
            transcript_root = root / "transcripts"
            transcript_root.mkdir()
            transcript = transcript_root / "session.jsonl"
            transcript.write_text("{}\n")

            with (
                mock.patch(
                    "synapt.recall.core.project_transcript_dir",
                    return_value=transcript_root,
                ),
                mock.patch(
                    "synapt.recall.codex.discover_codex_sessions",
                    return_value=None,
                ),
                mock.patch(
                    "synapt.recall.journal.extract_session_id",
                    return_value=SESSION_A,
                ),
                mock.patch(
                    "synapt.recall.resume._latest_event_timestamp",
                    side_effect=FileNotFoundError("removed after stat"),
                ),
            ):
                found = caller_transcripts(caller)

            self.assertEqual(found, [])

    def test_source_labels_do_not_print_absolute_non_worktree_paths(self):
        self.assertEqual(
            _source_label(
                "/Users/example/.claude/projects/"
                "-Users-example-Development-synapt/session.jsonl"
            ),
            "project:-Users-example-Development-synapt",
        )
        self.assertEqual(
            _source_label("/var/transcripts/session.jsonl"),
            "source:transcripts",
        )

    def test_bounded_session_listing_derives_compact_source_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _save_sqlite_index([
                _chunk(
                    SESSION_A,
                    0,
                    "question",
                    "answer",
                    transcript_path=(
                        "/Users/example/.claude/projects/"
                        "-Users-example-Development-synapt/session.jsonl"
                    ),
                )
            ], root)

            index = load_resume_index(root)
            try:
                rows = index.list_sessions()
            finally:
                index.close()

        self.assertEqual(
            rows[0]["source_root"],
            "project:-Users-example-Development-synapt",
        )

    def test_control_newest_is_not_the_named_session(self):
        """Without this the selection tests could pass by coincidence."""
        self.assertNotEqual(SESSION_A, resolve_session(self.index, None))

    def test_exact_session_id_resolves(self):
        self.assertEqual(resolve_session(self.index, SESSION_A), SESSION_A)

    def test_unique_prefix_resolves(self):
        """`recall sessions` prints 8 characters, so prefixes are the real input."""
        self.assertEqual(resolve_session(self.index, SESSION_A[:8]), SESSION_A)

    def test_ambiguous_prefix_errors_and_names_candidates(self):
        index = _index([
            _chunk("dup-1111", 0, "q", "a"),
            _chunk("dup-2222", 0, "q", "a"),
        ])
        with self.assertRaises(ResumeError) as ctx:
            resolve_session(index, "dup-")
        message = str(ctx.exception)
        self.assertIn("dup-1111", message)
        self.assertIn("dup-2222", message)

    def test_unknown_session_errors_rather_than_falling_back(self):
        with self.assertRaises(ResumeError) as ctx:
            resolve_session(self.index, "nosuchsession")
        self.assertIn("nosuchsession", str(ctx.exception))

    def test_unknown_session_does_not_return_the_newest(self):
        """The failure this guards is silent, so assert the fallback never happens."""
        with contextlib.suppress(ResumeError):
            result = resolve_session(self.index, "nosuchsession")
            self.fail(f"expected ResumeError, silently resolved to {result}")

    def test_empty_index_errors(self):
        with self.assertRaises(ResumeError):
            resolve_session(_index([]), None)


# ---------------------------------------------------------------------------
# The noise discriminator
# ---------------------------------------------------------------------------


class TestHarnessNoiseDiscriminator(unittest.TestCase):
    """Exclusion requires POSITIVE identification of harness authorship.

    The rule is a conjunction: the user text is entirely a harness control block
    AND nothing responded to it. Each half alone has a known false-reject, and
    both false-rejects are witnessed here so that weakening the AND to an OR reds
    a specific row rather than passing quietly.
    """

    def test_slash_command_echo_is_harness_authored(self):
        chunk = _chunk(
            SESSION_A, 5,
            user_text=(
                "<command-name>/compact</command-name>\n"
                "<command-message>compact</command-message>\n"
                "<command-args>some directive</command-args>"
            ),
        )
        self.assertTrue(is_harness_authored(chunk))

    def test_local_command_stdout_is_harness_authored(self):
        chunk = _chunk(
            SESSION_A, 6,
            user_text="<local-command-stdout>Compacted (ctrl+o)</local-command-stdout>",
        )
        self.assertTrue(is_harness_authored(chunk))

    def test_continuation_preamble_is_harness_authored(self):
        chunk = _chunk(SESSION_A, 7, user_text=CONTINUATION_PREAMBLE)
        self.assertTrue(is_harness_authored(chunk))

    def test_final_user_turn_with_no_reply_is_kept(self):
        """A dropped baton looks exactly like this. Deleting it defeats the feature."""
        chunk = _chunk(SESSION_A, 8, user_text="ship the release when CI goes green")
        self.assertFalse(is_harness_authored(chunk))

    def test_prose_mentioning_a_marker_is_kept_when_something_responded(self):
        """Kill-witness for the marker half: the marker ALONE must not reject."""
        chunk = _chunk(
            SESSION_A, 9,
            user_text="why does <command-name> leak into the index?",
            assistant_text="because scrub.py does not cover it",
        )
        self.assertFalse(is_harness_authored(chunk))

    def test_prose_mentioning_a_marker_is_kept_even_with_no_reply(self):
        """The residue outside the block is participant text, so it cannot be harness-only."""
        chunk = _chunk(
            SESSION_A, 10,
            user_text="look at <command-name>/compact</command-name> please",
        )
        self.assertFalse(is_harness_authored(chunk))

    def test_unanswered_turn_without_any_marker_is_kept(self):
        """Kill-witness for the emptiness half: emptiness ALONE must not reject."""
        chunk = _chunk(SESSION_A, 11, user_text="one last thought before I go")
        self.assertFalse(is_harness_authored(chunk))

    def test_harness_block_that_got_a_reply_is_kept(self):
        """A participant responded, so the turn is part of the conversation."""
        chunk = _chunk(
            SESSION_A, 12,
            user_text="<local-command-stdout>output</local-command-stdout>",
            assistant_text="I see the command output",
        )
        self.assertFalse(is_harness_authored(chunk))

    def test_harness_block_with_tools_is_kept(self):
        chunk = _chunk(
            SESSION_A, 13,
            user_text="<local-command-stdout>output</local-command-stdout>",
            tools_used=["Bash"],
        )
        self.assertFalse(is_harness_authored(chunk))

    def test_raw_slash_command_typed_by_a_participant_is_kept(self):
        """`/compact <directive>` is authored by a person; only its ECHO is harness text.

        Rejecting it would be widening the filter until the tail looks tidy,
        which is the failure mode recall#919 was about.
        """
        chunk = _chunk(SESSION_A, 14, user_text="/compact keep what matters")
        self.assertFalse(is_harness_authored(chunk))

    def test_empty_chunk_is_not_classified_as_harness_authored(self):
        """Emptiness is a separate reason for exclusion; the two must not be conflated."""
        self.assertFalse(is_harness_authored(_chunk(SESSION_A, 15)))

    def test_view_excludes_harness_turns_and_reports_how_many(self):
        index = _index([
            _chunk(SESSION_A, 0, "real question", "real answer"),
            _chunk(SESSION_A, 1, user_text=CONTINUATION_PREAMBLE),
            _chunk(SESSION_A, 2,
                   user_text="<command-name>/compact</command-name>"),
            _chunk(SESSION_A, 3, "last words"),
        ])
        view = build_resume_view(index, limit=10, journal_path=None)
        self.assertEqual([t.chunk_id for t in view.turns],
                         [f"{SESSION_A[:8]}:t0", f"{SESSION_A[:8]}:t3"])
        self.assertEqual(view.excluded_count, 2)

    def test_view_excludes_content_free_chunks(self):
        index = _index([
            _chunk(SESSION_A, 0, "question", "answer"),
            _chunk(SESSION_A, 1),  # no user, no assistant, no tools
        ])
        view = build_resume_view(index, limit=10, journal_path=None)
        self.assertEqual(len(view.turns), 1)


# ---------------------------------------------------------------------------
# Ordering, limit, traceability
# ---------------------------------------------------------------------------


class TestOrderingAndLimit(unittest.TestCase):
    def setUp(self):
        self.index = _index([
            _chunk(SESSION_A, i, f"q{i}", f"a{i}") for i in range(6)
        ])

    def test_turns_are_newest_last(self):
        view = build_resume_view(self.index, limit=3, journal_path=None)
        self.assertEqual([t.turn_index for t in view.turns], [3, 4, 5])

    def test_limit_takes_the_tail_not_the_head(self):
        view = build_resume_view(self.index, limit=2, journal_path=None)
        self.assertEqual([t.turn_index for t in view.turns], [4, 5])

    def test_limit_larger_than_available_returns_everything(self):
        view = build_resume_view(self.index, limit=100, journal_path=None)
        self.assertEqual(len(view.turns), 6)

    def test_ordering_is_by_turn_index_not_storage_order(self):
        shuffled = _index([
            _chunk(SESSION_A, 2, "q2", "a2"),
            _chunk(SESSION_A, 0, "q0", "a0"),
            _chunk(SESSION_A, 1, "q1", "a1"),
        ])
        view = build_resume_view(shuffled, limit=10, journal_path=None)
        self.assertEqual([t.turn_index for t in view.turns], [0, 1, 2])

    def test_journal_chunks_are_not_treated_as_turns(self):
        """Journal chunks carry turn_index=-1 as a sentinel; they are not conversation."""
        index = _index([
            _chunk(SESSION_A, -1, "journal chunk text", "x"),
            _chunk(SESSION_A, 0, "q0", "a0"),
        ])
        view = build_resume_view(index, limit=10, journal_path=None)
        self.assertEqual([t.turn_index for t in view.turns], [0])

    def test_invalid_limit_errors_rather_than_returning_empty(self):
        """A zero limit returning "nothing to resume" would misreport an empty session."""
        with self.assertRaises(ResumeError):
            build_resume_view(self.index, limit=0, journal_path=None)


class TestChunkTraceability(unittest.TestCase):
    """"Each traceable to its chunk" is a contract term, so the ids must be real."""

    def test_every_turn_carries_its_chunk_id(self):
        index = _index([_chunk(SESSION_A, i, f"q{i}", f"a{i}") for i in range(3)])
        view = build_resume_view(index, limit=10, journal_path=None)
        self.assertEqual(
            [t.chunk_id for t in view.turns],
            [f"{SESSION_A[:8]}:t0", f"{SESSION_A[:8]}:t1", f"{SESSION_A[:8]}:t2"],
        )

    def test_chunk_ids_appear_in_rendered_output(self):
        index = _index([_chunk(SESSION_A, 0, "q", "a")])
        text = format_resume(build_resume_view(index, limit=10, journal_path=None))
        self.assertIn(f"{SESSION_A[:8]}:t0", text)


# ---------------------------------------------------------------------------
# Journal binding
# ---------------------------------------------------------------------------


class TestJournalBinding(unittest.TestCase):
    """Which journal entry, if any, belongs with this session's tail.

    Provenance has three states and they are rendered differently, because a
    pairing that is inferred must not read like one that is proven. Entries
    written when no transcript was discoverable carry ``session_id=""`` (verified
    against real journals, 2026-08-05), so an exact-match-only rule would fail
    to bind in the most common case while every constructed test still passed.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.journal = Path(self.tmp.name) / "journal.jsonl"
        self.index = _index([
            _chunk(SESSION_A, 0, "old", "old", timestamp="2026-08-01T10:00:00Z"),
            _chunk(SESSION_B, 0, "new", "new", timestamp="2026-08-05T10:00:00Z"),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, session_id: str, focus: str, timestamp: str):
        append_entry(
            JournalEntry(timestamp=timestamp, session_id=session_id, focus=focus),
            self.journal,
        )

    def test_matching_session_id_binds(self):
        self._write(SESSION_B, "shipping the resume verb", "2026-08-05T11:00:00Z")
        view = build_resume_view(self.index, limit=10, journal_path=self.journal)
        self.assertIsNotNone(view.journal)
        self.assertEqual(view.journal.focus, "shipping the resume verb")
        self.assertEqual(view.journal_provenance, "bound")

    def test_entry_from_a_different_session_is_never_shown(self):
        """Positive contradiction. Showing it would pair one session's tail with another's intent."""
        self._write(SESSION_A, "a completely different session", "2026-08-05T11:00:00Z")
        view = build_resume_view(self.index, limit=10, journal_path=self.journal)
        self.assertIsNone(view.journal)
        self.assertIsNone(view.journal_provenance)

    def test_agent_selected_session_does_not_infer_a_foreign_unbound_journal(self):
        """Agent identity binds the session, not a sessionless shared-store journal."""
        self._write("", "foreign unbound intent", "2026-08-05T11:00:00Z")
        index = _index([
            _chunk(
                SESSION_A,
                0,
                "same agent but older",
                "a",
                timestamp="2026-08-01T10:00:00Z",
                agent_id="stable-agent",
            ),
            _chunk(
                SESSION_B,
                0,
                "newer foreign session",
                "b",
                timestamp="2026-08-05T10:00:00Z",
                agent_id="foreign-agent",
            ),
        ])

        view = build_resume_view(
            index,
            agent_id="stable-agent",
            limit=10,
            journal_path=self.journal,
        )

        self.assertEqual(view.session_id, SESSION_A)
        self.assertIsNone(view.journal)
        self.assertIsNone(view.journal_provenance)

    def test_agent_selected_session_keeps_exact_journal_binding(self):
        self._write(SESSION_A, "bound agent intent", "2026-08-01T11:00:00Z")
        index = _index([
            _chunk(
                SESSION_A,
                0,
                "same agent but older",
                "a",
                timestamp="2026-08-01T10:00:00Z",
                agent_id="stable-agent",
            ),
            _chunk(
                SESSION_B,
                0,
                "newer foreign session",
                "b",
                timestamp="2026-08-05T10:00:00Z",
                agent_id="foreign-agent",
            ),
        ])

        view = build_resume_view(
            index,
            agent_id="stable-agent",
            limit=10,
            journal_path=self.journal,
        )

        self.assertEqual(view.journal.focus, "bound agent intent")
        self.assertEqual(view.journal_provenance, "bound")

    def test_entry_without_a_session_id_is_shown_but_labelled_inferred(self):
        """Absence of evidence cannot contradict, so it is disclosed rather than hidden."""
        self._write("", "unbound entry", "2026-08-05T11:00:00Z")
        view = build_resume_view(self.index, limit=10, journal_path=self.journal)
        self.assertIsNotNone(view.journal)
        self.assertEqual(view.journal_provenance, "inferred")

    def test_inferred_binding_is_visible_in_the_output(self):
        """An inferred pairing that renders identically to a proven one is a fabrication."""
        self._write("", "unbound entry", "2026-08-05T11:00:00Z")
        view = build_resume_view(self.index, limit=10, journal_path=self.journal)
        self.assertIn("inferred", format_resume(view).lower())

    def test_named_older_session_never_takes_an_unbound_entry(self):
        """Inference is only defensible for the newest session; a named one must be exact."""
        self._write("", "unbound entry", "2026-08-05T11:00:00Z")
        view = build_resume_view(
            self.index, session_id=SESSION_A, limit=10, journal_path=self.journal
        )
        self.assertIsNone(view.journal)

    def test_named_older_session_binds_on_exact_match(self):
        self._write(SESSION_A, "the older session", "2026-08-01T11:00:00Z")
        view = build_resume_view(
            self.index, session_id=SESSION_A, limit=10, journal_path=self.journal
        )
        self.assertEqual(view.journal.focus, "the older session")
        self.assertEqual(view.journal_provenance, "bound")

    def test_exact_match_wins_over_an_unbound_entry(self):
        self._write("", "unbound and newer", "2026-08-05T12:00:00Z")
        self._write(SESSION_B, "bound and older", "2026-08-05T11:00:00Z")
        view = build_resume_view(self.index, limit=10, journal_path=self.journal)
        self.assertEqual(view.journal.focus, "bound and older")
        self.assertEqual(view.journal_provenance, "bound")

    def test_missing_journal_file_is_not_an_error(self):
        view = build_resume_view(
            self.index, limit=10, journal_path=Path(self.tmp.name) / "absent.jsonl"
        )
        self.assertIsNone(view.journal)


# ---------------------------------------------------------------------------
# Lazy hydration
# ---------------------------------------------------------------------------


class TestLazyHydration(unittest.TestCase):
    """A saved index loads HEADERS ONLY, so text must be hydrated on read.

    The control matters more than the assertion. An index built in memory from
    full chunks cannot distinguish hydrated from unhydrated, so a test that skips
    the disk round-trip would pass against a broken implementation.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        _save_sqlite_index([
            _chunk(SESSION_A, 0, "the seeded question", "the seeded answer"),
            _chunk(SESSION_A, 1, "second question", "second answer"),
        ], self.dir)
        self.loaded = TranscriptIndex.load(self.dir, use_embeddings=False)

    def tearDown(self):
        # Close the DB handle BEFORE removing the directory. ``load()`` opens a
        # connection to recall.db and holds it for as long as the index lives,
        # which here is the lifetime of this TestCase — so tearDown runs while
        # the file is still open. POSIX happily unlinks an open file; Windows
        # raises PermissionError (WinError 32) and the temp-dir cleanup fails,
        # reddening all four tests in this class for a reason that has nothing
        # to do with what they assert. Found on Windows CI, 2026-08-06; it
        # cannot reproduce on macOS or Linux.
        if self.loaded._db is not None:
            self.loaded._db.close()
        self.tmp.cleanup()

    def test_control_the_fixture_actually_loads_lazily(self):
        """If the index is not lazy, every assertion in this class is vacuous."""
        self.assertTrue(
            self.loaded._lazy_chunks,
            "fixture loaded eagerly — the hydration tests measure nothing",
        )

    def test_a_loaded_index_holds_a_db_handle_that_can_be_released(self):
        """Partial, platform-honest witness for a Windows-only failure.

        The failure itself — ``PermissionError`` when the temp directory is
        removed while ``recall.db`` is still open — cannot occur on POSIX, which
        unlinks open files happily. So **no assertion in this file can red on
        macOS or Linux when the tearDown close is removed**; Windows CI is the
        only instrument that detects it, and this test does not pretend
        otherwise.

        What IS assertable everywhere is the property whose absence caused it:
        loading opens a handle, and closing it actually releases it. That much
        is worth pinning, because a close that silently did nothing would look
        identical here and on macOS alike.
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _save_sqlite_index([_chunk(SESSION_A, 0, "q", "a")], directory)
            index = TranscriptIndex.load(directory, use_embeddings=False)

            self.assertIsNotNone(index._db, "load did not open a DB handle")
            index._db.close()
            with self.assertRaises(Exception):
                index._db.load_chunk_headers()

    def test_control_headers_really_are_empty_before_hydration(self):
        """Proves the hazard exists. If this ever fails, the witness below is inert."""
        headers = self.loaded.sessions[SESSION_A]
        self.assertTrue(
            all(not h.user_text and not h.assistant_text for h in headers),
            "headers already carry text — the hydration witness no longer measures anything",
        )

    def test_resume_returns_hydrated_text(self):
        view = build_resume_view(self.loaded, limit=10, journal_path=None)
        joined = " ".join(t.user_text + t.assistant_text for t in view.turns)
        self.assertIn("the seeded question", joined)
        self.assertIn("the seeded answer", joined)

    def test_hydrated_turns_survive_the_noise_filter(self):
        """Unhydrated chunks look content-free, so a broken read reports an empty session."""
        view = build_resume_view(self.loaded, limit=10, journal_path=None)
        self.assertEqual(len(view.turns), 2)


class TestBoundedResumeLoad(unittest.TestCase):
    """The CLI read should scale with sessions plus the selected tail, not all chunks."""

    def test_sqlite_loader_does_not_construct_the_full_transcript_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _save_sqlite_index([
                _chunk(SESSION_A, 0, "older question", "older answer",
                       timestamp="2026-08-01T10:00:00Z"),
                _chunk(SESSION_B, 0, "newer question", "newer answer",
                       timestamp="2026-08-05T10:00:00Z"),
            ], directory)

            with mock.patch.object(
                TranscriptIndex,
                "load",
                side_effect=AssertionError("full index load should not run"),
            ):
                index = load_resume_index(directory)
                try:
                    view = build_resume_view(index, limit=10, journal_path=None)
                finally:
                    index.close()

            self.assertEqual(view.session_id, SESSION_B)
            self.assertEqual(view.turns[0].user_text, "newer question")

    def test_session_order_ignores_newer_journal_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _save_sqlite_index([
                _chunk(SESSION_A, 0, "live", "work",
                       timestamp="2026-08-05T10:00:00Z"),
                _chunk(SESSION_B, 0, "old", "work",
                       timestamp="2026-08-01T10:00:00Z"),
                _chunk(SESSION_B, -1, "journal", "",
                       timestamp="2026-08-25T10:00:00Z"),
            ], directory)

            index = load_resume_index(directory)
            try:
                self.assertEqual(resolve_session(index, None), SESSION_A)
            finally:
                index.close()

    def test_only_selected_session_is_hydrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _save_sqlite_index([
                _chunk(SESSION_A, 0, "secret older question", "older answer",
                       timestamp="2026-08-01T10:00:00Z"),
                _chunk(SESSION_B, 0, "selected question", "selected answer",
                       timestamp="2026-08-05T10:00:00Z"),
            ], directory)

            index = load_resume_index(directory)
            try:
                with mock.patch.object(
                    index._db,
                    "load_session_chunks",
                    wraps=index._db.load_session_chunks,
                ) as load:
                    view = build_resume_view(index, limit=10, journal_path=None)
                load.assert_called_once_with(SESSION_B)
            finally:
                index.close()

            rendered = format_resume(view)
            self.assertIn("selected question", rendered)
            self.assertNotIn("secret older question", rendered)

    def test_session_without_a_timestamp_does_not_disappear(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _save_sqlite_index([
                _chunk(SESSION_A, 0, "question", "answer", timestamp=""),
            ], directory)

            index = load_resume_index(directory)
            try:
                self.assertEqual(resolve_session(index, None), SESSION_A)
            finally:
                index.close()

    def test_timestamp_spelling_does_not_decide_bounded_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _save_sqlite_index([
                _chunk(SESSION_A, 0, "later", "answer",
                       timestamp="2026-08-06T03:00:00.500000+00:00"),
                _chunk(SESSION_B, 0, "earlier", "answer",
                       timestamp="2026-08-06T03:00:00Z"),
            ], directory)

            index = load_resume_index(directory)
            try:
                self.assertEqual(resolve_session(index, None), SESSION_A)
            finally:
                index.close()


# ---------------------------------------------------------------------------
# Continuation segments
# ---------------------------------------------------------------------------


class TestContinuationSegments(unittest.TestCase):
    """One question can span several chunks; later ones restate it synthetically.

    Rendering that restatement as user speech would report a question the person
    never asked a second time.
    """

    CONTEXT_PREFIX = "(context: User previously asked: how do I ship this?...)"

    def test_continuation_segment_is_flagged(self):
        index = _index([
            _chunk(SESSION_A, 0, "how do I ship this?", "first part"),
            _chunk(SESSION_A, 1, self.CONTEXT_PREFIX, "second part"),
        ])
        view = build_resume_view(index, limit=10, journal_path=None)
        self.assertFalse(view.turns[0].is_continuation)
        self.assertTrue(view.turns[1].is_continuation)

    def test_continuation_text_is_suppressed_in_the_VIEW_not_only_the_rendering(self):
        """Two independent layers claim this; each needs its own witness.

        ``format_resume`` checks ``is_continuation`` before it ever looks at
        ``user_text``, so the rendering test below passes even when the view
        still carries the synthetic restatement — found by mutation, 2026-08-05.
        The view is a data surface, and a programmatic consumer reading
        ``turn.user_text`` would get a question nobody asked.
        """
        index = _index([
            _chunk(SESSION_A, 0, "how do I ship this?", "first part"),
            _chunk(SESSION_A, 1, self.CONTEXT_PREFIX, "second part"),
        ])
        view = build_resume_view(index, limit=10, journal_path=None)
        self.assertEqual(view.turns[1].user_text, "")

    def test_continuation_text_is_not_rendered_as_a_user_message(self):
        index = _index([
            _chunk(SESSION_A, 0, "how do I ship this?", "first part"),
            _chunk(SESSION_A, 1, self.CONTEXT_PREFIX, "second part"),
        ])
        text = format_resume(build_resume_view(index, limit=10, journal_path=None))
        self.assertNotIn("User previously asked", text)
        self.assertIn("second part", text)

    def test_continuation_is_marked_so_the_reader_knows_why_no_question_appears(self):
        """A separate claim from suppression: silence without a marker reads as a bug."""
        index = _index([
            _chunk(SESSION_A, 0, "how do I ship this?", "first part"),
            _chunk(SESSION_A, 1, self.CONTEXT_PREFIX, "second part"),
        ])
        text = format_resume(build_resume_view(index, limit=10, journal_path=None))
        self.assertIn("continues the previous question", text)

    def test_window_opening_mid_reply_is_anchored_to_its_question(self):
        """The defect this fixes was invisible to constructed tests and obvious in the fruit.

        On a real 324-chunk session every turn in the default window was a
        continuation, so the output carried no user message at all — the reader
        could not tell what was being answered. Verified 2026-08-05 by running
        the command against a real index rather than a fixture.
        """
        index = _index(
            [_chunk(SESSION_A, 0, "the question that started it", "part 0")]
            + [_chunk(SESSION_A, i, self.CONTEXT_PREFIX, f"part {i}") for i in range(1, 8)]
        )
        view = build_resume_view(index, limit=3, journal_path=None)
        self.assertFalse(view.turns[0].is_continuation)
        self.assertEqual(view.turns[0].user_text, "the question that started it")

    def test_the_anchor_costs_exactly_one_turn_however_long_the_reply_ran(self):
        """Bounded by construction: a 200-segment reply must not drag 200 turns in."""
        index = _index(
            [_chunk(SESSION_A, 0, "q", "part 0")]
            + [_chunk(SESSION_A, i, self.CONTEXT_PREFIX, f"part {i}") for i in range(1, 200)]
        )
        view = build_resume_view(index, limit=3, journal_path=None)
        self.assertEqual(len(view.turns), 4)

    def test_the_gap_created_by_anchoring_is_disclosed(self):
        """A tail that reads as contiguous while skipping turns misreports the session."""
        index = _index(
            [_chunk(SESSION_A, 0, "q", "part 0")]
            + [_chunk(SESSION_A, i, self.CONTEXT_PREFIX, f"part {i}") for i in range(1, 8)]
        )
        view = build_resume_view(index, limit=3, journal_path=None)
        self.assertEqual(view.omitted_between, 4)
        self.assertIn("4 intermediate turns omitted", format_resume(view))

    def test_no_anchoring_when_the_window_already_starts_at_a_question(self):
        """The anchor must not fire when it is not needed, or every window grows by one."""
        index = _index([_chunk(SESSION_A, i, f"q{i}", f"a{i}") for i in range(6)])
        view = build_resume_view(index, limit=3, journal_path=None)
        self.assertEqual(len(view.turns), 3)
        self.assertEqual(view.omitted_between, 0)

    def test_all_continuations_with_no_question_available_does_not_crash(self):
        """A session whose indexed head is already a continuation has no anchor to find."""
        index = _index([
            _chunk(SESSION_A, i, self.CONTEXT_PREFIX, f"part {i}") for i in range(4)
        ])
        view = build_resume_view(index, limit=2, journal_path=None)
        self.assertEqual(len(view.turns), 2)
        self.assertEqual(view.omitted_between, 0)

    def test_continuation_segments_are_still_kept(self):
        """They carry real assistant work; only their synthetic user line is suppressed."""
        index = _index([
            _chunk(SESSION_A, 0, "how do I ship this?", "first part"),
            _chunk(SESSION_A, 1, self.CONTEXT_PREFIX, "second part"),
        ])
        view = build_resume_view(index, limit=10, journal_path=None)
        self.assertEqual(len(view.turns), 2)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestFormatting(unittest.TestCase):
    def test_long_text_is_truncated_with_an_explicit_marker(self):
        index = _index([_chunk(SESSION_A, 0, "q", "y" * 5000)])
        text = format_resume(build_resume_view(index, limit=10, journal_path=None),
                             max_chars=200)
        self.assertIn("truncated", text.lower())
        self.assertLess(len(text), 2000)

    def test_short_text_is_not_marked_truncated(self):
        index = _index([_chunk(SESSION_A, 0, "q", "a short answer")])
        text = format_resume(build_resume_view(index, limit=10, journal_path=None),
                             max_chars=200)
        self.assertNotIn("truncated", text.lower())

    def test_tools_are_surfaced(self):
        index = _index([_chunk(SESSION_A, 0, "q", "a", tools_used=["Bash", "Edit"])])
        text = format_resume(build_resume_view(index, limit=10, journal_path=None))
        self.assertIn("Bash", text)

    def test_session_id_appears_in_the_header(self):
        index = _index([_chunk(SESSION_A, 0, "q", "a")])
        text = format_resume(build_resume_view(index, limit=10, journal_path=None))
        self.assertIn(SESSION_A[:8], text)


# ---------------------------------------------------------------------------
# Empty and error states at the CLI boundary
# ---------------------------------------------------------------------------


def _run_cli(index_dir: Path, session: str | None = None, turns: int = 10):
    """Run cmd_resume, returning (stdout, stderr, exit_code)."""
    from synapt.recall.cli import cmd_resume

    args = Namespace(index=str(index_dir), session=session, turns=turns)
    out, err, code = io.StringIO(), io.StringIO(), 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cmd_resume(args)
    except SystemExit as exc:
        code = exc.code or 0
    return out.getvalue(), err.getvalue(), code


class TestEmptyAndErrorStates(unittest.TestCase):
    """Three states that must stay distinguishable.

    "You never built an index", "you built one and it is empty", and "that
    session does not exist" have different fixes, so collapsing them into one
    message sends the reader down the wrong path.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # Ref #967: the CLI path under test resolves the recall data root
        # implicitly, so without an owned store these ran against a real
        # checkout and their "empty" states could come from the operator's
        # history rather than from the fixtures built here.
        self._store = owned_store()

    def tearDown(self):
        self._store.restore()
        self.tmp.cleanup()

    def test_missing_index_exits_nonzero_and_says_how_to_fix(self):
        _, err, code = _run_cli(self.dir / "does-not-exist")
        self.assertEqual(code, 1)
        self.assertIn("build", err.lower())

    def test_empty_index_is_an_honest_empty_state_not_an_error(self):
        _save_sqlite_index([], self.dir)
        out, _, code = _run_cli(self.dir)
        self.assertEqual(code, 0)
        self.assertIn("no session", (out).lower())

    def test_unknown_session_exits_nonzero(self):
        _save_sqlite_index([_chunk(SESSION_A, 0, "q", "a")], self.dir)
        _, err, code = _run_cli(self.dir, session="nope")
        self.assertEqual(code, 1)
        self.assertIn("nope", err)

    def test_session_of_only_harness_turns_reports_that_distinctly(self):
        """Not the same as an empty index: the session exists, its tail is noise."""
        _save_sqlite_index([
            _chunk(SESSION_A, 0, user_text=CONTINUATION_PREAMBLE),
            _chunk(SESSION_A, 1, user_text="<command-name>/x</command-name>"),
        ], self.dir)
        out, _, code = _run_cli(self.dir)
        self.assertEqual(code, 0)
        self.assertIn(SESSION_A[:8], out)

    def test_cli_reads_the_real_journal_path(self):
        """The library takes ``journal_path=None`` to mean "no journal".

        That default is deliberate — it keeps implicit I/O out of a library
        function — but it means the CLI is the ONLY thing that supplies a real
        path. Without this witness the journal could silently never appear in
        production while every library test above stayed green.
        """
        _save_sqlite_index([_chunk(SESSION_A, 0, "q", "a")], self.dir)
        journal = Path(self.tmp.name) / "real-journal.jsonl"
        append_entry(
            JournalEntry(timestamp="2026-08-05T11:00:00Z", session_id=SESSION_A,
                         focus="the intent that must survive"),
            journal,
        )
        with mock.patch("synapt.recall.journal._journal_path", return_value=journal):
            out, _, code = _run_cli(self.dir)
        self.assertEqual(code, 0)
        self.assertIn("the intent that must survive", out)

    def test_happy_path_prints_turns_oldest_first(self):
        """Named for what it measures. With two turns and a limit of ten it pins
        ORDERING, not tail-selection — the tail is pinned in TestOrderingAndLimit,
        and a mutation run showed this row stayed green under a head/tail swap."""
        _save_sqlite_index([
            _chunk(SESSION_A, 0, "first question", "first answer"),
            _chunk(SESSION_A, 1, "last question", "last answer"),
        ], self.dir)
        out, _, code = _run_cli(self.dir)
        self.assertEqual(code, 0)
        self.assertIn("last answer", out)
        self.assertLess(out.index("first answer"), out.index("last answer"),
                        "turns must read oldest-first so the tail ends at the newest")

    def test_cli_uses_stable_agent_identity_across_runtime_cwds(self):
        _save_sqlite_index([
            _chunk(
                SESSION_A,
                0,
                "same agent on another runtime cwd",
                "continue here",
                timestamp="2026-08-01T10:00:00Z",
                agent_id="stable-agent",
            ),
            _chunk(
                SESSION_B,
                0,
                "newer foreign cwd",
                "wrong continuity",
                timestamp="2026-08-05T10:00:00Z",
                agent_id="foreign-agent",
            ),
        ], self.dir)

        with mock.patch.dict(os.environ, {"SYNAPT_AGENT_ID": "stable-agent"}):
            out, _, code = _run_cli(self.dir)

        self.assertEqual(code, 0)
        self.assertIn("same agent on another runtime cwd", out)
        self.assertNotIn("newer foreign cwd", out)


# ---------------------------------------------------------------------------
# Cross-runtime
# ---------------------------------------------------------------------------


class TestCrossRuntime(unittest.TestCase):
    """Codex support holds by construction: resume reads chunks, never a parser.

    This is the relay's dropped-baton beat, gated by recall#926.
    """

    def test_codex_parsed_session_resumes(self):
        from synapt.recall.codex import parse_codex_transcript

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-2026-08-05T10-00-00-abc.jsonl"
            entries = [
                {"type": "session_meta",
                 "payload": {"id": "codex-session-1", "cwd": tmp,
                             "timestamp": "2026-08-05T10:00:00Z"}},
                {"type": "response_item", "timestamp": "2026-08-05T10:01:00Z",
                 "payload": {"type": "message", "role": "user",
                             "content": [{"type": "input_text",
                                          "text": "what is left to do?"}]}},
                {"type": "response_item", "timestamp": "2026-08-05T10:02:00Z",
                 "payload": {"type": "message", "role": "assistant",
                             "content": [{"type": "output_text",
                                          "text": "finish the migration"}]}},
            ]
            with open(path, "w", encoding="utf-8") as fh:
                for entry in entries:
                    fh.write(json.dumps(entry) + "\n")

            chunks = parse_codex_transcript(path)
            self.assertTrue(chunks, "codex fixture produced no chunks — test is inert")

            view = build_resume_view(_index(chunks), limit=10, journal_path=None)
            joined = " ".join(t.user_text + t.assistant_text for t in view.turns)
            self.assertIn("finish the migration", joined)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestTopLevelWiring(unittest.TestCase):
    """`synapt resume` is a front-door verb (ruled on recall#927, 2026-08-05).

    It routes through the same dispatcher `init` uses, so the long form
    `synapt recall resume` exists without a second code path.
    """

    def test_top_level_resume_dispatches_into_recall(self):
        import synapt.cli as top

        with mock.patch("synapt.recall.cli.main") as recall_main:
            with mock.patch.object(top.sys, "argv", ["synapt", "resume"]):
                top.main()
        recall_main.assert_called_once()

    def test_top_level_resume_forwards_its_arguments(self):
        import synapt.cli as top
        seen = {}

        def capture():
            seen["argv"] = list(top.sys.argv)

        with mock.patch("synapt.recall.cli.main", side_effect=capture):
            with mock.patch.object(top.sys, "argv",
                                   ["synapt", "resume", "abc123", "--turns", "3"]):
                top.main()
        self.assertIn("resume", seen["argv"])
        self.assertIn("abc123", seen["argv"])
        self.assertIn("--turns", seen["argv"])

    def test_resume_is_listed_in_top_level_help(self):
        import synapt.cli as top

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            top._print_help({})
        self.assertIn("resume", out.getvalue())

    def test_long_form_reaches_the_same_command(self):
        """`synapt recall resume` must reach cmd_resume, not a parallel implementation.

        Driven through the real argparse + dispatch path rather than a parser
        extracted for the test, so the wiring itself is what gets witnessed.
        """
        import synapt.recall.cli as recall_cli

        with mock.patch.object(recall_cli, "cmd_resume") as cmd:
            with mock.patch.object(recall_cli.sys, "argv",
                                   ["synapt recall", "resume", "--turns", "4"]):
                recall_cli.main()
        cmd.assert_called_once()
        self.assertEqual(cmd.call_args[0][0].turns, 4)

    def test_long_form_forwards_a_named_session(self):
        import synapt.recall.cli as recall_cli

        with mock.patch.object(recall_cli, "cmd_resume") as cmd:
            with mock.patch.object(recall_cli.sys, "argv",
                                   ["synapt recall", "resume", "abc123"]):
                recall_cli.main()
        self.assertEqual(cmd.call_args[0][0].session, "abc123")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# recall#935 / #937 — what the author's own cold run surfaced
#
# `synapt resume` shipped, and the first use of it from a genuinely empty
# context served a session two weeks stale with a journal made of harness
# markup. Three defects, and two of them masked each other: the ordering bug
# hid the fact that the newest thing in the index was a CHANNEL, so repairing
# ordering alone would have moved the default from "stale session" to "#dev",
# which looks more plausible and is more wrong.
# ---------------------------------------------------------------------------


CHANNEL_ID = "channel_dev"


def _journal_chunk(session_id: str, timestamp: str) -> TranscriptChunk:
    """A journal chunk: turn_index -1, timestamp = when the journal was WRITTEN."""
    return _chunk(session_id, -1, user_text="journal", timestamp=timestamp)


class TestSessionOrderingIgnoresJournalWriteTime(unittest.TestCase):
    """Recency must mean *activity*, not when a journal was written about it.

    Journal chunks carry write time. Indexing stamps them at build time, so a
    dead session acquires a timestamp of "now" and floats above a live one —
    which is exactly how a 2026-07-22 session won a "newest session" contract
    on 2026-08-06.
    """

    def test_journal_write_time_does_not_make_a_dead_session_newest(self):
        index = _index([
            _chunk(SESSION_A, 0, "old work", "old answer",
                   timestamp="2026-07-22T07:36:29Z"),
            _journal_chunk(SESSION_A, "2026-08-06T03:46:56Z"),  # build stamped it
            _chunk(SESSION_B, 0, "tonight's work", "tonight's answer",
                   timestamp="2026-08-06T03:14:44Z"),
        ])
        self.assertEqual(resolve_session(index, None), SESSION_B)

    def test_a_session_of_only_journal_chunks_is_still_resolvable(self):
        """Having no real turns must not make a session vanish from the index."""
        index = _index([_journal_chunk(SESSION_A, "2026-08-06T03:46:56Z")])
        self.assertEqual(resolve_session(index, None), SESSION_A)


class TestChannelsAreNotTheDefaultResumeTarget(unittest.TestCase):
    """A channel is not a session, and must never be the silent default.

    This defect was invisible until the ordering bug above was diagnosed: with
    journal chunks excluded, the newest entry in a real index was `channel_dev`.
    """

    def test_newest_channel_does_not_win_the_default(self):
        index = _index([
            _chunk(SESSION_A, 0, "real work", "real answer",
                   timestamp="2026-08-06T03:14:44Z"),
            _chunk(CHANNEL_ID, 0, "a channel post", "another post",
                   timestamp="2026-08-06T03:27:20Z"),
        ])
        self.assertEqual(resolve_session(index, None), SESSION_A)

    def test_a_channel_may_still_be_resumed_when_named_explicitly(self):
        """Asking for it by name is not the failure mode; defaulting to it is."""
        index = _index([
            _chunk(SESSION_A, 0, "real work", "real answer",
                   timestamp="2026-08-06T03:14:44Z"),
            _chunk(CHANNEL_ID, 0, "a channel post", "another post",
                   timestamp="2026-08-06T03:27:20Z"),
        ])
        self.assertEqual(resolve_session(index, CHANNEL_ID), CHANNEL_ID)

    def test_an_index_of_only_channels_says_so_rather_than_resuming_one(self):
        index = _index([
            _chunk(CHANNEL_ID, 0, "a channel post", "another post",
                   timestamp="2026-08-06T03:27:20Z"),
        ])
        with self.assertRaises(ResumeError):
            resolve_session(index, None)

    def test_channel_ids_are_rendered_in_full_so_they_are_tellable(self):
        """Truncated to eight characters, every channel renders as `channel_`."""
        index = _index([
            _chunk(CHANNEL_ID, 0, "a channel post", "another post",
                   timestamp="2026-08-06T03:27:20Z"),
        ])
        view = build_resume_view(index, session_id=CHANNEL_ID)
        self.assertIn(CHANNEL_ID, format_resume(view))


class TestStubJournalDoesNotShadowTheRealEntry(unittest.TestCase):
    """A file list is not a journal, and harness markup is not a focus.

    The auto-extractor writes a stub whose `focus` is the raw `/clear` command
    block and whose done/decisions/next_steps are empty. Because it carried a
    matching session id, it won the exact-match branch outright — so the
    "inferred" path, which exists precisely to surface rich entries whose
    session id is empty, was never reached.
    """

    def _stub(self, session_id: str) -> JournalEntry:
        return JournalEntry(
            timestamp="2026-08-06T03:00:00Z",
            session_id=session_id,
            focus="<command-name>/clear</command-name>\n<command-args></command-args>",
            done=[], decisions=[], next_steps=[],
            files_modified=["src/synapt/recall/resume.py"],
            auto=True,
        )

    def _real(self, session_id: str = "") -> JournalEntry:
        return JournalEntry(
            timestamp="2026-08-06T03:28:30Z",
            session_id=session_id,
            focus="recall#927 shipped — the session-tail verb merged",
            done=["merged the verb"],
            decisions=["strict xfail over skip"],
            next_steps=["rerun the nightly benchmark"],
            auto=False,
        )

    def _view_with(self, entries: list[JournalEntry], tmp: Path):
        path = tmp / "journal.jsonl"
        for entry in entries:
            append_entry(entry, path)
        index = _index([
            _chunk(SESSION_A, 0, "work", "answer", timestamp="2026-08-06T03:14:44Z"),
        ])
        return build_resume_view(index, journal_path=path)

    def test_a_files_only_stub_is_not_treated_as_a_journal(self):
        with tempfile.TemporaryDirectory() as td:
            view = self._view_with([self._stub(SESSION_A)], Path(td))
            self.assertIsNone(view.journal)

    def test_the_rich_unbound_entry_wins_over_a_matching_stub(self):
        with tempfile.TemporaryDirectory() as td:
            view = self._view_with(
                [self._stub(SESSION_A), self._real(session_id="")], Path(td)
            )
            self.assertIsNotNone(view.journal)
            self.assertEqual(view.journal.next_steps, ["rerun the nightly benchmark"])

    def test_a_genuinely_bound_rich_entry_still_binds(self):
        with tempfile.TemporaryDirectory() as td:
            view = self._view_with([self._real(session_id=SESSION_A)], Path(td))
            self.assertEqual(view.journal_provenance, "bound")

    def test_harness_markup_alone_is_not_intent_content(self):
        """Focus made only of a runtime control block carries no intent."""
        entry = JournalEntry(
            timestamp="2026-08-06T03:00:00Z", session_id=SESSION_A,
            focus="<command-name>/clear</command-name>",
            done=[], decisions=[], next_steps=[], auto=True,
        )
        with tempfile.TemporaryDirectory() as td:
            view = self._view_with([entry], Path(td))
            self.assertIsNone(view.journal)

    def test_an_auto_entry_is_not_labelled_as_written_by_the_session(self):
        """`auto=True` means the extractor wrote it, not the session."""
        entry = self._real(session_id=SESSION_A)
        entry.auto = True
        with tempfile.TemporaryDirectory() as td:
            view = self._view_with([entry], Path(td))
            self.assertIsNotNone(view.journal)
            self.assertNotIn("written by this session", format_resume(view))


class TestTimestampFormatDoesNotDecideOrdering(unittest.TestCase):
    """Identical instants must not order by spelling.

    Timestamps are compared as strings, and the index mixes `Z` with `+00:00`.
    `'Z'` (0x5A) sorts above `'+'` (0x2B), so the same moment written two ways
    compares unequal — a latent ordering landmine as `+00:00` writers grow.
    """

    def test_a_later_offset_timestamp_beats_an_earlier_z_timestamp(self):
        """The discriminating case, not a comfortable one.

        ``2026-08-06T03:00:00.500000+00:00`` is half a second AFTER
        ``2026-08-06T03:00:00Z``. Compared as text the fractional form loses,
        because ``'Z'`` (0x5A) outranks the ``'.'`` (0x2E) that begins the
        fraction — so the earlier moment wins on spelling alone.
        """
        index = _index([
            _chunk(SESSION_A, 0, "a", "b",
                   timestamp="2026-08-06T03:00:00.500000+00:00"),
            _chunk(SESSION_B, 0, "c", "d", timestamp="2026-08-06T03:00:00Z"),
        ])
        self.assertEqual(resolve_session(index, None), SESSION_A)


# ---------------------------------------------------------------------------
# Unclean end — a crash writes neither a journal nor a SessionEnd checkpoint
# ---------------------------------------------------------------------------


class TestUncleanEnd(unittest.TestCase):
    """A session that dies by host crash leaves no handoff of any kind.

    Measured 2026-08-31: the coordinator's session (last activity 12:06Z) had
    no journal after 04:54Z and no SessionEnd checkpoint, because a crash runs
    no SessionEnd. The on-disk checkpoint belonged to a DIFFERENT session (a
    subagent that ended cleanly at 11:58Z), and the wake rendered that tail
    under "LAST CHECKPOINT" as if it were the bridge. Nothing said that seven
    hours of work had no record.

    Atlas's r1 on the first version added the composition rule pinned here: a
    handoff is session-bound evidence. A journal from another session, however
    recent, certifies nothing about this one.
    """

    LAST = "2026-08-31T12:06:07Z"
    NOW = _timestamp_epoch("2026-08-31T13:10:24Z")

    @staticmethod
    def _source(session_id: str, latest: str) -> CallerTranscript:
        return CallerTranscript(session_id, Path(f"/t/{session_id[:8]}.jsonl"), 0.0, 10,
                                latest_timestamp=latest)

    @staticmethod
    def _journal(ts: str, session_id: str = "") -> JournalEntry:
        return JournalEntry(timestamp=ts, session_id=session_id, focus="handoff")

    def _judge(self, sources, *, checkpoint=None, journals=(), **kw):
        return detect_unclean_end(sources, checkpoint=checkpoint,
                                  authored_journals=list(journals), **kw)

    def test_no_previous_transcript_is_not_a_finding(self):
        self.assertIsNone(self._judge([]))

    def test_a_checkpoint_for_this_session_means_it_was_handed_off(self):
        found = self._judge([self._source(SESSION_A, self.LAST)],
                            checkpoint={"session_id": SESSION_A, "captured_at": self.LAST})
        self.assertIsNone(found)

    def test_a_checkpoint_from_another_session_does_not_cover_this_one(self):
        found = self._judge([self._source(SESSION_A, self.LAST)],
                            checkpoint={"session_id": SESSION_B, "captured_at": "2026-08-31T11:58:20Z"})
        self.assertIsNotNone(found)
        self.assertEqual(found.session_id, SESSION_A)
        self.assertEqual(found.last_activity, self.LAST)
        self.assertEqual(found.checkpoint_session, SESSION_B)
        self.assertIsNone(found.last_authored_journal)
        self.assertIsNone(found.gap_seconds)

    def test_a_bound_journal_inside_the_grace_window_covers_the_session(self):
        found = self._judge([self._source(SESSION_A, self.LAST)],
                            journals=[self._journal("2026-08-31T12:00:00Z", SESSION_A)])
        self.assertIsNone(found)

    def test_a_bound_journal_older_than_the_grace_window_does_not(self):
        found = self._judge([self._source(SESSION_A, self.LAST)],
                            journals=[self._journal("2026-08-31T04:54:00Z", SESSION_A)])
        self.assertIsNotNone(found)
        self.assertEqual(found.last_authored_journal, "2026-08-31T04:54:00Z")
        self.assertEqual(found.gap_seconds, 7 * 3600 + 12 * 60 + 7)
        self.assertIsNone(found.checkpoint_session)

    def test_a_later_journal_from_another_session_does_not_suppress_the_finding(self):
        """Atlas's reproducer: crash at 12:06Z, foreign journal at 13:00Z.
        The first version reduced journals to a timestamp and returned None."""
        found = self._judge(
            [self._source(SESSION_A, self.LAST)],
            checkpoint={"session_id": SESSION_B, "captured_at": "2026-08-31T11:58:20Z"},
            journals=[self._journal("2026-08-31T13:00:00Z", SESSION_B)],
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.session_id, SESSION_A)
        self.assertIsNone(found.last_authored_journal)
        self.assertEqual(found.foreign_journal, "2026-08-31T13:00:00Z")

    def test_a_later_journal_bound_to_the_session_does_cover_it(self):
        """Positive control for the test above: same time, own session."""
        found = self._judge(
            [self._source(SESSION_A, self.LAST)],
            checkpoint={"session_id": SESSION_B, "captured_at": "2026-08-31T11:58:20Z"},
            journals=[self._journal("2026-08-31T13:00:00Z", SESSION_A)],
        )
        self.assertIsNone(found)

    def test_a_sessionless_legacy_journal_covers_only_inside_the_symmetric_window(self):
        """The explicit fallback for entries that name no session, witnessed
        separately: inside the window it covers, an hour later it does not."""
        inside = self._judge([self._source(SESSION_A, self.LAST)],
                             journals=[self._journal("2026-08-31T12:10:00Z")])
        later = self._judge([self._source(SESSION_A, self.LAST)],
                            journals=[self._journal("2026-08-31T13:00:00Z")])
        self.assertIsNone(inside)
        self.assertIsNotNone(later)
        self.assertEqual(later.last_authored_journal, "2026-08-31T13:00:00Z")
        self.assertIsNone(later.gap_seconds)

    def test_the_grace_boundary_is_inclusive_and_discriminating(self):
        """gap == grace is covered; gap == grace + 1 s is not. A pair, so a
        mutation that drops the comparison turns exactly one side red."""
        sources = [self._source(SESSION_A, "2026-08-31T12:15:00Z")]
        at_grace = self._judge(sources, journals=[self._journal("2026-08-31T12:00:00Z", SESSION_A)],
                               grace_seconds=900)
        past_grace = self._judge(sources, journals=[self._journal("2026-08-31T11:59:59Z", SESSION_A)],
                                 grace_seconds=900)
        self.assertIsNone(at_grace)
        self.assertIsNotNone(past_grace)
        self.assertEqual(past_grace.gap_seconds, 901)

    def test_the_current_session_is_excluded_so_the_previous_one_is_judged(self):
        """At SessionStart the newest transcript is the session that is
        starting. Without the exclusion every wake would report itself."""
        found = self._judge(
            [self._source(SESSION_A, "2026-08-31T12:30:00Z"), self._source(SESSION_B, self.LAST)],
            exclude_session_id=SESSION_A,
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.session_id, SESSION_B)

    def test_a_crash_followed_by_a_fast_restart_is_still_judged(self):
        """Recency is not liveness evidence (Atlas, r1): a session that
        crashed seconds before this wake must not be treated as live. Only
        identity excludes."""
        found = self._judge(
            [self._source(SESSION_A, "2026-08-31T13:10:00Z"),  # 24 s before the wake
             self._source(SESSION_B, self.LAST)],
            exclude_session_id=SESSION_B,
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.session_id, SESSION_A)

    def test_a_source_without_a_timestamp_cannot_be_the_newest(self):
        found = self._judge([self._source(SESSION_A, ""), self._source(SESSION_B, self.LAST)])
        self.assertEqual(found.session_id, SESSION_B)

    def test_format_resume_names_an_unclean_end_in_the_header(self):
        index = _index([_chunk(SESSION_A, 0, "last question", "")])
        view = build_resume_view(index, session_id=SESSION_A, journal_path=None)
        self.assertNotIn("UNCLEAN END", format_resume(view))
        view.unclean_end = UncleanEnd(
            session_id=SESSION_A, transcript_path=Path("/t/a.jsonl"),
            last_activity=self.LAST, last_authored_journal="2026-08-31T04:54:00Z",
            gap_seconds=25927.0, checkpoint_session=None,
        )
        text = format_resume(view)
        self.assertIn("UNCLEAN END", text.splitlines()[0])
        self.assertIn("7h12m", text.splitlines()[0])
        self.assertIn("no SessionEnd checkpoint", text.splitlines()[0])

    def test_the_foreign_journal_wording_says_later_only_when_it_is(self):
        from synapt.recall.resume import format_unclean_end
        base = dict(session_id=SESSION_A, transcript_path=Path("/t/a.jsonl"),
                    last_activity=self.LAST, last_authored_journal=None,
                    gap_seconds=None, checkpoint_session=None)
        after = format_unclean_end(UncleanEnd(**base, foreign_journal="2026-08-31T13:00:00Z"), None)
        before = format_unclean_end(UncleanEnd(**base, foreign_journal="2026-08-31T11:00:00Z"), None)
        self.assertIn("A later journal at 2026-08-31T13:00:00Z", after)
        self.assertIn("A journal at 2026-08-31T11:00:00Z", before)
        self.assertNotIn("later", before)
