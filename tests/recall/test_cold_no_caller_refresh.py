"""Cold no-caller resume refresh (durable-checkpoint follow-on).

On a cold cross-runtime launch there is no caller transcript, so the caller-tail
refresh has nothing to act on and the shared index can be months stale. This
feature does an incremental refresh of the newest source before render, and
labels what it refreshed. The load-bearing guarantees, each with a witness here:

  * the read NEVER WAITS on a held build lock: a held lock (recall#1018 ghost
    lock included) builds NOTHING and returns immediately (control). The
    free-lock build is synchronous and delays the render by its own duration --
    it never waits on ANOTHER holder, but is not free on the success path;
  * it NEVER RAISES: every refresh failure (discovery, data-dir, lock acquire,
    build, lock release) degrades to the stale render;
  * the refresh builds into the SAME store the load reads, not a cwd-derived
    secondary one (this change's R2 store split);
  * a freshened render is legible as freshened: the label names the source and
    the index cursor before -> after.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from synapt.recall import cli
from synapt.recall.cli import ColdRefreshOutcome, cold_no_caller_refresh
from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.freshness import IndexFreshness
from synapt.recall.resume import ResumeView, ResumeTurn, format_resume

from _isolation_helpers import owned_store


def _patches(*, lock_fd, build_side_effect=None, ts_before="2026-06-01T00:00:00", ts_after="2026-09-01T00:00:00",
             newest=Path("/w/.codex/sessions/2026/09/01/rollout-a.jsonl")):
    """Patch the three seams cold_no_caller_refresh touches. build_calls records
    whether the real build ran, so the ghost-lock control can assert it did not."""
    build_calls = []

    def fake_build(*a, **k):
        build_calls.append((a, k))
        if build_side_effect:
            raise build_side_effect
        return None

    fresh = iter([mock.Mock(build_timestamp=ts_before), mock.Mock(build_timestamp=ts_after)])
    cms = [
        mock.patch.object(cli, "_newest_source_file", return_value=newest),
        mock.patch.object(cli, "_acquire_build_lock", return_value=lock_fd),
        mock.patch.object(cli, "_release_build_lock"),
        mock.patch.object(cli, "_archive_and_build_locked", side_effect=fake_build),
        # The STORE is derived from index_dir (the load target) and its data dir
        # mocked to a pytest-owned path so no real store is touched (this change's R2
        # store split); the cwd SOURCE is threaded separately.
        mock.patch.object(cli, "project_data_dir", return_value=Path("/tmp/x")),
        mock.patch("synapt.recall.freshness.check_index_freshness",
                   side_effect=lambda *a, **k: next(fresh)),
    ]
    return cms, build_calls


class TestColdNoCallerRefresh(unittest.TestCase):
    def _run(self, cms):
        for c in cms:
            c.start()
        self.addCleanup(mock.patch.stopall)
        return cold_no_caller_refresh(Path("/proj"), Path("/proj/.synapt/recall/index"))

    def test_refreshed_advances_and_reports_cursor(self):
        cms, build_calls = _patches(lock_fd=7)
        out = self._run(cms)
        self.assertTrue(out.refreshed)
        self.assertEqual(out.reason, "refreshed")
        self.assertEqual(out.source, "rollout-a.jsonl")
        self.assertEqual(out.cursor_before, "2026-06-01T00:00:00")
        self.assertEqual(out.cursor_after, "2026-09-01T00:00:00")
        self.assertEqual(len(build_calls), 1)  # the build ran

    def test_a_real_build_names_both_store_and_source_distinctly_on_stderr(self):
        # recall#1123 hardening: even the agreeing case (the only one that
        # reaches the build today, cmd_resume's own refusal having ruled out
        # the disagreeing one) leaves a legible trail of which store was
        # rebuilt from which source. store_root and source_dir can differ in
        # cold_no_caller_refresh's own signature (see TestColdRefreshStoreSplit
        # above), so this fixture uses genuinely different paths to prove the
        # line names BOTH, not one value printed twice.
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            store = tmp / "canonical"
            source = tmp / "agent-desk"
            (store / ".synapt" / "recall" / "index").mkdir(parents=True)
            source.mkdir()
            index_dir = store / ".synapt" / "recall" / "index"

            fresh = iter([mock.Mock(build_timestamp="old"), mock.Mock(build_timestamp="new")])
            with mock.patch.object(cli, "_newest_source_file",
                                   return_value=source / ".codex" / "rollout.jsonl"), \
                 mock.patch.object(cli, "_acquire_build_lock", return_value=7), \
                 mock.patch.object(cli, "_release_build_lock"), \
                 mock.patch.object(cli, "_archive_and_build_locked"), \
                 mock.patch.object(cli, "project_data_dir", return_value=store), \
                 mock.patch("synapt.recall.freshness.check_index_freshness",
                            side_effect=lambda *a, **k: next(fresh)):
                captured = io.StringIO()
                with mock.patch("sys.stderr", captured):
                    cold_no_caller_refresh(source, index_dir)

        self.assertIn("refreshing store", captured.getvalue())
        self.assertIn(str(store.resolve()), captured.getvalue())
        self.assertIn(str(source), captured.getvalue())

    def test_ghost_lock_and_no_source_print_nothing(self):
        # Negative control: the print sits AFTER lock acquisition succeeds,
        # so paths that never get that far (held lock, nothing to refresh)
        # must stay silent -- proves the positive test above is reading the
        # NEW line, not some unrelated print elsewhere in the function.
        import io

        for lock_fd, newest in ((None, Path("/w/rollout.jsonl")), (7, None)):
            with self.subTest(lock_fd=lock_fd, newest=newest):
                cms, _ = _patches(lock_fd=lock_fd, newest=newest)
                captured = io.StringIO()
                with mock.patch("sys.stderr", captured):
                    self._run(cms)
                self.assertEqual(captured.getvalue(), "")

    def test_ghost_lock_control_builds_nothing_and_waits_for_nothing(self):
        # recall#1018: lock held (possibly by a dead holder). The read must not
        # build and must not wait. This is THE control.
        cms, build_calls = _patches(lock_fd=None)
        out = self._run(cms)
        self.assertFalse(out.refreshed)
        self.assertEqual(out.reason, "lock_held")
        self.assertEqual(build_calls, [])  # nothing built under a held lock
        self.assertEqual(out.cursor_before, out.cursor_after)  # index untouched

    def test_up_to_date_when_build_ran_but_cursor_unchanged(self):
        cms, build_calls = _patches(lock_fd=7, ts_before="2026-09-01T00:00:00",
                                    ts_after="2026-09-01T00:00:00")
        out = self._run(cms)
        self.assertFalse(out.refreshed)
        self.assertEqual(out.reason, "up_to_date")
        self.assertEqual(len(build_calls), 1)

    def test_no_source_skips_lock_and_build(self):
        cms, build_calls = _patches(lock_fd=7, newest=None)
        out = self._run(cms)
        self.assertEqual(out.reason, "no_source")
        self.assertFalse(out.refreshed)
        self.assertEqual(build_calls, [])

    def test_build_error_degrades_without_raising_and_releases_lock(self):
        cms, build_calls = _patches(lock_fd=7, build_side_effect=RuntimeError("boom"))
        with mock.patch.object(cli, "_release_build_lock") as rel:
            for c in cms:
                if getattr(c, "attribute", None) == "_release_build_lock":
                    continue  # use our own release spy
                c.start()
            self.addCleanup(mock.patch.stopall)
            out = cold_no_caller_refresh(Path("/proj"), Path("/proj/.synapt/recall/index"))
            self.assertEqual(out.reason, "error")
            self.assertFalse(out.refreshed)
            rel.assert_called_once()  # lock released even on build failure

    def test_discovery_oserror_degrades_to_error_before_lock(self):
        # this change's R2 (Atlas): source discovery walks the filesystem and can
        # itself raise OSError (denied root, racing unlink). That is BEFORE the
        # lock, so it must be caught into an error outcome, never escape — and
        # the lock/build must not run.
        with mock.patch.object(cli, "_newest_source_file", side_effect=OSError("denied")), \
             mock.patch.object(cli, "_acquire_build_lock") as lock, \
             mock.patch.object(cli, "_archive_and_build_locked") as build, \
             mock.patch.object(cli, "project_data_dir", return_value=Path("/tmp/x")), \
             mock.patch("synapt.recall.freshness.check_index_freshness",
                        return_value=mock.Mock(build_timestamp="2026-06-01T00:00:00")):
            out = cold_no_caller_refresh(Path("/proj"), Path("/proj/.synapt/recall/index"))
        self.assertEqual(out.reason, "error")
        self.assertFalse(out.refreshed)
        lock.assert_not_called()   # degraded BEFORE lock acquisition
        build.assert_not_called()  # nothing built

    def test_acquire_oserror_degrades_to_error_without_raising(self):
        # this change's R2 (Atlas): resolving the data dir and acquiring the lock
        # touch the filesystem (mkdir + os.open on the lock parent). A lock parent
        # that is a regular file raises NotADirectoryError out of the REAL
        # _acquire_build_lock (no mock). It sat outside the degradation boundary
        # and escaped cmd_resume; it must degrade to an error outcome, never raise,
        # and build nothing.
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            notadir = Path(td) / "notadir"
            notadir.write_text("i am a file, not a directory")
            data_dir = notadir / "data"  # its parent is a file -> mkdir raises ENOTDIR
            with mock.patch.object(cli, "_newest_source_file",
                                   return_value=Path("/w/.codex/sessions/2026/09/01/rollout-a.jsonl")), \
                 mock.patch.object(cli, "project_data_dir", return_value=data_dir), \
                 mock.patch.object(cli, "_archive_and_build_locked") as build, \
                 mock.patch("synapt.recall.freshness.check_index_freshness",
                            return_value=mock.Mock(build_timestamp="2026-06-01T00:00:00")):
                out = cold_no_caller_refresh(Path("/proj"), Path("/proj/.synapt/recall/index"))
        self.assertEqual(out.reason, "error")   # degraded, did not escape
        self.assertFalse(out.refreshed)
        self.assertEqual(out.source, "rollout-a.jsonl")  # got PAST discovery
        build.assert_not_called()  # never reached the build

    def test_release_oserror_does_not_override_a_successful_read(self):
        # this change's R2 (Atlas): _release_build_lock runs in a finally and can
        # raise OSError (a closed/invalid fd, an unlink on a vanished lock file).
        # In a finally that would OVERRIDE the outcome: a release failure after a
        # SUCCESSFUL build must still be a successful refresh, not an escaped
        # exception. Swallow it; keep the read's outcome.
        cms, build_calls = _patches(lock_fd=7)  # build succeeds, cursor advances
        for c in cms:
            if getattr(c, "attribute", None) == "_release_build_lock":
                continue  # replace the no-op release with a raising one
            c.start()
        self.addCleanup(mock.patch.stopall)
        with mock.patch.object(cli, "_release_build_lock", side_effect=OSError("bad fd")):
            out = cold_no_caller_refresh(Path("/proj"), Path("/proj/.synapt/recall/index"))
        self.assertTrue(out.refreshed)          # success survived the release error
        self.assertEqual(out.reason, "refreshed")
        self.assertEqual(len(build_calls), 1)   # the build ran

    def test_release_oserror_does_not_override_a_build_error(self):
        # The other half of the finally-override hazard: when the build itself
        # failed, a release OSError must not replace that build-error outcome with
        # an escaped exception. reason stays "error", nothing raises.
        cms, build_calls = _patches(lock_fd=7, build_side_effect=RuntimeError("boom"))
        for c in cms:
            if getattr(c, "attribute", None) == "_release_build_lock":
                continue
            c.start()
        self.addCleanup(mock.patch.stopall)
        with mock.patch.object(cli, "_release_build_lock", side_effect=OSError("bad fd")):
            out = cold_no_caller_refresh(Path("/proj"), Path("/proj/.synapt/recall/index"))
        self.assertEqual(out.reason, "error")   # build error survived, did not escape
        self.assertFalse(out.refreshed)
        self.assertEqual(len(build_calls), 1)   # the build was attempted


class TestColdRefreshStoreSplit(unittest.TestCase):
    """this change's R2 (Atlas): resume LOADS the index from the GRIPSPACE_ROOT
    store, but the refresh used to resolve data_dir/build from cwd. On a spawned
    desk whose cwd is a filesystem SIBLING of GRIPSPACE_ROOT, that locked/built a
    cwd-derived SECONDARY store and reloaded the untouched stale one, leaving the
    durable-checkpoint fruit stale. The refresh must target the SAME store the load reads.

    Real GRIPSPACE_ROOT env and a real build lock (only the heavy build is
    spied), so the store paths are exercised and asserted ON DISK.
    """

    def test_refresh_targets_gripspace_root_store_source_stays_cwd(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            a = tmp / "canonical"      # GRIPSPACE_ROOT store (A) — what load reads
            b = tmp / "agent-desk"     # cwd, a filesystem SIBLING of A (B)
            (a / ".synapt" / "recall" / "index").mkdir(parents=True)
            b.mkdir()
            index_dir = a / ".synapt" / "recall" / "index"

            captured: dict[str, Path | None] = {}

            def spy_build(project_dir, source_dirs, **kw):
                captured["store"] = Path(project_dir)
                captured["source"] = kw.get("source_dir")
                return None

            with mock.patch.dict(os.environ, {"GRIPSPACE_ROOT": str(a)}, clear=False), \
                 mock.patch.object(cli, "_newest_source_file",
                                   return_value=b / ".codex" / "rollout-x.jsonl"), \
                 mock.patch.object(cli, "_archive_and_build_locked", side_effect=spy_build), \
                 mock.patch("synapt.recall.freshness.check_index_freshness",
                            return_value=mock.Mock(build_timestamp="old")):
                os.environ.pop("SYNAPT_RECALL_ROOT", None)
                out = cold_no_caller_refresh(b, index_dir)  # real lock + real release

            self.assertNotEqual(out.reason, "error")            # did not degrade/escape
            # project_root/project_data_dir resolve() the path (macOS /var -> /private/var)
            self.assertEqual(captured["store"], a.resolve())    # BUILD store == A (GRIPSPACE_ROOT)
            self.assertEqual(captured["source"], b)             # SOURCE == cwd (B), threaded as-is
            self.assertFalse((b / ".synapt").exists())          # no cwd-derived secondary store
            self.assertTrue((a / ".synapt" / "recall").is_dir())  # the real lock touched A


class TestResumeRefusesIndexSourceGripspaceMismatch(unittest.TestCase):
    """recall#1123: TestColdRefreshStoreSplit above proves cold_no_caller_refresh
    correctly targets whichever store it is TOLD to target -- that is right for
    a function that trusts its caller. The incident this closes is one layer up:
    a bare `synapt resume` with a stale GRIPSPACE_ROOT (left over from another
    desk) resolves index_dir from THAT env var while source stays cwd, so
    cmd_resume handed cold_no_caller_refresh two paths naming DIFFERENT
    workspaces -- and cold_no_caller_refresh, trusting its caller as designed,
    took the REAL other workspace's build.lock and rebuilt its index using this
    cwd's content. Real GRIPSPACE_ROOT env and two real on-disk workspaces (not
    mocked): cmd_resume must refuse before anything reaches the lock.

    v3 (Stromus's R2 on v2, m_4422aa09): the check below is a DISCRIMINATOR
    (is source_dir's own gripspace populated?) and not a COMPARISON
    (does source_root equal index_root?) -- every real agent desk on this
    host is a filesystem sibling of the shared GRIPSPACE_ROOT team root, so
    root inequality alone is the normal, correct shape for a legitimate
    desk, not a defect. test_sibling_desk_with_a_registered_repo_proceeds
    below is the witness the comparison-shaped v2 lacked; the two tests
    after it are the unpopulated-scratch and same-root negative/positive
    controls the discriminator must still get right.
    """

    def _cmd_resume_args(self):
        import argparse
        return argparse.Namespace(session=None, index=None, turns=10)

    def _build_real_loadable_index(self, index_dir: Path) -> None:
        """A genuinely loadable index (SQLite-backed, per test_resume.py's own
        _save_sqlite_index pattern) -- not an empty stub. Without this, resume
        would crash on schema errors before ever reaching caller_transcripts /
        the freshness check / cold_no_caller_refresh, and the RED-before-fix
        run below would prove nothing about the actual incident path."""
        from synapt.recall.storage import RecallDB

        index_dir.mkdir(parents=True, exist_ok=True)
        chunk = TranscriptChunk(
            id="fake0000:t0", session_id="fake0000", timestamp="2026-01-01T00:00:00Z",
            turn_index=0, user_text="hi", assistant_text="hello",
        )
        db = RecallDB(index_dir / "recall.db")
        try:
            TranscriptIndex([chunk], use_embeddings=False, db=db).save(index_dir)
        finally:
            db.close()

    def test_ambient_mismatch_refuses_before_any_lock_is_taken(self):
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            a = tmp / "canonical"       # GRIPSPACE_ROOT store -- the REAL other workspace
            b = tmp / "agent-desk"      # cwd, a filesystem SIBLING of A
            index_dir = a / ".synapt" / "recall" / "index"
            self._build_real_loadable_index(index_dir)
            b.mkdir()

            orig_cwd = Path.cwd()
            os.chdir(b)
            captured = io.StringIO()
            try:
                stale_verdict = IndexFreshness(
                    stale=True, build_timestamp="old", scanned="archive+sources"
                )
                with mock.patch.dict(os.environ, {"GRIPSPACE_ROOT": str(a)}, clear=False), \
                     mock.patch("synapt.recall.resume.caller_transcripts", return_value=[]), \
                     mock.patch(
                        "synapt.recall.freshness.check_index_freshness",
                        return_value=stale_verdict,
                     ), \
                     mock.patch.object(
                        cli, "_newest_source_file", return_value=b / "fake-rollout.jsonl"
                     ), \
                     mock.patch.object(cli, "_archive_and_build_locked") as build, \
                     mock.patch.object(cli, "_acquire_build_lock", return_value=999) as lock, \
                     mock.patch.object(cli, "_release_build_lock") as release, \
                     mock.patch("sys.stderr", captured):
                    os.environ.pop("SYNAPT_RECALL_ROOT", None)
                    with self.assertRaises(SystemExit) as ctx:
                        cli.cmd_resume(self._cmd_resume_args())
            finally:
                os.chdir(orig_cwd)

            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("disagree on which workspace", captured.getvalue())
            # RESOLVED, not raw: the message now prints source_dir.resolve() /
            # index_dir.resolve() consistently (recall#1136's fix), so on
            # Windows the raw str(b)/str(a) can be a SHORT (8.3) form no
            # longer present anywhere in a message that is now consistently
            # LONG-form -- checking the raw form here is exactly the bug this
            # follow-up corrects.
            self.assertIn(str(b.resolve()), captured.getvalue())  # names the source path
            self.assertIn(str(a.resolve()), captured.getvalue())  # names the index path
            lock.assert_not_called()   # never reached the lock
            build.assert_not_called()  # never built
            self.assertFalse(
                (a / ".synapt" / "recall" / "build.lock").exists(),
                "the real other workspace's build.lock must never be created",
            )

    def test_sibling_desk_with_a_registered_repo_proceeds(self):
        # v3's new witness: cwd (b) is a POPULATED gripspace -- it has a
        # registered child repo, the same shape a real `gr spawn`/`gr repo
        # add` desk always has (recall#1124's own positive control uses the
        # identical fixture shape). GRIPSPACE_ROOT (a) still names a
        # DIFFERENT workspace -- root inequality persists exactly as in the
        # scratch-dir mismatch test above -- but a populated source must
        # PROCEED, not refuse: this is the sibling-desk binding every real
        # desk on this host relies on. Real dirs, no mocking of the
        # discriminator itself, so a mutation that drops the populated
        # check has nowhere to hide.
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            a = tmp / "canonical"       # GRIPSPACE_ROOT -- a DIFFERENT, real workspace
            b = tmp / "agent-desk"      # cwd, populated (registered child repo)
            a.mkdir()
            b.mkdir()
            (b / "repo" / ".git").mkdir(parents=True)

            orig_cwd = Path.cwd()
            os.chdir(b)
            captured = io.StringIO()
            try:
                with mock.patch.dict(os.environ, {"GRIPSPACE_ROOT": str(a)}, clear=False), \
                     mock.patch("sys.stderr", captured):
                    os.environ.pop("SYNAPT_RECALL_ROOT", None)
                    with self.assertRaises(SystemExit) as ctx:
                        cli.cmd_resume(self._cmd_resume_args())
            finally:
                os.chdir(orig_cwd)

            # Reaches the SAME downstream outcome as the agreeing-root case
            # below ("no index found" against a, since a's real index was
            # never built) -- proving the refusal did not fire, not merely
            # that some other error happened to also exit 1.
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("no index found", captured.getvalue())
            self.assertNotIn("disagree on which workspace", captured.getvalue())

    def test_explicit_index_flag_bypasses_the_refusal(self):
        # The fix this refusal points a caller at: passing --index explicitly
        # is a deliberate choice recall already treats as suppressing ambient
        # resolution elsewhere (project_data_dir's own env-override rule).
        # Both this test and the mismatch test above end in the SAME exit
        # code (1), from two DIFFERENT checks -- so the message on stderr,
        # not the exit code, is what proves which one fired.
        import argparse
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            a = tmp / "canonical"
            b = tmp / "agent-desk"
            (a / ".synapt" / "recall" / "index").mkdir(parents=True)
            b.mkdir()

            orig_cwd = Path.cwd()
            os.chdir(b)
            captured = io.StringIO()
            try:
                args = argparse.Namespace(
                    session=None, index=str(a / ".synapt" / "recall" / "index"), turns=10
                )
                with mock.patch("sys.stderr", captured):
                    # Real function, no mocking: an explicit --index skips the
                    # mismatch check entirely and falls through to "no index
                    # found" (a real .db was never written under it) -- that
                    # SystemExit(1) is expected; it is the OTHER check.
                    with self.assertRaises(SystemExit) as ctx:
                        cli.cmd_resume(args)
            finally:
                os.chdir(orig_cwd)

            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("no index found", captured.getvalue())
            self.assertNotIn("disagree on which workspace", captured.getvalue())

    def test_agreeing_ambient_index_and_source_do_not_refuse(self):
        # Negative control for the mismatch test above: SAME GRIPSPACE_ROOT env
        # mechanism, but cwd IS the GRIPSPACE_ROOT workspace, so index and
        # source name the SAME root. Must proceed past the refusal to the
        # "no index found" branch -- same exit code as the mismatch test,
        # disambiguated here by the stderr message, not the code.
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            a = tmp / "canonical"
            a.mkdir()

            orig_cwd = Path.cwd()
            os.chdir(a)
            captured = io.StringIO()
            try:
                with mock.patch.dict(os.environ, {"GRIPSPACE_ROOT": str(a)}, clear=False), \
                     mock.patch("sys.stderr", captured):
                    os.environ.pop("SYNAPT_RECALL_ROOT", None)
                    with self.assertRaises(SystemExit) as ctx:
                        cli.cmd_resume(self._cmd_resume_args())
            finally:
                os.chdir(orig_cwd)

            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("no index found", captured.getvalue())
            self.assertNotIn("disagree on which workspace", captured.getvalue())

    def test_message_names_both_paths_in_their_resolved_form(self):
        # A portable witness for the Windows short/long-name incident: on a
        # windows-latest CI runner, an UNRESOLVED path can render in the short
        # 8.3 form (e.g. RUNNER~1) while its RESOLVED sibling renders long
        # (runneradmin) -- source_root/index_root are already resolved
        # (project_data_dir / _index_gripspace_root each call .resolve()
        # internally), so printing the raw source_dir/index_dir arguments
        # beside them made the SAME error message spell the SAME directory two
        # different ways on the SAME line, and a test asserting the raw
        # (unresolved) string was a substring failed deterministically on
        # every windows-latest job. A symlink reproduces the identical
        # SHAPE cross-platform: the raw path string differs from its
        # os-canonical form, on every OS this suite runs on.
        import io

        with self._tmp_symlink_world() as (index_dir, source_link):
            captured = io.StringIO()
            args = self._cmd_resume_args()
            with mock.patch("sys.stderr", captured):
                with self.assertRaises(SystemExit) as ctx:
                    cli._refuse_if_index_disagrees_with_source(args, index_dir, source_link)

            self.assertEqual(ctx.exception.code, 1)
            message = captured.getvalue()
            self.assertIn(f"source (cwd):     {source_link.resolve()}", message)
            self.assertIn(f"index (ambient):  {index_dir.resolve()}", message)

    def _tmp_symlink_world(self):
        import contextlib
        import tempfile

        @contextlib.contextmanager
        def _make():
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                real_index_root = tmp / "real-index-root"
                index_link = tmp / "index-alias"
                index_link.symlink_to(real_index_root, target_is_directory=True)
                self._build_real_loadable_index(real_index_root / ".synapt" / "recall" / "index")

                real_source_target = tmp / "real-source-target"
                real_source_target.mkdir()
                source_link = tmp / "source-alias"
                source_link.symlink_to(real_source_target, target_is_directory=True)

                yield index_link / ".synapt" / "recall" / "index", source_link

        return _make()


class TestNewestSourceProjectScope(unittest.TestCase):
    """this change's R1 (Sentinel): the Codex sessions dir holds EVERY project's
    rollouts. _newest_source_file must select only rollouts that belong to this
    project — an unrelated newer rollout must not be picked (which would trigger a
    build for a project that owns no sources). Real filesystem, real filter.

    Ref #967: _newest_source_file now resolves the archive/store roots it
    scans (all_worktree_archive_dirs, project_archive_dir) with an ambient
    project_dir=None rather than the tmp `project` fixture dir these tests
    pass to cli._newest_source_file, so without an owned store this class
    escaped into whichever real gripspace the test process happened to run
    from -- exactly the isolation break this ref is about, one level up.
    """

    def setUp(self):
        self._store = owned_store()

    def tearDown(self):
        self._store.restore()

    def _rollout(self, codex_dir, name, cwd, mtime):
        import json
        import os
        day = codex_dir / "2026" / "09" / "01"
        day.mkdir(parents=True, exist_ok=True)
        p = day / name
        meta = {"type": "session_meta", "payload": {"cwd": str(cwd)}}
        p.write_text(json.dumps(meta) + "\n", encoding="utf-8")
        os.utime(p, (mtime, mtime))
        return p

    def test_unrelated_newer_rollout_is_not_selected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "myproject"
            project.mkdir()
            other = base / "otherproject"
            other.mkdir()
            codex = base / ".codex" / "sessions"
            codex.mkdir(parents=True)
            # An unrelated, NEWER rollout for a different project.
            self._rollout(codex, "rollout-other.jsonl", other, mtime=2_000_000_000)
            with mock.patch("synapt.recall.codex.discover_codex_sessions", return_value=codex):
                got = cli._newest_source_file(project)
            self.assertIsNone(got)  # project owns nothing; the unrelated file is out of scope

    def test_owned_older_rollout_beats_unrelated_newer(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "myproject"
            project.mkdir()
            other = base / "otherproject"
            other.mkdir()
            codex = base / ".codex" / "sessions"
            codex.mkdir(parents=True)
            self._rollout(codex, "rollout-other.jsonl", other, mtime=2_000_000_000)  # newer, unrelated
            owned = self._rollout(codex, "rollout-mine.jsonl", project, mtime=1_000_000_000)  # older, owned
            with mock.patch("synapt.recall.codex.discover_codex_sessions", return_value=codex):
                got = cli._newest_source_file(project)
            self.assertEqual(got, owned)  # owned wins despite being older; scope, not mtime alone

    def test_unrelated_rollout_yields_no_source_no_lock_no_build(self):
        # The full contract Sentinel named: an unrelated newer rollout must NOT
        # drive cold_no_caller_refresh into a build. no_source, no lock, no build.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "myproject"
            project.mkdir()
            other = base / "otherproject"
            other.mkdir()
            codex = base / ".codex" / "sessions"
            codex.mkdir(parents=True)
            self._rollout(codex, "rollout-other.jsonl", other, mtime=2_000_000_000)
            with mock.patch("synapt.recall.codex.discover_codex_sessions", return_value=codex), \
                 mock.patch.object(cli, "_acquire_build_lock") as lock, \
                 mock.patch.object(cli, "_archive_and_build_locked") as build:
                out = cold_no_caller_refresh(project, project / "index")
        self.assertEqual(out.reason, "no_source")
        self.assertFalse(out.refreshed)
        lock.assert_not_called()   # never reached the lock
        build.assert_not_called()  # never built for a project it does not own


class TestNewestSourceRespectsEnvStoreOverride(unittest.TestCase):
    """Ref #967, the actual mechanism: _newest_source_file's archive/store
    lookups (all_worktree_archive_dirs, project_archive_dir) used to thread
    the passed project_dir straight into project_data_dir, which treats ANY
    passed project_dir as a deliberate override that suppresses
    SYNAPT_RECALL_ROOT -- so an implicit cwd (cmd_resume's fallback, resume
    has no --project flag) silently defeated a caller's env-based store
    isolation. Real filesystem, real env var, no mocking of project_data_dir
    itself: the archive must be found under the ENV-designated store even
    though a different directory is passed as project_dir.
    """

    def test_archived_source_under_env_store_is_found_despite_different_project_dir(
        self,
    ):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = base / "env-designated-store"
            store.mkdir()
            cwd_project = base / "agent-desk"  # sibling of `store`, NOT inside it
            cwd_project.mkdir()

            # An archived transcript under the store's worktree layout --
            # exactly where all_worktree_archive_dirs/project_archive_dir look
            # once project_data_dir resolves the ENV-designated root.
            archive_dir = (
                store / ".synapt" / "recall" / "worktrees" / "main" / "transcripts"
            )
            archive_dir.mkdir(parents=True)
            source = archive_dir / "rollout-archived.jsonl"
            source.write_text("{}\n", encoding="utf-8")

            with (
                mock.patch.dict(
                    os.environ, {"SYNAPT_RECALL_ROOT": str(store)}, clear=False
                ),
                mock.patch(
                    "synapt.recall.codex.discover_codex_sessions", return_value=None
                ),
            ):
                os.environ.pop("GRIPSPACE_ROOT", None)
                got = cli._newest_source_file(cwd_project)

            # found under the ENV store, not silently dropped
            self.assertEqual(got, source.resolve())

    def test_without_the_env_var_the_same_layout_is_not_found(self):
        # Negative control: identical fixture, no SYNAPT_RECALL_ROOT set and
        # cwd_project itself owns no .synapt structure -- proves the positive
        # case above is actually locating the archive via the env var, not
        # via some other path that happens to reach it regardless.
        #
        # With no env var, project_data_dir(None) falls through to a
        # Path.cwd()-based walk-up -- so the process cwd must itself be
        # chdir'd into a tmp-owned, marker-free directory here, or this
        # negative control would walk up into whatever real gripspace the
        # test happened to run from (the store-isolation guard would then
        # legitimately refuse it as an escape, same as any other implicit
        # resolution reaching a real store).
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = base / "env-designated-store"
            store.mkdir()
            cwd_project = base / "agent-desk"
            cwd_project.mkdir()

            archive_dir = (
                store / ".synapt" / "recall" / "worktrees" / "main" / "transcripts"
            )
            archive_dir.mkdir(parents=True)
            (archive_dir / "rollout-archived.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )

            original_cwd = Path.cwd()
            os.chdir(cwd_project)
            try:
                with (
                    mock.patch.dict(os.environ, {}, clear=False),
                    mock.patch(
                        "synapt.recall.codex.discover_codex_sessions",
                        return_value=None,
                    ),
                ):
                    os.environ.pop("SYNAPT_RECALL_ROOT", None)
                    os.environ.pop("GRIPSPACE_ROOT", None)
                    got = cli._newest_source_file(cwd_project)
            finally:
                os.chdir(original_cwd)

            self.assertIsNone(got)  # store never named; nothing to find


class TestRefreshLabelRender(unittest.TestCase):
    def _view(self, refresh_label):
        v = ResumeView(
            session_id="0af31c22", selection_scope="store",
            turns=[ResumeTurn(chunk_id="c0", turn_index=0, timestamp="2026-09-01T00:00:00Z",
                              user_text="q", assistant_text="a", tools_used=[])],
            total_turns=1,
        )
        v.refresh_label = refresh_label
        return v

    def test_label_names_source_and_cursor(self):
        out = format_resume(self._view("source rollout-a.jsonl, index 2026-06-01 → 2026-09-01"))
        self.assertIn("REFRESHED before render — source rollout-a.jsonl", out)
        self.assertIn("2026-06-01 → 2026-09-01", out)

    def test_no_label_when_not_refreshed(self):
        out = format_resume(self._view(None))
        self.assertNotIn("REFRESHED before render", out)


if __name__ == "__main__":
    unittest.main()
