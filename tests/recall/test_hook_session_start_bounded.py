"""The session-start hook is bounded by construction.

Invariant under test: ``synapt recall hook session-start`` does O(1) work in
transcript and index size, prints inside a fixed byte budget, and reports
its own death.

Why (measured 2026-08-25, Ref #856, #119):

* The hook was killed at the 60s timeout and emitted NOTHING — a timed-out
  hook's output is discarded, not truncated, so the fresh session started
  with no continuity context and no signal that any was missing.
* Step 0 (archive + journal catch-up) ran inline. ``extract_session_id``
  only recognised ``progress``/``session_meta`` lines; modern transcripts
  carry ``sessionId`` on line 1 (``custom-title``). On 111 of 160 archived
  files it scanned the ENTIRE file (1.4 GB, 818 MB, ...) every session
  start, found nothing, and never journaled those sessions.
* The contradictions step opened the index through the WRITER path
  (schema DDL + migrations, 30s busy timeout) and blocked 13.9s behind a
  concurrent build; a read-only reader answers in 0.01s.
* When the hook did finish it emitted 227 KB. The harness previews ~2 KB
  and saves the rest to a file the model is never told to read.

Each failure here has a control that can go red: bounded scans have a
line at bound+1, the run log has a seeded unfinished record, the budget
has a journal that is 40x larger than the budget.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from synapt.recall import cli
from synapt.recall.journal import (
    JournalEntry,
    SESSION_ID_SCAN_MAX_LINES,
    _journal_path,
    append_entry,
    auto_extract_entry,
    extract_session_id,
)


# ---------------------------------------------------------------------------
# extract_session_id — modern format + bounded scan
# ---------------------------------------------------------------------------


def _write_lines(path: Path, lines: list[dict | str]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line if isinstance(line, str) else json.dumps(line))
            f.write("\n")
    return path


class TestExtractSessionId:
    def test_modern_first_line_custom_title(self, tmp_path):
        p = _write_lines(tmp_path / "t.jsonl", [
            {"type": "custom-title", "customTitle": "opus", "sessionId": "sid-modern-1"},
            {"type": "mode", "mode": "auto"},
        ])
        assert extract_session_id(p) == "sid-modern-1"

    def test_any_line_carrying_session_id(self, tmp_path):
        p = _write_lines(tmp_path / "t.jsonl", [
            {"type": "mode", "mode": "auto"},
            {"type": "file-history-snapshot", "snapshot": {}},
            {"type": "user", "sessionId": "sid-user-2", "message": {"role": "user", "content": "hi"}},
        ])
        assert extract_session_id(p) == "sid-user-2"

    def test_legacy_progress_and_session_meta_still_resolve(self, tmp_path):
        p1 = _write_lines(tmp_path / "legacy1.jsonl", [
            {"type": "summary", "summary": "x"},
            {"type": "progress", "sessionId": "sid-legacy-progress"},
        ])
        p2 = _write_lines(tmp_path / "legacy2.jsonl", [
            {"type": "session_meta", "payload": {"id": "sid-legacy-meta"}},
        ])
        assert extract_session_id(p1) == "sid-legacy-progress"
        assert extract_session_id(p2) == "sid-legacy-meta"

    def test_scan_is_bounded_by_lines(self, tmp_path):
        """A transcript with no recognisable id must NOT be read to the end.

        Control pair: the id at line ``bound`` resolves, at ``bound + 1`` it
        does not. Without the bound both would resolve and the pair could not
        discriminate.
        """
        bound = 50
        filler = [{"type": "attachment", "n": i} for i in range(bound - 1)]
        at_bound = _write_lines(tmp_path / "at.jsonl", filler + [{"type": "user", "sessionId": "sid-at-bound"}])
        past_bound = _write_lines(tmp_path / "past.jsonl", filler + [{"type": "attachment"}, {"type": "user", "sessionId": "sid-past"}])
        assert extract_session_id(at_bound, max_lines=bound) == "sid-at-bound"
        assert extract_session_id(past_bound, max_lines=bound) == ""

    def test_default_bound_is_finite(self, tmp_path):
        assert isinstance(SESSION_ID_SCAN_MAX_LINES, int) and SESSION_ID_SCAN_MAX_LINES > 0
        filler = [{"type": "attachment", "n": i} for i in range(SESSION_ID_SCAN_MAX_LINES)]
        p = _write_lines(tmp_path / "deep.jsonl", filler + [{"type": "user", "sessionId": "sid-too-deep"}])
        assert extract_session_id(p) == ""

    def test_scan_is_bounded_by_bytes(self, tmp_path):
        """Line bound alone is not enough: one line can be megabytes."""
        big = {"type": "file-history-snapshot", "blob": "x" * 200_000}
        p = _write_lines(tmp_path / "fat.jsonl", [big, big, big, {"type": "user", "sessionId": "sid-after-fat"}])
        assert extract_session_id(p, max_bytes=300_000) == ""
        assert extract_session_id(p, max_bytes=1_000_000) == "sid-after-fat"


class TestAutoExtractEntryModernFormat:
    def test_session_id_from_modern_line(self, tmp_path):
        p = _write_lines(tmp_path / "t.jsonl", [
            {"type": "custom-title", "customTitle": "opus", "sessionId": "sid-auto-1"},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "name": "Edit", "input": {"file_path": str(tmp_path / "a.py")}},
            ]}},
        ])
        entry = auto_extract_entry(transcript_path=str(p), cwd=str(tmp_path))
        assert entry.session_id == "sid-auto-1"


# ---------------------------------------------------------------------------
# hook session-start — no inline catch-up, one detached catchup, run log
# ---------------------------------------------------------------------------


def _iso_ago(*, seconds: float) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(timespec="seconds")


def _begin_finish_one(log_path: str) -> str:
    """Worker for the concurrent witness: one full hook-run record pair.
    Module-level so the spawn start method can import it."""
    from synapt.recall.session_start import HookRun
    run = HookRun("session-start", "startup", log_path=Path(log_path))
    run.begin()
    run.finish(output_bytes=1)
    return run.run_id


def _run_hook(monkeypatch, tmp_path, *, source="startup", context_lines=None, transcript_dirs=None,
              channel_unread=None, channel_read="", directives=""):
    """Drive cmd_hook('session-start') with a hook payload on stdin.

    Returns (stdout_text, popen_calls). Everything with a side effect outside
    the owned root is patched; the pieces the invariant is about are recorded.
    """
    payload = json.dumps({
        "session_id": "hook-test-session",
        "hook_event_name": "SessionStart",
        "source": source,
        "cwd": str(tmp_path),
    })
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    popen_calls: list[list[str]] = []

    class _FakePopen:
        def __init__(self, argv, **kwargs):
            popen_calls.append(list(argv))
            self.kwargs = kwargs

    out = io.StringIO()
    with patch.object(cli.subprocess, "Popen", _FakePopen), \
         patch.object(cli, "project_transcript_dirs", return_value=transcript_dirs if transcript_dirs is not None else [tmp_path]), \
         patch("synapt.recall.server._check_version_stale", return_value=None), \
         patch("synapt.recall.journal.compact_journal", return_value=0), \
         patch.object(cli, "_dev_loop_activation_prompt", return_value=None), \
         patch("synapt.recall.reminders._reminders_path", return_value=tmp_path / "reminders.json"), \
         patch("synapt.recall.channel.channel_join"), \
         patch("synapt.recall.channel.channel_unread", return_value=channel_unread or {}), \
         patch("synapt.recall.channel.channel_read", return_value=channel_read), \
         patch("synapt.recall.channel.check_directives", return_value=directives), \
         patch("synapt.recall.server.format_contradictions_for_session_start", return_value=""), \
         patch("synapt.recall.knowledge.read_nodes", return_value=[]), \
         patch("synapt.recall.journal._get_branch", return_value=None), \
         patch.object(sys, "stdout", out):
        if context_lines is not None:
            with patch.object(cli, "generate_startup_context", return_value=list(context_lines)):
                cli.cmd_hook(argparse.Namespace(event="session-start"))
        else:
            cli.cmd_hook(argparse.Namespace(event="session-start"))
    return out.getvalue(), popen_calls


class TestSessionStartContinuityPolicy:
    @pytest.mark.parametrize(
        ("mode", "source", "allowed"),
        [
            ("off", "startup", False),
            ("explicit", "startup", False),
            ("explicit", "resume", True),
            ("automatic", "startup", True),
            ("automatic", "resume", True),
            ("automatic", "fork", True),
            ("automatic", "clear", False),
            ("always", "clear", True),
            ("always", "compact", False),
        ],
    )
    def test_policy(self, mode, source, allowed):
        config = type("Config", (), {
            "get_session_start_continuity": lambda self: mode,
        })()
        with patch("synapt.recall.config.load_config", return_value=config):
            assert cli._session_start_continuity_allowed(source) is allowed

    def test_compact_hook_is_absolute_noop(
        self, owned_recall_root, monkeypatch, tmp_path, capsys,
    ):
        payload = json.dumps({"hook_event_name": "SessionStart", "source": "compact"})
        monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
        monkeypatch.chdir(tmp_path)
        with patch.object(cli, "_session_start_continuity_allowed") as policy, \
             patch.object(cli, "project_data_dir") as data_dir, \
             patch.object(cli, "_spawn_session_start_catchup") as catchup, \
             patch.object(cli, "generate_startup_context") as context:
            cli.cmd_hook(argparse.Namespace(event="session-start"))
        policy.assert_not_called()
        data_dir.assert_not_called()
        catchup.assert_not_called()
        context.assert_not_called()
        assert capsys.readouterr().out == ""

    def test_clear_keeps_ambient_startup_but_disables_continuity(
        self, owned_recall_root, monkeypatch, tmp_path,
    ):
        with patch.object(cli, "generate_startup_context", return_value=["ambient"]) as context:
            out, _ = _run_hook(monkeypatch, tmp_path, source="clear")

        context.assert_called_once_with(
            tmp_path.resolve(),
            include_continuity=False,
            current_session_id="hook-test-session",
        )
        assert "ambient" in out


class TestHookDoesNoUnboundedWorkInline:
    def test_catchup_is_not_called_inline(self, owned_recall_root, monkeypatch, tmp_path):
        with patch.object(cli, "_catchup_archive_and_journal") as catchup:
            _run_hook(monkeypatch, tmp_path, context_lines=["Next steps:\n  - x"])
        catchup.assert_not_called()

    def test_exactly_one_detached_catchup_is_spawned(self, owned_recall_root, monkeypatch, tmp_path):
        _, calls = _run_hook(monkeypatch, tmp_path, context_lines=["Next steps:\n  - x"])
        assert len(calls) == 1, calls
        argv = calls[0]
        assert argv[:3] == [sys.executable, "-m", "synapt.recall.cli"]
        assert argv[3] == "catchup"
        # The old shape spawned build AND enrich separately; both are now
        # sequenced inside catchup so they cannot fight for the lock.
        assert "build" not in argv and "enrich" not in argv

    def test_no_transcripts_no_spawn(self, owned_recall_root, monkeypatch, tmp_path):
        _, calls = _run_hook(monkeypatch, tmp_path, context_lines=["Next steps:\n  - x"], transcript_dirs=[])
        assert calls == []


class TestHookRunLog:
    def test_run_is_recorded_started_then_completed(self, owned_recall_root, monkeypatch, tmp_path):
        from synapt.recall.session_start import hook_run_log_path
        _run_hook(monkeypatch, tmp_path, context_lines=["Next steps:\n  - x"])
        log = hook_run_log_path()
        assert log.exists(), log
        records = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        assert [r["status"] for r in records] == ["started", "completed"]
        assert records[0]["run_id"] == records[1]["run_id"] and records[0]["run_id"]
        done = records[-1]
        assert done["event"] == "session-start"
        assert done["source"] == "startup"
        assert isinstance(done["phases"], dict) and done["phases"]
        assert done["total_s"] >= 0
        assert done["output_bytes"] > 0

    def test_unfinished_previous_run_is_reported_at_the_top(self, owned_recall_root, monkeypatch, tmp_path):
        from synapt.recall.session_start import hook_run_log_path
        log = hook_run_log_path()
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({
            "event": "session-start", "source": "startup", "status": "started",
            "started": "2026-08-25T04:35:51+00:00", "pid": 99999,
        }) + "\n")
        out, _ = _run_hook(monkeypatch, tmp_path, context_lines=["Next steps:\n  - x"])
        first_line = out.splitlines()[0]
        assert "did not finish" in first_line
        assert "2026-08-25T04:35:51" in out

    def test_completed_previous_run_is_not_a_warning(self, owned_recall_root, monkeypatch, tmp_path):
        """Control for the warning: a finished record must not trip it."""
        from synapt.recall.session_start import hook_run_log_path
        log = hook_run_log_path()
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            json.dumps({"event": "session-start", "source": "startup", "status": "started", "started": "x", "pid": 1}) + "\n"
            + json.dumps({"event": "session-start", "source": "startup", "status": "completed", "total_s": 1.5, "phases": {}, "output_bytes": 10}) + "\n"
        )
        out, _ = _run_hook(monkeypatch, tmp_path, context_lines=["Next steps:\n  - x"])
        assert "did not finish" not in out

    def test_concurrent_runs_all_survive(self, tmp_path):
        """Simultaneous session starts are the production case the detached
        catchup exists for; the health log must survive the same scenario.
        Atlas's r2 probe (2026-08-25): 16 concurrent begin/finish pairs kept
        2 of 32 records under the read-modify-replace form. This witness is
        red on that form and green only when every record survives with its
        own run identity.
        """
        from synapt.recall.session_start import HOOK_RUN_LOG_MAX_RECORDS
        log = tmp_path / "hook-runs.jsonl"
        n = 16
        assert 2 * n <= HOOK_RUN_LOG_MAX_RECORDS  # trimming must not be the explanation
        with concurrent.futures.ProcessPoolExecutor(max_workers=n) as pool:
            run_ids = list(pool.map(_begin_finish_one, [str(log)] * n))
        records = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        assert len(records) == 2 * n, f"kept {len(records)} of {2 * n}"
        assert len(set(run_ids)) == n
        started = {r["run_id"] for r in records if r["status"] == "started"}
        completed = {r["run_id"] for r in records if r["status"] == "completed"}
        assert started == completed == set(run_ids)

    def test_live_sibling_is_not_reported_as_unfinished(self, tmp_path):
        """A started record with no completed yet is a DEATH only if its
        process is gone. A sibling hook still running (this very process, here)
        must not trip the warning, or every concurrent start reports the other
        as dead."""
        from synapt.recall.session_start import HookRun
        log = tmp_path / "hook-runs.jsonl"
        log.write_text(json.dumps({
            "event": "session-start", "source": "startup", "status": "started",
            "started": _iso_ago(seconds=5), "pid": os.getpid(), "run_id": "sibling-alive",
        }) + "\n")
        assert HookRun("session-start", "startup", log_path=log).begin() is None

    def test_stale_started_is_dead_even_if_its_pid_is_alive(self, tmp_path):
        """Pids are reused. A started record older than STALE_RUN_SECONDS with
        no completion is reported even though (here) its pid is this very
        process. Control: the same record five seconds old is not."""
        from synapt.recall.session_start import STALE_RUN_SECONDS, HookRun
        log = tmp_path / "hook-runs.jsonl"
        log.write_text(json.dumps({
            "event": "session-start", "status": "started", "pid": os.getpid(), "run_id": "stale-run",
            "started": _iso_ago(seconds=STALE_RUN_SECONDS + 60),
        }) + "\n")
        warning = HookRun("session-start", "startup", log_path=log).begin()
        assert warning and "did not finish" in warning and "stale-run" in warning

    def test_pid_probe_never_calls_os_kill_on_win32(self, monkeypatch):
        """On Windows os.kill(pid, sig) is TerminateProcess for every sig but the
        two CTRL events — signal 0 included. The probe must make that call
        unrepresentable on win32 and keep the Unix idiom elsewhere."""
        from synapt.recall import session_start as ss

        def forbidden(*a, **k):
            raise AssertionError(f"os.kill{a} would TerminateProcess on win32")

        monkeypatch.setattr(ss.sys, "platform", "win32")
        with patch.object(os, "kill", forbidden), \
             patch.object(ss, "_pid_alive_win32", return_value=True) as win:
            assert ss._pid_alive(424242) is True
        win.assert_called_once_with(424242)

        # Control: on a POSIX platform the probe IS os.kill(pid, 0) and nothing else.
        monkeypatch.setattr(ss.sys, "platform", "darwin")
        calls: list[tuple] = []
        with patch.object(os, "kill", lambda *a: calls.append(a)), \
             patch.object(ss, "_pid_alive_win32", side_effect=AssertionError("win32 path on POSIX")):
            assert ss._pid_alive(424242) is True
        assert calls == [(424242, 0)]

    def test_win32_probe_is_query_only_and_closes_its_handle(self):
        """The Windows primitive: OpenProcess with QUERY_LIMITED_INFORMATION,
        GetExitCodeProcess, CloseHandle — never a terminate right. Faked
        kernel32, because this suite runs on POSIX too; the CI windows-latest
        leg exercises the real one through the live-sibling test."""
        import ctypes
        from ctypes import wintypes
        from types import SimpleNamespace
        from synapt.recall import session_start as ss

        # A HANDLE is PVOID. This one does not fit in 32 bits, so a call that
        # narrowed it to C int (ctypes' default) could not hand it back intact.
        WIDE_HANDLE = 0x1_0000_0007
        log: list[tuple] = []

        class FakeFn:
            """A foreign-function stand-in that records the prototypes the
            code declares on it. Without argtypes/restype the real ctypes
            call would narrow the handle; the fake cannot narrow anything,
            so the DECLARATIONS are what this witness checks."""
            def __init__(self, impl):
                self.impl = impl
                self.argtypes = None
                self.restype = None

            def __call__(self, *args):
                return self.impl(*[a.value if hasattr(a, "value") and not hasattr(a, "_obj") else a for a in args])

        def open_process(access, inherit, pid):
            log.append(("OpenProcess", access, inherit, pid))
            return WIDE_HANDLE if pid == 4242 else None

        def get_exit_code(handle, code_ref):
            log.append(("GetExitCodeProcess", handle))
            code_ref._obj.value = 259 if handle == WIDE_HANDLE else 1
            return 1

        def close_handle(handle):
            log.append(("CloseHandle", handle))
            return 1

        k32 = SimpleNamespace(
            OpenProcess=FakeFn(open_process),
            GetExitCodeProcess=FakeFn(get_exit_code),
            CloseHandle=FakeFn(close_handle),
        )
        with patch.object(ctypes, "windll", SimpleNamespace(kernel32=k32), create=True):
            assert ss._pid_alive_win32(4242) is True
            assert ss._pid_alive_win32(9999) is False  # OpenProcess returns no handle

        # Query-only access, and the wide handle reaches both consumers intact.
        assert log[0] == ("OpenProcess", 0x1000, 0, 4242)
        assert all(e[1] == 0x1000 for e in log if e[0] == "OpenProcess")
        assert ("GetExitCodeProcess", WIDE_HANDLE) in log
        assert ("CloseHandle", WIDE_HANDLE) in log

        # The native contract is DECLARED: HANDLE is pointer-sized in and out.
        assert k32.OpenProcess.restype is wintypes.HANDLE
        assert k32.OpenProcess.argtypes == [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        assert k32.GetExitCodeProcess.argtypes == [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        assert k32.GetExitCodeProcess.restype is wintypes.BOOL
        assert k32.CloseHandle.argtypes == [wintypes.HANDLE]
        assert k32.CloseHandle.restype is wintypes.BOOL
        assert ctypes.sizeof(wintypes.HANDLE) == ctypes.sizeof(ctypes.c_void_p)

    def test_completion_binds_to_its_own_run(self, tmp_path):
        """A completed record from a DIFFERENT run must not clear a dead run's
        started record. Pairing is by run_id, not by 'the last record'."""
        from synapt.recall.session_start import HookRun
        log = tmp_path / "hook-runs.jsonl"
        log.write_text(
            json.dumps({"event": "session-start", "status": "started", "started": "2026-08-25T04:35:51+00:00",
                        "pid": 99999, "run_id": "dead-run"}) + "\n"
            + json.dumps({"event": "session-start", "status": "started", "started": "2026-08-25T04:36:00+00:00",
                          "pid": 99998, "run_id": "other-run"}) + "\n"
            + json.dumps({"event": "session-start", "status": "completed", "pid": 99998, "run_id": "other-run",
                          "total_s": 1.0, "phases": {}, "output_bytes": 1}) + "\n"
        )
        warning = HookRun("session-start", "startup", log_path=log).begin()
        assert warning and "did not finish" in warning and "dead-run" in warning

    def test_log_is_bounded(self, owned_recall_root, monkeypatch, tmp_path):
        from synapt.recall.session_start import HOOK_RUN_LOG_MAX_RECORDS, hook_run_log_path
        log = hook_run_log_path()
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("".join(
            json.dumps({"event": "session-start", "status": "completed", "phases": {}, "total_s": 0, "output_bytes": 1}) + "\n"
            for _ in range(HOOK_RUN_LOG_MAX_RECORDS * 3)
        ))
        _run_hook(monkeypatch, tmp_path, context_lines=["Next steps:\n  - x"])
        n = len([l for l in log.read_text().splitlines() if l.strip()])
        assert n <= HOOK_RUN_LOG_MAX_RECORDS


# ---------------------------------------------------------------------------
# output budget — head first, per-source clipping, full text on disk
# ---------------------------------------------------------------------------


class TestWakeOutputBudget:
    def _seed_big_journal(self, tmp_path, n_items=400, item_len=200):
        # The wake reads the AMBIENT journal store (owned_recall_root sets it via
        # SYNAPT_RECALL_ROOT); seed the same store, not project=tmp_path, so the
        # wake sees what we wrote (dual-use wake fix).
        jf = _journal_path()
        jf.parent.mkdir(parents=True, exist_ok=True)
        entry = JournalEntry(
            timestamp="2026-08-22T03:40:00Z",
            session_id="big-session",
            focus="EOD close",
            done=[f"done item {i} " + "d" * item_len for i in range(n_items)],
            decisions=[f"decision {i} " + "c" * item_len for i in range(n_items)],
            next_steps=[f"NEXT-STEP-{i:04d} " + "n" * item_len for i in range(n_items)],
        )
        append_entry(entry, jf)
        return jf

    def test_stdout_is_within_budget_and_points_at_full_text(self, owned_recall_root, monkeypatch, tmp_path):
        from synapt.recall.session_start import WAKE_BUDGET_BYTES, wake_file_path
        self._seed_big_journal(tmp_path)
        out, _ = _run_hook(monkeypatch, tmp_path)
        full = wake_file_path(tmp_path)
        assert full.exists()
        full_text = full.read_text()
        # Everything is on disk...
        assert full_text.count("NEXT-STEP-") == 400
        # ...and stdout is bounded, says so, and says where the rest is.
        assert len(out.encode()) <= WAKE_BUDGET_BYTES
        assert "withheld" in out
        assert str(full) in out
        assert out.count("NEXT-STEP-") < 400

    def test_later_sources_survive_a_huge_journal(self, owned_recall_root, monkeypatch, tmp_path):
        """Per-source caps, not just a total cap. Without them the journal
        (300 KB here) consumes the whole budget and the channel state and
        directives — the parts that change every morning — never appear.
        Mutation witness: raising _CAP_JOURNAL_LATEST to 60 MB turns this red
        while the total-budget test above stays green."""
        self._seed_big_journal(tmp_path)
        out, _ = _run_hook(monkeypatch, tmp_path, channel_unread={"dev": 5},
                           channel_read="  2026-08-25 apollo: DEV-MESSAGE-ONE",
                           directives="DIRECTIVE-LINE-ONE\nDIRECTIVE-LINE-TWO")
        assert "#dev: 5 unread" in out
        assert "DEV-MESSAGE-ONE" in out
        assert "DIRECTIVE-LINE-ONE" in out
        assert "NEXT-STEP-0000" in out  # and the journal is still there, clipped
        assert "(truncated to budget)" not in out  # per-source caps did the work, not the guard

    def test_first_two_kb_carry_the_head_and_next_steps(self, owned_recall_root, monkeypatch, tmp_path):
        """The harness previews ~2 KB. That window must hold the head line and
        the start of the open threads, not a warning banner or a reminder hoard."""
        self._seed_big_journal(tmp_path)
        out, _ = _run_hook(monkeypatch, tmp_path)
        head = out.encode()[:2048].decode(errors="ignore")
        assert head.startswith("synapt wake"), head[:80]
        assert "source=startup" in head
        assert "NEXT-STEP-0000" in head

    def test_first_two_kb_report_journal_selection_coverage(
        self, owned_recall_root, monkeypatch, tmp_path,
    ):
        """The coverage report cannot sit behind the content whose omission it reports."""
        coverage = (
            'Journal read: {"shown":3,"withheld":9,'
            '"oldest_shown_at":"2026-08-03T12:00:00Z"}'
        )
        huge_journal = "Next steps:\n" + "\n".join(
            f"  - load-bearing-{i:04d} " + "x" * 180 for i in range(100)
        )
        out, _ = _run_hook(
            monkeypatch,
            tmp_path,
            context_lines=[huge_journal, coverage],
        )

        preview = out.encode()[:2048].decode(errors="ignore")
        assert coverage in preview
        assert "load-bearing-0000" in preview

    def test_source_is_carried_from_the_payload(self, owned_recall_root, monkeypatch, tmp_path):
        self._seed_big_journal(tmp_path)
        out, _ = _run_hook(monkeypatch, tmp_path, source="resume")
        assert "source=resume" in out.splitlines()[0]

    def test_small_context_is_not_clipped(self, owned_recall_root, monkeypatch, tmp_path):
        """Control: a context under budget passes through whole."""
        # The wake reads the AMBIENT journal store (owned_recall_root sets it via
        # SYNAPT_RECALL_ROOT); seed the same store, not project=tmp_path, so the
        # wake sees what we wrote (dual-use wake fix).
        jf = _journal_path()
        jf.parent.mkdir(parents=True, exist_ok=True)
        append_entry(JournalEntry(timestamp="2026-08-22T03:40:00Z", session_id="s", focus="f",
                                  next_steps=["one small step"]), jf)
        out, _ = _run_hook(monkeypatch, tmp_path)
        assert "one small step" in out
        assert '"shown":1' in out
        assert '"withheld":0' in out
        assert " B withheld" not in out
        assert out.rstrip().endswith("B, complete")

    def test_reminder_hoard_is_counted_not_dumped(self, owned_recall_root, monkeypatch, tmp_path):
        from synapt.recall.reminders import add_reminder
        rfile = tmp_path / "reminders.json"
        with patch("synapt.recall.reminders._reminders_path", return_value=rfile):
            for i in range(60):
                add_reminder(f"STICKY-REMINDER-{i:03d} " + "r" * 500, sticky=True)
        out, _ = _run_hook(monkeypatch, tmp_path)
        assert "60 reminders" in out.splitlines()[0]
        assert out.count("STICKY-REMINDER-") < 60


# ---------------------------------------------------------------------------
# catchup — the deferred half, sequenced under one lock
# ---------------------------------------------------------------------------


class TestCatchupCommand:
    def test_runs_archive_journal_compact_build_enrich_in_order(self, owned_recall_root, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        order: list[str] = []

        def rec(name):
            def _f(*a, **k):
                order.append(name)
                return 0
            return _f

        def fake_run(argv, **kwargs):
            order.append(" ".join(argv[3:]))
            return subprocess.CompletedProcess(argv, 0)

        with patch.object(cli, "project_transcript_dirs", return_value=[tmp_path / "a", tmp_path / "b"]), \
             patch.object(cli, "_catchup_archive_and_journal", rec("catchup")), \
             patch("synapt.recall.journal.compact_journal", rec("compact")), \
             patch.object(cli.subprocess, "run", fake_run):
            cli.cmd_catchup(argparse.Namespace())
        assert order == ["catchup", "catchup", "compact", "build --incremental", "enrich --max-entries 1"]

    def test_no_build_flag_skips_build_and_enrich(self, owned_recall_root, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        order: list[str] = []
        with patch.object(cli, "project_transcript_dirs", return_value=[tmp_path]), \
             patch.object(cli, "_catchup_archive_and_journal", lambda *a, **k: order.append("catchup")), \
             patch("synapt.recall.journal.compact_journal", lambda: order.append("compact")), \
             patch.object(cli.subprocess, "run", lambda argv, **k: order.append(" ".join(argv[3:]))):
            cli.cmd_catchup(argparse.Namespace(no_build=True))
        assert order == ["catchup", "compact"]

    def test_second_concurrent_catchup_yields(self, owned_recall_root, monkeypatch, tmp_path, capsys):
        """Two catchups at once would double-journal and fight for the build
        lock. The second one must step aside, and SAY so on stderr."""
        monkeypatch.chdir(tmp_path)
        with patch.object(cli, "_acquire_build_lock", return_value=None), \
             patch.object(cli, "_catchup_archive_and_journal") as catchup:
            cli.cmd_catchup(argparse.Namespace())
        catchup.assert_not_called()
        assert "already running" in capsys.readouterr().err

    def test_catchup_is_registered(self):
        parser = cli.make_parser()
        ns = parser.parse_args(["catchup", "--no-build"])
        assert ns.command == "catchup" and ns.no_build is True


# ---------------------------------------------------------------------------
# read-only index access from the hook path
# ---------------------------------------------------------------------------


class TestReadOnlyIndexAccess:
    def test_open_readonly_skips_schema_and_refuses_writes(self, tmp_path):
        import sqlite3
        from synapt.recall.storage import RecallDB
        path = tmp_path / "recall.db"
        RecallDB(path).close()  # create with the real schema once
        with patch.object(RecallDB, "_ensure_schema", side_effect=AssertionError("DDL ran on the read path")):
            db = RecallDB.open_readonly(path)
        try:
            assert db.pending_contradiction_count() == 0
            with pytest.raises(sqlite3.OperationalError):
                db._conn.execute("INSERT INTO pending_contradictions (old_node_id, new_content, category, reason, source_sessions, status, detected_at) VALUES ('a','b','c','d','[]','pending','now')")
        finally:
            db.close()

    def test_open_readonly_on_missing_db_raises_not_creates(self, tmp_path):
        from synapt.recall.storage import RecallDB
        missing = tmp_path / "nope" / "recall.db"
        with pytest.raises(Exception):
            RecallDB.open_readonly(missing)
        assert not missing.exists()
        assert not missing.parent.exists()

    def test_contradictions_for_session_start_use_readonly(self, tmp_path):
        from synapt.recall.storage import RecallDB
        from synapt.recall import server
        path = tmp_path / "index" / "recall.db"
        path.parent.mkdir(parents=True)
        db = RecallDB(path)
        db._conn.execute(
            "INSERT INTO pending_contradictions (old_node_id, new_content, category, reason, source_sessions, status, detected_at) "
            "VALUES ('n1', 'CONTRA-READONLY-PROBE', 'c', 'r', '[]', 'pending', '2026-08-25')")
        db._conn.commit()
        db.close()
        opened: list[Path] = []
        real = RecallDB.open_readonly

        def spy(p, *a, **k):
            opened.append(Path(p))
            return real(p, *a, **k)

        # The server swallows exceptions and returns "", so an EMPTY result is
        # not evidence the read-only path worked: the seeded row must come
        # back. Mutation witness: open_readonly delegating to __init__ raises
        # here, the server returns "", and this goes red.
        with patch("synapt.recall.server.project_index_dir", return_value=path.parent), \
             patch.object(RecallDB, "open_readonly", staticmethod(spy)), \
             patch.object(RecallDB, "__init__", side_effect=AssertionError("writer path used on the hook read")):
            text = server.format_contradictions_for_session_start()
        assert "CONTRA-READONLY-PROBE" in text
        assert "Pending contradictions (1)" in text
        assert opened == [path]

    def test_contradictions_output_is_capped_with_count(self, tmp_path):
        from synapt.recall.storage import RecallDB
        from synapt.recall import server
        path = tmp_path / "index" / "recall.db"
        path.parent.mkdir(parents=True)
        db = RecallDB(path)
        for i in range(12):
            db._conn.execute(
                "INSERT INTO pending_contradictions (old_node_id, new_content, category, reason, source_sessions, status, detected_at) "
                "VALUES (?, ?, 'c', 'r', '[]', 'pending', ?)", (f"node-{i}", f"CONTRA-{i:02d}", f"2026-08-{i+1:02d}"))
        db._conn.commit()
        db.close()
        with patch("synapt.recall.server.project_index_dir", return_value=path.parent):
            text = server.format_contradictions_for_session_start(limit=5)
        assert "Pending contradictions (12" in text
        assert text.count("CONTRA-") == 5


# ---------------------------------------------------------------------------
# build lock — say who holds it
# ---------------------------------------------------------------------------


class TestBuildLockHolder:
    def test_acquire_stamps_holder_and_busy_message_names_it(self, tmp_path):
        import errno
        data_dir = tmp_path / "data"
        fd = cli._acquire_build_lock(data_dir, timeout=0)
        assert fd is not None
        try:
            stamp = (data_dir / "build.lock").read_text()
            assert str(os.getpid()) in stamp
            # Simulate contention from another process: the lock call raises
            # EAGAIN, and the message must name the holder from the stamp.
            with patch("synapt.recall._filelock.lock_exclusive_nb", side_effect=OSError(errno.EAGAIN, "busy")):
                assert cli._acquire_build_lock(data_dir, timeout=0) is None
            msg = cli._build_lock_busy_message(data_dir)
            assert f"pid {os.getpid()}" in msg
        finally:
            cli._release_build_lock(fd)

    def test_lock_name_is_parameterised(self, tmp_path):
        data_dir = tmp_path / "data"
        fd = cli._acquire_build_lock(data_dir, timeout=0, name="catchup.lock")
        assert fd is not None
        try:
            assert (data_dir / "catchup.lock").exists()
            assert not (data_dir / "build.lock").exists()
        finally:
            cli._release_build_lock(fd)

    def test_a_real_second_holder_is_excluded_then_reacquirable(self, tmp_path):
        """Single-flight holds against a REAL second acquire, no patching.

        This is the contention half of the lock's contract, exercised through
        the real lock rather than a mocked ``lock_exclusive_nb``: while one
        holder has it, a second non-blocking acquire returns None; after
        release it is acquirable again. On POSIX ``flock`` excludes across two
        open descriptions of the same file even in one process; on Windows the
        same exclusion must hold through ``msvcrt.locking``. It is therefore the
        cross-platform witness that the Windows sentinel-byte lock (which moved
        off byte 0 so the stamp stays readable) still SERIALIZES rather than
        merely relocating the lock somewhere uncontended — the guarantee a
        readability fix must not quietly break.
        """
        data_dir = tmp_path / "data"
        first = cli._acquire_build_lock(data_dir, timeout=0)
        assert first is not None
        try:
            # The stamp remains readable WHILE the lock is held (the exact case
            # that failed on Windows: msvcrt locks are mandatory, so locking the
            # stamp bytes made this read raise PermissionError).
            assert str(os.getpid()) in (data_dir / "build.lock").read_text()
            second = cli._acquire_build_lock(data_dir, timeout=0)
            assert second is None, "a second holder must be excluded while the lock is held"
        finally:
            cli._release_build_lock(first)
        reacquired = cli._acquire_build_lock(data_dir, timeout=0)
        assert reacquired is not None, "the lock must be acquirable again after release"
        cli._release_build_lock(reacquired)
