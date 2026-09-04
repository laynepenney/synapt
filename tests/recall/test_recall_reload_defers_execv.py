"""recall_reload must not call os.execv() before returning its own response.

Calling os.execv() inline, before the function returns, preempts
the JSON-RPC response for the reload call itself — a live MCP client waiting
on that call's reply gets nothing and times out, then the fresh (exec'd)
process rejects the client's next call because its MCP session was never
initialized by that connection. Measured with a minimal stdio client against
the real server subprocess: the reload call timed out, and the next call on
the same connection raised "Invalid request parameters" (received before
initialization was complete).

recall_reload must return its response BEFORE the process is replaced, so
os.execv() is deferred a short moment via a background timer.

CAUTION for anyone extending these tests: the deferred timer must never be
left pending when a test's ``patch("os.execv", ...)`` context exits. A stray
timer that fires after the patch is gone hits the REAL os.execv() and
replaces the pytest process itself, not a fixture -- caught in review
(Stromus, R2 on the original version of this file): test 1 originally
exited its patch block immediately after the ordering assertion, so its
timer fired ~0.2s later against the unpatched, real os.execv, re-executing
pytest with the same argv. Measured effects of that version: the test alone
looped forever (each re-exec re-ran the test, which scheduled a new stray
timer); the two-test file together happened to pass only because test 2's
own patch caught test 1's leftover timer by pure timing luck; a third file
appended after this one got no pytest summary at all, because by then
nothing was patched when the timer fired. Every test below stays inside its
patch context until its timer has already fired, and the regression witness
runs the whole module in a real subprocess so a stray timer can never be
masked by a neighbor's patch.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

from synapt.recall.server import recall_reload


def test_recall_reload_returns_before_execv_is_invoked():
    execv_called = threading.Event()

    def fake_execv(*_args, **_kwargs):
        execv_called.set()

    with patch("os.execv", side_effect=fake_execv):
        result = recall_reload()

        assert isinstance(result, str) and result, (
            "recall_reload must return a non-empty response synchronously"
        )
        assert not execv_called.is_set(), (
            "os.execv must not have run yet at the moment recall_reload() "
            "returns -- calling it inline before return is exactly the "
            "defect: the reload call's own response never reaches "
            "the client because the process is replaced before it can be sent"
        )
        # Stay inside the patch context until the deferred timer has fired.
        # Leaving the block here (as the original version of this test did)
        # lets the timer fire later against the REAL os.execv, which
        # replaces the process running this test suite.
        assert execv_called.wait(timeout=2.0), (
            "the deferred os.execv() never fired within 2s of recall_reload() "
            "returning -- the reload must still happen, just not inline"
        )


def test_recall_reload_still_execs_shortly_after_returning():
    execv_called = threading.Event()

    def fake_execv(*_args, **_kwargs):
        execv_called.set()

    with patch("os.execv", side_effect=fake_execv):
        recall_reload()
        # The deferred execv must still actually fire -- this isn't a no-op,
        # only a delay long enough for the response to flush first. Waiting
        # here, still inside the patch context, is what keeps this test's
        # own timer from ever reaching the real os.execv.
        assert execv_called.wait(timeout=2.0), (
            "the deferred os.execv() never fired within 2s of recall_reload() "
            "returning -- the reload must still happen, just not inline"
        )


def test_zzz_padding_keeps_the_process_alive_past_the_timer_window():
    """Padding fixture for the subprocess witnesses below.

    A trivial single-test pytest run finishes and the interpreter exits in
    well under recall_reload's ~0.2s deferred-execv window -- even a
    non-daemon pending timer thread can lose that race if nothing keeps the
    process alive a little longer, which would silently hide a stray timer
    rather than exercise it. This test does nothing but sleep past that
    window, run alongside the target test in the SAME subprocess/session, so
    a stray timer left pending by a buggy test has time to actually fire
    while something still patches os.execv (a passing run) or does not (the
    defect this file guards against).
    """
    time.sleep(0.3)


def _run_reload_test_alone(test_name: str) -> subprocess.CompletedProcess:
    """Run exactly one of the two tests above, isolated, in a fresh
    subprocess -- no neighboring test's own patch context is present to
    accidentally mask a stray timer -- padded so the process outlives the
    deferred-execv window regardless of how fast the target test itself is."""
    return subprocess.run(
        [
            sys.executable, "-m", "pytest",
            f"{Path(__file__)}::{test_name}",
            f"{Path(__file__)}::test_zzz_padding_keeps_the_process_alive_past_the_timer_window",
            "-q", "--no-header",
        ],
        capture_output=True, text=True, timeout=8,
    )


def _assert_one_clean_summary(result: subprocess.CompletedProcess, test_name: str) -> None:
    summary_lines = [
        line for line in result.stdout.splitlines()
        if " passed" in line or " failed" in line or " error" in line
    ]
    assert len(summary_lines) == 1, (
        f"{test_name}: expected exactly one pytest summary line from the "
        f"isolated subprocess; got {summary_lines!r} in stdout="
        f"{result.stdout!r} -- more than one summary (each further one is "
        f"a re-exec re-running the suite) means a stray unpatched "
        f"os.execv() fired after this test's own patch context closed"
    )
    assert "2 passed" in summary_lines[0], summary_lines[0]


def test_reload_test_1_runs_cleanly_alone_in_a_real_subprocess():
    """Regression witness for the self-re-exec class of bug, test 1.

    An in-process test's own patch context can accidentally mask a
    NEIGHBORING test's stray timer by pure timing luck -- measured: that is
    exactly what made the original two-test file pass together while test 1
    alone looped forever (each re-exec re-ran it, scheduling a new stray
    timer, producing dozens of "1 passed" summaries before an outer bound
    cut it off). Running test 1 by itself, padded past the timer window,
    with no test 2 present to catch its timer by accident, is what actually
    discriminates the fix from the original bug -- running both tests
    together (an earlier version of this witness) does not, because it
    reproduces the same lucky-catch shape as the original broken file, and
    running test 1 alone with no padding does not either, because the
    process can exit before the stray timer's window even elapses.
    """
    result = _run_reload_test_alone("test_recall_reload_returns_before_execv_is_invoked")
    _assert_one_clean_summary(result, "test 1")


def test_reload_test_2_runs_cleanly_alone_in_a_real_subprocess():
    """Same witness, test 2: it was already safe on its own (its wait
    happens inside its own patch context), so this is a control -- it
    should pass whether or not test 1's structure is correct."""
    result = _run_reload_test_alone("test_recall_reload_still_execs_shortly_after_returning")
    _assert_one_clean_summary(result, "test 2")
