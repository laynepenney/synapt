"""Session-wide on-disk isolation for the recall test suite.

Ref #955 — closes at promotion.

Channel tests that exercise the post path without isolating resolution fall
through to tier-2 resolution and write their fixture messages into the real
home-level store. The environment override to prevent that already existed;
what was missing is that *nothing failed* when a test reached the real store,
so the contamination was silent and cumulative.

This file installs the policy in two layers:

* **Layer 1, the import window only** — ``pytest_configure`` runs before any
  test module is imported, so module-level code and collection-time helpers
  need somewhere safe to resolve. An autouse fixture then hands ordinary
  resolution back to each test.
* **Layer 2, at the seams** — refusal. The resolver and every write surface
  consult the policy before creating or opening anything.

Layer 1 is deliberately narrow, and that narrowness was learned rather than
designed. Leaving a session-wide ``SYNAPT_SHARED_CHANNELS_DIR`` in place for
the tests themselves broke thirty of them: the override is tier 1 and outranks
the tier-3 local resolution some tests deliberately exercise, and one shared
directory turned independent tests into shared-state ones. An environment
default is a *semantic* change to path resolution, not a neutral safety net.

So the guarantee rests on Layer 2, which refuses without redirecting. A harness
that changes what the suite measures has stopped being a harness.

This lives at the repository root rather than under ``tests/`` on purpose: a
root ``conftest.py`` is always an *initial* conftest, so its ``pytest_configure``
is guaranteed to run before collection begins regardless of which path
arguments pytest was invoked with. A conftest inside ``tests/`` is loaded
during collection, which is the ordering the guard exists to get ahead of.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "tests"))

from recall_store_isolation import (  # noqa: E402
    RecallStoreIsolationError,
    StoreIsolationPolicy,
    contained_by,
    protected_channel_root,
)

POLICY = StoreIsolationPolicy()


# ---------------------------------------------------------------------------
# options and markers
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--allow-live-channel-store-tests",
        action="store_true",
        default=False,
        help=(
            "Authorize tests marked live_channel_store to touch the real "
            "home-level channel store. Integration use only; never in the "
            "ordinary CI command."
        ),
    )
    parser.addoption(
        "--strict-recall-data-root",
        action="store_true",
        default=True,
        help=(
            "Accepted and now the default; kept so existing invocations do not "
            "break. Implicit recall data paths (journal, index, archive, "
            "knowledge) that resolve outside a pytest-owned root are refused."
        ),
    )
    parser.addoption(
        "--no-strict-recall-data-root",
        action="store_false",
        dest="strict_recall_data_root",
        help=(
            "Disable the data-root guard for a debugging run. Not for CI: with "
            "it off, a test that resolves into a real checkout passes silently, "
            "which is the condition this guard exists to end. If both this and "
            "--strict-recall-data-root are given, the LAST one wins."
        ),
    )


# ---------------------------------------------------------------------------
# Layer 1 — safe defaults, installed before collection
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_channel_store: test deliberately touches the real channel store; "
        "requires --allow-live-channel-store-tests",
    )

    # Under the system temp directory, not the repository. A session root
    # inside the checkout would leave an untracked directory after every run,
    # and "no dirty workspaces" exists so that dirty state stays meaningful —
    # a harness that manufactures noise every run trains people to ignore it.
    session_root = Path(tempfile.mkdtemp(prefix="recall-store-isolation-"))
    channel_root = session_root / "channels"
    data_root = session_root / "data"
    channel_root.mkdir(parents=True, exist_ok=True)
    # SYNAPT_RECALL_ROOT refuses a root that does not exist, so create it
    # before pointing at it rather than letting the first reader discover a
    # mistyped path as an empty history.
    data_root.mkdir(parents=True, exist_ok=True)
    POLICY.session_root = session_root

    POLICY.session_channel_root = channel_root
    POLICY.session_data_root = data_root
    POLICY.allow_live = config.getoption("--allow-live-channel-store-tests")

    # Only supply a default; an invocation that already chose a test root keeps it.
    os.environ.setdefault("SYNAPT_SHARED_CHANNELS_DIR", str(channel_root))
    os.environ.setdefault("SYNAPT_RECALL_ROOT", str(data_root))
    # The companion override matters here: redirecting only the root would file
    # this run's per-worktree data under the cwd basename inside the shared
    # store, which is another workspace's namespace rather than our own.
    os.environ.setdefault("SYNAPT_RECALL_WORKTREE", "pytest-isolated")

    _install_policy(config.getoption("strict_recall_data_root"))


def _install_policy(strict_data_root: bool):
    """Arm the guards.

    The channel guard is always on — that is the contract this harness exists
    for. It refuses without redirecting, so it does not change where anything
    legitimately resolves.

    The data-root guard is now ALSO on by default. It shipped opt-in because
    arming it failed 30 tests across 10 files — true positives, since
    ``tests/recall`` deliberately strips the root override so tests measure
    path inference, and that inference resolved into a real checkout. Those 30
    were isolated and the flag was flipped.

    ONE WARNING FOR WHOEVER TOUCHES THIS NEXT, because the obvious evidence is
    the wrong evidence. Before the burn-down, running with the flag differed
    from running without it, and THAT DIFFERENCE was the proof the flag was
    wired. Now the two agree — which is the success criterion and, identically,
    the signature of a guard that has stopped working. Modal agreement can no
    longer distinguish them, so it must never be cited as evidence this guard
    functions. The evidence is the direct witnesses in
    ``tests/recall/test_store_isolation_guard.py``, which exercise the policy
    itself rather than observing the flag.

    ``--no-strict-recall-data-root`` exists for debugging and is not for CI.
    """
    from synapt.recall import channel as channel_mod
    from synapt.recall import core as core_mod

    channel_mod.set_store_path_policy(POLICY.check_channel_path)
    if strict_data_root:
        core_mod.set_data_root_policy(POLICY.check_data_root)


def pytest_unconfigure(config):
    from synapt.recall import channel as channel_mod
    from synapt.recall import core as core_mod

    channel_mod.set_store_path_policy(None)
    core_mod.set_data_root_policy(None)

    # Removing the session root is safe only because the guard already proved
    # it is outside every protected root — re-checked here rather than assumed,
    # since this is the one place the harness deletes anything.
    root = POLICY.session_root
    if root is None:
        return
    if any(contained_by(root, protected) for protected in POLICY.roots()):
        return
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# current-item tracking
# ---------------------------------------------------------------------------
# The refusal names the test that caused it, and the live-store opt-in needs the
# item's markers. Plain phase hooks are used rather than a wrapper so this stays
# correct across pytest's hookwrapper API changes.

def pytest_runtest_setup(item):
    POLICY.current_item = item
    _rearm_if_disarmed(item.config)


def _rearm_if_disarmed(config):
    """Re-install the policy before each test if anything cleared it.

    The guard lives in a module global, so ANY mechanism that replaces or
    resets that module silently disarms it — and a disarmed guard is invisible:
    the suite stays green while the protection it advertises is gone. Two such
    mechanisms are known. A nested in-process pytest run whose unconfigure
    cleared the global (fixed at its source with save-and-restore), and
    ``importlib.reload``, which rebinds the module dict wholesale and cannot be
    fixed at the source at all.

    Enumerating those two and patching each is the weaker move; the next
    mechanism arrives unannounced with the same silent signature. Re-arming per
    test closes over the whole class instead — whatever cleared it, the next
    test starts armed. Cost is one identity check against None.

    SCOPE, stated rather than left to inference: this closes CROSS-test
    leakage, not intra-test. A disarm that happens *during* a test leaves the
    guard off for the remainder of THAT test; the next one starts armed. So the
    blast radius drops from "every test that follows, indefinitely" to "the
    rest of this one," which is the right trade at this cost — but it is a
    reduction, not an elimination. Saying "the class is closed" without this
    sentence would invite exactly the inference this whole harness exists to
    prevent: reading a partial guarantee as a total one.
    """
    from synapt.recall import channel as channel_mod
    from synapt.recall import core as core_mod

    if channel_mod._store_path_policy is None:
        channel_mod.set_store_path_policy(POLICY.check_channel_path)
    if config.getoption("--strict-recall-data-root") and core_mod._data_root_policy is None:
        core_mod.set_data_root_policy(POLICY.check_data_root)


def pytest_runtest_call(item):
    POLICY.current_item = item


def pytest_runtest_teardown(item, nextitem):
    POLICY.current_item = item


def pytest_runtest_logfinish(nodeid, location):
    POLICY.current_item = None


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _natural_resolution_per_test(monkeypatch):
    """Hand each test back its ordinary resolution, and let the guard do the work.

    The session defaults set in ``pytest_configure`` cover *import and
    collection* time, when no fixture has run yet. They are deliberately not
    left in place for the tests themselves, for two reasons found empirically
    rather than reasoned in advance:

    * ``SYNAPT_SHARED_CHANNELS_DIR`` is tier 1 and outranks everything. Tests
      that build their own project directory and expect tier-3 local
      resolution (``TestMessageFormatValidation``) then write somewhere else
      and cannot find their own file.
    * A single session-wide root also converts independent tests into
      shared-state tests — ``test_join_spam`` asserted one join event and saw
      three, then five, then six as earlier tests piled into the same JSONL.

    Both are the same mistake in different clothes: an environment default is a
    *semantic* change to path resolution, not a neutral safety net. So Layer 1
    narrows to the window it is actually needed for, and the guarantee moves
    entirely to Layer 2 — which refuses the real store without altering where
    anything legitimately resolves. A harness that changes what the suite
    measures has stopped being a harness.
    """
    monkeypatch.delenv("SYNAPT_SHARED_CHANNELS_DIR", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)


@pytest.fixture(autouse=True)
def _unwind_extra_protected_roots():
    """Decoy roots never outlive the test that registered them."""
    yield
    POLICY.clear_extra_roots()


@pytest.fixture(name="protected_channel_root")
def _protected_channel_root() -> Path:
    return protected_channel_root()


@pytest.fixture
def register_protected_root():
    """Add a decoy protected root for the duration of one test.

    Adds only. The real home store is permanent in the policy and cannot be
    removed by any fixture, so registering a decoy buys a witness without
    buying a bypass.
    """
    def _register(root: Path) -> Path:
        POLICY.register_extra_root(Path(root))
        return Path(root)

    return _register


@pytest.fixture
def rearm_guard(request):
    """The per-test re-arm routine, for the witness that a reload cannot disarm it.

    Exposed as a fixture rather than imported: ``import conftest`` resolves to
    the nearest conftest on sys.path, which is not this one.
    """
    def _rearm():
        _rearm_if_disarmed(request.config)

    return _rearm


@pytest.fixture
def pytester_precedence():
    """Measure flag precedence in a SEPARATE PROCESS, both orderings.

    Deliberately a subprocess and not ``pytester``'s in-process runner. The
    hazard that rules out nested in-process runs elsewhere in this file is
    shared module globals — a nested run can disarm the outer guard, and a
    green result would then be indistinguishable from the hazard firing. A
    subprocess has no globals in common, so it is the one shape that can answer
    an argument-parsing question without risking the thing being asked about.

    Returns (armed_when_strict_last, armed_when_no_strict_last).
    """
    import subprocess

    root = Path(__file__).parent
    probe = "tests/recall/test_store_isolation_guard.py::test_the_data_root_guard_is_armed_by_default"

    def _run(*flags):
        out = subprocess.run(
            [sys.executable, "-m", "pytest", probe, *flags, "-q", "--tb=no",
             "-p", "no:cacheprovider"],
            cwd=root, capture_output=True, text=True, timeout=300,
        ).stdout
        # The witness PASSES when armed and SKIPS when deliberately disarmed,
        # so the outcome word is the measurement.
        return "passed" in out and "skipped" not in out

    return (
        _run("--no-strict-recall-data-root", "--strict-recall-data-root"),
        _run("--strict-recall-data-root", "--no-strict-recall-data-root"),
    )


@pytest.fixture
def install_policy():
    """The real arming routine, for the witness that the escape hatch is wired."""
    return _install_policy


@pytest.fixture
def isolation_policy():
    """The live policy object, for witnesses that inspect the harness itself."""
    return POLICY


@pytest.fixture
def store_isolation_error():
    return RecallStoreIsolationError


@pytest.fixture
def strict_data_root():
    """Arm the data-root guard for one test.

    The witnesses for the data-root half must exercise the policy whether or
    not the suite was invoked with ``--strict-recall-data-root``, otherwise the
    mechanism ships with tests that silently no-op in the default CI command —
    which is the "check that cannot fail" shape this whole harness is against.
    """
    from synapt.recall import core as core_mod

    previous = core_mod.set_data_root_policy(POLICY.check_data_root)
    try:
        yield POLICY
    finally:
        core_mod.set_data_root_policy(previous)


@pytest.fixture
def rederive_protected_root():
    """Re-derive the protected root *now*, under whatever the test has patched.

    Returned as a callable rather than a value so the derivation runs after the
    test's monkeypatching, which is the only way to witness that the boundary
    does not move when HOME and ``Path.home()`` do.
    """
    return protected_channel_root


pytest_plugins = ["pytester"]


_NESTED_CONFTEST = '''
import sys
from pathlib import Path

import pytest

sys.path.insert(0, {tests_dir!r})
from recall_store_isolation import StoreIsolationPolicy

POLICY = StoreIsolationPolicy()
DECOY = Path(__file__).parent / "decoy-home" / ".synapt" / "channels"


def pytest_addoption(parser):
    parser.addoption(
        "--allow-live-channel-store-tests", action="store_true", default=False
    )


_PREVIOUS = []


def pytest_configure(config):
    config.addinivalue_line("markers", "live_channel_store: touches the live store")
    POLICY.register_extra_root(DECOY)
    POLICY.allow_live = config.getoption("--allow-live-channel-store-tests")
    from synapt.recall import channel as channel_mod
    # Save and RESTORE rather than clearing: pytester runs in-process, so this
    # nested run shares module globals with the parent session. Setting the
    # policy to None on unconfigure would disarm the outer guard for every test
    # that follows, and nothing would report it.
    _PREVIOUS.append(channel_mod.set_store_path_policy(POLICY.check_channel_path))


def pytest_unconfigure(config):
    from synapt.recall import channel as channel_mod
    channel_mod.set_store_path_policy(_PREVIOUS.pop() if _PREVIOUS else None)


def pytest_runtest_setup(item):
    POLICY.current_item = item


def pytest_runtest_call(item):
    POLICY.current_item = item


@pytest.fixture
def decoy_root():
    return DECOY
'''

_NESTED_TEST = '''
from synapt.recall import channel as channel_mod

{decorator}
def test_writes_to_the_live_store(decoy_root):
    msg = channel_mod.ChannelMessage(
        timestamp="2026-08-07T00:00:00Z",
        channel="dev",
        type="message",
        body="deliberate live write",
        from_agent="s_optin",
    )
    channel_mod._append_message(msg, channels_dir=decoy_root)
    assert (decoy_root / "dev.jsonl").exists()
'''


def _build_nested(pytester, decorator: str):
    """Scaffold a nested pytest run that installs the *real* policy.

    The mark and the option are pytest-level facts, so the only faithful way to
    witness their interaction is to run pytest. Importing the shared policy
    module rather than restating the rule keeps this a test of the guard rather
    than a test of a second implementation of the guard.
    """
    tests_dir = str(Path(__file__).parent / "tests")
    pytester.makeconftest(_NESTED_CONFTEST.format(tests_dir=tests_dir))
    pytester.makepyfile(_NESTED_TEST.format(decorator=decorator))
    return pytester


@pytest.fixture
def pytester_isolated(pytester):
    """A nested run whose single test carries the live_channel_store mark."""
    return _build_nested(pytester, "@__import__('pytest').mark.live_channel_store")


@pytest.fixture
def pytester_isolated_unmarked(pytester):
    """A nested run whose single test carries no mark."""
    return _build_nested(pytester, "")


@pytest.fixture
def isolated_channels(tmp_path, monkeypatch) -> Path:
    """A per-test channel root, stronger than the session default."""
    channels = tmp_path / "channels"
    channels.mkdir()
    monkeypatch.setenv("SYNAPT_SHARED_CHANNELS_DIR", str(channels))
    return channels
