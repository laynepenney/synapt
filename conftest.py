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
        default=False,
        help=(
            "Also refuse implicit recall data paths (journal, index, archive, "
            "knowledge) that resolve outside a pytest-owned root. Off by "
            "default: the suite currently has real sites that resolve into the "
            "operator's live store, and arming this before they are isolated "
            "would turn a true finding into a red suite."
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

    _install_policy(config.getoption("--strict-recall-data-root"))


def _install_policy(strict_data_root: bool):
    """Arm the guards.

    The channel guard is always on — that is the contract this harness exists
    for. It refuses without redirecting, so it does not change where anything
    legitimately resolves.

    The data-root guard is opt-in *for now*, and the reason is worth stating
    plainly rather than hiding behind a default. Arming it fails 30 further
    tests across 10 files, and those failures are TRUE POSITIVES:
    ``tests/recall`` deliberately strips the root override so tests measure
    path inference, and that inference currently lands in a real checkout's
    ``.synapt/recall``. Isolating those sites is real work and a separate
    reviewable change. Shipping it armed would mean either a red suite or 30
    edits buried in a guard PR; shipping it absent would lose the mechanism. So
    it ships built, witnessed, and one flag away — and the count is reported
    rather than quietly carried.

    That count is 30 measured at base 1245f62, not the ~60 an earlier draft
    carried. The larger figure was taken while this file still installed a
    session-wide channel override, which was itself breaking 30 tests; half the
    "true positives" were the harness's own doing. A measurement inherits the
    configuration it was taken under, and a number quoted after the
    configuration changed is a stale claim wearing a precise costume.
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
