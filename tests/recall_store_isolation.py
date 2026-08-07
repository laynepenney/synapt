"""Test-only store-isolation policy for the recall suite.

Ref #955 — closes at promotion.

This module holds the policy itself so that both the root ``conftest.py`` and
any nested pytest run (the mark/option witnesses use ``pytester``) can install
the *same* object rather than a lookalike. A witness that exercises a
reimplementation of the guard proves the reimplementation.

Nothing here is imported by production code. The production side owns only a
pair of no-op seams that call whatever policy is installed; with no policy
installed they do nothing at all.
"""

from __future__ import annotations

import os
import pwd
from pathlib import Path


class RecallStoreIsolationError(AssertionError):
    """Raised when a test would resolve a write target into a protected store.

    Deliberately an ``AssertionError`` subclass: this is a test-suite contract
    violation, not a runtime error in the code under test, and it must fail the
    test rather than be swallowed by an ``except Exception`` in a fixture.
    """


def account_home() -> Path:
    """The home directory of the operating-system account.

    Read from the passwd database rather than ``Path.home()`` or ``$HOME``
    precisely because a fixture can move both of those. The protected boundary
    must not be derivable from the value a test is currently asking the code to
    use, or every guarantee here becomes circular.
    """
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


def protected_channel_root() -> Path:
    """The real home-level channel store — the thing being protected."""
    return account_home() / ".synapt" / "channels"


def _normalise(path: Path) -> Path:
    """Resolve symlinks and ``..`` without requiring the path to exist.

    Containment is a path property, not a string property. A candidate that
    merely *looks* outside the store while resolving inside it is the case a
    ``startswith`` check waves through.
    """
    try:
        return Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return Path(os.path.normpath(os.path.abspath(str(path))))


def contained_by(candidate: Path, root: Path) -> bool:
    """True when *candidate* is *root* itself or lives beneath it.

    ``~/.synapt/channels-backup`` is not beneath ``~/.synapt/channels`` even
    though its text starts with it. The boundary is closed at the root itself,
    so the root is refused too rather than only its children.
    """
    c = _normalise(candidate)
    r = _normalise(root)
    return c == r or r in c.parents


class StoreIsolationPolicy:
    """Decides whether a resolved store path may be created or opened.

    Two protected sets, deliberately asymmetric:

    * ``_permanent`` always contains the real home store and can never be
      removed. A test may *add* a decoy root to witness refusal mechanics
      without ever being able to *remove* the real one — so the witnesses can
      prove prevention without any of them attempting a write at the real
      store.
    * ``_extra`` holds those per-test decoys and is unwound after each test.
    """

    def __init__(self) -> None:
        self._permanent: list[Path] = [protected_channel_root()]
        self._extra: list[Path] = []
        self.session_root: Path | None = None
        self.session_channel_root: Path | None = None
        self.session_data_root: Path | None = None
        self.allow_live: bool = False
        self.current_item = None
        self._announced: set[str] = set()

    # -- protected-root bookkeeping ---------------------------------------

    def roots(self) -> list[Path]:
        return [*self._permanent, *self._extra]

    def register_extra_root(self, root: Path) -> None:
        self._extra.append(Path(root))

    def clear_extra_roots(self) -> None:
        self._extra.clear()

    # -- the decision ------------------------------------------------------

    def _nodeid(self) -> str:
        item = self.current_item
        return getattr(item, "nodeid", "<collection>") if item is not None else "<collection>"

    def _is_marked_live(self) -> bool:
        item = self.current_item
        if item is None:
            return False
        return item.get_closest_marker("live_channel_store") is not None

    def check_channel_path(self, operation: str, path: Path) -> None:
        """Refuse a channel-store write target that lands in a protected root."""
        for root in self.roots():
            if contained_by(path, root):
                if self._is_marked_live() and self.allow_live:
                    self._announce(root, path)
                    return
                raise RecallStoreIsolationError(
                    self._message(operation, path, root)
                )

    def check_data_root(self, operation: str, path: Path) -> None:
        """Require an implicit recall data root to be pytest-owned.

        Opposite polarity to the channel check: channels are refused when they
        land somewhere named, data roots are refused unless they land somewhere
        owned. A test that avoids the channel guard entirely can still read the
        operator's real journal and make an assertion depend on history that
        is not in the repository.
        """
        if self.session_data_root is None:
            return
        if contained_by(path, self.session_data_root):
            return
        if contained_by(path, _tmp_root()):
            return
        raise RecallStoreIsolationError(
            "recall test isolation violation\n"
            f"nodeid={self._nodeid()}\n"
            f"operation={operation}\n"
            f"resolved={_normalise(path)}\n"
            f"expected under={self.session_data_root}\n"
            "an implicit recall data path escaped the pytest-owned root; pass a "
            "tmp_path-based project_dir, or set SYNAPT_RECALL_ROOT to a "
            "pytest-owned directory"
        )

    # -- reporting ---------------------------------------------------------

    def _message(self, operation: str, path: Path, root: Path) -> str:
        """The refusal text.

        Names the resolved candidate and the test, because the failure is
        usually read by someone who did not write the fixture that caused it.
        The remedy offered is isolation. The opt-in is deliberately absent: a
        message that leads with the bypass turns the guard into a documented
        workaround, and the next reader reaches for the flag instead of fixing
        the fixture.
        """
        return (
            "recall test isolation violation\n"
            f"nodeid={self._nodeid()}\n"
            f"operation={operation}\n"
            f"resolved={_normalise(path)}\n"
            f"protected={_normalise(root)}\n"
            "set SYNAPT_SHARED_CHANNELS_DIR to a pytest-owned directory, or "
            "pass channels_dir=tmp_path/'channels'"
        )

    def _announce(self, root: Path, path: Path) -> None:
        """Say plainly that a test was let through to a real store.

        The opt-in path stays loud so a reviewer can tell an authorized
        integration test from the ordinary isolation contract.
        """
        key = f"{self._nodeid()}:{_normalise(path)}"
        if key in self._announced:
            return
        self._announced.add(key)
        text = (
            f"[recall] {self._nodeid()} is authorized to touch the live "
            f"channel store at {_normalise(root)} "
            f"(--allow-live-channel-store-tests + live_channel_store mark)"
        )
        # Written through the terminal reporter rather than print(): pytest
        # captures stdout and only replays it for FAILING tests, so a printed
        # announcement is invisible in exactly the case it exists for — an
        # authorized write that succeeds. "Loud" has to mean loud on success.
        item = self.current_item
        if item is not None:
            reporter = item.config.pluginmanager.get_plugin("terminalreporter")
            if reporter is not None:
                reporter.write_line(text)
                return
        print(text)


def _tmp_root() -> Path:
    """The platform temporary directory.

    ``tmp_path`` lives beneath this, so an explicit ``project_dir=tmp_path``
    stays allowed without each test having to opt in. It is a safe allowance
    because no real store lives here — the risk this guard exists for is a
    resolved path escaping *into a checkout*, not into a scratch directory.
    """
    import tempfile

    return _normalise(Path(tempfile.gettempdir()))
