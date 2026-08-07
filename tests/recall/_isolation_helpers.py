"""Store-isolation helpers for ``unittest.TestCase`` recall tests.

Ref #967.

Pytest-style tests use the ``owned_recall_root`` fixture in ``conftest.py``.
``unittest.TestCase`` methods cannot receive fixtures, so those classes reach
for this instead — one helper rather than a hand-rolled setUp/tearDown pair per
class, because five copies of an environment save/restore is five chances to
restore the wrong thing.

The distinction the fixture docstring draws applies here identically: this is
for a test that NEEDS a store, not for one measuring where a store is inferred
from. A test genuinely asserting inference should chdir to an owned directory
so the inference still runs, just from a starting point it owns.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_VARS = ("SYNAPT_RECALL_ROOT", "SYNAPT_RECALL_WORKTREE", "SYNAPT_SHARED_CHANNELS_DIR")


class OwnedStore:
    """A restorable set of store overrides pointing at a throwaway directory.

    Restores the PREVIOUS value rather than deleting, because these tests run
    under an autouse fixture that has already removed any ambient override —
    deleting on teardown would work today and silently diverge from that
    fixture's intent the moment it changes.
    """

    def __init__(self, root: Path, previous: dict[str, str | None]) -> None:
        self.root = root
        self._previous = previous

    def restore(self) -> None:
        for name, prev in self._previous.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev


def owned_store() -> OwnedStore:
    """Point recall data and channel resolution at a directory this test owns.

    The data root is created before it is pointed at: the override refuses a
    root that does not exist, on the grounds that silently minting a fresh
    store under a mistyped path presents an empty history as a real answer.
    """
    base = Path(tempfile.mkdtemp())
    data_root = base / "recall-root"
    data_root.mkdir(parents=True, exist_ok=True)

    previous = {name: os.environ.get(name) for name in _VARS}
    os.environ["SYNAPT_RECALL_ROOT"] = str(data_root)
    os.environ["SYNAPT_RECALL_WORKTREE"] = "pytest-owned"
    os.environ["SYNAPT_SHARED_CHANNELS_DIR"] = str(base / "channels")
    return OwnedStore(base, previous)
