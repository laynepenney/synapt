"""The store-isolation policy is safe to construct off-POSIX.

Ref #955.

The isolation boundary is derived from the passwd database (``pwd`` +
``os.getuid``), which is POSIX-only. ``conftest`` constructs the policy at
import time, so an off-POSIX platform (Windows) used to hit that derivation
during pytest *collection* and fail the entire run with
``ModuleNotFoundError: No module named 'pwd'`` before any skip marker could
apply.

These witnesses run on every platform (they SIMULATE the off-POSIX condition by
patching ``POSIX_ACCOUNT_HOME``), so the POSIX legs prove the fallback that the
Windows CI leg then exercises for real. They complement
``test_store_isolation_guard.py``: there, the few witnesses that derive the
boundary from the passwd home skip off-POSIX through the
``protected_channel_root`` / ``rederive_protected_root`` fixtures, while its
portable witnesses — the data-root guarantees and the decoy-root refusal
mechanics — keep running on every platform. This file pins that the off-POSIX
ABSENCE of the guarantee is inert rather than fatal.
"""

from __future__ import annotations

import os

import pytest

import recall_store_isolation as rsi


def test_construction_off_posix_never_derives_the_boundary(monkeypatch):
    """With no passwd database, the policy constructs without ever computing the
    protected root — the permanent set is empty and the guard is inert.

    Mutation witness: revert ``__init__`` to the unconditional
    ``[protected_channel_root()]`` and the patched ``account_home`` fires, since
    construction would then reach the boundary the way it did during the Windows
    collection failure.
    """
    monkeypatch.setattr(rsi, "POSIX_ACCOUNT_HOME", False)

    def _explode() -> None:
        raise AssertionError("account_home() reached during off-POSIX construction")

    monkeypatch.setattr(rsi, "account_home", _explode)

    policy = rsi.StoreIsolationPolicy()
    assert policy._permanent == []
    assert policy.roots() == []


def test_channel_guard_is_inert_off_posix(monkeypatch):
    """An empty protected set means the channel check refuses nothing — the
    honest behaviour when the POSIX-only guarantee cannot be established."""
    monkeypatch.setattr(rsi, "POSIX_ACCOUNT_HOME", False)
    policy = rsi.StoreIsolationPolicy()
    # Any path at all: with no protected roots there is nothing to be contained by.
    policy.check_channel_path("open", os.getcwd() + "/anything/.synapt/channels/dev.jsonl")


def test_account_home_off_posix_raises_a_clean_error(monkeypatch):
    """Off-POSIX, ``account_home`` raises ``RecallStoreIsolationError`` at the
    point of use — a catchable contract error, not a bare ``ModuleNotFoundError``
    or ``AttributeError`` from ``os.getuid``.

    Mutation witness: remove the ``POSIX_ACCOUNT_HOME`` guard in ``account_home``
    and on a POSIX host this returns the real home instead of raising, going red.
    """
    monkeypatch.setattr(rsi, "POSIX_ACCOUNT_HOME", False)
    with pytest.raises(rsi.RecallStoreIsolationError):
        rsi.account_home()


@pytest.mark.skipif(
    not hasattr(os, "getuid"),
    reason="the POSIX positive control needs a real passwd database",
)
def test_construction_on_posix_still_protects_the_real_root():
    """Positive control: on POSIX the permanent set is the real protected root,
    exactly as before this fix — the off-POSIX branch changes nothing here."""
    policy = rsi.StoreIsolationPolicy()
    assert policy._permanent == [rsi.protected_channel_root()]
    assert policy._permanent  # non-empty
