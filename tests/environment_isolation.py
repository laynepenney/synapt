"""Shared process-environment isolation for the test harness.

Tests inherit the shell that launched pytest.  Values exported by a real
agent session are therefore ambient inputs unless the harness removes them
before each test.  Keep that policy in one importable seam so a new suite does
not have to rediscover which variables change recall semantics.

A test that deliberately exercises one of these variables sets it after the
autouse fixture runs.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest


AMBIENT_PROCESS_ENV_VARS = (
    "SYNAPT_SHARED_CHANNELS_DIR",
    "SYNAPT_RECALL_ROOT",
    "SYNAPT_RECALL_WORKTREE",
    "GRIPSPACE_ROOT",
    "SYNAPT_AGENT_ID",
)


def scrub_ambient_process_env(
    monkeypatch: pytest.MonkeyPatch,
    variables: Iterable[str] = AMBIENT_PROCESS_ENV_VARS,
) -> None:
    """Remove agent-session inputs before a test establishes its own state."""
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)
