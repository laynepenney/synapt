"""Meta-witnesses for the repository-global process environment scrub."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from environment_isolation import (
    AMBIENT_PROCESS_ENV_VARS,
    scrub_ambient_process_env,
)


ATTRIBUTION_OMISSION_CASES = (
    "tests/recall/test_attribution.py::TestTranscriptChunkAttribution::test_chunk_agent_id_defaults_to_none",
    "tests/recall/test_attribution.py::TestStorageAttribution::test_save_and_load_preserves_agent_id",
    "tests/recall/test_attribution.py::TestScopedSearch::test_lookup_legacy_chunks_visible_to_all",
    "tests/recall/test_attribution.py::TestAttributionMigration::test_existing_db_without_agent_id_still_works",
)


def test_shared_helper_owns_every_declared_variable(monkeypatch):
    """One list drives every consumer while unrelated process state survives."""
    required = {
        "SYNAPT_SHARED_CHANNELS_DIR",
        "SYNAPT_RECALL_ROOT",
        "SYNAPT_RECALL_WORKTREE",
        "GRIPSPACE_ROOT",
        "SYNAPT_AGENT_ID",
    }
    assert set(AMBIENT_PROCESS_ENV_VARS) == required

    for variable in AMBIENT_PROCESS_ENV_VARS:
        monkeypatch.setenv(variable, f"poisoned-{variable.lower()}")
    monkeypatch.setenv("SYNAPT_UNRELATED_CONTROL", "preserved")

    scrub_ambient_process_env(monkeypatch)

    assert all(os.environ.get(variable) is None for variable in AMBIENT_PROCESS_ENV_VARS)
    assert os.environ["SYNAPT_UNRELATED_CONTROL"] == "preserved"


def test_agent_session_identity_is_scrubbed_before_attribution_contracts():
    """The four omission contracts stay green inside a real agent shell."""
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["SYNAPT_AGENT_ID"] = "ambient-agent-that-must-not-own-fixtures"

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *ATTRIBUTION_OMISSION_CASES],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
