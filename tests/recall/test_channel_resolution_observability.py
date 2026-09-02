"""Every channel operation must name the state store it resolved.

The operational failure behind recall#996 returned a confident membership
error from one ``channels.db`` while the intended shared store already held
the membership. Store provenance belongs at the common MCP boundary so new
actions inherit it automatically. Caller identity is deliberately absent:
answering who an agent is belongs to the premium identity layer.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _isolation_helpers import owned_store


class TestChannelResolutionObservability(unittest.TestCase):
    def setUp(self) -> None:
        self._store = owned_store()
        self._previous_agent_id = os.environ.get("SYNAPT_AGENT_ID")
        os.environ["SYNAPT_AGENT_ID"] = "sentinel-witness"

        from synapt.recall.actions import reset_action_registry

        reset_action_registry()

    def tearDown(self) -> None:
        from synapt.recall.actions import reset_action_registry

        reset_action_registry()
        if self._previous_agent_id is None:
            os.environ.pop("SYNAPT_AGENT_ID", None)
        else:
            os.environ["SYNAPT_AGENT_ID"] = self._previous_agent_id
        self._store.restore()

    @property
    def expected_state_db(self) -> Path:
        return (
            self._store.root
            / "recall-root"
            / ".synapt"
            / "recall"
            / "channels"
            / "channels.db"
        ).resolve()

    def test_who_names_store_without_inventing_caller_identity(self) -> None:
        from synapt.recall.server import recall_channel

        result = recall_channel(action="who")

        self.assertIn(f"Channel state store: {self.expected_state_db}", result)
        self.assertIn("No agents online.", result)
        self.assertNotIn("Caller identity:", result)

    def test_unread_names_store_on_membership_error_without_identity(self) -> None:
        from synapt.recall.server import recall_channel

        result = recall_channel(action="unread", detail="low")

        self.assertIn(f"Channel state store: {self.expected_state_db}", result)
        self.assertIn("No channel memberships", result)
        self.assertNotIn("Caller identity:", result)

    def test_common_boundary_covers_every_registered_action(self) -> None:
        from synapt.recall.actions import ActionRegistry, get_action_registry
        from synapt.recall.server import recall_channel

        state_db = Path("/distinct/witness/channels.db")
        resolved_state_db = state_db.resolve()
        log_dir = Path("/distinct/witness/log-store")
        resolved_log_dir = log_dir.resolve()
        actions = sorted(get_action_registry().known_actions)
        with (
            patch("synapt.recall.channel._db_path", return_value=state_db),
            patch("synapt.recall.channel._channels_dir", return_value=log_dir),
            patch.object(ActionRegistry, "dispatch", return_value="payload") as dispatch,
        ):
            for action in actions:
                with self.subTest(action=action):
                    result = recall_channel(action=action)
                    self.assertEqual(
                        result,
                        f"Channel state store: {resolved_state_db}\n"
                        f"Channel log store: {resolved_log_dir}\npayload",
                    )

        self.assertEqual(dispatch.call_count, len(actions))

    def test_dispatch_failure_still_names_resolved_store(self) -> None:
        from synapt.recall.actions import ActionRegistry
        from synapt.recall.server import recall_channel

        state_db = Path("/distinct/failure/channels.db")
        resolved_state_db = state_db.resolve()
        log_dir = Path("/distinct/failure/log-store")
        resolved_log_dir = log_dir.resolve()
        with (
            patch("synapt.recall.channel._db_path", return_value=state_db),
            patch("synapt.recall.channel._channels_dir", return_value=log_dir),
            patch.object(ActionRegistry, "dispatch", side_effect=RuntimeError("boom")),
        ):
            result = recall_channel(action="read")

        self.assertEqual(
            result,
            f"Channel state store: {resolved_state_db}\n"
            f"Channel log store: {resolved_log_dir}\nChannel failed: boom",
        )

    def test_store_resolution_failure_reports_the_original_error(self) -> None:
        from synapt.recall.server import recall_channel

        with patch(
            "synapt.recall.channel._db_path",
            side_effect=RuntimeError("root unavailable"),
        ):
            result = recall_channel(action="read")

        self.assertEqual(result, "Channel failed: root unavailable")


class TestLogStoreDivergesFromStateStore(unittest.TestCase):
    """The log directory (Tier-2 global, when a gripspace's manifest
    resolves) and the state db directory (ALWAYS Tier-3 local, by
    channel._db_path's own documented design -- "presence, cursors, pins,
    and mutes are per-gripspace even when channels are shared") can be two
    different directories for the very same channel, from the very same cwd.

    This is the mechanism behind a reported production symptom: a real
    gripspace-local dev.jsonl "stayed behind looking live" after the actual
    live log moved to the org/project Tier-2 path -- recall_channel kept
    printing only the (correctly local) state store, so nothing in its
    output ever told a reader the log itself lives somewhere else. The
    resolver divergence recall#916/#1071 fixed was cwd-vs-GRIPSPACE_ROOT
    disagreement; this is a structural divergence between two DIFFERENT
    resolvers (_channels_dir vs _db_path) that GRIPSPACE_ROOT precedence
    does not touch, because it was never a bug in either resolver
    individually -- _db_path's local-only behavior is deliberate.
    """

    def setUp(self) -> None:
        from synapt.recall.actions import reset_action_registry
        from synapt.recall.core import _gripspace_cache

        self._tmp = tempfile.mkdtemp()
        tmp_path = Path(self._tmp)
        self._fake_home = tmp_path / "home"
        self._fake_home.mkdir()

        self._grip_root = tmp_path / "gripspace"
        (self._grip_root / ".gitgrip" / "spaces" / "main").mkdir(parents=True)
        (self._grip_root / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
        (self._grip_root / ".gitgrip" / "spaces" / "main" / "gripspace.yml").write_text(
            "manifest:\n  url: git@github.com:synapt-dev/synapt-gripspace.git\n"
        )

        self._previous_cwd = os.getcwd()
        os.chdir(self._grip_root)

        self._env_vars = (
            "SYNAPT_RECALL_ROOT",
            "SYNAPT_SHARED_CHANNELS_DIR",
            "GRIPSPACE_ROOT",
            "SYNAPT_AGENT_ID",
        )
        self._previous_env = {name: os.environ.get(name) for name in self._env_vars}
        for name in self._env_vars:
            os.environ.pop(name, None)
        os.environ["SYNAPT_AGENT_ID"] = "sentinel-witness"

        self._home_patch = patch(
            "synapt.recall.core.Path.home", return_value=self._fake_home
        )
        self._home_patch.start()
        _gripspace_cache.clear()

        reset_action_registry()

    def tearDown(self) -> None:
        from synapt.recall.actions import reset_action_registry
        from synapt.recall.core import _gripspace_cache

        reset_action_registry()
        self._home_patch.stop()
        _gripspace_cache.clear()
        os.chdir(self._previous_cwd)
        for name, prev in self._previous_env.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_state_store_and_log_store_can_diverge_and_both_must_be_named(self) -> None:
        from synapt.recall.channel import _channels_dir, _db_path
        from synapt.recall.server import recall_channel

        log_store = _channels_dir().resolve()
        state_store = _db_path().resolve()

        # Sanity: this scenario only demonstrates the defect if the two
        # resolvers actually land in different directories -- Tier-2 global
        # for the log, Tier-3 local (by design) for the state db.
        self.assertNotEqual(
            log_store,
            state_store.parent,
            "fixture did not reproduce the divergence -- log store and "
            "state store already agree, so this test cannot demonstrate "
            "the reported failure mode",
        )

        result = recall_channel(action="who")

        self.assertIn(f"Channel state store: {state_store}", result)
        self.assertIn(
            f"Channel log store: {log_store}",
            result,
            "the live JSONL log's directory diverged from the printed "
            "state store and recall_channel's output never said so -- "
            "exactly the reported failure mode: a reader trusts the "
            "printed path as the whole picture and never learns the log "
            "lives elsewhere",
        )


if __name__ == "__main__":
    unittest.main()
