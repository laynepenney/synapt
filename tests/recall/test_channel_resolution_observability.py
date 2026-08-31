"""Every channel operation must name the state store it resolved.

The operational failure behind recall#996 returned a confident membership
error from one ``channels.db`` while the intended shared store already held
the membership. Store provenance belongs at the common MCP boundary so new
actions inherit it automatically. Caller identity is deliberately absent:
answering who an agent is belongs to the premium identity layer.
"""

from __future__ import annotations

import os
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
        actions = sorted(get_action_registry().known_actions)
        with (
            patch("synapt.recall.channel._db_path", return_value=state_db),
            patch.object(ActionRegistry, "dispatch", return_value="payload") as dispatch,
        ):
            for action in actions:
                with self.subTest(action=action):
                    result = recall_channel(action=action)
                    self.assertEqual(
                        result,
                        f"Channel state store: {resolved_state_db}\npayload",
                    )

        self.assertEqual(dispatch.call_count, len(actions))

    def test_dispatch_failure_still_names_resolved_store(self) -> None:
        from synapt.recall.actions import ActionRegistry
        from synapt.recall.server import recall_channel

        state_db = Path("/distinct/failure/channels.db")
        resolved_state_db = state_db.resolve()
        with (
            patch("synapt.recall.channel._db_path", return_value=state_db),
            patch.object(ActionRegistry, "dispatch", side_effect=RuntimeError("boom")),
        ):
            result = recall_channel(action="read")

        self.assertEqual(
            result,
            f"Channel state store: {resolved_state_db}\nChannel failed: boom",
        )

    def test_store_resolution_failure_reports_the_original_error(self) -> None:
        from synapt.recall.server import recall_channel

        with patch(
            "synapt.recall.channel._db_path",
            side_effect=RuntimeError("root unavailable"),
        ):
            result = recall_channel(action="read")

        self.assertEqual(result, "Channel failed: root unavailable")


if __name__ == "__main__":
    unittest.main()
