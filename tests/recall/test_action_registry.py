"""Tests for MCP action registry gating seam — plugin-aware channel dispatch.

TDD spec for the action registry gating feature. These tests define
the contract for how recall_channel actions are registered, dispatched,
and gated across OSS and premium boundaries.

All tests are expected to FAIL until the implementation lands.
"""

import unittest
from unittest.mock import MagicMock, patch

from _isolation_helpers import owned_store




# OSS base actions that must always be available
OSS_ACTIONS = {
    "join", "leave", "post", "read", "read_message", "who",
    "heartbeat", "unread", "pin", "unpin", "list", "search", "rename",
}

# Premium actions that require premium plugin
PREMIUM_ACTIONS = {
    "directive", "claim", "unclaim", "intent", "board",
    "mute", "unmute", "kick", "broadcast",
}


class TestActionRegistryExists(unittest.TestCase):
    """ActionRegistry should be importable and constructable."""

    def test_registry_importable(self):
        """ActionRegistry should be importable from synapt.recall.channel."""
        from synapt.recall.actions import ActionRegistry

    def test_registry_has_register_method(self):
        """Registry should support registering action handlers."""
        from synapt.recall.actions import ActionRegistry

        reg = ActionRegistry()
        self.assertTrue(callable(getattr(reg, "register", None)))

    def test_registry_has_dispatch_method(self):
        """Registry should support dispatching an action by name."""
        from synapt.recall.actions import ActionRegistry

        reg = ActionRegistry()
        self.assertTrue(callable(getattr(reg, "dispatch", None)))


class TestOSSBaseActions(unittest.TestCase):
    """OSS base actions must be registered by default."""

    def setUp(self):
        self._store = owned_store()

    def tearDown(self):
        self._store.restore()

    def test_oss_actions_registered(self):
        """All OSS base actions should be in the default registry."""
        from synapt.recall.actions import get_default_registry

        reg = get_default_registry()
        for action in OSS_ACTIONS:
            self.assertIn(action, reg.actions, f"OSS action '{action}' not registered")

    def test_oss_action_dispatch(self):
        """Dispatching an OSS action should call the registered handler."""
        from synapt.recall.actions import get_default_registry

        reg = get_default_registry()
        # "list" is a simple action that returns channel names
        result = reg.dispatch("list")
        # Should return a string (channel list or "No channels yet")
        self.assertIsInstance(result, str)

    def test_oss_action_handler_receives_kwargs(self):
        """Handlers should receive keyword args passed through dispatch."""
        from synapt.recall.actions import ActionRegistry

        handler = MagicMock(return_value="ok")
        reg = ActionRegistry()
        reg.register("test_action", handler)

        reg.dispatch("test_action", channel="dev", limit=10)
        handler.assert_called_once()
        _, kwargs = handler.call_args
        self.assertEqual(kwargs.get("channel"), "dev")
        self.assertEqual(kwargs.get("limit"), 10)


class TestPremiumActionGating(unittest.TestCase):
    """Premium actions should be gated — clear error without premium."""

    def test_premium_actions_not_in_oss_registry(self):
        """Premium actions should NOT be in the default OSS registry."""
        from synapt.recall.actions import get_default_registry

        reg = get_default_registry()
        for action in PREMIUM_ACTIONS:
            self.assertNotIn(action, reg.actions,
                             f"Premium action '{action}' should not be in OSS registry")

    def test_dispatch_unknown_action_returns_error(self):
        """Dispatching an unregistered action should return a clear error string."""
        from synapt.recall.actions import get_default_registry

        reg = get_default_registry()
        result = reg.dispatch("nonexistent_action")
        self.assertIn("unknown", result.lower())

    def test_dispatch_premium_action_without_premium_returns_upgrade_message(self):
        """Dispatching a premium action on OSS should mention premium/upgrade."""
        from synapt.recall.actions import get_default_registry

        reg = get_default_registry()
        result = reg.dispatch("directive", message="test", to="opus")
        # Should not just say "unknown action" — should indicate it's a premium feature
        self.assertIn("premium", result.lower())

    def test_premium_action_error_names_the_action(self):
        """The error message should name which action requires premium."""
        from synapt.recall.actions import get_default_registry

        reg = get_default_registry()
        result = reg.dispatch("directive", message="test", to="opus")
        self.assertIn("directive", result.lower())


class TestPremiumActionRegistration(unittest.TestCase):
    """A downstream plugin should be able to register additional actions at import time."""

    def test_register_premium_action(self):
        """A downstream plugin can register a new action on the registry."""
        from synapt.recall.actions import ActionRegistry

        reg = ActionRegistry()
        handler = MagicMock(return_value="directive sent")
        reg.register("directive", handler, tier="premium")

        self.assertIn("directive", reg.actions)

    def test_premium_action_dispatch_after_registration(self):
        """After a downstream tier registers an action, it should be dispatchable."""
        from synapt.recall.actions import ActionRegistry

        reg = ActionRegistry()
        handler = MagicMock(return_value="directive sent")
        reg.register("directive", handler, tier="premium")

        result = reg.dispatch("directive", message="build the thing", to="opus")
        self.assertEqual(result, "directive sent")
        handler.assert_called_once()

    def test_premium_override_oss_action(self):
        """A downstream tier can override an OSS action with enriched behavior."""
        from synapt.recall.actions import ActionRegistry

        oss_handler = MagicMock(return_value="oss who")
        premium_handler = MagicMock(return_value="premium who with roles")

        reg = ActionRegistry()
        reg.register("who", oss_handler, tier="oss")
        reg.register("who", premium_handler, tier="premium")

        result = reg.dispatch("who")
        self.assertEqual(result, "premium who with roles")

    def test_override_does_not_silently_fail(self):
        """If a premium override fails, it should raise, not fall back silently."""
        from synapt.recall.actions import ActionRegistry

        oss_handler = MagicMock(return_value="oss fallback")
        premium_handler = MagicMock(side_effect=RuntimeError("premium broke"))

        reg = ActionRegistry()
        reg.register("who", oss_handler, tier="oss")
        reg.register("who", premium_handler, tier="premium")

        with self.assertRaises(RuntimeError):
            reg.dispatch("who")


class TestActionMetadata(unittest.TestCase):
    """Actions should carry metadata for discovery and documentation."""

    def test_action_has_tier(self):
        """Each registered action should know its tier (oss/premium)."""
        from synapt.recall.actions import ActionRegistry

        reg = ActionRegistry()
        reg.register("join", lambda: "ok", tier="oss")
        reg.register("directive", lambda: "ok", tier="premium")

        self.assertEqual(reg.tier("join"), "oss")
        self.assertEqual(reg.tier("directive"), "premium")

    def test_list_actions_by_tier(self):
        """Registry should list actions filtered by tier."""
        from synapt.recall.actions import ActionRegistry

        reg = ActionRegistry()
        reg.register("join", lambda: "ok", tier="oss")
        reg.register("post", lambda: "ok", tier="oss")
        reg.register("directive", lambda: "ok", tier="premium")

        oss = reg.actions_by_tier("oss")
        premium = reg.actions_by_tier("premium")

        self.assertEqual(oss, {"join", "post"})
        self.assertEqual(premium, {"directive"})

    def test_list_all_known_actions(self):
        """Registry should list all known action names (for help/docs)."""
        from synapt.recall.actions import ActionRegistry

        reg = ActionRegistry()
        reg.register("join", lambda: "ok", tier="oss")
        reg.register("directive", lambda: "ok", tier="premium")

        # known_actions should include premium stubs even when handlers aren't registered
        self.assertIn("join", reg.actions)
        self.assertIn("directive", reg.actions)

    def test_action_description(self):
        """Actions can have an optional description for discoverability."""
        from synapt.recall.actions import ActionRegistry

        reg = ActionRegistry()
        reg.register("join", lambda: "ok", tier="oss",
                     description="Join a channel")

        self.assertEqual(reg.description("join"), "Join a channel")


class TestPremiumStubs(unittest.TestCase):
    """OSS registry should know about premium actions even without premium installed."""

    def test_oss_registry_knows_premium_action_names(self):
        """The default registry should list premium action names as known-but-locked."""
        from synapt.recall.actions import get_default_registry

        reg = get_default_registry()
        known = reg.known_actions
        for action in PREMIUM_ACTIONS:
            self.assertIn(action, known,
                          f"Premium action '{action}' should be listed as known")

    def test_known_premium_action_shows_as_locked(self):
        """A known-but-unregistered premium action should show as locked, not unknown."""
        from synapt.recall.actions import get_default_registry

        reg = get_default_registry()
        self.assertEqual(reg.status("directive"), "locked")
        self.assertEqual(reg.status("join"), "available")

    def test_truly_unknown_action_shows_as_unknown(self):
        """An action that's not even in the known list should show as unknown."""
        from synapt.recall.actions import get_default_registry

        reg = get_default_registry()
        self.assertEqual(reg.status("totally_fake_action"), "unknown")


class TestRecallChannelIntegration(unittest.TestCase):
    """The live MCP tool should dispatch through the shared action registry."""

    def setUp(self):
        # These dispatch through the real channel post path. With no override
        # they resolve to the home-level store and their fixture messages land
        # in live channels (Ref #955).
        self._store = owned_store()

    def tearDown(self):
        from synapt.recall.actions import reset_action_registry

        reset_action_registry()
        self._store.restore()

    def test_recall_channel_uses_registry_dispatch(self):
        """recall_channel should route OSS actions through the shared registry."""
        from synapt.recall.actions import get_action_registry
        from synapt.recall.server import recall_channel

        reg = get_action_registry()
        handler = MagicMock(return_value="joined via registry")
        reg.register("join", handler, tier="oss")

        result = recall_channel(action="join", channel="dev", name="Atlas")
        self.assertIn("Channel state store:", result)
        self.assertTrue(result.endswith("\njoined via registry"))
        _, kwargs = handler.call_args
        self.assertEqual(kwargs["channel"], "dev")
        self.assertEqual(kwargs["name"], "Atlas")

    def test_coordination_actions_gated_without_premium(self):
        """Coordination actions should return 'requires premium' without plugin registered."""
        from synapt.recall.server import recall_channel

        result = recall_channel(action="directive", channel="dev", message="test", to="opus")
        self.assertIn("requires premium", result)

    def test_coordination_actions_work_after_registration(self):
        """Coordination actions should work after register_coordination_handlers() is called."""
        from synapt.recall.actions import register_coordination_handlers
        from synapt.recall.server import recall_channel

        register_coordination_handlers()
        result = recall_channel(action="directive", channel="dev", message="test", to="opus")
        self.assertIn("#dev", result)
        self.assertIn("@opus", result)

    def test_recall_channel_uses_shared_registry_overrides(self):
        """Premium-style overrides on the shared registry should affect the live dispatcher."""
        from synapt.recall.actions import get_action_registry
        from synapt.recall.server import recall_channel

        reg = get_action_registry()
        reg.register("who", lambda **_kwargs: "premium who", tier="premium")

        result = recall_channel(action="who")
        self.assertIn("Channel state store:", result)
        self.assertTrue(result.endswith("\npremium who"))


if __name__ == "__main__":
    unittest.main()
