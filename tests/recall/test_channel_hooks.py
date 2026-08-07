"""Tests for the channel message hook mechanism (#553).

Verifies:
- Hooks fire on message post
- Custom hooks can be registered
- _clear_message_hooks resets to empty
- Default hooks provide backward-compatible mention/wake behavior
- Hooks are best-effort (exceptions don't break posting)
"""

import os
import tempfile
import unittest

from synapt.recall.channel import (
    ChannelMessage,
    _append_message,
    _clear_message_hooks,
    _default_mention_hook,
    _default_wake_hook,
    _message_posted_hooks,
    register_message_hook,
)


class TestMessageHooks(unittest.TestCase):
    """Tests for the message hook registration and dispatch."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        # SYNAPT_DATA_DIR is read nowhere in the package; it isolated nothing and
        # these tests appended into the real channel store (Ref #955).
        os.environ["SYNAPT_SHARED_CHANNELS_DIR"] = self._tmp
        # Save original hooks
        self._original_hooks = list(_message_posted_hooks)

    def tearDown(self):
        # Restore original hooks
        _clear_message_hooks()
        for hook in self._original_hooks:
            register_message_hook(hook)
        os.environ.pop("SYNAPT_SHARED_CHANNELS_DIR", None)

    def test_default_hooks_registered(self):
        """Default mention and wake hooks should be registered at import time."""
        self.assertIn(_default_mention_hook, _message_posted_hooks)
        self.assertIn(_default_wake_hook, _message_posted_hooks)

    def test_clear_hooks(self):
        """_clear_message_hooks should empty the hook list."""
        self.assertGreater(len(_message_posted_hooks), 0)
        _clear_message_hooks()
        self.assertEqual(len(_message_posted_hooks), 0)

    def test_register_custom_hook(self):
        """register_message_hook should add a hook to the list."""
        _clear_message_hooks()
        calls = []

        def my_hook(msg, project_dir):
            calls.append(msg.body)

        register_message_hook(my_hook)
        self.assertIn(my_hook, _message_posted_hooks)

    def test_hooks_fire_on_append(self):
        """Hooks should fire when _append_message is called."""
        _clear_message_hooks()
        calls = []

        def my_hook(msg, project_dir):
            calls.append(msg.body)

        register_message_hook(my_hook)

        msg = ChannelMessage(
            timestamp="2026-04-11T12:00:00Z",
            from_agent="s_test",
            from_display="Test",
            channel="test-hooks",
            type="message",
            body="hello from hooks test",
        )
        _append_message(msg, project_dir=None)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], "hello from hooks test")

    def test_hook_exception_does_not_break_post(self):
        """A hook that raises should not prevent other hooks or the post."""
        _clear_message_hooks()
        calls = []

        def bad_hook(msg, project_dir):
            raise RuntimeError("boom")

        def good_hook(msg, project_dir):
            calls.append(msg.body)

        register_message_hook(bad_hook)
        register_message_hook(good_hook)

        msg = ChannelMessage(
            timestamp="2026-04-11T12:00:00Z",
            from_agent="s_test",
            from_display="Test",
            channel="test-hooks",
            type="message",
            body="should still work",
        )
        # Should not raise
        _append_message(msg, project_dir=None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], "should still work")

    def test_multiple_hooks_fire_in_order(self):
        """Multiple hooks should fire in registration order."""
        _clear_message_hooks()
        order = []

        register_message_hook(lambda msg, pd: order.append("first"))
        register_message_hook(lambda msg, pd: order.append("second"))
        register_message_hook(lambda msg, pd: order.append("third"))

        msg = ChannelMessage(
            timestamp="2026-04-11T12:00:00Z",
            from_agent="s_test",
            from_display="Test",
            channel="test-hooks",
            type="message",
            body="order test",
        )
        _append_message(msg, project_dir=None)
        self.assertEqual(order, ["first", "second", "third"])


class TestActionRegistryGating(unittest.TestCase):
    """Tests for coordination action gating without premium."""

    def setUp(self):
        from synapt.recall.actions import reset_action_registry
        reset_action_registry()

    def tearDown(self):
        from synapt.recall.actions import reset_action_registry
        reset_action_registry()

    def test_oss_actions_available(self):
        """OSS actions should be available without premium."""
        from synapt.recall.actions import get_action_registry

        reg = get_action_registry()
        oss_actions = {"join", "leave", "post", "read", "read_message",
                       "who", "heartbeat", "unread", "pin", "unpin",
                       "list", "search", "rename"}
        for action in oss_actions:
            self.assertEqual(reg.status(action), "available",
                             f"{action} should be available")

    def test_coordination_actions_locked_without_premium(self):
        """Coordination actions should be locked without premium plugin."""
        from synapt.recall.actions import get_action_registry, PREMIUM_ACTION_NAMES

        reg = get_action_registry()
        for action in PREMIUM_ACTION_NAMES:
            self.assertEqual(reg.status(action), "locked",
                             f"{action} should be locked without premium")

    def test_coordination_actions_available_after_registration(self):
        """Coordination actions should be available after register_coordination_handlers."""
        from synapt.recall.actions import (
            get_action_registry,
            register_coordination_handlers,
            PREMIUM_ACTION_NAMES,
        )

        register_coordination_handlers()
        reg = get_action_registry()
        for action in PREMIUM_ACTION_NAMES:
            self.assertEqual(reg.status(action), "available",
                             f"{action} should be available after registration")

    def test_coordination_handlers_registered_as_premium_tier(self):
        """Coordination handlers should be registered with tier='premium'."""
        from synapt.recall.actions import (
            get_action_registry,
            register_coordination_handlers,
            PREMIUM_ACTION_NAMES,
        )

        register_coordination_handlers()
        reg = get_action_registry()
        for action in PREMIUM_ACTION_NAMES:
            self.assertEqual(reg.tier(action), "premium",
                             f"{action} should have tier 'premium'")

    def test_locked_action_dispatch_returns_upgrade_message(self):
        """Dispatching a locked action should return an upgrade message."""
        from synapt.recall.actions import get_action_registry

        reg = get_action_registry()
        result = reg.dispatch("directive", message="test", to="opus")
        self.assertIn("requires premium", result)


if __name__ == "__main__":
    unittest.main()
