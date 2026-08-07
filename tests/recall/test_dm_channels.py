"""Tests for direct messages between agents — private 1:1 channels.

TDD spec for recall#488 (DMs between agents). These tests define
the contract for private agent-to-agent messaging via the dm: channel prefix.

All tests are expected to FAIL until the implementation lands.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestDMChannelNaming(unittest.TestCase):
    """DM channels should use a canonical sorted-pair naming scheme."""

    def test_dm_channel_name_is_sorted_pair(self):
        """dm:atlas from opus should resolve to the same channel as dm:opus from atlas."""
        from synapt.recall.channel import resolve_dm_channel

        # Both directions should produce the same canonical name
        name_a = resolve_dm_channel("opus", "atlas")
        name_b = resolve_dm_channel("atlas", "opus")
        self.assertEqual(name_a, name_b)

    def test_dm_channel_name_format(self):
        """DM channel names should follow dm:{agent_a}:{agent_b} (sorted)."""
        from synapt.recall.channel import resolve_dm_channel

        name = resolve_dm_channel("opus", "atlas")
        self.assertEqual(name, "dm:atlas:opus")

    def test_dm_channel_name_is_deterministic(self):
        """Same pair always produces the same channel name."""
        from synapt.recall.channel import resolve_dm_channel

        results = {resolve_dm_channel("sentinel", "opus") for _ in range(10)}
        self.assertEqual(len(results), 1)

    def test_dm_self_is_rejected(self):
        """An agent cannot DM itself."""
        from synapt.recall.channel import resolve_dm_channel

        with self.assertRaises(ValueError):
            resolve_dm_channel("opus", "opus")


class TestDMPostAndRead(unittest.TestCase):
    """DMs should use the same post/read mechanics as regular channels."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # SYNAPT_PROJECT_DIR is read nowhere in the package, so it isolated
        # nothing — these tests resolved to the real home-level channel store.
        # SYNAPT_SHARED_CHANNELS_DIR is the override the resolver actually
        # consults (Ref #955).
        self._env_patch = patch.dict(os.environ, {
            "SYNAPT_SHARED_CHANNELS_DIR": str(Path(self.tmpdir) / "channels"),
        })
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_post_to_dm_channel(self):
        """Posting to dm:atlas should create a message in the DM channel."""
        from synapt.recall.channel import channel_post, resolve_dm_channel

        dm_channel = resolve_dm_channel("opus", "atlas")
        result = channel_post(
            channel=dm_channel,
            message="thoughts on the worktree model?",
            display_name="opus",
        )
        self.assertIsInstance(result, str)
        self.assertNotIn("error", result.lower())

    def test_read_dm_channel(self):
        """Reading a DM channel should return messages between the two agents."""
        from synapt.recall.channel import channel_post, channel_read, resolve_dm_channel

        dm_channel = resolve_dm_channel("opus", "atlas")
        channel_post(
            channel=dm_channel,
            message="design review?",
            display_name="opus",
        )
        result = channel_read(channel=dm_channel, limit=10)
        self.assertIn("design review", result.lower())

    def test_dm_shorthand_resolves_to_canonical(self):
        """action='post', channel='dm:atlas' should resolve via the sender's identity."""
        from synapt.recall.channel import resolve_dm_channel_from_shorthand

        # When opus types channel="dm:atlas", we need to resolve that
        # to the canonical dm:atlas:opus channel name
        resolved = resolve_dm_channel_from_shorthand("dm:atlas", sender="opus")
        self.assertEqual(resolved, "dm:atlas:opus")

    def test_dm_shorthand_symmetric(self):
        """dm:opus from atlas resolves to the same channel as dm:atlas from opus."""
        from synapt.recall.channel import resolve_dm_channel_from_shorthand

        a = resolve_dm_channel_from_shorthand("dm:atlas", sender="opus")
        b = resolve_dm_channel_from_shorthand("dm:opus", sender="atlas")
        self.assertEqual(a, b)


class TestDMPrivacy(unittest.TestCase):
    """DMs should only be readable by the two participants."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # SYNAPT_PROJECT_DIR is read nowhere in the package, so it isolated
        # nothing — these tests resolved to the real home-level channel store.
        # SYNAPT_SHARED_CHANNELS_DIR is the override the resolver actually
        # consults (Ref #955).
        self._env_patch = patch.dict(os.environ, {
            "SYNAPT_SHARED_CHANNELS_DIR": str(Path(self.tmpdir) / "channels"),
        })
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_dm_not_visible_in_channel_list_for_non_participants(self):
        """DM channels should not appear in channel_list_channels for non-participants."""
        from synapt.recall.channel import (
            channel_list_channels,
            channel_post,
            resolve_dm_channel,
        )

        dm_channel = resolve_dm_channel("opus", "atlas")
        channel_post(channel=dm_channel, message="private msg", display_name="opus")

        # sentinel is not a participant — DM should not appear in their list
        channels = channel_list_channels()
        dm_channels = [c for c in channels if c.startswith("dm:")]
        # DMs should either not appear at all in the global list,
        # or be filtered when queried by a non-participant
        # Implementation may choose either approach
        self.assertTrue(
            len(dm_channels) == 0 or dm_channel not in dm_channels,
            "DM channel should not be visible to non-participants in channel list"
        )

    def test_dm_readable_by_participant(self):
        """A DM participant should be able to read their DM channel."""
        from synapt.recall.channel import (
            channel_post,
            channel_read,
            is_dm_participant,
            resolve_dm_channel,
        )

        dm_channel = resolve_dm_channel("opus", "atlas")
        channel_post(channel=dm_channel, message="secret plan", display_name="opus")

        self.assertTrue(is_dm_participant(dm_channel, "opus"))
        self.assertTrue(is_dm_participant(dm_channel, "atlas"))
        self.assertFalse(is_dm_participant(dm_channel, "sentinel"))

    def test_dm_not_in_global_unread(self):
        """DMs should not appear in the global unread check for non-participants."""
        from synapt.recall.channel import (
            channel_post,
            resolve_dm_channel,
        )

        dm_channel = resolve_dm_channel("opus", "atlas")
        channel_post(channel=dm_channel, message="private", display_name="opus")

        # Non-participant unread should not include this DM
        # (exact API depends on how unread is scoped — may need agent_id param)


class TestDMDiscovery(unittest.TestCase):
    """Agents should be able to discover their own DM conversations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # SYNAPT_PROJECT_DIR is read nowhere in the package, so it isolated
        # nothing — these tests resolved to the real home-level channel store.
        # SYNAPT_SHARED_CHANNELS_DIR is the override the resolver actually
        # consults (Ref #955).
        self._env_patch = patch.dict(os.environ, {
            "SYNAPT_SHARED_CHANNELS_DIR": str(Path(self.tmpdir) / "channels"),
        })
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_list_my_dms(self):
        """An agent should be able to list their active DM conversations."""
        from synapt.recall.channel import (
            channel_post,
            list_dm_channels,
            resolve_dm_channel,
        )

        dm1 = resolve_dm_channel("opus", "atlas")
        dm2 = resolve_dm_channel("opus", "sentinel")
        channel_post(channel=dm1, message="hello atlas", display_name="opus")
        channel_post(channel=dm2, message="hello sentinel", display_name="opus")

        # opus should see both DM channels
        opus_dms = list_dm_channels(agent_id="opus")
        self.assertEqual(len(opus_dms), 2)

        # atlas should see only their DM with opus
        atlas_dms = list_dm_channels(agent_id="atlas")
        self.assertEqual(len(atlas_dms), 1)
        self.assertIn(dm1, atlas_dms)

    def test_list_dms_returns_empty_for_no_conversations(self):
        """An agent with no DMs should get an empty list."""
        from synapt.recall.channel import list_dm_channels

        dms = list_dm_channels(agent_id="apollo")
        self.assertEqual(len(dms), 0)


class TestDMChannelDetection(unittest.TestCase):
    """The system should be able to distinguish DM channels from regular channels."""

    def test_is_dm_channel(self):
        """Channels prefixed with dm: should be detected as DM channels."""
        from synapt.recall.channel import is_dm_channel

        self.assertTrue(is_dm_channel("dm:atlas:opus"))
        self.assertTrue(is_dm_channel("dm:opus:sentinel"))
        self.assertFalse(is_dm_channel("dev"))
        self.assertFalse(is_dm_channel("general"))
        self.assertFalse(is_dm_channel("dm-discussion"))

    def test_dm_participants_from_channel_name(self):
        """Should extract the two participant names from a DM channel name."""
        from synapt.recall.channel import dm_participants

        a, b = dm_participants("dm:atlas:opus")
        self.assertEqual({a, b}, {"atlas", "opus"})

    def test_dm_participants_invalid_channel_raises(self):
        """Extracting participants from a non-DM channel should raise."""
        from synapt.recall.channel import dm_participants

        with self.assertRaises(ValueError):
            dm_participants("dev")


class TestDMInRecallSearch(unittest.TestCase):
    """DM content should be searchable by participants but not by others."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # SYNAPT_PROJECT_DIR is read nowhere in the package, so it isolated
        # nothing — these tests resolved to the real home-level channel store.
        # SYNAPT_SHARED_CHANNELS_DIR is the override the resolver actually
        # consults (Ref #955).
        self._env_patch = patch.dict(os.environ, {
            "SYNAPT_SHARED_CHANNELS_DIR": str(Path(self.tmpdir) / "channels"),
        })
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_dm_content_searchable_by_participant(self):
        """DM messages should appear in channel_search for participants."""
        from synapt.recall.channel import (
            channel_post,
            channel_search,
            resolve_dm_channel,
        )

        dm_channel = resolve_dm_channel("opus", "atlas")
        channel_post(
            channel=dm_channel,
            message="the worktree migration needs atomic swaps",
            display_name="opus",
        )

        results = channel_search("worktree migration", agent_id="opus")
        bodies = [r["body"] for r in results]
        self.assertTrue(
            any("worktree migration" in b.lower() for b in bodies),
            "DM content should be searchable by participant"
        )

    def test_dm_content_not_searchable_by_non_participant(self):
        """DM messages should NOT appear in channel_search for non-participants."""
        from synapt.recall.channel import (
            channel_post,
            channel_search,
            resolve_dm_channel,
        )

        dm_channel = resolve_dm_channel("opus", "atlas")
        channel_post(
            channel=dm_channel,
            message="secret licensing discussion about vendor X",
            display_name="opus",
        )

        results = channel_search("secret licensing", agent_id="sentinel")
        bodies = [r["body"] for r in results] if results else []
        self.assertFalse(
            any("secret licensing" in b.lower() for b in bodies),
            "DM content should NOT be searchable by non-participants"
        )


if __name__ == "__main__":
    unittest.main()
