"""Session/command bookkeeping rejection at the envelope-to-node boundary.

The extract path produces a junk class the legacy prompt used to suppress:
session-ID/date bookkeeping and one-off command echoes dressed up as durable
facts. ``_lacks_specificity`` does not catch them -- it treats a bare date as a
specificity *signal*, so a bare "Session <hex-id>, <date>" tuple passes straight
through it.

This spec pins a filter that targets the metadata SHAPE, in both directions.
The KEEP half is the load-bearing half: a filter that rejects everything passes
every REJECT case and silently destroys real memory.

FIXTURES ARE SYNTHETIC. The SHAPES below (hex-id + date, dated command echo,
ephemeral temp path + deletion verb) are drawn from observed extract-path
output, but every string is invented -- no real session hash, person, path, or
internal detail. Shape is what a pattern filter discriminates on, so synthetic
content costs the spec nothing.
"""

import unittest

from synapt.recall.consolidate import (
    _create_content_passes_filters,
    _evaluate_create_content,
    _is_metadata_noise,
)


class TestIsMetadataNoiseRejects(unittest.TestCase):
    """The junk shape the extract path emits. Each MUST be rejected."""

    def test_rejects_session_id_date_tuple(self):
        self.assertTrue(_is_metadata_noise("Session a1b2c3d4, 2020-01-02"))

    def test_rejects_session_occurred_on_date(self):
        self.assertTrue(_is_metadata_noise(
            "Session beef1234 occurred on 2020-01-02 with focus on clearing a command"))

    def test_rejects_session_from_date(self):
        self.assertTrue(_is_metadata_noise(
            "Session cafe5678 from 2020-01-02 focused on recapping the editor v1.2.3"))

    def test_rejects_execute_command_log(self):
        self.assertTrue(_is_metadata_noise("Execute `/clear` command on 2020-01-02"))
        self.assertTrue(_is_metadata_noise("Execute `/exit` command on 2020-01-03"))

    def test_rejects_raw_focus_command_echo(self):
        self.assertTrue(_is_metadata_noise(
            "Focus: Run this exact shell command and report its exit code: "
            "rm -rf /tmp/scratch-run-00000"))

    def test_rejects_bare_session_marker(self):
        self.assertTrue(_is_metadata_noise("Session a1b2c3d4, 2020-01-02"))
        self.assertTrue(_is_metadata_noise("Session a1b2c3d5, 2020-01-02"))

    def test_rejects_reworded_execution_event(self):
        # A small model rewords forbidden junk into declarative prose that an
        # anchored regex misses. A shape filter must catch the reworded form too,
        # or the quality counter reports a win the nodes do not show.
        self.assertTrue(_is_metadata_noise(
            "The /tmp/scratch-run-00000 directory was explicitly wiped with rm -rf "
            "(forceful recursive deletion) to ensure no lingering artifacts from the "
            "trial run, with exit code 0 confirmed."))


class TestIsMetadataNoiseKeeps(unittest.TestCase):
    """Precision controls. A filter that fails these is worse than no filter:
    it deletes real memory while reporting a cleanliness improvement."""

    def test_keeps_real_durable_fact_with_a_date(self):
        # The filter targets metadata SHAPE, never the mere presence of a date.
        self.assertFalse(_is_metadata_noise(
            "The indexer now consumes the tokenizer package for parsing as of 2020-01-02"))

    def test_keeps_synthesized_fact(self):
        self.assertFalse(_is_metadata_noise(
            "A reviewer verified the published listing showed version 1.2.1 "
            "rather than the expected 1.2.3"))

    def test_keeps_preference_fact(self):
        self.assertFalse(_is_metadata_noise("Dana prefers tabs over spaces"))

    def test_keeps_session_config_facts(self):
        # "Session" as a domain noun, not a journal-session hash.
        self.assertFalse(_is_metadata_noise(
            "Session v2 stores OAuth refresh tokens in encrypted cookies"))
        self.assertFalse(_is_metadata_noise("Session 30 timeout is 15 minutes"))

    def test_keeps_durable_command_convention(self):
        # "Execute /migrate ..." with no date is a durable rule, not a dated log.
        self.assertFalse(_is_metadata_noise("Execute /migrate after every schema upgrade"))

    def test_keeps_focus_prefixed_real_fact(self):
        self.assertFalse(_is_metadata_noise(
            "Focus: visible focus rings are required for keyboard accessibility"))

    def test_keeps_rm_rf_safety_rule(self):
        # A durable RULE about rm -rf (no ephemeral temp path) is not an event.
        #
        # NOTE: this string does NOT survive a create today. _lacks_specificity
        # runs EARLIER in the same is_create block and rejects it, so the node
        # dies before this filter is reached -- a separate pre-existing precision
        # defect, tracked on its own. That is why this guarantee is asserted at
        # the PREDICATE level only: a pipeline assertion would go red for a
        # reason this filter cannot fix, and would read as a failure of this
        # change. A keep guarantee on ONE filter is not a keep guarantee for the
        # pipeline.
        self.assertFalse(_is_metadata_noise(
            "Run cleanup with rm -rf only inside the generated build directory"))

    def test_session_token_must_be_hex_id_not_any_digit(self):
        # Regression against an over-broad ^session<digit>: a short numeric or
        # version-y token after "Session" is not a journal session hash.
        self.assertFalse(_is_metadata_noise("Session 7 introduced the new auth flow"))


class TestMetadataNoiseIsWiredIntoCreateEvaluation(unittest.TestCase):
    """The predicate existing is not the guarantee -- the guarantee is that the
    create path READS it. A guard whose result nothing consumes does not exist,
    so these travel the real call site rather than poking the predicate."""

    def test_create_path_rejects_session_bookkeeping(self):
        # _evaluate_create_content returns the normalized string on acceptance,
        # None on rejection -- so None IS the rejection assertion.
        self.assertIsNone(_evaluate_create_content("Session a1b2c3d4, 2020-01-02"))

    def test_create_path_rejects_reworded_execution_event(self):
        self.assertIsNone(_evaluate_create_content(
            "The /tmp/scratch-run-00000 directory was explicitly wiped with rm -rf "
            "(forceful recursive deletion) to ensure no lingering artifacts from the "
            "trial run, with exit code 0 confirmed."))

    def test_create_path_keeps_a_real_fact_that_mentions_a_date(self):
        # Precision at the call site, not just in the predicate. Guards against a
        # filter that passes every REJECT case by rejecting everything.
        self.assertIsNotNone(_evaluate_create_content(
            "The indexer now consumes the tokenizer package for parsing as of 2020-01-02"))

    def test_filter_bank_rejects_metadata_noise_on_create(self):
        # The shared seam B3 and B4 both call -- wiring it here is what makes the
        # two paths incapable of drifting apart on this filter.
        self.assertFalse(_create_content_passes_filters(
            "Session a1b2c3d4, 2020-01-02", is_create=True))

    def test_non_create_actions_are_not_subjected_to_the_filter(self):
        # Matches the convention of every sibling filter in the bank, which all
        # gate on is_create. Corroborate/contradict carry an existing node's
        # content and are not re-judged for creation quality.
        self.assertTrue(_create_content_passes_filters(
            "Session a1b2c3d4, 2020-01-02", is_create=False))


if __name__ == "__main__":
    unittest.main()
