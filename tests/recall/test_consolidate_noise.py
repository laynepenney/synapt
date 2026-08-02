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

import tempfile
import unittest
from pathlib import Path

from synapt.recall.consolidate import (
    _apply_consolidation_result,
    _create_content_passes_filters,
    _evaluate_create_content,
    _is_garbled_content,
    _is_generic_node,
    _is_metadata_noise,
    _lacks_specificity,
)
from synapt.recall.journal import JournalEntry
from synapt.recall.knowledge import KnowledgeNode, append_node, read_nodes
from synapt.recall.storage import RecallDB


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

    def test_rejects_journal_row_vocabulary(self):
        # Earns the journal-field predicates (stated focus / focus for /
        # duration / dated). Each of these is a FIELD of a session log row; the
        # id shape gate is what makes them safe, since the same words outside it
        # are ordinary prose.
        self.assertTrue(_is_metadata_noise(
            "Session d0c1e5b7, dated 2024-03-14, had as its stated focus the review "
            "of the pending changes on the current branch."))
        self.assertTrue(_is_metadata_noise(
            "The duration between the first and last recorded events of session "
            "e91b6d3a spanned just under two hours on the evening of 2024-03-22."))
        self.assertTrue(_is_metadata_noise(
            "Focus for session b7d2c904: recap the previous session and write the "
            "journal entry."))

    def test_rejects_the_same_execution_event_on_windows(self):
        # The byte-identical event with a Windows temp path. recall is public and
        # cross-platform; a POSIX-only pattern set passes this while the class
        # STATED in the docstring claims to cover it, so the claim and the code
        # would disagree in a direction no test could see (Atlas, P2).
        self.assertTrue(_is_metadata_noise(
            r"The C:\Users\runner\AppData\Local\Temp\scratch-run directory was "
            r"explicitly removed with Remove-Item -Recurse -Force after the trial "
            r"run on 2026-08-01."))


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

    def test_keeps_decimal_session_id_that_is_accidentally_valid_hex(self):
        # [0-9a-f]{6,} matches six or more DIGITS, so an ordinary decimal
        # constant trips the id shape with no hex character anywhere in it.
        # The verdict has to come from what is PREDICATED of the id, not from
        # what the id looks like.
        self.assertFalse(_is_metadata_noise(
            "Session 604800 is the ceiling the auth service enforces on remember-me "
            "lifetimes, measured in seconds, and a larger value in a tenant config "
            "is silently clamped rather than rejected"))

    def test_keeps_stable_server_path_containing_tmp_as_a_substring(self):
        # "/var/lib/nginx/tmp/client_body" is a stable server directory. An
        # unanchored "/tmp/" matches it as a SUBSTRING -- the ephemeral-path
        # shape has to anchor to a real path start or it fires on any path with
        # a "tmp" component.
        self.assertFalse(_is_metadata_noise(
            "Files in /var/lib/nginx/tmp/client_body are removed by nginx itself "
            "once a request completes, so a cron sweeper pointed at that directory "
            "will race the worker and produce spurious 400s"))

    def test_keeps_deprecation_sense_of_removed(self):
        # "removed" here deletes a FEATURE, not a directory: the object of the
        # verb is "support", and the temp path only sits inside the noun phrase
        # naming it. Shape and verb co-occur without being related to each other.
        self.assertFalse(_is_metadata_noise(
            "Support for the TMPDIR=/tmp/legacy-scratch override was removed in "
            "agent 2.4, and a job that still sets it falls back to the system temp "
            "directory without warning"))

    def test_keeps_hedged_empirical_tendency(self):
        # Calibrated observation: no modal, no quantifier, no conditional. A
        # keep-override built only from modality and quantification misfiles this
        # whole class as episodic.
        self.assertFalse(_is_metadata_noise(
            "In practice /tmp/build-cache is removed within a second of a green "
            "build, but on the ARM runners the reaper is niced so heavily that it "
            "lingers for minutes, which is why the disk alert carries a grace window"))


class TestKeepControlsSurviveTheWholePipeline(unittest.TestCase):
    """The six durable facts two reviewers found this filter eating.

    Asserted at the PIPELINE level, not the predicate level, and each one also
    asserts that the three sibling filters pass it. Without that second half a
    "pipeline keeps it" assertion is ambiguous: it would read as a win for this
    filter even if the content only survived because some other filter happened
    to be lenient, and it would go red for a reason this change cannot fix if
    some other filter tightened later.
    """

    CASES = {
        "session id with a domain predicate": (
            "Session deadbeef uses AES-GCM to encrypt refresh tokens, rotates "
            "signing keys every 24 hours, and expires after 15 minutes of "
            "inactivity across every authenticated client."),
        "session id as a named fixture": (
            "Session deadbeef is the stable OAuth replay fixture used by GitHub "
            "Actions to verify token rotation across the production "
            "authentication boundary."),
        "dated command convention": (
            "Execute /migrate after every schema upgrade starting 2026-08-01; "
            "keep the operation idempotent and verify schema version 12 before "
            "serving production traffic."),
        "future-dated instruction": (
            "Execute /migrate on 2027-01-15 only after the PostgreSQL backup has "
            "completed and the migration owner has approved the production "
            "window."),
        "temp-path lifecycle rule": (
            "The /tmp/build-cache directory is removed only after artifact upload "
            "succeeds, retained after failures, and recreated with mode 0700 "
            "before the next build begins."),
        "deletion prohibition": (
            "GitHub Actions must never run rm -rf /tmp/shared-cache because the "
            "path is shared by concurrent release jobs and deletion corrupts "
            "their signed artifacts."),
    }

    def test_every_keep_case_survives_the_real_create_pipeline(self):
        for name, content in self.CASES.items():
            with self.subTest(case=name):
                self.assertFalse(
                    _is_generic_node(content), "sibling filter _is_generic_node")
                self.assertFalse(
                    _lacks_specificity(content), "sibling filter _lacks_specificity")
                self.assertFalse(
                    _is_garbled_content(content), "sibling filter _is_garbled_content")
                self.assertFalse(
                    _is_metadata_noise(content), "this filter")
                self.assertIsNotNone(
                    _evaluate_create_content(content),
                    "survives the predicate but dies in the real create pipeline")


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

    def test_filter_bank_rejects_metadata_noise_regardless_of_action(self):
        # This REPLACES a test that pinned the opposite as desired behaviour.
        #
        # The old test reasoned that corroborate/contradict "carry an existing
        # node's content and are not re-judged for creation quality", matching
        # the convention of every sibling filter. That claim is false of the
        # production flow, and consistency with the siblings was the wrong thing
        # to be consistent with: ``is_create`` is the action the MODEL chose, not
        # a statement about whether this path lets the text become knowledge.
        # TestMetadataNoiseRejectsOnEveryPersistingRoute below drives the real
        # branches that made it false.
        self.assertFalse(_create_content_passes_filters(
            "Session a1b2c3d4, 2020-01-02", is_create=False))


class TestMetadataNoiseRejectsOnEveryPersistingRoute(unittest.TestCase):
    """The four routes through ``_apply_consolidation_result`` that let a
    non-create candidate reach durable knowledge.

    Poking ``_create_content_passes_filters(..., is_create=False)`` is not enough
    to pin this: the two fallthrough routes reach ``action = "create"`` only
    AFTER the filter bank has already run under the model's original label, so
    the bypass lives in the ORDER of the real function and is invisible to any
    test that calls the gate directly.

    Every test here carries a POSITIVE CONTROL running the identical route with
    a durable fact. Without it, each assertion would pass just as well against a
    route that was broken outright, and "nothing was persisted" would read as
    the guard working when it might mean the code never ran.
    """

    NOISE = "Session a1b2c3d4, 2020-01-02"
    DURABLE = "The indexer consumes the tokenizer package for parsing as of 2020-01-02"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.kn = self.tmp / "knowledge.jsonl"
        self.cluster = [
            JournalEntry(session_id="s-new", timestamp="2026-03-01", focus="testing"),
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def _existing(self, content="use unittest", sources=("s-old",)):
        node = KnowledgeNode.create(content, "tooling", source_sessions=list(sources))
        append_node(node, self.kn)
        return node

    def _apply(self, action, content, existing_id, existing, db=None):
        parsed = {"nodes": [{
            "action": action,
            "existing_id": existing_id,
            "content": content,
            "category": "tooling",
            "contradiction_note": "superseded",
            "tags": [],
        }]}
        return _apply_consolidation_result(
            parsed, [existing] if existing else [], self.cluster, self.kn, db=db,
        )

    # -- route 1: legacy contradiction against a real target, no DB ----------
    # The candidate becomes the new ACTIVE node and the old one is marked
    # contradicted. Bookkeeping must not be able to retire a real memory.

    def test_legacy_contradict_does_not_persist_bookkeeping(self):
        existing = self._existing()
        result = self._apply("contradict", self.NOISE, existing.id, existing)
        contents = [n.content for n in read_nodes(self.kn)]
        self.assertNotIn(self.NOISE, contents)
        self.assertEqual(result.nodes_created, 0)
        self.assertEqual(result.nodes_contradicted, 0)
        self.assertTrue(all(n.status == "active" for n in read_nodes(self.kn)))

    def test_legacy_contradict_still_works_for_a_durable_fact(self):
        existing = self._existing()
        result = self._apply("contradict", self.DURABLE, existing.id, existing)
        contents = [n.content for n in read_nodes(self.kn)]
        self.assertIn(self.DURABLE, contents)
        self.assertEqual(result.nodes_contradicted, 1)

    # -- route 2: corroboration against a real target ------------------------
    # The candidate's TEXT is not persisted here, but the candidate's session is
    # added as a source and confidence rises. Content that must never become
    # knowledge must not be able to serve as evidence FOR knowledge either --
    # a bookkeeping line is not a second witness to anything.

    def test_corroborate_does_not_let_bookkeeping_raise_confidence(self):
        existing = self._existing()
        before = existing.confidence
        result = self._apply("corroborate", self.NOISE, existing.id, existing)
        after = read_nodes(self.kn)[0]
        self.assertEqual(result.nodes_corroborated, 0)
        self.assertEqual(after.confidence, before)
        self.assertNotIn("s-new", after.source_sessions)

    def test_corroborate_still_works_for_a_durable_fact(self):
        existing = self._existing()
        before = existing.confidence
        result = self._apply("corroborate", self.DURABLE, existing.id, existing)
        after = read_nodes(self.kn)[0]
        self.assertEqual(result.nodes_corroborated, 1)
        self.assertGreater(after.confidence, before)

    # -- routes 3 and 4: missing target falls through to create --------------
    # These are the bypass proper: the filter bank already ran with
    # is_create=False, then the action is reassigned to "create" and the
    # candidate is persisted through a gate that declined to look at it.

    def test_missing_target_corroborate_does_not_create_bookkeeping(self):
        existing = self._existing()
        result = self._apply("corroborate", self.NOISE, "kn_does_not_exist", existing)
        self.assertNotIn(self.NOISE, [n.content for n in read_nodes(self.kn)])
        self.assertEqual(result.nodes_created, 0)

    def test_missing_target_corroborate_still_creates_a_durable_fact(self):
        existing = self._existing()
        result = self._apply("corroborate", self.DURABLE, "kn_does_not_exist", existing)
        self.assertIn(self.DURABLE, [n.content for n in read_nodes(self.kn)])
        self.assertEqual(result.nodes_created, 1)

    def test_missing_target_contradict_does_not_create_bookkeeping(self):
        existing = self._existing()
        result = self._apply("contradict", self.NOISE, "kn_does_not_exist", existing)
        self.assertNotIn(self.NOISE, [n.content for n in read_nodes(self.kn)])
        self.assertEqual(result.nodes_created, 0)

    def test_missing_target_contradict_still_creates_a_durable_fact(self):
        existing = self._existing()
        result = self._apply("contradict", self.DURABLE, "kn_does_not_exist", existing)
        self.assertIn(self.DURABLE, [n.content for n in read_nodes(self.kn)])
        self.assertEqual(result.nodes_created, 1)

    # -- route 5: DB contradiction queues the candidate for later apply ------
    # Queued, not persisted -- but a confirm-time materialization writes it, so
    # the gate has to hold before it reaches the queue rather than after.

    def test_db_contradict_does_not_queue_bookkeeping(self):
        db = RecallDB(self.tmp / "test.db")
        existing = self._existing()
        db.save_knowledge_nodes([existing.to_dict()])
        result = self._apply("contradict", self.NOISE, existing.id, existing, db=db)
        self.assertEqual(db.list_pending_contradictions(), [])
        self.assertEqual(result.nodes_contradicted, 0)

    def test_db_contradict_still_queues_a_durable_fact(self):
        db = RecallDB(self.tmp / "test.db")
        existing = self._existing()
        db.save_knowledge_nodes([existing.to_dict()])
        result = self._apply("contradict", self.DURABLE, existing.id, existing, db=db)
        pending = db.list_pending_contradictions()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["new_content"], self.DURABLE)
        self.assertEqual(result.nodes_contradicted, 1)


if __name__ == "__main__":
    unittest.main()
