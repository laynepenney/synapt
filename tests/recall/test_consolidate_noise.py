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
    _has_unsplit_boundary_residue,
    _lacks_specificity,
    _PROPOSITION_SPLIT_RE,
    _split_into_propositions,
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

    def test_a_multi_colon_command_echo_is_now_a_DOCUMENTED_FALSE_KEEP(self):
        # THIS TEST ASSERTED THE OPPOSITE UNTIL ROUND 6, AND THE CHANGE IS
        # RATIFIED RATHER THAN CONCEDED.
        #
        # This string is genuine bookkeeping and it now survives. It carries a
        # colon with material after it INSIDE a clause that would otherwise be
        # class B, which is exactly the signal that says the segmenter may have
        # missed a proposition boundary -- so the clause is demoted to unknown,
        # and unknown keeps.
        #
        # The alternative was retaining a conditional colon-split rule that a
        # reviewer proved wrong in BOTH directions: it refused to split
        # lowercase prose after a colon (deleting a durable production
        # invariant) and it DID split a plural noun read as a finite verb
        # (rescuing pure bookkeeping). Trading one junk node for a class of
        # silent deletions is the wrong side of the asymmetry, and the reviewer
        # said so explicitly: "do not preserve three reject counts by retaining
        # a discriminator that demonstrably deletes durable clauses."
        #
        # Recorded here as a limitation, not left to be rediscovered as a
        # regression. See limitation (d) in the _is_metadata_noise docstring.
        self.assertFalse(_is_metadata_noise(
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
        #
        # DELIBERATELY MARKER-FREE. The natural way to write this fixture ("...
        # removed by nginx itself ONCE a request completes") carries a
        # conditional, so the standing-rule override rescues it and the
        # assertion holds whether or not the anchor exists -- it would pass
        # against the very bug it names. Mutating the anchor away was run and
        # left this test green until the marker was removed. A precision control
        # has to fail for the reason it claims, so every keep-marker is stripped
        # and the anchor is the ONLY thing standing between this string and
        # deletion.
        self.assertFalse(_is_metadata_noise(
            "Files in /var/lib/nginx/tmp/client_body are removed by the worker "
            "process itself, immediately after the response body is flushed to "
            "the client socket"))

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


class TestResultBearingOccurrencesSurvive(unittest.TestCase):
    """An occurrence can carry a durable residue, and when it does the residue
    IS the memory.

    This is the class the whole-string version destroyed: it saw a dated command
    echo or a temp-path deletion in ONE clause and deleted every clause,
    including clauses recording a migration result, a schema state, and an
    incident root cause. A shape match on one clause is evidence about that
    clause only.

    Asserted at the PIPELINE level with the sibling filters also checked, so a
    keep here cannot be credited to this filter when another one earned it.
    """

    CASES = {
        "migration result": (
            "Session deadbeef concluded the migration from SHA-1 to SHA-256; "
            "production rejects SHA-1 signatures across API endpoints and stores "
            "SHA-256 digests for audit verification."),
        "persistent schema state": (
            "The /migrate command ran on 2026-08-01 and permanently upgraded the "
            "tenant schema to version 12, which fixes the lock-order defect in "
            "account creation."),
        "incident root cause": (
            "/tmp/build-cache was removed after artifact upload failed, exposing "
            "the cleanup race that truncated release manifests and corrupted "
            "checksum records for build 42."),
        # The residue's grammatical subject is the SESSION. Class A is about
        # what outlives the occurrence, never about who the subject is --
        # durable facts routinely arrive wearing occurrence clothing.
        "residue with an occurrence subject": (
            "Session deadbeef focused on migration and exposed a lock-order bug."),
    }

    def test_every_result_bearing_occurrence_survives_the_real_pipeline(self):
        for name, content in self.CASES.items():
            with self.subTest(case=name):
                self.assertFalse(_is_generic_node(content), "sibling _is_generic_node")
                self.assertFalse(_lacks_specificity(content), "sibling _lacks_specificity")
                self.assertFalse(_is_garbled_content(content), "sibling _is_garbled_content")
                self.assertFalse(_is_metadata_noise(content), "this filter")
                self.assertIsNotNone(
                    _evaluate_create_content(content),
                    "survives the predicate but dies in the real create pipeline")


class TestEveryClaimedPropositionBoundarySplits(unittest.TestCase):
    """One witness per punctuation form the segmenter claims to treat as a
    proposition boundary.

    A boundary the segmenter does NOT split fuses a durable residue back into
    the occurrence clause and recreates the whole-string false reject the clause
    contract exists to remove -- so the boundary set is not documentation, it is
    load-bearing, and each member needs its own kill-witness. Removing any one
    split must red exactly its own case; a table of six that only four tests can
    detect is four guarantees wearing six names.

    The semicolon row is the CONTROL: it was already supported and already
    passing, so it proves the fixture shape detects a working boundary rather
    than passing for some unrelated reason.
    """

    # Same 141-character fact either side of each separator: an occurrence
    # clause, then a durable production invariant.
    OCCURRENCE = "Session deadbeef concluded the migration from SHA-1 to SHA-256"
    RESIDUE = "Production rejects SHA-1 signatures across API endpoints for signed requests."

    SEPARATORS = {
        "semicolon (control, already supported)": "; ",
        "sentence period": ". ",
        # BARE newline, deliberately with NO period. ".\n" would split on the
        # PERIOD rule, whose trailing \s+ already matches a newline -- so a
        # ".\n" fixture leaves the newline branch entirely unpinned while
        # appearing to cover it. Caught by the kill-witness battery below: the
        # first version of this row stayed green with the newline pattern
        # deleted. That is the same defect as an unpinned platform branch,
        # committed inside the test written to prevent it.
        "bare newline": "\n",
        "em dash": " — ",
        # Closing quotes are absorbed by the sentence-punctuation rule. This row
        # asserts the SPLIT, not merely the keep: without it the residue
        # detector still rescues this string by demotion, so removing quote
        # absorption from the split pattern would change no observable outcome
        # and the claim would sit unpinned behind a mechanism that happens to
        # cover for it. The clause-count assertion below is what isolates it.
        "period then closing quote": '." ',
    }
    # The COLON is deliberately absent from this table as of round 6. It is no
    # longer a split at all; a residue-bearing colon reaches the same outcome by
    # DEMOTION instead, which the next test pins. Leaving it here would claim a
    # mechanism the code does not have.

    def test_a_residue_across_each_boundary_survives_the_real_pipeline(self):
        for name, sep in self.SEPARATORS.items():
            content = f"{self.OCCURRENCE}{sep}{self.RESIDUE}"
            with self.subTest(boundary=name):
                self.assertGreater(
                    len(_split_into_propositions(content)), 1,
                    "the segmenter did not treat this as a proposition boundary")
                self.assertFalse(_is_generic_node(content), "sibling _is_generic_node")
                self.assertFalse(_lacks_specificity(content), "sibling _lacks_specificity")
                self.assertFalse(_is_garbled_content(content), "sibling _is_garbled_content")
                self.assertFalse(_is_metadata_noise(content), "this filter")
                self.assertIsNotNone(
                    _evaluate_create_content(content),
                    "the durable production invariant is deleted at the real seam")

    def test_a_colon_reaches_the_same_outcome_by_demotion_not_by_splitting(self):
        # Round 6 deleted the conditional colon split after a reviewer proved
        # the condition wrong in BOTH directions with one probe each. A colon
        # now survives as a RESIDUE signal: it demotes the clause to unknown,
        # and unknown keeps. Same outcome for the durable case, reached by a
        # mechanism that cannot false-split a plural noun into a finite verb.
        for name, content in {
            "capitalised subject after the colon": (
                "Session deadbeef concluded the migration from SHA-1 to SHA-256: "
                "Production rejects SHA-1 signatures across API endpoints."),
            # This one is the reason the guard had to go: lowercase prose after
            # a colon is ordinary, the old capital guard refused to split it,
            # and the real pipeline deleted a durable production invariant.
            "lowercase prose after the colon": (
                "Session deadbeef concluded the migration from SHA-1 to SHA-256: "
                "production rejects SHA-1 signatures across API endpoints."),
        }.items():
            with self.subTest(case=name):
                self.assertFalse(_is_metadata_noise(content))
                self.assertIsNotNone(_evaluate_create_content(content))

    def test_a_plural_noun_after_a_colon_no_longer_rescues_bookkeeping(self):
        # The other direction the old guard failed. "records" satisfies a
        # finite-verb approximation while being a plural NOUN, so the colon
        # split a reported topic into an asserted-looking fragment and rescued
        # pure session bookkeeping. With no colon split, it stays bookkeeping.
        self.assertTrue(_is_metadata_noise(
            "Session deadbeef focused on the production incident archive: "
            "Production records from the incident"))

    def test_a_colon_introducing_a_label_complement_is_not_a_boundary(self):
        # The other half of the colon rule, and the reason it is conditional.
        # A colon far more often introduces a LABEL'S COMPLEMENT than a new
        # proposition. Splitting these would strand a bare label or an
        # imperative fragment as an unclassifiable clause -- and unknown keeps,
        # so an unconditional colon rule would turn three pinned REJECTS into
        # keeps while fixing one.
        # Asserted as "no proposition BEGINS at the colon", not as a clause
        # count: these strings also contain coordination, which is a separate
        # and legitimate boundary. Counting clauses would test the wrong thing
        # and fail for a reason this case is not about.
        for name, content, first_word_after_colon in [
            ("label on a durable rule",
             "Focus: visible focus rings are required for keyboard accessibility",
             "visible"),
            ("label on a journal row",
             "Focus for session b7d2c904: recap the previous session and write "
             "the journal entry.",
             "recap"),
            ("imperative command echo",
             "Focus: Run this exact shell command and report its exit code: "
             "rm -rf /tmp/scratch-run-00000",
             "Run"),
        ]:
            with self.subTest(case=name):
                starts = [p.split()[0] for p in _split_into_propositions(content)]
                self.assertNotIn(
                    first_word_after_colon, starts,
                    "the colon was treated as a proposition boundary; the label's "
                    "complement was stranded as its own clause")


class TestEverySentencePunctuationAndClosingGlyphSplits(unittest.TestCase):
    """One split-asserting row per claimed glyph.

    EVERY ROW ASSERTS THE SPLIT COUNT, NOT THE VERDICT, and that is the whole
    design of this class. The residue detector demotes on unsplit sentence
    punctuation, so a `!` fixture whose split silently disappeared would STILL
    keep -- via demotion -- and a verdict-shaped assertion would stay green over
    a boundary that no longer exists. Reviewers proved this by deleting `!`,
    `?`, and the curly closing quote and finding everything still passing.

    That is the third time in this PR that a general mechanism has masked the
    specific claims it overlaps. A row that can be satisfied by the fallback is
    not a witness for the thing it names.
    """

    OCC = "Session deadbeef concluded the migration from SHA-1 to SHA-256"
    RES = "Production rejects SHA-1 signatures across API endpoints."

    # The full character class the splitter claims: sentence punctuation, then
    # any run of closing characters, then whitespace.
    TERMINATORS = {"period": ".", "exclamation": "!", "question": "?"}
    CLOSERS = {
        "none": "",
        "ASCII double quote": '"',
        "ASCII single quote": "'",
        "curly double quote": "”",
        "curly single quote": "’",
        "closing paren": ")",
        "closing bracket": "]",
    }

    def test_each_terminator_splits(self):
        for name, mark in self.TERMINATORS.items():
            with self.subTest(terminator=name):
                content = f"{self.OCC}{mark} {self.RES}"
                self.assertEqual(
                    len(_split_into_propositions(content)), 2,
                    "this terminator is not splitting; the residue detector "
                    "would still keep the string, so a verdict assertion here "
                    "would pass over a missing boundary")

    def test_a_RUN_of_stacked_closers_is_absorbed(self):
        # THE QUANTIFIER IS ITSELF A CLAIM. The rule absorbs closers with `*`,
        # meaning arbitrarily many -- but every row above supplies exactly ONE,
        # so `*` could be weakened to `?` with all 59 tests and 55 subtests
        # staying green. Under that mutant an occurrence followed by stacked
        # `.")` and a durable residue stays FUSED, and the residue fallback
        # masks the verdict so nothing reds.
        #
        # Evidence about one is not evidence about many. This is the same
        # unpinned-claim defect found at the glyph level and the branch level,
        # now at the level of a repetition operator.
        for name, closers in {
            "quote then paren": '")',
            "quote then bracket": '"]',
            "curly quote then paren": "”)",
            "three stacked": "\"')",
            # A reviewer's own fixture, and the strongest of the set: a MIXED
            # run exercises the quantifier and the character class at once, so
            # one row covers both claims rather than assuming the other holds.
            "mixed run, quote paren bracket": '")]',
        }.items():
            with self.subTest(closers=name):
                content = f"{self.OCC}.{closers} {self.RES}"
                self.assertEqual(
                    len(_split_into_propositions(content)), 2,
                    "a run of closers was not absorbed; the residue detector "
                    "would still keep this, so only a split assertion can see it")

    def test_each_closing_character_is_absorbed_after_a_terminator(self):
        for name, closer in self.CLOSERS.items():
            with self.subTest(closer=name):
                content = f"{self.OCC}.{closer} {self.RES}"
                self.assertEqual(
                    len(_split_into_propositions(content)), 2,
                    "this closing character is not absorbed by the "
                    "sentence-punctuation rule")


def _count_top_level_alternation_branches(pattern: str) -> int:
    """Count the alternation branches at the depth of the OUTERMOST group.

    Deliberately reads the compiled pattern rather than a hand-maintained list.
    See TestEveryBranchOfTheSplitterHasARow for why that distinction is the
    whole point of this helper.
    """
    depth = 0
    target_depth = None
    branches = 1
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\":
            i += 2
            continue
        if char == "[":                      # character class: | inside is literal
            i += 1
            while i < len(pattern) and pattern[i] != "]":
                i += 2 if pattern[i] == "\\" else 1
        elif char == "(":
            depth += 1
            if target_depth is None:
                target_depth = depth
        elif char == ")":
            depth -= 1
        elif char == "|" and depth == target_depth:
            branches += 1
        i += 1
    return branches


class TestEveryBranchOfTheSplitterHasARow(unittest.TestCase):
    """One split-asserting row per branch of the splitter, with the BRANCH COUNT
    itself asserted against the compiled pattern.

    WHY THE COUNT IS AN ASSERTION AND NOT A COMMENT. The previous round closed
    what looked like the last open space: six literal word alternatives, six
    rows, none unaccounted. That enumeration was complete -- over LITERAL word
    alternatives. The participial branch is a word-FORM branch, `\\w+ing`, and it
    sat outside the category while looking like a member of it. Deleting it left
    the whole spec green.

    So the enumeration was true of the space AS DRAWN, and the space was drawn
    one member short. That is the failure mode of an exhaustiveness claim, and
    it is not the one anyone was guarding against: we checked whether the
    enumeration covered its category and nobody checked whether the CATEGORY
    covered the mechanism. A boundary is chosen by the same understanding that
    missed the gap, which is why a claim of completeness cannot be verified by
    the person drawing the boundary.

    The remedy is to stop drawing it. This class derives its obligation from the
    SPLITTER'S OWN ALTERNATION: count the top-level branches in the compiled
    pattern, and require exactly one row per branch. A seventh branch added
    later fails this spec until someone gives it a row -- the enumeration is now
    closed against the CODE rather than against a human's idea of the category,
    and it stays closed without anyone remembering to re-draw it.

    ------------------------------------------------------------------------
    UNFINISHED: THIS PATTERN IS DONE, THREE OTHERS ARE NOT. READ THIS FIRST.
    ------------------------------------------------------------------------

    THE DIAGNOSIS, which is what makes the remaining work obvious rather than
    open-ended. Witnesses in this file were historically written PER BEHAVIOUR;
    degrees of freedom live PER PATTERN. Those two do not line up. A test set
    that is complete over behaviours can therefore leave an arbitrary number of
    pattern DOFs unpinned, and only a mutation sweep finds them. Five rounds of
    review found that same class at five granularities -- branch, glyph,
    quantifier, word-alternative, and regex-internal grammar -- because each
    round pinned the DOFs someone happened to sweep. It is a units mismatch, not
    a carelessness story, which is why more care never closed it.

    THIS CLASS IS THE WORKING EXAMPLE of the fix, proven in both directions:
    deleting a branch reds only its row, and adding an unwitnessed branch reds
    the count assertion. Apply the same shape to the three patterns below.

      _SESSION_RECORD_TUPLE_RE  -- DOFs are its separator character class, the
          optional-ness of the separator, each timestamp alternative
          (YYYY-MM-DD, HH:MM, the optional :SS), and the repetition allowing
          zero, one or many timestamp fields. A reviewer found five of these
          mutable with the whole spec green. NOTE: whether a BARE ID with zero
          timestamp fields should bypass at all is an OPEN CONTRACT QUESTION,
          not a coverage gap -- the regex accepts it while the source prose says
          "an id and a timestamp". Do not pin it until that is ruled on.
      _BOUNDARY_RESIDUE_RE      -- DOFs are its two alternatives, the closing
          character class, and the quantifier on that class.
      _EPHEMERAL_PATH_RE        -- DOFs are its platform alternatives and the
          leading boundary group. The platform alternatives already have rows in
          TestEveryEphemeralPathBranchIsPinned; what is missing is the COUNT
          ASSERTION tying that row set to the pattern's own structure.

    THE COUNT ASSERTION IS THE LOAD-BEARING PART. Rows alone close the DOFs that
    exist today; the count is what makes a NEW DOF fail the spec until someone
    pins it. Without it this is one refactor away from starting over, and the
    class reopens in whatever pattern nobody swept.
    """

    OCC = "Session deadbeef concluded the migration from SHA-1 to SHA-256"
    RES = "production rejects SHA-1 signatures across API endpoints"

    # One entry per top-level branch of _PROPOSITION_SPLIT_RE, in pattern order.
    BRANCH_ROWS = {
        "semicolon": "; ",
        "sentence punctuation": ". ",
        "newline": "\n",
        "dash family": " — ",
        "relative or causal comma": ", because ",
        # The branch the literal-word enumeration did not cover. NOTE THE
        # FIXTURE CARRIES NO `and`: with one, a later coordination split creates
        # an unknown clause and the string keeps for the WRONG REASON, passing
        # the row while this branch is deleted.
        "participial comma": ", exposing a defect in ",
        "coordination": " and ",
    }

    def test_the_row_set_matches_the_compiled_branch_count(self):
        actual = _count_top_level_alternation_branches(_PROPOSITION_SPLIT_RE.pattern)
        self.assertEqual(
            actual, len(self.BRANCH_ROWS),
            f"the splitter has {actual} top-level branches and this class has "
            f"{len(self.BRANCH_ROWS)} rows. A branch without a row is a claim "
            f"without a witness; add the row rather than adjusting this count.")

    def test_each_branch_separates_an_occurrence_from_its_residue(self):
        for name, joiner in self.BRANCH_ROWS.items():
            with self.subTest(branch=name):
                content = f"{self.OCC}{joiner}{self.RES}"
                self.assertGreater(
                    len(_split_into_propositions(content)), 1,
                    "this branch is not splitting")
                self.assertFalse(_is_metadata_noise(content))


class TestEveryWordBranchSplits(unittest.TestCase):
    """One split-asserting row per word alternative in the splitter.

    THE SPACE IS CLOSED, and that is what makes this class different from the
    ones above. The splitter offers exactly six word alternatives -- `and` and
    `but` for coordination, `which`, `that`, `so` and `because` for relative
    and causal clauses. Two were pinned; a reviewer swept the remaining four and
    found each removable with all 60 tests and 60 subtests green, while each
    removal flipped a durable residue from keep to REJECT.

    Six alternatives, six rows, none unaccounted. Enumerating a small closed
    space is what turns "no more were found" into "none remain" -- the
    difference between a search that stopped and a search that finished.

    Every row asserts SPLIT COUNT, because these branches are exactly where the
    residue detector cannot rescue the outcome: removing one FUSES an occurrence
    with a durable residue and the fused clause has no boundary-shaped material
    left in asserted position, so it rejects silently.
    """

    OCC = "Session deadbeef concluded the migration from SHA-1 to SHA-256"
    RES = "production rejects SHA-1 signatures across API endpoints"

    JOINERS = {
        "and (coordination)": " and ",
        "but (coordination)": " but ",
        "which (relative)": ", which means ",
        "that (relative)": ", that is why ",
        "so (causal)": ", so ",
        "because (causal)": ", because ",
    }

    def test_each_word_alternative_separates_occurrence_from_residue(self):
        for name, joiner in self.JOINERS.items():
            with self.subTest(joiner=name):
                content = f"{self.OCC}{joiner}{self.RES}"
                self.assertGreater(
                    len(_split_into_propositions(content)), 1,
                    "this word alternative is not splitting; the residue "
                    "detector cannot rescue it, so the durable residue is "
                    "deleted with the occurrence and nothing reds")
                self.assertFalse(_is_metadata_noise(content))


class TestExactJournalTuplesBypassDemotion(unittest.TestCase):
    """An anchored whole-string journal tuple is not ambiguous, so the detector
    does not get to be uncertain about it.

    Residue demotion reopened the round-1 core case: the comma tuple rejected
    while the colon tuple kept, same content, because a colon fired the
    detector. A safety net that opens the case it was built over is worse than
    the gap it covered.

    The bypass is licensed by ANCHORING, not by length. `_SESSION_RECORD_TUPLE_RE`
    matches at both ends, so a whole-string match means the clause is entirely
    an id and a timestamp with no unaccounted material for a missed boundary to
    hide in. There is deliberately no short-token exemption: length is not
    evidence that something is metadata.
    """

    def test_a_colon_separated_tuple_rejects_like_its_comma_twin(self):
        for name, content in {
            "colon + date": "Session deadbeef: 2026-08-01",
            "comma + date (the twin)": "Session deadbeef, 2026-08-01",
            "colon + time": "Session a1b2c3d4: 09:03",
        }.items():
            with self.subTest(case=name):
                self.assertTrue(_is_metadata_noise(content))
                self.assertIsNone(_evaluate_create_content(content))

    def test_the_bypass_does_not_leak_to_anything_unanchored(self):
        # The control that keeps the bypass narrow. Real content after a colon
        # is NOT a whole-string tuple, so the tuple never matches, demotion
        # still applies, and ambiguity still keeps. Without this, a bypass that
        # widened to "starts with a session id" would pass the tests above while
        # deleting durable facts.
        for name, content in {
            "durable clause after the colon": (
                "Session deadbeef: the gateway now rejects SHA-1 across every "
                "endpoint"),
            "residue-bearing occurrence": (
                "Session deadbeef concluded the migration: production rejects "
                "SHA-1 signatures"),
        }.items():
            with self.subTest(case=name):
                self.assertFalse(_is_metadata_noise(content))


class TestResidueDemotionIsOneDirectionalAndDownstream(unittest.TestCase):
    """The residue detector: when boundary-shaped material remains inside a
    clause about to be called B, the clause is demoted to unknown, and unknown
    keeps.

    It exists because the set of punctuation forms that can separate two
    propositions has no bound a lexical layer can enumerate. Reviewers produced
    closing quotes, U+2015, a closed em dash, non-Latin capitals and lowercase
    prose after a colon, and there is no reason to think that list is finished.
    Chasing forms one at a time is a race with no finish line where every miss
    deletes a durable fact silently; this makes segmentation failure fail-open
    in the direction the contract already chose.
    """

    def test_a_clean_class_B_clause_with_no_residue_still_rejects(self):
        # THE CONTROL THAT STOPS THE DETECTOR BEING A BLANKET DEMOTION. Without
        # it, a mechanism that demoted EVERY clause would satisfy every keep
        # case in this file and look like a triumph.
        for name, content in {
            "bare tuple": "Session a1b2c3d4, 2020-01-02",
            "occurrence predicate": (
                "Session beef1234 occurred on 2020-01-02 with focus on clearing "
                "a command"),
            "dated command echo": "Execute `/clear` command on 2020-01-02",
        }.items():
            with self.subTest(case=name):
                self.assertFalse(
                    _has_unsplit_boundary_residue(content),
                    "clean class-B content must carry no residue signal")
                self.assertTrue(_is_metadata_noise(content))

    def test_residue_only_ever_turns_a_reject_into_a_keep(self):
        # One-directional by construction: the detector is consulted ONLY on the
        # class-B branch. A clause that was already going to keep cannot be
        # pushed into a reject by it, whatever residue it carries.
        already_keeps = "The indexer consumes the tokenizer package as of 2020-01-02"
        self.assertFalse(_is_metadata_noise(already_keeps))
        self.assertFalse(_is_metadata_noise(already_keeps + ": And more prose."))

    def test_the_detector_runs_downstream_of_segmentation_not_as_a_pre_scan(self):
        # A pre-scan over the RAW string would see boundaries the segmenter
        # would have split cleanly and demote content that was never at risk.
        # This string has a semicolon the segmenter handles perfectly; both
        # resulting clauses are clean, so nothing is demoted and the verdict
        # comes from the clause classes rather than from raw punctuation.
        # Both clauses carry a predicate on purpose: a bare "Session <id>, <date>"
        # tuple is verbless by construction and would be folded back into its
        # neighbour, which would test the fold rather than the pre-scan question.
        content = ("Session beef1234 occurred on 2020-01-02; "
                   "Session cafe5678 occurred on 2020-01-03")
        self.assertEqual(len(_split_into_propositions(content)), 2)
        for clause in _split_into_propositions(content):
            self.assertFalse(_has_unsplit_boundary_residue(clause))
        self.assertTrue(
            _is_metadata_noise(content),
            "two clean class-B clauses must still reject; a pre-scan would have "
            "seen the semicolon in the raw string and wrongly demoted this")


class TestNonAsciiClausesAreNotFoldedBackIntoOccurrences(unittest.TestCase):
    """The verbless-fold heuristic reads ASCII suffixes, so it must not judge
    text it cannot read.

    A reviewer's Greek probe kept fusing back into its occurrence clause and the
    SPLITTER was innocent -- it split correctly and the FOLD undid it. Any
    non-Latin clause looks verbless to a -ed/-ing/-s test, gets folded backwards,
    and its durable residue is deleted with the occurrence. The ASCII assumption
    sat one layer below the [A-Z] guards that were being examined.
    """

    def test_a_non_latin_residue_clause_stands_on_its_own(self):
        content = ("Session deadbeef concluded the migration from SHA-1 to "
                   "SHA-256. Παραγωγή απορρίπτει SHA-1.")
        self.assertEqual(len(_split_into_propositions(content)), 2,
                         "the non-Latin clause was folded back into the occurrence")
        self.assertFalse(_is_metadata_noise(content))
        self.assertIsNotNone(_evaluate_create_content(content))

    def test_an_ascii_verbless_fragment_is_still_folded(self):
        # The control. Folding is correct for ASCII noun coordination -- without
        # it, "the first and last recorded events" strands a verbless fragment
        # that keeps a journal row alive.
        self.assertTrue(_is_metadata_noise(
            "The duration between the first and last recorded events of session "
            "e91b6d3a spanned just under two hours on the evening of 2024-03-22."))


class TestEveryDashGlyphIsPinnedSeparately(unittest.TestCase):
    """One witness per dash glyph, and per spacing form.

    Reviewer finding: the en dash and the double hyphen were each removable from
    the pattern with all 46 tests and 26 subtests staying green. Both were
    pristine survivors behind a single em-dash witness -- the same
    one-witness-for-many-claims collapse found in the platform branches, and it
    was read as a nit by two people before being proved a blocker.
    """

    OCC = "Session deadbeef concluded the migration from SHA-1 to SHA-256"
    RES = "Production rejects SHA-1 signatures across API endpoints."

    GLYPHS = {
        "em dash spaced": " — ",
        "em dash CLOSED": "—",          # the common form; it false-rejected before
        "en dash spaced": " – ",
        "U+2015 horizontal bar": " ― ",
        "U+2012 figure dash": " ‒ ",
        "double hyphen spaced": " -- ",
    }

    def test_each_dash_form_separates_an_occurrence_from_its_residue(self):
        for name, glyph in self.GLYPHS.items():
            with self.subTest(dash=name):
                content = f"{self.OCC}{glyph}{self.RES}"
                self.assertGreater(len(_split_into_propositions(content)), 1)
                self.assertFalse(_is_metadata_noise(content))
                self.assertIsNotNone(_evaluate_create_content(content))


class TestCoordinatorsSplitRegardlessOfCase(unittest.TestCase):
    """Locally scoped case-insensitivity on the word branches.

    A pattern-wide (?i) was dropped so `[A-Z]` guards would work, and it
    silently made the PRE-EXISTING coordinator and relative branches
    case-sensitive too -- so generated prose that capitalised "AND" or "Which"
    stopped splitting and the durable residue behind it was deleted. Disclosed
    as benign; proved concrete by both reviewers.
    """

    def test_uppercase_and_capitalised_coordinators_still_split(self):
        for name, content in {
            "uppercase AND": (
                "Session deadbeef occurred on 2026-08-01 AND Production rejects "
                "SHA-1 signatures across API endpoints."),
            "capitalised Which": (
                "Session deadbeef concluded the migration, Which now rejects "
                "SHA-1 signatures."),
            "lowercase control": (
                "Session deadbeef occurred on 2026-08-01 and production rejects "
                "SHA-1 signatures across API endpoints."),
        }.items():
            with self.subTest(case=name):
                self.assertFalse(_is_metadata_noise(content))
                self.assertIsNotNone(_evaluate_create_content(content))


class TestEveryEphemeralPathBranchIsPinned(unittest.TestCase):
    """One witness per platform branch of the cross-platform claim.

    Reviewer finding: the macOS /var/folders branch could be DELETED outright
    and every test stayed green. A claim covering six platform forms that only
    two tests can detect is two guarantees wearing six names, and the four
    undetected branches can rot silently -- which is exactly the failure mode
    that made the original POSIX-only gap invisible.

    The fixture text is otherwise identical across rows so the PATH is the only
    variable, and the same-shaped control below shows a non-ephemeral path is
    not caught by accident.
    """

    SUFFIX = " directory was wiped with rm -rf after the trial run"

    PATHS = {
        "POSIX /tmp": "The /tmp/scratch-run-00000",
        "macOS /var/folders": "The /var/folders/aa/bbbbb/T/scratch-run",
        "$TMPDIR": "The $TMPDIR/scratch-run-00000",
        "%TEMP%": "The %TEMP%\\scratch-run-00000",
        "%TMP%": "The %TMP%\\scratch-run-00000",
        "Windows AppData Temp": r"The C:\Users\runner\AppData\Local\Temp\scratch-run",
    }

    def test_each_platform_branch_rejects_its_own_execution_event(self):
        for name, head in self.PATHS.items():
            with self.subTest(platform=name):
                content = head + self.SUFFIX
                self.assertTrue(_is_metadata_noise(content), "predicate")
                self.assertIsNone(_evaluate_create_content(content), "real create path")

    def test_a_stable_path_with_the_same_sentence_shape_is_not_caught(self):
        # The control. Without it, a mutation making the path pattern match
        # EVERYTHING would satisfy all six rows above.
        self.assertFalse(_is_metadata_noise(
            "The /var/lib/postgres/data" + self.SUFFIX))


class TestMentionedVocabularyCannotRescueBookkeeping(unittest.TestCase):
    """Vocabulary inside what a session merely focused ON is MENTIONED, not
    asserted, and licenses nothing about the matrix claim.

    Both strings below are pure journal rows. Each was rescued by a keep-word
    sitting inside the reported topic -- "policy" in one, "should" in the other.
    Neither is anyone asserting a policy or an obligation.

    Contrast the documented false keep in the predicate docstring: "Session
    9ab4f012 concluded at 17:42; any remaining follow-up items should be picked
    up" asserts its modal in a clause of its own and is therefore kept. The
    difference is POSITION, which is the whole distinction this class pins.
    """

    def test_keep_vocabulary_in_a_reported_topic_does_not_rescue(self):
        for name, content in {
            "rule noun in the topic": "Session a1b2c3d4 focused on the policy review.",
            "modal in the topic": "Session a1b2c3d4 focused on what should happen next.",
        }.items():
            with self.subTest(case=name):
                self.assertTrue(_is_metadata_noise(content))
                self.assertIsNone(_evaluate_create_content(content))

    def test_the_same_words_asserted_outside_a_topic_do_rescue(self):
        # The control that makes the test above mean POSITION rather than
        # vocabulary. Same words, asserted rather than reported: kept.
        self.assertFalse(_is_metadata_noise(
            "Session a1b2c3d4 is governed by the retention policy."))
        self.assertFalse(_is_metadata_noise(
            "Session a1b2c3d4 should be replayed before every release."))


class TestReportedContentAloneCannotCondemn(unittest.TestCase):
    """Class C is inert. Rejection needs a POSITIVELY identified occurrence
    clause; a reported topic on its own is not evidence that anything is
    bookkeeping."""

    def test_a_topic_label_on_a_durable_rule_is_kept(self):
        self.assertFalse(_is_metadata_noise(
            "Focus: visible focus rings are required for keyboard accessibility"))

    def test_an_occurrence_with_no_durable_clause_still_rejects(self):
        # The other half: C is inert, but B is not. This one has a real
        # occurrence clause, so it dies.
        self.assertTrue(_is_metadata_noise(
            "Focus for session b7d2c904: recap the previous session and write "
            "the journal entry."))


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
