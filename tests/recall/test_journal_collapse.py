"""Tests for journal field-collapse detection, serving-filter, and repair.

The defect: when a tool-call parameter's closing tag is dropped, the emitter
consumes every following parameter -- tags and all -- into the unclosed value.
The write succeeds; the trailing fields simply arrive empty. So a journal entry
looks partially filled rather than malformed, and nothing anywhere goes red.

The fixtures below reproduce the two shapes found in real stored data, not
invented ones:

    A:  <real text></decisions>\\n<parameter name="next_steps">swallowed...
    B:  <real text></decisions>\\n<next_steps>swallowed...

Three independent layers are under test, and the independence is the point --
each holds when the other two are absent:

    guard    -- refuses a NEW contaminated write at the single append point
    serving  -- never SERVES a contaminated step, whatever is already stored
    repair   -- recovers the swallowed content into a corrective entry

The serving layer is what stops the bleeding on data already on disk, because
a contaminated next_step can never be matched by a done item, so carry-forward
re-injects it every session forever. Measured in a real store: one malformed
step present in 13 distinct entries, 38 of 265 entries contaminated.
"""

import json
import tempfile
import unittest
from pathlib import Path

from synapt.recall.journal import (
    COLLAPSE_SIGNATURES,
    JournalEntry,
    JournalFieldCollapse,
    append_entry,
    is_collapsed,
    merge_carried_forward_next_steps,
    pending_next_steps,
    read_entries,
    recover_collapsed,
    repair_journal,
)

# Shape A -- the majority variant in real data.
COLLAPSED_A = (
    "Depth is only depth if the layers are INDEPENDENT."
    '</decisions>\n<parameter name="next_steps">'
    "Await the post-compact spark: cleanup, an unblocked lane, or rest."
    "</next_steps>\n</invoke>"
)
# Shape B -- same defect, bare tag instead of the parameter form.
COLLAPSED_B = (
    "Measure on main, do not claim."
    "</decisions>\n<next_steps>"
    "Re-verify the remote tree once the push lands."
    "</next_steps>\n</invoke>"
)
# Total collapse: done swallows BOTH decisions and next_steps.
COLLAPSED_TRIPLE = (
    "Merged the re-cut and closed the tail."
    '</done>\n<parameter name="decisions">'
    "Bind the gate on the tree, never the commit."
    '</decisions>\n<parameter name="next_steps">'
    "Claim the guard in the morning."
    "</next_steps>\n</invoke>"
)
CLEAN = "A perfectly ordinary decision with no markup in it at all."


def _write(path: Path, entries: list[JournalEntry]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")


class TestCollapseDetection(unittest.TestCase):
    def test_all_six_signatures_are_declared(self):
        for sig in (
            "</focus>", "</done>", "</decisions>",
            "</next_steps>", "</invoke>", "<parameter name=",
        ):
            self.assertIn(sig, COLLAPSE_SIGNATURES)

    def test_detects_both_real_world_shapes(self):
        self.assertTrue(is_collapsed(COLLAPSED_A))
        self.assertTrue(is_collapsed(COLLAPSED_B))
        self.assertTrue(is_collapsed(COLLAPSED_TRIPLE))

    def test_clean_prose_is_not_flagged(self):
        # The control. A detector that flags everything is not a detector.
        self.assertFalse(is_collapsed(CLEAN))
        self.assertFalse(is_collapsed(""))
        self.assertFalse(is_collapsed("a < b and c > d, 3<4"))
        self.assertFalse(is_collapsed("use <angle brackets> in prose freely"))


class TestRecovery(unittest.TestCase):
    def test_recovers_head_and_swallowed_field_shape_a(self):
        head, recovered = recover_collapsed(COLLAPSED_A)
        self.assertEqual(head, "Depth is only depth if the layers are INDEPENDENT.")
        self.assertEqual(
            recovered["next_steps"],
            ["Await the post-compact spark: cleanup, an unblocked lane, or rest."],
        )

    def test_recovers_head_and_swallowed_field_shape_b(self):
        head, recovered = recover_collapsed(COLLAPSED_B)
        self.assertEqual(head, "Measure on main, do not claim.")
        self.assertEqual(
            recovered["next_steps"],
            ["Re-verify the remote tree once the push lands."],
        )

    def test_recovers_two_swallowed_fields_from_a_triple_collapse(self):
        head, recovered = recover_collapsed(COLLAPSED_TRIPLE)
        self.assertEqual(head, "Merged the re-cut and closed the tail.")
        self.assertEqual(
            recovered["decisions"], ["Bind the gate on the tree, never the commit."]
        )
        self.assertEqual(recovered["next_steps"], ["Claim the guard in the morning."])

    def test_recovery_never_loses_content(self):
        # Every word of the original must survive somewhere. Truncating at the
        # tag would pass a "no tags remain" check while discarding real work.
        head, recovered = recover_collapsed(COLLAPSED_TRIPLE)
        rebuilt = head + " " + " ".join(v for vs in recovered.values() for v in vs)
        for phrase in (
            "Merged the re-cut", "Bind the gate on the tree", "Claim the guard"
        ):
            self.assertIn(phrase, rebuilt)

    def test_clean_text_recovers_to_itself(self):
        head, recovered = recover_collapsed(CLEAN)
        self.assertEqual(head, CLEAN)
        self.assertEqual(recovered, {})


class TestWriteGuard(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "journal.jsonl"

    def test_append_refuses_a_contaminated_entry(self):
        entry = JournalEntry(timestamp="2026-08-07T10:00:00+00:00",
                             session_id="s1", decisions=[COLLAPSED_A])
        with self.assertRaises(JournalFieldCollapse):
            append_entry(entry, self.path)

    def test_refusal_names_the_field_and_the_signature(self):
        # An error that does not say WHERE sends the caller hunting.
        entry = JournalEntry(timestamp="2026-08-07T10:00:00+00:00",
                             session_id="s1", done=[COLLAPSED_TRIPLE])
        with self.assertRaises(JournalFieldCollapse) as ctx:
            append_entry(entry, self.path)
        self.assertIn("done", str(ctx.exception))
        self.assertIn("</done>", str(ctx.exception))

    def test_refused_write_leaves_no_partial_line(self):
        entry = JournalEntry(timestamp="2026-08-07T10:00:00+00:00",
                             session_id="s1", decisions=[COLLAPSED_A])
        with self.assertRaises(JournalFieldCollapse):
            append_entry(entry, self.path)
        self.assertFalse(self.path.exists() and self.path.read_text().strip())

    def test_guard_checks_every_text_field_including_focus(self):
        for field, value in (
            ("focus", COLLAPSED_A),
            ("done", [COLLAPSED_A]),
            ("decisions", [COLLAPSED_A]),
            ("next_steps", [COLLAPSED_A]),
        ):
            entry = JournalEntry(timestamp="2026-08-07T10:00:00+00:00", session_id="s")
            setattr(entry, field, value)
            with self.assertRaises(JournalFieldCollapse, msg=f"{field} unguarded"):
                append_entry(entry, self.path)

    def test_clean_entry_still_appends(self):
        # Control: the guard must not block ordinary writes.
        entry = JournalEntry(timestamp="2026-08-07T10:00:00+00:00",
                             session_id="s1", decisions=[CLEAN], next_steps=["ship it"])
        append_entry(entry, self.path)
        self.assertEqual(len(read_entries(self.path, n=10)), 1)

    def test_repair_path_may_write_contaminated_text_deliberately(self):
        # The corrective entry has to be able to reference the original text.
        # Without this escape hatch the repair tool trips its own guard.
        entry = JournalEntry(timestamp="2026-08-07T10:00:00+00:00",
                             session_id="s1", done=[COLLAPSED_A])
        append_entry(entry, self.path, allow_collapsed=True)
        self.assertEqual(len(read_entries(self.path, n=10)), 1)


class TestServingFilter(unittest.TestCase):
    """Layer 2: whatever is already on disk, a contaminated step is never served."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "journal.jsonl"

    def test_pending_does_not_serve_a_contaminated_step(self):
        _write(self.path, [
            JournalEntry(timestamp="2026-08-01T10:00:00+00:00", session_id="s1",
                         next_steps=[COLLAPSED_A, "a genuinely clean step"]),
        ])
        pending = pending_next_steps(self.path)
        self.assertNotIn(COLLAPSED_A, pending)
        for sig in COLLAPSE_SIGNATURES:
            self.assertFalse(any(sig in p for p in pending))
        # Control: the clean step from the SAME entry is still served, so the
        # filter is removing contamination and not just emptying the list.
        self.assertIn("a genuinely clean step", pending)

    def test_carry_forward_does_not_propagate_a_contaminated_step(self):
        # This is the replication vector: a malformed step can never appear in
        # a done list, so it is unresolvable and rides forward every session.
        previous = JournalEntry(timestamp="2026-08-01T10:00:00+00:00", session_id="s1",
                                next_steps=[COLLAPSED_A, "carry me forward"])
        merged = merge_carried_forward_next_steps(["today's step"], [], previous)
        self.assertNotIn(COLLAPSED_A, merged)
        self.assertTrue(any(s.startswith("carry me forward") for s in merged))   # control (stamped, recall#984)
        self.assertIn("today's step", merged)

    def test_thirteen_entry_propagation_serves_zero(self):
        # Regression pinned to measured reality: the same malformed step was
        # found in 13 distinct entries of one real store.
        _write(self.path, [
            JournalEntry(timestamp=f"2026-07-{d:02d}T10:00:00+00:00",
                         session_id=f"s{d}", next_steps=[COLLAPSED_A])
            for d in range(1, 14)
        ])
        self.assertEqual(len(read_entries(self.path, n=50)), 13)  # control: all read
        self.assertEqual(pending_next_steps(self.path), [])


class TestRepair(unittest.TestCase):
    """Layer 3: recover the swallowed content, append-only."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "journal.jsonl"
        _write(self.path, [
            # Contamination in BOTH a decisions value and a next_steps value.
            # The next_steps one is what the carry-forward loop replicates, so
            # without it the loop-break witness would have nothing to break.
            JournalEntry(timestamp="2026-08-01T10:00:00+00:00", session_id="s1",
                         decisions=[COLLAPSED_A], next_steps=[COLLAPSED_B]),
            JournalEntry(timestamp="2026-08-02T10:00:00+00:00", session_id="s2",
                         decisions=[CLEAN]),
        ])
        self.original_lines = self.path.read_text(encoding="utf-8").splitlines()

    def test_repair_is_append_only(self):
        repair_journal(self.path)
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[:len(self.original_lines)], self.original_lines)
        self.assertGreater(len(lines), len(self.original_lines))

    def test_dry_run_changes_nothing_but_still_reports(self):
        report = repair_journal(self.path, dry_run=True)
        self.assertEqual(
            self.path.read_text(encoding="utf-8").splitlines(), self.original_lines
        )
        self.assertEqual(report["contaminated_entries"], 1)

    def test_repair_reports_what_it_found(self):
        report = repair_journal(self.path)
        self.assertEqual(report["contaminated_entries"], 1)
        self.assertEqual(report["total_entries"], 2)

    def test_recovered_content_is_served_after_repair(self):
        # OUTCOME assertion on a served surface -- not "the row was rewritten".
        repair_journal(self.path)
        pending = pending_next_steps(self.path)
        self.assertTrue(
            any("Await the post-compact spark" in p for p in pending),
            f"swallowed next_step was not recovered into the served surface: {pending}",
        )

    def test_no_signature_survives_on_any_served_surface(self):
        from synapt.recall.journal import format_for_session_start, read_latest
        repair_journal(self.path)
        served = "\n".join(
            format_for_session_start(e) for e in read_entries(self.path, n=50)
        )
        served += "\n" + "\n".join(pending_next_steps(self.path))
        served += "\n" + format_for_session_start(read_latest(self.path))
        for sig in COLLAPSE_SIGNATURES:
            self.assertNotIn(sig, served)
        # Control: the served text is not empty, so the assertion above had
        # something to be true about.
        self.assertIn("Await the post-compact spark", served)

    def test_repair_is_idempotent(self):
        repair_journal(self.path)
        after_first = self.path.read_text(encoding="utf-8")
        second = repair_journal(self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8"), after_first)
        self.assertEqual(second["repaired_entries"], 0)

    def test_loop_break_survives_a_simulated_next_session(self):
        # THE witness. Marking the step resolved is a mechanism; what has to be
        # true is that the malformed step never rides forward again. So simulate
        # the next session end-to-end -- read the previous entry, run the real
        # carry-forward merge, write the new entry -- and assert on what the
        # served surfaces produce, not on the marking having happened.
        from synapt.recall.journal import read_previous_meaningful
        repair_journal(self.path)

        previous = read_previous_meaningful(current_session_id="s_next", path=self.path)
        carried = merge_carried_forward_next_steps(["tomorrow's work"], [], previous)
        self.assertNotIn(COLLAPSED_B, carried)   # the step that WAS replicating
        self.assertNotIn(COLLAPSED_A, carried)
        for sig in COLLAPSE_SIGNATURES:
            self.assertFalse(
                any(sig in step for step in carried),
                f"{sig!r} rode forward into the next session's entry",
            )

        # The next session's entry is written through the real guarded path.
        append_entry(
            JournalEntry(timestamp="2026-08-09T10:00:00+00:00",
                         session_id="s_next", next_steps=carried),
            self.path,
        )
        pending = pending_next_steps(self.path)
        for sig in COLLAPSE_SIGNATURES:
            self.assertFalse(any(sig in p for p in pending))
        self.assertIn("tomorrow's work", pending)   # control: pending is live

    def test_repaired_text_itself_passes_the_forward_guard(self):
        # A repair pass must not inject what the guard would refuse. If the
        # corrective entry's own next_steps could not be written through the
        # ordinary path, the repair is laundering contamination.
        repair_journal(self.path)
        corrective = read_entries(self.path, n=50)[0]
        replay = Path(self.tmpdir) / "replay.jsonl"
        append_entry(   # no allow_collapsed hatch -- must pass on its own merits
            JournalEntry(timestamp=corrective.timestamp, session_id="replay",
                         next_steps=corrective.next_steps,
                         decisions=corrective.decisions),
            replay,
        )
        self.assertEqual(len(read_entries(replay, n=10)), 1)

    def test_clean_store_is_left_alone(self):
        clean_path = Path(self.tmpdir) / "clean.jsonl"
        _write(clean_path, [
            JournalEntry(timestamp="2026-08-01T10:00:00+00:00", session_id="s1",
                         decisions=[CLEAN], next_steps=["ship it"]),
        ])
        before = clean_path.read_text(encoding="utf-8")
        report = repair_journal(clean_path)
        self.assertEqual(clean_path.read_text(encoding="utf-8"), before)
        self.assertEqual(report["contaminated_entries"], 0)


if __name__ == "__main__":
    unittest.main()
