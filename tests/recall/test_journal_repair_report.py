"""A repair report must name the store it examined — empty results most of all.

The defect this pins was found in operation, not in review. Running the repair
from a desk produced:

    Nothing to repair.

That report was clean, confident, and false. The CLI resolved its data root
from the working directory and landed on a store holding 15 entries and zero
contamination, while the journal that actually receives that desk's writes —
265 entries, 38 contaminated — sits under a different root the command never
looked at. Every agent would have reported a clean desk.

An empty result is a fact about the query until the query says what it asked.
"Nothing to repair" and "I examined the wrong store" are the same sentence
unless the report names the store, which makes this the same class as a
control that cannot fail: the output is indistinguishable between success and
having done nothing at all.

So the contract is a reporting contract, not a repair contract:

    every output line carries the examined store path, ALWAYS, and the
    empty case is the one that needs it most.
"""

import json
import tempfile
import unittest
from pathlib import Path

from synapt.recall.journal import (
    JournalEntry,
    format_repair_report,
    repair_journal,
    sweep_stores,
)

COLLAPSED = (
    "A real decision that was written down."
    '</decisions>\n<parameter name="next_steps">'
    "A real next step that got swallowed."
    "</next_steps>\n</invoke>"
)
CLEAN = "An ordinary decision with no markup in it."


def _store(root: Path, worktree: str, entries: list[JournalEntry]) -> Path:
    path = root / "worktrees" / worktree / "journal.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")
    return path


class TestReportNamesItsStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # The shape that actually happened: the root a desk resolves to is
        # clean, and the root its writes land in is not.
        self.resolved_root = self.tmp / "resolved"      # what cwd resolves to
        self.real_root = self.tmp / "real"              # where writes land
        self.empty_store = _store(
            self.resolved_root, "desk",
            [JournalEntry(timestamp="2026-08-07T09:00:00+00:00", session_id="c1",
                          decisions=[CLEAN])],
        )
        self.real_store = _store(
            self.real_root, "desk",
            [JournalEntry(timestamp="2026-08-01T09:00:00+00:00", session_id="d1",
                          decisions=[COLLAPSED]),
             JournalEntry(timestamp="2026-08-02T09:00:00+00:00", session_id="d2",
                          decisions=[CLEAN])],
        )

    # --- THE FALSE-CLEAN WITNESS -------------------------------------------
    def test_clean_report_names_the_store_it_examined(self):
        report = repair_journal(self.empty_store, dry_run=True)
        text = format_repair_report(report)
        self.assertEqual(report["contaminated_entries"], 0)
        self.assertIn(
            str(self.empty_store), text,
            "a clean report that does not name its store is indistinguishable "
            "from having examined the wrong one",
        )

    def test_the_bare_clean_sentence_is_never_emitted_alone(self):
        # The old behaviour, verbatim, was a bare "Nothing to repair."
        text = format_repair_report(repair_journal(self.empty_store, dry_run=True))
        stripped = text.replace(str(self.empty_store), "")
        self.assertNotEqual(stripped.strip(), text.strip(),
                            "the path must actually appear, not be implied")

    def test_the_same_desk_pointed_at_its_real_store_finds_the_contamination(self):
        # Same command, explicit path, different answer. This is the pair that
        # proves the clean result above was about the QUERY, not the data.
        report = repair_journal(self.real_store, dry_run=True)
        self.assertEqual(report["contaminated_entries"], 1)
        self.assertIn(str(self.real_store), format_repair_report(report))

    # --- non-empty case still names it -------------------------------------
    def test_report_names_the_store_when_it_finds_something(self):
        text = format_repair_report(repair_journal(self.real_store, dry_run=True))
        self.assertIn(str(self.real_store), text)

    def test_report_distinguishes_dry_run_from_applied(self):
        dry = format_repair_report(repair_journal(self.real_store, dry_run=True))
        applied = format_repair_report(repair_journal(self.real_store))
        self.assertNotEqual(dry, applied)
        self.assertIn(str(self.real_store), dry)
        self.assertIn(str(self.real_store), applied)

    # --- sweep names the root AND every store ------------------------------
    def test_sweep_names_the_root_it_swept_under(self):
        # NOT asserted by searching the text for the root string: the root is
        # a prefix of every store path, so that assertion passes even if the
        # root is never reported. Assert the report carries it as its own
        # field, which is the thing a caller can actually act on.
        reports = sweep_stores(self.real_root, dry_run=True)
        self.assertTrue(reports)
        for r in reports:
            self.assertEqual(r["root"], str(self.real_root))

    def test_sweep_names_every_store_it_touched(self):
        _store(self.real_root, "second",
               [JournalEntry(timestamp="2026-08-03T09:00:00+00:00", session_id="e1",
                             decisions=[COLLAPSED])])
        reports = sweep_stores(self.real_root, dry_run=True)
        text = "\n".join(format_repair_report(r) for r in reports)
        self.assertEqual(len(reports), 2)
        for name in ("desk", "second"):
            self.assertIn(
                str(self.real_root / "worktrees" / name / "journal.jsonl"), text
            )

    def test_sweep_of_a_root_with_no_stores_says_so_and_names_the_root(self):
        # The emptiest possible result is the one most likely to be misread:
        # "swept, found nothing" and "swept the wrong root" read identically.
        empty_root = self.tmp / "nothing-here"
        (empty_root / "worktrees").mkdir(parents=True)
        reports = sweep_stores(empty_root, dry_run=True)
        text = "\n".join(format_repair_report(r) for r in reports)
        self.assertIn(str(empty_root), text)      # the text alone, not text+root
        self.assertEqual([r for r in reports if r["total_entries"]], [])

    # --- Sentinel's addendum: resolved path AND line count, every line -----
    def test_every_report_carries_the_raw_line_count(self):
        # Line count is not entry count. Unparseable lines are skipped
        # silently, so a store whose line count exceeds its entry count has
        # lost records that no other number would reveal.
        for store in (self.empty_store, self.real_store):
            report = repair_journal(store, dry_run=True)
            expected = sum(1 for line in open(store, encoding="utf-8") if line.strip())
            self.assertEqual(report["line_count"], expected)
            self.assertIn(str(expected), format_repair_report(report))

    def test_line_count_exposes_records_the_parser_dropped(self):
        broken = self.tmp / "broken" / "worktrees" / "d" / "journal.jsonl"
        broken.parent.mkdir(parents=True)
        good = JournalEntry(timestamp="2026-08-01T09:00:00+00:00", session_id="g",
                            decisions=[CLEAN])
        with open(broken, "w", encoding="utf-8") as f:
            f.write(json.dumps(good.to_dict()) + "\n")
            f.write("{ this is not json\n")
        report = repair_journal(broken, dry_run=True)
        self.assertEqual(report["line_count"], 2)
        self.assertEqual(report["total_entries"], 1)   # one line was dropped
        text = format_repair_report(report)
        self.assertIn("2", text)
        self.assertIn("1", text)

    # --- control: the assertion is capable of failing ----------------------
    def test_a_path_that_was_not_examined_is_absent_from_the_report(self):
        text = format_repair_report(repair_journal(self.empty_store, dry_run=True))
        self.assertNotIn(str(self.real_store), text)


if __name__ == "__main__":
    unittest.main()
