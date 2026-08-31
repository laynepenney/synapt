"""Run one query-freshness mutation without editing the measured worktree."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Mutation:
    path: str
    witness: str
    replacements: tuple[tuple[str, str], ...]


MUTATIONS = {
    "caller-root": Mutation(
        "query_freshness.py",
        "test_refresh_uses_the_explicit_caller_root_without_a_store_fallback",
        (
            (
                "sources = caller_transcripts(caller_root or Path.cwd())",
                "sources = caller_transcripts(Path.cwd())",
            ),
        ),
    ),
    "age-trigger": Mutation(
        "query_freshness.py",
        "test_age_trigger_measures_ahead_of_index_not_age_of_live_source",
        (("return max(0.0, (latest - indexed).total_seconds())", "return 0.0"),),
    ),
    "mixed-offset": Mutation(
        "query_freshness.py",
        "test_projected_timestamp_orders_mixed_offsets_by_instant",
        (("key=lambda item: item[1]", "key=lambda item: item[0]"),),
    ),
    "threshold": Mutation(
        "query_freshness.py",
        "test_below_threshold_gap_is_labelled_without_a_write",
        (
            (
                "        if (\n            gap < policy.byte_trigger",
                "        if False and (\n            gap < policy.byte_trigger",
            ),
        ),
    ),
    "byte-trigger": Mutation(
        "query_freshness.py",
        "test_byte_trigger_refreshes_even_when_temporal_gap_is_recent",
        (("gap < policy.byte_trigger", "True"),),
    ),
    "open-turn": Mutation(
        "query_freshness.py",
        "test_growing_open_turn_replaces_the_earlier_overlay_row",
        (("start=rewind,", "start=observed,"),),
    ),
    "complete-record": Mutation(
        "query_freshness.py",
        "test_incomplete_final_record_does_not_advance_the_cursor",
        (
            (
                "end = _complete_end(\n"
                "                source.path,\n"
                "                observed,\n"
                "                observed + allowance,\n"
                "                hard_end,\n"
                "            )",
                "end = observed + allowance",
            ),
        ),
    ),
    "record-boundary": Mutation(
        "query_freshness.py",
        "test_complete_record_larger_than_step_advances_and_becomes_current",
        (("hard_end,\n            )", "observed + allowance,\n            )"),),
    ),
    "malformed-record": Mutation(
        "query_freshness.py",
        "test_malformed_complete_record_does_not_advance_the_cursor",
        (("json.loads(raw_line)", "None"),),
    ),
    "lock-wait": Mutation(
        "query_freshness.py",
        "test_held_build_lock_returns_busy_without_parsing",
        (
            (
                "_acquire_build_lock(index_dir.parent, timeout=0)",
                "_acquire_build_lock(index_dir.parent, timeout=60)",
            ),
        ),
    ),
    "byte-cap": Mutation(
        "query_freshness.py",
        "test_byte_cap_is_truthful_and_retryable",
        (
            (
                "state=(\n                QueryFreshnessState.REFRESHED\n                if complete\n                else QueryFreshnessState.PARTIAL\n            ),",
                "state=QueryFreshnessState.REFRESHED,",
            ),
        ),
    ),
    "wall-cap": Mutation(
        "query_freshness.py",
        "test_wall_cap_prevents_starting_another_atomic_step",
        (
            (
                "if time.monotonic() - started >= policy.wall_seconds:",
                "if False and time.monotonic() - started >= policy.wall_seconds:",
            ),
        ),
    ),
    "transaction": Mutation(
        "storage.py",
        "test_cursor_and_overlay_rollback_together_on_commit_failure",
        (
            (
                '            self._conn.execute(\n                "INSERT OR REPLACE INTO query_tail_cursors "',
                '            self._conn.commit()\n            self._conn.execute(\n                "INSERT OR REPLACE INTO query_tail_cursors "',
            ),
        ),
    ),
    "sharded-overlay": Mutation(
        "sharded_db.py",
        "test_sharded_layout_reads_overlay_without_rewriting_base",
        (
            (
                "    def _merge_overlay_chunks(self, chunks):  # noqa: ANN001, ANN201\n"
                "        overlay = self._index.load_query_tail_chunks()",
                "    def _merge_overlay_chunks(self, chunks):  # noqa: ANN001, ANN201\n"
                "        overlay = (\n"
                "            [] if self._data_dbs else self._index.load_query_tail_chunks()\n"
                "        )",
            ),
        ),
    ),
    "base-suppression": Mutation(
        "sharded_db.py",
        "test_source_shrink_suppresses_stale_base_rows_until_rebuilt",
        (("return self._index.query_tail_suppressed_sessions()", "return set()"),),
    ),
    "base-provenance": Mutation(
        "query_freshness.py",
        "test_same_session_in_a_different_path_never_seeds_from_the_old_base",
        (
            (
                "starting_extent = cursor if cursor_proven else {}",
                "starting_extent = cursor or base_extent or {}",
            ),
        ),
    ),
    "base-suppression-boundary": Mutation(
        "query_freshness.py",
        "test_partial_reparse_keeps_prior_base_visible_until_atomic_completion",
        (
            (
                "step_suppresses_base = suppresses_base or (\n"
                "                base_requires_suppression and end >= live_size\n"
                "            )",
                "step_suppresses_base = base_requires_suppression",
            ),
        ),
    ),
    "empty-replacement": Mutation(
        "query_freshness.py",
        "test_empty_replacement_suppresses_stale_base_but_empty_without_base_is_clean",
        (("and base_requires_suppression\n            and not suppresses_base", "and False\n            and not suppresses_base"),),
    ),
    "empty-suppression-read": Mutation(
        "sharded_db.py",
        "test_empty_replacement_suppresses_stale_base_but_empty_without_base_is_clean",
        (
            (
                "    def _merge_overlay_chunks(self, chunks):  # noqa: ANN001, ANN201\n"
                "        overlay = self._index.load_query_tail_chunks()\n"
                "        overlay_ids = {chunk.id for chunk in overlay}",
                "    def _merge_overlay_chunks(self, chunks):  # noqa: ANN001, ANN201\n"
                "        overlay = self._index.load_query_tail_chunks()\n"
                "        if not overlay:\n"
                "            return chunks\n"
                "        overlay_ids = {chunk.id for chunk in overlay}",
            ),
        ),
    ),
    "incomplete-replacement": Mutation(
        "query_freshness.py",
        "test_incomplete_first_record_keeps_prior_base_visible_and_reports_partial",
        (("            live_size == 0\n            and base_requires_suppression", "            observed == 0\n            and base_requires_suppression"),),
    ),
    "cursor-prefix": Mutation(
        "query_freshness.py",
        "test_same_path_same_length_replacement_invalidates_the_cursor",
        (
            (
                "                and int(cursor.get(\"source_mtime_ns\", -1))\n"
                "                == source_stat.st_mtime_ns",
                "                and True",
            ),
        ),
    ),
    "error-coverage": Mutation(
        "query_freshness.py",
        "test_error_after_a_durable_step_reports_that_coverage",
        (("known_observed = locals().get(\"observed\")", "known_observed = 0"),),
    ),
    "resume-listing-overlay": Mutation(
        "sharded_db.py",
        "test_overlay_only_session_hydrates_bounded_resume_listing",
        (
            (
                "        overlay = self._index.load_query_tail_chunks()\n"
                "        overlay_ids = {chunk.id for chunk in overlay}\n"
                "        for session_id, rows in grouped.items():",
                "        overlay = []\n"
                "        overlay_ids = {chunk.id for chunk in overlay}\n"
                "        for session_id, rows in grouped.items():",
            ),
        ),
    ),
    "overlay-retirement": Mutation(
        "sharded_db.py",
        "test_base_rebuild_retires_overlay_only_after_matching_coverage",
        (
            (
                "        for cursor in self._index.load_query_tail_cursors():\n",
                '        for cursor in self._index.load_query_tail_cursors():\n            self._index.clear_query_tail(cursor["source_key"])\n            continue\n',
            ),
        ),
    ),
    "cache-invalidation": Mutation(
        "server.py",
        "test_successful_refresh_invalidates_the_server_cache",
        (
            (
                "    if result.index_changed:\n        _invalidate_cache()",
                "    if result.index_changed:\n        pass",
            ),
        ),
    ),
    "preflight-order": Mutation(
        "server.py",
        "test_search_and_quick_label_success_not_only_empty_results",
        (
            (
                "    index_dir = project_index_dir()\n"
                "    freshness_line = _query_freshness_line(index_dir)\n"
                "    # min_score takes precedence",
                "    index_dir = project_index_dir()\n    # min_score takes precedence",
            ),
            (
                "    index = _get_index()\n\n    # Search the live transcript",
                "    index = _get_index()\n"
                "    freshness_line = _query_freshness_line(index_dir)\n\n"
                "    # Search the live transcript",
            ),
        ),
    ),
}


def _run(source_root: Path, witness: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/recall/test_query_freshness.py",
            "-k",
            witness,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mutation", choices=sorted(MUTATIONS))
    args = parser.parse_args()
    mutation = MUTATIONS[args.mutation]

    with tempfile.TemporaryDirectory(prefix="query-freshness-mutation-") as tmp:
        copied_src = Path(tmp) / "src"
        shutil.copytree(ROOT / "src" / "synapt", copied_src / "synapt")
        target = copied_src / "synapt" / "recall" / mutation.path
        text = target.read_text()
        for old, new in mutation.replacements:
            count = text.count(old)
            if count != 1:
                print(f"mutation={args.mutation} replacement_count={count}")
                return 2
            text = text.replace(old, new, 1)
        target.write_text(text)

        mutant = _run(copied_src, mutation.witness)
        killed = (
            mutant.returncode != 0
            and f"::{mutation.witness}" in mutant.stdout
            and "1 failed" in mutant.stdout
            and "ERROR collecting" not in mutant.stdout
        )
        pristine = _run(ROOT / "src", mutation.witness)
        survived = pristine.returncode == 0 and "1 passed" in pristine.stdout

    print(f"mutation={args.mutation}")
    print(f"witness={mutation.witness}")
    print(f"mutant={'KILLED' if killed else 'SURVIVED'}")
    print(f"pristine={'PASSED' if survived else 'FAILED'}")
    return 0 if killed and survived else 1


if __name__ == "__main__":
    raise SystemExit(main())
