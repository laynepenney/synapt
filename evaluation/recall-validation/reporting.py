"""Three-way diff report generation for recall-validation.

Produces human-readable and machine-parseable reports comparing
baseline (locked previous version) / actual (current run) / expected
(ground truth) results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    CATEGORY_LABELS,
    Category,
    Expected,
    ExpectedMatch,
    FixtureResult,
    SuiteResult,
)


def _format_float(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}"


def _match_symbol(record_id: str, expected_ids: set[str], retrieved_ids: set[str]) -> str:
    in_expected = record_id in expected_ids
    in_retrieved = record_id in retrieved_ids
    if in_expected and in_retrieved:
        return "+"
    if in_expected and not in_retrieved:
        return "MISS"
    if not in_expected and in_retrieved:
        return "FP"
    return ""


def render_fixture_diff(
    result: FixtureResult,
    expected: Expected,
    baseline: FixtureResult | None = None,
) -> str:
    lines: list[str] = []
    expected_ids = {m.record_id for m in expected.matches}
    expected_by_id = {m.record_id: m for m in expected.matches}
    retrieved_ids = {r.record_id for r in result.retrieved}

    lines.append(f"=== Fixture: {result.fixture_id} ===")
    lines.append(f"Category: {CATEGORY_LABELS.get(result.category.value, result.category.value)}")
    lines.append("")

    lines.append("--- EXPECTED ---")
    if expected.expect_empty:
        lines.append("  (no matches expected)")
    else:
        for m in sorted(expected.matches, key=lambda x: x.rank):
            lines.append(f"  {m.rank}. {m.record_id} (relevance: {m.relevance})")
    if expected.response_routing:
        lines.append(f"  Routing: {expected.response_routing.classification.value}"
                      f" (safety_critical: {expected.response_routing.safety_critical})")
    lines.append("")

    lines.append("--- ACTUAL ---")
    if not result.retrieved:
        lines.append("  (no results returned)")
    else:
        for i, r in enumerate(result.retrieved, 1):
            symbol = _match_symbol(r.record_id, expected_ids, retrieved_ids)
            lines.append(f"  {i}. {r.record_id} (score: {_format_float(r.score)}) [{symbol}]")
    if result.routing:
        lines.append(f"  Routing: {result.routing.classification.value}"
                      f" (confidence: {_format_float(result.routing.confidence)})")
    lines.append("")

    lines.append("--- BASELINE ---")
    if baseline is None:
        lines.append("  (no baseline yet)")
    else:
        if not baseline.retrieved:
            lines.append("  (no results returned)")
        else:
            for i, r in enumerate(baseline.retrieved, 1):
                lines.append(f"  {i}. {r.record_id} (score: {_format_float(r.score)})")
        lines.append(f"  P@5: {_format_float(baseline.precision_at_5)}"
                      f"  R@10: {_format_float(baseline.recall_at_10)}")
    lines.append("")

    lines.append("--- SCORES ---")
    lines.append(f"P@5: {_format_float(result.precision_at_5)}")
    lines.append(f"R@10: {_format_float(result.recall_at_10)}")
    if result.rank_correlation is not None:
        lines.append(f"Rank correlation (tau): {_format_float(result.rank_correlation)}")
    if result.safety_correct is not None:
        lines.append(f"Safety classification: {'CORRECT' if result.safety_correct else 'WRONG'}")
    if result.negative_correct is not None:
        lines.append(f"Negative case: {'CORRECT' if result.negative_correct else 'FALSE POSITIVE'}")
    if result.passed is not None:
        lines.append(f"Threshold: {'PASS' if result.passed else 'FAIL'}")
    lines.append("")

    lines.append("--- DIFF ---")
    all_ids = expected_ids | retrieved_ids
    for pid in sorted(all_ids):
        in_exp = pid in expected_ids
        in_ret = pid in retrieved_ids
        if in_exp and in_ret:
            exp_rank = expected_by_id[pid].rank
            act_rank = next(
                i + 1 for i, r in enumerate(result.retrieved) if r.record_id == pid
            )
            if exp_rank == act_rank:
                lines.append(f"  + {pid}: expected rank {exp_rank}, actual rank {act_rank}")
            else:
                lines.append(f"  ~ {pid}: expected rank {exp_rank}, actual rank {act_rank} (drift)")
        elif in_exp and not in_ret:
            lines.append(f"  - {pid}: MISSING (expected rank {expected_by_id[pid].rank})")
        elif not in_exp and in_ret:
            act_rank = next(
                i + 1 for i, r in enumerate(result.retrieved) if r.record_id == pid
            )
            lines.append(f"  ! {pid}: FALSE POSITIVE at rank {act_rank}")
    lines.append("")

    return "\n".join(lines)


def render_summary(suite_result: SuiteResult) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"RECALL VALIDATION REPORT: {suite_result.suite_name}")
    lines.append(f"Timestamp: {suite_result.timestamp}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("OVERALL SCORES")
    lines.append("-" * 40)
    for key, val in sorted(suite_result.overall_scores.items()):
        lines.append(f"  {key}: {_format_float(val)}")
    lines.append("")

    lines.append("PER-CATEGORY SCORES")
    lines.append("-" * 40)
    for cat in Category:
        cat_key = cat.value
        if cat_key not in suite_result.category_scores:
            continue
        scores = suite_result.category_scores[cat_key]
        label = CATEGORY_LABELS.get(cat_key, cat_key)
        lines.append(f"  {label} (n={int(scores['count'])})")
        for key, val in sorted(scores.items()):
            if key == "count":
                continue
            lines.append(f"    {key}: {_format_float(val)}")
        lines.append("")

    return "\n".join(lines)


def write_report(
    suite_result: SuiteResult,
    fixture_diffs: list[str],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"report-{suite_result.suite_name}-{ts}.txt"

    summary = render_summary(suite_result)
    full_report = summary + "\n" + "=" * 60 + "\nFIXTURE DETAILS\n" + "=" * 60 + "\n\n"
    full_report += "\n".join(fixture_diffs)

    report_path.write_text(full_report)

    json_path = output_dir / f"report-{suite_result.suite_name}-{ts}.json"
    json_data = {
        "suite_name": suite_result.suite_name,
        "timestamp": suite_result.timestamp,
        "overall_scores": suite_result.overall_scores,
        "category_scores": suite_result.category_scores,
        "fixture_results": [
            {
                "fixture_id": r.fixture_id,
                "category": r.category.value,
                "precision_at_5": r.precision_at_5,
                "recall_at_10": r.recall_at_10,
                "rank_correlation": r.rank_correlation,
                "safety_correct": r.safety_correct,
                "negative_correct": r.negative_correct,
                "passed": r.passed,
                "retrieved": [
                    {"record_id": ret.record_id, "score": ret.score}
                    for ret in r.retrieved
                ],
            }
            for r in suite_result.fixture_results
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2) + "\n")

    return report_path
