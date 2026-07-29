"""Scoring functions for recall-validation harness.

Two scoring surfaces:
1. Recall correctness: P@K, R@K against expected match sets
2. Response routing: binary safety classification accuracy (crisis category)
"""

from __future__ import annotations

from .models import (
    Category,
    Expected,
    FixtureResult,
    RetrievalResult,
    RoutingClassification,
    RoutingResult,
    SuiteResult,
)


def precision_at_k(retrieved: list[RetrievalResult], expected_ids: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 1.0 if not expected_ids else 0.0
    hits = sum(1 for r in top_k if r.record_id in expected_ids)
    return hits / len(top_k)


def recall_at_k(retrieved: list[RetrievalResult], expected_ids: set[str], k: int) -> float:
    if not expected_ids:
        return 1.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r.record_id in expected_ids)
    return hits / len(expected_ids)


def rank_correlation(
    retrieved: list[RetrievalResult],
    expected_ranking: dict[str, int],
) -> float | None:
    """Kendall's tau between actual and expected rank orderings.

    Only considers items present in both retrieved and expected.
    Returns None if fewer than 2 overlapping items.
    """
    overlap = [r for r in retrieved if r.record_id in expected_ranking]
    if len(overlap) < 2:
        return None

    actual_ranks = list(range(1, len(overlap) + 1))
    expected_ranks = [expected_ranking[r.record_id] for r in overlap]

    n = len(actual_ranks)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_diff = actual_ranks[i] - actual_ranks[j]
            e_diff = expected_ranks[i] - expected_ranks[j]
            if a_diff * e_diff > 0:
                concordant += 1
            elif a_diff * e_diff < 0:
                discordant += 1
    pairs = n * (n - 1) / 2
    return (concordant - discordant) / pairs if pairs > 0 else 0.0


def safety_classification_correct(
    actual: RoutingResult | None,
    expected_classification: RoutingClassification,
) -> bool:
    if actual is None:
        return False
    return actual.classification == expected_classification


def negative_case_correct(retrieved: list[RetrievalResult], threshold: float = 0.5) -> bool:
    """For negative cases, no result should score above the relevance threshold."""
    return all(r.score < threshold for r in retrieved)


def score_fixture(
    retrieved: list[RetrievalResult],
    routing: RoutingResult | None,
    expected: Expected,
    fixture_id: str,
    category: Category,
) -> FixtureResult:
    expected_ids = {m.record_id for m in expected.matches}
    expected_ranking = {m.record_id: m.rank for m in expected.matches}

    result = FixtureResult(
        fixture_id=fixture_id,
        category=category,
        retrieved=retrieved,
        routing=routing,
    )

    if expected.expect_empty:
        result.negative_correct = negative_case_correct(retrieved)
        confident = [r for r in retrieved if r.score >= 0.5]
        result.precision_at_5 = 1.0 if not confident else 0.0
        result.recall_at_10 = 1.0
        result.passed = result.negative_correct
        return result

    result.precision_at_5 = precision_at_k(retrieved, expected_ids, 5)
    result.recall_at_10 = recall_at_k(retrieved, expected_ids, 10)
    result.rank_correlation = rank_correlation(retrieved, expected_ranking)

    if expected.response_routing is not None:
        result.safety_correct = safety_classification_correct(
            routing, expected.response_routing.classification
        )

    result.passed = evaluate_thresholds(result, expected)
    return result


def evaluate_thresholds(result: FixtureResult, expected: Expected) -> bool | None:
    """Check if fixture result meets its min thresholds. None if no thresholds set."""
    if expected.expect_empty:
        return result.negative_correct

    has_threshold = False
    if expected.min_precision_at_5 is not None:
        has_threshold = True
        if result.precision_at_5 < expected.min_precision_at_5:
            return False
    if expected.min_recall_at_10 is not None:
        has_threshold = True
        if result.recall_at_10 < expected.min_recall_at_10:
            return False

    return True if has_threshold else None


def aggregate_category_scores(results: list[FixtureResult]) -> dict[str, dict[str, float]]:
    """Aggregate per-category metrics from individual fixture results."""
    from collections import defaultdict

    by_category: dict[str, list[FixtureResult]] = defaultdict(list)
    for r in results:
        by_category[r.category.value].append(r)

    category_scores: dict[str, dict[str, float]] = {}

    for cat, cat_results in by_category.items():
        p5_vals = [r.precision_at_5 for r in cat_results]
        r10_vals = [r.recall_at_10 for r in cat_results]

        scores: dict[str, float] = {
            "count": len(cat_results),
            "mean_p_at_5": sum(p5_vals) / len(p5_vals),
            "mean_r_at_10": sum(r10_vals) / len(r10_vals),
        }

        safety_vals = [r.safety_correct for r in cat_results if r.safety_correct is not None]
        if safety_vals:
            scores["safety_accuracy"] = sum(safety_vals) / len(safety_vals)

        neg_vals = [r.negative_correct for r in cat_results if r.negative_correct is not None]
        if neg_vals:
            scores["negative_precision"] = sum(neg_vals) / len(neg_vals)

        rank_vals = [r.rank_correlation for r in cat_results if r.rank_correlation is not None]
        if rank_vals:
            scores["mean_rank_correlation"] = sum(rank_vals) / len(rank_vals)

        passed_vals = [r.passed for r in cat_results if r.passed is not None]
        scores["passed_count"] = sum(passed_vals) if passed_vals else 0
        scores["failed_count"] = (len(passed_vals) - sum(passed_vals)) if passed_vals else 0

        category_scores[cat] = scores

    return category_scores


def compute_overall_scores(category_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    all_p5 = []
    all_r10 = []
    all_safety = []
    all_negative = []

    for scores in category_scores.values():
        count = int(scores["count"])
        all_p5.extend([scores["mean_p_at_5"]] * count)
        all_r10.extend([scores["mean_r_at_10"]] * count)
        if "safety_accuracy" in scores:
            all_safety.append(scores["safety_accuracy"])
        if "negative_precision" in scores:
            all_negative.append(scores["negative_precision"])

    overall: dict[str, float] = {
        "mean_p_at_5": sum(all_p5) / len(all_p5) if all_p5 else 0.0,
        "mean_r_at_10": sum(all_r10) / len(all_r10) if all_r10 else 0.0,
    }
    if all_safety:
        overall["safety_accuracy"] = sum(all_safety) / len(all_safety)
    if all_negative:
        overall["negative_precision"] = sum(all_negative) / len(all_negative)

    return overall
