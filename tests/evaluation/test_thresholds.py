"""Tests for threshold-aware reporting in recall-validation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

import importlib

rv = importlib.import_module("recall-validation.models")
rv_scoring = importlib.import_module("recall-validation.scoring")
rv_harness = importlib.import_module("recall-validation.harness")

Category = rv.Category
Expected = rv.Expected
ExpectedMatch = rv.ExpectedMatch
FixtureResult = rv.FixtureResult
Mode = rv.Mode
RetrievalResult = rv.RetrievalResult
SuiteResult = rv.SuiteResult

evaluate_thresholds = rv_scoring.evaluate_thresholds
score_fixture = rv_scoring.score_fixture
aggregate_category_scores = rv_scoring.aggregate_category_scores

check_ship_gate = rv_harness.check_ship_gate


def _make_expected(
    match_ids: list[str],
    min_p5: float | None = None,
    min_r10: float | None = None,
    expect_empty: bool = False,
) -> Expected:
    matches = [
        ExpectedMatch(record_id=pid, rank=i + 1, relevance="high")
        for i, pid in enumerate(match_ids)
    ]
    return Expected(
        matches=matches,
        expect_empty=expect_empty,
        min_precision_at_5=min_p5,
        min_recall_at_10=min_r10,
    )


def _make_retrieved(ids: list[str], base_score: float = 0.9) -> list[RetrievalResult]:
    return [
        RetrievalResult(record_id=pid, score=base_score - i * 0.1)
        for i, pid in enumerate(ids)
    ]


class TestEvaluateThresholds:
    def test_no_thresholds_returns_none(self):
        expected = _make_expected(["a", "b"])
        result = FixtureResult(
            fixture_id="test-1", category=Category.DIRECT_LOOKUP,
            precision_at_5=0.4, recall_at_10=0.5,
        )
        assert evaluate_thresholds(result, expected) is None

    def test_passes_when_above_thresholds(self):
        expected = _make_expected(["a", "b"], min_p5=0.3, min_r10=0.5)
        result = FixtureResult(
            fixture_id="test-1", category=Category.DIRECT_LOOKUP,
            precision_at_5=0.4, recall_at_10=1.0,
        )
        assert evaluate_thresholds(result, expected) is True

    def test_fails_on_precision_below_threshold(self):
        expected = _make_expected(["a", "b"], min_p5=0.8, min_r10=0.5)
        result = FixtureResult(
            fixture_id="test-1", category=Category.DIRECT_LOOKUP,
            precision_at_5=0.4, recall_at_10=1.0,
        )
        assert evaluate_thresholds(result, expected) is False

    def test_fails_on_recall_below_threshold(self):
        expected = _make_expected(["a", "b"], min_p5=0.3, min_r10=0.9)
        result = FixtureResult(
            fixture_id="test-1", category=Category.DIRECT_LOOKUP,
            precision_at_5=0.4, recall_at_10=0.5,
        )
        assert evaluate_thresholds(result, expected) is False

    def test_negative_case_passes_when_correct(self):
        expected = _make_expected([], expect_empty=True)
        result = FixtureResult(
            fixture_id="test-1", category=Category.NEGATIVE_CASE,
            negative_correct=True,
        )
        assert evaluate_thresholds(result, expected) is True

    def test_negative_case_fails_when_false_positive(self):
        expected = _make_expected([], expect_empty=True)
        result = FixtureResult(
            fixture_id="test-1", category=Category.NEGATIVE_CASE,
            negative_correct=False,
        )
        assert evaluate_thresholds(result, expected) is False


class TestScoreFixturePassField:
    def test_score_fixture_sets_passed_with_thresholds(self):
        expected = _make_expected(["a", "b"], min_p5=0.3, min_r10=0.5)
        retrieved = _make_retrieved(["a", "b", "c", "d", "e"])
        result = score_fixture(
            retrieved=retrieved, routing=None,
            expected=expected, fixture_id="t-1",
            category=Category.DIRECT_LOOKUP,
        )
        assert result.passed is True
        assert result.precision_at_5 == 0.4

    def test_score_fixture_no_thresholds_passed_is_none(self):
        expected = _make_expected(["a", "b"])
        retrieved = _make_retrieved(["a", "b", "c"])
        result = score_fixture(
            retrieved=retrieved, routing=None,
            expected=expected, fixture_id="t-1",
            category=Category.DIRECT_LOOKUP,
        )
        assert result.passed is None

    def test_score_fixture_negative_case_sets_passed(self):
        expected = Expected(expect_empty=True)
        result = score_fixture(
            retrieved=[], routing=None,
            expected=expected, fixture_id="t-1",
            category=Category.NEGATIVE_CASE,
        )
        assert result.passed is True

    def test_score_fixture_negative_case_low_scores_p5_correct(self):
        expected = Expected(expect_empty=True)
        low_score_results = _make_retrieved(["x", "y", "z"], base_score=0.3)
        result = score_fixture(
            retrieved=low_score_results, routing=None,
            expected=expected, fixture_id="t-1",
            category=Category.NEGATIVE_CASE,
        )
        assert result.precision_at_5 == 1.0
        assert result.negative_correct is True

    def test_score_fixture_negative_case_high_scores_p5_zero(self):
        expected = Expected(expect_empty=True)
        high_score_results = _make_retrieved(["x", "y", "z"], base_score=0.9)
        result = score_fixture(
            retrieved=high_score_results, routing=None,
            expected=expected, fixture_id="t-1",
            category=Category.NEGATIVE_CASE,
        )
        assert result.precision_at_5 == 0.0
        assert result.negative_correct is False


class TestAggregatePassFail:
    def test_category_scores_include_pass_fail_counts(self):
        results = [
            FixtureResult(
                fixture_id="t-1", category=Category.DIRECT_LOOKUP,
                precision_at_5=0.8, recall_at_10=1.0, passed=True,
            ),
            FixtureResult(
                fixture_id="t-2", category=Category.DIRECT_LOOKUP,
                precision_at_5=0.2, recall_at_10=0.5, passed=False,
            ),
            FixtureResult(
                fixture_id="t-3", category=Category.DIRECT_LOOKUP,
                precision_at_5=0.6, recall_at_10=0.8, passed=True,
            ),
        ]
        scores = aggregate_category_scores(results)
        assert scores["direct_lookup"]["passed_count"] == 2
        assert scores["direct_lookup"]["failed_count"] == 1

    def test_zero_counts_when_no_thresholds(self):
        results = [
            FixtureResult(
                fixture_id="t-1", category=Category.DIRECT_LOOKUP,
                precision_at_5=0.8, recall_at_10=1.0, passed=None,
            ),
        ]
        scores = aggregate_category_scores(results)
        assert scores["direct_lookup"]["passed_count"] == 0
        assert scores["direct_lookup"]["failed_count"] == 0


class TestShipGate:
    def _make_suite(self, category_scores: dict) -> SuiteResult:
        return SuiteResult(
            suite_name="test",
            timestamp="2026-01-01",
            category_scores=category_scores,
        )

    def test_ship_gate_passes_when_no_failures(self):
        suite = self._make_suite({
            "direct_lookup": {"count": 10, "mean_p_at_5": 0.8, "mean_r_at_10": 0.9, "passed_count": 10, "failed_count": 0},
            "negative_case": {"count": 8, "mean_p_at_5": 1.0, "mean_r_at_10": 1.0, "passed_count": 8, "failed_count": 0},
        })
        failures = check_ship_gate(suite, {"direct_lookup", "negative_case"})
        assert failures == []

    def test_ship_gate_fails_when_gated_category_has_failures(self):
        suite = self._make_suite({
            "direct_lookup": {"count": 10, "mean_p_at_5": 0.3, "mean_r_at_10": 0.5, "passed_count": 3, "failed_count": 7},
            "negative_case": {"count": 8, "mean_p_at_5": 1.0, "mean_r_at_10": 1.0, "passed_count": 8, "failed_count": 0},
        })
        failures = check_ship_gate(suite, {"direct_lookup", "negative_case"})
        assert len(failures) == 1
        assert "Direct Lookup" in failures[0]

    def test_non_gated_categories_dont_trigger_failure(self):
        suite = self._make_suite({
            "direct_lookup": {"count": 10, "mean_p_at_5": 0.3, "mean_r_at_10": 0.5, "passed_count": 3, "failed_count": 7},
            "temporal_queries": {"count": 8, "mean_p_at_5": 0.0, "mean_r_at_10": 0.0, "passed_count": 0, "failed_count": 8},
        })
        failures = check_ship_gate(suite, {"direct_lookup"})
        assert len(failures) == 1
        assert "Temporal" not in failures[0]

    def test_research_mode_enum_exists(self):
        assert Mode.RESEARCH.value == "research"
        assert Mode.SHIP_GATE.value == "ship-gate"
