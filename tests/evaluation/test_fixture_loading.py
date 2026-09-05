"""Tests for per-fixture user_history and alternate-format compatibility."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

import importlib

rv = importlib.import_module("recall-validation.models")
rv_harness = importlib.import_module("recall-validation.harness")

Category = rv.Category
Expected = rv.Expected
Fixture = rv.Fixture
Record = rv.Record
fixture_from_dict = rv.fixture_from_dict
_record_from_dict = rv._record_from_dict
_parse_expected_matches = rv._parse_expected_matches
CATEGORY_ALIASES = rv.CATEGORY_ALIASES

load_suite = rv_harness.load_suite
_load_fixture_file = rv_harness._load_fixture_file


class TestRecordFromDict:
    def test_harness_native_format(self):
        d = {
            "id": "record-1",
            "date": "2023-01-15",
            "text": "A record about hope.",
            "themes": ["hope"],
            "entities": ["God"],
            "summary": "Hope record",
        }
        p = _record_from_dict(d)
        assert p.id == "record-1"
        assert p.date == "2023-01-15"
        assert p.text == "A record about hope."
        assert p.themes == ["hope"]
        assert p.entities == ["God"]
        assert p.summary == "Hope record"

    def test_alt_format_uses_synthetic_date(self):
        d = {
            "id": "Calvin:conv-50:session_1:event_summary:Calvin:1:1",
            "text": "A record text.",
            "synthetic_date": "2023-03-23",
            "categories": ["transition", "work"],
            "emotional_register": "thanksgiving",
            "themes": ["thanksgiving", "transition"],
            "source_situation_id": "conv-50:session_1:event_summary:Calvin:1",
        }
        p = _record_from_dict(d)
        assert p.id == "Calvin:conv-50:session_1:event_summary:Calvin:1:1"
        assert p.date == "2023-03-23"
        assert p.themes == ["thanksgiving", "transition"]
        assert p.entities == []
        assert p.summary == ""

    def test_missing_optional_fields_default(self):
        d = {"id": "p-1", "text": "Minimal record."}
        p = _record_from_dict(d)
        assert p.date == ""
        assert p.themes == []
        assert p.entities == []
        assert p.summary == ""

    def test_date_takes_precedence_over_synthetic_date(self):
        d = {
            "id": "p-1",
            "text": "Text.",
            "date": "2023-01-01",
            "synthetic_date": "2023-06-01",
        }
        p = _record_from_dict(d)
        assert p.date == "2023-01-01"


class TestParseExpectedMatches:
    def test_ranked_list_becomes_expected_matches(self):
        d = {
            "ranked": ["record-a", "record-b", "record-c"],
            "min_precision_at_5": 0.6,
            "min_recall_at_10": 1.0,
        }
        expected = _parse_expected_matches(d)
        assert len(expected.matches) == 3
        assert expected.matches[0].record_id == "record-a"
        assert expected.matches[0].rank == 1
        assert expected.matches[1].rank == 2
        assert expected.matches[2].rank == 3
        assert all(m.relevance == "high" for m in expected.matches)
        assert expected.min_precision_at_5 == 0.6
        assert expected.min_recall_at_10 == 1.0
        assert expected.expect_empty is False

    def test_empty_ranked_with_max_results_zero_is_negative(self):
        d = {
            "ranked": [],
            "min_precision_at_5": 1,
            "min_recall_at_10": 1,
            "max_results": 0,
        }
        expected = _parse_expected_matches(d)
        assert expected.expect_empty is True
        assert len(expected.matches) == 0

    def test_empty_ranked_without_max_results_not_negative(self):
        d = {"ranked": []}
        expected = _parse_expected_matches(d)
        assert expected.expect_empty is False


class TestCategoryAliases:
    def test_all_short_names_resolve(self):
        assert CATEGORY_ALIASES["direct"] == Category.DIRECT_LOOKUP
        assert CATEGORY_ALIASES["thematic"] == Category.THEMATIC_RECALL
        assert CATEGORY_ALIASES["negative"] == Category.NEGATIVE_CASE
        assert CATEGORY_ALIASES["temporal"] == Category.TEMPORAL_QUERIES
        assert CATEGORY_ALIASES["pattern"] == Category.PATTERN_RECOGNITION
        assert CATEGORY_ALIASES["action_followup"] == Category.ACTION_FOLLOWUP


class TestFixtureFromDict:
    def test_alt_format(self):
        d = {
            "id": "direct:Calvin:01",
            "category": "direct",
            "persona_id": "Calvin",
            "query": "What about my mansion?",
            "user_history": [
                {
                    "id": "Calvin:p1",
                    "text": "A record about my mansion.",
                    "synthetic_date": "2023-03-23",
                    "themes": ["transition"],
                },
                {
                    "id": "Calvin:p2",
                    "text": "Another record.",
                    "synthetic_date": "2023-04-01",
                    "themes": ["gratitude"],
                },
            ],
            "expected_matches": {
                "ranked": ["Calvin:p1"],
                "min_precision_at_5": 0.8,
                "min_recall_at_10": 1.0,
            },
            "expected_routing": None,
            "notes": "Direct lookup for mansion purchase.",
        }
        f = fixture_from_dict(d)
        assert f.id == "direct:Calvin:01"
        assert f.category == Category.DIRECT_LOOKUP
        assert len(f.history) == 2
        assert f.history[0].date == "2023-03-23"
        assert f.expected.matches[0].record_id == "Calvin:p1"
        assert f.expected.min_precision_at_5 == 0.8
        assert f.description == "Direct lookup for mansion purchase."
        assert f.query_date == ""

    def test_harness_native_format(self):
        d = {
            "id": "test-1",
            "category": "direct_lookup",
            "description": "Test fixture",
            "history": [
                {
                    "id": "p-1",
                    "date": "2023-01-01",
                    "text": "A record.",
                    "themes": [],
                    "entities": [],
                    "summary": "",
                },
            ],
            "query": "Test query",
            "query_date": "2023-06-01",
            "expected": {
                "matches": [
                    {"record_id": "p-1", "rank": 1, "relevance": "high"},
                ],
            },
        }
        f = fixture_from_dict(d)
        assert f.id == "test-1"
        assert f.category == Category.DIRECT_LOOKUP
        assert f.history[0].date == "2023-01-01"
        assert f.expected.matches[0].record_id == "p-1"
        assert f.description == "Test fixture"
        assert f.query_date == "2023-06-01"

    def test_negative_case_alt_format(self):
        d = {
            "id": "negative:Calvin:01",
            "category": "negative",
            "persona_id": "Calvin",
            "query": "What about the Mars rover?",
            "user_history": [
                {"id": "Calvin:p1", "text": "A record.", "synthetic_date": "2023-01-01"},
            ],
            "expected_matches": {
                "ranked": [],
                "min_precision_at_5": 1,
                "min_recall_at_10": 1,
                "max_results": 0,
            },
            "expected_routing": None,
            "notes": "Unrelated query.",
        }
        f = fixture_from_dict(d)
        assert f.expected.expect_empty is True
        assert len(f.expected.matches) == 0


class TestLoadFixtureFile:
    def test_json_array(self, tmp_path):
        data = [{"id": "t-1", "query": "q1"}, {"id": "t-2", "query": "q2"}]
        p = tmp_path / "fixtures.json"
        p.write_text(json.dumps(data))
        result = _load_fixture_file(p)
        assert len(result) == 2
        assert result[0]["id"] == "t-1"

    def test_jsonl(self, tmp_path):
        lines = [
            json.dumps({"id": "t-1", "query": "q1"}),
            json.dumps({"id": "t-2", "query": "q2"}),
        ]
        p = tmp_path / "fixtures.jsonl"
        p.write_text("\n".join(lines))
        result = _load_fixture_file(p)
        assert len(result) == 2
        assert result[1]["id"] == "t-2"

    def test_jsonl_with_blank_lines(self, tmp_path):
        lines = [
            json.dumps({"id": "t-1"}),
            "",
            json.dumps({"id": "t-2"}),
            "",
        ]
        p = tmp_path / "data.jsonl"
        p.write_text("\n".join(lines))
        result = _load_fixture_file(p)
        assert len(result) == 2


class TestLoadSuitePerFixtureHistory:
    def _create_suite(self, tmp_path, *, shared_history=True, fixture_has_history=False):
        suite_dir = tmp_path / "test-suite"
        suite_dir.mkdir()

        suite_meta = {
            "name": "test-suite",
            "version": "0.1.0",
            "description": "Test suite",
            "methodology_version": "0.2",
            "categories": ["direct_lookup"],
            "fixture_files": {"direct_lookup": "direct.json"},
        }

        shared_records = [
            {"id": "shared-p1", "date": "2023-01-01", "text": "Shared record one.",
             "themes": [], "entities": [], "summary": ""},
            {"id": "shared-p2", "date": "2023-02-01", "text": "Shared record two.",
             "themes": [], "entities": [], "summary": ""},
        ]

        if shared_history:
            suite_meta["shared_history"] = "history.json"
            (suite_dir / "history.json").write_text(json.dumps(shared_records))

        fixture = {
            "id": "test-1",
            "category": "direct_lookup",
            "description": "Test",
            "query": "What about record one?",
            "query_date": "2023-06-01",
            "expected": {
                "matches": [{"record_id": "shared-p1", "rank": 1, "relevance": "high"}],
            },
        }
        if fixture_has_history:
            fixture["user_history"] = [
                {"id": "own-p1", "text": "Own record.", "synthetic_date": "2023-05-01"},
            ]

        (suite_dir / "direct.json").write_text(json.dumps([fixture]))
        (suite_dir / "suite.json").write_text(json.dumps(suite_meta))
        return suite_dir

    def test_shared_history_injected_when_fixture_lacks_own(self, tmp_path, monkeypatch):
        self._create_suite(tmp_path, shared_history=True, fixture_has_history=False)
        monkeypatch.setattr(rv_harness, "FIXTURES_DIR", tmp_path)
        _, history, fixtures = load_suite("test-suite")
        assert len(history) == 2
        assert len(fixtures) == 1
        assert len(fixtures[0].history) == 2
        assert fixtures[0].history[0].id == "shared-p1"

    def test_per_fixture_history_used_when_present(self, tmp_path, monkeypatch):
        self._create_suite(tmp_path, shared_history=True, fixture_has_history=True)
        monkeypatch.setattr(rv_harness, "FIXTURES_DIR", tmp_path)
        _, history, fixtures = load_suite("test-suite")
        assert len(history) == 2
        assert len(fixtures[0].history) == 1
        assert fixtures[0].history[0].id == "own-p1"

    def test_no_shared_history_with_per_fixture(self, tmp_path, monkeypatch):
        self._create_suite(tmp_path, shared_history=False, fixture_has_history=True)
        monkeypatch.setattr(rv_harness, "FIXTURES_DIR", tmp_path)
        _, history, fixtures = load_suite("test-suite")
        assert len(history) == 0
        assert len(fixtures[0].history) == 1
        assert fixtures[0].history[0].id == "own-p1"

    def test_jsonl_fixture_loading(self, tmp_path, monkeypatch):
        suite_dir = tmp_path / "jsonl-suite"
        suite_dir.mkdir()
        suite_meta = {
            "name": "jsonl-suite",
            "version": "0.1.0",
            "description": "JSONL test",
            "methodology_version": "0.2",
            "categories": ["direct_lookup"],
            "fixture_files": {"direct_lookup": "direct.jsonl"},
        }
        (suite_dir / "suite.json").write_text(json.dumps(suite_meta))

        fixtures_jsonl = "\n".join([
            json.dumps({
                "id": "direct:Test:01",
                "category": "direct",
                "query": "Test query",
                "user_history": [
                    {"id": "p-1", "text": "Record text.", "synthetic_date": "2023-01-01"},
                ],
                "expected_matches": {
                    "ranked": ["p-1"],
                    "min_precision_at_5": 0.8,
                    "min_recall_at_10": 1.0,
                },
                "expected_routing": None,
                "notes": "Test fixture.",
            }),
        ])
        (suite_dir / "direct.jsonl").write_text(fixtures_jsonl)

        monkeypatch.setattr(rv_harness, "FIXTURES_DIR", tmp_path)
        _, _, fixtures = load_suite("jsonl-suite")
        assert len(fixtures) == 1
        assert fixtures[0].id == "direct:Test:01"
        assert fixtures[0].category == Category.DIRECT_LOOKUP
        assert fixtures[0].history[0].date == "2023-01-01"
        assert fixtures[0].expected.matches[0].record_id == "p-1"
